"""One generation step with an NLA intervention at the last position.

`run(body, call)` handles one stepper API action. `call(role, endpoint, request)`
talks to the nla.cpp server for role 'base' (Qwen), 'av' (verbalizer) or
'ar' (reconstructor).
"""
import math
from nla import editor

LAYER = 20
CONTEXT = 2048  # base-model context; 148 tokens are reserved for generation
AV_INPUT_IDS = None  # AV prompt token IDs, set by the server from the pinned frontend
VARIANTS = ('original', 'roundtrip', 'edited_raw', 'edited_corrected')


class InvalidActivation(ValueError):
    pass


class InvalidPrediction(ValueError):
    pass


def prediction_token_id(result):
    """The backend's chosen token, including single-token UTF-8 fragments.

    The pinned server records token IDs before checking UTF-8 completeness but
    only reports probabilities for complete text, so a one-token response can
    legitimately omit them.
    """
    if not isinstance(result, dict):
        raise InvalidPrediction('Invalid next-token response')
    tokens, count = result.get('tokens'), result.get('tokens_predicted')
    native = (tokens[0] if isinstance(tokens, list) and len(tokens) == 1
              and type(tokens[0]) is int and count == 1 else None)
    probabilities = result.get('completion_probabilities')
    if probabilities not in (None, []):
        if not isinstance(probabilities, list) or not isinstance(probabilities[0], dict):
            raise InvalidPrediction('Invalid next-token probabilities')
        token = probabilities[0].get('id')
        if type(token) is not int or (native is not None and native != token):
            raise InvalidPrediction('Invalid or conflicting next-token ID')
        return token
    if native is not None:
        return native
    raise InvalidPrediction('Missing next-token ID')


def validate_activation(vector, size=None):
    if (not isinstance(vector, list) or not vector or (size is not None and len(vector) != size)
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)):
        raise InvalidActivation('Invalid or non-finite activation')


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def intervention(h, target, position):
    """Ask the backend to replace activation h with target at one position."""
    delta = [b-a for a, b in zip(h, target)]
    # The backend applies h + alpha*||h||*delta/||delta||; choose alpha so the
    # applied change is exactly delta.
    return {'layer': LAYER, 'token_pos': position, 'direction': delta,
            'alpha': norm(delta)/norm(h) if norm(h) else 0}


def metrics(h, r):
    e = [a-b for a, b in zip(h, r)]
    return {'l2': norm(e), 'relative_l2': norm(e)/max(norm(h), 1e-30),
            'mse': sum(x*x for x in e)/len(e),
            'cosine': sum(a*b for a, b in zip(h, r))/max(norm(h)*norm(r), 1e-30),
            'original_norm': norm(h), 'reconstructed_norm': norm(r)}


def run(body, call):
    action = body['action']
    if action == 'start':
        prompt = ('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
                  + body['prompt'] + '<|im_end|>\n<|im_start|>assistant\n')
        return {'tokens': call('base', '/tokenize', {'content': prompt, 'add_special': False,
                                                     'parse_special': True})['tokens']}
    if action == 'propose':
        return editor.propose(body.get('editor') or 'codex', str(body.get('instruction', '')),
                              str(body.get('text', '')))
    tokens = body['tokens']
    if (not isinstance(tokens, list) or not tokens or len(tokens) > CONTEXT-148
            or any(type(t) is not int for t in tokens)):
        raise ValueError('Token context is invalid or exceeds the configured limit')
    if action == 'decode':
        return call('base', '/detokenize', {'tokens': tokens})

    def extract():
        return call('base', '/extract', {'prompt': tokens, 'layer': LAYER, 'token_pos': -1,
                                         'add_special': False})['activation']

    def predict(h=None, target=None):
        request = {'prompt': tokens, 'n_predict': 1, 'temperature': 0, 'seed': 42,
                   'cache_prompt': False, 'n_probs': 30, 'post_sampling_probs': False,
                   'return_tokens': True, 'special': True}
        if target is not None:
            request['steering'] = [intervention(h, target, len(tokens)-1)]
        return call('base', '/completion', request)

    def reconstruct(text):
        return call('ar', '/reconstruct', {'explanation': text})['activation']

    if action == 'inspect':
        h = extract()
        return {'h': h, **({'baseline': predict()} if body.get('baseline', True) else {})}
    if action == 'explain':
        h = body['h']
        validate_activation(h)
        text = call('av', '/explain', {'activation': h, 'input_ids': AV_INPUT_IDS, 'inject_pos': 111,
                                       'scale': 150, 'n_predict': 200, 'temperature': 0, 'seed': 42})
        text = text.get('explanation') or text.get('text')
        r = reconstruct(text)
        validate_activation(r, len(h))
        return {'text': text, 'reconstruction': r, 'error': [a-b for a, b in zip(h, r)],
                'metrics': metrics(h, r)}
    if action == 'compare':
        variants = body.get('variants', VARIANTS)
        if not variants or any(v not in VARIANTS for v in variants):
            raise ValueError('Unknown comparison variant')
        multiplier = body.get('multiplier', 1.0)
        if type(multiplier) not in (int, float) or not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError('Steering multiplier must be a finite non-negative number')
        # Re-extract h for this exact context rather than trusting the client.
        h, r0 = extract(), body['reconstruction']
        r1 = reconstruct(body['edited']) if any(v.startswith('edited') for v in variants) else r0
        for vector in (h, r0, r1):
            validate_activation(vector, len(h))
        error = [a-b for a, b in zip(h, r0)]
        delta = [a-b for a, b in zip(r1, r0)]
        # h + m*(AR(t') - AR(t)); m = 1 is written as AR(t') + error so that
        # existing checkpoints reproduce bit-for-bit.
        corrected = ([a+b for a, b in zip(r1, error)] if multiplier == 1
                     else [a + multiplier*d for a, d in zip(h, delta)])
        targets = {'original': None, 'roundtrip': r0, 'edited_raw': r1, 'edited_corrected': corrected}
        natural = norm(delta)/norm(h) if norm(h) else 0.0
        return {**{v: predict(h, targets[v]) for v in variants}, 'edited_reconstruction': r1,
                'error': error, 'metrics': metrics(h, r0),
                'steering': {'multiplier': multiplier, 'alpha_natural': natural,
                             'alpha_applied': multiplier*natural}}
    raise ValueError('Unknown action')
