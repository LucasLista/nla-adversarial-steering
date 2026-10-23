"""Four independent greedy trajectories per prompt, checkpointed after every stage.

Each branch generates its own continuation. At every step the activation at
the last position is explained, the explanation edited, and the branch's next
token predicted from:

  original          h
  roundtrip         AR(t)
  edited_raw        AR(t')
  edited_corrected  AR(t') + h - AR(t)

Every API call is saved with its request under DIR/BRANCH/step-NNNN/, so an
interrupted run resumes exactly where it stopped.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from nla import stepper

VARIANTS = stepper.VARIANTS
EOS = {151643, 151645}
REPEAT_LIMIT = 50  # identical tokens in a row that end a branch as degenerate
UNPARSEABLE = 'Failed to parse input at pos'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def recorded(path, request, operation, validate=None):
    """Return the checkpointed response for this exact request, or compute and save it."""
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['request'] != request:
            raise ValueError(f'Checkpoint request changed: {path}')
        if validate is None or validate(saved['response']):
            return saved['response']
    start = time.time()
    try:
        result = operation()
        if validate is not None and not validate(result):
            raise ValueError('Backend returned an invalid response')
    except Exception as exc:
        write_json(path.with_name(f'{path.stem}-error-{time.time_ns()}.json'),
                   {'request': request, 'error': str(exc), 'started_at': start, 'finished_at': time.time()})
        raise
    write_json(path, {'request': request, 'response': result, 'started_at': start,
                      'finished_at': time.time(), 'duration_seconds': time.time() - start})
    return result


def valid_prediction(result, allow_empty=False):
    if allow_empty and result == {}:
        return True
    try:
        stepper.prediction_token_id(result)
        return True
    except stepper.InvalidPrediction:
        return False


class API:
    """Client for a running nla.server."""

    def __init__(self, url, token=None):
        self.url = url.rstrip('/') + '/stepper-api'
        self.token = token

    def __call__(self, action, **kwargs):
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = 'Bearer ' + self.token
        request = urllib.request.Request(self.url, json.dumps({'action': action, **kwargs}).encode(), headers)
        try:
            with urllib.request.urlopen(request, timeout=1200) as r:
                result = json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code in (408, 502, 503, 504):
                raise ConnectionError(f'Server temporarily unavailable (HTTP {exc.code})') from None
            try:
                result = json.loads(exc.read())
            except ValueError:
                result = {'error': f'Server HTTP {exc.code}'}
        if isinstance(result.get('error'), str):
            raise RuntimeError(result['error'])
        return result


def progress(directory, limit):
    """Write branch token counts and completion flags, without any model text."""
    branches = {}
    for variant in VARIANTS:
        path = directory / variant / 'state.json'
        state = json.loads(path.read_text()) if path.exists() else {'generated': [], 'ended': False}
        count = len(state['generated'])
        branches[variant] = {'tokens': count, 'ended': state['ended'],
                             'complete': state['ended'] or count >= limit}
    write_json(directory / 'progress.json',
               {'branches': branches, 'complete': (directory / 'complete.json').exists()})


def degenerate(generated):
    return len(generated) >= REPEAT_LIMIT and len(set(generated[-REPEAT_LIMIT:])) == 1


def generate(config, directory, api, editor, variants=None):
    """Run (or resume) the selected branches for one prompt.

    config: prompt, instruction, max_tokens, plus optional comparison_mode='selected'
    (predict only the branch being generated) and steering_multiplier.
    editor(instruction, text) returns {'proposal': edited_text, ...}.
    """
    directory = Path(directory)
    metadata = directory / 'config.json'
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        if previous != config:
            # The only allowed change is extending the token limit.
            same = {k: v for k, v in previous.items() if k != 'max_tokens'} == \
                   {k: v for k, v in config.items() if k != 'max_tokens'}
            if not same or config['max_tokens'] <= previous['max_tokens']:
                raise ValueError('Run configuration differs; only increasing max_tokens is supported')
            (directory / 'complete.json').unlink(missing_ok=True)
    write_json(metadata, config)
    started = recorded(directory / 'start.json', {'prompt': config['prompt']},
                       lambda: api('start', prompt=config['prompt']))
    eos = set(started.get('eos_token_ids', EOS))
    selected = tuple(variants or VARIANTS)
    if not selected or any(v not in VARIANTS for v in selected):
        raise ValueError('Unknown trajectory variant')
    fast = config.get('comparison_mode') == 'selected'
    # Branches share prefixes until they diverge; reuse identical deterministic
    # inspect/explain responses across them (never editor proposals).
    cache = {}
    outputs = {}
    for variant in selected:
        state_path = directory / variant / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            'tokens': list(started['tokens']), 'generated': [], 'ended': False, 'response': '', 'steps': []}

        def end(reason):
            state.update(ended=True, stop_reason=reason)
            write_json(state_path, state)

        if not state['ended'] and degenerate(state['generated']):
            end('degenerate_repeat')
        while len(state['generated']) < config['max_tokens'] and not state['ended']:
            step = directory / variant / f"step-{len(state['generated']):04d}"

            def call(action, validate=None, **kwargs):
                request = {'action': action, **kwargs}
                key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
                operation = ((lambda: cache[key]) if key in cache else (lambda: api(action, **kwargs)))
                result = recorded(step / f'{action}.json', request, operation, validate)
                if action in ('inspect', 'explain'):
                    cache[key] = result
                return result

            baseline = not fast or variant == 'original'
            try:
                inspected = call('inspect', tokens=state['tokens'],
                                 validate=lambda r: isinstance(r.get('h'), list) and
                                 (not baseline or valid_prediction(r.get('baseline'))),
                                 **({} if baseline else {'baseline': False}))
                if variant == 'original':
                    prediction = inspected['baseline']
                else:
                    explained = call('explain', tokens=state['tokens'], h=inspected['h'])
                    edited = (explained['text'] if variant == 'roundtrip' else recorded(
                        step / 'editor.json', {'instruction': config['instruction'], 'text': explained['text']},
                        lambda: editor(config['instruction'], explained['text']))['proposal'])
                    wanted = [variant] if fast else list(VARIANTS)
                    compared = call('compare', tokens=state['tokens'], edited=edited,
                                    reconstruction=explained['reconstruction'],
                                    validate=lambda r: all(valid_prediction(r.get(v), variants is not None)
                                                           for v in wanted),
                                    **({'variants': [variant]} if fast else {}),
                                    **({'multiplier': config['steering_multiplier']}
                                       if 'steering_multiplier' in config else {}))
                    prediction = compared[variant]
            except RuntimeError as exc:
                # Deterministic: the same context fails the same way on every resume.
                if UNPARSEABLE not in str(exc):
                    raise
                end('backend_parse_error')
                break
            if prediction == {}:
                end('backend_no_prediction')
                break
            token = stepper.prediction_token_id(prediction)
            generated = state['generated'] + [token]
            state['steps'].append({'index': len(state['generated']), 'chosen_id': token, 'variant': variant,
                                   'artifacts': str(step.relative_to(directory))})
            state.update(tokens=state['tokens'] + [token], generated=generated, ended=token in eos,
                         response=call('decode', tokens=generated)['content'])
            if token in eos:
                state['stop_reason'] = 'eos'
            elif degenerate(generated):
                state.update(ended=True, stop_reason='degenerate_repeat')
            write_json(state_path, state)
            progress(directory, config['max_tokens'])
            print(f'{directory.name} {variant}: {len(generated)}/{config["max_tokens"]}', flush=True)
        outputs[variant] = {'response': state['response'], 'generated_token_ids': state['generated'],
                            'stop_reason': state.get('stop_reason', 'eos') if state['ended'] else 'token_limit'}
        write_json(directory / 'responses.json', outputs)
    write_json(directory / 'complete.json', {'completed_at': datetime.datetime.now(datetime.UTC).isoformat(),
                                             'variants': list(outputs)})
    progress(directory, config['max_tokens'])
    return outputs
