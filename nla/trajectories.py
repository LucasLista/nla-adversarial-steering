"""Four independent greedy NLA trajectories, with durable per-stage checkpoints.

Run on the signed-in laptop against an SSH-forwarded HPC launcher. No model
weights or activation computation are needed on the laptop.
"""
import argparse
from collections import OrderedDict
import copy
import hashlib
import datetime
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error
from nla import codex_editor, stepper

REPEAT_LIMIT = 50  # identical tokens in a row that end a branch as degenerate


UNPARSEABLE = 'Failed to parse input at pos'


def degenerate(generated):
    return len(generated) >= REPEAT_LIMIT and len(set(generated[-REPEAT_LIMIT:])) == 1


VARIANTS = ('original', 'roundtrip', 'edited_raw', 'edited_corrected')
EOS = {151643, 151645}

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def _valid_prediction_response(result):
    try:
        stepper.validate_prediction(result)
        return True
    except stepper.InvalidPredictionResponse:
        return False

def _valid_comparison_response(result, variants, allow_no_prediction=False):
    return (isinstance(result, dict)
            and all(variant in result and (_valid_prediction_response(result[variant])
                    or (allow_no_prediction and result[variant] == {}))
                    for variant in variants))

def _valid_inspection_response(result, baseline_required):
    return (isinstance(result, dict) and 'h' in result and isinstance(result['h'], list)
            and (not baseline_required or _valid_prediction_response(result.get('baseline'))))

def recorded(path, request, operation, validate=None):
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
            raise ValueError('Backend returned an invalid checkpoint response')
    except Exception as exc:
        write_json(path.with_name(path.stem + f'-error-{time.time_ns()}.json'),
                   {'request': request, 'error': str(exc), 'started_at': start,
                    'finished_at': time.time()})
        raise
    write_json(path, {'request': request, 'response': result,
                      'started_at': start, 'finished_at': time.time(),
                      'duration_seconds': time.time() - start})
    return result

class API:
    def __init__(self, url):
        self.url = url.rstrip('/') + '/stepper-api'
    def __call__(self, action, **kwargs):
        req = urllib.request.Request(self.url, json.dumps({'action': action, **kwargs}).encode(),
                                     {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=1200) as r:
                result = json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code in (408, 502, 503, 504):
                # Let controllers retry the same durable checkpoint after a
                # temporary gateway/service failure. Never print error bodies.
                raise ConnectionError(f'Backend temporarily unavailable (HTTP {exc.code})') from None
            raise RuntimeError(f'Backend HTTP {exc.code}: {exc.read().decode()}') from exc
        if isinstance(result.get('error'), str):
            raise RuntimeError(result['error'])
        return result

def write_progress(directory, limit):
    branches = {}
    for variant in VARIANTS:
        path = directory / variant / 'state.json'
        if path.exists():
            state = json.loads(path.read_text())
            count = len(state['generated'])
            branches[variant] = {'tokens': count, 'ended': state['ended'],
                                 'complete': state['ended'] or count >= limit}
        else:
            branches[variant] = {'tokens': 0, 'ended': False, 'complete': False}
    write_json(directory / 'progress.json', {'branches': branches,
               'complete': (directory / 'complete.json').exists()})


class PrefixCache:
    """Bounded, run-local reuse of deterministic requests, never editor proposals.

    Full request hashes include tokens, activations and baseline flags. Entries
    are populated only after checkpoint validation; failed calls are not cached.
    No KV state or interventions are carried across independent prefixes.
    """
    def __init__(self, capacity=2048):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = 0

    def recorded(self, path, request, operation, validate=None):
        eligible = request['action'] in ('inspect', 'explain')
        key = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).digest()
        def compute():
            if eligible and key in self.entries:
                self.hits += 1
                self.entries.move_to_end(key)
                return copy.deepcopy(self.entries[key])
            return operation()
        result = recorded(path, request, compute, validate)
        if eligible:
            self.entries[key] = copy.deepcopy(result)
            self.entries.move_to_end(key)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        return result


def generate(config, directory, api, editor, variants=None, execution_order=None, cache_shared_prefixes=False):
    directory = Path(directory)
    metadata = directory / 'config.json'
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        if previous != config:
            old_limit = previous.get('max_tokens', 0)
            new_limit = config.get('max_tokens', 0)
            old_settings = {k:v for k,v in previous.items() if k != 'max_tokens'}
            new_settings = {k:v for k,v in config.items() if k != 'max_tokens'}
            if old_settings != new_settings or new_limit <= old_limit:
                raise ValueError('Run configuration differs; only increasing the token limit is supported')
            write_json(metadata, config)
            (directory / 'complete.json').unlink(missing_ok=True)
            write_progress(directory, new_limit)
    else:
        write_json(metadata, config)
    started = recorded(directory / 'start.json', {'prompt': config['prompt']},
                       lambda: api('start', prompt=config['prompt']))
    initial = started['tokens']
    eos = set(started.get('eos_token_ids', EOS))
    outputs = {}
    prefix_cache = PrefixCache() if cache_shared_prefixes else None
    selected = tuple(variants or VARIANTS)
    if not selected or any(v not in VARIANTS for v in selected):
        raise ValueError("Unknown trajectory variant")
    if execution_order is not None:
        if len(execution_order) != len(selected) or set(execution_order) != set(selected):
            raise ValueError('Execution order must contain each selected branch exactly once')
        selected = tuple(execution_order)
    for variant in selected:
        state_path = directory / variant / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            'tokens': initial.copy(), 'generated': [], 'ended': False, 'response': '', 'steps': []}
        if not state['ended'] and degenerate(state['generated']):
            state.update(ended=True, stop_reason='degenerate_repeat')
            write_json(state_path, state)
        while len(state['generated']) < config['max_tokens'] and not state['ended']:
          try:
            step = directory / variant / f"step-{len(state['generated']):04d}"
            def call(action, **kwargs):
                request = {'action': action, **kwargs}
                validator = None
                if action == 'compare':
                    validator = lambda result: _valid_comparison_response(
                        result, kwargs.get('variants', VARIANTS),
                        allow_no_prediction=variants is not None)
                elif action == 'inspect':
                    validator = lambda result: _valid_inspection_response(
                        result, kwargs.get('baseline', True))
                recorder = prefix_cache.recorded if prefix_cache else recorded
                return recorder(step / (action + '.json'), request,
                                lambda: api(action, **kwargs), validator)
            fast = config.get('comparison_mode') == 'selected'
            inspected = call('inspect', tokens=state['tokens'],
                             **({'baseline': False} if fast and variant != 'original' else {}))
            if variant == 'original':
                prediction = inspected['baseline']
            else:
                explained = call('explain', tokens=state['tokens'], h=inspected['h'])
                if variant == 'roundtrip':
                    edited = {'proposal': explained['text']}
                else:
                    edited = recorded(step / 'editor.json',
                                      {'instruction': config['instruction'], 'text': explained['text']},
                                      lambda: editor(config['instruction'], explained['text']))
                compared = call('compare', tokens=state['tokens'], edited=edited['proposal'],
                                reconstruction=explained['reconstruction'],
                                **({'variants': [variant]} if fast else {}),
                                **({'multiplier': config['steering_multiplier']}
                                   if 'steering_multiplier' in config else {}))
                prediction = compared[variant]
            if variants is not None and prediction == {}:
                state.update(ended=True, stop_reason='backend_no_prediction')
                write_json(state_path, state)
                write_progress(directory, config['max_tokens'])
                print(f'{variant}: stopped, backend returned no next-token prediction', flush=True)
                break
            token = stepper.prediction_token_id(prediction)
            generated = state['generated'] + [token]
            decoded = call('decode', tokens=generated)['content']
            state['steps'].append({'index': len(state['generated']), 'chosen_id': token,
                                   'variant': variant, 'artifacts': str(step.relative_to(directory))})
            state.update(tokens=state['tokens'] + [token], generated=generated,
                         ended=token in eos, response=decoded)
            if token in eos:
                state['stop_reason'] = 'eos'
            elif degenerate(generated):
                state.update(ended=True, stop_reason='degenerate_repeat')
            write_json(state_path, state)
            write_progress(directory, config['max_tokens'])
            print(f'{variant}: {len(generated)}/{config["max_tokens"]}, token={token}', flush=True)
          except RuntimeError as exc:
            # Deterministic: the same context reproduces it on every resume.
            if UNPARSEABLE not in str(exc):
                raise
            state.update(ended=True, stop_reason='backend_parse_error')
            write_json(state_path, state)
            write_progress(directory, config['max_tokens'])
            print(f'{variant}: stopped, backend could not parse the generated token', flush=True)
        outputs[variant] = {'response': state['response'], 'generated_token_ids': state['generated'],
                            'stop_reason': state.get('stop_reason', 'eos') if state['ended'] else 'token_limit'}
        write_json(directory / 'responses.json', outputs)
    write_json(directory / 'complete.json', {'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                           'variants': list(outputs)})
    write_progress(directory, config['max_tokens'])
    if prefix_cache:
        write_json(directory / 'cache-stats.json', {'hits': prefix_cache.hits, 'entries': len(prefix_cache.entries)})
    return outputs

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:18090')
    p.add_argument('--prompt', required=True)
    p.add_argument('--instruction', required=True)
    p.add_argument('--max-tokens', type=int, default=64)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.max_tokens <= 10000:
        p.error('--max-tokens must be between 1 and 10000')
    config = {'prompt': args.prompt, 'instruction': args.instruction, 'max_tokens': args.max_tokens,
              'editor_model': codex_editor.MODEL, 'variants': list(VARIANTS),
              'policy': 'independent prefixes; greedy; one-position intervention; no KV-edit replay'}
    generate(config, args.output, API(args.url), codex_editor.propose)

if __name__ == '__main__':
    main()
