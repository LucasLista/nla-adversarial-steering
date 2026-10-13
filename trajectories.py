"""Four independent greedy NLA trajectories, with durable per-stage checkpoints.

Run on the signed-in laptop against an SSH-forwarded HPC launcher. No model
weights or activation computation are needed on the laptop.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error
import codex_editor

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

def recorded(path, request, operation):
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['request'] != request:
            raise ValueError(f'Checkpoint request changed: {path}')
        return saved['response']
    start = time.time()
    try:
        result = operation()
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
            raise RuntimeError(f'Backend HTTP {exc.code}: {exc.read().decode()}') from exc
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

def generate(config, directory, api, editor):
    directory = Path(directory)
    metadata = directory / 'config.json'
    if metadata.exists():
        if json.loads(metadata.read_text()) != config:
            raise ValueError('Run configuration differs; use another output directory')
    else:
        write_json(metadata, config)
    initial = recorded(directory / 'start.json', {'prompt': config['prompt']},
                       lambda: api('start', prompt=config['prompt']))['tokens']
    outputs = {}
    for variant in VARIANTS:
        state_path = directory / variant / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            'tokens': initial.copy(), 'generated': [], 'ended': False, 'response': '', 'steps': []}
        while len(state['generated']) < config['max_tokens'] and not state['ended']:
            step = directory / variant / f"step-{len(state['generated']):04d}"
            def call(action, **kwargs):
                request = {'action': action, **kwargs}
                return recorded(step / (action + '.json'), request, lambda: api(action, **kwargs))
            inspected = call('inspect', tokens=state['tokens'])
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
                                reconstruction=explained['reconstruction'])
                prediction = compared[variant]
            token = prediction['completion_probabilities'][0]['id']
            if type(token) is not int:
                raise ValueError('Backend did not return an integer next-token ID')
            generated = state['generated'] + [token]
            decoded = call('decode', tokens=generated)['content']
            state['steps'].append({'index': len(state['generated']), 'chosen_id': token,
                                   'variant': variant, 'artifacts': str(step.relative_to(directory))})
            state.update(tokens=state['tokens'] + [token], generated=generated,
                         ended=token in EOS, response=decoded)
            write_json(state_path, state)
            print(f'{variant}: {len(generated)}/{config["max_tokens"]}, token={token}', flush=True)
        outputs[variant] = {'response': state['response'], 'generated_token_ids': state['generated'],
                            'stop_reason': 'eos' if state['ended'] else 'token_limit'}
        write_json(directory / 'responses.json', outputs)
    write_json(directory / 'complete.json', {'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                           'variants': list(outputs)})
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
