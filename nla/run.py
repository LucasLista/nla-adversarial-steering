"""Run four-branch NLA trajectories for a set of prompts against a running nla.server.

  python -m nla.run --cases runs/x/cases.json --instruction-file runs/x/instruction.txt --output runs/x
  python -m nla.run --prompt "..." --instruction "..." --output runs/one
  python -m nla.run --status runs/x

A cases file is a JSON list of {"id", "prompt"} objects, optionally with a
per-case "instruction". "{prompt}" in an instruction is replaced by the case's
prompt. Runs resume from their checkpoints; rerunning a command is always safe.
"""
import argparse
import http.client
import json
from pathlib import Path
import time
import urllib.request
from nla import editor, trajectories

POLICY = 'independent prefixes; greedy; one-position intervention; no KV-edit replay'


def wait_until_ready(url, token, timeout=3600):
    headers = {'Authorization': 'Bearer ' + token} if token else {}
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(urllib.request.Request(url + '/health', headers=headers), timeout=10) as r:
                if json.load(r).get('ready'):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(10)
    raise RuntimeError('Server did not become ready')


def steering_self_test(api):
    """Fail fast unless an edit changes the prediction beyond the first microbatch."""
    tokens = api('start', prompt='Repeat this sentence back to me exactly: '
                 + 'the quick brown fox jumps over the lazy dog. ' * 20)['tokens']
    explained = api('explain', tokens=tokens, h=api('inspect', tokens=tokens)['h'])
    compared = api('compare', tokens=tokens, reconstruction=explained['reconstruction'],
                   edited='A detailed description of tropical fish, coral reefs, and ocean currents.')
    top = lambda key: compared[key]['completion_probabilities'][0]['top_logprobs']
    if top('original') == top('edited_raw'):
        raise RuntimeError(f'Steering self-test failed at context {len(tokens)}')


def status(output):
    """Print per-case progress metadata only."""
    for path in sorted(Path(output).rglob('progress.json')):
        data = json.loads(path.read_text())
        tokens = {v: b['tokens'] for v, b in data['branches'].items()}
        print(path.parent.relative_to(output), 'complete' if data['complete'] else tokens)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--status', type=Path, metavar='OUTPUT', help='summarize progress and exit')
    p.add_argument('--cases', type=Path)
    p.add_argument('--prompt')
    p.add_argument('--instruction')
    p.add_argument('--instruction-file', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--url', default='http://127.0.0.1:18090')
    p.add_argument('--token-file', type=Path)
    p.add_argument('--editor', choices=sorted(editor.EDITORS), default='codex')
    p.add_argument('--editor-model', default=editor.MODEL)
    p.add_argument('--variants', nargs='+', choices=trajectories.VARIANTS,
                   help='run only these branches; "original" alone screens prompts for refusals')
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--fast', action='store_true', help='predict only the branch being generated')
    p.add_argument('--multipliers', type=float, nargs='+',
                   help='sweep h + m*(AR(edited) - AR(original)) over m; m=0 is the unsteered baseline')
    p.add_argument('--shard', default='0/1', help='i/n: run every n-th case starting at i')
    args = p.parse_args()
    if args.status:
        return status(args.status)
    if not args.output or bool(args.cases) == bool(args.prompt):
        p.error('give --output and exactly one of --cases or --prompt')
    if args.multipliers and args.variants:
        p.error('--multipliers chooses its own branches')
    shard, shards = map(int, args.shard.split('/'))
    cases = json.loads(args.cases.read_text()) if args.cases else [{'id': '', 'prompt': args.prompt}]
    shared = args.instruction_file.read_text() if args.instruction_file else args.instruction
    token = args.token_file.read_text().strip() if args.token_file else None
    api = trajectories.API(args.url, token)
    variants = args.variants or list(trajectories.VARIANTS)
    sweep = [(m, f'm{m:g}') for m in args.multipliers] if args.multipliers else [(None, '')]
    editing = any(v.startswith('edited') for v in variants) or any(m for m, _ in sweep)

    wait_until_ready(args.url.rstrip('/'), token)
    if editing:
        steering_self_test(api)
    edit = lambda instruction, text: editor.propose(args.editor, instruction, text, args.editor_model)
    # Case-major, so an interrupted sweep still leaves complete curves for finished cases.
    for case in cases[shard::shards]:
        for multiplier, label in sweep:
            instruction = case.get('instruction', shared) or ''
            if editing and not instruction.strip():
                p.error(f"case {case['id']!r} has no editing instruction")
            config = {'prompt': case['prompt'], 'instruction': instruction.replace('{prompt}', case['prompt']),
                      'max_tokens': args.max_tokens, 'editor_model': args.editor_model,
                      'variants': variants, 'policy': POLICY}
            if args.fast:
                config['comparison_mode'] = 'selected'
            if multiplier is not None:
                config['variants'] = ['original'] if multiplier == 0 else ['edited_corrected']
                if multiplier:
                    config['steering_multiplier'] = multiplier
            directory = args.output / label / case['id']
            selected = None if config['variants'] == list(trajectories.VARIANTS) else config['variants']
            for attempt in range(20):
                try:
                    trajectories.generate(config, directory, api, edit, selected)
                    break
                except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
                    # Connection drops and editor network errors resume from the checkpoint.
                    if attempt == 19:
                        raise
                    print(f'{type(exc).__name__}; retrying from checkpoint in 30 s', flush=True)
                    time.sleep(30)


if __name__ == '__main__':
    main()
