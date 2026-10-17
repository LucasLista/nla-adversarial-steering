"""Run checkpointed trajectories on the GPU node itself, with the local-Qwen editor.

No laptop, SSH tunnel or external editor is involved, so the run survives laptop
disconnects. Produces the same artifacts and configs as hpc.hpc_control --editor local.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from nla import openai_editor
from nla import trajectories
from nla.benchmark import manifest_configs

EDITOR_MODEL = 'Qwen2.5-7B-Instruct-Q4_K_M'


class AuthAPI(trajectories.API):
    def __init__(self, url, token):
        super().__init__(url)
        self.token = token

    def __call__(self, action, **kwargs):
        req = urllib.request.Request(self.url, json.dumps({'action': action, **kwargs}).encode(),
                                     {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.token})
        try:
            with urllib.request.urlopen(req, timeout=1200) as r:
                result = json.load(r)
        except urllib.error.HTTPError as exc:
            # Keep the server's own error string (diagnostic), truncated; never the request.
            try:
                detail = str(json.loads(exc.read()).get('error', ''))[:300]
            except (ValueError, AttributeError):
                detail = ''
            raise RuntimeError(f'Backend HTTP {exc.code}: {detail}') from None
        if isinstance(result.get('error'), str):
            raise RuntimeError(result['error'])
        return result


def steering_self_test(api):
    """Abort unless an intervention changes the prediction beyond the first microbatch."""
    prompt = 'Repeat this sentence back to me exactly: ' + 'the quick brown fox jumps over the lazy dog. ' * 20
    tokens = api('start', prompt=prompt)['tokens']
    if len(tokens) <= 128:
        raise RuntimeError('Self-test context too short')
    h = api('inspect', tokens=tokens)['h']
    explained = api('explain', tokens=tokens, h=h)
    compared = api('compare', tokens=tokens, reconstruction=explained['reconstruction'],
                   edited='A detailed description of tropical fish, coral reefs, and ocean currents.')
    top = lambda key: compared[key]['completion_probabilities'][0]['top_logprobs']
    if top('original') == top('edited_raw'):
        raise RuntimeError(f'Steering self-test failed: intervention had no effect at context {len(tokens)}')
    print(f'Steering self-test passed at context {len(tokens)}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--configs', type=Path, nargs='+')
    inputs.add_argument('--benchmark-manifest', type=Path)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--max-tokens', type=int, default=10000)
    p.add_argument('--editor', choices=('local', 'openai'), default='local')
    p.add_argument('--editor-model', help='API editor model, e.g. gpt-6-luna')
    p.add_argument('--variants', nargs='+', choices=trajectories.VARIANTS)
    p.add_argument('--fast', action='store_true', help='Compute only the selected branch distribution')
    p.add_argument('--multipliers', type=float, nargs='+',
                   help='Sweep h + m*delta for each m; 0 runs the unsteered branch')
    p.add_argument('--shard', help='i/n: run only this slice of the cases, for parallel jobs')
    args = p.parse_args()
    shard, shards = 0, 1
    if args.shard:
        try:
            shard, shards = (int(part) for part in args.shard.split('/'))
        except ValueError:
            p.error('--shard must look like i/n')
        if not 0 <= shard < shards:
            p.error('--shard index must be below the shard count')
    if args.multipliers is not None:
        if not args.multipliers or any(not 0 <= m <= 100 for m in args.multipliers):
            p.error('--multipliers must be between 0 and 100')
        if len(set(args.multipliers)) != len(args.multipliers):
            p.error('--multipliers must be distinct')
        if args.variants:
            p.error('--multipliers selects its own variant per multiplier')
    manifest = None
    if args.benchmark_manifest:
        manifest, args.configs = manifest_configs(args.benchmark_manifest)
        edited_requested = any(v.startswith('edited') for v in (args.variants or [])) or \
            any(m != 0 for m in (args.multipliers or []))
        if not manifest['editing_enabled'] and edited_requested:
            p.error('Edited benchmark branches require a private --instruction-file during preparation')
    variants = args.variants or (['original'] if manifest and not manifest['editing_enabled'] else list(trajectories.VARIANTS))
    if not 1 <= args.max_tokens <= 10000:
        p.error('--max-tokens must be between 1 and 10000')
    token = args.token_file.read_text().strip()
    url = f'http://127.0.0.1:{args.port}'
    api = AuthAPI(url, token)
    for _ in range(360):
        try:
            req = urllib.request.Request(url + '/setup-status', headers={'Authorization': 'Bearer ' + token})
            with urllib.request.urlopen(req, timeout=10):
                break
        except OSError:
            time.sleep(10)
    else:
        raise RuntimeError('Server did not become ready')
    editing = any(v.startswith('edited') for v in variants)
    if args.editor == 'openai' and editing:
        if args.editor_model:
            os.environ['NLA_OPENAI_MODEL'] = args.editor_model
        openai_editor.api_key()  # fail fast if .env lacks the key
        print(f'Editor preflight passed: {openai_editor.preflight()}', flush=True)
    if variants != ['original']:
        steering_self_test(api)
    args.configs = args.configs[shard::shards]
    if not args.configs:
        raise RuntimeError('Shard selected no cases')
    # Each multiplier is an independent trajectory set under its own directory.
    sweep = [(m, f'm{m:g}') for m in args.multipliers] if args.multipliers else [(None, '')]
    # Case-major: an interrupted job still leaves complete alpha curves for finished cases.
    for multiplier, label, path in [(m, l, c) for c in args.configs for m, l in sweep]:
        private = json.loads(path.read_text())
        if not private.get('prompt', '').strip() or not private.get('instruction', '').strip():
            raise ValueError('Empty experiment prompt or editor instruction')
        # Must equal hpc.hpc_control's config so existing checkpoints resume.
        cfg = {'prompt': private['prompt'], 'instruction': private['instruction'], 'max_tokens': args.max_tokens,
               'editor_model': EDITOR_MODEL if args.editor == 'local' else openai_editor.model(),
               'variants': list(variants),
               'policy': 'independent prefixes; greedy; one-position intervention; no KV-edit replay'}
        if args.fast:
            cfg['comparison_mode'] = 'selected'
        destination = args.output / label / path.parent.parent.name if label else args.output / path.parent.parent.name
        if multiplier is not None:
            # m = 0 is the unsteered baseline, so it needs no explanation or edit.
            cfg['variants'] = ['original'] if multiplier == 0 else ['edited_corrected']
            if multiplier != 0:
                cfg['steering_multiplier'] = multiplier
        if args.editor == 'openai':
            editor = openai_editor.propose
        else:
            editor = lambda instruction, text: api('propose', tokens=[1], editor='local', instruction=instruction, text=text)
        for retry in range(20):
            try:
                trajectories.generate(cfg, destination, api, editor, variants=cfg['variants'])
                break
            except RuntimeError as exc:
                # Occasional NaN activations are transient; the checkpoint resumes the same step.
                if not any(m in str(exc) for m in ('Non-finite', 'NoneType')) or retry == 19:
                    raise
                print(f'Transient non-finite activation; retrying step ({retry + 1})', flush=True)
        print(f'{label or "run"}/{path.parent.parent.name} complete', flush=True)


if __name__ == '__main__':
    import sys
    import traceback
    try:
        main()
    except Exception as exc:  # not SystemExit: --help and clean exits must stay zero
        # Sanctioned diagnostics: exception class and source frames only, never
        # messages or payloads, so failures are traceable without reading logs.
        health = {'phase': 'batch_failed', 'error_type': type(exc).__name__,
                  'job': os.environ.get('LSB_JOBID', 'local'),
                  'frames': [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno}
                             for f in traceback.extract_tb(exc.__traceback__)][-8:]}
        path = Path('runs') / f"batch-health-{health['job']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(health, indent=2))
        raise SystemExit(1)
