"""Run checkpointed trajectories on the GPU node itself, with the local-Qwen editor.

No laptop, SSH tunnel or external editor is involved, so the run survives laptop
disconnects. Produces the same artifacts and configs as hpc_control.py --editor local.
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from nla import openai_editor
import trajectories

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
    p.add_argument('--configs', type=Path, nargs='+', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--max-tokens', type=int, default=10000)
    p.add_argument('--editor', choices=('local', 'openai'), default='local')
    args = p.parse_args()
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
    if args.editor == 'openai':
        openai_editor.api_key()  # fail fast if .env lacks the key
    steering_self_test(api)
    for path in args.configs:
        private = json.loads(path.read_text())
        if not private.get('prompt', '').strip() or not private.get('instruction', '').strip():
            raise ValueError('Empty experiment prompt or editor instruction')
        # Must equal hpc_control.py's config so existing checkpoints resume.
        cfg = {'prompt': private['prompt'], 'instruction': private['instruction'], 'max_tokens': args.max_tokens,
               'editor_model': EDITOR_MODEL if args.editor == 'local' else openai_editor.MODEL, 'variants': list(trajectories.VARIANTS),
               'policy': 'independent prefixes; greedy; one-position intervention; no KV-edit replay'}
        if args.editor == 'openai':
            editor = openai_editor.propose
        else:
            editor = lambda instruction, text: api('propose', tokens=[1], editor='local', instruction=instruction, text=text)
        for retry in range(20):
            try:
                trajectories.generate(cfg, args.output / path.parent.parent.name, api, editor)
                break
            except RuntimeError as exc:
                # Occasional NaN activations are transient; the checkpoint resumes the same step.
                if not any(m in str(exc) for m in ('Non-finite', 'NoneType')) or retry == 19:
                    raise
                print(f'Transient non-finite activation; retrying step ({retry + 1})', flush=True)
        print(f'{path.parent.parent.name} complete', flush=True)


if __name__ == '__main__':
    main()
