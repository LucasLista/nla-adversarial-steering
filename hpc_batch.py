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
            # Status only: bodies can echo request content.
            raise RuntimeError(f'Backend HTTP {exc.code}') from None
        if isinstance(result.get('error'), str):
            raise RuntimeError(result['error'])
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--configs', type=Path, nargs='+', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--max-tokens', type=int, default=10000)
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
    for path in args.configs:
        private = json.loads(path.read_text())
        if not private.get('prompt', '').strip() or not private.get('instruction', '').strip():
            raise ValueError('Empty experiment prompt or editor instruction')
        # Must equal hpc_control.py's config so existing checkpoints resume.
        cfg = {'prompt': private['prompt'], 'instruction': private['instruction'], 'max_tokens': args.max_tokens,
               'editor_model': EDITOR_MODEL, 'variants': list(trajectories.VARIANTS),
               'policy': 'independent prefixes; greedy; one-position intervention; no KV-edit replay'}
        editor = lambda instruction, text: api('propose', tokens=[1], editor='local', instruction=instruction, text=text)
        trajectories.generate(cfg, args.output / path.parent.parent.name, api, editor)
        print(f'{path.parent.parent.name} complete', flush=True)


if __name__ == '__main__':
    main()
