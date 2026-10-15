"""Portable experiment inputs and interface-compatible result snapshots."""
import json
import re
import time
from pathlib import Path

def run(root, body):
    root = Path(root) / 'experiments'
    action = body['action']
    if action == 'experiments':
        return {'experiments': sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []}
    name = body.get('experiment', '')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', name):
        raise ValueError('Use an experiment name containing letters, numbers, hyphens or underscores')
    folder = root / name
    if folder.is_symlink():
        raise ValueError('Symlink experiments are unsupported')
    if action == 'experiment-input':
        return json.loads((folder / 'inputs' / 'config.json').read_text())
    results = folder / 'results'
    if action == 'results':
        return {'results': sorted(p.name for p in results.glob('*.json'))}
    if action == 'save-result':
        results.mkdir(parents=True, exist_ok=True)
        path = results / f'{time.time_ns()}.json'
        path.write_text(json.dumps(body['session'], ensure_ascii=False, indent=2))
        return {'saved': path.name}
    name = body.get('result', '')
    if action == 'open-result' and re.fullmatch(r'[a-zA-Z0-9_.-]+\.json', name):
        path = results / name
        if path.is_symlink():
            raise ValueError('Symlink results are unsupported')
        return json.loads(path.read_text())
    raise ValueError('Unknown experiment action')
