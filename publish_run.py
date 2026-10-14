"""Copy an authorized response snapshot to Git logs without displaying its text."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ('original', 'roundtrip', 'edited_raw', 'edited_corrected')


def publish(run, name, limit=256, branch_runs=None):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Use a single directory name for the log')
    source = Path(run)
    destination = ROOT / 'logs' / name
    destination.mkdir(parents=True, exist_ok=True)
    progress = {}
    responses = {}
    for variant in VARIANTS:
        path = Path((branch_runs or {}).get(variant, source)) / variant / 'state.json'
        if path.is_file():
            state = json.loads(path.read_text())
            count = len(state['generated'])
            progress[variant] = {'tokens': count, 'ended': state['ended']}
            # Runtime-only transfer: do not print or interpret response content.
            responses[variant] = {
                'response': state['response'],
                'generated_token_ids': state['generated'],
                'stop_reason': 'eos' if state['ended'] else 'token_limit' if count >= limit else 'in_progress',
            }
    if not responses:
        raise ValueError('No branch checkpoints available yet')
    temporary = destination / 'responses.tmp'
    temporary.write_text(json.dumps(responses, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(destination / 'responses.json')
    snapshot = {
        'complete': len(progress)==len(VARIANTS) and all(v['ended'] or v['tokens']>=limit for v in progress.values()),
        'branches': progress,
        'responses_sha256': hashlib.sha256((destination / 'responses.json').read_bytes()).hexdigest(),
    }
    (destination / 'status.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    print('Response snapshot copied; content was not displayed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--name', required=True)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--branch-run', action='append', default=[], help='variant=worker-directory')
    args = parser.parse_args()
    publish(args.run, args.name, args.max_tokens, dict(item.split('=',1) for item in args.branch_run))
