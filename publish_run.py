"""Copy an authorized response snapshot to Git logs without displaying its text."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
VARIANTS = ('original', 'roundtrip', 'edited_raw', 'edited_corrected')


def publish(run, name):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Use a single directory name for the log')
    source = Path(run)
    responses = source / 'responses.json'
    if not responses.is_file():
        raise ValueError('No completed branch response snapshot yet')
    destination = ROOT / 'logs' / name
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / 'responses.tmp'
    shutil.copyfile(responses, temporary)
    temporary.replace(destination / 'responses.json')
    progress = {}
    for variant in VARIANTS:
        path = source / variant / 'state.json'
        if path.is_file():
            state = json.loads(path.read_text())
            progress[variant] = {'tokens': len(state['generated']), 'ended': state['ended']}
    snapshot = {
        'complete': (source / 'complete.json').is_file(),
        'branches': progress,
        'responses_sha256': hashlib.sha256((destination / 'responses.json').read_bytes()).hexdigest(),
    }
    (destination / 'status.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    print('Response snapshot copied; content was not displayed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    publish(args.run, args.name)
