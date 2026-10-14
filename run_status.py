"""Report experiment progress metadata without displaying logs or model text."""
import argparse
import json
from pathlib import Path

VARIANTS = ('original', 'roundtrip', 'edited_raw', 'edited_corrected')


def status(directory, limit=256):
    progress = directory / 'progress.json'
    if progress.exists():
        data = json.loads(progress.read_text())
        source = data['branches']
    else:
        source = {}
        for variant in VARIANTS:
            path = directory / variant / 'state.json'
            if path.exists():
                state = json.loads(path.read_text())
                count = len(state['generated'])
                source[variant] = {'tokens': count, 'ended': bool(state['ended']),
                                   'complete': bool(state['ended']) or count >= limit}
    branches = {}
    for variant in VARIANTS:
        item = source.get(variant, {})
        branches[variant] = {'tokens': int(item.get('tokens', 0)),
                             'ended': bool(item.get('ended', False)),
                             'complete': bool(item.get('complete', False))}
    return {'branches': branches, 'complete': (directory / 'complete.json').is_file()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(status(args.run, args.max_tokens), indent=2))
