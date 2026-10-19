"""Prepare private inputs for the five confirmed DECCP cases. Does not submit jobs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT / 'config/deccp-refusal-v2.json').read_text())
    assert hashlib.sha256(plan['instruction'].encode()).hexdigest() == plan['instruction_sha256']
    # Refuse to alter an existing campaign's inputs/checkpoints.
    args.output.mkdir(parents=True, exist_ok=False)
    for case in plan['cases']:
        path = args.output / 'inputs' / case['id'] / 'inputs' / 'config.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'prompt': case['prompt'], 'instruction': plan['instruction'] +
                        '\nQuestion (context only): ' + json.dumps(case['prompt'], ensure_ascii=False), 'editor': 'codex'},
                        ensure_ascii=False, indent=2) + '\n')
        case['config'] = str(path.resolve())
    (args.output / 'inputs/manifest.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(f'Prepared {len(plan["cases"])} case configs; no jobs submitted.')


if __name__ == '__main__':
    main()
