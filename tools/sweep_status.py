"""Report alpha-sweep progress as counts only, never response content."""
import argparse
import json
from pathlib import Path


def branch(directory):
    """Token count, ended flag and stop reason for one trajectory directory."""
    for variant in ('original', 'edited_corrected'):
        state = directory / variant / 'state.json'
        if state.exists():
            saved = json.loads(state.read_text())
            return {'variant': variant, 'tokens': len(saved.get('generated', [])),
                    'ended': bool(saved.get('ended')), 'stop_reason': saved.get('stop_reason')}
    return None


def collect(root):
    root = Path(root)
    rows = {}
    for multiplier in sorted(root.glob('m*')):
        if not multiplier.is_dir():
            continue
        for case in sorted(multiplier.iterdir()):
            if case.is_dir() and (status := branch(case)):
                rows.setdefault(multiplier.name, {})[case.name] = status
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/alpha-sweep'))
    parser.add_argument('--expected-cases', type=int, default=10)
    args = parser.parse_args()
    rows = collect(args.root)
    summary = []
    for multiplier, cases in rows.items():
        done = [c for c in cases.values() if c['ended']]
        reasons = {}
        for case in done:
            reasons[case['stop_reason'] or 'unknown'] = reasons.get(case['stop_reason'] or 'unknown', 0) + 1
        summary.append({'multiplier': multiplier, 'started': len(cases),
                        'complete': len(done), 'expected': args.expected_cases,
                        'tokens': sum(c['tokens'] for c in cases.values()),
                        'stop_reasons': reasons})
    total_done = sum(row['complete'] for row in summary)
    print(json.dumps({'branches_complete': total_done,
                      'branches_expected': args.expected_cases * 10,
                      'by_multiplier': summary}, indent=2))


if __name__ == '__main__':
    main()
