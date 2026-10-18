"""Summarise the alpha sweep as per-multiplier statistics.

Reports only counts, token-identity comparisons and numeric metrics. Generated
text, explanations and edits are never read or printed; use the viewer for those.
"""
import argparse
import json
import statistics as st
from pathlib import Path

VARIANTS = ('original', 'edited_corrected')
WINDOW = 100  # tokens scored for the length-independent degeneracy proxy


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def state_of(case):
    for variant in VARIANTS:
        saved = read(case / variant / 'state.json')
        if saved:
            return variant, saved
    return None, {}


def longest_run(tokens):
    best = run = 1 if tokens else 0
    for previous, current in zip(tokens, tokens[1:]):
        run = run + 1 if current == previous else 1
        best = max(best, run)
    return best


def first_divergence(tokens, baseline):
    for index, (a, b) in enumerate(zip(tokens, baseline)):
        if a != b:
            return index
    return None if tokens == baseline else min(len(tokens), len(baseline))


def alpha_sample(case, variant, steps, stride):
    """Applied/natural alpha from a strided sample of steps (compare files are large)."""
    applied, natural = [], []
    for index in range(0, steps, max(1, stride)):
        steering = read(case / variant / f'step-{index:04d}' / 'compare.json').get('response', {}).get('steering')
        if steering:
            applied.append(steering['alpha_applied'])
            natural.append(steering['alpha_natural'])
    return applied, natural


def collect(root, stride):
    root = Path(root)
    multipliers = {}
    for directory in sorted(root.glob('m*')):
        if not directory.is_dir():
            continue
        for case in sorted(p for p in directory.iterdir() if p.is_dir()):
            variant, saved = state_of(case)
            if not saved:
                continue
            tokens = saved.get('generated', [])
            row = {'case': case.name, 'variant': variant, 'tokens': len(tokens),
                   'ended': bool(saved.get('ended')), 'stop_reason': saved.get('stop_reason'),
                   # Fixed window: a raw unique ratio falls with length, so it is not
                   # comparable across multipliers whose branches differ in length.
                   'unique_ratio': (round(len(set(tokens[:WINDOW]))/min(len(tokens), WINDOW), 3)
                                    if tokens else None),
                   'longest_repeat': longest_run(tokens), 'ids': tokens}
            if variant != 'original':
                applied, natural = alpha_sample(case, variant, len(tokens), stride)
                row['alpha_applied'] = round(st.median(applied), 3) if applied else None
                row['alpha_natural'] = round(st.median(natural), 3) if natural else None
            multipliers.setdefault(directory.name, []).append(row)
    return multipliers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/alpha-sweep'))
    parser.add_argument('--baseline', default='m0', help='Multiplier directory used as the unsteered reference')
    parser.add_argument('--stride', type=int, default=25, help='Sample every Nth step for alpha metrics')
    parser.add_argument('--json', action='store_true', help='Emit JSON instead of a table')
    args = parser.parse_args()
    multipliers = collect(args.root, args.stride)
    baseline = {row['case']: row['ids'] for row in multipliers.get(args.baseline, [])}

    summary = []
    for name, rows in sorted(multipliers.items(), key=lambda kv: float(kv[0][1:])):
        divergences, identical = [], 0
        for row in rows:
            reference = baseline.get(row['case'])
            if reference is None:
                continue
            index = first_divergence(row['ids'], reference)
            if index is None:
                identical += 1
            else:
                divergences.append(index)
        finished = [r for r in rows if r['ended']]
        reasons = {}
        for row in finished:
            key = row['stop_reason'] or 'unknown'
            reasons[key] = reasons.get(key, 0) + 1
        alphas = [r['alpha_applied'] for r in rows if r.get('alpha_applied') is not None]
        summary.append({
            'multiplier': float(name[1:]), 'branches': len(rows), 'complete': len(finished),
            'median_tokens': round(st.median([r['tokens'] for r in rows])) if rows else 0,
            'median_alpha_applied': round(st.median(alphas), 3) if alphas else None,
            'identical_to_baseline': identical,
            'median_first_divergence': round(st.median(divergences)) if divergences else None,
            'median_unique_ratio': round(st.median([r['unique_ratio'] for r in rows if r['unique_ratio']]), 3)
                                   if any(r['unique_ratio'] for r in rows) else None,
            'max_repeat_run': max((r['longest_repeat'] for r in rows), default=0),
            'stop_reasons': reasons})

    if args.json:
        print(json.dumps(summary, indent=2))
        return
    header = f"{'m':>5} {'done':>7} {'tok':>5} {'alpha':>6} {'same':>5} {'diverge':>8} {'uniq':>5} {'rep':>4}  stop reasons"
    print(header)
    print('-' * len(header))
    dash = lambda value: '-' if value is None else value
    for row in summary:
        reasons = ', '.join(f'{k}:{v}' for k, v in sorted(row['stop_reasons'].items())) or '-'
        print(f"{row['multiplier']:>5g} {row['complete']:>3}/{row['branches']:<3} {row['median_tokens']:>5} "
              f"{dash(row['median_alpha_applied']):>6} {row['identical_to_baseline']:>5} "
              f"{dash(row['median_first_divergence']):>8} {dash(row['median_unique_ratio']):>5} "
              f"{row['max_repeat_run']:>4}  {reasons}")
    print(f'\nsame = branches identical to the unsteered baseline; diverge = median first differing token index;'
          f'\nuniq = unique tokens in the first {WINDOW}, rep = longest identical-token run (degeneracy proxies).'
          '\nstop reasons: eos | degenerate_repeat (50 identical tokens) | token_limit |'
          '\n              backend_parse_error / backend_no_prediction (steering broke the output).')


if __name__ == '__main__':
    main()
