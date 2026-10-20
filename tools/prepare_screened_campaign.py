"""Prepare exact screened prompts in label/length priority; no submissions."""
import argparse
import hashlib
import json
from pathlib import Path

PRIORITY = {'Refusal': 0, 'Deflection': 1, 'Opposite position': 2}

def select(cases):
    selected = [c for c in cases if c.get('classification') in PRIORITY]
    if len({c['id'] for c in selected}) != len(selected):
        raise ValueError('Duplicate case identifiers')
    return sorted(selected, key=lambda c: (PRIORITY[c['classification']], c['tokens_predicted'], c['id']))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screening', required=True, type=Path)
    p.add_argument('--prior-plan', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    plan = json.loads(args.prior_plan.read_text())
    assert hashlib.sha256(plan['instruction'].encode()).hexdigest() == plan['instruction_sha256']
    cases = select(json.loads(args.screening.read_text())['cases'])
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for c in cases:
        path = args.output/'inputs'/c['id']/'inputs/config.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'prompt': c['prompt'], 'instruction': plan['instruction'] +
                        '\nQuestion (context only): ' + json.dumps(c['prompt'], ensure_ascii=False),
                        'editor': 'codex'}, ensure_ascii=False, indent=2))
        manifest.append({'id': c['id'], 'classification': c['classification'],
                         'tokens': c['tokens_predicted'], 'config': str(path.resolve()), 'phase':'pending'})
    (args.output/'selection.json').write_text(json.dumps({'cases':manifest,
        'max_tokens':plan['max_tokens'], 'instruction_sha256':plan['instruction_sha256'],
        'screening_sha256':hashlib.sha256(args.screening.read_bytes()).hexdigest()}, indent=2))
    print(json.dumps({'selected':len(cases),'counts':{k:sum(c['classification']==k for c in cases) for k in PRIORITY}}))

if __name__ == '__main__':
    main()
