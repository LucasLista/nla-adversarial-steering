"""Write a cases file from a pinned public dataset, without printing any prompts.

  python -m nla.cases jbb --split harmful --category Malware/Hacking --output runs/jbb/cases.json
  uv run --with pyarrow python -m nla.cases deccp --output runs/deccp/cases.json
"""
import argparse
import csv
import io
import json
from pathlib import Path
import urllib.request

JBB = ('JailbreakBench/JBB-Behaviors', '886acc352a31533ffbcf4ef22c744658688086fc')
DECCP = ('augmxnt/deccp', '2c964c6957c5b1692992eb0d0409633155417056')
# Physical-harm behaviors are out of scope for this project.
JBB_HARMFUL_CATEGORIES = ('Harassment/Discrimination', 'Malware/Hacking', 'Economic harm', 'Fraud/Deception',
                          'Disinformation', 'Sexual/Adult content', 'Privacy', 'Expert advice',
                          'Government decision-making')


def fetch(dataset, path):
    url = f'https://huggingface.co/datasets/{dataset[0]}/resolve/{dataset[1]}/{path}'
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def jbb(split, categories):
    rows = csv.DictReader(io.StringIO(fetch(JBB, f'data/{split}-behaviors.csv').decode('utf-8-sig')))
    return [{'id': f"jbb-{split}-{int(row['Index']):03d}", 'category': row['Category'], 'prompt': row['Goal']}
            for row in rows if not categories or row['Category'] in categories]


def deccp(split):
    import pyarrow.parquet
    table = pyarrow.parquet.read_table(io.BytesIO(fetch(DECCP, f'data/{split}-00000-of-00001.parquet')))
    return [{'id': f'deccp-{split}-{i:03d}', 'prompt': text} for i, text in enumerate(table.column('text').to_pylist())]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', choices=('jbb', 'deccp'))
    p.add_argument('--split', help='jbb: benign|harmful (default benign); deccp: censored|uncensored (default censored)')
    p.add_argument('--category', action='append', default=[], help='jbb only; repeatable')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.source == 'jbb' and args.split == 'harmful' and (
            not args.category or any(c not in JBB_HARMFUL_CATEGORIES for c in args.category)):
        p.error('the harmful split needs explicit --category values from: ' + ', '.join(JBB_HARMFUL_CATEGORIES))
    if args.output.exists():
        p.error(f'{args.output} exists; never overwrite inputs under existing checkpoints')
    cases = (jbb(args.split or 'benign', args.category) if args.source == 'jbb'
             else deccp(args.split or 'censored'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n')
    print(f'wrote {len(cases)} cases to {args.output}')


if __name__ == '__main__':
    main()
