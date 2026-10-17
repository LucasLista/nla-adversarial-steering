"""Prepare pinned JailbreakBench cases without displaying their contents."""
import argparse
import csv
import hashlib
import io
import json
import random
from pathlib import Path
import urllib.request

from nla.trajectories import write_json

ROOT = Path(__file__).resolve().parents[1]
REVISION = '886acc352a31533ffbcf4ef22c744658688086fc'
DATASET = 'JailbreakBench/JBB-Behaviors'
# Physical-harm cases are outside this integration's evaluation scope.
CATEGORIES = ('Harassment/Discrimination', 'Malware/Hacking', 'Economic harm',
              'Fraud/Deception', 'Disinformation', 'Sexual/Adult content',
              'Privacy', 'Expert advice', 'Government decision-making')


def load_rows(data, split, categories=()):
    rows = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
    required = {'Index', 'Goal', 'Category'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Unexpected benchmark schema')
    if split == 'harmful' and not categories:
        raise ValueError('Select explicit evaluation categories for the harmful split')
    if any(category not in CATEGORIES for category in categories):
        raise ValueError('Category outside the supported evaluation scope')
    selected = []
    seen = set()
    for row in rows:
        index = int(row['Index'])
        if index < 0 or index in seen or not row['Goal'].strip():
            raise ValueError('Invalid or duplicate benchmark item')
        seen.add(index)
        if categories and row['Category'] not in categories:
            continue
        selected.append({'id': f'jbb-{split}-{index:03d}', 'category': row['Category'],
                         'prompt': row['Goal']})
    return sorted(selected, key=lambda row: row['id'])


def prepare(data, directory, split='benign', categories=(), limit=None, instruction=None,
            sample=None, seed=None):
    directory = Path(directory)
    rows = load_rows(data, split, categories)
    if limit is not None:
        if limit < 1:
            raise ValueError('Limit must be positive')
        rows = rows[:limit]
    if sample is not None:
        if not 1 <= sample <= len(rows):
            raise ValueError('Sample size must fit the selected cases')
        # Seeded draw so the selected case IDs are reproducible from the manifest.
        rows = sorted(random.Random(seed).sample(rows, sample), key=lambda row: row['id'])
    if not rows:
        raise ValueError('No benchmark cases selected')
    if instruction is not None and not instruction.strip():
        raise ValueError('Editing instruction is empty')
    manifest = {'benchmark': DATASET, 'revision': REVISION,
                'source_sha256': hashlib.sha256(data).hexdigest(), 'split': split,
                'categories': list(categories), 'limit': limit, 'count': len(rows),
                'sample': sample, 'seed': seed,
                'model': 'qwen2.5-7b', 'subset': len(rows) != 100,
                'editing_enabled': instruction is not None, 'cases': []}
    payloads = {}
    for row in rows:
        relative = Path(row['id']) / 'inputs/config.json'
        payloads[relative] = {'prompt': row['prompt'],
                             'instruction': instruction or 'Preserve the description unchanged.'}
        manifest['cases'].append({'id': row['id'], 'category': row['category'],
                                  'config': str(relative)})
    payloads[Path('manifest.json')] = manifest
    payloads[Path('catalog.json')] = [{'id': row['id'], 'prompt': row['prompt']} for row in rows]
    # Do not silently replace inputs underneath existing checkpoints.
    for relative, payload in payloads.items():
        path = directory / relative
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ValueError('Prepared benchmark differs; choose a new output directory')
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    for relative, payload in payloads.items():
        write_json(directory / relative, payload)
        (directory / relative).chmod(0o600)
    return manifest


def manifest_configs(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('benchmark') != DATASET or manifest.get('revision') != REVISION:
        raise ValueError('Unsupported benchmark manifest')
    if manifest.get('model') != 'qwen2.5-7b' or not manifest.get('cases'):
        raise ValueError('Invalid benchmark model or cases')
    paths = []
    for case in manifest['cases']:
        config = (path.parent / case['config']).resolve()
        if not config.is_relative_to(path.parent):
            raise ValueError('Benchmark config must remain inside its run directory')
        paths.append(config)
    return manifest, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('benign', 'harmful'), default='benign')
    parser.add_argument('--category', action='append', choices=CATEGORIES, default=[])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--limit', type=int, help='Take the first N cases')
    selection.add_argument('--sample', type=int, help='Draw N cases at random')
    parser.add_argument('--seed', type=int, default=0, help='Seed for --sample')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/benchmarks/jbb-benign')
    parser.add_argument('--instruction-file', type=Path, help='Private editor instruction; never printed')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'runs'):
        parser.error('Prepared inputs must stay in the ignored runs directory')
    if args.split == 'harmful' and not args.category:
        parser.error('The harmful split requires explicit --category selection')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    cache = ROOT/'.runtime/benchmarks'/REVISION/(args.split+'.csv')
    if not cache.exists():
        url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{args.split}-behaviors.csv'
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise ValueError('Unexpected benchmark size')
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
        cache.chmod(0o600)
    instruction = args.instruction_file.read_text() if args.instruction_file else None
    manifest = prepare(cache.read_bytes(), args.output, args.split, args.category, args.limit,
                       instruction, args.sample, args.seed)
    print(json.dumps({key: manifest[key] for key in
                      ('benchmark', 'revision', 'split', 'count', 'sample', 'seed', 'subset', 'editing_enabled')}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Dataset/parser errors must never echo input text.
        raise SystemExit('Benchmark preparation failed: ' + type(error).__name__) from None
