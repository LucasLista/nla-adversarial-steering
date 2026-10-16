"""Summarize benchmark completion and optional external grades without reading text."""
import argparse
import json
from pathlib import Path

from nla.benchmark import manifest_configs


def report(manifest_path, results, grades=None):
    manifest, _ = manifest_configs(manifest_path)
    grades = grades or {}
    totals = {}
    cases = []
    for case in manifest['cases']:
        folder = Path(results)/case['id']
        progress_path = folder/'progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        item = {'id': case['id'], 'category': case['category'],
                'complete': bool(progress.get('complete')), 'branches': {}}
        for branch, status in progress.get('branches', {}).items():
            if not status.get('tokens') and not status.get('complete'):
                continue
            score = grades.get(case['id'], {}).get(branch)
            if score is not None and (type(score) not in (int, float) or not 0 <= score <= 1):
                raise ValueError('Grades must be numeric scores between zero and one')
            item['branches'][branch] = {'tokens': status['tokens'], 'complete': status['complete'], 'score': score}
            aggregate = totals.setdefault(branch, {'completed': 0, 'graded': 0, 'score_sum': 0})
            aggregate['completed'] += bool(status['complete'])
            if score is not None and status['complete']:
                aggregate['graded'] += 1
                aggregate['score_sum'] += score
        cases.append(item)
    for aggregate in totals.values():
        aggregate['mean_score'] = aggregate.pop('score_sum')/aggregate['graded'] if aggregate['graded'] else None
    return {'benchmark': manifest['benchmark'], 'revision': manifest['revision'],
            'split': manifest['split'], 'subset': manifest['subset'], 'count': manifest['count'],
            'completed_cases': sum(case['complete'] for case in cases), 'branches': totals,
            'grading': 'external scores; specify evaluator and version in your write-up', 'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--grades', type=Path, help='Scores only: {case_id: {branch: score}}')
    args = parser.parse_args()
    grades = json.loads(args.grades.read_text()) if args.grades else None
    print(json.dumps(report(args.manifest, args.results, grades), indent=2))


if __name__ == '__main__':
    main()
