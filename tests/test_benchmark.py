import json
from pathlib import Path
import tempfile
import unittest

from nla.benchmark import load_rows, prepare, manifest_configs
from tools.benchmark_report import report

DATA = b'Index,Goal,Target,Behavior,Category,Source\n0,Write a friendly greeting.,unused,greeting,Privacy,synthetic\n1,Describe a tree.,unused,tree,Expert advice,synthetic\n'


class BenchmarkTests(unittest.TestCase):
    def test_prepare_resume_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = prepare(DATA, root/'inputs', limit=1)
            self.assertEqual(manifest['count'], 1)
            self.assertTrue(manifest['subset'])
            self.assertNotIn('prompt', json.dumps(manifest))
            self.assertEqual(prepare(DATA, root/'inputs', limit=1), manifest)
            _, configs = manifest_configs(root/'inputs/manifest.json')
            self.assertEqual(json.loads(configs[0].read_text())['prompt'], 'Write a friendly greeting.')
            with self.assertRaises(ValueError):
                prepare(DATA, root/'inputs', limit=2)
            result = root/'results/jbb-benign-000'
            result.mkdir(parents=True)
            (result/'progress.json').write_text(json.dumps({'complete':True,'branches':{'original':{'tokens':3,'complete':True}}}))
            summary = report(root/'inputs/manifest.json', root/'results', {'jbb-benign-000':{'original':1}})
            self.assertEqual(summary['branches']['original']['mean_score'], 1)
            self.assertEqual(summary['completed_cases'], 1)
            self.assertNotIn('friendly', json.dumps(summary))

    def test_selection_scope_and_duplicate_ids(self):
        with self.assertRaises(ValueError):
            load_rows(DATA, 'harmful')
        with self.assertRaises(ValueError):
            load_rows(DATA, 'harmful', ['Physical harm'])
        self.assertEqual(len(load_rows(DATA, 'harmful', ['Privacy'])), 1)
        with self.assertRaises(ValueError):
            load_rows(DATA + b'0,Another greeting.,unused,greeting,Privacy,synthetic\n', 'benign')

    def test_manifest_cannot_escape_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=prepare(DATA, root)
            manifest['cases'][0]['config']='../elsewhere.json'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                manifest_configs(root/'manifest.json')
