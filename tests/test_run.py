import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from nla import run, trajectories


class RunTests(unittest.TestCase):
    def configs(self, *argv):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp)/'cases.json'
            cases.write_text(json.dumps([{'id': 'a', 'prompt': 'P1'}, {'id': 'b', 'prompt': 'P2', 'instruction': 'own'}]))
            with patch.object(sys, 'argv', ['run', '--cases', str(cases), '--output', tmp, *argv]), \
                    patch.object(run, 'wait_until_ready'), patch.object(run, 'steering_self_test') as test, \
                    patch.object(trajectories, 'generate', lambda config, directory, *rest: seen.append(
                        (str(Path(directory).relative_to(tmp)), config, rest[-1]))):
                run.main()
        return seen, test

    def test_config_keys_match_existing_checkpoints(self):
        seen, test = self.configs('--instruction', 'Edit for {prompt}', '--max-tokens', '64')
        test.assert_called_once()
        self.assertEqual(seen[0], ('a', {'prompt': 'P1', 'instruction': 'Edit for P1', 'max_tokens': 64,
                                         'editor_model': 'gpt-6.1-sol', 'variants': list(trajectories.VARIANTS),
                                         'policy': run.POLICY}, None))
        self.assertEqual(seen[1][1]['instruction'], 'own')

    def test_screening_runs_only_the_original_branch(self):
        seen, test = self.configs('--variants', 'original', '--shard', '1/2')
        test.assert_not_called()
        self.assertEqual([(d, c['variants'], v) for d, c, v in seen], [('b', ['original'], ['original'])])

    def test_multiplier_sweep(self):
        seen, _ = self.configs('--instruction', 'x', '--multipliers', '0', '2', '--fast')
        self.assertEqual([d for d, _, _ in seen], ['m0/a', 'm2/a', 'm0/b', 'm2/b'])
        self.assertEqual(seen[0][1]['variants'], ['original'])
        self.assertNotIn('steering_multiplier', seen[0][1])
        self.assertEqual(seen[1][1]['steering_multiplier'], 2)
        self.assertEqual(seen[1][1]['comparison_mode'], 'selected')


if __name__ == '__main__':
    unittest.main()
