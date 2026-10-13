import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import trajectories as t

def prediction(token):
    return {'completion_probabilities': [{'id': token, 'top_logprobs': [{'id': token, 'logprob': 0}]}]}

class TrajectoryTests(unittest.TestCase):
    def test_independent_prefixes_checkpoint_and_resume(self):
        seen = []
        def api(action, **kwargs):
            seen.append((action, kwargs))
            if action == 'start': return {'tokens': [1]}
            if action == 'inspect': return {'h': [1.0], 'baseline': prediction(2)}
            if action == 'explain': return {'text': 'original', 'reconstruction': [.9], 'error': [.1], 'metrics': {}}
            if action == 'compare': return {k: prediction(i + 2) for i, k in enumerate(t.VARIANTS)}
            if action == 'decode': return {'content': str(kwargs['tokens'])}
            raise AssertionError(action)
        editor = Mock(return_value={'proposal': 'edited', 'editor_provider': 'codex-chatgpt'})
        config = {'prompt': 'test', 'instruction': 'edit', 'max_tokens': 2}
        with tempfile.TemporaryDirectory() as tmp:
            result = t.generate(config, tmp, api, editor)
            self.assertEqual([x['generated_token_ids'] for x in result.values()], [[2,2],[3,3],[4,4],[5,5]])
            for token in range(2,6):
                self.assertIn(('inspect', {'tokens': [1,token]}), seen)
            self.assertEqual(editor.call_count, 4)
            self.assertEqual(t.generate(config, tmp, Mock(side_effect=AssertionError('Recomputed')), editor), result)
            self.assertEqual(editor.call_count, 4)
            saved = json.loads((Path(tmp)/'edited_raw/step-0000/explain.json').read_text())
            self.assertEqual(saved['response']['error'], [.1])
            self.assertIn('duration_seconds', saved)

    def test_failure_preserves_successful_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'stage.json'
            first = Mock(return_value={'activation': [1,2]})
            t.recorded(path, {'tokens': [1]}, first)
            self.assertEqual(t.recorded(path, {'tokens': [1]}, Mock(side_effect=RuntimeError())), {'activation': [1,2]})
            with self.assertRaises(ValueError): t.recorded(path, {'tokens': [2]}, first)
            with self.assertRaises(RuntimeError): t.recorded(Path(tmp)/'fail.json', {}, Mock(side_effect=RuntimeError('offline')))
            self.assertEqual(len(list(Path(tmp).glob('fail-error-*.json'))), 1)

    def test_eos_stops_branch(self):
        def api(action, **kwargs):
            if action == 'start': return {'tokens': [1]}
            if action == 'inspect': return {'h': [1.], 'baseline': prediction(151645)}
            if action == 'explain': return {'text': 't', 'reconstruction': [1.]}
            if action == 'compare': return {k: prediction(151645) for k in t.VARIANTS}
            if action == 'decode': return {'content': ''}
        with tempfile.TemporaryDirectory() as tmp:
            outputs=t.generate({'prompt':'p','instruction':'i','max_tokens':2},tmp,api,lambda *_:{'proposal':'e'})
            self.assertTrue(all(v['stop_reason']=='eos' and len(v['generated_token_ids'])==1 for v in outputs.values()))

if __name__ == '__main__': unittest.main()
