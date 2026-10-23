import json
import unittest
from unittest.mock import Mock, patch
from nla import editor


class EditorTests(unittest.TestCase):
    def test_codex_failures_are_classified_from_error_events_only(self):
        for event in ({'type': 'error', 'message': 'Usage limit reached'},
                      {'type': 'turn.failed', 'error': {'message': 'Not logged in'}}):
            self.assertIs(editor.codex_failure(json.dumps(event)), editor.EditorUnavailable)
        self.assertIs(editor.codex_failure(json.dumps({'type': 'error', 'message': 'stream disconnected'})),
                      editor.EditorConnectionError)
        # Model output that happens to mention a limit is not an error envelope.
        message = json.dumps({'type': 'item.completed', 'item': {'text': 'usage limit'}})
        self.assertIs(editor.codex_failure(message), RuntimeError)
        self.assertIs(editor.codex_failure('usage limit'), RuntimeError)

    def test_connection_errors_are_retryable(self):
        self.assertTrue(issubclass(editor.EditorConnectionError, OSError))
        self.assertFalse(issubclass(editor.EditorUnavailable, OSError))

    def test_codex_requires_chatgpt_sign_in(self):
        with patch('shutil.which', return_value='/bin/codex'), \
                patch('subprocess.run', return_value=Mock(returncode=0, stdout='Logged in using an API key', stderr='')):
            with self.assertRaisesRegex(editor.EditorUnavailable, 'ChatGPT'):
                editor.codex('rewrite', 'rabbit')

    def test_propose_validates_and_reports_provenance(self):
        with patch.dict(editor.EDITORS, {'codex': lambda instruction, text, model: ' mouse '}):
            result = editor.propose('codex', 'rewrite', 'rabbit')
        self.assertEqual(result, {'proposal': 'mouse', 'editor_model': editor.MODEL,
                                  'editor_provider': 'codex', 'instruction': 'rewrite'})
        with self.assertRaises(ValueError):
            editor.propose('other', 'rewrite', 'rabbit')
        with self.assertRaises(ValueError):
            editor.propose('codex', ' ', 'rabbit')


if __name__ == '__main__':
    unittest.main()
