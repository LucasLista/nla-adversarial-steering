import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import publish_run


class PublishRunTests(unittest.TestCase):
    def test_export_preserves_stop_reasons_without_private_checkpoint_fields(self):
        for ended, reason, count, expected in (
            (True, 'degenerate_repeat', 3, 'degenerate_repeat'),
            (True, 'backend_no_prediction', 3, 'backend_no_prediction'),
            (True, 'backend_parse_error', 3, 'backend_parse_error'),
            (True, 'eos', 3, 'eos'),
            (True, None, 3, 'eos'),  # Legacy checkpoints lack a reason.
            (False, None, 4, 'token_limit'),
            (False, None, 3, 'in_progress'),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                branch = root / 'run' / 'original'
                branch.mkdir(parents=True)
                state = {'generated': [1] * count, 'ended': ended,
                         'response': 'Synthetic response.', 'private': 'DO_NOT_EXPORT'}
                if reason is not None:
                    state['stop_reason'] = reason
                (branch / 'state.json').write_text(json.dumps(state))
                with patch.object(publish_run, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                    publish_run.publish(root / 'run', 'snapshot', limit=4)
                data = (root / 'logs/snapshot/responses.json').read_bytes()
                exported = json.loads(data)['original']
                self.assertEqual(exported['stop_reason'], expected)
                self.assertNotIn(b'DO_NOT_EXPORT', data)
                metadata = json.loads((root / 'logs/snapshot/status.json').read_text())
                self.assertEqual(metadata['responses_sha256'], hashlib.sha256(data).hexdigest())
