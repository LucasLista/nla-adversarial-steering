import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from nla import openai_editor, trajectories
from hpc import hpc_batch, hpc_control


class HPCRecoveryTests(unittest.TestCase):
    def test_controller_releases_allocation_before_waiting_on_bad_editor(self):
        args = ['controller', '--job', '123', '--ssh-host', 'synthetic-host',
                '--ssh-socket', '/synthetic/socket', '--remote-root', '/synthetic',
                '--output', '/synthetic/results', '--prompt', 'Synthetic greeting.',
                '--instruction', 'Preserve the greeting.', '--editor', 'openai']
        with patch('sys.argv', args), \
             patch.object(openai_editor, 'preflight', side_effect=openai_editor.EditorCreditExhausted()), \
             patch('subprocess.run') as run, patch('subprocess.check_output') as query:
            with self.assertRaises(openai_editor.EditorCreditExhausted):
                hpc_control.main()
        query.assert_not_called()
        self.assertIn('bkill 123', run.call_args.args[0][-1])

    def test_credit_exhaustion_is_terminal_and_has_no_payload(self):
        body = json.dumps({'error': {'code': 'credit_balance_exhausted',
                                      'message': 'PRIVATE_SENTINEL'}}).encode()
        error = urllib.error.HTTPError('https://api.openai.com', 429, '', {}, io.BytesIO(body))
        with patch.object(openai_editor, 'api_key', return_value='synthetic'), \
             patch('urllib.request.urlopen', side_effect=error) as call, \
             patch.object(openai_editor.time, 'sleep') as sleep:
            with self.assertRaises(openai_editor.EditorCreditExhausted) as raised:
                openai_editor.propose('Change blue to green.', 'A blue square.')
        self.assertNotIn('PRIVATE_SENTINEL', str(raised.exception))
        self.assertEqual(call.call_count, 1)
        sleep.assert_not_called()

    def test_terminal_editor_failure_returns_configuration_exit(self):
        for error in (openai_editor.EditorCreditExhausted,
                      openai_editor.EditorAccessDenied, openai_editor.EditorRefusal):
            with self.subTest(error=error), patch.object(hpc_batch, 'main', side_effect=error()), \
                 patch('sys.stdout', new=io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    hpc_batch.cli()
                self.assertEqual(raised.exception.code, 78)

    def test_temporary_backend_http_failure_can_be_retried(self):
        for api in (trajectories.API('http://localhost'), hpc_batch.AuthAPI('http://localhost', 'synthetic')):
            for status in (408, 502, 503, 504):
                with self.subTest(api=type(api).__name__, status=status):
                    error = urllib.error.HTTPError('http://localhost', status, '', {},
                                                   io.BytesIO(b'PRIVATE_SENTINEL'))
                    with patch('urllib.request.urlopen', side_effect=error):
                        with self.assertRaises(ConnectionError) as raised:
                            api('inspect', tokens=[1])
                    self.assertNotIn('PRIVATE_SENTINEL', str(raised.exception))

    def test_transient_editor_preflight_does_not_cancel_allocation(self):
        args = ['controller', '--job', '123', '--ssh-host', 'synthetic-host',
                '--ssh-socket', '/synthetic/socket', '--remote-root', '/synthetic',
                '--output', '/synthetic/results', '--prompt', 'Synthetic greeting.',
                '--instruction', 'Preserve the greeting.', '--editor', 'openai']
        with patch('sys.argv', args), \
             patch.object(openai_editor, 'preflight', side_effect=TimeoutError()), \
             patch('subprocess.run') as run:
            with self.assertRaises(TimeoutError):
                hpc_control.main()
        run.assert_not_called()
