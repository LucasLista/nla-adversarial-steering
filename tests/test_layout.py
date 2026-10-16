"""Check public entry points and static assets without models or credentials."""
import http.client
from http.server import ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

import launch

ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def test_commands_import_and_show_help(self):
        commands = [[name + '.py'] for name in ('setup', 'launch', 'trajectories')]
        commands += [['-m', name] for name in (
            'nla.codex_editor', 'hpc.hpc_control', 'hpc.hpc_batch',
            'hpc.hf_server', 'hpc.hf_stage',
            'hpc.background_controller', 'hpc.model_preflight',
            'tools.run_status', 'tools.publish_run', 'tools.batch_viewer',
        )]
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, *command, '--help'], cwd=ROOT,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_relocated_stepper_is_served(self):
        with patch.object(launch, 'AUTH_TOKEN', None):
            server = ThreadingHTTPServer(('127.0.0.1', 0), launch.Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for path in ('/stepper', '/stepper.html'):
                    connection = http.client.HTTPConnection('127.0.0.1', server.server_port)
                    try:
                        connection.request('GET', path)
                        response = connection.getresponse()
                        self.assertEqual(response.status, 200)
                        self.assertEqual(response.getheader('Content-Type'), 'text/html; charset=utf-8')
                        self.assertEqual(response.read(), (ROOT / 'ui/stepper.html').read_bytes())
                    finally:
                        connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
