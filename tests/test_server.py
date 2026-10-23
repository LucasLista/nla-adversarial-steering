import unittest
from unittest.mock import Mock, patch
from nla import server, stepper


def backend():
    b = server.Backend(build=None, port=18100, threads=1, gpu_layers=0, context=2048, resident=True)
    b.start, b.stop = Mock(), Mock()
    return b


class BackendTests(unittest.TestCase):
    def test_transient_failures_restart_the_model_and_replay_the_request(self):
        b = backend()
        good = {'completion_probabilities': [{'id': 7}]}
        invalid = {'completion_probabilities': [{'id': None}]}
        for first in (server.BackendError(500, 'Compute error.'), server.BackendError(503, 'loading'),
                      ConnectionRefusedError(), invalid):
            b.stop.reset_mock()
            with patch.object(b, 'request', side_effect=[first, good]) as request, patch('time.sleep'):
                self.assertIs(b('base', '/completion', {'n_probs': 30}), good)
            self.assertEqual(request.call_count, 2)
            b.stop.assert_called_once()

    def test_retries_are_bounded_and_deterministic_errors_are_not_retried(self):
        b = backend()
        with patch.object(b, 'request', side_effect=server.BackendError(500, 'Compute error.')) as request, \
                patch('time.sleep'), self.assertRaises(server.BackendError):
            b('base', '/completion', {})
        self.assertEqual(request.call_count, 3)
        with patch.object(b, 'request', side_effect=server.BackendError(500, 'Failed to parse input')) as request, \
                self.assertRaises(server.BackendError):
            b('base', '/completion', {})
        request.assert_called_once()

    def test_non_finite_activations_are_retried(self):
        b = backend()
        with patch.object(b, 'request', side_effect=[{'activation': [float('nan')]}, {'activation': [1.]}]), \
                patch('time.sleep'):
            self.assertEqual(b('base', '/extract', {}), {'activation': [1.]})


class AuthTests(unittest.TestCase):
    def request(self, token, path='/health', headers=None):
        handler = server.handler(Mock(), token)
        h = handler.__new__(handler)
        h.path, h.command, h.headers = path, 'GET', headers or {}
        h.send_response, h.send_header, h.end_headers = Mock(), Mock(), Mock()
        return h.authenticated(), h

    def test_token_is_required_and_can_be_set_as_a_cookie(self):
        token = 'ab' * 32
        self.assertTrue(self.request(None)[0])
        self.assertFalse(self.request(token)[0])
        self.assertTrue(self.request(token, headers={'Authorization': 'Bearer ' + token})[0])
        self.assertTrue(self.request(token, headers={'Cookie': 'nla_token=' + token})[0])
        ok, h = self.request(token, path='/?token=' + token)
        self.assertFalse(ok)
        h.send_response.assert_called_once_with(303)
        self.assertFalse(self.request(token, path='/?token=' + 'cd' * 32)[0])


if __name__ == '__main__':
    unittest.main()
