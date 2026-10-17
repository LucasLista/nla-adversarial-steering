import unittest
from unittest.mock import patch
from nla import launch, stepper

class NumericalRecoveryTests(unittest.TestCase):
    def test_retries_identical_request_after_reload(self):
        request = {'explanation': 'Synthetic benign test.'}
        with patch.object(launch, '_backend_call_once', side_effect=[
            {'activation': [None]}, {'activation': [1., 2.]}]) as call, patch.object(launch, 'stop') as stop:
            self.assertEqual(launch.backend_call('ar', '/reconstruct', request), {'activation': [1., 2.]})
            self.assertEqual(call.call_count, 2)
            for invocation in call.call_args_list:
                self.assertEqual(invocation.args, ('ar', '/reconstruct', request))
            stop.assert_called_once()

    def test_persistent_invalid_values_fail_without_substitution(self):
        with patch.object(launch, '_backend_call_once', return_value={'activation': [float('nan')]}) as call, patch.object(launch, 'stop') as stop:
            with self.assertRaises(stepper.NonFiniteActivation):
                launch.backend_call('ar', '/reconstruct', {})
            self.assertEqual(call.call_count, 3)
            self.assertEqual(stop.call_count, 2)

    def test_valid_output_has_no_reload(self):
        with patch.object(launch, '_backend_call_once', return_value={'activation': [1.]}) as call, patch.object(launch, 'stop') as stop:
            self.assertEqual(launch.backend_call('base', '/extract', {}), {'activation': [1.]})
            call.assert_called_once()
            stop.assert_not_called()
