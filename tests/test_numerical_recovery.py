import unittest
from unittest.mock import patch
from nla import launch, stepper

class NumericalRecoveryTests(unittest.TestCase):
    def test_startup_failure_reloads_identical_request_and_is_bounded(self):
        request = {'prompt':[1,2], 'n_probs':30}
        result = {'completion_probabilities':[{'id':7}]}
        with patch.object(launch,'_backend_call_once',side_effect=[launch.BackendStartupFailure(),result]) as call, patch.object(launch,'stop') as stop, patch.object(launch.time,'sleep'):
            self.assertIs(launch.backend_call('base','/completion',request),result)
            self.assertTrue(all(c.args==('base','/completion',request) for c in call.call_args_list))
            stop.assert_called_once()
        with patch.object(launch,'_backend_call_once',side_effect=launch.BackendStartupFailure()) as call, patch.object(launch,'stop') as stop, patch.object(launch.time,'sleep'):
            with self.assertRaises(launch.BackendStartupFailure):
                launch.backend_call('base','/completion',request)
            self.assertEqual(call.call_count,3)
            self.assertEqual(stop.call_count,2)
    def test_native_json_diagnostic_retains_only_exception_identifier(self):
        error = launch.backend_http_error(500, {'error': {'message': '[json.exception.type_error.316] synthetic private value'}})
        self.assertEqual(error.__name__, 'BackendJSONTypeError316')
        self.assertEqual(str(error()), '')
        for message in ('prefix [json.exception.type_error.316]', '[json.exception.unknown.316]', '[json.exception.type_error.999]'):
            self.assertIs(launch.backend_http_error(500, {'error': {'message': message}}), launch.BACKEND_HTTP_ERRORS[500])
        self.assertIs(launch.backend_http_error(400, {'error': {'message': '[json.exception.type_error.316] synthetic'}}), launch.BACKEND_HTTP_ERRORS[400])

    def test_native_decode_failure_classification_does_not_echo_body(self):
        self.assertIs(launch.backend_http_error(500,{'error':{'message':'Invalid input batch.'}}),launch.BackendInvalidInputBatch)
        self.assertIs(launch.backend_http_error(500,{'error':{'message':'Compute error.'}}),launch.BackendComputeError)
        self.assertIs(launch.backend_http_error(500,{'error':{'message':'Context size has been exceeded.'}}),launch.BackendContextExceeded)
        self.assertIs(launch.backend_http_error(500,{'error':{'message':'Unrecognized synthetic error'}}),launch.BACKEND_HTTP_ERRORS[500])

    def test_native_invalid_batch_reloads_without_changing_request(self):
        request={'prompt':[1,2], 'n_probs':30}
        response={'completion_probabilities':[{'id':7}]}
        with patch.object(launch,'_backend_call_once',side_effect=[launch.BackendInvalidInputBatch(),response]) as call, patch.object(launch,'stop') as stop, patch.object(launch.time,'sleep'):
            self.assertIs(launch.backend_call('base','/completion',request),response)
            self.assertTrue(all(c.args==('base','/completion',request) for c in call.call_args_list))
            stop.assert_called_once()

    def test_persistent_native_compute_error_is_bounded(self):
        with patch.object(launch,'_backend_call_once',side_effect=launch.BackendComputeError()) as call, patch.object(launch,'stop') as stop, patch.object(launch.time,'sleep'):
            with self.assertRaises(launch.BackendComputeError):launch.backend_call('base','/completion',{'n_probs':30})
            self.assertEqual(call.call_count,3)
            self.assertEqual(stop.call_count,2)

    def test_temporary_backend_http_failure_replays_identical_request(self):
        request = {'n_probs':30}
        response = {'completion_probabilities':[{'id':7}]}
        with patch.object(launch,'_backend_call_once',side_effect=[launch.BACKEND_HTTP_ERRORS[503](),response]) as call, patch.object(launch,'stop') as stop, patch.object(launch.time,'sleep'):
            self.assertIs(launch.backend_call('base','/completion',request),response)
            self.assertTrue(all(c.args==('base','/completion',request) for c in call.call_args_list))
            stop.assert_called_once()

    def test_backend_http_internal_failure_does_not_retry(self):
        with patch.object(launch,'_backend_call_once',side_effect=launch.BACKEND_HTTP_ERRORS[500]()) as call:
            with self.assertRaises(launch.BACKEND_HTTP_ERRORS[500]):
                launch.backend_call('base','/completion',{'n_probs':30})
            call.assert_called_once()

    def test_native_utf8_fragment_uses_returned_id_without_reload(self):
        response = {'tokens': [7], 'tokens_predicted': 1, 'content': ''}
        with patch.object(launch, '_backend_call_once', return_value=response) as call, patch.object(launch, 'stop') as stop:
            self.assertIs(launch.backend_call('base', '/completion', {'n_probs':30}), response)
            self.assertEqual(stepper.prediction_token_id(response), 7)
            call.assert_called_once()
            stop.assert_not_called()
        self.assertNotIn('completion_probabilities', response)

    def test_native_token_requires_single_explicit_id(self):
        for response in ({'tokens': [], 'tokens_predicted': 1},
                         {'tokens': [7, 8], 'tokens_predicted': 2},
                         {'tokens': [True], 'tokens_predicted': 1},
                         {'tokens': [7]},
                         {'tokens': [7], 'tokens_predicted': 1,
                          'completion_probabilities': [{'id':8}]}):
            with self.subTest(response=response), self.assertRaises(stepper.InvalidPredictionResponse):
                stepper.validate_prediction(response)

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

    def test_invalid_prediction_replays_identical_request_after_reload(self):
        request={'prompt':[1,2], 'n_predict':1, 'n_probs':30}
        valid={'completion_probabilities':[{'id':7}]}
        with patch.object(launch, '_backend_call_once', side_effect=[{}, valid]) as call, patch.object(launch, 'stop') as stop:
            self.assertEqual(launch.backend_call('base', '/completion', request), valid)
            self.assertEqual(call.call_count, 2)
            for invocation in call.call_args_list:
                self.assertEqual(invocation.args, ('base', '/completion', request))
            stop.assert_called_once()

    def test_persistent_invalid_prediction_fails_without_substitution(self):
        with patch.object(launch, '_backend_call_once', return_value={}) as call, patch.object(launch, 'stop') as stop:
            with self.assertRaises(stepper.InvalidPredictionResponse):
                launch.backend_call('base', '/completion', {'n_probs':30})
            self.assertEqual(call.call_count, 3)
            self.assertEqual(stop.call_count, 2)
