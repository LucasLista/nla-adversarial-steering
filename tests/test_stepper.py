import math
import unittest
from unittest.mock import patch
from nla import stepper
from nla.stepper import run


def backend(calls):
    def call(role, endpoint, request):
        calls.append((role, endpoint))
        if endpoint == '/reconstruct': return {'activation': [2., 3.]}
        if endpoint == '/extract': return {'activation': [1., 2.]}
        if endpoint == '/completion': return {'steering': request.get('steering')}
        raise AssertionError(endpoint)
    return call


class StepperTests(unittest.TestCase):
    def test_prediction_token_id(self):
        self.assertEqual(stepper.prediction_token_id({'completion_probabilities': [{'id': 7}]}), 7)
        # A one-token UTF-8 fragment carries its ID without probabilities.
        self.assertEqual(stepper.prediction_token_id({'tokens': [9], 'tokens_predicted': 1,
                                                      'completion_probabilities': []}), 9)
        for bad in ({}, {'completion_probabilities': [{'id': 'x'}]},
                    {'tokens': [8], 'tokens_predicted': 1, 'completion_probabilities': [{'id': 7}]}):
            with self.assertRaises(stepper.InvalidPrediction):
                stepper.prediction_token_id(bad)

    def test_intervention_applies_exact_delta(self):
        h, target = [3., 4.], [4., 6.]
        entry = stepper.intervention(h, target, 5)
        scale = entry['alpha'] * stepper.norm(h) / stepper.norm(entry['direction'])
        self.assertEqual(entry['token_pos'], 5)
        for got, want in zip([a + scale*d for a, d in zip(h, entry['direction'])], target):
            self.assertAlmostEqual(got, want)

    def test_invalid_activations_are_rejected(self):
        for bad in ([], [1, float('nan')], [1, '2'], None):
            with self.assertRaises(stepper.InvalidActivation):
                stepper.validate_activation(bad)
        with self.assertRaises(stepper.InvalidActivation):
            stepper.validate_activation([1., 2.], 3)

    def test_selected_variant_matches_full_comparison(self):
        body = {'action': 'compare', 'tokens': [1, 2], 'reconstruction': [.5, 1.], 'edited': 'text'}
        full = run(body, backend([]))
        for variant in stepper.VARIANTS:
            calls = []
            self.assertEqual(run(dict(body, variants=[variant]), backend(calls))[variant], full[variant])
            self.assertEqual(calls.count(('base', '/completion')), 1)
            self.assertEqual(calls.count(('ar', '/reconstruct')), int(variant.startswith('edited')))

    def test_inspect_can_skip_baseline(self):
        self.assertEqual(run({'action': 'inspect', 'tokens': [1], 'baseline': False}, backend([])), {'h': [1., 2.]})

    def test_context_limit(self):
        with self.assertRaises(ValueError):
            run({'action': 'inspect', 'tokens': [1] * stepper.CONTEXT}, backend([]))

    def test_propose_routes_to_editor(self):
        with patch('nla.editor.propose', return_value={'proposal': 'mouse'}) as propose:
            run({'action': 'propose', 'editor': 'openai', 'instruction': 'rewrite', 'text': 'rabbit'}, None)
        propose.assert_called_once_with('openai', 'rewrite', 'rabbit')


if __name__ == '__main__':
    unittest.main()
