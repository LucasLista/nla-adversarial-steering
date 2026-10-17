import math
import tempfile
import unittest
from unittest.mock import Mock

from nla import trajectories as t
from nla.stepper import run

H = [3., 4.]
R0 = [1., 1.]
R1 = [2., 5.]
DELTA = [R1[0]-R0[0], R1[1]-R0[1]]


def backend(role, endpoint, request):
    if endpoint == '/reconstruct':
        return {'activation': list(R1)}
    if endpoint == '/extract':
        return {'activation': list(H)}
    if endpoint == '/completion':
        return {'steering': request.get('steering')}
    raise AssertionError(endpoint)


def target(result):
    """Reconstruct the patched activation the backend was asked to apply."""
    entry = result['edited_corrected']['steering'][0]
    direction_norm = math.sqrt(sum(x*x for x in entry['direction']))
    if direction_norm < 1e-12:
        return list(H)  # the backend skips zero-length directions
    scale = entry['alpha'] * math.sqrt(sum(x*x for x in H)) / direction_norm
    return [h + scale*d for h, d in zip(H, entry['direction'])]


class SteeringMultiplierTests(unittest.TestCase):
    def compare(self, **extra):
        body = {'action': 'compare', 'tokens': [1, 2], 'reconstruction': list(R0),
                'edited': 'Synthetic description.', 'variants': ['edited_corrected'], **extra}
        return run(body, backend, '')

    def test_default_matches_parameter_free_method(self):
        """Without a multiplier the patch stays h + (AR(edited) - AR(original))."""
        for got, want in zip(target(self.compare()), [h+d for h, d in zip(H, DELTA)]):
            self.assertAlmostEqual(got, want)
        for got, want in zip(target(self.compare(multiplier=1)), [h+d for h, d in zip(H, DELTA)]):
            self.assertAlmostEqual(got, want)

    def test_multiplier_scales_the_edit_direction(self):
        for multiplier in (0.25, 2.0, 6.0):
            expected = [h + multiplier*d for h, d in zip(H, DELTA)]
            for got, want in zip(target(self.compare(multiplier=multiplier)), expected):
                self.assertAlmostEqual(got, want)

    def test_zero_multiplier_leaves_the_activation_unchanged(self):
        for got, want in zip(target(self.compare(multiplier=0)), H):
            self.assertAlmostEqual(got, want)

    def test_reported_alpha_is_relative_to_the_activation_norm(self):
        natural = math.sqrt(sum(d*d for d in DELTA))/math.sqrt(sum(x*x for x in H))
        for multiplier in (0, 1, 3.0):
            steering = self.compare(multiplier=multiplier)['steering']
            self.assertAlmostEqual(steering['alpha_natural'], natural)
            self.assertAlmostEqual(steering['alpha_applied'], multiplier*natural)
            self.assertEqual(steering['multiplier'], multiplier)

    def test_invalid_multipliers_are_rejected(self):
        for value in (-1, float('nan'), float('inf'), '1', True, None):
            with self.assertRaises(ValueError):
                self.compare(multiplier=value)



class SweepWiringTests(unittest.TestCase):
    """The configured multiplier must reach the backend's compare call."""

    def generate(self, config):
        seen = []
        def prediction(token):
            return {'completion_probabilities': [{'id': token, 'top_logprobs': [{'id': token, 'logprob': 0}]}]}
        def api(action, **kwargs):
            seen.append((action, kwargs))
            if action == 'start': return {'tokens': [1]}
            if action == 'inspect': return {'h': [1.0], 'baseline': prediction(2)}
            if action == 'explain': return {'text': 'o', 'reconstruction': [.9], 'error': [.1], 'metrics': {}}
            if action == 'compare': return {k: prediction(5) for k in t.VARIANTS}
            if action == 'decode': return {'content': ''}
            raise AssertionError(action)
        with tempfile.TemporaryDirectory() as tmp:
            t.generate(config, tmp, api, Mock(return_value={'proposal': 'e'}),
                       variants=['edited_corrected'])
        return [kwargs for action, kwargs in seen if action == 'compare']

    def test_multiplier_is_forwarded_when_configured(self):
        base = {'prompt': 'p', 'instruction': 'i', 'max_tokens': 1}
        self.assertEqual(self.generate(dict(base, steering_multiplier=2.5))[0]['multiplier'], 2.5)

    def test_absent_multiplier_keeps_requests_unchanged(self):
        base = {'prompt': 'p', 'instruction': 'i', 'max_tokens': 1}
        self.assertNotIn('multiplier', self.generate(base)[0])

if __name__ == '__main__':
    unittest.main()
