import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from nla.trajectories import PrefixCache

class PrefixCacheTests(unittest.TestCase):
    def test_exact_requests_only_and_independent_copies(self):
        c = PrefixCache(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            req = {'action': 'inspect', 'tokens': [1], 'baseline': False}
            op = Mock(return_value={'h': [1.]})
            result = c.recorded(root/'a.json', req, op)
            result['h'][0] = 9
            self.assertEqual(c.recorded(root/'b.json', req, op), {'h': [1.]})
            self.assertEqual(op.call_count, 1)
            c.recorded(root/'c.json', dict(req, tokens=[2]), op)
            c.recorded(root/'d.json', dict(req, baseline=True), op)
            self.assertEqual(op.call_count, 3)
            c.recorded(root/'e.json', req, op)
            self.assertEqual(op.call_count, 4)
            self.assertEqual(len(c.entries), 2)

    def test_failed_validation_never_cached_and_editors_never_cached(self):
        c = PrefixCache()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            req = {'action': 'inspect', 'tokens': [1]}
            with self.assertRaises(ValueError):
                c.recorded(root/'a.json', req, lambda: {}, lambda r: 'h' in r)
            self.assertFalse(c.entries)
            op = Mock(return_value={'proposal':'synthetic'})
            for i in range(2):
                c.recorded(root/f'e{i}.json', {'action':'propose','text':'synthetic'}, op)
            self.assertEqual(op.call_count, 2)

    def test_resume_populates_cache_without_backend_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            req = {'action':'explain', 'tokens':[1], 'h':[1.]}
            PrefixCache().recorded(root/'a.json', req, lambda: {'text':'synthetic'})
            c = PrefixCache()
            fail = Mock(side_effect=AssertionError('backend called'))
            c.recorded(root/'a.json', req, fail)
            self.assertEqual(c.recorded(root/'b.json', req, fail), {'text':'synthetic'})
