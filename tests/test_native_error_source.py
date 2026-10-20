import tempfile
import unittest
from pathlib import Path
from nla.native_error_source import source_frame, error_sites


class NativeErrorSourceTests(unittest.TestCase):
    def tearDown(self):
        error_sites.cache_clear()

    def test_unique_source_prefix_exposes_only_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'src').mkdir()
            (root / 'src/test.cpp').write_text('throw std::runtime_error("Synthetic source failure: " + value);')
            result = source_frame(root, 'Synthetic source failure: private synthetic value')
            self.assertEqual(result, {'file':'test.cpp','function':'native_runtime_error','line':1})
            self.assertNotIn('private', str(result))

    def test_ambiguous_or_unknown_error_has_no_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'src').mkdir()
            (root / 'src/test.cpp').write_text('throw std::runtime_error("Repeated synthetic failure");\nthrow std::runtime_error("Repeated synthetic failure");')
            self.assertIsNone(source_frame(root, 'Repeated synthetic failure'))
            self.assertIsNone(source_frame(root, 'Unknown synthetic failure'))
