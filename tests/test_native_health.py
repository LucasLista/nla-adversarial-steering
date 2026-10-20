import unittest
from nla.fix_native_health import instrument, HELPER


class NativeHealthTests(unittest.TestCase):
    def test_health_schema_has_no_message_or_model_content(self):
        self.assertIn('"error_type", category', HELPER)
        self.assertIn('"frames", json::array', HELPER)
        for forbidden in ('e.what()', 'req.body', 'error.c_str()', 'activation'):
            self.assertNotIn(forbidden, HELPER)

    def test_unknown_pinned_source_fails_closed(self):
        for kind in ('context', 'http'):
            with self.assertRaises(RuntimeError):
                instrument('synthetic unexpected source', kind)
