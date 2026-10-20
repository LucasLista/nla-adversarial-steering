import unittest
from nla.fix_native_completion import ORIGINAL, REPLACEMENT, patched


class NativeCompletionPatchTests(unittest.TestCase):
    def test_only_native_response_bypasses_chat_parsing(self):
        source = 'prefix\n' + ORIGINAL + '\nsuffix'
        result = patched(source)
        self.assertEqual(result, 'prefix\n' + REPLACEMENT + '\nsuffix')
        self.assertLess(result.index('is_updated = true'), result.index('if (res_type'))
        self.assertLess(result.index('return;'), result.index('state.update_chat_msg'))

    def test_unknown_source_fails_closed(self):
        with self.assertRaises(RuntimeError):
            patched('unexpected synthetic source')
