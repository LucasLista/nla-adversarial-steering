import unittest
from nla import patches


class PatchTests(unittest.TestCase):
    def test_completion_patch_returns_before_chat_parsing(self):
        result = patches.patched_completion('prefix\n' + patches.COMPLETION + '\nsuffix')
        self.assertLess(result.index('return;'), result.index('state.update_chat_msg'))

    def test_unexpected_sources_fail_closed(self):
        for patch in (patches.patched_completion, patches.patched_capture, patches.patched_steering):
            with self.assertRaises(RuntimeError):
                patch('unexpected source')


if __name__ == '__main__':
    unittest.main()
