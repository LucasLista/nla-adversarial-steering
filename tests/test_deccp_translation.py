import json
from pathlib import Path
import tempfile
import unittest

from tools.export_deccp_viewer import render
from tools.translate_deccp_review import (
    CJK, NUMBER, source_digest, texts, translate_segment, translate_text,
)


class TranslationTests(unittest.TestCase):
    def test_preserves_surrounding_english_and_fragment_punctuation(self):
        source = 'Final token "事实)" follows "信息\\n".'
        overrides = {'texts': {}, 'segments': {
            '事实': 'facts', '信息\\n': 'information\\n',
        }}
        self.assertEqual(
            translate_text(source, {}, overrides),
            'Final token "facts)" follows "information\\n".',
        )

    def test_preserves_list_numbers_and_exact_addresses(self):
        self.assertEqual(
            translate_segment('5. 使用', {'5. 使用': 'Use'}, {}),
            '5. Use',
        )
        self.assertEqual(
            translate_segment('地址8.8.8.8', {'地址8.8.8.8': 'Address 2.8.8.8'}, {}),
            'Address 8.8.8.8',
        )

    def test_snapshot_coverage_and_literal_numbers(self):
        directory = Path(__file__).resolve().parents[1] / 'research/2026-10-19/deccp'
        data = json.loads((directory / 'review.json').read_text())
        translated = json.loads((directory / 'review.en.json').read_text())
        sources = texts(data)
        self.assertEqual(set(translated['texts']), set(sources))
        self.assertEqual(translated['source_digest'], source_digest(sources))
        for source, english in translated['texts'].items():
            self.assertTrue(english.strip())
            self.assertIsNone(CJK.search(english))
            self.assertEqual(NUMBER.findall(source), NUMBER.findall(english))

    def test_rejects_stale_translation_before_overwriting_viewer(self):
        data = {'cases': [{'prompt': '中国', 'instruction': '', 'branches': {}}]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'review.html'
            output.write_text('previous viewer')
            with self.assertRaisesRegex(ValueError, 'refreshing'):
                render(data, output, {'source_digest': 'old', 'texts': {}})
            self.assertEqual(output.read_text(), 'previous viewer')


if __name__ == '__main__':
    unittest.main()
