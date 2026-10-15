import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hpc import hf_remote_download


class Response(io.BytesIO):
    def __init__(self, data, offset=0):
        super().__init__(data[offset:])
        self.status = 206 if offset else 200
        self.headers = {'Content-Range': f'bytes {offset}-{len(data)-1}/{len(data)}'}


class SnapshotTests(unittest.TestCase):
    def test_verified_resume_and_progress(self):
        payload = b'synthetic checkpoint bytes'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'ar').mkdir()
            (root / 'ar/head.safetensors.partial').write_bytes(payload[:4])
            plan = {'destination': directory, 'model': 'synthetic', 'roles': {}, 'proxy': 'http://localhost',
                    'files': [{'role': 'ar', 'file': 'head.safetensors', 'size': len(payload),
                               'sha256': hashlib.sha256(payload).hexdigest()}]}
            def fetch(request, **kwargs):
                self.assertEqual(request.headers['Range'], 'bytes=4-')
                return Response(payload, 4)
            with patch('sys.stdin', io.StringIO(json.dumps(plan))), patch('urllib.request.urlopen', side_effect=fetch):
                hf_remote_download.main()
            self.assertEqual((root / 'ar/head.safetensors').read_bytes(), payload)
            self.assertTrue(json.loads((root / 'download-progress.json').read_text())['complete'])
            with patch('sys.stdin', io.StringIO(json.dumps(plan))), patch('urllib.request.urlopen', side_effect=AssertionError('Redownloaded')):
                hf_remote_download.main()

    def test_bad_digest_never_marks_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {'destination': directory, 'model': 'synthetic', 'proxy': 'http://localhost',
                    'files': [{'role': 'base', 'file': 'model.safetensors', 'size': 4, 'sha256': '0'*64}]}
            with patch('sys.stdin', io.StringIO(json.dumps(plan))), patch('urllib.request.urlopen', return_value=Response(b'test')):
                with self.assertRaises(ValueError):
                    hf_remote_download.main()
            self.assertFalse((Path(directory) / 'download-progress.json').exists())


if __name__ == '__main__':
    unittest.main()
