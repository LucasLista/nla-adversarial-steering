import tempfile
import unittest
from pathlib import Path
from nla.setup import cuda_link_options


class CudaLinkTests(unittest.TestCase):
    def test_stub_is_link_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / 'runtime'
            runtime.mkdir()
            library = root / 'cuda'
            (library / 'stubs').mkdir(parents=True)
            (library / 'stubs/libcuda.so').touch()
            first = cuda_link_options(runtime, library)
            self.assertEqual(first, cuda_link_options(runtime, library))
            self.assertTrue((runtime / 'cuda-link-stubs/libcuda.so.1').is_symlink())
            self.assertNotIn(',-rpath,', first[0])
            self.assertIn(',-rpath-link,', first[0])

    def test_missing_toolkit_stub_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                cuda_link_options(Path(temporary), temporary)
