import tempfile
import unittest
from pathlib import Path
from nla.setup import prepare_compiler_cache


class CompilerCacheTests(unittest.TestCase):
    def test_compiler_alias_does_not_invalidate_shared_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compiler = root / 'compiler'
            compiler.touch()
            alias = root / 'alias'
            alias.symlink_to(compiler)
            cache = root / 'CMakeCache.txt'
            cache.write_text('CMAKE_CXX_COMPILER:FILEPATH=' + str(compiler) + '\n')
            prepare_compiler_cache(root, str(alias))
            self.assertTrue(cache.exists())

    def test_different_compiler_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / 'CMakeCache.txt'
            cache.write_text('CMAKE_CXX_COMPILER:FILEPATH=/synthetic/old-compiler\n')
            prepare_compiler_cache(root, '/synthetic/new-compiler')
            self.assertFalse(cache.exists())
