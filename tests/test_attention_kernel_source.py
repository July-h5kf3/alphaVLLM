from pathlib import Path
import unittest


class StoreKVCacheKernelSourceTest(unittest.TestCase):
    def test_store_kvcache_kernel_uses_block_pointers(self):
        source = Path("nanovllm/layers/Attention.py").read_text()

        self.assertGreaterEqual(source.count("tl.make_block_ptr"), 4)
        self.assertIn("key_block_ptr", source)
        self.assertIn("value_block_ptr", source)
        self.assertIn("k_cache_block_ptr", source)
        self.assertIn("v_cache_block_ptr", source)


if __name__ == "__main__":
    unittest.main()
