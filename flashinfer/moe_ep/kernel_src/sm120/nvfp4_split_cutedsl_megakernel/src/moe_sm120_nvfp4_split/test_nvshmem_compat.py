"""Compatibility guard tests; run inside the supported CuTeDSL image."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

try:
    from .transport.nvshmem_compat import prepare_nvshmem_hybrid_compat
except ImportError:
    from transport.nvshmem_compat import prepare_nvshmem_hybrid_compat


class NvshmemCompatTest(unittest.TestCase):
    def test_installed_artifacts_are_guarded_and_compilable(self):
        artifacts = prepare_nvshmem_hybrid_compat()
        self.assertEqual(artifacts.version, (3, 7, 0))
        self.assertTrue(Path(artifacts.helper_bitcode).is_file())
        self.assertIn(artifacts.device_bitcode, artifacts.link_libraries)
        self.assertIn(artifacts.helper_bitcode, artifacts.link_libraries)

    def test_unknown_bitcode_hash_fails_closed(self):
        with tempfile.NamedTemporaryFile() as unknown:
            unknown.write(b"not an NVSHMEM device library")
            unknown.flush()
            with self.assertRaisesRegex(RuntimeError, "refuses unvalidated"):
                prepare_nvshmem_hybrid_compat(unknown.name)

    def test_helper_source_has_no_raw_state_offset(self):
        source = (
            Path(__file__).parent / "transport" / "nvshmem_compat.cu"
        ).read_text()
        self.assertNotIn("nvshmemi_device_state_d+56", source)
        self.assertIn("offsetof", source)
        self.assertIn("peer_heap_base_p2p", source)


if __name__ == "__main__":
    unittest.main()
