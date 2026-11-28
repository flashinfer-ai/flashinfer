# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU dependency tests; runnable directly without importing Torch/FlashInfer."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SOURCE = Path(__file__).resolve().parents[3] / "flashinfer/jit/rdma_moe_alltoall.py"
_SPEC = importlib.util.spec_from_file_location("rdma_moe_jit_under_test", _SOURCE)
_JIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_JIT)


class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.prefix = Path(self.directory.name)
        self.include = self.prefix / "include"
        self.library = self.prefix / "lib"
        self.include.mkdir()
        self.library.mkdir()
        self.header = self.include / "nccl.h"
        self.header.write_text(
            "#define NCCL_MAJOR 2\n#define NCCL_MINOR 30\n#define NCCL_PATCH 7\n"
        )
        (self.include / "nccl_device.h").write_text("// device API\n")
        (self.library / "libnccl.so.2").write_bytes(b"fixture-runtime-v1")

    def test_prefix_discovery_and_exact_version_rejection(self):
        with mock.patch.dict(
            os.environ,
            {
                "NCCL_HOME": str(self.prefix),
                "NCCL_INCLUDE_DIR": str(self.include),
                "NCCL_LIB_DIR": str(self.library),
            },
        ):
            self.assertEqual(_JIT.resolve_nccl_paths(), (self.include, self.library))
            self.header.write_text(
                "#define NCCL_MAJOR 2\n#define NCCL_MINOR 30\n#define NCCL_PATCH 8\n"
            )
            with self.assertRaisesRegex(RuntimeError, "NCCL 2.30.7"):
                _JIT.resolve_nccl_paths()

    def test_missing_device_headers_rejected(self):
        (self.include / "nccl_device.h").unlink()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "NCCL_HOME": str(self.prefix),
                    "NCCL_INCLUDE_DIR": str(self.include),
                    "NCCL_LIB_DIR": str(self.library),
                },
            ),
            self.assertRaisesRegex(RuntimeError, "host/device headers"),
        ):
            _JIT.resolve_nccl_paths()

    def test_content_identity_ignores_preserved_timestamps(self):
        initial = _JIT._nccl_identity(self.include, self.library)
        stat = self.header.stat()
        self.header.write_text(self.header.read_text() + "// changed device contract\n")
        os.utime(self.header, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        changed_header = _JIT._nccl_identity(self.include, self.library)
        self.assertNotEqual(initial, changed_header)
        (self.library / "libnccl.so.2").write_bytes(b"fixture-runtime-v2")
        self.assertNotEqual(
            changed_header, _JIT._nccl_identity(self.include, self.library)
        )

    def test_linked_device_headers_and_directory_cycle(self):
        device = self.prefix / "device_headers"
        device.mkdir()
        header = device / "gin.cuh"
        header.write_text("// first contract\n")
        (self.include / "nccl_device").symlink_to(device, target_is_directory=True)
        (device / "parent").symlink_to(self.include, target_is_directory=True)
        initial = _JIT._nccl_identity(self.include, self.library)
        header.write_text("// second contract\n")
        self.assertNotEqual(initial, _JIT._nccl_identity(self.include, self.library))


if __name__ == "__main__":
    unittest.main()
