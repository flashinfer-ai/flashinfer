# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the Blackwell next/ MegaMoE source snapshot.

Run directly with Python to avoid loading the GPU test suite's conftest.
Set FLASHINFER_TEST_WHEEL to also check the built wheel's source payload.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import zipfile


_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_PATH = "flashinfer/moe_ep/kernel_src/sm100/next_cutedsl_megamoe"
_PACKAGE = _ROOT / _PACKAGE_PATH


class BlackwellNextMegaMoePackagingTests(unittest.TestCase):
    def test_sources_match_export_manifest(self):
        manifest = json.loads((_PACKAGE / "export_manifest.json").read_text())
        source = _PACKAGE / "src"
        files = {
            path.relative_to(source).as_posix(): path
            for path in source.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        self.assertEqual(set(files), set(manifest["files"]))
        for name, digest in manifest["files"].items():
            with self.subTest(file=name):
                self.assertEqual(
                    hashlib.sha256(files[name].read_bytes()).hexdigest(), digest
                )

    def test_package_import_does_not_load_runtime_dependencies(self):
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                """
import importlib.abc
import importlib.util
import sys

class RejectRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'cutlass', 'cuda', 'nvshmem', 'tvm_ffi', 'sources'
        } or '.src' in fullname:
            raise AssertionError(f'Unexpected runtime import: {fullname}')

sys.meta_path.insert(0, RejectRuntimeImports())
original_path = list(sys.path)
spec = importlib.util.spec_from_file_location('_blackwell_snapshot', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert sys.path == original_path, 'Package import modified sys.path'
""",
                str(_PACKAGE / "__init__.py"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(os.environ.get("FLASHINFER_TEST_WHEEL"), "No wheel supplied")
    def test_wheel_contains_export_and_provenance(self):
        manifest = json.loads((_PACKAGE / "export_manifest.json").read_text())
        with zipfile.ZipFile(os.environ["FLASHINFER_TEST_WHEEL"]) as wheel:
            prefix = f"{_PACKAGE_PATH}/src/"
            files = {
                name.removeprefix(prefix)
                for name in wheel.namelist()
                if name.startswith(prefix) and not name.endswith("/")
            }
            self.assertEqual(files, set(manifest["files"]))
            for name, digest in manifest["files"].items():
                with self.subTest(file=name):
                    self.assertEqual(
                        hashlib.sha256(wheel.read(prefix + name)).hexdigest(), digest
                    )
            for name in (
                "__init__.py",
                "VENDOR.md",
                "SKILL.md",
                "export_manifest.json",
            ):
                with self.subTest(file=name):
                    self.assertEqual(
                        wheel.read(f"{_PACKAGE_PATH}/{name}"),
                        (_PACKAGE / name).read_bytes(),
                    )


if __name__ == "__main__":
    unittest.main()
