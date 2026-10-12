# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project

"""Exercise the real scalar launch cache without importing CUDA libraries."""

import ast
import functools
import unittest
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py"
)


class LaunchResolutionContract(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        self.functions = {
            x.name: x for x in tree.body if isinstance(x, ast.FunctionDef)
        }
        self.calls = []

        def compile_launch(**options):
            self.calls.append(options)
            if options.get("force_tile_config") == (0, 0, 0, 0):
                raise ValueError("invalid tile")
            return object()

        self.ns = {
            "lru_cache": functools.lru_cache,
            "Any": object,
            "W4A16FusedMoeCompileResult": object,
            "compile_w4a16_fused_moe": compile_launch,
        }
        exec(
            compile(
                ast.Module(
                    body=[self.functions["_resolve_w4a16_fused_launch"]],
                    type_ignores=[],
                ),
                str(SOURCE),
                "exec",
            ),
            self.ns,
        )
        self.resolve = self.ns["_resolve_w4a16_fused_launch"]

    def test_identical_requests_resolve_once(self):
        result = self.resolve(0, size_m=1, force_tile_config=(128, 128, 32, 512))
        for _ in range(20):
            self.assertIs(
                self.resolve(0, size_m=1, force_tile_config=(128, 128, 32, 512)), result
            )
        self.assertEqual(len(self.calls), 1)

    def test_device_and_every_forwarded_option_separate_entries(self):
        flat = self.functions["_w4a16_fused_moe_launch_flat"]
        call = next(
            x
            for x in ast.walk(flat)
            if isinstance(x, ast.Call)
            and isinstance(x.func, ast.Name)
            and x.func.id == "_resolve_w4a16_fused_launch"
        )
        options = {
            x.arg: (1, 1, 1, 1) if x.arg == "force_tile_config" else 1
            for x in call.keywords
        }
        baseline = self.resolve(0, **options)
        self.assertIsNot(self.resolve(1, **options), baseline)
        for name in options:
            changed = dict(options)
            changed[name] = (2, 2, 2, 2) if name == "force_tile_config" else 2
            self.assertIsNot(self.resolve(0, **changed), baseline, name)
        self.assertEqual(len(self.calls), len(options) + 2)

    def test_invalid_requests_are_not_cached(self):
        for _ in range(2):
            with self.assertRaises(ValueError):
                self.resolve(0, force_tile_config=(0, 0, 0, 0))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.resolve.cache_info().currsize, 0)

    def test_cache_is_bounded(self):
        for m in range(300):
            self.resolve(0, size_m=m)
        self.assertEqual(self.resolve.cache_info().currsize, 256)

    def test_public_clear_invalidates_launch_resolution(self):
        first = self.resolve(0, size_m=1)
        for name in ["_CACHE", "_FUSED_CACHE", "_ACTIVATION_CACHE", "_SUM_CACHE"]:
            self.ns[name] = {"sentinel": True}
        self.ns["clear_compile_cache"] = lambda: None
        exec(
            compile(
                ast.Module(
                    body=[self.functions["clear_w4a16_kernel_cache"]], type_ignores=[]
                ),
                str(SOURCE),
                "exec",
            ),
            self.ns,
        )
        self.ns["clear_w4a16_kernel_cache"]()
        self.assertEqual(self.resolve.cache_info().currsize, 0)
        self.assertIsNot(self.resolve(0, size_m=1), first)
        self.assertEqual(len(self.calls), 2)


if __name__ == "__main__":
    unittest.main()
