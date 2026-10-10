# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project

"""CPU regression for re-pinning selected tiles; no CUDA compilation or execution."""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REL = "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py"


def validate_pin(forced, *, selected=(128, 128, 32, 512), memory=100000):
    tree = ast.parse((ROOT / REL).read_text())
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    compile_fn = functions["compile_w4a16_fused_moe"]
    branch = next(
        node
        for node in reversed(compile_fn.body)
        if isinstance(node, ast.If) and "force_tile_config" in ast.unparse(node.test)
    )
    ns = {
        "_STAGES": 4,
        "_PACK_FACTOR": 8,
        "_covering_count": lambda n, d: (n + d - 1) // d,
        "_scale_group_size": lambda fmt: 32 if fmt == "e8m0_k32" else 16,
        "_normalize_scale_format": lambda fmt: fmt,
        "force_tile_config": forced,
        "fc1_tile_k": selected[0],
        "fc1_tile_n": selected[1],
        "fc2_tile_k": selected[2],
        "fc2_tile_n": selected[3],
        "fc1_cta_threads": 256,
        "fc2_cta_threads": 256,
        "fc1_cols": 1024,
        "hidden_size": 2048,
        "intermediate_size": 512,
        "moe_block_size": 8,
        "max_shared_mem": memory,
        "scale_format": "e8m0_k32",
        "weight_layout": "packed",
        "allow_native_logical_tail": False,
    }
    module = ast.Module(
        body=[
            functions["_shared_memory_footprint"],
            functions["_candidate_tile_fits"],
            branch,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / REL), "exec"), ns)
    return tuple(
        ns[x] for x in ("fc1_tile_k", "fc1_tile_n", "fc2_tile_k", "fc2_tile_n")
    )


class TileRoundtripContract(unittest.TestCase):
    def test_selected_ultrawide_geometry_roundtrips(self):
        self.assertEqual(validate_pin((128, 128, 32, 512)), (128, 128, 32, 512))

    def test_regular_explicit_geometry_still_valid(self):
        self.assertEqual(validate_pin((128, 128, 64, 256)), (128, 128, 64, 256))

    def test_mismatched_thread_count_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "thread counts must match"):
            validate_pin((64, 128, 64, 256))

    def test_unselected_small_k_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not fit"):
            validate_pin((64, 256, 32, 512))

    def test_alternate_shared_memory_overflow_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not fit"):
            validate_pin((128, 128, 64, 256), memory=1024)


if __name__ == "__main__":
    unittest.main()
