# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Delivered source package integrity without loading GPU kernels."""

import hashlib
import json
from pathlib import Path

from flashinfer.experimental.sm110_xqa import jit
from flashinfer.experimental.sm110_xqa.backend import _TREE_FAMILIES

PACKAGE = Path(jit.__file__).resolve().parent
LEDGER = PACKAGE.parents[2] / "benchmarks" / "sm110_xqa_shapes.json"
PLAN_KINDS = {
    "parameter",
    "buffer",
    "tma_buffer",
    "pod_field",
    "nullable_raw_pointer",
    "grid",
}


def test_every_route_names_a_delivered_program():
    programs = set(jit.MODULES) | set(jit.FROZEN)
    assert set(jit.ROUTES.values()) <= programs
    assert not set(jit.MODULES) & set(jit.FROZEN)
    for name, record in (*jit.MODULES.items(), *jit.FROZEN.items()):
        kernel, binding = record["sources"]
        assert kernel.endswith("_kernel.cu") and binding.endswith("_binding.cu"), name
        for source in record["sources"]:
            assert (PACKAGE / source).is_file(), f"{name}: missing {source}"
        assert "--device-int128" not in record["compile_flags"], name
        assert record["ffi_entry"].isidentifier(), name


def test_generated_programs_declare_bindable_argument_plans():
    for name, record in jit.MODULES.items():
        # The registry is a JSON-like literal: every plan entry is a [kind, key] pair.
        plan = [tuple(item) for item in record["arg_plan"]]
        assert {kind for kind, _ in plan} <= PLAN_KINDS, name
        assert [key for kind, key in plan if kind == "grid"] == [
            "grid_x",
            "grid_y",
            "grid_z",
        ], name
        assert ("parameter", "head_group_size") in plan, name
        if record["ratios"] is not None:
            assert record["ratios"] == [2, 4, 8, 16], name


# grid.x of the host's tree launch, per family; the programs' clusters are recorded in
# jit.MODULES: (1, 1, 1) register families, (2, 2, 1) / (2, 1, 1) tmem, (4, 1, 1) pair.
TREE_GRID_WIDTH = {"register_mma": 1, "register_mma_split": 1, "tmem": 2, "pair": 4}


def test_tree_family_grid_width_is_the_program_cluster_width():
    # backend.prepare launches a tree program on grid.x = ctas_per_tile * tiles_per_row;
    # that extent must be exactly the cluster width of every program serving the family,
    # or clusters would repeat (or miss) output tiles.
    assert set(_TREE_FAMILIES) == set(TREE_GRID_WIDTH)
    for family, (suffix, _rows, ctas_per_tile, tiles_per_row) in _TREE_FAMILIES.items():
        grid_x = ctas_per_tile * tiles_per_row
        assert grid_x == TREE_GRID_WIDTH[family], family
        programs = {
            program
            for key, program in jit.ROUTES.items()
            if key.split("__")[0].endswith(suffix)
        }
        assert programs, family
        for program in sorted(programs):
            cluster = jit.MODULES[program]["cluster"]
            assert grid_x == cluster[0], (family, program, cluster)


def test_frozen_sources_match_their_recorded_hash():
    for name, record in jit.FROZEN.items():
        digest = hashlib.sha256()
        for source in record["sources"]:
            digest.update((PACKAGE / source).read_bytes())
        assert digest.hexdigest() == record["sources_sha256"], name


def test_ledger_tree_rows_are_served():
    ledger = json.loads(LEDGER.read_text())
    assert ledger["architecture"] == "sm_110a"
    assert ledger["shape_count"] == len(ledger["shapes"])
    for shape in ledger["shapes"]:
        if shape["head_dim"] != 512:
            assert shape["name"].startswith("decode_")
            assert "decode_fp16_contiguous" in jit.FROZEN
            continue
        route = shape["name"]
        if route in jit.FROZEN:
            continue
        forms = ("any", "even", "odd")
        assert any(f"{route}__{form}" in jit.ROUTES for form in forms), route
