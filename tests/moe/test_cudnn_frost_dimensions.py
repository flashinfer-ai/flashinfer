# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Cartesian catalog coverage and device-dependent launch ABI regression."""

import ast
import json
import shutil
from copy import deepcopy

import pytest

from flashinfer.fused_moe.backends.cudnn_frost.device import (
    DeviceFacts,
    specialize_parameters,
)
from flashinfer.fused_moe.backends.cudnn_frost.dimension_catalog import expand_catalog
from flashinfer.fused_moe.backends.cudnn_frost.manifest import (
    MANIFEST_FILENAME,
    read_manifest,
    write_manifest,
)
from flashinfer.fused_moe.backends.cudnn_frost.runtime import artifact_root
from flashinfer.fused_moe.backends.cudnn_frost.source_template import render_source

DTYPES = ("bf16", "mxfp8", "nvfp4", "mxfp8_mxfp4")


@pytest.mark.parametrize("dtype", DTYPES)
def test_dimension_catalog_covers_activation_tile_product(dtype):
    payload = json.loads((artifact_root(dtype) / MANIFEST_FILENAME).read_text())
    records = expand_catalog(payload)["kernels"]
    assert len({r["id"] for r in records}) == len(records)
    expected = 0
    for family in payload["families"]:
        activations = (
            ["identity"]
            if family["stage"] == "fc2"
            else payload["activation_groups"][family["activation_group"]]
        )
        for variant in family["variants"]:
            assert set(variant["activation_fragments"]) == set(activations)
            assert variant["tile_parameters"].keys() <= payload["tiles"].keys()
            expected += len(activations) * len(variant["tile_parameters"])
    assert len(records) == expected
    # Resource snapshots are never part of the tile recipes.
    for record in records:
        constants = dict(record["source"]["parameters"]["constants"])
        assert all(
            constants[name] == "1"
            for name in ("ab_stages", "grid_num_clusters", "swizzle_l2_budget_bytes")
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_dimension_catalog_writer_preserves_format_and_ids(dtype, tmp_path):
    original = artifact_root(dtype)
    root = tmp_path / dtype
    shutil.copytree(original, root)
    path = root / MANIFEST_FILENAME
    before = read_manifest(path)
    write_manifest(path, before)
    assert json.loads(path.read_text())["schema_version"] == 4
    assert read_manifest(path) == before
    encoded = path.read_bytes()
    write_manifest(path, read_manifest(path))
    assert path.read_bytes() == encoded


@pytest.mark.parametrize("dtype", DTYPES)
def test_device_specialization_keeps_host_workspace_and_activation_abi(dtype):
    root = artifact_root(dtype)
    records = read_manifest(root / MANIFEST_FILENAME)["kernels"]
    templates = {}
    for record in records:
        source = record["source"]
        path = source["path"]
        if path not in templates:
            templates[path] = (root / path).read_text()
        parameters = source["parameters"]
        original = deepcopy(parameters)
        for grid, smem, l2 in ((84, 232448, 50331648), (102, 334848, 117440512)):
            facts = DeviceFacts(record["arch"], smem, l2, (grid,) * 16)
            resolved, workspace = specialize_parameters(parameters, facts)
            constants = dict(resolved["constants"])
            concrete = render_source(templates[path], resolved)
            tree = ast.parse(concrete)
            host = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_host"
            )
            names = [a.arg for a in host.args.args]
            # SiTU alone has the two auxiliary activation scalars. This checks
            # every synthesized activation arm, including fused quantization.
            assert ("gate_scale" in names) == (
                record["contract"]["activation"] == "situ"
            )
            assert ("linear_scale" in names) == ("gate_scale" in names)
            assert int(constants["grid_num_clusters"]) == grid
            assert int(constants["swizzle_l2_budget_bytes"]) == l2 // 3
            assert 1 <= int(constants["ab_stages"]) <= 16
            cm, cn, _ = ast.literal_eval(constants["cluster_shape_mnk"])
            if record["arch"] == "sm_120a":
                slots = 0
            else:
                token_operands = int(
                    constants[
                        "num_b_operands" if resolved["swap_ab"] else "num_a_operands"
                    ]
                )
                slots = token_operands * (1 if dtype == "bf16" else 2)
                slots += int(constants.get("n_tma_outputs", "0"))
            assert workspace == (grid * cm * cn * slots + 1) * 128
        assert parameters == original
