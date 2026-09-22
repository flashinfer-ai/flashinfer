# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental source-package validation without loading GPU kernels."""

import hashlib
import json

import pytest

from flashinfer.experimental.sm110_xqa.jit import (
    DECODE_SINGLE_PARTITION_ROUTE,
    ROUTE_ENTRIES,
    SCHEMA,
    _read_manifest,
)


def _package(tmp_path):
    # This is source-package metadata test data, not an executable attention kernel.
    source = b"// source-package integrity fixture\n"
    (tmp_path / "integrity.cu").write_bytes(source)
    manifest = {
        "schema": SCHEMA,
        "architecture": "sm_110a",
        "files": {"integrity.cu": {"sha256": hashlib.sha256(source).hexdigest()}},
        "routes": {
            name: {
                "ffi_entry": entry,
                "sources": ["integrity.cu"],
                "nvcc_options": [],
                **(
                    {"tile_rows": 64, "output_tile_columns": 128, "copy_warps": 8}
                    if name.startswith("tree_")
                    else {}
                ),
                **(
                    {
                        "partition_tokens": 256,
                        "fused_merge": False,
                        "half_warp_merge": False,
                        "merge_stats_cache": False,
                        "counter_initialization": "zero_at_prepare",
                        "workspace_replay": "ordered",
                    }
                    if name == "decode_fp16_contiguous"
                    else {}
                ),
            }
            for name, entry in ROUTE_ENTRIES.items()
        },
    }
    return manifest


def _stats_cache_package(tmp_path):
    manifest = _package(tmp_path)
    producer = manifest["routes"]["decode_fp16_contiguous"]
    producer.update(fused_merge=True, half_warp_merge=True, merge_stats_cache=True)
    manifest["routes"][DECODE_SINGLE_PARTITION_ROUTE] = dict(
        producer, merge_stats_cache=False
    )
    producer["single_partition_route"] = DECODE_SINGLE_PARTITION_ROUTE
    return manifest


def _write(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_missing_sources_are_not_an_executable_package(tmp_path):
    with pytest.raises(FileNotFoundError, match="scaffold cannot execute"):
        _read_manifest(tmp_path)


def test_exact_source_package_round_trip(tmp_path):
    manifest = _package(tmp_path)
    _write(tmp_path, manifest)
    assert _read_manifest(tmp_path) == manifest


@pytest.mark.parametrize("arch", ["sm_100a", "sm_107a", "sm_110", "sm_110f", "sm_120a"])
def test_other_targets_do_not_alias_sm110a(tmp_path, arch):
    manifest = _package(tmp_path)
    manifest["architecture"] = arch
    _write(tmp_path, manifest)
    with pytest.raises(ValueError, match="exact-sm_110a"):
        _read_manifest(tmp_path)


def test_modified_device_source_is_rejected(tmp_path):
    manifest = _package(tmp_path)
    _write(tmp_path, manifest)
    (tmp_path / "integrity.cu").write_text("// changed source\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        _read_manifest(tmp_path)


def test_incomplete_route_inventory_is_rejected(tmp_path):
    manifest = _package(tmp_path)
    del manifest["routes"]["decode_merge"]
    _write(tmp_path, manifest)
    with pytest.raises(ValueError, match="all six"):
        _read_manifest(tmp_path)


def test_route_compiler_options_cannot_change_architecture(tmp_path):
    manifest = _package(tmp_path)
    manifest["routes"]["decode_merge"]["nvcc_options"] = ["-arch=sm_100a"]
    _write(tmp_path, manifest)
    with pytest.raises(ValueError, match="override"):
        _read_manifest(tmp_path)


@pytest.mark.parametrize("fused_merge", [False, True])
def test_explicit_decode_physical_candidate_round_trip(tmp_path, fused_merge):
    manifest = _package(tmp_path)
    manifest["routes"]["decode_fp16_contiguous"].update(
        fused_merge=fused_merge, partition_tokens=512
    )
    _write(tmp_path, manifest)
    assert _read_manifest(tmp_path) == manifest


def test_stats_cache_source_package_round_trip(tmp_path):
    manifest = _stats_cache_package(tmp_path)
    _write(tmp_path, manifest)
    assert _read_manifest(tmp_path) == manifest


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_direct",
        "unexpected_direct",
        "missing_redirect",
        "wrong_redirect",
        "disabled_redirect",
        "direct_cache",
        "direct_redirect",
        "unfused",
        "full_warp",
    ],
)
def test_stats_cache_route_inventory_and_dispatch_rejected(tmp_path, mutation):
    manifest = _stats_cache_package(tmp_path)
    routes = manifest["routes"]
    producer, direct = (
        routes["decode_fp16_contiguous"],
        routes[DECODE_SINGLE_PARTITION_ROUTE],
    )
    if mutation == "missing_direct":
        del routes[DECODE_SINGLE_PARTITION_ROUTE]
    elif mutation == "unexpected_direct":
        producer["merge_stats_cache"] = False
    elif mutation == "missing_redirect":
        del producer["single_partition_route"]
    elif mutation == "wrong_redirect":
        producer["single_partition_route"] = "decode_merge"
    elif mutation == "disabled_redirect":
        producer["merge_stats_cache"] = False
        del routes[DECODE_SINGLE_PARTITION_ROUTE]
    elif mutation == "direct_cache":
        direct["merge_stats_cache"] = True
    elif mutation == "direct_redirect":
        direct["single_partition_route"] = "decode_fp16_contiguous"
    elif mutation == "unfused":
        producer["fused_merge"] = False
    else:
        producer["half_warp_merge"] = False
    _write(tmp_path, manifest)
    with pytest.raises(ValueError, match="stats-cache|single-partition"):
        _read_manifest(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("partition_tokens", 512),
        ("fused_merge", False),
        ("half_warp_merge", False),
        ("nvcc_options", ["--use_fast_math"]),
    ],
)
def test_single_partition_specialization_contract_must_match(tmp_path, field, value):
    manifest = _stats_cache_package(tmp_path)
    manifest["routes"][DECODE_SINGLE_PARTITION_ROUTE][field] = value
    _write(tmp_path, manifest)
    with pytest.raises(ValueError, match="single-partition route disagrees"):
        _read_manifest(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fused_merge", None),
        ("fused_merge", 1),
        ("half_warp_merge", None),
        ("merge_stats_cache", None),
        ("merge_stats_cache", 1),
        ("partition_tokens", 65),
        ("partition_tokens", 1 << 32),
        ("counter_initialization", "zero_each_replay"),
        ("workspace_replay", "concurrent"),
    ],
)
def test_invalid_decode_workspace_or_candidate_contract(tmp_path, field, value):
    manifest = _package(tmp_path)
    manifest["routes"]["decode_fp16_contiguous"][field] = value
    _write(tmp_path, manifest)
    with pytest.raises(ValueError, match="decode"):
        _read_manifest(tmp_path)
