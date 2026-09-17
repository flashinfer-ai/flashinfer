# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Native JIT loader for frozen SM110a attention sources."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import JitSpec, gen_jit_spec, sm110a_nvcc_flags

SCHEMA = "flashinfer.sm110_xqa.v1"
DECODE_SINGLE_PARTITION_ROUTE = "decode_fp16_contiguous_single_partition"
ROUTE_ENTRIES = {
    "tree_fp16_contiguous": "run_tree",
    "tree_fp16_paged": "run_tree",
    "tree_fp8_contiguous": "run_tree",
    "tree_fp8_paged": "run_tree",
    "decode_fp16_contiguous": "run_decode",
    "decode_merge": "run_decode_merge",
}


def _source_root() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "sm110_xqa"


def _read_manifest(source_root: Path) -> dict[str, Any]:
    path = source_root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Frozen SM110 XQA sources have not been installed; "
            "this integration scaffold cannot execute attention."
        )
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("architecture") != "sm_110a":
        raise ValueError("SM110 XQA requires the versioned exact-sm_110a manifest")
    routes = manifest.get("routes")
    if not isinstance(routes, dict) or not set(ROUTE_ENTRIES).issubset(routes):
        raise ValueError(
            "SM110 XQA manifest must enumerate all six base physical routes"
        )
    producer = routes["decode_fp16_contiguous"]
    expected_entries = dict(ROUTE_ENTRIES)
    if producer.get("merge_stats_cache") is True:
        expected_entries[DECODE_SINGLE_PARTITION_ROUTE] = "run_decode"
    if set(routes) != set(expected_entries):
        raise ValueError(
            "stats-cache selection requires exactly its declared physical routes"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("SM110 XQA manifest has no frozen source files")
    for name, metadata in files.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix not in {".cu", ".cuh", ".h"}
        ):
            raise ValueError(f"invalid source path: {name}")
        source = source_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"frozen source file is missing: {name}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"frozen source digest mismatch: {name}")
    for name, route in routes.items():
        if route.get("ffi_entry") != expected_entries[name]:
            raise ValueError(f"unexpected FFI entry for {name}")
        sources = route.get("sources")
        if (
            not isinstance(sources, list)
            or not sources
            or any(
                source not in files or Path(source).suffix != ".cu"
                for source in sources
            )
        ):
            raise ValueError(f"invalid CUDA translation-unit list for {name}")
        options = route.get("nvcc_options")
        if not isinstance(options, list) or any(
            not isinstance(option, str) for option in options
        ):
            raise ValueError(f"invalid compiler options for {name}")
        if any(
            "gpu-architecture" in option
            or "gencode" in option
            or option.startswith("-arch")
            for option in options
        ):
            raise ValueError("route options must not override the exact SM110a target")
        if name.startswith("tree_"):
            rows, columns = route.get("tile_rows"), route.get("output_tile_columns")
            if (
                rows not in (64, 128)
                or columns not in (128, 256)
                or route.get("copy_warps") not in (4, 8)
            ):
                raise ValueError(f"unsupported tree launch geometry for {name}")
        elif name in ("decode_fp16_contiguous", DECODE_SINGLE_PARTITION_ROUTE):
            tokens = route.get("partition_tokens")
            if (
                type(tokens) is not int
                or not 0 < tokens <= (1 << 32) - 1
                or tokens % 64
            ):
                raise ValueError(
                    "decode partition size must fit uint32 and be a positive multiple of 64"
                )
            for field in ("fused_merge", "half_warp_merge", "merge_stats_cache"):
                if type(route.get(field)) is not bool:
                    raise ValueError(
                        f"decode route must declare its physical {field} selection"
                    )
            if (
                route.get("counter_initialization") != "zero_at_prepare"
                or route.get("workspace_replay") != "ordered"
            ):
                raise ValueError(
                    "decode route requires zero-at-prepare counters and ordered workspace replay"
                )
    if producer["merge_stats_cache"]:
        if not producer["fused_merge"] or not producer["half_warp_merge"]:
            raise ValueError("stats-cache route requires a fused half-warp merge")
        if producer.get("single_partition_route") != DECODE_SINGLE_PARTITION_ROUTE:
            raise ValueError(
                "stats-cache route must name its single-partition specialization"
            )
        direct = routes[DECODE_SINGLE_PARTITION_ROUTE]
        if direct["merge_stats_cache"] or "single_partition_route" in direct:
            raise ValueError(
                "single-partition route must disable stats cache and cannot redirect"
            )
        for field in (
            "partition_tokens",
            "fused_merge",
            "half_warp_merge",
            "counter_initialization",
            "workspace_replay",
            "nvcc_options",
        ):
            if direct[field] != producer[field]:
                raise ValueError(
                    f"single-partition route disagrees with producer {field}"
                )
    elif "single_partition_route" in producer:
        raise ValueError(
            "disabled stats-cache route cannot redirect single-partition launches"
        )
    return manifest


@functools.cache
def get_manifest() -> dict[str, Any]:
    return _read_manifest(_source_root())


def _header_directories() -> list[Path]:
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    if (checkout / "csrc" / "tvm_ffi_utils.h").is_file():
        return [checkout / "csrc", checkout / "include"]
    raise FileNotFoundError("FlashInfer's native TVM-FFI headers are unavailable")


@functools.cache
def gen_sm110_xqa_module(route_name: str) -> JitSpec:
    manifest = get_manifest()
    if route_name not in manifest["routes"]:
        raise ValueError(f"unknown SM110 XQA route: {route_name}")
    route = manifest["routes"][route_name]
    identity = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return gen_jit_spec(
        name=f"sm110_xqa_{route_name}_{identity}",
        sources=[_source_root() / source for source in route["sources"]],
        extra_cuda_cflags=[*sm110a_nvcc_flags, *route["nvcc_options"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            _source_root(),
            _source_root() / "sm_110a",
            *_header_directories(),
        ],
        # Math flags belong to the frozen route, not the JIT helper's default.
        use_fast_math=False,
    )


def require_sm110(device: Any) -> None:
    from ...utils import get_compute_capability, is_sm110a_supported

    capability = get_compute_capability(device)
    if capability != (11, 0) or not is_sm110a_supported(device):
        raise RuntimeError(
            "SM110 XQA requires exact compute capability 11.0 and CUDA 13.0 or newer"
        )


@functools.cache
def load_sm110_xqa_module(route_name: str) -> Any:
    return gen_sm110_xqa_module(route_name).build_and_load()
