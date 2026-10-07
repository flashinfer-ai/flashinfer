"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# JIT loader for the Cake fused MLA context K/V pack (``concat_mla_kv_quant_fp8``).
#
# The fused kernel is ONE generated Cake program per head group -- the number of
# consecutive head pairs a warp owns (1, 2, 3 or 4), chosen on the host from
# ``(num_tokens, num_heads)`` by ``flashinfer.mla_kv_pack._plan_head_group`` --
# and every program serves every token count and head count (both are runtime
# launch arguments) on both exact targets, compiled with the exact flag set of
# the device it runs on.  ``MODULES`` and ``ROUTES`` are written by the Cake
# export; the empty tables are the source-only placeholder and every route
# lookup fails until they are filled.

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags, sm103a_nvcc_flags

ConcatMlaKvQuantFp8Target = Literal["sm100a", "sm103a"]

_TARGET_FLAGS: dict[str, list[str]] = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
# Exact compute capabilities the delivered programs are compiled for.  Any other
# device (including other 10.x / 12.x parts) takes the composable torch path of
# the public API; the loader never cross-routes a cubin.
_TARGET_BY_CAPABILITY: dict[tuple[int, int], ConcatMlaKvQuantFp8Target] = {
    (10, 0): "sm100a",
    (10, 3): "sm103a",
}
GENERATED_DIR = "cake_concat_mla_kv_quant_fp8"

# program name -> {"sources": [kernel, binding] (relative to the csrc family
# directory), "template", "ffi_entry", "compile_flags", "arg_plan",
# "launch_block", "cluster", "dynamic_smem_bytes", "use_pdl", "head_group",
# "min_blocks", "closure": {target: sealed closure identity}}.
MODULES: dict[str, dict[str, Any]] = {
    "cake_concat_mla_kv_quant_fp8_75d7ea09e4023fab4fa8": {
        "sources": [
            "cake_concat_mla_kv_quant_fp8_75d7ea09e4023fab4fa8_kernel.cu",
            "cake_concat_mla_kv_quant_fp8_75d7ea09e4023fab4fa8_binding.cu",
        ],
        "template": "concat_mla_kv_quant_fp8_hg4",
        "head_group": 4,
        "min_blocks": 3,
        "ffi_entry": "run",
        "compile_flags": [],
        "arg_plan": [
            ["buffer", "kv_nope"],
            ["buffer", "k_pe"],
            ["buffer", "key"],
            ["buffer", "value"],
            ["parameter", "num_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "head_pairs"],
            ["parameter", "warps_per_token"],
            ["parameter", "k_pe_row_stride"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch_block": [256, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 0,
        "use_pdl": False,
        "closure": {
            "sm100a": "4d0fc9fbd52203dda988e955e25c3f00cbf5d48d5795a6545baea1c9db7f1efe",
            "sm103a": "b47d84102a29aac63b18ab4d7a1e6e6b3f9cc01a3233df2dd381df7abe79f4d5",
        },
    },
    "cake_concat_mla_kv_quant_fp8_8dea7038f02767ca1cbf": {
        "sources": [
            "cake_concat_mla_kv_quant_fp8_8dea7038f02767ca1cbf_kernel.cu",
            "cake_concat_mla_kv_quant_fp8_8dea7038f02767ca1cbf_binding.cu",
        ],
        "template": "concat_mla_kv_quant_fp8_hg2",
        "head_group": 2,
        "min_blocks": 6,
        "ffi_entry": "run",
        "compile_flags": [],
        "arg_plan": [
            ["buffer", "kv_nope"],
            ["buffer", "k_pe"],
            ["buffer", "key"],
            ["buffer", "value"],
            ["parameter", "num_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "head_pairs"],
            ["parameter", "warps_per_token"],
            ["parameter", "k_pe_row_stride"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch_block": [256, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 0,
        "use_pdl": False,
        "closure": {
            "sm100a": "9df116f7f74bc719948c7f303c2632d5bf6772eec411f377f5ebb902a1006376",
            "sm103a": "018e27f2fa192ab283ce601a329778ca40935f306099201a865a68a3a298a47b",
        },
    },
    "cake_concat_mla_kv_quant_fp8_982430f4a5a8e3e38334": {
        "sources": [
            "cake_concat_mla_kv_quant_fp8_982430f4a5a8e3e38334_kernel.cu",
            "cake_concat_mla_kv_quant_fp8_982430f4a5a8e3e38334_binding.cu",
        ],
        "template": "concat_mla_kv_quant_fp8_hg3",
        "head_group": 3,
        "min_blocks": 3,
        "ffi_entry": "run",
        "compile_flags": [],
        "arg_plan": [
            ["buffer", "kv_nope"],
            ["buffer", "k_pe"],
            ["buffer", "key"],
            ["buffer", "value"],
            ["parameter", "num_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "head_pairs"],
            ["parameter", "warps_per_token"],
            ["parameter", "k_pe_row_stride"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch_block": [256, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 0,
        "use_pdl": False,
        "closure": {
            "sm100a": "862a5678fb5712efe82bfef2b9e3ae701306a21d2c8707e805ae7df7b238358e",
            "sm103a": "afce15ae436f6c8a95f754f14f5e8cbb004f8fa5182a40584f77fba2f3c464f2",
        },
    },
    "cake_concat_mla_kv_quant_fp8_bff13b38c6c98eb5c9a9": {
        "sources": [
            "cake_concat_mla_kv_quant_fp8_bff13b38c6c98eb5c9a9_kernel.cu",
            "cake_concat_mla_kv_quant_fp8_bff13b38c6c98eb5c9a9_binding.cu",
        ],
        "template": "concat_mla_kv_quant_fp8_hg1",
        "head_group": 1,
        "min_blocks": 8,
        "ffi_entry": "run",
        "compile_flags": [],
        "arg_plan": [
            ["buffer", "kv_nope"],
            ["buffer", "k_pe"],
            ["buffer", "key"],
            ["buffer", "value"],
            ["parameter", "num_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "head_pairs"],
            ["parameter", "warps_per_token"],
            ["parameter", "k_pe_row_stride"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch_block": [256, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 0,
        "use_pdl": False,
        "closure": {
            "sm100a": "a07ec972d38808df8add4e8cf3f825fda151f18b61bcccd0f79c4d5e9c9daf7c",
            "sm103a": "c3f2ba98842df276e8d7b5a5aae7d106d16421f8db1f04e9e4686921b8b5227d",
        },
    },
}
# "hg<head_group>" -> {"module": program name}.
ROUTES: dict[str, dict[str, Any]] = {
    "hg1": {"module": "cake_concat_mla_kv_quant_fp8_bff13b38c6c98eb5c9a9"},
    "hg2": {"module": "cake_concat_mla_kv_quant_fp8_8dea7038f02767ca1cbf"},
    "hg3": {"module": "cake_concat_mla_kv_quant_fp8_982430f4a5a8e3e38334"},
    "hg4": {"module": "cake_concat_mla_kv_quant_fp8_75d7ea09e4023fab4fa8"},
}


def route_key(head_group: int) -> str:
    """Route-table key of one head group."""
    return f"hg{int(head_group)}"


def head_groups() -> tuple[int, ...]:
    """Head groups with a delivered program, ascending (empty in the source-only placeholder)."""
    return tuple(sorted(int(key[2:]) for key in ROUTES))


def concat_mla_kv_quant_fp8_target_for_capability(
    capability: Sequence[int],
) -> Optional[ConcatMlaKvQuantFp8Target]:
    """Exact target of a ``(major, minor)`` compute capability, ``None`` when no program is built for it."""
    return _TARGET_BY_CAPABILITY.get((int(capability[0]), int(capability[1])))


def concat_mla_kv_quant_fp8_target(device: torch.device) -> ConcatMlaKvQuantFp8Target:
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    target = concat_mla_kv_quant_fp8_target_for_capability(capability)
    if target is None:
        raise RuntimeError(
            "concat_mla_kv_quant_fp8 Cake programs are built for exact compute capability "
            f"10.0 or 10.3, got {capability[0]}.{capability[1]}"
        )
    return target


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / GENERATED_DIR
    checkout = Path(__file__).resolve().parents[2] / "csrc" / GENERATED_DIR
    for candidate in (installed, checkout):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "concat_mla_kv_quant_fp8 CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


def _route(head_group: int) -> dict[str, Any]:
    key = route_key(head_group)
    try:
        return ROUTES[key]
    except KeyError as exc:
        raise RuntimeError(
            f"no exact concat_mla_kv_quant_fp8 route for head group {key}"
        ) from exc


def concat_mla_kv_quant_fp8_route_record(
    device: torch.device, head_group: int
) -> dict[str, Any]:
    """``{"head_group", "target", "module"}`` for one head group on ``device``.

    ``module`` is the generated program record plus its ``name``; the public
    operation binds the launch arguments from its ``arg_plan`` and passes the
    host-planned grid.
    """
    target = concat_mla_kv_quant_fp8_target(device)
    route = _route(head_group)
    name = str(route["module"])
    return {
        "head_group": int(head_group),
        "target": target,
        "module": {**MODULES[name], "name": name},
    }


@functools.cache
def gen_concat_mla_kv_quant_fp8_module(
    target: ConcatMlaKvQuantFp8Target, head_group: int
) -> JitSpec:
    """JIT spec of one (target, head group) instantiation of the head group's program."""
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported concat_mla_kv_quant_fp8 target: {target}")
    route = _route(head_group)
    name = str(route["module"])
    record = MODULES[name]
    csrc = _get_csrc_dir()
    spec = gen_jit_spec(
        name=f"cake_concat_mla_kv_quant_fp8_{target}_hg{int(head_group)}_{record['closure'][target][:20]}",
        sources=[csrc / source for source in record["sources"]],
        extra_cuda_cflags=[*_TARGET_FLAGS[target], *record["compile_flags"]],
        extra_include_paths=[csrc, csrc.parent, _get_include_dir()],
        needs_device_linking=True,
    )
    logger.info(
        "Generated concat_mla_kv_quant_fp8 %s head group %d JIT spec: %s",
        target,
        int(head_group),
        spec.name,
    )
    return spec


@functools.cache
def load_concat_mla_kv_quant_fp8_build(
    target: ConcatMlaKvQuantFp8Target, head_group: int
):
    return gen_concat_mla_kv_quant_fp8_module(target, head_group).build_and_load()


def load_concat_mla_kv_quant_fp8_module(device: torch.device, head_group: int):
    return load_concat_mla_kv_quant_fp8_build(
        concat_mla_kv_quant_fp8_target(device), head_group
    )


def gen_concat_mla_kv_quant_fp8_aot_modules(
    target: ConcatMlaKvQuantFp8Target,
) -> tuple[JitSpec, ...]:
    """Deduplicated JIT specs of every head group for one exact physical target (AOT inventory)."""
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported concat_mla_kv_quant_fp8 target: {target}")
    specs: dict[str, JitSpec] = {}
    for head_group in head_groups():
        spec = gen_concat_mla_kv_quant_fp8_module(target, head_group)
        specs.setdefault(spec.name, spec)
    return tuple(specs.values())


__all__ = [
    "GENERATED_DIR",
    "MODULES",
    "ROUTES",
    "ConcatMlaKvQuantFp8Target",
    "concat_mla_kv_quant_fp8_route_record",
    "concat_mla_kv_quant_fp8_target",
    "concat_mla_kv_quant_fp8_target_for_capability",
    "gen_concat_mla_kv_quant_fp8_aot_modules",
    "gen_concat_mla_kv_quant_fp8_module",
    "head_groups",
    "load_concat_mla_kv_quant_fp8_build",
    "load_concat_mla_kv_quant_fp8_module",
    "route_key",
]
