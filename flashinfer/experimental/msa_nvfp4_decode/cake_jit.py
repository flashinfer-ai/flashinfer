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

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Explicit target-owned registration of the generated programs.  One record per
# (architecture, route, split-KV factor): the ``swap_tsk`` records (one per
# split factor) carry the persistent decode kernel with its translation units,
# compile flags, FFI entry and argument plan, plus the resident-CTA count per SM
# the host uses to size the persistent grid; the optional ``short`` record of an
# architecture carries the eight-CTA-cluster register-MMA program that serves
# work items of at most ``max_pages`` selected pages (``cluster`` CTAs per item,
# at most ``max_clusters`` clusters per launch).  Populated verbatim by the
# generated-program export; do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_msa_nvfp4_decode_sm_100a_short": {
        "arch": "sm_100a",
        "route": "short",
        "max_pages": 4,
        "cluster": 8,
        "max_clusters": 16,
        "main": {
            "module": "cake_msa_nvfp4_decode_b96a36ad944b04b6eaea",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_b96a36ad944b04b6eaea_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_b96a36ad944b04b6eaea_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "Q"],
                ["buffer", "K"],
                ["buffer", "K_scale"],
                ["buffer", "V"],
                ["buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["parameter", "k_page_stride"],
                ["parameter", "k_head_stride"],
                ["parameter", "ks_page_stride"],
                ["parameter", "ks_head_stride"],
                ["parameter", "v_page_stride"],
                ["parameter", "v_head_stride"],
                ["parameter", "vs_page_stride"],
                ["parameter", "vs_head_stride"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "888c29d9cd567d9895fbed0d9262d5c494b5002c460aa57ce2cd73590b85cd4a",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "888c29d9cd567d9895fbed0d9262d5c494b5002c460aa57ce2cd73590b85cd4a",
    },
    "cake_msa_nvfp4_decode_sm_100a_split1": {
        "arch": "sm_100a",
        "route": "swap_tsk",
        "splits": 1,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_3364243f5992ccf9ac63",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_3364243f5992ccf9ac63_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_3364243f5992ccf9ac63_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "12a60c8ef7df2dcf01d71fc6f79ee8644994c7fe1b987a5bb77282bba204e3a1",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "12a60c8ef7df2dcf01d71fc6f79ee8644994c7fe1b987a5bb77282bba204e3a1",
    },
    "cake_msa_nvfp4_decode_sm_100a_split2": {
        "arch": "sm_100a",
        "route": "swap_tsk",
        "splits": 2,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_08a06dbdabbe2724b57b",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_08a06dbdabbe2724b57b_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_08a06dbdabbe2724b57b_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "3864570379b6dc650302919e9d89a098cc1eeeb881ca774435a8941ca0a57d86",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "3864570379b6dc650302919e9d89a098cc1eeeb881ca774435a8941ca0a57d86",
    },
    "cake_msa_nvfp4_decode_sm_100a_split4": {
        "arch": "sm_100a",
        "route": "swap_tsk",
        "splits": 4,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_195875cbef12a3199df3",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_195875cbef12a3199df3_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_195875cbef12a3199df3_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "dc160a04e604f6290014cf4a481263d813f01dca46f3186e8aa9795c674f5b68",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "dc160a04e604f6290014cf4a481263d813f01dca46f3186e8aa9795c674f5b68",
    },
    "cake_msa_nvfp4_decode_sm_100a_split8": {
        "arch": "sm_100a",
        "route": "swap_tsk",
        "splits": 8,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_45b58b7383003e4ae409",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_45b58b7383003e4ae409_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_45b58b7383003e4ae409_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "6db37c08efd047a771b0c65f875311e4d3a3e8e2450ef10cde2330518f899f39",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "6db37c08efd047a771b0c65f875311e4d3a3e8e2450ef10cde2330518f899f39",
    },
    "cake_msa_nvfp4_decode_sm_103a_short": {
        "arch": "sm_103a",
        "route": "short",
        "max_pages": 4,
        "cluster": 8,
        "max_clusters": 16,
        "main": {
            "module": "cake_msa_nvfp4_decode_4ccf4bb3f41cf8a00195",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_4ccf4bb3f41cf8a00195_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_4ccf4bb3f41cf8a00195_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "Q"],
                ["buffer", "K"],
                ["buffer", "K_scale"],
                ["buffer", "V"],
                ["buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["parameter", "k_page_stride"],
                ["parameter", "k_head_stride"],
                ["parameter", "ks_page_stride"],
                ["parameter", "ks_head_stride"],
                ["parameter", "v_page_stride"],
                ["parameter", "v_head_stride"],
                ["parameter", "vs_page_stride"],
                ["parameter", "vs_head_stride"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "90ae446e504a24b95d66f267e3e6c7ee5291c5ee0f9bd0f23c3fa94f5e1d137b",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "90ae446e504a24b95d66f267e3e6c7ee5291c5ee0f9bd0f23c3fa94f5e1d137b",
    },
    "cake_msa_nvfp4_decode_sm_103a_split1": {
        "arch": "sm_103a",
        "route": "swap_tsk",
        "splits": 1,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_1593eec93d43e0decdfc",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_1593eec93d43e0decdfc_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_1593eec93d43e0decdfc_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "ede62c5ff8ff02c078dc07395669372b1cbd2ee94e889d678a0692ee12cffcbf",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "ede62c5ff8ff02c078dc07395669372b1cbd2ee94e889d678a0692ee12cffcbf",
    },
    "cake_msa_nvfp4_decode_sm_103a_split2": {
        "arch": "sm_103a",
        "route": "swap_tsk",
        "splits": 2,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_6ebd6c45a5ad143dc80c",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_6ebd6c45a5ad143dc80c_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_6ebd6c45a5ad143dc80c_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "a9a0d70c640f16755e78513759f3b0df3fdfc8dd426dbe8069148382a525f1aa",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "a9a0d70c640f16755e78513759f3b0df3fdfc8dd426dbe8069148382a525f1aa",
    },
    "cake_msa_nvfp4_decode_sm_103a_split4": {
        "arch": "sm_103a",
        "route": "swap_tsk",
        "splits": 4,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_f996a9dd751808dedee4",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_f996a9dd751808dedee4_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_f996a9dd751808dedee4_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "db597f2ff58402404ddab5fc34a3d1bbe0e8521b5d8c2cd1632bf7d5369132a9",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "db597f2ff58402404ddab5fc34a3d1bbe0e8521b5d8c2cd1632bf7d5369132a9",
    },
    "cake_msa_nvfp4_decode_sm_103a_split8": {
        "arch": "sm_103a",
        "route": "swap_tsk",
        "splits": 8,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_64ab063a449951d65342",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_64ab063a449951d65342_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_64ab063a449951d65342_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "8372bba6bc5fc003f03fca4723a1f9d4120bb8163c83ddff3ad597561768b88e",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "8372bba6bc5fc003f03fca4723a1f9d4120bb8163c83ddff3ad597561768b88e",
    },
}

STAGES = ("main",)
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def route_of(record: dict[str, Any]) -> str:
    """Route family of a registry record (``swap_tsk`` persistent or ``short`` cluster program)."""
    return str(record.get("route", "swap_tsk"))


def registered_split_factors(arch: str) -> tuple[int, ...]:
    """Split-KV factors with a registered persistent program for ``arch``."""
    return tuple(
        sorted(
            int(record["splits"])
            for record in MODULES.values()
            if record["arch"] == arch and route_of(record) == "swap_tsk"
        )
    )


def select_module(arch: str, splits: int) -> str:
    """Return the registered persistent module name for ``arch`` and split factor ``splits``."""
    for name, record in MODULES.items():
        if (
            record["arch"] == arch
            and route_of(record) == "swap_tsk"
            and int(record["splits"]) == int(splits)
        ):
            return name
    raise NotImplementedError(
        "The generated NVFP4 MSA decode program for "
        f"{arch} with split factor {splits} is not registered in this checkout"
    )


def select_short_module(arch: str) -> str | None:
    """Return the short-item cluster program registered for ``arch``, or ``None``.

    The short program serves work items whose requests span at most
    ``MODULES[name]["max_pages"]`` pages; an architecture without it serves
    those items with the persistent program.
    """
    names = [
        name
        for name, record in MODULES.items()
        if record["arch"] == arch and route_of(record) == "short"
    ]
    if len(names) > 1:
        raise NotImplementedError(
            f"{arch} registers more than one short-item NVFP4 MSA decode program: {names}"
        )
    return names[0] if names else None


def _header_dirs():
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer/layout.cuh"
    ).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer/layout.cuh"
    ).is_file():
        return source
    raise FileNotFoundError("FlashInfer binding headers were not found")


@functools.cache
def gen_cake_msa_nvfp4_decode_module(name, stage):
    record = MODULES[name]
    physical = record[stage]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in physical["sources"]]
    return gen_jit_spec(
        name=f"{name}_{stage}_" + physical["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *physical["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_msa_nvfp4_decode_module(name, stage):
    return gen_cake_msa_nvfp4_decode_module(name, stage).build_and_load()
