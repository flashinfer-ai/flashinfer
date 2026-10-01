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

# Explicit target-owned registration of the generated training programs.  One
# record per architecture.  A record carries ``arch``, the host binding
# profile ``abi`` (the keyword set its kernels expect, see ``cake_backend``),
# the list of kernel ``stages`` it registers, for a backward the layout of its
# FP32 dK/dV accumulators (``dkv_acc_layout``: ``"natural"`` row-major or
# ``"permuted"``, internal to the kernels and un-permuted by ``bwd_cast``),
# for a program with key-range passes the host policy that selects them
# (``key_pass_policy``, see ``cake_backend.KeyPassPolicy``), and one physical
# entry per stage (translation units, compile flags, FFI entry, argument
# plan, grid rule and closure identity).  Populated verbatim by the
# generated-program export; do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_dsa_h64_train_sm_100a": {
        "arch": "sm_100a",
        "abi": "dsa_h64_v1",
        "stages": [
            "fwd",
            "bwd_delta",
            "bwd_main",
            "bwd_compact",
            "bwd_main_pass",
            "bwd_cast",
        ],
        "dkv_acc_layout": "permuted",
        "key_pass_policy": {
            "l2_budget_bytes": 104857600,
            "key_bytes": 2304,
            "workspace_budget_bytes": 671088640,
            "token_chunk_multiple": 128,
        },
        "fwd": {
            "module": "cake_dsa_h64_train_b919726fbcb4b487a4bb",
            "sources": [
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_b919726fbcb4b487a4bb_kernel.cu",
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_b919726fbcb4b487a4bb_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "q_latent"],
                ["tma_buffer", "q_rope"],
                ["tma_buffer", "kv_latent"],
                ["buffer", "k_rope"],
                ["tma_buffer", "out"],
                ["buffer", "o_lo"],
                ["buffer", "lse"],
                ["buffer", "indices"],
                ["buffer", "topk_length"],
                ["parameter", "num_queries"],
                ["parameter", "num_kv"],
                ["parameter", "topk"],
                ["parameter", "idx_stride"],
                ["parameter", "indices_offset"],
                ["parameter", "k_rope_stride"],
                ["parameter", "k_rope_offset"],
                ["parameter", "has_topk_length"],
                ["parameter", "scale_log2"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "2ff7c2a8a97455dfe2c5d4f0215e8ea72c0058ad36286da2a7847a33d29ad756",
            "tma_workspace_bytes": 512,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_delta": {
            "module": "cake_dsa_h64_train_2061d8b248c23a071d54",
            "sources": [
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_2061d8b248c23a071d54_kernel.cu",
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_2061d8b248c23a071d54_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "dout"],
                ["buffer", "out"],
                ["buffer", "o_lo"],
                ["buffer", "delta"],
                ["parameter", "num_rows"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "1df6d33e687f409940b6aee156feebad55d105f2ab87778979d143ae6ca1aaa5",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_queries*8", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main": {
            "module": "cake_dsa_h64_train_b30777f5c742bca51a5a",
            "sources": [
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_b30777f5c742bca51a5a_kernel.cu",
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_b30777f5c742bca51a5a_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "q_latent"],
                ["tma_buffer", "q_rope"],
                ["tma_buffer", "dout"],
                ["tma_buffer", "dq_latent"],
                ["tma_buffer", "dq_rope"],
                ["tma_buffer", "kv_latent"],
                ["tma_buffer", "k_rope"],
                ["buffer", "lse"],
                ["buffer", "delta"],
                ["buffer", "indices"],
                ["buffer", "topk_length"],
                ["buffer", "dkv_f32"],
                ["buffer", "dkr_f32"],
                ["parameter", "num_queries"],
                ["parameter", "num_kv"],
                ["parameter", "topk"],
                ["parameter", "idx_stride"],
                ["parameter", "indices_offset"],
                ["parameter", "has_topk_length"],
                ["parameter", "token_base"],
                ["parameter", "token_step"],
                ["parameter", "scale_log2"],
                ["parameter", "sm_scale"],
                ["parameter", "pass_lo"],
                ["parameter", "pass_hi"],
                ["parameter", "dq_mode"],
                ["buffer", "dq_partial"],
                ["buffer", "key_scratch"],
                ["buffer", "pass_counts"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "38df6df200fa6071370858808cba5965528c7f6539f82fd8e5502bca7e4bcb11",
            "tma_workspace_bytes": 896,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_compact": {
            "module": "cake_dsa_h64_train_44e6567b736f319834ea",
            "sources": [
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_44e6567b736f319834ea_kernel.cu",
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_44e6567b736f319834ea_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "indices"],
                ["buffer", "topk_length"],
                ["buffer", "key_scratch"],
                ["buffer", "pass_counts"],
                ["parameter", "num_tokens"],
                ["parameter", "topk"],
                ["parameter", "idx_stride"],
                ["parameter", "indices_offset"],
                ["parameter", "has_topk_length"],
                ["parameter", "token_base"],
                ["parameter", "token_step"],
                ["parameter", "pass_lo"],
                ["parameter", "pass_hi"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "46fbb31517ea2f6c19a48fe7c7411415824bdadf811a8b53aba50e7658195e21",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_queries/4", 1, 1],
            "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main_pass": {
            "module": "cake_dsa_h64_train_3b4d979404748d282f34",
            "sources": [
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_3b4d979404748d282f34_kernel.cu",
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_3b4d979404748d282f34_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "q_latent"],
                ["tma_buffer", "q_rope"],
                ["tma_buffer", "dout"],
                ["tma_buffer", "dq_latent"],
                ["tma_buffer", "dq_rope"],
                ["tma_buffer", "kv_latent"],
                ["tma_buffer", "k_rope"],
                ["buffer", "lse"],
                ["buffer", "delta"],
                ["buffer", "indices"],
                ["buffer", "topk_length"],
                ["buffer", "dkv_f32"],
                ["buffer", "dkr_f32"],
                ["parameter", "num_queries"],
                ["parameter", "num_kv"],
                ["parameter", "topk"],
                ["parameter", "idx_stride"],
                ["parameter", "indices_offset"],
                ["parameter", "has_topk_length"],
                ["parameter", "token_base"],
                ["parameter", "token_step"],
                ["parameter", "scale_log2"],
                ["parameter", "sm_scale"],
                ["parameter", "pass_lo"],
                ["parameter", "pass_hi"],
                ["parameter", "dq_mode"],
                ["buffer", "dq_partial"],
                ["buffer", "key_scratch"],
                ["buffer", "pass_counts"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "ee084aceacbc0935a60c546c29349b5692459ea4ce52e2a056cdca1397f69b99",
            "tma_workspace_bytes": 896,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_cast": {
            "module": "cake_dsa_h64_train_8c9d80dc4d003d3b3d8c",
            "sources": [
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_8c9d80dc4d003d3b3d8c_kernel.cu",
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_8c9d80dc4d003d3b3d8c_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "src_latent"],
                ["buffer", "src_rope"],
                ["buffer", "dst_latent"],
                ["buffer", "dst_rope"],
                ["buffer", "dst_latent_f32"],
                ["buffer", "dst_rope_f32"],
                ["parameter", "latent_groups"],
                ["parameter", "rope_groups"],
                ["parameter", "out_f32"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "80cf640c6444bd4e0001447fdeed7ffdbb485d2294f7fabdca838e9778b76fad",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_kv*18/256", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "58e3826cf85e7e85bea1585201938bef723fde96c1eaf7bdb90f2ecb9baaf98d",
    },
    "cake_dsa_h64_train_sm_103a": {
        "arch": "sm_103a",
        "abi": "dsa_h64_v1",
        "stages": [
            "fwd",
            "bwd_delta",
            "bwd_main",
            "bwd_compact",
            "bwd_main_pass",
            "bwd_cast",
        ],
        "dkv_acc_layout": "permuted",
        "key_pass_policy": {
            "l2_budget_bytes": 104857600,
            "key_bytes": 2304,
            "workspace_budget_bytes": 671088640,
            "token_chunk_multiple": 128,
        },
        "fwd": {
            "module": "cake_dsa_h64_train_47ab04b028d5c2ec650e",
            "sources": [
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_47ab04b028d5c2ec650e_kernel.cu",
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_47ab04b028d5c2ec650e_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "q_latent"],
                ["tma_buffer", "q_rope"],
                ["tma_buffer", "kv_latent"],
                ["buffer", "k_rope"],
                ["tma_buffer", "out"],
                ["buffer", "o_lo"],
                ["buffer", "lse"],
                ["buffer", "indices"],
                ["buffer", "topk_length"],
                ["parameter", "num_queries"],
                ["parameter", "num_kv"],
                ["parameter", "topk"],
                ["parameter", "idx_stride"],
                ["parameter", "indices_offset"],
                ["parameter", "k_rope_stride"],
                ["parameter", "k_rope_offset"],
                ["parameter", "has_topk_length"],
                ["parameter", "scale_log2"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "0357ed5b455fe454ce7c0a7a195432078b8d5bcd60f20ce6ada930a70835a77d",
            "tma_workspace_bytes": 512,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_delta": {
            "module": "cake_dsa_h64_train_e5a5d9697c2d53b81502",
            "sources": [
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_e5a5d9697c2d53b81502_kernel.cu",
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_e5a5d9697c2d53b81502_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "dout"],
                ["buffer", "out"],
                ["buffer", "o_lo"],
                ["buffer", "delta"],
                ["parameter", "num_rows"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "5d7cf59e4c495e05eeccead89b6df6a8051142b8e290efc51f162e2fca411ec4",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_queries*8", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main": {
            "module": "cake_dsa_h64_train_61a1676e9d6ac1755272",
            "sources": [
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_61a1676e9d6ac1755272_kernel.cu",
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_61a1676e9d6ac1755272_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "q_latent"],
                ["tma_buffer", "q_rope"],
                ["tma_buffer", "dout"],
                ["tma_buffer", "dq_latent"],
                ["tma_buffer", "dq_rope"],
                ["tma_buffer", "kv_latent"],
                ["tma_buffer", "k_rope"],
                ["buffer", "lse"],
                ["buffer", "delta"],
                ["buffer", "indices"],
                ["buffer", "topk_length"],
                ["buffer", "dkv_f32"],
                ["buffer", "dkr_f32"],
                ["parameter", "num_queries"],
                ["parameter", "num_kv"],
                ["parameter", "topk"],
                ["parameter", "idx_stride"],
                ["parameter", "indices_offset"],
                ["parameter", "has_topk_length"],
                ["parameter", "token_base"],
                ["parameter", "token_step"],
                ["parameter", "scale_log2"],
                ["parameter", "sm_scale"],
                ["parameter", "pass_lo"],
                ["parameter", "pass_hi"],
                ["parameter", "dq_mode"],
                ["buffer", "dq_partial"],
                ["buffer", "key_scratch"],
                ["buffer", "pass_counts"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "9ba5308137a0d9577a01334499609a081498e05dd030095dff6f324a96904508",
            "tma_workspace_bytes": 896,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_compact": {
            "module": "cake_dsa_h64_train_1914091710f34400bc72",
            "sources": [
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_1914091710f34400bc72_kernel.cu",
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_1914091710f34400bc72_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "indices"],
                ["buffer", "topk_length"],
                ["buffer", "key_scratch"],
                ["buffer", "pass_counts"],
                ["parameter", "num_tokens"],
                ["parameter", "topk"],
                ["parameter", "idx_stride"],
                ["parameter", "indices_offset"],
                ["parameter", "has_topk_length"],
                ["parameter", "token_base"],
                ["parameter", "token_step"],
                ["parameter", "pass_lo"],
                ["parameter", "pass_hi"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "23e067b08e60a6d34a9d73d81cd4db45f4d2145a8506c32965cd1b8fd2135e5c",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_queries/4", 1, 1],
            "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main_pass": {
            "module": "cake_dsa_h64_train_d771b00c49e3bdfa1874",
            "sources": [
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_d771b00c49e3bdfa1874_kernel.cu",
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_d771b00c49e3bdfa1874_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "q_latent"],
                ["tma_buffer", "q_rope"],
                ["tma_buffer", "dout"],
                ["tma_buffer", "dq_latent"],
                ["tma_buffer", "dq_rope"],
                ["tma_buffer", "kv_latent"],
                ["tma_buffer", "k_rope"],
                ["buffer", "lse"],
                ["buffer", "delta"],
                ["buffer", "indices"],
                ["buffer", "topk_length"],
                ["buffer", "dkv_f32"],
                ["buffer", "dkr_f32"],
                ["parameter", "num_queries"],
                ["parameter", "num_kv"],
                ["parameter", "topk"],
                ["parameter", "idx_stride"],
                ["parameter", "indices_offset"],
                ["parameter", "has_topk_length"],
                ["parameter", "token_base"],
                ["parameter", "token_step"],
                ["parameter", "scale_log2"],
                ["parameter", "sm_scale"],
                ["parameter", "pass_lo"],
                ["parameter", "pass_hi"],
                ["parameter", "dq_mode"],
                ["buffer", "dq_partial"],
                ["buffer", "key_scratch"],
                ["buffer", "pass_counts"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "07d1f37526462a355b4a86e81e58ab701ce5d66cb8a009b8d42af497d0f7d28f",
            "tma_workspace_bytes": 896,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_cast": {
            "module": "cake_dsa_h64_train_d8b3954b05a7afab575c",
            "sources": [
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_d8b3954b05a7afab575c_kernel.cu",
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_d8b3954b05a7afab575c_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "src_latent"],
                ["buffer", "src_rope"],
                ["buffer", "dst_latent"],
                ["buffer", "dst_rope"],
                ["buffer", "dst_latent_f32"],
                ["buffer", "dst_rope_f32"],
                ["parameter", "latent_groups"],
                ["parameter", "rope_groups"],
                ["parameter", "out_f32"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "7f909367dba066afcb8eb265f4d47a735ceddd4759622ec653fa8c140fb65e3c",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_kv*18/256", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "cdfb9f0c88dd62be34b14c23960f80b40f55e1fd039e304b2b0becd8938d323d",
    },
}

# Kernel stages of one training step, in launch order.  ``fwd`` writes the
# output, the natural-log LSE and the output residual; ``bwd_delta`` forms
# delta = rowsum(dO * (O + O_lo)); ``bwd_main`` recomputes P, accumulates the
# FP32 dK/dV partials and, when ``bwd_dq`` is absent, also dQ; ``bwd_dq`` is
# the separate dQ pass of a two-pass backward; ``bwd_compact`` and
# ``bwd_main_pass`` are the key-range-pass form of the main stage (per pass:
# compact each row's keys of the pass range, then the main stage over that
# range, carrying dQ through an FP32 partial) that the host selects instead of
# ``bwd_main`` when the record's ``key_pass_policy`` yields more than one
# pass; ``bwd_cast`` turns the FP32 dK/dV accumulators into natural-layout
# BF16 (or FP32) outputs.  A record registers the subset its program uses
# (``bwd_dq``, the pass stages and ``bwd_cast`` are optional; a ``permuted``
# accumulator layout requires ``bwd_cast``).
STAGES = (
    "fwd",
    "bwd_delta",
    "bwd_main",
    "bwd_dq",
    "bwd_compact",
    "bwd_main_pass",
    "bwd_cast",
)
FORWARD_STAGES = ("fwd",)
BACKWARD_STAGES = (
    "bwd_delta",
    "bwd_main",
    "bwd_dq",
    "bwd_compact",
    "bwd_main_pass",
    "bwd_cast",
)
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def toolchain_supports(arch: str) -> bool:
    """Can the nvcc this checkout invokes emit ``arch``?  (SM100 / SM103 only.)"""
    return arch in ARCH_NVCC_FLAGS


def select_module(arch: str) -> str:
    """Return the registered module name for ``arch``."""
    names = [name for name, record in MODULES.items() if record["arch"] == arch]
    if len(names) > 1:
        raise NotImplementedError(
            f"{arch} registers more than one DSA training program: {names}"
        )
    if not names:
        raise NotImplementedError(
            f"The generated DSA sparse-attention training program for {arch} is "
            "not registered in this checkout yet (see flashinfer-ai/flashinfer#5657)"
        )
    return names[0]


def registered_stages(name: str) -> tuple[str, ...]:
    """Stages a record registers, in launch order."""
    present = tuple(stage for stage in STAGES if stage in MODULES[name])
    declared = tuple(MODULES[name].get("stages", present))
    if tuple(s for s in STAGES if s in declared) != present:
        raise ValueError(
            f"registry record {name!r} declares stages {declared} but carries {present}"
        )
    return present


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
def gen_cake_dsa_train_module(name: str, stage: str):
    record = MODULES[name]
    if not toolchain_supports(record["arch"]):
        raise RuntimeError(
            f"generated DSA training program {name!r} targets {record['arch']}, "
            "which this checkout cannot compile"
        )
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
def load_cake_dsa_train_module(name: str, stage: str):
    return gen_cake_dsa_train_module(name, stage).build_and_load()
