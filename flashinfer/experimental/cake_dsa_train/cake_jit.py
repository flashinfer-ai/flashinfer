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
from ...jit.core import (
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

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
            "module": "cake_dsa_h64_train_1dabf72b2a2d213a3788",
            "sources": [
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_1dabf72b2a2d213a3788_kernel.cu",
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_1dabf72b2a2d213a3788_binding.cu",
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
            "closure_sha256": "1a2c4599fd7adc245a738ff14a4e4cf123c415ac7fc53ca2603084587bcb77f0",
            "tma_workspace_bytes": 512,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_delta": {
            "module": "cake_dsa_h64_train_8e77ef15129fbcc59a67",
            "sources": [
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_8e77ef15129fbcc59a67_kernel.cu",
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_8e77ef15129fbcc59a67_binding.cu",
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
            "closure_sha256": "fa4f8f9caa1231a3ed455b4aa0fb89d74baeac9538caac8d4d02d1263e152c99",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_queries*8", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main": {
            "module": "cake_dsa_h64_train_111f9507bb9e6afbac20",
            "sources": [
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_111f9507bb9e6afbac20_kernel.cu",
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_111f9507bb9e6afbac20_binding.cu",
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
            "closure_sha256": "deda49ead38b900f280dbb4156b0ce8c5b35a3ee0edc12f7472f7570ed0d33f3",
            "tma_workspace_bytes": 896,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_compact": {
            "module": "cake_dsa_h64_train_7ebdf91b84ccbba0b3d9",
            "sources": [
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_7ebdf91b84ccbba0b3d9_kernel.cu",
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_7ebdf91b84ccbba0b3d9_binding.cu",
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
            "closure_sha256": "ea11cd287d54f7a857fc755a6551708ca01f8dd8420934ce9954cd52b696ff00",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_queries/4", 1, 1],
            "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main_pass": {
            "module": "cake_dsa_h64_train_7c425c49edd1a50f29c7",
            "sources": [
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_7c425c49edd1a50f29c7_kernel.cu",
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_7c425c49edd1a50f29c7_binding.cu",
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
            "closure_sha256": "8413adff8d6f5cfd60bd99b11a401e57dd22789fc1c55c145f34a61b8989e044",
            "tma_workspace_bytes": 896,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_cast": {
            "module": "cake_dsa_h64_train_4d4c2e668b233b795b94",
            "sources": [
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_4d4c2e668b233b795b94_kernel.cu",
                "cake_dsa_h64_train/sm_100a/cake_dsa_h64_train_4d4c2e668b233b795b94_binding.cu",
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
                ["buffer", "dst_packed"],
                ["parameter", "dst_row_stride"],
                ["buffer", "dst_map"],
                ["parameter", "has_dst_map"],
                ["parameter", "accumulate"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "32d7947fea61250ffe6fe5cd4a968994a4e251da5c8dccd6f88ad05ef073f0d7",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_kv*18/256", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "49c06b6fa18691f2b7d7320bdc61776f5b6db573c3954c1e5f5d0e3fde781305",
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
            "module": "cake_dsa_h64_train_0b6054bf52663f80236b",
            "sources": [
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_0b6054bf52663f80236b_kernel.cu",
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_0b6054bf52663f80236b_binding.cu",
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
            "closure_sha256": "e02c5e09bab6a742a1dacd1a9ecec9a6b45d8d3cf65c9304572e4a4a1e40279c",
            "tma_workspace_bytes": 512,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_delta": {
            "module": "cake_dsa_h64_train_4acab9732281ac23243a",
            "sources": [
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_4acab9732281ac23243a_kernel.cu",
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_4acab9732281ac23243a_binding.cu",
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
            "closure_sha256": "59aab7b4979e3092f16dc20df6201c351723ffc7353b6f8aafb2ac4e52779c2e",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_queries*8", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main": {
            "module": "cake_dsa_h64_train_9785f85d8c8d2acae461",
            "sources": [
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_9785f85d8c8d2acae461_kernel.cu",
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_9785f85d8c8d2acae461_binding.cu",
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
            "closure_sha256": "37917f67ddd49b0324f912b15eca2d9474d8276aa6b324b900d6832b90b4916b",
            "tma_workspace_bytes": 896,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_compact": {
            "module": "cake_dsa_h64_train_a84f0662a82028c7dcfe",
            "sources": [
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_a84f0662a82028c7dcfe_kernel.cu",
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_a84f0662a82028c7dcfe_binding.cu",
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
            "closure_sha256": "12e69d3d4b7f7f4198826bfc6d422dcbb2378d8a78e43f28f27a461dec299c78",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_queries/4", 1, 1],
            "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main_pass": {
            "module": "cake_dsa_h64_train_5b1accdf3cbb4915bed7",
            "sources": [
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_5b1accdf3cbb4915bed7_kernel.cu",
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_5b1accdf3cbb4915bed7_binding.cu",
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
            "closure_sha256": "4dc194322e044873b66669f3ee109cf14eb9be02c197a3cd7d49076b0c16ae01",
            "tma_workspace_bytes": 896,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_cast": {
            "module": "cake_dsa_h64_train_8528ac191367cda50727",
            "sources": [
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_8528ac191367cda50727_kernel.cu",
                "cake_dsa_h64_train/sm_103a/cake_dsa_h64_train_8528ac191367cda50727_binding.cu",
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
                ["buffer", "dst_packed"],
                ["parameter", "dst_row_stride"],
                ["buffer", "dst_map"],
                ["parameter", "has_dst_map"],
                ["parameter", "accumulate"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "5a15967cd56d128f4190e91acb80c981f3049e01c5014618ee062d144457f102",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_kv*18/256", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "b0e6a5dd04e551a9001681c0b261ad7605ae393c89346e27f38e6260bc5840ec",
    },
    "cake_dsa_h64_train_sm_107a": {
        "arch": "sm_107a",
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
            "module": "cake_dsa_h64_train_df8c520551bb6f16858a",
            "sources": [
                "cake_dsa_h64_train/sm_107a/cake_dsa_h64_train_df8c520551bb6f16858a_kernel.cu",
                "cake_dsa_h64_train/sm_107a/cake_dsa_h64_train_df8c520551bb6f16858a_binding.cu",
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
            "closure_sha256": "d50a0201725d184dcf4598f6b9598c9fc7f9f46741939482cb8f50dc9975bfcc",
            "tma_workspace_bytes": 512,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_delta": {
            "module": "cake_dsa_h64_train_7d250ffc8cda4a421383",
            "sources": [
                "cake_dsa_h64_train/sm_107a/cake_dsa_h64_train_7d250ffc8cda4a421383_kernel.cu",
                "cake_dsa_h64_train/sm_107a/cake_dsa_h64_train_7d250ffc8cda4a421383_binding.cu",
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
            "closure_sha256": "7a467882294501fd85818dcfe4f059b02adc0648ecb4e31175df0cbbd57742b0",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_queries*8", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main": {
            "module": "cake_dsa_h64_train_c224ca9fa942cd5a5f94",
            "sources": [
                "cake_dsa_h64_train/sm_107a/cake_dsa_h64_train_c224ca9fa942cd5a5f94_kernel.cu",
                "cake_dsa_h64_train/sm_107a/cake_dsa_h64_train_c224ca9fa942cd5a5f94_binding.cu",
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
            "closure_sha256": "ca882afdf9f6354199bce4b1db095dc75d00ece986eec0d3aa703ec25b407a62",
            "tma_workspace_bytes": 896,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_compact": {
            "module": "cake_dsa_h64_train_e753deec417a3a54b4a5",
            "sources": [
                "cake_dsa_h64_train/sm_107a/cake_dsa_h64_train_e753deec417a3a54b4a5_kernel.cu",
                "cake_dsa_h64_train/sm_107a/cake_dsa_h64_train_e753deec417a3a54b4a5_binding.cu",
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
            "closure_sha256": "86e1e065bbe548b9f5e4cc303fe7e29b94462f972730bf3765ec76a8d694ee9d",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_queries/4", 1, 1],
            "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main_pass": {
            "module": "cake_dsa_h64_train_3d0c37428e7991ab4c07",
            "sources": [
                "cake_dsa_h64_train/sm_107a/cake_dsa_h64_train_3d0c37428e7991ab4c07_kernel.cu",
                "cake_dsa_h64_train/sm_107a/cake_dsa_h64_train_3d0c37428e7991ab4c07_binding.cu",
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
            "closure_sha256": "75d8d1cfc9c4f08427f60cf31a10db7aaf5064b4651fb331d7bc494728ae60f8",
            "tma_workspace_bytes": 896,
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_cast": {
            "module": "cake_dsa_h64_train_464e7007fcb470954f0c",
            "sources": [
                "cake_dsa_h64_train/sm_107a/cake_dsa_h64_train_464e7007fcb470954f0c_kernel.cu",
                "cake_dsa_h64_train/sm_107a/cake_dsa_h64_train_464e7007fcb470954f0c_binding.cu",
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
                ["buffer", "dst_packed"],
                ["parameter", "dst_row_stride"],
                ["buffer", "dst_map"],
                ["parameter", "has_dst_map"],
                ["parameter", "accumulate"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "c3b787334a27e2ba71ec1dc3a702c80e84624162ad28dc4ec17d496cf35b45c1",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["num_kv*18/256", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "21c2176c40e16f286d6c1be63baf4e804ccb2354434e7150393fcd7c619d8842",
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
# BF16 (or FP32) outputs or, when its argument plan declares the
# packed-accumulate operands (``cake_backend.CAST_ACCUMULATE_OPERANDS``), adds
# them into a caller-provided packed FP32 buffer with an optional
# destination-row map.  A record registers the subset its program uses
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
    "sm_107a": sm107a_nvcc_flags,
}


def toolchain_supports(arch: str) -> bool:
    """Can the nvcc this checkout invokes emit ``arch``?

    SM100 / SM103 compile with any CUDA 12.8+ toolkit; ``sm_107a`` needs an nvcc that
    lists ``compute_107`` (public CUDA 13.x toolkits do not), so it is probed once.
    """
    if arch not in ARCH_NVCC_FLAGS:
        return False
    if arch == "sm_107a":
        from ...compilation_context import _nvcc_supports_sm107

        return _nvcc_supports_sm107()
    return True


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
