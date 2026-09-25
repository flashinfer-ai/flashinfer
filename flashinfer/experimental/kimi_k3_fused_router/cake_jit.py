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
import os
import re
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags
from ...jit.utils import write_if_different

# Explicit target-owned registration of the generated program.  One record per
# (architecture, dispatch arm, block_m); each record carries the single
# physical stage (the fused router kernel of that arm) with its own
# translation units, compile flags, FFI entry, argument plan and launch
# resources (block, cluster, cooperative flag, dynamic shared memory).
# Populated verbatim by the generated-program export; do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_kimi_k3_fused_router_g_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "G",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_71c16c145876dbdbd3ba",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_71c16c145876dbdbd3ba_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_71c16c145876dbdbd3ba_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "5bb753ff9acdcdba23701a7a77f853039d01ee2fec32d4f465fb90b045d7e2f8",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "5bb753ff9acdcdba23701a7a77f853039d01ee2fec32d4f465fb90b045d7e2f8",
    },
    "cake_kimi_k3_fused_router_g_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "G",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_42ceb12a571870c8e0cf",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_42ceb12a571870c8e0cf_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_42ceb12a571870c8e0cf_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "a5678c46e94c55513bb1a8eddd142fa78cf724eb6ea8f7ccbc05ab6f57f6ae0e",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "a5678c46e94c55513bb1a8eddd142fa78cf724eb6ea8f7ccbc05ab6f57f6ae0e",
    },
    "cake_kimi_k3_fused_router_g_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "G",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_e55bb5262eab36776468",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_e55bb5262eab36776468_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_e55bb5262eab36776468_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "29a2255fa17d403e3146a0e51b2533f4f6e2cf069cc001871fb7ed604a08db81",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "29a2255fa17d403e3146a0e51b2533f4f6e2cf069cc001871fb7ed604a08db81",
    },
    "cake_kimi_k3_fused_router_g_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "G",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_814ffc82d3bda169588d",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_814ffc82d3bda169588d_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_814ffc82d3bda169588d_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "265065c8499af5a1698450be52193409d65f76e80c1d5360ee43717ae392695b",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "265065c8499af5a1698450be52193409d65f76e80c1d5360ee43717ae392695b",
    },
    "cake_kimi_k3_fused_router_l_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "L",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_3b29c29e5355b43c4121",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_3b29c29e5355b43c4121_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_3b29c29e5355b43c4121_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "d009e88d3dd2392fbf6bfe61fad24767dba8f567e6d2c7a83e091469b9ff4be1",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "d009e88d3dd2392fbf6bfe61fad24767dba8f567e6d2c7a83e091469b9ff4be1",
    },
    "cake_kimi_k3_fused_router_l_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "L",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_4e77e090a7664d23769a",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_4e77e090a7664d23769a_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_4e77e090a7664d23769a_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "094e59098e08c99fdd68ce0e4c805bc9c1753c6f0ea7f33d05580091e5832118",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "094e59098e08c99fdd68ce0e4c805bc9c1753c6f0ea7f33d05580091e5832118",
    },
    "cake_kimi_k3_fused_router_l_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "L",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_70906f4273411ba316cd",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_70906f4273411ba316cd_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_70906f4273411ba316cd_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "b177e1f733ca93a6285f66c8ed84a0d65440458c3531236f71785b626cb46985",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "b177e1f733ca93a6285f66c8ed84a0d65440458c3531236f71785b626cb46985",
    },
    "cake_kimi_k3_fused_router_l_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "L",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_d4ec14b2e4e5bcb184ce",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_d4ec14b2e4e5bcb184ce_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_d4ec14b2e4e5bcb184ce_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "ef32591c1f626031a84aabe04d633d605865561de14251c36fb7a33c99dee777",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "ef32591c1f626031a84aabe04d633d605865561de14251c36fb7a33c99dee777",
    },
    "cake_kimi_k3_fused_router_lc2_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_75201632adc2f29843c0",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_75201632adc2f29843c0_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_75201632adc2f29843c0_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "87758cff0d28f31b72084f653411093859491b724cd8af74327dc84823fe1226",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "87758cff0d28f31b72084f653411093859491b724cd8af74327dc84823fe1226",
    },
    "cake_kimi_k3_fused_router_lc2_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_1518abeded240fc8242e",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_1518abeded240fc8242e_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_1518abeded240fc8242e_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "d74d361ad5327346e7a27c6332c2c90737a86f30b609cb1dc61e6909ebb1a74b",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "d74d361ad5327346e7a27c6332c2c90737a86f30b609cb1dc61e6909ebb1a74b",
    },
    "cake_kimi_k3_fused_router_lc2_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_aa68b32950dfab8b58bb",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_aa68b32950dfab8b58bb_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_aa68b32950dfab8b58bb_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "d4e93f95b8e675d08b25f11c62d35687defe4f27d2bb1a24a07381c3907fd012",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "d4e93f95b8e675d08b25f11c62d35687defe4f27d2bb1a24a07381c3907fd012",
    },
    "cake_kimi_k3_fused_router_lc2_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_89297ef561224339c50e",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_89297ef561224339c50e_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_89297ef561224339c50e_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "18b735caac7875cfa911f932998400c2fea4bb697378c0e0b565ae0b4163b4b6",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "18b735caac7875cfa911f932998400c2fea4bb697378c0e0b565ae0b4163b4b6",
    },
    "cake_kimi_k3_fused_router_lc4_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_e69978ea3491a9b2c0bb",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_e69978ea3491a9b2c0bb_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_e69978ea3491a9b2c0bb_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "c8cbf946d1d0d20e634e83d57dd937b58e5194858fe114894e28b91559cd1aaf",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "c8cbf946d1d0d20e634e83d57dd937b58e5194858fe114894e28b91559cd1aaf",
    },
    "cake_kimi_k3_fused_router_lc4_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_e81b263b3503a6162f88",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_e81b263b3503a6162f88_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_e81b263b3503a6162f88_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "357698d29bb63d73a80b119ef638ce9ecf5ac2f81f129b2f536c4fbd2f79c7ea",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "357698d29bb63d73a80b119ef638ce9ecf5ac2f81f129b2f536c4fbd2f79c7ea",
    },
    "cake_kimi_k3_fused_router_lc4_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_adea05c4dae98eecc015",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_adea05c4dae98eecc015_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_adea05c4dae98eecc015_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "3b4c9e325c5dc6fc23fdb49ce609fa663f792d2586cb6d644223cd08989a295b",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "3b4c9e325c5dc6fc23fdb49ce609fa663f792d2586cb6d644223cd08989a295b",
    },
    "cake_kimi_k3_fused_router_lc4_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_09ce077215664f532771",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_09ce077215664f532771_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_09ce077215664f532771_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "0cd8ef64638bd20845c40226b1b581393fb63a26354edd5a15c657a6dee13f07",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "0cd8ef64638bd20845c40226b1b581393fb63a26354edd5a15c657a6dee13f07",
    },
    "cake_kimi_k3_fused_router_lc8_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_627bbffb55c29ffdd890",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_627bbffb55c29ffdd890_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_627bbffb55c29ffdd890_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "a4a77e6c91ec91fa7988bdbbe2bd28ad842c67764cc70fca0df7a256ba7789f0",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "a4a77e6c91ec91fa7988bdbbe2bd28ad842c67764cc70fca0df7a256ba7789f0",
    },
    "cake_kimi_k3_fused_router_lc8_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_a78c494e09ae72be901a",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_a78c494e09ae72be901a_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_a78c494e09ae72be901a_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "27d178cf8ebe24d3f956ab9f2f9eaa703d9f5ff875fa8633e77fef2feb7f0916",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "27d178cf8ebe24d3f956ab9f2f9eaa703d9f5ff875fa8633e77fef2feb7f0916",
    },
    "cake_kimi_k3_fused_router_lc8_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_0487353c3fdc5b5014ef",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_0487353c3fdc5b5014ef_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_0487353c3fdc5b5014ef_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "1ba00260b89865f5d465f72051b4733d014d5b783d1152d6068d2669aca9e701",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "1ba00260b89865f5d465f72051b4733d014d5b783d1152d6068d2669aca9e701",
    },
    "cake_kimi_k3_fused_router_lc8_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_5d9ee5461b192b265e46",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_5d9ee5461b192b265e46_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_5d9ee5461b192b265e46_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "58763fedb99bc74109a3601f89ac62afbdaed235b12723ae0424966d470bb645",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "58763fedb99bc74109a3601f89ac62afbdaed235b12723ae0424966d470bb645",
    },
    "cake_kimi_k3_fused_router_m_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "M",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_a4c4268c7580b6bd00ef",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_a4c4268c7580b6bd00ef_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_a4c4268c7580b6bd00ef_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "0bb1d3bfc05dada46e973fa68556ec65f0b43bcfb6e7948287851c6708fca8bb",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "0bb1d3bfc05dada46e973fa68556ec65f0b43bcfb6e7948287851c6708fca8bb",
    },
    "cake_kimi_k3_fused_router_m_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "M",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_fb1fdb58f546377ef3da",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_fb1fdb58f546377ef3da_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_fb1fdb58f546377ef3da_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "a3ea6a3897ae889993a693193c640ff41473bbbe4e38ef58e02df0096d77b5cf",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "a3ea6a3897ae889993a693193c640ff41473bbbe4e38ef58e02df0096d77b5cf",
    },
    "cake_kimi_k3_fused_router_m_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "M",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_f32e3d7beaca76dfa149",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_f32e3d7beaca76dfa149_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_f32e3d7beaca76dfa149_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "f1cd455b69301f3e469d15040e8a4455e9669a8d8abdb33b0ac7721f2e9ce780",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "f1cd455b69301f3e469d15040e8a4455e9669a8d8abdb33b0ac7721f2e9ce780",
    },
    "cake_kimi_k3_fused_router_m_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "M",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_5ea8397393c26c013b6c",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_5ea8397393c26c013b6c_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_5ea8397393c26c013b6c_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "044c4688d7bc3f2dfdead90e11f9ddf74e5e817cbaec1e77408505f5613b85c2",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "044c4688d7bc3f2dfdead90e11f9ddf74e5e817cbaec1e77408505f5613b85c2",
    },
    "cake_kimi_k3_fused_router_q4s_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "Q4S",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_537eae853235a32e4997",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_537eae853235a32e4997_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_537eae853235a32e4997_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "239a697795a4151ec345c7339680349cdee73bd0829c0a681e46276853127b8d",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "239a697795a4151ec345c7339680349cdee73bd0829c0a681e46276853127b8d",
    },
    "cake_kimi_k3_fused_router_q4s_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "Q4S",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_105e3452054c9384311c",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_105e3452054c9384311c_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_105e3452054c9384311c_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "a268ed692391f9bb43a83d5f43fc8cb26354b9ffbfc322cbaa67a6376e00e8a5",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "a268ed692391f9bb43a83d5f43fc8cb26354b9ffbfc322cbaa67a6376e00e8a5",
    },
    "cake_kimi_k3_fused_router_q4s_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "Q4S",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_ee744df0c7f354ab0a82",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_ee744df0c7f354ab0a82_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_ee744df0c7f354ab0a82_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "dd0e264ad00fe445effd47bafed79c16e443b8913d77de7998141d4adcf92d8e",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "dd0e264ad00fe445effd47bafed79c16e443b8913d77de7998141d4adcf92d8e",
    },
    "cake_kimi_k3_fused_router_q4s_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "Q4S",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_7e7638c56c92f730bfdd",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_7e7638c56c92f730bfdd_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_7e7638c56c92f730bfdd_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "logits"],
                ["buffer", "bias"],
                ["buffer", "topk_weights"],
                ["buffer", "topk_ids"],
                ["buffer", "sorted_token_ids"],
                ["buffer", "expert_ids"],
                ["buffer", "num_tokens_post_padded"],
                ["buffer", "expert_counts"],
                ["buffer", "expert_offsets"],
                ["buffer", "expert_scatter_offsets"],
                ["parameter", "M"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "00a56fefbbcd5fb5100193841d5ee2c9d0432de63bb4f16cce244312be63c0ed",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "00a56fefbbcd5fb5100193841d5ee2c9d0432de63bb4f16cce244312be63c0ed",
    },
}

STAGES = ("main",)
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}

# Dispatch arms of the generated program.  The per-shape route tables in
# ``cake_backend`` name one arm per (architecture, num_tokens, block_m):
#   L   : one-join plan builder, num_tokens <= 512, at least 128 CTAs launched
#   LC  : one cluster of num_tokens CTAs (2, 4 or 8) exchanging selected ids
#         through distributed shared memory; one kernel per row count,
#         non-cooperative cluster launch
#   M   : one-join bitmap plan builder, num_tokens = 256
#   Q4S : arm M's cp.async ID stream in 4-CTA clusters (cooperative cluster
#         launch), compiled with __launch_bounds__(224, 4): four CTAs per SM,
#         512 <= num_tokens <= 2048
#   G   : two-join persistent kernel for the largest batches, compiled with
#         per-architecture launch bounds (4 CTAs/SM on SM100, 6 on SM103)
ARMS = ("L", "LC", "M", "Q4S", "G")
# Arms registered per row count (one kernel per num_tokens); every other arm
# registers one module per (arch, block_m) and serves all of its rows.
PER_ROW_COUNT_ARMS = ("LC",)


def module_num_tokens(arm: str, num_tokens: int):
    """``num_tokens`` key of the module serving ``arm`` (``None`` for shared kernels)."""
    return int(num_tokens) if arm in PER_ROW_COUNT_ARMS else None


def select_module(arch: str, arm: str, block_m: int, num_tokens=None) -> str:
    """Registered module name for ``arch``, dispatch ``arm``, ``block_m`` (and, for
    per-row-count arms, ``num_tokens``)."""
    for name, record in MODULES.items():
        if (
            record["arch"] == arch
            and record["arm"] == arm
            and int(record["block_m"]) == int(block_m)
            and record.get("num_tokens") == num_tokens
        ):
            return name
    rows = "" if num_tokens is None else f", num_tokens {num_tokens}"
    raise NotImplementedError(
        f"The generated Kimi-K3 fused router program (arm {arm}, block_m {block_m}{rows}) "
        f"for {arch} is not registered in this checkout yet"
    )


def registered_programs(arch: str) -> set[tuple[str, int, Any]]:
    """``{(arm, block_m, num_tokens_or_None)}`` registered for ``arch``."""
    return {
        (record["arm"], int(record["block_m"]), record.get("num_tokens"))
        for record in MODULES.values()
        if record["arch"] == arch
    }


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


# ---------------------------------------------------------------------------
# Occupancy helper for cluster-launched arms
# ---------------------------------------------------------------------------
#
# The generated binding launches the kernel (cooperative and cluster attributes
# included) but exposes no occupancy query.  Arm Q's persistent grid must be
# bounded by the number of co-resident 4-CTA clusters the driver admits (GPC
# placement, not just CTAs per SM), so this package adds one small translation
# unit per cluster module that forwards the kernel's host stub to
# ``cudaOccupancyMaxActiveClusters`` with the same one-cluster configuration the
# launch uses (block, cluster dims, dynamic shared memory, cooperative flag).

_KERNEL_DECLARATION = re.compile(
    r'^extern "C" __global__ void (?P<symbol>kernel_[A-Za-z0-9_]+)\((?P<params>[^;]*)\);\s*$',
    re.MULTILINE,
)

_OCCUPANCY_TEMPLATE = """\
// Occupancy query for the generated cluster kernel {symbol}; rendered by
// cake_jit.py from the kernel declaration of the generated binding.
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

#include <cstdint>

{declaration}

namespace {{

int64_t MaxActiveClusters(int64_t device, int64_t block_x, int64_t block_y, int64_t block_z,
                          int64_t cluster_x, int64_t cluster_y, int64_t cluster_z,
                          int64_t dynamic_smem_bytes, bool cooperative) {{
  int previous_device = -1;
  TVM_FFI_CHECK(cudaGetDevice(&previous_device) == cudaSuccess, RuntimeError)
      << "cudaGetDevice failed";
  if (static_cast<int64_t>(previous_device) != device) {{
    TVM_FFI_CHECK(cudaSetDevice(static_cast<int>(device)) == cudaSuccess, RuntimeError)
        << "cudaSetDevice failed";
  }}
  const void* function = reinterpret_cast<const void*>(&{symbol});
  if (dynamic_smem_bytes > 48 * 1024) {{
    TVM_FFI_CHECK(cudaFuncSetAttribute(function, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(dynamic_smem_bytes)) == cudaSuccess,
                  RuntimeError)
        << "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed for {symbol}";
  }}
  cudaLaunchAttribute attrs[2]{{}};
  int n = 0;
  attrs[n].id = cudaLaunchAttributeClusterDimension;
  attrs[n].val.clusterDim.x = static_cast<unsigned>(cluster_x);
  attrs[n].val.clusterDim.y = static_cast<unsigned>(cluster_y);
  attrs[n].val.clusterDim.z = static_cast<unsigned>(cluster_z);
  ++n;
  if (cooperative) {{
    attrs[n].id = cudaLaunchAttributeCooperative;
    attrs[n].val.cooperative = 1;
    ++n;
  }}
  cudaLaunchConfig_t config{{}};
  config.gridDim = dim3(static_cast<unsigned>(cluster_x), static_cast<unsigned>(cluster_y),
                        static_cast<unsigned>(cluster_z));
  config.blockDim = dim3(static_cast<unsigned>(block_x), static_cast<unsigned>(block_y),
                         static_cast<unsigned>(block_z));
  config.dynamicSmemBytes = static_cast<size_t>(dynamic_smem_bytes);
  config.attrs = attrs;
  config.numAttrs = static_cast<unsigned>(n);
  int clusters = 0;
  cudaError_t status = cudaOccupancyMaxActiveClusters(&clusters, function, &config);
  if (static_cast<int64_t>(previous_device) != device) {{
    cudaSetDevice(previous_device);
  }}
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaOccupancyMaxActiveClusters for {symbol} failed: " << cudaGetErrorString(status);
  return static_cast<int64_t>(clusters);
}}

}}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(max_active_clusters, MaxActiveClusters);
"""


def _kernel_declaration(binding_path: Path) -> tuple[str, str]:
    """``(symbol, declaration)`` of the kernel declared by a generated binding."""
    matches = _KERNEL_DECLARATION.findall(binding_path.read_text())
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one generated kernel declaration in {binding_path}, "
            f"found {len(matches)}"
        )
    symbol, params = matches[0]
    return symbol, f'extern "C" __global__ void {symbol}({params});'


def uses_cluster_launch(record: dict[str, Any]) -> bool:
    launch = record["main"]["launch"]
    return any(int(dim) > 1 for dim in launch["cluster"])


def _occupancy_source(spec_name: str, binding_path: Path) -> Path:
    symbol, declaration = _kernel_declaration(binding_path)
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / spec_name
    os.makedirs(gen_directory, exist_ok=True)
    path = gen_directory / f"{symbol}_occupancy.cu"
    write_if_different(
        path, _OCCUPANCY_TEMPLATE.format(symbol=symbol, declaration=declaration)
    )
    return path


@functools.cache
def gen_kimi_k3_fused_router_module(name: str, stage: str):
    record = MODULES[name]
    physical = record[stage]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in physical["sources"]]
    spec_name = f"{name}_{stage}_" + physical["closure_sha256"][:20]
    if uses_cluster_launch(record):
        binding = next(path for path in sources if path.name.endswith("_binding.cu"))
        sources.append(_occupancy_source(spec_name, binding))
    return gen_jit_spec(
        name=spec_name,
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
def load_kimi_k3_fused_router_module(name: str, stage: str):
    return gen_kimi_k3_fused_router_module(name, stage).build_and_load()
