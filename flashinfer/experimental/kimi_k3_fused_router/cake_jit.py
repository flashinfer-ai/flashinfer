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
    "cake_kimi_k3_fused_router_gw_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "GW",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_f79fc5268e38e3297512",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_f79fc5268e38e3297512_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_f79fc5268e38e3297512_binding.cu",
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
            "closure_sha256": "cff32a7d9bc91b556065c68bb2500b3c669b49fe87a764c2c0eb9d2a52891749",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "cff32a7d9bc91b556065c68bb2500b3c669b49fe87a764c2c0eb9d2a52891749",
    },
    "cake_kimi_k3_fused_router_gw_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "GW",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_d502b6a6bbcdd0a041bb",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_d502b6a6bbcdd0a041bb_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_d502b6a6bbcdd0a041bb_binding.cu",
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
            "closure_sha256": "0832e4f0d7dc4534deddc0798d51b56da536ed24665fa61807c70adaf8fa8fbf",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "0832e4f0d7dc4534deddc0798d51b56da536ed24665fa61807c70adaf8fa8fbf",
    },
    "cake_kimi_k3_fused_router_gw_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "GW",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_2954072a99c6fbe78d85",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_2954072a99c6fbe78d85_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_2954072a99c6fbe78d85_binding.cu",
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
            "closure_sha256": "bee9b84c46985d3821fa8cbb73cdecf7af58d9d60ca6f6ccbd7dfb21d0c22a6e",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "bee9b84c46985d3821fa8cbb73cdecf7af58d9d60ca6f6ccbd7dfb21d0c22a6e",
    },
    "cake_kimi_k3_fused_router_gw_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "GW",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_d8044a2578a033619aea",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_d8044a2578a033619aea_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_d8044a2578a033619aea_binding.cu",
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
            "closure_sha256": "36a41d2ea8d4116d9a752c705bd34e02ef8a7c68150d779213b8c63cb4918e78",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "36a41d2ea8d4116d9a752c705bd34e02ef8a7c68150d779213b8c63cb4918e78",
    },
    "cake_kimi_k3_fused_router_l_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "L",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_7c6dcf483657e6e5d1fe",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_7c6dcf483657e6e5d1fe_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_7c6dcf483657e6e5d1fe_binding.cu",
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
            "closure_sha256": "6f7dc2921a555c48fd77969b3504526d7ca4283afc753238ea33f3933ca36aec",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "6f7dc2921a555c48fd77969b3504526d7ca4283afc753238ea33f3933ca36aec",
    },
    "cake_kimi_k3_fused_router_l_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "L",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_a358e19e0ba01c6ec6e6",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_a358e19e0ba01c6ec6e6_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_a358e19e0ba01c6ec6e6_binding.cu",
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
            "closure_sha256": "c0ae271b35fc07d87cb7e3e9fd54a16639d1d2bc37667fc2ea59a49212062b0b",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "c0ae271b35fc07d87cb7e3e9fd54a16639d1d2bc37667fc2ea59a49212062b0b",
    },
    "cake_kimi_k3_fused_router_l_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "L",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_49461d5ab5829e2d3244",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_49461d5ab5829e2d3244_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_49461d5ab5829e2d3244_binding.cu",
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
            "closure_sha256": "8b0ab3bed386cc75e53bad0cb9452f57596ff817907d9f5aa4f6c81c9050abbd",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "8b0ab3bed386cc75e53bad0cb9452f57596ff817907d9f5aa4f6c81c9050abbd",
    },
    "cake_kimi_k3_fused_router_l_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "L",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_d58aa328ca957ffdae23",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_d58aa328ca957ffdae23_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_d58aa328ca957ffdae23_binding.cu",
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
            "closure_sha256": "163ce8a01265a43499d59ede3ffe95f76577e8225f0b9a18c89629c75ac0c62c",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "163ce8a01265a43499d59ede3ffe95f76577e8225f0b9a18c89629c75ac0c62c",
    },
    "cake_kimi_k3_fused_router_lc16_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 16,
        "main": {
            "module": "cake_kimi_k3_fused_router_4017b289ad93cc957417",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_4017b289ad93cc957417_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_4017b289ad93cc957417_binding.cu",
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
            "closure_sha256": "9825eef8938742b0c951535ce8cbb56852813e1683a68d89108097ff8cc8adaf",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [16, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "9825eef8938742b0c951535ce8cbb56852813e1683a68d89108097ff8cc8adaf",
    },
    "cake_kimi_k3_fused_router_lc16_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 16,
        "main": {
            "module": "cake_kimi_k3_fused_router_d12fb8d32d70f983d645",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_d12fb8d32d70f983d645_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_d12fb8d32d70f983d645_binding.cu",
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
            "closure_sha256": "8619cd2d6e92a895a05f71421075ec701ef0289fecd117ea7d7bf860d77d53dc",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [16, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "8619cd2d6e92a895a05f71421075ec701ef0289fecd117ea7d7bf860d77d53dc",
    },
    "cake_kimi_k3_fused_router_lc16_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 16,
        "main": {
            "module": "cake_kimi_k3_fused_router_8e7b3f3dfa020a6ae9d9",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_8e7b3f3dfa020a6ae9d9_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_8e7b3f3dfa020a6ae9d9_binding.cu",
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
            "closure_sha256": "b55ace7d92178a97844b9370333f9b2c2e9248321522c7ff9ab7662f845b43b3",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [16, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "b55ace7d92178a97844b9370333f9b2c2e9248321522c7ff9ab7662f845b43b3",
    },
    "cake_kimi_k3_fused_router_lc16_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 16,
        "main": {
            "module": "cake_kimi_k3_fused_router_3a5319e7ee55a7fa1e5c",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_3a5319e7ee55a7fa1e5c_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_3a5319e7ee55a7fa1e5c_binding.cu",
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
            "closure_sha256": "bce4a11c4fd7d06ac0a51f147862ad1cab7baba10cda161cf74a10a281681ff0",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [16, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "bce4a11c4fd7d06ac0a51f147862ad1cab7baba10cda161cf74a10a281681ff0",
    },
    "cake_kimi_k3_fused_router_lc1_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 1,
        "main": {
            "module": "cake_kimi_k3_fused_router_45c8e2fadd1a9bb077ad",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_45c8e2fadd1a9bb077ad_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_45c8e2fadd1a9bb077ad_binding.cu",
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
            "closure_sha256": "b621f68a809a9e934327fcbc316961826c43d184f72bdec7001a5984dcc75350",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "b621f68a809a9e934327fcbc316961826c43d184f72bdec7001a5984dcc75350",
    },
    "cake_kimi_k3_fused_router_lc1_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 1,
        "main": {
            "module": "cake_kimi_k3_fused_router_a56ddeaca4d13ba0b3d2",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_a56ddeaca4d13ba0b3d2_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_a56ddeaca4d13ba0b3d2_binding.cu",
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
            "closure_sha256": "b5e6473b39814ca171b91bbf1d8da9f606e68c9ca08dc9590687f89cbbc8ad58",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "b5e6473b39814ca171b91bbf1d8da9f606e68c9ca08dc9590687f89cbbc8ad58",
    },
    "cake_kimi_k3_fused_router_lc1_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 1,
        "main": {
            "module": "cake_kimi_k3_fused_router_265956a33db4fefe7cc5",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_265956a33db4fefe7cc5_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_265956a33db4fefe7cc5_binding.cu",
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
            "closure_sha256": "c90a58ba980cf654401f7884bcfe07ff6db59d335cd60cd7bec88950956a801e",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "c90a58ba980cf654401f7884bcfe07ff6db59d335cd60cd7bec88950956a801e",
    },
    "cake_kimi_k3_fused_router_lc1_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 1,
        "main": {
            "module": "cake_kimi_k3_fused_router_2d82264fdb83f4f009ce",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_2d82264fdb83f4f009ce_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_2d82264fdb83f4f009ce_binding.cu",
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
            "closure_sha256": "0f1f5a528824f9e44242e48061ee60ca83f7412190f1a22576865fc61711bcf4",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "0f1f5a528824f9e44242e48061ee60ca83f7412190f1a22576865fc61711bcf4",
    },
    "cake_kimi_k3_fused_router_lc2_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_472dd7224ae155206d46",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_472dd7224ae155206d46_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_472dd7224ae155206d46_binding.cu",
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
            "closure_sha256": "5c31b3124e987bb0bb5d7105fca5bf95045903cb9cebadc9e9f1f0794f0ef492",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "5c31b3124e987bb0bb5d7105fca5bf95045903cb9cebadc9e9f1f0794f0ef492",
    },
    "cake_kimi_k3_fused_router_lc2_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_e429f6ab2aa95c75b968",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_e429f6ab2aa95c75b968_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_e429f6ab2aa95c75b968_binding.cu",
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
            "closure_sha256": "2aa1c950ceb0d6e683da36844e5b44f0aeafbfb9631e266115ecb9cee6b9d261",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "2aa1c950ceb0d6e683da36844e5b44f0aeafbfb9631e266115ecb9cee6b9d261",
    },
    "cake_kimi_k3_fused_router_lc2_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_27186893942662009c9a",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_27186893942662009c9a_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_27186893942662009c9a_binding.cu",
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
            "closure_sha256": "967a548ef4058ed4ea7fcbff634a1a7d51e1553d20ff739d188e34a854bd070d",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "967a548ef4058ed4ea7fcbff634a1a7d51e1553d20ff739d188e34a854bd070d",
    },
    "cake_kimi_k3_fused_router_lc2_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_9121e44aa9ca19842290",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_9121e44aa9ca19842290_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_9121e44aa9ca19842290_binding.cu",
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
            "closure_sha256": "8a5551f62d949331b71c12293765fdd03fcaa0edabaab4d04ad02b71fe6ac5fc",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "8a5551f62d949331b71c12293765fdd03fcaa0edabaab4d04ad02b71fe6ac5fc",
    },
    "cake_kimi_k3_fused_router_lc4_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_d94a7ea1a88024ae3d57",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_d94a7ea1a88024ae3d57_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_d94a7ea1a88024ae3d57_binding.cu",
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
            "closure_sha256": "3986b8462a279286b28f5af003e6848d2ea87c1992df5b7792bb7331b967ec2c",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "3986b8462a279286b28f5af003e6848d2ea87c1992df5b7792bb7331b967ec2c",
    },
    "cake_kimi_k3_fused_router_lc4_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_a5d34a9799f137b0b126",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_a5d34a9799f137b0b126_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_a5d34a9799f137b0b126_binding.cu",
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
            "closure_sha256": "55b5eadc8d5cc8d3e89fc59933a200aa5fe02122bdb49d8aa8a38d633f9466b5",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "55b5eadc8d5cc8d3e89fc59933a200aa5fe02122bdb49d8aa8a38d633f9466b5",
    },
    "cake_kimi_k3_fused_router_lc4_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_683abb4442dab9898a66",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_683abb4442dab9898a66_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_683abb4442dab9898a66_binding.cu",
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
            "closure_sha256": "60841dcd78104a88295826e4fed01e6514943466c6dba8aa0489dbadb05ad4b8",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "60841dcd78104a88295826e4fed01e6514943466c6dba8aa0489dbadb05ad4b8",
    },
    "cake_kimi_k3_fused_router_lc4_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_2470f3a5ddf86b62ddf7",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_2470f3a5ddf86b62ddf7_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_2470f3a5ddf86b62ddf7_binding.cu",
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
            "closure_sha256": "413ba01ed1063970053c24dc6677bab27520af5f4db38610c6898b3356ce95c7",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "413ba01ed1063970053c24dc6677bab27520af5f4db38610c6898b3356ce95c7",
    },
    "cake_kimi_k3_fused_router_lc8_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_a76346a5b73a29dfeec9",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_a76346a5b73a29dfeec9_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_a76346a5b73a29dfeec9_binding.cu",
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
            "closure_sha256": "1970d3a0e88538f479a4c7c63cd5dba555c9d6f376ef9d9dd618019301bf2374",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "1970d3a0e88538f479a4c7c63cd5dba555c9d6f376ef9d9dd618019301bf2374",
    },
    "cake_kimi_k3_fused_router_lc8_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_55dadd4f9b4b3e146da8",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_55dadd4f9b4b3e146da8_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_55dadd4f9b4b3e146da8_binding.cu",
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
            "closure_sha256": "d7769fe79f9c1baf2ae5496fc685711d8c9e4862307fb7752d6ae8044bd59511",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "d7769fe79f9c1baf2ae5496fc685711d8c9e4862307fb7752d6ae8044bd59511",
    },
    "cake_kimi_k3_fused_router_lc8_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_7c597d814131c1348b90",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_7c597d814131c1348b90_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_7c597d814131c1348b90_binding.cu",
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
            "closure_sha256": "82674fd3c7ecf485650948304e6c2d24d803241016b5d7321ff87fbc0c1e55d4",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "82674fd3c7ecf485650948304e6c2d24d803241016b5d7321ff87fbc0c1e55d4",
    },
    "cake_kimi_k3_fused_router_lc8_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_6148ffc932cedc17eafa",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_6148ffc932cedc17eafa_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_6148ffc932cedc17eafa_binding.cu",
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
            "closure_sha256": "cdf523aed55d6ab46ae8b052dbdeff6e95a935045fdfcfa78c5645528c8592b5",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "cdf523aed55d6ab46ae8b052dbdeff6e95a935045fdfcfa78c5645528c8592b5",
    },
    "cake_kimi_k3_fused_router_m_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "M",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_602efd15f39203a75b7e",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_602efd15f39203a75b7e_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_602efd15f39203a75b7e_binding.cu",
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
            "closure_sha256": "a3e3316b7fdfeb1384d7b6af3b63d8c8aecbd33efbd732fd59ca393a81d11320",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "a3e3316b7fdfeb1384d7b6af3b63d8c8aecbd33efbd732fd59ca393a81d11320",
    },
    "cake_kimi_k3_fused_router_m_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "M",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_0296e37edd23439504f8",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_0296e37edd23439504f8_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_0296e37edd23439504f8_binding.cu",
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
            "closure_sha256": "fcc46aaaeae54086dd093fc9141873c43d259dfe5a4f433f40277bc4b47dda8a",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "fcc46aaaeae54086dd093fc9141873c43d259dfe5a4f433f40277bc4b47dda8a",
    },
    "cake_kimi_k3_fused_router_m_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "M",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_eae61ef2bee46fd55821",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_eae61ef2bee46fd55821_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_eae61ef2bee46fd55821_binding.cu",
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
            "closure_sha256": "536f8c9567db99dafec48ac6828dfd20b672d65125820e4dc2ca52a807765b39",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "536f8c9567db99dafec48ac6828dfd20b672d65125820e4dc2ca52a807765b39",
    },
    "cake_kimi_k3_fused_router_m_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "M",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_794a3e561e8f0f635b4e",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_794a3e561e8f0f635b4e_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_794a3e561e8f0f635b4e_binding.cu",
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
            "closure_sha256": "f5dd1a971a69d917fef8ed29cac0cd8f9d00934434881e8d7decab39ba55c8d5",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "f5dd1a971a69d917fef8ed29cac0cd8f9d00934434881e8d7decab39ba55c8d5",
    },
    "cake_kimi_k3_fused_router_q4s_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "Q4S",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_b4769c721de863485e8a",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_b4769c721de863485e8a_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_b4769c721de863485e8a_binding.cu",
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
            "closure_sha256": "e6bb4e5cd0918b1e4da224dc90be71e0b867ba976bba8a50f12ff7919957157e",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "e6bb4e5cd0918b1e4da224dc90be71e0b867ba976bba8a50f12ff7919957157e",
    },
    "cake_kimi_k3_fused_router_q4s_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "Q4S",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_e14b304dd12be528b8d9",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_e14b304dd12be528b8d9_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_e14b304dd12be528b8d9_binding.cu",
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
            "closure_sha256": "51f38d19ef5ca60e70c1cf716a47270802382fff78e75a008e7507ceb0a9d98a",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "51f38d19ef5ca60e70c1cf716a47270802382fff78e75a008e7507ceb0a9d98a",
    },
    "cake_kimi_k3_fused_router_q4s_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "Q4S",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_581c19b3face9437332d",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_581c19b3face9437332d_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_581c19b3face9437332d_binding.cu",
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
            "closure_sha256": "0f497c67facb968ae1a7f68c64f618085071010f35117536470fb1bb895f5ca1",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "0f497c67facb968ae1a7f68c64f618085071010f35117536470fb1bb895f5ca1",
    },
    "cake_kimi_k3_fused_router_q4s_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "Q4S",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_ff82f4246005af5a34b8",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_ff82f4246005af5a34b8_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_ff82f4246005af5a34b8_binding.cu",
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
            "closure_sha256": "84de2a0476785fd1aed93f715416c081dbd186e0d3f353cc62fc038f1adc88d2",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "84de2a0476785fd1aed93f715416c081dbd186e0d3f353cc62fc038f1adc88d2",
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
