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
            "module": "cake_kimi_k3_fused_router_07cdba1397d2b574f330",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_07cdba1397d2b574f330_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_07cdba1397d2b574f330_binding.cu",
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
            "closure_sha256": "0baf4c94e0127821a78f48cc6060bca2b4281a5d211277f444d25442ad470c02",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "0baf4c94e0127821a78f48cc6060bca2b4281a5d211277f444d25442ad470c02",
    },
    "cake_kimi_k3_fused_router_gw_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "GW",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_b6c760c9dde2e5748c0e",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_b6c760c9dde2e5748c0e_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_b6c760c9dde2e5748c0e_binding.cu",
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
            "closure_sha256": "85d7455cbdd760d8db953337426ad73234b94002b6d3391c853b3bf50a4806d8",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "85d7455cbdd760d8db953337426ad73234b94002b6d3391c853b3bf50a4806d8",
    },
    "cake_kimi_k3_fused_router_gw_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "GW",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_6a3d908b990127be087f",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_6a3d908b990127be087f_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_6a3d908b990127be087f_binding.cu",
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
            "closure_sha256": "80ece14f6702aa4047dc91e68504cdd202f3b26368dfd4622745fa263f6d59de",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "80ece14f6702aa4047dc91e68504cdd202f3b26368dfd4622745fa263f6d59de",
    },
    "cake_kimi_k3_fused_router_gw_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "GW",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_4072c7b378c9b8db1932",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_4072c7b378c9b8db1932_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_4072c7b378c9b8db1932_binding.cu",
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
            "closure_sha256": "f5bfe7cfe3e79f42eaf7418e4a8a16639128615b27b1d033e1ec23cf47dd1335",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "f5bfe7cfe3e79f42eaf7418e4a8a16639128615b27b1d033e1ec23cf47dd1335",
    },
    "cake_kimi_k3_fused_router_l_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "L",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_102f058bbcaff298dd00",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_102f058bbcaff298dd00_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_102f058bbcaff298dd00_binding.cu",
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
            "closure_sha256": "cec84952ecfcfbcf3571317052bb775cfd805b7ac4d5549d2cddaaee63077bb1",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "cec84952ecfcfbcf3571317052bb775cfd805b7ac4d5549d2cddaaee63077bb1",
    },
    "cake_kimi_k3_fused_router_l_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "L",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_4c989bb14b58d901b14f",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_4c989bb14b58d901b14f_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_4c989bb14b58d901b14f_binding.cu",
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
            "closure_sha256": "ab553e43ee5bda5f0c6add564c96c2abff36a051aa0a06e50430d27f739d7e21",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "ab553e43ee5bda5f0c6add564c96c2abff36a051aa0a06e50430d27f739d7e21",
    },
    "cake_kimi_k3_fused_router_l_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "L",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_6db942769fab64daf439",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_6db942769fab64daf439_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_6db942769fab64daf439_binding.cu",
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
            "closure_sha256": "a4cd6b3f9685389435aaeae3bf939c80815334e94c1d24cf0db30059765cee15",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "a4cd6b3f9685389435aaeae3bf939c80815334e94c1d24cf0db30059765cee15",
    },
    "cake_kimi_k3_fused_router_l_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "L",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_dedd9cbf2a2e54aeb71a",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_dedd9cbf2a2e54aeb71a_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_dedd9cbf2a2e54aeb71a_binding.cu",
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
            "closure_sha256": "61a52cf158a1d9ec0a376d4f2ca90d28a183282786a33085c4fd08c3fcb2daf0",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "61a52cf158a1d9ec0a376d4f2ca90d28a183282786a33085c4fd08c3fcb2daf0",
    },
    "cake_kimi_k3_fused_router_lc2_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_e638b2fbb73d168afa50",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_e638b2fbb73d168afa50_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_e638b2fbb73d168afa50_binding.cu",
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
            "closure_sha256": "449aae3c40afa3b811f1e57acacb2f6cd9c270143f4a8348cdd185486ba4a79c",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "449aae3c40afa3b811f1e57acacb2f6cd9c270143f4a8348cdd185486ba4a79c",
    },
    "cake_kimi_k3_fused_router_lc2_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_37a7df3b444c308ca945",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_37a7df3b444c308ca945_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_37a7df3b444c308ca945_binding.cu",
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
            "closure_sha256": "6e4b4510997495b2061ad4b50bf69b8a9c599ee0aeac938e89d07b653090d551",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "6e4b4510997495b2061ad4b50bf69b8a9c599ee0aeac938e89d07b653090d551",
    },
    "cake_kimi_k3_fused_router_lc2_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_0bde7b0a06e6f38ab4e5",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_0bde7b0a06e6f38ab4e5_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_0bde7b0a06e6f38ab4e5_binding.cu",
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
            "closure_sha256": "bb35e76d17c9cf7f8e38857ae0e84f773ba099fd4afe37027efbc192e27990b3",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "bb35e76d17c9cf7f8e38857ae0e84f773ba099fd4afe37027efbc192e27990b3",
    },
    "cake_kimi_k3_fused_router_lc2_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_55ae849337547eff3ce7",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_55ae849337547eff3ce7_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_55ae849337547eff3ce7_binding.cu",
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
            "closure_sha256": "29157516ee58d20a734e77adc2e52eadb6fb3a759da5335274db5f09095d6f42",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "29157516ee58d20a734e77adc2e52eadb6fb3a759da5335274db5f09095d6f42",
    },
    "cake_kimi_k3_fused_router_lc4_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_517fe5089b9f0d78e2f9",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_517fe5089b9f0d78e2f9_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_517fe5089b9f0d78e2f9_binding.cu",
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
            "closure_sha256": "fbee6cb9eeef23a5a0732e934c01ecec654b61a7060f13354561b946a2b62431",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "fbee6cb9eeef23a5a0732e934c01ecec654b61a7060f13354561b946a2b62431",
    },
    "cake_kimi_k3_fused_router_lc4_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_ac4edc568d53b5cf835a",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_ac4edc568d53b5cf835a_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_ac4edc568d53b5cf835a_binding.cu",
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
            "closure_sha256": "05b406add3977ef49a0f91af82d8b21e6afc738d1fef944428ac58df92616c4d",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "05b406add3977ef49a0f91af82d8b21e6afc738d1fef944428ac58df92616c4d",
    },
    "cake_kimi_k3_fused_router_lc4_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_3ca5b129f78f00dd429b",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_3ca5b129f78f00dd429b_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_3ca5b129f78f00dd429b_binding.cu",
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
            "closure_sha256": "7a70db5f1f59030fe4d6493a4c5011645affe77856112e78b627fdbdab2ee56b",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "7a70db5f1f59030fe4d6493a4c5011645affe77856112e78b627fdbdab2ee56b",
    },
    "cake_kimi_k3_fused_router_lc4_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_44360f10d432f153db00",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_44360f10d432f153db00_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_44360f10d432f153db00_binding.cu",
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
            "closure_sha256": "ef3e28a9fbd3c282c0657ac1da0f3a92b72a94de5bf7d0f84f46316767ca13a0",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "ef3e28a9fbd3c282c0657ac1da0f3a92b72a94de5bf7d0f84f46316767ca13a0",
    },
    "cake_kimi_k3_fused_router_lc8_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_dfc0cf64a925d85827ea",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_dfc0cf64a925d85827ea_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_dfc0cf64a925d85827ea_binding.cu",
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
            "closure_sha256": "0bc7a753771d68833253e45e57b72cb0350b6cbfcc091cfc0d77f1eb01bad73c",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "0bc7a753771d68833253e45e57b72cb0350b6cbfcc091cfc0d77f1eb01bad73c",
    },
    "cake_kimi_k3_fused_router_lc8_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_739c0acfedb22a837166",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_739c0acfedb22a837166_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_739c0acfedb22a837166_binding.cu",
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
            "closure_sha256": "c3d56122dfcacd919f5fe1726f5f760a38601509cbb0ae9444f08f024c66d175",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "c3d56122dfcacd919f5fe1726f5f760a38601509cbb0ae9444f08f024c66d175",
    },
    "cake_kimi_k3_fused_router_lc8_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_7d3b45a1a4beac9477e2",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_7d3b45a1a4beac9477e2_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_7d3b45a1a4beac9477e2_binding.cu",
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
            "closure_sha256": "f51d27b8c921356cba70204b87cc235500274f8432d47a5382ba0577dfc3ac4d",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "f51d27b8c921356cba70204b87cc235500274f8432d47a5382ba0577dfc3ac4d",
    },
    "cake_kimi_k3_fused_router_lc8_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_8e28449694e1a739dabf",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_8e28449694e1a739dabf_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_8e28449694e1a739dabf_binding.cu",
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
            "closure_sha256": "f76361e81b2c8490593795cbac9913cc0976371a00f663a7308ca6da573939ce",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "f76361e81b2c8490593795cbac9913cc0976371a00f663a7308ca6da573939ce",
    },
    "cake_kimi_k3_fused_router_m_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "M",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_e4f337433abc128800e4",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_e4f337433abc128800e4_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_e4f337433abc128800e4_binding.cu",
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
            "closure_sha256": "cce4f0b8917b1f27d21da4e7a48ed32d299b5876bd9a87711b6465babedc6942",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "cce4f0b8917b1f27d21da4e7a48ed32d299b5876bd9a87711b6465babedc6942",
    },
    "cake_kimi_k3_fused_router_m_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "M",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_d42ceac9854c7a339581",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_d42ceac9854c7a339581_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_d42ceac9854c7a339581_binding.cu",
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
            "closure_sha256": "14f1846b6949c68dd44bb146e76fd5cfc6a9ac20b6c6c65257b6018b4d29578d",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "14f1846b6949c68dd44bb146e76fd5cfc6a9ac20b6c6c65257b6018b4d29578d",
    },
    "cake_kimi_k3_fused_router_m_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "M",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_421da233b459c4da3f96",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_421da233b459c4da3f96_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_421da233b459c4da3f96_binding.cu",
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
            "closure_sha256": "89cdebf236f7b3ba684e456dc74c0324be76f25e76f02c9249f62404b6428a38",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "89cdebf236f7b3ba684e456dc74c0324be76f25e76f02c9249f62404b6428a38",
    },
    "cake_kimi_k3_fused_router_m_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "M",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_c7aec6c7cb086f97540e",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_c7aec6c7cb086f97540e_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_c7aec6c7cb086f97540e_binding.cu",
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
            "closure_sha256": "242abe44526dd4e30087479ab88a6e8c2aabe997b53f4a0a7002503b696c3ca6",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "242abe44526dd4e30087479ab88a6e8c2aabe997b53f4a0a7002503b696c3ca6",
    },
    "cake_kimi_k3_fused_router_q4s_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "Q4S",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_0313c6cbad861be08246",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_0313c6cbad861be08246_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_0313c6cbad861be08246_binding.cu",
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
            "closure_sha256": "3e3fc9f5eb2df916c8feec929fed077fdd1ef959ddacbdc3f63bfabf7662933a",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "3e3fc9f5eb2df916c8feec929fed077fdd1ef959ddacbdc3f63bfabf7662933a",
    },
    "cake_kimi_k3_fused_router_q4s_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "Q4S",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_0abb4bf1a75075608fa0",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_0abb4bf1a75075608fa0_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_0abb4bf1a75075608fa0_binding.cu",
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
            "closure_sha256": "788c7883836084c712ba8ad961b304d755130745b9af20db6aa1babb8add73e0",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "788c7883836084c712ba8ad961b304d755130745b9af20db6aa1babb8add73e0",
    },
    "cake_kimi_k3_fused_router_q4s_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "Q4S",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_cf56795d077b30cd1733",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_cf56795d077b30cd1733_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_cf56795d077b30cd1733_binding.cu",
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
            "closure_sha256": "d8ce8d3fa24f41254bf56e80524446013ee0a0c1766e1629c9b7b0a5d7921c1e",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "d8ce8d3fa24f41254bf56e80524446013ee0a0c1766e1629c9b7b0a5d7921c1e",
    },
    "cake_kimi_k3_fused_router_q4s_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "Q4S",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_939c478d3e1895f177ee",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_939c478d3e1895f177ee_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_939c478d3e1895f177ee_binding.cu",
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
            "closure_sha256": "0d92399e989a244b66e6dffa74760d35e12467af9db08758029ad84b7e313875",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "0d92399e989a244b66e6dffa74760d35e12467af9db08758029ad84b7e313875",
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
