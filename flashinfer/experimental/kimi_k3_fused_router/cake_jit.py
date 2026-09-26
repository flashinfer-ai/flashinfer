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
            "module": "cake_kimi_k3_fused_router_ee5ceecd9d07a3b62bf4",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_ee5ceecd9d07a3b62bf4_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_ee5ceecd9d07a3b62bf4_binding.cu",
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
            "closure_sha256": "403ce28682c93ae49754793be5af4384d51eee01226f4ae864bcc02a0a8643eb",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "403ce28682c93ae49754793be5af4384d51eee01226f4ae864bcc02a0a8643eb",
    },
    "cake_kimi_k3_fused_router_gw_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "GW",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_eba0da520bfae74745f3",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_eba0da520bfae74745f3_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_eba0da520bfae74745f3_binding.cu",
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
            "closure_sha256": "802e2f93dd842e8236bad9e8bc6fa55c02721090cfb8c07b98531c0247b00ada",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "802e2f93dd842e8236bad9e8bc6fa55c02721090cfb8c07b98531c0247b00ada",
    },
    "cake_kimi_k3_fused_router_gw_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "GW",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_ef42df39862269c07a1d",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_ef42df39862269c07a1d_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_ef42df39862269c07a1d_binding.cu",
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
            "closure_sha256": "79422e1f30971decae07ddfce667bb834d42616d085512e2e8666c45bb32cb47",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "79422e1f30971decae07ddfce667bb834d42616d085512e2e8666c45bb32cb47",
    },
    "cake_kimi_k3_fused_router_gw_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "GW",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_e7b548f804a751cb0c4a",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_e7b548f804a751cb0c4a_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_e7b548f804a751cb0c4a_binding.cu",
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
            "closure_sha256": "71badf070e5466b766b94c6e393d422fd36f3950ba4c2b5b4f2779518dd33f0e",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 32768,
            },
        },
        "closure_sha256": "71badf070e5466b766b94c6e393d422fd36f3950ba4c2b5b4f2779518dd33f0e",
    },
    "cake_kimi_k3_fused_router_l_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "L",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_b0aaa1c292a0cd36af7a",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_b0aaa1c292a0cd36af7a_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_b0aaa1c292a0cd36af7a_binding.cu",
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
            "closure_sha256": "5a811a41acc1bd708f6f623fbf7baebfcc2293f5b7f18c45ceb1096ade332307",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "5a811a41acc1bd708f6f623fbf7baebfcc2293f5b7f18c45ceb1096ade332307",
    },
    "cake_kimi_k3_fused_router_l_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "L",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_ba18b31a2d8e9f0b6dd0",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_ba18b31a2d8e9f0b6dd0_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_ba18b31a2d8e9f0b6dd0_binding.cu",
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
            "closure_sha256": "6788ffe5e5d6e60e103863d637b886b040abd885fe27a1343266802fe79188a6",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "6788ffe5e5d6e60e103863d637b886b040abd885fe27a1343266802fe79188a6",
    },
    "cake_kimi_k3_fused_router_l_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "L",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_b0dbf6b1a3d313907bc4",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_b0dbf6b1a3d313907bc4_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_b0dbf6b1a3d313907bc4_binding.cu",
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
            "closure_sha256": "8db99021316a901734d82bf802e3a34ffc90f585d19c1f9bb3a3af345c7858cc",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "8db99021316a901734d82bf802e3a34ffc90f585d19c1f9bb3a3af345c7858cc",
    },
    "cake_kimi_k3_fused_router_l_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "L",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_7a7b6e6201edd47f584e",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_7a7b6e6201edd47f584e_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_7a7b6e6201edd47f584e_binding.cu",
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
            "closure_sha256": "5e2ace57051b3189cdf86fdc665abb12515e269a3266bf4846909513d62a7a4c",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 40960,
            },
        },
        "closure_sha256": "5e2ace57051b3189cdf86fdc665abb12515e269a3266bf4846909513d62a7a4c",
    },
    "cake_kimi_k3_fused_router_lc2_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_d25730ea5c4d228ae7d9",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_d25730ea5c4d228ae7d9_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_d25730ea5c4d228ae7d9_binding.cu",
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
            "closure_sha256": "925eb5d5a64e9e517a58bbb806c05453883d1385f05498ee8b45054067649e3d",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "925eb5d5a64e9e517a58bbb806c05453883d1385f05498ee8b45054067649e3d",
    },
    "cake_kimi_k3_fused_router_lc2_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_b88e6859fedc60bbe0ac",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_b88e6859fedc60bbe0ac_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_b88e6859fedc60bbe0ac_binding.cu",
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
            "closure_sha256": "02d774c866e5b27e324f98f7c064842ae36e4bcc94ec1d3c7f6bfd283edb6926",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "02d774c866e5b27e324f98f7c064842ae36e4bcc94ec1d3c7f6bfd283edb6926",
    },
    "cake_kimi_k3_fused_router_lc2_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_0380cba8e73d67bb4ab9",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_0380cba8e73d67bb4ab9_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_0380cba8e73d67bb4ab9_binding.cu",
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
            "closure_sha256": "95ec8cec021d57c159bd57cc58acb07c4d32d3c423aeb6c22b63bf33b427a398",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "95ec8cec021d57c159bd57cc58acb07c4d32d3c423aeb6c22b63bf33b427a398",
    },
    "cake_kimi_k3_fused_router_lc2_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 2,
        "main": {
            "module": "cake_kimi_k3_fused_router_484c77b4884ede5f52a2",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_484c77b4884ede5f52a2_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_484c77b4884ede5f52a2_binding.cu",
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
            "closure_sha256": "18d5a640be4c56b28359c70be4ed731d39dee73238b59053c3ae4bfbfd2a22d4",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "18d5a640be4c56b28359c70be4ed731d39dee73238b59053c3ae4bfbfd2a22d4",
    },
    "cake_kimi_k3_fused_router_lc4_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_421b8a75d822521d2144",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_421b8a75d822521d2144_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_421b8a75d822521d2144_binding.cu",
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
            "closure_sha256": "79a23a71a04c3a89278b5b33d08872454c29b6c1c2e3b002ad188872720b453d",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "79a23a71a04c3a89278b5b33d08872454c29b6c1c2e3b002ad188872720b453d",
    },
    "cake_kimi_k3_fused_router_lc4_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_bf4f02b31c3d3ee7603f",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_bf4f02b31c3d3ee7603f_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_bf4f02b31c3d3ee7603f_binding.cu",
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
            "closure_sha256": "bd3647ab8a97d11f92c15e64be3a1926a9c5081a3130c65e0fbf3c884faf8412",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "bd3647ab8a97d11f92c15e64be3a1926a9c5081a3130c65e0fbf3c884faf8412",
    },
    "cake_kimi_k3_fused_router_lc4_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_038a86148244206306e8",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_038a86148244206306e8_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_038a86148244206306e8_binding.cu",
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
            "closure_sha256": "1c5cd09964c5fe921f8078b6dd50bc23b7af4ea2e0fd0dada05c3027d7396119",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "1c5cd09964c5fe921f8078b6dd50bc23b7af4ea2e0fd0dada05c3027d7396119",
    },
    "cake_kimi_k3_fused_router_lc4_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 4,
        "main": {
            "module": "cake_kimi_k3_fused_router_72095cb0854773f74d00",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_72095cb0854773f74d00_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_72095cb0854773f74d00_binding.cu",
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
            "closure_sha256": "62a2537d9be3e5ad15dce6d8ef229246d963357b8a6b622e4540ee687bfd7975",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "62a2537d9be3e5ad15dce6d8ef229246d963357b8a6b622e4540ee687bfd7975",
    },
    "cake_kimi_k3_fused_router_lc8_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_92e1d780d24dc73598a0",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_92e1d780d24dc73598a0_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_92e1d780d24dc73598a0_binding.cu",
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
            "closure_sha256": "cdf8d8d6ebce5e7e21c7567ad016203a286648cb3d613decdcbd192383876a1c",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "cdf8d8d6ebce5e7e21c7567ad016203a286648cb3d613decdcbd192383876a1c",
    },
    "cake_kimi_k3_fused_router_lc8_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 16,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_babfef5ddc8cad6731ad",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_babfef5ddc8cad6731ad_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_babfef5ddc8cad6731ad_binding.cu",
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
            "closure_sha256": "7d0f7d966a3be39fb1474610c5f8b5556b869946209c669b7d0320820f743e67",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "7d0f7d966a3be39fb1474610c5f8b5556b869946209c669b7d0320820f743e67",
    },
    "cake_kimi_k3_fused_router_lc8_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_687d3eb31e88dd3d4fe4",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_687d3eb31e88dd3d4fe4_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_687d3eb31e88dd3d4fe4_binding.cu",
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
            "closure_sha256": "ef39c0d9349d852e3222c3faad46bd69b8eb0b399b017eb6b0d6d20c95e7c494",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "ef39c0d9349d852e3222c3faad46bd69b8eb0b399b017eb6b0d6d20c95e7c494",
    },
    "cake_kimi_k3_fused_router_lc8_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "LC",
        "block_m": 8,
        "num_tokens": 8,
        "main": {
            "module": "cake_kimi_k3_fused_router_4b77f12f97bdf73aace5",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_4b77f12f97bdf73aace5_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_4b77f12f97bdf73aace5_binding.cu",
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
            "closure_sha256": "d3b9baa76ac41171e2d0247bf0bcef2393485eecde0f56a5cbf8c70ab97d2782",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [8, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 16896,
            },
        },
        "closure_sha256": "d3b9baa76ac41171e2d0247bf0bcef2393485eecde0f56a5cbf8c70ab97d2782",
    },
    "cake_kimi_k3_fused_router_m_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "M",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_e0cc8f0932d5baed8fe1",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_e0cc8f0932d5baed8fe1_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_e0cc8f0932d5baed8fe1_binding.cu",
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
            "closure_sha256": "19369612b7f68289c6d9026daed61aa3871fee0300ff68683ab3daa81444446f",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "19369612b7f68289c6d9026daed61aa3871fee0300ff68683ab3daa81444446f",
    },
    "cake_kimi_k3_fused_router_m_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "M",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_87037221a467ba60ef3e",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_87037221a467ba60ef3e_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_87037221a467ba60ef3e_binding.cu",
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
            "closure_sha256": "d52448105d66c5a0db316a2e0890256966006eb467cfbbd621121a4caf2e6b17",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "d52448105d66c5a0db316a2e0890256966006eb467cfbbd621121a4caf2e6b17",
    },
    "cake_kimi_k3_fused_router_m_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "M",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_8df07d0382f0a9cc0018",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_8df07d0382f0a9cc0018_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_8df07d0382f0a9cc0018_binding.cu",
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
            "closure_sha256": "1982d8f2f6a5edf0b033ba7112b35d0be86cd29c498522e31ffd3bf3c104991a",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "1982d8f2f6a5edf0b033ba7112b35d0be86cd29c498522e31ffd3bf3c104991a",
    },
    "cake_kimi_k3_fused_router_m_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "M",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_d9b2d4218bb4f5ed1681",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_d9b2d4218bb4f5ed1681_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_d9b2d4218bb4f5ed1681_binding.cu",
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
            "closure_sha256": "53c6e08c5d81e9ab080c387b7032c63d844cbfe1c7c24dfdc29d51ef2918b1fb",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 17152,
            },
        },
        "closure_sha256": "53c6e08c5d81e9ab080c387b7032c63d844cbfe1c7c24dfdc29d51ef2918b1fb",
    },
    "cake_kimi_k3_fused_router_q4s_bm16_sm_100a": {
        "arch": "sm_100a",
        "arm": "Q4S",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_f2f835f5e1255ba2f435",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_f2f835f5e1255ba2f435_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_f2f835f5e1255ba2f435_binding.cu",
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
            "closure_sha256": "3e8f6a8609e067ea4434e71f1fcf05417a7d6e6c92c97ad261369bb9833b8df5",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "3e8f6a8609e067ea4434e71f1fcf05417a7d6e6c92c97ad261369bb9833b8df5",
    },
    "cake_kimi_k3_fused_router_q4s_bm16_sm_103a": {
        "arch": "sm_103a",
        "arm": "Q4S",
        "block_m": 16,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_79711f0ec38a6d1cde59",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_79711f0ec38a6d1cde59_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_79711f0ec38a6d1cde59_binding.cu",
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
            "closure_sha256": "18296e39dda320050af7b85e35a37deea9b4e5f1197116fcc8179b6acf13d8db",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "18296e39dda320050af7b85e35a37deea9b4e5f1197116fcc8179b6acf13d8db",
    },
    "cake_kimi_k3_fused_router_q4s_bm8_sm_100a": {
        "arch": "sm_100a",
        "arm": "Q4S",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_f2a8eeb7288f49a287ab",
            "sources": [
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_f2a8eeb7288f49a287ab_kernel.cu",
                "cake_kimi_k3_fused_router/sm_100a/cake_kimi_k3_fused_router_f2a8eeb7288f49a287ab_binding.cu",
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
            "closure_sha256": "f81df76a6284a6f2d7bba4b6ca78d438db9ff66437c20f87e774b870cc46e5ce",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "f81df76a6284a6f2d7bba4b6ca78d438db9ff66437c20f87e774b870cc46e5ce",
    },
    "cake_kimi_k3_fused_router_q4s_bm8_sm_103a": {
        "arch": "sm_103a",
        "arm": "Q4S",
        "block_m": 8,
        "num_tokens": None,
        "main": {
            "module": "cake_kimi_k3_fused_router_3c1cb22024b4758cfadb",
            "sources": [
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_3c1cb22024b4758cfadb_kernel.cu",
                "cake_kimi_k3_fused_router/sm_103a/cake_kimi_k3_fused_router_3c1cb22024b4758cfadb_binding.cu",
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
            "closure_sha256": "416de1a8c549bf0a2e9963a4e5caf90f4469cfdc5a34dc0b5157eb86a210d911",
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [224, 1, 1],
                "cluster": [4, 1, 1],
                "cooperative": True,
                "dynamic_smem_bytes": 49408,
            },
        },
        "closure_sha256": "416de1a8c549bf0a2e9963a4e5caf90f4469cfdc5a34dc0b5157eb86a210d911",
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
