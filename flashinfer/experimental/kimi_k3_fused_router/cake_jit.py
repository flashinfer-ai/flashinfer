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
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags
from ...jit.utils import write_if_different

# Registration of the generated Kimi-K3 fused router programs (SM100 / SM103).
#
# ``MODULES`` holds one record per physical program (a kernel source plus its
# host launcher): the dispatch ``arm`` it implements, the ``num_tokens`` it is
# bound to (arm LC only; ``None`` for the shared kernels), the architectures the
# source compiles for, the two translation units under ``csrc/``, compile flags,
# FFI entry, argument plan, launch geometry (block, cluster, cooperative flag,
# dynamic shared memory) and the closure identity (SHA-256 over the translation
# units and the shared headers they include).  One source serves every listed
# architecture: architecture-specific lowering lines sit behind
# ``__CUDA_ARCH__`` guards inside the source and the loader compiles it with
# the flag set of the device it runs on.  Programs whose persistent grid is
# bounded by the driver's co-resident cluster capacity (the 4-CTA-cluster arms
# and the 16-CTA cluster of the sixteen-token kernel) also carry the
# declaration of their kernel; the occupancy unit below forwards it to
# ``cudaOccupancyMaxActiveClusters``.
#
# ``ROUTES`` maps, per architecture, the logical kernel key the host dispatcher
# resolves at preparation (``"L"``, ``"LP"``, ``"M"``, ``"Q4S"``, ``"Q4SP"``,
# ``"GW"`` or ``"LC:<num_tokens>"``) to its program.  Every program takes the
# route block alignment on the compile line (``-DBLOCK_M=8`` / ``-DBLOCK_M=16``);
# the source carries no default.
#
# Both literals are populated by the generated-program export; do not edit
# them by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_kimi_k3_fused_router_045ff7650cba571a4f5a": {
        "arm": "M",
        "num_tokens": None,
        "arches": ["sm_100a"],
        "sources": [
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_045ff7650cba571a4f5a_kernel.cu",
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_045ff7650cba571a4f5a_binding.cu",
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
        "launch": {
            "block": [224, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": True,
            "dynamic_smem_bytes": 9344,
        },
        "closure_sha256": {
            "sm_100a": "0060c619f382165aef92e7ceac9bd43141212042131f2df47a37bb2ee24a45d4",
        },
        "kernel_symbol": "kernel_cake_kimi_k3_fused_router_045ff7650cba571a4f5a",
    },
    "cake_kimi_k3_fused_router_4c42741998bdc491ccee": {
        "arm": "GW",
        "num_tokens": None,
        "arches": ["sm_100a", "sm_103a"],
        "sources": [
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_4c42741998bdc491ccee_kernel.cu",
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_4c42741998bdc491ccee_binding.cu",
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
        "launch": {
            "block": [224, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": True,
            "dynamic_smem_bytes": 32768,
        },
        "closure_sha256": {
            "sm_100a": "1901391002bc44980bc6525011d3d0ecbe325c39d5df67e5539e8a9e867c44c8",
            "sm_103a": "7719879f1e5ac92abac4eb43beeb378c07d9654fb4cff77fc89083d7ca94cc05",
        },
        "kernel_symbol": "kernel_cake_kimi_k3_fused_router_4c42741998bdc491ccee",
    },
    "cake_kimi_k3_fused_router_57eabcfcbd9fb3d39f75": {
        "arm": "LC",
        "num_tokens": 4,
        "arches": ["sm_100a", "sm_103a"],
        "sources": [
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_57eabcfcbd9fb3d39f75_kernel.cu",
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_57eabcfcbd9fb3d39f75_binding.cu",
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
        "launch": {
            "block": [224, 1, 1],
            "cluster": [4, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 16896,
        },
        "closure_sha256": {
            "sm_100a": "8b8668782f76af4a1c1b29fd88d12f7c7d52766661cf006779450aacc6633cb6",
            "sm_103a": "031a24e20480529f3897560a5fcb1ff8d860cf06d35ad867b7d508a894c7c700",
        },
        "kernel_symbol": "kernel_cake_kimi_k3_fused_router_57eabcfcbd9fb3d39f75",
    },
    "cake_kimi_k3_fused_router_60dbc59ea4518ad97b9f": {
        "arm": "L",
        "num_tokens": None,
        "arches": ["sm_103a"],
        "sources": [
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_60dbc59ea4518ad97b9f_kernel.cu",
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_60dbc59ea4518ad97b9f_binding.cu",
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
        "launch": {
            "block": [224, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": True,
            "dynamic_smem_bytes": 40960,
        },
        "closure_sha256": {
            "sm_103a": "e0828940ca0d0c0ebd5e4a953858a3fb6e5b21dee3dd38635e188e9bfb7fcd09",
        },
        "kernel_symbol": "kernel_cake_kimi_k3_fused_router_60dbc59ea4518ad97b9f",
    },
    "cake_kimi_k3_fused_router_708d694a9c7844cdce53": {
        "arm": "LP",
        "num_tokens": None,
        "arches": ["sm_103a"],
        "sources": [
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_708d694a9c7844cdce53_kernel.cu",
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_708d694a9c7844cdce53_binding.cu",
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
        "launch": {
            "block": [224, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": True,
            "dynamic_smem_bytes": 40960,
        },
        "closure_sha256": {
            "sm_103a": "55fc8bfd9bb7c630c55ceb001cd3c7839250b67f67f22451706c2a42f292543f",
        },
        "kernel_symbol": "kernel_cake_kimi_k3_fused_router_708d694a9c7844cdce53",
    },
    "cake_kimi_k3_fused_router_71d1c647e2fbe0eeb09d": {
        "arm": "M",
        "num_tokens": None,
        "arches": ["sm_103a"],
        "sources": [
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_71d1c647e2fbe0eeb09d_kernel.cu",
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_71d1c647e2fbe0eeb09d_binding.cu",
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
        "launch": {
            "block": [224, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": True,
            "dynamic_smem_bytes": 17152,
        },
        "closure_sha256": {
            "sm_103a": "60dc6b8dc2f64c9bd79158fef79d39bae5512d5f1c45c2a30931abcf670d559a",
        },
        "kernel_symbol": "kernel_cake_kimi_k3_fused_router_71d1c647e2fbe0eeb09d",
    },
    "cake_kimi_k3_fused_router_86bbf6d4f01730e4bb80": {
        "arm": "LC",
        "num_tokens": 16,
        "arches": ["sm_100a", "sm_103a"],
        "sources": [
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_86bbf6d4f01730e4bb80_kernel.cu",
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_86bbf6d4f01730e4bb80_binding.cu",
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
        "launch": {
            "block": [224, 1, 1],
            "cluster": [16, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 16896,
        },
        "closure_sha256": {
            "sm_100a": "63d110f444a8fa67a31a3865c530fe302e9d74dcdad3c6f2200d7b008b713c7c",
            "sm_103a": "592455209648cba8464bdbcd6fa040ee5d95d8ef5f5eef8354ac648332238e65",
        },
        "kernel_symbol": "kernel_cake_kimi_k3_fused_router_86bbf6d4f01730e4bb80",
        "kernel_declaration": 'extern "C" __global__ void kernel_cake_kimi_k3_fused_router_86bbf6d4f01730e4bb80(float* __restrict__ logits, float* __restrict__ bias, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M);',
    },
    "cake_kimi_k3_fused_router_8f5c40be6676ad1c9117": {
        "arm": "LC",
        "num_tokens": 2,
        "arches": ["sm_100a", "sm_103a"],
        "sources": [
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_8f5c40be6676ad1c9117_kernel.cu",
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_8f5c40be6676ad1c9117_binding.cu",
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
        "launch": {
            "block": [224, 1, 1],
            "cluster": [2, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 16896,
        },
        "closure_sha256": {
            "sm_100a": "852ff2065772571da0d2aff820f1662fb8aff62b25940c35306f73c4e3f7f952",
            "sm_103a": "ff6219f824168546b6a4eb64b7ba8e97e8d6b17b571c0f62a4918f89dedcd968",
        },
        "kernel_symbol": "kernel_cake_kimi_k3_fused_router_8f5c40be6676ad1c9117",
    },
    "cake_kimi_k3_fused_router_962f15b28e84fee89ca4": {
        "arm": "LC",
        "num_tokens": 8,
        "arches": ["sm_100a", "sm_103a"],
        "sources": [
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_962f15b28e84fee89ca4_kernel.cu",
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_962f15b28e84fee89ca4_binding.cu",
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
        "launch": {
            "block": [224, 1, 1],
            "cluster": [8, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 16896,
        },
        "closure_sha256": {
            "sm_100a": "5b40e008758e0e71194035ec318d995e7eef63033158d3d7936226a450e8f0f3",
            "sm_103a": "ddd2717f5c06c5a7871d70df81b9f7034b53ebcf0c40737d62b49e78b296b8e0",
        },
        "kernel_symbol": "kernel_cake_kimi_k3_fused_router_962f15b28e84fee89ca4",
    },
    "cake_kimi_k3_fused_router_b68f1c43a8ff937bf31b": {
        "arm": "LP",
        "num_tokens": None,
        "arches": ["sm_100a"],
        "sources": [
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_b68f1c43a8ff937bf31b_kernel.cu",
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_b68f1c43a8ff937bf31b_binding.cu",
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
        "launch": {
            "block": [224, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": True,
            "dynamic_smem_bytes": 40960,
        },
        "closure_sha256": {
            "sm_100a": "0aed3eaf0b5ba410e5f654222b451636abb42afdeb70cd547f555b92e9958d42",
        },
        "kernel_symbol": "kernel_cake_kimi_k3_fused_router_b68f1c43a8ff937bf31b",
    },
    "cake_kimi_k3_fused_router_d1cc33620067c5d15476": {
        "arm": "LC",
        "num_tokens": 1,
        "arches": ["sm_100a", "sm_103a"],
        "sources": [
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_d1cc33620067c5d15476_kernel.cu",
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_d1cc33620067c5d15476_binding.cu",
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
        "launch": {
            "block": [224, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 16896,
        },
        "closure_sha256": {
            "sm_100a": "a1edaf2779ff21d9b35cdcea22cfcf395990667b553cc25dd58f0d08823cc6ee",
            "sm_103a": "5fa92fb8552ac81c91e95101c5703f6ae3b11e74f1137fd7d32c21005701f846",
        },
        "kernel_symbol": "kernel_cake_kimi_k3_fused_router_d1cc33620067c5d15476",
    },
    "cake_kimi_k3_fused_router_d9e433c790c24a24bab1": {
        "arm": "Q4SP",
        "num_tokens": None,
        "arches": ["sm_100a", "sm_103a"],
        "sources": [
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_d9e433c790c24a24bab1_kernel.cu",
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_d9e433c790c24a24bab1_binding.cu",
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
        "launch": {
            "block": [224, 1, 1],
            "cluster": [4, 1, 1],
            "cooperative": True,
            "dynamic_smem_bytes": 49408,
        },
        "closure_sha256": {
            "sm_100a": "ea084330c173b2bb3d6d271ae9f431de366d42967282f9869cf46680075d7372",
            "sm_103a": "d3ef54adf1ffaf702016660085ed2675fc33f751484b7d52d0c1d817a33e4521",
        },
        "kernel_symbol": "kernel_cake_kimi_k3_fused_router_d9e433c790c24a24bab1",
        "kernel_declaration": 'extern "C" __global__ void kernel_cake_kimi_k3_fused_router_d9e433c790c24a24bab1(float* __restrict__ logits, float* __restrict__ bias, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M);',
    },
    "cake_kimi_k3_fused_router_e9d0e24ee4dbca06542a": {
        "arm": "L",
        "num_tokens": None,
        "arches": ["sm_100a"],
        "sources": [
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_e9d0e24ee4dbca06542a_kernel.cu",
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_e9d0e24ee4dbca06542a_binding.cu",
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
        "launch": {
            "block": [224, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": True,
            "dynamic_smem_bytes": 40960,
        },
        "closure_sha256": {
            "sm_100a": "632b751202e5de56501cda183246a47401ab3a1c31efc5009b1bf544286f3ff4",
        },
        "kernel_symbol": "kernel_cake_kimi_k3_fused_router_e9d0e24ee4dbca06542a",
    },
    "cake_kimi_k3_fused_router_f6c232446f12d0466fdf": {
        "arm": "Q4S",
        "num_tokens": None,
        "arches": ["sm_100a", "sm_103a"],
        "sources": [
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_f6c232446f12d0466fdf_kernel.cu",
            "cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_f6c232446f12d0466fdf_binding.cu",
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
        "launch": {
            "block": [224, 1, 1],
            "cluster": [4, 1, 1],
            "cooperative": True,
            "dynamic_smem_bytes": 49408,
        },
        "closure_sha256": {
            "sm_100a": "3365c98493905847d72ec87b6ee5b8418509ad99274969be58959d5680804dde",
            "sm_103a": "ed1b4b553471a07ca485331c41d23859749fd4effc0b690ea4531cc86e03333a",
        },
        "kernel_symbol": "kernel_cake_kimi_k3_fused_router_f6c232446f12d0466fdf",
        "kernel_declaration": 'extern "C" __global__ void kernel_cake_kimi_k3_fused_router_f6c232446f12d0466fdf(float* __restrict__ logits, float* __restrict__ bias, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M);',
    },
}
ROUTES: dict[str, dict[str, str]] = {
    "sm_100a": {
        "GW": "cake_kimi_k3_fused_router_4c42741998bdc491ccee",
        "L": "cake_kimi_k3_fused_router_e9d0e24ee4dbca06542a",
        "LC:1": "cake_kimi_k3_fused_router_d1cc33620067c5d15476",
        "LC:16": "cake_kimi_k3_fused_router_86bbf6d4f01730e4bb80",
        "LC:2": "cake_kimi_k3_fused_router_8f5c40be6676ad1c9117",
        "LC:4": "cake_kimi_k3_fused_router_57eabcfcbd9fb3d39f75",
        "LC:8": "cake_kimi_k3_fused_router_962f15b28e84fee89ca4",
        "LP": "cake_kimi_k3_fused_router_b68f1c43a8ff937bf31b",
        "M": "cake_kimi_k3_fused_router_045ff7650cba571a4f5a",
        "Q4S": "cake_kimi_k3_fused_router_f6c232446f12d0466fdf",
        "Q4SP": "cake_kimi_k3_fused_router_d9e433c790c24a24bab1",
    },
    "sm_103a": {
        "GW": "cake_kimi_k3_fused_router_4c42741998bdc491ccee",
        "L": "cake_kimi_k3_fused_router_60dbc59ea4518ad97b9f",
        "LC:1": "cake_kimi_k3_fused_router_d1cc33620067c5d15476",
        "LC:16": "cake_kimi_k3_fused_router_86bbf6d4f01730e4bb80",
        "LC:2": "cake_kimi_k3_fused_router_8f5c40be6676ad1c9117",
        "LC:4": "cake_kimi_k3_fused_router_57eabcfcbd9fb3d39f75",
        "LC:8": "cake_kimi_k3_fused_router_962f15b28e84fee89ca4",
        "LP": "cake_kimi_k3_fused_router_708d694a9c7844cdce53",
        "M": "cake_kimi_k3_fused_router_71d1c647e2fbe0eeb09d",
        "Q4S": "cake_kimi_k3_fused_router_f6c232446f12d0466fdf",
        "Q4SP": "cake_kimi_k3_fused_router_d9e433c790c24a24bab1",
    },
}

ARCHES = ("sm_100a", "sm_103a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
BLOCK_M_VALUES = (8, 16)

# Dispatch arms of the generated program (see ``cake_backend`` for the route
# tables and launch rules):
#   L   : one-join plan builder, num_tokens <= 128, at least 128 CTAs launched
#   LP  : the L kernel with a register prefetch of the bias and the first
#         row's logits above its prologue barrier, same grid rule
#   LC  : one cluster of num_tokens CTAs (a single CTA for one token, 2, 4, 8
#         or 16 CTAs otherwise) exchanging selected ids through distributed
#         shared memory; one kernel per row count, non-cooperative launch
#   M   : one-join bitmap plan builder, 128 <= num_tokens <= 256 (SM100) /
#         2048 (SM103)
#   Q4S : arm M's cp.async ID stream in 4-CTA clusters (cooperative cluster
#         launch) at four CTAs per SM, 128 <= num_tokens <= 2048
#   Q4SP: the Q4S kernel with a register prefetch of the next row's logits,
#         same cluster / launch bounds / grid rule
#   GW  : two-join persistent warp-per-row kernel for the largest batches
ARMS = ("L", "LP", "LC", "M", "Q4S", "Q4SP", "GW")
# Arms registered per row count (one kernel per num_tokens); every other arm
# registers one program and serves all of its rows.
PER_ROW_COUNT_ARMS = ("LC",)


def kernel_key(arm: str, num_tokens: int | None = None) -> str:
    """Logical kernel key of ``arm`` (``"LC:<num_tokens>"`` for the per-row-count arm)."""
    if arm not in ARMS:
        raise ValueError(f"unknown dispatch arm {arm!r}")
    if arm in PER_ROW_COUNT_ARMS:
        if num_tokens is None:
            raise ValueError(f"arm {arm} is registered per num_tokens")
        return f"{arm}:{int(num_tokens)}"
    return arm


def program_for(arch: str, arm: str, num_tokens: int | None = None) -> str:
    """Program name serving ``arm`` (and ``num_tokens`` for arm LC) on ``arch``."""
    key = kernel_key(arm, num_tokens)
    name = ROUTES.get(arch, {}).get(key)
    if name is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 fused router program {key!r} for {arch} is not "
            "registered in this checkout"
        )
    if arch not in MODULES[name]["arches"]:
        raise NotImplementedError(
            f"The generated Kimi-K3 fused router program of {key!r} is not built for "
            f"{arch} (registered: {MODULES[name]['arches']})"
        )
    return name


def registered_keys(arch: str) -> set[str]:
    """Logical kernel keys with a program built for ``arch``."""
    return {
        key
        for key, name in ROUTES.get(arch, {}).items()
        if arch in MODULES[name]["arches"]
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
# Occupancy query for the cluster-bounded programs
# ---------------------------------------------------------------------------
#
# The generated binding launches the kernel (cooperative and cluster attributes
# included) but exposes no occupancy query.  The persistent grid of the
# 4-CTA-cluster arms must be bounded by the number of co-resident clusters the
# driver admits (GPC placement, not just CTAs per SM), and the 16-CTA
# non-portable cluster of the sixteen-token kernel needs at least one.  For
# exactly those programs the registry record carries the kernel declaration
# (``kernel_declaration``, taken from the generated source by the export), and
# this package adds one translation unit that forwards it to
# ``cudaOccupancyMaxActiveClusters`` with the one-cluster configuration the
# launch uses (block, cluster dims, dynamic shared memory, cooperative flag).

_OCCUPANCY_TEMPLATE = """\
// Occupancy query for the generated cluster kernel {symbol}.
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
  if (cluster_x * cluster_y * cluster_z > 8) {{
    TVM_FFI_CHECK(cudaFuncSetAttribute(function, cudaFuncAttributeNonPortableClusterSizeAllowed,
                                       1) == cudaSuccess,
                  RuntimeError)
        << "cudaFuncSetAttribute(NonPortableClusterSizeAllowed) failed for {symbol}";
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


def queries_occupancy(name: str) -> bool:
    """True when program ``name`` ships the cluster occupancy query."""
    return "kernel_declaration" in MODULES[name]


def _occupancy_source(spec_name: str, record: dict[str, Any]) -> Path:
    declaration = record["kernel_declaration"]
    symbol = record["kernel_symbol"]
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / spec_name
    os.makedirs(gen_directory, exist_ok=True)
    path = gen_directory / f"{symbol}_occupancy.cu"
    write_if_different(
        path, _OCCUPANCY_TEMPLATE.format(symbol=symbol, declaration=declaration)
    )
    return path


@functools.cache
def gen_kimi_k3_fused_router_module(name: str, arch: str, block_m: int):
    """JIT spec of program ``name`` compiled for ``arch`` with ``-DBLOCK_M=block_m``.

    The architecture and the define are part of the spec name, so two
    architectures (or two alignments) never share one cached library; the
    closure digest covers the translation units and their shared headers.
    """
    record = MODULES[name]
    if arch not in record["arches"]:
        raise ValueError(
            f"program {name!r} is not built for {arch} ({record['arches']})"
        )
    if int(block_m) not in BLOCK_M_VALUES:
        raise ValueError(f"block_m must be one of {BLOCK_M_VALUES}, got {block_m}")
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    spec_name = f"{name}_{arch}_bm{int(block_m)}_" + record["closure_sha256"][arch][:20]
    if queries_occupancy(name):
        sources.append(_occupancy_source(spec_name, record))
    return gen_jit_spec(
        name=spec_name,
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[arch],
            *record["compile_flags"],
            f"-DBLOCK_M={int(block_m)}",
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            root,
            *dict.fromkeys(p.parent for p in sources),
            *_header_dirs(),
        ],
        use_fast_math=False,
    )


@functools.cache
def load_kimi_k3_fused_router_module(name: str, arch: str, block_m: int):
    return gen_kimi_k3_fused_router_module(name, arch, block_m).build_and_load()
