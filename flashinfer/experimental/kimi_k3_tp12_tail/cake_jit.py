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

# Explicit target-owned registration of the generated programs of the
# Kimi-K3 TP12 fused LatentMoE communication tail (SM100 / SM103, twelve
# ranks in one multi-node NVLink domain).
#
# ``MODULES`` holds one record per physical generated module (a kernel plus
# its host binding): translation units, compile flags, FFI entry, argument
# plan and closure identity.  ``KERNELS`` maps ``"<arch>"`` to the logical
# kernel key -> module assignment the host runtime resolves at preparation:
#
# * ``k1_oneshot:r<rank>``  the one-shot Lamport all-reduce of the routed
#   partial fused with KimiRMSNorm for ``M <= 16`` tokens; the rank is a
#   trace-time specialisation (reduction order and local packet position),
#   so one module per rank;
# * ``k1_twoshot:<grouped|pinned>``  the token-sliced two-shot form (owner
#   reduce + norm, multicast broadcast) for ``M > 16``; ``grouped`` issues all
#   twelve remote loads of the owner retry body before the first test
#   (``M <= 256``), ``pinned`` keeps the per-rank pinned body (``M > 256``);
# * ``k3:<grouped|pinned>``  the column reduce-scatter of the shared partial,
#   fused add of this rank's up-projection slice, one BF16 rounding and the
#   multicast all-gather of the output row, with the same poll schedule
#   selection by ``M``, one CTA per token and column half (``4 < M < 256``,
#   grouped poll schedule only);
# * ``k3_persist:<grouped|pinned>``  the same K3 on a persistent grid of
#   ``min(M, SM count)`` CTAs per column half, each walking its tokens as a
#   three-stage pipeline (scatter ``t``, owner reduce + multicast ``t - P``,
#   gather ``t - 2P``) so the fabric hops of consecutive tokens overlap
#   (``M >= 256``; identical buffers, flags and numerics);
# * ``k2_stream:n<640|512>``  the SIMT weight-streaming up-projection slice
#   GEMM (fp32 output) that replaces cuBLAS for ``M <= 4``; one module per
#   column width of the rank partition;
# * ``k3_f32:grouped``  the fp32-add form of K3 that consumes the K2-stream
#   slice (``M <= 4``).
#
# Every module is an exact-architecture program launched with programmatic
# dependent launch.  Both literals are populated verbatim by the
# generated-program export; do not edit them by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_kimi_k3_tp12_tail_059199fc13dd03bacb12": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_059199fc13dd03bacb12_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_059199fc13dd03bacb12_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "3e02baaa77c505701fce86e04afcdc430235fe2e0896c23b1ff55ceb999a13d3",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_0bdace2db5c1ea58de22": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_0bdace2db5c1ea58de22_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_0bdace2db5c1ea58de22_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "71b566155932259aceea3eaf0f231d7adb4fbfdcf8c4c17d4b71300eb55364a4",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_1aa73d0e225f019716c9": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_1aa73d0e225f019716c9_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_1aa73d0e225f019716c9_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "c44074e0f9954c105e77e880bb9190372a7bb2e9c15c52eb4869efd03e7bb630",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_208f95020e43bfd6b028": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_208f95020e43bfd6b028_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_208f95020e43bfd6b028_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "a15dff1c6bda093bda6b7964f19e14c35aee7011c178d1150431f182de0ac8ac",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_2612fc0c046ff8f7e1d7": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_2612fc0c046ff8f7e1d7_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_2612fc0c046ff8f7e1d7_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "bafc92b72a89b6a22dfa8bfc4f0971b47065e318e96a7c9dd67c4d13d35f48f1",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_38a23ee0f849bd40a266": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_38a23ee0f849bd40a266_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_38a23ee0f849bd40a266_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "956c24d2babed0e31ba9c51aa3c986b80b3322ef4474e337bc109951ac505a1b",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_3d4ea8055e0c38defafe": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_3d4ea8055e0c38defafe_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_3d4ea8055e0c38defafe_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "0c03448a215c83e8d1e7d1996d1b320ceb65e8511fc81254e66a8c69c46c0329",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_3e44777e07d6a76cb70b": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_3e44777e07d6a76cb70b_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_3e44777e07d6a76cb70b_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "86585257092cb99c5dea64424a8cdd5d0a5afd8a0dbee9775fa7f53f474ac2e0",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_3ffbf6f10c254327b6b4": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_3ffbf6f10c254327b6b4_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_3ffbf6f10c254327b6b4_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "6bc601bee8d3d61b6ec4ae4e1f999e74f54d81dead5f3eea5d834b693cf4b8f4",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_478739c83690aaedd5e5": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_478739c83690aaedd5e5_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_478739c83690aaedd5e5_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "51ab21aa02779f087982c8e5c57f8c563afb79b86e261e09d37d18945de2f7c4",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_4d472a0c4bfd124ac918": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_4d472a0c4bfd124ac918_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_4d472a0c4bfd124ac918_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "db34227c348c426f537a7aa12b681dbe7cb13e95ebc0d27cba20c8e47ebbb63b",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_5659a33e7f6575b3b6b8": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_5659a33e7f6575b3b6b8_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_5659a33e7f6575b3b6b8_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "58ec94e444f03ba41b23d520df158e89e9304e855f56f0cab240bd580363a9ea",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_5b0f2c34849ef87ac24b": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_5b0f2c34849ef87ac24b_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_5b0f2c34849ef87ac24b_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "2aa2ed9666b02df33a4a489a40832d63f3ed34791d879ced41d15a59ec5b7366",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_655245e4ea6a0566f7c2": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_655245e4ea6a0566f7c2_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_655245e4ea6a0566f7c2_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "1e3dcb63f657304b1efb07a05ac6a8a829cbb8806107c31138a623378a6c77bf",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_734d3ec84d75fbfad81d": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_734d3ec84d75fbfad81d_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_734d3ec84d75fbfad81d_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "0ca8f935971e2d00ab24c522cf700999c3618b259d00576447249e9daf64c83a",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_7d065b2e62d8c1a3d15a": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_7d065b2e62d8c1a3d15a_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_7d065b2e62d8c1a3d15a_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "shared"],
            ["buffer", "gemm_slice"],
            ["buffer", "out"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "my_col_begin"],
            ["parameter", "my_cols"],
            ["parameter", "gemm_plane_stride"],
            ["parameter", "num_gemm_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "c9fd3a6cfffa22897f82631be17e97a88e01e2565f1cc522d35a66a0da650edf",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_7de4539e58943861165d": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_7de4539e58943861165d_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_7de4539e58943861165d_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "eded13e1198815c0b062adc3f2f85ac1e7ddd8e7e3141367c7178a3b5494807d",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_854cb9b1586c72761b2f": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_854cb9b1586c72761b2f_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_854cb9b1586c72761b2f_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "fe5db193555a21a22da60f415176f62d045877bb8722223641b0e7c9c171d7d0",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_8621affbb892d25c21c4": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_8621affbb892d25c21c4_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_8621affbb892d25c21c4_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "529d5f8a04f7a5d16beb11e4cdaf8bcf743b712515787ee1d4ece1f40fa92286",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_9df6effe73530626c497": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_9df6effe73530626c497_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_9df6effe73530626c497_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "385e557d2eb01efa0ca0407d2db17635f6f0426956c44e0ce80d8cf9c1ee15c7",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_a6e3998e39ed767bb130": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_a6e3998e39ed767bb130_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_a6e3998e39ed767bb130_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "40bedb59091e74926778203ae6c62d44dbf8cdaa42f2d32fa4b4314fff6a1c76",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_afc049beb5340f185e43": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_afc049beb5340f185e43_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_afc049beb5340f185e43_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "shared"],
            ["buffer", "gemm_slice"],
            ["buffer", "out"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "my_col_begin"],
            ["parameter", "my_cols"],
            ["parameter", "gemm_plane_stride"],
            ["parameter", "num_gemm_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "6dca65a136f2879f65ae3e0535f1465dd803440191ff5b6edeceb41c3468f1e0",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_b18646c92b839845eebe": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_b18646c92b839845eebe_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_b18646c92b839845eebe_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "214600fde209598f73fc6aa788b7e837f70ce6188c3334a36e0267f574b8fd1d",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_b276b8216af1ef607f85": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_b276b8216af1ef607f85_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_b276b8216af1ef607f85_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "3827c63920edabcbe3aa751a460fd29466e980c5b515fd79a9ebd4413508ca99",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_bff3b68ca7b2da0a79f0": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_bff3b68ca7b2da0a79f0_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_bff3b68ca7b2da0a79f0_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "d9758ad64ed54ccdc90098e3686b778f5eee5d304c6b1cdc634912ecb80ad2da",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_c728b4dbc601934e79fc": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_c728b4dbc601934e79fc_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_c728b4dbc601934e79fc_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "c5a7c3d18c5d0f6507f617eeee776928ebb5451c0d4baafcd0f09579802a9374",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_c946edf026dbdb4ff989": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_c946edf026dbdb4ff989_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_c946edf026dbdb4ff989_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "shared"],
            ["buffer", "gemm_slice"],
            ["buffer", "out"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "my_col_begin"],
            ["parameter", "my_cols"],
            ["parameter", "gemm_plane_stride"],
            ["parameter", "num_gemm_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "6cdd84a110f889c322369e6cd111e5a28bd118f2c311eda089b9a84605d8dd4d",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_d0a751f069e7886b85ce": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_d0a751f069e7886b85ce_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_d0a751f069e7886b85ce_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "8b83e53e469f67e65d53c85a8b1d4a9faf43843a19ef30e28f0987fd16371086",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_d5dd99ea2f13462deec5": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_d5dd99ea2f13462deec5_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_d5dd99ea2f13462deec5_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "shared"],
            ["buffer", "gemm_slice"],
            ["buffer", "out"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "my_col_begin"],
            ["parameter", "my_cols"],
            ["parameter", "gemm_plane_stride"],
            ["parameter", "num_gemm_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "5ef908b8a547e3a7057246ae6939bc4ec461ac8de851bec546712e66baebc66e",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_d87ce0e07959bc6a9a94": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_d87ce0e07959bc6a9a94_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_100a/cake_kimi_k3_tp12_tail_d87ce0e07959bc6a9a94_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "d4872921eacf7471cdc034fe79874774e9e1bbfa0c2af49e748456b481b5e107",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_d94322a9f13a2d1a870d": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_d94322a9f13a2d1a870d_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_d94322a9f13a2d1a870d_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "dcd954df139e16cc0f9cdbcd9a8b1087923a99c96b1e6609cfa8730fd9069d45",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
    "cake_kimi_k3_tp12_tail_d99c3ecd0edd0ba892b0": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_d99c3ecd0edd0ba892b0_kernel.cu",
            "cake_kimi_k3_tp12_tail/sm_103a/cake_kimi_k3_tp12_tail_d99c3ecd0edd0ba892b0_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "7a57a1d40b3f3b4188db5fa3f41c275d16de315c863b5b8eba014b79bd8c9d21",
        "tma_workspace_bytes": 0,
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
    },
}

KERNELS: dict[str, dict[str, str]] = {
    "sm_100a": {
        "k1_oneshot:r0": "cake_kimi_k3_tp12_tail_478739c83690aaedd5e5",
        "k1_oneshot:r1": "cake_kimi_k3_tp12_tail_5659a33e7f6575b3b6b8",
        "k1_oneshot:r10": "cake_kimi_k3_tp12_tail_38a23ee0f849bd40a266",
        "k1_oneshot:r11": "cake_kimi_k3_tp12_tail_059199fc13dd03bacb12",
        "k1_oneshot:r2": "cake_kimi_k3_tp12_tail_a6e3998e39ed767bb130",
        "k1_oneshot:r3": "cake_kimi_k3_tp12_tail_bff3b68ca7b2da0a79f0",
        "k1_oneshot:r4": "cake_kimi_k3_tp12_tail_734d3ec84d75fbfad81d",
        "k1_oneshot:r5": "cake_kimi_k3_tp12_tail_8621affbb892d25c21c4",
        "k1_oneshot:r6": "cake_kimi_k3_tp12_tail_b276b8216af1ef607f85",
        "k1_oneshot:r7": "cake_kimi_k3_tp12_tail_d87ce0e07959bc6a9a94",
        "k1_oneshot:r8": "cake_kimi_k3_tp12_tail_4d472a0c4bfd124ac918",
        "k1_oneshot:r9": "cake_kimi_k3_tp12_tail_854cb9b1586c72761b2f",
        "k1_twoshot:grouped": "cake_kimi_k3_tp12_tail_0bdace2db5c1ea58de22",
        "k1_twoshot:pinned": "cake_kimi_k3_tp12_tail_655245e4ea6a0566f7c2",
        "k3:grouped": "cake_kimi_k3_tp12_tail_c946edf026dbdb4ff989",
        "k3:pinned": "cake_kimi_k3_tp12_tail_7d065b2e62d8c1a3d15a",
    },
    "sm_103a": {
        "k1_oneshot:r0": "cake_kimi_k3_tp12_tail_3ffbf6f10c254327b6b4",
        "k1_oneshot:r1": "cake_kimi_k3_tp12_tail_9df6effe73530626c497",
        "k1_oneshot:r10": "cake_kimi_k3_tp12_tail_7de4539e58943861165d",
        "k1_oneshot:r11": "cake_kimi_k3_tp12_tail_d0a751f069e7886b85ce",
        "k1_oneshot:r2": "cake_kimi_k3_tp12_tail_2612fc0c046ff8f7e1d7",
        "k1_oneshot:r3": "cake_kimi_k3_tp12_tail_3d4ea8055e0c38defafe",
        "k1_oneshot:r4": "cake_kimi_k3_tp12_tail_d94322a9f13a2d1a870d",
        "k1_oneshot:r5": "cake_kimi_k3_tp12_tail_208f95020e43bfd6b028",
        "k1_oneshot:r6": "cake_kimi_k3_tp12_tail_1aa73d0e225f019716c9",
        "k1_oneshot:r7": "cake_kimi_k3_tp12_tail_5b0f2c34849ef87ac24b",
        "k1_oneshot:r8": "cake_kimi_k3_tp12_tail_b18646c92b839845eebe",
        "k1_oneshot:r9": "cake_kimi_k3_tp12_tail_d99c3ecd0edd0ba892b0",
        "k1_twoshot:grouped": "cake_kimi_k3_tp12_tail_3e44777e07d6a76cb70b",
        "k1_twoshot:pinned": "cake_kimi_k3_tp12_tail_c728b4dbc601934e79fc",
        "k3:grouped": "cake_kimi_k3_tp12_tail_afc049beb5340f185e43",
        "k3:pinned": "cake_kimi_k3_tp12_tail_d5dd99ea2f13462deec5",
    },
}

ARCHES = ("sm_100a", "sm_103a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
WORLD_SIZE = 12
POLL_SCHEDULES = ("grouped", "pinned")


def required_kernel_keys() -> tuple[str, ...]:
    """Every logical kernel the runtime can select on one architecture."""
    return (
        *(f"k1_oneshot:r{rank}" for rank in range(WORLD_SIZE)),
        *(f"k1_twoshot:{schedule}" for schedule in POLL_SCHEDULES),
        # the one-CTA-per-token K3 only below 256 tokens (grouped); K3-P owns every M >= 256
        "k3:grouped",
        *(f"k3_persist:{schedule}" for schedule in POLL_SCHEDULES),
        "k2_stream:n640",
        "k2_stream:n512",
        "k3_f32:grouped",
    )


def route_available(arch: str, required_keys: tuple[str, ...] = ()) -> bool:
    """True when ``arch`` is registered and carries every key in ``required_keys``."""
    table = KERNELS.get(arch)
    return table is not None and all(key in table for key in required_keys)


def kernel_module_name(arch: str, key: str) -> str:
    """Return the registered physical module for ``key`` on ``arch``."""
    table = KERNELS.get(arch)
    if table is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 TP12 tail programs for {arch} are not "
            "registered in this checkout yet (see flashinfer-ai/flashinfer#4542)"
        )
    name = table.get(key)
    if name is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 TP12 tail kernel {key!r} for {arch} is not "
            "registered in this checkout (see flashinfer-ai/flashinfer#4542)"
        )
    record = MODULES[name]
    if record["arch"] != arch:
        raise RuntimeError(
            f"registered module {name!r} is an {record['arch']} program bound to {arch}"
        )
    return name


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
def gen_cake_kimi_k3_tp12_tail_module(name: str):
    record = MODULES[name]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{name}_" + record["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *record["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_kimi_k3_tp12_tail_module(name: str):
    return gen_cake_kimi_k3_tp12_tail_module(name).build_and_load()
