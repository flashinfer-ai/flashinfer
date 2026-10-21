# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JIT registry and loader for the Cake GDN CP-prefill kernels.

One generated program (device source + thin TVM-FFI launcher) compiles
for every architecture it lists; the exact architecture flag set is
chosen per device and is part of the JIT library name, so two
architectures never share one cached library.  ``KERNELS`` maps each
logical kernel the host dispatches on to its program, one table per
architecture.  A loaded kernel is the program's TVM-FFI entry itself: the host
calls it positionally with the arguments of ``MODULES[program]["arg_plan"]``
(grid triple last); the launcher validates them, encodes its tensor maps by
value and derives the CP-prefill fast-division constants from the plain divisor
arguments, so no Python work sits between the host and the launch.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

GDNCPArch = Literal["sm_100a", "sm_103a", "sm_107a"]

# Filled mechanically by the Cake exporter from the complete production builds.
MODULES: dict[str, dict[str, Any]] = {
    "cake_gdn_cp_0988da242f8528c9fdec": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_0988da242f8528c9fdec_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_0988da242f8528c9fdec_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "alpha"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "initial_state"],
            ["buffer", "final_state"],
            ["buffer", "output"],
            ["parameter", "scale"],
            ["parameter", "normalize_qk"],
            ["parameter", "write_output"],
            ["parameter", "write_final_state"],
            ["parameter", "use_block64_final_state"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_v_heads"],
            ["parameter", "num_state_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_1284aa3c9e3f34eb5df5": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_1284aa3c9e3f34eb5df5_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_1284aa3c9e3f34eb5df5_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "source"],
            ["buffer", "state_indices"],
            ["buffer", "packed"],
            ["parameter", "pool_stride0"],
            ["parameter", "pool_stride1"],
            ["parameter", "pool_stride2"],
            ["parameter", "pool_stride3"],
            ["parameter", "num_heads"],
            ["parameter", "total_values"],
            ["parameter", "use_indices"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_28b5fee38bf967a68cd8": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_28b5fee38bf967a68cd8_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_28b5fee38bf967a68cd8_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "K"],
            ["buffer", "beta"],
            ["buffer", "t"],
            ["buffer", "cu_seqlens"],
            ["buffer", "cu_seqlens_i64"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "total_t_blocks"],
            ["parameter", "num_seqs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_29c857ce26576248b1c8": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_29c857ce26576248b1c8_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_29c857ce26576248b1c8_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "alpha"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "initial_state"],
            ["buffer", "final_state"],
            ["buffer", "output"],
            ["parameter", "scale"],
            ["parameter", "normalize_qk"],
            ["parameter", "write_output"],
            ["parameter", "write_final_state"],
            ["parameter", "use_block64_final_state"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_v_heads"],
            ["parameter", "num_state_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_107a"],
    },
    "cake_gdn_cp_2eb4e3c69f8d5339cba3": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_2eb4e3c69f8d5339cba3_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_2eb4e3c69f8d5339cba3_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "packed"],
            ["buffer", "state_indices"],
            ["buffer", "output"],
            ["parameter", "pool_stride0"],
            ["parameter", "pool_stride1"],
            ["parameter", "pool_stride2"],
            ["parameter", "pool_stride3"],
            ["parameter", "num_heads"],
            ["parameter", "total_values"],
            ["parameter", "use_indices"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_306612b8c71d26cd1d83": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_306612b8c71d26cd1d83_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_306612b8c71d26cd1d83_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "K"],
            ["buffer", "beta"],
            ["buffer", "t"],
            ["buffer", "cu_seqlens"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "total_t_blocks"],
            ["parameter", "num_seqs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_31a251fd8cbae7fb9d60": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_31a251fd8cbae7fb9d60_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_31a251fd8cbae7fb9d60_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "T"],
            ["tma_buffer", "O"],
            ["buffer", "alpha"],
            ["buffer", "cu_seqlens"],
            ["buffer", "fixed_state"],
            ["buffer", "initial_state_workspace"],
            ["buffer", "tensormap_workspace"],
            ["parameter", "cp_chunk_len"],
            ["parameter", "source_cp_chunk_len"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_v_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_438c7804657ab305e23f": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_438c7804657ab305e23f_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_438c7804657ab305e23f_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "T"],
            ["tma_buffer", "O"],
            ["buffer", "alpha"],
            ["buffer", "cu_seqlens"],
            ["buffer", "fixed_state"],
            ["buffer", "initial_state_workspace"],
            ["buffer", "tensormap_workspace"],
            ["parameter", "cp_chunk_len"],
            ["parameter", "source_cp_chunk_len"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_v_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_496899c05a73707f3f72": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_496899c05a73707f3f72_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_496899c05a73707f3f72_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "source"],
            ["buffer", "state_indices"],
            ["buffer", "packed"],
            ["parameter", "pool_stride0"],
            ["parameter", "pool_stride1"],
            ["parameter", "pool_stride2"],
            ["parameter", "pool_stride3"],
            ["parameter", "num_heads"],
            ["parameter", "total_values"],
            ["parameter", "use_indices"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_4cce14a6b72822cb9ff5": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_4cce14a6b72822cb9ff5_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_4cce14a6b72822cb9ff5_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "packed"],
            ["buffer", "state_indices"],
            ["buffer", "output"],
            ["parameter", "pool_stride0"],
            ["parameter", "pool_stride1"],
            ["parameter", "pool_stride2"],
            ["parameter", "pool_stride3"],
            ["parameter", "num_heads"],
            ["parameter", "total_values"],
            ["parameter", "use_indices"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_510a0627fc99f6b9429c": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_510a0627fc99f6b9429c_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_510a0627fc99f6b9429c_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "alpha"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "initial_state"],
            ["buffer", "final_state"],
            ["buffer", "output"],
            ["parameter", "scale"],
            ["parameter", "normalize_qk"],
            ["parameter", "write_output"],
            ["parameter", "write_final_state"],
            ["parameter", "use_block64_final_state"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_v_heads"],
            ["parameter", "num_state_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_58df50f20a8e6a75e788": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_58df50f20a8e6a75e788_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_58df50f20a8e6a75e788_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "q_normalized"],
            ["buffer", "k_normalized"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_k_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_5e8583a983c7aa197239": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_5e8583a983c7aa197239_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_5e8583a983c7aa197239_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "packed"],
            ["buffer", "state_indices"],
            ["buffer", "output"],
            ["parameter", "pool_stride0"],
            ["parameter", "pool_stride1"],
            ["parameter", "pool_stride2"],
            ["parameter", "pool_stride3"],
            ["parameter", "num_heads"],
            ["parameter", "total_values"],
            ["parameter", "use_indices"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_65642b3b1c2872ac7959": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_65642b3b1c2872ac7959_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_65642b3b1c2872ac7959_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "T"],
            ["tma_buffer", "O"],
            ["buffer", "alpha"],
            ["buffer", "cu_seqlens"],
            ["buffer", "fixed_state"],
            ["buffer", "initial_state_workspace"],
            ["buffer", "tensormap_workspace"],
            ["parameter", "cp_chunk_len"],
            ["parameter", "source_cp_chunk_len"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_v_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_6856273b89c0049e09b9": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_6856273b89c0049e09b9_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_6856273b89c0049e09b9_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "T"],
            ["tma_buffer", "O"],
            ["buffer", "alpha"],
            ["buffer", "cu_seqlens"],
            ["buffer", "fixed_state"],
            ["buffer", "initial_state_workspace"],
            ["buffer", "tensormap_workspace"],
            ["parameter", "cp_chunk_len"],
            ["parameter", "source_cp_chunk_len"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_v_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_107a"],
    },
    "cake_gdn_cp_6cb33987eab2fdd2a40a": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_6cb33987eab2fdd2a40a_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_6cb33987eab2fdd2a40a_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "packed"],
            ["buffer", "state_indices"],
            ["buffer", "output"],
            ["parameter", "pool_stride0"],
            ["parameter", "pool_stride1"],
            ["parameter", "pool_stride2"],
            ["parameter", "pool_stride3"],
            ["parameter", "num_heads"],
            ["parameter", "total_values"],
            ["parameter", "use_indices"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_72564a892ad77a80bc74": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_72564a892ad77a80bc74_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_72564a892ad77a80bc74_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "local_transfer"],
            ["tma_buffer", "local_state"],
            ["buffer", "initial_state"],
            ["buffer", "initial_state_workspace"],
            ["buffer", "fixed_state"],
            ["buffer", "output_state"],
            ["buffer", "cu_seqlens"],
            ["parameter", "chunk_len"],
            ["parameter", "total_cp_chunks"],
            ["parameter", "num_seqs"],
            ["parameter", "num_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_7c18fb3d6e06a4d25a64": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_7c18fb3d6e06a4d25a64_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_7c18fb3d6e06a4d25a64_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "T"],
            ["buffer", "alpha"],
            ["buffer", "local_transfer"],
            ["buffer", "local_state"],
            ["buffer", "cu_seqlens"],
            ["parameter", "chunk_len"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_v_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "total_cp_chunks"],
            ["parameter", "num_seqs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_107a"],
    },
    "cake_gdn_cp_862720015e8760a08ccc": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_862720015e8760a08ccc_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_862720015e8760a08ccc_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "source"],
            ["buffer", "state_indices"],
            ["buffer", "packed"],
            ["parameter", "pool_stride0"],
            ["parameter", "pool_stride1"],
            ["parameter", "pool_stride2"],
            ["parameter", "pool_stride3"],
            ["parameter", "num_heads"],
            ["parameter", "total_values"],
            ["parameter", "use_indices"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_86f4471fb290c9489313": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_86f4471fb290c9489313_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_86f4471fb290c9489313_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "T"],
            ["tma_buffer", "O"],
            ["buffer", "alpha"],
            ["buffer", "cu_seqlens"],
            ["buffer", "fixed_state"],
            ["buffer", "initial_state_workspace"],
            ["buffer", "tensormap_workspace"],
            ["parameter", "cp_chunk_len"],
            ["parameter", "source_cp_chunk_len"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_v_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_8d18c9a7d634c6df6e7e": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_8d18c9a7d634c6df6e7e_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_8d18c9a7d634c6df6e7e_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "source"],
            ["buffer", "state_indices"],
            ["buffer", "packed"],
            ["parameter", "pool_stride0"],
            ["parameter", "pool_stride1"],
            ["parameter", "pool_stride2"],
            ["parameter", "pool_stride3"],
            ["parameter", "num_heads"],
            ["parameter", "total_values"],
            ["parameter", "use_indices"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_902c2149cbdfd8f74a79": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_902c2149cbdfd8f74a79_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_902c2149cbdfd8f74a79_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "T"],
            ["tma_buffer", "O"],
            ["buffer", "alpha"],
            ["buffer", "cu_seqlens"],
            ["buffer", "fixed_state"],
            ["buffer", "initial_state_workspace"],
            ["buffer", "tensormap_workspace"],
            ["parameter", "cp_chunk_len"],
            ["parameter", "source_cp_chunk_len"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_v_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_107a"],
    },
    "cake_gdn_cp_91580fabfb9eae4b9cb6": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_91580fabfb9eae4b9cb6_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_91580fabfb9eae4b9cb6_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "packed"],
            ["buffer", "state_indices"],
            ["buffer", "output"],
            ["parameter", "pool_stride0"],
            ["parameter", "pool_stride1"],
            ["parameter", "pool_stride2"],
            ["parameter", "pool_stride3"],
            ["parameter", "num_heads"],
            ["parameter", "total_values"],
            ["parameter", "use_indices"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_a0c0fe113dc3d4a2c02d": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_a0c0fe113dc3d4a2c02d_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_a0c0fe113dc3d4a2c02d_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "local_transfer"],
            ["tma_buffer", "local_state"],
            ["buffer", "initial_state"],
            ["buffer", "initial_state_workspace"],
            ["buffer", "fixed_state"],
            ["buffer", "output_state"],
            ["buffer", "cu_seqlens"],
            ["parameter", "chunk_len"],
            ["parameter", "total_cp_chunks"],
            ["parameter", "num_seqs"],
            ["parameter", "num_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_a3c89471a34205cf5fa6": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_a3c89471a34205cf5fa6_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_a3c89471a34205cf5fa6_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "source"],
            ["buffer", "state_indices"],
            ["buffer", "packed"],
            ["parameter", "pool_stride0"],
            ["parameter", "pool_stride1"],
            ["parameter", "pool_stride2"],
            ["parameter", "pool_stride3"],
            ["parameter", "num_heads"],
            ["parameter", "total_values"],
            ["parameter", "use_indices"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_b89a7ff12a25db400df8": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_b89a7ff12a25db400df8_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_b89a7ff12a25db400df8_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "T"],
            ["buffer", "alpha"],
            ["buffer", "local_transfer"],
            ["buffer", "local_state"],
            ["buffer", "cu_seqlens"],
            ["parameter", "chunk_len"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_v_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "total_cp_chunks"],
            ["parameter", "num_seqs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_bbdd97c3096cd360b467": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_bbdd97c3096cd360b467_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_bbdd97c3096cd360b467_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "packed"],
            ["buffer", "state_indices"],
            ["buffer", "output"],
            ["parameter", "pool_stride0"],
            ["parameter", "pool_stride1"],
            ["parameter", "pool_stride2"],
            ["parameter", "pool_stride3"],
            ["parameter", "num_heads"],
            ["parameter", "total_values"],
            ["parameter", "use_indices"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_c1c2bbbd5cd640e1becb": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_c1c2bbbd5cd640e1becb_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_c1c2bbbd5cd640e1becb_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "T"],
            ["tma_buffer", "O"],
            ["buffer", "alpha"],
            ["buffer", "cu_seqlens"],
            ["buffer", "fixed_state"],
            ["buffer", "initial_state_workspace"],
            ["buffer", "tensormap_workspace"],
            ["parameter", "cp_chunk_len"],
            ["parameter", "source_cp_chunk_len"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_v_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_c2f66f9e7a1e3b12cabd": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_c2f66f9e7a1e3b12cabd_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_c2f66f9e7a1e3b12cabd_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "q_normalized"],
            ["buffer", "k_normalized"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_k_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_d5827d0f1ecedc535e35": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_d5827d0f1ecedc535e35_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_d5827d0f1ecedc535e35_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "K"],
            ["buffer", "beta"],
            ["buffer", "t"],
            ["buffer", "cu_seqlens"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "total_t_blocks"],
            ["parameter", "num_seqs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_dc9fd29403170907f3a6": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_dc9fd29403170907f3a6_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_dc9fd29403170907f3a6_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "local_transfer"],
            ["buffer", "local_state"],
            ["buffer", "initial_state"],
            ["buffer", "initial_state_workspace"],
            ["buffer", "fixed_state"],
            ["buffer", "output_state"],
            ["buffer", "cu_seqlens"],
            ["parameter", "chunk_len"],
            ["parameter", "total_cp_chunks"],
            ["parameter", "num_seqs"],
            ["parameter", "num_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_e19f8702607bd4d78e49": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_e19f8702607bd4d78e49_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_e19f8702607bd4d78e49_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "T"],
            ["buffer", "alpha"],
            ["buffer", "local_transfer"],
            ["buffer", "local_state"],
            ["buffer", "cu_seqlens"],
            ["parameter", "chunk_len"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_v_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "total_cp_chunks"],
            ["parameter", "num_seqs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_e22ac3051c70e802f177": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_e22ac3051c70e802f177_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_e22ac3051c70e802f177_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "source"],
            ["buffer", "state_indices"],
            ["buffer", "packed"],
            ["parameter", "pool_stride0"],
            ["parameter", "pool_stride1"],
            ["parameter", "pool_stride2"],
            ["parameter", "pool_stride3"],
            ["parameter", "num_heads"],
            ["parameter", "total_values"],
            ["parameter", "use_indices"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_gdn_cp_f297019c7bec02dd6183": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_f297019c7bec02dd6183_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_f297019c7bec02dd6183_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "K"],
            ["buffer", "beta"],
            ["buffer", "t"],
            ["buffer", "cu_seqlens"],
            ["buffer", "cu_seqlens_i64"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "total_t_blocks"],
            ["parameter", "num_seqs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_ffe233cc2d497efa304a": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_ffe233cc2d497efa304a_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_ffe233cc2d497efa304a_binding.cu",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "T"],
            ["tma_buffer", "O"],
            ["buffer", "alpha"],
            ["buffer", "cu_seqlens"],
            ["buffer", "fixed_state"],
            ["buffer", "initial_state_workspace"],
            ["buffer", "tensormap_workspace"],
            ["parameter", "cp_chunk_len"],
            ["parameter", "source_cp_chunk_len"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_k_heads"],
            ["parameter", "num_v_heads"],
            ["parameter", "num_sab_heads"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_103a"],
    },
}
KERNELS: dict[str, Any] = {
    "sm_100a": {
        "cp_prefill": "cake_gdn_cp_438c7804657ab305e23f",
        "cp_prefill_bf16": "cake_gdn_cp_65642b3b1c2872ac7959",
        "cp_prefill_equal_head": "cake_gdn_cp_c1c2bbbd5cd640e1becb",
        "cp_prefill_generic": "cake_gdn_cp_31a251fd8cbae7fb9d60",
        "cp_prefill_generic_bf16": "cake_gdn_cp_86f4471fb290c9489313",
        "mn_precompute": "cake_gdn_cp_e19f8702607bd4d78e49",
        "mn_precompute_bf16": "cake_gdn_cp_b89a7ff12a25db400df8",
        "normalized_final_state": "cake_gdn_cp_510a0627fc99f6b9429c",
        "normalized_final_state_bf16": "cake_gdn_cp_0988da242f8528c9fdec",
        "qk_norm": "cake_gdn_cp_58df50f20a8e6a75e788",
        "qk_norm_bf16": "cake_gdn_cp_c2f66f9e7a1e3b12cabd",
        "state_fixup_simt_row4": "cake_gdn_cp_dc9fd29403170907f3a6",
        "state_fixup_utcmma128": "cake_gdn_cp_a0c0fe113dc3d4a2c02d",
        "state_fixup_utcmma64": "cake_gdn_cp_72564a892ad77a80bc74",
        "state_gather_bf16": "cake_gdn_cp_a3c89471a34205cf5fa6",
        "state_gather_bf16_int64": "cake_gdn_cp_8d18c9a7d634c6df6e7e",
        "state_gather_fp16": "cake_gdn_cp_1284aa3c9e3f34eb5df5",
        "state_gather_fp16_int64": "cake_gdn_cp_e22ac3051c70e802f177",
        "state_gather_fp32": "cake_gdn_cp_862720015e8760a08ccc",
        "state_gather_fp32_int64": "cake_gdn_cp_496899c05a73707f3f72",
        "state_scatter_bf16": "cake_gdn_cp_4cce14a6b72822cb9ff5",
        "state_scatter_bf16_int64": "cake_gdn_cp_5e8583a983c7aa197239",
        "state_scatter_fp16": "cake_gdn_cp_6cb33987eab2fdd2a40a",
        "state_scatter_fp16_int64": "cake_gdn_cp_bbdd97c3096cd360b467",
        "state_scatter_fp32": "cake_gdn_cp_2eb4e3c69f8d5339cba3",
        "state_scatter_fp32_int64": "cake_gdn_cp_91580fabfb9eae4b9cb6",
        "t_precompute": "cake_gdn_cp_d5827d0f1ecedc535e35",
        "t_precompute_bf16": "cake_gdn_cp_306612b8c71d26cd1d83",
        "t_precompute_bf16_cu32": "cake_gdn_cp_f297019c7bec02dd6183",
        "t_precompute_cu32": "cake_gdn_cp_28b5fee38bf967a68cd8",
    },
    "sm_103a": {
        "cp_prefill": "cake_gdn_cp_438c7804657ab305e23f",
        "cp_prefill_bf16": "cake_gdn_cp_65642b3b1c2872ac7959",
        "cp_prefill_equal_head": "cake_gdn_cp_c1c2bbbd5cd640e1becb",
        "cp_prefill_equal_head_h32": "cake_gdn_cp_ffe233cc2d497efa304a",
        "cp_prefill_generic": "cake_gdn_cp_31a251fd8cbae7fb9d60",
        "cp_prefill_generic_bf16": "cake_gdn_cp_86f4471fb290c9489313",
        "mn_precompute": "cake_gdn_cp_e19f8702607bd4d78e49",
        "mn_precompute_bf16": "cake_gdn_cp_b89a7ff12a25db400df8",
        "normalized_final_state": "cake_gdn_cp_510a0627fc99f6b9429c",
        "normalized_final_state_bf16": "cake_gdn_cp_0988da242f8528c9fdec",
        "qk_norm": "cake_gdn_cp_58df50f20a8e6a75e788",
        "qk_norm_bf16": "cake_gdn_cp_c2f66f9e7a1e3b12cabd",
        "state_fixup_simt_row4": "cake_gdn_cp_dc9fd29403170907f3a6",
        "state_fixup_utcmma128": "cake_gdn_cp_a0c0fe113dc3d4a2c02d",
        "state_fixup_utcmma64": "cake_gdn_cp_72564a892ad77a80bc74",
        "state_gather_bf16": "cake_gdn_cp_a3c89471a34205cf5fa6",
        "state_gather_bf16_int64": "cake_gdn_cp_8d18c9a7d634c6df6e7e",
        "state_gather_fp16": "cake_gdn_cp_1284aa3c9e3f34eb5df5",
        "state_gather_fp16_int64": "cake_gdn_cp_e22ac3051c70e802f177",
        "state_gather_fp32": "cake_gdn_cp_862720015e8760a08ccc",
        "state_gather_fp32_int64": "cake_gdn_cp_496899c05a73707f3f72",
        "state_scatter_bf16": "cake_gdn_cp_4cce14a6b72822cb9ff5",
        "state_scatter_bf16_int64": "cake_gdn_cp_5e8583a983c7aa197239",
        "state_scatter_fp16": "cake_gdn_cp_6cb33987eab2fdd2a40a",
        "state_scatter_fp16_int64": "cake_gdn_cp_bbdd97c3096cd360b467",
        "state_scatter_fp32": "cake_gdn_cp_2eb4e3c69f8d5339cba3",
        "state_scatter_fp32_int64": "cake_gdn_cp_91580fabfb9eae4b9cb6",
        "t_precompute": "cake_gdn_cp_d5827d0f1ecedc535e35",
        "t_precompute_bf16": "cake_gdn_cp_306612b8c71d26cd1d83",
        "t_precompute_bf16_cu32": "cake_gdn_cp_f297019c7bec02dd6183",
        "t_precompute_cu32": "cake_gdn_cp_28b5fee38bf967a68cd8",
    },
    "sm_107a": {
        "cp_prefill": "cake_gdn_cp_438c7804657ab305e23f",
        "cp_prefill_bf16": "cake_gdn_cp_6856273b89c0049e09b9",
        "cp_prefill_equal_head": "cake_gdn_cp_c1c2bbbd5cd640e1becb",
        "cp_prefill_generic": "cake_gdn_cp_31a251fd8cbae7fb9d60",
        "cp_prefill_generic_bf16": "cake_gdn_cp_902c2149cbdfd8f74a79",
        "mn_precompute": "cake_gdn_cp_e19f8702607bd4d78e49",
        "mn_precompute_bf16": "cake_gdn_cp_7c18fb3d6e06a4d25a64",
        "normalized_final_state": "cake_gdn_cp_510a0627fc99f6b9429c",
        "normalized_final_state_bf16": "cake_gdn_cp_29c857ce26576248b1c8",
        "qk_norm": "cake_gdn_cp_58df50f20a8e6a75e788",
        "qk_norm_bf16": "cake_gdn_cp_c2f66f9e7a1e3b12cabd",
        "state_fixup_simt_row4": "cake_gdn_cp_dc9fd29403170907f3a6",
        "state_gather_bf16": "cake_gdn_cp_a3c89471a34205cf5fa6",
        "state_gather_bf16_int64": "cake_gdn_cp_8d18c9a7d634c6df6e7e",
        "state_gather_fp16": "cake_gdn_cp_1284aa3c9e3f34eb5df5",
        "state_gather_fp16_int64": "cake_gdn_cp_e22ac3051c70e802f177",
        "state_gather_fp32": "cake_gdn_cp_862720015e8760a08ccc",
        "state_gather_fp32_int64": "cake_gdn_cp_496899c05a73707f3f72",
        "state_scatter_bf16": "cake_gdn_cp_4cce14a6b72822cb9ff5",
        "state_scatter_bf16_int64": "cake_gdn_cp_5e8583a983c7aa197239",
        "state_scatter_fp16": "cake_gdn_cp_6cb33987eab2fdd2a40a",
        "state_scatter_fp16_int64": "cake_gdn_cp_bbdd97c3096cd360b467",
        "state_scatter_fp32": "cake_gdn_cp_2eb4e3c69f8d5339cba3",
        "state_scatter_fp32_int64": "cake_gdn_cp_91580fabfb9eae4b9cb6",
        "t_precompute": "cake_gdn_cp_d5827d0f1ecedc535e35",
        "t_precompute_bf16": "cake_gdn_cp_306612b8c71d26cd1d83",
    },
}

_ARCH_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}


def _source_path(relative: str) -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / relative
    if installed.is_file():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / relative
    if checkout.is_file():
        return checkout
    raise FileNotFoundError(
        f"GDN CP-prefill source {relative!r} was not found under {installed.parent} or {checkout.parent}"
    )


def _include_dirs() -> list[Path]:
    include = jit_env.FLASHINFER_INCLUDE_DIR
    if not include.is_dir():
        include = Path(__file__).resolve().parents[2] / "include"
    return [_source_path("tvm_ffi_utils.h").parent, include]


def program_for(name: str, arch: GDNCPArch) -> str:
    """Program implementing logical kernel ``name`` on ``arch``."""

    try:
        program = KERNELS[arch][name]
    except KeyError as error:
        raise ValueError(
            f"unknown GDN CP-prefill kernel {name!r} for {arch}"
        ) from error
    if arch not in MODULES[program]["arches"]:
        raise ValueError(f"GDN CP-prefill program {program!r} does not support {arch}")
    return program


def has_gdn_cp_kernel(name: str, arch: GDNCPArch) -> bool:
    """Whether logical kernel ``name`` has a program for ``arch``.

    Same per-architecture ``KERNELS`` lookup as :func:`program_for`, answering
    ``False`` where that raises, so the host can select an optional program
    (the cu32 Stage-1 variants) only on the architectures that ship it.
    """

    try:
        program_for(name, arch)
    except ValueError:
        return False
    return True


@functools.cache
def gen_gdn_cp_spec(program: str, arch: GDNCPArch) -> JitSpec:
    record = MODULES[program]
    if arch not in record["arches"]:
        raise ValueError(f"GDN CP-prefill program {program!r} does not support {arch}")
    sources = [_source_path(path) for path in record["sources"]]
    # The shipped kernels were compiled with plain ``nvcc -O3`` (IEEE division/sqrt, no FTZ); the
    # JIT default ``-use_fast_math`` changes the FP32 final-state recurrence bitwise, so it stays off.
    return gen_jit_spec(
        name=f"cake_gdn_cp_{program}_{arch}",
        sources=sources,
        extra_cflags=["-std=c++17", "-DNDEBUG", "-O3"],
        extra_cuda_cflags=[
            "-std=c++17",
            "-DNDEBUG",
            "-O3",
            *_ARCH_FLAGS[arch],
        ],
        extra_include_paths=[sources[0].parent, *_include_dirs()],
        needs_device_linking=True,
        use_fast_math=False,
    )


@functools.cache
def load_gdn_cp_kernel_module(program: str, arch: GDNCPArch):
    return gen_gdn_cp_spec(program, arch).build_and_load()


@functools.cache
def load_gdn_cp_kernel(name: str, arch: GDNCPArch):
    """Compile (once per process) and return the launcher of one logical kernel.

    The launcher is the program's TVM-FFI entry; call it positionally with the
    arguments of ``MODULES[program]["arg_plan"]``, grid triple last.
    """

    if arch not in _ARCH_FLAGS:
        raise ValueError(f"unsupported GDN CP-prefill architecture: {arch!r}")
    program = program_for(name, arch)
    launcher = getattr(
        load_gdn_cp_kernel_module(program, arch), MODULES[program]["ffi_entry"]
    )
    return launcher


__all__ = [
    "GDNCPArch",
    "KERNELS",
    "MODULES",
    "gen_gdn_cp_spec",
    "has_gdn_cp_kernel",
    "load_gdn_cp_kernel",
    "load_gdn_cp_kernel_module",
    "program_for",
]
