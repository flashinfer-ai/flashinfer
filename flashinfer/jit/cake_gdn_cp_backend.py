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

One generated program (device source + thin TVM-FFI launcher) compiles for every
architecture it lists; the exact architecture flag set is chosen per device and
is part of the JIT library name, so two architectures never share one cached
library.  ``KERNELS`` maps each logical kernel the host dispatches on to its
program (one table when both architectures agree, otherwise one table per
architecture).  A loaded kernel is the program's TVM-FFI entry itself: the host
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
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

GDNCPArch = Literal["sm_100a", "sm_103a"]

# Filled mechanically by the Cake exporter from the complete production builds.
MODULES: dict[str, dict[str, Any]] = {
    "cake_gdn_cp_0d2152b1fca65e744033": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_0d2152b1fca65e744033_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_0d2152b1fca65e744033_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_0d2152b1fca65e744033",
        "compile_flags": [],
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
        "launch": {
            "block": [128, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 24960,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_11cf3c609ca59a007e96": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_11cf3c609ca59a007e96_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_11cf3c609ca59a007e96_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_11cf3c609ca59a007e96",
        "compile_flags": [],
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
        "launch": {
            "block": [128, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 640,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_13ae1311f2f0ee1b7813": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_13ae1311f2f0ee1b7813_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_13ae1311f2f0ee1b7813_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_13ae1311f2f0ee1b7813",
        "compile_flags": [],
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
        "launch": {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 0,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_173b7a01d35b79f84731": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_173b7a01d35b79f84731_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_173b7a01d35b79f84731_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_173b7a01d35b79f84731",
        "compile_flags": [],
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
        "launch": {
            "block": [32, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 0,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_182ccae8c184a6945d27": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_182ccae8c184a6945d27_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_182ccae8c184a6945d27_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_182ccae8c184a6945d27",
        "compile_flags": [],
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
        "launch": {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 0,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_29539d13a8dd2cd0568f": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_29539d13a8dd2cd0568f_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_29539d13a8dd2cd0568f_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_29539d13a8dd2cd0568f",
        "compile_flags": [],
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
        "launch": {
            "block": [384, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 159744,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_29c857ce26576248b1c8": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_29c857ce26576248b1c8_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_29c857ce26576248b1c8_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_29c857ce26576248b1c8",
        "compile_flags": [],
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
        "launch": {
            "block": [128, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 82560,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_3c808f2c5dac8046844b": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_3c808f2c5dac8046844b_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_3c808f2c5dac8046844b_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_3c808f2c5dac8046844b",
        "compile_flags": [],
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
        "launch": {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 164864,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_3df53d2d65fb731b7e09": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_3df53d2d65fb731b7e09_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_3df53d2d65fb731b7e09_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_3df53d2d65fb731b7e09",
        "compile_flags": [],
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
        "launch": {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 0,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_427bbbbbb5253da4a9c6": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_427bbbbbb5253da4a9c6_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_427bbbbbb5253da4a9c6_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_427bbbbbb5253da4a9c6",
        "compile_flags": [],
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
        "launch": {
            "block": [384, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 224768,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_103a"],
    },
    "cake_gdn_cp_4913bea45166d7c45234": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_4913bea45166d7c45234_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_4913bea45166d7c45234_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_4913bea45166d7c45234",
        "compile_flags": [],
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
        "launch": {
            "block": [128, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 2048,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_4e5f37aac06abc7b42a0": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_4e5f37aac06abc7b42a0_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_4e5f37aac06abc7b42a0_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_4e5f37aac06abc7b42a0",
        "compile_flags": [],
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
        "launch": {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 0,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_5b18710c217aae02ee0e": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_5b18710c217aae02ee0e_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_5b18710c217aae02ee0e_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_5b18710c217aae02ee0e",
        "compile_flags": [],
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
        "launch": {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 0,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_6856273b89c0049e09b9": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_6856273b89c0049e09b9_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_6856273b89c0049e09b9_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_6856273b89c0049e09b9",
        "compile_flags": [],
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
        "launch": {
            "block": [384, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 224768,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_6a6a9d040a98ccea86c9": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_6a6a9d040a98ccea86c9_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_6a6a9d040a98ccea86c9_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_6a6a9d040a98ccea86c9",
        "compile_flags": [],
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
        "launch": {
            "block": [128, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 24960,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_6d92c5b1c26421eb420f": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_6d92c5b1c26421eb420f_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_6d92c5b1c26421eb420f_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_6d92c5b1c26421eb420f",
        "compile_flags": [],
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
        "launch": {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 0,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_6e3ae3e850e9989ccdc2": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_6e3ae3e850e9989ccdc2_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_6e3ae3e850e9989ccdc2_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_6e3ae3e850e9989ccdc2",
        "compile_flags": [],
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
        "launch": {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 0,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_72e10fdf3b60fe467d88": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_72e10fdf3b60fe467d88_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_72e10fdf3b60fe467d88_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_72e10fdf3b60fe467d88",
        "compile_flags": [],
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
        "launch": {
            "block": [384, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 224768,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_7c18fb3d6e06a4d25a64": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_7c18fb3d6e06a4d25a64_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_7c18fb3d6e06a4d25a64_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_7c18fb3d6e06a4d25a64",
        "compile_flags": [],
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
        "launch": {
            "block": [384, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 225280,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_8c895924c32f9c71dcf6": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_8c895924c32f9c71dcf6_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_8c895924c32f9c71dcf6_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_8c895924c32f9c71dcf6",
        "compile_flags": [],
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
        "launch": {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 0,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_902c2149cbdfd8f74a79": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_902c2149cbdfd8f74a79_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_902c2149cbdfd8f74a79_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_902c2149cbdfd8f74a79",
        "compile_flags": [],
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
        "launch": {
            "block": [384, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 224768,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_902cedff35fc12c2a357": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_902cedff35fc12c2a357_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_902cedff35fc12c2a357_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_902cedff35fc12c2a357",
        "compile_flags": [],
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
        "launch": {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 0,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_acaf46daf1b4b166f7f2": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_acaf46daf1b4b166f7f2_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_acaf46daf1b4b166f7f2_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_acaf46daf1b4b166f7f2",
        "compile_flags": [],
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
        "launch": {
            "block": [384, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 224768,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_bb7a8a4735606de585d8": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_bb7a8a4735606de585d8_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_bb7a8a4735606de585d8_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_bb7a8a4735606de585d8",
        "compile_flags": [],
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
        "launch": {
            "block": [32, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 0,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_d4736ea62c7fb5df0e35": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_d4736ea62c7fb5df0e35_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_d4736ea62c7fb5df0e35_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_d4736ea62c7fb5df0e35",
        "compile_flags": [],
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
        "launch": {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 0,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_d5727d042939d5b90479": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_d5727d042939d5b90479_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_d5727d042939d5b90479_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_d5727d042939d5b90479",
        "compile_flags": [],
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
        "launch": {
            "block": [384, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 224768,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_dd2c081134dd8b47073c": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_dd2c081134dd8b47073c_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_dd2c081134dd8b47073c_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_dd2c081134dd8b47073c",
        "compile_flags": [],
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
        "launch": {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 132096,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_ef19bc051e3664f9ba15": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_ef19bc051e3664f9ba15_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_ef19bc051e3664f9ba15_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_ef19bc051e3664f9ba15",
        "compile_flags": [],
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
        "launch": {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 0,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_gdn_cp_f3a9ede2eb4e48a3e107": {
        "sources": [
            "gdn/gdn_cp/cake_gdn_cp_f3a9ede2eb4e48a3e107_kernel.cu",
            "gdn/gdn_cp/cake_gdn_cp_f3a9ede2eb4e48a3e107_binding.cu",
        ],
        "kernel": "kernel_cake_gdn_cp_f3a9ede2eb4e48a3e107",
        "compile_flags": [],
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
        "launch": {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "dynamic_smem_bytes": 0,
            "cooperative": False,
            "use_pdl": False,
        },
        "arches": ["sm_100a", "sm_103a"],
    },
}
KERNELS: dict[str, Any] = {
    "sm_100a": {
        "cp_prefill": "cake_gdn_cp_d5727d042939d5b90479",
        "cp_prefill_bf16": "cake_gdn_cp_6856273b89c0049e09b9",
        "cp_prefill_checkpoint": "cake_gdn_cp_d5727d042939d5b90479",
        "cp_prefill_equal_head": "cake_gdn_cp_72e10fdf3b60fe467d88",
        "cp_prefill_equal_head_checkpoint": "cake_gdn_cp_72e10fdf3b60fe467d88",
        "cp_prefill_generic": "cake_gdn_cp_acaf46daf1b4b166f7f2",
        "cp_prefill_generic_bf16": "cake_gdn_cp_902c2149cbdfd8f74a79",
        "cp_prefill_generic_checkpoint": "cake_gdn_cp_acaf46daf1b4b166f7f2",
        "mn_precompute": "cake_gdn_cp_29539d13a8dd2cd0568f",
        "mn_precompute_bf16": "cake_gdn_cp_7c18fb3d6e06a4d25a64",
        "normalized_final_state": "cake_gdn_cp_11cf3c609ca59a007e96",
        "normalized_final_state_bf16": "cake_gdn_cp_29c857ce26576248b1c8",
        "qk_norm": "cake_gdn_cp_bb7a8a4735606de585d8",
        "qk_norm_bf16": "cake_gdn_cp_173b7a01d35b79f84731",
        "state_fixup_simt_row4": "cake_gdn_cp_4913bea45166d7c45234",
        "state_fixup_utcmma128": "cake_gdn_cp_dd2c081134dd8b47073c",
        "state_fixup_utcmma64": "cake_gdn_cp_3c808f2c5dac8046844b",
        "state_gather_bf16": "cake_gdn_cp_5b18710c217aae02ee0e",
        "state_gather_bf16_int64": "cake_gdn_cp_f3a9ede2eb4e48a3e107",
        "state_gather_fp16": "cake_gdn_cp_d4736ea62c7fb5df0e35",
        "state_gather_fp16_int64": "cake_gdn_cp_902cedff35fc12c2a357",
        "state_gather_fp32": "cake_gdn_cp_ef19bc051e3664f9ba15",
        "state_gather_fp32_int64": "cake_gdn_cp_13ae1311f2f0ee1b7813",
        "state_scatter_bf16": "cake_gdn_cp_6e3ae3e850e9989ccdc2",
        "state_scatter_bf16_int64": "cake_gdn_cp_6d92c5b1c26421eb420f",
        "state_scatter_fp16": "cake_gdn_cp_8c895924c32f9c71dcf6",
        "state_scatter_fp16_int64": "cake_gdn_cp_4e5f37aac06abc7b42a0",
        "state_scatter_fp32": "cake_gdn_cp_3df53d2d65fb731b7e09",
        "state_scatter_fp32_int64": "cake_gdn_cp_182ccae8c184a6945d27",
        "t_precompute": "cake_gdn_cp_6a6a9d040a98ccea86c9",
        "t_precompute_bf16": "cake_gdn_cp_0d2152b1fca65e744033",
        "t_precompute_gb300_hv48_min6": "cake_gdn_cp_6a6a9d040a98ccea86c9",
    },
    "sm_103a": {
        "cp_prefill": "cake_gdn_cp_d5727d042939d5b90479",
        "cp_prefill_bf16": "cake_gdn_cp_6856273b89c0049e09b9",
        "cp_prefill_checkpoint": "cake_gdn_cp_d5727d042939d5b90479",
        "cp_prefill_equal_head": "cake_gdn_cp_72e10fdf3b60fe467d88",
        "cp_prefill_equal_head_checkpoint": "cake_gdn_cp_72e10fdf3b60fe467d88",
        "cp_prefill_equal_head_h32": "cake_gdn_cp_427bbbbbb5253da4a9c6",
        "cp_prefill_generic": "cake_gdn_cp_acaf46daf1b4b166f7f2",
        "cp_prefill_generic_bf16": "cake_gdn_cp_902c2149cbdfd8f74a79",
        "cp_prefill_generic_checkpoint": "cake_gdn_cp_acaf46daf1b4b166f7f2",
        "mn_precompute": "cake_gdn_cp_29539d13a8dd2cd0568f",
        "mn_precompute_bf16": "cake_gdn_cp_7c18fb3d6e06a4d25a64",
        "normalized_final_state": "cake_gdn_cp_11cf3c609ca59a007e96",
        "normalized_final_state_bf16": "cake_gdn_cp_29c857ce26576248b1c8",
        "qk_norm": "cake_gdn_cp_bb7a8a4735606de585d8",
        "qk_norm_bf16": "cake_gdn_cp_173b7a01d35b79f84731",
        "state_fixup_simt_row4": "cake_gdn_cp_4913bea45166d7c45234",
        "state_fixup_utcmma128": "cake_gdn_cp_dd2c081134dd8b47073c",
        "state_fixup_utcmma64": "cake_gdn_cp_3c808f2c5dac8046844b",
        "state_gather_bf16": "cake_gdn_cp_5b18710c217aae02ee0e",
        "state_gather_bf16_int64": "cake_gdn_cp_f3a9ede2eb4e48a3e107",
        "state_gather_fp16": "cake_gdn_cp_d4736ea62c7fb5df0e35",
        "state_gather_fp16_int64": "cake_gdn_cp_902cedff35fc12c2a357",
        "state_gather_fp32": "cake_gdn_cp_ef19bc051e3664f9ba15",
        "state_gather_fp32_int64": "cake_gdn_cp_13ae1311f2f0ee1b7813",
        "state_scatter_bf16": "cake_gdn_cp_6e3ae3e850e9989ccdc2",
        "state_scatter_bf16_int64": "cake_gdn_cp_6d92c5b1c26421eb420f",
        "state_scatter_fp16": "cake_gdn_cp_8c895924c32f9c71dcf6",
        "state_scatter_fp16_int64": "cake_gdn_cp_4e5f37aac06abc7b42a0",
        "state_scatter_fp32": "cake_gdn_cp_3df53d2d65fb731b7e09",
        "state_scatter_fp32_int64": "cake_gdn_cp_182ccae8c184a6945d27",
        "t_precompute": "cake_gdn_cp_6a6a9d040a98ccea86c9",
        "t_precompute_bf16": "cake_gdn_cp_0d2152b1fca65e744033",
        "t_precompute_gb300_hv48_min6": "cake_gdn_cp_6a6a9d040a98ccea86c9",
    },
}

_ARCH_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}


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

    table = (
        KERNELS.get(arch, KERNELS)
        if any(isinstance(v, dict) for v in KERNELS.values())
        else KERNELS
    )
    try:
        program = table[name]
    except KeyError as error:
        raise ValueError(
            f"unknown GDN CP-prefill kernel {name!r} for {arch}"
        ) from error
    if arch not in MODULES[program]["arches"]:
        raise ValueError(f"GDN CP-prefill program {program!r} does not support {arch}")
    return program


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
            *record["compile_flags"],
        ],
        extra_include_paths=[sources[0].parent, *_include_dirs()],
        needs_device_linking=True,
        use_fast_math=False,
    )


@functools.cache
def load_gdn_cp_kernel_module(program: str, arch: GDNCPArch):
    return gen_gdn_cp_spec(program, arch).build_and_load()


# Logical kernel name per launcher handed out by ``load_gdn_cp_kernel`` (the
# cache below keeps every launcher alive, so its identity is stable).
_KERNEL_NAMES: dict[int, str] = {}


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
    _KERNEL_NAMES[id(launcher)] = name
    return launcher


def kernel_name(launcher) -> str:
    """Logical kernel name of a launcher returned by ``load_gdn_cp_kernel``."""

    try:
        return _KERNEL_NAMES[id(launcher)]
    except KeyError:
        raise ValueError("not a launcher returned by load_gdn_cp_kernel") from None


def prepare_gdn_cp_kernel(name: str, arch: GDNCPArch, *, device) -> tuple[Any, tuple]:
    """Load one kernel for a launcher instance.

    Tensor maps travel by value in kernel parameters, so no per-instance
    descriptor workspace exists; the second element is always empty.
    """

    del device
    return load_gdn_cp_kernel(name, arch), ()


__all__ = [
    "GDNCPArch",
    "KERNELS",
    "MODULES",
    "gen_gdn_cp_spec",
    "kernel_name",
    "load_gdn_cp_kernel",
    "load_gdn_cp_kernel_module",
    "prepare_gdn_cp_kernel",
    "program_for",
]
