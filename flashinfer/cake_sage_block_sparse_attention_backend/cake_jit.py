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

Host plan and loader glue of the Cake Sage block-sparse attention kernels.

Three programs serve SM100 / SM103 (B200, B300 / GB300): the Sage-FP8
block-sparse attention forward (``attention``) and the two fused Sage-FP8
quantizers (``quantize_qk_vamax``, ``quantize_v``).  Twenty programs serve
SM120 (RTX PRO 6000 / RTX 5090): the ``mma.sync`` INT8-QK / FP8-PV kernel, one
per dispatched specialization of the five metadata flags (``sm120_module_key``
derives them from a call; the record's ``select_rule`` fixes them).  The SM120
libraries share one kernel source and one launcher, compiled per
specialization under the compile definitions ``build.json`` records.

Every record below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` carries the host plan only: the kernel's
``library`` stem, the architectures it is delivered for, its launch and grid
rules, the selection rule of an SM120 specialization and the kernel-parameter
-> logical-argument map.  What to compile -- sources, digests, nvcc flags -- is
recorded once in ``build.json`` next to this file; :mod:`.cake_build` serves
the library ``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder
compiled it and otherwise through FlashInfer's JIT from the same sources and
flags, verifying every source digest before the first build.

Tensor maps are kernel parameters: every binding encodes its ``CUtensorMap``
descriptors on the host per call and passes them by value, so a call writes no
device buffer besides its outputs, needs no workspace and captures into a CUDA
Graph as one kernel node.

The derived leaves of a record (``kernel_symbol``, ``launch``, ``arg_plan``,
``logical``) are rewritten by the exporter from the rendered binding; the
other fields are the host plan and never change.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import tvm_ffi

from . import cake_build

# Filled mechanically from the complete verified program bundle.
MODULES: dict[str, dict[str, Any]] = {
    "attention": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "out"],
            ["buffer", "lse"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "block_index"],
            ["buffer", "block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "num_kv_heads"],
            ["parameter", "q_blocks"],
            ["parameter", "k_blocks"],
            ["parameter", "k_scale_groups"],
            ["parameter", "capacity"],
            ["parameter", "block_sparse_num"],
            ["parameter", "has_block_nums"],
            ["parameter", "block_sizes_mode"],
            ["parameter", "v_scale_batch_stride"],
            ["parameter", "total_tiles"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "return_lse"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "capped_persistent",
            "ctas_per_sm": 1,
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_9884891b129880048274",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 77824,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_9884891b129880048274",
        "logical": {
            "block_index": "block_index",
            "block_nums": "block_nums",
            "block_sizes": "block_sizes",
            "block_sizes_mode": "block_sizes_mode",
            "block_sparse_num": "block_sparse_num",
            "capacity": "capacity",
            "has_block_nums": "has_block_nums",
            "k": "k",
            "k_blocks": "k_blocks",
            "k_scale": "k_scale",
            "k_scale_groups": "k_scale_groups",
            "lse": "lse",
            "num_heads": "num_heads",
            "num_kv_heads": "num_kv_heads",
            "out": "out",
            "q": "q",
            "q_blocks": "q_blocks",
            "q_scale": "q_scale",
            "return_lse": "return_lse",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale_log2": "softmax_scale_log2",
            "total_tiles": "total_tiles",
            "v": "v",
            "v_scale": "v_scale",
            "v_scale_batch_stride": "v_scale_batch_stride",
        },
        "role": "attention",
        "select_rule": None,
    },
    "quantize_qk_vamax": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [
            ["buffer", "q_in"],
            ["buffer", "k_in"],
            ["buffer", "v_in"],
            ["buffer", "q_out"],
            ["buffer", "k_out"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_amax"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "num_kv_heads"],
            ["parameter", "grid_heads"],
            ["parameter", "k_scale_groups"],
            ["parameter", "chunks_per_bh"],
            ["parameter", "q_chunks"],
            ["parameter", "k_chunks"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "batch_heads_chunks",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_ac8d3223970f7e4fddd8",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4224,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_ac8d3223970f7e4fddd8",
        "logical": {
            "chunks_per_bh": "chunks_per_bh",
            "grid_heads": "grid_heads",
            "k_chunks": "k_chunks",
            "k_in": "k_in",
            "k_out": "k_out",
            "k_scale": "k_scale",
            "k_scale_groups": "k_scale_groups",
            "num_heads": "num_heads",
            "num_kv_heads": "num_kv_heads",
            "q_chunks": "q_chunks",
            "q_in": "q_in",
            "q_out": "q_out",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "v_amax": "v_amax",
            "v_in": "v_in",
        },
        "role": "quantize_qk_vamax",
        "select_rule": None,
    },
    "quantize_v": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [
            ["buffer", "v_in"],
            ["buffer", "v_out"],
            ["buffer", "v_amax"],
            ["buffer", "v_scale"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "chunks_per_bh"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "batch_heads_chunks",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_ce646595859ababcf2bd",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_ce646595859ababcf2bd",
        "logical": {
            "chunks_per_bh": "chunks_per_bh",
            "num_heads": "num_heads",
            "seqlen_k": "seqlen_k",
            "v_amax": "v_amax",
            "v_in": "v_in",
            "v_out": "v_out",
            "v_scale": "v_scale",
        },
        "role": "quantize_v",
        "select_rule": None,
    },
    "attention_sm120_00000": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_c214204e0e4b13466e20",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_c214204e0e4b13466e20",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 0,
            "full_k64_tiles": 0,
            "uniform_nonempty": 0,
            "contiguous_block_indices": 0,
        },
    },
    "attention_sm120_00010": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_5e1d9bd7d208ae70cfc3",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_5e1d9bd7d208ae70cfc3",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 0,
            "full_k64_tiles": 0,
            "uniform_nonempty": 1,
            "contiguous_block_indices": 0,
        },
    },
    "attention_sm120_00011": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_883c6c3450cc823b2d9b",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_883c6c3450cc823b2d9b",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 0,
            "full_k64_tiles": 0,
            "uniform_nonempty": 1,
            "contiguous_block_indices": 1,
        },
    },
    "attention_sm120_00100": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_ac8c0628f84e77fb565d",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_ac8c0628f84e77fb565d",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 0,
            "full_k64_tiles": 1,
            "uniform_nonempty": 0,
            "contiguous_block_indices": 0,
        },
    },
    "attention_sm120_00110": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_0a00c8bb89c685b0c222",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_0a00c8bb89c685b0c222",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 0,
            "full_k64_tiles": 1,
            "uniform_nonempty": 1,
            "contiguous_block_indices": 0,
        },
    },
    "attention_sm120_00111": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_2adb38e9fb64cab29bf4",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_2adb38e9fb64cab29bf4",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 0,
            "full_k64_tiles": 1,
            "uniform_nonempty": 1,
            "contiguous_block_indices": 1,
        },
    },
    "attention_sm120_01000": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_70d9ca316721813db07f",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_70d9ca316721813db07f",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 1,
            "full_k64_tiles": 0,
            "uniform_nonempty": 0,
            "contiguous_block_indices": 0,
        },
    },
    "attention_sm120_01010": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_08683144a7b6c64433ad",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_08683144a7b6c64433ad",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 1,
            "full_k64_tiles": 0,
            "uniform_nonempty": 1,
            "contiguous_block_indices": 0,
        },
    },
    "attention_sm120_01011": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_aa9589fa9d54f340e6b1",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_aa9589fa9d54f340e6b1",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 1,
            "full_k64_tiles": 0,
            "uniform_nonempty": 1,
            "contiguous_block_indices": 1,
        },
    },
    "attention_sm120_02000": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_6a1fcc34a68d31545b23",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_6a1fcc34a68d31545b23",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 2,
            "full_k64_tiles": 0,
            "uniform_nonempty": 0,
            "contiguous_block_indices": 0,
        },
    },
    "attention_sm120_02010": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_3e61674cc1a4b6793646",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_3e61674cc1a4b6793646",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 2,
            "full_k64_tiles": 0,
            "uniform_nonempty": 1,
            "contiguous_block_indices": 0,
        },
    },
    "attention_sm120_02011": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_8b24fd2b689f1ead23c9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_8b24fd2b689f1ead23c9",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 2,
            "full_k64_tiles": 0,
            "uniform_nonempty": 1,
            "contiguous_block_indices": 1,
        },
    },
    "attention_sm120_03000": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_82751a008faeeeb37778",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_82751a008faeeeb37778",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 3,
            "full_k64_tiles": 0,
            "uniform_nonempty": 0,
            "contiguous_block_indices": 0,
        },
    },
    "attention_sm120_03010": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_a17d0e6f30423c654a19",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_a17d0e6f30423c654a19",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 3,
            "full_k64_tiles": 0,
            "uniform_nonempty": 1,
            "contiguous_block_indices": 0,
        },
    },
    "attention_sm120_03011": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_813b7e41eaabcd64b34d",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_813b7e41eaabcd64b34d",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 0,
            "block_sizes_mode": 3,
            "full_k64_tiles": 0,
            "uniform_nonempty": 1,
            "contiguous_block_indices": 1,
        },
    },
    "attention_sm120_10000": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_386f636f65fcfae07807",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_386f636f65fcfae07807",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 1,
            "block_sizes_mode": 0,
            "full_k64_tiles": 0,
            "uniform_nonempty": 0,
            "contiguous_block_indices": 0,
        },
    },
    "attention_sm120_10100": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_2069e62b1e24fb1be284",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_2069e62b1e24fb1be284",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 1,
            "block_sizes_mode": 0,
            "full_k64_tiles": 1,
            "uniform_nonempty": 0,
            "contiguous_block_indices": 0,
        },
    },
    "attention_sm120_11000": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_bd48d91a04a229511902",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_bd48d91a04a229511902",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 1,
            "block_sizes_mode": 1,
            "full_k64_tiles": 0,
            "uniform_nonempty": 0,
            "contiguous_block_indices": 0,
        },
    },
    "attention_sm120_12000": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_073e7b711576560b267c",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_073e7b711576560b267c",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 1,
            "block_sizes_mode": 2,
            "full_k64_tiles": 0,
            "uniform_nonempty": 0,
            "contiguous_block_indices": 0,
        },
    },
    "attention_sm120_13000": {
        "arches": ["sm_120a"],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "V_map"],
            ["buffer", "O_map"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q2k_block_index"],
            ["buffer", "q2k_block_nums"],
            ["buffer", "block_sizes"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "num_heads"],
            ["parameter", "q2k_capacity"],
            ["parameter", "num_k_blocks"],
            ["parameter", "block_sparse_num"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "q_blocks_heads_batch",
        },
        "kernel_symbol": "kernel_cake_sage_block_sparse_attention_f9e50348d20ed28e5654",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "use_pdl": False,
        },
        "library": "cake_sage_block_sparse_attention_f9e50348d20ed28e5654",
        "logical": {
            "K_map": "k",
            "O_map": "out",
            "Q_map": "q",
            "V_map": "v",
            "block_sizes": "block_sizes",
            "block_sparse_num": "block_sparse_num",
            "k_scale": "k_scale",
            "num_heads": "num_heads",
            "num_k_blocks": "k_blocks",
            "q2k_block_index": "block_index",
            "q2k_block_nums": "block_nums",
            "q2k_capacity": "capacity",
            "q_scale": "q_scale",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "softmax_scale": "softmax_scale",
            "v_scale": "v_scale",
        },
        "role": "attention_sm120",
        "select_rule": {
            "has_block_nums": 1,
            "block_sizes_mode": 3,
            "full_k64_tiles": 0,
            "uniform_nonempty": 0,
            "contiguous_block_indices": 0,
        },
    },
}


# One verified build per architecture: B200 (``sm_100a``), B300 / GB300
# (``sm_103a``) and RTX PRO 6000 (``sm_120a``).  Every architecture is compiled
# with its exact ``-gencode`` flag set (``build.json`` ``arch_flags``); there is
# no family fallback.
ARCHES = ("sm_100a", "sm_103a", "sm_120a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (12, 0): "sm_120a",
}
SM100_ARCHES = ("sm_100a", "sm_103a")
SM120_ARCH = "sm_120a"
# The roles of the records: the SM100-family attention and quantizers, one
# record each, and the SM120 attention, one record per specialization.
ROLES = ("attention", "quantize_qk_vamax", "quantize_v", "attention_sm120")
# The five metadata flags an SM120 record's ``select_rule`` fixes, in the order
# the dispatcher names them.
SM120_FLAGS = (
    "has_block_nums",
    "block_sizes_mode",
    "full_k64_tiles",
    "uniform_nonempty",
    "contiguous_block_indices",
)
HEAD_DIM = 128
BLOCK_M = 64
KV_BLOCK = 64
K_SCALE_GROUP = 16
TOKENS_PER_CTA = 64
# Logical argument names the public API binds; every module record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).
LOGICAL_TENSORS = (
    "q",
    "k",
    "v",
    "out",
    "lse",
    "q_scale",
    "k_scale",
    "v_scale",
    "block_index",
    "block_nums",
    "block_sizes",
    "q_in",
    "k_in",
    "v_in",
    "q_out",
    "k_out",
    "v_out",
    "v_amax",
)
LOGICAL_SCALARS = (
    "seqlen_q",
    "seqlen_k",
    "num_heads",
    "num_kv_heads",
    "q_blocks",
    "k_blocks",
    "k_scale_groups",
    "capacity",
    "block_sparse_num",
    "has_block_nums",
    "block_sizes_mode",
    "v_scale_batch_stride",
    "total_tiles",
    "softmax_scale_log2",
    "softmax_scale",
    "return_lse",
    "grid_heads",
    "chunks_per_bh",
    "q_chunks",
    "k_chunks",
)


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def records_of_role(role: str) -> tuple[str, ...]:
    """The record names of ``role`` in registry order."""

    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; the roles are {ROLES}")
    return tuple(name for name, record in MODULES.items() if record["role"] == role)


def _sm120_index() -> dict[tuple[int, int, int, int, int], str]:
    index: dict[tuple[int, int, int, int, int], str] = {}
    for name in records_of_role("attention_sm120"):
        rule = MODULES[name]["select_rule"]
        key = tuple(int(rule[flag]) for flag in SM120_FLAGS)
        if key in index:
            raise RuntimeError(
                f"two SM120 records select the same specialization {key}: {index[key]} and {name}"
            )
        index[key] = name  # type: ignore[index]
    return index


# Specialization key -> record name, built once from the records' select rules.
SM120_RECORDS: dict[tuple[int, int, int, int, int], str] = _sm120_index()


def sm120_module_key(
    *,
    has_block_nums: bool,
    block_sizes_mode: int,
    seqlen_k: int,
    uniform_block_count: bool,
    contiguous_block_indices: bool,
    block_sparse_num: int,
) -> tuple[int, int, int, int, int]:
    """The SM120 specialization one call dispatches to, as the five flags of
    ``SM120_FLAGS``: ``(has_block_nums, block_sizes_mode, full_k64_tiles,
    uniform_nonempty, contiguous_block_indices)``.

    Per-row counts are unused under a uniform count, whole 64-token K tiles are
    only assumed without declared block sizes, and a contiguous selection only
    changes the kernel once a row has a block to load: with
    ``block_sparse_num == 0`` every specialization writes zeros, so the
    contiguous flag is cleared together with ``uniform_nonempty`` (no record
    selects ``contiguous_block_indices`` without ``uniform_nonempty``).
    """

    uniform_nonempty = int(uniform_block_count and block_sparse_num > 0)
    return (
        int(has_block_nums and not uniform_block_count),
        int(block_sizes_mode),
        int(block_sizes_mode == 0 and seqlen_k % KV_BLOCK == 0),
        uniform_nonempty,
        int(contiguous_block_indices and uniform_nonempty),
    )


def select_sm120(
    *,
    has_block_nums: int,
    block_sizes_mode: int,
    full_k64_tiles: int,
    uniform_nonempty: int,
    contiguous_block_indices: int,
) -> str:
    """The SM120 record whose ``select_rule`` fixes exactly these five flags."""

    key = (
        int(has_block_nums),
        int(block_sizes_mode),
        int(full_k64_tiles),
        int(uniform_nonempty),
        int(contiguous_block_indices),
    )
    name = SM120_RECORDS.get(key)
    if name is None:
        raise RuntimeError(
            "no exported SM120 Sage block-sparse attention specialization for "
            f"{dict(zip(SM120_FLAGS, key, strict=True))}"
        )
    return name


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def launch_grid(
    record: dict[str, Any], values: dict[str, Any], *, device_index: int
) -> tuple[int, int, int]:
    """Launch grid of one module for the logical ``values`` under its recorded grid rule.

    ``capped_persistent`` (the SM100 attention): one CTA per
    ``(batch, head, query block)`` tile, capped at ``ctas_per_sm`` CTAs per SM
    of the device (the kernel walks the remaining tiles), at least one.
    ``batch_heads_chunks`` (the quantizers): ``batch * heads * chunks`` CTAs,
    where ``heads`` is the kernel's ``grid_heads`` / ``num_heads`` and
    ``chunks`` its ``chunks_per_bh``.  ``q_blocks_heads_batch`` (SM120): the
    3-D grid ``(q_blocks, num_heads, batch)``.  Every input is a host scalar
    the caller already holds; nothing is read from a tensor.
    """

    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind == "capped_persistent":
        tiles = int(values["total_tiles"])
        cap = int(rule["ctas_per_sm"]) * sm_count(device_index)
        return (max(1, min(tiles, cap)), 1, 1)
    if kind == "batch_heads_chunks":
        heads = int(
            values["grid_heads"] if "grid_heads" in values else values["num_heads"]
        )
        return (
            max(1, int(values["batch"]) * heads * int(values["chunks_per_bh"])),
            1,
            1,
        )
    if kind == "q_blocks_heads_batch":
        return (
            max(1, int(values["q_blocks"])),
            max(1, int(values["num_heads"])),
            max(1, int(values["batch"])),
        )
    raise ValueError(f"unknown Cake Sage block-sparse attention grid rule: {kind!r}")


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake Sage block-sparse attention target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake Sage block-sparse attention module {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``.

    The prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's
    JIT build of the same sources and flags (``cake_build.load_library``).
    """

    return cake_build.load_library(library_name(name, arch))


@functools.cache
def entry(name: str, arch: str) -> Any:
    """The bound ``ffi_entry`` callable of record ``name`` on ``arch`` (one attribute lookup per process)."""

    return getattr(load(name, arch), MODULES[name]["ffi_entry"])


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""

    return cake_build.jit_specs(arch)


def run_module(name: str, values: dict[str, Any], *, device_index: int) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` /
    ``LOGICAL_SCALARS`` (plus ``batch`` for the grid) to tensors and Python
    scalars; the module record maps its kernel parameter names onto that
    vocabulary.  The launch grid follows the module's recorded grid rule and
    the kernel runs on the current torch stream.  The public API binds the
    same arguments positionally (``record["arg_plan"]`` order) without this
    per-call interpretation; this entry serves generic callers such as the
    artifact-parity harness.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(record, values, device_index=device_index)
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        logical_name = logical[key]
        if logical_name not in values:
            raise ValueError(
                f"module {name} needs {logical_name!r} (kernel parameter {key!r})"
            )
        args.append(values[logical_name])
    run = entry(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        run(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "BLOCK_M",
    "HEAD_DIM",
    "KV_BLOCK",
    "K_SCALE_GROUP",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "ROLES",
    "SM100_ARCHES",
    "SM120_ARCH",
    "SM120_FLAGS",
    "SM120_RECORDS",
    "TOKENS_PER_CTA",
    "arch_for_capability",
    "device_arch",
    "entry",
    "launch_grid",
    "library_name",
    "load",
    "records_of_role",
    "run_module",
    "select_sm120",
    "sm120_module_key",
    "sm_count",
    "specs_for_arch",
]
