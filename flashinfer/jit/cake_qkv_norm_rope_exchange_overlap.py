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

JIT loader of the Cake fused QKV projection + norm + RoPE + head-exchange
programs (SM90 / SM100 / SM103).

``PROGRAMS`` records every delivered program: its device and binding
translation units, the architectures it compiles for, its compile flags (and,
where a program is built from another program's sources, the compile
definitions naming its kernel symbol and host namespace), the launch block,
the cluster shape and the positional argument names of its ``run``
launcher.  ``ROUTES`` maps, per architecture, the
logical stage keys to programs: ``barrier_p0``, ``barrier_p1`` and ``wait`` on
every architecture, ``producer_bn<bn>[_pair][_<variant>]`` for the producer of
a tile width (``_pair``: the 2-CTA pair tile; ``<variant>``: a producer build
other than the baseline -- ``ow256`` wide operand loads, ``bulk_ow256`` wide
operand loads with bulk row stores, ``bulk`` bulk row stores alone -- the
baseline build has no suffix), and -- where the architecture's default launch
path folds the cross-rank rendezvous into the producer
(``FOLD_BARRIER_DEFAULT_BY_ARCH``) --
``producer_bn<bn>[_pair][_<variant>]_fold<phase>`` for the producer builds that
run the barrier of mailbox phase ``phase`` in a rendezvous warp (one extra warp
per CTA on SM100 / SM103, the otherwise idle TMA warp on SM90).  The default
producer build is selected per architecture and world size
(``PRODUCER_VARIANT_DEFAULT_BY_ARCH_WORLD``); wherever a variant is the
default, the baseline build of the same tile width is delivered beside it (the
``FLASHINFER_QKV_EXCHANGE_PRODUCER_VARIANT=shipped`` opt-out of the Cake
backend).  ``PROGRAMS`` and ``ROUTES`` are filled by the exporter from the
program bundle, ``PRODUCER_VARIANT_DEFAULT_BY_ARCH_WORLD`` from the exporter's
per-(architecture, world size) pins; the loader never hashes sources or
interprets argument plans.  A program is compiled once per architecture with
that architecture's exact flag set, and the architecture is part of the JIT
spec name so two architectures never share one cached library.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, NamedTuple, Optional

import torch

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    sm90a_nvcc_flags,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

# Filled mechanically from the program bundle.
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_qkv_norm_rope_exchange_overlap_02b1f09c700c0c153a86": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_02b1f09c700c0c153a86_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_02b1f09c700c0c153a86_binding.cu",
        ],
        "arches": ["sm_100a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [192, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_092734d18ca9cefe21b8": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b7fcf611d8f7bdc09ca2_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b7fcf611d8f7bdc09ca2_binding.cu",
        ],
        "arches": ["sm_90a"],
        "compile_flags": [],
        "defines": [
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_qkv_norm_rope_exchange_overlap_092734d18ca9cefe21b8",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_a417cf321e06ca27",
        ],
        "args": [
            "device_index",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [32, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_0cc6f56e47ebab3fdb82": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_ee7af16c3fae716e45d2_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_ee7af16c3fae716e45d2_binding.cu",
        ],
        "arches": ["sm_90a"],
        "compile_flags": [],
        "defines": [
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_qkv_norm_rope_exchange_overlap_0cc6f56e47ebab3fdb82",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_66e5f8bc245a7df8",
        ],
        "args": [
            "device_index",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [32, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_0f87d1e5925cd81d327b": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_0f87d1e5925cd81d327b_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_0f87d1e5925cd81d327b_binding.cu",
        ],
        "arches": ["sm_100a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_170e1046f21fa0612b2c": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_170e1046f21fa0612b2c_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_170e1046f21fa0612b2c_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_1f991a087fec97c4af30": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_1f991a087fec97c4af30_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_1f991a087fec97c4af30_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_2005b8b40b1a4ee10ef9": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_2005b8b40b1a4ee10ef9_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_2005b8b40b1a4ee10ef9_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_27a9e2ff563af5644d27": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_27a9e2ff563af5644d27_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_27a9e2ff563af5644d27_binding.cu",
        ],
        "arches": ["sm_107a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [192, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_2a944243ea81979c5d5c": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_2a944243ea81979c5d5c_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_2a944243ea81979c5d5c_binding.cu",
        ],
        "arches": ["sm_90a"],
        "compile_flags": [],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [384, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_2b9c189b86ebb1d20c79": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_2b9c189b86ebb1d20c79_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_2b9c189b86ebb1d20c79_binding.cu",
        ],
        "arches": ["sm_100a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_2ecceb2f544e52504015": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_2ecceb2f544e52504015_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_2ecceb2f544e52504015_binding.cu",
        ],
        "arches": ["sm_100a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [192, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_338093757d2ad7fe7cf4": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_338093757d2ad7fe7cf4_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_338093757d2ad7fe7cf4_binding.cu",
        ],
        "arches": ["sm_90a"],
        "compile_flags": [],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [384, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_4054da4210fed46e7645": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_4054da4210fed46e7645_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_4054da4210fed46e7645_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_45ed8ce8f57f3a656890": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_45ed8ce8f57f3a656890_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_45ed8ce8f57f3a656890_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_502e52054565032af6ad": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b1d866d329f2f38fa312_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b1d866d329f2f38fa312_binding.cu",
        ],
        "arches": ["sm_90a"],
        "compile_flags": [],
        "defines": [
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_qkv_norm_rope_exchange_overlap_502e52054565032af6ad",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_a7ef2527018941cd",
        ],
        "args": [
            "ready",
            "num_words_total",
            "slot_base",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [128, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_5d4bb6b19c2e690f3024": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_5d4bb6b19c2e690f3024_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_5d4bb6b19c2e690f3024_binding.cu",
        ],
        "arches": ["sm_90a"],
        "compile_flags": [],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [384, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_61a999397a049761ea60": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_61a999397a049761ea60_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_61a999397a049761ea60_binding.cu",
        ],
        "arches": ["sm_100a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_68b414261baeee963c2a": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_68b414261baeee963c2a_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_68b414261baeee963c2a_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_69efd34074e608023116": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_69efd34074e608023116_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_69efd34074e608023116_binding.cu",
        ],
        "arches": ["sm_107a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [192, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_6cea4b1cead955382caa": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_6cea4b1cead955382caa_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_6cea4b1cead955382caa_binding.cu",
        ],
        "arches": ["sm_107a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_70830c62a710866bb7c7": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_70830c62a710866bb7c7_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_70830c62a710866bb7c7_binding.cu",
        ],
        "arches": ["sm_100a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_70a5b6a09569b80313de": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_70a5b6a09569b80313de_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_70a5b6a09569b80313de_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [192, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_85994b80021f37783cfb": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_85994b80021f37783cfb_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_85994b80021f37783cfb_binding.cu",
        ],
        "arches": ["sm_107a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_8f59df10e4d9f8889a6a": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_8f59df10e4d9f8889a6a_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_8f59df10e4d9f8889a6a_binding.cu",
        ],
        "arches": ["sm_90a"],
        "compile_flags": [],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [384, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_9342da14c254f5bc59c6": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_9342da14c254f5bc59c6_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_9342da14c254f5bc59c6_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [192, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_9a0d9b341b5180f7e493": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_9a0d9b341b5180f7e493_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_9a0d9b341b5180f7e493_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_9eae0db2863a18c91274": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_9eae0db2863a18c91274_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_9eae0db2863a18c91274_binding.cu",
        ],
        "arches": ["sm_90a"],
        "compile_flags": [],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [384, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_a4c464facca189a5a803": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_a4c464facca189a5a803_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_a4c464facca189a5a803_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [192, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_a80e15c6e7962bb6e449": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_a80e15c6e7962bb6e449_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_a80e15c6e7962bb6e449_binding.cu",
        ],
        "arches": ["sm_100a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_b1937d956988deed0eda": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b1937d956988deed0eda_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b1937d956988deed0eda_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [192, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_b1d866d329f2f38fa312": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b1d866d329f2f38fa312_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b1d866d329f2f38fa312_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "ready",
            "num_words_total",
            "slot_base",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [128, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_b7fcf611d8f7bdc09ca2": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b7fcf611d8f7bdc09ca2_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b7fcf611d8f7bdc09ca2_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "device_index",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [32, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_b8d1d4c4f468ac997963": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b8d1d4c4f468ac997963_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b8d1d4c4f468ac997963_binding.cu",
        ],
        "arches": ["sm_100a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_c1dab51f71ed9c0b7410": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_c1dab51f71ed9c0b7410_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_c1dab51f71ed9c0b7410_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_c320e77573b04a75ecd2": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_c320e77573b04a75ecd2_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_c320e77573b04a75ecd2_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [192, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_c7cccaf9e086af62f948": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_c7cccaf9e086af62f948_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_c7cccaf9e086af62f948_binding.cu",
        ],
        "arches": ["sm_90a"],
        "compile_flags": [],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [384, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_c941e7d765074f1713d9": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_c941e7d765074f1713d9_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_c941e7d765074f1713d9_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_e484812469d655377101": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_e484812469d655377101_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_e484812469d655377101_binding.cu",
        ],
        "arches": ["sm_100a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [192, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_e75374f0daa1274b01f2": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_e75374f0daa1274b01f2_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_e75374f0daa1274b01f2_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [192, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_ee7af16c3fae716e45d2": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_ee7af16c3fae716e45d2_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_ee7af16c3fae716e45d2_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "device_index",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [32, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_ee977d94a3c09de15cbf": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_ee977d94a3c09de15cbf_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_ee977d94a3c09de15cbf_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_f3ef71043fa04f37683f": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_f3ef71043fa04f37683f_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_f3ef71043fa04f37683f_binding.cu",
        ],
        "arches": ["sm_107a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_f680859b58db85fe3dac": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_f680859b58db85fe3dac_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_f680859b58db85fe3dac_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [1, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_f86aebd1e176ce36e6d2": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_f86aebd1e176ce36e6d2_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_f86aebd1e176ce36e6d2_binding.cu",
        ],
        "arches": ["sm_107a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [2, 1, 1],
    },
    "cake_qkv_norm_rope_exchange_overlap_fcd50bf61b8135ffbbbd": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_fcd50bf61b8135ffbbbd_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_fcd50bf61b8135ffbbbd_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [224, 1, 1],
        "cluster": [1, 1, 1],
    },
}
ROUTES: dict[str, dict[str, str]] = {
    "sm_90a": {
        "barrier_p0": "cake_qkv_norm_rope_exchange_overlap_0cc6f56e47ebab3fdb82",
        "barrier_p1": "cake_qkv_norm_rope_exchange_overlap_092734d18ca9cefe21b8",
        "producer_bn128": "cake_qkv_norm_rope_exchange_overlap_2a944243ea81979c5d5c",
        "producer_bn128_fold0": "cake_qkv_norm_rope_exchange_overlap_9eae0db2863a18c91274",
        "producer_bn128_fold1": "cake_qkv_norm_rope_exchange_overlap_338093757d2ad7fe7cf4",
        "producer_bn256": "cake_qkv_norm_rope_exchange_overlap_5d4bb6b19c2e690f3024",
        "producer_bn256_fold0": "cake_qkv_norm_rope_exchange_overlap_8f59df10e4d9f8889a6a",
        "producer_bn256_fold1": "cake_qkv_norm_rope_exchange_overlap_c7cccaf9e086af62f948",
        "wait": "cake_qkv_norm_rope_exchange_overlap_502e52054565032af6ad",
    },
    "sm_100a": {
        "barrier_p0": "cake_qkv_norm_rope_exchange_overlap_ee7af16c3fae716e45d2",
        "barrier_p1": "cake_qkv_norm_rope_exchange_overlap_b7fcf611d8f7bdc09ca2",
        "producer_bn128": "cake_qkv_norm_rope_exchange_overlap_b1937d956988deed0eda",
        "producer_bn128_bulk_ow256": "cake_qkv_norm_rope_exchange_overlap_e484812469d655377101",
        "producer_bn128_bulk_ow256_fold0": "cake_qkv_norm_rope_exchange_overlap_2b9c189b86ebb1d20c79",
        "producer_bn128_bulk_ow256_fold1": "cake_qkv_norm_rope_exchange_overlap_61a999397a049761ea60",
        "producer_bn128_fold0": "cake_qkv_norm_rope_exchange_overlap_c1dab51f71ed9c0b7410",
        "producer_bn128_fold1": "cake_qkv_norm_rope_exchange_overlap_9a0d9b341b5180f7e493",
        "producer_bn128_pair": "cake_qkv_norm_rope_exchange_overlap_a4c464facca189a5a803",
        "producer_bn128_pair_bulk_ow256": "cake_qkv_norm_rope_exchange_overlap_02b1f09c700c0c153a86",
        "producer_bn128_pair_bulk_ow256_fold0": "cake_qkv_norm_rope_exchange_overlap_0f87d1e5925cd81d327b",
        "producer_bn128_pair_bulk_ow256_fold1": "cake_qkv_norm_rope_exchange_overlap_70830c62a710866bb7c7",
        "producer_bn128_pair_fold0": "cake_qkv_norm_rope_exchange_overlap_170e1046f21fa0612b2c",
        "producer_bn128_pair_fold1": "cake_qkv_norm_rope_exchange_overlap_c941e7d765074f1713d9",
        "producer_bn256_pair": "cake_qkv_norm_rope_exchange_overlap_70a5b6a09569b80313de",
        "producer_bn256_pair_bulk_ow256": "cake_qkv_norm_rope_exchange_overlap_2ecceb2f544e52504015",
        "producer_bn256_pair_bulk_ow256_fold0": "cake_qkv_norm_rope_exchange_overlap_a80e15c6e7962bb6e449",
        "producer_bn256_pair_bulk_ow256_fold1": "cake_qkv_norm_rope_exchange_overlap_b8d1d4c4f468ac997963",
        "producer_bn256_pair_fold0": "cake_qkv_norm_rope_exchange_overlap_1f991a087fec97c4af30",
        "producer_bn256_pair_fold1": "cake_qkv_norm_rope_exchange_overlap_68b414261baeee963c2a",
        "producer_bn256_pair_ow256": "cake_qkv_norm_rope_exchange_overlap_c320e77573b04a75ecd2",
        "producer_bn256_pair_ow256_fold0": "cake_qkv_norm_rope_exchange_overlap_4054da4210fed46e7645",
        "producer_bn256_pair_ow256_fold1": "cake_qkv_norm_rope_exchange_overlap_2005b8b40b1a4ee10ef9",
        "wait": "cake_qkv_norm_rope_exchange_overlap_b1d866d329f2f38fa312",
    },
    "sm_103a": {
        "barrier_p0": "cake_qkv_norm_rope_exchange_overlap_ee7af16c3fae716e45d2",
        "barrier_p1": "cake_qkv_norm_rope_exchange_overlap_b7fcf611d8f7bdc09ca2",
        "producer_bn128": "cake_qkv_norm_rope_exchange_overlap_b1937d956988deed0eda",
        "producer_bn128_fold0": "cake_qkv_norm_rope_exchange_overlap_c1dab51f71ed9c0b7410",
        "producer_bn128_fold1": "cake_qkv_norm_rope_exchange_overlap_9a0d9b341b5180f7e493",
        "producer_bn128_ow256": "cake_qkv_norm_rope_exchange_overlap_e75374f0daa1274b01f2",
        "producer_bn128_ow256_fold0": "cake_qkv_norm_rope_exchange_overlap_f680859b58db85fe3dac",
        "producer_bn128_ow256_fold1": "cake_qkv_norm_rope_exchange_overlap_fcd50bf61b8135ffbbbd",
        "producer_bn128_pair": "cake_qkv_norm_rope_exchange_overlap_a4c464facca189a5a803",
        "producer_bn128_pair_fold0": "cake_qkv_norm_rope_exchange_overlap_170e1046f21fa0612b2c",
        "producer_bn128_pair_fold1": "cake_qkv_norm_rope_exchange_overlap_c941e7d765074f1713d9",
        "producer_bn128_pair_ow256": "cake_qkv_norm_rope_exchange_overlap_9342da14c254f5bc59c6",
        "producer_bn128_pair_ow256_fold0": "cake_qkv_norm_rope_exchange_overlap_45ed8ce8f57f3a656890",
        "producer_bn128_pair_ow256_fold1": "cake_qkv_norm_rope_exchange_overlap_ee977d94a3c09de15cbf",
        "producer_bn256_pair": "cake_qkv_norm_rope_exchange_overlap_70a5b6a09569b80313de",
        "producer_bn256_pair_fold0": "cake_qkv_norm_rope_exchange_overlap_1f991a087fec97c4af30",
        "producer_bn256_pair_fold1": "cake_qkv_norm_rope_exchange_overlap_68b414261baeee963c2a",
        "producer_bn256_pair_ow256": "cake_qkv_norm_rope_exchange_overlap_c320e77573b04a75ecd2",
        "producer_bn256_pair_ow256_fold0": "cake_qkv_norm_rope_exchange_overlap_4054da4210fed46e7645",
        "producer_bn256_pair_ow256_fold1": "cake_qkv_norm_rope_exchange_overlap_2005b8b40b1a4ee10ef9",
        "wait": "cake_qkv_norm_rope_exchange_overlap_b1d866d329f2f38fa312",
    },
    "sm_107a": {
        "barrier_p0": "cake_qkv_norm_rope_exchange_overlap_ee7af16c3fae716e45d2",
        "barrier_p1": "cake_qkv_norm_rope_exchange_overlap_b7fcf611d8f7bdc09ca2",
        "producer_bn256_pair": "cake_qkv_norm_rope_exchange_overlap_27a9e2ff563af5644d27",
        "producer_bn256_pair_bulk": "cake_qkv_norm_rope_exchange_overlap_69efd34074e608023116",
        "producer_bn256_pair_bulk_fold0": "cake_qkv_norm_rope_exchange_overlap_85994b80021f37783cfb",
        "producer_bn256_pair_bulk_fold1": "cake_qkv_norm_rope_exchange_overlap_f3ef71043fa04f37683f",
        "producer_bn256_pair_fold0": "cake_qkv_norm_rope_exchange_overlap_f86aebd1e176ce36e6d2",
        "producer_bn256_pair_fold1": "cake_qkv_norm_rope_exchange_overlap_6cea4b1cead955382caa",
        "wait": "cake_qkv_norm_rope_exchange_overlap_b1d866d329f2f38fa312",
    },
}
# Default producer build per architecture and world size (``""`` = the baseline build), filled mechanically from
# the exporter's pins; ``default_producer_variant`` reads it.
PRODUCER_VARIANT_DEFAULT_BY_ARCH_WORLD: dict[str, dict[int, str]] = {
    "sm_90a": {2: "", 4: "", 8: ""},
    "sm_100a": {2: "", 4: "ow256", 8: "bulk_ow256"},
    "sm_103a": {2: "", 4: "ow256", 8: "ow256"},
    "sm_107a": {2: "", 4: "bulk", 8: ""},
}

# Producer builds a route table may carry: the baseline build (``SHIPPED_VARIANT``) first, then wide (256-bit) operand
# loads, wide operand loads with bulk row stores, and bulk row stores alone (the baseline operand loads).  Every
# build stores identical bits; the wide-load builds read their ``cos_sin`` / norm-weight operands through 32-byte
# aligned bases (``PRODUCER_VARIANT_OPERAND_LOAD_WIDTH``; the Cake backend checks the alignment).
PRODUCER_VARIANTS: tuple[str, ...] = ("", "ow256", "bulk_ow256", "bulk")
SHIPPED_VARIANT = ""
PRODUCER_VARIANT_OPERAND_LOAD_WIDTH: dict[str, int] = {
    "": 128,
    "ow256": 256,
    "bulk_ow256": 256,
    "bulk": 128,
}

SOURCE_PACKAGE = "cake_qkv_norm_rope_exchange_overlap"
ARCH_FLAGS: dict[str, list[str]] = {
    "sm_90a": sm90a_nvcc_flags,
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}
CAPABILITY_ARCH: dict[tuple[int, int], str] = {
    (9, 0): "sm_90a",
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
SUPPORTED_WORLD_SIZES: tuple[int, ...] = (2, 4, 8)
# Per-architecture world sizes: SM107 sessions run on 4-GPU nodes, so a
# session there has 2 or 4 ranks.
SUPPORTED_WORLD_SIZES_BY_ARCH: dict[str, tuple[int, ...]] = {
    "sm_90a": (2, 4, 8),
    "sm_100a": (2, 4, 8),
    "sm_103a": (2, 4, 8),
    "sm_107a": (2, 4),
}
# Barrier flag pad: two phase epochs plus two sender-indexed mailbox banks per
# phase and rank (``2 + 2 * 2 * world_size`` uint32 words).
BARRIER_PHASES = 2
BARRIER_BANKS = 2
# Stage keys every architecture's route table carries.
STAGE_KEYS: tuple[str, ...] = ("barrier_p0", "barrier_p1", "wait")
# Default launch path per architecture.  ``True``: a call launches the producer
# build with the folded rendezvous (``producer_bn<bn>[_pair]_fold<phase>``,
# alternating phase) and the wait kernel -- two launches.  ``False``: the
# barrier kernel of the phase, then the plain producer, then the wait kernel --
# three launches.  The folded producers, the barrier programs and the plain
# producers are delivered on every architecture, so the barrier path is always
# available as the explicit ``fold_barrier=False`` choice; the folded path is the
# default on SM100 / SM103 (first), SM90 and SM107 (each since its own measurement;
# SM107 = Rubin: its large W = 2 row measured neutral within noise and
# is disclosed as such, not as a gain).  Both paths store identical bits.
FOLD_BARRIER_DEFAULT_BY_ARCH: dict[str, bool] = {
    "sm_90a": True,
    "sm_100a": True,
    "sm_103a": True,
    "sm_107a": True,
}
FOLD_PHASES: tuple[int, ...] = (0, 1)


class DeviceFacts(NamedTuple):
    capability: tuple[int, int]
    arch: str | None
    sm_count: int


@functools.cache
def device_facts(device_index: int) -> DeviceFacts:
    """Compute capability, exported architecture and SM count, queried once per device."""

    properties = torch.cuda.get_device_properties(device_index)
    capability = (int(properties.major), int(properties.minor))
    return DeviceFacts(
        capability=capability,
        arch=CAPABILITY_ARCH.get(capability),
        sm_count=int(properties.multi_processor_count),
    )


def supported_world_sizes(arch: str) -> tuple[int, ...]:
    """World sizes the delivered programs support on ``arch``."""

    try:
        return SUPPORTED_WORLD_SIZES_BY_ARCH[arch]
    except KeyError:
        raise ValueError(
            f"no fused QKV exchange programs are delivered for {arch!r}"
        ) from None


def barrier_flag_words(world_size: int) -> int:
    """Number of uint32 words of one rank's barrier flag pad."""

    return BARRIER_PHASES + BARRIER_PHASES * BARRIER_BANKS * int(world_size)


def default_fold_barrier(arch: str) -> bool:
    """The default launch path of ``arch`` (:data:`FOLD_BARRIER_DEFAULT_BY_ARCH`); an unknown architecture is an error."""

    try:
        return FOLD_BARRIER_DEFAULT_BY_ARCH[arch]
    except KeyError:
        raise ValueError(
            f"no default launch path for architecture {arch!r}; known: {tuple(FOLD_BARRIER_DEFAULT_BY_ARCH)}"
        ) from None


def default_producer_variant(arch: str, world_size: int) -> str:
    """The default producer build of ``arch`` for a group of ``world_size`` ranks
    (:data:`PRODUCER_VARIANT_DEFAULT_BY_ARCH_WORLD`; ``""`` = the baseline build); an unknown architecture or world
    size is an error."""

    try:
        row = PRODUCER_VARIANT_DEFAULT_BY_ARCH_WORLD[arch]
    except KeyError:
        raise ValueError(
            f"no default producer build for architecture {arch!r}; known: "
            f"{tuple(PRODUCER_VARIANT_DEFAULT_BY_ARCH_WORLD)}"
        ) from None
    try:
        return row[world_size]
    except KeyError:
        raise ValueError(
            f"no default producer build for architecture {arch!r} at world_size={world_size!r}; known: {tuple(row)}"
        ) from None


def _check_variant(variant: str) -> str:
    if variant not in PRODUCER_VARIANTS:
        raise ValueError(
            f"producer build variant must be one of {PRODUCER_VARIANTS}, got {variant!r}"
        )
    return variant


def producer_key(block_n: int, pair: bool, variant: str = SHIPPED_VARIANT) -> str:
    """Route key of a producer build: ``producer_bn<bn>[_pair][_<variant>]``."""

    suffix = f"_{variant}" if _check_variant(variant) else ""
    return f"producer_bn{int(block_n)}{'_pair' if pair else ''}{suffix}"


def fold_producer_key(
    block_n: int, pair: bool, phase: int, variant: str = SHIPPED_VARIANT
) -> str:
    """Route key of the producer build that folds the rendezvous of mailbox ``phase`` into the producer."""

    if phase not in FOLD_PHASES:
        raise ValueError(
            f"rendezvous phase must be one of {FOLD_PHASES}, got {phase!r}"
        )
    return f"{producer_key(block_n, pair, variant)}_fold{int(phase)}"


class ProducerKey(NamedTuple):
    """The parsed form of a producer route key (``fold_phase`` is ``None`` for a plain producer key)."""

    block_n: int
    pair: bool
    variant: str
    fold_phase: Optional[int]


def parse_producer_key(key: str) -> ProducerKey:
    """Parse ``producer_bn<bn>[_pair][_<variant>][_fold<phase>]``; any other spelling is an error."""

    text = str(key)
    body, phase = text, None
    if "_fold" in text:
        body, _, tail = text.rpartition("_fold")
        if not tail.isdigit() or int(tail) not in FOLD_PHASES:
            raise ValueError(
                f"producer route key {key!r} has an unknown rendezvous phase {tail!r}"
            )
        phase = int(tail)
    prefix = "producer_bn"
    if not body.startswith(prefix):
        raise ValueError(
            f"{key!r} is not a producer route key "
            f"({prefix}<bn>[_pair][_<variant>][_fold<phase>])"
        )
    head, _, tail = body[len(prefix) :].partition("_")
    if not head.isdigit():
        raise ValueError(f"producer route key {key!r} has no tile width")
    pair = tail == "pair" or tail.startswith("pair_")
    if tail.startswith("pair_"):
        variant = tail[len("pair_") :]
    else:
        variant = "" if pair else tail
    parsed = ProducerKey(int(head), pair, _check_variant(variant), phase)
    rebuilt = (
        producer_key(parsed.block_n, parsed.pair, parsed.variant)
        if phase is None
        else fold_producer_key(parsed.block_n, parsed.pair, phase, parsed.variant)
    )
    if rebuilt != text:
        raise ValueError(
            f"producer route key {key!r} is not in canonical form ({rebuilt!r})"
        )
    return parsed


def program_for(arch: str, key: str) -> str:
    """The program bound to one stage key on one architecture."""

    try:
        return ROUTES[arch][key]
    except KeyError:
        raise ValueError(
            f"no fused QKV exchange program is delivered for {key!r} on {arch}"
        ) from None


def has_producer(
    arch: str, block_n: int, pair: bool, variant: str = SHIPPED_VARIANT
) -> bool:
    return producer_key(block_n, pair, variant) in ROUTES.get(arch, {})


def has_fold_producer(
    arch: str, block_n: int, pair: bool, variant: str = SHIPPED_VARIANT
) -> bool:
    """Whether both rendezvous phases of the folded producer are delivered for this tile width and build on
    ``arch``."""

    table = ROUTES.get(arch, {})
    return all(
        fold_producer_key(block_n, pair, phase, variant) in table
        for phase in FOLD_PHASES
    )


def fold_producers_delivered(arch: str) -> bool:
    """Whether ``arch`` delivers any folded producer (the ``fold_barrier=True`` launch path is selectable)."""

    return any("_fold" in key for key in ROUTES.get(arch, {}))


def delivered_producer_variants(arch: str, block_n: int, pair: bool) -> tuple[str, ...]:
    """The producer builds delivered for one tile width and pair flag on ``arch``, in :data:`PRODUCER_VARIANTS`
    order (``""`` names the baseline build)."""

    return tuple(
        variant
        for variant in PRODUCER_VARIANTS
        if has_producer(arch, block_n, pair, variant)
    )


def producer_program(
    arch: str, block_n: int, pair: bool, variant: str = SHIPPED_VARIANT
) -> str:
    return program_for(arch, producer_key(block_n, pair, variant))


def fold_producer_program(
    arch: str, block_n: int, pair: bool, phase: int, variant: str = SHIPPED_VARIANT
) -> str:
    return program_for(arch, fold_producer_key(block_n, pair, phase, variant))


def barrier_program(arch: str, phase: int) -> str:
    if phase not in (0, 1):
        raise ValueError(f"barrier phase must be 0 or 1, got {phase!r}")
    return program_for(arch, f"barrier_p{phase}")


def wait_program(arch: str) -> str:
    return program_for(arch, "wait")


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake fused QKV exchange CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def _source_path(relative: str) -> Path:
    parts = Path(relative).parts
    if parts[:2] != ("csrc", SOURCE_PACKAGE) or len(parts) != 3:
        raise ValueError(
            f"exported source path is outside the fused QKV exchange package: {relative!r}"
        )
    return _source_dir() / parts[2]


def _include_paths() -> list[str | Path]:
    checkout_root = Path(__file__).resolve().parents[2]
    include_dir = jit_env.FLASHINFER_INCLUDE_DIR
    if not include_dir.is_dir():
        include_dir = checkout_root / "include"
    # ``gen_jit_spec`` declares ``extra_include_paths: list[str | Path]``; the
    # list is built under that annotation (mypy's ``list`` is invariant).
    paths: list[str | Path] = [
        _source_dir(),
        _source_dir().parent,
        jit_env.FLASHINFER_CSRC_DIR,
        include_dir,
    ]
    return list(dict.fromkeys(paths))


@functools.cache
def spec(program: str, arch: str) -> JitSpec:
    """Build specification of one program for one exact architecture."""

    record = PROGRAMS[program]
    if arch not in record["arches"]:
        raise ValueError(
            f"program {program!r} is delivered for {record['arches']}, not {arch!r}"
        )
    return gen_jit_spec(
        name=f"{SOURCE_PACKAGE}_{arch}_{program}",
        sources=[_source_path(relative) for relative in record["sources"]],
        # ``defines``: the compile definitions of a program built from another
        # program's sources (its kernel symbol and host-shim namespace).
        extra_cuda_cflags=[
            *ARCH_FLAGS[arch],
            *record["compile_flags"],
            *record.get("defines", []),
        ],
        extra_include_paths=_include_paths(),
        # The source build carries its own math flags in ``compile_flags``
        # (no fast math); the default ``-use_fast_math`` would change them.
        use_fast_math=False,
    )


@functools.cache
def load(program: str, arch: str):
    return spec(program, arch).build_and_load()


__all__ = [
    "ARCH_FLAGS",
    "BARRIER_BANKS",
    "BARRIER_PHASES",
    "CAPABILITY_ARCH",
    "FOLD_BARRIER_DEFAULT_BY_ARCH",
    "FOLD_PHASES",
    "PRODUCER_VARIANTS",
    "PRODUCER_VARIANT_DEFAULT_BY_ARCH_WORLD",
    "PRODUCER_VARIANT_OPERAND_LOAD_WIDTH",
    "PROGRAMS",
    "ROUTES",
    "SHIPPED_VARIANT",
    "SOURCE_PACKAGE",
    "STAGE_KEYS",
    "SUPPORTED_WORLD_SIZES",
    "SUPPORTED_WORLD_SIZES_BY_ARCH",
    "DeviceFacts",
    "ProducerKey",
    "barrier_flag_words",
    "barrier_program",
    "default_fold_barrier",
    "default_producer_variant",
    "delivered_producer_variants",
    "device_facts",
    "fold_producer_key",
    "fold_producer_program",
    "fold_producers_delivered",
    "has_fold_producer",
    "has_producer",
    "load",
    "parse_producer_key",
    "producer_key",
    "producer_program",
    "program_for",
    "spec",
    "supported_world_sizes",
    "wait_program",
]
