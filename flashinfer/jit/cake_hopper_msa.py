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

JIT registration of the Cake-generated Hopper (compute capability 9.0) MSA
programs.  One device / binding source pair per physical schedule, compiled
with FlashInfer's exact ``sm_90a`` flag set (``_ARCHES``); ``ROUTES`` maps a
logical route (``<route key>:<stage>``) to the program that serves it and
``MODULES`` carries each program's neutral compile flags, FFI entry and
argument order.  A program's two sources are derived from its name
(``csrc/cake_hopper_msa/<name>_{kernel,binding}.cu``); its JIT key is the
name, the target and the first twenty hex digits of the SHA-256 over the two
sources and its compile flags (``get_hopper_msa_uri``).
"""

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm90a_nvcc_flags

HopperMSAVariant = str
_ARCHES = ("sm_90a",)
_TARGET = "sm90a"

# fmt: off
MODULES: dict[str, dict[str, Any]] = {
    "cake_hopper_msa_025cfebce8c49b67ec28": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_02949879f73b80e4a89d": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_10913214edb8c25d60e6": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_10a5ce76abdf61206e96": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_112f343295150aa53f1b": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_1533648d99b5503e11d3": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "cu_seqlens_q"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "total_q"], ["parameter", "batch"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "use_q_offset"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_17cd0cc3a08e4f2a8aab": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_200ffb0330dbc8893eeb": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_20ef3836378e7e27b4cc": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_2336edef056031b8d505": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_26311d226b72e91edb96": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_276e272a54b4efd20df6": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_2841be2dbf5dd0850755": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_28d169fe4dab7b3b30ce": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_32dffc6bff721ed84d72": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_33db3e98efe0657eaf8c": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_38ae2dcd5875ff4f4492": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "cu_seqlens_q"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "nsplit"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_3ac2f034fe9238ad3b82": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_434683e642fc2353d459": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_43e957550739d30313fb": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_47452510f979fcf77372": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_4c19e05879003d7441c6": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_4c525bd96e3b3a405ca9": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_56dcad5539ad9283e4c4": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_5a3b3a1697633a0e8b04": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_5adf7239b83aa93c4897": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_5b4a0b9e9f07928582f0": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_5d570a1e487133c6dca6": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_5ed0c641bcea3a524077": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_5fdc63d045d78d90a65b": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_6400f9b940af49476321": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_68735943ee82c916aba4": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_68ed6c2ee8c7e4e9b9e7": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_6a18dd4a679b2805be09": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_6bc022e40283794cee7f": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_6c4ab03fa64cb21663ed": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "cu_seqlens_q"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "total_q"], ["parameter", "batch"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "use_q_offset"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_6f654886d6934fcaf89a": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_72b2aa6d03c5f8f9ad54": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_74205ef0b2ce7da8e6d8": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_79943b91410024d2d669": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_7b36cd26ef2fda8ca238": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_7b41c417f2155cba303c": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_7bbffa90057de145b905": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_7bf42068b30a7198903c": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_859fe30a3573f05eae1c": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "cu_seqlens_q"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "total_q"], ["parameter", "batch"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "use_q_offset"], ["parameter", "softmax_scale_log2"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_8b6a95d561090f227dab": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_93a9c57f2a8dddc0caab": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "cu_seqlens_q"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "total_q"], ["parameter", "batch"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "use_q_offset"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_9836e2dce49c4f15dfce": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "cu_seqlens_q"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "total_q"], ["parameter", "batch"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "use_q_offset"], ["parameter", "softmax_scale_log2"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_98d948252b6f068d1c81": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_9a5ac33b7664412dcc01": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "cu_seqlens_q"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "total_q"], ["parameter", "batch"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "use_q_offset"], ["parameter", "softmax_scale_log2"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_9e8f576d0962c01e6d38": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "num_items"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_9f2b923fef8d64829d34": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_a1ae7ef9e264d4f504dd": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_a68e884144b35797092c": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_aa3467cf0eac90f51afb": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "cu_seqlens_q"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "nsplit"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_aaeebb555d02603f6d7d": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_ac2766e9d4ba2584d7ae": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_b2f3a1305ff2fa815c81": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_b6a2a20bc02e391d3b45": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_b8a663af850de1edb0d9": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_ba99bfe0afd9ce76cad5": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_bb22f86b3d99286ad074": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "cu_seqlens_q"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "nsplit"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_bc95c6fe0ba17ca60452": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_bcdd1a5a7ff34d9bd896": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_bee3f87e8352e3c15b6d": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_c068424133e9f294259e": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_c331c4b69479b4b27ddd": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "cu_seqlens_q"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "nsplit"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_c3b2a4f1fd24297a9d2b": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_d0a467f526d3867ac5e1": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_d11a076134fd62b4b353": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_d569914059e33efb7b5d": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_d5f749fb65a2dcacfe45": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_db50816bd273fd52da82": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_db6c9d3115914fecfacf": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_dbd355b7b172f79b91c1": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_de0f3642c1c821652669": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "cu_seqlens_q"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "nsplit"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_e3290a79330977e2e676": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_e3f7ae829be0044f3725": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_e44b187248ccb2036429": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_e7ceaf13bd710abb0e4e": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_e7de79c713bd81585160": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_e81547a295d9d917baea": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_e9b601567cc3a6dd8033": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_e9f9ae0d08ebbf603198": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "Q"], ["buffer", "K"], ["buffer", "out"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "q_offset"], ["parameter", "has_qoff"], ["parameter", "pt_stride"], ["parameter", "max_k_tiles"], ["parameter", "total_q"], ["parameter", "batch_fast"], ["parameter", "evict_first"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_eadeab9fa70f479ad877": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_ec73383b420033711c12": {"compile_flags": ["--use_fast_math"], "ffi_entry": "run", "arg_plan": [["buffer", "Q32"], ["buffer", "K"], ["buffer", "V"], ["buffer", "O"], ["buffer", "q2k_indices"], ["buffer", "page_table"], ["buffer", "seqused_k"], ["buffer", "part_o"], ["buffer", "part_ml"], ["buffer", "counters"], ["parameter", "total_q"], ["parameter", "seqlen_q"], ["parameter", "num_q_heads"], ["parameter", "num_kv_heads"], ["parameter", "max_pages"], ["parameter", "num_chunks"], ["parameter", "softmax_scale_log2"], ["parameter", "zero_u32"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_f6b7c28db5340c5e38bf": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_f7cca058bf8c73b01aee": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
    "cake_hopper_msa_fc14209ebc6cac7e0653": {"compile_flags": [], "ffi_entry": "run", "arg_plan": [["buffer", "S"], ["buffer", "nvp"], ["buffer", "out"], ["parameter", "tiles"], ["parameter", "total_q"], ["parameter", "num_heads"], ["parameter", "num_chunks"], ["parameter", "fb"], ["parameter", "fe"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]},
}
# fmt: on

# fmt: off
ROUTES: dict[str, str] = {
    "decode_v5:nc16:s1:st2:evict_first:half:vseq:lean-qk-v-vlate:main": "cake_hopper_msa_56dcad5539ad9283e4c4",
    "decode_v5:nc16:s1:st2:evict_first:main": "cake_hopper_msa_e44b187248ccb2036429",
    "decode_v5:nc16:s1:st2:none:main": "cake_hopper_msa_aaeebb555d02603f6d7d",
    "decode_v5:nc16:s1:st3:none:persist:main": "cake_hopper_msa_9e8f576d0962c01e6d38",
    "decode_v5:nc16:s2:st2:evict_first:main": "cake_hopper_msa_d5f749fb65a2dcacfe45",
    "decode_v5:nc16:s2:st2:none:main": "cake_hopper_msa_ec73383b420033711c12",
    "decode_v5:nc16:s2:st3:none:main": "cake_hopper_msa_5adf7239b83aa93c4897",
    "decode_v5:nc16:s4:st2:evict_first:main": "cake_hopper_msa_5d570a1e487133c6dca6",
    "decode_v5:nc16:s4:st2:none:main": "cake_hopper_msa_3ac2f034fe9238ad3b82",
    "decode_v5:nc16:s8:st2:evict_first:main": "cake_hopper_msa_db6c9d3115914fecfacf",
    "decode_v5:nc16:s8:st2:none:main": "cake_hopper_msa_7b36cd26ef2fda8ca238",
    "decode_v5:nc8:s16:st2:evict_first:main": "cake_hopper_msa_200ffb0330dbc8893eeb",
    "decode_v5:nc8:s16:st2:none:main": "cake_hopper_msa_9f2b923fef8d64829d34",
    "decode_v5:nc8:s1:st2:evict_first:half:vseq:main": "cake_hopper_msa_7b41c417f2155cba303c",
    "decode_v5:nc8:s1:st2:evict_first:main": "cake_hopper_msa_d569914059e33efb7b5d",
    "decode_v5:nc8:s1:st2:none:main": "cake_hopper_msa_c068424133e9f294259e",
    "decode_v5:nc8:s2:st2:evict_first:main": "cake_hopper_msa_434683e642fc2353d459",
    "decode_v5:nc8:s2:st2:none:main": "cake_hopper_msa_68ed6c2ee8c7e4e9b9e7",
    "decode_v5:nc8:s4:st2:evict_first:main": "cake_hopper_msa_98d948252b6f068d1c81",
    "decode_v5:nc8:s4:st2:none:main": "cake_hopper_msa_5a3b3a1697633a0e8b04",
    "decode_v5:nc8:s8:st4:evict_first:half:main": "cake_hopper_msa_2841be2dbf5dd0850755",
    "decode_v5:nc8:s8:st4:none:half:main": "cake_hopper_msa_17cd0cc3a08e4f2a8aab",
    "prefill_v1:g16:nr32:st2:none:ksplit:main": "cake_hopper_msa_6c4ab03fa64cb21663ed",
    "prefill_v1:g4:nr32:st2:none:ksplit:main": "cake_hopper_msa_1533648d99b5503e11d3",
    "prefill_v1:g8:nr32:st2:none:ksplit:main": "cake_hopper_msa_93a9c57f2a8dddc0caab",
    "prefill_v2:g16:nc2:st4:fb2:none:qrs:pp:pvwait:role:amma:main": "cake_hopper_msa_9a5ac33b7664412dcc01",
    "prefill_v2:g4:nc2:st4:fb2:none:qrs:pp:pvwait:role:amma:main": "cake_hopper_msa_9836e2dce49c4f15dfce",
    "prefill_v2:g8:nc2:st4:fb2:none:qrs:pp:pvwait:role:amma:main": "cake_hopper_msa_859fe30a3573f05eae1c",
    "proxy_decode:bf16:hq1:sq1:main": "cake_hopper_msa_6a18dd4a679b2805be09",
    "proxy_decode:bf16:hq1:sq2:main": "cake_hopper_msa_112f343295150aa53f1b",
    "proxy_decode:bf16:hq1:sq3:main": "cake_hopper_msa_bcdd1a5a7ff34d9bd896",
    "proxy_decode:bf16:hq1:sq4:main": "cake_hopper_msa_79943b91410024d2d669",
    "proxy_decode:bf16:hq2:sq1:main": "cake_hopper_msa_e7ceaf13bd710abb0e4e",
    "proxy_decode:bf16:hq2:sq2:main": "cake_hopper_msa_e7de79c713bd81585160",
    "proxy_decode:bf16:hq2:sq3:main": "cake_hopper_msa_bc95c6fe0ba17ca60452",
    "proxy_decode:bf16:hq2:sq4:main": "cake_hopper_msa_d0a467f526d3867ac5e1",
    "proxy_decode:bf16:hq4:sq1:main": "cake_hopper_msa_72b2aa6d03c5f8f9ad54",
    "proxy_decode:bf16:hq4:sq2:main": "cake_hopper_msa_33db3e98efe0657eaf8c",
    "proxy_decode:bf16:hq4:sq3:main": "cake_hopper_msa_28d169fe4dab7b3b30ce",
    "proxy_decode:bf16:hq4:sq4:main": "cake_hopper_msa_5ed0c641bcea3a524077",
    "proxy_decode:fp8:hq1:sq1:main": "cake_hopper_msa_e9f9ae0d08ebbf603198",
    "proxy_decode:fp8:hq1:sq2:main": "cake_hopper_msa_e81547a295d9d917baea",
    "proxy_decode:fp8:hq1:sq3:main": "cake_hopper_msa_ba99bfe0afd9ce76cad5",
    "proxy_decode:fp8:hq1:sq4:main": "cake_hopper_msa_7bf42068b30a7198903c",
    "proxy_decode:fp8:hq2:sq1:main": "cake_hopper_msa_2336edef056031b8d505",
    "proxy_decode:fp8:hq2:sq2:main": "cake_hopper_msa_47452510f979fcf77372",
    "proxy_decode:fp8:hq2:sq3:main": "cake_hopper_msa_02949879f73b80e4a89d",
    "proxy_decode:fp8:hq2:sq4:main": "cake_hopper_msa_5b4a0b9e9f07928582f0",
    "proxy_decode:fp8:hq4:sq1:main": "cake_hopper_msa_4c19e05879003d7441c6",
    "proxy_decode:fp8:hq4:sq2:main": "cake_hopper_msa_4c525bd96e3b3a405ca9",
    "proxy_decode:fp8:hq4:sq3:main": "cake_hopper_msa_5fdc63d045d78d90a65b",
    "proxy_decode:fp8:hq4:sq4:main": "cake_hopper_msa_10a5ce76abdf61206e96",
    "proxy_prefill:fp8:st4:sb:f16db:main": "cake_hopper_msa_de0f3642c1c821652669",
    "proxy_prefill:fp8:st4:sb:f16t4:main": "cake_hopper_msa_38ae2dcd5875ff4f4492",
    "proxy_prefill:fp8:st4:sb:f16t4F:main": "cake_hopper_msa_aa3467cf0eac90f51afb",
    "proxy_prefill:fp8:st4:sb:f16w3L:main": "cake_hopper_msa_c331c4b69479b4b27ddd",
    "proxy_prefill:fp8:st4:sb:wg4st:main": "cake_hopper_msa_bb22f86b3d99286ad074",
    "topk_select:c128:w4:l6:masked:nvp:lat:main": "cake_hopper_msa_ac2766e9d4ba2584d7ae",
    "topk_select:c128:w4:l7:masked:nvp:lat:main": "cake_hopper_msa_b2f3a1305ff2fa815c81",
    "topk_select:c128:w4:l8:masked:nvp:lat:main": "cake_hopper_msa_f6b7c28db5340c5e38bf",
    "topk_select:c16:w16:l9:masked:nvp:lat:t32:main": "cake_hopper_msa_e3290a79330977e2e676",
    "topk_select:c16:w16:l9:plain:t32:main": "cake_hopper_msa_e3f7ae829be0044f3725",
    "topk_select:c16:w2:l5:masked:nvp:lat:main": "cake_hopper_msa_6400f9b940af49476321",
    "topk_select:c16:w8:l10:masked:nvp:lat:main": "cake_hopper_msa_68735943ee82c916aba4",
    "topk_select:c1:w128:l11:masked:nvp:main": "cake_hopper_msa_7bbffa90057de145b905",
    "topk_select:c1:w128:l11:plain:main": "cake_hopper_msa_276e272a54b4efd20df6",
    "topk_select:c1:w32:l9:masked:nvp:main": "cake_hopper_msa_db50816bd273fd52da82",
    "topk_select:c1:w32:l9:plain:main": "cake_hopper_msa_a1ae7ef9e264d4f504dd",
    "topk_select:c1:w64:l10:masked:nvp:main": "cake_hopper_msa_dbd355b7b172f79b91c1",
    "topk_select:c256:w1:l1:masked:nvp:lat:main": "cake_hopper_msa_b8a663af850de1edb0d9",
    "topk_select:c256:w1:l3:masked:nvp:lat:main": "cake_hopper_msa_a68e884144b35797092c",
    "topk_select:c2:w128:l11:masked:nvp:lat:main": "cake_hopper_msa_f7cca058bf8c73b01aee",
    "topk_select:c2:w128:l11:plain:main": "cake_hopper_msa_025cfebce8c49b67ec28",
    "topk_select:c2:w16:l8:masked:nvp:lat:main": "cake_hopper_msa_fc14209ebc6cac7e0653",
    "topk_select:c2:w32:l9:masked:nvp:lat:main": "cake_hopper_msa_e9b601567cc3a6dd8033",
    "topk_select:c2:w32:l9:plain:main": "cake_hopper_msa_bee3f87e8352e3c15b6d",
    "topk_select:c32:w16:l11:masked:nvp:lat:main": "cake_hopper_msa_eadeab9fa70f479ad877",
    "topk_select:c32:w16:l11:plain:main": "cake_hopper_msa_6bc022e40283794cee7f",
    "topk_select:c32:w1:l1:masked:nvp:lat:main": "cake_hopper_msa_6f654886d6934fcaf89a",
    "topk_select:c32:w1:l3:masked:nvp:lat:main": "cake_hopper_msa_b6a2a20bc02e391d3b45",
    "topk_select:c32:w4:l6:masked:nvp:lat:main": "cake_hopper_msa_74205ef0b2ce7da8e6d8",
    "topk_select:c32:w4:l6:plain:main": "cake_hopper_msa_20ef3836378e7e27b4cc",
    "topk_select:c4:w8:l7:masked:nvp:lat:main": "cake_hopper_msa_c3b2a4f1fd24297a9d2b",
    "topk_select:c64:w8:l10:masked:nvp:lat:main": "cake_hopper_msa_26311d226b72e91edb96",
    "topk_select:c64:w8:l11:masked:nvp:lat:main": "cake_hopper_msa_10913214edb8c25d60e6",
    "topk_select:c8:w32:l9:masked:nvp:lat:main": "cake_hopper_msa_8b6a95d561090f227dab",
    "topk_select:c8:w32:l9:plain:main": "cake_hopper_msa_43e957550739d30313fb",
    "topk_select:c8:w4:l6:masked:nvp:lat:main": "cake_hopper_msa_32dffc6bff721ed84d72",
    "topk_select:c8:w4:l6:plain:main": "cake_hopper_msa_d11a076134fd62b4b353",
}
# fmt: on

HOPPER_MSA_VARIANTS: tuple[HopperMSAVariant, ...] = tuple(MODULES)


def _csrc_dir() -> Path:
    for root in (
        jit_env.FLASHINFER_CSRC_DIR,
        Path(__file__).resolve().parents[2] / "csrc",
    ):
        if (root / "cake_hopper_msa").exists():
            return root
    raise FileNotFoundError(
        "Hopper MSA CUDA sources were not found under csrc/cake_hopper_msa"
    )


def route_program(key: str) -> str:
    """Program name of one logical route (``<route key>:<stage>``)."""
    try:
        return ROUTES[key]
    except KeyError:
        raise KeyError(f"no Hopper MSA program for route {key!r}") from None


@functools.cache
def _content_digest(name: HopperMSAVariant) -> str:
    """First twenty hex digits of the SHA-256 over the program's two sources and
    the compact JSON of its compile flags and target, computed once per process
    at first use."""
    csrc = _csrc_dir() / "cake_hopper_msa"
    digest = hashlib.sha256()
    digest.update((csrc / f"{name}_kernel.cu").read_bytes())
    digest.update((csrc / f"{name}_binding.cu").read_bytes())
    identity = {"options": list(MODULES[name]["compile_flags"]), "target": _ARCHES[0]}
    digest.update(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()[:20]


def get_hopper_msa_uri(name: HopperMSAVariant) -> str:
    """JIT / AOT key of one program: its name, the target and its content digest."""
    if name not in MODULES:
        raise ValueError(f"unsupported Hopper MSA program: {name}")
    return f"{name}_{_TARGET}_{_content_digest(name)}"


@functools.cache
def gen_hopper_msa_module(name: HopperMSAVariant) -> JitSpec:
    uri = get_hopper_msa_uri(name)
    record = MODULES[name]
    csrc = _csrc_dir()
    return gen_jit_spec(
        name=uri,
        sources=[
            csrc / "cake_hopper_msa" / f"{name}_kernel.cu",
            csrc / "cake_hopper_msa" / f"{name}_binding.cu",
        ],
        extra_cuda_cflags=[*sm90a_nvcc_flags, *record["compile_flags"]],
        extra_include_paths=[csrc / "cake_hopper_msa"],
    )


@functools.cache
def load_hopper_msa_module(name: HopperMSAVariant):
    return gen_hopper_msa_module(name).build_and_load()


def get_hopper_msa_module(name: HopperMSAVariant):
    """The loaded module of one program."""
    return load_hopper_msa_module(name)


__all__ = [
    "HOPPER_MSA_VARIANTS",
    "MODULES",
    "ROUTES",
    "HopperMSAVariant",
    "gen_hopper_msa_module",
    "get_hopper_msa_module",
    "get_hopper_msa_uri",
    "load_hopper_msa_module",
    "route_program",
]
