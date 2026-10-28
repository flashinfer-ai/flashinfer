# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cake selective-state-update programs for Blackwell (SM100 / SM103).

The architecture-neutral CUDA programs under ``csrc/cake_selective_state_update``
serve the Cake backend of :func:`flashinfer.mamba.selective_state_update`.
``MODULES`` records each generated program (sources, compile flags, FFI entry,
argument plan, the names of its compile-time defines and every delivered
instantiation keyed by the define-set digest that names its JIT module, and
``arches`` for a program delivered for one architecture only);
``PROGRAMS`` maps the logical program names to those records.  Routes are
chosen from host-known tensor metadata only: no device read-back, no device
allocation and no stream synchronization happen on the call path, so every
route can be captured into a CUDA graph.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

import torch

from .. import env as jit_env
from ..core import (
    JitSpec,
    MissingJITCacheError,
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)
from ...utils import get_compute_capability, get_device_sm_count

# Filled by the generated-program export.
MODULES: dict[str, dict[str, Any]] = {
    "cake_selective_state_update_03e316d2e39f9165ae23": {
        "sources": [
            "cake_selective_state_update_03e316d2e39f9165ae23_kernel.cu",
            "cake_selective_state_update_03e316d2e39f9165ae23_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "output"],
            ["buffer", "intermediate_state"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["parameter", "d_addr"],
            ["parameter", "dt_bias_addr"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "intermediate_state_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "intermediate_stride_slot"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_069ad267429cac3e0710": {
        "sources": [
            "cake_selective_state_update_069ad267429cac3e0710_kernel.cu",
            "cake_selective_state_update_069ad267429cac3e0710_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_092088eb41a05061f1ce": {
        "sources": [
            "cake_selective_state_update_092088eb41a05061f1ce_kernel.cu",
            "cake_selective_state_update_092088eb41a05061f1ce_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_0c4d07dba49698124934": {
        "sources": [
            "cake_selective_state_update_0c4d07dba49698124934_kernel.cu",
            "cake_selective_state_update_0c4d07dba49698124934_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_0d056c8f4ac55933266e": {
        "sources": [
            "cake_selective_state_update_0d056c8f4ac55933266e_kernel.cu",
            "cake_selective_state_update_0d056c8f4ac55933266e_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_100a",
        ],
    },
    "cake_selective_state_update_0e370764097daff22a04": {
        "sources": [
            "cake_selective_state_update_0e370764097daff22a04_kernel.cu",
            "cake_selective_state_update_0e370764097daff22a04_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_1620e61f3308698efb02": {
        "sources": [
            "cake_selective_state_update_1620e61f3308698efb02_kernel.cu",
            "cake_selective_state_update_1620e61f3308698efb02_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_1b1cd93b17f356ca2dfe": {
        "sources": [
            "cake_selective_state_update_1b1cd93b17f356ca2dfe_kernel.cu",
            "cake_selective_state_update_1b1cd93b17f356ca2dfe_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_103a",
        ],
    },
    "cake_selective_state_update_1f10def4297052a15f78": {
        "sources": [
            "cake_selective_state_update_1f10def4297052a15f78_kernel.cu",
            "cake_selective_state_update_1f10def4297052a15f78_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "batch_size"],
            ["parameter", "token_steps"],
            ["parameter", "state_stride_slot"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_1fda6b386a87bbfe2c03": {
        "sources": [
            "cake_selective_state_update_1fda6b386a87bbfe2c03_kernel.cu",
            "cake_selective_state_update_1fda6b386a87bbfe2c03_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_215e5b54ac3db1884c38": {
        "sources": [
            "cake_selective_state_update_215e5b54ac3db1884c38_kernel.cu",
            "cake_selective_state_update_215e5b54ac3db1884c38_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_2b07ca215b31742b79ab": {
        "sources": [
            "cake_selective_state_update_2b07ca215b31742b79ab_kernel.cu",
            "cake_selective_state_update_2b07ca215b31742b79ab_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_2daf8a1c27be3854989c": {
        "sources": [
            "cake_selective_state_update_2daf8a1c27be3854989c_kernel.cu",
            "cake_selective_state_update_2daf8a1c27be3854989c_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["tma_buffer", "x_tma"],
            ["tma_buffer", "b_tma"],
            ["tma_buffer", "c_tma"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "intermediate_state"],
            ["buffer", "intermediate_state_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "total_tiles"],
            ["parameter", "intermediate_stride_slot"],
            ["parameter", "dt_softplus"],
            ["parameter", "cache_intermediate"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_2ea76bcfb658269baad5": {
        "sources": [
            "cake_selective_state_update_2ea76bcfb658269baad5_kernel.cu",
            "cake_selective_state_update_2ea76bcfb658269baad5_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "batch_size"],
            ["parameter", "token_steps"],
            ["parameter", "state_stride_slot"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_37b861f7812f3eeb556a": {
        "sources": [
            "cake_selective_state_update_37b861f7812f3eeb556a_kernel.cu",
            "cake_selective_state_update_37b861f7812f3eeb556a_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_103a",
        ],
    },
    "cake_selective_state_update_3f3150aae42f892f1c84": {
        "sources": [
            "cake_selective_state_update_3f3150aae42f892f1c84_kernel.cu",
            "cake_selective_state_update_3f3150aae42f892f1c84_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["parameter", "batch_size"],
            ["parameter", "nheads"],
            ["parameter", "dim"],
            ["parameter", "dstate"],
            ["parameter", "ngroups"],
            ["parameter", "token_steps"],
            ["parameter", "head_tiles"],
            ["parameter", "dt_softplus"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_425d23b976b2c934449b": {
        "sources": [
            "cake_selective_state_update_425d23b976b2c934449b_kernel.cu",
            "cake_selective_state_update_425d23b976b2c934449b_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_51e0af2f05c4f6e5880f": {
        "sources": [
            "cake_selective_state_update_51e0af2f05c4f6e5880f_kernel.cu",
            "cake_selective_state_update_51e0af2f05c4f6e5880f_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_53bfeb43fd704ac3d6c5": {
        "sources": [
            "cake_selective_state_update_53bfeb43fd704ac3d6c5_kernel.cu",
            "cake_selective_state_update_53bfeb43fd704ac3d6c5_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_100a",
        ],
    },
    "cake_selective_state_update_5b534b81cdee477c1feb": {
        "sources": [
            "cake_selective_state_update_5b534b81cdee477c1feb_kernel.cu",
            "cake_selective_state_update_5b534b81cdee477c1feb_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_103a",
        ],
    },
    "cake_selective_state_update_5efde2a65925e2c3c9d3": {
        "sources": [
            "cake_selective_state_update_5efde2a65925e2c3c9d3_kernel.cu",
            "cake_selective_state_update_5efde2a65925e2c3c9d3_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_650060655f4353a908df": {
        "sources": [
            "cake_selective_state_update_650060655f4353a908df_kernel.cu",
            "cake_selective_state_update_650060655f4353a908df_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_66f42b672ba733b5afd1": {
        "sources": [
            "cake_selective_state_update_66f42b672ba733b5afd1_kernel.cu",
            "cake_selective_state_update_66f42b672ba733b5afd1_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_690e2f567f0c6c4be78d": {
        "sources": [
            "cake_selective_state_update_690e2f567f0c6c4be78d_kernel.cu",
            "cake_selective_state_update_690e2f567f0c6c4be78d_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "D"],
            ["buffer", "z"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["parameter", "batch_size"],
            ["parameter", "nheads"],
            ["parameter", "dim"],
            ["parameter", "dstate"],
            ["parameter", "ngroups"],
            ["parameter", "token_steps"],
            ["parameter", "head_tiles"],
            ["parameter", "dt_softplus"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_100a",
        ],
    },
    "cake_selective_state_update_6947b451e086ed2a3b90": {
        "sources": [
            "cake_selective_state_update_6947b451e086ed2a3b90_kernel.cu",
            "cake_selective_state_update_6947b451e086ed2a3b90_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_6b625e5aa92cae6c618d": {
        "sources": [
            "cake_selective_state_update_6b625e5aa92cae6c618d_kernel.cu",
            "cake_selective_state_update_6b625e5aa92cae6c618d_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_100a",
        ],
    },
    "cake_selective_state_update_75d6c61d14a99c37b5a3": {
        "sources": [
            "cake_selective_state_update_75d6c61d14a99c37b5a3_kernel.cu",
            "cake_selective_state_update_75d6c61d14a99c37b5a3_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_103a",
        ],
    },
    "cake_selective_state_update_7880aa81b86665feb0ec": {
        "sources": [
            "cake_selective_state_update_7880aa81b86665feb0ec_kernel.cu",
            "cake_selective_state_update_7880aa81b86665feb0ec_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "output"],
            ["buffer", "intermediate_state"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["parameter", "d_addr"],
            ["parameter", "dt_bias_addr"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "intermediate_state_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "intermediate_stride_slot"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_79b7ad05a6250777090c": {
        "sources": [
            "cake_selective_state_update_79b7ad05a6250777090c_kernel.cu",
            "cake_selective_state_update_79b7ad05a6250777090c_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_7a283e0596e5009db58a": {
        "sources": [
            "cake_selective_state_update_7a283e0596e5009db58a_kernel.cu",
            "cake_selective_state_update_7a283e0596e5009db58a_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_7aa77bb2bcd225bc7c4c": {
        "sources": [
            "cake_selective_state_update_7aa77bb2bcd225bc7c4c_kernel.cu",
            "cake_selective_state_update_7aa77bb2bcd225bc7c4c_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_7bf2ab4137e58a5005ce": {
        "sources": [
            "cake_selective_state_update_7bf2ab4137e58a5005ce_kernel.cu",
            "cake_selective_state_update_7bf2ab4137e58a5005ce_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_7df813cef15099ef6b6c": {
        "sources": [
            "cake_selective_state_update_7df813cef15099ef6b6c_kernel.cu",
            "cake_selective_state_update_7df813cef15099ef6b6c_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_7e148cb3204c6ae7b537": {
        "sources": [
            "cake_selective_state_update_7e148cb3204c6ae7b537_kernel.cu",
            "cake_selective_state_update_7e148cb3204c6ae7b537_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["parameter", "batch_size"],
            ["parameter", "nheads"],
            ["parameter", "dim"],
            ["parameter", "dstate"],
            ["parameter", "ngroups"],
            ["parameter", "token_steps"],
            ["parameter", "head_tiles"],
            ["parameter", "dt_softplus"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_100a",
        ],
    },
    "cake_selective_state_update_8135210c44963374bccd": {
        "sources": [
            "cake_selective_state_update_8135210c44963374bccd_kernel.cu",
            "cake_selective_state_update_8135210c44963374bccd_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["tma_buffer", "x_tma"],
            ["tma_buffer", "b_tma"],
            ["tma_buffer", "c_tma"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "intermediate_state"],
            ["buffer", "intermediate_state_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "total_tiles"],
            ["parameter", "intermediate_stride_slot"],
            ["parameter", "dt_softplus"],
            ["parameter", "cache_intermediate"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_85985e28d97736821e4c": {
        "sources": [
            "cake_selective_state_update_85985e28d97736821e4c_kernel.cu",
            "cake_selective_state_update_85985e28d97736821e4c_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_8e707f2ce4abdf70d66f": {
        "sources": [
            "cake_selective_state_update_8e707f2ce4abdf70d66f_kernel.cu",
            "cake_selective_state_update_8e707f2ce4abdf70d66f_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "output"],
            ["buffer", "intermediate_state"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["parameter", "d_addr"],
            ["parameter", "dt_bias_addr"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "intermediate_state_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "intermediate_stride_slot"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_8ede1128580d84e2f75c": {
        "sources": [
            "cake_selective_state_update_8ede1128580d84e2f75c_kernel.cu",
            "cake_selective_state_update_8ede1128580d84e2f75c_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_9264ca07d890e146617b": {
        "sources": [
            "cake_selective_state_update_9264ca07d890e146617b_kernel.cu",
            "cake_selective_state_update_9264ca07d890e146617b_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["parameter", "batch_size"],
            ["parameter", "nheads"],
            ["parameter", "dim"],
            ["parameter", "dstate"],
            ["parameter", "ngroups"],
            ["parameter", "token_steps"],
            ["parameter", "head_tiles"],
            ["parameter", "dt_softplus"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_100a",
        ],
    },
    "cake_selective_state_update_93a0e0315a4697642884": {
        "sources": [
            "cake_selective_state_update_93a0e0315a4697642884_kernel.cu",
            "cake_selective_state_update_93a0e0315a4697642884_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_103a",
        ],
    },
    "cake_selective_state_update_a1b343381deb84763333": {
        "sources": [
            "cake_selective_state_update_a1b343381deb84763333_kernel.cu",
            "cake_selective_state_update_a1b343381deb84763333_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_a38772ef81c782c2ddcf": {
        "sources": [
            "cake_selective_state_update_a38772ef81c782c2ddcf_kernel.cu",
            "cake_selective_state_update_a38772ef81c782c2ddcf_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_103a",
        ],
    },
    "cake_selective_state_update_a746af67567a508fa3c2": {
        "sources": [
            "cake_selective_state_update_a746af67567a508fa3c2_kernel.cu",
            "cake_selective_state_update_a746af67567a508fa3c2_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "units"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_b07c2a52beccd1a47a41": {
        "sources": [
            "cake_selective_state_update_b07c2a52beccd1a47a41_kernel.cu",
            "cake_selective_state_update_b07c2a52beccd1a47a41_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_103a",
        ],
    },
    "cake_selective_state_update_b1306a044543e22b04f3": {
        "sources": [
            "cake_selective_state_update_b1306a044543e22b04f3_kernel.cu",
            "cake_selective_state_update_b1306a044543e22b04f3_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_b4f79470cd2af6081722": {
        "sources": [
            "cake_selective_state_update_b4f79470cd2af6081722_kernel.cu",
            "cake_selective_state_update_b4f79470cd2af6081722_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_bb0dce3cad66d31fbd71": {
        "sources": [
            "cake_selective_state_update_bb0dce3cad66d31fbd71_kernel.cu",
            "cake_selective_state_update_bb0dce3cad66d31fbd71_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_bf3b407276910455ace8": {
        "sources": [
            "cake_selective_state_update_bf3b407276910455ace8_kernel.cu",
            "cake_selective_state_update_bf3b407276910455ace8_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_103a",
        ],
    },
    "cake_selective_state_update_bf9d8f51eaa98a9b55e0": {
        "sources": [
            "cake_selective_state_update_bf9d8f51eaa98a9b55e0_kernel.cu",
            "cake_selective_state_update_bf9d8f51eaa98a9b55e0_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_c250f06894e17ae7105f": {
        "sources": [
            "cake_selective_state_update_c250f06894e17ae7105f_kernel.cu",
            "cake_selective_state_update_c250f06894e17ae7105f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["parameter", "batch_size"],
            ["parameter", "nheads"],
            ["parameter", "dim"],
            ["parameter", "dstate"],
            ["parameter", "ngroups"],
            ["parameter", "token_steps"],
            ["parameter", "state_stride_slot"],
            ["parameter", "dt_softplus"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_c417c072b913bfb0b080": {
        "sources": [
            "cake_selective_state_update_c417c072b913bfb0b080_kernel.cu",
            "cake_selective_state_update_c417c072b913bfb0b080_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_c50c0d087342726cb2c9": {
        "sources": [
            "cake_selective_state_update_c50c0d087342726cb2c9_kernel.cu",
            "cake_selective_state_update_c50c0d087342726cb2c9_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_c6359778c662f82edaea": {
        "sources": [
            "cake_selective_state_update_c6359778c662f82edaea_kernel.cu",
            "cake_selective_state_update_c6359778c662f82edaea_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_c70704cfda5f53df5935": {
        "sources": [
            "cake_selective_state_update_c70704cfda5f53df5935_kernel.cu",
            "cake_selective_state_update_c70704cfda5f53df5935_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_cf1848ae62ab6ca3b7f0": {
        "sources": [
            "cake_selective_state_update_cf1848ae62ab6ca3b7f0_kernel.cu",
            "cake_selective_state_update_cf1848ae62ab6ca3b7f0_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_cfaa7ed6d28c5e210f1f": {
        "sources": [
            "cake_selective_state_update_cfaa7ed6d28c5e210f1f_kernel.cu",
            "cake_selective_state_update_cfaa7ed6d28c5e210f1f_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_d0caa4515bbe78db52f9": {
        "sources": [
            "cake_selective_state_update_d0caa4515bbe78db52f9_kernel.cu",
            "cake_selective_state_update_d0caa4515bbe78db52f9_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_100a",
        ],
    },
    "cake_selective_state_update_d3495f7b5362267e04c4": {
        "sources": [
            "cake_selective_state_update_d3495f7b5362267e04c4_kernel.cu",
            "cake_selective_state_update_d3495f7b5362267e04c4_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_d3991bda62384e60d45d": {
        "sources": [
            "cake_selective_state_update_d3991bda62384e60d45d_kernel.cu",
            "cake_selective_state_update_d3991bda62384e60d45d_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_103a",
        ],
    },
    "cake_selective_state_update_d5bed033482836800b3c": {
        "sources": [
            "cake_selective_state_update_d5bed033482836800b3c_kernel.cu",
            "cake_selective_state_update_d5bed033482836800b3c_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "batch_size"],
            ["parameter", "token_steps"],
            ["parameter", "state_stride_slot"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_d640bc795863da1c5fb4": {
        "sources": [
            "cake_selective_state_update_d640bc795863da1c5fb4_kernel.cu",
            "cake_selective_state_update_d640bc795863da1c5fb4_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["parameter", "batch_size"],
            ["parameter", "nheads"],
            ["parameter", "dim"],
            ["parameter", "dstate"],
            ["parameter", "ngroups"],
            ["parameter", "token_steps"],
            ["parameter", "state_stride_slot"],
            ["parameter", "dt_softplus"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_d83be0f311b5f9806ebe": {
        "sources": [
            "cake_selective_state_update_d83be0f311b5f9806ebe_kernel.cu",
            "cake_selective_state_update_d83be0f311b5f9806ebe_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["parameter", "batch_size"],
            ["parameter", "nheads"],
            ["parameter", "dim"],
            ["parameter", "dstate"],
            ["parameter", "ngroups"],
            ["parameter", "token_steps"],
            ["parameter", "head_tiles"],
            ["parameter", "dt_softplus"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_da654c98ebe8c52ad280": {
        "sources": [
            "cake_selective_state_update_da654c98ebe8c52ad280_kernel.cu",
            "cake_selective_state_update_da654c98ebe8c52ad280_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "output"],
            ["buffer", "intermediate_state"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["parameter", "d_addr"],
            ["parameter", "dt_bias_addr"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "intermediate_state_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "intermediate_stride_slot"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [
            "COEFFICIENT_BF16",
            "INDEX_I32",
            "NHEADS_STATIC",
            "NGROUPS_STATIC",
            "X_BATCH_STRIDE",
            "X_STEP_STRIDE",
            "DT_BATCH_STRIDE",
            "DT_STEP_STRIDE",
            "B_BATCH_STRIDE",
            "B_STEP_STRIDE",
            "C_BATCH_STRIDE",
            "C_STEP_STRIDE",
        ],
        "instantiations": {
            "8cb7048fb7793f88": {
                "COEFFICIENT_BF16": 0,
                "INDEX_I32": 0,
                "NHEADS_STATIC": 0,
                "NGROUPS_STATIC": 0,
                "X_BATCH_STRIDE": 24576,
                "X_STEP_STRIDE": 4096,
                "DT_BATCH_STRIDE": 384,
                "DT_STEP_STRIDE": 64,
                "B_BATCH_STRIDE": 6144,
                "B_STEP_STRIDE": 1024,
                "C_BATCH_STRIDE": 6144,
                "C_STEP_STRIDE": 1024,
            },
            "2682678a821d9bed": {
                "COEFFICIENT_BF16": 0,
                "INDEX_I32": 0,
                "NHEADS_STATIC": 0,
                "NGROUPS_STATIC": 0,
                "X_BATCH_STRIDE": 24576,
                "X_STEP_STRIDE": 4096,
                "DT_BATCH_STRIDE": 384,
                "DT_STEP_STRIDE": 64,
                "B_BATCH_STRIDE": 768,
                "B_STEP_STRIDE": 128,
                "C_BATCH_STRIDE": 768,
                "C_STEP_STRIDE": 128,
            },
            "e5f56c56ab9d4db4": {
                "COEFFICIENT_BF16": 1,
                "INDEX_I32": 1,
                "NHEADS_STATIC": 64,
                "NGROUPS_STATIC": 1,
                "X_BATCH_STRIDE": 26112,
                "X_STEP_STRIDE": 4352,
                "DT_BATCH_STRIDE": 51072,
                "DT_STEP_STRIDE": 8512,
                "B_BATCH_STRIDE": 26112,
                "B_STEP_STRIDE": 4352,
                "C_BATCH_STRIDE": 26112,
                "C_STEP_STRIDE": 4352,
            },
        },
    },
    "cake_selective_state_update_df2ae3087dbfec1d66dd": {
        "sources": [
            "cake_selective_state_update_df2ae3087dbfec1d66dd_kernel.cu",
            "cake_selective_state_update_df2ae3087dbfec1d66dd_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["parameter", "batch_size"],
            ["parameter", "nheads"],
            ["parameter", "dim"],
            ["parameter", "dstate"],
            ["parameter", "ngroups"],
            ["parameter", "token_steps"],
            ["parameter", "head_tiles"],
            ["parameter", "dt_softplus"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_103a",
        ],
    },
    "cake_selective_state_update_e0443592bce0b8b1b911": {
        "sources": [
            "cake_selective_state_update_e0443592bce0b8b1b911_kernel.cu",
            "cake_selective_state_update_e0443592bce0b8b1b911_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "output"],
            ["buffer", "intermediate_state"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["parameter", "d_addr"],
            ["parameter", "dt_bias_addr"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "intermediate_state_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "intermediate_stride_slot"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_e0a6f0a910168d91472d": {
        "sources": [
            "cake_selective_state_update_e0a6f0a910168d91472d_kernel.cu",
            "cake_selective_state_update_e0a6f0a910168d91472d_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_e6b877dd79b4a59b8825": {
        "sources": [
            "cake_selective_state_update_e6b877dd79b4a59b8825_kernel.cu",
            "cake_selective_state_update_e6b877dd79b4a59b8825_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_e7d7d58d9c8c459c5f85": {
        "sources": [
            "cake_selective_state_update_e7d7d58d9c8c459c5f85_kernel.cu",
            "cake_selective_state_update_e7d7d58d9c8c459c5f85_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_e9c1044201dcdb724f52": {
        "sources": [
            "cake_selective_state_update_e9c1044201dcdb724f52_kernel.cu",
            "cake_selective_state_update_e9c1044201dcdb724f52_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_e9c17ab5afa18774829d": {
        "sources": [
            "cake_selective_state_update_e9c17ab5afa18774829d_kernel.cu",
            "cake_selective_state_update_e9c17ab5afa18774829d_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "output"],
            ["buffer", "intermediate_state"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["parameter", "d_addr"],
            ["parameter", "dt_bias_addr"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "intermediate_state_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "intermediate_stride_slot"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_eb7ec508e23e9f9cc884": {
        "sources": [
            "cake_selective_state_update_eb7ec508e23e9f9cc884_kernel.cu",
            "cake_selective_state_update_eb7ec508e23e9f9cc884_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_ebfde0cc1cfa338d7aaf": {
        "sources": [
            "cake_selective_state_update_ebfde0cc1cfa338d7aaf_kernel.cu",
            "cake_selective_state_update_ebfde0cc1cfa338d7aaf_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_100a",
        ],
    },
    "cake_selective_state_update_f75fba6195c641c8ea09": {
        "sources": [
            "cake_selective_state_update_f75fba6195c641c8ea09_kernel.cu",
            "cake_selective_state_update_f75fba6195c641c8ea09_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": ["COEFFICIENT_BF16", "INDEX_I32", "PREFETCH_ROWS", "DIM_TILES"],
        "instantiations": {
            "3b6308a422404887": {
                "COEFFICIENT_BF16": 1,
                "INDEX_I32": 1,
                "PREFETCH_ROWS": 1,
                "DIM_TILES": 8,
            },
            "648cce0221e406c5": {
                "COEFFICIENT_BF16": 1,
                "INDEX_I32": 1,
                "PREFETCH_ROWS": 4,
                "DIM_TILES": 1,
            },
            "4107f9a7589d0c16": {
                "COEFFICIENT_BF16": 0,
                "INDEX_I32": 0,
                "PREFETCH_ROWS": 4,
                "DIM_TILES": 1,
            },
        },
    },
    "cake_selective_state_update_f8ebb15623f0fd11be1a": {
        "sources": [
            "cake_selective_state_update_f8ebb15623f0fd11be1a_kernel.cu",
            "cake_selective_state_update_f8ebb15623f0fd11be1a_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_fa4d8236383f6cfa5858": {
        "sources": [
            "cake_selective_state_update_fa4d8236383f6cfa5858_kernel.cu",
            "cake_selective_state_update_fa4d8236383f6cfa5858_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
        "arches": [
            "sm_100a",
        ],
    },
    "cake_selective_state_update_ffc75b2432b6aa1e1b7d": {
        "sources": [
            "cake_selective_state_update_ffc75b2432b6aa1e1b7d_kernel.cu",
            "cake_selective_state_update_ffc75b2432b6aa1e1b7d_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "dst_state_batch_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "head_tiles"],
            ["parameter", "x_batch_stride"],
            ["parameter", "b_batch_stride"],
            ["parameter", "c_batch_stride"],
            ["parameter", "out_batch_stride"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "defines": [],
        "instantiations": {},
    },
}
PROGRAMS: dict[str, str] = {
    "dynamic_lanes16": "cake_selective_state_update_2ea76bcfb658269baad5",
    "dynamic_lanes16_pr": "cake_selective_state_update_1f10def4297052a15f78",
    "dynamic_lanes8": "cake_selective_state_update_d5bed033482836800b3c",
    "mtp_cache_bf16_i32": "cake_selective_state_update_e0443592bce0b8b1b911",
    "mtp_cache_c4_t6": "cake_selective_state_update_da654c98ebe8c52ad280",
    "mtp_cache_groups1": "cake_selective_state_update_7880aa81b86665feb0ec",
    "mtp_cache_groups1_heads64": "cake_selective_state_update_8e707f2ce4abdf70d66f",
    "mtp_cache_groups8": "cake_selective_state_update_e9c17ab5afa18774829d",
    "mtp_cache_groups8_heads64": "cake_selective_state_update_03e316d2e39f9165ae23",
    "mtp_horizontal_direct": "cake_selective_state_update_8135210c44963374bccd",
    "mtp_horizontal_early": "cake_selective_state_update_2daf8a1c27be3854989c",
    "mtp_short_prefetch": "cake_selective_state_update_d640bc795863da1c5fb4",
    "mtp_short_prefetch_heads64_groups8": "cake_selective_state_update_c250f06894e17ae7105f",
    "mtp_short_ring_heads64_groups8_t1": "cake_selective_state_update_d83be0f311b5f9806ebe",
    "mtp_short_ring_heads64_groups8_t1_stp": "cake_selective_state_update_9264ca07d890e146617b",
    "mtp_short_ring_heads64_groups8_t1_stp_z": "cake_selective_state_update_690e2f567f0c6c4be78d",
    "mtp_short_ring_heads64_groups8_t2": "cake_selective_state_update_3f3150aae42f892f1c84",
    "mtp_short_ring_heads64_groups8_t2_jr": "cake_selective_state_update_7e148cb3204c6ae7b537",
    "mtp_short_ring_heads64_groups8_t2_jr_hb": "cake_selective_state_update_df2ae3087dbfec1d66dd",
    "stp_dim128_fp32": "cake_selective_state_update_0c4d07dba49698124934",
    "stp_dim128_fp32_l2_prefetch": "cake_selective_state_update_eb7ec508e23e9f9cc884",
    "stp_dim128_hpc1_mb4": "cake_selective_state_update_1620e61f3308698efb02",
    "stp_dim128_hpc1_s2w4": "cake_selective_state_update_53bfeb43fd704ac3d6c5",
    "stp_dim128_hpc1_s2w4_lean": "cake_selective_state_update_1b1cd93b17f356ca2dfe",
    "stp_dim128_hpc2_lean": "cake_selective_state_update_cfaa7ed6d28c5e210f1f",
    "stp_dim128_hpc2_s2w4_mb5_lean": "cake_selective_state_update_0d056c8f4ac55933266e",
    "stp_dim128_z_balanced_lean": "cake_selective_state_update_a746af67567a508fa3c2",
    "stp_dim128_z_gated": "cake_selective_state_update_c50c0d087342726cb2c9",
    "stp_hd64_paired_hpc1": "cake_selective_state_update_85985e28d97736821e4c",
    "stp_hd64_paired_hpc1_bf16_i32": "cake_selective_state_update_8ede1128580d84e2f75c",
    "stp_hd64_paired_hpc1_ratio8": "cake_selective_state_update_1fda6b386a87bbfe2c03",
    "stp_hd64_paired_hpc1_ratio8_bf16_i32": "cake_selective_state_update_66f42b672ba733b5afd1",
    "stp_hd64_paired_hpc1_ratio8_s4w2": "cake_selective_state_update_fa4d8236383f6cfa5858",
    "stp_hd64_paired_hpc1_ratio8_s4w2_bf16_i32": "cake_selective_state_update_d0caa4515bbe78db52f9",
    "stp_hd64_paired_hpc1_ratio8_s4w2_lean": "cake_selective_state_update_5b534b81cdee477c1feb",
    "stp_hd64_paired_hpc1_ratio8_s4w2_lean_bf16_i32": "cake_selective_state_update_b07c2a52beccd1a47a41",
    "stp_hd64_paired_hpc1_s4w2": "cake_selective_state_update_6b625e5aa92cae6c618d",
    "stp_hd64_paired_hpc1_s4w2_bf16_i32": "cake_selective_state_update_ebfde0cc1cfa338d7aaf",
    "stp_hd64_paired_hpc1_s4w2_lean": "cake_selective_state_update_75d6c61d14a99c37b5a3",
    "stp_hd64_paired_hpc1_s4w2_lean_bf16_i32": "cake_selective_state_update_a38772ef81c782c2ddcf",
    "stp_hd64_paired_hpc2": "cake_selective_state_update_d3991bda62384e60d45d",
    "stp_hd64_paired_hpc2_bf16_i32": "cake_selective_state_update_bf3b407276910455ace8",
    "stp_hd64_paired_hpc2_lean": "cake_selective_state_update_e7d7d58d9c8c459c5f85",
    "stp_hd64_paired_hpc2_lean_bf16_i32": "cake_selective_state_update_51e0af2f05c4f6e5880f",
    "stp_hd64_paired_hpc2_mb2_lean": "cake_selective_state_update_6947b451e086ed2a3b90",
    "stp_hd64_paired_hpc2_mb2_lean_bf16_i32": "cake_selective_state_update_069ad267429cac3e0710",
    "stp_hd64_paired_hpc2_ratio8_s2w4_lean": "cake_selective_state_update_425d23b976b2c934449b",
    "stp_hd64_paired_hpc2_ratio8_s2w4_lean_bf16_i32": "cake_selective_state_update_e6b877dd79b4a59b8825",
    "stp_hd64_paired_hpc2_s2w4_lean": "cake_selective_state_update_e9c1044201dcdb724f52",
    "stp_hd64_paired_hpc2_s2w4_lean_bf16_i32": "cake_selective_state_update_b4f79470cd2af6081722",
    "stp_hd64_paired_hpc2_s2w4_mb5_lean": "cake_selective_state_update_bf9d8f51eaa98a9b55e0",
    "stp_hd64_paired_hpc2_s2w4_mb5_lean_bf16_i32": "cake_selective_state_update_7a283e0596e5009db58a",
    "stp_hd64_paired_hpc4_lean": "cake_selective_state_update_f8ebb15623f0fd11be1a",
    "stp_hd64_paired_hpc4_lean_bf16_i32": "cake_selective_state_update_c70704cfda5f53df5935",
    "stp_hd64_paired_hpc4_s2w4_lean": "cake_selective_state_update_215e5b54ac3db1884c38",
    "stp_hd64_paired_hpc4_s2w4_lean_bf16_i32": "cake_selective_state_update_c417c072b913bfb0b080",
    "stp_hd64_paired_hpc4_s4w2_lean": "cake_selective_state_update_b1306a044543e22b04f3",
    "stp_hd64_paired_hpc4_s4w2_lean_bf16_i32": "cake_selective_state_update_ffc75b2432b6aa1e1b7d",
    "stp_hd64_rows_bf16_dt4_od": "cake_selective_state_update_650060655f4353a908df",
    "stp_hd64_rows_bf16_dt4_od_bf16_i32": "cake_selective_state_update_092088eb41a05061f1ce",
    "stp_hd64_rows_bf16_dt4_xa_bf16_i32": "cake_selective_state_update_79b7ad05a6250777090c",
    "stp_hd64_rows_bf16_dt4_xa_od": "cake_selective_state_update_bb0dce3cad66d31fbd71",
    "stp_hd64_rows_bf16_dt8": "cake_selective_state_update_a1b343381deb84763333",
    "stp_hd64_rows_bf16_dt8_bf16_i32": "cake_selective_state_update_7df813cef15099ef6b6c",
    "stp_hd64_rows_fp32": "cake_selective_state_update_f75fba6195c641c8ea09",
    "stp_hd64_rows_fp32_dt1": "cake_selective_state_update_7bf2ab4137e58a5005ce",
    "stp_hd64_rows_fp32_dt1_bf16_i32": "cake_selective_state_update_0e370764097daff22a04",
    "stp_hd64_rows_fp32_dt1_l2_prefetch_bf16_i32": "cake_selective_state_update_e0a6f0a910168d91472d",
    "stp_hd64_rows_fp32_dt1_large": "cake_selective_state_update_c6359778c662f82edaea",
    "stp_hd64_rows_fp32_dt1_large_bf16_i32": "cake_selective_state_update_5efde2a65925e2c3c9d3",
    "stp_hd64_rows_fp32_dt1_large_sl2first": "cake_selective_state_update_93a0e0315a4697642884",
    "stp_hd64_rows_fp32_dt1_large_sl2first_bf16_i32": "cake_selective_state_update_37b861f7812f3eeb556a",
    "stp_hd64_rows_fp32_dt2_bf16_i32": "cake_selective_state_update_cf1848ae62ab6ca3b7f0",
    "stp_hd64_rows_fp32_dt4_bf16_i32": "cake_selective_state_update_7aa77bb2bcd225bc7c4c",
    "stp_hd64_rows_fp32_dt8": "cake_selective_state_update_2b07ca215b31742b79ab",
    "stp_hd64_rows_fp32_dt8_bf16_i32": "cake_selective_state_update_d3495f7b5362267e04c4",
}

_ARCH_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}
_ARCH_OF_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
_GRID_AXES = ("grid_x", "grid_y", "grid_z")
_MTP_HEADS_PER_CTA = 2
_MTP_RESIDENT_CTAS_PER_SM = 4
_MTP_DIRECT_MAX_RESIDENT_WAVES = 2
_DYNAMIC_MAX_TOKEN_STEPS = 8
# The six-token cache producer's (batch, step) strides of x, dt, B and C in the raw SGLang ABI: the fused
# projection's padded views the raw build folds.
_MTP_CACHE_RAW_STRIDES = (26112, 4352, 51072, 8512, 26112, 4352, 26112, 4352)
# The TMA-ring builds of the short program (64 heads of 8 groups, one or two folded tokens): eight heads of one
# group per CTA, two CTAs per SM.  Per (architecture, token count): the smallest batch, the largest batch (None: no
# bound) and the smallest fill of the ring's last wave of CTAs in quarters of a wave (whole waves always qualify).
_MTP_SHORT_RING_HEADS_PER_CTA = 8
_MTP_SHORT_RING_CTAS_PER_SM = 2
_MTP_SHORT_RING_DOMAIN: dict[tuple[str, int], tuple[int, Optional[int], int]] = {
    ("sm_100a", 1): (32, 96, 2),
    ("sm_100a", 2): (32, None, 1),
    ("sm_103a", 1): (64, 96, 2),
    ("sm_103a", 2): (32, None, 0),
}
# The two-token rings that reduce both tokens' projections together (``joint_reduce``; on sm_103a with each
# consumer's state base set before the head's first wait, ``head_base``; output and state bitwise the two-token
# ring's), by architecture, and per (architecture, SM count) the inclusive batch intervals in which a two-token call
# the ring serves runs the architecture's one instead, each batch measured faster than the ring.
_MTP_SHORT_RING_JR_PROGRAMS: dict[str, str] = {
    "sm_100a": "mtp_short_ring_heads64_groups8_t2_jr",
    "sm_103a": "mtp_short_ring_heads64_groups8_t2_jr_hb",
}
_MTP_SHORT_RING_JR_BATCHES: dict[tuple[str, int], tuple[tuple[int, int], ...]] = {
    ("sm_100a", 148): (
        (33, 35),
        (51, 65),
        (68, 68),
        (86, 86),
        (88, 101),
        (104, 104),
        (107, 108),
        (110, 111),
        (122, 138),
        (140, 141),
        (158, 168),
        (170, 170),
        (172, 178),
        (195, 202),
        (204, 206),
        (232, 237),
        (239, 245),
        (248, 248),
        (254, 256),
        (270, 270),
        (285, 285),
        (296, 296),
        (306, 306),
    ),
    ("sm_103a", 152): (
        (32, 46),
        (50, 50),
        (54, 74),
        (76, 78),
        (81, 81),
        (83, 106),
        (108, 108),
        (110, 110),
        (112, 118),
        (123, 148),
        (154, 154),
        (156, 184),
        (186, 188),
        (190, 256),
        (260, 260),
        (266, 266),
        (270, 270),
        (280, 280),
        (285, 285),
        (296, 296),
        (304, 304),
        (320, 320),
    ),
}


def _source_dir() -> Path:
    packaged = jit_env.FLASHINFER_CSRC_DIR / "cake_selective_state_update"
    if packaged.is_dir():
        return packaged
    return Path(__file__).resolve().parents[3] / "csrc" / "cake_selective_state_update"


def _include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.is_dir():
        return jit_env.FLASHINFER_INCLUDE_DIR
    return Path(__file__).resolve().parents[3] / "include"


@functools.cache
def define_digest(defines: Sequence[tuple[str, int]]) -> str:
    """Stable 16-hex digest of a define set (sorted ``NAME=value`` pairs)."""
    return hashlib.sha256(
        "|".join(f"{name}={value}" for name, value in sorted(defines)).encode()
    ).hexdigest()[:16]


def jit_spec(
    module: str, arch: str, defines: tuple[tuple[str, int], ...] = ()
) -> JitSpec:
    """Build specification of one delivered program for ``arch`` and its compile-time defines; a record
    listing ``arches`` is delivered for those architectures only."""
    record = MODULES[module]
    if arch not in record.get("arches", _ARCH_FLAGS):
        raise ValueError(f"{module} is not delivered for {arch}")
    source_dir = _source_dir()
    # The define set names the instantiation through its digest (spelling twelve defines out exceeds
    # the 255-byte file-name limit of the JIT cache); ``MODULES[module]["instantiations"]`` maps every
    # delivered digest back to its define values.
    suffix = f"_{define_digest(defines)}" if defines else ""
    return gen_jit_spec(
        name=f"{module}_{arch}{suffix}",
        sources=[source_dir / path for path in record["sources"]],
        extra_cuda_cflags=[
            *_ARCH_FLAGS[arch],
            *record["compile_flags"],
            *[f"-D{name}={value}" for name, value in defines],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[source_dir, source_dir.parent, _include_dir()],
        use_fast_math=False,
    )


def gen_cake_selective_state_update_modules(arch: str) -> list[JitSpec]:
    """Build specifications of every program delivered for ``arch`` and of every instantiation (the AOT table)."""
    specs: list[JitSpec] = []
    for module, record in MODULES.items():
        if arch not in record.get("arches", _ARCH_FLAGS):
            continue
        instantiations = record["instantiations"] or {"": {}}
        for values in instantiations.values():
            defines = tuple((name, int(values[name])) for name in record["defines"])
            specs.append(jit_spec(module, arch, defines))
    return specs


class _Program:
    """A loaded FFI entry bound to its argument plan (``buffer`` / ``parameter`` bindings and grid axes)."""

    __slots__ = ("_call", "_names", "_grid_slots")

    def __init__(self, call: Any, arg_plan: Sequence[Sequence[str]]):
        self._call = call
        names: list[Optional[str]] = []
        grid_slots: list[tuple[int, int]] = []
        for slot, (kind, name) in enumerate(arg_plan):
            if kind == "grid":
                names.append(None)
                grid_slots.append((slot, _GRID_AXES.index(name)))
            else:
                names.append(name)
        self._names = tuple(names)
        self._grid_slots = tuple(grid_slots)

    def launch(self, grid: tuple[int, int, int], bindings: dict[str, Any]) -> None:
        args = [bindings[name] if name is not None else 0 for name in self._names]
        for slot, axis in self._grid_slots:
            args[slot] = grid[axis]
        self._call(*args)


def _generated_program(
    program: str, arch: str, defines: tuple[tuple[str, int], ...]
) -> _Program:
    record = MODULES[PROGRAMS[program]]
    call = getattr(
        jit_spec(PROGRAMS[program], arch, defines).build_and_load(), record["ffi_entry"]
    )
    return _Program(call, record["arg_plan"])


@functools.cache
def _program(program: str, arch: str, defines: tuple[tuple[str, int], ...]) -> _Program:
    return _generated_program(program, arch, defines)


@functools.cache
def _device_arch(device_index: int) -> Optional[str]:
    return _ARCH_OF_CAPABILITY.get(
        get_compute_capability(torch.device("cuda", device_index))
    )


@functools.cache
def _sm_count(device_index: int) -> int:
    return int(get_device_sm_count(torch.device("cuda", device_index)))


class RoutePlan(NamedTuple):
    """One resolved launch: program, compile-time defines, grid and kernel bindings."""

    program: str
    defines: tuple[tuple[str, int], ...]
    grid: tuple[int, int, int]
    bindings: dict[str, Any]
    arch: str
    device_index: int


def _coefficients(dt, A, D, dt_bias, x_shape, nheads: int, dim: int, dstate: int):
    """Compact per-head views of the coefficient tensors, or ``None`` when a shape or stride is not the broadcast layout.

    The programs read one ``A`` / ``D`` / ``dt_bias`` value per head and one ``dt`` value per row (zero strides on
    the broadcast dims). One ``shape`` and one ``stride()`` read per tensor: the plan path is attribute traffic.
    """
    dt_stride, a_stride, d_stride, bias_stride = (
        dt.stride(),
        A.stride(),
        D.stride(),
        dt_bias.stride(),
    )
    if (
        dt.shape != x_shape
        or A.shape != (nheads, dim, dstate)
        or D.shape != (nheads, dim)
        or dt_bias.shape != (nheads, dim)
        or dt_stride[-1] != 0
        or a_stride[1:] != (0, 0)
        or d_stride[-1] != 0
        or bias_stride[-1] != 0
    ):
        return None
    return (
        dt.as_strided(x_shape[:-1], dt_stride[:-1]),
        A.as_strided((nheads,), (a_stride[0],)),
        D.as_strided((nheads,), (d_stride[0],)),
        dt_bias.as_strided((nheads,), (bias_stride[0],)),
    )


def rows_defines(
    coefficient_dtype: torch.dtype,
    index_dtype: torch.dtype,
    prefetch_rows: int,
    dim_tiles: int,
) -> tuple[tuple[str, int], ...]:
    """Defines of one row-owner tile instantiation: the storage ABI and the plan's tile geometry."""
    return (
        ("COEFFICIENT_BF16", int(coefficient_dtype == torch.bfloat16)),
        ("INDEX_I32", int(index_dtype == torch.int32)),
        ("PREFETCH_ROWS", int(prefetch_rows)),
        ("DIM_TILES", int(dim_tiles)),
    )


def cache_defines(
    coefficient_dtype: torch.dtype,
    index_dtype: torch.dtype,
    nheads: int,
    ngroups: int,
    x_stride: Sequence[int],
    dt_stride: Sequence[int],
    b_stride: Sequence[int],
    c_stride: Sequence[int],
) -> tuple[tuple[str, int], ...]:
    """Compile-time defines of the six-token cache program for one launch.

    BF16 ``dt``/``D``/``dt_bias`` storage and int32 slot tables (the SGLang
    ABI) select the folded loads, and that ABI also folds its model constants
    ``nheads`` / ``ngroups``; the canonical FP32/int64 storage keeps runtime
    extents.  The projection (batch, step) strides are always folded: one
    instantiation per projection layout.
    """
    raw_sglang = coefficient_dtype == torch.bfloat16 and index_dtype == torch.int32
    return (
        ("COEFFICIENT_BF16", int(coefficient_dtype == torch.bfloat16)),
        ("INDEX_I32", int(index_dtype == torch.int32)),
        ("NHEADS_STATIC", int(nheads) if raw_sglang else 0),
        ("NGROUPS_STATIC", int(ngroups) if raw_sglang else 0),
        ("X_BATCH_STRIDE", int(x_stride[0])),
        ("X_STEP_STRIDE", int(x_stride[1])),
        ("DT_BATCH_STRIDE", int(dt_stride[0])),
        ("DT_STEP_STRIDE", int(dt_stride[1])),
        ("B_BATCH_STRIDE", int(b_stride[0])),
        ("B_STEP_STRIDE", int(b_stride[1])),
        ("C_BATCH_STRIDE", int(c_stride[0])),
        ("C_STEP_STRIDE", int(c_stride[1])),
    )


def mtp_cache_program(
    defines: tuple[tuple[str, int], ...],
    *,
    nheads: int,
    ngroups: int,
    dense_slots: bool,
) -> tuple[str, tuple[tuple[str, int], ...]]:
    """``(program, defines)`` of one six-token cache launch from its define set (:func:`cache_defines`).

    The three projection layouts the cache producer ships prebuilt run its rendered builds (cached states stored
    ahead of the reduction, outputs after the last token), one build per layout: the canonical ABI on 64 dense
    heads with one group or eight, and the raw SGLang ABI on 64 heads with one group at the fused projection's
    strides.  A canonical call with exactly 64 heads, the layout's group count and slot-contiguous state and
    intermediate state (``dense_slots``) runs the layout's build that also folds the head and group counts and the
    slot strides.  Every other layout keeps the carried program instantiated under its defines.
    """
    raw_coefficients, raw_indices, nheads_static, ngroups_static, *strides = (
        value for _, value in defines
    )
    if raw_coefficients and raw_indices:
        if (nheads_static, ngroups_static) == (64, 1) and tuple(
            strides
        ) == _MTP_CACHE_RAW_STRIDES:
            return "mtp_cache_bf16_i32", ()
    elif not (raw_coefficients or raw_indices or nheads_static or ngroups_static):
        x_dt = (6 * 64 * 64, 64 * 64, 6 * 64, 64)
        for groups in (1, 8):
            rows = (6 * groups * 128, groups * 128)
            if tuple(strides) == (*x_dt, *rows, *rows):
                if dense_slots and (nheads, ngroups) == (64, groups):
                    return f"mtp_cache_groups{groups}_heads64", ()
                return f"mtp_cache_groups{groups}", ()
    return "mtp_cache_c4_t6", defines


def mtp_short_program(
    batch_size: int,
    nheads: int,
    ngroups: int,
    token_steps: int,
    sm_count: int,
    arch: str,
) -> tuple[str, int]:
    """``(program, head_tiles)`` of one short (one- or two-token, 128 x 128) launch; ``head_tiles`` is 0 for the
    one-CTA-per-head builds.

    On 64 heads of 8 groups the TMA-ring build of the launch's token count runs inside its domain
    (``_MTP_SHORT_RING_DOMAIN``): a batch in the range of the architecture and token count, and a last wave of
    eight-head CTAs (two per SM) filled at least to the bound, or a whole number of waves; a two-token call the ring
    serves at a batch of ``_MTP_SHORT_RING_JR_BATCHES`` for the architecture and SM count runs the architecture's
    joint-reduction ring (``_MTP_SHORT_RING_JR_PROGRAMS``).  Smaller batches, and batches that leave the ring's last
    wave nearly empty, run the build folding those counts with one CTA per batch row and head.  Every other head and
    group count runs the prefetch build.
    """
    if (nheads, ngroups) != (64, 8):
        return "mtp_short_prefetch", 0
    bounds = _MTP_SHORT_RING_DOMAIN.get((arch, token_steps))
    if bounds is not None:
        low, high, quarters = bounds
        if batch_size >= low and (high is None or batch_size <= high):
            head_tiles = -(-(nheads // ngroups) // _MTP_SHORT_RING_HEADS_PER_CTA)
            wave = _MTP_SHORT_RING_CTAS_PER_SM * sm_count
            tail = batch_size * ngroups * head_tiles % wave
            if tail == 0 or 4 * tail >= quarters * wave:
                jr = _MTP_SHORT_RING_JR_BATCHES.get((arch, sm_count), ())
                if token_steps == 2 and any(
                    low <= batch_size <= high for low, high in jr
                ):
                    return _MTP_SHORT_RING_JR_PROGRAMS[arch], head_tiles
                return f"mtp_short_ring_heads64_groups8_t{token_steps}", head_tiles
    return "mtp_short_prefetch_heads64_groups8", 0


# The token counts at which a one-row launch of the 16-lane dynamic build runs its pipelined-reduction build, per
# architecture.
_DYNAMIC_PR_TOKEN_STEPS: dict[str, tuple[int, ...]] = {
    "sm_100a": (4, 5, 6, 7, 8),
    "sm_103a": (5, 6, 7, 8),
}


def dynamic_program(batch_size: int, token_steps: int, arch: str) -> str:
    """The build of one launch of the dynamic program: below eight batch rows a multi-token call runs 16 lanes per
    state row (eight dimension tiles per head), otherwise 8 lanes per row (four tiles).  A one-row launch of the 16-lane
    build at a token count ``_DYNAMIC_PR_TOKEN_STEPS`` lists for the architecture runs its pipelined-reduction build
    (``dynamic_lanes16_pr``: each token's shuffle reduction and output store run in the next token's body, after its
    update; output and state bitwise the plain build's)."""
    if token_steps < 2 or batch_size >= 8:
        return "dynamic_lanes8"
    if batch_size == 1 and token_steps in _DYNAMIC_PR_TOKEN_STEPS.get(arch, ()):
        return "dynamic_lanes16_pr"
    return "dynamic_lanes16"


def _plan_mtp_cache(
    *,
    state,
    x,
    dt,
    A,
    B,
    C,
    D,
    dt_bias,
    output,
    source,
    intermediate,
    intermediate_indices,
    nheads,
    ngroups,
    pad_slot_id,
    arch,
    device_index,
) -> Optional[RoutePlan]:
    """Six-token D64/N128 cache program on any projection layout with dense head/group rows.

    One ``shape``/``stride()`` read per tensor: the per-call host path of this
    route is Python attribute traffic, so every metadata value is read once.
    """
    x_shape, x_stride = x.shape, x.stride()
    batch_size = x_shape[0]
    b_shape, b_stride, c_stride = B.shape, B.stride(), C.stride()
    dt_stride, a_stride, d_stride, bias_stride = (
        dt.stride(),
        A.stride(),
        D.stride(),
        dt_bias.stride(),
    )
    if (
        x_shape[1:] != (6, nheads, 64)
        or dt.shape != x_shape
        or b_shape != (batch_size, 6, ngroups, 128)
        or C.shape != b_shape
        or output.shape != x_shape
        or state.shape[1:] != (nheads, 64, 128)
        or A.shape != (nheads, 64, 128)
        or D.shape != (nheads, 64)
        or dt_bias.shape != (nheads, 64)
        or intermediate.shape[1:] != (6, nheads, 64, 128)
        or source.shape != (batch_size,)
        or intermediate_indices.shape != (batch_size,)
        or output.dtype != torch.bfloat16
        or intermediate.dtype != state.dtype
        or dt.dtype != D.dtype
        or dt.dtype != dt_bias.dtype
    ):
        return None
    # Broadcast coefficients (one element per head, unit head stride) and dense projection rows
    # (unit last stride, head/group stride = row length, 16-byte aligned batch/step strides).
    if (
        dt_stride[3] != 0
        or dt_stride[2] != 1
        or a_stride != (1, 0, 0)
        or d_stride != (1, 0)
        or bias_stride != (1, 0)
        or x_stride[3] != 1
        or x_stride[2] != 64
        or (x_stride[0] | x_stride[1]) & 7
        or b_stride[3] != 1
        or b_stride[2] != 128
        or (b_stride[0] | b_stride[1]) & 7
        or c_stride[3] != 1
        or c_stride[2] != 128
        or (c_stride[0] | c_stride[1]) & 7
        or (x.data_ptr() | B.data_ptr() | C.data_ptr()) & 15
    ):
        return None
    if not (
        state.is_contiguous()
        and output.is_contiguous()
        and intermediate.is_contiguous()
        and source.is_contiguous()
        and intermediate_indices.is_contiguous()
    ):
        return None
    state_stride_slot, intermediate_stride_slot = (
        state.stride(0),
        intermediate.stride(0),
    )
    bindings = dict(
        state=state,
        x=x,
        B=B,
        C=C,
        output=output,
        intermediate_state=intermediate,
        dt_addr=dt.data_ptr(),
        a_addr=A.data_ptr(),
        d_addr=D.data_ptr(),
        dt_bias_addr=dt_bias.data_ptr(),
        state_batch_indices_addr=source.data_ptr(),
        intermediate_state_indices_addr=intermediate_indices.data_ptr(),
        nheads=nheads,
        ngroups=ngroups,
        state_stride_slot=state_stride_slot,
        intermediate_stride_slot=intermediate_stride_slot,
        pad_slot_id=pad_slot_id,
    )
    # A contiguous state of one slot may spell any slot stride; the folded builds need the dense one.
    program, defines = mtp_cache_program(
        cache_defines(
            dt.dtype,
            source.dtype,
            nheads,
            ngroups,
            x_stride,
            dt_stride,
            b_stride,
            c_stride,
        ),
        nheads=nheads,
        ngroups=ngroups,
        dense_slots=state_stride_slot == nheads * 64 * 128
        and intermediate_stride_slot == 6 * nheads * 64 * 128,
    )
    return RoutePlan(
        program,
        defines,
        (batch_size, nheads, 4),
        bindings,
        arch,
        device_index,
    )


def _dense_rows(shape, stride, batch_size: int, rows: int, row: int) -> Optional[int]:
    """Batch stride of a ``(batch, rows, row)`` view with dense rows, or ``None``."""
    if shape != (batch_size, rows, row) or stride[2] != 1 or stride[1] != row:
        return None
    if batch_size > 1 and stride[0] < rows * row:
        return None
    return stride[0]


def _rows_batch_stride(
    tensor: torch.Tensor, batch_size: int, rows: int, row: int
) -> Optional[int]:
    """Batch stride the paired row-major programs read a ``(batch, rows, row)`` tensor at: dense rows at a
    padded batch stride, or a contiguous tensor (whatever the stride of a size-1 axis), else ``None``."""
    shape = tensor.shape
    if shape != (batch_size, rows, row):
        return None
    if tensor.is_contiguous():
        return rows * row
    return _dense_rows(shape, tensor.stride(), batch_size, rows, row)


def _work_tiles(
    row_groups: int, program_heads_per_group: int, heads_per_cta: int
) -> int:
    """CTAs of one launch: one per tile of ``heads_per_cta`` program heads of one (batch, group) row."""
    return row_groups * -(-program_heads_per_group // heads_per_cta)


def _busiest_sm_heads(
    row_groups: int, program_heads_per_group: int, heads_per_cta: int, sm_count: int
) -> int:
    """Head slots of the busiest SM when the launch runs ``heads_per_cta`` program heads per CTA."""
    ctas = _work_tiles(row_groups, program_heads_per_group, heads_per_cta)
    return -(-ctas // sm_count) * heads_per_cta


def hd64_paired_program(
    row_groups: int, program_heads_per_group: int, sm_count: int, raw_abi: bool
) -> tuple[str, int]:
    """``(program, heads_per_cta)`` of the paired headdim-64 route for one launch.

    ``row_groups`` is batch x groups; each group holds ``program_heads_per_group`` program heads (two 64-row
    heads each), so the launch covers ``row_groups x program_heads_per_group`` program heads and runs one CTA per
    tile of ``heads_per_cta`` heads of one group.  Below three program heads per SM a CTA owns one program head
    (the build with the folded ratio at eight program heads per group); below six it owns two, at two CTAs per SM
    while those CTAs fit two per SM; below twelve it owns four unless two put fewer head slots on the busiest SM;
    from twelve on it owns two.  ``raw_abi`` selects the build for BF16 ``dt`` / ``D`` / ``dt_bias`` and int32
    slot tables.
    """
    program_heads = row_groups * program_heads_per_group
    if program_heads < 3 * sm_count:
        heads_per_cta = 1
        variant = "hpc1_ratio8" if program_heads_per_group == 8 else "hpc1"
    elif program_heads < 6 * sm_count:
        heads_per_cta = 2
        fits_two = _work_tiles(row_groups, program_heads_per_group, 2) <= 2 * sm_count
        variant = "hpc2_mb2" if fits_two else "hpc2"
    elif program_heads < 12 * sm_count and _busiest_sm_heads(
        row_groups, program_heads_per_group, 2, sm_count
    ) >= _busiest_sm_heads(row_groups, program_heads_per_group, 4, sm_count):
        heads_per_cta, variant = 4, "hpc4"
    else:
        heads_per_cta, variant = 2, "hpc2"
    return f"stp_hd64_paired_{variant}{'_bf16_i32' if raw_abi else ''}", heads_per_cta


def hd64_unsplit_program(
    row_groups: int,
    program_heads_per_group: int,
    sm_count: int,
    arch: str,
    raw_abi: bool,
) -> tuple[str, int]:
    """``(program, heads_per_cta)`` of the unsplit paired headdim-64 build for one launch: the build
    :func:`hd64_paired_program` gives it, except that a launch it gives the two-head build at three CTAs per SM, with
    groups of a multiple of four program heads and that build's CTAs inside the architecture's band
    (``_HD64_HPC4_BAND_HUNDREDTHS``: from 3.05 to 3.46 waves of three CTAs per SM on sm_100a, from 3.09 to 3.37 on
    sm_103a), runs the four-head build."""
    program, heads_per_cta = hd64_paired_program(
        row_groups, program_heads_per_group, sm_count, raw_abi
    )
    suffix = _RAW_ABI_SUFFIX if raw_abi else ""
    band = _HD64_HPC4_BAND_HUNDREDTHS.get(arch)
    if (
        band is None
        or program != f"stp_hd64_paired_hpc2{suffix}"
        or program_heads_per_group % 4
    ):
        return program, heads_per_cta
    low, high = band
    tiles = _work_tiles(row_groups, program_heads_per_group, 2)
    if low * 3 * sm_count <= 100 * tiles <= high * 3 * sm_count:
        return f"stp_hd64_paired_hpc4{suffix}", 4
    return program, heads_per_cta


def dim128_program(heads_per_group: int, gated: bool) -> tuple[str, int]:
    """``(program, heads_per_cta)`` of the headdim-128 / dstate-128 single-token route: the z-gated build at four
    CTAs per SM for a call with ``z`` (outside the persistent build's domain, :func:`dim128_balanced_units`),
    otherwise the early-input build at one head per CTA (four CTAs per SM) for one head per group and at two heads
    per CTA for more."""
    if gated:
        return "stp_dim128_z_gated", 2
    if heads_per_group == 1:
        return "stp_dim128_hpc1_mb4", 1
    return "stp_dim128_hpc2", 2


# The persistent build of the z-gated 128x128 single-token program: per (architecture, SM count) the inclusive
# intervals of ``batch x nheads`` in which a call with ``z`` runs it, and its CTAs per SM.
_DIM128_BALANCED_UNITS: dict[tuple[str, int], tuple[tuple[int, int], ...]] = {
    ("sm_100a", 148): ((64, 384), (1280, 1664), (2560, 2816), (3328, 4096)),
    ("sm_103a", 152): ((64, 448), (1280, 1792), (2560, 3328), (3840, 4352)),
}
_DIM128_BALANCED_CTAS_PER_SM = 3


def dim128_balanced_units(
    batch_size: int, nheads: int, sm_count: int, arch: str
) -> int:
    """The heads (``batch_size x nheads``) a z-gated headdim-128 / dstate-128 single-token launch hands the
    persistent build, or 0 when the launch keeps the z-gated build: the build runs inside the measured intervals
    of the architecture at its SM count (``_DIM128_BALANCED_UNITS``), one launch of at most three CTAs per SM, each
    CTA streaming whole heads."""
    units = batch_size * nheads
    intervals = _DIM128_BALANCED_UNITS.get((arch, sm_count), ())
    return units if any(low <= units <= high for low, high in intervals) else 0


# The L2 read-ahead build of the 128x128 FP32 single-token program: the inclusive range of ``batch x nheads`` in
# which a launch runs it, on either architecture.
_DIM128_FP32_L2_PREFETCH_BATCH_HEADS = (1408, 6656)


def dim128_fp32_program(batch_heads: int) -> str:
    """The build of one headdim-128 / dstate-128 FP32 single-token launch over ``batch_heads`` (``batch x nheads``)
    heads: inside ``_DIM128_FP32_L2_PREFETCH_BATCH_HEADS`` the build that also reads each CTA's state tile 2480 CTAs
    ahead into L2 (the plain build's grid, bindings and arithmetic), outside it the plain build."""
    low, high = _DIM128_FP32_L2_PREFETCH_BATCH_HEADS
    if low <= batch_heads <= high:
        return "stp_dim128_fp32_l2_prefetch"
    return "stp_dim128_fp32"


# The one-token TMA ring of the short program built in the single-token programs' arithmetic: per (architecture, SM
# count) the inclusive batch intervals in which a BF16 single-token call of 64 heads in 8 groups runs it.
_DIM128_STP_RING_BATCHES: dict[tuple[str, int], tuple[tuple[int, int], ...]] = {
    ("sm_100a", 148): ((16, 18), (28, 29), (31, 37), (56, 67), (70, 71)),
}
_DIM128_STP_RING_PROGRAM = "mtp_short_ring_heads64_groups8_t1_stp"
# The same ring with the gate ``z`` (64 heads in 8 groups): per (architecture, SM count) the batches at which a BF16
# single-token call of that shape with ``z`` runs it.
_DIM128_STP_RING_Z_BATCHES: dict[tuple[str, int], tuple[int, ...]] = {
    ("sm_100a", 148): (64,),
}
_DIM128_STP_RING_Z_PROGRAM = "mtp_short_ring_heads64_groups8_t1_stp_z"


def dim128_stp_ring(
    batch_size: int, nheads: int, ngroups: int, sm_count: int, arch: str
) -> bool:
    """Whether a headdim-128 / dstate-128 BF16 single-token launch of ``batch_size`` rows without ``z`` runs the
    one-token ring of 64 heads: 64 heads in 8 groups, and a batch inside the intervals ``_DIM128_STP_RING_BATCHES``
    lists for the architecture at its SM count (none for another architecture or SM count)."""
    if (nheads, ngroups) != (64, 8):
        return False
    intervals = _DIM128_STP_RING_BATCHES.get((arch, sm_count), ())
    return any(low <= batch_size <= high for low, high in intervals)


def dim128_stp_ring_program(
    batch_size: int,
    nheads: int,
    ngroups: int,
    sm_count: int,
    arch: str,
    gated: bool,
) -> Optional[str]:
    """The one-token ring a headdim-128 / dstate-128 BF16 single-token launch of ``batch_size`` rows runs, or
    ``None``: without ``z`` (``gated`` false), the ring of 64 heads (:func:`dim128_stp_ring`); with ``z``, for 64
    heads in 8 groups, the z-gated ring at a batch ``_DIM128_STP_RING_Z_BATCHES`` lists for the architecture at its
    SM count (none for another architecture or SM count)."""
    if gated:
        gated_batches = _DIM128_STP_RING_Z_BATCHES.get((arch, sm_count), ())
        if (nheads, ngroups) == (64, 8) and batch_size in gated_batches:
            return _DIM128_STP_RING_Z_PROGRAM
        return None
    if dim128_stp_ring(batch_size, nheads, ngroups, sm_count, arch):
        return _DIM128_STP_RING_PROGRAM
    return None


def dim128_stp_ring_view(
    *,
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor],
    dt_bias: torch.Tensor,
    output: torch.Tensor,
    source: torch.Tensor,
    dst_state_batch_indices: Optional[torch.Tensor],
    intermediate_states_buffer: Optional[torch.Tensor],
    nheads: int,
    ngroups: int,
    dt_softplus: bool,
    disable_state_update: bool,
    pad_slot_id: int,
    arch: str,
    device_index: int,
    sms: int,
) -> Optional[RoutePlan]:
    """The one-token ring launch of a canonical BF16 single-token call, or ``None`` when the call keeps the
    single-token programs.

    The ring runs a call :func:`dim128_stp_ring_program` admits without destination table, cached states or a
    disabled update, on the call's dense one-token view: ``x`` / ``B`` / ``C`` / ``output`` (and ``z`` of a gated
    call) contiguous, ``dt`` the broadcast of a contiguous ``[batch, nheads]`` block, ``A`` / ``D`` / ``dt_bias`` one
    contiguous value per head, a contiguous state pool and slot table.  Its output and state are bitwise the
    single-token programs'.
    """
    batch_size = x.shape[0]
    program = dim128_stp_ring_program(
        batch_size, nheads, ngroups, sms, arch, z is not None
    )
    if (
        program is None
        or dst_state_batch_indices is not None
        or intermediate_states_buffer is not None
        or disable_state_update
    ):
        return None
    dim = 128
    x_shape = (batch_size, nheads, dim)
    b_shape = (batch_size, ngroups, dim)
    if (
        x.shape != x_shape
        or B.shape != b_shape
        or C.shape != b_shape
        or output.shape != x_shape
        or output.dtype != torch.bfloat16
        or source.shape != (batch_size,)
        or dt.shape != x_shape
        or dt.stride() != (nheads, 1, 0)
        or A.shape != (nheads, dim, dim)
        or A.stride() != (1, 0, 0)
        or D.shape != (nheads, dim)
        or D.stride() != (1, 0)
        or dt_bias.shape != (nheads, dim)
        or dt_bias.stride() != (1, 0)
        or not (
            x.is_contiguous()
            and B.is_contiguous()
            and C.is_contiguous()
            and output.is_contiguous()
            and state.is_contiguous()
            and source.is_contiguous()
        )
        or (
            z is not None
            and (
                z.shape != x_shape or z.dtype != torch.bfloat16 or not z.is_contiguous()
            )
        )
    ):
        return None
    head_tiles = -(-(nheads // ngroups) // _MTP_SHORT_RING_HEADS_PER_CTA)
    gate = {} if z is None else {"z": z.view(batch_size, 1, nheads, dim)}
    bindings = dict(
        state_tma=state,
        x=x.view(batch_size, 1, nheads, dim),
        dt=dt.as_strided((batch_size, 1, nheads), (nheads, nheads, 1)),
        A=A.as_strided((nheads,), (1,)),
        B=B.view(batch_size, 1, ngroups, dim),
        C=C.view(batch_size, 1, ngroups, dim),
        D=D.as_strided((nheads,), (1,)),
        **gate,
        dt_bias=dt_bias.as_strided((nheads,), (1,)),
        output=output.view(batch_size, 1, nheads, dim),
        state_batch_indices=source,
        batch_size=batch_size,
        nheads=nheads,
        dim=dim,
        dstate=dim,
        ngroups=ngroups,
        token_steps=1,
        head_tiles=head_tiles,
        dt_softplus=int(dt_softplus),
        pad_slot_id=pad_slot_id,
    )
    # The TMA ring: a CTA per (batch row, group, tile of heads), the state a tensor map of the slot table.
    return RoutePlan(
        program,
        (),
        (batch_size * ngroups * head_tiles, 1, 1),
        bindings,
        arch,
        device_index,
    )


# CTAs per SM of the two-head build from which a launch runs that build split two ways at five CTAs per SM, per
# architecture: headdim 64, and headdim 128 (sm_100a only).
_HD64_SPLIT_MB5_MIN_CTAS_PER_SM = {"sm_100a": 24, "sm_103a": 48}
# The lower band of that split at headdim 64, per architecture: tenths of a CTA per SM of the two-head build from
# which (inclusive) and below which a launch runs it.
_HD64_SPLIT_MB5_BAND_TENTHS = {"sm_100a": (60, 75), "sm_103a": (60, 75)}
# The two-head two-CTA build's launches (``hpc2_mb2``) that run two heads per CTA split two ways (the build of the
# launches below three program heads per SM; two CTAs per tile, four per SM), per architecture: the band of that
# build's CTAs per SM, in hundredths, inclusive (sm_103a: 1.68 to 1.90, one wave of the split grid).
_HD64_C2_BAND_HUNDREDTHS = {"sm_103a": (168, 190)}
_DIM128_SPLIT_MB5_MIN_CTAS_PER_SM = {"sm_100a": 48}
# The headdim-64 launches the two-head build serves at three CTAs per SM from six CTAs per SM that run four heads per
# CTA split two ways instead (the same CTAs, four per SM) when that grid ends in a nearly full wave, per architecture:
# the waves of four CTAs per SM the grid may fill, and the hundredths of a wave below each of them above which its
# last wave must end.
_HD64_C4_WAVES = {"sm_100a": (3, 4), "sm_103a": (3, 4)}
_HD64_C4_TAIL_HUNDREDTHS = {"sm_100a": 25, "sm_103a": 40}
# The band of the two-head build at three CTAs per SM in which a headdim-64 launch runs the four-head build instead,
# per architecture: inclusive bounds in hundredths of a wave of three CTAs per SM.
_HD64_HPC4_BAND_HUNDREDTHS = {"sm_100a": (305, 346), "sm_103a": (309, 337)}
# The builds whose consumers skip the proxy fence after their wait on a full state stage, per architecture: a
# launch of one of these programs (either ABI of a headdim-64 build) runs its ``_lean`` build, whose output and state
# are bitwise the plain build's.
_LEAN_PROGRAMS: dict[str, frozenset[str]] = {
    "sm_100a": frozenset(
        {
            "stp_dim128_hpc2",
            "stp_dim128_hpc2_s2w4_mb5",
            "stp_dim128_z_balanced",
            "stp_hd64_paired_hpc2",
            "stp_hd64_paired_hpc2_mb2",
            "stp_hd64_paired_hpc2_ratio8_s2w4",
            "stp_hd64_paired_hpc2_s2w4",
            "stp_hd64_paired_hpc2_s2w4_mb5",
            "stp_hd64_paired_hpc4",
            "stp_hd64_paired_hpc4_s2w4",
            "stp_hd64_paired_hpc4_s4w2",
        }
    ),
    "sm_103a": frozenset(
        {
            "stp_dim128_hpc1_s2w4",
            "stp_dim128_hpc2",
            "stp_dim128_z_balanced",
            "stp_hd64_paired_hpc1_ratio8_s4w2",
            "stp_hd64_paired_hpc1_s4w2",
            "stp_hd64_paired_hpc2_mb2",
            "stp_hd64_paired_hpc2_ratio8_s2w4",
            "stp_hd64_paired_hpc2_s2w4",
            "stp_hd64_paired_hpc2_s2w4_mb5",
            "stp_hd64_paired_hpc4",
            "stp_hd64_paired_hpc4_s2w4",
            "stp_hd64_paired_hpc4_s4w2",
        }
    ),
}
_RAW_ABI_SUFFIX = "_bf16_i32"
# Programs that run their ``_lean`` build on an architecture only inside a band of the launch's CTAs per SM, in
# hundredths, inclusive: the two-head paired build at three CTAs per SM on sm_103a from 6.10 to 26.95 CTAs per SM.
_LEAN_BAND_HUNDREDTHS: dict[str, dict[str, tuple[int, int]]] = {
    "sm_103a": {"stp_hd64_paired_hpc2": (610, 2695)},
}


def lean_program(program: str, arch: str, ctas: int = 0, sm_count: int = 0) -> str:
    """The build that runs a launch of ``program`` on ``arch``: its ``_lean`` build (``<program>_lean``, the raw ABI
    suffix kept last) when the architecture lists the program in ``_LEAN_PROGRAMS``, or when the launch's ``ctas`` on
    ``sm_count`` SMs fall inside the program's band of ``_LEAN_BAND_HUNDREDTHS``, else ``program``."""
    raw_abi = program.endswith(_RAW_ABI_SUFFIX)
    base = program.removesuffix(_RAW_ABI_SUFFIX)
    if base not in _LEAN_PROGRAMS.get(arch, ()):
        band = _LEAN_BAND_HUNDREDTHS.get(arch, {}).get(base)
        if (
            band is None
            or sm_count <= 0
            or not band[0] * sm_count <= 100 * ctas <= band[1] * sm_count
        ):
            return program
    return f"{base}_lean{_RAW_ABI_SUFFIX if raw_abi else ''}"


def hd64_split_program(
    row_groups: int,
    program_heads_per_group: int,
    sm_count: int,
    arch: str,
    raw_abi: bool,
) -> Optional[tuple[str, int, int]]:
    """``(program, heads_per_cta, p_split)`` of the P-split paired headdim-64 build for one launch, or ``None``
    when the launch keeps the build :func:`hd64_paired_program` gives it (below two program heads per SM, the
    row-owner tile).

    A P-split build runs ``p_split`` CTAs on every work tile, each owning ``128 / p_split`` state rows of the
    tile's program heads with its own consumer warps, at a fixed number of CTAs per SM; output and state are
    bitwise those of the unsplit build.  The split follows the build :func:`hd64_paired_program` gives the
    launch and that build's CTAs (``tiles``):

    - one head per CTA: from one to two program heads per SM split four ways; above two, two heads per CTA
      split two ways when the groups hold an even number of program heads (both with the folded ratio at eight
      program heads per group);
    - four heads per CTA: split two ways up to two ``tiles`` per SM;
    - two heads per CTA at two CTAs per SM: no split, except inside the architecture's band of that build's CTAs per
      SM (``_HD64_C2_BAND_HUNDREDTHS``, sm_103a 1.68 to 1.90) with groups of an even number of program heads: two
      heads per CTA split two ways (four CTAs per SM, the folded ratio at eight program heads per group);
    - two heads per CTA at three CTAs per SM, below six ``tiles`` per SM and groups of a multiple of four
      program heads: four heads per CTA split two ways up to three ``tiles`` per SM, above that split four ways
      at nine CTAs per SM while the launch holds at most nine program heads per SM;
    - two heads per CTA at three CTAs per SM from six ``tiles`` per SM: split two ways at five CTAs per SM from
      6 to 7.5 ``tiles`` per SM and from 24 (sm_100a) or 48 (sm_103a) ``tiles`` per
      SM; with groups of a multiple of four program heads, four heads per CTA split two ways (``tiles`` CTAs, four
      per SM) when those CTAs fill all but the last quarter (sm_100a) or 40 % (sm_103a) of a wave of four CTAs per
      SM up to a whole third or fourth wave (``_HD64_C4_WAVES``, ``_HD64_C4_TAIL_HUNDREDTHS``).

    ``raw_abi`` selects the build for BF16 ``dt`` / ``D`` / ``dt_bias`` and int32 slot tables.
    """
    suffix = "_bf16_i32" if raw_abi else ""
    program_heads = row_groups * program_heads_per_group
    if program_heads < 3 * sm_count:
        ratio8 = "_ratio8" if program_heads_per_group == 8 else ""
        if program_heads < sm_count:
            return None
        if program_heads <= 2 * sm_count:
            return f"stp_hd64_paired_hpc1{ratio8}_s4w2{suffix}", 1, 4
        if program_heads_per_group % 2:
            return None
        return f"stp_hd64_paired_hpc2{ratio8}_s2w4{suffix}", 2, 2
    unsplit, heads_per_cta = hd64_paired_program(
        row_groups, program_heads_per_group, sm_count, raw_abi
    )
    tiles = _work_tiles(row_groups, program_heads_per_group, heads_per_cta)
    c2 = _HD64_C2_BAND_HUNDREDTHS.get(arch)
    if (
        c2 is not None
        and unsplit == f"stp_hd64_paired_hpc2_mb2{suffix}"
        and program_heads_per_group % 2 == 0
        and c2[0] * sm_count <= 100 * tiles <= c2[1] * sm_count
    ):
        ratio8 = "_ratio8" if program_heads_per_group == 8 else ""
        return f"stp_hd64_paired_hpc2{ratio8}_s2w4{suffix}", 2, 2
    if heads_per_cta == 4:
        if tiles > 2 * sm_count:
            return None
        return f"stp_hd64_paired_hpc4_s2w4{suffix}", 4, 2
    if program_heads < 6 * sm_count and tiles <= 2 * sm_count:
        return None
    if tiles < 6 * sm_count:
        if program_heads_per_group % 4:
            return None
        if tiles <= 3 * sm_count:
            return f"stp_hd64_paired_hpc4_s2w4{suffix}", 4, 2
        if program_heads > 9 * sm_count:
            return None
        return f"stp_hd64_paired_hpc4_s4w2{suffix}", 4, 4
    tail = _HD64_C4_TAIL_HUNDREDTHS.get(arch)
    if (
        tail is not None
        and program_heads_per_group % 4 == 0
        and any(
            4 * sm_count * (100 * waves - tail) < 100 * tiles <= 400 * sm_count * waves
            for waves in _HD64_C4_WAVES[arch]
        )
    ):
        return f"stp_hd64_paired_hpc4_s2w4{suffix}", 4, 2
    low, high = _HD64_SPLIT_MB5_BAND_TENTHS[arch]
    if (
        low * sm_count <= 10 * tiles < high * sm_count
        or tiles >= _HD64_SPLIT_MB5_MIN_CTAS_PER_SM[arch] * sm_count
    ):
        return f"stp_hd64_paired_hpc2_s2w4_mb5{suffix}", 2, 2
    return None


def dim128_split_program(
    row_groups: int, heads_per_group: int, sm_count: int, arch: str, gated: bool
) -> Optional[tuple[str, int, int]]:
    """``(program, heads_per_cta, p_split)`` of the P-split headdim-128 / dstate-128 build for one launch, or
    ``None`` when the launch keeps the build :func:`dim128_program` gives it.  At one head per group the one-head
    build splits two ways up to 3.5 CTAs per SM; from 48 CTAs per SM of the two-head build (16 waves of three
    CTAs per SM), on sm_100a only, that build splits two ways at five CTAs per SM.  A call with ``z`` never
    splits."""
    if gated:
        return None
    if heads_per_group == 1:
        if 2 * row_groups > 7 * sm_count:
            return None
        return "stp_dim128_hpc1_s2w4", 1, 2
    min_ctas_per_sm = _DIM128_SPLIT_MB5_MIN_CTAS_PER_SM.get(arch)
    if (
        min_ctas_per_sm is None
        or _work_tiles(row_groups, heads_per_group, 2) < min_ctas_per_sm * sm_count
    ):
        return None
    return "stp_dim128_hpc2_s2w4_mb5", 2, 2


def _plan_stp_rows(
    *,
    state,
    x,
    dt,
    A,
    B,
    C,
    D,
    z,
    dt_bias,
    output,
    source,
    destination,
    nheads,
    ngroups,
    dim,
    dt_softplus,
    disable_state_update,
    pad_slot_id,
    arch,
    device_index,
    sms,
) -> Optional[RoutePlan]:
    """One single-token launch of the paired row-major program family on dense rows at runtime batch strides.

    Every program head is one 128x128 BF16 state tile: with ``dim`` 64 two adjacent 64-row heads fold into one
    program head and the state is viewed as ``(slots, nheads / 2, 128, 128)``; with ``dim`` 128 one head is one
    program head and the tensor map covers the state as passed.  ``dt``, ``A``, ``D`` and ``dt_bias`` carry one
    value per head at a unit head stride; slot tables are contiguous and cover the batch.  A launch the P-split
    rule takes (:func:`hd64_split_program`, :func:`dim128_split_program`) runs the split build under the same
    bindings, ``p_split`` CTAs per work tile; an unsplit headdim-64 launch runs :func:`hd64_unsplit_program`'s
    build.  A call with ``z`` inside the persistent build's domain (:func:`dim128_balanced_units`) runs that
    build: the same bindings with ``units`` (the batch's heads) for the head tiles, at most three CTAs per SM.
    Every program runs its build of the device's architecture (:func:`lean_program`).  One ``shape`` /
    ``stride()`` read per tensor.
    """
    x_shape = x.shape
    batch_size = x_shape[0]
    paired = dim == 64
    x_batch = _rows_batch_stride(x, batch_size, nheads, dim)
    b_batch = _rows_batch_stride(B, batch_size, ngroups, 128)
    c_batch = _rows_batch_stride(C, batch_size, ngroups, 128)
    out_batch = _rows_batch_stride(output, batch_size, nheads, dim)
    if (
        x_batch is None
        or b_batch is None
        or c_batch is None
        or out_batch is None
        or output.dtype != torch.bfloat16
        or dt.shape != x_shape
        or A.shape != (nheads, dim, 128)
        or D.shape != (nheads, dim)
        or dt_bias.shape != (nheads, dim)
        or source.shape[0] < batch_size
        or destination.shape[0] < batch_size
        or source.dtype != destination.dtype
        or not (source.is_contiguous() and destination.is_contiguous())
        or (paired and not state.is_contiguous())
        or (B.data_ptr() | C.data_ptr()) & 15
        or (b_batch | c_batch) & 7
    ):
        return None
    dt_stride, a_stride, d_stride, bias_stride = (
        dt.stride(),
        A.stride(),
        D.stride(),
        dt_bias.stride(),
    )
    # One coefficient per head at a unit head stride (a single head never steps it).
    head_strides = (dt_stride[1], a_stride[0], d_stride[0], bias_stride[0])
    if (
        dt_stride[2] != 0
        or a_stride[1:] != (0, 0)
        or d_stride[1] != 0
        or bias_stride[1] != 0
        or (nheads > 1 and head_strides != (1, 1, 1, 1))
    ):
        return None
    if z is not None and (
        z.dtype != torch.bfloat16
        or _rows_batch_stride(z, batch_size, nheads, dim) != x_batch
    ):
        return None
    if dt.dtype == torch.bfloat16 and source.dtype == torch.int32:
        raw_abi = True
    elif dt.dtype == torch.float32 and source.dtype == torch.int64:
        raw_abi = False
    else:
        return None
    if paired:
        program_heads = nheads >> 1
        heads_per_group = program_heads // ngroups
        split = hd64_split_program(
            batch_size * ngroups, heads_per_group, sms, arch, raw_abi
        )
        if split is None:
            program, heads_per_cta = hd64_unsplit_program(
                batch_size * ngroups, heads_per_group, sms, arch, raw_abi
            )
        state_tma = state.view(state.shape[0], program_heads, 128, 128)
    else:
        if raw_abi:
            return None
        program_heads = nheads
        heads_per_group = nheads // ngroups
        split = dim128_split_program(
            batch_size * ngroups, heads_per_group, sms, arch, z is not None
        )
        if split is None:
            program, heads_per_cta = dim128_program(heads_per_group, z is not None)
        state_tma = state
    p_split = 1
    if split is not None:
        program, heads_per_cta, p_split = split
    units = (
        dim128_balanced_units(batch_size, nheads, sms, arch)
        if not paired and z is not None
        else 0
    )
    head_tiles = (heads_per_group + heads_per_cta - 1) // heads_per_cta
    bindings = dict(
        state_tma=state_tma,
        x=x,
        dt_addr=dt.data_ptr(),
        a_addr=A.data_ptr(),
        B=B,
        C=C,
        d_addr=D.data_ptr(),
        z=x if z is None else z,
        dt_bias_addr=dt_bias.data_ptr(),
        output=output,
        state_batch_indices_addr=source.data_ptr(),
        dst_state_batch_indices_addr=destination.data_ptr(),
        nheads=program_heads,
        ngroups=ngroups,
        head_tiles=head_tiles,
        x_batch_stride=x_batch,
        b_batch_stride=b_batch,
        c_batch_stride=c_batch,
        out_batch_stride=out_batch,
        dt_batch_stride=dt_stride[0],
        dt_softplus=int(dt_softplus),
        has_z=int(z is not None),
        disable_state_update=int(disable_state_update),
        pad_slot_id=pad_slot_id,
    )
    if units:
        # The persistent build: CTA ``c`` of the grid streams heads ``c``, ``c + grid``, ... of the batch.
        del bindings["head_tiles"]
        bindings["units"] = units
        return RoutePlan(
            lean_program("stp_dim128_z_balanced", arch),
            (),
            (min(units, _DIM128_BALANCED_CTAS_PER_SM * sms), 1, 1),
            bindings,
            arch,
            device_index,
        )
    # ``p_split`` CTAs per work tile of ``heads_per_cta`` program heads of one (batch, group).
    ctas = batch_size * ngroups * head_tiles * p_split
    return RoutePlan(
        lean_program(program, arch, ctas, sms),
        (),
        (ctas, 1, 1),
        bindings,
        arch,
        device_index,
    )


HD64_ROWS_PER_CTA_ITERATION = 8
# (allowed prefetch depths, wanted below two batch-heads per SM, wanted above), mirrors of
# ``seed_stp_generic.{BF16,FP32}_PREFETCH_ROWS`` and the per-band targets
HD64_BF16_PREFETCH = ((1, 2), 1, 1)
HD64_FP32_PREFETCH = ((1, 4), 1, 4)


def hd64_rows_plan(
    batch_size: int, nheads: int, state_dtype: torch.dtype, sm_count: int
) -> Optional[tuple[int, int]]:
    """``(dim_tiles, prefetch_rows)`` of the 64x128 row-owner tile for one launch, or ``None`` when no delivered
    tile serves it.

    Both tiles own eight rows per CTA iteration (BF16: two warps of four 8-lane row owners; FP32: four warps
    of two 16-lane row owners).  BF16: under-saturated grids (below two batch-heads per SM) split the 64 rows
    into eight tiles, mid-range grids (below eight) into four; the saturated grid belongs to the paired batch
    programs (odd head pairings at that work fall back to FlashInfer).  FP32: eight tiles below one
    batch-head per SM, four below two, two below four, then one tile per head.  ``prefetch_rows`` (state rows
    per lane in flight per trip, a compile-time define of the instantiation) is the largest allowed depth at most the
    band's target that divides the tile's iterations (``seed_stp_generic.tile_plan``).
    """
    total_batch_heads = batch_size * nheads
    if state_dtype == torch.float32:
        if total_batch_heads < sm_count:
            dim_tiles = 8
        elif total_batch_heads < 2 * sm_count:
            dim_tiles = 4
        elif total_batch_heads < 4 * sm_count:
            dim_tiles = 2
        else:
            dim_tiles = 1
        allowed, low_wanted, wanted = HD64_FP32_PREFETCH
    elif total_batch_heads < 2 * sm_count:
        dim_tiles = 8
        allowed, low_wanted, wanted = HD64_BF16_PREFETCH
    elif total_batch_heads < 8 * sm_count:
        dim_tiles = 4
        allowed, low_wanted, wanted = HD64_BF16_PREFETCH
    else:
        return None
    target = low_wanted if total_batch_heads < 2 * sm_count else wanted
    iterations = (64 // dim_tiles) // HD64_ROWS_PER_CTA_ITERATION
    prefetch_rows = max(
        depth for depth in allowed if depth <= target and iterations % depth == 0
    )
    return dim_tiles, prefetch_rows


# The FP32 one-tile batch-heads, inclusive, at which the four-tile build reading the state 2176 CTAs ahead stores the
# state with an L2 evict-first policy (``_sl2first``), per architecture.
_HD64_ROWS_SL2FIRST_BATCH_HEADS = {"sm_103a": (28672, 40960)}


def hd64_rows_program(
    state_dtype: torch.dtype,
    raw_abi: bool,
    dim_tiles: int,
    total_batch_heads: int,
    arch: str,
    nheads: int,
) -> Optional[tuple[str, int]]:
    """``(program, CTAs per head)`` of the rendered row-owner build serving one band of :func:`hd64_rows_plan`, or
    ``None`` when the band keeps the carried FP32 row-owner program under the plan's defines.

    A band is the state dtype, FlashInfer's tile count and, at one tile per head, the batch-heads; ``raw_abi``
    (BF16 ``dt`` / ``D`` / ``dt_bias``, int32 slot tables) selects the ``_bf16_i32`` build.  BF16 state at eight
    tiles runs its band's build in either ABI.  BF16 state at four tiles runs one of two builds per ABI: in the
    FP32/int64 ABI the build with x loaded in all lanes and the one-division group index (``xa_od``) on sm_100a and
    the one-division build (``od``) on sm_103a; in the BF16/int32 ABI the one-division build, but on sm_103a below
    64 heads the build with x loaded in all lanes (``xa``).  FP32 state at eight tiles runs its band's build in
    either ABI; FP32 at four and two tiles runs its build in the raw ABI only; FP32 at one tile runs one build up to
    4096 batch-heads in the canonical ABI, up to 2048 in the raw ABI and above 2048 up to 4096 the raw build that
    reads the state tile 2304 CTAs ahead into L2, and from 8192 in either ABI the four-tile build that reads it
    2176 CTAs ahead, on sm_103a from 28672 to 40960 batch-heads its build that stores the state with an L2 evict-first
    policy.  The canonical calls at four and two tiles and the calls between 4096 and 8192 batch-heads keep the
    carried program.
    """
    suffix = "_bf16_i32" if raw_abi else ""
    if state_dtype == torch.bfloat16:
        if dim_tiles == 8:
            return f"stp_hd64_rows_bf16_dt8{suffix}", 4
        if raw_abi:
            build = "xa" if arch == "sm_103a" and nheads < 64 else "od"
        else:
            build = "od" if arch == "sm_103a" else "xa_od"
        return f"stp_hd64_rows_bf16_dt4_{build}{suffix}", 4
    if dim_tiles == 8:
        return f"stp_hd64_rows_fp32_dt8{suffix}", 4
    if dim_tiles in (4, 2):
        return (f"stp_hd64_rows_fp32_dt{dim_tiles}_bf16_i32", 4) if raw_abi else None
    if total_batch_heads >= 8192:
        band = _HD64_ROWS_SL2FIRST_BATCH_HEADS.get(arch)
        if band is not None and band[0] <= total_batch_heads <= band[1]:
            return f"stp_hd64_rows_fp32_dt1_large_sl2first{suffix}", 4
        return f"stp_hd64_rows_fp32_dt1_large{suffix}", 4
    if not raw_abi:
        return ("stp_hd64_rows_fp32_dt1", 4) if total_batch_heads <= 4096 else None
    if total_batch_heads <= 2048:
        return "stp_hd64_rows_fp32_dt1_bf16_i32", 2
    if total_batch_heads <= 4096:
        return "stp_hd64_rows_fp32_dt1_l2_prefetch_bf16_i32", 4
    return None


def _plan_stp_hd64(
    *,
    state,
    x,
    dt,
    A,
    B,
    C,
    D,
    z,
    dt_bias,
    output,
    source,
    destination,
    nheads,
    ngroups,
    dt_softplus,
    disable_state_update,
    pad_slot_id,
    arch,
    device_index,
) -> Optional[RoutePlan]:
    """Headdim-64 / dstate-128 single-token decode (Nemotron-H, granite-4.0-h) on both ABIs and state dtypes.

    BF16 state with an even head count and even heads per group pairs two heads into one 128x128 program head
    of the paired row-major programs at saturating work (at least two program heads per SM) and wherever the
    P-split rule takes the launch below that (one to two program heads per SM, :func:`hd64_split_program`);
    every other case, and a split launch whose layout the paired programs cannot read, runs the direct
    row-owner tile.  Dense x/B/C/out rows may carry padded batch strides (the SGLang fused-projection views).
    """
    sms = _sm_count(device_index)
    if (
        state.dtype == torch.bfloat16
        and nheads % 2 == 0
        and (nheads // ngroups) % 2 == 0
    ):
        saturating = x.shape[0] * (nheads >> 1) >= 2 * sms
        split = hd64_split_program(
            x.shape[0] * ngroups, (nheads >> 1) // ngroups, sms, arch, False
        )
        if saturating or split is not None:
            plan = _plan_stp_rows(
                state=state,
                x=x,
                dt=dt,
                A=A,
                B=B,
                C=C,
                D=D,
                z=z,
                dt_bias=dt_bias,
                output=output,
                source=source,
                destination=destination,
                nheads=nheads,
                ngroups=ngroups,
                dim=64,
                dt_softplus=dt_softplus,
                disable_state_update=disable_state_update,
                pad_slot_id=pad_slot_id,
                arch=arch,
                device_index=device_index,
                sms=sms,
            )
            if plan is not None or saturating:
                return plan
    x_shape, x_stride = x.shape, x.stride()
    batch_size = x_shape[0]
    rows_plan = hd64_rows_plan(batch_size, nheads, state.dtype, sms)
    if rows_plan is None:
        return None
    dim_tiles, prefetch_rows = rows_plan
    b_shape, b_stride, c_shape, c_stride = B.shape, B.stride(), C.shape, C.stride()
    out_shape, out_stride = output.shape, output.stride()
    dt_stride, a_stride, d_stride, bias_stride = (
        dt.stride(),
        A.stride(),
        D.stride(),
        dt_bias.stride(),
    )
    x_batch = _dense_rows(x_shape, x_stride, batch_size, nheads, 64)
    b_batch = _dense_rows(b_shape, b_stride, batch_size, ngroups, 128)
    c_batch = _dense_rows(c_shape, c_stride, batch_size, ngroups, 128)
    out_batch = _dense_rows(out_shape, out_stride, batch_size, nheads, 64)
    if (
        x_batch is None
        or b_batch is None
        or c_batch is None
        or out_batch is None
        or output.dtype != torch.bfloat16
        or dt.shape != x_shape
        or dt_stride[2] != 0
        or A.shape != (nheads, 64, 128)
        or a_stride[1:] != (0, 0)
        or D.shape != (nheads, 64)
        or d_stride[1] != 0
        or dt_bias.shape != (nheads, 64)
        or bias_stride[1] != 0
        or source.shape != (batch_size,)
        or destination.shape != (batch_size,)
        or source.dtype != destination.dtype
        or not (
            state.is_contiguous()
            and source.is_contiguous()
            and destination.is_contiguous()
        )
    ):
        return None
    if z is not None and (
        z.dtype != torch.bfloat16
        or _dense_rows(z.shape, z.stride(), batch_size, nheads, 64) != x_batch
    ):
        return None
    bindings = dict(
        state=state,
        x=x,
        dt_addr=dt.data_ptr(),
        a_addr=A.data_ptr(),
        B=B,
        C=C,
        d_addr=D.data_ptr(),
        z=x if z is None else z,
        dt_bias_addr=dt_bias.data_ptr(),
        output=output,
        state_batch_indices_addr=source.data_ptr(),
        dst_state_batch_indices_addr=destination.data_ptr(),
        nheads=nheads,
        ngroups=ngroups,
        state_stride_slot=state.stride(0),
        x_batch_stride=x_batch,
        b_batch_stride=b_batch,
        c_batch_stride=c_batch,
        out_batch_stride=out_batch,
        dt_batch_stride=dt_stride[0],
        dt_head_stride=dt_stride[1],
        a_head_stride=a_stride[0],
        d_head_stride=d_stride[0],
        dt_bias_head_stride=bias_stride[0],
        dt_softplus=int(dt_softplus),
        has_z=int(z is not None),
        disable_state_update=int(disable_state_update),
        pad_slot_id=pad_slot_id,
    )
    rendered = hd64_rows_program(
        state.dtype,
        dt.dtype == torch.bfloat16,
        dim_tiles,
        batch_size * nheads,
        arch,
        nheads,
    )
    if rendered is None:
        return RoutePlan(
            "stp_hd64_rows_fp32",
            rows_defines(dt.dtype, source.dtype, prefetch_rows, dim_tiles),
            (dim_tiles, nheads, batch_size),
            bindings,
            arch,
            device_index,
        )
    program, tiles = rendered
    bindings["state_batch_indices"] = source
    bindings["dst_state_batch_indices"] = destination
    # One CTA per (tile of the head's 64 rows, head, batch row).
    return RoutePlan(
        program, (), (tiles, nheads, batch_size), bindings, arch, device_index
    )


def plan_route(
    *,
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    output: torch.Tensor,
    state_batch_indices: Optional[torch.Tensor],
    dst_state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    disable_state_update: bool,
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    state_scale: Optional[torch.Tensor],
    intermediate_state_scales: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    cache_steps: int,
    cu_seqlens: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    algorithm: str,
    dt_softplus: bool,
) -> Optional[RoutePlan]:
    """Resolve the Cake route of one call from tensor metadata, or ``None`` when no program serves it."""
    if (
        dt_bias is None
        or state_batch_indices is None
        or state_scale is not None
        or intermediate_state_scales is not None
        or rand_seed is not None
        or cu_seqlens is not None
        or num_accepted_tokens is not None
        or state.ndim != 4
        or state.dtype not in (torch.bfloat16, torch.float32)
        or x.dtype != torch.bfloat16
        or B.dtype != torch.bfloat16
        or C.dtype != torch.bfloat16
        or A.dtype != torch.float32
        or state_batch_indices.ndim != 1
    ):
        return None
    # One dtype read per coefficient and index tensor; an absent index table takes the dtype of the required one.
    dt_dtype, d_dtype, bias_dtype = dt.dtype, D.dtype, dt_bias.dtype
    index_dtype = state_batch_indices.dtype
    dst_dtype = (
        index_dtype
        if dst_state_batch_indices is None
        else dst_state_batch_indices.dtype
    )
    intermediate_dtype = (
        index_dtype
        if intermediate_state_indices is None
        else intermediate_state_indices.dtype
    )
    canonical = (
        dt_dtype == torch.float32
        and d_dtype == torch.float32
        and bias_dtype == torch.float32
        and index_dtype == dst_dtype == intermediate_dtype == torch.int64
    )
    raw_coefficients = (
        dt_dtype == torch.bfloat16
        and d_dtype == torch.bfloat16
        and bias_dtype == torch.bfloat16
        and index_dtype == dst_dtype == intermediate_dtype == torch.int32
    )
    if not (canonical or raw_coefficients):
        return None
    device_index = state.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    arch = _device_arch(device_index)
    if arch is None:
        return None
    # One shape read per tensor (``torch.Size`` holds plain ints); the routes below compare against these
    # tuples. Layouts the shapes admit but the programs cannot read (non-contiguous buffers) are rejected by
    # each binding's argument validation before any launch, and ``try_cake_selective_state_update`` falls
    # back on that.
    x_shape, b_shape, state_shape = x.shape, B.shape, state.shape
    batch_size = x_shape[0]
    nheads, dim, dstate = state_shape[1], state_shape[2], state_shape[3]
    ngroups = b_shape[-2]
    if ngroups <= 0 or nheads % ngroups:
        return None

    if (
        len(x_shape) == 3
        and cache_steps == 0
        and dim == 128
        and dstate == 128
        and canonical
    ):
        destination = (
            state_batch_indices
            if dst_state_batch_indices is None
            else dst_state_batch_indices
        )
        if destination.ndim != 1:
            return None
        if state.dtype == torch.bfloat16:
            sms = _sm_count(device_index)
            ring = dim128_stp_ring_view(
                state=state,
                x=x,
                dt=dt,
                A=A,
                B=B,
                C=C,
                D=D,
                z=z,
                dt_bias=dt_bias,
                output=output,
                source=state_batch_indices,
                dst_state_batch_indices=dst_state_batch_indices,
                intermediate_states_buffer=intermediate_states_buffer,
                nheads=nheads,
                ngroups=ngroups,
                dt_softplus=dt_softplus,
                disable_state_update=disable_state_update,
                pad_slot_id=pad_slot_id,
                arch=arch,
                device_index=device_index,
                sms=sms,
            )
            if ring is not None:
                return ring
            return _plan_stp_rows(
                state=state,
                x=x,
                dt=dt,
                A=A,
                B=B,
                C=C,
                D=D,
                z=z,
                dt_bias=dt_bias,
                output=output,
                source=state_batch_indices,
                destination=destination,
                nheads=nheads,
                ngroups=ngroups,
                dim=128,
                dt_softplus=dt_softplus,
                disable_state_update=disable_state_update,
                pad_slot_id=pad_slot_id,
                arch=arch,
                device_index=device_index,
                sms=sms,
            )
        if (
            z is None
            and not dt_softplus
            and not disable_state_update
            and destination.data_ptr() == state_batch_indices.data_ptr()
            and batch_size * nheads >= 8 * _sm_count(device_index)
        ):
            # The program reads one coefficient per head (broadcast dt / A / D / dt_bias, passed by address
            # and head stride) at the shapes below; any other shape or a dense coefficient tensor plans no
            # launch. Shapes are compared before the strides are indexed, and each stride tuple is read once
            # and reused by the bindings.
            dt_stride, a_stride, d_stride, bias_stride = (
                dt.stride(),
                A.stride(),
                D.stride(),
                dt_bias.stride(),
            )
            if (
                x_shape != (batch_size, nheads, dim)
                or b_shape != (batch_size, ngroups, dstate)
                or C.shape != b_shape
                or output.shape != x_shape
                or output.dtype != torch.bfloat16
                or state_batch_indices.shape != (batch_size,)
                or dt.shape != x_shape
                or A.shape != (nheads, dim, dstate)
                or D.shape != (nheads, dim)
                or dt_bias.shape != (nheads, dim)
                or dt_stride[2] != 0
                or a_stride[1] != 0
                or a_stride[2] != 0
                or d_stride[1] != 0
                or bias_stride[1] != 0
                or not (
                    x.is_contiguous()
                    and B.is_contiguous()
                    and C.is_contiguous()
                    and output.is_contiguous()
                )
            ):
                # The program reads x, B, C and the output at their batch strides, but this route serves
                # contiguous rows only (padded rows run on FlashInfer).
                return None
            bindings = dict(
                state=state,
                x=x,
                dt_addr=dt.data_ptr(),
                a_addr=A.data_ptr(),
                B=B,
                C=C,
                d_addr=D.data_ptr(),
                z=x,
                dt_bias_addr=dt_bias.data_ptr(),
                output=output,
                state_batch_indices=state_batch_indices,
                dst_state_batch_indices=destination,
                nheads=nheads,
                ngroups=ngroups,
                state_stride_slot=state.stride(0),
                x_batch_stride=x.stride(0),
                b_batch_stride=B.stride(0),
                c_batch_stride=C.stride(0),
                out_batch_stride=output.stride(0),
                dt_batch_stride=dt_stride[0],
                dt_head_stride=dt_stride[1],
                a_head_stride=a_stride[0],
                d_head_stride=d_stride[0],
                dt_bias_head_stride=bias_stride[0],
                dt_softplus=0,
                has_z=0,
                disable_state_update=0,
                pad_slot_id=pad_slot_id,
            )
            # Eight CTAs per (head, batch row), each owning 16 of the head's 128 rows.
            return RoutePlan(
                dim128_fp32_program(batch_size * nheads),
                (),
                (8, nheads, batch_size),
                bindings,
                arch,
                device_index,
            )
        return None

    if (
        len(x_shape) == 3
        and cache_steps == 0
        and dim == 64
        and dstate == 128
        and intermediate_states_buffer is None
    ):
        destination = (
            state_batch_indices
            if dst_state_batch_indices is None
            else dst_state_batch_indices
        )
        if destination.ndim != 1:
            return None
        return _plan_stp_hd64(
            state=state,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            output=output,
            source=state_batch_indices,
            destination=destination,
            nheads=nheads,
            ngroups=ngroups,
            dt_softplus=dt_softplus,
            disable_state_update=disable_state_update,
            pad_slot_id=pad_slot_id,
            arch=arch,
            device_index=device_index,
        )

    if len(x_shape) != 4:
        return None
    c_shape = C.shape
    token_steps = x_shape[1]

    if (
        canonical
        and state.dtype == torch.bfloat16
        and (dim, dstate) == (128, 128)
        and token_steps in (1, 2)
        and z is None
        and dst_state_batch_indices is None
        and intermediate_states_buffer is None
        and not disable_state_update
    ):
        # The shapes the program indexes; the compact coefficient views carry the broadcast check.
        if (
            x_shape != (batch_size, token_steps, nheads, dim)
            or b_shape != (batch_size, token_steps, ngroups, dstate)
            or c_shape != b_shape
            or output.shape != x_shape
            or output.dtype != torch.bfloat16
            or state_batch_indices.shape != (batch_size,)
        ):
            return None
        compact = _coefficients(dt, A, D, dt_bias, x_shape, nheads, dim, dstate)
        if compact is None:
            return None
        dt_compact, A_compact, D_compact, bias_compact = compact
        program, head_tiles = mtp_short_program(
            batch_size,
            nheads,
            ngroups,
            token_steps,
            _sm_count(device_index),
            arch,
        )
        bindings = dict(
            x=x,
            dt=dt_compact,
            A=A_compact,
            B=B,
            C=C,
            D=D_compact,
            dt_bias=bias_compact,
            output=output,
            state_batch_indices=state_batch_indices,
            batch_size=batch_size,
            nheads=nheads,
            dim=dim,
            dstate=dstate,
            ngroups=ngroups,
            token_steps=token_steps,
            dt_softplus=int(dt_softplus),
            pad_slot_id=pad_slot_id,
        )
        if head_tiles:
            # The TMA ring: a CTA per (batch row, group, tile of heads), the state a tensor map of the slot table.
            bindings.update(state_tma=state, head_tiles=head_tiles)
            grid = (batch_size * ngroups * head_tiles, 1, 1)
        else:
            # A CTA per (batch row, head); 64 heads of 8 groups fold those counts into each CTA's (batch, head, group).
            bindings.update(state=state, state_stride_slot=state.stride(0))
            grid = (batch_size * nheads, 1, 1)
        return RoutePlan(program, (), grid, bindings, arch, device_index)

    if (
        state.dtype == torch.bfloat16
        and (dim, dstate, token_steps) == (64, 128, 6)
        and z is None
        and dst_state_batch_indices is None
        and dt_softplus
        and disable_state_update
        and intermediate_states_buffer is not None
        and intermediate_state_indices is not None
    ):
        total_tiles = batch_size * nheads
        sms = _sm_count(device_index)
        if batch_size < 32 and (sms * 10) // total_tiles >= 4:
            return _plan_mtp_cache(
                state=state,
                x=x,
                dt=dt,
                A=A,
                B=B,
                C=C,
                D=D,
                dt_bias=dt_bias,
                output=output,
                source=state_batch_indices,
                intermediate=intermediate_states_buffer,
                intermediate_indices=intermediate_state_indices,
                nheads=nheads,
                ngroups=ngroups,
                pad_slot_id=pad_slot_id,
                arch=arch,
                device_index=device_index,
            )
        if canonical and batch_size >= 32 and algorithm == "horizontal":
            # The shapes the program indexes (state, x, B and C arrive through tensor maps encoded from
            # their strides); the compact coefficient views carry the broadcast check.
            if (
                x_shape != (batch_size, token_steps, nheads, dim)
                or b_shape != (batch_size, token_steps, ngroups, dstate)
                or c_shape != b_shape
                or output.shape != x_shape
                or output.dtype != torch.bfloat16
                or state_batch_indices.shape != (batch_size,)
                or intermediate_states_buffer.shape[1:]
                != (token_steps, nheads, dim, dstate)
                or intermediate_states_buffer.dtype != torch.bfloat16
                or intermediate_state_indices.shape != (batch_size,)
            ):
                return None
            compact = _coefficients(dt, A, D, dt_bias, x_shape, nheads, dim, dstate)
            if compact is None:
                return None
            dt_compact, A_compact, D_compact, bias_compact = compact
            work_groups = (total_tiles + _MTP_HEADS_PER_CTA - 1) // _MTP_HEADS_PER_CTA
            # One CTA per work group. Up to two resident waves the early-index build reads each head's
            # cache slot ahead of the state and stores the outputs after the last token; above them the
            # plain build runs.
            program = (
                "mtp_horizontal_early"
                if work_groups
                <= _MTP_DIRECT_MAX_RESIDENT_WAVES * _MTP_RESIDENT_CTAS_PER_SM * sms
                else "mtp_horizontal_direct"
            )
            bindings = dict(
                state_tma=state,
                x_tma=x,
                b_tma=B,
                c_tma=C,
                dt=dt_compact,
                A=A_compact,
                D=D_compact,
                dt_bias=bias_compact,
                output=output,
                state_batch_indices=state_batch_indices,
                intermediate_state=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices,
                nheads=nheads,
                ngroups=ngroups,
                total_tiles=total_tiles,
                intermediate_stride_slot=intermediate_states_buffer.stride(0),
                dt_softplus=int(dt_softplus),
                cache_intermediate=1,
            )
            return RoutePlan(
                program, (), (work_groups, 1, 1), bindings, arch, device_index
            )
        return None

    if (
        canonical
        and state.dtype == torch.float32
        and (nheads, dim, dstate, ngroups) == (16, 64, 128, 1)
        and 1 <= token_steps <= _DYNAMIC_MAX_TOKEN_STEPS
        and algorithm == "simple"
        and dt_softplus
        and z is None
        and dst_state_batch_indices is not None
        and dst_state_batch_indices.shape == (batch_size, token_steps)
        and intermediate_states_buffer is None
        and not disable_state_update
    ):
        # The fixed-index program indexes every buffer at the shapes below; the compact coefficient
        # views carry the broadcast check.
        if (
            x_shape != (batch_size, token_steps, 16, 64)
            or b_shape != (batch_size, token_steps, 1, 128)
            or c_shape != b_shape
            or output.shape != x_shape
            or output.dtype != torch.bfloat16
            or state_batch_indices.shape != (batch_size,)
        ):
            return None
        compact = _coefficients(dt, A, D, dt_bias, x_shape, nheads, dim, dstate)
        if compact is None:
            return None
        dt_compact, A_compact, D_compact, bias_compact = compact
        bindings = dict(
            state=state,
            x=x,
            dt=dt_compact,
            A=A_compact,
            B=B,
            C=C,
            D=D_compact,
            dt_bias=bias_compact,
            output=output,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            batch_size=batch_size,
            token_steps=token_steps,
            state_stride_slot=state.stride(0),
            pad_slot_id=pad_slot_id,
        )
        program = dynamic_program(batch_size, token_steps, arch)
        return RoutePlan(
            program,
            (),
            (batch_size, nheads, 4 if program == "dynamic_lanes8" else 8),
            bindings,
            arch,
            device_index,
        )
    return None


def try_cake_selective_state_update(**kwargs: Any) -> bool:
    """Run the call through a Cake program when one serves it; ``False`` leaves it to FlashInfer."""
    plan = plan_route(**kwargs)
    if plan is None:
        return False
    with torch.cuda.device(plan.device_index):
        try:
            program = _program(plan.program, plan.arch, plan.defines)
        except MissingJITCacheError:
            # JIT builds are disabled and the cache holds no build of this program: the call runs on FlashInfer.
            return False
        try:
            program.launch(plan.grid, plan.bindings)
        except ValueError:
            # The binding validates every argument (device, dtype, contiguity, tensor-map strides) before
            # it launches, so nothing has run when it rejects a layout: the call goes to FlashInfer.
            return False
    return True


__all__ = [
    "MODULES",
    "PROGRAMS",
    "RoutePlan",
    "define_digest",
    "dim128_fp32_program",
    "dim128_program",
    "dim128_split_program",
    "dim128_stp_ring",
    "dim128_stp_ring_program",
    "dynamic_program",
    "hd64_paired_program",
    "hd64_rows_plan",
    "hd64_split_program",
    "hd64_unsplit_program",
    "jit_spec",
    "lean_program",
    "mtp_short_program",
    "plan_route",
    "rows_defines",
    "try_cake_selective_state_update",
]
