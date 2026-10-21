# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JIT loader for the generated Cake quantized-base + runtime BF16 LoRA GEMM programs.

One generated program per schedule template, one kernel source and one host
binding each under ``csrc/cake_quant_lora_gemm``, compiled once per exact target:

* ``nvfp4_blackwell_pair`` -- NVFP4 base (E2M1 codes, UE4M3 128x4 block scales) on the 256 x 128
  cta_group::2 pair tile, SM100 / SM103.
* ``nvfp4_blackwell_n256`` -- the same base on the 256 x 256 pair tile, SM100 / SM103.
* ``nvfp4_blackwell_n192`` / ``nvfp4_blackwell_n192h`` -- the same base on the 256 x 192 pair tile
  (SM100 / SM103), and the same schedule with TMA L2 evict_last operand hints and l2_256b
  descriptor promotion (SM100); the host's per-architecture dispatch rule picks one of the
  NVFP4 programs per problem (see :mod:`flashinfer.gemm.cake_quant_lora_gemm`).
* ``nvfp4_blackwell_svdq`` -- the prepared-adapter program on the 256 x 256 pair tile
  (SM100 / SM103): ``mm_nvfp4_lora(..., lora_adapter=prepare_nvfp4_lora_adapter(...))``
  runs it for every prepared call (K >= 2048) at its own raster rule; the raw path never
  reaches it (``PROGRAMS[...]["adapter_states"]`` names the API forms a record serves).
* ``fp8_blackwell_n256`` / ``fp8_blackwell_n128`` -- FP8 E4M3 base with per-tensor
  scales, SM100 / SM103; the pair-tile width is resolved from the shape and the
  device SM count (see :mod:`flashinfer.gemm.cake_quant_lora_gemm`).
* ``fp8_blackwell_n256_clc_blocked`` -- the 256-wide FP8 program with the cluster-launch-control
  dynamic tile scheduler and the 2-D blocked tile raster, SM100 / SM103; the host's per-problem
  rule picks it on long large-K runs and the static program everywhere else.
* ``fp8_blackwell_n256_m256_clc_blocked`` -- the same scheduler and raster on the 512-row pair
  tile (256 token rows per CTA), SM103 only; the host's per-problem rule picks it on the longest
  large-K runs (at least 20 waves of 512 x 256 tiles, K >= 32768) of the SM103 device.
* ``fp8_sm90`` / ``fp8_sm90_hint`` -- FP8 E4M3 base, SM90 (persistent, cluster pairs); the
  host's per-problem rule picks the second program (TMA L2 policies, raster group 1) on
  long runs of large-K panels and the first one everywhere else (see
  :mod:`flashinfer.gemm.cake_quant_lora_gemm`).

The Blackwell kernels are ``template <int RANK>`` instantiated for ranks 32, 64 and
128; every binding dispatches on the runtime ``rank`` argument.  Every tensor map
travels by value in the kernel parameters, so a call is self-contained and
CUDA-graph capturable.  ``lora_scale`` must be a normal float or ±0; denormal
scales are unsupported.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

import torch

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm90a_nvcc_flags,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)
from ..utils import get_compute_capability

CakeQuantLoraTarget = Literal["sm90a", "sm100a", "sm103a"]
_TARGET_FLAGS = {
    "sm90a": sm90a_nvcc_flags,
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
TARGET_ARCH: dict[str, str] = {
    "sm90a": "sm_90a",
    "sm100a": "sm_100a",
    "sm103a": "sm_103a",
}
TARGET_OF_ARCH: dict[str, str] = {arch: target for target, arch in TARGET_ARCH.items()}
_CAPABILITY_TARGET: dict[tuple[int, int], CakeQuantLoraTarget] = {
    (9, 0): "sm90a",
    (10, 0): "sm100a",
    (10, 3): "sm103a",
}

# Values shared by several PROGRAMS records: one module-level object per distinct
# value, referenced by every record that carries it (readers take them by value).
# A record that needs its own value gets its own literal; a shared constant is
# never edited in place.

# Compile flags of the SM100 / SM103 programs and of the SM90 programs.
_COMPILE_FLAGS_BLACKWELL: list[str] = ["--use_fast_math"]
_COMPILE_FLAGS_SM90: list[str] = []

# Kernel argument order: the FP8 Blackwell programs; the SM90 programs; the
# 192- and 256-wide raw NVFP4 programs; the NVFP4 pair-tile program; the
# prepared-adapter NVFP4 program.
_ARG_PLAN_FP8_BLACKWELL: list[list[str]] = [
    ["tma_buffer", "A"],
    ["tma_buffer", "D"],
    ["buffer", "a_scale"],
    ["buffer", "b_scale"],
    ["buffer", "lora_scale"],
    ["buffer", "bias"],
    ["tma_buffer", "out"],
    ["parameter", "M"],
    ["parameter", "N"],
    ["parameter", "K_tiles"],
    ["parameter", "tok_tiles"],
    ["parameter", "w_tiles"],
    ["parameter", "num_tiles"],
    ["parameter", "raster_group"],
    ["parameter", "has_bias"],
    ["parameter", "has_lora"],
    ["parameter", "rank"],
    ["tma_buffer", "B"],
    ["tma_buffer", "LORA_B"],
    ["grid", "grid_x"],
    ["grid", "grid_y"],
    ["grid", "grid_z"],
]
_ARG_PLAN_FP8_SM90: list[list[str]] = [
    ["parameter", "M"],
    ["parameter", "N"],
    ["parameter", "K"],
    ["parameter", "rank"],
    ["parameter", "group_m"],
    ["parameter", "has_adapter"],
    ["parameter", "has_bias"],
    ["tma_buffer", "A"],
    ["tma_buffer", "W"],
    ["tma_buffer", "D"],
    ["tma_buffer", "BL"],
    ["buffer", "a_scale"],
    ["buffer", "b_scale"],
    ["buffer", "lora_scale"],
    ["buffer", "bias"],
    ["buffer", "out"],
    ["grid", "grid_x"],
    ["grid", "grid_y"],
    ["grid", "grid_z"],
]
_ARG_PLAN_NVFP4_BLACKWELL_WIDE: list[list[str]] = [
    ["tma_buffer", "D"],
    ["buffer", "alpha"],
    ["buffer", "lora_scale"],
    ["buffer", "bias"],
    ["tma_buffer", "out"],
    ["parameter", "M"],
    ["parameter", "N"],
    ["parameter", "K_tiles"],
    ["parameter", "tok_tiles"],
    ["parameter", "w_tiles"],
    ["parameter", "num_tiles"],
    ["parameter", "raster_group"],
    ["parameter", "has_bias"],
    ["parameter", "has_lora"],
    ["parameter", "rank"],
    ["tma_buffer", "A"],
    ["tma_buffer", "B"],
    ["tma_buffer", "SFA"],
    ["tma_buffer", "SFB"],
    ["tma_buffer", "LORA_B"],
    ["grid", "grid_x"],
    ["grid", "grid_y"],
    ["grid", "grid_z"],
]
_ARG_PLAN_NVFP4_BLACKWELL_PAIR: list[list[str]] = [
    ["tma_buffer", "A"],
    ["tma_buffer", "SFA"],
    ["tma_buffer", "SFB"],
    ["tma_buffer", "D"],
    ["buffer", "alpha"],
    ["buffer", "lora_scale"],
    ["buffer", "bias"],
    ["tma_buffer", "out"],
    ["parameter", "M"],
    ["parameter", "N"],
    ["parameter", "K_tiles"],
    ["parameter", "tok_tiles"],
    ["parameter", "w_tiles"],
    ["parameter", "num_tiles"],
    ["parameter", "raster_group"],
    ["parameter", "has_bias"],
    ["parameter", "has_lora"],
    ["parameter", "rank"],
    ["tma_buffer", "B"],
    ["tma_buffer", "LORA_B"],
    ["grid", "grid_x"],
    ["grid", "grid_y"],
    ["grid", "grid_z"],
]
_ARG_PLAN_NVFP4_BLACKWELL_SVDQ: list[list[str]] = [
    ["tma_buffer", "D"],
    ["buffer", "alpha"],
    ["buffer", "lora_scale"],
    ["buffer", "bias"],
    ["tma_buffer", "out"],
    ["parameter", "M"],
    ["parameter", "N"],
    ["parameter", "K_tiles"],
    ["parameter", "tok_tiles"],
    ["parameter", "w_tiles"],
    ["parameter", "num_tiles"],
    ["parameter", "raster_group"],
    ["parameter", "has_bias"],
    ["parameter", "has_lora"],
    ["parameter", "rank"],
    ["tma_buffer", "A"],
    ["tma_buffer", "B"],
    ["tma_buffer", "SFA"],
    ["tma_buffer", "SFB"],
    ["tma_buffer", "LORA_B_PRE"],
    ["tma_buffer", "LORA_B_TAIL"],
    ["grid", "grid_x"],
    ["grid", "grid_y"],
    ["grid", "grid_z"],
]

# Per-family dispatch constants (identical on every record of a family).
_DISPATCH_FP8_BLACKWELL: dict[str, Any] = {
    "rule_id": "fp8_blackwell_l2_dispatch_v0",
    "arches": ["sm_100a", "sm_103a"],
    "form": "rescale",
    "gated_template": "fp8_blackwell_n256_clc_blocked",
    "gated_tile_n": 256,
    "default_sched": "static",
    "default_raster_mode": "grouped",
    "gated_sched": "clc",
    "gated_raster_mode": "blocked",
    "min_tiles_per_pair": 10,
    "min_k": 16384,
    "pair_m": 256,
    "block_pack": 256,
    "block_max": 255,
    "tile_m_arches": ["sm_103a"],
    "tile_m_default": 128,
    "tile_m_wide": 256,
    "tile_m_min_waves": 20,
    "tile_m_min_k": 32768,
    "tile_m_template": "fp8_blackwell_n256_m256_clc_blocked",
}
_DISPATCH_FP8_SM90: dict[str, Any] = {
    "rule_id": "sm90_fp8_lora_dispatch_v0",
    "default_template": "fp8_sm90",
    "gated_template": "fp8_sm90_hint",
    "default_group_m": 8,
    "gated_group_m": 2,
    "panel_budget_bytes": 50331648,
    "max_k": 32768,
    "waves_min": 192,
    "block_m": 128,
    "block_n": 192,
    "cluster_m": 2,
}
_DISPATCH_NVFP4_BLACKWELL: dict[str, Any] = {
    "sm_100a": {
        "arch": "sm_100a",
        "sku": "B200",
        "l2_total_bytes": 132644864,
        "l2_partitions": 2,
        "l2_resident_fraction": 0.67,
        "l2_budget_bytes": 44436029,
        "k_threshold": 8192,
        "K_THRESHOLD": 8192,
        "SMALL_M": 4096,
        "SMALL_N": 4096,
        "SMALL_K_MIN": 16384,
        "rule_id": "DISPATCH_RULE_v2 (probes 2cdf47e3 / d8b6a313, tree 1dbdd0847aba91d9cdd5974bd3ee608ed32005c0, rule sha256 e219fb3fe716561582885b3caddb3850a9224113c376e46fbfb247bab414c2c8)",
        "choice_rule_order": "ordered, first match wins; every condition in an entry's 'if' must hold (keys <M|N|K|rank>_<lt|le|ge|gt|eq>); the final {'else': program} entry is the default",
        "schema_version": 2,
        "kernels": {
            "pair": {
                "form": "acc2",
                "tile_m": 256,
                "tile_n": 128,
                "sfb_mcast": True,
                "raster_candidates": [0, 1, 8, 16],
                "mainline_order": "along_m",
                "min_k": 0,
            },
            "n256": {
                "form": "rescale",
                "tile_m": 256,
                "tile_n": 256,
                "sfb_mcast": True,
                "hints": False,
                "raster_candidates": [0, 8, 16],
                "mainline_order": "along_m",
                "min_k": 8192,
            },
            "n192": {
                "form": "rescale",
                "tile_m": 256,
                "tile_n": 192,
                "sfb_mcast": True,
                "hints": False,
                "raster_candidates": [0, 8, 16],
                "mainline_order": "along_m",
                "min_k": 0,
            },
            "n192h": {
                "form": "rescale",
                "tile_m": 256,
                "tile_n": 192,
                "sfb_mcast": True,
                "hints": True,
                "raster_candidates": [0, 8, 16],
                "mainline_order": "along_m",
                "min_k": 0,
            },
            "svdq_folded": {
                "form": "folded_last_ordered",
                "tile_m": 256,
                "tile_n": 256,
                "raster_candidates": [0, 8, 16],
                "mainline_order": "along_m",
                "min_k": 2048,
                "dispatched": False,
                "adapter_state": "prepared",
            },
        },
        "choice_rule": [
            {"if": {"K_le": 3072, "M_le": 4096}, "program": "n192h"},
            {"if": {"K_le": 3072}, "program": "n192h"},
            {"if": {"K_le": 4096, "M_le": 4096}, "program": "pair"},
            {"if": {"K_le": 4096}, "program": "n192"},
            {"if": {"K_le": 6144, "M_le": 4096}, "program": "n192h"},
            {"if": {"K_le": 6144}, "program": "n192"},
            {"if": {"K_le": 8192, "N_le": 8192}, "program": "n256"},
            {"if": {"K_le": 8192}, "program": "n256"},
            {
                "if": {"M_le": 4096, "N_le": 4096, "K_ge": 16384},
                "program": "pair",
            },
            {"else": "n256"},
        ],
        "prepared": {
            "kernel": "svdq_folded",
            "min_k": 2048,
            "raster_rule": "k_band_then_n256_bytes_model",
            "k_band_max_k": 4096,
            "k_band_min_m": 32768,
            "k_band_group": 1,
            "bytes_model_candidates": [0, 8, 16],
            "bytes_model_tile_n": 256,
            "bytes_model_pair_m": 256,
            "bytes_model_budget_bytes": 44436029,
            "bytes_model_tie": "fewest modeled DRAM bytes; ties -> the smallest group >= 1; G = 0 last",
        },
    },
    "sm_103a": {
        "arch": "sm_103a",
        "sku": "B300",
        "l2_total_bytes": 135528448,
        "l2_partitions": 2,
        "l2_resident_fraction": 0.67,
        "l2_budget_bytes": 45402030,
        "k_threshold": 8192,
        "K_THRESHOLD": 8192,
        "SMALL_M": 4096,
        "SMALL_N": 4096,
        "SMALL_K_MIN": 16384,
        "rule_id": "DISPATCH_RULE_v2 (sm_103a: pass 010e07f0 / rerun 50ad704c / old-G 9cce24d2, tree 15939fa63879ca8735429702b34ab2bf3d392129, rule sha256 e219fb3fe716561582885b3caddb3850a9224113c376e46fbfb247bab414c2c8)",
        "choice_rule_order": "ordered, first match wins; every condition in an entry's 'if' must hold (keys <M|N|K|rank>_<lt|le|ge|gt|eq>); the final {'else': program} entry is the default",
        "schema_version": 2,
        "kernels": {
            "pair": {
                "form": "acc2",
                "tile_m": 256,
                "tile_n": 128,
                "sfb_mcast": True,
                "raster_candidates": [0, 1, 8, 16],
                "mainline_order": "along_m",
                "min_k": 0,
            },
            "n256": {
                "form": "rescale",
                "tile_m": 256,
                "tile_n": 256,
                "sfb_mcast": True,
                "hints": False,
                "raster_candidates": [0, 8, 16],
                "mainline_order": "along_m",
                "min_k": 8192,
            },
            "n192": {
                "form": "rescale",
                "tile_m": 256,
                "tile_n": 192,
                "sfb_mcast": True,
                "hints": False,
                "raster_candidates": [0, 8, 16],
                "mainline_order": "along_m",
                "min_k": 0,
            },
            "n192h": {
                "form": "rescale",
                "tile_m": 256,
                "tile_n": 192,
                "sfb_mcast": True,
                "hints": True,
                "raster_candidates": [0, 8, 16],
                "mainline_order": "along_m",
                "min_k": 0,
            },
            "svdq_folded": {
                "form": "folded_last_ordered",
                "tile_m": 256,
                "tile_n": 256,
                "raster_candidates": [0, 8, 16],
                "mainline_order": "along_m",
                "min_k": 2048,
                "dispatched": False,
                "adapter_state": "prepared",
            },
        },
        "choice_rule": [
            {"if": {"K_le": 3072, "M_le": 4096}, "program": "pair"},
            {"if": {"K_le": 3072}, "program": "pair"},
            {"if": {"K_le": 4096, "M_le": 4096}, "program": "pair"},
            {"if": {"K_le": 4096}, "program": "pair"},
            {"if": {"K_le": 6144, "M_le": 4096}, "program": "n192"},
            {"if": {"K_le": 6144}, "program": "n192"},
            {"if": {"K_le": 8192, "N_le": 8192}, "program": "n256"},
            {"if": {"K_le": 8192}, "program": "n256"},
            {
                "if": {"M_le": 4096, "N_le": 4096, "K_ge": 16384},
                "program": "pair",
            },
            {"else": "n256"},
        ],
        "prepared": {
            "kernel": "svdq_folded",
            "min_k": 2048,
            "raster_rule": "k_band_then_n256_bytes_model",
            "k_band_max_k": 4096,
            "k_band_min_m": 32768,
            "k_band_group": 1,
            "bytes_model_candidates": [0, 8, 16],
            "bytes_model_tile_n": 256,
            "bytes_model_pair_m": 256,
            "bytes_model_budget_bytes": 44436029,
            "bytes_model_tie": "fewest modeled DRAM bytes; ties -> the smallest group >= 1; G = 0 last",
        },
    },
}

# Generated by Cake for FlashInfer. Exporter output, hand-edited here pending exporter support.
# One record per generated program, keyed by its schedule template.
# Hand edit: the dispatch, arg_plan and compile_flags values that several records
# share are the module-level constants above, referenced by the records.
PROGRAMS: dict[str, dict[str, Any]] = {
    "fp8_blackwell_n128": {
        "module": "cake_quant_lora_gemm_ca6afb23cac2aa9105fb",
        "device": "cake_quant_lora_gemm_ca6afb23cac2aa9105fb_kernel.cu",
        "binding": "cake_quant_lora_gemm_ca6afb23cac2aa9105fb_binding.cu",
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": _COMPILE_FLAGS_BLACKWELL,
        "ffi_entry": "run",
        "arg_plan": _ARG_PLAN_FP8_BLACKWELL,
        "pdl": True,
        "family": "fp8",
        "form": "rescale",
        "tile_n": 128,
        "grid": "fp8_pairs",
        "operand_view": "uint8",
        "ranks": [32, 64, 128],
        "block_m": 128,
        "block_k": 128,
        "lora_carrier": [128, 32],
        "adapter_states": ["none", "raw"],
        "tile_m": 128,
        "pair_m": 256,
        "raster_group_tokens": 16,
        "tile_ns": [256, 128],
        "sched": "static",
        "raster_mode": "grouped",
        "rules": {"m_min": 1, "n_multiple": 8, "k_multiple": 128, "max_mn": None},
        "dispatch": _DISPATCH_FP8_BLACKWELL,
    },
    "fp8_blackwell_n256": {
        "module": "cake_quant_lora_gemm_33d1d9e4954630ce127f",
        "device": "cake_quant_lora_gemm_33d1d9e4954630ce127f_kernel.cu",
        "binding": "cake_quant_lora_gemm_33d1d9e4954630ce127f_binding.cu",
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": _COMPILE_FLAGS_BLACKWELL,
        "ffi_entry": "run",
        "arg_plan": _ARG_PLAN_FP8_BLACKWELL,
        "pdl": True,
        "family": "fp8",
        "form": "rescale",
        "tile_n": 256,
        "grid": "fp8_pairs",
        "operand_view": "uint8",
        "ranks": [32, 64, 128],
        "block_m": 128,
        "block_k": 128,
        "lora_carrier": [128, 32],
        "adapter_states": ["none", "raw"],
        "tile_m": 128,
        "pair_m": 256,
        "raster_group_tokens": 16,
        "tile_ns": [256, 128],
        "sched": "static",
        "raster_mode": "grouped",
        "rules": {"m_min": 1, "n_multiple": 8, "k_multiple": 128, "max_mn": None},
        "dispatch": _DISPATCH_FP8_BLACKWELL,
    },
    "fp8_blackwell_n256_clc_blocked": {
        "module": "cake_quant_lora_gemm_79090638dfd94335349d",
        "device": "cake_quant_lora_gemm_79090638dfd94335349d_kernel.cu",
        "binding": "cake_quant_lora_gemm_79090638dfd94335349d_binding.cu",
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": _COMPILE_FLAGS_BLACKWELL,
        "ffi_entry": "run",
        "arg_plan": _ARG_PLAN_FP8_BLACKWELL,
        "pdl": True,
        "family": "fp8",
        "form": "rescale",
        "tile_n": 256,
        "grid": "fp8_pairs",
        "operand_view": "uint8",
        "ranks": [32, 64, 128],
        "block_m": 128,
        "block_k": 128,
        "lora_carrier": [128, 32],
        "adapter_states": ["none", "raw"],
        "tile_m": 128,
        "pair_m": 256,
        "raster_group_tokens": 16,
        "tile_ns": [256, 128],
        "sched": "clc",
        "raster_mode": "blocked",
        "rules": {"m_min": 1, "n_multiple": 8, "k_multiple": 128, "max_mn": None},
        "dispatch": _DISPATCH_FP8_BLACKWELL,
    },
    "fp8_blackwell_n256_m256_clc_blocked": {
        "module": "cake_quant_lora_gemm_e58e380eacc1adcd5d34",
        "device": "cake_quant_lora_gemm_e58e380eacc1adcd5d34_kernel.cu",
        "binding": "cake_quant_lora_gemm_e58e380eacc1adcd5d34_binding.cu",
        "arches": ["sm_103a"],
        "compile_flags": _COMPILE_FLAGS_BLACKWELL,
        "ffi_entry": "run",
        "arg_plan": _ARG_PLAN_FP8_BLACKWELL,
        "pdl": True,
        "family": "fp8",
        "form": "rescale",
        "tile_n": 256,
        "grid": "fp8_pairs",
        "operand_view": "uint8",
        "ranks": [32, 64, 128],
        "block_m": 128,
        "block_k": 128,
        "lora_carrier": [128, 32],
        "adapter_states": ["none", "raw"],
        "tile_m": 256,
        "pair_m": 512,
        "raster_group_tokens": 16,
        "tile_ns": [256, 128],
        "sched": "clc",
        "raster_mode": "blocked",
        "rules": {"m_min": 1, "n_multiple": 8, "k_multiple": 128, "max_mn": None},
        "dispatch": _DISPATCH_FP8_BLACKWELL,
    },
    "fp8_sm90": {
        "module": "cake_quant_lora_gemm_58229fa296714c9ace87",
        "device": "cake_quant_lora_gemm_58229fa296714c9ace87_kernel.cu",
        "binding": "cake_quant_lora_gemm_58229fa296714c9ace87_binding.cu",
        "arches": ["sm_90a"],
        "compile_flags": _COMPILE_FLAGS_SM90,
        "ffi_entry": "run",
        "arg_plan": _ARG_PLAN_FP8_SM90,
        "pdl": True,
        "family": "fp8",
        "form": "acc2",
        "tile_n": None,
        "grid": "sm90_persistent",
        "operand_view": None,
        "ranks": [32, 64, 128],
        "block_m": 128,
        "block_k": 128,
        "lora_carrier": [128, 64],
        "adapter_states": ["none", "raw"],
        "block_n": 192,
        "cluster_m": 2,
        "group_m": 8,
        "tiles_per_cluster": None,
        "rules": {"m_min": 1, "n_multiple": 8, "k_multiple": 16, "max_mn": 4294967296},
        "dispatch": _DISPATCH_FP8_SM90,
    },
    "fp8_sm90_hint": {
        "module": "cake_quant_lora_gemm_b24e291cc9bbcb6adad1",
        # Shares the fp8_sm90 kernel source; compile_flags select this program.
        "device": "cake_quant_lora_gemm_58229fa296714c9ace87_kernel.cu",
        "binding": "cake_quant_lora_gemm_b24e291cc9bbcb6adad1_binding.cu",
        "arches": ["sm_90a"],
        "compile_flags": [
            *_COMPILE_FLAGS_SM90,
            "-DCAKE_QLG_PROGRAM_KERNEL=kernel_cake_quant_lora_gemm_b24e291cc9bbcb6adad1",
            "-DCAKE_QLG_FP8_SM90_HINT=1",
        ],
        "ffi_entry": "run",
        "arg_plan": _ARG_PLAN_FP8_SM90,
        "pdl": True,
        "family": "fp8",
        "form": "acc2_hint",
        "tile_n": None,
        "grid": "sm90_persistent",
        "operand_view": None,
        "ranks": [32, 64, 128],
        "block_m": 128,
        "block_k": 128,
        "lora_carrier": [128, 64],
        "adapter_states": ["none", "raw"],
        "block_n": 192,
        "cluster_m": 2,
        "group_m": 2,
        "tiles_per_cluster": 4,
        "rules": {"m_min": 1, "n_multiple": 8, "k_multiple": 16, "max_mn": 4294967296},
        "dispatch": _DISPATCH_FP8_SM90,
    },
    "nvfp4_blackwell_n192": {
        "module": "cake_quant_lora_gemm_aa5410d8a8acee43d949",
        "device": "cake_quant_lora_gemm_aa5410d8a8acee43d949_kernel.cu",
        "binding": "cake_quant_lora_gemm_aa5410d8a8acee43d949_binding.cu",
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": _COMPILE_FLAGS_BLACKWELL,
        "ffi_entry": "run",
        "arg_plan": _ARG_PLAN_NVFP4_BLACKWELL_WIDE,
        "pdl": True,
        "family": "nvfp4",
        "form": "rescale",
        "tile_n": 192,
        "grid": "nvfp4_pairs",
        "operand_view": None,
        "ranks": [32, 64, 128],
        "block_m": 128,
        "block_k": 256,
        "lora_carrier": [128, 32],
        "adapter_states": ["none", "raw"],
        "pair_m": 256,
        "tile_ns": [192],
        "sf_row_pad": 128,
        "sf_mma_k": 64,
        "rules": {"m_min": 1, "n_multiple": 1, "k_multiple": 256, "max_mn": None},
        "dispatch_kernel": "n192",
        "dispatch": _DISPATCH_NVFP4_BLACKWELL,
    },
    "nvfp4_blackwell_n192h": {
        "module": "cake_quant_lora_gemm_1c144a4162f4ba39de6c",
        # Shares the nvfp4_blackwell_n192 kernel source; compile_flags select this program.
        "device": "cake_quant_lora_gemm_aa5410d8a8acee43d949_kernel.cu",
        "binding": "cake_quant_lora_gemm_1c144a4162f4ba39de6c_binding.cu",
        "arches": ["sm_100a"],
        "compile_flags": [
            *_COMPILE_FLAGS_BLACKWELL,
            "-DCAKE_QLG_PROGRAM_KERNEL=kernel_cake_quant_lora_gemm_1c144a4162f4ba39de6c",
            "-DCAKE_QLG_NVFP4_N192_HINT=1",
        ],
        "ffi_entry": "run",
        "arg_plan": _ARG_PLAN_NVFP4_BLACKWELL_WIDE,
        "pdl": True,
        "family": "nvfp4",
        "form": "rescale",
        "tile_n": 192,
        "grid": "nvfp4_pairs",
        "operand_view": None,
        "ranks": [32, 64, 128],
        "block_m": 128,
        "block_k": 256,
        "lora_carrier": [128, 32],
        "adapter_states": ["none", "raw"],
        "pair_m": 256,
        "tile_ns": [192],
        "sf_row_pad": 128,
        "sf_mma_k": 64,
        "rules": {"m_min": 1, "n_multiple": 1, "k_multiple": 256, "max_mn": None},
        "dispatch_kernel": "n192h",
        "dispatch": _DISPATCH_NVFP4_BLACKWELL,
    },
    "nvfp4_blackwell_n256": {
        "module": "cake_quant_lora_gemm_90d64729259715a65e04",
        "device": "cake_quant_lora_gemm_90d64729259715a65e04_kernel.cu",
        "binding": "cake_quant_lora_gemm_90d64729259715a65e04_binding.cu",
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": _COMPILE_FLAGS_BLACKWELL,
        "ffi_entry": "run",
        "arg_plan": _ARG_PLAN_NVFP4_BLACKWELL_WIDE,
        "pdl": True,
        "family": "nvfp4",
        "form": "rescale",
        "tile_n": 256,
        "grid": "nvfp4_pairs",
        "operand_view": None,
        "ranks": [32, 64, 128],
        "block_m": 128,
        "block_k": 256,
        "lora_carrier": [128, 32],
        "adapter_states": ["none", "raw"],
        "pair_m": 256,
        "tile_ns": [256],
        "sf_row_pad": 128,
        "sf_mma_k": 64,
        "rules": {"m_min": 1, "n_multiple": 1, "k_multiple": 256, "max_mn": None},
        "dispatch_kernel": "n256",
        "dispatch": _DISPATCH_NVFP4_BLACKWELL,
    },
    "nvfp4_blackwell_pair": {
        "module": "cake_quant_lora_gemm_f03d94e2d19d6f3c274d",
        "device": "cake_quant_lora_gemm_f03d94e2d19d6f3c274d_kernel.cu",
        "binding": "cake_quant_lora_gemm_f03d94e2d19d6f3c274d_binding.cu",
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": _COMPILE_FLAGS_BLACKWELL,
        "ffi_entry": "run",
        "arg_plan": _ARG_PLAN_NVFP4_BLACKWELL_PAIR,
        "pdl": True,
        "family": "nvfp4",
        "form": "acc2",
        "tile_n": 128,
        "grid": "nvfp4_pairs",
        "operand_view": None,
        "ranks": [32, 64, 128],
        "block_m": 128,
        "block_k": 256,
        "lora_carrier": [128, 32],
        "adapter_states": ["none", "raw"],
        "pair_m": 256,
        "tile_ns": [128],
        "sf_row_pad": 128,
        "sf_mma_k": 64,
        "rules": {"m_min": 1, "n_multiple": 1, "k_multiple": 256, "max_mn": None},
        "dispatch_kernel": "pair",
        "dispatch": _DISPATCH_NVFP4_BLACKWELL,
    },
    "nvfp4_blackwell_svdq": {
        "module": "cake_quant_lora_gemm_140e35bb5ecf92090cf9",
        "device": "cake_quant_lora_gemm_140e35bb5ecf92090cf9_kernel.cu",
        "binding": "cake_quant_lora_gemm_140e35bb5ecf92090cf9_binding.cu",
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": _COMPILE_FLAGS_BLACKWELL,
        "ffi_entry": "run",
        "arg_plan": _ARG_PLAN_NVFP4_BLACKWELL_SVDQ,
        "pdl": True,
        "family": "nvfp4",
        "form": "folded_last_ordered",
        "tile_n": 256,
        "grid": "nvfp4_pairs",
        "operand_view": None,
        "ranks": [32, 64, 128],
        "block_m": 128,
        "block_k": 256,
        "lora_carrier": [128, 32],
        "adapter_states": ["prepared"],
        "pair_m": 256,
        "tile_ns": [256],
        "sf_row_pad": 128,
        "sf_mma_k": 64,
        "rules": {"m_min": 1, "n_multiple": 1, "k_multiple": 256, "max_mn": None},
        "dispatch_kernel": "svdq_folded",
        "dispatch": _DISPATCH_NVFP4_BLACKWELL,
    },
}


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_quant_lora_gemm"
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_quant_lora_gemm"
    for candidate in (installed, checkout):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cake quantized-base LoRA GEMM sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


@functools.cache
def _target_of_device_index(index: int) -> CakeQuantLoraTarget:
    major, minor = get_compute_capability(torch.device("cuda", index))
    target = _CAPABILITY_TARGET.get((major, minor))
    if target is None:
        raise RuntimeError(
            "the Cake quantized-base LoRA GEMM requires exact compute capability "
            f"9.0, 10.0 or 10.3, got {major}.{minor}"
        )
    return target


def cake_quant_lora_target(device: torch.device) -> CakeQuantLoraTarget:
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _target_of_device_index(int(index))


@functools.cache
def gen_cake_quant_lora_module(template: str, target: CakeQuantLoraTarget) -> JitSpec:
    record = PROGRAMS[template]
    arch = TARGET_ARCH[target]
    if arch not in record["arches"]:
        raise RuntimeError(f"generated program {template} does not target {arch}")
    root = _get_csrc_dir()
    spec = gen_jit_spec(
        name=f"{record['module']}_{target}",
        sources=[root / record["device"], root / record["binding"]],
        extra_cuda_cflags=[*_TARGET_FLAGS[target], *record["compile_flags"]],
        extra_include_paths=[root, root.parent, _get_include_dir()],
        needs_device_linking=True,
    )
    logger.info("Generated Cake quantized-base LoRA GEMM JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_quant_lora_module(template: str, target: CakeQuantLoraTarget):
    return gen_cake_quant_lora_module(template, target).build_and_load()


@functools.cache
def cake_quant_lora_launcher(template: str, target: CakeQuantLoraTarget):
    module = load_cake_quant_lora_module(template, target)
    return getattr(module, PROGRAMS[template]["ffi_entry"])


__all__ = [
    "PROGRAMS",
    "TARGET_ARCH",
    "TARGET_OF_ARCH",
    "CakeQuantLoraTarget",
    "cake_quant_lora_launcher",
    "cake_quant_lora_target",
    "gen_cake_quant_lora_module",
    "load_cake_quant_lora_module",
]
