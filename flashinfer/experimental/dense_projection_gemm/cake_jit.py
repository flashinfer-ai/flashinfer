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
import hashlib
import json
from typing import Any

from . import cake_build

# Registry of the generated GEMM programs (schema v2).  Populated verbatim by
# the generated-program export; do not edit by hand.  Three literals, each
# wrapped in ``# fmt: off`` / ``# fmt: on`` and rendered one record per line,
# so the export replaces a literal from its head line up to the next
# ``# fmt: on`` line:
#
# * ``ARG_PLANS``: the distinct argument plans, keyed by the first 12 hex digits
#   of the SHA-256 of the plan's compact JSON (``arg_plan_key``).  A plan is the
#   ordered list of ``[kind, name]`` pairs the generated binding's entry takes
#   (``tma_buffer`` / ``buffer`` tensors, ``parameter`` scalars, the ``grid``
#   dims); the host binds it by keyword (``cake_backend.bind_launch``).  Every
#   K1 program shares one plan and every K2 program another, so a plan is
#   stored once instead of once per record.
# * ``MODULES``: one record per delivered program = one traced kernel instance
#   (``template``: the symbol the host planner derives from the operand
#   layouts, the tile shape, the output kind, the epilogue path, the TMA L2
#   eviction hints and the per-launch knobs -- see
#   ``cake_backend.instance_symbol`` and ``cake_backend.router_symbol``).  A
#   program is architecture-neutral (one source pair, built per device
#   architecture); its record carries the kernel symbol, the architecture-free
#   ``library`` stem (``build.json`` names the per-architecture build
#   ``<library>_<arch>`` and records its translation units, compile flags and
#   closure digest -- see ``cake_build``), the FFI entry, the ``arg_plan`` key
#   (``arg_plan_of`` resolves it; an inline plan list is still accepted), the
#   launch geometry baked into the module (block, cluster) and the
#   architectures it serves.
# * ``KERNELS``: ``arch -> template -> module name``, the table the host
#   planner resolves a launch through.
# fmt: off
ARG_PLANS: dict[str, list[list[str]]] = {
    "775c109309ff": [["tma_buffer", "A"], ["tma_buffer", "B"], ["tma_buffer", "OUT32"], ["tma_buffer", "OUT16"], ["buffer", "out"], ["buffer", "out32"], ["buffer", "ws"], ["buffer", "counters"], ["parameter", "M"], ["parameter", "N"], ["parameter", "m_tiles"], ["parameter", "n_tiles"], ["parameter", "group_m"], ["parameter", "promo_code"], ["parameter", "k_iters"], ["parameter", "ldo"], ["parameter", "out_l"], ["parameter", "num_cluster_tiles"], ["parameter", "num_l"], ["parameter", "num_full"], ["parameter", "iters_per_unit"], ["parameter", "sk_iters"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
    "d898389c4086": [["tma_buffer", "A"], ["tma_buffer", "B"], ["buffer", "out32"], ["parameter", "M"], ["parameter", "N"], ["parameter", "m_tiles"], ["parameter", "n_tiles"], ["parameter", "k_iters_split"], ["parameter", "ldo"], ["parameter", "out_l"], ["parameter", "num_cluster_tiles"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
}
# fmt: on

# fmt: off
MODULES: dict[str, dict[str, Any]] = {
    "cake_dense_projection_gemm_011c85b820ec0cf6863d": {"template": "dense_proj_gemm_kn_n128_f32_tma1_c1", "kernel": "cake::dense_projection_gemm::kn_n128_f32_tma1_c1__pd2<false>", "library": "cake_dense_projection_gemm_011c85b820ec0cf6863d", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [1, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_0618ae0aec1af120afd0": {"template": "dense_proj_gemm_kn_n256_tma1_pd2_pro15", "kernel": "kernel_cake_dense_projection_gemm_0618ae0aec1af120afd0", "library": "cake_dense_projection_gemm_0618ae0aec1af120afd0", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_087ba4f157914949c26a": {"template": "dense_proj_gemm_kk_n256_ht_pro15", "kernel": "kernel_cake_dense_projection_gemm_087ba4f157914949c26a", "library": "cake_dense_projection_gemm_087ba4f157914949c26a", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_0a71a49da71c7d34ab37": {"template": "dense_proj_gemm_kk_n128_hen_pro15", "kernel": "kernel_cake_dense_projection_gemm_0a71a49da71c7d34ab37", "library": "cake_dense_projection_gemm_0a71a49da71c7d34ab37", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_0c72e864a4afb5347da6": {"template": "dense_proj_gemm_nn_n128_m64_t_pro7", "kernel": "cake::dense_projection_gemm::nn_n128_m64_t_pro7__f32<false>", "library": "cake_dense_projection_gemm_0c72e864a4afb5347da6", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_0db65f3ddd780ed42d4d": {"template": "dense_proj_gemm_kk_n256_q_pro7", "kernel": "kernel_cake_dense_projection_gemm_0db65f3ddd780ed42d4d", "library": "cake_dense_projection_gemm_0db65f3ddd780ed42d4d", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_1350861ef426bd310e0f": {"template": "dense_proj_gemm_kn_n256_f32_tma1_pro7", "kernel": "cake::dense_projection_gemm::kn_n256_tma1_pro7__f32<true>", "library": "cake_dense_projection_gemm_1350861ef426bd310e0f", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_13d9310f37868a18eace": {"template": "dense_proj_gemm_kn_n256_m256_cg_or_pro7", "kernel": "kernel_cake_dense_projection_gemm_13d9310f37868a18eace", "library": "cake_dense_projection_gemm_13d9310f37868a18eace", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_14192244441c1003db74": {"template": "dense_proj_gemm_kn_n256_cg_pro7", "kernel": "kernel_cake_dense_projection_gemm_14192244441c1003db74", "library": "cake_dense_projection_gemm_14192244441c1003db74", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_1c1874c560375d29237d": {"template": "dense_proj_gemm_kn_n256_pro15", "kernel": "kernel_cake_dense_projection_gemm_1c1874c560375d29237d", "library": "cake_dense_projection_gemm_1c1874c560375d29237d", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_295aef2f2e0332de2372": {"template": "dense_proj_gemm_kk_n256_m256_ov_ht_pro7", "kernel": "kernel_cake_dense_projection_gemm_295aef2f2e0332de2372", "library": "cake_dense_projection_gemm_295aef2f2e0332de2372", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_29d013dfb72e932eaee5": {"template": "dense_proj_gemm_kn_n128_f32_tma2_s4_c1", "kernel": "kernel_cake_dense_projection_gemm_29d013dfb72e932eaee5", "library": "cake_dense_projection_gemm_29d013dfb72e932eaee5", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [1, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_3088b8056e0b7f9ca32c": {"template": "dense_proj_gemm_kn_n256_tma2_pd2_pro15", "kernel": "kernel_cake_dense_projection_gemm_3088b8056e0b7f9ca32c", "library": "cake_dense_projection_gemm_3088b8056e0b7f9ca32c", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_31be7faa868eabcedada": {"template": "dense_proj_gemm_kn_n256_f32_v8_ef_s9_pro15", "kernel": "kernel_cake_dense_projection_gemm_31be7faa868eabcedada", "library": "cake_dense_projection_gemm_31be7faa868eabcedada", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_332b601bf82c22b05ef2": {"template": "dense_proj_gemm_kn_n256_tma1_bg8_sol_pro7", "kernel": "cake::dense_projection_gemm::kn_n256_tma1_bg8_pro7__sol<true>", "library": "cake_dense_projection_gemm_332b601bf82c22b05ef2", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_3507ceddbf4e88a13231": {"template": "dense_proj_gemm_nn_n128_m64_hen_f32_t_pro7", "kernel": "cake::dense_projection_gemm::nn_n128_m64_hen_t_pro7__f32<true>", "library": "cake_dense_projection_gemm_3507ceddbf4e88a13231", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_367fc93d3342f82a1c9f": {"template": "dense_proj_gemm_nn_n256_m256_pro7", "kernel": "cake::dense_projection_gemm::nn_n256_m256_pro7__ht<false>", "library": "cake_dense_projection_gemm_367fc93d3342f82a1c9f", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_36ae2b727d4b1d18694c": {"template": "dense_proj_gemm_nn_n192_bz64_hee_f32_t_sks_sb2_pro7", "kernel": "kernel_cake_dense_projection_gemm_36ae2b727d4b1d18694c", "library": "cake_dense_projection_gemm_36ae2b727d4b1d18694c", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_36d0401698df21c9425f": {"template": "dense_proj_gemm_nn_n128_t_pro7", "kernel": "cake::dense_projection_gemm::nn_n128_t_pro7__f32<false>", "library": "cake_dense_projection_gemm_36d0401698df21c9425f", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_36e10fde6497f82171d4": {"template": "dense_proj_gemm_nn_n192_hee_f32_t_tma1_pro7", "kernel": "cake::dense_projection_gemm::nn_n192_hee_t_tma1_pro7__f32<1>", "library": "cake_dense_projection_gemm_36e10fde6497f82171d4", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_39200edabebcd9eb2edd": {"template": "dense_proj_gemm_kn_n256_tma2_pd2_pro7", "kernel": "kernel_cake_dense_projection_gemm_39200edabebcd9eb2edd", "library": "cake_dense_projection_gemm_39200edabebcd9eb2edd", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_438841f42a0743b63fb5": {"template": "dense_proj_gemm_kk_n256_cg_pro15", "kernel": "kernel_cake_dense_projection_gemm_438841f42a0743b63fb5", "library": "cake_dense_projection_gemm_438841f42a0743b63fb5", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_446497e750658dfb057b": {"template": "dense_proj_gemm_nn_n256_m256_hee_t_tma1_s6_pro7", "kernel": "kernel_cake_dense_projection_gemm_446497e750658dfb057b", "library": "cake_dense_projection_gemm_446497e750658dfb057b", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_451a0f92e10205e345eb": {"template": "dense_proj_gemm_kk_n128_pro7", "kernel": "cake::dense_projection_gemm::kk_n128_pro7__hen<false>", "library": "cake_dense_projection_gemm_451a0f92e10205e345eb", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_49acb91191bbcced9051": {"template": "dense_proj_gemm_kn_n256_tma2_pd3_c1", "kernel": "kernel_cake_dense_projection_gemm_49acb91191bbcced9051", "library": "cake_dense_projection_gemm_49acb91191bbcced9051", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [1, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_4af009e00a1d5c0ad794": {"template": "dense_proj_gemm_kk_n256_m256_cg_or_pro7", "kernel": "cake::dense_projection_gemm::kk_n256_m256_cg_or_pro7__ht<false>", "library": "cake_dense_projection_gemm_4af009e00a1d5c0ad794", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_4bf834aaa9005cdedd12": {"template": "dense_proj_gemm_nn_n256_sks_sb2_pro7", "kernel": "kernel_cake_dense_projection_gemm_4bf834aaa9005cdedd12", "library": "cake_dense_projection_gemm_4bf834aaa9005cdedd12", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_4f31f3828842f20176d6": {"template": "dense_proj_gemm_nn_n128_m64_hen_t_skx_pro15", "kernel": "cake::dense_projection_gemm::nn_n128_m64_hen_t_skx_pro15__f32<false>", "library": "cake_dense_projection_gemm_4f31f3828842f20176d6", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_5003bfbff9a5eab4693c": {"template": "dense_proj_gemm_nn_n256_f32_v8_ef_sks_sb2_pro7", "kernel": "kernel_cake_dense_projection_gemm_5003bfbff9a5eab4693c", "library": "cake_dense_projection_gemm_5003bfbff9a5eab4693c", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_505fbe05ce8b62f0a589": {"template": "dense_proj_gemm_nn_n160_m256_bz64_pro15", "kernel": "kernel_cake_dense_projection_gemm_505fbe05ce8b62f0a589", "library": "cake_dense_projection_gemm_505fbe05ce8b62f0a589", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_5226b6c854d7db395d7e": {"template": "dense_proj_gemm_kn_n256_m256_tma1_bg4_pro7", "kernel": "kernel_cake_dense_projection_gemm_5226b6c854d7db395d7e", "library": "cake_dense_projection_gemm_5226b6c854d7db395d7e", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_53a63652be2155219d76": {"template": "dense_proj_gemm_kk_n256_m256_ht_cg_or_pro7", "kernel": "cake::dense_projection_gemm::kk_n256_m256_cg_or_pro7__ht<true>", "library": "cake_dense_projection_gemm_53a63652be2155219d76", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_53ec24d6b4983bd1f9c0": {"template": "dense_proj_gemm_kk_n192_m256_hen_s5_pro15", "kernel": "kernel_cake_dense_projection_gemm_53ec24d6b4983bd1f9c0", "library": "cake_dense_projection_gemm_53ec24d6b4983bd1f9c0", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_5445e0f16ad84868fa01": {"template": "dense_proj_gemm_nn_n256_m256_f32_v8_ef_ht_pro7", "kernel": "kernel_cake_dense_projection_gemm_5445e0f16ad84868fa01", "library": "cake_dense_projection_gemm_5445e0f16ad84868fa01", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_5568657a526e2040f0bc": {"template": "dense_proj_gemm_nn_n256_m256_ov_ht_pro7", "kernel": "kernel_cake_dense_projection_gemm_5568657a526e2040f0bc", "library": "cake_dense_projection_gemm_5568657a526e2040f0bc", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_55c1b40f71e3fe4a892c": {"template": "dense_proj_gemm_kn_n128_f32_tma1_pd2_c1", "kernel": "cake::dense_projection_gemm::kn_n128_f32_tma1_c1__pd2<true>", "library": "cake_dense_projection_gemm_55c1b40f71e3fe4a892c", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [1, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_563dd919de0e74c71bde": {"template": "dense_proj_gemm_nn_n128_f32_t_pro7", "kernel": "cake::dense_projection_gemm::nn_n128_t_pro7__f32<true>", "library": "cake_dense_projection_gemm_563dd919de0e74c71bde", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_56be8af43d5d0822da87": {"template": "dense_proj_gemm_kk_n192_hee_s8_pro15", "kernel": "kernel_cake_dense_projection_gemm_56be8af43d5d0822da87", "library": "cake_dense_projection_gemm_56be8af43d5d0822da87", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_56c1924c362457a1c3fe": {"template": "dense_proj_gemm_nn_n128_m64_hen_f32_t_skx_pro15", "kernel": "cake::dense_projection_gemm::nn_n128_m64_hen_t_skx_pro15__f32<true>", "library": "cake_dense_projection_gemm_56c1924c362457a1c3fe", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_5a06c5e2f33772a33174": {"template": "dense_proj_gemm_kn_n128_f32_tma1_pro7", "kernel": "kernel_cake_dense_projection_gemm_5a06c5e2f33772a33174", "library": "cake_dense_projection_gemm_5a06c5e2f33772a33174", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_5e77ef909e88a99c4357": {"template": "dense_proj_gemm_nn_n160_m256_bz64_f32_v8_ef_pro15", "kernel": "kernel_cake_dense_projection_gemm_5e77ef909e88a99c4357", "library": "cake_dense_projection_gemm_5e77ef909e88a99c4357", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_6235599f21b10d4ab9f2": {"template": "dense_proj_gemm_nn_n256_m256_f32_ht_tma1_pro7", "kernel": "cake::dense_projection_gemm::nn_n256_m256_f32_tma1_pro7__ht<true>", "library": "cake_dense_projection_gemm_6235599f21b10d4ab9f2", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_66df6446dd2cca3106d3": {"template": "dense_proj_gemm_kn_n256_hne_pro7", "kernel": "kernel_cake_dense_projection_gemm_66df6446dd2cca3106d3", "library": "cake_dense_projection_gemm_66df6446dd2cca3106d3", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_6e17162acb2711a87d39": {"template": "dense_proj_gemm_nn_n256_m256_f32_tma1_pro7", "kernel": "cake::dense_projection_gemm::nn_n256_m256_f32_tma1_pro7__ht<false>", "library": "cake_dense_projection_gemm_6e17162acb2711a87d39", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_6f17e515b2e1b7e2fde9": {"template": "dense_proj_gemm_nn_n128_hen_t_skx_pro7", "kernel": "cake::dense_projection_gemm::nn_n128_hen_t_skx_pro7__f32<false>", "library": "cake_dense_projection_gemm_6f17e515b2e1b7e2fde9", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_73a471b3db6b5aaf0c32": {"template": "dense_proj_gemm_kk_n256_cg_pro7", "kernel": "cake::dense_projection_gemm::kk_n256_pro7__cg<true>", "library": "cake_dense_projection_gemm_73a471b3db6b5aaf0c32", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_765ece06955b70a95b29": {"template": "dense_proj_gemm_kn_n256_m256_f32_v8_ef_ov_ht_pro7", "kernel": "kernel_cake_dense_projection_gemm_765ece06955b70a95b29", "library": "cake_dense_projection_gemm_765ece06955b70a95b29", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_7ad01af3a13775199254": {"template": "dense_proj_gemm_kk_n128_hen_pro7", "kernel": "cake::dense_projection_gemm::kk_n128_pro7__hen<true>", "library": "cake_dense_projection_gemm_7ad01af3a13775199254", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_7c6747659ccbe75ad5d3": {"template": "router_fp32_gemm_nn_t", "kernel": "kernel_cake_dense_projection_gemm_7c6747659ccbe75ad5d3", "library": "cake_dense_projection_gemm_7c6747659ccbe75ad5d3", "ffi_entry": "run", "arg_plan": "d898389c4086", "launch": {"block": [448, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_7d403fa7b167188ccc24": {"template": "dense_proj_gemm_kn_n256_f32_v8_pro7", "kernel": "kernel_cake_dense_projection_gemm_7d403fa7b167188ccc24", "library": "cake_dense_projection_gemm_7d403fa7b167188ccc24", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_8053ebc84a22444c52af": {"template": "dense_proj_gemm_kn_n256_tma1_pro7", "kernel": "cake::dense_projection_gemm::kn_n256_tma1_pro7__f32<false>", "library": "cake_dense_projection_gemm_8053ebc84a22444c52af", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_806927d101fe72fc1317": {"template": "dense_proj_gemm_nn_n256_m256_ht_tma1_pro7", "kernel": "kernel_cake_dense_projection_gemm_806927d101fe72fc1317", "library": "cake_dense_projection_gemm_806927d101fe72fc1317", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_82f698f67a1cca6e4291": {"template": "dense_proj_gemm_nn_n256_pro15", "kernel": "kernel_cake_dense_projection_gemm_82f698f67a1cca6e4291", "library": "cake_dense_projection_gemm_82f698f67a1cca6e4291", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_895f9ba7159bea07aedd": {"template": "dense_proj_gemm_kn_n256_f32_v8_ef_pro7", "kernel": "kernel_cake_dense_projection_gemm_895f9ba7159bea07aedd", "library": "cake_dense_projection_gemm_895f9ba7159bea07aedd", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_8e1f6805342f620cca8c": {"template": "dense_proj_gemm_nn_n256_f32_v8_pro15", "kernel": "kernel_cake_dense_projection_gemm_8e1f6805342f620cca8c", "library": "cake_dense_projection_gemm_8e1f6805342f620cca8c", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_9aa9aa60b0c544bfab06": {"template": "dense_proj_gemm_nn_n192_m256_bz64_hee_t_tma1_s6_pro7", "kernel": "kernel_cake_dense_projection_gemm_9aa9aa60b0c544bfab06", "library": "cake_dense_projection_gemm_9aa9aa60b0c544bfab06", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_9cc6363566760a38b4fe": {"template": "dense_proj_gemm_kn_n256_m256_ov_ht_pro7", "kernel": "kernel_cake_dense_projection_gemm_9cc6363566760a38b4fe", "library": "cake_dense_projection_gemm_9cc6363566760a38b4fe", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_a6ff52c845eac9296b34": {"template": "dense_proj_gemm_kk_n256_pro7", "kernel": "cake::dense_projection_gemm::kk_n256_pro7__cg<false>", "library": "cake_dense_projection_gemm_a6ff52c845eac9296b34", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_b09222ffcfa7a7e4ead7": {"template": "dense_proj_gemm_nn_n128_hen_f32_t_skx_pro7", "kernel": "cake::dense_projection_gemm::nn_n128_hen_t_skx_pro7__f32<true>", "library": "cake_dense_projection_gemm_b09222ffcfa7a7e4ead7", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_b26ac3f3ed56a78cb052": {"template": "dense_proj_gemm_nn_n192_m256_t_tma1_pro7", "kernel": "kernel_cake_dense_projection_gemm_b26ac3f3ed56a78cb052", "library": "cake_dense_projection_gemm_b26ac3f3ed56a78cb052", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_b43a23d219b0196769dc": {"template": "dense_proj_gemm_nn_n256_f32_v8_pro7", "kernel": "kernel_cake_dense_projection_gemm_b43a23d219b0196769dc", "library": "cake_dense_projection_gemm_b43a23d219b0196769dc", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_b58324007fd26df8cf2c": {"template": "dense_proj_gemm_kn_n256_ht_pro15", "kernel": "kernel_cake_dense_projection_gemm_b58324007fd26df8cf2c", "library": "cake_dense_projection_gemm_b58324007fd26df8cf2c", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_b58b50619ef5ed46e109": {"template": "dense_proj_gemm_kn_n256_f32_tma1_pro15", "kernel": "kernel_cake_dense_projection_gemm_b58b50619ef5ed46e109", "library": "cake_dense_projection_gemm_b58b50619ef5ed46e109", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_b9c3f4eb125b638d7c87": {"template": "dense_proj_gemm_kn_n256_f32_v8_ht_pro7", "kernel": "kernel_cake_dense_projection_gemm_b9c3f4eb125b638d7c87", "library": "cake_dense_projection_gemm_b9c3f4eb125b638d7c87", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_c15429b8d7d808a760ea": {"template": "dense_proj_gemm_kn_n256_pro7", "kernel": "kernel_cake_dense_projection_gemm_c15429b8d7d808a760ea", "library": "cake_dense_projection_gemm_c15429b8d7d808a760ea", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_c1568b6644ff567f3b1e": {"template": "router_fp32_gemm_kn", "kernel": "kernel_cake_dense_projection_gemm_c1568b6644ff567f3b1e", "library": "cake_dense_projection_gemm_c1568b6644ff567f3b1e", "ffi_entry": "run", "arg_plan": "d898389c4086", "launch": {"block": [448, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_c195846d649ef9968762": {"template": "dense_proj_gemm_kk_n128_hen_c1", "kernel": "cake::dense_projection_gemm::kk_n128_c1__hen<true>", "library": "cake_dense_projection_gemm_c195846d649ef9968762", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [1, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_ca9a846738b1443f7318": {"template": "dense_proj_gemm_nn_n256_m256_t_tma1_pro7", "kernel": "kernel_cake_dense_projection_gemm_ca9a846738b1443f7318", "library": "cake_dense_projection_gemm_ca9a846738b1443f7318", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_cb45dbd62c7edd4410f1": {"template": "dense_proj_gemm_kn_n256_tma1_bg8_pro7", "kernel": "cake::dense_projection_gemm::kn_n256_tma1_bg8_pro7__sol<false>", "library": "cake_dense_projection_gemm_cb45dbd62c7edd4410f1", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_cc2c62bd9132f8273125": {"template": "dense_proj_gemm_kk_n128_c1", "kernel": "cake::dense_projection_gemm::kk_n128_c1__hen<false>", "library": "cake_dense_projection_gemm_cc2c62bd9132f8273125", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [1, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_d2a5e70bd49083888179": {"template": "dense_proj_gemm_nk_n256_t_pro7", "kernel": "kernel_cake_dense_projection_gemm_d2a5e70bd49083888179", "library": "cake_dense_projection_gemm_d2a5e70bd49083888179", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_d6ebaab2415ea7b00de9": {"template": "dense_proj_gemm_nn_n192_bz64_hee_f32_t_pro7", "kernel": "kernel_cake_dense_projection_gemm_d6ebaab2415ea7b00de9", "library": "cake_dense_projection_gemm_d6ebaab2415ea7b00de9", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_ddff2b911137449c1dfc": {"template": "dense_proj_gemm_nn_n192_bz64_hee_t_sks_stt_pro15", "kernel": "kernel_cake_dense_projection_gemm_ddff2b911137449c1dfc", "library": "cake_dense_projection_gemm_ddff2b911137449c1dfc", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_de7492eeb6e24b31687a": {"template": "dense_proj_gemm_kn_n256_m256_f32_tma1_pro7", "kernel": "kernel_cake_dense_projection_gemm_de7492eeb6e24b31687a", "library": "cake_dense_projection_gemm_de7492eeb6e24b31687a", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_df649506b6ba6f6e47ef": {"template": "dense_proj_gemm_nn_n192_hee_t_tma1_pro7", "kernel": "cake::dense_projection_gemm::nn_n192_hee_t_tma1_pro7__f32<0>", "library": "cake_dense_projection_gemm_df649506b6ba6f6e47ef", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_e2692c71bc236502ba68": {"template": "dense_proj_gemm_nn_n256_m256_tma1_pro7", "kernel": "kernel_cake_dense_projection_gemm_e2692c71bc236502ba68", "library": "cake_dense_projection_gemm_e2692c71bc236502ba68", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_e51cf65dcafa642e2c51": {"template": "dense_proj_gemm_kk_n256_pro15", "kernel": "kernel_cake_dense_projection_gemm_e51cf65dcafa642e2c51", "library": "cake_dense_projection_gemm_e51cf65dcafa642e2c51", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_e98b3ee7b774c63d7544": {"template": "dense_proj_gemm_kn_n256", "kernel": "kernel_cake_dense_projection_gemm_e98b3ee7b774c63d7544", "library": "cake_dense_projection_gemm_e98b3ee7b774c63d7544", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_ed6ef54b238c5da2f5a7": {"template": "dense_proj_gemm_kn_n256_m256_ov_ht_pro15", "kernel": "kernel_cake_dense_projection_gemm_ed6ef54b238c5da2f5a7", "library": "cake_dense_projection_gemm_ed6ef54b238c5da2f5a7", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_eed91d80f7b7ad8493f4": {"template": "router_fp32_gemm_kk", "kernel": "kernel_cake_dense_projection_gemm_eed91d80f7b7ad8493f4", "library": "cake_dense_projection_gemm_eed91d80f7b7ad8493f4", "ffi_entry": "run", "arg_plan": "d898389c4086", "launch": {"block": [448, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_f052eb094121b03461ab": {"template": "dense_proj_gemm_nn_n256_m256_ht_pro7", "kernel": "cake::dense_projection_gemm::nn_n256_m256_pro7__ht<true>", "library": "cake_dense_projection_gemm_f052eb094121b03461ab", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_f258d45b7e98001e0c09": {"template": "dense_proj_gemm_nn_n128_m64_f32_t_pro7", "kernel": "cake::dense_projection_gemm::nn_n128_m64_t_pro7__f32<true>", "library": "cake_dense_projection_gemm_f258d45b7e98001e0c09", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_f2e6d4e79cf21105fa1e": {"template": "dense_proj_gemm_kk_n192_pro7", "kernel": "kernel_cake_dense_projection_gemm_f2e6d4e79cf21105fa1e", "library": "cake_dense_projection_gemm_f2e6d4e79cf21105fa1e", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_107a"]},
    "cake_dense_projection_gemm_f31e1dcb1df00973dc0c": {"template": "dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7", "kernel": "kernel_cake_dense_projection_gemm_f31e1dcb1df00973dc0c", "library": "cake_dense_projection_gemm_f31e1dcb1df00973dc0c", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_f861bf1312b9c7fe78c1": {"template": "dense_proj_gemm_nn_n128_m64_hen_t_pro7", "kernel": "cake::dense_projection_gemm::nn_n128_m64_hen_t_pro7__f32<false>", "library": "cake_dense_projection_gemm_f861bf1312b9c7fe78c1", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
    "cake_dense_projection_gemm_fb4bcdf13915dfb96c59": {"template": "dense_proj_gemm_kn_n256_q_pd1_pro7", "kernel": "kernel_cake_dense_projection_gemm_fb4bcdf13915dfb96c59", "library": "cake_dense_projection_gemm_fb4bcdf13915dfb96c59", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_107a"]},
    "cake_dense_projection_gemm_fd8a34ec5fffdef4f902": {"template": "dense_proj_gemm_nn_n256_m256_sks_sb2_pro7", "kernel": "kernel_cake_dense_projection_gemm_fd8a34ec5fffdef4f902", "library": "cake_dense_projection_gemm_fd8a34ec5fffdef4f902", "ffi_entry": "run", "arg_plan": "775c109309ff", "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a"]},
}
# fmt: on

# fmt: off
KERNELS: dict[str, dict[str, str]] = {
    "sm_100a": {"dense_proj_gemm_kk_n128_c1": "cake_dense_projection_gemm_cc2c62bd9132f8273125", "dense_proj_gemm_kk_n128_hen_c1": "cake_dense_projection_gemm_c195846d649ef9968762", "dense_proj_gemm_kk_n192_hee_s8_pro15": "cake_dense_projection_gemm_56be8af43d5d0822da87", "dense_proj_gemm_kk_n192_pro7": "cake_dense_projection_gemm_f2e6d4e79cf21105fa1e", "dense_proj_gemm_kk_n256_cg_pro7": "cake_dense_projection_gemm_73a471b3db6b5aaf0c32", "dense_proj_gemm_kk_n256_m256_ov_ht_pro7": "cake_dense_projection_gemm_295aef2f2e0332de2372", "dense_proj_gemm_kk_n256_pro15": "cake_dense_projection_gemm_e51cf65dcafa642e2c51", "dense_proj_gemm_kk_n256_pro7": "cake_dense_projection_gemm_a6ff52c845eac9296b34", "dense_proj_gemm_kk_n256_q_pro7": "cake_dense_projection_gemm_0db65f3ddd780ed42d4d", "dense_proj_gemm_kn_n128_f32_tma1_pro7": "cake_dense_projection_gemm_5a06c5e2f33772a33174", "dense_proj_gemm_kn_n128_f32_tma2_s4_c1": "cake_dense_projection_gemm_29d013dfb72e932eaee5", "dense_proj_gemm_kn_n256": "cake_dense_projection_gemm_e98b3ee7b774c63d7544", "dense_proj_gemm_kn_n256_f32_tma1_pro7": "cake_dense_projection_gemm_1350861ef426bd310e0f", "dense_proj_gemm_kn_n256_f32_v8_ef_pro7": "cake_dense_projection_gemm_895f9ba7159bea07aedd", "dense_proj_gemm_kn_n256_f32_v8_pro7": "cake_dense_projection_gemm_7d403fa7b167188ccc24", "dense_proj_gemm_kn_n256_m256_f32_v8_ef_ov_ht_pro7": "cake_dense_projection_gemm_765ece06955b70a95b29", "dense_proj_gemm_kn_n256_m256_ov_ht_pro7": "cake_dense_projection_gemm_9cc6363566760a38b4fe", "dense_proj_gemm_kn_n256_m256_tma1_bg4_pro7": "cake_dense_projection_gemm_5226b6c854d7db395d7e", "dense_proj_gemm_kn_n256_pro15": "cake_dense_projection_gemm_1c1874c560375d29237d", "dense_proj_gemm_kn_n256_pro7": "cake_dense_projection_gemm_c15429b8d7d808a760ea", "dense_proj_gemm_kn_n256_tma1_pd2_pro15": "cake_dense_projection_gemm_0618ae0aec1af120afd0", "dense_proj_gemm_kn_n256_tma1_pro7": "cake_dense_projection_gemm_8053ebc84a22444c52af", "dense_proj_gemm_kn_n256_tma2_pd2_pro15": "cake_dense_projection_gemm_3088b8056e0b7f9ca32c", "dense_proj_gemm_kn_n256_tma2_pd2_pro7": "cake_dense_projection_gemm_39200edabebcd9eb2edd", "dense_proj_gemm_nk_n256_t_pro7": "cake_dense_projection_gemm_d2a5e70bd49083888179", "dense_proj_gemm_nn_n128_m64_f32_t_pro7": "cake_dense_projection_gemm_f258d45b7e98001e0c09", "dense_proj_gemm_nn_n128_m64_hen_f32_t_pro7": "cake_dense_projection_gemm_3507ceddbf4e88a13231", "dense_proj_gemm_nn_n128_m64_hen_t_pro7": "cake_dense_projection_gemm_f861bf1312b9c7fe78c1", "dense_proj_gemm_nn_n128_m64_t_pro7": "cake_dense_projection_gemm_0c72e864a4afb5347da6", "dense_proj_gemm_nn_n192_hee_f32_t_tma1_pro7": "cake_dense_projection_gemm_36e10fde6497f82171d4", "dense_proj_gemm_nn_n192_hee_t_tma1_pro7": "cake_dense_projection_gemm_df649506b6ba6f6e47ef", "dense_proj_gemm_nn_n192_m256_t_tma1_pro7": "cake_dense_projection_gemm_b26ac3f3ed56a78cb052", "dense_proj_gemm_nn_n256_f32_v8_ef_sks_sb2_pro7": "cake_dense_projection_gemm_5003bfbff9a5eab4693c", "dense_proj_gemm_nn_n256_f32_v8_pro7": "cake_dense_projection_gemm_b43a23d219b0196769dc", "dense_proj_gemm_nn_n256_m256_f32_ht_tma1_pro7": "cake_dense_projection_gemm_6235599f21b10d4ab9f2", "dense_proj_gemm_nn_n256_m256_f32_tma1_pro7": "cake_dense_projection_gemm_6e17162acb2711a87d39", "dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7": "cake_dense_projection_gemm_f31e1dcb1df00973dc0c", "dense_proj_gemm_nn_n256_m256_ht_pro7": "cake_dense_projection_gemm_f052eb094121b03461ab", "dense_proj_gemm_nn_n256_m256_ht_tma1_pro7": "cake_dense_projection_gemm_806927d101fe72fc1317", "dense_proj_gemm_nn_n256_m256_ov_ht_pro7": "cake_dense_projection_gemm_5568657a526e2040f0bc", "dense_proj_gemm_nn_n256_m256_pro7": "cake_dense_projection_gemm_367fc93d3342f82a1c9f", "dense_proj_gemm_nn_n256_m256_sks_sb2_pro7": "cake_dense_projection_gemm_fd8a34ec5fffdef4f902", "dense_proj_gemm_nn_n256_m256_t_tma1_pro7": "cake_dense_projection_gemm_ca9a846738b1443f7318", "dense_proj_gemm_nn_n256_sks_sb2_pro7": "cake_dense_projection_gemm_4bf834aaa9005cdedd12", "router_fp32_gemm_kk": "cake_dense_projection_gemm_eed91d80f7b7ad8493f4", "router_fp32_gemm_kn": "cake_dense_projection_gemm_c1568b6644ff567f3b1e", "router_fp32_gemm_nn_t": "cake_dense_projection_gemm_7c6747659ccbe75ad5d3"},
    "sm_107a": {"dense_proj_gemm_kk_n128_hen_pro15": "cake_dense_projection_gemm_0a71a49da71c7d34ab37", "dense_proj_gemm_kk_n128_hen_pro7": "cake_dense_projection_gemm_7ad01af3a13775199254", "dense_proj_gemm_kk_n128_pro7": "cake_dense_projection_gemm_451a0f92e10205e345eb", "dense_proj_gemm_kk_n192_m256_hen_s5_pro15": "cake_dense_projection_gemm_53ec24d6b4983bd1f9c0", "dense_proj_gemm_kk_n192_pro7": "cake_dense_projection_gemm_f2e6d4e79cf21105fa1e", "dense_proj_gemm_kk_n256_cg_pro15": "cake_dense_projection_gemm_438841f42a0743b63fb5", "dense_proj_gemm_kk_n256_ht_pro15": "cake_dense_projection_gemm_087ba4f157914949c26a", "dense_proj_gemm_kk_n256_m256_cg_or_pro7": "cake_dense_projection_gemm_4af009e00a1d5c0ad794", "dense_proj_gemm_kk_n256_m256_ht_cg_or_pro7": "cake_dense_projection_gemm_53a63652be2155219d76", "dense_proj_gemm_kk_n256_pro15": "cake_dense_projection_gemm_e51cf65dcafa642e2c51", "dense_proj_gemm_kk_n256_q_pro7": "cake_dense_projection_gemm_0db65f3ddd780ed42d4d", "dense_proj_gemm_kn_n128_f32_tma1_c1": "cake_dense_projection_gemm_011c85b820ec0cf6863d", "dense_proj_gemm_kn_n128_f32_tma1_pd2_c1": "cake_dense_projection_gemm_55c1b40f71e3fe4a892c", "dense_proj_gemm_kn_n256_cg_pro7": "cake_dense_projection_gemm_14192244441c1003db74", "dense_proj_gemm_kn_n256_f32_tma1_pro15": "cake_dense_projection_gemm_b58b50619ef5ed46e109", "dense_proj_gemm_kn_n256_f32_v8_ef_s9_pro15": "cake_dense_projection_gemm_31be7faa868eabcedada", "dense_proj_gemm_kn_n256_f32_v8_ht_pro7": "cake_dense_projection_gemm_b9c3f4eb125b638d7c87", "dense_proj_gemm_kn_n256_f32_v8_pro7": "cake_dense_projection_gemm_7d403fa7b167188ccc24", "dense_proj_gemm_kn_n256_hne_pro7": "cake_dense_projection_gemm_66df6446dd2cca3106d3", "dense_proj_gemm_kn_n256_ht_pro15": "cake_dense_projection_gemm_b58324007fd26df8cf2c", "dense_proj_gemm_kn_n256_m256_cg_or_pro7": "cake_dense_projection_gemm_13d9310f37868a18eace", "dense_proj_gemm_kn_n256_m256_f32_tma1_pro7": "cake_dense_projection_gemm_de7492eeb6e24b31687a", "dense_proj_gemm_kn_n256_m256_ov_ht_pro15": "cake_dense_projection_gemm_ed6ef54b238c5da2f5a7", "dense_proj_gemm_kn_n256_pro15": "cake_dense_projection_gemm_1c1874c560375d29237d", "dense_proj_gemm_kn_n256_pro7": "cake_dense_projection_gemm_c15429b8d7d808a760ea", "dense_proj_gemm_kn_n256_q_pd1_pro7": "cake_dense_projection_gemm_fb4bcdf13915dfb96c59", "dense_proj_gemm_kn_n256_tma1_bg8_pro7": "cake_dense_projection_gemm_cb45dbd62c7edd4410f1", "dense_proj_gemm_kn_n256_tma1_bg8_sol_pro7": "cake_dense_projection_gemm_332b601bf82c22b05ef2", "dense_proj_gemm_kn_n256_tma2_pd2_pro15": "cake_dense_projection_gemm_3088b8056e0b7f9ca32c", "dense_proj_gemm_kn_n256_tma2_pd3_c1": "cake_dense_projection_gemm_49acb91191bbcced9051", "dense_proj_gemm_nk_n256_t_pro7": "cake_dense_projection_gemm_d2a5e70bd49083888179", "dense_proj_gemm_nn_n128_f32_t_pro7": "cake_dense_projection_gemm_563dd919de0e74c71bde", "dense_proj_gemm_nn_n128_hen_f32_t_skx_pro7": "cake_dense_projection_gemm_b09222ffcfa7a7e4ead7", "dense_proj_gemm_nn_n128_hen_t_skx_pro7": "cake_dense_projection_gemm_6f17e515b2e1b7e2fde9", "dense_proj_gemm_nn_n128_m64_hen_f32_t_skx_pro15": "cake_dense_projection_gemm_56c1924c362457a1c3fe", "dense_proj_gemm_nn_n128_m64_hen_t_skx_pro15": "cake_dense_projection_gemm_4f31f3828842f20176d6", "dense_proj_gemm_nn_n128_t_pro7": "cake_dense_projection_gemm_36d0401698df21c9425f", "dense_proj_gemm_nn_n160_m256_bz64_f32_v8_ef_pro15": "cake_dense_projection_gemm_5e77ef909e88a99c4357", "dense_proj_gemm_nn_n160_m256_bz64_pro15": "cake_dense_projection_gemm_505fbe05ce8b62f0a589", "dense_proj_gemm_nn_n192_bz64_hee_f32_t_pro7": "cake_dense_projection_gemm_d6ebaab2415ea7b00de9", "dense_proj_gemm_nn_n192_bz64_hee_f32_t_sks_sb2_pro7": "cake_dense_projection_gemm_36ae2b727d4b1d18694c", "dense_proj_gemm_nn_n192_bz64_hee_t_sks_stt_pro15": "cake_dense_projection_gemm_ddff2b911137449c1dfc", "dense_proj_gemm_nn_n192_m256_bz64_hee_t_tma1_s6_pro7": "cake_dense_projection_gemm_9aa9aa60b0c544bfab06", "dense_proj_gemm_nn_n256_f32_v8_ef_sks_sb2_pro7": "cake_dense_projection_gemm_5003bfbff9a5eab4693c", "dense_proj_gemm_nn_n256_f32_v8_pro15": "cake_dense_projection_gemm_8e1f6805342f620cca8c", "dense_proj_gemm_nn_n256_m256_f32_ht_tma1_pro7": "cake_dense_projection_gemm_6235599f21b10d4ab9f2", "dense_proj_gemm_nn_n256_m256_f32_tma1_pro7": "cake_dense_projection_gemm_6e17162acb2711a87d39", "dense_proj_gemm_nn_n256_m256_f32_v8_ef_ht_pro7": "cake_dense_projection_gemm_5445e0f16ad84868fa01", "dense_proj_gemm_nn_n256_m256_hee_t_tma1_s6_pro7": "cake_dense_projection_gemm_446497e750658dfb057b", "dense_proj_gemm_nn_n256_m256_ht_pro7": "cake_dense_projection_gemm_f052eb094121b03461ab", "dense_proj_gemm_nn_n256_m256_pro7": "cake_dense_projection_gemm_367fc93d3342f82a1c9f", "dense_proj_gemm_nn_n256_m256_tma1_pro7": "cake_dense_projection_gemm_e2692c71bc236502ba68", "dense_proj_gemm_nn_n256_pro15": "cake_dense_projection_gemm_82f698f67a1cca6e4291", "dense_proj_gemm_nn_n256_sks_sb2_pro7": "cake_dense_projection_gemm_4bf834aaa9005cdedd12", "router_fp32_gemm_kk": "cake_dense_projection_gemm_eed91d80f7b7ad8493f4", "router_fp32_gemm_kn": "cake_dense_projection_gemm_c1568b6644ff567f3b1e", "router_fp32_gemm_nn_t": "cake_dense_projection_gemm_7c6747659ccbe75ad5d3"},
}
# fmt: on

# Generated file list: the device / binding translation units of every registered
# module live flat under ``csrc/cake_dense_projection_gemm/`` next to this file
# (one architecture-neutral pair per program, built for every architecture in
# the record's ``arches``); ``build.json`` lists them per library entry.  They
# are written by the Cake exporter together with the three registries above and
# the manifest.

# The architectures this checkout registers programs for (``KERNELS`` keys and
# every record's ``arches`` are drawn from it).  SM103 (B300 / GB300) has no
# registered program: a CC 10.3 device ends in ``NotImplementedError`` from
# ``select_module``.
ARCHES = ("sm_100a", "sm_107a")
TRACKING_ISSUE = "flashinfer-ai/flashinfer#5677"


def arg_plan_key(plan) -> str:
    """Registry key of an argument plan: the first 12 hex digits of the SHA-256 of its compact JSON
    (``[[kind, name], ...]`` with ``,`` / ``:`` separators) -- the export computes the same key."""
    normalised = [[str(kind), str(name)] for kind, name in plan]
    return hashlib.sha256(
        json.dumps(normalised, separators=(",", ":")).encode()
    ).hexdigest()[:12]


def arg_plan_of(record: dict[str, Any]) -> list[list[str]]:
    """The argument plan of a registry record: a ``str`` is a key into ``ARG_PLANS`` (schema v2); an
    inline list of ``[kind, name]`` pairs (a record from before the shared plans, or a test double) is
    taken as is.  Fails closed on a key ``ARG_PLANS`` does not hold."""
    plan = record["arg_plan"]
    if isinstance(plan, str):
        try:
            return ARG_PLANS[plan]
        except KeyError:
            raise KeyError(
                f"registry record {record.get('library', record.get('kernel'))!r} names the "
                f"argument plan {plan!r}, which ARG_PLANS does not hold (keys: {sorted(ARG_PLANS)})"
            ) from None
    return [list(item) for item in plan]


def registered_templates(arch: str) -> tuple[str, ...]:
    """Instance templates registered for ``arch``, sorted."""
    return tuple(sorted(KERNELS.get(arch, {})))


def select_module(arch: str, template: str) -> str:
    """Return the registered module name serving ``template`` on ``arch``."""
    name = KERNELS.get(arch, {}).get(template)
    if name is None:
        registered = registered_templates(arch)
        raise NotImplementedError(
            f"The generated dense projection GEMM instance {template!r} for {arch} is "
            f"not registered in this checkout (registered: {list(registered) or 'none'}; "
            f"see {TRACKING_ISSUE})"
        )
    record = MODULES.get(name)
    if (
        record is None
        or arch not in record.get("arches", ())
        or record.get("template") != template
    ):
        raise ValueError(
            f"registry record {name!r} does not serve template {template!r} on {arch}"
        )
    return name


def library_name(name: str, arch: str) -> str:
    """``build.json`` entry of program ``name`` built for ``arch``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise ValueError(
            f"generated dense projection GEMM module {name!r} is not registered for {arch} "
            f"(registered: {list(record['arches'])})"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load_cake_dense_projection_gemm_module(name: str, arch: str):
    """Load (once per program and architecture) the generated program ``name`` for ``arch``: the
    prebuilt library under ``aot/`` when the package ships it, else FlashInfer's JIT build of its
    ``build.json`` entry."""
    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str):
    """JIT specs of every ``build.json`` entry of ``arch`` without a prebuilt library (the
    ``flashinfer.jit.build_jit_specs`` warm-up)."""
    return cake_build.jit_specs(arch)
