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
"""Generated programs of the Blackwell fused MLA context K/V pack with the saturating fp8
e4m3 cast: one program per head group (the consecutive head pairs a warp owns: 1, 2, 3
or 4), each shipped for sm_100a and sm_103a. The host module ``flashinfer.mla_kv_pack``
selects the program from the token and head counts, computes the grid and launches it.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

HG1 = cake_program.CakeProgram(
    name="cake_concat_mla_kv_quant_fp8_73c47238832b85fec054",
    operator_dir="cake_concat_mla_kv_quant_fp8",
    kernel="kernel_cake_concat_mla_kv_quant_fp8_73c47238832b85fec054",
    sources=(
        "cake_concat_mla_kv_quant_fp8_73c47238832b85fec054_kernel.cu",
        "cake_concat_mla_kv_quant_fp8_73c47238832b85fec054_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "kv_nope",
        "k_pe",
        "key",
        "value",
        "num_tokens",
        "num_heads",
        "head_pairs",
        "warps_per_token",
        "k_pe_row_stride",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("kv_nope", "k_pe", "key", "value"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

HG2 = cake_program.CakeProgram(
    name="cake_concat_mla_kv_quant_fp8_20bd2bcab9e7e31596d8",
    operator_dir="cake_concat_mla_kv_quant_fp8",
    kernel="kernel_cake_concat_mla_kv_quant_fp8_20bd2bcab9e7e31596d8",
    sources=(
        "cake_concat_mla_kv_quant_fp8_20bd2bcab9e7e31596d8_kernel.cu",
        "cake_concat_mla_kv_quant_fp8_20bd2bcab9e7e31596d8_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "kv_nope",
        "k_pe",
        "key",
        "value",
        "num_tokens",
        "num_heads",
        "head_pairs",
        "warps_per_token",
        "k_pe_row_stride",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("kv_nope", "k_pe", "key", "value"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

HG3 = cake_program.CakeProgram(
    name="cake_concat_mla_kv_quant_fp8_10275775b561370999fa",
    operator_dir="cake_concat_mla_kv_quant_fp8",
    kernel="kernel_cake_concat_mla_kv_quant_fp8_10275775b561370999fa",
    sources=(
        "cake_concat_mla_kv_quant_fp8_10275775b561370999fa_kernel.cu",
        "cake_concat_mla_kv_quant_fp8_10275775b561370999fa_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "kv_nope",
        "k_pe",
        "key",
        "value",
        "num_tokens",
        "num_heads",
        "head_pairs",
        "warps_per_token",
        "k_pe_row_stride",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("kv_nope", "k_pe", "key", "value"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

HG4 = cake_program.CakeProgram(
    name="cake_concat_mla_kv_quant_fp8_bbe9bbca6e1f08471502",
    operator_dir="cake_concat_mla_kv_quant_fp8",
    kernel="kernel_cake_concat_mla_kv_quant_fp8_bbe9bbca6e1f08471502",
    sources=(
        "cake_concat_mla_kv_quant_fp8_bbe9bbca6e1f08471502_kernel.cu",
        "cake_concat_mla_kv_quant_fp8_bbe9bbca6e1f08471502_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "kv_nope",
        "k_pe",
        "key",
        "value",
        "num_tokens",
        "num_heads",
        "head_pairs",
        "warps_per_token",
        "k_pe_row_stride",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("kv_nope", "k_pe", "key", "value"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (HG1, HG2, HG3, HG4)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = ["HG1", "HG2", "HG3", "HG4", "PROGRAMS", "aot_specs"]
