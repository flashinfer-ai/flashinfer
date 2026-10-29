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
"""Generated programs of the MiniMax-H3 SM120 NVFP4-QK / FP8-PV packed-varlen attention:
the fused K/V statistics launch (KV_STATS), the quantizer in two launch-bounds builds
(QUANTIZE, QUANTIZE_MB3) and the attention launch in its base and short schedules
(ATTENTION_BASE, ATTENTION_SHORT), each shipped for sm_120a. The host module
``flashinfer.diffusion_ops.cake_minimax_h3_sm120_nvfp4_varlen_attention`` selects the
programs, computes the grids and launches them behind ``pv_mode="fp8"``.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

KV_STATS = cake_program.CakeProgram(
    name="cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_889b9d3514af07aa659e",
    operator_dir="cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention",
    kernel="kernel_cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_889b9d3514af07aa659e",
    sources=(
        "cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_889b9d3514af07aa659e_kernel.cu",
        "cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_889b9d3514af07aa659e_binding.cu",
    ),
    targets=("sm_120a",),
    compile_options=(("sm_120a", ()),),
    args=(
        "K",
        "V",
        "tile_table",
        "partials",
        "knorm_part",
        "counters",
        "mean_k",
        "v_scale",
        "k_scale",
        "num_tiles",
        "num_heads",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "K",
        "V",
        "tile_table",
        "partials",
        "knorm_part",
        "counters",
        "mean_k",
        "v_scale",
        "k_scale",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=16512,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

QUANTIZE = cake_program.CakeProgram(
    name="cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_a9cc2416e8a621d5cc13",
    operator_dir="cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention",
    kernel="kernel_cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_a9cc2416e8a621d5cc13",
    sources=(
        "cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_a9cc2416e8a621d5cc13_kernel.cu",
        "cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_a9cc2416e8a621d5cc13_binding.cu",
    ),
    targets=("sm_120a",),
    compile_options=(("sm_120a", ()),),
    args=(
        "Q",
        "K",
        "V",
        "tok_seg",
        "mean_k",
        "v_scale",
        "q4",
        "k4",
        "q_sf",
        "k_sf",
        "vt8",
        "q_scale",
        "k_scale",
        "total_tokens",
        "total_padded",
        "num_heads",
        "num_kblocks",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Q",
        "K",
        "V",
        "tok_seg",
        "mean_k",
        "v_scale",
        "q4",
        "k4",
        "q_sf",
        "k_sf",
        "vt8",
        "q_scale",
        "k_scale",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

QUANTIZE_MB3 = cake_program.CakeProgram(
    name="cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_bbf8dd026d615bd67e12",
    operator_dir="cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention",
    kernel="kernel_cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_bbf8dd026d615bd67e12",
    sources=(
        "cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_bbf8dd026d615bd67e12_kernel.cu",
        "cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_bbf8dd026d615bd67e12_binding.cu",
    ),
    targets=("sm_120a",),
    compile_options=(("sm_120a", ()),),
    args=(
        "Q",
        "K",
        "V",
        "tok_seg",
        "mean_k",
        "v_scale",
        "q4",
        "k4",
        "q_sf",
        "k_sf",
        "vt8",
        "q_scale",
        "k_scale",
        "total_tokens",
        "total_padded",
        "num_heads",
        "num_kblocks",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Q",
        "K",
        "V",
        "tok_seg",
        "mean_k",
        "v_scale",
        "q4",
        "k4",
        "q_sf",
        "k_sf",
        "vt8",
        "q_scale",
        "k_scale",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

ATTENTION_BASE = cake_program.CakeProgram(
    name="cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_0e987fea14036520ea94",
    operator_dir="cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention",
    kernel="kernel_cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_0e987fea14036520ea94",
    sources=(
        "cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_0e987fea14036520ea94_kernel.cu",
        "cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_0e987fea14036520ea94_binding.cu",
    ),
    targets=("sm_120a",),
    compile_options=(("sm_120a", ()),),
    args=(
        "Q_map",
        "K_map",
        "KSF_map",
        "V_map",
        "O",
        "q_scale",
        "k_scale",
        "v_scale",
        "q_sf",
        "unit_table",
        "total_units",
        "num_heads",
        "num_kblocks",
        "softmax_scale_log2",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Q_map",
        "K_map",
        "KSF_map",
        "V_map",
        "O",
        "q_scale",
        "k_scale",
        "v_scale",
        "q_sf",
        "unit_table",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=60416,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

ATTENTION_SHORT = cake_program.CakeProgram(
    name="cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_be0f418eaf8afee6ba8d",
    operator_dir="cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention",
    kernel="kernel_cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_be0f418eaf8afee6ba8d",
    sources=(
        "cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_be0f418eaf8afee6ba8d_kernel.cu",
        "cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_be0f418eaf8afee6ba8d_binding.cu",
    ),
    targets=("sm_120a",),
    compile_options=(("sm_120a", ()),),
    args=(
        "Q_map",
        "K_map",
        "KSF_map",
        "V_map",
        "O",
        "q_scale",
        "k_scale",
        "v_scale",
        "q_sf",
        "unit_table",
        "total_units",
        "num_heads",
        "num_kblocks",
        "softmax_scale_log2",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Q_map",
        "K_map",
        "KSF_map",
        "V_map",
        "O",
        "q_scale",
        "k_scale",
        "v_scale",
        "q_sf",
        "unit_table",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=60416,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    KV_STATS,
    QUANTIZE,
    QUANTIZE_MB3,
    ATTENTION_BASE,
    ATTENTION_SHORT,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "KV_STATS",
    "QUANTIZE",
    "QUANTIZE_MB3",
    "ATTENTION_BASE",
    "ATTENTION_SHORT",
    "PROGRAMS",
    "aot_specs",
]
