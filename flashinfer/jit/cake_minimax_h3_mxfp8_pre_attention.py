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
"""Generated programs of the MiniMax-H3 MXFP8 pre-attention: the RMSNorm + AdaLN + MXFP8
quantize stage (NORM) and the fused MXFP8 QKV GEMM with the Q/K RMSNorm, the RoPE and
the destination-major MXFP8 pack in its epilogue (FUSED), both for sm_100a and sm_103a,
and the fused GEMM whose epilogue takes its TMEM lane address per tile
(FUSED_TILE_LANE), for sm_103a. The host module
``flashinfer.diffusion_ops.cake_minimax_h3_mxfp8`` computes the grids and launches them.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

NORM = cake_program.CakeProgram(
    name="cake_minimax_h3_mxfp8_pre_attention_0ca9783189af24b14501",
    operator_dir="cake_minimax_h3_mxfp8_pre_attention",
    kernel="kernel_cake_minimax_h3_mxfp8_pre_attention_0ca9783189af24b14501",
    sources=(
        "cake_minimax_h3_mxfp8_pre_attention_0ca9783189af24b14501_kernel.cu",
        "cake_minimax_h3_mxfp8_pre_attention_0ca9783189af24b14501_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "x",
        "x_norm_weight",
        "adaln_scale",
        "adaln_shift",
        "adaln_index",
        "activation_q",
        "activation_sf",
        "eps",
        "M",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "x",
        "x_norm_weight",
        "adaln_scale",
        "adaln_shift",
        "adaln_index",
        "activation_q",
        "activation_sf",
    ),
    block=(672, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=128,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

FUSED = cake_program.CakeProgram(
    name="cake_minimax_h3_mxfp8_pre_attention_d6abb58e32252678be5c",
    operator_dir="cake_minimax_h3_mxfp8_pre_attention",
    kernel="kernel_cake_minimax_h3_mxfp8_pre_attention_d6abb58e32252678be5c",
    sources=(
        "cake_minimax_h3_mxfp8_pre_attention_d6abb58e32252678be5c_kernel.cu",
        "cake_minimax_h3_mxfp8_pre_attention_d6abb58e32252678be5c_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "A",
        "B",
        "SFA",
        "SFB",
        "q_norm_weight",
        "k_norm_weight",
        "rope_cos_sin",
        "out_sf",
        "OUTQ",
        "qkv_words",
        "debug_q_words",
        "debug_k_words",
        "write_debug",
        "eps",
        "M",
        "m_tiles",
        "HEADS_PER_DESTINATION",
        "ROWS_PER_DESTINATION",
        "SCALE_STRIDE",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "A",
        "B",
        "SFA",
        "SFB",
        "q_norm_weight",
        "k_norm_weight",
        "rope_cos_sin",
        "out_sf",
        "OUTQ",
        "qkv_words",
        "debug_q_words",
        "debug_k_words",
    ),
    block=(320, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=208000,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

FUSED_TILE_LANE = cake_program.CakeProgram(
    name="cake_minimax_h3_mxfp8_pre_attention_5b7b585cc94b67ac31fb",
    operator_dir="cake_minimax_h3_mxfp8_pre_attention",
    kernel="kernel_cake_minimax_h3_mxfp8_pre_attention_5b7b585cc94b67ac31fb",
    sources=(
        "cake_minimax_h3_mxfp8_pre_attention_5b7b585cc94b67ac31fb_kernel.cu",
        "cake_minimax_h3_mxfp8_pre_attention_5b7b585cc94b67ac31fb_binding.cu",
    ),
    targets=("sm_103a",),
    compile_options=(("sm_103a", ("--use_fast_math", "--fmad=true")),),
    args=(
        "A",
        "B",
        "SFA",
        "SFB",
        "q_norm_weight",
        "k_norm_weight",
        "rope_cos_sin",
        "out_sf",
        "OUTQ",
        "qkv_words",
        "debug_q_words",
        "debug_k_words",
        "write_debug",
        "eps",
        "M",
        "m_tiles",
        "HEADS_PER_DESTINATION",
        "ROWS_PER_DESTINATION",
        "SCALE_STRIDE",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "A",
        "B",
        "SFA",
        "SFB",
        "q_norm_weight",
        "k_norm_weight",
        "rope_cos_sin",
        "out_sf",
        "OUTQ",
        "qkv_words",
        "debug_q_words",
        "debug_k_words",
    ),
    block=(320, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=208000,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (NORM, FUSED, FUSED_TILE_LANE)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = ["NORM", "FUSED", "FUSED_TILE_LANE", "PROGRAMS", "aot_specs"]
