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
"""Generated programs of the MiniMax-H3 SM120 FP8 packed-varlen attention: the K/V
statistics with their per-segment fold into the K means and V scales (KV_STATS), the
E4M3 quantizer (QUANTIZE) and the attention launch in the unit-scaled block-scale MMA
form (ATTENTION), and the overlapped form of the plans in the overlap band, launched in
this order: the head-major pre-processing producers (PREPROCESS_HEAD_MAJOR, a plain
launch), the lead attention launch (ATTENTION_LEAD) and the tail attention launch
(ATTENTION_TAIL), each a programmatic dependent of the launch before it; each shipped
for sm_120a. An overlapped CTA waits only on CTAs of the launches before it in its call,
so a lazy module load of any of the three only delays the call. The host module
``flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention`` selects the
form, computes the grids and launches the programs behind
``minimax_h3_sm120_varlen_attention_fp8``.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

KV_STATS = cake_program.CakeProgram(
    name="cake_minimax_h3_sm120_quant_varlen_attention_ff7a4a68163ed9779075",
    operator_dir="cake_minimax_h3_sm120_quant_varlen_attention",
    kernel="kernel_cake_minimax_h3_sm120_quant_varlen_attention_ff7a4a68163ed9779075",
    sources=(
        "cake_minimax_h3_sm120_quant_varlen_attention_ff7a4a68163ed9779075_kernel.cu",
        "cake_minimax_h3_sm120_quant_varlen_attention_ff7a4a68163ed9779075_binding.cu",
    ),
    targets=("sm_120a",),
    compile_options=(("sm_120a", ()),),
    args=(
        "K",
        "V",
        "tile_table",
        "seg_begin",
        "seg_len",
        "seg_tile_begin",
        "partials",
        "counters",
        "mean_k",
        "v_scale",
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
        "seg_begin",
        "seg_len",
        "seg_tile_begin",
        "partials",
        "counters",
        "mean_k",
        "v_scale",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=16512,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

QUANTIZE = cake_program.CakeProgram(
    name="cake_minimax_h3_sm120_quant_varlen_attention_e2abeffeb2ef21daba83",
    operator_dir="cake_minimax_h3_sm120_quant_varlen_attention",
    kernel="kernel_cake_minimax_h3_sm120_quant_varlen_attention_e2abeffeb2ef21daba83",
    sources=(
        "cake_minimax_h3_sm120_quant_varlen_attention_e2abeffeb2ef21daba83_kernel.cu",
        "cake_minimax_h3_sm120_quant_varlen_attention_e2abeffeb2ef21daba83_binding.cu",
    ),
    targets=("sm_120a",),
    compile_options=(("sm_120a", ()),),
    args=(
        "Q",
        "K",
        "V",
        "cu_seqlens",
        "mean_k",
        "v_scale",
        "q8",
        "k8",
        "vt8",
        "q_scale",
        "k_scale",
        "total_tokens",
        "total_padded",
        "num_segments",
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
        "cu_seqlens",
        "mean_k",
        "v_scale",
        "q8",
        "k8",
        "vt8",
        "q_scale",
        "k_scale",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=128,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ATTENTION = cake_program.CakeProgram(
    name="cake_minimax_h3_sm120_quant_varlen_attention_2fcf8d095e20c53cdc4b",
    operator_dir="cake_minimax_h3_sm120_quant_varlen_attention",
    kernel="kernel_cake_minimax_h3_sm120_quant_varlen_attention_2fcf8d095e20c53cdc4b",
    sources=(
        "cake_minimax_h3_sm120_quant_varlen_attention_2fcf8d095e20c53cdc4b_kernel.cu",
        "cake_minimax_h3_sm120_quant_varlen_attention_2fcf8d095e20c53cdc4b_binding.cu",
    ),
    targets=("sm_120a",),
    compile_options=(("sm_120a", ()),),
    args=(
        "Q_map",
        "K_map",
        "V_map",
        "O",
        "q_scale",
        "k_scale",
        "v_scale",
        "seg_begin",
        "seg_len",
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
        "V_map",
        "O",
        "q_scale",
        "k_scale",
        "v_scale",
        "seg_begin",
        "seg_len",
        "unit_table",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=82944,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PREPROCESS_HEAD_MAJOR = cake_program.CakeProgram(
    name="cake_minimax_h3_sm120_quant_varlen_attention_bffafaaa13fe97200512",
    operator_dir="cake_minimax_h3_sm120_quant_varlen_attention",
    kernel="kernel_cake_minimax_h3_sm120_quant_varlen_attention_bffafaaa13fe97200512",
    sources=(
        "cake_minimax_h3_sm120_quant_varlen_attention_bffafaaa13fe97200512_kernel.cu",
        "cake_minimax_h3_sm120_quant_varlen_attention_bffafaaa13fe97200512_binding.cu",
    ),
    targets=("sm_120a",),
    compile_options=(("sm_120a", ()),),
    args=(
        "Q",
        "K",
        "V",
        "tile_table",
        "seg_begin",
        "seg_len",
        "seg_tile_begin",
        "cu_seqlens",
        "partials",
        "counters",
        "mean_k",
        "v_scale",
        "q8",
        "k8",
        "vt8",
        "q_scale",
        "k_scale",
        "hctr",
        "num_tiles",
        "total_tokens",
        "total_padded",
        "num_segments",
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
        "tile_table",
        "seg_begin",
        "seg_len",
        "seg_tile_begin",
        "cu_seqlens",
        "partials",
        "counters",
        "mean_k",
        "v_scale",
        "q8",
        "k8",
        "vt8",
        "q_scale",
        "k_scale",
        "hctr",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=33024,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ATTENTION_LEAD = cake_program.CakeProgram(
    name="cake_minimax_h3_sm120_quant_varlen_attention_d2be4103bb05de829bb5",
    operator_dir="cake_minimax_h3_sm120_quant_varlen_attention",
    kernel="kernel_cake_minimax_h3_sm120_quant_varlen_attention_d2be4103bb05de829bb5",
    sources=(
        "cake_minimax_h3_sm120_quant_varlen_attention_d2be4103bb05de829bb5_kernel.cu",
        "cake_minimax_h3_sm120_quant_varlen_attention_d2be4103bb05de829bb5_binding.cu",
    ),
    targets=("sm_120a",),
    compile_options=(("sm_120a", ()),),
    args=(
        "Q_map",
        "K_map",
        "V_map",
        "O",
        "q_scale",
        "k_scale",
        "v_scale",
        "seg_begin",
        "seg_len",
        "unit_table",
        "total_units",
        "num_heads",
        "num_kblocks",
        "softmax_scale_log2",
        "hctr",
        "lead_ctas",
        "tail_ctas",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Q_map",
        "K_map",
        "V_map",
        "O",
        "q_scale",
        "k_scale",
        "v_scale",
        "seg_begin",
        "seg_len",
        "unit_table",
        "hctr",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=82944,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

ATTENTION_TAIL = cake_program.CakeProgram(
    name="cake_minimax_h3_sm120_quant_varlen_attention_033fa67a100e827f1eb5",
    operator_dir="cake_minimax_h3_sm120_quant_varlen_attention",
    kernel="kernel_cake_minimax_h3_sm120_quant_varlen_attention_033fa67a100e827f1eb5",
    sources=(
        "cake_minimax_h3_sm120_quant_varlen_attention_033fa67a100e827f1eb5_kernel.cu",
        "cake_minimax_h3_sm120_quant_varlen_attention_033fa67a100e827f1eb5_binding.cu",
    ),
    targets=("sm_120a",),
    compile_options=(("sm_120a", ()),),
    args=(
        "Q_map",
        "K_map",
        "V_map",
        "O",
        "q_scale",
        "k_scale",
        "v_scale",
        "seg_begin",
        "seg_len",
        "unit_table",
        "total_units",
        "num_heads",
        "num_kblocks",
        "softmax_scale_log2",
        "hctr",
        "lead_ctas",
        "tail_ctas",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Q_map",
        "K_map",
        "V_map",
        "O",
        "q_scale",
        "k_scale",
        "v_scale",
        "seg_begin",
        "seg_len",
        "unit_table",
        "hctr",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=82944,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    KV_STATS,
    QUANTIZE,
    ATTENTION,
    PREPROCESS_HEAD_MAJOR,
    ATTENTION_LEAD,
    ATTENTION_TAIL,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "KV_STATS",
    "QUANTIZE",
    "ATTENTION",
    "PREPROCESS_HEAD_MAJOR",
    "ATTENTION_LEAD",
    "ATTENTION_TAIL",
    "PROGRAMS",
    "aot_specs",
]
