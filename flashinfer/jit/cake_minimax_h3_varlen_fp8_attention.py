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
"""Generated programs of the MiniMax-H3 all-FP8 packed-varlen attention: the general
program (ATTENTION) and its tiny-plan build (ATTENTION_TINY, bound when every segment
has at most 32 tokens), each shipped for sm_100a and sm_103a. The host module
``flashinfer.experimental.minimax_h3_varlen_attention.cake_backend`` binds them as the
``variant="fp8"`` attention stage over the BF16 planner's unit table and launches the
BF16 route's combine for the K/V-split partials.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

ATTENTION = cake_program.CakeProgram(
    name="cake_minimax_h3_varlen_fp8_attention_7fad085e4e9ddbed0a98",
    operator_dir="cake_minimax_h3_varlen_fp8_attention",
    kernel="kernel_cake_minimax_h3_varlen_fp8_attention_7fad085e4e9ddbed0a98",
    sources=(
        "cake_minimax_h3_varlen_fp8_attention_7fad085e4e9ddbed0a98_kernel.cu",
        "cake_minimax_h3_varlen_fp8_attention_7fad085e4e9ddbed0a98_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "Q",
        "K",
        "V",
        "O",
        "O_raw",
        "unit_table",
        "partial_O",
        "partial_ML",
        "q_descale",
        "k_descale",
        "v_descale",
        "total_tiles",
        "num_heads",
        "softmax_scale_log2",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Q",
        "K",
        "V",
        "O",
        "O_raw",
        "unit_table",
        "partial_O",
        "partial_ML",
        "q_descale",
        "k_descale",
        "v_descale",
    ),
    block=(512, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=168960,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ATTENTION_TINY = cake_program.CakeProgram(
    name="cake_minimax_h3_varlen_fp8_attention_529a83afa416443c4fb4",
    operator_dir="cake_minimax_h3_varlen_fp8_attention",
    kernel="kernel_cake_minimax_h3_varlen_fp8_attention_529a83afa416443c4fb4",
    sources=(
        "cake_minimax_h3_varlen_fp8_attention_529a83afa416443c4fb4_kernel.cu",
        "cake_minimax_h3_varlen_fp8_attention_529a83afa416443c4fb4_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "Q",
        "K",
        "V",
        "O",
        "O_raw",
        "unit_table",
        "partial_O",
        "partial_ML",
        "q_descale",
        "k_descale",
        "v_descale",
        "total_tiles",
        "num_heads",
        "softmax_scale_log2",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Q",
        "K",
        "V",
        "O",
        "O_raw",
        "unit_table",
        "partial_O",
        "partial_ML",
        "q_descale",
        "k_descale",
        "v_descale",
    ),
    block=(512, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=168960,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (ATTENTION, ATTENTION_TINY)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = ["ATTENTION", "ATTENTION_TINY", "PROGRAMS", "aot_specs"]
