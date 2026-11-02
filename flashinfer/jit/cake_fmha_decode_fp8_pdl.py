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
"""Generated split-KV reducer of the E4M3 paged decode of the cake_fmha module: it merges
the eight FP32 partials of every (request, query head) that a fixed-plan decode program
of cake_fmha_decode_fp8 stored and writes the E4M3 output, launched with programmatic
dependent launch right after that program. The host module ``flashinfer.cake_fmha``
launches it after the decode program it pairs with; other launches keep the module's
other decode routes.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

REDUCE_S8 = cake_program.CakeProgram(
    name="cake_fmha_decode_fp8_pdl_8cb30ca58dc96a96e10c",
    operator_dir="cake_fmha_decode_fp8_pdl",
    kernel="kernel_cake_fmha_decode_fp8_pdl_8cb30ca58dc96a96e10c",
    sources=(
        "cake_fmha_decode_fp8_pdl_8cb30ca58dc96a96e10c_kernel.cu",
        "cake_fmha_decode_fp8_pdl_8cb30ca58dc96a96e10c_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "partial_O",
        "partial_max",
        "partial_sum",
        "O",
        "bmm2_scale_ptr",
        "num_split",
        "bmm2_scalar",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("partial_O", "partial_max", "partial_sum", "O", "bmm2_scale_ptr"),
    block=(32, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (REDUCE_S8,)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = ["REDUCE_S8", "PROGRAMS", "aot_specs"]
