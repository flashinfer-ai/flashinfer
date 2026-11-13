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
"""Generated programs of the Blackwell MLA K concatenation (each head's k_nope row and the
token's broadcast k_rope row copied byte-exact into the caller-owned K buffer through
byte strides): the one-CTA-per-window program for sm_100a, sm_103a and sm_107a and the
slot-split program for sm_107a, where several CTAs share one token window. The host
module ``flashinfer.concat_ops`` validates the tensors, selects the program and the grid
and launches it.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

ONE_CTA = cake_program.CakeProgram(
    name="cake_concat_mla_k_2f4712b0770953f7b9ae",
    operator_dir="cake_concat_mla_k",
    kernel="kernel_cake_concat_mla_k_2f4712b0770953f7b9ae",
    sources=(
        "cake_concat_mla_k_2f4712b0770953f7b9ae_kernel.cu",
        "cake_concat_mla_k_2f4712b0770953f7b9ae_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--fmad=true",)),
        ("sm_103a", ("--fmad=true",)),
        ("sm_107a", ()),
    ),
    args=(
        "k",
        "k_nope",
        "k_rope",
        "element_bytes",
        "k_stride_0_bytes",
        "k_stride_1_bytes",
        "k_nope_stride_0_bytes",
        "k_nope_stride_1_bytes",
        "k_rope_stride_0_bytes",
        "tokens",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("k", "k_nope", "k_rope"),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

SPLIT = cake_program.CakeProgram(
    name="cake_concat_mla_k_1e22ce3ebf0f18dbe15f",
    operator_dir="cake_concat_mla_k",
    kernel="kernel_cake_concat_mla_k_1e22ce3ebf0f18dbe15f",
    sources=(
        "cake_concat_mla_k_1e22ce3ebf0f18dbe15f_kernel.cu",
        "cake_concat_mla_k_1e22ce3ebf0f18dbe15f_binding.cu",
    ),
    targets=("sm_107a",),
    compile_options=(("sm_107a", ()),),
    args=(
        "k",
        "k_nope",
        "k_rope",
        "element_bytes",
        "k_stride_0_bytes",
        "k_stride_1_bytes",
        "k_nope_stride_0_bytes",
        "k_nope_stride_1_bytes",
        "k_rope_stride_0_bytes",
        "tokens",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("k", "k_nope", "k_rope"),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (ONE_CTA, SPLIT)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = ["ONE_CTA", "SPLIT", "PROGRAMS", "aot_specs"]
