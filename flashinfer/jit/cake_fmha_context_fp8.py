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
"""Generated programs of the FP8 E4M3 paged context attention of the cake_fmha module (head
dim 128, bottom-right causal, the GQA group packed into the M tile), each shipped for
sm_100a and sm_103a: B256 takes the softmax and output scales as host scalars,
B256_SCALE_PTR reads them from device scalars. The host module ``flashinfer.cake_fmha``
selects the program from the head counts, the page size, the planned maximum lengths and
the kind of scale arguments of a launch, computes the grid and launches it; other shapes
keep the module's other context routes.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

B256 = cake_program.CakeProgram(
    name="cake_fmha_context_fp8_6f7e8b77c2309a9c63a2",
    operator_dir="cake_fmha_context_fp8",
    kernel="kernel_cake_fmha_context_fp8_6f7e8b77c2309a9c63a2",
    sources=(
        "cake_fmha_context_fp8_6f7e8b77c2309a9c63a2_kernel.cu",
        "cake_fmha_context_fp8_6f7e8b77c2309a9c63a2_binding.cu",
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
        "O_ptr",
        "LSE_ptr",
        "sinks",
        "page_table_k",
        "page_table_v",
        "seq_lens_q",
        "seq_lens_kv",
        "cu_seq_lens_q",
        "softmax_scale_log2",
        "output_scale",
        "total_bh",
        "page_row_stride",
        "num_ctas",
        "dynamic_counter",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
        "bmm1_is_log2",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Q",
        "K",
        "V",
        "O_ptr",
        "LSE_ptr",
        "sinks",
        "page_table_k",
        "page_table_v",
        "seq_lens_q",
        "seq_lens_kv",
        "cu_seq_lens_q",
        "dynamic_counter",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=86528,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

B256_SCALE_PTR = cake_program.CakeProgram(
    name="cake_fmha_context_fp8_a586c83f6eca2799c39e",
    operator_dir="cake_fmha_context_fp8",
    kernel="kernel_cake_fmha_context_fp8_a586c83f6eca2799c39e",
    sources=(
        "cake_fmha_context_fp8_a586c83f6eca2799c39e_kernel.cu",
        "cake_fmha_context_fp8_a586c83f6eca2799c39e_binding.cu",
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
        "O_ptr",
        "LSE_ptr",
        "sinks",
        "page_table_k",
        "page_table_v",
        "seq_lens_q",
        "seq_lens_kv",
        "cu_seq_lens_q",
        "softmax_scale_log2",
        "output_scale",
        "total_bh",
        "page_row_stride",
        "num_ctas",
        "dynamic_counter",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
        "bmm1_is_log2",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Q",
        "K",
        "V",
        "O_ptr",
        "LSE_ptr",
        "sinks",
        "page_table_k",
        "page_table_v",
        "seq_lens_q",
        "seq_lens_kv",
        "cu_seq_lens_q",
        "dynamic_counter",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=86528,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (B256, B256_SCALE_PTR)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = ["B256", "B256_SCALE_PTR", "PROGRAMS", "aot_specs"]
