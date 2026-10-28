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
"""Generated programs of the BF16 paged context attention of the cake_fmha module (head dim
128, bottom-right causal, the GQA group packed into the M tile): the persistent programs
Q8192 and B128 for sm_100a and sm_103a, and the one-tile programs Q511 and Q257, one per
target. The host module ``flashinfer.cake_fmha`` selects the program from the head
counts, the page size and the planned maximum lengths of a launch, computes the grid and
launches it; other shapes keep the module's other context routes.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

Q8192 = cake_program.CakeProgram(
    name="cake_fmha_context_bf16_eb5b942877557703876d",
    operator_dir="cake_fmha_context_bf16",
    kernel="kernel_cake_fmha_context_bf16_eb5b942877557703876d",
    sources=(
        "cake_fmha_context_bf16_eb5b942877557703876d_kernel.cu",
        "cake_fmha_context_bf16_eb5b942877557703876d_binding.cu",
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
        "total_bh",
        "page_row_stride",
        "num_ctas",
        "dynamic_counter",
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
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=201216,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

B128 = cake_program.CakeProgram(
    name="cake_fmha_context_bf16_31373706c9fcb1a2b1db",
    operator_dir="cake_fmha_context_bf16",
    kernel="kernel_cake_fmha_context_bf16_31373706c9fcb1a2b1db",
    sources=(
        "cake_fmha_context_bf16_31373706c9fcb1a2b1db_kernel.cu",
        "cake_fmha_context_bf16_31373706c9fcb1a2b1db_binding.cu",
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
        "total_bh",
        "page_row_stride",
        "num_ctas",
        "dynamic_counter",
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
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=201216,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

Q511_SM100A = cake_program.CakeProgram(
    name="cake_fmha_context_bf16_fe2dfd2eb24d75a1bccc",
    operator_dir="cake_fmha_context_bf16",
    kernel="kernel_cake_fmha_context_bf16_fe2dfd2eb24d75a1bccc",
    sources=(
        "cake_fmha_context_bf16_fe2dfd2eb24d75a1bccc_kernel.cu",
        "cake_fmha_context_bf16_fe2dfd2eb24d75a1bccc_binding.cu",
    ),
    targets=("sm_100a",),
    compile_options=(("sm_100a", ("--use_fast_math", "--fmad=true")),),
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
        "total_bh",
        "page_row_stride",
        "num_ctas",
        "dynamic_counter",
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
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=168448,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

Q511_SM103A = cake_program.CakeProgram(
    name="cake_fmha_context_bf16_8bd7d679916961178c03",
    operator_dir="cake_fmha_context_bf16",
    kernel="kernel_cake_fmha_context_bf16_8bd7d679916961178c03",
    sources=(
        "cake_fmha_context_bf16_8bd7d679916961178c03_kernel.cu",
        "cake_fmha_context_bf16_8bd7d679916961178c03_binding.cu",
    ),
    targets=("sm_103a",),
    compile_options=(("sm_103a", ("--use_fast_math", "--fmad=true")),),
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
        "total_bh",
        "page_row_stride",
        "num_ctas",
        "dynamic_counter",
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
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=168448,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

Q257_SM100A = cake_program.CakeProgram(
    name="cake_fmha_context_bf16_d400e81a0185922f0097",
    operator_dir="cake_fmha_context_bf16",
    kernel="kernel_cake_fmha_context_bf16_d400e81a0185922f0097",
    sources=(
        "cake_fmha_context_bf16_d400e81a0185922f0097_kernel.cu",
        "cake_fmha_context_bf16_d400e81a0185922f0097_binding.cu",
    ),
    targets=("sm_100a",),
    compile_options=(("sm_100a", ("--use_fast_math", "--fmad=true")),),
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
        "total_bh",
        "page_row_stride",
        "num_ctas",
        "dynamic_counter",
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
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=168448,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

Q257_SM103A = cake_program.CakeProgram(
    name="cake_fmha_context_bf16_5fa928327be121e0c987",
    operator_dir="cake_fmha_context_bf16",
    kernel="kernel_cake_fmha_context_bf16_5fa928327be121e0c987",
    sources=(
        "cake_fmha_context_bf16_5fa928327be121e0c987_kernel.cu",
        "cake_fmha_context_bf16_5fa928327be121e0c987_binding.cu",
    ),
    targets=("sm_103a",),
    compile_options=(("sm_103a", ("--use_fast_math", "--fmad=true")),),
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
        "total_bh",
        "page_row_stride",
        "num_ctas",
        "dynamic_counter",
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
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=168448,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    Q8192,
    B128,
    Q511_SM100A,
    Q511_SM103A,
    Q257_SM100A,
    Q257_SM103A,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "Q8192",
    "B128",
    "Q511_SM100A",
    "Q511_SM103A",
    "Q257_SM100A",
    "Q257_SM103A",
    "PROGRAMS",
    "aot_specs",
]
