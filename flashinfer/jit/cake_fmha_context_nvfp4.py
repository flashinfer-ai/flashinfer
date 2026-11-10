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
"""Generated programs of the paged context attention of the cake_fmha module over NVFP4 K
and V pages (E4M3 Q and O, head dim 128, bottom-right causal, the GQA group packed into
the M tile), one per target and scale ABI: Q511_SM100A / Q511_SM103A take the softmax
and output scales as host scalars, the SCALE_PTR programs read them from device scalars.
The host module ``flashinfer.cake_fmha`` selects the program from the head counts, the
page size, the planned maximum lengths and the kind of scale arguments of a launch,
computes the grid and launches it; other shapes keep the module's other context routes.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

Q511_SM100A = cake_program.CakeProgram(
    name="cake_fmha_context_nvfp4_41e0f7b224fcc4acfc54",
    operator_dir="cake_fmha_context_nvfp4",
    kernel="kernel_cake_fmha_context_nvfp4_41e0f7b224fcc4acfc54",
    sources=(
        "cake_fmha_context_nvfp4_41e0f7b224fcc4acfc54_kernel.cu",
        "cake_fmha_context_nvfp4_41e0f7b224fcc4acfc54_binding.cu",
    ),
    targets=("sm_100a",),
    compile_options=(("sm_100a", ("--use_fast_math", "--fmad=true")),),
    args=(
        "Q",
        "Kp",
        "Vp",
        "Ksf",
        "Vsf",
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
        "Kp",
        "Vp",
        "Ksf",
        "Vsf",
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
    dynamic_smem_bytes=104960,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

Q511_SM103A = cake_program.CakeProgram(
    name="cake_fmha_context_nvfp4_c916619510183749b162",
    operator_dir="cake_fmha_context_nvfp4",
    kernel="kernel_cake_fmha_context_nvfp4_c916619510183749b162",
    sources=(
        "cake_fmha_context_nvfp4_c916619510183749b162_kernel.cu",
        "cake_fmha_context_nvfp4_c916619510183749b162_binding.cu",
    ),
    targets=("sm_103a",),
    compile_options=(("sm_103a", ("--use_fast_math", "--fmad=true")),),
    args=(
        "Q",
        "Kp",
        "Vp",
        "Ksf",
        "Vsf",
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
        "Kp",
        "Vp",
        "Ksf",
        "Vsf",
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
    dynamic_smem_bytes=104960,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

Q511_SCALE_PTR_SM100A = cake_program.CakeProgram(
    name="cake_fmha_context_nvfp4_ab413a0149f11e7a8463",
    operator_dir="cake_fmha_context_nvfp4",
    kernel="kernel_cake_fmha_context_nvfp4_ab413a0149f11e7a8463",
    sources=(
        "cake_fmha_context_nvfp4_ab413a0149f11e7a8463_kernel.cu",
        "cake_fmha_context_nvfp4_ab413a0149f11e7a8463_binding.cu",
    ),
    targets=("sm_100a",),
    compile_options=(("sm_100a", ("--use_fast_math", "--fmad=true")),),
    args=(
        "Q",
        "Kp",
        "Vp",
        "Ksf",
        "Vsf",
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
        "Kp",
        "Vp",
        "Ksf",
        "Vsf",
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
    dynamic_smem_bytes=104960,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

Q511_SCALE_PTR_SM103A = cake_program.CakeProgram(
    name="cake_fmha_context_nvfp4_1b4750660638d644a971",
    operator_dir="cake_fmha_context_nvfp4",
    kernel="kernel_cake_fmha_context_nvfp4_1b4750660638d644a971",
    sources=(
        "cake_fmha_context_nvfp4_1b4750660638d644a971_kernel.cu",
        "cake_fmha_context_nvfp4_1b4750660638d644a971_binding.cu",
    ),
    targets=("sm_103a",),
    compile_options=(("sm_103a", ("--use_fast_math", "--fmad=true")),),
    args=(
        "Q",
        "Kp",
        "Vp",
        "Ksf",
        "Vsf",
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
        "Kp",
        "Vp",
        "Ksf",
        "Vsf",
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
    dynamic_smem_bytes=104960,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    Q511_SM100A,
    Q511_SM103A,
    Q511_SCALE_PTR_SM100A,
    Q511_SCALE_PTR_SM103A,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "Q511_SM100A",
    "Q511_SM103A",
    "Q511_SCALE_PTR_SM100A",
    "Q511_SCALE_PTR_SM103A",
    "PROGRAMS",
    "aot_specs",
]
