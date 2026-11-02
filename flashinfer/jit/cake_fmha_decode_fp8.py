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
"""Generated programs of the E4M3 paged decode of the cake_fmha module (E4M3 query, K/V
pages and output, head dim 128, HND pages, 32 query heads over 4 KV heads), one per
target and shape: 256 requests over 16-token pages for a launch the host plans as one
split, and 4 requests over 32-token pages built for the fixed split plan of eight splits
of sixteen 128-token blocks, whose partials the split-KV reducer of
cake_fmha_decode_fp8_pdl merges. The host module ``flashinfer.cake_fmha`` selects the
program from the batch, the head counts, the page size, the split plan and enable_pdl of
a launch, computes the grid and launches it; other shapes keep the module's other decode
routes.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

B256_SM100A = cake_program.CakeProgram(
    name="cake_fmha_decode_fp8_768aa719bcc0c8dc6de5",
    operator_dir="cake_fmha_decode_fp8",
    kernel="kernel_cake_fmha_decode_fp8_768aa719bcc0c8dc6de5",
    sources=(
        "cake_fmha_decode_fp8_768aa719bcc0c8dc6de5_kernel.cu",
        "cake_fmha_decode_fp8_768aa719bcc0c8dc6de5_binding.cu",
    ),
    targets=("sm_100a",),
    compile_options=(("sm_100a", ("--use_fast_math", "--fmad=true")),),
    args=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
        "partial_O",
        "partial_max",
        "partial_sum",
        "pt_batch_stride",
        "pt_v_offset",
        "bmm1_is_log2",
        "num_splits",
        "blocks_per_split",
        "bmm1_scalar",
        "bmm2_scalar",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
        "partial_O",
        "partial_max",
        "partial_sum",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=157952,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

B256_SM103A = cake_program.CakeProgram(
    name="cake_fmha_decode_fp8_2bf8c20db84aa7b45230",
    operator_dir="cake_fmha_decode_fp8",
    kernel="kernel_cake_fmha_decode_fp8_2bf8c20db84aa7b45230",
    sources=(
        "cake_fmha_decode_fp8_2bf8c20db84aa7b45230_kernel.cu",
        "cake_fmha_decode_fp8_2bf8c20db84aa7b45230_binding.cu",
    ),
    targets=("sm_103a",),
    compile_options=(("sm_103a", ("--use_fast_math", "--fmad=true")),),
    args=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
        "partial_O",
        "partial_max",
        "partial_sum",
        "pt_batch_stride",
        "pt_v_offset",
        "bmm1_is_log2",
        "num_splits",
        "blocks_per_split",
        "bmm1_scalar",
        "bmm2_scalar",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
        "partial_O",
        "partial_max",
        "partial_sum",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=157952,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

B4_S8X16_SM100A = cake_program.CakeProgram(
    name="cake_fmha_decode_fp8_56920e5612bc123d66c9",
    operator_dir="cake_fmha_decode_fp8",
    kernel="kernel_cake_fmha_decode_fp8_56920e5612bc123d66c9",
    sources=(
        "cake_fmha_decode_fp8_56920e5612bc123d66c9_kernel.cu",
        "cake_fmha_decode_fp8_56920e5612bc123d66c9_binding.cu",
    ),
    targets=("sm_100a",),
    compile_options=(("sm_100a", ("--use_fast_math", "--fmad=true")),),
    args=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
        "partial_O",
        "partial_max",
        "partial_sum",
        "pt_batch_stride",
        "pt_v_offset",
        "bmm1_is_log2",
        "num_splits",
        "blocks_per_split",
        "bmm1_scalar",
        "bmm2_scalar",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
        "partial_O",
        "partial_max",
        "partial_sum",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=157952,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

B4_S8X16_SM103A = cake_program.CakeProgram(
    name="cake_fmha_decode_fp8_4a15e1e5d592f3ec8833",
    operator_dir="cake_fmha_decode_fp8",
    kernel="kernel_cake_fmha_decode_fp8_4a15e1e5d592f3ec8833",
    sources=(
        "cake_fmha_decode_fp8_4a15e1e5d592f3ec8833_kernel.cu",
        "cake_fmha_decode_fp8_4a15e1e5d592f3ec8833_binding.cu",
    ),
    targets=("sm_103a",),
    compile_options=(("sm_103a", ("--use_fast_math", "--fmad=true")),),
    args=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
        "partial_O",
        "partial_max",
        "partial_sum",
        "pt_batch_stride",
        "pt_v_offset",
        "bmm1_is_log2",
        "num_splits",
        "blocks_per_split",
        "bmm1_scalar",
        "bmm2_scalar",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
        "partial_O",
        "partial_max",
        "partial_sum",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=157952,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    B256_SM100A,
    B256_SM103A,
    B4_S8X16_SM100A,
    B4_S8X16_SM103A,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "B256_SM100A",
    "B256_SM103A",
    "B4_S8X16_SM100A",
    "B4_S8X16_SM103A",
    "PROGRAMS",
    "aot_specs",
]
