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
"""Generated programs of the aligned head-dim-128 single-query paged GQA decode of the
cake_fmha module: one program per cluster size (8, 4, 2 or 1 CTAs per (request, KV
head)) and KV-cache dtype (bfloat16, float16), each shipped for sm_100a and sm_103a. The
host module ``flashinfer.cake_fmha`` selects the program from the request count, the KV-
head count and the maximum sequence length of a launch, computes the grid (requests x KV
heads, 1, cluster size) and launches it; shapes outside the selector's bands keep the
module's other decode routes.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

ALIGNED_C8_BF16 = cake_program.CakeProgram(
    name="cake_fmha_aligned_a9cc814b948505a2cc1b",
    operator_dir="cake_fmha_aligned",
    kernel="kernel_cake_fmha_aligned_a9cc814b948505a2cc1b",
    sources=(
        "cake_fmha_aligned_a9cc814b948505a2cc1b_kernel.cu",
        "cake_fmha_aligned_a9cc814b948505a2cc1b_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "uniform_seq_len",
        "max_pages_per_seq",
        "softmax_scale_log2",
        "num_q_heads",
        "num_kv_heads",
        "batch_size",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("Qt", "K", "V", "O", "page_table", "seq_lens_kv"),
    block=(256, 1, 1),
    cluster=(1, 1, 8),
    dynamic_smem_bytes=206720,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ALIGNED_C4_BF16 = cake_program.CakeProgram(
    name="cake_fmha_aligned_2bd88c52bb7a16a1814f",
    operator_dir="cake_fmha_aligned",
    kernel="kernel_cake_fmha_aligned_2bd88c52bb7a16a1814f",
    sources=(
        "cake_fmha_aligned_2bd88c52bb7a16a1814f_kernel.cu",
        "cake_fmha_aligned_2bd88c52bb7a16a1814f_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "uniform_seq_len",
        "max_pages_per_seq",
        "softmax_scale_log2",
        "num_q_heads",
        "num_kv_heads",
        "batch_size",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("Qt", "K", "V", "O", "page_table", "seq_lens_kv"),
    block=(256, 1, 1),
    cluster=(1, 1, 4),
    dynamic_smem_bytes=206720,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ALIGNED_C2_BF16 = cake_program.CakeProgram(
    name="cake_fmha_aligned_1052b6ae93d516e78ed6",
    operator_dir="cake_fmha_aligned",
    kernel="kernel_cake_fmha_aligned_1052b6ae93d516e78ed6",
    sources=(
        "cake_fmha_aligned_1052b6ae93d516e78ed6_kernel.cu",
        "cake_fmha_aligned_1052b6ae93d516e78ed6_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "uniform_seq_len",
        "max_pages_per_seq",
        "softmax_scale_log2",
        "num_q_heads",
        "num_kv_heads",
        "batch_size",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("Qt", "K", "V", "O", "page_table", "seq_lens_kv"),
    block=(256, 1, 1),
    cluster=(1, 1, 2),
    dynamic_smem_bytes=206720,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ALIGNED_C1_BF16 = cake_program.CakeProgram(
    name="cake_fmha_aligned_6f88390dacdd946afa74",
    operator_dir="cake_fmha_aligned",
    kernel="kernel_cake_fmha_aligned_6f88390dacdd946afa74",
    sources=(
        "cake_fmha_aligned_6f88390dacdd946afa74_kernel.cu",
        "cake_fmha_aligned_6f88390dacdd946afa74_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "uniform_seq_len",
        "max_pages_per_seq",
        "softmax_scale_log2",
        "num_q_heads",
        "num_kv_heads",
        "batch_size",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("Qt", "K", "V", "O", "page_table", "seq_lens_kv"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=206720,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ALIGNED_C8_FP16 = cake_program.CakeProgram(
    name="cake_fmha_aligned_8d67475a5eaad8c4abc2",
    operator_dir="cake_fmha_aligned",
    kernel="kernel_cake_fmha_aligned_8d67475a5eaad8c4abc2",
    sources=(
        "cake_fmha_aligned_8d67475a5eaad8c4abc2_kernel.cu",
        "cake_fmha_aligned_8d67475a5eaad8c4abc2_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "uniform_seq_len",
        "max_pages_per_seq",
        "softmax_scale_log2",
        "num_q_heads",
        "num_kv_heads",
        "batch_size",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("Qt", "K", "V", "O", "page_table", "seq_lens_kv"),
    block=(256, 1, 1),
    cluster=(1, 1, 8),
    dynamic_smem_bytes=206720,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ALIGNED_C4_FP16 = cake_program.CakeProgram(
    name="cake_fmha_aligned_b100f226ebd9ce6884cc",
    operator_dir="cake_fmha_aligned",
    kernel="kernel_cake_fmha_aligned_b100f226ebd9ce6884cc",
    sources=(
        "cake_fmha_aligned_b100f226ebd9ce6884cc_kernel.cu",
        "cake_fmha_aligned_b100f226ebd9ce6884cc_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "uniform_seq_len",
        "max_pages_per_seq",
        "softmax_scale_log2",
        "num_q_heads",
        "num_kv_heads",
        "batch_size",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("Qt", "K", "V", "O", "page_table", "seq_lens_kv"),
    block=(256, 1, 1),
    cluster=(1, 1, 4),
    dynamic_smem_bytes=206720,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ALIGNED_C2_FP16 = cake_program.CakeProgram(
    name="cake_fmha_aligned_6b02ec3e27916ce5eee5",
    operator_dir="cake_fmha_aligned",
    kernel="kernel_cake_fmha_aligned_6b02ec3e27916ce5eee5",
    sources=(
        "cake_fmha_aligned_6b02ec3e27916ce5eee5_kernel.cu",
        "cake_fmha_aligned_6b02ec3e27916ce5eee5_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "uniform_seq_len",
        "max_pages_per_seq",
        "softmax_scale_log2",
        "num_q_heads",
        "num_kv_heads",
        "batch_size",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("Qt", "K", "V", "O", "page_table", "seq_lens_kv"),
    block=(256, 1, 1),
    cluster=(1, 1, 2),
    dynamic_smem_bytes=206720,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ALIGNED_C1_FP16 = cake_program.CakeProgram(
    name="cake_fmha_aligned_300f2001dc3a9f642eeb",
    operator_dir="cake_fmha_aligned",
    kernel="kernel_cake_fmha_aligned_300f2001dc3a9f642eeb",
    sources=(
        "cake_fmha_aligned_300f2001dc3a9f642eeb_kernel.cu",
        "cake_fmha_aligned_300f2001dc3a9f642eeb_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "Qt",
        "K",
        "V",
        "O",
        "page_table",
        "seq_lens_kv",
        "uniform_seq_len",
        "max_pages_per_seq",
        "softmax_scale_log2",
        "num_q_heads",
        "num_kv_heads",
        "batch_size",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("Qt", "K", "V", "O", "page_table", "seq_lens_kv"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=206720,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    ALIGNED_C8_BF16,
    ALIGNED_C4_BF16,
    ALIGNED_C2_BF16,
    ALIGNED_C1_BF16,
    ALIGNED_C8_FP16,
    ALIGNED_C4_FP16,
    ALIGNED_C2_FP16,
    ALIGNED_C1_FP16,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "ALIGNED_C8_BF16",
    "ALIGNED_C4_BF16",
    "ALIGNED_C2_BF16",
    "ALIGNED_C1_BF16",
    "ALIGNED_C8_FP16",
    "ALIGNED_C4_FP16",
    "ALIGNED_C2_FP16",
    "ALIGNED_C1_FP16",
    "PROGRAMS",
    "aot_specs",
]
