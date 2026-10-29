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
"""Generated programs of the BF16 paged decode of the cake_fmha module (head dim 128,
16-token HND pages, eight query heads per KV head): the stream-K 128-request program,
the four-request program with 4-CTA clusters and the 256-request attention-sink program,
each with host scales and with FP32 device scales, shipped for sm_100a and sm_103a. The
host module ``flashinfer.cake_fmha`` selects the program from the batch, the head
counts, the page size, the sink, the scale form and the maximum sequence length of a
launch, computes the grid and launches it; other shapes keep the module's other decode
routes.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

DEC_B128 = cake_program.CakeProgram(
    name="cake_fmha_decode_bf16_4a8ccd779ab91c5d2cb5",
    operator_dir="cake_fmha_decode_bf16",
    kernel="kernel_cake_fmha_decode_bf16_4a8ccd779ab91c5d2cb5",
    sources=(
        "cake_fmha_decode_bf16_4a8ccd779ab91c5d2cb5_kernel.cu",
        "cake_fmha_decode_bf16_4a8ccd779ab91c5d2cb5_binding.cu",
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
        "O_ptr",
        "LSE_ptr",
        "page_table",
        "causal_seqlens_kv_global",
        "scale_log2_ptr",
        "bmm2_ptr",
        "sinks_ptr",
        "split_counter",
        "split_ws",
        "max_pages_per_seq",
        "max_local_seq_len",
        "softmax_scale_log2",
        "window_left",
        "num_q_heads",
        "num_kv_heads",
        "batch_size",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Qt",
        "K",
        "V",
        "O_ptr",
        "LSE_ptr",
        "page_table",
        "causal_seqlens_kv_global",
        "scale_log2_ptr",
        "bmm2_ptr",
        "sinks_ptr",
        "split_counter",
        "split_ws",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=210944,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

DEC_B128_DS = cake_program.CakeProgram(
    name="cake_fmha_decode_bf16_4771b35b3b41de52fc15",
    operator_dir="cake_fmha_decode_bf16",
    kernel="kernel_cake_fmha_decode_bf16_4771b35b3b41de52fc15",
    sources=(
        "cake_fmha_decode_bf16_4771b35b3b41de52fc15_kernel.cu",
        "cake_fmha_decode_bf16_4771b35b3b41de52fc15_binding.cu",
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
        "O_ptr",
        "LSE_ptr",
        "page_table",
        "causal_seqlens_kv_global",
        "scale_log2_ptr",
        "bmm2_ptr",
        "sinks_ptr",
        "split_counter",
        "split_ws",
        "max_pages_per_seq",
        "max_local_seq_len",
        "softmax_scale_log2",
        "window_left",
        "num_q_heads",
        "num_kv_heads",
        "batch_size",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Qt",
        "K",
        "V",
        "O_ptr",
        "LSE_ptr",
        "page_table",
        "causal_seqlens_kv_global",
        "scale_log2_ptr",
        "bmm2_ptr",
        "sinks_ptr",
        "split_counter",
        "split_ws",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=210944,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

DEC_B4C4 = cake_program.CakeProgram(
    name="cake_fmha_decode_bf16_da0b91411447542f0766",
    operator_dir="cake_fmha_decode_bf16",
    kernel="kernel_cake_fmha_decode_bf16_da0b91411447542f0766",
    sources=(
        "cake_fmha_decode_bf16_da0b91411447542f0766_kernel.cu",
        "cake_fmha_decode_bf16_da0b91411447542f0766_binding.cu",
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
        "O_ptr",
        "LSE_ptr",
        "page_table",
        "causal_seqlens_kv_global",
        "scale_log2_ptr",
        "bmm2_ptr",
        "sinks_ptr",
        "max_pages_per_seq",
        "max_local_seq_len",
        "softmax_scale_log2",
        "window_left",
        "num_q_heads",
        "num_kv_heads",
        "batch_size",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Qt",
        "K",
        "V",
        "O_ptr",
        "LSE_ptr",
        "page_table",
        "causal_seqlens_kv_global",
        "scale_log2_ptr",
        "bmm2_ptr",
        "sinks_ptr",
    ),
    block=(512, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=175360,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

DEC_B4C4_DS = cake_program.CakeProgram(
    name="cake_fmha_decode_bf16_41a50922430aef5ca651",
    operator_dir="cake_fmha_decode_bf16",
    kernel="kernel_cake_fmha_decode_bf16_41a50922430aef5ca651",
    sources=(
        "cake_fmha_decode_bf16_41a50922430aef5ca651_kernel.cu",
        "cake_fmha_decode_bf16_41a50922430aef5ca651_binding.cu",
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
        "O_ptr",
        "LSE_ptr",
        "page_table",
        "causal_seqlens_kv_global",
        "scale_log2_ptr",
        "bmm2_ptr",
        "sinks_ptr",
        "max_pages_per_seq",
        "max_local_seq_len",
        "softmax_scale_log2",
        "window_left",
        "num_q_heads",
        "num_kv_heads",
        "batch_size",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Qt",
        "K",
        "V",
        "O_ptr",
        "LSE_ptr",
        "page_table",
        "causal_seqlens_kv_global",
        "scale_log2_ptr",
        "bmm2_ptr",
        "sinks_ptr",
    ),
    block=(512, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=175360,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

DEC_SINK = cake_program.CakeProgram(
    name="cake_fmha_decode_bf16_eb613495c94cae21226f",
    operator_dir="cake_fmha_decode_bf16",
    kernel="kernel_cake_fmha_decode_bf16_eb613495c94cae21226f",
    sources=(
        "cake_fmha_decode_bf16_eb613495c94cae21226f_kernel.cu",
        "cake_fmha_decode_bf16_eb613495c94cae21226f_binding.cu",
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
        "O_ptr",
        "LSE_ptr",
        "page_table",
        "causal_seqlens_kv_global",
        "scale_log2_ptr",
        "bmm2_ptr",
        "sinks_ptr",
        "max_pages_per_seq",
        "max_local_seq_len",
        "softmax_scale_log2",
        "window_left",
        "num_q_heads",
        "num_kv_heads",
        "batch_size",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Qt",
        "K",
        "V",
        "O_ptr",
        "LSE_ptr",
        "page_table",
        "causal_seqlens_kv_global",
        "scale_log2_ptr",
        "bmm2_ptr",
        "sinks_ptr",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=210944,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

DEC_SINK_DS = cake_program.CakeProgram(
    name="cake_fmha_decode_bf16_394b0caa8e0362fe9988",
    operator_dir="cake_fmha_decode_bf16",
    kernel="kernel_cake_fmha_decode_bf16_394b0caa8e0362fe9988",
    sources=(
        "cake_fmha_decode_bf16_394b0caa8e0362fe9988_kernel.cu",
        "cake_fmha_decode_bf16_394b0caa8e0362fe9988_binding.cu",
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
        "O_ptr",
        "LSE_ptr",
        "page_table",
        "causal_seqlens_kv_global",
        "scale_log2_ptr",
        "bmm2_ptr",
        "sinks_ptr",
        "max_pages_per_seq",
        "max_local_seq_len",
        "softmax_scale_log2",
        "window_left",
        "num_q_heads",
        "num_kv_heads",
        "batch_size",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Qt",
        "K",
        "V",
        "O_ptr",
        "LSE_ptr",
        "page_table",
        "causal_seqlens_kv_global",
        "scale_log2_ptr",
        "bmm2_ptr",
        "sinks_ptr",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=210944,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    DEC_B128,
    DEC_B128_DS,
    DEC_B4C4,
    DEC_B4C4_DS,
    DEC_SINK,
    DEC_SINK_DS,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "DEC_B128",
    "DEC_B128_DS",
    "DEC_B4C4",
    "DEC_B4C4_DS",
    "DEC_SINK",
    "DEC_SINK_DS",
    "PROGRAMS",
    "aot_specs",
]
