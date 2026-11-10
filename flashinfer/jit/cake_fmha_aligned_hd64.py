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
"""Generated program of the aligned head-dim-64 single-query grouped paged BF16 GQA decode
of the cake_fmha module: one program, shipped for sm_100a and sm_103a, one CTA per
(request, KV head) whose 8-row query tile holds the KV head's query heads, launched at
the grid (1, KV heads, requests). The host module ``flashinfer.cake_fmha`` routes a
launch to it through its band selector over (requests x KV heads, maximum KV length) and
the pinned per-row verdicts; launches outside the bands keep the module's present head-
dim-64 decode route.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

ALIGNED_HD64_BF16 = cake_program.CakeProgram(
    name="cake_fmha_aligned_hd64_f922a4aaa2d1161d3a26",
    operator_dir="cake_fmha_aligned_hd64",
    kernel="kernel_cake_fmha_aligned_hd64_f922a4aaa2d1161d3a26",
    sources=(
        "cake_fmha_aligned_hd64_f922a4aaa2d1161d3a26_kernel.cu",
        "cake_fmha_aligned_hd64_f922a4aaa2d1161d3a26_binding.cu",
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
        "page_table",
        "seq_lens_kv",
        "max_pages_per_seq",
        "softmax_scale_log2",
        "num_q_heads",
        "num_kv_heads",
        "heads_per_kv",
        "batch_size",
        "skip_corr_threshold",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("Q", "K", "V", "O", "page_table", "seq_lens_kv"),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=156544,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (ALIGNED_HD64_BF16,)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = ["ALIGNED_HD64_BF16", "PROGRAMS", "aot_specs"]
