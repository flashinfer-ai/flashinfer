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
"""Generated program of the aligned speculative-decode grouped paged BF16 GQA decode of the
cake_fmha module: one program, shipped for sm_100a and sm_103a, whose 32-row query tile
holds the 1 to 4 query tokens of one KV head's 8 query heads. It splits one (request, KV
head)'s KV sequence over a cluster of 2 CTAs at the grid (2, KV heads, requests). The
host module ``flashinfer.cake_fmha`` routes a launch to it through its selector over
(requests x KV heads, maximum KV length, query tokens, multiprocessor count) and the
pinned per-row verdicts; other launches keep the module's present speculative-decode
route.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

ALIGNED_MTP_Q32_CGA2_BF16 = cake_program.CakeProgram(
    name="cake_fmha_aligned_mtp_q32_a1e5ddfa63d266952503",
    operator_dir="cake_fmha_aligned_mtp_q32",
    kernel="kernel_cake_fmha_aligned_mtp_q32_a1e5ddfa63d266952503",
    sources=(
        "cake_fmha_aligned_mtp_q32_a1e5ddfa63d266952503_kernel.cu",
        "cake_fmha_aligned_mtp_q32_a1e5ddfa63d266952503_binding.cu",
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
        "seq_len_q",
        "skip_corr_threshold",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("Q", "K", "V", "O", "page_table", "seq_lens_kv"),
    block=(512, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=201088,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (ALIGNED_MTP_Q32_CGA2_BF16,)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = ["ALIGNED_MTP_Q32_CGA2_BF16", "PROGRAMS", "aot_specs"]
