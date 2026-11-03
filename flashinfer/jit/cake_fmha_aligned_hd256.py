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
"""Generated programs of the aligned head-dim-256 single-query grouped paged BF16 GQA
decode of the cake_fmha module: four programs, each shipped for sm_100a and sm_103a,
whose 8-row query tile holds one KV head's query heads. The persistent program runs one
CTA per (request, KV head) at the grid (1, KV heads, requests); the cluster programs
split one (request, KV head)'s KV sequence over a cluster of 2, 4 or 8 CTAs at the grid
(cluster size, KV heads, requests). The host module ``flashinfer.cake_fmha`` selects a
program through its selector over (requests x KV heads, maximum KV length,
multiprocessor count) and the pinned per-row verdicts; launches outside the bands keep
the module's present head-dim-256 decode route.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

ALIGNED_HD256_BF16 = cake_program.CakeProgram(
    name="cake_fmha_aligned_hd256_50326788c79371d913ba",
    operator_dir="cake_fmha_aligned_hd256",
    kernel="kernel_cake_fmha_aligned_hd256_50326788c79371d913ba",
    sources=(
        "cake_fmha_aligned_hd256_50326788c79371d913ba_kernel.cu",
        "cake_fmha_aligned_hd256_50326788c79371d913ba_binding.cu",
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
    dynamic_smem_bytes=147328,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ALIGNED_HD256_CGA2_BF16 = cake_program.CakeProgram(
    name="cake_fmha_aligned_hd256_3d8340c8d395be9c7587",
    operator_dir="cake_fmha_aligned_hd256",
    kernel="kernel_cake_fmha_aligned_hd256_3d8340c8d395be9c7587",
    sources=(
        "cake_fmha_aligned_hd256_3d8340c8d395be9c7587_kernel.cu",
        "cake_fmha_aligned_hd256_3d8340c8d395be9c7587_binding.cu",
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
    cluster=(2, 1, 1),
    dynamic_smem_bytes=178560,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ALIGNED_HD256_CGA4_BF16 = cake_program.CakeProgram(
    name="cake_fmha_aligned_hd256_7ba007d1a2873ce28e12",
    operator_dir="cake_fmha_aligned_hd256",
    kernel="kernel_cake_fmha_aligned_hd256_7ba007d1a2873ce28e12",
    sources=(
        "cake_fmha_aligned_hd256_7ba007d1a2873ce28e12_kernel.cu",
        "cake_fmha_aligned_hd256_7ba007d1a2873ce28e12_binding.cu",
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
    cluster=(4, 1, 1),
    dynamic_smem_bytes=178560,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ALIGNED_HD256_CGA8_BF16 = cake_program.CakeProgram(
    name="cake_fmha_aligned_hd256_dc665a26118a54d4ccb7",
    operator_dir="cake_fmha_aligned_hd256",
    kernel="kernel_cake_fmha_aligned_hd256_dc665a26118a54d4ccb7",
    sources=(
        "cake_fmha_aligned_hd256_dc665a26118a54d4ccb7_kernel.cu",
        "cake_fmha_aligned_hd256_dc665a26118a54d4ccb7_binding.cu",
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
    cluster=(8, 1, 1),
    dynamic_smem_bytes=178560,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    ALIGNED_HD256_BF16,
    ALIGNED_HD256_CGA2_BF16,
    ALIGNED_HD256_CGA4_BF16,
    ALIGNED_HD256_CGA8_BF16,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "ALIGNED_HD256_BF16",
    "ALIGNED_HD256_CGA2_BF16",
    "ALIGNED_HD256_CGA4_BF16",
    "ALIGNED_HD256_CGA8_BF16",
    "PROGRAMS",
    "aot_specs",
]
