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
"""Generated programs of the fused Q/K RMSNorm + NeoX RoPE + FP8 E4M3 quantization + paged
KV append kernel: one program for every supported architecture (sm_90a, sm_100a,
sm_103a). The host module ``flashinfer.cake_fused_qk_rope_fp8_append`` computes the grid
and launches it.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

FP8_E4M3 = cake_program.CakeProgram(
    name="cake_fused_qk_rope_fp8_append_e8a15268bf7234456084",
    operator_dir="cake_fused_qk_rope_fp8_append",
    kernel="kernel_cake_fused_qk_rope_fp8_append_e8a15268bf7234456084",
    sources=(
        "cake_fused_qk_rope_fp8_append_e8a15268bf7234456084_kernel.cu",
        "cake_fused_qk_rope_fp8_append_e8a15268bf7234456084_binding.cu",
    ),
    targets=("sm_90a", "sm_100a", "sm_103a"),
    compile_options=(
        ("sm_90a", ()),
        ("sm_100a", ("--fmad=true",)),
        ("sm_103a", ("--fmad=true",)),
    ),
    args=(
        "qkv",
        "cos_sin",
        "seq_lens",
        "q_indptr",
        "page_indices",
        "q_norm_weight",
        "k_norm_weight",
        "k_scale",
        "v_scale",
        "q_scale_inv",
        "out_q",
        "key_cache",
        "value_cache",
        "out_k",
        "out_v",
        "q_scale",
        "split_k_flag",
        "num_rows",
        "num_requests",
        "num_q_heads",
        "num_kv_heads",
        "page_size",
        "max_pages_per_request",
        "max_seqlen",
        "max_seqlen_aligned",
        "quant_policy",
        "norm_policy",
        "is_prefill",
        "has_out_kv",
        "upper_max",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "qkv",
        "cos_sin",
        "seq_lens",
        "q_indptr",
        "page_indices",
        "q_norm_weight",
        "k_norm_weight",
        "k_scale",
        "v_scale",
        "q_scale_inv",
        "out_q",
        "key_cache",
        "value_cache",
        "out_k",
        "out_v",
        "q_scale",
        "split_k_flag",
    ),
    block=(96, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (FP8_E4M3,)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = ["FP8_E4M3", "PROGRAMS", "aot_specs"]
