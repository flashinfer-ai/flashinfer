"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Cake SM100/SM103 Sage-FP8 block-sparse attention and fused quantization.

The public surface of the SM100-family programs of
:mod:`flashinfer.cake_sage_block_sparse_attention_backend`, the self-contained
package that holds the host API bodies, the host plan and the generated CUDA
sources.  Compared with the ``sm100_blk64`` CuTe-DSL Sage-FP8 path this backend
accepts any batch size and head count, GQA (``num_heads`` a multiple of
``num_kv_heads``), ragged ``seqlen_q`` / ``seqlen_k`` (no 64-token alignment),
per-row KV counts (``q2k_block_nums``, empty rows produce exact zeros), rank-1/2/3
``block_sizes`` token masks and an optional LSE.  Inputs and outputs use the
``bsa_attn_sm100_blk64_fwd`` contract: BSHD ``float8_e4m3fn`` Q/K/V,
``q_scale[B,H,Sq]``, ``k_scale[B,Hkv,ceil(Sk/16)]``, ``v_scale[Hkv,D]`` or
``v_scale[B,Hkv,D]`` (float32), BF16 BSHD output.

:func:`sage_fp8_quantize_sm100` is the matching fused quantizer (two launches):
per-token Q scales, per-16-token K scales and per-channel V scales, replacing
the ~30-kernel torch recipe.
"""

from flashinfer.cake_sage_block_sparse_attention_backend import (
    bsa_attn_sm100_sage_fwd_cake,
    is_cake_sage_sm100_supported,
    sage_fp8_quantize_sm100,
)

__all__ = [
    "bsa_attn_sm100_sage_fwd_cake",
    "is_cake_sage_sm100_supported",
    "sage_fp8_quantize_sm100",
]
