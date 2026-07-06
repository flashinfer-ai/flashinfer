# Copyright (c) 2025 by FlashInfer team.
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

import math
from typing import Optional, Tuple

import torch

from flashinfer.api_logging import flashinfer_api


@flashinfer_api
def bsa_attn_blk64_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Forward pass for BSA block-sparse attention using the blk64 CUDA C++ kernel (SM100 only).

    Block granularity is 64 tokens (kSparseBlockSize=64, kRows=64).  Only bfloat16
    inputs are supported and head_dim must be 128.

    Args:
        q: Query tensor (batch, seqlen_q, num_heads, head_dim).
        k: Key tensor (batch, seqlen_k, num_heads_kv, head_dim).
        v: Value tensor (batch, seqlen_k, num_heads_kv, head_dim).
        q2k_block_index: Block index tensor (batch, num_heads, num_q_blocks, max_kv_blocks), int32.
        block_sparse_num: Number of KV blocks each Q block attends to (>= 1).
            Ignored when q2k_block_nums is provided.
        block_sizes: Actual token count per KV block (num_kv_blocks,), int32.  Pass None to
            skip per-block padding masking (assumes all blocks are full).
        q2k_block_nums: Per-(batch, head, q_block) number of KV blocks to attend to,
            (batch, num_heads, num_q_blocks) int32.  When None, uses fixed block_sparse_num.
        softmax_scale: Softmax scale (default: 1/sqrt(head_dim)).
        return_lse: Whether to return log-sum-exp.
        out: Pre-allocated output tensor (batch, seqlen_q, num_heads, head_dim).
        lse: Pre-allocated LSE tensor (batch, num_heads, seqlen_q).

    Returns:
        (out, lse) where lse is None if return_lse is False.
    """
    from .blk64 import load_blk64_ext  # noqa: PLC0415

    batch_size, seqlen_q, num_head, head_dim = q.shape
    seqlen_k = k.shape[1]

    assert q.dtype == torch.bfloat16, "blk64 kernel only supports bfloat16"
    assert k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert head_dim == 128, f"blk64 kernel requires head_dim=128, got {head_dim}"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    major, minor = torch.cuda.get_device_capability(q.device)
    arch = major * 10 + minor
    if arch // 10 != 10:
        raise RuntimeError(f"BSA blk64 only supports SM100, current device is SM{arch}")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Make inputs contiguous in BSHD layout.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    has_variable_block_nums = q2k_block_nums is not None

    # Phantom block masking: when using variable block nums, the kernel pads each row's KV count
    # to a multiple of 8 and fills phantom slots with the last real block.  Phantom blocks are
    # only masked correctly when HasBlockSizes=True.  Passing a full-block sizes tensor (all 64)
    # activates that path and ensures phantom blocks are zeroed in softmax.
    if block_sizes is None and has_variable_block_nums:
        num_kv_blocks = (seqlen_k + 63) // 64
        block_sizes = torch.full(
            (num_kv_blocks,), 64, dtype=torch.int32, device=q.device
        )
    has_block_sizes = block_sizes is not None

    # Prepare optional tensors: pass empty tensors for undefined args (C++ binding checks .defined()).
    block_sizes_arg = block_sizes.contiguous() if has_block_sizes else torch.Tensor()
    q2k_block_nums_arg = (
        q2k_block_nums.contiguous() if has_variable_block_nums else torch.Tensor()
    )

    ext = load_blk64_ext()
    results = ext.bsa_fused_fwd_blk64(
        q,
        k,
        v,
        q2k_block_index.contiguous(),
        block_sparse_num,
        block_sizes_arg,
        softmax_scale,
        q2k_block_nums_arg,
    )

    # Kernel returns [out_bshd, lse_bhs] with:
    #   out_bshd: (batch, seqlen_q, num_heads, head_dim)
    #   lse_bhs:  (batch, num_heads, seqlen_q)
    kernel_out, kernel_lse = results[0], results[1]

    if out is not None:
        out.copy_(kernel_out)
    else:
        out = kernel_out

    if return_lse:
        if lse is not None:
            lse.copy_(kernel_lse)
        else:
            lse = kernel_lse
        return out, lse

    return out, None
