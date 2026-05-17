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

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .fwd import VideoSparseAttentionForwardGroup2QInterleaveKV as VideoSparseAttentionForward

__all__ = [
    "VideoSparseAttentionForward",
    "dsl_block_sparse_attn_forward",
]


def dsl_block_sparse_attn_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-sparse attention forward pass (Blackwell CuTe-DSL kernel).

    Parameters
    ----------
    Q, K, V : torch.Tensor
        Shape ``[batch, head, seqlen, dim]``, fp16 or bf16.
    q2k_index : torch.Tensor
        Shape ``[batch, head, MB, NB]``, int32.  KV-block indices per Q-block,
        padded with -1.
    q2k_num : torch.Tensor
        Shape ``[batch, head, MB]``, int32.  Number of attended KV-blocks per
        Q-block.
    variable_block_sizes : torch.Tensor
        Shape ``[MB]``, int32.  Token count per Q-block (uniform = 64).
    sm_scale : float, optional
        Softmax scale.  Defaults to ``1 / sqrt(head_dim)``.

    Returns
    -------
    output : torch.Tensor  shape ``[batch, head, seqlen, dim]``
    lse    : torch.Tensor  shape ``[batch, head, seqlen]``, float32
    """
    B, H, T, D = Q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    output = torch.empty_like(Q)
    lse = torch.empty((B, H, T), device=Q.device, dtype=torch.float32)

    cuda_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def _to_cute(t: torch.Tensor) -> cute.Tensor:
        return (
            from_dlpack(t.detach(), assumed_align=128)
            .mark_compact_shape_dynamic(mode=0, stride_order=t.dim_order())
            .mark_compact_shape_dynamic(mode=1, stride_order=t.dim_order())
            .mark_compact_shape_dynamic(mode=2, stride_order=t.dim_order())
        )

    Q_packed = _to_cute(Q)
    K_packed = _to_cute(K)
    V_packed = _to_cute(V)
    O_packed = _to_cute(output)

    LSE_packed = from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=2)
    idx_packed = from_dlpack(q2k_index.detach()).mark_layout_dynamic(leading_dim=3)
    num_packed = from_dlpack(q2k_num.detach()).mark_layout_dynamic(leading_dim=2)
    vbs_packed = from_dlpack(variable_block_sizes.detach()).mark_layout_dynamic(leading_dim=0)

    compile_key = (H, D, Q.dtype, sm_scale)
    cache = dsl_block_sparse_attn_forward.compile_cache
    if compile_key not in cache:
        kernel = VideoSparseAttentionForward(block_m=64, block_n=64, headdim=D)
        cache[compile_key] = cute.compile(
            kernel,
            Q_packed, K_packed, V_packed,
            sm_scale,
            O_packed, LSE_packed,
            idx_packed, num_packed, vbs_packed,
            cuda_stream,
        )

    cache[compile_key](
        Q_packed, K_packed, V_packed,
        sm_scale,
        O_packed, LSE_packed,
        idx_packed, num_packed, vbs_packed,
        cuda_stream,
    )

    return output, lse


dsl_block_sparse_attn_forward.compile_cache = {}

