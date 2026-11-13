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

Pure-torch oracle of the MiniMax-H3 attention output projection
(``flashinfer.diffusion_ops.minimax_h3_out_proj``): the sequence-parallel
receive-layout unpacking, the FP32 GEMM (TF32 disabled) rounded to BF16 and
the indexed gate / residual epilogue, each stage rounded to BF16 as the
operators do.  Shared by ``tests/diffusion_ops/test_minimax_h3_out_proj.py``
and ``tests/diffusion_ops/test_minimax_h3_mlp.py`` (FC2 + epilogue stage).
"""

from typing import Optional

import torch

from flashinfer.diffusion_ops.minimax_h3_out_proj import (
    MINIMAX_H3_ATTN_DIM,
    MINIMAX_H3_HIDDEN,
)

_REFERENCE_CHUNK_ROWS = 1024


def minimax_h3_unpack_attn_out(attn_out: torch.Tensor) -> torch.Tensor:
    r"""``[P, M, 56 // P, 128]`` receive layout -> the logical activation ``A`` ``[M, 7168]``
    with ``A[m, h * 128 + d] = attn_out[h // (56 // P), m, h % (56 // P), d]`` (a copy; the
    fused kernels never materialise it)."""
    degree, rows = int(attn_out.shape[0]), int(attn_out.shape[1])
    return (
        attn_out.reshape(degree, rows, MINIMAX_H3_ATTN_DIM // degree)
        .permute(1, 0, 2)
        .reshape(rows, MINIMAX_H3_ATTN_DIM)
    )


def _gate_residual_epilogue(
    o: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    gate_rows = int(gate.shape[0])
    idx = gate_index.to(torch.int64)
    valid = (idx >= 0) & (idx < gate_rows)
    g = gate.index_select(0, idx.clamp(0, gate_rows - 1))
    g = torch.where(valid[:, None], g, torch.zeros_like(g))
    p = (g * o).to(torch.bfloat16)  # BF16 x BF16: FP32 product, one rounding
    return (residual + p).to(torch.bfloat16)


def _reference_from_operands(
    a: torch.Tensor,
    w: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """``out = BF16(residual + BF16(gate[idx] * BF16(alpha * a @ w^T)))`` with an FP32 GEMM
    (TF32 disabled) over any float-convertible operands ``a`` ``[M, K]`` / ``w`` ``[N, K]``."""
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        w_t = w.float().t()
        o = torch.empty(
            (a.shape[0], MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=a.device
        )
        for r0 in range(0, a.shape[0], _REFERENCE_CHUNK_ROWS):
            r1 = min(a.shape[0], r0 + _REFERENCE_CHUNK_ROWS)
            h = a[r0:r1].float() @ w_t
            if alpha is not None:
                h = h * float(alpha)
            o[r0:r1] = h.to(torch.bfloat16)
        return _gate_residual_epilogue(o, gate, gate_index, residual)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def minimax_h3_out_proj_reference(
    attn_out: torch.Tensor,
    o_weight: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    r"""Pure-torch BF16 reference of :func:`minimax_h3_out_proj`: unpack the receive layout,
    FP32 GEMM (TF32 disabled) rounded to BF16, indexed gate product rounded to BF16, residual sum
    rounded to BF16."""
    return _reference_from_operands(
        minimax_h3_unpack_attn_out(attn_out), o_weight, gate, gate_index, residual
    )
