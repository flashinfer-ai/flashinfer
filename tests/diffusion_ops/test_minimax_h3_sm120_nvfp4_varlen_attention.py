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
"""Tests for the SM120 (GB202) experimental NVFP4-QK / FP8-PV MiniMax-H3 packed-varlen attention.

The operator evaluates ``softmax(Q K^T * scale) V`` per ``cu_seqlens`` segment with E2M1 Q / K
(one UE4M3 scale per 16 channels after the segment-mean shift and a fixed orthonormal Hadamard
rotation), E4M3 P / V and an FP32 softmax.  The E2M1 scores carry the FP4 block-scaled error
budget: the package tolerance is ``atol = 1.0, rtol = 0.1`` on every element (the same contract as
the SM100 NVFP4 attention routes) plus a relative-L2 bound about three times the FP8 operator's,
checked on packed boundaries, tails, empty segments and multi-segment streams.  Every test needs
an SM120 GPU (the JIT module only builds for compute capability 12.x).
"""

import math

import pytest
import torch

from flashinfer.diffusion_ops import (
    minimax_h3_sm120_varlen_attention_fp8,
    minimax_h3_sm120_varlen_attention_nvfp4,
)
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_nvfp4_varlen_attention import (
    workspace_bytes_nvfp4,
)
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention import (
    MINIMAX_H3_HEAD_DIM,
    MINIMAX_H3_NUM_HEADS,
    workspace_bytes,
)
from flashinfer.utils import get_compute_capability

ATOL = 1.0
RTOL = 0.1
# Measured relative L2 error on Gaussian inputs is 0.14-0.15 (FP8 operator: 0.05); the bound leaves
# headroom for the boundary shapes without accepting a broken kernel.
REL_L2_MAX = 0.25
# Packed boundaries: single tokens, exact / one-past tile edges, empty segments, ragged
# multi-segment streams, and a multi-tile single segment.
CU_SEQLENS = [
    (0, 1),
    (0, 128),
    (0, 129),
    (0, 0, 129, 129, 500),
    (0, 133, 300, 900),
    (0, 257, 4567, 4824),
    (0, 4097),
]


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = get_compute_capability(torch.device("cuda:0"))
    return major == 12


requires_sm120 = pytest.mark.skipif(
    not _supported(), reason="requires an SM120 (RTX 5090 / RTX PRO 6000 Blackwell) GPU"
)


def synthetic_inputs(tokens: int, heads: int, seed: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(
        torch.randn(
            tokens, heads, MINIMAX_H3_HEAD_DIM, generator=generator, dtype=torch.float32
        )
        .to(torch.bfloat16)
        .to(device)
        for _ in range(3)
    )


def fp32_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: tuple,
    query_chunk: int = 1024,
) -> torch.Tensor:
    """Exact-softmax FP32 attention per segment and head over the BF16 operands (TF32 off)."""
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    scale = 1.0 / math.sqrt(MINIMAX_H3_HEAD_DIM)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for a, b in zip(cu_seqlens, cu_seqlens[1:], strict=False):
            if b == a:
                continue
            # One head at a time: a [query_chunk, L] FP32 score block stays below 0.5 GiB even at
            # the 110k-token rows, so the oracle fits next to the operands on a 32 GiB board.
            for h in range(q.shape[1]):
                kf = k[a:b, h].float()  # [L, D]
                vf = v[a:b, h].float()
                for start in range(a, b, query_chunk):
                    stop = min(start + query_chunk, b)
                    scores = (
                        torch.matmul(q[start:stop, h].float(), kf.transpose(0, 1))
                        * scale
                    )
                    out[start:stop, h] = torch.matmul(torch.softmax(scores, dim=-1), vf)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return out


def _check(actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (actual.float() - expected).abs()
    bound = ATOL + RTOL * expected.abs()
    assert torch.isfinite(actual.float()).all(), "non-finite output"
    bad = int((diff > bound).sum().item())
    assert bad == 0, (
        f"{bad} elements outside atol={ATOL} rtol={RTOL} (max abs err {diff.max().item():.4f})"
    )
    rel_l2 = (diff.norm() / expected.norm().clamp_min(1e-6)).item()
    assert rel_l2 < REL_L2_MAX, f"relative L2 error {rel_l2:.4f} exceeds {REL_L2_MAX}"


@requires_sm120
@pytest.mark.parametrize("cu_seqlens", CU_SEQLENS)
@pytest.mark.parametrize("heads", [MINIMAX_H3_NUM_HEADS, 3])
def test_minimax_h3_sm120_varlen_attention_nvfp4_matches_fp32_oracle(
    cu_seqlens: tuple, heads: int
) -> None:
    if heads != MINIMAX_H3_NUM_HEADS and cu_seqlens[-1] > 1000:
        pytest.skip("small-head sweep covers the boundary shapes only")
    device = torch.device("cuda:0")
    tokens = cu_seqlens[-1]
    q, k, v = synthetic_inputs(tokens, heads, seed=tokens + heads, device=device)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    out = minimax_h3_sm120_varlen_attention_nvfp4(
        q, k, v, cu, cu_seqlens_host=cu_seqlens
    )
    torch.cuda.synchronize()
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    _check(out, fp32_oracle(q, k, v, cu_seqlens))
    # Idempotence: the quantized operands and block scales are rebuilt every call, bit-exactly.
    again = minimax_h3_sm120_varlen_attention_nvfp4(
        q, k, v, cu, cu_seqlens_host=cu_seqlens
    )
    torch.cuda.synchronize()
    assert torch.equal(out, again)


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_nvfp4_preallocated_output_and_device_cu() -> (
    None
):
    device = torch.device("cuda:0")
    cu_seqlens = (0, 300, 700)
    q, k, v = synthetic_inputs(
        cu_seqlens[-1], MINIMAX_H3_NUM_HEADS, seed=7, device=device
    )
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    # No host copy of cu_seqlens: the plan reads the bounds from the device once.
    result = minimax_h3_sm120_varlen_attention_nvfp4(q, k, v, cu, out)
    torch.cuda.synchronize()
    assert result.data_ptr() == out.data_ptr()
    _check(out, fp32_oracle(q, k, v, cu_seqlens))


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_nvfp4_custom_scale_and_empty_stream() -> (
    None
):
    device = torch.device("cuda:0")
    cu_seqlens = (0, 200)
    q, k, v = synthetic_inputs(200, 4, seed=11, device=device)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    scale = 0.05
    out = minimax_h3_sm120_varlen_attention_nvfp4(
        q, k, v, cu, cu_seqlens_host=cu_seqlens, softmax_scale=scale
    )
    torch.cuda.synchronize()
    expected = torch.softmax(
        torch.einsum("mhd,nhd->hmn", q.float(), k.float()) * scale, dim=-1
    )
    expected = torch.einsum("hmn,nhd->mhd", expected, v.float())
    _check(out, expected)
    # Every segment empty: nothing is written and no kernel is launched.
    empty = torch.zeros(0, 4, MINIMAX_H3_HEAD_DIM, dtype=torch.bfloat16, device=device)
    cu0 = torch.tensor([0, 0, 0], dtype=torch.int32, device=device)
    assert (
        minimax_h3_sm120_varlen_attention_nvfp4(empty, empty, empty, cu0).shape
        == empty.shape
    )


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_nvfp4_rejects_bad_inputs() -> None:
    device = torch.device("cuda:0")
    q, k, v = synthetic_inputs(300, 4, seed=3, device=device)
    cu = torch.tensor([0, 300], dtype=torch.int32, device=device)
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_nvfp4(q.float(), k, v, cu)
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_nvfp4(q, k[:299], v, cu)
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_nvfp4(q, k, v, cu.to(torch.int64))
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_nvfp4(q, k, v, cu, cu_seqlens_host=(0, 299))
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_nvfp4(q, k, v, cu, cu_seqlens_host=(5, 300))
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_nvfp4(
            q, k, v, cu, cu_seqlens_host=(0, 200, 100, 300)
        )
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_nvfp4(
            q, k, v, cu, out=torch.empty_like(q)[:, :3]
        )


def test_minimax_h3_sm120_varlen_attention_nvfp4_workspace_is_smaller_than_fp8() -> (
    None
):
    # E2M1 codes + UE4M3 block scales are 9/16 of the E4M3 rows; the V^T tile and the scale /
    # partial buffers are shared with the FP8 operator.  The K block scales are stored as full
    # 1 KiB (128-key) tiles, so only streams below ~8 tokens per 128-key block pay more.
    assert workspace_bytes_nvfp4(1, 1, 1) > 0
    for tokens, heads, segments in ((128, 1, 1), (4824, 56, 3), (109952, 56, 1)):
        assert (
            0
            < workspace_bytes_nvfp4(tokens, heads, segments)
            < workspace_bytes(tokens, heads, segments)
        )


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_nvfp4_error_vs_fp8_operator() -> None:
    """Diagnostic contract: the NVFP4 route is within the FP4 tolerance and its error stays within
    a bounded factor of the shipped FP8 operator on the same inputs (both against the FP32 oracle)."""
    device = torch.device("cuda:0")
    cu_seqlens = (0, 257, 4567, 4824)
    q, k, v = synthetic_inputs(
        cu_seqlens[-1], MINIMAX_H3_NUM_HEADS, seed=5, device=device
    )
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    expected = fp32_oracle(q, k, v, cu_seqlens)
    out4 = minimax_h3_sm120_varlen_attention_nvfp4(
        q, k, v, cu, cu_seqlens_host=cu_seqlens
    )
    out8 = minimax_h3_sm120_varlen_attention_fp8(
        q, k, v, cu, cu_seqlens_host=cu_seqlens
    )
    torch.cuda.synchronize()
    _check(out4, expected)
    rel4 = ((out4.float() - expected).norm() / expected.norm()).item()
    rel8 = ((out8.float() - expected).norm() / expected.norm()).item()
    assert rel8 < rel4 < 5.0 * rel8, (rel4, rel8)
