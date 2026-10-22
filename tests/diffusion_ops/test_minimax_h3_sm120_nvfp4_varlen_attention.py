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
"""Tests for the SM120 (GB202) experimental NVFP4 (SageAttention3-recipe) MiniMax-H3 packed-varlen attention.

The operator evaluates ``softmax(Q K^T * scale) V`` per ``cu_seqlens`` segment with E2M1 Q / K / V / P
operands and one UE4M3 scale per 16 elements (SageAttention3's recipe: per-segment K mean and
per-128-row Q block mean removal with in-kernel compensation, block amax / 6 scales, two-level P
quantization) and an FP32 softmax.  The E2M1 scores and probabilities carry the FP4 block-scaled
error budget: the package tolerance is ``atol = 1.0, rtol = 0.1`` on every element (the same contract
as the SM100 NVFP4 attention routes) plus a relative-L2 bound about four times the FP8 operator's,
checked on packed boundaries, tails, empty segments and multi-segment streams.  The opt-in
``pv_mode="fp8"`` route (NVFP4 scores, E4M3 probabilities and V; three generated programs) shares
the API and the FP4 tolerance and is held to a smaller relative-L2 error than the default route on
the same inputs, to bitwise idempotence, to stream-private workspaces and to its program routing.
Every test needs an SM120 GPU (the JIT modules only build for compute capability 12.x) except the
workspace estimators and the ``pv_mode`` validation.
"""

import math

import pytest
import torch

from flashinfer.diffusion_ops import (
    minimax_h3_sm120_varlen_attention_fp8,
    minimax_h3_sm120_varlen_attention_nvfp4,
)
import flashinfer.diffusion_ops.cake_minimax_h3_sm120_nvfp4_varlen_attention as nvfp4_ops
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_nvfp4_varlen_attention import (
    minimax_h3_sm120_nvfp4_fp8pv_programs,
    workspace_bytes_nvfp4,
    workspace_bytes_nvfp4_fp8pv,
)
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention import (
    MINIMAX_H3_HEAD_DIM,
    MINIMAX_H3_NUM_HEADS,
    _plan,
    workspace_bytes,
)
from flashinfer.jit.cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention import (
    ATTENTION_BASE,
    ATTENTION_SHORT,
    QUANTIZE,
    QUANTIZE_MB3,
)
from flashinfer.utils import get_compute_capability

ATOL = 1.0
RTOL = 0.1
# Measured relative L2 error on Gaussian inputs is 0.186-0.193 (SageAttention3 itself: 0.19-0.24; the FP8
# operator: 0.05); the bound leaves headroom for the boundary shapes without accepting a broken kernel.
REL_L2_MAX = 0.30
# Packed boundaries: single tokens, exact / one-past tile edges, empty segments, ragged
# multi-segment streams, and a multi-tile single segment.
CU_SEQLENS = [
    (0, 1),
    (0, 128),
    (0, 129),
    (0, 0, 129, 129, 500),
    (0, 133, 300, 900),
    (0, 257, 4567, 4824),
    (0, 4096),
    (0, 4097),
]
PV_MODES = ("fp4", "fp8")


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
@pytest.mark.parametrize("pv_mode", PV_MODES)
@pytest.mark.parametrize("cu_seqlens", CU_SEQLENS)
@pytest.mark.parametrize("heads", [MINIMAX_H3_NUM_HEADS, 3])
def test_minimax_h3_sm120_varlen_attention_nvfp4_matches_fp32_oracle(
    cu_seqlens: tuple, heads: int, pv_mode: str
) -> None:
    if heads != MINIMAX_H3_NUM_HEADS and cu_seqlens[-1] > 1000:
        pytest.skip("small-head sweep covers the boundary shapes only")
    device = torch.device("cuda:0")
    tokens = cu_seqlens[-1]
    q, k, v = synthetic_inputs(tokens, heads, seed=tokens + heads, device=device)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    out = minimax_h3_sm120_varlen_attention_nvfp4(
        q, k, v, cu, cu_seqlens_host=cu_seqlens, pv_mode=pv_mode
    )
    torch.cuda.synchronize()
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    _check(out, fp32_oracle(q, k, v, cu_seqlens))
    # Idempotence: the quantized operands and block scales are rebuilt every call, bit-exactly.
    again = minimax_h3_sm120_varlen_attention_nvfp4(
        q, k, v, cu, cu_seqlens_host=cu_seqlens, pv_mode=pv_mode
    )
    torch.cuda.synchronize()
    assert torch.equal(out, again)
    if pv_mode == "fp4":
        # The default is the fp4 route, bit for bit.
        default = minimax_h3_sm120_varlen_attention_nvfp4(
            q, k, v, cu, cu_seqlens_host=cu_seqlens
        )
        torch.cuda.synchronize()
        assert torch.equal(out, default)


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
    # E2M1 codes + UE4M3 block scales are 9/16 of the E4M3 rows for Q / K and V^T; the block-mean
    # rows and channel means add one row per (Q block, head).  The K / V block scales are stored as
    # full 1 KiB (128-key) tiles, so only streams below ~8 tokens per 128-key block pay more.
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
    assert rel8 < rel4 < 6.0 * rel8, (rel4, rel8)


def test_minimax_h3_sm120_varlen_attention_nvfp4_rejects_unknown_pv_mode() -> None:
    # Validated before any device work: CPU tensors reach the pv_mode check first.
    q = torch.empty(4, 2, MINIMAX_H3_HEAD_DIM, dtype=torch.bfloat16)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="pv_mode must be one of"):
        minimax_h3_sm120_varlen_attention_nvfp4(q, q, q, cu, pv_mode="fp16")


def test_minimax_h3_sm120_nvfp4_fp8pv_workspace_is_smaller_than_fp8() -> None:
    # Packed E2M1 Q / K (half the E4M3 rows) plus their block scales and the K scale tiles against the FP8
    # operator's E4M3 Q / K; V^T is E4M3 in both.
    assert workspace_bytes_nvfp4_fp8pv(1, 1, 1) > 0
    for tokens, heads, segments in ((128, 1, 1), (4824, 56, 3), (109952, 56, 1)):
        assert (
            0
            < workspace_bytes_nvfp4_fp8pv(tokens, heads, segments)
            < workspace_bytes(tokens, heads, segments)
        )


@requires_sm120
@pytest.mark.parametrize(
    ("cu_seqlens", "quantize", "attention"),
    [
        ((0, 4096), QUANTIZE, ATTENTION_SHORT),
        ((0, 4097), QUANTIZE, ATTENTION_BASE),
        ((0, 1024), QUANTIZE_MB3, ATTENTION_SHORT),
        ((0, 1025), QUANTIZE, ATTENTION_SHORT),
        ((0, 0, 129, 129, 500), QUANTIZE_MB3, ATTENTION_SHORT),
        ((0, 257, 4567, 4824), QUANTIZE, ATTENTION_BASE),
    ],
)
def test_minimax_h3_sm120_nvfp4_fp8pv_routes_its_programs(
    cu_seqlens: tuple, quantize, attention, monkeypatch
) -> None:
    device = torch.device("cuda:0")
    plan = _plan(cu_seqlens, MINIMAX_H3_NUM_HEADS, device)
    assert minimax_h3_sm120_nvfp4_fp8pv_programs(plan) == (quantize, attention)
    # The launches the operator issues follow the same selection.
    seen = []
    original = nvfp4_ops._minimax_h3_sm120_varlen_attention_nvfp4_fp8pv_impl

    def recording(*args):
        seen.append((args[-7], args[-6]))  # quantize_small, attention_short
        return original(*args)

    monkeypatch.setattr(
        nvfp4_ops, "_minimax_h3_sm120_varlen_attention_nvfp4_fp8pv_impl", recording
    )
    q, k, v = synthetic_inputs(
        cu_seqlens[-1], MINIMAX_H3_NUM_HEADS, seed=13, device=device
    )
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    out = minimax_h3_sm120_varlen_attention_nvfp4(
        q, k, v, cu, cu_seqlens_host=cu_seqlens, pv_mode="fp8"
    )
    torch.cuda.synchronize()
    assert seen == [(quantize is QUANTIZE_MB3, attention is ATTENTION_SHORT)]
    _check(out, fp32_oracle(q, k, v, cu_seqlens))


@requires_sm120
def test_minimax_h3_sm120_nvfp4_fp8pv_is_more_accurate_than_the_default_route() -> None:
    """``pv_mode="fp8"`` keeps the probabilities and V in E4M3: its relative-L2 error against the FP32
    oracle is below the default (Sage3-recipe) route's on the same inputs, and it meets the FP4
    tolerance on every element."""
    device = torch.device("cuda:0")
    for cu_seqlens, seed in (((0, 257, 4567, 4824), 5), ((0, 8192), 17)):
        q, k, v = synthetic_inputs(
            cu_seqlens[-1], MINIMAX_H3_NUM_HEADS, seed=seed, device=device
        )
        cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        expected = fp32_oracle(q, k, v, cu_seqlens)
        out8 = minimax_h3_sm120_varlen_attention_nvfp4(
            q, k, v, cu, cu_seqlens_host=cu_seqlens, pv_mode="fp8"
        )
        out4 = minimax_h3_sm120_varlen_attention_nvfp4(
            q, k, v, cu, cu_seqlens_host=cu_seqlens, pv_mode="fp4"
        )
        torch.cuda.synchronize()
        _check(out8, expected)
        rel8 = ((out8.float() - expected).norm() / expected.norm()).item()
        rel4 = ((out4.float() - expected).norm() / expected.norm()).item()
        assert rel8 < rel4, (cu_seqlens, rel8, rel4)


@requires_sm120
def test_minimax_h3_sm120_nvfp4_fp8pv_two_streams_keep_private_counters() -> None:
    """The statistics launch's arrival counters are per (device, stream): two streams running the
    route concurrently on different inputs each reproduce their single-stream result bit for bit."""
    device = torch.device("cuda:0")
    cu_seqlens = (0, 700, 1400, 2100)  # folded statistics (segments above 512 tokens)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    inputs = [
        synthetic_inputs(cu_seqlens[-1], MINIMAX_H3_NUM_HEADS, seed=seed, device=device)
        for seed in (21, 22)
    ]
    expected = []
    for q, k, v in inputs:
        expected.append(
            minimax_h3_sm120_varlen_attention_nvfp4(
                q, k, v, cu, cu_seqlens_host=cu_seqlens, pv_mode="fp8"
            )
        )
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(device=device) for _ in inputs]
    outs = [torch.empty_like(q) for q, _k, _v in inputs]
    for _round in range(4):
        for stream, (q, k, v), out in zip(streams, inputs, outs, strict=True):
            with torch.cuda.stream(stream):
                minimax_h3_sm120_varlen_attention_nvfp4(
                    q, k, v, cu, out, cu_seqlens_host=cu_seqlens, pv_mode="fp8"
                )
    torch.cuda.synchronize()
    for out, want in zip(outs, expected, strict=True):
        assert torch.equal(out, want)
