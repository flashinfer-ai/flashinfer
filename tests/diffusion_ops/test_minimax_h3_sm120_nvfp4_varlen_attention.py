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
Concurrent use of the NVFP4 routes (several streams, CUDA graphs captured on one stream or with forked
branches, two host threads on one stream, serialized launches) completes and reproduces every call's
single-stream result bit for bit; each such test fails at a progress deadline instead of hanging.
Numerical and concurrency tests require exact compute capability 12.0: the generated programs
ship only sm_120a. Input validation and unsupported-device rejection remain covered on other GPUs.
"""

import math
import os
import subprocess
import sys
import textwrap
import threading
import time
from collections import OrderedDict

import pytest
import torch

from flashinfer.diffusion_ops import (
    minimax_h3_sm120_varlen_attention_fp8,
    minimax_h3_sm120_varlen_attention_nvfp4,
    minimax_h3_sm120_varlen_attention_nvfp4_nodelta,
)
import flashinfer.diffusion_ops.cake_minimax_h3_sm120_nvfp4_varlen_attention as nvfp4_ops
import flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention as quant_ops
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
    return get_compute_capability(torch.device("cuda:0")) == (12, 0)


requires_sm120 = pytest.mark.skipif(
    not _supported(),
    reason="generated programs require exact compute capability 12.0 (sm_120a)",
)


@pytest.mark.parametrize(
    "cuda_available, capability, expected",
    [
        (False, (12, 0), False),
        (True, (12, 0), True),
        (True, (12, 1), False),
        (True, (10, 0), False),
    ],
)
def test_supported_requires_exact_sm120(
    monkeypatch, cuda_available, capability, expected
):
    # The generated programs ship sm_120a only; Spark must not enter the numerical suite.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setitem(
        _supported.__globals__, "get_compute_capability", lambda device: capability
    )
    assert _supported() is expected


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("pv_mode", PV_MODES)
def test_minimax_h3_sm120_varlen_attention_nvfp4_rejects_unsupported_device(
    pv_mode,
) -> None:
    device = torch.device("cuda:0")
    if get_compute_capability(device) == (12, 0):
        pytest.skip("SM120 is supported; numerical tests cover this device")
    q, k, v = synthetic_inputs(1, 1, seed=0, device=device)
    cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
    with pytest.raises(RuntimeError, match=r"requires exact compute capability 12\.0"):
        minimax_h3_sm120_varlen_attention_nvfp4(
            q, k, v, cu, cu_seqlens_host=(0, 1), pv_mode=pv_mode
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
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


def _route(delta: bool):
    return (
        minimax_h3_sm120_varlen_attention_nvfp4
        if delta
        else minimax_h3_sm120_varlen_attention_nvfp4_nodelta
    )


def _await(streams, what: str, seconds: float = 60.0) -> None:
    """Fail instead of hanging: the work queued so far on ``streams`` must complete within ``seconds``."""
    ends = []
    for stream in streams:
        end = torch.cuda.Event()
        end.record(stream)
        ends.append(end)
    deadline = time.monotonic() + seconds
    while not all(end.query() for end in ends):
        if time.monotonic() > deadline:
            pytest.fail(f"{what} made no progress in {seconds:.0f} s")
        time.sleep(0.001)


def _control_words_are_zero(owners) -> bool:
    """Whether the zero-start words (arrival counters, grid-barrier word, head-control words) of every
    workspace owner in ``owners`` are zero again, as the kernels leave them after every complete call (the
    grid barrier's top bit is its sense, which flips at every use)."""
    for owner in owners:
        for name, buffer in nvfp4_ops._ZERO_STREAM_WORKSPACES[owner].items():
            words = buffer.view(torch.int32)
            if name == "barrier":
                words = words & 0x7FFFFFFF
            if int(torch.count_nonzero(words)) != 0:
                return False
    return True


def _new_capture_owners(before) -> list:
    """The workspace owners created by graph captures since the owner set ``before`` was taken."""
    return [
        owner
        for owner in nvfp4_ops._ZERO_STREAM_WORKSPACES
        if len(owner) == 3 and owner not in before
    ]


# A plan of the fused pre-processing and a long plan (the statistics and long-plan quantizer launches), both of
# which take the overlapped form on 170 and 188 SMs.
CONCURRENT_CU_SEQLENS = [
    (0, 1100, 2100),
    (0, 5000, 5000, 21000, 21129),
]


@requires_sm120
@pytest.mark.parametrize("cu_seqlens", CONCURRENT_CU_SEQLENS)
@pytest.mark.parametrize("delta", [True, False])
@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("num_streams", [2, 4])
def test_minimax_h3_sm120_varlen_attention_nvfp4_concurrent_streams_are_bitwise(
    cu_seqlens, delta, overlap, num_streams, monkeypatch
) -> None:
    """Streams running the route concurrently on different inputs each reproduce their single-stream result
    bit for bit, and every round completes: the arrival counters, the grid-barrier word and the head-control
    words are private to the (device, stream), and concurrent overlapped calls make progress.
    ``overlap=False`` empties the overlap table, so the plans take the separate launches."""
    device = torch.device("cuda:0")
    # fresh launch caches for this test (restored afterwards), so no launch form leaks between cases
    monkeypatch.setattr(nvfp4_ops, "_NVFP4_LAUNCHES", OrderedDict())
    if not overlap:
        monkeypatch.setattr(nvfp4_ops, "_OVERLAP_WIDTH", {})
    fn = _route(delta)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    inputs = [
        synthetic_inputs(cu_seqlens[-1], MINIMAX_H3_NUM_HEADS, seed=seed, device=device)
        for seed in range(31, 31 + num_streams)
    ]
    expected = [fn(q, k, v, cu, cu_seqlens_host=cu_seqlens) for q, k, v in inputs]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(device=device) for _ in inputs]
    outs = [torch.zeros_like(q) for q, _k, _v in inputs]
    for _round in range(8):
        for stream, (q, k, v), out in zip(streams, inputs, outs, strict=True):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fn(q, k, v, cu, out, cu_seqlens_host=cu_seqlens)
        _await(streams, f"concurrent calls on {num_streams} streams")
        for out, want in zip(outs, expected, strict=True):
            assert torch.equal(out, want)
            out.zero_()
    assert _control_words_are_zero([(device.index, s.cuda_stream) for s in streams])


# The plan of the graph, host-thread and serialized-launch tests: the fused pre-processing (the overlapped form
# on 170 and 188 SMs).
FUSED_CU_SEQLENS = (0, 1100, 2100)


def _fused_case(device: torch.device, seeds):
    """``cu_seqlens``, one input triple per seed and each triple's single-stream result."""
    cu = torch.tensor(FUSED_CU_SEQLENS, dtype=torch.int32, device=device)
    inputs = [
        synthetic_inputs(
            FUSED_CU_SEQLENS[-1], MINIMAX_H3_NUM_HEADS, seed=seed, device=device
        )
        for seed in seeds
    ]
    expected = [
        minimax_h3_sm120_varlen_attention_nvfp4(
            q, k, v, cu, cu_seqlens_host=FUSED_CU_SEQLENS
        )
        for q, k, v in inputs
    ]
    torch.cuda.synchronize()
    return cu, inputs, expected


def _fused_call(q, k, v, cu, out) -> None:
    minimax_h3_sm120_varlen_attention_nvfp4(
        q, k, v, cu, out, cu_seqlens_host=FUSED_CU_SEQLENS
    )


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_nvfp4_graphs_one_capture_stream() -> None:
    """Two graphs captured one after the other on torch's default capture stream get separate workspace
    sets (their owners carry the capture id) and, replayed concurrently on two streams, each reproduce
    their eager result bit for bit."""
    device = torch.device("cuda:0")
    cu, inputs, expected = _fused_case(device, (41, 42))
    outs = [torch.zeros_like(q) for q, _k, _v in inputs]
    before = set(nvfp4_ops._ZERO_STREAM_WORKSPACES)
    graphs = []
    for (q, k, v), out in zip(inputs, outs, strict=True):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _fused_call(q, k, v, cu, out)
        graphs.append(graph)
    owners = _new_capture_owners(before)
    # one capture stream, two captures
    assert len(owners) == 2
    assert owners[0][:2] == owners[1][:2] and owners[0][2] != owners[1][2]
    streams = [torch.cuda.Stream(device=device) for _ in graphs]
    for _round in range(16):
        for stream, graph in zip(streams, graphs, strict=True):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
        _await(streams, "two graphs replayed concurrently")
        for out, want in zip(outs, expected, strict=True):
            assert torch.equal(out, want)
            out.zero_()
    assert _control_words_are_zero(owners)


@requires_sm120
def test_minimax_h3_sm120_owner_asks_cuda_when_the_call_device_is_not_current(
    monkeypatch,
) -> None:
    """While another device is current, torch's capture query reads that device's stream; the owner of a
    call then asks CUDA about the call's own stream, which sees the capture (one GPU: the other current
    device and its idle stream are reported by patched torch queries inside a real capture)."""
    x = torch.zeros(16, device="cuda:0")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x.add_(1)
        raw = torch.cuda.current_stream().cuda_stream
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "current_device", lambda: 1)
            patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
            owner = quant_ops._owner(0, raw)
    assert len(owner) == 3 and owner[:2] == (0, raw) and owner[2] != 0
    # the same stream after the capture is an eager owner
    assert quant_ops._owner(0, raw) == (0, raw)


@requires_sm120
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_minimax_h3_sm120_varlen_attention_nvfp4_capture_on_another_device(
    monkeypatch,
) -> None:
    """A call on device 1 captured into a graph while device 0 is current gets a capture owner and no eager
    warm-up launch inside the capture, and its replays reproduce the eager result bit for bit."""
    device = torch.device("cuda:1")
    with torch.cuda.device(device):
        cu, ((q, k, v),), (want,) = _fused_case(device, (43,))
    out = torch.zeros_like(q)
    # the plan's first call on this device again: the warm-up gate must see the capture
    monkeypatch.setattr(nvfp4_ops, "_NVFP4_OVERLAP_LOADED", set())
    before = set(nvfp4_ops._ZERO_STREAM_WORKSPACES)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.device(device), torch.cuda.graph(graph), torch.cuda.device(0):
        _fused_call(q, k, v, cu, out)
    owners = _new_capture_owners(before)
    assert len(owners) == 1 and owners[0][0] == device.index
    assert not nvfp4_ops._NVFP4_OVERLAP_LOADED
    with torch.cuda.device(device):
        for _round in range(4):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, want)
            out.zero_()
    assert _control_words_are_zero(owners)


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_nvfp4_graph_beside_eager_calls() -> None:
    """A graph captured on a stream, replayed on another stream concurrently with eager calls on its capture
    stream: the eager calls use the stream's workspace set, the graph its capture's, and both outputs are
    their single-stream results bit for bit."""
    device = torch.device("cuda:0")
    cu, inputs, expected = _fused_case(device, (43, 44))
    (qg, kg, vg), (qe, ke, ve) = inputs
    capture_stream = torch.cuda.Stream(device=device)
    replay_stream = torch.cuda.Stream(device=device)
    out_graph, out_eager = torch.zeros_like(qg), torch.zeros_like(qe)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        _fused_call(qg, kg, vg, cu, out_graph)
    streams = (capture_stream, replay_stream)
    for _round in range(16):
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(replay_stream):
            graph.replay()
        with torch.cuda.stream(capture_stream):
            _fused_call(qe, ke, ve, cu, out_eager)
        _await(streams, "a graph replayed beside eager calls on its capture stream")
        assert torch.equal(out_graph, expected[0])
        assert torch.equal(out_eager, expected[1])
        out_graph.zero_()
        out_eager.zero_()
    assert _control_words_are_zero([(device.index, capture_stream.cuda_stream)])


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_nvfp4_two_threads_one_stream() -> None:
    """Two host threads calling on one stream: the per-(device, stream) host lock keeps the launches of
    every call contiguous on the stream, so the calls take the stream's workspace set in turn and each
    thread's output is its single-stream result bit for bit."""
    device = torch.device("cuda:0")
    cu, inputs, expected = _fused_case(device, (45, 46))
    stream = torch.cuda.Stream(device=device)
    outs = [torch.zeros_like(q) for q, _k, _v in inputs]
    stream.wait_stream(torch.cuda.current_stream())
    start = threading.Barrier(len(inputs), timeout=60.0)
    errors = []

    def caller(q, k, v, out) -> None:
        try:
            torch.cuda.set_device(device)
            with torch.cuda.stream(stream):
                start.wait()
                for _call in range(16):
                    _fused_call(q, k, v, cu, out)
        except Exception as error:
            errors.append(error)

    threads = [
        threading.Thread(target=caller, args=(q, k, v, out))
        for (q, k, v), out in zip(inputs, outs, strict=True)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60.0)
    if any(thread.is_alive() for thread in threads):
        pytest.fail("two host threads calling on one stream did not finish in 60 s")
    if errors:
        raise errors[0]
    _await((stream,), "two host threads calling on one stream")
    for out, want in zip(outs, expected, strict=True):
        assert torch.equal(out, want)
    assert _control_words_are_zero([(device.index, stream.cuda_stream)])


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_nvfp4_forked_capture_branches() -> None:
    """One capture with the operator on two forked branches joined back into the capture stream: each branch
    gets its own workspace set (one capture id, two streams), the branches run concurrently at every replay
    and both outputs are their eager results bit for bit."""
    device = torch.device("cuda:0")
    cu, inputs, expected = _fused_case(device, (47, 48))
    outs = [torch.zeros_like(q) for q, _k, _v in inputs]
    capture_stream = torch.cuda.Stream(device=device)
    branch = torch.cuda.Stream(device=device)
    before = set(nvfp4_ops._ZERO_STREAM_WORKSPACES)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        branch.wait_stream(capture_stream)
        _fused_call(*inputs[0], cu, outs[0])
        with torch.cuda.stream(branch):
            _fused_call(*inputs[1], cu, outs[1])
        capture_stream.wait_stream(branch)
    owners = _new_capture_owners(before)
    # two branch streams, one capture
    assert len(owners) == 2
    assert owners[0][1] != owners[1][1] and owners[0][2] == owners[1][2]
    replay_stream = torch.cuda.Stream(device=device)
    for _round in range(16):
        replay_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(replay_stream):
            graph.replay()
        _await((replay_stream,), "a graph with the operator on two forked branches")
        for out, want in zip(outs, expected, strict=True):
            assert torch.equal(out, want)
            out.zero_()
    assert _control_words_are_zero(owners)


# Run under ``CUDA_LAUNCH_BLOCKING=1``: every launch completes before the next one starts.
_SERIALIZED_SCRIPT = textwrap.dedent(
    """
    import torch

    import flashinfer.diffusion_ops.cake_minimax_h3_sm120_nvfp4_varlen_attention as ops

    bounds = (0, 1100, 2100)
    device = torch.device("cuda:0")
    generator = torch.Generator(device="cpu").manual_seed(49)
    q, k, v = (
        torch.randn(bounds[-1], 56, 128, generator=generator)
        .to(torch.bfloat16)
        .to(device)
        for _ in range(3)
    )
    cu = torch.tensor(bounds, dtype=torch.int32, device=device)
    routes = (
        ops.minimax_h3_sm120_varlen_attention_nvfp4,
        ops.minimax_h3_sm120_varlen_attention_nvfp4_nodelta,
    )
    outs = [
        (fn(q, k, v, cu, cu_seqlens_host=bounds), fn(q, k, v, cu, cu_seqlens_host=bounds))
        for fn in routes
    ]
    torch.cuda.synchronize()
    ops._OVERLAP_WIDTH = {}
    ops._NVFP4_LAUNCHES.clear()
    for fn, (first, again) in zip(routes, outs):
        separate = fn(q, k, v, cu, cu_seqlens_host=bounds)
        torch.cuda.synchronize()
        assert torch.equal(first, separate) and torch.equal(again, separate)
    print("SERIALIZED_OK")
    """
)


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_nvfp4_serialized_launches(
    monkeypatch,
) -> None:
    """With ``CUDA_LAUNCH_BLOCKING=1`` every launch completes before the next one starts, so no programmatic
    dependent overlaps its primary: the overlapped form still completes (its producers run first and finish
    on their own) and matches the separate launches bit for bit, both routes, the first call included.  Run
    in a subprocess with a deadline."""
    device = torch.device("cuda:0")
    # build the programs of both forms here first, so the subprocess only loads them
    cu, ((q, k, v),), _ = _fused_case(device, (49,))
    for delta in (True, False):
        _route(delta)(q, k, v, cu, cu_seqlens_host=FUSED_CU_SEQLENS)
    monkeypatch.setattr(nvfp4_ops, "_NVFP4_LAUNCHES", OrderedDict())
    monkeypatch.setattr(nvfp4_ops, "_OVERLAP_WIDTH", {})
    for delta in (True, False):
        _route(delta)(q, k, v, cu, cu_seqlens_host=FUSED_CU_SEQLENS)
    torch.cuda.synchronize()
    try:
        result = subprocess.run(
            [sys.executable, "-c", _SERIALIZED_SCRIPT],
            env=dict(os.environ, CUDA_LAUNCH_BLOCKING="1"),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("overlapped calls with serialized launches did not finish in 600 s")
    stderr_tail = result.stderr[-4000:]
    assert result.returncode == 0 and "SERIALIZED_OK" in result.stdout, stderr_tail


@requires_sm120
@pytest.mark.parametrize(
    "cu_seqlens",
    [
        (0, 1100, 2100),
        (0, 257, 4567, 4824),
        (0, 4096),
        (0, 8368, 16736, 25104, 33472),
    ],
)
def test_minimax_h3_sm120_varlen_attention_nvfp4_nodelta_producer_builds_are_bitwise(
    cu_seqlens, monkeypatch
) -> None:
    """The overlapped form's two producer builds, ``PREPROCESS_HEAD_MAJOR`` and ``PREPROCESS_HEAD_MAJOR_KA``
    (a K item waits for its head's K sums in the worker's second warp), give bit-identical outputs of the
    uncompensated route: each in turn produces for the lead and tail the host routes on this device (on 188
    SMs the host's own producers are ``PREPROCESS_HEAD_MAJOR_KA``), two calls each, the first included."""
    device = torch.device("cuda:0")
    num_sms = nvfp4_ops._sm_count(0)
    if num_sms not in nvfp4_ops._OVERLAP_WIDTH:
        pytest.skip(f"no overlapped form on {num_sms} SMs")
    q, k, v = synthetic_inputs(
        cu_seqlens[-1], MINIMAX_H3_NUM_HEADS, seed=67, device=device
    )
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    host_programs = nvfp4_ops.minimax_h3_sm120_nvfp4_programs
    producers = (nvfp4_ops.PREPROCESS_HEAD_MAJOR, nvfp4_ops.PREPROCESS_HEAD_MAJOR_KA)
    outs, routed_producers = [], []
    for producer in producers:

        def programs(plan, sms, *, delta=True, _producer=producer):
            routed = host_programs(plan, sms, delta=delta)
            if routed[0] in producers:
                routed_producers.append(routed[0])
                return (_producer, *routed[1:])
            return routed

        monkeypatch.setattr(nvfp4_ops, "minimax_h3_sm120_nvfp4_programs", programs)
        monkeypatch.setattr(nvfp4_ops, "_NVFP4_LAUNCHES", OrderedDict())
        monkeypatch.setattr(nvfp4_ops, "_NVFP4_OVERLAP_LOADED", set())
        first = minimax_h3_sm120_varlen_attention_nvfp4_nodelta(
            q, k, v, cu, cu_seqlens_host=cu_seqlens
        )
        again = minimax_h3_sm120_varlen_attention_nvfp4_nodelta(
            q, k, v, cu, cu_seqlens_host=cu_seqlens
        )
        torch.cuda.synchronize()
        assert torch.equal(first, again)
        outs.append(first)
    assert len(routed_producers) >= 2, "the plan did not take the overlapped form"
    if num_sms in nvfp4_ops._OVERLAP_WS_NODELTA_SMS:
        assert all(p is nvfp4_ops.PREPROCESS_HEAD_MAJOR_KA for p in routed_producers)
    else:
        assert all(p is nvfp4_ops.PREPROCESS_HEAD_MAJOR for p in routed_producers)
    assert torch.equal(outs[0], outs[1])


@requires_sm120
@pytest.mark.parametrize(
    "cu_seqlens",
    [
        (0, 1100, 2100),
        (0, 257, 4567, 4824),
        (0, 4096),
        (0, 8368, 16736, 25104, 33472),
    ],
)
def test_minimax_h3_sm120_varlen_attention_nvfp4_sfsel_lead_tail_are_bitwise(
    cu_seqlens, monkeypatch
) -> None:
    """The compensated overlapped form's two lead / tail builds, ``ATTENTION_LEAD`` / ``ATTENTION_TAIL`` and
    ``ATTENTION_SFSEL_LEAD`` / ``ATTENTION_SFSEL_TAIL`` (each MMA selects its tile's B scale record), give
    bit-identical outputs: each pair in turn runs as the lead and tail of the overlapped form the host routes on
    this device (on 170 SMs the host's own pair is the second for a plan above the fused pre-processing, the
    first for a plan of it), two calls each, the first included."""
    device = torch.device("cuda:0")
    num_sms = nvfp4_ops._sm_count(0)
    if num_sms not in nvfp4_ops._OVERLAP_WIDTH:
        pytest.skip(f"no overlapped form on {num_sms} SMs")
    q, k, v = synthetic_inputs(
        cu_seqlens[-1], MINIMAX_H3_NUM_HEADS, seed=71, device=device
    )
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    host_programs = nvfp4_ops.minimax_h3_sm120_nvfp4_programs
    pairs = (
        (nvfp4_ops.ATTENTION_LEAD, nvfp4_ops.ATTENTION_TAIL),
        (nvfp4_ops.ATTENTION_SFSEL_LEAD, nvfp4_ops.ATTENTION_SFSEL_TAIL),
    )
    outs, routed_pairs = [], []
    for pair in pairs:

        def programs(plan, sms, *, delta=True, _pair=pair):
            routed = host_programs(plan, sms, delta=delta)
            if tuple(routed[1:]) in pairs:
                routed_pairs.append(tuple(routed[1:]))
                return (routed[0], *_pair)
            return routed

        monkeypatch.setattr(nvfp4_ops, "minimax_h3_sm120_nvfp4_programs", programs)
        monkeypatch.setattr(nvfp4_ops, "_NVFP4_LAUNCHES", OrderedDict())
        monkeypatch.setattr(nvfp4_ops, "_NVFP4_OVERLAP_LOADED", set())
        first = minimax_h3_sm120_varlen_attention_nvfp4(
            q, k, v, cu, cu_seqlens_host=cu_seqlens
        )
        again = minimax_h3_sm120_varlen_attention_nvfp4(
            q, k, v, cu, cu_seqlens_host=cu_seqlens
        )
        torch.cuda.synchronize()
        assert torch.equal(first, again)
        outs.append(first)
    assert len(routed_pairs) >= 2, "the plan did not take the overlapped form"
    padded = -(-cu_seqlens[-1] // 128) * 128
    fused = (
        nvfp4_ops._PREPROCESS_MIN_TOKENS <= padded <= nvfp4_ops._PREPROCESS_MAX_TOKENS
    )
    sfsel = num_sms in nvfp4_ops._OVERLAP_SFSEL_SMS and not fused
    expected = pairs[1] if sfsel else pairs[0]
    assert all(p[0] is expected[0] and p[1] is expected[1] for p in routed_pairs)
    assert torch.equal(outs[0], outs[1])
