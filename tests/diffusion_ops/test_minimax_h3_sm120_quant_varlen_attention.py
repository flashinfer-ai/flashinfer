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
"""SM120 (GB202) FP8 MiniMax-H3 packed-varlen attention against an FP32 exact-softmax oracle.

Q / K / P / V are E4M3 tensor-core operands (per-token Q scale, per-128-key-block K scale after
the segment-mean shift, 2^8-biased P, per-(segment, channel) V scale) with an FP32 softmax, so the
BF16 output carries FP8 quantization noise: the package tolerance is ``atol = rtol = 0.1`` on every
element plus a relative-L2 bound, checked on packed boundaries, tails, empty segments and
multi-segment streams.  A plan of the overlap band runs the overlapped form on 170 and 188 SMs and
must equal the sequential launches bit for bit.  Concurrent use (several streams, CUDA graphs
captured on one stream or with forked branches, two host threads on one stream, serialized
launches) completes and reproduces every call's single-stream result bit for bit; each such test
fails at a progress deadline instead of hanging.
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

from flashinfer.diffusion_ops import minimax_h3_sm120_varlen_attention_fp8
import flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention as fp8_ops
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention import (
    _STREAM_WORKSPACES,
    MINIMAX_H3_HEAD_DIM,
    MINIMAX_H3_NUM_HEADS,
    MiniMaxH3VarlenPlan,
    assign_unit_slots,
    normalize_cu_seqlens,
    workspace_bytes,
)
from flashinfer.utils import get_compute_capability

ATOL = 0.1
RTOL = 0.1
REL_L2_MAX = 0.08
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
@pytest.mark.parametrize("cu_seqlens", CU_SEQLENS)
@pytest.mark.parametrize("heads", [MINIMAX_H3_NUM_HEADS, 3])
def test_minimax_h3_sm120_varlen_attention_fp8_matches_fp32_oracle(
    cu_seqlens: tuple, heads: int
) -> None:
    if heads != MINIMAX_H3_NUM_HEADS and cu_seqlens[-1] > 1000:
        pytest.skip("small-head sweep covers the boundary shapes only")
    device = torch.device("cuda:0")
    tokens = cu_seqlens[-1]
    q, k, v = synthetic_inputs(tokens, heads, seed=tokens + heads, device=device)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    out = minimax_h3_sm120_varlen_attention_fp8(q, k, v, cu, cu_seqlens_host=cu_seqlens)
    torch.cuda.synchronize()
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    _check(out, fp32_oracle(q, k, v, cu_seqlens))
    # Idempotence: the quantized operands and scales are rebuilt every call, bit-exactly.
    again = minimax_h3_sm120_varlen_attention_fp8(
        q, k, v, cu, cu_seqlens_host=cu_seqlens
    )
    torch.cuda.synchronize()
    assert torch.equal(out, again)


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_fp8_preallocated_output_and_device_cu() -> (
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
    result = minimax_h3_sm120_varlen_attention_fp8(q, k, v, cu, out)
    torch.cuda.synchronize()
    assert result.data_ptr() == out.data_ptr()
    _check(out, fp32_oracle(q, k, v, cu_seqlens))


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_fp8_custom_scale_and_empty_stream() -> None:
    device = torch.device("cuda:0")
    cu_seqlens = (0, 200)
    q, k, v = synthetic_inputs(200, 4, seed=11, device=device)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    scale = 0.05
    out = minimax_h3_sm120_varlen_attention_fp8(
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
        minimax_h3_sm120_varlen_attention_fp8(empty, empty, empty, cu0).shape
        == empty.shape
    )


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_fp8_workspace_reuse_is_bitwise() -> None:
    # The launch arguments of a plan are sliced from its stream's workspace once; a larger plan grows
    # the workspace and drops them.  On a stream with a fresh workspace, the small plan called again
    # after the growth, and on a second stream (its own workspace and arrival counters), is bitwise
    # its first call; so is a call between the two plans' interleaved calls (the counters return to
    # zero after every call).
    device = torch.device("cuda:0")
    heads = 8
    small, large = (0, 133, 300, 900), (0, 257, 4567, 4824)
    q, k, v = synthetic_inputs(small[-1], heads, seed=5, device=device)
    ql, kl, vl = synthetic_inputs(large[-1], heads, seed=6, device=device)
    cu = torch.tensor(small, dtype=torch.int32, device=device)
    cul = torch.tensor(large, dtype=torch.int32, device=device)

    def on_fresh_stream(calls):
        stream = torch.cuda.Stream(device)
        key = (device.index, stream.cuda_stream)
        _STREAM_WORKSPACES.pop(key, None)
        stream.wait_stream(torch.cuda.current_stream(device))
        outs, sizes = [], []
        with torch.cuda.stream(stream):
            for args, bounds in calls:
                outs.append(
                    minimax_h3_sm120_varlen_attention_fp8(*args, cu_seqlens_host=bounds)
                )
                sizes.append(_STREAM_WORKSPACES[key].buffers["q8"].numel())
        torch.cuda.current_stream(device).wait_stream(stream)
        return outs, sizes

    small_call, large_call = ((q, k, v, cu), small), ((ql, kl, vl, cul), large)
    (first, large_first, again, large_again), sizes = on_fresh_stream(
        (small_call, large_call, small_call, large_call)
    )
    (on_side,), _ = on_fresh_stream((small_call,))
    torch.cuda.synchronize()
    # the large plan grew the workspace
    assert sizes[0] < sizes[1] == sizes[2] == sizes[3]
    assert torch.equal(first, again) and torch.equal(first, on_side)
    assert torch.equal(large_first, large_again)
    _check(first, fp32_oracle(q, k, v, small))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_minimax_h3_sm120_varlen_attention_fp8_rejects_unsupported_device() -> None:
    device = torch.device("cuda:0")
    if get_compute_capability(device) == (12, 0):
        pytest.skip("SM120 is supported; numerical tests cover this device")
    q, k, v = synthetic_inputs(1, 1, seed=0, device=device)
    cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
    with pytest.raises(RuntimeError, match=r"requires exact compute capability 12\.0"):
        minimax_h3_sm120_varlen_attention_fp8(q, k, v, cu, cu_seqlens_host=(0, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_minimax_h3_sm120_varlen_attention_fp8_rejects_bad_inputs() -> None:
    device = torch.device("cuda:0")
    q, k, v = synthetic_inputs(300, 4, seed=3, device=device)
    cu = torch.tensor([0, 300], dtype=torch.int32, device=device)
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_fp8(q.float(), k, v, cu)
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_fp8(q, k[:299], v, cu)
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_fp8(q, k, v, cu.to(torch.int64))
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_fp8(q, k, v, cu, cu_seqlens_host=(0, 299))
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_fp8(q, k, v, cu, cu_seqlens_host=(5, 300))
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_fp8(
            q, k, v, cu, cu_seqlens_host=(0, 200, 100, 300)
        )
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_fp8(
            q, k, v, cu, out=torch.empty_like(q)[:, :3]
        )


def test_minimax_h3_sm120_varlen_plan_covers_every_unit_once() -> None:
    bounds = normalize_cu_seqlens((0, 0, 129, 129, 500, 4824))
    heads = 5
    plan = MiniMaxH3VarlenPlan(bounds, heads, torch.device("cpu"), num_ctas=170)
    assert plan.num_segments == 5 and plan.total_tokens == 4824
    # Q tiles per segment: 0, 2, 0, 3, 34 -> 39 tiles, one unit per (non-empty tile, head).
    assert plan.num_tiles == 39 and plan.num_units == 39 * heads
    assert plan.grid == 170
    assert plan.seg_tile_begin.tolist() == [0, 0, 2, 2, 5, 39]
    table = plan.unit_table.view(-1, 2).tolist()
    units = {(seg, packed >> 16, packed & 0xFFFF) for seg, packed in table}
    assert len(units) == plan.num_units
    for seg, head, tile in units:
        assert 0 <= head < heads
        assert bounds[seg + 1] > bounds[seg]
        assert tile * 128 < bounds[seg + 1] - bounds[seg]
    # Slot order: the heaviest units land in the first full round.
    first_round = [seg for seg, _ in table[: plan.grid]]
    assert all(seg == 4 for seg in first_round)
    assert workspace_bytes(4824, heads, 5) > 0
    assert assign_unit_slots([], 8) == []
    assert sorted(assign_unit_slots([3, 1, 2, 5, 4], 2)) == [0, 1, 2, 3, 4]


def test_minimax_h3_sm120_varlen_plan_rejects_32_bit_offset_overflow() -> None:
    # The largest token count whose padded [tokens, heads, 128] extent fits a signed 32-bit offset.
    heads = MINIMAX_H3_NUM_HEADS
    row = heads * MINIMAX_H3_HEAD_DIM
    fits = ((1 << 31) - 1) // row // 128 * 128
    plan = MiniMaxH3VarlenPlan((0, fits), heads, torch.device("cpu"), num_ctas=170)
    assert plan.padded_tokens * row <= (1 << 31) - 1
    with pytest.raises(ValueError, match="32-bit offsets"):
        MiniMaxH3VarlenPlan((0, fits + 1), heads, torch.device("cpu"), num_ctas=170)


def test_minimax_h3_sm120_varlen_plan_rejects_32_bit_statistics_overflow() -> None:
    heads = MINIMAX_H3_NUM_HEADS
    limit = (1 << 31) - 1
    # Empty segments: no tokens, one [heads, 128] row of K means and V scales each.
    segments = limit // (heads * MINIMAX_H3_HEAD_DIM) + 1
    with pytest.raises(ValueError, match="statistics kernels' 32-bit offsets"):
        MiniMaxH3VarlenPlan(
            (0,) * (segments + 1), heads, torch.device("cpu"), num_ctas=170
        )
    # One-token segments: the tokens fit, but each segment is a query tile with
    # 2 * 128 partial floats per head.
    tiles = limit // (heads * 2 * MINIMAX_H3_HEAD_DIM) + 1
    with pytest.raises(ValueError, match="statistics kernels' 32-bit offsets"):
        MiniMaxH3VarlenPlan(
            tuple(range(tiles + 1)), heads, torch.device("cpu"), num_ctas=170
        )


# A plan of the overlap band (56 heads, attention work 257^2 + 4310^2 + 257^2): the overlapped form
# on 170 and 188 SMs.
OVERLAP_CU_SEQLENS = (0, 257, 4567, 4824)


@pytest.fixture
def fresh_caches(monkeypatch):
    """Fresh plan and workspace caches for one test (restored afterwards), so no launch form leaks
    between cases."""
    monkeypatch.setattr(fp8_ops, "_PLANS", OrderedDict())
    monkeypatch.setattr(fp8_ops, "_STREAM_WORKSPACES", OrderedDict())


def _skip_without_overlap(device: torch.device) -> None:
    if fp8_ops._sm_count(device.index) not in fp8_ops._OVERLAP_WIDTH:
        pytest.skip("the overlapped form needs a part of 170 or 188 SMs")


def _await(streams, what: str, seconds: float = 60.0) -> None:
    """Fail instead of hanging: the work queued so far on ``streams`` must complete within
    ``seconds``."""
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
    """Whether the zero-start words (arrival counters, head-control words) of every workspace owner
    in ``owners`` are zero again, as the kernels leave them after every complete call."""
    for owner in owners:
        buffers = fp8_ops._STREAM_WORKSPACES[owner].buffers
        for name in ("counters", "hctr"):
            if int(torch.count_nonzero(buffers[name].view(torch.int32))) != 0:
                return False
    return True


def _new_capture_owners(before) -> list:
    """The workspace owners created by graph captures since the owner set ``before`` was taken."""
    return [
        owner
        for owner in fp8_ops._STREAM_WORKSPACES
        if len(owner) == 3 and owner not in before
    ]


def _overlap_case(device: torch.device, seeds):
    """``cu_seqlens``, one input triple per seed and each triple's single-stream result."""
    cu = torch.tensor(OVERLAP_CU_SEQLENS, dtype=torch.int32, device=device)
    inputs = [
        synthetic_inputs(
            OVERLAP_CU_SEQLENS[-1], MINIMAX_H3_NUM_HEADS, seed=seed, device=device
        )
        for seed in seeds
    ]
    expected = [_overlap_call(q, k, v, cu) for q, k, v in inputs]
    torch.cuda.synchronize()
    return cu, inputs, expected


def _overlap_call(q, k, v, cu, out=None) -> torch.Tensor:
    return minimax_h3_sm120_varlen_attention_fp8(
        q, k, v, cu, out, cu_seqlens_host=OVERLAP_CU_SEQLENS
    )


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_fp8_overlapped_form_is_bitwise_sequential(
    fresh_caches, monkeypatch
) -> None:
    """A plan of the overlap band takes the overlapped form (the producers, the lead and the tail
    attention), leaves the head-control words zero after every call and matches the sequential
    launches bit for bit, its first call (with the programs' zero-work first run) included."""
    device = torch.device("cuda:0")
    _skip_without_overlap(device)
    monkeypatch.setattr(fp8_ops, "_OVERLAP_LOADED", set())
    cu = torch.tensor(OVERLAP_CU_SEQLENS, dtype=torch.int32, device=device)
    q, k, v = synthetic_inputs(
        OVERLAP_CU_SEQLENS[-1], MINIMAX_H3_NUM_HEADS, seed=21, device=device
    )
    plan = fp8_ops._plan_at(OVERLAP_CU_SEQLENS, MINIMAX_H3_NUM_HEADS, device.index)
    assert plan.overlap
    assert fp8_ops.minimax_h3_sm120_quant_varlen_attention_programs(plan) == (
        fp8_ops.PREPROCESS_HEAD_MAJOR,
        fp8_ops.ATTENTION_LEAD,
        fp8_ops.ATTENTION_TAIL,
    )
    overlapped = [_overlap_call(q, k, v, cu) for _call in range(3)]
    _await((torch.cuda.current_stream(),), "overlapped calls")
    owner = (device.index, torch.cuda.current_stream().cuda_stream)
    assert _control_words_are_zero([owner])
    monkeypatch.setattr(fp8_ops, "_OVERLAP_WIDTH", {})
    monkeypatch.setattr(fp8_ops, "_PLANS", OrderedDict())
    monkeypatch.setattr(fp8_ops, "_STREAM_WORKSPACES", OrderedDict())
    sequential = _overlap_call(q, k, v, cu)
    torch.cuda.synchronize()
    assert not fp8_ops._plan_at(
        OVERLAP_CU_SEQLENS, MINIMAX_H3_NUM_HEADS, device.index
    ).overlap
    for out in overlapped:
        assert torch.equal(out, sequential)
    _check(sequential, fp32_oracle(q, k, v, OVERLAP_CU_SEQLENS))


@requires_sm120
@pytest.mark.parametrize(
    "cu_seqlens",
    [
        # a short attention-weighted mean segment length: the light overlap width
        (0, 1536, 3072, 4608, 6144),
        # a long mean over more than 16384 padded tokens: the long overlap width
        (0, 8368, 16736, 25104, 33472),
    ],
)
def test_minimax_h3_sm120_varlen_attention_fp8_overlap_widths_are_bitwise_sequential(
    cu_seqlens, fresh_caches, monkeypatch
) -> None:
    """The plans whose overlap width is not the shipped 32 match the sequential launches bit for bit
    and leave the head-control words zero."""
    device = torch.device("cuda:0")
    _skip_without_overlap(device)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    q, k, v = synthetic_inputs(
        cu_seqlens[-1], MINIMAX_H3_NUM_HEADS, seed=23, device=device
    )
    plan = fp8_ops._plan_at(cu_seqlens, MINIMAX_H3_NUM_HEADS, device.index)
    assert plan.overlap
    grids = fp8_ops.minimax_h3_sm120_quant_varlen_attention_overlap_grids(
        fp8_ops._sm_count(device.index),
        num_units=plan.num_units,
        total_tokens=plan.total_tokens,
        work=plan.work,
        padded_tokens=plan.padded_tokens,
    )
    assert grids[2] > 32

    def call() -> torch.Tensor:
        return minimax_h3_sm120_varlen_attention_fp8(
            q, k, v, cu, cu_seqlens_host=cu_seqlens
        )

    overlapped = [call() for _call in range(2)]
    _await((torch.cuda.current_stream(),), "overlapped calls")
    owner = (device.index, torch.cuda.current_stream().cuda_stream)
    assert _control_words_are_zero([owner])
    monkeypatch.setattr(fp8_ops, "_OVERLAP_WIDTH", {})
    monkeypatch.setattr(fp8_ops, "_PLANS", OrderedDict())
    monkeypatch.setattr(fp8_ops, "_STREAM_WORKSPACES", OrderedDict())
    sequential = call()
    torch.cuda.synchronize()
    assert not fp8_ops._plan_at(cu_seqlens, MINIMAX_H3_NUM_HEADS, device.index).overlap
    for out in overlapped:
        assert torch.equal(out, sequential)


@requires_sm120
@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("num_streams", [2, 4])
def test_minimax_h3_sm120_varlen_attention_fp8_concurrent_streams_are_bitwise(
    overlap, num_streams, fresh_caches, monkeypatch
) -> None:
    """Streams running the operator concurrently on different inputs each reproduce their
    single-stream result bit for bit, and every round completes: the arrival counters and the
    head-control words are private to the (device, stream), and concurrent overlapped calls make
    progress.  ``overlap=False`` empties the overlap table, so the plan takes the sequential
    launches."""
    device = torch.device("cuda:0")
    if overlap:
        _skip_without_overlap(device)
    else:
        monkeypatch.setattr(fp8_ops, "_OVERLAP_WIDTH", {})
    cu, inputs, expected = _overlap_case(device, range(31, 31 + num_streams))
    streams = [torch.cuda.Stream(device=device) for _ in inputs]
    outs = [torch.zeros_like(q) for q, _k, _v in inputs]
    for _round in range(8):
        for stream, (q, k, v), out in zip(streams, inputs, outs, strict=True):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _overlap_call(q, k, v, cu, out)
        _await(streams, f"concurrent calls on {num_streams} streams")
        for out, want in zip(outs, expected, strict=True):
            assert torch.equal(out, want)
            out.zero_()
    assert _control_words_are_zero([(device.index, s.cuda_stream) for s in streams])


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_fp8_graphs_one_capture_stream(
    fresh_caches,
) -> None:
    """Two graphs captured one after the other on torch's default capture stream get separate
    workspace sets (their owners carry the capture id) and, replayed concurrently on two streams,
    each reproduce their eager result bit for bit."""
    device = torch.device("cuda:0")
    cu, inputs, expected = _overlap_case(device, (41, 42))
    outs = [torch.zeros_like(q) for q, _k, _v in inputs]
    before = set(fp8_ops._STREAM_WORKSPACES)
    graphs = []
    for (q, k, v), out in zip(inputs, outs, strict=True):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _overlap_call(q, k, v, cu, out)
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
def test_minimax_h3_sm120_varlen_attention_fp8_graph_beside_eager_calls(
    fresh_caches,
) -> None:
    """A graph captured on a stream, replayed on another stream concurrently with eager calls on
    its capture stream: the eager calls use the stream's workspace set, the graph its capture's,
    and both outputs are their single-stream results bit for bit."""
    device = torch.device("cuda:0")
    cu, inputs, expected = _overlap_case(device, (43, 44))
    (qg, kg, vg), (qe, ke, ve) = inputs
    capture_stream = torch.cuda.Stream(device=device)
    replay_stream = torch.cuda.Stream(device=device)
    out_graph, out_eager = torch.zeros_like(qg), torch.zeros_like(qe)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        _overlap_call(qg, kg, vg, cu, out_graph)
    streams = (capture_stream, replay_stream)
    for _round in range(16):
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(replay_stream):
            graph.replay()
        with torch.cuda.stream(capture_stream):
            _overlap_call(qe, ke, ve, cu, out_eager)
        _await(streams, "a graph replayed beside eager calls on its capture stream")
        assert torch.equal(out_graph, expected[0])
        assert torch.equal(out_eager, expected[1])
        out_graph.zero_()
        out_eager.zero_()
    assert _control_words_are_zero([(device.index, capture_stream.cuda_stream)])


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_fp8_two_threads_one_stream(
    fresh_caches,
) -> None:
    """Two host threads calling on one stream: the per-(device, stream) host lock keeps the
    launches of every call contiguous on the stream, so the calls take the stream's workspace set
    in turn and each thread's output is its single-stream result bit for bit."""
    device = torch.device("cuda:0")
    cu, inputs, expected = _overlap_case(device, (45, 46))
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
                    _overlap_call(q, k, v, cu, out)
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
def test_minimax_h3_sm120_varlen_attention_fp8_forked_capture_branches(
    fresh_caches,
) -> None:
    """One capture with the operator on two forked branches joined back into the capture stream:
    each branch gets its own workspace set (one capture id, two streams), the branches run
    concurrently at every replay and both outputs are their eager results bit for bit."""
    device = torch.device("cuda:0")
    cu, inputs, expected = _overlap_case(device, (47, 48))
    outs = [torch.zeros_like(q) for q, _k, _v in inputs]
    capture_stream = torch.cuda.Stream(device=device)
    branch = torch.cuda.Stream(device=device)
    before = set(fp8_ops._STREAM_WORKSPACES)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        branch.wait_stream(capture_stream)
        _overlap_call(*inputs[0], cu, outs[0])
        with torch.cuda.stream(branch):
            _overlap_call(*inputs[1], cu, outs[1])
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

    import flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention as ops

    bounds = (0, 257, 4567, 4824)
    device = torch.device("cuda:0")
    generator = torch.Generator(device="cpu").manual_seed(49)
    q, k, v = (
        torch.randn(bounds[-1], 56, 128, generator=generator)
        .to(torch.bfloat16)
        .to(device)
        for _ in range(3)
    )
    cu = torch.tensor(bounds, dtype=torch.int32, device=device)
    fn = ops.minimax_h3_sm120_varlen_attention_fp8
    first = fn(q, k, v, cu, cu_seqlens_host=bounds)
    again = fn(q, k, v, cu, cu_seqlens_host=bounds)
    torch.cuda.synchronize()
    ops._OVERLAP_WIDTH = {}
    ops._PLANS.clear()
    ops._STREAM_WORKSPACES.clear()
    sequential = fn(q, k, v, cu, cu_seqlens_host=bounds)
    torch.cuda.synchronize()
    assert torch.equal(first, sequential) and torch.equal(again, sequential)
    print("SERIALIZED_OK")
    """
)


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_fp8_serialized_launches(
    fresh_caches, monkeypatch
) -> None:
    """With ``CUDA_LAUNCH_BLOCKING=1`` every launch completes before the next one starts, so no
    programmatic dependent overlaps its primary: the overlapped form still completes (its producers
    run first and finish on their own) and matches the sequential launches bit for bit, the first
    call included.  Run in a subprocess with a deadline."""
    device = torch.device("cuda:0")
    _skip_without_overlap(device)
    # build the programs of both forms here first, so the subprocess only loads them
    cu, ((q, k, v),), _ = _overlap_case(device, (49,))
    monkeypatch.setattr(fp8_ops, "_OVERLAP_WIDTH", {})
    monkeypatch.setattr(fp8_ops, "_PLANS", OrderedDict())
    monkeypatch.setattr(fp8_ops, "_STREAM_WORKSPACES", OrderedDict())
    _overlap_call(q, k, v, cu)
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
    tail = result.stderr[-4000:]
    assert result.returncode == 0 and "SERIALIZED_OK" in result.stdout, tail
