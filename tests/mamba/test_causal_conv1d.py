# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project

import pytest
import torch
import torch.nn.functional as F

from flashinfer.mamba import causal_conv1d_fn, causal_conv1d_update

from .triton_reference.causal_conv1d_triton import (
    causal_conv1d_fn as triton_prefill,
    causal_conv1d_update as triton_update,
)


@pytest.fixture(autouse=True)
def _require_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("inference causal conv1d targets SM100/SM103")
    pytest.importorskip("cutlass.cute")
    torch.manual_seed(7967)


def _state(slots, channels, history, layout, padding=0):
    padding = 16 if layout == "padded" else padding
    stride = channels * history + padding
    backing = torch.full((slots, stride), -7, device="cuda", dtype=torch.bfloat16)
    strides = (stride, 1, channels) if layout == "sd" else (stride, history, 1)
    state = backing.as_strided((slots, channels, history), strides)
    state.copy_(torch.randn(state.shape, device="cuda", dtype=torch.bfloat16) * 0.25)
    return state


def _offsets(lengths):
    return torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )


def _reference(
    x,
    w,
    state,
    lengths,
    slots,
    *,
    prefill=False,
    initial=None,
    accepted=None,
    bias=None,
):
    raw, weights, pool = x.cpu(), w.cpu().double(), state.cpu().clone()
    result = torch.zeros_like(raw)
    b = (
        torch.zeros(w.shape[0], dtype=torch.float64)
        if bias is None
        else bias.cpu().double()
    )
    start = 0
    for i, (length, slot) in enumerate(zip(lengths, slots, strict=True)):
        stop = start + length
        if slot == -1 or length == 0 or (accepted is not None and accepted[i] == 0):
            start = stop
            continue
        offset = 0 if accepted is None else accepted[i] - 1
        history = pool[slot, :, offset : offset + 3].clone()
        if prefill and (initial is None or not initial[i]):
            history.zero_()
        extended = torch.cat((history, raw[start:stop].T), dim=1)
        for t in range(length):
            value = (extended[:, t : t + 4].double() * weights).sum(1) + b
            result[start + t] = F.silu(value).bfloat16()
        if accepted is None:
            pool[slot, :, :3] = extended[:, -3:]
        else:
            pool[slot, :, : length + 2] = extended[:, 1:]
        start = stop
    return result.to(x.device), pool.to(state.device)


def _assert_result(out, state, expected):
    torch.testing.assert_close(out, expected[0], atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(
        state.view(torch.int16), expected[1].view(torch.int16), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "layout,channels,weight_dtype,bias_dtype,offset",
    [
        ("sd", 129, torch.float32, None, 0),
        ("ds", 512, torch.bfloat16, torch.float32, 0),
        ("padded", 129, torch.float32, torch.bfloat16, 0),
        ("sd", 4096, torch.float32, None, 0),
        ("sd", 129, torch.float32, None, 1),
        ("ds", 129, torch.bfloat16, None, 1),
    ],
)
def test_prefill_matches_references(layout, channels, weight_dtype, bias_dtype, offset):
    lengths = [1, 2, 3, 17, 1024 if channels == 4096 else 35]
    slot_ids, initial = [0, 3, 1, 5, 4], [True, False, True, False, True]
    if layout == "padded":
        initial = [False] * len(lengths)
    x = torch.randn(sum(lengths), channels, device="cuda", dtype=torch.bfloat16) * 0.2
    w = torch.randn(channels * 4 + offset, device="cuda", dtype=weight_dtype)[
        offset:
    ].view(channels, 4)
    w.mul_(0.2)
    bias = (
        None
        if bias_dtype is None
        else torch.randn(channels, device="cuda", dtype=bias_dtype) * 0.1
    )
    state = _state(7, channels, 3, layout)
    reference_state = state.clone()
    expected = _reference(
        x, w, state, lengths, slot_ids, prefill=True, initial=initial, bias=bias
    )
    slots = torch.tensor(slot_ids, device="cuda", dtype=torch.int32)
    mask, offsets = torch.tensor(initial, device="cuda"), _offsets(lengths)
    reference = triton_prefill(
        x.T,
        w,
        bias,
        reference_state,
        offsets,
        lengths,
        cache_indices=slots,
        has_initial_state=mask,
        activation="silu",
    )
    out = causal_conv1d_fn(
        x, w, state, offsets, slots, None if layout == "padded" else mask, bias=bias
    )
    _assert_result(out, state, expected)
    _assert_result(reference.T, reference_state, expected)


@pytest.mark.parametrize("layout", ["sd", "ds"])
@pytest.mark.parametrize(
    "tokens,speculative,channels",
    [
        (1, False, 4096),
        (5, False, 129),
        (1, True, 4096),
        (5, True, 256),
        (8, True, 129),
        (9, True, 129),
        (16, True, 256),
        (17, True, 129),
    ],
)
def test_update_matches_references(layout, tokens, speculative, channels):
    x = torch.randn(4, tokens, channels, device="cuda", dtype=torch.bfloat16) * 0.2
    weight_dtype = torch.bfloat16 if speculative and tokens == 5 else torch.float32
    w = torch.randn(channels, 4, device="cuda", dtype=weight_dtype) * 0.2
    bias = None if channels == 4096 else torch.randn(channels, device="cuda") * 0.1
    state = _state(6, channels, tokens + 2, layout)
    reference_state = state.clone()
    slot_ids = [3, 0, 4, 1]
    slots = torch.tensor(slot_ids, device="cuda", dtype=torch.int32)
    accepted = torch.ones(4, device="cuda", dtype=torch.int32) if speculative else None
    offsets = None if tokens == 1 else _offsets([tokens] * 4)
    for step in range(3):
        counts = [1, tokens, 1, max(1, tokens - 1)] if step else [1] * 4
        if speculative:
            accepted.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        x.mul_(0.75)
        expected = _reference(
            x.flatten(0, 1),
            w,
            state,
            [tokens] * 4,
            slot_ids,
            accepted=counts if speculative else None,
            bias=bias,
        )
        reference = triton_update(
            x.transpose(1, 2),
            reference_state,
            w,
            bias=bias,
            activation="silu",
            conv_state_indices=slots,
            num_accepted_tokens=accepted,
        )
        out = causal_conv1d_update(
            x.flatten(0, 1),
            w,
            state,
            slots,
            cu_seqlens=offsets,
            num_accepted_tokens=accepted,
            bias=bias,
        )
        _assert_result(out, state, expected)
        _assert_result(
            reference.transpose(1, 2).flatten(0, 1), reference_state, expected
        )


def test_prefill_chunking_and_decode_are_identical():
    x = torch.randn(43, 65, device="cuda", dtype=torch.bfloat16) * 0.1
    x[0, 0], x[-1, 0] = -0.0, -0.0
    w = torch.randn(65, 4, device="cuda") * 0.1
    full = _state(2, 65, 3, "padded")
    split, decode = full.clone(), full.clone()
    slots = torch.tensor([0], device="cuda", dtype=torch.int32)
    mask = torch.ones(1, device="cuda", dtype=torch.bool)
    expected = causal_conv1d_fn(x, w, full, _offsets([43]), slots, mask)
    pieces, start = [], 0
    for count in (1, 2, 17, 23):
        pieces.append(
            causal_conv1d_fn(
                x[start : start + count], w, split, _offsets([count]), slots, mask
            )
        )
        start += count
    steps = [causal_conv1d_update(row.unsqueeze(0), w, decode, slots) for row in x]
    for out, pool in ((torch.cat(pieces), split), (torch.cat(steps), decode)):
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            pool.view(torch.int16), full.view(torch.int16), atol=0, rtol=0
        )


@pytest.mark.parametrize(
    "kind,n,channels,layout,padding,strided",
    [
        ("prefill", 1, 256, "sd", 0, True),
        ("prefill", 3, 129, "ds", 1, True),
        ("prefill", 32, 24, "sd", 0, False),
        ("prefill", 33, 256, "sd", 8, False),
        ("prefill", 97, 24, "ds", 0, False),
        ("spec", 3, 256, "sd", 8, True),
        ("spec", 3, 129, "ds", 1, False),
        ("spec", 3, 128, "sd", 1, False),
        ("decode", 3, 129, "sd", 0, True),
    ],
)
def test_graph_replay_updates_live_state(kind, n, channels, layout, padding, strided):
    prefill, speculative = kind == "prefill", kind == "spec"
    if n == 1:
        lengths = [35]
    elif n >= 32:
        lengths = [32] * n
    elif kind == "decode":
        lengths = [1] * n
    else:
        lengths = [1, 0, 16]
    rows = sum(lengths)
    history = 18 if speculative else 7 if prefill else 3
    backing = (
        torch.randn(
            rows, channels + (23 if strided else 0), device="cuda", dtype=torch.bfloat16
        )
        * 0.1
    )
    x = backing[:, 5 : 5 + channels] if strided else backing
    target = torch.full(
        (rows, channels + (13 if strided else 0)),
        -11,
        device="cuda",
        dtype=torch.bfloat16,
    )
    out = target[:, :channels]
    w = torch.randn(channels, 4, device="cuda") * 0.1
    state = _state(n + 3, channels, history, layout, padding)
    offsets, slots = (
        _offsets(lengths),
        torch.arange(n, device="cuda", dtype=torch.int32),
    )
    mask = torch.ones(n, device="cuda", dtype=torch.bool)
    accepted = torch.ones(n, device="cuda", dtype=torch.int32) if speculative else None
    workspace = torch.full((n + 4,), -123, device="cuda", dtype=torch.int32)

    def run(pool, output):
        if prefill:
            return causal_conv1d_fn(
                x, w, pool, offsets, slots, mask, out=output, workspace=workspace
            )
        return causal_conv1d_update(
            x,
            w,
            pool,
            slots,
            cu_seqlens=None if kind == "decode" else offsets,
            num_accepted_tokens=accepted,
            out=output,
        )

    assert run(state, out).data_ptr() == out.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(state, out)
    oracle = state.clone()
    if n >= 32:
        variants = [[1] * (n - 1) + [rows - n + 1], [0] * (n - 1) + [rows], lengths]
    elif n == 1 or kind == "decode":
        variants = [lengths] * 3
    else:
        variants = [[5, 7, 5], [0, 16, 1], [1, 0, 16]]
    for step, lengths in enumerate(variants):
        slot_ids = [(i + step) % (n + 3) for i in range(n)]
        if step == 1:
            slot_ids[0] = -1
        counts = None
        if speculative:
            slot_ids, counts = (
                ([0, 1, 2], [1, 0, 16]),
                ([-1, 3, 2], [0, 2, 5]),
                ([4, 0, -1], [3, 1, 1]),
            )[step]
            accepted.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        offsets.copy_(_offsets(lengths))
        slots.copy_(torch.tensor(slot_ids, device="cuda", dtype=torch.int32))
        mask.logical_not_()
        x.mul_(0.75)
        w.mul_(0.875)
        input_bits = backing.clone().view(torch.int16)
        expected = _reference(
            x,
            w,
            oracle,
            lengths,
            slot_ids,
            prefill=prefill,
            initial=mask.cpu().tolist(),
            accepted=counts,
        )
        eager_state, eager_out = oracle.clone(), torch.empty_like(out)
        run(eager_state, eager_out)
        workspace.fill_(-123)
        graph.replay()
        _assert_result(out, state, expected)
        torch.testing.assert_close(
            out.view(torch.int16), eager_out.view(torch.int16), atol=0, rtol=0
        )
        torch.testing.assert_close(
            backing.view(torch.int16), input_bits, atol=0, rtol=0
        )
        assert torch.all(workspace[-3:] == -123)
        assert torch.all(target[:, channels:] == -11)
        if padding:
            tail = state.as_strided(
                (n + 3, padding), (state.stride(0), 1), channels * history
            )
            assert torch.all(tail == -7)
        oracle.copy_(expected[1])


def test_empty_calls_output_lifetime_and_aliasing():
    state = _state(2, 32, 3, "ds")
    original = state.clone()
    w = torch.randn(32, 4, device="cuda")
    slots = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    empty = torch.empty(0, 32, device="cuda", dtype=torch.bfloat16)
    offsets = _offsets([0, 0])
    causal_conv1d_fn(empty, w, state, offsets, slots)
    causal_conv1d_update(empty, w, state, slots, cu_seqlens=offsets)
    torch.testing.assert_close(
        state.view(torch.int16), original.view(torch.int16), atol=0, rtol=0
    )
    x = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
    first = causal_conv1d_update(x, w, state, slots)
    snapshot = first.clone()
    second = causal_conv1d_update(x, w, state, slots)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, snapshot, atol=0, rtol=0)
    with pytest.raises(ValueError, match="share storage"):
        causal_conv1d_update(x, w, state, slots, out=x)
    offsets = _offsets([1, 1])
    with pytest.raises(ValueError, match="share storage"):
        causal_conv1d_fn(x, w, state, offsets, slots, workspace=offsets)
