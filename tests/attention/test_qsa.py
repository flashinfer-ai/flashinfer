# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Packed QSA numerics, validation and changed-selection graph replay."""

import sys
from types import SimpleNamespace

import pytest
import torch

import flashinfer
from flashinfer import qsa


def _inputs(lengths=(3, 5, 9), heads=12, device="cuda"):
    total = sum(lengths)
    torch.manual_seed(42)
    data = {
        "q": torch.randn(total, heads, 256, device=device, dtype=torch.bfloat16),
        "k": torch.randn(total, heads // 12, 256, device=device, dtype=torch.bfloat16),
        "v": torch.randn(total, heads // 12, 256, device=device, dtype=torch.bfloat16),
    }
    starts, ends, token_bases, block_bases = [], [], [], []
    offset = 0
    for length in lengths:
        token_bases.extend([offset] * length)
        block_bases.extend([len(starts)] * length)
        for start in range(offset, offset + length, 4):
            starts.append(start)
            ends.append(min(start + 3, offset + length - 1))
        offset += length
    for name, values in {
        "lo": token_bases,
        "hi": list(range(total)),
        "tok_base": token_bases,
        "blk_base": block_bases,
        "block_first": starts,
        "block_last": ends,
    }.items():
        data[name] = torch.tensor(values, device=device, dtype=torch.int32)
    selected = torch.randint(
        0, 2, (total, len(starts)), device=device, dtype=torch.uint8
    )
    selected[0].zero_()
    data["sel_blocks"] = selected
    return data


def _allowed(data):
    total = data["q"].shape[0]
    token = torch.arange(total, device=data["q"].device)
    block = (
        data["blk_base"][:, None] + (token[None, :] - data["tok_base"][:, None]) // 4
    )
    return (
        (token[None, :] >= data["lo"][:, None])
        & (token[None, :] <= data["hi"][:, None])
        & data["sel_blocks"]
        .gather(1, block.clamp(0, data["sel_blocks"].shape[1] - 1).long())
        .bool()
    )


def _indices(data):
    tokens = data["block_first"][None, :, None] + torch.arange(
        4, device=data["q"].device, dtype=torch.int32
    )
    valid = (
        data["sel_blocks"][:, :, None].bool()
        & (tokens <= data["block_last"][None, :, None])
        & (tokens >= data["lo"][:, None, None])
        & (tokens <= data["hi"][:, None, None])
    )
    return torch.where(valid, tokens, -1).flatten(1).contiguous()


def _reference(data, scale):
    q, k, v = (data[name].float() for name in ("q", "k", "v"))
    k, v = (x.repeat_interleave(q.shape[1] // k.shape[1], dim=1) for x in (k, v))
    scores = torch.einsum("thd,shd->hts", q, k) * scale
    scores.masked_fill_(~_allowed(data)[None, :, :], -torch.inf)
    lse = scores.logsumexp(-1)
    probs = scores.softmax(-1).nan_to_num()
    out32 = torch.einsum("hts,shd->thd", probs, v)
    return out32.to(data["q"].dtype), out32, lse


@pytest.fixture
def supported_gpu():
    if not torch.cuda.is_available():
        pytest.skip("QSA requires CUDA")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("QSA requires SM100, SM103 or SM107")


@pytest.mark.parametrize("heads", [12, 24])
@pytest.mark.parametrize("scale", [1 / 16, 0.0, -0.1])
# FE 1.32 selects its gated bitmap launcher above 512 tokens.
@pytest.mark.parametrize("lengths", [(3, 5, 9), (65, 129), (255, 257), (255, 258)])
def test_bitmap_forward(supported_gpu, heads, scale, lengths):
    data = _inputs(lengths=lengths, heads=heads)
    actual = flashinfer.qsa_forward(**data, scale=scale)
    for result, expected in zip(actual, _reference(data, scale), strict=True):
        torch.testing.assert_close(result, expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("lengths", [(3, 5, 9), (65, 129)])
def test_indexed_forward(supported_gpu, lengths):
    if torch.cuda.get_device_capability() == (10, 7):
        pytest.skip("Indexed QSA requires SM100 or SM103")
    data = _inputs(lengths=lengths)
    actual = flashinfer.qsa_indexed_forward(
        data["q"], data["k"], data["v"], _indices(data)
    )
    for result, expected in zip(actual, _reference(data, 1 / 16), strict=True):
        torch.testing.assert_close(result, expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("indexed", [False, True])
def test_changed_selection_replay(supported_gpu, indexed):
    if indexed and torch.cuda.get_device_capability() == (10, 7):
        pytest.skip("Indexed QSA requires SM100 or SM103")
    data = _inputs()
    indices = _indices(data)

    def run():
        if indexed:
            return flashinfer.qsa_indexed_forward(
                data["q"], data["k"], data["v"], indices
            )
        return flashinfer.qsa_forward(**data)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run()
    graph.replay()
    for actual, reference in zip(captured, expected, strict=True):
        torch.testing.assert_close(actual, reference)
    if indexed:
        indices.fill_(-1)
    else:
        data["sel_blocks"].zero_()
    graph.replay()
    assert torch.count_nonzero(captured[0]) == 0
    assert torch.count_nonzero(captured[1]) == 0
    assert torch.isneginf(captured[2]).all()


def test_invalid_inputs(supported_gpu):
    data = _inputs()
    with pytest.raises(ValueError, match="finite"):
        flashinfer.qsa_forward(**data, scale=float("nan"))
    with pytest.raises((ValueError, NotImplementedError), match="block_size"):
        flashinfer.qsa_forward(**data, block_size=8)
    with pytest.raises(NotImplementedError, match="BF16"):
        flashinfer.qsa_forward(**{**data, "q": data["q"].float()})


@pytest.mark.parametrize("version", ["1.31.0", "1.32.0.dev1"])
def test_frontend_version_gate(monkeypatch, version):
    qsa._cudnn_ops.cache_clear()
    monkeypatch.setitem(sys.modules, "cudnn", SimpleNamespace(__version__=version))
    try:
        with pytest.raises(RuntimeError, match="1.32.0"):
            qsa._cudnn_ops()
    finally:
        qsa._cudnn_ops.cache_clear()


def test_trace_contracts():
    data = _inputs(device="cpu")
    bitmap = flashinfer.qsa_forward.fi_trace(**data)
    indexed = flashinfer.qsa_indexed_forward.fi_trace(
        q=data["q"], k=data["k"], v=data["v"], indices=_indices(data)
    )
    for trace in (bitmap, indexed):
        assert trace["outputs"]["lse"]["shape"] == ["num_q_heads", "num_tokens"]
        assert trace["outputs"]["out_fp32"]["dtype"] == "float32"
        assert trace["axes"]["head_dim"]["value"] == 256
