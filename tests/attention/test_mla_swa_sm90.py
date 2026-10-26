# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

import flashinfer
from flashinfer.mla import batch_decode_with_kv_cache_mla_swa as run
from flashinfer.mla import get_mla_swa_workspace_size_bytes
from flashinfer.mla._swa import _runner

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 required",
)


def inputs(b=2, sq=4, context=1055):
    torch.manual_seed(123)
    pages = math.ceil(context / 64)
    q = torch.randn(b, sq, 64, 1088, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(b * pages, 64, 1088, device="cuda", dtype=torch.bfloat16)
    table = torch.arange(b * pages, device="cuda", dtype=torch.int32).reshape(b, pages)
    lengths = torch.full((b,), context, device="cuda", dtype=torch.int32)
    workspace = torch.empty(
        get_mla_swa_workspace_size_bytes(b, sq), device="cuda", dtype=torch.uint8
    )
    out = torch.empty(b, sq, 64, 1024, device="cuda", dtype=torch.bfloat16)
    return [q, kv, workspace, table, lengths, out]


def reference(args):
    q, kv, _, table, lengths, _ = args
    result = torch.zeros((*q.shape[:-1], 1024), device=q.device, dtype=torch.float64)
    weighted_abs = torch.zeros_like(result)
    for b in range(q.shape[0]):
        length = int(lengths[b])
        cache = kv[table[b].long()].reshape(-1, 1088).double()
        for i in range(q.shape[1]):
            pos = length - q.shape[1] + i
            if pos < 0:
                continue
            k = cache[max(0, pos - 512) : pos + 1]
            p = torch.softmax(q[b, i].double() @ k.T * 0.0625, dim=-1)
            result[b, i] = p @ k[:, :1024]
            weighted_abs[b, i] = p @ k[:, :1024].abs()
    return result, weighted_abs


def assert_accuracy(actual, args):
    gold, weighted_abs = reference(args)
    actual = actual.double()
    assert torch.isfinite(actual).all()
    diff = actual - gold
    norm = gold.norm(dim=-1)
    floor = 1e-5 * 1024**0.5
    assert (diff.norm(dim=-1) / norm.clamp_min(floor)).max().item() <= 2 * 2**-8
    assert (diff.abs() / (4 * 2**-8 * weighted_abs + 1e-5)).max().item() <= 1
    nonzero = norm > floor
    if nonzero.any():
        assert (
            torch.nn.functional.cosine_similarity(
                actual[nonzero], gold[nonzero], dim=-1
            )
            .min()
            .item()
            >= 0.99999
        )


@pytest.mark.parametrize("tactic", range(6))
@pytest.mark.parametrize("sq", [1, 4])
@pytest.mark.parametrize("offset", [0, 1, 31, 63])
def test_correctness(tactic, sq, offset):
    args = inputs(sq=sq, context=1024 + offset)
    _runner(0.0625, 512)(args, tactic=tactic)
    assert_accuracy(args[-1], args)


@pytest.mark.parametrize("tactic", range(6))
@pytest.mark.parametrize("sq", [1, 4])
@pytest.mark.parametrize("context", [63, 65, 191, 255, 511, 1024, 1025, 1055, 1087])
def test_nonfinite_masked_cache_slots(tactic, sq, context):
    args = inputs(sq=sq, context=context)
    kv, table = args[1], args[3]
    tokens = torch.arange(table.shape[1] * 64, device=kv.device).reshape(-1, 64)
    poison = torch.tensor(
        [float("nan"), float("inf"), -float("inf")],
        device=kv.device,
        dtype=kv.dtype,
    ).repeat(363)[:1088]
    for b in range(table.shape[0]):
        # Poison only slots excluded by every query in the request. Valid
        # K/V values must remain finite; the last sq tokens are not padding.
        left = max(context - sq - 512, 0)
        masked = (tokens < left) | (tokens >= context)
        page_values = kv[table[b].long()].clone()
        page_values[masked] = poison
        kv[table[b].long()] = page_values
    original = kv.clone()
    _runner(0.0625, 512)(args, tactic=tactic)
    assert_accuracy(args[-1], args)
    torch.testing.assert_close(kv, original, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("b", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("sq", [1, 4])
def test_graph_and_dummy_rows(b, sq):
    args = inputs(b=b, sq=sq, context=1055)
    if b > 1:
        args[4][-1] = 0
    run(*args[:-1], window_left=512, sm_scale=0.0625, out=args[-1])
    expected = args[-1].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(*args[:-1], window_left=512, sm_scale=0.0625, out=args[-1])
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(args[-1], expected, rtol=0, atol=0)
    assert_accuracy(args[-1], args)
    if b > 1:
        assert torch.count_nonzero(args[-1][-1]) == 0


def test_autotune(tmp_path):
    args = inputs(b=1, sq=1)
    metadata = args[3].clone(), args[4].clone()
    cache = tmp_path / "mla_swa.json"
    with flashinfer.autotune(True, cache=str(cache)):
        run(*args[:-1], window_left=512, sm_scale=0.0625, out=args[-1])
    assert cache.is_file()
    assert "mla_swa_sm90" in cache.read_text()
    torch.testing.assert_close(args[3], metadata[0], rtol=0, atol=0)
    torch.testing.assert_close(args[4], metadata[1], rtol=0, atol=0)
    assert_accuracy(args[-1], args)


def test_bad_workspace_and_alias():
    args = inputs()
    with pytest.raises(ValueError, match="workspace"):
        run(
            args[0],
            args[1],
            args[2][:256],
            args[3],
            args[4],
            window_left=512,
            sm_scale=0.0625,
        )
    with pytest.raises(ValueError, match="window_left"):
        run(*args[:-1], window_left=511, sm_scale=0.0625)
    alias = args[0].view(-1)[: args[-1].numel()].view(args[-1].shape)
    with pytest.raises(ValueError, match="overlap"):
        run(*args[:-1], window_left=512, sm_scale=0.0625, out=alias)


@pytest.mark.parametrize("tactic", range(6))
def test_short_and_empty(tactic):
    args = inputs(b=2, sq=4, context=63)
    args[4][-1] = 0
    _runner(0.0625, 512)(args, tactic=tactic)
    assert_accuracy(args[-1], args)


@pytest.mark.parametrize("tactic", range(6))
def test_large_scores(tactic):
    args = inputs(b=2, sq=4, context=1055)
    args[0].mul_(3)
    args[1].mul_(3)
    _runner(0.0625, 512)(args, tactic=tactic)
    assert_accuracy(args[-1], args)


def test_nondefault_stream_strides_and_updated_metadata():
    args = inputs(b=2, sq=4)
    q = torch.empty(2, 5, 64, 1088, dtype=torch.bfloat16, device="cuda")[:, :4]
    q.copy_(args[0])
    args[0] = q
    table = torch.empty(2, args[3].shape[1] + 3, dtype=torch.int32, device="cuda")[
        :, : args[3].shape[1]
    ]
    table.copy_(args[3])
    args[3] = table
    run(*args[:-1], window_left=512, sm_scale=0.0625, out=args[-1])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(*args[:-1], window_left=512, sm_scale=0.0625, out=args[-1])
        args[4].fill_(63)
        graph.replay()
    stream.synchronize()
    assert_accuracy(args[-1], args)
