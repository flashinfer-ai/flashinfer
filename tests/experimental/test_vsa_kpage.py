# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""KPage's source contracts exercised through FI's public plan/run interface."""

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


@pytest.fixture(autouse=True)
def supported_gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3)):
        pytest.skip("KPage requires SM100/SM103")
    pytest.importorskip("cutlass", minversion="4.6.2")


def inputs(block, *, gqa=1, tail=False, hkv=1):
    torch.manual_seed(42)
    m, n = 3 * block - (7 if tail else 0), 5 * block - (13 if tail else 0)
    hq = gqa * hkv
    q = torch.randn(m, hq, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [
        torch.randn(n, hkv, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    # Unsorted, odd row counts, ignored sentinel slots and an empty row.
    idx = torch.tensor(
        [[[[4, 0, 2], [1, -1, -1], [-1, -1, -1]]]], device="cuda", dtype=torch.int32
    )
    idx = idx.expand(1, hq, -1, -1).clone()
    if hq > 1:
        idx[0, 1, 0] = torch.tensor([1, 3, 0], device="cuda")
    counts = (
        torch.tensor([[[3, 1, 0]]], device="cuda", dtype=torch.int32)
        .expand(1, hq, -1)
        .contiguous()
    )
    kv_sizes = torch.tensor([block, 7, 17, 0, block], device="cuda", dtype=torch.int32)
    q_sizes = (
        torch.tensor([64, 65, 0], device="cuda", dtype=torch.int32)
        if block == 128
        else None
    )
    return q, k, v, idx, counts, kv_sizes, q_sizes


def wrapper(block):
    return flashinfer.BlockSparseAttentionWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend=f"vsa_sm100_blk{block}",
    )


def plan(w, block, data, *, scale=None, **kwargs):
    q, k, _, idx, counts, kv_sizes, q_sizes = data
    options = dict(
        indptr=None,
        indices=None,
        M=q.shape[0],
        N=k.shape[0],
        R=block,
        C=block,
        num_qo_heads=q.shape[1],
        num_kv_heads=k.shape[1],
        head_dim=128,
        q_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        q2k_indices=idx,
        q2k_num=counts,
        kv_block_lens=kv_sizes,
        q_block_lens=q_sizes,
        sm_scale=scale,
        kpage=True,
    )
    options.update(kwargs)
    w.plan(**options)


def reference(data, block, scale):
    q, k, v, idx, counts, kv_sizes, q_sizes = data
    hq = q.shape[1]
    if idx.ndim == 3:
        idx = idx.unsqueeze(0)
    if counts is None:
        counts = torch.full(idx.shape[:-1], idx.shape[-1], device=idx.device)
    elif counts.ndim == 2:
        counts = counts.unsqueeze(0)
    if kv_sizes is None:
        kv_sizes = torch.full(((k.shape[0] + block - 1) // block,), block)
    mask = torch.zeros(hq, q.shape[0], k.shape[0], device="cuda", dtype=torch.bool)
    for h in range(hq):
        for row in range(counts.shape[2]):
            for col in idx[0, h, row, : int(counts[0, h, row])].tolist():
                mask[
                    h,
                    row * block : min((row + 1) * block, q.shape[0]),
                    col * block : min(col * block + int(kv_sizes[col]), k.shape[0]),
                ] = True
    return dense_reference(q, k, v, mask, scale, query_valid_rows(data, block))


def query_valid_rows(data, block):
    q, *_, q_sizes = data
    rows = torch.arange(q.shape[0], device=q.device)
    return (
        torch.ones_like(rows, dtype=torch.bool)
        if q_sizes is None
        else rows % block < q_sizes[rows // block]
    )


def dense_reference(q, k, v, mask, scale, valid):
    # FP64 reference avoids inheriting the kernel's precision or TF32 settings.
    hq, hkv = q.shape[1], k.shape[1]
    key_heads = k.repeat_interleave(hq // hkv, dim=1).double()
    value_heads = v.repeat_interleave(hq // hkv, dim=1).double()
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.einsum("mhd,nhd->hmn", q.double(), key_heads) * scale
    scores.masked_fill_(~mask, -torch.inf)
    lse = scores.logsumexp(-1).T.contiguous()
    # Only empty rows get a zero probability vector; NaNs on valid rows must
    # remain visible to the assertions rather than being silently sanitized.
    empty = ~mask.any(-1)
    scores[empty] = 0
    probs = scores.softmax(-1).masked_fill(empty.unsqueeze(-1), 0)
    out = torch.einsum("hmn,nhd->mhd", probs, value_heads)
    out[~valid] = 0
    lse[~valid] = 0
    return out.float(), lse.float()


def assert_reference(actual, actual_lse, expected, expected_lse, valid):
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=1e-3, rtol=1e-3)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[~valid], torch.zeros_like(actual[~valid]))
    assert torch.equal(actual_lse[~valid], torch.zeros_like(actual_lse[~valid]))
    empty = torch.isneginf(expected_lse) & valid.unsqueeze(-1)
    assert torch.equal(actual[empty], torch.zeros_like(actual[empty]))
    assert torch.isneginf(actual_lse[empty]).all()


@pytest.mark.parametrize(
    "block,gqa,tail", [(64, 1, False), (128, 1, False), (128, 2, False), (128, 2, True)]
)
@pytest.mark.parametrize("scale", [128**-0.5, 0.0, -0.125])
def test_reference_reuse_and_output_buffers(block, gqa, tail, scale):
    data = inputs(block, gqa=gqa, tail=tail)
    w = wrapper(block)
    plan(w, block, data, scale=scale)
    layout = w._kpage_plan.layout
    q, k, v = data[:3]
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], device="cuda", dtype=torch.float32)
    for _ in range(2):
        q.normal_()  # Q/K/V values can change without rebuilding the layout.
        expected, expected_lse = reference(data, block, scale)
        actual, actual_lse = w.run(q, k, v, out=out, lse=lse, return_lse=True)
        assert actual is out and actual_lse is lse
        assert w._kpage_plan.layout is layout
        assert_reference(
            actual, actual_lse, expected, expected_lse, query_valid_rows(data, block)
        )


@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("kind", ["mask", "bsr"])
def test_sparse_input_conversion_and_switch_back(block, kind):
    data = list(inputs(block))
    data[-1] = None
    q, k, v = data[:3]
    w = wrapper(block)
    # Two full KV blocks, no padding, so ordinary VSA has the same semantics.
    mask = torch.zeros(q.shape[1], 3, 5, dtype=torch.bool, device="cuda")
    mask[..., :2] = True
    args = (
        dict(block_mask=mask)
        if kind == "mask"
        else dict(
            indptr=torch.tensor([0, 2, 4, 6], device="cuda", dtype=torch.int32),
            indices=torch.tensor([0, 1] * 3, device="cuda", dtype=torch.int32),
        )
    )
    base = dict(
        indptr=None,
        indices=None,
        M=q.shape[0],
        N=k.shape[0],
        R=block,
        C=block,
        num_qo_heads=q.shape[1],
        num_kv_heads=k.shape[1],
        head_dim=128,
        q_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
    )
    base.update(args)
    w.plan(**base, kpage=True)
    paged = w.run(q, k, v)
    w.plan(**base, kpage=False)
    assert w._kpage_plan is None
    ordinary = w.run(q, k, v)
    torch.testing.assert_close(paged, ordinary, atol=2e-2, rtol=2e-2)
    w.plan(**base, kpage=True)
    torch.testing.assert_close(w.run(q, k, v), paged, atol=0, rtol=0)


@pytest.mark.parametrize("block", [64, 128])
def test_invalid_metadata_and_failed_replan(block):
    data = inputs(block)
    w = wrapper(block)
    plan(w, block, data)
    data[3][0, 0, 0, 1] = data[3][0, 0, 0, 0]
    with pytest.raises(ValueError, match="duplicate"):
        plan(w, block, data)
    with pytest.raises(RuntimeError, match="successful plan"):
        w.run(*data[:3])
    data[3][0, 0, 0, 1] = 0
    plan(w, block, data)
    with pytest.raises(ValueError, match="scale"):
        w.run(*data[:3], scale_q=torch.ones(1, device="cuda"))
    with pytest.raises(ValueError, match="overlap"):
        w.run(*data[:3], out=data[0])


@pytest.mark.parametrize("backend", ["cake", "auto", "fa2"])
def test_reject_other_backends(backend):
    w = flashinfer.BlockSparseAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8, device="cuda"), backend=backend
    )
    with pytest.raises(ValueError, match="kpage=True requires"):
        w.plan(None, None, 128, 128, 128, 128, 1, 1, 128, kpage=True)


def test_reject_incompatible_options():
    data = inputs(64)
    w = wrapper(64)
    for kwargs in ({"kv_splits": 2}, {"use_clc": False}, {"causal": True}):
        with pytest.raises(ValueError):
            plan(w, 64, data, **kwargs)


@pytest.mark.parametrize("block", [64, 128])
def test_native_fi_shapes_and_cuda_graph(block):
    data = list(inputs(block))
    data[3], data[4] = data[3][0], data[4][0]
    w = wrapper(block)
    plan(w, block, data)
    q, k, v = data[:3]
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], device="cuda", dtype=torch.float32)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            w.run(q, k, v, out=out, lse=lse, return_lse=True)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(q, k, v, out=out, lse=lse, return_lse=True)
    for _ in range(2):
        for tensor in (q, k, v):
            tensor.normal_()
        graph.replay()
        captured, captured_lse = out.clone(), lse.clone()
        eager, eager_lse = w.run(q, k, v, return_lse=True)
        torch.testing.assert_close(captured, eager, atol=0, rtol=0)
        torch.testing.assert_close(captured_lse, eager_lse, atol=0, rtol=0)
        expected, expected_lse = reference(data, block, None)
        assert_reference(
            captured,
            captured_lse,
            expected,
            expected_lse,
            query_valid_rows(data, block),
        )


@pytest.mark.parametrize("tail_valid", [17, 64, 65, 121])
def test_blk128_active_query_tail(tail_valid):
    data = inputs(128, tail=True)
    data[-1][-1] = tail_valid
    data[3][0, 0, -1] = torch.tensor([4, 0, -1], device="cuda")
    data[4][0, 0, -1] = 2
    w = wrapper(128)
    plan(w, 128, data)
    actual, lse = w.run(*data[:3], return_lse=True)
    expected, expected_lse = reference(data, 128, None)
    assert torch.isfinite(expected_lse[256 : 256 + tail_valid]).all()
    assert_reference(actual, lse, expected, expected_lse, query_valid_rows(data, 128))


def test_blk128_gqa_multiple_kv_heads():
    data = inputs(128, gqa=2, hkv=2)
    data[2][:, 1].add_(2)  # A wrong KV-head mapping must change output visibly.
    data[3][0, :, 0] = torch.tensor(
        [[0, 2, 4], [1, 4, 2], [2, 0, 1], [4, 1, 0]], device="cuda"
    )
    w = wrapper(128)
    plan(w, 128, data)
    actual, lse = w.run(*data[:3], return_lse=True)
    expected, expected_lse = reference(data, 128, None)
    assert_reference(actual, lse, expected, expected_lse, query_valid_rows(data, 128))


@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("empty_kind", ["no_connections", "zero_valid_kv"])
def test_exact_empty_attention_and_query_padding(block, empty_kind):
    data = inputs(block)
    if empty_kind == "no_connections":
        data[4][0, 0, 1] = 0
    else:
        data[3][0, 0, 1] = torch.tensor([3, -1, -1], device="cuda")
        assert data[5][3] == 0
    w = wrapper(block)
    plan(w, block, data)
    q = data[0]
    out = torch.full_like(q, torch.nan)
    lse = torch.full(q.shape[:2], torch.nan, device="cuda", dtype=torch.float32)
    actual, actual_lse = w.run(*data[:3], out=out, lse=lse, return_lse=True)
    expected, expected_lse = reference(data, block, None)
    valid = query_valid_rows(data, block)
    assert (torch.isneginf(expected_lse) & valid.unsqueeze(-1)).any()
    if block == 128:
        assert (~valid).any()
    assert_reference(actual, actual_lse, expected, expected_lse, valid)


@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("kind", ["mask_q", "mask_kv", "bsr"])
def test_sparse_conversion_against_independent_reference(block, kind):
    data = list(inputs(block, hkv=2, gqa=2 if block == 128 else 1))
    data[-1] = None
    q, k, v = data[:3]
    hq, hkv = q.shape[1], k.shape[1]
    heads = 1 if kind == "bsr" else hkv if kind == "mask_kv" else hq
    pattern = torch.zeros(heads, 3, 5, dtype=torch.bool, device="cuda")
    for h in range(heads):
        for row, selected in enumerate(([0, 4], [2], [1, 3])):
            pattern[h, row, [(col + h) % 5 for col in selected]] = True
    if kind == "bsr":
        options = dict(
            indptr=torch.tensor([0, 2, 3, 5], device="cuda", dtype=torch.int32),
            indices=torch.tensor([0, 4, 2, 1, 3], device="cuda", dtype=torch.int32),
        )
        expected_pattern = pattern.expand(hq, -1, -1)
    else:
        options = dict(block_mask=pattern)
        expected_pattern = pattern.repeat_interleave(hq // heads, dim=0)
    # Expand the original boolean pattern directly; never call FI's sparse
    # converters or reconstruct the reference from their resulting indices.
    dense = expected_pattern.repeat_interleave(block, 1).repeat_interleave(block, 2)
    key_rows = torch.arange(k.shape[0], device="cuda")
    dense &= (key_rows % block < data[5][key_rows // block])[None, None, :]
    valid = query_valid_rows(data, block)
    expected, expected_lse = dense_reference(q, k, v, dense, None, valid)
    w = wrapper(block)
    plan(w, block, data, q2k_indices=None, q2k_num=None, **options)
    actual, lse = w.run(q, k, v, return_lse=True)
    assert_reference(actual, lse, expected, expected_lse, valid)


@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("defaults", ["counts", "lengths", "all"])
def test_default_metadata_and_scale(block, defaults):
    data = list(inputs(block, tail=block == 128))
    data[3] = torch.tensor(
        [[[[4, 0], [1, 2], [0, 4]]]], dtype=torch.int32, device="cuda"
    )
    data[4] = torch.full((1, 1, 3), 2, dtype=torch.int32, device="cuda")
    if defaults in ("counts", "all"):
        data[4] = None
    if defaults in ("lengths", "all"):
        data[5], data[6] = None, None
    w = wrapper(block)
    plan(w, block, data)  # sm_scale=None must use 1/sqrt(D).
    actual, lse = w.run(*data[:3], return_lse=True)
    expected, expected_lse = reference(data, block, None)
    assert_reference(actual, lse, expected, expected_lse, query_valid_rows(data, block))


@pytest.mark.parametrize("block", [64, 128])
def test_replan_changed_metadata_and_shape(block):
    data = inputs(block)
    w = wrapper(block)
    plan(w, block, data)
    previous = w.run(*data[:3]).clone()
    data[3][0, 0, 0] = torch.tensor([1, 2, 4], device="cuda")
    data[5][1] = block
    if data[6] is not None:
        data[6][0] = 16
    plan(w, block, data)
    actual, lse = w.run(*data[:3], return_lse=True)
    expected, expected_lse = reference(data, block, None)
    assert not torch.allclose(previous, actual)
    assert_reference(actual, lse, expected, expected_lse, query_valid_rows(data, block))
    m, n = (
        4 * block - (5 if block == 128 else 0),
        3 * block - (3 if block == 128 else 0),
    )
    q = torch.randn(m, 1, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [
        torch.randn(n, 1, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    idx = torch.tensor(
        [[[[0, 2], [1, 0], [2, 1], [1, 2]]]], dtype=torch.int32, device="cuda"
    )
    changed = (q, k, v, idx, None, None, None)
    plan(w, block, changed)
    actual, lse = w.run(q, k, v, return_lse=True)
    expected, expected_lse = reference(changed, block, None)
    assert_reference(
        actual, lse, expected, expected_lse, query_valid_rows(changed, block)
    )
