# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 by FlashInfer team.
"""Accuracy and public-contract checks for native Prims-TS sparse MLA.

The test-only preparer adapts vLLM's page-stride mapping and stable compaction
(0aee727ff6131a1b647941186ceef8f4777bdac2); native Q/KV are never repacked.
"""

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")
triton = pytest.importorskip("triton")
import triton.language as tl
from flashinfer.attention.prims_ts import (
    BatchSparseMLADecodePagedTSWrapper,
    batch_sparse_mla_decode_with_paged_kv_cache,
    get_prims_ts_sparse_mla_decode_workspace_size,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


# Reference: FP64 math, explicit masks, and an unchanged FP8 rounding bound.
def _reference(
    query,
    kv_cache,
    extra_kv_cache,
    indices,
    extra_indices=None,
    *,
    lengths=None,
    extra_lengths=None,
    q_scale=1.0,
    kv_scale=1.0,
    output_scale=1.0,
    attention_sinks=None,
    softmax_scale=512**-0.5,
):
    rows, heads = query.numel() // (query.shape[-2] * 512), query.shape[-2]
    q = query.reshape(rows, heads, 512)
    output = torch.empty((rows, heads, 512), device=query.device, dtype=torch.float64)
    lse = torch.empty((rows, heads), device=query.device, dtype=torch.float64)
    bound = torch.empty_like(output)
    for start in range(0, rows, 32):
        end = min(start + 32, rows)
        values, masks = [], []
        for cache, ids, lens in (
            (kv_cache, indices, lengths),
            (extra_kv_cache, extra_indices, extra_lengths),
        ):
            if cache is None:
                continue
            ids = ids.reshape(rows, -1)[start:end]
            valid = ids >= 0
            if lens is not None:
                valid &= (
                    torch.arange(ids.shape[-1], device=query.device)[None, :]
                    < lens.reshape(-1)[start:end, None]
                )
            safe = ids.masked_fill(~valid, 0).long()
            value = (
                cache[safe // cache.shape[1], safe % cache.shape[1]].double() * kv_scale
            )
            values.append(value.masked_fill(~valid[..., None], 0))
            masks.append(valid)
        kv, valid = torch.cat(values, dim=1), torch.cat(masks, dim=1)
        empty = ~valid.any(-1)
        logits = (
            torch.bmm(q[start:end].double() * q_scale, kv.transpose(1, 2))
            * softmax_scale
        )
        logits.masked_fill_(~valid[:, None, :], -torch.inf)
        row_lse = logits.logsumexp(-1)
        normalizer = (
            row_lse
            if attention_sinks is None
            else torch.logaddexp(row_lse, attention_sinks.double()[None, :])
        )
        normalizer = normalizer.masked_fill(empty[:, None], 0)
        weights = (logits - normalizer[..., None]).exp()
        output[start:end] = torch.bmm(weights, kv) * output_scale
        lse[start:end] = row_lse
        absolute = kv.abs()
        sensitivity = torch.bmm(weights, absolute)
        underflow = (logits.amax(-1) - normalizer).exp()[..., None] * absolute.sum(1)[
            :, None, :
        ]
        bound[start:end] = (
            output_scale
            * ((2**-4 + 3 * 2**-8 + 1e-5) * sensitivity + (2**-10 / 448) * underflow)
            + 1e-6
        ).masked_fill(empty[:, None, None], 0)
    return (
        output.reshape(query.shape),
        lse.reshape(query.shape[:-1]),
        bound.reshape(query.shape),
    )


# Preparation: convert physical page slots into compact storage-row lists.
@triton.jit
def _map_indices(
    indices,
    lengths,
    out,
    counts,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    length = tl.load(lengths + row) if HAS_LENGTH else K
    token = tl.load(indices + row * STRIDE + col, col < K, -1)
    valid = (col < K) & (col < length) & (token >= 0)
    destination = tl.cumsum(valid.to(tl.int32)) - valid.to(tl.int32)
    storage_row = token // PAGE * PAGE_ROWS + token % PAGE
    tl.store(out + row * K + destination, storage_row, valid)
    count = tl.sum(valid.to(tl.int32))
    tl.store(counts + row, count)
    tl.store(out + row * K + col, -1, (col >= count) & (col < K))


def map_sparse_indices(
    indices, lengths, *, page_size, page_stride_rows, out=None, counts=None
):
    rows, width = indices.shape
    if out is None:
        out = torch.empty_like(indices, memory_format=torch.contiguous_format)
    if counts is None:
        counts = torch.empty(rows, device=indices.device, dtype=torch.int32)
    if rows:
        _map_indices[(rows,)](
            indices,
            lengths,
            out,
            counts,
            indices.stride(0),
            width,
            page_size,
            page_stride_rows,
            lengths is not None,
            triton.next_power_of_2(width),
            num_warps=4,
        )
    return out, counts


def prepare_sparse_mla_indices(
    wrapper,
    query,
    kv_cache,
    indices,
    lengths=None,
    *,
    extra_kv_cache=None,
    extra_indices=None,
    extra_lengths=None,
    out=None,
):
    """Test adapter mapping/compacting each source into storage-row indices.

    Inputs are physical page-slot indices, as in the original model fixtures.
    All allocations/preparation belong outside measured attention. Supplying
    out reuses addresses and supports preparation in graph-lifetime tests.
    Scales and attention_sinks are passed directly to run(); this adapter does not copy them.
    """
    state = wrapper._state
    if state is None:
        raise ValueError("a prepared attention plan is required")
    rows = query.numel() // (state["num_qo_heads"] * 512)
    ks, kc = state["max_topk"], state["max_extra_topk"]
    device = query.device

    def empty(shape, dtype=torch.int32):
        return torch.empty(shape, device=device, dtype=dtype)

    if out is None:
        out = dict(
            indices=empty((rows, ks)),
            lengths=empty((rows,)),
            extra_indices=empty((rows, kc)) if kc else None,
            extra_lengths=empty((rows,)) if kc else None,
        )

    def map_pool(cache, idx, lens, width, dst, count):
        axis = 2 if cache.ndim == 4 and state["kv_layout"] == "HND" else 1
        page = cache.shape[axis]
        map_sparse_indices(
            idx.view(rows, width),
            None if lens is None else lens.view(rows),
            page_size=page,
            page_stride_rows=cache.stride(0) // 512,
            out=dst,
            counts=count,
        )

    map_pool(kv_cache, indices, lengths, ks, out["indices"], out["lengths"])
    if kc:
        if extra_kv_cache is None or extra_indices is None:
            raise ValueError("the extra pool and indices are required")
        map_pool(
            extra_kv_cache,
            extra_indices,
            extra_lengths,
            kc,
            out["extra_indices"],
            out["extra_lengths"],
        )
    return out


def _inverse_output_rope(values, positions, cos_sin):
    result = values.reshape(positions.numel(), values.shape[-2], 512).clone()
    table = cos_sin[positions.reshape(-1)].to(result.dtype)
    c, s = table[:, None, :32], table[:, None, 32:]
    x, y = result[..., 448::2].clone(), result[..., 449::2].clone()
    result[..., 448::2] = x * c + y * s
    result[..., 449::2] = y * c - x * s
    return result


def _dequant_output(output, scales, block_size):
    if scales.dtype == torch.int32:
        exponent = torch.stack([(scales >> i) & 255 for i in (0, 8, 16, 24)], -1)
        scales = torch.exp2(exponent.flatten(-2).float() - 127)
    values = (
        output.float().unflatten(-1, (-1, block_size)) * scales[..., None]
    ).flatten(-2)
    return values, scales


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "batch,queries,heads,topk,block,fmt,packed",
    [
        (64, 1, 128, 513, 32, "ue8m0", False),
        (64, 1, 128, 513, 128, "fp32", False),
        (128, 1, 8, 513, 32, "fp32", False),
        (128, 1, 16, 513, 128, "ue8m0", False),
        (32, 1, 32, 513, 32, "ue8m0", False),
        (32, 1, 32, 513, 128, "fp32", False),
        (1, 1, 16, 513, 128, "ue8m0", False),
        (1, 1, 8, 8193, 32, "fp32", False),
        (2, 2, 24, 513, 32, "ue8m0", True),
        (1, 3, 64, 129, 128, "ue8m0", False),
        (1, 1, 128, 2049, 128, "fp32", False),
        # T == G and T != G, T > 1: token-major and group-major differ here.
        (16, 1, 128, 513, 128, "ue8m0", False),
        (5, 1, 128, 513, 32, "fp32", False),
        (5, 1, 16, 513, 128, "ue8m0", False),
        # Short lists on the H128 2CTA schedule (capacity 256), fused output.
        (80, 1, 128, 64, 128, "fp32", False),
        (80, 1, 128, 64, 32, "ue8m0", False),
        (1, 161, 128, 64, 128, "fp32", False),
        (1, 161, 64, 513, 32, "ue8m0", False),
        # Packed multiwave queries reuse the quantized output's shared stages.
        (2, 81, 128, 513, 32, "fp32", True),
    ],
)
def test_rope_quant_graph(
    dtype,
    batch,
    queries,
    heads,
    topk,
    block,
    fmt,
    packed,
    groups=None,
    output_padding=0,
):
    """Exercise final stores and reductions through public automatic dispatch."""
    torch.manual_seed(317)
    rows = batch * queries - int(packed)
    shape = (rows, heads, 512) if packed else (batch, queries, heads, 512)
    q = (torch.randn(shape, device="cuda") * 0.5).to(dtype)
    kv = (torch.randn(1024, 1, 512, device="cuda") * 0.5).to(dtype)
    extra = (torch.randn(256, 1, 512, device="cuda") * 0.5).to(dtype)
    # Duplicates are intentional: output fusion must preserve sparse semantics.
    indices = torch.randint(1024, (rows, topk), device="cuda", dtype=torch.int32)
    extra_indices = torch.randint(256, (rows, 128), device="cuda", dtype=torch.int32)
    lengths = torch.full((rows,), topk, device="cuda", dtype=torch.int32)
    extra_lengths = torch.full((rows,), 128, device="cuda", dtype=torch.int32)
    if rows > 1:
        lengths[0], extra_lengths[0] = 0, 0
        lengths[-1] = min(topk, 17)
    attention_sinks = torch.linspace(-3, 4, heads, device="cuda")
    attention_sinks[0] = torch.inf
    descales = dict(
        kv_scale=0.5 if dtype == torch.float8_e4m3fn else 1.0,
        output_scale=0.75,
    )
    positions = (torch.arange(rows, device="cuda", dtype=torch.int32) * 3 + 19).view(
        shape[:-2]
    )
    theta = torch.randn(rows * 3 + 64, 32, device="cuda")
    # Vectorized RoPE loads must honor the public 16-byte alignment contract.
    storage = torch.empty(theta.numel() * 2 + 4, device="cuda")
    cos_sin = storage[4:].view(-1, 64)
    cos_sin.copy_(torch.cat((theta.cos(), theta.sin()), -1))
    groups = max(1, heads // 8) if groups is None else groups
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        "cuda",
        batch,
        heads,
        max_topk=topk,
        max_extra_topk=128,
        max_seq_len_q=queries,
        packed_query=packed,
        q_dtype=dtype,
        use_attention_sinks=True,
        return_lse=True,
        output_quant_block_size=block,
        out_scale_format=fmt,
        num_output_groups=groups,
    )
    metadata = prepare_sparse_mla_indices(
        wrapper,
        q,
        kv,
        indices,
        lengths,
        extra_kv_cache=extra,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
    )
    indptr = (
        torch.tensor([0, queries, rows], device="cuda", dtype=torch.int32)
        if packed
        else None
    )
    kwargs = dict(
        qo_indptr=indptr,
        attention_sinks=attention_sinks,
        token_positions=positions,
        cos_sin_cache=cos_sin,
        softmax_scale=512**-0.5 * descales["kv_scale"],
        output_scale=descales["output_scale"] * descales["kv_scale"],
    )
    if packed:
        # Cover graph-live effective scales with fused output as well as the
        # Python-number path exercised by the fixed-query cases.
        kwargs["softmax_scale"] = torch.tensor(kwargs["softmax_scale"], device=q.device)
        kwargs["output_scale"] = torch.tensor(kwargs["output_scale"], device=q.device)
    output, scales, lse = wrapper.run(q, kv, **metadata, extra_kv_cache=extra, **kwargs)
    channels = heads // groups * 512
    assert output.shape == (rows, groups, channels)
    assert output.dtype == torch.float8_e4m3fn
    # Physical storage is group-major [G, T, C]; the public view is token-major.
    assert output.stride() == (channels, rows * channels, 1)
    assert output.transpose(0, 1).is_contiguous()
    padded = (rows + 3) // 4 * 4
    columns = heads // groups * 512 // block // (4 if fmt == "ue8m0" else 1)
    assert scales.shape == (rows, groups, columns)
    assert scales.stride() == (1, columns * padded, padded)
    # Graph replay also exercises caller-owned output aligned to 16, not 32.
    output_storage = torch.empty(
        groups * (rows + output_padding) * channels + 16,
        device=output.device,
        dtype=output.dtype,
    )
    output_backing = (
        output_storage[16:]
        .view(groups, rows + output_padding, channels)
        .transpose(0, 1)
    )
    output_backing.view(torch.uint8).fill_(123)
    output = output_backing[:rows]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            q,
            kv,
            **metadata,
            extra_kv_cache=extra,
            out=output,
            out_scale=scales,
            lse=lse,
            validate=False,
            **kwargs,
        )
    # Query positions are graph-live, including packed/chunked query batches.
    positions.add_(7)
    graph.replay()
    if output_padding:
        assert torch.all(output_backing[rows:].view(torch.uint8) == 123)
    ref, ref_lse, bound = _reference(
        q,
        kv,
        extra,
        indices,
        extra_indices,
        lengths=lengths,
        extra_lengths=extra_lengths,
        attention_sinks=attention_sinks,
        **descales,
    )
    ref = _inverse_output_rope(ref, positions, cos_sin)
    # Rotate the componentwise attention error bound using absolute coefficients.
    bound = bound.reshape(rows, heads, 512)
    c, s = (
        cos_sin[positions.flatten(), :32].abs()[:, None],
        cos_sin[positions.flatten(), 32:].abs()[:, None],
    )
    bx, by = bound[..., 448::2].clone(), bound[..., 449::2].clone()
    bound[..., 448::2] = bx * c + by * s
    bound[..., 449::2] = by * c + bx * s
    actual, dequant_scales = _dequant_output(output, scales, block)
    actual = actual.reshape(rows, heads, 512)
    # E4M3 RN adds at most 1/16 relative error plus half a subnormal step.
    rounding = dequant_scales.repeat_interleave(block, dim=-1).view_as(actual) / 1024
    allowed = bound + (ref.abs() + bound) / 16 + rounding + 1e-6
    assert torch.all((actual - ref).abs() <= allowed)
    assert (actual - ref).norm() / ref.norm().clamp_min(1e-20) < 0.06
    torch.testing.assert_close(lse.double(), ref_lse, atol=0.03, rtol=0.01)
    assert torch.count_nonzero(actual[:, 0]) == 0
    if rows > 1:
        assert torch.count_nonzero(actual[0]) == 0


@pytest.mark.parametrize("block", [32, 128])
@pytest.mark.parametrize("fmt", ["fp32", "ue8m0"])
def test_rope_quant_scales_and_contract(block, fmt):
    """One selected KV makes output/scale arithmetic independently checkable."""
    q = torch.zeros((1, 1, 8, 512), device="cuda", dtype=torch.bfloat16)
    kv = torch.linspace(-1, 1, 512, device="cuda").to(torch.bfloat16).view(1, 1, 512)
    kv[..., :32] = 0
    kv[..., 32:64] = 1e-8
    indices = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    positions = torch.ones((1, 1), device="cuda", dtype=torch.int32)
    cos_sin = torch.tensor(
        # Exact quarter-turn: isolate scale/packing checks from FMA rounding
        # at FP8 ties. The graph test above covers arbitrary angles.
        [[1.0] * 32 + [0.0] * 32, [0.0] * 32 + [1.0] * 32],
        device="cuda",
    )
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        "cuda",
        1,
        8,
        max_topk=1,
        output_quant_block_size=block,
        out_scale_format=fmt,
        num_output_groups=2,
    )
    metadata = prepare_sparse_mla_indices(wrapper, q, kv, indices)
    output, scales = wrapper.run(
        q, kv, **metadata, token_positions=positions, cos_sin_cache=cos_sin
    )
    reference = _inverse_output_rope(kv.expand(1, 8, 512).float(), positions, cos_sin)
    grouped = reference.reshape(1, 2, 2048).unflatten(-1, (-1, block))
    expected_scales = grouped.abs().amax(-1).clamp_min(1e-4) / 448
    if fmt == "ue8m0":
        expected_scales = expected_scales.log2().ceil().exp2()
    expected = (grouped / expected_scales[..., None]).to(torch.float8_e4m3fn).float()
    expected = (expected * expected_scales[..., None]).flatten(-2)
    actual, actual_scales = _dequant_output(output, scales, block)
    torch.testing.assert_close(actual_scales, expected_scales, atol=0, rtol=2e-6)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    if block == 32 and fmt == "ue8m0":
        eager_out, eager_scales = batch_sparse_mla_decode_with_paged_kv_cache(
            q,
            kv,
            **metadata,
            output_quant_block_size=block,
            out_scale_format=fmt,
            num_output_groups=2,
            token_positions=positions,
            cos_sin_cache=cos_sin,
        )
        torch.testing.assert_close(eager_out.float(), output.float(), atol=0, rtol=0)
        torch.testing.assert_close(eager_scales, scales, atol=0, rtol=0)
        misaligned = (
            torch.empty(output.numel() + 1, dtype=output.dtype, device="cuda")[1:]
            .view(2, 1, 2048)
            .transpose(0, 1)
        )
        with pytest.raises(ValueError, match="aligned"):
            wrapper.run(
                q,
                kv,
                **metadata,
                token_positions=positions,
                cos_sin_cache=cos_sin,
                out=misaligned,
            )
        scale_storage = torch.empty(
            scales.untyped_storage().nbytes() // 4 + 1,
            dtype=scales.dtype,
            device="cuda",
        )
        misaligned_scales = scale_storage.as_strided(scales.shape, scales.stride(), 1)
        with pytest.raises(ValueError, match="out_scale"):
            wrapper.run(
                q,
                kv,
                **metadata,
                token_positions=positions,
                cos_sin_cache=cos_sin,
                out_scale=misaligned_scales,
            )
    with pytest.raises(ValueError, match="token_positions"):
        wrapper.run(
            q, kv, **metadata, token_positions=positions + 2, cos_sin_cache=cos_sin
        )
    with pytest.raises(ValueError, match="out_scale"):
        wrapper.run(
            q,
            kv,
            **metadata,
            token_positions=positions,
            cos_sin_cache=cos_sin,
            out_scale=scales.contiguous(),
        )

    # A packed zero-token batch needs valid empty views, but no device launch.
    wrapper.plan(
        "cuda",
        1,
        8,
        max_topk=1,
        packed_query=True,
        output_quant_block_size=block,
        out_scale_format=fmt,
        num_output_groups=2,
    )
    empty_q = q.view(1, 8, 512)[:0]
    empty_metadata = prepare_sparse_mla_indices(wrapper, empty_q, kv, indices[:0])
    empty_out, empty_scales = wrapper.run(
        empty_q,
        kv,
        **empty_metadata,
        qo_indptr=torch.zeros(2, device="cuda", dtype=torch.int32),
        token_positions=positions.view(-1)[:0],
        cos_sin_cache=cos_sin,
    )
    assert empty_out.shape == (0, 2, 2048)
    assert empty_scales.shape == (0, 2, 2048 // block // (4 if fmt == "ue8m0" else 1))


@pytest.mark.parametrize("rows,groups", [(3, 2), (1, 2), (3, 1)])
def test_rope_quant_output_layout_contract(rows, groups):
    """Require physical [G,T,C] storage while ignoring singleton-axis strides."""
    heads = 8
    q = torch.zeros((rows, 1, heads, 512), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(4, 1, 512, device="cuda").to(torch.bfloat16)
    indices = torch.zeros((rows, 1), device="cuda", dtype=torch.int32)
    positions = torch.ones((rows, 1), device="cuda", dtype=torch.int32)
    cos_sin = torch.zeros(4, 64, device="cuda")
    cos_sin[:, :32] = 1
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        "cuda",
        rows,
        heads,
        max_topk=1,
        output_quant_block_size=128,
        out_scale_format="fp32",
        num_output_groups=groups,
    )
    metadata = prepare_sparse_mla_indices(wrapper, q, kv, indices)
    token_major = torch.empty(
        (rows, groups, heads // groups * 512), dtype=torch.float8_e4m3fn, device="cuda"
    )

    def run(out):
        return wrapper.run(
            q, kv, **metadata, token_positions=positions, cos_sin_cache=cos_sin, out=out
        )

    if rows > 1 and groups > 1:
        with pytest.raises(ValueError, match="group-major"):
            run(token_major)
    else:
        out, _ = run(token_major)
        assert out.data_ptr() == token_major.data_ptr()
    if rows == 1 and groups > 1:
        overlapping = token_major.as_strided(
            token_major.shape, (heads // groups * 512, 0, 1)
        )
        with pytest.raises(ValueError, match="group-major"):
            run(overlapping)
    group_major = token_major.view(groups, rows, -1).transpose(0, 1)
    out, _ = run(group_major)
    assert out.data_ptr() == group_major.data_ptr()


@pytest.mark.parametrize(
    "heads,groups,block,fmt",
    [(128, 1, 128, "fp32"), (128, 2, 32, "ue8m0"), (24, 8, 128, "fp32")],
)
def test_rope_quant_group_layout(heads, groups, block, fmt):
    """Cover partial/whole-group TMA boxes and non-power-of-two head groups."""
    rows = (
        torch.cuda.get_device_properties(0).multi_processor_count if heads == 128 else 3
    )
    test_rope_quant_graph(
        torch.float8_e4m3fn,
        rows,
        1,
        heads,
        513,
        block,
        fmt,
        False,
        groups=groups,
        output_padding=3,
    )


def test_rope_quant_plan_options():
    plan = dict(
        device="cuda", batch_size=1, num_qo_heads=8, max_topk=1, num_output_groups=2
    )
    for block, error in ((True, TypeError), (32.0, TypeError), (64, ValueError)):
        with pytest.raises(error, match="output_quant_block_size"):
            get_prims_ts_sparse_mla_decode_workspace_size(
                **plan, output_quant_block_size=block
            )
    with pytest.raises(ValueError, match="out_scale_format"):
        get_prims_ts_sparse_mla_decode_workspace_size(
            **plan, output_quant_block_size=32, out_scale_format="e4m3"
        )
    with pytest.raises(ValueError, match="num_output_groups"):
        get_prims_ts_sparse_mla_decode_workspace_size(
            **dict(plan, num_output_groups=3), output_quant_block_size=32
        )


# Accuracy and important public validation contracts.


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "batch,queries,heads,topk,extra,packed,prefix",
    [
        pytest.param(1, 3, 8, 129, True, False, False, id="small-head-tail"),
        pytest.param(2, 2, 24, 513, True, True, False, id="packed-partial-heads"),
        pytest.param(2, 2, 6, 65, True, True, False, id="odd-head-packed"),
        pytest.param(2, 3, 64, 129, False, False, False, id="single-source"),
        pytest.param(4, 1, 64, 513, True, False, True, id="wide-query"),
        pytest.param(64, 1, 128, 513, True, False, True, id="two-cta"),
        pytest.param(16, 1, 128, 2049, True, False, False, id="small-grid-wide-kv"),
        pytest.param(0, 1, 96, 513, True, False, True, id="partial-wide-heads"),
        pytest.param(0, 1, 8, 513, True, False, True, id="single-stream-short-kv"),
        pytest.param(0, 2, 8, 2049, True, False, True, id="small-head-long-kv-reuse"),
        pytest.param(1, 0, 64, 513, True, False, True, id="multiwave-prefill"),
        pytest.param(0, 4, 128, 2049, True, False, True, id="multiwave-decode"),
        # Short lists (capacity 256 and 384) with enough rows for the H128 2CTA tile.
        pytest.param(0, 1, 128, 64, True, False, True, id="short-two-cta-decode"),
        pytest.param(1, 0, 128, 64, True, False, True, id="short-two-cta-prefill"),
        pytest.param(0, 1, 128, 256, True, False, True, id="short-two-cta-384"),
        pytest.param(1, 3, 8, 8193, True, False, False, id="long-split-reduction"),
    ],
)
def test_native_graph(dtype, batch, queries, heads, topk, extra, packed, prefix):
    # Scale throughput cases with hardware, without asserting exact
    # tiles, split counts, register budgets or private policy identities.
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    batch = batch or sms
    queries = queries or (sms + 1)
    rows = batch * queries - int(packed)
    shape = (rows, heads, 512) if packed else (batch, queries, heads, 512)
    torch.manual_seed(131)
    q = (torch.randn(shape, device="cuda") * 0.15).to(dtype)
    a_storage = (torch.randn(4, 33, 512, device="cuda") * 0.15).to(dtype)
    b_storage = (torch.randn(7, 17, 512, device="cuda") * 0.15).to(dtype)
    a_storage[:, 32] = torch.nan
    b_storage[:, 16] = torch.nan
    a, b = a_storage[:, :32], b_storage[:, :16]
    ka, kb = (128, topk) if extra else (topk, 0)
    ai = (torch.arange(ka, device="cuda", dtype=torch.int32) % 128).repeat(rows, 1)
    bi = (
        (torch.arange(kb, device="cuda", dtype=torch.int32) % 112).repeat(rows, 1)
        if extra
        else None
    )
    al = torch.full((rows,), ka, device="cuda", dtype=torch.int32)
    bl = torch.full((rows,), kb, device="cuda", dtype=torch.int32) if extra else None
    al[0] = 0
    if extra:
        bl[0] = 0
    attention_sinks = torch.randn(heads, device="cuda")
    attention_sinks[0], attention_sinks[1] = torch.inf, -torch.inf
    kv_scale = torch.tensor(
        0.75 if dtype == torch.float8_e4m3fn else 1.0, device="cuda"
    )
    q_scale = torch.tensor(1.25 if dtype == torch.float8_e4m3fn else 1.0, device="cuda")
    offsets = (
        torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32) if packed else None
    )
    # Packed cases also cover the rank-four HND pool interface; the oracle
    # receives the same underlying padded storage as rank-three views.
    primary, secondary = (a.unsqueeze(1), b.unsqueeze(1)) if packed else (a, b)
    if not extra:
        primary = a.unsqueeze(2)  # NHD, singleton KV head
    w = BatchSparseMLADecodePagedTSWrapper()
    w.plan(
        q.device,
        batch,
        heads,
        max_topk=ka,
        max_extra_topk=kb,
        max_seq_len_q=queries,
        packed_query=packed,
        q_dtype=dtype,
        kv_layout="HND" if packed else "NHD",
        use_attention_sinks=True,
        return_lse=True,
        assume_valid_prefix=prefix,
    )
    attention_scale = torch.tensor(512**-0.5, device=q.device)
    output_multiplier = torch.tensor(0.75, device=q.device)
    effective_softmax_scale = torch.empty_like(attention_scale)
    effective_output_scale = torch.empty_like(output_multiplier)
    scales = dict(
        softmax_scale=effective_softmax_scale,
        output_scale=effective_output_scale,
        attention_sinks=attention_sinks,
    )

    def prepare(out=None):
        # Upstream work combines descales into stable graph inputs. The oracle
        # below keeps the original Q/K/V factors to check equivalence.
        torch.mul(attention_scale, q_scale, out=effective_softmax_scale)
        effective_softmax_scale.mul_(kv_scale)
        torch.mul(output_multiplier, kv_scale, out=effective_output_scale)
        return prepare_sparse_mla_indices(
            w,
            q,
            primary,
            ai,
            al,
            extra_kv_cache=secondary if extra else None,
            extra_indices=bi,
            extra_lengths=bl,
            out=out,
        )

    meta = prepare()
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:-1], device=q.device)

    def run(checked=False, target_lse=lse):
        return w.run(
            q,
            primary,
            **meta,
            extra_kv_cache=secondary if extra else None,
            qo_indptr=offsets,
            out=out,
            lse=target_lse,
            validate=checked,
            **scales,
        )

    def check(target_lse=lse):
        prefix_shape = q.shape[:-2]
        ref, ref_lse, bound = _reference(
            q,
            a,
            b if extra else None,
            ai.reshape(*prefix_shape, ka),
            bi.reshape(*prefix_shape, kb) if extra else None,
            lengths=al.reshape(prefix_shape),
            extra_lengths=bl.reshape(prefix_shape) if extra else None,
            q_scale=q_scale.item(),
            kv_scale=kv_scale.item(),
            softmax_scale=attention_scale.item(),
            output_scale=output_multiplier.item(),
            attention_sinks=attention_sinks,
        )
        if dtype == torch.float8_e4m3fn:
            assert ((out.double() - ref).abs() <= bound).all()
        else:
            torch.testing.assert_close(out.double(), ref, atol=8e-4, rtol=0.01)
        torch.testing.assert_close(target_lse.double(), ref_lse, atol=2e-4, rtol=1e-4)

    run(checked=True)
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare(out=meta)
        run()
    # Transition from contiguous bulk loads to irregular gathers, activate an
    # empty row, and empty the last row. Preparation refreshes stable buffers.
    ai.copy_(ai.flip(-1))
    ai[:, 5::17] = -1
    al[0], al[-1] = 1, 0
    if extra:
        bi.copy_(bi.flip(-1))
        bi[:, 3::11] = -1
        bl[0], bl[-1] = 1, 0
    if dtype == torch.float8_e4m3fn:
        kv_scale.fill_(1.25)
        q_scale.fill_(0.5)
    attention_scale.mul_(0.75)
    output_multiplier.fill_(1.25)
    graph.replay()
    torch.cuda.synchronize()
    check()
    assert torch.count_nonzero(out.reshape(rows, heads, 512)[-1]) == 0
    assert torch.isneginf(lse.reshape(rows, heads)[-1]).all()
    if packed:
        unaligned = torch.empty(lse.numel() + 1, device=q.device)[1:].view_as(lse)
        with pytest.raises(ValueError, match="16-byte-aligned"):
            run(target_lse=unaligned)
    if packed and heads == 6 and dtype == torch.bfloat16:
        empty_meta = prepare_sparse_mla_indices(
            w,
            q[:0],
            primary,
            ai[:0],
            al[:0],
            extra_kv_cache=secondary,
            extra_indices=bi[:0],
            extra_lengths=bl[:0],
        )
        empty, empty_lse = w.run(
            q[:0],
            primary,
            **empty_meta,
            extra_kv_cache=secondary,
            qo_indptr=torch.zeros_like(offsets),
            **scales,
        )
        assert empty.numel() == empty_lse.numel() == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_large_query_native_graph(dtype):
    # A long Q grid exercises persistent work-tile advancement, including
    # BF16's static scheduler and FP8's CLC scheduler on the same public inputs.
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    test_native_graph(dtype, 1, 8 * sms, 32, 513, True, False, True)


def test_caller_workspace_single_source_and_eager_api():
    q = torch.zeros(1, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.ones(2, 1, 512, device="cuda", dtype=torch.bfloat16)
    kv[1] *= 2
    indices = torch.tensor([[[0, 1, -1]]], device="cuda", dtype=torch.int32)
    args = dict(
        device=q.device, batch_size=1, num_qo_heads=16, max_topk=3, return_lse=True
    )
    size = get_prims_ts_sparse_mla_decode_workspace_size(**args)
    workspace = torch.empty(size, device=q.device, dtype=torch.uint8)
    wrapper = BatchSparseMLADecodePagedTSWrapper(workspace)
    wrapper.plan(**args)
    metadata = prepare_sparse_mla_indices(wrapper, q, kv, indices)
    result, lse = wrapper.run(q, kv, **metadata)
    torch.testing.assert_close(result, torch.full_like(q, 1.5))
    eager, eager_lse = batch_sparse_mla_decode_with_paged_kv_cache(
        q, kv, **metadata, return_lse=True
    )
    torch.testing.assert_close(result, eager, atol=0, rtol=0)
    torch.testing.assert_close(lse, eager_lse, atol=0, rtol=0)
    first_lse = lse.clone()
    shorter = prepare_sparse_mla_indices(
        wrapper, q, kv, indices, torch.ones((1, 1), device=q.device, dtype=torch.int32)
    )
    _, next_lse = wrapper.run(q, kv, **shorter)
    torch.testing.assert_close(next_lse, torch.zeros_like(next_lse), atol=0, rtol=0)
    torch.testing.assert_close(lse, first_lse, atol=0, rtol=0)
    short = BatchSparseMLADecodePagedTSWrapper(workspace[:-1])
    with pytest.raises(ValueError, match="workspace"):
        short.plan(**args)
    bad = indices.clone()
    bad[..., 0] = 2
    with pytest.raises(ValueError, match="outside pool"):
        wrapper.run(q, kv, **dict(metadata, indices=bad.view(1, 3)))
    with pytest.raises(ValueError, match="positive"):
        wrapper.run(q, kv, **metadata, softmax_scale=1e-100)


def test_valid_prefix_eager_contract():
    q = torch.zeros(1, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.ones(2, 1, 512, device=q.device, dtype=q.dtype)
    swa[1] *= 2
    compressed = swa * 10
    si = torch.tensor([[[0, 1, -1]]], device=q.device, dtype=torch.int32)
    ci = torch.tensor([[[1, -1, 0]]], device=q.device, dtype=torch.int32)
    lengths = torch.tensor([[2]], device=q.device, dtype=torch.int32)
    ci[..., 1] = 0
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        q.device, 1, 16, max_topk=3, max_extra_topk=3, assume_valid_prefix=True
    )
    metadata = prepare_sparse_mla_indices(
        wrapper,
        q,
        swa,
        si,
        lengths,
        extra_kv_cache=compressed,
        extra_indices=ci,
        extra_lengths=lengths,
    )
    metadata["extra_indices"][0, 1] = -1
    kwargs = dict(assume_valid_prefix=True, return_lse=True)
    with pytest.raises(ValueError, match="extra active prefix contains a hole"):
        batch_sparse_mla_decode_with_paged_kv_cache(
            q, swa, **metadata, extra_kv_cache=compressed, **kwargs
        )
    metadata["extra_indices"][0, 1] = 0
    result, lse = batch_sparse_mla_decode_with_paged_kv_cache(
        q, swa, **metadata, extra_kv_cache=compressed, **kwargs
    )
    # Q=0 gives uniform attention over values 1, 2, 20, 10.
    torch.testing.assert_close(result, torch.full_like(result, 8.25))
    torch.testing.assert_close(lse, torch.full_like(lse, 4.0).log())


def test_sparse_indices_graph():
    """Caller index/length tensors remain live through graph replay."""
    q = torch.zeros(1, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.ones(2, 1, 512, device=q.device, dtype=q.dtype)
    kv[1] *= 3
    indices = torch.tensor([[0, -1, 1]], device=q.device, dtype=torch.int32)
    lengths = torch.tensor([3], device=q.device, dtype=torch.int32)
    w = BatchSparseMLADecodePagedTSWrapper()
    w.plan(q.device, 1, 16, max_topk=3, return_lse=True)
    out, lse = w.run(q, kv, indices, lengths)
    torch.testing.assert_close(out, torch.full_like(out, 2.0))
    torch.testing.assert_close(lse, torch.full_like(lse, 2.0).log())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(q, kv, indices, lengths, out=out, lse=lse, validate=False)
    indices[0, 2] = -1
    graph.replay()
    torch.testing.assert_close(out, torch.ones_like(out))
    torch.testing.assert_close(lse, torch.zeros_like(lse))
    lengths.zero_()
    graph.replay()
    assert torch.count_nonzero(out) == 0 and torch.isneginf(lse).all()


@pytest.mark.parametrize(
    "heads,prefix", [(8, False), (8, True), (128, False), (128, True)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_source_tile_boundary_graph(heads, prefix, dtype):
    """Move the extra-source boundary between 128/256 during graph replay."""
    batch = (
        1 if heads == 8 else torch.cuda.get_device_properties(0).multi_processor_count
    )
    q = torch.zeros((batch, 1, heads, 512), device="cuda", dtype=dtype)
    # Distinct pool values expose selection of the wrong source descriptor.
    primary = torch.ones((129, 1, 512), device="cuda", dtype=dtype)
    extra = torch.full((5, 1, 512), 8, device="cuda", dtype=dtype)
    indices = torch.arange(129, device="cuda", dtype=torch.int32).repeat(batch, 1)
    extra_indices = torch.arange(5, device="cuda", dtype=torch.int32).repeat(batch, 1)
    lengths = torch.full((batch,), 127, device="cuda", dtype=torch.int32)
    extra_lengths = torch.full_like(lengths, 5)
    kv_scale = 0.5 if dtype == torch.float8_e4m3fn else 1.0
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        q.device,
        batch,
        heads,
        max_topk=129,
        max_extra_topk=5,
        q_dtype=dtype,
        return_lse=True,
        use_attention_sinks=True,
        assume_valid_prefix=prefix,
    )

    # No packed routes or source padding: the kernel derives the boundary.
    metadata = dict(
        indices=indices,
        lengths=lengths,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
    )
    attention_sinks = torch.full((heads,), -torch.inf, device=q.device)
    out, lse = wrapper.run(
        q,
        primary,
        **metadata,
        extra_kv_cache=extra,
        attention_sinks=attention_sinks,
        softmax_scale=512 ** (-0.5) * kv_scale,
        output_scale=kv_scale,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            q,
            primary,
            **metadata,
            extra_kv_cache=extra,
            attention_sinks=attention_sinks,
            out=out,
            lse=lse,
            validate=False,
            softmax_scale=512 ** (-0.5) * kv_scale,
            output_scale=kv_scale,
        )
    for primary_length, extra_length in (
        (127, 5),
        (129, 5),
        (128, 5),
        (0, 5),
        (129, 0),
        (0, 0),
    ):
        lengths.fill_(primary_length)
        extra_lengths.fill_(extra_length)
        graph.replay()
        count = primary_length + extra_length
        expected = (
            kv_scale * (primary_length + 8 * extra_length) / count if count else 0
        )
        torch.testing.assert_close(
            out.float(), torch.full_like(out.float(), expected), rtol=0.01, atol=0.002
        )
        expected_lse = torch.tensor(float(count), device=q.device).log()
        torch.testing.assert_close(
            lse, expected_lse.expand_as(lse), rtol=1e-4, atol=2e-4
        )

    # An invalid first route must not redirect later rows to the other pool.
    # Sinks are graph-live inputs and need no metadata refresh either.
    lengths.fill_(129)
    extra_lengths.fill_(5)
    if not prefix:
        indices[:, 0] = -1
        extra_indices[:, 0] = -1
    attention_sinks.zero_()
    graph.replay()
    primary_count, extra_count = (129, 5) if prefix else (128, 4)
    count = primary_count + extra_count
    expected = kv_scale * (primary_count + 8 * extra_count) / (count + 1)
    torch.testing.assert_close(
        out.float(), torch.full_like(out.float(), expected), rtol=0.01, atol=0.002
    )
    expected_lse = torch.tensor(float(count), device=q.device).log()
    torch.testing.assert_close(lse, expected_lse.expand_as(lse), rtol=1e-4, atol=2e-4)


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_softmax_score_jumps(heads, dtype):
    batch = torch.cuda.get_device_properties(0).multi_processor_count + 1
    # The first two BK128 tiles initialize both softmax streams at zero.
    # The following two tiles straddle the six-log2 update boundary. This
    # checks both retained-anchor and rescaling paths.
    q = torch.zeros((batch, 1, heads, 512), device="cuda", dtype=dtype)
    q[..., 1] = 1
    swa = torch.zeros((1, 256, 512), device="cuda", dtype=q.dtype)
    kv = torch.zeros((384, 1, 512), device="cuda", dtype=q.dtype)
    kv[128:, :, 0] = 1
    si = (
        torch.arange(128, device="cuda", dtype=torch.int32)
        .view(1, 1, -1)
        .repeat(batch, 1, 1)
    )
    ci = (
        torch.arange(384, device="cuda", dtype=torch.int32)
        .view(1, 1, -1)
        .repeat(batch, 1, 1)
    )
    attention_sinks = torch.zeros(heads, device="cuda")
    attention_sinks[0], attention_sinks[1] = -torch.inf, torch.inf
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:-1], device="cuda")
    w = BatchSparseMLADecodePagedTSWrapper()
    w.plan(
        q.device,
        batch,
        heads,
        q_dtype=q.dtype,
        max_topk=128,
        max_extra_topk=384,
        use_attention_sinks=True,
        return_lse=True,
    )
    softmax_scale = 0.6931471805599453
    meta = prepare_sparse_mla_indices(
        w,
        q,
        swa,
        si,
        extra_kv_cache=kv,
        extra_indices=ci,
    )
    w.run(
        q,
        swa,
        **meta,
        extra_kv_cache=kv,
        attention_sinks=attention_sinks,
        softmax_scale=softmax_scale,
        out=out,
        lse=lse,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(
            q,
            swa,
            **meta,
            extra_kv_cache=kv,
            attention_sinks=attention_sinks,
            softmax_scale=softmax_scale,
            out=out,
            lse=lse,
            validate=False,
        )
    for delta in (5.5, 6.0, 6.5, 12.0):
        kv[128:, :, 1] = delta
        graph.replay()
        ref, ref_lse, bound = _reference(
            q,
            swa,
            kv,
            si,
            ci,
            attention_sinks=attention_sinks,
            softmax_scale=softmax_scale,
        )
        assert torch.isfinite(out).all()
        if dtype == torch.float8_e4m3fn:
            assert ((out.double() - ref).abs() <= bound).all()
        else:
            torch.testing.assert_close(out.double(), ref, atol=8e-4, rtol=0.01)
        torch.testing.assert_close(lse.double(), ref_lse, atol=2e-4, rtol=1e-4)
