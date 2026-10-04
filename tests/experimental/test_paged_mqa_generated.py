"""Paged prepared API (DeepGEMM paged signatures): metadata against the scalar
DeepGEMM schedule specification, logits against the dequantized PyTorch paged
reference (and DeepGEMM when importable), the clean_logits=False mask, graph
replay with changed contents and chunk equivalence on SM100a/SM103a.

The tests skip while the installed catalog carries no paged route for the
shape (``paged_route_available``); the host-only helpers are tested always.
"""

import pytest
import torch

from flashinfer.experimental.deepgemm_dense_mqa import paged_mqa as _runtime
from flashinfer.paged_mqa import (
    fp8_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
    prepare_paged_mqa_logits,
)

HEAD_DIM = 128
# (heads, page, next_n, batch, avg context)
CASES = [
    (64, 64, 1, 1, 1024),
    (64, 64, 1, 16, 8192),
    (64, 64, 2, 16, 4096),
    (64, 64, 4, 3, 2048),
    (64, 64, 1, 256, 1024),
    (32, 64, 1, 16, 4096),
    (64, 32, 1, 16, 4096),
]


def _skip_unless_route(heads, page, next_n):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flashinfer.experimental.deepgemm_dense_mqa import dense_mqa

        dense_mqa.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(str(error))
    if not _runtime.paged_route_available(heads, page, next_n):
        pytest.skip(
            f"catalog has no paged route for H={heads}, page {page}, next_n {next_n}"
        )


def _inputs(heads, page, next_n, batch, avg_ctx, seed=7):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    lo, hi = max(page, int(0.7 * avg_ctx)), max(page + 1, int(1.3 * avg_ctx))
    ctx_max = torch.randint(
        lo, hi, (batch,), device=device, dtype=torch.int32, generator=generator
    )
    offsets = (next_n - 1 - torch.arange(next_n, device=device, dtype=torch.int32))[
        None, :
    ]
    ctx_2d = (ctx_max[:, None] - offsets).clamp_min(1).contiguous()
    max_context_len = int(ctx_max.max())
    blocks = (ctx_max + page - 1) // page
    width = int(blocks.max())
    pages = int(blocks.sum()) + batch
    perm = torch.randperm(pages, device=device, generator=generator).to(torch.int32)
    block_table = torch.zeros(batch, width, device=device, dtype=torch.int32)
    offset = 0
    for b in range(batch):
        n = int(blocks[b])
        block_table[b, :n] = perm[offset : offset + n]
        offset += n
    q = torch.randn(
        batch, next_n, heads, HEAD_DIM, device=device, generator=generator
    ).to(torch.float8_e4m3fn)
    kv = torch.randn(pages, page, HEAD_DIM, device=device, generator=generator)
    scale = (kv.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / 448.0).squeeze(-1)
    kv_fp8 = (kv / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    fused = torch.empty(pages, page * (HEAD_DIM + 4), device=device, dtype=torch.uint8)
    fused[:, : page * HEAD_DIM] = kv_fp8.reshape(pages, -1).view(torch.uint8)
    fused[:, page * HEAD_DIM :] = scale.reshape(pages, page).view(torch.uint8)
    kv_cache = fused.view(pages, page, 1, HEAD_DIM + 4)
    weights = (
        torch.rand(batch * next_n, heads, device=device, generator=generator) + 0.1
    )
    return q, kv_cache, kv_fp8, scale, weights, ctx_2d, block_table, max_context_len


def _reference(q, kv_fp8, scale, weights, ctx_2d, block_table, max_context_len, page):
    """logits[b*next_n + t, k] = sum_h relu(q . kv[k]) * w * scale[k] for k < ctx[b, t], -inf otherwise."""
    batch, next_n, heads, _ = q.shape
    device = q.device
    out = torch.full(
        (batch * next_n, max_context_len),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    for b in range(batch):
        ctx_last = int(ctx_2d[b, -1])
        n_blocks = (ctx_last + page - 1) // page
        phys = block_table[b, :n_blocks].long()
        keys = kv_fp8[phys].float().reshape(-1, HEAD_DIM)[:ctx_last]
        kscale = scale[phys].reshape(-1)[:ctx_last]
        for t in range(next_n):
            qt = q[b, t].float()
            logits = (torch.relu(qt @ keys.T) * weights[b * next_n + t][:, None]).sum(
                0
            ) * kscale
            ctx_t = int(ctx_2d[b, t])
            out[b * next_n + t, :ctx_t] = logits[:ctx_t]
    return out


def _metadata_reference(ctx_2d, next_n, num_sms, split_kv, atoms):
    """Scalar specification of the paged schedule: (q_atom_idx, kv_split_idx) per CTA boundary."""
    lens = [int(ctx_2d[b, -1]) for b in range(ctx_2d.shape[0])]
    segs = [(c + split_kv - 1) // split_kv for c in lens]
    prefix, acc = [], 0
    for s in segs:
        acc += s
        prefix.append(acc)
    total = acc * atoms
    qd, rd = total // num_sms, total % num_sms
    meta = []
    for sm in range(num_sms + 1):
        start = sm * qd + min(sm, rd)
        q_idx = 0
        while q_idx < len(lens) and prefix[q_idx] * atoms <= start:
            q_idx += 1
        prev = prefix[q_idx - 1] if q_idx else 0
        cur = prefix[q_idx] if q_idx < len(lens) else acc
        offset, n = start - prev * atoms, cur - prev
        atom = offset // n if n else 0
        split = offset % n if n else 0
        meta.append([q_idx * atoms + atom, split])
    return meta


def _check(got, ref, ctx_2d, max_context_len, stride_rows):
    lengths = ctx_2d.reshape(-1)
    position = torch.arange(max_context_len, device=got.device)[None, :]
    inside = position < lengths[:, None]
    assert torch.isfinite(got[inside]).all()
    row_max = (
        ref.masked_fill(~inside, 0.0).abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    )
    err = (got - ref).abs().masked_fill(~inside, 0.0)
    assert bool(
        (err <= 1e-2 * row_max + 1e-2 * ref.abs().masked_fill(~inside, 0.0)).all()
    )
    # clean_logits=False semantics: inside the aligned range of each request the cells at or past the
    # row's length are exact -inf (DeepGEMM stores finite don't-care there; -inf is the strictly safer
    # value for the engine's top-k). Which cells are written at all is checked against DeepGEMM's own
    # behaviour in _check_write_extent.
    # The extent is align(ctx_last, SPLIT_KV), which exceeds max_context_len for the longest request(s) when
    # max_context_len % SPLIT_KV != 0: check it on the physical rows, not on the [:, :max_context_len] view.
    for row in range(got.shape[0]):
        aligned = min(_aligned_extent(ctx_2d, row), stride_rows.shape[1])
        assert bool(
            torch.isneginf(stride_rows[row, int(lengths[row]) : aligned]).all()
        ), row


def _aligned_extent(ctx_2d, row):
    """DeepGEMM's per-request store extent read from its source: the SM100 paged scheduler issues
    ``ceil(context_len / SPLIT_KV)`` KV splits of ``SPLIT_KV`` columns per request
    (``sm100_paged_mqa_logits.cuh``, ``num_kv_splits``) and with ``clean_logits=False`` nothing touches the
    columns beyond (only the clean_logits cleaner fills ``[coverage_end, logits_stride)``)."""
    split_kv = int(_runtime.paged_policy()["split_kv"])
    ctx_last = int(ctx_2d[row // ctx_2d.shape[1], -1])
    return (ctx_last + split_kv - 1) // split_kv * split_kv


def _deep_gemm_measured(q, kv_cache, weights, ctx_2d, block_table, max_len, page):
    """``(native_logits, written)`` with ``written`` the mask of cells DeepGEMM actually stores over its whole
    output storage, measured: DeepGEMM allocates its output internally, so the block it will receive is
    poisoned with NaN first (allocate a buffer of the identical byte size, free it, call DeepGEMM; the caching
    allocator hands the same block back) and the never-written storage columns past ``max_len`` prove the
    poison survived. ``None`` when DeepGEMM is unavailable or the poison is not observable (another block
    was handed out); the caller then falls back to the extent read from DeepGEMM's source."""
    try:
        import deep_gemm
    except ImportError:
        return None
    if not hasattr(deep_gemm, "fp8_paged_mqa_logits"):
        return None

    def run():
        meta = deep_gemm.get_paged_mqa_logits_metadata(
            ctx_2d, page, deep_gemm.get_num_sms()
        )
        return deep_gemm.fp8_paged_mqa_logits(
            q, kv_cache, weights, ctx_2d, block_table, meta, max_len, clean_logits=False
        )

    probe = run()
    torch.cuda.synchronize()
    nbytes, rows, stride = (
        probe.untyped_storage().nbytes(),
        int(probe.shape[0]),
        int(probe.stride(0)),
    )
    del probe
    poison = torch.full(
        (nbytes // 4,), float("nan"), dtype=torch.float32, device=q.device
    )
    del poison
    native = run()
    torch.cuda.synchronize()
    if native.untyped_storage().nbytes() != nbytes or rows * stride * 4 > nbytes:
        return None
    full = torch.empty(0, dtype=torch.float32, device=q.device).set_(
        native.untyped_storage(), 0, (rows * stride,)
    )
    full = full.view(rows, stride)
    # Witness that the poison survived: columns past each request's source-derived extent are never written
    # by DeepGEMM (the buffer is [rows, align(max_context_len, SPLIT_KV)], so rows of shorter requests have
    # such columns); without any witness cell the measurement is not observable.
    witness = [
        full[row, min(_aligned_extent(ctx_2d, row), stride) :] for row in range(rows)
    ]
    witness_cells = sum(int(w.numel()) for w in witness)
    if witness_cells == 0 or not all(
        bool(torch.isnan(w).all()) for w in witness if w.numel()
    ):
        return None
    return native, ~torch.isnan(full)


def _check_write_extent(cake_output, measured, ctx_2d, max_len):
    """Cake writes exactly the cells DeepGEMM writes (cell for cell on the measured DeepGEMM storage when
    observable, else DeepGEMM's source-derived extent): a kernel that writes more or less than DeepGEMM is
    wrong, the test is not loosened to it. ``cake_output`` is the plan's physical [rows, stride] buffer,
    NaN-poisoned before the launch."""
    cake_written = ~torch.isnan(cake_output)
    # Beyond max_context_len the written cells of the longest request(s) are the approved deviation: -inf.
    for row in range(cake_output.shape[0]):
        aligned = min(_aligned_extent(ctx_2d, row), cake_output.shape[1])
        tail = cake_output[row, max_len:aligned]
        assert bool(torch.isneginf(tail).all()), (
            f"row {row}: cells in [{max_len}, {aligned}) are not -inf"
        )
    if measured is None:
        for row in range(cake_output.shape[0]):
            aligned = min(_aligned_extent(ctx_2d, row), cake_output.shape[1])
            assert bool(cake_written[row, :aligned].all()), (
                f"row {row}: Cake left cells inside DeepGEMM's extent unwritten"
            )
            assert not bool(cake_written[row, aligned:].any()), (
                f"row {row}: Cake writes {int(cake_written[row, aligned:].sum())} cells past DeepGEMM's extent {aligned}"
            )
        return
    _native, dg_written = measured
    common = min(cake_output.shape[1], dg_written.shape[1])
    differs = cake_written[:, :common] ^ dg_written[:, :common]
    if bool(differs.any()):
        row = int(differs.any(dim=1).nonzero()[0])
        cols = differs[row].nonzero().flatten()
        raise AssertionError(
            f"{int(differs.sum())} cells written by exactly one of Cake / DeepGEMM; row {row} columns "
            f"{int(cols[0])}..{int(cols[-1])} (Cake wrote {int(cake_written[row, cols].sum())} of them, DeepGEMM "
            f"{int(dg_written[row, cols].sum())}); request length {int(ctx_2d.reshape(-1)[row])}"
        )
    assert not bool(cake_written[:, common:].any()) and not bool(
        dg_written[:, common:].any()
    )


@pytest.mark.parametrize("heads,page,next_n,batch,avg_ctx", CASES)
def test_paged_mqa_logits(heads, page, next_n, batch, avg_ctx):
    _skip_unless_route(heads, page, next_n)
    q, kv_cache, kv_fp8, scale, weights, ctx_2d, block_table, max_len = _inputs(
        heads, page, next_n, batch, avg_ctx
    )
    plan = prepare_paged_mqa_logits(q, kv_cache, weights, ctx_2d, block_table, max_len)
    assert plan.route_name == _runtime.paged_route_name(heads, page, next_n)
    num_sms = plan.num_sms
    plan.output.fill_(float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        weights.mul_(0.5)  # a device-side input dependency must reach the launch
        plan.run()
    stream.synchronize()
    ref = _reference(q, kv_fp8, scale, weights, ctx_2d, block_table, max_len, page)
    _check(plan.logical_output, ref, ctx_2d, max_len, plan.output)
    policy = _runtime.paged_policy()
    expected_meta = _metadata_reference(
        ctx_2d,
        next_n,
        num_sms,
        int(policy["split_kv"]),
        _runtime.paged_next_n_atoms(next_n),
    )
    assert plan.schedule_meta.tolist() == expected_meta
    # One-shot entries on the same operands: identical bits.
    meta = get_paged_mqa_logits_metadata(ctx_2d, page, num_sms)
    assert torch.equal(meta, plan.schedule_meta)
    one_shot = fp8_paged_mqa_logits(
        q, kv_cache, weights, ctx_2d, block_table, meta, max_len
    )
    # Identical bits on every written cell; the one-shot entry's fresh buffer is unspecified elsewhere.
    written = ~torch.isnan(plan.logical_output)
    assert torch.equal(one_shot[written], plan.logical_output[written])
    # Graph replay with changed contents.
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    with torch.cuda.stream(stream):
        weights.neg_()
        ctx_2d.copy_((ctx_2d - 3).clamp_min(1))
        plan.output.fill_(float("nan"))
        plan.schedule_meta.fill_(0x55555555)
        graph.replay()
    stream.synchronize()
    ref = _reference(q, kv_fp8, scale, weights, ctx_2d, block_table, max_len, page)
    _check(plan.logical_output, ref, ctx_2d, max_len, plan.output)
    # Written extent vs DeepGEMM on the same operands (the replayed graph wrote the current contents).
    measured = _deep_gemm_measured(
        q, kv_cache, weights, ctx_2d, block_table, max_len, page
    )
    _check_write_extent(plan.output, measured, ctx_2d, max_len)
    if measured is None:
        return
    native, _written = measured
    inside = (
        torch.arange(max_len, device=q.device)[None, :] < ctx_2d.reshape(-1)[:, None]
    )
    torch.testing.assert_close(
        plan.logical_output[inside], native[inside], atol=1e-2, rtol=1e-2
    )


def test_paged_chunking_is_equivalent():
    """One call above the CTA budget equals the engine-style per-chunk calls."""
    heads, page, next_n = 64, 64, 1
    _skip_unless_route(heads, page, next_n)
    batch = 2 * torch.cuda.get_device_properties(0).multi_processor_count + 5
    q, kv_cache, _kv, _scale, weights, ctx_2d, block_table, max_len = _inputs(
        heads, page, next_n, batch, 1024
    )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    whole = fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        ctx_2d,
        block_table,
        get_paged_mqa_logits_metadata(ctx_2d, page, num_sms),
        max_len,
    )
    chunks = []
    for start in range(0, batch, num_sms):
        end = min(start + num_sms, batch)
        ctx = ctx_2d[start:end].contiguous()
        chunks.append(
            fp8_paged_mqa_logits(
                q[start:end],
                kv_cache,
                weights[start:end],
                ctx,
                block_table[start:end],
                get_paged_mqa_logits_metadata(ctx, page, num_sms),
                max_len,
            )
        )
    chunked = torch.cat(chunks, dim=0)
    torch.cuda.synchronize()
    inside = (
        torch.arange(max_len, device=q.device)[None, :] < ctx_2d.reshape(-1)[:, None]
    )
    assert torch.equal(whole[inside], chunked[inside])


def test_paged_host_helpers_and_rejections():
    assert _runtime.metadata_shape(148) == (149, 2)
    assert _runtime.paged_route_name(64, 64, 2) == "paged:fp8:h64:p64:n2"
    stride = _runtime.paged_logits_stride(1000)
    assert stride % 256 == 0 and stride >= 1000
    assert not _runtime.paged_route_available(16, 64, 1)
    assert not _runtime.paged_route_available(64, 128, 1)
    # Paged head counts come from policy["paged"]["heads"], not the dense heads(); one metadata atom per
    # exported next_n (whole-request programs); next_n = 3 has no program.
    policy = _runtime.paged_policy()
    assert tuple(_runtime.paged_heads()) == tuple(int(h) for h in policy["heads"])
    assert all(_runtime.paged_next_n_atoms(int(n)) == 1 for n in policy["next_n_atoms"])
    with pytest.raises(ValueError):
        _runtime.paged_next_n_atoms(3)
    assert not _runtime.paged_route_available(64, 64, 3)
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    ctx_1d = torch.full((4,), 100, device=device, dtype=torch.int32)
    with pytest.raises(ValueError):
        get_paged_mqa_logits_metadata(ctx_1d, 64, 148)
    with pytest.raises(ValueError):
        get_paged_mqa_logits_metadata(
            ctx_1d[:, None].contiguous(),
            64,
            148,
            torch.zeros(4, device=device, dtype=torch.int32),
        )
    if _runtime.paged_route_available(64, 64, 1):
        q, kv_cache, _kv, _scale, weights, ctx_2d, block_table, max_len = _inputs(
            64, 64, 1, 2, 512
        )
        meta = get_paged_mqa_logits_metadata(ctx_2d, 64, 148)
        with pytest.raises(ValueError):
            fp8_paged_mqa_logits(
                q,
                kv_cache,
                weights,
                ctx_2d,
                block_table,
                meta,
                max_len,
                clean_logits=True,
            )


@pytest.mark.parametrize("heads,page,next_n,batch,avg_ctx", CASES[:1])
def test_paged_one_shot_graph_replay(heads, page, next_n, batch, avg_ctx):
    """``get_paged_mqa_logits_metadata`` + ``fp8_paged_mqa_logits`` captured in
    a CUDA graph (the engine's decode path) launch on the capture stream: a
    replay after poisoning the captured buffers reproduces the eager bits on
    every cell inside the rows' lengths and the logits are non-trivial; an
    empty capture leaves the poison and fails."""
    _skip_unless_route(heads, page, next_n)
    q, kv_cache, _kv_fp8, _scale, weights, ctx_2d, block_table, max_len = _inputs(
        heads, page, next_n, batch, avg_ctx
    )
    num_sms = _runtime._resolve_device(q, None)[1]
    meta = get_paged_mqa_logits_metadata(ctx_2d, page, num_sms)
    eager = fp8_paged_mqa_logits(
        q, kv_cache, weights, ctx_2d, block_table, meta, max_len
    )
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_meta = get_paged_mqa_logits_metadata(ctx_2d, page, num_sms)
            captured = fp8_paged_mqa_logits(
                q, kv_cache, weights, ctx_2d, block_table, captured_meta, max_len
            )
    with torch.cuda.stream(stream):
        captured_meta.fill_(0x55555555)
        captured.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    assert torch.equal(captured_meta, meta)
    inside = torch.arange(max_len, device="cuda")[None, :] < ctx_2d.reshape(-1)[:, None]
    assert torch.equal(captured[inside], eager[inside]), (
        "graph replay does not reproduce the eager logits"
    )
    assert torch.isfinite(captured[inside]).all() and bool(
        (captured[inside] != 0).any()
    )
    with torch.cuda.stream(stream):
        weights.neg_()
        captured.fill_(float("nan"))
        graph.replay()
        changed = fp8_paged_mqa_logits(
            q, kv_cache, weights, ctx_2d, block_table, meta, max_len
        )
    stream.synchronize()
    assert torch.equal(captured[inside], changed[inside]) and not torch.equal(
        captured[inside], eager[inside]
    )
