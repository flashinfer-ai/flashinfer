"""SM120a paged FP8 MQA lightning-indexer logits (DeepGEMM signatures).

Covers every exported ``(heads, page_kv, next_n, Q-atom variant)`` program --
the batch selects the variant, so each geometry with two variants has a
small-batch and a large-batch case -- against a dequantized PyTorch reference
and an independent Python mirror of the schedule, the ``clean_logits=False``
write extent, CUDA Graph replay with changed contents, one-shot/plan
equivalence, the padded block-stride cache view, the head-count rule of the
metadata entry, the optional long-context hint (``max_context_hint``, which
applies the large-batch atom rule at batches 2-8 on every entry; 64-head
next_n 5/6 keep the reference atoms), and the host-only
route/target helpers.

The GPU tests skip while the installed catalog carries no route for the shape
or the device is not an exported architecture; the host-only helpers and the
architecture-target gate run everywhere.
"""

import pytest
import torch
from flashinfer.experimental.deepgemm_sm120_paged_mqa_logits import (
    sm120_paged_mqa as _runtime,
)
from flashinfer.sm120_paged_mqa_logits import (
    fp8_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
    prepare_sm120_paged_mqa_logits,
)

HEAD_DIM = 128
FUSED_ROW_BYTES = HEAD_DIM + 4
ATOL, RTOL = 0.1, 0.1  # FP8 e4m3
# (heads, page_kv, next_n, batch, average context).  The batch selects the
# Q-atom variant (catalog ``atom_rule``): batches up to the small-batch bound
# take the reference atomization (2-token atoms, ``a2``; ``a1`` for next_n 1),
# larger batches the native atom (``a3``..``a6``), and the 64-head next_n 5/6
# geometries take two 3-token atoms (``a3``) from the large-batch bound up.
CASES = [
    # Reference atomization: every geometry at a small batch.
    (32, 128, 1, 4, 1024),
    (32, 128, 2, 6, 2048),
    (32, 128, 3, 3, 1024),
    (32, 128, 4, 3, 1024),
    (32, 128, 5, 2, 1024),
    (32, 128, 6, 2, 512),
    (32, 64, 1, 5, 1536),
    (32, 64, 2, 8, 2048),
    (32, 64, 3, 3, 1024),
    (32, 64, 4, 3, 512),
    (32, 64, 5, 2, 1024),
    (32, 64, 6, 2, 1024),
    (64, 64, 1, 8, 2048),
    (64, 64, 2, 4, 1024),
    (64, 64, 3, 3, 1024),
    (64, 64, 4, 2, 512),
    (64, 64, 5, 2, 512),
    (64, 64, 6, 2, 1024),
    # Native atoms (one Q atom per request) above the small-batch bound.
    (32, 128, 3, 12, 512),
    (32, 128, 4, 12, 512),
    (32, 128, 5, 12, 512),
    (32, 128, 6, 12, 512),
    (32, 64, 3, 16, 512),
    (32, 64, 4, 16, 512),
    (32, 64, 5, 12, 512),
    (32, 64, 6, 12, 512),
    (64, 64, 3, 12, 512),
    (64, 64, 4, 12, 512),
    # 64 heads, next_n 5/6: two 3-token atoms from the large-batch bound up.
    (64, 64, 5, 128, 256),
    (64, 64, 6, 128, 256),
]
# (heads, page_kv, next_n, batch, average context) for the optional long-context
# hint (CAKE-1373 r2 R3).  With ``max_context_hint >= long_context_min_kv`` the
# catalog's ``selection_long`` applies: batch class 1 (2 <= B <= 8) takes the
# native atom where it fits the TMA Q box (next_n 3-6 at 32 heads, 3-4 at 64
# heads) and keeps the reference ``a2`` for the 64-head next_n 5/6 geometries.
# No new route: every hinted variant is a program the batch rule already ships.
HINT_CASES = [
    (32, 128, 3, 8, 16384),
    (32, 64, 4, 8, 16384),
    (32, 64, 5, 8, 16384),
    (64, 64, 3, 8, 16384),
    (64, 64, 6, 8, 16384),
]
# ``_inputs`` draws every request's last-token length from ``[0.7, 1.3) *
# avg_ctx``, so ``data["max_context_len"]`` is not guaranteed to reach 16384 at
# avg_ctx 16384.  The hint is an upper bound of the batch's context lengths, so
# the hint cases pass an explicit bound above every drawn length instead.
LONG_HINT = 65536
# ``selection_long`` for batch class 1 (2 <= B <= 8): the native atom wherever
# it fits the 256-row TMA Q box, else the reference pairing.
LONG_CLASS1 = {
    (32, 1): "a1",
    (32, 2): "a2",
    (32, 3): "a3",
    (32, 4): "a4",
    (32, 5): "a5",
    (32, 6): "a6",
    (64, 1): "a1",
    (64, 2): "a2",
    (64, 3): "a3",
    (64, 4): "a4",
    (64, 5): "a2",
    (64, 6): "a2",
}


def _skip_unless_route(heads, page_kv, next_n):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        _runtime.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(str(error))
    if not _runtime.route_available(heads, page_kv, next_n):
        pytest.skip(
            f"catalog has no route for H={heads}, page {page_kv}, next_n {next_n}"
        )


def _inputs(heads, page_kv, next_n, batch, avg_ctx, *, pad=0, seed=11):
    """Decode operands in the DeepGEMM/vLLM fused-page contract."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    lo = max(page_kv, int(0.7 * avg_ctx))
    hi = max(page_kv + 1, int(1.3 * avg_ctx))
    ctx_last = torch.randint(
        lo, hi, (batch,), device=device, dtype=torch.int32, generator=generator
    )
    offsets = (next_n - 1 - torch.arange(next_n, device=device, dtype=torch.int32))[
        None, :
    ]
    context_lens = (ctx_last[:, None] - offsets).clamp_min(1).contiguous()
    max_context_len = int(ctx_last.max())
    blocks = (ctx_last + page_kv - 1) // page_kv
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
    kv = torch.randn(pages, page_kv, HEAD_DIM, device=device, generator=generator)
    scale = (kv.abs().amax(dim=-1).clamp_min(1e-4) / 448.0).contiguous()
    kv_fp8 = (kv / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    row_bytes = page_kv * FUSED_ROW_BYTES + pad
    fused = torch.zeros(pages, row_bytes, device=device, dtype=torch.uint8)
    fused[:, : page_kv * HEAD_DIM] = kv_fp8.reshape(pages, -1).view(torch.uint8)
    fused[:, page_kv * HEAD_DIM : page_kv * FUSED_ROW_BYTES] = scale.view(torch.uint8)
    kv_cache = fused if pad else fused.view(pages, page_kv, 1, FUSED_ROW_BYTES)
    weights = (
        torch.rand(batch * next_n, heads, device=device, generator=generator) + 0.1
    )
    return dict(
        q=q,
        kv_cache=kv_cache,
        kv_fp8=kv_fp8,
        scale=scale,
        weights=weights,
        context_lens=context_lens,
        block_table=block_table,
        max_context_len=max_context_len,
        page_kv=page_kv,
    )


def _reference(data):
    """logits[b * next_n + t, k] = sum_h relu(q . kv[k]) * w[h] * scale[k]."""
    q, kv_fp8, scale = data["q"], data["kv_fp8"], data["scale"]
    weights, context_lens = data["weights"], data["context_lens"]
    block_table, page_kv = data["block_table"], data["page_kv"]
    max_context_len = data["max_context_len"]
    batch, next_n, _heads, _dim = q.shape
    out = torch.full(
        (batch * next_n, max_context_len),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for b in range(batch):
        ctx_last = int(context_lens[b, -1])
        n_blocks = (ctx_last + page_kv - 1) // page_kv
        phys = block_table[b, :n_blocks].long()
        keys = kv_fp8[phys].float().reshape(-1, HEAD_DIM)[:ctx_last]
        key_scale = scale[phys].reshape(-1)[:ctx_last]
        for t in range(next_n):
            scores = torch.relu(q[b, t].float() @ keys.T)
            row = (scores * weights[b * next_n + t][:, None]).sum(0) * key_scale
            ctx_t = int(context_lens[b, t])
            out[b * next_n + t, :ctx_t] = row[:ctx_t]
    return out


def _schedule_reference(context_lens, split_kv, atoms, num_sms):
    """Independent mirror of the SM120 scheduler (non-varlen branch)."""
    rows = context_lens.cpu().tolist()
    batch = len(rows)
    prefix, total_segs = [], 0
    for row in rows:
        total_segs += (row[-1] + split_kv - 1) // split_kv
        prefix.append(total_segs)
    total = total_segs * atoms
    if total == 0:
        return [[batch * atoms, 0] for _ in range(num_sms + 1)]
    per_sm, remainder = divmod(total, num_sms)
    pivot = num_sms - remainder
    meta = []
    for sm in range(num_sms):
        start = sm * per_sm + (sm - pivot if sm > pivot else 0)
        low, high = 0, batch
        while low < high:
            mid = (low + high) // 2
            if prefix[mid] * atoms <= start:
                low = mid + 1
            else:
                high = mid
        index = min(low, batch - 1)
        before = prefix[index - 1] if index > 0 else 0
        offset = start - before * atoms
        segments = prefix[index] - before
        atom = offset // segments if segments > 0 else 0
        split = offset % segments if segments > 0 else 0
        meta.append([index * atoms + atom, split])
    meta.append([batch * atoms, 0])
    return meta


def _written_extent(context_lens, row, columns):
    """Columns the logits program touches for one output row.

    The schedule issues ``ceil(context_lens[b, -1] / split_kv)`` segments of
    ``split_kv`` columns per request and nothing writes past them
    (``clean_logits=False``).
    """
    split_kv = _runtime.split_kv()
    next_n = int(context_lens.shape[1])
    ctx_last = int(context_lens[row // next_n, -1])
    return min((ctx_last + split_kv - 1) // split_kv * split_kv, columns)


def _check(logical, reference, data, physical):
    lengths = data["context_lens"].reshape(-1)
    position = torch.arange(data["max_context_len"], device=logical.device)[None, :]
    inside = position < lengths[:, None]
    assert torch.isfinite(logical[inside]).all()
    # FP8 e4m3 tolerance, the same rule the producer's validator applies.
    error = (logical - reference).abs().masked_fill(~inside, 0.0)
    tolerance = ATOL + RTOL * reference.abs().masked_fill(~inside, 0.0)
    assert bool((error <= tolerance).all())
    # Nothing is written past the schedule's per-request extent.
    for row in range(physical.shape[0]):
        extent = _written_extent(data["context_lens"], row, physical.shape[1])
        tail = physical[row, extent:]
        assert bool(torch.isnan(tail).all()), row


@pytest.mark.parametrize("heads,page_kv,next_n,batch,avg_ctx", CASES)
def test_sm120_paged_mqa_logits(heads, page_kv, next_n, batch, avg_ctx):
    _skip_unless_route(heads, page_kv, next_n)
    data = _inputs(heads, page_kv, next_n, batch, avg_ctx)
    plan = prepare_sm120_paged_mqa_logits(
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["max_context_len"],
    )
    assert plan.route_name == _runtime.route_name(heads, page_kv, next_n, batch)
    assert plan.atom_variant == _runtime.atom_variant(heads, next_n, batch)
    assert plan.route["atom_variant"] == plan.atom_variant
    assert plan.launch_count == 1 and plan.kernel_launches == 2
    assert plan.program_names == [plan.route["sequence"]]
    plan.output.fill_(float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        data["weights"].mul_(0.5)  # a device-side dependency must reach the launch
        plan.run()
    stream.synchronize()
    _check(plan.logical_output, _reference(data), data, plan.output)
    # The sequence rebuilt the schedule; it matches the independent mirror.
    atoms = _runtime.next_n_atoms(next_n, batch, heads)
    assert atoms == plan.num_next_n_atoms
    expected = _schedule_reference(
        data["context_lens"], _runtime.split_kv(), atoms, plan.num_sms
    )
    assert plan.schedule_meta.cpu().tolist() == expected
    # The standalone metadata entry produces the same schedule.
    meta = get_paged_mqa_logits_metadata(
        data["context_lens"], page_kv, plan.num_sms, num_heads=heads
    )
    assert meta.dtype == torch.int32
    assert tuple(meta.shape) == _runtime.metadata_shape(plan.num_sms)
    assert torch.equal(meta, plan.schedule_meta)
    # Without the head count the entry resolves exactly when every exported
    # head count selects this schedule, and names ``num_heads`` otherwise.
    variants = {
        _runtime.atom_variant(h, next_n, batch) for h in _runtime.exported_heads()
    }
    if len(variants) == 1:
        heads_free = get_paged_mqa_logits_metadata(
            data["context_lens"], page_kv, plan.num_sms
        )
        assert torch.equal(heads_free, plan.schedule_meta)
    else:
        with pytest.raises(ValueError, match="num_heads"):
            get_paged_mqa_logits_metadata(data["context_lens"], page_kv, plan.num_sms)
    # One-shot entry on the same operands: identical bits where both wrote.
    one_shot = fp8_paged_mqa_logits(
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        meta,
        data["max_context_len"],
    )
    written = ~torch.isnan(plan.logical_output)
    assert torch.equal(one_shot[written], plan.logical_output[written])


@pytest.mark.parametrize("heads,page_kv,next_n,batch,avg_ctx", HINT_CASES)
def test_sm120_paged_mqa_logits_long_context_hint(
    heads, page_kv, next_n, batch, avg_ctx
):
    """``max_context_hint`` selects ``selection_long`` for batch 2-8 on every entry.

    The hint is host-only: the schedule buffer carries no atom count, so the
    metadata call and the logits call of one decode step must receive the same
    hint (exactly as ``num_heads``).  The test also records the hazard: the
    same operands without the hint build a different schedule wherever the
    variant changes, and nothing at runtime can tell the two buffers apart.
    """
    _skip_unless_route(heads, page_kv, next_n)
    data = _inputs(heads, page_kv, next_n, batch, avg_ctx)
    hint = LONG_HINT
    assert hint >= data["max_context_len"]
    assert hint >= int(_runtime.atom_rule()["long_context_min_kv"])
    plan = prepare_sm120_paged_mqa_logits(
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["max_context_len"],
        max_context_hint=hint,
    )
    hinted = _runtime.atom_variant(heads, next_n, batch, max_context_hint=hint)
    unhinted = _runtime.atom_variant(heads, next_n, batch)
    assert plan.max_context_hint == hint
    assert plan.atom_variant == hinted
    assert plan.route_name == _runtime.route_name(
        heads, page_kv, next_n, batch, max_context_hint=hint
    )
    assert plan.route["atom_variant"] == plan.atom_variant
    assert plan.launch_count == 1 and plan.kernel_launches == 2
    # The 64-head 5-/6-token native atom exceeds the TMA Q box, so that
    # geometry keeps ``a2`` with or without the hint; every other case moves
    # from the reference pairing to its native atom.
    if (heads, next_n) in ((64, 5), (64, 6)):
        assert hinted == unhinted == "a2"
    else:
        assert hinted != unhinted
    plan.output.fill_(float("nan"))
    plan.run()
    torch.cuda.synchronize()
    _check(plan.logical_output, _reference(data), data, plan.output)
    # The sequence rebuilt the schedule with the hinted atom count.
    atoms = _runtime.next_n_atoms(next_n, batch, heads, max_context_hint=hint)
    assert atoms == plan.num_next_n_atoms
    expected = _schedule_reference(
        data["context_lens"], _runtime.split_kv(), atoms, plan.num_sms
    )
    assert plan.schedule_meta.cpu().tolist() == expected
    # The standalone metadata entry with the same hint produces the same schedule.
    meta = get_paged_mqa_logits_metadata(
        data["context_lens"],
        page_kv,
        plan.num_sms,
        num_heads=heads,
        max_context_hint=hint,
    )
    assert torch.equal(meta, plan.schedule_meta)
    # With the hint at batch 2-8 the exported head counts select different
    # schedules for next_n 5/6 (a5/a6 against a2), so the head-count-free call
    # names ``num_heads`` there; next_n 3/4 resolve without it.
    if next_n in (5, 6):
        with pytest.raises(ValueError, match="num_heads"):
            get_paged_mqa_logits_metadata(
                data["context_lens"], page_kv, plan.num_sms, max_context_hint=hint
            )
    else:
        heads_free = get_paged_mqa_logits_metadata(
            data["context_lens"], page_kv, plan.num_sms, max_context_hint=hint
        )
        assert torch.equal(heads_free, plan.schedule_meta)
    # The same operands WITHOUT the hint schedule the reference atoms, a
    # different buffer wherever the variant changes.  The buffer is
    # DeepGEMM-identical and carries no atom count, so a mismatch is not
    # detectable from it; the logits entries rebuild the schedule under their
    # own hint (checked above), which is why the sequence stays consistent.
    unhinted_meta = get_paged_mqa_logits_metadata(
        data["context_lens"], page_kv, plan.num_sms, num_heads=heads
    )
    if hinted == unhinted:
        assert torch.equal(unhinted_meta, plan.schedule_meta)
    else:
        assert not torch.equal(unhinted_meta, plan.schedule_meta)
    # One-shot entry with the same hint: identical bits where both wrote.
    one_shot = fp8_paged_mqa_logits(
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        meta,
        data["max_context_len"],
        max_context_hint=hint,
    )
    written = ~torch.isnan(plan.logical_output)
    assert torch.equal(one_shot[written], plan.logical_output[written])


@pytest.mark.parametrize("heads,page_kv,next_n,batch,avg_ctx", CASES[:1])
def test_sm120_paged_graph_replay(heads, page_kv, next_n, batch, avg_ctx):
    """Replay reproduces the eager bits and follows changed tensor contents."""
    _skip_unless_route(heads, page_kv, next_n)
    data = _inputs(heads, page_kv, next_n, batch, avg_ctx, seed=23)
    plan = prepare_sm120_paged_mqa_logits(
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["max_context_len"],
    )
    plan.output.fill_(float("nan"))
    plan.run()
    torch.cuda.synchronize()
    eager = plan.logical_output.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    with torch.cuda.stream(stream):
        plan.output.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    inside = (
        torch.arange(data["max_context_len"], device="cuda")[None, :]
        < data["context_lens"].reshape(-1)[:, None]
    )
    assert torch.equal(plan.logical_output[inside], eager[inside])
    # Changed inputs, including the context lengths the scheduler reads.
    with torch.cuda.stream(stream):
        data["weights"].neg_()
        data["context_lens"].copy_((data["context_lens"] - 3).clamp_min(1))
        plan.output.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    _check(plan.logical_output, _reference(data), data, plan.output)
    assert not torch.equal(plan.logical_output[inside], eager[inside])


def test_sm120_paged_padded_block_stride():
    """A page row padded past ``page_kv * 132`` bytes is accepted as a 2-D view."""
    heads, page_kv, next_n = 32, 128, 1
    _skip_unless_route(heads, page_kv, next_n)
    data = _inputs(heads, page_kv, next_n, 4, 1024, pad=512, seed=31)
    assert data["kv_cache"].ndim == 2
    plan = prepare_sm120_paged_mqa_logits(
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["max_context_len"],
        page_kv=page_kv,
    )
    assert plan.block_stride_bytes == page_kv * FUSED_ROW_BYTES + 512
    plan.output.fill_(float("nan"))
    plan.run()
    torch.cuda.synchronize()
    _check(plan.logical_output, _reference(data), data, plan.output)
    with pytest.raises(ValueError):
        # The padded view cannot be routed without an explicit page size.
        prepare_sm120_paged_mqa_logits(
            data["q"],
            data["kv_cache"],
            data["weights"],
            data["context_lens"],
            data["block_table"],
            data["max_context_len"],
        )


def test_sm120_paged_strided_per_layer_view():
    """The per-layer view of a block-outermost engine layout (every layer's page
    in one block: ``stride(0) > page_kv * 132``, not contiguous) is accepted as
    the 4-D cache and scores bitwise what the dense layout scores."""
    heads, page_kv, next_n = 32, 128, 1
    _skip_unless_route(heads, page_kv, next_n)
    data = _inputs(heads, page_kv, next_n, 4, 1024, seed=37)
    dense = data["kv_cache"]
    assert dense.ndim == 4 and dense.is_contiguous()
    pages = int(dense.shape[0])
    layer_bytes = page_kv * FUSED_ROW_BYTES
    # Two layers per block; the indexer's pages sit in the second layer slot.
    block = torch.zeros(pages, 2 * layer_bytes, device=dense.device, dtype=torch.uint8)
    block[:, layer_bytes:] = dense.reshape(pages, layer_bytes)
    strided = torch.as_strided(
        block,
        (pages, page_kv, 1, FUSED_ROW_BYTES),
        (2 * layer_bytes, FUSED_ROW_BYTES, FUSED_ROW_BYTES, 1),
        storage_offset=layer_bytes,
    )
    assert not strided.is_contiguous() and strided.stride(0) == 2 * layer_bytes
    args = (
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["max_context_len"],
    )
    plan_dense = prepare_sm120_paged_mqa_logits(data["q"], dense, *args)
    plan_dense.output.fill_(float("nan"))
    plan_dense.run()
    plan = prepare_sm120_paged_mqa_logits(data["q"], strided, *args)
    assert plan.page_kv == page_kv and plan.block_stride_bytes == 2 * layer_bytes
    plan.output.fill_(float("nan"))
    plan.run()
    torch.cuda.synchronize()
    _check(plan.logical_output, _reference(data), data, plan.output)
    lengths = data["context_lens"].reshape(-1)
    position = torch.arange(data["max_context_len"], device=dense.device)[None, :]
    inside = position < lengths[:, None]
    assert torch.equal(plan.logical_output[inside], plan_dense.logical_output[inside])
    with pytest.raises(ValueError):
        # Token rows must stay dense: a view that skips every other row is refused.
        prepare_sm120_paged_mqa_logits(
            data["q"],
            torch.as_strided(
                block,
                (pages, page_kv // 2, 1, FUSED_ROW_BYTES),
                (2 * layer_bytes, 2 * FUSED_ROW_BYTES, FUSED_ROW_BYTES, 1),
                storage_offset=layer_bytes,
            ),
            *args,
        )


def test_sm120_cases_cover_every_route():
    """Every exported (heads, page, next_n, atom) route has a GPU correctness case."""
    catalog = _runtime._catalog()
    exported = {
        name for name, record in catalog["routes"].items() if record["sequence"]
    }
    covered = {
        _runtime.route_name(heads, page_kv, next_n, batch)
        for heads, page_kv, next_n, batch, _c in CASES
    }
    assert exported, "the installed catalog exports no logits route"
    assert exported <= covered, sorted(exported - covered)
    for name in exported:
        record = catalog["routes"][name]
        assert name.endswith(f":{record['atom_variant']}")
        assert record["atom_variant"] == f"a{record['next_n_atom']}"
    # The long-context hint introduces no route: every variant it can select
    # is an exported program with a batch-only case above, and the hint cases
    # themselves resolve to exported routes.
    hinted = {
        _runtime.route_name(heads, page_kv, next_n, batch, max_context_hint=LONG_HINT)
        for heads, page_kv, next_n, batch, _c in CASES + HINT_CASES
    }
    assert hinted <= exported, sorted(hinted - exported)


def _batch_of_variant(heads, next_n, route):
    """A batch of the first class whose variant the route name carries."""
    rule = _runtime.atom_rule()
    variants = rule["selection"][f"h{heads}:n{next_n}"]
    index = variants.index(route.rsplit(":", 1)[1])
    return int(rule["batch_class_min"][index])


def test_sm120_paged_host_helpers_and_rejections():
    catalog = _runtime._catalog()
    assert catalog["schema"] == _runtime.CATALOG_SCHEMA
    assert _runtime.metadata_shape(84) == (85, 2)
    assert _runtime.split_kv() == 128
    stride = _runtime.paged_logits_stride(1000)
    assert stride % 256 == 0 and stride >= 1000
    assert not _runtime.route_available(16, 64, 1)
    assert not _runtime.route_available(64, 128, 1)
    assert _runtime.route_available(32, 64, 3)
    assert _runtime.route_available(32, 64, 3, 8)
    assert _runtime.route_available(32, 64, 3, 64)
    assert not _runtime.route_available(32, 64, 7)
    assert not _runtime.route_available(32, 64, 3, 0)
    assert _runtime.route_available(32, 64, 3, 8, max_context_hint=65536)
    with pytest.raises(ValueError):
        _runtime.next_n_atoms(7, 8)
    with pytest.raises(ValueError):
        _runtime.atom_variant(16, 1, 8)
    # A hint below one KV token is not an upper bound of anything.
    with pytest.raises(ValueError):
        _runtime.atom_variant(32, 3, 8, max_context_hint=0)
    with pytest.raises(ValueError):
        _runtime.next_n_atoms(3, 8, max_context_hint=-1)
    # The route carries the realised Q atom: the reference pairing (a2) for
    # next_n 2, the native atom above the small-batch bound, and the 64-head
    # two-3-token-atom schedule from the large-batch bound up.
    assert _runtime.route_name(64, 64, 2, 8) == "sm120:fp8:h64:p64:n2:a2"
    assert _runtime.route_name(64, 64, 2, 256) == "sm120:fp8:h64:p64:n2:a2"
    assert _runtime.route_name(32, 128, 1, 64) == "sm120:fp8:h32:p128:n1:a1"
    assert _runtime.route_name(32, 128, 4, 8) == "sm120:fp8:h32:p128:n4:a2"
    assert _runtime.route_name(32, 128, 4, 9) == "sm120:fp8:h32:p128:n4:a4"
    assert _runtime.route_name(32, 128, 5, 64) == "sm120:fp8:h32:p128:n5:a5"
    assert _runtime.route_name(64, 64, 5, 64) == "sm120:fp8:h64:p64:n5:a2"
    assert _runtime.route_name(64, 64, 5, 127) == "sm120:fp8:h64:p64:n5:a2"
    assert _runtime.route_name(64, 64, 5, 128) == "sm120:fp8:h64:p64:n5:a3"
    # The long-context hint moves batch 2-8 to the native atom (R3); the same
    # call without it keeps the reference pairing.
    assert _runtime.route_name(32, 128, 3, 8) == "sm120:fp8:h32:p128:n3:a2"
    assert (
        _runtime.route_name(32, 128, 3, 8, max_context_hint=65536)
        == "sm120:fp8:h32:p128:n3:a3"
    )
    assert (
        _runtime.route_name(64, 64, 6, 8, max_context_hint=65536)
        == "sm120:fp8:h64:p64:n6:a2"
    )
    for name in catalog["routes"]:
        if name != _runtime.metadata_route_name():
            record = catalog["routes"][name]
            assert name == _runtime.route_name(
                record["num_heads"],
                record["page_kv"],
                record["next_n"],
                _batch_of_variant(record["num_heads"], record["next_n"], name),
            )
    # Every logits route is the ordered (metadata, logits) pair behind one
    # prepared sequence; the metadata route is the standalone scheduler.
    metadata_route = _runtime.metadata_route_name()
    for name, record in catalog["routes"].items():
        stages = [stage for stage, _program in record["stages"]]
        if name == metadata_route:
            assert stages == ["metadata"] and not record["sequence"]
        else:
            assert stages == ["metadata", "logits"] and record["sequence"]
            assert record["kernel_launches"] == 2
    # The Q-atom rule is catalog data over (heads, next_n, batch class).  Table
    # check: every (heads, next_n, class) resolves to the catalog's variant and
    # atom count, and the head-count-free form resolves exactly where the
    # exported head counts agree.
    rule = _runtime.atom_rule()
    starts = [int(v) for v in rule["batch_class_min"]]
    assert starts[0] == 1 and starts == sorted(set(starts))
    assert set(_runtime.exported_next_n()) == {
        int(key.split(":")[1][1:]) for key in rule["selection"]
    }
    for key, variants in rule["selection"].items():
        heads, next_n = (int(part[1:]) for part in key.split(":"))
        assert len(variants) == len(starts)
        for index, start in enumerate(starts):
            batches = [start]
            if index + 1 < len(starts):
                batches.append(starts[index + 1] - 1)
            for batch in batches:
                assert _runtime.batch_class(batch) == index
                assert _runtime.atom_variant(heads, next_n, batch) == variants[index]
                atoms = rule["num_atoms"][f"n{next_n}:{variants[index]}"]
                assert _runtime.next_n_atoms(next_n, batch, heads) == atoms
                agreed = {
                    rule["selection"][f"h{h}:n{next_n}"][index]
                    for h in _runtime.exported_heads()
                }
                if len(agreed) == 1:
                    assert _runtime.next_n_atoms(next_n, batch) == atoms
                else:
                    with pytest.raises(ValueError, match="num_heads"):
                        _runtime.next_n_atoms(next_n, batch)
    # The optional long-context hint (CAKE-1373 r2 R3) is a second selection
    # table, ``selection_long``, that applies from ``long_context_min_kv`` KV
    # tokens up.  It differs from ``selection`` in batch class 1 (2 <= B <= 8)
    # only, where it takes the native atom that fits the TMA Q box; classes 0,
    # 2 and 3 and the batch classes themselves are unaffected, and no hint
    # (``None``) or a hint below the bound keeps the batch-only selection.
    long_min = int(rule["long_context_min_kv"])
    assert long_min == 16384
    assert set(rule["selection_long"]) == set(rule["selection"])
    geometries = {
        (record["num_heads"], record["page_kv"], record["next_n"])
        for name, record in catalog["routes"].items()
        if name != metadata_route
    }
    for heads, page_kv, next_n in geometries:
        assert _runtime.route_available(
            heads, page_kv, next_n, 8, max_context_hint=65536
        )
    for key, variants in rule["selection"].items():
        heads, next_n = (int(part[1:]) for part in key.split(":"))
        long_variants = rule["selection_long"][key]
        assert len(long_variants) == len(starts)
        for index, start in enumerate(starts):
            if index == 1:
                assert long_variants[index] == LONG_CLASS1[(heads, next_n)]
            else:
                assert long_variants[index] == variants[index]
            batches = [start]
            if index + 1 < len(starts):
                batches.append(starts[index + 1] - 1)
            for batch in batches:
                assert _runtime.batch_class(batch) == index
                for hint, table_key in (
                    (None, "selection"),
                    (long_min - 1, "selection"),
                    (long_min, "selection_long"),
                    (65536, "selection_long"),
                ):
                    table = rule[table_key]
                    variant = table[key][index]
                    assert (
                        _runtime.atom_variant(
                            heads, next_n, batch, max_context_hint=hint
                        )
                        == variant
                    )
                    atoms = rule["num_atoms"][f"n{next_n}:{variant}"]
                    assert (
                        _runtime.next_n_atoms(
                            next_n, batch, heads, max_context_hint=hint
                        )
                        == atoms
                    )
                    agreed = {
                        table[f"h{h}:n{next_n}"][index]
                        for h in _runtime.exported_heads()
                    }
                    if len(agreed) == 1:
                        assert (
                            _runtime.next_n_atoms(next_n, batch, max_context_hint=hint)
                            == atoms
                        )
                    else:
                        with pytest.raises(ValueError, match="num_heads"):
                            _runtime.next_n_atoms(next_n, batch, max_context_hint=hint)
    # Spot checks of the measured policy (CAKE-1373 r2): next_n 6 at batch 256
    # is one native atom for 32 heads and two 3-token atoms for 64 heads, so the
    # head count is required there; at batch 8 both head counts run the
    # reference 2 + 2 + 2 atomization; next_n 1-4 never need the head count.
    # Batch 1 is its own class: a single 3-token request takes one native atom,
    # every other next_n keeps the reference atoms there.
    assert _runtime.batch_class(1) == 0
    assert _runtime.batch_class(2) == 1
    assert _runtime.batch_class(8) == 1
    assert _runtime.batch_class(9) == 2
    assert _runtime.batch_class(128) == 3
    assert _runtime.next_n_atoms(3, 1) == 1
    assert _runtime.next_n_atoms(3, 2) == 2
    assert _runtime.next_n_atoms(4, 1) == 2
    assert _runtime.next_n_atoms(5, 1) == 3
    assert _runtime.next_n_atoms(6, 1, 64) == 3
    assert _runtime.next_n_atoms(6, 256, 32) == 1
    assert _runtime.next_n_atoms(6, 256, 64) == 2
    assert _runtime.next_n_atoms(6, 8, 32) == 3
    assert _runtime.next_n_atoms(6, 8, 64) == 3
    assert _runtime.next_n_atoms(6, 8) == 3
    with pytest.raises(ValueError, match="num_heads"):
        _runtime.next_n_atoms(6, 256)
    with pytest.raises(ValueError, match="num_heads"):
        _runtime.next_n_atoms(5, 64)
    assert _runtime.next_n_atoms(4, 256) == 1
    assert _runtime.next_n_atoms(4, 8) == 2
    assert _runtime.next_n_atoms(3, 8) == 2
    assert _runtime.next_n_atoms(3, 9) == 1
    assert _runtime.next_n_atoms(5, 8) == 3
    assert _runtime.next_n_atoms(1, 4096) == 1
    assert _runtime.next_n_atoms(2, 4096) == 1
    # Spot checks of the long-context hint (R3): from 16384 KV tokens up a
    # batch of 2-8 takes one native atom for next_n 3/4 at both head counts
    # and for next_n 5/6 at 32 heads, so the head count becomes required for
    # next_n 5/6 there; batch 1 and the classes from batch 9 up are unchanged,
    # and a hint below the bound is the batch-only selection.
    assert _runtime.next_n_atoms(3, 8, max_context_hint=16384) == 1
    assert _runtime.next_n_atoms(3, 8, max_context_hint=16383) == 2
    assert _runtime.next_n_atoms(4, 8, max_context_hint=65536) == 1
    assert _runtime.next_n_atoms(5, 8, 32, max_context_hint=65536) == 1
    assert _runtime.next_n_atoms(6, 8, 64, max_context_hint=65536) == 3
    with pytest.raises(ValueError, match="num_heads"):
        _runtime.next_n_atoms(5, 8, max_context_hint=65536)
    assert _runtime.next_n_atoms(3, 1, max_context_hint=65536) == 1
    assert _runtime.next_n_atoms(4, 1, max_context_hint=65536) == 2
    assert _runtime.next_n_atoms(3, 9, max_context_hint=4096) == 1
    assert _runtime.next_n_atoms(3, 9, max_context_hint=65536) == 1
    assert _runtime.next_n_atoms(6, 256, 64, max_context_hint=65536) == 2
    assert _runtime.atom_variant(32, 3, 8, max_context_hint=16384) == "a3"
    assert _runtime.atom_variant(64, 6, 8, max_context_hint=16384) == "a2"
    assert _runtime.atom_variant(32, 5, 8, max_context_hint=16384) == "a5"
    assert _runtime.atom_variant(32, 5, 8, max_context_hint=16383) == "a2"
    assert _runtime.atom_variant(32, 5, 8) == "a2"


def test_sm120_arch_targets_follow_the_compilation_context(monkeypatch):
    """Targets come from FLASHINFER_CUDA_ARCH_LIST, never a toolkit probe."""
    from flashinfer.jit.core import sm120a_nvcc_flags

    assert "-gencode=arch=compute_120a,code=sm_120a" in sm120a_nvcc_flags
    assert _runtime._nvcc_flags("sm_120a") == sm120a_nvcc_flags
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "12.0f")
    assert _runtime.supported_capabilities() == ((12, 0),)
    assert _runtime.require_supported_capability((12, 0)) == (12, 0)
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "12.0a")
    assert _runtime.supported_capabilities() == ((12, 0),)
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "10.0a 10.3a")
    assert _runtime.supported_capabilities() == ()
    with pytest.raises(RuntimeError, match="not a build target"):
        _runtime.require_supported_capability((12, 0))
    with pytest.raises(RuntimeError, match="requires compute capability"):
        _runtime.require_supported_capability((10, 0))


def test_sm120_bindings_cover_every_program_argument():
    """Every argument of every program's launcher contract is bound by name.

    Contract-level check that fails before a JIT build or a launch would: the
    sequence binding addresses its stages as ``"<stage>.<name>"`` and the
    standalone scheduler by bare argument name.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    catalog = _runtime._catalog()
    device = torch.device("cuda")
    num_sms, batch, pages, max_context_len = 4, 2, 4, 128
    checked = 0
    for name, record in catalog["routes"].items():
        if name == _runtime.metadata_route_name():
            heads, page_kv, next_n = 32, 64, 1
        else:
            heads = int(record["num_heads"])
            page_kv = int(record["page_kv"])
            next_n = int(record["next_n"])
        context_lens = torch.full(
            (batch, next_n), max_context_len, dtype=torch.int32, device=device
        )
        schedule_meta = torch.zeros(
            _runtime.metadata_shape(num_sms), dtype=torch.int32, device=device
        )
        bindings = {
            "metadata": _runtime.metadata_bindings(
                context_lens, schedule_meta, num_sms=num_sms, num_heads=heads
            )
        }
        if name != _runtime.metadata_route_name():
            q = torch.zeros(
                (batch, next_n, heads, HEAD_DIM),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            fused = torch.zeros(
                (pages, page_kv * FUSED_ROW_BYTES), dtype=torch.uint8, device=device
            )
            weights = torch.zeros(
                (batch * next_n, heads), dtype=torch.float32, device=device
            )
            block_table = torch.zeros((batch, pages), dtype=torch.int32, device=device)
            output = torch.zeros(
                (batch * next_n, _runtime.paged_logits_stride(max_context_len)),
                dtype=torch.float32,
                device=device,
            )
            bindings["logits"] = _runtime.logits_bindings(
                q,
                fused,
                weights,
                context_lens,
                block_table,
                schedule_meta,
                output,
                num_sms=num_sms,
            )
        sequence = record.get("sequence")
        if sequence:
            plan = catalog["programs"][sequence]["arg_plan"]
            missing = [
                key
                for _kind, key in plan
                if key.split(".", 1)[1] not in bindings[key.split(".", 1)[0]]
            ]
        else:
            stage, program = record["stages"][0]
            plan = catalog["programs"][program]["arg_plan"]
            missing = [key for _kind, key in plan if key not in bindings[stage]]
        assert not missing, f"{name}: unbound arguments {missing}"
        checked += 1
    assert checked == len(catalog["routes"])
