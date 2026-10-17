"""Paged prepared API (DeepGEMM paged signatures): logits against the
dequantized PyTorch paged reference (and DeepGEMM when importable), the
clean_logits=False mask, graph replay with changed contents and chunk
equivalence on SM100a/SM103a, on BOTH paths of the optional schedule-meta
input. Default path: every paged call is ONE kernel launch (the schedule is
derived in-kernel); ``get_paged_mqa_logits_metadata`` is a no-launch
DeepGEMM-signature placeholder and a tensor passed as ``schedule_meta`` keeps
this path. Table path (opt-in): ``prepare_paged_mqa_schedule_meta`` returns the
one-launch producer plan of the per-forward schedule table; a logits plan given
it as ``schedule_meta`` serves the table-consuming ``:table`` program inside the
catalog's ``sched_meta_windows`` (one logits launch, bit-identical logits) and
the default program elsewhere (``plan.schedule_meta_used``).

The tests skip while the installed catalog carries no paged route for the
shape (``paged_route_available``); the host-only helpers are tested always and
their device-dependent checks run only on a catalogued architecture.
Correctness is validated on EVERY shipped shape through
``PagedMqaPlan(enforce_admission=False)`` (the export protocol's view: the
catalog's admission rules are routing policy, not a validity bound); where the
per-architecture rules withhold a (batch, max_context_len) the public entries
must refuse it, which the same tests assert.
"""

import pytest
import torch
from flashinfer.experimental.deepgemm_dense_mqa import paged_mqa as _runtime
from flashinfer.paged_mqa import (
    fp8_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
    prepare_paged_mqa_logits,
    prepare_paged_mqa_schedule_meta,
)

HEAD_DIM = 128
# (heads, page, next_n, batch, avg context)
CASES = [
    (64, 64, 1, 1, 1024),
    (64, 64, 1, 16, 8192),
    (64, 64, 2, 16, 4096),
    (64, 64, 4, 3, 2048),
    (64, 64, 1, 256, 1024),
    (
        64,
        64,
        1,
        512,
        1024,
    ),  # batch > dynamic_scheduler.max_batch: the static program of a single-atom geometry
    (32, 64, 1, 16, 4096),
    (64, 32, 1, 16, 4096),
    # next_n-2 coverage geometries: static-preferred and admitted at batch <= 16 through
    # 5376 context on both architectures; ``_inputs`` draws max_context_len <= 5323 for
    # an average context of 4096, so every draw of these cases lands inside that window.
    (32, 64, 2, 16, 4096),
    (64, 32, 2, 16, 4096),
    # (64, 64, 2) at batch 256 and an average context of 98304: ``_inputs`` draws max_context_len in
    # 68812..127795, inside the sm_100a static-preferred window [256, 256, 43008, 169728], whose text
    # route ``paged:fp8:h64:p64:n2:hyb8c4`` serves it there (the hybrid tail pool of the static
    # program, one scheduler-state word per SM); the dynamic route on sm_103a.
    (64, 64, 2, 256, 98304),
]


def _text_routes(scheduler):
    """Every window-selected text route of ``paged_dynamic_scheduler()``."""
    return {
        route
        for table in scheduler["static_window_routes"].values()
        for pairs in table.values()
        for _window, route in pairs
    }


def _is_meta_route(route_name):
    """A schedule-meta producer route (``paged:fp8:meta:v<N>``): one ``schedule_meta`` stage, no logits program, no geometry
    in its name -- the geometry-route checks below skip it (``test_paged_sched_meta_policy_and_producer_routes`` covers it)."""
    return ":meta:" in route_name


def _catalog_paged_routes():
    from flashinfer.experimental.deepgemm_dense_mqa.dense_mqa import _catalog

    return [
        route
        for name, route in _catalog()["paged_routes"].items()
        if not _is_meta_route(name)
    ]


def _consumer_routes():
    """The table-consuming ``:table`` routes (``policy.paged.sched_meta.consumer_routes``): served only to a plan given a
    ``PagedScheduleMetaPlan`` inside the windows, never by ``paged_route_name`` (test_paged_schedule_meta_two_paths)."""
    record = _runtime.paged_sched_meta()
    return set() if record is None else set(record["consumer_routes"].values())


def _schedule_meta_reference(ctx_2d, num_sms, split_kv):
    """The schedule table the producer writes, in pure Python: ``num_segs[q] = ceil(ctx_last(q) / split_kv)`` over the
    requests' last-token lengths, the inclusive prefix, ``total`` segments split as ``sm * (total // num_sms) + min(sm,
    total % num_sms)`` and each start mapped back to its ``(request, kv_split)`` pair for ``sm`` in ``0..num_sms`` (one
    atom per request: the paged programs iterate the next_n atoms in-kernel). Returns ``num_sms + 1`` pairs."""
    lens = [int(v) for v in ctx_2d.reshape(-1).tolist()]
    batch, next_n = (int(v) for v in ctx_2d.shape)
    num_segs = [
        (lens[q * next_n + next_n - 1] + split_kv - 1) // split_kv for q in range(batch)
    ]
    prefix, acc = [], 0
    for n in num_segs:
        acc += n
        prefix.append(acc)
    total = prefix[-1] if batch else 0
    qd, rd = (total // num_sms, total % num_sms) if num_sms > 0 else (0, 0)
    table = []
    for sm in range(num_sms + 1):
        start = sm * qd + min(sm, rd)
        q = 0
        while q < batch and prefix[q] <= start:
            q += 1
        prev = prefix[q - 1] if q > 0 else 0
        cur = prefix[q] if q < batch else total
        offset, n = start - prev, cur - prev
        table.append([q + (offset // n if n > 0 else 0), offset % n if n > 0 else 0])
    return table


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


def _catalogued_device_arch():
    """The catalog architecture of the current CUDA device, or ``None`` without a CUDA device or on a
    device whose architecture ships no programs: the device-dependent half of a host test then ends."""
    if not torch.cuda.is_available():
        return None
    from flashinfer.experimental.deepgemm_dense_mqa import dense_mqa

    try:
        return dense_mqa.device_arch(torch.device("cuda"))
    except RuntimeError:
        return None


def _admitted(q, ctx_2d, max_len, heads, page, next_n):
    """The shipped per-architecture verdict for this exact call (batch = ``ctx_2d.shape[0]``)."""
    arch = _runtime._resolve_device(q, None)[0]
    return _runtime.paged_route_admitted(
        arch,
        _runtime.paged_route_name(
            heads,
            page,
            next_n,
            batch=int(ctx_2d.shape[0]),
            max_context_len=int(max_len),
            arch=arch,
        ),
        int(ctx_2d.shape[0]),
        int(max_len),
    )


def _validation_plan(q, kv_cache, weights, ctx_2d, block_table, max_len, **kwargs):
    """The plan the export validates with: every shipped shape, admission rules bypassed."""
    return _runtime.PagedMqaPlan(
        q,
        kv_cache,
        weights,
        ctx_2d,
        block_table,
        max_len,
        enforce_admission=False,
        **kwargs,
    )


def _one_shot(q, kv_cache, weights, ctx_2d, block_table, num_sms, max_len):
    """``fp8_paged_mqa_logits`` minus the admission check (same plan, same single ``run``)."""
    return _validation_plan(
        q, kv_cache, weights, ctx_2d, block_table, max_len, sm_count=num_sms
    ).run()


def _inputs(heads, page, next_n, batch, avg_ctx, seed=7, uniform=False):
    """Engine-layout operands; ``uniform`` pins every request's length to ``avg_ctx`` (the extent is then exactly ``avg_ctx``),
    otherwise the lengths are drawn from ``[0.7 avg_ctx, 1.3 avg_ctx)``."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    lo, hi = max(page, int(0.7 * avg_ctx)), max(page + 1, int(1.3 * avg_ctx))
    if uniform:
        ctx_max = torch.full((batch,), int(avg_ctx), device=device, dtype=torch.int32)
    else:
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
    admitted = _admitted(q, ctx_2d, max_len, heads, page, next_n)
    if admitted:
        plan = prepare_paged_mqa_logits(
            q, kv_cache, weights, ctx_2d, block_table, max_len
        )
    else:
        # Withheld on this architecture: the public entry refuses, the kernel is validated through the plan.
        with pytest.raises(ValueError, match="is not admitted on"):
            prepare_paged_mqa_logits(q, kv_cache, weights, ctx_2d, block_table, max_len)
        plan = _validation_plan(q, kv_cache, weights, ctx_2d, block_table, max_len)
    # The plan's route is the shipped per-call rule under the device's CTA budget: the rect / compact routes when
    # the (request, KV split) rectangle fits one wave (catalog v11), else the dynamic-scheduler route by batch and
    # the static route inside this architecture's static-preferred windows (batch, max_context_len).
    assert plan.route_name == _runtime.paged_route_name(
        heads,
        page,
        next_n,
        batch=batch,
        max_context_len=max_len,
        arch=plan.arch,
        num_sms=plan.num_sms,
    )
    static = _runtime.paged_route_name(heads, page, next_n)
    scheduler = _runtime.paged_dynamic_scheduler()
    if plan.route_name == static and static in scheduler["routes"]:
        preferred = _runtime.paged_static_preferred(plan.arch, static, batch, max_len)
        assert batch > scheduler["max_batch"] or preferred
    if plan.route_name in _text_routes(scheduler):
        # a window-selected text of the static program: inside a static-preferred window that names it, the
        # plan's scheduler state is the route's (one word per SM for the hybrid texts)
        assert _runtime.paged_static_preferred(plan.arch, static, batch, max_len)
        assert (
            _runtime.paged_static_window_route(plan.arch, static, batch, max_len)
            == plan.route_name
        )
    assert plan.sched_counters.numel() == _runtime.paged_sched_state_words(
        plan.route_name, plan.num_sms
    )
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
    assert plan.launch_count == 1 and plan.program_names == [
        plan.route["stages"][0][1][plan.arch]
    ]
    # One-shot entries on the same operands: identical bits. The metadata entry is a no-launch
    # placeholder of the DeepGEMM signature (zeros of the CTA-budget shape).
    meta = get_paged_mqa_logits_metadata(ctx_2d, page, num_sms)
    assert (
        meta.dtype == torch.int32
        and tuple(meta.shape) == (num_sms + 1, 2)
        and not meta.any()
    )
    if admitted:
        one_shot = fp8_paged_mqa_logits(
            q, kv_cache, weights, ctx_2d, block_table, meta, max_len
        )
    else:
        with pytest.raises(ValueError, match="is not admitted on"):
            fp8_paged_mqa_logits(
                q, kv_cache, weights, ctx_2d, block_table, meta, max_len
            )
        one_shot = _one_shot(
            q, kv_cache, weights, ctx_2d, block_table, num_sms, max_len
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
    """One call above the CTA budget equals the engine-style per-chunk calls (kernel property, validated
    through the plan on every architecture; the public entry's verdict for the whole call follows the
    shipped admission rules for batch 2 * SMs + 5)."""
    heads, page, next_n = 64, 64, 1
    _skip_unless_route(heads, page, next_n)
    batch = 2 * torch.cuda.get_device_properties(0).multi_processor_count + 5
    q, kv_cache, _kv, _scale, weights, ctx_2d, block_table, max_len = _inputs(
        heads, page, next_n, batch, 1024
    )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    whole_meta = get_paged_mqa_logits_metadata(ctx_2d, page, num_sms)
    if _admitted(q, ctx_2d, max_len, heads, page, next_n):
        whole = fp8_paged_mqa_logits(
            q, kv_cache, weights, ctx_2d, block_table, whole_meta, max_len
        )
    else:
        with pytest.raises(ValueError, match="is not admitted on"):
            fp8_paged_mqa_logits(
                q, kv_cache, weights, ctx_2d, block_table, whole_meta, max_len
            )
        whole = _one_shot(q, kv_cache, weights, ctx_2d, block_table, num_sms, max_len)
    chunks = []
    for start in range(0, batch, num_sms):
        end = min(start + num_sms, batch)
        ctx = ctx_2d[start:end].contiguous()
        chunks.append(
            _one_shot(
                q[start:end],
                kv_cache,
                weights[start:end],
                ctx,
                block_table[start:end],
                num_sms,
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
    # Paged head counts come from policy["paged"]["heads"], not the dense heads(); one schedule atom per
    # exported next_n (whole-request programs); next_n = 3 has no program; no metadata program at all.
    policy = _runtime.paged_policy()
    assert policy["metadata_program"] is None
    assert all(
        len(route["stages"]) == 1 and route["stages"][0][0] == "logits"
        for route in _catalog_paged_routes()
    )
    assert tuple(_runtime.paged_heads()) == tuple(int(h) for h in policy["heads"])
    assert all(_runtime.paged_next_n_atoms(int(n)) == 1 for n in policy["next_n_atoms"])
    with pytest.raises(ValueError):
        _runtime.paged_next_n_atoms(3)
    assert not _runtime.paged_route_available(64, 64, 3)
    if _catalogued_device_arch() is None:
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
        assert tuple(meta.shape) == (149, 2) and not meta.any()
        out = torch.full((149, 2), 7, dtype=torch.int32, device=device)
        assert (
            get_paged_mqa_logits_metadata(ctx_2d, 64, 148, out=out) is out
            and not out.any()
        )
        with pytest.raises(ValueError):
            get_paged_mqa_logits_metadata(ctx_2d, 3, 148)  # block_kv 3 is not exported
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
        with pytest.raises(ValueError):
            fp8_paged_mqa_logits(
                q,
                kv_cache,
                weights,
                ctx_2d,
                block_table,
                torch.zeros(1, 2, dtype=torch.int32, device=device),
                max_len,
            )


@pytest.mark.parametrize("heads,page,next_n,batch,avg_ctx", CASES[:1])
def test_paged_one_shot_graph_replay(heads, page, next_n, batch, avg_ctx):
    """``fp8_paged_mqa_logits`` (the engine's decode call, one launch) captured in
    a CUDA graph launches on the capture stream: a replay after poisoning the
    captured buffer reproduces the eager bits on every cell inside the rows'
    lengths and the logits are non-trivial; an empty capture leaves the poison
    and fails. ``schedule_meta=None`` is the signature-parity path."""
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
            captured = fp8_paged_mqa_logits(
                q, kv_cache, weights, ctx_2d, block_table, None, max_len
            )
    with torch.cuda.stream(stream):
        captured.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
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


def test_paged_bindings_cover_every_program_argument():
    """Every operand of every paged program's launcher contract (the catalog ``arg_plan``) is produced
    by the runtime's binding builders, and every paged route is the single ``logits`` stage. This is the
    contract-level check that fails before a prebuild or launch would: the single-kernel logits programs
    take the CTA budget ``num_sms`` as a kernel parameter, which the runtime must bind as well as use for
    the launch grid (an earlier export left it unbound); no program binds a schedule buffer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    from flashinfer.experimental.deepgemm_dense_mqa.dense_mqa import _catalog

    catalog = _catalog()
    device = torch.device("cuda")
    num_sms, batch, pages, max_len = 4, 2, 4, 100
    checked = 0
    counters = torch.zeros(2, dtype=torch.uint32, device=device)
    for route_name, route in catalog["paged_routes"].items():
        if _is_meta_route(route_name):
            continue  # the producer routes bind the schedule table by design (their own test)
        fields = {field[0]: int(field[1:]) for field in route_name.split(":")[2:5]}
        heads, page, next_n = fields["h"], fields["p"], fields["n"]
        ctx_2d = torch.full((batch, next_n), max_len, dtype=torch.int32, device=device)
        q = torch.zeros(
            (batch, next_n, heads, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
        )
        kv_cache = torch.zeros(
            (pages, page, 1, _runtime.FUSED_ROW_BYTES), dtype=torch.uint8, device=device
        )
        weights = torch.zeros(
            (batch * next_n, heads), dtype=torch.float32, device=device
        )
        block_table = torch.zeros((batch, pages), dtype=torch.int32, device=device)
        output = torch.zeros(
            (batch * next_n, _runtime.paged_logits_stride(max_len)),
            dtype=torch.float32,
            device=device,
        )
        bindings = {
            "logits": _runtime.logits_bindings(
                q,
                kv_cache,
                weights,
                ctx_2d,
                block_table,
                output,
                num_sms=num_sms,
                sched_counters=counters,
            ),
        }
        assert [stage for stage, _programs in route["stages"]] == ["logits"], route_name
        for stage, program in (
            (stage, program)
            for stage, by_arch in route["stages"]
            for program in sorted(set(by_arch.values()))
        ):
            arg_plan = catalog["programs"][program]["arg_plan"]
            missing = [name for _kind, name in arg_plan if name not in bindings[stage]]
            assert not missing, (
                f"{route_name} stage {stage} ({program}): unbound arguments {missing}"
            )
            assert "schedule_meta" not in {name for _kind, name in arg_plan}, (
                f"{program} binds a schedule buffer"
            )
            checked += 1
        if route_name in _consumer_routes():
            # the table-consuming program takes the table through ``sched_counters`` (the same contract, no extra operand)
            # and is a program of its own next to the static route's on every architecture it is delivered for
            static = route_name.rsplit(":", 1)[0]
            for arch, program in route["stages"][0][1].items():
                assert (
                    program != catalog["paged_routes"][static]["stages"][0][1][arch]
                ), (route_name, arch)
                assert "sched_counters" in {
                    name for _kind, name in catalog["programs"][program]["arg_plan"]
                }
    assert checked > 0


def test_paged_dynamic_scheduler_routes():
    """``policy.paged.dynamic_scheduler`` maps single-atom static routes to their ``:dyn`` routes, both in the
    catalog with one logits stage whose programs differ; ``paged_route_name(..., batch=)`` selects the dynamic
    route up to ``max_batch`` and the static route above it (and without a batch); every program of a dynamic
    route binds the scheduler state ``sched_counters``."""
    from flashinfer.experimental.deepgemm_dense_mqa.dense_mqa import _catalog

    catalog = _catalog()
    dynamic = _runtime.paged_dynamic_scheduler()
    if not dynamic["routes"]:
        pytest.skip("catalog ships no dynamic-scheduler program")
    assert dynamic["max_batch"] >= 1
    for static, dyn in dynamic["routes"].items():
        assert static in catalog["paged_routes"] and dyn in catalog["paged_routes"]
        assert dyn == f"{static}:dyn"
        for arch in sorted(catalog["arches"]):
            static_program = catalog["paged_routes"][static]["stages"][0][1][arch]
            dyn_program = catalog["paged_routes"][dyn]["stages"][0][1][arch]
            # Program ids are content hashes of the generated text: the two routes serve distinct programs on
            # every architecture, and only the dynamic one binds the scheduler state.
            assert dyn_program != static_program, (static, arch)
            assert (
                catalog["programs"][dyn_program]["role"]
                == catalog["programs"][static_program]["role"]
            )
            assert "sched_counters" in {
                name for _kind, name in catalog["programs"][dyn_program]["arg_plan"]
            }
        fields = {field[0]: int(field[1:]) for field in static.split(":")[2:5]}
        heads, page, next_n = fields["h"], fields["p"], fields["n"]
        assert _runtime.paged_route_name(heads, page, next_n) == static
        assert _runtime.paged_route_name(heads, page, next_n, batch=1) == dyn
        assert (
            _runtime.paged_route_name(heads, page, next_n, batch=dynamic["max_batch"])
            == dyn
        )
        assert (
            _runtime.paged_route_name(
                heads, page, next_n, batch=dynamic["max_batch"] + 1
            )
            == static
        )
    text_routes = _text_routes(dynamic)
    for route_name in catalog["paged_routes"]:
        if (
            route_name.endswith(":dyn")
            or route_name.endswith(
                ":rect"
            )  # catalog v11: selected only with num_sms (test_paged_rect_routes)
            or route_name.endswith(
                ":compact"
            )  # catalog v11 (round-4 compact-walk adoption): selected only with num_sms, above one wave
            or _is_meta_route(
                route_name
            )  # the schedule-meta producer routes: no geometry, never a logits route
            or route_name
            in _consumer_routes()  # the table path: served only to a plan given a PagedScheduleMetaPlan
            or route_name in dynamic["routes"]
            or route_name in text_routes
        ):
            continue
        fields = {field[0]: int(field[1:]) for field in route_name.split(":")[2:5]}
        assert (
            _runtime.paged_route_name(fields["h"], fields["p"], fields["n"], batch=1)
            == route_name
        )
    # window-selected text routes (catalog v10): each is catalogued, delivered exactly on the architectures whose table
    # names it, serves a program distinct from the static and dynamic ones there, binds the scheduler state its
    # ``sched_state`` record sizes, and inherits the static route's admission rules
    for arch, table in dynamic["static_window_routes"].items():
        for static, pairs in table.items():
            assert static in dynamic["routes"], (arch, static)
            for window, route in pairs:
                assert route in catalog["paged_routes"] and route.startswith(
                    static + ":"
                ), route
                assert window in dynamic["static_windows"][arch][static], (
                    route,
                    window,
                )
                programs = catalog["paged_routes"][route]["stages"][0][1]
                assert arch in programs, (route, arch)
                for other in catalog["arches"]:
                    named = any(
                        route == r
                        for _s, ps in dynamic["static_window_routes"]
                        .get(other, {})
                        .items()
                        for _w, r in ps
                    )
                    assert (other in programs) == named, (route, other)
                program = programs[arch]
                static_program = catalog["paged_routes"][static]["stages"][0][1][arch]
                dyn_program = catalog["paged_routes"][dynamic["routes"][static]][
                    "stages"
                ][0][1][arch]
                assert program not in (static_program, dyn_program), route
                assert "sched_counters" in {
                    name for _kind, name in catalog["programs"][program]["arg_plan"]
                }
                base, per_sm, max_sms = dynamic["sched_state"][route]
                assert base >= 2 and per_sm >= 1 and max_sms >= SM_COUNT[arch], route
                assert (
                    _runtime.paged_sched_state_words(route, SM_COUNT[arch])
                    == base + per_sm * SM_COUNT[arch]
                )
                admission = catalog["policy"]["paged"]["admission"][arch]
                assert admission.get(route) == admission.get(static), (route, static)
                min_batch, max_batch, min_ctx, max_ctx = window
                batch = min_batch if max_batch is None else max_batch
                ctx = max(min_ctx, 256) if max_ctx is None else max_ctx
                assert (
                    _runtime.paged_route_name(
                        int(static.split(":h")[1].split(":")[0]),
                        int(static.split(":p")[1].split(":")[0]),
                        int(static.split(":n")[1].split(":")[0]),
                        batch=batch,
                        max_context_len=ctx,
                        arch=arch,
                    )
                    == route
                )
    assert _runtime.paged_sched_state_words("paged:fp8:h64:p64:n1", 148) == 2
    # static-preferred windows: inside one, a batch the dynamic route would take stays on the static program when the
    # call's architecture and max_context_len are given; without them (and outside the windows) the rule above holds.
    # Each parsed window is (min_batch, max_batch, min_context_len, max_context_len): the catalog's two-field
    # entries start at batch 1 and context 1, its three-field entries carry a lower batch bound, its four-field
    # entries a context lower bound as well (test_paged_static_window_lower_bound).
    for arch, table in dynamic["static_windows"].items():
        for static, windows in table.items():
            assert static in dynamic["routes"] and windows, (arch, static)
            fields = {field[0]: int(field[1:]) for field in static.split(":")[2:5]}
            for min_batch, max_batch, min_ctx, max_ctx in windows:
                assert 1 <= min_batch <= dynamic["max_batch"]
                assert (
                    max_batch is None or min_batch <= max_batch <= dynamic["max_batch"]
                )
                assert min_ctx >= 1 and (max_ctx is None or max_ctx >= min_ctx)
                batch = min_batch if max_batch is None else max_batch
                ctx = max(min_ctx, 256) if max_ctx is None else max_ctx
                assert _runtime.paged_static_preferred(arch, static, batch, ctx)
                h, p, n = fields["h"], fields["p"], fields["n"]
                at_ctx = {"batch": batch, "max_context_len": ctx, "arch": arch}
                # the static route, or the text route the window names (catalog v10)
                expected = (
                    _runtime.paged_static_window_route(arch, static, batch, ctx)
                    or static
                )
                assert _runtime.paged_route_name(h, p, n, **at_ctx) == expected
                dyn_route = dynamic["routes"][static]
                assert _runtime.paged_route_name(h, p, n, batch=batch) == dyn_route
                if max_ctx is not None:
                    ctx1 = max_ctx + 1
                    past = {"batch": batch, "max_context_len": ctx1, "arch": arch}
                    beyond = _runtime.paged_route_name(h, p, n, **past) == dyn_route
                    pref = _runtime.paged_static_preferred(arch, static, batch, ctx1)
                    assert beyond or pref


# The SM count of each catalogued architecture (B200 148, GB300 152): the batch of the (64, 64, 2)
# batch-lower-bounded static window, pinned so the probe below is deterministic per architecture.
SM_COUNT = {"sm_100a": 148, "sm_103a": 152}


def test_paged_static_window_lower_bound(monkeypatch):
    """A ``static_windows`` entry may carry a lower batch bound (``[min_batch, max_batch, max_context_len]``):
    the (64, 64, 2) route keeps the static program at batch == the SM count through 5376 context on both
    architectures, while batch 64 / 128, the neighbouring batches and the contexts between the ctx-4k and ctx-8k
    classes take the dynamic route; the static route admits the window; an entry of another length is refused when
    the policy is read. Round 4 (H35-b): the n1 route carries the same SM-count literal ``[sms, sms, 5376]`` and
    ``[65, 128, 5376]``; round 4 (H35-c): ``[128, sms, 5632, 10752]`` on n2 / ``[64, sms, 5632, 10752]`` on n1 keep
    the static program from 5632 through 10752 context, above it the dynamic route."""
    from flashinfer.experimental.deepgemm_dense_mqa import dense_mqa

    dynamic = _runtime.paged_dynamic_scheduler()
    static, dyn = "paged:fp8:h64:p64:n2", "paged:fp8:h64:p64:n2:dyn"
    if dynamic["routes"].get(static) != dyn:
        pytest.skip("catalog ships no dynamic-scheduler program for h64 p64 n2")

    def route(arch, batch, ctx):
        return _runtime.paged_route_name(
            64, 64, 2, batch=batch, max_context_len=ctx, arch=arch
        )

    real = dense_mqa._catalog()
    for arch in sorted(real["arches"]):
        sms = SM_COUNT[arch]
        windows = dynamic["static_windows"][arch][static]
        assert (sms, sms, 1, 5376) in windows and (1, 64, 1, 3840) in windows, (
            arch,
            windows,
        )
        # the engine extents of an average-4096 call (``_inputs`` draws max_context_len <= 5323)
        for ctx in (2880, 4096, 5323, 5376):
            assert route(arch, sms, ctx) == static, (arch, ctx)
        assert _runtime.paged_static_preferred(arch, static, sms, 5376)
        assert (
            route(arch, sms, 5377) == dyn and route(arch, sms, 5631) == dyn
        )  # between the ctx-4k and ctx-8k classes
        assert (
            route(arch, sms, 8192) == static and route(arch, sms, 10753) == dyn
        )  # r4 H35-c: the n2 ctx-8k window [128, sms, 5632, 10752]
        for batch in (64, 128, sms - 1, sms + 1):
            assert route(arch, batch, 4096) == dyn, (arch, batch)
        # the batch-64 edge from batch 1 is unchanged; the other architecture's SM count is not this one's window
        assert route(arch, 64, 3840) == static
        other = next(count for a, count in SM_COUNT.items() if a != arch)
        assert route(arch, other, 4096) == dyn
        assert (
            _runtime.paged_route_name(
                64, 64, 1, batch=sms, max_context_len=4096, arch=arch
            )
            == "paged:fp8:h64:p64:n1"  # r4 H35-b: the n1 SM-count window [sms, sms, 5376]
        )
        assert (
            _runtime.paged_route_name(
                64, 64, 1, batch=sms, max_context_len=5377, arch=arch
            )
            == "paged:fp8:h64:p64:n1:dyn"
        )
        assert (
            _runtime.paged_route_name(
                64, 64, 1, batch=128, max_context_len=4096, arch=arch
            )
            == "paged:fp8:h64:p64:n1"
        )  # r4 H35-b: [65, 128, 5376]
        assert (
            _runtime.paged_route_name(
                64, 64, 1, batch=129, max_context_len=4096, arch=arch
            )
            == "paged:fp8:h64:p64:n1:dyn"
        )
        # the static route admits every call its window selects (the catalog invariant)
        assert _runtime.paged_route_admitted(arch, static, sms, 5376)
        # the context-lower-bounded four-field form (catalog v10): sm_100a keeps the static program at batch 256 from
        # 43008 through 169728 context (the measured 128k rows) and the dynamic one below (the measured 32k rows)
        if arch == "sm_100a":
            assert (256, 256, 43008, 169728) in windows, windows
            # the window names the hybrid-tail-pool text of the static program (catalog v10): its own route, the
            # static route's admission, one scheduler-state word per SM on top of the two
            hyb = "paged:fp8:h64:p64:n2:hyb8c4"
            assert ((256, 256, 43008, 169728), hyb) in dynamic["static_window_routes"][
                arch
            ][static]
            for ctx in (43008, 131072, 169664, 169728):
                assert route(arch, 256, ctx) == hyb, ctx
                assert _runtime.paged_static_window_route(arch, static, 256, ctx) == hyb
                assert _runtime.paged_static_preferred(arch, static, 256, ctx)
                assert _runtime.paged_route_admitted(arch, hyb, 256, ctx)
                assert _runtime.paged_route_admitted(arch, static, 256, ctx)
            for ctx in (32768, 42624, 42752, 43007, 169729, 169984):
                assert route(arch, 256, ctx) == dyn, ctx
                assert (
                    _runtime.paged_static_window_route(arch, static, 256, ctx) is None
                )
            for batch in (255, 257):
                assert route(arch, batch, 131072) == dyn, batch
            assert _runtime.paged_sched_state_words(hyb, sms) == 2 + sms
            assert _runtime.paged_sched_state_words(static, sms) == 2
            with pytest.raises(ValueError, match="at most"):
                _runtime.paged_sched_state_words(hyb, 257)
        else:
            # sm_103a carries no catalog-v10 b256 window and no text route; its only context-lower-bounded window is
            # the r4 H35-c n2 ctx-8k window pinned above (every published window reads as the four-field form here)
            assert [w for w in windows if w[2] > 1] == [(128, sms, 5632, 10752)], (
                windows
            )
            assert route(arch, 256, 131072) == dyn
            assert not dynamic["static_window_routes"].get(arch), arch
    # an entry of another length is a catalog error, refused when the policy is read
    patched = dict(real)
    patched["policy"] = dict(real["policy"])
    paged = real["policy"]["paged"]
    patched["policy"]["paged"] = dict(
        paged,
        dynamic_scheduler=dict(
            paged["dynamic_scheduler"],
            static_windows={"sm_100a": {static: [[1, 2, 3, 4, 5]]}},
        ),
    )
    monkeypatch.setattr(dense_mqa, "_catalog", lambda: patched)
    monkeypatch.setattr(_runtime, "_catalog", lambda: patched)
    with pytest.raises(ValueError, match="static window"):
        _runtime.paged_dynamic_scheduler()
    # a window route that is not a text route of its static route, and a malformed sched_state record, are refused too
    for bad in (
        dict(
            static_window_routes={
                "sm_100a": {
                    static: [[[256, 256, 43008, 169728], "paged:fp8:h64:p64:n1"]]
                }
            }
        ),
        dict(static_window_routes={"sm_100a": {static: [[[256, 256, 43008, 169728]]]}}),
        dict(sched_state={"paged:fp8:h64:p64:n2:hyb8c4": {"base_words": 2}}),
        dict(
            sched_state={
                "paged:fp8:h64:p64:n2:hyb8c4": {
                    "base_words": 0,
                    "words_per_sm": 1,
                    "max_sms": 256,
                }
            }
        ),
    ):
        patched["policy"]["paged"] = dict(
            paged, dynamic_scheduler=dict(paged["dynamic_scheduler"], **bad)
        )
        with pytest.raises(ValueError, match="static window route|sched_state"):
            _runtime.paged_dynamic_scheduler()


def test_paged_admission_rules_are_applied_per_arch(monkeypatch):
    """``policy.paged.admission`` rules a (arch, route) pair by (batch, max_context_len): admitted iff some
    ``(max_batch, max_context_len)`` rule covers the call, unbounded where absent; the plan constructor
    refuses a withheld call before any build (the export's ``enforce_admission=False`` bypasses it)."""
    from flashinfer.experimental.deepgemm_dense_mqa import dense_mqa

    real = dense_mqa._catalog()
    heads, page = 64, 64
    route_n2, route_n1 = (
        _runtime.paged_route_name(heads, page, 2),
        _runtime.paged_route_name(heads, page, 1),
    )
    if route_n2 not in real["paged_routes"] or route_n1 not in real["paged_routes"]:
        pytest.skip("catalog lacks the h64 p64 n1/n2 paged routes")
    # The shipped policy itself: well-formed rules on catalogued architectures and routes.
    paged = real["policy"]["paged"]
    assert int(paged["measured_max_context_len"]) % int(paged["split_kv"]) == 0
    for arch, table in paged["admission"].items():
        assert arch in real["arches"], arch
        assert isinstance(paged["admission_reason"].get(arch), str) or not table
        for route, rules in table.items():
            assert route in real["paged_routes"], route
            batches = [rule[0] for rule in rules]
            finite = [b for b in batches if b is not None]
            assert finite == sorted(finite) and all(b >= 1 for b in finite), (
                route,
                rules,
            )
            assert None not in batches[:-1], (route, rules)
            for _max_batch, max_ctx in rules:
                assert max_ctx is None or (
                    int(max_ctx) >= 1 and int(max_ctx) % int(paged["split_kv"]) == 0
                )
    # Rule semantics on a patched policy.
    patched = dict(real)
    patched["policy"] = dict(real["policy"])
    # The rules are exercised on the static route: the patched policy ships no dynamic-scheduler map, so the
    # batch-aware route resolution of ``paged_route_available`` selects ``route_n2`` at every batch.
    patched["policy"]["paged"] = dict(
        paged,
        dynamic_scheduler={},
        admission={"sm_100a": {route_n2: [[16, None], [64, 32768], [None, 8192]]}},
    )
    monkeypatch.setattr(dense_mqa, "_catalog", lambda: patched)
    monkeypatch.setattr(_runtime, "_catalog", lambda: patched)
    assert _runtime.paged_admission_rules("sm_100a", route_n2) == [
        (16, None),
        (64, 32768),
        (None, 8192),
    ]
    assert _runtime.paged_admission_rules("sm_103a", route_n2) is None
    assert _runtime.paged_admission_rules("sm_100a", route_n1) is None
    admitted = lambda batch, ctx: _runtime.paged_route_available(
        heads, page, 2, arch="sm_100a", batch=batch, max_context_len=ctx
    )
    assert _runtime.paged_route_available(
        heads, page, 2
    )  # no arch: table availability only
    assert admitted(16, 1 << 20) and admitted(1, 131072)
    assert admitted(17, 32768) and not admitted(17, 32769)
    assert admitted(64, 32768) and not admitted(65, 32768)
    assert admitted(65, 8192) and admitted(4096, 8192) and not admitted(4096, 8448)
    assert _runtime.paged_route_available(
        heads, page, 2, arch="sm_103a", batch=4096, max_context_len=1 << 20
    )
    assert _runtime.paged_route_available(
        heads, page, 1, arch="sm_100a", batch=4096, max_context_len=1 << 20
    )
    with pytest.raises(ValueError, match="requires batch and max_context_len"):
        _runtime.paged_route_available(heads, page, 2, arch="sm_100a")
    monkeypatch.undo()
    arch = _catalogued_device_arch()
    if arch is None:
        return
    # Live refusal under the shipped rules of this device's architecture (and the export bypass). The plan
    # resolves the route from the call's batch (the dynamic-scheduler route up to its max_batch, the static
    # route above it), so every probe is judged on the route the plan would select.
    probes = ((17, 1024), (17, 32769), (65, 8448), (129, 8448), (4096, 131072))
    route_for = lambda b, c: _runtime.paged_route_name(
        heads, page, 2, batch=b, max_context_len=c, arch=arch
    )
    rules = {
        route_for(b, c): _runtime.paged_admission_rules(arch, route_for(b, c))
        for b, c in probes
    }
    if all(r is None for r in rules.values()):
        return
    withheld = next(
        (
            (b, c)
            for b, c in probes
            if not _runtime.paged_route_admitted(arch, route_for(b, c), b, c)
        ),
        None,
    )
    assert withheld is not None, (
        f"shipped rules {rules} admit every probe; extend the probe list"
    )
    batch, ctx = withheld
    q, kv_cache, _kv, _scale, weights, ctx_2d, block_table, _max_len = _inputs(
        heads, page, 2, batch, 1024
    )
    # A block table wide enough for both the generated rows and the probed max_context_len.
    width = max(int(block_table.shape[1]), (ctx + page - 1) // page)
    wide = torch.zeros((batch, width), dtype=torch.int32, device=q.device)
    wide[:, : block_table.shape[1]] = block_table
    with pytest.raises(ValueError, match="is not admitted on"):
        _runtime.PagedMqaPlan(q, kv_cache, weights, ctx_2d, wide, ctx)
    plan = _runtime.PagedMqaPlan(
        q, kv_cache, weights, ctx_2d, wide, ctx, enforce_admission=False
    )
    assert plan.batch == batch and plan.max_context_len == ctx


def test_paged_rect_routes():
    """``policy.paged.rect`` (catalog v11) maps every single-atom static route to its ``:rect`` route, catalogued with one
    logits stage whose program differs from the static and dynamic ones on every architecture; ``paged_route_name(...,
    num_sms=)`` selects it FIRST exactly when ``batch * ceil(max_context_len / split_kv) <= max_waves * num_sms`` (the
    boundary cases of the producer's rule table), without ``num_sms`` the pre-rect selection holds, and the rect route
    carries the static route's admission rules."""
    from flashinfer.experimental.deepgemm_dense_mqa.dense_mqa import _catalog

    catalog = _catalog()
    rect = _runtime.paged_rect()
    if not rect["routes"]:
        pytest.skip("catalog ships no rectangle-scheduled program")
    dynamic = _runtime.paged_dynamic_scheduler()
    assert (
        rect["max_waves"] == 1
        and rect["split_kv"] == int(_runtime.paged_policy()["split_kv"]) == 256
    )
    assert set(rect["routes"]) == set(dynamic["routes"])
    for static, rect_route in rect["routes"].items():
        assert rect_route == f"{static}:rect" and rect_route in catalog["paged_routes"]
        assert [
            stage for stage, _p in catalog["paged_routes"][rect_route]["stages"]
        ] == ["logits"]
        for arch in sorted(catalog["arches"]):
            programs = {
                kind: catalog["paged_routes"][route]["stages"][0][1][arch]
                for kind, route in (
                    ("static", static),
                    ("dyn", dynamic["routes"][static]),
                    ("rect", rect_route),
                )
            }
            assert len(set(programs.values())) == 3, (static, arch, programs)
            assert _runtime.paged_admission_rules(
                arch, rect_route
            ) == _runtime.paged_admission_rules(arch, static)
    name = _runtime.paged_route_name
    n1, n2 = "paged:fp8:h64:p64:n1", "paged:fp8:h64:p64:n2"
    # the producer's boundary table: one wave = batch x ceil(max_context_len / 256) <= num_sms
    assert name(64, 64, 1, batch=4, max_context_len=4096, num_sms=148) == f"{n1}:rect"
    assert name(64, 64, 1, batch=4, max_context_len=4096, num_sms=152) == f"{n1}:rect"
    assert (
        name(64, 64, 1, batch=1, max_context_len=1344, num_sms=148) == f"{n1}:rect"
    )  # ceil: 6 splits
    assert (
        name(64, 64, 1, batch=16, max_context_len=1024, num_sms=148) == f"{n1}:rect"
    )  # 64 units
    assert (
        name(64, 64, 2, batch=16, max_context_len=2304, num_sms=148) == f"{n2}:rect"
    )  # 144 units
    # above one wave (the round-4 compact-walk adoption, catalog v11 ``rect.compact``): the compact-walk route up to two waves at
    # batch <= min(num_sms, 160); batch 1 takes the plain rect route at two waves; above two waves / above the batch bound the
    # pre-rect selection holds (test_paged_rect_compact_routes pins the record itself)
    compact = _runtime.paged_rect()["compact"]
    if compact["routes"]:
        assert (
            name(64, 64, 2, batch=16, max_context_len=2560, num_sms=148)
            == f"{n2}:compact"
        )  # 160 units
        assert (
            name(64, 64, 2, batch=16, max_context_len=2560, num_sms=152)
            == f"{n2}:compact"
        )
        assert (
            name(64, 64, 1, batch=16, max_context_len=4096, num_sms=152)
            == f"{n1}:compact"
        )  # 256 units
        assert (
            name(64, 32, 1, batch=1, max_context_len=37920, num_sms=148)
            == "paged:fp8:h64:p32:n1:rect"
        )  # 149 splits, b1
    else:
        assert (
            name(64, 64, 2, batch=16, max_context_len=2560, num_sms=148) == f"{n2}:dyn"
        )  # 160 units
        assert (
            name(64, 64, 2, batch=16, max_context_len=2560, num_sms=152) == f"{n2}:dyn"
        )
        assert (
            name(64, 64, 1, batch=16, max_context_len=4096, num_sms=152) == f"{n1}:dyn"
        )  # 256 units
        assert (
            name(64, 32, 1, batch=1, max_context_len=37920, num_sms=148)
            == "paged:fp8:h64:p32:n1:dyn"
        )  # 149 splits
    assert name(64, 64, 1, batch=148, max_context_len=256, num_sms=148) == f"{n1}:rect"
    assert (
        name(64, 64, 1, batch=149, max_context_len=256, num_sms=148) == f"{n1}:dyn"
    )  # 149 units, batch > num_sms
    assert (
        name(64, 64, 1, batch=16, max_context_len=8192, num_sms=148) == f"{n1}:dyn"
    )  # 512 units: above two waves
    assert (
        name(64, 32, 1, batch=1, max_context_len=37888, num_sms=148)
        == "paged:fp8:h64:p32:n1:rect"
    )  # 148 splits
    assert (
        name(32, 64, 2, batch=4, max_context_len=1024, num_sms=152)
        == "paged:fp8:h32:p64:n2:rect"
    )
    assert (
        name(64, 64, 4, batch=4, max_context_len=1024, num_sms=152)
        == "paged:fp8:h64:p64:n4"
    )  # multi-atom
    assert (
        _runtime.paged_rect_units(4, 4096) == 64
        and _runtime.paged_rect_units(1, 1330) == 6
    )
    assert _runtime.paged_rect_applicable(n1, 148, 256, 148)
    assert not _runtime.paged_rect_applicable(n1, 149, 256, 148)
    assert not _runtime.paged_rect_applicable("paged:fp8:h64:p64:n4", 1, 256, 148)
    # without num_sms (or without max_context_len) the pre-rect selection holds
    assert name(64, 64, 1, batch=4, max_context_len=4096) == f"{n1}:dyn"
    assert name(64, 64, 1, batch=4, num_sms=148) == f"{n1}:dyn"
    assert name(64, 64, 1, num_sms=148) == n1
    # with an architecture the rect route is served only where its (inherited) admission rules admit the call; elsewhere
    # the pre-rect route of the same call is kept
    for arch in sorted(catalog["arches"]):
        assert (
            name(64, 64, 1, batch=4, max_context_len=4096, arch=arch, num_sms=148)
            == f"{n1}:rect"
        )
        for heads, page, next_n, batch, ctx in (
            (32, 64, 2, 4, 8192),
            (64, 32, 2, 4, 8192),
            (64, 64, 2, 4, 1024),
        ):
            static = name(heads, page, next_n)
            pre_rect = name(
                heads, page, next_n, batch=batch, max_context_len=ctx, arch=arch
            )
            served = name(
                heads,
                page,
                next_n,
                batch=batch,
                max_context_len=ctx,
                arch=arch,
                num_sms=148,
            )
            if _runtime.paged_route_admitted(arch, rect["routes"][static], batch, ctx):
                assert served == rect["routes"][static], (arch, static)
            else:
                assert served == pre_rect, (arch, static)


def test_paged_rect_compact_routes():
    """``policy.paged.rect.compact`` (catalog v11, the round-4 compact-walk adoption) maps every rect-capable static route to its
    ``:compact`` route, catalogued with one logits stage whose program differs from the static / dyn / rect ones on every
    architecture, admitted by the static route's rules; ``paged_route_name(..., num_sms=)`` selects it exactly when the
    producer's shape-only rule does (``num_sms < U <= 2 num_sms``, ``batch <= min(num_sms, 160)``; batch 1 -> the rect route)."""
    from flashinfer.experimental.deepgemm_dense_mqa.dense_mqa import _catalog

    catalog = _catalog()
    rect = _runtime.paged_rect()
    compact = rect["compact"]
    if not compact["routes"]:
        pytest.skip("catalog ships no compact-walk program")
    assert (
        compact["max_waves"] == 2
        and compact["max_batch"] == 160
        and set(compact["routes"]) == set(rect["routes"])
    )
    for static, route in compact["routes"].items():
        assert route == f"{static}:compact" and route in catalog["paged_routes"]
        assert [stage for stage, _p in catalog["paged_routes"][route]["stages"]] == [
            "logits"
        ]
        for arch in sorted(catalog["arches"]):
            programs = {
                kind: catalog["paged_routes"][r]["stages"][0][1][arch]
                for kind, r in (
                    ("static", static),
                    ("rect", rect["routes"][static]),
                    ("compact", route),
                )
            }
            assert len(set(programs.values())) == 3, (static, arch, programs)
            assert _runtime.paged_admission_rules(
                arch, route
            ) == _runtime.paged_admission_rules(arch, static)
    kind = _runtime.paged_rect_compact_kind
    n1 = "paged:fp8:h64:p64:n1"
    assert (
        kind(n1, 16, 4096, 148) == "compact" and kind(n1, 16, 4096, 152) == "compact"
    )  # 256 units
    assert kind(n1, 4, 10624, 148) == "compact"  # S_max 42, 168 units
    assert kind(n1, 4, 4096, 148) is None  # 64 units: one wave (the rect rule's)
    assert kind(n1, 16, 8192, 148) is None  # 512 units: above two waves
    assert (
        kind(n1, 1, 42752, 148) == "rect" and kind(n1, 1, 37888, 148) is None
    )  # 167 / 148 splits
    assert (
        kind(n1, 149, 256, 148) is None and kind(n1, 148, 512, 148) == "compact"
    )  # the batch bound min(num_sms, 160)
    assert kind(n1, 160, 512, 160) == "compact" and kind(n1, 161, 512, 161) is None
    assert (
        kind("paged:fp8:h64:p64:n4", 16, 4096, 148) is None
    )  # multi-atom: no rect / compact program
    name = _runtime.paged_route_name
    assert (
        name(64, 64, 1, batch=16, max_context_len=4096, num_sms=148) == f"{n1}:compact"
    )
    assert (
        name(64, 64, 1, batch=4, max_context_len=10624, num_sms=148) == f"{n1}:compact"
    )
    assert name(64, 64, 1, batch=1, max_context_len=42752, num_sms=148) == f"{n1}:rect"
    assert (
        name(32, 64, 1, batch=16, max_context_len=4096, num_sms=152)
        == "paged:fp8:h32:p64:n1:compact"
    )
    assert (
        name(64, 32, 2, batch=16, max_context_len=4096, num_sms=148)
        == "paged:fp8:h64:p32:n2:compact"
    )
    assert (
        name(64, 64, 1, batch=16, max_context_len=4096) == f"{n1}:dyn"
    )  # without num_sms the pre-rect selection holds
    for arch in sorted(catalog["arches"]):
        for heads, page, next_n, batch, ctx in (
            (64, 64, 1, 16, 4096),
            (32, 64, 1, 16, 4096),
            (64, 32, 2, 16, 4096),
            (32, 64, 2, 16, 4096),
        ):
            static = name(heads, page, next_n)
            pre = name(heads, page, next_n, batch=batch, max_context_len=ctx, arch=arch)
            served = name(
                heads,
                page,
                next_n,
                batch=batch,
                max_context_len=ctx,
                arch=arch,
                num_sms=148,
            )
            if _runtime.paged_route_admitted(
                arch, compact["routes"][static], batch, ctx
            ):
                assert served == compact["routes"][static], (arch, static)
            else:
                assert served == pre, (arch, static)


def test_paged_sched_meta_policy_and_producer_routes():
    """``policy.paged.sched_meta`` and ``policy.paged.dynamic_scheduler.sched_meta_windows`` (the unpublished v11, extended):
    the producer routes ``paged:fp8:meta:v<N>`` are catalogued with one ``schedule_meta`` stage whose program binds the
    schedule table and the producer's launch arguments on every architecture and whose record carries the launch geometry
    (``threads``, ``dynamic_smem_bytes``, ``grid``) instead of the geometry fields; the selection names one of them for
    every batch the paged programs accept, monotone in the batch (a larger batch never picks a smaller-capacity version);
    the windows parse like ``static_windows``, lie inside the static route's admission and keep ``batch <= the SM count``;
    ``consumer_routes`` maps exactly the windowed static routes to their ``:table`` routes (one logits stage, a program of
    their own on the architectures whose windows name them, the static route's admission).  The producer launch and the
    table path themselves are exercised by ``test_paged_schedule_meta_two_paths`` / ``test_paged_schedule_meta_producer_plan``."""
    from flashinfer.experimental.deepgemm_dense_mqa.dense_mqa import _catalog

    catalog = _catalog()
    record = _runtime.paged_sched_meta()
    if record is None:
        pytest.skip("catalog carries no schedule-meta producer record")
    assert record["table_shape"] == ("num_sms_plus_1", 2) and record["dtype"] == "int32"
    args = record["producer_args"]
    assert (
        args["num_next_n_atoms"] == 1
        and args["is_context_lens_2d"] == 1
        and list(args["grid"]) == [1, 1, 1]
    )
    assert int(args["split_kv"]) == int(_runtime.paged_policy()["split_kv"])
    programs = record["producer_programs"]
    assert programs and set(programs) == {v for _b, _s, v in record["selection"]}
    producer_args = {
        "context_lens",
        "schedule_meta",
        "batch_size",
        "next_n",
        "num_next_n_atoms",
        "split_kv",
        "num_sms",
        "is_context_lens_2d",
    }
    for version, route in programs.items():
        assert (
            route == f"paged:fp8:meta:v{version}" and route in catalog["paged_routes"]
        ), route
        rec = catalog["paged_routes"][route]
        assert [stage for stage, _p in rec["stages"]] == ["schedule_meta"], route
        assert rec["sequence"] is None
        assert (
            rec["num_heads"] is None
            and rec["block_q"] is None
            and rec["clean_logits"] is None
        )
        assert rec["kv_alignment"] is None and rec["kv_min"] is None
        assert (
            rec["threads"] % 32 == 0
            and rec["threads"] >= 32
            and rec["dynamic_smem_bytes"] >= 0
            and rec["grid"] == [1, 1, 1]
        )
        by_arch = rec["stages"][0][1]
        assert sorted(by_arch) == sorted(catalog["arches"]), route
        for arch, program in by_arch.items():
            bound = {name for _kind, name in catalog["programs"][program]["arg_plan"]}
            assert producer_args <= bound, (route, arch, sorted(producer_args - bound))
            assert arch in catalog["programs"][program]["arches"]
        # the thread budget covers every SM index the selection hands this version (one thread per table entry)
        for _max_batch, max_sms_plus_1, named in record["selection"]:
            if named == version and max_sms_plus_1 is not None:
                assert rec["threads"] >= max_sms_plus_1, (route, max_sms_plus_1)
    # the selection: a total rule over the accepted batches at both catalogued SM counts, monotone in the batch
    for num_sms in sorted(SM_COUNT.values()):
        picked = [
            int(
                _runtime.paged_sched_meta_producer_route(batch, num_sms).rsplit(
                    ":v", 1
                )[1]
            )
            for batch in range(1, int(_runtime.paged_policy()["max_batch"]) + 1)
        ]
        capacity = {v: mb for mb, _s, v in record["selection"]}
        assert all(
            capacity[a] <= capacity[b] for a, b in zip(picked, picked[1:], strict=False)
        ), num_sms
        assert set(picked) <= set(programs)
    with pytest.raises(ValueError):
        _runtime.paged_sched_meta_producer_route(
            int(_runtime.paged_policy()["max_batch"]) + 1, 148
        )
    # the windows: parsed like the static-preferred windows, inside the static route's admission, batch-capped by the SM count
    windows = _runtime.paged_sched_meta_windows()
    dynamic = _runtime.paged_dynamic_scheduler()
    assert windows == dynamic["sched_meta_windows"]
    for arch, table in windows.items():
        assert arch in catalog["arches"]
        for static, boxes in table.items():
            assert static in catalog["paged_routes"] and boxes, (arch, static)
            assert (
                ":" not in static.split(":n", 1)[1]
            )  # a static route, never a text / rect / dyn route
            for min_batch, max_batch, min_ctx, max_ctx in boxes:
                assert 1 <= min_batch <= max_batch <= SM_COUNT[arch], (
                    arch,
                    static,
                    (min_batch, max_batch),
                )
                assert 2 <= min_ctx <= max_ctx, (arch, static, (min_ctx, max_ctx))
                assert _runtime.paged_sched_meta_admitted(
                    arch, static, min_batch, min_ctx
                )
                assert _runtime.paged_sched_meta_admitted(
                    arch, static, max_batch, max_ctx
                )
                assert not _runtime.paged_sched_meta_admitted(
                    arch, static, max_batch + 1, max_ctx
                )
                assert not _runtime.paged_sched_meta_admitted(
                    arch, static, max_batch, max_ctx + 1
                )
                assert not _runtime.paged_sched_meta_admitted(
                    arch, static, min_batch, min_ctx - 1
                )
                for batch, ctx in ((min_batch, min_ctx), (max_batch, max_ctx)):
                    assert _runtime.paged_route_admitted(arch, static, batch, ctx), (
                        arch,
                        static,
                        batch,
                        ctx,
                    )
    assert not _runtime.paged_sched_meta_admitted(
        "sm_90a", "paged:fp8:h64:p64:n4", 16, 4096
    )
    # the consumer routes: one per windowed static route, the table-consuming program behind the ``:table`` suffix
    consumers = record["consumer_routes"]
    assert set(consumers) == {static for table in windows.values() for static in table}
    for static, route in consumers.items():
        assert route == f"{static}:table" and route in catalog["paged_routes"], route
        assert _runtime.paged_sched_meta_consumer_route(static) == route
        rec = catalog["paged_routes"][route]
        assert [stage for stage, _p in rec["stages"]] == ["logits"] and rec[
            "sequence"
        ] is None
        assert (
            rec["num_heads"] == catalog["paged_routes"][static]["num_heads"]
            and rec["clean_logits"] == "raw"
        )
        assert set(rec["stages"][0][1]) == {
            arch for arch, table in windows.items() if static in table
        }, route
        for arch, program in rec["stages"][0][1].items():
            assert program != catalog["paged_routes"][static]["stages"][0][1][arch], (
                route,
                arch,
            )
            assert arch in catalog["programs"][program]["arches"]
            assert _runtime.paged_admission_rules(
                arch, route
            ) == _runtime.paged_admission_rules(arch, static), (route, arch)
    unwindowed = [
        name
        for name in catalog["paged_routes"]
        if not _is_meta_route(name)
        and ":" not in name.split(":n", 1)[1]
        and name not in consumers
    ]
    assert all(
        _runtime.paged_sched_meta_consumer_route(name) is None for name in unwindowed
    )


def test_paged_rect_route_matches_the_pre_rect_route_bitwise():
    """A b4 ctx-4k call (64 (request, KV split) cells) takes the rect route under the device's CTA budget; the same
    call under a one-CTA budget (``sm_count=1``: 64 cells do not fit one wave) takes the pre-rect route; both plans
    compute the same logits bitwise, and the rect plan's grid is one CTA per cell."""
    heads, page, next_n, batch, avg_ctx = 64, 64, 1, 4, 4096
    _skip_unless_route(heads, page, next_n)
    rect = _runtime.paged_rect()
    static = _runtime.paged_route_name(heads, page, next_n)
    if static not in rect["routes"]:
        pytest.skip("catalog ships no rectangle-scheduled program for h64 p64 n1")
    q, kv_cache, _kv_fp8, _scale, weights, ctx_2d, block_table, max_len = _inputs(
        heads, page, next_n, batch, avg_ctx
    )
    plan = _validation_plan(q, kv_cache, weights, ctx_2d, block_table, max_len)
    assert (
        plan.route_name
        == rect["routes"][static]
        == _runtime.paged_route_name(
            heads,
            page,
            next_n,
            batch=batch,
            max_context_len=max_len,
            arch=plan.arch,
            num_sms=plan.num_sms,
        )
    )
    cells = _runtime.paged_rect_units(batch, int(block_table.shape[1]) * page)
    assert cells <= plan.num_sms and plan.grid_x == cells and plan.rect
    one_cta = _validation_plan(
        q, kv_cache, weights, ctx_2d, block_table, max_len, sm_count=1
    )
    assert one_cta.route_name == _runtime.paged_route_name(
        heads, page, next_n, batch=batch, max_context_len=max_len, arch=plan.arch
    )
    assert (
        one_cta.route_name != plan.route_name
        and one_cta.grid_x == 1
        and not one_cta.rect
    )
    # both programs store exactly DeepGEMM's per-request extent: the same cells written, the same bits on them; cells
    # past a shorter request's extent are never written and stay unspecified in both (torch.empty) buffers
    plan.output.fill_(float("nan"))
    one_cta.output.fill_(float("nan"))
    plan.run()
    one_cta.run()
    torch.cuda.synchronize()
    written = ~torch.isnan(one_cta.logical_output)
    assert torch.equal(~torch.isnan(plan.logical_output), written)
    assert written.any() and torch.equal(
        plan.logical_output[written], one_cta.logical_output[written]
    )
    assert torch.isfinite(plan.logical_output[written]).any()


# (heads, page, next_n, batch, avg context, uniform): the two-path cases of the optional schedule-meta input.  The first four
# lie inside a ``sched_meta_windows`` box of the architecture(s) TABLE_ADMITTED names (``_inputs(uniform=True)`` pins the
# extent to the average context): the n4 window ``[128, sms, 4096, 170304]`` on both architectures (random extents
# 5734..10648 / the uniform floor 4096), the sm_100a n1 window ``[16, 16, 4096, 5248]`` at its ceiling and the sm_103a n1
# window ``[64, 64, 4096, 5184]`` at its floor; the last two are accepted calls no window admits (a static-route call at
# batch 3 and a rect-route call), on which the meta plan is accepted and NOT consumed.
TABLE_CASES = [
    (64, 64, 4, 128, 8192, False),
    (64, 64, 4, 128, 4096, True),
    (64, 64, 1, 16, 5248, True),
    (64, 64, 1, 64, 4096, True),
    (64, 64, 4, 3, 2048, False),
    (64, 64, 1, 1, 1024, False),
]
TABLE_ADMITTED = {"sm_100a": {0, 1, 2}, "sm_103a": {0, 1, 3}}


@pytest.mark.parametrize("index,case", list(enumerate(TABLE_CASES)))
def test_paged_schedule_meta_two_paths(index, case):
    """Both paths of the optional schedule-meta input on one call.  (i) The default path: ``schedule_meta=None`` is one launch
    of the default program (``schedule_meta_used`` False); (ii) the DeepGEMM placeholder tensor is the same path, program for
    program and bit for bit (no silent switch inside a window); (iii) ``prepare_paged_mqa_schedule_meta`` is one launch of
    the producer route the selection names and writes the pure-Python reference table bit for bit; (iv) a plan given the
    meta plan serves the ``:table`` route exactly where the catalog admits the call on the plain static route -- a program of
    its own, one launch, the table bound through ``sched_counters`` -- and the default plan elsewhere, with bit-identical
    logits either way; (v) the one-shot entry given the meta plan reproduces those bits; (vi) a CUDA graph of producer +
    logits replayed on changed contents rewrites the table and reproduces an eager default-path plan bit for bit."""
    heads, page, next_n, batch, avg_ctx, uniform = case
    _skip_unless_route(heads, page, next_n)
    if _runtime.paged_sched_meta() is None:
        pytest.skip("catalog carries no schedule-meta producer record")
    q, kv_cache, kv_fp8, scale, weights, ctx_2d, block_table, max_len = _inputs(
        heads, page, next_n, batch, avg_ctx, uniform=uniform
    )
    admitted = _admitted(q, ctx_2d, max_len, heads, page, next_n)
    arch, num_sms = _runtime._resolve_device(q, None)
    static = _runtime.paged_route_name(heads, page, next_n)
    split_kv = int(_runtime.paged_policy()["split_kv"])

    def logits_plan(**kwargs):
        if admitted:
            return prepare_paged_mqa_logits(
                q, kv_cache, weights, ctx_2d, block_table, max_len, **kwargs
            )
        return _validation_plan(
            q, kv_cache, weights, ctx_2d, block_table, max_len, **kwargs
        )

    # (i) the default path
    plan = logits_plan()
    assert (
        plan.launch_count == 1
        and plan.schedule_meta_used is False
        and plan.schedule_meta is None
    )
    assert plan.program_names == [plan.route["stages"][0][1][arch]]
    plan.output.fill_(float("nan"))
    plan.run()
    torch.cuda.synchronize()
    ref = _reference(q, kv_fp8, scale, weights, ctx_2d, block_table, max_len, page)
    _check(plan.logical_output, ref, ctx_2d, max_len, plan.output)
    written = ~torch.isnan(plan.logical_output)
    default_bits = plan.logical_output.clone()
    # (ii) the placeholder tensor: the default path, program for program and bit for bit
    placeholder = get_paged_mqa_logits_metadata(ctx_2d, page, num_sms)
    assert not placeholder.any()
    plan_p = logits_plan(schedule_meta=placeholder)
    assert (
        plan_p.route_name == plan.route_name
        and plan_p.program_names == plan.program_names
    )
    assert plan_p.schedule_meta_used is False and plan_p.schedule_meta is placeholder
    assert plan_p.sched_counters.numel() == plan.sched_counters.numel()
    plan_p.output.fill_(float("nan"))
    plan_p.run()
    torch.cuda.synchronize()
    assert torch.equal(plan_p.logical_output[written], default_bits[written])
    # (iii) the producer plan
    meta = prepare_paged_mqa_schedule_meta(ctx_2d, num_sms=num_sms)
    assert (
        meta.launch_count == 1
        and meta.num_sms == num_sms
        and (meta.batch, meta.next_n) == (batch, next_n)
    )
    assert (
        meta.route_name == _runtime.paged_sched_meta_producer_route(batch, num_sms)
        and meta.arch == arch
    )
    assert (
        _runtime.paged_sched_meta()["producer_programs"][meta.version]
        == meta.route_name
    )
    assert meta.program_name == meta_program(meta.route_name, arch)
    assert meta.table.dtype == torch.int32 and tuple(meta.table.shape) == (
        num_sms + 1,
        2,
    )
    assert meta.table.device == q.device and meta.table.is_contiguous()
    meta.table.zero_()  # a wrong-but-valid table (the reference is non-zero past entry 0): a producer that did not run shows
    assert meta.run() is meta.table
    torch.cuda.synchronize()
    assert meta.table.cpu().tolist() == _schedule_meta_reference(
        ctx_2d.cpu(), num_sms, split_kv
    )
    # (iv) the logits plan given the producer plan
    plan_t = logits_plan(schedule_meta=meta)
    expected_used = plan.route_name == static and _runtime.paged_sched_meta_admitted(
        arch, static, batch, max_len
    )
    assert expected_used == (index in TABLE_ADMITTED[arch]), (
        arch,
        index,
        plan.route_name,
        max_len,
    )
    assert plan_t.schedule_meta_used is expected_used and plan_t.schedule_meta is meta
    assert plan_t.launch_count == 1 and plan_t.num_sms == num_sms
    if expected_used:
        consumer = _runtime.paged_sched_meta_consumer_route(static)
        assert plan_t.route_name == consumer == f"{static}:table"
        assert plan_t.program_names == [plan_t.route["stages"][0][1][arch]]
        assert plan_t.program_names != plan.program_names
        assert plan_t.sched_counters.data_ptr() == meta.table.data_ptr()
        assert (
            plan_t.sched_counters.numel() == 2 * (num_sms + 1)
            and plan_t.sched_counters.dtype == torch.uint32
        )
        assert not plan_t.rect and plan_t.grid_x == num_sms
    else:
        assert (
            plan_t.route_name == plan.route_name
            and plan_t.program_names == plan.program_names
        )
        assert plan_t.sched_counters.numel() == plan.sched_counters.numel()
        assert plan_t.sched_counters.data_ptr() != meta.table.data_ptr()
    plan_t.output.fill_(float("nan"))
    plan_t.run()
    torch.cuda.synchronize()
    _check(plan_t.logical_output, ref, ctx_2d, max_len, plan_t.output)
    assert torch.equal(~torch.isnan(plan_t.logical_output), written)
    assert torch.equal(plan_t.logical_output[written], default_bits[written])
    # (v) the one-shot entry given the producer plan: the same bits (its CTA budget is the meta plan's)
    if admitted:
        shot = fp8_paged_mqa_logits(
            q, kv_cache, weights, ctx_2d, block_table, meta, max_len
        )
    else:
        shot = _validation_plan(
            q, kv_cache, weights, ctx_2d, block_table, max_len, schedule_meta=meta
        ).run()
    torch.cuda.synchronize()
    assert torch.equal(shot[written], default_bits[written])
    # (vi) graph replay of producer + logits with changed contents
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            meta.run()
            plan_t.run()
    with torch.cuda.stream(stream):
        weights.neg_()
        ctx_2d.copy_((ctx_2d - 3).clamp_min(1))
        plan_t.output.fill_(float("nan"))
        meta.table.zero_()  # every CTA an empty walk until the replayed producer rewrites the table
        graph.replay()
    stream.synchronize()
    assert meta.table.cpu().tolist() == _schedule_meta_reference(
        ctx_2d.cpu(), num_sms, split_kv
    )
    ref = _reference(q, kv_fp8, scale, weights, ctx_2d, block_table, max_len, page)
    _check(plan_t.logical_output, ref, ctx_2d, max_len, plan_t.output)
    eager = _validation_plan(
        q, kv_cache, weights, ctx_2d, block_table, max_len, sm_count=num_sms
    )
    assert eager.schedule_meta_used is False and eager.route_name == plan.route_name
    eager.output.fill_(float("nan"))
    eager.run()
    torch.cuda.synchronize()
    written = ~torch.isnan(eager.logical_output)
    assert torch.equal(~torch.isnan(plan_t.logical_output), written)
    assert torch.equal(plan_t.logical_output[written], eager.logical_output[written])


def meta_program(route_name, arch):
    """The producer program a ``paged:fp8:meta:v<N>`` route launches on ``arch``."""
    from flashinfer.experimental.deepgemm_dense_mqa.dense_mqa import _catalog

    return _catalog()["paged_routes"][route_name]["stages"][0][1][arch]


def test_paged_schedule_meta_producer_plan():
    """The producer side of the table path.  Host half: ``schedule_meta_bindings`` covers every producer program's launcher
    contract; ``consumer_routes`` maps exactly the windowed static routes to ``:table`` routes catalogued with one logits
    stage, a program of their own on the architectures whose windows name them and the static route's admission; a static
    route without windows has no consumer.  Device half: ``out=`` reuse, the default CTA budget, the ``ValueError`` /
    ``TypeError`` contract of the producer plan (1-D / CPU / non-int32 / non-contiguous lengths, a batch above the ceiling, an
    ``out`` of the wrong dtype / shape / layout / device) and of a logits plan given one (another CTA budget, another
    ``context_lens`` tensor, a foreign ``schedule_meta`` type), and a meta plan fixing the CTA budget of the plans that consume
    it."""
    from flashinfer.experimental.deepgemm_dense_mqa.dense_mqa import _catalog

    catalog = _catalog()
    record = _runtime.paged_sched_meta()
    if record is None:
        pytest.skip("catalog carries no schedule-meta producer record")
    lens = torch.full((3, 2), 300, dtype=torch.int32)
    table = torch.zeros((5, 2), dtype=torch.int32)
    bindings = _runtime.schedule_meta_bindings(lens, table, num_sms=4)
    assert (bindings["batch_size"], bindings["next_n"], bindings["num_sms"]) == (
        3,
        2,
        4,
    )
    assert (bindings["grid_x"], bindings["grid_y"], bindings["grid_z"]) == tuple(
        record["producer_args"]["grid"]
    )
    assert (
        bindings["split_kv"] == int(_runtime.paged_policy()["split_kv"])
        and bindings["num_next_n_atoms"] == 1
    )
    assert bindings["is_context_lens_2d"] == 1
    assert (
        bindings["schedule_meta"].data_ptr() == table.data_ptr()
        and bindings["schedule_meta"].ndim == 1
    )
    assert bindings["context_lens"].ndim == 1 and bindings["context_lens"].numel() == 6
    for route in record["producer_programs"].values():
        for arch, program in catalog["paged_routes"][route]["stages"][0][1].items():
            missing = [
                name
                for _kind, name in catalog["programs"][program]["arg_plan"]
                if name not in bindings
            ]
            assert not missing, (route, arch, missing)
    windows = _runtime.paged_sched_meta_windows()
    consumers = record["consumer_routes"]
    assert set(consumers) == {static for table in windows.values() for static in table}
    for static, route in consumers.items():
        assert (
            route == f"{static}:table"
            and _runtime.paged_sched_meta_consumer_route(static) == route
        )
        rec = catalog["paged_routes"][route]
        assert [stage for stage, _p in rec["stages"]] == ["logits"] and rec[
            "sequence"
        ] is None
        assert set(rec["stages"][0][1]) == {
            arch for arch, table in windows.items() if static in table
        }, route
        for arch, program in rec["stages"][0][1].items():
            assert program != catalog["paged_routes"][static]["stages"][0][1][arch], (
                route,
                arch,
            )
            assert _runtime.paged_admission_rules(
                arch, route
            ) == _runtime.paged_admission_rules(arch, static)
    unwindowed = [
        name
        for name in catalog["paged_routes"]
        if not _is_meta_route(name)
        and ":" not in name.split(":n", 1)[1]
        and name not in consumers
    ]
    assert all(
        _runtime.paged_sched_meta_consumer_route(name) is None for name in unwindowed
    )
    assert _runtime.paged_sched_meta_consumer_route("paged:fp8:h64:p64:n2:dyn") is None
    if _catalogued_device_arch() is None:
        return
    _skip_unless_route(64, 64, 1)
    q, kv_cache, _kv, _scale, weights, ctx_2d, block_table, max_len = _inputs(
        64, 64, 1, 8, 2048
    )
    arch, num_sms = _runtime._resolve_device(q, None)
    device = q.device
    split_kv = int(_runtime.paged_policy()["split_kv"])
    out = torch.full((num_sms + 1, 2), -1, dtype=torch.int32, device=device)
    meta = prepare_paged_mqa_schedule_meta(ctx_2d, num_sms=num_sms, out=out)
    assert meta.table is out and meta.run() is out
    torch.cuda.synchronize()
    assert out.cpu().tolist() == _schedule_meta_reference(
        ctx_2d.cpu(), num_sms, split_kv
    )
    assert (
        meta.arch == arch
        and meta.version in record["producer_programs"]
        and meta.launch_count == 1
    )
    assert (
        prepare_paged_mqa_schedule_meta(ctx_2d).num_sms == num_sms
    )  # the device's SM count by default
    with pytest.raises(ValueError):
        prepare_paged_mqa_schedule_meta(
            ctx_2d.reshape(-1), num_sms=num_sms
        )  # 1-D lengths
    with pytest.raises(ValueError):
        prepare_paged_mqa_schedule_meta(
            ctx_2d.cpu(), num_sms=num_sms
        )  # CPU lengths (an operand error, not a device error)
    with pytest.raises(ValueError):
        prepare_paged_mqa_schedule_meta(
            ctx_2d.to(torch.int64), num_sms=num_sms
        )  # non-int32 lengths
    strided = torch.zeros((ctx_2d.shape[0], 2), dtype=torch.int32, device=device)[
        :, : ctx_2d.shape[1]
    ]
    assert not strided.is_contiguous()
    with pytest.raises(ValueError):
        prepare_paged_mqa_schedule_meta(
            strided, num_sms=num_sms
        )  # non-contiguous lengths
    with pytest.raises(ValueError):
        prepare_paged_mqa_schedule_meta(
            ctx_2d,
            num_sms=num_sms,
            out=torch.zeros((num_sms + 1, 2), dtype=torch.int32),
        )  # out on another device
    with pytest.raises(ValueError):
        prepare_paged_mqa_schedule_meta(
            ctx_2d,
            num_sms=num_sms,
            out=torch.zeros((num_sms + 1, 2), dtype=torch.int64, device=device),
        )
    with pytest.raises(ValueError):
        prepare_paged_mqa_schedule_meta(
            ctx_2d,
            num_sms=num_sms,
            out=torch.zeros((num_sms, 2), dtype=torch.int32, device=device),
        )
    with pytest.raises(ValueError):
        prepare_paged_mqa_schedule_meta(
            ctx_2d,
            num_sms=num_sms,
            out=torch.zeros((2, num_sms + 1), dtype=torch.int32, device=device).t(),
        )
    too_many = torch.ones(
        (int(_runtime.paged_policy()["max_batch"]) + 1, 1),
        dtype=torch.int32,
        device=device,
    )
    with pytest.raises(ValueError):
        prepare_paged_mqa_schedule_meta(too_many, num_sms=num_sms)
    # a meta plan is bound to its CTA budget and to its context_lens tensor; any other schedule_meta type is refused
    with pytest.raises(ValueError):
        _validation_plan(
            q,
            kv_cache,
            weights,
            ctx_2d,
            block_table,
            max_len,
            schedule_meta=meta,
            sm_count=num_sms - 1,
        )
    foreign = prepare_paged_mqa_schedule_meta(ctx_2d.clone(), num_sms=num_sms)
    with pytest.raises(ValueError):
        _validation_plan(
            q, kv_cache, weights, ctx_2d, block_table, max_len, schedule_meta=foreign
        )
    with pytest.raises(TypeError):
        _validation_plan(
            q, kv_cache, weights, ctx_2d, block_table, max_len, schedule_meta=object()
        )
    with pytest.raises(TypeError):
        fp8_paged_mqa_logits(
            q, kv_cache, weights, ctx_2d, block_table, object(), max_len
        )
    # a meta plan fixes the CTA budget of the plans that consume it (sm_count defaults to the meta plan's)
    smaller = prepare_paged_mqa_schedule_meta(ctx_2d, num_sms=num_sms - 1)
    smaller.run()
    plan = _validation_plan(
        q, kv_cache, weights, ctx_2d, block_table, max_len, schedule_meta=smaller
    )
    assert plan.num_sms == num_sms - 1 and plan.launch_count == 1
    torch.cuda.synchronize()
