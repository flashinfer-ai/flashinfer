"""Paged prepared API (DeepGEMM paged signatures): logits against the
dequantized PyTorch paged reference (and DeepGEMM when importable), the
clean_logits=False mask, graph replay with changed contents and chunk
equivalence on SM100a/SM103a. Every paged call is ONE kernel launch (the
schedule is derived in-kernel); ``get_paged_mqa_logits_metadata`` is a
no-launch DeepGEMM-signature placeholder.

The tests skip while the installed catalog carries no paged route for the
shape (``paged_route_available``); the host-only helpers are tested always.
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
]


def _catalog_paged_routes():
    from flashinfer.experimental.deepgemm_dense_mqa.dense_mqa import _catalog

    return list(_catalog()["paged_routes"].values())


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


def _admitted(q, ctx_2d, max_len, heads, page, next_n):
    """The shipped per-architecture verdict for this exact call (batch = ``ctx_2d.shape[0]``)."""
    arch = _runtime._resolve_device(q, None)[0]
    return _runtime.paged_route_admitted(
        arch,
        _runtime.paged_route_name(heads, page, next_n, batch=int(ctx_2d.shape[0])),
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
    assert plan.route_name == _runtime.paged_route_name(
        heads, page, next_n, batch=batch
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
    assert plan.launch_count == 1 and plan.program_names == [plan.route["stages"][0][1]]
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
        assert [stage for stage, _program in route["stages"]] == ["logits"], route_name
        for stage, program in route["stages"]:
            arg_plan = catalog["programs"][program]["arg_plan"]
            missing = [name for _kind, name in arg_plan if name not in bindings[stage]]
            assert not missing, (
                f"{route_name} stage {stage} ({program}): unbound arguments {missing}"
            )
            assert "schedule_meta" not in {name for _kind, name in arg_plan}, (
                f"{program} binds a schedule buffer"
            )
            checked += 1
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
        static_program = catalog["paged_routes"][static]["stages"][0][1]
        dyn_program = catalog["paged_routes"][dyn]["stages"][0][1]
        # Program ids are content hashes of the generated text: the two routes serve distinct programs, and only
        # the dynamic one binds the scheduler state.
        assert dyn_program != static_program
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
    for route_name in catalog["paged_routes"]:
        if route_name.endswith(":dyn") or route_name in dynamic["routes"]:
            continue
        fields = {field[0]: int(field[1:]) for field in route_name.split(":")[2:5]}
        assert (
            _runtime.paged_route_name(fields["h"], fields["p"], fields["n"], batch=1)
            == route_name
        )


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
    if not torch.cuda.is_available():
        return
    # Live refusal under the shipped rules of this device's architecture (and the export bypass). The plan
    # resolves the route from the call's batch (the dynamic-scheduler route up to its max_batch, the static
    # route above it), so every probe is judged on the route the plan would select.
    arch = dense_mqa.device_arch(torch.device("cuda"))
    probes = ((17, 1024), (17, 32769), (65, 8448), (129, 8448), (4096, 131072))
    route_for = lambda b: _runtime.paged_route_name(heads, page, 2, batch=b)
    rules = {
        route_for(b): _runtime.paged_admission_rules(arch, route_for(b))
        for b, _c in probes
    }
    if all(r is None for r in rules.values()):
        return
    withheld = next(
        (
            (b, c)
            for b, c in probes
            if not _runtime.paged_route_admitted(arch, route_for(b), b, c)
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
