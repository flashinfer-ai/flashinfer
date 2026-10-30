# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The EXTEND / PREFILL forms of the Qwen 3.8 Flash-Next block (the in-block indexer's EXTEND step over the pools; the cuDNN Frontend
4c8e24355 re-vendor) against the TWO-STAGE oracle: a uniform ``[batch, T]`` chunk of every sequence behind its cached prefix -- the
prefix primed into the pools, the compressed-key cache (through the vendored compress kernel) and the ring (the last ``cap`` positions
below the chunk: the form's ring INPUT contract) by a torch reference prefill; then (1) the chunk rows' selection vs the fp64 top-k of
the kernel's own operands up to ties, the counts rule, the canonical list, (2) ``out`` within the block suites' bf16 OUTPUT bound of the
fp32 oracle fed THAT list; EVERY complete block of the stream bitwise the vendored compress from scratch after the chunk (the prefix
untouched, the chunk's blocks -- the straddling one included -- written), the ring = the sequence's last ``cap`` raw keys, the pools'
chunk rows vs the reference.  The ladder on the 24 / 2 geometry: ``T 128`` at batch 1 behind ``P 0`` (the ``prefill`` form), 6144, 6145
and 6146 (the chunk start at every residue of the block grid); batch 2 x ``T 64`` with a dead entry on page 32 / cap 8; batch 4 x ``T 20``;
one- and two-row chunks that complete a block or carry the tail alone; the e4m3 compressed-key cache; the ``"hf"`` rounding convention.
Then: a chunked prefill ``[125, 2, 1, 45, 128]`` of a 301-token prompt with the caches STARTING EMPTY (``prefill`` for the first chunk,
``extend`` for the rest; every chunk's own writes prime the next) followed by three carried decode steps with a rejected draft re-stepped
-- after every chunk and step the oracle holds and the cache's complete blocks are bitwise the from-scratch compress of the accepted
stream; the typed declines at plan time (the five vendored ``indexer_extend_*`` rows by text, the 128-row bound, the chunk bound, a page
off the powers of two, the e4m3 pool, the body vocabulary) and at run time; CUDA-graph capture after ONE eager warm-up on the capture
stream under ``set_sync_debug_mode("error")``, replayed bitwise and over a new chunk; the launch census by kernel NAME (the chunk
compress, the gather, the canonical kernel, two cache writes; + 2 per further select window at a lowered budget); the workspace query
honest for the extend buckets and three runs allocating nothing; ``"prefill"`` == ``"extend"`` (equal declarations but the form, bitwise
runs on the same inputs)."""

import pytest
import torch

from tests.test_helpers.qwen_3_8_block_cases import (
    Qsa38ExtendCase,
    Qsa38PagedCase,
    bits,
)

pytestmark = [pytest.mark.arch_rubin, pytest.mark.usefixtures("rubin")]

GEOM = dict(
    d_model=2560, num_qo_heads=24, num_kv_heads=2
)  # Qwen3.8-Flash-Next at TP 1 (the served geometry)


def _geometry(**over):
    from flashinfer.qwen_3_8_block import Qwen38BlockGeometry

    return Qwen38BlockGeometry(**dict(GEOM, **over))


def _plan(case, g, form="extend", **kw):
    from flashinfer.qwen_3_8_block import prepare_qwen_3_8_block_bf16

    args = dict(
        geometry=g,
        device="cuda",
        weights=case.weights38,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        page_size=case.page_size,
        batch_size=case.b,
        q_len_per_req=case.q_len,
        block_table=case.block_table,
        kv_layout=case.layout,
        **case.plan_args(),
    )
    args.update(kw)
    return prepare_qwen_3_8_block_bf16(form, **args)


def _run(plan, case, **kw):
    """One chunk through ``plan`` on the case's tensors (the per-token tables, the stack's ring slots); the selection into the case's
    sentinel buffers.  Returns ``out``."""
    case.reset_outputs()
    out = plan.run(
        case.hidden_states,
        case.rope_cos,
        case.rope_sin,
        case.weights38,
        case.w_o,
        **case.extend_args(),
        block_ids_out=case.block_ids_out,
        block_lens_out=case.block_lens_out,
        **kw,
    )
    torch.cuda.synchronize()
    return out


def _qi(plan):
    """The indexer queries the chunk left in the plan's workspace (the scorer's own operand): read through the vendored extend stage."""
    blk = plan._block
    return blk._indexer_extend.views(plan._workspace, blk._ws)["qi"]


def _raw_keys(plan, case):
    """The chunk's raw indexer keys as the block's own slab GEMM left them in the workspace (the fifth band's key head)."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gated_attention_block import (
        index_k_raw_view,
    )
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gated_attention_block.api import (
        _view,
    )

    blk = plan._block
    g = blk.geom
    t = case.b * case.q_len
    proj = _view(plan._workspace, blk._ws.proj, (t, g.n_qkvg), case.dtype)
    return index_k_raw_view(proj, g, case.b, case.q_len)[:, :, 0].reshape(t, -1)


def _check_chunk(case, plan, out, label):
    """The two-stage oracle + the whole-stream state checks on one chunk."""
    stats = case.check_selection(
        case.block_ids_out,
        case.block_lens_out,
        _qi(plan),
        n_blocks_max=plan.declaration["n_blocks_max"],
        label=label,
    )
    case.check_out_from_list(out, case.block_ids_out, label=label)
    completed = case.check_state(_raw_keys(plan, case), label=label)
    assert (case.kv_cache[0] == 0).all(), (
        "the NULL page (block 0) must never be written"
    )
    return stats, completed


def _check_declaration(plan, case, form):
    d = plan.declaration
    assert plan.form == form and d["form"] == form
    assert (
        plan.index_select_form == "extend"
        and d["index_select_form"] == d["index_select_form_resolved"] == "extend"
    )
    assert (
        plan.ring_slots is True
        and plan.rope_position_tables_only is False
        and plan.fuse_prepare is False
    )
    assert (
        d["ring_slots"] is True
        and d["rope_position_tables_only"] is False
        and d["fuse_prepare"] is False
    )
    assert (
        plan.ring_cap == case.cap
        and plan.ring_cap_exact is True
        and plan.capture_warm_up_required is True
    )
    assert d["ring_cap_exact"] is True and d["capture_warm_up_required"] is True
    assert (
        "_IndexerExtend" in d["stages"]
        and "_CacheWrite" in d["stages"]
        and "_SparseSdpa" in d["stages"]
    ), d["stages"]
    assert "_IndexerDecode" not in d["stages"] and "_QsaPrepare" not in d["stages"], d[
        "stages"
    ]
    assert (
        d["n_windows"] >= 1
        and d["window_rows"] >= 1
        and d["select_budget_bytes"] == 512 << 20
    )
    assert d["indexer_launches"] == 4 + 2 * d["n_windows"]
    assert d["index_compress_rounding"] == plan.index_compress_rounding == case.rounding
    assert d["index_cache_dtype"] == plan.index_cache_dtype == case.cache_dtype
    assert d["n_blocks_max"] == case.block_table.shape[1] * (case.page_size // 4)
    assert d["proj_capture_safe"] is True and d["out_proj_capture_safe"] is True, (
        "under the 128-row bound both projections ride the JIT-only route (the vendored block's own reading)"
    )
    assert d["capture_safe_rows"] is True


def _two_runs_bitwise(plan, case, label):
    """Two runs with the side caches restored in between are bitwise on ``out``, the list, the cache and the ring; the oracle after the first."""
    case.snapshot()
    out = _run(plan, case).clone()
    ids, lens = case.block_ids_out.clone(), case.block_lens_out.clone()
    cache, ring = (
        bits(case.index_kv_cache).clone(),
        bits(case.index_state_cache).clone(),
    )
    stats, completed = _check_chunk(case, plan, out, label)
    case.restore()
    again = _run(plan, case)
    assert (
        torch.equal(out, again)
        and torch.equal(ids, case.block_ids_out)
        and torch.equal(lens, case.block_lens_out)
    ), f"{label}: two runs are not bitwise"
    assert torch.equal(cache, bits(case.index_kv_cache)) and torch.equal(
        ring, bits(case.index_state_cache)
    ), f"{label}: the side caches differ between two runs"
    return stats, completed


# (id, geometry overrides, kv_lens AFTER the chunk, T, page, cap, form)
LADDER = [
    ("T128-P0-prefill-p16-cap4", {}, (128,), 128, 16, 4, "prefill"),
    ("T128-P6144-res0-p16-cap4", {}, (6272,), 128, 16, 4, "extend"),
    ("T128-P6145-res1-p16-cap4", {}, (6273,), 128, 16, 4, "extend"),
    ("T128-P6146-res2-p16-cap4", {}, (6274,), 128, 16, 4, "extend"),
    ("B2-T64-P6146-dead-p32-cap8", {}, (6210, 0), 64, 32, 8, "extend"),
    ("B4-T20-residues-p16-cap4", {}, (321, 2072, 0, 6166), 20, 16, 4, "extend"),
    ("T2-P8194-res2-one-block-completes", {}, (8196,), 2, 16, 4, "extend"),
    ("T1-P8194-no-block-completes-tail-alone", {}, (8195,), 1, 16, 4, "extend"),
    (
        "T128-P6146-e4m3-cache",
        dict(index_cache_dtype=torch.float8_e4m3fn),
        (6274,),
        128,
        16,
        4,
        "extend",
    ),
    (
        "T100-P8195-res3-hf-rounding",
        dict(index_compress_rounding="hf"),
        (8295,),
        100,
        16,
        4,
        "extend",
    ),
]


@pytest.mark.parametrize(
    "ident, over, kv_lens, t, page, cap, form", LADDER, ids=[c[0] for c in LADDER]
)
def test_extend_matches_the_two_stage_oracle(ident, over, kv_lens, t, page, cap, form):
    g = _geometry(**over)
    case = Qsa38ExtendCase(g, kv_lens, t, page_size=page, seed=101 + t, cap=cap)
    case.prime()
    plan = _plan(case, g, form)
    _check_declaration(plan, case, form)
    stats, completed = _two_runs_bitwise(plan, case, ident)
    # the blocks the chunk completes: every 4-token block whose last position lies in the chunk
    want = sum(max(0, L // 4 - max(L - t, 0) // 4) for L in kv_lens if L)
    assert completed == want, (completed, want)
    if over.get("index_cache_dtype") is not None:
        assert case.index_kv_cache.dtype == _qi(plan).dtype == torch.float8_e4m3fn
    assert plan.supports(
        hidden_states=case.hidden_states,
        rope_cos=case.rope_cos,
        rope_sin=case.rope_sin,
        **case.extend_args(),
    )
    assert not plan.supports(hidden_states=case.hidden_states, **case.extend_args()), (
        "the per-token tables are required"
    )


def test_prefill_is_the_extend_form_at_an_empty_prefix():
    """``"prefill"`` and ``"extend"`` at one bucket: equal declarations but the form; a prefill plan's run with ``seq_lens[b] == T`` for every
    live sequence is BITWISE an extend plan's run on the same inputs (one vendored form, two names)."""
    g = _geometry()
    a = Qsa38ExtendCase(g, (32, 32, 0), 32, page_size=16, seed=111, cap=4)
    b = Qsa38ExtendCase(g, (32, 32, 0), 32, page_size=16, seed=111, cap=4)
    pa, pb = _plan(a, g, "prefill"), _plan(b, g, "extend")
    da, db = dict(pa.declaration), dict(pb.declaration)
    assert da.pop("form") == "prefill" and db.pop("form") == "extend" and da == db
    assert pa.workspace_size == pb.workspace_size
    out_a = _run(pa, a)
    out_b = _run(pb, b)
    assert torch.equal(out_a, out_b) and torch.equal(a.block_ids_out, b.block_ids_out)
    assert torch.equal(bits(a.index_kv_cache), bits(b.index_kv_cache)) and torch.equal(
        bits(a.index_state_cache), bits(b.index_state_cache)
    )
    _check_chunk(a, pa, out_a, "prefill == extend")


def test_chunked_prefill_then_carried_decode_steps_are_one_chain():
    """A 301-token prompt as FIVE CHUNKS ``[125, 2, 1, 45, 128]`` (chunk starts at residues 0 / 1 / 3 / 0 / 1 of the block grid; two chunks
    shorter than ``cap``) through ``prepare("prefill")`` for the first and ``prepare("extend")`` for the rest, every chunk planned at its own
    ``T`` -- the caches START EMPTY and every chunk's own writes prime the next --, then THREE decode steps over the same pools / ring / cache:
    ``q_len`` 1, 1, then 4 with two of four rows accepted and RE-STEPPED over the same positions.  After every chunk and step the two-stage
    oracle holds, the ring holds the stream's last ``cap`` raw keys and the cache's rows for every complete block of the ACCEPTED stream are
    bitwise the from-scratch compress of that stream (the chunks' rows, the rows a step completes and a re-stepped block are one chain)."""
    from tests.test_helpers.qwen_3_5_block_reference import (
        gated_attention_block_reference,
    )

    g = _geometry()
    chunks = [125, 2, 1, 45, 128]
    assert sum(chunks) == 301
    cap, page, width = (
        8,
        16,
        24,
    )  # cap 8: the decode step at q_len 4 needs cap >= 7; 24 pages = 384 positions hold the 310-token stream
    first = Qsa38ExtendCase(
        g,
        (chunks[0],),
        chunks[0],
        page_size=page,
        seed=131,
        table_width=width,
        cap=cap,
        extra_pages=-(-(301 + 1 + 1 + 4) // page)
        - (-(-chunks[0] // page))
        + 2,  # the pools hold the whole stream: 20 pages + 2
    )
    assert (
        not first._primed
        and torch.isnan(first.index_kv_cache.float()).all()
        and torch.isnan(first.index_state_cache.float()).all()
    )
    first._primed = True  # P = 0: nothing to prime -- the caches start empty
    # the stream's pages: the first case maps its 125 tokens; the later chunks and steps take the spare pages (the table has room: width 24)
    pages = list(first.pages[0])
    spare = sorted(set(range(1, first.num_pages)) - set(pages))
    pages_t = lambda: torch.tensor(pages, device="cuda")  # noqa: E731
    weights = (first.weights38, first.w_o)
    cos_all, sin_all = (
        first.position_tables
    )  # [1, n_pos, rope]: the stream's per-position tables
    gen = torch.Generator(device="cuda").manual_seed(79)
    h_all = first.h_full[
        0
    ]  # [1, L, d_model]: the stream's inputs so far (the first chunk's draw), extended chunk by chunk
    L = 0
    k_raw_acc = None  # the stream's raw keys as the BLOCK's slabs emitted them
    case = first
    for ci, T in enumerate(chunks):
        pos0, Lf = L, L + T
        while len(pages) * page < Lf:
            pages.append(spare.pop(0))
        if ci == 0:
            nxt = first
        else:
            nxt = Qsa38ExtendCase(
                g,
                (Lf,),
                T,
                page_size=page,
                seed=131,
                table_width=width,
                weights=weights,
                cap=cap,
            )
            # re-aim the case at THIS chunk of the carried stream: the stream's inputs drawn once (extended by this chunk's rows) and the
            # dense reference over them, the carried caches, pages and table
            h_chunk = torch.randn(
                1, T, g.d_model, generator=gen, device="cuda", dtype=torch.float32
            ).to(torch.bfloat16)
            h_all = torch.cat([h_all, h_chunk], 1)
            nxt.refs[0] = gated_attention_block_reference(
                h_all,
                case.weights.w_qkvg,
                case.weights.w_q_norm,
                case.weights.w_k_norm,
                cos_all[:, :Lf],
                sin_all[:, :Lf],
                case.w_o,
                case.ref_geom,
            )
            nxt.h_full[0] = h_all
            nxt.hidden_states = h_chunk.view(T, g.d_model).contiguous()
            nxt.rope_cos = cos_all[0, pos0:Lf].contiguous()
            nxt.rope_sin = sin_all[0, pos0:Lf].contiguous()
            nxt.kv_cache, nxt.index_kv_cache, nxt.index_state_cache = (
                case.kv_cache,
                case.index_kv_cache,
                case.index_state_cache,
            )
            nxt.index_state_block_table = case.index_state_block_table
            nxt.k_raw_full[0] = torch.cat(
                [
                    k_raw_acc,
                    (h_chunk[0].float() @ case.w_band_k.float().t()).to(torch.bfloat16),
                ],
                0,
            )  # the prefix as the blocks wrote it (the chunk's own rows come from its slab)
            nxt.prefix_bits = (bits(case.kv_cache).clone(),)
            nxt._primed = True
        nxt.pages = [list(pages)]
        nxt.block_table.zero_()
        nxt.block_table[0, : len(pages)] = pages_t().to(torch.int32)
        pos = torch.arange(pos0, Lf, device="cuda")
        nxt.slot_mapping = (pages_t()[pos // page] * page + pos % page).to(torch.int32)
        nxt.ring_slots = torch.where(
            pos + cap >= Lf, 1 * cap + pos % cap, torch.full_like(pos, -1)
        ).to(torch.int32)
        nxt.seq_lens = torch.tensor([Lf], dtype=torch.int32, device="cuda")
        plan = _plan(nxt, g, "prefill" if ci == 0 else "extend")
        out = _run(plan, nxt)
        label = f"chunk {ci} T={T} pos {pos0}..{Lf - 1}"
        _check_chunk(nxt, plan, out, label)
        raw = _raw_keys(plan, nxt)
        k_raw_acc = raw.clone() if k_raw_acc is None else torch.cat([k_raw_acc, raw], 0)
        case, L = nxt, Lf
    assert L == 301
    # the three carried decode steps (the decode module's carried cell, over THIS stream): q 1 (position 301), q 1 (302: completes block 75),
    # q 4 with two of four accepted and re-stepped
    h_acc = h_all.clone()
    plans = {}
    for step, (s_q, n_acc) in enumerate([(1, 1), (1, 1), (4, 2), (4, 4)], 1):
        pos0 = L
        h_step = torch.randn(
            1, s_q, g.d_model, generator=gen, device="cuda", dtype=torch.float32
        ).to(torch.bfloat16)
        h_full = torch.cat([h_acc, h_step], 1)
        Lf = pos0 + s_q
        while len(pages) * page < Lf:
            pages.append(spare.pop(0))
        ref = gated_attention_block_reference(
            h_full,
            case.weights.w_qkvg,
            case.weights.w_q_norm,
            case.weights.w_k_norm,
            case.position_tables[0][:, :Lf],
            case.position_tables[1][:, :Lf],
            case.w_o,
            case.ref_geom,
        )
        dec = Qsa38PagedCase(
            g,
            (Lf,),
            s_q,
            page_size=page,
            seed=131,
            table_width=width,
            weights=weights,
            cap=cap,
        )
        dec.refs[0] = ref
        dec.h_full[0] = h_full
        dec.hidden_states = h_step.view(s_q, g.d_model).contiguous()
        dec.kv_cache, dec.index_kv_cache, dec.index_state_cache = (
            case.kv_cache,
            case.index_kv_cache,
            case.index_state_cache,
        )
        dec.index_state_block_table = case.index_state_block_table
        dec.pages = [list(pages)]
        dec.block_table.zero_()
        dec.block_table[0, : len(pages)] = pages_t().to(torch.int32)
        pos = torch.arange(pos0, Lf, device="cuda")
        dec.slot_mapping = (pages_t()[pos // page] * page + pos % page).to(torch.int32)
        dec.seq_lens = torch.tensor([Lf], dtype=torch.int32, device="cuda")
        dec.k_raw_full[0] = torch.cat(
            [
                k_raw_acc,
                (h_step[0].float() @ case.w_band_k.float().t()).to(torch.bfloat16),
            ],
            0,
        )
        dec.prefix_bits = (bits(case.kv_cache).clone(),)
        dec._primed = True
        from flashinfer.qwen_3_8_block import prepare_qwen_3_8_block_bf16

        if s_q not in plans:
            plans[s_q] = prepare_qwen_3_8_block_bf16(
                "decode",
                geometry=g,
                device="cuda",
                weights=dec.weights38,
                w_o=dec.w_o,
                kv_cache=dec.kv_cache,
                page_size=page,
                batch_size=1,
                q_len_per_req=s_q,
                block_table=dec.block_table,
                **dec.plan_args(),
            )
        dec.reset_outputs()
        out = plans[s_q].run(
            dec.hidden_states,
            None,
            None,
            dec.weights38,
            dec.w_o,
            **dec.decode_args(),
            block_ids_out=dec.block_ids_out,
            block_lens_out=dec.block_lens_out,
        )
        torch.cuda.synchronize()
        label = f"carried decode step {step} (q{s_q}, accept {n_acc})"
        blk = plans[s_q]._block
        qi = blk._indexer_decode.views(plans[s_q]._workspace, blk._ws)["qi"]
        dec.check_selection(
            dec.block_ids_out,
            dec.block_lens_out,
            qi,
            n_blocks_max=plans[s_q].declaration["n_blocks_max"],
            label=label,
        )
        dec.check_out_from_list(out, dec.block_ids_out, label=label)
        raw_step = _raw_keys(plans[s_q], dec)
        dec.check_state(raw_step, label=label)
        # accept n_acc rows: the cache's complete blocks of the ACCEPTED stream vs the from-scratch compress of its raw keys
        L = pos0 + n_acc
        h_acc = torch.cat([h_acc, h_step[:, :n_acc]], 1)
        k_raw_acc = torch.cat([k_raw_acc, raw_step[:n_acc]], 0)
        nb = L // 4
        kb = case.compress(k_raw_acc[None, : nb * 4], *case.position_tables)[0]
        j = torch.arange(nb, device="cuda")
        rows = case.index_kv_cache[pages_t()[j // 4], j % 4, 0]
        assert torch.equal(bits(rows), bits(kb)), (
            f"{label}: the cache's {nb} complete blocks are not the from-scratch compress of the accepted stream"
        )
        case = dec
    assert L == 301 + 1 + 1 + 2 + 4


def test_plan_time_declines_are_typed_and_named(monkeypatch):
    """The extend / prefill forms' typed declines at plan time: the light gates BEFORE any vendored code (the 128-row bound, a page off the
    powers of two, the e4m3 pool, the body vocabulary, the chunk bound), then the vendored rows raised with the envelope's own text
    (``indexer_extend_ring_slots`` / ``indexer_extend_rope_tables_only`` / ``indexer_extend_fuse_prepare`` on an explicit flag; the ring
    floor ``cap >= 3`` -- a cap-2 ring --; ``use_pdl=True`` under the indexer)."""
    from flashinfer.experimental.qwen_3_8_block import api as impl
    from flashinfer.qwen_3_8_block import qwen_3_8_block_support

    g = _geometry()
    sup = qwen_3_8_block_support(torch.device("cuda", 0), g)
    env = sup.extend.declines
    case = Qsa38ExtendCase(g, (300,), 16, page_size=16, seed=141, cap=4)
    for form in ("extend", "prefill"):
        with pytest.raises(
            NotImplementedError,
            match=r"indexer_extend_ring_slots|takes the stack's ring slots",
        ) as info:
            _plan(case, g, form, ring_slots=False)
        assert str(info.value) == env["indexer_extend_ring_slots"], str(info.value)
        with pytest.raises(NotImplementedError) as info:
            _plan(case, g, form, rope_position_tables_only=True)
        assert str(info.value) == env["indexer_extend_rope_tables_only"], str(
            info.value
        )
        with pytest.raises(NotImplementedError) as info:
            _plan(case, g, form, fuse_prepare=True)
        assert str(info.value) == env["indexer_extend_fuse_prepare"], str(info.value)
        with pytest.raises(
            ValueError, match="at least block_size - 1 = 3 ring rows"
        ) as info:
            _plan(case, g, form, index_state_cache=case.index_state_cache[:, :2])
        with pytest.raises(NotImplementedError, match="use_pdl=True"):
            _plan(case, g, form, use_pdl=True)
        # q_len_per_req > max_pages x page_size: the chunk bound (the vendored indexer_extend_chunk_rows class), before any vendored code
        short = Qsa38ExtendCase(g, (16,), 16, page_size=16, seed=142, cap=4)
        assert int(short.block_table.shape[1]) == 1
        with pytest.raises(ValueError, match="exceed the block table's reach"):
            _plan(
                short,
                g,
                form,
                q_len_per_req=17,
                index_state_cache=short.index_state_cache,
            )
    # the light gates never reach the vendored tree
    monkeypatch.setattr(
        impl,
        "_frost_block",
        lambda: (_ for _ in ()).throw(
            AssertionError("the vendored tree was reached by a light-gate decline")
        ),
    )
    for form in ("extend", "prefill"):
        with pytest.raises(NotImplementedError, match="129 rows per run exceed 128"):
            _plan(case, g, form, q_len_per_req=129)
        with pytest.raises(NotImplementedError, match="130 rows per run exceed 128"):
            _plan(case, g, form, batch_size=2, q_len_per_req=65)
        with pytest.raises(NotImplementedError, match="page_size must be one of"):
            _plan(case, g, form, page_size=48)
        with pytest.raises(NotImplementedError, match="e4m3 pool"):
            _plan(case, g, form, kv_cache_dtype=torch.float8_e4m3fn)
        with pytest.raises(ValueError, match="sparse_tile_form must be one of"):
            _plan(case, g, form, sparse_tile_form="fast")
        with pytest.raises(TypeError, match="ring_slots must be None"):
            _plan(case, g, form, ring_slots=1)


def test_run_time_declines_are_typed_and_named():
    g = _geometry()
    case = Qsa38ExtendCase(g, (300, 77), 16, page_size=16, seed=151, cap=4)
    case.prime()
    plan = _plan(case, g)
    a = case.extend_args()
    w = (case.hidden_states, case.rope_cos, case.rope_sin, case.weights38, case.w_o)
    with pytest.raises(ValueError, match="run needs index_state_slot_mapping"):
        plan.run(*w, **case.decode_args(ring_slots=False))
    with pytest.raises(ValueError, match="rope_cos / rope_sin"):
        plan.run(case.hidden_states, None, None, case.weights38, case.w_o, **a)
    with pytest.raises(ValueError, match="was prepared on a ring of cap=4"):
        plan.run(
            *w,
            **dict(
                a,
                index_state_cache=torch.zeros(
                    case.b + 1, 8, 1, 128, dtype=torch.bfloat16, device="cuda"
                ),
            ),
        )
    with pytest.raises(ValueError, match="at least block_size - 1 = 3 ring rows"):
        plan.run(*w, **dict(a, index_state_cache=a["index_state_cache"][:, :2]))
    with pytest.raises(ValueError, match="seq_lens must be torch.int32"):
        plan.run(*w, **dict(a, seq_lens=a["seq_lens"].to(torch.int64)))
    with pytest.raises(ValueError, match="index_kv_cache must be"):
        plan.run(*w, **dict(a, index_kv_cache=a["index_kv_cache"][:-1]))
    with pytest.raises(ValueError, match="query_start_loc / num_actual_tokens"):
        plan._bind(
            hidden_states=case.hidden_states,
            query_start_loc=torch.zeros(3, dtype=torch.int32, device="cuda"),
        )
    assert not plan.supports(
        hidden_states=case.hidden_states[:1],
        rope_cos=case.rope_cos,
        rope_sin=case.rope_sin,
        **a,
    )


def test_capture_after_one_warm_up_replays_bitwise_and_takes_a_new_chunk():
    """CUDA-graph capture of the extend form AFTER one eager warm-up ON THE CAPTURE STREAM (the first run uploads the select window plan's
    constants): the capture under ``set_sync_debug_mode("error")`` launches nothing; the replay is BITWISE the eager run (``out``, the list,
    the cache, the ring); a NEW chunk's tensors (other lengths, another seed) written in place into the captured buffers replay bitwise an
    eager run of that chunk and pass the oracle; ``prepare`` under capture is refused typed."""
    g = _geometry()
    lens_a, lens_b = (321, 0, 2075), (322, 2074, 0)
    pages_a, pages_b = (
        sum(-(-L // 16) for L in lens_a),
        sum(-(-L // 16) for L in lens_b),
    )
    case = Qsa38ExtendCase(
        g,
        lens_a,
        24,
        page_size=16,
        seed=161,
        cap=4,
        table_width=140,
        extra_pages=max(pages_b - pages_a, 0) + 2,
    )
    case.prime()
    plan = _plan(case, g)
    t = case.b * case.q_len
    out = torch.full((t, g.d_model), float("nan"), dtype=torch.bfloat16, device="cuda")
    ids = torch.full((t, g.top_k), -7, dtype=torch.int32, device="cuda")
    h, cos, sin = (
        case.hidden_states.clone(),
        case.rope_cos.clone(),
        case.rope_sin.clone(),
    )
    slots, ring_slots = (
        case.slot_mapping.clone(),
        case.ring_slots.to(case.slot_mapping.dtype).clone(),
    )
    table, lens, ring_table = (
        case.block_table.clone(),
        case.seq_lens.clone(),
        case.index_state_block_table.clone(),
    )
    args = dict(
        kv_cache=case.kv_cache,
        slot_mapping=slots,
        block_table=table,
        seq_lens=lens,
        index_kv_cache=case.index_kv_cache,
        index_state_cache=case.index_state_cache,
        index_state_block_table=ring_table,
        index_state_slot_mapping=ring_slots,
        out=out,
        block_ids_out=ids,
    )
    case.snapshot()
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):  # the eager warm-up ON the capture stream
        plan.run(h, cos, sin, case.weights38, case.w_o, **args)
    torch.cuda.synchronize()
    eager_out, eager_ids = out.clone(), ids.clone()
    eager_cache, eager_ring = (
        bits(case.index_kv_cache).clone(),
        bits(case.index_state_cache).clone(),
    )
    case.restore()
    out.fill_(float("nan"))
    ids.fill_(-7)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                plan.run(h, cos, sin, case.weights38, case.w_o, **args)
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(out).all() and (ids == -7).all(), "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager_out) and torch.equal(ids, eager_ids), (
            "the replay differs from the eager run"
        )
        assert torch.equal(bits(case.index_kv_cache), eager_cache) and torch.equal(
            bits(case.index_state_cache), eager_ring
        )
        case.block_ids_out.copy_(ids)
        case.check_out_from_list(out, ids, label="replay")
        with (
            pytest.raises(RuntimeError, match="capture"),
            torch.cuda.graph(torch.cuda.CUDAGraph(), stream=torch.cuda.Stream()),
        ):
            _plan(case, g)
        # a NEW chunk through the captured tensors: another seed, other lengths over the same shapes
        sib = Qsa38ExtendCase(
            g,
            lens_b,
            24,
            page_size=16,
            seed=163,
            cap=4,
            table_width=140,
            weights=(case.weights38, case.w_o),
            extra_pages=2,
        )
        sib.prime()
        assert (
            sib.kv_cache.shape == case.kv_cache.shape
            and sib.index_kv_cache.shape == case.index_kv_cache.shape
        )
        case.kv_cache.copy_(sib.kv_cache)
        case.index_kv_cache.copy_(sib.index_kv_cache)
        case.index_state_cache.copy_(sib.index_state_cache)
        table.copy_(sib.block_table)
        lens.copy_(sib.seq_lens)
        ring_table.copy_(sib.index_state_block_table)
        slots.copy_(sib.slot_mapping)
        ring_slots.copy_(sib.ring_slots.to(ring_slots.dtype))
        h.copy_(sib.hidden_states)
        cos.copy_(sib.rope_cos)
        sin.copy_(sib.rope_sin)
        sib.snapshot()
        out.fill_(float("nan"))
        ids.fill_(-7)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        rep_out, rep_ids = out.clone(), ids.clone()
        rep_cache, rep_ring = (
            bits(case.index_kv_cache).clone(),
            bits(case.index_state_cache).clone(),
        )
        sib.kv_cache, sib.index_kv_cache, sib.index_state_cache = (
            case.kv_cache,
            case.index_kv_cache,
            case.index_state_cache,
        )
        sib.block_ids_out.copy_(rep_ids)
        sib.check_out_from_list(rep_out, rep_ids, label="replay over a new chunk")
        assert (rep_out[sib.q_len * 2 :] == 0).all(), (
            "the dead entry must replay as exact zeros"
        )
        sib.restore()
        out.fill_(float("nan"))
        plan.run(h, cos, sin, case.weights38, case.w_o, **args)
        torch.cuda.synchronize()
        assert torch.equal(out, rep_out) and torch.equal(ids, rep_ids), (
            "the replay over a new chunk differs from the eager run over it"
        )
        assert torch.equal(bits(case.index_kv_cache), rep_cache) and torch.equal(
            bits(case.index_state_cache), rep_ring
        )
    finally:
        graph.reset()


def _kernel_names(plan, case):
    case.reset_outputs()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        plan.run(
            case.hidden_states,
            case.rope_cos,
            case.rope_sin,
            case.weights38,
            case.w_o,
            **case.extend_args(),
            block_ids_out=case.block_ids_out,
            block_lens_out=case.block_lens_out,
        )
        torch.cuda.synchronize()
    return [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]


def test_launch_census_by_kernel_name_matches_the_declaration():
    """The CUDA kernels one chunk launches (the profiler's names) at batch 4 x T 20 (80 rows): the pool-1 query compress and the CHUNK compress
    (``frost_qsa_compress`` x 2, one of them ``frost_qsa_compress_step``), the dense kbar gather, the canonical-order kernel, two cache writes
    (the K / V write and the ring arm), no paged scorer / decode top-k / fused prepare, no memset / memcpy; the count equals the declaration's
    ``launches_per_execute``; a lowered select-window budget (two windows) adds exactly the scorer + top-k pair and computes the same bytes."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gated_attention_block import (
        api as vend,
    )

    g = _geometry()
    case = Qsa38ExtendCase(g, (4116, 320, 0, 2071), 20, page_size=16, seed=171, cap=4)
    case.prime()
    plan = _plan(case, g)
    case.snapshot()
    _run(plan, case)  # the warm-up: the constants' upload
    case.restore()
    names = _kernel_names(plan, case)
    if not names:
        pytest.skip(
            "torch.profiler recorded no CUDA event (CUPTI unavailable on this node)"
        )
    has = lambda ns, tok: sum(tok in n for n in ns)  # noqa: E731
    memops = [n for n in names if "memset" in n.lower() or "memcpy" in n.lower()]
    assert not memops, f"memset / memcpy on the execute path: {memops}"
    assert len(names) == plan.declaration["launches_per_execute"], (
        len(names),
        plan.declaration["launches_per_execute"],
        names,
    )
    assert (
        plan.declaration["n_windows"] == 1 and plan.declaration["indexer_launches"] == 6
    )
    assert (
        has(names, "frost_qsa_compress_step") == 1
        and has(names, "frost_qsa_compress") == 2
    ), names
    assert (
        has(names, "qsa_kbar_gather") == 1
        and has(names, "qsa_canonical_list") == 1
        and has(names, "frost_cache_write") == 2
    ), names
    assert (
        has(names, "qsa_score_paged") == 0
        and has(names, "qsa_topk_decode") == 0
        and has(names, "frost_qsa_prepare") == 0
    ), names
    assert not any("nvjet" in n.lower() or "cublas" in n.lower() for n in names), (
        "the slab left the FROST route"
    )
    # a lowered budget: 40 rows per select window over 80 rows -> two windows (+ 2 launches: the second scorer + top-k pair), the same bytes;
    # every run starts from the restored caches (one run leaves the chunk's last cap raw keys in the ring: the next chunk's input, not this one's)
    case.restore()
    out_one = _run(plan, case).clone()
    ids_one = case.block_ids_out.clone()
    case.restore()
    n_blocks_max = plan.declaration["n_blocks_max"]
    saved = vend.QSA_EXTEND_SELECT_BUDGET_BYTES
    vend.QSA_EXTEND_SELECT_BUDGET_BYTES = 40 * n_blocks_max * 4
    try:
        two = _plan(case, g)
    finally:
        vend.QSA_EXTEND_SELECT_BUDGET_BYTES = saved
    assert (
        two.declaration["n_windows"] == 2
        and two.declaration["window_rows"] == 40
        and two.declaration["indexer_launches"] == 8
    )
    assert (
        two.declaration["launches_per_execute"]
        == plan.declaration["launches_per_execute"] + 2
    )
    _run(two, case)
    case.restore()
    out_two = _run(two, case)
    assert torch.equal(out_two, out_one) and torch.equal(case.block_ids_out, ids_one), (
        "the window count changed the function"
    )
    case.restore()
    names_two = _kernel_names(two, case)
    assert (
        len(names_two) == len(names) + 2 == two.declaration["launches_per_execute"]
    ), (len(names_two), len(names))


def test_workspace_query_is_honest_and_runs_allocate_nothing():
    """``qwen_3_8_block_workspace_size("extend" / "prefill", ...)`` equals the compiled plan's carve (bf16 and e4m3 caches; the ring cap does
    not enter the carve: cap 4 and cap 8 equal); the wrapper over ONE shared workspace serves a decode bucket and an extend bucket in turn;
    after the warm-up three runs into a caller-provided ``out`` make no CUDA allocation under ``set_sync_debug_mode("error")`` (``run`` allocates
    the output only when the caller passes none)."""
    from flashinfer.qwen_3_8_block import (
        Qwen38BlockWrapper,
        qwen_3_8_block_workspace_size,
    )

    sizes = {}
    for cache_dtype in (None, torch.float8_e4m3fn):
        g = _geometry(index_cache_dtype=cache_dtype)
        case = Qsa38ExtendCase(g, (4224, 300, 0), 32, page_size=16, seed=181, cap=4)
        case.prime()
        plan = _plan(case, g)
        for form in ("extend", "prefill"):
            for cap in (4, 8):
                sized = qwen_3_8_block_workspace_size(
                    form,
                    geometry=g,
                    device="cuda",
                    page_size=16,
                    batch_size=case.b,
                    q_len_per_req=32,
                    max_pages=int(case.block_table.shape[1]),
                    ring_cap=cap,
                )
                assert sized == plan.workspace_size, (
                    form,
                    cap,
                    cache_dtype,
                    sized,
                    plan.workspace_size,
                )
        with pytest.raises(ValueError, match="below the extend step's floor"):
            qwen_3_8_block_workspace_size(
                "extend",
                geometry=g,
                device="cuda",
                page_size=16,
                batch_size=1,
                q_len_per_req=16,
                max_pages=4,
                ring_cap=2,
            )
        sizes[cache_dtype] = plan.workspace_size
        case.snapshot()
        out_buf = torch.empty(case.t, g.d_model, dtype=torch.bfloat16, device="cuda")
        _run(plan, case, out=out_buf)  # the warm-up
        torch.cuda.synchronize()
        stat = lambda: torch.cuda.memory_stats()["allocation.all.allocated"]  # noqa: E731
        n0 = stat()
        prev = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(3):
                case.restore()
                _run(plan, case, out=out_buf)
        finally:
            torch.cuda.set_sync_debug_mode(prev)
        assert stat() - n0 == 0, (
            f"the extend plan made {stat() - n0} CUDA allocation(s) over three runs (cache {cache_dtype})"
        )
        _check_chunk(case, plan, _run(plan, case), f"workspace ({cache_dtype})")
    # the serving idiom: one workspace at the MAX over a decode bucket and an extend bucket
    g = _geometry()
    dec = Qsa38PagedCase(g, (300, 2300, 0), 4, page_size=16, seed=182)
    dec.prime()
    ext = Qsa38ExtendCase(
        g, (2300 + 32, 300 + 32, 0), 32, page_size=16, seed=183, cap=4
    )
    ext.prime()
    need = max(
        qwen_3_8_block_workspace_size(
            "decode",
            geometry=g,
            device="cuda",
            page_size=16,
            batch_size=3,
            q_len_per_req=4,
            max_pages=int(dec.block_table.shape[1]),
            ring_cap=dec.cap,
        ),
        qwen_3_8_block_workspace_size(
            "extend",
            geometry=g,
            device="cuda",
            page_size=16,
            batch_size=3,
            q_len_per_req=32,
            max_pages=int(ext.block_table.shape[1]),
        ),
    )
    ws = torch.empty(need, dtype=torch.uint8, device="cuda")
    w = Qwen38BlockWrapper(ws, "NHD", geometry=g)
    w.plan(
        "decode",
        weights=dec.weights38,
        w_o=dec.w_o,
        kv_cache=dec.kv_cache,
        page_size=16,
        batch_size=3,
        q_len_per_req=4,
        block_table=dec.block_table,
        **dec.plan_args(),
    )
    assert w.plan_info.form == "decode" and w.plan_info._workspace is None
    dec.reset_outputs()
    out = w.run(
        dec.hidden_states,
        None,
        None,
        dec.weights38,
        dec.w_o,
        block_ids_out=dec.block_ids_out,
        block_lens_out=dec.block_lens_out,
        **dec.decode_args(),
    )
    torch.cuda.synchronize()
    dec.check_out_from_list(out, dec.block_ids_out, label="wrapper decode")
    w.plan(
        "extend",
        weights=ext.weights38,
        w_o=ext.w_o,
        kv_cache=ext.kv_cache,
        page_size=16,
        batch_size=3,
        q_len_per_req=32,
        block_table=ext.block_table,
        **ext.plan_args(),
    )
    assert (
        w.plan_info.form == "extend"
        and w.plan_info.index_select_form == "extend"
        and w.plan_info.workspace_size <= ws.numel()
    )
    ext.reset_outputs()
    out = w.run(
        ext.hidden_states,
        ext.rope_cos,
        ext.rope_sin,
        ext.weights38,
        ext.w_o,
        block_ids_out=ext.block_ids_out,
        block_lens_out=ext.block_lens_out,
        **ext.extend_args(),
    )
    torch.cuda.synchronize()
    ext.check_selection(
        ext.block_ids_out,
        ext.block_lens_out,
        w.plan_info._block._indexer_extend.views(ws, w.plan_info._block._ws)["qi"],
        n_blocks_max=w.plan_info.declaration["n_blocks_max"],
        label="wrapper extend",
    )
    ext.check_out_from_list(out, ext.block_ids_out, label="wrapper extend")
