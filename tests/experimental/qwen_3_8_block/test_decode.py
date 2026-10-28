# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The DECODE form of the Qwen 3.8 Flash-Next block against the TWO-STAGE oracle: a torch reference prefill primes the pools, the ring and
the paged compressed-key cache (through the vendored compress kernel), then one decode step at q_len 1 and 4 on the 24 / 2 shard (and
the TP 2 / 4 shards) -- (1) the kernel's selection vs the fp64 top-k of its own operands up to ties, the counts rule, the canonical
list; (2) ``out`` within the block suite's budget of the fp32 oracle fed THAT list; the written pool rows / ring rows / completed cache
rows vs the reference; padded entries exactly 0, the null page / ring block untouched; page 16 / 32 / 64 / 128; the UNIFIED cache spelling
vs the NHD one bitwise; the ring-slots arm vs the derived one bitwise; the fused prepare vs the unfused twin bitwise; the position-table
arm vs the per-token arm bitwise; the declaration readbacks; the launch census by kernel count and by kernel NAME (the lean decode
tile at batch 1 under the AUTO body knob, the tcgen05 core at batch 4 and under the explicit pin); the two bodies of the sparse core
within the budget of each other at batch 1; three carried steps with a rejected draft re-stepped; the run-time typed declines; a
cached step under ``set_sync_debug_mode("error")``; the compiled-plan cache's reload."""

import pytest
import torch

from tests.test_helpers.qwen_3_8_block_cases import (
    Qsa38PagedCase,
    bits,
    unified_kv_cache,
)

pytestmark = [pytest.mark.arch_rubin, pytest.mark.usefixtures("rubin")]

GEOMETRIES = {
    "24_2": dict(
        d_model=2560, num_qo_heads=24, num_kv_heads=2
    ),  # Qwen3.8-Flash-Next at TP 1
    "12_1": dict(
        d_model=2560, num_qo_heads=12, num_kv_heads=1
    ),  # TP 2: declined by name (the slab N 7296 is off the capture-safe projection route)
    "6_1": dict(
        d_model=2560, num_qo_heads=6, num_kv_heads=1
    ),  # TP 4: declined by name (N 4224)
}
SERVED = ("24_2",)
SHARDS = ("12_1", "6_1")
# one KV tile with a tail, a long sequence past the identity bound (2051 visible tokens: the list no longer holds every block), a
# PADDED entry (block 0, slot -1, seq_len 0, ring block 0), a sequence whose step completes a block
LENS = (100, 2300, 0, 303)


def _geometry(name):
    from flashinfer.qwen_3_8_block import Qwen38BlockGeometry

    return Qwen38BlockGeometry(**GEOMETRIES[name])


def _plan(case, g, **kw):
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
    return prepare_qwen_3_8_block_bf16("decode", **args)


def _run(plan, case, *, ring_slots=False, per_token=False, **kw):
    """One step through ``plan`` on the case's tensors; the selection into the case's sentinel buffers.  Returns ``out``."""
    case.reset_outputs()
    cos, sin = (case.rope_cos, case.rope_sin) if per_token else (None, None)
    out = plan.run(
        case.hidden_states,
        cos,
        sin,
        case.weights38,
        case.w_o,
        **case.decode_args(ring_slots=ring_slots),
        block_ids_out=case.block_ids_out,
        block_lens_out=case.block_lens_out,
        **kw,
    )
    torch.cuda.synchronize()
    return out


def _qi(plan):
    """The indexer queries the step left in the plan's workspace (the kernel's own scorer operand): read through the vendored stage."""
    blk = plan._block
    return blk._indexer_decode.views(plan._workspace, blk._ws)["qi"]


def _raw_keys(plan, case):
    """The step's raw indexer keys as the block's own slab GEMM left them in the workspace (the fifth band's key head)."""
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


def _check_step(case, plan, out, label):
    """The two-stage oracle + the state checks on one step."""
    n_max = plan.declaration["n_blocks_max"]
    stats = case.check_selection(
        case.block_ids_out,
        case.block_lens_out,
        _qi(plan),
        n_blocks_max=n_max,
        label=label,
    )
    case.check_out_from_list(out, case.block_ids_out, label=label)
    completed = case.check_state(_raw_keys(plan, case), label=label)
    assert (case.kv_cache[0] == 0).all(), (
        "the NULL page (block 0) must never be written"
    )
    return stats, completed


@pytest.mark.parametrize("name", SERVED)
@pytest.mark.parametrize("q_len", [1, 4])
def test_decode_matches_the_two_stage_oracle(name, q_len):
    g = _geometry(name)
    case = Qsa38PagedCase(g, LENS, q_len, page_size=16, seed=1)
    case.prime()
    plan = _plan(case, g)
    d = plan.declaration
    assert (
        d["index_list_form"] == "per_row"
        and d["fuse_prepare"] is True
        and d["pdl_stages"] == []
        and d["use_pdl"] is False
    ), d
    assert (
        d["fuse_norm_rope_effective"] is False
        and d["gate_fused"] is True
        and d["ring_slots"] is False
        and d["rope_position_tables_only"] is True
    )
    assert (
        "_QsaPrepare" in d["stages"]
        and "_IndexerDecode" in d["stages"]
        and "_SparseSdpa" in d["stages"]
    ), d["stages"]
    assert d["indexer_launches"] == 2 and d["n_blocks_max"] == case.block_table.shape[
        1
    ] * (16 // 4)
    # the sparse core's body under the AUTO knob at batch 4 (8 CTAs per chunk at two KV heads, past the lean bound of 4): the tcgen05 core
    assert (
        d["sparse_tile_form_requested"] == "auto"
        and plan.sparse_tile_form == d["sparse_tile_form"] == "tcgen05"
        and plan.sparse_tile == d["sparse_tile"] == "sparse_d256_f16"
        and d["sparse_sub_tiles_per_cta"] is None
    ), d
    out = _run(plan, case)
    assert out.shape == (case.b * q_len, g.d_model) and out.dtype == torch.bfloat16
    stats, completed = _check_step(case, plan, out, f"decode {name} q{q_len}")
    # the blocks the step completes: one per live row at a position p with (p + 1) % 4 == 0
    want = sum(
        1 for L in LENS if L for j in range(q_len) if (L - q_len + j + 1) % 4 == 0
    )
    assert completed == want, (completed, want)
    assert plan.supports(hidden_states=case.hidden_states, **case.decode_args())
    assert not plan.supports(hidden_states=case.hidden_states[:1], **case.decode_args())


@pytest.mark.parametrize("name", SHARDS)
def test_tp_shard_slabs_are_declined_by_name(name, monkeypatch):
    """The TP 2 / TP 4 shard geometries: the probe declines them under ``proj_slab_route`` (their slabs are off the capture-safe
    small-M projection route on this part -- the vendored route's own arithmetic at the device's SM count), and ``prepare`` raises the
    same typed ``NotImplementedError`` before any vendored declaration."""
    from flashinfer.experimental.qwen_3_8_block import api as impl
    from flashinfer.experimental.qwen_3_8_block import support
    from flashinfer.qwen_3_8_block import qwen_3_8_block_support

    g = _geometry(name)
    sup = qwen_3_8_block_support(torch.device("cuda", 0), g)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    assert (
        not sup.served
        and sup.reason == sup.decode.declines[support.PROJ_SLAB_ROUTE_KEY]
    )
    assert f"N={g.n_qkvg}" in sup.reason and f"{sm}-SM" in sup.reason
    assert qwen_3_8_block_support(torch.device("cuda", 0), _geometry("24_2")).served

    def reached():
        raise AssertionError(
            "the vendored declaration was reached for a shard the route gate declines"
        )

    monkeypatch.setattr(impl, "_frost_block", reached)
    case = Qsa38PagedCase(g, (64,), 1, page_size=16, seed=31)
    with pytest.raises(NotImplementedError, match="small-M projection route"):
        _plan(case, g)


@pytest.mark.parametrize("page", [32, 64, 128])
def test_decode_page_sizes(page):
    g = _geometry("24_2")
    case = Qsa38PagedCase(g, (300, 1100, 0), 2, page_size=page, seed=2)
    case.prime()
    plan = _plan(case, g)
    assert plan.page_size == page and plan.declaration[
        "n_blocks_max"
    ] == case.block_table.shape[1] * (page // 4)
    out = _run(plan, case)
    _check_step(case, plan, out, f"decode p{page}")


def test_unified_cache_spelling_is_bitwise_the_nhd_one():
    """A serving stack's UNIFIED row [num_pages, page_size, H_kv, 2 x D] bound by strides (K / V halves) against the NHD 5-D cache holding the
    same values: the same bytes in ``out``, the list, the pools."""
    g = _geometry("24_2")
    a = Qsa38PagedCase(g, (300, 77, 0, 2052), 2, page_size=16, seed=3)
    a.prime()
    plan_a = _plan(a, g)
    out_a = _run(plan_a, a)
    _check_step(a, plan_a, out_a, "decode NHD")
    b = Qsa38PagedCase(g, (300, 77, 0, 2052), 2, page_size=16, seed=3)
    b.prime()
    unified = unified_kv_cache(b.kv_cache)
    plan_b = _plan(b, g, kv_cache=unified, kv_layout="UNIFIED")
    assert (
        plan_b.kv_layout == "UNIFIED" and plan_b.declaration["kv_layout"] == "UNIFIED"
    )
    b.reset_outputs()
    out_b = plan_b.run(
        b.hidden_states,
        None,
        None,
        b.weights38,
        b.w_o,
        block_ids_out=b.block_ids_out,
        block_lens_out=b.block_lens_out,
        **dict(b.decode_args(), kv_cache=unified),
    )
    torch.cuda.synchronize()
    assert torch.equal(out_a, out_b) and torch.equal(
        a.block_ids_out, b.block_ids_out
    ), "the two cache spellings compute different bytes"
    k_u, v_u = unified[..., :256], unified[..., 256:]
    assert torch.equal(bits(k_u), bits(a.kv_cache[:, 0])) and torch.equal(
        bits(v_u), bits(a.kv_cache[:, 1])
    ), "the unified pools' rows differ"
    assert torch.equal(bits(b.index_kv_cache), bits(a.index_kv_cache)) and torch.equal(
        bits(b.index_state_cache), bits(a.index_state_cache)
    )


def test_ring_slots_arm_is_bitwise_the_derived_one():
    """The stack's own ring WRITE rows (``index_state_slot_mapping = block x cap + pos % cap``) against the rows derived in the kernel from
    the ring table and ``seq_lens``: bitwise on ``out``, the list, the ring and the cache."""
    g = _geometry("24_2")
    outs, states = [], []
    for ring_slots in (False, True):
        case = Qsa38PagedCase(g, (303, 77, 0, 1001), 4, page_size=16, seed=4)
        case.prime()
        plan = _plan(case, g, ring_slots=ring_slots)
        assert (
            plan.ring_slots is ring_slots
            and plan.declaration["ring_slots"] is ring_slots
        )
        out = _run(plan, case, ring_slots=ring_slots)
        _check_step(case, plan, out, f"decode ring_slots={ring_slots}")
        outs.append((out.clone(), case.block_ids_out.clone()))
        states.append(
            (bits(case.index_state_cache).clone(), bits(case.index_kv_cache).clone())
        )
        # the arm's refusals in both directions
        with pytest.raises(ValueError, match="ring_slots"):
            plan.run(
                case.hidden_states,
                None,
                None,
                case.weights38,
                case.w_o,
                **case.decode_args(ring_slots=not ring_slots),
            )
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    assert torch.equal(states[0][0], states[1][0]) and torch.equal(
        states[0][1], states[1][1]
    )


def test_fused_prepare_is_bitwise_the_unfused_twin_and_the_table_arm_the_per_token_arm():
    """The prepare fusion is a performance knob: the fused launch (the position-table arm) == the fused launch on per-token rows == the
    unfused four-launch twin (per-token rows: the only form it serves), bitwise on ``out``, the list, the pools, the ring and the cache;
    the declarations read the knob back and the stage lists differ as documented."""
    g = _geometry("24_2")
    results = {}
    for tag, kw, per_token in (
        ("fused_tables", dict(), False),
        ("fused_per_token", dict(rope_position_tables_only=False), True),
        (
            "unfused_per_token",
            dict(fuse_prepare=False, rope_position_tables_only=False),
            True,
        ),
    ):
        case = Qsa38PagedCase(g, (303, 77, 0, 2300), 4, page_size=16, seed=6)
        case.prime()
        plan = _plan(case, g, **kw)
        d = plan.declaration
        if tag.startswith("fused"):
            assert (
                d["fuse_prepare"] is True
                and "_QsaPrepare" in d["stages"]
                and d["indexer_launches"] == 2
            ), d
        else:
            assert (
                d["fuse_prepare"] is False
                and d["fuse_prepare_requested"] is False
                and d["indexer_launches"] == 4
            ), d
            assert (
                "_QkNormRope" in d["stages"]
                and "_CacheWrite" in d["stages"]
                and "_QsaPrepare" not in d["stages"]
            ), d["stages"]
            assert d["launches_per_execute"] == results["fused_tables"][3] + 3, (
                d["launches_per_execute"],
                results["fused_tables"][3],
            )
        out = _run(plan, case, per_token=per_token)
        _check_step(case, plan, out, f"decode {tag}")
        results[tag] = (
            out.clone(),
            case.block_ids_out.clone(),
            (
                bits(case.kv_cache).clone(),
                bits(case.index_state_cache).clone(),
                bits(case.index_kv_cache).clone(),
            ),
            d["launches_per_execute"],
        )
    base = results["fused_tables"]
    for tag in ("fused_per_token", "unfused_per_token"):
        got = results[tag]
        assert torch.equal(base[0], got[0]) and torch.equal(base[1], got[1]), (
            f"{tag}: out / the list differ from the fused table arm"
        )
        for x, y in zip(base[2], got[2], strict=True):
            assert torch.equal(x, y), (
                f"{tag}: the pools / ring / cache differ from the fused table arm"
            )
    # the unfused twin with the table arm is the vendored decline, worded by the envelope
    case = Qsa38PagedCase(g, (303,), 1, page_size=16, seed=6)
    with pytest.raises(
        NotImplementedError, match="unfused query norm reads the per-token"
    ):
        _plan(case, g, fuse_prepare=False, rope_position_tables_only=True)


def _kernel_names(plan, case, **outputs):
    """The CUDA kernels one ``plan.run`` launches on the case's tensors (the profiler's names), the case's sentinel fills outside the window."""
    case.reset_outputs()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        plan.run(
            case.hidden_states,
            None,
            None,
            case.weights38,
            case.w_o,
            **case.decode_args(),
            **outputs,
        )
        torch.cuda.synchronize()
    return [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]


def test_launch_census_matches_the_declaration():
    """The number of kernels one step launches (the profiler's count) equals the declaration's ``launches_per_execute``: 8 at a split
    (the slab GEMM, the prepare, the scorer, the top-k, the sparse tile, the combine, the out projection, its reducer), no memset / memcpy.
    At batch 1 (2 CTAs per chunk) the AUTO body knob resolves the LEAN decode tile on cc 10.7: the census names ONE ``cudnn_sparse_lean_kernel``
    and NO ``cudnn_kernel__kernel`` sparse core; the explicit ``"tcgen05"`` pin on the same bucket names the core and no lean kernel at
    the same launch count (the lean tile replaces the core one for one; both routes run their split combine)."""
    g = _geometry("24_2")
    case = Qsa38PagedCase(g, (8192,), 1, page_size=16, seed=7)
    case.prime()
    plan = _plan(case, g)
    assert (
        plan.sparse_tile == "sparse_lean_d256_f16"
        and plan.sparse_tile_form == "lean"
        and plan.decode_split_kv == 65
    ), (
        plan.sparse_tile,
        plan.decode_split_kv,
    )
    _run(plan, case)  # warm (the launch stream, the first-execute paths)

    def census(**outputs):
        case.reset_outputs()  # the test's own sentinel fills: outside the profiled window
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            plan.run(
                case.hidden_states,
                None,
                None,
                case.weights38,
                case.w_o,
                **case.decode_args(),
                **outputs,
            )
            torch.cuda.synchronize()
        return [
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]

    # the serving path (no optional outputs) and the test path (the list requested) issue the SAME launches: the list is written by
    # the top-k itself, no fill / memset of its own
    names = census()
    with_list = census(
        block_ids_out=case.block_ids_out, block_lens_out=case.block_lens_out
    )
    assert len(with_list) == len(names), (with_list, names)
    memops = [n for n in names if "memset" in n.lower() or "memcpy" in n.lower()]
    assert not memops, f"memset / memcpy on the execute path: {memops}"
    assert not [n for n in names if "FillFunctor" in n], names
    assert len(names) == plan.declaration["launches_per_execute"], (
        len(names),
        plan.declaration["launches_per_execute"],
        names,
    )
    assert plan.declaration["launches_per_execute"] == 8, (
        plan.declaration["launches_per_execute"],
        names,
    )
    assert (
        any("qsa_prepare" in n for n in names)
        and any("qsa_score_paged" in n for n in names)
        and any("qsa_topk_decode" in n for n in names)
    ), names
    if not names:  # CUPTI unavailable: the count assertions above would have failed first; the names are the pin below
        pytest.skip(
            "torch.profiler recorded no CUDA event (CUPTI unavailable on this node)"
        )
    has = lambda ns, tok: sum(tok in n for n in ns)  # noqa: E731
    assert (
        has(names, "cudnn_sparse_lean_kernel") == 1
        and has(names, "cudnn_kernel__kernel") == 0
    ), names
    assert has(names, "cudnn_kernel__combine_decode_grid_kernel") == 1, names
    # the explicit tcgen05 pin on the same bucket: the core by name, no lean kernel, the same launch count
    core = _plan(case, g, sparse_tile_form="tcgen05")
    assert (
        core.sparse_tile == "sparse_d256_f16"
        and core.sparse_tile_form == "tcgen05"
        and 2 <= core.decode_split_kv <= 17
    )
    _run(core, case)
    core_names = _kernel_names(core, case)
    assert (
        has(core_names, "cudnn_kernel__kernel") == 1
        and has(core_names, "cudnn_sparse_lean_kernel") == 0
    ), core_names
    assert (
        len(core_names) == len(names) == core.declaration["launches_per_execute"] == 8
    ), (len(core_names), len(names))
    assert has(core_names, "cudnn_kernel__combine_decode_grid_kernel") == 1, core_names


def test_the_two_bodies_of_the_sparse_core_agree_within_the_budget_at_batch_1():
    """The sparse core's two bodies on the same primed step at batch 1 x q_len 1 (the serving stacks' decode cell): ``"auto"`` (= the lean
    tile on cc 10.7) and the explicit ``"lean"`` are the SAME body (``out``, the list, the pools / ring / cache bitwise); ``"tcgen05"`` is
    the former function -- the list and every written row bitwise (the same prepare / scorer / top-k kernels run on both routes), ``out``
    within the block suite's budget of the lean body's (cosine; never asserted bitwise across the bodies: a 4 x 64-d split-K sum in fixed
    order vs one 256-d MMA chain), each body against the two-stage oracle; the readbacks and the workspace carve of each body
    (``qwen_3_8_block_workspace_size`` with the same knob equals the plan's; the lean split's wider slab exceeds the core's)."""
    from flashinfer.qwen_3_8_block import qwen_3_8_block_workspace_size
    from tests.test_helpers.qwen_3_8_block_cases import OUT_COS

    g = _geometry("24_2")
    results = {}
    for form in ("auto", "lean", "tcgen05"):
        case = Qsa38PagedCase(g, (8192,), 1, page_size=16, seed=9)
        case.prime()
        plan = _plan(case, g, sparse_tile_form=form)
        d = plan.declaration
        body = "tcgen05" if form == "tcgen05" else "lean"
        template = "sparse_d256_f16" if body == "tcgen05" else "sparse_lean_d256_f16"
        assert (
            d["sparse_tile_form_requested"] == form
            and plan.sparse_tile_form == d["sparse_tile_form"] == body
        ), d
        assert plan.sparse_tile == d["sparse_tile"] == template, d
        assert (
            (plan.decode_split_kv == 65 and d["sparse_sub_tiles_per_cta"] == 1)
            if body == "lean"
            else (d["sparse_sub_tiles_per_cta"] is None)
        ), d
        sized = qwen_3_8_block_workspace_size(
            "decode",
            geometry=g,
            device="cuda",
            page_size=16,
            batch_size=1,
            q_len_per_req=1,
            max_pages=int(case.block_table.shape[1]),
            sparse_tile_form=form,
        )
        assert sized == plan.workspace_size, (form, sized, plan.workspace_size)
        out = _run(plan, case)
        _check_step(case, plan, out, f"decode body {form}")
        results[form] = (
            out.clone(),
            case.block_ids_out.clone(),
            case.block_lens_out.clone(),
            (
                bits(case.kv_cache).clone(),
                bits(case.index_state_cache).clone(),
                bits(case.index_kv_cache).clone(),
            ),
            plan.workspace_size,
        )
    auto, lean, core = results["auto"], results["lean"], results["tcgen05"]
    assert torch.equal(auto[0], lean[0]) and torch.equal(auto[1], lean[1]), (
        "auto resolves the lean body at batch 1: the same bytes"
    )
    for x, y in zip(auto[3], lean[3], strict=True):
        assert torch.equal(x, y)
    assert torch.equal(lean[1], core[1]) and torch.equal(lean[2], core[2]), (
        "the list is the indexer's on both bodies: bitwise"
    )
    for x, y in zip(lean[3], core[3], strict=True):
        assert torch.equal(x, y), (
            "the pools / ring / cache rows are the prepare's on both bodies: bitwise"
        )
    a, b = lean[0].float().flatten(), core[0].float().flatten()
    cos = float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))
    md = float((a - b).abs().max())
    bitwise = bool(torch.equal(bits(lean[0]), bits(core[0])))
    print(
        f"\nlean vs tcgen05 at batch 1 x q_len 1: cos(out) {cos:.6f} max|diff| {md:.3e} (bitwise across the bodies {bitwise}: not required)"
    )
    assert cos > OUT_COS, f"the two bodies disagree past the budget: cos {cos}"
    assert lean[4] > core[4], (
        "the lean split's partial slab (65 partials) is wider than the core's"
    )


def test_three_carried_steps_with_a_rejected_draft():
    """ONE token stream across three steps over the same pools / ring / cache: S_q 1 (completes block 75 at position 302), S_q 1, then S_q 4
    with two of four rows accepted and RE-STEPPED over the same positions (the draft's completing row had written block 76 from rejected
    keys; the re-step rewrites it from the accepted ones).  After every step the two-stage oracle holds; the cache rows of every complete
    block of the ACCEPTED stream equal a from-scratch compress of that stream's raw keys (the prefill rows, the rows a step completes and
    a re-stepped block are one chain)."""
    from tests.test_helpers.qwen_3_5_block_reference import (
        gated_attention_block_reference,
    )

    g = _geometry("24_2")
    S0 = 302
    case = Qsa38PagedCase(g, (S0 + 1,), 1, page_size=16, seed=8, table_width=24, cap=8)
    case.prime()  # the prefix 0 .. 301 primed; the step's row is position 302
    # the stream grows to 310 tokens: positions 304 .. 309 live in a twentieth page the 303-token case never mapped -- hand it one of
    # the case's spare pages (the table has room: width 24)
    spare = sorted(set(range(1, case.num_pages)) - set(case.pages[0]))
    case.pages[0].append(spare[0])
    case.block_table[0, len(case.pages[0]) - 1] = spare[0]
    plans = {1: _plan(case, g), 4: None}
    L = S0 + 1
    h_acc = case.h_full[
        0
    ].clone()  # [1, L, d_model]: the ACCEPTED stream's hidden states (the oracle's full sequence)
    k_raw_acc = case.k_raw_full[0].clone()  # [L, d_i]
    out = _run(plans[1], case)
    _check_step(case, plans[1], out, "carried step 1 (q1, completes block 75)")
    stream = [(1, 1), (4, 2), (4, 4)]
    gen = torch.Generator(device="cuda").manual_seed(80)
    for step, (s_q, n_acc) in enumerate(stream, 2):
        pos0 = L
        h_step = torch.randn(
            1, s_q, g.d_model, generator=gen, device="cuda", dtype=torch.float32
        ).to(torch.bfloat16)
        h_full = torch.cat([h_acc, h_step], 1)
        Lf = pos0 + s_q
        # the oracle of the step: the dense reference over the accepted stream + the step's rows
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
        # re-aim the case at the new step (same caches, pages and table): a fresh case object for its oracles, the carried buffers reused
        nxt = Qsa38PagedCase(
            g,
            (Lf,),
            s_q,
            page_size=16,
            seed=8,
            table_width=24,
            weights=(case.weights38, case.w_o),
            cap=8,
        )
        nxt.refs[0] = ref
        nxt.h_full[0] = h_full
        nxt.hidden_states = h_step.view(s_q, g.d_model).contiguous()
        nxt.kv_cache, nxt.index_kv_cache, nxt.index_state_cache = (
            case.kv_cache,
            case.index_kv_cache,
            case.index_state_cache,
        )
        nxt.block_table, nxt.index_state_block_table, nxt.pages = (
            case.block_table,
            case.index_state_block_table,
            case.pages,
        )
        pos = torch.arange(pos0, Lf, device="cuda")
        pages_t = torch.tensor(case.pages[0], device="cuda")
        nxt.slot_mapping = (pages_t[pos // 16] * 16 + pos % 16).to(torch.int32)
        nxt.seq_lens = torch.tensor([Lf], dtype=torch.int32, device="cuda")
        nxt.k_raw_full[0] = (h_full[0].float() @ case.w_band_k.float().t()).to(
            torch.bfloat16
        )
        nxt.prefix_bits = (bits(case.kv_cache).clone(),)
        nxt._primed = True
        if plans[s_q] is None:
            plans[s_q] = _plan(nxt, g)
        out = _run(plans[s_q], nxt)
        label = f"carried step {step} (q{s_q}, accept {n_acc})"
        _check_step(nxt, plans[s_q], out, label)
        # accept n_acc rows: the stream, its raw keys; the cache's complete blocks of the ACCEPTED stream vs a from-scratch compress
        L = pos0 + n_acc
        h_acc = torch.cat([h_acc, h_step[:, :n_acc]], 1)
        k_raw_acc = torch.cat([k_raw_acc, _raw_keys(plans[s_q], nxt)[:n_acc]], 0)
        nb = L // 4
        kb = case.compress(k_raw_acc[None, : nb * 4], *case.position_tables)[0]
        j = torch.arange(nb, device="cuda")
        rows = case.index_kv_cache[pages_t[j // 4], j % 4, 0]
        assert torch.equal(bits(rows), bits(kb)), (
            f"{label}: the cache's {nb} complete blocks are not the from-scratch compress of the accepted stream"
        )
        case = nxt
    assert L == S0 + 1 + 1 + 2 + 4


def test_run_time_declines_are_typed_and_named():
    g = _geometry("24_2")
    case = Qsa38PagedCase(g, (300, 77), 1, page_size=16, seed=9)
    case.prime()
    plan = _plan(case, g)
    a = case.decode_args()
    w = (case.hidden_states, None, None, case.weights38, case.w_o)
    with pytest.raises(ValueError, match="rope_cos / rope_sin are refused"):
        plan.run(
            case.hidden_states,
            case.rope_cos,
            case.rope_sin,
            case.weights38,
            case.w_o,
            **a,
        )
    with pytest.raises(ValueError, match="index_kv_cache must be"):
        plan.run(*w, **dict(a, index_kv_cache=a["index_kv_cache"][:-1]))
    with pytest.raises(ValueError, match="index_state_cache holds cap=2"):
        plan.run(*w, **dict(a, index_state_cache=a["index_state_cache"][:, :2]))
    with pytest.raises(ValueError, match="index_state_block_table must be"):
        plan.run(
            *w, **dict(a, index_state_block_table=a["index_state_block_table"][:1])
        )
    with pytest.raises(ValueError, match="ring_slots=False"):
        plan.run(*w, **dict(a, index_state_slot_mapping=case.ring_slots))
    with pytest.raises(ValueError, match="k_scale / v_scale"):
        plan.run(
            *w,
            **a,
            k_scale=torch.ones(1, device="cuda"),
            v_scale=torch.ones(1, device="cuda"),
        )
    with pytest.raises(ValueError, match="block_ids_out must be"):
        plan.run(
            *w, **a, block_ids_out=torch.zeros(2, 100, dtype=torch.int32, device="cuda")
        )
    with pytest.raises(ValueError, match="weights.w_iq_norm must be"):
        bad = case.weights38._replace(w_iq_norm=case.weights38.w_iq_norm[:-1])
        plan.run(case.hidden_states, None, None, bad, case.w_o, **a)
    with pytest.raises(ValueError, match="seq_lens must be torch.int32"):
        plan.run(*w, **dict(a, seq_lens=a["seq_lens"].to(torch.int64)))
    with pytest.raises(ValueError, match="kv_cache does not match the plan's pools"):
        plan.run(*w, **dict(a, kv_cache=a["kv_cache"][:-1]))
    # the position tables one row short of the table's reach: refused at prepare by form
    short = (
        case.position_tables[0][:, :-1].contiguous(),
        case.position_tables[1][:, :-1].contiguous(),
    )
    with pytest.raises(ValueError, match="positions; the plan's block table reaches"):
        _plan(case, g, position_tables=short)
    # a ring too small for the step at prepare
    with pytest.raises(ValueError, match="needs at least q_len_per_req"):
        _plan(case, g, index_state_cache=case.index_state_cache[:, :3])
    assert not plan.supports(hidden_states=case.hidden_states.float(), **a)


def test_functional_entry_and_a_cached_step_run_without_a_host_sync():
    """The functional entry (the LRU) over the same step twice: the second call is a cache hit under ``set_sync_debug_mode("error")`` (no host
    read of device data) and bitwise the first."""
    from flashinfer.experimental.qwen_3_8_block import api
    from flashinfer.qwen_3_8_block import qwen_3_8_block_bf16_decode

    api.clear_plan_cache()
    g = _geometry("24_2")
    case = Qsa38PagedCase(g, (300, 77, 0), 2, page_size=16, seed=10)
    case.prime()
    args = dict(
        geometry=g,
        q_len_per_req=2,
        position_tables=case.position_tables,
        **case.decode_args(),
    )
    w = (case.hidden_states, None, None, case.weights38, case.w_o)
    first = qwen_3_8_block_bf16_decode(*w, **args)  # the plan compiles here
    torch.cuda.synchronize()
    assert len(api._PLANS) == 1
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        again = qwen_3_8_block_bf16_decode(*w, **args)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    torch.cuda.synchronize()
    assert torch.equal(first, again) and len(api._PLANS) == 1
    # the ring-slots arm through the functional entry is another bucket
    qwen_3_8_block_bf16_decode(
        *w,
        **dict(
            args, index_state_slot_mapping=case.ring_slots.to(case.slot_mapping.dtype)
        ),
    )
    assert len(api._PLANS) == 2
    api.clear_plan_cache()


def test_wrapper_over_a_shared_workspace_serves_two_buckets():
    """The serving idiom: ONE caller-owned workspace sized at the largest bucket, two plans (q_len 1 and 4) over it, run in turn; the
    workspace sizing read before the compile equals the compiled plan's."""
    from flashinfer.qwen_3_8_block import Qwen38BlockWrapper

    g = _geometry("24_2")
    cases = {
        q: Qsa38PagedCase(g, (300, 2300, 0), q, page_size=16, seed=11 + q)
        for q in (1, 4)
    }
    for c in cases.values():
        c.prime()
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for q, case in cases.items():
        w = Qwen38BlockWrapper(ws, "NHD", geometry=g)
        w.plan(
            "decode",
            weights=case.weights38,
            w_o=case.w_o,
            kv_cache=case.kv_cache,
            page_size=16,
            batch_size=case.b,
            q_len_per_req=q,
            block_table=case.block_table,
            **case.plan_args(),
        )
        plan = w.plan_info
        assert plan.workspace_size <= ws.numel() and plan._workspace is None
        from flashinfer.qwen_3_8_block import qwen_3_8_block_workspace_size

        sized = qwen_3_8_block_workspace_size(
            "decode",
            geometry=g,
            device="cuda",
            page_size=16,
            batch_size=case.b,
            q_len_per_req=q,
            max_pages=int(case.block_table.shape[1]),
            ring_cap=case.cap,
        )
        assert sized == plan.workspace_size, (sized, plan.workspace_size)
        case.reset_outputs()
        out = w.run(
            case.hidden_states,
            None,
            None,
            case.weights38,
            case.w_o,
            block_ids_out=case.block_ids_out,
            block_lens_out=case.block_lens_out,
            **case.decode_args(),
        )
        torch.cuda.synchronize()
        case.check_selection(
            case.block_ids_out,
            case.block_lens_out,
            plan._block._indexer_decode.views(ws, plan._block._ws)["qi"],
            n_blocks_max=plan.declaration["n_blocks_max"],
            label=f"wrapper q{q}",
        )
        case.check_out_from_list(out, case.block_ids_out, label=f"wrapper q{q}")
    with pytest.raises(ValueError, match="workspace_size"):
        Qwen38BlockWrapper(
            torch.empty(16, dtype=torch.uint8, device="cuda"), "NHD", geometry=g
        ).plan(
            "decode",
            weights=case.weights38,
            w_o=case.w_o,
            kv_cache=case.kv_cache,
            page_size=16,
            batch_size=case.b,
            q_len_per_req=4,
            block_table=case.block_table,
            **case.plan_args(),
        )


def test_compiled_plan_cache_reloads_the_decode_form():
    """A second plan of the same bucket reloads the exported kernels from FlashInfer's cache root once the in-process memo is cleared
    (the serving process's next start-up) and computes the same bytes."""
    import pathlib

    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost import compiled_cache
    from flashinfer.jit.env import FLASHINFER_CACHE_DIR

    g = _geometry("24_2")
    case = Qsa38PagedCase(g, (300, 77), 1, page_size=16, seed=13)
    case.prime()
    plan = _plan(case, g)
    caches = plan.declaration["caches"]
    assert (
        caches["compiled_plans"].startswith(
            str(FLASHINFER_CACHE_DIR / "qwen_3_5_block")
        )
        or "CUDNN_FRONTEND_COMPILED_CACHE" in __import__("os").environ
    )
    out = _run(plan, case)
    if not caches["compiled_plans_enabled"]:
        pytest.skip(
            "the compiled-plan cache is disabled (CUDNN_FRONTEND_DISABLE_COMPILED_CACHE)"
        )
    assert any(pathlib.Path(caches["compiled_plans"]).rglob("kernel.o")), (
        "the exported kernel objects"
    )
    compiled_cache.clear_memo()
    compiled_cache.reset_stats()
    again = _plan(case, g)
    assert compiled_cache.stats()["hits"] >= 1, compiled_cache.stats()
    case2 = Qsa38PagedCase(g, (300, 77), 1, page_size=16, seed=13)
    case2.prime()
    out2 = _run(again, case2)
    assert torch.equal(out, out2), (
        "a plan reloaded from the files differs from the compiled one"
    )


def test_shared_workspace_is_sized_over_every_planned_bucket_not_the_largest_batch():
    """Under ``sparse_tile_form="auto"`` the workspace need is NOT monotonic in the batch: batch 2 (and 1) take the lean body, whose
    split-KV partial slab (split 65 at the default k) is wider than the tcgen05 core's at batch 4, so the serving idiom sizes the ONE
    shared workspace at the MAX of ``qwen_3_8_block_workspace_size`` over every planned bucket.  A buffer sized from the batch-4 query
    alone is rejected by the batch-2 plan with the typed workspace error (never an unsafe launch); sized from the maximum it serves both."""
    from flashinfer.qwen_3_8_block import (
        Qwen38BlockWrapper,
        qwen_3_8_block_workspace_size,
    )

    g = _geometry("24_2")
    cases = {
        b: Qsa38PagedCase(g, (8192,) * b, 1, page_size=16, seed=21 + b) for b in (2, 4)
    }
    for c in cases.values():
        c.prime()
    max_pages = int(cases[4].block_table.shape[1])
    sized = {
        b: qwen_3_8_block_workspace_size(
            "decode",
            geometry=g,
            device="cuda",
            page_size=16,
            batch_size=b,
            q_len_per_req=1,
            max_pages=max_pages,
        )
        for b in (2, 4)
    }
    core4 = qwen_3_8_block_workspace_size(
        "decode",
        geometry=g,
        device="cuda",
        page_size=16,
        batch_size=4,
        q_len_per_req=1,
        max_pages=max_pages,
        sparse_tile_form="tcgen05",
    )
    assert sized[4] == core4 and sized[2] > sized[4], (
        sized
    )  # AUTO: the core at batch 4, the lean body (the wider slab) at batch 2

    def _plan_into(ws, case):
        w = Qwen38BlockWrapper(ws, "NHD", geometry=g)
        w.plan(
            "decode",
            weights=case.weights38,
            w_o=case.w_o,
            kv_cache=case.kv_cache,
            page_size=16,
            batch_size=case.b,
            q_len_per_req=1,
            block_table=case.block_table,
            **case.plan_args(),
        )
        return w

    too_small = torch.empty(
        sized[4], dtype=torch.uint8, device="cuda"
    )  # the largest-batch idiom
    with pytest.raises(ValueError, match="workspace"):
        _plan_into(too_small, cases[2])
    ws = torch.empty(
        max(sized.values()), dtype=torch.uint8, device="cuda"
    )  # the max over the planned buckets
    for b in (4, 2):
        w = _plan_into(ws, cases[b])
        assert (
            w.plan_info.workspace_size == sized[b]
            and w.plan_info.workspace_size <= ws.numel()
        )
        # a wrapper plan owns no workspace: the shared buffer rides every run; the oracle helper reads a plan's OWN workspace, so the
        # bucket is also run through a plan that owns one and the two must agree bitwise (the same body, the same bytes)
        out = _run(w.plan_info, cases[b], workspace_buffer=ws).clone()
        ids_w = cases[b].block_ids_out.clone()
        lens_w = cases[b].block_lens_out.clone()
        ref = _plan(cases[b], g)
        out_ref = _run(ref, cases[b])
        _check_step(cases[b], ref, out_ref, f"shared workspace over the max, batch {b}")
        assert (
            torch.equal(out, out_ref)
            and torch.equal(ids_w, cases[b].block_ids_out)
            and torch.equal(lens_w, cases[b].block_lens_out)
        ), (
            f"batch {b}: the wrapper plan over the shared buffer and the owning plan differ"
        )
