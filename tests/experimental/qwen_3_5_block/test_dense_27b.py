# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8-27B dense attention layer through the package -- the 24 / 4 x 256 geometry at ``d_model`` 5120 (``rope_dim`` 64, GQA group 6,
``n_qkvg`` 14336: the row cuDNN Frontend 4c8e24355 pins in its layout contract and family table) -- and the serving stacks' packed K|V pool
(``kv_layout="UNIFIED"``: ONE ``[num_pages, H_kv, page_size, 2 x D]`` tensor bound as two half views).

CPU: the geometry facts, the probe's answer by device and its cache spellings, the packed pool's form checks.  GPU (SM107): the probe
served and the typed declines around the geometry; DECODE ``q_len`` 1 at page 16 and 4 at page 32, batch 1 and 4 (a padded entry among
them, contexts to 2048 keys) on the packed decode tile's claimed routes; EXTEND as a 512-token step with an empty prefix and as a 512-token
chunk behind a 512-row cached prefix; the dense PREFILL at 512 tokens; the packed pool bitwise the two-tensor HND pool holding the same
bytes on the decode form (the functional entry point) and the extend form (a prepared plan), its half views' identity and strides, every
malformed packed pool refused by name at run time."""

import pytest
import torch

from tests.test_helpers.qwen_3_5_block_cases import (
    PagedCase,
    cache_rows,
    new_kv_cache,
    random_weights,
    ref_geometry,
)
from tests.test_helpers.qwen_3_5_block_reference import (
    build_rope_tables,
    cosine,
    gated_attention_block_reference,
)

# the Qwen3.8-27B dense attention layer at TP 1: d_model 5120, 24 query / 4 KV heads x 256, rope_dim 64 (a quarter of the head), rms eps 1e-6
GEOM_27B = dict(d_model=5120, num_qo_heads=24, num_kv_heads=4)
N_QKVG_27B = 14336
D = 256
H_KV = 4
UNIFIED_ROW = "[num_pages, H_kv, page_size, 2 x D]"
# (page, q_len): the plain decode step at page 16, the four-token speculative step at page 32 -- the stack's page per step width
DECODE_CELLS = [pytest.param(16, 1, id="p16-q1"), pytest.param(32, 4, id="p32-q4")]
# batch 1: the stack's longest context; batch 4: that context, a PADDED entry, a short one (kv_len >= q_len at q_len 4), a tail
LENS_B1 = (2048,)
LENS_B4 = (2048, 0, 5, 1000)
# the packed decode tile's claimed routes for group 6 (24 / 4) under the single-unit policy: (n_q, token_units) per q_len
ROUTE_27B = {1: (16, 1), 4: (32, 1)}


def _geometry(**overrides):
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry

    return Qwen35BlockGeometry(**dict(GEOM_27B, **overrides))


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _unified_of(kv_cache_hnd: torch.Tensor) -> torch.Tensor:
    """The stack's packed K|V pool ``[num_pages, H_kv, page_size, 2 x D]`` holding the same values as an HND 5-D cache -- a new,
    contiguous buffer (K in the first ``D`` columns of every row, V in the last)."""
    return torch.cat((kv_cache_hnd[:, 0], kv_cache_hnd[:, 1]), dim=-1).contiguous()


def _uniform_step(case, q_len):
    """The decode step's tensors at a UNIFORM ``[batch x q_len]``: the live sequences' rows from the case, a padded entry (kv_len 0)
    finite zero rows with cos 1 / sin 0 and slot -1.  Returns ``(h, cos, sin, slots, live)`` with ``live`` the live sequence indices."""
    g, b = case.geometry, case.b
    h = torch.zeros(b * q_len, g.d_model, dtype=torch.bfloat16, device="cuda")
    cos = torch.ones(b * q_len, g.rope_dim, dtype=torch.bfloat16, device="cuda")
    sin = torch.zeros_like(cos)
    slots = torch.full((b * q_len,), -1, dtype=torch.int32, device="cuda")
    live = [i for i, L in enumerate(case.kv_lens) if L]
    for j, i in enumerate(live):
        h[i * q_len : (i + 1) * q_len] = case.hidden_states[j * q_len : (j + 1) * q_len]
        cos[i * q_len : (i + 1) * q_len] = case.rope_cos[j * q_len : (j + 1) * q_len]
        sin[i * q_len : (i + 1) * q_len] = case.rope_sin[j * q_len : (j + 1) * q_len]
        slots[i * q_len : (i + 1) * q_len] = case.slot_mapping[
            j * q_len : (j + 1) * q_len
        ]
    return h, cos, sin, slots, live


def _live_rows(out, live, q_len):
    return torch.cat([out[i * q_len : (i + 1) * q_len] for i in live])


# --------------------------------------------------------------------------- CPU


def test_geometry_facts_of_the_27b_member():
    """The 24 / 4 x 256 geometry at d_model 5120: group 6, n_qkvg 14336, the default scale, served at rope_dim 64 and 128; the declines
    around the row name the field (a rope_dim off the envelope, another head dim, a d_model off the projection tile, a ragged group)."""
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry

    g = Qwen35BlockGeometry(**GEOM_27B)
    assert (g.group, g.n_qkvg, g.head_dim, g.rope_dim) == (6, N_QKVG_27B, 256, 64)
    assert g.qk_norm and g.qk_norm_eps == 1e-6 and g.scale == 0.0625
    assert g.unsupported_reason() is None
    assert (
        Qwen35BlockGeometry(**dict(GEOM_27B, rope_dim=128)).unsupported_reason() is None
    )
    for kw, why in (
        (dict(rope_dim=32), "rope_dim=32"),
        (dict(rope_dim=192), "rope_dim=192"),
        (dict(rope_dim=256), "rope_dim=256"),
        (dict(head_dim=128), "head_dim=256"),
        (dict(d_model=5000), "d_model=5000"),
    ):
        reason = Qwen35BlockGeometry(**dict(GEOM_27B, **kw)).unsupported_reason()
        assert reason is not None and why in reason, (kw, reason)
    with pytest.raises(ValueError, match="multiple of num_kv_heads"):
        Qwen35BlockGeometry(d_model=5120, num_qo_heads=24, num_kv_heads=5)


def test_probe_on_cpu_names_the_device_and_the_cache_spellings():
    """The probe never raises: on a CPU device the 27B member is declined by device for every form, and the static facts carry the three
    cache spellings with one tensor shape each (the UNIFIED row = the vendored tree's unified pool layout)."""
    from flashinfer.qwen_3_5_block import qwen_3_5_block_support
    from flashinfer.experimental.qwen_3_5_block import support

    sup = qwen_3_5_block_support(torch.device("cpu"), _geometry())
    assert not sup.served and sup.forms == () and "SM107" in sup.reason
    assert all(
        not sup.form(f).served and sup.form(f).reason == sup.reason
        for f in ("prefill", "extend", "decode")
    )
    assert sup.kv_layouts == ("NHD", "HND", "UNIFIED") == support.KV_LAYOUTS
    assert support.KV_LAYOUT_UNIFIED == "UNIFIED"
    assert sup.kv_layout_shapes == support.KV_LAYOUT_SHAPES
    assert set(sup.kv_layout_shapes) == set(sup.kv_layouts)
    assert sup.kv_layout_shapes["UNIFIED"] == UNIFIED_ROW
    assert sup.kv_layout_shapes["HND"] == "[num_pages, 2, H_kv, page_size, D]"


def test_packed_pool_form_checks_are_typed_on_cpu():
    """Every malformed packed pool is a ``ValueError`` naming the operand and the expected form, host-side: a 5-D buffer or a ``(k, v)``
    pair under the UNIFIED spelling (pointed at the spelling they bind under), another trailing shape -- the sibling package's NHD-ordered
    row among them --, a page size off the shape, a non-tensor, an unknown spelling; a well-formed host pool fails the device row last.  The
    functional entry points' page-size reading takes the packed pool's second axis and keeps the 5-D buffer's / the pair's page axis."""
    from flashinfer.experimental.qwen_3_5_block.api import (
        _page_size_of,
        kv_views,
        kv_views_unified,
    )

    g, bf = _geometry(), torch.bfloat16
    ok = torch.zeros(
        3, H_KV, 16, 2 * D, dtype=bf
    )  # the stack's pool at page 16, in host memory
    with pytest.raises(ValueError, match="kv_cache must be a CUDA tensor"):
        kv_views(ok, "UNIFIED", geometry=g, page_size=16, dtype=bf)
    with pytest.raises(ValueError, match="kv_cache must be a CUDA tensor"):
        kv_views_unified(ok, geometry=g, page_size=16, dtype=bf)
    five = torch.zeros(3, 2, H_KV, 16, D, dtype=bf)
    with pytest.raises(
        ValueError,
        match=r"a 5-D \[num_pages, 2, \.\.\.\] buffer binds under kv_layout='NHD' / 'HND'",
    ):
        kv_views(five, "UNIFIED", geometry=g, page_size=16, dtype=bf)
    with pytest.raises(ValueError, match=r"not a \(k_cache, v_cache\) pair"):
        kv_views(
            (five[:, 0], five[:, 1]), "UNIFIED", geometry=g, page_size=16, dtype=bf
        )
    for bad in (
        torch.zeros(3, H_KV, 16, D, dtype=bf),  # two-tensor rows, not the packed one
        torch.zeros(
            3, 16, H_KV, 2 * D, dtype=bf
        ),  # the sibling package's NHD-ordered row
        torch.zeros(3, 2, 16, 2 * D, dtype=bf),  # another head count
    ):
        with pytest.raises(ValueError, match=r"\[\*, 4, 16, 512\]"):
            kv_views(bad, "UNIFIED", geometry=g, page_size=16, dtype=bf)
    with pytest.raises(ValueError, match=r"\[\*, 4, 32, 512\]"):
        kv_views(ok, "UNIFIED", geometry=g, page_size=32, dtype=bf)
    with pytest.raises(ValueError, match="kv_cache must be a torch.Tensor"):
        kv_views(3, "UNIFIED", geometry=g, page_size=16, dtype=bf)
    with pytest.raises(ValueError, match="kv_layout must be 'NHD', 'HND' or 'UNIFIED'"):
        kv_views(ok, "PACKED", geometry=g, page_size=16, dtype=bf)
    # the page size the functional entry points read off the cache's shape
    assert _page_size_of(ok, "UNIFIED", g) == 16
    assert _page_size_of(torch.zeros(3, H_KV, 32, 2 * D, dtype=bf), "unified", g) == 32
    with pytest.raises(ValueError, match="a UNIFIED kv_cache is"):
        _page_size_of(five, "UNIFIED", g)
    with pytest.raises(ValueError, match=r"not a \(k_cache, v_cache\) pair"):
        _page_size_of((ok, ok), "UNIFIED", g)
    assert _page_size_of(torch.zeros(3, 2, 16, H_KV, D, dtype=bf), "NHD", g) == 16
    assert _page_size_of(five, "HND", g) == 16
    assert _page_size_of((five[:, 0], five[:, 1]), "HND", g) == 16


# --------------------------------------------------------------------------- GPU (SM107)


def _plan(form, case, g, **kw):
    from flashinfer.qwen_3_5_block import prepare_qwen_3_5_block_bf16

    return prepare_qwen_3_5_block_bf16(
        form,
        geometry=g,
        device="cuda",
        weights=case.weights,
        w_o=case.w_o,
        kv_cache=kw.pop("kv_cache", case.kv_cache),
        kv_layout=kw.pop("kv_layout", case.layout),
        page_size=case.page_size,
        batch_size=case.b,
        block_table=case.block_table,
        **kw,
    )


@pytest.mark.arch_rubin
@pytest.mark.usefixtures("rubin")
def test_probe_serves_the_27b_member_and_declines_around_it_by_name(monkeypatch):
    """On the served device the probe advertises the 27B member on every form (rope_dim 64 and 128) with the three cache spellings, and
    the declines around the row agree between the probe and ``prepare``: a step past 32 decode rows is the decode envelope's typed
    decline, a declined rope_dim / head dim / d_model is the geometry gate's before any vendored code, f16 the entry's dtype decline."""
    from flashinfer.qwen_3_5_block import (
        prepare_qwen_3_5_block_bf16,
        qwen_3_5_block_support,
    )
    from flashinfer.experimental.qwen_3_5_block import api as impl

    dev = torch.device("cuda", 0)
    g = _geometry()
    sup = qwen_3_5_block_support(dev, g)
    assert sup.served and sup.reason is None, sup.reason
    assert sup.forms == ("prefill", "extend", "decode")
    assert sup.prefill.served and sup.extend.served and sup.decode.served
    assert sup.extend.declines and "head_dim" in sup.decode.declines
    assert sup.kv_layouts == ("NHD", "HND", "UNIFIED")
    assert sup.kv_layout_shapes["UNIFIED"] == UNIFIED_ROW
    assert sup.decode_max_q_len == 4 and sup.decode_max_rows == 32
    assert qwen_3_5_block_support(dev, _geometry(rope_dim=128)).served
    for bad, why in (
        (_geometry(rope_dim=32), "rope_dim=32"),
        (_geometry(rope_dim=192), "rope_dim=192"),
        (_geometry(head_dim=128), "head_dim=256"),
        (_geometry(d_model=5000), "d_model=5000"),
    ):
        s = qwen_3_5_block_support(dev, bad)
        assert not s.served and s.forms == () and why in s.reason, (why, s.reason)
    weights, w_o = random_weights(g, device="cuda", seed=3)
    kv = new_kv_cache(8, H_KV, 16, D, "NHD", device="cuda")
    common = dict(
        device=dev,
        weights=weights,
        w_o=w_o,
        kv_cache=kv,
        page_size=16,
        batch_size=1,
        max_pages=4,
    )
    with pytest.raises(NotImplementedError, match="decode envelope"):
        prepare_qwen_3_5_block_bf16("decode", geometry=g, q_len_per_req=33, **common)

    def reached():
        raise AssertionError(
            "the vendored tree was reached for a geometry / dtype the gate declines"
        )

    monkeypatch.setattr(impl, "_frost_block", reached)
    for form, extra in (
        ("prefill", dict(seq_len=8, max_pages=None)),
        ("extend", dict(token_capacity=16)),
        ("decode", {}),
    ):
        with pytest.raises(NotImplementedError, match="rope_dim=32") as info:
            prepare_qwen_3_5_block_bf16(
                form, geometry=_geometry(rope_dim=32), **dict(common, **extra)
            )
        assert str(info.value).startswith(f"qwen_3_5_block ({form}):")
    with pytest.raises(NotImplementedError, match="bfloat16"):
        impl.prepare("decode", geometry=g, dtype=torch.float16, **common)


@pytest.mark.arch_rubin
@pytest.mark.usefixtures("rubin")
@pytest.mark.parametrize("lens", [LENS_B1, LENS_B4], ids=["b1", "b4"])
@pytest.mark.parametrize("page, q_len", DECODE_CELLS)
def test_decode_27b_matches_the_oracle_on_the_claimed_route(page, q_len, lens):
    """The DECODE form at the 27B geometry on the HND cache: q_len 1 at page 16 and 4 at page 32, batch 1 (a 2048-key context) and batch 4
    (that context, a padded entry, a 5-key and a 1000-key sequence).  The route is the packed decode tile's CLAIMED single-unit route for
    group 6 (16 columns at q_len 1, 32 at q_len 4, one token unit, one KV stream), stage (1) the lowered decode chain with the write-through
    folded into the norm + RoPE launch; the oracle on every live sequence, the padded entry exactly 0, the pools' step rows, the NULL page
    untouched, two launches bitwise."""
    g = _geometry()
    case = PagedCase(
        g,
        lens,
        [q_len if L else 0 for L in lens],
        page_size=page,
        layout="HND",
        seed=70 + page + q_len + len(lens),
    )
    plan = _plan("decode", case, g, q_len_per_req=q_len)
    d = plan.declaration
    route = d["decode_route"]
    n_q, units = ROUTE_27B[q_len]
    assert (
        route["tile"],
        route["s_q"],
        route["n_q"],
        route["token_units"],
        route["pack_gqa"],
        route["claimed"],
        route["kv_streams"],
        route["policy"],
    ) == ("decode_d256_f16", q_len, n_q, units, True, True, 1, "single_unit"), route
    assert d["fuse_norm_rope_effective"] is False, d
    assert d["cache_write_in_norm_rope"] is True and "_CacheWrite" not in d["stages"], d
    assert "_Sdpa" in d["stages"] and d["kv_layout"] == "HND", d
    h, cos, sin, slots, live = _uniform_step(case, q_len)
    args = dict(case.decode_args(), slot_mapping=slots)
    out = plan.run(h, cos, sin, case.weights, case.w_o, **args)
    torch.cuda.synchronize()
    assert out.shape == (case.b * q_len, g.d_model) and out.dtype == torch.bfloat16
    for i, L in enumerate(lens):
        if L == 0:
            assert (out[i * q_len : (i + 1) * q_len] == 0).all(), (
                "a padded entry (seq_len 0) must yield out rows exactly 0"
            )
    label = f"decode 27B p{page} q{q_len} b{case.b}"
    case.check_out(_live_rows(out, live, q_len), label=label)
    case.check_written_rows(label=label)
    assert (case.kv_cache[0] == 0).all(), (
        "the NULL page (block 0) must never be written"
    )
    again = plan.run(h, cos, sin, case.weights, case.w_o, **args)
    torch.cuda.synchronize()
    assert torch.equal(out, again), f"{label}: two launches differ"


@pytest.mark.arch_rubin
@pytest.mark.usefixtures("rubin")
@pytest.mark.parametrize("page, q_len", DECODE_CELLS)
def test_decode_27b_packed_pool_is_bitwise_the_hnd_pool(page, q_len):
    """The stack's ONE ``[pages, 4, P, 512]`` tensor as ``kv_layout="UNIFIED"`` against the two-tensor HND cache holding the same bytes,
    through the functional entry point (the page size read off the 4-D shape, the plan cache keyed on the layout): the half views are the
    two halves of the one tensor (the same storage, the V base D elements past the K base, strides (H x P x 2D, P x 2D, 2D, 1)); ``out``
    bitwise the HND run's, the pool's halves bitwise the HND pools after the step (the write-through landed in the right halves and
    nowhere else), the oracle, two launches bitwise."""
    from flashinfer.qwen_3_5_block import qwen_3_5_block_bf16_decode
    from flashinfer.experimental.qwen_3_5_block.api import kv_views, kv_views_unified

    g = _geometry()
    case = PagedCase(
        g,
        LENS_B4,
        [q_len if L else 0 for L in LENS_B4],
        page_size=page,
        layout="HND",
        seed=90 + page,
    )
    unified = _unified_of(case.kv_cache)
    n = int(unified.shape[0])
    assert tuple(unified.shape) == (n, H_KV, page, 2 * D) and unified.is_contiguous()
    k, v = kv_views(
        unified, "UNIFIED", geometry=g, page_size=page, dtype=torch.bfloat16
    )
    k2, v2 = kv_views_unified(unified, geometry=g, page_size=page, dtype=torch.bfloat16)
    es = unified.element_size()
    assert k.data_ptr() == k2.data_ptr() == unified.data_ptr()
    assert v.data_ptr() == v2.data_ptr() == unified.data_ptr() + D * es
    assert k._base is unified and v._base is unified, "the views must alias the pool"
    assert tuple(k.shape) == tuple(v.shape) == (n, H_KV, page, D)
    assert (
        tuple(k.stride())
        == tuple(v.stride())
        == (H_KV * page * 2 * D, page * 2 * D, 2 * D, 1)
    )
    h, cos, sin, slots, live = _uniform_step(case, q_len)
    args = dict(
        geometry=g,
        q_len_per_req=q_len,
        slot_mapping=slots,
        block_table=case.block_table,
        seq_lens=case.seq_lens,
    )
    w = (h, cos, sin, case.weights, case.w_o)
    out_h = qwen_3_5_block_bf16_decode(
        *w, kv_cache=case.kv_cache, kv_layout="HND", **args
    )
    torch.cuda.synchronize()
    out_u = qwen_3_5_block_bf16_decode(
        *w, kv_cache=unified, kv_layout="UNIFIED", **args
    )
    torch.cuda.synchronize()
    label = f"decode 27B packed pool p{page} q{q_len}"
    case.check_out(_live_rows(out_u, live, q_len), label=label)
    case.check_written_rows(label=label + " (HND twin)")
    assert torch.equal(out_u, out_h), (
        f"{label}: out differs from the HND twin's: max|diff| {(out_u.float() - out_h.float()).abs().max().item():.3e}"
    )
    assert torch.equal(_bits(unified[..., :D]), _bits(case.kv_cache[:, 0])), (
        f"{label}: the K half differs from the HND K pool after the step"
    )
    assert torch.equal(_bits(unified[..., D:]), _bits(case.kv_cache[:, 1])), (
        f"{label}: the V half differs from the HND V pool after the step"
    )
    again = qwen_3_5_block_bf16_decode(
        *w, kv_cache=unified, kv_layout="UNIFIED", **args
    )
    torch.cuda.synchronize()
    assert torch.equal(out_u, again), f"{label}: two launches differ"


@pytest.mark.arch_rubin
@pytest.mark.usefixtures("rubin")
def test_packed_pool_plan_facts_and_the_run_time_declines():
    """A prepared decode plan over the packed pool reads the layout back (the declaration, the compiled-in half-view strides) and runs the
    oracle; at run time every malformed packed pool is a ``ValueError`` naming the operand before any launch: the HND 5-D cache handed to
    the UNIFIED plan (pointed at the spelling it binds under), a ``(k, v)`` pair, an f16 pool, a pool whose head dim is not contiguous, a
    stride off the 16-byte rule, a misaligned base, two-tensor rows, another page count; ``supports`` says no."""
    g = _geometry()
    page = 16
    case = PagedCase(g, (300, 77), (1, 1), page_size=page, layout="HND", seed=101)
    unified = _unified_of(case.kv_cache)
    n = int(unified.shape[0])
    plan = _plan("decode", case, g, kv_cache=unified, kv_layout="UNIFIED")
    assert plan.kv_layout == "UNIFIED" and plan.declaration["kv_layout"] == "UNIFIED"
    assert plan.num_pages == n and plan.page_size == page
    assert plan._pool_shape == (n, H_KV, page, D)
    assert (
        plan._k_strides
        == plan._v_strides
        == (H_KV * page * 2 * D, page * 2 * D, 2 * D, 1)
    )
    a = dict(case.decode_args(), kv_cache=unified)
    w = (case.hidden_states, case.rope_cos, case.rope_sin, case.weights, case.w_o)
    out = plan.run(*w, **a)
    torch.cuda.synchronize()
    case.check_out(out, label="decode 27B packed pool (prepared plan)")
    bf = torch.bfloat16
    with pytest.raises(
        ValueError,
        match=r"a 5-D \[num_pages, 2, \.\.\.\] buffer binds under kv_layout='NHD' / 'HND'",
    ):
        plan.run(*w, **dict(a, kv_cache=case.kv_cache))
    with pytest.raises(ValueError, match=r"not a \(k_cache, v_cache\) pair"):
        plan.run(*w, **dict(a, kv_cache=(case.kv_cache[:, 0], case.kv_cache[:, 1])))
    with pytest.raises(ValueError, match="kv_cache must be torch.bfloat16"):
        plan.run(*w, **dict(a, kv_cache=unified.to(torch.float16)))
    with pytest.raises(ValueError, match="the head dim must be contiguous"):
        plan.run(
            *w,
            **dict(
                a,
                kv_cache=torch.zeros(n, H_KV, page, 4 * D, dtype=bf, device="cuda")[
                    ..., ::2
                ],
            ),
        )
    with pytest.raises(ValueError, match="16-byte multiple"):
        plan.run(
            *w,
            **dict(
                a,
                kv_cache=torch.zeros(n, H_KV, page, 2 * D + 4, dtype=bf, device="cuda")[
                    ..., : 2 * D
                ],
            ),
        )
    buf = torch.zeros(n * H_KV * page * 2 * D + 4, dtype=bf, device="cuda")
    with pytest.raises(ValueError, match="16-byte aligned"):
        plan.run(*w, **dict(a, kv_cache=buf[4:].view(n, H_KV, page, 2 * D)))
    with pytest.raises(ValueError, match=r"\[\*, 4, 16, 512\]"):
        plan.run(*w, **dict(a, kv_cache=unified[..., :D].contiguous()))
    with pytest.raises(ValueError, match="does not match the plan's pools"):
        plan.run(*w, **dict(a, kv_cache=unified[:-1]))
    assert not plan.supports(
        hidden_states=case.hidden_states,
        rope_cos=case.rope_cos,
        rope_sin=case.rope_sin,
        **dict(a, kv_cache=case.kv_cache),
    )
    assert plan.supports(
        hidden_states=case.hidden_states,
        rope_cos=case.rope_cos,
        rope_sin=case.rope_sin,
        **a,
    )


@pytest.mark.arch_rubin
@pytest.mark.usefixtures("rubin")
def test_extend_27b_512_token_step_with_an_empty_prefix():
    """The EXTEND form as a 512-token prefill step (batch 1, an empty prefix, page 16, NHD) on a 512-token plan: stage (1) the fused norm +
    RoPE projection (above decode row counts), the forced 256-wide out-projection tile (any live count), the oracle over the sequence, the
    pools' rows, the NULL page untouched."""
    g = _geometry()
    case = PagedCase(g, (512,), (512,), page_size=16, layout="NHD", seed=111)
    plan = _plan("extend", case, g, token_capacity=512)
    d = plan.declaration
    assert d["fuse_norm_rope_effective"] is True and plan.live_rows_multiple == 1, d
    assert d["kv_layout"] == "NHD" and plan.q_len == 512, d
    out = plan.run(
        case.hidden_states,
        case.rope_cos,
        case.rope_sin,
        case.weights,
        case.w_o,
        num_actual_tokens=512,
        **case.extend_args(),
    )
    torch.cuda.synchronize()
    assert out.shape == (512, g.d_model)
    case.check_out(out, label="extend 27B T512 empty prefix")
    case.check_written_rows(label="extend 27B T512 empty prefix")
    assert (case.kv_cache[0] == 0).all(), "the NULL page must never be written"


@pytest.mark.arch_rubin
@pytest.mark.usefixtures("rubin")
def test_extend_27b_chunk_behind_a_cached_prefix_on_the_hnd_and_the_packed_pool():
    """A 512-token chunk behind a 512-row cached prefix (kv_len 1024, page 32, batch 1) on a 512-token plan: the oracle over the whole
    1024-token sequence (the prefix rows written by the oracle, the chunk's rows by the block); the same chunk through the packed K|V pool
    holding the same bytes (a prepared ``UNIFIED`` plan) is bitwise the HND run on ``out`` and on the pools' halves."""
    g = _geometry()
    case = PagedCase(g, (1024,), (512,), page_size=32, layout="HND", seed=121)
    unified = _unified_of(
        case.kv_cache
    )  # the prefix rows in both pools before either run
    plan = _plan("extend", case, g, token_capacity=512)
    d = plan.declaration
    assert d["fuse_norm_rope_effective"] is True and plan.live_rows_multiple == 1, d
    assert plan.max_pages == 32 and d["kv_layout"] == "HND", d
    w = (case.hidden_states, case.rope_cos, case.rope_sin, case.weights, case.w_o)
    out = plan.run(*w, num_actual_tokens=512, **case.extend_args())
    torch.cuda.synchronize()
    label = "extend 27B chunk 512 behind prefix 512 p32"
    case.check_out(out, label=label)
    case.check_written_rows(label=label)
    plan_u = _plan(
        "extend", case, g, token_capacity=512, kv_cache=unified, kv_layout="UNIFIED"
    )
    assert (
        plan_u.kv_layout == "UNIFIED" and plan_u.declaration["kv_layout"] == "UNIFIED"
    )
    assert plan_u._k_strides == (H_KV * 32 * 2 * D, 32 * 2 * D, 2 * D, 1)
    out_u = plan_u.run(
        *w, num_actual_tokens=512, **dict(case.extend_args(), kv_cache=unified)
    )
    torch.cuda.synchronize()
    assert torch.equal(out_u, out), (
        f"{label}: the packed pool's out differs from the HND run's: max|diff| {(out_u.float() - out.float()).abs().max().item():.3e}"
    )
    assert torch.equal(_bits(unified[..., :D]), _bits(case.kv_cache[:, 0])), (
        f"{label}: the K half differs from the HND K pool after the chunk"
    )
    assert torch.equal(_bits(unified[..., D:]), _bits(case.kv_cache[:, 1])), (
        f"{label}: the V half differs from the HND V pool after the chunk"
    )


@pytest.mark.arch_rubin
@pytest.mark.usefixtures("rubin")
def test_prefill_27b_dense_form_at_512_tokens():
    """The dense PREFILL form at batch 1 x 512 tokens with the write-through into an NHD cache at page 16: the oracle on ``out``, the
    pools' K / V rows within the fused norm + RoPE rounding, the NULL page untouched."""
    from flashinfer.qwen_3_5_block import qwen_3_5_block_bf16_prefill

    g = _geometry()
    rg = ref_geometry(g)
    weights, w_o = random_weights(g, device="cuda", seed=131)
    gen = torch.Generator(device="cuda").manual_seed(131)
    seq, page = 512, 16
    h = torch.randn(
        1, seq, g.d_model, generator=gen, device="cuda", dtype=torch.float32
    ).to(torch.bfloat16)
    cos, sin = build_rope_tables(
        seq, g.rope_dim, base=rg.rope_base, batch=1, device="cuda", dtype=torch.bfloat16
    )
    ref = gated_attention_block_reference(
        h, weights.w_qkvg, weights.w_q_norm, weights.w_k_norm, cos, sin, w_o, rg
    )
    kv = new_kv_cache(seq // page + 3, H_KV, page, D, "NHD", device="cuda")
    pos = torch.arange(seq, device="cuda")
    slots = ((1 + pos // page) * page + pos % page).to(torch.int32)
    out = qwen_3_5_block_bf16_prefill(
        h,
        cos,
        sin,
        weights,
        w_o,
        geometry=g,
        kv_cache=kv,
        slot_mapping=slots,
        kv_layout="NHD",
    )
    torch.cuda.synchronize()
    c = cosine(out, ref.out)
    print(f"[prefill 27B 1x{seq}] vs oracle cos {c:.6f}")
    assert out.shape == h.shape and torch.isfinite(out.float()).all() and c > 0.999
    k_rows, v_rows = cache_rows(kv, "NHD", slots.to(torch.long), page)
    torch.testing.assert_close(
        k_rows.float(), ref.k.reshape(-1, H_KV, D).float(), rtol=2.0**-7, atol=1e-2
    )
    torch.testing.assert_close(
        v_rows.float(), ref.v.reshape(-1, H_KV, D).float(), rtol=2.0**-7, atol=1e-2
    )
    assert (kv[0] == 0).all(), "page 0 was never addressed and must stay untouched"
