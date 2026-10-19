# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Admission consistency: what ``qwen_3_5_block_support`` advertises is what ``prepare`` admits.

The probe and ``prepare`` share one geometry gate (``Qwen35BlockGeometry.unsupported_reason``); these tests pin that gate against
the vendored tree's own checks -- the ones ``prepare`` runs through ``GatedAttentionBlockFwd.check_support`` for each form and row
bucket -- on a CPU host with the device capability mocked to Rubin: the real declarations on host tensors, the real stage checks,
no kernel compiled.  The SDPA stage's ``check_support`` binds a live device and is left out here; its geometry facts (``head_dim``,
the mask, the window) are the rows of the vendored envelope table, which the probe reads verbatim into ``declines``.  On the GPU:
a served geometry at each served ``rope_dim`` prepares and runs, and a declined ``rope_dim`` is a typed ``NotImplementedError``
before any vendored code is reached.
"""

from unittest import mock

import pytest
import torch

from tests.test_helpers.qwen_3_5_block_cases import PagedCase

FORMS = ("prefill", "extend", "decode")
# (d_model, num_qo_heads, num_kv_heads, rope_dim): the two Qwen3.5 shards, both served rope_dims, several GQA groups (16, 12,
# 8, 4, 3, 1 = MHA), the smallest projection tile, and the rope_dim values each stage-(1) rendering declines
GRID = [
    (2560, 24, 2, 64),
    (4096, 32, 2, 64),
    (4096, 32, 2, 128),
    (512, 4, 1, 128),
    (512, 8, 1, 64),
    (768, 6, 2, 64),
    (512, 4, 4, 64),
    (256, 1, 1, 64),
    (512, 8, 1, 16),  # both renderings decline
    (512, 4, 1, 32),  # the fused projection declines (whole 64-column subtile pairs)
    (
        512,
        4,
        1,
        192,
    ),  # the lowered decode chain declines (16 x a power of two) -- the fused projection takes it
    (512, 4, 1, 256),  # the fused projection declines (below the head dim)
]
# the declaration buckets of the three forms: dense prefill, the packed extend step, the DECODE form at decode row counts (the
# vendored declaration lowers stage (1) to the unfused chain there) and above them (the fused projection)
BUCKETS = (
    ("prefill", dict(batch=2, seq=64)),
    ("extend", dict(tokens=64)),
    ("decode", dict(batch=4, seq=2)),
    ("decode", dict(batch=129, seq=1)),
)


def _passes(check, *args) -> bool:
    try:
        check(*args)
    except (ValueError, NotImplementedError):
        return False
    return True


def _declare(form, geometry, **bucket):
    """The vendored declaration ``api.prepare`` builds for ``form`` (host tensors; no device, nothing compiled)."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gated_attention_block import (
        GatedAttentionBlockFwd as Fwd,
    )

    g = geometry.frost()
    dt, d, dm, rope, hq, hkv, page = (
        torch.bfloat16,
        geometry.head_dim,
        geometry.d_model,
        geometry.rope_dim,
        geometry.num_qo_heads,
        geometry.num_kv_heads,
        16,
    )

    def mk(*shape):
        return torch.empty(*shape, dtype=dt)

    w = (
        mk(g.n_qkvg, dm),
        torch.ones(d, dtype=dt),
        torch.ones(d, dtype=dt),
        mk(dm, hq * d),
    )
    pools = (
        torch.zeros(9, hkv, page, d, dtype=dt),
        torch.zeros(9, hkv, page, d, dtype=dt),
    )
    knobs = dict(fuse_norm_rope=True, fuse_gate=True)  # what api.prepare declares
    if form == "prefill":
        b, s = bucket["batch"], bucket["seq"]
        return Fwd(
            mk(b, s, dm),
            *w[:3],
            mk(b, s, rope),
            mk(b, s, rope),
            w[3],
            mk(b, s, dm),
            g,
            paged_kv_page_size=page,
            **knobs,
        )
    if form == "extend":
        t = bucket["tokens"]
        return Fwd(
            mk(t, dm),
            *w[:3],
            mk(t, rope),
            mk(t, rope),
            w[3],
            mk(t, dm),
            g,
            thd=True,
            num_sequences=2,
            max_seq_len=t,
            cu_seqlens=True,
            token_capacity=t,
            paged_kv_page_size=page,
            kv_source="paged",
            sample_k_cache=pools[0],
            sample_v_cache=pools[1],
            sample_block_table=torch.zeros(2, 4, dtype=torch.int32),
            **knobs,
        )
    b, s = bucket["batch"], bucket["seq"]
    return Fwd(
        mk(b, s, dm),
        *w[:3],
        mk(b, s, rope),
        mk(b, s, rope),
        w[3],
        mk(b, s, dm),
        g,
        paged_kv_page_size=page,
        kv_source="paged",
        sample_k_cache=pools[0],
        sample_v_cache=pools[1],
        sample_block_table=torch.zeros(b, 4, dtype=torch.int32),
        **knobs,
    )


def _rubin_mocks():
    """Device-capability discovery mocked to Rubin: torch's probes and the vendored tree's own (``frost.device``, what the norm +
    RoPE stage reads to pick its kernel).  Nothing else is mocked -- the declarations and the stage checks are the real ones."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost import (
        device as frost_device,
    )

    return (
        mock.patch("torch.cuda.get_device_capability", return_value=(10, 7)),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch.object(frost_device, "ambient_device", return_value=0),
        mock.patch.object(frost_device, "compute_capability", return_value=(10, 7)),
    )


def _stage_checks_pass(blk):
    """``check_support`` as ``prepare`` runs it, minus the SDPA stage (it binds a live device): the declaration contract and
    every other stage's own check.  Returns ``(admitted, the first typed decline or None)``."""
    try:
        blk._check_declaration()
        for st in blk._stages:
            if type(st).__name__ == "_Sdpa":
                continue
            st.check_support()
    except (ValueError, NotImplementedError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


@pytest.mark.parametrize(
    "num_qo_heads, num_kv_heads", [(32, 2), (24, 2), (8, 1), (6, 2), (4, 4)]
)
def test_rope_gate_is_the_vendored_stage_rules(num_qo_heads, num_kv_heads):
    """Every even rope_dim up to the head dim: the gate admits it exactly when BOTH stage-(1) renderings' real checks do -- the
    fused norm + RoPE projection fork (every form above decode row counts) and the norm + RoPE stage of the lowered decode chain --
    and declines it with a reason naming the value."""
    pytest.importorskip("cutlass")
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry
    from flashinfer.experimental.qwen_3_5_block import support
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gated_attention_block import (
        api as vendored,
    )

    p1, p2, p3, p4 = _rubin_mocks()
    served = []
    with p1, p2, p3, p4:
        for rope in range(2, support.HEAD_DIM + 1, 2):
            g = Qwen35BlockGeometry(
                d_model=512,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                rope_dim=rope,
            )
            fg = g.frost()
            fork_ok = _passes(
                vendored._FusedQkvProjection(
                    fg, batch=1, seq_len=1, dtype=torch.bfloat16, want_rstd=False
                ).check_support
            )
            chain_ok = _passes(
                vendored._QkNormRope(
                    fg, batch=1, seq_len=1, dtype=torch.bfloat16, want_rstd=False
                ).check_support
            )
            reason = g.unsupported_reason()
            assert (reason is None) == (fork_ok and chain_ok), (
                rope,
                fork_ok,
                chain_ok,
                reason,
            )
            if reason is None:
                served.append(rope)
            else:
                assert f"rope_dim={rope}" in reason
    assert tuple(served) == support.ROPE_DIMS == (64, 128)


def test_vendored_rule_tripwire_passes_and_fires_on_a_drift():
    """The mirrored constants are re-read from the vendored tree (the probe's vendored read, ``prepare``); a moved rule is a loud
    ``RuntimeError`` naming the fact, never a false verdict."""
    pytest.importorskip("cutlass")
    from flashinfer.experimental.qwen_3_5_block import support

    support.check_vendored_rules.cache_clear()
    facts = support.check_vendored_rules()
    assert facts["head_dim"] == 256 and facts["rope_subtile_pair"] == 64
    assert facts["rope_access_elems"] == 8 and facts["rope_row_lanes"] == 32
    assert facts["decode_max_rows"] == 32 and facts["page_sizes"] == (16, 32, 64, 128)
    keep = support._ROPE_SUBTILE_PAIR
    try:
        support._ROPE_SUBTILE_PAIR = 32
        support.check_vendored_rules.cache_clear()
        with pytest.raises(RuntimeError, match="rope_subtile_pair"):
            support.check_vendored_rules()
    finally:
        support._ROPE_SUBTILE_PAIR = keep
        support.check_vendored_rules.cache_clear()
    assert support.check_vendored_rules()["rope_subtile_pair"] == 64


@pytest.mark.parametrize("d_model, num_qo_heads, num_kv_heads, rope_dim", GRID)
def test_probe_verdict_matches_the_declaration_checks(
    d_model, num_qo_heads, num_kv_heads, rope_dim
):
    """``qwen_3_5_block_support(...).served`` (capability mocked to Rubin) is True exactly when every form's declaration passes
    ``prepare``'s checks in every row bucket -- the verdict is per GEOMETRY, the dispatch predicate's granularity, so a geometry one
    bucket declines (rope_dim 192: the lowered decode chain) is declined for every form, with one reason."""
    pytest.importorskip("cutlass")
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry, qwen_3_5_block_support

    g = Qwen35BlockGeometry(
        d_model=d_model,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        rope_dim=rope_dim,
    )
    declared = [(form, _declare(form, g, **bucket), bucket) for form, bucket in BUCKETS]
    p1, p2, p3, p4 = _rubin_mocks()
    with p1, p2, p3, p4:
        sup = qwen_3_5_block_support(torch.device("cuda", 0), g)
        if not sup.cutedsl_ok:
            from flashinfer.experimental.qwen_3_5_block import support

            pytest.skip(
                f"the installed CuTe DSL does not serve the block: {support.cutedsl_reason()}"
            )
        verdicts = {
            (form, tuple(sorted(bucket.items()))): _stage_checks_pass(blk)
            for form, blk, bucket in declared
        }
    admitted = all(ok for ok, _ in verdicts.values())
    assert sup.served == admitted, (sup.reason, verdicts)
    assert all(sup.form(f).served == sup.served for f in FORMS), sup
    if sup.served:
        assert sup.forms == FORMS and sup.reason is None
        assert sup.extend.declines and sup.decode.declines
    else:
        assert f"rope_dim={rope_dim}" in sup.reason, sup.reason
        assert all(sup.form(f).reason == sup.reason for f in FORMS)
        # the declines the vendored stages raise for this geometry are typed too, and name the same field (the fused projection
        # and the TMA norm + RoPE kernel spell the value, the LDG kernel its half-lane count)
        assert any(
            why is not None and "rope_dim" in why for _, why in verdicts.values()
        ), verdicts


def test_probe_declines_a_d_model_off_the_projection_tile_up_front():
    """``d_model % 256`` is the adapter's own rule (the capture-safe projection route runs the forced 256-wide tile only): declined
    by the gate on every device, named, before any vendored code; the vendored stages read it only at compile."""
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry, qwen_3_5_block_support

    for dm in (4000, 384):
        sup = qwen_3_5_block_support(
            torch.device("cpu"),
            Qwen35BlockGeometry(d_model=dm, num_qo_heads=32, num_kv_heads=2),
        )
        assert not sup.served and f"d_model={dm}" in sup.reason and "256" in sup.reason
    with pytest.raises(ValueError, match="rope_dim"):
        Qwen35BlockGeometry(d_model=4096, num_qo_heads=32, num_kv_heads=2, rope_dim=65)


# --------------------------------------------------------------------------- on the GPU


@pytest.mark.arch_rubin
@pytest.mark.usefixtures("rubin")
@pytest.mark.parametrize("rope_dim", [64, 128])
def test_served_rope_dims_prepare_and_run(rope_dim):
    """Each served rope_dim at the 24 / 2 shard: the probe advertises it, a decode step at decode row counts (the lowered chain's
    norm + RoPE kernel) and an extend step above them (the fused projection) prepare and match the oracle."""
    from flashinfer.qwen_3_5_block import (
        Qwen35BlockGeometry,
        prepare_qwen_3_5_block_bf16,
        qwen_3_5_block_support,
    )

    g = Qwen35BlockGeometry(
        d_model=2560, num_qo_heads=24, num_kv_heads=2, rope_dim=rope_dim
    )
    sup = qwen_3_5_block_support(torch.device("cuda", 0), g)
    assert sup.served and sup.forms == FORMS and rope_dim in sup.rope_dims, sup.reason
    dev = torch.device("cuda", 0)
    # decode: B 2 x q_len 1 (the lowered small-M chain)
    case = PagedCase(
        g, (100, 37), (1, 1), page_size=16, layout="NHD", seed=11 + rope_dim
    )
    plan = prepare_qwen_3_5_block_bf16(
        "decode",
        geometry=g,
        device=dev,
        weights=case.weights,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        page_size=16,
        batch_size=case.b,
        block_table=case.block_table,
    )
    assert plan.declaration["fuse_norm_rope_effective"] is False
    out = plan.run(
        case.hidden_states,
        case.rope_cos,
        case.rope_sin,
        case.weights,
        case.w_o,
        **case.decode_args(),
    )
    torch.cuda.synchronize()
    case.check_out(out, label=f"admission decode rope {rope_dim}")
    case.check_written_rows(label=f"admission decode rope {rope_dim}")
    # extend above decode row counts: the fused norm + RoPE projection (every sequence from position 0)
    lens = (130, 70)
    case = PagedCase(g, lens, lens, page_size=16, layout="NHD", seed=23 + rope_dim)
    plan = prepare_qwen_3_5_block_bf16(
        "extend",
        geometry=g,
        device=dev,
        weights=case.weights,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        page_size=16,
        batch_size=case.b,
        token_capacity=case.t,
        block_table=case.block_table,
    )
    assert plan.declaration["fuse_norm_rope_effective"] is True
    out = plan.run(
        case.hidden_states,
        case.rope_cos,
        case.rope_sin,
        case.weights,
        case.w_o,
        **case.extend_args(),
    )
    torch.cuda.synchronize()
    case.check_out(out, label=f"admission extend rope {rope_dim}")
    case.check_written_rows(label=f"admission extend rope {rope_dim}")


@pytest.mark.arch_rubin
@pytest.mark.usefixtures("rubin")
@pytest.mark.parametrize("rope_dim", [32, 192, 256])
def test_declined_rope_dim_is_typed_before_any_vendored_code(rope_dim, monkeypatch):
    """A rope_dim the gate declines: the probe says so for every form, and every form's ``prepare`` raises the typed
    ``NotImplementedError`` naming it before the vendored tree is touched -- no declaration, no compile."""
    from flashinfer.qwen_3_5_block import (
        Qwen35BlockGeometry,
        prepare_qwen_3_5_block_bf16,
        qwen_3_5_block_support,
    )
    from flashinfer.experimental.qwen_3_5_block import api as impl
    from tests.test_helpers.qwen_3_5_block_cases import new_kv_cache, random_weights

    g = Qwen35BlockGeometry(
        d_model=2560, num_qo_heads=24, num_kv_heads=2, rope_dim=rope_dim
    )
    sup = qwen_3_5_block_support(torch.device("cuda", 0), g)
    assert not sup.served and sup.forms == () and f"rope_dim={rope_dim}" in sup.reason
    assert all(
        not sup.form(f).served and sup.form(f).reason == sup.reason for f in FORMS
    )

    def reached():
        raise AssertionError(
            "the vendored tree was reached for a geometry the gate declines"
        )

    monkeypatch.setattr(impl, "_frost_block", reached)
    weights, w_o = random_weights(g, device="cuda", seed=3)
    kv = new_kv_cache(8, g.num_kv_heads, 16, g.head_dim, "NHD", device="cuda")
    common = dict(
        geometry=g,
        device=torch.device("cuda", 0),
        weights=weights,
        w_o=w_o,
        kv_cache=kv,
        page_size=16,
        batch_size=2,
    )
    for form, extra in (
        ("prefill", dict(seq_len=8)),
        ("extend", dict(token_capacity=16, max_pages=4)),
        ("decode", dict(max_pages=4)),
    ):
        with pytest.raises(NotImplementedError, match=f"rope_dim={rope_dim}") as info:
            prepare_qwen_3_5_block_bf16(form, **common, **extra)
        assert str(info.value).startswith(f"qwen_3_5_block ({form}):")
