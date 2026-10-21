# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Admission consistency: what ``qwen_3_8_block_support`` advertises is what ``prepare`` admits.

The probe and ``prepare`` share one geometry gate (``Qwen38BlockGeometry.unsupported_reason``) whose indexer rows are MIRRORED from the
vendored tree; these tests pin the mirror against the tree (``check_vendored_rules`` passes and FIRES on a planted drift of every
mirrored constant), the probe's verdict against the vendored declaration's checks on a grid of geometries x step widths x list forms x
cache dtypes x page sizes (the device capability mocked to Rubin: the real declarations on host tensors, the real stage checks, no kernel
compiled), the Errors table's keys against ``decode.declines``, and -- on the GPU -- that a declined geometry is typed BEFORE any vendored
code is reached."""

import contextlib
from unittest import mock

import pytest
import torch

GRID = [
    # (geometry overrides, q_len, page): served
    (dict(), 1, 16, True),
    (dict(), 4, 16, True),
    (dict(), 16, 16, True),
    (dict(), 17, 16, False),  # the scorer's row envelope: 17 x 4 = 68 > 64
    (dict(index_heads=5), 8, 16, True),  # 8 x next_pow2(5) = 64
    (dict(index_heads=5), 9, 16, False),
    (dict(), 1, 32, True),
    (dict(), 1, 64, True),
    (dict(), 1, 128, True),
    (
        dict(),
        1,
        48,
        False,
    ),  # a page off the powers of two: legal for the block, not for the paged scorer's shift
    (dict(index_list_form="per_row"), 4, 16, True),
    (dict(index_list_form="shared"), 1, 16, False),
    (dict(index_cache_dtype=torch.float8_e4m3fn), 1, 16, False),
    (dict(index_cache_dtype=torch.bfloat16), 1, 16, True),
    # the TP 2 / TP 4 shards: the declaration admits them, the probe declines them by name -- their slabs (N 7296 / 4224) are off
    # the capture-safe small-M projection route (`proj_slab_route`, decided at the part's SM count; mocked to 204 below)
    (dict(num_qo_heads=12, num_kv_heads=1), 1, 16, False),
    (dict(num_qo_heads=6, num_kv_heads=1), 4, 16, False),
    (dict(top_k=256), 1, 16, True),
    (dict(top_k=516), 1, 16, False),
    (dict(rope_dim=128), 1, 16, True),
    (dict(rope_dim=32), 1, 16, False),
]


def _geometry(**over):
    from flashinfer.qwen_3_8_block import Qwen38BlockGeometry

    kw = dict(d_model=2560, num_qo_heads=24, num_kv_heads=2)
    kw.update(over)
    return Qwen38BlockGeometry(**kw)


def _declare(geometry, q_len, page):
    """The vendored declaration ``api.prepare`` builds for the decode form (host tensors; no device, nothing compiled): the two side
    caches, the ring table, the position-table arm, the fused prepare."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gated_attention_block import (
        GatedAttentionBlockFwd as Fwd,
    )

    g = geometry.frost()
    dt, d, dm, hq, hkv = (
        torch.bfloat16,
        geometry.head_dim,
        geometry.d_model,
        geometry.num_qo_heads,
        geometry.num_kv_heads,
    )
    mk = lambda *shape: torch.empty(*shape, dtype=dt)  # noqa: E731
    b = 2
    cap = max(8, q_len + 3)
    rows = max(page // geometry.block_size, 1)
    return Fwd(
        mk(b, q_len, dm),
        mk(g.n_qkvg, dm),
        torch.ones(d, dtype=dt),
        torch.ones(d, dtype=dt),
        None,
        None,
        mk(dm, hq * d),
        mk(b, q_len, dm),
        g,
        paged_kv_page_size=page,
        kv_source="paged",
        sample_k_cache=torch.zeros(9, hkv, page, d, dtype=dt),
        sample_v_cache=torch.zeros(9, hkv, page, d, dtype=dt),
        sample_block_table=torch.zeros(b, 4, dtype=torch.int32),
        fuse_norm_rope=False,
        fuse_gate=True,
        fuse_prepare=True,
        sample_index_kv_cache=torch.zeros(
            9, rows, 1, geometry.index_head_dim, dtype=dt
        ),
        sample_index_state_cache=torch.zeros(
            b + 1, cap, 1, geometry.index_head_dim, dtype=dt
        ),
        sample_index_state_block_table=torch.zeros(b, 1, dtype=torch.int32),
        rope_position_tables_only=True,
    )


def _rubin_mocks():
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost import (
        device as frost_device,
    )

    return (
        mock.patch("torch.cuda.get_device_capability", return_value=(10, 7)),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch.object(frost_device, "ambient_device", return_value=0),
        mock.patch.object(frost_device, "compute_capability", return_value=(10, 7)),
        # the projection route gate asks the part's SM count: a 204-SM Rubin part (the 24 / 2 slab on the route, the shards off it)
        mock.patch(
            "flashinfer.experimental.qwen_3_8_block.support._sm_count", return_value=204
        ),
    )


@contextlib.contextmanager
def _rubin():
    with contextlib.ExitStack() as stack:
        for patch in _rubin_mocks():
            stack.enter_context(patch)
        yield


def test_vendored_rule_tripwire_passes_and_fires_on_a_drift_of_every_mirrored_constant():
    pytest.importorskip("cutlass")
    from flashinfer.experimental.qwen_3_8_block import support

    support.check_vendored_rules.cache_clear()
    facts = support.check_vendored_rules()
    assert (
        facts["index_head_dims"] == (128,)
        and facts["index_kv_heads"] == (1,)
        and facts["index_block_sizes"] == (4,)
    )
    assert (
        facts["index_top_k_range"] == (4, 512, 4)
        and facts["index_query_rows_max"] == 64
        and facts["index_state_rows_extra"] == 3
    )
    assert (
        facts["index_page_sizes"] == (16, 32, 64, 128)
        and facts["decode_capture_rows_max"] == 128
    )
    plants = [
        ("INDEX_HEAD_DIM", 64, "index_head_dims"),
        ("INDEX_KV_HEADS", (2,), "index_kv_heads"),
        ("INDEX_BLOCK_SIZES", (8,), "index_block_sizes"),
        ("INDEX_TOP_K_RANGE", (4, 1024, 4), "index_top_k_range"),
        ("INDEX_QUERY_ROWS_MAX", 128, "index_query_rows_max"),
        ("INDEX_STATE_ROWS_EXTRA", 7, "index_state_rows_extra"),
        ("INDEX_PAGE_SIZES", (16, 32), "index_page_sizes"),
        ("DECODE_CAPTURE_ROWS_MAX", 256, "decode_capture_rows_max"),
        (
            "DECLINE_KEYS",
            support.DECLINE_KEYS + ("indexer_moved_row",),
            "indexer_moved_row",
        ),
    ]
    for name, planted, token in plants:
        keep = getattr(support, name)
        try:
            setattr(support, name, planted)
            support.check_vendored_rules.cache_clear()
            with pytest.raises(RuntimeError, match=token):
                support.check_vendored_rules()
        finally:
            setattr(support, name, keep)
            support.check_vendored_rules.cache_clear()
    assert support.check_vendored_rules()["index_query_rows_max"] == 64


@pytest.mark.parametrize("over, q_len, page, served", GRID)
def test_probe_verdict_matches_the_vendored_declaration(over, q_len, page, served):
    """On the grid: the probe (capability mocked to Rubin) AND the vendored declaration + ``_check_declaration`` + the DSL-free stage checks
    agree on the verdict; a declined cell names its reason through the one gate or the envelope's decline row."""
    pytest.importorskip("cutlass")
    from flashinfer.experimental.qwen_3_8_block import api as impl
    from flashinfer.qwen_3_8_block import qwen_3_8_block_support

    g = _geometry(**over)
    # the light gate of prepare (what runs before any vendored code)
    light_ok, why = True, None
    try:
        impl._light_decode_gates(g, page, q_len, None, None)
    except NotImplementedError as exc:
        light_ok, why = False, str(exc)
    # the vendored declaration on host tensors (outside the mocks: the vendored device probe short-circuits on a CPU host); the typed
    # declines of the decode indexer are raised by the CONSTRUCTOR (the envelope's rows), then the declaration contract
    decl_ok = True
    if g.unsupported_reason() is None:
        try:
            blk = _declare(g, q_len, page)
            blk._check_declaration()
        except (ValueError, NotImplementedError) as exc:
            decl_ok, why = False, f"{type(exc).__name__}: {exc}"
    with _rubin():
        sup = qwen_3_8_block_support(torch.device("cuda", 0), g)
    if not sup.cutedsl_ok:
        from flashinfer.experimental.qwen_3_8_block import support

        pytest.skip(
            f"the installed CuTe DSL does not serve the block: {support.cutedsl_reason()}"
        )
    geometry_ok = sup.served
    if served:
        assert geometry_ok and light_ok and decl_ok, (sup.reason, why)
    else:
        assert not (geometry_ok and light_ok and decl_ok), (sup.reason, why)
        assert (sup.reason is not None) or (why is not None), (
            "a declined cell names its reason somewhere"
        )


def test_every_errors_table_key_is_a_decline_row_and_the_texts_are_the_envelope_s():
    """The README's Errors table keys are present in ``decode.declines`` (the tripwire checks them too), and the probe hands the FE texts
    through verbatim: the one spelling for the probe and the plan-time refusal."""
    pytest.importorskip("cutlass")
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gated_attention_block import (
        GatedAttentionBlockFwd,
    )
    from flashinfer.experimental.qwen_3_8_block import support
    from flashinfer.qwen_3_8_block import qwen_3_8_block_support

    g = _geometry()
    with _rubin():
        sup = qwen_3_8_block_support(torch.device("cuda", 0), g)
        if not sup.cutedsl_ok:
            pytest.skip(support.cutedsl_reason())
        env = GatedAttentionBlockFwd.serving_envelope(
            g.frost(), torch.bfloat16, torch.device("cuda", 0), device_cc=(10, 7)
        ).decode
    assert sup.served and sup.forms == ("decode",)
    for key in support.DECLINE_KEYS:
        assert (
            key in sup.decode.declines and sup.decode.declines[key] == env.declines[key]
        ), key
    assert "selects per row" in sup.decode.declines["indexer_shared_list"]
    assert "e4m3 read arm" in sup.decode.declines["indexer_e4m3_cache"]
    assert "in-kernel dequant arm" in sup.decode.declines["indexer_e4m3_pool"]
    assert (
        sup.index_query_rows_max == env.index_query_rows_max == 64
        and sup.fuse_prepare_default == env.fuse_prepare_default
    )
    # the FlashInfer-side row: absent at TP 1, present (and the verdict) for a shard whose slab is off the projection route
    assert support.PROJ_SLAB_ROUTE_KEY not in sup.decode.declines
    with _rubin():
        shard = qwen_3_8_block_support(
            torch.device("cuda", 0), _geometry(num_qo_heads=12, num_kv_heads=1)
        )
    assert (
        not shard.served
        and shard.reason == shard.decode.declines[support.PROJ_SLAB_ROUTE_KEY]
    )
    assert (
        "N=7296" in shard.reason
        and "204-SM" in shard.reason
        and "small-M projection route" in shard.reason
    )


def test_probe_declines_up_front_on_every_device():
    """The gate's own rows (the indexer geometry, the list form, the cache dtype, d_model, rope_dim) are declined on any device, named,
    before any vendored code; the prefill / extend forms are refused by the plan-time gate with the follow-up named."""
    from flashinfer.experimental.qwen_3_8_block import support
    from flashinfer.qwen_3_8_block import qwen_3_8_block_support

    for over, token in (
        (dict(index_list_form="shared"), "selects per row"),
        (dict(index_cache_dtype=torch.float8_e4m3fn), "e4m3"),
        (dict(index_head_dim=64), "index_head_dim=64"),
        (dict(block_size=8), "block_size=8"),
        (dict(top_k=6), "top_k=6"),
        (dict(d_model=4000), "d_model=4000"),
        (dict(rope_dim=32), "rope_dim=32"),
    ):
        sup = qwen_3_8_block_support(torch.device("cpu"), _geometry(**over))
        assert not sup.served and token in sup.reason, (over, sup.reason)
        assert sup.decode.reason == sup.reason
    for form in ("prefill", "extend"):
        with pytest.raises(NotImplementedError, match="later cuDNN Frontend PR"):
            support.require_served(
                torch.device("cpu"), _geometry(), torch.bfloat16, form
            )


# --------------------------------------------------------------------------- on the GPU


@pytest.mark.arch_rubin
@pytest.mark.usefixtures("rubin")
def test_declined_geometry_is_typed_before_any_vendored_code(monkeypatch):
    """A geometry the gate declines (the shared list form; the e4m3 indexer cache): the probe says so, and ``prepare`` raises the typed
    ``NotImplementedError`` naming it before the vendored tree is touched -- no declaration, no compile; likewise a step wider than the
    scorer's envelope and a page the scorer does not address (the light gates)."""
    from flashinfer.experimental.qwen_3_8_block import api as impl
    from flashinfer.qwen_3_8_block import (
        prepare_qwen_3_8_block_bf16,
        qwen_3_8_block_support,
    )
    from tests.test_helpers.qwen_3_8_block_cases import Qsa38PagedCase

    def reached():
        raise AssertionError(
            "the vendored tree was reached for a geometry the gate declines"
        )

    monkeypatch.setattr(impl, "_frost_block", reached)
    case = Qsa38PagedCase(_geometry(), (64,), 1, page_size=16, seed=5)
    for over, token in (
        (dict(index_list_form="shared"), "selects per row"),
        (dict(index_cache_dtype=torch.float8_e4m3fn), "e4m3"),
    ):
        g = _geometry(**over)
        sup = qwen_3_8_block_support(torch.device("cuda", 0), g)
        assert not sup.served and token in sup.reason
        with pytest.raises(NotImplementedError, match=token) as info:
            prepare_qwen_3_8_block_bf16(
                "decode",
                geometry=g,
                device="cuda",
                weights=case.weights38,
                w_o=case.w_o,
                kv_cache=case.kv_cache,
                page_size=16,
                batch_size=1,
                block_table=case.block_table,
                **case.plan_args(),
            )
        assert str(info.value).startswith("qwen_3_8_block (decode):")
    common = dict(
        geometry=_geometry(),
        device="cuda",
        weights=case.weights38,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        batch_size=1,
        block_table=case.block_table,
    )
    with pytest.raises(NotImplementedError, match="query rows per sequence"):
        prepare_qwen_3_8_block_bf16(
            "decode", page_size=16, q_len_per_req=17, **common, **case.plan_args()
        )
    with pytest.raises(NotImplementedError, match="page_size must be one of"):
        prepare_qwen_3_8_block_bf16(
            "decode", page_size=48, **common, **case.plan_args()
        )
    with pytest.raises(NotImplementedError, match="e4m3"):
        prepare_qwen_3_8_block_bf16(
            "decode",
            page_size=16,
            kv_cache_dtype=torch.float8_e4m3fn,
            **common,
            **case.plan_args(),
        )
    for form in ("prefill", "extend"):
        with pytest.raises(NotImplementedError, match="later cuDNN Frontend PR"):
            prepare_qwen_3_8_block_bf16(
                form, page_size=16, **common, **case.plan_args()
            )
