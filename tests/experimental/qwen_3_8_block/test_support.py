# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU: the geometry (the Qwen 3.5 fields first, the indexer appended), the support probe (never raises; the three forms served
together), the version probe shared with the Qwen 3.5 package (one vendored tree, one manifest), the mirrored constants."""

import pytest
import torch


def _flash_next(**over):
    from flashinfer.qwen_3_8_block import Qwen38BlockGeometry

    kw = dict(d_model=2560, num_qo_heads=24, num_kv_heads=2)
    kw.update(over)
    return Qwen38BlockGeometry(**kw)


def test_geometry_validation_and_derived_facts():
    from flashinfer.experimental.qwen_3_8_block import support
    from flashinfer.qwen_3_8_block import Qwen38BlockGeometry

    g = _flash_next()
    assert g.group == 12 and g.n_qkvg_dense == 13312 and g.index_band_cols == 640
    assert g.n_qkvg == 13952 and g.scale == 0.0625 and g.unsupported_reason() is None
    assert g.resolved_index_list_form == "per_row" and g.decode_max_q_len == 16
    assert _flash_next(num_qo_heads=12, num_kv_heads=1).n_qkvg == 6656 + 640
    assert _flash_next(num_qo_heads=6, num_kv_heads=1).n_qkvg == 3584 + 640
    # the 3.5 rows
    with pytest.raises(ValueError, match="multiple of num_kv_heads"):
        Qwen38BlockGeometry(d_model=2560, num_qo_heads=30, num_kv_heads=4)
    with pytest.raises(ValueError, match="rope_dim"):
        _flash_next(rope_dim=65)
    with pytest.raises(ValueError, match="positive int"):
        _flash_next(top_k=0)
    assert "head_dim=256" in _flash_next(head_dim=128).unsupported_reason()
    assert "multiple of 256" in _flash_next(d_model=4000).unsupported_reason()
    assert "rope_dim=32" in _flash_next(rope_dim=32).unsupported_reason()
    # the indexer rows (mirrored from the vendored tree: test_admission.py pins them against it)
    assert "index_head_dim=64" in _flash_next(index_head_dim=64).unsupported_reason()
    assert "index_kv_heads=2" in _flash_next(index_kv_heads=2).unsupported_reason()
    assert "block_size=8" in _flash_next(block_size=8).unsupported_reason()
    assert "top_k=516" in _flash_next(top_k=516).unsupported_reason()
    assert "top_k=6" in _flash_next(top_k=6).unsupported_reason()
    assert _flash_next(top_k=4).unsupported_reason() is None
    # the compressed-key cache dtypes: bf16 and e4m3 served on every form (the 4c8e24355 re-vendor); another dtype declined by name
    e4 = _flash_next(index_cache_dtype=torch.float8_e4m3fn)
    assert (
        e4.unsupported_reason() is None
        and e4.resolved_index_cache_dtype == torch.float8_e4m3fn
    )
    assert _flash_next(index_cache_dtype=torch.bfloat16).unsupported_reason() is None
    assert _flash_next().resolved_index_cache_dtype == torch.bfloat16
    assert (
        "index_cache_dtype=torch.float16"
        in _flash_next(index_cache_dtype=torch.float16).unsupported_reason()
    )
    # the indexer writers' rounding convention: a frozen geometry field (part of every plan key), validated against the mirrored vocabulary
    assert (
        _flash_next().index_compress_rounding == "single"
        and _flash_next(index_compress_rounding="hf").unsupported_reason() is None
    )
    assert _flash_next(index_compress_rounding="hf") != _flash_next()
    with pytest.raises(ValueError, match="index_compress_rounding must be one of"):
        _flash_next(index_compress_rounding="bf16")
    assert (
        "selects per row" in _flash_next(index_list_form="shared").unsupported_reason()
    )
    assert _flash_next(index_list_form="per_row").unsupported_reason() is None
    with pytest.raises(ValueError, match="index_list_form"):
        _flash_next(index_list_form="rows")
    assert (
        support.next_pow2(4) == 4
        and support.next_pow2(5) == 8
        and support.next_pow2(1) == 1
    )
    assert _flash_next(index_heads=5).decode_max_q_len == 8
    assert _flash_next(index_heads=16).decode_max_q_len == 4


def test_support_probe_on_cpu_never_raises_and_names_the_device():
    from flashinfer.experimental.qwen_3_8_block import support
    from flashinfer.qwen_3_8_block import qwen_3_8_block_support

    g = _flash_next()
    sup = qwen_3_8_block_support(torch.device("cpu"), g)
    assert qwen_3_8_block_support.is_experimental is True
    assert sup.served is False and sup.forms == () and "SM107" in sup.reason
    assert sup.decode.reason == sup.reason and sup.decode.declines == {}
    # the prefill / extend forms are ONE vendored form with the decode step's: not served here for the same (device) reason, no table
    assert (
        sup.prefill.served is False
        and sup.prefill.reason == sup.reason
        and sup.prefill.declines == {}
    )
    assert (
        sup.extend.served is False
        and sup.extend.reason == sup.reason
        and sup.extend.declines == {}
    )
    # no FlashInfer-side rows bound on the extend forms since the 6b21be0e0 pin (the previous pin's field / constants are retired)
    assert not hasattr(sup, "extend_rows_max") and not hasattr(
        support, "EXTEND_ROWS_MAX"
    )
    assert support.DECODE_CAPTURE_ROWS_MAX == 128
    assert sup.page_sizes == (16, 32, 64, 128) and sup.kv_layouts == (
        "NHD",
        "HND",
        "UNIFIED",
    )
    assert sup.decode_max_q_len == 16 and sup.decode_capture_rows_max == 128
    assert sup.dtypes == (torch.bfloat16,) and sup.cutedsl_required == "4.8.0"
    assert sup.rope_dims == (64, 128)
    # the appended indexer fields are READ off the vendored envelope: empty where the tree was not read (a CPU probe)
    assert (
        sup.index_sources == ()
        and sup.index_page_sizes == ()
        and sup.kv_cache_dtypes == ()
    )
    assert sup.index_query_rows_max == 0 and sup.fuse_prepare_default is False
    assert (
        sup.sparse_tiles == () and sup.sparse_tile_lean_default is False
    )  # the lean-tile rows: READ off the envelope, empty on a CPU probe
    # the extend step's rows: READ off the envelope too, empty / 0 on a CPU probe
    assert sup.index_select_forms == () and sup.index_extend_ring_rows_min == 0
    assert (
        sup.index_extend_select_budget_bytes == 0 and sup.index_compress_roundings == ()
    )
    assert (
        "bfloat16"
        in qwen_3_8_block_support(torch.device("cpu"), g, dtype=torch.float16).reason
    )
    # the geometry gate speaks before the device: an indexer row is the reason on every device
    shared = qwen_3_8_block_support(
        torch.device("cpu"), _flash_next(index_list_form="shared")
    )
    assert not shared.served and "selects per row" in shared.reason
    e4m3 = qwen_3_8_block_support(
        torch.device("cpu"), _flash_next(index_cache_dtype=torch.float8_e4m3fn)
    )
    assert (
        not e4m3.served and "SM107" in e4m3.reason
    )  # the e4m3 cache passes the geometry gate: the device speaks
    f16c = qwen_3_8_block_support(
        torch.device("cpu"), _flash_next(index_cache_dtype=torch.float16)
    )
    assert not f16c.served and "index_cache_dtype=torch.float16" in f16c.reason
    assert (
        "head_dim=256"
        in qwen_3_8_block_support(torch.device("cpu"), _flash_next(head_dim=128)).reason
    )


def test_support_probe_on_the_local_device():
    """Wherever this runs: the probe answers; served exactly on SM107 with a Rubin-capable DSL; the decode record then carries the
    vendored envelope's indexer rows and decline table."""
    from flashinfer.qwen_3_8_block import qwen_3_8_block_support

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    g = _flash_next()
    sup = qwen_3_8_block_support(torch.device("cuda", 0), g)
    cc = torch.cuda.get_device_capability(0)
    assert sup.device_cc == cc
    if cc != (10, 7):
        assert not sup.served and f"SM{cc[0]}{cc[1]}" in sup.reason
    elif not sup.cutedsl_ok:
        assert not sup.served and "cutlass-dsl" in sup.reason
    else:
        assert sup.served and sup.forms == ("prefill", "extend", "decode")
        assert (
            sup.prefill.served
            and sup.extend.served
            and sup.prefill.declines == sup.extend.declines
        )
        assert sup.index_sources == ("caller", "indexer") and sup.index_list_forms == (
            "per_row",
        )
        assert sup.index_list_form_default == "per_row"
        assert sup.index_cache_dtypes == (
            torch.bfloat16,
            torch.float8_e4m3fn,
        ) and sup.kv_cache_dtypes == (torch.bfloat16,)
        # the extend step's rows off the envelope
        assert sup.index_select_forms == ("auto", "decode", "extend")
        assert sup.index_extend_ring_rows_min == 3 == sup.index_state_rows_extra
        assert sup.index_extend_select_budget_bytes == 512 << 20
        assert sup.index_compress_roundings == ("single", "hf")
        assert (
            sup.index_head_dims == (128,)
            and sup.index_kv_heads == (1,)
            and sup.index_block_sizes == (4,)
        )
        assert sup.index_top_k_range == (4, 512, 4) and sup.index_query_rows_max == 64
        assert sup.index_state_rows_extra == 3 and sup.index_page_sizes == (
            16,
            32,
            64,
            128,
        )
        assert sup.fuse_prepare_default is True and sup.pdl_default is False
        # the sparse core's two bodies on cc 10.7 and the AUTO pin consulting the lean rule there (the envelope's rows, mirrored by name)
        from flashinfer.experimental.qwen_3_8_block import support

        assert (
            sup.sparse_tiles
            == (support.SPARSE_TILE_TCGEN05, support.SPARSE_TILE_LEAN)
            == ("sparse_d256_f16", "sparse_lean_d256_f16")
        )
        assert sup.sparse_tile_lean_default is True
        for key in (
            "indexer_e4m3_pool",
            "indexer_shared_list",
            "indexer_query_rows",
            "indexer_page_size",
            "indexer_extend_fuse_prepare",
            "indexer_extend_ring_slots",
            "indexer_extend_rope_tables_only",
            "indexer_extend_ring_rows",
            "indexer_extend_chunk_rows",
        ):
            assert key in sup.decode.declines and key in sup.extend.declines, key
        assert (
            "indexer_e4m3_cache" not in sup.decode.declines
        )  # the e4m3 cache is served (the FlashInfer-side row retired)
        # ONE decline table for the three forms: no FlashInfer-side rows bound on the extend / prefill records since the 6b21be0e0 pin
        assert sup.extend.declines == sup.prefill.declines == sup.decode.declines
        assert "extend_proj_slab_rows" not in sup.extend.declines
        # the e4m3 compressed-key cache: served on the device too
        assert qwen_3_8_block_support(
            torch.device("cuda", 0), _flash_next(index_cache_dtype=torch.float8_e4m3fn)
        ).served
        # the e4m3 pool: declined by the vendored text through the probe's kv_cache_dtype
        fp8 = qwen_3_8_block_support(
            torch.device("cuda", 0), g, kv_cache_dtype=torch.float8_e4m3fn
        )
        assert (
            not fp8.served
            and fp8.decode.reason == sup.decode.declines["indexer_e4m3_pool"]
        )


def test_mirrored_sparse_tile_vocabulary():
    """The sparse core's body knob as this module mirrors it (``check_vendored_rules`` re-reads every value off the vendored rules module
    ``sdpa.fwd.sparse_tile_rules``): the vocabulary, the two bodies' kernel_template names, the AUTO placement bound in CTAs per chunk and
    its unit as the rule's reading over (batch, q_len) cells; the light gate refuses another spelling by name."""
    from flashinfer.experimental.qwen_3_8_block import api, support

    assert (
        support.SPARSE_TILE_FORMS == ("auto", "lean", "tcgen05")
        and support.SPARSE_TILE_FORM_AUTO == "auto"
    )
    assert (
        support.SPARSE_TILE_LEAN == "sparse_lean_d256_f16"
        and support.SPARSE_TILE_TCGEN05 == "sparse_d256_f16"
    )
    assert support.SPARSE_TILE_LEAN_MAX_UNITS == 4
    # the unit: True iff batch x 2 <= 4 at every step width (the 7301fad21 pin's B x H_kv x q_len count read (1, 3) / (1, 4) / (2, 4) False)
    assert support.SPARSE_TILE_LEAN_RULE_CELLS == (
        (1, 1, True),
        (1, 3, True),
        (1, 4, True),
        (2, 1, True),
        (2, 4, True),
        (3, 1, False),
        (4, 1, False),
        (4, 4, False),
    )
    g = _flash_next()
    for form in support.SPARSE_TILE_FORMS:
        api._light_decode_gates(g, 16, 1, None, None, form)
    with pytest.raises(ValueError, match="sparse_tile_form must be one of"):
        api._light_decode_gates(g, 16, 1, None, None, "fast")
    with pytest.raises(ValueError, match="sparse_tile_form must be one of"):
        api._light_decode_gates(g, 16, 1, None, None, None)
    with pytest.raises(ValueError, match="sparse_tile_form must be one of"):
        support.sparse_tile(
            torch.device("cpu"), g, batch_size=1, sparse_tile_form="fast"
        )


def test_mirrored_extend_vocabulary():
    """The extend step's rows as this module mirrors them (``check_vendored_rules`` re-reads every value off the vendored envelope and the
    tree's DSL-free constants): the step forms, the ring floor (one spelling with the ring's extra rows), the select-window budget, the
    rounding conventions; the FlashInfer-side rows bound and its reason; the served forms."""
    from flashinfer.experimental.qwen_3_8_block import api, support

    assert support.FORMS == ("prefill", "extend", "decode") == support.ALL_FORMS
    assert support.EXTEND_FORMS == ("prefill", "extend")
    assert support.INDEX_SELECT_FORMS == ("auto", "decode", "extend")
    assert (
        support.INDEX_SELECT_FORM_EXTEND == "extend"
        and support.INDEX_SELECT_FORM_AUTO == "auto"
    )
    assert support.INDEX_EXTEND_RING_ROWS_MIN == 3 == support.INDEX_STATE_ROWS_EXTRA
    assert support.INDEX_EXTEND_SELECT_BUDGET_BYTES == 512 << 20
    assert (
        support.INDEX_COMPRESS_ROUNDINGS == ("single", "hf")
        and support.INDEX_COMPRESS_ROUNDING_DEFAULT == "single"
    )
    assert set(support.DECLINE_KEYS) >= {
        "indexer_extend_fuse_prepare",
        "indexer_extend_ring_slots",
        "indexer_extend_rope_tables_only",
        "indexer_extend_ring_rows",
        "indexer_extend_chunk_rows",
    }
    g = _flash_next()
    # the rows bound of the previous pin is retired: no reason function, no rows argument on the light gates
    assert not hasattr(support, "extend_rows_reason") and not hasattr(
        support, "EXTEND_ROWS_KEY"
    )
    # the light gates of the extend forms: the page sizes, the pool dtype, the types -- before any vendored code
    api._light_extend_gates(g, "extend", 16, None, None, None, "auto")
    with pytest.raises(NotImplementedError, match="page_size must be one of"):
        api._light_extend_gates(g, "extend", 48, None, None, None, "auto")
    with pytest.raises(NotImplementedError, match="e4m3 pool"):
        api._light_extend_gates(
            g, "extend", 16, torch.float8_e4m3fn, None, None, "auto"
        )
    with pytest.raises(TypeError, match="use_pdl"):
        api._light_extend_gates(g, "extend", 16, None, 1, None, "auto")
    with pytest.raises(TypeError, match="fuse_prepare"):
        api._light_extend_gates(g, "extend", 16, None, None, 1, "auto")
    with pytest.raises(ValueError, match="sparse_tile_form must be one of"):
        api._light_extend_gates(g, "extend", 16, None, None, None, "fast")
    with pytest.raises(ValueError, match="exceed the block table's reach"):
        api._check_extend_chunk_rows("extend", 129, 8, 16)
    api._check_extend_chunk_rows("extend", 128, 8, 16)
    # the contract flags per form: None reads the form's default; an explicit value is kept; another type refused
    assert api._resolve_contract_flags("decode", None, None) == (True, False)
    assert api._resolve_contract_flags("extend", None, None) == (False, True)
    assert api._resolve_contract_flags("prefill", None, None) == (False, True)
    assert api._resolve_contract_flags("extend", True, False) == (True, False)
    with pytest.raises(TypeError, match="ring_slots must be None"):
        api._resolve_contract_flags("extend", None, 1)


def test_version_probe_is_the_qwen_3_5_tree():
    """ONE vendored tree for both packages: the 3.8 source record equals the 3.5 one field for field, and names the pinned commit."""
    from flashinfer.experimental.qwen_3_5_block import support as s35
    from flashinfer.experimental.qwen_3_8_block import support as s38
    from flashinfer.qwen_3_8_block import Qwen38BlockSource, qwen_3_8_block_support

    a, b = s38.source(), s35.source()
    assert Qwen38BlockSource is type(b) and a == b
    assert a.commit.startswith("6b21be0e0") and a.version == "1.33.0" and a.files == 161
    assert a.validated_cutedsl == "4.8.0" and a.validated_compute_capability == "10.7"
    assert qwen_3_8_block_support.is_experimental is True
