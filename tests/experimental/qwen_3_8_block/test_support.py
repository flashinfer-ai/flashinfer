# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU: the geometry (the Qwen 3.5 fields first, the indexer appended), the support probe (never raises; the decode form only), the
version probe shared with the Qwen 3.5 package (one vendored tree, one manifest), the mirrored constants."""

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
    assert (
        "e4m3"
        in _flash_next(index_cache_dtype=torch.float8_e4m3fn).unsupported_reason()
    )
    assert _flash_next(index_cache_dtype=torch.bfloat16).unsupported_reason() is None
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
    from flashinfer.qwen_3_8_block import qwen_3_8_block_support

    g = _flash_next()
    sup = qwen_3_8_block_support(torch.device("cpu"), g)
    assert qwen_3_8_block_support.is_experimental is True
    assert sup.served is False and sup.forms == () and "SM107" in sup.reason
    assert sup.decode.reason == sup.reason and sup.decode.declines == {}
    # the prefill / extend forms are never served by this package: the follow-up named
    assert (
        sup.prefill.served is False and "later cuDNN Frontend PR" in sup.prefill.reason
    )
    assert sup.extend.served is False and sup.extend.reason == sup.prefill.reason
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
    assert not e4m3.served and "e4m3" in e4m3.reason
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
        assert sup.served and sup.forms == ("decode",)
        assert sup.index_sources == ("caller", "indexer") and sup.index_list_forms == (
            "per_row",
        )
        assert sup.index_list_form_default == "per_row"
        assert sup.index_cache_dtypes == (torch.bfloat16,) and sup.kv_cache_dtypes == (
            torch.bfloat16,
        )
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
        for key in (
            "indexer_e4m3_cache",
            "indexer_e4m3_pool",
            "indexer_shared_list",
            "indexer_query_rows",
            "indexer_page_size",
        ):
            assert key in sup.decode.declines, key
        # the e4m3 pool: declined by the vendored text through the probe's kv_cache_dtype
        fp8 = qwen_3_8_block_support(
            torch.device("cuda", 0), g, kv_cache_dtype=torch.float8_e4m3fn
        )
        assert (
            not fp8.served
            and fp8.decode.reason == sup.decode.declines["indexer_e4m3_pool"]
        )


def test_version_probe_is_the_qwen_3_5_tree():
    """ONE vendored tree for both packages: the 3.8 source record equals the 3.5 one field for field, and names the pinned commit."""
    from flashinfer.experimental.qwen_3_5_block import support as s35
    from flashinfer.experimental.qwen_3_8_block import support as s38
    from flashinfer.qwen_3_8_block import Qwen38BlockSource, qwen_3_8_block_support

    a, b = s38.source(), s35.source()
    assert Qwen38BlockSource is type(b) and a == b
    assert a.commit.startswith("f2aa360e5") and a.version == "1.32.0" and a.files == 114
    assert a.validated_cutedsl == "4.8.0" and a.validated_compute_capability == "10.7"
    assert qwen_3_8_block_support.is_experimental is True
