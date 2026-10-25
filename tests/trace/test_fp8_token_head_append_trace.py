"""Regression tests for layout-specific FP8 token/head cache traces."""

import pytest
import torch

from flashinfer.trace.templates.page import (
    fp8_token_head_quantize_append_paged_kv_cache_with_slot_mapping_trace as dispatch,
)


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp8_token_head_trace_layout_roundtrip(kv_layout, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    template = dispatch(kv_layout=kv_layout)
    init_kwargs = dict(
        nnz_kv=4,
        num_pages=4,
        num_kv_heads=8,
        head_dim=128,
        page_size=16,
        device=device,
    )
    inputs = template.init(**init_kwargs)
    assert inputs["kv_layout"] == kv_layout
    trace = template.build_fi_trace_fn(
        "flashinfer.page.fp8_token_head_quantize_append_paged_kv_cache_with_slot_mapping"
    )
    if device == "cuda":
        from flashinfer import (
            fp8_token_head_quantize_append_paged_kv_cache_with_slot_mapping as op,
        )

        assert op.fi_trace(**inputs) == trace(**inputs)
        assert op.fi_init(kv_layout=kv_layout, **init_kwargs)["kv_layout"] == kv_layout
    definition = trace(**inputs)
    assert definition["axes"]["page_size"]["value"] == 16
    assert definition["axes"]["num_kv_heads"]["value"] == 8
    cache_dims = (
        ["num_pages", "page_size", "num_kv_heads"]
        if kv_layout == "NHD"
        else ["num_pages", "num_kv_heads", "page_size"]
    )
    for section in ("inputs", "outputs"):
        for name in ("paged_k_cache", "paged_v_cache"):
            assert definition[section][name]["shape"] == cache_dims + ["slot_last"]
        for name in ("k_scale_cache", "v_scale_cache"):
            assert definition[section][name]["shape"] == cache_dims
    assert f"kv_layout == {kv_layout!r}" in definition["constraints"]

    # Execute the exported init in a fresh namespace: HND must survive without
    # the live module's globals or an explicit layout argument from the caller.
    namespace = {}
    exec(definition["init"], namespace)
    restored = namespace[template.init.__name__](**init_kwargs)
    assert restored["kv_layout"] == kv_layout
    assert trace(**restored) == definition
    for slots, scales in zip(
        restored["paged_kv_cache"], restored["kv_cache_sf"], strict=True
    ):
        assert scales.untyped_storage().data_ptr() == slots.untyped_storage().data_ptr()
        assert scales.storage_offset() * 4 == 128
        assert slots.shape == (
            (4, 16, 8, 144) if kv_layout == "NHD" else (4, 8, 16, 144)
        )


def test_fp8_token_head_trace_default_and_names():
    nhd = dispatch()
    hnd = dispatch(kv_layout="HND")
    assert nhd is dispatch(kv_layout="NHD")
    assert nhd.name_prefix != hnd.name_prefix
    assert tuple(dispatch.templates) == (nhd, hnd)
