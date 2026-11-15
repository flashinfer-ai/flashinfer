"""Regression tests for standalone FP8 token/head cache trace replay."""

import ast
import json
from pathlib import Path

import pytest
import torch

from flashinfer.trace.templates.page import (
    fp8_token_head_quantize_append_paged_kv_cache_with_slot_mapping_trace as dispatch,
)


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("kv_data_type", ["e4m3", "e5m2"])
@pytest.mark.parametrize("shape", [(2, 64, 4), (6, 256, 8)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp8_token_head_trace_layout_roundtrip(kv_layout, kv_data_type, shape, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    num_kv_heads, head_dim, page_size = shape
    template = dispatch(kv_layout=kv_layout, kv_data_type=kv_data_type)
    init_kwargs = dict(
        nnz_kv=4,
        num_pages=4,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        device=device,
    )
    inputs = template.init(**init_kwargs)
    assert inputs["kv_layout"] == kv_layout
    assert inputs["kv_data_type"] == kv_data_type
    trace = template.build_fi_trace_fn(
        "flashinfer.page.fp8_token_head_quantize_append_paged_kv_cache_with_slot_mapping"
    )
    if device == "cuda":
        from flashinfer import (
            fp8_token_head_quantize_append_paged_kv_cache_with_slot_mapping as op,
        )

        assert op.fi_trace(**inputs) == trace(**inputs)
        live_inputs = op.fi_init(
            kv_layout=kv_layout, kv_data_type=kv_data_type, **init_kwargs
        )
        assert live_inputs["kv_layout"] == kv_layout
        assert live_inputs["kv_data_type"] == kv_data_type
    definition = trace(**inputs)
    assert definition["axes"]["page_size"]["value"] == page_size
    assert definition["axes"]["num_kv_heads"]["value"] == num_kv_heads
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
    assert f"kv_data_type == {kv_data_type!r}" in definition["constraints"]
    assert definition["inputs"]["kv_data_type"]["dtype"] == "string"

    # Execute the exported init in a fresh namespace: HND must survive without
    # the live module's globals or caller overrides for Const axes or format.
    namespace = {}
    exec(definition["init"], namespace)
    restored = namespace[template.init.__name__](nnz_kv=4, num_pages=4, device=device)
    assert restored["kv_layout"] == kv_layout
    assert restored["kv_data_type"] == kv_data_type
    assert trace(**restored) == definition
    for slots, scales in zip(
        restored["paged_kv_cache"], restored["kv_cache_sf"], strict=True
    ):
        assert scales.untyped_storage().data_ptr() == slots.untyped_storage().data_ptr()
        assert scales.storage_offset() * 4 == head_dim
        assert slots.shape == (
            (4, page_size, num_kv_heads, head_dim + 16)
            if kv_layout == "NHD"
            else (4, num_kv_heads, page_size, head_dim + 16)
        )

    if device == "cuda":
        # Constant heads make the expected FP8 bytes and scale exact, and
        # distinguish e5m2 from silently replaying the default e4m3 format.
        restored["append_key"].fill_(1)
        restored["append_value"].fill_(-1)
        op(**restored)
        dtype = torch.float8_e4m3fn if kv_data_type == "e4m3" else torch.float8_e5m2
        fp8_max = torch.finfo(dtype).max
        for slots, scales, sign in zip(
            restored["paged_kv_cache"], restored["kv_cache_sf"], (1, -1), strict=True
        ):
            if kv_layout == "HND":
                slots = slots.transpose(1, 2)
                scales = scales.transpose(1, 2)
            data = slots[..., :head_dim].contiguous().view(dtype).float()
            data = data.reshape(-1, num_kv_heads, head_dim)[:4]
            scales = scales.reshape(-1, num_kv_heads)[:4]
            torch.testing.assert_close(data, torch.full_like(data, sign * fp8_max))
            torch.testing.assert_close(
                scales, torch.full_like(scales, 1 / fp8_max), rtol=2e-7, atol=0
            )


def test_fp8_token_head_trace_default_and_names():
    nhd = dispatch()
    hnd = dispatch(kv_layout="HND")
    assert nhd is dispatch(kv_layout="NHD")
    assert nhd.name_prefix != hnd.name_prefix
    assert nhd.name_prefix == (
        "fp8_token_head_quantize_append_paged_kv_cache_with_slot_mapping"
    )
    templates = [
        dispatch(kv_layout=layout, kv_data_type=fmt)
        for fmt in ("e4m3", "e5m2")
        for layout in ("NHD", "HND")
    ]
    assert tuple(dispatch.templates) == tuple(templates)
    assert len({template.name_prefix for template in templates}) == 4


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "aliases",
    [
        ("e4m3", "float8_e4m3fn", torch.float8_e4m3fn),
        ("e5m2", "float8_e5m2", torch.float8_e5m2),
    ],
)
def test_fp8_token_head_trace_dtype_aliases(kv_layout, aliases):
    from flashinfer import (
        fp8_token_head_quantize_append_paged_kv_cache_with_slot_mapping as op,
    )

    inputs = op.fi_init(nnz_kv=4, kv_layout=kv_layout, device="cpu")
    definitions = [op.fi_trace(**dict(inputs, kv_data_type=alias)) for alias in aliases]
    assert definitions[0] == definitions[1] == definitions[2]


@pytest.mark.parametrize(
    "path",
    sorted((Path(__file__).parent / "fi_trace_out").glob("fp8_token_head_*.json")),
    ids=lambda path: path.stem,
)
def test_fp8_token_head_committed_trace_replay(path):
    from flashinfer import (
        fp8_token_head_quantize_append_paged_kv_cache_with_slot_mapping as op,
    )

    definition = json.loads(path.read_text())
    namespace = {}
    exec(definition["init"], namespace)
    template = next(
        t
        for t in dispatch.templates
        if definition["name"].startswith(t.name_prefix + "_kv")
    )
    inputs = namespace[template.init.__name__](nnz_kv=4, num_pages=4, device="cpu")
    replayed = op.fi_trace(**inputs)
    # Compare ASTs because ast.unparse formatting varies across Python versions.
    assert ast.dump(ast.parse(replayed.pop("init"))) == ast.dump(
        ast.parse(definition.pop("init"))
    )
    assert replayed == definition
