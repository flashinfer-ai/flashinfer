# SPDX-License-Identifier: Apache-2.0
"""Trace replay must preserve section, layout, dimensions and precision constants."""

import torch
import pytest
from flashinfer import apply_mrope_with_cos_sin_cache_inplace as api
from flashinfer.trace.templates.mrope import _mrope_init, _mrope_reference, _mrope_round


@pytest.mark.parametrize("interleaved", [0, 1])
@pytest.mark.parametrize("native", [0, 1])
def test_exported_mrope_init_and_reference(interleaved, native):
    inputs = _mrope_init(
        nnz=7,
        max_seq_len=31,
        num_q_heads_x_head_size=240,
        num_k_heads_x_head_size=80,
        head_size=80,
        rotary_dim=60,
        section_t=12,
        section_h=9,
        section_w=9,
        mrope_interleaved=interleaved,
        query_bf16=0,
        cache_fp32=not native,
        position_int64=0,
        device="cpu",
    )
    definition = api.fi_trace(**inputs)
    assert definition["axes"]["section_t"]["value"] == 12
    assert definition["axes"]["mrope_interleaved"]["value"] == interleaved
    assert definition["inputs"]["cos_sin_cache"]["dtype"] == (
        "float16" if native else "float32"
    )
    init_scope, ref_scope = {}, {}
    exec(definition["init"], init_scope)
    exec(definition["reference"], ref_scope)
    # Supply only Var axes: every Const must come from the exported defaults.
    replay = init_scope["_mrope_init"](nnz=5, max_seq_len=19, device="cpu")
    assert replay["head_size"] == 80 and replay["query"].shape == (5, 240)
    assert replay["mrope_section"] == (12, 9, 9)
    assert replay["mrope_interleaved"] == bool(interleaved)
    assert replay["cos_sin_cache"].dtype == inputs["cos_sin_cache"].dtype
    assert replay["positions"].dtype == torch.int32
    expected = _mrope_reference(**replay)
    actual = ref_scope["_mrope_reference"](**replay)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reference_rounding_overflow_boundary(dtype):
    # Check both sides of the IEEE round-to-nearest overflow midpoint, using
    # FP64 neighbors that a cast through FP32 cannot distinguish.
    largest = torch.tensor(torch.finfo(dtype).max, dtype=dtype)
    previous = torch.nextafter(largest, torch.zeros_like(largest))
    threshold = largest.double() + (largest.double() - previous.double()) / 2
    below = torch.nextafter(threshold, torch.zeros_like(threshold))
    above = torch.nextafter(threshold, torch.full_like(threshold, float("inf")))
    values = torch.stack((below, threshold, above))
    expected = torch.tensor([largest.item(), float("inf"), float("inf")], dtype=dtype)
    for sign in (-1, 1):
        torch.testing.assert_close(
            _mrope_round(sign * values, dtype), sign * expected, rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reference_rounding_midpoint_and_nonfinite(dtype):
    one = torch.tensor(1.0, dtype=dtype)
    successor = torch.nextafter(one, torch.full_like(one, float("inf")))
    midpoint = (one.double() + successor.double()) / 2
    below = torch.nextafter(midpoint, torch.zeros_like(midpoint))
    above = torch.nextafter(midpoint, torch.full_like(midpoint, float("inf")))
    values = torch.stack((below, midpoint, above))
    expected = torch.tensor([1.0, 1.0, successor.item()], dtype=dtype)
    for sign in (-1, 1):
        torch.testing.assert_close(
            _mrope_round(sign * values, dtype), sign * expected, rtol=0, atol=0
        )
    special = torch.tensor(
        [float("inf"), -float("inf"), float("nan"), -0.0], dtype=torch.float64
    )
    actual = _mrope_round(special, dtype)
    torch.testing.assert_close(
        actual, special.to(dtype), rtol=0, atol=0, equal_nan=True
    )
    assert torch.signbit(actual[-1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_exported_reference_rotation_overflow(dtype):
    largest = torch.finfo(dtype).max
    query = torch.tensor([[largest, largest, largest, -largest]], dtype=dtype)
    inputs = dict(
        positions=torch.zeros(3, 1, dtype=torch.int64),
        query=query,
        key=-query,
        head_size=2,
        cos_sin_cache=torch.full((1, 2), 2**-0.5, dtype=dtype),
        mrope_section=(1, 0, 0),
    )
    scope = {}
    exec(api.fi_trace(**inputs)["reference"], scope)
    # The first head's sum and the second head's difference overflow. This
    # checks the exported reference against an analytic result, not itself.
    for reference in (_mrope_reference, scope["_mrope_reference"]):
        q, k = reference(**inputs)
        assert torch.isfinite(q[0, [0, 3]]).all()
        assert torch.isposinf(q[0, [1, 2]]).all()
        assert torch.isneginf(k[0, [1, 2]]).all()
    assert torch.isfinite(query).all()
