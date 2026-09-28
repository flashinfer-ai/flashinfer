# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Direct padded stores must not change add-RMSNorm/NVFP4 arithmetic."""

import pytest
import torch

from flashinfer.cute_dsl.utils import is_cute_dsl_available, is_cute_dsl_arch_supported

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cute_dsl_available()
    or torch.cuda.get_device_capability()[0] < 10
    or not is_cute_dsl_arch_supported(*torch.cuda.get_device_capability()),
    reason="Requires Blackwell or newer with CuTe-DSL support",
)


def _inputs(rows, hidden, dtype):
    torch.manual_seed(42)
    x = torch.randn((rows, hidden), device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn((hidden,), device="cuda", dtype=dtype)
    scale = torch.tensor([3.7], device="cuda")
    return x, residual, weight, scale


def _check(x, residual_source, weight, scale, residual, normalized, fp4, sf):
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant

    reference_residual = residual_source.clone()
    reference_norm = torch.empty_like(x)
    reference_fp4, reference_sf = add_rmsnorm_fp4quant(
        x, reference_residual, weight, global_scale=scale, y_out=reference_norm
    )
    h = x.shape[-1]
    torch.testing.assert_close(residual, reference_residual, rtol=0, atol=0)
    torch.testing.assert_close(normalized, reference_norm, rtol=0, atol=0)
    torch.testing.assert_close(
        fp4.view(torch.uint8)[..., : h // 2],
        reference_fp4.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        sf.view(torch.uint8)[..., : h // 16],
        reference_sf.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    assert not fp4.view(torch.uint8)[..., h // 2 :].count_nonzero().item()
    assert not sf.view(torch.uint8)[..., h // 16 :].count_nonzero().item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [0, 1, 7, 228])
@pytest.mark.parametrize(
    "hidden, physical", [(512, 512), (512, 528), (2880, 3072), (4096, 5120)]
)
def test_nvfp4_padded_output_exact(dtype, rows, hidden, physical):
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant

    x, residual, weight, scale = _inputs(rows, hidden, dtype)
    original_residual = residual.clone()
    normalized = torch.empty_like(x)
    fp4 = torch.empty(
        (rows, physical // 2), dtype=torch.float4_e2m1fn_x2, device="cuda"
    )
    sf = torch.empty((rows, physical // 16), dtype=torch.float8_e4m3fn, device="cuda")
    fp4.view(torch.uint8).fill_(255)
    sf.view(torch.uint8).fill_(255)
    actual_fp4, actual_sf = add_rmsnorm_fp4quant(
        x,
        residual,
        weight,
        global_scale=scale,
        y_out=normalized,
        output_hidden_size=physical,
        y_fp4=fp4,
        block_scale=sf,
    )
    assert actual_fp4 is fp4 and actual_sf is sf
    _check(x, original_residual, weight, scale, residual, normalized, fp4, sf)


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_nvfp4_padded_output_graph(enable_pdl):
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant

    x, residual, weight, scale = _inputs(7, 2880, torch.bfloat16)
    normalized = torch.empty_like(x)

    def run():
        return add_rmsnorm_fp4quant(
            x,
            residual,
            weight,
            global_scale=scale,
            y_out=normalized,
            output_hidden_size=3072,
            enable_pdl=enable_pdl,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fp4, sf = run()
    for value in [3.7, 1.0, 512.0]:
        residual.normal_()
        if value == 1.0:
            residual.copy_(-x)
        original_residual = residual.clone()
        scale.fill_(value)
        fp4.view(torch.uint8).fill_(255)
        sf.view(torch.uint8).fill_(255)
        graph.replay()
        _check(x, original_residual, weight, scale, residual, normalized, fp4, sf)


def test_nvfp4_padded_output_3d():
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant

    x, residual, weight, scale = _inputs(6, 2880, torch.bfloat16)
    x, residual = x.view(2, 3, 2880), residual.view(2, 3, 2880)
    source = residual.clone()
    normalized = torch.empty_like(x)
    fp4, sf = add_rmsnorm_fp4quant(
        x,
        residual,
        weight,
        global_scale=scale,
        y_out=normalized,
        output_hidden_size=3072,
    )
    assert fp4.shape == (2, 3, 1536) and sf.shape == (2, 3, 192)
    _check(x, source, weight, scale, residual, normalized, fp4, sf)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_hidden_size": 2864},
        {"output_hidden_size": 3073},
        {"output_hidden_size": 3072, "block_size": 32},
        {"output_hidden_size": 3072, "is_sf_swizzled_layout": True},
        {"output_hidden_size": 3072, "output_both_sf_layouts": True},
    ],
)
def test_nvfp4_padded_output_rejects_unsupported(kwargs):
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant

    x, residual, weight, scale = _inputs(1, 2880, torch.bfloat16)
    source = residual.clone()
    with pytest.raises(ValueError, match="Padded outputs require"):
        add_rmsnorm_fp4quant(x, residual, weight, global_scale=scale, **kwargs)
    torch.testing.assert_close(residual, source, rtol=0, atol=0)


def test_nvfp4_padded_output_rejects_compact_buffer():
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant

    x, residual, weight, scale = _inputs(1, 2880, torch.bfloat16)
    compact = torch.empty((1, 1440), dtype=torch.float4_e2m1fn_x2, device="cuda")
    with pytest.raises(ValueError, match="y_fp4 must be"):
        add_rmsnorm_fp4quant(
            x,
            residual,
            weight,
            global_scale=scale,
            output_hidden_size=3072,
            y_fp4=compact,
        )


def test_nvfp4_padded_output_without_norm_store():
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant

    x, residual, weight, scale = _inputs(7, 2880, torch.bfloat16)
    ref_residual = residual.clone()
    ref_norm = torch.empty_like(x)
    expected = add_rmsnorm_fp4quant(
        x,
        ref_residual,
        weight,
        global_scale=scale,
        output_hidden_size=3072,
        y_out=ref_norm,
    )
    actual = add_rmsnorm_fp4quant(
        x,
        residual,
        weight,
        global_scale=scale,
        output_hidden_size=3072,
    )
    for out, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            out.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0
        )
    torch.testing.assert_close(residual, ref_residual, rtol=0, atol=0)


def test_nvfp4_padded_output_rejects_multi_cta():
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant

    x, residual, weight, scale = _inputs(1, 65536, torch.bfloat16)
    before = residual.clone()
    with pytest.raises(ValueError, match="Padded outputs require single-CTA"):
        add_rmsnorm_fp4quant(
            x,
            residual,
            weight,
            global_scale=scale,
            output_hidden_size=65552,
        )
    torch.testing.assert_close(residual, before, rtol=0, atol=0)


def test_nvfp4_padded_output_trace_not_misrepresented():
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant

    x, residual, weight, _ = _inputs(1, 2880, torch.bfloat16)
    assert add_rmsnorm_fp4quant.fi_trace(input=x, residual=residual, weight=weight)
    assert (
        add_rmsnorm_fp4quant.fi_trace(
            input=x,
            residual=residual,
            weight=weight,
            output_hidden_size=3072,
        )
        == {}
    )
