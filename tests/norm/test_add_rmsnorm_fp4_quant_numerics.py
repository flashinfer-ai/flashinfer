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
"""Numerical boundaries in the single-CTA add-RMSNorm + NVFP4 path."""

import pytest
import torch

from flashinfer.cute_dsl.utils import (
    is_cute_dsl_arch_supported,
    is_cute_dsl_available,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cute_dsl_available()
    or torch.cuda.get_device_capability()[0] < 10
    or not is_cute_dsl_arch_supported(*torch.cuda.get_device_capability()),
    reason="Requires Blackwell or newer with CuTe-DSL support",
)


def _require_single_cta(hidden_size, dtype):
    import cutlass

    from flashinfer.cute_dsl.add_rmsnorm_fp4quant import AddRMSNormFP4QuantKernel

    kernel = AddRMSNormFP4QuantKernel(
        cutlass.Float16 if dtype == torch.float16 else cutlass.BFloat16,
        hidden_size,
        block_size=16,
        output_swizzled=False,
        is_fp16=dtype == torch.float16,
    )
    if kernel.cluster_n != 1:
        pytest.skip("The numerical correction targets the single-CTA NVFP4 path")


def _assert_quantized_norm(y_fp4, scales, y_norm, global_scale, swizzled=False):
    from flashinfer import fp4_quantize

    # Quantize the exact materialized norm: this isolates FP4 arithmetic from
    # differences in the RMSNorm reduction order, without a loose error budget.
    ref_fp4, ref_scales = fp4_quantize(
        y_norm,
        global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
        backend="cuda",
    )
    torch.testing.assert_close(
        y_fp4.view(torch.uint8), ref_fp4.view(torch.uint8), rtol=0, atol=0
    )
    num_rows, hidden_size = y_norm.shape
    num_scales = hidden_size // 16
    actual_scales = scales.view(torch.uint8)
    if swizzled:
        # Only live scale slots are defined; do not compare swizzle padding.
        row = torch.arange(num_rows, device="cuda")[:, None]
        col = torch.arange(num_scales, device="cuda")[None, :]
        offset = (
            (row // 128) * ((num_scales + 3) // 4) * 512
            + (col // 4) * 512
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + col % 4
        )
        actual_scales = actual_scales.flatten()[offset]
    torch.testing.assert_close(
        actual_scales.reshape(num_rows, num_scales),
        ref_scales.view(torch.uint8).reshape(num_rows, num_scales),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [2880, 4096])
def test_nvfp4_preserves_residual_sum_precision(dtype, hidden_size):
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant

    _require_single_cta(hidden_size, dtype)
    x = torch.ones((7, hidden_size), dtype=dtype, device="cuda")
    residual = torch.full_like(x, torch.finfo(dtype).eps / 2)
    residual[:, hidden_size // 2 :] *= -1
    weight = torch.full((hidden_size,), 0.75, dtype=dtype, device="cuda")
    h = x.float() + residual.float()
    expected = (
        h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)
    ) * weight.float()
    y_norm = torch.empty_like(x)

    add_rmsnorm_fp4quant(x, residual, weight, y_out=y_norm)

    # The positive sum lies halfway between input-dtype values. Rounding it
    # before normalization loses information that changes the final result.
    torch.testing.assert_close(residual, h.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(y_norm, expected.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 7, 228])
@pytest.mark.parametrize("hidden_size", [512, 2880, 8192])
@pytest.mark.parametrize("scale", [1.0, 3.7, 512.0])
def test_nvfp4_quantizes_materialized_norm(dtype, batch_size, hidden_size, scale):
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant

    _require_single_cta(hidden_size, dtype)
    torch.manual_seed(42)
    x = torch.randn((batch_size, hidden_size), dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn((hidden_size,), dtype=dtype, device="cuda")
    y_norm = torch.empty_like(x)
    global_scale = torch.tensor([scale], dtype=torch.float32, device="cuda")
    y_fp4, scales = add_rmsnorm_fp4quant(
        x, residual, weight, global_scale=global_scale, y_out=y_norm
    )

    _assert_quantized_norm(y_fp4, scales, y_norm, global_scale)


@pytest.mark.parametrize("batch_size", [7, 228])
@pytest.mark.parametrize("both_layouts", [False, True])
def test_nvfp4_materialized_norm_swizzled_scales(batch_size, both_layouts):
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant

    torch.manual_seed(42)
    x = torch.randn((batch_size, 2880), dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn((2880,), dtype=x.dtype, device="cuda")
    y_norm = torch.empty_like(x)
    global_scale = torch.tensor([3.7], dtype=torch.float32, device="cuda")
    result = add_rmsnorm_fp4quant(
        x,
        residual,
        weight,
        global_scale=global_scale,
        y_out=y_norm,
        is_sf_swizzled_layout=True,
        output_both_sf_layouts=both_layouts,
    )

    _assert_quantized_norm(result[0], result[1], y_norm, global_scale, swizzled=True)
    if both_layouts:
        _assert_quantized_norm(result[0], result[2], y_norm, global_scale)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_nvfp4_materialized_norm_cuda_graph(dtype, enable_pdl):
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant

    torch.manual_seed(42)
    x = torch.randn((7, 2880), dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn((2880,), dtype=dtype, device="cuda")
    y_norm = torch.empty_like(x)
    global_scale = torch.tensor([3.7], dtype=torch.float32, device="cuda")

    def run():
        return add_rmsnorm_fp4quant(
            x,
            residual,
            weight,
            global_scale=global_scale,
            y_out=y_norm,
            enable_pdl=enable_pdl,
        )

    run()  # Compile before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_fp4, scales = run()

    # Replay with changed inputs and a device-resident global scale, including
    # a zero-norm case to cover the zero-scale reciprocal guard.
    for scale in (3.7, 512.0, 1.0):
        global_scale.fill_(scale)
        residual.normal_()
        if scale == 1.0:
            residual.copy_(-x)
        h = x.float() + residual.float()
        expected = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)
        expected *= weight.float()
        graph.replay()
        torch.testing.assert_close(residual, h.to(dtype), rtol=0, atol=0)
        torch.testing.assert_close(
            y_norm.float(),
            expected.to(dtype).float(),
            rtol=torch.finfo(dtype).eps,
            atol=1e-6,
        )
        _assert_quantized_norm(y_fp4, scales, y_norm, global_scale)
