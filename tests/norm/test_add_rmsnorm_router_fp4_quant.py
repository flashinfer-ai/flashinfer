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
"""Experimental router accuracy is distinct from exact quantization parity."""

import math

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


def _run_case(rows, hidden, physical, experts, dtype, bias, graph=False, pdl=False):
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant
    from flashinfer.cute_dsl.add_rmsnorm_router_fp4quant import (
        _experimental_add_rmsnorm_router_fp4quant as fused,
    )

    torch.manual_seed(123)
    x = torch.randn((rows, hidden), dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn((hidden,), dtype=dtype, device="cuda")
    router = (torch.randn((experts, hidden), device="cuda") / math.sqrt(hidden)).to(
        dtype
    )
    router_bias = torch.randn((experts,), device="cuda", dtype=dtype) if bias else None
    scale = torch.tensor([3.7], device="cuda")
    normalized = torch.empty_like(x)
    fp4 = torch.empty(
        (rows, physical // 2), dtype=torch.float4_e2m1fn_x2, device="cuda"
    )
    sf = torch.empty((rows, physical // 16), dtype=torch.float8_e4m3fn, device="cuda")
    logits = torch.empty((rows, experts), device="cuda", dtype=dtype)

    def run():
        return fused(
            x,
            residual,
            weight,
            router,
            router_bias=router_bias,
            global_scale=scale,
            output_hidden_size=physical,
            y_out=normalized,
            y_fp4=fp4,
            block_scale=sf,
            router_logits=logits,
            enable_pdl=pdl,
        )

    run()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            outputs = run()
        invoke = capture.replay
    else:
        outputs = run()
        invoke = run
    assert outputs[0] is fp4 and outputs[1] is sf and outputs[2] is logits

    for value in (3.7, 1.0, 512.0):
        residual.normal_()
        if value == 1.0:
            residual.copy_(-x)
        reference_residual = residual.clone()
        router.normal_(std=1 / math.sqrt(hidden))
        if bias:
            router_bias.normal_()
        scale.fill_(value)
        fp4.view(torch.uint8).fill_(255)
        sf.view(torch.uint8).fill_(255)
        logits.fill_(float("nan"))
        normalized.fill_(float("nan"))
        reference_norm = torch.empty_like(x)
        ref_fp4, ref_sf = add_rmsnorm_fp4quant(
            x,
            reference_residual,
            weight,
            global_scale=scale,
            output_hidden_size=physical,
            y_out=reference_norm,
            enable_pdl=pdl,
        )
        invoke()
        torch.testing.assert_close(residual, reference_residual, rtol=0, atol=0)
        torch.testing.assert_close(normalized, reference_norm, rtol=0, atol=0)
        torch.testing.assert_close(
            fp4.view(torch.uint8), ref_fp4.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(
            sf.view(torch.uint8), ref_sf.view(torch.uint8), rtol=0, atol=0
        )
        assert torch.isfinite(logits).all()

        # Independent high-precision dot product of the materialized norm.
        exact = reference_norm.double() @ router.double().T
        magnitude = reference_norm.double().abs() @ router.double().abs().T
        if bias:
            exact += router_bias.double()
            magnitude += router_bias.double().abs()
        # At most ceil(H/32) multiply/adds plus five warp-reduction additions
        # and bias. Account for FP32 accumulation and final FP16/BF16 rounding.
        u = torch.finfo(torch.float32).eps / 2
        n = 2 * math.ceil(hidden / 32) + 6
        gamma = n * u / (1 - n * u)
        output_u = torch.finfo(dtype).eps / 2
        bound = (
            (1 + output_u) * gamma * magnitude
            + output_u * exact.abs()
            + torch.finfo(dtype).tiny
        )
        assert ((logits.double() - exact).abs() <= bound).all()
        if rows and experts >= 4:
            ordered = exact.sort(dim=-1, descending=True).values
            safe = ordered[:, 2] - ordered[:, 3] > 2 * bound.max(dim=-1).values
            torch.testing.assert_close(
                logits[safe].topk(3, dim=-1).indices.sort(dim=-1).values,
                exact[safe].topk(3, dim=-1).indices.sort(dim=-1).values,
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 7, 228])
@pytest.mark.parametrize(
    "hidden, physical, experts", [(512, 528, 5), (2880, 3072, 128)]
)
@pytest.mark.parametrize("bias", [False, True])
def test_router_outputs(rows, hidden, physical, experts, dtype, bias):
    _run_case(rows, hidden, physical, experts, dtype, bias)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("rows", [1, 7, 228])
def test_router_cuda_graph(rows, pdl):
    _run_case(rows, 2880, 3072, 128, torch.bfloat16, True, graph=True, pdl=pdl)


def test_router_empty():
    _run_case(0, 512, 528, 5, torch.bfloat16, True)


def test_router_rejects_overlapping_output():
    from flashinfer.cute_dsl.add_rmsnorm_router_fp4quant import (
        _experimental_add_rmsnorm_router_fp4quant as fused,
    )

    x = torch.randn((1, 512), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    before = residual.clone()
    with pytest.raises(ValueError, match="non-overlapping"):
        fused(
            x,
            residual,
            torch.ones_like(x[0]),
            torch.ones((5, 512), device=x.device, dtype=x.dtype),
            global_scale=torch.ones(1, device=x.device),
            y_out=residual,
        )
    torch.testing.assert_close(residual, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad",
    [
        "router_weight",
        "router_bias",
        "router_logits",
        "global_scale",
        "output_hidden_size",
    ],
)
def test_router_rejects_bad_arguments(bad):
    from flashinfer.cute_dsl.add_rmsnorm_router_fp4quant import (
        _experimental_add_rmsnorm_router_fp4quant as fused,
    )

    x = torch.randn((1, 512), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    before = residual.clone()
    kwargs = dict(
        input=x,
        residual=residual,
        weight=torch.ones_like(x[0]),
        router_weight=torch.ones((5, 512), device="cuda", dtype=x.dtype),
        global_scale=torch.ones(1, device="cuda"),
    )
    kwargs[bad] = 511 if bad == "output_hidden_size" else torch.empty(3, device="cuda")
    with pytest.raises(ValueError, match=bad):
        fused(**kwargs)
    torch.testing.assert_close(residual, before, rtol=0, atol=0)
