"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""The w8a16_fp8_gemm trace describes the ranks it was called with: act is
(..., K) and weight_scale is () or (1,).  CPU tensors; no kernel runs."""

import pytest
import torch

from flashinfer.gemm import w8a16_fp8_gemm
from flashinfer.trace.templates.gemm import (
    _w8a16_fp8_gemm_reference,
    w8a16_fp8_gemm_trace_dispatch,
)

_N, _K = 128, 64


def _call(act_shape, scale_shape, bias=False):
    act = torch.zeros(*act_shape, dtype=torch.bfloat16)
    weight = torch.zeros(_N, _K, dtype=torch.float8_e4m3fn)
    scale = torch.ones(scale_shape, dtype=torch.float32)
    kwargs = {"act": act, "weight": weight, "weight_scale": scale}
    if bias:
        kwargs["bias"] = torch.zeros(_N, dtype=torch.bfloat16)
    return kwargs


@pytest.mark.parametrize(
    "act_shape,act_dims",
    [
        ((_K,), ["K"]),
        ((5, _K), ["M", "K"]),
        ((2, 3, _K), ["batch", "seq_len", "K"]),
        ((2, 3, 4, _K), ["lead0", "lead1", "lead2", "K"]),
    ],
)
@pytest.mark.parametrize("scale_shape", [(), (1,)])
def test_w8a16_fp8_gemm_trace_ranks(act_shape, act_dims, scale_shape):
    kwargs = _call(act_shape, scale_shape)
    defn = w8a16_fp8_gemm.fi_trace(**kwargs)
    assert defn["op_type"] == "gemm_w8a16_fp8"
    assert defn["inputs"]["act"]["shape"] == act_dims
    assert defn["inputs"]["act"]["dtype"] == "bfloat16"
    assert defn["inputs"]["weight"]["shape"] == ["N", "K"]
    assert defn["inputs"]["weight"]["dtype"] == "float8_e4m3fn"
    assert defn["outputs"]["out"]["shape"] == act_dims[:-1] + ["N"]
    assert defn["outputs"]["out"]["dtype"] == "bfloat16"
    # K comes from act's last dimension whatever its rank; N from the weight.
    assert defn["axes"]["K"] == {"type": "const", "value": _K}
    assert defn["axes"]["N"] == {"type": "const", "value": _N}
    for lead in act_dims[:-1]:
        assert defn["axes"][lead]["type"] == "var"
    scale_entry = defn["inputs"]["weight_scale"]
    assert scale_entry["shape"] == (["scalar"] if scale_shape == (1,) else [])
    assert scale_entry["dtype"] == "float32"
    # The name says which ranks it describes; the common 2-D case keeps the plain name.
    expected = "w8a16_fp8_gemm"
    if act_shape != (5, _K) or scale_shape != (1,):
        expected += f"_act{len(act_shape)}d" + ("_scale0d" if scale_shape == () else "")
    assert defn["name"].startswith(expected + "_N"), defn["name"]
    assert "_N128_K64" in defn["name"]


def test_w8a16_fp8_gemm_trace_dispatch_is_cached_per_rank():
    a = w8a16_fp8_gemm_trace_dispatch(**_call((2, 3, _K), (1,)))
    b = w8a16_fp8_gemm_trace_dispatch(**_call((7, 9, _K), (1,)))
    c = w8a16_fp8_gemm_trace_dispatch(**_call((2, 3, _K), ()))
    assert a is b
    assert a is not c


def test_w8a16_fp8_gemm_trace_bias_optional():
    defn = w8a16_fp8_gemm.fi_trace(**_call((5, _K), (1,), bias=True))
    assert defn["inputs"]["bias"]["optional"] is True
    assert defn["inputs"]["bias"]["dtype"] == "bfloat16"
    defn = w8a16_fp8_gemm.fi_trace(**_call((5, _K), (1,)))
    assert defn["inputs"]["bias"]["dtype"] == "unknown"  # absent in this call


@pytest.mark.parametrize("act_shape", [(_K,), (5, _K), (2, 3, _K)])
@pytest.mark.parametrize("scale_shape", [(), (1,)])
def test_w8a16_fp8_gemm_trace_reference_keeps_leading_shape(act_shape, scale_shape):
    torch.manual_seed(0)
    act = torch.randn(*act_shape, dtype=torch.bfloat16)
    weight = (torch.randn(_N, _K) * 4).to(torch.float8_e4m3fn)
    scale = torch.full(scale_shape, 0.25, dtype=torch.float32)
    bias = torch.randn(_N, dtype=torch.bfloat16)
    out = _w8a16_fp8_gemm_reference(act, weight, scale, bias)
    assert out.shape == act_shape[:-1] + (_N,)
    assert out.dtype == torch.bfloat16
    expected = torch.nn.functional.linear(
        act.float(), weight.float() * 0.25, bias.float()
    )
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
