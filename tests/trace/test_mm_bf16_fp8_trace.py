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

"""The mm_bf16_fp8 / mm_fp16_fp8 traces describe the ranks they were called
with: a is (..., K) and alpha is () or (1,).  CPU tensors; no kernel runs."""

import pytest
import torch

from flashinfer.gemm import mm_bf16_fp8, mm_fp16_fp8
from flashinfer.trace.templates.gemm import (
    _mm_16bit_fp8_reference,
    mm_bf16_fp8_trace_dispatch,
    mm_fp16_fp8_trace_dispatch,
)

_N, _K = 128, 64
_FNS = [
    (mm_bf16_fp8, mm_bf16_fp8_trace_dispatch, torch.bfloat16, "bf16"),
    (mm_fp16_fp8, mm_fp16_fp8_trace_dispatch, torch.float16, "fp16"),
]


def _call(a_shape, scale_shape, dtype, bias=False):
    a = torch.zeros(*a_shape, dtype=dtype)
    b = torch.zeros(_N, _K, dtype=torch.float8_e4m3fn)
    alpha = torch.ones(scale_shape, dtype=torch.float32)
    kwargs = {"a": a, "b": b, "alpha": alpha}
    if bias:
        kwargs["bias"] = torch.zeros(_N, dtype=dtype)
    return kwargs


@pytest.mark.parametrize("fn,dispatch,dtype,short", _FNS)
@pytest.mark.parametrize(
    "a_shape,a_dims",
    [
        ((_K,), ["K"]),
        ((5, _K), ["M", "K"]),
        ((2, 3, _K), ["batch", "seq_len", "K"]),
        ((2, 3, 4, _K), ["lead0", "lead1", "lead2", "K"]),
    ],
)
@pytest.mark.parametrize("scale_shape", [(), (1,)])
def test_mm_16bit_fp8_trace_ranks(
    fn, dispatch, dtype, short, a_shape, a_dims, scale_shape
):
    dtype_name = str(dtype).removeprefix("torch.")
    kwargs = _call(a_shape, scale_shape, dtype)
    defn = fn.fi_trace(**kwargs)
    assert defn["op_type"] == f"gemm_{short}_fp8"
    assert defn["inputs"]["a"]["shape"] == a_dims
    assert defn["inputs"]["a"]["dtype"] == dtype_name
    assert defn["inputs"]["b"]["shape"] == ["N", "K"]
    assert defn["inputs"]["b"]["dtype"] == "float8_e4m3fn"
    assert defn["outputs"]["out"]["shape"] == a_dims[:-1] + ["N"]
    assert defn["outputs"]["out"]["dtype"] == dtype_name
    # K comes from a's last dimension whatever its rank; N from the weight.
    assert defn["axes"]["K"] == {"type": "const", "value": _K}
    assert defn["axes"]["N"] == {"type": "const", "value": _N}
    for lead in a_dims[:-1]:
        assert defn["axes"][lead]["type"] == "var"
    scale_entry = defn["inputs"]["alpha"]
    assert scale_entry["shape"] == (["scalar"] if scale_shape == (1,) else [])
    assert scale_entry["dtype"] == "float32"
    # The name says which ranks it describes; the common 2-D case keeps the plain name.
    expected = fn.__name__
    if a_shape != (5, _K) or scale_shape != (1,):
        expected += f"_a{len(a_shape)}d" + ("_alpha0d" if scale_shape == () else "")
    assert defn["name"].startswith(expected + "_N"), defn["name"]
    assert "_N128_K64" in defn["name"]


@pytest.mark.parametrize("fn,dispatch,dtype,short", _FNS)
def test_mm_16bit_fp8_trace_dispatch_is_cached_per_rank(fn, dispatch, dtype, short):
    a = dispatch(**_call((2, 3, _K), (1,), dtype))
    b = dispatch(**_call((7, 9, _K), (1,), dtype))
    c = dispatch(**_call((2, 3, _K), (), dtype))
    assert a is b
    assert a is not c


def test_mm_16bit_fp8_trace_templates_differ_by_dtype():
    bf16 = mm_bf16_fp8_trace_dispatch(**_call((5, _K), (1,), torch.bfloat16))
    fp16 = mm_fp16_fp8_trace_dispatch(**_call((5, _K), (1,), torch.float16))
    assert bf16 is not fp16
    assert bf16.name_prefix == "mm_bf16_fp8"
    assert fp16.name_prefix == "mm_fp16_fp8"


@pytest.mark.parametrize("fn,dispatch,dtype,short", _FNS)
def test_mm_16bit_fp8_trace_bias_optional(fn, dispatch, dtype, short):
    dtype_name = str(dtype).removeprefix("torch.")
    defn = fn.fi_trace(**_call((5, _K), (1,), dtype, bias=True))
    assert defn["inputs"]["bias"]["optional"] is True
    assert defn["inputs"]["bias"]["dtype"] == dtype_name
    defn = fn.fi_trace(**_call((5, _K), (1,), dtype))
    assert defn["inputs"]["bias"]["optional"] is True
    assert defn["inputs"]["bias"]["dtype"] == dtype_name  # declared, not inferred


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("a_shape", [(_K,), (5, _K), (2, 3, _K)])
@pytest.mark.parametrize("scale_shape", [(), (1,)])
def test_mm_16bit_fp8_trace_reference_keeps_leading_shape(dtype, a_shape, scale_shape):
    torch.manual_seed(0)
    a = torch.randn(*a_shape, dtype=dtype)
    b = (torch.randn(_N, _K) * 4).to(torch.float8_e4m3fn)
    alpha = torch.full(scale_shape, 0.25, dtype=torch.float32)
    bias = torch.randn(_N, dtype=dtype)
    out = _mm_16bit_fp8_reference(a, b, alpha, bias)
    assert out.shape == a_shape[:-1] + (_N,)
    assert out.dtype == dtype
    expected = torch.nn.functional.linear(a.float(), b.float() * 0.25, bias.float())
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
