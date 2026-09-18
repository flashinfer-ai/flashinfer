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

Regression coverage for normalization through the public prefill API.
"""

import re
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

# GDN's varlen helpers also import CuTe DSL during module initialization.
pytest.importorskip("cutlass.cute")

import flashinfer.gdn_prefill as gdn_prefill
from flashinfer import chunk_gated_delta_rule
from flashinfer.cute_dsl.availability import is_cute_dsl_arch_supported
from flashinfer.utils import get_compute_capability

from .reference_delta_rule import blockwise_delta_rule


def _supported_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    major, minor = get_compute_capability(device)
    if major not in (9, 10, 12):
        pytest.skip("GDN prefill requires SM90, SM100, or SM12x")
    if major == 10 and int(torch.version.cuda.split(".")[0]) < 13:
        pytest.skip("SM100 GDN prefill requires CUDA 13+")
    if not is_cute_dsl_arch_supported(major, minor):
        pytest.skip("CuTe DSL does not support this GPU")
    return device


def _normalize_reference(x):
    x = x.double()
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + 1e-6)


@pytest.mark.parametrize("use_cp", [False, True])
@pytest.mark.parametrize(
    "capability,cuda_version,dsl_available",
    [
        pytest.param((8, 0), "13.0", True, id="unsupported-gdn-arch"),
        pytest.param((10, 0), "12.8", True, id="unsupported-cuda-version"),
        pytest.param((9, 0), "13.0", False, id="unavailable-cute-dsl"),
    ],
)
def test_prefill_normalization_preserves_backend_rejection(
    monkeypatch, use_cp, capability, cuda_version, dsl_available
):
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda _: capability)
    monkeypatch.setattr(gdn_prefill, "get_device_sm_count", lambda _: 132)
    monkeypatch.setattr(gdn_prefill, "get_device_name", lambda _: "test GPU")
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(
        gdn_prefill, "is_cute_dsl_arch_supported", lambda *_: dsl_available
    )
    if not dsl_available:
        monkeypatch.setattr(gdn_prefill, "chunk_gated_delta_rule_sm90", None)
        monkeypatch.setattr(gdn_prefill, "cp_delta_rule_dsl_sm90", None)

    normalize = Mock(side_effect=AssertionError("normalization must not run"))
    normalization_module = ModuleType("flashinfer.gdn_kernels.qk_l2norm")
    normalization_module.normalize_qk = normalize
    monkeypatch.setitem(
        sys.modules, normalization_module.__name__, normalization_module
    )
    q = torch.zeros(8, 1, 128, dtype=torch.float16)
    kwargs = dict(
        q=q,
        k=q,
        v=q,
        cu_seqlens=torch.tensor([0, 8], dtype=torch.int64),
        use_cp=use_cp,
        backend="flashinfer",
    )
    error_type = ValueError if use_cp else NotImplementedError
    with pytest.raises(error_type) as expected:
        chunk_gated_delta_rule(**kwargs)
    with pytest.raises(error_type, match=re.escape(str(expected.value))):
        chunk_gated_delta_rule(**kwargs, use_qk_l2norm_in_kernel=True)
    normalize.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_qk_normalization_reference_and_graph(dtype):
    device = _supported_device()
    from flashinfer.gdn_kernels.qk_l2norm import normalize_qk

    torch.manual_seed(2026)
    q = torch.randn(37, 32, 128, device=device, dtype=dtype) * 3
    k = torch.randn(37, 16, 128, device=device, dtype=dtype) * 4
    q[0] = 0
    k[0] = 1e-5
    expected = [_normalize_reference(x).to(dtype) for x in (q, k)]
    outputs = normalize_qk(q, k)
    for actual, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = normalize_qk(q, k)
    graph.replay()
    for actual, reference in zip(graph_outputs, outputs, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("q_rows,k_rows", [(0, 0), (0, 7), (7, 0)])
def test_qk_normalization_empty_inputs(dtype, q_rows, k_rows):
    device = _supported_device()
    from flashinfer.gdn_kernels.qk_l2norm import normalize_qk

    torch.manual_seed(2026)
    q = torch.randn(q_rows, 32, 128, device=device, dtype=dtype)
    k = torch.randn(k_rows, 16, 128, device=device, dtype=dtype)
    outputs = normalize_qk(q, k)
    for actual, source in zip(outputs, (q, k), strict=True):
        torch.testing.assert_close(
            actual, _normalize_reference(source).to(dtype), rtol=1e-3, atol=1e-3
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prefill_normalization_all_empty(dtype):
    device = _supported_device()
    q = torch.empty(0, 16, 128, device=device, dtype=dtype)
    k = torch.empty(0, 16, 128, device=device, dtype=dtype)
    v = torch.empty(0, 32, 128, device=device, dtype=dtype)
    gate = torch.empty(0, 32, device=device, dtype=torch.float32)
    output_state = torch.full((2, 32, 128, 128), 123.0, device=device)
    output, state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=gate,
        cu_seqlens=torch.zeros(3, device=device, dtype=torch.int64),
        output_final_state=True,
        output_state=output_state,
        use_qk_l2norm_in_kernel=True,
        use_cp=False,
        backend="flashinfer",
    )
    torch.testing.assert_close(output, torch.empty_like(v))
    torch.testing.assert_close(state, torch.full_like(output_state, 123.0))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_cp", [False, True])
@pytest.mark.parametrize("heads", [(16, 16, 32), (32, 16, 16)])
@pytest.mark.parametrize("use_qk_l2norm", [False, True])
def test_prefill_normalization_flag(dtype, use_cp, heads, use_qk_l2norm):
    device = _supported_device()
    torch.manual_seed(2026)
    hq, hk, hv = heads
    q = torch.randn(104, hq, 128, device=device, dtype=dtype) * 3
    # Keep the recurrence stable when normalization is disabled.
    k = torch.randn(104, hk, 128, device=device, dtype=dtype) * 0.05
    v = torch.randn(104, hv, 128, device=device, dtype=dtype)
    h = max(hq, hv)
    kwargs = dict(
        v=v,
        g=torch.rand(104, h, device=device),
        beta=torch.rand(104, h, device=device),
        initial_state=torch.randn(2, h, 128, 128, device=device) * 0.01,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 37, 104], device=device, dtype=torch.int64),
        use_cp=use_cp,
        backend="flashinfer",
    )
    ref_q, ref_k = (
        (_normalize_reference(q).to(dtype), _normalize_reference(k).to(dtype))
        if use_qk_l2norm
        else (q, k)
    )
    ref_output, ref_state = blockwise_delta_rule(
        ref_q.float(),
        ref_k.float(),
        v.float(),
        [37, 67],
        scale_factor=128**-0.5,
        alpha=kwargs["g"],
        beta=kwargs["beta"],
        initial_state=kwargs["initial_state"].transpose(-1, -2),
        state_dtype=torch.float32,
    )
    actual = chunk_gated_delta_rule(
        q=q, k=k, use_qk_l2norm_in_kernel=use_qk_l2norm, **kwargs
    )
    for result in actual:
        assert torch.isfinite(result).all()
    torch.testing.assert_close(
        actual[0],
        ref_output.to(dtype),
        atol=1e-2 if dtype == torch.bfloat16 else 2e-3,
        rtol=1e-2 if dtype == torch.bfloat16 else 1e-3,
    )
    torch.testing.assert_close(
        actual[1].transpose(-1, -2),
        ref_state,
        atol=5e-3 if dtype == torch.bfloat16 else 1e-3,
        rtol=1e-3 if dtype == torch.bfloat16 else 1e-4,
    )
    if use_qk_l2norm:
        expected = chunk_gated_delta_rule(
            q=ref_q, k=ref_k, use_qk_l2norm_in_kernel=False, **kwargs
        )
        for result, reference in zip(actual, expected, strict=True):
            relative_error = (
                result.float() - reference.float()
            ).norm() / reference.float().norm()
            assert relative_error < 1e-3

    if dtype == torch.bfloat16 and not use_cp and heads == (16, 16, 32):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = chunk_gated_delta_rule(
                q=q, k=k, use_qk_l2norm_in_kernel=use_qk_l2norm, **kwargs
            )
        graph.replay()
        for result, reference in zip(graph_outputs, actual, strict=True):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
