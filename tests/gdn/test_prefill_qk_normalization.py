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
from unittest.mock import Mock

import pytest
import torch

import flashinfer.gdn_prefill as gdn_prefill
from flashinfer import chunk_gated_delta_rule
from flashinfer.gdn_kernels.qk_l2norm import normalize_qk
from flashinfer.utils import get_compute_capability


def _supported_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    major, _ = get_compute_capability(device)
    if major not in (9, 10, 12):
        pytest.skip("GDN prefill requires SM90, SM100, or SM12x")
    if major == 10 and int(torch.version.cuda.split(".")[0]) < 13:
        pytest.skip("SM100 GDN prefill requires CUDA 13+")
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
    monkeypatch.setattr("flashinfer.gdn_kernels.qk_l2norm.normalize_qk", normalize)
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
@pytest.mark.parametrize("use_cp", [False, True])
@pytest.mark.parametrize("heads", [(16, 16, 32), (32, 16, 16)])
def test_prefill_normalization_flag(dtype, use_cp, heads):
    device = _supported_device()
    torch.manual_seed(2026)
    hq, hk, hv = heads
    q = torch.randn(104, hq, 128, device=device, dtype=dtype) * 3
    k = torch.randn(104, hk, 128, device=device, dtype=dtype) * 4
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
    expected = chunk_gated_delta_rule(
        q=_normalize_reference(q).to(dtype),
        k=_normalize_reference(k).to(dtype),
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    )
    actual = chunk_gated_delta_rule(q=q, k=k, use_qk_l2norm_in_kernel=True, **kwargs)
    for result, reference in zip(actual, expected, strict=True):
        assert torch.isfinite(result).all()
        relative_error = (
            result.float() - reference.float()
        ).norm() / reference.float().norm()
        assert relative_error < 1e-3

    if dtype == torch.bfloat16 and not use_cp and heads == (16, 16, 32):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = chunk_gated_delta_rule(
                q=q, k=k, use_qk_l2norm_in_kernel=True, **kwargs
            )
        graph.replay()
        for result, reference in zip(graph_outputs, actual, strict=True):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
