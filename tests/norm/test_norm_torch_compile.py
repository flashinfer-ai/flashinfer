# Copyright (c) 2025 by FlashInfer team.
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
"""
Tests for norm functions compatibility with torch.compile.

Verifies that rmsnorm / fused_add_rmsnorm / gemma_rmsnorm /
gemma_fused_add_rmsnorm can be traced by Dynamo with fullgraph=True
(i.e. no graph breaks, no posix.stat issues).
"""

import pytest
import torch
from torch.torch_version import TorchVersion
from torch.torch_version import __version__ as torch_version

if TorchVersion(torch_version) < TorchVersion("2.4"):
    pytest.skip("torch.compile support requires torch >= 2.4", allow_module_level=True)

import flashinfer


def _rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * w.float()).to(orig)


def _gemma_rmsnorm_ref(
    x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    orig = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * (w.float() + 1.0)).to(orig)


def _fused_add_rmsnorm_ref(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
):
    residual = residual + x
    orig = residual.dtype
    r = residual.float()
    r = r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps)
    return (r * w.float()).to(orig), residual


def _gemma_fused_add_rmsnorm_ref(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
):
    residual = residual + x
    orig = residual.dtype
    r = residual.float()
    r = r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps)
    return (r * (w.float() + 1.0)).to(orig), residual


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("hidden_size", [128, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_torch_compile(batch_size, hidden_size, dtype):
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    w = torch.ones(hidden_size, dtype=dtype, device="cuda")

    # warmup: trigger JIT compilation before torch.compile tracing
    flashinfer.rmsnorm(x, w)

    compiled = torch.compile(flashinfer.rmsnorm, fullgraph=True, backend="eager")
    out = compiled(x, w)
    ref = _rmsnorm_ref(x, w)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("hidden_size", [128, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rmsnorm_torch_compile(batch_size, hidden_size, dtype):
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    w = torch.ones(hidden_size, dtype=dtype, device="cuda")

    # warmup
    flashinfer.fused_add_rmsnorm(x.clone(), residual.clone(), w)

    def fn(x, residual, w):
        flashinfer.fused_add_rmsnorm(x, residual, w)
        return x, residual

    compiled = torch.compile(fn, fullgraph=True, backend="eager")
    x2, r2 = x.clone(), residual.clone()
    out_x, out_r = compiled(x2, r2, w)

    ref_x, ref_r = _fused_add_rmsnorm_ref(x.clone(), residual.clone(), w)
    torch.testing.assert_close(out_x, ref_x, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out_r, ref_r, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("hidden_size", [128, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemma_rmsnorm_torch_compile(batch_size, hidden_size, dtype):
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    w = torch.zeros(hidden_size, dtype=dtype, device="cuda")  # gemma: (w+1)*norm

    # warmup
    flashinfer.gemma_rmsnorm(x, w)

    compiled = torch.compile(flashinfer.gemma_rmsnorm, fullgraph=True, backend="eager")
    out = compiled(x, w)
    ref = _gemma_rmsnorm_ref(x, w)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("hidden_size", [128, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemma_fused_add_rmsnorm_torch_compile(batch_size, hidden_size, dtype):
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    w = torch.zeros(hidden_size, dtype=dtype, device="cuda")

    # warmup
    flashinfer.gemma_fused_add_rmsnorm(x.clone(), residual.clone(), w)

    def fn(x, residual, w):
        flashinfer.gemma_fused_add_rmsnorm(x, residual, w)
        return x, residual

    compiled = torch.compile(fn, fullgraph=True, backend="eager")
    x2, r2 = x.clone(), residual.clone()
    out_x, out_r = compiled(x2, r2, w)

    ref_x, ref_r = _gemma_fused_add_rmsnorm_ref(x.clone(), residual.clone(), w)
    torch.testing.assert_close(out_x, ref_x, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out_r, ref_r, atol=1e-2, rtol=1e-2)
