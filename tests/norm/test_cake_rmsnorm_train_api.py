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

Tests of the flashinfer public surface of the Cake fused BF16 RMSNorm
training kernels (SM100 / SM103 / SM107): the ``flashinfer.cake_rmsnorm_train``
facade and its re-export from the ``flashinfer`` top-level package.  The
facade's three entry points are pass-throughs to
``flashinfer.cake_rmsnorm_train_backend``; this module checks that identity
(same objects at the top level, same bits as the backend) and leaves the
numerical contract to ``test_cake_rmsnorm_train.py``, which imports only the
backend package.
"""

from __future__ import annotations

import pytest
import torch

import flashinfer
import flashinfer.cake_rmsnorm_train_backend as backend
from flashinfer.cake_rmsnorm_train import (
    CakeRMSNormFunction,
    cake_rmsnorm,
    cake_rmsnorm_train_backward,
    cake_rmsnorm_train_backward_workspace_bytes,
    cake_rmsnorm_train_forward,
    is_cake_rmsnorm_train_supported,
)
from flashinfer.cake_rmsnorm_train_backend import cake_jit as loader
from flashinfer.jit.env import has_flashinfer_jit_cache
from flashinfer.utils import get_compute_capability

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTHS = ((6144, 1e-5), (2048, 1e-6), (512, 1e-6))
HIDDEN_SIZES = tuple(hidden for hidden, _eps in WIDTHS)
EPS = {hidden: eps for hidden, eps in WIDTHS}
SEED = 760


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(_device()) in SUPPORTED_CAPABILITIES
        and is_cake_rmsnorm_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(get_compute_capability(_device()))
        flashinfer.jit.build_jit_specs(loader.specs_for_arch(arch), verbose=False)
    yield


def _skip_unless_hidden(hidden: int) -> None:
    if not is_cake_rmsnorm_train_supported(_device(), hidden):
        pytest.skip(f"hidden={hidden} has no exported route on this device")


def _randn(*shape: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        *shape, dtype=torch.float32, device=_device(), generator=generator
    ).to(torch.bfloat16)


def test_function_signature_is_exported() -> None:
    assert flashinfer.cake_rmsnorm is cake_rmsnorm
    assert flashinfer.cake_rmsnorm_train_forward is cake_rmsnorm_train_forward
    assert flashinfer.cake_rmsnorm_train_backward is cake_rmsnorm_train_backward
    assert (
        flashinfer.cake_rmsnorm_train_backward_workspace_bytes
        is cake_rmsnorm_train_backward_workspace_bytes
    )
    assert flashinfer.CakeRMSNormFunction is CakeRMSNormFunction


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_facade_entry_points_match_the_backend(hidden: int) -> None:
    """The facade forwards every argument: its forward, backward and autograd
    entry points return the same bits as the backend package's."""
    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    generator = torch.Generator(device=_device()).manual_seed(SEED + hidden)
    x = _randn(129, hidden, generator=generator)
    g = _randn(129, hidden, generator=generator)
    w = _randn(hidden, generator=generator)
    y, rstd = cake_rmsnorm_train_forward(x, w, eps)
    y_backend, rstd_backend = backend.cake_rmsnorm_train_forward(x, w, eps)
    assert torch.equal(y, y_backend) and torch.equal(rstd, rstd_backend)
    dx, dw = cake_rmsnorm_train_backward(g, x, w, rstd)
    dx_backend, dw_backend = backend.cake_rmsnorm_train_backward(g, x, w, rstd)
    assert torch.equal(dx, dx_backend) and torch.equal(dw, dw_backend)
    assert torch.equal(cake_rmsnorm(x, w, eps), backend.cake_rmsnorm(x, w, eps))
