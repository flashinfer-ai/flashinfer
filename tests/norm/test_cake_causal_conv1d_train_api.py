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

Tests of the flashinfer public surface of the Cake packed width-4 depthwise
causal convolution + SiLU training kernels (SM100 / SM103 / SM107): the
``flashinfer.cake_causal_conv1d_train`` facade and its re-export from the
``flashinfer`` top-level package.  The facade's three entry points are
pass-throughs to ``flashinfer.cake_causal_conv1d_train_backend``; this module
checks that identity (same objects at the top level, the backend's exact
signatures, same bits as the backend) and leaves the numerical contract to
``test_cake_causal_conv1d_train.py``, which imports only the backend package.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import flashinfer
import flashinfer.cake_causal_conv1d_train
import flashinfer.cake_causal_conv1d_train_backend as backend
from flashinfer.cake_causal_conv1d_train import (
    CakeCausalConv1dFunction,
    cake_causal_conv1d,
    cake_causal_conv1d_train_backward,
    cake_causal_conv1d_train_backward_workspace_bytes,
    cake_causal_conv1d_train_forward,
    cake_causal_conv1d_train_forward_workspace,
    cake_causal_conv1d_train_forward_workspace_bytes,
    is_cake_causal_conv1d_train_supported,
)
from flashinfer.cake_causal_conv1d_train_backend import cake_jit as loader
from flashinfer.jit.env import has_flashinfer_jit_cache
from flashinfer.utils import get_compute_capability

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
SEED = 2033


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(_device()) in SUPPORTED_CAPABILITIES
        and is_cake_causal_conv1d_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_causal_conv1d_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(get_compute_capability(_device()))
        flashinfer.jit.build_jit_specs(loader.specs_for_arch(arch), verbose=False)
    yield


def test_function_signature_is_exported() -> None:
    assert flashinfer.cake_causal_conv1d is cake_causal_conv1d
    assert (
        flashinfer.cake_causal_conv1d_train_forward is cake_causal_conv1d_train_forward
    )
    assert (
        flashinfer.cake_causal_conv1d_train_backward
        is cake_causal_conv1d_train_backward
    )
    assert (
        flashinfer.cake_causal_conv1d_train_backward_workspace_bytes
        is cake_causal_conv1d_train_backward_workspace_bytes
    )
    assert (
        flashinfer.cake_causal_conv1d_train_forward_workspace_bytes
        is cake_causal_conv1d_train_forward_workspace_bytes
        is backend.cake_causal_conv1d_train_forward_workspace_bytes
    )
    assert (
        cake_causal_conv1d_train_forward_workspace
        is backend.cake_causal_conv1d_train_forward_workspace
    )
    assert flashinfer.CakeCausalConv1dFunction is CakeCausalConv1dFunction


def test_facade_signatures_mirror_the_backend() -> None:
    """Every keyword of the backend's entry points (``bias``, ``silu_grad_preactivation``, ``dbias``, ...) is
    spelled identically on the ``@flashinfer_api`` facade, which public-API tooling reads."""

    for name in (
        "cake_causal_conv1d_train_forward",
        "cake_causal_conv1d_train_backward",
        "cake_causal_conv1d",
    ):
        facade = inspect.signature(getattr(flashinfer.cake_causal_conv1d_train, name))
        backend_signature = inspect.signature(getattr(backend, name))
        assert facade == backend_signature, (name, facade, backend_signature)
    backward = inspect.signature(cake_causal_conv1d_train_backward).parameters
    assert backward["bias"].default is None
    assert backward["silu_grad_preactivation"].default == "bf16"
    assert backward["dbias"].default is None
    assert backward["bias"].kind is inspect.Parameter.KEYWORD_ONLY
    autograd = inspect.signature(cake_causal_conv1d).parameters
    assert autograd["bias"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert autograd["silu_grad_preactivation"].kind is inspect.Parameter.KEYWORD_ONLY
    assert backend.SILU_GRAD_PREACTIVATIONS == ("bf16", "fp32")


def test_facade_entry_points_match_the_backend() -> None:
    """The facade forwards every argument: its forward, backward and autograd
    entry points return the same bits as the backend package's."""

    S, C = 1000, 512
    generator = torch.Generator(device=_device()).manual_seed(SEED)
    x = torch.randn(
        S, C, dtype=torch.float32, device=_device(), generator=generator
    ).to(torch.bfloat16)
    dy = torch.randn(
        S, C, dtype=torch.float32, device=_device(), generator=generator
    ).to(torch.bfloat16)
    w = 0.5 * torch.randn(
        C, 4, dtype=torch.float32, device=_device(), generator=generator
    )
    cu = torch.tensor([0, 369, 659, 868, S], dtype=torch.int32, device=_device())
    y = cake_causal_conv1d_train_forward(x, w, cu)
    assert torch.equal(y, backend.cake_causal_conv1d_train_forward(x, w, cu))
    workspace = cake_causal_conv1d_train_forward_workspace(S, C, _device())
    assert torch.equal(
        y, cake_causal_conv1d_train_forward(x, w, cu, workspace=workspace)
    )
    assert int(torch.count_nonzero(workspace)) == 0
    dx, dw = cake_causal_conv1d_train_backward(x, w, cu, dy)
    dx_backend, dw_backend = backend.cake_causal_conv1d_train_backward(x, w, cu, dy)
    assert torch.equal(dx, dx_backend) and torch.equal(dw, dw_backend)
    assert torch.equal(
        cake_causal_conv1d(x, w, cu), backend.cake_causal_conv1d(x, w, cu)
    )
