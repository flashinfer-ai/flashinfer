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

Tests of the flashinfer public surface of the Cake SiLU-gated RMSNorm training
kernels (SM100 / SM103 / SM107): the ``flashinfer.cake_rmsnorm_silu_gate_train``
facade and its re-export from the ``flashinfer`` top-level package.  The
facade's entry points are pass-throughs to
``flashinfer.cake_rmsnorm_silu_gate_train_backend``; this module checks that
identity (same objects at the top level, same bits as the backend) and leaves
the numerical contract to ``test_cake_rmsnorm_silu_gate_train.py``.
"""

from __future__ import annotations

import pytest
import torch

import flashinfer
import flashinfer.cake_rmsnorm_silu_gate_train_backend as backend
from flashinfer.cake_rmsnorm_silu_gate_train import (
    CakeRMSNormSiLUGateFunction,
    cake_rmsnorm_silu_gate,
    cake_rmsnorm_silu_gate_train_backward,
    cake_rmsnorm_silu_gate_train_backward_workspace_bytes,
    cake_rmsnorm_silu_gate_train_forward,
    is_cake_rmsnorm_silu_gate_train_supported,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cake_rmsnorm_silu_gate_train_supported(torch.device("cuda")),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


def test_top_level_exports_are_the_facade():
    assert flashinfer.cake_rmsnorm_silu_gate is cake_rmsnorm_silu_gate
    assert (
        flashinfer.cake_rmsnorm_silu_gate_train_forward
        is cake_rmsnorm_silu_gate_train_forward
    )
    assert (
        flashinfer.cake_rmsnorm_silu_gate_train_backward
        is cake_rmsnorm_silu_gate_train_backward
    )
    assert (
        flashinfer.cake_rmsnorm_silu_gate_train_backward_workspace_bytes
        is cake_rmsnorm_silu_gate_train_backward_workspace_bytes
    )
    assert flashinfer.CakeRMSNormSiLUGateFunction is CakeRMSNormSiLUGateFunction
    assert CakeRMSNormSiLUGateFunction is backend.CakeRMSNormSiLUGateFunction


def test_facade_matches_backend_bits():
    g = torch.Generator(device="cuda").manual_seed(11)
    x, z, dy = (
        torch.randn(129, 48, 128, device="cuda", generator=g).bfloat16()
        for _ in range(3)
    )
    w = (1.0 + 0.5 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    y = cake_rmsnorm_silu_gate_train_forward(x, z, w, 1e-6)
    assert torch.equal(y, backend.cake_rmsnorm_silu_gate_train_forward(x, z, w, 1e-6))
    got = cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, 1e-6)
    want = backend.cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, 1e-6)
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    ya = cake_rmsnorm_silu_gate(xa, za, wa, 1e-6)
    ya.backward(dy)
    assert torch.equal(ya, y)
    assert (
        torch.equal(xa.grad, got[0])
        and torch.equal(za.grad, got[1])
        and torch.equal(wa.grad, got[2])
    )
