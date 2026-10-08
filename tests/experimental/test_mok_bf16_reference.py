# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The toy reference's expert weight gradients must keep every routed row.

torch's BF16 ``grad_out.T @ x`` (K = routed row count) silently drops the last
``K mod 256`` rows for some K on current cuBLAS builds (seen for K = 9473..9475
and 9730 on sm_100a, sm_103a and sm_107a).  ``mok_bf16_toy.expert`` therefore
forms its weight gradients as FP32 products; this test pins that behaviour by
zeroing every row except the tail, so a dropped tail yields a zero gradient.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch


def _load_expert():
    path = Path(__file__).resolve().parents[2] / "examples" / "mok_bf16_toy.py"
    spec = importlib.util.spec_from_file_location("mok_reference_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.expert


def _bf16(value):
    return value.to(torch.bfloat16).double()


@pytest.mark.parametrize("rows", [9473, 9474, 9475, 9730, 9476])
def test_expert_weight_gradients_keep_the_row_tail(rows):
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA device")
    expert = _load_expert()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    gen = torch.Generator(device=device).manual_seed(rows)
    hidden = intermediate = 256
    tail = rows % 256
    x = torch.zeros(rows, hidden, device=device, dtype=torch.bfloat16)
    x[rows - tail :] = (
        torch.randn(tail, hidden, device=device, generator=gen) * 0.125
    ).to(torch.bfloat16)
    dy = (torch.randn(rows, hidden, device=device, generator=gen) * 0.125).to(
        torch.bfloat16
    )
    shapes = [(intermediate, hidden), (intermediate, hidden), (hidden, intermediate)]
    weights = [
        (torch.randn(*shape, device=device, generator=gen) / shape[1] ** 0.5)
        .to(torch.bfloat16)
        .requires_grad_(True)
        for shape in shapes
    ]
    out = expert(x, *weights)
    actual = torch.autograd.grad(out, weights, dy)

    # FP64 oracle over the tail rows with the reference's BF16 rounding points.
    xt, dyt = x[rows - tail :].double(), dy[rows - tail :].double()
    gate, up, down = (w.detach().double() for w in weights)
    a, b = _bf16(xt @ gate.T), _bf16(xt @ up.T)
    h = _bf16(torch.nn.functional.silu(a) * b)
    dh = _bf16(dyt @ down)
    sig = torch.sigmoid(a)
    dsilu = sig * (1 + a * (1 - sig))
    dg, du = _bf16(dh * b * dsilu), _bf16(dh * torch.nn.functional.silu(a))
    expected = (dg.T @ xt, du.T @ xt, dyt.T @ h)
    for name, got, want in zip(("gate", "up", "down"), actual, expected, strict=True):
        error = float((got.double() - want).norm() / want.norm())
        assert want.norm() > 0
        assert error <= 1e-2, (
            f"{name} weight gradient lost the {tail}-row tail: rel {error:.3e}"
        )
