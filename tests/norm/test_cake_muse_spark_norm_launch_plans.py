# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""CPU tests of the launch plans of ``flashinfer.cake_muse_spark_norm_backend``: a repeated call of one signature
selects its record, resolves its grid and binds its entry once, a new signature prepares its own plan, a contract
breach after a plan exists still raises the package's error, and the plan's argument order is the loader's."""

import pytest
import torch

import flashinfer.cake_muse_spark_norm_backend as pkg
from flashinfer.cake_muse_spark_norm_backend import cake_jit as loader

HIDDEN_SIZE = pkg.HIDDEN_SIZE
HEAD_DIM = pkg.HEAD_DIM
ARCH = "sm_103a"


class _Module:
    """Stands in for a loaded tvm-ffi module: every attribute is a callable that records its arguments."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def entry(*args):
            self.calls.append((name, args))
            return 1

        return entry


@pytest.fixture
def plans(monkeypatch):
    """The package on a CPU device that pretends to be an exported architecture, with the loaded module, the
    launch and the device index stubbed, and the loader's record selection counted."""
    pkg.clear_launch_plans()
    module = _Module()
    monkeypatch.setattr(pkg, "_arch", lambda device: ARCH)
    monkeypatch.setattr(pkg, "_device_index", lambda device: 0)
    monkeypatch.setattr(loader, "load", lambda name, arch: module)
    monkeypatch.setattr(loader, "device_arch", lambda index: ARCH)
    launched = []
    monkeypatch.setattr(
        pkg, "_launch", lambda plan, args: launched.append((plan.name, args))
    )
    selections = []
    original = loader.selected_module

    def counted(*args, **kwargs):
        selections.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(loader, "selected_module", counted)
    yield module, launched, selections
    pkg.clear_launch_plans()
    loader.cluster_capacity.cache_clear()  # nothing the stub module answered survives this test


def _gated_inputs(rows, weight=False):
    branch = torch.zeros(rows, HIDDEN_SIZE, dtype=torch.bfloat16)
    residual = torch.zeros(rows, HIDDEN_SIZE, dtype=torch.float32)
    alpha = torch.ones(HIDDEN_SIZE, dtype=torch.float32)
    beta = torch.ones(HIDDEN_SIZE, dtype=torch.float32)
    out = torch.empty_like(residual)
    w = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16) if weight else None
    return branch, residual, w, alpha, beta, out


def test_a_repeated_gated_residual_rmsnorm_call_reuses_its_plan(plans):
    module, launched, selections = plans
    branch, residual, _, alpha, beta, out = _gated_inputs(64)
    for eps in (1e-5, 1e-3, 1e-5):
        pkg.cake_gated_residual_rmsnorm(
            branch, residual, None, alpha, beta, out, eps=eps
        )
    assert len(selections) == 1 and len(launched) == 3
    route = pkg.gated_residual_rmsnorm_route(False, False)
    name = loader.selected_module(ARCH, route, 64)
    assert [n for n, _ in launched] == [name] * 3
    # every launch carries the call's own scalars and tensors
    assert 1e-3 in launched[1][1] and 1e-3 not in launched[0][1]
    assert any(t is out for t in launched[0][1] if isinstance(t, torch.Tensor))
    # the mode-0 weight slot is unread: branch itself keeps the BF16 tensor ABI
    assert sum(t is branch for t in launched[0][1] if isinstance(t, torch.Tensor)) == 2
    # another signature prepares its own plan; the first stays
    branch1, residual1, weight, alpha1, beta1, out1 = _gated_inputs(64, weight=True)
    pkg.cake_gated_residual_rmsnorm(
        branch1, residual1, weight, alpha1, beta1, out1, eps=1e-5, gain_center=1.0
    )
    assert len(selections) == 3  # the plan's selection plus this test's own call above
    pkg.cake_gated_residual_rmsnorm(branch, residual, None, alpha, beta, out, eps=1e-5)
    assert len(selections) == 3 and len(launched) == 5
    # a contract breach after a plan exists raises the package's error and launches nothing
    with pytest.raises(ValueError, match="residual must be"):
        pkg.cake_gated_residual_rmsnorm(
            branch, residual.half(), None, alpha, beta, out, eps=1e-5
        )
    with pytest.raises(ValueError, match="out must be"):
        pkg.cake_gated_residual_rmsnorm(
            branch, residual, None, alpha, beta, out[:32], eps=1e-5
        )
    assert len(launched) == 5
    # an empty batch launches nothing and prepares nothing
    pkg.cake_gated_residual_rmsnorm(*_gated_inputs(0), eps=1e-5)
    assert len(launched) == 5 and len(selections) == 3
    # forgetting the plans selects again
    pkg.clear_launch_plans()
    pkg.cake_gated_residual_rmsnorm(branch, residual, None, alpha, beta, out, eps=1e-5)
    assert len(selections) == 4


def test_a_plan_orders_its_arguments_like_the_loader(plans):
    module, launched, selections = plans
    branch, residual, weight, alpha, beta, out = _gated_inputs(64, weight=True)
    pkg.cake_gated_residual_rmsnorm(
        branch, residual, weight, alpha, beta, out, eps=1e-5, gain_center=1.0
    )
    (plan,) = [p for p in pkg._LAUNCH_PLANS.values()]
    record = loader.MODULES[plan.name]
    values = {
        "branch": branch,
        "residual": residual,
        "weight": weight,
        "alpha": alpha,
        "beta": beta,
        "out": out,
        "rows": 64,
        "eps": 1e-5,
        "gain_center": 1.0,
    }
    assert plan.args(values) == loader.launch_args(record, values, plan.grid)
    assert plan.grid == loader.launch_grid(record, 64, device_index=0, values=values)
    assert launched[0][1] == plan.args(values)


def test_a_repeated_rmsnorm_sigmoid_gate_call_reuses_its_plan(plans):
    module, launched, selections = plans
    tokens, heads = 64, 128
    projection = torch.zeros(tokens, 3 * heads * HEAD_DIM, dtype=torch.bfloat16)
    gate = projection[:, : heads * HEAD_DIM].view(tokens, heads, HEAD_DIM)
    x = torch.zeros(tokens, heads, HEAD_DIM, dtype=torch.bfloat16)
    for _ in range(3):
        pkg.cake_rmsnorm_sigmoid_gate(x, gate, eps=1e-5)
    assert len(selections) == 1 and len(launched) == 3
    name = loader.selected_module(ARCH, pkg.rmsnorm_sigmoid_gate_route(), tokens, heads)
    assert [n for n, _ in launched] == [name] * 3
    args = launched[0][1]
    assert gate.stride(0) in args and x.stride(0) in args and tokens in args
    # the token stride is part of the signature: a contiguous gate is another plan
    pkg.cake_rmsnorm_sigmoid_gate(x, gate.contiguous(), eps=1e-5)
    assert len(selections) == 3  # the plan's selection plus this test's own call above
    # a 2-D call is one head, which no route of the architecture names: no plan, the package's error
    with pytest.raises(ValueError, match="admits 64 tokens of 1 heads"):
        pkg.cake_rmsnorm_sigmoid_gate(
            x.view(-1, HEAD_DIM)[:tokens].clone(),
            gate.reshape(-1, HEAD_DIM)[:tokens].clone(),
            eps=1e-5,
        )
    # a contract breach after a plan exists raises the package's error
    with pytest.raises(ValueError, match="gate must have the shape"):
        pkg.cake_rmsnorm_sigmoid_gate(x, gate[:1], eps=1e-5)
    with pytest.raises(ValueError, match="must be a bfloat16"):
        pkg.cake_rmsnorm_sigmoid_gate(x, gate.float(), eps=1e-5)
    assert len(launched) == 4
