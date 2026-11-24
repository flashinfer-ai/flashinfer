# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""CPU tests of the launch plans of ``flashinfer.cake_muse_spark_norm_backend``: a repeated call of one signature
selects its record, resolves its grid and binds its entry once, a new signature (the launch form among its keys)
prepares its own plan, a contract breach after a plan exists still raises the package's error, a PDL route's window
naming the plain build is that record's plan, and the plan's argument order is the loader's."""

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


def _scalars(args):
    """The non-tensor entries of a launch's positional arguments."""
    return [a for a in args if not isinstance(a, torch.Tensor)]


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
    assert 1e-3 in _scalars(launched[1][1]) and 1e-3 not in _scalars(launched[0][1])
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


def test_a_same_rows_other_hidden_size_takes_the_slow_path(plans):
    """The plan key carries the row count, not the hidden size: a later call with the admitted row count and
    another hidden size (or, for the sigmoid gate, the same strides and fewer columns) must not reuse the plan."""
    module, launched, selections = plans
    branch, residual, _, alpha, beta, out = _gated_inputs(64)
    pkg.cake_gated_residual_rmsnorm(branch, residual, None, alpha, beta, out, eps=1e-5)
    assert len(selections) == 1 and len(launched) == 1
    narrow = torch.zeros(64, HIDDEN_SIZE // 2, dtype=torch.bfloat16)
    narrow_f32 = torch.zeros(64, HIDDEN_SIZE // 2, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"branch must be \[rows, 8192\]"):
        pkg.cake_gated_residual_rmsnorm(
            narrow,
            narrow_f32,
            None,
            alpha,
            beta,
            torch.empty_like(narrow_f32),
            eps=1e-5,
        )
    assert len(selections) == 1 and len(launched) == 1
    tokens, heads = 64, 128
    x = torch.zeros(tokens, heads, HEAD_DIM, dtype=torch.bfloat16)
    gate = torch.zeros(tokens, heads, HEAD_DIM, dtype=torch.bfloat16)
    pkg.cake_rmsnorm_sigmoid_gate(x, gate, eps=1e-5)
    assert len(selections) == 2 and len(launched) == 2
    # the same token and head strides with half the columns: the plan's key matches, the contract does not
    half = slice(None, HEAD_DIM // 2)
    with pytest.raises(
        ValueError, match=r"must be \[tokens, 64\] or \[tokens, heads, 64\]"
    ):
        pkg.cake_rmsnorm_sigmoid_gate(x[:, :, half], gate[:, :, half], eps=1e-5)
    assert len(selections) == 2 and len(launched) == 2


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
    assert plan.pack(tuple(values.values()) + plan.constants) == plan.args(values)


def test_a_repeated_rmsnorm_sigmoid_gate_call_reuses_its_plan(plans):
    module, launched, selections = plans
    tokens, heads = 32, 128
    projection = torch.zeros(tokens, 3 * heads * HEAD_DIM, dtype=torch.bfloat16)
    gate = projection[:, : heads * HEAD_DIM].view(tokens, heads, HEAD_DIM)
    x = torch.zeros(tokens, heads, HEAD_DIM, dtype=torch.bfloat16)
    for _ in range(3):
        pkg.cake_rmsnorm_sigmoid_gate(x, gate, eps=1e-5)
    assert len(selections) == 1 and len(launched) == 3
    name = loader.selected_module(ARCH, pkg.rmsnorm_sigmoid_gate_route(), tokens, heads)
    assert [n for n, _ in launched] == [name] * 3
    scalars = _scalars(launched[0][1])
    assert gate.stride(0) in scalars and x.stride(0) in scalars and tokens in scalars
    # the token stride is part of the signature: a contiguous gate is another plan
    pkg.cake_rmsnorm_sigmoid_gate(x, gate.contiguous(), eps=1e-5)
    assert len(selections) == 3  # the plan's selection plus this test's own call above
    # a 2-D call is one head, which no route of the architecture names: no plan, the package's error
    with pytest.raises(ValueError, match="admits 32 tokens of 1 heads"):
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
    # the launch form is part of the signature: the PDL route prepares a plan of its own, keyed by enable_pdl
    before_selections, before_launches = len(selections), len(launched)
    for _ in range(3):
        pkg.cake_rmsnorm_sigmoid_gate(x, gate, eps=1e-5, enable_pdl=True)
    assert len(selections) == before_selections + 1
    assert len(launched) == before_launches + 3
    pdl_route = pkg.rmsnorm_sigmoid_gate_route(True)
    name_pdl = loader.selected_module(ARCH, pdl_route, tokens, heads)
    assert [n for n, _ in launched[-3:]] == [name_pdl] * 3 and name_pdl != name
    assert loader.MODULES[name_pdl]["launch"]["use_pdl"] is True
    assert loader.MODULES[name_pdl]["build"] == {
        **loader.MODULES[name]["build"],
        "pdl": True,
    }
    # a window of the PDL route naming the plain build: the plain record's plan, launched without the attribute
    tokens_plain, heads_plain = 1025, 16
    projection_plain = torch.zeros(
        tokens_plain, 3 * heads_plain * HEAD_DIM, dtype=torch.bfloat16
    )
    gate_plain = projection_plain[:, : heads_plain * HEAD_DIM].view(
        tokens_plain, heads_plain, HEAD_DIM
    )
    x_plain = torch.zeros(tokens_plain, heads_plain, HEAD_DIM, dtype=torch.bfloat16)
    pkg.cake_rmsnorm_sigmoid_gate(x_plain, gate_plain, eps=1e-5, enable_pdl=True)
    borrowed = launched[-1][0]
    assert borrowed == loader.selected_module(
        ARCH, pdl_route, tokens_plain, heads_plain
    )
    assert borrowed == loader.selected_module(
        ARCH, pkg.rmsnorm_sigmoid_gate_route(), tokens_plain, heads_plain
    )
    assert loader.MODULES[borrowed]["launch"]["use_pdl"] is False
    assert loader.MODULES[borrowed]["build"]["pdl"] is False
    # the PDL route's first window of 128 heads is a declared gap (one token: the CuTe DSL kernel under PDL wins
    # there), so the PDL call is refused and the plain call runs the program the plain route selects; two tokens
    # select the PDL build of that program
    one = torch.zeros(1, 128, HEAD_DIM, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="admits 1 tokens of 128 heads"):
        pkg.cake_rmsnorm_sigmoid_gate(one, one.clone(), eps=1e-5, enable_pdl=True)
    with pytest.raises(NotImplementedError, match="admits 1 rows"):
        loader.selected_module(ARCH, pdl_route, 1, 128)
    two = torch.zeros(2, 128, HEAD_DIM, dtype=torch.bfloat16)
    pkg.cake_rmsnorm_sigmoid_gate(two, two.clone(), eps=1e-5, enable_pdl=True)
    first_pdl = launched[-1][0]
    assert first_pdl == loader.selected_module(ARCH, pdl_route, 2, 128)
    assert loader.MODULES[first_pdl]["launch"]["use_pdl"] is True
    pkg.cake_rmsnorm_sigmoid_gate(one, one.clone(), eps=1e-5)
    first_plain = launched[-1][0]
    assert first_plain == loader.selected_module(
        ARCH, pkg.rmsnorm_sigmoid_gate_route(), 1, 128
    )
    assert loader.MODULES[first_plain]["launch"]["use_pdl"] is False
    assert loader.MODULES[first_pdl]["build"] == {
        **loader.MODULES[first_plain]["build"],
        "pdl": True,
    }


def test_the_admit_index_admits_exactly_the_row_counts_the_loader_scan_admits():
    """``is_cake_*_supported`` answers a call's row count from the package's admit index (one bisection); the loader's
    record-by-record ``admits`` is the reference, over every row or token count up to 20000 of every route of every
    architecture -- the sigmoid gate per head count its route names and one it does not -- and over the calls
    without a row count (route available; head count named)."""
    counts = range(1, 20001)
    for arch in loader.ARCHES:
        for route, names in loader.ROUTES[arch].items():
            rules = [loader.MODULES[name].get("select_rule") for name in names]
            assert pkg._admits(arch, route, None) is loader.route_available(
                arch, route
            ), (arch, route)
            if any(
                rule is not None and rule["kind"] != loader.SELECT_KIND
                for rule in rules
            ):
                named = sorted(
                    {
                        int(window[0])
                        for rule in rules
                        if rule is not None
                        for window in rule.get("routes", {})
                        .get(route, rule["windows"])
                        .get(arch, ())
                    }
                )
                for heads in (*named, 64):
                    scan = [
                        any(
                            loader.admits(rule, tokens, arch, heads, route)
                            for rule in rules
                        )
                        for tokens in counts
                    ]
                    index = [
                        pkg._admits(arch, route, tokens, heads) for tokens in counts
                    ]
                    assert index == scan, (arch, route, heads)
                    assert pkg._admits(arch, route, None, heads) is (heads in named), (
                        arch,
                        route,
                        heads,
                    )
            else:
                scan = [
                    any(loader.admits(rule, rows, arch, route=route) for rule in rules)
                    for rows in counts
                ]
                index = [pkg._admits(arch, route, rows) for rows in counts]
                assert index == scan, (arch, route)
    assert not pkg._admits("sm_100a", "no_such_route", 1)
    assert not pkg._admits("sm_100a", "no_such_route", None)
