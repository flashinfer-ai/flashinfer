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

"""Dispatch-logic tests for the "specialized" SM120 NVFP4-W4A16 MoE backend.

These tests exercise the routing decisions only — which backend a call
reaches under ``auto``, the kill switch, the allowlist, and the
never-swap-explicit rule — through BOTH reachability paths:

  * the flat entry (``b12x_fused_moe``) / ``B12xMoEWrapper`` directly, and
  * the unified ``MoELayer`` path (``BackendOptions((B12xW4A16Config(),))``
    -> ``B12xW4A16Runner`` -> ``B12xMoEWrapper.run``).

The kernel launches and the stock SM12x machinery are monkeypatched, and the
SM120 device/toolkit probes are mocked, so everything here runs CPU-side on
any machine (no GPU, no CuTe DSL, no sm120 silicon). Numeric correctness and
real-silicon reachability live in ``test_specialized_nvfp4_w4a16_sm120.py``.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe import specialized_moe_nvfp4_w4a16_sm120 as spec

E, TOPK, H, I = 128, 6, 2688, 1856
DISPATCH_PKG = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x"


# ---------------------------------------------------------------------------
# Environment mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def record():
    return {"spec_calls": 0, "stock_calls": 0}


@pytest.fixture
def sm120_env(monkeypatch, record):
    """Mock the SM120 device/toolkit probes and the kernel launches.

    The specialized guard's own logic (allowlist, layout, kill switch,
    never-swap) stays real; only what needs silicon or the CuTe-DSL stack is
    replaced.
    """
    # --- SM120 device + CUDA >= 13.1 toolkit probes
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    monkeypatch.setattr(
        "flashinfer.jit.cpp_ext.get_cuda_version",
        lambda: SimpleNamespace(major=13, minor=1),
    )
    # Unified-runner environment probes (resolved at call time).
    monkeypatch.setattr(
        "flashinfer.utils.get_compute_capability", lambda device: (12, 0)
    )
    monkeypatch.setattr(
        "flashinfer.fused_moe.layer.get_compute_capability", lambda device: (12, 0)
    )
    monkeypatch.setattr("flashinfer.cute_dsl.is_cute_dsl_available", lambda: True)
    # Without the CuTe-DSL package installed, fused_moe.cute_dsl's __init__
    # does not re-export B12xMoEWrapper (evaluated at package import); bind
    # it so B12xW4A16Runner._build's `from .cute_dsl import B12xMoEWrapper`
    # resolves. A no-op rebinding in a full environment.
    import flashinfer.fused_moe.cute_dsl as _fm_cute_dsl
    from flashinfer.fused_moe.cute_dsl.b12x_moe import B12xMoEWrapper

    monkeypatch.setattr(_fm_cute_dsl, "B12xMoEWrapper", B12xMoEWrapper, raising=False)

    # --- specialized backend: fake the JIT module and the kernel launch
    monkeypatch.setattr(spec, "_module", lambda: object())

    def fake_specialized(x, w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha, ids, tw, output):
        record["spec_calls"] += 1
        output.fill_(7.0)
        return output

    monkeypatch.setattr(spec, "run_specialized_moe_nvfp4_w4a16", fake_specialized)

    # --- stock SM12x machinery: a stub module (the real one needs the CUDA
    # python bindings and CuTe DSL). All call sites import it lazily, so a
    # sys.modules override intercepts them.
    stub = types.ModuleType(f"{DISPATCH_PKG}.moe_dispatch")

    def _normalize_quant_mode(quant_mode, activation_precision):
        if quant_mode is not None:
            return "nvfp4" if quant_mode == "w4a4" else quant_mode
        return "nvfp4" if activation_precision == "fp4" else "w4a16"

    def launch_sm120_moe(**kwargs):
        record["stock_calls"] += 1
        return kwargs["scatter_output"]

    stub._normalize_quant_mode = _normalize_quant_mode
    stub._activation_precision_from_quant_mode = (
        lambda quant_mode: "bf16" if quant_mode == "w4a16" else "fp4"
    )
    stub._validate_w4a4_dimensions = lambda *args, **kwargs: None
    stub.launch_sm120_moe = launch_sm120_moe
    stub.allocate_sm120_moe_workspace = lambda **kwargs: object()
    stub.select_sm120_moe_backend = lambda **kwargs: "static"
    stub._prepare_weight_views = lambda **kwargs: None
    stub._get_static_compact_cutover_pairs = lambda **kwargs: 0
    pkg = types.ModuleType(DISPATCH_PKG)
    pkg.moe_dispatch = stub
    monkeypatch.setitem(sys.modules, DISPATCH_PKG, pkg)
    monkeypatch.setitem(sys.modules, f"{DISPATCH_PKG}.moe_dispatch", stub)
    return record


@pytest.fixture(scope="module")
def weights():
    """Contract-shaped CPU weight views (contents never read: launches are
    mocked, only the guard's shape/dtype/layout checks consume these)."""
    return {
        "w1_weight": torch.empty(E, I, H // 2, dtype=torch.uint8),
        "w1_weight_sf": torch.empty(E, I, H // 16, dtype=torch.float8_e4m3fn),
        "w1_alpha": torch.ones(E, dtype=torch.float32),
        "w2_weight": torch.empty(E, H, I // 2, dtype=torch.uint8),
        "w2_weight_sf": torch.empty(E, H, I // 16, dtype=torch.float8_e4m3fn),
        "w2_alpha": torch.ones(E, dtype=torch.float32),
    }


def _activations(t):
    x = torch.zeros(t, H, dtype=torch.bfloat16)
    ids = torch.zeros(t, TOPK, dtype=torch.int32)
    tw = torch.full((t, TOPK), 1.0 / TOPK, dtype=torch.float32)
    return x, ids, tw


# ---------------------------------------------------------------------------
# The shared dispatch gate (maybe_run_specialized)
# ---------------------------------------------------------------------------


class TestSharedDispatchGate:
    def _gate(self, backend, t, weights, **overrides):
        x, ids, tw = _activations(t)
        kwargs = dict(
            w1_alpha=weights["w1_alpha"],
            w2_alpha=weights["w2_alpha"],
            output=torch.empty(t, H, dtype=torch.bfloat16),
            activation="relu2",
            quant_mode="w4a16",
            source_format="modelopt",
        )
        kwargs.update(overrides)
        return spec.maybe_run_specialized(
            backend,
            x,
            weights["w1_weight"],
            weights["w1_weight_sf"],
            weights["w2_weight"],
            weights["w2_weight_sf"],
            ids,
            tw,
            E,
            TOPK,
            **kwargs,
        )

    def test_canonical_backends_reexported(self):
        from flashinfer.fused_moe.cute_dsl.b12x_moe import BACKENDS

        assert BACKENDS is spec.BACKENDS
        assert BACKENDS == ("auto", "specialized", "b12x")

    def test_bogus_backend_raises(self, sm120_env, weights):
        with pytest.raises(ValueError, match="backend must be one of"):
            self._gate("fa2", 16, weights)

    def test_auto_on_allowlist_serves_specialized(self, sm120_env, weights):
        out = self._gate("auto", 16, weights)
        assert out is not None and bool((out == 7.0).all())
        assert sm120_env["spec_calls"] == 1

    def test_auto_off_allowlist_declines_with_reason(self, sm120_env, weights):
        before_b12x = spec.dispatch_stats.get("b12x", 0)
        assert self._gate("auto", 8, weights) is None
        assert sm120_env["spec_calls"] == 0
        assert spec.dispatch_stats.get("b12x", 0) == before_b12x + 1
        assert any(
            any(r.startswith("token_count") for r in reasons)
            for reasons in spec.fallback_reasons
        )

    def test_explicit_stock_backend_skips_the_guard_entirely(self, sm120_env, weights):
        n_reasons = len(spec.fallback_reasons)
        assert self._gate("b12x", 16, weights) is None
        assert sm120_env["spec_calls"] == 0
        assert len(spec.fallback_reasons) == n_reasons

    def test_explicit_specialized_raises_instead_of_swapping(self, sm120_env, weights):
        with pytest.raises(ValueError, match="token_count"):
            self._gate("specialized", 8, weights)
        assert sm120_env["spec_calls"] == 0

    def test_kill_switch(self, sm120_env, weights, monkeypatch):
        monkeypatch.setenv(spec.ENV_DISABLE, "1")
        assert self._gate("auto", 16, weights) is None
        with pytest.raises(ValueError, match="kill_switch"):
            self._gate("specialized", 16, weights)
        assert sm120_env["spec_calls"] == 0

    def test_expert_parallel_declines(self, sm120_env, weights):
        assert self._gate("auto", 16, weights, expert_parallel=True) is None
        assert sm120_env["spec_calls"] == 0
        assert ("expert_parallelism",) in spec.fallback_reasons

    def test_missing_alpha_declines(self, sm120_env, weights):
        assert self._gate("auto", 16, weights, w1_alpha=None) is None
        assert sm120_env["spec_calls"] == 0
        assert any("w1_alpha_missing" in reasons for reasons in spec.fallback_reasons)


# ---------------------------------------------------------------------------
# B12xMoEWrapper reachability (the hook the unified runner traverses)
# ---------------------------------------------------------------------------


def _make_wrapper(**overrides):
    from flashinfer.fused_moe.cute_dsl.b12x_moe import B12xMoEWrapper

    kwargs = dict(
        num_experts=E,
        top_k=TOPK,
        hidden_size=H,
        intermediate_size=I,
        device="cpu",
        activation="relu2",
        quant_mode="w4a16",
    )
    kwargs.update(overrides)
    return B12xMoEWrapper(**kwargs)


def _run_wrapper(wrapper, weights, t):
    x, ids, tw = _activations(t)
    return wrapper.run(
        x,
        weights["w1_weight"],
        weights["w1_weight_sf"],
        weights["w2_weight"],
        weights["w2_weight_sf"],
        ids,
        tw,
        w1_alpha=weights["w1_alpha"],
        w2_alpha=weights["w2_alpha"],
    )


class TestWrapperReachability:
    def test_wrapper_rejects_bogus_backend(self, sm120_env):
        with pytest.raises(ValueError, match="backend must be one of"):
            _make_wrapper(backend="fa2")

    def test_wrapper_auto_reaches_specialized_on_allowlisted_shape(
        self, sm120_env, weights
    ):
        before = spec.dispatch_stats.get("specialized", 0)
        out = _run_wrapper(_make_wrapper(), weights, 16)
        assert sm120_env["spec_calls"] == 1
        assert sm120_env["stock_calls"] == 0
        assert spec.dispatch_stats.get("specialized", 0) == before + 1
        assert bool((out == 7.0).all())

    def test_wrapper_auto_off_allowlist_keeps_stock(self, sm120_env, weights):
        _run_wrapper(_make_wrapper(), weights, 8)
        assert sm120_env["spec_calls"] == 0
        assert sm120_env["stock_calls"] == 1

    def test_wrapper_explicit_stock_never_receives_specialized(
        self, sm120_env, weights
    ):
        _run_wrapper(_make_wrapper(backend="b12x"), weights, 16)
        assert sm120_env["spec_calls"] == 0
        assert sm120_env["stock_calls"] == 1

    def test_wrapper_kill_switch_keeps_stock(self, sm120_env, weights, monkeypatch):
        monkeypatch.setenv(spec.ENV_DISABLE, "1")
        _run_wrapper(_make_wrapper(), weights, 16)
        assert sm120_env["spec_calls"] == 0
        assert sm120_env["stock_calls"] == 1

    def test_wrapper_explicit_specialized_off_allowlist_raises(
        self, sm120_env, weights
    ):
        with pytest.raises(ValueError, match="token_count"):
            _run_wrapper(_make_wrapper(backend="specialized"), weights, 8)
        assert sm120_env["stock_calls"] == 0

    def test_wrapper_expert_parallel_keeps_stock(self, sm120_env, weights):
        expert_map = torch.full((E,), -1, dtype=torch.int32)
        expert_map[: E // 2] = torch.arange(E // 2, dtype=torch.int32)
        wrapper = _make_wrapper(num_local_experts=E // 2, expert_map=expert_map)
        _run_wrapper(wrapper, weights, 16)
        assert sm120_env["spec_calls"] == 0
        assert sm120_env["stock_calls"] == 1
        assert ("expert_parallelism",) in spec.fallback_reasons


# ---------------------------------------------------------------------------
# MoELayer reachability (BackendOptions((B12xW4A16Config(),)) on sm120)
# ---------------------------------------------------------------------------


def _make_layer_and_packs(weights, t):
    from flashinfer.fused_moe import (
        B12xW4A16Config,
        MoEActivationPack,
        MoELayer,
        MoEWeightPack,
    )
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        ReLU2,
        RoutingConfig,
    )

    config = MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=TOPK),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.BF16),
        experts=ExpertConfig(intermediate_size=I, local_num_experts=E),
        activation=ReLU2(),
        backend=BackendOptions((B12xW4A16Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=t),
    )
    layer = MoELayer(config, device=torch.device("cpu"))
    assert len(layer.runners) == 1
    layer._select_winner = lambda *_: (layer.runners[0], -1)

    weight_pack = MoEWeightPack()
    weight_pack.prepare_for("b12x_w4a16", weights)
    x, ids, tw = _activations(t)
    act_pack = MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=None,
        topk_ids=ids,
        topk_weights=tw,
    )
    return layer, act_pack, weight_pack


class TestMoELayerReachability:
    def test_moe_layer_auto_reaches_specialized_on_allowlisted_shape(
        self, sm120_env, weights
    ):
        layer, act_pack, weight_pack = _make_layer_and_packs(weights, 16)
        before = spec.dispatch_stats.get("specialized", 0)
        out = layer(act_pack, weight_pack)
        assert sm120_env["spec_calls"] == 1
        assert sm120_env["stock_calls"] == 0
        assert spec.dispatch_stats.get("specialized", 0) == before + 1
        assert bool((out == 7.0).all())
        assert layer.winner_backend == "b12x_w4a16"

    def test_moe_layer_off_allowlist_keeps_stock(self, sm120_env, weights):
        layer, act_pack, weight_pack = _make_layer_and_packs(weights, 8)
        layer(act_pack, weight_pack)
        assert sm120_env["spec_calls"] == 0
        assert sm120_env["stock_calls"] == 1
        assert any(
            any(r.startswith("token_count") for r in reasons)
            for reasons in spec.fallback_reasons
        )

    def test_moe_layer_kill_switch_keeps_stock(self, sm120_env, weights, monkeypatch):
        monkeypatch.setenv(spec.ENV_DISABLE, "1")
        layer, act_pack, weight_pack = _make_layer_and_packs(weights, 16)
        layer(act_pack, weight_pack)
        assert sm120_env["spec_calls"] == 0
        assert sm120_env["stock_calls"] == 1
        assert any("kill_switch" in reasons for reasons in spec.fallback_reasons)
