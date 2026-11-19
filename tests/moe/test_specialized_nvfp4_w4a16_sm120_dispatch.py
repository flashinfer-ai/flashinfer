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

"""Dispatch-logic tests for the specialized SM120 NVFP4-W4A16 MoE backend.

The kernels' sole public home is the unified MoE API: these tests exercise
the guard (:func:`specialized_support_reasons`), the unified
``SpecializedW4A16Runner`` (direct module dispatch, allowlist eligibility,
kill switch, capture guard, no-fallback honesty), default-list visibility,
and the NEGATIVE CONTROLS for the removed facade hook (``b12x_fused_moe`` /
``B12xMoEWrapper`` carry no ``backend`` kwarg and never reach the
specialized kernels).

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
    # --- SM120 device + CUDA >= 13.2 toolkit probes
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    monkeypatch.setattr(
        "flashinfer.jit.cpp_ext.get_cuda_version",
        lambda: SimpleNamespace(major=13, minor=2),
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
    stub._activation_precision_from_quant_mode = lambda quant_mode: (
        "bf16" if quant_mode == "w4a16" else "fp4"
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
# The guard (specialized_support_reasons)
# ---------------------------------------------------------------------------


class TestGuard:
    def _reasons(self, t, weights, **overrides):
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
        return spec.specialized_support_reasons(
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

    def test_on_allowlist_supported(self, sm120_env, weights):
        ok, reasons = self._reasons(16, weights)
        assert ok and not reasons

    @pytest.mark.parametrize("minor", [0, 1])
    def test_unsupported_toolkit_declines_before_build(
        self, sm120_env, weights, monkeypatch, minor
    ):
        monkeypatch.setattr(
            "flashinfer.jit.cpp_ext.get_cuda_version",
            lambda: SimpleNamespace(major=13, minor=minor),
        )
        ok, reasons = self._reasons(16, weights)
        assert not ok and f"cuda_version:13.{minor}" in reasons
        assert sm120_env["spec_calls"] == 0

    def test_off_allowlist_token_count(self, sm120_env, weights):
        ok, reasons = self._reasons(8, weights)
        assert not ok
        assert any(r.startswith("token_count") for r in reasons)

    def test_kill_switch(self, sm120_env, weights, monkeypatch):
        monkeypatch.setenv(spec.ENV_DISABLE, "1")
        ok, reasons = self._reasons(16, weights)
        assert not ok and "kill_switch" in reasons

    def test_missing_alpha(self, sm120_env, weights):
        ok, reasons = self._reasons(16, weights, w1_alpha=None)
        assert not ok and "w1_alpha_missing" in reasons

    def test_runs_under_autotune_tuning_mode(self, sm120_env, weights, monkeypatch):
        """The guard has no autotune bail: the unified runner is a
        first-class autotune candidate, so the specialized kernels are what
        the shootout times."""
        from flashinfer.autotuner import AutoTuner

        monkeypatch.setattr(AutoTuner.get(), "is_tuning_mode", True)
        ok, reasons = self._reasons(16, weights)
        assert ok and not reasons


# ---------------------------------------------------------------------------
# Facade negative controls: the kernels are NOT a b12x backend
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


class TestFacadeNegativeControls:
    def test_no_backend_kwarg_on_the_b12x_entry_points(self):
        """The specialized kernels are not b12x kernels, so the vendor entry
        points carry no backend kwarg and no re-exported BACKENDS tuple."""
        import inspect

        from flashinfer.fused_moe.cute_dsl import b12x_moe

        assert "backend" not in inspect.signature(b12x_moe.b12x_fused_moe).parameters
        assert (
            "backend"
            not in inspect.signature(b12x_moe.B12xMoEWrapper.__init__).parameters
        )
        assert not hasattr(b12x_moe, "BACKENDS")
        assert not hasattr(spec, "BACKENDS")
        assert not hasattr(spec, "maybe_run_specialized")

    def test_wrapper_never_reaches_the_specialized_kernels(self, sm120_env, weights):
        """Even on an allowlisted shape, the b12x wrapper is the stock
        path, period."""
        _run_wrapper(_make_wrapper(), weights, 16)
        assert sm120_env["spec_calls"] == 0
        assert sm120_env["stock_calls"] == 1


# ---------------------------------------------------------------------------
# Unified MoELayer surface: the kernels' public home
# ---------------------------------------------------------------------------


def _moe_config(t, backends=None, top_k=TOPK):
    from flashinfer.fused_moe import B12xW4A16Config
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

    if backends is None:
        backends = (B12xW4A16Config(),)
    return MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=top_k),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.BF16),
        experts=ExpertConfig(intermediate_size=I, local_num_experts=E),
        activation=ReLU2(),
        backend=BackendOptions(backends),
        execution=ExecutionConfig(tune_max_num_tokens=t),
    )


def _make_layer_and_packs(weights, t, backends=None, winner_key=None):
    from flashinfer.fused_moe import (
        MoEActivationPack,
        MoELayer,
        MoEWeightPack,
    )

    layer = MoELayer(_moe_config(t, backends), device=torch.device("cpu"))
    if winner_key is None:
        layer._select_winner = lambda *_, **__: (layer.runners[0], -1)
    else:
        winner = next(r for r in layer.runners if r.backend_key == winner_key)
        layer._select_winner = lambda *_, **__: (winner, -1)

    weight_pack = MoEWeightPack()
    # The two views carry DIFFERENT layouts (stock: swizzled 128-row-padded
    # scales; specialized: row-major unpadded) and neither aliases the
    # other, so each candidate gets its own registration. The mocked
    # contract-shaped tensors satisfy the specialized runner's metadata
    # check, which is all these CPU tests consume.
    weight_pack.prepare_for("b12x_w4a16", weights)
    weight_pack.prepare_for("specialized_w4a16", weights)
    x, ids, tw = _activations(t)
    act_pack = MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=None,
        topk_ids=ids,
        topk_weights=tw,
    )
    return layer, act_pack, weight_pack


def _specialized_configs():
    from flashinfer.fused_moe import B12xW4A16Config, SpecializedW4A16Config

    return B12xW4A16Config, SpecializedW4A16Config


class TestUnifiedSurface:
    def test_default_visible(self):
        from flashinfer.fused_moe.api import _DEFAULT_BACKEND

        _, SpecializedW4A16Config = _specialized_configs()
        assert any(isinstance(c, SpecializedW4A16Config) for c in _DEFAULT_BACKEND)
        assert any(
            isinstance(c, SpecializedW4A16Config)
            for c in _DEFAULT_BACKEND.valid_for(120)
        )
        for arch in (90, 100, 103, 121):
            assert not any(
                isinstance(c, SpecializedW4A16Config)
                for c in _DEFAULT_BACKEND.valid_for(arch)
            )

    def test_both_candidates_constructed(self, sm120_env, weights):
        B12xW4A16Config, SpecializedW4A16Config = _specialized_configs()
        layer, _, _ = _make_layer_and_packs(
            weights, 16, backends=(B12xW4A16Config(), SpecializedW4A16Config())
        )
        assert [r.backend_key for r in layer.runners] == [
            "b12x_w4a16",
            "specialized_w4a16",
        ]

    @pytest.mark.parametrize("minor", [0, 1])
    def test_unsupported_toolkit_keeps_stock_candidate(
        self, sm120_env, weights, monkeypatch, minor
    ):
        monkeypatch.setattr(
            "flashinfer.jit.cpp_ext.get_cuda_version",
            lambda: SimpleNamespace(major=13, minor=minor),
        )
        B12xW4A16Config, SpecializedW4A16Config = _specialized_configs()
        layer, act_pack, weight_pack = _make_layer_and_packs(
            weights, 16, backends=(B12xW4A16Config(), SpecializedW4A16Config())
        )
        assert [r.backend_key for r in layer.runners] == ["b12x_w4a16"]
        layer(act_pack, weight_pack)
        assert sm120_env["spec_calls"] == 0
        assert sm120_env["stock_calls"] == 1

    def test_specialized_runner_dispatches_its_own_module(self, sm120_env, weights):
        """The runner calls the kernels' own module directly, served by its
        own ``specialized_w4a16`` weight view (the stock swizzled view is a
        different layout and is not an alias)."""
        B12xW4A16Config, SpecializedW4A16Config = _specialized_configs()
        layer, act_pack, weight_pack = _make_layer_and_packs(
            weights,
            16,
            backends=(B12xW4A16Config(), SpecializedW4A16Config()),
            winner_key="specialized_w4a16",
        )
        before = spec.dispatch_stats.get("specialized", 0)
        out = layer(act_pack, weight_pack)
        assert sm120_env["spec_calls"] == 1
        assert sm120_env["stock_calls"] == 0
        assert spec.dispatch_stats.get("specialized", 0) == before + 1
        assert bool((out == 7.0).all())
        assert layer.winner_backend == "specialized_w4a16"

    def test_runs_under_autotune_tuning_mode(self, sm120_env, weights, monkeypatch):
        """First-class candidacy: the specialized kernel is what runs while
        the autotuner times this runner (no bail, no tuner-state games)."""
        from flashinfer.autotuner import AutoTuner

        _, SpecializedW4A16Config = _specialized_configs()
        monkeypatch.setattr(AutoTuner.get(), "is_tuning_mode", True)
        layer, act_pack, weight_pack = _make_layer_and_packs(
            weights, 16, backends=(SpecializedW4A16Config(),)
        )
        runner = layer.runners[0]
        out = runner.forward(runner.pack_inputs(act_pack, weight_pack))
        assert sm120_env["spec_calls"] == 1
        assert bool((out == 7.0).all())

    def test_off_allowlist_declined_and_stock_serves(self, sm120_env, weights):
        """Out-of-allowlist token counts are declined at eligibility time:
        auto routes them to the stock b12x backend (negative control for
        the removed in-runner torch fallback — nothing serves through the
        specialized runner)."""
        B12xW4A16Config, SpecializedW4A16Config = _specialized_configs()
        layer, act_pack, weight_pack = _make_layer_and_packs(
            weights, 8, backends=(B12xW4A16Config(), SpecializedW4A16Config())
        )
        eligible = layer._eligible_runners(act_pack, weight_pack)
        assert [layer.runners[i].backend_key for i in eligible] == ["b12x_w4a16"]
        layer(act_pack, weight_pack)
        assert sm120_env["spec_calls"] == 0
        assert sm120_env["stock_calls"] == 1

    def test_off_allowlist_specialized_only_layer_refuses(self, sm120_env, weights):
        """A specialized-only layer refuses an out-of-allowlist call rather
        than serving it with a different kernel."""
        _, SpecializedW4A16Config = _specialized_configs()
        layer, act_pack, weight_pack = _make_layer_and_packs(
            weights, 8, backends=(SpecializedW4A16Config(),)
        )
        with pytest.raises(NotImplementedError, match="routing_input_mode"):
            layer(act_pack, weight_pack)
        assert sm120_env["spec_calls"] == 0

    def test_declined_accepted_call_raises_instead_of_fallback(
        self, sm120_env, weights, monkeypatch
    ):
        """A call-time guard refusal after acceptance raises: the runner
        never serves a call with another kernel (no torch fallback)."""
        _, SpecializedW4A16Config = _specialized_configs()
        layer, act_pack, weight_pack = _make_layer_and_packs(
            weights, 16, backends=(SpecializedW4A16Config(),)
        )
        runner = layer.runners[0]
        packed = runner.pack_inputs(act_pack, weight_pack)
        monkeypatch.setattr(
            spec,
            "specialized_support_reasons",
            lambda *a, **k: (False, ["token_count:16"]),
        )
        with pytest.raises(RuntimeError, match="declined an accepted call"):
            runner.forward(packed)
        assert sm120_env["spec_calls"] == 0

    def test_capture_guard_raises_instead_of_capturing_fallback(
        self, sm120_env, weights, monkeypatch
    ):
        _, SpecializedW4A16Config = _specialized_configs()
        layer, act_pack, weight_pack = _make_layer_and_packs(
            weights, 16, backends=(SpecializedW4A16Config(),)
        )
        runner = layer.runners[0]
        packed = runner.pack_inputs(act_pack, weight_pack)
        monkeypatch.setattr(spec, "_is_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="CUDA-graph capture"):
            runner.forward(packed)
        assert sm120_env["spec_calls"] == 0

    def test_kill_switch_rejects_candidate_at_construction(
        self, sm120_env, weights, monkeypatch
    ):
        from flashinfer.fused_moe.layer import NoEligibleBackend

        _, SpecializedW4A16Config = _specialized_configs()
        monkeypatch.setenv(spec.ENV_DISABLE, "1")
        with pytest.raises(NoEligibleBackend, match="DISABLE"):
            _make_layer_and_packs(weights, 16, backends=(SpecializedW4A16Config(),))

    def test_off_contract_geometry_rejected_at_construction(self, sm120_env, weights):
        from flashinfer.fused_moe import MoELayer
        from flashinfer.fused_moe.layer import NoEligibleBackend

        _, SpecializedW4A16Config = _specialized_configs()
        with pytest.raises(NoEligibleBackend, match="top_k"):
            MoELayer(
                _moe_config(16, (SpecializedW4A16Config(),), top_k=4),
                device=torch.device("cpu"),
            )


class TestWeightViewEligibility:
    """The specialized candidate is eligible only with a registered,
    layout-compatible ``specialized_w4a16`` view (row-major unpadded
    group-16 scales): an absent view or the stock swizzled storage leaves
    it out of the shootout instead of raising while profiling."""

    def test_b12x_only_pack_leaves_specialized_ineligible(self, sm120_env, weights):
        B12xW4A16Config, SpecializedW4A16Config = _specialized_configs()
        layer, act_pack, weight_pack = _make_layer_and_packs(
            weights, 16, backends=(B12xW4A16Config(), SpecializedW4A16Config())
        )
        del weight_pack.native_views["specialized_w4a16"]
        eligible = layer._eligible_runners(act_pack, weight_pack)
        assert [layer.runners[i].backend_key for i in eligible] == ["b12x_w4a16"]

    def test_swizzled_scale_view_leaves_specialized_ineligible(
        self, sm120_env, weights
    ):
        B12xW4A16Config, SpecializedW4A16Config = _specialized_configs()
        layer, act_pack, weight_pack = _make_layer_and_packs(
            weights, 16, backends=(B12xW4A16Config(), SpecializedW4A16Config())
        )
        swizzled = dict(weights)
        swizzled["w1_weight_sf"] = torch.empty(
            E, 1920, 168, dtype=torch.float8_e4m3fn
        )  # 128-row-padded stock storage
        weight_pack.native_views["specialized_w4a16"] = swizzled
        eligible = layer._eligible_runners(act_pack, weight_pack)
        assert [layer.runners[i].backend_key for i in eligible] == ["b12x_w4a16"]

    def test_prepare_weights_no_longer_delegates_to_stock(self):
        from flashinfer.fused_moe import B12xW4A16Config, SpecializedW4A16Config

        assert (
            SpecializedW4A16Config.prepare_weights
            is not B12xW4A16Config.prepare_weights
        )
