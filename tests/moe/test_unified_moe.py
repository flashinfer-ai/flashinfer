"""Tests for the unified MoE API (config dataclasses + MoELayer + Packs).

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

Two sections:

  * CPU-only config/dataclass tests (no GPU or JIT). These track the actual MVP
    API surface (single-knob ``QuantVariant``, explicit
    ``BackendOptions(candidates=(...))``); see
    ``docs/design_docs/flashinfer_moe_api.md`` §10 CR1.

  * SM100 (Blackwell) GPU tests for ``MoELayer`` + Packs, parametrized per
    ``QuantVariant`` via ``VariantSpec`` (currently NVFP4 + BF16, pre-routed
    path): accuracy vs an independent reference, direct-runner conformance,
    CUDA-graph replay, autotune candidate visitation, and the packed-topk-id
    contract (CR3). Adding a variant = registering one spec.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Callable, ClassVar

import pytest
import torch
import torch.nn.functional as F

from flashinfer.autotuner import autotune
from flashinfer.autotuner.autotuner import ProfilingCacheKey
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.fused_moe import (
    # Typed activation values
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    ReLU,
    ReLU2,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
    # Unified configs, packs, and runners
    ActivationType,
    BackendOptions,
    CuteDslConfig,
    CuteDslRunner,
    CutlassBf16Config,
    CutlassBf16Runner,
    CutlassFp8BlockConfig,
    CutlassFp8PerTensorConfig,
    CutlassMxfp8Config,
    CutlassNvfp4Config,
    CutlassW4A16Runner,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    RoutingInputMode,
    RoutingMethodType,
    TrtllmBf16Config,
    TrtllmBf16RoutedRunner,
    TrtllmFp4Config,
    TrtllmFp4RoutedRunner,
    TrtllmFp8BlockConfig,
    TrtllmFp8BlockRunner,
    TrtllmFp8PerTensorConfig,
    TrtllmFp8PerTensorRunner,
    TrtllmMxInt4Config,
    TrtllmMxInt4RoutedRunner,
)
from flashinfer.fused_moe.runners import MoERunner
from flashinfer.fused_moe.core import _fake_trtllm_moe_output
from flashinfer.tllm_enums import DEFAULT_SITU_BETA, DEFAULT_SITU_LINEAR_BETA
from flashinfer.utils import get_compute_capability


def _build_direct_runner(runner_type, config, device):
    runner = runner_type(config, device=device)
    runner.check_support()
    runner.build()
    return runner


# Reuse the canonical reference implementation + accuracy helpers from the
# existing CuteDSL test — keeps tolerance bounds consistent across tests.
from tests.moe.test_cute_dsl_fused_moe import (  # noqa: E402
    check_accuracy,
    compute_reference_moe_fp4,
    create_moe_tensors,
)
from tests.moe.utils import create_relu2_moe_tensors  # noqa: E402


def test_noaux_tc_ref_excludes_unselected_groups_with_negative_scores():
    from tests.moe.trtllm_gen_fused_moe_utils import noaux_tc_ref

    logits = torch.zeros((1, 8), dtype=torch.float32)
    bias = torch.tensor(
        [[2.5, 1.5, 0.5, -1.5, 0.0, -0.1, -0.2, -0.3]],
        dtype=torch.float32,
    )

    scores = noaux_tc_ref(
        logits,
        bias,
        n_group=2,
        topk_group=1,
        top_k=4,
        routed_scaling_factor=1.0,
    )

    selected = torch.where(scores[0] != 0)[0]
    assert set(selected.tolist()) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Enum repr round-trip
# ---------------------------------------------------------------------------


class TestEnumRepr:
    @pytest.mark.parametrize("member", list(RoutingMethodType))
    def test_routing_method_repr(self, member):
        assert eval(repr(member)) == member

    @pytest.mark.parametrize("member", list(ActivationType))
    def test_activation_repr(self, member):
        assert eval(repr(member)) == member

    @pytest.mark.parametrize("member", list(QuantVariant))
    def test_quant_variant_repr(self, member):
        assert eval(repr(member)) == member


class TestTrtllmFakeOutputContract:
    class _FakeContext:
        def __init__(self):
            self._next = 16

        def new_dynamic_size(self):
            self._next += 1
            return self._next

    def test_unfinalized_generated_weights(self, monkeypatch):
        monkeypatch.setattr(torch.library, "get_ctx", lambda: self._FakeContext())
        hidden_states = torch.empty((4, 32), device="meta")
        result = _fake_trtllm_moe_output(
            hidden_states,
            hidden_size=32,
            intermediate_size=64,
            top_k=2,
            do_finalize=False,
        )

        assert len(result) == 3
        assert result[0].shape == (17, 32)
        assert result[1].shape == (4, 2)
        assert result[1].dtype == torch.bfloat16
        assert result[2].shape == (8,)
        assert result[2].dtype == torch.int32

    def test_unfinalized_preserves_precomputed_weights(self, monkeypatch):
        monkeypatch.setattr(torch.library, "get_ctx", lambda: self._FakeContext())
        hidden_states = torch.empty((4, 32), device="meta")
        weights = torch.empty((4, 2), dtype=torch.float32, device="meta")
        result = _fake_trtllm_moe_output(
            hidden_states,
            hidden_size=32,
            intermediate_size=64,
            top_k=2,
            do_finalize=False,
            expert_weights=weights,
        )

        assert result[1] is weights

    def test_finalized_lora_arity(self, monkeypatch):
        monkeypatch.setattr(torch.library, "get_ctx", lambda: self._FakeContext())
        hidden_states = torch.empty((4, 32), device="meta")
        lora_delta = torch.empty((4, 128), device="meta")
        result = _fake_trtllm_moe_output(
            hidden_states,
            hidden_size=32,
            intermediate_size=64,
            top_k=2,
            do_finalize=True,
            gemm1_lora_delta=lora_delta,
        )

        assert len(result) == 3
        assert result[0].shape == (4, 32)
        assert result[1].shape == (8,)
        assert result[2].shape == (17, 64)


# ---------------------------------------------------------------------------
# ActivationType helpers
# ---------------------------------------------------------------------------


class TestActivation:
    def test_is_gated(self):
        assert ActivationType.Swiglu.is_gated
        assert ActivationType.Geglu.is_gated
        assert ActivationType.SwigluBias.is_gated
        assert ActivationType.SwigluStep.is_gated
        assert not ActivationType.Identity.is_gated
        assert not ActivationType.Relu2.is_gated
        assert not ActivationType.Gelu.is_gated


# ---------------------------------------------------------------------------
# Config immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_routing_config_frozen(self):
        cfg = RoutingConfig(num_experts=64, top_k=8)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.top_k = 4

    def test_quant_config_frozen(self):
        cfg = QuantConfig(variant=QuantVariant.FP8PerTensor)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.variant = QuantVariant.BF16

    def test_moe_config_frozen(self):
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=8, top_k=2),
            quant=QuantConfig(variant=QuantVariant.BF16),
            experts=ExpertConfig(intermediate_size=512),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.routing = RoutingConfig(num_experts=16, top_k=2)


# ---------------------------------------------------------------------------
# Config repr round-trip (critical for repro serialization)
# ---------------------------------------------------------------------------


def _eval_repr(obj):
    """Evaluate repr(obj) in the config namespace — must reconstruct the object."""
    from flashinfer.fused_moe import api as ns

    return eval(
        repr(obj), {k: getattr(ns, k) for k in dir(ns) if not k.startswith("_")}
    )


class TestReprRoundTrip:
    def test_routing_config_minimal(self):
        cfg = RoutingConfig(num_experts=64, top_k=8)
        assert _eval_repr(cfg) == cfg

    def test_routing_config_full(self):
        cfg = RoutingConfig(
            num_experts=256,
            top_k=8,
            method=RoutingMethodType.DeepSeekV3,
            n_group=8,
            topk_group=4,
            routed_scaling_factor=1.0,
        )
        assert _eval_repr(cfg) == cfg

    @pytest.mark.parametrize("variant", list(QuantVariant))
    def test_quant_config(self, variant):
        cfg = QuantConfig(variant=variant)
        assert _eval_repr(cfg) == cfg

    def test_activation_config(self):
        for cfg in (
            SwiGLU(),
            SwiGLU(alpha=1.5, beta=0.25, limit=7.0),
            SiTU(gate_scale=2.0, linear_scale=3.0, clamp_limit=4.0),
            GeGLU(),
            ReLU2(),
            GeGLUTanh(),
            SwiGLUStep(),
            Identity(),
            GELU(),
            ReLU(),
            SiLU(),
        ):
            assert _eval_repr(cfg) == cfg
            assert hash(_eval_repr(cfg)) == hash(cfg)

    def test_expert_config(self):
        cfg = ExpertConfig(
            intermediate_size=2048,
            local_expert_offset=4,
            local_num_experts=8,
            num_fused_shared_experts=2,
        )
        assert _eval_repr(cfg) == cfg

    def test_execution_config_default(self):
        cfg = ExecutionConfig()
        assert _eval_repr(cfg) == cfg

    def test_execution_config_custom(self):
        cfg = ExecutionConfig(enable_pdl=True, tune_max_num_tokens=1024)
        assert _eval_repr(cfg) == cfg

    def test_finalize_config_default(self):
        cfg = MoEFinalizeConfig()
        assert _eval_repr(cfg) == cfg

    def test_finalize_config_custom(self):
        cfg = MoEFinalizeConfig(do_finalize=False, use_fused_finalize=False)
        assert _eval_repr(cfg) == cfg

    def test_backend_options_multi(self):
        opts = BackendOptions(candidates=(TrtllmFp4Config(), CutlassNvfp4Config()))
        reconstructed = _eval_repr(opts)
        assert len(reconstructed) == 2
        assert isinstance(reconstructed.candidates[0], TrtllmFp4Config)
        assert isinstance(reconstructed.candidates[1], CutlassNvfp4Config)

    def test_backend_options_single(self):
        opts = BackendOptions(candidates=(TrtllmFp8PerTensorConfig(),))
        reconstructed = _eval_repr(opts)
        assert len(reconstructed) == 1
        assert isinstance(reconstructed.candidates[0], TrtllmFp8PerTensorConfig)

    def test_moe_config_minimal(self):
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=8, top_k=2),
            quant=QuantConfig(variant=QuantVariant.BF16),
            experts=ExpertConfig(intermediate_size=512),
        )
        assert _eval_repr(cfg) == cfg

    def test_moe_config_full(self):
        cfg = MoEConfig(
            routing=RoutingConfig(
                num_experts=256,
                top_k=8,
                method=RoutingMethodType.DeepSeekV3,
                n_group=8,
                topk_group=4,
                routed_scaling_factor=1.0,
            ),
            quant=QuantConfig(variant=QuantVariant.MxFp8),
            experts=ExpertConfig(intermediate_size=2048, local_num_experts=32),
            activation=GeGLU(),
            backend=BackendOptions(
                candidates=(TrtllmFp8BlockConfig(), CutlassMxfp8Config())
            ),
            execution=ExecutionConfig(enable_pdl=True, tune_max_num_tokens=4096),
        )
        assert _eval_repr(cfg) == cfg


# ---------------------------------------------------------------------------
# BackendOptions
# ---------------------------------------------------------------------------


class TestBackendOptions:
    def test_explicit_candidates(self):
        opts = BackendOptions(candidates=(TrtllmFp4Config(), CutlassNvfp4Config()))
        assert isinstance(opts, BackendOptions)
        assert len(opts) == 2

    def test_multiple_candidates(self):
        opts = BackendOptions(
            candidates=(TrtllmFp4Config(), TrtllmFp8BlockConfig(), CutlassNvfp4Config())
        )
        assert len(opts) == 3

    def test_valid_for_filtering(self):
        opts = BackendOptions(
            candidates=(TrtllmBf16Config(), TrtllmFp8BlockConfig(), CutlassBf16Config())
        )
        # SM90 is supported by CUTLASS BF16 but not the TRTLLM runners above.
        valid = opts.valid_for(90)
        assert len(valid) == 1
        assert isinstance(valid[0], CutlassBf16Config)

    def test_valid_for_blackwell(self):
        opts = BackendOptions(
            candidates=(TrtllmBf16Config(), TrtllmFp8BlockConfig(), CutlassBf16Config())
        )
        valid = opts.valid_for(100)
        assert len(valid) == 3
        assert CutlassBf16Config.supported(100)
        assert TrtllmBf16Config.supported(100)
        assert TrtllmBf16Config.supported(103)
        assert TrtllmFp4Config.supported(107)
        assert not TrtllmFp8BlockConfig.supported(107)
        assert TrtllmBf16Config.supported(107)
        assert not TrtllmBf16Config.supported(110)
        assert not TrtllmBf16Config.supported(120)
        assert not TrtllmBf16Config.supported(121)
        assert not TrtllmFp8BlockConfig.supported(110)
        assert not TrtllmFp8BlockConfig.supported(120)
        assert TrtllmFp8PerTensorConfig.supported(100)
        assert TrtllmFp8PerTensorConfig.supported(103)
        assert not TrtllmFp8PerTensorConfig.supported(107)
        assert not TrtllmFp8PerTensorConfig.supported(90)
        assert not TrtllmFp8PerTensorConfig.supported(120)

    def test_iteration(self):
        opts = BackendOptions(candidates=(TrtllmFp4Config(), CutlassNvfp4Config()))
        items = list(opts)
        assert len(items) == 2
        assert any(isinstance(c, TrtllmFp4Config) for c in items)
        assert any(isinstance(c, CutlassNvfp4Config) for c in items)

    def test_empty(self):
        opts = BackendOptions()
        assert len(opts) == 0
        assert opts.valid_for(100) == []


# ---------------------------------------------------------------------------
# QuantConfig
# ---------------------------------------------------------------------------


class TestQuantConfig:
    def test_default_is_bf16(self):
        assert QuantConfig().variant == QuantVariant.BF16

    def test_explicit_variant(self):
        assert QuantConfig(variant=QuantVariant.NVFP4).variant == QuantVariant.NVFP4

    @pytest.mark.parametrize("variant", list(QuantVariant))
    def test_all_variants_constructible(self, variant):
        assert QuantConfig(variant=variant).variant is variant


# ---------------------------------------------------------------------------
# MoEConfig dict-unpacking protocol
# ---------------------------------------------------------------------------


class TestMoEConfigDictProtocol:
    def test_keys(self):
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=8, top_k=2),
            quant=QuantConfig(variant=QuantVariant.BF16),
            experts=ExpertConfig(intermediate_size=512),
        )
        keys = list(cfg.keys())
        assert "routing" in keys
        assert "quant" in keys
        assert "experts" in keys
        assert "activation" in keys
        assert "backend" in keys
        assert "execution" in keys

    def test_getitem(self):
        routing = RoutingConfig(num_experts=8, top_k=2)
        cfg = MoEConfig(
            routing=routing,
            quant=QuantConfig(variant=QuantVariant.BF16),
            experts=ExpertConfig(intermediate_size=512),
        )
        assert cfg["routing"] is routing

    def test_unpack(self):
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=8, top_k=2),
            quant=QuantConfig(variant=QuantVariant.BF16),
            experts=ExpertConfig(intermediate_size=512),
        )
        d = dict(**cfg)
        assert isinstance(d["routing"], RoutingConfig)
        assert isinstance(d["backend"], BackendOptions)


# ---------------------------------------------------------------------------
# Dataclasses.replace for immutable overrides
# ---------------------------------------------------------------------------


class TestImmutableReplace:
    def test_replace_quant(self):
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=64, top_k=8),
            quant=QuantConfig(variant=QuantVariant.BF16),
            experts=ExpertConfig(intermediate_size=2048),
        )
        fp8_cfg = dataclasses.replace(
            cfg,
            quant=QuantConfig(variant=QuantVariant.DeepSeekFp8),
        )
        assert fp8_cfg.quant.variant == QuantVariant.DeepSeekFp8
        assert cfg.quant.variant == QuantVariant.BF16  # original unchanged

    def test_replace_backend(self):
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=8, top_k=2),
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=ExpertConfig(intermediate_size=512),
        )
        narrow = dataclasses.replace(
            cfg, backend=BackendOptions((CutlassNvfp4Config(),))
        )
        assert len(narrow.backend) == 1


# ---------------------------------------------------------------------------
# Hashability (needed for cache keys)
# ---------------------------------------------------------------------------


class TestHashability:
    def test_routing_config_hashable(self):
        a = RoutingConfig(num_experts=64, top_k=8)
        b = RoutingConfig(num_experts=64, top_k=8)
        assert hash(a) == hash(b)
        assert {a, b} == {a}

    def test_moe_config_hashable(self):
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=8, top_k=2),
            quant=QuantConfig(variant=QuantVariant.BF16),
            experts=ExpertConfig(intermediate_size=512),
            backend=BackendOptions(
                candidates=(TrtllmBf16Config(), CutlassBf16Config())
            ),
        )
        # Must not raise
        h = hash(cfg)
        assert isinstance(h, int)

    def test_moe_config_as_dict_key(self):
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=8, top_k=2),
            quant=QuantConfig(variant=QuantVariant.BF16),
            experts=ExpertConfig(intermediate_size=512),
        )
        d = {cfg: "value"}
        assert d[cfg] == "value"


# ---------------------------------------------------------------------------
# Typed activation values
# ---------------------------------------------------------------------------


class TestTypedActivationConfig:
    def test_parameter_free_activations_are_public(self):
        import flashinfer.fused_moe as fused_moe

        for name, activation_type in (
            ("GELU", ActivationType.Gelu),
            ("ReLU", ActivationType.Relu),
            ("SiLU", ActivationType.Silu),
        ):
            activation_cls = getattr(fused_moe, name)
            activation = activation_cls()
            assert name in fused_moe.__all__
            assert activation.type is activation_type
            assert not activation.is_gated
            assert eval(repr(activation), vars(fused_moe)) == activation
            assert hash(activation) == hash(activation_cls())

    def test_type_and_gating(self):
        assert SwiGLU().type is ActivationType.Swiglu
        assert GeGLU().type is ActivationType.Geglu
        assert ReLU2().type is ActivationType.Relu2
        assert Identity().type is ActivationType.Identity
        assert GELU().type is ActivationType.Gelu
        assert ReLU().type is ActivationType.Relu
        assert SiLU().type is ActivationType.Silu
        assert SwiGLU().is_gated
        assert not Identity().is_gated
        assert not GELU().is_gated
        assert not ReLU().is_gated
        assert not SiLU().is_gated

    def test_common_base_is_not_a_concrete_activation(self):
        from flashinfer.fused_moe import ActivationConfig

        with pytest.raises(TypeError, match="common base"):
            ActivationConfig()

    @pytest.mark.parametrize(
        "factory",
        (
            lambda: SwiGLU(alpha=float("nan")),
            lambda: SwiGLU(limit=0),
            lambda: SiTU(gate_scale=0),
            lambda: SiTU(linear_scale=float("inf")),
            lambda: SiTU(clamp_limit=-1),
            lambda: SwiGLUStep(limit=0),
        ),
    )
    def test_scalar_validation(self, factory):
        with pytest.raises(ValueError):
            factory()

    def test_trtllm_scalar_expansion_keeps_per_expert_abi(self):
        from flashinfer.fused_moe.prepare import _activation_param_view

        assert _activation_param_view(SwiGLU(), 3, torch.device("cpu")) == {}
        view = _activation_param_view(
            SwiGLU(alpha=1.7, beta=0.25, limit=6.0),
            3,
            torch.device("cpu"),
        )
        torch.testing.assert_close(view["gemm1_alpha"], torch.full((3,), 1.7))
        torch.testing.assert_close(view["gemm1_beta"], torch.full((3,), 0.25))
        torch.testing.assert_close(view["gemm1_clamp_limit"], torch.full((3,), 6.0))

        situ = _activation_param_view(
            SiTU(gate_scale=2.0, linear_scale=3.0, clamp_limit=4.0),
            2,
            torch.device("cpu"),
        )
        torch.testing.assert_close(situ["gemm1_alpha"], torch.full((2,), 2.0))
        torch.testing.assert_close(situ["gemm1_beta"], torch.full((2,), 3.0))
        torch.testing.assert_close(situ["gemm1_clamp_limit"], torch.full((2,), 4.0))

    def test_cute_dsl_scalar_mapping(self):
        from flashinfer.fused_moe.runners import _cute_dsl_activation_kwargs

        assert _cute_dsl_activation_kwargs(SwiGLU(alpha=1.5, beta=0.5, limit=7.0)) == {
            "activation_type": int(ActivationType.Swiglu),
            "swiglu_alpha": 1.5,
            "swiglu_beta": 0.5,
            "swiglu_limit": 7.0,
        }
        assert _cute_dsl_activation_kwargs(SiTU(gate_scale=2.0, linear_scale=3.0)) == {
            "activation_type": int(ActivationType.Swiglu),
            "situ_beta": 2.0,
            "situ_linear_beta": 3.0,
        }
        # linear_scale=None reaches the CuTe-DSL ABI as "no linear-branch clamp".
        assert _cute_dsl_activation_kwargs(SiTU(linear_scale=None)) == {
            "activation_type": int(ActivationType.Swiglu),
            "situ_beta": DEFAULT_SITU_BETA,
            "situ_linear_beta": None,
        }

    def test_situ_unclamped_linear_branch_is_expressible(self):
        activation = SiTU(linear_scale=None)
        assert activation.linear_scale is None
        # The unclamped mode must stay hashable/frozen like every other value.
        assert activation == SiTU(linear_scale=None)
        assert hash(activation) == hash(SiTU(linear_scale=None))
        assert activation != SiTU()
        # The default stays the clamped canonical scale.
        assert SiTU().linear_scale == DEFAULT_SITU_LINEAR_BETA

    def test_situ_unclamped_linear_branch_has_no_per_expert_tensor(self):
        from flashinfer.fused_moe.prepare import _activation_param_view

        view = _activation_param_view(
            SiTU(gate_scale=2.0, linear_scale=None), 2, torch.device("cpu")
        )
        torch.testing.assert_close(view["gemm1_alpha"], torch.full((2,), 2.0))
        assert "gemm1_beta" not in view

    def test_typed_scalars_require_matching_prepared_metadata(self):
        from flashinfer.fused_moe.runners import (
            _validate_prepared_activation_params,
        )

        with pytest.raises(ValueError, match="missing activation parameters"):
            _validate_prepared_activation_params({}, SwiGLU(alpha=1.5), "TestRunner")
        with pytest.raises(ValueError, match="missing activation parameters"):
            _validate_prepared_activation_params({}, SiTU(), "TestRunner")

        overrides = {
            "gemm1_alpha": torch.ones(2),
            "gemm1_beta": torch.ones(2),
            "gemm1_clamp_limit": torch.ones(2),
        }
        _validate_prepared_activation_params(overrides, SwiGLU(alpha=1.5), "TestRunner")
        _validate_prepared_activation_params(
            overrides, SiTU(clamp_limit=4.0), "TestRunner"
        )
        # Only the parameters that actually differ from the defaults need a
        # tensor: a view carrying just gemm1_alpha is valid for SwiGLU(alpha=..),
        # whose beta/limit are exactly the kernel's neutral values.
        _validate_prepared_activation_params(
            {"gemm1_alpha": torch.ones(2)}, SwiGLU(alpha=1.5), "TestRunner"
        )
        _validate_prepared_activation_params(
            {"gemm1_clamp_limit": torch.ones(2)}, SwiGLU(limit=7.0), "TestRunner"
        )
        with pytest.raises(ValueError, match=r"\['gemm1_beta'\]"):
            _validate_prepared_activation_params(
                {"gemm1_alpha": torch.ones(2)},
                SwiGLU(alpha=1.5, beta=1.0),
                "TestRunner",
            )

        # linear_scale=None has no per-expert encoding, so gemm1_beta is not
        # required; gemm1_alpha still is.
        _validate_prepared_activation_params(
            {"gemm1_alpha": torch.ones(2)}, SiTU(linear_scale=None), "TestRunner"
        )
        with pytest.raises(ValueError, match="missing activation parameters"):
            _validate_prepared_activation_params(
                {}, SiTU(linear_scale=None), "TestRunner"
            )

    def test_none_valued_scalar_key_counts_as_missing(self):
        """A key present with value None is absent, not supplied.

        The launcher reads a null pointer as "use the neutral value", which is
        exactly what a non-default typed scalar says it must not do, so keying
        on presence alone would let it through.
        """
        from flashinfer.fused_moe.runners import (
            _validate_prepared_activation_params,
        )

        with pytest.raises(ValueError, match="missing activation parameters"):
            _validate_prepared_activation_params(
                {"gemm1_alpha": None}, SwiGLU(alpha=1.5), "TestRunner"
            )

    def test_scalar_overrides_rejected_for_non_gated_activation(self):
        """The gated epilogue is what reads the per-expert scalar tensors.

        GeGLU is gated and does consume alpha/beta -- its formula is
        (x0 + beta) * (x1 * phi(alpha * x1)) -- so only a non-gated activation
        should reject them. Accepting one there reads as a working override
        while the kernel discards it.
        """
        from flashinfer.fused_moe.runners import (
            _validate_prepared_activation_params,
        )

        assert not ReLU2().is_gated
        with pytest.raises(ValueError, match="does not consume"):
            _validate_prepared_activation_params(
                {"gemm1_alpha": torch.ones(2)}, ReLU2(), "TestRunner"
            )
        # No override is still fine.
        _validate_prepared_activation_params({}, ReLU2(), "TestRunner")
        # Deliberately no positive case for a gated non-SwiGLU activation.
        # Whether one accepts these overrides is backend-specific, so this
        # shared helper -- which serves the TRTLLM preparation contract -- must
        # not be read as asserting generic acceptance.

    @pytest.mark.parametrize(
        "activation,expected_rows",
        (
            (SwiGLU(), 256),
            (SiTU(), 256),
            (GeGLU(), 256),
            (GeGLUTanh(), 256),
            (SwiGLUStep(), 256),
            (ReLU2(), 128),
            (Identity(), 128),
            (GELU(), 128),
            (ReLU(), 128),
            (SiLU(), 128),
        ),
    )
    def test_typed_activation_controls_gemm1_rows(self, activation, expected_rows):
        from flashinfer.fused_moe.prepare import _gemm1_rows

        assert _gemm1_rows(128, activation) == expected_rows

    def test_defaults_match_flat_abi(self):
        activation = SwiGLU()
        assert activation.alpha == 1.0
        assert activation.beta == 0.0
        assert activation.limit == torch.finfo(torch.float32).max
        # The canonical SiTU (Kimi-K3) scales, matching the CUTLASS
        # SituAdaptor compile-time defaults.
        assert SiTU() == SiTU(
            gate_scale=DEFAULT_SITU_BETA, linear_scale=DEFAULT_SITU_LINEAR_BETA
        )
        assert SwiGLUStep().limit == 7.0


@pytest.mark.parametrize("activation", (SwiGLU(), ReLU2()))
def test_trtllm_bf16_preparation_shapes_for_declared_activations(activation):
    experts, hidden, intermediate = 2, 128, 128
    rows = intermediate * (2 if activation.is_gated else 1)
    view = TrtllmBf16Config.prepare_weights(
        torch.randn(experts, rows, hidden, dtype=torch.bfloat16),
        torch.randn(experts, hidden, intermediate, dtype=torch.bfloat16),
        num_local_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation=activation,
        device="cpu",
    )
    assert view["gemm1_weights"].shape[0] == experts
    assert view["gemm1_weights"].numel() == experts * rows * hidden
    assert view["gemm2_weights"].numel() == experts * hidden * intermediate


@pytest.mark.parametrize("activation", (SwiGLU(), ReLU2()))
def test_trtllm_fp8_per_tensor_preparation_shapes_for_declared_activations(
    activation,
):
    experts, hidden, intermediate = 2, 128, 128
    rows = intermediate * (2 if activation.is_gated else 1)
    view = TrtllmFp8PerTensorConfig.prepare_weights(
        torch.randn(experts, rows, hidden, dtype=torch.bfloat16),
        torch.randn(experts, hidden, intermediate, dtype=torch.bfloat16),
        hidden_states_scale_global=1.0,
        intermediate_scale_global=1.0,
        num_local_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation=activation,
        device="cpu",
    )
    assert view["gemm1_weights"].shape == (experts, rows, hidden)
    assert view["gemm2_weights"].shape == (experts, hidden, intermediate)


# ---------------------------------------------------------------------------
# Expressiveness: can we represent the existing test configurations?
# ---------------------------------------------------------------------------


class TestExpressiveness:
    """Verify that the unified config can express every existing test scenario.

    Each scenario maps a legacy flat-API configuration onto the single-knob
    ``QuantVariant`` surface.
    """

    def test_trtllm_fp4_deepseekv3(self):
        """The most common DeepSeek-V3 FP4 config from test_trtllm_gen_fused_moe.py."""
        cfg = MoEConfig(
            routing=RoutingConfig(
                num_experts=256,
                top_k=8,
                method=RoutingMethodType.DeepSeekV3,
                n_group=8,
                topk_group=4,
                routed_scaling_factor=1.0,
            ),
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=ExpertConfig(intermediate_size=1024),
            activation=SwiGLU(),
            backend=BackendOptions(
                candidates=(TrtllmFp4Config(), CutlassNvfp4Config())
            ),
        )
        assert cfg.routing.method == RoutingMethodType.DeepSeekV3
        assert cfg.quant.variant == QuantVariant.NVFP4
        assert cfg.activation.is_gated

    def test_trtllm_fp8_block_mxfp8(self):
        """MxFP8 block-scale config."""
        cfg = MoEConfig(
            routing=RoutingConfig(
                num_experts=64,
                top_k=8,
                method=RoutingMethodType.Renormalize,
            ),
            quant=QuantConfig(variant=QuantVariant.MxFp8),
            experts=ExpertConfig(intermediate_size=512),
            activation=SwiGLU(),
            backend=BackendOptions(
                candidates=(TrtllmFp8BlockConfig(), CutlassMxfp8Config())
            ),
        )
        assert cfg.quant.variant == QuantVariant.MxFp8

    def test_trtllm_fp8_per_tensor(self):
        """Per-tensor FP8 config."""
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=8, top_k=2),
            quant=QuantConfig(variant=QuantVariant.FP8PerTensor),
            experts=ExpertConfig(intermediate_size=512),
            backend=BackendOptions(
                candidates=(TrtllmFp8PerTensorConfig(), CutlassFp8PerTensorConfig())
            ),
        )
        assert cfg.quant.variant == QuantVariant.FP8PerTensor

    def test_trtllm_bf16(self):
        """BF16 unquantized config."""
        cfg = MoEConfig(
            routing=RoutingConfig(
                num_experts=8,
                top_k=2,
                method=RoutingMethodType.Renormalize,
            ),
            quant=QuantConfig(variant=QuantVariant.BF16),
            experts=ExpertConfig(intermediate_size=512),
            backend=BackendOptions(
                candidates=(TrtllmBf16Config(), CutlassBf16Config())
            ),
        )
        assert cfg.quant.variant == QuantVariant.BF16

    def test_trtllm_mxint4(self):
        """MxInt4 config."""
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=8, top_k=2),
            quant=QuantConfig(variant=QuantVariant.MxInt4),
            experts=ExpertConfig(intermediate_size=512),
            backend=BackendOptions((TrtllmMxInt4Config(),)),
        )
        assert cfg.quant.variant == QuantVariant.MxInt4

    def test_cutlass_modular_fp8(self):
        """CUTLASS DeepSeek block-scale FP8 config."""
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=64, top_k=8),
            quant=QuantConfig(variant=QuantVariant.DeepSeekFp8),
            experts=ExpertConfig(intermediate_size=2048),
            activation=SwiGLU(),
            backend=BackendOptions((CutlassFp8BlockConfig(),)),
        )
        assert any(isinstance(c, CutlassFp8BlockConfig) for c in cfg.backend)

    def test_cutedsl_nvfp4(self):
        """CuteDSL NVFP4 config."""
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=64, top_k=8),
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=ExpertConfig(intermediate_size=1024),
            activation=SwiGLU(),
            backend=BackendOptions(candidates=(CuteDslConfig(), CutlassNvfp4Config())),
        )
        assert any(isinstance(c, CuteDslConfig) for c in cfg.backend)

    def test_expert_parallel(self):
        """Config with expert parallelism (EP)."""
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=256, top_k=8),
            quant=QuantConfig(variant=QuantVariant.DeepSeekFp8),
            experts=ExpertConfig(
                intermediate_size=2048,
                local_expert_offset=32,
                local_num_experts=32,
            ),
        )
        assert cfg.experts.local_expert_offset == 32
        assert cfg.experts.local_num_experts == 32

    def test_llama4_routing(self):
        """Llama4 top-1 sigmoid routing."""
        cfg = MoEConfig(
            routing=RoutingConfig(
                num_experts=16,
                top_k=1,
                method=RoutingMethodType.Llama4,
            ),
            quant=QuantConfig(variant=QuantVariant.BF16),
            experts=ExpertConfig(intermediate_size=4096),
        )
        assert cfg.routing.method == RoutingMethodType.Llama4
        assert cfg.routing.top_k == 1

    def test_qwen3_renormalize_naive(self):
        """Qwen3 RenormalizeNaive routing."""
        cfg = MoEConfig(
            routing=RoutingConfig(
                num_experts=64,
                top_k=8,
                method=RoutingMethodType.RenormalizeNaive,
            ),
            quant=QuantConfig(variant=QuantVariant.DeepSeekFp8),
            experts=ExpertConfig(intermediate_size=1024),
        )
        assert cfg.routing.method == RoutingMethodType.RenormalizeNaive


# ---------------------------------------------------------------------------
# Unified runner support validation
# ---------------------------------------------------------------------------


class TestMoERunnerSupport:
    def test_registered_runners_declare_activation_capabilities(self):
        for runner_cls in set(_BACKEND_RUNNERS.values()):
            by_quant = runner_cls.supported_activation_classes_by_quant
            if by_quant:
                assert set(by_quant) == set(runner_cls.supported_quant_variants)
                assert all(by_quant.values())
            else:
                assert runner_cls.supported_activation_classes, (
                    f"{runner_cls.__name__} must explicitly declare supported "
                    "activation classes"
                )

    def test_declarative_activation_capabilities(self):
        cutlass = (
            SwiGLU,
            SwiGLUStep,
            GeGLU,
            GeGLUTanh,
            ReLU2,
            SiTU,
            Identity,
            GELU,
            ReLU,
            SiLU,
        )
        assert CutlassBf16Runner.supported_activation_classes == cutlass
        assert CutlassW4A16Runner.supported_activation_classes == cutlass
        assert CuteDslRunner.supported_activation_classes == (
            SwiGLU,
            GeGLUTanh,
            ReLU2,
            SiTU,
        )
        assert TrtllmFp4RoutedRunner.supported_activation_classes_by_quant == {
            QuantVariant.NVFP4: (SwiGLU, GeGLU, SiTU, ReLU2),
            QuantVariant.MXFP4: (SwiGLU, GeGLU, SiTU, ReLU2),
            QuantVariant.W4A16: (SwiGLU,),
        }
        assert TrtllmBf16RoutedRunner.supported_activation_classes == (
            SwiGLU,
            ReLU2,
        )
        assert TrtllmFp8PerTensorRunner.supported_activation_classes == (
            SwiGLU,
            ReLU2,
        )
        assert TrtllmFp8BlockRunner.supported_activation_classes_by_quant == {
            QuantVariant.DeepSeekFp8: (SwiGLU,),
            QuantVariant.MxFp8: (SwiGLU, GeGLU, ReLU2),
        }
        assert TrtllmMxInt4RoutedRunner.supported_activation_classes == (SwiGLU,)

    def _nvfp4_swiglu(self, **overrides):
        base = dict(
            routing=RoutingConfig(num_experts=32, top_k=2),
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=ExpertConfig(intermediate_size=512),
            activation=SwiGLU(),
        )
        base.update(overrides)
        return MoEConfig(**base)

    @pytest.mark.parametrize(
        ("compute_capability", "supported"),
        [((10, 0), True), ((10, 3), True), ((10, 7), False)],
    )
    def test_trtllm_fp4_situ_rejected_on_rubin(
        self, monkeypatch, compute_capability, supported
    ):
        """SM107 must reject SiTU while the Rubin BMM pin predates SiTuGlu.

        Rubin's gemmGatedAct::ActType is {SwiGlu, GeGlu, None}, so None holds
        the value SiTuGlu carries elsewhere. activationTypeToGatedActType still
        maps Situ to it and the static asserts are compiled out under
        TLLM_RUBIN_FEATURES, so without this guard SM107 would run SiTU with no
        activation at all rather than fail.
        """
        import flashinfer.utils as utils

        runner = TrtllmFp4RoutedRunner.__new__(TrtllmFp4RoutedRunner)
        runner.config = self._nvfp4_swiglu(activation=SiTU())
        runner.device = torch.device("cuda")
        monkeypatch.setattr(
            utils, "get_compute_capability", lambda _: compute_capability
        )
        if supported:
            assert runner.check_support() is None
        else:
            with pytest.raises(NotImplementedError, match="SiTU on SM107"):
                runner.check_support()

    @pytest.mark.parametrize(
        "variant", (QuantVariant.NVFP4, QuantVariant.MXFP4, QuantVariant.W4A16)
    )
    def test_cute_dsl_quant_variants_supported(self, variant):
        runner = CuteDslRunner.__new__(CuteDslRunner)
        runner.config = self._nvfp4_swiglu(quant=QuantConfig(variant=variant))
        assert runner.check_support() is None

    def test_cute_dsl_w4a8_rejected_on_rubin(self, monkeypatch):
        import flashinfer.utils as utils

        runner = CuteDslRunner.__new__(CuteDslRunner)
        runner.config = self._nvfp4_swiglu(
            quant=QuantConfig(variant=QuantVariant.MXFP4)
        )
        runner.device = torch.device("cuda")
        monkeypatch.setattr(utils, "get_compute_capability", lambda _: (10, 7))
        with pytest.raises(NotImplementedError, match=r"W4A8.*SM107"):
            runner.check_support()

    def test_cute_dsl_w4a8_requires_fused_finalize(self):
        runner = CuteDslRunner.__new__(CuteDslRunner)
        runner.config = self._nvfp4_swiglu(
            quant=QuantConfig(variant=QuantVariant.MXFP4),
            finalize=MoEFinalizeConfig(use_fused_finalize=False),
        )
        with pytest.raises(NotImplementedError, match="requires fused finalize"):
            runner.check_support()

    def test_cute_dsl_rejects_gated_rows_for_non_gated_activation(self):
        """A ReLU2 config paired with a default-prepared (SwiGLU) view.

        prepare_weights defaults to SwiGLU, so the view carries 2I rows while
        the config wants I. The tuner infers intermediate_size from this tensor,
        so without a boundary check the mismatch surfaces deep in the kernel.
        """
        runner = CuteDslRunner.__new__(CuteDslRunner)
        runner.config = self._nvfp4_swiglu(activation=ReLU2())
        runner._built = True
        runner._inner = SimpleNamespace(top_k=2)

        intermediate = runner.config.experts.intermediate_size
        weights = MoEWeightPack()
        weights.prepare_for(
            "cute_dsl",
            {"w1_weight": torch.empty(32, 2 * intermediate, 64, dtype=torch.uint8)},
        )
        act = MoEActivationPack(
            hidden_states_q=torch.empty(4, 64, dtype=torch.uint8),
            hidden_states_scale=torch.empty(4, 4, dtype=torch.uint8),
            topk_ids=torch.zeros(4, 2, dtype=torch.int32),
            topk_weights=torch.ones(4, 2, dtype=torch.bfloat16),
        )
        with pytest.raises(ValueError, match="GEMM1 rows"):
            runner.pack_inputs(act, weights)

    def test_cute_dsl_mxfp4_pack_uses_unpacked_mxfp8_and_no_fc2_scale(self):
        runner = CuteDslRunner.__new__(CuteDslRunner)
        runner.config = self._nvfp4_swiglu(
            quant=QuantConfig(variant=QuantVariant.MXFP4)
        )
        runner._built = True
        runner._inner = SimpleNamespace(top_k=2)
        intermediate = runner.config.experts.intermediate_size
        weights = MoEWeightPack()
        weights.prepare_for(
            "cute_dsl",
            {
                "w1_weight": torch.empty(32, 2 * intermediate, 64, dtype=torch.uint8),
                "w1_weight_sf": torch.empty(1, dtype=torch.uint8),
                "w1_alpha": torch.ones(32),
                "w2_weight": torch.empty(32, 128, intermediate // 2, dtype=torch.uint8),
                "w2_weight_sf": torch.empty(1, dtype=torch.uint8),
                "w2_alpha": torch.ones(32),
            },
        )
        act = MoEActivationPack(
            hidden_states_q=torch.empty(4, 128, dtype=torch.float8_e4m3fn),
            hidden_states_scale=torch.empty(4, 4, dtype=torch.uint8),
            topk_ids=torch.zeros(4, 2, dtype=torch.int32),
            topk_weights=torch.ones(4, 2, dtype=torch.float32),
        )
        packed = runner.pack_inputs(act, weights)
        assert packed[0].shape == (4, 128)
        assert packed[1].shape == (4, 4)
        assert packed[7] is None
        assert packed[-1].shape == (4, 128)

    def test_cute_dsl_mxfp4_pack_rejects_unaligned_geometry(self):
        w1 = torch.zeros(2, 2 * 96, 256, dtype=torch.bfloat16)
        w2 = torch.zeros(2, 256, 96, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="divisible by 128"):
            CuteDslConfig.prepare_weights(
                w1,
                w2,
                variant=QuantVariant.MXFP4,
                num_local_experts=2,
                hidden_size=256,
                intermediate_size=96,
                device="cpu",
            )

    def test_cute_dsl_rejects_unrepresentable_situ_clamp(self):
        runner = CuteDslRunner.__new__(CuteDslRunner)
        runner.config = self._nvfp4_swiglu(activation=SiTU(clamp_limit=4.0))
        with pytest.raises(NotImplementedError, match="clamp_limit"):
            runner.check_support()

    def test_cute_dsl_accepts_unclamped_situ_linear_branch(self):
        runner = CuteDslRunner.__new__(CuteDslRunner)
        runner.config = self._nvfp4_swiglu(activation=SiTU(linear_scale=None))
        assert runner.check_support() is None

    def test_trtllm_rejects_unclamped_situ_linear_branch(self):
        # The TRT-LLM per-expert gemm1_beta tensor cannot encode "no clamp",
        # so the mode CuTe-DSL accepts must be rejected here rather than
        # silently dropping the parameter.
        runner = TrtllmFp4RoutedRunner.__new__(TrtllmFp4RoutedRunner)
        runner.config = self._nvfp4_swiglu(activation=SiTU(linear_scale=None))
        with pytest.raises(NotImplementedError, match="linear_scale=None"):
            runner.check_support()

    def test_b12x_construction_does_not_validate_activation(self):
        # Regression: resolving the b12x activation name in __init__ raised
        # before backend selection could filter the runner, so an unsupported
        # activation aborted MoELayer construction instead of falling back to
        # another backend that supports it.
        from flashinfer.fused_moe.runners import B12xNvfp4Runner

        config = self._nvfp4_swiglu(activation=GeGLU())
        runner = B12xNvfp4Runner(config, device=torch.device("cpu"))
        assert runner.activation is None

    def test_missing_per_quant_capability_entry_is_rejected(self):
        # A runner declaring per-quant activation support must declare it for
        # every variant it accepts; an unmapped variant must not fall back to
        # the permissive class default.
        class _UnmappedRunner(TrtllmFp4RoutedRunner):
            supported_quant_variants: ClassVar[tuple[QuantVariant, ...]] = (
                QuantVariant.NVFP4,
                QuantVariant.MXFP4,
            )
            supported_activation_classes_by_quant: ClassVar[dict] = {
                QuantVariant.NVFP4: (SwiGLU,),
            }

        runner = _UnmappedRunner.__new__(_UnmappedRunner)
        runner.config = self._nvfp4_swiglu(
            quant=QuantConfig(variant=QuantVariant.MXFP4), activation=GeGLU()
        )
        with pytest.raises(NotImplementedError, match="no entry for QuantVariant"):
            runner.check_support()

    @pytest.mark.parametrize(
        "runner_type,variant",
        (
            (CuteDslRunner, QuantVariant.BF16),
            (TrtllmFp4RoutedRunner, QuantVariant.BF16),
            (TrtllmBf16RoutedRunner, QuantVariant.NVFP4),
            (TrtllmFp8BlockRunner, QuantVariant.BF16),
        ),
    )
    def test_unsupported_quant_variant_rejected(self, runner_type, variant):
        cfg = self._nvfp4_swiglu(quant=QuantConfig(variant=variant))
        runner = runner_type.__new__(runner_type)
        runner.config = cfg
        with pytest.raises(NotImplementedError, match=f"QuantVariant.{variant.name}"):
            runner.check_support()

    @pytest.mark.parametrize(
        "act",
        (Identity(),),
    )
    @pytest.mark.parametrize(
        "runner_type,variant",
        (
            (CuteDslRunner, QuantVariant.NVFP4),
            (TrtllmFp4RoutedRunner, QuantVariant.NVFP4),
            (TrtllmBf16RoutedRunner, QuantVariant.BF16),
            (TrtllmFp8BlockRunner, QuantVariant.DeepSeekFp8),
        ),
    )
    def test_not_supported_activation(self, runner_type, variant, act):
        cfg = self._nvfp4_swiglu(
            quant=QuantConfig(variant=variant),
            activation=act,
        )
        runner = runner_type.__new__(runner_type)
        runner.config = cfg
        with pytest.raises(NotImplementedError, match="supported activations"):
            runner.check_support()

    def test_fp8_block_unfinalized_supported(self, monkeypatch):
        import flashinfer.utils as utils

        cfg = self._nvfp4_swiglu(
            quant=QuantConfig(variant=QuantVariant.DeepSeekFp8),
            finalize=MoEFinalizeConfig(do_finalize=False),
        )
        runner = TrtllmFp8BlockRunner.__new__(TrtllmFp8BlockRunner)
        runner.config = cfg
        runner.device = torch.device("cuda")
        monkeypatch.setattr(utils, "get_compute_capability", lambda _: (10, 0))
        runner.check_support()

    @pytest.mark.parametrize(
        ("runner_type", "variant"),
        [
            (TrtllmFp8BlockRunner, QuantVariant.DeepSeekFp8),
            (TrtllmFp8PerTensorRunner, QuantVariant.FP8PerTensor),
        ],
    )
    @pytest.mark.parametrize(
        ("compute_capability", "supported"),
        [((10, 0), True), ((10, 3), True), ((10, 7), False)],
    )
    def test_fp8_runner_arch_support(
        self, monkeypatch, runner_type, variant, compute_capability, supported
    ):
        import flashinfer.utils as utils

        cfg = self._nvfp4_swiglu(quant=QuantConfig(variant=variant))
        runner = runner_type.__new__(runner_type)
        runner.config = cfg
        runner.device = torch.device("cuda")
        monkeypatch.setattr(
            utils, "get_compute_capability", lambda _: compute_capability
        )
        if supported:
            runner.check_support()
        else:
            with pytest.raises(NotImplementedError, match="SM100/SM103"):
                runner.check_support()

    def test_bf16_unfinalized_supported(self, monkeypatch):
        import flashinfer.utils as utils

        cfg = self._nvfp4_swiglu(
            quant=QuantConfig(variant=QuantVariant.BF16),
            finalize=MoEFinalizeConfig(do_finalize=False),
        )
        runner = TrtllmBf16RoutedRunner.__new__(TrtllmBf16RoutedRunner)
        runner.config = cfg
        runner.device = torch.device("cuda")
        monkeypatch.setattr(utils, "get_compute_capability", lambda _: (10, 0))
        runner.check_support()

    def test_bf16_sm120_rejected_before_launch(self, monkeypatch):
        import flashinfer.utils as utils

        cfg = self._nvfp4_swiglu(quant=QuantConfig(variant=QuantVariant.BF16))
        runner = TrtllmBf16RoutedRunner.__new__(TrtllmBf16RoutedRunner)
        runner.config = cfg
        runner.device = torch.device("cuda")
        monkeypatch.setattr(utils, "get_compute_capability", lambda _: (12, 0))
        with pytest.raises(NotImplementedError, match="SM100/SM103/SM107"):
            runner.check_support()

    def test_bf16_sm107_supported_after_reland(self, monkeypatch):
        import flashinfer.utils as utils

        cfg = self._nvfp4_swiglu(quant=QuantConfig(variant=QuantVariant.BF16))
        runner = TrtllmBf16RoutedRunner.__new__(TrtllmBf16RoutedRunner)
        runner.config = cfg
        runner.device = torch.device("cuda")
        monkeypatch.setattr(utils, "get_compute_capability", lambda _: (10, 7))
        runner.check_support()

    @pytest.mark.parametrize(
        "variant",
        [QuantVariant.NVFP4, QuantVariant.MXFP4, QuantVariant.W4A16],
    )
    def test_fp4_sm107_variant_support_after_reland(self, monkeypatch, variant):
        import flashinfer.utils as utils

        cfg = self._nvfp4_swiglu(quant=QuantConfig(variant=variant))
        runner = TrtllmFp4RoutedRunner.__new__(TrtllmFp4RoutedRunner)
        runner.config = cfg
        runner.device = torch.device("cuda")
        monkeypatch.setattr(utils, "get_compute_capability", lambda _: (10, 7))
        runner.check_support()

    def test_moe_runner_quant_support_check(self):
        class Runner(MoERunner):
            supported_quant_variants = (QuantVariant.NVFP4,)
            supported_activation_classes = (SwiGLU,)

            def get_valid_tactics(self, inputs, profile):
                return []

            def forward(self, inputs, **kwargs):
                return None

        runner = Runner()
        runner.config = self._nvfp4_swiglu()
        assert runner.check_support() is None

    def test_moe_runner_without_activation_capability_is_rejected(self):
        class Runner(MoERunner):
            supported_quant_variants = (QuantVariant.NVFP4,)

            def get_valid_tactics(self, inputs, profile):
                return []

            def forward(self, inputs, **kwargs):
                return None

        runner = Runner()
        runner.config = self._nvfp4_swiglu()
        with pytest.raises(
            NotImplementedError, match="declares no supported activation classes"
        ):
            runner.check_support()


class TestBuiltInRunnerLifecycle:
    @staticmethod
    def _config(variant):
        return MoEConfig(
            routing=RoutingConfig(num_experts=32, top_k=2),
            quant=QuantConfig(variant=variant),
            experts=ExpertConfig(intermediate_size=512),
            activation=SwiGLU(),
            execution=ExecutionConfig(enable_pdl=False),
        )

    @pytest.mark.parametrize(
        "runner_type,variant",
        (
            (TrtllmFp4RoutedRunner, QuantVariant.NVFP4),
            (TrtllmFp8BlockRunner, QuantVariant.DeepSeekFp8),
            (TrtllmFp8PerTensorRunner, QuantVariant.FP8PerTensor),
            (TrtllmBf16RoutedRunner, QuantVariant.BF16),
            (TrtllmMxInt4RoutedRunner, QuantVariant.MxInt4),
        ),
    )
    def test_trtllm_constructor_defers_idempotent_module_build(
        self, monkeypatch, runner_type, variant
    ):
        from flashinfer.fused_moe import core

        module = object()
        loads = []

        def load_module():
            loads.append("module")
            return module

        monkeypatch.setattr(core, "get_trtllm_moe_sm100_module", load_module)
        runner = runner_type(self._config(variant), torch.device("cuda:0"))
        runner._check_support = lambda: None

        assert runner._module is None
        assert loads == []

        runner.check_support()
        runner.build()
        runner.build()

        assert runner._module is module
        assert runner._built
        assert loads == ["module"]

    def test_cute_dsl_constructor_defers_idempotent_inner_build(self, monkeypatch):
        from flashinfer.fused_moe.cute_dsl import fused_moe, tuner

        events = []
        tuning_config = object()

        class Inner:
            def __init__(self, **kwargs):
                events.append(("build", kwargs))
                self.tuning_config = tuning_config
                self.use_fused_finalize = kwargs["use_fused_finalize"]
                self.enable_pdl = kwargs["enable_pdl"]

        monkeypatch.setattr(tuner, "CuteDslFusedMoERunner", Inner)
        monkeypatch.setattr(fused_moe, "_cute_dsl_fused_moe_impl", object())
        runner = CuteDslRunner(self._config(QuantVariant.NVFP4), torch.device("cuda:0"))
        runner._check_support = lambda: None

        assert runner._inner is None
        assert events == []

        runner.check_support()
        runner.build()
        runner.build()

        assert len(events) == 1
        assert runner._built
        assert runner.tuning_config is tuning_config
        assert hash(runner) == hash(runner.get_cache_key_extras([]))


# ---------------------------------------------------------------------------
# MoEActivationPack construction + runner-boundary validation (CPU-only)
# ---------------------------------------------------------------------------
# The runner helpers are tested DIRECTLY (private imports) on purpose: the
# public path (pack_inputs) needs a JIT'd runner + GPU, which would push these
# regressions out of the always-on CPU tier.


def _pack_tensors(num_tokens=4, top_k=2, hidden_packed=8, num_experts=16):
    x = torch.zeros(num_tokens, hidden_packed, dtype=torch.uint8)
    sf = torch.zeros(num_tokens, 1, dtype=torch.uint8)
    ids = torch.zeros(num_tokens, top_k, dtype=torch.int32)
    w = torch.ones(num_tokens, top_k)
    logits = torch.zeros(num_tokens, num_experts, dtype=torch.float32)
    return x, sf, ids, w, logits


class TestActivationPackValidation:
    """``MoEActivationPack.__post_init__`` contract (raises, survives -O)."""

    def test_valid_prerouted_and_positional_compat(self):
        x, sf, ids, w, _ = _pack_tensors()
        pack = MoEActivationPack(x, sf, ids, w)  # positional, pre-rename order
        assert pack.topk_ids is ids and pack.topk_weights is w

    @pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
    def test_valid_unpacked_prerouted(self, weights_dtype):
        from flashinfer.fused_moe.core import RoutingInputMode

        x, sf, ids, w, _ = _pack_tensors()
        pack = MoEActivationPack(
            x,
            sf,
            ids,
            w.to(weights_dtype),
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        )
        assert pack.topk_ids is ids
        assert pack.topk_weights.dtype is weights_dtype

    def test_routing_fields_are_keyword_only(self):
        x, sf, ids, w, _ = _pack_tensors()
        with pytest.raises(TypeError):
            MoEActivationPack(x, sf, ids, w, torch.zeros(4, 16))

    def test_valid_fromlogits_mixed_dtypes(self):
        from flashinfer.fused_moe.core import RoutingInputMode

        x, sf, _, _, logits = _pack_tensors()
        # fp32 logits + bf16 bias is the standard DeepSeek-V3 shape; dtypes
        # are independent (test_routing_dtype_flexibility).
        pack = MoEActivationPack(
            x,
            sf,
            routing_input_mode=RoutingInputMode.FromLogits,
            routing_logits=logits,
            routing_bias=torch.zeros(16, dtype=torch.bfloat16),
        )
        assert pack.topk_ids is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(topk_ids=None),  # missing ids in pre-routed
            dict(topk_weights=None),  # missing weights in pre-routed
            dict(routing_logits="LOGITS"),  # logits smuggled into pre-routed
            dict(routing_bias="BIAS"),  # bias smuggled into pre-routed
        ],
    )
    def test_prerouted_field_mismatch_raises(self, kwargs):
        x, sf, ids, w, logits = _pack_tensors()
        fields = dict(topk_ids=ids, topk_weights=w)
        for k, v in kwargs.items():
            fields[k] = (
                logits
                if v == "LOGITS"
                else torch.zeros(16, dtype=torch.bfloat16)
                if v == "BIAS"
                else v
            )
        with pytest.raises(ValueError):
            MoEActivationPack(x, sf, **fields)

    def test_fromlogits_field_mismatch_raises(self):
        from flashinfer.fused_moe.core import RoutingInputMode

        x, sf, ids, w, logits = _pack_tensors()
        with pytest.raises(ValueError):  # missing logits
            MoEActivationPack(x, sf, routing_input_mode=RoutingInputMode.FromLogits)
        with pytest.raises(ValueError):  # topk fields must stay None
            MoEActivationPack(
                x,
                sf,
                ids,
                w,
                routing_input_mode=RoutingInputMode.FromLogits,
                routing_logits=logits,
            )

    @pytest.mark.parametrize(
        ("ids_transform", "weights_transform", "match"),
        [
            pytest.param(
                lambda x: x.long(),
                lambda x: x.to(torch.bfloat16),
                "int32",
                id="int64-ids",
            ),
            pytest.param(
                lambda x: x,
                lambda x: x.half(),
                "bfloat16",
                id="fp16-weights",
            ),
            pytest.param(
                lambda x: x[:, :1],
                lambda x: x.to(torch.bfloat16),
                "matching",
                id="mismatched-shapes",
            ),
            pytest.param(
                lambda x: x,
                lambda x: x.to(torch.bfloat16)[:, :1].expand_as(x),
                "contiguous",
                id="noncontiguous-weights",
            ),
        ],
    )
    def test_unpacked_contract_rejected(self, ids_transform, weights_transform, match):
        """Reject invalid unpacked dtypes, shapes, and layouts before launch."""
        from flashinfer.fused_moe.core import RoutingInputMode

        x, sf, ids, w, _ = _pack_tensors()
        with pytest.raises((TypeError, ValueError), match=match):
            MoEActivationPack(
                x,
                sf,
                ids_transform(ids),
                weights_transform(w),
                routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
            )

    def test_int64_topk_ids_rejected(self):
        # torch.topk returns int64; the launcher casts data_ptr without a
        # dtype ICHECK, so int64 reaching it is read as int32 bytes (silent
        # garbage routing) -- must fail loudly at construction.
        x, sf, ids, w, _ = _pack_tensors()
        with pytest.raises(TypeError, match="int32"):
            MoEActivationPack(x, sf, ids.long(), w)

    @pytest.mark.parametrize(
        "field_name",
        ["topk_ids", "topk_weights", "hidden_states_scale", "per_token_scale"],
    )
    def test_device_mismatch_rejected(self, field_name):
        # meta-device tensors give a second device without needing a GPU.
        x, sf, ids, w, _ = _pack_tensors()
        fields = dict(
            hidden_states_scale=sf,
            topk_ids=ids,
            topk_weights=w,
            per_token_scale=torch.ones(x.shape[0]),
        )
        t = fields[field_name]
        fields[field_name] = torch.zeros(t.shape, dtype=t.dtype, device="meta")
        with pytest.raises(ValueError, match="device"):
            MoEActivationPack(x, **fields)


class TestRunnerBoundaryValidation:
    """The shared ``_validate_*`` helpers, called directly (CPU, no JIT).

    They duplicate ``__post_init__`` BY DESIGN: the pack is mutable, so the
    launch boundary is the authoritative validation layer. The mutation tests
    below pin exactly the bypass that motivates the duplication -- do not
    "deduplicate" these checks against ``__post_init__``.
    """

    def test_trtllm_fp4_rejects_ungated_gemm1_rows_below_epilogue_tile(self):
        """A non-gated activation halves GEMM1 rows past the tile constraint.

        intermediate_size=64 clears the 16-element NVFP4 alignment check, and
        SwiGLU is fine because gating gives 128 rows. ReLU2 gives 64, which the
        scale permutation cannot tile -- previously a bare AssertionError from
        inside the permutation rather than a diagnosable rejection.
        """
        E, H, I = 2, 128, 64
        w2 = torch.empty(E, H, I, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="GEMM1 rows divisible by 128"):
            TrtllmFp4Config.prepare_weights(
                torch.empty(E, I, H, dtype=torch.bfloat16),
                w2,
                num_local_experts=E,
                hidden_size=H,
                intermediate_size=I,
                activation=ReLU2(),
            )

    def test_prerouted_valid_passes(self):
        from flashinfer.fused_moe.runners import _validate_prerouted_inputs

        x, sf, ids, w, _ = _pack_tensors()
        _validate_prerouted_inputs(MoEActivationPack(x, sf, ids, w), 4, 2, "T")

    def test_prerouted_column_mismatch_raises(self):
        from flashinfer.fused_moe.runners import _validate_prerouted_inputs

        x, sf, _, _, _ = _pack_tensors()
        ids3 = torch.zeros(4, 3, dtype=torch.int32)
        w3 = torch.ones(4, 3)
        pack = MoEActivationPack(x, sf, ids3, w3)
        # config top_k=2 but the pack carries 3 columns: mis-packs against the
        # kernel's top_k-sized buffers.
        with pytest.raises(ValueError, match="top_k"):
            _validate_prerouted_inputs(pack, 4, 2, "T")

    def test_mutation_to_int64_caught_at_runner_boundary(self):
        from flashinfer.fused_moe.runners import _validate_prerouted_inputs

        x, sf, ids, w, _ = _pack_tensors()
        pack = MoEActivationPack(x, sf, ids, w)  # valid at construction
        pack.topk_ids = pack.topk_ids.long()  # bypasses __post_init__
        with pytest.raises(TypeError, match="int32"):
            _validate_prerouted_inputs(pack, 4, 2, "T")

    def test_mutation_smuggling_logits_caught_at_runner_boundary(self):
        from flashinfer.fused_moe.runners import _validate_prerouted_inputs

        x, sf, ids, w, logits = _pack_tensors()
        pack = MoEActivationPack(x, sf, ids, w)
        pack.routing_logits = logits  # bypasses __post_init__
        with pytest.raises(ValueError, match="FromLogits"):
            _validate_prerouted_inputs(pack, 4, 2, "T")

    def test_prerouted_device_mutation_caught_at_runner_boundary(self):
        from flashinfer.fused_moe.runners import _validate_prerouted_inputs

        x, sf, ids, w, _ = _pack_tensors()
        pack = MoEActivationPack(x, sf, ids, w)
        pack.topk_ids = torch.zeros(
            pack.topk_ids.shape, dtype=torch.int32, device="meta"
        )
        with pytest.raises(ValueError, match="device"):
            _validate_prerouted_inputs(pack, 4, 2, "T")

    def _logits_pack(self, logits, bias=None):
        from flashinfer.fused_moe.core import RoutingInputMode

        x, sf, _, _, _ = _pack_tensors()
        return MoEActivationPack(
            x,
            sf,
            routing_input_mode=RoutingInputMode.FromLogits,
            routing_logits=logits,
            routing_bias=bias,
        )

    def test_logits_valid_passes_including_mixed_dtypes(self):
        from flashinfer.fused_moe.runners import _validate_logits_inputs

        pack = self._logits_pack(
            torch.zeros(4, 16, dtype=torch.float32),
            bias=torch.zeros(16, dtype=torch.bfloat16),
        )
        _validate_logits_inputs(pack, 4, 16, "T")

    def test_logits_shape_mismatch_raises(self):
        from flashinfer.fused_moe.runners import _validate_logits_inputs

        pack = self._logits_pack(torch.zeros(4, 9))
        with pytest.raises(ValueError, match="num_experts"):
            _validate_logits_inputs(pack, 4, 16, "T")

    def test_noncontiguous_logits_rejected(self):
        from flashinfer.fused_moe.runners import _validate_logits_inputs

        logits = torch.zeros(16, 4, dtype=torch.float32).T
        assert logits.shape == (4, 16) and not logits.is_contiguous()
        with pytest.raises(ValueError, match="routing_logits must be contiguous"):
            _validate_logits_inputs(self._logits_pack(logits), 4, 16, "T")

    @pytest.mark.parametrize("bad_dtype", [torch.float16, torch.float64])
    def test_logits_dtype_rejected(self, bad_dtype):
        from flashinfer.fused_moe.runners import _validate_logits_inputs

        pack = self._logits_pack(torch.zeros(4, 16, dtype=torch.float32))
        pack.routing_logits = pack.routing_logits.to(bad_dtype)  # mutation
        with pytest.raises(TypeError, match="float32 or bfloat16"):
            _validate_logits_inputs(pack, 4, 16, "T")

    def test_bias_dtype_rejected(self):
        # The launcher maps bf16->Bfloat16 and anything-else->Fp32 with no
        # ICHECK: an fp16 bias would be silently reinterpreted as fp32 bits.
        from flashinfer.fused_moe.runners import _validate_logits_inputs

        pack = self._logits_pack(torch.zeros(4, 16, dtype=torch.float32))
        pack.routing_bias = torch.zeros(16, dtype=torch.float16)  # mutation
        with pytest.raises(TypeError, match="bfloat16 or float32"):
            _validate_logits_inputs(pack, 4, 16, "T")

    def test_bias_shape_rejected(self):
        from flashinfer.fused_moe.runners import _validate_logits_inputs

        pack = self._logits_pack(
            torch.zeros(4, 16, dtype=torch.float32),
            bias=torch.zeros(15, dtype=torch.bfloat16),
        )
        with pytest.raises(ValueError, match="num_experts"):
            _validate_logits_inputs(pack, 4, 16, "T")

    def test_noncontiguous_bias_rejected(self):
        from flashinfer.fused_moe.runners import _validate_logits_inputs

        bias = torch.zeros(32, dtype=torch.bfloat16)[::2]
        assert bias.shape == (16,) and not bias.is_contiguous()
        pack = self._logits_pack(torch.zeros(4, 16), bias=bias)
        with pytest.raises(ValueError, match="routing_bias must be contiguous"):
            _validate_logits_inputs(pack, 4, 16, "T")

    def test_logits_device_mutation_caught_at_runner_boundary(self):
        from flashinfer.fused_moe.runners import _validate_logits_inputs

        pack = self._logits_pack(torch.zeros(4, 16, dtype=torch.float32))
        pack.routing_logits = torch.zeros(
            pack.routing_logits.shape, dtype=torch.float32, device="meta"
        )
        with pytest.raises(ValueError, match="device"):
            _validate_logits_inputs(pack, 4, 16, "T")

    def test_packed_ids_normalize_noncontiguous_inputs(self):
        from flashinfer.fused_moe.runners import _pack_prerouted_topk_ids

        x, sf, _, _, _ = _pack_tensors()
        ids = torch.arange(8, dtype=torch.int32).reshape(2, 4).T
        weights = torch.linspace(-1, 1, 8).reshape(2, 4).T
        assert not ids.is_contiguous() and not weights.is_contiguous()
        packed = _pack_prerouted_topk_ids(MoEActivationPack(x, sf, ids, weights))
        expected_bits = (
            weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
        )
        assert packed.is_contiguous()
        assert torch.equal(packed >> 16, ids)
        assert torch.equal(packed & 0xFFFF, expected_bits)


def _is_unified_nvfp4_arch() -> bool:
    return torch.cuda.is_available() and get_compute_capability(
        torch.device("cuda")
    ) in ((10, 0), (10, 3), (10, 7))


sm100_required = pytest.mark.skipif(
    not _is_unified_nvfp4_arch(),
    reason="Unified NVFP4 MoE requires SM100, SM103, or SM107",
)


cute_dsl_sm100_required = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and get_compute_capability(torch.device("cuda")) in ((10, 0), (10, 3), (10, 7))
    ),
    reason="CuTeDSL unified MoE requires SM100, SM103, or SM107",
)


# Small-scale geometry for fast accuracy + dispatch tests.
SMALL = dict(hidden_size=1024, intermediate_size=512, num_experts=32, top_k=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_packs_and_config(
    num_tokens: int,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    local_num_experts: int | None = None,
    max_tokens: int | None = None,
    activation=None,
    variant: QuantVariant = QuantVariant.NVFP4,
):
    """Build (act_pack, weight_pack, config, tensors_dict) for a given shape.

    ``tensors_dict`` contains the original bf16 reference weights used to
    compute ground truth via ``compute_reference_moe_fp4``.
    """
    local_num_experts = local_num_experts or num_experts
    max_tokens = max_tokens or max(num_tokens, 8192)
    device = torch.device("cuda", torch.cuda.current_device())

    activation = activation or SwiGLU()
    # CuteDSL views come pre-built by the flat-test tensor factories.
    tensor_factory = (
        create_relu2_moe_tensors
        if isinstance(activation, ReLU2)
        else create_moe_tensors
    )
    tensors = tensor_factory(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=local_num_experts,
        top_k=top_k,
    )

    w4a16 = variant is QuantVariant.W4A16
    act_pack = MoEActivationPack(
        hidden_states_q=tensors["x_bf16"] if w4a16 else tensors["x"],
        hidden_states_scale=None if w4a16 else tensors["x_sf"].squeeze(-1),
        topk_ids=tensors["token_selected_experts"],
        topk_weights=tensors["token_final_scales"],
    )

    weight_pack = MoEWeightPack()
    weight_pack.prepare_for(
        "cute_dsl",
        {
            "w1_weight": tensors["w1_weight"],
            "w1_weight_sf": tensors["w1_weight_sf"],
            "w1_alpha": tensors["w1_alpha"],
            "fc2_input_scale": tensors["fc2_input_scale"],
            "w2_weight": tensors["w2_weight"],
            "w2_weight_sf": tensors["w2_weight_sf"],
            "w2_alpha": tensors["w2_alpha"],
        },
    )
    weight_pack.prepare_for(
        "trtllm_fp4_routed",
        TrtllmFp4Config.prepare_weights(
            tensors["w1_weight_bf16"],
            tensors["w2_weight_bf16"],
            variant=variant,
            num_local_experts=local_num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
        ),
    )

    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=variant),
        experts=ExpertConfig(
            intermediate_size=intermediate_size,
            local_num_experts=local_num_experts,
        ),
        activation=activation,
        backend=BackendOptions(candidates=(CuteDslConfig(), TrtllmFp4Config())),
        execution=ExecutionConfig(tune_max_num_tokens=max_tokens),
    )
    return act_pack, weight_pack, config, tensors


# ---------------------------------------------------------------------------
# 1. NVFP4 reference helper
# ---------------------------------------------------------------------------


def _compute_ref(act_pack, tensors, shape, activation=None, wrong_formula=False):
    """bf16 ground-truth MoE output for the given pack + shape.

    ``wrong_formula`` evaluates a deliberately different activation over the
    same weights (plain ReLU where ReLU^2 is called for), for use as a negative
    control that proves a tolerance can distinguish activation formulas.
    """
    activation = activation or SwiGLU()
    activation_kwargs = {
        "activation_type": int(activation.type),
        "wrong_formula": wrong_formula,
    }
    if isinstance(activation, SwiGLU):
        activation_kwargs.update(
            swiglu_alpha=activation.alpha,
            swiglu_beta=activation.beta,
            swiglu_limit=activation.limit,
        )
    elif isinstance(activation, SiTU):
        activation_kwargs.update(
            activation_type=int(ActivationType.Swiglu),
            situ_beta=activation.gate_scale,
            situ_linear_beta=activation.linear_scale,
        )
    return compute_reference_moe_fp4(
        hidden_states=tensors["x_bf16"].float().cuda(),
        gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
        gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
        gemm1_alpha=tensors["w1_alpha"],
        gemm2_alpha=tensors["w2_alpha"],
        token_selected_experts=act_pack.topk_ids,
        token_final_scales=act_pack.topk_weights,
        num_tokens=act_pack.num_tokens,
        num_experts=shape["num_experts"],
        top_k=shape["top_k"],
        hidden_size=shape["hidden_size"],
        intermediate_size=shape["intermediate_size"],
        fc2_input_scale=tensors["fc2_input_scale"],
        **activation_kwargs,
    )


@cute_dsl_sm100_required
@pytest.mark.parametrize("variant", (QuantVariant.NVFP4, QuantVariant.W4A16))
@pytest.mark.parametrize(
    "activation",
    (
        SwiGLU(alpha=1.7, beta=1.0, limit=7.0),
        SiTU(gate_scale=2.0, linear_scale=3.0),
        GeGLUTanh(),
        ReLU2(),
    ),
)
def test_cute_dsl_typed_activation_matches_flat_reference(variant, activation):
    shape = dict(
        hidden_size=1024,
        intermediate_size=512,
        num_experts=8,
        top_k=2,
    )
    act_pack, weight_pack, config, tensors = _make_packs_and_config(
        8,
        activation=activation,
        variant=variant,
        **shape,
    )
    config = dataclasses.replace(
        config,
        backend=BackendOptions(candidates=(CuteDslConfig(),)),
    )
    layer = MoELayer(config)
    runner = layer.runners[0]
    layer._select_winner = lambda *_: (runner, -1)
    actual = layer(act_pack, weight_pack)
    reference = _compute_ref(act_pack, tensors, shape, activation)
    passed, pct, atol = check_accuracy(actual, reference)
    assert passed, (
        f"{activation!r}: {pct * 100:.2f}% within tolerance "
        f"(atol={atol:.4f}) vs flat reference"
    )

    # Use a tighter percentage bound to distinguish activation formulas while
    # tolerating FP4 quantization outliers.
    def _agreement(a: torch.Tensor, b: torch.Tensor) -> float:
        a, b = a.float(), b.float()
        atol = 0.05 + 0.5 * b.std().item()
        close = (a - b).abs() < atol + 0.1 * b.abs()
        return close.float().mean().item()

    # Do not let the percentage allowance absorb non-finite values.
    for name, tensor in (("output", actual), ("reference", reference)):
        assert torch.isfinite(tensor).all(), (
            f"{activation!r}: {name} contains non-finite values "
            f"({(~torch.isfinite(tensor)).sum().item()} of {tensor.numel()})."
        )
    agreement = _agreement(actual, reference)
    assert agreement >= 0.97, (
        f"{activation!r}: only {agreement * 100:.2f}% of elements agree with the "
        f"flat reference under the discriminating bound."
    )

    # Verify that the tighter bound rejects a wrong formula with matching
    # gated/non-gated geometry.
    control = {
        ActivationType.Swiglu: SwiGLU(alpha=0.25, beta=2.0, limit=0.5),
        ActivationType.Situ: SiTU(gate_scale=0.25, linear_scale=0.5),
        ActivationType.GegluTanh: SwiGLU(alpha=0.25, beta=2.0, limit=0.5),
    }.get(activation.type)
    control_reference = (
        # Use plain ReLU as the non-gated control for ReLU2.
        _compute_ref(act_pack, tensors, shape, activation, wrong_formula=True)
        if control is None
        else _compute_ref(act_pack, tensors, shape, control)
    )
    control_agreement = _agreement(actual, control_reference)
    assert control_agreement < 0.97, (
        f"{activation!r}: output also agreed with a deliberately wrong reference "
        f"at {control_agreement * 100:.2f}%; the bound cannot detect a "
        f"wrong-formula regression."
    )


# ---------------------------------------------------------------------------
# 2. Dispatch plumbing
# ---------------------------------------------------------------------------


@sm100_required
class TestUnifiedMoEDispatch:
    """Plumbing tests — invariants MoELayer must guarantee."""

    def test_autotune_visits_all_candidate_backends(self):
        """The autotuner actually profiles every candidate backend.

        Shape-robust: doesn't commit to a specific winner (those change with
        kernel updates), just asserts each backend's `forward` was invoked
        during _select_winner.
        """
        act_pack, weight_pack, config, _ = _make_packs_and_config(256, **SMALL)
        layer = MoELayer(config)
        if len(layer.runners) < 2:
            pytest.skip(
                "cross-backend autotune needs >=2 instantiable backends on "
                "this device/stack (e.g. the installed CuTe DSL cannot "
                "target this arch)"
            )

        # Wrap each runner's forward to count invocations.
        call_counts: dict = {}
        for runner in layer.runners:
            key = runner.backend_key
            call_counts[key] = 0
            original = runner.forward

            def counted(*args, __key=key, __orig=original, **kwargs):
                call_counts[__key] += 1
                return __orig(*args, **kwargs)

            runner.forward = counted  # type: ignore[assignment]

        with autotune(True):
            _ = layer(act_pack, weight_pack)

        assert len(call_counts) >= 2, (
            f"Expected ≥2 candidate backends, got {list(call_counts)}"
        )
        for key, count in call_counts.items():
            assert count > 0, (
                f"Backend {key!r} was never invoked — autotuner skipped it "
                f"(call counts: {call_counts})"
            )


# ---------------------------------------------------------------------------
# 4. BF16 conformance (trtllm_bf16_routed)
# ---------------------------------------------------------------------------
# Pre-routed BF16 through the unified MoELayer. Every assertion here compares
# against an independent fp32 dense reference — deliberately NOT against the
# same kernel driven another way (e.g. EP vs non-EP), which would let a
# numerical bug cancel out.

# Starting point from tests/moe_ep/test_moe_ep_compute_correctness.py: weights at
# ~1/sqrt(fan_in) keep activations O(1), so the fp32-reference vs bf16-kernel gap
# is precision-bound, not scale-bound.  Recalibrate on SM100 if a kernel change
# legitimately shifts the floor.
BF16_RTOL = 3e-2
BF16_ATOL = 3e-2


def _bf16_dense_reference(
    x, w1, w2, selected_experts, final_scales, intermediate_size, expert_offset=0
):
    """fp32 dense MoE authority for the bf16 path.

    trtllm-gen gated-activation convention (same as the dense reference in
    trtllm_gen_fused_moe_utils.py): with ``a = x @ w1.T`` of shape [T, 2I],
    ``x1 = a[:, :I]`` is the linear half, ``x2 = a[:, I:]`` the gate, and the
    SwiGLU output is ``silu(x2) * x1``.  ``w1``/``w2`` hold only this rank's
    LOCAL experts; a token routed to global id ``g`` uses local weight
    ``g - expert_offset``.
    """
    final_scales = final_scales.to(torch.bfloat16).float()
    x32 = x.float()
    out = torch.zeros_like(x32)
    for local_e in range(w1.shape[0]):
        mask = selected_experts == local_e + expert_offset
        if not mask.any():
            continue
        tok, nth = torch.where(mask)
        a = x32[tok] @ w1[local_e].float().t()
        inter = F.silu(a[:, intermediate_size:]) * a[:, :intermediate_size]
        inter = inter.to(torch.bfloat16).float()  # gemm1 output is stored bf16
        expert_out = (inter @ w2[local_e].float().t()).to(torch.bfloat16).float()
        out[tok] += final_scales[tok, nth, None].float() * expert_out
    return out


def _make_bf16_packs_and_config(
    num_tokens: int,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    local_num_experts: int | None = None,
    local_expert_offset: int = 0,
    max_tokens: int | None = None,
    seed: int = 42,
):
    """Build (act_pack, weight_pack, config, tensors_dict) for the bf16 path.

    Mirrors ``_make_packs_and_config`` but with raw bf16 activations — no
    quantization and no scale tensors (the runner reads ``hidden_states_q``
    directly and ignores ``hidden_states_scale``).  ``tensors_dict`` holds the
    UNSHUFFLED weights for ``_bf16_dense_reference``.
    """
    local_num_experts = local_num_experts or num_experts
    max_tokens = max_tokens or max(num_tokens, 8192)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(seed)

    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    w1 = (
        torch.randn(
            local_num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / hidden_size**0.5
    )
    w2 = (
        torch.randn(
            local_num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / intermediate_size**0.5
    )

    # Distinct top-k global expert ids per token, drawn from this rank's local
    # shard [offset, offset + local_num_experts).
    logits = torch.rand(num_tokens, local_num_experts, device=device)
    selected_experts = (
        torch.topk(logits, top_k, dim=-1).indices + local_expert_offset
    ).to(torch.int32)
    # Snap gate weights to the bf16 grid: pack_inputs truncates them to bf16 bits
    # for the packed top-k ids, so unsnapped fp32 scales would add rounding noise
    # the reference cannot see.
    final_scales = torch.rand(num_tokens, top_k, device=device)
    final_scales = (
        (final_scales / final_scales.sum(-1, keepdim=True)).to(torch.bfloat16).float()
    )

    act_pack = MoEActivationPack(
        hidden_states_q=x,  # raw bf16 on this path
        hidden_states_scale=None,  # unused by trtllm_bf16_routed
        topk_ids=selected_experts,
        topk_weights=final_scales,
    )

    weight_pack = MoEWeightPack()
    weight_pack.prepare_for(
        "trtllm_bf16_routed",
        TrtllmBf16Config.prepare_weights(
            w1,
            w2,
            num_local_experts=local_num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=device,
        ),
    )

    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.BF16),
        experts=ExpertConfig(
            intermediate_size=intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
        ),
        activation=SwiGLU(),
        backend=BackendOptions(candidates=(TrtllmBf16Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=max_tokens),
    )
    return act_pack, weight_pack, config, {"x": x, "w1": w1, "w2": w2}


# ---------------------------------------------------------------------------
# 5. Variant-parametrized conformance + packing contract
# ---------------------------------------------------------------------------
# One VariantSpec per executable QuantVariant drives the shared GPU test
# bodies below; the variant shows up in the test id (e.g. ``[nvfp4-128]``).
# Adding a variant (FP8, MxInt4, ...) = register one spec.  ``check``
# deliberately preserves each variant's assertion semantics (percent-within
# for NVFP4's quantization noise, hard rtol/atol for BF16).


def _nvfp4_make(
    num_tokens, *, max_tokens=None, local_num_experts=None, local_expert_offset=0
):
    assert local_expert_offset == 0, "NVFP4 pack builder has no offset support yet"
    return _make_packs_and_config(
        num_tokens,
        max_tokens=max_tokens,
        local_num_experts=local_num_experts,
        **SMALL,
    )


def _nvfp4_ref(act_pack, tensors, expert_offset=0):
    assert expert_offset == 0
    return _compute_ref(act_pack, tensors, SMALL)


def _nvfp4_check(out, ref, label):
    passed, pct, atol = check_accuracy(out, ref)
    assert passed, (
        f"{label}: {pct * 100:.2f}% within tolerance (atol={atol:.4f}) vs reference"
    )


def _bf16_make(
    num_tokens, *, max_tokens=None, local_num_experts=None, local_expert_offset=0
):
    return _make_bf16_packs_and_config(
        num_tokens,
        max_tokens=max_tokens,
        local_num_experts=local_num_experts,
        local_expert_offset=local_expert_offset,
        **SMALL,
    )


def _bf16_ref(act_pack, tensors, expert_offset=0):
    return _bf16_dense_reference(
        tensors["x"],
        tensors["w1"],
        tensors["w2"],
        act_pack.topk_ids,
        act_pack.topk_weights,
        tensors["w2"].shape[-1],  # intermediate_size, derived not hardcoded
        expert_offset=expert_offset,
    )


def _bf16_check(out, ref, label):
    torch.testing.assert_close(out.float(), ref, rtol=BF16_RTOL, atol=BF16_ATOL)


@dataclasses.dataclass(frozen=True)
class VariantSpec:
    """Everything the shared conformance bodies need for one QuantVariant."""

    id: str
    backend_keys: tuple  # runner backend_key strings to exercise directly
    make: Callable  # (num_tokens, *, max_tokens, local_num_experts, local_expert_offset) -> (act, wp, config, tensors)
    reference: Callable  # (act_pack, tensors, expert_offset=0) -> fp32 [T, H]
    check: Callable  # (out, ref, label) -> asserts
    supports_runtime_offset: bool


_VARIANT_SPECS = (
    VariantSpec(
        id="nvfp4",
        backend_keys=("cute_dsl", "trtllm_fp4_routed"),
        make=_nvfp4_make,
        reference=_nvfp4_ref,
        check=_nvfp4_check,
        supports_runtime_offset=False,
    ),
    VariantSpec(
        id="bf16",
        backend_keys=("trtllm_bf16_routed",),
        make=_bf16_make,
        reference=_bf16_ref,
        check=_bf16_check,
        supports_runtime_offset=True,
    ),
)

_variant_params = pytest.mark.parametrize(
    "spec", _VARIANT_SPECS, ids=[s.id for s in _VARIANT_SPECS]
)


@sm100_required
@_variant_params
class TestUnifiedMoEConformance:
    """Every wired backend vs an independent reference, per variant.

    Catches a semantically wrong weight view or pack translation even when
    all backends agree with each other.
    """

    @pytest.mark.parametrize("num_tokens", [128, 512])
    def test_layer_output_matches_reference(self, spec, num_tokens):
        """MoELayer end-to-end output matches the variant's reference."""
        act_pack, weight_pack, config, tensors = spec.make(
            num_tokens, max_tokens=num_tokens
        )
        with autotune(True):
            layer = MoELayer(config)
            out = layer(act_pack, weight_pack)
        spec.check(out, spec.reference(act_pack, tensors), f"{spec.id} MoELayer")

    def test_each_backend_matches_reference(self, spec):
        """Each backend, driven directly (pack_inputs + forward), matches the
        same reference."""
        act_pack, weight_pack, config, tensors = spec.make(256, max_tokens=256)
        layer = MoELayer(config)
        ref = spec.reference(act_pack, tensors)
        checked = 0
        for backend_key in spec.backend_keys:
            runner = next(
                (r for r in layer.runners if r.backend_key == backend_key), None
            )
            if runner is None:
                # Backend not instantiable on this device/stack (e.g. the
                # installed CuTe DSL cannot target this arch) — MoELayer
                # already dropped it from the candidate list.
                continue
            out = runner.forward(runner.pack_inputs(act_pack, weight_pack), tactic=-1)
            spec.check(out, ref, backend_key)
            checked += 1
        if checked == 0:
            pytest.skip(
                f"none of {spec.backend_keys} is available on this device/stack"
            )

    def test_runner_with_local_expert_offset(self, spec):
        """Nonzero local shard offset through the real kernel: global ids in
        the pack + separately-passed offset must produce the local-shard MoE
        output."""
        if not spec.supports_runtime_offset:
            pytest.skip(f"{spec.id} pack builder has no local_expert_offset support")
        offset = 16
        act_pack, weight_pack, config, tensors = spec.make(
            256, max_tokens=256, local_num_experts=16, local_expert_offset=offset
        )
        layer = MoELayer(config)
        runner = next(r for r in layer.runners if r.backend_key == spec.backend_keys[0])
        inputs = runner.pack_inputs(act_pack, weight_pack)
        # The output buffer is new_empty(); zero it so the all-zero check below
        # reads what the kernel wrote, not uninitialized memory.
        inputs[0].zero_()
        out = runner.forward(inputs, tactic=-1)
        assert out.float().abs().max().item() > 0, (
            "all-zero output — the kernel treated every routed expert as "
            "non-local (offset handling broken)"
        )
        spec.check(
            out,
            spec.reference(act_pack, tensors, expert_offset=offset),
            f"{spec.id} offset={offset}",
        )

    def test_graph_capture_replay(self, spec):
        """CUDA-graph-captured replay matches eager output."""
        num_tokens = 256
        act_pack, weight_pack, config, _ = spec.make(num_tokens, max_tokens=num_tokens)

        # Warm up: populate autotune cache + stabilize allocator
        with autotune(True):
            layer = MoELayer(config)
            for _ in range(3):
                _ = layer(act_pack, weight_pack)
        for _ in range(3):
            _ = layer(act_pack, weight_pack)

        eager_out = layer(act_pack, weight_pack).clone()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            captured = layer(act_pack, weight_pack)

        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()

        spec.check(captured, eager_out.float(), f"{spec.id} graph replay")


def _fp4_dummy_hidden(num_tokens, hidden_size, device):
    return (
        torch.zeros(num_tokens, hidden_size // 2, dtype=torch.uint8, device=device),
        torch.zeros(
            num_tokens, hidden_size // 16, dtype=torch.uint8, device=device
        ).view(torch.float8_e4m3fn),
    )


def _fp4_dummy_weight_view(num_experts, hidden_size, intermediate_size, device):
    """Shape-valid FP4 placeholders for packing-only tests (never launched)."""
    return {
        "gemm1_weights": torch.empty(
            num_experts,
            2 * intermediate_size,
            hidden_size // 2,
            dtype=torch.uint8,
            device=device,
        ),
        "gemm1_weights_scale": torch.empty(
            num_experts,
            2 * intermediate_size,
            hidden_size // 16,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        "gemm1_alpha": torch.empty(num_experts, dtype=torch.float32, device=device),
        "gemm2_weights": torch.empty(
            num_experts,
            hidden_size,
            intermediate_size // 2,
            dtype=torch.uint8,
            device=device,
        ),
        "gemm2_weights_scale": torch.empty(
            num_experts,
            hidden_size,
            intermediate_size // 16,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
    }


def _bf16_dummy_hidden(num_tokens, hidden_size, device):
    return (
        torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device=device),
        None,
    )


@dataclasses.dataclass(frozen=True)
class PackingSpec:
    """Per-runner inputs for the packed-topk-id contract test."""

    id: str
    runner_cls: type
    variant: QuantVariant
    view_keys: tuple  # weight-view keys the runner's pack_inputs requires
    make_hidden: Callable  # (num_tokens, hidden_size, device) -> (q, scale)


_PACKING_SPECS = (
    PackingSpec(
        id="fp4",
        runner_cls=TrtllmFp4RoutedRunner,
        variant=QuantVariant.NVFP4,
        view_keys=(
            "gemm1_weights",
            "gemm1_weights_scale",
            "gemm1_alpha",
            "gemm2_weights",
            "gemm2_weights_scale",
        ),
        make_hidden=_fp4_dummy_hidden,
    ),
    PackingSpec(
        id="bf16",
        runner_cls=TrtllmBf16RoutedRunner,
        variant=QuantVariant.BF16,
        view_keys=("gemm1_weights", "gemm2_weights"),
        make_hidden=_bf16_dummy_hidden,
    ),
)


@sm100_required
@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda t: t.to(torch.float16), "must be float32"),
        (lambda t: t[:1], "shape"),
        (lambda t: t.cpu(), "expected"),
    ),
    ids=("dtype", "shape", "device"),
)
def test_bf16_runner_rejects_malformed_activation_override(mutation, match):
    """Verify BF16 ``pack_inputs`` validates activation override metadata.

    Presence checks alone let an invalid dtype, shape, or device reach the FFI
    boundary. Testing through the runner also pins the helper wiring and the
    expert count used for validation.
    """
    act_pack, weight_pack, config, _ = _make_bf16_packs_and_config(
        8, hidden_size=128, intermediate_size=256, num_experts=4, top_k=2
    )
    view = weight_pack.get_view("trtllm_bf16_routed")
    view["gemm1_alpha"] = mutation(
        torch.ones(4, dtype=torch.float32, device=act_pack.hidden_states_q.device)
    )

    layer = MoELayer(config)
    runner = layer.runners[0]
    with pytest.raises((ValueError, TypeError), match=match):
        runner.pack_inputs(act_pack, weight_pack)


@sm100_required
@pytest.mark.parametrize("spec", _PACKING_SPECS, ids=[s.id for s in _PACKING_SPECS])
class TestTrtllmRoutedPackingContract:
    """TRTLLM routed packing must keep GLOBAL expert ids.

    The packed int32 top-k id is ``(GLOBAL expert_id << 16) | bf16(weight)``;
    the kernel maps ids onto its local shard via the separately passed
    ``local_expert_offset``.  Pre-subtracting the offset yields ids the kernel
    treats as non-local and silently skips → zero output on offset>0 ranks
    (gh #3547).
    """

    @pytest.mark.parametrize("local_expert_offset", [0, 32, 96])
    def test_pack_inputs_keeps_global_ids(self, spec, local_expert_offset):
        device = torch.device("cuda", torch.cuda.current_device())
        num_experts = 128
        local_num_experts = 32
        top_k = 4
        num_tokens = 16
        hidden_size = 256

        config = MoEConfig(
            routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
            quant=QuantConfig(variant=spec.variant),
            experts=ExpertConfig(
                intermediate_size=512,
                local_expert_offset=local_expert_offset,
                local_num_experts=local_num_experts,
            ),
        )
        runner = _build_direct_runner(spec.runner_cls, config, device)

        # Global expert ids drawn from this rank's local shard.
        selected_experts = (
            torch.randint(0, local_num_experts, (num_tokens, top_k), device=device).to(
                torch.int32
            )
            + local_expert_offset
        )
        final_scales = torch.rand(num_tokens, top_k, device=device)
        # One negative weight: its bf16 sign bit makes the int16->int32 widen
        # sign-extend, so a dropped `& 0xFFFF` mask corrupts the id field and
        # fails the assertions below (all-positive scales mask that regression).
        final_scales[0, 0] = -final_scales[0, 0]
        hidden_q, hidden_scale = spec.make_hidden(num_tokens, hidden_size, device)
        act_pack = MoEActivationPack(
            hidden_states_q=hidden_q,
            hidden_states_scale=hidden_scale,
            topk_ids=selected_experts,
            topk_weights=final_scales,
        )

        # No kernel launches, but runners still validate their backend-native
        # weight contracts before exposing the packed routing buffers.
        if spec.variant is QuantVariant.NVFP4:
            weight_view = _fp4_dummy_weight_view(
                local_num_experts,
                hidden_size,
                config.experts.intermediate_size,
                device,
            )
        else:
            weight_view = {k: torch.empty(0, device=device) for k in spec.view_keys}
        weight_pack = MoEWeightPack()
        weight_pack.prepare_for(runner.backend_key, weight_view)

        from flashinfer.fused_moe.core import MoeRunnerInputs

        inputs = runner.pack_inputs(act_pack, weight_pack)
        topk_ids = MoeRunnerInputs.from_list(inputs).topk_ids

        # Upper 16 bits hold the GLOBAL expert id — NOT offset-shifted.
        decoded_ids = topk_ids >> 16
        assert torch.equal(decoded_ids, selected_experts), (
            f"{spec.id} offset={local_expert_offset}: packed ids {decoded_ids} != "
            f"global ids {selected_experts} — pre-subtracting the offset makes "
            f"the kernel skip these experts as non-local"
        )
        # Low 16 bits hold the bf16 gate-weight bits.
        expected_bits = (
            final_scales.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
        )
        assert torch.equal(topk_ids & 0xFFFF, expected_bits)
        # The offset travels to the kernel as a separate argument.
        assert runner._static_kwargs["local_expert_offset"] == local_expert_offset


@sm100_required
class TestTrtllmFp4UnpackedContract:
    @pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
    def test_pack_inputs_forwards_separate_routing_tensors(self, weights_dtype):
        from flashinfer.fused_moe.core import MoeRunnerInputs, RoutingInputMode

        device = torch.device("cuda", torch.cuda.current_device())
        num_tokens, hidden_size, top_k = 16, 256, 4
        config = MoEConfig(
            routing=RoutingConfig(num_experts=128, top_k=top_k),
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=ExpertConfig(
                intermediate_size=512,
                local_expert_offset=32,
                local_num_experts=32,
            ),
        )
        runner = _build_direct_runner(TrtllmFp4RoutedRunner, config, device)
        ids = torch.randint(
            32, 64, (num_tokens, top_k), dtype=torch.int32, device=device
        )
        weights = torch.rand(num_tokens, top_k, dtype=weights_dtype, device=device)
        act_pack = MoEActivationPack(
            hidden_states_q=torch.zeros(
                num_tokens, hidden_size // 2, dtype=torch.uint8, device=device
            ),
            hidden_states_scale=torch.zeros(
                num_tokens, hidden_size // 16, dtype=torch.uint8, device=device
            ).view(torch.float8_e4m3fn),
            topk_ids=ids,
            topk_weights=weights,
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        )
        weight_pack = MoEWeightPack()
        weight_pack.prepare_for(
            runner.backend_key,
            _fp4_dummy_weight_view(
                config.experts.local_num_experts,
                hidden_size,
                config.experts.intermediate_size,
                device,
            ),
        )

        moe_inputs = MoeRunnerInputs.from_list(
            runner.pack_inputs(act_pack, weight_pack)
        )
        assert moe_inputs.topk_ids is ids
        assert moe_inputs.expert_weights is weights
        assert (
            runner._static_kwargs["routing_input_mode"]
            == RoutingInputMode.UnpackedPrecomputed
        )
        assert runner._static_kwargs["local_expert_offset"] == 32

    @pytest.mark.parametrize(
        "activation",
        [
            pytest.param(SwiGLU(), id="swiglu"),
            pytest.param(ReLU2(), id="relu2"),
        ],
    )
    @pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
    def test_cuda_graph_replay_matches_eager(self, weights_dtype, activation):
        device = torch.device("cuda", torch.cuda.current_device())
        num_tokens, hidden_size, intermediate_size = 16, 1024, 512
        num_experts, top_k = 8, 2
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            gated=activation.is_gated,
            use_per_token_activation=True,
            use_nontrivial_alphas=False,
        )
        config = MoEConfig(
            routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
            quant=QuantConfig(
                variant=QuantVariant.NVFP4,
                per_token_scale=True,
            ),
            experts=ExpertConfig(
                intermediate_size=intermediate_size,
                local_num_experts=num_experts,
            ),
            activation=activation,
        )
        runner = _build_direct_runner(TrtllmFp4RoutedRunner, config, device)
        act_pack = MoEActivationPack(
            hidden_states_q=tensors["x"],
            hidden_states_scale=tensors["x_sf"].squeeze(-1),
            topk_ids=tensors["token_selected_experts"],
            topk_weights=tensors["token_final_scales"].to(weights_dtype),
            per_token_scale=tensors["x_per_token_scale"],
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        )
        weight_pack = MoEWeightPack()
        prepared_weights = TrtllmFp4Config.prepare_weights(
            tensors["w1_weight_bf16"],
            tensors["w2_weight_bf16"],
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
        )
        fc1_size = intermediate_size * (2 if activation.is_gated else 1)
        assert prepared_weights["gemm1_weights"].shape == (
            num_experts,
            fc1_size,
            hidden_size // 2,
        )
        assert prepared_weights["gemm1_weights_scale"].shape == (
            num_experts,
            fc1_size,
            hidden_size // 16,
        )
        weight_pack.prepare_for(
            runner.backend_key,
            prepared_weights,
        )
        inputs = runner.pack_inputs(act_pack, weight_pack)
        from flashinfer.fused_moe.core import MoeRunnerInputs

        moe_inputs = MoeRunnerInputs.from_list(inputs)
        assert moe_inputs.per_token_scale is act_pack.per_token_scale
        assert runner._static_kwargs["per_token_scale"] is act_pack.per_token_scale
        assert runner._inner.use_per_token_scaling is True
        for _ in range(3):
            runner.forward(inputs, tactic=-1)
        torch.cuda.synchronize()
        eager = runner.forward(inputs, tactic=-1).clone()

        ones = torch.ones(num_experts, device=device, dtype=torch.float32)
        reference = compute_reference_moe_fp4(
            hidden_states=tensors["x_ref"],
            gemm1_weights=tensors["w1_weight_bf16"],
            gemm2_weights=tensors["w2_weight_bf16"],
            gemm1_alpha=ones,
            gemm2_alpha=ones,
            token_selected_experts=act_pack.topk_ids,
            token_final_scales=act_pack.topk_weights,
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
            use_per_token_activation=True,
            activation_type=activation.type,
        )
        passed, pct, atol = check_accuracy(eager, reference)
        assert passed, (
            f"{activation.type.name}: only {pct * 100:.2f}% values within "
            f"tolerance (atol={atol:.4f})"
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = runner.forward(inputs, tactic=-1)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(captured, eager)


@sm100_required
class TestTrtllmEPOffset:
    """EP-shard forward regression (gh #3547): an offset>0 run over the same
    local-shard weights must reproduce the offset-0 baseline, not silently
    zero out.  The packed-id bit contract itself is covered per-variant by
    ``TestTrtllmRoutedPackingContract``.
    """

    @pytest.mark.parametrize("local_expert_offset", [32, 96])
    @pytest.mark.parametrize(
        "routing_input_mode",
        [
            RoutingInputMode.PackedPrecomputed,
            RoutingInputMode.UnpackedPrecomputed,
        ],
        ids=["packed", "unpacked"],
    )
    def test_ep_shard_forward_matches_offset_zero(
        self, local_expert_offset, routing_input_mode
    ):
        """Full EP-shard forward equals the identical offset-0 run.

        Same local-shard weights, same tokens, global ids shifted up by the
        shard offset — the output must match the offset-0 baseline.  Before
        the gh #3547 fix the EP run returned bit-exactly zero output.
        """
        device = torch.device("cuda", torch.cuda.current_device())
        num_experts = 128  # global expert count across all EP ranks
        local_num_experts = 32
        top_k = 4
        num_tokens = 64
        hidden_size = 512
        intermediate_size = 512

        # Sample routing within one shard so the same tensors serve both runs.
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=local_num_experts,
            num_local_experts=local_num_experts,
            top_k=top_k,
        )
        weight_pack = MoEWeightPack()
        weight_pack.prepare_for(
            "trtllm_fp4_routed",
            TrtllmFp4Config.prepare_weights(
                tensors["w1_weight_bf16"],
                tensors["w2_weight_bf16"],
                num_local_experts=local_num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                device=device,
            ),
        )

        def run(offset: int) -> torch.Tensor:
            config = MoEConfig(
                routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
                quant=QuantConfig(variant=QuantVariant.NVFP4),
                experts=ExpertConfig(
                    intermediate_size=intermediate_size,
                    local_expert_offset=offset,
                    local_num_experts=local_num_experts,
                ),
            )
            act_pack = MoEActivationPack(
                hidden_states_q=tensors["x"],
                hidden_states_scale=tensors["x_sf"].squeeze(-1),
                topk_ids=tensors["token_selected_experts"] + offset,
                topk_weights=(
                    tensors["token_final_scales"].to(torch.bfloat16)
                    if routing_input_mode == RoutingInputMode.UnpackedPrecomputed
                    else tensors["token_final_scales"]
                ),
                routing_input_mode=routing_input_mode,
            )
            runner = _build_direct_runner(TrtllmFp4RoutedRunner, config, device)
            inputs = runner.pack_inputs(act_pack, weight_pack)
            return runner.forward(inputs, tactic=-1).clone()

        baseline = run(0)
        ep_out = run(local_expert_offset)

        # gh #3547 symptom: the EP-shard output was bit-exactly zero.
        assert not bool((ep_out == 0).all()), (
            f"offset={local_expert_offset}: EP-shard output is all-zero (gh #3547)"
        )
        passed, pct, atol = check_accuracy(ep_out, baseline)
        assert passed, (
            f"offset={local_expert_offset}: EP-shard output diverges from the "
            f"offset-0 baseline ({pct * 100:.2f}% within tolerance, atol={atol:.4f})"
        )


# ---------------------------------------------------------------------------
# 4. FromLogits packing contract (gh #3595)
# ---------------------------------------------------------------------------


@sm100_required
class TestTrtllmFromLogitsPackingContract:
    """FromLogits buffer allocation must follow the kernel's output contract.

    TRTLLM routing kernels write bf16 expert weights regardless of the logits
    dtype; allocating ``expert_weights`` with ``routing_logits.dtype`` (fp32
    DeepSeekV3 logits) mislabels the kernel-filled buffer, so an unfinalized
    read interprets bf16 bits as fp32 garbage (gh #3595 — same bug previously
    fixed in the canonical ``trtllm_fp4_block_scale_moe`` wrapper).  Packing
    inspection only; no kernel launch.
    """

    @pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
    def test_expert_weights_buffer_is_bf16(self, logits_dtype):
        from flashinfer.fused_moe.core import MoeRunnerInputs, RoutingInputMode

        device = torch.device("cuda", torch.cuda.current_device())
        num_experts, top_k, num_tokens, hidden_size = 128, 4, 16, 256

        config = MoEConfig(
            routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=ExpertConfig(intermediate_size=512),
        )
        runner = _build_direct_runner(TrtllmFp4RoutedRunner, config, device)

        routing_logits = torch.randn(
            num_tokens, num_experts, dtype=logits_dtype, device=device
        )
        act_pack = MoEActivationPack(
            hidden_states_q=torch.zeros(
                num_tokens, hidden_size // 2, dtype=torch.uint8, device=device
            ),
            hidden_states_scale=torch.zeros(
                num_tokens, hidden_size // 16, dtype=torch.uint8, device=device
            ).view(torch.float8_e4m3fn),
            routing_input_mode=RoutingInputMode.FromLogits,
            routing_logits=routing_logits,
        )
        weight_pack = MoEWeightPack()
        weight_pack.prepare_for(
            "trtllm_fp4_routed",
            _fp4_dummy_weight_view(
                num_experts,
                hidden_size,
                config.experts.intermediate_size,
                device,
            ),
        )

        moe_inputs = MoeRunnerInputs.from_list(
            runner.pack_inputs(act_pack, weight_pack)
        )

        # Kernel-filled OUTPUT buffers: bf16 weights (gh #3595), int32 ids.
        assert moe_inputs.expert_weights.dtype == torch.bfloat16, (
            f"logits_dtype={logits_dtype}: expert_weights buffer is "
            f"{moe_inputs.expert_weights.dtype}, but the fp4 routing kernel "
            f"writes bf16 — an unfinalized read would mislabel the data"
        )
        assert moe_inputs.expert_weights.shape == (num_tokens, top_k)
        assert moe_inputs.topk_ids.dtype == torch.int32
        # Logits thread through unchanged; mode reaches the kernel kwargs.
        assert moe_inputs.routing_logits is routing_logits
        assert (
            runner._static_kwargs["routing_input_mode"] == RoutingInputMode.FromLogits
        )

    def _make_bf16_from_logits_inputs(self, logits_dtype):
        from flashinfer.fused_moe.core import MoeRunnerInputs

        act, weights, config, _ = _make_bf16_packs_and_config(
            16,
            hidden_size=256,
            intermediate_size=512,
            num_experts=8,
            top_k=2,
            max_tokens=16,
        )
        logits = torch.randn(
            16, 8, dtype=logits_dtype, device=act.hidden_states_q.device
        )
        logits_act = MoEActivationPack(
            hidden_states_q=act.hidden_states_q,
            hidden_states_scale=None,
            routing_input_mode=RoutingInputMode.FromLogits,
            routing_logits=logits,
        )
        runner = _build_direct_runner(TrtllmBf16RoutedRunner, config, logits.device)
        inputs = runner.pack_inputs(logits_act, weights)
        return runner, inputs, MoeRunnerInputs.from_list(inputs), logits

    @pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
    def test_bf16_expert_weights_buffer_is_bf16(self, logits_dtype):
        runner, _, moe_inputs, logits = self._make_bf16_from_logits_inputs(logits_dtype)
        assert moe_inputs.routing_logits is logits
        assert moe_inputs.topk_ids.dtype == torch.int32
        assert moe_inputs.expert_weights.dtype == torch.bfloat16
        assert (
            runner._static_kwargs["routing_input_mode"] == RoutingInputMode.FromLogits
        )

    @pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "routing_method",
        [RoutingMethodType.Default, RoutingMethodType.DeepSeekV3],
        ids=["default", "deepseek-v3"],
    )
    def test_bf16_from_logits_matches_host_reference(
        self, routing_method, logits_dtype
    ):
        from tests.moe.trtllm_gen_fused_moe_utils import noaux_tc_ref

        num_tokens, num_experts, top_k = 16, 8, 2
        intermediate_size = 512
        act, weights, config, tensors = _make_bf16_packs_and_config(
            num_tokens,
            hidden_size=256,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            max_tokens=num_tokens,
        )
        logits = torch.randn(
            num_tokens,
            num_experts,
            dtype=logits_dtype,
            device=act.hidden_states_q.device,
        )
        routing_bias = None
        if routing_method is RoutingMethodType.DeepSeekV3:
            routing_bias = torch.randn(
                num_experts,
                dtype=torch.bfloat16,
                device=logits.device,
            )
            dense_scores = noaux_tc_ref(
                logits.float(),
                routing_bias.float(),
                n_group=4,
                topk_group=2,
                top_k=top_k,
                routed_scaling_factor=1.0,
            )
            topk_weights, topk_ids = torch.topk(dense_scores, top_k, dim=-1)
            routing = RoutingConfig(
                num_experts=num_experts,
                top_k=top_k,
                method=routing_method,
                n_group=4,
                topk_group=2,
                routed_scaling_factor=1.0,
            )
        else:
            topk_weights, topk_ids = torch.topk(
                torch.softmax(logits.float(), dim=-1), top_k, dim=-1
            )
            routing = RoutingConfig(
                num_experts=num_experts,
                top_k=top_k,
                method=routing_method,
            )

        logits_act = MoEActivationPack(
            hidden_states_q=act.hidden_states_q,
            hidden_states_scale=None,
            routing_input_mode=RoutingInputMode.FromLogits,
            routing_logits=logits,
            routing_bias=routing_bias,
        )
        config = dataclasses.replace(config, routing=routing)
        output = MoELayer(config)(logits_act, weights)
        reference = _bf16_dense_reference(
            tensors["x"],
            tensors["w1"],
            tensors["w2"],
            topk_ids.to(torch.int32),
            topk_weights,
            intermediate_size,
        )
        torch.testing.assert_close(
            output.float(), reference, rtol=BF16_RTOL, atol=BF16_ATOL
        )

    @pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
    def test_bf16_cuda_graph_replay_matches_eager(self, logits_dtype):
        runner, inputs, _, _ = self._make_bf16_from_logits_inputs(logits_dtype)
        for _ in range(3):
            runner.forward(inputs, tactic=-1)
        torch.cuda.synchronize()
        eager = runner.forward(inputs, tactic=-1).clone()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = runner.forward(inputs, tactic=-1)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(captured, eager)

    def test_bf16_from_logits_with_local_expert_offset(self):
        act, weights, config, tensors = _make_bf16_packs_and_config(
            16,
            hidden_size=256,
            intermediate_size=512,
            num_experts=8,
            top_k=2,
            local_num_experts=4,
            local_expert_offset=4,
            max_tokens=16,
        )
        logits = torch.full(
            (16, 8), -4.0, dtype=torch.float32, device=act.hidden_states_q.device
        )
        logits[:8, :4] = torch.randn_like(logits[:8, :4]) + 4.0
        logits[8:, 4:] = torch.randn_like(logits[8:, 4:]) + 4.0
        topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), 2, dim=-1)
        logits_act = MoEActivationPack(
            hidden_states_q=act.hidden_states_q,
            hidden_states_scale=None,
            routing_input_mode=RoutingInputMode.FromLogits,
            routing_logits=logits,
        )
        layer = MoELayer(config, device=act.hidden_states_q.device)
        output = layer(logits_act, weights)
        reference = _bf16_dense_reference(
            tensors["x"],
            tensors["w1"],
            tensors["w2"],
            topk_ids.to(torch.int32),
            topk_weights,
            512,
            expert_offset=4,
        )
        assert torch.count_nonzero(reference)
        torch.testing.assert_close(
            output.float(), reference, rtol=BF16_RTOL, atol=BF16_ATOL
        )


# 6. prepare_trtllm_bf16_weights input contract
# ---------------------------------------------------------------------------
# Validation fires before any CUDA work, so the negative tests are CPU-only.


class TestPrepareTrtllmBf16Weights:
    _E, _I, _H = 2, 64, 128

    def _weights(self, dtype=torch.bfloat16):
        E, I, H = self._E, self._I, self._H
        return (
            torch.randn(E, 2 * I, H).to(dtype),
            torch.randn(E, H, I).to(dtype),
        )

    def _prepare(self, w1, w2, **overrides):
        kwargs = dict(
            num_local_experts=self._E,
            hidden_size=self._H,
            intermediate_size=self._I,
        )
        kwargs.update(overrides)
        return TrtllmBf16Config.prepare_weights(w1, w2, **kwargs)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_rejects_non_bf16_dtype(self, dtype):
        w1, w2 = self._weights(dtype)
        with pytest.raises(ValueError, match="bf16"):
            self._prepare(w1, w2)

    def test_rejects_wrong_shape(self):
        w1, w2 = self._weights()
        with pytest.raises(ValueError, match="shape"):
            self._prepare(w1[:, : self._I], w2)  # missing the gate half of gemm1

    @sm100_required
    def test_normalizes_noncontiguous_and_cpu_inputs(self):
        """Non-contiguous and CPU-resident inputs yield the same views as the
        contiguous on-device call (the .to(device).contiguous() normalization)."""
        w1, w2 = self._weights()
        w1, w2 = w1.cuda(), w2.cuda()
        base = self._prepare(w1, w2)

        # Same values, non-contiguous layout.
        w1_nc = w1.transpose(1, 2).contiguous().transpose(1, 2)
        assert not w1_nc.is_contiguous()
        nc = self._prepare(w1_nc, w2)

        # CPU-resident inputs with an explicit device target.
        cpu = self._prepare(w1.cpu(), w2.cpu(), device=torch.device("cuda"))

        for view in (nc, cpu):
            for key in ("gemm1_weights", "gemm2_weights"):
                assert torch.equal(view[key], base[key])


# ---------------------------------------------------------------------------
# Fused shared experts — configuration boundaries and backend gating
# ---------------------------------------------------------------------------


class TestFusedSharedExpertsConfig:
    """Geometry rejections live on the config, not in ``check_support()``.

    ``MoELayer.__init__`` swallows ``check_support()`` exceptions to filter
    unusable backends, so a geometry error raised there would surface as
    "no backend available" rather than naming the offending value.
    """

    def _cfg(self, *, num_shared, **overrides):
        routing = dict(
            num_experts=64,
            top_k=8,
            method=RoutingMethodType.DeepSeekV3,
            n_group=8,
            topk_group=4,
            routed_scaling_factor=2.5,
        )
        experts = dict(intermediate_size=512, num_fused_shared_experts=num_shared)
        routing.update(overrides.pop("routing", {}))
        experts.update(overrides.pop("experts", {}))
        return MoEConfig(
            routing=RoutingConfig(**routing),
            quant=QuantConfig(variant=QuantVariant.DeepSeekFp8),
            experts=ExpertConfig(**experts),
        )

    def test_zero_is_the_default_and_stays_out_of_repr(self):
        cfg = ExpertConfig(intermediate_size=512)
        assert cfg.num_fused_shared_experts == 0
        assert "num_fused_shared_experts" not in repr(cfg)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            ExpertConfig(intermediate_size=512, num_fused_shared_experts=-1)

    def test_non_deepseek_routing_rejected(self):
        # Only the DeepSeek routing kernel emits the appended shared slots.
        with pytest.raises(ValueError, match="requires DeepSeekV3 routing"):
            self._cfg(num_shared=1, routing=dict(method=RoutingMethodType.Default))

    def test_top_k_plus_shared_bounded(self):
        # MaxSupportedTopExperts == 32 applies to the *fused* total.
        self._cfg(num_shared=24)  # 8 + 24 == 32, allowed
        with pytest.raises(ValueError, match=r"top_k \+ num_fused_shared_experts"):
            self._cfg(num_shared=25)

    def test_num_experts_plus_shared_bounded(self):
        # NumNemotronExperts == 512, likewise on the fused total.
        with pytest.raises(
            ValueError, match=r"num_experts \+ num_fused_shared_experts"
        ):
            self._cfg(num_shared=1, routing=dict(num_experts=512))

    @pytest.mark.parametrize(
        "experts_override",
        [
            dict(local_expert_offset=8),
            dict(local_num_experts=32),
        ],
    )
    def test_expert_parallelism_rejected(self, experts_override):
        # The routing kernel maps a shared id to a weight row as
        # (global_id - local_expert_offset), which only lands on the intended
        # slot when this rank holds the whole routed set.
        with pytest.raises(ValueError, match="does not support expert"):
            self._cfg(num_shared=1, experts=experts_override)

    def test_full_local_set_accepted(self):
        cfg = self._cfg(num_shared=1, experts=dict(local_num_experts=64))
        assert cfg.experts.num_fused_shared_experts == 1


class TestFusedSharedExpertsBackendGating:
    """Backends must opt in before ``check_support()`` accepts S > 0."""

    def _shared_cfg(self, variant):
        return MoEConfig(
            routing=RoutingConfig(
                num_experts=64,
                top_k=2,
                method=RoutingMethodType.DeepSeekV3,
                n_group=8,
                topk_group=4,
                routed_scaling_factor=2.5,
            ),
            quant=QuantConfig(variant=variant),
            experts=ExpertConfig(intermediate_size=512, num_fused_shared_experts=2),
        )

    def test_registry_is_split_into_supporting_and_not(self):
        from flashinfer.fused_moe.layer import _BACKEND_RUNNERS

        supporting = {
            r.__name__
            for r in _BACKEND_RUNNERS.values()
            if r.supports_fused_shared_experts
        }
        assert supporting == {
            "TrtllmFp4RoutedRunner",
            "TrtllmFp8BlockRunner",
        }, (
            "backends claiming fused shared-expert support changed; confirm the "
            f"new backend forwards num_fused_shared_experts: {supporting}"
        )

    def test_every_non_supporting_runner_rejects_at_check_support(self):
        from flashinfer.fused_moe.layer import _BACKEND_RUNNERS

        checked = 0
        for runner_cls in set(_BACKEND_RUNNERS.values()):
            if runner_cls.supports_fused_shared_experts:
                continue
            variant = runner_cls.supported_quant_variants[0]
            runner = runner_cls.__new__(runner_cls)
            runner.config = self._shared_cfg(variant)
            with pytest.raises(
                NotImplementedError, match="does not support fused shared experts"
            ):
                runner.check_support()
            checked += 1
        assert checked, "no non-supporting runners exercised"


# ---------------------------------------------------------------------------
# Autotune cache keying
# ---------------------------------------------------------------------------
#
# In-memory tuning keys include runner_hash, nearest_profile, and extras.
# nearest_profile sees only profiled activation/routing tensors, so weight and
# config geometry must come from _cache_key_extras(). Persisted file_key drops
# runner_hash, making those same extras the config discriminator on disk. Every
# tactic-relevant field must therefore appear in the shared extras tuple.

# FP8 and FP4 exercise cross-variant keys; MxInt4 covers the single-variant case.
_RUNNERS = [
    pytest.param(
        TrtllmFp8BlockRunner,
        TrtllmFp8BlockConfig(),
        QuantVariant.DeepSeekFp8,
        QuantVariant.MxFp8,
        id="fp8_block",
    ),
    pytest.param(
        TrtllmFp4RoutedRunner,
        TrtllmFp4Config(),
        QuantVariant.NVFP4,
        QuantVariant.MXFP4,
        id="fp4",
    ),
    pytest.param(
        TrtllmMxInt4RoutedRunner,
        TrtllmMxInt4Config(),
        QuantVariant.MxInt4,
        None,
        id="mxint4",
    ),
]


def _cache_key_config(backend_cfg, variant, **overrides):
    fields = dict(
        num_experts=64,
        top_k=2,
        intermediate_size=256,
        local_expert_offset=0,
        local_num_experts=None,
        activation=SwiGLU(),
    )
    fields.update(overrides)
    return MoEConfig(
        routing=RoutingConfig(
            num_experts=fields["num_experts"],
            top_k=fields["top_k"],
            method=RoutingMethodType.DeepSeekV3,
            n_group=8,
            topk_group=4,
            routed_scaling_factor=2.5,
        ),
        quant=QuantConfig(variant=variant),
        experts=ExpertConfig(
            intermediate_size=fields["intermediate_size"],
            local_expert_offset=fields["local_expert_offset"],
            local_num_experts=fields["local_num_experts"],
        ),
        activation=fields["activation"],
        backend=BackendOptions(candidates=(backend_cfg,)),
    )


def _cache_key_runner(runner_cls, config):
    """Build a config-only runner without loading architecture-specific modules."""
    runner = runner_cls.__new__(runner_cls)
    runner.config = config
    return runner


# Dimensions that change which tactics are legal but are NOT recoverable from
# the profiled tensor shapes. Each must move both cache keys.
_DIMENSIONS = [
    ("intermediate_size", dict(intermediate_size=512)),
    ("top_k", dict(top_k=4)),
    ("num_experts", dict(num_experts=128)),
    ("local_num_experts", dict(local_num_experts=32)),
    ("local_expert_offset", dict(local_expert_offset=8)),
    ("activation", dict(activation=GeGLU())),
]


@pytest.mark.parametrize("runner_cls,backend_cfg,variant,alt_variant", _RUNNERS)
@pytest.mark.parametrize(
    "dimension,override", _DIMENSIONS, ids=[d[0] for d in _DIMENSIONS]
)
def test_tactic_dimension_changes_both_cache_keys(
    runner_cls, backend_cfg, variant, alt_variant, dimension, override
):
    """A tactic-relevant dimension must separate memory *and* file cache keys."""
    base = _cache_key_runner(runner_cls, _cache_key_config(backend_cfg, variant))
    other = _cache_key_runner(
        runner_cls, _cache_key_config(backend_cfg, variant, **override)
    )

    assert hash(base) != hash(other), (
        f"{runner_cls.__name__}: {dimension} does not change runner_hash, so two "
        "configurations share one in-memory tuned tactic."
    )
    # Compare serialized extras because the persisted key uses their string form.
    assert str(base.get_cache_key_extras([])) != str(other.get_cache_key_extras([])), (
        f"{runner_cls.__name__}: {dimension} does not change get_cache_key_extras(), "
        "so the on-disk cache (which drops runner_hash) collides."
    )


@pytest.mark.parametrize("runner_cls,backend_cfg,variant,alt_variant", _RUNNERS)
def test_quant_variant_changes_both_cache_keys(
    runner_cls, backend_cfg, variant, alt_variant
):
    if alt_variant is None:
        pytest.skip(f"{runner_cls.__name__} supports a single quant variant")
    base = _cache_key_runner(runner_cls, _cache_key_config(backend_cfg, variant))
    other = _cache_key_runner(runner_cls, _cache_key_config(backend_cfg, alt_variant))
    assert hash(base) != hash(other)
    assert str(base.get_cache_key_extras([])) != str(other.get_cache_key_extras([]))


@pytest.mark.parametrize("runner_cls,backend_cfg,variant,alt_variant", _RUNNERS)
def test_identical_config_shares_cache_key(
    runner_cls, backend_cfg, variant, alt_variant
):
    """Two runners built from the same configuration must still share a key.

    Guards the opposite failure: over-keying (e.g. folding in object identity)
    would make every layer re-tune from scratch.
    """
    a = _cache_key_runner(runner_cls, _cache_key_config(backend_cfg, variant))
    b = _cache_key_runner(runner_cls, _cache_key_config(backend_cfg, variant))
    assert hash(a) == hash(b)
    assert str(a.get_cache_key_extras([])) == str(b.get_cache_key_extras([]))


@pytest.mark.parametrize("runner_cls,backend_cfg,variant,alt_variant", _RUNNERS)
def test_cache_key_extras_are_str_stable(runner_cls, backend_cfg, variant, alt_variant):
    """Every element must survive the str() round trip the file cache performs.

    ``ProfilingCacheKey.file_key`` is ``str((custom_op, class, profile, extras))``
    and is written to disk, so an element whose repr embeds an address (or
    otherwise varies per process) would never match on reload.
    """
    runner = _cache_key_runner(runner_cls, _cache_key_config(backend_cfg, variant))
    extras = runner.get_cache_key_extras([])
    assert isinstance(extras, tuple)
    rendered = str(extras)
    assert "0x" not in rendered, (
        f"{runner_cls.__name__}: cache-key extras contain what looks like an "
        f"object address and will not match across processes: {rendered}"
    )
    for element in extras:
        # None is permitted: str(None) == "None" is stable across processes,
        # and optional config fields (n_group, per_token_scale, ...) are keyed.
        assert isinstance(element, (int, float, str, bool, tuple, type(None))), (
            f"{runner_cls.__name__}: cache-key element {element!r} of type "
            f"{type(element).__name__} has no guaranteed stable repr."
        )


def test_every_registered_runner_defines_stable_extras():
    """Every runner must use the unified in-memory and persisted-key entrypoints."""
    assert _BACKEND_RUNNERS, "no unified runners registered"
    for runner_cls in set(_BACKEND_RUNNERS.values()):
        assert runner_cls.__hash__ is MoERunner.__hash__
        assert runner_cls.get_cache_key_extras is MoERunner.get_cache_key_extras


def test_cute_dsl_cache_key_extends_unified_fields():
    class Inner:
        use_fused_finalize = False
        enable_pdl = True

    runner = CuteDslRunner.__new__(CuteDslRunner)
    runner.config = _cache_key_config(CuteDslConfig(), QuantVariant.NVFP4)
    runner._inner = Inner()

    shared = MoERunner._cache_key_extras(runner)
    assert runner.get_cache_key_extras([]) == shared + (False, True)


@pytest.mark.parametrize(
    "runner_cls,backend_cfg,variant,first,second",
    (
        (
            CutlassBf16Runner,
            CutlassBf16Config(),
            QuantVariant.BF16,
            SwiGLU(),
            SwiGLU(alpha=1.7, beta=1.0, limit=7.0),
        ),
        (
            CutlassBf16Runner,
            CutlassBf16Config(),
            QuantVariant.BF16,
            SwiGLUStep(limit=7.0),
            SwiGLUStep(limit=6.0),
        ),
        (
            CuteDslRunner,
            CuteDslConfig(),
            QuantVariant.NVFP4,
            SwiGLU(),
            SwiGLU(alpha=1.7, beta=1.0, limit=7.0),
        ),
        (
            CuteDslRunner,
            CuteDslConfig(),
            QuantVariant.NVFP4,
            SiTU(gate_scale=1.0, linear_scale=1.0),
            SiTU(gate_scale=2.0, linear_scale=3.0),
        ),
    ),
)
def test_scalar_activation_values_separate_cache_identity(
    runner_cls, backend_cfg, variant, first, second
):
    first_runner = _cache_key_runner(
        runner_cls,
        _cache_key_config(backend_cfg, variant, activation=first),
    )
    second_runner = _cache_key_runner(
        runner_cls,
        _cache_key_config(backend_cfg, variant, activation=second),
    )
    if issubclass(runner_cls, CutlassBf16Runner):
        first_runner._device_arch = second_runner._device_arch = 100
        first_runner._enable_pdl = second_runner._enable_pdl = False
    elif runner_cls is CuteDslRunner:

        class Inner:
            use_fused_finalize = True
            enable_pdl = False

        first_runner._inner = second_runner._inner = Inner()
    assert hash(first_runner) != hash(second_runner)
    assert first_runner.get_cache_key_extras([]) != second_runner.get_cache_key_extras(
        []
    )


def test_profiling_cache_key_file_key_separates_configs():
    """The on-disk key must separate configs that share a profile.

    Builds ``ProfilingCacheKey`` directly with an identical
    ``nearest_profile`` — the situation the adapters actually hit, since the
    weights are not in the profiled tensor list — and asserts ``file_key``
    still differs. ``file_key`` drops ``runner_hash``, so this is the check
    that the in-memory assertions above cannot make.
    """
    base = _cache_key_runner(
        TrtllmFp8BlockRunner,
        _cache_key_config(TrtllmFp8BlockConfig(), QuantVariant.DeepSeekFp8),
    )
    other = _cache_key_runner(
        TrtllmFp8BlockRunner,
        _cache_key_config(
            TrtllmFp8BlockConfig(), QuantVariant.DeepSeekFp8, intermediate_size=512
        ),
    )
    shared_profile = ((64, 256), (64, 2))

    def key_for(runner):
        return ProfilingCacheKey(
            custom_op="moe_trtllm_fp8_block",
            runner_class_name=type(runner).__name__,
            runner_hash=hash(runner),
            nearest_profile=shared_profile,
            extras=runner.get_cache_key_extras([]),
        )

    key_a, key_b = key_for(base), key_for(other)
    assert key_a != key_b
    assert key_a.file_key != key_b.file_key, (
        "identical profiles with different intermediate_size produced the same "
        "file_key; the persisted tactic would be reused across both."
    )


def test_fused_shared_experts_change_both_cache_keys():
    """S must separate cache entries.

    S widens the counts the tactic enumeration keys on (top_k + S,
    num_local_experts + S) but changes no profiled shape, so without it in the
    extras an S=0 layer and an S>0 layer in the same process would share one
    tuned tactic -- and the persisted cache would reuse it across runs.
    """
    base = _cache_key_runner(
        TrtllmFp8BlockRunner,
        _cache_key_config(TrtllmFp8BlockConfig(), QuantVariant.DeepSeekFp8),
    )
    keys = {0: (hash(base), str(base.get_cache_key_extras([])))}
    for num_shared in (1, 2):
        cfg = _cache_key_config(TrtllmFp8BlockConfig(), QuantVariant.DeepSeekFp8)
        cfg = dataclasses.replace(
            cfg,
            experts=dataclasses.replace(
                cfg.experts, num_fused_shared_experts=num_shared
            ),
        )
        runner = _cache_key_runner(TrtllmFp8BlockRunner, cfg)
        keys[num_shared] = (hash(runner), str(runner.get_cache_key_extras([])))

    assert len({h for h, _ in keys.values()}) == 3, (
        f"S values share a runner_hash: {keys}"
    )
    assert len({e for _, e in keys.values()}) == 3, (
        f"S values share cache-key extras: {keys}"
    )


# Routing shape and declared-but-unconsumed quant flags. Neither changes which
# tactics are *legal*, so they are not in _DIMENSIONS above; they are keyed
# because they change how a tactic ranks (routing shape alters the expert-token
# distribution) or will feed tactic enumeration once honored (quant flags).
_SOFT_DIMENSIONS = [
    ("routing_method", dict(method=RoutingMethodType.Renormalize)),
    ("n_group", dict(n_group=4)),
    ("topk_group", dict(topk_group=2)),
    ("per_token_scale", dict(per_token_scale=True)),
    ("swizzled_scale_factors", dict(swizzled_scale_factors=True)),
]


def _cache_key_config_with(backend_cfg, variant, **overrides):
    quant_kwargs = {
        k: overrides.pop(k)
        for k in ("per_token_scale", "swizzled_scale_factors")
        if k in overrides
    }
    routing_kwargs = {
        k: overrides.pop(k)
        for k in ("method", "n_group", "topk_group")
        if k in overrides
    }
    cfg = _cache_key_config(backend_cfg, variant, **overrides)
    if routing_kwargs:
        cfg = dataclasses.replace(
            cfg, routing=dataclasses.replace(cfg.routing, **routing_kwargs)
        )
    if quant_kwargs:
        cfg = dataclasses.replace(
            cfg, quant=dataclasses.replace(cfg.quant, **quant_kwargs)
        )
    return cfg


@pytest.mark.parametrize(
    "dimension,override", _SOFT_DIMENSIONS, ids=[d[0] for d in _SOFT_DIMENSIONS]
)
def test_ranking_relevant_dimension_changes_both_cache_keys(dimension, override):
    base = _cache_key_runner(
        TrtllmFp8BlockRunner,
        _cache_key_config(TrtllmFp8BlockConfig(), QuantVariant.DeepSeekFp8),
    )
    other = _cache_key_runner(
        TrtllmFp8BlockRunner,
        _cache_key_config_with(
            TrtllmFp8BlockConfig(), QuantVariant.DeepSeekFp8, **override
        ),
    )
    assert hash(base) != hash(other), f"{dimension} does not change runner_hash"
    assert str(base.get_cache_key_extras([])) != str(other.get_cache_key_extras([])), (
        f"{dimension} does not change get_cache_key_extras()"
    )
