"""
Copyright (c) 2025 by FlashInfer team.

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

"""Unit tests for the unified MoE config and tensor dataclasses.

These tests are CPU-only — no GPU or JIT compilation required.

They track the *actual* MVP API surface (see
``docs/design_docs/flashinfer_moe_api.md`` §10 CR1):

  * Quantization is a single-knob ``QuantVariant`` enum (not the older
    ``QuantDtype`` + ``QuantGranularity`` + ``Fp8Variant`` triple).
  * Backend candidate sets are spelled explicitly as
    ``BackendOptions(candidates=(...))``.  The ``|`` pipe-operator sugar is a
    documented post-MVP non-goal and is intentionally not exercised here.
"""

import dataclasses

import pytest
import torch

from flashinfer.fused_moe.api import (
    Activation,
    ActivationConfig,
    BackendOptions,
    CuteDslConfig,
    CutlassConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEConfig,
    MoETensors,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    RoutingMethod,
    TrtllmBf16Config,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
    TrtllmMxInt4Config,
)


# ---------------------------------------------------------------------------
# Enum repr round-trip
# ---------------------------------------------------------------------------


class TestEnumRepr:
    @pytest.mark.parametrize("member", list(RoutingMethod))
    def test_routing_method_repr(self, member):
        assert eval(repr(member)) == member

    @pytest.mark.parametrize("member", list(Activation))
    def test_activation_repr(self, member):
        assert eval(repr(member)) == member

    @pytest.mark.parametrize("member", list(QuantVariant))
    def test_quant_variant_repr(self, member):
        assert eval(repr(member)) == member


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------


class TestActivation:
    def test_is_gated(self):
        assert Activation.Swiglu.is_gated
        assert Activation.Geglu.is_gated
        assert Activation.SwigluBias.is_gated
        assert not Activation.Identity.is_gated
        assert not Activation.Relu2.is_gated
        assert not Activation.Gelu.is_gated


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
            method=RoutingMethod.DeepSeekV3,
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
        for act in Activation:
            cfg = ActivationConfig(type=act)
            assert _eval_repr(cfg) == cfg

    def test_expert_config(self):
        cfg = ExpertConfig(
            intermediate_size=2048, local_expert_offset=4, local_num_experts=8
        )
        assert _eval_repr(cfg) == cfg

    def test_execution_config_default(self):
        cfg = ExecutionConfig()
        assert _eval_repr(cfg) == cfg

    def test_execution_config_custom(self):
        cfg = ExecutionConfig(
            do_finalize=False, enable_pdl=True, tune_max_num_tokens=1024
        )
        assert _eval_repr(cfg) == cfg

    def test_backend_options_multi(self):
        opts = BackendOptions(candidates=(TrtllmFp4Config(), CutlassConfig()))
        reconstructed = _eval_repr(opts)
        assert len(reconstructed) == 2
        assert isinstance(reconstructed.candidates[0], TrtllmFp4Config)
        assert isinstance(reconstructed.candidates[1], CutlassConfig)

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
                method=RoutingMethod.DeepSeekV3,
                n_group=8,
                topk_group=4,
                routed_scaling_factor=1.0,
            ),
            quant=QuantConfig(variant=QuantVariant.MxFp8),
            experts=ExpertConfig(intermediate_size=2048, local_num_experts=32),
            activation=ActivationConfig(type=Activation.Geglu),
            backend=BackendOptions(
                candidates=(TrtllmFp8BlockConfig(), CutlassConfig())
            ),
            execution=ExecutionConfig(enable_pdl=True, tune_max_num_tokens=4096),
        )
        assert _eval_repr(cfg) == cfg

    def test_moe_config_from_repr(self):
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=64, top_k=4),
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=ExpertConfig(intermediate_size=1024),
            backend=BackendOptions(candidates=(TrtllmFp4Config(), CuteDslConfig())),
        )
        reconstructed = MoEConfig.from_repr(repr(cfg))
        assert reconstructed == cfg


# ---------------------------------------------------------------------------
# BackendOptions
# ---------------------------------------------------------------------------


class TestBackendOptions:
    def test_explicit_candidates(self):
        opts = BackendOptions(candidates=(TrtllmFp4Config(), CutlassConfig()))
        assert isinstance(opts, BackendOptions)
        assert len(opts) == 2

    def test_multiple_candidates(self):
        opts = BackendOptions(
            candidates=(TrtllmFp4Config(), TrtllmFp8BlockConfig(), CutlassConfig())
        )
        assert len(opts) == 3

    def test_valid_for_filtering(self):
        opts = BackendOptions(
            candidates=(TrtllmBf16Config(), TrtllmFp8BlockConfig(), CutlassConfig())
        )
        # sm80: BF16 requires 100+, FP8Block requires 80+, Cutlass is universal
        valid = opts.valid_for(80)
        assert len(valid) == 2
        assert isinstance(valid[0], TrtllmFp8BlockConfig)
        assert isinstance(valid[1], CutlassConfig)

    def test_valid_for_blackwell(self):
        opts = BackendOptions(
            candidates=(TrtllmBf16Config(), TrtllmFp8BlockConfig(), CutlassConfig())
        )
        valid = opts.valid_for(100)
        assert len(valid) == 3

    def test_iteration(self):
        opts = BackendOptions(candidates=(TrtllmFp4Config(), CutlassConfig()))
        items = list(opts)
        assert len(items) == 2
        assert any(isinstance(c, TrtllmFp4Config) for c in items)
        assert any(isinstance(c, CutlassConfig) for c in items)

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
        narrow = dataclasses.replace(cfg, backend=BackendOptions((CutlassConfig(),)))
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
            backend=BackendOptions(candidates=(TrtllmBf16Config(), CutlassConfig())),
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
# ActivationConfig singletons
# ---------------------------------------------------------------------------


class TestActivationConfigSingletons:
    def test_singletons_exist(self):
        assert ActivationConfig.swiglu == ActivationConfig(Activation.Swiglu)
        assert ActivationConfig.geglu == ActivationConfig(Activation.Geglu)
        assert ActivationConfig.relu2 == ActivationConfig(Activation.Relu2)
        assert ActivationConfig.identity == ActivationConfig(Activation.Identity)

    def test_singleton_is_gated(self):
        assert ActivationConfig.swiglu.is_gated
        assert not ActivationConfig.identity.is_gated


# ---------------------------------------------------------------------------
# MoETensors validation
# ---------------------------------------------------------------------------


class TestMoETensors:
    def test_monolithic_path(self):
        t = MoETensors(
            hidden_states=torch.randn(4, 128),
            routing_logits=torch.randn(4, 8),
        )
        assert t.is_monolithic
        assert not t.is_modular
        t.validate()

    def test_modular_path(self):
        t = MoETensors(
            hidden_states=torch.randn(4, 128),
            token_selected_experts=torch.randint(0, 8, (4, 2)),
            token_final_scales=torch.randn(4, 2),
        )
        assert not t.is_monolithic
        assert t.is_modular
        t.validate()

    def test_both_paths_raises(self):
        t = MoETensors(
            hidden_states=torch.randn(4, 128),
            routing_logits=torch.randn(4, 8),
            token_selected_experts=torch.randint(0, 8, (4, 2)),
            token_final_scales=torch.randn(4, 2),
        )
        with pytest.raises(ValueError, match="Cannot provide both"):
            t.validate()

    def test_neither_path_raises(self):
        t = MoETensors(hidden_states=torch.randn(4, 128))
        with pytest.raises(ValueError, match="Must provide either"):
            t.validate()

    def test_modular_missing_scales_raises(self):
        t = MoETensors(
            hidden_states=torch.randn(4, 128),
            token_selected_experts=torch.randint(0, 8, (4, 2)),
        )
        with pytest.raises(ValueError, match="token_final_scales is required"):
            t.validate()

    def test_missing_hidden_states_raises(self):
        t = MoETensors(routing_logits=torch.randn(4, 8))
        with pytest.raises(ValueError, match="hidden_states is required"):
            t.validate()


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
                method=RoutingMethod.DeepSeekV3,
                n_group=8,
                topk_group=4,
                routed_scaling_factor=1.0,
            ),
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=ExpertConfig(intermediate_size=1024),
            activation=ActivationConfig(type=Activation.Swiglu),
            backend=BackendOptions(candidates=(TrtllmFp4Config(), CutlassConfig())),
        )
        assert cfg.routing.method == RoutingMethod.DeepSeekV3
        assert cfg.quant.variant == QuantVariant.NVFP4
        assert cfg.activation.is_gated

    def test_trtllm_fp8_block_mxfp8(self):
        """MxFP8 block-scale config."""
        cfg = MoEConfig(
            routing=RoutingConfig(
                num_experts=64,
                top_k=8,
                method=RoutingMethod.Renormalize,
            ),
            quant=QuantConfig(variant=QuantVariant.MxFp8),
            experts=ExpertConfig(intermediate_size=512),
            activation=ActivationConfig(type=Activation.Swiglu),
            backend=BackendOptions(
                candidates=(TrtllmFp8BlockConfig(), CutlassConfig())
            ),
        )
        assert cfg.quant.variant == QuantVariant.MxFp8

    def test_trtllm_fp8_per_tensor(self):
        """Per-tensor FP8 config."""
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=8, top_k=2),
            quant=QuantConfig(variant=QuantVariant.FP8PerTensor),
            experts=ExpertConfig(intermediate_size=512),
            backend=BackendOptions((TrtllmFp8PerTensorConfig(),)),
        )
        assert cfg.quant.variant == QuantVariant.FP8PerTensor

    def test_trtllm_bf16(self):
        """BF16 unquantized config."""
        cfg = MoEConfig(
            routing=RoutingConfig(
                num_experts=8,
                top_k=2,
                method=RoutingMethod.Renormalize,
            ),
            quant=QuantConfig(variant=QuantVariant.BF16),
            experts=ExpertConfig(intermediate_size=512),
            backend=BackendOptions(candidates=(TrtllmBf16Config(), CutlassConfig())),
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
        """CUTLASS modular (pre-routed) FP8 config."""
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=64, top_k=8),
            quant=QuantConfig(variant=QuantVariant.DeepSeekFp8),
            experts=ExpertConfig(intermediate_size=2048),
            activation=ActivationConfig(type=Activation.Swiglu),
            backend=BackendOptions((CutlassConfig(),)),
        )
        # CUTLASS uses modular dispatch — expressed via MoETensors, not config
        assert any(isinstance(c, CutlassConfig) for c in cfg.backend)

    def test_cutedsl_nvfp4(self):
        """CuteDSL NVFP4 config."""
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=64, top_k=8),
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=ExpertConfig(intermediate_size=1024),
            activation=ActivationConfig(type=Activation.Swiglu),
            backend=BackendOptions(candidates=(CuteDslConfig(), CutlassConfig())),
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
                method=RoutingMethod.Llama4,
            ),
            quant=QuantConfig(variant=QuantVariant.BF16),
            experts=ExpertConfig(intermediate_size=4096),
        )
        assert cfg.routing.method == RoutingMethod.Llama4
        assert cfg.routing.top_k == 1

    def test_qwen3_renormalize_naive(self):
        """Qwen3 RenormalizeNaive routing."""
        cfg = MoEConfig(
            routing=RoutingConfig(
                num_experts=64,
                top_k=8,
                method=RoutingMethod.RenormalizeNaive,
            ),
            quant=QuantConfig(variant=QuantVariant.DeepSeekFp8),
            experts=ExpertConfig(intermediate_size=1024),
        )
        assert cfg.routing.method == RoutingMethod.RenormalizeNaive


# ---------------------------------------------------------------------------
# MoELayer MVP fail-fast validation (CR6)
# ---------------------------------------------------------------------------
# These exercise MoELayer._validate_mvp_scope, which runs at construction time
# before any device/runner setup, so they need no GPU.


class TestMoELayerMVPValidation:
    def _nvfp4_swiglu(self, **overrides):
        base = dict(
            routing=RoutingConfig(num_experts=32, top_k=2),
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=ExpertConfig(intermediate_size=512),
            activation=ActivationConfig(type=Activation.Swiglu),
        )
        base.update(overrides)
        return MoEConfig(**base)

    @pytest.mark.parametrize(
        "variant",
        [v for v in QuantVariant if v is not QuantVariant.NVFP4],
    )
    def test_non_nvfp4_quant_rejected(self, variant):
        from flashinfer.fused_moe import MoELayer

        cfg = self._nvfp4_swiglu(quant=QuantConfig(variant=variant))
        with pytest.raises(NotImplementedError, match="NVFP4"):
            MoELayer(cfg)

    @pytest.mark.parametrize(
        "act",
        [a for a in Activation if a is not Activation.Swiglu],
    )
    def test_non_swiglu_activation_rejected(self, act):
        from flashinfer.fused_moe import MoELayer

        cfg = self._nvfp4_swiglu(activation=ActivationConfig(type=act))
        with pytest.raises(NotImplementedError, match="Swiglu"):
            MoELayer(cfg)
