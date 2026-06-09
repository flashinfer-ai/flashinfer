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

Two halves:

  * CPU-only config/dataclass tests (no GPU or JIT). These track the actual MVP
    API surface (single-knob ``QuantVariant``, explicit
    ``BackendOptions(candidates=(...))``); see
    ``docs/design_docs/flashinfer_moe_api.md`` §10 CR1.

  * SM100 (Blackwell) GPU tests for ``MoELayer`` + Packs: shared-bf16-reference
    accuracy, autotune candidate visitation, CUDA-graph replay, and the
    expert-parallel offset packing (CR3). Scope: NVFP4, pre-routed path.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    MoEActivationPack,
    MoELayer,
    MoEWeightPack,
    TrtllmFp4RoutedRunner,
)
from flashinfer.fused_moe.api import (
    ActivationConfig,
    ActivationType,
    BackendOptions,
    CuteDslConfig,
    CutlassConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEConfig,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    RoutingMethodType,
    TrtllmBf16Config,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
    TrtllmMxInt4Config,
)

# Reuse the canonical reference implementation + accuracy helpers from the
# existing CuteDSL test — keeps tolerance bounds consistent across tests.
from tests.moe.test_cute_dsl_fused_moe import (  # noqa: E402
    check_accuracy,
    compute_reference_moe_fp4,
    create_moe_tensors,
    is_sm100_family,
)


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


# ---------------------------------------------------------------------------
# ActivationType helpers
# ---------------------------------------------------------------------------


class TestActivation:
    def test_is_gated(self):
        assert ActivationType.Swiglu.is_gated
        assert ActivationType.Geglu.is_gated
        assert ActivationType.SwigluBias.is_gated
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
        for act in ActivationType:
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
                method=RoutingMethodType.DeepSeekV3,
                n_group=8,
                topk_group=4,
                routed_scaling_factor=1.0,
            ),
            quant=QuantConfig(variant=QuantVariant.MxFp8),
            experts=ExpertConfig(intermediate_size=2048, local_num_experts=32),
            activation=ActivationConfig(type=ActivationType.Geglu),
            backend=BackendOptions(
                candidates=(TrtllmFp8BlockConfig(), CutlassConfig())
            ),
            execution=ExecutionConfig(enable_pdl=True, tune_max_num_tokens=4096),
        )
        assert _eval_repr(cfg) == cfg


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
        assert ActivationConfig.swiglu == ActivationConfig(ActivationType.Swiglu)
        assert ActivationConfig.geglu == ActivationConfig(ActivationType.Geglu)
        assert ActivationConfig.relu2 == ActivationConfig(ActivationType.Relu2)
        assert ActivationConfig.identity == ActivationConfig(ActivationType.Identity)

    def test_singleton_is_gated(self):
        assert ActivationConfig.swiglu.is_gated
        assert not ActivationConfig.identity.is_gated


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
            activation=ActivationConfig(type=ActivationType.Swiglu),
            backend=BackendOptions(candidates=(TrtllmFp4Config(), CutlassConfig())),
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
            activation=ActivationConfig(type=ActivationType.Swiglu),
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
                method=RoutingMethodType.Renormalize,
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
            activation=ActivationConfig(type=ActivationType.Swiglu),
            backend=BackendOptions((CutlassConfig(),)),
        )
        # CUTLASS uses modular (pre-routed) dispatch — supplied at call time via
        # MoEActivationPack (selected_experts/final_scales), not via config
        assert any(isinstance(c, CutlassConfig) for c in cfg.backend)

    def test_cutedsl_nvfp4(self):
        """CuteDSL NVFP4 config."""
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=64, top_k=8),
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=ExpertConfig(intermediate_size=1024),
            activation=ActivationConfig(type=ActivationType.Swiglu),
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
            activation=ActivationConfig(type=ActivationType.Swiglu),
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
        [a for a in ActivationType if a is not ActivationType.Swiglu],
    )
    def test_non_swiglu_activation_rejected(self, act):
        from flashinfer.fused_moe import MoELayer

        cfg = self._nvfp4_swiglu(activation=ActivationConfig(type=act))
        with pytest.raises(NotImplementedError, match="Swiglu"):
            MoELayer(cfg)


sm100_required = pytest.mark.skipif(
    not is_sm100_family(),
    reason="Unified NVFP4 MoE requires SM100 family (Blackwell SM100/SM103)",
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
):
    """Build (act_pack, weight_pack, config, tensors_dict) for a given shape.

    ``tensors_dict`` contains the original bf16 reference weights used to
    compute ground truth via ``compute_reference_moe_fp4``.
    """
    local_num_experts = local_num_experts or num_experts
    max_tokens = max_tokens or max(num_tokens, 8192)
    device = torch.device("cuda", torch.cuda.current_device())

    # CuteDSL view comes pre-built by create_moe_tensors + bf16 refs
    tensors = create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=local_num_experts,
        top_k=top_k,
    )

    act_pack = MoEActivationPack(
        hidden_states_q=tensors["x"],
        hidden_states_scale=tensors["x_sf"].squeeze(-1),
        selected_experts=tensors["token_selected_experts"],
        final_scales=tensors["token_final_scales"],
    )

    weight_pack = MoEWeightPack()
    weight_pack.prepare_for(
        "cute_dsl_nvfp4",
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
            num_local_experts=local_num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=device,
        ),
    )

    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.NVFP4),
        experts=ExpertConfig(
            intermediate_size=intermediate_size,
            local_num_experts=local_num_experts,
        ),
        activation=ActivationConfig(),
        backend=BackendOptions(candidates=(CuteDslConfig(), TrtllmFp4Config())),
        execution=ExecutionConfig(tune_max_num_tokens=max_tokens),
    )
    return act_pack, weight_pack, config, tensors


# ---------------------------------------------------------------------------
# 1. Accuracy
# ---------------------------------------------------------------------------


def _compute_ref(act_pack, tensors, shape):
    """bf16 ground-truth MoE output for the given pack + shape."""
    return compute_reference_moe_fp4(
        hidden_states=tensors["x_bf16"].float().cuda(),
        gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
        gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
        token_selected_experts=act_pack.selected_experts,
        token_final_scales=act_pack.final_scales,
        num_tokens=act_pack.num_tokens,
        num_experts=shape["num_experts"],
        top_k=shape["top_k"],
        hidden_size=shape["hidden_size"],
        intermediate_size=shape["intermediate_size"],
        fc2_input_scale=tensors["fc2_input_scale"],
    )


@sm100_required
class TestUnifiedMoEAccuracy:
    """Every path compares against the same bf16 reference.

    Catches cases where both backends are wrong in the same way, which a
    cross-backend agreement test would miss.
    """

    @pytest.mark.parametrize("num_tokens", [128, 512])
    def test_layer_output_matches_reference(self, num_tokens):
        """MoELayer end-to-end output matches bf16 reference."""
        act_pack, weight_pack, config, tensors = _make_packs_and_config(
            num_tokens, **SMALL
        )

        with autotune(True):
            layer = MoELayer(config)
            out = layer(act_pack, weight_pack)

        ref = _compute_ref(act_pack, tensors, SMALL)
        passed, pct, atol = check_accuracy(out, ref)
        assert passed, (
            f"MoELayer output {pct * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}) vs bf16 reference at num_tokens={num_tokens}"
        )

    @pytest.mark.parametrize("backend_key", ["cute_dsl_nvfp4", "trtllm_fp4_routed"])
    def test_each_backend_matches_reference(self, backend_key):
        """Each candidate backend individually matches the same bf16 reference.

        If either backend's weight view were semantically wrong, its output
        would diverge from the shared reference — even in cases where it
        might agree with the other backend.
        """
        act_pack, weight_pack, config, tensors = _make_packs_and_config(256, **SMALL)
        layer = MoELayer(config)
        runner = next(r for r in layer.runners if r.backend_key == backend_key)

        inputs = runner.pack_inputs(act_pack, weight_pack)
        out = runner.forward(inputs, tactic=-1)

        ref = _compute_ref(act_pack, tensors, SMALL)
        passed, pct, atol = check_accuracy(out, ref)
        assert passed, (
            f"{backend_key}: {pct * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}) vs bf16 reference"
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

    def test_graph_capture_replay(self):
        """CUDA-graph-captured replay matches eager output."""
        num_tokens = 256
        act_pack, weight_pack, config, _ = _make_packs_and_config(
            num_tokens, max_tokens=num_tokens, **SMALL
        )

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

        passed, pct, atol = check_accuracy(captured, eager_out)
        assert passed, (
            f"Graph replay diverged from eager: {pct * 100:.2f}% within "
            f"tolerance (atol={atol:.4f})"
        )


# ---------------------------------------------------------------------------
# 3. Expert-parallel offset (CR3)
# ---------------------------------------------------------------------------


@sm100_required
class TestTrtllmEPOffset:
    """TRTLLM routed packing must shift global expert ids down by the local
    shard offset.

    The packed int32 top-k id is ``((expert_id - offset) << 16) | bf16(weight)``;
    the kernel indexes local experts as ``[0, local_num_experts)``.  Before the
    CR3 fix, ``pack_inputs`` defaulted the offset to 0, so a nonzero local shard
    produced out-of-range local expert ids.
    """

    @pytest.mark.parametrize("local_expert_offset", [0, 32, 96])
    def test_pack_inputs_applies_local_expert_offset(self, local_expert_offset):
        device = torch.device("cuda", torch.cuda.current_device())
        num_experts = 128
        local_num_experts = 32
        top_k = 4
        num_tokens = 16
        hidden_size = 256
        sf_vec_size = 16

        config = MoEConfig(
            routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=ExpertConfig(
                intermediate_size=512,
                local_expert_offset=local_expert_offset,
                local_num_experts=local_num_experts,
            ),
        )
        runner = TrtllmFp4RoutedRunner(config, device=device)

        # Global expert ids drawn from this rank's local shard.
        selected_experts = (
            torch.randint(0, local_num_experts, (num_tokens, top_k), device=device).to(
                torch.int32
            )
            + local_expert_offset
        )
        final_scales = torch.rand(num_tokens, top_k, device=device)
        act_pack = MoEActivationPack(
            hidden_states_q=torch.zeros(
                num_tokens, hidden_size // 2, dtype=torch.uint8, device=device
            ),
            hidden_states_scale=torch.zeros(
                num_tokens, hidden_size // sf_vec_size, dtype=torch.uint8, device=device
            ).view(torch.float8_e4m3fn),
            selected_experts=selected_experts,
            final_scales=final_scales,
        )

        # pack_inputs only passes weight tensors through; dummies suffice since
        # we inspect topk_ids only (no kernel launch).
        weight_pack = MoEWeightPack()
        weight_pack.prepare_for(
            "trtllm_fp4_routed",
            {
                "gemm1_weights": torch.empty(0, device=device),
                "gemm1_weights_scale": torch.empty(0, device=device),
                "gemm1_alpha": torch.empty(0, device=device),
                "gemm2_weights": torch.empty(0, device=device),
                "gemm2_weights_scale": torch.empty(0, device=device),
            },
        )

        from flashinfer.fused_moe.core import MoEInputs

        inputs = runner.pack_inputs(act_pack, weight_pack)
        topk_ids = MoEInputs.from_list(inputs).topk_ids

        # Upper 16 bits hold the (offset-shifted) local expert id.
        decoded_local_ids = topk_ids >> 16
        expected_local_ids = selected_experts - local_expert_offset
        assert torch.equal(decoded_local_ids, expected_local_ids), (
            f"offset={local_expert_offset}: packed local ids "
            f"{decoded_local_ids} != expected {expected_local_ids}"
        )
        # Local ids must land inside the kernel's [0, local_num_experts) range.
        assert int(decoded_local_ids.min()) >= 0
        assert int(decoded_local_ids.max()) < local_num_experts
