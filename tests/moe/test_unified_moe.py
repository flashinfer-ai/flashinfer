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

  * SM100 (Blackwell) GPU tests for ``MoELayer`` + Packs, parametrized per
    ``QuantVariant`` via ``VariantSpec`` (currently NVFP4 + BF16, pre-routed
    path): accuracy vs an independent reference, direct-runner conformance,
    CUDA-graph replay, autotune candidate visitation, and the packed-topk-id
    contract (CR3). Adding a variant = registering one spec.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    MoEActivationPack,
    MoELayer,
    MoEWeightPack,
    TrtllmFp4RoutedRunner,
)
from flashinfer.fused_moe.runners import TrtllmBf16RoutedRunner
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
        # NVFP4 (CuteDSL/TRTLLM-FP4) and BF16 (TRTLLM-BF16, the EP grouped-GEMM
        # path) are both MVP-supported now; everything else is still rejected.
        [v for v in QuantVariant if v not in (QuantVariant.NVFP4, QuantVariant.BF16)],
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
# 1. NVFP4 reference helper
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


def _bf16_kernel_weight_views(w1: torch.Tensor, w2: torch.Tensor):
    """BlockMajorK weight views for the trtllm bf16 routed runner (per-expert).

    The authoritative recipe is ``BF16Moe.prepare_static_weights_for_kernel``
    (tests/moe/trtllm_gen_fused_moe_utils.py) — the only bf16 prep validated
    against independent dense math: gemm1 rows get the fused-gated-activation
    reorder chained with the ``epilogue_tile_m=128`` shuffle (the w3_w1
    permute), gemm2 gets the plain shuffle, then both convert to BlockMajorK
    (``block_k=128`` on the uint8 view).  Pure layout transform: the dense
    reference uses the unshuffled weights.  (Do NOT copy the moe_ep /
    routed-parity recipe of shuffle(64) without the gated reorder — those
    tests compare kernel-vs-same-kernel, so their layout was never validated
    against dense math and disagrees with the kernel's gate/linear pairing.)
    """
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    epilogue_tile_m = 128
    block_k = 128
    cache: dict = {}
    w1_views, w2_views = [], []
    for i in range(w1.shape[0]):
        p1 = _maybe_get_cached_w3_w1_permute_indices(
            cache, w1[i].view(torch.uint8), epilogue_tile_m
        )
        s1 = w1[i].view(torch.uint8)[p1.to(w1.device)].contiguous()
        p2 = get_w2_permute_indices_with_cache(
            cache, w2[i].view(torch.uint8), epilogue_tile_m
        )
        s2 = w2[i].view(torch.uint8)[p2.to(w2.device)].contiguous()
        w1_views.append(convert_to_block_layout(s1, block_k))
        w2_views.append(convert_to_block_layout(s2, block_k))
    return (
        torch.stack(w1_views).view(torch.bfloat16),
        torch.stack(w2_views).view(torch.bfloat16),
    )


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
    x32 = x.float()
    out = torch.zeros_like(x32)
    for local_e in range(w1.shape[0]):
        mask = selected_experts == local_e + expert_offset
        if not mask.any():
            continue
        tok, nth = torch.where(mask)
        a = x32[tok] @ w1[local_e].float().t()
        inter = F.silu(a[:, intermediate_size:]) * a[:, :intermediate_size]
        out[tok] += final_scales[tok, nth, None].float() * (
            inter @ w2[local_e].float().t()
        )
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
        selected_experts=selected_experts,
        final_scales=final_scales,
    )

    w1_view, w2_view = _bf16_kernel_weight_views(w1, w2)
    weight_pack = MoEWeightPack()
    weight_pack.prepare_for(
        "trtllm_bf16_routed",
        {
            "gemm1_weights": w1_view,
            "gemm2_weights": w2_view,
        },
    )

    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.BF16),
        experts=ExpertConfig(
            intermediate_size=intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
        ),
        activation=ActivationConfig(),
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
        act_pack.selected_experts,
        act_pack.final_scales,
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
        backend_keys=("cute_dsl_nvfp4", "trtllm_fp4_routed"),
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
        for backend_key in spec.backend_keys:
            runner = next(r for r in layer.runners if r.backend_key == backend_key)
            out = runner.forward(runner.pack_inputs(act_pack, weight_pack), tactic=-1)
            spec.check(out, ref, backend_key)

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
        runner = spec.runner_cls(config, device=device)

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
            selected_experts=selected_experts,
            final_scales=final_scales,
        )

        # pack_inputs only threads weights into static kwargs; dummies suffice
        # since we inspect topk_ids only (no kernel launch).
        weight_pack = MoEWeightPack()
        weight_pack.prepare_for(
            runner.backend_key,
            {k: torch.empty(0, device=device) for k in spec.view_keys},
        )

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
class TestTrtllmEPOffset:
    """EP-shard forward regression (gh #3547): an offset>0 run over the same
    local-shard weights must reproduce the offset-0 baseline, not silently
    zero out.  The packed-id bit contract itself is covered per-variant by
    ``TestTrtllmRoutedPackingContract``.
    """

    @pytest.mark.parametrize("local_expert_offset", [32, 96])
    def test_ep_shard_forward_matches_offset_zero(self, local_expert_offset):
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
                selected_experts=tensors["token_selected_experts"] + offset,
                final_scales=tensors["token_final_scales"],
            )
            runner = TrtllmFp4RoutedRunner(config, device=device)
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
