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
from flashinfer.fused_moe.runners import (
    CuteDslNvfp4Runner,
    MoERunner,
    TrtllmBf16RoutedRunner,
    TrtllmFp8BlockRunner,
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
        # TRTLLM-gen BF16 and block-FP8 require SM100+; Cutlass is universal.
        valid = opts.valid_for(80)
        assert len(valid) == 1
        assert isinstance(valid[0], CutlassConfig)

    def test_valid_for_blackwell(self):
        opts = BackendOptions(
            candidates=(TrtllmBf16Config(), TrtllmFp8BlockConfig(), CutlassConfig())
        )
        valid = opts.valid_for(100)
        assert len(valid) == 3
        assert not TrtllmFp8BlockConfig.supported(110)
        assert not TrtllmFp8BlockConfig.supported(120)

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
        # MoEActivationPack (topk_ids/topk_weights), not via config
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
# Unified runner support validation
# ---------------------------------------------------------------------------


class TestMoERunnerSupport:
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
        "runner_type,variant",
        (
            (CuteDslNvfp4Runner, QuantVariant.BF16),
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
        [a for a in ActivationType if a is not ActivationType.Swiglu],
    )
    @pytest.mark.parametrize(
        "runner_type,variant",
        (
            (CuteDslNvfp4Runner, QuantVariant.NVFP4),
            (TrtllmFp4RoutedRunner, QuantVariant.NVFP4),
            (TrtllmBf16RoutedRunner, QuantVariant.BF16),
            (TrtllmFp8BlockRunner, QuantVariant.DeepSeekFp8),
        ),
    )
    def test_non_swiglu_activation_not_supported(self, runner_type, variant, act):
        cfg = self._nvfp4_swiglu(
            quant=QuantConfig(variant=variant),
            activation=ActivationConfig(type=act),
        )
        runner = runner_type.__new__(runner_type)
        runner.config = cfg
        with pytest.raises(NotImplementedError, match="Swiglu"):
            runner.check_support()

    def test_fp8_block_unfinalized_not_supported(self):
        cfg = self._nvfp4_swiglu(
            quant=QuantConfig(variant=QuantVariant.DeepSeekFp8),
            execution=ExecutionConfig(do_finalize=False),
        )
        runner = TrtllmFp8BlockRunner.__new__(TrtllmFp8BlockRunner)
        runner.config = cfg
        with pytest.raises(NotImplementedError, match="do_finalize=True"):
            runner.check_support()

    def test_moe_runner_quant_support_check(self):
        class Runner(MoERunner):
            supported_quant_variants = (QuantVariant.NVFP4,)

            def get_valid_tactics(self, inputs, profile):
                return []

            def forward(self, inputs, **kwargs):
                return None

        runner = Runner()
        runner.config = self._nvfp4_swiglu()
        assert runner.check_support() is None


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

    def test_int64_topk_ids_rejected(self):
        # torch.topk returns int64; the launcher casts data_ptr without a
        # dtype ICHECK, so int64 reaching it is read as int32 bytes (silent
        # garbage routing) -- must fail loudly at construction.
        x, sf, ids, w, _ = _pack_tensors()
        with pytest.raises(TypeError, match="int32"):
            MoEActivationPack(x, sf, ids.long(), w)

    @pytest.mark.parametrize(
        "field_name", ["topk_ids", "topk_weights", "hidden_states_scale"]
    )
    def test_device_mismatch_rejected(self, field_name):
        # meta-device tensors give a second device without needing a GPU.
        x, sf, ids, w, _ = _pack_tensors()
        fields = dict(hidden_states_scale=sf, topk_ids=ids, topk_weights=w)
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

    def test_logits_device_mutation_caught_at_runner_boundary(self):
        from flashinfer.fused_moe.runners import _validate_logits_inputs

        pack = self._logits_pack(torch.zeros(4, 16, dtype=torch.float32))
        pack.routing_logits = torch.zeros(
            pack.routing_logits.shape, dtype=torch.float32, device="meta"
        )
        with pytest.raises(ValueError, match="device"):
            _validate_logits_inputs(pack, 4, 16, "T")


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
        topk_ids=tensors["token_selected_experts"],
        topk_weights=tensors["token_final_scales"],
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
            topk_ids=selected_experts,
            topk_weights=final_scales,
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
                topk_ids=tensors["token_selected_experts"] + offset,
                topk_weights=tensors["token_final_scales"],
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


# ---------------------------------------------------------------------------
# 4. FromLogits packing contract (gh #3595)
# ---------------------------------------------------------------------------


@sm100_required
class TestTrtllmFromLogitsPackingContract:
    """FromLogits buffer allocation must follow the kernel's output contract.

    The fp4 routing kernel writes bf16 expert weights regardless of the logits
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
        runner = TrtllmFp4RoutedRunner(config, device=device)

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
            {
                "gemm1_weights": torch.empty(0, device=device),
                "gemm1_weights_scale": torch.empty(0, device=device),
                "gemm1_alpha": torch.empty(0, device=device),
                "gemm2_weights": torch.empty(0, device=device),
                "gemm2_weights_scale": torch.empty(0, device=device),
            },
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
