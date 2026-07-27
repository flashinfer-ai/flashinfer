"""Tests for the b12x backends through the unified MoE API.

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

  * CPU-only configuration and runner-support tests.

  * SM120/SM121 GPU tests for the b12x NVFP4 and W4A16 runners: accuracy,
    CUDA-graph replay, legacy-wrapper consistency, and backend regressions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe import (
    B12xNvfp4Config,
    B12xNvfp4Runner,
    B12xW4A16Config,
    B12xW4A16Runner,
    MoEActivationPack,
    MoELayer,
    MoEWeightPack,
)
from flashinfer.fused_moe.api import (
    ActivationConfig,
    ActivationType,
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    MoEConfig,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    RoutingInputMode,
)
from tests.moe.test_b12x_fused_moe import (  # noqa: E402
    check_accuracy as check_b12x_accuracy,
    compute_reference_moe_fp4 as compute_b12x_reference_moe_fp4,
    compute_reference_moe_relu2 as compute_b12x_reference_moe_relu2,
    create_moe_tensors as create_b12x_moe_tensors,
    create_relu2_moe_tensors as create_b12x_relu2_moe_tensors,
    cuda_13_required,
    cute_dsl_available,
    sm120_required,
)


# ---------------------------------------------------------------------------
# B12x backend configuration
# ---------------------------------------------------------------------------


class TestB12xBackendOptions:
    def test_b12x_is_sm12x_only(self):
        opts = BackendOptions((B12xNvfp4Config(), B12xW4A16Config()))
        assert eval(repr(opts)) == opts
        assert opts.valid_for(100) == []
        assert opts.valid_for(120) == list(opts)
        assert opts.valid_for(121) == list(opts)


# ---------------------------------------------------------------------------
# B12x runner support validation
# ---------------------------------------------------------------------------


class TestB12xUnifiedValidation:
    @staticmethod
    def _runner(config, runner_type=B12xNvfp4Runner):
        runner = runner_type.__new__(runner_type)
        runner.config = config
        runner.device = torch.device("cuda", 0)
        return runner

    @staticmethod
    def _mock_environment(
        monkeypatch, *, cuda_major=13, cute_dsl_available=True, capability=(12, 0)
    ):
        monkeypatch.setattr(
            "flashinfer.jit.cpp_ext.get_cuda_version",
            lambda: SimpleNamespace(major=cuda_major),
        )
        monkeypatch.setattr(
            "flashinfer.cute_dsl.is_cute_dsl_available",
            lambda: cute_dsl_available,
        )
        monkeypatch.setattr(
            "flashinfer.utils.get_compute_capability",
            lambda _: capability,
        )

    def _config(
        self,
        backend,
        variant,
        *,
        activation=ActivationConfig.swiglu,
        experts=None,
        execution=None,
    ):
        return MoEConfig(
            routing=RoutingConfig(num_experts=8, top_k=2),
            quant=QuantConfig(variant=variant),
            experts=experts or ExpertConfig(intermediate_size=512),
            activation=activation,
            backend=BackendOptions((backend,)),
            execution=execution or ExecutionConfig(),
        )

    @pytest.mark.parametrize("runner_type", (B12xNvfp4Runner, B12xW4A16Runner))
    def test_b12x_supports_precomputed_routing(self, runner_type):
        assert runner_type.supported_routing_modes == (
            RoutingInputMode.PackedPrecomputed,
        )

    @pytest.mark.parametrize(
        "runner_type,expected",
        ((B12xNvfp4Runner, "nvfp4"), (B12xW4A16Runner, "w4a16")),
    )
    def test_b12x_quant_mode_name(self, runner_type, expected):
        runner = object.__new__(runner_type)
        assert runner._get_quant_mode_name() == expected

    @pytest.mark.parametrize("variants", ((), (QuantVariant.NVFP4, QuantVariant.W4A16)))
    def test_b12x_quant_mode_requires_one_variant(self, variants):
        runner = object.__new__(B12xNvfp4Runner)
        runner.supported_quant_variants = variants
        with pytest.raises(ValueError, match="exactly one"):
            runner._get_quant_mode_name()

    @pytest.mark.parametrize(
        "runner_type,backend,variant",
        (
            (B12xNvfp4Runner, B12xNvfp4Config(), QuantVariant.NVFP4),
            (B12xW4A16Runner, B12xW4A16Config(), QuantVariant.W4A16),
        ),
    )
    @pytest.mark.parametrize(
        "activation", (ActivationConfig.swiglu, ActivationConfig.relu2)
    )
    def test_b12x_accepts_implemented_activations(
        self, monkeypatch, runner_type, backend, variant, activation
    ):
        config = self._config(
            backend,
            variant,
            activation=activation,
        )
        self._mock_environment(monkeypatch)
        assert self._runner(config, runner_type).check_support() is None

    def test_b12x_w4a16_rejects_geglu_tanh(self, monkeypatch):
        config = self._config(
            B12xW4A16Config(),
            QuantVariant.W4A16,
            activation=ActivationConfig(ActivationType.GegluTanh),
        )
        self._mock_environment(monkeypatch)
        runner = self._runner(config, B12xW4A16Runner)
        with pytest.raises(NotImplementedError, match="Swiglu or Relu2"):
            runner.check_support()

    @pytest.mark.parametrize(
        "environment,error,match",
        (
            ({"cuda_major": 12}, ValueError, "CUDA 13"),
            (
                {"cute_dsl_available": False},
                RuntimeError,
                "CuTe DSL package",
            ),
            ({"capability": (10, 0)}, RuntimeError, "SM120 or SM121"),
        ),
    )
    def test_b12x_environment_not_supported(
        self, monkeypatch, environment, error, match
    ):
        config = self._config(B12xNvfp4Config(), QuantVariant.NVFP4)
        self._mock_environment(monkeypatch, **environment)
        with pytest.raises(error, match=match):
            self._runner(config).check_support()

    def test_b12x_does_not_support_expert_parallelism(self, monkeypatch):
        config = self._config(
            B12xNvfp4Config(),
            QuantVariant.NVFP4,
            experts=ExpertConfig(
                intermediate_size=512,
                local_expert_offset=4,
                local_num_experts=4,
            ),
        )
        self._mock_environment(monkeypatch)
        runner = self._runner(config)
        with pytest.raises(NotImplementedError, match="expert parallelism"):
            runner.check_support()

    def test_b12x_does_not_support_unfinalized_output(self, monkeypatch):
        config = self._config(
            B12xNvfp4Config(),
            QuantVariant.NVFP4,
            execution=ExecutionConfig(do_finalize=False),
        )
        self._mock_environment(monkeypatch)
        runner = self._runner(config)
        with pytest.raises(NotImplementedError, match="do_finalize"):
            runner.check_support()

    def test_layer_skips_runner_when_support_check_fails(self, monkeypatch):
        config = self._config(
            B12xNvfp4Config(),
            QuantVariant.NVFP4,
        )
        monkeypatch.setattr(
            "flashinfer.fused_moe.layer.get_compute_capability", lambda _: (12, 0)
        )

        def init_runner(runner, config, device):
            runner.config = config

        def check_support(runner):
            raise ValueError("unsupported")

        monkeypatch.setattr(B12xNvfp4Runner, "__init__", init_runner)
        monkeypatch.setattr(B12xNvfp4Runner, "check_support", check_support)
        with pytest.raises(RuntimeError, match="for this configuration"):
            MoELayer(config, device=torch.device("cpu"))

    @pytest.mark.parametrize(
        "runner_type,backend,variant",
        (
            (B12xNvfp4Runner, B12xNvfp4Config(), QuantVariant.W4A16),
            (B12xW4A16Runner, B12xW4A16Config(), QuantVariant.NVFP4),
        ),
    )
    def test_quantization_backend_mismatch_rejected(
        self, runner_type, backend, variant
    ):
        config = self._config(backend, variant)
        with pytest.raises(NotImplementedError, match=f"QuantVariant.{variant.name}"):
            self._runner(config, runner_type).check_support()

    def test_w4a16_prepare_does_not_expose_fp16_mode(self):
        with pytest.raises(TypeError, match="params_dtype"):
            B12xW4A16Config.prepare_weights(
                None,
                None,
                None,
                None,
                None,
                None,
                params_dtype=torch.float16,
            )

    @pytest.mark.parametrize(
        "activation_type,expected",
        (
            (ActivationType.Swiglu, "silu"),
            (ActivationType.GegluTanh, "gelu_tanh"),
            (ActivationType.Relu2, "relu2"),
        ),
    )
    def test_get_b12x_activation_name(self, activation_type, expected):
        from flashinfer.fused_moe.utils import get_b12x_activation_name

        assert get_b12x_activation_name(activation_type) == expected

    @pytest.mark.parametrize(
        "activation_type",
        (
            ActivationType.Geglu,
            ActivationType.SwigluBias,
            ActivationType.Identity,
            ActivationType.SwigluStep,
        ),
    )
    def test_get_b12x_activation_name_rejects_unsupported(self, activation_type):
        from flashinfer.fused_moe.utils import get_b12x_activation_name

        with pytest.raises(ValueError, match="Unsupported b12x activation type"):
            get_b12x_activation_name(activation_type)

    @pytest.mark.parametrize(
        "runner_type,has_fc2_input_scale",
        (
            (B12xNvfp4Runner, True),
            (B12xW4A16Runner, False),
        ),
    )
    def test_b12x_runner_delegates_to_wrapper(self, runner_type, has_fc2_input_scale):
        calls = {}

        class Wrapper:
            def run(self, **kwargs):
                calls.update(kwargs)
                return kwargs["x"]

        weight = torch.empty(0)
        prepared = {
            "w1_weight": weight,
            "w1_weight_sf": weight,
            "w1_alpha": weight,
            "w2_weight": weight,
            "w2_weight_sf": weight,
            "w2_alpha": weight,
        }
        if has_fc2_input_scale:
            prepared["fc2_input_scale"] = weight

        runner = object.__new__(runner_type)
        runner._prepared_weights = prepared
        runner._inner = Wrapper()
        hidden = torch.empty(1, 16)
        expert_ids = torch.zeros(1, 1, dtype=torch.int32)
        routing_weights = torch.ones(1, 1)

        assert runner.forward([hidden, expert_ids, routing_weights]) is hidden
        assert calls["fc2_input_scale"] is prepared.get("fc2_input_scale")


# ---------------------------------------------------------------------------
# SM120/SM121 b12x conformance
# ---------------------------------------------------------------------------
_B12X_ACTIVATIONS = {
    "silu": ActivationConfig.swiglu,
    "relu2": ActivationConfig.relu2,
}


def _make_b12x_tensors(
    *,
    activation: str,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    seed: int = 42,
):
    make_tensors = (
        create_b12x_relu2_moe_tensors
        if activation == "relu2"
        else create_b12x_moe_tensors
    )
    return make_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=num_experts,
        top_k=top_k,
        seed=seed,
    )


def _make_b12x_layer_and_packs(
    tensors,
    *,
    variant: QuantVariant,
    activation: str,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    source_format: str = "modelopt",
):
    activation_config = _B12X_ACTIVATIONS[activation]
    hidden_size = tensors["x_bf16"].shape[1]
    if variant is QuantVariant.NVFP4:
        backend_config = B12xNvfp4Config()
        prepared = backend_config.prepare_weights(
            tensors["w1_weight_bf16"],
            tensors["w2_weight_bf16"],
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation_config,
        )
        backend_key = "b12x_nvfp4"
    else:
        backend_config = B12xW4A16Config()
        w1_alpha = tensors["w1_alpha"]
        w2_alpha = tensors["w2_alpha"]
        if source_format == "compressed_tensors":
            w1_alpha = w1_alpha.reciprocal()
            w2_alpha = w2_alpha.reciprocal()
        prepared = backend_config.prepare_weights(
            tensors["w1_weight"],
            tensors["w1_weight_sf"],
            w1_alpha,
            tensors["w2_weight"],
            tensors["w2_weight_sf"],
            w2_alpha,
            activation=activation_config,
            source_format=source_format,
        )
        backend_key = "b12x_w4a16"

    assert all(isinstance(value, torch.Tensor) for value in prepared.values())

    act_pack = MoEActivationPack(
        hidden_states_q=tensors["x_bf16"],
        hidden_states_scale=torch.empty(0, device="cuda"),
        topk_ids=tensors["token_selected_experts"],
        topk_weights=tensors["token_final_scales"],
    )
    weight_pack = MoEWeightPack()
    weight_pack.prepare_for(backend_key, prepared)
    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=variant),
        experts=ExpertConfig(
            intermediate_size=intermediate_size,
            local_num_experts=num_experts,
        ),
        activation=activation_config,
        backend=BackendOptions((backend_config,)),
        execution=ExecutionConfig(tune_max_num_tokens=tensors["x_bf16"].shape[0]),
    )
    return MoELayer(config), act_pack, weight_pack


def _run_b12x_unified(layer, act_pack, weight_pack):
    runner = layer.runners[0]
    layer._select_winner = lambda *_: (runner, -1)
    output = layer(act_pack, weight_pack)
    assert layer.winner_backend == runner.backend_key
    return output


def _b12x_reference(
    tensors,
    *,
    variant: QuantVariant,
    activation: str,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
):
    common = dict(
        hidden_states=tensors["x_bf16"].float(),
        token_selected_experts=tensors["token_selected_experts"],
        token_final_scales=tensors["token_final_scales"],
        num_tokens=tensors["x_bf16"].shape[0],
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=tensors["x_bf16"].shape[1],
        intermediate_size=intermediate_size,
        fc2_input_scale=(
            tensors["fc2_input_scale"] if variant is QuantVariant.NVFP4 else None
        ),
    )
    if activation == "relu2":
        return compute_b12x_reference_moe_relu2(
            fc1_weights=tensors["w1_weight_bf16"].float(),
            fc2_weights=tensors["w2_weight_bf16"].float(),
            **common,
        )
    return compute_b12x_reference_moe_fp4(
        gemm1_weights=tensors["w1_weight_bf16"].float(),
        gemm2_weights=tensors["w2_weight_bf16"].float(),
        **common,
    )


def _assert_b12x_accurate(actual, expected):
    assert torch.isfinite(actual).all()
    passed, percent_within, atol = check_b12x_accuracy(actual, expected)
    assert passed, f"{percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"


def _assert_b12x_case(
    *,
    variant: QuantVariant,
    activation: str,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
):
    tensors = _make_b12x_tensors(
        activation=activation,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
    )
    layer, act_pack, weight_pack = _make_b12x_layer_and_packs(
        tensors,
        variant=variant,
        activation=activation,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
    )
    actual = _run_b12x_unified(layer, act_pack, weight_pack)
    expected = _b12x_reference(
        tensors,
        variant=variant,
        activation=activation,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
    )
    _assert_b12x_accurate(actual, expected)


_B12X_DISPATCH_CASES = (
    # Existing numerical, micro, static, dynamic, and ReLU2 paths.
    (QuantVariant.NVFP4, "silu", 1, 8, 256, 256, 512),
    (QuantVariant.NVFP4, "silu", 128, 1, 256, 256, 512),
    (QuantVariant.NVFP4, "silu", 515, 8, 384, 256, 512),
    (QuantVariant.NVFP4, "silu", 128, 2, 8, 1024, 2048),
    (QuantVariant.NVFP4, "relu2", 1, 1, 256, 256, 512),
    (QuantVariant.NVFP4, "relu2", 128, 2, 256, 256, 512),
    (QuantVariant.NVFP4, "relu2", 512, 8, 256, 256, 512),
    (QuantVariant.W4A16, "silu", 1, 2, 64, 512, 256),
    (QuantVariant.W4A16, "silu", 64, 2, 256, 256, 512),
    (QuantVariant.W4A16, "silu", 384, 2, 256, 256, 512),
    (QuantVariant.W4A16, "relu2", 2, 2, 64, 512, 256),
    (QuantVariant.W4A16, "relu2", 64, 2, 256, 256, 512),
    (QuantVariant.W4A16, "relu2", 384, 2, 256, 256, 512),
)


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestUnifiedB12xConformance:
    @pytest.mark.parametrize(
        "variant,activation,num_tokens,top_k,num_experts,hidden_size,intermediate_size",
        _B12X_DISPATCH_CASES,
    )
    def test_dispatch_accuracy(
        self,
        variant,
        activation,
        num_tokens,
        top_k,
        num_experts,
        hidden_size,
        intermediate_size,
    ):
        _assert_b12x_case(
            variant=variant,
            activation=activation,
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    @pytest.mark.parametrize(
        "variant,activation,num_tokens,intermediate_size",
        (
            (QuantVariant.NVFP4, "silu", 8, 704),
            (QuantVariant.NVFP4, "silu", 128, 704),
            (QuantVariant.W4A16, "silu", 32, 192),
        ),
    )
    def test_ragged_intermediate(
        self, variant, activation, num_tokens, intermediate_size
    ):
        _assert_b12x_case(
            variant=variant,
            activation=activation,
            num_tokens=num_tokens,
            top_k=2,
            num_experts=8 if intermediate_size == 704 else 256,
            hidden_size=512 if intermediate_size == 704 else 256,
            intermediate_size=intermediate_size,
        )

    @pytest.mark.parametrize(
        "num_tokens,top_k,num_experts", ((2, 8, 8), (4, 8, 16), (4, 4, 8))
    )
    def test_micro_pairs_exceed_experts(self, num_tokens, top_k, num_experts):
        _assert_b12x_case(
            variant=QuantVariant.NVFP4,
            activation="silu",
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=256,
            intermediate_size=512,
        )

    @pytest.mark.parametrize("num_tokens", (64, 384))
    def test_w4a16_is_more_accurate_than_w4a4(self, num_tokens):
        tensors = _make_b12x_tensors(
            activation="silu",
            num_tokens=num_tokens,
            hidden_size=256,
            intermediate_size=512,
            num_experts=256,
            top_k=2,
            seed=123,
        )
        outputs = {}
        for variant in (QuantVariant.NVFP4, QuantVariant.W4A16):
            layer, act_pack, weight_pack = _make_b12x_layer_and_packs(
                tensors,
                variant=variant,
                activation="silu",
                intermediate_size=512,
                num_experts=256,
                top_k=2,
            )
            outputs[variant] = _run_b12x_unified(layer, act_pack, weight_pack).float()
        expected = _b12x_reference(
            tensors,
            variant=QuantVariant.W4A16,
            activation="silu",
            intermediate_size=512,
            num_experts=256,
            top_k=2,
        )
        a4_error = (outputs[QuantVariant.NVFP4] - expected).abs()
        a16_error = (outputs[QuantVariant.W4A16] - expected).abs()
        assert a16_error.mean() < 0.75 * a4_error.mean()
        assert a16_error.square().mean() < 0.5 * a4_error.square().mean()

    @pytest.mark.parametrize(
        "variant,activation,num_tokens,hidden_size,intermediate_size,num_experts",
        (
            (QuantVariant.NVFP4, "silu", 128, 256, 512, 256),
            (QuantVariant.NVFP4, "relu2", 128, 256, 512, 256),
            (QuantVariant.W4A16, "silu", 2, 512, 256, 64),
        ),
    )
    def test_cuda_graph(
        self,
        variant,
        activation,
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
    ):
        tensors = _make_b12x_tensors(
            activation=activation,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=2,
            seed=780,
        )
        layer, act_pack, weight_pack = _make_b12x_layer_and_packs(
            tensors,
            variant=variant,
            activation=activation,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=2,
        )
        runner = layer.runners[0]
        inputs = runner.pack_inputs(act_pack, weight_pack)
        for _ in range(3):
            runner.forward(inputs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = runner.forward(inputs)
        graph.replay()
        torch.cuda.synchronize()
        assert output.shape == tensors["x_bf16"].shape
        assert torch.isfinite(output).all()
        assert not (output == 0).all()

    @pytest.mark.parametrize(
        "variant,source_format",
        (
            (QuantVariant.NVFP4, "modelopt"),
            (QuantVariant.W4A16, "modelopt"),
            (QuantVariant.W4A16, "compressed_tensors"),
        ),
    )
    def test_matches_legacy_wrapper(self, variant, source_format):
        from flashinfer.fused_moe import B12xMoEWrapper

        tensors = _make_b12x_tensors(
            activation="silu",
            num_tokens=64,
            hidden_size=256,
            intermediate_size=512,
            num_experts=256,
            top_k=2,
            seed=321,
        )
        layer, act_pack, weight_pack = _make_b12x_layer_and_packs(
            tensors,
            variant=variant,
            activation="silu",
            intermediate_size=512,
            num_experts=256,
            top_k=2,
            source_format=source_format,
        )
        unified = _run_b12x_unified(layer, act_pack, weight_pack).clone()
        w1_alpha = tensors["w1_alpha"]
        w2_alpha = tensors["w2_alpha"]
        if source_format == "compressed_tensors":
            w1_alpha = w1_alpha.reciprocal()
            w2_alpha = w2_alpha.reciprocal()

        wrapper = B12xMoEWrapper(
            num_experts=256,
            top_k=2,
            hidden_size=256,
            intermediate_size=512,
            quant_mode="w4a16" if variant is QuantVariant.W4A16 else "nvfp4",
            source_format=source_format,
        )
        legacy = wrapper.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=w1_alpha,
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=w2_alpha,
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )
        torch.testing.assert_close(unified, legacy, rtol=0.05, atol=0.05)

    def test_w4a16_source_formats_match(self):
        tensors = _make_b12x_tensors(
            activation="silu",
            num_tokens=8,
            hidden_size=256,
            intermediate_size=512,
            num_experts=8,
            top_k=2,
        )
        tensors["w1_alpha"].fill_(1.75)
        tensors["w2_alpha"].fill_(0.75)
        prepared_views = []
        for source_format in ("modelopt", "compressed_tensors"):
            _, _, weight_pack = _make_b12x_layer_and_packs(
                tensors,
                variant=QuantVariant.W4A16,
                activation="silu",
                intermediate_size=512,
                num_experts=8,
                top_k=2,
                source_format=source_format,
            )
            prepared_views.append(weight_pack.get_view("b12x_w4a16"))
        for field in (
            "w1_weight",
            "w1_weight_sf",
            "w1_alpha",
            "w2_weight",
            "w2_weight_sf",
            "w2_alpha",
        ):
            rtol = 1e-6 if field in ("w1_alpha", "w2_alpha") else 0
            torch.testing.assert_close(
                prepared_views[0][field],
                prepared_views[1][field],
                rtol=rtol,
                atol=0,
            )
