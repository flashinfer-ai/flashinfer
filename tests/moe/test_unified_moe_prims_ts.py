"""Tests for the Prims-TS backend through the unified MoE API.

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

  * SM100/SM103 GPU tests that require nvidia-cutlass-dsl: accuracy against
    the existing unified NVFP4/BF16 references, direct pack_inputs+forward,
    and CUDA-graph replay.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from flashinfer.fused_moe import (
    MoELayer,
    MoEWeightPack,
    PrimsTsConfig,
    PrimsTsRunner,
)
from flashinfer.fused_moe.api import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    GeGLU,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    QuantConfig,
    QuantFormat,
    ReLU2,
    RoutingConfig,
    RoutingInputMode,
    SiTU,
    SwiGLU,
    TrtllmBf16Config,
    TrtllmFp4Config,
    _DEFAULT_BACKEND,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.prims_ts.utils import is_prims_ts_available
from flashinfer.tllm_enums import RoutingMethodType
from flashinfer.utils import get_compute_capability
from tests.moe.test_unified_moe import (
    SMALL,
    _bf16_check,
    _bf16_ref,
    _build_direct_runner,
    _compute_ref,
    _make_bf16_packs_and_config,
    _make_packs_and_config,
    _nvfp4_check,
)

_NVFP4 = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
_BF16 = QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16)


def _prims_ts_gpu_available() -> bool:
    if not torch.cuda.is_available() or not is_prims_ts_available():
        return False
    return get_compute_capability(torch.device("cuda")) in ((10, 0), (10, 3))


prims_ts_unified_required = pytest.mark.skipif(
    not _prims_ts_gpu_available(),
    reason="Unified Prims-TS MoE requires SM100/SM103 and nvidia-cutlass-dsl",
)


def _config(backend=None, variant=_NVFP4, **overrides):
    fields = dict(
        routing=RoutingConfig(num_experts=32, top_k=2),
        quant=variant,
        experts=ExpertConfig(intermediate_size=512),
        activation=SwiGLU(),
        backend=BackendOptions((backend or PrimsTsConfig(),)),
        execution=ExecutionConfig(),
        finalize=MoEFinalizeConfig(),
    )
    fields.update(overrides)
    return MoEConfig(**fields)


def _attach_prims_ts_view(weights: MoEWeightPack, source_key: str) -> None:
    weights.prepare_for("prims_ts", weights.get_view(source_key))


def _with_prims_ts_backend(act, weights, config, source_key: str):
    _attach_prims_ts_view(weights, source_key)
    config = dataclasses.replace(
        config,
        backend=BackendOptions((PrimsTsConfig(),)),
    )
    return act, weights, config


class TestPrimsTsBackendOptions:
    def test_activation_and_quant_capabilities(self):
        assert PrimsTsRunner.backend_key == "prims_ts"
        assert PrimsTsRunner.supported_quant_variants == (
            (QuantFormat.NVFP4, QuantFormat.NVFP4),
            (QuantFormat.BF16, QuantFormat.BF16),
        )
        assert PrimsTsRunner.supported_activation_classes_by_quant == {
            (QuantFormat.NVFP4, QuantFormat.NVFP4): (SwiGLU, GeGLU, SiTU, ReLU2),
            (QuantFormat.BF16, QuantFormat.BF16): (SwiGLU, ReLU2),
        }
        assert not PrimsTsRunner.supports_fused_shared_experts

    def test_sm100_family_only(self):
        opts = BackendOptions((PrimsTsConfig(),))
        assert eval(repr(opts)) == opts
        assert opts.valid_for(90) == []
        assert opts.valid_for(100) == list(opts)
        assert opts.valid_for(103) == list(opts)
        assert opts.valid_for(107) == []
        assert opts.valid_for(120) == []

    def test_repr_round_trip(self):
        import flashinfer.fused_moe as fused_moe

        cfg = PrimsTsConfig()
        assert repr(cfg) == "PrimsTsConfig()"
        assert eval(repr(cfg), vars(fused_moe)) == cfg

    def test_registered_and_opt_in(self):
        assert _BACKEND_RUNNERS[PrimsTsConfig] is PrimsTsRunner
        assert not any(isinstance(item, PrimsTsConfig) for item in _DEFAULT_BACKEND)


class TestPrimsTsPrepare:
    def test_prepare_weights_rejects_unsupported_pairs(self):
        with pytest.raises(ValueError, match="NVFP4×NVFP4 and BF16×BF16"):
            PrimsTsConfig.prepare_weights(
                torch.empty(0),
                torch.empty(0),
                quant=QuantConfig(
                    weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8
                ),
                num_local_experts=1,
                hidden_size=128,
                intermediate_size=128,
            )

    def test_prepare_activations_rejects_unsupported_pairs(self):
        with pytest.raises(ValueError, match="NVFP4×NVFP4 and BF16×BF16"):
            PrimsTsConfig.prepare_activations(
                torch.empty(1, 128, dtype=torch.bfloat16),
                quant=QuantConfig(
                    weight=QuantFormat.DeepSeekFp8,
                    activation=QuantFormat.DeepSeekFp8,
                ),
            )


class TestPrimsTsUnifiedValidation:
    @staticmethod
    def _runner(config):
        runner = PrimsTsRunner.__new__(PrimsTsRunner)
        runner.config = config
        runner.device = torch.device("cpu")
        return runner

    @pytest.fixture(autouse=True)
    def _prims_ts_cpu_support_env(self, monkeypatch):
        monkeypatch.setattr(
            "flashinfer.prims_ts.utils.is_prims_ts_available", lambda: True
        )
        monkeypatch.setattr(
            "flashinfer.utils.get_compute_capability", lambda _: (10, 0)
        )

    def test_missing_dsl_is_skipped(self, monkeypatch):
        monkeypatch.setattr(
            "flashinfer.prims_ts.utils.is_prims_ts_available", lambda: False
        )
        runner = self._runner(_config())
        with pytest.raises(RuntimeError, match="nvidia-cutlass-dsl"):
            runner.check_support()

    def test_unsupported_arch_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "flashinfer.utils.get_compute_capability", lambda _: (10, 7)
        )
        runner = self._runner(_config())
        with pytest.raises(NotImplementedError, match="SM100/SM103"):
            runner.check_support()

    def test_intermediate_size_alignment(self):
        runner = self._runner(_config(experts=ExpertConfig(intermediate_size=64)))
        with pytest.raises(NotImplementedError, match="intermediate_size"):
            runner.check_support()

    def test_bf16_rejects_sigmoid_routing(self):
        runner = self._runner(
            _config(
                variant=_BF16,
                routing=RoutingConfig(
                    num_experts=32,
                    top_k=2,
                    method=RoutingMethodType.Sigmoid,
                ),
            )
        )
        with pytest.raises(NotImplementedError, match="Sigmoid"):
            runner.check_support()

    def test_geglu_does_not_forward_prepare_alpha(self):
        alpha = torch.ones(32, dtype=torch.float32)
        view = {"gemm1_alpha": alpha, "gemm1_beta": None, "gemm1_clamp_limit": None}
        geglu = self._runner(_config(activation=GeGLU()))._gemm1_oa_launch_kwargs(view)
        assert geglu["gemm1_alpha"] is None
        assert geglu["gemm1_beta"] is None
        swiglu = self._runner(_config())._gemm1_oa_launch_kwargs(view)
        assert swiglu["gemm1_alpha"] is alpha

    def test_staged_routing_placeholders_match_bf16_runner(self):
        runner = self._runner(_config(execution=ExecutionConfig(enable_pdl=False)))
        hidden = torch.zeros(4, 8, dtype=torch.bfloat16)
        packed_act = MoEActivationPack(
            hidden,
            None,
            torch.arange(8, dtype=torch.int32).reshape(4, 2),
            torch.ones(4, 2, dtype=torch.float32),
        )
        _, _, packed_ids, packed_weights = runner._pack_routing(packed_act)
        assert packed_ids.shape == (4, 2)
        assert packed_ids.dtype == torch.int32
        assert packed_weights.shape == (4, 2)
        assert packed_weights.dtype == torch.bfloat16

        logits_act = MoEActivationPack(
            hidden,
            None,
            routing_input_mode=RoutingInputMode.FromLogits,
            routing_logits=torch.zeros(4, 32, dtype=torch.float32),
        )
        _, _, logits_ids, logits_weights = runner._pack_routing(logits_act)
        assert logits_ids.shape == (4, 2)
        assert logits_weights.shape == (4, 2)

        unpacked_act = MoEActivationPack(
            hidden,
            None,
            torch.arange(8, dtype=torch.int32).reshape(4, 2),
            torch.ones(4, 2, dtype=torch.bfloat16),
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        )
        _, _, unpacked_ids, unpacked_weights = runner._pack_routing(unpacked_act)
        assert unpacked_ids.shape == (4, 2)
        assert unpacked_weights.shape == (4, 2)
        assert unpacked_weights is unpacked_act.topk_weights

    def test_situ_unclamped_linear_rejected(self):
        runner = self._runner(_config(activation=SiTU(linear_scale=None)))
        with pytest.raises(NotImplementedError, match="linear_scale=None"):
            runner.check_support()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="MoELayer needs a CUDA device"
    )
    def test_moe_layer_skips_when_dsl_missing(self, monkeypatch):
        monkeypatch.setattr(
            "flashinfer.prims_ts.utils.is_prims_ts_available", lambda: False
        )
        monkeypatch.setattr(
            "flashinfer.fused_moe.layer.get_compute_capability", lambda _: (10, 0)
        )
        with pytest.raises(RuntimeError, match="none of the configured backends"):
            MoELayer(_config())


@prims_ts_unified_required
class TestPrimsTsUnifiedGpu:
    def test_nvfp4_layer_and_direct_runner_match_reference(self):
        act, weights, config, tensors = _make_packs_and_config(
            16, max_tokens=16, **SMALL
        )
        act, weights, config = _with_prims_ts_backend(
            act, weights, config, "trtllm_fp4_routed"
        )
        reference = _compute_ref(act, tensors, SMALL)
        layer_out = MoELayer(config)(act, weights)
        _nvfp4_check(layer_out, reference, "prims_ts nvfp4 layer")

        runner = _build_direct_runner(PrimsTsRunner, config, act.hidden_states_q.device)
        packed = runner.pack_inputs(act, weights)
        direct_out = runner.forward(packed, tactic=-1)
        _nvfp4_check(direct_out, reference, "prims_ts nvfp4 direct")

    def test_nvfp4_geglu_matches_trtllm_without_oa_placeholder(self):
        # Cute-DSL's FP4 reference does not implement GeGLU. TRT-LLM does, and
        # prepare_weights still inserts gemm1_alpha=ones; the adapter must drop
        # that placeholder or the first Prims-TS forward fails support.
        act, weights, config, _tensors = _make_packs_and_config(
            16, max_tokens=16, activation=GeGLU(), **SMALL
        )
        _attach_prims_ts_view(weights, "trtllm_fp4_routed")
        prims_out = MoELayer(
            dataclasses.replace(config, backend=BackendOptions((PrimsTsConfig(),)))
        )(act, weights)
        trtllm_out = MoELayer(
            dataclasses.replace(config, backend=BackendOptions((TrtllmFp4Config(),)))
        )(act, weights)
        _nvfp4_check(prims_out, trtllm_out.float(), "prims_ts nvfp4 geglu vs trtllm")

    def test_bf16_layer_and_direct_runner_match_reference(self):
        act, weights, config, tensors = _make_bf16_packs_and_config(
            16,
            hidden_size=256,
            intermediate_size=256,
            num_experts=8,
            top_k=2,
            max_tokens=16,
        )
        act, weights, config = _with_prims_ts_backend(
            act, weights, config, "trtllm_bf16_routed"
        )
        reference = _bf16_ref(act, tensors)
        layer_out = MoELayer(config)(act, weights)
        _bf16_check(layer_out, reference, "prims_ts bf16 layer")

        runner = _build_direct_runner(PrimsTsRunner, config, act.hidden_states_q.device)
        packed = runner.pack_inputs(act, weights)
        direct_out = runner.forward(packed, tactic=-1)
        _bf16_check(direct_out, reference, "prims_ts bf16 direct")

    def test_nvfp4_cuda_graph_replay_matches_eager(self):
        from flashinfer.autotuner import autotune

        act, weights, config, tensors = _make_packs_and_config(
            16, max_tokens=16, **SMALL
        )
        act, weights, config = _with_prims_ts_backend(
            act, weights, config, "trtllm_fp4_routed"
        )
        with autotune(True):
            layer = MoELayer(config)
            for _ in range(2):
                _ = layer(act, weights)
        eager = layer(act, weights).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(act, weights)
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        _nvfp4_check(captured, eager.float(), "prims_ts nvfp4 graph replay")
        _nvfp4_check(
            captured, _compute_ref(act, tensors, SMALL), "prims_ts nvfp4 graph vs ref"
        )

    def test_prepare_helpers_match_trtllm_views(self):
        act, weights, config, tensors = _make_packs_and_config(8, max_tokens=8, **SMALL)
        del act, config
        device = torch.device("cuda", torch.cuda.current_device())
        view = PrimsTsConfig.prepare_weights(
            tensors["w1_weight_bf16"],
            tensors["w2_weight_bf16"],
            quant=_NVFP4,
            num_local_experts=SMALL["num_experts"],
            hidden_size=SMALL["hidden_size"],
            intermediate_size=SMALL["intermediate_size"],
            activation=SwiGLU(),
            device=device,
        )
        trtllm = TrtllmFp4Config.prepare_weights(
            tensors["w1_weight_bf16"],
            tensors["w2_weight_bf16"],
            quant=_NVFP4,
            num_local_experts=SMALL["num_experts"],
            hidden_size=SMALL["hidden_size"],
            intermediate_size=SMALL["intermediate_size"],
            activation=SwiGLU(),
            device=device,
        )
        assert view.keys() == trtllm.keys()
        for key, tensor in view.items():
            torch.testing.assert_close(tensor, trtllm[key])

        bf16_w1 = torch.randn(4, 256, 128, device=device, dtype=torch.bfloat16)
        bf16_w2 = torch.randn(4, 128, 128, device=device, dtype=torch.bfloat16)
        prims_bf16 = PrimsTsConfig.prepare_weights(
            bf16_w1,
            bf16_w2,
            quant=_BF16,
            num_local_experts=4,
            hidden_size=128,
            intermediate_size=128,
            device=device,
        )
        trtllm_bf16 = TrtllmBf16Config.prepare_weights(
            bf16_w1,
            bf16_w2,
            num_local_experts=4,
            hidden_size=128,
            intermediate_size=128,
            device=device,
        )
        assert prims_bf16.keys() == trtllm_bf16.keys()
        for key, tensor in prims_bf16.items():
            torch.testing.assert_close(tensor, trtllm_bf16[key])
