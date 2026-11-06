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

  * SM100/SM103 GPU tests that require nvidia-cutlass-dsl: accuracy across
    supported quantization and activation pairs, direct pack_inputs+forward,
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
    SiLU,
    SiTU,
    SwiGLU,
    TrtllmBf16Config,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
    _DEFAULT_BACKEND,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.prims_ts.utils import is_prims_ts_available
from flashinfer.tllm_enums import RoutingMethodType, WeightLayout
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
from tests.moe.utils import fp8_per_tensor_global_scale

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


def _silu_reference(x, w1, w2, topk_ids, routing_weights):
    out = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        token, slot = torch.where(topk_ids == expert)
        if token.numel() == 0:
            continue
        fc1 = x[token].float() @ w1[expert].float().T
        fc2 = torch.nn.functional.silu(fc1) @ w2[expert].float().T
        out[token] += routing_weights[token, slot, None].float() * fc2
    return out


class TestPrimsTsBackendOptions:
    def test_activation_and_quant_capabilities(self):
        assert PrimsTsRunner.backend_key == "prims_ts"
        assert PrimsTsRunner.supported_quant_variants == (
            (QuantFormat.NVFP4, QuantFormat.NVFP4),
            (QuantFormat.MXFP4, QuantFormat.MXFP8),
            (QuantFormat.MXFP4, QuantFormat.BF16),
            (QuantFormat.BF16, QuantFormat.BF16),
            (QuantFormat.FP8PerTensor, QuantFormat.FP8PerTensor),
            (QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8),
            (QuantFormat.MXFP8, QuantFormat.MXFP8),
        )
        assert PrimsTsRunner.supported_activation_classes_by_quant == {
            (QuantFormat.NVFP4, QuantFormat.NVFP4): (
                SwiGLU,
                GeGLU,
                SiTU,
                ReLU2,
                SiLU,
            ),
            (QuantFormat.MXFP4, QuantFormat.MXFP8): (
                SwiGLU,
                GeGLU,
                SiTU,
                ReLU2,
                SiLU,
            ),
            (QuantFormat.MXFP4, QuantFormat.BF16): (SwiGLU,),
            (QuantFormat.BF16, QuantFormat.BF16): (SwiGLU, ReLU2, SiLU),
            (QuantFormat.FP8PerTensor, QuantFormat.FP8PerTensor): (
                SwiGLU,
                ReLU2,
                SiLU,
            ),
            (QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8): (SwiGLU,),
            (QuantFormat.MXFP8, QuantFormat.MXFP8): (
                SwiGLU,
                GeGLU,
                ReLU2,
                SiLU,
            ),
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
        with pytest.raises(ValueError, match="does not support"):
            PrimsTsConfig.prepare_weights(
                torch.empty(0),
                torch.empty(0),
                quant=QuantConfig(
                    weight=QuantFormat.MXINT4, activation=QuantFormat.BF16
                ),
                num_local_experts=1,
                hidden_size=128,
                intermediate_size=128,
            )
        with pytest.raises(ValueError, match="hidden_states_scale_global"):
            PrimsTsConfig.prepare_weights(
                torch.empty(0),
                torch.empty(0),
                quant=QuantConfig(
                    weight=QuantFormat.FP8PerTensor,
                    activation=QuantFormat.FP8PerTensor,
                ),
                num_local_experts=1,
                hidden_size=128,
                intermediate_size=128,
            )

    def test_prepare_activations_rejects_unsupported_pairs(self):
        with pytest.raises(ValueError, match="does not support"):
            PrimsTsConfig.prepare_activations(
                torch.empty(1, 128, dtype=torch.bfloat16),
                quant=QuantConfig(
                    weight=QuantFormat.MXINT4, activation=QuantFormat.BF16
                ),
            )
        with pytest.raises(ValueError, match="hidden_states_scale_global"):
            PrimsTsConfig.prepare_activations(
                torch.empty(1, 128, dtype=torch.bfloat16),
                quant=QuantConfig(
                    weight=QuantFormat.FP8PerTensor,
                    activation=QuantFormat.FP8PerTensor,
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

    def test_bf16_rejects_per_token_scale_config(self):
        runner = self._runner(
            _config(variant=dataclasses.replace(_BF16, per_token_scale=True))
        )
        with pytest.raises(NotImplementedError, match="per-token scale"):
            runner.check_support()

    @pytest.mark.parametrize("variant", [_BF16, _NVFP4], ids=["bf16", "nvfp4"])
    def test_pack_rejects_per_token_scale_when_not_configured(self, variant):
        runner = self._runner(_config(variant=variant))
        runner._built = True
        runner._per_token = False
        hidden = torch.zeros(4, 8, dtype=torch.bfloat16)
        act = MoEActivationPack(
            hidden,
            None,
            torch.zeros(4, 2, dtype=torch.int32),
            torch.ones(4, 2, dtype=torch.float32),
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
            per_token_scale=torch.ones(4, dtype=torch.float32),
        )
        with pytest.raises(ValueError, match="per_token_scale"):
            runner.pack_inputs(act, weights=None)

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

    @pytest.mark.parametrize(
        "method,activation",
        [
            (RoutingMethodType.Sigmoid, SwiGLU()),
            (RoutingMethodType.DeepSeekV3, ReLU2()),
        ],
        ids=["sigmoid", "deepseekv3-relu2"],
    )
    def test_fp8_per_tensor_rejects_inner_routing_limits(self, method, activation):
        runner = self._runner(
            _config(
                variant=QuantConfig(
                    weight=QuantFormat.FP8PerTensor,
                    activation=QuantFormat.FP8PerTensor,
                ),
                routing=RoutingConfig(num_experts=32, top_k=2, method=method),
                activation=activation,
            )
        )
        with pytest.raises(NotImplementedError, match=method.name):
            runner.check_support()

    def test_fp8_per_tensor_llama4_requires_from_logits(self):
        runner = self._runner(
            _config(
                variant=QuantConfig(
                    weight=QuantFormat.FP8PerTensor,
                    activation=QuantFormat.FP8PerTensor,
                ),
                routing=RoutingConfig(
                    num_experts=32, top_k=1, method=RoutingMethodType.Llama4
                ),
            )
        )
        runner._built = True
        runner._pair = runner.config.quant.pair
        hidden = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)
        act = MoEActivationPack(
            hidden,
            None,
            torch.zeros(4, 1, dtype=torch.int32),
            torch.ones(4, 1, dtype=torch.float32),
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )
        with pytest.raises(NotImplementedError, match="FromLogits"):
            runner.pack_inputs(act, weights=None)

    def test_deepseek_rejects_nondefault_swiglu(self):
        runner = self._runner(
            _config(
                variant=QuantConfig(
                    weight=QuantFormat.DeepSeekFp8,
                    activation=QuantFormat.DeepSeekFp8,
                ),
                activation=SwiGLU(alpha=1.5),
            )
        )
        with pytest.raises(NotImplementedError, match="non-default"):
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

    def test_layer_writes_the_packs_output_buffer(self):
        """A pack's ``output`` buffer receives the finalized output in place."""
        act, weights, config, _tensors = _make_bf16_packs_and_config(
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
        layer = MoELayer(config)
        expected = layer(act, weights)
        buffer = torch.full_like(expected, float("nan"))
        out = layer(dataclasses.replace(act, output=buffer), weights)
        assert out.data_ptr() == buffer.data_ptr()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        rows, hidden = buffer.shape
        for invalid in (
            buffer[:, :128],
            buffer.new_empty((rows, 2 * hidden))[:, ::2],
            buffer.cpu(),
            buffer.new_empty(rows * hidden + 1)[1:].view(rows, hidden),
        ):
            with pytest.raises(ValueError, match="output must be a contiguous"):
                layer(dataclasses.replace(act, output=invalid), weights)

    def test_tuning_pre_hook_binds_each_packs_static_extras(self):
        """Two packs on one runner: each pack's tuning pre-hook installs its own
        inner static extras, so tactic enumeration never reads the other's."""
        shape = dict(
            hidden_size=256,
            intermediate_size=256,
            num_experts=8,
            top_k=2,
            max_tokens=16,
        )
        packed_act, weights, config, _ = _make_bf16_packs_and_config(16, **shape)
        unpacked_act, _, _, _ = _make_bf16_packs_and_config(
            16, routing_input_mode=RoutingInputMode.UnpackedPrecomputed, **shape
        )
        _, weights, config = _with_prims_ts_backend(
            packed_act, weights, config, "trtllm_bf16_routed"
        )
        runner = _build_direct_runner(
            PrimsTsRunner, config, packed_act.hidden_states_q.device
        )
        packed = runner.pack_inputs(packed_act, weights)
        unpacked = runner.pack_inputs(unpacked_act, weights)

        def mode_after_pre_hook(inputs):
            hook = runner.tuning_config_for(inputs).inputs_pre_hook
            hook([t.clone() if t is not None else None for t in inputs])
            return dict(runner._inner._cache_key_static_extras)["routing_input_mode"]

        assert mode_after_pre_hook(packed) == int(RoutingInputMode.PackedPrecomputed)
        assert mode_after_pre_hook(unpacked) == int(
            RoutingInputMode.UnpackedPrecomputed
        )
        assert mode_after_pre_hook(packed) == int(RoutingInputMode.PackedPrecomputed)

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

        mxfp4 = QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
        mx_w1 = torch.randn(4, 512, 256, device=device, dtype=torch.bfloat16)
        mx_w2 = torch.randn(4, 256, 256, device=device, dtype=torch.bfloat16)
        mx_kwargs = dict(
            quant=mxfp4,
            num_local_experts=4,
            hidden_size=256,
            intermediate_size=256,
            activation=SwiGLU(),
            device=device,
        )
        prims_mx = PrimsTsConfig.prepare_weights(mx_w1, mx_w2, **mx_kwargs)
        trtllm_mx = TrtllmFp4Config.prepare_weights(mx_w1, mx_w2, **mx_kwargs)
        assert prims_mx.keys() == trtllm_mx.keys()
        for key in ("gemm1_weights", "gemm2_weights"):
            # BlockMajorK [experts, K_bytes / 64, Mn, 64] of the MajorK weights.
            num_experts, mn, k_bytes = trtllm_mx[key].shape
            assert prims_mx[key].shape == (num_experts, k_bytes // 64, mn, 64)

    @pytest.mark.parametrize("num_tokens", [6, 300])
    def test_mxfp4_mxfp8_local_expert_subset_matches_trtllm(
        self, num_tokens, monkeypatch
    ):
        """Tokens with no, one and two local experts.

        Two of sixteen experts per token over four local experts average under
        one local expert per token, so the FC2 epilogue stores the
        single-expert tokens and the finalize zeroes and reduces the rest.
        """
        from flashinfer.prims_ts.moe import runner as prims_runner

        fused = []
        apply_fused_finalize = prims_runner.apply_fused_finalize

        def recording_apply(pair, **kwargs):
            pair = apply_fused_finalize(pair, **kwargs)
            fused.append(pair.fc2.cfg.build().fused_finalize_top_k)
            return pair

        monkeypatch.setattr(prims_runner, "apply_fused_finalize", recording_apply)

        torch.manual_seed(0)
        device = torch.device("cuda", torch.cuda.current_device())
        num_experts, local, offset, top_k = 16, 4, 4, 2
        hidden, intermediate = 256, 256
        quant = QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
        x = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
        w1 = torch.randn(
            local, intermediate * 2, hidden, device=device, dtype=torch.bfloat16
        )
        w2 = torch.randn(
            local, hidden, intermediate, device=device, dtype=torch.bfloat16
        )
        topk_ids = torch.stack(
            [
                torch.randperm(num_experts, device=device)[:top_k]
                for _ in range(num_tokens)
            ]
        ).to(torch.int32)
        # Local experts are 4..7: no, one and two local experts.
        topk_ids[:3] = torch.tensor(
            [[0, 9], [5, 12], [4, 7]], device=device, dtype=torch.int32
        )
        routing_weights = torch.softmax(
            torch.randn(num_tokens, top_k, device=device), dim=-1
        ).to(torch.bfloat16)

        x_q, x_scale = PrimsTsConfig.prepare_activations(x, quant=quant)
        view_kwargs = dict(
            quant=quant,
            num_local_experts=local,
            hidden_size=hidden,
            intermediate_size=intermediate,
            activation=SwiGLU(),
            device=device,
        )
        prims_weights = MoEWeightPack()
        prims_weights.prepare_for(
            "prims_ts",
            PrimsTsConfig.prepare_weights(w1.clone(), w2.clone(), **view_kwargs),
        )
        trtllm_weights = MoEWeightPack()
        trtllm_weights.prepare_for(
            "trtllm_fp4_routed",
            TrtllmFp4Config.prepare_weights(w1.clone(), w2.clone(), **view_kwargs),
        )
        act = MoEActivationPack(
            x_q,
            x_scale,
            topk_ids,
            routing_weights,
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        )
        config = MoEConfig(
            routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
            quant=quant,
            experts=ExpertConfig(
                intermediate_size=intermediate,
                local_expert_offset=offset,
                local_num_experts=local,
            ),
            activation=SwiGLU(),
            execution=ExecutionConfig(tune_max_num_tokens=num_tokens),
        )
        prims_layer = MoELayer(
            dataclasses.replace(config, backend=BackendOptions((PrimsTsConfig(),)))
        )
        prims_out = prims_layer(act, prims_weights)
        trtllm_out = MoELayer(
            dataclasses.replace(config, backend=BackendOptions((TrtllmFp4Config(),)))
        )(act, trtllm_weights)

        assert fused and all(k == top_k for k in fused)
        _nvfp4_check(prims_out, trtllm_out.float(), "prims_ts local expert subset")
        is_local = (topk_ids >= offset) & (topk_ids < offset + local)
        no_local = ~is_local.any(dim=1)
        assert torch.equal(prims_out[no_local], torch.zeros_like(prims_out[no_local]))

        # The same routing weights in FP32, through the same layer and tactics.
        fp32_out = prims_layer(
            dataclasses.replace(act, topk_weights=routing_weights.float()),
            prims_weights,
        )
        assert torch.equal(fp32_out, prims_out)

        # Without zero rows, the rows of tokens without a local expert keep what
        # the output buffer held, and every other row is the same.
        buffer = torch.full_like(prims_out, float("nan"))
        kept = prims_layer(
            dataclasses.replace(
                act, output=buffer, zero_rows_without_local_expert=False
            ),
            prims_weights,
        )
        assert kept.data_ptr() == buffer.data_ptr()
        assert torch.isnan(kept[no_local]).all()
        assert torch.equal(kept[~no_local], prims_out[~no_local])

    def test_mxfp4_mxfp8_fused_finalize_two_cta_fc2_past_hidden(self, monkeypatch):
        """A two-CTA FC2 tile that ends past the hidden size.

        Hidden size 384 is three 128-row tiles, so the second CTA of the last
        two-CTA tile covers rows 384..511. The fused epilogue stores only the
        rows below the hidden size; with unit routing weights it matches the
        unfused finalize bit for bit. The tactic is pinned because the
        autotuner need not pick a two-CTA FC2 tile.
        """
        from flashinfer.prims_ts.moe import runner as prims_runner
        from flashinfer.prims_ts.moe.config_mapper import (
            map_trtllm_mxfp4_mxfp8_moe_tactic,
            valid_prims_ts_mxfp4_mxfp8_moe_tactics,
        )

        torch.manual_seed(0)
        device = torch.device("cuda", torch.cuda.current_device())
        num_tokens, num_experts, local, offset, top_k = 300, 16, 4, 4, 2
        hidden, intermediate = 384, 256
        quant = QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
        x = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
        w1 = torch.randn(
            local, intermediate * 2, hidden, device=device, dtype=torch.bfloat16
        )
        w2 = torch.randn(
            local, hidden, intermediate, device=device, dtype=torch.bfloat16
        )
        topk_ids = torch.stack(
            [
                torch.randperm(num_experts, device=device)[:top_k]
                for _ in range(num_tokens)
            ]
        ).to(torch.int32)

        x_q, x_scale = PrimsTsConfig.prepare_activations(x, quant=quant)
        weights = MoEWeightPack()
        weights.prepare_for(
            "prims_ts",
            PrimsTsConfig.prepare_weights(
                w1,
                w2,
                quant=quant,
                num_local_experts=local,
                hidden_size=hidden,
                intermediate_size=intermediate,
                activation=SwiGLU(),
                device=device,
            ),
        )
        act = MoEActivationPack(
            x_q,
            x_scale,
            topk_ids,
            torch.ones(num_tokens, top_k, device=device, dtype=torch.bfloat16),
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        )
        config = MoEConfig(
            routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
            quant=quant,
            experts=ExpertConfig(
                intermediate_size=intermediate,
                local_expert_offset=offset,
                local_num_experts=local,
            ),
            activation=SwiGLU(),
            execution=ExecutionConfig(tune_max_num_tokens=num_tokens),
            backend=BackendOptions((PrimsTsConfig(),)),
        )
        runner = _build_direct_runner(PrimsTsRunner, config, device)
        inputs = runner.pack_inputs(act, weights)

        major, minor = get_compute_capability(device)
        tactic_kwargs = dict(
            sm_arch=major * 10 + minor,
            num_tokens=num_tokens,
            top_k=top_k,
            num_local_experts=local,
            num_experts=num_experts,
            weight_layout=int(WeightLayout.BlockMajorK),
        )
        two_cta = [
            tactic
            for tactic in valid_prims_ts_mxfp4_mxfp8_moe_tactics(**tactic_kwargs)
            if int(
                map_trtllm_mxfp4_mxfp8_moe_tactic(
                    tactic, **tactic_kwargs
                ).fc2.cfg.kwargs.get("umma_cta_count", 1)
            )
            == 2
        ]
        assert two_cta

        fused = []
        apply_fused_finalize = prims_runner.apply_fused_finalize

        def recording_apply(pair, **kwargs):
            pair = apply_fused_finalize(pair, **kwargs)
            fused.append(pair.fc2.cfg.build().fused_finalize_top_k)
            return pair

        def forward(apply):
            with monkeypatch.context() as patch:
                patch.setattr(prims_runner, "apply_fused_finalize", apply)
                return runner.forward(
                    inputs, tactic=two_cta[0], **runner.launch_kwargs_for(inputs)
                ).clone()

        fused_out = forward(recording_apply)
        plain_out = forward(lambda pair, **kwargs: pair)
        assert fused and all(k == top_k for k in fused)
        assert torch.equal(fused_out, plain_out)

    @pytest.mark.parametrize(
        "quant",
        (
            _NVFP4,
            QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
            _BF16,
            QuantConfig(
                weight=QuantFormat.FP8PerTensor,
                activation=QuantFormat.FP8PerTensor,
            ),
            QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8),
        ),
        ids=("nvfp4", "mxfp4-mxfp8", "bf16", "fp8-per-tensor", "mxfp8"),
    )
    def test_silu_supported_quant_pairs_end_to_end(self, quant):
        torch.manual_seed(0)
        device = torch.device("cuda", torch.cuda.current_device())
        num_experts, hidden, intermediate, tokens, top_k = 4, 256, 256, 8, 2
        activation = SiLU()
        x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16) / 2
        w1 = (
            torch.randn(
                num_experts,
                intermediate,
                hidden,
                device=device,
                dtype=torch.bfloat16,
            )
            / hidden**0.5
        )
        w2 = (
            torch.randn(
                num_experts,
                hidden,
                intermediate,
                device=device,
                dtype=torch.bfloat16,
            )
            / intermediate**0.5
        )
        routing_weights, topk_ids = torch.topk(
            torch.softmax(torch.randn(tokens, num_experts, device=device), dim=-1),
            top_k,
            dim=-1,
        )
        topk_ids = topk_ids.to(torch.int32)
        routing_weights = routing_weights.to(torch.bfloat16)
        reference = _silu_reference(x, w1, w2, topk_ids, routing_weights)

        prepare_kwargs: dict = {}
        if quant.pair == (QuantFormat.FP8PerTensor, QuantFormat.FP8PerTensor):
            prepare_kwargs = {
                "hidden_states_scale_global": fp8_per_tensor_global_scale(x),
                "intermediate_scale_global": torch.tensor(32.0, device=device),
            }
        act_kwargs = {
            key: value
            for key, value in prepare_kwargs.items()
            if key == "hidden_states_scale_global"
        }
        x_q, x_scale = PrimsTsConfig.prepare_activations(x, quant=quant, **act_kwargs)
        prims_view = PrimsTsConfig.prepare_weights(
            w1.clone(),
            w2.clone(),
            quant=quant,
            num_local_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            activation=activation,
            device=device,
            **prepare_kwargs,
        )

        act = MoEActivationPack(
            x_q,
            x_scale,
            topk_ids,
            routing_weights,
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )
        weights = MoEWeightPack()
        weights.prepare_for("prims_ts", prims_view)
        config = MoEConfig(
            routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
            quant=quant,
            experts=ExpertConfig(intermediate_size=intermediate),
            activation=activation,
            backend=BackendOptions((PrimsTsConfig(),)),
            execution=ExecutionConfig(tune_max_num_tokens=tokens),
        )
        layer = MoELayer(config)
        actual = layer(act, weights)

        assert layer.winner_backend == "prims_ts"
        if quant == _BF16:
            _bf16_check(actual, reference, "prims_ts bf16 silu")
        else:
            _nvfp4_check(actual, reference, f"prims_ts {quant.pair} silu")

    @pytest.mark.parametrize(
        "quant",
        (
            QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
            QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.BF16),
            QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8),
            QuantConfig(
                weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8
            ),
            QuantConfig(
                weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
            ),
        ),
    )
    def test_added_quant_matches_trtllm(self, quant):
        if quant.pair == (QuantFormat.MXFP4, QuantFormat.BF16) and (
            get_compute_capability(torch.device("cuda")) == (10, 3)
        ):
            pytest.xfail("TRTLLM MXFP4×BF16 is disabled on SM103")

        torch.manual_seed(0)
        device = torch.device("cuda", torch.cuda.current_device())
        num_experts, hidden, intermediate, tokens, top_k = 4, 256, 256, 8, 2
        x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
        w1 = torch.randn(
            num_experts, intermediate * 2, hidden, device=device, dtype=torch.bfloat16
        )
        w2 = torch.randn(
            num_experts, hidden, intermediate, device=device, dtype=torch.bfloat16
        )
        routing_weights, topk_ids = torch.topk(
            torch.softmax(torch.randn(tokens, num_experts, device=device), dim=-1),
            top_k,
            dim=-1,
        )
        topk_ids = topk_ids.to(torch.int32)
        routing_weights = routing_weights.to(torch.bfloat16)

        prepare_kwargs: dict = {}
        if quant.pair == (QuantFormat.FP8PerTensor, QuantFormat.FP8PerTensor):
            prepare_kwargs["hidden_states_scale_global"] = fp8_per_tensor_global_scale(
                x
            )
            prepare_kwargs["intermediate_scale_global"] = torch.tensor(
                64.0, device=device
            )
            trtllm_config = TrtllmFp8PerTensorConfig()
            trtllm_key = "trtllm_fp8_per_tensor"
        elif quant.pair[0] is QuantFormat.MXFP4:
            trtllm_config = TrtllmFp4Config()
            trtllm_key = "trtllm_fp4_routed"
        else:
            trtllm_config = TrtllmFp8BlockConfig()
            trtllm_key = "trtllm_fp8_block"

        act_kwargs = {
            key: value
            for key, value in prepare_kwargs.items()
            if key == "hidden_states_scale_global"
        }
        x_q, x_scale = PrimsTsConfig.prepare_activations(x, quant=quant, **act_kwargs)
        prims_view = PrimsTsConfig.prepare_weights(
            w1.clone(),
            w2.clone(),
            quant=quant,
            num_local_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            activation=SwiGLU(),
            device=device,
            **prepare_kwargs,
        )
        trtllm_prepare = (
            dict(
                hidden_states_scale_global=prepare_kwargs["hidden_states_scale_global"],
                intermediate_scale_global=prepare_kwargs["intermediate_scale_global"],
            )
            if trtllm_key == "trtllm_fp8_per_tensor"
            else {}
        )
        if trtllm_key == "trtllm_fp4_routed":
            trtllm_view = TrtllmFp4Config.prepare_weights(
                w1.clone(),
                w2.clone(),
                quant=quant,
                num_local_experts=num_experts,
                hidden_size=hidden,
                intermediate_size=intermediate,
                activation=SwiGLU(),
                device=device,
            )
        elif trtllm_key == "trtllm_fp8_per_tensor":
            trtllm_view = TrtllmFp8PerTensorConfig.prepare_weights(
                w1.clone(),
                w2.clone(),
                num_local_experts=num_experts,
                hidden_size=hidden,
                intermediate_size=intermediate,
                activation=SwiGLU(),
                device=device,
                **trtllm_prepare,
            )
        else:
            trtllm_view = TrtllmFp8BlockConfig.prepare_weights(
                w1.clone(),
                w2.clone(),
                quant=quant,
                num_local_experts=num_experts,
                hidden_size=hidden,
                intermediate_size=intermediate,
                activation=SwiGLU(),
                device=device,
            )
        if quant.pair == (QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8):
            # Both GEMM payloads are shuffled; both block scales stay TRT-LLM.
            for name in ("gemm1", "gemm2"):
                assert not torch.equal(
                    prims_view[f"{name}_weights"].view(torch.uint8),
                    trtllm_view[f"{name}_weights"].view(torch.uint8),
                )
                torch.testing.assert_close(
                    prims_view[f"{name}_weights_scale"],
                    trtllm_view[f"{name}_weights_scale"],
                )
        else:
            assert prims_view.keys() == trtllm_view.keys()
            for key, tensor in prims_view.items():
                other = trtllm_view[key]
                if tensor.ndim == 4 and other.ndim == 3:
                    # BlockMajorK MXFP4 weights, [experts, K_bytes / block, Mn, block].
                    tensor = tensor.permute(0, 2, 1, 3).reshape(other.shape)
                assert tensor.dtype == other.dtype and tensor.shape == other.shape
                # MXFP4 stores UE8M0 bytes in float8_e4m3fn. assert_close
                # rejects those encodings even when the bytes match. 0-dim
                # calibration scalars in the per-tensor view cannot be viewed
                # as uint8.
                if tensor.ndim == 0:
                    assert torch.equal(tensor, other)
                else:
                    assert torch.equal(
                        tensor.contiguous().view(torch.uint8),
                        other.contiguous().view(torch.uint8),
                    )

        act = MoEActivationPack(
            x_q,
            x_scale,
            topk_ids,
            routing_weights,
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )
        config = MoEConfig(
            routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
            quant=quant,
            experts=ExpertConfig(intermediate_size=intermediate),
            activation=SwiGLU(),
            execution=ExecutionConfig(tune_max_num_tokens=tokens),
        )
        prims_weights = MoEWeightPack()
        prims_weights.prepare_for("prims_ts", prims_view)
        trtllm_weights = MoEWeightPack()
        trtllm_weights.prepare_for(trtllm_key, trtllm_view)
        prims_out = MoELayer(
            dataclasses.replace(config, backend=BackendOptions((PrimsTsConfig(),)))
        )(act, prims_weights)
        trtllm_out = MoELayer(
            dataclasses.replace(config, backend=BackendOptions((trtllm_config,)))
        )(act, trtllm_weights)
        _nvfp4_check(prims_out, trtllm_out.float(), f"prims_ts {quant.pair} vs trtllm")
        if quant.pair == (QuantFormat.MXFP4, QuantFormat.MXFP8):
            # Prims-TS also takes the MajorK weights of the TRT-LLM view.
            major_k_weights = MoEWeightPack()
            major_k_weights.prepare_for("prims_ts", trtllm_view)
            major_k_out = MoELayer(
                dataclasses.replace(config, backend=BackendOptions((PrimsTsConfig(),)))
            )(act, major_k_weights)
            assert torch.equal(major_k_out, prims_out)
