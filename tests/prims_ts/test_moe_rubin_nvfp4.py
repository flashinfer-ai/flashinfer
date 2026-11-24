# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from flashinfer.tllm_enums import ActivationType, WeightLayout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rubin_unified_moe_shared_weights(monkeypatch):
    """One layer can use either weight pack across decode and native prefill."""
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Native Batch-M MoE requires Rubin")
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe import MoELayer, MoEWeightPack, PrimsTsConfig
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEActivationPack,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        RoutingInputMode,
    )
    from flashinfer.prims_ts.moe.runner import PrimsTsNvfp4MoERunner
    from flashinfer.tllm_enums import RoutingMethodType

    torch.manual_seed(31)
    experts, hidden, intermediate, top_k = 8, 1024, 512, 2
    quant = QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4)
    backend = PrimsTsConfig()
    w1 = (
        torch.randn(
            experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16
        )
        / hidden**0.5
    )
    w2 = (
        torch.randn(experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16)
        / intermediate**0.5
    )
    weights = {}
    for layout in (WeightLayout.MajorK, WeightLayout.BlockMajorK):
        view = backend.prepare_weights(
            w1,
            w2,
            quant=quant,
            num_local_experts=experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            nvfp4_weight_layout=layout.name,
        )
        weights[layout] = MoEWeightPack({"prims_ts": view})
    config = MoEConfig(
        routing=RoutingConfig(
            num_experts=experts, top_k=top_k, method=RoutingMethodType.Renormalize
        ),
        quant=quant,
        experts=ExpertConfig(intermediate_size=intermediate),
        backend=BackendOptions((backend,)),
        execution=ExecutionConfig(tune_max_num_tokens=4096, enable_pdl=True),
    )
    selected = [-1, -1]
    monkeypatch.setattr(
        AutoTuner.get(),
        "choose_one",
        lambda custom_op, runners, tuning_config, inputs, **kw: (runners[0], selected),
    )
    forward = PrimsTsNvfp4MoERunner.forward

    def track(runner, inputs, tactic=-1, **kwargs):
        view = weights[runner.weight_layout].get_view("prims_ts")
        for name in (
            "gemm1_weights",
            "gemm1_weights_scale",
            "gemm2_weights",
            "gemm2_weights_scale",
        ):
            assert kwargs[name].data_ptr() == view[name].data_ptr()
        return forward(runner, inputs, tactic=tactic, **kwargs)

    monkeypatch.setattr(PrimsTsNvfp4MoERunner, "forward", track)
    for tokens in (32, 4096):
        x, sf = backend.prepare_activations(
            torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16),
            quant=quant,
        )
        act = MoEActivationPack(
            hidden_states_q=x,
            hidden_states_scale=sf,
            routing_input_mode=RoutingInputMode.FromLogits,
            routing_logits=torch.randn(
                tokens, experts, device="cuda", dtype=torch.float32
            ),
        )
        selected = [-1, -1]
        layer = MoELayer(config)
        baseline = layer(act, weights[WeightLayout.MajorK]).clone()
        blocked = layer(act, weights[WeightLayout.BlockMajorK])
        assert (
            blocked.float() - baseline.float()
        ).norm() / baseline.float().norm() < 0.01
        if tokens == 32:
            continue
        for tactic in ([512, 0], [512, 1], [1024, 0]):
            selected = tactic
            layer = MoELayer(config)
            captures = []
            for pack in weights.values():
                actual = layer(act, pack)
                assert torch.isfinite(actual).all()
                assert (actual - baseline).norm() / baseline.norm() < 0.12
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = layer(act, pack)
                captures.append((graph, captured, actual))
            for _ in range(5):
                for graph, _, _ in captures:
                    graph.replay()
            torch.cuda.synchronize()
            for _, captured, actual in captures:
                assert torch.isfinite(captured).all()
                assert (
                    captured.float() - actual.float()
                ).norm() / actual.float().norm() < 0.01


@pytest.mark.parametrize("layout", (WeightLayout.MajorK, WeightLayout.BlockMajorK))
def test_rubin_nvfp4_batch_m_candidates(layout):
    from flashinfer.prims_ts.moe.config_mapper import (
        map_trtllm_nvfp4_moe_tactic,
        valid_prims_ts_nvfp4_moe_tactics,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        validate_config,
    )

    shape = dict(num_tokens=2048, top_k=2, num_local_experts=8)
    tactics = valid_prims_ts_nvfp4_moe_tactics(
        sm_arch=107, weight_layout=int(layout), **shape
    )
    native = [t for t in tactics if t[0] > 256]
    assert native == [[512, 0], [512, 1], [1024, 0]]
    for tactic in native:
        pair = map_trtllm_nvfp4_moe_tactic(
            tactic, sm_arch=107, weight_layout=int(layout), enable_pdl=True
        )
        for spec in (pair.fc1, pair.fc2):
            cfg = spec.cfg.build()
            validate_config(cfg, problem_mnk=(4096, 1024, 512))
            assert not cfg.is_swap_ab
            assert cfg.use_shuffled_matrix
            assert cfg.gated_weight_group_size == 1
            assert cfg.tile_token == tactic[0]
            assert cfg.use_global_scales
            assert cfg.use_early_exit
            assert not cfg.use_pdl
        assert pair.fc1.cfg.build().dense_preload_store_prefix_fragments == 0
        assert pair.fc2.cfg.build().dense_preload_store_prefix_fragments == 2
    for extra in (
        {"sm_arch": 100},
        {"sm_arch": 107, "fc1_has_bias": True},
        {"sm_arch": 107, "use_per_token_sf_b": True},
        {"sm_arch": 107, "activation_type": int(ActivationType.Situ)},
    ):
        assert all(
            t[0] <= 256 for t in valid_prims_ts_nvfp4_moe_tactics(**shape, **extra)
        )
    default = map_trtllm_nvfp4_moe_tactic([-1, -1], sm_arch=107, **shape)
    assert default.fc1.cfg.build().is_swap_ab
    # The B/SFB task excludes CTAs that neither issue nor arm its pipeline.
    wide = map_trtllm_nvfp4_moe_tactic([1024, 0], sm_arch=107)
    assert wide.fc1.cfg.build().num_stages_workid == 1
    assert wide.fc2.cfg.build().num_stages_workid == 1


@pytest.mark.parametrize("tile", (512, 1024))
@pytest.mark.parametrize(
    "hidden_size,intermediate_size,layout,error",
    [
        (
            896,
            512,
            WeightLayout.MajorK,
            "activation scale loads require problem K.*got 896",
        ),
        (
            1024,
            448,
            WeightLayout.MajorK,
            "weight scale indexing requires problem N.*got 896",
        ),
        (
            1024,
            384,
            WeightLayout.BlockMajorK,
            "BlockMajorK requires problem K.*problem_k=384",
        ),
    ],
)
def test_rubin_nvfp4_batch_m_rejects_unaligned_geometry(
    tile, hidden_size, intermediate_size, layout, error
):
    from flashinfer.prims_ts.moe.config_mapper import map_trtllm_nvfp4_moe_tactic
    from flashinfer.prims_ts.moe.runner import PrimsTsNvfp4MoERunner

    runner = PrimsTsNvfp4MoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        weight_layout=layout,
    )
    pair = map_trtllm_nvfp4_moe_tactic(
        [tile, 0], sm_arch=107, weight_layout=int(layout)
    )
    with pytest.raises(ValueError, match=error):
        runner._check_cga_tactic(pair.fc1.cfg.build(), pair.fc2.cfg.build())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "num_tokens,hidden_size,intermediate_size,num_experts,top_k,tactic",
    [
        pytest.param(tokens, 1024, 512, 8, 2, tactic, id=f"t{tokens}-{name}")
        for tokens in (32, 1152)
        for tactic, name in (
            ([512, 0], "2x2"),
            ([512, 1], "2x4"),
            ([1024, 0], "4x2"),
            (None, "autotune"),
        )
    ],
)
@pytest.mark.parametrize("layout", (WeightLayout.MajorK, WeightLayout.BlockMajorK))
def test_rubin_nvfp4_batch_m_block(
    monkeypatch,
    tactic,
    layout,
    num_tokens,
    hidden_size,
    intermediate_size,
    num_experts,
    top_k,
):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Native Batch-M MoE requires Rubin")
    if tactic is None and num_tokens == 32:
        pytest.skip("Native Batch-M candidates target prefill token counts")
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.prims_ts.moe.runner import PrimsTsNvfp4MoERunner
    from tests.moe.trtllm_gen_fused_moe_utils import (
        FP4Moe,
        MoeGemmBackend,
        QuantMode,
        RoutingMethodType,
        run_moe_test,
    )

    # Use the public MoE API and normal weight preparation. The shared helper
    # compares against dequantized PyTorch MoE and captures/replays a CUDA graph.
    actual_choose = AutoTuner.get().choose_one
    profiled = set()
    if tactic is None:
        # Bound this integration test to the existing default plus new rows.
        # Candidate generation and timing still run through the real tuner.
        get_valid = PrimsTsNvfp4MoERunner.get_valid_tactics

        def candidates(runner, inputs, profile):
            return [
                t for t in get_valid(runner, inputs, profile) if t == -1 or t[0] > 256
            ]

        monkeypatch.setattr(PrimsTsNvfp4MoERunner, "get_valid_tactics", candidates)
        forward = PrimsTsNvfp4MoERunner.forward

        def track_forward(runner, inputs, tactic=-1, **kwargs):
            result = forward(runner, inputs, tactic=tactic, **kwargs)
            if tactic != -1:
                profiled.add(tuple(tactic))
            return result

        monkeypatch.setattr(PrimsTsNvfp4MoERunner, "forward", track_forward)

    selected = []
    baseline = []
    weight_ptrs = []

    def choose_one(name, runners, config, inputs, **kwargs):
        assert "prims_ts" in name
        from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs

        # Decode and prefill receive exactly the same prepared weight tensors.
        weight_ptrs.append(
            tuple(
                kwargs[key].data_ptr()
                for key in (
                    "gemm1_weights",
                    "gemm1_weights_scale",
                    "gemm2_weights",
                    "gemm2_weights_scale",
                )
            )
        )
        if not baseline:
            assert not torch.cuda.is_current_stream_capturing()
            runners[0].forward(inputs, tactic=[-1, -1], **kwargs)
            baseline.append(MoeRunnerInputs.from_list(inputs).output.clone())
        if tactic is None:
            # The shared numerical helper disables Prims-TS autotuning. Enable
            # it explicitly during warmup, then exercise its cache in capture.
            with autotune(
                not torch.cuda.is_current_stream_capturing(),
                tuning_buckets=[num_tokens],
            ):
                runner, chosen = actual_choose(name, runners, config, inputs, **kwargs)
            selected.append([-1, -1] if chosen == -1 else chosen)
            return runner, chosen
        selected.append(tactic)
        return runners[0], tactic

    monkeypatch.setattr(AutoTuner.get(), "choose_one", choose_one)
    reference, actual, _ = run_moe_test(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_impl=FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4),
        routing_config={
            "num_experts": num_experts,
            "top_k": top_k,
            "padding": 8,
            "n_groups": None,
            "top_k_groups": None,
            "routed_scaling": None,
            "has_routing_bias": False,
            "routing_method_type": RoutingMethodType.Renormalize,
            "compatible_moe_impls": [FP4Moe],
            "compatible_intermediate_size": [intermediate_size],
            "enable_autotune": tactic is None,
            "autotune_tuning_buckets": [num_tokens],
            "tune_max_num_tokens": num_tokens,
        },
        weight_processing={
            "use_shuffled_weight": True,
            "layout": layout,
            "compatible_moe_impls": [FP4Moe],
            "compatible_gemm_backends": [MoeGemmBackend.PRIMS_TS],
        },
        activation_type=ActivationType.Swiglu,
        cache_permute_indices={},
        moe_gemm_backend=MoeGemmBackend.PRIMS_TS,
    )
    assert selected
    if tactic is None:
        assert {(512, index) for index in range(16)} <= profiled
        assert all(t == [-1, -1] or t[0] == 512 and 0 <= t[1] < 16 for t in selected)
        assert len({tuple(t) for t in selected}) == 1
    assert len(set(weight_ptrs)) == 1
    assert (actual - baseline[0]).norm() / baseline[0].norm() < 0.12
    assert torch.isfinite(actual).all()
    assert actual.abs().max() > 0
    # A norm check complements the reference helper's elementwise FP4 tolerance.
    assert (actual - reference).norm() / reference.norm() < 0.12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", (WeightLayout.MajorK, WeightLayout.BlockMajorK))
@pytest.mark.timeout(240)
def test_rubin_nvfp4_one_stage_early_exit_graph(monkeypatch, layout):
    """Reproduce the last-active-tile B/SFB parity race with the full E32 shape.

    Use the established swapAB path as the large-case oracle. Small cases
    above also compare to PyTorch; its CPU FP4 dequantization is prohibitively
    large here, and the shared helper normally skips these SwiGLU dimensions.
    """
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Native Batch-M MoE requires Rubin")
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe import prims_ts_fp4_block_scale_moe
    from flashinfer.fused_moe.api import QuantConfig, QuantFormat, SwiGLU
    from flashinfer.fused_moe.core import convert_to_block_layout
    from flashinfer.fused_moe.prepare import (
        prepare_trtllm_fp4_activations,
        prepare_trtllm_fp4_weights,
    )
    from flashinfer.tllm_enums import RoutingMethodType

    torch.manual_seed(17)
    experts, hidden, intermediate, tokens, top_k = 32, 10240, 4096, 4096, 8
    quant = QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4)
    w1 = (
        torch.randn(
            experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16
        )
        / hidden**0.5
    )
    w2 = (
        torch.randn(experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16)
        / intermediate**0.5
    )
    weights = prepare_trtllm_fp4_weights(
        w1,
        w2,
        quant=quant,
        num_local_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation=SwiGLU(),
    )
    del w1, w2
    if layout == WeightLayout.BlockMajorK:
        for key in ("gemm1_weights", "gemm2_weights"):
            weights[key] = torch.stack(
                [convert_to_block_layout(w, blockK=128) for w in weights[key]]
            )
    weights["output2_scale_scalar"] = torch.linspace(0.7, 1.3, experts, device="cuda")
    x, sf = prepare_trtllm_fp4_activations(
        torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16),
        quant=quant,
    )
    kwargs = dict(
        routing_logits=torch.randn(tokens, experts, device="cuda", dtype=torch.float32),
        routing_bias=None,
        hidden_states=x,
        hidden_states_scale=sf,
        **weights,
        gemm1_bias=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_bias=None,
        num_experts=experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate,
        local_expert_offset=0,
        local_num_experts=experts,
        routed_scaling_factor=None,
        routing_method_type=int(RoutingMethodType.Renormalize),
        weight_layout=int(layout),
        enable_pdl=True,
    )
    tactic = [-1, -1]
    monkeypatch.setattr(
        AutoTuner.get(),
        "choose_one",
        lambda name, runners, config, inputs, **kw: (runners[0], tactic),
    )
    reference = prims_ts_fp4_block_scale_moe(**kwargs)[0].clone()
    tactic = [1024, 0]
    output = prims_ts_fp4_block_scale_moe(**kwargs)[0]
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, rtol=0.01, atol=0.01)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            output = prims_ts_fp4_block_scale_moe(**kwargs)[0]
    for _ in range(50):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=0.01, atol=0.01)
