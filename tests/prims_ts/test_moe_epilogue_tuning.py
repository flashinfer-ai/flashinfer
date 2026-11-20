# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Production tactic admission, resource profiles and replay for epilogue tuning."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.tllm_enums import ActivationType, WeightLayout


@pytest.mark.parametrize("layout", (WeightLayout.MajorK, WeightLayout.BlockMajorK))
def test_shortlist_matches_swept_kernel_configs(layout):
    from flashinfer.prims_ts.batched_gemm.epilogue_search import iter_epilogue_configs
    from flashinfer.prims_ts.moe.compile_cache import stable_config_hash
    from flashinfer.prims_ts.moe.config_mapper import (
        _nvfp4_epilogue_tactics,
        map_trtllm_nvfp4_moe_tactic,
        valid_prims_ts_nvfp4_moe_tactics,
    )
    from flashinfer.prims_ts.moe.epilogue_policy import fused_finalize_config

    options = dict(sm_arch=107, weight_layout=int(layout))
    shape = dict(num_tokens=24576, top_k=6, num_local_experts=384)
    baseline = valid_prims_ts_nvfp4_moe_tactics(**options, **shape)
    tactics = valid_prims_ts_nvfp4_moe_tactics(
        **options, **shape, use_fused_finalize=True
    )
    assert tactics[: len(baseline)] == baseline
    assert [t for t in tactics if t[0] == 512] == [[512, i] for i in range(16)]
    assert [t for t in tactics if t[0] != 512] == [t for t in baseline if t[0] != 512]
    catalog = _nvfp4_epilogue_tactics(
        **options,
        activation_type=int(ActivationType.Swiglu),
        use_per_token_sf_b=False,
        per_token_sf_dtype=1,
    )
    expected = {}
    for base_index in (0, 1):
        pair = map_trtllm_nvfp4_moe_tactic([512, base_index], **options)
        first = pair.fc1.cfg.build()
        second = fused_finalize_config(pair.fc2.cfg.build())
        # Compare to the independent sweep generator's frozen candidate order.
        fc1 = [first, list(iter_epilogue_configs(first))[23]]
        generated = list(iter_epilogue_configs(second))
        fc2 = [second, generated[19], generated[23], generated[27]]
        expected[base_index] = [
            (stable_config_hash(a), stable_config_hash(b)) for a in fc1 for b in fc2
        ]
    for index, (base, policy) in enumerate(catalog):
        pair = map_trtllm_nvfp4_moe_tactic(
            [512, index], **options, use_fused_finalize=True
        )
        assert pair.moe_config_index == index
        first = pair.fc1.cfg.build()
        second = fused_finalize_config(pair.fc2.cfg.build())
        assert (stable_config_hash(first), stable_config_hash(second)) == expected[
            base
        ][policy]
        assert not first.is_swap_ab and not second.is_swap_ab
        assert second.fuse_moe_finalize == 1
        assert (second.num_stages_a, second.num_stages_b, second.cluster_dim_y) == (
            3,
            4,
            4,
        )
        assert fused_finalize_config(second) is second
        if index >= 2:
            with pytest.raises(ValueError, match="config index"):
                map_trtllm_nvfp4_moe_tactic([512, index], **options)


@pytest.mark.parametrize(
    "extra",
    (
        {"sm_arch": 100},
        {"sm_arch": 103},
        {"activation_type": int(ActivationType.Geglu)},
        {"use_per_token_sf_b": True},
        {"fc1_has_bias": True},
        {"fc2_has_bias": True},
    ),
)
def test_shortlist_leaves_other_profiles_unchanged(extra):
    from flashinfer.prims_ts.moe.config_mapper import valid_prims_ts_nvfp4_moe_tactics

    options = dict(sm_arch=107, num_tokens=24576, top_k=6, num_local_experts=384)
    options.update(extra)
    assert valid_prims_ts_nvfp4_moe_tactics(
        **options, use_fused_finalize=True
    ) == valid_prims_ts_nvfp4_moe_tactics(**options)


def test_runner_candidate_cache_tracks_finalize_mode(monkeypatch):
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
    from flashinfer.prims_ts.moe import runner as module
    from flashinfer.prims_ts.moe.epilogue_policy import fused_finalize_requested

    monkeypatch.setattr(module, "_sm_arch_for_tensor", lambda tensor: 107)
    instance = module.PrimsTsNvfp4MoERunner(
        None, top_k=6, num_local_experts=384, hidden_size=7168, intermediate_size=3072
    )
    inputs = MoeRunnerInputs(
        output=torch.empty(24576, 7168, device="meta", dtype=torch.bfloat16),
        routing_logits=torch.empty(24576, 384, device="meta"),
        topk_ids=None,
        expert_weights=None,
        hidden_states=torch.empty(24576, 3584, device="meta", dtype=torch.uint8),
        hidden_states_scale=torch.empty(24576, 448, device="meta", dtype=torch.uint8),
        gemm1_lora_delta=None,
        per_token_scale=None,
    ).to_list()

    def native(**kwargs):
        instance.set_cache_key_static_extras(**kwargs)
        return [
            t
            for t in instance.get_valid_tactics(inputs, None)
            if t != -1 and t[0] == 512
        ]

    assert len(native()) == 16
    fused_key = instance.get_cache_key_extras(inputs)
    assert native(use_fused_finalize=False) == [[512, 0], [512, 1]]
    assert instance.get_cache_key_extras(inputs) != fused_key
    assert native(do_finalize=False) == [[512, 0], [512, 1]]
    assert native(zero_rows_without_local_expert=False) == [[512, 0], [512, 1]]
    assert instance.get_cache_key_extras(inputs) != fused_key
    assert not fused_finalize_requested(
        instance, {"zero_rows_without_local_expert": False}
    )
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        assert native() == [[512, 0], [512, 1]]
        assert instance.get_cache_key_extras(inputs) != fused_key
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    assert len(native()) == 16
    assert not fused_finalize_requested(SimpleNamespace(_is_partitioned=True), {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", ("MajorK", "BlockMajorK"))
def test_shortlist_production_dispatch_replay(monkeypatch, layout):
    """Use mapped tactics without replacing configs or the finalizer selector."""
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Native epilogue tuning requires Rubin")
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe import PrimsTsConfig, prims_ts_fp4_block_scale_moe
    from flashinfer.fused_moe.api import QuantConfig, QuantFormat
    from flashinfer.fused_moe.shared.inputs import RoutingInputMode
    from flashinfer.prims_ts.moe import runner as module
    from flashinfer.tllm_enums import RoutingMethodType

    torch.manual_seed(83)
    tokens, hidden, intermediate, experts, local, offset, top_k = (
        1031,
        1024,
        512,
        8,
        4,
        2,
        2,
    )
    quant = QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4)
    weights = PrimsTsConfig.prepare_weights(
        torch.randn(
            local, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16
        )
        / hidden**0.5,
        torch.randn(local, hidden, intermediate, device="cuda", dtype=torch.bfloat16)
        / intermediate**0.5,
        quant=quant,
        num_local_experts=local,
        hidden_size=hidden,
        intermediate_size=intermediate,
        nvfp4_weight_layout=layout,
    )
    weights["output2_scale_scalar"] = torch.linspace(0.7, 1.3, local, device="cuda")
    x, sf = PrimsTsConfig.prepare_activations(
        torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16), quant=quant
    )
    ids = torch.empty((tokens, top_k), device="cuda", dtype=torch.int32)
    coefficients = torch.empty_like(ids, dtype=torch.bfloat16)
    output = torch.empty((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    kwargs = dict(
        routing_logits=None,
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
        local_expert_offset=offset,
        local_num_experts=local,
        routed_scaling_factor=None,
        routing_method_type=int(RoutingMethodType.Renormalize),
        weight_layout=int(getattr(WeightLayout, layout)),
        enable_pdl=False,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        topk_ids=ids,
        topk_weights=coefficients,
        output=output,
    )
    selected = [512, 0]
    observed = set()
    pointers = set()
    builder = module.build_nvfp4_launch_io

    def inspect_io(**options):
        if options["fc"] == "fc2":
            cfg = options["cfg"]
            observed.add((tuple(selected), bool(cfg.fuse_moe_finalize)))
            if cfg.fuse_moe_finalize:
                assert cfg.dense_epilogue_warps == 4
        pointers.add(
            tuple(
                options[name].data_ptr() for name in ("gemm1_weights", "gemm2_weights")
            )
        )
        return builder(**options)

    def choose(name, runners, config, inputs, **options):
        return runners[0], selected if options["use_fused_finalize"] else [512, 0]

    monkeypatch.setattr(AutoTuner.get(), "choose_one", choose)
    monkeypatch.setattr(module, "build_nvfp4_launch_io", inspect_io)

    def route(step):
        row = torch.arange(tokens, device="cuda", dtype=torch.int32)
        ids[:, 0] = (row + step) % experts
        ids[:, 1] = (row + step + 3) % experts
        coefficients.copy_(torch.rand_like(coefficients) * 1.7 + 0.1)
        weights["output2_scale_scalar"].mul_(1.03)

    def check(reference):
        assert torch.isfinite(output).all()
        assert (
            output.float() - reference.float()
        ).norm() / reference.float().norm() < 0.015

    route(0)
    reference = prims_ts_fp4_block_scale_moe(**kwargs, use_fused_finalize=False)[
        0
    ].clone()
    graphs = []
    for index in range(16):
        selected = [512, index]
        prims_ts_fp4_block_scale_moe(**kwargs)
        check(reference)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prims_ts_fp4_block_scale_moe(**kwargs)
        graphs.append(graph)
    assert all(((512, index), True) in observed for index in range(16))
    assert len(pointers) == 1
    for step in (1, 5):
        route(step)
        reference = prims_ts_fp4_block_scale_moe(**kwargs, use_fused_finalize=False)[
            0
        ].clone()
        for graph in graphs:
            output.fill_(float("nan"))
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            check(reference)
    ids[:, 0] = 0
    ids[:, 1] = 1
    for graph in graphs:
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(output) == 0
