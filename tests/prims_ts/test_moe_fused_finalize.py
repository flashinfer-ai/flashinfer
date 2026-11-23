# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Native Rubin FC2 weighting/scatter, including replay and local expert tails."""

from dataclasses import replace

import pytest
import torch


@pytest.mark.parametrize(
    "warps,prefix,stages", ((8, 0, 1), (8, 2, 1), (8, 0, 2), (8, 2, 2), (4, 0, 2))
)
def test_fused_finalize_config(warps, prefix, stages):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.moe.config_mapper import map_trtllm_nvfp4_moe_tactic

    pair = map_trtllm_nvfp4_moe_tactic([1024, 0], sm_arch=107)
    cfg = replace(
        pair.fc2.cfg.build(),
        fuse_moe_finalize=1,
        dense_epilogue_warps=warps,
        epilogue_regs=248 if warps == 4 else 168,
        dense_preload_store_prefix_fragments=prefix,
        num_stages_c_smem=stages,
        num_stages_a=2,
        num_stages_b=2,
        num_stages_smem_sfa=2,
        num_stages_smem_sfb=2,
    )
    validate_config(cfg, problem_mnk=(8192, 1024, 512))
    for invalid in (
        replace(cfg, dense_epilogue_warps=4, dense_preload_store_prefix_fragments=2),
        replace(cfg, dense_epilogue_warps=4, num_stages_c_smem=1),
        replace(cfg, dense_preload_epilogue=0),
        replace(pair.fc1.cfg.build(), fuse_moe_finalize=1),
        replace(cfg, fuse_moe_finalize=2),
    ):
        with pytest.raises(ValueError):
            validate_config(invalid, problem_mnk=(8192, 1024, 512))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "layout,tactic,prefix,stages,warps,release,ahead",
    (
        ("MajorK", [512, 0], 2, 1, 8, 0, -1),
        ("BlockMajorK", [512, 0], 2, 1, 8, 0, -1),
        ("MajorK", [1024, 0], 2, 1, 8, 0, -1),
        ("BlockMajorK", [1024, 0], 2, 1, 8, 0, -1),
        ("BlockMajorK", [512, 0], 0, 2, 8, 0, -1),
        ("MajorK", [512, 0], 0, 2, 4, 0, -1),
        ("BlockMajorK", [512, 0], 0, 2, 4, 0, -1),
        ("MajorK", [1024, 0], 0, 2, 4, 0, -1),
        ("BlockMajorK", [1024, 0], 0, 2, 4, 0, -1),
        ("BlockMajorK", [512, 0], 0, 2, 8, 0, 0),
        ("BlockMajorK", [512, 0], 2, 1, 8, 2, -1),
        ("BlockMajorK", [512, 0], 0, 2, 4, 2, -1),
    ),
)
def test_fused_finalize_replay(
    monkeypatch,
    layout,
    tactic,
    prefix,
    stages,
    warps,
    release,
    ahead,
    steps=(),
    fc1_overrides=None,
    router_weight_dtype=torch.bfloat16,
):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Native BatchM finalization requires Rubin")
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe import PrimsTsConfig, prims_ts_fp4_block_scale_moe
    from flashinfer.fused_moe.api import QuantConfig, QuantFormat
    from flashinfer.fused_moe.shared.inputs import RoutingInputMode
    from flashinfer.tllm_enums import RoutingMethodType, WeightLayout

    from flashinfer.prims_ts.moe import runner

    build_io = runner.build_nvfp4_launch_io
    fc1_enabled = False

    def with_epilogue(**kwargs):
        if kwargs["fc"] == "fc1" and fc1_enabled:
            kwargs["cfg"] = replace(kwargs["cfg"], **fc1_overrides)
        if kwargs["cfg"].fuse_moe_finalize:
            kwargs["cfg"] = replace(
                kwargs["cfg"],
                dense_epilogue_warps=warps,
                epilogue_regs=248 if warps == 4 else 168,
                dense_preload_store_prefix_fragments=prefix,
                num_stages_c_smem=stages,
                num_stages_a=2,
                num_stages_b=2,
                num_stages_smem_sfa=2,
                num_stages_smem_sfb=2,
                epilogue_smem_staging=0,
                epilogue_tmem_load_width=0,
                epilogue_tmem_load_ahead=ahead,
                epilogue_tmem_release=release,
                epilogue_steps=steps,
            )
        return build_io(**kwargs)

    monkeypatch.setattr(runner, "build_nvfp4_launch_io", with_epilogue)

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
    router_weights = torch.empty_like(ids, dtype=router_weight_dtype)
    out = torch.empty((tokens, hidden), device="cuda", dtype=torch.bfloat16)
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
        topk_weights=router_weights,
    )
    monkeypatch.setattr(
        AutoTuner.get(),
        "choose_one",
        lambda name, runners, config, inputs, **kw: (runners[0], tactic),
    )

    def inputs_for(step):
        base = torch.arange(tokens, device="cuda", dtype=torch.int32)
        ids[:, 0] = (base + step) % experts
        ids[:, 1] = (base + step + 3) % experts
        router_weights.copy_(torch.rand_like(router_weights) * 1.7 + 0.1)
        weights["output2_scale_scalar"].mul_(1.03)

    def run(fused=True, **extra):
        nonlocal fc1_enabled
        # FC1 step tests compare against the retained FC1 on every reference
        # run, including changed routes/scales and captured graph replay.
        fc1_enabled = fused and fc1_overrides is not None
        return prims_ts_fp4_block_scale_moe(**kwargs, use_fused_finalize=fused, **extra)

    inputs_for(0)
    reference = run(False)[0]
    run(output=out)
    assert torch.isfinite(out).all() and out.abs().max() > 0
    assert (out.float() - reference.float()).norm() / reference.float().norm() < 0.015
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(output=out)
    for step in (1, 5):
        inputs_for(step)
        reference = run(False)[0]
        out.fill_(float("nan"))
        # Exercise repeated scratch reuse with independent epilogue warps,
        # including the partly populated last expert tiles.
        for _ in range(8 if warps == 4 else 1):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        assert (
            out.float() - reference.float()
        ).norm() / reference.float().norm() < 0.015

    # No selected expert is local: output must be zero, even after a replay
    # that populated it. Padded metadata must never be consumed by FC2.
    ids[:, 0] = 0
    ids[:, 1] = 1
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) == 0

    inputs_for(3)
    reference = run(False)[0]
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        actual = run()[0]
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    partials = run(do_finalize=False)
    assert len(partials) == 3 and partials[0].numel() > 0

    if layout == "BlockMajorK" and tactic == [1024, 0]:
        # Retain two captures and replay them on independent caller streams.
        # Both must keep their own zeroing dependencies and destination buffer.
        other_out = torch.empty_like(out)
        other_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(other_graph):
            run(output=other_out)
        caller = torch.cuda.current_stream()
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        for replay, target, worker in zip(
            (graph, other_graph), (out, other_out), streams, strict=True
        ):
            worker.wait_stream(caller)
            with torch.cuda.stream(worker):
                target.fill_(float("nan"))
                replay.replay()
            caller.wait_stream(worker)
        for target in (out, other_out):
            assert torch.isfinite(target).all()
            assert (
                target.float() - reference.float()
            ).norm() / reference.float().norm() < 0.015

    # Reuse compiled code at a different batch size after the captured work.
    # The one-token destination has no TMA dimension/alignment padding.
    if layout == "BlockMajorK" and tactic == [1024, 0]:
        tiny_x, tiny_sf = PrimsTsConfig.prepare_activations(
            torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16), quant=quant
        )
        kwargs.update(
            hidden_states=tiny_x,
            hidden_states_scale=tiny_sf,
            topk_ids=ids[:1],
            topk_weights=router_weights[:1],
        )
        reference = run(False)[0]
        actual = run()[0]
        assert torch.isfinite(actual).all()
        assert (
            actual.float() - reference.float()
        ).norm() / reference.float().norm() < 0.015

        # TopK uses coefficients selected by the Python wrapper. FC2 must
        # index that buffer, rather than a stale router-produced coefficient.
        logits = torch.full((1, experts), -10.0, device="cuda")
        logits[0, offset : offset + local] = torch.arange(
            local, device="cuda", dtype=torch.float32
        )
        kwargs.update(
            routing_logits=logits,
            routing_input_mode=RoutingInputMode.FromLogits,
            routing_method_type=int(RoutingMethodType.TopK),
            topk_ids=None,
            topk_weights=None,
        )
        run()
        topk_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(topk_graph):
            topk_output = run()[0]
        logits[0, offset : offset + local].add_(0.5)
        reference = run(False)[0]
        topk_graph.replay()
        torch.cuda.synchronize()
        assert (
            topk_output.float() - reference.float()
        ).norm() / reference.float().norm() < 0.015


@pytest.mark.arch_rubin
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_finalize_fp32_router_weights_replay(monkeypatch):
    from flashinfer.prims_ts.moe import runner

    build_io = runner.build_nvfp4_launch_io
    finalize_dtypes = []

    def record_finalize_dtype(**kwargs):
        if kwargs["cfg"].fuse_moe_finalize:
            finalize_dtypes.append(kwargs["finalize_scales"].dtype)
        return build_io(**kwargs)

    monkeypatch.setattr(runner, "build_nvfp4_launch_io", record_finalize_dtype)
    test_fused_finalize_replay(
        monkeypatch,
        layout="BlockMajorK",
        tactic=[1024, 0],
        prefix=2,
        stages=1,
        warps=8,
        release=0,
        ahead=-1,
        router_weight_dtype=torch.float32,
    )
    assert finalize_dtypes and set(finalize_dtypes) == {torch.bfloat16}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zero_output_failure_cleanup():
    from flashinfer.prims_ts.moe.finalize import zero_output_during_fc1

    caller = torch.cuda.current_stream()
    worker = torch.cuda.Stream()
    output = torch.empty(16 * 1024 * 1024, device="cuda", dtype=torch.bfloat16)
    worker.wait_stream(caller)
    allocated = torch.cuda.memory_allocated()
    with torch.cuda.stream(worker):
        for _ in range(3):
            torch.cuda._sleep(10000)
            output.fill_(float("nan"))
            with (
                pytest.raises(RuntimeError, match="launch failed"),
                zero_output_during_fc1(output),
            ):
                raise RuntimeError("launch failed")
            # Cleanup must join the outstanding memset before the next use.
            output.add_(1)
            assert torch.all(output == 1)
    caller.wait_stream(worker)
    assert torch.cuda.memory_allocated() == allocated
