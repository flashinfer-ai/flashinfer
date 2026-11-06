"""Mixed Blackwell weight layouts, preparation and shared-workspace replay."""

import dataclasses
import os

import pytest
import torch
import torch.distributed as dist

from flashinfer import moe_ep
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg


def test_mixed_config_and_candidates(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim.tuning import (
        sm100_standard_candidates,
        is_valid_sm100_knobs,
    )

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    config = moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=256,
        top_k=3,
        knobs="auto",
        finalize_mode=moe_ep.FinalizeMode.NONE,
    )
    backend = create_mega_kernel(config)
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(world_size=2, rank=0, auto_bootstrap=False)
    )
    fleet = moe_ep.FleetParams(
        num_experts=16, max_tokens_per_rank=129, token_hidden_size=384
    )
    native = backend._workspace_config(fleet)
    assert native.quant_kind == "mxfp4_mxfp8"
    assert native.instruction_k == 32
    assert native.torch_act_data_dtype == torch.float8_e4m3fn
    assert native.torch_act_sf_dtype == torch.float8_e8m0fnu
    assert native.kernel_variant == "inference"
    candidates = sm100_standard_candidates(native)
    assert candidates
    assert all(is_valid_sm100_knobs(candidate, native) for candidate in candidates)
    for changes in ({"hidden": 192}, {"intermediate": 192}):
        with pytest.raises(ValueError, match="128"):
            dataclasses.replace(native, **changes)
    with pytest.raises(ValueError, match="standard provider"):
        moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=256, top_k=3, kind="mxfp4_mxfp8"
        )


def test_mixed_cli_dispatch(monkeypatch):
    from flashinfer.moe_ep import tune
    from flashinfer.moe_ep.backends.mega.kernel.sm100 import next_tuning

    called = []
    monkeypatch.setattr(
        next_tuning, "run_tuning", lambda args: called.append(args) or 0
    )
    args = [
        "--arch",
        "sm100",
        "--dtype",
        "mxfp4_mxfp8",
        "--hidden",
        "384",
        "--intermediate",
        "256",
        "--num-experts",
        "16",
        "--topk",
        "3",
        "--max-tokens",
        "129",
    ]
    assert tune.main(args) == 2
    args += ["--implementation", "next"]
    assert tune.main(args) == 0
    assert called[0].dtype == "mxfp4_mxfp8"
    assert tune.main(args + ["--kernel-variant", "genphase"]) == 0
    assert called[1].kernel_variant == "genphase"
    assert len(called) == 2


@pytest.mark.arch_blackwell
@pytest.mark.gpu_2
@pytest.mark.parametrize("quantize_input", [False, True])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("early", [False, True])
def test_mixed_layer_replay(quantize_input, fused, early, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires Blackwell")
    rank, world = (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103")
    if world == 1:
        monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "1" if fused else "0")
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import (
        kernel_helpers,
    )
    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as staging_pkg
    from flashinfer.moe_ep.backends.mega.kernel import routing_staging

    calls = []
    original_quant, original_routes = (
        staging_pkg.fused_quant_stage,
        routing_staging.stage_routes,
    )

    def quant(*args, **kwargs):
        calls.append("quant")
        return original_quant(*args, **kwargs)

    def routes(*args, **kwargs):
        calls.append("routes")
        return original_routes(*args, **kwargs)

    monkeypatch.setattr(staging_pkg, "fused_quant_stage", quant)
    monkeypatch.setattr(routing_staging, "stage_routes", routes)
    capacity, hidden, intermediate, topk, local_experts = 129, 384, 256, 3, 4
    experts = local_experts * world
    rng = torch.Generator(device="cuda").manual_seed(971 + rank)
    w13 = (
        torch.randn(
            local_experts, 2 * intermediate, hidden, device="cuda", generator=rng
        )
        / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(local_experts, hidden, intermediate, device="cuda", generator=rng)
        / intermediate**0.5
    ).bfloat16()
    q13, sf13 = kernel_helpers.quantize_mxfp4_block32(w13)
    q2, sf2 = kernel_helpers.quantize_mxfp4_block32(w2)
    weights = (
        moe_ep.MoEWeightPack(w13, w2)
        if quantize_input
        else moe_ep.MoEWeightPack(
            q13, q2, sf13.view(torch.uint8), sf2.view(torch.uint8)
        )
    )
    config = moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=intermediate,
        top_k=topk,
        apply_routing_weights_before_fc2=early,
    )
    bootstrap = moe_ep.BootstrapConfig(world_size=world, rank=rank)
    fleet = moe_ep.FleetParams(
        num_experts=experts, max_tokens_per_rank=capacity, token_hidden_size=hidden
    )
    layer = moe_ep.MoEEpLayer(
        bootstrap,
        fleet,
        weights,
        backend=moe_ep.MegaConfig(megakernel=config, quantize_input=quantize_input),
    )
    peer = None
    captures = []
    try:
        expected_weights = pkg.preprocess_prequantized_block_scaled_weights(
            q13,
            q2,
            sf13,
            sf2,
            quant_kind="mxfp4_mxfp8",
            hidden_size=hidden,
            intermediate_size=intermediate,
        )
        for leg, expected_leg in zip(layer._transformed, expected_weights, strict=True):
            for actual, expected in zip(leg, expected_leg, strict=True):
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        peer = moe_ep.MoEEpLayer(
            bootstrap,
            fleet,
            weights,
            backend=moe_ep.MegaConfig(
                megakernel=config,
                quantize_input=False,
                preprocess_weights=False,
                transformed_weights=layer._transformed,
            ),
        )

        def inputs(n, quantize, dtype):
            x = torch.randn(n, hidden, device="cuda", generator=rng).bfloat16()
            if n:
                x[0].zero_()
                x[1].fill_(2.0**-120)
                q, sf = pkg.quantize_mxfp8_block32(x, torch.float8_e4m3fn)
            else:
                q = torch.empty(0, hidden, device="cuda", dtype=torch.float8_e4m3fn)
                sf = torch.empty(
                    0, hidden // 32, device="cuda", dtype=torch.float8_e8m0fnu
                )
            ids = (
                torch.arange(n * topk, device="cuda").reshape(n, topk) + rank * 3
            ) % experts
            ids[:, 0] = -1
            scores = torch.full((n, topk), 1 / topk, device="cuda", dtype=dtype)
            scores[:, 0] = float("nan")
            return (
                moe_ep.MoEEpTensors(
                    hidden_states=x if quantize else q,
                    scales=None if quantize else sf,
                    topk_ids=ids.to(torch.int32 if quantize else torch.int64),
                    topk_weights=scores,
                ),
                q,
                sf,
            )

        def check(current_layer, t, out, q, sf):
            ws = current_layer._ensure_workspace()
            n = t.hidden_states.shape[0]
            assert torch.equal(ws.x[:n].view(torch.uint8), q.view(torch.uint8))
            assert torch.equal(
                ws.x_sf[:n, : hidden // 32].view(torch.uint8), sf.view(torch.uint8)
            )
            assert torch.equal(ws.topk_idx[:n], t.topk_ids.to(torch.int32))
            torch.testing.assert_close(
                ws.topk_weights[:n],
                t.topk_weights.float().masked_fill(t.topk_ids == -1, 0),
                rtol=0,
                atol=0,
            )
            assert (ws.topk_idx[n:] == -1).all()
            assert t.topk_weights[:, 0].isnan().all()
            # The collective oracle needs equal payload sizes even on empty ranks.
            indices, expected = pkg.sampled_reference(
                ws, *current_layer._transformed, n, sample_size=capacity
            )
            error = pkg.output_error(out, indices, expected)
            error = torch.tensor(error, device="cuda")
            if world > 1:
                dist.all_reduce(error, op=dist.ReduceOp.MAX)
            assert error.item() < 0.03
            assert torch.isfinite(out).all()

        for current_layer, n, quantize, dtype in (
            (layer, capacity, quantize_input, torch.float32),
            (peer, 65, False, torch.float16),
            (layer, 7, quantize_input, torch.float32),
        ):
            t, q, sf = inputs(n, quantize, dtype)
            current_layer.warmup(t)
            result = current_layer.forward(t, return_workspace_view=True)
            check(current_layer, t, result, q, sf)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = current_layer.forward(t, return_workspace_view=True)
            captures.append((graph, current_layer, t, output, q, sf))
        assert layer._ensure_workspace() is peer._ensure_workspace()
        assert ("quant" in calls) == (fused and quantize_input)
        assert "routes" in calls
        for index in (0, 2, 1, 2, 0):
            graph, current_layer, t, output, q, sf = captures[index]
            t.topk_ids[:, 1:] = (t.topk_ids[:, 1:] + 1) % experts
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            check(current_layer, t, output, q, sf)
        t, q, sf = inputs(0 if rank == 0 else 7, quantize_input, torch.float32)
        output = layer.forward(t, return_workspace_view=True)
        check(layer, t, output, q, sf)
    finally:
        for graph, *_ in captures:
            graph.reset()
        if peer is not None:
            peer.destroy()
        layer.destroy()
