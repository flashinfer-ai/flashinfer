"""Blackwell activation and finalization through the public MegaMoE layer."""

import os

import pytest
import torch
import torch.distributed as dist

from flashinfer import moe_ep
from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg


@pytest.mark.arch_blackwell
@pytest.mark.gpu_4
@pytest.mark.parametrize(
    "kind,activation,mode,early,wire,in_kernel",
    [
        ("nvfp4", "minimax_scale", "DEFAULT", True, "bf16", False),
        ("nvfp4", "minimax", "PERSISTENT", False, "bf16", False),
        ("nvfp4", "minimax", "NONE", True, "bf16", False),
        ("nvfp4", "situ", "DEFAULT", False, "bf16", False),
        ("nvfp4", "situ_unbounded", "PERSISTENT", True, "bf16", False),
        ("nvfp4", "situ_unbounded", "NONE", False, "bf16", False),
        ("mxfp8_e4m3", "minimax", "DEFAULT", True, "bf16", False),
        ("mxfp8_e4m3", "minimax", "PERSISTENT", False, "bf16", False),
        ("mxfp8_e4m3", "situ_unbounded", "NONE", True, "bf16", False),
        ("mxfp8_e5m2", "minimax", "NONE", False, "bf16", False),
        ("mxfp8_e5m2", "situ", "PERSISTENT", True, "bf16", False),
        ("mxfp8_e5m2", "situ_unbounded", "DEFAULT", False, "bf16", False),
        ("nvfp4", "minimax", "DEFAULT", True, "mxfp8", False),
        ("nvfp4", "minimax", "PERSISTENT", False, "mxfp8", False),
        ("nvfp4", "situ_unbounded", "DEFAULT", False, "nvfp4", False),
        ("nvfp4", "situ", "PERSISTENT", True, "nvfp4", False),
        ("nvfp4", "minimax", "DEFAULT", True, "bf16", True),
        ("mxfp8_e4m3", "situ_unbounded", "DEFAULT", True, "bf16", True),
    ],
)
def test_standard_features(
    kind, activation, mode, early, wire, in_kernel, monkeypatch, *, autotune=False
):
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
    capacity, hidden, intermediate, topk, experts = 129, 384, 192, 3, 8 * world
    rng = torch.Generator(device="cuda").manual_seed(731 + rank)
    x = torch.randn(capacity, hidden, generator=rng, device="cuda").bfloat16()
    weights = moe_ep.MoEWeightPack(
        (
            torch.randn(8, 2 * intermediate, hidden, generator=rng, device="cuda")
            / hidden**0.5
        ).bfloat16(),
        (
            torch.randn(8, hidden, intermediate, generator=rng, device="cuda")
            / intermediate**0.5
        ).bfloat16(),
    )
    ids = (
        torch.arange(capacity * topk, device="cuda").view(capacity, topk) + rank * 5
    ) % experts
    ids[0] = -1
    ids[1, 1] = experts
    scores = torch.full((capacity, topk), 1 / topk, device="cuda")
    scores[(ids < 0) | (ids >= experts)] = float("nan")
    count = torch.tensor([capacity], dtype=torch.int32, device="cuda")
    options = dict(
        intermediate_size=intermediate,
        top_k=topk,
        finalize_mode=moe_ep.FinalizeMode[mode],
        num_valid_tokens_tensor=count if mode == "PERSISTENT" else None,
        apply_routing_weights_before_fc2=early,
        enable_in_kernel_fc2_reduce=in_kernel,
        knobs={"in_kernel_fc2_reduce": True} if in_kernel else None,
    )
    if autotune:
        options["knobs"] = "auto"
    if activation.startswith("minimax"):
        options.update(swiglu_alpha=1.702, swiglu_beta=1.0, gate_up_clamp=7.0)
    else:
        options.update(
            activation="situ",
            situ_beta=7.0,
            situ_linear_beta=None if activation.endswith("unbounded") else 5.0,
        )
    if kind == "nvfp4":
        config = moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next",
            combine_dtype=wire,
            enable_fc1_activation_per_token_scale=activation.endswith("scale"),
            **options,
        )
    else:
        config = moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            kernel_name="sm100_mxfp8_mxfp8_bf16_cutedsl_next",
            kind=kind,
            **options,
        )
    layer = moe_ep.MoEEpLayer(
        moe_ep.BootstrapConfig(world_size=world, rank=rank),
        moe_ep.FleetParams(
            num_experts=experts, max_tokens_per_rank=capacity, token_hidden_size=hidden
        ),
        weights,
        backend=moe_ep.MegaConfig(megakernel=config),
    )
    scales = (
        torch.linspace(0.5, 1.5, capacity, device="cuda")
        if activation.endswith("scale")
        else None
    )
    current = moe_ep.MoEEpTensors(
        hidden_states=x,
        topk_ids=ids.clone(),
        topk_weights=scores.clone(),
        fc1_activation_per_token_scale=scales,
    )

    def forward(t):
        if mode == "NONE":
            return layer.forward_unfinalized(t)
        return layer.forward(t, return_workspace_view=True)

    def check(result, tokens):
        if mode == "NONE":
            rows, remaining_weights, route_map = result
            assert rows.shape == (current.num_tokens * topk, hidden)
            assert torch.equal(
                route_map.flatten(),
                torch.arange(rows.shape[0], device="cuda", dtype=torch.int32),
            )
            values = rows[route_map].float()
            actual = (values * remaining_weights[:, :, None]).sum(1).bfloat16()
        else:
            actual = result
        indices, expected = pkg.sampled_reference(
            ws, *layer._transformed, tokens, sample_size=32
        )
        error = torch.tensor(
            pkg.output_error(actual, indices, expected) if tokens else 0.0,
            device="cuda",
        )
        if world > 1:
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
        assert error.item() < 0.03
        assert torch.isfinite(actual[:tokens]).all()

    graph = None
    try:
        layer.warmup(current)
        if autotune:
            assert layer.get_tuned_knobs()
        ws = layer._ensure_workspace()
        if mode != "NONE":
            owned = layer.forward(current)
            assert isinstance(owned, torch.Tensor)
            check(owned, capacity)
            current.output = torch.empty_like(owned)
            supplied = layer.forward(current)
            assert supplied is current.output
            check(supplied, capacity)
            current.output = None
        check(forward(current), capacity)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            captured = forward(current)
        torch.cuda.current_stream().wait_stream(stream)
        for live, masked in (
            (capacity, False),
            (7, True),
            (0, False),
            (65, False),
            (capacity, False),
        ):
            current.topk_ids.copy_(ids)
            if masked:
                current.topk_ids.fill_(-1)
            count.fill_(live)
            if ws._pre_reduced_activation is not None:
                ws._pre_reduced_activation.view(torch.bfloat16).fill_(float("nan"))
            ws.output_activation.fill_(-19)
            graph.replay()
            torch.cuda.synchronize()
            valid = live if mode == "PERSISTENT" else capacity
            check(captured, valid)
            if mode == "PERSISTENT":
                assert torch.all(ws.output_activation[valid:] == -19)
            assert count.item() == live
        for _ in range(16):
            graph.replay()
        torch.cuda.synchronize()
        check(captured, capacity)
        if mode == "NONE":
            graph = None
            n = 0 if rank == 0 else 7
            current = moe_ep.MoEEpTensors(
                hidden_states=x[:n],
                topk_ids=ids[:n],
                topk_weights=scores[:n],
            )
            check(forward(current), n)
    finally:
        graph = None
        layer.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("kind", ["mxfp8_e4m3", "mxfp8_e5m2"])
def test_standard_padded_mxfp8_staging(kind, fused, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "1" if fused else "0")
    x = torch.randn(65, 192, device="cuda", dtype=torch.bfloat16)
    weights = moe_ep.MoEWeightPack(
        torch.randn(4, 256, 192, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(4, 192, 128, device="cuda", dtype=torch.bfloat16) / 16,
    )
    layer = moe_ep.MoEEpLayer(
        moe_ep.BootstrapConfig(world_size=1, rank=0),
        moe_ep.FleetParams(
            num_experts=4, max_tokens_per_rank=65, token_hidden_size=192
        ),
        weights,
        backend=moe_ep.MegaConfig(
            megakernel=moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=128,
                top_k=2,
                kind=kind,
                kernel_name="sm100_mxfp8_mxfp8_bf16_cutedsl_next",
            )
        ),
    )
    try:
        from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import (
            kernel_helpers as helpers,
        )

        dtype = torch.float8_e4m3fn if kind == "mxfp8_e4m3" else torch.float8_e5m2
        q13, sf13 = helpers.quantize_mxfp8_block32(weights.w13.float(), dtype)
        q2, sf2 = helpers.quantize_mxfp8_block32(weights.w2.float(), dtype)
        prequantized = helpers.preprocess_prequantized_block_scaled_weights(
            q13,
            q2,
            sf13,
            sf2,
            quant_kind=kind,
            hidden_size=192,
            intermediate_size=128,
        )
        for actual, expected_leg in zip(layer._transformed, prequantized, strict=True):
            for a, b in zip(actual, expected_leg, strict=True):
                assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        result = layer.forward(
            moe_ep.MoEEpTensors(
                hidden_states=x,
                topk_ids=torch.zeros(65, 2, dtype=torch.int64, device="cuda"),
                topk_weights=torch.full((65, 2), 0.5, device="cuda"),
            ),
            return_workspace_view=True,
        )
        indices, expected = pkg.sampled_reference(
            layer._ensure_workspace(), *layer._transformed, 65
        )
        assert pkg.output_error(result, indices, expected) < 0.03
    finally:
        layer.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("mode", ["AUTO", "CAKE"])
def test_standard_cake_finalizer(mode, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    torch.manual_seed(935)
    layer = moe_ep.MoEEpLayer(
        moe_ep.BootstrapConfig(world_size=1, rank=0),
        moe_ep.FleetParams(
            num_experts=8, max_tokens_per_rank=256, token_hidden_size=4096
        ),
        moe_ep.MoEWeightPack(
            torch.randn(8, 256, 4096, device="cuda", dtype=torch.bfloat16) / 64,
            torch.randn(8, 4096, 128, device="cuda", dtype=torch.bfloat16) / 12,
        ),
        backend=moe_ep.MegaConfig(
            megakernel=moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=128,
                top_k=6,
                finalize_mode=moe_ep.FinalizeMode[mode],
                kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next",
            )
        ),
    )
    current = moe_ep.MoEEpTensors(
        hidden_states=torch.randn(17, 4096, device="cuda", dtype=torch.bfloat16),
        topk_ids=torch.arange(17 * 6, device="cuda").view(17, 6) % 8,
        topk_weights=torch.full((17, 6), 1 / 6, device="cuda"),
    )
    try:
        layer.warmup(current)
        ws = layer._ensure_workspace()
        assert ws.config.use_custom_finalize
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            output = layer.forward(current, return_workspace_view=True)
        torch.cuda.current_stream().wait_stream(stream)
        for masked in (False, True):
            if masked:
                current.topk_ids.fill_(-1)
            ws.custom_finalize_workspace().fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            indices, expected = pkg.sampled_reference(ws, *layer._transformed, 17)
            assert pkg.output_error(output, indices, expected) < 0.03
    finally:
        layer.destroy()
