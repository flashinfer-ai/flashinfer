"""clamped ReLU2 public-layer EP4 regression; run under torchrun --nproc_per_node=4."""

import os

import pytest
import torch

pytestmark = [pytest.mark.gpu_4, pytest.mark.arch_rubin]


def test_clamped_relu2_layer_ep4():
    capacity = 65
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("Requires torchrun with four ranks")
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        PrequantizedMoEWeights,
        Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
        ensure_moe_ep_cuda_device,
    )
    from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as pkg

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Requires SM107")
    rank = int(os.environ["RANK"])
    boot = BootstrapConfig(rank=rank, world_size=4)
    ensure_moe_ep_cuda_device(boot)
    torch.manual_seed(146 + rank)
    experts, hidden, intermediate, topk = 32, 512, 1024, 4
    # Canonical quantized checkpoint weights exercise the no-requantization path.
    w1 = (
        torch.randn(experts // 4, intermediate, hidden, device="cuda") / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(experts // 4, hidden, intermediate, device="cuda")
        / intermediate**0.5
    ).bfloat16()
    q1, sf1 = pkg.quantize_mxfp8_block32(w1, torch.float8_e4m3fn)
    q2, sf2 = pkg.quantize_mxfp8_block32(w2, torch.float8_e4m3fn)
    del w1, w2
    layer = MoEEpLayer(
        bootstrap=boot,
        fleet_params=FleetParams(
            num_experts=experts, max_tokens_per_rank=capacity, token_hidden_size=hidden
        ),
        weights=PrequantizedMoEWeights(q1, q2, sf1, sf2),
        backend=MegaConfig(
            megakernel=Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
                intermediate,
                topk,
                activation="clamped_relu2",
                kernel_variant="genphase",
                cluster_shape_mn=(4, 1),
                mma_tiler_mnk=(256, 64, 128),
                fc2_use_bulk=True,
            ),
            quantize_input=True,
        ),
    )
    graph = None
    try:
        x = torch.randn(capacity, hidden, device="cuda").bfloat16()
        ids = torch.stack(
            [torch.randperm(experts, device="cuda")[:topk] for _ in range(capacity)]
        ).int()
        scores = torch.softmax(torch.randn(capacity, topk, device="cuda"), dim=-1)
        t = MoEEpTensors(x, ids, scores)
        layer.forward(t)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = layer.forward(t)
        original_ids, original_scores = ids.clone(), scores.clone()
        for iteration in range(5):
            if iteration == 1:
                ids[:, :4] = torch.arange(
                    4, device="cuda"
                )  # concentrate work on rank 0
                ids[::3, 0] = -1
                scores[:, 1] = 0
            if iteration == 2:
                layer._workspace.output_activation.fill_(float("nan"))
                ids.fill_(-1)
                scores.zero_()
            if iteration == 3:
                ids.copy_(original_ids)
                scores.copy_(original_scores)
            if iteration == 4:
                x.zero_()
            graph.replay()
            torch.cuda.synchronize()
            indices, expected = pkg.sampled_reference(
                layer._workspace, *layer._transformed, capacity, sample_size=capacity
            )
            error = pkg.output_error(y, indices, expected)
            print(
                f"CLAMPED_RELU2_EP4 rank={rank} capacity={capacity} iteration={iteration} relative_l2={error}",
                flush=True,
            )
            assert error < 0.02
    finally:
        if graph is not None:
            graph.reset()
        layer.destroy()
