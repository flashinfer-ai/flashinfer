"""clamped ReLU2 nongated MXFP8 correctness and retained graph replay on Rubin."""

import pytest
import torch

pytestmark = pytest.mark.arch_rubin


# Cover both tile widths and routing-weight placements without a Cartesian sweep.
@pytest.mark.parametrize(
    "early,tile_n,hidden,intermediate",
    [(False, 64, 384, 256), (True, 128, 512, 1024)],
)
def test_clamped_relu2_graph_replay(monkeypatch, early, tile_n, hidden, intermediate):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Requires SM107")
    from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as pkg

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    torch.manual_seed(147)
    experts, capacity, topk = 8, 65, 3
    w1 = (
        torch.randn(experts, intermediate, hidden, device="cuda") / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(experts, hidden, intermediate, device="cuda") / intermediate**0.5
    ).bfloat16()
    weights = pkg.preprocess_block_scaled_weights(
        w1, w2, quant_kind="mxfp8_e4m3", intermediate_size=intermediate, gated=False
    )
    ws = pkg.get_symm_buffer_for_sm107_block_scaled_mega_moe(
        experts,
        capacity,
        topk,
        hidden,
        intermediate,
        0,
        1,
        quant_kind="mxfp8_e4m3",
        activation="clamped_relu2",
        kernel_variant="genphase",
        cluster_shape_mn=(4, 1),
        mma_tiler_mnk=(256, tile_n, 128),
        fc2_use_bulk=True,
        apply_routing_weights_before_fc2=early,
    )
    graph = None
    try:
        x = torch.randn(capacity, hidden, device="cuda").bfloat16()
        ids = (
            torch.arange(capacity * topk, device="cuda", dtype=torch.int32).view(
                capacity, topk
            )
            % experts
        )
        scores = torch.full((capacity, topk), 1 / topk, device="cuda")
        for iteration in range(3):
            if iteration == 1:
                ids[::3, 0] = -1
                scores[:, 1] = 0
            if iteration == 2:
                x.zero_()
            stage_mega_moe_inputs(
                x,
                scores,
                ids,
                ws.x,
                ws.x_sf,
                ws.topk_idx,
                ws.topk_weights,
                kind="mxfp8_e4m3",
            )
            ws.note_staged_tokens(capacity)
            if graph is None:
                pkg.sm107_block_scaled_mega_moe(None, *weights, ws, num_tokens=capacity)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    y = pkg.sm107_block_scaled_mega_moe(
                        None, *weights, ws, num_tokens=capacity
                    )
            graph.replay()
            torch.cuda.synchronize()
            indices, expected = pkg.sampled_reference(
                ws, *weights, capacity, sample_size=capacity
            )
            error = pkg.output_error(y, indices, expected)
            assert error < 0.02, (iteration, early, tile_n, error)
    finally:
        if graph is not None:
            graph.reset()
        ws.destroy()


def test_clamped_relu2_rejects_unaligned_intermediate():
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
        Sm107BlockScaledMoeConfig,
    )

    with pytest.raises(ValueError, match="intermediate must be a multiple of 128"):
        Sm107BlockScaledMoeConfig(
            num_total_experts=8,
            max_tokens_per_rank=65,
            num_topk=3,
            hidden=384,
            intermediate=192,
            rank=0,
            world_size=1,
            activation="clamped_relu2",
            kernel_variant="genphase",
            cluster_shape_mn=(4, 1),
            fc2_use_bulk=True,
        )
