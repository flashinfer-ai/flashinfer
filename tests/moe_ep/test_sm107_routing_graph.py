# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Changed-route CUDA Graph replay with partial dumps and empty expert buckets."""

import pytest
import torch


@pytest.mark.arch_rubin
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp4_mxfp8"])
@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize(
    "family,capacity", [("inference", 1024), ("inference", 4096), ("genphase", 1024)]
)
def test_routing_graph_changes(monkeypatch, kind, early, family, capacity):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Requires SM107")
    from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl.staging import (
        stage_mega_moe_inputs as stage_mx,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs as stage_nv,
    )
    from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as pkg

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    torch.manual_seed(6197)
    hidden, intermediate, experts, topk, live = 512, 256, 8, 8, 33
    w13 = (
        torch.randn(experts, 2 * intermediate, hidden, device="cuda") / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(experts, hidden, intermediate, device="cuda") / intermediate**0.5
    ).bfloat16()
    weights = pkg.preprocess_block_scaled_weights(
        w13, w2, quant_kind=kind, intermediate_size=intermediate
    )
    knobs = dict(
        kernel_variant=family,
        apply_topk_at_fc1=early,
        reduce_topk_in_kernel=False,
    )
    if family == "genphase":
        knobs.update(cluster_shape_mn=(4, 1), fc2_use_bulk=True)
    ws = pkg.get_symm_buffer_for_sm107_block_scaled_mega_moe(
        experts, capacity, topk, hidden, intermediate, 0, 1, quant_kind=kind, **knobs
    )
    graph = None
    try:
        if family == "inference":
            # Exercise both the contiguous dump and the segmented multi-CTA dump.
            assert ws.kernel.token_comm._router.router_data_cta_count == (
                2 if capacity == 4096 else 1
            )
        x = torch.randn(live, hidden, device="cuda").bfloat16()
        base_ids = (
            torch.arange(live * topk, device="cuda", dtype=torch.int32).view(live, topk)
            % experts
        )
        scores = torch.rand(live, topk, device="cuda")
        scores /= scores.sum(dim=-1, keepdim=True)
        stage = stage_nv if kind == "nvfp4" else stage_mx
        extra = {} if kind == "nvfp4" else {"kind": kind}

        def stage_state(ids, iteration):
            count = stage(
                x * (1 + 0.125 * iteration),
                scores,
                ids,
                ws.x,
                ws.x_sf,
                ws.topk_idx,
                ws.topk_weights,
                **extra,
            )
            assert count == live
            ws.note_staged_tokens(count)

        stage_state(base_ids, 0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                ws.launch(*weights)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            ws.launch(*weights)

        for iteration, state in enumerate(
            ("active", "partial", "last-only", "empty", "restore")
        ):
            ids = (base_ids + iteration).remainder(experts)
            if state == "partial":
                ids[::3, ::2] = -1
            elif state == "last-only":
                ids.fill_(-1)
                ids[-1, -1] = experts - 1
            elif state == "empty":
                ids.fill_(-1)
            stage_state(ids, iteration)
            ws.output_activation[:live].fill_(float("nan"))
            indices, expected = pkg.sampled_reference(
                ws, *weights, live, sample_size=live
            )
            assert indices.numel() == live
            # Replay the same graph after changing the data and masking routes.
            # Stale combine slots from earlier IDs must not affect the output.
            for _ in range(2):
                graph.replay()
                torch.cuda.synchronize()
                output = ws.output_activation[:live]
                assert torch.isfinite(output).all(), state
                error = pkg.output_error(output, indices, expected)
                assert error < (0.06 if kind == "nvfp4" else 0.02), (state, error)
                assert torch.count_nonzero(output[~(ids >= 0).any(dim=-1)]) == 0
    finally:
        if graph is not None:
            graph.reset()
        ws.destroy()
