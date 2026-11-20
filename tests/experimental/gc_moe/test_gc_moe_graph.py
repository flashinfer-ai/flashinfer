# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph capture and replay of the experimental GC MoE call."""

import torch

import flashinfer

from .reference import (
    GEOMETRY,
    checkpoint_weights,
    cosine,
    decode_fp4_payload,
    reference_forward,
    stage,
    stage_tokens,
)


def test_capture_and_replay(gc_group, backend):
    device = torch.device("cuda", torch.cuda.current_device())
    experts, hidden, intermediate = (
        GEOMETRY["num_experts"],
        GEOMETRY["hidden"],
        GEOMETRY["intermediate_hidden"],
    )
    topk = GEOMETRY["num_topk"]
    l1_checkpoint, l2_checkpoint = checkpoint_weights(
        backend, experts, hidden, intermediate, device
    )
    w1 = decode_fp4_payload(l1_checkpoint[0], hidden)
    w2 = decode_fp4_payload(l2_checkpoint[0], intermediate)

    with flashinfer.GCMoEWorkspace(
        gc_group, experts, GEOMETRY["capacity"], topk, hidden, intermediate
    ) as workspace:
        l1_weights, l2_weights = flashinfer.transform_gc_moe_weights(
            l1_checkpoint, l2_checkpoint
        )
        y = torch.empty(8, hidden, dtype=torch.bfloat16, device=device)

        def case(seed):
            x, topk_idx, topk_weights = stage_tokens(
                8, hidden, experts, topk, device, seed
            )
            stage(workspace, x, topk_idx, topk_weights)
            expected = reference_forward(
                x, topk_idx, topk_weights, w1, w2, intermediate
            )
            return expected

        first = case(7)
        flashinfer.gc_moe(y, l1_weights, l2_weights, workspace)
        torch.cuda.synchronize()
        eager_first = y.clone()
        assert cosine(eager_first, first) > 0.97

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                flashinfer.gc_moe(y, l1_weights, l2_weights, workspace)

            y.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(y, eager_first), (
                "graph replay diverged from the eager run"
            )

            second_staging = stage_tokens(8, hidden, experts, topk, device, seed=11)
            stage(workspace, *second_staging)
            second = reference_forward(*second_staging, w1, w2, intermediate)
            y.zero_()
            flashinfer.gc_moe(y, l1_weights, l2_weights, workspace)
            torch.cuda.synchronize()
            eager_second = y.clone()
            assert cosine(eager_second, second) > 0.97

            y.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(y, eager_second), (
                "graph replay ignored the re-staged tokens"
            )
            assert not torch.equal(eager_second, eager_first)
        finally:
            # Retire the graph before the workspace allocation is destroyed.
            del graph
            torch.cuda.synchronize()
