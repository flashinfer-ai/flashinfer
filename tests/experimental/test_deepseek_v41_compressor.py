# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from flashinfer.deepseek_v41 import deepseek_v41_compressor_decode


def gate():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("CSA2 decode requires SM100/SM103")


def check(actual, expected):
    a, e = actual.double(), expected.double()
    difference = a - e
    assert bool(torch.isfinite(a).all())
    assert float(difference.norm() / e.norm().clamp_min(1e-20)) <= 0.004
    assert float(difference.abs().max() / e.abs().max().clamp_min(1e-20)) <= 0.008


@pytest.mark.parametrize("norm_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mixed_parity", [False, True])
def test_csa2_state_fp64_pool_round_norm_and_changing_graph(norm_dtype, mixed_parity):
    gate()
    torch.manual_seed(4129)
    kv, score = [torch.randn(4, 512, device="cuda") for _ in range(2)]
    kv_state, score_state = [torch.randn(4, 2, 512, device="cuda") for _ in range(2)]
    ref_kv, ref_score = kv_state.clone(), score_state.clone()
    weight = (torch.rand(512, device="cuda") + 0.5).to(norm_dtype)
    starts = torch.tensor([0, 2, 126, 65534], device="cuda", dtype=torch.int32)
    if mixed_parity:
        starts[1::2].add_(1)
    out = torch.full((4, 512), 17.0, device="cuda", dtype=torch.bfloat16)
    positions = torch.empty(4, device="cuda", dtype=torch.int32)
    expected_out = out.clone()

    def run():
        deepseek_v41_compressor_decode(
            kv,
            score,
            kv_state,
            score_state,
            weight,
            starts,
            out=out,
            positions=positions,
        )

    def verify():
        for row, start in enumerate(starts.cpu().tolist()):
            ref_kv[row, start % 2] = kv[row]
            ref_score[row, start % 2] = score[row]
            if start % 2:
                # FP64 independent pooling, then the required BF16 boundary.
                pooled = (
                    (ref_kv[row].double() * ref_score[row].double().softmax(0))
                    .sum(0)
                    .bfloat16()
                    .double()
                )
                expected_out[row] = (
                    pooled
                    * torch.rsqrt(pooled.square().mean() + 1e-20)
                    * weight.double()
                ).bfloat16()
        check(out, expected_out)
        torch.testing.assert_close(kv_state, ref_kv, rtol=0, atol=0)
        torch.testing.assert_close(score_state, ref_score, rtol=0, atol=0)
        torch.testing.assert_close(
            positions, torch.where(starts % 2 == 1, starts // 2, -1), rtol=0, atol=0
        )

    run()
    verify()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for step in range(5):
        starts.add_(1)
        kv.copy_(torch.randn_like(kv))
        score.copy_(torch.randn_like(score) * (120 if step == 2 else 1))
        weight.mul_(0.875)
        graph.replay()
        verify()
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_compressor_decode(
            kv,
            score,
            kv_state,
            score_state,
            weight,
            starts,
            out=kv.view(torch.bfloat16).view(-1, 512)[:4],
            positions=positions,
        )


def test_csa2_incomplete_does_not_read_uninitialized_partner_or_write_output():
    gate()
    kv = torch.zeros(1, 512, device="cuda")
    score = torch.zeros_like(kv)
    kv_state = torch.full((1, 2, 512), torch.nan, device="cuda")
    score_state = torch.full_like(kv_state, torch.nan)
    weight = torch.ones(512, device="cuda")
    starts = torch.zeros(1, device="cuda", dtype=torch.int32)
    out = torch.full((1, 512), -17.0, device="cuda", dtype=torch.bfloat16)
    positions = torch.empty(1, device="cuda", dtype=torch.int32)
    deepseek_v41_compressor_decode(
        kv, score, kv_state, score_state, weight, starts, out=out, positions=positions
    )
    torch.testing.assert_close(out, torch.full_like(out, -17), rtol=0, atol=0)
    assert positions.item() == -1
    assert torch.isnan(kv_state[:, 1]).all() and torch.isnan(score_state[:, 1]).all()
    starts.fill_(1)
    deepseek_v41_compressor_decode(
        kv, score, kv_state, score_state, weight, starts, out=out, positions=positions
    )
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
    assert positions.item() == 0
    with pytest.raises(ValueError, match="inference-only"):
        deepseek_v41_compressor_decode(
            kv.requires_grad_(), score, kv_state, score_state, weight, starts
        )
