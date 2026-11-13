# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Raw signed INT4 -> packed MegaMoE vs an independent FP8 operand oracle."""

import pytest


def _decode(raw, scales):
    import torch

    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2).int()
    q = torch.where(codes < 8, codes, codes - 16).float()
    return (
        (q * scales.float().repeat_interleave(128, dim=-1))
        .to(torch.bfloat16)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )


def _reference(buffer, legs, n, intermediate, clamp):
    import torch

    from tests.moe_ep.test_sm90_pull_mxfp4_kernel_vs_reference import (
        _quantize_fc2_k64_reference,
    )

    from tests.moe_ep._sm90_fp8_wgmma_reference import fma_add, rs_k32_mm
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        _swiglu_pair_hw_match_cuda,
    )

    w1, w2 = legs
    terms = torch.zeros(
        n, buffer.num_topk, buffer.hidden, dtype=torch.bfloat16, device="cuda"
    )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for expert in range(w1.shape[0]):
            token, slot = (buffer.topk_idx[:n] == expert).nonzero(as_tuple=True)
            if token.numel() == 0:
                continue
            fc1 = rs_k32_mm(buffer.x[token], w1[expert].T)
            fc1 *= buffer.x_sf[token, :1]
            gate, up = fc1.split(intermediate, dim=-1)
            hidden = _swiglu_pair_hw_match_cuda(
                gate.clamp(max=clamp).contiguous(), up.clamp(-clamp, clamp).contiguous()
            )
            hidden *= buffer.topk_weights[token, slot, None]
            x2, s2 = _quantize_fc2_k64_reference(hidden)
            fc2 = torch.zeros(token.numel(), buffer.hidden, device="cuda")
            for k in range(0, intermediate, 64):
                partial = rs_k32_mm(x2[:, k : k + 64], w2[expert, :, k : k + 64].T)
                fc2 = fma_add(fc2, partial, s2[:, k // 64, None])
            terms[token, slot] = fc2.bfloat16()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return terms.float().sum(1).bfloat16()


@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "tile,hidden", [((128, 32, 128), 384), ((256, 64, 128), 384), ((256, 64, 256), 512)]
)
def test_int4_full_moe_reference_reuse_and_graph(monkeypatch, tile, hidden):
    import torch

    from flashinfer.utils import is_sm90a_supported

    if not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("requires SM90 and CUDA 12.3+")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel as pkg
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.int4_weights import (
        Int4WeightPack,
        preprocess_int4_mega_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl.staging import (
        stage_mega_moe_inputs,
    )

    experts, intermediate, topk, capacity = 4, 256, 2, 96
    torch.manual_seed(8431)

    def raw(rows, k):
        packed = torch.randint(
            256, (experts, rows, k // 2), dtype=torch.uint8, device="cuda"
        )
        scales = (
            torch.rand(experts, rows, k // 128, device="cuda") * 0.025 + 0.005
        ).bfloat16()
        return packed, scales

    w1, s1 = raw(2 * intermediate, hidden)
    w2, s2 = raw(hidden, intermediate)
    transformed = preprocess_int4_mega_weights(
        Int4WeightPack(w1, s1, w2, s2),
        hidden_size=hidden,
        intermediate_size=intermediate,
    )
    reference_weights = (_decode(w1, s1), _decode(w2, s2))
    buffer = pkg.get_symm_buffer_for_hopper_int4_mega_moe(
        experts,
        capacity,
        topk,
        hidden,
        intermediate,
        0,
        1,
        mma_tiler_mnk=tile,
        gate_up_clamp=4.0,
    )
    try:
        # A shorter second request must clear all inactive rows from the first.
        for n in (73, 13):
            x = torch.randn(n, hidden, device="cuda").bfloat16()
            token = torch.arange(n, device="cuda")
            ids = torch.stack((token % experts, (token + 1) % experts), -1).long()
            routing = torch.rand(n, topk, device="cuda").softmax(-1)
            stage_mega_moe_inputs(
                x,
                routing,
                ids,
                buffer.x,
                buffer.x_sf,
                buffer.topk_idx,
                buffer.topk_weights,
                quantize_input=True,
            )
            expected = _reference(buffer, reference_weights, n, intermediate, 4.0)
            actual = pkg.hopper_int4_mega_moe(
                None, *transformed, buffer, num_tokens=n, gate_up_clamp=4.0, sync=True
            ).clone()
            assert torch.isfinite(actual).all()
            rel_l2 = (
                actual.float() - expected.float()
            ).norm() / expected.float().norm()
            print(f"INT4 {tile} H={hidden} T={n} rel_l2={float(rel_l2):.6f}")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                thunk = pkg.hopper_int4_mega_launch_thunk(*transformed, buffer)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                thunk()
            torch.cuda.current_stream().wait_stream(stream)
            buffer.output_activation.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(buffer.output_activation[:n], actual)
    finally:
        buffer.destroy()
