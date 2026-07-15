"""Single-GPU check: DeepGEMM ``fp8_fp4_mega_moe`` vs a pure-torch oracle.

DeepGEMM counterpart of ``test_nvfp4_cutedsl_kernel_vs_reference.py``: a
single-rank ``DeepGemmMegaKernelBackend.compute`` launch must match a
pure-torch dequant reference (fp32 GEMMs + DeepGEMM's SwiGLU convention +
fc1-out per-32 e4m3/UE8M0 round-trip).

DeepGEMM's own correctness test compares the fused kernel against an unfused
dispatch+GEMM+tilelang-swiglu pipeline (kernel-vs-kernel); this file is the
independent torch-math anchor for the FlashInfer dg mega path.

Oracle conventions (pinned against ``deep_gemm`` sources):
  * activations: fp8 e4m3, per-32 UE8M0 scales (``per_token_cast_to_fp8``)
  * weights: fp4 e2m1 packed int8, per-32 scales (``per_token_cast_to_fp4``)
  * fc1 acc → bf16 round → gate = first half, up = second half
  * clamp: ``gate = min(gate, c)``; ``up = clamp(up, -c, c)`` (bf16, pre-SiLU)
  * ``silu(gate) * up * topk_weight`` folded BEFORE the fc1-out fp8 round-trip
  * combine: plain sum over topk (weight already folded)

The deep_gemm symmetric buffer requires an initialized process group, so run
under a single-process torchrun::

    torchrun --standalone --nproc_per_node=1 -m pytest \\
        tests/moe_ep/test_deep_gemm_mega_kernel_vs_reference.py -v \\
        -m arch_blackwell
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

_PG_TIMEOUT = timedelta(minutes=60)

HIDDEN = 4096
INTERMEDIATE = 2048
NUM_TOKENS = 64
MAX_TOKENS = 64
NUM_EXPERTS = 8
TOP_K = 4
GRAN_K = 32
CLAMP = 10.0


def _e2m1_decode_table(device):
    import torch

    return torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=device,
    )


def _dequant_fp4_gran32(packed_int8, sf_fp32):
    """(rows, K//2) packed e2m1 + (rows, K//32) fp32 scales → fp32 (rows, K)."""
    import torch

    raw = packed_int8.view(torch.uint8)
    lut = _e2m1_decode_table(raw.device)
    lo = lut[(raw & 0x0F).to(torch.int64)]
    hi = lut[(raw >> 4).to(torch.int64)]
    vals = torch.empty(
        raw.shape[0], raw.shape[1] * 2, dtype=torch.float32, device=raw.device
    )
    vals[:, ::2] = lo
    vals[:, 1::2] = hi
    return vals * sf_fp32.repeat_interleave(GRAN_K, dim=-1)


def _fp8_gran32_roundtrip(x_fp32):
    """Per-32 e4m3 + UE8M0 quantize→dequantize, mirroring the kernel epilogue."""
    from deep_gemm.utils import per_token_cast_to_fp8

    q, sf = per_token_cast_to_fp8(x_fp32, use_ue8m0=True, gran_k=GRAN_K)
    return q.float() * sf.repeat_interleave(GRAN_K, dim=-1)


def _torch_dg_mega_reference(problem):
    """Pure-torch oracle for the fp8_fp4 DeepGEMM mega-MoE (single rank)."""
    import torch
    from deep_gemm.utils import per_token_cast_to_fp8

    x = problem["hidden_states"]
    topk_ids = problem["topk_ids"]
    topk_weights = problem["topk_weights"]
    wp = problem["weights"]
    num_tokens = x.shape[0]

    # Same activation quant recipe as the FI dg staging (packed-vs-plain SF
    # layout differs, the values do not).
    x_q, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=GRAN_K)
    x_deq = x_q.float() * x_sf.repeat_interleave(GRAN_K, dim=-1)

    out = torch.zeros(num_tokens, HIDDEN, dtype=torch.float32, device=x.device)
    for expert in range(NUM_EXPERTS):
        routing_mask = topk_ids == expert
        if not routing_mask.any():
            continue
        routed = routing_mask.nonzero(as_tuple=False)
        tokens, slots = routed[:, 0], routed[:, 1]

        w13_d = _dequant_fp4_gran32(wp.w13[expert], wp.w13_scale[expert])
        w2_d = _dequant_fp4_gran32(wp.w2[expert], wp.w2_scale[expert])

        fc1 = x_deq[tokens] @ w13_d.transpose(0, 1)  # (R, 2I) fp32
        # The kernel rounds the fc1 accumulator to bf16 before clamp + SiLU.
        fc1 = fc1.to(torch.bfloat16).float()
        gate = fc1[:, :INTERMEDIATE]
        up = fc1[:, INTERMEDIATE:]
        gate = gate.clamp(max=CLAMP)
        up = up.clamp(min=-CLAMP, max=CLAMP)
        swiglu = gate * torch.sigmoid(gate) * up
        # topk weight folded before the fc1-out fp8 round-trip.
        swiglu = swiglu * topk_weights[tokens, slots].unsqueeze(-1)

        swiglu_rt = _fp8_gran32_roundtrip(swiglu)
        out.index_put_((tokens,), swiglu_rt @ w2_d.transpose(0, 1), accumulate=True)

    return out.to(torch.bfloat16)


def _make_problem():
    import torch
    from deep_gemm.utils import per_token_cast_to_fp4

    from flashinfer.moe_ep import MoEWeightPack

    g = torch.Generator(device="cuda").manual_seed(7)
    hidden_states = torch.randn(
        NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=g
    )
    scores = torch.randn(
        NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32, device="cuda", generator=g
    )
    topk_weights, topk_ids = torch.topk(
        scores, TOP_K, dim=-1, largest=True, sorted=False
    )
    topk_weights = torch.softmax(topk_weights, dim=-1)

    gw = torch.Generator(device="cuda").manual_seed(13)
    w13_bf16 = torch.randn(
        NUM_EXPERTS,
        2 * INTERMEDIATE,
        HIDDEN,
        dtype=torch.bfloat16,
        device="cuda",
        generator=gw,
    ) * (HIDDEN**-0.5)
    w2_bf16 = torch.randn(
        NUM_EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=gw,
    ) * (INTERMEDIATE**-0.5)

    w13 = torch.empty(
        NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2, dtype=torch.int8, device="cuda"
    )
    w2 = torch.empty(
        NUM_EXPERTS, HIDDEN, INTERMEDIATE // 2, dtype=torch.int8, device="cuda"
    )
    w13_sf = torch.empty(
        NUM_EXPERTS,
        2 * INTERMEDIATE,
        HIDDEN // GRAN_K,
        dtype=torch.float32,
        device="cuda",
    )
    w2_sf = torch.empty(
        NUM_EXPERTS,
        HIDDEN,
        INTERMEDIATE // GRAN_K,
        dtype=torch.float32,
        device="cuda",
    )
    for e in range(NUM_EXPERTS):
        w13[e], w13_sf[e] = per_token_cast_to_fp4(
            w13_bf16[e], use_ue8m0=True, gran_k=GRAN_K
        )
        w2[e], w2_sf[e] = per_token_cast_to_fp4(
            w2_bf16[e], use_ue8m0=True, gran_k=GRAN_K
        )

    return dict(
        hidden_states=hidden_states,
        topk_ids=topk_ids.to(torch.int64),
        topk_weights=topk_weights.to(torch.float32),
        weights=MoEWeightPack(w13=w13, w2=w2, w13_scale=w13_sf, w2_scale=w2_sf),
    )


@pytest.mark.arch_blackwell
def test_deep_gemm_mega_kernel_matches_torch_reference():
    pytest.importorskip("deep_gemm")
    pytest.importorskip("triton")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("fp8_fp4_mega_moe requires SM100-family Blackwell")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        pytest.skip("single-rank oracle test; run with --nproc_per_node=1")

    import deep_gemm
    import torch.distributed as dist

    from flashinfer.moe_ep import DeepGemmMegaMoeConfig, preprocess_mega_weights
    from flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.backend import (
        DeepGemmMegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.staging import (
        stage_mega_moe_inputs,
    )

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=_PG_TIMEOUT)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    problem = _make_problem()

    symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        NUM_EXPERTS,
        MAX_TOKENS,
        TOP_K,
        HIDDEN,
        INTERMEDIATE,
    )
    try:
        stage_mega_moe_inputs(
            problem["hidden_states"],
            problem["topk_weights"],
            problem["topk_ids"],
            symm_buffer.x[:NUM_TOKENS],
            symm_buffer.x_sf[:NUM_TOKENS],
            symm_buffer.topk_idx[:NUM_TOKENS],
            symm_buffer.topk_weights[:NUM_TOKENS],
        )

        transformed = preprocess_mega_weights(
            problem["weights"],
            intermediate_size=INTERMEDIATE,
            hidden_size=HIDDEN,
        )

        y_kernel = torch.empty(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
        kernel = DeepGemmMegaKernelBackend(
            DeepGemmMegaMoeConfig(
                intermediate_size=INTERMEDIATE,
                top_k=TOP_K,
                activation_clamp=CLAMP,
                # fast_math uses __expf/fast-rcp SiLU; disable for a tighter
                # oracle band (the multirank tests cover fast_math=True).
                fast_math=False,
            )
        )
        kernel.compute(symm_buffer, transformed, output=y_kernel)
        torch.cuda.synchronize()

        y_ref = _torch_dg_mega_reference(problem)

        assert torch.isfinite(y_kernel).all()
        yk = y_kernel.to(torch.float32)
        yr = y_ref.to(torch.float32)
        rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
        print(
            f"[dg oracle] rel_l2={rel_l2.item():.4g} "
            f"max|Δ|={(yk - yr).abs().max().item():.4g} "
            f"amax(ref)={yr.abs().max().item():.4g}"
        )
        # Both sides consume identical quantized operands; the residual is fp8
        # RTNE flips at fc1-out plus GEMM accumulation-order noise on |y|~O(1)
        # (weights are dim^-0.5 scaled).
        torch.testing.assert_close(yk, yr, atol=0.15, rtol=0.05)
        assert rel_l2.item() < 0.02
    finally:
        symm_buffer.destroy()
