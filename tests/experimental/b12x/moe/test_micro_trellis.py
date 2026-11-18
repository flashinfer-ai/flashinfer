"""Trellis weight-source parity for the direct micro kernel."""

from __future__ import annotations

import pytest
import torch

from b12x._lib.quant.lut_e4m3 import lut_e4m3_value_table_cpu
from cutlass.cute.runtime import from_dlpack

from b12x.moe._shared.kernels.micro import MoEMicroKernelBackend
from b12x.testing.reference.helpers import require_b12x
from b12x.testing.reference.trellis_moe import (
    build_trellis_weight,
    trellis_moe_reference,
)

require_b12x(allow_module_level=True)

_BITS = 2


def _run_micro_trellis(
    *,
    E: int,
    m: int,
    K: int,
    n: int,
    top_k: int,
    seed: int,
    mac: int = 64,
    intermediate_hadamard: bool = False,
):
    device = torch.device("cuda")
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed * 11 + 3)
    gate_payload, gate_w = build_trellis_weight(gen, E, n, K, _BITS, device)
    up_payload, up_w = build_trellis_weight(gen, E, n, K, _BITS, device)
    down_payload, down_w = build_trellis_weight(gen, E, K, n, _BITS, device)
    # Micro layout is projection-major [2][E][K16][N16].
    w1_payload = (
        torch.stack((gate_payload, up_payload), dim=0).contiguous().view(torch.uint8)
    )
    w2_payload = down_payload.contiguous().view(torch.uint8)
    w13_ref = torch.stack((gate_w, up_w), dim=0)

    x = (torch.randn(m, K, device=device) * 1.0e-2).to(torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(m)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(m, top_k, device=device), dim=-1).float()
    rot_segments = 6 if intermediate_hadamard else 3
    rotations = (
        torch.rand((E, rot_segments * n), generator=gen, dtype=torch.float32).add_(0.5)
    ).to(device=device, dtype=torch.float16)

    oracle = trellis_moe_reference(
        x.float(),
        w13_ref,
        down_w,
        rotations,
        topk_ids,
        topk_weights,
        intermediate_hadamard=intermediate_hadamard,
    )

    kernel = MoEMicroKernelBackend(
        16,
        (64, 128),
        1,
        activation="situ",
        a8_mx_mode=True,
        scale_format="e8m0_k32",
        weight_layout="trellis_t256",
        trellis_bits=_BITS,
        trellis_intermediate_hadamard=intermediate_hadamard,
        share_input_across_experts=True,
        single_token=m == 1,
    )
    kernel.configure(m=m, k=K, n=n, num_topk=top_k, weight_E=E, max_active_ctas=mac)
    cfg = kernel._cfg

    ones = torch.ones(E, dtype=torch.float32, device=device)
    dummy_scale = torch.zeros(64, dtype=torch.uint8, device=device)
    inter = torch.zeros(kernel.inter_alloc_u32, dtype=torch.float32, device=device)
    out = torch.zeros(m, K, dtype=torch.bfloat16, device=device)
    barrier_count = torch.zeros(m * top_k + m * 16, dtype=torch.int32, device=device)
    barrier_epoch = torch.zeros_like(barrier_count)
    value_table = lut_e4m3_value_table_cpu().to(device)
    rot_flat = rotations.reshape(-1).contiguous()

    MoEMicroKernelBackend.launch(
        kernel,
        x=x.reshape(-1),
        w1_fp4=w1_payload.reshape(-1),
        w1_blockscale=dummy_scale,
        w1_alphas=ones,
        a1_gscale=ones,
        a2_gscale=ones,
        inter_fp32=inter,
        w2_fp4=w2_payload.reshape(-1),
        w2_blockscale=dummy_scale,
        w2_alphas=ones,
        topk_ids=topk_ids.reshape(-1).contiguous(),
        topk_weights=topk_weights.reshape(-1).contiguous(),
        out=out.reshape(-1),
        barrier_count=from_dlpack(barrier_count, assumed_align=16),
        barrier_epoch=from_dlpack(barrier_epoch, assumed_align=16),
        m=m,
        grid_x=kernel.grid_x,
        trellis_lut=value_table,
        trellis_rotations=rot_flat,
    )
    torch.cuda.synchronize()
    return out.float(), oracle


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1])
# mac=8 pins the single-CTA FC1 path (trellis_ksplit=1); mac=64 exercises
# the K-split global-scratch merge at the maximum split factor.
@pytest.mark.parametrize("mac", [8, 64])
@pytest.mark.parametrize("intermediate_hadamard", [False, True])
def test_micro_trellis_matches_reference(
    m: int, mac: int, intermediate_hadamard: bool
) -> None:
    got, want = _run_micro_trellis(
        E=8,
        m=m,
        K=512,
        n=256,
        top_k=4,
        seed=20260815,
        mac=mac,
        intermediate_hadamard=intermediate_hadamard,
    )
    assert torch.isfinite(got).all()
    cosine = torch.nn.functional.cosine_similarity(
        got.reshape(1, -1), want.reshape(1, -1)
    ).item()
    assert cosine > 0.995, cosine
    rel = ((got - want).norm() / want.norm().clamp_min(1e-9)).item()
    assert rel < 0.12, rel
