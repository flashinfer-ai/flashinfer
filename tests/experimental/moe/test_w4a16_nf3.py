"""Numerical reference tests for the NF3 ("nf3_2p1") W4A16 MoE kernel path.

Packs NF3 codes/scales with the production packer (prepare_nf3_moe_weights),
compiles the fused MoE kernel with weight_layout="nf3_2p1", runs it through
the production host entry (run_w4a16_moe), and compares against a pure-torch
reference MoE built from the same dequantized weights. Covers both the
TC-decode (small-M direct top-k, fused top-k sum) and route-packed paths.

The tile_n coupling: the flat-span NF3 layout is packed for a specific CTA
N-tile, so the test compiles the fused kernel FIRST, reads fc1_tile_n/fc2_tile_n
back off the compile result, and packs with exactly those. The same
(cached) kernel is then reused for the launch, so packing and kernel agree.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from flashinfer.experimental.sm12x.moe._shared.kernels.w4a16.kernel import (
    compile_w4a16_fused_moe,
    run_w4a16_moe,
)
from flashinfer.experimental.sm12x.moe._shared.kernels.w4a16.prepare import (
    _NF3_CODEBOOK,
    prepare_nf3_moe_weights,
)
from flashinfer.experimental.sm12x.moe._shared.kernels.w4a16.host import (
    make_w4a16_packed_buffers,
    max_packed_route_slots,
    select_route_block_size_m,
)

_DEVICE = torch.device("cuda")
_DTYPE = torch.bfloat16
_DEFAULT_MAX_SHARED_MEM = 101_376


def _round_to_e4m3_scale(t_s: torch.Tensor) -> torch.Tensor:
    """Round positive scales to 3 mantissa bits so the NF3 e4m3-style K/32 scale
    encoding is lossless (real checkpoints already have <=3 mantissa bits, so the
    kernel decode reproduces exactly these values -- the reference can then use
    them directly)."""
    t_s = t_s.to(torch.float32).clamp(min=2.0**-7)
    e = torch.floor(torch.log2(t_s))
    step = torch.pow(2.0, e - 3)
    return torch.round(t_s / step) * step


def _dequant(codes: torch.Tensor, t_s: torch.Tensor) -> torch.Tensor:
    """[E, N, K] codes + [E, N, K//32] scales -> bf16 [E, N, K] weights, matching
    the kernel: bf16 codebook value * (3-mantissa) scale, rounded to bf16."""
    cb = torch.tensor(_NF3_CODEBOOK, dtype=torch.bfloat16, device=codes.device)
    w = cb[codes.long()].to(torch.float32)  # [E, N, K] bf16-valued
    scale = t_s.to(torch.float32).repeat_interleave(32, dim=2)  # [E, N, K]
    return (w * scale).to(torch.bfloat16)


def _reference_moe(
    a: torch.Tensor,
    w13_deq: torch.Tensor,
    w2_deq: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    intermediate_size: int,
) -> torch.Tensor:
    m, hidden = a.shape
    topk = topk_ids.shape[1]
    out = torch.zeros((m, hidden), dtype=torch.float32, device=a.device)
    a_f = a.to(torch.float32)
    for t in range(m):
        for k in range(topk):
            e = int(topk_ids[t, k])
            w13 = w13_deq[e].to(torch.float32)  # [2I, hidden]
            fc1 = a_f[t] @ w13.T  # [2I]
            gate = fc1[:intermediate_size]
            up = fc1[intermediate_size:]
            silu = gate * torch.sigmoid(gate)
            act = (silu.to(torch.bfloat16) * up.to(torch.bfloat16)).to(torch.float32)
            w2 = w2_deq[e].to(torch.float32)  # [hidden, I]
            fc2 = act @ w2.T  # [hidden]
            out[t] += float(topk_weights[t, k]) * fc2
    return out


def _build_problem(m: int, seed: int = 0):
    torch.manual_seed(seed)
    num_experts = 4
    topk = 8
    intermediate_size = 64
    hidden = 256
    w13_rows = 2 * intermediate_size  # 128

    w13_codes = torch.randint(
        0, 8, (num_experts, w13_rows, hidden), dtype=torch.int32, device=_DEVICE
    )
    w2_codes = torch.randint(
        0,
        8,
        (num_experts, hidden, intermediate_size),
        dtype=torch.int32,
        device=_DEVICE,
    )
    w13_scale = _round_to_e4m3_scale(
        0.01 + 0.24 * torch.rand(num_experts, w13_rows, hidden // 32, device=_DEVICE)
    )
    w2_scale = _round_to_e4m3_scale(
        0.01
        + 0.24
        * torch.rand(num_experts, hidden, intermediate_size // 32, device=_DEVICE)
    )
    a = torch.randn(m, hidden, dtype=_DTYPE, device=_DEVICE) * 0.1
    topk_ids = torch.randint(
        0, num_experts, (m, topk), dtype=torch.int32, device=_DEVICE
    )
    topk_weights = torch.rand(m, topk, dtype=torch.float32, device=_DEVICE)
    return dict(
        num_experts=num_experts,
        topk=topk,
        intermediate_size=intermediate_size,
        hidden=hidden,
        w13_rows=w13_rows,
        w13_codes=w13_codes,
        w2_codes=w2_codes,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        a=a,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )


def _device_limits():
    props = torch.cuda.get_device_properties(_DEVICE)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(
        getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
    )
    return sms, max_shared_mem


def _run_case(m: int, *, tc_decode: bool) -> None:
    p = _build_problem(m, seed=1234 + m + (100 if tc_decode else 0))
    sms, max_shared_mem = _device_limits()

    if tc_decode:
        block_size_m = 8
        direct_topk = True
        max_m_blocks = m * p["topk"]
    else:
        block_size_m = select_route_block_size_m(m, p["topk"], p["num_experts"])
        direct_topk = False
        route_slots = max_packed_route_slots(
            m * p["topk"], block_size_m, p["num_experts"]
        )
        max_m_blocks = (route_slots + block_size_m - 1) // block_size_m

    fused = compile_w4a16_fused_moe(
        size_m=m,
        hidden_size=p["hidden"],
        intermediate_size=p["intermediate_size"],
        num_experts=p["num_experts"],
        top_k=p["topk"],
        activation="silu",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=block_size_m,
        max_m_blocks=int(max_m_blocks),
        element_dtype="bf16",
        fast_math=True,
        sms=sms,
        max_shared_mem=max_shared_mem,
        weight_layout="nf3_2p1",
        scale_format="e4m3_k32",
        w13_layout="w13",
        direct_topk_routes=direct_topk,
        tc_decode_fused_sum=tc_decode,
    )

    prepared = prepare_nf3_moe_weights(
        p["w13_codes"],
        p["w13_scale"],
        p["w2_codes"],
        p["w2_scale"],
        activation="silu",
        fc1_tile_n=int(fused.fc1_tile_n),
        fc2_tile_n=int(fused.fc2_tile_n),
        params_dtype=_DTYPE,
    )

    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=p["topk"],
        dtype=_DTYPE,
        device=_DEVICE,
    )

    out = run_w4a16_moe(
        p["a"],
        prepared,
        p["topk_weights"],
        p["topk_ids"],
        activation="silu",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        fused_launch=fused,
    )
    torch.cuda.synchronize()

    w13_deq = _dequant(p["w13_codes"], p["w13_scale"])
    w2_deq = _dequant(p["w2_codes"], p["w2_scale"])
    ref = _reference_moe(
        p["a"],
        w13_deq,
        w2_deq,
        p["topk_ids"],
        p["topk_weights"],
        p["intermediate_size"],
    )
    got = out.to(torch.float32)
    denom = ref.abs().amax().clamp(min=1e-6)
    rel = (got - ref).abs().amax() / denom
    tag = "tc_decode" if tc_decode else "packed_route"
    assert rel < 2e-2, f"NF3 {tag} m={m} rel err {float(rel):.4f} exceeds 2e-2"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("m", "tc_decode"),
    [
        # Small-M decode path: fused top-k sum (tc_zero_output prologue) +
        # direct top-k routes on the NF3 weights.
        (1, True),
        (5, True),
        # Route-packed path + shared w4a16_topk_sum: the prefill-class NF3 GEMM.
        (33, False),
    ],
)
def test_nf3_matches_dequant_reference(m: int, tc_decode: bool) -> None:
    _run_case(m, tc_decode=tc_decode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nf3_tc_decode_production_shape_with_tier_mask() -> None:
    """Cover the GLM-5.2 TP4 hybrid contract used by the vLLM integration.

    The serving path compiles one m=8 launch at pinned 256-wide tiles, reuses
    it at m=1, and maps routes belonging to the other precision tier to -1.
    """
    torch.manual_seed(20260714)
    m, capacity_m = 1, 8
    num_experts, topk = 4, 8
    hidden, intermediate = 6144, 512
    w13_rows = 2 * intermediate

    w13_codes = torch.randint(
        0,
        8,
        (num_experts, w13_rows, hidden),
        dtype=torch.int32,
        device=_DEVICE,
    )
    w2_codes = torch.randint(
        0,
        8,
        (num_experts, hidden, intermediate),
        dtype=torch.int32,
        device=_DEVICE,
    )
    w13_scale = _round_to_e4m3_scale(
        0.01 + 0.24 * torch.rand(num_experts, w13_rows, hidden // 32, device=_DEVICE)
    )
    w2_scale = _round_to_e4m3_scale(
        0.01
        + 0.24 * torch.rand(num_experts, hidden, intermediate // 32, device=_DEVICE)
    )
    x = (torch.randn(m, hidden, device=_DEVICE) * 0.1).to(_DTYPE)
    topk_ids = torch.tensor(
        [[0, -1, 1, -1, 2, -1, 3, -1]], dtype=torch.int32, device=_DEVICE
    )
    topk_weights = torch.softmax(torch.randn(m, topk, device=_DEVICE), dim=-1)

    sms, max_shared_mem = _device_limits()
    fused = compile_w4a16_fused_moe(
        size_m=capacity_m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=num_experts,
        top_k=topk,
        activation="silu",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=8,
        max_m_blocks=capacity_m * topk,
        element_dtype="bf16",
        fast_math=True,
        sms=sms,
        max_shared_mem=max_shared_mem,
        weight_layout="nf3_2p1",
        scale_format="e4m3_k32",
        w13_layout="w13",
        direct_topk_routes=True,
        tc_decode_fused_sum=True,
        force_tile_config=(64, 256, 64, 256),
    )
    prepared = prepare_nf3_moe_weights(
        w13_codes,
        w13_scale,
        w2_codes,
        w2_scale,
        activation="silu",
        fc1_tile_n=256,
        fc2_tile_n=256,
        params_dtype=_DTYPE,
    )
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=capacity_m,
        topk=topk,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    out = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation="silu",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output[:m],
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        fused_launch=fused,
    )
    torch.cuda.synchronize()

    w13_deq = _dequant(w13_codes, w13_scale)
    w2_deq = _dequant(w2_codes, w2_scale)
    ref = torch.zeros((m, hidden), dtype=torch.float32, device=_DEVICE)
    x_f = x.float()
    for route in range(topk):
        expert = int(topk_ids[0, route])
        if expert < 0:
            continue
        fc1 = x_f[0] @ w13_deq[expert].float().T
        gate, up = fc1[:intermediate], fc1[intermediate:]
        act = (gate * torch.sigmoid(gate)).to(_DTYPE) * up.to(_DTYPE)
        ref[0] += float(topk_weights[0, route]) * (
            act.float() @ w2_deq[expert].float().T
        )

    denom = ref.abs().amax().clamp(min=1e-6)
    rel = (out.float() - ref).abs().amax() / denom
    assert rel < 2e-2, f"production-shape NF3 TC decode rel err {float(rel):.4f}"

    mismatched = replace(prepared, fc1_tile_n=128)
    with pytest.raises(RuntimeError, match="NF3 packing geometry"):
        run_w4a16_moe(
            x,
            mismatched,
            topk_weights,
            topk_ids,
            activation="silu",
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output[:m],
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            fused_launch=fused,
        )
