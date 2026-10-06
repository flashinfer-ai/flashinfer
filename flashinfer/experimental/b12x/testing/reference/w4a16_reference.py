from __future__ import annotations

from dataclasses import dataclass

import torch
from b12x._lib.intrinsics import fp4_quantize_values_torch


@dataclass(frozen=True)
class OracleMetrics:
    max_abs: float
    rmse: float
    mean_abs: float
    cos: float


@dataclass(frozen=True)
class MoERouteTrace:
    token_idx: int
    route_idx: int
    expert_idx: int
    activation: str
    router_weight: float
    alpha_fc1: float
    alpha_fc2: float
    gs_fc1: float
    gs_fc2: float
    x_dequant: torch.Tensor
    fc1_out: torch.Tensor | None
    gate_out: torch.Tensor | None
    up_out: torch.Tensor | None
    intermediate: torch.Tensor
    int_dequant: torch.Tensor
    down_out: torch.Tensor
    routed_out: torch.Tensor


def compare_to_reference(actual: torch.Tensor, reference: torch.Tensor) -> OracleMetrics:
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    diff = actual_fp32 - reference_fp32
    actual_rows = actual_fp32.reshape(actual_fp32.shape[0], -1)
    reference_rows = reference_fp32.reshape(reference_fp32.shape[0], -1)
    dot = (actual_rows * reference_rows).sum(dim=1)
    actual_norm = actual_rows.norm(dim=1)
    reference_norm = reference_rows.norm(dim=1)
    denom = actual_norm * reference_norm
    both_zero = (actual_norm <= 1e-12) & (reference_norm <= 1e-12)
    cos_rows = torch.where(
        both_zero,
        torch.ones_like(dot),
        torch.where(denom > 1e-24, dot / denom, torch.zeros_like(dot)),
    )
    cos = cos_rows.mean().item()
    return OracleMetrics(
        max_abs=diff.abs().max().item(),
        rmse=diff.square().mean().sqrt().item(),
        mean_abs=diff.abs().mean().item(),
        cos=cos,
    )


def unswizzle_block_scale(swizzled_scale: torch.Tensor, rows: int, cols_blocks: int) -> torch.Tensor:
    cols_padded = ((cols_blocks + 3) // 4) * 4
    rows_padded = ((rows + 127) // 128) * 128
    unswizzled = swizzled_scale.view(torch.float8_e4m3fn).reshape(
        rows_padded // 128, cols_padded // 4, 32, 4, 4,
    )
    unswizzled = unswizzled.permute(0, 3, 2, 1, 4).contiguous()
    unswizzled = unswizzled.reshape(rows_padded, cols_padded)
    return unswizzled[:rows, :cols_blocks].to(torch.float32)


def _validate_reference_inputs(
    w1_fp4: torch.Tensor,
    I_tp: int,
    activation: str,
) -> None:
    if activation not in {"silu", "situ", "relu2"}:
        raise ValueError(f"unsupported activation {activation!r}")
    expected_w1_rows = 2 * I_tp if activation in {"silu", "situ"} else I_tp
    if w1_fp4.shape[1] != expected_w1_rows:
        raise ValueError(
            f"expected w1_fp4.shape[1] == {expected_w1_rows} for activation "
            f"{activation!r}, got {w1_fp4.shape[1]}"
        )


def _apply_gated_activation(
    gate: torch.Tensor,
    up: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    if activation == "situ":
        beta = 4.0
        linear_beta = 25.0
        situ_gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        situ_up = linear_beta * torch.tanh(up / linear_beta)
        return situ_gate * situ_up
    return torch.sigmoid(gate) * gate * up


def _make_fp4_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        dtype=torch.float32,
        device=device,
    )


def _dequant_fp4(
    packed_u8: torch.Tensor,
    rows: int,
    cols: int,
    fp4_lut: torch.Tensor,
) -> torch.Tensor:
    lo = (packed_u8 & 0x0F).to(torch.int64)
    hi = ((packed_u8 >> 4) & 0x0F).to(torch.int64)
    return torch.stack([fp4_lut[lo], fp4_lut[hi]], dim=-1).reshape(rows, cols)


def _apply_block_scales(
    raw: torch.Tensor,
    sf_f32: torch.Tensor,
    rows: int,
    cols: int,
    *,
    block_size: int,
) -> torch.Tensor:
    n_blocks = cols // block_size
    sf = sf_f32[:rows, :n_blocks]
    return raw * sf.unsqueeze(-1).expand(rows, n_blocks, block_size).reshape(rows, cols)


def _quantize_vec_to_fp4_dequant(
    vals_f32: torch.Tensor,
    global_scale: float,
    *,
    block_size: int,
    fp8_e4m3_max: float,
) -> torch.Tensor:
    cols = vals_f32.shape[0]
    n_blocks = cols // block_size
    blocked = vals_f32.reshape(n_blocks, block_size)
    block_max = blocked.abs().amax(dim=-1)

    raw_scale = (block_max * global_scale / 6.0).clamp(max=fp8_e4m3_max)
    sf_e4m3 = raw_scale.to(torch.float8_e4m3fn).to(torch.float32)

    sf_times_gs = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols) / global_scale
    scaled = vals_f32 / sf_times_gs.clamp(min=1e-30)
    quant = fp4_quantize_values_torch(scaled)
    sf_only = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols)
    return quant * sf_only


def _trace_nvfp4_route(
    *,
    x_f32: torch.Tensor,
    w1_fp4_eid: torch.Tensor,
    w1_blockscale_eid: torch.Tensor,
    alpha_fc1: float,
    w2_fp4_eid: torch.Tensor,
    w2_blockscale_eid: torch.Tensor,
    alpha_fc2: float,
    gs_fc1: float,
    gs_fc2: float,
    K: int,
    I_tp: int,
    token_idx: int,
    route_idx: int,
    expert_idx: int,
    router_weight: float,
    activation: str,
) -> MoERouteTrace:
    block_size = 16
    fp8_e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)
    fp4_lut = _make_fp4_lut(x_f32.device)
    is_gated = activation in {"silu", "situ"}

    x_dequant = _quantize_vec_to_fp4_dequant(
        x_f32,
        gs_fc1,
        block_size=block_size,
        fp8_e4m3_max=fp8_e4m3_max,
    )
    w2_sf = unswizzle_block_scale(w2_blockscale_eid, K, I_tp // block_size)

    fc1_out = None
    gate_out = None
    up_out = None
    if is_gated:
        w13_sf = unswizzle_block_scale(w1_blockscale_eid, 2 * I_tp, K // block_size)
        up_dequant = _apply_block_scales(
            _dequant_fp4(w1_fp4_eid[:I_tp], I_tp, K, fp4_lut),
            w13_sf[:I_tp],
            I_tp,
            K,
            block_size=block_size,
        )
        gate_dequant = _apply_block_scales(
            _dequant_fp4(w1_fp4_eid[I_tp:], I_tp, K, fp4_lut),
            w13_sf[I_tp:],
            I_tp,
            K,
            block_size=block_size,
        )
        gate_out = (gate_dequant @ x_dequant) * alpha_fc1
        up_out = (up_dequant @ x_dequant) * alpha_fc1
        intermediate = _apply_gated_activation(
            gate_out,
            up_out,
            activation,
        ).to(torch.bfloat16).float()
    else:
        w1_sf = unswizzle_block_scale(w1_blockscale_eid, I_tp, K // block_size)
        fc1_dequant = _apply_block_scales(
            _dequant_fp4(w1_fp4_eid[:I_tp], I_tp, K, fp4_lut),
            w1_sf[:I_tp],
            I_tp,
            K,
            block_size=block_size,
        )
        fc1_out = (fc1_dequant @ x_dequant) * alpha_fc1
        intermediate = torch.square(torch.relu(fc1_out)).to(torch.bfloat16).float()

    int_dequant = _quantize_vec_to_fp4_dequant(
        intermediate,
        gs_fc2,
        block_size=block_size,
        fp8_e4m3_max=fp8_e4m3_max,
    )
    down_dequant = _apply_block_scales(
        _dequant_fp4(w2_fp4_eid, K, I_tp, fp4_lut),
        w2_sf,
        K,
        I_tp,
        block_size=block_size,
    )
    down_out = ((down_dequant @ int_dequant) * alpha_fc2).to(torch.bfloat16)
    routed_out = (router_weight * down_out.float()).to(torch.bfloat16)
    return MoERouteTrace(
        token_idx=token_idx,
        route_idx=route_idx,
        expert_idx=expert_idx,
        activation=activation,
        router_weight=router_weight,
        alpha_fc1=alpha_fc1,
        alpha_fc2=alpha_fc2,
        gs_fc1=gs_fc1,
        gs_fc2=gs_fc2,
        x_dequant=x_dequant,
        fc1_out=fc1_out,
        gate_out=gate_out,
        up_out=up_out,
        intermediate=intermediate,
        int_dequant=int_dequant,
        down_out=down_out,
        routed_out=routed_out,
    )


def _trace_w4a16_route(
    *,
    x_f32: torch.Tensor,
    w1_fp4_eid: torch.Tensor,
    w1_blockscale_eid: torch.Tensor,
    alpha_fc1: float,
    w2_fp4_eid: torch.Tensor,
    w2_blockscale_eid: torch.Tensor,
    alpha_fc2: float,
    K: int,
    I_tp: int,
    token_idx: int,
    route_idx: int,
    expert_idx: int,
    router_weight: float,
    activation: str,
    swiglu_limit: float | None = None,
) -> MoERouteTrace:
    block_size = 16
    fp4_lut = _make_fp4_lut(x_f32.device)
    is_gated = activation in {"silu", "situ"}

    x_bf16 = x_f32.to(torch.bfloat16).float()
    w2_sf = unswizzle_block_scale(w2_blockscale_eid, K, I_tp // block_size)

    fc1_out = None
    gate_out = None
    up_out = None
    if is_gated:
        w13_sf = unswizzle_block_scale(w1_blockscale_eid, 2 * I_tp, K // block_size)
        up_dequant = _apply_block_scales(
            _dequant_fp4(w1_fp4_eid[:I_tp], I_tp, K, fp4_lut),
            w13_sf[:I_tp],
            I_tp,
            K,
            block_size=block_size,
        ).to(torch.bfloat16).float()
        gate_dequant = _apply_block_scales(
            _dequant_fp4(w1_fp4_eid[I_tp:], I_tp, K, fp4_lut),
            w13_sf[I_tp:],
            I_tp,
            K,
            block_size=block_size,
        ).to(torch.bfloat16).float()
        gate_out = (gate_dequant @ x_bf16) * alpha_fc1
        up_out = (up_dequant @ x_bf16) * alpha_fc1
        if swiglu_limit is not None:
            gate_out = torch.clamp(gate_out, max=swiglu_limit)
            up_out = torch.clamp(up_out, min=-swiglu_limit, max=swiglu_limit)
        intermediate = _apply_gated_activation(
            gate_out,
            up_out,
            activation,
        ).to(torch.bfloat16).float()
    else:
        w1_sf = unswizzle_block_scale(w1_blockscale_eid, I_tp, K // block_size)
        fc1_dequant = _apply_block_scales(
            _dequant_fp4(w1_fp4_eid[:I_tp], I_tp, K, fp4_lut),
            w1_sf[:I_tp],
            I_tp,
            K,
            block_size=block_size,
        ).to(torch.bfloat16).float()
        fc1_out = (fc1_dequant @ x_bf16) * alpha_fc1
        intermediate = torch.square(torch.relu(fc1_out)).to(torch.bfloat16).float()

    down_dequant = _apply_block_scales(
        _dequant_fp4(w2_fp4_eid, K, I_tp, fp4_lut),
        w2_sf,
        K,
        I_tp,
        block_size=block_size,
    ).to(torch.bfloat16).float()
    down_out = ((down_dequant @ intermediate) * alpha_fc2).to(torch.bfloat16)
    routed_out = (router_weight * down_out.float()).to(torch.bfloat16)
    return MoERouteTrace(
        token_idx=token_idx,
        route_idx=route_idx,
        expert_idx=expert_idx,
        activation=activation,
        router_weight=router_weight,
        alpha_fc1=alpha_fc1,
        alpha_fc2=alpha_fc2,
        gs_fc1=1.0,
        gs_fc2=1.0,
        x_dequant=x_bf16,
        fc1_out=fc1_out,
        gate_out=gate_out,
        up_out=up_out,
        intermediate=intermediate,
        int_dequant=intermediate,
        down_out=down_out,
        routed_out=routed_out,
    )


def trace_moe_reference_nvfp4_route(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    token_idx: int,
    route_idx: int,
    activation: str = "silu",
) -> MoERouteTrace:
    del E
    _validate_reference_inputs(w1_fp4, I_tp, activation)
    if token_idx < 0 or token_idx >= x.shape[0]:
        raise IndexError(f"token_idx {token_idx} is out of range for batch {x.shape[0]}")
    if route_idx < 0 or route_idx >= topk_ids.shape[1]:
        raise IndexError(f"route_idx {route_idx} is out of range for top_k {topk_ids.shape[1]}")

    x_f32 = x[token_idx].float()
    expert_idx = int(topk_ids[token_idx, route_idx].item())
    router_weight = float(topk_weights[token_idx, route_idx].item())
    alpha_fc1 = float(w1_alphas[expert_idx].item())
    alpha_fc2 = float(w2_alphas[expert_idx].item())
    gs_fc1 = float(a1_gscale[expert_idx].item()) if a1_gscale.numel() > 1 else float(a1_gscale.item())
    gs_fc2 = float(a2_gscale[expert_idx].item()) if a2_gscale.numel() > 1 else float(a2_gscale.item())
    return _trace_nvfp4_route(
        x_f32=x_f32,
        w1_fp4_eid=w1_fp4[expert_idx],
        w1_blockscale_eid=w1_blockscale[expert_idx],
        alpha_fc1=alpha_fc1,
        w2_fp4_eid=w2_fp4[expert_idx],
        w2_blockscale_eid=w2_blockscale[expert_idx],
        alpha_fc2=alpha_fc2,
        gs_fc1=gs_fc1,
        gs_fc2=gs_fc2,
        K=K,
        I_tp=I_tp,
        token_idx=token_idx,
        route_idx=route_idx,
        expert_idx=expert_idx,
        router_weight=router_weight,
        activation=activation,
    )


def trace_moe_reference_w4a16_route(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    token_idx: int,
    route_idx: int,
    activation: str = "silu",
    swiglu_limit: float | None = None,
) -> MoERouteTrace:
    del E
    _validate_reference_inputs(w1_fp4, I_tp, activation)
    if token_idx < 0 or token_idx >= x.shape[0]:
        raise IndexError(f"token_idx {token_idx} is out of range for batch {x.shape[0]}")
    if route_idx < 0 or route_idx >= topk_ids.shape[1]:
        raise IndexError(f"route_idx {route_idx} is out of range for top_k {topk_ids.shape[1]}")

    expert_idx = int(topk_ids[token_idx, route_idx].item())
    return _trace_w4a16_route(
        x_f32=x[token_idx].float(),
        w1_fp4_eid=w1_fp4[expert_idx],
        w1_blockscale_eid=w1_blockscale[expert_idx],
        alpha_fc1=float(w1_alphas[expert_idx].item()),
        w2_fp4_eid=w2_fp4[expert_idx],
        w2_blockscale_eid=w2_blockscale[expert_idx],
        alpha_fc2=float(w2_alphas[expert_idx].item()),
        K=K,
        I_tp=I_tp,
        token_idx=token_idx,
        route_idx=route_idx,
        expert_idx=expert_idx,
        router_weight=float(topk_weights[token_idx, route_idx].item()),
        activation=activation,
        swiglu_limit=swiglu_limit,
    )


def moe_reference_f32(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    _validate_reference_inputs(w1_fp4, I_tp, activation)
    del E
    is_gated = activation in {"silu", "situ"}
    block_size = 16
    fp8_e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)

    fp4_lut = torch.tensor(
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        dtype=torch.float32,
        device=x.device,
    )
    def dequant_fp4(packed_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        lo = (packed_u8 & 0x0F).to(torch.int64)
        hi = ((packed_u8 >> 4) & 0x0F).to(torch.int64)
        return torch.stack([fp4_lut[lo], fp4_lut[hi]], dim=-1).reshape(rows, cols)

    def apply_block_scales(raw: torch.Tensor, sf_f32: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        n_blocks = cols // block_size
        sf = sf_f32[:rows, :n_blocks]
        return raw * sf.unsqueeze(-1).expand(rows, n_blocks, block_size).reshape(rows, cols)

    def quantize_vec_to_fp4_dequant(vals_f32: torch.Tensor, global_scale: float) -> torch.Tensor:
        cols = vals_f32.shape[0]
        n_blocks = cols // block_size
        blocked = vals_f32.reshape(n_blocks, block_size)
        block_max = blocked.abs().amax(dim=-1)

        raw_scale = (block_max * global_scale / 6.0).clamp(max=fp8_e4m3_max)
        sf_e4m3 = raw_scale.to(torch.float8_e4m3fn).to(torch.float32)

        sf_times_gs = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols) / global_scale
        scaled = vals_f32 / sf_times_gs.clamp(min=1e-30)
        quant = fp4_quantize_values_torch(scaled)
        sf_only = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols)
        return quant * sf_only

    device = x.device
    m = x.shape[0]
    top_k = topk_ids.shape[1]
    output = torch.zeros(m, K, dtype=torch.float32, device=device)

    for t in range(m):
        x_f32 = x[t].float()
        for k_idx in range(top_k):
            eid = int(topk_ids[t, k_idx].item())
            router_w = float(topk_weights[t, k_idx].item())
            alpha_fc1 = float(w1_alphas[eid].item())
            alpha_fc2 = float(w2_alphas[eid].item())

            gs_fc1 = float(a1_gscale[eid].item()) if a1_gscale.numel() > 1 else float(a1_gscale.item())
            gs_fc2 = float(a2_gscale[eid].item()) if a2_gscale.numel() > 1 else float(a2_gscale.item())

            x_dequant = quantize_vec_to_fp4_dequant(x_f32, gs_fc1)

            w2_sf = unswizzle_block_scale(w2_blockscale[eid], K, I_tp // block_size)

            if is_gated:
                w13_sf = unswizzle_block_scale(w1_blockscale[eid], 2 * I_tp, K // block_size)
                up_dequant = apply_block_scales(
                    dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K), w13_sf[:I_tp], I_tp, K,
                )
                gate_dequant = apply_block_scales(
                    dequant_fp4(w1_fp4[eid, I_tp:], I_tp, K), w13_sf[I_tp:], I_tp, K,
                )
                gate_out = (gate_dequant @ x_dequant) * alpha_fc1
                up_out = (up_dequant @ x_dequant) * alpha_fc1
                intermediate = _apply_gated_activation(
                    gate_out,
                    up_out,
                    activation,
                )
            else:
                w1_sf = unswizzle_block_scale(w1_blockscale[eid], I_tp, K // block_size)
                fc1_dequant = apply_block_scales(
                    dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K), w1_sf[:I_tp], I_tp, K,
                )
                fc1_out = (fc1_dequant @ x_dequant) * alpha_fc1
                intermediate = torch.square(torch.relu(fc1_out))

            int_dequant = quantize_vec_to_fp4_dequant(intermediate, gs_fc2)
            down_dequant = apply_block_scales(
                dequant_fp4(w2_fp4[eid], K, I_tp), w2_sf, K, I_tp,
            )
            down_out = (down_dequant @ int_dequant) * alpha_fc2
            output[t] += router_w * down_out

    return output


def moe_reference_w4a16(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    activation: str = "silu",
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    _validate_reference_inputs(w1_fp4, I_tp, activation)
    m = x.shape[0]
    top_k = topk_ids.shape[1]
    output = torch.zeros(m, K, dtype=torch.bfloat16, device=x.device)
    contribs: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(E)]

    for t in range(m):
        for k_idx in range(top_k):
            eid = int(topk_ids[t, k_idx].item())
            trace = trace_moe_reference_w4a16_route(
                x,
                w1_fp4,
                w1_blockscale,
                w1_alphas,
                w2_fp4,
                w2_blockscale,
                w2_alphas,
                topk_ids,
                topk_weights,
                E,
                K,
                I_tp,
                token_idx=t,
                route_idx=k_idx,
                activation=activation,
                swiglu_limit=swiglu_limit,
            )
            assert trace.expert_idx == eid
            contribs[eid].append((t, trace.routed_out))

    for eid in range(E):
        for t, contrib in contribs[eid]:
            output[t] = (output[t].float() + contrib.float()).to(torch.bfloat16)

    return output


def moe_reference_nvfp4(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    _validate_reference_inputs(w1_fp4, I_tp, activation)
    m = x.shape[0]
    top_k = topk_ids.shape[1]
    output = torch.zeros(m, K, dtype=torch.bfloat16, device=x.device)
    contribs: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(E)]

    for t in range(m):
        for k_idx in range(top_k):
            eid = int(topk_ids[t, k_idx].item())
            trace = trace_moe_reference_nvfp4_route(
                x,
                w1_fp4,
                w1_blockscale,
                w1_alphas,
                w2_fp4,
                w2_blockscale,
                w2_alphas,
                a1_gscale,
                a2_gscale,
                topk_ids,
                topk_weights,
                E,
                K,
                I_tp,
                token_idx=t,
                route_idx=k_idx,
                activation=activation,
            )
            assert trace.expert_idx == eid
            contribs[eid].append((t, trace.routed_out))

    for eid in range(E):
        for t, contrib in contribs[eid]:
            output[t] = (output[t].float() + contrib.float()).to(torch.bfloat16)

    return output
