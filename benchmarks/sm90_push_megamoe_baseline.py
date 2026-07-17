"""

Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Non-fused SM90 MoE baseline compute path (local grouped GEMMs).
"""

from typing import Dict, Tuple

import torch

COMPACT_EMPTY_EXPERTS = False  # the grouped GEMM handles zero-token groups
EXPERT_PAD = 1  # the kernel needs no per-expert row alignment

_runner = None
_weight_cache: Dict[int, Tuple[torch.Tensor, ...]] = {}


def get_runner():
    global _runner
    if _runner is None:
        from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim.gemm import (
            create_sm90_push_fp8_moe_gemm_runner,
        )

        _runner = create_sm90_push_fp8_moe_gemm_runner()
    return _runner


def quant_weights(w13: torch.Tensor, w2: torch.Tensor):
    """Per-expert 128x128 block quant, cached by weight identity (offline cost)."""
    key = w13.data_ptr()
    if key in _weight_cache:
        return _weight_cache[key]
    from flashinfer.testing.utils import per_block_cast_to_fp8

    E, two_i, H = w13.shape
    _, H2, I = w2.shape
    assert H2 == H
    w13_fp8 = torch.empty(E, two_i, H, device=w13.device, dtype=torch.float8_e4m3fn)
    w13_sf = torch.empty(
        E, two_i // 128, H // 128, device=w13.device, dtype=torch.float32
    )
    w2_fp8 = torch.empty(E, H, I, device=w2.device, dtype=torch.float8_e4m3fn)
    w2_sf = torch.empty(E, H // 128, I // 128, device=w2.device, dtype=torch.float32)
    for e in range(E):
        f, s = per_block_cast_to_fp8(w13[e])
        w13_fp8[e].copy_(f)
        w13_sf[e].copy_(s)
        f, s = per_block_cast_to_fp8(w2[e])
        w2_fp8[e].copy_(f)
        w2_sf[e].copy_(s)
    _weight_cache[key] = (w13_fp8, w13_sf, w2_fp8, w2_sf)
    return _weight_cache[key]


def quant_act(runner, x_bf16: torch.Tensor):
    """1x128 activation quant via the runner's bound fp8_quantize_1x128."""
    m, k = x_bf16.shape
    a = torch.empty(m, k, device=x_bf16.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty(m, k // 128, device=x_bf16.device, dtype=torch.float32)
    runner.fp8_quantize_1x128(x_bf16.contiguous(), a, sf, True)
    return a, sf


def padded_offset(off: int, g: int) -> int:
    """Mirror of deep_gemm::compute_padded_offset (32-align with group skew)."""
    return (off + g * 31) // 32 * 32


def quant_act_grouped(x_bf16: torch.Tensor, offsets: torch.Tensor):
    """1x128 activation quant + SFA packing for the GROUPED external-scale path."""
    M, K = x_bf16.shape
    nkb = K // 128
    G = offsets.numel() - 1
    off = offsets.tolist()
    P = max(padded_offset(off[-1], G), 1)
    xb = x_bf16.float().reshape(M, nkb, 128)
    amax = xb.abs().amax(dim=-1)
    sc = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))  # dequant scale
    q = (xb / sc.unsqueeze(-1)).clamp(-448, 448).reshape(M, K).to(torch.float8_e4m3fn)
    sfa = torch.zeros(nkb, P, dtype=torch.float32, device=x_bf16.device)
    for g in range(G):
        s, t = off[g], off[g + 1]
        if t > s:
            ps = padded_offset(s, g)
            sfa[:, ps : ps + (t - s)] = sc[s:t].T
    return q, sfa.contiguous(), sc


def grouped_ffn(
    runner,
    x_rep: torch.Tensor,  # (M, H) bf16, rows expert-contiguous per `offsets`
    offsets: torch.Tensor,  # (G+1,) int64 cumulative per-expert row offsets
    w13_fp8: torch.Tensor,
    w13_sf: torch.Tensor,  # (G, 2I, H) / (G, 2I/128, H/128)
    w2_fp8: torch.Tensor,
    w2_sf: torch.Tensor,  # (G, H, I)  / (G, H/128,  I/128)
) -> torch.Tensor:
    """FC1 -> SwiGLU -> FC2 over expert-contiguous rows. Returns (M, H) bf16."""
    from flashinfer.activation import silu_and_mul

    M, H = x_rep.shape
    G, two_i, _ = w13_fp8.shape
    I = two_i // 2
    dev = x_rep.device
    if M == 0:
        return torch.empty(0, H, device=dev, dtype=torch.bfloat16)

    sz = runner.get_moe_workspace_size(M, M, max(two_i, H), max(H, I), G, True, True)
    runner.configure_workspace(
        torch.empty(max(int(sz), 1), device=dev, dtype=torch.uint8)
    )

    a1, a1_sfa, _ = quant_act_grouped(x_rep, offsets)
    h = torch.empty(M, two_i, device=dev, dtype=torch.bfloat16)
    runner.moe_gemm(h, a1, w13_fp8, offsets, two_i, H, a1_sfa, w13_sf, False)

    g = silu_and_mul(h)  # gate = h[:, :I], up = h[:, I:] -> (M, I)

    a2, a2_sfa, _ = quant_act_grouped(g, offsets)
    y = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    runner.moe_gemm(y, a2, w2_fp8, offsets, H, I, a2_sfa, w2_sf, False)
    return y


def build_expert_contiguous(
    topk_ids: torch.Tensor,  # (T, k) int
    topk_weights: torch.Tensor,  # (T, k) float
    num_experts: int,
):
    """Sort (token, k) pairs by expert. Returns (tok_of, w_of, counts, offsets)."""
    T, k = topk_ids.shape
    dev = topk_ids.device
    flat = topk_ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    tok_of = torch.arange(T, device=dev).repeat_interleave(k)[order]
    w_of = topk_weights.reshape(-1)[order]
    counts = torch.bincount(flat, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=dev)
    offsets[1:] = counts.cumsum(0)
    return tok_of, w_of, counts, offsets


def sm90_moe_baseline_local(inp) -> torch.Tensor:
    """inp: namespace with hidden_states/w13/w2/topk_ids/topk_weights; returns (T, H) fp32."""
    runner = get_runner()
    hidden_states, w13, w2 = inp.hidden_states, inp.w13, inp.w2
    T, H = hidden_states.shape
    E = w13.shape[0]

    w13_fp8, w13_sf, w2_fp8, w2_sf = quant_weights(w13, w2)
    tok_of, w_of, counts, offsets = build_expert_contiguous(
        inp.topk_ids, inp.topk_weights, E
    )

    if EXPERT_PAD > 1:
        raise NotImplementedError("expert padding is not implemented")
    if COMPACT_EMPTY_EXPERTS:
        nz = (counts > 0).nonzero(as_tuple=True)[0]
        offsets = torch.zeros(
            nz.numel() + 1, dtype=torch.int64, device=hidden_states.device
        )
        offsets[1:] = counts[nz].cumsum(0)
        w13_fp8, w13_sf = w13_fp8[nz], w13_sf[nz]
        w2_fp8, w2_sf = w2_fp8[nz], w2_sf[nz]

    x_rep = hidden_states[tok_of]  # (T*k, H) bf16, expert-contiguous
    y = grouped_ffn(runner, x_rep, offsets, w13_fp8, w13_sf, w2_fp8, w2_sf)

    out = torch.zeros(T, H, device=hidden_states.device, dtype=torch.float32)
    out.index_add_(0, tok_of, y.float() * w_of.unsqueeze(1).float())
    return out
