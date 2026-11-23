# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Independent numerical primitives for the GLM TP8 attention/indexer tests.

FP64 component references round to BF16 at the same boundaries as the kernels.
Attention uses FP32 scores/softmax, BF16 probabilities, then FP32 PV. TP output
sums BF16 rank partials in rank order in FP32 and rounds only after adding the
residual. The ``fp32_*`` helpers reproduce a kernel's FP32 accumulation order
where a stage is checked at that precision.
"""

from __future__ import annotations

import torch

F64 = torch.float64
BF16 = torch.bfloat16
HIDDEN, QLORA, KVLORA = 6144, 2048, 512
NOPE, ROPE, VDIM, HEADS = 192, 64, 256, 10
SELECTED, IDX_HEADS, IDX_DIM, WORLD = 2048, 32, 128, 8
SOFTMAX_SCALE = 0.0625
X_EPS = Q_EPS = KV_EPS = 1e-5
LN_EPS = 1e-6
INV_SQRT128 = 0.08838834613561630249
LOG2E_F32 = 1.4426950216293335


def _f32(v: float) -> float:
    return float(torch.tensor(v, dtype=torch.float32))


def softmax_c(softmax_scale: float = SOFTMAX_SCALE) -> float:
    c = torch.tensor(softmax_scale, dtype=torch.float32) * torch.tensor(
        LOG2E_F32, dtype=torch.float32
    )
    return float(c)


def B(v: torch.Tensor) -> torch.Tensor:
    return v.to(BF16)


def f64(v: torch.Tensor) -> torch.Tensor:
    if v.dtype == torch.float8_e4m3fn:
        return v.to(F64)
    if v.dtype == torch.uint8:
        return v.view(torch.float8_e4m3fn).to(F64)
    return v.to(F64)


def ref_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    xd = f64(x)
    dim = xd.shape[-1]
    ss = (xd * xd).sum(dim=-1, keepdim=True)
    rstd = torch.rsqrt(ss / dim + _f32(eps))
    return B(xd * rstd * f64(gamma))


def _dequant_w8(
    w_fp8: torch.Tensor, scales: torch.Tensor, row_block: int, col_block: int
) -> torch.Tensor:
    wd = f64(w_fp8)
    N, K = (wd.shape[-2], wd.shape[-1])
    sd = f64(scales)
    nrb, nkb = (sd.shape[-2], sd.shape[-1])
    if nkb * col_block != K or nrb != (N + row_block - 1) // row_block:
        raise ValueError(
            f"scale shape {tuple(scales.shape)} inconsistent with W {tuple(w_fp8.shape)} rb={row_block} cb={col_block}"
        )
    s_full = sd.repeat_interleave(row_block, dim=-2)[..., :N, :].repeat_interleave(
        col_block, dim=-1
    )
    return wd * s_full


def ref_gemv_w8a16(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    scales: torch.Tensor,
    row_block: int = 128,
    col_block: int = 128,
) -> torch.Tensor:
    wd = _dequant_w8(w_fp8, scales, row_block, col_block)
    y = torch.matmul(f64(x), wd.transpose(-1, -2))
    return B(y)


def ref_gemv_bf16(x: torch.Tensor, w_bf16: torch.Tensor) -> torch.Tensor:
    return B(torch.matmul(f64(x), f64(w_bf16).transpose(-1, -2)))


def projection(weights: dict, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 weight and FP32 block scales of checkpoint projection ``self_attn.<name>``."""
    prefix = f"self_attn.{name}"
    return weights[f"{prefix}.weight"], weights[f"{prefix}.weight_scale_inv"]


def kv_b_halves(weights: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 W_UK ``[10, 192, 512]`` and W_UV ``[10, 256, 512]`` of each head slot.

    Dequantizes the rank-local ``kv_b_proj`` ``[10 * 448, 512]`` with its
    128 x 128 block scales; each head has 192 W_UK rows, then 256 W_UV rows.
    """
    w = _dequant_w8(*projection(weights, "kv_b_proj"), 128, 128)
    w = w.reshape(HEADS, NOPE + VDIM, KVLORA)
    return w[:, :NOPE], w[:, NOPE:]


def ref_gemv_uk(q_nope_rows: torch.Tensor, weights: dict) -> torch.Tensor:
    """Absorb ``[heads, q, 192]`` query rows into the latent space with W_UK."""
    w_uk, _ = kv_b_halves(weights)
    return B(torch.matmul(f64(q_nope_rows), w_uk))


def ref_gemv_uv(latent: torch.Tensor, weights: dict) -> torch.Tensor:
    """Project ``[heads, q, 512]`` latent outputs to the value space with W_UV."""
    _, w_uv = kv_b_halves(weights)
    return B(torch.matmul(f64(latent), w_uv.transpose(-1, -2)))


def _rope64_f64(v: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    fd = f64(freqs)
    shape = [fd.shape[0]] + [1] * (v.dim() - 2) + [ROPE]
    fd = fd.reshape(shape)
    c, s = (fd[..., 0::2], fd[..., 1::2])
    re, im = (v[..., 0:ROPE:2], v[..., 1:ROPE:2])
    out = v.clone()
    out[..., 0:ROPE:2] = re * c - im * s
    out[..., 1:ROPE:2] = im * c + re * s
    return out


def ref_rope(rows64: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    if rows64.shape[-1] != ROPE:
        raise ValueError("ref_rope expects trailing dim 64")
    return B(_rope64_f64(f64(rows64), freqs))


def hadamard128_f64(v: torch.Tensor) -> torch.Tensor:
    out = v.clone()
    stride = 1
    while stride < IDX_DIM:
        o = out.reshape(*out.shape[:-1], IDX_DIM // (2 * stride), 2, stride)
        a, b = (o[..., 0, :].clone(), o[..., 1, :].clone())
        o[..., 0, :] = a + b
        o[..., 1, :] = a - b
        out = o.reshape(*v.shape)
        stride *= 2
    return out


def ref_rotate_q(iq: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    v = _rope64_f64(f64(iq), freqs)
    return B(hadamard128_f64(v) * INV_SQRT128)


def ref_k_ln_rope_had(
    ki: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    freqs: torch.Tensor,
) -> torch.Tensor:
    v = f64(ki)
    mean = v.mean(dim=-1, keepdim=True)
    xc = v - mean
    var = (xc * xc).mean(dim=-1, keepdim=True)
    y = f64(gamma) * (xc * torch.rsqrt(var + _f32(eps))) + f64(beta)
    y = _rope64_f64(y, freqs)
    return B(hadamard128_f64(y) * INV_SQRT128)


def ref_index_score(
    k_cache: torch.Tensor, L: int, iq_rt: torch.Tensor, idx_scores: torch.Tensor
) -> torch.Tensor:
    kc = f64(k_cache[:L])
    q = f64(iq_rt)
    w = f64(idx_scores)
    dots = torch.matmul(q, kc.transpose(0, 1))
    dots = torch.clamp_min(dots, 0.0)
    logits = (w.unsqueeze(-1) * dots).sum(dim=1) * (1.0 / 64.0)
    return logits.to(torch.float32)


def topk_key(v: torch.Tensor) -> torch.Tensor:
    b = v.to(torch.float32).contiguous().view(torch.int32).to(torch.int64) & 4294967295
    k = torch.where(b & 2147483648 != 0, ~b & 4294967295, b | 2147483648)
    return torch.where(torch.isnan(v), torch.zeros_like(k), k)


def ref_topk_select(logits: torch.Tensor, cur_pos: int, topk: int = SELECTED):
    Tn = logits.shape[0]
    dev = logits.device
    ids = torch.full((Tn, topk), -1, dtype=torch.int32, device=dev)
    lengths = torch.zeros(Tn, dtype=torch.int32, device=dev)
    for t in range(Tn):
        visible = cur_pos + t + 1
        if visible <= topk:
            ids[t, :visible] = torch.arange(visible, dtype=torch.int32, device=dev)
            lengths[t] = visible
            continue
        keys = topk_key(logits[t, :visible])
        _, order = torch.sort(keys, descending=True, stable=True)
        sel = order[:topk]
        ids[t] = torch.sort(sel)[0].to(torch.int32)
        lengths[t] = topk
    return (ids, lengths)


def ref_tp_exit_sum(partials, residual: torch.Tensor | None) -> torch.Tensor:
    acc = torch.zeros(partials[0].shape, dtype=torch.float32)
    for part in partials:
        acc += part.float().cpu()
    if residual is not None:
        acc += residual.float().cpu()
    return acc.to(torch.bfloat16)


def ref_sparse_attention(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    pe_cache: torch.Tensor,
    ids: torch.Tensor | None,
    cur_pos: int,
    topk: int = SELECTED,
    scale: float = SOFTMAX_SCALE,
) -> torch.Tensor:
    T, H, _ = q_nope.shape
    kv, pe = (kv_cache.float(), pe_cache.float())
    rows = kv.shape[0]
    out = torch.zeros(T, H, KVLORA, dtype=torch.float32, device=q_nope.device)
    for t in range(T):
        visible = cur_pos + 1 + t
        logits = (q_nope[t].float() @ kv.T + q_pe[t].float() @ pe.T) * scale
        bias = torch.full((rows,), float("-inf"), device=q_nope.device)
        if visible > topk:
            assert ids is not None, "sparse query needs selected ids"
            bias[ids[t].long()] = 0.0
        else:
            bias[:visible] = 0.0
        p = torch.softmax(logits + bias, dim=-1, dtype=torch.float32)
        out[t] = p.to(torch.bfloat16).float() @ kv
    return out.reshape(T, H * KVLORA).to(torch.bfloat16)


def bf16_ulp_of(mag: torch.Tensor) -> torch.Tensor:
    bits = mag.contiguous().view(torch.int64)
    e = (bits >> 52 & 2047) - 1023
    return (e - 7 + 1023 << 52).view(torch.float64)


def fp32_indexer_rmsnorm(x, gamma):
    """RMSNorm with the indexer kernel's FP32 square-sum order.

    For each 2048-column chunk, 256 lanes each add four squared pairs; a warp
    shuffle reduction and an ordered sum over the eight warps follow.
    """
    q_len = x.shape[0]
    cols = x.shape[-1]
    chunks = cols // 2048
    words = x.float().reshape(q_len, chunks, 256, 8)
    s = torch.zeros((q_len, 256), dtype=torch.float32, device=x.device)
    for c in range(chunks):
        for j in range(0, 8, 2):
            s = s + (words[:, c, :, j].square() + words[:, c, :, j + 1].square())
    s = s.reshape(q_len, 8, 32)
    for off in [16, 8, 4, 2, 1]:
        shifted = torch.cat([s[:, :, off:], s[:, :, -off:]], dim=-1)
        s = s + shifted
    total = torch.zeros((q_len, 1), dtype=torch.float32, device=x.device)
    for w in range(8):
        total = total + s[:, w, 0, None]
    mean = (total.double() / cols).float()
    rstd = torch.rsqrt(mean + 1e-05)
    return (x.float() * (gamma.float()[None, :] * rstd)).bfloat16()


def fp32_attention_rmsnorm(x, gamma, eps=1e-05):
    """q4 MLA input RMSNorm with the kernel's FP32 square-sum order.

    Each of 256 lanes adds eight squares from each of three 2048-column chunks;
    a warp shuffle reduction and an ordered sum over the eight warps follow.
    """
    q_len = x.shape[0]
    words = x.float().reshape(q_len, 3, 256, 8)
    s = torch.zeros((q_len, 256), dtype=torch.float32, device=x.device)
    for c in range(3):
        for j in range(8):
            s = s + words[:, c, :, j].square()
    s = s.reshape(q_len, 8, 32)
    for off in [16, 8, 4, 2, 1]:
        s = s + torch.cat([s[:, :, off:], s[:, :, -off:]], dim=-1)
    total = s[:, 0, 0, None]
    for w in range(1, 8):
        total = total + s[:, w, 0, None]
    inv = torch.tensor(1.0 / 6144, dtype=torch.float32, device=x.device)
    ep = torch.tensor(eps, dtype=torch.float32, device=x.device)
    mean_eps = (total.double() * inv.double() + ep.double()).float()
    rstd = torch.rsqrt(mean_eps)
    return (x.float() * rstd * gamma.float()[None, :]).bfloat16()


def fp32_o_proj(x, weights, scales):
    """Output projection with the q4 kernel's FP32 accumulation order.

    Each of eight warps accumulates scaled 64-column partial products over five
    512-column pages; the eight warp partials are then summed in order.
    """
    q_len = x.shape[0]
    assert x.dtype == torch.bfloat16 and x.shape == (q_len, 2560)
    assert weights.shape == (6144, 2560) and scales.shape == (48, 20)
    w = (
        weights.view(torch.float8_e4m3fn).bfloat16()
        if weights.dtype == torch.uint8
        else weights.bfloat16()
    )
    row = torch.arange(6144, device=x.device) // 128
    partial = [
        torch.zeros((q_len, 6144), dtype=torch.float32, device=x.device)
        for _ in range(8)
    ]
    for page in range(5):
        for warp in range(8):
            col = page * 512 + warp * 64
            dot = torch.mm(
                x[:, col : col + 64].contiguous(),
                w[:, col : col + 64].t().contiguous(),
                out_dtype=torch.float32,
            )
            assert dot.dtype == torch.float32
            scale = scales[row, col // 128].float()[None, :]
            partial[warp] = (
                dot.double() * scale.double() + partial[warp].double()
            ).float()
    total = torch.zeros_like(partial[0])
    for warp in range(8):
        total = total + partial[warp]
    return total.bfloat16()


def assert_close(actual, expected, budget, floor_fraction, *, per_row=False):
    """Max error in BF16 ULPs of max(|ref|, floor_fraction * max|ref|)."""
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    out, ref = actual.double(), expected.double()
    if per_row:
        out, ref = out.reshape(-1, out.shape[-1]), ref.reshape(-1, ref.shape[-1])
    else:
        out, ref = out.reshape(1, -1), ref.reshape(1, -1)
    magnitude = torch.maximum(
        ref.abs(), floor_fraction * ref.abs().amax(-1, keepdim=True)
    ).clamp_min(2**-126)
    error = (out - ref).abs() / bf16_ulp_of(magnitude)
    assert bool(torch.isfinite(error).all()), "nonfinite attention comparison"
    peak = float(error.max())
    assert peak <= budget, {
        "max_floored_bf16_ulp": peak,
        "budget": budget,
        "floor_fraction": floor_fraction,
    }
    return peak


def assert_rope_close(actual, expected, budget=2, floor_fraction=2**-14):
    """BF16 ULP error of RoPE pairs, scaled by the pair norm (rotation invariant)."""
    out, ref = actual.double().reshape(-1, 2), expected.double().reshape(-1, 2)
    pair = ref.norm(dim=-1, keepdim=True).expand_as(ref)
    magnitude = torch.maximum(
        torch.maximum(ref.abs(), pair), floor_fraction * ref.abs().amax()
    ).clamp_min(2**-126)
    error = (out - ref).abs() / bf16_ulp_of(magnitude)
    assert bool(torch.isfinite(error).all()), "nonfinite RoPE comparison"
    peak = float(error.max())
    assert peak <= budget, {"max_rope_pair_bf16_ulp": peak, "budget": budget}
    return peak


def check_exact_topk(logits, selected_ids, cur_pos, q_len):
    """Check the exact causal top-2048 set; equal-threshold order is not required."""
    result = selected_ids[:q_len].clone()
    for token in range(q_len):
        visible = cur_pos + token + 1
        if visible <= SELECTED:
            # These rows need no publication: the MLA kernel generates its own
            # dense causal prefix, even if old selection storage is untouched.
            result[token].fill_(-1)
            result[token, :visible] = torch.arange(visible, device=result.device)
            continue
        ids = result[token].long()
        assert bool(((ids >= 0) & (ids < visible)).all())
        assert torch.unique(ids).numel() == SELECTED
        values = logits[token, :visible]
        threshold = torch.topk(values, SELECTED).values[-1]
        present = torch.zeros(visible, dtype=torch.bool, device=values.device)
        present[ids] = True
        assert bool((values[ids] >= threshold).all()), (
            "selected a value below the exact threshold"
        )
        assert bool(present[values > threshold].all()), (
            "omitted a value above the exact threshold"
        )
    return result


def attention_pipeline(weights, state, x, cur_pos, freqs, ids):
    """Independent MLA reference up to the latent attention output.

    ``weights`` holds the rank-local checkpoint tensors (see
    ``shard_attention_weights``). q4 normalizes the input with the kernel's
    FP32 square-sum order; q1 uses the FP64 RMSNorm reference. Every
    projection and normalization rounds to BF16. Callers compare the
    intermediate stages; an end-to-end FP64 output is not used as a pass/fail
    criterion.
    """
    w, q_len = weights, x.shape[0]
    gamma = w["input_layernorm.weight"]
    xn = (
        fp32_attention_rmsnorm(x, gamma) if q_len == 4 else ref_rmsnorm(x, gamma, X_EPS)
    )
    qa = ref_gemv_w8a16(xn, *projection(w, "q_a_proj"))
    kv_pe = ref_gemv_w8a16(xn, *projection(w, "kv_a_proj_with_mqa"))
    kv, pe = kv_pe[:, :KVLORA].contiguous(), kv_pe[:, KVLORA:].contiguous()
    kv_row = ref_rmsnorm(kv, w["self_attn.kv_a_layernorm.weight"], KV_EPS)
    pe_row = ref_rope(pe, freqs)
    qn = ref_rmsnorm(qa, w["self_attn.q_a_layernorm.weight"], Q_EPS)
    qb = ref_gemv_w8a16(qn, *projection(w, "q_b_proj")).reshape(
        q_len, HEADS, NOPE + ROPE
    )
    q_pe = ref_rope(qb[..., NOPE:].contiguous(), freqs)
    q_nd = qb[..., :NOPE].permute(1, 0, 2).contiguous()
    q_nope = ref_gemv_uk(q_nd, w).permute(1, 0, 2).contiguous()
    kv_cache, pe_cache = state["kv_cache"].clone(), state["pe_cache"].clone()
    kv_cache[cur_pos : cur_pos + q_len] = kv_row
    pe_cache[cur_pos : cur_pos + q_len] = pe_row
    latent = ref_sparse_attention(q_nope, q_pe, kv_cache, pe_cache, ids, cur_pos)
    return {
        "xn": xn,
        "kv": kv,
        "pe_row": pe_row,
        "latent": latent.reshape(q_len, HEADS, KVLORA),
    }
