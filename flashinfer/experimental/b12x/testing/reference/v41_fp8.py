"""Independent reference for native V4.1 FP8 operands and 64-key split math."""

import torch


def canonical_fp8_rows(payload: torch.Tensor, kind: str):
    """Return unscaled E4M3 values and per-64 scales from compact native rows."""
    if kind == "swa":
        rows = payload.reshape(-1, 528)
        codes = rows[:, :512].contiguous().view(torch.float8_e4m3fn).double()
        original = rows[:, 512:528].double()
        exponent = original.reshape(-1, 8, 2).amax(-1).clamp_min(1) - 127
        normalized = codes * torch.exp2(original - 127).repeat_interleave(32, dim=1)
    else:
        rows = payload.reshape(-1, 288)
        packed = rows[:, :256].long()
        codes = torch.stack((packed & 15, packed >> 4), -1).flatten(1)
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float64,
            device=rows.device,
        )
        original = rows[:, 256:288].contiguous().view(torch.float8_e4m3fn).double()
        normalized = lut[codes] * original.repeat_interleave(16, dim=1)
        bound = original.reshape(-1, 8, 4).amax(-1) * (6 / 448)
        exponent = torch.ceil(torch.log2(bound.clamp_min(2.0**-126)))
    scales = torch.exp2(exponent)
    normalized = normalized / scales.repeat_interleave(64, dim=1)
    values = normalized.float().to(torch.float8_e4m3fn).float()
    return values, scales.float()


def split64_fp8_attention(
    q, key_values, key_scales, valid, sm_scale, attn_sink=None,
    *, qk_fp8=True, round_split_outputs=True,
):
    """FP8 Q/K dot and per-output-group FP8 probability/V products.

    The caller supplies one independently normalized split per 64 candidates,
    matching the test plans. Partial outputs round to BF16 before the LSE merge.
    """
    rows, heads, dim = q.shape
    qdq = q.float()
    if qk_fp8:
        grouped_q = qdq.reshape(rows, heads, dim // 64, 64)
        raw = grouped_q.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
        qscale = torch.exp2(torch.ceil(torch.log2(raw)))
        qdq = ((grouped_q / qscale).to(torch.float8_e4m3fn).float() * qscale).reshape_as(q)
    kdq = key_values * key_scales.repeat_interleave(64, dim=-1)
    if not qk_fp8:
        kdq = kdq.bfloat16().float()
    logits = torch.einsum("mhd,mkd->mhk", qdq, kdq) * sm_scale
    logits = logits.masked_fill(~valid[:, None], -torch.inf)
    partials, lses = [], []
    for first in range(0, key_values.shape[1], 64):
        last = first + 64
        local = logits[:, :, first:last]
        maximum = local.amax(-1)
        probability = torch.exp(local - maximum[:, :, None])
        probability = torch.where(valid[:, None, first:last], probability, 0)
        denominator = probability.sum(-1)
        groups = []
        for group in range(dim // 64):
            weighted = probability * key_scales[:, None, first:last, group]
            wscale = weighted.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
            wq = (weighted / wscale).to(torch.float8_e4m3fn).float()
            vq = key_values[:, first:last, group * 64 : (group + 1) * 64]
            product = torch.einsum("mhk,mkd->mhd", wq, vq) * wscale
            groups.append(product)
        partial = torch.cat(groups, -1) / denominator.clamp_min(1e-30)[:, :, None]
        partials.append(partial.bfloat16().float() if round_split_outputs else partial)
        lses.append(
            torch.where(denominator > 0, maximum + denominator.log(), -torch.inf)
        )
    lse = torch.stack(lses, -1)
    maximum = lse.amax(-1)
    if attn_sink is not None:
        maximum = torch.maximum(maximum, attn_sink.float()[None])
    weights = torch.where(torch.isfinite(lse), torch.exp(lse - maximum[:, :, None]), 0)
    denominator = weights.sum(-1)
    if attn_sink is not None:
        denominator = denominator + torch.exp(attn_sink.float()[None] - maximum)
    output = (torch.stack(partials, -2) * weights[..., None]).sum(-2)
    output = output / denominator.clamp_min(1e-30)[..., None]
    final_lse = torch.where(denominator > 0, maximum + denominator.log(), -torch.inf)
    return output.bfloat16(), final_lse
