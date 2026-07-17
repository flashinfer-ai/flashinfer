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

Pure-torch reference (oracle) for SM90 MoE, plus dequant + metric helpers.
"""

import torch
import torch.nn.functional as F


def reference_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Full MoE forward in fp32 on a single device (all experts present)."""
    T, H = hidden_states.shape
    E, two_i, h2 = w13.shape
    assert h2 == H and two_i % 2 == 0, f"w13 {tuple(w13.shape)} vs H={H}"
    I = two_i // 2
    assert tuple(w2.shape) == (E, H, I), f"w2 {tuple(w2.shape)} vs ({E},{H},{I})"

    x = hidden_states.to(compute_dtype)
    w13 = w13.to(compute_dtype)
    w2 = w2.to(compute_dtype)
    out = torch.zeros(T, H, dtype=compute_dtype, device=hidden_states.device)

    for e in range(E):
        sel = topk_ids == e  # (T, top_k) bool
        if not bool(sel.any()):
            continue
        tok_idx, slot_idx = sel.nonzero(as_tuple=True)  # tokens routed to expert e
        xe = x[tok_idx]  # (Me, H)
        fc1 = xe @ w13[e].T  # (Me, 2I)
        gate, up = fc1[:, :I], fc1[:, I:]
        act = F.silu(gate) * up  # (Me, I)
        fc2 = act @ w2[e].T  # (Me, H)
        w = topk_weights[tok_idx, slot_idx].unsqueeze(1).to(compute_dtype)
        out.index_add_(0, tok_idx, fc2 * w)

    return out


def quant_dequant_128x128(w: torch.Tensor) -> torch.Tensor:
    """Pure-torch 128x128 fp8-blockscale quantize->dequantize of ONE matrix."""
    n, k = w.shape
    assert n % 128 == 0 and k % 128 == 0, f"weight ({n},{k}) not 128-aligned"
    t = w.float().reshape(n // 128, 128, k // 128, 128)
    amax = t.abs().amax(dim=(1, 3))
    sc = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    q = (t / sc[:, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (q.float() * sc[:, None, :, None]).reshape(n, k)


def reference_moe_fp8_weights_streaming(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """reference_moe on fp8-blockscale-dequantized weights (streaming, low-mem)."""
    T, H = hidden_states.shape
    E, two_i, h2 = w13.shape
    assert h2 == H and two_i % 2 == 0
    I = two_i // 2
    assert tuple(w2.shape) == (E, H, I)

    x = hidden_states.float()
    out = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)
    for e in range(E):
        sel = topk_ids == e
        if not bool(sel.any()):
            continue
        tok_idx, slot_idx = sel.nonzero(as_tuple=True)
        w13d = quant_dequant_128x128(w13[e])
        w2d = quant_dequant_128x128(w2[e])
        fc1 = x[tok_idx] @ w13d.T
        act = F.silu(fc1[:, :I]) * fc1[:, I:]
        fc2 = act @ w2d.T
        w = topk_weights[tok_idx, slot_idx].unsqueeze(1).float()
        out.index_add_(0, tok_idx, fc2 * w)
    return out


def dequant_act_1x128(a_fp8: torch.Tensor, a_sf: torch.Tensor) -> torch.Tensor:
    """Dequantize 1x128 per-token-per-128-channel FP8 activations -> fp32 (M, K)."""
    m, k = a_fp8.shape
    assert k % 128 == 0, f"K={k} not a multiple of 128"
    assert tuple(a_sf.shape) == (m, k // 128), (
        f"a_sf {tuple(a_sf.shape)} != {(m, k // 128)}"
    )
    return (a_fp8.float().reshape(m, k // 128, 128) * a_sf.unsqueeze(-1)).reshape(m, k)


def dequant_weight_128x128(w_fp8: torch.Tensor, w_sf: torch.Tensor) -> torch.Tensor:
    """Dequantize 128x128 per-block FP8 weights -> fp32 (N, K)."""
    n, k = w_fp8.shape
    assert n % 128 == 0 and k % 128 == 0, f"weight ({n},{k}) not 128-aligned"
    assert tuple(w_sf.shape) == (n // 128, k // 128), (
        f"w_sf {tuple(w_sf.shape)} != {(n // 128, k // 128)}"
    )
    return (
        w_fp8.float().reshape(n // 128, 128, k // 128, 128)
        * w_sf.reshape(n // 128, 1, k // 128, 1)
    ).reshape(n, k)


def compare(
    out: torch.Tensor,
    ref: torch.Tensor,
    *,
    err_ratio: float = 0.15,
    cos_min: float = 0.98,
) -> dict:
    """Error metrics + PASS flag, judged on NORMALIZED error (err_rms / ref_rms)"""
    out = out.float()
    ref = ref.float()
    abs_err = (out - ref).abs()
    nrm = ref.pow(2).mean().sqrt().clamp_min(1e-6)
    e_rms = (out - ref).pow(2).mean().sqrt()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    m = {
        "max_abs": abs_err.max().item(),
        "mean_abs": abs_err.mean().item(),
        "err_ratio": float(e_rms / nrm),
        "cos": float(cos),
    }
    m["passed"] = (m["err_ratio"] < err_ratio) and (m["cos"] > cos_min)
    return m
