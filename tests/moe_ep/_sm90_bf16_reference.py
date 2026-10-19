"""Torch references for the SM90 native BF16 MegaMoE backend.

``reference_moe_bf16`` reproduces the backend's precision contract exactly:
bf16 activations and weights, fp32 accumulation in both expert GEMMs, SwiGLU
in fp32, the FC1 output rounded to bf16 (the kernel's bf16 intermediate), the
FC2 output rounded to bf16, then the top-k weighted sum in fp32 with fp32
route weights, rounded once to bf16.  Nothing is quantized to FP8.

``reference_moe_fp32`` keeps every intermediate in fp32 (only the final
output is bf16) so both rounding models can be reported side by side.
Routes with ``topk_ids < 0`` are masked and contribute nothing.
"""

from __future__ import annotations

import torch


def _expert_rows(topk_ids: torch.Tensor, num_experts: int):
    """Yield ``(expert, token_index, route_index)`` index tensors per expert."""
    for e in range(num_experts):
        hit = topk_ids == e
        if not bool(hit.any()):
            continue
        token_idx, route_idx = hit.nonzero(as_tuple=True)
        yield e, token_idx, route_idx


def reference_moe_bf16(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    clamp: float | None = None,
) -> torch.Tensor:
    """BF16-contract reference; ``w13`` is canonical ``[E, 2I, H]`` (gate | up)."""
    num_tokens, hidden = x.shape
    num_experts, two_i, _ = w13.shape
    intermediate = two_i // 2
    acc = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)
    ids = topk_ids.to(torch.int64)
    for e, token_idx, route_idx in _expert_rows(ids, num_experts):
        a = x[token_idx].float()
        h = a @ w13[e].float().T  # fp32 accumulate
        gate, up = h[:, :intermediate], h[:, intermediate:]
        if clamp is not None:
            gate = gate.clamp(-clamp, clamp)
            up = up.clamp(-clamp, clamp)
        inter = (torch.nn.functional.silu(gate) * up).to(
            torch.bfloat16
        )  # bf16 intermediate
        y = (inter.float() @ w2[e].float().T).to(torch.bfloat16)  # bf16 expert output
        w = topk_weights[token_idx, route_idx].float().unsqueeze(1)
        acc.index_add_(0, token_idx, y.float() * w)
    return acc.to(torch.bfloat16)


def reference_moe_fp32(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    clamp: float | None = None,
) -> torch.Tensor:
    """All-fp32 reference (no intermediate rounding); output rounded to bf16 once."""
    num_tokens, hidden = x.shape
    num_experts, two_i, _ = w13.shape
    intermediate = two_i // 2
    acc = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)
    ids = topk_ids.to(torch.int64)
    for e, token_idx, route_idx in _expert_rows(ids, num_experts):
        a = x[token_idx].float()
        h = a @ w13[e].float().T
        gate, up = h[:, :intermediate], h[:, intermediate:]
        if clamp is not None:
            gate = gate.clamp(-clamp, clamp)
            up = up.clamp(-clamp, clamp)
        inter = torch.nn.functional.silu(gate) * up
        y = inter @ w2[e].float().T
        w = topk_weights[token_idx, route_idx].float().unsqueeze(1)
        acc.index_add_(0, token_idx, y * w)
    return acc.to(torch.bfloat16)


def compare_bf16(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict[str, float]:
    """Elementwise ``|out - ref| <= atol + rtol * |ref|`` statistics."""
    got = output.float()
    exp = reference.float()
    err = (got - exp).abs()
    tol = atol + rtol * exp.abs()
    bad = int((err > tol).sum().item())
    return {
        "mismatches": bad,
        "numel": int(err.numel()),
        "max_abs_err": float(err.max().item()) if err.numel() else 0.0,
        "mean_abs_err": float(err.mean().item()) if err.numel() else 0.0,
        "rel_l2": float((err.norm() / exp.norm().clamp_min(1e-12)).item())
        if err.numel()
        else 0.0,
    }
