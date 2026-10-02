"""Pure-torch oracle for the SM90 pull-style megakernel with BF16 operands.

Mirrors the kernel's compute graph (the drop's ``deepgemm`` graph) with
FP32 math and the kernel's two BF16 rounding points:

* FC1 ``x @ w13[e].T`` in FP32; gate clamped to ``<= clamp``, up to
  ``[-clamp, clamp]``; ``up * silu(gate)`` times the routing weight, then
  rounded to BF16 (the FC1 output the FC2 GEMM reads).
* FC2 in FP32, rounded to BF16 per ``(token, topk)`` term (the combine wire).
* The top-k terms summed in FP32 (``TopkReduce``).

Masked (``-1``) routes contribute nothing.  Independent of every kernel tree.
"""

from __future__ import annotations

from typing import Optional

import torch


def sm90_bf16_moe_reference(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    gate_up_clamp: Optional[float] = None,
) -> torch.Tensor:
    """FP32 ``(T, H)`` output; ``w13`` ``(E, 2I, H)`` gate-first, ``w2`` ``(E, H, I)``.

    Expert ids index ``w13``/``w2`` directly (pass the GLOBAL expert bank for
    multi-rank checks).  Weights may be any float dtype (e.g. dequantized).
    """
    num_tokens, top_k = topk_ids.shape
    intermediate = w2.shape[-1]
    x = hidden_states.float()
    terms = torch.zeros(
        num_tokens, top_k, w2.shape[1], dtype=torch.float32, device=x.device
    )
    for expert in range(w13.shape[0]):
        token_idx, slot_idx = (topk_ids == expert).nonzero(as_tuple=True)
        if token_idx.numel() == 0:
            continue
        h = x[token_idx] @ w13[expert].float().T
        gate, up = h[:, :intermediate], h[:, intermediate:]
        if gate_up_clamp is not None:
            gate = gate.clamp(max=gate_up_clamp)
            up = up.clamp(-gate_up_clamp, gate_up_clamp)
        route = topk_weights[token_idx, slot_idx].float()[:, None]
        fc1_out = (up * gate * torch.sigmoid(gate) * route).to(torch.bfloat16)
        fc2 = fc1_out.float() @ w2[expert].float().T
        terms[token_idx, slot_idx] = fc2.to(torch.bfloat16).float()
    return terms.sum(dim=1)


def assert_bf16_close(
    output: torch.Tensor, reference: torch.Tensor, *, label: str = ""
) -> float:
    """BF16-level agreement for O(1) outputs; returns rel_l2.

    Kernel and oracle round at the same points but accumulate in different
    orders, so a BF16 rounding boundary can flip by one ulp (2**-7 relative)
    and propagate through FC2; well inside 2e-2 abs + 2e-2 rel.
    """
    out = output.float()
    ref = reference.float()
    assert torch.isfinite(out).all(), f"{label}: non-finite output"
    rel_l2 = ((out - ref).norm() / ref.norm().clamp_min(1e-6)).item()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2, msg=label)
    assert rel_l2 < 5e-3, (label, rel_l2)
    return rel_l2
