"""Allocation-using mathematical oracle; never used by the native runtime.

Adapted math: DeepSeek-V4.1-Flash inference/model.py:281-293,429-485,
snapshot fb2764a5cf321eaa5070ca8f9e892818f477c16d. Streaming metadata extends
the model's uniform-batch example to packed independently advancing requests.
"""
from __future__ import annotations

import torch


def normalized_latent(values: torch.Tensor, weight: torch.Tensor,
                      gates: torch.Tensor | None = None) -> torch.Tensor:
    """Pool per channel, round BF16, then ordinary weighted FP32 RMSNorm."""
    pooled = values if gates is None else (values.float() * gates.float().softmax(dim=0)).sum(dim=0)
    rounded = pooled.to(torch.bfloat16).float()
    return (rounded * torch.rsqrt(rounded.square().mean(dim=-1, keepdim=True) + 1e-20)
            * weight.float()).to(torch.bfloat16)


def streaming_reference(*, values, gates, weight, starts, positions, state_ids,
                        slots, ratio, state):
    """Return fixed-row outputs and mutate a CPU-oracle state dictionary.

    ``state`` maps stable IDs to (absolute_even_position, value, gate). Starts
    and the other metadata are host lists of the live packed request rows.
    """
    out = torch.zeros(values.shape, dtype=torch.bfloat16, device=values.device)
    emitted = torch.zeros(values.shape[0], dtype=torch.bool, device=values.device)
    emitted_slots = torch.full((values.shape[0],), -1, dtype=torch.int64, device=values.device)
    for r, sid in enumerate(state_ids):
        if sid < 0 or positions[r] < 0:
            continue
        for token in range(starts[r], starts[r + 1]):
            pos = positions[r] + token - starts[r]
            if ratio == 1:
                out[token] = normalized_latent(values[token], weight)
            elif pos % 2 == 0:
                state[sid] = (pos, values[token].clone(), gates[token].clone())
                continue
            else:
                pending = state.pop(sid, None)
                if pending is None or pending[0] != pos - 1:
                    continue
                out[token] = normalized_latent(
                    torch.stack((pending[1], values[token])), weight,
                    torch.stack((pending[2], gates[token])))
            emitted[token] = True
            emitted_slots[token] = slots[token]
    return out, emitted, emitted_slots
