# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial finite FP32 rows and an independent full-key set oracle."""

import torch

from .attention_reference import topk_key


def make_scores(pattern, capacity, past, device, phase=0):
    ids = torch.arange(capacity, dtype=torch.int64, device=device)
    rows = []
    for query in range(4):
        code = (ids * 104729 + 101 * phase + 17 * query) % 4096
        if pattern == "fp32_collision":
            values = 1.0 + code.float() * 2**-23
        elif pattern == "negative_collision":
            values = -1.0 - code.float() * 2**-23
        elif pattern == "wide":
            values = ((ids * 37 + phase) % 257 - 128).float() * 2.0**96
        elif pattern in ("signed_zero", "high_ties"):
            start = (1 << 18) + 17 + phase
            if pattern == "signed_zero":
                values = (
                    torch.where(ids >= start, 0, -2147483648).int().view(torch.float32)
                )
            else:
                values = torch.where(ids >= start, 1.0 + phase, -1.0)
        elif pattern == "all_equal":
            # Equal keys force the exact full-row fallback after pool overflow.
            values = torch.full((capacity,), 0.5 + phase, device=device)
        elif pattern == "group_spikes":
            peak = ids % 32 == (phase + query) % 32
            score = ((ids // 32) * 15485863 + phase * 101) % 1048573
            values = torch.where(peak, score.float() + 1, -(ids % 1024).float() - 1)
        elif pattern == "subnormal":
            values = (code.int() + 1).view(torch.float32)
        elif pattern == "ascending":
            values = ids.float() if phase % 2 == 0 else -ids.float()
        else:
            raise ValueError(pattern)
        values = values.float().contiguous()
        # These finite, dominating values must never be selected past the causal end.
        values[past + query + 1 :] = torch.finfo(torch.float32).max
        rows.append(values)
    return torch.stack(rows)


def exact_ids(values, k=2048):
    """Rank unsigned key32, then ID; never sort u64 composites as signed int64."""
    keys = topk_key(values)
    cutoff = torch.topk(keys, k).values[-1]
    above = torch.nonzero(keys > cutoff, as_tuple=False).flatten()
    tied = torch.nonzero(keys == cutoff, as_tuple=False).flatten()
    return torch.cat((above, tied[: k - above.numel()])).sort().values


def check_selection(logits, selected, past):
    assert logits.dtype == torch.float32 and selected.shape == (4, 2048)
    for query in range(4):
        visible = past + query + 1
        ids = selected[query].long()
        if visible <= 2048:
            assert torch.equal(ids[:visible], torch.arange(visible, device=ids.device))
            assert bool((ids[visible:] == -1).all())
        else:
            values = logits[query, :visible]
            assert bool(torch.isfinite(values).all())
            assert bool(((ids >= 0) & (ids < visible)).all())
            assert torch.unique(ids).numel() == 2048
            assert torch.equal(ids.sort().values, exact_ids(values)), (
                "full FP32 ordering key / lower-ID tie rule",
                query,
                past,
            )
