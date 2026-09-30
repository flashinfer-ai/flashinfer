"""Deterministic, untimed routing inputs for MoE candidate comparisons."""

from __future__ import annotations

from collections import Counter
import random

import torch


TUNING_WORKLOAD_VERSION = "shared_0_20_40_60_80_v1"
ROUTING_WORKLOADS = ("disjoint", *(f"shared_{p}" for p in (0, 20, 40, 60, 80, 100)))


def make_routing_ids(
    tokens: int,
    top_k: int,
    num_experts: int,
    *,
    workload: str = "shared_40",
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build routes with distinct experts per token.

    ``shared_N`` reuses N% of the batch's token/expert assignments, rounded
    to the nearest realizable expert count. Reuse favors already popular
    experts, rather than giving every expert exactly one or two tokens.
    One-token batches cannot share; a small expert pool may force more reuse.
    ``disjoint`` retains the cyclic, maximally spread baseline.
    """
    if tokens < 1 or not 1 <= top_k <= num_experts:
        raise ValueError("require positive tokens and 1 <= top_k <= num_experts")
    if workload == "disjoint":
        return (
            torch.arange(tokens * top_k, device=device, dtype=torch.int32)
            .reshape(tokens, top_k)
            .remainder_(num_experts)
        )
    if workload not in ROUTING_WORKLOADS:
        raise ValueError(f"unknown routing workload: {workload!r}")
    return _make_shared_routing_ids(
        tokens, top_k, num_experts, sharing_percent=int(workload.removeprefix("shared_")),
        seed=seed, device=device,
    )


def _make_shared_routing_ids(
    tokens: int,
    top_k: int,
    num_experts: int,
    *,
    sharing_percent: int,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    if tokens < 1 or not 1 <= top_k <= num_experts:
        raise ValueError("require positive tokens and 1 <= top_k <= num_experts")
    unique = min(num_experts, max(
        top_k, (tokens * top_k * (100 - sharing_percent) + 50) // 100
    ))
    rng = random.Random(seed)
    expert_ids = rng.sample(range(num_experts), unique)
    counts: Counter[int] = Counter()
    rows = []
    introduced = 0
    for token in range(tokens):
        target = top_k + (unique - top_k) * token // max(tokens - 1, 1)
        new_count = target - introduced
        row = expert_ids[introduced:target]
        candidates = list(counts)
        for _ in range(top_k - new_count):
            expert = rng.choices(
                candidates, weights=[counts[e] ** 2 for e in candidates], k=1
            )[0]
            row.append(expert)
            candidates.remove(expert)
        rng.shuffle(row)
        rows.append(row)
        counts.update(row)
        introduced = target
    return torch.tensor(rows, dtype=torch.int32, device=device)


def make_tuning_routes(
    tokens: int, top_k: int, num_experts: int, *, device: torch.device | str
) -> torch.Tensor:
    """Cover route-sharing variation in small verification batches.

    Five equally weighted sharing levels retain a nominal 40% mean while
    avoiding selection against a single distinct-expert count. Token and
    expert-pool constraints can clamp the realizable sharing levels.
    """
    if 2 <= tokens <= 8:
        return torch.stack([
            _make_shared_routing_ids(
                tokens, top_k, num_experts, sharing_percent=sharing,
                seed=42 + index, device=device,
            )
            for index, sharing in enumerate((0, 20, 40, 60, 80))
        ])
    return make_routing_ids(
        tokens, top_k, num_experts, workload="disjoint", device=device
    ).unsqueeze(0)
