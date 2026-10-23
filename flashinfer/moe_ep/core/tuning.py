"""Format version and token bucketing shared by the offline MegaMoE caches."""

from typing import Any

# Every backend shares one cache file; bump this for all of them together.
KNOB_CACHE_VERSION = 2


def tuning_live_tokens(tokens: int | None, capacity: int, group: Any = None) -> int:
    """Use the same EP-wide workload bucket that serving uses for selection."""
    import torch
    import torch.distributed as dist

    count = capacity if tokens is None else tokens
    if dist.is_available() and dist.is_initialized():
        value = torch.tensor(count, device="cuda", dtype=torch.int64)
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
        count = int(value.item())
    return count


def live_buckets(entries: list[dict[str, Any]], capacity: int) -> list[int]:
    return sorted(
        {
            e.get("live_tokens", capacity)
            for e in entries
            if e["max_tokens"] == capacity
            and type(e.get("live_tokens", capacity)) is int
            and 0 <= e.get("live_tokens", capacity) <= capacity
        }
    )


def select_token_entries(
    entries: list[dict[str, Any]], capacity: int, live_tokens: int | None
) -> list[dict[str, Any]]:
    """Select a live bucket at exact capacity, or legacy capacity candidates."""
    if live_tokens is not None and (
        type(live_tokens) is not int or not 0 <= live_tokens <= capacity
    ):
        raise ValueError("live_tokens must be between zero and max_tokens")
    buckets = live_buckets(entries, capacity)
    if buckets:
        workload = capacity if live_tokens is None else live_tokens
        bucket = next((b for b in buckets if b >= workload), buckets[-1])
        return [
            e
            for e in entries
            if e["max_tokens"] == capacity and e.get("live_tokens", capacity) == bucket
        ]
    # Old files had no live-token dimension. Full-capacity records have the
    # same meaning, so both retain the old capacity-rounding fallback.
    return [
        e
        for e in entries
        if e.get("live_tokens") is None or e.get("live_tokens") == e["max_tokens"]
    ]
