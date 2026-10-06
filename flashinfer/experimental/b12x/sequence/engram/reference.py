"""Explicit CPU integer oracle, never a runtime fallback."""
import torch

from .geometry import Geometry


def hash_reference(tokens, mask, starts, slots, history, token_map,
                   geometry: Geometry, layer_id: int) -> torch.Tensor:
    """Reference cache/gather equation from DeepSeek inference/engram.py 163-184."""
    layer = geometry.layer_ids.index(layer_id)
    pad = token_map[2]
    compressed = [token_map[t] if keep else -1 for t, keep in zip(tokens, mask, strict=True)]
    rows = []
    for request, (begin, end) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
        cache = list(history[slots[request]]) + compressed[begin:end]
        for position in range(3, len(cache)):
            rolling, blocked, result = 0, False, []
            for lag in range(4):
                source = cache[position - lag]
                blocked |= source == -1
                rolling ^= (pad if blocked else source) * geometry.multipliers[layer][lag]
                if lag:
                    for head in range((lag - 1) * 8, lag * 8):
                        result.append(rolling % geometry.primes[layer][head] + geometry.offsets[layer][head])
            rows.append(result)
    return torch.tensor(rows, dtype=torch.int64).reshape(-1, 24)
