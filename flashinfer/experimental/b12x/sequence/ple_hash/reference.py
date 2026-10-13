"""Exact, intentionally unoptimized PyTorch oracles for PLE hashing."""

from __future__ import annotations

import torch

MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB


def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def is_prime_64(value: int) -> bool:
    """Deterministically test primality over the unsigned 64-bit range."""
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    odd = value - 1
    shifts = 0
    while odd % 2 == 0:
        odd //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, odd, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    current = int(start)
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    for _ in range(int(count)):
        candidate = current + 1
        if candidate <= 2:
            current = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not is_prime_64(candidate):
            candidate += 2
        current = candidate
    return current


def ple_multipliers(
    *,
    vocab_size: int,
    max_order: int,
    dense_layer_ordinal: int,
    seed: int = 1234,
) -> torch.Tensor:
    """Generate deterministic odd multipliers with safe token products."""
    vocab_size = int(vocab_size)
    max_order = int(max_order)
    dense_layer_ordinal = int(dense_layer_ordinal)
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if max_order < 2:
        raise ValueError(f"max_order must be at least 2, got {max_order}")
    if dense_layer_ordinal < 0:
        raise ValueError(
            f"dense_layer_ordinal must be nonnegative, got {dense_layer_ordinal}"
        )
    max_multiplier = ((1 << 63) - 1) // vocab_size
    half_bound = max(1, max_multiplier // 2)
    base_seed = seed + 10007 * dense_layer_ordinal
    values = []
    for index in range(max_order):
        source = base_seed + SPLITMIX_GAMMA * (index + 1)
        values.append(2 * (splitmix64(source) % half_bound) + 1)
    return torch.tensor(values, dtype=torch.int64)


def ple_table_geometry(
    *,
    base_size: int,
    dense_layer_ordinal: int,
    total_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return distinct per-head prime sizes and cumulative offsets."""
    base_size = int(base_size)
    dense_layer_ordinal = int(dense_layer_ordinal)
    total_heads = int(total_heads)
    if base_size <= 0:
        raise ValueError(f"base_size must be positive, got {base_size}")
    if dense_layer_ordinal < 0:
        raise ValueError(
            f"dense_layer_ordinal must be nonnegative, got {dense_layer_ordinal}"
        )
    if total_heads <= 0:
        raise ValueError(f"total_heads must be positive, got {total_heads}")
    sizes: list[int] = []
    offsets: list[int] = []
    offset = 0
    for local_head in range(total_heads):
        global_head = dense_layer_ordinal * total_heads + local_head
        size = nth_prime_after(base_size - 1, global_head + 1)
        sizes.append(size)
        offsets.append(offset)
        offset += size
    return (
        torch.tensor(sizes, dtype=torch.int64),
        torch.tensor(offsets, dtype=torch.int64),
    )


def eos_bounded_windows(
    request_tokens: torch.Tensor,
    *,
    eos_token_id: int,
    max_order: int,
) -> dict[int, torch.Tensor]:
    """Build oldest-to-current n-gram windows for one complete request."""
    if request_tokens.ndim != 1 or request_tokens.dtype != torch.int64:
        raise TypeError("request_tokens must be a rank-1 torch.int64 tensor")
    tokens = [int(token) for token in request_tokens.detach().cpu().tolist()]
    rows: dict[int, list[list[int]]] = {
        order: [] for order in range(2, int(max_order) + 1)
    }
    segment_start = 0
    for current, token in enumerate(tokens):
        for order in rows:
            start = current - order + 1
            window = []
            for position in range(start, current + 1):
                valid = position >= 0 and position >= segment_start
                window.append(tokens[position] if valid else int(eos_token_id))
            rows[order].append(window)
        if token == int(eos_token_id):
            segment_start = current + 1
    return {
        order: torch.tensor(values, dtype=torch.int64, device=request_tokens.device)
        for order, values in rows.items()
    }


def ple_hash_ids_reference(
    windows: dict[int, torch.Tensor],
    *,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    heads_per_order: int,
) -> torch.Tensor:
    """Hash chronological windows with multipliers indexed by token lag.

    ``multipliers[0]`` belongs to the current token, the last column in each
    window; ``multipliers[i]`` belongs to its predecessor at distance ``i``.
    """
    if not windows:
        raise ValueError("windows must contain at least one n-gram order")
    orders = sorted(windows)
    if orders != list(range(2, max(orders) + 1)):
        raise ValueError(f"windows orders must be contiguous from 2, got {orders}")
    max_order = max(orders)
    heads_per_order = int(heads_per_order)
    total_heads = (max_order - 1) * heads_per_order
    if tuple(multipliers.shape) != (max_order,):
        raise ValueError(f"multipliers must have shape {(max_order,)}")
    if tuple(prime_sizes.shape) != (total_heads,):
        raise ValueError(f"prime_sizes must have shape {(total_heads,)}")
    if tuple(table_offsets.shape) != (total_heads,):
        raise ValueError(f"table_offsets must have shape {(total_heads,)}")
    if any(
        tensor.dtype != torch.int64
        for tensor in (multipliers, prime_sizes, table_offsets)
    ):
        raise TypeError("PLE hash geometry tensors must use torch.int64")
    token_count = int(windows[orders[0]].shape[0])
    id_blocks = []
    for order in orders:
        token_window = windows[order].to(torch.int64)
        if tuple(token_window.shape) != (token_count, order):
            raise ValueError(
                f"windows[{order}] must have shape {(token_count, order)}, "
                f"got {tuple(token_window.shape)}"
            )
        mixed = token_window[:, -1] * multipliers[0]
        for index in range(1, order):
            mixed = torch.bitwise_xor(
                mixed,
                token_window[:, -1 - index] * multipliers[index],
            )
        first = (order - 2) * heads_per_order
        last = first + heads_per_order
        sizes = prime_sizes[first:last]
        offsets = table_offsets[first:last]
        ids = torch.remainder(mixed[:, None], sizes[None, :])
        id_blocks.append(ids + offsets[None, :])
    return torch.cat(id_blocks, dim=-1)


def ple_hash_packed_reference(
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    *,
    eos_token_id: int,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    heads_per_order: int,
) -> torch.Tensor:
    """Hash packed query tokens using each request's committed left context."""
    if token_ids.ndim != 1 or token_ids.dtype != torch.int64:
        raise TypeError("token_ids must be a rank-1 torch.int64 tensor")
    if query_start_loc.ndim != 1 or query_start_loc.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("query_start_loc must be a rank-1 int32/int64 tensor")
    if committed_history.ndim != 2 or committed_history.dtype != torch.int64:
        raise TypeError("committed_history must be a rank-2 torch.int64 tensor")
    max_order = int(multipliers.numel())
    num_seqs = int(query_start_loc.numel()) - 1
    if tuple(committed_history.shape) != (num_seqs, max_order - 1):
        raise ValueError(
            "committed_history must have shape "
            f"{(num_seqs, max_order - 1)}, got {tuple(committed_history.shape)}"
        )
    starts = [int(value) for value in query_start_loc.detach().cpu().tolist()]
    if starts[0] != 0 or starts[-1] != int(token_ids.numel()):
        raise ValueError("query_start_loc must span exactly all packed token_ids")
    if any(left > right for left, right in zip(starts, starts[1:], strict=False)):
        raise ValueError("query_start_loc must be nondecreasing")

    outputs = []
    history_length = max_order - 1
    for request in range(num_seqs):
        query = token_ids[starts[request] : starts[request + 1]]
        complete = torch.cat((committed_history[request], query))
        windows = eos_bounded_windows(
            complete,
            eos_token_id=eos_token_id,
            max_order=max_order,
        )
        query_windows = {
            order: values[history_length:] for order, values in windows.items()
        }
        outputs.append(
            ple_hash_ids_reference(
                query_windows,
                multipliers=multipliers,
                prime_sizes=prime_sizes,
                table_offsets=table_offsets,
                heads_per_order=heads_per_order,
            )
        )
    if outputs:
        return torch.cat(outputs, dim=0)
    return torch.empty(
        (0, (max_order - 1) * int(heads_per_order)),
        dtype=torch.int64,
        device=token_ids.device,
    )


__all__ = [
    "MASK64",
    "SPLITMIX_GAMMA",
    "SPLITMIX_M1",
    "SPLITMIX_M2",
    "splitmix64",
    "is_prime_64",
    "nth_prime_after",
    "ple_multipliers",
    "ple_table_geometry",
    "eos_bounded_windows",
    "ple_hash_ids_reference",
    "ple_hash_packed_reference",
]
