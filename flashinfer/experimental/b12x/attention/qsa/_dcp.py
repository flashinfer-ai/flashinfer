"""Static context-parallel geometry for QSA's compressed groups."""

from __future__ import annotations


def validate_geometry(
    *,
    size: int,
    rank: int,
    token_interleave: int,
    compress_ratio: int,
) -> None:
    """Validate that DCP ownership preserves whole compressed groups."""
    if size <= 0:
        raise ValueError("dcp_size must be positive")
    if not 0 <= rank < size:
        raise ValueError("dcp_rank must be in [0, dcp_size)")
    if token_interleave <= 0:
        raise ValueError("cp_kv_cache_interleave_size must be positive")
    if compress_ratio <= 0:
        raise ValueError("compress_ratio must be positive")
    if size > 1 and token_interleave % compress_ratio:
        raise ValueError(
            "QSA DCP requires cp_kv_cache_interleave_size divisible by "
            "compress_ratio"
        )


def local_length(
    global_length: int,
    *,
    size: int,
    rank: int,
    interleave: int,
) -> int:
    """Return the number of striped positions owned by one DCP rank."""
    if global_length < 0:
        raise ValueError("global_length must be nonnegative")
    if size <= 0 or not 0 <= rank < size or interleave <= 0:
        raise ValueError("invalid DCP geometry")
    round_width = size * interleave
    full_rounds, remainder = divmod(global_length, round_width)
    return full_rounds * interleave + min(
        max(remainder - rank * interleave, 0), interleave
    )


def global_to_local(
    position: int,
    *,
    size: int,
    interleave: int,
) -> tuple[int, int]:
    """Map a global striped position to ``(owner_rank, local_position)``."""
    if position < 0:
        raise ValueError("position must be nonnegative")
    if size <= 0 or interleave <= 0:
        raise ValueError("invalid DCP geometry")
    stripe, stripe_offset = divmod(position, interleave)
    round_index, owner = divmod(stripe, size)
    return owner, round_index * interleave + stripe_offset


def local_to_global(
    position: int,
    *,
    size: int,
    rank: int,
    interleave: int,
) -> int:
    """Map one rank-local striped position back to global position space."""
    if position < 0:
        raise ValueError("position must be nonnegative")
    if size <= 0 or not 0 <= rank < size or interleave <= 0:
        raise ValueError("invalid DCP geometry")
    round_index, stripe_offset = divmod(position, interleave)
    return (round_index * size + rank) * interleave + stripe_offset


__all__ = [
    "global_to_local",
    "local_length",
    "local_to_global",
    "validate_geometry",
]
