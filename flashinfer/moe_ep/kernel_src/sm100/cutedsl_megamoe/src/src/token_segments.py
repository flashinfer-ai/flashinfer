"""CuTe helpers implementing the adaptive expert token-segment policy."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32


@cute.jit
def merged_token_segment(intra, relay, tile: cutlass.Constexpr):
    # Counts may be Int64 in Step 5c's overflow guard; do not narrow them.
    return (intra + relay + tile - 1) // tile < (
        (intra + tile - 1) // tile + (relay + tile - 1) // tile
    )


@cute.jit
def scheduled_token_segments(intra, relay, tile: cutlass.Constexpr):
    """Merged work occupies the relay scheduling pass, with full-expert rows."""
    first = intra
    second = relay
    if merged_token_segment(intra, relay, tile):
        first = Int32(0)
        second = intra + relay
    return first, second


@cute.jit
def expert_token_tiles(intra, relay, tile: cutlass.Constexpr):
    first, second = scheduled_token_segments(intra, relay, tile)
    return (first + tile - 1) // tile + (second + tile - 1) // tile
