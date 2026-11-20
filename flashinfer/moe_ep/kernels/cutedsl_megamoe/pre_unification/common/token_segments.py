"""Host policy for packing intranode and relay rows of one expert."""


def merge_token_segments(intra: int, relay: int, tile: int) -> bool:
    if intra < 0 or relay < 0 or tile <= 0:
        raise ValueError("counts must be non-negative and tile must be positive")
    return (intra + relay + tile - 1) // tile < (
        (intra + tile - 1) // tile + (relay + tile - 1) // tile
    )
