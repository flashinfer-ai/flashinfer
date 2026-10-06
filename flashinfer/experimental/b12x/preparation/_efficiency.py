"""Declaration-time controls and finite ladders for search efficiency."""
import os


def capture_exhaustive_search() -> bool:
    value = os.environ.get("B12X_AUTOTUNE_EXHAUSTIVE", "0")
    if value not in ("0", "1"):
        raise ValueError("B12X_AUTOTUNE_EXHAUSTIVE must be 0 or 1")
    return value == "1"


def powers_of_two(capacity: int) -> frozenset[int]:
    return frozenset(1 << bit for bit in range(capacity.bit_length()))
