"""Packed Step 5c bucket metadata for the direct-L1 fan-out reservation.

One Int64 per ``(local expert, source-node slot)`` on the expert owner:
``count << 44 | sf_base << 22 | data_base``. The relay peer reads it with a
single system-acquire load instead of three dependent acquire loads of
``internode_recv_count_per_source``, ``relay_pool_base`` and
``relay_sf_pool_base``. Step 5c traps if any field does not fit.
"""

from __future__ import annotations


DIRECT_META_FIELD_BITS = 22
DIRECT_META_FIELD_MASK = (1 << DIRECT_META_FIELD_BITS) - 1
DIRECT_META_SF_SHIFT = DIRECT_META_FIELD_BITS
DIRECT_META_COUNT_SHIFT = 2 * DIRECT_META_FIELD_BITS
DIRECT_META_COUNT_BITS = 64 - DIRECT_META_COUNT_SHIFT
DIRECT_META_COUNT_LIMIT = 1 << DIRECT_META_COUNT_BITS


def pack_direct_bucket_meta(data_base: int, sf_base: int, count: int) -> int:
    """Host replica of the Step 5c packing; raises on the same ranges the kernel traps."""
    if not (0 <= data_base <= DIRECT_META_FIELD_MASK):
        raise ValueError(f"data_base {data_base} does not fit {DIRECT_META_FIELD_BITS} bits")
    if not (0 <= sf_base <= DIRECT_META_FIELD_MASK):
        raise ValueError(f"sf_base {sf_base} does not fit {DIRECT_META_FIELD_BITS} bits")
    if not (0 <= count < DIRECT_META_COUNT_LIMIT):
        raise ValueError(f"count {count} does not fit {DIRECT_META_COUNT_BITS} bits")
    return data_base | (sf_base << DIRECT_META_SF_SHIFT) | (count << DIRECT_META_COUNT_SHIFT)


def unpack_direct_bucket_meta(packed: int) -> tuple[int, int, int]:
    """Return ``(data_base, sf_base, count)`` exactly as the device unpacks them."""
    return (
        packed & DIRECT_META_FIELD_MASK,
        (packed >> DIRECT_META_SF_SHIFT) & DIRECT_META_FIELD_MASK,
        packed >> DIRECT_META_COUNT_SHIFT,
    )
