"""Shared-memory helpers for the CUTLASS DSL 4.6 contract."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute


def make_smem_memrange_alias(dtype, num_elems: int, ptr):
    """Return a typed, non-owning view over an existing shared-memory range.

    CUTLASS DSL 4.6 does not expose a public constructor for this alias form.
    Keep the one required private API touchpoint here so payload carving can be
    audited and updated without duplicating version-sensitive construction in
    individual kernels. Allocation and byte offsets remain owned by the caller.
    """
    return cute.struct._MemRangeData(dtype, num_elems, ptr)


def make_tma_aligned_payload_storage(*, payload_bytes: int, num_stages: int):
    """Build the canonical two-barrier-array TMA shared-memory struct.

    Returning the real CUTLASS type keeps its alignment and tail-padding rules
    authoritative for both launch accounting and the kernel allocation.
    """
    if payload_bytes <= 0:
        raise ValueError(f"payload_bytes must be positive, got {payload_bytes}")
    if num_stages <= 0:
        raise ValueError(f"num_stages must be positive, got {num_stages}")

    class SharedStorage:
        pass

    mbar_struct = cute.struct.MemRange[cutlass.Int64, 2 * int(num_stages)]
    SharedStorage.__annotations__ = {
        "mbar_ptr_K": mbar_struct,
        "mbar_ptr_V": mbar_struct,
        "payload": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, int(payload_bytes)],
            1024,
        ],
    }
    return cute.struct(SharedStorage)


__all__ = ["make_smem_memrange_alias", "make_tma_aligned_payload_storage"]
