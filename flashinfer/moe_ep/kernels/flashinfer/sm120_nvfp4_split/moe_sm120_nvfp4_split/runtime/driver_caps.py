"""Driver capabilities that select between transports (no torch, no CUDA context).

The GIN-ring hybrid (``gin_hybrid``) runs K1, COMM-TX, COMM-RX and K2 in four
disjoint Green Contexts created by one heterogeneous SM split
(``NativeGreenContextResources``). That split is the structured
``cuDevSmResourceSplit`` API of CUDA 13.1; a driver without it (e.g. R580 /
CUDA 13.0 without the forward-compatibility libcuda) cannot run the hybrid.
"""

from __future__ import annotations

import os
from functools import lru_cache

# "0" / "1" force the answer (tests, or a host whose driver must not be probed).
HYBRID_SM_SPLIT_ENV = "MEGAMOE_HYBRID_SM_SPLIT"
STRUCTURED_SM_SPLIT_CUDA_VERSION = 13010  # cuDevSmResourceSplit: CUDA 13.1
FLASHINFER_UNIT_SPLIT_CUDA_VERSION = 13000  # FlashInfer unit-split fallback


@lru_cache(maxsize=1)
def hybrid_sm_split_available() -> bool:
    """Whether the driver provides the structured SM split the hybrid needs.

    Queries the driver version and the ``cuDevSmResourceSplit`` entry point
    (``cuGetProcAddress`` works without a device or context). Any failure,
    including missing CUDA bindings, reads as "not available".
    """
    forced = os.environ.get(HYBRID_SM_SPLIT_ENV, "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    try:
        from cuda.bindings import driver as cuda

        cuda.cuInit(0)  # may report no device; the queries below still work
        err, version = cuda.cuDriverGetVersion()
        if int(err) != 0:
            return False
        # FlashInfer: NativeGreenContextResources also builds the four disjoint
        # partitions on CUDA 13.0 (equal SM units from cuDevSmResourceSplitByCount
        # merged per partition), so the hybrid runs on any CUDA 13 driver.
        if int(version) >= FLASHINFER_UNIT_SPLIT_CUDA_VERSION:
            return True
        if int(version) < STRUCTURED_SM_SPLIT_CUDA_VERSION:
            return False
        err, pfn, status = cuda.cuGetProcAddress(
            b"cuDevSmResourceSplit", STRUCTURED_SM_SPLIT_CUDA_VERSION, 0)
        return int(err) == 0 and int(pfn) != 0 and int(status) == 0
    except Exception:  # no bindings / no driver
        return False


def driver_cuda_version() -> int:
    """Driver CUDA version (1000 * major + 10 * minor), 0 if unknown."""
    try:
        from cuda.bindings import driver as cuda

        err, version = cuda.cuDriverGetVersion()
        return int(version) if int(err) == 0 else 0
    except Exception:
        return 0


__all__ = ["HYBRID_SM_SPLIT_ENV", "driver_cuda_version", "hybrid_sm_split_available"]
