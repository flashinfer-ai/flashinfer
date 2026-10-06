"""Keep benchmark allocations off small-page device memory on integrated GPUs.

On GB10 the driver builds device allocations from system RAM and serves the
first requests from the smallest free buddy blocks (64 KB-1 MB fragments left
by earlier activity). The GPU cannot map those with 2 MB pages, so a buffer
backed by them streams about 2-4% slower than one on 2 MB+ blocks. A benchmark
usually allocates its weights first, so its result depends on how fragmented
the node happened to be. Claiming the fragment inventory before anything else
makes later allocations land on large blocks and the measurement repeatable.
"""

from __future__ import annotations

import os
import sys

_SPONGE: list[object] = []
_KEEP_FREE_BYTES = 16 << 30


def _small_fragment_bytes() -> int:
    """Free bytes held in buddy blocks smaller than 2 MB (orders 0-8)."""
    total = 0
    try:
        with open("/proc/buddyinfo") as f:
            for line in f:
                if " Normal " not in line + " ":
                    continue
                counts = [int(v) for v in line.split()[4:]]
                total += sum(n * (4096 << order) for order, n in enumerate(counts[:9]))
    except OSError:
        return 0
    return total


def absorb_small_page_fragments(device: int = 0) -> int:
    """Hold driver memory for the process lifetime; return the bytes held.

    No-op on discrete GPUs. ``B12X_BENCH_SPONGE=0`` disables it and a number
    sets a fixed size in GB; otherwise the size follows the current fragment
    inventory, always leaving at least 16 GB free.
    """
    request = os.environ.get("B12X_BENCH_SPONGE", "auto").strip().lower()
    if request in ("0", "off", "false", "no") or _SPONGE:
        return 0
    import torch

    if not torch.cuda.is_available():
        return 0
    if not getattr(torch.cuda.get_device_properties(device), "is_integrated", False):
        return 0
    from cuda.bindings import runtime as cudart

    wanted = int(float(request) * (1 << 30)) if request != "auto" else _small_fragment_bytes()
    torch.cuda.set_device(device)
    free_bytes = torch.cuda.mem_get_info(device)[0]
    size = min(wanted, max(0, free_bytes - _KEEP_FREE_BYTES)) & ~((2 << 20) - 1)
    if size <= 0:
        return 0
    err, ptr = cudart.cudaMalloc(size)
    if err != cudart.cudaError_t.cudaSuccess:
        return 0
    # Touch every page so the driver commits the fragments now.
    cudart.cudaMemset(ptr, 0, size)
    cudart.cudaDeviceSynchronize()
    _SPONGE.append(ptr)
    print(
        f"[b12x] holding {size / (1 << 30):.1f} GB of small-page device memory "
        f"(B12X_BENCH_SPONGE={request}) so benchmark buffers land on 2 MB pages",
        file=sys.stderr,
    )
    return size


__all__ = ["absorb_small_page_fragments"]
