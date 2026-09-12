# Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
All-gather matmul fused using a push-wait algorithm.

For testing, see test_all_gather_matmul.py.

Problem:
    Each rank holds a local input of shape (M, K) and a weight tensor of shape (K, N).
    The goal is to compute output of shape (M * world_size, N), where slice [i] is
    input[i] @ weight.

Algorithm (push-wait):
    1. Push: each rank writes its local input into the symmetric memory buffer of every
       peer rank, chunk by chunk, and sets a per-chunk ready signal on the destination.
    2. Wait-matmul: the matmul kernel on each rank iterates over all peers. For the local
       input chunks it starts immediately; for remote chunks it spin-waits on the
       corresponding signal before loading and computing the tile.

    A barrier kernel (one block per peer) is used to synchronize the start of the push
    phase: each block deposits a signal to its peer and waits for the peer's deposit,
    ensuring all ranks have entered the kernel before any pushing begins.

Routing:
    - SM >= 100 (Blackwell+): cuTile implementation (all_gather_matmul_cutile)
    - SM <  100             : Triton implementation  (all_gather_matmul_triton)
    - ``backend="cake"``    : exact source-built SM100/SM103 implementation

Example (run with torchrun or mp.spawn across all GPU ranks)::

    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    from flashinfer.comm import all_gather_matmul

    # --- per-rank setup ---
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    # --- inputs ---
    M, K, N = 16384, 8192, 2048
    dtype = torch.bfloat16
    inp = symm_mem.empty(M, K, device=device, dtype=dtype).normal_()
    w   = torch.randn(K, N, device=device, dtype=dtype)

    # --- fused all-gather matmul ---
    # out shape: (M * world_size, N)
    out = all_gather_matmul(inp, w, group)
"""

from typing import Callable

import torch
import torch.distributed as dist

from flashinfer.utils import register_custom_op


@register_custom_op(
    "flashinfer::all_gather_matmul",
    mutates_args=[],
)
def all_gather_matmul(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    backend: str = "auto",
    verbose: bool = False,
):
    """Run push-wait all-gather matmul.

    ``backend="auto"`` preserves the existing cuTile-on-Blackwell and
    Triton-otherwise routing. ``backend="cake"`` selects the exact
    source-built Blackwell implementation and rejects unsupported inputs.
    """
    if backend == "cake":
        from .cake_all_gather_matmul import all_gather_matmul_cake

        return all_gather_matmul_cake(inp, w, group, backend="cake", verbose=verbose)
    if backend != "auto":
        raise ValueError("backend must be exactly 'auto' or 'cake'")
    major, _ = torch.cuda.get_device_capability(inp.device)
    if major >= 10:
        from .all_gather_matmul_cutile import all_gather_matmul_cutile

        return all_gather_matmul_cutile(inp, w, group, verbose=verbose)
    from .all_gather_matmul_triton import all_gather_matmul_triton

    return all_gather_matmul_triton(inp, w, group, verbose=verbose)


def prepare_all_gather_matmul(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    backend: str = "auto",
    verbose: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Prepare the packed-QKV all-gather matmul launcher.

    The returned callable binds ``w`` and ``group`` and accepts a new input
    tensor with the same shape, dtype, and device as ``inp``. Both
    ``backend="auto"`` and ``backend="cake"`` select the source-built
    prepared SM103/BF16 launcher for the exact TP4/N=2560 or TP8/N=1280
    profile. Unsupported inputs raise during preparation instead of falling
    back to another implementation.
    """
    if backend not in {"auto", "cake"}:
        raise ValueError("backend must be exactly 'auto' or 'cake'")

    from .cake_all_gather_matmul import (
        _prepare_all_gather_matmul_cake_packed_qkv_sm103,
    )

    return _prepare_all_gather_matmul_cake_packed_qkv_sm103(
        inp, w, group, verbose=verbose
    )
