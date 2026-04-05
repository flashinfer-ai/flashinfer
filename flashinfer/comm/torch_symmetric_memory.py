from typing import Any

import torch
import torch.distributed._symmetric_memory as symm_mem


def _alloc_symm_buffer_bytes(
    size_bytes: int,
    world_size: int,
    dtype: torch.dtype,
    device: torch.device,
    group_name: str,
) -> tuple[list[int], torch.Tensor, Any]:
    """Allocate a symmetric memory buffer and return per-peer pointers.

    Args:
        size_bytes: Total buffer size in bytes.
        world_size: Number of peers in the communication group.
        dtype: Element type used to interpret the buffer.
        device: CUDA device for the allocation.
        group_name: Process group name for the rendezvous.

    Returns:
        Tuple of (per-peer data pointers, local tensor, symmetric memory handle).
    """
    elem_size = torch.empty(0, dtype=dtype).element_size()
    numel = size_bytes // elem_size
    tensor = symm_mem.empty(numel, dtype=dtype, device=device)
    handle = symm_mem.rendezvous(tensor, group=group_name)
    ptrs: list[int] = [
        handle.get_buffer(peer, (numel,), dtype, storage_offset=0).data_ptr()
        for peer in range(world_size)
    ]
    return ptrs, tensor, handle
