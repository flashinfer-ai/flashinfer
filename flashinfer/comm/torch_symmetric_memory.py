import functools
from typing import Any

import torch
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed.distributed_c10d as c10d

_compat_patched = False


def _patch_group_count_reset() -> None:
    """Prevent group_count from resetting to 0 on WORLD destruction (2.10 and below)."""
    global _compat_patched
    if _compat_patched:
        return
    _compat_patched = True

    import torch.distributed as dist

    _original_destroy = dist.destroy_process_group

    @functools.wraps(_original_destroy)
    def _patched_destroy(group=None):
        saved_count = c10d._world.group_count
        _original_destroy(group)
        # WORLD destruction resets group_count to 0 – restore it so the next
        # init_process_group picks a name that is fresh in the C++ map.
        if group is None:
            c10d._world.group_count = saved_count

    dist.destroy_process_group = _patched_destroy


def _enable_symm_mem_for_group(group_name: str) -> None:
    """Enable symmetric memory for a process group (PyTorch 2.11+)."""
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version >= (2, 11):
        return
    from torch.distributed._symmetric_memory import enable_symm_mem_for_group

    _patch_group_count_reset()
    enable_symm_mem_for_group(group_name)


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
    # Ensure symmetric memory is set up with the correct store before
    # rendezvous on PyTorch older than 2.11.
    _enable_symm_mem_for_group(group_name)

    elem_size = torch.empty(0, dtype=dtype).element_size()
    numel = size_bytes // elem_size
    tensor = symm_mem.empty(numel, dtype=dtype, device=device)
    handle = symm_mem.rendezvous(tensor, group=group_name)
    ptrs: list[int] = [
        handle.get_buffer(peer, (numel,), dtype, storage_offset=0).data_ptr()
        for peer in range(world_size)
    ]
    return ptrs, tensor, handle
