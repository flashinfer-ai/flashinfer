import functools
from typing import Any

import torch
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed.distributed_c10d as c10d

_compat_patched = False


def _patch_group_count_reset() -> None:
    """Prevent group_count from resetting to 0 on WORLD destruction (2.10 only).

    On PyTorch 2.10, ``destroy_process_group(WORLD)`` resets
    ``_world.group_count`` to 0.  When the WORLD group is later recreated via
    ``init_process_group``, it receives the same name (e.g. ``"0"``) as the
    previous incarnation.  The C++ ``group_info_map`` still holds the old
    entry whose store is now a zombie (the underlying TCPStore server was torn
    down with the old process group).  Because ``set_group_info`` enforces
    single-init per name, we cannot update the stale entry.

    By preserving the counter, each recreated WORLD group gets a unique name
    (``"0"``, ``"1"``, ``"2"``, …).  ``set_group_info`` always sees a fresh
    key, the rendezvous uses the live store, and the old zombie entries are
    simply never looked up again.
    """
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


def _enable_symm_mem_compat(group_name: str) -> None:
    """Pre-initialize symmetric memory for a process group (PyTorch 2.10 compat).

    PyTorch 2.10's ``enable_symm_mem_for_group`` wraps the group store in an
    extra ``PrefixStore("symmetric_memory-{ranks}", ...)``.  PyTorch 2.11
    removed this indirection and resolves the group store directly in C++ via
    ``group->getStore()``.  On 2.10 the extra layer can cause the store-based
    handle exchange inside ``make_peer_alloc_info`` to hang on certain
    topologies (Blackwell MNNVL fabric handles).

    This helper mimics the 2.11 behaviour: it calls ``set_group_info`` with the
    group's native store (no extra prefix) and populates the Python-side guard
    dict so that ``enable_symm_mem_for_group`` becomes a no-op for this group.

    It also patches ``destroy_process_group`` to preserve the group counter so
    that if the WORLD group is destroyed and recreated, the new group receives a
    unique name that does not collide with stale entries in the C++ map.
    """
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version >= (2, 11):
        # 2.11+ resolves the store in C++ – nothing to patch.
        return

    from torch.distributed._symmetric_memory import _group_name_to_store
    from torch._C._distributed_c10d import _SymmetricMemory

    # Ensure the group counter survives WORLD destruction so recreated groups
    # get unique names that won't collide in the C++ group_info_map.
    _patch_group_count_reset()

    if group_name in _group_name_to_store:
        return  # Already initialised for this group name.

    group = c10d._resolve_process_group(group_name)
    # Use the store that the C++ ProcessGroup owns – identical to what 2.11's
    # resolve_process_group(group_name)->getStore() returns.
    current_store = c10d._get_process_group_store(group)

    _group_name_to_store[group_name] = current_store
    _SymmetricMemory.set_group_info(
        group_name,
        group.rank(),
        group.size(),
        current_store,
    )


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
    # rendezvous.  On PyTorch 2.10 this avoids an extra PrefixStore layer
    # that can cause hangs on Blackwell/MNNVL topologies.
    _enable_symm_mem_compat(group_name)

    elem_size = torch.empty(0, dtype=dtype).element_size()
    numel = size_bytes // elem_size
    tensor = symm_mem.empty(numel, dtype=dtype, device=device)
    handle = symm_mem.rendezvous(tensor, group=group_name)
    ptrs: list[int] = [
        handle.get_buffer(peer, (numel,), dtype, storage_offset=0).data_ptr()
        for peer in range(world_size)
    ]
    return ptrs, tensor, handle
