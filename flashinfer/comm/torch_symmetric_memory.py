import torch
import torch.distributed._symmetric_memory as symm_mem


def _alloc_symm_buffer_bytes(
    size_bytes: int, tp_size: int, dtype: torch.dtype, device: torch.device, group_name: str
) -> tuple[list[int], torch.Tensor]:
    numel = size_bytes // torch.tensor([], dtype=dtype).element_size()
    tensor = symm_mem.empty(numel, dtype=dtype, device=device)
    handle = symm_mem.rendezvous(tensor, group=group_name)
    ptrs: list[int] = []
    for peer in range(tp_size):
        buf = handle.get_buffer(peer, (numel,), dtype, storage_offset=0)
        ptrs.append(buf.data_ptr())
    return ptrs, tensor, handle

