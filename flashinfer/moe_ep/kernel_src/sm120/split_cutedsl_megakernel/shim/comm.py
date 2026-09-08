"""NVSHMEM allocation and peer-pointer helpers for the W4A8 split kernel."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch


def _no_dist() -> bool:
    return bool(int(os.environ.get("MEGA_NO_DIST", "0")))


def ensure_not_capturing(operation: str) -> None:
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{operation} cannot run during CUDA Graph capture; call warmup() "
            "collectively on every EP rank before capture."
        )


def sym_zeros(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if _no_dist():
        tensor = torch.zeros(shape, dtype=dtype, device="cuda")
        tensor._mega_plain_alloc = True
        return tensor
    import nvshmem.core

    tensor = nvshmem.core.tensor(shape, dtype=dtype)
    tensor.zero_()
    return tensor


def sym_byte_view(
    shape: Tuple[int, ...], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    nbytes = dtype.itemsize
    for extent in shape:
        nbytes *= extent
    root = sym_zeros((nbytes,), torch.uint8)
    return root.view(dtype).reshape(shape), root


def free_sym_tensor(tensor: Optional[torch.Tensor]) -> None:
    if tensor is None or getattr(tensor, "_mega_plain_alloc", False) or _no_dist():
        return
    import nvshmem.core

    try:
        nvshmem.core.free_tensor(tensor)
    except (RuntimeError, TypeError, ValueError) as exc:
        if any(word in str(exc).lower() for word in ("already", "freed", "invalid")):
            return
        raise


def compute_peer_offsets(
    symmetric_tensor: torch.Tensor, world_size: int
) -> tuple[int, tuple[int, ...]]:
    local_base = int(symmetric_tensor.data_ptr())
    if _no_dist():
        return local_base, tuple(0 for _ in range(world_size))

    import nvshmem.core
    from nvshmem.core.interop.torch import tensor_get_buffer

    my_pe = int(nvshmem.core.my_pe())
    buffer, _size, _dtype = tensor_get_buffer(symmetric_tensor)

    def peer_base(peer: int) -> int:
        if peer == my_pe:
            return local_base
        peer_buffer = nvshmem.core.get_peer_buffer(buffer, peer)
        return int(torch.utils.dlpack.from_dlpack(peer_buffer).data_ptr())

    return local_base, tuple(
        peer_base(peer) - local_base for peer in range(world_size)
    )


__all__ = [
    "compute_peer_offsets",
    "ensure_not_capturing",
    "free_sym_tensor",
    "sym_byte_view",
    "sym_zeros",
]
