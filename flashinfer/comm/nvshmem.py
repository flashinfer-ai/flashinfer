from typing import Sequence

import torch

import nvshmem.core
from cuda.core import Device


def get_unique_id() -> "nvshmem.bindings.nvshmem.uniqueid":
    """Get a new NVSHMEM unique ID for initialization.

    Returns a nvshmem4py uniqueid object. To broadcast across ranks,
    serialize uid._data via numpy/torch and reconstruct with
    UniqueID.from_data().
    """
    return nvshmem.core.get_unique_id(empty=False)


def alloc_empty_unique_id() -> "nvshmem.bindings.nvshmem.uniqueid":
    """Allocate an empty unique ID (for non-root ranks before broadcast)."""
    return nvshmem.core.get_unique_id(empty=True)


def init(uid: "nvshmem.bindings.nvshmem.uniqueid", rank: int, world_size: int) -> None:
    device = Device(torch.cuda.current_device())
    nvshmem.core.init(
        device=device,
        uid=uid,
        rank=rank,
        nranks=world_size,
        initializer_method="uid",
    )
    torch.cuda.synchronize()


def alltoall(dest: torch.Tensor, source: torch.Tensor) -> None:
    stream = torch.cuda.current_stream()
    nvshmem.core.alltoall(nvshmem.core.Teams.TEAM_WORLD, dest, source, stream=stream)


def finalize() -> None:
    torch.cuda.synchronize()
    nvshmem.core.finalize()


def my_pe() -> int:
    return nvshmem.core.my_pe()


def n_pes() -> int:
    return nvshmem.core.n_pes()


def malloc(
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Allocates memory using NVSHMEM collective malloc operation.

    This is a collective operation that requires participation by all PEs (Processing Elements).
    All participants must call this function with the same parameters.

    Note: Use free_tensor(tensor) to free the returned tensor
    rather than relying on garbage collection.

    Args:
        shape: The shape of the tensor to allocate.
        dtype: The data type of the tensor.
        device: The device to allocate the tensor on.

    Returns:
        A tensor allocated using NVSHMEM collective malloc.
    """
    if isinstance(device, torch.device):
        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
    else:
        device_index = torch.cuda.current_device()
    if device_index != torch.cuda.current_device():
        raise ValueError(
            f"NVSHMEM malloc requested on device {device_index}, "
            f"but current CUDA device is {torch.cuda.current_device()}. "
            "NVSHMEM allocates on the current device."
        )
    return nvshmem.core.tensor(tuple(shape), dtype=dtype)


def free_tensor(tensor: torch.Tensor) -> None:
    """Free a tensor allocated by malloc()."""
    nvshmem.core.free_tensor(tensor)


def barrier_all() -> None:
    stream = torch.cuda.current_stream()
    nvshmem.core.barrier_all(stream=stream)
    stream.synchronize()


def barrier_all_on_current_stream() -> None:
    stream = torch.cuda.current_stream()
    nvshmem.core.barrier_all(stream=stream)
