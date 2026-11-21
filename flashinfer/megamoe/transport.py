"""NCCL-window transport of the SM90 MegaMoE kernels.

The kernels exchange tokens through one symmetric buffer per rank. Inside an
NVLink domain every rank reads and writes its peers' buffers through mapped
pointers, which torch symmetric memory provides for ranks on one node. This
transport provides the same buffers as NCCL symmetric windows instead: the
load/store pointers of the ranks on this node come from the window, and ranks
on other nodes are reached with GPU-initiated network puts on the NCCL device
communicator the transport creates.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.distributed as dist

from ..jit.megamoe import gen_megamoe_sm90_transport_module
from ._jit import _get_jit_module


class _PeerPointers:
    """Per-rank pointers of one window, in the shape the kernels read them."""

    def __init__(self, pointers: List[int]) -> None:
        """Keep ``pointers`` as ``buffer_ptrs``, the attribute the kernel calls read."""
        self.buffer_ptrs = list(pointers)


class _WindowArray:
    """``__cuda_array_interface__`` view of a window, so torch wraps it without a copy."""

    def __init__(self, pointer: int, nbytes: int) -> None:
        """Describe ``nbytes`` of device memory at ``pointer`` as a byte array."""
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|i1",
            "data": (pointer, False),
            "strides": None,
            "version": 3,
        }


class MegaMoENcclTransport:
    """Symmetric windows of one expert-parallel process group on NCCL.

    Creation, every :meth:`alloc_window` and :meth:`close` are collective over
    the group and must precede CUDA graph capture. The transport must outlive
    every tensor obtained from :meth:`window_tensor`.

    Parameters
    ----------
    group : torch.distributed.ProcessGroup
        Expert-parallel process group backed by NCCL.
    gin_contexts : int
        GIN contexts (one queue pair per peer each) of the device communicator.
        0 creates no device communicator, which is enough when every rank is
        on one node. Defaults to 0.
    gin_signals : int
        Signals of the device communicator. Defaults to 0.
    gin_barriers : int
        World barriers of the device communicator. Defaults to 0.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        gin_contexts: int = 0,
        gin_signals: int = 0,
        gin_barriers: int = 0,
    ) -> None:
        """Borrow the group's communicator and create the device communicator when GIN is requested."""
        self.group = group
        self.rank = group.rank()
        self.world_size = group.size()
        self.device = torch.cuda.current_device()
        self.module = _get_jit_module(gen_megamoe_sm90_transport_module)
        self.handle: Optional[int] = int(
            self.module.megamoe_sm90_transport_create(
                self._comm_pointer(group),
                self.rank,
                self.world_size,
                self.device,
                gin_contexts,
                gin_signals,
                gin_barriers,
            )
        )
        self._tensors: List[torch.Tensor] = []

    @staticmethod
    def _comm_pointer(group: dist.ProcessGroup) -> int:
        """``ncclComm_t`` of the group's NCCL backend, created on demand by a barrier."""
        dist.barrier(group=group)
        backend = group._get_backend(torch.device("cuda"))
        if not hasattr(backend, "_comm_ptr"):
            raise RuntimeError(
                "the expert-parallel group is not a ProcessGroupNCCL exposing its communicator"
            )
        pointer = int(backend._comm_ptr())
        if pointer <= 0:
            raise RuntimeError("ProcessGroupNCCL returned a null communicator pointer")
        return pointer

    def _live(self) -> int:
        """The session handle, or an error after :meth:`close`."""
        if self.handle is None:
            raise ValueError("the NCCL transport has been closed")
        return self.handle

    def alloc_window(self, nbytes: int) -> int:
        """Allocate and register one zeroed symmetric window of ``nbytes`` bytes (collective).

        Parameters
        ----------
        nbytes : int
            Size in bytes, identical on every rank; rounded up to the window alignment.

        Returns
        -------
        int
            Index of the window in this transport.
        """
        return int(
            self.module.megamoe_sm90_transport_alloc_window(self._live(), nbytes)
        )

    def window_tensor(
        self, index: int, dtype: torch.dtype = torch.int8
    ) -> torch.Tensor:
        """This rank's window ``index`` as a flat tensor of ``dtype`` sharing the window memory."""
        handle = self._live()
        pointer = int(self.module.megamoe_sm90_transport_window_pointer(handle, index))
        nbytes = int(self.module.megamoe_sm90_transport_window_bytes(handle, index))
        with torch.cuda.device(self.device):
            tensor = torch.as_tensor(_WindowArray(pointer, nbytes), device="cuda")
        tensor = tensor.view(dtype)
        self._tensors.append(tensor)
        return tensor

    def peer_pointers(self, index: int) -> List[int]:
        """Load/store pointers of window ``index`` on every rank; 0 for ranks on other nodes."""
        return [
            int(p)
            for p in self.module.megamoe_sm90_transport_peer_pointers(
                self._live(), index
            )
        ]

    def peer_handle(self, index: int) -> _PeerPointers:
        """The peer pointers of window ``index`` in the shape the kernel calls read."""
        return _PeerPointers(self.peer_pointers(index))

    def window_handle(self, index: int) -> int:
        """``ncclWindow_t`` of window ``index`` as an integer, for GIN puts."""
        return int(
            self.module.megamoe_sm90_transport_window_handle(self._live(), index)
        )

    @property
    def dev_comm_pointer(self) -> int:
        """Device address of the ``ncclDevComm`` the kernels read, 0 without GIN."""
        return int(self.module.megamoe_sm90_transport_dev_comm_pointer(self._live()))

    @property
    def gin_context_count(self) -> int:
        """GIN contexts of the device communicator, 0 without GIN."""
        return int(self.module.megamoe_sm90_transport_gin_context_count(self._live()))

    def close(self) -> None:
        """Deregister and free every window and destroy the device communicator (collective)."""
        if self.handle is None:
            return
        self._tensors.clear()
        self.module.megamoe_sm90_transport_close(self.handle)
        self.handle = None
