"""Handle — per-iteration EP dispatch/combine state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import (
        CombineInputParams,
        CombineOutput,
        DispatchInputParams,
        DispatchOutput,
        HandleParams,
    )


class Handle(ABC):
    @abstractmethod
    def dispatch(self, params: "DispatchInputParams") -> "DispatchOutput":
        """Scatter token tensors to their expert-home ranks."""

    @abstractmethod
    def combine(self, params: "CombineInputParams") -> "CombineOutput":
        """Gather expert outputs back to the originating ranks."""

    @abstractmethod
    def complete(self) -> None:
        """Wait on a staged operation. No-op when ``kSplitOperation`` was unset."""

    def destroy(self) -> None:  # noqa: B027 - intentional no-op default
        """Release per-iteration native resources. Idempotent."""

    def update(self, params: "HandleParams") -> None:
        """Rebind this handle to a new step's routing, reusing its buffers.

        Optional capability. A Handle is normally created per forward, but a
        CUDA graph records the device pointers it sees at capture time, so a
        handle that is destroyed at the end of the captured forward leaves the
        replayed graph pointing at freed memory. Backends that implement
        ``update`` let one long-lived handle serve many forwards: create it
        once *outside* the capture, then call ``update`` per step so the
        routing metadata is recomputed by a kernel recorded *inside* it.

        This mirrors NCCL-EP's own graph recipe, where ``ncclEpInitHandle``
        stays outside the capture and ``ncclEpUpdateHandle`` goes in
        (``contrib/nccl_ep/ep_test.cu``, ``--use_cuda_graph``).

        Buffers are NOT reallocated, so the routing shape must stay within
        what the handle was created for; in particular ``top_k`` is fixed at
        creation and cannot change here.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement update")

    def dispatch_send_only(self, params: "DispatchInputParams") -> "DispatchOutput":
        """Optional send-only dispatch for kSplitOperation; default raises."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement dispatch_send_only"
        )

    def dispatch_recv_only(self) -> "DispatchOutput":
        """Optional recv-only dispatch for kSplitOperation; default raises."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement dispatch_recv_only"
        )
