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
