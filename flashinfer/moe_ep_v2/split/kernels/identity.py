"""Identity inner kernel — dispatch/combine roundtrip."""

from __future__ import annotations

from .base import SplitKernelContext


class IdentitySplitKernel:
    def __call__(self, ctx: SplitKernelContext):
        return ctx.expert_tensors
