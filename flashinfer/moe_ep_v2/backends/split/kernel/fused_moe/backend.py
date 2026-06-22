"""Fused MoE split kernel placeholder."""

from __future__ import annotations

from .....core.kernel.base import SplitKernelBackend, SplitKernelContext
from .....core.kernel.registry import register_split_kernel


@register_split_kernel("fused_moe")
class FusedMoeSplitKernelBackend(SplitKernelBackend):
    @classmethod
    def kernel_name(cls) -> str:
        return "fused_moe"

    def compute(self, ctx: SplitKernelContext):
        raise NotImplementedError(
            "FusedMoeKernelConfig is not wired yet; use IdentityConfig for now"
        )
