"""Identity split kernel — returns a copy of the expert tensors."""

from __future__ import annotations

from .....core.kernel.base import SplitKernelBackend, SplitKernelContext
from .....core.kernel.registry import register_split_kernel
from .config import IdentityConfig


@register_split_kernel("identity")
class IdentitySplitKernelBackend(SplitKernelBackend):
    def __init__(self, config: IdentityConfig) -> None:
        super().__init__(config)

    @classmethod
    def kernel_name(cls) -> str:
        return "identity"

    @classmethod
    def requires_weights(cls) -> bool:
        return False

    def compute(self, ctx: SplitKernelContext):
        # A copy, not the dispatched tensors themselves: a communication may
        # reuse its receive rows before its combine reads the expert output.
        tokens = ctx.expert_tensors
        out = ctx.output
        if (
            out is not None
            and out.dtype == tokens.dtype
            and out.numel() == tokens.numel()
            and out.is_contiguous()
        ):
            return out.view(tokens.shape).copy_(tokens)
        return tokens.clone()
