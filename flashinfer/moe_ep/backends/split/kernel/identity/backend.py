"""Identity split kernel — passes expert tensors through unchanged."""

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

    def requires_weights(self) -> bool:
        cfg = self._config
        if isinstance(cfg, IdentityConfig):
            return cfg.require_weights
        return bool(getattr(cfg, "require_weights", False))

    def preprocess_weights(self, weights, fleet_params):
        return super().preprocess_weights(weights, fleet_params)

    def compute(self, ctx: SplitKernelContext):
        return ctx.expert_tensors
