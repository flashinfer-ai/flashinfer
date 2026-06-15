"""Resolve ``SplitConfig.kernel`` objects to runnable inner kernels."""

from __future__ import annotations

from ..config import FusedMoeKernelConfig, IdentityConfig
from .base import SplitKernel, SplitKernelContext
from .identity import IdentitySplitKernel

_IDENTITY = IdentitySplitKernel()


def kernel_requires_weights(kernel_config: object) -> bool:
    if isinstance(kernel_config, IdentityConfig):
        return False
    if getattr(kernel_config, "kernel_name", None) == "identity":
        return False
    return True


def resolve_split_kernel(kernel_config: object) -> SplitKernel:
    if isinstance(kernel_config, IdentityConfig):
        return _IDENTITY
    if getattr(kernel_config, "kernel_name", None) == "identity":
        return _IDENTITY
    if isinstance(kernel_config, FusedMoeKernelConfig):
        raise NotImplementedError(
            "FusedMoeKernelConfig is not wired yet; use IdentityConfig for now"
        )
    raise NotImplementedError(
        f"split inner kernel {type(kernel_config).__name__!r} is not implemented"
    )


def run_split_kernel(kernel_config: object, ctx: SplitKernelContext):
    return resolve_split_kernel(kernel_config)(ctx)
