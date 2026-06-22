"""Resolve kernel configs to backend instances."""

from __future__ import annotations

from typing import Type

from .base import MegaKernelBackend, SplitKernelBackend, SplitKernelContext

_SPLIT_KERNEL_REGISTRY: dict[str, Type[SplitKernelBackend]] = {}
_MEGA_KERNEL_REGISTRY: dict[str, Type[MegaKernelBackend]] = {}


def register_split_kernel(name: str):
    def decorator(cls: Type[SplitKernelBackend]) -> Type[SplitKernelBackend]:
        _SPLIT_KERNEL_REGISTRY[name] = cls
        return cls

    return decorator


def register_mega_kernel(name: str):
    def decorator(cls: Type[MegaKernelBackend]) -> Type[MegaKernelBackend]:
        _MEGA_KERNEL_REGISTRY[name] = cls
        return cls

    return decorator


def _kernel_name(config: object) -> str:
    name = getattr(config, "kernel_name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(
            f"kernel config {type(config).__name__!r} must define a non-empty "
            "kernel_name str"
        )
    return name


def create_split_kernel(config: object) -> SplitKernelBackend:
    name = _kernel_name(config)
    if name not in _SPLIT_KERNEL_REGISTRY:
        available = sorted(_SPLIT_KERNEL_REGISTRY)
        raise KeyError(f"unknown split kernel {name!r}; available: {available}")
    return _SPLIT_KERNEL_REGISTRY[name](config)


def create_mega_kernel(config: object) -> MegaKernelBackend:
    name = _kernel_name(config)
    if name not in _MEGA_KERNEL_REGISTRY:
        available = sorted(_MEGA_KERNEL_REGISTRY)
        raise KeyError(f"unknown mega kernel {name!r}; available: {available}")
    return _MEGA_KERNEL_REGISTRY[name](config)


def resolve_split_kernel(config: object) -> SplitKernelBackend:
    return create_split_kernel(config)


def resolve_mega_kernel(config: object) -> MegaKernelBackend:
    return create_mega_kernel(config)


def kernel_requires_weights(config: object) -> bool:
    return create_split_kernel(config).requires_weights()


def run_split_kernel(config: object, ctx: SplitKernelContext):
    return create_split_kernel(config).compute(ctx)
