"""Resolve kernel configs to backend instances."""

from __future__ import annotations

import warnings
from typing import Type

from .base import MegaKernelBackend, SplitKernelBackend, SplitKernelContext

_SPLIT_KERNEL_REGISTRY: dict[str, Type[SplitKernelBackend]] = {}
_MEGA_KERNEL_REGISTRY: dict[str, Type[MegaKernelBackend]] = {}
# Pre-taxonomy kernel_name spellings -> canonical name. Resolvable for
# backward compatibility, but WILL BE REMOVED in a future release; using one
# emits a DeprecationWarning.
_MEGA_KERNEL_DEPRECATED_ALIASES: dict[str, str] = {}


def register_split_kernel(name: str):
    def decorator(cls: Type[SplitKernelBackend]) -> Type[SplitKernelBackend]:
        _SPLIT_KERNEL_REGISTRY[name] = cls
        return cls

    return decorator


def register_mega_kernel(name: str, *, deprecated_aliases: tuple[str, ...] = ()):
    def decorator(cls: Type[MegaKernelBackend]) -> Type[MegaKernelBackend]:
        _MEGA_KERNEL_REGISTRY[name] = cls
        for alias in deprecated_aliases:
            _MEGA_KERNEL_REGISTRY[alias] = cls
            _MEGA_KERNEL_DEPRECATED_ALIASES[alias] = name
        return cls

    return decorator


def _warn_if_deprecated_mega_name(name: str) -> None:
    canonical = _MEGA_KERNEL_DEPRECATED_ALIASES.get(name)
    if canonical is not None:
        warnings.warn(
            f"mega kernel name {name!r} is deprecated and will be removed in "
            f"a future release; use {canonical!r}",
            DeprecationWarning,
            stacklevel=3,
        )


def _available_mega_kernels() -> list[str]:
    return sorted(
        n for n in _MEGA_KERNEL_REGISTRY if n not in _MEGA_KERNEL_DEPRECATED_ALIASES
    )


def _kernel_name(config: object) -> str:
    name = getattr(config, "kernel_name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(
            f"kernel config {type(config).__name__!r} must define a non-empty "
            "kernel_name str"
        )
    return name


def is_split_kernel_config(config: object) -> bool:
    """True when ``config`` names a registered split-path inner kernel."""
    try:
        name = _kernel_name(config)
    except TypeError:
        return False
    return name in _SPLIT_KERNEL_REGISTRY


def create_split_kernel(config: object) -> SplitKernelBackend:
    name = _kernel_name(config)
    if name not in _SPLIT_KERNEL_REGISTRY:
        available = sorted(_SPLIT_KERNEL_REGISTRY)
        raise KeyError(f"unknown split kernel {name!r}; available: {available}")
    return _SPLIT_KERNEL_REGISTRY[name](config)


def is_mega_kernel_config(config: object) -> bool:
    """True when ``config`` names a registered mega-kernel plugin."""
    try:
        name = _kernel_name(config)
    except TypeError:
        return False
    return name in _MEGA_KERNEL_REGISTRY


def create_mega_kernel(config: object) -> MegaKernelBackend:
    name = _kernel_name(config)
    if name not in _MEGA_KERNEL_REGISTRY:
        available = _available_mega_kernels()
        raise KeyError(f"unknown mega kernel {name!r}; available: {available}")
    _warn_if_deprecated_mega_name(name)
    return _MEGA_KERNEL_REGISTRY[name](config)


def kernel_requires_weights(config: object) -> bool:
    name = _kernel_name(config)
    if name not in _SPLIT_KERNEL_REGISTRY:
        available = sorted(_SPLIT_KERNEL_REGISTRY)
        raise KeyError(f"unknown split kernel {name!r}; available: {available}")
    return _SPLIT_KERNEL_REGISTRY[name].requires_weights()


def run_split_kernel(config: object, ctx: SplitKernelContext):
    return create_split_kernel(config).compute(ctx)
