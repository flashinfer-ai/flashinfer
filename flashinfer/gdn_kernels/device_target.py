"""Per-device compile target and launch policy for the GDN CuTe-DSL kernels.

The DSL resolves its default ``GPUArch`` from ``CUTE_DSL_ARCH`` or CUDA device 0,
so kernels compiled for operands on another device could be built and tuned for
the wrong GPU. Resolve one target from the operand device and use it for both the
``cute.compile`` options and the compile-cache key so the two always agree.
"""

import functools
import os
from typing import NamedTuple, Union

import cutlass.cute as cute
import torch


class GdnDeviceTarget(NamedTuple):
    """Architecture and launch policy of a single CUDA device."""

    arch: str
    major: int
    minor: int
    num_sms: int
    use_packed_fma: bool


@functools.lru_cache(maxsize=None)
def _resolve(device_index: int) -> GdnDeviceTarget:
    major, minor = torch.cuda.get_device_capability(device_index)
    # CUTE_DSL_ARCH is the DSL's own cross-compile override; keep it authoritative.
    env_arch = os.environ.get("CUTE_DSL_ARCH")
    suffix = "a" if major >= 9 else ""
    return GdnDeviceTarget(
        arch=env_arch or f"sm_{major}{minor}{suffix}",
        major=major,
        minor=minor,
        num_sms=torch.cuda.get_device_properties(device_index).multi_processor_count,
        use_packed_fma=major >= 10,
    )


def gdn_device_target(device: Union[str, torch.device]) -> GdnDeviceTarget:
    """Resolve the compile target from an operand's device."""
    d = torch.device(device)
    if d.type != "cuda":
        raise ValueError(f"GDN CuTe-DSL kernels require CUDA tensors, got {d}")
    return _resolve(torch.cuda.current_device() if d.index is None else d.index)


@functools.lru_cache(maxsize=None)
def _arch_options(arch: str) -> tuple:
    return (cute.GPUArch(arch),)


def gdn_compile_options(device: Union[str, torch.device], *extra) -> tuple:
    """``cute.compile`` options pinned to ``device``'s architecture.

    Apply via ``cute.compile[options](...)``: a string ``options=`` kwarg replaces
    these wholesale rather than merging, which would silently drop the arch.
    """
    return _arch_options(gdn_device_target(device).arch) + tuple(extra)


__all__ = ["GdnDeviceTarget", "gdn_compile_options", "gdn_device_target"]
