"""Shared CuteDSL compile helpers for GDN decode kernels."""

from __future__ import annotations

import torch
import cutlass.cute as cute


def device_compute_capability(device: torch.device | str) -> tuple[int, int]:
    """Return ``(major, minor)`` for ``device``, resolving index-less ``cuda``."""
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return torch.cuda.get_device_capability(d)


def cute_compile_options(device: torch.device | str) -> tuple:
    """Device-matched CuteDSL ``GPUArch`` options for ``device``.

    Prefer the tensor's device over process ``current_device`` so multi-GPU
    callers compile for the GPU that owns the inputs (flashinfer#3960, #4117).
    """
    major, minor = device_compute_capability(device)
    # Match nvidia-cutlass-dsl gpu_arch_map: Hopper+ primary targets use "a".
    if major >= 9:
        return (cute.GPUArch(f"sm_{major}{minor}a"),)
    return (cute.GPUArch(f"sm_{major}{minor}"),)


def cute_compile(func, *args, device: torch.device | str, **kwargs):
    """``cute.compile`` with ``GPUArch`` taken from ``device``."""
    return cute.compile[cute_compile_options(device)](func, *args, **kwargs)
