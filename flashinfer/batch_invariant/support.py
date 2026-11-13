# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The gates of the batch-invariant mode: the device check of an operator's class and the "unsupported raises" rule.
Light imports only: the operators import this module on every call with ``batch_invariant=True``."""

from __future__ import annotations

import functools
from typing import Iterable, Tuple

import torch


class BatchInvariantUnsupportedError(ValueError):
    """``batch_invariant=True`` was asked for an input the batch-invariant class does not cover.

    The mode never falls back to a non-invariant kernel: an unsupported shape, dtype, layout, device or architecture
    raises this error, whose message names the class and the limit."""


def unsupported(
    class_name: str, what: str, limit: str
) -> BatchInvariantUnsupportedError:
    return BatchInvariantUnsupportedError(
        f"batch_invariant=True: the class `{class_name}` does not support {what}; the limit is {limit}. "
        "The batch-invariant mode never falls back to another kernel; call without batch_invariant=True to use "
        "the regular backends."
    )


def compute_capability(device: torch.device) -> Tuple[int, int]:
    """The device's compute capability as FlashInfer reports it."""
    from ..utils import get_compute_capability

    major, minor = get_compute_capability(device)
    return (int(major), int(minor))


def _arch_names(archs: Iterable[Tuple[int, int]]) -> str:
    return " / ".join(f"SM{a}{b}" for a, b in archs)


def is_supported_device(device: torch.device, archs: Iterable[Tuple[int, int]]) -> bool:
    """Whether ``device`` is a CUDA device of one of the compute capabilities ``archs`` that the installed CuTe DSL can
    target. Each class states its own ``archs``."""
    if device.type != "cuda":
        return False
    from ..utils import get_compute_capability

    cc = get_compute_capability(device)
    if cc not in tuple(archs):
        return False
    from ..cute_dsl.availability import (
        is_cute_dsl_arch_supported,
        is_cute_dsl_available,
    )

    return is_cute_dsl_available() and is_cute_dsl_arch_supported(*cc)


def check_device(
    device: torch.device, class_name: str, archs: Iterable[Tuple[int, int]]
) -> Tuple[int, int]:
    """The compute capability of ``device``, or :class:`BatchInvariantUnsupportedError` naming the limit; ``archs`` are
    the compute capabilities the class ``class_name`` supports.

    Runs on every call with ``batch_invariant=True``; a successful check is cached per (device, class, archs), since
    neither the device's compute capability nor the installed CuTe DSL changes within a process (a failing check is
    not cached and raises again on the next call)."""
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _check_device_cached(device, class_name, tuple(archs))


@functools.lru_cache(maxsize=None)
def _check_device_cached(
    device: torch.device, class_name: str, archs: Tuple[Tuple[int, int], ...]
) -> Tuple[int, int]:
    if device.type != "cuda":
        raise unsupported(
            class_name,
            f"device {device}",
            f"a CUDA device of architecture {_arch_names(archs)}",
        )
    from ..utils import get_compute_capability

    cc = get_compute_capability(device)
    if cc not in archs:
        raise unsupported(
            class_name,
            f"compute capability {cc[0]}.{cc[1]}",
            f"the architectures {_arch_names(archs)}",
        )
    from ..cute_dsl.availability import (
        is_cute_dsl_arch_supported,
        is_cute_dsl_available,
    )

    if not is_cute_dsl_available():
        raise unsupported(
            class_name,
            "an environment without the CuTe DSL",
            "the nvidia-cutlass-dsl package being installed (the kernels are CuTe-DSL kernels)",
        )
    if not is_cute_dsl_arch_supported(*cc):
        raise unsupported(
            class_name,
            f"the installed CuTe DSL on compute capability {cc[0]}.{cc[1]}",
            "a CuTe DSL release that can target this device",
        )
    return cc
