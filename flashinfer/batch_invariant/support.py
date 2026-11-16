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


def out_buffer(
    t,
    name: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    like: torch.Tensor,
    align: int = 16,
):
    """A caller-owned output buffer (an ``out=`` argument): ``None`` passes through; otherwise the tensor is checked and
    returned as given -- on ``like``'s device, of ``dtype`` and exactly ``shape``, contiguous and ``align``-byte aligned
    (the kernels store through vector copies). A buffer that fails is refused with ``ValueError``, never copied: an
    ``out=`` is written in place or not at all (CUDA graphs and framework-owned structures rely on the address)."""
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"{name} must be a tensor, got {type(t).__name__}")
    if t.device != like.device:
        raise ValueError(f"{name} must be on {like.device}, got {t.device}")
    if t.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {t.dtype}")
    if tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(t.shape)}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if t.data_ptr() % align:
        raise ValueError(
            f"{name} must be {align}-byte aligned (its storage offset is not)"
        )
    return t


def byte_range(t: torch.Tensor) -> Tuple[int, int]:
    """The half-open byte range ``[start, end)`` a tensor's elements occupy, for any strides (the extent is the largest
    element offset plus one, times the element size); an empty tensor spans nothing."""
    start = t.data_ptr()
    if t.numel() == 0:
        return (start, start)
    extent = 1 + sum(
        (n - 1) * s for n, s in zip(t.shape, t.stride(), strict=True) if n > 1
    )
    return (start, start + extent * t.element_size())


def check_no_overlap(outputs, inputs, in_place=()):
    """An output buffer may overlap neither another output nor any input -- a partial overlap is a corruption (a row's
    values read after another row's store) -- except an output that IS the input it is allowed to replace in place:
    ``in_place`` lists the ``(output name, input name)`` pairs whose byte ranges must then coincide exactly (the kernel
    writes every element after the thread that owns it read it). ``outputs`` / ``inputs`` are ``(name, tensor | None)``
    pairs; a violation raises ``ValueError``."""
    outs = [(n, byte_range(t)) for n, t in outputs if t is not None]
    ins = [(n, byte_range(t)) for n, t in inputs if t is not None]

    def overlap(a, b):
        return a[0] < b[1] and b[0] < a[1]

    for i, (n, r) in enumerate(outs):
        for m, q in outs[i + 1 :]:
            if overlap(r, q):
                raise ValueError(
                    f"{n} and {m} overlap in memory: the outputs need distinct buffers"
                )
        for m, q in ins:
            if overlap(r, q):
                if (n, m) in in_place and r == q:
                    continue
                how = " without aliasing it exactly" if (n, m) in in_place else ""
                raise ValueError(
                    f"{n} overlaps {m} in memory{how}: an output buffer may not overlap an input"
                )


_scratch: dict = {}


def scratch(device: torch.device, dtype: torch.dtype, n: int) -> torch.Tensor:
    """A per-device scratch of at least ``n`` elements for a kernel output nobody reads (the draws' validity word, the
    truncated draw's kept mass when not requested): grown to the largest size seen and kept, so a call allocates nothing
    in steady state. Its contents are dead writes, so calls on different streams may share it. Inside a CUDA-graph
    capture a fresh buffer is used and not cached (a capture-time allocation belongs to the graph's pool and would
    dangle after the graph is freed)."""
    key = (device, dtype)
    t = _scratch.get(key)
    if t is not None and t.numel() >= n:
        return t
    size = max(n, 2 * (t.numel() if t is not None else 0), 1024)
    fresh = torch.empty(size, dtype=dtype, device=device)
    if not torch.cuda.is_current_stream_capturing():
        _scratch[key] = fresh
    return fresh
