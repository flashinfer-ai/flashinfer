"""Caller-supplied host copy of ``cu_seqlens`` for ``recurrent_kda`` (avoids a per-call device-to-host read)."""

import contextlib
import contextvars
import functools
import weakref
from typing import Optional, Sequence, Union

import torch

HostHint = Union[torch.Tensor, Sequence[int]]

_HOST_CU_SEQLENS: contextvars.ContextVar[Optional[HostHint]] = contextvars.ContextVar(
    "flashinfer_kda_host_cu_seqlens", default=None
)
_NORMALIZED: contextvars.ContextVar[Optional[tuple[int, ...]]] = contextvars.ContextVar(
    "flashinfer_kda_host_cu_seqlens_normalized", default=None
)


def normalize_host_cu_seqlens(host: HostHint) -> tuple[int, ...]:
    if isinstance(host, torch.Tensor):
        if host.device.type != "cpu" or host.ndim != 1:
            raise ValueError("cu_seqlens_host must be a 1-D CPU tensor or a sequence")
        host = host.tolist()
    return tuple(int(value) for value in host)


def host_cu_seqlens_for(cu_seqlens: torch.Tensor) -> Optional[tuple[int, ...]]:
    """Return the caller's host offsets for ``cu_seqlens`` when a hint is active."""

    host = _HOST_CU_SEQLENS.get()
    if host is None:
        return None
    normalized = _NORMALIZED.get()
    if normalized is None:
        normalized = normalize_host_cu_seqlens(host)
        _NORMALIZED.set(normalized)
    if len(normalized) != cu_seqlens.numel():
        raise ValueError(
            f"cu_seqlens_host has {len(normalized)} entries but cu_seqlens has "
            f"{cu_seqlens.numel()}"
        )
    return normalized


_CAST_SOURCES: dict = {}


def register_cu_seqlens_cast(converted: torch.Tensor, source: torch.Tensor) -> None:
    """Record that ``converted`` is a fresh cast of ``source`` so its contents can be read from ``source``."""

    _CAST_SOURCES[converted.data_ptr()] = (
        weakref.ref(converted),
        weakref.ref(source),
    )


def cast_source_of(converted: torch.Tensor) -> Optional[torch.Tensor]:
    entry = _CAST_SOURCES.get(converted.data_ptr())
    if entry is None or entry[0]() is not converted:
        return None
    return entry[1]()


def cu_seqlens_to_list(cu_seqlens: torch.Tensor) -> list:
    """``cu_seqlens.tolist()`` unless the caller supplied host offsets."""

    host = host_cu_seqlens_for(cu_seqlens)
    return cu_seqlens.tolist() if host is None else list(host)


@contextlib.contextmanager
def host_cu_seqlens_scope(host: Optional[HostHint]):
    """Make ``host`` the active host copy of ``cu_seqlens`` for the enclosed calls."""

    token = _HOST_CU_SEQLENS.set(host)
    norm_token = _NORMALIZED.set(None)
    try:
        yield
    finally:
        _NORMALIZED.reset(norm_token)
        _HOST_CU_SEQLENS.reset(token)


def accepts_cu_seqlens_host(func):
    """Scope the keyword-only ``cu_seqlens_host`` argument around ``func``."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        host = kwargs.get("cu_seqlens_host")
        if host is None:
            return func(*args, **kwargs)
        with host_cu_seqlens_scope(host):
            return func(*args, **kwargs)

    return wrapper
