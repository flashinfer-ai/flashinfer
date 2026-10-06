from __future__ import annotations

import hashlib
import inspect
from contextlib import contextmanager
from threading import Lock


class KernelResolutionFrozenError(RuntimeError):
    """Raised when b12x is asked to resolve a new kernel after freeze."""


_STATE_LOCK = Lock()
_GUARDS: dict[object, str | None] = {}


@contextmanager
def kernel_resolution_guard(reason: str | None = None):
    """Each capture/session owns its guard; leaving one cannot unfreeze another."""
    token = object()
    with _STATE_LOCK:
        _GUARDS[token] = reason
    try:
        yield
    finally:
        with _STATE_LOCK:
            del _GUARDS[token]


def kernel_resolution_frozen() -> bool:
    with _STATE_LOCK:
        return bool(_GUARDS)



def raise_if_kernel_resolution_frozen(
    kind: str,
    *,
    target: object | None = None,
    cache_key: object | None = None,
) -> None:
    with _STATE_LOCK:
        frozen = bool(_GUARDS)
        reason = next(reversed(_GUARDS.values()), None)
    if not frozen:
        return

    details = [f"b12x kernel resolution is frozen; refusing {kind}"]
    target_name = _describe_target(target)
    if target_name is not None:
        details.append(f"target={target_name}")
    if cache_key is not None:
        details.append(f"key={_summarize_cache_key(cache_key)}")
    if reason is not None:
        details.append(f"reason={reason}")
    details.append(
        "prepare this execution before entering capture or freezing its session"
    )
    raise KernelResolutionFrozenError("; ".join(details))


def _describe_target(target: object | None) -> str | None:
    if target is None:
        return None
    if inspect.ismethod(target):
        module = getattr(target.__func__, "__module__", "")
        qualname = getattr(
            target.__func__, "__qualname__", getattr(target.__func__, "__name__", "")
        )
        return f"{module}.{qualname}" if module else qualname
    if inspect.isfunction(target):
        module = getattr(target, "__module__", "")
        qualname = getattr(target, "__qualname__", getattr(target, "__name__", ""))
        return f"{module}.{qualname}" if module else qualname
    target_type = type(target)
    module = getattr(target_type, "__module__", "")
    qualname = getattr(target_type, "__qualname__", target_type.__name__)
    return f"{module}.{qualname}" if module else qualname


def _summarize_cache_key(cache_key: object) -> str:
    text = repr(cache_key)
    if len(text) > 120:
        text = text[:117] + "..."
    digest = hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()[:12]
    return f"{text} [{digest}]"
