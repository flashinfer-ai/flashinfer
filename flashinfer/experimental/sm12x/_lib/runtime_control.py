# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/cute/runtime_control.py @ 9f2eb830 (2026-05-27) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
from __future__ import annotations

import hashlib
import inspect
from threading import Lock


class KernelResolutionFrozenError(RuntimeError):
    """Raised when sm12x is asked to resolve a new kernel after freeze."""


_STATE_LOCK = Lock()
_FROZEN = False
_FREEZE_REASON: str | None = None


def freeze_kernel_resolution(reason: str | None = None) -> None:
    global _FROZEN, _FREEZE_REASON
    with _STATE_LOCK:
        _FROZEN = True
        _FREEZE_REASON = reason


def unfreeze_kernel_resolution() -> None:
    global _FROZEN, _FREEZE_REASON
    with _STATE_LOCK:
        _FROZEN = False
        _FREEZE_REASON = None


def kernel_resolution_frozen() -> bool:
    with _STATE_LOCK:
        return _FROZEN


freeze_compilation = freeze_kernel_resolution
unfreeze_compilation = unfreeze_kernel_resolution
compilation_frozen = kernel_resolution_frozen


def raise_if_kernel_resolution_frozen(
    kind: str,
    *,
    target: object | None = None,
    cache_key: object | None = None,
) -> None:
    with _STATE_LOCK:
        frozen = _FROZEN
        reason = _FREEZE_REASON
    if not frozen:
        return

    details = [f"sm12x kernel resolution is frozen; refusing {kind}"]
    target_name = _describe_target(target)
    if target_name is not None:
        details.append(f"target={target_name}")
    if cache_key is not None:
        details.append(f"key={_summarize_cache_key(cache_key)}")
    if reason is not None:
        details.append(f"reason={reason}")
    details.append(
        "warm up this kernel shape before calling flashinfer.experimental.sm12x.freeze_kernel_resolution()"
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
