# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Persistent knob cache: offline-tuned winners feeding a pure-lookup hot path.

The online ``knobs="auto"`` sweep is unusable inside a serving engine (~24
``cute.compile``\\ s + collective barriers at first compute — see
the 2026-07-16 vLLM e2e runs).  This module makes tuning
a write-once/offline step: winners land in a small JSON file keyed by
(device, dtype, world_size, geometry, combine wire, token-bucket), and
``get_symm_buffer_for_*`` resolves ``knobs=None`` through :func:`lookup_knobs`
before falling back to the built-in :func:`.tuner.default_knobs` heuristic.
Resolution is a dict lookup — no compiles, no collectives, no timing.

Populate the cache with the offline CLI (``python -m flashinfer.moe_ep.tune``)
or implicitly by running ``knobs="auto"`` once outside the engine (autotune
winners are recorded here by rank 0).

File location: ``FLASHINFER_MOE_EP_KNOB_CACHE`` (a path, or ``0``/``off`` to
disable the cache entirely), default
``~/.cache/flashinfer/moe_ep_knob_cache.json``.  Format::

    {"version": 1,
     "entries": [{"device": "NVIDIA GB200", "dtype": "nvfp4",
                  "world_size": 4, "hidden": 7168, "intermediate": 2048,
                  "num_experts": 256, "topk": 8, "combine_dtype": "bf16",
                  "max_tokens": 2048, "knobs": {...},
                  "p50_us": 585.0, "source": "autotune",
                  "tuned_at": "2026-07-16T12:00:00"}, ...]}

``max_tokens`` is the compile-time buffer capacity (the kernel compiles once
per buffer size); lookup picks the exact bucket when present, else the
smallest recorded bucket >= the requested size, else the largest below it.
All other key fields must match exactly — an untuned geometry deliberately
falls back to the heuristic instead of borrowing a neighbour's knobs.

If enabling ikr finds a faster config, each shape can have two winners:
On for each of ikr on/off
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Tuple

_CACHE_VERSION = 1
_KEY_FIELDS = (
    "device",
    "dtype",
    "world_size",
    "hidden",
    "intermediate",
    "num_experts",
    "topk",
    "combine_dtype",
)


def _cache_path() -> Optional[str]:
    raw = os.environ.get("FLASHINFER_MOE_EP_KNOB_CACHE", "")
    if raw.strip().lower() in ("0", "off", "none", "disable", "disabled"):
        return None
    if raw:
        return os.path.expanduser(raw)
    return os.path.expanduser("~/.cache/flashinfer/moe_ep_knob_cache.json")


def knob_cache_path() -> Optional[str]:
    """Resolved knob-cache file path, or None when the cache is disabled.

    Public wrapper so callers outside the shim never import ``_cache_path``
    (tests monkeypatch that name; routing through this wrapper at call time
    keeps their patches effective).
    """
    return _cache_path()


def _current_device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    return "cpu"


def _load_entries(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError) as exc:
        warnings.warn(
            f"[moe_ep-knob-cache] unreadable cache {path!r} ({exc}); "
            "ignoring it (lookups fall back to the built-in heuristic).",
            RuntimeWarning,
            stacklevel=3,
        )
        return []
    if not isinstance(data, dict) or data.get("version") != _CACHE_VERSION:
        warnings.warn(
            f"[moe_ep-knob-cache] {path!r} has unsupported version "
            f"{data.get('version') if isinstance(data, dict) else '?'}; ignoring.",
            RuntimeWarning,
            stacklevel=3,
        )
        return []
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        return []
    # Drop non-dict elements too: lookup_knobs/record_knobs call e.get() on
    # every entry, and a corrupted-but-valid-JSON cache must degrade, not raise.
    return [e for e in entries if isinstance(e, dict)]


def _knobs_to_json(knobs: Dict[str, Any]) -> Dict[str, Any]:
    # Tuples (mma tile / cluster / epi_flag_batch) become JSON lists.
    return {k: list(v) if isinstance(v, tuple) else v for k, v in knobs.items()}


def _knobs_from_json(knobs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in knobs.items()}


def _entry_needs_ikr(entry: Dict[str, Any]) -> bool:
    """Whether a recorded winner requires ikr"""
    knobs = entry.get("knobs")
    return bool(isinstance(knobs, dict) and knobs.get("in_kernel_fc2_reduce"))


def lookup_knobs(
    *,
    dtype: str,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    max_tokens: int,
    combine_dtype: str = "bf16",
    enable_in_kernel_fc2_reduce: bool = False,
    device: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the cached knob dict for this session key, or ``None`` on miss.

    ``enable_in_kernel_fc2_reduce`` indicates whether the user has opted in to in_kernel_fc2_reduce.
    """
    path = _cache_path()
    if path is None:
        return None
    key = dict(
        device=device if device is not None else _current_device_name(),
        dtype=dtype,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        combine_dtype=combine_dtype,
    )
    # BF16 and BF16-MXFP8 have different user interfaces for in_kernel_fc2_reduce, so we filter by exact match
    # MXFP8 and NVFP4 can turn it on or off, without changing the API surface
    # TODO: Remove this if we fuse the reduction into the kernel without IKR, like for MXFP8/NVFP4
    session_owns_ikr = dtype in (
        "bf16",
        "bf16_mxfp8",
        "bf16_mxfp8_e4m3",
        "bf16_mxfp8_e5m2",
    )
    matches = [
        e
        for e in _load_entries(path)
        if all(e.get(f) == key[f] for f in _KEY_FIELDS)
        and isinstance(e.get("knobs"), dict)
        and isinstance(e.get("max_tokens"), int)
        and (
            _entry_needs_ikr(e) == bool(enable_in_kernel_fc2_reduce)
            if session_owns_ikr
            else (enable_in_kernel_fc2_reduce or not _entry_needs_ikr(e))
        )
    ]
    if not matches:
        return None
    # We use IKR to break ties. If IKR is not allowed, that will be filtered out when constructing matches
    at_or_above = [e for e in matches if e["max_tokens"] >= max_tokens]
    if at_or_above:
        best = min(
            at_or_above, key=lambda e: (e["max_tokens"], not _entry_needs_ikr(e))
        )
    else:
        best = max(matches, key=lambda e: (e["max_tokens"], _entry_needs_ikr(e)))
    return _knobs_from_json(best["knobs"])


def record_knobs(
    knobs: Dict[str, Any],
    *,
    dtype: str,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    max_tokens: int,
    combine_dtype: str = "bf16",
    device: Optional[str] = None,
    p50_us: Optional[float] = None,
    source: str = "autotune",
) -> Optional[str]:
    """Upsert one tuned entry (exact key incl. ``max_tokens``); atomic write.

    Deterministic and ikr-tuned winners for the same geometry are separate
    entries (see :func:`_entry_needs_ikr`), so a ``--allow-nondeterministic``
    sweep does not evict the deterministic winner a reproducible session needs,
    or vice versa.

    Returns the cache path written, or ``None`` when the cache is disabled or
    the write failed (recording is best-effort — a read-only home directory
    must not break tuning).
    """
    path = _cache_path()
    if path is None:
        return None
    entry = dict(
        device=device if device is not None else _current_device_name(),
        dtype=dtype,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        combine_dtype=combine_dtype,
        max_tokens=max_tokens,
        knobs=_knobs_to_json(knobs),
        p50_us=p50_us,
        source=source,
        tuned_at=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    try:
        entries = _load_entries(path)
        entries = [
            e
            for e in entries
            if not (
                all(e.get(f) == entry[f] for f in _KEY_FIELDS)
                and e.get("max_tokens") == max_tokens
                and _entry_needs_ikr(e) == _entry_needs_ikr(entry)
            )
        ]
        entries.append(entry)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(path), prefix=".moe_ep_knob_cache."
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"version": _CACHE_VERSION, "entries": entries}, f, indent=1)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    except (OSError, TypeError, ValueError) as exc:
        warnings.warn(
            f"[moe_ep-knob-cache] could not write {path!r}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return path


def resolve_knobs(
    *,
    dtype: str,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    max_tokens: int,
    combine_dtype: str = "bf16",
    enable_in_kernel_fc2_reduce: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """Pure-lookup knob resolution: cache hit, else built-in heuristic.

    Returns ``(knobs, source)`` where source is ``"cache"`` or ``"heuristic"``.
    Cheap and deterministic — safe on the engine hot path (called once per
    buffer creation, never per forward).

    ``combine_dtype`` and ``enable_in_kernel_fc2_reduce`` describe the calling
    session, so resolution can never hand back a knob set the session's config
    would reject: the heuristic is built against them (see
    :func:`.tuner.default_knobs`) and the lookup only considers entries tuned
    under a compatible ikr objective (see :func:`lookup_knobs`), so a cache
    populated with ``--allow-nondeterministic`` stays safe to share with
    sessions that need a reproducible combine.
    """
    cached = lookup_knobs(
        dtype=dtype,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        max_tokens=max_tokens,
        combine_dtype=combine_dtype,
        enable_in_kernel_fc2_reduce=enable_in_kernel_fc2_reduce,
    )
    if cached is not None:
        return cached, "cache"
    from .tuner import default_knobs

    # The cache keys on the precise element kind (``mxfp8_e4m3`` and
    # ``mxfp8_e5m2`` are separate compiles); the profiles are per kernel.
    if dtype.startswith("bf16_mxfp8"):
        heuristic_dtype = "bf16_mxfp8"
    elif dtype.startswith("mxfp8"):
        heuristic_dtype = "mxfp8"
    else:
        heuristic_dtype = dtype
    return (
        default_knobs(
            max_tokens,
            dtype=heuristic_dtype,
            combine_dtype=combine_dtype,
            enable_in_kernel_fc2_reduce=enable_in_kernel_fc2_reduce,
        ),
        "heuristic",
    )
