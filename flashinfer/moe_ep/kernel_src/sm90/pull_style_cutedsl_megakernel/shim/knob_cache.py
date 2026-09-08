# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Persistent knob cache for the SM90 Hopper MegaMoE frontends.

Sibling-fork mirror of ``kernel_src/cutedsl_megamoe/shim/knob_cache.py``:
offline-tuned winners land in a small JSON file and
``get_symm_buffer_for_hopper_fp8_mega_moe`` resolves ``knobs=None`` through
:func:`lookup_knobs` before falling back to the built-in
:func:`.tuner.default_knobs` heuristic (the kernel drop's token-bucket
table).  Resolution is a dict lookup — no compiles, no collectives.

The cache file is shared with the SM100 tree (same
``FLASHINFER_MOE_EP_KNOB_CACHE`` env / default path / JSON version); entries
never cross-match because this tree's keys carry the SM90 ``dtype`` values
plus an ``fp8_scale_mode`` field the SM100 entries do not have. MXFP4 dtype
identities encode fused versus split and the weight/activation ABI. MXFP4
entries additionally carry exact compute-capability, SM-count, and tuning
provenance axes, so a winner measured on another device or qualified against a
stale candidate domain cannot be replayed. Legacy MXFP4 entries without those
axes fail closed. Ordinary FP8 v1 entries retain their original schema and
matching behavior.

``max_tokens`` is the compile-time buffer capacity (the kernel compiles once
per buffer size); lookup picks the exact bucket when present, else the
smallest recorded bucket >= the requested size, else the largest below it.
All other key fields, including an explicitly supplied routing profile, must
match exactly — an untuned session deliberately falls back to the heuristic
instead of borrowing a neighbour's knobs.
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import json
import os
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Tuple

_CACHE_VERSION = 1
_MXFP4_DTYPE_PREFIX = "sm90_w_mxfp4_"
_MXFP4_HARDWARE_KEY_FIELDS = ("compute_capability", "sm_count")
_MXFP4_PROVENANCE_KEY_FIELD = "tuning_provenance_sha256"
_KEY_FIELDS = (
    "device",
    "dtype",
    "fp8_scale_mode",
    "world_size",
    "hidden",
    "intermediate",
    "num_experts",
    "topk",
    "gate_up_clamp",
)


def _cache_path() -> Optional[str]:
    raw = os.environ.get("FLASHINFER_MOE_EP_KNOB_CACHE", "")
    if raw.strip().lower() in ("0", "off", "none", "disable", "disabled"):
        return None
    if raw:
        return os.path.expanduser(raw)
    return os.path.expanduser("~/.cache/flashinfer/moe_ep_knob_cache.json")


def knob_cache_path() -> Optional[str]:
    """Resolved knob-cache file path, or None when the cache is disabled."""
    return _cache_path()


def _current_device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    return "cpu"


def _normalize_compute_capability(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(isinstance(part, bool) or not isinstance(part, int) for part in value)
    ):
        raise ValueError(
            f"compute_capability must be a (major, minor) integer pair, got {value!r}"
        )
    return [int(value[0]), int(value[1])]


def _mxfp4_hardware_identity(
    *, compute_capability: Any, sm_count: Optional[int]
) -> Tuple[Optional[List[int]], Optional[int]]:
    """Resolve extra MXFP4 hardware axes without changing FP8 cache keys."""
    capability = _normalize_compute_capability(compute_capability)
    sms = sm_count
    if capability is None or sms is None:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            if capability is None:
                capability = _normalize_compute_capability(
                    torch.cuda.get_device_capability(device)
                )
            if sms is None:
                sms = int(
                    torch.cuda.get_device_properties(device).multi_processor_count
                )
    if sms is not None and (
        isinstance(sms, bool) or not isinstance(sms, int) or sms <= 0
    ):
        raise ValueError(f"sm_count must be a positive integer, got {sms!r}")
    return capability, sms


def _is_mxfp4_dtype(dtype: Any) -> bool:
    return isinstance(dtype, str) and dtype.startswith(_MXFP4_DTYPE_PREFIX)


def _is_valid_tuning_provenance_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


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
    return [e for e in entries if isinstance(e, dict)]


def _knobs_to_json(knobs: Dict[str, Any]) -> Dict[str, Any]:
    # Tuples (mma tile / cluster / epi_flag_batch) become JSON lists.
    return {k: list(v) if isinstance(v, tuple) else v for k, v in knobs.items()}


def _knobs_from_json(knobs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in knobs.items()}


def _entry_matches_key(
    entry: Dict[str, Any],
    key: Dict[str, Any],
    *,
    routing_profile: Optional[str],
) -> bool:
    """Match one v1 entry, including the append-only routing-profile axis.

    Historical FP8 entries remain byte-for-byte compatible: a caller that
    does not request a routing profile matches only entries where the field is
    absent. MXFP4 additionally requires exact compute-capability, SM-count, and
    structured tuning-provenance fields; entries written before those axes were
    added fail closed.
    """

    if not all(entry.get(field) == key[field] for field in _KEY_FIELDS):
        return False
    if _is_mxfp4_dtype(key["dtype"]) and not all(
        entry.get(field) == key.get(field) for field in _MXFP4_HARDWARE_KEY_FIELDS
    ):
        return False
    if _is_mxfp4_dtype(key["dtype"]) and entry.get(
        _MXFP4_PROVENANCE_KEY_FIELD
    ) != key.get(_MXFP4_PROVENANCE_KEY_FIELD):
        return False
    if "routing_profile" in entry:
        return entry["routing_profile"] == routing_profile
    return routing_profile is None


def lookup_knobs(
    *,
    dtype: str,
    fp8_scale_mode: str,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    max_tokens: int,
    device: Optional[str] = None,
    gate_up_clamp: Optional[float] = None,
    routing_profile: Optional[str] = None,
    compute_capability: Any = None,
    sm_count: Optional[int] = None,
    tuning_provenance_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the cached knob dict for this session key, or ``None`` on miss."""
    path = _cache_path()
    if path is None:
        return None
    capability = resolved_sms = None
    if _is_mxfp4_dtype(dtype):
        if not _is_valid_tuning_provenance_sha256(tuning_provenance_sha256):
            warnings.warn(
                "[moe_ep-knob-cache] ignoring MXFP4 cache lookup without a "
                "valid tuning-provenance SHA-256.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        capability, resolved_sms = _mxfp4_hardware_identity(
            compute_capability=compute_capability,
            sm_count=sm_count,
        )
        if capability is None or resolved_sms is None:
            return None
    key = dict(
        device=device if device is not None else _current_device_name(),
        dtype=dtype,
        fp8_scale_mode=fp8_scale_mode,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        gate_up_clamp=gate_up_clamp,
    )
    if _is_mxfp4_dtype(dtype):
        key.update(
            compute_capability=capability,
            sm_count=resolved_sms,
            tuning_provenance_sha256=tuning_provenance_sha256,
        )
    matches = [
        e
        for e in _load_entries(path)
        if _entry_matches_key(e, key, routing_profile=routing_profile)
        and isinstance(e.get("knobs"), dict)
        and isinstance(e.get("max_tokens"), int)
    ]
    if not matches:
        return None
    at_or_above = [e for e in matches if e["max_tokens"] >= max_tokens]
    if at_or_above:
        best = min(at_or_above, key=lambda e: e["max_tokens"])
    else:
        best = max(matches, key=lambda e: e["max_tokens"])
    return _knobs_from_json(best["knobs"])


def record_knobs(
    knobs: Dict[str, Any],
    *,
    dtype: str,
    fp8_scale_mode: str,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    max_tokens: int,
    device: Optional[str] = None,
    gate_up_clamp: Optional[float] = None,
    routing_profile: Optional[str] = None,
    compute_capability: Any = None,
    sm_count: Optional[int] = None,
    tuning_provenance_sha256: Optional[str] = None,
    p50_us: Optional[float] = None,
    source: str = "autotune",
) -> Optional[str]:
    """Upsert one tuned entry (exact key incl. ``max_tokens``); atomic write.

    Returns the cache path written, or ``None`` when the cache is disabled or
    the write failed (recording is best-effort).
    """
    path = _cache_path()
    if path is None:
        return None
    capability = resolved_sms = None
    if _is_mxfp4_dtype(dtype):
        if not _is_valid_tuning_provenance_sha256(tuning_provenance_sha256):
            warnings.warn(
                "[moe_ep-knob-cache] refusing to record MXFP4 knobs without a "
                "valid tuning-provenance SHA-256.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        capability, resolved_sms = _mxfp4_hardware_identity(
            compute_capability=compute_capability,
            sm_count=sm_count,
        )
        if capability is None or resolved_sms is None:
            warnings.warn(
                "[moe_ep-knob-cache] refusing to record MXFP4 knobs without "
                "compute-capability and SM-count identity.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
    entry = dict(
        device=device if device is not None else _current_device_name(),
        dtype=dtype,
        fp8_scale_mode=fp8_scale_mode,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        max_tokens=max_tokens,
        gate_up_clamp=gate_up_clamp,
        knobs=_knobs_to_json(knobs),
        p50_us=p50_us,
        source=source,
        tuned_at=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    if _is_mxfp4_dtype(dtype):
        entry.update(
            compute_capability=capability,
            sm_count=resolved_sms,
            tuning_provenance_sha256=tuning_provenance_sha256,
        )
    if routing_profile is not None:
        entry["routing_profile"] = routing_profile
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        lock_path = path + ".lock"
        with open(lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                entries = _load_entries(path)
                entries = [
                    e
                    for e in entries
                    if not (
                        _entry_matches_key(
                            e,
                            entry,
                            routing_profile=routing_profile,
                        )
                        and e.get("max_tokens") == max_tokens
                    )
                ]
                entries.append(entry)
                fd, tmp = tempfile.mkstemp(dir=directory, prefix=".moe_ep_knob_cache.")
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(
                            {"version": _CACHE_VERSION, "entries": entries}, f, indent=1
                        )
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp, path)
                except BaseException:
                    with contextlib.suppress(OSError):
                        os.unlink(tmp)
                    raise
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
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
    fp8_scale_mode: str,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    max_tokens: int,
    gate_up_clamp: Optional[float] = None,
) -> Tuple[Dict[str, Any], str]:
    """Pure-lookup knob resolution: cache hit, else built-in heuristic.

    Returns ``(knobs, source)`` where source is ``"cache"`` or
    ``"heuristic"``.  Cheap and deterministic — safe on the engine hot path
    (called once per buffer creation, never per forward).
    """
    cached = lookup_knobs(
        dtype=dtype,
        fp8_scale_mode=fp8_scale_mode,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        max_tokens=max_tokens,
        gate_up_clamp=gate_up_clamp,
    )
    if cached is not None:
        return cached, "cache"
    from .tuner import default_knobs

    return default_knobs(max_tokens, fp8_scale_mode=fp8_scale_mode), "heuristic"


__all__ = [
    "knob_cache_path",
    "lookup_knobs",
    "record_knobs",
    "resolve_knobs",
]
