# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Persistent knob cache for the SM107 block-scaled mega kernel.

Same file format, version, and location as the SM100 tree's
``cutedsl_megamoe/shim/knob_cache.py`` (the two trees are process-exclusive,
so the module is reimplemented here rather than imported): winners land in a
small JSON file keyed by (device, dtype, world_size, geometry, combine wire,
token-bucket), and knob resolution is a pure dict lookup — no compiles, no
collectives.  SM107 entries never collide with SM100 ones because the
``device`` key differs (and so do the knob key names).

Populate with the offline CLI (``python -m flashinfer.moe_ep.tune`` on a
Rubin node) — see ``.autotune`` for the collective sweep.

File location: ``FLASHINFER_MOE_EP_KNOB_CACHE`` (a path, or ``0``/``off`` to
disable), default ``~/.cache/flashinfer/moe_ep_knob_cache.json``.

``max_tokens`` is the session buffer capacity; lookup picks the exact bucket
when present, else the smallest recorded bucket >= the requested size, else
the largest below it.  All other key fields must match exactly.
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
    """Resolved knob-cache file path, or None when the cache is disabled."""
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
            f"[moe_ep-knob-cache] unreadable cache {path!r} ({exc}); ignoring.",
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
    # Tuples (tile / cluster / epi flag batches / schedule_policy) -> lists.
    return {k: list(v) if isinstance(v, tuple) else v for k, v in knobs.items()}


def _knobs_from_json(knobs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in knobs.items()}


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
    device: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the cached knob dict for this session key, or ``None`` on miss."""
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
    matches = [
        e
        for e in _load_entries(path)
        if all(e.get(f) == key[f] for f in _KEY_FIELDS)
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

    Best-effort: returns the cache path written, or ``None`` when the cache is
    disabled or the write failed.
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


def default_knobs(max_tokens: int, *, quant_kind: str = "nvfp4") -> Dict[str, Any]:
    """Built-in SM107 heuristic: the upstream Rubin perf-report selected-best profile.

    Two token profiles (see TUNING.md): <2048 tokens/rank keeps the 128-wide
    N tile, >=2048 the 256-wide.  Both use mixed CGA (4x1 preferred, 2x1
    fallback), phase-interleave with the kernel-resolved minimum hint, atomic
    work IDs, FC2 bulk TMA stage 2, epi-warp token back, and separate top-k
    reduction.  Tile K is the kind's 2x-mode depth (2 x instruction K).

    Returns a fresh dict each call.
    """
    tile_k = 256 if quant_kind == "nvfp4" else 128
    tile_n = 128 if max_tokens < 2048 else 256
    return {
        "mma_tiler_mnk": (256, tile_n, tile_k),
        "cluster_shape_mn": (4, 1),
        "fallback_cluster_shape_mn": (2, 1),
        "schedule_policy": ("phase_interleave", None),
        "work_id_mode": "atomic_counter",
        "fc2_use_bulk": True,
        "fc2_tma_stages": 2,
        "epi_flag_batches": (1, 4),
        "token_in_flag_batch": 1,
        "token_back_mode": "epi_warps",
        "reduce_topk_in_kernel": False,
    }


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
) -> Tuple[Dict[str, Any], str]:
    """Pure-lookup knob resolution: cache hit, else the built-in heuristic.

    Returns ``(knobs, source)`` where source is ``"cache"`` or ``"heuristic"``.
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
    )
    if cached is not None:
        return cached, "cache"
    return default_knobs(max_tokens, quant_kind=dtype), "heuristic"


__all__ = [
    "default_knobs",
    "knob_cache_path",
    "lookup_knobs",
    "record_knobs",
    "resolve_knobs",
]
