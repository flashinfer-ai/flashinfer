# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persistent tuned-knob cache of the Cake SM120 MegaMoE backend.

The offline tuner (``python -m flashinfer.moe_ep.tune --dtype sm120_mxfp4_mxfp8``,
see :mod:`.tuner`) times every feasible transport table of a deployment and records
the winner here; ``transport_table="tuned"`` resolves it at workspace creation with a
dict lookup (no timing, no compile; rank 0 reads the file and broadcasts).  Nothing
runs on the per-call path.

The file is the moe_ep knob cache shared with the other mega backends
(``FLASHINFER_MOE_EP_KNOB_CACHE``, default ``~/.cache/flashinfer/moe_ep_knob_cache.json``,
format version ``KNOB_CACHE_VERSION``).  Entries of this backend carry ``dtype="sm120_mxfp4_mxfp8_cake"``
plus a ``topology`` digest of the gathered LSA-reachability matrix and GIN
availability, so they never match another backend's key and never leak to a host
whose peer topology differs.  ``max_tokens`` is the workspace row capacity; lookup
picks the exact bucket, else the smallest recorded bucket above it, else the largest
below it.  Every other key field must match exactly.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import tempfile
import warnings
from typing import Any, Optional

DTYPE = "sm120_mxfp4_mxfp8_cake"
_KEY_FIELDS = (
    "device",
    "dtype",
    "variant",
    "world_size",
    "hidden",
    "intermediate",
    "num_experts",
    "topk",
    "topology",
)


def knob_cache_path() -> Optional[str]:
    """Resolved knob-cache file path, or ``None`` when the cache is disabled."""

    raw = os.environ.get("FLASHINFER_MOE_EP_KNOB_CACHE", "")
    if raw.strip().lower() in ("0", "off", "none", "disable", "disabled"):
        return None
    if raw:
        return os.path.expanduser(raw)
    return os.path.expanduser("~/.cache/flashinfer/moe_ep_knob_cache.json")


def topology_digest(lsa_pair_matrix: Any, gin_available: bool) -> str:
    """Short digest of the peer topology a transport table is valid for."""

    rows = [[bool(v) for v in row] for row in lsa_pair_matrix]
    text = json.dumps({"lsa": rows, "gin": bool(gin_available)}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def current_device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    return "cpu"


def _load_entries(path: str) -> list[dict[str, Any]]:
    from flashinfer.moe_ep.core.tuning import KNOB_CACHE_VERSION

    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError) as exc:
        warnings.warn(
            f"[moe_ep-knob-cache] unreadable cache {path!r} ({exc}); ignoring it.",
            RuntimeWarning,
            stacklevel=3,
        )
        return []
    if not isinstance(data, dict) or data.get("version") != KNOB_CACHE_VERSION:
        return []
    entries = data.get("entries", [])
    return (
        [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []
    )


def _key(**fields: Any) -> dict[str, Any]:
    key = {"dtype": DTYPE, **fields}
    if key.get("device") is None:
        key["device"] = current_device_name()
    missing = [f for f in _KEY_FIELDS if f not in key]
    if missing:
        raise TypeError(f"missing knob-cache key fields {missing}")
    return key


def lookup_knobs(*, max_tokens: int, **fields: Any) -> Optional[dict[str, Any]]:
    """The cached knob dict for this deployment key, or ``None`` on a miss."""

    path = knob_cache_path()
    if path is None:
        return None
    key = _key(**fields)
    matches = [
        e
        for e in _load_entries(path)
        if all(e.get(f) == key[f] for f in _KEY_FIELDS)
        and isinstance(e.get("knobs"), dict)
        and isinstance(e.get("max_tokens"), int)
    ]
    if not matches:
        return None
    above = [e for e in matches if e["max_tokens"] >= max_tokens]
    best = (
        min(above, key=lambda e: e["max_tokens"])
        if above
        else max(matches, key=lambda e: e["max_tokens"])
    )
    return dict(best["knobs"])


def record_knobs(
    knobs: dict[str, Any],
    *,
    max_tokens: int,
    p50_us: Optional[float] = None,
    candidates_us: Optional[dict[str, float]] = None,
    **fields: Any,
) -> Optional[str]:
    """Upsert one tuned entry (exact key incl. ``max_tokens``); atomic write.

    Returns the path written, or ``None`` when the cache is disabled or the write
    failed (recording is best-effort)."""

    path = knob_cache_path()
    if path is None:
        return None
    entry = {
        **_key(**fields),
        "max_tokens": int(max_tokens),
        "knobs": knobs,
        "p50_us": p50_us,
        "candidates_us": candidates_us,
        "source": "autotune",
        "tuned_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        entries = [
            e
            for e in _load_entries(path)
            if not (
                all(e.get(f) == entry[f] for f in _KEY_FIELDS)
                and e.get("max_tokens") == entry["max_tokens"]
            )
        ]
        entries.append(entry)
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".moe_ep_knob_cache.")
        from flashinfer.moe_ep.core.tuning import KNOB_CACHE_VERSION

        try:
            with os.fdopen(fd, "w") as f:
                json.dump(
                    {"version": KNOB_CACHE_VERSION, "entries": entries}, f, indent=1
                )
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


__all__ = [
    "DTYPE",
    "current_device_name",
    "knob_cache_path",
    "lookup_knobs",
    "record_knobs",
    "topology_digest",
]
