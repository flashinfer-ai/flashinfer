# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Exact-workload cache for Blackwell GenPhase performance knobs.

Entries share the MoE EP cache file but use a distinct device namespace.
Capacity, live-token bucket, operation attributes, hardware and compiler must
match. Missing or invalid entries fall back to the geometry estimate.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

from .tuning import KNOB_KEYS, is_valid_sm100_genphase_knobs, sm100_genphase_candidates

_REVISION = "sm100-genphase-3c1d1c2f-auto-v3"


def knob_cache_path() -> str | None:
    """Resolve the shared knob-cache path, or None when disabled by environment."""
    value = os.environ.get("FLASHINFER_MOE_EP_KNOB_CACHE", "")
    if value.strip().lower() in ("0", "off", "none", "disable", "disabled"):
        return None
    return os.path.expanduser(value or "~/.cache/flashinfer/moe_ep_knob_cache.json")


def device_identity() -> str:
    """Identify the current GPU and compiler stack for exact cache matching.

    Native target aliases normalize before DSL initialization so a fresh
    process can reuse prior tuning. Compiler options remain part of the key.
    """
    import torch
    from flashinfer.jit.cute_dsl_core import _get_cute_dsl_version

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    native_arch = f"sm_{props.major}{props.minor}a"
    compile_arch = os.environ.get("CUTE_DSL_ARCH", native_arch)
    if compile_arch == native_arch.removesuffix("a"):
        compile_arch = native_arch
    return json.dumps(
        dict(
            name=props.name,
            cc=[props.major, props.minor],
            sms=props.multi_processor_count,
            cuda=torch.version.cuda,
            torch=torch.__version__,
            dsl_stack=_get_cute_dsl_version(),
            compile_arch=compile_arch,
            compiler_options=os.environ.get("CUTE_DSL_COMPILER_OPT", ""),
            instrumentation=os.environ.get("DKG_IKET_INSTRUMENTATION_METHOD", ""),
        ),
        sort_keys=True,
    )


def _key(config, device: str | None = None) -> dict:
    """Key operation attributes and capacity independently of rank and knobs."""
    spec = dataclasses.asdict(config)
    for name in ("rank", *KNOB_KEYS):
        spec.pop(name)
    return dict(
        backend_revision=_REVISION,
        device="sm100-genphase:" + (device_identity() if device is None else device),
        problem=json.loads(json.dumps(spec)),
    )


def _read(path: str) -> dict:
    """Read a supported cache document, treating a missing file as an empty cache."""
    from flashinfer.moe_ep.core.tuning import KNOB_CACHE_VERSION

    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return dict(version=KNOB_CACHE_VERSION, entries=[])
    if (
        not isinstance(data, dict)
        or data.get("version") != KNOB_CACHE_VERSION
        or not isinstance(data.get("entries"), list)
    ):
        raise ValueError("unsupported MoE EP knob-cache format")
    return data


def _entries(config, device=None) -> list[dict]:
    """Find exact workload entries; warn and ignore unreadable cache files."""
    path = knob_cache_path()
    if path is None:
        return []
    try:
        data = _read(path)
    except (ValueError, OSError) as exc:
        warnings.warn(
            f"Ignoring Blackwell GenPhase knob cache: {exc}",
            RuntimeWarning,
            stacklevel=3,
        )
        return []
    key = _key(config, device)
    return [
        e
        for e in data["entries"]
        if isinstance(e, dict) and all(e.get(k) == v for k, v in key.items())
    ]


def _decode(entry, config):
    """Restore knob tuples and reject malformed or inadmissible cached values."""
    raw = entry.get("knobs")
    if not isinstance(raw, dict) or set(raw) != set(KNOB_KEYS):
        return None
    knobs = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    try:
        return knobs if is_valid_sm100_genphase_knobs(knobs, config) else None
    except (TypeError, ValueError):
        return None


def live_token_buckets(config, *, device=None) -> list[int]:
    """List valid cached live-token counts for this capacity and device identity."""
    return sorted(
        {
            e["live_tokens"]
            for e in _entries(config, device)
            if type(e.get("live_tokens")) is int
            and 0 <= e["live_tokens"] <= config.max_tokens_per_rank
            and _decode(e, config) is not None
        }
    )


def resolve_knobs(
    config,
    *,
    sm_count: int,
    live_tokens=None,
    constraints=None,
    device=None,
    use_cache=True,
) -> tuple[dict[str, Any], str]:
    """Return exact cached knobs or a heuristic, together with their source.

    Cache hits must match the live-token count and explicit constraints.
    ``use_cache=False`` starts from the heuristic for a fresh tuning sweep.
    """
    candidates = sm100_genphase_candidates(
        config, sm_count=sm_count, live_tokens=live_tokens, constraints=constraints
    )
    tokens = config.max_tokens_per_rank if live_tokens is None else live_tokens
    if use_cache:
        for entry in reversed(_entries(config, device)):
            if (
                type(entry.get("live_tokens")) is not int
                or entry["live_tokens"] != tokens
            ):
                continue
            knobs = _decode(entry, config)
            if knobs is not None and all(
                knobs[k] == v for k, v in (constraints or {}).items()
            ):
                return knobs, "cache"
    return candidates[0], "heuristic"


def record_knobs(
    knobs, config, *, live_tokens: int, p50_us: float, device=None
) -> bool:
    """Atomically preserve unrelated entries; cache I/O failure is nonfatal."""
    if set(knobs) != set(KNOB_KEYS) or not is_valid_sm100_genphase_knobs(knobs, config):
        raise ValueError("cannot cache invalid Blackwell GenPhase knobs")
    if (
        type(live_tokens) is not int
        or not 0 <= live_tokens <= config.max_tokens_per_rank
    ):
        raise ValueError("live_tokens must be within workspace capacity")
    if not math.isfinite(p50_us) or p50_us <= 0:
        raise ValueError("p50_us must be finite and positive")
    path = knob_cache_path()
    if path is None:
        return False
    entry = dict(
        **_key(config, device),
        live_tokens=live_tokens,
        knobs=knobs,
        p50_us=p50_us,
        source="cuda-graph-max-rank-sampled-oracle",
    )
    identity = ("backend_revision", "device", "problem", "live_tokens")
    temporary = None
    try:
        import fcntl

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = _read(path)
            data["entries"] = [
                e
                for e in data["entries"]
                if not isinstance(e, dict)
                or not all(e.get(k) == entry[k] for k in identity)
            ]
            data["entries"].append(entry)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=Path(path).parent, prefix=".genphase-knobs-", delete=False
            ) as stream:
                temporary = stream.name
                json.dump(data, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
        return True
    except (ValueError, OSError) as exc:
        warnings.warn(
            f"Could not persist Blackwell GenPhase knobs: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
