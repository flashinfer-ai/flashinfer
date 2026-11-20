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

    {"version": 2,
     "entries": [{"device": "NVIDIA GB200", "dtype": "nvfp4",
                  "world_size": 4, "hidden": 7168, "intermediate": 2048,
                  "num_experts": 256, "topk": 8, "combine_dtype": "bf16",
                  "apply_routing_weights_before_fc2": true,
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from flashinfer.moe_ep._compat import deprecated_aliases

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


def _batch_invariant_version(
    dtype: str, batch_invariant: bool, version: Optional[int]
) -> Optional[int]:
    if not batch_invariant:
        return None
    if dtype in ("bf16_nvfp4", "bf16_mxfp8_e4m3", "bf16_mxfp8_e5m2"):
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.a16_batch_invariant import (
            A16_BATCH_INVARIANT_VERSION,
        )

        return A16_BATCH_INVARIANT_VERSION if version is None else version
    if dtype in ("mxfp8_e4m3", "mxfp8_e5m2"):
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.mxfp8_batch_invariant import (
            MXFP8_BATCH_INVARIANT_VERSION,
        )

        return MXFP8_BATCH_INVARIANT_VERSION if version is None else version
    if dtype not in ("bf16", "nvfp4"):
        raise ValueError("unsupported batch_invariant knob-cache dtype")
    if dtype == "nvfp4":
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.nvfp4_batch_invariant import (
            NVFP4_BATCH_INVARIANT_VERSION,
        )

        return NVFP4_BATCH_INVARIANT_VERSION if version is None else version
    if version is None:
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
            BF16_BATCH_INVARIANT_VERSION,
        )

        return BF16_BATCH_INVARIANT_VERSION
    return version


def _arithmetic_fingerprint(
    *,
    batch_invariant: bool,
    dtype: str,
    input_norm_const: float,
    activation: str = "swiglu",
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    per_token_scale: bool = False,
    handoff_norm: bool = False,
    output_mode: str = "finalized",
    combine_dtype: str = "bf16",
    hidden: int,
    intermediate: int,
    topk: int,
    num_experts: int,
    apply_routing_weights_before_fc2: bool,
    gate_up_clamp: Optional[float],
) -> Optional[str]:
    if not batch_invariant:
        return None
    from types import SimpleNamespace
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.activation_batch_invariant import (
        batch_invariant_variant_kwargs,
    )

    if per_token_scale and dtype != "nvfp4":
        raise ValueError("per-token FC1 scaling is only supported by NVFP4")
    variants = batch_invariant_variant_kwargs(
        SimpleNamespace(
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            enable_fc1_activation_per_token_scale=per_token_scale,
        )
    )
    if dtype in ("bf16_nvfp4", "bf16_mxfp8_e4m3", "bf16_mxfp8_e5m2"):
        from flashinfer.moe_ep.batch_invariant.a16_spec import (
            spec_fingerprint as a16_fingerprint,
        )
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
            bf16_batch_invariant_arch,
        )

        kind = {
            "bf16_nvfp4": "w4a16",
            "bf16_mxfp8_e4m3": "w8a16_e4m3",
            "bf16_mxfp8_e5m2": "w8a16_e5m2",
        }[dtype]
        return a16_fingerprint(
            hidden,
            intermediate,
            topk,
            num_experts,
            kind=kind,
            arch=bf16_batch_invariant_arch(),
            apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
            gate_up_clamp=gate_up_clamp,
            handoff_norm=handoff_norm,
            output_mode=output_mode,
            **variants,
        )
    if output_mode != "finalized":
        variants["output_mode"] = output_mode
    if combine_dtype != "bf16":
        variants["combine_dtype"] = combine_dtype
    if dtype in ("mxfp8_e4m3", "mxfp8_e5m2"):
        from flashinfer.moe_ep.batch_invariant.mxfp8_spec import (
            spec_fingerprint as mxfp8_spec_fingerprint,
        )
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.nvfp4_batch_invariant import (
            nvfp4_batch_invariant_arch,
        )

        if isinstance(input_norm_const, bool) or input_norm_const != 1.0:
            raise ValueError("batch_invariant MXFP8 requires input_norm_const=1.0")
        return mxfp8_spec_fingerprint(
            hidden=hidden,
            intermediate=intermediate,
            top_k=topk,
            num_experts=num_experts,
            arch=nvfp4_batch_invariant_arch(),
            quant_kind=dtype,
            implementation="legacy",
            apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
            gate_up_clamp=gate_up_clamp,
            **variants,
        )
    if dtype == "nvfp4":
        from flashinfer.moe_ep.batch_invariant.nvfp4_spec import (
            spec_fingerprint as nvfp4_spec_fingerprint,
        )
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.nvfp4_batch_invariant import (
            nvfp4_batch_invariant_arch,
        )

        return nvfp4_spec_fingerprint(
            hidden=hidden,
            intermediate=intermediate // 2,
            top_k=topk,
            num_experts=num_experts,
            arch=nvfp4_batch_invariant_arch(),
            implementation="legacy",
            apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
            gate_up_clamp=gate_up_clamp,
            input_norm_const=input_norm_const,
            has_fc1_alpha=True,
            has_fc2_alpha=True,
            has_fc1_norm_const=True,
            **variants,
        )
    from flashinfer.moe_ep.batch_invariant.spec import spec_fingerprint
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
        bf16_batch_invariant_arch,
    )

    return spec_fingerprint(
        hidden,
        intermediate,
        topk,
        num_experts,
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
        gate_up_clamp=gate_up_clamp,
        arch=bf16_batch_invariant_arch(),
        **variants,
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
    from flashinfer.moe_ep.core.tuning import KNOB_CACHE_VERSION

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
    if not isinstance(data, dict) or data.get("version") != KNOB_CACHE_VERSION:
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


def _nvfp4_bi_plan_valid(knobs, combine_dtype="bf16"):
    if (
        combine_dtype != "bf16"
        and knobs.get("token_back_mode", "epi_warps") != "reuse_dispatch_warps"
    ):
        return False
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.nvfp4_batch_invariant import (
        nvfp4_batch_invariant_knobs_valid,
    )

    return nvfp4_batch_invariant_knobs_valid(
        _knobs_from_json(knobs), implementation="legacy"
    )


def _a16_bi_plan_valid(knobs, dtype):
    from types import SimpleNamespace
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.a16_batch_invariant import (
        validate_a16_batch_invariant,
    )

    try:
        from .tuner import reject_explicit_knobs

        reject_explicit_knobs(knobs)
        validate_a16_batch_invariant(
            SimpleNamespace(batch_invariant=True, kind=dtype), _knobs_from_json(knobs)
        )
    except (TypeError, ValueError, KeyError):
        return False
    return True


@dataclass(frozen=True)
class _Bf16CacheProblem:
    hidden: int
    intermediate: int
    num_total_experts: int
    batch_invariant: bool = True


def _bf16_bi_plan_valid(knobs, hidden, intermediate, num_experts):
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
        validate_bf16_batch_invariant,
    )

    try:
        validate_bf16_batch_invariant(
            _Bf16CacheProblem(hidden, intermediate, num_experts),
            _knobs_from_json(knobs),
        )
    except (TypeError, ValueError):
        return False
    return True


def _mxfp8_bi_plan_valid(knobs, dtype):
    from types import SimpleNamespace
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.mxfp8_batch_invariant import (
        validate_mxfp8_batch_invariant,
    )

    try:
        validate_mxfp8_batch_invariant(
            SimpleNamespace(batch_invariant=True, kind=dtype),
            _knobs_from_json(knobs),
            implementation="legacy",
        )
    except (TypeError, ValueError):
        return False
    return True


@deprecated_aliases(
    apply_topk_in_fc1="apply_routing_weights_before_fc2",
    defer_topk_reduce="use_custom_finalize",
)
def _matching_entries(
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
    apply_routing_weights_before_fc2: bool = False,
    use_custom_finalize: bool = False,
    batch_invariant: bool = False,
    batch_invariant_version: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    input_norm_const: float = 1.0,
    activation: str = "swiglu",
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    per_token_scale: bool = False,
    handoff_norm: bool = False,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the cached knob dict for this session key, or ``None`` on miss.

    ``enable_in_kernel_fc2_reduce`` indicates whether the user has opted in to in_kernel_fc2_reduce.
    """
    policy_version = _batch_invariant_version(
        dtype, batch_invariant, batch_invariant_version
    )
    arithmetic_fingerprint = _arithmetic_fingerprint(
        dtype=dtype,
        batch_invariant=batch_invariant,
        hidden=hidden,
        intermediate=intermediate,
        topk=topk,
        num_experts=num_experts,
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
        gate_up_clamp=gate_up_clamp,
        input_norm_const=input_norm_const,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        per_token_scale=per_token_scale,
        handoff_norm=handoff_norm,
        output_mode="deferred" if use_custom_finalize else "finalized",
        combine_dtype=combine_dtype,
    )
    path = _cache_path()
    if path is None:
        return []
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
        and e.get("apply_routing_weights_before_fc2")
        == apply_routing_weights_before_fc2
        and e.get("use_custom_finalize", False) == use_custom_finalize
        and e.get("batch_invariant", False) == batch_invariant
        and (not batch_invariant or e.get("batch_invariant_version") == policy_version)
        and (
            not batch_invariant
            or e.get("batch_invariant_fingerprint") == arithmetic_fingerprint
        )
        and e.get("profile") in (None, "live_tokens")
        and isinstance(e.get("knobs"), dict)
        and (
            not batch_invariant
            or dtype not in ("bf16_nvfp4", "bf16_mxfp8_e4m3", "bf16_mxfp8_e5m2")
            or _a16_bi_plan_valid(e["knobs"], dtype)
        )
        and (
            not batch_invariant
            or dtype != "bf16"
            or _bf16_bi_plan_valid(e["knobs"], hidden, intermediate, num_experts)
        )
        and (
            not batch_invariant
            or dtype != "nvfp4"
            or _nvfp4_bi_plan_valid(e["knobs"], combine_dtype)
        )
        and (
            not batch_invariant
            or dtype not in ("mxfp8_e4m3", "mxfp8_e5m2")
            or _mxfp8_bi_plan_valid(e["knobs"], dtype)
        )
        and isinstance(e.get("max_tokens"), int)
        and (enable_in_kernel_fc2_reduce or not _entry_needs_ikr(e))
    ]
    return matches


@deprecated_aliases(
    apply_topk_in_fc1="apply_routing_weights_before_fc2",
    defer_topk_reduce="use_custom_finalize",
)
def live_token_buckets(**kwargs: Any) -> List[int]:
    from flashinfer.moe_ep.core.tuning import live_buckets

    return live_buckets(_matching_entries(**kwargs), kwargs["max_tokens"])


@deprecated_aliases(
    apply_topk_in_fc1="apply_routing_weights_before_fc2",
    defer_topk_reduce="use_custom_finalize",
)
def lookup_knobs(
    *,
    dtype: str,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    max_tokens: int,
    live_tokens: Optional[int] = None,
    combine_dtype: str = "bf16",
    enable_in_kernel_fc2_reduce: bool = False,
    apply_routing_weights_before_fc2: bool = False,
    use_custom_finalize: bool = False,
    batch_invariant: bool = False,
    batch_invariant_version: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    input_norm_const: float = 1.0,
    activation: str = "swiglu",
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    per_token_scale: bool = False,
    handoff_norm: bool = False,
    device: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    from flashinfer.moe_ep.core.tuning import select_token_entries

    matches = _matching_entries(
        dtype=dtype,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        max_tokens=max_tokens,
        combine_dtype=combine_dtype,
        enable_in_kernel_fc2_reduce=enable_in_kernel_fc2_reduce,
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
        use_custom_finalize=use_custom_finalize,
        batch_invariant=batch_invariant,
        batch_invariant_version=batch_invariant_version,
        gate_up_clamp=gate_up_clamp,
        input_norm_const=input_norm_const,
        device=device,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        per_token_scale=per_token_scale,
        handoff_norm=handoff_norm,
    )
    matches = select_token_entries(matches, max_tokens, live_tokens)
    if not matches:
        return None
    # Key of `(tokens, ~ikr)` causes the IKR config to be selected by min()
    # An IKR entry may not exist if
    #  1. the offline autotuner ran with IKR disabled
    #  2. the fastest config is IKR disabled
    #  3. `enable_in_kernel_fc2_reduce` is False and was filtered out above
    at_or_above = [e for e in matches if e["max_tokens"] >= max_tokens]
    if at_or_above:
        best = min(
            at_or_above, key=lambda e: (e["max_tokens"], not _entry_needs_ikr(e))
        )
    else:
        best = max(matches, key=lambda e: (e["max_tokens"], _entry_needs_ikr(e)))
    return _knobs_from_json(best["knobs"])


@deprecated_aliases(
    apply_topk_in_fc1="apply_routing_weights_before_fc2",
    defer_topk_reduce="use_custom_finalize",
)
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
    live_tokens: Optional[int] = None,
    combine_dtype: str = "bf16",
    apply_routing_weights_before_fc2: bool = False,
    use_custom_finalize: bool = False,
    batch_invariant: bool = False,
    batch_invariant_version: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    input_norm_const: float = 1.0,
    activation: str = "swiglu",
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    per_token_scale: bool = False,
    handoff_norm: bool = False,
    device: Optional[str] = None,
    p50_us: Optional[float] = None,
    source: str = "autotune",
) -> Optional[str]:
    """Upsert one tuned entry (exact key incl. ``max_tokens``); atomic write.

    Deterministic and ikr-tuned winners for the same geometry are separate
    entries (see :func:`_entry_needs_ikr`); FC1 and post-FC2 routing-weight
    placement and reduced/deferred output also have separate winners. BF16
    batch-invariant entries additionally isolate the numerical policy version. A
    ``--allow-nondeterministic`` sweep does not evict the deterministic winner
    a reproducible session needs, or vice versa.

    ``live_tokens`` distinguishes workloads at the same workspace capacity.

    Returns the cache path written, or ``None`` when the cache is disabled or
    the write failed (recording is best-effort — a read-only home directory
    must not break tuning).
    """
    if (
        batch_invariant
        and dtype == "nvfp4"
        and not _nvfp4_bi_plan_valid(knobs, combine_dtype)
    ):
        raise ValueError("cannot record an inadmissible batch-invariant NVFP4 plan")
    if (
        batch_invariant
        and dtype == "bf16"
        and not _bf16_bi_plan_valid(knobs, hidden, intermediate, num_experts)
    ):
        raise ValueError("cannot record an inadmissible batch-invariant BF16 plan")
    if (
        batch_invariant
        and dtype in ("mxfp8_e4m3", "mxfp8_e5m2")
        and not _mxfp8_bi_plan_valid(knobs, dtype)
    ):
        raise ValueError("cannot record an inadmissible batch-invariant MXFP8 plan")
    if (
        batch_invariant
        and dtype in ("bf16_nvfp4", "bf16_mxfp8_e4m3", "bf16_mxfp8_e5m2")
        and not _a16_bi_plan_valid(knobs, dtype)
    ):
        raise ValueError("cannot record an inadmissible batch-invariant A16 plan")
    policy_version = _batch_invariant_version(
        dtype, batch_invariant, batch_invariant_version
    )
    arithmetic_fingerprint = _arithmetic_fingerprint(
        dtype=dtype,
        batch_invariant=batch_invariant,
        hidden=hidden,
        intermediate=intermediate,
        topk=topk,
        num_experts=num_experts,
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
        gate_up_clamp=gate_up_clamp,
        input_norm_const=input_norm_const,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        per_token_scale=per_token_scale,
        handoff_norm=handoff_norm,
        output_mode="deferred" if use_custom_finalize else "finalized",
        combine_dtype=combine_dtype,
    )
    if live_tokens is None:
        live_tokens = max_tokens
    if type(live_tokens) is not int or not 0 <= live_tokens <= max_tokens:
        raise ValueError("live_tokens must be between zero and max_tokens")
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
        live_tokens=live_tokens,
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
        use_custom_finalize=use_custom_finalize,
        knobs=_knobs_to_json(knobs),
        p50_us=p50_us,
        source=source,
        tuned_at=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    if dtype in ("bf16", "nvfp4") or batch_invariant:
        entry.update(
            batch_invariant=batch_invariant,
            batch_invariant_version=policy_version,
            batch_invariant_fingerprint=arithmetic_fingerprint,
        )
    try:
        entries = _load_entries(path)
        entries = [
            e
            for e in entries
            if not (
                all(e.get(f) == entry[f] for f in _KEY_FIELDS)
                and e.get("max_tokens") == max_tokens
                and e.get("live_tokens", e["max_tokens"]) == live_tokens
                and e.get("apply_routing_weights_before_fc2")
                == apply_routing_weights_before_fc2
                and e.get("use_custom_finalize", False) == use_custom_finalize
                and e.get("batch_invariant", False) == batch_invariant
                and (
                    not batch_invariant
                    or e.get("batch_invariant_version") == policy_version
                )
                and (
                    not batch_invariant
                    or e.get("batch_invariant_fingerprint") == arithmetic_fingerprint
                )
                and e.get("profile") in (None, "live_tokens")
                and _entry_needs_ikr(e) == _entry_needs_ikr(entry)
            )
        ]
        entries.append(entry)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(path), prefix=".moe_ep_knob_cache."
        )
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


@deprecated_aliases(
    apply_topk_in_fc1="apply_routing_weights_before_fc2",
    defer_topk_reduce="use_custom_finalize",
)
def resolve_knobs(
    *,
    dtype: str,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    max_tokens: int,
    live_tokens: Optional[int] = None,
    combine_dtype: str = "bf16",
    enable_in_kernel_fc2_reduce: bool = False,
    apply_routing_weights_before_fc2: bool = False,
    use_custom_finalize: bool = False,
    batch_invariant: bool = False,
    batch_invariant_version: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    input_norm_const: float = 1.0,
    activation: str = "swiglu",
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    per_token_scale: bool = False,
    handoff_norm: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """Pure-lookup knob resolution: cache hit, else built-in heuristic.

    Returns ``(knobs, source)`` where source is ``"cache"`` or ``"heuristic"``.
    Cheap and deterministic — safe on the engine hot path (called once per
    buffer creation, never per forward).

    ``combine_dtype`` and ``enable_in_kernel_fc2_reduce`` describe the calling
    session, so resolution can never hand back a knob set the session's config
    would reject: the heuristic is built against them (see
    :func:`.tuner.default_knobs`) and the lookup only considers entries the
    session permits (see :func:`lookup_knobs`), so a cache populated with
    ``--allow-nondeterministic`` stays safe to share with sessions that need a
    reproducible combine.
    """
    cached = lookup_knobs(
        dtype=dtype,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        max_tokens=max_tokens,
        live_tokens=live_tokens,
        combine_dtype=combine_dtype,
        enable_in_kernel_fc2_reduce=enable_in_kernel_fc2_reduce,
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
        use_custom_finalize=use_custom_finalize,
        batch_invariant=batch_invariant,
        batch_invariant_version=batch_invariant_version,
        gate_up_clamp=gate_up_clamp,
        input_norm_const=input_norm_const,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        per_token_scale=per_token_scale,
        handoff_norm=handoff_norm,
    )
    if cached is not None:
        return cached, "cache"
    if batch_invariant and dtype == "nvfp4":
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.nvfp4_batch_invariant import (
            nvfp4_batch_invariant_default_knobs,
        )

        fallback = nvfp4_batch_invariant_default_knobs("legacy")
        if combine_dtype != "bf16":
            fallback["token_back_mode"] = "reuse_dispatch_warps"
        return fallback, "heuristic"
    if batch_invariant and dtype in ("mxfp8_e4m3", "mxfp8_e5m2"):
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.mxfp8_batch_invariant import (
            mxfp8_batch_invariant_default_knobs,
        )

        return mxfp8_batch_invariant_default_knobs(
            "legacy", quant_kind=dtype
        ), "heuristic"
    if batch_invariant and dtype in (
        "bf16_nvfp4",
        "bf16_mxfp8_e4m3",
        "bf16_mxfp8_e5m2",
    ):
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.a16_batch_invariant import (
            a16_batch_invariant_default_knobs,
        )

        return a16_batch_invariant_default_knobs(dtype), "heuristic"
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
