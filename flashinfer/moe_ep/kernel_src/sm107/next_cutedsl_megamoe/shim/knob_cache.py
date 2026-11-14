# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Persistent tuning cache for the SM107 block-scaled kernel.

The JSON format and file location are shared with SM100. An ``sm107:`` device
prefix keeps the architectures' entries separate. Implementation revision,
quantization, EP size, geometry, combine format, activation and beta parameters,
gate/up clamp, nondeterminism permission, and routing-weight policy must match.

Populate entries with ``python -m flashinfer.moe_ep.tune``. Set
``FLASHINFER_MOE_EP_KNOB_CACHE`` to a path, or ``0``/``off`` to disable it;
the default is ``~/.cache/flashinfer/moe_ep_knob_cache.json``.

``max_tokens`` is workspace capacity. Lookup prefers the exact bucket, then
the smallest larger bucket, then the largest smaller bucket.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Tuple

from flashinfer.moe_ep._compat import deprecated_aliases

# Bump when kernel or knob semantics invalidate previous tuning results.
BACKEND_REVISION = "sm107-block-scaled-1667b47a-runtime-options-v1"
_KEY_FIELDS = (
    "backend_revision",
    "kernel_variant",
    "allow_nondeterministic",
    "apply_routing_weights_before_fc2",
    "device",
    "dtype",
    "world_size",
    "hidden",
    "intermediate",
    "num_experts",
    "topk",
    "combine_dtype",
    "activation",
    "situ_beta",
    "situ_linear_beta",
    "gate_up_clamp",
    "batch_invariant",
    "batch_invariant_version",
    "batch_invariant_fingerprint",
)


def _bi_key(
    dtype,
    hidden,
    intermediate,
    topk,
    num_experts,
    variant,
    early,
    clamp,
    norm,
    enabled,
    version,
    *,
    activation="swiglu",
    swiglu_alpha=None,
    swiglu_beta=None,
    situ_beta=None,
    situ_linear_beta=None,
    per_token_scale=False,
    combine_dtype="bf16",
):
    if not enabled:
        return dict(
            batch_invariant=False,
            batch_invariant_version=None,
            batch_invariant_fingerprint=None,
        )
    from types import SimpleNamespace
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.activation_batch_invariant import (
        batch_invariant_variant_kwargs,
    )

    if per_token_scale:
        raise ValueError("per-token FC1 scaling is not implemented by SM107")
    variants = batch_invariant_variant_kwargs(
        SimpleNamespace(
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )
    )
    if combine_dtype != "bf16":
        variants["combine_dtype"] = combine_dtype
    if dtype in ("mxfp4", "mxfp4_mxfp8"):
        if isinstance(norm, bool) or norm != 1.0:
            raise ValueError("batch_invariant MXFP4 requires input_norm_const=1.0")
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.mxfp4_batch_invariant import (
            MXFP4_BATCH_INVARIANT_VERSION,
        )
        from flashinfer.moe_ep.batch_invariant.mxfp4_spec import (
            spec_fingerprint as mxfp4_fingerprint,
        )

        return dict(
            batch_invariant=True,
            batch_invariant_version=MXFP4_BATCH_INVARIANT_VERSION
            if version is None
            else version,
            batch_invariant_fingerprint=mxfp4_fingerprint(
                hidden=hidden,
                intermediate=intermediate,
                top_k=topk,
                num_experts=num_experts,
                arch="sm_107",
                quant_kind=dtype,
                implementation="genphase" if variant == "genphase" else "standard",
                apply_routing_weights_before_fc2=early,
                gate_up_clamp=clamp,
                **variants,
            ),
        )
    if dtype in ("mxfp8_e4m3", "mxfp8_e5m2"):
        if isinstance(norm, bool) or norm != 1.0:
            raise ValueError("batch_invariant MXFP8 requires input_norm_const=1.0")
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.mxfp8_batch_invariant import (
            MXFP8_BATCH_INVARIANT_VERSION,
        )
        from flashinfer.moe_ep.batch_invariant.mxfp8_spec import (
            spec_fingerprint as mxfp8_fingerprint,
        )

        return dict(
            batch_invariant=True,
            batch_invariant_version=MXFP8_BATCH_INVARIANT_VERSION
            if version is None
            else version,
            batch_invariant_fingerprint=mxfp8_fingerprint(
                hidden=hidden,
                intermediate=intermediate,
                top_k=topk,
                num_experts=num_experts,
                arch="sm_107",
                quant_kind=dtype,
                implementation="genphase" if variant == "genphase" else "standard",
                apply_routing_weights_before_fc2=early,
                gate_up_clamp=clamp,
                **variants,
            ),
        )
    if dtype != "nvfp4":
        raise ValueError("SM107 batch_invariant cache requires NVFP4 or MXFP4")
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.nvfp4_batch_invariant import (
        NVFP4_BATCH_INVARIANT_VERSION,
    )
    from flashinfer.moe_ep.batch_invariant.nvfp4_spec import spec_fingerprint

    return dict(
        batch_invariant=True,
        batch_invariant_version=NVFP4_BATCH_INVARIANT_VERSION
        if version is None
        else version,
        batch_invariant_fingerprint=spec_fingerprint(
            hidden=hidden,
            intermediate=intermediate,
            top_k=topk,
            num_experts=num_experts,
            arch="sm_107",
            implementation="genphase" if variant == "genphase" else "standard",
            apply_routing_weights_before_fc2=early,
            gate_up_clamp=clamp,
            input_norm_const=norm,
            has_fc1_alpha=True,
            has_fc2_alpha=True,
            has_fc1_norm_const=True,
            **variants,
        ),
    )


def _bi_plan_valid(
    knobs,
    variant,
    dtype="nvfp4",
    *,
    hidden=None,
    intermediate=None,
    combine_dtype="bf16",
):
    from types import SimpleNamespace
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.block_scaled_batch_invariant import (
        validate_block_scaled_batch_invariant,
    )

    values = dict(batch_invariant=True, quant_kind=dtype, combine_dtype=combine_dtype)
    if hidden is not None:
        values["hidden"] = hidden
    if intermediate is not None:
        values["intermediate"] = intermediate
    try:
        validate_block_scaled_batch_invariant(
            SimpleNamespace(**values),
            _knobs_from_json(knobs),
            implementation="genphase" if variant == "genphase" else "standard",
            arch="sm_107",
        )
    except (TypeError, ValueError):
        return False
    return True


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


def _device_key(device: Optional[str]) -> str:
    # Older backend readers ignore new JSON fields. Namespace an existing
    # key too, so generic pre-production GPU names cannot collide with SM100.
    return "sm107:" + (device if device is not None else _current_device_name())


def _load_entries(path: str) -> List[Dict[str, Any]]:
    from flashinfer.moe_ep.core.tuning import KNOB_CACHE_VERSION

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
    return [e for e in entries if isinstance(e, dict)]


def _knobs_to_json(knobs: Dict[str, Any]) -> Dict[str, Any]:
    # Tuples (tile / cluster / epi flag batches / schedule_policy) -> lists.
    return {k: list(v) if isinstance(v, tuple) else v for k, v in knobs.items()}


def _knobs_from_json(knobs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in knobs.items()}


def _matching_entries(
    *,
    entries: Optional[List[Dict[str, Any]]] = None,
    dtype: str,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    max_tokens: int,
    combine_dtype: str = "bf16",
    kernel_variant: str = "inference",
    device: Optional[str] = None,
    allow_nondeterministic: bool = False,
    apply_routing_weights_before_fc2: bool = True,
    activation: str = "swiglu",
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    gate_up_clamp: Optional[float] = None,
    batch_invariant: bool = False,
    batch_invariant_version: Optional[int] = None,
    input_norm_const: float = 1.0,
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    per_token_scale: bool = False,
) -> List[Dict[str, Any]]:
    """Return records with a compatible numerical and execution contract."""
    if entries is None:
        path = _cache_path()
        entries = [] if path is None else _load_entries(path)
    key = dict(
        **_bi_key(
            dtype,
            hidden,
            intermediate,
            topk,
            num_experts,
            kernel_variant,
            apply_routing_weights_before_fc2,
            gate_up_clamp,
            input_norm_const,
            batch_invariant,
            batch_invariant_version,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            per_token_scale=per_token_scale,
            combine_dtype=combine_dtype,
        ),
        backend_revision=BACKEND_REVISION,
        kernel_variant=kernel_variant,
        allow_nondeterministic=allow_nondeterministic,
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
        activation=activation,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        gate_up_clamp=gate_up_clamp,
        device=_device_key(device),
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
        for e in entries
        if all(
            e.get(f, False if f == "batch_invariant" else None) == key[f]
            for f in _KEY_FIELDS
        )
        and isinstance(e.get("knobs"), dict)
        and (
            not batch_invariant
            or _bi_plan_valid(
                e["knobs"],
                kernel_variant,
                dtype,
                hidden=hidden,
                intermediate=intermediate,
                combine_dtype=combine_dtype,
            )
        )
        and isinstance(e.get("max_tokens"), int)
        and (
            allow_nondeterministic or not e["knobs"].get("reduce_topk_in_kernel", False)
        )
    ]
    return matches


@deprecated_aliases(apply_topk_at_fc1="apply_routing_weights_before_fc2")
def live_token_buckets(**kwargs: Any) -> List[int]:
    from flashinfer.moe_ep.core.tuning import live_buckets

    return live_buckets(_matching_entries(**kwargs), kwargs["max_tokens"])


def lookup_record(
    *,
    entries: Optional[List[Dict[str, Any]]] = None,
    dtype: str,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    max_tokens: int,
    live_tokens: Optional[int] = None,
    combine_dtype: str = "bf16",
    kernel_variant: str = "inference",
    device: Optional[str] = None,
    allow_nondeterministic: bool = False,
    apply_routing_weights_before_fc2: bool = True,
    activation: str = "swiglu",
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    gate_up_clamp: Optional[float] = None,
    batch_invariant: bool = False,
    batch_invariant_version: Optional[int] = None,
    input_norm_const: float = 1.0,
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    per_token_scale: bool = False,
) -> Optional[Dict[str, Any]]:
    from flashinfer.moe_ep.core.tuning import select_token_entries

    matches = _matching_entries(
        entries=entries,
        dtype=dtype,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        max_tokens=max_tokens,
        combine_dtype=combine_dtype,
        kernel_variant=kernel_variant,
        device=device,
        allow_nondeterministic=allow_nondeterministic,
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
        activation=activation,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        gate_up_clamp=gate_up_clamp,
        batch_invariant=batch_invariant,
        batch_invariant_version=batch_invariant_version,
        input_norm_const=input_norm_const,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        per_token_scale=per_token_scale,
    )
    matches = select_token_entries(matches, max_tokens, live_tokens)
    if not matches:
        return None
    at_or_above = [e for e in matches if e["max_tokens"] >= max_tokens]
    if at_or_above:
        best = min(at_or_above, key=lambda e: e["max_tokens"])
    else:
        best = max(matches, key=lambda e: e["max_tokens"])
    return best


@deprecated_aliases(apply_topk_at_fc1="apply_routing_weights_before_fc2")
def lookup_knobs(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Resolve knobs using the same record lookup as the shared autotuner."""
    record = lookup_record(**kwargs)
    return None if record is None else _knobs_from_json(record["knobs"])


def make_record(
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
    kernel_variant: str = "inference",
    device: Optional[str] = None,
    p50_us: Optional[float] = None,
    source: str = "autotune",
    allow_nondeterministic: bool = False,
    apply_routing_weights_before_fc2: bool = True,
    activation: str = "swiglu",
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    gate_up_clamp: Optional[float] = None,
    batch_invariant: bool = False,
    batch_invariant_version: Optional[int] = None,
    input_norm_const: float = 1.0,
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    per_token_scale: bool = False,
) -> Dict[str, Any]:
    """Build a tuning record shared by FlashInfer and the MegaMoE file."""
    if batch_invariant and not _bi_plan_valid(
        knobs,
        kernel_variant,
        dtype,
        hidden=hidden,
        intermediate=intermediate,
        combine_dtype=combine_dtype,
    ):
        raise ValueError("cannot record an inadmissible batch-invariant NVFP4 plan")
    if knobs.get("reduce_topk_in_kernel", False) and not allow_nondeterministic:
        raise ValueError(
            "recording in-kernel reduction requires allow_nondeterministic=True"
        )
    if live_tokens is None:
        live_tokens = max_tokens
    if type(live_tokens) is not int or not 0 <= live_tokens <= max_tokens:
        raise ValueError("live_tokens must be between zero and max_tokens")
    entry = dict(
        **_bi_key(
            dtype,
            hidden,
            intermediate,
            topk,
            num_experts,
            kernel_variant,
            apply_routing_weights_before_fc2,
            gate_up_clamp,
            input_norm_const,
            batch_invariant,
            batch_invariant_version,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            per_token_scale=per_token_scale,
            combine_dtype=combine_dtype,
        ),
        backend_revision=BACKEND_REVISION,
        kernel_variant=kernel_variant,
        allow_nondeterministic=allow_nondeterministic,
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
        activation=activation,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        gate_up_clamp=gate_up_clamp,
        device=_device_key(device),
        dtype=dtype,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        combine_dtype=combine_dtype,
        max_tokens=max_tokens,
        live_tokens=live_tokens,
        knobs=_knobs_to_json(knobs),
        p50_us=p50_us,
        source=source,
        tuned_at=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    return entry


def write_record(entry: Dict[str, Any]) -> Optional[str]:
    """Write a selected record to the configured MegaMoE cache file."""
    path = _cache_path()
    if path is None:
        return None
    max_tokens = entry["max_tokens"]
    live_tokens = entry.get("live_tokens", max_tokens)
    try:
        entries = _load_entries(path)
        entries = [
            e
            for e in entries
            if not (
                all(
                    e.get(f, False if f == "batch_invariant" else None) == entry[f]
                    for f in _KEY_FIELDS
                )
                and e.get("max_tokens") == max_tokens
                and e.get("live_tokens", e["max_tokens"]) == live_tokens
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


@deprecated_aliases(apply_topk_at_fc1="apply_routing_weights_before_fc2")
def record_knobs(knobs: Dict[str, Any], **kwargs: Any) -> Optional[str]:
    """Record a winner in the MegaMoE file; return its path if written."""
    return write_record(make_record(knobs, **kwargs))


def default_knobs(
    max_tokens: int,
    *,
    quant_kind: str = "nvfp4",
    kernel_variant: str = "inference",
    combine_dtype: str = "bf16",
) -> Dict[str, Any]:
    """Built-in SM107 configurations for cache misses.

    Two token profiles (see TUNING.md): <2048 tokens/rank keeps the 128-wide
    N tile, >=2048 the 256-wide.  Both use mixed CGA (4x1 preferred, 2x1
    fallback), phase-interleave with the kernel-resolved minimum hint, atomic
    work IDs, FC2 bulk TMA stage 2, epi-warp token back, and separate top-k
    reduction.  Tile K is the kind's 2x-mode depth (2 x instruction K).

    GenPhase uses uniform (4, 1) clusters and its required FC2 bulk path.
    NVFP4 combine disables the peer-facing bulk store, which cannot send FP4.
    Returns a fresh dict each call.
    """
    if kernel_variant == "genphase":
        if combine_dtype != "bf16":
            raise ValueError("GenPhase requires BF16 combine.")
        return {
            "cluster_shape_mn": (4, 1),
            "fc2_use_bulk": True,
            "reduce_topk_in_kernel": False,
        }
    if kernel_variant != "inference":
        raise ValueError(f"unsupported kernel_variant {kernel_variant!r}")
    tile_k = 256 if quant_kind in ("nvfp4", "mxfp4") else 128
    tile_n = 128 if max_tokens < 2048 else 256
    knobs = {
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

    if combine_dtype == "nvfp4":
        # The peer-facing bulk store cannot send packed FP4 data.
        knobs["fc2_use_bulk"] = False
        knobs["fc2_tma_stages"] = None
    return knobs


@deprecated_aliases(apply_topk_at_fc1="apply_routing_weights_before_fc2")
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
    kernel_variant: str = "inference",
    allow_nondeterministic: bool = False,
    apply_routing_weights_before_fc2: bool = True,
    activation: str = "swiglu",
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    gate_up_clamp: Optional[float] = None,
    batch_invariant: bool = False,
    batch_invariant_version: Optional[int] = None,
    input_norm_const: float = 1.0,
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    per_token_scale: bool = False,
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
        live_tokens=live_tokens,
        combine_dtype=combine_dtype,
        kernel_variant=kernel_variant,
        allow_nondeterministic=allow_nondeterministic,
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
        activation=activation,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        gate_up_clamp=gate_up_clamp,
        batch_invariant=batch_invariant,
        batch_invariant_version=batch_invariant_version,
        input_norm_const=input_norm_const,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        per_token_scale=per_token_scale,
    )
    if cached is not None:
        return cached, "cache"
    if batch_invariant:
        from types import SimpleNamespace
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.block_scaled_batch_invariant import (
            block_scaled_batch_invariant_default_knobs,
        )

        return block_scaled_batch_invariant_default_knobs(
            SimpleNamespace(quant_kind=dtype),
            "genphase" if kernel_variant == "genphase" else "standard",
            arch="sm_107",
        ), "heuristic"
    return default_knobs(
        max_tokens,
        quant_kind=dtype,
        kernel_variant=kernel_variant,
        combine_dtype=combine_dtype,
    ), "heuristic"


__all__ = [
    "default_knobs",
    "knob_cache_path",
    "lookup_knobs",
    "record_knobs",
    "resolve_knobs",
]
