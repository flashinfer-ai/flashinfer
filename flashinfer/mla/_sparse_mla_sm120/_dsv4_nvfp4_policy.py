# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calibrated decode/prefill planner for SM120 NVFP4 sparse MLA.

The shared decode-token cutoff is policy, not a kernel capability.
Inside that policy range, a separately keyed NVFP4
calibration compares split-K decode (including merge) with streaming prefill
and records the measured variant plus the best decode CPB for each probe
bucket, persisted as a per-configuration profile (the same record shape the
DSV4.1 planner uses).  Per-bucket records preserve GPU wave-boundary
non-monotonicity that a single threshold cannot express.  The dispatch
policy and persistence primitives are shared with the existing FP8 planner;
FP8 measurements are never reused for NVFP4.
"""

from __future__ import annotations

import enum
import functools
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypedDict

import torch

from ...autotuner import AutoTuner
from . import _calibration as _cpb
from ._calibration import CalibrationError
from ._policy import _select_calibrated_variant, _DECODE_MAX_TOKENS
from ._execution import dsv4_nvfp4_format_info

logger = logging.getLogger(__name__)

# Publish family for NVFP4 profile records; the JSON request key carries the
# exact cache configuration.
_FAMILY = "dsv4_nvfp4"
_AUTOTUNE_OP = "sparse_mla_sm120_nvfp4"


class _StaticFormat(TypedDict):
    query_dim: int
    value_dim: int
    bytes_per_token: int
    chunk_width: int
    page_size: int
    heads: tuple[int, ...]
    topks: tuple[int, ...]
    extra_page_sizes: tuple[int, ...]


# Coarse static capability envelope, pure Python (the ordinary families keep
# theirs in ``_policy``). Init-time probes such as
# ``supported_sparse_mla_sm120_configs(kv_cache_format="nvfp4")`` must work
# without a GPU or JIT compilation. The C++ dispatcher stays authoritative for
# member shapes; this is a pre-filter, not a mirror of the compiled instances,
# so no static-vs-compiled parity test pins it.
_STATIC_FORMAT: _StaticFormat = {
    "query_dim": 512,
    "value_dim": 512,
    "bytes_per_token": 384,
    "chunk_width": 64,
    "page_size": 64,
    "heads": (16, 32, 64, 128),
    "topks": (128, 512),
    "extra_page_sizes": (2, 64),
}


@functools.cache
def _supports(
    heads: int, topk: int, page: int, extra_topk: int, extra_page: int
) -> bool:
    from ._execution import query

    return query(
        "supports_attention",
        heads,
        topk,
        page,
        extra_topk,
        extra_page,
        is_dsv4_nvfp4=True,
    )


# One profile bucket per canonical grid point; the store validator requires
# the full grid. Non-probe decode batches use an exact refined entry when
# available, otherwise the next larger canonical bucket's measured CPB.
_CROSSOVER_PROBED_T = _cpb._PROFILE_T
_CROSSOVER_MARGIN = _cpb._CROSSOVER_MARGIN

# The calibration timing protocol queues calls and rotates index sets, keeping
# launch latency overlap and cache residency close to steady-state serving.
# OOM halves the aggregate toward _POOL_BYTES_MIN (the same floor semantics as
# the ordinary pools); each dual-cache segment additionally floors at twice
# the device L2 plus one page so the capacity guard cannot misfire on
# large-L2 parts (e.g. 96 MiB on GB202).
_POOL_BYTES_TARGET = 2 << 30
_POOL_BYTES_MIN = 512 << 20
_MIN_POOL_PER_SEGMENT = 128 << 20


class NVFP4KernelVariant(enum.Enum):
    """NVFP4 sparse-MLA implementation selected for one call."""

    DECODE_SPLITK = "decode_splitk"
    PREFILL_STREAMING = "prefill_streaming"


@dataclass(frozen=True)
class NVFP4PlannedCall:
    """Planner output; CPB is meaningful only for split-K decode."""

    variant: NVFP4KernelVariant
    cpb: int  # 0 asks the C++ decode launcher for its safe heuristic
    compute_precision: str = "nvfp4"


@dataclass(frozen=True)
class NVFP4CalibrationReport:
    """Measured result for one exact NVFP4 sparse-MLA cache configuration."""

    family: str
    num_heads: int
    topk: int
    extra_topk: int
    extra_page_size: int
    decode_max_tokens: Optional[int]
    phase_by_token_bucket: dict[int, str]
    cpb_by_token_bucket: dict[int, int]
    decode_latency_us: dict[int, float]
    prefill_latency_us: dict[int, float]
    persisted: bool = False


_plan_memo: dict[tuple, NVFP4KernelVariant] = {}
_plan_epoch = -1
_calibration_lock = threading.RLock()
_calibrating: set[tuple[str | None, str, str]] = set()


def _request_key(
    *,
    num_heads: int,
    topk: int,
    primary_page_size: int,
    extra_topk: int,
    extra_page_size: int,
    has_topk_length: bool,
    has_extra_topk_length: bool,
    has_attn_sink: bool,
) -> str:
    """Profile key for one exact NVFP4 cache configuration."""
    return json.dumps(
        {
            "family": _FAMILY,
            "strategy": _cpb._TIMING_PROTOCOL,
            "heads": num_heads,
            "topk": topk,
            "primary_page_size": primary_page_size,
            "extra_topk": extra_topk,
            "extra_page_size": extra_page_size,
            "has_topk_length": bool(has_topk_length),
            "has_extra_topk_length": bool(has_extra_topk_length),
            "has_attn_sink": bool(has_attn_sink),
        },
        sort_keys=True,
    )


def _eligible(
    num_tokens: int,
    num_heads: int,
    topk: int,
    primary_page_size: int,
    extra_topk: int,
    extra_page_size: int,
) -> tuple[bool, bool]:
    common = num_tokens >= 1 and _supports(
        num_heads, topk, primary_page_size, extra_topk, extra_page_size
    )
    return common and num_tokens <= _DECODE_MAX_TOKENS, common


def _canonical_profile_layout(
    primary_page_size: int,
    page_stride_bytes: Optional[int],
    extra_topk: int,
    extra_page_size: int,
    extra_page_stride_bytes: Optional[int],
) -> bool:
    """True iff the cache layout matches the canonical calibration sample.

    Profiles are measured on packed pages (pitch == page_size * 384, which is
    inherently 16B-aligned; indices and LSE are dense by resolve contract).
    Other legal layouts — a padded page pitch — skip profile lookup, lazy
    calibration and exact-T refinement, mirroring the ordinary policy's
    ``canonical_profile_layout`` gate. ``None`` strides (direct planner
    callers without tensor metadata) are treated as canonical; the
    wrapper/resolve boundary passes the real strides from inspected metadata.
    """
    bytes_per_token = _STATIC_FORMAT["bytes_per_token"]
    if (
        page_stride_bytes is not None
        and page_stride_bytes != primary_page_size * bytes_per_token
    ):
        return False
    if extra_topk and extra_page_stride_bytes is not None:
        return extra_page_stride_bytes == extra_page_size * bytes_per_token
    return True


def _autotune_skipped() -> bool:
    tuner = AutoTuner.get()
    skip_stack = tuner._get_skip_ops_stack()
    return bool(skip_stack) and _AUTOTUNE_OP in skip_stack[-1]


def _maybe_calibrate(
    *,
    device: torch.device,
    key: str,
    num_heads: int,
    topk: int,
    primary_page_size: int,
    extra_topk: int,
    extra_page_size: int,
    has_topk_length: bool,
    has_extra_topk_length: bool,
    has_attn_sink: bool,
) -> None:
    tuner = AutoTuner.get()
    if (
        not tuner.is_tuning_mode
        or _autotune_skipped()
        or _cpb._target_capturing(device)
        or _cpb.is_calibration_failed(device, key)
    ):
        return

    with _cpb._store_lock:
        _cpb.refresh_store()
        scope = _cpb._active_scope
    guard_key = (scope, _cpb._device_key(device), key)
    with _calibration_lock:
        if _cpb.get_profile(key, device) is not None or _cpb.is_calibration_failed(
            device, key
        ):
            return
        if guard_key in _calibrating:
            return
        _calibrating.add(guard_key)
        try:
            calibrate_nvfp4_sparse_mla_sm120(
                device,
                num_heads=num_heads,
                topk=topk,
                primary_page_size=primary_page_size,
                extra_topk=extra_topk,
                extra_page_size=extra_page_size,
                has_topk_length=has_topk_length,
                has_extra_topk_length=has_extra_topk_length,
                has_attn_sink=has_attn_sink,
                force=True,
            )
        except (CalibrationError, torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(
                "SM120 NVFP4 sparse-MLA calibration failed for H=%d, topk=%d, "
                "extra_topk=%d, extra_page_size=%d (%s); using decode-first "
                "routing and the C++ CPB heuristic.",
                num_heads,
                topk,
                extra_topk,
                extra_page_size,
                e,
            )
            _cpb.mark_calibration_failed(device, key)
        finally:
            _calibrating.discard(guard_key)


def plan_nvfp4_sparse_mla_sm120(
    num_tokens: int,
    num_heads: int,
    topk: int,
    primary_page_size: int,
    device: torch.device,
    *,
    extra_topk: int = 0,
    extra_page_size: int = 0,
    has_topk_length: bool = False,
    has_extra_topk_length: bool = False,
    has_attn_sink: bool = False,
    page_stride_bytes: Optional[int] = None,
    extra_page_stride_bytes: Optional[int] = None,
) -> Optional[NVFP4PlannedCall]:
    """Select independently calibrated NVFP4 prefill/decode execution.

    The measured profiles apply only to the canonical packed page layout
    (``page stride == page_size * 384``, inherently 16B-aligned); the
    wrapper/resolve boundary passes the real page strides, and other legal
    layouts keep the decode-first fallback without profile lookup, lazy
    calibration, or refinement. ``None`` strides mean the caller has no
    layout metadata and are treated as canonical.
    """
    device = torch.device(device)
    if has_extra_topk_length and extra_topk == 0:
        raise ValueError("has_extra_topk_length requires extra_topk > 0")
    decode_ok, prefill_ok = _eligible(
        num_tokens,
        num_heads,
        topk,
        primary_page_size,
        extra_topk,
        extra_page_size,
    )
    if not decode_ok and not prefill_ok:
        return None

    canonical = _canonical_profile_layout(
        primary_page_size,
        page_stride_bytes,
        extra_topk,
        extra_page_size,
        extra_page_stride_bytes,
    )
    key = _request_key(
        num_heads=num_heads,
        topk=topk,
        primary_page_size=primary_page_size,
        extra_topk=extra_topk,
        extra_page_size=extra_page_size,
        has_topk_length=has_topk_length,
        has_extra_topk_length=has_extra_topk_length,
        has_attn_sink=has_attn_sink,
    )
    with _cpb._store_lock:
        profile = _cpb.get_profile(key, device) if canonical else None
        scope_epoch = _cpb._constants_version
    if canonical and decode_ok and profile is None:
        _maybe_calibrate(
            device=device,
            key=key,
            num_heads=num_heads,
            topk=topk,
            primary_page_size=primary_page_size,
            extra_topk=extra_topk,
            extra_page_size=extra_page_size,
            has_topk_length=has_topk_length,
            has_extra_topk_length=has_extra_topk_length,
            has_attn_sink=has_attn_sink,
        )
        with _cpb._store_lock:
            profile = _cpb.get_profile(key, device)
            scope_epoch = _cpb._constants_version

    if (
        canonical
        and profile is not None
        and decode_ok
        and str(num_tokens) not in profile["buckets"]
        and AutoTuner.get().is_tuning_mode
        and not _autotune_skipped()
        and not _cpb._target_capturing(device)
        and not _cpb.is_refine_failed(device, key, num_tokens)
    ):
        # Tuning mode: measure this exact token count once so the selection
        # stops interpolating from the canonical grid. A failed measurement is
        # suppressed in-process instead of being retimed on every call.
        try:
            refined = refine_nvfp4(
                device,
                tokens=num_tokens,
                num_heads=num_heads,
                topk=topk,
                primary_page_size=primary_page_size,
                extra_topk=extra_topk,
                extra_page_size=extra_page_size,
                has_topk_length=has_topk_length,
                has_extra_topk_length=has_extra_topk_length,
                has_attn_sink=has_attn_sink,
            )
        except (CalibrationError, RuntimeError) as error:
            logger.debug("NVFP4 refine skipped at tokens=%d: %s", num_tokens, error)
            _cpb.mark_refine_failed(device, key, num_tokens)
            refined = None
        if refined is not None:
            with _cpb._store_lock:
                profile = _cpb.get_profile(key, device)
                scope_epoch = _cpb._constants_version

    # Each bucket stores its own selection; a missing profile keeps the
    # decode-first fallback.
    bucket = (
        None
        if profile is None or num_tokens > _DECODE_MAX_TOKENS
        else _cpb._profile_bucket(profile, num_tokens)
    )
    memo_key = (
        scope_epoch,
        key,
        num_tokens,
        _cpb._device_key(device),
    )
    global _plan_epoch
    if _plan_epoch != _cpb._constants_version:
        _plan_memo.clear()
        _plan_epoch = _cpb._constants_version
    variant = _plan_memo.get(memo_key)
    if variant is None:
        variant = _select_calibrated_variant(
            decode_eligible=decode_ok,
            decode_variant=NVFP4KernelVariant.DECODE_SPLITK,
            prefill_variant=(
                NVFP4KernelVariant.PREFILL_STREAMING if prefill_ok else None
            ),
            decode_preferred=(
                None
                if bucket is None
                else bucket["variant"] == NVFP4KernelVariant.DECODE_SPLITK.value
            ),
        )
        if variant is None:
            return None
        _plan_memo[memo_key] = variant

    if variant is NVFP4KernelVariant.PREFILL_STREAMING:
        return NVFP4PlannedCall(variant, 0)
    return NVFP4PlannedCall(variant, 0 if bucket is None else bucket["cpb"])


def _allocate_cache_pool(
    page_size: int, pool_bytes: int, device: torch.device
) -> tuple[torch.Tensor, int]:
    from ._dsv4_nvfp4 import nvfp4_quantize_pack_sparse_mla_cache

    facts = dsv4_nvfp4_format_info()
    page_bytes = page_size * facts["bytes_per_token"]
    num_pages = max(1, pool_bytes // page_bytes)
    _cpb._check_pool_capacity(device, num_pages * page_bytes)
    cache = torch.empty(
        (num_pages, page_size, facts["bytes_per_token"]),
        dtype=torch.uint8,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(0)
    chunk = max(1, _cpb._SAMPLE_CHUNK_BYTES // (page_size * facts["query_dim"] * 8))
    for start in range(0, num_pages, chunk):
        pages = min(chunk, num_pages - start)
        latent = (
            torch.randn(
                (pages, page_size, facts["query_dim"]),
                device=device,
                generator=generator,
            )
            .mul_(0.1)
            .clamp_(-1, 1)
            .to(torch.bfloat16)
        )
        packed = nvfp4_quantize_pack_sparse_mla_cache(latent)
        cache[start : start + pages].copy_(
            packed.view(pages, page_size, facts["bytes_per_token"])
        )
        del latent, packed
    return cache, num_pages * page_size


def _segment_floor_bytes(device: torch.device, page_bytes: int) -> int:
    """Per-segment pool floor for dual-cache calibration.

    Cold-L2 sampling needs each cache segment to exceed twice the device L2
    capacity even after page-count rounding (the guard
    ``_check_pool_capacity`` enforces on the allocated bytes); one page of
    headroom absorbs the rounding-down. The static 128 MiB floor covers
    smaller-L2 parts.
    """
    return max(_MIN_POOL_PER_SEGMENT, 2 * _cpb._device_l2(device) + page_bytes)


def _allocate_calibration_pools(
    device: torch.device,
    primary_page_size: int,
    topk: int,
    extra_topk: int,
    extra_page_size: int,
) -> tuple[torch.Tensor, int, Optional[torch.Tensor], int]:
    facts = dsv4_nvfp4_format_info()
    if extra_topk:
        primary_floor = _segment_floor_bytes(
            device, primary_page_size * facts["bytes_per_token"]
        )
        extra_floor = _segment_floor_bytes(
            device, extra_page_size * facts["bytes_per_token"]
        )
        min_total = max(_POOL_BYTES_MIN, primary_floor + extra_floor)
    else:
        primary_floor = extra_floor = 0
        min_total = _POOL_BYTES_MIN
    total_bytes = _POOL_BYTES_TARGET
    while True:
        total_topk = topk + extra_topk
        if extra_topk:
            primary_bytes = max(primary_floor, total_bytes * topk // total_topk)
            extra_bytes = max(extra_floor, total_bytes * extra_topk // total_topk)
        else:
            primary_bytes, extra_bytes = total_bytes, 0
        primary_cache = None
        extra_cache = None
        try:
            primary_cache, primary_slots = _allocate_cache_pool(
                primary_page_size, primary_bytes, device
            )
            if extra_topk:
                extra_cache, extra_slots = _allocate_cache_pool(
                    extra_page_size, extra_bytes, device
                )
            else:
                extra_slots = 0
            return primary_cache, primary_slots, extra_cache, extra_slots
        except RuntimeError as error:
            if not _cpb._is_cuda_oom(error):
                raise
            del primary_cache, extra_cache
            if total_bytes <= min_total:
                if extra_topk:
                    l2_mib = _cpb._device_l2(device) >> 20
                    raise CalibrationError(
                        "cannot allocate NVFP4 sparse-MLA calibration KV pools: "
                        f"out of memory above the {min_total >> 20} MiB aggregate "
                        "floor (each dual-cache segment must exceed twice the "
                        f"device L2 capacity, {l2_mib} MiB here, after page "
                        "rounding)"
                    ) from None
                raise CalibrationError(
                    "cannot allocate an NVFP4 sparse-MLA calibration KV pool: "
                    f"out of memory above the {min_total >> 20} MiB floor"
                ) from None
            total_bytes //= 2
        torch.cuda.empty_cache()


def _make_index_sets(
    *,
    num_tokens: int,
    topk: int,
    primary_slots: int,
    extra_topk: int,
    extra_slots: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    count = _cpb.calibration_batch_count(
        num_tokens,
        topk + extra_topk,
        dsv4_nvfp4_format_info()["bytes_per_token"],
        device,
    )
    generator = torch.Generator(device=device).manual_seed(0)
    result = []
    for _ in range(count):
        primary = torch.randint(
            0,
            primary_slots,
            (num_tokens, topk),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )
        extra = (
            torch.randint(
                0,
                extra_slots,
                (num_tokens, extra_topk),
                dtype=torch.int32,
                device=device,
                generator=generator,
            )
            if extra_topk
            else None
        )
        result.append((primary, extra))
    return result


def _time_indexed_calls(
    call: Callable[[torch.Tensor, Optional[torch.Tensor]], None],
    index_sets: list[tuple[torch.Tensor, Optional[torch.Tensor]]],
    device: torch.device,
) -> float:
    return _cpb.time_calibration_calls(call, index_sets, device) * 1e6


def _make_calibration_calls(
    *,
    module,
    device: torch.device,
    num_tokens: int,
    num_heads: int,
    topk: int,
    primary_cache: torch.Tensor,
    extra_topk: int,
    extra_cache: Optional[torch.Tensor],
    has_topk_length: bool,
    has_extra_topk_length: bool,
    has_attn_sink: bool,
) -> tuple[
    Callable[[int], Callable[[torch.Tensor, Optional[torch.Tensor]], None]],
    Callable[[torch.Tensor, Optional[torch.Tensor]], None],
]:
    num_splits = (
        topk + dsv4_nvfp4_format_info()["chunk_width"] - 1
    ) // dsv4_nvfp4_format_info()["chunk_width"]
    num_splits += (
        extra_topk + dsv4_nvfp4_format_info()["chunk_width"] - 1
    ) // dsv4_nvfp4_format_info()["chunk_width"]
    q = (
        (
            torch.randn(
                num_tokens,
                num_heads,
                dsv4_nvfp4_format_info()["query_dim"],
                dtype=torch.float32,
                device=device,
                generator=torch.Generator(device=device).manual_seed(1),
            )
            / 10.0
        )
        .clamp(-1, 1)
        .to(torch.bfloat16)
    )
    mid_out = torch.empty(
        (num_tokens, num_heads, num_splits, dsv4_nvfp4_format_info()["value_dim"]),
        dtype=torch.bfloat16,
        device=device,
    )
    mid_lse = torch.empty(
        (num_tokens, num_heads, num_splits),
        dtype=torch.float32,
        device=device,
    )
    output = torch.empty_like(q)
    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=device)
    topk_length = (
        torch.full((num_tokens,), topk, dtype=torch.int32, device=device)
        if has_topk_length
        else None
    )
    extra_topk_length = (
        torch.full((num_tokens,), extra_topk, dtype=torch.int32, device=device)
        if has_extra_topk_length
        else None
    )
    attn_sink = (
        torch.zeros((num_heads,), dtype=torch.float32, device=device)
        if has_attn_sink
        else None
    )
    sm_scale = dsv4_nvfp4_format_info()["query_dim"] ** -0.5

    def build_decode(
        cpb: int,
    ) -> Callable[[torch.Tensor, Optional[torch.Tensor]], None]:
        def call(indices: torch.Tensor, extra_indices: Optional[torch.Tensor]) -> None:
            module.sparse_mla_sm120_nvfp4_decode(
                q,
                primary_cache,
                indices,
                mid_out,
                mid_lse,
                output,
                out_lse,
                num_splits,
                sm_scale,
                topk_length,
                attn_sink,
                extra_cache,
                extra_indices,
                extra_topk_length,
                cpb,
                False,
            )

        return call

    def prefill(indices: torch.Tensor, extra_indices: Optional[torch.Tensor]) -> None:
        module.sparse_mla_sm120_nvfp4_prefill(
            q,
            primary_cache,
            indices,
            output,
            out_lse,
            sm_scale,
            topk_length,
            attn_sink,
            extra_cache,
            extra_indices,
            extra_topk_length,
        )

    return build_decode, prefill


@dataclass(frozen=True)
class _Nvfp4MeasureContext:
    """Device-resident pools and the loaded module shared across buckets."""

    device: torch.device
    module: Any
    primary_cache: torch.Tensor
    primary_slots: int
    extra_cache: Optional[torch.Tensor]
    extra_slots: int


def _nvfp4_measure_context(
    device: torch.device,
    primary_page_size: int,
    topk: int,
    extra_topk: int,
    extra_page_size: int,
) -> _Nvfp4MeasureContext:
    from ._execution import get_sparse_mla_dsv4_nvfp4_module

    primary_cache, primary_slots, extra_cache, extra_slots = (
        _allocate_calibration_pools(
            device, primary_page_size, topk, extra_topk, extra_page_size
        )
    )
    return _Nvfp4MeasureContext(
        device=device,
        module=get_sparse_mla_dsv4_nvfp4_module(),
        primary_cache=primary_cache,
        primary_slots=primary_slots,
        extra_cache=extra_cache,
        extra_slots=extra_slots,
    )


def _measure_nvfp4_bucket(
    ctx: _Nvfp4MeasureContext,
    *,
    num_heads: int,
    topk: int,
    extra_topk: int,
    has_topk_length: bool,
    has_extra_topk_length: bool,
    has_attn_sink: bool,
    num_tokens: int,
) -> dict:
    index_sets = _make_index_sets(
        num_tokens=num_tokens,
        topk=topk,
        primary_slots=ctx.primary_slots,
        extra_topk=extra_topk,
        extra_slots=ctx.extra_slots,
        device=ctx.device,
    )
    build_decode, prefill = _make_calibration_calls(
        module=ctx.module,
        device=ctx.device,
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        primary_cache=ctx.primary_cache,
        extra_topk=extra_topk,
        extra_cache=ctx.extra_cache,
        has_topk_length=has_topk_length,
        has_extra_topk_length=has_extra_topk_length,
        has_attn_sink=has_attn_sink,
    )
    chunk = dsv4_nvfp4_format_info()["chunk_width"]
    num_splits = (topk + chunk - 1) // chunk
    num_splits += (extra_topk + chunk - 1) // chunk
    candidates = list(range(1, num_splits + 1)) + [0]
    measured = _cpb._balanced_timings(
        candidates,
        lambda cpb: _time_indexed_calls(
            build_decode(cpb) if cpb else prefill, index_sets, ctx.device
        ),
    )
    best_cpb = min(range(1, num_splits + 1), key=lambda cpb: (measured[cpb], -cpb))
    best_decode_us = measured[best_cpb]
    measured_prefill_us = measured[0]
    use_decode = best_decode_us <= _CROSSOVER_MARGIN * measured_prefill_us
    return {
        "variant": (
            NVFP4KernelVariant.DECODE_SPLITK.value
            if use_decode
            else NVFP4KernelVariant.PREFILL_STREAMING.value
        ),
        "cpb": best_cpb,
        "decode_s": best_decode_us / 1e6,
        "prefill_s": measured_prefill_us / 1e6,
    }


@_cpb._target_calibration
def calibrate_nvfp4_sparse_mla_sm120(
    device: torch.device,
    *,
    num_heads: int,
    topk: int,
    primary_page_size: int = 64,
    extra_topk: int = 0,
    extra_page_size: int = 0,
    has_topk_length: bool = False,
    has_extra_topk_length: bool = False,
    has_attn_sink: bool = False,
    force: bool = False,
) -> NVFP4CalibrationReport:
    """Calibrate one exact NVFP4 sparse-MLA shape family on an idle GPU.

    This is intentionally kept in the internal module while the cache ABI and
    public calibration surface are under upstream review. Normal serving
    warmup invokes it lazily inside ``autotune(True)``. To avoid calibration
    competing with startup memory profiling, stop serving, run this in a
    separate process on an idle GPU, check the report's ``persisted`` field,
    and exit that process before service startup. In service warmup use
    ``autotune(True, skip_ops={"sparse_mla_sm120_nvfp4"})`` to consume cached
    profiles without new measurement or off-grid token refinement. Use the
    same cache directory and target device. Ordinary eager scratch allocations
    remain; concurrent GPU work can still contaminate calibration measurements.

    This API is for NVFP4 KV, not merely NVFP4 model weights. The profile key
    includes the device and this call's shape/cache/presence configuration.
    Old schema or timing-protocol caches are not reused. Explicit calls retry
    missing or failed configurations; ``force=True`` re-measures present ones,
    retaining old profiles if measurement raises.

    Profiles are measured on the canonical packed page layout (page pitch
    ``page_size * 384`` bytes, inherently 16B-aligned, with dense indices and
    LSE) and apply only to it: calls with padded page strides keep the
    decode-first fallback instead of consuming or extending the profile.
    """
    device = torch.device(device)
    if torch.cuda.is_current_stream_capturing():
        raise CalibrationError(
            "NVFP4 sparse-MLA calibration must not run under CUDA graph capture"
        )
    decode_ok, prefill_ok = _eligible(
        1,
        num_heads,
        topk,
        primary_page_size,
        extra_topk,
        extra_page_size,
    )
    if not decode_ok or not prefill_ok:
        raise ValueError(
            "unsupported NVFP4 calibration shape: "
            f"heads={num_heads}, topk={topk}, primary_page_size="
            f"{primary_page_size}, extra_topk={extra_topk}, "
            f"extra_page_size={extra_page_size}"
        )
    if has_extra_topk_length and extra_topk == 0:
        raise ValueError("has_extra_topk_length requires extra_topk > 0")

    key = _request_key(
        num_heads=num_heads,
        topk=topk,
        primary_page_size=primary_page_size,
        extra_topk=extra_topk,
        extra_page_size=extra_page_size,
        has_topk_length=has_topk_length,
        has_extra_topk_length=has_extra_topk_length,
        has_attn_sink=has_attn_sink,
    )
    existing = _cpb.get_profile(key, device)
    if existing is not None and not force:
        buckets = existing["buckets"]
        reused_decode_by_t = {
            t: buckets[str(t)]["variant"] == NVFP4KernelVariant.DECODE_SPLITK.value
            for t in _CROSSOVER_PROBED_T
        }
        with _cpb._store_lock:
            overlay_pending = _cpb._activate_store()[1]["overlay"]
        return NVFP4CalibrationReport(
            key,
            num_heads,
            topk,
            extra_topk,
            extra_page_size,
            max(
                (t for t, decode in reused_decode_by_t.items() if decode), default=None
            ),
            {
                t: "decode" if decode else "prefill"
                for t, decode in reused_decode_by_t.items()
            },
            {t: buckets[str(t)]["cpb"] for t in _CROSSOVER_PROBED_T},
            {t: buckets[str(t)]["decode_s"] * 1e6 for t in _CROSSOVER_PROBED_T},
            {t: buckets[str(t)]["prefill_s"] * 1e6 for t in _CROSSOVER_PROBED_T},
            not overlay_pending,
        )

    ctx = _nvfp4_measure_context(
        device, primary_page_size, topk, extra_topk, extra_page_size
    )
    buckets = {
        str(num_tokens): _measure_nvfp4_bucket(
            ctx,
            num_heads=num_heads,
            topk=topk,
            extra_topk=extra_topk,
            has_topk_length=has_topk_length,
            has_extra_topk_length=has_extra_topk_length,
            has_attn_sink=has_attn_sink,
            num_tokens=num_tokens,
        )
        for num_tokens in _CROSSOVER_PROBED_T
    }
    cpb_by_t = {t: buckets[str(t)]["cpb"] for t in _CROSSOVER_PROBED_T}
    decode_us = {t: buckets[str(t)]["decode_s"] * 1e6 for t in _CROSSOVER_PROBED_T}
    prefill_us = {t: buckets[str(t)]["prefill_s"] * 1e6 for t in _CROSSOVER_PROBED_T}
    decode_by_t = {
        t: buckets[str(t)]["variant"] == NVFP4KernelVariant.DECODE_SPLITK.value
        for t in _CROSSOVER_PROBED_T
    }
    profile = {
        "request": json.loads(key),
        "sample_kind": _cpb._SAMPLE_KIND,
        "timing_protocol": _cpb._TIMING_PROTOCOL,
        "buckets": buckets,
    }
    persisted = _cpb.publish_calibration(device, _FAMILY, profiles={key: profile})
    _cpb._clear_calibration_failed(device, key)
    phase_by_t = {
        token_bucket: "decode" if use_decode else "prefill"
        for token_bucket, use_decode in decode_by_t.items()
    }
    logger.info(
        "Calibrated SM120 NVFP4 sparse MLA H=%d topk=%d extra_topk=%d "
        "extra_page_size=%d: phases=%s, cpb=%s",
        num_heads,
        topk,
        extra_topk,
        extra_page_size,
        phase_by_t,
        cpb_by_t,
    )
    return NVFP4CalibrationReport(
        key,
        num_heads,
        topk,
        extra_topk,
        extra_page_size,
        max((t for t, decode in decode_by_t.items() if decode), default=None),
        phase_by_t,
        cpb_by_t,
        decode_us,
        prefill_us,
        persisted,
    )


@_cpb._target_calibration
def refine_nvfp4(
    device: torch.device,
    *,
    tokens: int,
    num_heads: int,
    topk: int,
    primary_page_size: int,
    extra_topk: int,
    extra_page_size: int,
    has_topk_length: bool,
    has_extra_topk_length: bool,
    has_attn_sink: bool,
) -> Optional[dict]:
    """Measure one exact token count and merge it into the stored profile.

    Returns the refined bucket entry, or ``None`` when refinement does not
    apply (outside the decode envelope, no stored profile, or the exact entry
    already measured). A successful call clears the in-process failure
    suppression a failed tuning-time attempt left for this (key, tokens)
    point.
    """
    if not 1 <= tokens <= _DECODE_MAX_TOKENS:
        return None
    key = _request_key(
        num_heads=num_heads,
        topk=topk,
        primary_page_size=primary_page_size,
        extra_topk=extra_topk,
        extra_page_size=extra_page_size,
        has_topk_length=has_topk_length,
        has_extra_topk_length=has_extra_topk_length,
        has_attn_sink=has_attn_sink,
    )
    profile = _cpb.get_profile(key, device)
    if profile is None or str(tokens) in profile["buckets"]:
        return None
    ctx = _nvfp4_measure_context(
        device, primary_page_size, topk, extra_topk, extra_page_size
    )
    entry = _measure_nvfp4_bucket(
        ctx,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        has_topk_length=has_topk_length,
        has_extra_topk_length=has_extra_topk_length,
        has_attn_sink=has_attn_sink,
        num_tokens=tokens,
    )
    entry["provenance"] = "refined"
    with _cpb._store_lock:
        profile = _cpb.get_profile(key, device)
        if profile is None or str(tokens) in profile["buckets"]:
            return None
        profile["buckets"][str(tokens)] = entry
        persisted = _cpb.publish_calibration(
            device,
            _FAMILY,
            profiles={key: profile},
            profile_buckets={key: {str(tokens): entry}},
        )
    # A successful measurement clears the in-process failure suppression for
    # this (key, tokens) point.
    _cpb.clear_refine_failed(device, key, tokens)
    return {**entry, "persisted": persisted}


__all__ = [
    "NVFP4CalibrationReport",
    "NVFP4KernelVariant",
    "NVFP4PlannedCall",
    "calibrate_nvfp4_sparse_mla_sm120",
    "plan_nvfp4_sparse_mla_sm120",
]
