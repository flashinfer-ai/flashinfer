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
and records the measured phase plus the best decode CPB for each probe bucket.
Per-bucket phase records preserve GPU wave-boundary non-monotonicity that a
single threshold cannot express.  The dispatch policy and persistence
primitives are shared with the existing FP8 planner; FP8 measurements are
never reused for NVFP4.
"""

from __future__ import annotations

import enum
import functools
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from ..autotuner import AutoTuner
from . import _sparse_mla_sm120_calibration as _cpb
from ._sparse_mla_sm120_calibration import CalibrationError
from ._sparse_mla_sm120_policy import _select_calibrated_variant, _DECODE_MAX_TOKENS
from ._sparse_mla_sm120_execution import dsv4_nvfp4_format_info

logger = logging.getLogger(__name__)

_PLAN_VERSION = 1
_FAMILY_PREFIX = f"dsv4_nvfp4_v{_PLAN_VERSION}"
_AUTOTUNE_OP = "sparse_mla_sm120_nvfp4"


@functools.cache
def _supports(
    heads: int, topk: int, page: int, extra_topk: int, extra_page: int
) -> bool:
    from ._sparse_mla_sm120_execution import query

    return query(
        "supports_attention",
        heads,
        topk,
        page,
        extra_topk,
        extra_page,
        is_dsv4_nvfp4=True,
    )


# Match the FP8 crossover policy, adding T=1 because single-request decode is
# an important NVFP4 latency corner.  Non-probe decode batches use the next
# larger bucket's measured CPB.
_CROSSOVER_PROBED_T = (1, 4, 8, 16, 24, 32, 48, 64)
_CROSSOVER_MARGIN = _cpb._CROSSOVER_MARGIN

# The calibration timing protocol queues calls and rotates index sets, keeping
# launch latency overlap and cache residency close to steady-state serving.
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
_calibrating: set[tuple[str, str, int, int]] = set()


def _token_bucket(num_tokens: int) -> int:
    for bucket in _CROSSOVER_PROBED_T:
        if num_tokens <= bucket:
            return bucket
    return _CROSSOVER_PROBED_T[-1]


def _family_key(
    *,
    primary_page_size: int,
    extra_topk: int,
    extra_page_size: int,
    has_topk_length: bool,
    has_extra_topk_length: bool,
    has_attn_sink: bool,
) -> str:
    """Build a format- and ABI-specific calibration namespace."""
    return (
        f"{_FAMILY_PREFIX}_p{primary_page_size}_e{extra_topk}"
        f"p{extra_page_size}_l{int(has_topk_length)}"
        f"x{int(has_extra_topk_length)}_s{int(has_attn_sink)}"
    )


def _phase_family(family: str, token_bucket: int) -> str:
    """Namespace one measured phase decision without assuming monotonicity."""
    return f"{family}_t{token_bucket}"


def _phase_grid_complete(
    device: torch.device, family: str, num_heads: int, topk: int
) -> bool:
    return all(
        _cpb.get_decode_max_tokens(
            device, _phase_family(family, token_bucket), num_heads, topk
        )
        is not None
        for token_bucket in _CROSSOVER_PROBED_T
    )


def _monotonic_decode_max_tokens(
    decode_by_token_bucket: dict[int, bool],
) -> Optional[int]:
    """Return a threshold only when measured phase choices form a prefix."""
    decode_max_tokens = 0
    saw_prefill = False
    for token_bucket in sorted(decode_by_token_bucket):
        if decode_by_token_bucket[token_bucket]:
            if saw_prefill:
                return None
            decode_max_tokens = token_bucket
        else:
            saw_prefill = True
    return decode_max_tokens


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


def _autotune_skipped() -> bool:
    tuner = AutoTuner.get()
    skip_stack = tuner._get_skip_ops_stack()
    return bool(skip_stack) and _AUTOTUNE_OP in skip_stack[-1]


def _maybe_calibrate(
    *,
    device: torch.device,
    family: str,
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
        or _cpb.is_crossover_failed(device, family)
    ):
        return

    with _cpb._store_lock:
        _cpb.refresh_store()
        scope = _cpb._active_scope
    key = (scope, _cpb._device_key(device), family, num_heads, topk)
    with _calibration_lock:
        if _phase_grid_complete(device, family, num_heads, topk):
            return
        if key in _calibrating:
            return
        _calibrating.add(key)
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
            _cpb.mark_crossover_failed(device, family)
        finally:
            _calibrating.discard(key)


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
) -> Optional[NVFP4PlannedCall]:
    """Select independently calibrated NVFP4 prefill/decode execution."""
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

    family = _family_key(
        primary_page_size=primary_page_size,
        extra_topk=extra_topk,
        extra_page_size=extra_page_size,
        has_topk_length=has_topk_length,
        has_extra_topk_length=has_extra_topk_length,
        has_attn_sink=has_attn_sink,
    )
    token_bucket = _token_bucket(num_tokens)
    phase_family = _phase_family(family, token_bucket)
    # This family's decode_max_tokens record is an encoded bucket phase,
    # not a threshold for other token buckets.
    with _cpb._store_lock:
        encoded_bucket_phase = _cpb.get_decode_max_tokens(
            device, phase_family, num_heads, topk
        )
        scope_epoch = _cpb._constants_version
    if decode_ok and encoded_bucket_phase is None:
        _maybe_calibrate(
            device=device,
            family=family,
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
            encoded_bucket_phase = _cpb.get_decode_max_tokens(
                device, phase_family, num_heads, topk
            )
            scope_epoch = _cpb._constants_version

    t_bucket = token_bucket if num_tokens <= _DECODE_MAX_TOKENS else -1
    memo_key = (
        scope_epoch,
        family,
        num_heads,
        topk,
        t_bucket,
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
            # Each bucket stores its own phase.  The shared selector consumes
            # the decision directly, so NVFP4 does not inherit FP8's
            # monotonic crossover assumption.
            decode_preferred=(
                None if encoded_bucket_phase is None else encoded_bucket_phase > 0
            ),
        )
        if variant is None:
            return None
        _plan_memo[memo_key] = variant

    if variant is NVFP4KernelVariant.PREFILL_STREAMING:
        return NVFP4PlannedCall(variant, 0)
    cpb = _cpb.get_cpb_override(
        device, family, num_heads, topk, _token_bucket(num_tokens)
    )
    return NVFP4PlannedCall(variant, 0 if cpb is None else cpb)


def _allocate_cache_pool(
    page_size: int, pool_bytes: int, device: torch.device
) -> tuple[torch.Tensor, int]:
    from ._sparse_mla_sm120_dsv4_nvfp4 import nvfp4_quantize_pack_sparse_mla_cache

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


def _allocate_calibration_pools(
    device: torch.device,
    primary_page_size: int,
    topk: int,
    extra_topk: int,
    extra_page_size: int,
) -> tuple[torch.Tensor, int, Optional[torch.Tensor], int]:
    total_bytes = _POOL_BYTES_TARGET
    while True:
        total_topk = topk + extra_topk
        if extra_topk:
            primary_bytes = max(_MIN_POOL_PER_SEGMENT, total_bytes * topk // total_topk)
            extra_bytes = max(
                _MIN_POOL_PER_SEGMENT, total_bytes * extra_topk // total_topk
            )
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
        except torch.cuda.OutOfMemoryError:
            del primary_cache, extra_cache
            if total_bytes <= _POOL_BYTES_MIN:
                raise CalibrationError(
                    "cannot allocate a >=512 MiB aggregate NVFP4 KV pool for "
                    "sparse-MLA calibration"
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
    public calibration surface are under upstream review.  Normal serving
    warmup invokes it lazily inside ``autotune(True)``.
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

    family = _family_key(
        primary_page_size=primary_page_size,
        extra_topk=extra_topk,
        extra_page_size=extra_page_size,
        has_topk_length=has_topk_length,
        has_extra_topk_length=has_extra_topk_length,
        has_attn_sink=has_attn_sink,
    )
    existing_phase = {
        token_bucket: _cpb.get_decode_max_tokens(
            device,
            _phase_family(family, token_bucket),
            num_heads,
            topk,
        )
        for token_bucket in _CROSSOVER_PROBED_T
    }
    if all(value is not None for value in existing_phase.values()) and not force:
        cpbs = {
            t: cpb
            for t in _CROSSOVER_PROBED_T
            if (cpb := _cpb.get_cpb_override(device, family, num_heads, topk, t))
            is not None
        }
        return NVFP4CalibrationReport(
            family,
            num_heads,
            topk,
            extra_topk,
            extra_page_size,
            _monotonic_decode_max_tokens(
                {
                    token_bucket: bool(value)
                    for token_bucket, value in existing_phase.items()
                }
            ),
            {
                token_bucket: "decode" if value else "prefill"
                for token_bucket, value in existing_phase.items()
            },
            cpbs,
            {},
            {},
            not _cpb._activate_store()[1]["overlay"],
        )

    from ._sparse_mla_sm120_execution import get_sparse_mla_dsv4_nvfp4_module

    module = get_sparse_mla_dsv4_nvfp4_module()
    primary_cache, primary_slots, extra_cache, extra_slots = (
        _allocate_calibration_pools(
            device,
            primary_page_size,
            topk,
            extra_topk,
            extra_page_size,
        )
    )
    num_splits = (
        topk + dsv4_nvfp4_format_info()["chunk_width"] - 1
    ) // dsv4_nvfp4_format_info()["chunk_width"]
    num_splits += (
        extra_topk + dsv4_nvfp4_format_info()["chunk_width"] - 1
    ) // dsv4_nvfp4_format_info()["chunk_width"]
    cpb_by_t: dict[int, int] = {}
    decode_us: dict[int, float] = {}
    prefill_us: dict[int, float] = {}
    decode_by_t: dict[int, bool] = {}

    for num_tokens in _CROSSOVER_PROBED_T:
        index_sets = _make_index_sets(
            num_tokens=num_tokens,
            topk=topk,
            primary_slots=primary_slots,
            extra_topk=extra_topk,
            extra_slots=extra_slots,
            device=device,
        )
        build_decode, prefill = _make_calibration_calls(
            module=module,
            device=device,
            num_tokens=num_tokens,
            num_heads=num_heads,
            topk=topk,
            primary_cache=primary_cache,
            extra_topk=extra_topk,
            extra_cache=extra_cache,
            has_topk_length=has_topk_length,
            has_extra_topk_length=has_extra_topk_length,
            has_attn_sink=has_attn_sink,
        )
        best_cpb = 1
        best_decode_us = float("inf")
        for cpb in range(1, num_splits + 1):
            latency_us = _time_indexed_calls(build_decode(cpb), index_sets, device)
            if latency_us <= best_decode_us:
                best_cpb, best_decode_us = cpb, latency_us
        measured_prefill_us = _time_indexed_calls(prefill, index_sets, device)
        cpb_by_t[num_tokens] = best_cpb
        decode_us[num_tokens] = best_decode_us
        prefill_us[num_tokens] = measured_prefill_us
        decode_by_t[num_tokens] = (
            best_decode_us <= _CROSSOVER_MARGIN * measured_prefill_us
        )

    # The persisted decode_max_tokens field encodes one bucket's phase here,
    # not a monotonic threshold: positive means decode, zero means prefill.
    encoded_bucket_phases = {
        f"{_phase_family(family, token_bucket)}|{num_heads}|{topk}": (
            token_bucket if use_decode else 0
        )
        for token_bucket, use_decode in decode_by_t.items()
    }
    persisted = _cpb.publish_calibration(
        device,
        family,
        crossover=encoded_bucket_phases,
        overrides={
            f"{family}|{num_heads}|{topk}|{t}": cpb for t, cpb in cpb_by_t.items()
        },
    )
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
        family,
        num_heads,
        topk,
        extra_topk,
        extra_page_size,
        _monotonic_decode_max_tokens(decode_by_t),
        phase_by_t,
        cpb_by_t,
        decode_us,
        prefill_us,
        persisted,
    )


__all__ = [
    "NVFP4CalibrationReport",
    "NVFP4KernelVariant",
    "NVFP4PlannedCall",
    "calibrate_nvfp4_sparse_mla_sm120",
    "plan_nvfp4_sparse_mla_sm120",
]
