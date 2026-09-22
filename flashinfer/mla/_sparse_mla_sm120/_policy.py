# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Ordinary sparse-MLA policy: calibrated defaults and explicit FP8/BF16 routes.

The shared prepared facade consumes this policy after metadata validation.
Calibration completes before the current call chooses its phase. The C++ host
resolver owns concrete numerical/kernel resources and split geometry; neither
Wrapper nor functional execution reconstructs them. Calibration state is shared,
but owned scratch and LSE remain per Wrapper, never a global tensor pool.
"""

from __future__ import annotations

import enum
import functools
import logging
import threading
from dataclasses import dataclass
from typing import Iterator, Optional, TypeVar

import torch

from ...autotuner import AutoTuner
from . import _calibration as _cpb
from ._calibration import CalibrationError
from ._execution import FormatValues, format_info

logger = logging.getLogger(__name__)

_VariantT = TypeVar("_VariantT")

# Kernel-side constants. Mirrored from
# include/flashinfer/attention/sparse_mla_sm120/{arch,model}/*.cuh.
_D_V = 512  # value head dim (universal across DSV3_2 and DSV4)
_BI = 64  # KV partition tile size in candidates (BLOCK_SIZE_N)

# Decode-form cutoff: decode kernels are only ever offered num_tokens <= 64
# (policy; prefill serves any num_tokens >= 1).
_DECODE_MAX_TOKENS = 64

_MODEL_TYPE_DSV3_2 = 0
_MODEL_TYPE_DSV4 = 1
_MODEL_TYPE_GLM_NSA = 2
_MODEL_TYPE_GLM53_NOPE = 3
_MODEL_TYPE_DOTS3_SWA = 4
# DeepSeek-V4.1: GLM53_NOPE geometry (512-wide all-FP8 K, no BF16 rope) with a
# DSV4-style UE8M0 footer, but 32-wide quant groups (16 scales, 16B/token).
# Its 528B payload collides with GLM53_NOPE's, so it is only ever selected
# explicitly (kv_scale_format="ue8m0_g32"), never inferred from widths.
_MODEL_TYPE_DSV4_1 = 5

# d_v per model type. Every DeepSeek-family type is 512; DOTS3_SWA is the one
# divergence (its latent V is the full 1024-wide latent, rope excluded).
_D_V_BY_MODEL_TYPE = FormatValues({model: model for model in range(6)}, "value_dim")

# Kernel-family names used in the public config query and error messages.
_MODEL_TYPE_TO_FAMILY = _cpb.MODEL_FAMILIES


def _decode_chunk_width(model_type: int) -> int:
    """Kernel candidate-tile width (BI): candidates consumed per loop
    iteration. DOTS3_SWA halves it to 32 because its 1040-byte KV smem stride
    does not fit BI=64 on SM120 (see DecodeTileCfg in decode_dsv4_kernel.cuh).
    """
    return format_info(model_type)["chunk_width"]


# Default page size for calibration probes, not the eligibility boundary.
_PAGE_BLOCK_SIZE = 64

_DECODE_MAX_HEADS = 128


@dataclass(frozen=True)
class _StaticFamilyEnvelope:
    """Coarse static decode envelope of one ordinary kernel family.

    Pure Python by construction: init-time capability probes (vLLM imports the
    ``_DECODE_*_DISPATCH`` objects and ``supported_sparse_mla_sm120_configs``)
    must work on hosts without a GPU and under ``FLASHINFER_DISABLE_JIT``, so
    membership never resolves through the compiled module. The C++ dispatcher
    stays authoritative — a member shape can still be rejected at dispatch
    (layout, precision, dual-cache rules); this table is a coarse pre-filter,
    not a mirror of the compiled instantiation set, so no static-vs-compiled
    parity test pins it. ``page_block_sizes``/``page_block_size_is_runtime``
    describe the main-cache page envelope: every ordinary family currently
    accepts independent positive page sizes.
    """

    d_qk: int
    bytes_per_token: int
    min_topk: int = 1
    max_num_heads: int = _DECODE_MAX_HEADS
    page_block_size: int = 64  # default/probe page size
    page_block_sizes: frozenset[int] = frozenset()
    page_block_size_is_runtime: bool = True
    dedicated_heads: frozenset[int] = frozenset({8, 16, 32, 64, 128})


_STATIC_DECODE_ENVELOPE = {
    _MODEL_TYPE_DSV3_2: _StaticFamilyEnvelope(d_qk=576, bytes_per_token=656),
    _MODEL_TYPE_DSV4: _StaticFamilyEnvelope(d_qk=512, bytes_per_token=584),
    _MODEL_TYPE_GLM_NSA: _StaticFamilyEnvelope(d_qk=576, bytes_per_token=656),
    _MODEL_TYPE_GLM53_NOPE: _StaticFamilyEnvelope(
        d_qk=512, bytes_per_token=528, dedicated_heads=frozenset({8, 16, 32, 64})
    ),
    _MODEL_TYPE_DOTS3_SWA: _StaticFamilyEnvelope(
        d_qk=1088,
        bytes_per_token=1160,
        min_topk=513,
        dedicated_heads=frozenset({8, 16, 32, 64}),
    ),
    _MODEL_TYPE_DSV4_1: _StaticFamilyEnvelope(d_qk=512, bytes_per_token=528),
}


class _DecodeDispatchEnvelope:
    """Coarse ``(num_heads, topk)`` decode membership for one kernel family.

    topk is a runtime kernel argument, so the envelope is a predicate, not a
    pair set: ``(h, k) in envelope`` iff ``1 <= h <= max_num_heads`` and
    ``k >= min_topk`` in the family's static probe table (a pre-filter; the
    C++ dispatcher remains authoritative for member shapes). Iteration yields
    each dedicated head specialization paired with the minimum legal runtime
    topk, not every legal pair: ordinary decode has no topk specializations.
    """

    __slots__ = ("model",)

    def __init__(self, model: int) -> None:
        self.model = model

    def __iter__(self) -> Iterator[tuple[int, int]]:
        envelope = _STATIC_DECODE_ENVELOPE[self.model]
        topk = envelope.min_topk
        return ((head, topk) for head in sorted(envelope.dedicated_heads))

    def __contains__(self, pair: object) -> bool:
        if not isinstance(pair, tuple) or len(pair) != 2:
            return False
        h, k = pair
        if not isinstance(h, int) or not isinstance(k, int):
            return False
        envelope = _STATIC_DECODE_ENVELOPE[self.model]
        return 1 <= h <= envelope.max_num_heads and k >= envelope.min_topk

    def __repr__(self) -> str:
        envelope = _STATIC_DECODE_ENVELOPE[self.model]
        return (
            f"_DecodeDispatchEnvelope(num_heads<={envelope.max_num_heads}, "
            f"topk>={envelope.min_topk})"
        )


_DECODE_DSV4_DISPATCH = _DecodeDispatchEnvelope(_MODEL_TYPE_DSV4)
_DECODE_DSV3_2_DISPATCH = _DecodeDispatchEnvelope(_MODEL_TYPE_DSV3_2)
_DECODE_GLM53_NOPE_DISPATCH = _DecodeDispatchEnvelope(_MODEL_TYPE_GLM53_NOPE)
_DECODE_DOTS3_SWA_DISPATCH = _DecodeDispatchEnvelope(_MODEL_TYPE_DOTS3_SWA)
_DECODE_DSV4_1_DISPATCH = _DecodeDispatchEnvelope(_MODEL_TYPE_DSV4_1)

# Calibration/documented topk values per family (the crossover sweep points).
# Any width >= min_topk above is served; these are the values with measured
# crossover data.
_DECODE_DSV4_TOPKS = frozenset({128, 192, 256, 512, 1024})
_DECODE_DSV3_2_TOPKS = frozenset({128, 512, 1024, 2048})
_DECODE_GLM53_NOPE_TOPK = 2176
_DECODE_DOTS3_SWA_TOPK = 576
_DECODE_DSV4_1_TOPK = 512  # the V4.1 indexer topk

# Default profile sampling grids, not the eligibility envelope. Explicit
# calibration requests may use any shape allowed by the compiled capabilities.
_CALIBRATION_HEADS = (8, 16, 32, 64, 128)
_DECODE_DSV4_CALIBRATION_GRID = frozenset(
    (h, k) for h in _CALIBRATION_HEADS for k in _DECODE_DSV4_TOPKS
)
_DECODE_DSV3_2_CALIBRATION_GRID = frozenset(
    (h, k) for h in _CALIBRATION_HEADS for k in _DECODE_DSV3_2_TOPKS
)
_DECODE_GLM53_NOPE_CALIBRATION_GRID = frozenset(
    {(32, _DECODE_GLM53_NOPE_TOPK), (64, _DECODE_GLM53_NOPE_TOPK)}
)
_DECODE_DOTS3_SWA_CALIBRATION_GRID = frozenset(
    (h, _DECODE_DOTS3_SWA_TOPK) for h in (8, 16, 32, 64)
)
_DECODE_DSV4_1_CALIBRATION_GRID = frozenset(
    (h, _DECODE_DSV4_1_TOPK) for h in _CALIBRATION_HEADS
)


@functools.cache
def _decode_scratch_heads(num_heads: int) -> int:
    from ._execution import query

    return query("decode_scratch_heads", num_heads)


class KernelVariant(enum.IntEnum):
    """Launchable kernel families; the prefill values cross the FFI boundary."""

    DECODE_SPLITK = 0
    PREFILL_SG = 1
    PREFILL_MG = 2
    PREFILL_MG_DUAL = 3
    PREFILL_SWAPAB = 4


@dataclass(frozen=True)
class PlannedCall:
    """One routed call: the kernel variant and, for decode, the model cpb."""

    variant: KernelVariant
    cpb: int  # decode only; -1 selects the C++ heuristic


canonical_profile_layout = _cpb.canonical_profile_layout

# Lazy tuning-time calibration guard, symmetric with the NVFP4 policy's
# ``_calibration_lock``/``_calibrating``: a re-entrant or concurrent
# profile_selection for the same key must not start a second measurement.
_calibration_lock = threading.RLock()
_calibrating: set[tuple[str | None, str, str]] = set()


def _lazy_calibrated_profile(request, device) -> Optional[dict]:
    """Calibrate one ordinary request at tuning time, guarded against re-entry.

    Returns the profile dict, or ``None`` when measurement failed, is already
    in flight on this thread or another, or a concurrent caller produced the
    profile first.
    """
    with _cpb._store_lock:
        _cpb.refresh_store()
        scope = _cpb._active_scope
    guard_key = (scope, _cpb._device_key(device), request.key)
    with _calibration_lock:
        if guard_key in _calibrating:
            return None
        profile = _cpb.get_ordinary_profile(request, device)
        if profile is not None:
            return profile
        if _cpb.is_calibration_failed(device, request.key):
            return None
        _calibrating.add(guard_key)
        try:
            result = _cpb._calibrate_ordinary(request, device, False)
        finally:
            _calibrating.discard(guard_key)
    return None if result["status"] == "failed" else result


def profile_selection(metadata, device, precision: str) -> Optional[PlannedCall]:
    if precision == "bf16_qk":
        # This route has one SG specialization and no split-K/CPB choice.
        # Legacy FP8 profiles must not select its arithmetic or execution phase.
        return None
    if not canonical_profile_layout(metadata):
        return None
    m = metadata
    request = _cpb._OrdinaryRequest(
        m.heads,
        m.topk,
        precision,
        m.page_size,
        m.extra_topk,
        m.extra_page_size,
        m.extra_fp4,
        m.has_lengths,
        m.has_extra_lengths,
        m.has_sink,
        family=_MODEL_TYPE_TO_FAMILY[m.model],
    )
    profile = _cpb.get_ordinary_profile(request, device)
    tuner = AutoTuner.get()
    stack = tuner._get_skip_ops_stack()
    allowed = tuner.is_tuning_mode and not (stack and "sparse_mla_sm120" in stack[-1])
    if (
        profile is None
        and allowed
        and not _cpb._target_capturing(device)
        and not _cpb.is_calibration_failed(device, request.key)
    ):
        profile = _lazy_calibrated_profile(request, device)
    if profile is None:
        return None
    if (
        allowed
        and m.tokens <= _DECODE_MAX_TOKENS
        and str(m.tokens) not in profile["buckets"]
        and not _cpb._target_capturing(device)
        and not _cpb.is_refine_failed(device, request.key, m.tokens)
    ):
        # Tuning mode: measure this exact token count once so the selection
        # stops interpolating from the canonical grid. A failed measurement is
        # suppressed in-process instead of being retimed on every call.
        try:
            refined = _cpb.refine_ordinary(request, device, m.tokens)
        except (CalibrationError, RuntimeError) as error:
            logger.debug(
                "%s refine skipped at tokens=%d: %s", request.family, m.tokens, error
            )
            _cpb.mark_refine_failed(device, request.key, m.tokens)
            refined = None
        if refined is not None:
            profile = _cpb.get_ordinary_profile(request, device)
    bucket = _cpb._profile_bucket(profile, m.tokens)
    if bucket is None:
        pf = (
            KernelVariant.PREFILL_SG
            if precision == "bf16"
            else prefill_variant(
                m.model, m.heads, m.topk, m.page_size, m.extra_topk > 0, 0
            )
        )
        return None if pf is None else PlannedCall(pf, -1)
    return PlannedCall(KernelVariant(bucket["variant"]), bucket["cpb"])


def filter_metadata_selection(
    selected: Optional[PlannedCall],
    metadata,
    precision: str,
    sm_count: int,
    max_shared_bytes: int,
    preference: int = 0,
) -> Optional[PlannedCall]:
    from ._execution import metadata_candidates

    legal = metadata_candidates(metadata, precision, sm_count, max_shared_bytes)
    if (
        preference == _PREFILL_IMPL_SWAPAB
        and (selected is None or selected.variant is not KernelVariant.DECODE_SPLITK)
        and int(KernelVariant.PREFILL_SWAPAB) not in legal
    ):
        raise ValueError("prefill_impl='swapab' is unsupported for actual metadata")
    if selected is not None and int(selected.variant) in legal:
        return selected
    if int(KernelVariant.DECODE_SPLITK) in legal:
        return PlannedCall(
            KernelVariant.DECODE_SPLITK, 1 if precision != "default" else -1
        )
    return None


def _select_calibrated_variant(
    *,
    decode_eligible: bool,
    decode_variant: _VariantT,
    prefill_variant: Optional[_VariantT],
    decode_preferred: Optional[bool],
) -> Optional[_VariantT]:
    """Shared decode/prefill crossover policy.

    ``decode_eligible`` is an implementation envelope, not a performance
    threshold.  When both implementations can serve the call, the caller
    converts its independently keyed calibration record into
    ``decode_preferred``.  Missing calibration deliberately retains the safe
    historical decode-first fallback.

    The helper is format agnostic: FP8 and NVFP4 share this policy while
    looking up independently keyed measurements.
    """
    if not decode_eligible:
        return prefill_variant
    if prefill_variant is None:
        return decode_variant
    if decode_preferred is None or decode_preferred:
        return decode_variant
    return prefill_variant


@functools.cache
def _candidates(
    model_type: int,
    num_heads: int,
    topk: int,
    page_block_size: int,
    has_extra: bool,
    extra_page_size: int = 64,
    precision: str = "default",
) -> frozenset[int]:
    from ._execution import query

    return frozenset(
        query(
            "candidates",
            model_type,
            num_heads,
            topk,
            page_block_size,
            has_extra,
            extra_page_size,
            precision,
        )
    )


def decode_splitk_eligible(
    model_type: int,
    num_heads: int,
    topk: int,
    page_block_size: int,
    has_extra: bool,
    num_tokens: int,
) -> bool:
    return num_tokens <= _DECODE_MAX_TOKENS and 0 in _candidates(
        model_type, num_heads, topk, page_block_size, has_extra
    )


def prefill_swapab_eligible(
    model_type: int,
    num_heads: int,
    topk: int,
    page_block_size: int,
    has_extra: bool,
) -> bool:
    return 4 in _candidates(model_type, num_heads, topk, page_block_size, has_extra)


def prefill_sg_eligible(
    model_type: int,
    num_heads: int,
    topk: int,
    page_block_size: int,
    has_extra: bool,
) -> bool:
    return 1 in _candidates(model_type, num_heads, topk, page_block_size, has_extra)


def prefill_mg_eligible(
    model_type: int,
    num_heads: int,
    topk: int,
    page_block_size: int,
    has_extra: bool,
) -> bool:
    return 2 in _candidates(model_type, num_heads, topk, page_block_size, has_extra)


def prefill_mg_dual_eligible(
    model_type: int,
    num_heads: int,
    topk: int,
    page_block_size: int,
    has_extra: bool,
) -> bool:
    return 3 in _candidates(model_type, num_heads, topk, page_block_size, has_extra)


# ── prefill_impl override ──────────────────────────────────────────────────

_PREFILL_IMPL_AUTO = 0
_PREFILL_IMPL_SWAPAB = 1
_PREFILL_IMPL_MG = 2
_PREFILL_IMPL_FROM_STR = {
    "auto": _PREFILL_IMPL_AUTO,
    "swapab": _PREFILL_IMPL_SWAPAB,
    "mg": _PREFILL_IMPL_MG,
}


def _normalize_prefill_impl(prefill_impl: Optional[str]) -> int:
    if prefill_impl is None:
        return _PREFILL_IMPL_AUTO
    impl = _PREFILL_IMPL_FROM_STR.get(prefill_impl)
    if impl is None:
        raise ValueError(
            f"prefill_impl must be one of None, 'auto', 'swapab', 'mg'; "
            f"got {prefill_impl!r}"
        )
    return impl


def prefill_variant(
    model_type: int,
    num_heads: int,
    topk: int,
    page_block_size: int,
    has_extra: bool,
    prefill_impl_pref: int,
) -> Optional[KernelVariant]:
    """The prefill variant auto/forced routing selects; None if the prefill
    envelope does not serve the shape.

    Auto order: swapAB (where instantiated) then SG / MG / dual-MG.
    ``prefill_impl='swapab'`` raises ValueError when swapAB is ineligible;
    ``'mg'`` excludes swapAB (SG/MG/dual as before)."""
    candidates = _candidates(model_type, num_heads, topk, page_block_size, has_extra)
    if prefill_impl_pref == _PREFILL_IMPL_SWAPAB:
        if KernelVariant.PREFILL_SWAPAB not in _candidates(
            model_type, num_heads, topk, format_info(model_type)["page_size"], has_extra
        ):
            raise ValueError(
                "prefill_impl='swapab' unsupported V32-family/num_heads/topk % 64/dual-cache metadata"
            )
        if KernelVariant.PREFILL_SWAPAB not in candidates:
            raise ValueError(
                "prefill_impl='swapab' is not instantiated for "
                f"page_block_size={page_block_size}"
            )
        return KernelVariant.PREFILL_SWAPAB
    order: tuple[KernelVariant, ...] = (
        KernelVariant.PREFILL_SG,
        KernelVariant.PREFILL_MG,
        KernelVariant.PREFILL_MG_DUAL,
    )
    if prefill_impl_pref != _PREFILL_IMPL_MG:
        order = (KernelVariant.PREFILL_SWAPAB, *order)
    return next((variant for variant in order if variant in candidates), None)


# ── cpb resolution (decode launch parameter) ───────────────────────────────

_cpb_hot_cache: dict = {}
_cpb_hot_epoch = -1

# GLM_NSA shares the dsv3_2 kernel, ABI, and calibrated cpb constants; its
# crossover entries stay keyed "glm_nsa" (scale format moves prefill time).
_CPB_FAMILY_ALIAS = {"glm_nsa": "dsv3_2"}


def _resolve_cpb(
    device: torch.device,
    family: str,
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int,
) -> int:
    """Model-picked chunks_per_block; -1 selects the C++ heuristic fallback."""
    cpb_family = _CPB_FAMILY_ALIAS.get(family, family)
    with _cpb._store_lock:
        c = _cpb.get_constants(device, cpb_family)
        scope_epoch = _cpb._constants_version
    if c is None:
        return -1
    hot_key = (
        scope_epoch,
        _cpb._device_key(device),
        cpb_family,
        num_tokens,
        num_heads,
        topk,
        extra_topk,
    )
    global _cpb_hot_epoch
    if _cpb_hot_epoch != _cpb._constants_version:
        _cpb_hot_cache.clear()
        _cpb_hot_epoch = _cpb._constants_version
    cpb = _cpb_hot_cache.get(hot_key)
    if cpb is None:
        cpb = _cpb.select_cpb(
            num_tokens,
            num_heads,
            topk,
            extra_topk,
            c,
            chunk_width=_cpb._CHUNK_WIDTH[cpb_family],
        )
        _cpb_hot_cache[hot_key] = cpb
    return cpb


# ── The planner ────────────────────────────────────────────────────────────

_precision_plan_memo: dict = {}


def plan(
    num_tokens: int,
    num_heads: int,
    topk: int,
    model_type: int,
    page_block_size: int,
    has_extra: bool,
    prefill_impl_pref: int,
    device: torch.device,
    *,
    extra_topk: int = 0,
    extra_fp4: bool = False,
    compute_precision: str = "default",
    extra_page_block_size: int = 0,
) -> Optional[PlannedCall]:
    """Decode-first fallback when no configuration profile selects the call.

    Compiled capabilities define eligibility; analytical constants only estimate
    decode CPB. Forced prefill preferences do not change a legal decode phase.
    """
    if compute_precision == "bf16_qk":
        if (
            model_type != _MODEL_TYPE_GLM53_NOPE
            or num_tokens <= 0
            or num_heads != 16
            or topk not in (2112, 2176)
            or page_block_size != 64
            or has_extra
            or extra_topk
            or extra_fp4
        ):
            raise ValueError(
                "bf16_qk requires GLM53 NoPE, T>0, H=16, PBS=64, "
                "topk=2112/2176 and a single cache"
            )
        if prefill_impl_pref != _PREFILL_IMPL_AUTO:
            raise ValueError("bf16_qk prefill_impl must be None or auto")
        return PlannedCall(KernelVariant.PREFILL_SG, -1)
    if compute_precision != "default":
        if compute_precision not in ("fp8", "bf16"):
            raise ValueError(f"unsupported compute_precision={compute_precision!r}")
        if model_type != _MODEL_TYPE_DSV4_1:
            raise ValueError("explicit compute_precision requires DSV4.1 storage")
        eligible_decode = decode_splitk_eligible(
            model_type, num_heads, topk, page_block_size, has_extra, num_tokens
        )
        if compute_precision == "bf16":
            if prefill_impl_pref != _PREFILL_IMPL_AUTO:
                raise ValueError("BF16 prefill_impl must be None or auto")
            selected = (
                KernelVariant.DECODE_SPLITK
                if eligible_decode
                else KernelVariant.PREFILL_SG
            )
        else:
            pf = prefill_variant(
                model_type,
                num_heads,
                topk,
                page_block_size,
                has_extra,
                prefill_impl_pref,
            )
            selected = KernelVariant.DECODE_SPLITK if eligible_decode else pf
            if selected is None:
                raise ValueError(
                    "FP8 prefill requires heads in {8,16,32,64}, topk multiple of 64 and positive PBS"
                )
        key = (
            compute_precision,
            model_type,
            num_tokens,
            num_heads,
            topk,
            page_block_size,
            has_extra,
            extra_topk,
            extra_fp4,
            extra_page_block_size,
            str(device),
            prefill_impl_pref,
            "untuned_decode_first_v2",
        )
        if key not in _precision_plan_memo:
            _precision_plan_memo[key] = PlannedCall(
                selected, 1 if selected is KernelVariant.DECODE_SPLITK else -1
            )
        return _precision_plan_memo[key]
    decode_ok = decode_splitk_eligible(
        model_type, num_heads, topk, page_block_size, has_extra, num_tokens
    )
    pf = prefill_variant(
        model_type, num_heads, topk, page_block_size, has_extra, prefill_impl_pref
    )
    if pf is not None and pf not in _candidates(
        model_type,
        num_heads,
        topk,
        page_block_size,
        has_extra,
        extra_page_block_size or _PAGE_BLOCK_SIZE,
    ):
        pf = None
    if not decode_ok and pf is None:
        return None
    cpb = -1
    if decode_ok and model_type != _MODEL_TYPE_DSV4_1:
        cpb = _resolve_cpb(
            device,
            _MODEL_TYPE_TO_FAMILY[model_type],
            num_tokens,
            num_heads,
            topk,
            extra_topk,
        )
    variant = KernelVariant.DECODE_SPLITK if decode_ok else pf
    return PlannedCall(variant, cpb if variant is KernelVariant.DECODE_SPLITK else -1)
