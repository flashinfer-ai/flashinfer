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
from dataclasses import dataclass
from typing import Optional, TypeVar

import torch

from ..autotuner import AutoTuner
from . import _sparse_mla_sm120_calibration as _cpb
from ._sparse_mla_sm120_calibration import CalibrationError
from ._sparse_mla_sm120_execution import FormatValues, format_info

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


class _DecodeDispatchEnvelope:
    """Compatibility membership probe backed by compiled resolver capabilities."""

    def __init__(self, model: int) -> None:
        self.model = model

    def __contains__(self, pair: object) -> bool:
        if not isinstance(pair, tuple) or len(pair) != 2:
            return False
        h, k = pair
        return (
            isinstance(h, int)
            and isinstance(k, int)
            and 0 in _candidates(self.model, h, k, _PAGE_BLOCK_SIZE, False)
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

# Crossover-calibration grids: the (num_heads, topk) pairs the tuning-mode
# sweep times on both paths. Deliberately NOT the full eligibility envelope —
# calibrating every head count and topk width would explode the sweep. Every
# grid head count hits a dedicated instantiation, so the sweep times exactly
# the kernels production decode calls launch; off-grid shapes keep the
# decode-first default until a measured entry exists.
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
    from ._sparse_mla_sm120_execution import query

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


def profile_selection(metadata, device, precision: str) -> Optional[PlannedCall]:
    if metadata.model != _MODEL_TYPE_DSV4_1:
        return None
    m = metadata
    request = _cpb._Dsv41Request(
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
    )
    profile = _cpb.get_dsv41_profile(request, device)
    tuner = AutoTuner.get()
    stack = tuner._get_skip_ops_stack()
    allowed = tuner.is_tuning_mode and not (stack and "sparse_mla_sm120" in stack[-1])
    if (
        profile is None
        and allowed
        and not _cpb._target_capturing(device)
        and not _cpb.is_calibration_failed(device, request.key)
    ):
        result = _cpb._calibrate_dsv41(request, device, False)
        profile = result if result["status"] != "failed" else None
    if profile is None:
        return None
    bucket = _cpb._profile_bucket(profile, m.tokens)
    if bucket is None:
        return PlannedCall(KernelVariant.PREFILL_SG, -1)
    return PlannedCall(KernelVariant(bucket["variant"]), bucket["cpb"])


def filter_metadata_selection(
    selected: Optional[PlannedCall],
    metadata,
    precision: str,
    sm_count: int,
    max_shared_bytes: int,
) -> Optional[PlannedCall]:
    from ._sparse_mla_sm120_execution import metadata_candidates

    if metadata.model != _MODEL_TYPE_DSV4_1:
        return selected
    legal = metadata_candidates(metadata, precision, sm_count, max_shared_bytes)
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
    from ._sparse_mla_sm120_execution import query

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
        return (
            KernelVariant.PREFILL_SWAPAB
            if KernelVariant.PREFILL_SWAPAB in candidates
            else None
        )
    order = (
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
    tuner = AutoTuner.get()
    tuning = (
        family != "dsv4_1"
        and tuner.is_tuning_mode
        and not _cpb._target_capturing(device)
    )
    # autotune(skip_ops={"sparse_mla_sm120"}) opts out of the multi-second
    # calibration passes too, not only of choose_one.
    skip_stack = tuner._get_skip_ops_stack()
    skipped = bool(skip_stack) and "sparse_mla_sm120" in skip_stack[-1]
    if (
        c is None
        and tuning
        and not skipped
        and not _cpb.is_calibration_failed(device, cpb_family)
    ):
        from ._sparse_mla_sm120_execution import (
            get_sparse_mla_sm120_module as _get_sparse_mla_sm120_decode_module,
        )

        try:
            c = _cpb.calibrate(_get_sparse_mla_sm120_decode_module, cpb_family, device)
        except (CalibrationError, torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(
                "SM120 sparse-MLA %s cpb calibration failed (%s); "
                "falling back to the C++ heuristic for this process.",
                cpb_family,
                e,
            )
            _cpb.mark_calibration_failed(device, cpb_family)
        else:
            _cpb.save_constants(device, cpb_family, c)
    if (
        c is not None
        and tuning
        and not skipped
        and not _cpb.is_crossover_failed(device, family)
        and not _cpb.crossover_grid_complete(device, family)
    ):
        from ._sparse_mla_sm120_execution import (
            get_sparse_mla_sm120_module as _get_sparse_mla_sm120_decode_module,
        )

        try:
            # glm_nsa crossover entries are produced by the dsv3_2 crossover
            # calibration (shared kernel; separate key space).
            table = _cpb.calibrate_crossover(
                _get_sparse_mla_sm120_decode_module(), device, cpb_family, c
            )
        except (CalibrationError, torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(
                "SM120 sparse-MLA %s crossover calibration failed (%s); "
                "keeping the decode-first routing default for this process.",
                family,
                e,
            )
            _cpb.mark_crossover_failed(device, family)
        else:
            _cpb.save_crossover(device, table)
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
    if cpb is None or (tuning and not skipped):
        single_cache_cpb_override = (
            _cpb.get_cpb_override(device, cpb_family, num_heads, topk, num_tokens)
            if extra_topk == 0
            else None
        )
        cpb = single_cache_cpb_override
        if (
            cpb is None
            and tuning
            and not skipped
            # Refinement measures single-cache shapes only; dual-cache picks
            # stay on the model (their measured pick error is within ~6%).
            and extra_topk == 0
        ):
            from ._sparse_mla_sm120_execution import (
                get_sparse_mla_sm120_module as _get_sparse_mla_sm120_decode_module,
            )

            try:
                cpb = _cpb.refine_cpb(
                    _get_sparse_mla_sm120_decode_module,
                    cpb_family,
                    device,
                    c,
                    num_tokens,
                    num_heads,
                    topk,
                )
            except (CalibrationError, torch.cuda.OutOfMemoryError, RuntimeError) as e:
                logger.warning(
                    "SM120 sparse-MLA %s cpb refinement failed at T=%d H=%d "
                    "topk=%d (%s); using the model pick.",
                    cpb_family,
                    num_tokens,
                    num_heads,
                    topk,
                    e,
                )
                cpb = None
            else:
                _cpb.save_cpb_override(
                    device, cpb_family, num_heads, topk, num_tokens, cpb
                )
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
    """Route one call to a kernel variant; None when no envelope serves it.

    Policy: a decode-instantiated decode-form call takes DECODE_SPLITK up to
    the calibrated ``decode_max_tokens`` crossover for
    ``(model_type, num_heads, topk)`` (decode-first when uncalibrated);
    everything else takes the prefill variant from :func:`prefill_variant`.
    A forced swapab preference raises ValueError on ineligible shapes rather
    than returning None."""
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
    if decode_ok and not (
        model_type == _MODEL_TYPE_DSV4_1 and (has_extra or page_block_size != 64)
    ):
        cpb = _resolve_cpb(
            device,
            _MODEL_TYPE_TO_FAMILY[model_type],
            num_tokens,
            num_heads,
            topk,
            extra_topk,
        )
    # Dual DSV4_1 gathers need separate calibration, including the FP4 format.
    crossover = (
        None
        if model_type == _MODEL_TYPE_DSV4_1 and (has_extra or page_block_size != 64)
        else _cpb.get_decode_max_tokens(
            device, _MODEL_TYPE_TO_FAMILY[model_type], num_heads, topk
        )
    )
    variant = _select_calibrated_variant(
        decode_eligible=decode_ok,
        decode_variant=KernelVariant.DECODE_SPLITK,
        prefill_variant=pf,
        decode_preferred=(None if crossover is None else num_tokens <= crossover),
    )
    return PlannedCall(variant, cpb if variant is KernelVariant.DECODE_SPLITK else -1)
