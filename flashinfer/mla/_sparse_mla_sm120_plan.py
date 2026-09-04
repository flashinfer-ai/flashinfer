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

"""Dispatch planner for SM120 sparse-MLA: the single dispatch authority.

Owns the kernel-variant envelopes (extracted from the C++ launchers in
``csrc/sparse_mla_sm120_{prefill,decode_dsv3_2,decode_dsv4}.cu`` and kept in
sync by ``test_sparse_mla_sm120_envelope_consistency``) and all routing
policy: decode-vs-prefill crossover, swapAB preference, and the
``prefill_impl`` override. C++ keeps only per-variant envelope checks and
launches what the plan names.

All state is module-level (vLLM constructs a fresh runner per call).
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Optional

import torch

from ..autotuner import AutoTuner
from . import _sparse_mla_sm120_cpb as _cpb
from ._sparse_mla_sm120_cpb import CalibrationError

logger = logging.getLogger(__name__)

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
# The V32 kernel family: the 656B/token inline-scale cache ABI. GLM53_NOPE is
# the rope-free member (d_qk=512; bytes [528:656) are reserved padding).
# swapAB is instantiated for all V32 model types.
_V32_MODEL_TYPES = frozenset(
    {_MODEL_TYPE_DSV3_2, _MODEL_TYPE_GLM_NSA, _MODEL_TYPE_GLM53_NOPE}
)
_BPT_DSV3_2 = 656
_BPT_DSV4 = 584
_BPT_DOTS3_SWA = 1160

# d_v per model type. Every DeepSeek-family type is 512; DOTS3_SWA is the one
# divergence (its latent V is the full 1024-wide latent, rope excluded).
_D_V_BY_MODEL_TYPE = {
    _MODEL_TYPE_DSV3_2: 512,
    _MODEL_TYPE_DSV4: 512,
    _MODEL_TYPE_GLM_NSA: 512,
    _MODEL_TYPE_GLM53_NOPE: 512,
    _MODEL_TYPE_DOTS3_SWA: 1024,
}

# Kernel-family names used in the public config query and error messages.
_MODEL_TYPE_TO_FAMILY = {
    _MODEL_TYPE_DSV3_2: "dsv3_2",
    _MODEL_TYPE_DSV4: "dsv4",
    _MODEL_TYPE_GLM_NSA: "glm_nsa",
    _MODEL_TYPE_GLM53_NOPE: "glm53_nope",
    _MODEL_TYPE_DOTS3_SWA: "dots3_swa",
}


def _decode_chunk_width(model_type: int) -> int:
    """Kernel candidate-tile width (BI): candidates consumed per loop
    iteration. DOTS3_SWA halves it to 32 because its 1040-byte KV smem stride
    does not fit BI=64 on SM120 (see DecodeTileCfg in decode_dsv4_kernel.cuh).
    """
    return 32 if model_type == _MODEL_TYPE_DOTS3_SWA else _BI


# Every instantiated kernel (decode and prefill, both families) is compiled
# for the 64-token page layout; the C++ prefill launchers hardcode PBS=64.
_PAGE_BLOCK_SIZE = 64

# decode-dsv4 eligibility. num_heads in {8, 16, 32, 64, 128} keeps dedicated
# instantiations (measured 0.9-2.5% faster than runtime-H on hot shapes);
# every other num_heads in [1, 128] is served by one runtime-H instantiation
# (NUM_HEADS=0; the kernel zero-Q-pads the head tile and HPB-aligns the mid
# scratch). topk is a runtime kernel argument (the indices-row width), so
# eligibility is "num_heads in [1, 128] and topk >= min_topk" — not an
# enumerable pair set. The _DECODE_*_DISPATCH objects below implement that
# membership; vLLM's has_flashinfer_sparse_mla_sm120_config probes
# ``(num_heads, topk) in _DECODE_DSV4_DISPATCH``.
_DECODE_MAX_HEADS = 128


class _DecodeDispatchEnvelope:
    """(num_heads, topk) membership for one decode kernel family.

    topk is runtime, so the envelope is a predicate, not a pair set:
    ``(h, k) in envelope`` iff ``1 <= h <= _DECODE_MAX_HEADS and k >=
    min_topk``. ``min_topk`` is 513 for the sliding-window family (the window
    must fit the indices buffer) and 1 elsewhere.
    """

    __slots__ = ("min_topk",)

    def __init__(self, min_topk: int) -> None:
        self.min_topk = min_topk

    def __contains__(self, pair: object) -> bool:
        if not isinstance(pair, tuple) or len(pair) != 2:
            return False
        h, k = pair
        if not isinstance(h, int) or not isinstance(k, int):
            return False
        return 1 <= h <= _DECODE_MAX_HEADS and k >= self.min_topk

    def __repr__(self) -> str:
        return f"_DecodeDispatchEnvelope(num_heads<={_DECODE_MAX_HEADS}, topk>={self.min_topk})"


_DECODE_DSV4_DISPATCH = _DecodeDispatchEnvelope(1)

# decode-dsv3_2 eligibility (shared with GLM-NSA).
_DECODE_DSV3_2_DISPATCH = _DecodeDispatchEnvelope(1)

# GLM-5.3 native NoPE decode: topk=2176 folds the 128-token indexer tail
# into the 2048 sparse selection; any runtime width is served (2176 remains
# the calibration point).
_DECODE_GLM53_NOPE_DISPATCH = _DecodeDispatchEnvelope(1)

# DOTS3_SWA sliding-window decode, served by the decode-dsv4 kernel at BI=32 /
# 4 math warps. topk is the buffer width and must cover the 513-wide window;
# the window itself is clamped inside the kernel
# (DecodeTileCfg<DOTS3_SWA>::WINDOW). Runtime-H covers TP shards of a 64-head
# layer (TP4 -> 16) and any other count up to 128.
_DECODE_DOTS3_SWA_DISPATCH = _DecodeDispatchEnvelope(513)

# Calibration/documented topk values per family (the crossover sweep points).
# Any width >= min_topk above is served; these are the values with measured
# crossover data.
_DECODE_DSV4_TOPKS = frozenset({128, 192, 256, 512, 1024})
_DECODE_DSV3_2_TOPKS = frozenset({128, 512, 1024, 2048})
_DECODE_GLM53_NOPE_TOPK = 2176
_DECODE_DOTS3_SWA_TOPK = 576

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


def _decode_scratch_heads(num_heads: int) -> int:
    """Head rows the split-K decode scratch must carry for ``num_heads``.

    The dedicated num_heads=8 instantiation strides mid_out/mid_lse by the
    true head count; the runtime-H instantiation (all other eligible counts)
    HPB-aligns the scratch so both halves of the 16-head tile always exist.
    """
    if num_heads == 8:
        return 8
    return ((num_heads + 15) // 16) * 16


# Prefill instantiation envelope (single cache unless noted).
# topk (the indices row width) is a runtime kernel argument for every prefill
# variant: any topk >= 1 made of whole _BI=64 index tiles is served (the
# kernels issue whole tiles and the binding rejects ragged widths), so one
# instantiation per (model, variant, num_heads) covers every width — e.g. the
# V32 family's 2048, GLM53_NOPE's 2176, DOTS3_SWA's 576, and DSV4's
# 128..2048 all ride the same kernels.
_SG_HEADS = frozenset({8, 16})  # SG: 16-head CTA, NH=8 zero-padded
_MG_V32_HEADS = frozenset({32, 64, 128})  # MG: 32-head CTA
_PREFILL_DSV4_HEADS = frozenset({8, 16, 32, 64, 128})
_SWAPAB_HEADS = frozenset({64, 128})  # swapAB fills whole 64-head CTAs


def _prefill_topk_ok(topk: int) -> bool:
    """Prefill topk contract: whole 64-wide index tiles (binding-enforced)."""
    return topk >= 1 and topk % _BI == 0


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


# ── Envelopes (single source of truth; C++ re-checks them defensively) ────


def decode_splitk_eligible(
    model_type: int,
    num_heads: int,
    topk: int,
    page_block_size: int,
    has_extra: bool,
    num_tokens: int,
) -> bool:
    """True iff a standalone decode kernel is instantiated for this shape."""
    if num_tokens > _DECODE_MAX_TOKENS or page_block_size != _PAGE_BLOCK_SIZE:
        return False
    if model_type == _MODEL_TYPE_DSV4:
        # The decode-dsv4 kernel takes the secondary cache as runtime args.
        return (num_heads, topk) in _DECODE_DSV4_DISPATCH
    if model_type == _MODEL_TYPE_GLM53_NOPE:
        # decode-v32 has no dual-cache form.
        return not has_extra and (num_heads, topk) in _DECODE_GLM53_NOPE_DISPATCH
    if model_type == _MODEL_TYPE_DOTS3_SWA:
        # decode-dsv4 kernel, DOTS3_SWA tile; no dual-cache form for this family.
        return not has_extra and (num_heads, topk) in _DECODE_DOTS3_SWA_DISPATCH
    if model_type in _V32_MODEL_TYPES:
        # decode-dsv3_2 has no dual-cache form.
        return not has_extra and (num_heads, topk) in _DECODE_DSV3_2_DISPATCH
    return False


def prefill_swapab_eligible(
    model_type: int, num_heads: int, topk: int, page_block_size: int, has_extra: bool
) -> bool:
    # Single-cache only.
    return (
        model_type in _V32_MODEL_TYPES
        and not has_extra
        and page_block_size == _PAGE_BLOCK_SIZE
        and _prefill_topk_ok(topk)
        and num_heads in _SWAPAB_HEADS
    )


# DOTS3_SWA prefill is SG-only (D_NOPE=1024 does not fit the MG smem layout);
# num_heads > 16 is served by CTA replication. The indices buffer must cover
# the 513-wide sliding window (the kernel clamps the scan to the window).
_DOTS3_SWA_MIN_TOPK = 513
_DOTS3_SWA_SG_HEADS = frozenset({8, 16, 32, 64})


def prefill_sg_eligible(
    model_type: int, num_heads: int, topk: int, page_block_size: int, has_extra: bool
) -> bool:
    if model_type == _MODEL_TYPE_DOTS3_SWA:
        return (
            not has_extra
            and page_block_size == _PAGE_BLOCK_SIZE
            and topk >= _DOTS3_SWA_MIN_TOPK
            and topk % _BI == 0
            and num_heads in _DOTS3_SWA_SG_HEADS
        )
    return (
        model_type in _V32_MODEL_TYPES
        and not has_extra
        and page_block_size == _PAGE_BLOCK_SIZE
        and _prefill_topk_ok(topk)
        and num_heads in _SG_HEADS
    )


def prefill_mg_eligible(
    model_type: int, num_heads: int, topk: int, page_block_size: int, has_extra: bool
) -> bool:
    if has_extra or page_block_size != _PAGE_BLOCK_SIZE or not _prefill_topk_ok(topk):
        return False
    if model_type in _V32_MODEL_TYPES:
        return num_heads in _MG_V32_HEADS
    if model_type == _MODEL_TYPE_DSV4:
        return num_heads in _PREFILL_DSV4_HEADS
    return False


def prefill_mg_dual_eligible(
    model_type: int, num_heads: int, topk: int, page_block_size: int, has_extra: bool
) -> bool:
    return (
        model_type == _MODEL_TYPE_DSV4
        and has_extra
        and page_block_size == _PAGE_BLOCK_SIZE
        and _prefill_topk_ok(topk)
        and num_heads in _PREFILL_DSV4_HEADS
    )


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


def _check_swapab_eligible(
    model_type: int,
    num_heads: int,
    topk: int,
    has_extra: bool,
) -> None:
    """Raise ValueError when prefill_impl='swapab' meets an ineligible shape."""
    if has_extra:
        raise ValueError("prefill_impl='swapab' does not support dual-cache")
    if model_type not in _V32_MODEL_TYPES:
        raise ValueError(
            "prefill_impl='swapab' requires a V32-family model type "
            f"(dsv3_2, glm_nsa, or glm53_nope); got family={_MODEL_TYPE_TO_FAMILY[model_type]!r}"
        )
    if not _prefill_topk_ok(topk):
        raise ValueError(
            f"prefill_impl='swapab' requires topk >= 1 with topk % 64 == 0 "
            f"(whole index tiles); got topk={topk}"
        )
    if num_heads not in _SWAPAB_HEADS:
        raise ValueError(
            f"prefill_impl='swapab' requires num_heads in {sorted(_SWAPAB_HEADS)}; "
            f"got num_heads={num_heads}"
        )


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
    if prefill_impl_pref == _PREFILL_IMPL_SWAPAB:
        _check_swapab_eligible(model_type, num_heads, topk, has_extra)
        if page_block_size != _PAGE_BLOCK_SIZE:
            return None
        return KernelVariant.PREFILL_SWAPAB
    if prefill_impl_pref != _PREFILL_IMPL_MG and prefill_swapab_eligible(
        model_type, num_heads, topk, page_block_size, has_extra
    ):
        return KernelVariant.PREFILL_SWAPAB
    if prefill_sg_eligible(model_type, num_heads, topk, page_block_size, has_extra):
        return KernelVariant.PREFILL_SG
    if prefill_mg_eligible(model_type, num_heads, topk, page_block_size, has_extra):
        return KernelVariant.PREFILL_MG
    if prefill_mg_dual_eligible(
        model_type, num_heads, topk, page_block_size, has_extra
    ):
        return KernelVariant.PREFILL_MG_DUAL
    return None


# ── cpb resolution (decode launch parameter) ───────────────────────────────

# Memoized select_cpb results; the trailing constants-version entry makes
# entries self-invalidating when calibration stores new constants.
_cpb_hot_cache: dict = {}

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
    c = _cpb.get_constants(device, cpb_family)
    tuner = AutoTuner.get()
    tuning = tuner.is_tuning_mode
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
        from ._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module

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
        from ._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module

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
        _cpb._device_key(device),
        cpb_family,
        num_tokens,
        num_heads,
        topk,
        extra_topk,
        _cpb._constants_version,
    )
    cpb = _cpb_hot_cache.get(hot_key)
    if cpb is None:
        cpb = _cpb.get_cpb_override(device, cpb_family, num_heads, topk, num_tokens)
        if (
            cpb is None
            and tuning
            and not skipped
            # Refinement measures single-cache shapes only; dual-cache picks
            # stay on the model (their measured pick error is within ~6%).
            and extra_topk == 0
            and not torch.cuda.is_current_stream_capturing()
        ):
            from ._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module

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

# num_tokens bucket for the memo: exact T inside the decode envelope, one
# sentinel above it (the plan is T-independent once T > _DECODE_MAX_TOKENS).
_T_LARGE = -1

# Memoized variant decisions, keyed per the contract in plan(); the trailing
# constants-version entry self-invalidates when calibration stores new
# crossover data. cpb is NOT memoized here — it depends on the exact
# extra_topk and rides the exact-key _cpb_hot_cache instead.
_plan_memo: dict = {}


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
) -> Optional[PlannedCall]:
    """Route one call to a kernel variant; None when no envelope serves it.

    Policy: a decode-instantiated decode-form call takes DECODE_SPLITK up to
    the calibrated ``decode_max_tokens`` crossover for
    ``(model_type, num_heads, topk)`` (decode-first when uncalibrated);
    everything else takes the prefill variant from :func:`prefill_variant`.
    A forced swapab preference raises ValueError on ineligible shapes rather
    than returning None."""
    t_bucket = num_tokens if num_tokens <= _DECODE_MAX_TOKENS else _T_LARGE
    key = (
        model_type,
        num_heads,
        topk,
        page_block_size,
        has_extra,
        t_bucket,
        prefill_impl_pref,
        _cpb._device_key(device),
        _cpb._constants_version,
    )
    variant = _plan_memo.get(key)
    if variant is None and key not in _plan_memo:
        variant = _decide(
            num_tokens,
            num_heads,
            topk,
            model_type,
            page_block_size,
            has_extra,
            prefill_impl_pref,
            device,
        )
        _plan_memo[key] = variant
    if variant is None:
        return None
    if variant is KernelVariant.DECODE_SPLITK:
        cpb = _resolve_cpb(
            device,
            _MODEL_TYPE_TO_FAMILY[model_type],
            num_tokens,
            num_heads,
            topk,
            extra_topk,
        )
        return PlannedCall(variant, cpb)
    return PlannedCall(variant, -1)


def _decide(
    num_tokens: int,
    num_heads: int,
    topk: int,
    model_type: int,
    page_block_size: int,
    has_extra: bool,
    prefill_impl_pref: int,
    device: torch.device,
) -> Optional[KernelVariant]:
    pf = prefill_variant(
        model_type, num_heads, topk, page_block_size, has_extra, prefill_impl_pref
    )
    if not decode_splitk_eligible(
        model_type, num_heads, topk, page_block_size, has_extra, num_tokens
    ):
        return pf
    if pf is None:
        return KernelVariant.DECODE_SPLITK
    crossover = _cpb.get_decode_max_tokens(
        device, _MODEL_TYPE_TO_FAMILY[model_type], num_heads, topk
    )
    if crossover is None or num_tokens <= crossover:
        return KernelVariant.DECODE_SPLITK
    return pf
