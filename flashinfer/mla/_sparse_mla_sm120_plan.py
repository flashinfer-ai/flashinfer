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
_V32_MODEL_TYPES = frozenset(
    {_MODEL_TYPE_DSV3_2, _MODEL_TYPE_GLM_NSA, _MODEL_TYPE_GLM53_NOPE}
)
# swapAB is instantiated for all V32 model types; GLM53_NOPE serves
# topk=2176, the others topk=2048.
_SWAPAB_MODEL_TYPES = _V32_MODEL_TYPES
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

# decode-dsv4 instantiation set. NH=8 is the small-TP corner case; the kernel
# pads the head tile to HPB=16 with zero-Q rows and gates writes by NUM_HEADS.
_DECODE_DSV4_DISPATCH = frozenset(
    {
        (8, 128),
        (8, 192),
        (8, 256),
        (8, 512),
        (8, 1024),
        (16, 128),
        (16, 192),
        (16, 256),
        (16, 512),
        (16, 1024),
        (32, 128),
        (32, 192),
        (32, 256),
        (32, 512),
        (32, 1024),
        (64, 128),
        (64, 192),
        (64, 256),
        (64, 512),
        (64, 1024),
        (128, 128),
        (128, 192),
        (128, 256),
        (128, 512),
        (128, 1024),
    }
)

# decode-dsv3_2 instantiation set (shared with GLM-NSA).
_DECODE_DSV3_2_DISPATCH = frozenset(
    {
        (8, 128),
        (8, 512),
        (8, 1024),
        (8, 2048),
        (16, 128),
        (16, 512),
        (16, 1024),
        (16, 2048),
        (32, 128),
        (32, 512),
        (32, 1024),
        (32, 2048),
        (64, 128),
        (64, 512),
        (64, 1024),
        (64, 2048),
        (128, 128),
        (128, 512),
        (128, 1024),
        (128, 2048),
    }
)

# GLM-5.3 native NoPE decode: topk=2176 folds the 128-token indexer tail
# into the 2048 sparse selection. 64 heads is the TP1 shape; (32, 2176) is
# the TP2 shape of the same model.
_DECODE_GLM53_NOPE_DISPATCH = frozenset({(32, 2176), (64, 2176)})

# DOTS3_SWA sliding-window decode, served by the decode-dsv4 kernel at BI=32 /
# 4 math warps. topk=576 is the tightest multiple of the BI=32 tile (18
# chunks) covering the 513-wide window; the window itself is clamped inside
# the kernel (DecodeTileCfg<DOTS3_SWA>::WINDOW). Head counts cover TP shards
# of a 64-head layer (TP4 -> 16).
_DECODE_DOTS3_SWA_DISPATCH = frozenset({(8, 576), (16, 576), (32, 576), (64, 576)})

# Prefill instantiation envelope (single cache unless noted).
# DSV3_2-family prefill topk (SG, MG, and swapAB); GLM53_NOPE serves 2176.
_V32_TOPK = {
    _MODEL_TYPE_DSV3_2: 2048,
    _MODEL_TYPE_GLM_NSA: 2048,
    _MODEL_TYPE_GLM53_NOPE: 2176,
}
_SG_HEADS = frozenset({8, 16})  # SG: 16-head CTA, NH=8 zero-padded
_MG_V32_HEADS = frozenset({32, 64, 128})  # MG: 32-head CTA
_PREFILL_DSV4_TOPKS = frozenset({128, 192, 256, 512, 1024, 2048})
_PREFILL_DSV4_HEADS = frozenset({8, 16, 32, 64, 128})
_SWAPAB_HEADS = frozenset({64, 128})  # swapAB fills whole 64-head CTAs
_DUAL_TOPK = 128  # dual-cache MG (DSV4 C4A / C128A layers)


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
    # Single-cache only; GLM53_NOPE is included at topk=2176.
    return (
        model_type in _SWAPAB_MODEL_TYPES
        and not has_extra
        and page_block_size == _PAGE_BLOCK_SIZE
        and topk == _V32_TOPK[model_type]
        and num_heads in _SWAPAB_HEADS
    )


# DOTS3_SWA prefill is SG-only (D_NOPE=1024 does not fit the MG smem layout);
# num_heads > 16 is served by CTA replication.
_DOTS3_SWA_TOPK = 576
_DOTS3_SWA_SG_HEADS = frozenset({8, 16, 32, 64})


def prefill_sg_eligible(
    model_type: int, num_heads: int, topk: int, page_block_size: int, has_extra: bool
) -> bool:
    if model_type == _MODEL_TYPE_DOTS3_SWA:
        return (
            not has_extra
            and page_block_size == _PAGE_BLOCK_SIZE
            and topk == _DOTS3_SWA_TOPK
            and num_heads in _DOTS3_SWA_SG_HEADS
        )
    return (
        model_type in _V32_MODEL_TYPES
        and not has_extra
        and page_block_size == _PAGE_BLOCK_SIZE
        and topk == _V32_TOPK[model_type]
        and num_heads in _SG_HEADS
    )


def prefill_mg_eligible(
    model_type: int, num_heads: int, topk: int, page_block_size: int, has_extra: bool
) -> bool:
    if has_extra or page_block_size != _PAGE_BLOCK_SIZE:
        return False
    if model_type in _V32_MODEL_TYPES:
        return topk == _V32_TOPK[model_type] and num_heads in _MG_V32_HEADS
    if model_type == _MODEL_TYPE_DSV4:
        return topk in _PREFILL_DSV4_TOPKS and num_heads in _PREFILL_DSV4_HEADS
    return False


def prefill_mg_dual_eligible(
    model_type: int, num_heads: int, topk: int, page_block_size: int, has_extra: bool
) -> bool:
    return (
        model_type == _MODEL_TYPE_DSV4
        and has_extra
        and page_block_size == _PAGE_BLOCK_SIZE
        and topk == _DUAL_TOPK
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
    if model_type not in _SWAPAB_MODEL_TYPES:
        raise ValueError(
            "prefill_impl='swapab' requires a V32-family model type "
            f"(dsv3_2, glm_nsa, or glm53_nope); got family={_MODEL_TYPE_TO_FAMILY[model_type]!r}"
        )
    if topk != _V32_TOPK[model_type]:
        raise ValueError(
            f"prefill_impl='swapab' requires topk={_V32_TOPK[model_type]}; got topk={topk}"
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
    if (
        c is None
        and AutoTuner.get().is_tuning_mode
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
        and AutoTuner.get().is_tuning_mode
        and not _cpb.is_crossover_failed(device, family)
        and not _cpb.has_crossover(device, family)
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
