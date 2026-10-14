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

"""Built-in routing configurations for the MNNVL CuTe DSL backend."""

from collections.abc import Callable

import torch

from .config import (
    KernelTarget,
    MNNVLCuteDSLConfig,
    MRangeDispatch,
    ProtocolKind,
    StaticProfile,
)
from .kernel_bt import (
    BT_ALL_REDUCE_GB300_TP4_H5120_PRESET_0,
    BT_ALL_REDUCE_GB300_TP4_H5120_PRESET_1,
    BT_ALL_REDUCE_GB300_TP8_H5120_PRESET_0,
    BT_ALL_REDUCE_GB300_TP8_H5120_PRESET_1,
    BT_FINALIZE_GB300_TP4_H5120_K3_PRESET_0,
    BT_FINALIZE_GB300_TP4_H5120_K3_PRESET_1,
    BT_FINALIZE_GB300_TP4_H5120_K6_PRESET_0,
    BT_FINALIZE_GB300_TP4_H5120_K6_PRESET_1,
    BT_FINALIZE_GB300_TP8_H5120_K3_PRESET_0,
    BT_FINALIZE_GB300_TP8_H5120_K3_PRESET_1,
    BT_FINALIZE_GB300_TP8_H5120_K6_PRESET_0,
    BT_FINALIZE_GB300_TP8_H5120_K6_PRESET_1,
    BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_0,
    BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_1,
    BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0,
    BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1,
    BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_0,
    BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_1,
    BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0,
    BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1,
    BT_ALL_REDUCE_B300_TP4_H6144_PRESET_0,
    BT_ALL_REDUCE_B300_TP4_H6144_PRESET_1,
    BT_ALL_REDUCE_B300_TP8_H6144_PRESET_0,
    BT_ALL_REDUCE_B300_TP8_H6144_PRESET_1,
    BT_FINALIZE_B300_TP4_H6144_K8_PRESET_0,
    BT_FINALIZE_B300_TP4_H6144_K8_PRESET_1,
    BT_FINALIZE_B300_TP8_H6144_K8_PRESET_0,
    BT_FINALIZE_B300_TP8_H6144_K8_PRESET_1,
)
from .kernel_ll import (
    LL_ALL_REDUCE_GB300_TP4_H5120,
    LL_ALL_REDUCE_GB300_TP8_H5120,
    LL_FINALIZE_GB300_TP4_H5120_K3,
    LL_FINALIZE_GB300_TP4_H5120_K6,
    LL_FINALIZE_GB300_TP8_H5120_K3,
    LL_FINALIZE_GB300_TP8_H5120_K6,
    LL_ALL_REDUCE_GB300_TP16_H8192,
    LL_ALL_REDUCE_GB300_TP8_H8192,
    LL_FINALIZE_GB300_TP16_H8192_K10,
    LL_FINALIZE_GB300_TP8_H8192_K10,
    LL_FINALIZE_B300_TP4_H6144_K8,
    LL_FINALIZE_B300_TP8_H6144_K8,
    LL_ALL_REDUCE_B300_TP4_H6144,
    LL_ALL_REDUCE_B300_TP8_H6144,
)
from .kernel_ht import (
    HT_ALL_REDUCE_GB300_H3584,
    HT_FINALIZE_GB300_H3584_K16,
    HT_ALL_REDUCE_GB300_TP4_H5120,
    HT_FINALIZE_GB300_TP4_H5120_K3,
    HT_FINALIZE_GB300_TP4_H5120_K6,
    HT_ALL_REDUCE_GB300_TP16_H8192,
    HT_ALL_REDUCE_GB300_TP8_H8192,
    HT_FINALIZE_GB300_TP16_H8192_K10,
    HT_FINALIZE_GB300_TP8_H8192_K10,
    HT_ALL_REDUCE_B300_TP4_H6144,
    HT_ALL_REDUCE_B300_TP8_H6144,
    HT_FINALIZE_B300_TP4_H6144_K8,
    HT_FINALIZE_B300_TP8_H6144_K8,
)

__all__ = [
    "BT_ONLY_CONFIG",
    "DEFAULT_CONFIG",
    "HT_ONLY_CONFIG",
    "LL_ONLY_CONFIG",
    "NO_NORM_CONFIG",
]


def _target(protocol: ProtocolKind, preset: object) -> KernelTarget[object]:
    return KernelTarget(protocol=protocol, preset=preset)


# hidden_size 5120, bf16, at top_k 6 and 3. The two top_k values belong to MoE
# stages that share a hidden size and a dtype, so one table generates both.
_H5120_HIDDEN = 5120
_H5120_TOP_K = (6, 3)

# Only tp=4 and tp=8 are published. HT is structurally unreachable at
# hidden=5120 for tp>=8 (see the note in kernel_ht/protocol.py), so the tp=8
# routes below hand their large-M ranges to BT instead of HT.
_H5120_TP_SIZES = (4, 8)


_LL_H5120_FINALIZE = {
    (4, 6): LL_FINALIZE_GB300_TP4_H5120_K6,
    (4, 3): LL_FINALIZE_GB300_TP4_H5120_K3,
    (8, 6): LL_FINALIZE_GB300_TP8_H5120_K6,
    (8, 3): LL_FINALIZE_GB300_TP8_H5120_K3,
}
_LL_H5120_ALL_REDUCE = {
    4: LL_ALL_REDUCE_GB300_TP4_H5120,
    8: LL_ALL_REDUCE_GB300_TP8_H5120,
}
_BT_H5120_FINALIZE = {
    (4, 6): (
        BT_FINALIZE_GB300_TP4_H5120_K6_PRESET_0,
        BT_FINALIZE_GB300_TP4_H5120_K6_PRESET_1,
    ),
    (4, 3): (
        BT_FINALIZE_GB300_TP4_H5120_K3_PRESET_0,
        BT_FINALIZE_GB300_TP4_H5120_K3_PRESET_1,
    ),
    (8, 6): (
        BT_FINALIZE_GB300_TP8_H5120_K6_PRESET_0,
        BT_FINALIZE_GB300_TP8_H5120_K6_PRESET_1,
    ),
    (8, 3): (
        BT_FINALIZE_GB300_TP8_H5120_K3_PRESET_0,
        BT_FINALIZE_GB300_TP8_H5120_K3_PRESET_1,
    ),
}
_BT_H5120_ALL_REDUCE = {
    4: (BT_ALL_REDUCE_GB300_TP4_H5120_PRESET_0, BT_ALL_REDUCE_GB300_TP4_H5120_PRESET_1),
    8: (BT_ALL_REDUCE_GB300_TP8_H5120_PRESET_0, BT_ALL_REDUCE_GB300_TP8_H5120_PRESET_1),
}
_HT_H5120_FINALIZE = {
    (4, 6): HT_FINALIZE_GB300_TP4_H5120_K6,
    (4, 3): HT_FINALIZE_GB300_TP4_H5120_K3,
}
_HT_H5120_ALL_REDUCE = {
    4: HT_ALL_REDUCE_GB300_TP4_H5120,
}

# M-range boundaries for the H5120 routes.
#
# Measured on NVIDIA GB200 with `benchmarks/comm/bench_mnnvl_cutedsl_h5120.py`
# -- tp=4 on one node, tp=8 across two nodes over multi-node NVLink. Each bound
# is the largest swept M at which the lower-M protocol was still ahead.
#
# tp=8 spans two nodes over multi-node NVLink; its crossovers sit at much
# smaller M than tp=4 because LL's per-rank mailbox traffic grows with tp.
_H5120_FINALIZE_LL_MAX = {(4, 6): 42, (4, 3): 38, (8, 6): 24, (8, 3): 24}
_H5120_ALL_REDUCE_LL_MAX = {4: 36, 8: 20}
# BT PRESET_0 -> PRESET_1. The two patterns cross over in very different
# places -- the finalize kernel leaves the narrow 2-element/256-thread tiling as
# soon as the gather dominates, while the plain all-reduce stays on it four
# times longer -- so, as in the H8192 profiles, each pattern carries its own
# split rather than sharing one number.
_H5120_FINALIZE_BT_SPLIT = {(4, 6): 56, (4, 3): 40, (8, 6): 64, (8, 3): 32}
_H5120_ALL_REDUCE_BT_SPLIT = {4: 192, 8: 512}
# BT -> HT, tp=4 only: HT is structurally unreachable at hidden=5120 for tp>=8
# (see the note in kernel_ht/protocol.py), so the tp=8 routes stay on BT all the
# way up. The finalize crossover is strongly top_k dependent -- HT hides the
# top_k gather in its producer warps, so its edge over BT arrives far earlier at
# k=6 (1024) than at k=3 (2048) -- which is why this one is keyed by (tp, k).
_H5120_FINALIZE_BT_MAX = {(4, 6): 1024, (4, 3): 2048}
_H5120_ALL_REDUCE_BT_MAX = {4: 2048}


def _h5120_ll_only_finalize(tp: int, k: int) -> MRangeDispatch:
    return MRangeDispatch(
        upper_bounds=(None,),
        targets=(_target(ProtocolKind.LL, _LL_H5120_FINALIZE[(tp, k)]),),
    )


def _h5120_ll_only_all_reduce(tp: int) -> MRangeDispatch:
    return MRangeDispatch(
        upper_bounds=(None,),
        targets=(_target(ProtocolKind.LL, _LL_H5120_ALL_REDUCE[tp]),),
    )


def _h5120_bt_only_finalize(tp: int, k: int) -> MRangeDispatch:
    preset_0, preset_1 = _BT_H5120_FINALIZE[(tp, k)]
    return MRangeDispatch(
        upper_bounds=(_H5120_FINALIZE_BT_SPLIT[(tp, k)], None),
        targets=(
            _target(ProtocolKind.BT, preset_0),
            _target(ProtocolKind.BT, preset_1),
        ),
    )


def _h5120_bt_only_all_reduce(tp: int) -> MRangeDispatch:
    preset_0, preset_1 = _BT_H5120_ALL_REDUCE[tp]
    return MRangeDispatch(
        upper_bounds=(_H5120_ALL_REDUCE_BT_SPLIT[tp], None),
        targets=(
            _target(ProtocolKind.BT, preset_0),
            _target(ProtocolKind.BT, preset_1),
        ),
    )


def _h5120_ht_only_finalize(tp: int, k: int) -> MRangeDispatch:
    return MRangeDispatch(
        upper_bounds=(None,),
        targets=(_target(ProtocolKind.HT, _HT_H5120_FINALIZE[(tp, k)]),),
    )


def _h5120_ht_only_all_reduce(tp: int) -> MRangeDispatch:
    return MRangeDispatch(
        upper_bounds=(None,),
        targets=(_target(ProtocolKind.HT, _HT_H5120_ALL_REDUCE[tp]),),
    )


def _h5120_default_finalize(tp: int, k: int) -> MRangeDispatch:
    preset_0, preset_1 = _BT_H5120_FINALIZE[(tp, k)]
    bounds: tuple[int | None, ...] = (
        _H5120_FINALIZE_LL_MAX[(tp, k)],
        _H5120_FINALIZE_BT_SPLIT[(tp, k)],
    )
    targets: tuple[KernelTarget[object], ...] = (
        _target(ProtocolKind.LL, _LL_H5120_FINALIZE[(tp, k)]),
        _target(ProtocolKind.BT, preset_0),
    )
    if (tp, k) in _HT_H5120_FINALIZE:
        bounds += (_H5120_FINALIZE_BT_MAX[(tp, k)], None)
        targets += (
            _target(ProtocolKind.BT, preset_1),
            _target(ProtocolKind.HT, _HT_H5120_FINALIZE[(tp, k)]),
        )
    else:
        bounds += (None,)
        targets += (_target(ProtocolKind.BT, preset_1),)
    return MRangeDispatch(upper_bounds=bounds, targets=targets)


def _h5120_default_all_reduce(tp: int) -> MRangeDispatch:
    preset_0, preset_1 = _BT_H5120_ALL_REDUCE[tp]
    bounds: tuple[int | None, ...] = (
        _H5120_ALL_REDUCE_LL_MAX[tp],
        _H5120_ALL_REDUCE_BT_SPLIT[tp],
    )
    targets: tuple[KernelTarget[object], ...] = (
        _target(ProtocolKind.LL, _LL_H5120_ALL_REDUCE[tp]),
        _target(ProtocolKind.BT, preset_0),
    )
    if tp in _HT_H5120_ALL_REDUCE:
        bounds += (_H5120_ALL_REDUCE_BT_MAX[tp], None)
        targets += (
            _target(ProtocolKind.BT, preset_1),
            _target(ProtocolKind.HT, _HT_H5120_ALL_REDUCE[tp]),
        )
    else:
        bounds += (None,)
        targets += (_target(ProtocolKind.BT, preset_1),)
    return MRangeDispatch(upper_bounds=bounds, targets=targets)


# Norm-free boundaries, measured with `apply_rms_norm=False` and
# `add_residual=True`. They differ from the tables above because dropping the
# norm frees LL of a cluster-wide barrier while BT only loses a store, so LL
# stays ahead further (tp=4 top_k=6 finalize: 42 -> 64); turning off
# add_residual shifts them again (-> ~96), so re-measure for that case.
#
# LL -> BT only. HT is structurally unreachable at tp >= 8 here, and at tp = 4
# it has no norm-free kernel yet -- unimplemented, not impossible, and worth
# ~11% at M=2048 and ~25% at M=8192. See kernel_ht/protocol.py.
_H5120_NO_NORM_FINALIZE_LL_MAX = {(4, 6): 64, (4, 3): 56, (8, 6): 28, (8, 3): 28}
_H5120_NO_NORM_ALL_REDUCE_LL_MAX = {4: 40, 8: 28}
_H5120_NO_NORM_FINALIZE_BT_SPLIT = {(4, 6): 64, (4, 3): 32, (8, 6): 96, (8, 3): 48}
_H5120_NO_NORM_ALL_REDUCE_BT_SPLIT = {4: 128, 8: 512}


def _no_norm_routes(ll_preset, bt_presets, ll_max: int, bt_split: int):
    """LL up to `ll_max`, then BT, splitting presets at `bt_split`.

    When the split falls at or below the LL bound (tp=4 finalize), LL already
    covers everything PRESET_0 would win, so BT starts directly on PRESET_1.
    """
    preset_0, preset_1 = bt_presets
    if bt_split <= ll_max:
        return MRangeDispatch(
            upper_bounds=(ll_max, None),
            targets=(
                _target(ProtocolKind.LL, ll_preset),
                _target(ProtocolKind.BT, preset_1),
            ),
        )
    return MRangeDispatch(
        upper_bounds=(ll_max, bt_split, None),
        targets=(
            _target(ProtocolKind.LL, ll_preset),
            _target(ProtocolKind.BT, preset_0),
            _target(ProtocolKind.BT, preset_1),
        ),
    )


def _h5120_no_norm_finalize(tp: int, k: int) -> MRangeDispatch:
    return _no_norm_routes(
        _LL_H5120_FINALIZE[(tp, k)],
        _BT_H5120_FINALIZE[(tp, k)],
        _H5120_NO_NORM_FINALIZE_LL_MAX[(tp, k)],
        _H5120_NO_NORM_FINALIZE_BT_SPLIT[(tp, k)],
    )


def _h5120_no_norm_all_reduce(tp: int) -> MRangeDispatch:
    return _no_norm_routes(
        _LL_H5120_ALL_REDUCE[tp],
        _BT_H5120_ALL_REDUCE[tp],
        _H5120_NO_NORM_ALL_REDUCE_LL_MAX[tp],
        _H5120_NO_NORM_ALL_REDUCE_BT_SPLIT[tp],
    )


def _h5120_profiles(
    finalize_routes: Callable[[int, int], MRangeDispatch],
    all_reduce_routes: Callable[[int], MRangeDispatch],
    tp_sizes: tuple[int, ...] = _H5120_TP_SIZES,
) -> tuple[StaticProfile, ...]:
    """Build one H5120 profile per (tp_size, top_k) pair.

    Both top_k values share a hidden size and a dtype, so their routes are
    identical apart from the finalize preset's prefetch_group. Generating both
    from one callable keeps them from drifting apart as the M-range boundaries
    are retuned.
    """
    return tuple(
        StaticProfile(
            tp_size=tp,
            hidden_size=_H5120_HIDDEN,
            top_k=k,
            dtype=torch.bfloat16,
            finalize_routes=finalize_routes(tp, k),
            all_reduce_routes=all_reduce_routes(tp),
        )
        for tp in tp_sizes
        for k in _H5120_TOP_K
    )


# hidden_size 6144, bf16, at top_k 8 (GLM-5.2: one MoE stage, so one top_k).
# Unlike H5120, HT is reachable at tp=8 here -- 6144 has no factor of 5 to
# strand the HT shard -- so both tp sizes carry HT presets.
_H6144_HIDDEN = 6144
_H6144_TOP_K = (8,)
_H6144_TP_SIZES = (4, 8)

_LL_H6144_FINALIZE = {
    (4, 8): LL_FINALIZE_B300_TP4_H6144_K8,
    (8, 8): LL_FINALIZE_B300_TP8_H6144_K8,
}
_LL_H6144_ALL_REDUCE = {
    4: LL_ALL_REDUCE_B300_TP4_H6144,
    8: LL_ALL_REDUCE_B300_TP8_H6144,
}
_BT_H6144_FINALIZE = {
    (4, 8): (
        BT_FINALIZE_B300_TP4_H6144_K8_PRESET_0,
        BT_FINALIZE_B300_TP4_H6144_K8_PRESET_1,
    ),
    (8, 8): (
        BT_FINALIZE_B300_TP8_H6144_K8_PRESET_0,
        BT_FINALIZE_B300_TP8_H6144_K8_PRESET_1,
    ),
}
_BT_H6144_ALL_REDUCE = {
    4: (BT_ALL_REDUCE_B300_TP4_H6144_PRESET_0, BT_ALL_REDUCE_B300_TP4_H6144_PRESET_1),
    8: (BT_ALL_REDUCE_B300_TP8_H6144_PRESET_0, BT_ALL_REDUCE_B300_TP8_H6144_PRESET_1),
}
_HT_H6144_FINALIZE = {
    (4, 8): HT_FINALIZE_B300_TP4_H6144_K8,
    (8, 8): HT_FINALIZE_B300_TP8_H6144_K8,
}
_HT_H6144_ALL_REDUCE = {
    4: HT_ALL_REDUCE_B300_TP4_H6144,
    8: HT_ALL_REDUCE_B300_TP8_H6144,
}

# M-range boundaries for the H6144 routes.
#
# Measured on 8x NVIDIA B300 SXM6 (single node, NVSwitch, no multi-node
# NVLink) with `benchmarks/comm/bench_mnnvl_cutedsl_h6144.py`, two runs each;
# 8x B200 was swept as a cross-check. Each bound is the largest swept M at
# which the lower-M protocol was still ahead. LL -> BT sat at the same ladder
# step on both GPUs; the finalize BT -> HT edge is the one that moved, arriving
# a step later on B200 (1536 vs 1024).
_H6144_FINALIZE_LL_MAX = {(4, 8): 28, (8, 8): 12}
_H6144_ALL_REDUCE_LL_MAX = {4: 28, 8: 12}
# BT PRESET_0 -> PRESET_1, measured between the two BT presets alone so the
# same split serves BT_ONLY_CONFIG. At top_k=8 the finalize gather outgrows the
# narrow 2-element tiling almost at once; at tp=4 that happens inside LL's
# range, so the default route has no PRESET_0 band there. For the all-reduce
# the presets stay within ~4% of each other past M=256 and trade places
# run to run, so the split sits where PRESET_0 stops winning clearly.
_H6144_FINALIZE_BT_SPLIT = {(4, 8): 20, (8, 8): 20}
_H6144_ALL_REDUCE_BT_SPLIT = {4: 256, 8: 256}
# BT -> HT. Reachable at both tp sizes here, unlike hidden 5120; HT's edge at
# M=4096, tp=8 is ~26% on the finalize pattern and ~24% on the all-reduce.
_H6144_FINALIZE_BT_MAX = {(4, 8): 768, (8, 8): 768}
_H6144_ALL_REDUCE_BT_MAX = {4: 1536, 8: 1024}

# Norm-free boundaries (`apply_rms_norm=False`, `add_residual=True`). LL stays
# ahead further, as at hidden 5120. HT has no norm-free kernel, so these routes
# end on BT even where the norm-on route reaches HT. PRESET_0 never wins the
# norm-free finalize past LL at either tp size (a tie at tp=8, M=20), so those
# splits sit at the LL bound and the route skips it.
_H6144_NO_NORM_FINALIZE_LL_MAX = {(4, 8): 40, (8, 8): 16}
_H6144_NO_NORM_ALL_REDUCE_LL_MAX = {4: 32, 8: 16}
_H6144_NO_NORM_FINALIZE_BT_SPLIT = {(4, 8): 16, (8, 8): 16}
_H6144_NO_NORM_ALL_REDUCE_BT_SPLIT = {4: 256, 8: 256}


def _h6144_ll_only_finalize(tp: int, k: int) -> MRangeDispatch:
    return MRangeDispatch(
        upper_bounds=(None,),
        targets=(_target(ProtocolKind.LL, _LL_H6144_FINALIZE[(tp, k)]),),
    )


def _h6144_ll_only_all_reduce(tp: int) -> MRangeDispatch:
    return MRangeDispatch(
        upper_bounds=(None,),
        targets=(_target(ProtocolKind.LL, _LL_H6144_ALL_REDUCE[tp]),),
    )


def _h6144_bt_only_finalize(tp: int, k: int) -> MRangeDispatch:
    preset_0, preset_1 = _BT_H6144_FINALIZE[(tp, k)]
    return MRangeDispatch(
        upper_bounds=(_H6144_FINALIZE_BT_SPLIT[(tp, k)], None),
        targets=(
            _target(ProtocolKind.BT, preset_0),
            _target(ProtocolKind.BT, preset_1),
        ),
    )


def _h6144_bt_only_all_reduce(tp: int) -> MRangeDispatch:
    preset_0, preset_1 = _BT_H6144_ALL_REDUCE[tp]
    return MRangeDispatch(
        upper_bounds=(_H6144_ALL_REDUCE_BT_SPLIT[tp], None),
        targets=(
            _target(ProtocolKind.BT, preset_0),
            _target(ProtocolKind.BT, preset_1),
        ),
    )


def _h6144_ht_only_finalize(tp: int, k: int) -> MRangeDispatch:
    return MRangeDispatch(
        upper_bounds=(None,),
        targets=(_target(ProtocolKind.HT, _HT_H6144_FINALIZE[(tp, k)]),),
    )


def _h6144_ht_only_all_reduce(tp: int) -> MRangeDispatch:
    return MRangeDispatch(
        upper_bounds=(None,),
        targets=(_target(ProtocolKind.HT, _HT_H6144_ALL_REDUCE[tp]),),
    )


def _h6144_default_routes(
    ll_preset, bt_presets, ht_preset, ll_max: int, bt_split: int, bt_max: int
) -> MRangeDispatch:
    """LL, then BT (split at `bt_split`), then HT above `bt_max`.

    As in `_no_norm_routes`, a split at or below the LL bound means LL already
    covers everything PRESET_0 would win, so BT starts on PRESET_1.
    """
    preset_0, preset_1 = bt_presets
    bounds: tuple[int | None, ...] = (ll_max,)
    targets: tuple[KernelTarget[object], ...] = (_target(ProtocolKind.LL, ll_preset),)
    if bt_split > ll_max:
        bounds += (bt_split,)
        targets += (_target(ProtocolKind.BT, preset_0),)
    bounds += (bt_max, None)
    targets += (
        _target(ProtocolKind.BT, preset_1),
        _target(ProtocolKind.HT, ht_preset),
    )
    return MRangeDispatch(upper_bounds=bounds, targets=targets)


def _h6144_default_finalize(tp: int, k: int) -> MRangeDispatch:
    return _h6144_default_routes(
        _LL_H6144_FINALIZE[(tp, k)],
        _BT_H6144_FINALIZE[(tp, k)],
        _HT_H6144_FINALIZE[(tp, k)],
        _H6144_FINALIZE_LL_MAX[(tp, k)],
        _H6144_FINALIZE_BT_SPLIT[(tp, k)],
        _H6144_FINALIZE_BT_MAX[(tp, k)],
    )


def _h6144_default_all_reduce(tp: int) -> MRangeDispatch:
    return _h6144_default_routes(
        _LL_H6144_ALL_REDUCE[tp],
        _BT_H6144_ALL_REDUCE[tp],
        _HT_H6144_ALL_REDUCE[tp],
        _H6144_ALL_REDUCE_LL_MAX[tp],
        _H6144_ALL_REDUCE_BT_SPLIT[tp],
        _H6144_ALL_REDUCE_BT_MAX[tp],
    )


def _h6144_no_norm_finalize(tp: int, k: int) -> MRangeDispatch:
    return _no_norm_routes(
        _LL_H6144_FINALIZE[(tp, k)],
        _BT_H6144_FINALIZE[(tp, k)],
        _H6144_NO_NORM_FINALIZE_LL_MAX[(tp, k)],
        _H6144_NO_NORM_FINALIZE_BT_SPLIT[(tp, k)],
    )


def _h6144_no_norm_all_reduce(tp: int) -> MRangeDispatch:
    return _no_norm_routes(
        _LL_H6144_ALL_REDUCE[tp],
        _BT_H6144_ALL_REDUCE[tp],
        _H6144_NO_NORM_ALL_REDUCE_LL_MAX[tp],
        _H6144_NO_NORM_ALL_REDUCE_BT_SPLIT[tp],
    )


def _h6144_profiles(
    finalize_routes: Callable[[int, int], MRangeDispatch],
    all_reduce_routes: Callable[[int], MRangeDispatch],
    tp_sizes: tuple[int, ...] = _H6144_TP_SIZES,
) -> tuple[StaticProfile, ...]:
    """Build one H6144 profile per (tp_size, top_k) pair; see _h5120_profiles."""
    return tuple(
        StaticProfile(
            tp_size=tp,
            hidden_size=_H6144_HIDDEN,
            top_k=k,
            dtype=torch.bfloat16,
            finalize_routes=finalize_routes(tp, k),
            all_reduce_routes=all_reduce_routes(tp),
        )
        for tp in tp_sizes
        for k in _H6144_TOP_K
    )


LL_ONLY_CONFIG = MNNVLCuteDSLConfig(
    profiles=(
        StaticProfile(
            tp_size=8,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_FINALIZE_GB300_TP8_H8192_K10,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_ALL_REDUCE_GB300_TP8_H8192,
                    ),
                ),
            ),
        ),
        StaticProfile(
            tp_size=16,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_FINALIZE_GB300_TP16_H8192_K10,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_ALL_REDUCE_GB300_TP16_H8192,
                    ),
                ),
            ),
        ),
        *_h5120_profiles(_h5120_ll_only_finalize, _h5120_ll_only_all_reduce),
        *_h6144_profiles(_h6144_ll_only_finalize, _h6144_ll_only_all_reduce),
    ),
    # Protocol-pinned: these ranges force one protocol rather than describe a
    # crossover, so the config suits either apply_rms_norm setting.
    applies_rms_norm=None,
)


BT_ONLY_CONFIG = MNNVLCuteDSLConfig(
    profiles=(
        StaticProfile(
            tp_size=8,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(48, 1024),
                targets=(
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(256, 1024),
                targets=(
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1,
                    ),
                ),
            ),
        ),
        StaticProfile(
            tp_size=16,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(52, 1024),
                targets=(
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_1,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(512, 1024),
                targets=(
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_1,
                    ),
                ),
            ),
        ),
        *_h5120_profiles(_h5120_bt_only_finalize, _h5120_bt_only_all_reduce),
        *_h6144_profiles(_h6144_bt_only_finalize, _h6144_bt_only_all_reduce),
    ),
    # Protocol-pinned: these ranges force one protocol rather than describe a
    # crossover, so the config suits either apply_rms_norm setting.
    applies_rms_norm=None,
)


HT_ONLY_CONFIG = MNNVLCuteDSLConfig(
    profiles=(
        StaticProfile(
            tp_size=4,
            hidden_size=3584,
            top_k=16,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(_target(ProtocolKind.HT, HT_FINALIZE_GB300_H3584_K16),),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(_target(ProtocolKind.HT, HT_ALL_REDUCE_GB300_H3584),),
            ),
        ),
        StaticProfile(
            tp_size=8,
            hidden_size=3584,
            top_k=16,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(_target(ProtocolKind.HT, HT_FINALIZE_GB300_H3584_K16),),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(_target(ProtocolKind.HT, HT_ALL_REDUCE_GB300_H3584),),
            ),
        ),
        StaticProfile(
            tp_size=16,
            hidden_size=3584,
            top_k=16,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(_target(ProtocolKind.HT, HT_FINALIZE_GB300_H3584_K16),),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(_target(ProtocolKind.HT, HT_ALL_REDUCE_GB300_H3584),),
            ),
        ),
        StaticProfile(
            tp_size=8,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.HT,
                        HT_FINALIZE_GB300_TP8_H8192_K10,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.HT,
                        HT_ALL_REDUCE_GB300_TP8_H8192,
                    ),
                ),
            ),
        ),
        StaticProfile(
            tp_size=16,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.HT,
                        HT_FINALIZE_GB300_TP16_H8192_K10,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.HT,
                        HT_ALL_REDUCE_GB300_TP16_H8192,
                    ),
                ),
            ),
        ),
        *_h5120_profiles(
            _h5120_ht_only_finalize, _h5120_ht_only_all_reduce, tp_sizes=(4,)
        ),
        *_h6144_profiles(_h6144_ht_only_finalize, _h6144_ht_only_all_reduce),
    ),
    # Protocol-pinned: these ranges force one protocol rather than describe a
    # crossover, so the config suits either apply_rms_norm setting.
    applies_rms_norm=None,
)


DEFAULT_CONFIG = MNNVLCuteDSLConfig(
    profiles=(
        StaticProfile(
            tp_size=8,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(23, 48, 703, None),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_FINALIZE_GB300_TP8_H8192_K10,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1,
                    ),
                    _target(
                        ProtocolKind.HT,
                        HT_FINALIZE_GB300_TP8_H8192_K10,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(15, 256, 1024, None),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_ALL_REDUCE_GB300_TP8_H8192,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1,
                    ),
                    _target(
                        ProtocolKind.HT,
                        HT_ALL_REDUCE_GB300_TP8_H8192,
                    ),
                ),
            ),
        ),
        StaticProfile(
            tp_size=16,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(7, 52, 703, None),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_FINALIZE_GB300_TP16_H8192_K10,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_1,
                    ),
                    _target(
                        ProtocolKind.HT,
                        HT_FINALIZE_GB300_TP16_H8192_K10,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(5, 512, 959, None),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_ALL_REDUCE_GB300_TP16_H8192,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_1,
                    ),
                    _target(
                        ProtocolKind.HT,
                        HT_ALL_REDUCE_GB300_TP16_H8192,
                    ),
                ),
            ),
        ),
        *_h5120_profiles(_h5120_default_finalize, _h5120_default_all_reduce),
        *_h6144_profiles(_h6144_default_finalize, _h6144_default_all_reduce),
    )
)


NO_NORM_CONFIG = MNNVLCuteDSLConfig(
    profiles=(
        *_h5120_profiles(_h5120_no_norm_finalize, _h5120_no_norm_all_reduce),
        *_h6144_profiles(_h6144_no_norm_finalize, _h6144_no_norm_all_reduce),
    ),
    applies_rms_norm=False,
)
