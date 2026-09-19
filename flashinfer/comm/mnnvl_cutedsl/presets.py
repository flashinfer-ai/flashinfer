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
)
from .kernel_ht import (
    HT_ALL_REDUCE_GB300_TP4_H5120,
    HT_FINALIZE_GB300_TP4_H5120_K3,
    HT_FINALIZE_GB300_TP4_H5120_K6,
    HT_ALL_REDUCE_GB300_TP16_H8192,
    HT_ALL_REDUCE_GB300_TP8_H8192,
    HT_FINALIZE_GB300_TP16_H8192_K10,
    HT_FINALIZE_GB300_TP8_H8192_K10,
)

__all__ = [
    "BT_ONLY_CONFIG",
    "DEFAULT_CONFIG",
    "HT_ONLY_CONFIG",
    "LL_ONLY_CONFIG",
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
    )
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
    )
)


HT_ONLY_CONFIG = MNNVLCuteDSLConfig(
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
    )
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
    )
)
