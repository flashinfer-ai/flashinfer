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

"""Tests for MNNVL CuTe DSL backend routing."""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="requires nvidia-cutlass-dsl")

from flashinfer.comm.mnnvl_cutedsl import (
    BT_ONLY_CONFIG,
    DEFAULT_CONFIG,
    HT_ONLY_CONFIG,
    LL_ONLY_CONFIG,
)
from flashinfer.comm.mnnvl_cutedsl.config import (
    KernelTarget,
    MNNVLCuteDSLConfig,
    MRangeDispatch,
    ProtocolKind,
    StaticProfile,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_bt import (
    BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_0,
    BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_1,
    BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0,
    BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1,
    BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_0,
    BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_1,
    BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0,
    BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1,
)


def _target(protocol: ProtocolKind, preset: str) -> KernelTarget[str]:
    return KernelTarget(protocol=protocol, preset=preset)


def test_static_profile_rejects_hidden_size_without_vector_alignment():
    routes = MRangeDispatch(
        upper_bounds=(None,),
        targets=(_target(ProtocolKind.LL, "ll"),),
    )
    with pytest.raises(ValueError, match="multiple of 8"):
        StaticProfile(
            tp_size=8,
            hidden_size=8191,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=routes,
            all_reduce_routes=routes,
        )


def test_m_range_dispatch_selects_contiguous_ranges():
    dispatch = MRangeDispatch(
        upper_bounds=(4, 64, None),
        targets=(
            _target(ProtocolKind.LL, "ll-small"),
            _target(ProtocolKind.BT, "bt"),
            _target(ProtocolKind.HT, "ht"),
        ),
    )

    assert dispatch.select(1).preset == "ll-small"
    assert dispatch.select(4).preset == "ll-small"
    assert dispatch.select(5).preset == "bt"
    assert dispatch.select(64).preset == "bt"
    assert dispatch.select(65).preset == "ht"
    assert dispatch.select(8192).preset == "ht"


@pytest.mark.parametrize(
    "upper_bounds",
    [(), (0,), (8, 8), (None, 8)],
)
def test_m_range_dispatch_rejects_invalid_ranges(upper_bounds):
    targets = tuple(
        _target(ProtocolKind.LL, f"preset-{index}")
        for index in range(len(upper_bounds))
    )
    with pytest.raises(ValueError):
        MRangeDispatch(upper_bounds=upper_bounds, targets=targets)


def test_config_derives_protocol_capacities():
    profile = StaticProfile(
        tp_size=8,
        hidden_size=8192,
        top_k=10,
        dtype=torch.bfloat16,
        finalize_routes=MRangeDispatch(
            upper_bounds=(4, 1024, None),
            targets=(
                _target(ProtocolKind.LL, "ll"),
                _target(ProtocolKind.BT, "bt"),
                _target(ProtocolKind.HT, "ht"),
            ),
        ),
        all_reduce_routes=MRangeDispatch(
            upper_bounds=(8, 512, None),
            targets=(
                _target(ProtocolKind.LL, "ll-shared"),
                _target(ProtocolKind.BT, "bt-shared"),
                _target(ProtocolKind.HT, "ht-shared"),
            ),
        ),
    )
    config = MNNVLCuteDSLConfig(profiles=(profile,))

    resolved = config.resolve(
        tp_size=8,
        hidden_size=8192,
        top_k=10,
        dtype=torch.bfloat16,
        capacity_m=8192,
    )
    assert resolved.protocol_capacity(ProtocolKind.LL, capacity_m=8192) == 8
    assert resolved.protocol_capacity(ProtocolKind.BT, capacity_m=8192) == 1024
    assert resolved.protocol_capacity(ProtocolKind.HT, capacity_m=8192) == 8192


def test_static_profile_supports_protocol_internal_preset_switch():
    profile = StaticProfile(
        tp_size=8,
        hidden_size=8192,
        top_k=10,
        dtype=torch.bfloat16,
        finalize_routes=MRangeDispatch(
            upper_bounds=(8, None),
            targets=(
                _target(ProtocolKind.LL, "preset-0"),
                _target(ProtocolKind.LL, "preset-1"),
            ),
        ),
        all_reduce_routes=MRangeDispatch(
            upper_bounds=(None,),
            targets=(_target(ProtocolKind.LL, "preset-0"),),
        ),
    )

    assert profile.finalize_routes.select(8).preset == "preset-0"
    assert profile.finalize_routes.select(9).preset == "preset-1"


def test_bounded_only_config_rejects_larger_capacity():
    bt = _target(ProtocolKind.BT, "bt")
    profile = StaticProfile(
        tp_size=8,
        hidden_size=8192,
        top_k=10,
        dtype=torch.bfloat16,
        finalize_routes=MRangeDispatch(upper_bounds=(1024,), targets=(bt,)),
        all_reduce_routes=MRangeDispatch(upper_bounds=(1024,), targets=(bt,)),
    )
    config = MNNVLCuteDSLConfig(profiles=(profile,))

    config.resolve(
        tp_size=8,
        hidden_size=8192,
        top_k=10,
        dtype=torch.bfloat16,
        capacity_m=1024,
    )
    with pytest.raises(ValueError):
        config.resolve(
            tp_size=8,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            capacity_m=1025,
        )


def test_default_protocol_and_preset_boundaries():
    expected = {
        8: {
            "finalize": (
                (23, ProtocolKind.LL, None),
                (24, ProtocolKind.BT, BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0),
                (48, ProtocolKind.BT, BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0),
                (49, ProtocolKind.BT, BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1),
                (703, ProtocolKind.BT, BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1),
                (704, ProtocolKind.HT, None),
            ),
            "all_reduce": (
                (15, ProtocolKind.LL, None),
                (16, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0),
                (256, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0),
                (257, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1),
                (1024, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1),
                (1025, ProtocolKind.HT, None),
            ),
        },
        16: {
            "finalize": (
                (7, ProtocolKind.LL, None),
                (8, ProtocolKind.BT, BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_0),
                (52, ProtocolKind.BT, BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_0),
                (53, ProtocolKind.BT, BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_1),
                (703, ProtocolKind.BT, BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_1),
                (704, ProtocolKind.HT, None),
            ),
            "all_reduce": (
                (5, ProtocolKind.LL, None),
                (6, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_0),
                (512, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_0),
                (513, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_1),
                (959, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_1),
                (960, ProtocolKind.HT, None),
            ),
        },
    }

    for tp_size, routes in expected.items():
        profile = DEFAULT_CONFIG.resolve(
            tp_size=tp_size,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            capacity_m=8192,
        )
        for path, route in (
            ("finalize", profile.finalize_routes),
            ("all_reduce", profile.all_reduce_routes),
        ):
            for m, protocol, expected_preset in routes[path]:
                target = route.select(m)
                assert target.protocol is protocol
                if expected_preset is not None:
                    assert target.preset is expected_preset


@pytest.mark.parametrize("tp_size", [8, 16])
def test_default_finalize_presets_use_safe_shared_expert_ordering(tp_size):
    profile = DEFAULT_CONFIG.resolve(
        tp_size=tp_size,
        hidden_size=8192,
        top_k=10,
        dtype=torch.bfloat16,
        capacity_m=8192,
    )

    for target in profile.finalize_routes.targets:
        if target.protocol in (ProtocolKind.LL, ProtocolKind.BT):
            assert target.preset.load_shared_expert_before_pdl is False


@pytest.mark.parametrize(
    "config,expected_protocol,capacity_m",
    [
        (LL_ONLY_CONFIG, ProtocolKind.LL, 8192),
        (BT_ONLY_CONFIG, ProtocolKind.BT, 1024),
        (HT_ONLY_CONFIG, ProtocolKind.HT, 8192),
    ],
)
@pytest.mark.parametrize("tp_size", [8, 16])
def test_only_configs_select_one_protocol(
    config, expected_protocol, capacity_m, tp_size
):
    profile = config.resolve(
        tp_size=tp_size,
        hidden_size=8192,
        top_k=10,
        dtype=torch.bfloat16,
        capacity_m=capacity_m,
    )

    for routes in (profile.finalize_routes, profile.all_reduce_routes):
        assert all(target.protocol is expected_protocol for target in routes.targets)
