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
    NO_NORM_CONFIG,
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
    BT_ALL_REDUCE_GB300_TP4_H7168_PRESET_0,
    BT_ALL_REDUCE_GB300_TP4_H7168_PRESET_1,
    BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_0,
    BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_1,
    BT_FINALIZE_GB300_TP4_H7168_K16_PRESET_0,
    BT_FINALIZE_GB300_TP4_H7168_K16_PRESET_1,
    BT_FINALIZE_GB300_TP8_H7168_K16_PRESET_0,
    BT_FINALIZE_GB300_TP8_H7168_K16_PRESET_1,
    BT_ALL_REDUCE_GB300_TP4_H8192_FUSED,
    BT_ALL_REDUCE_GB300_TP4_H8192_PRESET_0,
    BT_FINALIZE_GB300_TP4_H8192_K10,
    BT_ALL_REDUCE_RUBIN_TP4_H8192_FUSED,
    BT_ALL_REDUCE_RUBIN_TP4_H8192_PRESET_0,
    BT_FINALIZE_RUBIN_TP4_H8192_K10,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_ht import (
    HT_ALL_REDUCE_GB300_TP4_H7168,
    HT_FINALIZE_GB300_TP4_H7168_K16,
    HT_FINALIZE_GB300_TP4_H8192_K10,
    HT_FINALIZE_RUBIN_TP4_H8192_K10,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_ll import (
    LL_ALL_REDUCE_GB300_TP4_H7168,
    LL_ALL_REDUCE_GB300_TP8_H7168,
    LL_FINALIZE_GB300_TP4_H7168_K16,
    LL_FINALIZE_GB300_TP8_H7168_K16,
    LL_ALL_REDUCE_GB300_TP4_H8192,
    LL_ALL_REDUCE_GB300_TP4_H8192_BULK,
    LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_25_TO_64,
    LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_65_TO_192,
    LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_193_TO_320,
    LL_FINALIZE_GB300_TP4_H8192_K10,
    LL_ALL_REDUCE_RUBIN_TP4_H8192,
    LL_ALL_REDUCE_RUBIN_TP4_H8192_BULK,
    LL_ALL_REDUCE_RUBIN_TP4_H8192_TWO_SHOT,
    LL_FINALIZE_RUBIN_TP4_H8192_K10,
)


def _target(protocol: ProtocolKind, preset: str) -> KernelTarget[str]:
    return KernelTarget(protocol=protocol, preset=preset)


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_kimi_k3_default_protocol_and_preset_boundaries():
    expected = {
        4: {
            "finalize": (
                (24, ProtocolKind.LL, LL_FINALIZE_GB300_TP4_H7168_K16),
                (25, ProtocolKind.BT, BT_FINALIZE_GB300_TP4_H7168_K16_PRESET_0),
                (512, ProtocolKind.BT, BT_FINALIZE_GB300_TP4_H7168_K16_PRESET_0),
                (513, ProtocolKind.BT, BT_FINALIZE_GB300_TP4_H7168_K16_PRESET_1),
                (768, ProtocolKind.BT, BT_FINALIZE_GB300_TP4_H7168_K16_PRESET_1),
                (769, ProtocolKind.HT, HT_FINALIZE_GB300_TP4_H7168_K16),
            ),
            "all_reduce": (
                (28, ProtocolKind.LL, LL_ALL_REDUCE_GB300_TP4_H7168),
                (29, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP4_H7168_PRESET_0),
                (512, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP4_H7168_PRESET_0),
                (513, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP4_H7168_PRESET_1),
                (2048, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP4_H7168_PRESET_1),
                (2049, ProtocolKind.HT, HT_ALL_REDUCE_GB300_TP4_H7168),
            ),
        },
        8: {
            "finalize": (
                (12, ProtocolKind.LL, LL_FINALIZE_GB300_TP8_H7168_K16),
                (13, ProtocolKind.BT, BT_FINALIZE_GB300_TP8_H7168_K16_PRESET_0),
                (512, ProtocolKind.BT, BT_FINALIZE_GB300_TP8_H7168_K16_PRESET_0),
                (513, ProtocolKind.BT, BT_FINALIZE_GB300_TP8_H7168_K16_PRESET_1),
            ),
            "all_reduce": (
                (12, ProtocolKind.LL, LL_ALL_REDUCE_GB300_TP8_H7168),
                (13, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_0),
                (256, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_0),
                (257, ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_1),
            ),
        },
    }

    for tp_size, routes in expected.items():
        profile = DEFAULT_CONFIG.resolve(
            tp_size=tp_size,
            hidden_size=7168,
            top_k=16,
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
                assert target.preset is expected_preset


_LL = ProtocolKind.LL
_BT = ProtocolKind.BT
_HT = ProtocolKind.HT
_RUBIN = (10, 7)
_GB300 = (10, 3)
_TP4_H8192_ROUTES = {
    (_RUBIN, True): (
        (
            (32, _LL, LL_FINALIZE_RUBIN_TP4_H8192_K10),
            (33, _BT, BT_FINALIZE_RUBIN_TP4_H8192_K10),
            (2048, _BT, BT_FINALIZE_RUBIN_TP4_H8192_K10),
            (2049, _HT, HT_FINALIZE_RUBIN_TP4_H8192_K10),
        ),
        (
            (16, _LL, LL_ALL_REDUCE_RUBIN_TP4_H8192),
            (17, _LL, LL_ALL_REDUCE_RUBIN_TP4_H8192_BULK),
            (48, _LL, LL_ALL_REDUCE_RUBIN_TP4_H8192_BULK),
            (49, _BT, BT_ALL_REDUCE_RUBIN_TP4_H8192_PRESET_0),
            (256, _BT, BT_ALL_REDUCE_RUBIN_TP4_H8192_PRESET_0),
            (257, _BT, BT_ALL_REDUCE_RUBIN_TP4_H8192_FUSED),
            (16384, _BT, BT_ALL_REDUCE_RUBIN_TP4_H8192_FUSED),
        ),
    ),
    (_RUBIN, False): (
        (
            (48, _LL, LL_FINALIZE_RUBIN_TP4_H8192_K10),
            (49, _BT, BT_FINALIZE_RUBIN_TP4_H8192_K10),
            (16384, _BT, BT_FINALIZE_RUBIN_TP4_H8192_K10),
        ),
        (
            (16, _LL, LL_ALL_REDUCE_RUBIN_TP4_H8192),
            (17, _LL, LL_ALL_REDUCE_RUBIN_TP4_H8192_BULK),
            (64, _LL, LL_ALL_REDUCE_RUBIN_TP4_H8192_BULK),
            (65, _BT, BT_ALL_REDUCE_RUBIN_TP4_H8192_PRESET_0),
            (128, _BT, BT_ALL_REDUCE_RUBIN_TP4_H8192_PRESET_0),
            (129, _LL, LL_ALL_REDUCE_RUBIN_TP4_H8192_TWO_SHOT),
            (384, _LL, LL_ALL_REDUCE_RUBIN_TP4_H8192_TWO_SHOT),
            (385, _BT, BT_ALL_REDUCE_RUBIN_TP4_H8192_FUSED),
            (16384, _BT, BT_ALL_REDUCE_RUBIN_TP4_H8192_FUSED),
        ),
    ),
    (_GB300, True): (
        (
            (24, _LL, LL_FINALIZE_GB300_TP4_H8192_K10),
            (25, _BT, BT_FINALIZE_GB300_TP4_H8192_K10),
            (768, _BT, BT_FINALIZE_GB300_TP4_H8192_K10),
            (769, _HT, HT_FINALIZE_GB300_TP4_H8192_K10),
            (16384, _HT, HT_FINALIZE_GB300_TP4_H8192_K10),
        ),
        (
            (16, _LL, LL_ALL_REDUCE_GB300_TP4_H8192),
            (17, _BT, BT_ALL_REDUCE_GB300_TP4_H8192_PRESET_0),
            (384, _BT, BT_ALL_REDUCE_GB300_TP4_H8192_PRESET_0),
            (385, _BT, BT_ALL_REDUCE_GB300_TP4_H8192_FUSED),
            (16384, _BT, BT_ALL_REDUCE_GB300_TP4_H8192_FUSED),
        ),
    ),
    (_GB300, False): (
        (
            (32, _LL, LL_FINALIZE_GB300_TP4_H8192_K10),
            (33, _BT, BT_FINALIZE_GB300_TP4_H8192_K10),
            (16384, _BT, BT_FINALIZE_GB300_TP4_H8192_K10),
        ),
        (
            (8, _LL, LL_ALL_REDUCE_GB300_TP4_H8192),
            (9, _LL, LL_ALL_REDUCE_GB300_TP4_H8192_BULK),
            (24, _LL, LL_ALL_REDUCE_GB300_TP4_H8192_BULK),
            (25, _LL, LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_25_TO_64),
            (64, _LL, LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_25_TO_64),
            (65, _LL, LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_65_TO_192),
            (192, _LL, LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_65_TO_192),
            (193, _LL, LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_193_TO_320),
            (320, _LL, LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_193_TO_320),
            (321, _BT, BT_ALL_REDUCE_GB300_TP4_H8192_FUSED),
            (16384, _BT, BT_ALL_REDUCE_GB300_TP4_H8192_FUSED),
        ),
    ),
}


def _tp4_h8192(config, compute_capability):
    return config.resolve(
        tp_size=4,
        hidden_size=8192,
        top_k=10,
        dtype=torch.bfloat16,
        capacity_m=16384,
        compute_capability=compute_capability,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "compute_capability,norm",
    tuple(_TP4_H8192_ROUTES),
    ids=("rubin-norm", "rubin-no-norm", "gb300-norm", "gb300-no-norm"),
)
def test_tp4_h8192_protocol_and_preset_boundaries(compute_capability, norm):
    config = DEFAULT_CONFIG if norm else NO_NORM_CONFIG
    profile = _tp4_h8192(config, compute_capability)
    assert profile.compute_capability == compute_capability
    finalize, all_reduce = _TP4_H8192_ROUTES[(compute_capability, norm)]
    for route, expected in (
        (profile.finalize_routes, finalize),
        (profile.all_reduce_routes, all_reduce),
    ):
        for m, protocol, preset in expected:
            target = route.select(m)
            assert target.protocol is protocol
            assert target.preset is preset


@pytest.mark.cpu_only
@pytest.mark.parametrize("compute_capability", (_RUBIN, _GB300))
def test_tp4_h8192_norm_routes_avoid_two_shot(compute_capability):
    """The norm-on routes also serve dynamic FP8, which two-shot rejects."""
    profile = _tp4_h8192(DEFAULT_CONFIG, compute_capability)
    for target in profile.all_reduce_routes.targets:
        assert not getattr(target.preset, "two_shot", False)


@pytest.mark.cpu_only
@pytest.mark.parametrize("compute_capability", (_RUBIN, _GB300))
def test_tp4_h8192_no_norm_routes_avoid_ht(compute_capability):
    """HT cannot compile the norm out, so norm-free routes never reach it."""
    profile = _tp4_h8192(NO_NORM_CONFIG, compute_capability)
    assert ProtocolKind.HT not in profile.referenced_protocols


@pytest.mark.cpu_only
@pytest.mark.parametrize("compute_capability", (None, (10, 0), (12, 0)))
def test_tp4_h8192_has_no_profile_for_other_architectures(compute_capability):
    for config in (DEFAULT_CONFIG, NO_NORM_CONFIG):
        with pytest.raises(ValueError, match="No MNNVL CuTe DSL profile"):
            _tp4_h8192(config, compute_capability)


def _single_route_profile(preset: str, compute_capability=None) -> StaticProfile:
    route = MRangeDispatch(
        upper_bounds=(None,), targets=(_target(ProtocolKind.LL, preset),)
    )
    return StaticProfile(
        tp_size=4,
        hidden_size=4096,
        top_k=8,
        dtype=torch.bfloat16,
        finalize_routes=route,
        all_reduce_routes=route,
        compute_capability=compute_capability,
    )


def _resolved_preset(config: MNNVLCuteDSLConfig, compute_capability) -> str:
    profile = config.resolve(
        tp_size=4,
        hidden_size=4096,
        top_k=8,
        dtype=torch.bfloat16,
        capacity_m=64,
        compute_capability=compute_capability,
    )
    return profile.all_reduce_routes.select(1).preset


@pytest.mark.cpu_only
@pytest.mark.parametrize("agnostic_first", (True, False))
def test_resolve_prefers_architecture_specific_profile(agnostic_first):
    agnostic = _single_route_profile("any")
    rubin = _single_route_profile("rubin", _RUBIN)
    gb300 = _single_route_profile("gb300", _GB300)
    profiles = (agnostic, rubin, gb300) if agnostic_first else (rubin, gb300, agnostic)
    config = MNNVLCuteDSLConfig(profiles=profiles)
    assert _resolved_preset(config, _RUBIN) == "rubin"
    assert _resolved_preset(config, _GB300) == "gb300"
    assert _resolved_preset(config, (10, 0)) == "any"
    assert _resolved_preset(config, None) == "any"


@pytest.mark.cpu_only
def test_architecture_specific_profile_needs_a_matching_device():
    config = MNNVLCuteDSLConfig(profiles=(_single_route_profile("rubin", _RUBIN),))
    assert _resolved_preset(config, _RUBIN) == "rubin"
    for compute_capability in (None, _GB300):
        with pytest.raises(ValueError, match="No MNNVL CuTe DSL profile"):
            _resolved_preset(config, compute_capability)
        assert (
            config.find(
                tp_size=4,
                hidden_size=4096,
                top_k=8,
                dtype=torch.bfloat16,
                compute_capability=compute_capability,
            )
            is None
        )


@pytest.mark.cpu_only
def test_config_rejects_duplicate_profile_per_architecture():
    MNNVLCuteDSLConfig(
        profiles=(_single_route_profile("a"), _single_route_profile("b", _RUBIN))
    )
    with pytest.raises(ValueError, match="unique static shapes"):
        MNNVLCuteDSLConfig(
            profiles=(
                _single_route_profile("a", _RUBIN),
                _single_route_profile("b", _RUBIN),
            )
        )
    with pytest.raises(ValueError, match="unique static shapes"):
        MNNVLCuteDSLConfig(
            profiles=(_single_route_profile("a"), _single_route_profile("b"))
        )


@pytest.mark.cpu_only
def test_static_profile_rejects_malformed_compute_capability():
    with pytest.raises(ValueError, match="compute_capability"):
        _single_route_profile("a", (10,))  # type: ignore[arg-type]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "config,expected_protocol,tp_sizes",
    [
        (LL_ONLY_CONFIG, ProtocolKind.LL, (4, 8)),
        (BT_ONLY_CONFIG, ProtocolKind.BT, (4, 8)),
        (HT_ONLY_CONFIG, ProtocolKind.HT, (4,)),
    ],
)
def test_kimi_k3_only_configs_select_one_protocol(config, expected_protocol, tp_sizes):
    for tp_size in tp_sizes:
        profile = config.resolve(
            tp_size=tp_size,
            hidden_size=7168,
            top_k=16,
            dtype=torch.bfloat16,
            capacity_m=8192,
        )
        for routes in (profile.finalize_routes, profile.all_reduce_routes):
            assert all(
                target.protocol is expected_protocol for target in routes.targets
            )


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "compute_capability,capacity_m,expected",
    (
        (_RUBIN, 16384, {LL_ALL_REDUCE_RUBIN_TP4_H8192_TWO_SHOT: 384}),
        (_RUBIN, 200, {LL_ALL_REDUCE_RUBIN_TP4_H8192_TWO_SHOT: 200}),
        (_RUBIN, 100, {}),
        (
            _GB300,
            16384,
            {
                LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_25_TO_64: 64,
                LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_65_TO_192: 192,
                LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_193_TO_320: 320,
            },
        ),
        (
            _GB300,
            100,
            {
                LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_25_TO_64: 64,
                LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_65_TO_192: 100,
            },
        ),
    ),
    ids=("rubin", "rubin-capped", "rubin-unreached", "gb300", "gb300-capped"),
)
def test_two_shot_capacities_follow_routes(compute_capability, capacity_m, expected):
    """Each LL two-shot preset is sized to the largest token count routed to it."""
    from flashinfer.comm.mnnvl_cutedsl_ar import _preset_capacities

    profile = _tp4_h8192(NO_NORM_CONFIG, compute_capability)
    capacities = _preset_capacities(
        profile.all_reduce_routes, ProtocolKind.LL, capacity_m
    )
    two_shot = {
        preset: capacity
        for preset, capacity in capacities.items()
        if getattr(preset, "two_shot", False)
    }
    assert two_shot == expected
