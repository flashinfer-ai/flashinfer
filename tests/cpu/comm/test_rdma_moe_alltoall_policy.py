# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""CPU-only tests for the conservative fixed-capacity pure-RDMA policy."""

import pytest

from flashinfer.comm.rdma_moe_alltoall_policy import select_rdma_strategy


def select(hidden=4096, dtype="nvfp4", capacity=8, **overrides):
    kwargs = dict(
        hidden=hidden,
        dtype=dtype,
        capacity=capacity,
        world_size=8,
        num_experts=288,
        device_name="NVIDIA RTX 6000D",
        device_capability=(12, 0),
        device_num_sms=156,
    )
    kwargs.update(overrides)
    return select_rdma_strategy(**kwargs)


def assert_thresholds(policy, capacity, dispatch_limit, combine_limit):
    for operation, limit in (("dispatch", dispatch_limit), ("combine", combine_limit)):
        coalesced = capacity <= limit
        assert policy[operation] == {
            "mode": "coalesced" if coalesced else "legacy",
            "num_sms": 32 if coalesced else None,
            "num_contexts": 8 if coalesced else None,
        }


@pytest.mark.parametrize(
    "hidden,dtype,dispatch_limit,combine_limit",
    [
        (4096, "bf16", 4, 16),
        (4096, "fp8", 16, 16),
        (4096, "nvfp4", 16, 16),
        (6144, "bf16", 4, 16),
        (6144, "fp8", 8, 16),
        (6144, "nvfp4", 16, 8),
        (7168, "bf16", 4, 16),
        (7168, "fp8", 4, 16),
        (7168, "nvfp4", 16, 8),
    ],
)
def test_threshold_boundaries(hidden, dtype, dispatch_limit, combine_limit):
    for capacity in sorted(
        {1, dispatch_limit, dispatch_limit + 1, combine_limit, combine_limit + 1, 4096}
    ):
        policy = select(hidden, dtype, capacity)
        assert policy["profile"] == "sm120_rtx6000d_ep8_e288_v1"
        assert_thresholds(policy, capacity, dispatch_limit, combine_limit)


@pytest.mark.parametrize(
    "overrides",
    [
        {"device_name": "NVIDIA RTX 6000 Ada"},
        {"device_capability": (12, 1)},
        {"device_num_sms": 144},
        {"world_size": 4},
        {"num_experts": 256},
        {"hidden": 8192},
        {"dtype": "float16"},
        {"capacity": 0},
        {"capacity": 4097},
        {"capacity": True},
        {"device_capability": (12.0, 0)},
        {"world_size": 8.0},
    ],
)
def test_unknown_profiles_preserve_legacy_defaults(overrides):
    policy = select(**overrides)
    assert policy["profile"] is None
    assert policy["reason"]
    assert (
        policy["dispatch"]
        == policy["combine"]
        == {"mode": "legacy", "num_sms": None, "num_contexts": None}
    )


def test_policy_uses_shared_capacity_and_returns_independent_settings():
    first = select(hidden=6144, dtype="nvfp4", capacity=16)
    assert first["dispatch"]["mode"] == "coalesced"
    assert first["combine"]["mode"] == "legacy"
    first["dispatch"]["mode"] = "modified"
    first["combine"]["num_sms"] = 1
    fresh = select(hidden=6144, dtype="nvfp4", capacity=16)
    assert fresh["dispatch"]["mode"] == "coalesced"
    assert fresh["combine"]["num_sms"] is None


@pytest.mark.parametrize(
    "dtype,dispatch_limit,combine_limit", [("bf16", 8, 16), ("nvfp4", 32, 16)]
)
def test_110sm_capacity_boundaries(dtype, dispatch_limit, combine_limit):
    for capacity in (1, 8, 9, 16, 17, 32, 33, 4096):
        policy = select(
            dtype=dtype,
            capacity=capacity,
            device_name="NVIDIA Graphics Device",
            device_num_sms=110,
        )
        assert policy["profile"] == "sm120_110sm_ep8_e288_h4096_v1"
        assert_thresholds(policy, capacity, dispatch_limit, combine_limit)


@pytest.mark.parametrize(
    "overrides",
    [
        {"device_num_sms": 156},
        {"device_name": "NVIDIA RTX 6000D"},
        {"hidden": 6144},
        {"dtype": "fp8"},
        {"device_capability": (12, 1)},
    ],
)
def test_110sm_does_not_generalize_to_unmeasured_profiles(overrides):
    args = dict(device_name="NVIDIA Graphics Device", device_num_sms=110)
    args.update(overrides)
    policy = select(**args)
    assert policy["profile"] is None
    assert policy["dispatch"]["mode"] == policy["combine"]["mode"] == "legacy"


@pytest.mark.parametrize("capacity", (1, 4, 5, 8, 9, 16, 17, 32, 33, 4096))
def test_110sm_ep4_nvfp4_capacity_boundaries(capacity):
    policy = select(
        capacity=capacity,
        world_size=4,
        device_name="NVIDIA Graphics Device",
        device_num_sms=110,
    )
    assert policy["profile"] == "sm120_110sm_ep4_e288_h4096_nvfp4_v2"
    assert_thresholds(policy, capacity, 16, 4)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dtype": "bf16"},
        {"dtype": "fp8"},
        {"hidden": 6144},
        {"hidden": 7168},
        {"hidden": 4096.0},
        {"num_experts": 256},
        {"num_experts": 288.0},
        {"device_num_sms": 156},
        {"device_num_sms": 110.0},
        {"device_name": "NVIDIA RTX 6000D", "device_num_sms": 156},
        {"device_name": "NVIDIA RTX 6000D"},
        {"device_capability": (12, 1)},
        {"device_capability": (12.0, 0)},
        {"world_size": 4.0},
        {"world_size": 2},
        {"capacity": 0},
        {"capacity": 4097},
        {"capacity": True},
    ],
)
def test_110sm_ep4_profile_is_nvfp4_only(overrides):
    args = dict(world_size=4, device_name="NVIDIA Graphics Device", device_num_sms=110)
    args.update(overrides)
    policy = select(**args)
    assert policy["profile"] is None
    assert (
        policy["dispatch"]
        == policy["combine"]
        == {"mode": "legacy", "num_sms": None, "num_contexts": None}
    )
