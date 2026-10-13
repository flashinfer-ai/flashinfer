# Copyright (c) 2026 by FlashInfer team.
# Portions adapted from DeepEP (MIT); see LICENSE.deepep.txt and NOTICE.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Conservative, host-side launch policy for the measured pure-RDMA profile.

The selection key is the fixed buffer/graph capacity, which must be identical
across ranks. It is not a rank's current token count. Select before graph
capture and keep the result fixed through replay and each send/receive hook.

The measurements used typed arenas, top-8 routing and hooks on EP4/EP8 with E288. These
describe the performance calibration; hooks and top-8 are not API correctness
restrictions imposed by this selector. The thresholds are conservative choices
from the measured shapes, not a promise for every routing distribution or for
small actual batches within a much larger fixed capacity.

This module only selects configuration. It does not modify a live GinBuffer,
allocate communication resources, or synchronize ranks. Unknown profiles keep
the C++ legacy launch defaults instead of imposing the measured GPU's budget.
"""

_PROFILE = "sm120_rtx6000d_ep8_e288_v1"
_PROFILE_110SM = "sm120_110sm_ep8_e288_h4096_v1"
_PROFILE_110SM_EP4 = "sm120_110sm_ep4_e288_h4096_nvfp4_v2"

# Maximum capacity for coalesced, as (dispatch, combine). The other path uses
# legacy defaults. Keep a margin around near-ties at T16/T32 rather than select
# the lowest latency independently at every sampled point.
_THRESHOLDS = {
    4096: {"bf16": (4, 16), "fp8": (16, 16), "nvfp4": (16, 16)},
    6144: {"bf16": (4, 16), "fp8": (8, 16), "nvfp4": (16, 8)},
    7168: {"bf16": (4, 16), "fp8": (4, 16), "nvfp4": (16, 8)},
}


# The measured RTX PRO 5000 reports this generic CUDA device name. Require
# both SM120 and 110 SMs as well; do not apply its budget to every unnamed GPU.
# FP8 and other hidden sizes have not been calibrated on this device.
_THRESHOLDS_110SM = {4096: {"bf16": (8, 16), "nvfp4": (32, 16)}}
# Same-binary boundary checks found regressions at D32 and a near-tie at C8.
_THRESHOLDS_110SM_EP4 = {4096: {"nvfp4": (16, 4)}}


def _launch(coalesced: bool) -> dict:
    return {
        "mode": "coalesced" if coalesced else "legacy",
        "num_sms": 32 if coalesced else None,
        "num_contexts": 8 if coalesced else None,
    }


def _fallback(reason: str) -> dict:
    return {
        "profile": None,
        "reason": reason,
        "dispatch": _launch(False),
        "combine": _launch(False),
    }


def select_rdma_strategy(
    hidden: int,
    dtype: str,
    capacity: int,
    world_size: int,
    num_experts: int,
    device_name: str,
    device_capability: tuple,
    device_num_sms: int,
) -> dict:
    """Return independent dispatch/combine launch settings for a fixed capacity.

    ``profile=None`` means that both operations use unmodified legacy defaults.
    On the profiled 156-SM GPU, the legacy default budget tiles 288 experts into
    144 CTAs and requests 36 contexts; do not hard-code that budget on other
    devices. ``reason`` describes either the profile mismatch or the applied
    thresholds. Each call returns fresh dictionaries that callers may copy or
    annotate without changing future selections.
    """
    if (
        not isinstance(device_capability, (tuple, list))
        or len(device_capability) != 2
        or any(type(value) is not int for value in device_capability)
        or tuple(device_capability) != (12, 0)
    ):
        return _fallback("unprofiled device capability")
    if type(device_num_sms) is not int:
        return _fallback("unprofiled device SM count")
    if device_name in ("NVIDIA RTX 6000D", "RTX 6000D") and device_num_sms == 156:
        profile, thresholds = _PROFILE, _THRESHOLDS
    elif device_name == "NVIDIA Graphics Device" and device_num_sms == 110:
        profile, thresholds = _PROFILE_110SM, _THRESHOLDS_110SM
    else:
        return _fallback("unprofiled device name/SM count")
    if type(world_size) is not int:
        return _fallback("unprofiled EP size")
    if (
        world_size == 4
        and profile == _PROFILE_110SM
        and type(num_experts) is int
        and num_experts == 288
        and type(hidden) is int
        and hidden == 4096
        and isinstance(dtype, str)
        and dtype == "nvfp4"
    ):
        profile, thresholds = _PROFILE_110SM_EP4, _THRESHOLDS_110SM_EP4
    elif world_size != 8:
        return _fallback("unprofiled EP size")
    if type(num_experts) is not int or num_experts != 288:
        return _fallback("unprofiled expert count")
    if type(hidden) is not int or hidden not in thresholds:
        return _fallback("unprofiled hidden size")
    if not isinstance(dtype, str) or dtype not in thresholds[hidden]:
        return _fallback("unprofiled communication dtype")
    if type(capacity) is not int or not 1 <= capacity <= 4096:
        return _fallback("capacity outside the profiled integer range [1, 4096]")

    dispatch_limit, combine_limit = thresholds[hidden][dtype]
    return {
        "profile": profile,
        "reason": (
            f"fixed capacity={capacity}; coalesced limits: "
            f"dispatch<={dispatch_limit}, combine<={combine_limit}"
        ),
        "dispatch": _launch(capacity <= dispatch_limit),
        "combine": _launch(capacity <= combine_limit),
    }
