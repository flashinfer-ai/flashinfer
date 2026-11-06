# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit NT4 local-expert API for GB300 NVLS composition (experimental)."""

from ..api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def pack_nt4_mxfp8_weights(fc1, fc1_scale, fc2, fc2_scale):
    """Pack 128 local experts without requantization, before graph capture.

    FC1/FC2 are contiguous CUDA E4M3 tensors [128,4096,768]/[128,768,4096].
    Scales are raw E8M0 bytes [128,4096,24]/[128,768,128], not swizzled.
    The returned immutable layout descriptor can be shared across bucket plans.
    """
    from ..experimental.nt4_mxfp8.plan import pack_weights

    return pack_weights(fc1, fc1_scale, fc2, fc2_scale)


@flashinfer_experimental_api
def prepare_nt4_mxfp8_local_experts(
    weights,
    *,
    max_tokens,
    local_expert_offset,
    workers=152,
    group_hint=1024,
    hardware_tanh=True,
):
    """Compile and allocate a native fixed-bucket NT4 local-expert plan.

    The plan owns fused staging, FC1/activation/FC2, lookup scheduling, detached
    cleanup and FP32 weighted assembly. ``run(x, ids, probs, out=partial)`` writes
    directly into the caller's NVLS buffer, on the caller's current stream.
    No distributed initialization or collective is performed by this API.

    Targets SM103, hidden 768, intermediate 4096, top-k 10, 128 local experts.
    ``max_tokens`` is the gathered bucket capacity, not the rank-local capacity.
    ``local_expert_offset`` is 0/128/256/384. Hardware tanh is part of the winning
    configuration; False is provided only for numerical attribution controls.
    Experimental numerical and end-to-end performance qualification is pending.
    """
    from ..experimental.nt4_mxfp8.plan import LocalExpertPlan

    return LocalExpertPlan(
        weights, max_tokens, local_expert_offset, workers, group_hint, hardware_tanh
    )
