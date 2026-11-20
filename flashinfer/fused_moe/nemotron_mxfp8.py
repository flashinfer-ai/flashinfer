# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit Nemotron local-expert API for GB300 NVLS composition (experimental)."""

from ..api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def pack_nemotron_mxfp8_weights(fc1, fc1_scale, fc2, fc2_scale):
    """Pack 128 local experts without requantization, before graph capture.

    FC1/FC2 are contiguous CUDA E4M3 tensors [128,I,H]/[128,H,I].
    Supported (H,I) pairs are (768,4096) and (1536,7168). Scales are raw
    E8M0 bytes [128,I,H/32]/[128,H,I/32], not swizzled.
    The returned immutable layout descriptor can be shared across bucket plans.
    """
    from ..experimental.nemotron_mxfp8.plan import pack_weights

    return pack_weights(fc1, fc1_scale, fc2, fc2_scale)


@flashinfer_experimental_api
def prepare_nemotron_mxfp8_local_experts(
    weights,
    *,
    max_tokens,
    local_expert_offset,
    workers=152,
    group_hint=1024,
    hardware_tanh=True,
):
    """Compile and allocate a native fixed-bucket Nemotron local-expert plan.

    The plan owns fused staging, FC1/activation/FC2, lookup scheduling, detached
    cleanup and FP32 weighted assembly. `run(x, ids, probs, out=partial)` writes
    directly into the caller's NVLS buffer, on the caller's current stream.
    No distributed initialization or collective is performed by this API.

    Targets SM103, (hidden, intermediate) = (768, 4096) or (1536, 7168),
    top-k 10, 128 local experts. Dimensions are inferred from packed weights.
    `max_tokens` is the gathered bucket capacity, not the rank-local capacity.
    `local_expert_offset` is 0/128/256/384. Hardware tanh is enabled by default;
    `hardware_tanh=False` selects the software numerical attribution control.
    The two activation choices compile separate plans. See the experimental
    backend README for quality diagnostics and remaining qualification limits.
    """
    from ..experimental.nemotron_mxfp8.plan import LocalExpertPlan

    return LocalExpertPlan(
        weights, max_tokens, local_expert_offset, workers, group_hint, hardware_tanh
    )
