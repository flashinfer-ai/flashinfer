"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from typing import Optional

import torch

from ..jit.mamba.ssu_incremental import gen_ssu_incremental_module
from ..utils import register_custom_op  # noqa: F401


@functools.cache
def _get_module(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    state_scale_dtype: Optional[torch.dtype],
    dim: int,
    dstate: int,
    ntokens_mtp: int,
    philox_rounds: int = 0,
):
    return gen_ssu_incremental_module(
        state_dtype,
        input_dtype,
        dt_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        state_scale_dtype,
        dim,
        dstate,
        ntokens_mtp,
        philox_rounds,
    ).build_and_load()


def ssu_incremental(
    state: torch.Tensor,
    old_x: torch.Tensor,
    old_B: torch.Tensor,
    old_dt_proc: torch.Tensor,
    old_cumAdt: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    state_batch_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    state_scale: Optional[torch.Tensor] = None,
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 10,
    d_split: Optional[int] = None,
) -> torch.Tensor:
    """Incremental SSU with MTP replay using matmul-based parallel token processing.

    Parameters
    ----------
    state : torch.Tensor
        SSM state, shape (cache, nheads, dim, dstate). Updated in-place.
    old_x : torch.Tensor
        Cached x from previous step, shape (cache, T, nheads, dim). Single-buffered.
    old_B : torch.Tensor
        Cached B, shape (cache, 2, T, ngroups, dstate). Double-buffered.
    old_dt_proc : torch.Tensor
        Cached processed dt, shape (cache, 2, nheads, T). Double-buffered, f32.
    old_cumAdt : torch.Tensor
        Cached cumulative A*dt, shape (cache, 2, nheads, T). Double-buffered, f32.
    cache_buf_idx : torch.Tensor
        Which buffer to read (0 or 1), shape (cache,), int32.
    prev_num_accepted_tokens : torch.Tensor
        Number of old tokens to replay, shape (cache,), int32.
    x : torch.Tensor
        New token inputs, shape (batch, T, nheads, dim).
    dt : torch.Tensor
        Delta time, shape (batch, T, nheads, dim) with tie_hdim (stride[-1]=0).
        Accepted in native dtype (e.g. bf16) — converted to f32 internally.
    A : torch.Tensor
        Decay rate, shape (nheads, dim, dstate) with tie_hdim.
    B : torch.Tensor
        Input projection, shape (batch, T, ngroups, dstate).
    C : torch.Tensor
        Output projection, shape (batch, T, ngroups, dstate).
    out : torch.Tensor
        Preallocated output, shape (batch, T, nheads, dim).
    D : Optional[torch.Tensor]
        Skip connection, shape (nheads, dim).
    z : Optional[torch.Tensor]
        Gate, shape (batch, T, nheads, dim).
    dt_bias : Optional[torch.Tensor]
        Bias added to dt, shape (nheads, dim) with tie_hdim.
    dt_softplus : bool
        Whether to apply softplus to dt.
    state_batch_indices : Optional[torch.Tensor]
        Maps batch index to cache slot, shape (batch,), int32.
    pad_slot_id : int
        Sentinel value for padded entries.
    state_scale : Optional[torch.Tensor]
        Block-scale decode factors for quantized state, shape (cache, nheads, dim), f32.
    rand_seed : Optional[torch.Tensor]
        Single-element int64 CUDA tensor for stochastic rounding seed.
    philox_rounds : int
        Philox PRNG rounds for stochastic rounding (default 10).
    d_split : Optional[int]
        Per-head DIM split factor (v12 §59).  Allowed: {1, 2, 4}.  When None
        (default), an auto-heuristic picks the largest pow2 ≤ 4 that brings
        total CTA count up to ~SMs * occupancy_estimate.  Use the override to
        force a specific value for benchmarking.

    Returns
    -------
    out : torch.Tensor
        Output tensor, shape (batch, T, nheads, dim).
    """
    # Validate rand_seed / philox_rounds
    if rand_seed is not None:
        assert isinstance(rand_seed, torch.Tensor), (
            "rand_seed must be a CUDA int64 tensor"
        )
        assert rand_seed.numel() == 1, (
            f"rand_seed must be single-element, got {rand_seed.numel()}"
        )
        assert rand_seed.dtype == torch.int64, (
            f"rand_seed must be int64, got {rand_seed.dtype}"
        )
        assert rand_seed.is_cuda, "rand_seed must be a CUDA tensor"
        assert philox_rounds > 0, (
            f"philox_rounds must be > 0 with rand_seed, got {philox_rounds}"
        )
    else:
        philox_rounds = 0

    # Extract JIT specialization keys
    dim = state.size(2)
    dstate = state.size(3)
    ntokens_mtp = x.size(1)
    assert ntokens_mtp <= 16, (
        f"ssu_incremental supports at most 16 MTP tokens, got {ntokens_mtp}"
    )

    # ── d_split selection (v12 §59) ──
    # Auto-heuristic: pick the largest pow2 ∈ {1, 2} that keeps total CTA
    # count <= SMs * occupancy_estimate.  occupancy_estimate=2 matches the
    # PAD_TOKENS path's CTAs/SM (see v10.7).  d_split=4 is deferred to
    # v12.x (needs warp-count restructure for output MMA).
    if d_split is None:
        # Auto-heuristic still clamped to 1 — re-enable as a separate
        # tuning change once the benchmarking is in place.  The override
        # knob is open and exercised by the d_split=2 correctness tests.
        d_split = 1
    assert d_split in (1, 2), (
        f"d_split must be in {{1, 2}} for v12 (d_split=4 deferred), got {d_split}"
    )
    assert dim % d_split == 0, f"dim={dim} must be divisible by d_split={d_split}"
    assert dim // d_split >= 32, (
        f"d_split={d_split} gives D_PER_CTA={dim // d_split} < 32 "
        "(output MMA m16n8 floor with _1×4 warp layout)"
    )

    stateIndex_dtype = torch.int32
    if state_batch_indices is not None:
        stateIndex_dtype = state_batch_indices.dtype

    state_scale_dtype = state_scale.dtype if state_scale is not None else None

    module = _get_module(
        state.dtype,
        x.dtype,
        dt.dtype,
        D.dtype
        if D is not None
        else (
            dt_bias.dtype if dt_bias is not None else dt.dtype
        ),  # weight_dtype (D, dt_bias)
        A.dtype,  # matrixA_dtype
        stateIndex_dtype,
        state_scale_dtype,
        dim,
        dstate,
        ntokens_mtp,
        philox_rounds,
    )

    module.ssu_incremental(
        state,
        x,
        dt,
        A,
        B,
        C,
        out,
        old_x,
        old_B,
        old_dt_proc,
        old_cumAdt,
        cache_buf_idx,
        prev_num_accepted_tokens,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        pad_slot_id,
        state_scale,
        rand_seed,
        d_split,
    )
    return out
