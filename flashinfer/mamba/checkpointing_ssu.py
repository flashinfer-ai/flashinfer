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

from ..api_logging import flashinfer_api
from ..jit.mamba.checkpointing_ssu import gen_checkpointing_ssu_module
from ..utils import register_custom_op, register_fake_op


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
    npredicted: int,
    max_window: int,
    heads_per_group: int,
    philox_rounds: int = 0,
    enable_pdl: bool = False,
):
    return gen_checkpointing_ssu_module(
        state_dtype,
        input_dtype,
        dt_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        state_scale_dtype,
        dim,
        dstate,
        npredicted,
        max_window,
        heads_per_group,
        philox_rounds,
        enable_pdl,
    ).build_and_load()


@register_custom_op(
    "flashinfer::checkpointing_ssu",
    mutates_args=(
        "state",
        "out",
        "old_x",
        "old_B",
        "old_dt",
        "old_cumAdt",
        "state_scale",
    ),
)
def _checkpointing_ssu(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    old_x: torch.Tensor,
    old_B: torch.Tensor,
    old_dt: torch.Tensor,
    old_cumAdt: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    state_scale: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    d_split: int,
    cu_seqlens: Optional[torch.Tensor],
    enable_pdl: bool,
    philox_rounds: int,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    npredicted: int,
    max_window: int,
    heads_per_group: int,
) -> None:
    """Internal function registered with torch.library for torch.compile() support."""
    module = _get_module(
        state_dtype,
        input_dtype,
        dt_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        state_scale.dtype if state_scale is not None else None,
        dim,
        dstate,
        npredicted,
        max_window,
        heads_per_group,
        philox_rounds,
        enable_pdl,
    )
    module.checkpointing_ssu(
        state,
        x,
        dt,
        A,
        B,
        C,
        out,
        old_x,
        old_B,
        old_dt,
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
        cu_seqlens,
    )


@register_fake_op("flashinfer::checkpointing_ssu")
def _checkpointing_ssu_fake(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    old_x: torch.Tensor,
    old_B: torch.Tensor,
    old_dt: torch.Tensor,
    old_cumAdt: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    state_scale: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    d_split: int,
    cu_seqlens: Optional[torch.Tensor],
    enable_pdl: bool,
    philox_rounds: int,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    npredicted: int,
    max_window: int,
    heads_per_group: int,
) -> None:
    """Fake implementation for torch.compile() meta tensor propagation."""
    pass


@flashinfer_api
def checkpointing_ssu(
    state: torch.Tensor,
    old_x: torch.Tensor,
    old_B: torch.Tensor,
    old_dt: torch.Tensor,
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
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Checkpointing SSU with MTP replay using matmul-based parallel token processing.

    Parameters
    ----------
    state : torch.Tensor
        SSM state, shape (state_cache_size, nheads, dim, dstate). Updated in-place.
    old_x : torch.Tensor
        Cached x from previous step, shape (state_cache_size, T, nheads, dim). Single-buffered.
    old_B : torch.Tensor
        Cached B, shape (state_cache_size, 2, T, ngroups, dstate). Double-buffered.
    old_dt : torch.Tensor
        Cached processed dt, shape (state_cache_size, 2, nheads, T). Double-buffered, f32.
    old_cumAdt : torch.Tensor
        Cached cumulative A*dt, shape (state_cache_size, 2, nheads, T). Double-buffered, f32.
    cache_buf_idx : torch.Tensor
        Which buffer to read (0 or 1), shape (state_cache_size,), int32.
    prev_num_accepted_tokens : torch.Tensor
        Number of old tokens to replay, shape (state_cache_size,), int32.
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
        Maps batch index to cache slot, shape (batch,), int32 | int64.
    pad_slot_id : int
        Sentinel value for padded entries.
    state_scale : Optional[torch.Tensor]
        Block-scale decode factors for quantized state, shape (state_cache_size, nheads, dim), f32.
    rand_seed : Optional[torch.Tensor]
        Single-element int64 CUDA tensor for stochastic rounding seed.
    philox_rounds : int
        Philox PRNG rounds for stochastic rounding (default 10).
    d_split : Optional[int]
        Per-head DIM split factor.  This is only exposed for benchmarking.
        Do not use it cause it will make things slow.
    cu_seqlens : Optional[torch.Tensor]
        Cumulative sequence lengths with shape ``(N + 1,)``, dtype
        ``torch.int32``, on the same CUDA device as ``x`` (the kernel
        asserts both). When provided, the new-token inputs (``x``, ``dt``,
        ``B``, ``C``, ``out``, optionally ``z``) are interpreted in varlen
        layout where tokens are packed along the **time** axis with batch
        fixed to 1 — i.e. ``x`` is 4-D with shape
        ``(1, total_tokens, nheads, dim)`` — instead of the default
        ``(batch, T, ...)`` layout.
    max_seqlen : Optional[int]
        Maximum sequence length present in ``cu_seqlens``, used by the kernel
        to size its per-sequence work tiles. Only meaningful in varlen mode
        (``cu_seqlens is not None``); falls back to ``max_window`` when
        omitted (wider smem than strictly needed but always safe). Must be
        ``None`` in non-varlen mode (the JIT key is taken from ``x.size(1)``).
    enable_pdl : bool
        When True the kernel is launched with
        `cudaLaunchAttributeProgrammaticStreamSerialization`, enabling the
        in-kernel `griddepcontrol.{wait,launch_dependents}` PTX to gate on
        the upstream (e.g. conv1d) and signal the downstream kernel.
        Caller's responsibility: upstream/downstream kernels must also be
        PDL-paired for the wait/signal to have effect.  Defaults to False.

    Returns
    -------
    out : torch.Tensor
        Output tensor, shape (batch, T, nheads, dim).

    Notes
    -----
    **In-place updates.** The custom op declares ``mutates_args =
    ("state", "out", "old_x", "old_B", "old_dt", "old_cumAdt",
    "state_scale")`` — the four ``old_*`` cache tensors are double-buffered
    and the kernel writes the *current* step's x / B / dt / cumulative-A·dt
    back into the slot selected by ``cache_buf_idx`` so the next call can
    replay them. ``state_scale`` is also written when ``state`` is
    quantized (int8 / fp8_e4m3fn): the kernel computes new per-block
    decode scales and stores them here for the caller to dequantize
    against on read-back.
    """
    # Validate quantized state ↔ state_scale combo.
    # int8 and fp8_e4m3fn use a per-(cache, head, dim) decode-scale tensor
    # (QUANT_MAX = 127 and 448 respectively).  Non-quantized dtypes must NOT
    # pass one (the kernel hardcodes the dispatch on whether `state_scale_t`
    # is `void`).
    _quantized_state_dtypes = (torch.int8, torch.float8_e4m3fn)
    if state.dtype in _quantized_state_dtypes:
        assert state_scale is not None, (
            f"state.dtype={state.dtype} requires a state_scale tensor "
            f"of shape (cache, nheads, dim) and dtype float32"
        )
        cache_size, nheads_state, dim_state = (
            state.size(0),
            state.size(1),
            state.size(2),
        )
        assert state_scale.shape == (cache_size, nheads_state, dim_state), (
            f"state_scale shape mismatch: expected "
            f"{(cache_size, nheads_state, dim_state)}, got {tuple(state_scale.shape)}"
        )
        assert state_scale.dtype == torch.float32, (
            f"state_scale must be float32 (got {state_scale.dtype})"
        )
        assert state_scale.is_cuda, "state_scale must be a CUDA tensor"
        # The 8-bit replay path uses a per-warp M-shard layout
        # (Layout<_4, _1>) that requires per-warp M = D_PER_CTA / 4 ≥ 16
        # (m16n8 atom M).  → D_PER_CTA ≥ 64 → d_split == 1.  d_split == 2
        # would give D_PER_CTA = 32 = 8 per warp, which doesn't fit the atom.
        assert d_split == 1 or d_split is None, (
            f"8-bit state (int8/fp8) requires d_split=1 (got d_split={d_split}); "
            f"the M-shard-per-warp layout needs D_PER_CTA / 4 >= 16."
        )
    else:
        assert state_scale is None, (
            f"state_scale must be None for non-quantized state.dtype={state.dtype}"
            f" (allowed quantized dtypes: {_quantized_state_dtypes})"
        )

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
    max_window = old_x.size(1)
    # Varlen: inputs are packed (1, total_tokens, ...) — `x.size(1)` is no
    # longer a JIT key (it varies per call).  The caller must promise an
    # upper bound on every cu_seqlens[i+1] - cu_seqlens[i] via `max_seqlen`,
    # which becomes the JIT-stamped NPREDICTED.  Default (when omitted) is
    # max_window — wider smem than strictly needed when actual seq_lens are
    # small, but always safe.
    if cu_seqlens is not None:
        assert x.dim() == 4 and x.size(0) == 1, (
            f"varlen mode: x must be (1, total_tokens, nheads, dim), got shape {tuple(x.shape)}"
        )
        assert cu_seqlens.dim() == 1 and cu_seqlens.dtype == torch.int32, (
            f"cu_seqlens must be a 1D int32 CUDA tensor, got shape "
            f"{tuple(cu_seqlens.shape)} dtype {cu_seqlens.dtype}"
        )
        assert cu_seqlens.is_cuda, "cu_seqlens must be a CUDA tensor"
        npredicted = max_seqlen if max_seqlen is not None else max_window
    else:
        assert max_seqlen is None, (
            "max_seqlen is only valid with cu_seqlens (varlen mode); for "
            "non-varlen the JIT key is taken from x.size(1)"
        )
        npredicted = x.size(1)
    assert max_window <= 16, (
        f"checkpointing_ssu supports at most 16 cache tokens (max_window), got {max_window}"
    )
    assert npredicted <= max_window, (
        f"npredicted ({npredicted}) must be <= max_window ({max_window})"
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

    # HEADS_PER_GROUP is JIT-stamped (was a runtime `dispatchRatio` over 7
    # candidate values).  Stamping it as a constexpr means each .so compiles
    # exactly one specialization instead of seven — ~7x faster per JIT.
    # The kernel asserts `params.nheads / params.ngroups == HEADS_PER_GROUP`
    # before launch.
    nheads = state.size(1)
    ngroups = B.size(-2)
    assert nheads % ngroups == 0, (
        f"nheads ({nheads}) must be divisible by ngroups ({ngroups})"
    )
    heads_per_group = nheads // ngroups

    # D and dt_bias share a single JIT `weight_dtype` specialization.  If
    # both are present, the dtypes must match — otherwise the kernel will
    # read one of them as the wrong type.
    if D is not None and dt_bias is not None:
        assert D.dtype == dt_bias.dtype, (
            f"D.dtype ({D.dtype}) and dt_bias.dtype ({dt_bias.dtype}) must match"
        )
    weight_dtype = (
        D.dtype
        if D is not None
        else (dt_bias.dtype if dt_bias is not None else dt.dtype)
    )

    _checkpointing_ssu(
        state,
        x,
        dt,
        A,
        B,
        C,
        out,
        old_x,
        old_B,
        old_dt,
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
        cu_seqlens,
        enable_pdl,
        philox_rounds=philox_rounds,
        state_dtype=state.dtype,
        input_dtype=x.dtype,
        dt_dtype=dt.dtype,
        weight_dtype=weight_dtype,
        matrixA_dtype=A.dtype,
        stateIndex_dtype=stateIndex_dtype,
        dim=dim,
        dstate=dstate,
        npredicted=npredicted,
        max_window=max_window,
        heads_per_group=heads_per_group,
    )
    return out
