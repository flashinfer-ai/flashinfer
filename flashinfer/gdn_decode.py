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

"""
Gated Delta Rule Decode - API Layer
====================================

This file provides the public API for gated delta rule decode operations.
Kernel implementations are in flashinfer/gdn_kernels/.

Three APIs are provided:
- gated_delta_rule_decode_pretranspose: V-major state layout [B, HV, V, K], T=1
- gated_delta_rule_decode: K-major state layout [B, HV, K, V], T=1
- gated_delta_rule_mtp: Multi-token processing (T > 1) for speculative decoding
"""

from typing import Optional, Tuple

import torch

from .jit.core import logger

try:
    from .api_logging import flashinfer_api
    from .trace.templates.gdn import (
        gated_delta_rule_decode_trace,
        gdn_mtp_trace,
    )

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False
    gated_delta_rule_decode_trace = None  # type: ignore[assignment]
    gdn_mtp_trace = None  # type: ignore[assignment]

    # Fallback decorator for standalone usage (accepts trace= kwarg)
    def flashinfer_api(func=None, *, trace=None):  # type: ignore[misc]
        if func is None:
            return lambda f: f
        return func


# GDN decode BF16 state kernels - optional backend
try:
    from .gdn_kernels.gdn_decode_bf16_state import (
        gated_delta_rule as _gated_delta_rule_bf16_state,
        gated_delta_rule_mtp as _gated_delta_rule_bf16_state_mtp,
    )

    _GDN_DECODE_BF16_STATE_AVAILABLE = True
except (ImportError, RuntimeError):
    _GDN_DECODE_BF16_STATE_AVAILABLE = False
    _gated_delta_rule_bf16_state = None
    _gated_delta_rule_bf16_state_mtp = None

# Pretranspose decode kernel (V-major state, T=1)
try:
    from .gdn_kernels.gdn_decode_pretranspose import run_pretranspose_decode

    _PRETRANSPOSE_AVAILABLE = True
except (ImportError, RuntimeError):
    _PRETRANSPOSE_AVAILABLE = False
    run_pretranspose_decode = None

# Nontranspose decode kernel (K-major state, T=1)
try:
    from .gdn_kernels.gdn_decode_nontranspose import (
        run_nontranspose_decode,
        TILE_V_NT,
    )

    _NONTRANSPOSE_AVAILABLE = True
except (ImportError, RuntimeError):
    _NONTRANSPOSE_AVAILABLE = False
    run_nontranspose_decode = None
    TILE_V_NT = 32  # fallback for assertions

# MTP kernel (T > 1, speculative decoding verification)
try:
    from .gdn_kernels.gdn_decode_mtp import (
        run_mtp_decode,
        get_tile_v_mtp,
        get_vec_size_mtp,
        get_mtp_config,
    )

    _MTP_AVAILABLE = True
except (ImportError, RuntimeError):
    _MTP_AVAILABLE = False
    run_mtp_decode = None
    get_tile_v_mtp = None
    get_vec_size_mtp = None
    get_mtp_config = None

# Constants for V-divisibility validation
TILE_V = 8  # pretranspose tile size


# ============================================================================
# API: Pretranspose Decode (V-major / K-last state layout)
# ============================================================================


@flashinfer_api(trace=gated_delta_rule_decode_trace)
def gated_delta_rule_decode_pretranspose(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: Optional[torch.Tensor],
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Gated Delta Rule Decode kernel for single-token generation.

    Implements the decode phase of gated delta rule linear attention,
    processing one token at a time and updating the recurrent state.

    Parameters
    ----------
    q : torch.Tensor
        Current query of shape ``[B, 1, H, K]``.  Must be float16/bfloat16.
    k : torch.Tensor
        Current key of shape ``[B, 1, H, K]``.  Must be float16/bfloat16.
    v : torch.Tensor
        Current value of shape ``[B, 1, HV, V]``.  Must be float16/bfloat16.
    state : torch.Tensor, optional
        Current state of shape ``[B, HV, V, K]`` (v-major / K-last layout).
        Float32: legacy kernel (T=1 only).  Bfloat16: BF16 state backend
        (T=1 or MTP for T>1) when K=V=128.  Updated in-place.  Pass ``None``
        when using ``initial_state`` / ``initial_state_indices`` instead.
    A_log : torch.Tensor
        Log decay parameter of shape ``[HV]``.  Must be float32.
    a : torch.Tensor
        Input-dependent decay of shape ``[B, 1, HV]``.  Must be
        float16/bfloat16.
    dt_bias : torch.Tensor
        Decay bias of shape ``[HV]``.  Must be bfloat16 or float32.
    b : torch.Tensor
        Update gate (beta) input of shape ``[B, 1, HV]``.  Must be
        float16/bfloat16.
    scale : float, optional
        Scale factor for queries.  If ``None``, defaults to
        ``1 / sqrt(K)``.
    output : torch.Tensor, optional
        Pre-allocated output tensor of shape ``[B, 1, HV, V]``.  Allocated
        automatically when ``None``.
    use_qk_l2norm : bool
        Whether to apply L2 normalization to q and k.  Default: ``True``.
    initial_state : torch.Tensor, optional
        State pool of shape ``[pool_size, HV, V, K]`` (K-last /
        K-contiguous, same layout as the per-batch ``state`` argument).
        When provided, the kernel gathers directly from the pool using
        ``initial_state_indices`` and writes updates back in-place,
        eliminating the caller-side gather/scatter overhead.  Requires
        bfloat16 state with K=V=128 (bf16 fast path).
    initial_state_indices : torch.Tensor, optional
        Per-batch indices of shape ``[B]`` (int32 or int64) mapping each
        batch entry to its slot in ``initial_state``.  Required when
        ``initial_state`` is provided.
    output_state_indices : torch.Tensor, optional
        Per-batch indices of shape ``[B]`` (int32 or int64) specifying
        where to write the updated state for each batch entry in the pool.
        Requires ``initial_state`` to be provided.  If ``None``, the kernel
        writes the updated state back to the same slot it read from (i.e.
        ``initial_state_indices``).

        **Padding / inactive sequences**: set the index to ``-1`` for any
        batch entry that should be treated as padding.  The two backends
        handle ``-1`` differently:

        - **bf16 fast path** (bfloat16 state, K=V=128): ``-1`` is redirected
          to ``initial_state[0]``, which acts as a sacrificial *null
          buffer*.  The kernel reads from and writes back to slot 0; the
          output for that batch entry is computed but **undefined** (caller
          should not use it).  The caller must therefore allocate the pool
          with an extra leading slot (``pool_size = num_real_slots + 1``)
          and keep real slots at indices ``1..pool_size-1``.
        - **float32 legacy path** (T=1): ``-1`` entries are skipped
          entirely; neither the state pool nor the output are touched for
          that batch entry; the output slot is written as **zero**.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(output, state_or_initial_state)`` where ``output`` has shape
        ``[B, 1, HV, V]`` and the second element is the updated state
        (mutated in place).

    Notes
    -----
    - Requires SM90+ (Hopper, Blackwell, etc.).
    - State is always updated in-place; the pool path writes directly into
      ``initial_state`` memory (no separate scatter step needed).
    - State layout is v-major (K-last): ``[B, HV, V, K]``.  When state is
      bfloat16 and ``K = V = 128``, the BF16 state kernel is used (T=1 or
      MTP for T>1); the pool+indices path routes through the MTP kernel.
    - Pool+indices (``initial_state`` / ``initial_state_indices``) are
      supported on both the bf16 fast path (K=V=128) and the float32 legacy
      path (T=1).  Both paths support ``-1`` padding indices (see
      ``initial_state_indices`` above for per-backend semantics).
    - Legacy path (float32 state, T=1): ``K`` and ``V`` must each be
      ``>= 128``, and ``V`` must be a multiple of 8 (the pretranspose tile
      size ``TILE_V``).
    """
    # Validate input shapes
    B, T, H, K = q.shape
    _, _, HV, V = v.shape

    use_pool = initial_state is not None
    assert use_pool == (initial_state_indices is not None), (
        "initial_state and initial_state_indices must be provided together"
    )
    if output_state_indices is not None:
        assert use_pool, (
            "output_state_indices can only be used with initial_state (pool mode)"
        )
        assert output_state_indices.shape == (B,), (
            f"Expected output_state_indices shape [{B}], "
            f"got {output_state_indices.shape}"
        )
        assert output_state_indices.dtype in (torch.int32, torch.int64), (
            f"output_state_indices must be int32 or int64, "
            f"got {output_state_indices.dtype}"
        )

    if use_pool:
        pool_size = initial_state.shape[0]
        assert initial_state.shape == (pool_size, HV, V, K), (
            f"Expected initial_state shape [pool_size={pool_size}, HV={HV}, V={V}, K={K}], "
            f"got {initial_state.shape}"
        )
        assert initial_state.stride(-1) == 1, (
            "initial_state must be K-contiguous (stride[-1] == 1) for pretranspose decode, "
            f"got stride={initial_state.stride()}"
        )
    else:
        assert state is not None, "Either state or initial_state must be provided"
        # Validate state shape (K-last: [B, HV, V, K])
        assert state.shape == (B, HV, V, K), (
            f"Expected state shape [B={B}, HV={HV}, V={V}, K={K}], got {state.shape}"
        )

    # Backend: BF16 state kernel when bf16 state, K=V=128
    state_dtype = initial_state.dtype if use_pool else state.dtype
    use_bf16_state = (
        _GDN_DECODE_BF16_STATE_AVAILABLE
        and state_dtype == torch.bfloat16
        and K == 128
        and V == 128
    )
    if use_bf16_state:
        assert q.dtype in (torch.float16, torch.bfloat16), (
            f"q must be float16/bfloat16, got {q.dtype}"
        )
        assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"
        scale_val = K**-0.5 if scale is None else scale
        # The BF16 path is pool-only. When the caller uses non-pool semantics
        # (passes ``state`` instead of ``initial_state``), treat ``state`` as
        # a pool of size B and synthesize sequential indices arange(B).
        if use_pool:
            bf16_pool = initial_state
            bf16_indices = initial_state_indices
        else:
            bf16_pool = state
            bf16_indices = torch.arange(B, dtype=torch.int32, device=q.device)
        # Forward user's `output=` straight into the kernel when its dtype
        # matches what the kernel writes (bf16). This avoids a redundant
        # device-to-device `output.copy_()` after every call.
        target_dtype = output.dtype if output is not None else q.dtype
        forward_output = (
            output if (output is not None and output.dtype == torch.bfloat16) else None
        )
        if T == 1:
            out = _gated_delta_rule_bf16_state(
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=bf16_pool,
                initial_state_indices=bf16_indices,
                output_state_indices=output_state_indices,
                use_qk_l2norm_in_kernel=use_qk_l2norm,
                scale=scale_val,
                output=forward_output,
            )
        else:
            # MTP kernel for T>1 (supports pool+indices and intermediate caching)
            out = _gated_delta_rule_bf16_state_mtp(
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=bf16_pool,
                initial_state_indices=bf16_indices,
                output_state_indices=output_state_indices,
                use_qk_l2norm_in_kernel=use_qk_l2norm,
                scale=scale_val,
                output=forward_output,
            )
        if forward_output is not None:
            # Kernel wrote directly into the user's buffer.
            output = forward_output
        elif output is None:
            output = out
        else:
            # User wants a non-bf16 dtype; cast on the way back.
            output.copy_(out.to(target_dtype))
        return_state = initial_state if use_pool else state
        return output, return_state

    # Legacy path: float32 state (T=1 via run_pretranspose_decode; T>1 routes
    # through gated_delta_rule_mtp when a pool is provided).
    use_pool_indexing = initial_state_indices is not None

    if use_pool:
        assert initial_state.dtype == torch.float32, (
            f"initial_state must be float32 for legacy path, got {initial_state.dtype}"
        )
    else:
        assert state is not None, "Either state or initial_state must be provided"
        assert state.dtype == torch.float32, f"state must be float32, got {state.dtype}"

    # Route fp32 + T>1 through the MTP kernel (supports separate read/write
    # indices via output_state_indices). Direct-state (non-pool) callers must
    # still use T=1 — wrapping a per-batch state as a pool of size B is the
    # caller's responsibility if they want T>1.
    if T > 1:
        assert use_pool, (
            f"fp32 state with T={T} > 1 requires pool mode (pass initial_state "
            f"and initial_state_indices instead of state)."
        )
        out, return_state = gated_delta_rule_mtp(
            q=q,
            k=k,
            v=v,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            scale=scale,
            output=output,
            intermediate_states_buffer=None,
            disable_state_update=False,
            use_qk_l2norm=use_qk_l2norm,
            output_state_indices=output_state_indices,
        )
        return out, return_state

    assert T == 1, f"Decode only supports T=1, got T={T}"

    # Validate K and V constraints
    assert K >= 128, f"K must be at least 128, got K={K}"
    assert V >= 128, f"V must be at least 128, got V={V}"
    assert V % TILE_V == 0, (
        f"V must be divisible by {TILE_V} to prevent out-of-bounds access, got V={V}"
    )

    # Validate dtypes
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"q must be float16/bfloat16, got {q.dtype}"
    )
    assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"

    # Set default scale
    if scale is None:
        scale = K**-0.5

    # Allocate output if not provided
    # Note: kernel outputs bfloat16, we'll convert to q.dtype if needed
    output_provided = output is not None
    target_dtype = output.dtype if output_provided else q.dtype

    if output is None:
        # Kernel outputs bfloat16, allocate in that dtype first
        output = torch.zeros((B, T, HV, V), dtype=torch.bfloat16, device=q.device)

    # Build h0_source for kernel.
    # - pool path: keep original [pool_size, HV, V, K] view so non-contiguous
    #   page-strided pools are supported.
    # - direct path: flatten to [B*HV, V, K].
    if use_pool:
        pool_size = initial_state.shape[0]
        h0_source = initial_state
        return_state = initial_state
    else:
        pool_size = B
        h0_source = state.reshape(pool_size * HV, V, K)
        return_state = state

    # Execute kernel
    run_pretranspose_decode(
        h0_source,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        output,
        B,
        T,
        H,
        HV,
        K,
        V,
        scale,
        use_qk_l2norm,
        use_pool_indexing=use_pool_indexing,
        initial_state_indices=initial_state_indices,
        output_state_indices=output_state_indices,
    )

    # Copy state back only if not using pool and state was not contiguous
    # (if contiguous, reshape returns a view and kernel updated state in-place)
    # Pool path: kernel writes directly into initial_state via pool indices
    if not use_pool and not state.is_contiguous():
        state.copy_(h0_source.reshape(B, HV, V, K))

    # Convert output to target dtype if needed (kernel outputs bfloat16)
    if output.dtype != target_dtype:
        output = output.to(target_dtype)

    return output, return_state


# ============================================================================
# API: Nontranspose Decode (K-major state layout, recommended)
# ============================================================================


@flashinfer_api(trace=gated_delta_rule_decode_trace)
def gated_delta_rule_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Gated Delta Rule Decode kernel (K-major layout, no transpose needed).

    Implements the decode phase of gated delta rule linear attention,
    processing one token at a time and updating the recurrent state.  This
    variant uses K-major state layout ``[B, HV, K, V]`` (no transposition).

    Parameters
    ----------
    q : torch.Tensor
        Current query of shape ``[B, 1, H, K]``.  Must be float16/bfloat16.
    k : torch.Tensor
        Current key of shape ``[B, 1, H, K]``.  Must be float16/bfloat16.
    v : torch.Tensor
        Current value of shape ``[B, 1, HV, V]``.  Must be float16/bfloat16.
    state : torch.Tensor
        Current state of shape ``[B, HV, K, V]`` (k-major layout).  Must be
        float32.  Updated in-place.
    A_log : torch.Tensor
        Log decay parameter of shape ``[HV]``.  Must be float32.
    a : torch.Tensor
        Input-dependent decay of shape ``[B, 1, HV]``.  Must be
        float16/bfloat16.
    dt_bias : torch.Tensor
        Decay bias of shape ``[HV]``.  Must be bfloat16 or float32.
    b : torch.Tensor
        Update gate (beta) input of shape ``[B, 1, HV]``.  Must be
        float16/bfloat16.
    scale : float, optional
        Scale factor for queries.  If ``None``, defaults to ``1 /
        sqrt(K)``.
    output : torch.Tensor, optional
        Pre-allocated output tensor of shape ``[B, 1, HV, V]``.  Allocated
        automatically when ``None``.
    use_qk_l2norm : bool
        Whether to apply L2 normalization to q and k.  Default: ``True``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(output, state)`` where ``output`` has shape ``[B, 1, HV, V]``
        and ``state`` has shape ``[B, HV, K, V]`` (updated in-place).

    Notes
    -----
    - Requires SM90 (Hopper) architecture.
    - State is updated in-place.
    - ``K`` and ``V`` must each be ``>= 128``.  ``V`` must be a multiple
      of 32 (``TILE_V_NT``): the launcher conservatively asserts the
      large-batch tile size to cover both code paths, even though the
      small-batch kernel could in principle accept ``V % 16 == 0``
      (``TILE_V_SMALL_NT``).
    - State layout is k-major: ``[B, HV, K, V]`` (no transpose needed).
    """
    # Validate input shapes
    B, T, H, K = q.shape
    assert T == 1, f"Decode only supports T=1, got T={T}"
    _, _, HV, V = v.shape

    # Validate state shape
    assert state.shape == (B, HV, K, V), (
        f"Expected state shape [B={B}, HV={HV}, K={K}, V={V}], got {state.shape}"
    )

    # Validate K and V constraints
    assert K >= 128, f"K must be at least 128, got K={K}"
    assert V >= 128, f"V must be at least 128, got V={V}"
    # V must be divisible by tile size to prevent out-of-bounds access
    # For small batch: TILE_V_SMALL_NT=16, for large batch: TILE_V_NT=32
    # Use the more restrictive constraint (32) to cover both cases
    assert V % TILE_V_NT == 0, (
        f"V must be divisible by {TILE_V_NT} to prevent out-of-bounds access, got V={V}"
    )

    # Validate dtypes
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"q must be float16/bfloat16, got {q.dtype}"
    )
    assert state.dtype == torch.float32, f"state must be float32, got {state.dtype}"
    assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"

    # Set default scale
    if scale is None:
        scale = K**-0.5

    # Allocate output if not provided
    output_provided = output is not None
    target_dtype = output.dtype if output_provided else q.dtype

    if output is None:
        # Kernel outputs bfloat16, allocate in that dtype first
        output = torch.zeros((B, T, HV, V), dtype=torch.bfloat16, device=q.device)

    # State is in K-major layout [B, HV, K, V]
    # Flatten to [B*HV, K, V] to ensure proper alignment for SIMT async copy
    # This avoids alignment issues when B=1 (zero strides cause alignment failures)
    state_contiguous = state.contiguous()
    h0_source = state_contiguous.view(B * HV, K, V)

    # Execute kernel
    run_nontranspose_decode(
        h0_source,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        output,
        B,
        T,
        H,
        HV,
        K,
        V,
        scale,
        use_qk_l2norm,
    )

    # Copy state back only if state was not contiguous
    # (if contiguous, state_contiguous is state itself, so kernel updated state in-place)
    if state_contiguous.data_ptr() != state.data_ptr():
        state.copy_(state_contiguous)

    # Convert output to target dtype if needed (kernel outputs bfloat16)
    if output.dtype != target_dtype:
        output = output.to(target_dtype)

    return output, state


# ============================================================================
# API: MTP Decode (Multiple Token Processing, T > 1)
# ============================================================================


@flashinfer_api(trace=gdn_mtp_trace)
def gated_delta_rule_mtp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    disable_state_update: Optional[bool] = None,
    use_qk_l2norm: bool = True,
    output_state_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Gated Delta Rule MTP kernel (Multiple Token Processing).

    Processes multiple tokens (``T > 1``) per call, typically used for
    speculative decoding verification.  Supports intermediate state caching
    for potential rollback scenarios.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape ``[B, T, H, K]``.
    k : torch.Tensor
        Key tensor of shape ``[B, T, H, K]``.
    v : torch.Tensor
        Value tensor of shape ``[B, T, HV, V]``.
    initial_state : torch.Tensor
        Initial state pool of shape ``[pool_size, HV, V, K]`` (K-last
        layout). **Must be float32** — this standalone MTP entry point
        does not support the BF16 fast path; for a BF16 K=V=128 state
        pool, call :func:`gated_delta_rule_decode_pretranspose` instead
        (which dispatches into the BF16 MTP kernel when ``T > 1``).
        When contiguous the kernel reads/writes the pool in-place via the
        free 4D→3D reshape view; a non-contiguous pool is dispatched
        through the native 4D ``use_pool_indexing=True`` path and the
        kernel writes the strided pool in place without densification.
    initial_state_indices : torch.Tensor
        Read indices mapping each batch to its slot in ``initial_state``,
        shape ``[B]``. Negative entries are treated as padding — the
        kernel skips both the read and the writeback for that batch and
        the output slot is left as the caller-allocated value (zero when
        ``output`` is ``None``).
    A_log : torch.Tensor
        Log decay parameter of shape ``[HV]``.
    a : torch.Tensor
        Input-dependent decay of shape ``[B, T, HV]``.
    dt_bias : torch.Tensor
        Decay bias of shape ``[HV]``.
    b : torch.Tensor
        Update gate input of shape ``[B, T, HV]``.
    scale : float, optional
        Scaling factor for queries.  If ``None``, uses ``1 / sqrt(K)``.
    output : torch.Tensor, optional
        Pre-allocated output tensor of shape ``[B, T, HV, V]``.
    intermediate_states_buffer : torch.Tensor, optional
        Buffer for caching intermediate states, shape ``[B, T, HV, V, K]``
        (first dim is indexed per-batch, not per-pool-slot — buffer must
        be at least ``B`` rows and contiguous; must be float32 when
        provided). When ``None``, intermediate states are not cached.
    disable_state_update : bool, optional
        If ``True``, the initial state is not updated.  Currently defaults
        to ``True``; pass this argument explicitly to silence the
        deprecation warning - the default will change to ``False`` in
        FlashInfer 0.7.0.

        .. deprecated::
            The implicit default of ``True`` is deprecated and will change
            to ``False`` in version 0.7.0.  Pass
            ``disable_state_update=True`` or ``disable_state_update=False``
            explicitly to silence the warning.
    use_qk_l2norm : bool
        Whether to apply L2 normalization to q and k.  Default: ``True``.
    output_state_indices : torch.Tensor, optional
        Write indices of shape ``[B]`` (int32 or int64) specifying the
        destination pool slot for each batch's updated state.  Defaults
        to ``initial_state_indices`` (read and write target the same
        slot).  Negative entries skip the writeback for that batch
        (the read still runs).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(output, initial_state)`` where ``output`` has shape
        ``[B, T, HV, V]`` and ``initial_state`` is the updated state
        (unchanged when ``disable_state_update=True``).

    Notes
    -----
    - Requires SM90 (Hopper) architecture.
    - Supports ``T > 1`` (multiple token processing).
    - State layout is K-last: ``[pool_size, HV, V, K]``.
    - Optimized for speculative decoding verification scenarios.
    """
    # Handle deprecation of disable_state_update default value
    if disable_state_update is None:
        logger.warning_once(
            "gated_delta_rule_mtp(): the 'disable_state_update' parameter currently "
            "defaults to True, but this default will change to False in FlashInfer "
            "0.7.0. Please pass disable_state_update=True or "
            "disable_state_update=False explicitly to suppress this warning."
        )
        disable_state_update = True

    # Validate input shapes
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    pool_size = initial_state.shape[0]

    # Dynamic TILE_V and vec_size selection based on batch size and sequence length
    tile_v = get_tile_v_mtp(B, T, num_v_heads=HV, v_dim=V)
    vec_size = get_vec_size_mtp(B, T)

    # Validate state shape
    assert initial_state.shape == (pool_size, HV, V, K), (
        f"Expected initial_state shape [pool_size={pool_size}, HV={HV}, V={V}, K={K}], got {initial_state.shape}"
    )

    # Validate K and V constraints
    assert K >= 128, f"K must be at least 128, got K={K}"
    assert V >= 128, f"V must be at least 128, got V={V}"
    assert V % tile_v == 0, (
        f"V must be divisible by {tile_v} to prevent out-of-bounds access, got V={V}"
    )

    # Validate dtypes
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"q must be float16/bfloat16, got {q.dtype}"
    )
    assert initial_state.dtype == torch.float32, (
        f"initial_state must be float32, got {initial_state.dtype}"
    )
    assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"

    # Validate output indices shape/dtype
    if output_state_indices is not None:
        assert output_state_indices.shape == (B,), (
            f"Expected output_state_indices shape [{B}], "
            f"got {output_state_indices.shape}"
        )
        assert output_state_indices.dtype in (torch.int32, torch.int64), (
            f"output_state_indices must be int32 or int64, "
            f"got {output_state_indices.dtype}"
        )

    # Set default scale
    if scale is None:
        scale = K**-0.5

    # Allocate output if not provided
    output_provided = output is not None
    target_dtype = output.dtype if output_provided else q.dtype

    if output is None:
        output = torch.zeros((B, T, HV, V), dtype=torch.bfloat16, device=q.device)

    # Build h0_source for the kernel.
    # - Contiguous 4D pool: `.reshape()` returns a free 3D view, kernel takes
    #   the flat-mode `use_pool_indexing=False` path (existing fast path).
    # - Non-contiguous 4D pool (e.g., vLLM page-strided SSM pool): pass the
    #   4D tensor through unchanged and tell the kernel to use 4D indexing,
    #   `use_pool_indexing=True`. The kernel reads/writes via the native
    #   `[pool, HV, V, K]` layout — no densification copy.
    pool_use_pool_indexing = not initial_state.is_contiguous()
    if pool_use_pool_indexing:
        h0_source = initial_state
    else:
        h0_source = initial_state.reshape(pool_size * HV, V, K)

    # Handle intermediate states. The kernel indexes the buffer by batch (i_n),
    # not by pool slot — see `flat_idx = i_n * T * HV + i_t * HV + i_hv` inside
    # the kernel. So the buffer's first dim MUST be at least B, and the buffer
    # MUST be contiguous so the reshape returns a free view. We make both
    # contracts explicit here to fail loudly on caller mistakes (the pre-existing
    # code silently did out-of-bounds writes when buffer.shape[0] < B).
    cache_intermediate_states = intermediate_states_buffer is not None
    if cache_intermediate_states:
        buffer_size = intermediate_states_buffer.shape[0]
        cache_steps = intermediate_states_buffer.shape[1]

        assert buffer_size >= B, (
            f"intermediate_states_buffer first dim ({buffer_size}) must be "
            f"at least B={B}: the kernel indexes it by batch (i_n in [0, B)), "
            f"so a smaller buffer causes out-of-bounds writes."
        )
        assert cache_steps >= T, (
            f"intermediate_states_buffer second dimension (cache_steps={cache_steps}) "
            f"must be at least T={T} to prevent out-of-bounds indexing"
        )
        assert intermediate_states_buffer.dtype == torch.float32, (
            f"intermediate_states_buffer must be float32, "
            f"got {intermediate_states_buffer.dtype}"
        )
        assert intermediate_states_buffer.is_contiguous(), (
            "intermediate_states_buffer must be contiguous so the kernel writes "
            "land in the caller-owned tensor (reshape would otherwise materialize "
            "a throwaway copy)."
        )

        intermediate_states = intermediate_states_buffer.view(
            buffer_size * cache_steps * HV, V, K
        )
    else:
        cache_steps = T
        intermediate_states = torch.zeros(1, 1, 1, dtype=torch.float32, device=q.device)

    # FLA-style per-token pool scatter. When provided, the kernel writes each
    # h_{t+1} directly to initial_state[ssm_state_indices[i, t]] instead of
    # to a dense intermediate_states_buffer. same_pool semantics only — the
    # final-state writeback to initial_state_indices[i] is skipped to avoid
    # clobbering h_0 (the per-token scatter at t=T-1 already wrote h_T to
    # its assigned pool slot). Caller pre-allocates B*T fresh slots from
    # the free-list and sizes the pool for at least B*(T+1) slots.
    per_token_pool_scatter = ssm_state_indices is not None
    if per_token_pool_scatter:
        assert intermediate_states_buffer is None, (
            "ssm_state_indices and intermediate_states_buffer are mutually exclusive"
        )
        assert not disable_state_update, (
            "ssm_state_indices requires state writes; disable_state_update must be False"
        )
        assert T >= 2, f"ssm_state_indices requires T >= 2 (got T={T})"
        assert ssm_state_indices.shape == (B, T), (
            f"ssm_state_indices must have shape [B={B}, T={T}], "
            f"got {tuple(ssm_state_indices.shape)}"
        )
        assert ssm_state_indices.dtype == torch.int32, (
            f"ssm_state_indices must be int32, got {ssm_state_indices.dtype}"
        )
        assert ssm_state_indices.device == q.device, (
            f"ssm_state_indices device {ssm_state_indices.device} != q device {q.device}"
        )

    # Execute kernel
    run_mtp_decode(
        h0_source,
        intermediate_states,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        output,
        initial_state_indices,
        B,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        cache_steps,
        tile_v,
        vec_size,
        scale,
        use_qk_l2norm,
        disable_state_update,
        cache_intermediate_states,
        ssm_state_indices=ssm_state_indices,
        output_state_indices=output_state_indices,
        use_pool_indexing=pool_use_pool_indexing,
    )

    # No post-kernel scatter step: the contiguity assert above guarantees
    # `intermediate_states` is a view of `intermediate_states_buffer`, and the
    # `use_pool_indexing=True` path makes the kernel write the strided pool
    # in place. Writes are visible to the caller directly.

    # Convert output to target dtype if needed
    if output.dtype != target_dtype:
        output = output.to(target_dtype)

    return output, initial_state
