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
except ImportError:
    _GDN_DECODE_BF16_STATE_AVAILABLE = False
    _gated_delta_rule_bf16_state = None
    _gated_delta_rule_bf16_state_mtp = None

# Pretranspose decode kernel (V-major state, T=1)
try:
    from .gdn_kernels.gdn_decode_pretranspose import run_pretranspose_decode

    _PRETRANSPOSE_AVAILABLE = True
except ImportError:
    _PRETRANSPOSE_AVAILABLE = False
    run_pretranspose_decode = None

# Nontranspose decode kernel (K-major state, T=1)
try:
    from .gdn_kernels.gdn_decode_nontranspose import (
        run_nontranspose_decode,
        TILE_V_NT,
    )

    _NONTRANSPOSE_AVAILABLE = True
except ImportError:
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
except ImportError:
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

    This implements the decode phase of gated delta rule linear attention,
    processing one token at a time and updating the recurrent state.

    Args:
        q (torch.Tensor):
            Current query of shape ``[B, 1, H, K]``. Must be float16/bfloat16.
        k (torch.Tensor):
            Current key of shape ``[B, 1, H, K]``. Must be float16/bfloat16.
        v (torch.Tensor):
            Current value of shape ``[B, 1, HV, V]``. Must be float16/bfloat16.
        state (Optional[torch.Tensor]):
            Current state of shape ``[B, HV, V, K]`` (v-major / K-last layout).
            Float32: legacy kernel (T=1 only). Bfloat16: BF16 state backend
            (T=1 or MTP for T>1) when K=V=128. Will be updated in-place.
            Pass ``None`` when using ``initial_state`` / ``initial_state_indices`` instead.
        A_log (torch.Tensor):
            Log decay parameter of shape ``[HV]``. Must be float32.
        a (torch.Tensor):
            Input-dependent decay of shape ``[B, 1, HV]``. Must be float16/bfloat16.
        dt_bias (torch.Tensor):
            Decay bias of shape ``[HV]``. Must be bfloat16 or float32.
        b (torch.Tensor):
            Update gate (beta) input of shape ``[B, 1, HV]``. Must be float16/bfloat16.
        scale (Optional[float]):
            Scale factor for queries. If None, defaults to ``1 / sqrt(K)``.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor of shape ``[B, 1, HV, V]``.
            If None, will be allocated automatically.
        use_qk_l2norm (bool):
            Whether to apply L2 normalization to q and k. Default: ``True``.
        initial_state (Optional[torch.Tensor]):
            State pool of shape ``[pool_size, HV, V, K]`` (K-last / K-contiguous,
            same layout as the per-batch ``state`` argument).
            When provided, the kernel gathers directly from the pool using
            ``initial_state_indices`` and writes updates back in-place — eliminating
            the caller-side gather/scatter overhead.
            Requires bfloat16 state with K=V=128 (bf16 fast path).
        initial_state_indices (Optional[torch.Tensor]):
            Per-batch indices of shape ``[B]`` (int32 or int64) mapping each batch
            entry to its slot in ``initial_state``.  Required when ``initial_state``
            is provided.
        output_state_indices (Optional[torch.Tensor]):
            Per-batch indices of shape ``[B]`` (int32 or int64) specifying where to write the updated state for each batch entry in the pool.
            Requires ``initial_state`` to be provided.
            If None, the kernel will write the updated state back to the same slot it read from (i.e., ``initial_state_indices``).

            **Padding / inactive sequences**: set the index to ``-1`` for any batch
            entry that should be treated as padding.  The two backends handle ``-1``
            differently:

            - **bf16 fast path** (bfloat16 state, K=V=128): ``-1`` is redirected
              to ``initial_state[0]``, which acts as a sacrificial *null buffer*.
              The kernel reads from and writes back to slot 0; the output for that
              batch entry is computed but **undefined** (caller should not use it).
              The caller must therefore allocate the pool with an extra leading slot
              (``pool_size = num_real_slots + 1``) and keep real slots at indices
              ``1..pool_size-1``.
            - **float32 legacy path** (T=1): ``-1`` entries are skipped entirely —
              neither the state pool nor the output are touched for that batch entry;
              the output slot is written as **zero**.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - output: Output tensor of shape ``[B, 1, HV, V]``
            - state or initial_state: Updated state (in-place).

    Note:
        - Requires SM90+ (Hopper, Blackwell, etc.)
        - State is always updated in-place; the pool path writes directly into
          ``initial_state`` memory (no separate scatter step needed)
        - State layout is v-major (K-last): [B, HV, V, K]. When state is bfloat16
          and K=V=128, the BF16 state kernel is used (T=1 or MTP for T>1).
          The pool+indices path routes through the MTP kernel.
        - pool+indices (``initial_state``/``initial_state_indices``) supported on
          both the bf16 fast path (K=V=128) and the float32 legacy path (T=1).
          Both paths support ``-1`` padding indices (see ``initial_state_indices``
          above for per-backend semantics).
        - Legacy path (float32 state, T=1): K and V must be multiples of 4.
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
        if T == 1 and not use_pool:
            # T=1 kernel does not accept initial_state_indices
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
                initial_state_source=state,
                use_qk_l2norm_in_kernel=use_qk_l2norm,
                scale=scale_val,
            )
        else:
            # MTP kernel supports T>=1 and pool+indices
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
                initial_state_source=initial_state if use_pool else state,
                initial_state_indices=initial_state_indices,
                output_state_indices=output_state_indices,
                use_qk_l2norm_in_kernel=use_qk_l2norm,
                scale=scale_val,
            )
        output_provided = output is not None
        target_dtype = output.dtype if output_provided else q.dtype
        if output is not None:
            output.copy_(out)
        else:
            output = out
        if output.dtype != target_dtype:
            output = output.to(target_dtype)
        return_state = initial_state if use_pool else state
        return output, return_state

    # Legacy path: T=1 only, float32 state (supports pool+indices via CuTe DSL kernel)
    use_pool_indexing = initial_state_indices is not None
    assert T == 1, f"Decode only supports T=1, got T={T}"

    if use_pool:
        assert initial_state.dtype == torch.float32, (
            f"initial_state must be float32 for legacy path, got {initial_state.dtype}"
        )
    else:
        assert state is not None, "Either state or initial_state must be provided"
        assert state.dtype == torch.float32, f"state must be float32, got {state.dtype}"

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

    This implements the decode phase of gated delta rule linear attention,
    processing one token at a time and updating the recurrent state.
    This version uses K-major state layout [B, HV, K, V] which is more natural
    and doesn't require transposition.

    Args:
        q (torch.Tensor):
            Current query of shape ``[B, 1, H, K]``. Must be float16/bfloat16.
        k (torch.Tensor):
            Current key of shape ``[B, 1, H, K]``. Must be float16/bfloat16.
        v (torch.Tensor):
            Current value of shape ``[B, 1, HV, V]``. Must be float16/bfloat16.
        state (torch.Tensor):
            Current state of shape ``[B, HV, K, V]`` (k-major layout).
            Must be float32. Will be updated in-place.
        A_log (torch.Tensor):
            Log decay parameter of shape ``[HV]``. Must be float32.
        a (torch.Tensor):
            Input-dependent decay of shape ``[B, 1, HV]``. Must be float16/bfloat16.
        dt_bias (torch.Tensor):
            Decay bias of shape ``[HV]``. Must be bfloat16 or float32.
        b (torch.Tensor):
            Update gate (beta) input of shape ``[B, 1, HV]``. Must be float16/bfloat16.
        scale (Optional[float]):
            Scale factor for queries. If None, defaults to ``1 / sqrt(K)``.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor of shape ``[B, 1, HV, V]``.
            If None, will be allocated automatically.
        use_qk_l2norm (bool):
            Whether to apply L2 normalization to q and k. Default: ``True``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - output: Output tensor of shape ``[B, 1, HV, V]``
            - state: Updated state tensor of shape ``[B, HV, K, V]``

    Note:
        - Requires SM90 (Hopper) architecture
        - State is updated in-place
        - K and V must be multiples of 4 for vectorized loads
        - State layout is k-major: [B, HV, K, V] (no transpose needed)
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
    disable_state_update: Optional[bool] = None,
    use_qk_l2norm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gated Delta Rule MTP Kernel (Multiple Token Processing).

    This function processes multiple tokens (T > 1) in sequence, typically used for
    speculative decoding verification. It supports intermediate state caching for
    potential rollback scenarios.

    Args:
        q (torch.Tensor):
            Query tensor of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Key tensor of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Value tensor of shape ``[B, T, HV, V]``.
        initial_state (torch.Tensor):
            Initial state tensor of shape ``[pool_size, HV, V, K]`` (K-last layout).
        initial_state_indices (torch.Tensor):
            Indices mapping each batch to its initial state, shape ``[B]``.
        A_log (torch.Tensor):
            Log decay parameter of shape ``[HV]``.
        a (torch.Tensor):
            Input-dependent decay of shape ``[B, T, HV]``.
        dt_bias (torch.Tensor):
            Decay bias of shape ``[HV]``.
        b (torch.Tensor):
            Update gate input of shape ``[B, T, HV]``.
        scale (Optional[float]):
            Scaling factor for queries. If None, uses ``1/sqrt(K)``.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor of shape ``[B, T, HV, V]``.
        intermediate_states_buffer (Optional[torch.Tensor]):
            Buffer for caching intermediate states, shape ``[pool_size, T, HV, V, K]``.
            If None, intermediate states are not cached.
        disable_state_update (Optional[bool]):
            If True, the initial state is not updated. Currently defaults to ``True``.
            Please pass this argument explicitly — the default will change to ``False``
            in FlashInfer 0.7.0.

            .. deprecated::
                The implicit default of ``True`` is deprecated and will change to
                ``False`` in version 0.7.0. Pass ``disable_state_update=True`` or
                ``disable_state_update=False`` explicitly to silence the warning.
        use_qk_l2norm (bool):
            Whether to apply L2 normalization to q and k. Default: ``True``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - output: Output tensor of shape ``[B, T, HV, V]``
            - initial_state: Updated state tensor (unchanged if disable_state_update=True)

    Note:
        - Requires SM90 (Hopper) architecture
        - Supports T > 1 (multiple token processing)
        - State layout is K-last: [pool_size, HV, V, K]
        - Optimized for speculative decoding verification scenarios
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

    # Set default scale
    if scale is None:
        scale = K**-0.5

    # Allocate output if not provided
    output_provided = output is not None
    target_dtype = output.dtype if output_provided else q.dtype

    if output is None:
        output = torch.zeros((B, T, HV, V), dtype=torch.bfloat16, device=q.device)

    # Reshape initial_state from [pool_size, HV, V, K] to [pool_size * HV, V, K]
    h0_source = initial_state.to(torch.float32).reshape(pool_size * HV, V, K)

    # Handle intermediate states
    cache_intermediate_states = intermediate_states_buffer is not None
    if cache_intermediate_states:
        buffer_size = intermediate_states_buffer.shape[0]
        cache_steps = intermediate_states_buffer.shape[1]

        # Validate buffer length matches query sequence length
        assert cache_steps >= T, (
            f"intermediate_states_buffer second dimension (cache_steps={cache_steps}) must be at least T={T} to prevent out-of-bounds indexing"
        )

        intermediate_states = (
            intermediate_states_buffer.to(torch.float32)
            .reshape(buffer_size * cache_steps * HV, V, K)
            .contiguous()
        )
    else:
        cache_steps = T
        intermediate_states = torch.zeros(1, 1, 1, dtype=torch.float32, device=q.device)

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
    )

    # Copy state back if needed (no sync needed - PyTorch handles stream ordering)
    # Only copy if state update is enabled AND initial_state was not contiguous
    # (if contiguous, reshape returns a view and kernel updated state in-place)
    if not disable_state_update and not initial_state.is_contiguous():
        initial_state.copy_(h0_source.reshape(pool_size, HV, V, K))

    # Convert output to target dtype if needed
    if output.dtype != target_dtype:
        output = output.to(target_dtype)

    return output, initial_state
