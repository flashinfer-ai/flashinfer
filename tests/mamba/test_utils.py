from typing import Any, Dict, Optional

import numpy as np
import torch


def clone_preserving_strides(tensor):
    """Clone a tensor while preserving its strides (non-contiguous layout)."""
    result = torch.empty_strided(
        tensor.size(), tensor.stride(), dtype=tensor.dtype, device=tensor.device
    )
    result.copy_(tensor)
    return result


def create_test_inputs(
    batch_size: int,
    nheads: int,
    dim: int,
    dstate: int,
    ngroups: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype = torch.float32,
    state_dtype: torch.dtype = torch.bfloat16,
    matrixA_dtype: torch.dtype = torch.float32,
    generate_z: bool = False,
    generate_intermediate_states_buffer: bool = False,
    cache_steps: Optional[int] = None,
    generate_retrieve_parent_token: bool = False,
    state_cache_batch_stride: Optional[int] = None,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Create test inputs for selective_state_update functions.

    This function generates all necessary tensors for testing selective state
    update kernels, supporting both single-token and multi-token (speculative
    decoding) scenarios.

    Arguments:
        batch_size: Number of sequences in the batch.
        nheads: Number of attention heads.
        dim: Head dimension (headdim).
        dstate: SSM state size.
        ngroups: Number of groups for B and C matrices.
        input_dtype: Data type for input tensors (x, B, C, z) - from model config.json (typically bf16).
        weight_dtype: Data type for weight tensors (D, dt, dt_bias) - hardcoded fp32 in mamba2_mixer.py.
        state_dtype: Data type for state tensor - user configurable (bf16/fp16/fp32). Defaults to input_dtype.
        matrixA_dtype: Data type for the A matrix - hardcoded fp32 in mamba2_mixer.py.
        generate_z: If True, generate z tensor for gating.
        generate_intermediate_states_buffer: If True, generate buffer for
            caching intermediate states during speculative decoding.
        cache_steps: Number of steps/tokens to cache. Required if
            generate_intermediate_states_buffer is True. Also determines
            T dimension when > 1 (multi-token mode).
        generate_retrieve_parent_token: If True, generate tensor for EAGLE
            tree attention parent token retrieval.
        state_cache_batch_stride: Optional batch stride for ssm_state_cache.
            If None, defaults to contiguous stride (nheads * dim * dstate).
            Must be >= nheads * dim * dstate if specified.
        device: Device to create tensors on.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing all generated tensors with the following keys:
        - state_cache: (total_entries, nheads, dim, dstate)
        - x: (batch_size, [T,] nheads, dim) - T present if cache_steps provided
        - dt: (batch_size, [T,] nheads, dim) - T present if cache_steps provided
        - A: (nheads, dim, dstate)
        - B: (batch_size, [T,] ngroups, dstate) - T present if cache_steps provided
        - C: (batch_size, [T,] ngroups, dstate) - T present if cache_steps provided
        - D: (nheads, dim)
        - dt_bias: (nheads, dim)
        - slot_idx: (batch_size,)
        - z: (batch_size, [T,] nheads, dim) - only if generate_z=True, T present if cache_steps provided
        - intermediate_states_buffer: (batch_size, cache_steps, nheads, dim, dstate)
            - only if generate_intermediate_states_buffer=True
        - intermediate_slot_idx: (batch_size,)
            - only if generate_intermediate_states_buffer=True
        - retrieve_parent_token: (batch_size, T)
            - only if generate_retrieve_parent_token=True
        - cache_steps: int - only if cache_steps is provided
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine if we're in multi-token mode
    # Always use 4D tensors when cache_steps is provided (even for cache_steps=1)
    T = cache_steps if cache_steps is not None else None

    # If we use the cache, then the state indices are taken from a specific slot
    # so the state in the kernel will have batch as the first dimension, but it will
    # only come from a particular slot; the full tensor first dim is larger
    ssm_state_cache_size = max(384, batch_size * 10)

    # State dtype defaults to input_dtype if not specified

    # SSM state cache: (total_entries, nheads, dim, dstate)
    # Calculate the contiguous batch stride
    contiguous_batch_stride = nheads * dim * dstate

    # Use provided batch stride or default to contiguous
    if state_cache_batch_stride is None:
        state_cache_batch_stride = contiguous_batch_stride

    # Validate that batch stride is large enough
    if state_cache_batch_stride < contiguous_batch_stride:
        raise ValueError(
            f"state_cache_batch_stride ({state_cache_batch_stride}) must be >= "
            f"contiguous stride ({contiguous_batch_stride} = nheads * dim * dstate)"
        )

    total_elements = ssm_state_cache_size * state_cache_batch_stride
    state_cache_flat = torch.randn(total_elements, dtype=state_dtype, device=device)
    state_cache = state_cache_flat.as_strided(
        (ssm_state_cache_size, nheads, dim, dstate),
        (state_cache_batch_stride, dim * dstate, dstate, 1),
    )

    # Input x: (batch_size, [T,] nheads, dim)
    if T is not None:
        x = torch.randn(batch_size, T, nheads, dim, device=device, dtype=input_dtype)
    else:
        x = torch.randn(batch_size, nheads, dim, dtype=input_dtype, device=device)

    # dt: (batch_size, [T,] nheads, dim) with strides that broadcast dim
    # dt uses weight_dtype (fp32) as per mamba2_mixer.py
    # dt has T dimension for multi-token mode, matching x shape
    if T is not None:
        dt_base = torch.randn(batch_size, T, nheads, dtype=weight_dtype, device=device)
        dt = dt_base.as_strided(
            (batch_size, T, nheads, dim), (T * nheads, nheads, 1, 0)
        )
    else:
        dt_base = torch.randn(batch_size, nheads, dtype=weight_dtype, device=device)
        dt = dt_base.as_strided((batch_size, nheads, dim), (nheads, 1, 0))

    # A matrix: (nheads, dim, dstate) with strides (1, 0, 0) - one value per head
    # A should be negative for stability
    A_base = -torch.rand(nheads, dtype=matrixA_dtype, device=device) - 1.0
    A = A_base.as_strided((nheads, dim, dstate), (1, 0, 0))

    # B: (batch_size, T, ngroups, dstate)
    # C: (batch_size, ngroups, dstate)
    if T is not None:
        B = torch.randn(
            batch_size, T, ngroups, dstate, device=device, dtype=input_dtype
        )
        C = torch.randn(
            batch_size, T, ngroups, dstate, device=device, dtype=input_dtype
        )
    else:
        B = torch.randn(batch_size, ngroups, dstate, dtype=input_dtype, device=device)
        C = torch.randn(batch_size, ngroups, dstate, dtype=input_dtype, device=device)

    # D: (nheads, dim) with strides (1, 0) - one value per head
    D_base = torch.randn(nheads, dtype=weight_dtype, device=device)
    D = D_base.as_strided((nheads, dim), (1, 0))

    # dt_bias: (nheads, dim) with strides (1, 0) - one value per head
    dt_bias_base = torch.rand(nheads, dtype=weight_dtype, device=device) - 4.0
    dt_bias = dt_bias_base.as_strided((nheads, dim), (1, 0))

    # Slot indices for state batching - (batch_size,)
    slot_idx = torch.randperm(ssm_state_cache_size, dtype=torch.int64, device=device)[
        :batch_size
    ]

    # Build result dictionary
    result = {
        "state_cache": state_cache,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "dt_bias": dt_bias,
        "slot_idx": slot_idx,
    }

    # Optional: z tensor for gating
    # z: (batch_size, [T,] nheads, dim) - has T dimension for multi-token mode, matching x shape
    if generate_z:
        if T is not None:
            z = torch.randn(
                batch_size, T, nheads, dim, dtype=input_dtype, device=device
            )
        else:
            z = torch.randn(batch_size, nheads, dim, dtype=input_dtype, device=device)
        result["z"] = z

    # Optional: intermediate states buffer for speculative decoding
    if generate_intermediate_states_buffer:
        if cache_steps is None:
            raise ValueError(
                "cache_steps must be provided when generate_intermediate_states_buffer=True"
            )
        intermediate_states_buffer = torch.zeros(
            batch_size,
            cache_steps,
            nheads,
            dim,
            dstate,
            dtype=state_dtype,
            device=device,
        )
        result["intermediate_states_buffer"] = intermediate_states_buffer
        result["cache_steps"] = cache_steps
        # Also generate indices mapping batch elements to intermediate state buffer positions
        intermediate_slot_idx = torch.arange(
            batch_size, dtype=torch.int64, device=device
        )
        result["intermediate_slot_idx"] = intermediate_slot_idx

    # Optional: retrieve_parent_token for EAGLE tree attention
    if generate_retrieve_parent_token:
        if T is None or T <= 1:
            raise ValueError(
                "cache_steps > 1 required when generate_retrieve_parent_token=True"
            )
        # Create a simple linear chain structure by default
        # Token 0: parent = -1 (initial state)
        # Token t: parent = t - 1 (previous token)
        retrieve_parent_token = torch.zeros(
            batch_size, T, dtype=torch.int64, device=device
        )
        retrieve_parent_token[:, 0] = -1  # First token uses initial state
        for t in range(1, T):
            retrieve_parent_token[:, t] = t - 1
        result["retrieve_parent_token"] = retrieve_parent_token

    return result
