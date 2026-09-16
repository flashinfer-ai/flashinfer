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
import os
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.mamba import selective_state_update_trace
from ..jit.mamba import (
    gen_selective_state_update_module,
    gen_selective_state_update_sm100_module,
    gen_selective_state_update_sm90_module,
)
from ..utils import get_compute_capability, register_custom_op, register_fake_op


@functools.cache
def _get_module(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
    cu_seqlens_dtype: torch.dtype,
    num_accepted_tokens_dtype: torch.dtype,
    sm_major: int,
    state_scale_dtype: Optional[torch.dtype] = None,
    philox_rounds: int = 0,
):
    args = (
        state_dtype,
        input_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        state_scale_dtype,
        dim,
        dstate,
        ntokens_mtp,
        cu_seqlens_dtype,
        num_accepted_tokens_dtype,
        philox_rounds,
    )
    if sm_major >= 10:
        return gen_selective_state_update_sm100_module(*args).build_and_load()
    elif sm_major >= 9:
        return gen_selective_state_update_sm90_module(*args).build_and_load()
    else:
        return gen_selective_state_update_module(*args).build_and_load()


def get_selective_state_update_module(
    device: torch.device,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
    cu_seqlens_dtype: torch.dtype,
    num_accepted_tokens_dtype: torch.dtype,
    state_scale_dtype: Optional[torch.dtype] = None,
    philox_rounds: int = 0,
):
    major, _ = get_compute_capability(device)
    return _get_module(
        state_dtype,
        input_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        dim,
        dstate,
        ntokens_mtp,
        cu_seqlens_dtype,
        num_accepted_tokens_dtype,
        major,
        state_scale_dtype,
        philox_rounds,
    )


@flashinfer_api(trace=selective_state_update_trace)
def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    state_batch_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    state_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    intermediate_state_scales: Optional[torch.Tensor] = None,
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 10,
    cache_steps: int = 0,
    algorithm: str = "auto",
    dst_state_batch_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    backend: str = "auto",
) -> torch.Tensor:
    r"""Selective state update operation for Mamba layers (the generation phase).

    Parameters
    ----------
    state : torch.Tensor
        State tensor with shape (state_cache_size, dim, dstate) or (state_cache_size, nheads, dim, dstate)
    x : torch.Tensor
        Input tensor with shape (batch, dim) or (batch, nheads, dim) for single-token,
        (batch, T, nheads, dim) for multi-token,
        or (total_tokens, nheads, dim) for varlen multi-token (with cu_seqlens)
    dt : torch.Tensor
        Delta time tensor, same layout as x
    A : torch.Tensor
        A matrix with shape (dim, dstate) or (nheads, dim, dstate)
    B : torch.Tensor
        B matrix with shape (batch, dstate) or (batch, ngroups, dstate) for single-token,
        (batch, T, ngroups, dstate) for multi-token,
        or (total_tokens, ngroups, dstate) for varlen multi-token
    C : torch.Tensor
        C matrix, same layout as B
    D : torch.Tensor
        D vector with shape (dim,) or (nheads, dim)
    z : Optional[torch.Tensor]
        Optional z tensor, same layout as x
    dt_bias : Optional[torch.Tensor]
        Optional dt bias with shape (dim,) or (nheads, dim)
    dt_softplus : bool
        Whether to apply softplus to dt
    state_batch_indices : Optional[torch.Tensor]
        Batch indices for state cache reading. Shape (batch,) or (N, max_seqlen).
        For speculative decoding with num_accepted_tokens, must be 2D.
    dst_state_batch_indices : Optional[torch.Tensor]
        Destination indices for state cache writing. Shape (batch,) or (N, max_seqlen).
        When provided, state is read from state_batch_indices and written to
        dst_state_batch_indices (enables separate read/write state slots).
    pad_slot_id : int
        Sentinel value for padded entries in state_batch_indices
    state_scale : Optional[torch.Tensor]
        Optional float32 scale tensor with shape (state_cache_size, nheads, dim)
        for int16 state quantization with block scaling. Listed in the custom
        op's ``mutates_args``: when ``state`` is quantized (int16), the kernel
        writes the new per-block scales here in place — the caller dequantizes
        ``state`` against this tensor on read-back (mirrors the
        ``intermediate_state_scales`` contract below).
    out : Optional[torch.Tensor]
        Optional output tensor (same shape as x)
    disable_state_update : bool
        If True, skip updating the state tensor (useful for speculative decoding verification)
    intermediate_states_buffer : Optional[torch.Tensor]
        Optional buffer for caching intermediate states during speculative decoding
        with shape (batch, cache_steps, nheads, dim, dstate). Also listed in
        ``mutates_args`` — the kernel writes intermediate states into it
        in place.
    intermediate_state_indices : Optional[torch.Tensor]
        Optional indices mapping batch elements to intermediate state buffer positions
        with shape (batch,)
    intermediate_state_scales : Optional[torch.Tensor]
        Optional per-block float32 scale tensor matching ``intermediate_states_buffer``.
        When provided alongside an int16 ``intermediate_states_buffer``, the kernel
        writes the computed scales into this tensor (it is listed in the custom
        op's ``mutates_args``), and the caller is responsible for *dequantizing*
        the intermediate states using these scales when reading them back.
        Mirrors the ``state_scale`` layout but for the speculative-decoding
        intermediate buffer.
    rand_seed : Optional[torch.Tensor]
        Optional single-element int64 CUDA tensor for stochastic rounding seed.
    philox_rounds : int
        Number of Philox-4x32 PRNG rounds for stochastic rounding (default 10).
    cache_steps : int
        Number of steps/tokens to cache for speculative decoding.
        For varlen mode (cu_seqlens provided), this specifies max_seqlen.
    cu_seqlens : Optional[torch.Tensor]
        Cumulative sequence lengths with shape (N + 1,), integer dtype
        (the JIT specializes on the actual dtype, so int32 or int64 is fine;
        int32 is the default when omitted). When provided, inputs are in
        packed-token varlen format: ``x`` / ``dt`` are 3-D
        ``(total_tokens, nheads, dim)``, ``B`` / ``C`` are 3-D
        ``(total_tokens, ngroups, dstate)``, with sequence boundaries given
        by ``cu_seqlens``.
    num_accepted_tokens : Optional[torch.Tensor]
        Number of accepted tokens per sequence with shape (N,).
        Determines which state to read as initial state for each sequence.
    algorithm : str
        Algorithm to use: "auto", "simple", "vertical", "horizontal"
    backend : str
        Backend to use: "auto", "flashinfer", or "cake". Both "auto" and
        "flashinfer" use FlashInfer. Cake is an opt-in source-built backend on
        SM100/SM103; backend="cake" falls back to FlashInfer outside Cake's
        promoted rows.

    Returns
    -------
    output : torch.Tensor
        Output tensor with same shape as x
    """
    is_varlen = cu_seqlens is not None and x.dim() == 3
    is_mtp = cache_steps >= 1 and not is_varlen

    if state.dim() == 3:
        state = state.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if D.dim() == 1:
        D = D.unsqueeze(0)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)

    if not is_varlen:
        # Handle x, dt, B, C, z dimensions based on mode
        # For single-token: 2D -> 3D (batch, nheads, dim)
        # For multi-token: 3D -> 4D (batch, T, nheads, dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if is_mtp and x.dim() == 3:
            # Add T dimension for MTP mode: (batch, nheads, dim) -> (batch, T, nheads, dim)
            x = x.unsqueeze(1)

        if dt.dim() == 2:
            dt = dt.unsqueeze(1)
        if is_mtp and dt.dim() == 3:
            dt = dt.unsqueeze(1)

        if B.dim() == 2:
            B = B.unsqueeze(1)
        if is_mtp and B.dim() == 3:
            B = B.unsqueeze(1)

        if C.dim() == 2:
            C = C.unsqueeze(1)
        if is_mtp and C.dim() == 3:
            C = C.unsqueeze(1)

        if z is not None:
            if z.dim() == 2:
                z = z.unsqueeze(1)
            if is_mtp and z.dim() == 3:
                z = z.unsqueeze(1)

    # Normalize state_scale to 3D: (state_cache_size, nheads, dim)
    if state_scale is not None and state_scale.dim() == 4 and state_scale.size(-1) == 1:
        state_scale = state_scale.squeeze(-1)

    # Validate rand_seed and philox_rounds
    if rand_seed is not None:
        if not isinstance(rand_seed, torch.Tensor):
            raise TypeError(
                f"rand_seed must be a CUDA/MUSA int64 tensor, got {type(rand_seed).__name__}"
            )
        if rand_seed.numel() != 1:
            raise ValueError(
                f"rand_seed must be a single-element tensor, got numel={rand_seed.numel()}"
            )
        if rand_seed.dtype != torch.int64:
            raise ValueError(f"rand_seed must have dtype int64, got {rand_seed.dtype}")
        if rand_seed.device.type not in ("cuda", "musa"):
            raise ValueError("rand_seed must be a CUDA or MUSA tensor")
        if state_scale is not None:
            raise ValueError("rand_seed and state_scale cannot both be provided")
        if philox_rounds <= 0:
            raise ValueError(
                f"philox_rounds must be > 0 when rand_seed is provided, got {philox_rounds}"
            )
    else:
        # No stochastic rounding when rand_seed is None
        philox_rounds = 0

    if intermediate_states_buffer is not None and dst_state_batch_indices is not None:
        raise ValueError(
            "intermediate_states_buffer and dst_state_batch_indices are mutually exclusive"
        )

    # A one-token decode may carry the v1 query-start metadata [0, 1].  The
    # native Simple STP kernel consumes already-flattened tensors and must not
    # silently ignore a different pair (which would turn a malformed offset
    # into an out-of-bounds reference fallback).  Multi-token varlen metadata
    # is handled by the varlen provider below.
    if (
        cu_seqlens is not None
        and x.dim() == 3
        and x.shape[0] == 1
        and cu_seqlens.numel() == 2
        and [int(v) for v in cu_seqlens.detach().cpu().tolist()] != [0, 1]
    ):
        raise ValueError(
            "single-token cu_seqlens must be the canonical [0, 1] pair; "
            "use a valid multi-token varlen request for other offsets"
        )

    if out is None:
        output = torch.empty_like(x)
    else:
        output = out

    # MUSA has a separate provider boundary.  Do not send MUSA tensors through
    # the CUDA JIT module: its launchers include CUDA runtime headers and its
    # architecture dispatch is NVIDIA-SM specific.  The initial MUSA provider
    # is deliberately a correctness scaffold; the native MUSA kernel will keep
    # this exact call boundary when it lands.
    if state.device.type == "musa":
        if algorithm not in (
            "auto",
            "simple",
            "vertical",
            "horizontal",
            "async_horizontal",
        ):
            raise ValueError(f"unknown MUSA SSU algorithm={algorithm!r}")
        fused_state_batch_indices = state_batch_indices
        fused_dst_state_batch_indices = dst_state_batch_indices
        fused_pad_slot_id = -1 if pad_slot_id is None else int(pad_slot_id)
        state_indices_are_one_token = state_batch_indices is not None and (
            state_batch_indices.dim() == 1
            or (state_batch_indices.dim() == 2 and state_batch_indices.shape[1] == 1)
        )
        dst_indices_are_one_token = (
            dst_state_batch_indices is None
            or dst_state_batch_indices.dim() == 1
            or (
                dst_state_batch_indices.dim() == 2
                and dst_state_batch_indices.shape[1] == 1
            )
        )
        if (
            x.dim() == 3
            and state_batch_indices is not None
            and state_batch_indices.dim() == 2
            and state_batch_indices.shape[1] == 1
        ):
            fused_state_batch_indices = state_batch_indices[:, -1].contiguous()
        if (
            x.dim() == 3
            and dst_state_batch_indices is not None
            and dst_state_batch_indices.dim() == 2
            and dst_state_batch_indices.shape[1] == 1
        ):
            fused_dst_state_batch_indices = dst_state_batch_indices[:, -1].contiguous()
        if (
            (z is None or (z.dim() == 3 and z.shape == x.shape))
            and (
                dt_bias is None
                or (dt_bias.dim() == 1 and dt_bias.shape[0] == x.shape[1])
                or (dt_bias.dim() == 2 and dt_bias.shape == x.shape[1:3])
            )
            and (
                fused_state_batch_indices is not None
                and (
                    fused_dst_state_batch_indices is None
                    or fused_dst_state_batch_indices.shape
                    == fused_state_batch_indices.shape
                )
            )
            and intermediate_states_buffer is None
            and (
                rand_seed is None
                or (
                    state.dtype == torch.float16
                    and state.shape[-1] in (64, 128, 256)
                    and rand_seed.dtype == torch.int64
                    and rand_seed.numel() == 1
                    and rand_seed.device == state.device
                    and philox_rounds in (5, 10)
                    and algorithm in ("auto", "simple")
                )
            )
            and num_accepted_tokens is None
            and not disable_state_update
            # vLLM's decode path supplies a two-entry query_start_loc even
            # for one token. The Simple-STP kernel consumes the already
            # flattened token tensors, so that metadata is safe to ignore.
            and (
                cu_seqlens is None
                or (
                    x.shape[0] == 1
                    and cu_seqlens.numel() == 2
                    and int(cu_seqlens[0].item()) == 0
                    and int(cu_seqlens[1].item()) == 1
                )
            )
            and state.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and state.dim() == 4
            and x.dim() == 3
            and state.shape[1:3] == x.shape[1:3]
            and state_indices_are_one_token
            and dst_indices_are_one_token
            and state_batch_indices.numel() == x.shape[0]
            and (
                dst_state_batch_indices is None
                or dst_state_batch_indices.numel() == x.shape[0]
            )
            and dt.dim() == 3
            and A.dim() == 3
            and B.dim() == 3
            and C.dim() == 3
            and dt.shape == x.shape
            and A.shape[:2] == x.shape[1:3]
            and B.shape == C.shape
            and B.shape[0] == x.shape[0]
            and B.shape[2] == A.shape[2]
            and D is not None
            and (
                (D.dim() == 1 and D.shape[0] == x.shape[1])
                or (D.dim() == 2 and D.shape == x.shape[1:3])
            )
            and fused_state_batch_indices is not None
            and fused_state_batch_indices.dim() == 1
            and x.shape[1] % B.shape[1] == 0
        ):
            # S5000's low-parallel Nemotron shape maps to the NVIDIA SM90
            # Simple STP contract: four D rows per 128-thread CTA. Keep this
            # shape-gated so every other MUSA shape retains the generic
            # Triton provider.
            native_stp_contract = (
                x.shape[0] == 1
                and x.shape[1:] == (64, 64)
                and state.shape[-1] == 128
                and B.shape[1] == 8
                and state.dtype == torch.float16
                and x.dtype == torch.bfloat16
                and B.dtype == x.dtype
                and C.dtype == x.dtype
                and dt.dtype == torch.float32
                and A.dtype == torch.float32
                and D.dtype in (torch.float32, x.dtype)
                and state.is_contiguous()
                and all(t.device == state.device for t in (x, dt, A, B, C, D))
                and fused_state_batch_indices.device == state.device
                and (
                    fused_dst_state_batch_indices is None
                    or fused_dst_state_batch_indices.device == state.device
                )
                and (z is None or (z.dtype == x.dtype and z.device == state.device))
                and (out is None or (out.dtype == x.dtype and out.device == state.device))
                and (
                    dt_bias is None
                    or (dt_bias.dtype == torch.float32 and dt_bias.device == state.device)
                )
                and A.stride(1) == 0
                and A.stride(2) == 0
                and dt.stride(2) == 0
                and (
                    dt_bias is None
                    or dt_bias.dim() == 1
                    or dt_bias.stride(1) == 0
                )
            )
            if native_stp_contract:
                if os.environ.get("FLASHINFER_MUSA_SIMPLE_STP_NATIVE") == "1":
                    from .musa_ssu_native import musa_ssu_one_token_native

                    return musa_ssu_one_token_native(
                        state,
                        x,
                        dt,
                        A,
                        B,
                        C,
                        D,
                        fused_state_batch_indices,
                        fused_dst_state_batch_indices
                        if fused_dst_state_batch_indices is not None
                        else fused_state_batch_indices,
                        dt_bias,
                        z,
                        dt_softplus,
                        fused_pad_slot_id,
                        out,
                        rand_seed,
                        philox_rounds,
                    )

                from .musa_ssu_simple import ssu_one_token_musa_simple

                return ssu_one_token_musa_simple(
                    state,
                    x,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    fused_state_batch_indices,
                    dt_bias=dt_bias,
                    z=z,
                    dt_softplus=dt_softplus,
                    pad_slot_id=fused_pad_slot_id,
                    dst_state_batch_indices=fused_dst_state_batch_indices,
                    out=out,
                    rand_seed=rand_seed,
                    philox_rounds=philox_rounds,
                )

            from .musa_ssu_triton import ssu_one_token_musa_triton

            return ssu_one_token_musa_triton(
                state,
                x,
                dt,
                A,
                B,
                C,
                D,
                fused_state_batch_indices,
                dt_bias=dt_bias,
                z=z,
                dt_softplus=dt_softplus,
                pad_slot_id=fused_pad_slot_id,
                dst_state_batch_indices=fused_dst_state_batch_indices,
                out=out,
                rand_seed=rand_seed,
                philox_rounds=philox_rounds,
            )
        # The correctness provider has one recurrence implementation.  The
        # algorithm value remains accepted so callers can use the upstream
        # API while native vertical/horizontal kernels are added.
        from .musa_reference import selective_state_update_musa_reference

        return selective_state_update_musa_reference(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            dt_softplus,
            state_batch_indices,
            dst_state_batch_indices,
            pad_slot_id,
            output,
            disable_state_update,
            intermediate_states_buffer,
            intermediate_state_indices,
            state_scale,
            intermediate_state_scales,
            rand_seed,
            philox_rounds,
            cache_steps,
            cu_seqlens,
            num_accepted_tokens,
        )

    # Determine stateIndex dtype from index tensors, default to int32
    stateIndex_dtype = torch.int32
    if state_batch_indices is not None:
        stateIndex_dtype = state_batch_indices.dtype
    elif dst_state_batch_indices is not None:
        stateIndex_dtype = dst_state_batch_indices.dtype
    elif intermediate_state_indices is not None:
        stateIndex_dtype = intermediate_state_indices.dtype

    # Extract dim/dstate/ntokens for JIT specialization
    dim = state.size(2)
    dstate = state.size(3)
    if is_varlen:
        ntokens_mtp = cache_steps
    elif x.dim() == 4:
        ntokens_mtp = x.size(1)
    else:
        ntokens_mtp = 1

    if algorithm == "auto":
        algorithm_int = 0
    elif algorithm == "simple":
        algorithm_int = 1
    elif algorithm == "vertical":
        algorithm_int = 2
    elif algorithm == "horizontal":
        algorithm_int = 3
    elif algorithm == "async_horizontal":
        # Backward compat: async_horizontal is now merged into simple
        algorithm_int = 1
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if backend not in {"auto", "flashinfer", "cake"}:
        raise ValueError(f"Unknown backend: {backend}")
    if backend == "cake":
        from ..jit.mamba.cake_selective_state_update import (
            try_cake_selective_state_update,
        )

        if try_cake_selective_state_update(
            state=state,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            output=output,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            pad_slot_id=pad_slot_id,
            disable_state_update=disable_state_update,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            state_scale=state_scale,
            intermediate_state_scales=intermediate_state_scales,
            rand_seed=rand_seed,
            cache_steps=cache_steps,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
            algorithm=algorithm,
            dt_softplus=dt_softplus,
        ):
            return output

    _selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        dst_state_batch_indices,
        pad_slot_id,
        state_scale,
        output,
        disable_state_update,
        intermediate_states_buffer,
        intermediate_state_indices,
        intermediate_state_scales,
        rand_seed,
        cache_steps,
        cu_seqlens,
        num_accepted_tokens,
        algorithm_int,
        philox_rounds,
        state.dtype,
        x.dtype,
        dt.dtype,
        A.dtype,
        stateIndex_dtype,
        dim,
        dstate,
        ntokens_mtp,
    )
    return output


@register_custom_op(
    "flashinfer::selective_state_update",
    mutates_args=(
        "state",
        "output",
        "intermediate_states_buffer",
        "state_scale",
        "intermediate_state_scales",
    ),
)
def _selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    dst_state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    state_scale: Optional[torch.Tensor],
    output: torch.Tensor,
    disable_state_update: bool,
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    intermediate_state_scales: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    cache_steps: int,
    cu_seqlens: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    algorithm: int,
    philox_rounds: int,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
) -> None:
    """Internal function registered with torch.library for torch.compile() support."""
    get_selective_state_update_module(
        state.device,
        state_dtype,
        input_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        dim,
        dstate,
        ntokens_mtp,
        cu_seqlens.dtype if cu_seqlens is not None else torch.int32,
        num_accepted_tokens.dtype if num_accepted_tokens is not None else torch.int64,
        state_scale_dtype=state_scale.dtype if state_scale is not None else None,
        philox_rounds=philox_rounds,
    ).selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        dst_state_batch_indices,
        pad_slot_id,
        state_scale,
        output,
        disable_state_update,
        intermediate_states_buffer,
        intermediate_state_indices,
        intermediate_state_scales,
        rand_seed,
        cache_steps,
        cu_seqlens,
        num_accepted_tokens,
        algorithm,
    )


@register_fake_op("flashinfer::selective_state_update")
def _selective_state_update_fake(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    dst_state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    state_scale: Optional[torch.Tensor],
    output: torch.Tensor,
    disable_state_update: bool,
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    intermediate_state_scales: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    cache_steps: int,
    cu_seqlens: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    algorithm: int,
    philox_rounds: int,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
) -> None:
    """Fake implementation for torch.compile() meta tensor propagation."""
    pass


# Build/import the opt-in native extension before vLLM starts graph capture.
# Generic Triton/CUDA imports never execute this branch.
if os.environ.get("FLASHINFER_MUSA_SIMPLE_STP_NATIVE") == "1":
    from .musa_ssu_native import preload_musa_simple_stp

    preload_musa_simple_stp()
