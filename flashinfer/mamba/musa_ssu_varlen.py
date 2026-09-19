"""Packed-varlen Mamba SSU provider for MUSA.

The public FlashInfer API uses the same packed ``x/dt/B/C`` and
``cu_seqlens`` contract as vLLM's Mamba2 decoder.  This module keeps the
recurrence on device, including the accepted-token initial-state selection,
so graph capture never synchronizes metadata to the host.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .musa_ssd_helpers import fast_exp
from .musa_stochastic import cvt_rs_f16, philox4x32


@triton.jit
def softplus(dt):
    # Keep the finite branch compatible with Triton 3.2 on MUSA.
    return tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)


@triton.jit
def _convert_rs_fp16x2(x, rand):
    return cvt_rs_f16(x, rand)


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"IS_SPEC_DECODING": lambda args: args["num_accepted_tokens_ptr"] is not None}
)
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens_ptr"] is not None})
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit(do_not_specialize=["N"])
def _musa_varlen_ssu_kernel(
    # Pointers to matrices
    state_ptr,
    rand_seed_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_batch_indices_ptr,
    dst_state_batch_indices_ptr,
    null_block_id,
    num_accepted_tokens_ptr,
    cu_seqlens_ptr,
    # Matrix dimensions
    N,
    nheads,
    dim,
    dstate,
    nheads_ngroups_ratio,
    # Strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    stride_state_indices_batch,
    stride_state_indices_T,
    stride_dst_state_indices_batch,
    stride_dst_state_indices_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    USE_RS_ROUNDING: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
        seq_len = eos - bos

        if seq_len == 0:
            return
    else:
        bos = pid_b
        seq_len = 1

    state_ptr_base = state_ptr

    # If HAS_STATE_BATCH_INDICES is true, then the ssm state's batch coordinate
    # is taken from the state_batch_indices_ptr Otherwise, the state coordinate
    # is the same as the batch id.
    if HAS_STATE_BATCH_INDICES:
        if IS_SPEC_DECODING:
            num_accepted = tl.load(num_accepted_tokens_ptr + pid_b).to(tl.int64)
            init_token_idx = tl.maximum(num_accepted - 1, 0)
        else:
            init_token_idx = 0

        dst_state_batch_indices_ptr += pid_b * stride_dst_state_indices_batch
        if not IS_SPEC_DECODING:
            dst_state_batch_idx = tl.load(
                dst_state_batch_indices_ptr
                + init_token_idx * stride_dst_state_indices_T
            ).to(tl.int64)
            dst_state_ptr = state_ptr + (
                dst_state_batch_idx * stride_state_batch + pid_h * stride_state_head
            )

        state_batch_indices_ptr += (
            pid_b * stride_state_indices_batch + init_token_idx * stride_state_indices_T
        )
        state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    else:
        dst_state_ptr = (
            state_ptr + pid_b * stride_state_batch + pid_h * stride_state_head
        )
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    x_ptr += bos * stride_x_batch + pid_h * stride_x_head
    dt_ptr += bos * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += bos * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += bos * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += bos * stride_z_batch + pid_h * stride_z_head
    out_ptr += bos * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    if not IS_SPEC_DECODING:
        dst_state_ptrs = dst_state_ptr + (
            offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        )

    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    if HAS_STATE_BATCH_INDICES:
        mask &= state_batch_idx != null_block_id
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D_ptrs = D_ptr + offs_m * stride_D_dim
    A_ptrs = A_ptr + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate

    for i_t in range(seq_len):
        x_ptrs = x_ptr + offs_m * stride_x_dim
        dt_ptrs = dt_ptr + offs_m * stride_dt_dim
        B_ptrs = B_ptr + offs_n * stride_B_dstate
        C_ptrs = C_ptr + offs_n * stride_C_dstate
        if HAS_Z:
            z_ptrs = z_ptr + offs_m * stride_z_dim
        out_ptrs = out_ptr + offs_m * stride_out_dim

        x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if not TIE_HDIM:
            dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(
                A_ptrs,
                mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
                other=0.0,
            ).to(tl.float32)
            dA = fast_exp(A * dt[:, None])
        else:
            dt = tl.load(dt_ptr).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(A_ptr).to(tl.float32)
            dA = fast_exp(A * dt)  # scalar, not a matrix

        B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        if HAS_D:
            D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_Z:
            z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

        dB = B[None, :] * dt[:, None] if not TIE_HDIM else B * dt
        state = state * dA + dB * x[:, None]

        if IS_SPEC_DECODING:
            dst_idx_ptr = dst_state_batch_indices_ptr + i_t * stride_dst_state_indices_T
            token_dst_idx = tl.load(dst_idx_ptr).to(tl.int64)
            if token_dst_idx != null_block_id:
                token_dst_ptrs = (
                    state_ptr_base
                    + token_dst_idx * stride_state_batch
                    + pid_h * stride_state_head
                    + offs_m[:, None] * stride_state_dim
                    + offs_n[None, :] * stride_state_dstate
                )
                tl.store(
                    token_dst_ptrs, state.to(token_dst_ptrs.dtype.element_ty), mask=mask
                )

        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D
        if HAS_Z:
            out *= z * tl.sigmoid(z)
        tl.store(out_ptrs, out, mask=offs_m < dim)

        x_ptr += stride_x_batch
        dt_ptr += stride_dt_batch
        B_ptr += stride_B_batch
        C_ptr += stride_C_batch
        out_ptr += stride_out_batch
        if HAS_Z:
            z_ptr += stride_z_batch

    if not IS_SPEC_DECODING:
        if USE_RS_ROUNDING:
            # Load random seed
            rand_seed = tl.load(rand_seed_ptr)
            # Generate random offsets for each element in state
            if HAS_STATE_BATCH_INDICES:
                rand_offsets = (
                    state_batch_idx * stride_state_batch + pid_h * stride_state_head
                )
            else:
                rand_offsets = pid_b * stride_state_batch + pid_h * stride_state_head
            rand_offsets += (
                offs_m[:, None] * stride_state_dim
                + offs_n[None, :] * stride_state_dstate
            )
            # Use the same explicit Philox4x32 lowering as the other MUSA
            # providers.  Triton 3.2 and 3.6 expose different tl.randint
            # signatures on MUSA; keeping the counter construction here makes
            # the capture path version-stable.
            r0, _, _, _ = philox4x32(
                rand_seed, rand_offsets, PHILOX_ROUNDS if PHILOX_ROUNDS > 0 else 10
            )
            rand = r0
            # Convert state to fp16 with RS rounding
            state = _convert_rs_fp16x2(state, rand)
            tl.static_assert(state.dtype == tl.float16, "state must be fp16")
            tl.static_assert(
                dst_state_ptrs.dtype.element_ty == tl.float16,
                "dst_state_ptrs must be fp16",
            )
        else:
            state = state.to(dst_state_ptrs.dtype.element_ty)
        tl.store(dst_state_ptrs, state, mask=mask)


def _select_block_config(dim: int, dstate: int, batch: int, nheads: int) -> tuple[int, int]:
    """Small deterministic launch table for the S5000 packed MTP shape."""
    # The Nemotron TP1 contract is D=64/N=128.  Four rows and four warps
    # match the existing MUSA decode provider and keep register pressure low.
    if dstate >= 128:
        return (4 if dim >= 64 else 8), 4
    if dstate >= 64:
        return (8 if dim >= 64 else 16), 4
    return (16 if dim >= 64 else 32), 2


def ssu_varlen_musa_triton(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    dst_state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = -1,
    out: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    enable_stochastic_rounding: bool = False,
    cache_philox_rounds: int = 0,
    rand_seed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run packed varlen Mamba2 SSU without host metadata reads.

    ``x/dt`` are ``[total_tokens, H, D]`` and ``B/C`` are
    ``[total_tokens, G, N]``.  ``state_batch_indices`` and its destination
    counterpart are ``[sequences, max_seqlen]`` for speculative decoding.
    """
    if cu_seqlens is None:
        raise ValueError("packed varlen SSU requires cu_seqlens")
    if state.dim() != 4 or x.dim() != 3 or dt.shape != x.shape:
        raise ValueError("invalid packed varlen state/x/dt shapes")
    if A.dim() != 3 or B.dim() != 3 or C.shape != B.shape:
        raise ValueError("invalid packed varlen A/B/C shapes")
    if D.dim() not in (1, 2):
        raise ValueError("D must be rank 1 or 2")
    if dt_bias is not None and dt_bias.dim() not in (1, 2):
        raise ValueError("dt_bias must be rank 1 or 2")
    if state_batch_indices is None:
        raise ValueError("packed varlen SSU requires state_batch_indices")
    if state_batch_indices.dim() == 1:
        state_batch_indices = state_batch_indices[:, None]
    if dst_state_batch_indices is None:
        dst_state_batch_indices = state_batch_indices
    elif dst_state_batch_indices.dim() == 1:
        dst_state_batch_indices = dst_state_batch_indices[:, None]
    if state_batch_indices.dim() != 2 or dst_state_batch_indices.dim() != 2:
        raise ValueError("state index tensors must be rank 2 in packed varlen mode")
    tensors = (x, dt, A, B, C, D, state_batch_indices, dst_state_batch_indices, cu_seqlens)
    if any(t.device != state.device for t in tensors):
        raise ValueError("packed varlen tensors must share the state device")
    if state_batch_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("state_batch_indices must be int32 or int64")
    if dst_state_batch_indices.dtype != state_batch_indices.dtype:
        raise ValueError("source and destination index dtypes must match")
    if out is None:
        out = torch.empty_like(x)
    if out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError("out must match x in packed varlen mode")
    if cu_seqlens.dim() != 1 or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be a contiguous rank-1 tensor")
    if cu_seqlens.device != state.device or cu_seqlens.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("cu_seqlens must be int32/int64 on the state device")
    nseq = cu_seqlens.numel() - 1
    if nseq < 1 or state_batch_indices.shape[0] < nseq:
        raise ValueError("cu_seqlens/index batch mismatch")
    if dst_state_batch_indices.shape[0] < nseq:
        raise ValueError("cu_seqlens/destination-index batch mismatch")
    if num_accepted_tokens is not None:
        if num_accepted_tokens.dim() != 1 or num_accepted_tokens.numel() < nseq:
            raise ValueError("num_accepted_tokens must have one entry per sequence")
        if num_accepted_tokens.device != state.device:
            raise ValueError("num_accepted_tokens must share the state device")
        if num_accepted_tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("num_accepted_tokens must be int32 or int64")
    if enable_stochastic_rounding and state.dtype != torch.float16:
        raise ValueError("stochastic rounding requires FP16 state")
    if enable_stochastic_rounding and cache_philox_rounds not in (5, 10):
        raise ValueError("stochastic rounding supports Philox 5 or 10 rounds")

    _, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    if x.shape[1:] != (nheads, dim) or B.shape[1:] != (ngroups, dstate):
        raise ValueError("packed varlen tensor dimensions do not match state")
    if A.shape != (nheads, dim, dstate):
        raise ValueError("A shape does not match state")
    if D.shape not in ((nheads,), (nheads, dim)):
        raise ValueError("D shape does not match state")
    if dt_bias is not None and dt_bias.shape not in ((nheads,), (nheads, dim)):
        raise ValueError("dt_bias shape does not match state")
    if z is not None and (z.shape != x.shape or z.device != state.device):
        raise ValueError("z shape does not match x")
    if nheads % ngroups:
        raise ValueError("nheads must be divisible by ngroups")
    if dt_bias is not None and dt_bias.device != state.device:
        raise ValueError("dt_bias must share the state device")

    block_m, num_warps = _select_block_config(dim, dstate, nseq, nheads)
    idx_strides = (state_batch_indices.stride(0), state_batch_indices.stride(1))
    dst_strides = (dst_state_batch_indices.stride(0), dst_state_batch_indices.stride(1))
    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    bias_strides = (dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None and dt_bias.dim() == 2 else ((dt_bias.stride(0), 0) if dt_bias is not None else (0, 0))
    d_strides = (D.stride(0), D.stride(1) if D.dim() == 2 else 0)
    if enable_stochastic_rounding:
        if rand_seed is None:
            rand_seed = torch.randint(
                0, 2**32, (1,), device=state.device, dtype=torch.int64
            )
        elif rand_seed.dtype != torch.int64 or rand_seed.numel() != 1:
            raise ValueError("rand_seed must be one int64 value")
    else:
        rand_seed = None
    grid = (triton.cdiv(dim, block_m), nseq, nheads)
    _musa_varlen_ssu_kernel[grid](
        state,
        rand_seed,
        x,
        dt,
        dt_bias,
        A,
        B,
        C,
        D,
        z,
        out,
        state_batch_indices,
        dst_state_batch_indices,
        null_block_id,
        num_accepted_tokens,
        cu_seqlens,
        x.shape[0],
        nheads,
        dim,
        dstate,
        nheads // ngroups,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        dt.stride(0),
        dt.stride(1),
        dt.stride(2),
        bias_strides[0],
        bias_strides[1],
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        d_strides[0],
        d_strides[1],
        z_strides[0],
        z_strides[1],
        z_strides[2],
        out.stride(0),
        out.stride(1),
        out.stride(2),
        idx_strides[0],
        idx_strides[1],
        dst_strides[0],
        dst_strides[1],
        dt_softplus,
        A.stride(-1) == 0 and A.stride(-2) == 0 and dt.stride(-1) == 0 and (dt_bias is None or dt_bias.stride(-1) == 0),
        block_m,
        num_warps=num_warps,
        USE_RS_ROUNDING=enable_stochastic_rounding,
        PHILOX_ROUNDS=cache_philox_rounds,
        HAS_DT_BIAS=dt_bias is not None,
        HAS_D=True,
        HAS_Z=z is not None,
        HAS_STATE_BATCH_INDICES=True,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        IS_VARLEN=True,
        BLOCK_SIZE_DSTATE=triton.next_power_of_2(dstate),
    )
    return out


__all__ = ["ssu_varlen_musa_triton"]
