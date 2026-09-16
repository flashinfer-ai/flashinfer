"""Fused MUSA Triton selective state update kernel.

This kernel intentionally covers the hot decode contract first: one token per
request, tied-dimension dt, floating-point state, and no bias/softplus/z gate.
Other contracts continue to use the correctness reference provider.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .musa_stochastic import cvt_rs_f16, philox4x32


@triton.jit
def _ssu_one_token_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    dt_bias_ptr,
    z_ptr,
    src_slot_ptr,
    dst_slot_ptr,
    out_ptr,
    rand_seed_ptr,
    state_slot_stride,
    state_h_stride,
    state_d_stride,
    state_n_stride,
    x_b_stride,
    x_h_stride,
    x_d_stride,
    dt_b_stride,
    dt_h_stride,
    dt_d_stride,
    a_h_stride,
    a_d_stride,
    a_n_stride,
    b_b_stride,
    b_g_stride,
    b_n_stride,
    c_b_stride,
    c_g_stride,
    c_n_stride,
    d_h_stride,
    d_d_stride,
    bias_h_stride,
    bias_d_stride,
    out_b_stride,
    out_h_stride,
    out_d_stride,
    z_b_stride,
    z_h_stride,
    z_d_stride,
    pad_slot_id,
    H: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BIAS_IS_MATRIX: tl.constexpr,
    HAS_Z: tl.constexpr,
    SOFTPLUS: tl.constexpr,
    D_IS_VECTOR: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CACHE_SLOTS: tl.constexpr,
    USE_SR: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    SR_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    hd = H * D
    batch = pid // hd
    rem = pid % hd
    head = rem // D
    dim = rem % D
    group = head * G // H
    src_slot = tl.load(src_slot_ptr + batch).to(tl.int64)
    dst_slot = tl.load(dst_slot_ptr + batch).to(tl.int64)
    src_valid = (src_slot >= 0) & (src_slot < CACHE_SLOTS) & (src_slot != pad_slot_id)
    dst_valid = (dst_slot >= 0) & (dst_slot < CACHE_SLOTS) & (dst_slot != pad_slot_id)
    n = tl.arange(0, BLOCK_N)
    mask = n < N
    state_offset = (
        src_slot * state_slot_stride + head * state_h_stride + dim * state_d_stride
    )
    dst_state_offset = (
        dst_slot * state_slot_stride + head * state_h_stride + dim * state_d_stride
    )
    a_offset = head * a_h_stride + dim * a_d_stride
    bc_offset = batch * b_b_stride + group * b_g_stride
    s = tl.load(
        state_ptr + state_offset + n * state_n_stride, mask=mask & src_valid, other=0.0
    ).to(tl.float32)
    a = tl.load(a_ptr + a_offset + n * a_n_stride, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + bc_offset + n * b_n_stride, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(
        c_ptr + batch * c_b_stride + group * c_g_stride + n * c_n_stride,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x = tl.load(x_ptr + batch * x_b_stride + head * x_h_stride + dim * x_d_stride).to(
        tl.float32
    )
    dt = tl.load(
        dt_ptr + batch * dt_b_stride + head * dt_h_stride + dim * dt_d_stride
    ).to(tl.float32)
    if HAS_BIAS:
        bias_offset = (
            head * bias_h_stride + dim * bias_d_stride
            if BIAS_IS_MATRIX
            else head * bias_h_stride
        )
        dt += tl.load(dt_bias_ptr + bias_offset).to(tl.float32)
    if SOFTPLUS:
        dt = tl.where(dt > 20.0, dt, tl.log(1.0 + tl.exp(dt)))
    updated = s * tl.exp(a * dt) + (dt * b) * x
    stored_state = updated
    if USE_SR:
        seed = tl.load(rand_seed_ptr).to(tl.uint64)
        offset = state_offset + (n // SR_GROUP * SR_GROUP) * state_n_stride
        r0, r1, r2, r3 = philox4x32(seed, offset, PHILOX_ROUNDS)
        lane = n % SR_GROUP
        random_word = tl.where(
            lane == 0, r0, tl.where(lane == 1, r1, tl.where(lane == 2, r2, r3))
        )
        stored_state = cvt_rs_f16(updated, random_word).to(tl.float32)
    tl.store(
        state_ptr + dst_state_offset + n * state_n_stride,
        stored_state,
        mask=mask & dst_valid,
    )
    y = tl.sum(c * updated, axis=0)
    d_offset = (
        head * d_h_stride + dim * d_d_stride if not D_IS_VECTOR else head * d_h_stride
    )
    d = tl.load(d_ptr + d_offset).to(tl.float32)
    y += d * x
    if HAS_Z:
        z = tl.load(
            z_ptr + batch * z_b_stride + head * z_h_stride + dim * z_d_stride
        ).to(tl.float32)
        y *= z / (1.0 + tl.exp(-z))
    tl.store(
        out_ptr + batch * out_b_stride + head * out_h_stride + dim * out_d_stride, y
    )


def ssu_one_token_musa_triton(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    state_batch_indices: torch.Tensor,
    dst_state_batch_indices: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    pad_slot_id: int = -1,
    out: torch.Tensor | None = None,
    rand_seed: torch.Tensor | None = None,
    philox_rounds: int = 10,
) -> torch.Tensor:
    """Run the fused MUSA decode kernel for the supported contract."""
    batch, heads, dim = x.shape
    groups = B.shape[1]
    dstate = A.shape[-1]
    if rand_seed is not None:
        if state.dtype != torch.float16 or dstate not in (64, 128, 256):
            raise ValueError(
                "stochastic fused SSU requires FP16 state and dstate 64/128/256"
            )
        if (
            rand_seed.dtype != torch.int64
            or rand_seed.numel() != 1
            or rand_seed.device != state.device
        ):
            raise ValueError("rand_seed must be one int64 value on the state device")
        if philox_rounds not in (5, 10):
            raise ValueError("stochastic fused SSU supports Philox 5 or 10 rounds")
    block_n = triton.next_power_of_2(dstate)
    if dst_state_batch_indices is None:
        dst_state_batch_indices = state_batch_indices
    if dst_state_batch_indices.shape != state_batch_indices.shape:
        raise ValueError("MUSA fused SSU source/destination slot shapes differ")
    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError("MUSA fused SSU out must match x shape and dtype")
    if (
        any(t.dim() != 3 for t in (x, dt, B, C, out))
        or state.dim() != 4
        or A.dim() != 3
    ):
        raise ValueError(
            "MUSA fused SSU expects x/dt/B/C/out rank 3, state rank 4, A rank 3"
        )
    if D.dim() not in (1, 2) or (dt_bias is not None and dt_bias.dim() not in (1, 2)):
        raise ValueError("MUSA fused SSU D and dt_bias must be rank 1 or 2")
    _ssu_one_token_kernel[(batch * heads * dim,)](
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias if dt_bias is not None else x,
        z if z is not None else x,
        state_batch_indices,
        dst_state_batch_indices,
        out,
        rand_seed if rand_seed is not None else x,
        state_slot_stride=state.stride(0),
        state_h_stride=state.stride(1),
        state_d_stride=state.stride(2),
        state_n_stride=state.stride(3),
        x_b_stride=x.stride(0),
        x_h_stride=x.stride(1),
        x_d_stride=x.stride(2),
        dt_b_stride=dt.stride(0),
        dt_h_stride=dt.stride(1),
        dt_d_stride=dt.stride(2),
        a_h_stride=A.stride(0),
        a_d_stride=A.stride(1),
        a_n_stride=A.stride(2),
        b_b_stride=B.stride(0),
        b_g_stride=B.stride(1),
        b_n_stride=B.stride(2),
        c_b_stride=C.stride(0),
        c_g_stride=C.stride(1),
        c_n_stride=C.stride(2),
        d_h_stride=D.stride(0),
        d_d_stride=D.stride(1) if D.dim() == 2 else 0,
        bias_h_stride=dt_bias.stride(0) if dt_bias is not None else 0,
        bias_d_stride=dt_bias.stride(1)
        if dt_bias is not None and dt_bias.dim() == 2
        else 0,
        out_b_stride=out.stride(0),
        out_h_stride=out.stride(1),
        out_d_stride=out.stride(2),
        z_b_stride=z.stride(0) if z is not None else 0,
        z_h_stride=z.stride(1) if z is not None else 0,
        z_d_stride=z.stride(2) if z is not None else 0,
        pad_slot_id=pad_slot_id,
        H=heads,
        D=dim,
        N=dstate,
        G=groups,
        HAS_BIAS=dt_bias is not None,
        BIAS_IS_MATRIX=dt_bias is not None and dt_bias.dim() == 2,
        HAS_Z=z is not None,
        SOFTPLUS=dt_softplus,
        D_IS_VECTOR=D.dim() == 1,
        BLOCK_N=block_n,
        CACHE_SLOTS=state.shape[0],
        USE_SR=rand_seed is not None,
        PHILOX_ROUNDS=philox_rounds,
        SR_GROUP=2 if dstate == 64 else 4,
    )
    return out


__all__ = ["ssu_one_token_musa_triton"]
