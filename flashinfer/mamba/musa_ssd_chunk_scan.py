# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_scan.py

# ruff: noqa: E501,SIM102


import os

from packaging import version

from .musa_ssd_helpers import fast_exp
import triton
import triton.language as tl

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


def is_musa_triton_32():
    return triton.__version__.split(".")[:2] == ["3", "2"]


def _scan_config_override():
    """Return one explicit tile for isolated S5000 benchmark runs.

    The override is intentionally opt-in and evaluated before Triton's
    autotune decorator is built.  Production callers retain the existing
    version-gated configuration unless a benchmark process explicitly sets
    ``FLASHINFER_MUSA_SSD_SCAN_CONFIG`` to a value such as
    ``32x32x32x1x4``.  Keeping this hook in the provider lets us compare
    configs with identical Python wrappers and metadata, instead of copying
    the kernel into a separate benchmark-only implementation.
    """
    raw = os.environ.get("FLASHINFER_MUSA_SSD_SCAN_CONFIG", "").strip().lower()
    if not raw:
        return None
    fields = raw.replace("-", "x").split("x")
    if len(fields) != 5:
        raise ValueError(
            "FLASHINFER_MUSA_SSD_SCAN_CONFIG must be MxNxKxstagesxwarps"
        )
    try:
        m, n, k, stages, warps = (int(field) for field in fields)
    except ValueError as exc:
        raise ValueError(
            "FLASHINFER_MUSA_SSD_SCAN_CONFIG must be MxNxKxstagesxwarps"
        ) from exc
    if min(m, n, k, stages, warps) <= 0:
        raise ValueError("FLASHINFER_MUSA_SSD_SCAN_CONFIG values must be positive")
    return [
        triton.Config(
            {"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k},
            num_stages=stages,
            num_warps=warps,
        )
    ]


def _ssd_autotune_configs():
    override = _scan_config_override()
    if override is not None:
        return override
    # MUSA Triton 3.2.x cannot compile the broad upstream autotune search.
    # The 16x16x32 tile is retained, while four warps provide the latency
    # hiding measured on the Nemotron H=64, headdim=64, dstate=128 path. The
    # explicit environment override above remains available for shape-specific
    # replay until the full serving matrix is measured.

    if is_musa_triton_32():
        return [
            triton.Config(
                {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 32},
                num_stages=1,
                num_warps=4,
            )
        ]
    return [
        # =================================================================
        # Higher warp count configs for better latency hiding
        # More warps = more instructions in flight = better memory latency hiding
        # =================================================================
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=8,
        ),
        # Smaller tiles with more stages for software pipelining
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64},
            num_stages=2,
            num_warps=4,
        ),
        # =================================================================
        # Low register pressure configs (num_stages=1) for large dstate
        # =================================================================
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=1,
            num_warps=4,
        ),
        # num_stages=2 configs - moderate register pressure
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        # Original configs for larger dstate values
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
    ]


@triton.jit
def _load_previous_state(
    states_ptr,
    initstates_ptr,
    pid_c,
    pid_h,
    seq_idx,
    new_sequence,
    offs_n,
    offs_k,
    hdim,
    dstate,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_init_states_batch,
    stride_init_states_head,
    stride_init_states_hdim,
    stride_init_states_dstate,
    HAS_INITSTATES: tl.constexpr,
):
    mask = (offs_k[:, None] < dstate) & (offs_n[None, :] < hdim)
    if new_sequence:
        if HAS_INITSTATES:
            initial_ptrs = (
                initstates_ptr
                + seq_idx * stride_init_states_batch
                + pid_h * stride_init_states_head
                + offs_n[None, :] * stride_init_states_hdim
                + offs_k[:, None] * stride_init_states_dstate
            )
            value = tl.load(initial_ptrs, mask=mask, other=0.0).to(tl.float32)
        else:
            value = tl.full((offs_k.shape[0], offs_n.shape[0]), 0.0, tl.float32)
    else:
        chunk_ptrs = (
            states_ptr
            + (pid_c - 1) * stride_states_chunk
            + pid_h * stride_states_head
            + offs_n[None, :] * stride_states_hdim
            + offs_k[:, None] * stride_states_dstate
        )
        value = tl.load(chunk_ptrs, mask=mask, other=0.0).to(tl.float32)
    return value


@triton.autotune(
    configs=_ssd_autotune_configs(),
    key=["chunk_size", "hdim", "dstate", "IS_CAUSAL"],
)
@triton.jit
def _chunk_scan_fwd_kernel(
    # Pointers to matrices
    cb_ptr,
    x_ptr,
    z_ptr,
    out_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    seq_idx_ptr,
    C_ptr,
    states_ptr,
    D_ptr,
    initstates_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimensions
    chunk_size: tl.constexpr,
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides
    stride_cb_chunk: tl.int64,
    stride_cb_head: tl.int64,
    stride_cb_csize_m: tl.int64,
    stride_cb_csize_k: tl.constexpr,
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    stride_z_seqlen: tl.int64,
    stride_z_head: tl.int64,
    stride_z_hdim: tl.constexpr,
    stride_out_seqlen: tl.int64,
    stride_out_head: tl.int64,
    stride_out_hdim: tl.constexpr,
    stride_dt_chunk: tl.int64,
    stride_dt_head: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    stride_seq_idx_chunk: tl.constexpr,
    stride_C_seqlen: tl.int64,
    stride_C_head: tl.int64,
    stride_C_dstate: tl.constexpr,
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    stride_init_states_batch: tl.int64,
    stride_init_states_head: tl.int64,
    stride_init_states_hdim: tl.int64,
    stride_init_states_dstate: tl.constexpr,
    stride_D_head: tl.constexpr,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    IS_MUSA: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
):
    pid_c = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    cb_ptr += pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    x_ptr += chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += (
        chunk_seqlen_start * stride_C_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_C_head
    )

    # M-block offsets and prev states
    #  - logic in next block may override these if there is an active offset
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    seq_idx_ptr += pid_c * stride_seq_idx_chunk
    seq_idx = tl.load(seq_idx_ptr)
    seq_idx_prev = tl.load(
        seq_idx_ptr - stride_seq_idx_chunk, mask=pid_c >= 1, other=-1
    )

    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(
        dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0
    ).to(tl.float32)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    offs_k_dstate = tl.arange(
        0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K
    )
    C_ptrs = C_ptr + (
        offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate
    )

    scale_m = fast_exp(dA_cs_m)
    if BLOCK_SIZE_DSTATE <= 128:
        C = tl.load(
            C_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit)
            & (offs_k_dstate[None, :] < dstate),
            other=0.0,
        )

        prev_states = _load_previous_state(
            states_ptr,
            initstates_ptr,
            pid_c,
            pid_h,
            seq_idx,
            seq_idx != seq_idx_prev,
            offs_n,
            offs_k_dstate,
            hdim,
            dstate,
            stride_states_chunk,
            stride_states_head,
            stride_states_hdim,
            stride_states_dstate,
            stride_init_states_batch,
            stride_init_states_head,
            stride_init_states_hdim,
            stride_init_states_dstate,
            HAS_INITSTATES,
        ).to(C_ptr.dtype.element_ty)

        if IS_MUSA:
            acc = tl.dot(C.to(tl.float32), prev_states.to(tl.float32))
        else:
            acc = tl.dot(C, prev_states)
        acc *= scale_m[:, None]

    else:
        for k in range(0, dstate, BLOCK_SIZE_K):
            C = tl.load(
                C_ptrs,
                mask=(offs_m[:, None] < chunk_size_limit)
                & (offs_k_dstate[None, :] < dstate - k),
                other=0.0,
            )
            prev_states = _load_previous_state(
                states_ptr,
                initstates_ptr,
                pid_c,
                pid_h,
                seq_idx,
                seq_idx != seq_idx_prev,
                offs_n,
                offs_k_dstate + k,
                hdim,
                dstate,
                stride_states_chunk,
                stride_states_head,
                stride_states_hdim,
                stride_states_dstate,
                stride_init_states_batch,
                stride_init_states_head,
                stride_init_states_hdim,
                stride_init_states_dstate,
                HAS_INITSTATES,
            ).to(C_ptr.dtype.element_ty)
            if IS_MUSA:
                acc += tl.dot(C.to(tl.float32), prev_states.to(tl.float32))
            else:
                acc += tl.dot(C, prev_states)
            C_ptrs += BLOCK_SIZE_K
        acc *= scale_m[:, None]

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (
        offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k
    )
    x_ptrs = x_ptr + (
        offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = (
        chunk_size_limit
        if not IS_CAUSAL
        else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    )
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(
            cb_ptrs,
            mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(
            tl.float32
        )
        # If there's seq_idx, we already set cb[i, j] = 0 for seq_idx[i] != seq_idx[j].
        # So we don't need masking wrt seq_idx here.
        cb *= fast_exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= dt_k
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)
        x = tl.load(
            x_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim),
            other=0.0,
        )
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(
                D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0
            ).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(
            x_ptr
            + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc += x_residual * D

    if HAS_Z:
        z_ptr += chunk_seqlen_start * stride_z_seqlen + pid_h * stride_z_head
        z_ptrs = z_ptr + (
            stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :]
        )
        z = tl.load(
            z_ptrs,
            mask=(offs_out_m[:, None] < chunk_size_limit)
            & (offs_out_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    out_ptr += chunk_seqlen_start * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (
        stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim
    )
    tl.store(
        out_ptrs,
        acc,
        mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim),
    )


def _chunk_scan_fwd(
    cb,
    x,
    dt,
    dA_cumsum,
    C,
    states,
    cu_chunk_seqlens,
    out,
    seq_idx,
    D=None,
    z=None,
    initial_states=None,
):
    assert seq_idx is not None, "this implementation requires seq_idx"

    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (seqlen, ngroups, dstate)
    assert cb.shape == (nchunks, ngroups, chunk_size, chunk_size)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if z is not None:
        assert z.shape == x.shape
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    assert states.shape == (nchunks, nheads, headdim, dstate)
    assert seq_idx.shape == (nchunks,)

    grid = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        nchunks,
        nheads,
    )

    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    initial_states_strides = (
        (
            initial_states.stride(0),
            initial_states.stride(1),
            initial_states.stride(2),
            initial_states.stride(3),
        )
        if initial_states is not None
        else (0, 0, 0, 0)
    )

    # MUSA Triton 3.2.x requires optional pointer arguments to retain a
    # concrete pointer type even when the constexpr branch is disabled.
    initstates_ptr = initial_states
    if is_musa_triton_32() and initial_states is None:
        initstates_ptr = states
        initial_states_strides = states.stride()

    # Keep the BF16 dot lowering scoped to the affected MUSA Triton stack.
    is_musa = is_musa_triton_32()

    _chunk_scan_fwd_kernel[grid](
        cb_ptr=cb,
        x_ptr=x,
        z_ptr=z,
        out_ptr=out,
        dt_ptr=dt,
        dA_cumsum_ptr=dA_cumsum,
        seq_idx_ptr=seq_idx,
        C_ptr=C,
        states_ptr=states,
        D_ptr=D,
        initstates_ptr=initstates_ptr,
        cu_chunk_seqlens_ptr=cu_chunk_seqlens,
        chunk_size=chunk_size,
        hdim=headdim,
        dstate=dstate,
        nheads_ngroups_ratio=nheads // ngroups,
        stride_cb_chunk=cb.stride(0),
        stride_cb_head=cb.stride(1),
        stride_cb_csize_m=cb.stride(2),
        stride_cb_csize_k=cb.stride(3),
        stride_x_seqlen=x.stride(0),
        stride_x_head=x.stride(1),
        stride_x_hdim=x.stride(2),
        stride_z_seqlen=z_strides[0],
        stride_z_head=z_strides[1],
        stride_z_hdim=z_strides[2],
        stride_out_seqlen=out.stride(0),
        stride_out_head=out.stride(1),
        stride_out_hdim=out.stride(2),
        stride_dt_chunk=dt.stride(1),
        stride_dt_head=dt.stride(0),
        stride_dt_csize=dt.stride(2),
        stride_dA_cs_chunk=dA_cumsum.stride(1),
        stride_dA_cs_head=dA_cumsum.stride(0),
        stride_dA_cs_csize=dA_cumsum.stride(2),
        stride_seq_idx_chunk=seq_idx.stride(0),
        stride_C_seqlen=C.stride(0),
        stride_C_head=C.stride(1),
        stride_C_dstate=C.stride(2),
        stride_states_chunk=states.stride(0),
        stride_states_head=states.stride(1),
        stride_states_hdim=states.stride(2),
        stride_states_dstate=states.stride(3),
        stride_init_states_batch=initial_states_strides[0],
        stride_init_states_head=initial_states_strides[1],
        stride_init_states_hdim=initial_states_strides[2],
        stride_init_states_dstate=initial_states_strides[3],
        stride_D_head=D.stride(0) if D is not None else 0,
        IS_CAUSAL=True,
        IS_MUSA=is_musa,
        HAS_D=D is not None,
        D_HAS_HDIM=D.dim() == 2 if D is not None else True,
        HAS_Z=z is not None,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        IS_TRITON_22=TRITON_22,
        HAS_INITSTATES=initial_states is not None,
    )
    return
