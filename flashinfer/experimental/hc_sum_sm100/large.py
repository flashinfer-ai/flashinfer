# SPDX-License-Identifier: Apache-2.0
# Derived from sgl-project/sglang at 9cdd6624fe115979b0a0e6c588d9de5ddfeaafe2.
# Modified 2026-11-04: standalone FlashInfer backend; remove framework dependencies.

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _shuffle_inv_pair(values, source, ODD: gl.constexpr):
    # Layout is explicitly Blocked([2],[32],[4],[0]), not inferred from pack.
    # Each lane owns even/odd logical elements in two consecutive registers.
    # A logical group of32 elements occupies16 physical lanes. 'source' indexes
    # that group; branch parity selects the register, source//2 selects its lane.
    if ODD:
        result = gl.inline_asm_elementwise(
            asm="""
            {
                .reg .b32 lane, half, s0, s1;
                mov.u32 lane, %laneid;
                and.b32 half, lane, 16;
                shr.u32 s0, $4, 1;
                shr.u32 s1, $5, 1;
                or.b32 s0, s0, half;
                or.b32 s1, s1, half;
                shfl.sync.idx.b32 $0, $3, s0, 31, -1;
                shfl.sync.idx.b32 $1, $3, s1, 31, -1;
            }
            """,
            constraints="=&r,=&r,r,r,r,r",
            args=[values.to(gl.int32, bitcast=True), source],
            dtype=gl.int32,
            is_pure=True,
            pack=2,
        )
    else:
        result = gl.inline_asm_elementwise(
            asm="""
            {
                .reg .b32 lane, half, s0, s1;
                mov.u32 lane, %laneid;
                and.b32 half, lane, 16;
                shr.u32 s0, $4, 1;
                shr.u32 s1, $5, 1;
                or.b32 s0, s0, half;
                or.b32 s1, s1, half;
                shfl.sync.idx.b32 $0, $2, s0, 31, -1;
                shfl.sync.idx.b32 $1, $2, s1, 31, -1;
            }
            """,
            constraints="=&r,=&r,r,r,r,r",
            args=[values.to(gl.int32, bitcast=True), source],
            dtype=gl.int32,
            is_pure=True,
            pack=2,
        )
    return result.to(gl.float32, bitcast=True)


@gluon.jit(do_not_specialize=["BATCH_STRIDE"])
def _hc_down_batched_sum_post_kernel(
    partial,
    sums,
    down,
    alpha,
    numel,
    H: gl.constexpr,
    LOWRANK: gl.constexpr,
    C: gl.constexpr,
    NP: gl.constexpr,
    BATCH_STRIDE,
    WIDE_INDEX: gl.constexpr,
    EPS: gl.constexpr,
    BLOCK: gl.constexpr,
):
    gl.static_assert(C == 4)
    gl.static_assert(LOWRANK > 0)
    gl.static_assert(BLOCK == 256)
    layout: gl.constexpr = gl.BlockedLayout([2], [32], [4], [0])
    n: gl.constexpr = LOWRANK + C
    # Only index width specializes on T, not the exact branch stride. The host
    # includes tail lanes in its range check; global address products stay wide.
    index_dtype: gl.constexpr = gl.int64 if WIDE_INDEX else gl.int32
    x = gl.program_id(0).to(index_dtype) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    live = x < numel
    row = x // n
    col = x % n

    # n>=5: <=8 intersecting token rows per32-output logical group,4 slots each.
    group_start = (x // 32) * 32
    first_row = group_start // n
    last_row = (group_start + 31) // n
    carrier_row = first_row + (x % 32) // C
    carrier_branch = x % C
    carrier_live = (carrier_row <= last_row) & (carrier_row.to(gl.int64) * n < numel)
    ss = gl.load(
        sums + carrier_row.to(gl.int64) * C + carrier_branch, carrier_live, other=0.0
    )
    carrier_inv = gl.rsqrt(ss * (1.0 / H) + EPS)

    row64 = row.to(gl.int64)
    # Contiguous [C,T,NP] rows remain vector-aligned for every runtime T.
    batch_stride = gl.multiple_of(BATCH_STRIDE.to(gl.int64), NP & -NP)
    total = gl.full((BLOCK,), 0.0, gl.float32, layout)
    for branch in gl.static_range(C):
        # Shuffle lane IDs are local to a 32-value group, not global offsets.
        source = ((row - first_row) * C + branch).to(gl.int32)
        inv = _shuffle_inv_pair(carrier_inv, source, branch % 2)
        p = gl.load(
            partial + row64 * NP + col + branch * batch_stride,
            live,
            other=0.0,
        )
        # Keep FP32 FMA rounding; pair packing must not emit FMUL2 + FADD.
        total = gl.fma(p, inv, total)
    raw = total.to(down.dtype.element_ty).to(gl.float32)
    scaled = raw * (1.0 / C)
    sigmoid = 1 / (1 + gl.exp(-scaled))
    gl.store(down + row64 * LOWRANK + col, scaled * sigmoid, live & (col < LOWRANK))
    gl.store(alpha + row64 * C + col - LOWRANK, 2.0 * sigmoid, live & (col >= LOWRANK))


@gluon.jit
def _hc_mix_branch_inv(values, branch: gl.constexpr):
    # BlockedLayout([2], [32], [4], [0]): consecutive register pairs
    # belong to one token. Lanes 0..3 carry the four branch inverses.
    source = gl.full(values.shape, branch, gl.int32, values.type.layout)
    bits = gl.inline_asm_elementwise(
        asm="{ shfl.sync.idx.b32 $0, $2, $4, 31, -1; mov.b32 $1, $0; }",
        constraints="=&r,=r,r,r,r,r",
        args=[values.to(gl.int32, bitcast=True), source],
        dtype=gl.int32,
        is_pure=True,
        pack=2,
    )
    return bits.to(gl.float32, bitcast=True)


@gluon.jit
def _hc_mix_from_sum_kernel(
    logits,
    residual,
    sums,
    norm_weight,
    mixed,
    next_sums,
    rows,
    H: gl.constexpr,
    EPS: gl.constexpr,
    BLOCK: gl.constexpr,
):
    # Each warp covers 64 hidden coordinates within one token;
    # warp shuffle shares its four lane-carried inverses.
    gl.static_assert(H > 0 and H % 512 == 0)
    gl.static_assert(BLOCK == 256)
    layout: gl.constexpr = gl.BlockedLayout([2], [32], [4], [0])
    # H is divisible by BLOCK, so a CTA never straddles tokens. Compute local
    # coordinates in int32 without forming the potentially overflowing pid*BLOCK.
    pid = gl.program_id(0)
    lane_offset = gl.arange(0, BLOCK, layout=layout)
    row = pid // (H // BLOCK)
    hidden = (pid % (H // BLOCK)) * BLOCK + lane_offset
    row64 = row.to(gl.int64)
    live = row < rows
    carrier = (lane_offset // 2) % 4
    ss = gl.load(sums + row64 * 4 + carrier, live, other=0.0)
    # CURRENT and NEXT sums do not alias. Clear NEXT before the Mix math;
    # exactly one owner writes each (token, branch), before the later Apply.
    for branch in gl.static_range(4):
        gl.store(next_sums + row64 * 4 + branch, 0.0, live & (pid % (H // BLOCK) == 0))
    inv_carrier = gl.rsqrt(ss * (1.0 / H) + EPS)
    # Shared [H,4] RMS layout: one LDG128 loads two hidden coordinates per lane.
    # Repack their eight BF16/FP16 values into per-branch register pairs.
    w0, w1, w2, w3 = gl.inline_asm_elementwise(
        asm="""{
            .reg .b32 p0, p1, p2, p3;
            ld.global.v4.b32 {p0, p1, p2, p3}, [$4];
            prmt.b32 $0, p0, p2, 0x5410;
            prmt.b32 $1, p0, p2, 0x7632;
            prmt.b32 $2, p1, p3, 0x5410;
            prmt.b32 $3, p1, p3, 0x7632;
        }""",
        constraints="=r,=r,=r,=r,l,l",
        args=[norm_weight + hidden * 4],
        dtype=(
            norm_weight.dtype.element_ty,
            norm_weight.dtype.element_ty,
            norm_weight.dtype.element_ty,
            norm_weight.dtype.element_ty,
        ),
        is_pure=True,
        pack=2,
    )
    total = gl.full((BLOCK,), 0.0, gl.float32, layout)
    for branch in gl.static_range(4):
        offset = (row64 * 4 + branch) * H + hidden
        inv = _hc_mix_branch_inv(inv_carrier, branch)
        r = gl.load(residual + offset, live, other=0.0).to(gl.float32)
        if branch == 0:
            weight = w0
        elif branch == 1:
            weight = w1
        elif branch == 2:
            weight = w2
        else:
            weight = w3
        gamma = 1.0 + weight.to(gl.float32)
        norm = ((r * inv) * gamma).to(residual.dtype.element_ty).to(gl.float32)
        logit = gl.load(logits + offset, live, other=0.0).to(gl.float32)
        # Preserve the ordinary low-precision pointwise tensor boundaries.
        gate = (
            (1.0 / (1.0 + gl.exp(-logit))).to(residual.dtype.element_ty).to(gl.float32)
        )
        product = (gate * norm).to(residual.dtype.element_ty).to(gl.float32)
        total += product
    gl.store(mixed + row64 * H + hidden, total * 0.25, live)


def run_large(residual, sum_sq, plan, workspace):
    """Preserve the validated five-stage path without residual/weight copies."""
    rows = residual.shape[0]
    operand = residual.view(rows, 4, plan.hidden_size).transpose(0, 1)
    matrix = plan.packed_down.transpose(1, 2)
    if workspace.deterministic:
        from .deterministic import bmm

        bmm(operand, matrix, workspace.partial)
    else:
        torch.bmm(operand, matrix, out_dtype=torch.float32, out=workspace.partial)
    n = rows * (plan.lowrank + 4)
    _hc_down_batched_sum_post_kernel[(triton.cdiv(n, 256),)](
        workspace.partial,
        sum_sq,
        workspace.down,
        workspace.alpha,
        n,
        H=plan.hidden_size,
        LOWRANK=plan.lowrank,
        C=4,
        NP=matrix.shape[-1],
        BATCH_STRIDE=workspace.partial.stride(0),
        WIDE_INDEX=triton.cdiv(n, 256) * 256 > 2**31,
        EPS=plan.rms_eps,
        BLOCK=256,
        num_warps=4,
    )
    torch.mm(workspace.down, plan.up.T, out=workspace.logits)
    _hc_mix_from_sum_kernel[(triton.cdiv(rows * plan.hidden_size, 256),)](
        workspace.logits,
        residual,
        sum_sq,
        plan.norm,
        workspace.mixed,
        workspace.sum_sq,
        rows,
        H=plan.hidden_size,
        EPS=plan.rms_eps,
        BLOCK=256,
        num_warps=4,
        enable_fp_fusion=False,
    )
