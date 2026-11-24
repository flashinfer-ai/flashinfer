# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Narrow GQA decode with warp FP8 MMA and a split-KV softmax reduction."""

import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._layouts import (
    BlockedLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    SliceLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit
def _decode(
    Q,
    K,
    V,
    SF,
    TAB,
    LENS,
    PTR,
    OUT,
    TMP,
    ks: gl.constexpr,
    vs: gl.constexpr,
    ss: gl.constexpr,
    ts: gl.constexpr,
    qs: gl.constexpr,
    os: gl.constexpr,
    scale,
    vscale,
    H: gl.constexpr,
    G: gl.constexpr,
    PAGE: gl.constexpr,
    D: gl.constexpr,
    N: gl.constexpr,
    S: gl.constexpr,
    W: gl.constexpr,
):
    split, head, req = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    tok = gl.load(PTR + req)
    if gl.load(PTR + req + 1) == tok:
        return
    length = gl.load(LENS + req)
    tiles = gl.cdiv(length, N)
    chunk = gl.cdiv(tiles, S)
    L: gl.constexpr = BlockedLayout([1, 4], [4, 8], [W, 1], [1, 0])
    M: gl.constexpr = NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[1, W], instr_shape=[16, 8]
    )
    A: gl.constexpr = DotOperandLayout(0, M, k_width=4)
    B: gl.constexpr = DotOperandLayout(1, M, k_width=4)
    row = gl.arange(0, 16, layout=gl.constexpr(SliceLayout(1, L)))
    x = gl.arange(0, D, layout=gl.constexpr(SliceLayout(0, L)))
    q = gl.load(
        Q + tok.to(gl.int64) * qs[0] + (head * G + row[:, None]) * qs[1] + x[None, :],
        row[:, None] < G,
        0.0,
    )
    a = gl.convert_layout(q, A)
    m = gl.full(
        (16,), -float("inf"), gl.float32, layout=gl.constexpr(SliceLayout(1, L))
    )
    z = gl.full((16,), 0, gl.float32, layout=gl.constexpr(SliceLayout(1, L)))
    acc = gl.full((16, D), 0, gl.float32, layout=gl.constexpr(L))
    for tile in range(split * chunk, gl.minimum(tiles, (split + 1) * chunk)):
        r = gl.arange(0, N, layout=gl.constexpr(SliceLayout(1, L)))
        n = tile * N + r
        if PAGE % N == 0:
            pg = gl.load(TAB + req * ts + tile // (PAGE // N)).to(gl.int64)
        else:
            pg = gl.load(TAB + req * ts + n // PAGE, n < length, 0).to(gl.int64)
        token = n % PAGE
        kb = pg * ks[0] + head * ks[1] + token * ks[2]
        k = gl.load(K + kb[:, None] + x[None, :], n[:, None] < length, 0.0)
        dot = mma_v2(
            a,
            gl.convert_layout(gl.permute(k, (1, 0)), B),
            gl.full((16, N), 0, gl.float32, layout=gl.constexpr(M)),
        )
        y = gl.convert_layout(dot, gl.constexpr(L)) * scale
        col = gl.arange(0, N, layout=gl.constexpr(SliceLayout(0, L)))
        y = gl.where(tile * N + col[None, :] < length, y, -float("inf"))
        nm = gl.maximum(m, gl.max(y, 1))
        alpha = gl.exp2((m - nm) * 1.4426950408889634)
        p = gl.exp2((y - nm[:, None]) * 1.4426950408889634)
        z = z * alpha + gl.sum(p, 1)
        quart = gl.arange(0, D // 4, layout=gl.constexpr(SliceLayout(0, L)))
        vbase = pg * vs[0] + head * vs[1] + token * vs[2]
        raw = gl.load(V + vbase[:, None] + quart[None, :], n[:, None] < length, 0)
        group = quart // 4
        sr = (token // 4 * 4)[:, None] + group[None, :] // (D // 64)
        sc = (group % (D // 64) * 4)[None, :] + (token % 4)[:, None]
        sb = pg * ss[0] + head * ss[1] + token * 0
        sf = gl.load(SF + sb[:, None] + sr * ss[2] + sc, n[:, None] < length, 0)
        word = gl.inline_asm_elementwise(
            "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 $0, $1, $2;",
            "=r,h,r",
            [raw, sf.to(gl.uint32) * 0x01010101],
            dtype=gl.uint32,
            is_pure=True,
            pack=1,
        )
        ev = gl.join(word.to(gl.uint8), (word >> 16).to(gl.uint8))
        od = gl.join((word >> 8).to(gl.uint8), (word >> 24).to(gl.uint8))
        val = gl.reshape(gl.join(ev, od), (N, D)).to(gl.float8e4nv, bitcast=True)
        pv = mma_v2(
            gl.convert_layout((p * 448.0).to(gl.float8e4nv), A),
            gl.convert_layout(val, B),
            gl.full((16, D), 0, gl.float32, layout=gl.constexpr(M)),
        )
        acc = acc * alpha[:, None] + gl.convert_layout(pv, gl.constexpr(L)) * (
            1.0 / 448.0
        )
        m = nm
    if S == 1:
        gl.store(
            OUT
            + tok.to(gl.int64) * os[0]
            + (head * G + row[:, None]) * os[1]
            + x[None, :],
            acc * (vscale / gl.where(z > 0, z, 1.0))[:, None],
            row[:, None] < G,
        )
    else:
        base = ((tok.to(gl.int64) * H + head * G + row) * S + split) * (D + 2)
        gl.store(TMP + base[:, None] + x[None, :], acc, row[:, None] < G)
        gl.store(TMP + base + D, m, row < G)
        gl.store(TMP + base + D + 1, z, row < G)


@triton.jit
def _merge_split(
    P,
    OUT,
    PTR,
    v_scale,
    o_str: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    NREQ: tl.constexpr,
):
    t, h = tl.program_id(0), tl.program_id(1)
    if t >= tl.load(PTR + NREQ):
        return
    base = ((t.to(tl.int64) * tl.num_programs(1) + h) * S + tl.arange(0, S)) * (D + 2)
    m = tl.load(P + base + D)
    denom = tl.load(P + base + D + 1)
    max_m = tl.max(m, axis=0)
    w = tl.exp(m - tl.where(max_m == -float("inf"), 0.0, max_m))
    den = tl.sum(w * denom, axis=0)
    d = tl.arange(0, D)
    acc = tl.sum(tl.load(P + base[:, None] + d[None, :]) * w[:, None], axis=0)
    tl.store(
        OUT + t.to(tl.int64) * o_str[0] + h * o_str[1] + d,
        acc * v_scale / tl.where(den > 0, den, 1.0),
    )


def run_decode(q, k, v, sf, table, lens, ptr, out, scale, vscale):
    import torch

    heads, d = q.shape[1:]
    hkv, page = k.shape[1:3]
    n, s, w = 32, 32, 2
    tmp = torch.empty(
        (q.shape[0], heads, s, d + 2), device=q.device, dtype=torch.float32
    )
    v16 = v.view(torch.uint16)
    with torch.cuda.device(q.device):
        _decode[(s, hkv, table.shape[0])](
            q,
            k,
            v16,
            sf.view(torch.uint8),
            table,
            lens,
            ptr,
            out,
            tmp,
            k.stride(),
            v16.stride(),
            sf.stride(),
            table.stride(0),
            q.stride(),
            out.stride(),
            scale,
            vscale,
            heads,
            heads // hkv,
            page,
            d,
            n,
            s,
            w,
            num_warps=w,
            num_stages=1,
            ptx_version=94,
        )
        _merge_split[(q.shape[0], heads)](
            tmp, out, ptr, vscale, out.stride(), s, d, table.shape[0]
        )
    return out
