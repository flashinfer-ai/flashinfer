"""Experimental dense paged MLA decode for SM120, isolated from FlashInfer/SGLang.

All modes share the same algorithm and take BF16 Q. mode=0: BF16 KV/BF16 MMA;
mode=1: FP8 KV, fused dequantization to BF16 MMA; mode=2: FP8 QK and PV MMA
with FP32 softmax, per-head query scales and per-token KV scales.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_rows(
    X,
    Y,
    Scale,
    N: tl.constexpr,
    D: tl.constexpr,
    SX: tl.constexpr,
    SY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    x = tl.load(X + r * SX + d, d < D, 0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), 0), 1e-12) / 448.0
    tl.store(Y + r * SY + d, (x / s).to(Y.dtype.element_ty), d < D)
    tl.store(Scale + r, s)


def quantize_rows(x, out=None, scales=None):
    assert x.shape[-1] == 576 and x.stride(-1) == 1
    rows = x.numel() // 576
    if out is None:
        out = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    if scales is None:
        scales = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
    _quantize_rows[(rows,)](x, out, scales, rows, 576, 576, 576, 1024)
    return out, scales


@triton.jit
def _dequantize(X, S, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + i, i < N, 0.0).to(tl.float32)
    s = tl.load(S + i // 576, i < N, 1)
    tl.store(Y + i, v * s, i < N)


def dequantize(x, scales, out):
    _dequantize[(triton.cdiv(x.numel(), 1024),)](x, scales, out, x.numel(), 1024)
    return out


@triton.jit
def _mla_split(
    Q,
    QS,
    KV,
    KS,
    Pages,
    Lengths,
    Partial,
    LSE,
    HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    SPLITS: tl.constexpr,
    MODE: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr = 32,
):
    b, split = tl.program_id(0), tl.program_id(1)
    h = tl.program_id(2) * BM + tl.arange(0, BM)
    dc = tl.arange(0, 512)
    dr = tl.arange(0, 64)
    nk = tl.arange(0, BN)
    qc = tl.load(
        Q + b * HEADS * 576 + h[:, None] * 576 + dc[None, :], h[:, None] < HEADS, 0.0
    )
    qr = tl.load(
        Q + b * HEADS * 576 + h[:, None] * 576 + 512 + dr[None, :],
        h[:, None] < HEADS,
        0.0,
    )
    if MODE == 2:
        sq = tl.load(QS + b * HEADS + h, h < HEADS, 1)
    length = tl.load(Lengths + b)
    split_len = tl.cdiv(tl.cdiv(length, BN), SPLITS) * BN
    begin = split * split_len
    end = tl.minimum(begin + split_len, length)
    maximum = tl.full((BM,), -1.0e30, tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, 512), tl.float32)
    for start in range(begin, end, BN):
        n = start + nk
        valid = n < end
        page = tl.load(Pages + b * TABLE_STRIDE + n // PAGE_SIZE, valid, 0)
        physical = page * PAGE_SIZE + n % PAGE_SIZE
        kc = tl.load(KV + physical[:, None] * 576 + dc[None, :], valid[:, None], 0.0)
        kr = tl.load(
            KV + physical[:, None] * 576 + 512 + dr[None, :], valid[:, None], 0.0
        )
        if MODE != 0:
            sk = tl.load(KS + physical, valid, 1)
        if MODE == 1:
            kc = (kc.to(tl.float32) * sk[:, None]).to(tl.bfloat16)
            kr = (kr.to(tl.float32) * sk[:, None]).to(tl.bfloat16)
        scores = tl.dot(qc, tl.trans(kc)) + tl.dot(qr, tl.trans(kr))
        if MODE == 2:
            scores = scores * sq[:, None] * sk[None, :]
        scores = scores * (0.0625 * 1.4426950408889634)
        scores = tl.where(valid[None, :], scores, -1.0e30)
        mnew = tl.maximum(maximum, tl.max(scores, axis=1))
        alpha = tl.exp2(maximum - mnew)
        p = tl.where(valid[None, :], tl.exp2(scores - mnew[:, None]), 0.0)
        denominator = denominator * alpha + tl.sum(p, axis=1)
        if MODE == 2:
            # Each token's KV scale must be applied again in the PV product.
            p_scaled = p * sk[None, :]
            sp = tl.maximum(tl.max(tl.abs(p_scaled), axis=1), 1e-12) / 448.0
            p8 = (p_scaled / sp[:, None]).to(tl.float8e4nv)
            acc = acc * alpha[:, None] + tl.dot(p8, kc) * sp[:, None]
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kc)
        maximum = mnew
    inv = 1.0 / tl.maximum(denominator, 1e-30)
    o = acc * inv[:, None]
    lse = tl.where(denominator > 0, maximum + tl.log2(denominator), -float("inf"))
    tl.store(
        Partial + ((b * SPLITS + split) * HEADS + h[:, None]) * 512 + dc[None, :],
        o,
        h[:, None] < HEADS,
    )
    tl.store(LSE + (b * SPLITS + split) * HEADS + h, lse, h < HEADS)


@triton.jit
def _merge(
    Partial,
    LSE,
    Out,
    HEADS: tl.constexpr,
    SPLITS: tl.constexpr,
    SPAD: tl.constexpr,
    BD: tl.constexpr = 128,
):
    b, h, part_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    s = tl.arange(0, SPAD)
    d = part_d * BD + tl.arange(0, BD)
    l = tl.load(LSE + (b * SPLITS + s) * HEADS + h, s < SPLITS, -float("inf"))
    m = tl.max(l, 0)
    weights = tl.exp2(l - m)
    denom = tl.sum(weights, 0)
    weights /= denom
    o = tl.load(
        Partial + ((b * SPLITS + s[:, None]) * HEADS + h) * 512 + d[None, :],
        s[:, None] < SPLITS,
        0,
    )
    result = tl.sum(o * weights[:, None], axis=0)
    tl.store(Out + (b * HEADS + h) * 512 + d, result)


class ExperimentalMLA:
    def __init__(self, q, kv, pages, lengths, mode, splits, bn, bm=32, warps=8):
        self.q, self.kv, self.pages, self.lengths = q, kv, pages, lengths
        self.mode, self.splits, self.bn = mode, splits, bn
        self.bm, self.warps = bm, warps
        self.batch, self.heads = q.shape[:2]
        self.page_size = kv.shape[1]
        self.partial = torch.empty(
            self.batch, splits, self.heads, 512, device=q.device, dtype=torch.float32
        )
        self.lse = torch.empty(
            self.batch, splits, self.heads, device=q.device, dtype=torch.float32
        )
        self.out = torch.empty(
            self.batch, self.heads, 512, device=q.device, dtype=torch.bfloat16
        )
        self.q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn) if mode == 2 else q
        self.qs = torch.empty(
            self.batch, self.heads, device=q.device, dtype=torch.float32
        )
        self.ks = None
        self.compiled_kernel = None

    def run(self, kv_scales=None, include_q_quant=True):
        if self.mode == 2 and include_q_quant:
            quantize_rows(self.q, self.q8, self.qs)
        scales = kv_scales if kv_scales is not None else self.qs
        kernel = _mla_split[
            (self.batch, self.splits, triton.cdiv(self.heads, self.bm))
        ](
            self.q8,
            self.qs,
            self.kv,
            scales,
            self.pages,
            self.lengths,
            self.partial,
            self.lse,
            self.heads,
            self.page_size,
            self.pages.stride(0),
            self.splits,
            self.mode,
            self.bn,
            self.bm,
            num_warps=self.warps,
            num_stages=1,
        )
        if kernel is not None:
            self.compiled_kernel = kernel
        _merge[(self.batch, self.heads, 4)](
            self.partial,
            self.lse,
            self.out,
            self.heads,
            self.splits,
            triton.next_power_of_2(self.splits),
            num_warps=4,
        )
        return self.out
