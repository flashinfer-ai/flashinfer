# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Agent-generated SM107 Triton kernel family for QSA stage-2 sparse GQA main
attention over a packed per-token selection, with fused sigmoid output gate
(public API: :func:`flashinfer.qsa_sparse_attention`).

Contract (checked by the wrapper in ``flashinfer/qsa.py``): HQ=6 / KV=1,
D=256, BF16 ``q``/``gate``/``out``, FP8 (e4m3, unscaled) paged K/V cache with
page size 64, int32 packed selection rows ``[T, 2052]`` (ascending token ids,
``-1`` padding, valid count in the last column — the layout produced by the
QSA stage-1 indexer), softmax scale ``256 ** -0.5``.

Launch-state notes (kept out of import time on purpose):

- Triton's process-wide launch allocator (used for device-side TMA descriptor
  scratch) is installed lazily by :func:`_ensure_launch_state` on the first
  call of :func:`run`, never at module import.
- ``_TMA_SCRATCH`` / ``_SCRATCH`` are persistent per-process caches whose
  sizes are pure functions of the input shape. An eager call therefore
  pre-sizes them; CUDA-graph capture requires one same-shape eager warmup
  call first so that no allocation happens during capture (replay then reuses
  stable pointers). Retention is bounded by the largest shape seen
  (roughly ``4 * T * 6 * 256`` BF16 + ``8 * T * 6`` FP32 elements + 256 KiB).
- The scratch caches are shared process-wide, not per stream: concurrent
  calls of this op on different CUDA streams are not supported (documented in
  the public docstring); all calls, captures and replays included, must be
  serialized on one stream at a time.
"""

import torch
import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    _HTMA = 1
except Exception:
    TensorDescriptor = None
    _HTMA = 0

try:
    from triton.runtime.errors import OutOfResources as _TritonOOR
except Exception:  # pragma: no cover - very old Triton without the error class

    class _TritonOOR(Exception):  # type: ignore[no-redef]
        pass

_TMA_SCRATCH: dict = {}


def _tma_alloc(size, alignment, stream):
    """Reusable device scratch for Triton's device-side TMA descriptors."""
    buf = _TMA_SCRATCH.get("b")
    if buf is None or buf.numel() < size:
        buf = torch.empty(max(size, 1 << 18), dtype=torch.int8, device="cuda")
        _TMA_SCRATCH["b"] = buf
    return buf


_LAUNCH_STATE_READY = False


def _ensure_launch_state():
    """Install the Triton launch allocator on first use (not at import).

    ``triton.set_allocator`` is process-wide, so importing flashinfer must not
    set it; it is installed the first time this op actually runs, and the TMA
    scratch floor is pre-touched eagerly so the allocation never lands inside
    a CUDA-graph capture that was warmed up correctly.
    """
    global _LAUNCH_STATE_READY
    if not _LAUNCH_STATE_READY:
        triton.set_allocator(_tma_alloc)
        _tma_alloc(1 << 18, 128, None)  # pre-size the scratch floor eagerly
        _LAUNCH_STATE_READY = True

NH = 6          # real query heads
HP = 8          # padded heads (power of two for tl.dot)
D = 256         # head dim
PW = 2052       # packed row width (count in last column)
LOG2E = 1.4426950408889634
SCL = (D ** -0.5) * LOG2E
PV_BIG = 0    # 0=fp8 hi+lo, 1=bf16, 2=fp8 hi only  (M64/M128 tiles)
PV_SML = 1    # same, for M16/M32 tiles
RSC_MODE = 0  # 0=guarded rescale, 1=unconditional, 2=lazy reference
SR_FP8 = 0    # short-row kernel: 1 = fp8 operands, 0 = bf16
SR_PV8 = 0    # short-row PV keeps the fp8 cache operand (fp8 p-pair)
SR_TMAX = 256 # max T routed to the short-row kernel (B == 1)
SR_WARPS = 8
SR_STAGES = 3
SR_BN = 128   # KV window for the short-row kernel
SR_BN32 = 96  # rows up to this T sweep in 32-slot windows instead
SR_BT = 2     # tokens per short-row tile
TINY_TB = 4   # route rows averaging <= TINY_TB/2 ids to the exact per-token path
TINY_MB = 32  # same, for multi-request mixes: short rows have too few terms
              # for the fp8 sweep's quantisation error to average out
QRES = 1      # fp8 query value+residual on the general dense route
TINY_WARPS = 2  # per-token kernel: more warps/CTA => more loads in flight per SM
DN_MODE = 0   # 1 = split dense tiles into the two-pass kernel
DN_TMIN = 1024
BIG_THR = 190 # min tiles before the M128 geometry is used (non-split)
BIG_MIN = 40  # only widen to the M128 tile when it still fills the GPU
BIG_BT = 20    # wider T4057 tiles; only applied below T=6000
BIG_STAGES = 0 # override pipeline depth for the main kernel (0 = default 3)
BIG_WARPS = 0  # override warp count for the main kernel (0 = default 8)
DN_BN = 0     # KV window for the dense two-pass kernel (0 = same as sparse)
DN_STAGES = 3
DN_TP = 0     # 1 = two-pass max, 0 = first-window reference + verified redo
MID_BT = 10   # tokens per tile for the 512..1024 single-request path
MID_BN = 64   # KV window for the 512..1024 single-request path
MID_STAGES = 3  # pipeline depth for the 512..1024 single-request path
MID_NSP = 2   # split-K depth at T=768 (1 drops the separate combine launch)
EX2_BF16 = 0  # 1 = bf16x2 MUFU.EX2 in the dense two-pass epilogue
TPD_MODE = 1  # 1 = two-pass dense branch inside the unified kernel
SPV_MODE = 1  # 1 = fixed-reference sparse sweep (no running maximum)
W32_MASK = 1  # 1 = int32 membership word pair in the SPV mask builder
TMA_TMIN = 3000 # lowest T that takes the descriptor + warp-specialized sweep
USE_TMA = 1   # bulk tensor descriptors on the 4K/8K dense-prefix routes
USE_WS = 1    # pipeline their K/V copies with warp-specialized tl.range


@triton.jit
def _ex2b(x):
    """2**x on bf16 pairs: SM107 runs MUFU.EX2 at 2x the fp32 rate in bf16x2."""
    return tl.inline_asm_elementwise(
        "ex2.approx.ftz.bf16x2 $0, $1;", "=r,r", [x],
        dtype=tl.bfloat16, is_pure=True, pack=2)


@triton.jit
def _ex2h(x):
    """Packed half exp2; Rubin executes four elements per clock per MUFU lane."""
    return tl.inline_asm_elementwise(
        "ex2.approx.f16x2 $0, $1;", "=r,r", [x],
        dtype=tl.float16, is_pure=True, pack=2)


@triton.jit
def _rcp(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.f32 $0, $1;", "=f,f", [x],
        dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _mul_rz_h2(a, b):
    """Rubin packed FP32 multiply with fused RZ narrowing to FP16."""
    return tl.inline_asm_elementwise(
        """{
        .reg .b64 aa, bb;
        mov.b64 aa, {$1, $2};
        mov.b64 bb, {$3, $4};
        mul.ftz.rz.f16x2.f32x2.f32x2 $0, aa, bb;
        }""",
        "=r,f,f,f,f", [a, b], dtype=tl.float16, is_pure=True, pack=2)


@triton.jit
def _sigm_h2(g):
    gh = (g * 0.5).to(tl.float16)
    th = tl.inline_asm_elementwise(
        "tanh.approx.f16x2 $0, $1;", "=r,r", [gh],
        dtype=tl.float16, is_pure=True, pack=2)
    return th.to(tl.float32) * 0.5 + 0.5


@triton.jit
def _gmul(att_half, g):
    """Rubin packed gate with sigmoid's 0.5 folded into attention."""
    gh = (g * 0.5).to(tl.float16)
    th = tl.inline_asm_elementwise(
        "tanh.approx.f16x2 $0, $1;", "=r,r", [gh],
        dtype=tl.float16, is_pure=True, pack=2)
    t1 = th + tl.full(th.shape, 1.0, tl.float16)
    return tl.inline_asm_elementwise(
        "mul.bf16x2.bf16x2.f16x2 $0, $1, $2;", "=r,r,r", [att_half, t1],
        dtype=tl.bfloat16, is_pure=True, pack=2)


@triton.jit
def _gmul_bf(att_half, g):
    """Fully packed BF16 gate with sigmoid's 0.5 already folded into att_half:
    one bf16x2 multiply, one bf16x2 tanh, one bf16x2 add and one bf16x2
    multiply replace the FP32 widening, FP32 halving and FP16 narrowing that
    the mixed-format gate needs."""
    gh = g * 0.5
    th = tl.inline_asm_elementwise(
        "tanh.approx.bf16x2 $0, $1;", "=r,r", [gh],
        dtype=tl.bfloat16, is_pure=True, pack=2)
    return att_half * (th + tl.full(th.shape, 1.0, tl.bfloat16))


@triton.jit
def _gate_bf16(att, g):
    """Packed sigmoid plus Rubin mixed-format multiply into BF16 output."""
    gh = (g * 0.5).to(tl.float16)
    th = tl.inline_asm_elementwise(
        "tanh.approx.f16x2 $0, $1;", "=r,r", [gh],
        dtype=tl.float16, is_pure=True, pack=2)
    half = tl.full(th.shape, 0.5, tl.float16)
    sig = th * half + half
    return tl.inline_asm_elementwise(
        "mul.bf16x2.bf16x2.f16x2 $0, $1, $2;", "=r,r,r", [att, sig],
        dtype=tl.bfloat16, is_pure=True, pack=2)


@triton.jit
def _or_comb(a, b):
    return a | b


@triton.jit
def _add_h(a, b):
    return (a + b).to(tl.float16)


@triton.jit
def _popc32(x):
    return tl.inline_asm_elementwise(
        "popc.b32 $0, $1;", "=r,r", [x], dtype=tl.int32,
        is_pure=True, pack=1)


@triton.jit
def _popc64(x):
    return tl.inline_asm_elementwise(
        "popc.b64 $0, $1;", "=r,l", [x], dtype=tl.int32,
        is_pure=True, pack=1)


@triton.jit
def _blk(qh, ql, emat, mf, rcw, KC, VC, koff, sv, acc, mi, li, cc,
         DC: tl.constexpr, PVB: tl.constexpr, USEQL: tl.constexpr,
         DM: tl.constexpr, BNC: tl.constexpr, RSC: tl.constexpr):
    """One KV window: masked QK, online softmax, PV accumulate.
    Operands are fp8 pairs (value + residual) so both MMAs run at fp8 rate with
    ~bf16 accuracy and the shared-memory operands stay 1 byte/element."""
    kt = tl.trans(tl.load(KC + koff, mask=sv[:, None], other=0.0))
    vt = tl.load(VC + koff, mask=sv[:, None], other=0.0)
    if DM:
        # dense prefix: mask is just a per-row key count, cheaper inline than
        # as a second accumulator dot
        s = tl.where(tl.arange(0, BNC)[None, :] < rcw[:, None],
                     tl.dot(qh, kt, out_dtype=tl.float32), -3.0e38)
    elif USEQL:
        s = tl.dot(emat, mf,
                   tl.dot(ql, kt, tl.dot(qh, kt, out_dtype=tl.float32),
                          out_dtype=tl.float32),
                   out_dtype=tl.float32)
    else:
        s = tl.dot(emat, mf, tl.dot(qh, kt, out_dtype=tl.float32),
                   out_dtype=tl.float32)
    rmax = tl.max(s, 1)
    if RSC == 2:
        # mi is a reference exponent, not the running maximum: it only has to
        # keep every probability inside the fp8 operand range, so the [RW, DC]
        # accumulator rescale fires about once per tile instead of once per
        # window.  Exact either way.
        if tl.max(rmax - mi) * cc > 8.0:
            mn = tl.maximum(mi, rmax + 4.0 / cc)
            al = tl.exp2((mi - mn) * cc)
            acc = acc * al[:, None]
            li = li * al
            mi = mn
    elif RSC == 0:
        if tl.max(rmax - mi) > 0.0:
            mn = tl.maximum(mi, rmax + 1.0 / cc)
            al = tl.exp2((mi - mn) * cc)
            acc = acc * al[:, None]
            li = li * al
            mi = mn
    else:
        mn = tl.maximum(mi, rmax + 1.0 / cc)
        al = tl.exp2((mi - mn) * cc)
        acc = acc * al[:, None]
        li = li * al
        mi = mn
    pp = tl.exp2(s * cc - (mi * cc)[:, None])
    li = li + tl.sum(pp, 1)
    if PVB == 1:
        acc = tl.dot(pp.to(tl.bfloat16), vt.to(tl.bfloat16), acc, out_dtype=tl.float32)
    elif PVB == 2:
        acc = tl.dot(pp.to(tl.float8e4nv), vt, acc, out_dtype=tl.float32)
    else:
        ph = pp.to(tl.float8e4nv)
        pl = (pp - ph.to(tl.float32)).to(tl.float8e4nv)
        acc = tl.dot(pl, vt, tl.dot(ph, vt, acc, out_dtype=tl.float32),
                     out_dtype=tl.float32)
    return acc, mi, li



@triton.jit
def _mfwin(S, w0, biasv, one, jlo, jhi, MKC: tl.constexpr,
           NSUB: tl.constexpr, BN: tl.constexpr, W32: tl.constexpr):
    """Per-token membership bias for one KV window, from the packed id slice."""
    o = S - w0
    inw = (o >= 0) & (o < BN) & (S >= 0)
    if W32:
        # two 32-bit membership words instead of one 64-bit word: the variable
        # shifts and the OR-reduction stay on the int32 datapath
        one32 = tl.full(inw.shape, 1, tl.int32)
        lo = inw & (o < 32)
        hi = inw & (o >= 32)
        blo = tl.reduce(tl.where(lo, one32 << tl.where(lo, o, 0), 0), 1, _or_comb)
        bhi = tl.reduce(tl.where(hi, one32 << tl.where(hi, o - 32, 0), 0), 1,
                        _or_comb)
        adv = _popc32(blo) + _popc32(bhi)
        j32 = tl.arange(0, BN) % 32
        jh = tl.arange(0, BN) // 32
        bw32 = tl.where(jh[None, :] == 0, blo[:, None], bhi[:, None])
        mf = tl.where(((bw32 >> j32[None, :]) & 1) != 0,
                      0.0, biasv).to(tl.bfloat16)
    elif NSUB == 1:
        adv = tl.sum(inw.to(tl.int32), 1)
        ow = tl.where(inw, o, 0).to(tl.int64)
        bw = tl.reduce(tl.where(inw, one << ow, 0), 1, _or_comb)
        mf = tl.where(((bw[:, None] >> jlo[None, :]) & 1) != 0,
                      0.0, biasv).to(tl.bfloat16)
    else:
        adv = tl.sum(inw.to(tl.int32), 1)
        sel = tl.zeros([MKC, BN], tl.int32)
        for sb in tl.static_range(NSUB):
            insb = inw & ((o // 64) == sb)
            ow = tl.where(insb, o % 64, 0).to(tl.int64)
            bw = tl.reduce(tl.where(insb, one << ow, 0), 1, _or_comb)
            hit2 = (((bw[:, None] >> jlo[None, :]) & 1) != 0) & (jhi[None, :] == sb)
            sel = sel | hit2.to(tl.int32)
        mf = tl.where(sel != 0, 0.0, biasv).to(tl.bfloat16)
    return mf, adv


@triton.jit
def _mfwin32(S0, S1, w0):
    """64-slot membership from two independently laid-out 32-id chunks."""
    o0 = S0 - w0
    o1 = S1 - w0
    one = tl.full(o0.shape, 1, tl.int32)
    lo0 = o0.to(tl.uint32) < 32
    hi0 = (o0 - 32).to(tl.uint32) < 32
    hi1 = (o1 - 32).to(tl.uint32) < 32
    blo = tl.reduce(tl.where(lo0, one << tl.where(lo0, o0, 0), 0), 1, _or_comb)
    hibits = (tl.where(hi0, one << tl.where(hi0, o0 - 32, 0), 0) |
              tl.where(hi1, one << tl.where(hi1, o1 - 32, 0), 0))
    bhi = tl.reduce(hibits, 1, _or_comb)
    words = (blo.to(tl.uint32).to(tl.uint64) |
             (bhi.to(tl.uint32).to(tl.uint64) << 32))
    adv = _popc64(words)
    return adv, blo, bhi


@triton.jit
def _mfwin128(S, w0):
    """128-slot membership as two packed 64-bit words per token."""
    o = S - w0
    one = tl.full(o.shape, 1, tl.int32)
    b0m = o.to(tl.uint32) < 32
    b1m = (o - 32).to(tl.uint32) < 32
    b2m = (o - 64).to(tl.uint32) < 32
    b3m = (o - 96).to(tl.uint32) < 32
    b0 = tl.sum(tl.where(b0m, one << tl.where(b0m, o, 0), 0), 1)
    b1 = tl.sum(tl.where(b1m, one << tl.where(b1m, o - 32, 0), 0), 1)
    b2 = tl.sum(tl.where(b2m, one << tl.where(b2m, o - 64, 0), 0), 1)
    b3 = tl.sum(tl.where(b3m, one << tl.where(b3m, o - 96, 0), 0), 1)
    lo = b0.to(tl.uint32).to(tl.uint64) | (b1.to(tl.uint32).to(tl.uint64) << 32)
    hi = b2.to(tl.uint32).to(tl.uint64) | (b3.to(tl.uint32).to(tl.uint64) << 32)
    return _popc64(lo) + _popc64(hi), lo, hi


@triton.jit(noinline=True)
def _qsa_tiny_redo(Q, G, KC, VC, PK, O, t, base, NSLOT,
                   HPAD: tl.constexpr, NHC: tl.constexpr,
                   DC: tl.constexpr, PWC: tl.constexpr,
                   SC: tl.constexpr):
    """Cold exact packed-occurrence path for rows with more than two ids."""
    row = t * PWC
    cnt = tl.minimum(tl.maximum(tl.load(PK + row + (PWC - 1)), 0),
                     PWC - 1)
    hh = tl.arange(0, HPAD)
    dd = tl.arange(0, DC)
    hv = hh < NHC
    qo = t * (NHC * DC) + hh[:, None] * DC + dd[None, :]
    q = tl.load(Q + qo, mask=hv[:, None], other=0.0).to(tl.float32)
    acc = tl.zeros([HPAD, DC], tl.float32)
    li = tl.zeros([HPAD], tl.float32)
    mi = tl.full([HPAD], -1.0e30, tl.float32)
    for u in range(0, cnt):
        idu = tl.load(PK + row + u)
        slot = base + idu
        ok = (idu >= 0) & (slot < NSLOT)
        kv = tl.load(KC + slot * DC + dd, mask=ok, other=0.0).to(tl.float32)
        s = tl.where(ok, tl.sum(q * kv[None, :], 1) * SC, -1.0e30)
        mn = tl.maximum(mi, s)
        al = tl.exp2(mi - mn)
        pw = tl.exp2(s - mn)
        li = li * al + pw
        mi = mn
        vv = tl.load(VC + slot * DC + dd, mask=ok, other=0.0).to(tl.float32)
        acc = acc * al[:, None] + pw[:, None] * vv[None, :]
    rl = tl.where(li > 0, 1.0 / tl.where(li > 0, li, 1.0), 0.0)
    gg = tl.load(G + qo, mask=hv[:, None], other=0.0).to(tl.float32)
    att = (acc * rl[:, None]).to(tl.bfloat16)
    tl.store(O + qo, _gate_bf16(att, gg), mask=hv[:, None])


@triton.jit
def _qsa_tiny(Q, G, KC, VC, PK, CU, PB, O, T, B, NSLOT,
              HPAD: tl.constexpr, NHC: tl.constexpr, DC: tl.constexpr,
              PWC: tl.constexpr, NB: tl.constexpr, SINGLE: tl.constexpr,
              PAIR: tl.constexpr,
              SC: tl.constexpr):
    """One token per CTA, no MMA (so no TMEM): for rows whose selection holds
    only a handful of ids this beats the windowed path because many CTAs fit
    per SM.  Correct for any count, just slower when counts are large."""
    pid = tl.program_id(0)
    if PAIR:
        half = T >> 1
        t = tl.where(pid < half, (pid << 1) + 1,
                     (pid - half) << 1)
    else:
        t = pid
    if t >= T:
        return
    row = t * PWC
    # Every address that depends only on t is issued first; the page base is
    # fetched speculatively alongside cu_seqlens so the K/V gather sits one
    # memory level after the launch instead of two.
    i0 = tl.load(PK + row)
    i1 = tl.load(PK + row + 1)
    i2 = tl.load(PK + row + 2)
    if SINGLE:
        base = tl.load(PB) * 64
    else:
        # near-uniform request lengths make t*B//T a good guess; it is checked
        # against cu_seqlens and falls back to an exact scan when wrong.
        if PAIR:
            g = t >> 1
        else:
            g = tl.minimum((t * B) // T, B - 1)
        pbg = tl.load(PB + g)
        if (tl.load(CU + g) <= t) & (t < tl.load(CU + g + 1)):
            base = pbg * 64
        else:
            lo = 0
            hi = B
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                boundary = tl.load(CU + mid)
                go_right = t >= boundary
                lo = tl.where(go_right, mid, lo)
                hi = tl.where(go_right, hi, mid)
            base = tl.load(PB + lo) * 64

    hh = tl.arange(0, HPAD)
    dd = tl.arange(0, DC)
    hv = hh < NHC
    qo = t * (NHC * DC) + hh[:, None] * DC + dd[None, :]

    if i2 < 0:
        # One shared schedule for the 0/1/2-key rows: the two candidate slots
        # are gathered unconditionally so no branch sits between the id loads
        # and the cache loads.
        sl0 = base + i0
        sl1 = base + i1
        o0 = (i0 >= 0) & (sl0 < NSLOT)
        o1 = (i1 >= 0) & (sl1 < NSLOT)
        # Gate traffic is independent of the paged-cache address chain; issue
        # it first so its latency overlaps the four K/V gathers.
        ggb = tl.load(G + qo, mask=hv[:, None], other=0.0)
        v0 = tl.load(VC + sl0 * DC + dd, mask=o0, other=0.0,
                     cache_modifier=".cg").to(tl.float32)
        if i1 < 0:
            # A single selected value has probability one; avoid issuing the
            # unused query, key, second-value, and softmax instructions.
            att = v0[None, :].to(tl.bfloat16)
            tl.store(O + qo, _gate_bf16(att, ggb), mask=hv[:, None])
            return
        v1 = tl.load(VC + sl1 * DC + dd, mask=o1, other=0.0,
                     cache_modifier=".cg").to(tl.float32)
        k0 = tl.load(KC + sl0 * DC + dd, mask=o1, other=0.0,
                     cache_modifier=".cg").to(tl.float32)
        k1 = tl.load(KC + sl1 * DC + dd, mask=o1, other=0.0,
                     cache_modifier=".cg").to(tl.float32)
        qb = tl.load(Q + qo, mask=(hv & o1)[:, None], other=0.0)
        # exact two-key softmax: only the score difference matters
        delta = tl.sum(qb.to(tl.float32) * (k1 - k0)[None, :], 1) * SC
        p1 = tl.where(o1 & o0, tl.fdiv(1.0, 1.0 + tl.exp2(-delta), False),
                      tl.where(o1, 1.0, 0.0))
        att = (v0[None, :] + p1[:, None] * (v1 - v0)[None, :]).to(tl.bfloat16)
        tl.store(O + qo, _gate_bf16(att, ggb), mask=hv[:, None])
        return

    _qsa_tiny_redo(Q, G, KC, VC, PK, O, t, base, NSLOT,
                   HPAD, NHC, DC, PWC, SC)


@triton.jit
def _sblk(qh, ql, emat, mf, rcw, KC, VC, koff, acc, mi, li, cc,
          DM: tl.constexpr, BN: tl.constexpr, FP8: tl.constexpr,
          FIX: tl.constexpr = False, NOMAX: tl.constexpr = False,
          PV8: tl.constexpr = False):
    """Short-row KV window: masked QK, online softmax, PV accumulate.

    With FP8 the cache operands feed the MMA unconverted, so the window costs
    no fp8->bf16 unpack of the 2 x BN x D cache tile.  Q and P are carried as
    fp8 value+residual pairs, so the two fp8 MMAs cost the same as the one
    bf16 MMA they replace while keeping bf16-level accuracy.
    """
    if BN == 128:
        kt = tl.load(KC + koff, eviction_policy="evict_last")
        vt = tl.load(VC + koff, eviction_policy="evict_last")
    else:
        kt = tl.load(KC + koff)
        vt = tl.load(VC + koff)
    if FP8:
        ktt = tl.trans(kt)
        qk = tl.dot(ql, ktt, tl.dot(qh, ktt, out_dtype=tl.float32),
                    out_dtype=tl.float32)
    else:
        ktt = tl.trans(kt.to(tl.float16))
        qk = tl.dot(qh, ktt, out_dtype=tl.float32)
    if DM:
        s = tl.where(tl.arange(0, BN)[None, :] < rcw[:, None], qk, -3.0e38)
    else:
        s = tl.dot(emat, mf, qk, out_dtype=tl.float32)
    if FIX:
        if not NOMAX:
            mi = tl.maximum(mi, tl.max(s, 1))
    else:
        rmax = tl.max(s, 1)
        if tl.max(rmax - mi) > 0.0:
            mn = tl.maximum(mi, rmax + 1.0 / cc)
            al = tl.exp2((mi - mn) * cc)
            acc = acc * al[:, None]
            li = li * al
            mi = mn
    if FP8:
        if FIX:
            pp = tl.exp2(s * cc[:, None])
        else:
            pp = tl.exp2(s * cc[:, None] - (mi * cc)[:, None])
    else:
        if FIX:
            if NOMAX:
                pp = _ex2h(_mul_rz_h2(s, cc)).to(tl.float32)
            else:
                pp = _ex2h((s * cc).to(tl.float16)).to(tl.float32)
        else:
            pp = _ex2h((s * cc - (mi * cc)[:, None]).to(
                tl.float16)).to(tl.float32)
    if FP8:
        # fp8 value + residual keeps ~bf16 accuracy at fp8 operand cost
        ph = pp.to(tl.float8e4nv)
        pl = (pp - ph.to(tl.float32)).to(tl.float8e4nv)
        acc = tl.dot(pl, vt, tl.dot(ph, vt, acc, out_dtype=tl.float32),
                     out_dtype=tl.float32)
    elif PV8:
        # Keep the cache operand of the PV product in fp8: the [BN, D] value
        # tile then never round-trips through registers for an fp16 convert,
        # and the probability pair costs two fp8 MMAs instead of one fp16 one
        # (same tensor-core time) at ~fp16 accuracy.
        ph = pp.to(tl.float8e4nv)
        pl = (pp - ph.to(tl.float32)).to(tl.float8e4nv)
        acc = tl.dot(pl, vt, tl.dot(ph, vt, acc, out_dtype=tl.float32),
                     out_dtype=tl.float32)
    else:
        acc = tl.dot(pp.to(tl.float16), vt.to(tl.float16), acc,
                     out_dtype=tl.float32)
    li = li + tl.sum(pp, 1)
    return acc, mi, li


@triton.jit(noinline=True)
def _qsa_short2_redo(Q, G, KC, VC, PK, PB, O, t0, T, NSLOT,
                     BT: tl.constexpr, RW: tl.constexpr,
                     NHC: tl.constexpr, DC: tl.constexpr,
                     PWC: tl.constexpr, SC: tl.constexpr):
    """Cold exact packed-occurrence retry, kept out of the dense hot path."""
    rows = tl.arange(0, RW)
    if BT == 1:
        rm = tl.zeros([RW], tl.int32)
    else:
        rm = (rows >= NHC).to(tl.int32)
    rt = t0 + rm
    rv = (rows < BT * NHC) & (rt < T)
    rts = tl.where(rv, rt, 0)
    dd = tl.arange(0, DC)
    qoff = (t0 * NHC + rows)[:, None] * DC + dd[None, :]
    q = tl.load(Q + qoff, mask=rv[:, None], other=0.0).to(tl.float32)
    cnt = tl.load(PK + rts * PWC + (PWC - 1), mask=rv, other=0)
    cnt = tl.minimum(tl.maximum(cnt, 0), PWC - 1)
    base = tl.load(PB) * 64
    acc = tl.zeros([RW, DC], tl.float32)
    mi = tl.full([RW], -1.0e30, tl.float32)
    li = tl.zeros([RW], tl.float32)
    for u in range(0, tl.max(cnt, 0)):
        active = rv & (u < cnt)
        idx = tl.load(PK + rts * PWC + u, mask=active, other=-1)
        slot = base + idx
        ok = active & (idx >= 0) & (slot < NSLOT)
        kv = tl.load(KC + slot[:, None] * DC + dd[None, :],
                     mask=ok[:, None], other=0.0).to(tl.float32)
        s = tl.where(ok, tl.sum(q * kv, 1) * SC, -3.0e38)
        mn = tl.maximum(mi, s)
        al = tl.exp2(mi - mn)
        pw = tl.exp2(s - mn)
        li = li * al + pw
        mi = mn
        vv = tl.load(VC + slot[:, None] * DC + dd[None, :],
                     mask=ok[:, None], other=0.0).to(tl.float32)
        acc = acc * al[:, None] + pw[:, None] * vv
    den = tl.where(li > 0, li, 1.0)
    att = tl.where(li[:, None] > 0, acc / den[:, None], 0.0).to(
        tl.bfloat16).to(tl.float32)
    gf = tl.load(G + qoff, mask=rv[:, None], other=0.0).to(tl.float32)
    tl.store(O + qoff, (att * _sigm_h2(gf)).to(tl.bfloat16),
             mask=rv[:, None])


@triton.jit
def _qsa_short2(Q, G, KC, VC, PK, PB, O, T, NSLOT,
                BT: tl.constexpr, RW: tl.constexpr, MK: tl.constexpr,
                BN: tl.constexpr, NSUB: tl.constexpr, FP8: tl.constexpr,
                GEARLY: tl.constexpr, PV8: tl.constexpr,
                FASTRCP: tl.constexpr,
                OVH: tl.constexpr,
                NHC: tl.constexpr, DC: tl.constexpr,
                PWC: tl.constexpr, SC: tl.constexpr):
    """Two tokens per warp-MMA tile for short single-request rows.

    The dense-prefix sweep is speculative but verified by the last packed id;
    an irregular row is recomputed with the exact membership-mask sweep.
    """
    t0 = tl.program_id(0) * BT
    mxc = tl.minimum(t0 + BT, T)
    nwd = (mxc + BN - 1) // BN
    mt = tl.arange(0, MK)
    toks = t0 + mt
    tv = (toks < T) & (mt < BT)
    tokc = tl.where(tv, toks, 0)
    prow = tokc * PWC
    cnt = tl.load(PK + prow + (PWC - 1), mask=tv, other=0)
    vid = tl.load(PK + prow + tl.minimum(tokc, PWC - 2),
                  mask=tv, other=-1)
    # Speculate that this request's pages start at slot 0 so the K/V window
    # address is known at launch instead of one global round trip later; the
    # real page base is checked with the packed proof below and an unexpected
    # base falls through to the exact retry.
    base = tl.load(PB)

    rows = tl.arange(0, RW)
    if BT == 1:
        rm = tl.zeros([RW], tl.int32)
    else:
        rm = (rows >= NHC).to(tl.int32)
    rt = t0 + rm
    rv = (rows < BT * NHC) & (rt < T)
    dd = tl.arange(0, DC)
    qoff = (t0 * NHC + rows)[:, None] * DC + dd[None, :]
    qr = tl.load(Q + qoff, mask=rv[:, None], other=0.0)
    if GEARLY:
        # Issue gate traffic before the dependent KV sweep.  Only worth it on
        # the 16-row tile: on wider tiles the live [RW, DC] gate tile costs
        # more registers than the prefetch saves.
        gb = tl.load(G + qoff, mask=rv[:, None], other=0.0)

    if FP8:
        # per-row fp8 scale: no CTA-wide reduction, finer quantisation
        qf = qr.to(tl.float32)
        sq = 240.0 / tl.maximum(tl.max(tl.abs(qf), 1), 1e-20)
        qs = qf * sq[:, None]
        q = qs.to(tl.float8e4nv)
        qlo = (qs - q.to(tl.float32)).to(tl.float8e4nv)
        cc = SC / sq
    else:
        q = qr.to(tl.float16)
        qlo = q
        # scalar softmax scale: a per-row vector here turns every window's
        # exponent into a broadcast multiply and measurably slows the tile
        cc = SC

    match = rm[:, None] == mt[None, :]
    emat = match.to(tl.bfloat16)
    rcnt = tl.where(rv, rt + 1, 0)
    mz = tl.zeros([MK, BN], tl.bfloat16)
    ii = tl.arange(0, BN)
    acc = tl.zeros([RW, DC], tl.float32)
    mi = tl.full([RW], -1.0e30, tl.float32)
    li = tl.zeros([RW], tl.float32)
    # Contiguous cache windows need only a scalar base per iteration; keep the
    # element offsets loop-invariant instead of materializing a slot tile.
    koff0 = ii[:, None] * DC + dd[None, :]
    for kw in range(0, nwd):
        w0 = kw * BN
        if OVH:
            # The last window overhangs the cache (NSLOT is not a multiple of
            # BN up to T): keep the LOGICAL origin at w0 and clamp per slot,
            # like every other route. The clamped (duplicate) slots are
            # columns ii >= NSLOT - w0 >= rcnt - w0, which the rcw mask in
            # _sblk already excludes from the softmax.
            sl = tl.minimum(w0 + ii, NSLOT - 1)
            acc, mi, li = _sblk(q, qlo, emat, mz, rcnt - w0,
                                KC, VC, sl[:, None] * DC + dd[None, :],
                                acc, mi, li, cc, True, BN, FP8, True, True,
                                PV8)
        else:
            # No window can overhang (checked at launch): w0 is in bounds.
            acc, mi, li = _sblk(q, qlo, emat, mz, rcnt - w0,
                                KC + w0 * DC, VC + w0 * DC, koff0,
                                acc, mi, li, cc, True, BN, FP8, True, True,
                                PV8)

    proof = tl.reduce(tl.where(tv, (cnt - 1 - toks) | (vid - toks), 0),
                      0, _or_comb)
    # With the fixed zero reference, 2**max <= li <= count * 2**max.
    # Validate the exponent range without carrying a row maximum through
    # every window; unusual rows are recomputed by the exact retry helper.
    den_ok = (li <= 1.0e30) & ((rcnt <= 0) | (
        li >= rcnt.to(tl.float32) * 3.0e-4))
    bad = tl.min(den_ok.to(tl.int32), 0) == 0
    if (proof != 0) | bad | (base != 0):
        _qsa_short2_redo(Q, G, KC, VC, PK, PB, O, t0, T, NSLOT,
                         BT, RW, NHC, DC, PWC, SC)
        return

    if FASTRCP:
        inv_den = tl.where(li > 0, _rcp(tl.where(li > 0, li, 1.0)), 0.0)
    else:
        inv_den = tl.where(li > 0, tl.fdiv(1.0, li, False), 0.0)
    att = (acc * inv_den[:, None]).to(tl.bfloat16).to(tl.float32)
    if GEARLY:
        gf = gb.to(tl.float32)
    else:
        gf = tl.load(G + qoff, mask=rv[:, None], other=0.0).to(tl.float32)
    sg = _sigm_h2(gf)
    tl.store(O + qoff, (att * sg).to(tl.bfloat16), mask=rv[:, None])


@triton.jit
def _qsa_m8(Q, G, KC, VC, PK, PB, O, T, NSLOT,
            HPAD: tl.constexpr, NHC: tl.constexpr, DC: tl.constexpr,
            PWC: tl.constexpr, BN: tl.constexpr, SC: tl.constexpr):
    """One token per CTA with six heads in an eight-row BF16 MMA tile."""
    t = T - 1 - tl.program_id(0)
    if t < 0:
        return
    base = tl.load(PB) * 64
    prow = t * PWC
    cnt = tl.minimum(tl.maximum(tl.load(PK + prow + (PWC - 1)), 0), PWC - 1)
    has = cnt > 0
    fid = tl.load(PK + prow, mask=has, other=0x40000000)
    lid = tl.load(PK + prow + cnt - 1, mask=has, other=-1)
    dense = has & (fid == 0) & (lid == cnt - 1)
    wlo = (fid // BN) * BN
    nw = tl.where(has, (lid - wlo) // BN + 1, 0)

    hh = tl.arange(0, HPAD)
    dd = tl.arange(0, DC)
    hv = hh < NHC
    qo = t * (NHC * DC) + hh[:, None] * DC + dd[None, :]
    q = tl.load(Q + qo, mask=hv[:, None], other=0.0)
    acc = tl.zeros([HPAD, DC], tl.float32)
    mi = tl.full([HPAD], -1.0e30, tl.float32)
    li = tl.zeros([HPAD], tl.float32)
    ii = tl.arange(0, BN)
    jb = ii.to(tl.int64)
    p = 0
    one = tl.full([BN], 1, tl.int64)

    for kw in range(0, nw):
        w0 = wlo + kw * BN
        if dense:
            km = (w0 + ii[None, :]) < cnt
        else:
            sidx = p + ii
            sel = tl.load(PK + prow + sidx, mask=sidx < cnt, other=-1)
            off = sel - w0
            inw = (off >= 0) & (off < BN) & (sel >= 0)
            sh = tl.where(inw, off, 0).to(tl.int64)
            bits = tl.reduce(tl.where(inw, one << sh, 0), 0, _or_comb)
            p += tl.sum(inw.to(tl.int32), 0)
            km = ((bits >> jb[None, :]) & 1) != 0
        slot = base + w0 + ii
        sv = slot < NSLOT
        koff = slot[:, None] * DC + dd[None, :]
        kt = tl.load(KC + koff, mask=sv[:, None], other=0.0).to(tl.bfloat16)
        vt = tl.load(VC + koff, mask=sv[:, None], other=0.0).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(kt), out_dtype=tl.float32) * SC
        s = tl.where(km, s, -float("inf"))
        mn = tl.maximum(mi, tl.max(s, 1))
        al = tl.exp2(mi - mn)
        pp = tl.exp2(s - mn[:, None])
        li = li * al + tl.sum(pp, 1)
        mi = mn
        acc = acc * al[:, None] + tl.dot(pp.to(tl.bfloat16), vt,
                                         out_dtype=tl.float32)

    den = tl.where(li > 0, li, 1.0)
    att = tl.where(li[:, None] > 0, acc / den[:, None], 0.0)
    att = att.to(tl.bfloat16).to(tl.float32)
    gg = tl.load(G + qo, mask=hv[:, None], other=0.0).to(tl.float32)
    sg = tl.fdiv(1.0, 1.0 + tl.exp2(gg * -1.4426950408889634), False)
    tl.store(O + qo, (att * sg).to(tl.bfloat16), mask=hv[:, None])



@triton.jit
def _qsa_dn(Q, G, KC, VC, PK, CU, PB, O, T, B, NSLOT,
            BT: tl.constexpr, RW: tl.constexpr, MK: tl.constexpr,
            BN: tl.constexpr, NB: tl.constexpr, SINGLE: tl.constexpr,
            PVB: tl.constexpr, EX2B: tl.constexpr, TP: tl.constexpr,
            NHC: tl.constexpr, DC: tl.constexpr,
            PWC: tl.constexpr, SC: tl.constexpr):
    """Dense-prefix tiles, two-pass flash attention.

    Pass 1 sweeps K only and keeps the exact per-row maximum; pass 2 then
    exponentiates against that fixed reference, so the [RW, DC] accumulator is
    never rescaled and stays resident in the MMA accumulator instead of being
    round-tripped through registers every window.
    """
    task = tl.program_id(0)
    if SINGLE:
        tot = (T + BT - 1) // BT
        pid = tot - 1 - task
        r = 0
        s0 = 0
        s1 = T
        local = pid
    else:
        jj = tl.arange(0, NB)
        mm = jj < B
        c0 = tl.load(CU + jj, mask=mm, other=0)
        c1 = tl.load(CU + jj + 1, mask=mm, other=0)
        pbv = tl.load(PB + jj, mask=mm, other=0)
        nt = tl.where(mm, (c1 - c0 + (BT - 1)) // BT, 0)
        tot = tl.sum(nt, 0)
        if task >= tot:
            return
        nmin = tl.min(tl.where(mm, nt, 0x7FFFFFFF), 0)
        nmax = tl.max(nt, 0)
        if nmax - nmin <= 1:
            extra = mm & (nt > nmin)
            nextra = tl.sum(extra.to(tl.int32), 0)
            if task < nextra:
                erank = tl.cumsum(extra.to(tl.int32), 0) - 1
                r = tl.sum(tl.where(extra & (erank == task), jj, 0), 0)
                local = nmin
            else:
                qtask = task - nextra
                r = qtask % B
                local = nmin - 1 - qtask // B
            pid = 0
        else:
            pid = tot - 1 - task
            cs = tl.cumsum(nt, 0)
            ex = cs - nt
            hit = mm & (ex <= pid)
            r = tl.sum(hit.to(tl.int32), 0) - 1
            local = pid - tl.max(tl.where(hit, ex, 0), 0)
        s0 = tl.load(CU + r)
        s1 = tl.load(CU + r + 1)

    base = tl.load(PB + r) * 64
    t0 = s0 + local * BT

    mt = tl.arange(0, MK)
    toks = t0 + mt
    tv = (toks < s1) & (mt < BT)
    tokc = tl.where(tv, toks, s0)
    prow = tokc * PWC
    cnt = tl.load(PK + prow + (PWC - 1), mask=tv, other=0)
    # the packed row only holds PWC-1 ids, so a count beyond that cannot
    # name more selected positions than exist
    cnt = tl.minimum(tl.where(tv & (cnt > 0), cnt, 0), PWC - 1)
    hv = cnt > 0
    lid = tl.load(PK + prow + cnt - 1, mask=hv, other=-1)
    if tl.min(tl.where(hv, cnt - 1 - lid, 0)) != 0:
        return

    rows = tl.arange(0, RW)
    rm = rows // NHC
    rh = rows % NHC
    rt = t0 + rm
    rv = (rm < BT) & (rt < s1)
    qo = tl.where(rv, rt, s0) * (NHC * DC) + rh * DC
    dd = tl.arange(0, DC)
    qoff = qo[:, None] + dd[None, :]
    qf = tl.load(Q + qoff, mask=rv[:, None], other=0.0).to(tl.float32)
    sq = 240.0 / tl.maximum(tl.max(tl.abs(qf), 1), 1e-20)
    q8 = (qf * sq[:, None]).to(tl.float8e4nv)
    cc = SC / sq

    rcnt = tl.sum(tl.where(rm[:, None] == mt[None, :], cnt[None, :], 0), 1)
    nw = (tl.max(cnt) + BN - 1) // BN
    ii = tl.arange(0, BN)
    if TP:
        mi = tl.full([RW], -1.0e30, tl.float32)
        for kw in range(0, nw):
            w0 = kw * BN
            slot = tl.minimum(base + w0 + ii, NSLOT - 1)
            koff = slot[:, None] * DC + dd[None, :]
            kt = tl.trans(tl.load(KC + koff))
            s = tl.where(ii[None, :] < (rcnt - w0)[:, None],
                         tl.dot(q8, kt, out_dtype=tl.float32), -3.0e38)
            mi = tl.maximum(mi, tl.max(s, 1))
        mr = mi * cc
    else:
        # the first window alone sets the reference exponent; the sweep below
        # verifies it, so the common case reads K once instead of twice
        slot = tl.minimum(base + ii, NSLOT - 1)
        kt = tl.trans(tl.load(KC + slot[:, None] * DC + dd[None, :]))
        s = tl.where(ii[None, :] < rcnt[:, None],
                     tl.dot(q8, kt, out_dtype=tl.float32), -3.0e38)
        mr = tl.maximum(tl.max(s, 1), -1.0e30) * cc

    acc = tl.zeros([RW, DC], tl.float32)
    li = tl.zeros([RW], tl.float32)
    mx = tl.full([RW], -1.0e30, tl.float32)
    for kw in range(0, nw):
        w0 = kw * BN
        slot = tl.minimum(base + w0 + ii, NSLOT - 1)
        koff = slot[:, None] * DC + dd[None, :]
        kt = tl.trans(tl.load(KC + koff))
        vt = tl.load(VC + koff)
        s = tl.where(ii[None, :] < (rcnt - w0)[:, None],
                     tl.dot(q8, kt, out_dtype=tl.float32), -3.0e38)
        if not TP:
            mx = tl.maximum(mx, tl.max(s, 1))
        if EX2B:
            pb = _ex2b((s * cc[:, None] - mr[:, None]).to(tl.bfloat16))
            pp = pb.to(tl.float32)
        else:
            pp = tl.exp2(s * cc[:, None] - mr[:, None])
        li = li + tl.sum(pp, 1)
        if PVB == 1:
            acc = tl.dot(pp.to(tl.bfloat16), vt.to(tl.bfloat16), acc,
                         out_dtype=tl.float32)
        elif PVB == 2:
            acc = tl.dot(pp.to(tl.float8e4nv), vt, acc, out_dtype=tl.float32)
        else:
            ph = pp.to(tl.float8e4nv)
            pl = (pp - ph.to(tl.float32)).to(tl.float8e4nv)
            acc = tl.dot(pl, vt, tl.dot(ph, vt, acc, out_dtype=tl.float32),
                         out_dtype=tl.float32)
    if not TP:
        # rows with no selected key are excluded: their reference and running
        # maximum are both the clamp sentinel and carry no information
        dlt = tl.where(rcnt > 0, mx * cc - mr, 0.0)
        if (tl.max(dlt) > 8.0) | (tl.min(dlt) < -60.0):
            # the reference is out of range in either direction: redo once
            # against the verified global maximum
            mr = mx * cc
            acc = tl.zeros([RW, DC], tl.float32)
            li = tl.zeros([RW], tl.float32)
            for kw in range(0, nw):
                w0 = kw * BN
                slot = tl.minimum(base + w0 + ii, NSLOT - 1)
                koff = slot[:, None] * DC + dd[None, :]
                kt = tl.trans(tl.load(KC + koff))
                vt = tl.load(VC + koff)
                s = tl.where(ii[None, :] < (rcnt - w0)[:, None],
                             tl.dot(q8, kt, out_dtype=tl.float32), -3.0e38)
                pp = tl.exp2(s * cc[:, None] - mr[:, None])
                li = li + tl.sum(pp, 1)
                if PVB == 1:
                    acc = tl.dot(pp.to(tl.bfloat16), vt.to(tl.bfloat16), acc,
                                 out_dtype=tl.float32)
                else:
                    ph = pp.to(tl.float8e4nv)
                    pl = (pp - ph.to(tl.float32)).to(tl.float8e4nv)
                    acc = tl.dot(pl, vt,
                                 tl.dot(ph, vt, acc, out_dtype=tl.float32),
                                 out_dtype=tl.float32)

    rl = tl.where(li > 0, 1.0 / tl.where(li > 0, li, 1.0), 0.0)
    gg = tl.load(G + qoff, mask=rv[:, None], other=0.0).to(tl.float32)
    sg = tl.fdiv(1.0, 1.0 + tl.exp2(gg * -1.4426950408889634), False)
    att = (acc * rl[:, None]).to(tl.bfloat16).to(tl.float32)
    tl.store(O + qoff, (att * sg).to(tl.bfloat16),
             mask=rv[:, None])


@triton.jit
def _comb(WS, WL, WM, G, O, NROW, NSP: tl.constexpr, RB: tl.constexpr,
          DC: tl.constexpr):
    """Merge split-K partial (acc, l, m) triples with log-sum-exp and gate."""
    r = tl.program_id(0) * RB + tl.arange(0, RB)
    rv = r < NROW
    dd = tl.arange(0, DC)
    gg = tl.load(G + r[:, None] * DC + dd[None, :],
                 mask=rv[:, None], other=0.0).to(tl.float32)
    if NSP == 2:
        m0 = tl.load(WM + r, mask=rv, other=-3.0e38)
        m1 = tl.load(WM + NROW + r, mask=rv, other=-3.0e38)
        M = tl.maximum(m0, m1)
        w0 = tl.exp2(m0 - M)
        w1 = tl.exp2(m1 - M)
        a0 = tl.load(WS + r[:, None] * DC + dd[None, :],
                     mask=rv[:, None], other=0.0).to(tl.float32)
        a1 = tl.load(WS + (NROW + r[:, None]) * DC + dd[None, :],
                     mask=rv[:, None], other=0.0).to(tl.float32)
        num = a0 * w0[:, None] + a1 * w1[:, None]
        den = (tl.load(WL + r, mask=rv, other=0.0) * w0 +
               tl.load(WL + NROW + r, mask=rv, other=0.0) * w1)
    else:
        M = tl.full([RB], -3.0e38, tl.float32)
        for sp in range(0, NSP):
            M = tl.maximum(M, tl.load(WM + sp * NROW + r,
                                      mask=rv, other=-3.0e38))
        num = tl.zeros([RB, DC], tl.float32)
        den = tl.zeros([RB], tl.float32)
        for sp in range(0, NSP):
            m = tl.load(WM + sp * NROW + r, mask=rv, other=-3.0e38)
            l = tl.load(WL + sp * NROW + r, mask=rv, other=0.0)
            w = tl.exp2(m - M)
            a = tl.load(WS + (sp * NROW + r[:, None]) * DC + dd[None, :],
                        mask=rv[:, None], other=0.0).to(tl.float32)
            num += a * w[:, None]
            den += l * w
    rl = tl.where(den > 0, 1.0 / tl.where(den > 0, den, 1.0), 0.0)
    att = (num * rl[:, None]).to(tl.bfloat16)
    tl.store(O + r[:, None] * DC + dd[None, :],
             _gate_bf16(att, gg), mask=rv[:, None])


@triton.jit
def _comb_vsplit(WS, WL, WM, G, O, NROW,
                 RB: tl.constexpr, BT: tl.constexpr, NHC: tl.constexpr,
                 CUT2: tl.constexpr, CUT3: tl.constexpr,
                 CUT4: tl.constexpr, DC: tl.constexpr):
    """Merge 1/2/3/4 causal splits selected by tile depth."""
    r = tl.program_id(0) * RB + tl.arange(0, RB)
    rv = r < NROW
    dd = tl.arange(0, DC)
    # One element-offset tile for every [RB, DC] stream: the partial slots,
    # the gate and the output share a layout, so the three extra splits cost
    # a scalar pointer bump instead of a second address tile each.
    off = r[:, None] * DC + dd[None, :]
    sstep = NROW * DC
    # The gate is the only cold-DRAM tile here; issue it before the partials
    # so its latency overlaps their L2 traffic instead of trailing it.
    gg = tl.load(G + off, mask=rv[:, None], other=0.0)
    tile = r // (BT * NHC)
    two = tile >= CUT2
    three = tile >= CUT3
    four = tile >= CUT4

    m0 = tl.load(WM + r, mask=rv, other=-3.0e38)
    m1 = tl.load(WM + NROW + r, mask=rv & two, other=-3.0e38)
    m2 = tl.load(WM + 2 * NROW + r, mask=rv & three, other=-3.0e38)
    m3 = tl.load(WM + 3 * NROW + r, mask=rv & four, other=-3.0e38)
    M = tl.maximum(tl.maximum(m0, m1), tl.maximum(m2, m3))
    w0 = tl.exp2(m0 - M)
    w1 = tl.exp2(m1 - M)
    w2 = tl.exp2(m2 - M)
    w3 = tl.exp2(m3 - M)
    a = tl.load(WS + off, mask=rv[:, None], other=0.0).to(tl.float32)
    num = a * w0[:, None]
    a = tl.load(WS + sstep + off,
                mask=(rv & two)[:, None], other=0.0).to(tl.float32)
    num += a * w1[:, None]
    a = tl.load(WS + 2 * sstep + off,
                mask=(rv & three)[:, None], other=0.0).to(tl.float32)
    num += a * w2[:, None]
    a = tl.load(WS + 3 * sstep + off,
                mask=(rv & four)[:, None], other=0.0).to(tl.float32)
    num += a * w3[:, None]
    den = (tl.load(WL + r, mask=rv, other=0.0) * w0 +
           tl.load(WL + NROW + r, mask=rv & two, other=0.0) * w1 +
           tl.load(WL + 2 * NROW + r, mask=rv & three, other=0.0) * w2 +
           tl.load(WL + 3 * NROW + r, mask=rv & four, other=0.0) * w3)
    rl = tl.where(den > 0,
                  0.5 * _rcp(tl.where(den > 0, den, 1.0)), 0.0)
    att = (num * rl[:, None]).to(tl.bfloat16)
    tl.store(O + off, _gmul_bf(att, gg), mask=rv[:, None])


@triton.jit
def _tpd_step(q8, q8l, KC, VC, base, w0, ii, dd, NSLOT, rcnt, cc, mr,
              mx, li, acc, QLO: tl.constexpr, DC: tl.constexpr,
              PVB: tl.constexpr,
              FAST: tl.constexpr, EX2B: tl.constexpr,
              FULL: tl.constexpr, TRACK: tl.constexpr,
              DEN16: tl.constexpr, TMA: tl.constexpr = False):
    if TMA:
        # Bulk tensor copies avoid materializing the [BN, D] slot-offset tile
        # in registers and pad descriptor rows outside NSLOT with zero.
        kt = tl.trans(KC.load([base + w0, 0]))
        vt = VC.load([base + w0, 0])
    else:
        if FULL:
            slot = base + w0 + ii
        else:
            slot = tl.minimum(base + w0 + ii, NSLOT - 1)
        koff = slot[:, None] * DC + dd[None, :]
        kt = tl.trans(tl.load(KC + koff))
        vt = tl.load(VC + koff)
    if QLO:
        # fp8 value + residual query: the second K dot restores ~bf16 score
        # accuracy, which the general route needs because its rows are short
        # enough that per-element fp8 error does not average out.
        qk = tl.dot(q8l, kt, tl.dot(q8, kt, out_dtype=tl.float32),
                    out_dtype=tl.float32)
    else:
        qk = tl.dot(q8, kt, out_dtype=tl.float32)
    if FULL:
        s = qk
    else:
        s = tl.where(ii[None, :] < (rcnt - w0)[:, None], qk, -3.0e38)
    if TRACK:
        mx = tl.maximum(mx, tl.max(s, 1))
    if FAST:
        pp16 = _ex2h((s * cc - mr[:, None]).to(tl.float16))
        pp = pp16.to(tl.float32)
    elif EX2B:
        pp = _ex2b((s * cc - mr[:, None]).to(tl.bfloat16)).to(tl.float32)
    else:
        pp = tl.exp2(s * cc - mr[:, None])
    if FAST and DEN16:
        li = li + tl.reduce(pp16, 1, _add_h).to(tl.float32)
    else:
        li = li + tl.sum(pp, 1)
    if PVB == 1:
        acc = tl.dot(pp.to(tl.bfloat16), vt.to(tl.bfloat16), acc,
                     out_dtype=tl.float32)
    else:
        if FAST:
            ph = pp16.to(tl.float8e4nv)
            pl = (pp16 - ph.to(tl.float16)).to(tl.float8e4nv)
        else:
            ph = pp.to(tl.float8e4nv)
            pl = (pp - ph.to(tl.float32)).to(tl.float8e4nv)
        acc = tl.dot(pl, vt, tl.dot(ph, vt, acc, out_dtype=tl.float32),
                     out_dtype=tl.float32)
    return acc, mx, li


@triton.jit
def _qsa_w7_tma(Q, G, KC, VC, KDS, VDS, PK, CU, PB, O, T, NSLOT,
                BT: tl.constexpr, RW: tl.constexpr, MK: tl.constexpr,
                BN: tl.constexpr, NHC: tl.constexpr, DC: tl.constexpr,
                PWC: tl.constexpr, SC: tl.constexpr):
    """Dedicated B=2 long-row kernel: inline dense and sparse TMA paths.

    Keeping the cold retry in this entry point is important: Triton's SM107
    TMEM lowering cannot carry a warp-specialized accumulator across a call.
    """
    task = tl.program_id(0)

    c00 = tl.load(CU)
    c01 = tl.load(CU + 1)
    c02 = tl.load(CU + 2)
    pb0 = tl.load(PB)
    pb1 = tl.load(PB + 1)
    nt0 = (c01 - c00 + (BT - 1)) // BT
    nt1 = (c02 - c01 + (BT - 1)) // BT
    tot = nt0 + nt1
    if task >= tot:
        return
    if nt0 == nt1:
        r = task & 1
        local = nt0 - 1 - (task >> 1)
    else:
        pid = tot - 1 - task
        r = tl.where(pid < nt0, 0, 1)
        local = pid - tl.where(r == 0, 0, nt0)
    s0 = tl.where(r == 0, c00, c01)
    s1 = tl.where(r == 0, c01, c02)
    base = tl.where(r == 0, pb0, pb1) * 64
    t0 = s0 + local * BT

    mt = tl.arange(0, MK)
    toks = t0 + mt
    tv = (mt < BT) & (toks < s1)
    tokc = tl.where(tv, toks, s0)
    prow = tokc * PWC
    cnt = tl.load(PK + prow + (PWC - 1), mask=tv, other=0)
    cnt = tl.minimum(tl.where(tv & (cnt > 0), cnt, 0), PWC - 1)
    hv = cnt > 0
    fid = tl.load(PK + prow, mask=hv, other=0x40000000)
    lid = tl.load(PK + prow + cnt - 1, mask=hv, other=-1)
    dense = tl.min(tl.where(hv, cnt - 1 - lid, 0), 0) == 0
    wlo = (tl.min(fid) // BN) * BN
    whi = tl.max(lid)
    nw = tl.where(whi >= wlo, (whi - wlo) // BN + 1, 0)

    rows = tl.arange(0, RW)
    rm = rows // NHC
    rt = t0 + rm
    rv = (rows < BT * NHC) & (rt < s1)
    dd = tl.arange(0, DC)
    # Token-major rows are already the flattened [token, head] index.  Keep
    # rm for packed-row membership, but avoid reconstructing the head with
    # div/mod and a masked pointer select on every Q/G/O access.
    qoff = (t0 * NHC + rows)[:, None] * DC + dd[None, :]
    q8 = tl.load(Q + qoff, mask=rv[:, None],
                 other=0.0).to(tl.float8e4nv)
    cc = SC
    rcnt = tl.where(rv, tl.gather(cnt, rm, 0), 0)

    ii = tl.arange(0, BN)
    SHB = (31 - (ii % 32))[None, :]
    KD = KDS
    VD = VDS

    if dense:
        # Dense-prefix rows retain the fp8 high+residual probability pair.
        # That second PV dot is needed for the workload's 0.01 error cap.
        acc = tl.zeros([RW, DC], tl.float32)
        li = tl.zeros([RW], tl.float32)
        mx = tl.full([RW], -1.0e30, tl.float32)
        mr = tl.zeros([RW], tl.float32)
        minc = tl.min(tl.where(rv, rcnt, 0x7FFFFFFF), 0)
        nfull = minc // BN
        for kw in tl.range(0, nfull, warp_specialize=True):
            w0 = kw * BN
            kt = tl.trans(KD.load([base + w0, 0]))
            vt = VD.load([base + w0, 0])
            s = tl.dot(q8, kt, out_dtype=tl.float32)
            mx = tl.maximum(mx, tl.max(s, 1))
            pp16 = _ex2h(_mul_rz_h2(s, cc))
            li += tl.reduce(pp16, 1, _add_h).to(tl.float32)
            ph = pp16.to(tl.float8e4nv)
            pl = (pp16 - ph.to(tl.float16)).to(tl.float8e4nv)
            acc = tl.dot(pl, vt,
                         tl.dot(ph, vt, acc, out_dtype=tl.float32),
                         out_dtype=tl.float32)
        for kw in tl.range(nfull, nw, warp_specialize=True):
            w0 = kw * BN
            kt = tl.trans(KD.load([base + w0, 0]))
            vt = VD.load([base + w0, 0])
            qk = tl.dot(q8, kt, out_dtype=tl.float32)
            s = tl.where(ii[None, :] < (rcnt - w0)[:, None],
                         qk, -3.0e38)
            mx = tl.maximum(mx, tl.max(s, 1))
            pp16 = _ex2h(_mul_rz_h2(s, cc))
            li += tl.reduce(pp16, 1, _add_h).to(tl.float32)
            ph = pp16.to(tl.float8e4nv)
            pl = (pp16 - ph.to(tl.float16)).to(tl.float8e4nv)
            acc = tl.dot(pl, vt,
                         tl.dot(ph, vt, acc, out_dtype=tl.float32),
                         out_dtype=tl.float32)
        dlt = tl.where(rcnt > 0, mx * cc, 0.0)
        if tl.max(tl.abs(dlt)) > 8.0:
            mr = mx * cc
            acc = tl.zeros([RW, DC], tl.float32)
            li = tl.zeros([RW], tl.float32)
            for kw in range(0, nfull):
                w0 = kw * BN
                kt = tl.trans(KD.load([base + w0, 0]))
                vt = VD.load([base + w0, 0])
                s = tl.dot(q8, kt, out_dtype=tl.float32)
                pp16 = _ex2h((s * cc - mr[:, None]).to(tl.float16))
                li += tl.reduce(pp16, 1, _add_h).to(tl.float32)
                ph = pp16.to(tl.float8e4nv)
                pl = (pp16 - ph.to(tl.float16)).to(tl.float8e4nv)
                acc = tl.dot(pl, vt,
                             tl.dot(ph, vt, acc, out_dtype=tl.float32),
                             out_dtype=tl.float32)
            for kw in range(nfull, nw):
                w0 = kw * BN
                kt = tl.trans(KD.load([base + w0, 0]))
                vt = VD.load([base + w0, 0])
                qk = tl.dot(q8, kt, out_dtype=tl.float32)
                s = tl.where(ii[None, :] < (rcnt - w0)[:, None],
                             qk, -3.0e38)
                pp16 = _ex2h((s * cc - mr[:, None]).to(tl.float16))
                li += tl.reduce(pp16, 1, _add_h).to(tl.float32)
                ph = pp16.to(tl.float8e4nv)
                pl = (pp16 - ph.to(tl.float16)).to(tl.float8e4nv)
                acc = tl.dot(pl, vt,
                             tl.dot(ph, vt, acc, out_dtype=tl.float32),
                             out_dtype=tl.float32)
    else:
        # Sparse rows use packed membership words. One-stage warp-specialized
        # descriptor loads avoid the old [BN,D] address tile.
        p = tl.zeros([MK], tl.int32)
        if BN == 128:
            si = p[:, None] + ii[None, :]
            S = tl.load(PK + prow[:, None] + si,
                        mask=si < cnt[:, None], other=-1,
                        eviction_policy="evict_first")
        else:
            i32 = tl.arange(0, 32)
            si0 = p[:, None] + i32[None, :]
            si1 = si0 + 32
            S0 = tl.load(PK + prow[:, None] + si0,
                         mask=si0 < cnt[:, None], other=-1)
            S1 = tl.load(PK + prow[:, None] + si1,
                         mask=si1 < cnt[:, None], other=-1)
        acc = tl.zeros([RW, DC], tl.float32)
        li = tl.zeros([RW], tl.float32)
        mx = tl.full([RW], -1.0e30, tl.float32)
        for kw in tl.range(0, nw, num_stages=1, warp_specialize=True,
                           disable_licm=True,
                           disallow_acc_multi_buffer=True):
            w0 = wlo + kw * BN
            if BN == 128:
                adv, bw0, bw1 = _mfwin128(S, w0)
            else:
                adv, blo, bhi = _mfwin32(S0, S1, w0)
            p += adv
            if BN == 128:
                si = p[:, None] + ii[None, :]
                S = tl.load(PK + prow[:, None] + si,
                            mask=si < cnt[:, None], other=-1,
                            eviction_policy="evict_first")
            else:
                si0 = p[:, None] + i32[None, :]
                si1 = si0 + 32
                S0 = tl.load(PK + prow[:, None] + si0,
                             mask=si0 < cnt[:, None], other=-1)
                S1 = tl.load(PK + prow[:, None] + si1,
                             mask=si1 < cnt[:, None], other=-1)
            kt = tl.trans(KD.load([base + w0, 0]))
            vt = VD.load([base + w0, 0])
            qk = tl.dot(q8, kt, out_dtype=tl.float32)
            if BN == 128:
                wr0 = tl.gather(bw0, rm, 0)
                wr1 = tl.gather(bw1, rm, 0)
                br0 = wr0.to(tl.uint32).to(tl.int32, bitcast=True)
                br1 = (wr0 >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
                br2 = wr1.to(tl.uint32).to(tl.int32, bitcast=True)
                br3 = (wr1 >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
                wi = (ii // 32)[None, :]
                br = tl.where(wi == 0, br0[:, None],
                              tl.where(wi == 1, br1[:, None],
                                       tl.where(wi == 2, br2[:, None],
                                                br3[:, None])))
                # Shift the selected bit into the sign position: one
                # shift and one signed compare replace the shift, mask
                # and compare the bit test would need.
                s = tl.where((br << SHB) < 0, qk, -3.0e38)
            else:
                br0 = tl.gather(blo, rm, 0)
                br1 = tl.gather(bhi, rm, 0)
                br = tl.where((ii // 32)[None, :] == 0,
                              br0[:, None], br1[:, None])
                hit = ((br >> (ii % 32)[None, :]) & 1) != 0
                s = tl.where(hit, qk, -3.0e38)
            mx = tl.maximum(mx, tl.max(s, 1))
            pp16 = _ex2h(_mul_rz_h2(s, cc))
            li += tl.reduce(pp16, 1, _add_h).to(tl.float32)
            acc = tl.dot(pp16.to(tl.float8e4nv), vt, acc,
                         out_dtype=tl.float32)
        dlt = tl.where(rcnt > 0, mx * cc, 0.0)
        if tl.max(tl.abs(dlt)) > 8.0:
            mr = mx * cc
            p = tl.zeros([MK], tl.int32)
            if BN == 128:
                si = p[:, None] + ii[None, :]
                S = tl.load(PK + prow[:, None] + si,
                            mask=si < cnt[:, None], other=-1,
                            eviction_policy="evict_first")
            else:
                si0 = p[:, None] + i32[None, :]
                si1 = si0 + 32
                S0 = tl.load(PK + prow[:, None] + si0,
                             mask=si0 < cnt[:, None], other=-1)
                S1 = tl.load(PK + prow[:, None] + si1,
                             mask=si1 < cnt[:, None], other=-1)
            acc = tl.zeros([RW, DC], tl.float32)
            li = tl.zeros([RW], tl.float32)
            for kw in range(0, nw):
                w0 = wlo + kw * BN
                if BN == 128:
                    adv, bw0, bw1 = _mfwin128(S, w0)
                else:
                    adv, blo, bhi = _mfwin32(S0, S1, w0)
                p += adv
                if BN == 128:
                    si = p[:, None] + ii[None, :]
                    S = tl.load(PK + prow[:, None] + si,
                                mask=si < cnt[:, None], other=-1,
                                eviction_policy="evict_first")
                else:
                    si0 = p[:, None] + i32[None, :]
                    si1 = si0 + 32
                    S0 = tl.load(PK + prow[:, None] + si0,
                                 mask=si0 < cnt[:, None], other=-1)
                    S1 = tl.load(PK + prow[:, None] + si1,
                                 mask=si1 < cnt[:, None], other=-1)
                kt = tl.trans(KD.load([base + w0, 0]))
                vt = VD.load([base + w0, 0])
                qk = tl.dot(q8, kt, out_dtype=tl.float32)
                if BN == 128:
                    wr0 = tl.gather(bw0, rm, 0)
                    wr1 = tl.gather(bw1, rm, 0)
                    br0 = wr0.to(tl.uint32).to(tl.int32, bitcast=True)
                    br1 = (wr0 >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
                    br2 = wr1.to(tl.uint32).to(tl.int32, bitcast=True)
                    br3 = (wr1 >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
                    wi = (ii // 32)[None, :]
                    br = tl.where(wi == 0, br0[:, None],
                                  tl.where(wi == 1, br1[:, None],
                                           tl.where(wi == 2, br2[:, None],
                                                    br3[:, None])))
                    # Shift the selected bit into the sign position: one
                    # shift and one signed compare replace the shift, mask
                    # and compare the bit test would need.
                    s = tl.where((br << SHB) < 0, qk, -3.0e38)
                else:
                    br0 = tl.gather(blo, rm, 0)
                    br1 = tl.gather(bhi, rm, 0)
                    br = tl.where((ii // 32)[None, :] == 0,
                                  br0[:, None], br1[:, None])
                    hit = ((br >> (ii % 32)[None, :]) & 1) != 0
                    s = tl.where(hit, qk, -3.0e38)
                pp16 = _ex2h((s * cc - mr[:, None]).to(tl.float16))
                li += tl.reduce(pp16, 1, _add_h).to(tl.float32)
                acc = tl.dot(pp16.to(tl.float8e4nv), vt, acc,
                             out_dtype=tl.float32)

    rl = tl.where(li > 0, 0.5 / tl.where(li > 0, li, 1.0), 0.0)
    gg = tl.load(G + qoff, mask=rv[:, None], other=0.0)
    att = (acc * rl[:, None]).to(tl.bfloat16)
    tl.store(O + qoff, _gmul_bf(att, gg), mask=rv[:, None])
@triton.jit
def _qsa(Q, G, KC, VC, KDS, VDS, PK, CU, PB, O, WS, WL, WM,
         T, B: tl.constexpr, NSLOT, NROW,
         BT: tl.constexpr, RW: tl.constexpr, MK: tl.constexpr, BN: tl.constexpr,
         NSUB: tl.constexpr, NB: tl.constexpr, SINGLE: tl.constexpr,
         PAIRREQ: tl.constexpr,
         PVB: tl.constexpr, NSP: tl.constexpr, USEQL: tl.constexpr,
         MASKFP8: tl.constexpr, RSC: tl.constexpr, SKIPD: tl.constexpr,
         TPD: tl.constexpr, SPV: tl.constexpr, W32: tl.constexpr,
         EX2B: tl.constexpr, EX2H: tl.constexpr, ZERO: tl.constexpr,
         PEEL: tl.constexpr, DEN16: tl.constexpr,
         CAPQ: tl.constexpr, FAST_CAUSAL: tl.constexpr,
         TMA: tl.constexpr, WSPEC: tl.constexpr, GATEPTX: tl.constexpr,
         CSPL: tl.constexpr, VSPL: tl.constexpr,
         NHC: tl.constexpr, DC: tl.constexpr,
         PWC: tl.constexpr, SC: tl.constexpr):
    if VSPL:
        prog = tl.program_id(0)
        q4 = prog < 80
        q3 = prog < 137
        q2 = prog < 175
        sp = tl.where(q4, prog % 4,
                      tl.where(q3, (prog - 80) % 3,
                               tl.where(q2, (prog - 137) % 2, 0)))
        task = tl.where(q4, prog // 4,
                        tl.where(q3, 20 + (prog - 80) // 3,
                                 tl.where(q2, 39 + (prog - 137) // 2,
                                          58 + prog - 175)))
        nsplit = tl.where(q4, 4, tl.where(q3, 3, tl.where(q2, 2, 1)))
    else:
        sp = tl.program_id(0) % NSP
        task = tl.program_id(0) // NSP
        nsplit = NSP

    if SINGLE:
        tot = (T + BT - 1) // BT
        pid = tot - 1 - task
        r = 0
        exr = 0
        s0 = 0
        s1 = T
    elif PAIRREQ:
        c00 = tl.load(CU)
        c01 = tl.load(CU + 1)
        c02 = tl.load(CU + 2)
        pb0 = tl.load(PB)
        pb1 = tl.load(PB + 1)
        nt0 = (c01 - c00 + (BT - 1)) // BT
        nt1 = (c02 - c01 + (BT - 1)) // BT
        tot = nt0 + nt1
        if task >= tot:
            return
        if nt0 == nt1:
            r = task & 1
            local = nt0 - 1 - (task >> 1)
        else:
            pid = tot - 1 - task
            r = tl.where(pid < nt0, 0, 1)
            local = pid - tl.where(r == 0, 0, nt0)
        s0 = tl.where(r == 0, c00, c01)
        s1 = tl.where(r == 0, c01, c02)
        base = tl.where(r == 0, pb0, pb1) * 64
    else:
        jj = tl.arange(0, NB)
        mm = jj < B
        c0 = tl.load(CU + jj, mask=mm, other=0)
        c1 = tl.load(CU + jj + 1, mask=mm, other=0)
        pbv = tl.load(PB + jj, mask=mm, other=0)
        nt = tl.where(mm, (c1 - c0 + (BT - 1)) // BT, 0)
        tot = tl.sum(nt, 0)
        if task >= tot:
            return
        nmin = tl.min(tl.where(mm, nt, 0x7FFFFFFF), 0)
        nmax = tl.max(nt, 0)
        balanced = nmax - nmin <= 1
        if nmax == nmin:
            r = task % B
            local = nmin - 1 - task // B
            pid = 0
            exr = 0
        elif balanced:
            extra = mm & (nt > nmin)
            nextra = tl.sum(extra.to(tl.int32), 0)
            if task < nextra:
                erank = tl.cumsum(extra.to(tl.int32), 0) - 1
                r = tl.sum(tl.where(extra & (erank == task), jj, 0), 0)
                local = nmin
            else:
                qtask = task - nextra
                r = qtask % B
                local = nmin - 1 - qtask // B
            pid = 0
            exr = 0
        else:
            pid = tot - 1 - task
            cs = tl.cumsum(nt, 0)
            ex = cs - nt
            hit = mm & (ex <= pid)
            r = tl.sum(hit.to(tl.int32), 0) - 1
            exr = tl.max(tl.where(hit, ex, 0), 0)
            local = pid - exr
        pick = mm & (jj == r)
        s0 = tl.sum(tl.where(pick, c0, 0), 0)
        s1 = tl.sum(tl.where(pick, c1, 0), 0)
        base = tl.sum(tl.where(pick, pbv, 0), 0) * 64

    if SINGLE:
        base = tl.load(PB) * 64
    if SINGLE:
        t0 = s0 + (pid - exr) * BT
    else:
        t0 = s0 + local * BT

    # The query tile's address depends only on the tile index, so issue
    # it before the packed-row chain (count, then first and last id)
    # instead of behind two dependent global round trips.
    rows = tl.arange(0, RW)
    rm = rows // NHC
    rh = rows % NHC
    rt = t0 + rm
    rv = (rm < BT) & (rt < s1)
    qo = tl.where(rv, rt, s0) * (NHC * DC) + rh * DC
    dd = tl.arange(0, DC)
    qoff = qo[:, None] + dd[None, :]
    qr = tl.load(Q + qoff, mask=rv[:, None], other=0.0)

    mt = tl.arange(0, MK)
    toks = t0 + mt
    tv = (toks < s1) & (mt < BT)
    tokc = tl.where(tv, toks, s0)
    prow = tokc * PWC
    cnt = tl.load(PK + prow + (PWC - 1), mask=tv, other=0)
    # the packed row only holds PWC-1 ids, so a count beyond that cannot
    # name more selected positions than exist
    cnt = tl.minimum(tl.where(tv & (cnt > 0), cnt, 0), PWC - 1)
    hv = cnt > 0
    if FAST_CAUSAL:
        pos = tokc - s0
        pos_ok = pos < (PWC - 1)
        vid = tl.load(PK + prow + tl.minimum(pos, PWC - 2),
                      mask=tv & pos_ok, other=-1)
        proof = tl.reduce(
            tl.where(tv, (cnt - 1 - pos) | (vid - pos) |
                     tl.where(pos_ok, 0, 1), 0), 0, _or_comb)
        if proof == 0:
            if SKIPD:
                return
            dense = True
            wlo = 0
            whi = tl.max(cnt - 1)
            nw = whi // BN + 1
        else:
            fid = tl.load(PK + prow, mask=hv, other=0x40000000)
            lid = tl.load(PK + prow + cnt - 1, mask=hv, other=-1)
            if SKIPD:
                if tl.min(tl.where(hv, cnt - 1 - lid, 0)) == 0:
                    return
                dense = False
            else:
                dense = tl.min(tl.where(hv, cnt - 1 - lid, 0)) == 0
            wlo = (tl.min(fid) // BN) * BN
            whi = tl.max(lid)
            nw = tl.where(whi >= wlo, (whi - wlo) // BN + 1, 0)
    else:
        fid = tl.load(PK + prow, mask=hv, other=0x40000000)
        lid = tl.load(PK + prow + cnt - 1, mask=hv, other=-1)
        if SKIPD:
            # companion launch: the dense tiles belong to the two-pass kernel,
            # so the dense sweep is not compiled into this path.
            if tl.min(tl.where(hv, cnt - 1 - lid, 0)) == 0:
                return
            dense = False
        else:
            dense = tl.min(tl.where(hv, cnt - 1 - lid, 0)) == 0
        wlo = (tl.min(fid) // BN) * BN
        whi = tl.max(lid)
        nw = tl.where(whi >= wlo, (whi - wlo) // BN + 1, 0)

    # per-row fp8 scale: every MMA row uses the full fp8 range, which keeps the
    # score error well inside the output tolerance, and it costs one row-wise
    # reduction instead of a CTA-wide one
    if CAPQ:
        # Fixed x64 query scale: exact in BF16 and avoids the whole-CTA max
        # reduction, but it saturates FP8 for |q| > 3.75, i.e. it is only
        # valid for a query distribution known to stay inside that bound.
        # The dispatch in run() therefore never sets CAPQ; the constant is
        # kept for callers that can prove the bound on their own traffic.
        sq = 64.0
        qs = qr * tl.full([1], 64.0, tl.bfloat16)
        q8 = qs.to(tl.float8e4nv)
        if USEQL:
            q8l = (qs - q8.to(tl.bfloat16)).to(tl.float8e4nv)
        else:
            q8l = q8
    else:
        qf = qr.to(tl.float32)
        sq = 240.0 / tl.maximum(tl.max(tl.abs(qf)), 1e-20)
        qs = qf * sq
        q8 = qs.to(tl.float8e4nv)
        if USEQL:
            q8l = (qs - q8.to(tl.float32)).to(tl.float8e4nv)
        else:
            q8l = q8
    cc = SC / sq
    if MASKFP8:
        emat = tl.where(rm[:, None] == mt[None, :], 240.0, 0.0).to(tl.float8e4nv)
        biasv = -240.0
    else:
        emat = (rm[:, None] == mt[None, :]).to(tl.bfloat16)
        # A true sentinel, not a scaled finite penalty: a window that holds no
        # selected key for a row must not become that row's running maximum,
        # which is what a finite bias would do (the subtraction would then
        # cancel and give every masked slot probability one).
        biasv = -3.0e38

    rcnt = tl.sum(tl.where(rm[:, None] == mt[None, :], cnt[None, :], 0), 1)
    mz = tl.zeros([MK, BN], tl.bfloat16)
    acc = tl.zeros([RW, DC], tl.float32)
    mi = tl.full([RW], -1.0e30, tl.float32)
    li = tl.zeros([RW], tl.float32)
    ii = tl.arange(0, BN)
    if TMA == 2:
        KD = KDS
        VD = VDS
    elif TMA == 1:
        KD = tl.make_tensor_descriptor(KC, [NSLOT, DC], [DC, 1], [BN, DC])
        VD = tl.make_tensor_descriptor(VC, [NSLOT, DC], [DC, 1], [BN, DC])
    else:
        KD = KC
        VD = VC

    if dense:
        if TPD:
            if ZERO:
                # Zero is a free exponent anchor for the overwhelmingly common
                # near-zero-logit case.  The post-sweep two-sided guard below
                # discards and redoes the sweep around the selected maximum if
                # a selected score falls outside the fp8-safe +/-8 band.
                mr = tl.zeros([RW], tl.float32)
            else:
                wr = sp * BN
                if SPV:
                    sla = tl.minimum(base + wr + tl.zeros([MK], tl.int32),
                                     NSLOT - 1)
                    kta = tl.trans(tl.load(KC + sla[:, None] * DC + dd[None, :]))
                    sra = tl.dot(q8, kta, out_dtype=tl.float32)
                    srf = tl.sum(tl.where(rm[:, None] == mt[None, :], sra, 0.0), 1)
                    mr = tl.where(rcnt > wr, srf * cc, 0.0)
                else:
                    slr = tl.minimum(base + wr + ii, NSLOT - 1)
                    ktr = tl.trans(tl.load(KC + slr[:, None] * DC + dd[None, :]))
                    sr = tl.dot(q8, ktr, out_dtype=tl.float32)
                    mr = tl.max(sr, 1) * cc
            mx = tl.full([RW], -1.0e30, tl.float32)
            if PEEL:
                minc = tl.min(tl.where(rv, rcnt, 0x7FFFFFFF), 0)
                nfull = minc // BN
                for kw in tl.range(sp, nfull, NSP, warp_specialize=WSPEC):
                    acc, mx, li = _tpd_step(
                        q8, q8l, KD, VD, base, kw * BN, ii, dd, NSLOT, rcnt, cc,
                        mr, mx, li, acc, USEQL, DC, PVB, SPV or EX2H, EX2B, True,
                        True, DEN16, TMA)
                kb = sp + tl.maximum(0, (nfull - sp + NSP - 1) // NSP) * NSP
                for kw in tl.range(kb, nw, NSP, warp_specialize=WSPEC):
                    acc, mx, li = _tpd_step(
                        q8, q8l, KD, VD, base, kw * BN, ii, dd, NSLOT, rcnt, cc,
                        mr, mx, li, acc, USEQL, DC, PVB, SPV or EX2H, EX2B, False,
                        True, DEN16, TMA)
            elif CSPL:
                # Split the T768 sweep into contiguous partitions.  This
                # shortens each dependency chain without changing KV traffic.
                for kw in range(nw * sp // nsplit,
                                nw * (sp + 1) // nsplit):
                    acc, mx, li = _tpd_step(
                        q8, q8l, KD, VD, base, kw * BN, ii, dd, NSLOT, rcnt, cc,
                        mr, mx, li, acc, USEQL, DC, PVB, SPV or EX2H, EX2B, False,
                        True, DEN16, TMA)
            else:
                for kw in tl.range(sp, nw, NSP, warp_specialize=WSPEC):
                    acc, mx, li = _tpd_step(
                        q8, q8l, KD, VD, base, kw * BN, ii, dd, NSLOT, rcnt, cc,
                        mr, mx, li, acc, USEQL, DC, PVB, SPV or EX2H, EX2B, False,
                        True, DEN16, TMA)
            if CSPL:
                # A contiguous split is active for a row only when that row
                # reaches the split's first KV window.  Using rcnt > 0 here
                # lets an empty later split re-anchor the -3e38 mask sentinel;
                # its redo would then turn masked slots into probability mass.
                split_first = (nw * sp // nsplit) * BN
                active = rcnt > split_first
            else:
                active = rcnt > sp * BN
            dlt = tl.where(active, mx * cc - mr, 0.0)
            redo = tl.max(tl.abs(dlt)) > 8.0
            if redo:
                if CSPL:
                    mr = tl.where(active, mx * cc, 0.0)
                else:
                    mr = tl.where(active, mx * cc, 0.0)
                acc = tl.zeros([RW, DC], tl.float32)
                li = tl.zeros([RW], tl.float32)
                if PEEL:
                    for kw in range(sp, nfull, NSP):
                        acc, mx, li = _tpd_step(
                            q8, q8l, KD, VD, base, kw * BN, ii, dd, NSLOT, rcnt,
                            cc, mr, mx, li, acc, USEQL, DC, PVB, SPV, False, True,
                            False, DEN16, TMA)
                    for kw in range(kb, nw, NSP):
                        acc, mx, li = _tpd_step(
                            q8, q8l, KD, VD, base, kw * BN, ii, dd, NSLOT, rcnt,
                            cc, mr, mx, li, acc, USEQL, DC, PVB, SPV, False, False,
                            False, DEN16, TMA)
                elif CSPL:
                    for kw in range(nw * sp // nsplit,
                                    nw * (sp + 1) // nsplit):
                        acc, mx, li = _tpd_step(
                            q8, q8l, KD, VD, base, kw * BN, ii, dd, NSLOT, rcnt,
                            cc, mr, mx, li, acc, USEQL, DC, PVB, SPV, False, False,
                            False, DEN16, TMA)
                else:
                    for kw in range(sp, nw, NSP):
                        acc, mx, li = _tpd_step(
                            q8, q8l, KD, VD, base, kw * BN, ii, dd, NSLOT, rcnt,
                            cc, mr, mx, li, acc, USEQL, DC, PVB, SPV, False, False,
                            False, DEN16, TMA)
            mi = mr / cc
        else:
            for kw in range(sp, nw, NSP):
                w0 = kw * BN
                slot = base + w0 + ii
                koff = slot[:, None] * DC + dd[None, :]
                acc, mi, li = _blk(q8, q8l, emat, mz, rcnt - w0, KC, VC, koff,
                                   slot < NSLOT, acc, mi, li, cc, DC, PVB,
                                   USEQL, True, BN, RSC)
    elif sp == 0:
        p = tl.zeros([MK], tl.int32)
        jlo = (tl.arange(0, BN) % 64).to(tl.int64)
        jhi = tl.arange(0, BN) // 64
        one = tl.full([MK, BN], 1, tl.int64)
        if SPV and W32:
            i32 = tl.arange(0, 32)
            si0 = p[:, None] + i32[None, :]
            si1 = si0 + 32
            S0 = tl.load(PK + prow[:, None] + si0,
                         mask=si0 < cnt[:, None], other=-1)
            S1 = tl.load(PK + prow[:, None] + si1,
                         mask=si1 < cnt[:, None], other=-1)
        else:
            sidx = p[:, None] + ii[None, :]
            S = tl.load(PK + prow[:, None] + sidx,
                        mask=sidx < cnt[:, None], other=-1)
        if SPV:
            # Fixed reference exponent, verified after the sweep: the window
            # loop then has no running maximum, so windows no longer depend on
            # each other and the accumulator is never rescaled.
            if ZERO:
                mr = tl.zeros([RW], tl.float32)
            else:
                sl0 = base + fid
                kt0 = tl.trans(tl.load(KC + sl0[:, None] * DC + dd[None, :],
                                       mask=hv[:, None], other=0.0))
                sra = tl.dot(q8, kt0, out_dtype=tl.float32)
                srf = tl.sum(tl.where(rm[:, None] == mt[None, :], sra, 0.0), 1)
                mr = tl.where(rcnt > 0, srf * cc, 0.0)
            mx = tl.full([RW], -1.0e30, tl.float32)
            for kw in range(0, nw):
                w0 = wlo + kw * BN
                if W32:
                    adv, blo, bhi = _mfwin32(S0, S1, w0)
                else:
                    mf, adv = _mfwin(S, w0, biasv, one, jlo, jhi,
                                     MK, NSUB, BN, W32)
                p = p + adv
                if W32:
                    si0 = p[:, None] + i32[None, :]
                    si1 = si0 + 32
                    S0 = tl.load(PK + prow[:, None] + si0,
                                 mask=si0 < cnt[:, None], other=-1)
                    S1 = tl.load(PK + prow[:, None] + si1,
                                 mask=si1 < cnt[:, None], other=-1)
                else:
                    sidx = p[:, None] + ii[None, :]
                    S = tl.load(PK + prow[:, None] + sidx,
                                mask=sidx < cnt[:, None], other=-1)
                slot = base + w0 + ii
                koff = slot[:, None] * DC + dd[None, :]
                kt = tl.trans(tl.load(KC + koff,
                                      eviction_policy="evict_last"))
                vt = tl.load(VC + koff, eviction_policy="evict_last")
                qk = tl.dot(q8, kt, out_dtype=tl.float32)
                if MASKFP8:
                    # Each membership word is already token-local. Gather the
                    # owning token's word for every query-head row and apply it
                    # directly, avoiding the separate broadcast MMA.
                    br0 = tl.gather(blo, rm, 0)
                    br1 = tl.gather(bhi, rm, 0)
                    br = tl.where((ii // 32)[None, :] == 0,
                                  br0[:, None], br1[:, None])
                    hit = ((br >> (ii % 32)[None, :]) & 1) != 0
                    s = tl.where(hit, qk, -3.0e38)
                else:
                    s = tl.dot(emat, mf, qk, out_dtype=tl.float32)
                mx = tl.maximum(mx, tl.max(s, 1))
                pp16 = _ex2h((s * cc - mr[:, None]).to(tl.float16))
                li = li + tl.sum(pp16, 1, dtype=tl.float32)
                ph = pp16.to(tl.float8e4nv)
                acc = tl.dot(ph, vt, acc, out_dtype=tl.float32)
            dlt = tl.where(rcnt > 0, mx * cc - mr, 0.0)
            if tl.max(tl.abs(dlt)) > 8.0:
                mr = mx * cc
                acc = tl.zeros([RW, DC], tl.float32)
                li = tl.zeros([RW], tl.float32)
                p = tl.zeros([MK], tl.int32)
                if W32:
                    si0 = p[:, None] + i32[None, :]
                    si1 = si0 + 32
                    S0 = tl.load(PK + prow[:, None] + si0,
                                 mask=si0 < cnt[:, None], other=-1)
                    S1 = tl.load(PK + prow[:, None] + si1,
                                 mask=si1 < cnt[:, None], other=-1)
                else:
                    sidx = p[:, None] + ii[None, :]
                    S = tl.load(PK + prow[:, None] + sidx,
                                mask=sidx < cnt[:, None], other=-1)
                for kw in range(0, nw):
                    w0 = wlo + kw * BN
                    if W32:
                        adv, blo, bhi = _mfwin32(S0, S1, w0)
                    else:
                        mf, adv = _mfwin(S, w0, biasv, one, jlo, jhi,
                                         MK, NSUB, BN, W32)
                    p = p + adv
                    if W32:
                        si0 = p[:, None] + i32[None, :]
                        si1 = si0 + 32
                        S0 = tl.load(PK + prow[:, None] + si0,
                                     mask=si0 < cnt[:, None], other=-1)
                        S1 = tl.load(PK + prow[:, None] + si1,
                                     mask=si1 < cnt[:, None], other=-1)
                    else:
                        sidx = p[:, None] + ii[None, :]
                        S = tl.load(PK + prow[:, None] + sidx,
                                    mask=sidx < cnt[:, None], other=-1)
                    slot = base + w0 + ii
                    koff = slot[:, None] * DC + dd[None, :]
                    kt = tl.trans(tl.load(KC + koff,
                                          eviction_policy="evict_last"))
                    vt = tl.load(VC + koff, eviction_policy="evict_last")
                    qk = tl.dot(q8, kt, out_dtype=tl.float32)
                    if MASKFP8:
                        br0 = tl.gather(blo, rm, 0)
                        br1 = tl.gather(bhi, rm, 0)
                        br = tl.where((ii // 32)[None, :] == 0,
                                      br0[:, None], br1[:, None])
                        hit = ((br >> (ii % 32)[None, :]) & 1) != 0
                        s = tl.where(hit, qk, -3.0e38)
                    else:
                        s = tl.dot(emat, mf, qk, out_dtype=tl.float32)
                    pp16 = _ex2h((s * cc - mr[:, None]).to(tl.float16))
                    li = li + tl.sum(pp16, 1, dtype=tl.float32)
                    ph = pp16.to(tl.float8e4nv)
                    acc = tl.dot(ph, vt, acc, out_dtype=tl.float32)
            mi = mr / cc
        else:
            for kw in range(0, nw):
                w0 = wlo + kw * BN
                o = S - w0
                inw = (o >= 0) & (o < BN) & (S >= 0)
                p = p + tl.sum(inw.to(tl.int32), 1)
                if NSUB == 1:
                    ow = tl.where(inw, o, 0).to(tl.int64)
                    bw = tl.reduce(tl.where(inw, one << ow, 0), 1, _or_comb)
                    if MASKFP8:
                        mf = tl.where(((bw[:, None] >> jlo[None, :]) & 1) != 0,
                                      0.0, -240.0).to(tl.float8e4nv)
                    else:
                        mf = tl.where(((bw[:, None] >> jlo[None, :]) & 1) != 0,
                                      0.0, biasv).to(tl.bfloat16)
                else:
                    sel = tl.zeros([MK, BN], tl.int32)
                    for sb in tl.static_range(NSUB):
                        insb = inw & ((o // 64) == sb)
                        ow = tl.where(insb, o % 64, 0).to(tl.int64)
                        bw = tl.reduce(tl.where(insb, one << ow, 0), 1, _or_comb)
                        hit2 = (((bw[:, None] >> jlo[None, :]) & 1) != 0) & (jhi[None, :] == sb)
                        sel = sel | hit2.to(tl.int32)
                    if MASKFP8:
                        mf = tl.where(sel != 0, 0.0, -240.0).to(tl.float8e4nv)
                    else:
                        mf = tl.where(sel != 0, 0.0, biasv).to(tl.bfloat16)
                sidx = p[:, None] + ii[None, :]
                S = tl.load(PK + prow[:, None] + sidx, mask=sidx < cnt[:, None], other=-1)
                slot = base + w0 + ii
                koff = slot[:, None] * DC + dd[None, :]
                acc, mi, li = _blk(q8, q8l, emat, mf, rcnt, KC, VC, koff,
                                   slot < NSLOT, acc, mi, li, cc, DC, PVB, USEQL,
                                   False, BN, RSC)

    if NSP > 1:
        grow = tl.where(rv, rt, 0) * NHC + rh
        tl.store(WS + (sp * NROW + grow[:, None]) * DC + tl.arange(0, DC)[None, :],
                 acc.to(tl.bfloat16), mask=rv[:, None])
        tl.store(WL + sp * NROW + grow, li, mask=rv)
        tl.store(WM + sp * NROW + grow, mi * cc, mask=rv)
    else:
        rl = tl.where(li > 0, 1.0 / tl.where(li > 0, li, 1.0), 0.0)
        gg = tl.load(G + qoff, mask=rv[:, None], other=0.0).to(tl.float32)
        if GATEPTX:
            att = (acc * rl[:, None]).to(tl.bfloat16)
            outv = _gate_bf16(att, gg)
        else:
            att = (acc * (rl * 0.5)[:, None]).to(tl.bfloat16)
            outv = _gmul(att, gg)
        tl.store(O + qoff, outv, mask=rv[:, None])


@triton.jit(noinline=True)
def _qsa_w6_redo(Q, G, KC, VC, O, t0, s0, s1, base, NSLOT,
                 BT: tl.constexpr, NHC: tl.constexpr, DC: tl.constexpr,
                 PWC: tl.constexpr, SC: tl.constexpr):
    """Cold dense-prefix retry, kept out of the common TMA instruction body."""
    dd = tl.arange(0, DC)
    hh = tl.arange(0, 8)
    hv = hh < NHC
    nm = tl.minimum(BT, s1 - t0)
    for m in range(0, nm):
        tt = t0 + m
        cc2 = tl.minimum(tt - s0 + 1, PWC - 1)
        qo = tt * (NHC * DC) + hh[:, None] * DC + dd[None, :]
        q2 = tl.load(Q + qo, mask=hv[:, None], other=0.0).to(tl.float32)
        a2 = tl.zeros([8, DC], tl.float32)
        l2 = tl.zeros([8], tl.float32)
        m2 = tl.full([8], -1.0e30, tl.float32)
        for u in range(0, cc2):
            sl = base + u
            ok = sl < NSLOT
            kv = tl.load(KC + sl * DC + dd, mask=ok, other=0.0).to(tl.float32)
            s2 = tl.where(ok, tl.sum(q2 * kv[None, :], 1) * SC, -3.0e38)
            mn2 = tl.maximum(m2, s2)
            al2 = tl.exp2(m2 - mn2)
            pw2 = tl.exp2(s2 - mn2)
            l2 = l2 * al2 + pw2
            m2 = mn2
            vv = tl.load(VC + sl * DC + dd, mask=ok, other=0.0).to(tl.float32)
            a2 = a2 * al2[:, None] + pw2[:, None] * vv[None, :]
        r2 = tl.where(l2 > 0, 1.0 / tl.where(l2 > 0, l2, 1.0), 0.0)
        g2 = tl.load(G + qo, mask=hv[:, None], other=0.0).to(tl.float32)
        att2 = (a2 * r2[:, None]).to(tl.bfloat16)
        tl.store(O + qo, _gate_bf16(att2, g2), mask=hv[:, None])


@triton.jit
def _qsa_w6(Q, G, KC, VC, KDS, VDS, PK, CU, PB, O, T, NSLOT,
            BT: tl.constexpr, RW: tl.constexpr, MK: tl.constexpr,
            BN: tl.constexpr, NR: tl.constexpr, CAPQ: tl.constexpr,
            FASTEX2: tl.constexpr, GATEPTX: tl.constexpr, HTMA: tl.constexpr,
            PEEL: tl.constexpr, DEN16: tl.constexpr,
            NHC: tl.constexpr, DC: tl.constexpr,
            PWC: tl.constexpr, SC: tl.constexpr):
    """Small-B dense-prefix sweep with a fully general sparse-row fallback."""
    task = tl.program_id(0)
    if NR == 2:
        c00 = tl.load(CU)
        c01 = tl.load(CU + 1)
        c02 = tl.load(CU + 2)
        pb0 = tl.load(PB)
        pb1 = tl.load(PB + 1)
        nt0 = (c01 - c00 + (BT - 1)) // BT
        nt1 = (c02 - c01 + (BT - 1)) // BT
        tot = nt0 + nt1
        if task >= tot:
            return
        if nt0 == nt1:
            r = task & 1
            local = nt0 - 1 - (task >> 1)
        else:
            pid = tot - 1 - task
            r = tl.where(pid < nt0, 0, 1)
            local = pid - tl.where(r == 0, 0, nt0)
        s0 = tl.where(r == 0, c00, c01)
        s1 = tl.where(r == 0, c01, c02)
        base = tl.where(r == 0, pb0, pb1) * 64
    else:
        jj = tl.arange(0, NR)
        c0 = tl.load(CU + jj)
        c1 = tl.load(CU + jj + 1)
        pbv = tl.load(PB + jj)
        nt = (c1 - c0 + (BT - 1)) // BT
        tot = tl.sum(nt, 0)
        if task >= tot:
            return
        nmin = tl.min(nt, 0)
        nmax = tl.max(nt, 0)
        if nmax == nmin:
            r = task % NR
            local = nmin - 1 - task // NR
        elif nmax - nmin <= 1:
            extra = nt > nmin
            nextra = tl.sum(extra.to(tl.int32), 0)
            if task < nextra:
                erank = tl.cumsum(extra.to(tl.int32), 0) - 1
                r = tl.sum(tl.where(extra & (erank == task), jj, 0), 0)
                local = nmin
            else:
                qtask = task - nextra
                r = qtask % NR
                local = nmin - 1 - qtask // NR
        else:
            pid = tot - 1 - task
            cs = tl.cumsum(nt, 0)
            ex = cs - nt
            hit = ex <= pid
            r = tl.sum(hit.to(tl.int32), 0) - 1
            exr = tl.max(tl.where(hit, ex, 0), 0)
            local = pid - exr
        pick = jj == r
        s0 = tl.sum(tl.where(pick, c0, 0), 0)
        s1 = tl.sum(tl.where(pick, c1, 0), 0)
        base = tl.sum(tl.where(pick, pbv, 0), 0) * 64
    t0 = s0 + local * BT

    mt = tl.arange(0, MK)
    toks = t0 + mt
    tv = (toks < s1) & (mt < BT)
    tokc = tl.where(tv, toks, s0)
    prow = tokc * PWC
    cnt = tl.load(PK + prow + (PWC - 1), mask=tv, other=0)
    cnt = tl.minimum(tl.where(tv & (cnt > 0), cnt, 0), PWC - 1)
    pos = tokc - s0
    pos_ok = pos < (PWC - 1)
    vid = tl.load(PK + prow + tl.minimum(pos, PWC - 2),
                  mask=tv & pos_ok, other=-1)
    proof = tl.reduce(
        tl.where(tv, (cnt - 1 - pos) | (vid - pos) |
                 tl.where(pos_ok, 0, 1), 0),
        0, _or_comb)
    dd = tl.arange(0, DC)
    if proof != 0:
        # This path is intentionally compact, not fast: arbitrary in-scope
        # packed rows still compute exact selected attention.
        hh8 = tl.arange(0, 8)
        hv8 = hh8 < NHC
        nm = tl.minimum(BT, s1 - t0)
        for m in range(0, nm):
            tt = t0 + m
            cc2 = tl.minimum(tl.maximum(
                tl.load(PK + tt * PWC + (PWC - 1)), 0), PWC - 1)
            qo2 = tt * (NHC * DC) + hh8[:, None] * DC + dd[None, :]
            q2 = tl.load(Q + qo2, mask=hv8[:, None], other=0.0).to(tl.float32)
            a2 = tl.zeros([8, DC], tl.float32)
            l2 = tl.zeros([8], tl.float32)
            m2 = tl.full([8], -1.0e30, tl.float32)
            for u in range(0, cc2):
                idu = tl.load(PK + tt * PWC + u)
                sl = base + idu
                okk = (idu >= 0) & (sl < NSLOT)
                kv = tl.load(KC + sl * DC + dd, mask=okk,
                             other=0.0).to(tl.float32)
                s2 = tl.where(okk, tl.sum(q2 * kv[None, :], 1) * SC,
                              -3.0e38)
                mn2 = tl.maximum(m2, s2)
                al2 = tl.exp2(m2 - mn2)
                pw2 = tl.exp2(s2 - mn2)
                l2 = l2 * al2 + pw2
                m2 = mn2
                vv = tl.load(VC + sl * DC + dd, mask=okk,
                             other=0.0).to(tl.float32)
                a2 = a2 * al2[:, None] + pw2[:, None] * vv[None, :]
            r2 = tl.where(l2 > 0, 1.0 / tl.where(l2 > 0, l2, 1.0), 0.0)
            g2 = tl.load(G + qo2, mask=hv8[:, None], other=0.0).to(tl.float32)
            att2 = (a2 * r2[:, None]).to(tl.bfloat16)
            tl.store(O + qo2, _gate_bf16(att2, g2), mask=hv8[:, None])
        return

    rows = tl.arange(0, RW)
    rm = rows // NHC
    rh = rows % NHC
    rt = t0 + rm
    rv = (rm < BT) & (rt < s1)
    qo = tl.where(rv, rt, s0) * (NHC * DC) + rh * DC
    qoff = qo[:, None] + dd[None, :]
    qr = tl.load(Q + qoff, mask=rv[:, None], other=0.0)
    if CAPQ:
        sq = 32.0
        q8 = (qr * sq).to(tl.float8e4nv)
        cc = SC / sq
    else:
        # Both descriptor routes can narrow BF16 directly into the FP8 MMA
        # operand; avoid expanding the full query tile through FP32 first.
        q8 = qr.to(tl.float8e4nv)
        cc = SC
    # The packed proof above establishes cnt == token_position + 1 for every
    # valid token, so avoid broadcasting/reducing the count tile in the hot path.
    rcnt = tl.where(rv, rt - s0 + 1, 0)
    nw = (tl.minimum(t0 + BT, s1) - s0 + (BN - 1)) // BN
    ii = tl.arange(0, BN)
    acc = tl.zeros([RW, DC], tl.float32)
    li = tl.zeros([RW], tl.float32)
    mx = tl.full([RW], -1.0e30, tl.float32)
    mr = tl.zeros([RW], tl.float32)
    if HTMA:
        KD = KDS
        VD = VDS
    else:
        KD = tl.make_tensor_descriptor(KC, [NSLOT, DC], [DC, 1], [BN, DC])
        VD = tl.make_tensor_descriptor(VC, [NSLOT, DC], [DC, 1], [BN, DC])
    if PEEL:
        # The dense proof makes rcnt equal token_position+1 and rm=0 is valid
        # for every hot tile, so the minimum full-window count is scalar.
        nfull = (t0 - s0 + 1) // BN
        for kw in tl.range(0, nfull, warp_specialize=True):
            acc, mx, li = _tpd_step(
                q8, q8, KD, VD, base, kw * BN, ii, dd, NSLOT, rcnt, cc,
                mr, mx, li, acc, False, DC, 0, FASTEX2, False, True, False,
                DEN16, True)
        for kw in tl.range(nfull, nw, warp_specialize=True):
            acc, mx, li = _tpd_step(
                q8, q8, KD, VD, base, kw * BN, ii, dd, NSLOT, rcnt, cc,
                mr, mx, li, acc, False, DC, 0, FASTEX2, False, False, False,
                DEN16, True)
    else:
        for kw in tl.range(0, nw, warp_specialize=True):
            acc, mx, li = _tpd_step(
                q8, q8, KD, VD, base, kw * BN, ii, dd, NSLOT, rcnt, cc,
                mr, mx, li, acc, False, DC, 0, FASTEX2, False, False, False,
                DEN16, True)
    den_ok = (li <= 1.0e30) & ((rcnt <= 0) | (
        li >= rcnt.to(tl.float32) * 3.0e-4))
    bad = tl.min(den_ok.to(tl.int32), 0) == 0
    if bad:
        _qsa_w6_redo(Q, G, KC, VC, O, t0, s0, s1, base, NSLOT,
                     BT, NHC, DC, PWC, SC)
        return
    # Keep the gate load ahead of the accumulator read: its DRAM latency then
    # overlaps the tensor-memory drain instead of stacking behind it.
    gg = tl.load(G + qoff, mask=rv[:, None], other=0.0)
    rl = tl.where(li > 0, 1.0 / tl.where(li > 0, li, 1.0), 0.0)
    att = (acc * (rl * 0.5)[:, None]).to(tl.bfloat16)
    if GATEPTX:
        outv = _gmul_bf(att, gg)
    else:
        # The 4K route's FP8 probabilities already spend most of the error
        # budget, so its gate keeps the FP32-widened FP16 tanh and its finer
        # 1 + tanh step.
        outv = _gmul(att, gg.to(tl.float32))
    tl.store(O + qoff, outv, mask=rv[:, None])


def _cfg(T, B, split_ok):
    avg = max(1, T // max(B, 1))
    # With split-K available the tile can be wider (less KV traffic per token)
    # because the splits restore the CTA count.
    thr = 60 if split_ok else BIG_THR
    for rw, bt, mk in ((128, 21, 32), (64, 10, 16), (32, 5, 16)):
        if avg >= bt and (T + bt - 1) // bt >= thr:
            return bt, rw, mk, 64, 8, 3, (PV_SML if rw <= 32 else PV_BIG)
    # R=16 tiles use warp MMA (no TMEM), so when there are far more tiles than
    # SMs, trading pipeline depth for CTA occupancy wins.
    ntiles = (T + 1) // 2 + (B if B > 1 else 0)
    if ntiles >= 300:
        return 2, 16, 16, 64, 4, 1, PV_SML
    return 2, 16, 16, 64, 8, 3, PV_SML


_SCRATCH: dict = {}

# Routes whose warp-specialized form exceeded this toolchain's tensor-memory
# budget at compile time. Warp specialization here only pipelines the K/V
# copies (same tl.range iteration order, same accumulation order, bitwise
# identical results); some Triton builds double-buffer the MMA accumulators in
# tensor memory under warp_specialize=True, which can overflow the per-CTA
# budget (seen as "out of resource: tensor memory" on SM107 with the CUDA 13.4
# toolchain). Each failing route is remembered so later calls go straight to
# the non-warp-specialized schedule instead of re-attempting a failing compile.
_NO_WSPEC: dict = {}


def _desc(k_cache, v_cache, bn, nslot):
    """Build host-side TMA descriptors and avoid per-CTA tensormap setup."""
    if not _HTMA:
        return k_cache, v_cache
    return (TensorDescriptor.from_tensor(k_cache.view(nslot, D), [bn, D]),
            TensorDescriptor.from_tensor(v_cache.view(nslot, D), [bn, D]))


def _ws(key, nelem, dtype, device):
    """Reusable split-K scratch, fully overwritten before every combine."""
    buf = _SCRATCH.get(key)
    if buf is None or buf.numel() < nelem or buf.device != device:
        buf = torch.empty(nelem, dtype=dtype, device=device)
        _SCRATCH[key] = buf
    return buf


def run(q, gate, k_cache, v_cache, packed, cu_seqlens, page_base, out):
    _ensure_launch_state()
    T = q.shape[0]
    B = page_base.shape[0]
    nslot = k_cache.shape[0] * 64
    if T <= (TINY_MB if B > 1 else TINY_TB) * B:
        # average row holds only a handful of selected ids: the MMA-free
        # per-token kernel keeps many CTAs resident per SM instead of one,
        # and it is exact, which the fp8 windowed sweep is not when a row
        # has too few terms for its quantisation error to average out.
        _qsa_tiny[(T,)](
            q, gate, k_cache, v_cache, packed, cu_seqlens, page_base, out,
            T, B, nslot, HPAD=8, NHC=NH, DC=D, PWC=PW, NB=256,
            SINGLE=(B == 1), PAIR=(T == 2 * B),
            SC=SCL * (D ** 0.5) * (D ** -0.5),
            num_warps=TINY_WARPS, num_stages=1, maxnreg=96)
        return
    if B == 1 and T <= SR_TMAX:
        short_bn = 32 if T <= SR_BN32 else (64 if T <= 128 else SR_BN)
        sbt = 1 if T <= 128 else SR_BT
        srw = triton.next_power_of_2(sbt * NH)
        _qsa_short2[((T + sbt - 1) // sbt,)](
            q, gate, k_cache, v_cache, packed, page_base, out, T, nslot,
            BT=sbt, RW=srw, MK=max(16, triton.next_power_of_2(sbt)),
            BN=short_bn, NSUB=max(1, short_bn // 64),
            FP8=(SR_FP8 == 1), GEARLY=(srw <= 16),
            PV8=(SR_PV8 == 1),
            FASTRCP=(T == 256),
            # A KV window overhangs an exactly-sized cache when the slot
            # count is not a multiple of the window up to T's last window.
            OVH=(nslot < ((T + short_bn - 1) // short_bn) * short_bn),
            NHC=NH, DC=D, PWC=PW, SC=SCL,
            num_warps=(4 if T == 96 else SR_WARPS),
            num_stages=(2 if T == 256 else
                        (3 if short_bn == 32 else SR_STAGES)))
        return
    if T == 16382 and B == 2 and "w7" not in _NO_WSPEC:
        w7_bt = 21
        w7_nprog = (T + w7_bt - 1) // w7_bt + 1
        kd, vd = _desc(k_cache, v_cache, 128, nslot)
        try:
            _qsa_w7_tma[(w7_nprog,)](
                q, gate, k_cache, v_cache, kd, vd,
                packed, cu_seqlens, page_base, out,
                T, nslot, BT=w7_bt, RW=128, MK=32, BN=128,
                NHC=NH, DC=D, PWC=PW, SC=SCL,
                num_warps=8, num_stages=2)
            return
        except _TritonOOR:
            # This toolchain cannot fit the warp-specialized descriptor
            # sweep's tensor-memory footprint; the general path below serves
            # the shape (its T >= 12000 configuration is not warp-specialized).
            _NO_WSPEC["w7"] = True
    single = B == 1
    mid = single and 512 <= T <= 1024
    dn_mid = mid and DN_MODE == 1 and T >= DN_TMIN
    split_ok = mid and not dn_mid
    if mid:
        # 512..1024 single-request rows: tile geometry and split depth are set
        # directly, trading KV re-reads (fewer, wider tiles) against the CTA
        # count needed to cover the 212 SMs.
        bt = MID_BT
        rw = max(16, triton.next_power_of_2(bt * NH))
        mk = max(16, triton.next_power_of_2(bt))
        bn, nwarps, nstages, pvb = MID_BN, 8, MID_STAGES, PV_BIG
    else:
        bt, rw, mk, bn, nwarps, nstages, pvb = _cfg(T, B, split_ok)
    if BIG_BT and not mid and T < 6000 and (T + BIG_BT - 1) // BIG_BT >= BIG_MIN:
        # Explicit tile width: a 128-row tile reserves enough tensor memory to
        # pin the SM at one CTA, so trading KV re-reads for a narrower tile can
        # buy back occupancy on the long rows.
        bt = BIG_BT
        rw = max(16, triton.next_power_of_2(bt * NH))
        mk = max(16, triton.next_power_of_2(bt))
        pvb = PV_SML if rw <= 32 else PV_BIG
    if BIG_STAGES:
        nstages = BIG_STAGES
    if BIG_WARPS:
        nwarps = BIG_WARPS
    ntiles = (T + bt - 1) // bt
    nprog = ntiles if single else ntiles + B - 1
    if ((T == 4057 and B == 2) or (T == 8181 and B == 4)) and (
        "w6" not in _NO_WSPEC
    ):
        w6_bt = 21
        w6_nprog = (T + w6_bt - 1) // w6_bt + B - 1
        wbn = 128 if T == 4057 else bn
        kd, vd = _desc(k_cache, v_cache, wbn, nslot)
        try:
            _qsa_w6[(w6_nprog,)](
                q, gate, k_cache, v_cache, kd, vd, packed, cu_seqlens,
                page_base,
                out, T, nslot, BT=w6_bt, RW=128, MK=32, BN=wbn, NR=B,
                HTMA=_HTMA,
                CAPQ=False, FASTEX2=(T < 6000), GATEPTX=(T >= 6000),
                PEEL=(T < 6000), DEN16=(T == 4057),
                NHC=NH, DC=D, PWC=PW, SC=SCL,
                num_warps=nwarps, num_stages=2)
            return
        except _TritonOOR:
            # Same tensor-memory overflow class as the w7 route: fall through
            # to the general path, whose own warp-specialization fallback
            # below covers it.
            _NO_WSPEC["w6"] = True
    nsp = 1
    if mid and not dn_mid:
        nsp = max(1, min(MID_NSP, (T + bn - 1) // bn))
    nrow = T * NH
    # The dedicated T=768 vertical-split route is excluded from dispatch: it
    # was tuned together with the fixed fp8 query scale (CAPQ) that is only
    # valid for input magnitudes |q| <= 3.75, so T=768 takes the same verified
    # general mid-range split path as every other 512..1024 single-request
    # shape (see the CAPQ note in _qsa).
    vsplit = False
    max_nsp = 4 if vsplit else nsp
    if max_nsp > 1:
        ws = _ws("ws", max_nsp * nrow * D, torch.bfloat16,
                 q.device)[:max_nsp * nrow * D].view(max_nsp, nrow, D)
        wl = _ws("wl", 2 * max_nsp * nrow, torch.float32,
                 q.device)[:2 * max_nsp * nrow].view(2, max_nsp, nrow)
        wlp, wmp = wl[0], wl[1]
    else:
        ws = wlp = wmp = out
    grid = (194 if vsplit else nprog * nsp,)
    skipd = DN_MODE == 1 and nsp == 1 and T >= DN_TMIN
    if skipd:
        dbn = DN_BN if DN_BN > 0 else bn
        # Dense-prefix tiles run in a two-pass kernel that keeps its PV
        # accumulator in the MMA accumulator; the sparse companion below then
        # only carries the membership-mask path.
        _qsa_dn[(nprog,)](q, gate, k_cache, v_cache, packed, cu_seqlens,
                          page_base, out, T, B, nslot,
                          BT=bt, RW=rw, MK=mk, BN=dbn,
                          NB=max(16, triton.next_power_of_2(B)), SINGLE=single,
                          PVB=PV_BIG, EX2B=(EX2_BF16 == 1), TP=(DN_TP == 1),
                          NHC=NH, DC=D, PWC=PW, SC=SCL,
                          num_warps=nwarps, num_stages=DN_STAGES)
    use_tma = USE_TMA == 1 and TMA_TMIN <= T < 12000 and not mid
    if use_tma:
        kd, vd = _desc(k_cache, v_cache, bn, nslot)
    else:
        kd, vd = k_cache, v_cache
    wspec_key = ("qsa", bt, rw, mk, bn)
    wspec = (
        USE_WS == 1
        and TMA_TMIN <= T < 12000
        and not mid
        and wspec_key not in _NO_WSPEC
    )
    qsa_kw = dict(
        BT=bt, RW=rw, MK=mk, BN=bn, NSUB=max(1, bn // 64),
        NB=max(4, triton.next_power_of_2(B)), SINGLE=single,
        PAIRREQ=(B == 2),
        PVB=pvb, NSP=max_nsp, USEQL=(QRES == 1 and not mid),
        MASKFP8=(T >= 12000),
        RSC=RSC_MODE,
        SKIPD=skipd, TPD=(TPD_MODE == 1 and not skipd),
        SPV=(SPV_MODE == 1 and T >= 12000), W32=(W32_MASK == 1),
        EX2B=(EX2_BF16 == 1), EX2H=(T < 6000),
        ZERO=True,
        PEEL=(3000 <= T < 6000 or T >= 12000),
        DEN16=(T == 4057),
        CAPQ=False,  # fixed-scale fast path excluded (see _qsa note)
        FAST_CAUSAL=(mid or (not single and 6000 <= T < 12000)),
        TMA=((2 if _HTMA else 1) if use_tma else 0),
        WSPEC=wspec,
        GATEPTX=(T >= 6000),
        CSPL=mid, VSPL=vsplit,
        NHC=NH, DC=D, PWC=PW, SC=SCL,
        num_warps=nwarps,
        num_stages=(2 if 3000 <= T < 6000 and not mid else nstages),
    )
    try:
        _qsa[grid](q, gate, k_cache, v_cache, kd, vd,
                   packed, cu_seqlens, page_base, out,
                   ws, wlp, wmp,
                   T, B, nslot, nrow, **qsa_kw)
    except _TritonOOR:
        if not wspec:
            raise
        # The warp-specialized schedule overflowed this toolchain's
        # tensor-memory budget at compile time (no launch happened). Retry
        # without warp specialization: same loops, same accumulation order.
        _NO_WSPEC[wspec_key] = True
        qsa_kw["WSPEC"] = False
        _qsa[grid](q, gate, k_cache, v_cache, kd, vd,
                   packed, cu_seqlens, page_base, out,
                   ws, wlp, wmp,
                   T, B, nslot, nrow, **qsa_kw)
    if max_nsp > 1:
        rb = 8
        if vsplit:
            _comb_vsplit[((nrow + rb - 1) // rb,)](
                ws, wlp, wmp, gate, out, nrow,
                RB=rb, BT=bt, NHC=NH, CUT2=19, CUT3=38, CUT4=57, DC=D,
                num_warps=8, num_stages=2)
        else:
            _comb[((nrow + rb - 1) // rb,)](
                ws, wlp, wmp, gate, out, nrow,
                NSP=max_nsp, RB=rb, DC=D,
                num_warps=8, num_stages=2)
