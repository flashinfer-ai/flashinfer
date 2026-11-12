# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-tensor FP8 quantization pass over ``[T, H, D]``: ``dst = sat_fp8(src * scale)``.

The block's FP8 pipeline needs its SDPA operands (Q, K, V) and the out-projection's
A operand (``O_gated``) in E4M3 with a per-tensor scale.  This is that pass, as
a NAMED, MEASURED stage rather than a fusion (the fused-epilogue version is the
roadmap in ``api.py``): one streaming read of the bf16/f16 source, one write of
the fp8 destination.

* **Source may be a STRIDED column slice** of the fused projection slab (token
  stride ``N_qkvg``, head stride ``D``, element stride 1) — same dynamic
  token-stride fake as ``elementwise.py``, so no repack.  **Destination is
  COMPACT** ``[T, H, D]`` fp8 (the SDPA's zero-copy contract at 1 B/elem needs
  16-element-aligned seq/head strides, and compact ``H*D`` satisfies it).
* **``scale`` is a 1-element fp32 CUDA tensor read IN-KERNEL** (the
  ``make_array_view(t)[0]`` idiom of the FP8 SDPA kernels): a static per-tensor
  scale supplied by the caller, no host readback, so the launch stays
  CUDA-graph capturable (AGENTS.md Rule 3).  ``descale = 1/scale`` is what the
  consumer folds back.
* **Rounding**: ``cvt.rn.satfinite.e4m3x2.f32`` — round-to-nearest-even,
  saturating to +-448.  ``torch.Tensor.to(torch.float8_e4m3fn)`` saturates the
  same way on the torch pinned here, so the two agree BIT-FOR-BIT
  (``test_quantize.py`` asserts it; the reference clamps to +-448 first so it
  stays correct on a torch whose cast produces NaN on overflow).

Per-lane layout (the reason this is not literally ``elementwise.py``): a lane
moves **16 elements** — two ``ld.global.v4`` of bf16 (32 B, adjacent) in, one
``st.global.v4.b32`` of fp8 (16 B) out — because ``fp32_to_fp8_pack`` converts
exactly 16 fp32 into four packed words, and 16 B is the narrowest fully
vectorised fp8 store.  So ``LANES = D // 16`` lanes cover a row: at ``D = 256``
that is 16 lanes = HALF a warp per row, two rows per warp, the row's 512 B read
and 256 B write both contiguous across the half-warp.  Wider ``D`` adds chunks.

Traffic: ``2 B read + 1 B write`` per element (``moved_bytes``); bandwidth-bound.
**Needs sm_89+ for the fp8 ``cvt``** (Ada/Hopper/Blackwell/Rubin); an A100 has
no such instruction, so the tests skip there.

**The quantized BACKWARD's gradient quantization** rides on the same kernel plus
two small ones, all over fp32 SCALAR SLOTS (1-element fp32 CUDA views, 4-byte
aligned -- the kernels declare ``assumed_align=4`` for every slot pointer, so a
packed 4-byte slot stride is legal; ``check_scalar_slot`` is the one host spelling,
and ``_check_one_cuda_device`` pins every operand of a launch to ``src``'s device):

* **the amax pass** (``compile_amax`` / ``run_amax``): ``amax = max |fp32(src)|``
  over ``[T, H, D]`` in THIS kernel's lane layout (16 elements per lane, two
  ``ld.global.v4``), a ternary abs-max tree per lane, a warp butterfly, and ONE
  int32 ``atomicMax`` per warp of the fp32 bit pattern (``tile_dsl.pointwise.
  atomic_max_f32_bits``): non-negative fp32 patterns order as int32, so the slot
  ends at the fp32 max in any arrival order -- order-free, BITWISE
  ``x.float().abs().amax()``.  The slot is PRE-ZEROED by the caller on the same
  stream (0 is the identity of the max; a poisoned slot is the caller's bug).
* **the scale-from-amax arm** (``compile_quantize(scale_src="amax", margin_log2=)``):
  EVERY CTA derives ``scale = grad_scale_from_amax(amax, margin_log2)`` from the
  slot the amax pass filled -- ``2 ** (floor(log2(448 / amax)) - margin)``, ``1.0``
  at ``amax == 0``, as exact integer arithmetic on the fp32 bit pattern (never a
  division whose rounding could move the floor) -- and never reads the published
  copy (CTA 0 writes it concurrently).  So ``amax * scale <= 448`` for the amax the
  kernel read: no element saturates.
* **the publish** (``publish=True``, implied by ``scale_src="amax"`` or
  ``n_alpha > 0``): lane 0 of CTA 0 writes ``scale_out[0] = scale``, ``descale[0] =
  1 / scale`` (one ``div.rn.f32``: exact for a power of two) and ``alpha_outs[i][0] =
  descale * alpha_consts[i][0]`` for ``i < n_alpha`` (ONE fp32 RN multiply each:
  ``np.float32(descale) * np.float32(c)`` on the host gives the same bits) -- the
  GEMM epilogue scales of the backward.  BOTH arms publish through the same code,
  so a "delayed" recipe that feeds the caller's ``scale`` replays the "current"
  run bitwise, slots included.
* **the scalar-block init** (``compile_init_scalars`` / ``run_init_scalars``; ``descale_dp=False`` traces
  the body WITHOUT the reciprocal -- a pipeline with no dP scalar hands no ``scale_dp`` and gets none): ONE
  thread zeroes the ``n_slots`` fp32 slots (every amax slot must be zero before
  the first ``atomicMax`` of the first pass), then writes ``descale_dp_out[0] =
  1 / scale_dp[0]`` (``div.rn.f32``) -- so no second caller input can disagree
  with ``scale_dp`` -- and then the step's ``n_consts`` PLAN-TIME CONSTANTS
  (``consts``, Python floats) into ``slots[const_slot0 + i]``.  The constants are
  KERNEL ARGUMENTS (runtime fp32 values: one artifact per slot layout, never one
  per value), so they reach the device through the launch that consumes them --
  on the launch stream, on every execute, inside a CUDA-graph capture -- never as
  a device tensor filled at compile time, whose fill would sit on whatever stream
  was ambient THEN with nothing ordering it before an execute on another stream.
  Every store is an ordered inline-PTX store: the reciprocal and the constants
  may legally target slots just zeroed.
* **the amax PARTIALS pass** (``compile_amax_partials`` / ``run_amax_partials``): the
  same per-lane fold over a PERSISTENT grid of ``n_ctas = min(row groups, SMs x
  AMAX_CTAS_PER_SM)`` CTAs, each striding over its row groups and writing ONE fp32
  ``partials[cta] = max |x|`` of everything it read (warp butterfly, then the
  warps combined through a 4-word SMEM array and one barrier).  Every partial is
  WRITTEN unconditionally -- there is no slot to pre-zero, so the pass can share a
  launch with the kernel that would otherwise have to zero it.  The consumer is the
  quantize kernel's ``amax_src="partials"`` arm: every CTA reduces the ``n_partials``
  words in its prologue (128 threads, coalesced, one warp butterfly, one barrier),
  derives the same ``scale`` as the slot arm would from the same amax (``max`` is
  order-free: bitwise the slot pass), and lane 0 of CTA 0 PUBLISHES the reduced amax
  into ``amax_out`` -- so the scalar block's amax slot reads exactly what the slot
  pass would have left there.  ``scale_src="given"`` + ``amax_src="partials"`` is the
  "delayed" recipe's form: the caller's scale, the amax still published.

* **the persistent cast** (``compile_quantize(persistent=True)``): the grid is ``min(row groups,
  SMs x AMAX_CTAS_PER_SM)`` and every CTA strides over the row groups ``cta, cta + n_ctas, ...``
  (the partials pass's own loop), so a cast that reduces partials does that reduce ONCE per
  resident CTA instead of once per row group.  MEASURED on Rubin (cc 10.7, 204 SMs, SM clock
  locked at 2376 MHz), the dqkvg slab at S = 32K: a per-row-group reduce of 1632 partials costs
  +9 % on the cast and one of 6120 partials +35 %; persistent, the reduce is 1632 x 24 KiB of L2
  reads per launch and the cast is the plain cast's speed.  The default (``False``) is the
  one-row-group-per-block grid of every other launch.

The bodies of the quantize pass, of the partials pass and of the scalar-block init are ``@cute.jit`` functions
(``quantize_rows``, ``amax_partials_rows``, ``cta_max_of_partials``, ``init_scalars_body``)
taking a JOB-RELATIVE block index, so a fused launch (``fp8_bwd_fused.py``: several
small kernels behind one block-range dispatch) runs the very same code as the
standalone kernels here -- one body, two launch shapes, bitwise the same bytes.

The defaults (``scale_src="given"``, ``n_alpha=0``, ``margin_log2=0``,
``publish=False``) trace today's artifact: every new operand is ``None`` and every
new arm a ``const_expr`` folded out.  The amax pass and the init launch need no
fp8 instruction and run on every CUDA device; the fp8 ``cvt`` arms decline a
pre-sm_89 device BY NAME at ``compile_*`` (AGENTS.md Rule 7).

SMEM buffer table: NONE (no SMEM).  Barrier table: NONE (no mbarrier, no named
barrier, no TMA; the amax fold is one ``atomicMax`` per warp).  Nothing here can hang.
"""

import math
import numbers
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.datatypes import _convert_to_cutlass_data_type
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import compute_capability, current_device, multiprocessor_count
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import (
    abs_max_tree,
    atomic_max_f32_bits,
    div_rn_f32,
    f16x2_to_f32,
    fmax_f32,
    fp32_to_fp8_pack,
    opaque_f32_zero,
    warp_abs_max_f32_shfl,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import ld_global, ld_global_v4, st_global, st_global_v4
from cutlass.experimental import primitives as nvvm

from .qk_norm_rope import fake_rowmajor_dynamic_token_stride

ELEMS_PER_LANE = 16  # what fp32_to_fp8_pack converts in one call: 32 B of bf16 in, 16 B of fp8 out
SRC_BYTES_PER_LANE = ELEMS_PER_LANE * 2
DST_BYTES_PER_LANE = ELEMS_PER_LANE * 1
LOADS_PER_LANE = SRC_BYTES_PER_LANE // 16  # two ld.global.v4 (16 B each)
FP8_E4M3_MAX = 448.0
SCALE_SOURCES = ("given", "amax")  # where the quantize kernel takes its per-tensor scale: the caller's tensor, or derived from an amax slot
# Where the quantize kernel reads the amax it derives the scale from / publishes: "none" (a static-scale launch), "slot" (the
# pre-folded 1-element slot an amax pass or a producer filled), "partials" (the per-CTA maxima of a partials pass, reduced in
# every CTA's prologue and published to `amax_out`).  `None` at compile = resolved from scale_src ("slot" under "amax", else "none").
AMAX_SOURCES = ("none", "slot", "partials")
# Persistent-grid cap of the partials pass: SMs x AMAX_CTAS_PER_SM CTAs of 128 threads, each writing ONE partial.  8 is the
# residency the sibling 128-thread persistent kernel measured on Rubin (qk_norm_rope_bwd.CTAS_PER_SM); the consumer's prologue
# reads n_partials x 4 B per CTA, so the cap also bounds that read (6.5 KiB at 204 SMs).  Re-measure per part.
AMAX_CTAS_PER_SM = 8
# The "current" gradient-scale formula is spelled from frexp(448) = (0.875, 9) -- 448 = 0.875 * 2**9 -- on the host
# (grad_scale_from_amax) and in-kernel (_grad_scale_bits): never a literal 9 / 0.875 / 0x600000.
_E4M3_MAX_MANT, _E4M3_MAX_EXP = math.frexp(FP8_E4M3_MAX)
# amax = m * 2**e with m in [0.5, 1); m > 0.875  <=>  the fp32 mantissa field M > (2 * 0.875 - 1) * 2**23 (= 0x600000)
_E4M3_MAX_MANT_BITS = int(round((2.0 * _E4M3_MAX_MANT - 1.0) * (1 << 23)))
_FP32_FREXP_BIAS = 126  # the biased exponent field E of a NORMAL fp32 is frexp's e + 126
_SCALE_LOG2_MIN, _SCALE_LOG2_MAX = -126, 127  # the scale is clamped to a finite NORMAL power of two
_FP8_CVT_MIN_CC = (8, 9)  # cvt.rn.satfinite.e4m3x2.f32: Ada / Hopper / Blackwell / Rubin
MAX_ALPHA = 4  # alpha products one quantize launch can publish (the ABI reserves this many slot pairs; the block needs 2)
MAX_INIT_CONSTS = 16  # plan-time constants the scalar-init launch can store from its kernel arguments (the ABI reserves this many; the block needs 14)

DEFAULT_THREADS_PER_CTA = 128
DEFAULT_ROWS_PER_GROUP = 2
DEFAULT_CONST_HEAD_COUNT = True  # same knob, same reason as elementwise.py: a runtime H is a software divide per row

_FAKE_STREAM = None


def lanes_per_row(d: int) -> int:
    """Lanes cooperating on one head row, 16 elements each, capped at a warp."""
    return min(32, d // ELEMS_PER_LANE)


def chunks_per_lane(d: int) -> int:
    return d // (lanes_per_row(d) * ELEMS_PER_LANE)


def validate_shape(d: int, threads_per_cta: int) -> None:
    """Raise ``ValueError`` on any geometry this kernel cannot address (never an assert)."""
    if d % ELEMS_PER_LANE != 0:
        raise ValueError(f"d_head must be a multiple of {ELEMS_PER_LANE} (one fp32_to_fp8_pack per lane), got {d}")
    lanes = lanes_per_row(d)
    if 32 % lanes != 0:
        raise ValueError(f"d_head={d} gives {lanes} lanes/row, which must divide a warp")
    if d != lanes * ELEMS_PER_LANE * chunks_per_lane(d):
        raise ValueError(f"d_head={d} is not covered exactly by {lanes} lanes x {chunks_per_lane(d)} chunks x {ELEMS_PER_LANE} elements")
    if threads_per_cta % lanes != 0:
        raise ValueError(f"threads_per_cta={threads_per_cta} must be a multiple of the {lanes} lanes per row")


def grad_scale_from_amax(amax: float, margin_log2: int = 0) -> float:
    """``2 ** (floor(log2(FP8_E4M3_MAX / amax)) - margin_log2)`` as the kernel's ``scale_src="amax"`` arm computes it; ``1.0``
    at ``amax == 0``.  Exact integer arithmetic on the exponent (``math.frexp``), never an fp32 division whose rounding could
    move the floor at a boundary: with ``amax = m * 2**e`` (``m`` in ``[0.5, 1)``) and ``448 = 0.875 * 2**9``,
    ``floor(log2(448 / amax)) = (9 - e) - [m > 0.875]``.  The exponent is clamped to ``[-126, 127]`` -- a finite NORMAL
    power of two for ANY finite amax (a subnormal amax gives ``2**127``, a huge one ``2**-126``).  Properties: ``amax * scale
    <= 448`` for finite ``amax > 0`` (so nothing saturates), ``1 / scale`` is exact.  The amax is the kernel's own
    (``>= 0`` and finite): a negative or non-finite value is out of contract and raises."""
    if isinstance(amax, bool) or not isinstance(amax, (int, float)):
        raise ValueError(f"grad_scale_from_amax: amax must be a real number (the amax slot's value), got {type(amax).__name__}")
    amax = float(amax)
    if not math.isfinite(amax) or amax < 0.0:
        raise ValueError(f"grad_scale_from_amax: amax must be finite and >= 0 (an amax pass's max |x|), got {amax!r}")
    if isinstance(margin_log2, bool) or not isinstance(margin_log2, int):
        raise ValueError(f"grad_scale_from_amax: margin_log2 must be an int, got {margin_log2!r}")
    if amax == 0.0:
        return 1.0
    m, e = math.frexp(amax)
    k = (_E4M3_MAX_EXP - e) - (1 if m > _E4M3_MAX_MANT else 0) - margin_log2
    k = max(min(k, _SCALE_LOG2_MAX), _SCALE_LOG2_MIN)
    return math.ldexp(1.0, k)


@cute.jit
def _grad_scale_bits(amax: cutlass.Float32, margin_log2: cutlass.Constexpr[int]) -> cutlass.Float32:
    """:func:`grad_scale_from_amax` on the device, bit for bit, from the fp32 pattern of ``amax``: ``E = (bits >> 23) & 0xFF``
    is frexp's ``e + 126`` for a normal value and ``M = bits & 0x7FFFFF`` its mantissa field (``m > 0.875 <=> M > 0x600000``),
    so ``k = (9 + 126 - margin) - E - [M > 0x600000]``, clamped to ``[-126, 127]``, ``0`` for a zero amax; the scale is the
    pattern ``(k + 127) << 23``.  A subnormal amax (``E == 0``) lands on the ``2**127`` clamp like the host's.  Every
    operation is integer: no rounding anywhere."""
    bits = amax.bitcast(cutlass.Int32) & cutlass.Int32(0x7FFFFFFF)
    exp_field = (bits >> 23) & cutlass.Int32(0xFF)
    mant_field = bits & cutlass.Int32(0x7FFFFF)
    k = cutlass.Int32(_E4M3_MAX_EXP + _FP32_FREXP_BIAS - margin_log2) - exp_field
    k = k - (cutlass.Int32(1) if mant_field > cutlass.Int32(_E4M3_MAX_MANT_BITS) else cutlass.Int32(0))
    k = cutlass.Int32(_SCALE_LOG2_MAX) if k > cutlass.Int32(_SCALE_LOG2_MAX) else k
    k = cutlass.Int32(_SCALE_LOG2_MIN) if k < cutlass.Int32(_SCALE_LOG2_MIN) else k
    k = cutlass.Int32(0) if bits == cutlass.Int32(0) else k
    return ((k + cutlass.Int32(127)) << 23).bitcast(cutlass.Float32)


def require_fp8_cvt(who: str) -> None:
    """The fp8 ``cvt.rn.satfinite`` needs sm_89+: decline a pre-Ada device BY NAME at compile time (AGENTS.md Rule 7), never
    through the ptxas error the trace would otherwise die with."""
    cc = compute_capability(current_device())
    if tuple(cc) < _FP8_CVT_MIN_CC:
        raise NotImplementedError(
            f"{who}: the fp8 cvt.rn.satfinite instruction needs sm_{_FP8_CVT_MIN_CC[0]}{_FP8_CVT_MIN_CC[1]}+ (Ada / Hopper / Blackwell / Rubin); "
            f"the current device is sm_{cc[0]}{cc[1]}"
        )


def _slot_value(m: cute.Tensor) -> cutlass.Float32:
    """The fp32 a 1-element slot holds, read in-kernel (no host readback)."""
    return cutlass.Float32(cutlass.make_array_view(m)[0])


def check_scalar_slot(name: str, ten, *, numel: int = 1) -> None:
    """The scalar-slot contract of these kernels' fp32 side operands (an amax target, a published scale / descale / alpha,
    the scalar block): a ``numel``-element fp32 CUDA tensor (contiguous when ``numel > 1``) at a 4-byte-aligned address --
    the kernels declare ``assumed_align=4`` for every slot pointer, so a 4-byte slot stride is legal.  Raises ``ValueError``."""
    if not isinstance(ten, torch.Tensor):
        raise ValueError(f"{name} must be a {numel}-element fp32 CUDA tensor, got {type(ten).__name__}")
    if ten.dtype != torch.float32 or ten.numel() != numel or not ten.is_cuda:
        raise ValueError(
            f"{name} must be a {numel}-element fp32 CUDA tensor (read / written in-kernel; no host readback), got {ten.dtype} x {ten.numel()} on {ten.device}"
        )
    if numel > 1 and not ten.is_contiguous():
        raise ValueError(f"{name} must be contiguous (one fp32 [{numel}] view of the scalar block), got strides {tuple(ten.stride())}")
    if ten.data_ptr() % 4:
        raise ValueError(f"{name} must sit at a 4-byte-aligned address (the kernels' assumed_align for a slot), got {ten.data_ptr():#x}")


def _check_one_cuda_device(anchor_name: str, anchor, operands) -> None:
    """Every bound operand on ONE CUDA device, the anchor's -- the device half of every block launcher's host contract
    (``run_quantize`` here; ``sigmoid_gate_bwd`` and the qk-norm / RoPE backward import it).

    The kernel reads each operand through a device pointer.  A CPU operand (a ``seq_lens`` built as ``torch.tensor(lens,
    dtype=torch.int32)`` with no ``device=``, a scalar slot) passes the dtype / rank / length checks and would be
    dereferenced as a HOST address -- an illegal-address fault at the next synchronize, sticky for the process -- and an
    operand on ANOTHER GPU, which an ``is_cuda`` check cannot tell from a local one, is the same wild access on a foreign
    device (or silent peer traffic).  Named here -- the operand, the launch device and where it actually is -- before the
    launch.  ``operands`` is ``(name, tensor)`` pairs; a ``None`` (an unbound optional) is skipped."""
    if not anchor.is_cuda:
        raise ValueError(f"{anchor_name} must be a CUDA tensor (the kernel reads every operand through a device pointer), got device {anchor.device}")
    for name, ten in operands:
        if ten is not None and ten.device != anchor.device:
            raise ValueError(f"{name} must be on {anchor.device} with {anchor_name}, got {ten.device}")


# ---------------------------------------------------------------------------
# The quantize BODY -- one function, two launch shapes (the standalone kernel below, the fused launches of fp8_bwd_fused.py)
# ---------------------------------------------------------------------------


@cute.jit
def quantize_rows(
    mSrc: cute.Tensor,  # [T, H, D] bf16/f16, own token/head strides
    mDst: cute.Tensor,  # [T, H, D] fp8 e4m3, own token/head strides (compact in the block)
    scale: cutlass.Float32,  # the per-tensor scale, ALREADY in a register (read from a slot or derived from an amax by the caller)
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    cta: cutlass.Int32,  # JOB-RELATIVE block index: the standalone kernel passes blockIdx.x, a fused launch its arm's offset
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
) -> None:
    """``dst = sat_e4m3(src * scale)`` over the ``rows_per_cta`` rows of block ``cta`` (module docstring: 16 elements per
    lane, every load in flight first, tail rows clamp their LOADS to the last valid row and skip the store).  ``src`` and
    ``dst`` never alias (dtypes differ)."""
    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(chunks_per_lane(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    _h = cutlass.Int32(h_ct) if cutlass.const_expr(const_head_count) else h
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    row0 = (cta * cutlass.Int32(groups_per_cta) + grp) * cutlass.Int32(rows_per_group)
    src_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(SRC_BYTES_PER_LANE)
    dst_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(DST_BYTES_PER_LANE)

    # PASS 1: every load in flight first (pure memory-level parallelism -- no
    # reduction sits between a load and the next issue).  Tail rows clamp their
    # LOADS to the last valid row and skip the store.
    rows = []
    dsts = []
    srcs = []
    for r in cutlass.range_constexpr(rows_per_group):
        row = row0 + cutlass.Int32(r)
        row_r = row if row < n_rows else n_rows - cutlass.Int32(1)
        token = row_r // _h
        head = row_r % _h
        src_addr = mSrc.iterator.toint() + (
            token.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[1])
        ) * cutlass.Int64(2)
        dst_addr = mDst.iterator.toint() + (token.to(cutlass.Int64) * cutlass.Int64(mDst.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mDst.stride[1]))
        row_src = []
        for c in cutlass.range_constexpr(chunks):
            base = src_addr + cutlass.Int64((c * lanes) * SRC_BYTES_PER_LANE) + src_lane_off
            vals = []
            for j in cutlass.range_constexpr(LOADS_PER_LANE):
                for w in ld_global_v4(base + cutlass.Int64(j * 16), cutlass.Int32):
                    lo, hi = f16x2_to_f32(w, dtype=mSrc.element_type)
                    vals.append(lo)
                    vals.append(hi)
            row_src.append(vals)
        rows.append(row)
        dsts.append(dst_addr)
        srcs.append(row_src)

    # PASS 2: scale in fp32, convert (RNE, saturating) and pack 16 -> 4 words, one v4 store.
    for r in cutlass.range_constexpr(rows_per_group):
        if rows[r] < n_rows:
            for c in cutlass.range_constexpr(chunks):
                scaled = []
                for i in cutlass.range_constexpr(ELEMS_PER_LANE):
                    scaled.append(srcs[r][c][i] * scale)
                packed = fp32_to_fp8_pack(scaled, dtype=cutlass.Float8E4M3FN)
                st_global_v4(
                    dsts[r] + cutlass.Int64((c * lanes) * DST_BYTES_PER_LANE) + dst_lane_off, [packed[0], packed[1], packed[2], packed[3]], cutlass.Int32
                )


@cute.jit
def cta_max_of_partials(mPartials: cute.Tensor, n_partials: cutlass.Int32, sRed, threads_per_cta: cutlass.Constexpr[int]) -> cutlass.Float32:
    """The CTA-uniform ``max`` of the ``n_partials`` fp32 words of ``mPartials`` (non-negative: the partials pass's per-CTA
    maxima) -- EVERY thread of the CTA takes part: coalesced strided loads, a warp butterfly, the warps combined through
    ``sRed`` (fp32 ``[warps]`` SMEM) behind ONE ``barrier_cta_sync``, then every thread folds the warp maxima in the same order.
    Returns the same bits in every thread.  ``max`` is order-free, so the result is bitwise the slot pass's ``atomicMax`` fold.
    The strided loop takes FOUR words per thread per step (independent loads in flight; a single-word loop was ~50 dependent
    round trips at 6.5k partials -- most of a block's reduce time), then a one-word tail."""
    warps = cutlass.const_expr(threads_per_cta // 32)
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    warp_lane = tidx % cutlass.Int32(32)
    warp_id = tidx // cutlass.Int32(32)
    acc = _strided_max_of_words(mPartials.iterator.toint(), n_partials, tidx, threads_per_cta, opaque_f32_zero())
    wmax = warp_abs_max_f32_shfl(acc)  # |x| is the identity on the non-negative partials
    if warp_lane == cutlass.Int32(0):
        sRed.subview(warp_id).store(wmax)
    nvvm.barrier_cta_sync()
    m = sRed.subview(0).load()
    for w in cutlass.range_constexpr(1, warps):
        m = fmax_f32(m, sRed.subview(w).load())
    return m


@cute.jit
def _strided_max_of_words(
    base: cutlass.Int64, n: cutlass.Int32, tidx: cutlass.Int32, threads_per_cta: cutlass.Constexpr[int], acc: cutlass.Float32
) -> cutlass.Float32:
    """``max(acc, words[0 .. n))`` over the ``n`` fp32 words at ``base`` (16-byte aligned: the partials contract) by the whole CTA:
    thread ``tidx`` takes 16-byte VECTORS ``tidx, tidx + T, ...`` of the first ``4 * (n // 4)`` words, FOUR independent vector
    loads per step (16 words in flight per thread -- the reduce of ~6.5k partials is 3-4 round trips instead of ~50), then a
    one-vector tail, then the ``n % 4`` trailing words one per thread.  ``max`` is order-free, so the grouping changes no bit."""
    step = cutlass.Int32(threads_per_cta)
    n4 = n // cutlass.Int32(4)
    i = tidx
    while i + cutlass.Int32(3 * threads_per_cta) < n4:
        v0 = ld_global_v4(base + i.to(cutlass.Int64) * cutlass.Int64(16), cutlass.Float32)
        v1 = ld_global_v4(base + (i + step).to(cutlass.Int64) * cutlass.Int64(16), cutlass.Float32)
        v2 = ld_global_v4(base + (i + cutlass.Int32(2 * threads_per_cta)).to(cutlass.Int64) * cutlass.Int64(16), cutlass.Float32)
        v3 = ld_global_v4(base + (i + cutlass.Int32(3 * threads_per_cta)).to(cutlass.Int64) * cutlass.Int64(16), cutlass.Float32)
        m01 = fmax_f32(fmax_f32(fmax_f32(v0[0], v0[1]), fmax_f32(v0[2], v0[3])), fmax_f32(fmax_f32(v1[0], v1[1]), fmax_f32(v1[2], v1[3])))
        m23 = fmax_f32(fmax_f32(fmax_f32(v2[0], v2[1]), fmax_f32(v2[2], v2[3])), fmax_f32(fmax_f32(v3[0], v3[1]), fmax_f32(v3[2], v3[3])))
        acc = fmax_f32(acc, fmax_f32(m01, m23))
        i = i + cutlass.Int32(4 * threads_per_cta)
    while i < n4:
        v = ld_global_v4(base + i.to(cutlass.Int64) * cutlass.Int64(16), cutlass.Float32)
        acc = fmax_f32(acc, fmax_f32(fmax_f32(v[0], v[1]), fmax_f32(v[2], v[3])))
        i = i + step
    # the n % 4 trailing words, one per thread: the load clamps to the last word (always valid: n >= 1) and a SELECT keeps the fold
    j = n4 * cutlass.Int32(4) + tidx
    j_r = j if j < n else n - cutlass.Int32(1)
    w = ld_global(base + j_r.to(cutlass.Int64) * cutlass.Int64(4), cutlass.Float32)
    acc = fmax_f32(acc, w) if j < n else acc
    return acc


@cute.jit
def cta_max_of_partials_pair(
    mA: cute.Tensor, n_a: cutlass.Int32, mB: cute.Tensor, n_b: cutlass.Int32, sRed, threads_per_cta: cutlass.Constexpr[int]
) -> cutlass.Float32:
    """:func:`cta_max_of_partials` over TWO partials arrays (two producers' per-CTA maxima -- the gate backward's dG
    partials and the norm backward's band partials of the quantized block backward) behind ONE barrier: every thread
    strides over ``mA`` then ``mB``, then the one butterfly + SMEM combine.  ``max`` is order-free, so the result is bitwise
    ``max(cta_max_of_partials(mA), cta_max_of_partials(mB))`` and the slot pass's fold over both producers' words."""
    warps = cutlass.const_expr(threads_per_cta // 32)
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    warp_lane = tidx % cutlass.Int32(32)
    warp_id = tidx // cutlass.Int32(32)
    acc = _strided_max_of_words(mA.iterator.toint(), n_a, tidx, threads_per_cta, opaque_f32_zero())
    acc = _strided_max_of_words(mB.iterator.toint(), n_b, tidx, threads_per_cta, acc)
    wmax = warp_abs_max_f32_shfl(acc)
    if warp_lane == cutlass.Int32(0):
        sRed.subview(warp_id).store(wmax)
    nvvm.barrier_cta_sync()
    m = sRed.subview(0).load()
    for w in cutlass.range_constexpr(1, warps):
        m = fmax_f32(m, sRed.subview(w).load())
    return m


@cute.jit
def amax_partials_rows(
    mSrc: cute.Tensor,  # [T, H, D] bf16/f16, own token/head strides
    mPartials: cute.Tensor,  # [>= n_ctas] fp32 OUT: partials[cta] = max |x| over the rows this CTA read (written unconditionally)
    n_rows: cutlass.Int32,
    n_groups: cutlass.Int32,  # ceil(n_rows / rows_per_cta): the row groups the persistent CTAs stride over
    h: cutlass.Int32,
    cta: cutlass.Int32,  # JOB-RELATIVE block index in [0, n_ctas)
    n_ctas: cutlass.Int32,
    sRed,  # fp32 [warps] SMEM: the warp maxima
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
) -> None:
    """The amax pass as PER-CTA PARTIALS: CTA ``cta`` folds row groups ``cta, cta + n_ctas, ...`` (the quantize layout, a
    ternary abs-max tree per lane, tail rows SELECTED out), then the warp butterfly, the warps through ``sRed`` behind one
    barrier, and thread 0 stores the CTA's maximum -- no atomic, no pre-zeroed slot.  Every lane of every warp reaches the
    butterfly (the row loop bound is CTA-uniform; nothing above diverges)."""
    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(chunks_per_lane(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    warps = cutlass.const_expr(threads_per_cta // 32)
    _h = cutlass.Int32(h_ct) if cutlass.const_expr(const_head_count) else h
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    warp_lane = tidx % cutlass.Int32(32)
    warp_id = tidx // cutlass.Int32(32)
    src_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(SRC_BYTES_PER_LANE)
    acc = opaque_f32_zero()
    n_iters = (n_groups - cta + n_ctas - cutlass.Int32(1)) // n_ctas
    for it in cutlass.range(n_iters):
        g_idx = cta + it * n_ctas
        row0 = (g_idx * cutlass.Int32(groups_per_cta) + grp) * cutlass.Int32(rows_per_group)
        lives = []
        srcs = []
        for r in cutlass.range_constexpr(rows_per_group):
            row = row0 + cutlass.Int32(r)
            row_r = row if row < n_rows else n_rows - cutlass.Int32(1)
            token = row_r // _h
            head = row_r % _h
            src_addr = mSrc.iterator.toint() + (
                token.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[1])
            ) * cutlass.Int64(2)
            vals = []
            for c in cutlass.range_constexpr(chunks):
                base = src_addr + cutlass.Int64((c * lanes) * SRC_BYTES_PER_LANE) + src_lane_off
                for j in cutlass.range_constexpr(LOADS_PER_LANE):
                    for w in ld_global_v4(base + cutlass.Int64(j * 16), cutlass.Int32):
                        lo, hi = f16x2_to_f32(w, dtype=mSrc.element_type)
                        vals.append(lo)
                        vals.append(hi)
            lives.append(row < n_rows)
            srcs.append(vals)
        for r in cutlass.range_constexpr(rows_per_group):
            m = abs_max_tree(srcs[r])
            acc = fmax_f32(acc, m) if lives[r] else acc
    wmax = warp_abs_max_f32_shfl(acc)
    if warp_lane == cutlass.Int32(0):
        sRed.subview(warp_id).store(wmax)
    nvvm.barrier_cta_sync()
    if tidx == cutlass.Int32(0):
        m = sRed.subview(0).load()
        for w in cutlass.range_constexpr(1, warps):
            m = fmax_f32(m, sRed.subview(w).load())
        st_global(mPartials.iterator.toint() + cta.to(cutlass.Int64) * cutlass.Int64(4), m, cutlass.Float32)


@cute.jit
def publish_scale(
    scale: cutlass.Float32,
    amax_val: cutlass.Float32,
    mScaleOut: cute.Tensor,
    mDescale: cute.Tensor,
    mAmaxOut: Optional[cute.Tensor],
    mAlphaC0: Optional[cute.Tensor],
    mAlphaC1: Optional[cute.Tensor],
    mAlphaC2: Optional[cute.Tensor],
    mAlphaC3: Optional[cute.Tensor],
    mAlphaO0: Optional[cute.Tensor],
    mAlphaO1: Optional[cute.Tensor],
    mAlphaO2: Optional[cute.Tensor],
    mAlphaO3: Optional[cute.Tensor],
    n_alpha: cutlass.Constexpr[int],
) -> None:
    """ONE thread (the caller's lane 0 of its first block) publishes the scale, ``descale = 1 / scale`` (``div.rn.f32``: exact
    for a power of two, correctly rounded otherwise -- never a reciprocal-multiply), ``alpha_i = descale * c_i`` (one RN
    multiply each) and, when ``mAmaxOut`` is bound, the amax the scale was derived from (the partials arm: the slot pass's
    slot value, re-created)."""
    descale = div_rn_f32(opaque_f32_zero() + cutlass.Float32(1.0), scale)
    mScaleOut.iterator[0] = scale
    mDescale.iterator[0] = descale
    if cutlass.const_expr(mAmaxOut is not None):
        mAmaxOut.iterator[0] = amax_val
    # The alpha stores are spelled HERE, in a DSL-transformed body, never in a plain Python helper: the DSL transforms only the
    # decorated function's own source, so a helper's ops have no claim to the branch they are called from (python/cudnn/AGENTS.md,
    # "CuTeDSL kernel bodies"); range_constexpr's trace-time i indexes the ABI's slot tuples, so exactly the n_alpha bound pairs are touched.
    alpha_consts = (mAlphaC0, mAlphaC1, mAlphaC2, mAlphaC3)
    alpha_outs = (mAlphaO0, mAlphaO1, mAlphaO2, mAlphaO3)
    for i in cutlass.range_constexpr(n_alpha):
        alpha_outs[i].iterator[0] = descale * _slot_value(alpha_consts[i])


@cute.kernel
def frost_quantize_fp8(
    mSrc: cute.Tensor,  # [T, H, D] bf16/f16, own token/head strides
    mDst: cute.Tensor,  # [T, H, D] fp8 e4m3, own token/head strides (compact in the block)
    mScale: Optional[cute.Tensor],  # [1] fp32: the caller's scale (scale_src="given"); None under "amax"
    mAmax: Optional[cute.Tensor],  # [1] fp32: the slot the amax pass / the producer filled (amax_src="slot"); None otherwise
    mScaleOut: Optional[cute.Tensor],  # [1] fp32 OUT (publish): the scale this launch applied
    mDescale: Optional[cute.Tensor],  # [1] fp32 OUT (publish): 1 / scale
    mAlphaC0: Optional[cute.Tensor],  # [1] fp32 IN: alpha constant i (bound for i < n_alpha)
    mAlphaC1: Optional[cute.Tensor],
    mAlphaC2: Optional[cute.Tensor],
    mAlphaC3: Optional[cute.Tensor],
    mAlphaO0: Optional[cute.Tensor],  # [1] fp32 OUT: alpha_i = descale * alpha constant i
    mAlphaO1: Optional[cute.Tensor],
    mAlphaO2: Optional[cute.Tensor],
    mAlphaO3: Optional[cute.Tensor],
    mPartials: Optional[cute.Tensor],  # [n_partials] fp32: the partials pass's per-CTA maxima (amax_src="partials"); None otherwise
    mAmaxOut: Optional[cute.Tensor],  # [1] fp32 OUT: the reduced amax (amax_src="partials"); None otherwise
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    n_partials: cutlass.Int32,
    n_ctas: cutlass.Int32,  # the grid (persistent arm: every CTA strides over the row groups cta, cta + n_ctas, ...)
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    n_alpha: cutlass.Constexpr[int],
    margin_log2: cutlass.Constexpr[int],
    persistent: cutlass.Constexpr[bool],
) -> None:
    """``dst = sat_fp8(src * scale)`` (:func:`quantize_rows`); ``src`` and ``dst`` never alias (dtypes differ).

    ``scale`` is the caller's (``mScale``) or derived in EVERY CTA from the amax -- the pre-folded slot (``mAmax``) or the
    partials pass's maxima reduced in this CTA's prologue (``mPartials``; module docstring); with a publish (``mScaleOut``
    bound) lane 0 of CTA 0 also writes the scale, its reciprocal, the ``n_alpha`` alpha products and (partials arm) the amax.
    Under ``persistent`` the CTA loops over its row groups (``cta + it * n_ctas``); otherwise it owns exactly one.
    """
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    from_slot = cutlass.const_expr(mAmax is not None)
    from_partials = cutlass.const_expr(mPartials is not None)
    publish = cutlass.const_expr(mScaleOut is not None)
    derive = cutlass.const_expr(mScale is None)
    sRed = cutlass.Array(cutlass.Float32, threads_per_cta // 32, alignment=16, space=cutlass.AddressSpace.smem) if cutlass.const_expr(from_partials) else None

    # The amax this launch derives from / publishes: the partials reduced by the whole CTA (one barrier), or the slot, once per
    # thread from device memory (no host readback) -- never the published copy CTA 0 writes concurrently.
    amax_val = (
        cta_max_of_partials(mPartials, n_partials, sRed, threads_per_cta)
        if cutlass.const_expr(from_partials)
        else (_slot_value(mAmax) if cutlass.const_expr(from_slot) else opaque_f32_zero())
    )
    # The per-tensor scale: the "current" recipe's power of two from that amax, or the caller's.
    scale = _grad_scale_bits(amax_val, margin_log2) if cutlass.const_expr(derive) else _slot_value(mScale)

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    cta = cutlass.Int32(cute.arch.block_idx()[0])
    if cutlass.const_expr(publish):
        # Lane 0 of CTA 0 publishes.  The one BLOCK the scalar reads below depends on nothing this launch writes (the host
        # refuses an aliasing slot).
        if (cta == cutlass.Int32(0)) & (tidx == cutlass.Int32(0)):
            publish_scale(
                scale, amax_val, mScaleOut, mDescale, mAmaxOut, mAlphaC0, mAlphaC1, mAlphaC2, mAlphaC3, mAlphaO0, mAlphaO1, mAlphaO2, mAlphaO3, n_alpha
            )
    if cutlass.const_expr(persistent):
        rows_per_cta = cutlass.const_expr((threads_per_cta // lanes_per_row(d)) * rows_per_group)
        n_groups = (n_rows + cutlass.Int32(rows_per_cta - 1)) // cutlass.Int32(rows_per_cta)
        n_iters = (n_groups - cta + n_ctas - cutlass.Int32(1)) // n_ctas
        for it in cutlass.range(n_iters):
            quantize_rows(mSrc, mDst, scale, n_rows, h, cta + it * n_ctas, h_ct, const_head_count, d, threads_per_cta, rows_per_group)
    else:
        quantize_rows(mSrc, mDst, scale, n_rows, h, cta, h_ct, const_head_count, d, threads_per_cta, rows_per_group)

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()


@cute.jit
def quantize_fp8_launch(
    src: cute.Tensor,
    dst: cute.Tensor,
    scale: Optional[cute.Tensor],
    amax: Optional[cute.Tensor],
    scale_out: Optional[cute.Tensor],
    descale: Optional[cute.Tensor],
    alpha_c0: Optional[cute.Tensor],
    alpha_c1: Optional[cute.Tensor],
    alpha_c2: Optional[cute.Tensor],
    alpha_c3: Optional[cute.Tensor],
    alpha_o0: Optional[cute.Tensor],
    alpha_o1: Optional[cute.Tensor],
    alpha_o2: Optional[cute.Tensor],
    alpha_o3: Optional[cute.Tensor],
    partials: Optional[cute.Tensor],
    amax_out: Optional[cute.Tensor],
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    n_partials: cutlass.Int32,
    n_blocks: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    n_alpha: cutlass.Constexpr[int],
    margin_log2: cutlass.Constexpr[int],
    persistent: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_quantize_fp8(
        src,
        dst,
        scale,
        amax,
        scale_out,
        descale,
        alpha_c0,
        alpha_c1,
        alpha_c2,
        alpha_c3,
        alpha_o0,
        alpha_o1,
        alpha_o2,
        alpha_o3,
        partials,
        amax_out,
        n_rows,
        h,
        n_partials,
        n_blocks,
        h_ct,
        const_head_count,
        d,
        threads_per_cta,
        rows_per_group,
        use_pdl,
        n_alpha,
        margin_log2,
        persistent,
    ).launch(grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl)


compiled_cache = {}


def _fake_slot():
    """The trace-time shape of every fp32 scalar slot: ONE element, 4-byte aligned (a packed 4-byte slot stride is legal)."""
    return cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), stride_order=(0,), assumed_align=4)


def _fake_partials():
    """The trace-time shape of a partials array: fp32 ``[n]`` contiguous with a SYMBOLIC length (one artifact serves every
    grid), on a 16-byte-aligned base (a 256-B-aligned workspace region, or a fresh tensor)."""
    return cute.runtime.make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(),), stride_order=(0,), assumed_align=16)


def _slot_view(ten: Optional[torch.Tensor]):
    """The ``[1]`` view a slot is bound as (a 1-element reshape never copies -- Rule 1), or ``None``."""
    return None if ten is None else ten.reshape(1)


def resolve_amax_src(scale_src: str, amax_src: Optional[str]) -> str:
    """``None`` = the pre-partials construction: a ``scale_src="amax"`` artifact reads the slot, a ``"given"`` one reads no amax."""
    if amax_src is None:
        return "slot" if scale_src == "amax" else "none"
    return amax_src


class QuantizeRecipe(NamedTuple):
    """Build-time facts of one quantize launch; the token count rides in as a runtime ``Int32``."""

    compiled: object
    h: int
    d: int
    rows_per_cta: int
    dtype_in: object
    # Appended (defaults = today's artifact): where the scale comes from, how many alpha products the launch publishes,
    # the "current" recipe's power-of-two headroom, and whether the publish arm (scale_out / descale) is traced.  All four
    # are part of the artifact's ABI, so ``run_quantize`` checks its operands against them BOTH ways (Rule 1).
    scale_src: str = "given"
    n_alpha: int = 0
    margin_log2: int = 0
    publish: bool = False
    # Appended: where the amax comes from -- "slot" / "partials" / "none"; None = resolved from scale_src (resolve_amax_src),
    # so a recipe built the old way reads exactly as before.
    amax_src: Optional[str] = None
    # Appended (defaults = today's artifact): the persistent grid (module docstring) and its cap, SMs x AMAX_CTAS_PER_SM on
    # the compiling device (0 on a non-persistent recipe: the grid is the row-group count).
    persistent: bool = False
    n_ctas_cap: int = 0


def compile_quantize(
    *,
    dtype_in,
    h: int,
    d: int,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP,
    const_head_count: bool = DEFAULT_CONST_HEAD_COUNT,
    use_pdl: bool = False,
    scale_src: str = "given",
    n_alpha: int = 0,
    margin_log2: int = 0,
    publish: bool = False,
    amax_src: Optional[str] = None,
    persistent: bool = False,
) -> QuantizeRecipe:
    """Build from SHAPES ALONE -- no allocation, no launch.  E4M3 output only for now.

    Appended: ``scale_src="given"`` reads the caller's 1-element scale (today's artifact); ``"amax"`` derives ``scale =
    grad_scale_from_amax(amax, margin_log2)`` in EVERY CTA from the amax.  ``amax_src`` names where that amax comes from:
    ``"slot"`` (a pre-filled 1-element slot -- the default under ``"amax"``), ``"partials"`` (the partials pass's per-CTA
    maxima, reduced in every CTA's prologue and PUBLISHED to ``amax_out``; legal under both scale sources: with ``"given"``
    it is the "delayed" recipe's form, the caller's scale and the amax still published), ``"none"`` (the default under
    ``"given"``).  ``n_alpha`` (``<= MAX_ALPHA``) is how many ``descale * const`` products lane 0 of CTA 0 publishes (the GEMM
    epilogue scales).  ``publish`` traces the publish arm -- ``scale_out`` and ``descale`` become REQUIRED at execute -- and is
    IMPLIED by ``scale_src="amax"``, ``amax_src="partials"`` or ``n_alpha > 0``; pass it explicitly for a "given" launch that
    must publish its scale and reciprocal without alpha products (the backward's dO quantize under the delayed recipe).
    Every knob is in the cache key; the defaults share the forward's artifact.  The fp8 ``cvt`` needs sm_89+: a pre-Ada
    device is declined by name (Rule 7)."""
    global _FAKE_STREAM
    validate_shape(d, threads_per_cta)
    if dtype_in not in (torch.bfloat16, torch.float16):
        raise ValueError(f"quantize serves bf16/f16 sources only, got {dtype_in}")
    if scale_src not in SCALE_SOURCES:
        raise ValueError(f"scale_src must be one of {SCALE_SOURCES}, got {scale_src!r}")
    amax_src = resolve_amax_src(scale_src, amax_src)
    if amax_src not in AMAX_SOURCES:
        raise ValueError(f"amax_src must be one of {AMAX_SOURCES} (or None = resolved from scale_src), got {amax_src!r}")
    if scale_src == "amax" and amax_src == "none":
        raise ValueError("scale_src='amax' derives the scale from an amax: amax_src must be 'slot' or 'partials', not 'none'")
    if isinstance(n_alpha, bool) or not isinstance(n_alpha, int) or n_alpha < 0:
        raise ValueError(f"n_alpha must be a non-negative int (the alpha products the launch publishes), got {n_alpha!r}")
    if n_alpha > MAX_ALPHA:
        raise ValueError(f"n_alpha={n_alpha} exceeds the {MAX_ALPHA} alpha slot pairs the quantize launch's ABI reserves")
    if isinstance(margin_log2, bool) or not isinstance(margin_log2, int):
        raise ValueError(f"margin_log2 must be an int (the 'current' recipe's power-of-two headroom), got {margin_log2!r}")
    if not isinstance(publish, bool):
        raise ValueError(f"publish must be a bool (whether the launch publishes scale_out / descale), got {publish!r}")
    if amax_src == "partials" and threads_per_cta % 32 != 0:
        raise ValueError(f"amax_src='partials' needs threads_per_cta % 32 == 0 (the prologue reduce ends in a full-warp butterfly), got {threads_per_cta}")
    publish = publish or scale_src == "amax" or n_alpha > 0 or amax_src == "partials"
    if not isinstance(persistent, bool):
        raise ValueError(f"persistent must be a bool (whether the launch strides a capped grid over the row groups), got {persistent!r}")
    require_fp8_cvt("compile_quantize")
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    device = current_device()
    key = (
        str(dtype_in),
        h,
        d,
        int(threads_per_cta),
        int(rows_per_group),
        bool(const_head_count),
        bool(use_pdl),
        device,
        scale_src,
        int(n_alpha),
        int(margin_log2),
        publish,
        amax_src,
        persistent,
    )
    if key not in compiled_cache:
        tok = cute.sym_int()
        # Both operands carry a symbolic token stride: the source is a column
        # slice of the projection slab, the destination is compact -- one
        # artifact serves both (and a compact source, too).
        src = fake_rowmajor_dynamic_token_stride(dtype_in, tok, h, d)
        dst = cute.runtime.make_fake_tensor(dtype=cutlass.Float8E4M3FN, shape=(tok, h, d), stride=(cute.sym_int(), d, 1), assumed_align=16)
        # The scalar slots: each traced ONLY when its arm is on (None otherwise -- the ABI keeps the parameter, the kernel
        # folds the arm out), every one a 1-element fp32 at 4-byte alignment.
        scale = _fake_slot() if scale_src == "given" else None
        amax = _fake_slot() if amax_src == "slot" else None
        partials = _fake_partials() if amax_src == "partials" else None
        amax_out = _fake_slot() if amax_src == "partials" else None
        scale_out = _fake_slot() if publish else None
        descale = _fake_slot() if publish else None
        alpha_c = [_fake_slot() if i < n_alpha else None for i in range(MAX_ALPHA)]
        alpha_o = [_fake_slot() if i < n_alpha else None for i in range(MAX_ALPHA)]
        compiled_cache[key] = cute.compile(
            quantize_fp8_launch,
            src,
            dst,
            scale,
            amax,
            scale_out,
            descale,
            *alpha_c,
            *alpha_o,
            partials,
            amax_out,
            cutlass.Int32(0),  # n_rows     ) runtime; the zeros pin the TYPE only
            cutlass.Int32(h),  # h          )
            cutlass.Int32(0),  # n_partials )
            cutlass.Int32(1),  # n_blocks   )
            int(h),
            bool(const_head_count),
            d,
            int(threads_per_cta),
            int(rows_per_group),
            bool(use_pdl),
            int(n_alpha),
            int(margin_log2),
            persistent,
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return QuantizeRecipe(
        compiled=compiled_cache[key],
        h=h,
        d=d,
        rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group,
        dtype_in=dtype_in,
        scale_src=scale_src,
        n_alpha=int(n_alpha),
        margin_log2=int(margin_log2),
        publish=publish,
        amax_src=amax_src,
        persistent=persistent,
        n_ctas_cap=int(multiprocessor_count(device)) * AMAX_CTAS_PER_SM if persistent else 0,
    )


def quantize_grid(r: QuantizeRecipe, t: int) -> int:
    """The blocks one quantize launch over ``t`` tokens runs: the row groups ``ceil(t * h / rows_per_cta)``, capped at
    ``n_ctas_cap`` on a persistent recipe (at least 1)."""
    groups = (int(t) * r.h + r.rows_per_cta - 1) // r.rows_per_cta
    if getattr(r, "persistent", False):
        return max(1, min(groups, int(r.n_ctas_cap)))
    return max(1, groups)


def check_partials(name: str, ten, n_partials) -> int:
    """The partials-array contract (an amax partials pass's output, the partials arm's input): a contiguous fp32 1-D CUDA
    tensor on a 16-byte-aligned base holding at least ``n_partials >= 1`` words; returns ``int(n_partials)``.  Raises ``ValueError``."""
    if not isinstance(ten, torch.Tensor) or ten.dtype != torch.float32 or ten.dim() != 1 or not ten.is_cuda or not ten.is_contiguous():
        got = f"{ten.dtype} of shape {tuple(ten.shape)} on {ten.device}" if isinstance(ten, torch.Tensor) else type(ten).__name__
        raise ValueError(f"{name} must be a contiguous fp32 [n] CUDA tensor (the amax partials), got {got}")
    if ten.data_ptr() % 16:
        raise ValueError(f"{name} must sit on a 16-byte-aligned base (the kernels' assumed_align for the partials), got {ten.data_ptr():#x}")
    if isinstance(n_partials, bool) or not isinstance(n_partials, int) or n_partials < 1 or n_partials > int(ten.numel()):
        raise ValueError(f"n_partials must be an int in [1, {int(ten.numel())}] (the partials the pass wrote into {name}), got {n_partials!r}")
    return int(n_partials)


def run_quantize(
    r: QuantizeRecipe,
    src: torch.Tensor,
    dst: torch.Tensor,
    scale: Optional[torch.Tensor],
    *,
    stream,
    amax: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
    descale: Optional[torch.Tensor] = None,
    alpha_consts: tuple = (),
    alpha_outs: tuple = (),
    partials: Optional[torch.Tensor] = None,
    n_partials: Optional[int] = None,
    amax_out: Optional[torch.Tensor] = None,
) -> None:
    """Launch.  ``src`` ``[T, H, D]`` bf16/f16 (strided ok), ``dst`` ``[T, H, D]`` ``float8_e4m3fn``, ``scale`` 1-element fp32 CUDA.

    Cheap host checks only; every one of them guards a wild write or a silent
    wrong answer at the tvm-ffi boundary, which reports neither.

    Appended (the quantized backward's arm; every slot operand a 1-element fp32 CUDA view at a 4-byte-aligned address,
    checked BOTH ways against the recipe -- Rule 1): under ``scale_src="given"`` ``scale`` is READ; under ``"amax"`` ``scale``
    must be ``None`` and every CTA derives the scale itself, never reading ``scale_out``, which CTA 0 writes concurrently.
    The amax comes from the recipe's ``amax_src``: ``"slot"`` -- ``amax`` (the slot the amax pass filled) is REQUIRED;
    ``"partials"`` -- ``partials`` (the partials pass's contiguous fp32 array), ``n_partials`` (how many of its words the
    pass wrote) and ``amax_out`` (the slot the reduced amax is published to) are REQUIRED and ``amax`` must be ``None``;
    ``"none"`` -- none of them may be bound.  With the publish arm (``r.publish``: implied by ``"amax"``, ``"partials"`` or
    ``n_alpha > 0``, explicit otherwise) ``scale_out`` AND ``descale`` are REQUIRED and lane 0 of CTA 0 writes ``scale_out[0]
    = scale``, ``descale[0] = 1 / scale`` and ``alpha_outs[i][0] = descale * alpha_consts[i][0]`` for ``i < n_alpha`` (a length
    other than ``n_alpha`` is a typed error: the ABI is fixed per artifact); without it every one of them must be ``None``.
    No published slot may alias a slot the launch READS (``amax``, ``scale``, an alpha constant, a word of ``partials``) --
    the other CTAs read it concurrently -- nor another published slot.  Every
    operand -- ``src``, ``dst``, ``scale``, ``partials`` and each slot -- sits on ONE CUDA device, ``src``'s
    (``_check_one_cuda_device``): the kernel reads and writes them all through raw device pointers, and ``is_cuda`` alone
    cannot see a slot on another GPU.
    """
    if src.dtype != r.dtype_in:
        raise ValueError(f"src is {src.dtype} but this artifact was compiled for {r.dtype_in}")
    if dst.dtype != torch.float8_e4m3fn:
        raise ValueError(f"dst must be torch.float8_e4m3fn, got {dst.dtype}")
    for name, ten in (("src", src), ("dst", dst)):
        if ten.ndim != 3 or int(ten.shape[1]) != r.h or int(ten.shape[2]) != r.d:
            raise ValueError(f"{name} must be [T, H={r.h}, D={r.d}], got {tuple(ten.shape)}")
        if ten.stride(2) != 1 or ten.stride(1) != r.d:
            raise ValueError(f"{name} must have head stride D={r.d} and element stride 1 (a column slice of the slab or compact), got strides {ten.stride()}")
    if int(src.shape[0]) != int(dst.shape[0]):
        raise ValueError(f"src has T={int(src.shape[0])} rows, dst T={int(dst.shape[0])}")
    if int(src.shape[0]) < 1:
        raise ValueError("src has no rows (T == 0): nothing to quantize, and a zero-size grid is a launch error, not a no-op")
    if (src.stride(0) * 2) % 16 or (dst.stride(0)) % 16:
        raise ValueError(f"token strides must keep every row 16-byte aligned: src {src.stride(0)} elems (bf16), dst {dst.stride(0)} elems (fp8)")
    amax_src = resolve_amax_src(r.scale_src, r.amax_src)
    if r.scale_src == "given":
        if scale is None:
            raise ValueError("this artifact reads the caller's scale (scale_src='given'); scale must be bound at execute (Rule 1: no silent fallback)")
        if scale.dtype != torch.float32 or scale.numel() != 1 or not scale.is_cuda:
            raise ValueError("scale must be a 1-element fp32 CUDA tensor (read in-kernel; no host readback)")
        if amax is not None and amax_src != "slot":
            raise ValueError("this artifact reads the caller's scale (scale_src='given'); passing amax would silently ignore it (Rule 1)")
    else:
        if scale is not None:
            raise ValueError("this artifact derives its scale from amax (scale_src='amax'); passing a caller scale would silently ignore it (Rule 1)")
        if amax_src == "slot" and amax is None:
            raise ValueError(
                "this artifact derives its scale from amax (scale_src='amax'); amax (the slot the amax pass filled) must be bound at execute (Rule 1)"
            )
        if scale_out is None:
            raise ValueError(
                "this artifact derives its scale from amax (scale_src='amax'); scale_out (where the derived scale is published) must be bound at execute (Rule 1)"
            )
    if amax_src == "slot":
        if amax is None:
            raise ValueError("this artifact reads the amax SLOT (amax_src='slot'); amax must be bound at execute (Rule 1: no silent fallback)")
        if partials is not None or n_partials is not None or amax_out is not None:
            raise ValueError(
                "this artifact reads the amax SLOT (amax_src='slot'); passing partials / n_partials / amax_out would silently ignore them (Rule 1)"
            )
    elif amax_src == "partials":
        if partials is None or n_partials is None or amax_out is None:
            raise ValueError(
                "this artifact reduces the amax PARTIALS (amax_src='partials'): partials (the partials pass's fp32 array), n_partials (the words it "
                "wrote) and amax_out (the slot the reduced amax is published to) must all be bound at execute (Rule 1: no silent fallback)"
            )
        if amax is not None:
            raise ValueError("this artifact reduces the amax PARTIALS (amax_src='partials'); passing an amax slot would silently ignore it (Rule 1)")
        n_partials = check_partials("partials", partials, n_partials)
    else:
        if partials is not None or n_partials is not None or amax_out is not None:
            raise ValueError("this artifact reads no amax (amax_src='none'); passing partials / n_partials / amax_out would silently ignore them (Rule 1)")
    if len(alpha_consts) != r.n_alpha or len(alpha_outs) != r.n_alpha:
        raise ValueError(
            f"this artifact publishes n_alpha={r.n_alpha} alpha products; got {len(alpha_consts)} alpha_consts and {len(alpha_outs)} alpha_outs "
            "(the ABI is fixed per artifact)"
        )
    if r.publish:
        if scale_out is None or descale is None:
            raise ValueError(
                "this artifact was compiled WITH the publish arm (publish=True, implied by scale_src='amax', amax_src='partials' or n_alpha > 0): "
                "scale_out AND descale (1-element fp32 CUDA views) must be bound at execute (Rule 1: no silent fallback)"
            )
    elif scale_out is not None or descale is not None:
        raise ValueError(
            "this artifact was compiled WITHOUT the publish arm (publish=False); passing scale_out / descale would silently ignore them (Rule 1) -- "
            "compile with publish=True (it is implied by scale_src='amax' or n_alpha > 0, explicit for a 'given' launch that publishes)"
        )
    slots = [("amax", amax), ("scale_out", scale_out), ("descale", descale), ("amax_out", amax_out)]
    slots += [(f"alpha_consts[{i}]", c) for i, c in enumerate(alpha_consts)] + [(f"alpha_outs[{i}]", a) for i, a in enumerate(alpha_outs)]
    for name, ten in slots:
        if ten is not None:
            check_scalar_slot(name, ten)
    # ONE CUDA device, src's, for every operand -- dst, the scale source, the partials and each slot: the kernel reads and writes
    # them all through raw device pointers, and the is_cuda half of the slot contract cannot tell a slot on ANOTHER GPU from a local one.
    _check_one_cuda_device("src", src, [("dst", dst), ("scale", scale), ("partials", partials)] + slots)
    # A published slot must not be a slot this launch READS (every CTA reads amax / scale / the alpha constants / the partials
    # while CTA 0 writes), nor another published slot (two writers of one word).
    reads = [("scale", scale), ("amax", amax)] + [(f"alpha_consts[{i}]", c) for i, c in enumerate(alpha_consts)]
    writes = [("scale_out", scale_out), ("descale", descale), ("amax_out", amax_out)] + [(f"alpha_outs[{i}]", a) for i, a in enumerate(alpha_outs)]
    writes = [(n, w) for n, w in writes if w is not None]
    for wn, w in writes:
        for rn, rd in reads:
            if rd is not None and w.data_ptr() == rd.data_ptr():
                raise ValueError(f"{wn} aliases {rn}: a published slot cannot be a slot the launch reads (the other CTAs read it while CTA 0 writes)")
        if partials is not None and partials.data_ptr() <= w.data_ptr() < partials.data_ptr() + 4 * int(n_partials):
            raise ValueError(f"{wn} lies inside the partials this launch reads: a published slot cannot be a word the other CTAs read")
    for i, (wn, w) in enumerate(writes):
        for vn, v in writes[i + 1 :]:
            if w.data_ptr() == v.data_ptr():
                raise ValueError(f"{wn} and {vn} are the same slot: two published values cannot share one word")
    t = int(src.shape[0])
    n_rows = t * r.h
    n_blocks = quantize_grid(r, t)
    alpha_c = list(alpha_consts) + [None] * (MAX_ALPHA - r.n_alpha)
    alpha_o = list(alpha_outs) + [None] * (MAX_ALPHA - r.n_alpha)
    r.compiled(
        src,
        dst,
        _slot_view(scale),
        _slot_view(amax),
        _slot_view(scale_out),
        _slot_view(descale),
        *[_slot_view(c) for c in alpha_c],
        *[_slot_view(a) for a in alpha_o],
        partials,
        _slot_view(amax_out),
        cutlass.Int32(n_rows),
        cutlass.Int32(r.h),
        cutlass.Int32(int(n_partials) if n_partials is not None else 0),
        cutlass.Int32(n_blocks),
        cuda.CUstream(int(stream)),
    )


def moved_bytes(t: int, h: int, d: int, *, src_elem_bytes: int = 2) -> int:
    """HBM traffic of one launch: the bf16/f16 read plus the 1-byte fp8 write.  The 4-byte scale is noise."""
    return t * h * d * (src_elem_bytes + 1)


def amax_moved_bytes(t: int, h: int, d: int, *, src_elem_bytes: int = 2) -> int:
    """HBM traffic of one amax pass: the bf16/f16 read only (one 4-byte atomic per warp is noise)."""
    return t * h * d * src_elem_bytes


# ---------------------------------------------------------------------------
# The amax pass: amax = max |fp32(src)| over [T, H, D], into a pre-zeroed fp32 slot
# ---------------------------------------------------------------------------


@cute.kernel
def frost_amax_abs(
    mSrc: cute.Tensor,  # [T, H, D] bf16/f16, own token/head strides (compact or a slab column slice)
    mAmax: cute.Tensor,  # [1] fp32, PRE-ZEROED: one int32 atomicMax of the warp's max |x| bit pattern per warp
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
) -> None:
    """The quantize kernel's lane layout (16 elements per lane, two ``ld.global.v4``), a ternary abs-max tree per lane
    (``abs_max_tree``), the warp butterfly (``warp_abs_max_f32_shfl``) and ONE ``atomicMax`` per warp.  Tail rows clamp
    their loads to the last valid row (a redundant read, never a wild one) and are SELECTED out of the fold."""
    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(chunks_per_lane(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    _h = cutlass.Int32(h_ct) if cutlass.const_expr(const_head_count) else h

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    warp_lane = tidx % cutlass.Int32(32)
    row0 = (cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(groups_per_cta) + grp) * cutlass.Int32(rows_per_group)
    src_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(SRC_BYTES_PER_LANE)

    # PASS 1: every load in flight first (the quantize kernel's structure).
    lives = []
    srcs = []
    for r in cutlass.range_constexpr(rows_per_group):
        row = row0 + cutlass.Int32(r)
        row_r = row if row < n_rows else n_rows - cutlass.Int32(1)
        token = row_r // _h
        head = row_r % _h
        src_addr = mSrc.iterator.toint() + (
            token.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[1])
        ) * cutlass.Int64(2)
        vals = []
        for c in cutlass.range_constexpr(chunks):
            base = src_addr + cutlass.Int64((c * lanes) * SRC_BYTES_PER_LANE) + src_lane_off
            for j in cutlass.range_constexpr(LOADS_PER_LANE):
                for w in ld_global_v4(base + cutlass.Int64(j * 16), cutlass.Int32):
                    lo, hi = f16x2_to_f32(w, dtype=mSrc.element_type)
                    vals.append(lo)
                    vals.append(hi)
        lives.append(row < n_rows)
        srcs.append(vals)

    # PASS 2: the per-lane fold (a SELECT drops a clamped tail row -- it duplicates the last valid row, so the max would not
    # change, but the fold is over the rows that exist), then the warp max and one atomic from lane 0.  Every lane of the
    # warp reaches the butterfly: nothing above diverges.
    acc = opaque_f32_zero()
    for r in cutlass.range_constexpr(rows_per_group):
        m = abs_max_tree(srcs[r])
        acc = fmax_f32(acc, m) if lives[r] else acc
    wmax = warp_abs_max_f32_shfl(acc)
    if warp_lane == cutlass.Int32(0):
        atomic_max_f32_bits(mAmax, wmax)


@cute.jit
def amax_abs_launch(
    src: cute.Tensor,
    amax: cute.Tensor,
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    n_blocks: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    frost_amax_abs(src, amax, n_rows, h, h_ct, const_head_count, d, threads_per_cta, rows_per_group).launch(
        grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream
    )


amax_compiled_cache = {}


class AmaxRecipe(NamedTuple):
    """Build-time facts of one amax launch (``run_amax``): the quantize kernel's own lane layout over ``[T, H, D]``."""

    compiled: object
    h: int
    d: int
    rows_per_cta: int
    dtype_in: object


def compile_amax(
    *,
    dtype_in,
    h: int,
    d: int,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP,
    const_head_count: bool = DEFAULT_CONST_HEAD_COUNT,
) -> AmaxRecipe:
    """The amax pass ``amax = max |fp32(src)|`` over a ``[T, H, D]`` bf16 / f16 source (compact or a slab slice): one int32
    ``atomicMax`` per warp of non-negative fp32 bit patterns, which order as int32 -- order-free, bitwise the fp32 max.
    Build from SHAPES ALONE -- no allocation, no launch.  Needs no fp8 instruction: runs on every CUDA device."""
    global _FAKE_STREAM
    validate_shape(d, threads_per_cta)
    if dtype_in not in (torch.bfloat16, torch.float16):
        raise ValueError(f"amax serves bf16/f16 sources only, got {dtype_in}")
    if threads_per_cta % 32 != 0:
        raise ValueError(f"threads_per_cta={threads_per_cta} must be a multiple of 32: the fold ends in a full-warp butterfly (shfl.sync over 32 lanes)")
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    key = (str(dtype_in), h, d, int(threads_per_cta), int(rows_per_group), bool(const_head_count), current_device())
    if key not in amax_compiled_cache:
        tok = cute.sym_int()
        src = fake_rowmajor_dynamic_token_stride(dtype_in, tok, h, d)
        amax_compiled_cache[key] = cute.compile(
            amax_abs_launch,
            src,
            _fake_slot(),
            cutlass.Int32(0),  # n_rows   ) runtime; the zeros pin the TYPE only
            cutlass.Int32(h),  # h        )
            cutlass.Int32(0),  # n_blocks )
            int(h),
            bool(const_head_count),
            d,
            int(threads_per_cta),
            int(rows_per_group),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return AmaxRecipe(compiled=amax_compiled_cache[key], h=h, d=d, rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group, dtype_in=dtype_in)


def run_amax(r: AmaxRecipe, src: torch.Tensor, amax: torch.Tensor, *, stream) -> None:
    """Launch.  ``src`` ``[T, H, D]`` bf16 / f16 under the quantize kernel's layout rules (head stride ``D``, element stride 1,
    a 16-byte-aligned token stride); ``amax`` a 1-element fp32 CUDA view PRE-ZEROED by the caller on the same stream (an
    ``atomicMax`` target: a poisoned slot is the caller's bug -- the launch can only RAISE it).  Host checks only."""
    if src.dtype != r.dtype_in:
        raise ValueError(f"src is {src.dtype} but this artifact was compiled for {r.dtype_in}")
    if src.ndim != 3 or int(src.shape[1]) != r.h or int(src.shape[2]) != r.d:
        raise ValueError(f"src must be [T, H={r.h}, D={r.d}], got {tuple(src.shape)}")
    if src.stride(2) != 1 or src.stride(1) != r.d:
        raise ValueError(f"src must have head stride D={r.d} and element stride 1 (a column slice of the slab or compact), got strides {src.stride()}")
    if (src.stride(0) * 2) % 16:
        raise ValueError(f"the token stride must keep every row 16-byte aligned: src {src.stride(0)} elems (bf16/f16)")
    if int(src.shape[0]) < 1:
        raise ValueError("src has no rows (T == 0): the amax of nothing is the slot's zero, and a zero-size grid is a launch error, not a no-op")
    check_scalar_slot("amax", amax)
    if not src.is_cuda or amax.device != src.device:
        raise ValueError(f"src and amax must be on one CUDA device (the kernel reads both through device pointers), got {src.device} and {amax.device}")
    t = int(src.shape[0])
    n_rows = t * r.h
    n_blocks = (n_rows + r.rows_per_cta - 1) // r.rows_per_cta
    r.compiled(src, _slot_view(amax), cutlass.Int32(n_rows), cutlass.Int32(r.h), cutlass.Int32(n_blocks), cuda.CUstream(int(stream)))


# ---------------------------------------------------------------------------
# The amax PARTIALS pass: partials[cta] = max |x| over the row groups of persistent CTA `cta` (no slot, no atomic)
# ---------------------------------------------------------------------------


@cute.kernel
def frost_amax_partials(
    mSrc: cute.Tensor,  # [T, H, D] bf16/f16, own token/head strides
    mPartials: cute.Tensor,  # [>= n_ctas] fp32 OUT
    n_rows: cutlass.Int32,
    n_groups: cutlass.Int32,
    h: cutlass.Int32,
    n_ctas: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
) -> None:
    """The standalone launch shape of :func:`amax_partials_rows` (grid = ``n_ctas`` persistent CTAs)."""
    sRed = cutlass.Array(cutlass.Float32, threads_per_cta // 32, alignment=16, space=cutlass.AddressSpace.smem)
    cta = cutlass.Int32(cute.arch.block_idx()[0])
    amax_partials_rows(mSrc, mPartials, n_rows, n_groups, h, cta, n_ctas, sRed, h_ct, const_head_count, d, threads_per_cta, rows_per_group)


@cute.jit
def amax_partials_launch(
    src: cute.Tensor,
    partials: cute.Tensor,
    n_rows: cutlass.Int32,
    n_groups: cutlass.Int32,
    h: cutlass.Int32,
    n_ctas: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    frost_amax_partials(src, partials, n_rows, n_groups, h, n_ctas, h_ct, const_head_count, d, threads_per_cta, rows_per_group).launch(
        grid=(n_ctas, 1, 1), block=(threads_per_cta, 1, 1), stream=stream
    )


amax_partials_compiled_cache = {}


class AmaxPartialsRecipe(NamedTuple):
    """Build-time facts of one amax PARTIALS launch (``run_amax_partials``): the quantize layout over ``[T, H, D]`` on a
    persistent grid of at most ``n_ctas_cap`` CTAs (``SMs x AMAX_CTAS_PER_SM`` on the compiling device), one partial each."""

    compiled: object
    h: int
    d: int
    rows_per_cta: int
    dtype_in: object
    n_ctas_cap: int


def compile_amax_partials(
    *,
    dtype_in,
    h: int,
    d: int,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP,
    const_head_count: bool = DEFAULT_CONST_HEAD_COUNT,
) -> AmaxPartialsRecipe:
    """The amax pass as per-CTA PARTIALS over a ``[T, H, D]`` bf16 / f16 source (module docstring): ``n_ctas = n_partials_for(r, t)``
    persistent CTAs each write ``partials[cta] = max |x|`` of the row groups they strode over -- written unconditionally, so
    no slot has to be zeroed first; the consumer (``compile_quantize(amax_src="partials")``) reduces them.  Build from SHAPES
    ALONE -- no allocation, no launch.  Needs no fp8 instruction: runs on every CUDA device."""
    global _FAKE_STREAM
    validate_shape(d, threads_per_cta)
    if dtype_in not in (torch.bfloat16, torch.float16):
        raise ValueError(f"amax serves bf16/f16 sources only, got {dtype_in}")
    if threads_per_cta % 32 != 0:
        raise ValueError(f"threads_per_cta={threads_per_cta} must be a multiple of 32: the fold ends in a full-warp butterfly (shfl.sync over 32 lanes)")
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    device = current_device()
    key = (str(dtype_in), h, d, int(threads_per_cta), int(rows_per_group), bool(const_head_count), device)
    if key not in amax_partials_compiled_cache:
        tok = cute.sym_int()
        src = fake_rowmajor_dynamic_token_stride(dtype_in, tok, h, d)
        amax_partials_compiled_cache[key] = cute.compile(
            amax_partials_launch,
            src,
            _fake_partials(),
            cutlass.Int32(0),  # n_rows   ) runtime; the zeros pin the TYPE only
            cutlass.Int32(0),  # n_groups )
            cutlass.Int32(h),  # h        )
            cutlass.Int32(1),  # n_ctas   )
            int(h),
            bool(const_head_count),
            d,
            int(threads_per_cta),
            int(rows_per_group),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return AmaxPartialsRecipe(
        compiled=amax_partials_compiled_cache[key],
        h=h,
        d=d,
        rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group,
        dtype_in=dtype_in,
        n_ctas_cap=int(multiprocessor_count(device)) * AMAX_CTAS_PER_SM,
    )


def n_partials_for(r: AmaxPartialsRecipe, t: int) -> int:
    """The partials one launch over ``t`` tokens writes: ``min(ceil(t * h / rows_per_cta), n_ctas_cap)``, at least 1 -- the
    ``n_partials`` the consumer reduces and the minimum length of the partials array."""
    groups = (int(t) * r.h + r.rows_per_cta - 1) // r.rows_per_cta
    return max(1, min(groups, r.n_ctas_cap))


def run_amax_partials(r: AmaxPartialsRecipe, src: torch.Tensor, partials: torch.Tensor, *, stream) -> int:
    """Launch.  ``src`` ``[T, H, D]`` bf16 / f16 under the quantize layout rules; ``partials`` a contiguous fp32 ``[>= n_partials_for(r, T)]``
    CUDA tensor (16-byte-aligned base) -- every one of its first ``n_partials`` words is OVERWRITTEN (no pre-zero).  Returns
    ``n_partials``.  Host checks only."""
    if src.dtype != r.dtype_in:
        raise ValueError(f"src is {src.dtype} but this artifact was compiled for {r.dtype_in}")
    if src.ndim != 3 or int(src.shape[1]) != r.h or int(src.shape[2]) != r.d:
        raise ValueError(f"src must be [T, H={r.h}, D={r.d}], got {tuple(src.shape)}")
    if src.stride(2) != 1 or src.stride(1) != r.d:
        raise ValueError(f"src must have head stride D={r.d} and element stride 1 (a column slice of the slab or compact), got strides {src.stride()}")
    if (src.stride(0) * 2) % 16:
        raise ValueError(f"the token stride must keep every row 16-byte aligned: src {src.stride(0)} elems (bf16/f16)")
    if int(src.shape[0]) < 1:
        raise ValueError("src has no rows (T == 0): the amax of nothing has no partial to write, and a zero-size grid is a launch error, not a no-op")
    t = int(src.shape[0])
    n_rows = t * r.h
    n_groups = (n_rows + r.rows_per_cta - 1) // r.rows_per_cta
    n_ctas = n_partials_for(r, t)
    check_partials("partials", partials, n_ctas)
    if not src.is_cuda or partials.device != src.device:
        raise ValueError(f"src and partials must be on one CUDA device (the kernel reads both through device pointers), got {src.device} and {partials.device}")
    r.compiled(src, partials, cutlass.Int32(n_rows), cutlass.Int32(n_groups), cutlass.Int32(r.h), cutlass.Int32(n_ctas), cuda.CUstream(int(stream)))
    return n_ctas


# ---------------------------------------------------------------------------
# The scalar-block init: slots[0:n_slots] = 0, then descale_dp_out[0] = 1 / scale_dp[0]
# ---------------------------------------------------------------------------


@cute.jit
def init_scalars_body(
    mSlots: cute.Tensor,  # [n_slots] fp32 contiguous: the scalar block
    mScaleDp: cute.Tensor,  # [1] fp32: the caller's scale_dP (outside the block)
    mDescaleDpOut: cute.Tensor,  # [1] fp32 OUT: 1 / scale_dP (may be one of the slots)
    const0: cutlass.Float32,  # the plan-time constants, KERNEL ARGUMENTS (runtime values): const_i -> slots[const_slot0 + i] for i < n_consts
    const1: cutlass.Float32,
    const2: cutlass.Float32,
    const3: cutlass.Float32,
    const4: cutlass.Float32,
    const5: cutlass.Float32,
    const6: cutlass.Float32,
    const7: cutlass.Float32,
    const8: cutlass.Float32,
    const9: cutlass.Float32,
    const10: cutlass.Float32,
    const11: cutlass.Float32,
    const12: cutlass.Float32,
    const13: cutlass.Float32,
    const14: cutlass.Float32,
    const15: cutlass.Float32,
    n_slots: cutlass.Constexpr[int],
    const_slot0: cutlass.Constexpr[int],
    n_consts: cutlass.Constexpr[int],
    descale_dp: cutlass.Constexpr[
        bool
    ] = True,  # APPENDED (default = today's body): False traces NO reciprocal -- mScaleDp / mDescaleDpOut are bound and never touched
) -> None:
    """ONE thread, every store an ordered inline-PTX ``st.global`` -- a dedicated reset launch ahead of the first ``atomicMax``,
    the ``_fp8_setup`` idiom of the fp8 SDPA host: the zeroing first, then the reciprocal, then the constants -- which may legally
    land on slots just zeroed, because asm stores keep program order whatever the compiler assumes about the pointers.  The
    constants arrive as runtime fp32 kernel arguments (``const0 .. const15``; ``range_constexpr``'s trace-time ``i`` picks the
    ``n_consts`` bound ones), so ONE artifact serves every value and the values are written by THIS launch, on its stream -- the
    whole point: nothing the execute reads is filled anywhere else.  Shared by the
    standalone kernel (``_init_scalars``, one block) and the fused prologue launch (``fp8_bwd_fused.py``, its block 0): ONE
    thread -- thread 0 of the calling block -- does every store.  ``descale_dp=False`` (appended) folds the reciprocal out: a
    pipeline whose attention backward has no dP scalar (the block-scaled MXFP8 row) zeroes the slots and stores its constants,
    and the two scalar tensors are bound, never read, never written."""
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    if tidx == cutlass.Int32(0):
        zero = opaque_f32_zero()
        base = mSlots.iterator.toint()
        for i in cutlass.range_constexpr(n_slots):
            # a 4-B pitch = the fp32 element size: the block's slot stride (api_bwd.QUANT_SCALAR_STRIDE, pinned equal to it by
            # api_bwd._plan_bwd_workspace) -- the readers' slot views sit at that stride, so the two must move together
            st_global(base + cutlass.Int64(i * 4), zero, cutlass.Float32)
        if cutlass.const_expr(descale_dp):
            descale = div_rn_f32(zero + cutlass.Float32(1.0), _slot_value(mScaleDp))
            st_global(mDescaleDpOut.iterator.toint(), descale, cutlass.Float32)
        # the plan-time constants, in kernel-argument order, at the same 4-B pitch (the stores are spelled HERE, in the kernel
        # body -- a Python helper's ops would have no claim to this thread-0 branch; python/cudnn/AGENTS.md, "CuTeDSL kernel bodies")
        consts = (const0, const1, const2, const3, const4, const5, const6, const7, const8, const9, const10, const11, const12, const13, const14, const15)
        for i in cutlass.range_constexpr(n_consts):
            st_global(base + cutlass.Int64((const_slot0 + i) * 4), consts[i], cutlass.Float32)


@cute.kernel
def _init_scalars(
    mSlots: cute.Tensor,  # [n_slots] fp32 contiguous: the scalar block
    mScaleDp: cute.Tensor,  # [1] fp32: the caller's scale_dP (outside the block)
    mDescaleDpOut: cute.Tensor,  # [1] fp32 OUT: 1 / scale_dP (may be one of the slots)
    const0: cutlass.Float32,
    const1: cutlass.Float32,
    const2: cutlass.Float32,
    const3: cutlass.Float32,
    const4: cutlass.Float32,
    const5: cutlass.Float32,
    const6: cutlass.Float32,
    const7: cutlass.Float32,
    const8: cutlass.Float32,
    const9: cutlass.Float32,
    const10: cutlass.Float32,
    const11: cutlass.Float32,
    const12: cutlass.Float32,
    const13: cutlass.Float32,
    const14: cutlass.Float32,
    const15: cutlass.Float32,
    n_slots: cutlass.Constexpr[int],
    const_slot0: cutlass.Constexpr[int],
    n_consts: cutlass.Constexpr[int],
    descale_dp: cutlass.Constexpr[bool],
) -> None:
    """The standalone launch shape of :func:`init_scalars_body` (one block; the fused prologue's block 0 runs the same body)."""
    init_scalars_body(
        mSlots,
        mScaleDp,
        mDescaleDpOut,
        const0,
        const1,
        const2,
        const3,
        const4,
        const5,
        const6,
        const7,
        const8,
        const9,
        const10,
        const11,
        const12,
        const13,
        const14,
        const15,
        n_slots,
        const_slot0,
        n_consts,
        descale_dp,
    )


@cute.jit
def init_scalars_launch(
    slots: cute.Tensor,
    scale_dp: cute.Tensor,
    descale_dp_out: cute.Tensor,
    const0: cutlass.Float32,
    const1: cutlass.Float32,
    const2: cutlass.Float32,
    const3: cutlass.Float32,
    const4: cutlass.Float32,
    const5: cutlass.Float32,
    const6: cutlass.Float32,
    const7: cutlass.Float32,
    const8: cutlass.Float32,
    const9: cutlass.Float32,
    const10: cutlass.Float32,
    const11: cutlass.Float32,
    const12: cutlass.Float32,
    const13: cutlass.Float32,
    const14: cutlass.Float32,
    const15: cutlass.Float32,
    n_slots: cutlass.Constexpr[int],
    const_slot0: cutlass.Constexpr[int],
    n_consts: cutlass.Constexpr[int],
    descale_dp: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    _init_scalars(
        slots,
        scale_dp,
        descale_dp_out,
        const0,
        const1,
        const2,
        const3,
        const4,
        const5,
        const6,
        const7,
        const8,
        const9,
        const10,
        const11,
        const12,
        const13,
        const14,
        const15,
        n_slots,
        const_slot0,
        n_consts,
        descale_dp,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


init_compiled_cache = {}


class InitScalarsRecipe(NamedTuple):
    """Build-time facts of the scalar-block init launch: the number of fp32 slots it zeroes and -- appended, defaults = today's
    artifact without constants -- the slot its first plan-time constant lands in and how many it stores from its kernel
    arguments.  Both are the artifact's ABI, so ``run_init_scalars`` checks ``consts`` against them (Rule 1)."""

    compiled: object
    n_slots: int
    const_slot0: int = 0
    n_consts: int = 0
    descale_dp: bool = True  # APPENDED: False = the artifact stores no reciprocal (no dP scalar) and run_ takes no scale_dp


def compile_init_scalars(n_slots: int, const_slot0: int = 0, n_consts: int = 0, descale_dp: bool = True) -> InitScalarsRecipe:
    """ONE thread: ``slots[0:n_slots] = 0.0``, then ``descale_dp_out[0] = 1.0 / scale_dp[0]`` -- every amax slot must be zero
    before the first ``atomicMax`` of the first pass, and the reciprocal is derived on device (one ``div.rn.f32``, exact for
    a power of two) so no second caller input can disagree with ``scale_dp`` -- then (appended) ``slots[const_slot0 + i] =
    consts[i]`` for ``i < n_consts`` from the launch's fp32 KERNEL ARGUMENTS (``run_init_scalars(consts=)``; ``n_consts <=
    MAX_INIT_CONSTS``, the range inside the block).  Build from the slot layout alone: the VALUES are runtime arguments, so one
    artifact serves every QuantSpec and nothing is written to the device before the launch that reads it.

    ``descale_dp=False`` (appended; default = today's artifact): the body traces WITHOUT the reciprocal -- for a pipeline whose
    attention backward takes no dP scalar (the block-scaled MXFP8 row), the slots are zeroed and the constants stored, and
    ``run_init_scalars`` REFUSES a ``scale_dp`` / ``descale_dp_out`` (nothing would read or write them: Rule 1, both ways)."""
    global _FAKE_STREAM
    if not isinstance(descale_dp, bool):
        raise ValueError(f"descale_dp must be a bool (True: store 1 / scale_dp; False: no dP scalar), got {descale_dp!r}")
    if isinstance(n_slots, bool) or not isinstance(n_slots, int) or n_slots < 1:
        raise ValueError(f"n_slots must be a positive int (the fp32 slots of the scalar block), got {n_slots!r}")
    if isinstance(const_slot0, bool) or not isinstance(const_slot0, int) or const_slot0 < 0:
        raise ValueError(f"const_slot0 must be a non-negative int (the slot the first plan-time constant lands in), got {const_slot0!r}")
    if isinstance(n_consts, bool) or not isinstance(n_consts, int) or n_consts < 0:
        raise ValueError(f"n_consts must be a non-negative int (the plan-time constants the launch stores from its arguments), got {n_consts!r}")
    if n_consts > MAX_INIT_CONSTS:
        raise ValueError(f"n_consts={n_consts} exceeds the {MAX_INIT_CONSTS} constant arguments the scalar-init launch's ABI reserves")
    if const_slot0 + n_consts > n_slots:
        raise ValueError(f"the {n_consts} plan-time constants at slots [{const_slot0}, {const_slot0 + n_consts}) do not fit the {n_slots}-slot block")
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    key = (int(n_slots), int(const_slot0), int(n_consts), current_device(), bool(descale_dp))  # today's key is the prefix
    if key not in init_compiled_cache:
        slots = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (int(n_slots),), stride_order=(0,), assumed_align=4)
        init_compiled_cache[key] = cute.compile(
            init_scalars_launch,
            slots,
            _fake_slot(),
            _fake_slot(),
            *[cutlass.Float32(0.0) for _ in range(MAX_INIT_CONSTS)],  # the constants: runtime fp32 arguments (the zeros pin the TYPE only)
            int(n_slots),
            int(const_slot0),
            int(n_consts),
            bool(descale_dp),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return InitScalarsRecipe(
        compiled=init_compiled_cache[key], n_slots=int(n_slots), const_slot0=int(const_slot0), n_consts=int(n_consts), descale_dp=bool(descale_dp)
    )


def run_init_scalars(
    r: InitScalarsRecipe,
    slots: torch.Tensor,
    scale_dp: Optional[torch.Tensor] = None,
    descale_dp_out: Optional[torch.Tensor] = None,
    consts: tuple = (),
    *,
    stream,
) -> None:
    """Launch.  ``slots`` the contiguous fp32 ``[n_slots]`` view of the scalar block; ``scale_dp`` the caller's 1-element fp32
    CUDA scalar -- OUTSIDE the block (a slot would be zeroed before it is read: 1 / 0); ``descale_dp_out`` a 1-element fp32
    view, INTO the same block or not (written after the zeroing, sequentially in one thread, so no race); ``consts``
    (appended) the artifact's ``n_consts`` plan-time constants as finite Python numbers, in slot order -- handed to the kernel
    as fp32 ARGUMENTS (RN-rounded like ``torch.full``'s fp32 fill) and stored at ``slots[const_slot0 + i]`` after the
    reciprocal, so ``descale_dp_out`` may not lie in that range.  A length other than ``n_consts`` is a typed error (the ABI is
    fixed per artifact); a tensor is refused (a device fill is exactly what the argument path replaces).  Host checks only.

    ``scale_dp`` / ``descale_dp_out`` are REQUIRED iff the artifact divides (``r.descale_dp``) and REFUSED otherwise -- a
    ``descale_dp=False`` artifact binds the first slot in their place (never read, never written: the arm is folded out)."""
    check_scalar_slot("slots", slots, numel=int(r.n_slots))
    if r.descale_dp:
        if scale_dp is None or descale_dp_out is None:
            raise ValueError("this artifact stores descale_dp = 1 / scale_dp (compile_init_scalars(descale_dp=True)): pass scale_dp= and descale_dp_out=")
        check_scalar_slot("scale_dp", scale_dp)
        check_scalar_slot("descale_dp_out", descale_dp_out)
        lo, hi = slots.data_ptr(), slots.data_ptr() + 4 * int(r.n_slots)
        if lo <= scale_dp.data_ptr() < hi:
            raise ValueError("scale_dp lies inside the slot block this launch zeroes: it would be read as 0 (descale_dp = inf); pass the caller's own scalar")
        if scale_dp.device != slots.device or descale_dp_out.device != slots.device:
            raise ValueError(f"slots, scale_dp and descale_dp_out must be on one CUDA device, got {slots.device}, {scale_dp.device}, {descale_dp_out.device}")
    else:
        if scale_dp is not None or descale_dp_out is not None:
            raise ValueError(
                "this artifact was built with descale_dp=False (no dP scalar: the reciprocal is folded out of the body): "
                "scale_dp= and descale_dp_out= must be None -- nothing would read or write them"
            )
        lo = slots.data_ptr()
        scale_dp = descale_dp_out = slots[:1]  # the ABI's two slot tensors, bound to a slot the folded-out arm never touches
    if len(consts) != int(r.n_consts):
        raise ValueError(f"this artifact stores n_consts={r.n_consts} plan-time constants; got {len(consts)} consts (the ABI is fixed per artifact)")
    values = []
    for i, v in enumerate(consts):
        if isinstance(v, (bool, torch.Tensor)) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)):
            raise ValueError(
                f"consts[{i}] must be a finite Python number (a plan-time constant handed to the kernel as an argument -- never a device tensor), got {v!r}"
            )
        values.append(float(v))
    if r.n_consts and r.descale_dp:
        clo, chi = lo + 4 * int(r.const_slot0), lo + 4 * (int(r.const_slot0) + int(r.n_consts))
        if clo <= descale_dp_out.data_ptr() < chi:
            raise ValueError(
                f"descale_dp_out lies inside slots [{r.const_slot0}, {r.const_slot0 + r.n_consts}) the plan-time constants overwrite: it would hold a constant, not 1 / scale_dp"
            )
    args = [cutlass.Float32(v) for v in values] + [cutlass.Float32(0.0) for _ in range(MAX_INIT_CONSTS - int(r.n_consts))]
    r.compiled(slots, _slot_view(scale_dp), _slot_view(descale_dp_out), *args, cuda.CUstream(int(stream)))


frost_quantize_fp8.set_name_prefix("cudnn", remove_cutlass_symbol=True)
frost_amax_abs.set_name_prefix("cudnn", remove_cutlass_symbol=True)
frost_amax_partials.set_name_prefix("cudnn", remove_cutlass_symbol=True)
_init_scalars.set_name_prefix("cudnn", remove_cutlass_symbol=True)
