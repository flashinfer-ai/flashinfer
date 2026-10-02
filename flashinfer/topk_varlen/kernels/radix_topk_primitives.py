# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Coarse-histogram top-k kernel written against the CuTe-DSL *primitives* API.

This is the ``radix_primitives`` backend of ``top_k_varlen``.  Two deliberate
constraints shape the code:

1. **No CuTe layout algebra.**  The kernel body uses raw pointers, integer
   address arithmetic, ``cute.arch.*`` PTX wrappers, and ``llvm.inline_asm``.
   ``cute.Tensor`` appears only as an opaque argument carrier at the FFI
   boundary; the first thing the kernel does is extract ``.iterator`` base
   pointers and never touches the tensors again.  No ``make_layout``, no
   ``make_tensor``, no copy atoms.

2. **sglang's DeepSeek-V4 top-k algorithm** (sgl_kernel/deepseek_v4/
   topk_impl.cuh) rather than the multi-round radix select of the ``radix``
   backend:

     Pass 1   one coarse histogram over ordered keys (8192 bins for 16-bit
              dtypes, 4096 fp16-derived bins for fp32) -- replaces 2-4
              sequential radix rounds.
     Search   block-wide inclusive scan of the histogram (warp shuffles +
              ``redux.sync``) finds the threshold bin: the unique bin where
              ``above < k <= above + count`` (keys are ascending, so "above"
              means bins strictly greater).
     Pass 2   re-walk the row; elements in bins above the threshold are
              emitted immediately (their order among themselves is an atomic
              race, same contract as the ``radix`` backend); elements *in*
              the threshold bin are staged into a bounded tie buffer.
     Tie      exact radix select over the staged candidates resolves the
              remaining slots.
     Overflow if the tie set exceeds the buffer (near-constant rows), refine
              the pivot to an exact key with extra histogram rounds over the
              remaining key bits, then re-collect.  This keeps the result
              exact where sglang would silently select from a truncated
              candidate set.

Three execution shapes, chosen at compile time (N is static) and host side:

* **Register path** (N small enough that the row fits in <= 2 vector slots
  per thread and rows are 16B aligned): the row is read from gmem exactly
  once; both passes run over registers (sglang TopKRegister).
* **Streaming path** (single CTA, any N): two vectorized passes over gmem,
  no data staging at all -- no chunk_size / SMEM-capacity machinery.
* **Multi-CTA groups** (long rows, spare SMs; ``ctas_per_group > 1``): the
  row is column-chunked across a CTA group.  Each CTA histograms its chunk
  locally (register path per chunk when possible), merges into a per-group
  gmem histogram with one relaxed-atomic round, and the group synchronizes
  with a monotonic arrival counter (``red.release.gpu`` arrive +
  ``ld.acquire.gpu`` spin -- the FlashInfer inter-CTA barrier idiom, here
  through the native cute.arch wrappers).  gt-emits use a single global
  position counter (< top_k atomics total); ties stage into a per-group gmem
  buffer and rank 0 alone runs the exact tie select.  One merge round + 4
  barriers per row, vs 2-4 merge rounds + ~6 barriers in the ``radix``
  backend.

Implementation notes:

* sglang classifies fp32 elements in pass 2 by comparing against precomputed
  *float* bin boundaries (saves an F2F per element) -- but an element exactly
  at a boundary midpoint can round ties-to-even into the *other* bin than the
  compare picks, desynchronizing pass 2 from the histogram.  We instead
  recompute the integer bin per element in pass 2, which is bit-identical to
  pass 1 by construction and also classifies NaN consistently.

* The per-element work of each pass is dispatched via the constexpr ``mode``
  argument of ``_elem_op`` with ALL state passed as explicit function
  arguments, not as Python closures over kernel-scope variables: the DSL's
  region rewriter only threads a variable into a generated region function if
  it appears in one of that region's direct statements, so a variable
  referenced *only* inside a nested closure arrives as ``None``.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait

_ENABLE_PDL = True

# Capacity of the smem tie stage: 2048 uint32 keys + 2048 int32 indices =
# 16 KiB.  top_k <= TIE_CAP resolves every tie through the stage; larger
# top_k adds the compile-time-gated EQFILL arms, which fill the output
# directly from prefix-equal ties beyond it.
TIE_CAP = 2048

# Multi-CTA solo cutoff: a row this short is resolved by rank 0 alone with
# the single-CTA streaming algorithm instead of being split across the group.
# Splitting a short row leaves most ranks idle yet still charges every
# inter-CTA barrier; measured on B200, one CTA streaming a 32K row beats an
# 8-CTA group on the same row by ~1.5-2x (the mixed-scenario loss cells).
_MC_SOLO_ELEMS = 32768

# One block per row / chunk, fixed (FlashInfer convention).
NUM_THREADS = 1024
NUM_WARPS = NUM_THREADS // 32

# _elem_op modes (constexpr dispatch of the per-element pass body).
_OP_HIST = 0  # pass 1: coarse histogram (smem)
_OP_COLLECT = 1  # pass 2 single-CTA: emit gt, stage ties in smem
_OP_REFINE = 2  # overflow: byte histogram of threshold-bin members (smem)
_OP_FINAL = 3  # overflow single-CTA: final collect against the exact pivot
_OP_COLLECT_MC = 4  # pass 2 multi-CTA: gmem counters + gmem tie buffer
_OP_FINAL_MC = 5  # overflow multi-CTA: final collect with gmem counters
_OP_STAGE = 6  # overflow: emit prefix-above, stage prefix-matching (smem)
_OP_STAGE_MC = 7  # overflow multi-CTA: same via gmem counters/tie buffer
_OP_EQFILL = 8  # big-k overflow: fill output directly from prefix-equal ties
_OP_EQFILL_MC = 9  # big-k overflow multi-CTA: same via gmem counters

# Compile-time toggle for the collect-fused (12,12) refine histogram
# (single-CTA / solo fp32 paths): True builds it during the collect pass so
# the overflow refinement can skip that round's re-read; False makes the
# (12,12) round re-read the row like the other rounds.  OFF: the
# histogram is consumed only when more than TIE_CAP candidates tie at the
# boundary, yet every exact fp32 row paid its zeroing pass + barrier and one
# extra smem atomic per tie (ncu: exact-vs-approx +167 cycles in the threshold
# phase on 16 decode rows); flood rows now pay one L2-hot re-read instead.
_FUSE_COLLECT_R1 = False

# Multi-CTA per-group state layout (int32 units).  The full coarse histogram
# never round-trips through gmem: each CTA's local smem histogram is reduced
# in TWO tiny levels -- a 256-bin super-histogram (groups of hist_size/256
# adjacent bins), then only the winning super-bucket's fine bins -- cutting
# per-CTA global merge traffic from hist_size ints to ~300.
#   [0, 256)        merged super-histogram
#   [256, 320)      merged fine bins of the winning super-bucket (<= 32 used)
#   [320, 320+R)    refine-merge histograms, one (1 << bits)-bin slot per
#                   overflow round (fp32: 4096 + 4096 + 256; 16-bit: 256) --
#                   sized by ``refine_rounds``, so R is dtype-dependent
#   320+R+0 : arrival counter (monotonic)   +1 : out (gt) counter
#   320+R+2 : tie stage counter             +3 : final-eq / stage counter
#   320+R+4 : max(tie key) across the group +5 : max(~tie key) (i.e. min)
#   320+R+6 : departure counter (peers past their last barrier)
#   [320+R+7, ... +TIE_CAP)          staged tie keys (uint32 patterns)
#   [... , ... +2*TIE_CAP)           staged tie indices
# Row-refinement round geometry as (key_shift, bits); see __init__ for the
# derivation.  mc_state_size and the kernel layout both derive from these.
_REFINE_ROUNDS_F32 = [(24, 8), (12, 12), (0, 12)]
_REFINE_ROUNDS_16 = [(0, 8)]


def mc_state_size(is_f32: bool) -> int:
    """Per-group row_states size in int32 units for the multi-CTA mode."""
    rounds = _REFINE_ROUNDS_F32 if is_f32 else _REFINE_ROUNDS_16
    rbins = sum(1 << b for _s, b in rounds)
    return 256 + 64 + rbins + 7 + 2 * TIE_CAP  # 7 = arrive..kmaxn + depart


# ---------------------------------------------------------------------------
# PTX-level helpers
# ---------------------------------------------------------------------------
@dsl_user_op
def ld_global_v4_u32(gmem_addr: cutlass.Int64, *, loc=None, ip=None):
    """One 128-bit vectorized global load: ld.global.v4.b32.

    Takes a byte address (must be 16B aligned) and returns four Uint32 lanes.
    The multi-result pattern is an LLVM struct return unpacked with
    llvm.extractvalue -- the DSL's canonical way to get more than one value
    out of a single asm block.
    """
    st = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Uint32(llvm.extractvalue(T.i32(), st, [0])),
        cutlass.Uint32(llvm.extractvalue(T.i32(), st, [1])),
        cutlass.Uint32(llvm.extractvalue(T.i32(), st, [2])),
        cutlass.Uint32(llvm.extractvalue(T.i32(), st, [3])),
    )


@dsl_user_op
def st_shared_v4_zero(smem_addr: cutlass.Int32, *, loc=None, ip=None) -> None:
    """One 128-bit zero store to shared memory: st.shared.v4.b32 (16B aligned)."""
    llvm.inline_asm(
        None,
        [cutlass.Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "st.shared.v4.b32 [$0], {0, 0, 0, 0};",
        "r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_v4_s32(smem_addr: cutlass.Int32, *, loc=None, ip=None):
    """One 128-bit shared load: ld.shared.v4.b32 -> four Int32 (16B aligned).

    Marked as having side effects so the compiler cannot hoist or merge it
    across the block barriers and shared atomics that produce the words it
    reads (the refinement sites recorded exactly that miscompile with the
    pure form); the cost is one un-hoistable load per call site."""
    st = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [cutlass.Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Int32(llvm.extractvalue(T.i32(), st, [0])),
        cutlass.Int32(llvm.extractvalue(T.i32(), st, [1])),
        cutlass.Int32(llvm.extractvalue(T.i32(), st, [2])),
        cutlass.Int32(llvm.extractvalue(T.i32(), st, [3])),
    )


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None):
    """Read the 64-bit global nanosecond timer (%globaltimer). Debug only."""
    v = llvm.inline_asm(
        T.i64(),
        [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int64(v)


@dsl_user_op
def read_clock64(*, loc=None, ip=None):
    """Read the 64-bit SM cycle counter (%clock64).  Debug only: intra-CTA
    phase timing at cycle resolution (globaltimer ticks at ~0.3-1us on
    B200/L40S, too coarse for sub-microsecond phases).  Not comparable
    across SMs."""
    v = llvm.inline_asm(
        T.i64(),
        [],
        "mov.u64 $0, %clock64;",
        "=l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int64(v)


@cute.jit
def warp_inclusive_sum(val, lane_id):
    """Inclusive prefix sum across a warp via shfl.up.sync (5 steps)."""
    for i in cutlass.range(5, unroll_full=True):
        offset = 1 << i
        other = cute.arch.shuffle_sync_up(val, offset, mask_and_clamp=0)
        if lane_id >= offset:
            val = val + other
    return val


@cute.jit
def warp_sum(val):
    """Butterfly all-reduce sum over the full warp (single redux.sync.add)."""
    return cute.arch.warp_redux_sync(val, "add")


@cute.jit
def smem_atomic_add(ptr, val):
    """CTA-scope relaxed atomic add on shared memory, returning the old value.

    Lowers to atom.relaxed.cta.shared.add.  Explicit cta scope: the
    deprecated ``cutlass.utils.distributed.atomicAdd`` the ``radix`` backend
    uses is *system*-scope even for smem, stronger (slower) than needed.
    """
    return cute.arch.atomic_add(ptr, cutlass.Int32(val), sem="relaxed", scope="cta")


@cute.jit
def gmem_atomic_add(ptr, val):
    """Device-scope relaxed atomic add on global memory (returns old)."""
    return cute.arch.atomic_add(ptr, cutlass.Int32(val), sem="relaxed", scope="gpu")


@cute.jit
def gmem_red_add(ptr, val):
    """Fire-and-forget device-scope add (red.relaxed.gpu.global.add.s32)."""
    cute.arch.red(
        ptr, cutlass.Int32(val), op="add", dtype="s32", sem="relaxed", scope="gpu"
    )


# ---- Ampere warp-aggregated atomics (warp_agg=True kernels only) ----------
#
# B200/H100 execute same-address smem atomics at near-issue rate, and every
# warp-aggregation scheme MEASURED SLOWER there (see the negative-result
# notes below/in the project log).  Ampere-class SMs (SM80/86/89) serialize
# them instead: an all-tie row's per-element counter/histogram atomics
# dominate the kernel (constant fp32 N=65536 b=64: 187us vs sglang's 32us).
# These helpers trade ~3 warp instructions per element for a 32x reduction
# in atomic traffic; they are compiled ONLY into warp_agg=True (cc 8.x)
# kernels, so SM90+ codegen is untouched.


# NOTE (measured, do NOT retry): warp-collective aggregation of the walker's
# counter/histogram atomics (full-warp ballots + leader reservation + shfl
# broadcast, wa_full-gated) FIXED the Ampere flood cells' contention but
# serialized the double-buffered load pipeline: every stream-walker randn
# cell regressed 4-10x on A100/L40S (32K b=1: 17.3 -> 70us).  The
# thread-local vector batching in _elem_vec_wa gets the flood relief with
# zero warp synchronization instead.


# NOTE (measured, do NOT retry): a fire-and-forget smem red.add for the
# collect-fused histogram made saturated low-entropy cells 2-3x SLOWER than
# the returning smem atom.add (constant b=256: 73 -> 212us) while red.MAX on
# the same rows is ~free.  The returning atomic stays.


@cute.jit
def smem_red_max_u32(ptr, val):
    """Fire-and-forget CTA-scope unsigned max on smem (no return, no chain).

    Used to track the tie group's max ordered key (and max of ~key, i.e.
    the min) during collect: zero is the identity because every real
    float's ordered key and its complement are nonzero, so the slots need
    only the zero-init the counters already get."""
    cute.arch.red(
        ptr, cutlass.Uint32(val), op="max", dtype="u32", sem="relaxed", scope="cta"
    )


@cute.jit
def gmem_red_max_u32(ptr, val):
    """Fire-and-forget device-scope unsigned max on gmem."""
    cute.arch.red(
        ptr, cutlass.Uint32(val), op="max", dtype="u32", sem="relaxed", scope="gpu"
    )


@cute.jit
def mc_barrier(arrive_ptr, target, tidx):
    """Inter-CTA barrier over a monotonic gmem arrival counter.

    The LEADING bar.sync is load-bearing: without it, thread 0's release-red
    covers only thread 0's own prior accesses, and plain stores from the
    CTA's other warps (e.g. tie-buffer staging) can arrive at the reader
    AFTER the barrier "completes" -- an intermittent, load-dependent
    corruption (observed as a handful of wrong selections per row at high SM
    occupancy).  bar.sync establishes intra-CTA happens-before, so the
    release then publishes the whole CTA's writes; the acquire spin plus the
    trailing bar.sync symmetrically covers the reader side.
    """
    cute.arch.barrier()
    if tidx == 0:
        cute.arch.red(
            arrive_ptr,
            cutlass.Int32(1),
            op="add",
            dtype="s32",
            sem="release",
            scope="gpu",
        )
        while (
            cute.arch.load(arrive_ptr, cutlass.Int32, sem="acquire", scope="gpu")
            < target
        ):
            pass
    cute.arch.barrier()


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
class CoarseHistTopKPrimitivesKernel:
    """Coarse-histogram top-k at the primitives level (see module docstring)."""

    def __init__(
        self,
        dtype: cutlass.Numeric,
        top_k: int,
        next_n: int = 1,
        compress_ratio: int = 1,
        return_values: bool = False,
        ctas_per_group: int = 1,
        chunk_elems: int = 0,
        num_sms: int = 148,
        min_blocks_per_mp: int = 0,
        boundary_cls: bool = False,
        approx_ties: bool = False,
        enable_pdl: bool = True,
        warp_agg: bool = False,
        debug_solo_ts: bool = False,
        nt: int = 1024,
        reg_slots: int = 2,
        page_size: int = 0,
        pt_pages: int = 0,
    ):
        # Block size, parameterized for kernel-family specialization (the
        # k<=1024 walk-first family runs 512 threads for 2 CTAs/SM).  All
        # nt uses below are trace-time Python ints, so nt=1024 folds to
        # code bit-identical with the pre-parameterization kernels.
        assert nt in (512, 1024)
        self.nt = nt
        self.nw = nt // 32
        # Register-path capacity of the single-CTA kernel in 16-byte vector
        # slots per thread: 2 = sglang's TopKRegister<2> shape (8192 fp32 /
        # 16384 16-bit elements), 4 = TopKRegister<4> (16384 fp32 / 32768
        # 16-bit).  Trace-time cap for taking the register path instead of
        # the streaming loop; every slot's load is bounded by the REAL row
        # length, so a short row in a wide buffer never issues the extra
        # slots.  MEASURED 2026-09-29 (B200, 16 real decode rows of 500-2100
        # columns, K=512, raw SGLang kernel as reference): at a 16K width the
        # streaming loop + scan collect + emit re-walk cost 0.55 us of a 3.3 us
        # row (8 block barriers vs 5), while the same rows at an 8K width
        # (register path) tied SGLang (3.21 vs 3.15).  4 slots hold 16 data
        # words per thread and need the 64-register budget (min_blocks_per_mp
        # <= 1); the multi-CTA chunk paths keep 2.
        assert reg_slots in (1, 2, 3, 4)
        self.reg_slots = reg_slots
        # in-smem tie select holds exactly 2 candidates per thread
        self.tie_cap = 2 * nt
        # top_k > TIE_CAP is supported: those specializations compile extra
        # const-gated EQFILL arms; top_k <= TIE_CAP kernels are bit-identical
        # to before the big-k support.
        #
        # enable_pdl compiles the griddepcontrol (PDL) instructions in/out:
        # they exist on SM90+ only, so Ampere targets pass False.
        self.enable_pdl = enable_pdl
        # warp_agg switches the stream walker's HIST/COLLECT to the
        # thread-local vector-batched _elem_vec_wa (one counter reservation
        # per 16B vector instead of per element).  True on Ampere (SM8x)
        # only, where same-address smem atomics serialize; SM90+ measured
        # fastest with plain per-element atomics.
        self.warp_agg = warp_agg
        self.dtype = dtype
        self.top_k = top_k
        # sglang-DSv4-compatible relaxation: when > TIE_CAP candidates tie
        # at the coarse-bin boundary, fill the remaining slots from the
        # first-arrival staged prefix instead of the exact key refinement
        # (the strictly-greater part stays exact).  Rides on the
        # uniform-tie fast path's emission arm without its uniformity
        # requirement, so overflow rows run at fast-path cost.
        self.approx_ties = approx_ties
        # collect-fused (12,12) refine histogram: producers, their s_hist
        # re-zero + barrier and the overflow consumer all hang off this one
        # trace-time flag (see _FUSE_COLLECT_R1)
        self._fuse_r1 = (
            _FUSE_COLLECT_R1 and dtype == cutlass.Float32 and not approx_ties
        )
        # Paged output of the single-CTA kernel (the API's page_table
        # argument, SGLang's DSA decode contract): 0 = raw columns; a power
        # of two = every emit lands in a top_k-int smem stage and the row
        # epilogue stores physical slots page_table[q][c >> lg] << lg |
        # (c & (page_size - 1)) in one coalesced pass (-1 pads kept), q =
        # row // next_n.  pt_pages = ceil(N / page_size) entries per table
        # row; the leading pt_cache of them are staged in smem at kernel
        # entry (loads issued before the length load, stored after it), the
        # rest gather from global memory.  Same design as walk-first's
        # (measured there: mapping inside the scattered emits cost 0.26-0.4
        # us per call; the staged pass 0.0-0.2).
        assert page_size >= 0 and (page_size & (page_size - 1)) == 0
        self.page_size = page_size
        self.pt_pages = pt_pages
        self.pt_cache = min(pt_pages, 4 * NUM_THREADS, 2048) if page_size else 0
        # (A PTX `exit` right after the common tie paths, to dodge the far
        # exit stub ncu had flagged at 0.17 us, was built and measured
        # NEUTRAL on the decode trace -- 3.54 vs 3.56 us -- and removed, the
        # same verdict walk-first reached.)
        self.next_n = next_n
        self.compress_ratio = compress_ratio
        # Trace-time constant; every out_val_row use is guarded on it.
        self.has_values = return_values
        self.ctas_per_group = ctas_per_group
        # the group kernel stages at most TIE_CAP strictly-greater hits per
        # CTA (_OP_COLLECT_MC): gt < top_k <= TIE_CAP is its invariant, so the
        # host routes larger top_k to the single-CTA kernel (EQFILL arms)
        assert ctas_per_group == 1 or top_k <= TIE_CAP, (ctas_per_group, top_k)
        self.chunk_elems = chunk_elems  # only meaningful when ctas_per_group > 1
        self.num_sms = num_sms
        # __launch_bounds__(NUM_THREADS, min_blocks_per_mp) ptxas hint for the
        # single-CTA kernel (0 = unset).  At 32 regs/thread two 1024-thread
        # CTAs co-reside per SM100, halving the tail wave when the batch
        # oversubscribes the SMs (the regime where sglang's
        # __launch_bounds__(1024, 2) was ahead).
        self.min_blocks_per_mp = min_blocks_per_mp
        # fp32 collect passes classify with two float compares against
        # per-row bin-boundary values instead of recomputing the fp16 coarse
        # bin per element (sglang's coarse_bin_lower_bound trick, made exact
        # under ties-to-even by a parity bump -- see the helper).  The
        # boundaries cost ~20 instructions once per row.
        self.boundary_cls = boundary_cls and (dtype == cutlass.Float32)
        # Debug: rank-0 solo path stores %globaltimer low bits at phase
        # boundaries into g_state[g_tiek + 0..5] (content-only; counters
        # untouched, so replay safety is unaffected).  Compiled out unless
        # set; never enable in production compiles.
        self.debug_solo_ts = debug_solo_ts

        self.is_f32 = dtype == cutlass.Float32
        if self.is_f32:
            # fp32 is binned through an fp16 conversion (sglang's trick): the
            # 5-bit fp16 exponent leaves 6 mantissa bits inside a 12-bit bin,
            # much finer than the top 12 bits of the fp32 pattern itself
            # (8-bit exponent, only 3 mantissa bits) for typical logit ranges.
            self.hist_bits = 12
            self.elem_bytes = 4
            self.vec_elems = 4  # elements per 16-byte vector load
            # Exact keys are 32-bit ordered fp32.  tie_select's in-smem
            # radix keeps four byte rounds; the ROW-refinement rounds of the
            # overflow path are (shift, bits) digits chosen from the bin
            # geometry: one coarse (fp16-derived) bin spans at most ~2^18
            # fp32 key ULPs for every NORMAL fp16 bin, so key bits [24, 32)
            # are bin-determined and the (24, 8) round is SKIPPED at runtime
            # unless the threshold bin is one of the wide bins (+/-inf,
            # +/-0/denormal collapse) where the span bound fails.  The two
            # 12-bit rounds reuse the 4096-bin smem histogram +
            # find_threshold_coarse scan; the first executed round's
            # histogram is fused into the collect pass on the single-CTA and
            # solo paths (s_hist is idle there during collect).  After the
            # covered bits [0, 24) match, in-bin candidates are provably
            # key-identical, so a masked STAGE pass + exact in-smem
            # tie_select replaces the FINAL row pass.
            self.exact_shifts = [24, 16, 8, 0]
            self.refine_rounds = _REFINE_ROUNDS_F32
        else:
            # 16-bit dtypes bin on the top 13 bits of the ordered 16-bit key:
            # 8192 bins x 4B = 32 KiB smem, and only 3 key bits are left
            # unresolved inside a bin -- one byte round resolves them.
            self.hist_bits = 13
            self.elem_bytes = 2
            self.vec_elems = 8
            self.exact_shifts = [0]
            self.refine_rounds = _REFINE_ROUNDS_16
        self.hist_size = 1 << self.hist_bits
        self.coarse_shift = 16 - self.hist_bits  # coarse keys are 16-bit
        self.hist_items = self.hist_size // self.nt  # bins owned per thread

        # Multi-CTA per-group state offsets (int32 units).  The refine-merge
        # slots are sized per round ((1 << bits) bins each).
        self.super_w = self.hist_size // 256  # local bins per super-bucket
        self.g_super = 0
        self.g_fine = 256
        self.g_rhist = 320
        self.refine_slot_off = []
        off = self.g_rhist
        for _sh, _bits in self.refine_rounds:
            self.refine_slot_off.append(off)
            off += 1 << _bits
        self.g_arrive = off
        self.g_out = self.g_arrive + 1
        self.g_eq = self.g_arrive + 2
        self.g_eqf = self.g_arrive + 3
        self.g_kmax = self.g_arrive + 4  # max tie key across the group
        self.g_kmaxn = self.g_arrive + 5  # max ~tie key (encodes the min)
        self.g_depart = self.g_arrive + 6  # peers past their last barrier
        self.g_tiek = self.g_arrive + 7
        self.g_tiei = self.g_tiek + TIE_CAP
        self.state_size = mc_state_size(self.is_f32)
        assert self.state_size == self.g_tiei + TIE_CAP

    # ------------------------------------------------------------------
    # Ordered-key helpers (ascending: larger float => larger key)
    # ------------------------------------------------------------------
    @cute.jit
    def to_key16(self, bits):
        """Ascending ordered key of a 16-bit float pattern held in a Uint32.

        negative (sign set): flip all 16 bits; positive: set the sign bit.
        Branchless: xor_mask = 0x8000 | (0xFFFF if sign else 0).
        """
        sign = bits >> cutlass.Uint32(15)  # 0 or 1
        xor_mask = (
            (cutlass.Uint32(0) - sign) & cutlass.Uint32(0xFFFF)
        ) | cutlass.Uint32(0x8000)
        return (bits ^ xor_mask) & cutlass.Uint32(0xFFFF)

    @cute.jit
    def from_key16(self, key):
        """Inverse of to_key16: ordered key -> raw 16-bit float pattern.

        Key sign bit set <=> original was positive (mask was 0x8000), so
        invert with 0x8000; otherwise the whole 16 bits were flipped.
        """
        sign = key >> cutlass.Uint32(15)
        xor_mask = (
            (cutlass.Uint32(0) - (cutlass.Uint32(1) - sign)) & cutlass.Uint32(0xFFFF)
        ) | cutlass.Uint32(0x8000)
        return (key ^ xor_mask) & cutlass.Uint32(0xFFFF)

    @cute.jit
    def to_key32(self, bits):
        """Ascending ordered key of an fp32 pattern (CUB TwiddleIn idiom)."""
        sign = bits >> cutlass.Uint32(31)
        xor_mask = (cutlass.Uint32(0) - sign) | cutlass.Uint32(0x80000000)
        return bits ^ xor_mask

    @cute.jit
    def from_key32(self, key):
        """Inverse of to_key32."""
        sign = key >> cutlass.Uint32(31)  # 1 <=> original positive
        xor_mask = (cutlass.Uint32(0) - (cutlass.Uint32(1) - sign)) | cutlass.Uint32(
            0x80000000
        )
        return key ^ xor_mask

    @cute.jit
    def _canon_zero_f32(self, bits):
        """fp32 element bits with -0.0 mapped to +0.0.  The float-boundary
        collect compares VALUES, where -0.0 == +0.0, while the fp16 coarse
        bin and the ordered key keep the sign: a row whose rank-k value was
        -0.0 with +0.0 values ranked above it lost every +0.0 (the gt test
        ``v <= T`` with T = -0.0 holds for +0.0 too) and came back
        underfilled.  With one zero in every bin and key the histogram and
        the compares agree again; the two zeros are equal as values, so the
        selected value multiset is unchanged and any zero's index is exact.
        Two instructions per element (shift, select)."""
        bz = cutlass.Uint32(bits)
        if (bz << cutlass.Uint32(1)) == cutlass.Uint32(0):
            bz = cutlass.Uint32(0)
        return bz

    @cute.jit
    def coarse_bin(self, bits):
        """Element bits -> coarse histogram bin (Int32).

        fp32 goes through cvt.rn.f16.f32 first (after the signed-zero
        canonicalization of _canon_zero_f32); every pass uses this same
        function, so classification is bit-identical to the histogram.
        """
        if cutlass.const_expr(self.is_f32):
            f = self._canon_zero_f32(bits).bitcast(cutlass.Float32)
            h = f.to(cutlass.Float16)
            hbits = h.bitcast(cutlass.Uint16).to(cutlass.Uint32)
            key = self.to_key16(hbits)
        else:
            key = self.to_key16(bits)
        return cutlass.Int32(key >> cutlass.Uint32(self.coarse_shift))

    @cute.jit
    def coarse_bin_lower_bound_f32(self, b):
        """Smallest fp32 value v with coarse_bin(v) >= ``b`` (fp32 only).

        coarse_bin is monotone in v (fp32->fp16 cvt.rn preserves order), so
        the collect pass can classify with ``v >= bound`` float compares
        instead of recomputing the bin per element.  Construction (sglang's
        coarse_bin_lower_bound, plus a parity fix): the boundary is the
        midpoint of the fp16 values at ordered-16 keys ``b<<4`` and
        ``b<<4 - 1``; under round-to-nearest-EVEN an exact midpoint whose
        upper fp16 has an ODD mantissa LSB rounds DOWN, so the true boundary
        is then one fp32 ordered-key step above the midpoint.  Verified
        bit-exact host-side over 51k adversarial values x every achievable
        threshold bin (host-side proof script); the only deviations are
        equal-value reclassifications (-0 vs +0, +inf at the inf bin) that
        cannot change the selected value multiset.  Called once per row with
        b = threshold_bin and threshold_bin + 1 (range [63, 4033]).
        """
        res = cutlass.Uint32(0xFF800000).bitcast(cutlass.Float32)  # -inf
        key = cutlass.Uint32(b) << cutlass.Uint32(self.coarse_shift)
        if key > cutlass.Uint32(0x03FF):
            # keys at/below the -inf key: every real value qualifies.
            if key > cutlass.Uint32(0xFC00):
                # +NaN key space (only used for the TIE lower bound; the gt
                # threshold comes from coarse_bin_gt_threshold_f32): a
                # NaN-space threshold bin has no real ties, and qNaN makes
                # every ``v >= res`` compare false.
                res = cutlass.Uint32(0x7FC00000).bitcast(cutlass.Float32)
            else:
                # fp16 value at ordered key k; the +/-inf keys are treated
                # as +/-65536 so the midpoint lands on +/-65520, the exact
                # cvt.rn overflow threshold.
                hb_hi = self.from_key16(key)
                v_hi = cutlass.Float32(65536.0)
                if key < cutlass.Uint32(0xFC00):
                    v_hi = (
                        hb_hi.to(cutlass.Uint16)
                        .bitcast(cutlass.Float16)
                        .to(cutlass.Float32)
                    )
                keym1 = key - cutlass.Uint32(1)
                v_lo = cutlass.Float32(-65536.0)
                if keym1 > cutlass.Uint32(0x03FF):
                    v_lo = (
                        self.from_key16(keym1)
                        .to(cutlass.Uint16)
                        .bitcast(cutlass.Float16)
                        .to(cutlass.Float32)
                    )
                m = cutlass.Float32(0.5) * (v_hi + v_lo)
                if (hb_hi & cutlass.Uint32(1)) != 0:
                    # odd upper fp16: midpoint rounds DOWN; boundary is the
                    # next fp32 above m in ordered-key space.
                    m = self.from_key32(
                        self.to_key32(m.bitcast(cutlass.Uint32)) + cutlass.Uint32(1)
                    ).bitcast(cutlass.Float32)
                res = m
        return res

    @cute.jit
    def _track_tie_range(self, s_misc, kt):
        """Track the tie set's exact-key range [min, max] via
        fire-and-forget CTA reds (uniform-tie fast path + the range-driven
        round skip).  Compiled out in the approx_ties variant, whose
        unconditional clamp never reads the range -- the approx collect
        then costs one atomic per tie, like sglang's."""
        if cutlass.const_expr(not self.approx_ties):
            smem_red_max_u32(s_misc + 9, kt)
            smem_red_max_u32(s_misc + 8, ~kt)

    @cute.jit
    def coarse_bin_gt_threshold_f32(self, b):
        """Strict-GT threshold for the collect's NaN-robust gt test (fp32).

        The collect classifies with the INVERTED branch ``if val <= T:
        (tie check) else: (gt)`` so that every NaN -- which fails all
        ordered compares -- lands in the gt arm, consistent with the
        histogram placing +NaN keys above +inf (NaNs rank top, like
        torch.topk, instead of being silently dropped and underfilling
        the row).  For that branch shape, T must be the PREDECESSOR (one
        fp32 ordered-key step below) of coarse_bin_lower_bound_f32(b): no
        fp32 lies between them, so ``not (v <= T)`` == ``v >= bound`` for
        every real v.  NaN-space ``b`` returns +inf: reals and +/-inf then
        take the tie-check branch and only NaN can classify gt (this also
        removes the GE scheme's +inf gt reclassification at the inf bin --
        +inf now TIES exactly like the integer-bin path).  Proven bit-exact
        host-side over 51k adversarial values x 117 threshold bins
        (host-side proof script, 0 mismatches).
        """
        # key <= 0x03FF is unreachable here (threshold_bin >= bin(-inf) =
        # 63, so b = threshold_bin + 1 has key >= 0x400); qNaN (everything
        # gt) keeps that degenerate case safe regardless.
        res = cutlass.Uint32(0x7FC00000).bitcast(cutlass.Float32)
        key = cutlass.Uint32(b) << cutlass.Uint32(self.coarse_shift)
        if key > cutlass.Uint32(0x03FF):
            if key > cutlass.Uint32(0xFC00):
                res = cutlass.Uint32(0x7F800000).bitcast(cutlass.Float32)
            else:
                hb_hi = self.from_key16(key)
                v_hi = cutlass.Float32(65536.0)
                if key < cutlass.Uint32(0xFC00):
                    v_hi = (
                        hb_hi.to(cutlass.Uint16)
                        .bitcast(cutlass.Float16)
                        .to(cutlass.Float32)
                    )
                keym1 = key - cutlass.Uint32(1)
                v_lo = cutlass.Float32(-65536.0)
                if keym1 > cutlass.Uint32(0x03FF):
                    v_lo = (
                        self.from_key16(keym1)
                        .to(cutlass.Uint16)
                        .bitcast(cutlass.Float16)
                        .to(cutlass.Float32)
                    )
                m = cutlass.Float32(0.5) * (v_hi + v_lo)
                if (hb_hi & cutlass.Uint32(1)) == 0:
                    # even upper fp16: the inclusive boundary is the exact
                    # midpoint m, so its predecessor is one key step down.
                    # (Odd parity: the boundary is succ(m) after the
                    # ties-to-even bump, so the predecessor is m itself.)
                    m = self.from_key32(
                        self.to_key32(m.bitcast(cutlass.Uint32)) - cutlass.Uint32(1)
                    ).bitcast(cutlass.Float32)
                res = m
        # The predecessor of the +0.0 bin's bound is -0.0, which a float
        # compare cannot tell from +0.0 (both zeros live in that bin, see
        # _canon_zero_f32): the largest negative float is the predecessor
        # that separates them.
        if (res.bitcast(cutlass.Uint32) << cutlass.Uint32(1)) == cutlass.Uint32(0):
            res = cutlass.Uint32(0x80000001).bitcast(cutlass.Float32)
        return res

    @cute.jit
    def _dbg_ts(self, g_state, slot: cutlass.Constexpr, tidx):
        """Debug phase timestamp: store %globaltimer low bits (ns) to
        g_state[g_tiek + slot].  Compiled out unless debug_solo_ts."""
        if cutlass.const_expr(self.debug_solo_ts):
            if tidx == 0:
                (g_state + (self.g_tiek + slot)).store(
                    (read_globaltimer() & cutlass.Int64(0x7FFFFFFF)).to(cutlass.Int32)
                )

    @cute.jit
    def _block_excl_scan(self, cnt, s_warp_sums, s_count_gt, tidx):
        """Block-wide exclusive prefix sum of per-thread ``cnt`` (1024
        threads).  Returns this thread's exclusive offset and publishes the
        block total to ``s_count_gt[0]``.  Used by the scan-based collect
        emission: positions come from one scan instead of one serialized
        same-address atomic per emitted element."""
        lane = tidx % 32
        wid = tidx // 32
        incl = warp_inclusive_sum(cnt, lane)
        if lane == 31:
            s_warp_sums[wid] = incl
        cute.arch.barrier()
        if tidx < 32:
            w = s_warp_sums[tidx]
            winc = warp_inclusive_sum(w, tidx)
            s_warp_sums[tidx] = winc
        cute.arch.barrier()
        prev = cutlass.Int32(0)
        if wid > 0:
            prev = s_warp_sums[wid - 1]
        if tidx == 0:
            s_count_gt[0] = s_warp_sums[NUM_WARPS - 1]
        return prev + incl - cnt

    @cute.jit
    def exact_key(self, bits):
        """Element bits -> full-resolution ordered key (for tie breaking);
        fp32 maps -0.0 to +0.0 first (_canon_zero_f32), so the two zeros are
        one key inside the one coarse bin they share."""
        if cutlass.const_expr(self.is_f32):
            return self.to_key32(self._canon_zero_f32(bits))
        else:
            return self.to_key16(bits)

    @cute.jit
    def value_from_bits(self, bits):
        """Raw pattern (low elem_bytes of a Uint32) -> dtype value."""
        if cutlass.const_expr(self.is_f32):
            return bits.bitcast(cutlass.Float32)
        else:
            b16 = bits.to(cutlass.Uint16)
            if cutlass.const_expr(self.dtype == cutlass.Float16):
                return b16.bitcast(cutlass.Float16)
            else:
                return b16.bitcast(cutlass.BFloat16)

    @cute.jit
    def value_from_key(self, key):
        """Ordered key -> dtype value (exact; the key mapping is bijective)."""
        if cutlass.const_expr(self.is_f32):
            return self.from_key32(key).bitcast(cutlass.Float32)
        else:
            return self.value_from_bits(self.from_key16(key))

    # ------------------------------------------------------------------
    # Per-element pass body, constexpr-dispatched on ``mode``
    # ------------------------------------------------------------------
    @cute.jit
    def _collect_elem_plain(
        self,
        bits,
        idx,
        s_hist,
        s_count_gt,
        s_count_eq,
        s_tie_keys,
        s_tie_idx,
        s_misc,
        out_idx_row,
        out_val_row,
        threshold_bin,
        prefix,
        prefix_mask,
    ):
        """Per-lane-atomic collect body (all non-warp_agg kernels, and the
        warp_agg kernels' guarded/tail element slots)."""
        top_k = cutlass.const_expr(self.top_k)
        if cutlass.const_expr(self.boundary_cls):
            # prefix carries the fp32 strict-GT threshold T
            # (coarse_bin_gt_threshold_f32) and prefix_mask the tie
            # lower bound: two float compares replace the per-element
            # fp32->fp16 conversion + bin twiddle.  INVERTED branch so
            # the else arm = bin-above reals AND every NaN (NaN fails
            # all ordered compares): NaNs rank top like torch / the
            # histogram instead of being dropped (which underfilled
            # rows once NaNs outnumbered the threshold-bin slack).
            val = bits.bitcast(cutlass.Float32)
            if val <= prefix.bitcast(cutlass.Float32):  # real, bin <= tb
                if val >= prefix_mask.bitcast(cutlass.Float32):  # lo
                    kt = self.exact_key(bits)
                    # track the tie group's exact-key range (uniform-tie
                    # overflow fast path); red = no dependency chain
                    self._track_tie_range(s_misc, kt)
                    if cutlass.const_expr(self._fuse_r1):
                        # collect-fused (12,12) refine histogram: s_hist
                        # is idle here (single-CTA/solo zero it first);
                        # the approx variant never refines
                        smem_atomic_add(
                            s_hist
                            + cutlass.Int32(
                                (kt >> cutlass.Uint32(12)) & cutlass.Uint32(0xFFF)
                            ),
                            1,
                        )
                    c = smem_atomic_add(s_count_eq, 1)
                    if c < TIE_CAP:
                        s_tie_keys[c] = kt
                        s_tie_idx[c] = cutlass.Int32(idx)
            else:  # bin > tb, or NaN
                pos = smem_atomic_add(s_count_gt, 1)
                if pos < top_k:
                    out_idx_row[pos] = cutlass.Int32(idx)
                    if cutlass.const_expr(self.has_values):
                        out_val_row[pos] = self.value_from_bits(bits)
        else:
            b = self.coarse_bin(bits)
            if b > threshold_bin:
                pos = smem_atomic_add(s_count_gt, 1)
                if pos < top_k:  # guaranteed by the threshold invariant
                    out_idx_row[pos] = cutlass.Int32(idx)
                    if cutlass.const_expr(self.has_values):
                        out_val_row[pos] = self.value_from_bits(bits)
            else:
                if b == threshold_bin:
                    kt = self.exact_key(bits)
                    self._track_tie_range(s_misc, kt)
                    if cutlass.const_expr(self._fuse_r1):
                        smem_atomic_add(
                            s_hist
                            + cutlass.Int32(
                                (kt >> cutlass.Uint32(12)) & cutlass.Uint32(0xFFF)
                            ),
                            1,
                        )
                    c = smem_atomic_add(s_count_eq, 1)
                    if c < TIE_CAP:
                        s_tie_keys[c] = kt
                        s_tie_idx[c] = cutlass.Int32(idx)

    @cute.jit
    def _elem_vec_wa(
        self,
        mode: cutlass.Constexpr,
        w0,
        w1,
        w2,
        w3,
        base,
        hb,
        hc,
        sat,
        acc,
        s_hist,
        s_count_gt,
        s_count_eq,
        s_tie_keys,
        s_tie_idx,
        s_misc,
        out_idx_row,
        out_val_row,
        threshold_bin,
        prefix,
        prefix_mask,
    ):
        """Thread-local batched HIST/COLLECT over one 16B vector (warp_agg
        kernels only; semantics identical to per-element _elem_op).  Flood
        rows put every element in one bin / one counter, and Ampere-class
        SMs serialize same-address smem atomics, so the batching cuts the
        serialized traffic with ZERO warp synchronization -- warp
        collectives measured 4-10x slower here (they serialize the
        double-buffered load pipeline).

        Loop-carried thread-local state (caller initializes, threads flush
        after the vector loop):
          hb/hc  HIST run cache: pending count for bin hb.  A flood row
                 flushes its histogram once per THREAD instead of once per
                 element; randn vectors are bin-mixed, so the cache stays
                 empty and the cost is the plain per-element adds.
          sat/acc COLLECT saturating tie reservation: once the tie counter
                 has passed TIE_CAP nothing further can be staged, so ties
                 are counted locally (acc) and flushed in one atomic --
                 exactness of the final count is preserved, and the staged
                 subset contract is unchanged (a full stage only matters
                 when the group fits, i.e. before saturation).

        Element order within the vector is preserved; cross-thread order is
        atomic-race unspecified exactly like the per-element path."""
        top_k = cutlass.const_expr(self.top_k)
        if cutlass.const_expr(mode == _OP_HIST):
            same = cutlass.Boolean(True)
            b0 = cutlass.Int32(0)
            if cutlass.const_expr(self.is_f32):
                b0 = self.coarse_bin(w0)
                if self.coarse_bin(w1) != b0:
                    same = cutlass.Boolean(False)
                if self.coarse_bin(w2) != b0:
                    same = cutlass.Boolean(False)
                if self.coarse_bin(w3) != b0:
                    same = cutlass.Boolean(False)
            else:
                b0 = self.coarse_bin(w0 & cutlass.Uint32(0xFFFF))
                for j in cutlass.range_constexpr(4):
                    w = (w0, w1, w2, w3)[j]
                    if self.coarse_bin(w & cutlass.Uint32(0xFFFF)) != b0:
                        same = cutlass.Boolean(False)
                    if self.coarse_bin(w >> cutlass.Uint32(16)) != b0:
                        same = cutlass.Boolean(False)
            ne = cutlass.const_expr(4 if self.is_f32 else 8)
            if same:
                if b0 == hb:
                    hc = hc + ne
                else:
                    if hc > 0:
                        smem_atomic_add(s_hist + hb, hc)
                    hb = b0
                    hc = cutlass.Int32(ne)
            else:
                # bin-mixed vector: plain per-element adds; the run cache is
                # left as-is (adds commute)
                if cutlass.const_expr(self.is_f32):
                    smem_atomic_add(s_hist + self.coarse_bin(w0), 1)
                    smem_atomic_add(s_hist + self.coarse_bin(w1), 1)
                    smem_atomic_add(s_hist + self.coarse_bin(w2), 1)
                    smem_atomic_add(s_hist + self.coarse_bin(w3), 1)
                else:
                    for j in cutlass.range_constexpr(4):
                        w = (w0, w1, w2, w3)[j]
                        smem_atomic_add(
                            s_hist + self.coarse_bin(w & cutlass.Uint32(0xFFFF)), 1
                        )
                        smem_atomic_add(
                            s_hist + self.coarse_bin(w >> cutlass.Uint32(16)), 1
                        )
        if cutlass.const_expr(mode == _OP_COLLECT):
            if cutlass.const_expr(self.boundary_cls):
                # fp32 boundary classification (see _collect_elem_plain for
                # the threshold/NaN semantics).  Pass 1 counts the vector's
                # ties and gts; one reservation each; pass 2 assigns slots
                # in element order.
                tb = prefix.bitcast(cutlass.Float32)
                lo = prefix_mask.bitcast(cutlass.Float32)
                ntie = cutlass.Int32(0)
                ngt = cutlass.Int32(0)
                for j in cutlass.range_constexpr(4):
                    v = (w0, w1, w2, w3)[j].bitcast(cutlass.Float32)
                    if v <= tb:
                        if v >= lo:
                            ntie = ntie + 1
                    else:
                        ngt = ngt + 1
                cslot = cutlass.Int32(TIE_CAP)  # sentinel: skip staged stores
                gslot = cutlass.Int32(0)
                if ntie > 0:
                    if sat == 1:
                        acc = acc + ntie
                    else:
                        cslot = smem_atomic_add(s_count_eq, ntie)
                        if cslot >= TIE_CAP:
                            sat = cutlass.Int32(1)
                if ngt > 0:
                    gslot = smem_atomic_add(s_count_gt, ngt)
                for j in cutlass.range_constexpr(4):
                    w = (w0, w1, w2, w3)[j]
                    v = w.bitcast(cutlass.Float32)
                    if v <= tb:
                        if v >= lo:
                            kt = self.exact_key(w)
                            self._track_tie_range(s_misc, kt)
                            if cutlass.const_expr(self._fuse_r1):
                                smem_atomic_add(
                                    s_hist
                                    + cutlass.Int32(
                                        (kt >> cutlass.Uint32(12))
                                        & cutlass.Uint32(0xFFF)
                                    ),
                                    1,
                                )
                            if cslot < TIE_CAP:
                                s_tie_keys[cslot] = kt
                                s_tie_idx[cslot] = cutlass.Int32(base + j)
                            cslot = cslot + 1
                    else:  # bin > tb, or NaN
                        if gslot < top_k:
                            out_idx_row[gslot] = cutlass.Int32(base + j)
                            if cutlass.const_expr(self.has_values):
                                out_val_row[gslot] = self.value_from_bits(w)
                        gslot = gslot + 1
            else:
                # 16-bit bin classification, 8 elements per vector.
                ntie = cutlass.Int32(0)
                ngt = cutlass.Int32(0)
                for j in cutlass.range_constexpr(8):
                    w = (w0, w1, w2, w3)[j // 2]
                    e = (
                        (w & cutlass.Uint32(0xFFFF))
                        if cutlass.const_expr(j % 2 == 0)
                        else (w >> cutlass.Uint32(16))
                    )
                    b = self.coarse_bin(e)
                    if b > threshold_bin:
                        ngt = ngt + 1
                    else:
                        if b == threshold_bin:
                            ntie = ntie + 1
                cslot = cutlass.Int32(TIE_CAP)  # sentinel: skip staged stores
                gslot = cutlass.Int32(0)
                if ntie > 0:
                    if sat == 1:
                        acc = acc + ntie
                    else:
                        cslot = smem_atomic_add(s_count_eq, ntie)
                        if cslot >= TIE_CAP:
                            sat = cutlass.Int32(1)
                if ngt > 0:
                    gslot = smem_atomic_add(s_count_gt, ngt)
                for j in cutlass.range_constexpr(8):
                    w = (w0, w1, w2, w3)[j // 2]
                    e = (
                        (w & cutlass.Uint32(0xFFFF))
                        if cutlass.const_expr(j % 2 == 0)
                        else (w >> cutlass.Uint32(16))
                    )
                    b = self.coarse_bin(e)
                    if b > threshold_bin:
                        if gslot < top_k:
                            out_idx_row[gslot] = cutlass.Int32(base + j)
                            if cutlass.const_expr(self.has_values):
                                out_val_row[gslot] = self.value_from_bits(e)
                        gslot = gslot + 1
                    else:
                        if b == threshold_bin:
                            kt = self.exact_key(e)
                            self._track_tie_range(s_misc, kt)
                            if cutlass.const_expr(self._fuse_r1):
                                smem_atomic_add(
                                    s_hist
                                    + cutlass.Int32(
                                        (kt >> cutlass.Uint32(12))
                                        & cutlass.Uint32(0xFFF)
                                    ),
                                    1,
                                )
                            if cslot < TIE_CAP:
                                s_tie_keys[cslot] = kt
                                s_tie_idx[cslot] = cutlass.Int32(base + j)
                            cslot = cslot + 1
        return hb, hc, sat, acc

    @cute.jit
    def _elem_op(
        self,
        mode: cutlass.Constexpr,
        shift: cutlass.Constexpr,
        bits,
        idx,
        s_hist,
        s_count_gt,
        s_count_eq,
        s_tie_keys,
        s_tie_idx,
        s_misc,
        out_idx_row,
        out_val_row,
        threshold_bin,
        prefix,
        prefix_mask,
        pivot,
        eq_base,
        g_state,
    ):
        top_k = cutlass.const_expr(self.top_k)
        if cutlass.const_expr(mode == _OP_HIST):
            b = self.coarse_bin(bits)
            smem_atomic_add(s_hist + b, 1)
        if cutlass.const_expr(mode == _OP_COLLECT):
            self._collect_elem_plain(
                bits,
                idx,
                s_hist,
                s_count_gt,
                s_count_eq,
                s_tie_keys,
                s_tie_idx,
                s_misc,
                out_idx_row,
                out_val_row,
                threshold_bin,
                prefix,
                prefix_mask,
            )
        if cutlass.const_expr(mode == _OP_REFINE):
            # ``shift`` packs the round geometry: (bits << 8) | key_shift.
            rsh = cutlass.const_expr(shift & 0xFF)
            rmask = cutlass.const_expr((1 << (shift >> 8)) - 1)
            b = self.coarse_bin(bits)
            if b == threshold_bin:
                k = self.exact_key(bits)
                if (k & prefix_mask) == prefix:
                    sub = cutlass.Int32(
                        (k >> cutlass.Uint32(rsh)) & cutlass.Uint32(rmask)
                    )
                    smem_atomic_add(s_hist + sub, 1)
        if cutlass.const_expr(mode == _OP_FINAL):
            b = self.coarse_bin(bits)
            if b == threshold_bin:
                k = self.exact_key(bits)
                if k > pivot:
                    pos = smem_atomic_add(s_count_gt, 1)
                    if pos < top_k:
                        out_idx_row[pos] = cutlass.Int32(idx)
                        if cutlass.const_expr(self.has_values):
                            out_val_row[pos] = self.value_from_bits(bits)
                else:
                    if k == pivot:
                        e = smem_atomic_add(s_misc + 6, 1)
                        if eq_base + e < top_k:
                            out_idx_row[eq_base + e] = cutlass.Int32(idx)
                            if cutlass.const_expr(self.has_values):
                                out_val_row[eq_base + e] = self.value_from_bits(bits)
        if cutlass.const_expr(mode == _OP_COLLECT_MC):
            # Stage gt/tie hits in CTA-local smem -- gt indices (and keys,
            # when values are returned) in s_tie_*, ties in the two halves
            # of the dead coarse histogram -- with smem counters in
            # s_misc[4]/[5].  The caller reserves group output slots with
            # ONE device atomic per CTA per counter and flushes the stages.
            # Per-element device-scope atomics on the shared g_out/g_eq
            # counters were the multi-CTA collect bottleneck: hundreds of
            # same-address L2 RMWs serialized across the group (the radix
            # CuTe-DSL backend batches its gt-emit the same way).
            # Capacities: group-wide gt < top_k <= TIE_CAP by the threshold
            # invariant; ties beyond TIE_CAP only need the count (the
            # overflow path re-reads gmem and ignores the staged buffer).
            if cutlass.const_expr(self.boundary_cls):
                # Same inverted float-boundary classification as
                # _OP_COLLECT (NaNs fall into the gt arm).
                val = bits.bitcast(cutlass.Float32)
                if val <= prefix.bitcast(cutlass.Float32):  # real, bin <= tb
                    if val >= prefix_mask.bitcast(cutlass.Float32):  # lo
                        kt = self.exact_key(bits)
                        self._track_tie_range(s_misc, kt)
                        c = smem_atomic_add(s_misc + 5, 1)
                        if c < TIE_CAP:
                            s_hist[c] = kt.bitcast(cutlass.Int32)
                            s_hist[TIE_CAP + c] = cutlass.Int32(idx)
                else:  # bin > tb, or NaN
                    p = smem_atomic_add(s_misc + 4, 1)
                    if p < TIE_CAP:
                        s_tie_idx[p] = cutlass.Int32(idx)
                        if cutlass.const_expr(self.has_values):
                            s_tie_keys[p] = self.exact_key(bits)
            else:
                b = self.coarse_bin(bits)
                if b > threshold_bin:
                    p = smem_atomic_add(s_misc + 4, 1)
                    if p < TIE_CAP:
                        s_tie_idx[p] = cutlass.Int32(idx)
                        if cutlass.const_expr(self.has_values):
                            s_tie_keys[p] = self.exact_key(bits)
                else:
                    if b == threshold_bin:
                        kt = self.exact_key(bits)
                        self._track_tie_range(s_misc, kt)
                        c = smem_atomic_add(s_misc + 5, 1)
                        if c < TIE_CAP:
                            s_hist[c] = kt.bitcast(cutlass.Int32)
                            s_hist[TIE_CAP + c] = cutlass.Int32(idx)
        if cutlass.const_expr(mode == _OP_FINAL_MC):
            b = self.coarse_bin(bits)
            if b == threshold_bin:
                k = self.exact_key(bits)
                if k > pivot:
                    pos = gmem_atomic_add(g_state + self.g_out, 1)
                    if pos < top_k:
                        out_idx_row[pos] = cutlass.Int32(idx)
                        if cutlass.const_expr(self.has_values):
                            out_val_row[pos] = self.value_from_bits(bits)
                else:
                    if k == pivot:
                        e = gmem_atomic_add(g_state + self.g_eqf, 1)
                        if eq_base + e < top_k:
                            out_idx_row[eq_base + e] = cutlass.Int32(idx)
                            if cutlass.const_expr(self.has_values):
                                out_val_row[eq_base + e] = self.value_from_bits(bits)
        if cutlass.const_expr(mode == _OP_STAGE):
            # Terminal/early stage of the overflow path: candidates whose
            # covered key bits BEAT the refined prefix are winners; EQUAL
            # bits are staged (full key + index) for the exact in-smem
            # tie_select.  Caller zeroes s_count_eq first; staged count may
            # exceed TIE_CAP only when the survivors are provably
            # key-identical (bits [0,24) + bin membership determine the
            # key), where any staged subset is a valid answer.
            b = self.coarse_bin(bits)
            if b == threshold_bin:
                k = self.exact_key(bits)
                pm = k & prefix_mask
                if pm == prefix:
                    c = smem_atomic_add(s_count_eq, 1)
                    if c < TIE_CAP:
                        s_tie_keys[c] = k
                        s_tie_idx[c] = cutlass.Int32(idx)
                else:
                    if pm > prefix:
                        pos = smem_atomic_add(s_count_gt, 1)
                        if pos < top_k:
                            out_idx_row[pos] = cutlass.Int32(idx)
                            if cutlass.const_expr(self.has_values):
                                out_val_row[pos] = self.value_from_bits(bits)
        if cutlass.const_expr(mode == _OP_STAGE_MC):
            # Multi-CTA stage: winners through g_out (continues the group gt
            # counter), staged candidates through g_eqf into the gmem tie
            # buffer.  Only overflow rows pay these per-element atomics.
            b = self.coarse_bin(bits)
            if b == threshold_bin:
                k = self.exact_key(bits)
                pm = k & prefix_mask
                if pm == prefix:
                    c = gmem_atomic_add(g_state + self.g_eqf, 1)
                    if c < TIE_CAP:
                        (g_state + (self.g_tiek + c)).store(k.bitcast(cutlass.Int32))
                        (g_state + (self.g_tiei + c)).store(cutlass.Int32(idx))
                else:
                    if pm > prefix:
                        pos = gmem_atomic_add(g_state + self.g_out, 1)
                        if pos < top_k:
                            out_idx_row[pos] = cutlass.Int32(idx)
                            if cutlass.const_expr(self.has_values):
                                out_val_row[pos] = self.value_from_bits(bits)
        if cutlass.const_expr(mode == _OP_EQFILL):
            # Big-k (top_k > TIE_CAP) terminal when even the remaining fill
            # exceeds the tie stage: survivors of ALL refinement rounds are
            # provably key-identical, so prefix-equal candidates fill the
            # output tail directly in first-arrival order (any subset is a
            # valid exact answer).  Winners (prefix-above) continue through
            # the running gt counter exactly like _OP_STAGE.  Caller zeroes
            # the s_misc[6] fill cursor first; eq_base = top_k - remaining.
            b = self.coarse_bin(bits)
            if b == threshold_bin:
                k = self.exact_key(bits)
                pm = k & prefix_mask
                if pm == prefix:
                    e = smem_atomic_add(s_misc + 6, 1)
                    if eq_base + e < top_k:
                        out_idx_row[eq_base + e] = cutlass.Int32(idx)
                        if cutlass.const_expr(self.has_values):
                            out_val_row[eq_base + e] = self.value_from_bits(bits)
                else:
                    if pm > prefix:
                        pos = smem_atomic_add(s_count_gt, 1)
                        if pos < top_k:
                            out_idx_row[pos] = cutlass.Int32(idx)
                            if cutlass.const_expr(self.has_values):
                                out_val_row[pos] = self.value_from_bits(bits)
        if cutlass.const_expr(mode == _OP_EQFILL_MC):
            # Multi-CTA variant of _OP_EQFILL: the fill cursor is the shared
            # g_eqf counter (zeroed at grouped-row start) and winners go
            # through the group g_out counter, mirroring _OP_STAGE_MC.
            b = self.coarse_bin(bits)
            if b == threshold_bin:
                k = self.exact_key(bits)
                pm = k & prefix_mask
                if pm == prefix:
                    e = gmem_atomic_add(g_state + self.g_eqf, 1)
                    if eq_base + e < top_k:
                        out_idx_row[eq_base + e] = cutlass.Int32(idx)
                        if cutlass.const_expr(self.has_values):
                            out_val_row[eq_base + e] = self.value_from_bits(bits)
                else:
                    if pm > prefix:
                        pos = gmem_atomic_add(g_state + self.g_out, 1)
                        if pos < top_k:
                            out_idx_row[pos] = cutlass.Int32(idx)
                            if cutlass.const_expr(self.has_values):
                                out_val_row[pos] = self.value_from_bits(bits)

    @cute.jit
    def load_scalar(self, row_ptr, idx):
        """Scalar element load -> raw pattern in a Uint32."""
        v = row_ptr[idx]
        if cutlass.const_expr(self.is_f32):
            return v.bitcast(cutlass.Uint32)
        else:
            return v.bitcast(cutlass.Uint16).to(cutlass.Uint32)

    @cute.jit
    def _stream_row(
        self,
        mode: cutlass.Constexpr,
        shift: cutlass.Constexpr,
        row_ptr,
        length,
        col_base,
        tidx,
        s_hist,
        s_count_gt,
        s_count_eq,
        s_tie_keys,
        s_tie_idx,
        s_misc,
        out_idx_row,
        out_val_row,
        threshold_bin,
        prefix,
        prefix_mask,
        pivot,
        eq_base,
        g_state,
        wa_batch: cutlass.Constexpr = True,
    ):
        """Apply ``_elem_op(mode, ...)`` to elements [0, length) at row_ptr.

        Emitted indices are ``col_base + local`` (col_base > 0 for multi-CTA
        chunks).  The main body issues one ld.global.v4.b32 per 16 bytes; a
        scalar prologue re-aligns unaligned starts and a scalar tail covers
        the remainder.  Consecutive threads read consecutive 16B chunks.

        wa_batch (warp_agg kernels only): selects the thread-local batched
        HIST/COLLECT body (_elem_vec_wa) vs the per-element one.  Batching
        wins on flood rows but is pure per-vector tax on mixed rows, so
        callers instantiate both and pick per row (HIST: bin-uniformity
        probe; COLLECT: threshold-bin density).
        """
        addr = row_ptr.toint()  # Int64 byte address (gmem)
        mis = cutlass.Int32(addr & cutlass.Int64(15))
        prologue = cutlass.Int32(0)
        if mis != 0:
            prologue = (16 - mis) // self.elem_bytes
        if prologue > length:
            prologue = length

        # --- scalar prologue ---
        for i in range(tidx, prologue, NUM_THREADS):
            self._elem_op(
                mode,
                shift,
                self.load_scalar(row_ptr, i),
                col_base + i,
                s_hist,
                s_count_gt,
                s_count_eq,
                s_tie_keys,
                s_tie_idx,
                s_misc,
                out_idx_row,
                out_val_row,
                threshold_bin,
                prefix,
                prefix_mask,
                pivot,
                eq_base,
                g_state,
            )

        remaining = length - prologue
        num_vec = remaining // self.vec_elems
        tail = remaining - num_vec * self.vec_elems
        vec_base_addr = addr + cutlass.Int64(prologue) * self.elem_bytes

        # --- vectorized main (double-buffered, 2x unrolled) ---
        # Two alternating register sets A (w*) / B (m*): B's 16B load issues
        # before A's smem work and A's NEXT load before B's work, so one
        # global load is always in flight per thread (sglang's for_each_input
        # idiom) WITHOUT ``cur = next`` register copies -- ptxas kept the
        # single-buffer rotation as ~8 IMAD.MOVs per iteration (~2 pure
        # shuffle instructions per element, visible in SASS).  Addresses
        # advance by a fixed stride off one base pointer instead of a
        # per-load IMAD.WIDE.
        stride16 = cutlass.Int64(NUM_THREADS * 16)
        w0 = cutlass.Uint32(0)
        w1 = cutlass.Uint32(0)
        w2 = cutlass.Uint32(0)
        w3 = cutlass.Uint32(0)
        m0 = cutlass.Uint32(0)
        m1 = cutlass.Uint32(0)
        m2 = cutlass.Uint32(0)
        m3 = cutlass.Uint32(0)
        v = cutlass.Int32(tidx)
        va = vec_base_addr + cutlass.Int64(tidx) * 16
        # thread-local warp_agg state (see _elem_vec_wa); dead (DCE'd) for
        # non-warp_agg kernels and modes other than HIST/COLLECT
        hb = cutlass.Int32(-1)
        hc = cutlass.Int32(0)
        sat = cutlass.Int32(0)
        acc = cutlass.Int32(0)
        if v < num_vec:
            w0, w1, w2, w3 = ld_global_v4_u32(va)
        while v < num_vec:
            v2 = v + NUM_THREADS
            if v2 < num_vec:
                m0, m1, m2, m3 = ld_global_v4_u32(va + stride16)
            base = col_base + prologue + v * self.vec_elems
            if cutlass.const_expr(
                self.warp_agg and wa_batch and mode in (_OP_HIST, _OP_COLLECT)
            ):
                # Ampere: thread-local batched vector op (see _elem_vec_wa)
                hb, hc, sat, acc = self._elem_vec_wa(
                    mode,
                    w0,
                    w1,
                    w2,
                    w3,
                    base,
                    hb,
                    hc,
                    sat,
                    acc,
                    s_hist,
                    s_count_gt,
                    s_count_eq,
                    s_tie_keys,
                    s_tie_idx,
                    s_misc,
                    out_idx_row,
                    out_val_row,
                    threshold_bin,
                    prefix,
                    prefix_mask,
                )
            elif cutlass.const_expr(self.is_f32):
                for j in cutlass.range_constexpr(4):
                    w = (w0, w1, w2, w3)[j]
                    self._elem_op(
                        mode,
                        shift,
                        w,
                        base + j,
                        s_hist,
                        s_count_gt,
                        s_count_eq,
                        s_tie_keys,
                        s_tie_idx,
                        s_misc,
                        out_idx_row,
                        out_val_row,
                        threshold_bin,
                        prefix,
                        prefix_mask,
                        pivot,
                        eq_base,
                        g_state,
                    )
            else:
                for j in cutlass.range_constexpr(4):
                    w = (w0, w1, w2, w3)[j]
                    # little-endian: low half = lower column index
                    self._elem_op(
                        mode,
                        shift,
                        w & cutlass.Uint32(0xFFFF),
                        base + 2 * j,
                        s_hist,
                        s_count_gt,
                        s_count_eq,
                        s_tie_keys,
                        s_tie_idx,
                        s_misc,
                        out_idx_row,
                        out_val_row,
                        threshold_bin,
                        prefix,
                        prefix_mask,
                        pivot,
                        eq_base,
                        g_state,
                    )
                    self._elem_op(
                        mode,
                        shift,
                        w >> cutlass.Uint32(16),
                        base + 2 * j + 1,
                        s_hist,
                        s_count_gt,
                        s_count_eq,
                        s_tie_keys,
                        s_tie_idx,
                        s_misc,
                        out_idx_row,
                        out_val_row,
                        threshold_bin,
                        prefix,
                        prefix_mask,
                        pivot,
                        eq_base,
                        g_state,
                    )
            va = va + stride16 + stride16
            v = v2 + NUM_THREADS
            if v < num_vec:
                w0, w1, w2, w3 = ld_global_v4_u32(va)
            if v2 < num_vec:
                base2 = col_base + prologue + v2 * self.vec_elems
                if cutlass.const_expr(
                    self.warp_agg and wa_batch and mode in (_OP_HIST, _OP_COLLECT)
                ):
                    hb, hc, sat, acc = self._elem_vec_wa(
                        mode,
                        m0,
                        m1,
                        m2,
                        m3,
                        base2,
                        hb,
                        hc,
                        sat,
                        acc,
                        s_hist,
                        s_count_gt,
                        s_count_eq,
                        s_tie_keys,
                        s_tie_idx,
                        s_misc,
                        out_idx_row,
                        out_val_row,
                        threshold_bin,
                        prefix,
                        prefix_mask,
                    )
                elif cutlass.const_expr(self.is_f32):
                    for j in cutlass.range_constexpr(4):
                        w = (m0, m1, m2, m3)[j]
                        self._elem_op(
                            mode,
                            shift,
                            w,
                            base2 + j,
                            s_hist,
                            s_count_gt,
                            s_count_eq,
                            s_tie_keys,
                            s_tie_idx,
                            s_misc,
                            out_idx_row,
                            out_val_row,
                            threshold_bin,
                            prefix,
                            prefix_mask,
                            pivot,
                            eq_base,
                            g_state,
                        )
                else:
                    for j in cutlass.range_constexpr(4):
                        w = (m0, m1, m2, m3)[j]
                        self._elem_op(
                            mode,
                            shift,
                            w & cutlass.Uint32(0xFFFF),
                            base2 + 2 * j,
                            s_hist,
                            s_count_gt,
                            s_count_eq,
                            s_tie_keys,
                            s_tie_idx,
                            s_misc,
                            out_idx_row,
                            out_val_row,
                            threshold_bin,
                            prefix,
                            prefix_mask,
                            pivot,
                            eq_base,
                            g_state,
                        )
                        self._elem_op(
                            mode,
                            shift,
                            w >> cutlass.Uint32(16),
                            base2 + 2 * j + 1,
                            s_hist,
                            s_count_gt,
                            s_count_eq,
                            s_tie_keys,
                            s_tie_idx,
                            s_misc,
                            out_idx_row,
                            out_val_row,
                            threshold_bin,
                            prefix,
                            prefix_mask,
                            pivot,
                            eq_base,
                            g_state,
                        )

        # --- flush the warp_agg thread-local state (see _elem_vec_wa) ---
        if cutlass.const_expr(self.warp_agg and mode == _OP_HIST):
            if hc > 0:
                smem_atomic_add(s_hist + hb, hc)
        if cutlass.const_expr(self.warp_agg and mode == _OP_COLLECT):
            if acc > 0:
                smem_atomic_add(s_count_eq, acc)

        # --- scalar tail ---
        tail_base = prologue + num_vec * self.vec_elems
        for i in range(tidx, tail, NUM_THREADS):
            self._elem_op(
                mode,
                shift,
                self.load_scalar(row_ptr, tail_base + i),
                col_base + tail_base + i,
                s_hist,
                s_count_gt,
                s_count_eq,
                s_tie_keys,
                s_tie_idx,
                s_misc,
                out_idx_row,
                out_val_row,
                threshold_bin,
                prefix,
                prefix_mask,
                pivot,
                eq_base,
                g_state,
            )

    # ------------------------------------------------------------------
    # Register path: the whole row / chunk lives in per-thread registers
    # ------------------------------------------------------------------
    def _reg_path_ok(self, n_cols: int, span: int, max_slots: int = 2) -> bool:
        """Compile-time: every span start is 16B aligned (rows 16B aligned
        AND spans a multiple of the vector) and the span fits in
        ``max_slots`` vector slots per thread (2: bf16/fp16 <= 16384 elems,
        fp32 <= 8192; 4: twice that)."""
        return (
            ((n_cols * self.elem_bytes) % 16 == 0)
            and (span % self.vec_elems == 0 or span == n_cols)
            and (span <= max_slots * NUM_THREADS * self.vec_elems)
        )

    def _reg_two_slots(self, span: int) -> bool:
        return span > NUM_THREADS * self.vec_elems

    def _reg_num_slots(self, span: int) -> int:
        """Vector slots per thread that hold a ``span``-element row (1..4)."""
        return -(-span // (NUM_THREADS * self.vec_elems))

    @cute.jit
    def _reg_row(
        self,
        mode: cutlass.Constexpr,
        shift: cutlass.Constexpr,
        two_slots: cutlass.Constexpr,
        col_base,
        valid0,
        a0,
        a1,
        a2,
        a3,
        valid1,
        b0,
        b1,
        b2,
        b3,
        tail_thread,
        tbits,
        tail_idx,
        tidx,
        s_hist,
        s_count_gt,
        s_count_eq,
        s_tie_keys,
        s_tie_idx,
        s_misc,
        out_idx_row,
        out_val_row,
        threshold_bin,
        prefix,
        prefix_mask,
        pivot,
        eq_base,
        g_state,
    ):
        """Two-slot form of :meth:`_reg_row_n` (sglang TopKRegister<2>: thread
        t owns full vectors t and t + NUM_THREADS), kept for the multi-CTA
        chunk paths.  ``tail_idx`` is already col_base-relative-global."""
        self._reg_row_n(
            mode,
            shift,
            2 if two_slots else 1,
            col_base,
            [valid0, valid1],
            [a0, a1, a2, a3, b0, b1, b2, b3],
            tail_thread,
            tbits,
            tail_idx,
            tidx,
            s_hist,
            s_count_gt,
            s_count_eq,
            s_tie_keys,
            s_tie_idx,
            s_misc,
            out_idx_row,
            out_val_row,
            threshold_bin,
            prefix,
            prefix_mask,
            pivot,
            eq_base,
            g_state,
        )

    @cute.jit
    def _reg_row_n(
        self,
        mode: cutlass.Constexpr,
        shift: cutlass.Constexpr,
        nslots: cutlass.Constexpr,
        col_base,
        valids,
        words,
        tail_thread,
        tbits,
        tail_idx,
        tidx,
        s_hist,
        s_count_gt,
        s_count_eq,
        s_tie_keys,
        s_tie_idx,
        s_misc,
        out_idx_row,
        out_val_row,
        threshold_bin,
        prefix,
        prefix_mask,
        pivot,
        eq_base,
        g_state,
    ):
        """Apply ``_elem_op(mode, ...)`` to a row/chunk held in registers.

        Slot layout (sglang TopKRegister<nslots>): thread t owns full vectors
        t + s * NUM_THREADS for s < nslots, each guarded by its own validity
        flag ``valids[s]`` (the real-length bound: a dead slot costs one
        branch, none of its element code -- predicating the unrolled slot
        code off instead measured +35% warp-instructions on 1.5K rows);
        ``words[4 * s + j]`` is word j of slot s.  The < vec_elems tail is
        one scalar on each of the LAST ``tail`` threads.  Data was read from
        gmem exactly once.  ``tail_idx`` is already col_base-relative-global.
        """
        ve = cutlass.const_expr(self.vec_elems)
        for s in cutlass.range_constexpr(nslots):
            if valids[s]:
                base = col_base + (tidx + s * NUM_THREADS) * ve
                if cutlass.const_expr(self.is_f32):
                    for j in cutlass.range_constexpr(4):
                        self._elem_op(
                            mode,
                            shift,
                            words[4 * s + j],
                            base + j,
                            s_hist,
                            s_count_gt,
                            s_count_eq,
                            s_tie_keys,
                            s_tie_idx,
                            s_misc,
                            out_idx_row,
                            out_val_row,
                            threshold_bin,
                            prefix,
                            prefix_mask,
                            pivot,
                            eq_base,
                            g_state,
                        )
                else:
                    for j in cutlass.range_constexpr(4):
                        w = words[4 * s + j]
                        self._elem_op(
                            mode,
                            shift,
                            w & cutlass.Uint32(0xFFFF),
                            base + 2 * j,
                            s_hist,
                            s_count_gt,
                            s_count_eq,
                            s_tie_keys,
                            s_tie_idx,
                            s_misc,
                            out_idx_row,
                            out_val_row,
                            threshold_bin,
                            prefix,
                            prefix_mask,
                            pivot,
                            eq_base,
                            g_state,
                        )
                        self._elem_op(
                            mode,
                            shift,
                            w >> cutlass.Uint32(16),
                            base + 2 * j + 1,
                            s_hist,
                            s_count_gt,
                            s_count_eq,
                            s_tie_keys,
                            s_tie_idx,
                            s_misc,
                            out_idx_row,
                            out_val_row,
                            threshold_bin,
                            prefix,
                            prefix_mask,
                            pivot,
                            eq_base,
                            g_state,
                        )
        if tail_thread:
            self._elem_op(
                mode,
                shift,
                tbits,
                tail_idx,
                s_hist,
                s_count_gt,
                s_count_eq,
                s_tie_keys,
                s_tie_idx,
                s_misc,
                out_idx_row,
                out_val_row,
                threshold_bin,
                prefix,
                prefix_mask,
                pivot,
                eq_base,
                g_state,
            )

    @cute.jit
    def _reg_solve_row(
        self,
        nslots: cutlass.Constexpr,
        row_in,
        length,
        tidx,
        s_hist,
        s_count_gt,
        s_count_eq,
        s_tie_keys,
        s_tie_idx,
        s_warp_sums,
        s_misc,
        out_idx_row,
        out_val_row,
    ):
        """Register-resident solve of one row of at most nslots * NUM_THREADS
        * vec_elems elements (sglang TopKRegister<nslots>): ONE gmem read of
        the row, bounded by the REAL length at every slot (a 1K-column row in
        a 16K buffer issues one vector per thread on 250 threads and skips
        the other slots' code), then the coarse histogram and the classify +
        collect from registers with the crossing bin from
        find_threshold_coarse.  Leaves s_count_gt[0] / s_count_eq[0], the
        staged ties, s_misc[0] (threshold bin) and s_misc[8..9] (tie-key
        range) for the caller's tie phase.  Expects s_hist and the counters
        zeroed and fenced (the caller's init barrier); ends with a barrier.
        Shared by the single-CTA kernel and the multi-CTA kernel's solo arm."""
        top_k = cutlass.const_expr(self.top_k)
        ve = cutlass.const_expr(self.vec_elems)
        u0 = cutlass.Uint32(0)
        i0 = cutlass.Int32(0)
        addr = row_in.toint()
        num_full = length // ve
        tail = length - num_full * ve

        valids: list = []  # per-slot validity (trace-time list of DSL bools)
        words: list = []  # 4 words per slot
        for s in cutlass.range_constexpr(nslots):
            vs = (tidx + s * NUM_THREADS) < num_full
            w0 = cutlass.Uint32(0)
            w1 = cutlass.Uint32(0)
            w2 = cutlass.Uint32(0)
            w3 = cutlass.Uint32(0)
            if vs:
                w0, w1, w2, w3 = ld_global_v4_u32(
                    addr + cutlass.Int64(tidx + s * NUM_THREADS) * 16
                )
            valids.append(vs)
            words.extend((w0, w1, w2, w3))
        # Tail (< vec_elems elems): one scalar on each of the LAST ``tail``
        # threads.
        tail_thread = tidx >= NUM_THREADS - tail
        tail_idx = num_full * ve + (tidx - (NUM_THREADS - tail))
        tbits = cutlass.Uint32(0)
        if tail_thread:
            tbits = self.load_scalar(row_in, tail_idx)

        # ---- pass 1: coarse histogram (from registers) ----
        self._reg_row_n(
            _OP_HIST,
            0,
            nslots,
            i0,
            valids,
            words,
            tail_thread,
            tbits,
            tail_idx,
            tidx,
            s_hist,
            s_count_gt,
            s_count_eq,
            s_tie_keys,
            s_tie_idx,
            s_misc,
            out_idx_row,
            out_val_row,
            i0,
            u0,
            u0,
            u0,
            i0,
            s_misc,
        )
        cute.arch.barrier()

        # ---- threshold bin ----
        self.find_threshold_coarse(
            s_hist, length, cutlass.Int32(top_k), s_warp_sums, s_misc, tidx
        )
        threshold_bin = s_misc[0]
        hi_b = u0
        lo_b = u0
        if cutlass.const_expr(self.boundary_cls):
            hi_b = self.coarse_bin_gt_threshold_f32(threshold_bin + 1).bitcast(
                cutlass.Uint32
            )
            lo_b = self.coarse_bin_lower_bound_f32(threshold_bin).bitcast(
                cutlass.Uint32
            )
        if cutlass.const_expr(self._fuse_r1):
            # reset s_hist for the collect-fused (12,12) refine histogram
            # (the coarse counts are consumed by now)
            s_ha = s_hist.toint()
            for i in range(tidx * 4, self.hist_size, NUM_THREADS * 4):
                st_shared_v4_zero(s_ha + i * 4)
            cute.arch.barrier()

        # ---- pass 2: classify + collect (from registers) ----
        self._reg_row_n(
            _OP_COLLECT,
            0,
            nslots,
            i0,
            valids,
            words,
            tail_thread,
            tbits,
            tail_idx,
            tidx,
            s_hist,
            s_count_gt,
            s_count_eq,
            s_tie_keys,
            s_tie_idx,
            s_misc,
            out_idx_row,
            out_val_row,
            threshold_bin,
            hi_b,
            lo_b,
            u0,
            i0,
            s_misc,
        )
        cute.arch.barrier()

    @cute.jit
    def _collect_scan_row(
        self,
        row_ptr,
        length,
        tidx,
        s_tie_keys,
        s_tie_idx,
        s_warp_sums,
        s_count_gt,
        s_count_eq,
        s_misc,
        s_hist,
        out_idx_row,
        out_val_row,
        threshold_bin,
        hi_b,
        lo_b,
    ):
        """Scan-based collect for one CTA-local row (16B-aligned rows only).

        Phase attribution at b=1 k=2048 showed the classic collect paying
        ~1.45us in serialized same-address counter atomics plus ~1.7us in
        per-hit dependent stores (each store waits on its atomic's returned
        position) -- with only a few loop iterations there is nothing to
        hide those chains behind.  This walker removes both: classification
        sets bits in two per-thread 32-bit masks (my k-th vector's element j
        = bit k*ve + j; up to 64 elements/thread covers n_cols <= 65536, and
        every solo row), ONE block exclusive scan of popcounts assigns each
        thread its output range, and hits are emitted with plain stores and
        no atomics.  gt output becomes index-ordered (deterministic).  Ties
        are few and keep the staged-atomic path.  The caller must barrier
        afterwards before reading s_count_gt[0] (written by the scan) or
        reusing s_warp_sums.
        """
        ve = cutlass.const_expr(self.vec_elems)
        addr = row_ptr.toint()
        num_vec = length // ve
        tail = length - num_vec * ve

        m0 = cutlass.Uint32(0)
        m1 = cutlass.Uint32(0)
        w0 = cutlass.Uint32(0)
        w1 = cutlass.Uint32(0)
        w2 = cutlass.Uint32(0)
        w3 = cutlass.Uint32(0)
        n0 = cutlass.Uint32(0)
        n1 = cutlass.Uint32(0)
        n2 = cutlass.Uint32(0)
        n3 = cutlass.Uint32(0)
        stride16 = cutlass.Int64(NUM_THREADS * 16)
        v = cutlass.Int32(tidx)
        kord = cutlass.Int32(0)  # my vector ordinal (bit group)
        va = addr + cutlass.Int64(tidx) * 16
        if v < num_vec:
            w0, w1, w2, w3 = ld_global_v4_u32(va)
        while v < num_vec:
            v2 = v + NUM_THREADS
            if v2 < num_vec:
                n0, n1, n2, n3 = ld_global_v4_u32(va + stride16)
            for j in cutlass.range_constexpr(ve):
                if cutlass.const_expr(self.is_f32):
                    bits_e = (w0, w1, w2, w3)[j]
                else:
                    ww = (w0, w1, w2, w3)[j // 2]
                    if cutlass.const_expr(j % 2 == 0):
                        bits_e = ww & cutlass.Uint32(0xFFFF)
                    else:
                        bits_e = ww >> cutlass.Uint32(16)
                is_gt = cutlass.Int32(0)
                is_tie = cutlass.Int32(0)
                if cutlass.const_expr(self.boundary_cls):
                    # inverted branch: NaN fails both ordered compares and
                    # classifies gt (see coarse_bin_gt_threshold_f32)
                    val = bits_e.bitcast(cutlass.Float32)
                    if val <= hi_b.bitcast(cutlass.Float32):
                        if val >= lo_b.bitcast(cutlass.Float32):
                            is_tie = cutlass.Int32(1)
                    else:
                        is_gt = cutlass.Int32(1)
                else:
                    b = self.coarse_bin(bits_e)
                    if b > threshold_bin:
                        is_gt = cutlass.Int32(1)
                    else:
                        if b == threshold_bin:
                            is_tie = cutlass.Int32(1)
                if is_gt == 1:
                    bitpos = kord * ve + j
                    if bitpos < 32:
                        m0 = m0 | (cutlass.Uint32(1) << cutlass.Uint32(bitpos))
                    else:
                        m1 = m1 | (cutlass.Uint32(1) << cutlass.Uint32(bitpos - 32))
                if is_tie == 1:
                    kt = self.exact_key(bits_e)
                    self._track_tie_range(s_misc, kt)
                    if cutlass.const_expr(self._fuse_r1):
                        smem_atomic_add(
                            s_hist
                            + cutlass.Int32(
                                (kt >> cutlass.Uint32(12)) & cutlass.Uint32(0xFFF)
                            ),
                            1,
                        )
                    c = smem_atomic_add(s_count_eq, 1)
                    if c < TIE_CAP:
                        s_tie_keys[c] = kt
                        s_tie_idx[c] = cutlass.Int32(v * ve + j)
            kord = kord + 1
            va = va + stride16
            v = v2
            w0 = n0
            w1 = n1
            w2 = n2
            w3 = n3

        # my (at most one) tail element: separate flag, not a mask bit
        tail_hit = cutlass.Int32(0)
        tail_idx = num_vec * ve + (tidx - (NUM_THREADS - tail))
        if tidx >= NUM_THREADS - tail:
            tbits = self.load_scalar(row_ptr, tail_idx)
            if cutlass.const_expr(self.boundary_cls):
                tval = tbits.bitcast(cutlass.Float32)
                if tval <= hi_b.bitcast(cutlass.Float32):
                    if tval >= lo_b.bitcast(cutlass.Float32):
                        ktt = self.exact_key(tbits)
                        self._track_tie_range(s_misc, ktt)
                        if cutlass.const_expr(self._fuse_r1):
                            smem_atomic_add(
                                s_hist
                                + cutlass.Int32(
                                    (ktt >> cutlass.Uint32(12)) & cutlass.Uint32(0xFFF)
                                ),
                                1,
                            )
                        ct = smem_atomic_add(s_count_eq, 1)
                        if ct < TIE_CAP:
                            s_tie_keys[ct] = ktt
                            s_tie_idx[ct] = cutlass.Int32(tail_idx)
                else:  # bin > tb, or NaN
                    tail_hit = cutlass.Int32(1)
            else:
                tb = self.coarse_bin(tbits)
                if tb > threshold_bin:
                    tail_hit = cutlass.Int32(1)
                else:
                    if tb == threshold_bin:
                        ktt = self.exact_key(tbits)
                        self._track_tie_range(s_misc, ktt)
                        if cutlass.const_expr(self._fuse_r1):
                            smem_atomic_add(
                                s_hist
                                + cutlass.Int32(
                                    (ktt >> cutlass.Uint32(12)) & cutlass.Uint32(0xFFF)
                                ),
                                1,
                            )
                        ct = smem_atomic_add(s_count_eq, 1)
                        if ct < TIE_CAP:
                            s_tie_keys[ct] = ktt
                            s_tie_idx[ct] = cutlass.Int32(tail_idx)

        cnt = cute.arch.popc(m0) + cute.arch.popc(m1) + tail_hit
        base = self._block_excl_scan(cnt, s_warp_sums, s_count_gt, tidx)

        # emit my hits at [base, base + cnt): plain stores, no atomics.
        # Positional stores have no atomic cap, so a gt over-count (any
        # histogram/boundary classification disagreement) must never spill
        # into the NEXT row's output slots.  Under the threshold invariant
        # every thread satisfies base + cnt <= top_k and takes the unguarded
        # hot loop; only an overflowing thread pays the per-hit range check.
        if base + cnt <= cutlass.const_expr(self.top_k):
            r = cutlass.Int32(base)
            vv = cutlass.Int32(tidx)
            kk = cutlass.Int32(0)
            while vv < num_vec:
                bitbase = kk * ve
                mk = cutlass.Uint32(0)
                if bitbase < 32:
                    mk = (m0 >> cutlass.Uint32(bitbase)) & cutlass.Uint32((1 << ve) - 1)
                else:
                    mk = (m1 >> cutlass.Uint32(bitbase - 32)) & cutlass.Uint32(
                        (1 << ve) - 1
                    )
                if mk != 0:
                    ebase = vv * ve
                    for j in cutlass.range_constexpr(ve):
                        if ((mk >> cutlass.Uint32(j)) & cutlass.Uint32(1)) != 0:
                            out_idx_row[r] = cutlass.Int32(ebase + j)
                            if cutlass.const_expr(self.has_values):
                                out_val_row[r] = self.value_from_bits(
                                    self.load_scalar(row_ptr, ebase + j)
                                )
                            r = r + 1
                vv = vv + NUM_THREADS
                kk = kk + 1
            if tail_hit == 1:
                out_idx_row[r] = cutlass.Int32(tail_idx)
                if cutlass.const_expr(self.has_values):
                    out_val_row[r] = self.value_from_bits(
                        self.load_scalar(row_ptr, tail_idx)
                    )
        else:
            r = cutlass.Int32(base)
            vv = cutlass.Int32(tidx)
            kk = cutlass.Int32(0)
            while vv < num_vec:
                bitbase = kk * ve
                mk = cutlass.Uint32(0)
                if bitbase < 32:
                    mk = (m0 >> cutlass.Uint32(bitbase)) & cutlass.Uint32((1 << ve) - 1)
                else:
                    mk = (m1 >> cutlass.Uint32(bitbase - 32)) & cutlass.Uint32(
                        (1 << ve) - 1
                    )
                if mk != 0:
                    ebase = vv * ve
                    for j in cutlass.range_constexpr(ve):
                        if ((mk >> cutlass.Uint32(j)) & cutlass.Uint32(1)) != 0:
                            if r < cutlass.const_expr(self.top_k):
                                out_idx_row[r] = cutlass.Int32(ebase + j)
                                if cutlass.const_expr(self.has_values):
                                    out_val_row[r] = self.value_from_bits(
                                        self.load_scalar(row_ptr, ebase + j)
                                    )
                            r = r + 1
                vv = vv + NUM_THREADS
                kk = kk + 1
            if tail_hit == 1:
                if r < cutlass.const_expr(self.top_k):
                    out_idx_row[r] = cutlass.Int32(tail_idx)
                    if cutlass.const_expr(self.has_values):
                        out_val_row[r] = self.value_from_bits(
                            self.load_scalar(row_ptr, tail_idx)
                        )

    # ------------------------------------------------------------------
    # Threshold search over the coarse histogram
    # ------------------------------------------------------------------
    @cute.jit
    def find_threshold_coarse(self, s_hist, total, needed, s_warp_sums, s_misc, tidx):
        """Block-wide scan of the hist_size-bin histogram.

        Thread tx owns bins [tx*items, (tx+1)*items).  Warp-level inclusive
        scan of the per-thread sums, then each warp adds the total of the
        preceding warps (one masked butterfly redux per thread -- no second
        scan needed).  The unique bin with ``above < needed <= above+count``
        publishes itself:  s_misc[0] = threshold bin, s_misc[1] = above.
        ``above`` = elements in strictly greater bins
                  = total - inclusive_prefix(bin).
        All 1024 threads participate (hist_size >= NUM_THREADS).
        """
        items = cutlass.const_expr(self.hist_items)
        lane_id = tidx % 32
        warp_id = tidx // 32

        # Vectorized bin reads (ld.shared.v4), values kept in registers so
        # the threshold walk below re-uses them instead of re-loading.
        base_addr = s_hist.toint() + tidx * (items * 4)
        vals = []
        for c in cutlass.range_constexpr(items // 4):
            v0, v1, v2, v3 = ld_shared_v4_s32(base_addr + c * 16)
            vals += [v0, v1, v2, v3]
        local = cutlass.Int32(0)
        for i in cutlass.range_constexpr(items):
            local = local + vals[i]

        incl = warp_inclusive_sum(local, lane_id)
        if lane_id == 31:
            s_warp_sums[warp_id] = incl
        cute.arch.barrier()

        # Sum of preceding warps' totals: lane l contributes warp_sums[l]
        # when l < warp_id; the butterfly reduce broadcasts the sum to all
        # lanes of the warp.
        w = cutlass.Int32(0)
        if lane_id < warp_id:
            w = s_warp_sums[lane_id]
        prev_warps = warp_sum(w)

        # Walk my bins with a running inclusive prefix.
        prefix = prev_warps + (incl - local)
        for i in cutlass.range_constexpr(items):
            cnt = vals[i]
            prefix = prefix + cnt
            above = total - prefix
            if above < needed and above + cnt >= needed:
                s_misc[0] = tidx * items + i
                s_misc[1] = above
        cute.arch.barrier()

    @cute.jit
    def _terminal_stage_select(
        self,
        row_in,
        length,
        i0,
        u0,
        tidx,
        s_hist,
        s_count_gt,
        s_count_eq,
        s_tie_keys,
        s_tie_idx,
        s_warp_sums,
        s_misc,
        out_idx_row,
        out_val_row,
        threshold_bin,
        prefix,
        prefix_mask,
    ):
        """Terminal masked STAGE pass + exact in-smem tie_select (overflow
        tail of the single-CTA and solo paths).  Requires the row's
        remaining fill <= TIE_CAP -- guaranteed when top_k <= TIE_CAP and
        gated by the caller (EQFILL arm) otherwise."""
        top_k = cutlass.const_expr(self.top_k)
        if tidx == 0:
            s_count_eq[0] = cutlass.Int32(0)
        cute.arch.barrier()
        self._stream_row(
            _OP_STAGE,
            0,
            row_in,
            length,
            i0,
            tidx,
            s_hist,
            s_count_gt,
            s_count_eq,
            s_tie_keys,
            s_tie_idx,
            s_misc,
            out_idx_row,
            out_val_row,
            threshold_bin,
            prefix,
            prefix_mask,
            u0,
            i0,
            s_misc,
        )
        cute.arch.barrier()
        gt_total = s_count_gt[0]
        scnt = s_count_eq[0]
        if scnt > TIE_CAP:
            # only when survivors are key-identical: any staged subset is a
            # valid answer
            scnt = cutlass.Int32(TIE_CAP)
        rem2 = top_k - gt_total
        if rem2 < 0:
            rem2 = cutlass.Int32(0)
        self.tie_select(
            s_tie_keys,
            s_tie_idx,
            scnt,
            gt_total,
            rem2,
            s_hist,
            s_warp_sums,
            s_misc,
            out_idx_row,
            out_val_row,
            tidx,
        )

    @cute.jit
    def _mc_gather_select(
        self,
        g_state,
        tidx,
        s_hist,
        s_tie_keys,
        s_tie_idx,
        s_warp_sums,
        s_misc,
        out_idx_row,
        out_val_row,
    ):
        """Rank-0 resolution of the MC overflow terminal: gather the gmem
        tie stage into smem and run the exact tie_select (mirrors the
        eq_count <= TIE_CAP fast arm).  Requires remaining fill <= TIE_CAP
        -- guaranteed when top_k <= TIE_CAP and gated by the caller
        (EQFILL_MC arm) otherwise."""
        top_k = cutlass.const_expr(self.top_k)
        gt_total = (g_state + self.g_out).load()
        scnt = (g_state + self.g_eqf).load()
        if scnt > TIE_CAP:
            scnt = cutlass.Int32(TIE_CAP)
        rem2 = top_k - gt_total
        if rem2 < 0:
            rem2 = cutlass.Int32(0)
        for t in range(tidx, scnt, NUM_THREADS):
            s_tie_keys[t] = (g_state + (self.g_tiek + t)).load().bitcast(cutlass.Uint32)
            s_tie_idx[t] = (g_state + (self.g_tiei + t)).load()
        cute.arch.barrier()
        self.tie_select(
            s_tie_keys,
            s_tie_idx,
            scnt,
            gt_total,
            rem2,
            s_hist,
            s_warp_sums,
            s_misc,
            out_idx_row,
            out_val_row,
            tidx,
        )

    @cute.jit
    def find_threshold_wide(self, s_hist, total, needed, s_warp_sums, s_misc, tidx):
        """Threshold search over the hist_size-bin histogram with SCALAR bin
        loads; used by the wide refinement rounds.

        find_threshold_coarse's ld.shared.v4 register-caching returns stale
        bins at the refinement call sites (empirically bisected: identical
        flow passes with scalar loads and fails with the v4 variant -- the
        raw inline-asm loads appear to be hoisted across the round's
        histogram build inside the deeply nested refinement region).  The
        coarse call site keeps the vectorized version (proven there); the
        refinement runs only on tie-overflow rows, where two extra scalar
        smem reads per bin are noise.  ``total`` is advisory: the walk uses
        the histogram's OWN sum (and clamps ``needed`` to it), so a
        classified-count vs histogram-membership mismatch (e.g. -0.0
        boundary-classified as a tie of the +0.0 bin) degrades to selecting
        among the histogram members instead of publishing nothing and
        leaving stale s_misc behind.  Publishes s_misc[0] = bin,
        s_misc[1] = above, s_misc[2] = count."""
        items = cutlass.const_expr(self.hist_items)
        lane_id = tidx % 32
        warp_id = tidx // 32
        local = cutlass.Int32(0)
        for i in cutlass.range_constexpr(items):
            local = local + s_hist[tidx * items + i]
        incl = warp_inclusive_sum(local, lane_id)
        if lane_id == 31:
            s_warp_sums[warp_id] = incl
        cute.arch.barrier()
        w = cutlass.Int32(0)
        if lane_id < warp_id:
            w = s_warp_sums[lane_id]
        prev_warps = warp_sum(w)
        if cutlass.const_expr(self.nw == 32):
            hsum = warp_sum(s_warp_sums[lane_id])  # nw == 32 == lanes
        else:
            sw = cutlass.Int32(0)
            if lane_id < self.nw:
                sw = s_warp_sums[lane_id]
            hsum = warp_sum(sw)
        ne = needed
        if ne > hsum:
            ne = hsum
        prefix = prev_warps + (incl - local)
        for i in cutlass.range_constexpr(items):
            cnt = s_hist[tidx * items + i]
            prefix = prefix + cnt
            above = hsum - prefix
            if above < ne and above + cnt >= ne:
                s_misc[0] = tidx * items + i
                s_misc[1] = above
                s_misc[2] = cnt
        cute.arch.barrier()

    @cute.jit
    def scan256_and_find(self, s_hist256, total, needed, s_warp_sums, s_misc, tidx):
        """Threshold search over a 256-bin histogram (one bin per thread).

        Only threads 0..255 participate; they synchronize among themselves
        with a named barrier (bar.sync 1, 256) so all intermediate values
        stay in scope (the block_scan.py idiom).  Publishes
        s_misc[0] = bucket, s_misc[1] = above, s_misc[2] = count.
        Ends with a full-block barrier.
        """
        lane_id = tidx % 32
        warp_id = tidx // 32
        if tidx < 256:
            cnt = s_hist256[tidx]
            incl = warp_inclusive_sum(cnt, lane_id)
            if lane_id == 31:
                s_warp_sums[warp_id] = incl
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
            w = cutlass.Int32(0)
            if lane_id < 8:  # 256 bins = 8 warps
                if lane_id < warp_id:
                    w = s_warp_sums[lane_id]
            incl = incl + warp_sum(w)
            above = total - incl
            if above < needed and above + cnt >= needed:
                s_misc[0] = tidx
                s_misc[1] = above
                s_misc[2] = cnt
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # Exact tie select over staged smem candidates (sglang handle_tie)
    # ------------------------------------------------------------------
    @cute.jit
    def tie_select(
        self,
        s_tie_keys,
        s_tie_idx,
        num_ties,
        out_base,
        remaining,
        s_hist,
        s_warp_sums,
        s_misc,
        out_idx_row,
        out_val_row,
        tidx,
    ):
        """Emit the top ``remaining`` of ``num_ties`` staged candidates.

        Small tie sets take sglang's warp-ballot rank paths (the common case:
        with fine coarse bins the threshold bin holds tens of candidates):
        candidate c's output rank = number of candidates strictly greater
        than c, counted with one ballot+popc per (warp, candidate) pair.
        "Greater" breaks exact-key ties by ascending index, a total order, so
        ranks are unique and the result deterministic.  Large sets fall back
        to the block-wide byte-radix select.
        """
        lane = tidx % 32
        warp = tidx // 32
        if num_ties <= 32:
            # Warp w ranks candidate w; its 32 lanes hold all candidates.
            ck = cutlass.Uint32(0)  # sentinel: loses every comparison
            ci = cutlass.Int32(0x7FFFFFFF)
            if lane < num_ties:
                ck = cutlass.Uint32(s_tie_keys[lane])
                ci = cutlass.Int32(s_tie_idx[lane])
            # target coverage must span 32 candidates regardless of the
            # warp count: at nt=512 (16 warps) each warp ranks two
            # (folds to the original single-target form at nw=32)
            for w_ in cutlass.range_constexpr(32 // self.nw):
                t = warp + w_ * self.nw
                if t < num_ties:
                    tk = cutlass.Uint32(s_tie_keys[t])
                    ti = cutlass.Int32(s_tie_idx[t])
                    greater = (ck > tk) | ((ck == tk) & (ci < ti))
                    rank = cute.arch.popc(cute.arch.vote_ballot_sync(greater))
                    if lane == 0:
                        if rank < remaining:
                            out_idx_row[out_base + rank] = ti
                            if cutlass.const_expr(self.has_values):
                                out_val_row[out_base + rank] = self.value_from_key(tk)
        else:
            if num_ties <= 128:
                # 4 candidates per lane, 4 targets per warp (sglang 128x128).
                c0k = cutlass.Uint32(0)
                c1k = cutlass.Uint32(0)
                c2k = cutlass.Uint32(0)
                c3k = cutlass.Uint32(0)
                c0i = cutlass.Int32(0x7FFFFFFF)
                c1i = cutlass.Int32(0x7FFFFFFF)
                c2i = cutlass.Int32(0x7FFFFFFF)
                c3i = cutlass.Int32(0x7FFFFFFF)
                if lane < num_ties:
                    c0k = cutlass.Uint32(s_tie_keys[lane])
                    c0i = cutlass.Int32(s_tie_idx[lane])
                if lane + 32 < num_ties:
                    c1k = cutlass.Uint32(s_tie_keys[lane + 32])
                    c1i = cutlass.Int32(s_tie_idx[lane + 32])
                if lane + 64 < num_ties:
                    c2k = cutlass.Uint32(s_tie_keys[lane + 64])
                    c2i = cutlass.Int32(s_tie_idx[lane + 64])
                if lane + 96 < num_ties:
                    c3k = cutlass.Uint32(s_tie_keys[lane + 96])
                    c3i = cutlass.Int32(s_tie_idx[lane + 96])
                cks = (c0k, c1k, c2k, c3k)
                cis = (c0i, c1i, c2i, c3i)
                for i in cutlass.range_constexpr(128 // self.nw):
                    t = warp + i * self.nw
                    if t < num_ties:
                        tk = cutlass.Uint32(s_tie_keys[t])
                        ti = cutlass.Int32(s_tie_idx[t])
                        rank = cutlass.Int32(0)
                        for j in cutlass.range_constexpr(4):
                            g = (cks[j] > tk) | ((cks[j] == tk) & (cis[j] < ti))
                            rank = rank + cute.arch.popc(cute.arch.vote_ballot_sync(g))
                        if lane == 0:
                            if rank < remaining:
                                out_idx_row[out_base + rank] = ti
                                if cutlass.const_expr(self.has_values):
                                    out_val_row[out_base + rank] = self.value_from_key(
                                        tk
                                    )
            else:
                self._tie_select_radix(
                    s_tie_keys,
                    s_tie_idx,
                    num_ties,
                    out_base,
                    remaining,
                    s_hist,
                    s_warp_sums,
                    s_misc,
                    out_idx_row,
                    out_val_row,
                    tidx,
                )

    @cute.jit
    def _tie_select_radix(
        self,
        s_tie_keys,
        s_tie_idx,
        num_ties,
        out_base,
        remaining,
        s_hist,
        s_warp_sums,
        s_misc,
        out_idx_row,
        out_val_row,
        tidx,
    ):
        """Block-wide byte-radix select over the exact ordered keys, in smem.
        Each thread owns tie_cap/nt strided candidates in registers (2 at
        1024 threads, 4 at 512 -- the tie stage is sized by k, not by block
        shape).  Reuses s_hist[0..511] as the double-buffered 256-bin
        histogram (the coarse histogram is dead by now).  s_misc slots:
        [0..2] scan results, [6] = gt scatter counter, [7] = eq counter.
        """
        # Per-thread candidate state: FOUR strided slots, written out as plain
        # variables because the DSL's staged control flow tracks variable
        # names, not Python list elements.  Slot j holds candidate tidx +
        # j*nt; at 1024 threads slots 2/3 start beyond tie_cap = 2048 and are
        # never active (num_ties <= tie_cap), at 512 threads all four are.
        t0 = tidx
        t1 = tidx + self.nt
        t2 = tidx + 2 * self.nt
        t3 = tidx + 3 * self.nt
        active0 = t0 < num_ties
        active1 = t1 < num_ties
        active2 = t2 < num_ties
        active3 = t3 < num_ties
        key0 = cutlass.Uint32(0)
        key1 = cutlass.Uint32(0)
        key2 = cutlass.Uint32(0)
        key3 = cutlass.Uint32(0)
        idx0 = cutlass.Int32(0)
        idx1 = cutlass.Int32(0)
        idx2 = cutlass.Int32(0)
        idx3 = cutlass.Int32(0)
        if active0:
            key0 = cutlass.Uint32(s_tie_keys[t0])
            idx0 = cutlass.Int32(s_tie_idx[t0])
        if active1:
            key1 = cutlass.Uint32(s_tie_keys[t1])
            idx1 = cutlass.Int32(s_tie_idx[t1])
        if active2:
            key2 = cutlass.Uint32(s_tie_keys[t2])
            idx2 = cutlass.Int32(s_tie_idx[t2])
        if active3:
            key3 = cutlass.Uint32(s_tie_keys[t3])
            idx3 = cutlass.Int32(s_tie_idx[t3])
        pos0 = cutlass.Int32(remaining)  # sentinel: >= remaining => not selected
        pos1 = cutlass.Int32(remaining)
        pos2 = cutlass.Int32(remaining)
        pos3 = cutlass.Int32(remaining)

        if tidx < 256:
            s_hist[tidx] = cutlass.Int32(0)
        if tidx == 256:
            s_misc[6] = cutlass.Int32(0)
            s_misc[7] = cutlass.Int32(0)
        cute.arch.barrier()

        total_active = cutlass.Int32(num_ties)
        remain = cutlass.Int32(remaining)

        num_rounds = cutlass.const_expr(len(self.exact_shifts))
        for r in cutlass.range_constexpr(num_rounds):
            shift = cutlass.const_expr(self.exact_shifts[r])
            hist_off = cutlass.const_expr((r % 2) * 256)
            next_off = cutlass.const_expr(((r + 1) % 2) * 256)

            # ``remain`` is block-uniform, so barriers inside the guard are
            # safe.  It stays > 0 until resolution by the bin invariant; the
            # guard only skips dead rounds after early resolution.
            if remain > 0:
                if active0:
                    b = cutlass.Int32(
                        (key0 >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)
                    )
                    smem_atomic_add(s_hist + (hist_off + b), 1)
                if active1:
                    b = cutlass.Int32(
                        (key1 >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)
                    )
                    smem_atomic_add(s_hist + (hist_off + b), 1)
                if active2:
                    b = cutlass.Int32(
                        (key2 >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)
                    )
                    smem_atomic_add(s_hist + (hist_off + b), 1)
                if active3:
                    b = cutlass.Int32(
                        (key3 >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)
                    )
                    smem_atomic_add(s_hist + (hist_off + b), 1)
                if cutlass.const_expr(r + 1 < num_rounds):
                    if tidx < 256:
                        s_hist[next_off + tidx] = cutlass.Int32(0)
            cute.arch.barrier()

            if remain > 0:
                self.scan256_and_find(
                    s_hist + hist_off, total_active, remain, s_warp_sums, s_misc, tidx
                )
                bucket = s_misc[0]
                above = s_misc[1]
                cnt = s_misc[2]
                total_active = cnt
                remain_next = remain - above

                # Scatter: candidates in buckets above the threshold are
                # definite winners (positions from the shared gt counter);
                # bucket == threshold survives to the next round; the final
                # round places exact-equal keys in fill order after all
                # strictly-greater winners ((remaining - remain_next) of
                # them across all rounds).
                if active0:
                    b0 = cutlass.Int32(
                        (key0 >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)
                    )
                    if b0 > bucket:
                        pos0 = smem_atomic_add(s_misc + 6, 1)
                        active0 = False
                    else:
                        if b0 < bucket:
                            active0 = False
                        else:
                            if cutlass.const_expr(r + 1 == num_rounds):
                                eqp = smem_atomic_add(s_misc + 7, 1)
                                pos0 = (remaining - remain_next) + eqp
                if active1:
                    b1 = cutlass.Int32(
                        (key1 >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)
                    )
                    if b1 > bucket:
                        pos1 = smem_atomic_add(s_misc + 6, 1)
                        active1 = False
                    else:
                        if b1 < bucket:
                            active1 = False
                        else:
                            if cutlass.const_expr(r + 1 == num_rounds):
                                eqp = smem_atomic_add(s_misc + 7, 1)
                                pos1 = (remaining - remain_next) + eqp
                if active2:
                    b2 = cutlass.Int32(
                        (key2 >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)
                    )
                    if b2 > bucket:
                        pos2 = smem_atomic_add(s_misc + 6, 1)
                        active2 = False
                    else:
                        if b2 < bucket:
                            active2 = False
                        else:
                            if cutlass.const_expr(r + 1 == num_rounds):
                                eqp = smem_atomic_add(s_misc + 7, 1)
                                pos2 = (remaining - remain_next) + eqp
                if active3:
                    b3 = cutlass.Int32(
                        (key3 >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)
                    )
                    if b3 > bucket:
                        pos3 = smem_atomic_add(s_misc + 6, 1)
                        active3 = False
                    else:
                        if b3 < bucket:
                            active3 = False
                        else:
                            if cutlass.const_expr(r + 1 == num_rounds):
                                eqp = smem_atomic_add(s_misc + 7, 1)
                                pos3 = (remaining - remain_next) + eqp
                remain = remain_next
            cute.arch.barrier()

        # Emit winners.  gt winners hold positions [0, G) from the shared
        # counter; exact-equal survivors fill [G, remaining) capped.  Output
        # order within the tie block is unspecified (atomic race), matching
        # the ``radix`` backend contract.
        if pos0 < remaining:
            out_idx_row[out_base + pos0] = idx0
            if cutlass.const_expr(self.has_values):
                out_val_row[out_base + pos0] = self.value_from_key(key0)
        if pos1 < remaining:
            out_idx_row[out_base + pos1] = idx1
            if cutlass.const_expr(self.has_values):
                out_val_row[out_base + pos1] = self.value_from_key(key1)
        if pos2 < remaining:
            out_idx_row[out_base + pos2] = idx2
            if cutlass.const_expr(self.has_values):
                out_val_row[out_base + pos2] = self.value_from_key(key2)
        if pos3 < remaining:
            out_idx_row[out_base + pos3] = idx3
            if cutlass.const_expr(self.has_values):
                out_val_row[out_base + pos3] = self.value_from_key(key3)

    # ------------------------------------------------------------------
    # Single-CTA kernel (one block per row)
    # ------------------------------------------------------------------
    @cute.jit
    def _prim_row_done(self, out_gmem_row, s_stage, s_pt, pt_gmem, tidx):
        """Row epilogue of the single-CTA kernel.  Paged variants store the
        staged raw columns as physical slots in one coalesced pass (-1 pads
        kept; pages past the smem stage gather from global memory); then the
        PDL trigger."""
        top_k = cutlass.const_expr(self.top_k)
        if cutlass.const_expr(self.page_size > 0):
            lg = cutlass.const_expr(self.page_size.bit_length() - 1)
            cute.arch.barrier()  # every staged column is visible
            for i in range(tidx, top_k, NUM_THREADS):
                c = s_stage[i]
                if c < 0:
                    out_gmem_row[i] = cutlass.Int32(-1)
                else:
                    page = c >> cutlass.Int32(lg)
                    base = cutlass.Int32(0)
                    if cutlass.const_expr(self.pt_cache >= self.pt_pages):
                        base = s_pt[page]  # the whole table row is staged
                    else:
                        if page < cutlass.Int32(self.pt_cache):
                            base = s_pt[page]
                        else:
                            base = pt_gmem[page]
                    out_gmem_row[i] = (base << cutlass.Int32(lg)) + (
                        c & cutlass.Int32(self.page_size - 1)
                    )
        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_launch_dependents()

    @cute.kernel
    def topk_kernel(
        self,
        input_data: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
        page_table: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        top_k = cutlass.const_expr(self.top_k)
        next_n = cutlass.const_expr(self.next_n)
        compress_ratio = cutlass.const_expr(self.compress_ratio)
        n_cols = cutlass.const_expr(input_data.shape[1])  # static N

        # Tensors end here: raw base pointers only from this point on.
        in_ptr = input_data.iterator
        seq_ptr = seqlen.iterator
        oi_ptr = output_indices.iterator
        pt_ptr = page_table.iterator

        # ---- shared memory (layout-free raw arrays) ----
        smem = SmemAllocator()
        s_hist = smem.allocate_array(cutlass.Int32, self.hist_size, byte_alignment=128)
        s_tie_keys = smem.allocate_array(cutlass.Uint32, TIE_CAP, byte_alignment=128)
        s_tie_idx = smem.allocate_array(cutlass.Int32, TIE_CAP, byte_alignment=128)
        s_warp_sums = smem.allocate_array(cutlass.Int32, NUM_WARPS, byte_alignment=128)
        # s_misc: [0]=bin/bucket [1]=above [2]=count [4]/[5]=MC emit counters
        # [6]/[7]=tie counters [8]=max(~tie_key) [9]=max(tie_key)
        s_misc = smem.allocate_array(cutlass.Int32, 12, byte_alignment=128)
        # Contended output counters on their own cache lines (sglang
        # alignas(128) idiom).
        s_count_gt = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)
        s_count_eq = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)
        # paged output: the raw-column stage every emit targets, and the
        # staged leading entries of the request's page-table row (4-int
        # dummies when unpaged)
        s_stage = smem.allocate_array(
            cutlass.Int32, self.top_k if self.page_size > 0 else 4, byte_alignment=128
        )
        s_pt = smem.allocate_array(
            cutlass.Int32, max(self.pt_cache, 4), byte_alignment=128
        )

        row64 = cutlass.Int64(row)
        row_in = in_ptr + row64 * n_cols
        out_gmem_row = oi_ptr + row64 * top_k
        # every emit site below writes through out_idx_row: the global row
        # when unpaged, the smem stage when paged (trace-time choice; the
        # epilogue _prim_row_done maps the stage out)
        out_idx_row = s_stage if self.page_size > 0 else out_gmem_row
        # the request's page-table row; unpaged variants never touch the table
        # (not even its shape: every parameter word read here is a constant
        # load in the prologue, where ncu showed the MIO queue saturating on
        # parameter and special-register reads)
        # (trace-time ternary: a bare `if` here becomes a DSL region and the
        # name would not escape it)
        pt_gmem = (
            pt_ptr + cutlass.Int64(row // next_n) * cutlass.Int64(page_table.shape[1])
            if self.page_size > 0
            else pt_ptr
        )
        if cutlass.const_expr(self.has_values):
            out_val_row = output_values.iterator + row64 * top_k
        else:
            # Dummy binding, never dereferenced: every use is guarded by
            # const_expr(self.has_values).
            out_val_row = oi_ptr

        # PDL: wait before the FIRST global read -- the page table, the
        # length and the logits may all be produced by the kernel preceding
        # this one in the stream, and only griddepcontrol.wait makes its
        # writes visible.  griddepcontrol is SM90+; compiled out on Ampere
        # (enable_pdl=False).
        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_wait()

        # paged: load the leading page-table entries (up to four per thread)
        # BEFORE the length load and store them to smem AFTER it, so both
        # latencies overlap (in-order issue: a store right after its load
        # would stall the warp before the length load goes out)
        pv0 = cutlass.Int32(0)
        pv1 = cutlass.Int32(0)
        pv2 = cutlass.Int32(0)
        pv3 = cutlass.Int32(0)
        # The table may be NARROWER than ceil(N / page_size): a serving engine
        # sizes it for the batch's longest row while the logits buffer is padded
        # wider (SGLang: 129 pages of 64 under 8448 columns).  Never read past
        # its row; a row past its coverage is clamped below.  Trace-time
        # ternary: unpaged variants never touch the table.
        pt_w = (
            cutlass.Int32(page_table.shape[1])
            if self.page_size > 0
            else cutlass.Int32(0)
        )
        n_pre = cutlass.Int32(self.pt_cache)
        if cutlass.const_expr(self.pt_cache > 0):
            if n_pre > pt_w:
                n_pre = pt_w
            if tidx < n_pre:
                pv0 = pt_gmem[tidx]
            if cutlass.const_expr(self.pt_cache > NUM_THREADS):
                if tidx + NUM_THREADS < n_pre:
                    pv1 = pt_gmem[tidx + NUM_THREADS]
            if cutlass.const_expr(self.pt_cache > 2 * NUM_THREADS):
                if tidx + 2 * NUM_THREADS < n_pre:
                    pv2 = pt_gmem[tidx + 2 * NUM_THREADS]
            if cutlass.const_expr(self.pt_cache > 3 * NUM_THREADS):
                if tidx + 3 * NUM_THREADS < n_pre:
                    pv3 = pt_gmem[tidx + 3 * NUM_THREADS]

        # Effective row length: next_n adjustment in token space first, then
        # compression (identical to the ``radix`` backend), clamped to [0, N].
        seq_len = seq_ptr[row // next_n]
        length = (seq_len - next_n + (row % next_n) + 1) // compress_ratio
        if length < 0:
            length = cutlass.Int32(0)
        if length > n_cols:
            length = cutlass.Int32(n_cols)
        if cutlass.const_expr(self.page_size > 0):
            # a row past the table's coverage is clamped to the covered prefix;
            # a table wider than the row can need is counted up to the width's
            # own page count, so the shift cannot overflow int32
            pw = pt_w
            if pw > cutlass.Int32(self.pt_pages):
                pw = cutlass.Int32(self.pt_pages)
            cov = pw << cutlass.Int32(self.page_size.bit_length() - 1)
            if length > cov:
                length = cov

        if cutlass.const_expr(self.pt_cache > 0):
            if tidx < n_pre:
                s_pt[tidx] = pv0
            if cutlass.const_expr(self.pt_cache > NUM_THREADS):
                if tidx + NUM_THREADS < n_pre:
                    s_pt[tidx + NUM_THREADS] = pv1
            if cutlass.const_expr(self.pt_cache > 2 * NUM_THREADS):
                if tidx + 2 * NUM_THREADS < n_pre:
                    s_pt[tidx + 2 * NUM_THREADS] = pv2
            if cutlass.const_expr(self.pt_cache > 3 * NUM_THREADS):
                if tidx + 3 * NUM_THREADS < n_pre:
                    s_pt[tidx + 3 * NUM_THREADS] = pv3

        # Neutral defaults for mode-specific args.
        u0 = cutlass.Uint32(0)
        i0 = cutlass.Int32(0)

        if top_k >= length:
            # Degenerate row: the top-k is all valid indices, -1 padded (the
            # padded value slots hold 0, as radix_cutlass's do).
            for i in range(tidx, top_k, NUM_THREADS):
                if i < length:
                    out_idx_row[i] = cutlass.Int32(i)
                    if cutlass.const_expr(self.has_values):
                        out_val_row[i] = row_in[i]
                else:
                    out_idx_row[i] = cutlass.Int32(-1)
                    if cutlass.const_expr(self.has_values):
                        out_val_row[i] = self.dtype(0.0)
        else:
            # ---- init (vectorized 16B zero stores; hist_size % 4096 == 0) ----
            s_hist_addr = s_hist.toint()
            for i in range(tidx * 4, self.hist_size, NUM_THREADS * 4):
                st_shared_v4_zero(s_hist_addr + i * 4)
            if tidx == 0:
                s_count_gt[0] = cutlass.Int32(0)
                s_count_eq[0] = cutlass.Int32(0)
                s_misc[8] = cutlass.Int32(0)  # max(~tie key): min encoder
                s_misc[9] = cutlass.Int32(0)  # max(tie key)
            cute.arch.barrier()

            if cutlass.const_expr(self._reg_path_ok(n_cols, n_cols, self.reg_slots)):
                # ---- register path: read the row from gmem exactly once ----
                # sglang TopKRegister<nslots>: thread t owns vectors t + s * NT.
                # Every slot's load is bounded by the REAL length (num_full), so
                # a 1K-column row in a 16K buffer issues one vector per thread
                # on 250 threads and skips the other slots' code entirely.
                nslots = cutlass.const_expr(self._reg_num_slots(n_cols))
                self._reg_solve_row(
                    nslots,
                    row_in,
                    length,
                    tidx,
                    s_hist,
                    s_count_gt,
                    s_count_eq,
                    s_tie_keys,
                    s_tie_idx,
                    s_warp_sums,
                    s_misc,
                    out_idx_row,
                    out_val_row,
                )
                # the overflow arms below re-read the crossing bin and its
                # float boundaries (~20 instructions once per row)
                threshold_bin = s_misc[0]
                hi_b = u0
                lo_b = u0
                if cutlass.const_expr(self.boundary_cls):
                    hi_b = self.coarse_bin_gt_threshold_f32(threshold_bin + 1).bitcast(
                        cutlass.Uint32
                    )
                    lo_b = self.coarse_bin_lower_bound_f32(threshold_bin).bitcast(
                        cutlass.Uint32
                    )
            else:
                # ---- runtime register arm for widths past the register
                # capacity (the multi-CTA solo arm's form): a row within
                # reg_slots vector slots per thread takes the read-once
                # register solve, a longer row streams.  Without it every
                # decode row in a 64K / 256K buffer streamed the row twice
                # through gmem (paged decode trace 3.84 us at 64K against 3.66
                # for the same rows in a 16K buffer).
                reg_cap = self.reg_slots * NUM_THREADS * self.vec_elems
                use_reg = cutlass.Int32(0)
                if cutlass.const_expr(
                    self._reg_path_ok(n_cols, reg_cap, self.reg_slots)
                ):
                    if length <= cutlass.Int32(reg_cap):
                        use_reg = cutlass.Int32(1)
                threshold_bin = i0
                hi_b = u0
                lo_b = u0
                if use_reg == 1:
                    self._reg_solve_row(
                        self.reg_slots,
                        row_in,
                        length,
                        tidx,
                        s_hist,
                        s_count_gt,
                        s_count_eq,
                        s_tie_keys,
                        s_tie_idx,
                        s_warp_sums,
                        s_misc,
                        out_idx_row,
                        out_val_row,
                    )
                    threshold_bin = s_misc[0]
                    if cutlass.const_expr(self.boundary_cls):
                        hi_b = self.coarse_bin_gt_threshold_f32(
                            threshold_bin + 1
                        ).bitcast(cutlass.Uint32)
                        lo_b = self.coarse_bin_lower_bound_f32(threshold_bin).bitcast(
                            cutlass.Uint32
                        )
                else:
                    # ---- streaming path: two vectorized passes over gmem ----
                    if cutlass.const_expr(self.warp_agg):
                        # Row specialization: the batched walker wins on flood
                        # rows and is pure per-vector tax on mixed rows (the
                        # saturated-randn cell measured -9% from it).  A 4-point
                        # bin-uniformity probe (uniform addresses -> broadcast
                        # loads) picks the walker; a row that fools the probe
                        # only loses the optimization, never correctness.
                        pb = self.coarse_bin(self.load_scalar(row_in, i0))
                        hist_flood = cutlass.Int32(1)
                        if self.coarse_bin(self.load_scalar(row_in, length >> 2)) != pb:
                            hist_flood = cutlass.Int32(0)
                        if self.coarse_bin(self.load_scalar(row_in, length >> 1)) != pb:
                            hist_flood = cutlass.Int32(0)
                        if self.coarse_bin(self.load_scalar(row_in, length - 1)) != pb:
                            hist_flood = cutlass.Int32(0)
                        if hist_flood == 1:
                            self._stream_row(
                                _OP_HIST,
                                0,
                                row_in,
                                length,
                                i0,
                                tidx,
                                s_hist,
                                s_count_gt,
                                s_count_eq,
                                s_tie_keys,
                                s_tie_idx,
                                s_misc,
                                out_idx_row,
                                out_val_row,
                                i0,
                                u0,
                                u0,
                                u0,
                                i0,
                                s_misc,
                            )
                        else:
                            self._stream_row(
                                _OP_HIST,
                                0,
                                row_in,
                                length,
                                i0,
                                tidx,
                                s_hist,
                                s_count_gt,
                                s_count_eq,
                                s_tie_keys,
                                s_tie_idx,
                                s_misc,
                                out_idx_row,
                                out_val_row,
                                i0,
                                u0,
                                u0,
                                u0,
                                i0,
                                s_misc,
                                wa_batch=False,
                            )
                    else:
                        self._stream_row(
                            _OP_HIST,
                            0,
                            row_in,
                            length,
                            i0,
                            tidx,
                            s_hist,
                            s_count_gt,
                            s_count_eq,
                            s_tie_keys,
                            s_tie_idx,
                            s_misc,
                            out_idx_row,
                            out_val_row,
                            i0,
                            u0,
                            u0,
                            u0,
                            i0,
                            s_misc,
                        )
                    cute.arch.barrier()

                    self.find_threshold_coarse(
                        s_hist, length, cutlass.Int32(top_k), s_warp_sums, s_misc, tidx
                    )
                    threshold_bin = s_misc[0]
                    tb_cnt = i0
                    if cutlass.const_expr(self.warp_agg):
                        # threshold-bin population (read before the fused-refine
                        # zeroing below): exact tie density for the collect
                        # walker choice
                        tb_cnt = s_hist[threshold_bin]
                        if cutlass.const_expr(self._fuse_r1):
                            # order the read against the zeroing below: without
                            # this, a fast thread can clear threshold_bin's count
                            # before a slow warp has read it (warp-uniform stale
                            # zero -> that warp silently loses the batched-collect
                            # specialization; no correctness impact, but the
                            # ordering must not depend on that analysis)
                            cute.arch.barrier()
                    hi_b = u0
                    lo_b = u0
                    if cutlass.const_expr(self.boundary_cls):
                        hi_b = self.coarse_bin_gt_threshold_f32(
                            threshold_bin + 1
                        ).bitcast(cutlass.Uint32)
                        lo_b = self.coarse_bin_lower_bound_f32(threshold_bin).bitcast(
                            cutlass.Uint32
                        )
                    if cutlass.const_expr(self._fuse_r1):
                        # reset s_hist for the collect-fused (12,12) refine
                        # histogram (the coarse counts are consumed by now)
                        s_ha = s_hist.toint()
                        for i in range(tidx * 4, self.hist_size, NUM_THREADS * 4):
                            st_shared_v4_zero(s_ha + i * 4)
                        cute.arch.barrier()

                    if cutlass.const_expr(
                        ((n_cols * self.elem_bytes) % 16 == 0)
                        and (n_cols <= NUM_THREADS * 32)
                    ):
                        # <= 32 elements/thread: scan-based collect (no
                        # per-element atomics, no store-on-atomic chains).
                        # Capped at N=32768: larger single-CTA shapes only run
                        # in the SM-saturated regime (small batches go through
                        # the multi-CTA solo path), where the emit re-walk's
                        # extra ALU measurably loses to the atomic path.
                        self._collect_scan_row(
                            row_in,
                            length,
                            tidx,
                            s_tie_keys,
                            s_tie_idx,
                            s_warp_sums,
                            s_count_gt,
                            s_count_eq,
                            s_misc,
                            s_hist,
                            out_idx_row,
                            out_val_row,
                            threshold_bin,
                            hi_b,
                            lo_b,
                        )
                    else:
                        if cutlass.const_expr(self.warp_agg):
                            # batched collect only when ties are dense (>= 25%
                            # of the row); mixed rows take the tax-free
                            # per-element walker
                            if tb_cnt >= (length >> 2):
                                self._stream_row(
                                    _OP_COLLECT,
                                    0,
                                    row_in,
                                    length,
                                    i0,
                                    tidx,
                                    s_hist,
                                    s_count_gt,
                                    s_count_eq,
                                    s_tie_keys,
                                    s_tie_idx,
                                    s_misc,
                                    out_idx_row,
                                    out_val_row,
                                    threshold_bin,
                                    hi_b,
                                    lo_b,
                                    u0,
                                    i0,
                                    s_misc,
                                )
                            else:
                                self._stream_row(
                                    _OP_COLLECT,
                                    0,
                                    row_in,
                                    length,
                                    i0,
                                    tidx,
                                    s_hist,
                                    s_count_gt,
                                    s_count_eq,
                                    s_tie_keys,
                                    s_tie_idx,
                                    s_misc,
                                    out_idx_row,
                                    out_val_row,
                                    threshold_bin,
                                    hi_b,
                                    lo_b,
                                    u0,
                                    i0,
                                    s_misc,
                                    wa_batch=False,
                                )
                        else:
                            self._stream_row(
                                _OP_COLLECT,
                                0,
                                row_in,
                                length,
                                i0,
                                tidx,
                                s_hist,
                                s_count_gt,
                                s_count_eq,
                                s_tie_keys,
                                s_tie_idx,
                                s_misc,
                                out_idx_row,
                                out_val_row,
                                threshold_bin,
                                hi_b,
                                lo_b,
                                u0,
                                i0,
                                s_misc,
                            )
                    cute.arch.barrier()

            gt_count = s_count_gt[0]
            eq_count = s_count_eq[0]
            remaining = top_k - gt_count
            if remaining < 0:
                # NaN floods can push gt past top_k (the guarded stores
                # drop the excess); nothing is left for the tie phase.
                remaining = cutlass.Int32(0)
                eq_count = cutlass.Int32(0)

            # Uniform-tie fast path: when every tie candidate has the SAME
            # exact key (min == max), any `remaining` of them are an exact
            # answer, and the staged prefix (min(eq_count, TIE_CAP) >=
            # remaining) suffices -- clamp eq_count so the direct copy arm
            # below handles it.  Past TIE_CAP this skips the multi-pass gmem
            # refinement (the adversarial low-entropy cliff: constant /
            # few-valued / quantized logits); BELOW it, it skips the in-smem
            # radix select over identical keys -- ReLU-style rows tie
            # hundreds to ~2000 exact zeros at the boundary and ran the four
            # byte-radix rounds (2.4 us of a 6 us row, exactly what SGLang's
            # kernel does; walk-first's census arm measured 6.07 -> 3.80 us
            # from the same shortcut).
            if eq_count > remaining:
                if cutlass.const_expr(self.approx_ties):
                    # approx variant: past the cap any staged first-arrival
                    # subset fills the row (sglang tie-truncation semantics);
                    # the range reds are compiled out, so no uniform test
                    if eq_count > TIE_CAP:
                        if cutlass.const_expr(top_k > TIE_CAP):
                            if remaining <= TIE_CAP:
                                eq_count = remaining
                        else:
                            eq_count = remaining
                else:
                    # both clamps route through the staged-copy arm, which
                    # can only fill remaining <= TIE_CAP slots (big-k rows
                    # beyond that fall through to the overflow tail's EQFILL)
                    if (~cutlass.Uint32(s_misc[8])) == cutlass.Uint32(s_misc[9]):
                        if cutlass.const_expr(top_k > TIE_CAP):
                            if remaining <= TIE_CAP:
                                eq_count = remaining
                        else:
                            eq_count = remaining

            # direct copy only from the staged candidates: the stage holds
            # TIE_CAP entries, and with top_k > TIE_CAP a bin that exactly
            # fills the remainder can hold more (eq_count <= remaining but >
            # TIE_CAP); that case goes to the overflow refinement below
            copy_cap = remaining
            if copy_cap > TIE_CAP:
                copy_cap = cutlass.Int32(TIE_CAP)
            if eq_count <= copy_cap:
                # Every candidate is selected (and by the invariant this
                # fills the row exactly).
                for t in range(tidx, eq_count, NUM_THREADS):
                    out_idx_row[gt_count + t] = s_tie_idx[t]
                    if cutlass.const_expr(self.has_values):
                        out_val_row[gt_count + t] = self.value_from_key(s_tie_keys[t])
                # Defensive: unreachable while the invariant holds, but a -1
                # pad beats stale memory if it ever breaks.
                for t in range(tidx + eq_count, remaining, NUM_THREADS):
                    out_idx_row[gt_count + t] = cutlass.Int32(-1)
            else:
                if eq_count <= TIE_CAP:
                    # Exact in-smem select over the staged candidates.
                    self.tie_select(
                        s_tie_keys,
                        s_tie_idx,
                        eq_count,
                        gt_count,
                        remaining,
                        s_hist,
                        s_warp_sums,
                        s_misc,
                        out_idx_row,
                        out_val_row,
                        tidx,
                    )
                else:
                    # ---- overflow: exact refinement (streaming re-reads) ----
                    # Refine a bit-prefix of the exact key with (shift, bits)
                    # histogram rounds; STOP as soon as the surviving
                    # candidate count fits the smem stage, then one masked
                    # STAGE pass + tie_select finish exactly (staged entries
                    # carry FULL keys, so a partial prefix is fine).  The
                    # tie-key RANGE [kmin, kmax] (tracked by the uniform-tie
                    # red.max) drives a per-row exact round skip: a round's
                    # digit is constant across every tie iff
                    # kmax >> shift == kmin >> shift (the ties lie in a
                    # contiguous key range), and the digit then comes free
                    # from kmax -- so degenerate high rounds cost nothing on
                    # low-entropy rows, while binade-straddling rows (2.0 vs
                    # its predecessor in one bin: bounds differ at bit 24+)
                    # automatically run the high round.  The collect-fused
                    # (12, 12) histogram is valid whenever no earlier round
                    # narrowed the population (prefix_mask == 0).  If every
                    # round completes without fitting, survivors share all
                    # covered bits => key-identical, so the TIE_CAP-capped
                    # stage is still exact (top_k <= TIE_CAP is asserted).
                    prefix = cutlass.Uint32(0)
                    prefix_mask = cutlass.Uint32(0)
                    total = eq_count
                    remain = remaining
                    staged = cutlass.Int32(0)
                    kmax_u = cutlass.Uint32(s_misc[9])
                    kmin_u = ~cutlass.Uint32(s_misc[8])
                    num_refine = cutlass.const_expr(len(self.refine_rounds))
                    for r in cutlass.range_constexpr(num_refine):
                        rsh = cutlass.const_expr(self.refine_rounds[r][0])
                        rbits = cutlass.const_expr(self.refine_rounds[r][1])
                        rbins = cutlass.const_expr(1 << rbits)
                        packed = cutlass.const_expr((rbits << 8) | rsh)
                        skip = staged
                        if (kmax_u >> cutlass.Uint32(rsh)) == (
                            kmin_u >> cutlass.Uint32(rsh)
                        ):
                            # digit constant across all ties: absorb it into
                            # the prefix for free (no narrowing: total and
                            # remain are unchanged, exactly as if the round
                            # ran and found everything in one bucket)
                            if skip == 0:
                                prefix = prefix | (
                                    (
                                        (kmax_u >> cutlass.Uint32(rsh))
                                        & cutlass.Uint32(rbins - 1)
                                    )
                                    << cutlass.Uint32(rsh)
                                )
                                prefix_mask = prefix_mask | (
                                    cutlass.Uint32(rbins - 1) << cutlass.Uint32(rsh)
                                )
                            skip = cutlass.Int32(1)
                        if skip == 0:
                            fused = cutlass.Int32(0)
                            if cutlass.const_expr(
                                _FUSE_COLLECT_R1 and self.is_f32 and rsh == 12
                            ):
                                if prefix_mask == 0:
                                    fused = cutlass.Int32(1)
                            if fused == 0:
                                for i in range(tidx, rbins, NUM_THREADS):
                                    s_hist[i] = cutlass.Int32(0)
                                cute.arch.barrier()
                                self._stream_row(
                                    _OP_REFINE,
                                    packed,
                                    row_in,
                                    length,
                                    i0,
                                    tidx,
                                    s_hist,
                                    s_count_gt,
                                    s_count_eq,
                                    s_tie_keys,
                                    s_tie_idx,
                                    s_misc,
                                    out_idx_row,
                                    out_val_row,
                                    threshold_bin,
                                    prefix,
                                    prefix_mask,
                                    u0,
                                    i0,
                                    s_misc,
                                )
                                cute.arch.barrier()
                            if cutlass.const_expr(rbins == 256):
                                self.scan256_and_find(
                                    s_hist, total, remain, s_warp_sums, s_misc, tidx
                                )
                                bucket = s_misc[0]
                                above = s_misc[1]
                                cnt = s_misc[2]
                            else:
                                # 4096-bin round (fp32 hist_size == 4096)
                                self.find_threshold_wide(
                                    s_hist, total, remain, s_warp_sums, s_misc, tidx
                                )
                                bucket = s_misc[0]
                                above = s_misc[1]
                                cnt = s_misc[2]
                            prefix = prefix | (
                                cutlass.Uint32(bucket) << cutlass.Uint32(rsh)
                            )
                            prefix_mask = prefix_mask | (
                                cutlass.Uint32(rbins - 1) << cutlass.Uint32(rsh)
                            )
                            total = cnt
                            remain = remain - above
                            cute.arch.barrier()
                            if total <= TIE_CAP:
                                staged = cutlass.Int32(1)

                    # ---- terminal stage + exact in-smem select ----
                    if cutlass.const_expr(top_k > TIE_CAP):
                        if remain > TIE_CAP:
                            # ties exceed the staging buffer: all refinement
                            # survivors are key-equal (coverage proof), so
                            # masked-eq fill straight to gmem needs no
                            # ordering
                            if tidx == 0:
                                s_misc[6] = cutlass.Int32(0)
                            cute.arch.barrier()
                            self._stream_row(
                                _OP_EQFILL,
                                0,
                                row_in,
                                length,
                                i0,
                                tidx,
                                s_hist,
                                s_count_gt,
                                s_count_eq,
                                s_tie_keys,
                                s_tie_idx,
                                s_misc,
                                out_idx_row,
                                out_val_row,
                                threshold_bin,
                                prefix,
                                prefix_mask,
                                u0,
                                top_k - remain,
                                s_misc,
                            )
                        else:
                            self._terminal_stage_select(
                                row_in,
                                length,
                                i0,
                                u0,
                                tidx,
                                s_hist,
                                s_count_gt,
                                s_count_eq,
                                s_tie_keys,
                                s_tie_idx,
                                s_warp_sums,
                                s_misc,
                                out_idx_row,
                                out_val_row,
                                threshold_bin,
                                prefix,
                                prefix_mask,
                            )
                    else:
                        self._terminal_stage_select(
                            row_in,
                            length,
                            i0,
                            u0,
                            tidx,
                            s_hist,
                            s_count_gt,
                            s_count_eq,
                            s_tie_keys,
                            s_tie_idx,
                            s_warp_sums,
                            s_misc,
                            out_idx_row,
                            out_val_row,
                            threshold_bin,
                            prefix,
                            prefix_mask,
                        )

        # row epilogue: paged finish pass + PDL trigger
        self._prim_row_done(out_gmem_row, s_stage, s_pt, pt_gmem, tidx)

    # ------------------------------------------------------------------
    # Multi-CTA group kernel (ctas_per_group CTAs cooperate per row)
    # ------------------------------------------------------------------
    @cute.kernel
    def mc_topk_kernel(
        self,
        input_data: cute.Tensor,
        row_states: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_size, _, _ = cute.arch.grid_dim()

        top_k = cutlass.const_expr(self.top_k)
        next_n = cutlass.const_expr(self.next_n)
        compress_ratio = cutlass.const_expr(self.compress_ratio)
        n_cols = cutlass.const_expr(input_data.shape[1])
        cpg = cutlass.const_expr(self.ctas_per_group)
        chunk = cutlass.const_expr(self.chunk_elems)
        state_size = cutlass.const_expr(self.state_size)

        in_ptr = input_data.iterator
        seq_ptr = seqlen.iterator
        oi_ptr = output_indices.iterator

        group = bidx // cpg
        rank = bidx % cpg
        num_groups = grid_size // cpg
        num_rows = input_data.shape[0]
        g_state = row_states.iterator + group * state_size
        g_arrive = g_state + self.g_arrive

        # ---- shared memory ----
        smem = SmemAllocator()
        s_hist = smem.allocate_array(cutlass.Int32, self.hist_size, byte_alignment=128)
        s_tie_keys = smem.allocate_array(cutlass.Uint32, TIE_CAP, byte_alignment=128)
        s_tie_idx = smem.allocate_array(cutlass.Int32, TIE_CAP, byte_alignment=128)
        s_warp_sums = smem.allocate_array(cutlass.Int32, NUM_WARPS, byte_alignment=128)
        s_misc = smem.allocate_array(cutlass.Int32, 12, byte_alignment=128)
        s_count_gt = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)
        s_count_eq = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)
        # Scratch for merged super-histogram readbacks (s_hist must stay
        # intact for the fine-level merge).
        s_scan = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)

        u0 = cutlass.Uint32(0)
        i0 = cutlass.Int32(0)

        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_wait()

        row = cutlass.Int32(group)
        phase = cutlass.Int32(0)
        # 1 once any row of this launch used the group path (and therefore
        # dirtied g_state); gates the end-of-kernel state sweep.
        state_dirty = cutlass.Int32(0)
        while row < num_rows:
            row64 = cutlass.Int64(row)
            row_in = in_ptr + row64 * n_cols
            out_idx_row = oi_ptr + row64 * top_k
            if cutlass.const_expr(self.has_values):
                out_val_row = output_values.iterator + row64 * top_k
            else:
                out_val_row = oi_ptr

            seq_len = seq_ptr[row // next_n]
            length = (seq_len - next_n + (row % next_n) + 1) // compress_ratio
            if length < 0:
                length = cutlass.Int32(0)
            if length > n_cols:
                length = cutlass.Int32(n_cols)

            # My chunk of the row's columns.
            chunk_start = rank * chunk
            local_len = cutlass.Int32(0)
            if chunk_start < length:
                rem = length - chunk_start
                if rem > chunk:
                    local_len = cutlass.Int32(chunk)
                else:
                    local_len = rem
            chunk_in = row_in + chunk_start

            if (top_k >= length) | (length <= _MC_SOLO_ELEMS):
                if top_k >= length:
                    # Degenerate row: chunked direct write, rank 0 pads.  No
                    # group state touched, so no barriers (uniform per group).
                    for i in range(tidx, local_len, NUM_THREADS):
                        out_idx_row[chunk_start + i] = cutlass.Int32(chunk_start + i)
                        if cutlass.const_expr(self.has_values):
                            out_val_row[chunk_start + i] = row_in[chunk_start + i]
                    if rank == 0:
                        for i in range(tidx + length, top_k, NUM_THREADS):
                            out_idx_row[i] = cutlass.Int32(-1)
                            if cutlass.const_expr(self.has_values):
                                out_val_row[i] = self.dtype(0.0)
                else:
                    # ---- solo row: rank 0 resolves it alone ----
                    # The single-CTA streaming algorithm on CTA-local smem:
                    # no group state, no mc_barriers.  ``length`` is
                    # group-uniform, so every rank takes this branch together
                    # and the mc_barrier phase count stays consistent; the
                    # other ranks simply skip to the next row.
                    if rank == 0:
                        self._dbg_ts(g_state, 0, tidx)
                        s_hist_addr_s = s_hist.toint()
                        for i in range(tidx * 4, self.hist_size, NUM_THREADS * 4):
                            st_shared_v4_zero(s_hist_addr_s + i * 4)
                        if tidx == 0:
                            s_count_gt[0] = cutlass.Int32(0)
                            s_count_eq[0] = cutlass.Int32(0)
                            s_misc[8] = cutlass.Int32(0)
                            s_misc[9] = cutlass.Int32(0)
                        cute.arch.barrier()
                        self._dbg_ts(g_state, 1, tidx)

                        # Rows within the register capacity (reg_slots vector
                        # slots per thread, 16K fp32 at 4) take the single-CTA
                        # kernel's register-resident solve: one read bounded by
                        # the real length.  The streaming form below read the
                        # row twice through gmem and measured 3.99 us on the
                        # 64K-envelope decode trace against 3.51 for the same
                        # rows in a 16K buffer (raw SGLang 3.34).
                        threshold_bin_s = i0
                        hi_bs = u0
                        lo_bs = u0
                        use_reg = cutlass.Int32(0)
                        if cutlass.const_expr(
                            self._reg_path_ok(
                                n_cols,
                                self.reg_slots * NUM_THREADS * self.vec_elems,
                                self.reg_slots,
                            )
                        ):
                            if length <= cutlass.Int32(
                                self.reg_slots * NUM_THREADS * self.vec_elems
                            ):
                                use_reg = cutlass.Int32(1)
                        if use_reg == 1:
                            self._reg_solve_row(
                                self.reg_slots,
                                row_in,
                                length,
                                tidx,
                                s_hist,
                                s_count_gt,
                                s_count_eq,
                                s_tie_keys,
                                s_tie_idx,
                                s_warp_sums,
                                s_misc,
                                out_idx_row,
                                out_val_row,
                            )
                            threshold_bin_s = s_misc[0]
                            if cutlass.const_expr(self.boundary_cls):
                                hi_bs = self.coarse_bin_gt_threshold_f32(
                                    threshold_bin_s + 1
                                ).bitcast(cutlass.Uint32)
                                lo_bs = self.coarse_bin_lower_bound_f32(
                                    threshold_bin_s
                                ).bitcast(cutlass.Uint32)
                        else:
                            self._stream_row(
                                _OP_HIST,
                                0,
                                row_in,
                                length,
                                i0,
                                tidx,
                                s_hist,
                                s_count_gt,
                                s_count_eq,
                                s_tie_keys,
                                s_tie_idx,
                                s_misc,
                                out_idx_row,
                                out_val_row,
                                i0,
                                u0,
                                u0,
                                u0,
                                i0,
                                s_misc,
                            )
                            cute.arch.barrier()
                            self._dbg_ts(g_state, 2, tidx)

                            self.find_threshold_coarse(
                                s_hist,
                                length,
                                cutlass.Int32(top_k),
                                s_warp_sums,
                                s_misc,
                                tidx,
                            )
                            threshold_bin_s = s_misc[0]
                            if cutlass.const_expr(self.boundary_cls):
                                hi_bs = self.coarse_bin_gt_threshold_f32(
                                    threshold_bin_s + 1
                                ).bitcast(cutlass.Uint32)
                                lo_bs = self.coarse_bin_lower_bound_f32(
                                    threshold_bin_s
                                ).bitcast(cutlass.Uint32)
                            if cutlass.const_expr(self._fuse_r1):
                                # reset s_hist for the collect-fused (12,12)
                                # refine histogram
                                s_ha_s = s_hist.toint()
                                for i in range(
                                    tidx * 4, self.hist_size, NUM_THREADS * 4
                                ):
                                    st_shared_v4_zero(s_ha_s + i * 4)
                                cute.arch.barrier()

                            self._dbg_ts(g_state, 3, tidx)
                            # NOTE: the scan-based collect measured SLOWER
                            # here than the atomic path (3.7 vs 3.2us collect
                            # phase) -- the multi-CTA kernel's register
                            # pressure hurts the walker -- so the solo path
                            # keeps _stream_row; the scan variant is used by
                            # the single-CTA kernel at N <= 32768 where it
                            # wins at every batch size.
                            self._stream_row(
                                _OP_COLLECT,
                                0,
                                row_in,
                                length,
                                i0,
                                tidx,
                                s_hist,
                                s_count_gt,
                                s_count_eq,
                                s_tie_keys,
                                s_tie_idx,
                                s_misc,
                                out_idx_row,
                                out_val_row,
                                threshold_bin_s,
                                hi_bs,
                                lo_bs,
                                u0,
                                i0,
                                s_misc,
                            )
                            cute.arch.barrier()
                        self._dbg_ts(g_state, 4, tidx)

                        gt_count_s = s_count_gt[0]
                        eq_count_s = s_count_eq[0]
                        remaining_s = top_k - gt_count_s
                        if remaining_s < 0:
                            # NaN flood: gt overran top_k (guarded stores
                            # dropped the excess); no tie phase.
                            remaining_s = cutlass.Int32(0)
                            eq_count_s = cutlass.Int32(0)

                        # uniform-tie fast path (below AND past the cap): see
                        # the single-CTA tail
                        if eq_count_s > remaining_s:
                            if cutlass.const_expr(self.approx_ties):
                                if eq_count_s > TIE_CAP:
                                    if cutlass.const_expr(top_k > TIE_CAP):
                                        if remaining_s <= TIE_CAP:
                                            eq_count_s = remaining_s
                                    else:
                                        eq_count_s = remaining_s
                            else:
                                if (~cutlass.Uint32(s_misc[8])) == cutlass.Uint32(
                                    s_misc[9]
                                ):
                                    if cutlass.const_expr(top_k > TIE_CAP):
                                        if remaining_s <= TIE_CAP:
                                            eq_count_s = remaining_s
                                    else:
                                        eq_count_s = remaining_s

                        # staged copy only up to the stage capacity (see the
                        # single-CTA tail); beyond it the overflow refinement
                        copy_cap_s = remaining_s
                        if copy_cap_s > TIE_CAP:
                            copy_cap_s = cutlass.Int32(TIE_CAP)
                        if eq_count_s <= copy_cap_s:
                            for t in range(tidx, eq_count_s, NUM_THREADS):
                                out_idx_row[gt_count_s + t] = s_tie_idx[t]
                                if cutlass.const_expr(self.has_values):
                                    out_val_row[gt_count_s + t] = self.value_from_key(
                                        s_tie_keys[t]
                                    )
                            for t in range(tidx + eq_count_s, remaining_s, NUM_THREADS):
                                out_idx_row[gt_count_s + t] = cutlass.Int32(-1)
                        else:
                            if eq_count_s <= TIE_CAP:
                                self.tie_select(
                                    s_tie_keys,
                                    s_tie_idx,
                                    eq_count_s,
                                    gt_count_s,
                                    remaining_s,
                                    s_hist,
                                    s_warp_sums,
                                    s_misc,
                                    out_idx_row,
                                    out_val_row,
                                    tidx,
                                )
                            else:
                                # Overflow: exact refinement (mirrors the
                                # single-CTA kernel: wide skippable rounds,
                                # collect-fused (12,12) histogram, early
                                # stage escape, terminal STAGE + tie_select).
                                prefix_s = cutlass.Uint32(0)
                                pmask_s = cutlass.Uint32(0)
                                total_s = eq_count_s
                                remain_s = remaining_s
                                staged_s = cutlass.Int32(0)
                                kmax_us = cutlass.Uint32(s_misc[9])
                                kmin_us = ~cutlass.Uint32(s_misc[8])
                                nr = cutlass.const_expr(len(self.refine_rounds))
                                for r in cutlass.range_constexpr(nr):
                                    rsh = cutlass.const_expr(self.refine_rounds[r][0])
                                    rbits = cutlass.const_expr(self.refine_rounds[r][1])
                                    rbins = cutlass.const_expr(1 << rbits)
                                    packed = cutlass.const_expr((rbits << 8) | rsh)
                                    skip_s = staged_s
                                    if (kmax_us >> cutlass.Uint32(rsh)) == (
                                        kmin_us >> cutlass.Uint32(rsh)
                                    ):
                                        # constant digit across all ties:
                                        # absorb it into the prefix for free
                                        # (see the single-CTA comment)
                                        if skip_s == 0:
                                            prefix_s = prefix_s | (
                                                (
                                                    (kmax_us >> cutlass.Uint32(rsh))
                                                    & cutlass.Uint32(rbins - 1)
                                                )
                                                << cutlass.Uint32(rsh)
                                            )
                                            pmask_s = pmask_s | (
                                                cutlass.Uint32(rbins - 1)
                                                << cutlass.Uint32(rsh)
                                            )
                                        skip_s = cutlass.Int32(1)
                                    if skip_s == 0:
                                        fused_s = cutlass.Int32(0)
                                        if cutlass.const_expr(
                                            _FUSE_COLLECT_R1
                                            and self.is_f32
                                            and rsh == 12
                                        ):
                                            if pmask_s == 0:
                                                fused_s = cutlass.Int32(1)
                                        if fused_s == 0:
                                            for i in range(tidx, rbins, NUM_THREADS):
                                                s_hist[i] = cutlass.Int32(0)
                                            cute.arch.barrier()
                                            self._stream_row(
                                                _OP_REFINE,
                                                packed,
                                                row_in,
                                                length,
                                                i0,
                                                tidx,
                                                s_hist,
                                                s_count_gt,
                                                s_count_eq,
                                                s_tie_keys,
                                                s_tie_idx,
                                                s_misc,
                                                out_idx_row,
                                                out_val_row,
                                                threshold_bin_s,
                                                prefix_s,
                                                pmask_s,
                                                u0,
                                                i0,
                                                s_misc,
                                            )
                                            cute.arch.barrier()
                                        if cutlass.const_expr(rbins == 256):
                                            self.scan256_and_find(
                                                s_hist,
                                                total_s,
                                                remain_s,
                                                s_warp_sums,
                                                s_misc,
                                                tidx,
                                            )
                                            bucket_s = s_misc[0]
                                            above_s2 = s_misc[1]
                                            cnt_s = s_misc[2]
                                        else:
                                            self.find_threshold_wide(
                                                s_hist,
                                                total_s,
                                                remain_s,
                                                s_warp_sums,
                                                s_misc,
                                                tidx,
                                            )
                                            bucket_s = s_misc[0]
                                            above_s2 = s_misc[1]
                                            cnt_s = s_misc[2]
                                        prefix_s = prefix_s | (
                                            cutlass.Uint32(bucket_s)
                                            << cutlass.Uint32(rsh)
                                        )
                                        pmask_s = pmask_s | (
                                            cutlass.Uint32(rbins - 1)
                                            << cutlass.Uint32(rsh)
                                        )
                                        total_s = cnt_s
                                        remain_s = remain_s - above_s2
                                        cute.arch.barrier()
                                        if total_s <= TIE_CAP:
                                            staged_s = cutlass.Int32(1)

                                if cutlass.const_expr(top_k > TIE_CAP):
                                    if remain_s > TIE_CAP:
                                        # ties exceed the staging buffer: all
                                        # refinement survivors are key-equal
                                        # (coverage proof), so masked-eq fill
                                        # straight to gmem needs no ordering
                                        if tidx == 0:
                                            s_misc[6] = cutlass.Int32(0)
                                        cute.arch.barrier()
                                        self._stream_row(
                                            _OP_EQFILL,
                                            0,
                                            row_in,
                                            length,
                                            i0,
                                            tidx,
                                            s_hist,
                                            s_count_gt,
                                            s_count_eq,
                                            s_tie_keys,
                                            s_tie_idx,
                                            s_misc,
                                            out_idx_row,
                                            out_val_row,
                                            threshold_bin_s,
                                            prefix_s,
                                            pmask_s,
                                            u0,
                                            top_k - remain_s,
                                            s_misc,
                                        )
                                    else:
                                        self._terminal_stage_select(
                                            row_in,
                                            length,
                                            i0,
                                            u0,
                                            tidx,
                                            s_hist,
                                            s_count_gt,
                                            s_count_eq,
                                            s_tie_keys,
                                            s_tie_idx,
                                            s_warp_sums,
                                            s_misc,
                                            out_idx_row,
                                            out_val_row,
                                            threshold_bin_s,
                                            prefix_s,
                                            pmask_s,
                                        )
                                else:
                                    self._terminal_stage_select(
                                        row_in,
                                        length,
                                        i0,
                                        u0,
                                        tidx,
                                        s_hist,
                                        s_count_gt,
                                        s_count_eq,
                                        s_tie_keys,
                                        s_tie_idx,
                                        s_warp_sums,
                                        s_misc,
                                        out_idx_row,
                                        out_val_row,
                                        threshold_bin_s,
                                        prefix_s,
                                        pmask_s,
                                    )
                        self._dbg_ts(g_state, 5, tidx)
            else:
                state_dirty = cutlass.Int32(1)
                # ---- zero the group counters for THIS row (rank 0) ----
                # Deliberately at row START, not in the pre-B4 cleanup: the
                # fast-arm CTAs read g_out/g_eq/g_kmax(n) between B3 and B4,
                # so a pre-B4 zeroing by rank 0 races with those loads.  A
                # torn read (eq pre-zero, kmax post-zero) flips one CTA's
                # uniform-tie clamp, diverging the group's barrier schedule
                # -- deadlock.  Here nothing reads or writes the counters
                # until the post-B2 flush, and B1/B2 publish the zeros.
                if rank == 0:
                    if tidx == 0:
                        (g_state + self.g_out).store(cutlass.Int32(0))
                        (g_state + self.g_eq).store(cutlass.Int32(0))
                        (g_state + self.g_eqf).store(cutlass.Int32(0))
                        (g_state + self.g_kmax).store(cutlass.Int32(0))
                        (g_state + self.g_kmaxn).store(cutlass.Int32(0))
                # ---- init local hist (vectorized 16B zero stores) ----
                s_hist_addr = s_hist.toint()
                for i in range(tidx * 4, self.hist_size, NUM_THREADS * 4):
                    st_shared_v4_zero(s_hist_addr + i * 4)
                cute.arch.barrier()

                # ---- local coarse histogram over my chunk ----
                if cutlass.const_expr(self._reg_path_ok(n_cols, self.chunk_elems)):
                    two_slots = cutlass.const_expr(
                        self._reg_two_slots(self.chunk_elems)
                    )
                    ve = cutlass.const_expr(self.vec_elems)
                    addr = chunk_in.toint()
                    num_full = local_len // ve
                    tail = local_len - num_full * ve
                    valid0 = tidx < num_full
                    valid1 = (tidx + NUM_THREADS) < num_full
                    a0 = cutlass.Uint32(0)
                    a1 = cutlass.Uint32(0)
                    a2 = cutlass.Uint32(0)
                    a3 = cutlass.Uint32(0)
                    b0 = cutlass.Uint32(0)
                    b1 = cutlass.Uint32(0)
                    b2 = cutlass.Uint32(0)
                    b3 = cutlass.Uint32(0)
                    if valid0:
                        a0, a1, a2, a3 = ld_global_v4_u32(
                            addr + cutlass.Int64(tidx) * 16
                        )
                    if cutlass.const_expr(two_slots):
                        if valid1:
                            b0, b1, b2, b3 = ld_global_v4_u32(
                                addr + cutlass.Int64(tidx + NUM_THREADS) * 16
                            )
                    tail_thread = tidx >= NUM_THREADS - tail
                    tail_idx = (
                        chunk_start + num_full * ve + (tidx - (NUM_THREADS - tail))
                    )
                    tbits = cutlass.Uint32(0)
                    if tail_thread:
                        tbits = self.load_scalar(row_in, tail_idx)

                    self._reg_row(
                        _OP_HIST,
                        0,
                        two_slots,
                        chunk_start,
                        valid0,
                        a0,
                        a1,
                        a2,
                        a3,
                        valid1,
                        b0,
                        b1,
                        b2,
                        b3,
                        tail_thread,
                        tbits,
                        tail_idx,
                        tidx,
                        s_hist,
                        s_count_gt,
                        s_count_eq,
                        s_tie_keys,
                        s_tie_idx,
                        s_misc,
                        out_idx_row,
                        out_val_row,
                        i0,
                        u0,
                        u0,
                        u0,
                        i0,
                        g_state,
                    )
                else:
                    self._stream_row(
                        _OP_HIST,
                        0,
                        chunk_in,
                        local_len,
                        chunk_start,
                        tidx,
                        s_hist,
                        s_count_gt,
                        s_count_eq,
                        s_tie_keys,
                        s_tie_idx,
                        s_misc,
                        out_idx_row,
                        out_val_row,
                        i0,
                        u0,
                        u0,
                        u0,
                        i0,
                        g_state,
                    )
                cute.arch.barrier()

                # ---- two-level global threshold search ----
                # Level 1: each CTA folds its local histogram into a 256-bin
                # super-histogram (super_w adjacent bins per bucket) and
                # merges THAT -- ~256 ints of global traffic per CTA instead
                # of hist_size.
                sw = cutlass.const_expr(self.super_w)
                if tidx < 256:
                    s = cutlass.Int32(0)
                    for j in cutlass.range(sw, unroll_full=True):
                        s = s + s_hist[tidx * sw + j]
                    if s > 0:
                        gmem_red_add(g_state + (self.g_super + tidx), s)
                phase = phase + 1
                mc_barrier(g_arrive, phase * cpg, tidx)  # B1

                if tidx < 256:
                    s_scan[tidx] = (g_state + (self.g_super + tidx)).load()
                cute.arch.barrier()
                self.scan256_and_find(
                    s_scan, length, cutlass.Int32(top_k), s_warp_sums, s_misc, tidx
                )
                super_bucket = s_misc[0]
                above_s = s_misc[1]
                count_s = s_misc[2]

                # Level 2: merge only the winning super-bucket's fine bins
                # (<= 32 ints per CTA); every CTA derives the same exact
                # coarse threshold bin from them with one warp scan.
                if tidx < sw:
                    v = s_hist[super_bucket * sw + tidx]
                    if v > 0:
                        gmem_red_add(g_state + (self.g_fine + tidx), v)
                phase = phase + 1
                mc_barrier(g_arrive, phase * cpg, tidx)  # B2

                if tidx < 32:
                    f = cutlass.Int32(0)
                    if tidx < sw:
                        f = (g_state + (self.g_fine + tidx)).load()
                    incl = warp_inclusive_sum(f, tidx)
                    above_f = above_s + (count_s - incl)
                    if tidx < sw:
                        if above_f < top_k and above_f + f >= top_k:
                            s_misc[0] = super_bucket * sw + tidx
                            s_misc[1] = above_f
                cute.arch.barrier()
                threshold_bin = s_misc[0]
                hi_b = u0
                lo_b = u0
                if cutlass.const_expr(self.boundary_cls):
                    hi_b = self.coarse_bin_gt_threshold_f32(threshold_bin + 1).bitcast(
                        cutlass.Uint32
                    )
                    lo_b = self.coarse_bin_lower_bound_f32(threshold_bin).bitcast(
                        cutlass.Uint32
                    )

                # ---- collect over my chunk (smem staging, batched emit) ----
                if tidx == 0:
                    s_misc[4] = cutlass.Int32(0)  # CTA-local gt count
                    s_misc[5] = cutlass.Int32(0)  # CTA-local tie count
                    s_misc[8] = cutlass.Int32(0)  # max(~tie key)
                    s_misc[9] = cutlass.Int32(0)  # max(tie key)
                cute.arch.barrier()
                if cutlass.const_expr(self._reg_path_ok(n_cols, self.chunk_elems)):
                    self._reg_row(
                        _OP_COLLECT_MC,
                        0,
                        cutlass.const_expr(self._reg_two_slots(self.chunk_elems)),
                        chunk_start,
                        valid0,
                        a0,
                        a1,
                        a2,
                        a3,
                        valid1,
                        b0,
                        b1,
                        b2,
                        b3,
                        tail_thread,
                        tbits,
                        tail_idx,
                        tidx,
                        s_hist,
                        s_count_gt,
                        s_count_eq,
                        s_tie_keys,
                        s_tie_idx,
                        s_misc,
                        out_idx_row,
                        out_val_row,
                        threshold_bin,
                        hi_b,
                        lo_b,
                        u0,
                        i0,
                        g_state,
                    )
                else:
                    self._stream_row(
                        _OP_COLLECT_MC,
                        0,
                        chunk_in,
                        local_len,
                        chunk_start,
                        tidx,
                        s_hist,
                        s_count_gt,
                        s_count_eq,
                        s_tie_keys,
                        s_tie_idx,
                        s_misc,
                        out_idx_row,
                        out_val_row,
                        threshold_bin,
                        hi_b,
                        lo_b,
                        u0,
                        i0,
                        g_state,
                    )
                # ---- flush the CTA-local stages with ONE device atomic per
                # counter: reserve contiguous group slots, then copy.  The
                # gt flush writes final output positions; the tie flush
                # fills the group's gmem tie buffer (complete iff the group
                # total is <= TIE_CAP, which is exactly when it is used).
                cute.arch.barrier()
                if tidx == 0:
                    cg = s_misc[4]
                    if cg > TIE_CAP:
                        # A NaN flood can overrun the gt stage capacity;
                        # reserving only the staged count keeps g_out
                        # consistent with what gets written (the excess
                        # NaNs degrade to tie backfill instead of leaving
                        # reserved-but-unwritten output slots).
                        cg = cutlass.Int32(TIE_CAP)
                    s_misc[1] = gmem_atomic_add(g_state + self.g_out, cg)
                    s_misc[2] = gmem_atomic_add(g_state + self.g_eq, s_misc[5])
                    if cutlass.const_expr(not self.approx_ties):
                        # publish this CTA's tie-key range to the group
                        gmem_red_max_u32(
                            g_state + self.g_kmax, cutlass.Uint32(s_misc[9])
                        )
                        gmem_red_max_u32(
                            g_state + self.g_kmaxn, cutlass.Uint32(s_misc[8])
                        )
                cute.arch.barrier()
                base_gt = s_misc[1]
                base_eq = s_misc[2]
                cnt_gt = s_misc[4]
                if cnt_gt > TIE_CAP:
                    cnt_gt = cutlass.Int32(TIE_CAP)
                cnt_eq = s_misc[5]
                if cnt_eq > TIE_CAP:
                    cnt_eq = cutlass.Int32(TIE_CAP)
                for t in range(tidx, cnt_gt, NUM_THREADS):
                    pos = base_gt + t
                    if pos < top_k:
                        out_idx_row[pos] = s_tie_idx[t]
                        if cutlass.const_expr(self.has_values):
                            out_val_row[pos] = self.value_from_key(s_tie_keys[t])
                for t in range(tidx, cnt_eq, NUM_THREADS):
                    c = base_eq + t
                    if c < TIE_CAP:
                        (g_state + (self.g_tiek + c)).store(s_hist[t])
                        (g_state + (self.g_tiei + c)).store(s_hist[TIE_CAP + t])
                phase = phase + 1
                mc_barrier(g_arrive, phase * cpg, tidx)  # B3

                gt_count = (g_state + self.g_out).load()
                eq_count = (g_state + self.g_eq).load()
                remaining = top_k - gt_count
                if remaining < 0:
                    # NaN flood: gt overran top_k (guarded stores dropped
                    # the excess).  Group-uniform (gmem reads post-B3).
                    remaining = cutlass.Int32(0)
                    eq_count = cutlass.Int32(0)

                # uniform-tie overflow (group-uniform: gmem reads post-B3):
                # clamp so rank 0's direct staged-copy arm handles it, and
                # the multi-round gmem refinement is skipped.
                if eq_count > TIE_CAP:
                    if cutlass.const_expr(self.approx_ties):
                        if cutlass.const_expr(top_k > TIE_CAP):
                            if remaining <= TIE_CAP:
                                eq_count = remaining
                        else:
                            eq_count = remaining
                    else:
                        kmax_g = cutlass.Uint32(
                            (g_state + self.g_kmax).load().bitcast(cutlass.Uint32)
                        )
                        kmaxn_g = cutlass.Uint32(
                            (g_state + self.g_kmaxn).load().bitcast(cutlass.Uint32)
                        )
                        if (~kmaxn_g) == kmax_g:
                            if cutlass.const_expr(top_k > TIE_CAP):
                                if remaining <= TIE_CAP:
                                    eq_count = remaining
                            else:
                                eq_count = remaining

                ovf = cutlass.Int32(0)  # group-uniform: overflow rounds ran
                if eq_count <= TIE_CAP:
                    # rank 0 resolves the ties alone; other ranks go wait at
                    # B4 (the decision is group-uniform).
                    if rank == 0:
                        if eq_count <= remaining:
                            for t in range(tidx, eq_count, NUM_THREADS):
                                out_idx_row[gt_count + t] = (
                                    g_state + (self.g_tiei + t)
                                ).load()
                                if cutlass.const_expr(self.has_values):
                                    out_val_row[gt_count + t] = self.value_from_key(
                                        (g_state + (self.g_tiek + t))
                                        .load()
                                        .bitcast(cutlass.Uint32)
                                    )
                            for t in range(tidx + eq_count, remaining, NUM_THREADS):
                                out_idx_row[gt_count + t] = cutlass.Int32(-1)
                        else:
                            # Copy staged candidates to smem, exact select.
                            for t in range(tidx, eq_count, NUM_THREADS):
                                s_tie_keys[t] = (
                                    (g_state + (self.g_tiek + t))
                                    .load()
                                    .bitcast(cutlass.Uint32)
                                )
                                s_tie_idx[t] = (g_state + (self.g_tiei + t)).load()
                            cute.arch.barrier()
                            self.tie_select(
                                s_tie_keys,
                                s_tie_idx,
                                eq_count,
                                gt_count,
                                remaining,
                                s_hist,
                                s_warp_sums,
                                s_misc,
                                out_idx_row,
                                out_val_row,
                                tidx,
                            )
                else:
                    # ---- overflow: cooperative exact refinement ----
                    # Same structure as the single-CTA path (wide skippable
                    # rounds, early stage escape, terminal STAGE +
                    # tie_select) with per-round gmem merge slots; every
                    # skip/stage decision derives from group-uniform values
                    # (threshold_bin, merged counts) so the mc_barrier phase
                    # schedule stays identical across ranks.  No collect-
                    # fused round here: s_hist stages ties during the MC
                    # collect.
                    ovf = cutlass.Int32(1)
                    prefix = cutlass.Uint32(0)
                    prefix_mask = cutlass.Uint32(0)
                    total = eq_count
                    remain = remaining
                    staged = cutlass.Int32(0)
                    # group-merged tie-key range (post-B3, group-uniform):
                    # drives the per-row exact round skip (see the
                    # single-CTA comment); every rank derives the same skip
                    # schedule, so the mc_barrier phases stay aligned.
                    kmax_u = cutlass.Uint32(
                        (g_state + self.g_kmax).load().bitcast(cutlass.Uint32)
                    )
                    kmin_u = ~cutlass.Uint32(
                        (g_state + self.g_kmaxn).load().bitcast(cutlass.Uint32)
                    )
                    num_refine = cutlass.const_expr(len(self.refine_rounds))
                    for r in cutlass.range_constexpr(num_refine):
                        rsh = cutlass.const_expr(self.refine_rounds[r][0])
                        rbits = cutlass.const_expr(self.refine_rounds[r][1])
                        rbins = cutlass.const_expr(1 << rbits)
                        packed = cutlass.const_expr((rbits << 8) | rsh)
                        rh = cutlass.const_expr(self.refine_slot_off[r])
                        skip = staged
                        if (kmax_u >> cutlass.Uint32(rsh)) == (
                            kmin_u >> cutlass.Uint32(rsh)
                        ):
                            # constant digit across all ties: absorb it into
                            # the prefix for free
                            if skip == 0:
                                prefix = prefix | (
                                    (
                                        (kmax_u >> cutlass.Uint32(rsh))
                                        & cutlass.Uint32(rbins - 1)
                                    )
                                    << cutlass.Uint32(rsh)
                                )
                                prefix_mask = prefix_mask | (
                                    cutlass.Uint32(rbins - 1) << cutlass.Uint32(rsh)
                                )
                            skip = cutlass.Int32(1)
                        if skip == 0:
                            for i in range(tidx, rbins, NUM_THREADS):
                                s_hist[i] = cutlass.Int32(0)
                            cute.arch.barrier()
                            if cutlass.const_expr(
                                self._reg_path_ok(n_cols, self.chunk_elems)
                            ):
                                self._reg_row(
                                    _OP_REFINE,
                                    packed,
                                    cutlass.const_expr(
                                        self._reg_two_slots(self.chunk_elems)
                                    ),
                                    chunk_start,
                                    valid0,
                                    a0,
                                    a1,
                                    a2,
                                    a3,
                                    valid1,
                                    b0,
                                    b1,
                                    b2,
                                    b3,
                                    tail_thread,
                                    tbits,
                                    tail_idx,
                                    tidx,
                                    s_hist,
                                    s_count_gt,
                                    s_count_eq,
                                    s_tie_keys,
                                    s_tie_idx,
                                    s_misc,
                                    out_idx_row,
                                    out_val_row,
                                    threshold_bin,
                                    prefix,
                                    prefix_mask,
                                    u0,
                                    i0,
                                    g_state,
                                )
                            else:
                                self._stream_row(
                                    _OP_REFINE,
                                    packed,
                                    chunk_in,
                                    local_len,
                                    chunk_start,
                                    tidx,
                                    s_hist,
                                    s_count_gt,
                                    s_count_eq,
                                    s_tie_keys,
                                    s_tie_idx,
                                    s_misc,
                                    out_idx_row,
                                    out_val_row,
                                    threshold_bin,
                                    prefix,
                                    prefix_mask,
                                    u0,
                                    i0,
                                    g_state,
                                )
                            cute.arch.barrier()
                            # Merge my sub-bins into this round's gmem slot
                            # (occupied bins only), then read the group total
                            # back.
                            for i in range(tidx, rbins, NUM_THREADS):
                                v = s_hist[i]
                                if v > 0:
                                    gmem_red_add(g_state + (rh + i), v)
                            phase = phase + 1
                            mc_barrier(g_arrive, phase * cpg, tidx)
                            for i in range(tidx, rbins, NUM_THREADS):
                                s_hist[i] = (g_state + (rh + i)).load()
                            cute.arch.barrier()
                            if cutlass.const_expr(rbins == 256):
                                self.scan256_and_find(
                                    s_hist, total, remain, s_warp_sums, s_misc, tidx
                                )
                                bucket = s_misc[0]
                                above = s_misc[1]
                                cnt = s_misc[2]
                            else:
                                self.find_threshold_wide(
                                    s_hist, total, remain, s_warp_sums, s_misc, tidx
                                )
                                bucket = s_misc[0]
                                above = s_misc[1]
                                cnt = s_misc[2]
                            prefix = prefix | (
                                cutlass.Uint32(bucket) << cutlass.Uint32(rsh)
                            )
                            prefix_mask = prefix_mask | (
                                cutlass.Uint32(rbins - 1) << cutlass.Uint32(rsh)
                            )
                            total = cnt
                            remain = remain - above
                            cute.arch.barrier()
                            if total <= TIE_CAP:
                                staged = cutlass.Int32(1)
                    # ---- terminal stage: emit prefix-above through g_out,
                    # stage prefix-matching (full key + index) into the gmem
                    # tie buffer through g_eqf (zeroed at row start; FINAL_MC
                    # no longer runs).  Exact even with a partial prefix
                    # (staged keys are full) and even capped (survivors of
                    # ALL rounds are provably key-identical).
                    #
                    # Big-k (top_k > TIE_CAP) only: when even the post-round
                    # remaining fill exceeds the stage buffer, EQFILL_MC
                    # writes prefix-equal elements straight to the output
                    # through the shared g_eqf cursor (survivors are
                    # key-identical, so first-arrival order is a valid
                    # answer) and rank 0 skips the gather/select.  The
                    # branch is group-uniform: remain is group-uniform.
                    if cutlass.const_expr(top_k > TIE_CAP):
                        filled = cutlass.Int32(0)
                        if remain > TIE_CAP:
                            filled = cutlass.Int32(1)
                            if cutlass.const_expr(
                                self._reg_path_ok(n_cols, self.chunk_elems)
                            ):
                                self._reg_row(
                                    _OP_EQFILL_MC,
                                    0,
                                    cutlass.const_expr(
                                        self._reg_two_slots(self.chunk_elems)
                                    ),
                                    chunk_start,
                                    valid0,
                                    a0,
                                    a1,
                                    a2,
                                    a3,
                                    valid1,
                                    b0,
                                    b1,
                                    b2,
                                    b3,
                                    tail_thread,
                                    tbits,
                                    tail_idx,
                                    tidx,
                                    s_hist,
                                    s_count_gt,
                                    s_count_eq,
                                    s_tie_keys,
                                    s_tie_idx,
                                    s_misc,
                                    out_idx_row,
                                    out_val_row,
                                    threshold_bin,
                                    prefix,
                                    prefix_mask,
                                    u0,
                                    top_k - remain,
                                    g_state,
                                )
                            else:
                                self._stream_row(
                                    _OP_EQFILL_MC,
                                    0,
                                    chunk_in,
                                    local_len,
                                    chunk_start,
                                    tidx,
                                    s_hist,
                                    s_count_gt,
                                    s_count_eq,
                                    s_tie_keys,
                                    s_tie_idx,
                                    s_misc,
                                    out_idx_row,
                                    out_val_row,
                                    threshold_bin,
                                    prefix,
                                    prefix_mask,
                                    u0,
                                    top_k - remain,
                                    g_state,
                                )
                        else:
                            if cutlass.const_expr(
                                self._reg_path_ok(n_cols, self.chunk_elems)
                            ):
                                self._reg_row(
                                    _OP_STAGE_MC,
                                    0,
                                    cutlass.const_expr(
                                        self._reg_two_slots(self.chunk_elems)
                                    ),
                                    chunk_start,
                                    valid0,
                                    a0,
                                    a1,
                                    a2,
                                    a3,
                                    valid1,
                                    b0,
                                    b1,
                                    b2,
                                    b3,
                                    tail_thread,
                                    tbits,
                                    tail_idx,
                                    tidx,
                                    s_hist,
                                    s_count_gt,
                                    s_count_eq,
                                    s_tie_keys,
                                    s_tie_idx,
                                    s_misc,
                                    out_idx_row,
                                    out_val_row,
                                    threshold_bin,
                                    prefix,
                                    prefix_mask,
                                    u0,
                                    i0,
                                    g_state,
                                )
                            else:
                                self._stream_row(
                                    _OP_STAGE_MC,
                                    0,
                                    chunk_in,
                                    local_len,
                                    chunk_start,
                                    tidx,
                                    s_hist,
                                    s_count_gt,
                                    s_count_eq,
                                    s_tie_keys,
                                    s_tie_idx,
                                    s_misc,
                                    out_idx_row,
                                    out_val_row,
                                    threshold_bin,
                                    prefix,
                                    prefix_mask,
                                    u0,
                                    i0,
                                    g_state,
                                )
                        # The barrier keeps rank 0 from reading the counters
                        # (and later cleanup from touching state) while
                        # other ranks are still emitting.
                        phase = phase + 1
                        mc_barrier(g_arrive, phase * cpg, tidx)  # B3b
                        if rank == 0:
                            if filled == 0:
                                self._mc_gather_select(
                                    g_state,
                                    tidx,
                                    s_hist,
                                    s_tie_keys,
                                    s_tie_idx,
                                    s_warp_sums,
                                    s_misc,
                                    out_idx_row,
                                    out_val_row,
                                )
                    else:
                        if cutlass.const_expr(
                            self._reg_path_ok(n_cols, self.chunk_elems)
                        ):
                            self._reg_row(
                                _OP_STAGE_MC,
                                0,
                                cutlass.const_expr(
                                    self._reg_two_slots(self.chunk_elems)
                                ),
                                chunk_start,
                                valid0,
                                a0,
                                a1,
                                a2,
                                a3,
                                valid1,
                                b0,
                                b1,
                                b2,
                                b3,
                                tail_thread,
                                tbits,
                                tail_idx,
                                tidx,
                                s_hist,
                                s_count_gt,
                                s_count_eq,
                                s_tie_keys,
                                s_tie_idx,
                                s_misc,
                                out_idx_row,
                                out_val_row,
                                threshold_bin,
                                prefix,
                                prefix_mask,
                                u0,
                                i0,
                                g_state,
                            )
                        else:
                            self._stream_row(
                                _OP_STAGE_MC,
                                0,
                                chunk_in,
                                local_len,
                                chunk_start,
                                tidx,
                                s_hist,
                                s_count_gt,
                                s_count_eq,
                                s_tie_keys,
                                s_tie_idx,
                                s_misc,
                                out_idx_row,
                                out_val_row,
                                threshold_bin,
                                prefix,
                                prefix_mask,
                                u0,
                                i0,
                                g_state,
                            )
                        # STAGE_MC emits through the shared g_out/g_eqf
                        # atomics from EVERY rank; the barrier keeps rank 0
                        # from reading the counters (and later cleanup from
                        # touching state) while other ranks are still
                        # emitting.  Group-uniform: the overflow branch is
                        # taken by every rank or none.
                        phase = phase + 1
                        mc_barrier(g_arrive, phase * cpg, tidx)  # B3b

                        # rank 0 resolves the staged candidates alone
                        # (mirrors the eq_count <= TIE_CAP arm); others fall
                        # through to the cleanup + B4.
                        if rank == 0:
                            self._mc_gather_select(
                                g_state,
                                tidx,
                                s_hist,
                                s_tie_keys,
                                s_tie_idx,
                                s_warp_sums,
                                s_misc,
                                out_idx_row,
                                out_val_row,
                            )

                # ---- row cleanup (rank 0), published by B4 ----
                # Clears the super/fine histograms ([0, 320)) always, and
                # the (large) refine-merge slots only when the overflow
                # rounds actually dirtied them.  The counters
                # (g_out..g_kmaxn) are NOT cleared here -- fast-arm CTAs may
                # still be loading them between B3 and B4; they are zeroed
                # at the start of the next grouped row instead.
                if rank == 0:
                    for i in range(tidx, self.g_rhist, NUM_THREADS):
                        (g_state + i).store(cutlass.Int32(0))
                    if ovf == 1:
                        for i in range(tidx + self.g_rhist, self.g_arrive, NUM_THREADS):
                            (g_state + i).store(cutlass.Int32(0))
                phase = phase + 1
                mc_barrier(g_arrive, phase * cpg, tidx)  # B4

            row = row + num_groups

        # ---- end-of-kernel self-reset (CUDA-graph replay safety) ----
        # rank 0 re-clears the histogram region (already cleared per row; the
        # redundant sweep doubles as deliberate slack so the other CTAs of
        # the group observe the final barrier before the arrival counter is
        # reset -- same protocol and caveat as the ``radix`` backend), then
        # release-stores the arrival counter back to zero.  The bar.sync
        # orders the clears (from all of rank 0's warps) before the release.
        if state_dirty == 1:
            # Departure handshake.  The reset below zeroes the arrival
            # counter every peer spins on; a peer descheduled while still
            # spinning on the final barrier (GPU time-slicing with another
            # process -- observed on A100 and RTX 5080) would otherwise see
            # the counter reset underneath it and spin forever.  So every
            # peer announces that it is past its last barrier and its last
            # g_state access, and rank 0 waits for all of them first.  Rank
            # 0 only waits for CTAs that have nothing left to wait for, so
            # no cycle is possible; a descheduled peer just delays the
            # reset.  state_dirty is group-uniform (every rank took the
            # same per-row branches), so signal and wait always pair up.
            g_depart = g_state + self.g_depart
            if rank != 0:
                cute.arch.barrier()  # all of this CTA's g_state traffic is done
                if tidx == 0:
                    cute.arch.red(
                        g_depart,
                        cutlass.Int32(1),
                        op="add",
                        dtype="s32",
                        sem="release",
                        scope="gpu",
                    )
            else:
                if tidx == 0:
                    while cute.arch.load(
                        g_depart, cutlass.Int32, sem="acquire", scope="gpu"
                    ) < cutlass.Int32(cpg - 1):
                        pass
                cute.arch.barrier()
                # Only sweep when this launch actually entered the group
                # path: untouched state is still all-zero (worth ~0.5us at
                # b=1).  The bar.sync orders the clears (from all of rank 0's
                # warps) before the release-stores of both counters.
                for i in range(tidx, self.g_arrive, NUM_THREADS):
                    (g_state + i).store(cutlass.Int32(0))
                cute.arch.barrier()
                if tidx == 0:
                    cute.arch.store(
                        g_depart, cutlass.Int32(0), sem="release", scope="gpu"
                    )
                    cute.arch.store(
                        g_arrive, cutlass.Int32(0), sem="release", scope="gpu"
                    )

        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_launch_dependents()

    # ------------------------------------------------------------------
    # Host-side launcher
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        input_data: cute.Tensor,
        row_states: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
        page_table: cute.Tensor,
        stream,
    ):
        num_rows = input_data.shape[0]
        # both launchers run NUM_THREADS; nt != 1024 is the walk-first
        # subclass's shape, served by its own launcher (launch_prod)
        assert self.nt == NUM_THREADS, self.nt
        if cutlass.const_expr(self.ctas_per_group == 1):
            self.topk_kernel(
                input_data, seqlen, output_indices, output_values, page_table
            ).launch(
                grid=(num_rows, 1, 1),
                block=(NUM_THREADS, 1, 1),
                stream=stream,
                use_pdl=_ENABLE_PDL,
                min_blocks_per_mp=self.min_blocks_per_mp,
            )
        else:
            cpg = cutlass.const_expr(self.ctas_per_group)
            num_groups = min(self.num_sms // cpg, num_rows)
            # min_blocks_per_mp=1: the grid never exceeds one CTA per SM, and
            # the launch bound gives ptxas the 64-register budget the solo
            # arm's four-slot register path needs (unset, the single-CTA
            # kernel was squeezed to 32 registers with spills)
            self.mc_topk_kernel(
                input_data, row_states, seqlen, output_indices, output_values
            ).launch(
                grid=(num_groups * cpg, 1, 1),
                block=(NUM_THREADS, 1, 1),
                stream=stream,
                use_pdl=_ENABLE_PDL,
                min_blocks_per_mp=1,
            )
