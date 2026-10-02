"""Walk-first top-k on the primitives substrate -- a full
replication of gvr_2's streaming architecture, plus two things it lacks:
row-splitting at k > 1024 and an exact fallback.

gvr_2's structural speed (measured and source-analyzed) comes from
inverting the decide-first order a sample-calibrated selector uses: it
never pays a deliberation phase before the walk.

  1. PRE-SCALARS (~1us, deterministic): a 1024-element register sample ->
     smem min/max -> one 256-bin value histogram of the SAMPLE -> a
     provisional threshold TF at the bin whose sample rank targets
     ``aim ~ 2.5k`` survivors (conservatively LOW edge: overshooting the
     candidate count is recoverable, undershooting k is not), and a
     survivor-bin scale SC = 256 / (smax - TF).
  2. ONE FUSED WALK: element v survives iff NOT (v <= TF) (NaN survives,
     torch parity).  Each survivor is staged into the CTA-local candidate
     buffer AND binned into a CTA-local 256-bin survivor histogram --
     both per-element costs are a couple of instructions, no barriers.
     Per CTA at slice end: one gmem atomic reserves a slab range (bulk,
     never per-hit -- the serialization lesson), one bulk copy stages
     out, 256 red.adds merge the survivor histogram into the row's gmem
     histogram (kept in the slab's unused tail).
  3. POST-WALK SELECT (last-arriver CTA, no spins anywhere): with cnt
     survivors, k <= cnt <= SCAP and slots == cnt required (else status
     -> the gated exact fallback re-solves; duplicate floods land there
     by design).  The row's survivor histogram locates the bin B
     containing descending rank k AMONG CANDIDATES; candidates in bins
     above B are winners; rank ties WITHIN bin B (a value range, not a
     single value) get an exact key-space sub-select (ballot for small
     sets, byte-radix with degenerate-round skipping otherwise).
     Exactness never depends on TF or bin quality: candidates are
     exactly {v : !(v <= TF)} with an exact count, and the sub-select is
     key-exact -- TF/SC only steer performance.

mc_state: [0]=cand count, [1]=slab slots, [2]=arrive.  The row's
256-bin survivor histogram lives at slab_k[2*GCAP .. 2*GCAP+256), i.e.
the tail of the WF_ROW_INTS-wide slab row (zeroed by the epilogue's
self-reset; zero-init at first allocation).  Scope:
fp32 / fp16 / bf16, next_n == compress_ratio == 1, return_values=False,
k <= TIE_CAP, N a multiple of the 16-byte vector (4 fp32 or 8 16-bit
elements).  Outside the unpaged ``auto`` ranking; ``auto`` picks it first
for paged output.

NaN order: fp32 ranks every NaN top like torch.topk, whatever its sign
(the census bins through cvt.rn.f16.f32, which canonicalises NaN; the
register arm's float compares and the walk's ``not (v <= TF)`` survivor
test agree; tested with payload floods of either sign).  For the 16-bit
dtypes the short-row arms bin the integer key, where a sign-set NaN
pattern lands below -inf (bottom), while the walk pipeline ranks it top;
real logits do not carry negative NaNs, and canonicalising the 16-bit
sign would cost the hot census an instruction per element.  Should an
arm's winner count ever disagree with its census, the tail routes the row
to the exact fallback instead of dropping a winner.
"""

import os

import cutlass
import torch
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

from . import fallback_topk_primitives as _fallback_mod
from . import gvr2_topk_decode as _gvr2_mod
from . import radix_topk_primitives as _radix_mod
from .fallback_topk_primitives import GCAP, GatedExactFallback
from .radix_topk_primitives import (
    CoarseHistTopKPrimitivesKernel,
    gmem_atomic_add,
    gmem_red_add,
    read_clock64,
    smem_atomic_add,
    smem_red_max_u32,
    st_shared_v4_zero,
    warp_inclusive_sum,
)
from .gvr2_topk_decode import (
    RES_B,
    RES_B2,
    _atom_shared_cluster_add_i32,
    _cluster_sync_aligned,
    _ld_shared_cluster_i32,
    _mapa_shared_cluster,
    _pin_i32,
    _pin_i64,
    _st_shared_cluster_i32,
    lds128_i32,
    smem_atom_i32_128,
    sts128_i32,
    warp_incl_scan_add,
    warp_max_u32,
)

# DSv4-scale constants (sized for rows up to 1M): the original 64-128K
# era values (SCAP 8192 / LCAP 4096 / PSAMP 1024) collapsed at >=512K --
# the bracket rode on ~4 effective samples, candidates overflowed SCAP,
# and nearly every row fell back (382us at 1M vs gvr_2's 83).
SCAP = 16384  # candidate capacity == GCAP (full slab prefix)
# Short-row arms (see the SHORT-ROW ARM comment in the kernel): rows at or
# below WF_SMALL_N skip the walk pipeline entirely; within that domain rows
# at or below WF_CENSUS_N take the census arm (two L1-hot passes) and longer
# rows the register-resident arm (one read into registers, up to 4 vectors
# per thread).  Crossover measured (B200, 16 rows, K=512 and 2048, widths
# 16K and 64K, graph replay, vs SGLang's kernel): census wins at <= 4096
# (3.36 vs 3.77 us at 2K), register wins from 5000 up (4.18 vs 4.59 at 5K;
# 5.41 vs 7.05 census / 6.64 SGLang at 16K; with the census alone the bound
# had to stop at 8192: 7.45 vs the walk's 7.24 at 16K b=64).
WF_SMALL_N = 16384
WF_CENSUS_N = 4096
# The first WF_TIE_RED_DIRECT ties of a row publish their exact key to the tie
# range slots with direct smem reds (no fixed cost on tie-free rows); later
# ties fold into per-thread registers and are published once per warp (see
# _wf_publish_tie_range).  One warp's worth: a Gaussian crossing bin holds
# tens of ties, a plateau thousands.
WF_TIE_RED_DIRECT = 32
# paged output: the leading entries of the request's page-table row are staged
# in shared memory (at most this many, 8 KB, and at most four per thread);
# columns beyond the staged pages gather from global memory (wf_pt_cache).
# 8 KB keeps the K=512 kernel under the 100 KB smem carveout step; a 16 KB
# stage for the 256K-column cluster rows was measured (+0.41 us either way:
# the wider carveout takes L1 from the walk's survivor re-reads).
WF_PT_CACHE_MAX = 2048
LCAP = 8192  # per-CTA local candidate stage (pairs); smem ~83KB total
# doubled stage where the device has the shared memory (>= LCAP_BIG_SMEM
# bytes per block): ~147KB total, still 1 CTA/SM (registers pin occupancy),
# and the overshoot re-walk class disappears (see get_walkfirst_kernel)
LCAP_BIG = 16384
LCAP_BIG_SMEM = 160 * 1024
PSAMP = 4096  # pre-scalar sample size (one float4 per thread)
# wf slab row layout: [0..GCAP) keys, [GCAP..2*GCAP) idx, then the
# 256-int row survivor histogram (SCAP == GCAP leaves no prefix slack)
WF_SMAX = 32  # max row split supported by the gmem path
WF_TBL = 260  # per-CTA publish table: base, staged, prefix[0..256], pad
WF_ROW_INTS = 2 * GCAP + 256 + WF_SMAX * WF_TBL
# survivor-histogram span past the sample max (see the sample block)
WF_SPAN_EXT = 1.5

from cutlass._mlir import ir  # noqa: E402
from cutlass._mlir.extras import types as T  # noqa: E402


@dsl_user_op
def _wf_exit(*, loc=None, ip=None) -> None:
    """PTX ``exit``: ends the calling thread.  The short-row arms exit from
    their inline epilogue (CTA-uniform path, no barrier follows)."""
    llvm.inline_asm(
        None,
        [],
        "exit;",
        "",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_nc_v4_u32(gmem_addr: cutlass.Int64, *, loc=None, ip=None):
    """128-bit NON-COHERENT vectorized load: ld.global.nc.v4.b32.

    SASS comparison against gvr_2's streaming kernel showed its walk loads
    all compile to LDG.E.128.CONSTANT (the read-only data path, which
    sustains far more outstanding loads than the coherent LDG pipeline);
    ours were plain LDG.E.128 and moved ~1.6x fewer bytes/us.  The row is
    strictly read-only during the kernel, so .nc is safe here."""
    st = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
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


# Shared binning arithmetic: MUST be textually identical between wf_bin_u8
# (epilogue emit) and wf_emit_pred (walk histogram) -- any rounding
# difference between the two shifts the rank accounting.  NaN -> bin 255
# (torch sorts NaN on top; the saturating convert alone would send it to 0).
_WF_BIN_PTX = (
    "sub.rn.f32 d, {v}, {tf};\n"
    "mul.rn.f32 m, d, {sc};\n"
    "cvt.rzi.u32.f32 {b}, m;\n"
    "min.u32 {b}, {b}, 255;\n"
    "setp.neu.f32 q, {v}, {v};\n"
    "@q mov.u32 {b}, 255;\n"
)


@dsl_user_op
def wf_bin_u8(
    vf: cutlass.Float32, tf: cutlass.Float32, sc: cutlass.Float32, *, loc=None, ip=None
):
    """Survivor bin in [0,255] via the exact _WF_BIN_PTX sequence."""
    res = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Float32(vf).ir_value(loc=loc, ip=ip),
            cutlass.Float32(tf).ir_value(loc=loc, ip=ip),
            cutlass.Float32(sc).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .f32 d, m;\n.reg .pred q;\n"
        + _WF_BIN_PTX.format(v="$1", tf="$2", sc="$3", b="$0")
        + "}",
        "=r,f,f,f",
        has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(res)


@dsl_user_op
def wf_pack_tf16(tf: cutlass.Float32, bf16: bool, *, loc=None, ip=None):
    """fp32 walk threshold -> both halves of a 32-bit word holding the
    LARGEST 16-bit-grid value <= tf (cvt.rm), so ``x <= tf16`` in the
    packed compare equals ``f32(x) <= tf`` for every grid value x."""
    ty = "bf16" if bf16 else "f16"
    res = llvm.inline_asm(
        T.i32(),
        [cutlass.Float32(tf).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.rm." + ty + ".f32 h, $1;\nmov.b32 $0, {h, h};\n}",
        "=r,f",
        has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(res)


@dsl_user_op
def wf_dead_v4_16(
    dead: cutlass.Int32,
    w0: cutlass.Uint32,
    w1: cutlass.Uint32,
    w2: cutlass.Uint32,
    w3: cutlass.Uint32,
    tf2: cutlass.Uint32,
    base: int,
    bf16: bool,
    *,
    loc=None,
    ip=None,
):
    """Classify one 16-byte vector of 8 packed 16-bit elements: sets dead
    bit ``base + e`` for every element with x <= tf (4 packed setp + 8
    predicated ORs; element e = 2*word + half, low half first)."""
    ty = "bf16x2" if bf16 else "f16x2"
    body = "{\n.reg .pred p0, p1, p2, p3, p4, p5, p6, p7;\nmov.b32 $0, $1;\n"
    for j in range(4):
        # PTX spells the two-predicate destination as ``p|q``
        body += f"setp.le.{ty} p{2 * j}|p{2 * j + 1}, ${2 + j}, $6;\n"
    for e in range(8):
        body += f"@p{e} or.b32 $0, $0, 0x{(1 << (base + e)) & 0xFFFFFFFF:08x};\n"
    body += "}"
    res = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Int32(dead).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w0).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w1).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w2).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w3).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(tf2).ir_value(loc=loc, ip=ip),
        ],
        body,
        "=r,r,r,r,r,r,r",
        has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(res)


class WalkFirstTopK(CoarseHistTopKPrimitivesKernel):
    """See module docstring.  Inherits the substrate's key helpers and the
    tie selects; the exact fallback comes in through ProdWalkFirstTopK's
    GatedExactFallback mixin."""

    mc_splits: int = 2
    # Per-instance capacity knobs (defaults = the 1024-thread family; the
    # k<=1024 512-thread family halves them -- see get_walkfirst_kernel):
    lcap: int = LCAP
    psamp: int = PSAMP
    # Compile-time phase-telemetry switch (plain Python bool, read at
    # trace time): when False (production default) every
    # read_clock64() and the status-buffer phase writes (blocks 5-9)
    # are never traced -- zero instructions in the compiled kernel.
    # Enabled per-specialization by get_walkfirst_kernel(telemetry=True)
    # or FLASHINFER_TOPK_PRIM_TELEMETRY=1 (the one debug build flag).
    wf_telemetry: bool = False
    # Ampere/Ada (cc 8.x) build: ties past WF_TIE_RED_DIRECT fold into
    # per-thread registers and are published once per warp instead of two
    # same-address smem reds per tie (see _wf_note_tie / _wf_publish_tie_range).
    # Set by get_walkfirst_kernel from the device's compute capability.
    wf_tie_fold: bool = False
    # Cluster epilogue (SM90+, S in {2,3,4,6}): the S CTAs of a row form one
    # hardware cluster and coordinate through DSMEM instead of the gmem
    # slab + release/acquire arrival.  Set by get_walkfirst_kernel from
    # the device capability.
    wf_cluster: bool = False
    # short-row arm cutoff (rows at or below this length take the arms;
    # above it, the walk pipeline).  Compile-time.
    wf_small_n: int = WF_SMALL_N
    # packed-pair 16-bit walk classify (setp.le.f16x2 / .bf16x2); the
    # factory enables it for fp16 everywhere and bf16 on sm_90+
    wf_pair16: bool = False
    # Short-row arm tiers.  Rows <= wf_census_n: the census arm
    # (_wf_short_row, two L1-hot passes).  wf_census_n < rows <= wf_small_n:
    # the register-resident arm (_wf_reg_row: one read of the row bounded by
    # the real length, census + classify from registers -- SGLang's
    # TopKRegister shape).  MEASURED 2026-09-29 (B200, graph replay, vs
    # SGLang's kernel): on 300-2000-element decode rows the census arm is
    # faster (3.45 vs register 3.58 us; its second pass is L1-resident, so
    # register residency buys nothing and the arm's extra barrier + mask-walk
    # emit cost 0.1 us), from 5K up the register arm is (4.18 vs 4.79 at 5K,
    # 5.41 vs 7.05 at 16K, SGLang 4.59 / 6.64).
    wf_reg_arm: bool = True
    wf_census_n: int = WF_CENSUS_N
    # vectors per thread the register arm holds; the factory sizes it so
    # wf_small_n elements fit: ceil(wf_small_n / (nt * vec_elems)), at most 4
    wf_reg_vpt: int = 4
    # census arm: copy the staged prefix when all ties share one exact key
    # (ReLU-style rows: 6.07 -> 3.80 us).
    wf_uniform_tie: bool = True
    # Paged output (the API's page_table argument, SGLang's DSA decode
    # contract): 0 = raw columns; a power of two = every selected column c of
    # row q is stored as page_table[q, c >> lg] << lg | (c & (page_size - 1)),
    # the physical KV slot, -1 padding unchanged.  Compile-time (one kernel
    # variant per page size).  Emits go to a top_k-int smem stage (_wf_out)
    # and one coalesced pass at the end of the arm stores them mapped
    # (_wf_paged_finish); identity rows and the cluster epilogue's distributed
    # winners store mapped directly (_wf_out_gmem); the inherited exact
    # fallback, shared with the other primitives backends, stores raw columns
    # and is re-mapped in place (_wf_page_fixup).  Request q == row: next_n
    # is 1 here.
    wf_page_size: int = 0
    # Paged output geometry (factory-set): wf_pt_pages = ceil(N / page_size)
    # entries per table row; wf_pt_cache = min(that, 4 * nt, the cache cap)
    # LEADING entries staged in smem at kernel entry, the rest gathered from
    # global memory (real decode rows are short, so at production widths every
    # selected column falls inside the stage).  The staging loads are issued
    # before the length load and stored after it, so their latency hides
    # behind the length load's: gathering straight from global memory put a
    # cold L2 round trip on every row's critical path (measured +0.35-0.45 us
    # per call on every arm, identity rows included), and a store issued right
    # after its load stalls the warp before the length load can go out.
    wf_pt_pages: int = 0
    wf_pt_cache: int = 0
    # entries prefetched before the length load: enough for every row the
    # short-row arms serve (ceil(small_n / page_size), 256 at page size 64);
    # rows past small_n stage the rest of the leading entries in the long
    # pipeline.  Prefetching the whole stage on every row cost the wide
    # envelopes one load + one store per thread on all 32 warps (64K: 1024
    # entries, 256K: 2048) for decode rows that touch 7-33 of them.
    wf_pt_pre: int = 0

    @cute.jit
    def _wf_f32(self, bits):
        """Element bits (dtype pattern in the low bits of a Uint32) -> f32."""
        if cutlass.const_expr(self.is_f32):
            return bits.bitcast(cutlass.Float32)
        else:
            return self.value_from_bits(bits).to(cutlass.Float32)

    @cute.jit
    def _wf_val_of_key(self, key):
        """Ordered key -> f32 value (bijective per dtype)."""
        return self.value_from_key(key).to(cutlass.Float32)

    @cute.jit
    def _wf_key_of_f32(self, f):
        """Ordered key of an f32 threshold in this dtype's key space.  Used
        only for radix-round skip bounds (a wider range skips fewer rounds,
        never changes the result).  The f32->16-bit round-to-nearest can
        NARROW a bound, which is safe only because every staged member lies
        on the same 16-bit grid (RN(x) is at most the smallest grid point
        >= x), so no member falls outside the rounded range; callers must
        pass bounds that bracket the members on that grid."""
        if cutlass.const_expr(self.is_f32):
            return self.to_key32(f.bitcast(cutlass.Uint32))
        else:
            if cutlass.const_expr(self.dtype == cutlass.Float16):
                hb = f.to(cutlass.Float16).bitcast(cutlass.Uint16).to(cutlass.Uint32)
            else:
                hb = f.to(cutlass.BFloat16).bitcast(cutlass.Uint16).to(cutlass.Uint32)
            return self.to_key16(hb)

    @cute.jit
    def _wf_coarse_lb_key(self, b):
        """Exact key lower bound of coarse bin ``b`` (census arm skip bounds).
        fp32: the fp16-midpoint construction; 16-bit dtypes: coarse keys ARE
        the 16-bit keys, so the bound is simply b << coarse_shift."""
        if cutlass.const_expr(self.is_f32):
            return self.exact_key(
                self.coarse_bin_lower_bound_f32(b).bitcast(cutlass.Uint32)
            )
        else:
            return cutlass.Uint32(b) << cutlass.Uint32(self.coarse_shift)

    @cute.jit
    def _wf_elem_bits(self, w, h: cutlass.Constexpr):
        """Element ``h`` (0..EPW-1) of a loaded 32-bit word."""
        if cutlass.const_expr(self.is_f32):
            return w
        else:
            return (w >> cutlass.Uint32(16 * h)) & cutlass.Uint32(0xFFFF)

    @cute.jit
    def _wf_bin(self, val, tf_f, sc):
        """Survivor bin.  MUST be the single binning function for both the
        walk histogram and the epilogue emit: any disagreement between the
        two shifts the rank accounting.  NaN handled explicitly (a
        saturating float->int convert sends NaN to 0, i.e. ranked BOTTOM,
        breaking torch parity where NaN sorts on top).  Implemented as
        the shared _WF_BIN_PTX asm sequence so the rounding is
        bit-identical with the walk's wf_emit_pred by construction."""
        return wf_bin_u8(val, tf_f, sc)

    @cute.jit
    def _wf_out(self, out_idx_row, pt_row, pos, col):
        """Emit selected column ``col`` for output slot ``pos``.  Unpaged: the
        store itself.  Paged: the raw column goes to the smem stage
        (``pt_row[2]``) and _wf_paged_finish later stores the whole range as
        physical slots in one coalesced pass.  Mapping at every scattered
        emit instead -- a dependent page lookup inside the divergent classify
        loops -- measured +0.26-0.4 us per call (ncu: +16% warp-instructions,
        short-scoreboard stalls on every element slot of the loops)."""
        if cutlass.const_expr(self.wf_page_size > 0):
            pt_row[2][pos] = col
        else:
            out_idx_row[pos] = col

    @cute.jit
    def _wf_out_gmem(self, out_idx_row, pt_row, pos, col):
        """Store ``col`` at ``pos`` straight to global memory, as its physical
        slot in paged variants: one int32 gather from the request's page-table
        row (``pt_row[0]``: the smem stage of the leading wf_pt_cache entries;
        ``pt_row[1]``: the global row for pages past it).  For emits that
        cannot go through the stage: identity rows, the cluster epilogue's
        distributed winners (every CTA writes its own), the finish pass."""
        if cutlass.const_expr(self.wf_page_size > 0):
            s_pt, pt_gmem, _ = pt_row
            lg = cutlass.const_expr(self.wf_page_size.bit_length() - 1)
            page = col >> cutlass.Int32(lg)
            base = cutlass.Int32(0)
            if cutlass.const_expr(self.wf_pt_cache >= self.wf_pt_pages):
                base = s_pt[page]  # the whole row is staged: no branch
            else:
                if page < cutlass.Int32(self.wf_pt_cache):
                    base = s_pt[page]
                else:
                    base = pt_gmem[page]
            out_idx_row[pos] = (base << cutlass.Int32(lg)) + (
                col & cutlass.Int32(self.wf_page_size - 1)
            )
        else:
            out_idx_row[pos] = col

    @cute.jit
    def _wf_out_gmem_direct(self, out_idx_row, pt_row, pos, col):
        """_wf_out_gmem without the smem stage: the page-table entry comes
        straight from the global row (``pt_row[1]``, L1-hot after the
        prologue prefetch; 64 consecutive columns share one entry, so a
        warp's gather is one broadcast load).  For the identity arm, which
        has no barrier that could publish the stage."""
        if cutlass.const_expr(self.wf_page_size > 0):
            _, pt_gmem, _ = pt_row
            lg = cutlass.const_expr(self.wf_page_size.bit_length() - 1)
            base = pt_gmem[col >> cutlass.Int32(lg)]
            out_idx_row[pos] = (base << cutlass.Int32(lg)) + (
                col & cutlass.Int32(self.wf_page_size - 1)
            )
        else:
            out_idx_row[pos] = col

    @cute.jit
    def _wf_paged_finish(self, out_idx_row, pt_row, lo, n, tidx):
        """Paged variants: once the arm's emits landed in the smem stage, one
        barrier and one coalesced pass store ``out_idx_row[lo, lo + n)`` as
        physical slots -- SGLang's structure: the page lookups run as an
        independent-iteration loop, not as a dependent load inside every
        scattered emit.  CTA-uniform call sites only.  No code when unpaged."""
        if cutlass.const_expr(self.wf_page_size > 0):
            cute.arch.barrier()  # every staged column is visible
            for i in range(tidx, n, self.nt):
                self._wf_out_gmem(out_idx_row, pt_row, lo + i, pt_row[2][lo + i])

    @cute.jit
    def _wf_page_fixup(self, out_idx_row, pt_row, lo, n, tidx):
        """Paged variants: re-map in place the raw columns the inherited exact
        fallback (shared with the other primitives backends, so its emit
        sites stay raw) wrote to ``out_idx_row[lo, lo + n)``.  -1 padding is
        left alone.  CTA-uniform call sites only (block barrier inside).  No
        code when unpaged."""
        if cutlass.const_expr(self.wf_page_size > 0):
            cute.arch.barrier()  # the helper's stores are complete block-wide
            for i in range(tidx, n, self.nt):
                c = out_idx_row[lo + i]
                if c >= 0:
                    self._wf_out_gmem(out_idx_row, pt_row, lo + i, c)

    @cute.jit
    def _wf_tie_select_small(
        self,
        s_tk,
        s_ti,
        nb,
        above,
        remaining,
        s_scratch,
        s_warp_sums,
        s_misc,
        out_idx_row,
        pt_row,
        tidx,
    ):
        """The <= 128-tie warp-ballot select (inherited tie_select); paged
        variants rank straight into the smem stage (finished later by
        _wf_paged_finish), unpaged ones into the output."""
        if cutlass.const_expr(self.wf_page_size > 0):
            self.tie_select(
                s_tk,
                s_ti,
                nb,
                above,
                remaining,
                s_scratch,
                s_warp_sums,
                s_misc,
                pt_row[2],
                pt_row[2],  # dummy: has_values False
                tidx,
            )
        else:
            self.tie_select(
                s_tk,
                s_ti,
                nb,
                above,
                remaining,
                s_scratch,
                s_warp_sums,
                s_misc,
                out_idx_row,
                out_idx_row,  # dummy: has_values False
                tidx,
            )

    # ------------------------------------------------------------------
    # Register-resident short-row arm: emit helpers
    # ------------------------------------------------------------------
    @cute.jit
    def _rg_slot_idx(self, tidx, bp, lg: cutlass.Constexpr):
        """Element index of register slot ``bp`` (slot = u*E + e) of thread tidx."""
        return (
            (tidx + (bp >> cutlass.Int32(lg)) * cutlass.Int32(self.nt))
            << cutlass.Int32(lg)
        ) + (bp & cutlass.Int32(self.vec_elems - 1))

    @cute.jit
    def _rg_emit_winners(
        self, win, s_count, out_idx_row, pt_row, tidx, lane, lg: cutlass.Constexpr
    ):
        """Warp-aggregated tickets for this thread's winner mask, then the
        mask walk writes the indices.  ONE shared atomic per warp: the
        per-element same-address atomic serialized k times per row."""
        top_k = cutlass.const_expr(self.top_k)
        cnt = cutlass.Int32(cute.arch.popc(win))
        inc = warp_inclusive_sum(cnt, lane)
        bpos = cutlass.Int32(0)
        if lane == 31:
            if inc != 0:
                bpos = smem_atomic_add(s_count, inc)
        pos = cute.arch.shuffle_sync(bpos, cutlass.Int32(31)) + (inc - cnt)
        while win != 0:
            bp = cutlass.Int32(
                cute.arch.popc((win & (cutlass.Int32(0) - win)) - cutlass.Int32(1))
            )
            win = win & (win - cutlass.Int32(1))
            if pos < top_k:
                self._wf_out(out_idx_row, pt_row, pos, self._rg_slot_idx(tidx, bp, lg))
            pos = pos + cutlass.Int32(1)

    def _wf_census_cap(self) -> int:
        """Longest row the census arm serves (every row up to small_n when the
        register arm is disabled)."""
        return self.wf_census_n if self.wf_reg_arm else self.wf_small_n

    @cute.jit
    def _wf_bound_len(self, length, cap: cutlass.Constexpr):
        """``min(length, cap)`` as a select the loop analysis can see: the
        arms' row loops are then bounded by the arm's cap instead of the
        kernel-entry clamp to N (which unrolled them N / (nt * E) times)."""
        len_c = length
        if len_c > cutlass.Int32(cap):
            len_c = cutlass.Int32(cap)
        return len_c

    @cute.jit
    def _wf_zero_hist(self, s_h4k, tidx):
        """Zero the hist_size-bin coarse histogram with 16-byte stores: one
        STS.128 per thread per 4*nt bins instead of four scalar STS (the
        arms' prologue is MIO-bound: ncu attributed its stalls to the
        shared-memory instruction queue, where these stores compete with
        the parameter and special-register loads).  hist_size is a
        multiple of 4*nt and the stage is 128-byte aligned."""
        base = s_h4k.toint()
        for zz in cutlass.range_constexpr(self.hist_size // (4 * self.nt)):
            st_shared_v4_zero(base + (tidx * 4 + cutlass.Int32(zz * 4 * self.nt)) * 4)

    @cute.jit
    def _wf_reg_row(
        self,
        row_in,
        length,
        out_idx_row,
        pt_row,
        slab_k,
        slab_i,
        s_cv,
        s_h256,
        s_tk,
        s_ti,
        s_warp_sums,
        s_misc,
        s_count,
        tidx,
        vpt: cutlass.Constexpr,
    ):
        """Register-resident solve for one short row (length <= vpt * E * nt).

        SGLang's TopKRegister shape on the primitives substrate: the row is
        read ONCE into registers, bounded by the REAL length (never the
        padded width), and both passes -- the exact coarse census and the
        classify -- run from registers with one block barrier each.  It
        replaces the census arm's two L2 passes (measured on the shared
        radix machinery at a 16K width: the streaming loop + scan collect +
        emit re-walk cost 0.55 us of a 3.3 us row; the second read itself
        was L1-resident and free) and loads only the vectors the real length
        covers: a blind load of all vpt vectors (the earlier stand-alone
        register kernel's form) cost ~1 us on 1-4K rows inside a 16K width.

        Exactness is the census arm's: same coarse bins, same rank-k
        crossing, key-exact tie select; a crossing bin holding more than
        ``tie_cap`` members takes the fused exact fallback.  ``s_cv``
        (lcap+4 >= hist_size ints, dead until the tie select) doubles as the
        histogram and then as the tie select's scratch."""
        top_k = cutlass.const_expr(self.top_k)
        lg = cutlass.const_expr(self.vec_elems.bit_length() - 1)
        epw = cutlass.const_expr(self.vec_elems // 4)
        lane = tidx % 32
        s_h4k = s_cv
        # loops bounded by the arm's cap (see _wf_short_row: the visible bound
        # keeps the compiler from unrolling them N / (nt * E) times)
        len_c = self._wf_bound_len(length, self.wf_small_n)
        n4 = len_c >> cutlass.Int32(lg)  # full vectors in the valid row
        tail0 = n4 << cutlass.Int32(lg)

        # ---- row -> registers, only the vectors the real length covers ----
        w = [None] * (4 * vpt)
        for u in cutlass.range_constexpr(vpt):
            vi = tidx + cutlass.Int32(u * self.nt)
            a0 = cutlass.Uint32(0)
            a1 = cutlass.Uint32(0)
            a2 = cutlass.Uint32(0)
            a3 = cutlass.Uint32(0)
            if vi < n4:
                a0, a1, a2, a3 = ld_global_nc_v4_u32(
                    row_in.toint() + cutlass.Int64(vi) * cutlass.Int64(16)
                )
            w[4 * u + 0] = a0
            w[4 * u + 1] = a1
            w[4 * u + 2] = a2
            w[4 * u + 3] = a3

        # zero the census + cursors while the loads are in flight
        self._wf_zero_hist(s_h4k, tidx)
        if tidx == 0:
            s_count[0] = cutlass.Int32(0)  # winner ticket
            s_count[4] = cutlass.Int32(0)  # tie cursor
            s_misc[14] = cutlass.Int32(0)  # max tie key   (uniform-tie shortcut)
            s_misc[15] = cutlass.Int32(0)  # max ~tie key == ~min tie key
        cute.arch.barrier()  # R0: zeros visible

        # ---- exact coarse census from registers (+ the < E scalar tail) ----
        for u in cutlass.range_constexpr(vpt):
            vi = tidx + cutlass.Int32(u * self.nt)
            if vi < n4:
                for j in cutlass.range_constexpr(4):
                    for h in cutlass.range_constexpr(epw):
                        smem_atomic_add(
                            s_h4k
                            + self.coarse_bin(self._wf_elem_bits(w[4 * u + j], h)),
                            1,
                        )
        for i in cutlass.range(tail0 + tidx, len_c, self.nt, unroll=1):
            smem_atomic_add(s_h4k + self.coarse_bin(self.load_scalar(row_in, i)), 1)
        cute.arch.barrier()  # R1: census complete
        # crossing at rank k (ends with a block barrier)                    R2
        self.find_threshold_coarse(
            s_h4k, length, cutlass.Int32(top_k), s_warp_sums, s_misc, tidx
        )
        binb = s_misc[0]  # above-count stays in s_misc[1] for the shared tail

        # ---- classify from registers ----
        # fp32: two float compares against the crossing bin's exact float
        # boundaries (host-verified construction; NaN fails both ordered
        # compares and classifies as a winner, matching the +NaN-on-top
        # census bins).  16-bit dtypes: the coarse bin IS the key prefix.
        hi_f = cutlass.Float32(0.0)
        lo_f = cutlass.Float32(0.0)
        if cutlass.const_expr(self.is_f32):
            hi_f = self.coarse_bin_gt_threshold_f32(binb + cutlass.Int32(1))
            lo_f = self.coarse_bin_lower_bound_f32(binb)
        # Per-vector BRANCH on the thread's own validity (not a predicate
        # folded into the mask): a 1.5K row fills 375 of 1024 threads, and
        # running the unrolled slot code predicated-off on the other 650
        # measured +35% warp-instructions and a 10% slower row than the
        # census arm it replaces (ncu, B200).  Dead vectors contribute no
        # winner bits and no ties, so skipping them is exact.
        win = cutlass.Int32(0)
        # thread-local tie key range (max key, min key) of the ties folded
        # by _wf_note_tie_* (cc 8.x build only; dead otherwise), published
        # once per warp by _wf_publish_tie_range
        kmax = cutlass.Int32(
            -2147483648
        )  # signed keys (_wf_skey): INT_MIN / INT_MAX identities
        kmin = cutlass.Int32(2147483647)
        for u in cutlass.range_constexpr(vpt):
            vi = tidx + cutlass.Int32(u * self.nt)
            if vi < n4:
                for j in cutlass.range_constexpr(4):
                    for h in cutlass.range_constexpr(epw):
                        bits = self._wf_elem_bits(w[4 * u + j], h)
                        slot = cutlass.Int32(u * self.vec_elems + j * epw + h)
                        if cutlass.const_expr(self.is_f32):
                            val = bits.bitcast(cutlass.Float32)
                            le_hi = cutlass.Int32(val <= hi_f)
                            win = win | ((cutlass.Int32(1) - le_hi) << slot)
                            if le_hi == 1:
                                if val >= lo_f:
                                    t = smem_atomic_add(s_count + 4, 1)
                                    kx = self.exact_key(bits)
                                    kmax = self._wf_note_tie_max(t, kx, kmax, s_misc)
                                    kmin = self._wf_note_tie_min(t, kx, kmin, s_misc)
                                    if t < self.tie_cap:
                                        s_tk[t] = kx
                                        s_ti[t] = (
                                            vi << cutlass.Int32(lg)
                                        ) + cutlass.Int32(j * epw + h)
                        else:
                            bq = self.coarse_bin(bits)
                            win = win | (cutlass.Int32(bq > binb) << slot)
                            if bq == binb:
                                t = smem_atomic_add(s_count + 4, 1)
                                kx = self.exact_key(bits)
                                kmax = self._wf_note_tie_max(t, kx, kmax, s_misc)
                                kmin = self._wf_note_tie_min(t, kx, kmin, s_misc)
                                if t < self.tie_cap:
                                    s_tk[t] = kx
                                    s_ti[t] = (vi << cutlass.Int32(lg)) + cutlass.Int32(
                                        j * epw + h
                                    )
        self._rg_emit_winners(win, s_count, out_idx_row, pt_row, tidx, lane, lg)
        for i in cutlass.range(tail0 + tidx, len_c, self.nt, unroll=1):
            bits = self.load_scalar(row_in, i)
            b = self.coarse_bin(bits)
            if b > binb:
                p = smem_atomic_add(s_count, 1)
                if p < top_k:
                    self._wf_out(out_idx_row, pt_row, p, cutlass.Int32(i))
            else:
                if b == binb:
                    t = smem_atomic_add(s_count + 4, 1)
                    kx = self.exact_key(bits)
                    kmax = self._wf_note_tie_max(t, kx, kmax, s_misc)
                    kmin = self._wf_note_tie_min(t, kx, kmin, s_misc)
                    if t < self.tie_cap:
                        s_tk[t] = kx
                        s_ti[t] = cutlass.Int32(i)
        self._wf_publish_tie_range(kmax, kmin, s_misc, tidx)
        cute.arch.barrier()  # R3: winners written, ties staged
        # the shared tail (_wf_short_tail) selects among the staged ties or
        # takes the exact fallback; binb / above / nb stay in smem for it

    @cute.jit
    def _wf_note_tie_max(self, t, kx, kmax, s_misc):
        """Record tie number ``t`` (exact key ``kx``) in the row's max tie
        key, s_misc[14].  A direct same-address smem red, except in the cc 8.x
        build (wf_tie_fold) where ties past WF_TIE_RED_DIRECT fold into the
        thread-local ``kmax`` for _wf_publish_tie_range instead: Ampere/Ada
        serialize same-address smem reds, and two per tie made a 16K-tie
        plateau issue 32K of them per row (A100 const L=16384: 41.5 us vs
        SGLang's 11.8, one-bin 91 vs 22; L40S 24/29 vs 7.6/7.9; the fold:
        A100 18/27, L40S 11/16, and the decode trace's ReLU rows -22..-25%).
        The first WF_TIE_RED_DIRECT ties stay direct so a Gaussian row (tens
        of ties) never pays the per-warp publish; SM90+ issue same-address
        reds at near full rate and keep the direct path everywhere."""
        if cutlass.const_expr(self.wf_tie_fold):
            if t < WF_TIE_RED_DIRECT:
                smem_red_max_u32(s_misc + 14, kx)
            else:
                # fold on the SIGNED key (_wf_skey): a loop-carried Uint32 can
                # be re-wrapped signed by the DSL (see _fallback_row), which
                # would break a plain unsigned compare; one xor per tie
                sk = self._wf_skey(kx)
                if sk > kmax:
                    kmax = sk
        else:
            smem_red_max_u32(s_misc + 14, kx)
        return kmax

    @cute.jit
    def _wf_note_tie_min(self, t, kx, kmin, s_misc):
        """Min-key half of _wf_note_tie_max: s_misc[15] holds max(~key)."""
        if cutlass.const_expr(self.wf_tie_fold):
            if t < WF_TIE_RED_DIRECT:
                smem_red_max_u32(s_misc + 15, ~kx)
            else:
                sk = self._wf_skey(kx)
                if sk < kmin:
                    kmin = sk
        else:
            smem_red_max_u32(s_misc + 15, ~kx)
        return kmin

    @cute.jit
    def _wf_publish_tie_range(self, kmax, kmin, s_misc, tidx):
        """cc 8.x build: fold the thread-local tie key range (signed keys,
        see _wf_note_tie_max) into s_misc[14] (max key) and s_misc[15] (max
        ~key == ~min key) with one warp reduction and one lane-0 red per warp,
        only in warps that folded a tie (a folded tie moves kmax off INT_MIN
        or kmin off INT_MAX, so the vote is exact).  No-op in the other
        builds, where every tie published directly."""
        if cutlass.const_expr(self.wf_tie_fold):
            folded = cutlass.Int32(kmax != cutlass.Int32(-2147483648)) + cutlass.Int32(
                kmin != cutlass.Int32(2147483647)
            )
            if cute.arch.vote_any_sync(folded != 0):
                kmax = cute.arch.warp_redux_sync(kmax, "max")
                kmin = cute.arch.warp_redux_sync(kmin, "min")
                if tidx % 32 == 0:
                    # back to unsigned keys: flip the sign bit again
                    umax = cutlass.Uint32(kmax) ^ cutlass.Uint32(0x80000000)
                    umin = cutlass.Uint32(kmin) ^ cutlass.Uint32(0x80000000)
                    smem_red_max_u32(s_misc + 14, umax)
                    smem_red_max_u32(s_misc + 15, ~umin)

    @cute.jit
    def _wf_tie_select_32(
        self, s_tk, s_ti, nb, above, remaining, out_idx_row, pt_row, tidx
    ):
        """The <= 32-tie warp-ballot rank select ALONE: the inherited
        tie_select carries the 128-candidate and byte-radix bodies behind
        this case, which the hot path would have to jump over.  Warp w ranks
        candidate w while its 32 lanes hold all candidates; a candidate's
        rank is the number of candidates strictly greater (exact key, then
        ascending index -- a total order, so ranks are unique and the result
        deterministic).  Paged variants rank into the smem stage."""
        lane = tidx % 32
        warp = tidx // 32
        ck = cutlass.Uint32(0)  # sentinel: loses every comparison
        ci = cutlass.Int32(0x7FFFFFFF)
        if lane < nb:
            ck = cutlass.Uint32(s_tk[lane])
            ci = cutlass.Int32(s_ti[lane])
        # 32 targets regardless of the warp count (two per warp at nt=512)
        for w_ in cutlass.range_constexpr(32 // self.nw):
            t = warp + w_ * self.nw
            if t < nb:
                tk = cutlass.Uint32(s_tk[t])
                ti = cutlass.Int32(s_ti[t])
                greater = (ck > tk) | ((ck == tk) & (ci < ti))
                rank = cute.arch.popc(cute.arch.vote_ballot_sync(greater))
                if lane == 0:
                    if rank < remaining:
                        self._wf_out(out_idx_row, pt_row, above + rank, ti)

    @cute.jit
    def _wf_hot_tail(
        self, out_idx_row, pt_row, s_tk, s_ti, s_misc, s_count, tidx, length, epi
    ):
        """Hot tail of a short-row arm, inlined right after the arm's body:
        the common case -- a consistent count and at most 32 exact-key ties
        in the rank-k bin -- ranks them with the warp-ballot select, finishes
        the paged stage and EXITS here, so a decode row runs prologue ->
        census arm -> tail -> exit without a far jump (ncu on the previous
        layout: the census path jumped over the register arm's body at
        entry and over the larger tie-select bodies after the arm, no-
        instruction stalls 10x SGLang's linear TopKRegister).  Everything
        else falls through to the shared cold tail (_wf_short_tail).  Same
        contract as that tail: above-count in s_misc[1], tie count in
        s_count[4]."""
        top_k = cutlass.const_expr(self.top_k)
        above = s_misc[1]
        nb = s_count[4]
        remaining = cutlass.Int32(top_k) - above
        hot = (
            cutlass.Int32(remaining >= 0)
            & cutlass.Int32(remaining <= nb)
            & cutlass.Int32(nb <= 32)
            & cutlass.Int32(s_count[0] == above)  # winner tickets == census
        )
        if hot == 1:
            if remaining > 0:
                self._wf_tie_select_32(
                    s_tk, s_ti, nb, above, remaining, out_idx_row, pt_row, tidx
                )
            self._wf_paged_finish(out_idx_row, pt_row, cutlass.Int32(0), top_k, tidx)
            self._wf_arm_epilogue(epi, length, tidx, cutlass.Int32(0))

    @cute.jit
    def _wf_arm_epilogue(self, epi, length, tidx, fb):
        """Epilogue of the short-row arms, INLINE at the end of each tail
        branch so the hot path runs straight into it and exits there.  The
        region join after the tail sits past the exact fallback's body and
        the kernel's final block past the whole long pipeline: two far jumps
        on every decode row (ncu: the census path's no-instruction stall
        ratio 3.4 vs 0.3 for SGLang's linear TopKRegister, PREEXIT alone
        3% of the samples).  Writes the status word ``fb`` (0: fast path, 1:
        exact fallback, 2: flood refine -- the tests' contract, so a
        regression that sends decode rows down a slow path is visible), in
        telemetry builds the arm tag (5 = register arm, 4 = census arm),
        releases PDL dependents and exits the CTA.

        The slab's row-histogram tail (slab_h) needs NO reset here: only the
        S > 1 gmem publish writes it (red.add) and its last arriver zeroes it
        again; the exact fallback's harvest stays inside slab_k / slab_i
        (below 2 * GCAP).  The per-row reset this epilogue used to carry was
        256 redundant global stores per decode row."""
        st_ptr, row, num_rows_dbg = epi
        if tidx == 0:
            st_ptr[row] = fb
            if cutlass.const_expr(self.wf_telemetry):
                tag = cutlass.Int32(4)
                if cutlass.const_expr(self.wf_reg_arm) and length > cutlass.Int32(
                    cutlass.const_expr(self.wf_census_n)
                ):
                    tag = cutlass.Int32(5)
                st_ptr[num_rows_dbg + row] = tag
        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_launch_dependents()
        _wf_exit()

    @cute.jit
    def _wf_skey(self, kk):
        """Exact key as a SIGNED Int32 with the top bit flipped: signed
        compares of these follow the unsigned key order.  The refine's range
        tests use them because loop-carried Uint32 keys are re-wrapped signed
        by the DSL in some instantiations (see _fallback_row), which makes a
        plain unsigned compare unreliable; a signed compare of explicitly
        signed values cannot be lowered any other way.  Two instructions per
        key against the 64-bit compares' six."""
        return (kk ^ cutlass.Uint32(0x80000000)).bitcast(cutlass.Int32)

    @cute.jit
    def _wf_flood_bin(self, bits, binb, lo_k, slo, shi, sh64, s_h4k):
        """One element of a refine round: a member of the crossing coarse bin
        whose key lies in the round's range [slo, shi] counts in key bin
        (key - lo_k) >> shift.  The 32-bit difference needs no wrap (key >=
        lo_k inside the range); the shift runs zero-extended in 64 bits, where
        a signed lowering cannot turn it arithmetic."""
        if self.coarse_bin(bits) == binb:
            kk = cutlass.Uint32(self.exact_key(bits))
            sk = self._wf_skey(kk)
            if sk >= slo:
                if sk <= shi:
                    d = kk - cutlass.Uint32(lo_k)
                    smem_atomic_add(
                        s_h4k + cutlass.Int32(cutlass.Int64(cutlass.Uint32(d)) >> sh64),
                        1,
                    )

    @cute.jit
    def _wf_flood_harvest(
        self,
        bits,
        col,
        binb,
        slo,
        shi,
        above,
        n_gt,
        need,
        single,
        s_count,
        s_tk,
        s_ti,
        out_idx_row,
        pt_row,
    ):
        """One element of the refine's harvest: a member above the final key
        bin [slo, shi] is a winner (ticket s_count[8]); a member inside it is
        a tie -- staged for the key-exact select, or emitted by count when
        the bin is one key (ticket s_count[16])."""
        if self.coarse_bin(bits) == binb:
            kk = cutlass.Uint32(self.exact_key(bits))
            sk = self._wf_skey(kk)
            if sk > shi:
                p = smem_atomic_add(s_count + 8, 1)
                if p < n_gt:
                    self._wf_out(out_idx_row, pt_row, above + p, col)
            else:
                if sk >= slo:
                    c = smem_atomic_add(s_count + 16, 1)
                    if single == 1:
                        # one key: any ``need`` of them is exact
                        if c < need:
                            self._wf_out(out_idx_row, pt_row, above + n_gt + c, col)
                    else:
                        if c < self.tie_cap:
                            s_tk[c] = kk
                            s_ti[c] = col

    @cute.jit
    def _wf_flood_refine(
        self,
        row_in,
        length,
        out_idx_row,
        pt_row,
        binb,
        above,
        remaining,
        s_h4k,
        s_h256,
        s_tk,
        s_ti,
        s_warp_sums,
        s_misc,
        s_count,
        tidx,
    ):
        """Exact select of ``remaining`` members of the crossing coarse bin
        when more than tie_cap of them tied with DISTINCT keys (the uniform
        flood takes the staged-prefix shortcut instead).  The census's answer
        is kept -- the ``above`` winners already emitted, the bin index, the
        members' exact key range in s_misc[14..15] -- and the bin's members
        alone go through MSD key-radix rounds (hist_size bins over the
        current key range, members re-read from the L2-hot row) until the
        rank-``remaining`` key bin holds <= tie_cap members or a single key;
        one harvest pass then emits the members above that bin and stages
        the rest in the smem tie stage for the key-exact select (emitted by
        count for a single key).  Membership is the classify's own test
        (coarse_bin == binb), so the candidate set is bit-identical to the
        one the census ranked, and the key seeds are the members' true
        min/max, so no bin-boundary construction is involved.

        Replaces the from-scratch re-solve on this path (_fallback_row:
        three rounds over the whole row, a gmem slab harvest and a four-
        round slab select).  A 16384-wide one-bin row (every key inside one
        fp16 bin -- SGLang's 2048-candidate truncation case) resolves in ONE
        round: the members' key span fits the histogram, so the round's bins
        are single keys; four row passes in total instead of eight plus the
        slab rounds.  CTA-uniform call site; block barriers inside."""
        # seeds: the members' exact key range (0 is the identity of both reds)
        lo_k = ~cutlass.Uint32(s_misc[15])
        hi_k = cutlass.Uint32(s_misc[14])
        need = cutlass.Int32(remaining)
        done = cutlass.Int32(0)
        lg = cutlass.const_expr(self.vec_elems.bit_length() - 1)
        epw = cutlass.const_expr(self.vec_elems // 4)  # elements per 32-bit word
        n4 = length >> cutlass.Int32(lg)  # whole float4s of the row
        for _round in cutlass.range_constexpr(3):
            if done == 0:  # block-uniform: barriers inside are safe
                self._wf_zero_hist(s_h4k, tidx)
                cute.arch.barrier()
                # bin = (key - lo) >> shift, the span reduced to hist_size
                # bins (shift derived zero-extended in 64 bits, see the
                # signedness notes in _fallback_row; the per-element tests
                # use the flipped signed keys of _wf_skey)
                lo64 = cutlass.Int64(cutlass.Uint32(lo_k))
                span64 = cutlass.Int64(cutlass.Uint32(hi_k)) - lo64
                shift = cutlass.Uint32(0)
                spn = span64
                while spn > cutlass.Int64(self.hist_size - 1):
                    spn = spn >> 1
                    shift = shift + 1
                sh64 = cutlass.Int64(shift)
                slo = self._wf_skey(cutlass.Uint32(lo_k))
                shi = self._wf_skey(cutlass.Uint32(hi_k))
                # float4 passes with a scalar tail, as the census (the passes
                # are instruction-bound: ~40 instructions per element with
                # 64-bit range math, so the per-element work is kept lean)
                for i in cutlass.range(tidx, n4, self.nt, unroll=1):
                    w0, w1, w2, w3 = ld_global_nc_v4_u32(
                        row_in.toint() + cutlass.Int64(i) * 16
                    )
                    for j in cutlass.range_constexpr(4):
                        for h in cutlass.range_constexpr(epw):
                            self._wf_flood_bin(
                                self._wf_elem_bits((w0, w1, w2, w3)[j], h),
                                binb,
                                lo_k,
                                slo,
                                shi,
                                sh64,
                                s_h4k,
                            )
                for i in cutlass.range(
                    (n4 << cutlass.Int32(lg)) + tidx, length, self.nt, unroll=1
                ):
                    self._wf_flood_bin(
                        self.load_scalar(row_in, i), binb, lo_k, slo, shi, sh64, s_h4k
                    )
                cute.arch.barrier()
                # descending crossing at rank ``need`` (ends with a block
                # barrier; publishes bin/above/cnt)
                self.find_threshold_wide(
                    s_h4k, cutlass.Int32(0), need, s_warp_sums, s_misc, tidx
                )
                bkt = s_misc[0]
                ab = s_misc[1]
                cnt = s_misc[2]
                cute.arch.barrier()
                lo_k2 = lo_k + cutlass.Uint32(cutlass.Int64(bkt) << sh64)
                # the upper edge in 64 bits, clamped to the current max key:
                # lo_k is the members' exact MIN (not bin-aligned), so the last
                # partial bin's edge can pass 0xFFFFFFFF -- a +NaN payload
                # flood sits at the top of the key space -- and a 32-bit wrap
                # would empty the next round's range
                hi64 = lo64 + ((cutlass.Int64(bkt) + 1) << sh64) - cutlass.Int64(1)
                hi_cur = cutlass.Int64(cutlass.Uint32(hi_k))
                if hi64 > hi_cur:
                    hi64 = hi_cur
                hi_k2 = cutlass.Uint32(hi64)
                need = need - ab
                lo_k = lo_k2
                hi_k = hi_k2
                if cnt <= self.tie_cap:
                    done = cutlass.Int32(1)
                if lo_k == hi_k:
                    done = cutlass.Int32(1)
        # done == 1 here: three rounds cover 36 key bits
        # ---- harvest: members above the final key bin are winners, members
        # inside it are the ties (staged, or emitted by count for one key) ----
        if tidx == 0:
            s_count[8] = cutlass.Int32(0)  # winner ticket
            s_count[16] = cutlass.Int32(0)  # tie ticket
        cute.arch.barrier()
        slo = self._wf_skey(cutlass.Uint32(lo_k))
        shi = self._wf_skey(cutlass.Uint32(hi_k))
        n_gt = remaining - need  # members above the final bin
        single = cutlass.Int32(0)
        if lo_k == hi_k:
            single = cutlass.Int32(1)
        for i in cutlass.range(tidx, n4, self.nt, unroll=1):
            w0, w1, w2, w3 = ld_global_nc_v4_u32(row_in.toint() + cutlass.Int64(i) * 16)
            for j in cutlass.range_constexpr(4):
                for h in cutlass.range_constexpr(epw):
                    self._wf_flood_harvest(
                        self._wf_elem_bits((w0, w1, w2, w3)[j], h),
                        (i << cutlass.Int32(lg)) + cutlass.Int32(j * epw + h),
                        binb,
                        slo,
                        shi,
                        above,
                        n_gt,
                        need,
                        single,
                        s_count,
                        s_tk,
                        s_ti,
                        out_idx_row,
                        pt_row,
                    )
        for i in cutlass.range(
            (n4 << cutlass.Int32(lg)) + tidx, length, self.nt, unroll=1
        ):
            self._wf_flood_harvest(
                self.load_scalar(row_in, i),
                cutlass.Int32(i),
                binb,
                slo,
                shi,
                above,
                n_gt,
                need,
                single,
                s_count,
                s_tk,
                s_ti,
                out_idx_row,
                pt_row,
            )
        cute.arch.barrier()
        if single == 0:
            self._tie_select_smem_skip_wf(
                s_tk,
                s_ti,
                s_count[16],
                above + n_gt,
                need,
                hi_k,
                lo_k,
                s_h256,
                s_warp_sums,
                s_misc,
                out_idx_row,
                pt_row,
                tidx,
            )

    @cute.jit
    def _wf_short_tail(
        self,
        row_in,
        length,
        out_idx_row,
        pt_row,
        slab_k,
        slab_i,
        s_cv,
        s_h256,
        s_tk,
        s_ti,
        s_warp_sums,
        s_misc,
        s_count,
        tidx,
        epi,
    ):
        """Shared tail of both short-row arms, entered right after their last
        barrier with the crossing bin in s_misc[0], the above-count in
        s_misc[1], the tie count in s_count[4] and the tie-key range in
        s_misc[14..15]: resolve the rank-k bin's ties, refine a flooded bin
        (_wf_flood_refine), or take the from-scratch exact fallback on a
        count mismatch.  ONE copy: with the tail inlined into each arm the kernel
        carried two tie selects and two fallbacks, and ncu showed the
        no_instruction stall ratio rising 1.3 -> 2.3 on 12K rows (+0.3 us)
        and 5.2 -> 8.0 on decode rows -- instruction-cache pressure from the
        larger binary."""
        top_k = cutlass.const_expr(self.top_k)
        binb = s_misc[0]
        above = s_misc[1]
        nb = s_count[4]
        remaining = cutlass.Int32(top_k) - above
        # all ties share one exact key?  (0 is the identity of both red.max)
        uniform = cutlass.Int32(0)
        if cutlass.const_expr(self.wf_uniform_tie):
            if cutlass.Uint32(s_misc[14]) == ~cutlass.Uint32(s_misc[15]):
                uniform = cutlass.Int32(1)
        ok = cutlass.Int32(1)
        if remaining < 0:
            ok = cutlass.Int32(0)
        if remaining > nb:
            ok = cutlass.Int32(0)
        # the arm's winner tickets must agree with the census's above-count:
        # the register arm's fp32 compares rank a sign-set NaN as a winner
        # where the census binned it bottom (see the module docstring's NaN
        # order), and that row then takes the exact fallback instead of
        # dropping a real winner
        nwin = s_count[0]
        if nwin != above:
            ok = cutlass.Int32(0)
        if nb > self.tie_cap:
            # tie flood: exact from the staged prefix only when uniform
            # (remaining <= k <= tie_cap staged identical keys); otherwise
            # the exact MSD re-solve below
            if uniform == 0:
                ok = cutlass.Int32(0)
        if ok == 1:
            if remaining > 0:
                if nb <= 128:
                    self._wf_tie_select_small(
                        s_tk,
                        s_ti,
                        nb,
                        above,
                        remaining,
                        s_cv,  # >= 512 ints scratch (the histogram is dead)
                        s_warp_sums,
                        s_misc,
                        out_idx_row,
                        pt_row,
                        tidx,
                    )
                else:
                    # UNIFORM-TIE SHORTCUT: when every tie carries the same
                    # exact key (ReLU-style rows tie thousands of exact zeros
                    # at the boundary), any `remaining` of them is an exact
                    # top-k, so copy the staged prefix instead of four
                    # byte-radix rounds over identical keys (measured 2.4 us
                    # of a 6 us row, the same rounds SGLang's kernel runs).
                    # Distinct keys take the key-exact select.
                    if uniform == 1:
                        for i in range(tidx, remaining, self.nt):
                            self._wf_out(out_idx_row, pt_row, above + i, s_ti[i])
                    else:
                        # EXACT key range of the staged ties (nb <= tie_cap:
                        # every tie is staged), tracked by both arms in
                        # s_misc[14..15]; the select skips each leading byte
                        # round on which min and max agree -- a plateau of
                        # identical keys skips all four
                        k_lo = ~cutlass.Uint32(s_misc[15])
                        k_hi = cutlass.Uint32(s_misc[14])
                        self._tie_select_smem_skip_wf(
                            s_tk,
                            s_ti,
                            nb,
                            above,
                            remaining,
                            k_hi,
                            k_lo,
                            s_h256,
                            s_warp_sums,
                            s_misc,
                            out_idx_row,
                            pt_row,
                            tidx,
                        )
            self._wf_paged_finish(out_idx_row, pt_row, cutlass.Int32(0), top_k, tidx)
            self._wf_arm_epilogue(epi, length, tidx, cutlass.Int32(0))
        else:
            # count mismatch between the census and the classify (both read
            # the same row through the same bin function, so defensive only):
            # the from-scratch exact re-solve.  Tie flood of distinct keys
            # (> tie_cap in the crossing bin): refine that bin from the
            # census's answer.
            mismatch = cutlass.Int32(0)
            if remaining < 0:
                mismatch = cutlass.Int32(1)
            if remaining > nb:
                mismatch = cutlass.Int32(1)
            if nwin != above:
                mismatch = cutlass.Int32(1)
            fb = cutlass.Int32(2)  # status: flood refine
            if mismatch == 1:
                fb = cutlass.Int32(1)  # status: exact fallback
            if mismatch == 1:
                self._fallback_row(  # type: ignore[attr-defined]  # Prod MRO
                    row_in,
                    length,
                    out_idx_row,
                    slab_k,
                    slab_i,
                    s_cv,
                    s_h256,
                    s_warp_sums,
                    s_misc,
                    s_count + 8,
                    s_count + 16,
                    tidx,
                )
                self._wf_page_fixup(out_idx_row, pt_row, cutlass.Int32(0), top_k, tidx)
            else:
                self._wf_flood_refine(
                    row_in,
                    length,
                    out_idx_row,
                    pt_row,
                    binb,
                    above,
                    remaining,
                    s_cv,  # hist_size ints: the histogram is dead
                    s_h256,
                    s_tk,
                    s_ti,
                    s_warp_sums,
                    s_misc,
                    s_count,
                    tidx,
                )
                self._wf_paged_finish(
                    out_idx_row, pt_row, cutlass.Int32(0), top_k, tidx
                )
            self._wf_arm_epilogue(epi, length, tidx, fb)

    @cute.jit
    def _wf_short_row(
        self,
        row_in,
        length,
        out_idx_row,
        pt_row,
        slab_k,
        slab_i,
        s_cv,
        s_h256,
        s_tk,
        s_ti,
        s_warp_sums,
        s_misc,
        s_count,
        tidx,
    ):
        """Direct solve for one short row (length <= WF_SMALL_N), smem
        only: the parent's 4096-bin COARSE histogram (fp16-key bins --
        an 8-bit ordered-key hist was tried first and measured WORSE
        than the plain MSD re-solve: its top byte is an EXPONENT
        histogram, so randn's crossing bin held thousands of ties and
        always overflowed into the fallback) -> rank bin -> emit
        winners above the bin + stage bin ties in smem -> key-exact
        tie select seeded with the bin's lower-bound keys.  Two L2-hot
        gmem passes, no slab traffic -- the census structure that
        makes radix_primitives the short-row winner, built from this
        class's inherited helpers.  Tie overflow (> tie_cap in one
        coarse bin): identical keys take the staged-prefix copy, distinct
        keys the in-bin refine (_wf_flood_refine), and a count mismatch
        the exact re-solve.  s_cv doubles as the 4096-bin histogram (dead until
        the tie-select scratch phase consumes the hist)."""
        top_k = cutlass.const_expr(self.top_k)
        # coarse histogram: hist_size bins (4096 fp32 / 8192 16-bit), aliased
        # into the dead LCAP+4-int candidate stage
        s_h4k = s_cv
        self._wf_zero_hist(s_h4k, tidx)
        if tidx == 0:
            s_count[0] = cutlass.Int32(0)  # emit ticket
            s_count[4] = cutlass.Int32(0)  # tie cursor
            # tie-key range (max key, max ~key == ~min key) for the
            # uniform-tie shortcut; 0 is the identity for both (no real
            # key or complement is zero)
            s_misc[14] = cutlass.Int32(0)
            s_misc[15] = cutlass.Int32(0)
        cute.arch.barrier()
        # float4 census (the scalar form left ~1us on the 16K-length
        # boundary vs radix_primitives' vectorized stream)
        lg = cutlass.const_expr(self.vec_elems.bit_length() - 1)
        epw = cutlass.const_expr(self.vec_elems // 4)  # elements per 32-bit word
        # Bound the row loops by the arm's cap and keep them ROLLED
        # (unroll=1).  On this path length <= cap already, but the loop
        # analysis only sees the kernel-entry clamp to N and unrolled every
        # row loop N / (nt * E) times: the census phase was 206 SASS lines at
        # a 64K width against 66 at 16K and the classify 546 against 309, for
        # the same executed instructions, and the wider code paid
        # instruction-fetch stalls on every decode row (~0.15 us per launch
        # in graph replay).  Each loop runs at most once per thread here, so
        # a rolled loop costs a few instructions and no code.
        len_c = self._wf_bound_len(length, self._wf_census_cap())
        n4 = len_c >> cutlass.Int32(lg)
        for i in cutlass.range(tidx, n4, self.nt, unroll=1):
            w0, w1, w2, w3 = ld_global_nc_v4_u32(row_in.toint() + cutlass.Int64(i) * 16)
            for j in cutlass.range_constexpr(4):
                for h in cutlass.range_constexpr(epw):
                    smem_atomic_add(
                        s_h4k
                        + self.coarse_bin(self._wf_elem_bits((w0, w1, w2, w3)[j], h)),
                        1,
                    )
        for i in cutlass.range(
            (n4 << cutlass.Int32(lg)) + tidx, len_c, self.nt, unroll=1
        ):
            smem_atomic_add(s_h4k + self.coarse_bin(self.load_scalar(row_in, i)), 1)
        cute.arch.barrier()
        # crossing at rank top_k: bin order ascends with value
        # (coarse_bin is monotone); above = count in strictly greater bins
        self.find_threshold_coarse(
            s_h4k, length, cutlass.Int32(top_k), s_warp_sums, s_misc, tidx
        )
        # no post-find barrier: find_threshold_coarse ends with a
        # full-block barrier, the classify below touches disjoint smem
        # (cursors zeroed and fenced at arm entry), and s_misc[0..1] are
        # not rewritten until the tie select's own fenced machinery
        binb = s_misc[0]  # above-count stays in s_misc[1] for the shared tail
        # thread-local tie key range (cc 8.x build; see _wf_note_tie_*),
        # published once per warp before B3
        kmax = cutlass.Int32(
            -2147483648
        )  # signed keys (_wf_skey): INT_MIN / INT_MAX identities
        kmin = cutlass.Int32(2147483647)
        for i in cutlass.range(tidx, n4, self.nt, unroll=1):
            w0, w1, w2, w3 = ld_global_nc_v4_u32(row_in.toint() + cutlass.Int64(i) * 16)
            for j in cutlass.range_constexpr(4):
                for h in cutlass.range_constexpr(epw):
                    bits = self._wf_elem_bits((w0, w1, w2, w3)[j], h)
                    b = self.coarse_bin(bits)
                    eoff = cutlass.Int32(j * epw + h)
                    if b > binb:
                        p = smem_atomic_add(s_count, 1)
                        if p < top_k:
                            self._wf_out(
                                out_idx_row, pt_row, p, (i << cutlass.Int32(lg)) + eoff
                            )
                    else:
                        if b == binb:
                            t = smem_atomic_add(s_count + 4, 1)
                            kx = self.exact_key(bits)
                            # key range over ALL ties (staged or not): a
                            # uniform flood past the cap is still exact from
                            # the staged prefix
                            kmax = self._wf_note_tie_max(t, kx, kmax, s_misc)
                            kmin = self._wf_note_tie_min(t, kx, kmin, s_misc)
                            if t < self.tie_cap:
                                s_tk[t] = kx
                                s_ti[t] = (i << cutlass.Int32(lg)) + eoff
        for i in cutlass.range(
            (n4 << cutlass.Int32(lg)) + tidx, len_c, self.nt, unroll=1
        ):
            bits = self.load_scalar(row_in, i)
            b = self.coarse_bin(bits)
            if b > binb:
                p = smem_atomic_add(s_count, 1)
                if p < top_k:
                    self._wf_out(out_idx_row, pt_row, p, cutlass.Int32(i))
            else:
                if b == binb:
                    t = smem_atomic_add(s_count + 4, 1)
                    kx = self.exact_key(bits)
                    kmax = self._wf_note_tie_max(t, kx, kmax, s_misc)
                    kmin = self._wf_note_tie_min(t, kx, kmin, s_misc)
                    if t < self.tie_cap:
                        s_tk[t] = kx
                        s_ti[t] = cutlass.Int32(i)
        self._wf_publish_tie_range(kmax, kmin, s_misc, tidx)
        cute.arch.barrier()  # B3: winners written, ties staged; tail follows

    @cute.jit
    def _wf_elem(self, w, idx, tf_f, sc, s_count, s_hist, s_cv, s_ci):
        """Scalar classify+emit for TAIL elements only (at most 3 per
        row; the hot path is the mask walk in wf_topk_kernel).  Survivor
        iff NOT (v <= TF): NaN survives (torch parity).  Overflow lands
        in the trash slot LCAP; slots < count still routes the row to
        the exact fallback."""
        val = self._wf_f32(w)
        if not (val <= tf_f):
            c = smem_atomic_add(s_count, 1)
            ps = c
            if ps > self.lcap:
                ps = cutlass.Int32(self.lcap)  # trash slot (IMNMX)
            s_cv[ps] = w.bitcast(cutlass.Int32)
            s_ci[ps] = cutlass.Int32(idx)
            cute.arch.red(
                s_hist + self._wf_bin(val, tf_f, sc),
                cutlass.Int32(1),
                op="add",
                dtype="s32",
                sem="relaxed",
                scope="cta",
            )

    @cute.jit
    def _wf_scan_cross0_2t(self, s_hist, target, target2, tidx, s_res):
        """256-bin warp-0-only suffix scan (gvr_2's scan_cross0 idiom,
        zero=True, two=True): publishes the crossing bins for ``target``
        (-> s_res[RES_B]) and ``target2`` (-> s_res[RES_B2]) in one
        barrier-free warp pass and ZEROES the histogram on the way out --
        the walk phase's survivor-histogram clear is folded in for free.
        HOLD path (2 vectors/lane); caller pays exactly one barrier."""
        if tidx < cutlass.Int32(32):
            lane = tidx
            atom = smem_atom_i32_128()
            hbase = s_hist.toint()
            frag0 = cute.make_rmem_tensor((4,), cutlass.Int32)
            frag1 = cute.make_rmem_tensor((4,), cutlass.Int32)
            lds128_i32(atom, hbase, lane * cutlass.Int32(32), frag0)
            lds128_i32(atom, hbase, lane * cutlass.Int32(32) + 16, frag1)
            sm = (
                frag0[0]
                + frag0[1]
                + frag0[2]
                + frag0[3]
                + frag1[0]
                + frag1[1]
                + frag1[2]
                + frag1[3]
            )
            w = warp_incl_scan_add(sm, lane)
            tot = cute.arch.shuffle_sync(w, cutlass.Int32(31))
            after = tot - w  # bins strictly above my span
            base = lane * cutlass.Int32(8)
            zeros = cute.make_rmem_tensor((4,), cutlass.Int32)
            for j in cutlass.range_constexpr(4):
                zeros[j] = cutlass.Int32(0)
            for q in cutlass.range_constexpr(1, -1, -1):
                vv = frag1
                if cutlass.const_expr(q == 0):
                    vv = frag0
                for j in cutlass.range_constexpr(3, -1, -1):
                    cq = vv[j]
                    gb = base + cutlass.Int32(4 * q + j)
                    cross = cutlass.Int32(0)
                    if after < target:
                        if (after + cq) >= target:
                            cross = cutlass.Int32(1)
                        if gb == cutlass.Int32(0):
                            cross = cutlass.Int32(1)
                    if cross != cutlass.Int32(0):
                        s_res[RES_B] = gb
                    cross2 = cutlass.Int32(0)
                    if after < target2:
                        if (after + cq) >= target2:
                            cross2 = cutlass.Int32(1)
                        if gb == cutlass.Int32(0):
                            cross2 = cutlass.Int32(1)
                    if cross2 != cutlass.Int32(0):
                        s_res[RES_B2] = gb
                    after = after + cq
                sts128_i32(atom, zeros, hbase, lane * cutlass.Int32(32) + q * 16)

    @cute.jit
    def _wf_walk_slice(
        self,
        row_in,
        start,
        cnt_sl,
        tf_f,
        sc,
        degen,
        s_count,
        s_h256,
        s_cv,
        s_ci,
        tidx,
    ):
        """One staging pass over this CTA's slice: zero the local count +
        survivor histogram, then the mask walk (see the comment inside).
        Factored out so the T3 retry rung can re-run it with the floor
        threshold; ends with a block barrier (counts/hist visible)."""
        if tidx == 0:
            s_count[0] = cutlass.Int32(0)  # local candidates
        if tidx < 256:
            s_h256[tidx] = cutlass.Int32(0)  # local survivor hist
        cute.arch.barrier()
        if degen == 0:
            # ---- MASK WALK (gvr_2's structure, adopted after two
            # measured dead ends: unconditional per-element side effects
            # were 1.4-3x worse at ~3-6% survivor rates; a monolithic
            # predicated-asm emit lost ptxas's warp aggregation and
            # scheduling and ran ~20% behind this branchy form).
            # Per iteration each thread loads 4 float4s (16 elements)
            # and classifies them into a 16-bit survivor mask -- one
            # FSETP + OR per element, no memory side effects, nothing
            # for ptxas to fence.  The staging reservation is warp-
            # aggregated ONCE per iteration (popc + shfl scan + a
            # single lane-31 atomic), then a divergent bit-walk emits
            # only actual survivors, RELOADING each value with a
            # scalar load: holding all 16 floats live across the emit
            # spills, and the reload hits L1/L2 (gvr_2's measured
            # lesson, replicated here).  Dead-mask form (~dead) keeps
            # one compare per element with NaN surviving (NaN <= TF
            # is false).  The loop trip count is CTA-uniform (full
            # iterations unguarded; the boundary iteration clamps
            # addresses and masks validity) so the warp collectives
            # are always converged.
            # gvr_2's _pin discipline (D:1266): opaque identity movs stop
            # NVVM rematerializing the invariants' defining chains (param
            # base + start mul/add, the lv division ladder) inside the scf
            # while-region -- at the 64-register wall the rematerialized
            # chains cost registers the walk needs.  (Tile-0 register
            # priming on top of these pins still regressed: 16 live primes
            # exceed this kernel's budget even with pinned invariants.)
            lg = cutlass.const_expr(self.vec_elems.bit_length() - 1)
            epw = cutlass.const_expr(self.vec_elems // 4)  # elements per word
            addr = _pin_i64(
                row_in.toint() + cutlass.Int64(start) * cutlass.Int64(self.elem_bytes)
            )
            lv = cnt_sl >> cutlass.Int32(lg)
            lvm1 = _pin_i32(lv - 1)
            stride16 = cutlass.Int64(self.nt * 16)
            lane = tidx % 32
            # 4*nt float4s per iteration
            it_sh = cutlass.const_expr((4 * self.nt).bit_length() - 1)
            n_it = _pin_i32(
                (lv + cutlass.Int32(4 * self.nt - 1)) >> cutlass.Int32(it_sh)
            )
            n_full = _pin_i32(lv >> cutlass.Int32(it_sh))
            it = cutlass.Int32(0)
            vbase = cutlass.Int32(tidx)
            va = addr + cutlass.Int64(tidx) * 16
            tf2 = cutlass.Uint32(0)
            if cutlass.const_expr(self.wf_pair16):
                tf2 = wf_pack_tf16(tf_f, self.dtype == cutlass.BFloat16)
            while it < n_it:
                dead = cutlass.Int32(0)
                # all 4*E element bits valid (E=4 -> 0xFFFF; E=8 -> all 32)
                valid = cutlass.Int32(0xFFFF)
                if cutlass.const_expr(self.vec_elems == 8):
                    valid = cutlass.Int32(-1)
                if it < n_full:  # CTA-uniform: no bounds checks at all
                    a0, a1, a2, a3 = ld_global_nc_v4_u32(va)
                    b0, b1, b2, b3 = ld_global_nc_v4_u32(va + stride16)
                    c0, c1, c2, c3 = ld_global_nc_v4_u32(va + stride16 * 2)
                    d0, d1, d2, d3 = ld_global_nc_v4_u32(va + stride16 * 3)
                    # (An L2 prefetch of the next tile here -- 4 x
                    # prefetch.global.L2, no registers -- was built and
                    # MEASURED: walk 5.26 -> 5.48us at 64K.  The walk takes
                    # the same ~1us per 64KB tile with 8 CTAs or 148 on the
                    # GPU, so it is issue-bound on the classify + survivor
                    # bit-walk, not waiting on HBM; prefetching only adds
                    # instructions.  Removed.)
                    if cutlass.const_expr(self.wf_pair16):
                        bf = cutlass.const_expr(self.dtype == cutlass.BFloat16)
                        dead = wf_dead_v4_16(dead, a0, a1, a2, a3, tf2, 0, bf)
                        dead = wf_dead_v4_16(dead, b0, b1, b2, b3, tf2, 8, bf)
                        dead = wf_dead_v4_16(dead, c0, c1, c2, c3, tf2, 16, bf)
                        dead = wf_dead_v4_16(dead, d0, d1, d2, d3, tf2, 24, bf)
                    else:
                        for j in cutlass.range_constexpr(4):
                            for h in cutlass.range_constexpr(epw):
                                e = j * epw + h
                                fa = self._wf_f32(
                                    self._wf_elem_bits((a0, a1, a2, a3)[j], h)
                                )
                                fb = self._wf_f32(
                                    self._wf_elem_bits((b0, b1, b2, b3)[j], h)
                                )
                                fc = self._wf_f32(
                                    self._wf_elem_bits((c0, c1, c2, c3)[j], h)
                                )
                                fd = self._wf_f32(
                                    self._wf_elem_bits((d0, d1, d2, d3)[j], h)
                                )
                                dead = dead | (cutlass.Int32(fa <= tf_f) << e)
                                dead = dead | (
                                    cutlass.Int32(fb <= tf_f) << (self.vec_elems + e)
                                )
                                dead = dead | (
                                    cutlass.Int32(fc <= tf_f)
                                    << (2 * self.vec_elems + e)
                                )
                                dead = dead | (
                                    cutlass.Int32(fd <= tf_f)
                                    << (3 * self.vec_elems + e)
                                )
                else:  # boundary: clamped addresses + validity bits
                    valid = cutlass.Int32(0)
                    for uu in cutlass.range_constexpr(4):
                        vi = vbase + cutlass.Int32(uu * self.nt)
                        ic = vi
                        if ic > lvm1:
                            ic = lvm1  # clamp (IMNMX); load is harmless
                        e0, e1, e2, e3 = ld_global_nc_v4_u32(
                            addr + cutlass.Int64(ic) * 16
                        )
                        if vi < lv:
                            valid = valid | (
                                cutlass.Int32((1 << self.vec_elems) - 1)
                                << (uu * self.vec_elems)
                            )
                            if cutlass.const_expr(self.wf_pair16):
                                dead = wf_dead_v4_16(
                                    dead,
                                    e0,
                                    e1,
                                    e2,
                                    e3,
                                    tf2,
                                    uu * 8,
                                    self.dtype == cutlass.BFloat16,
                                )
                            else:
                                for j in cutlass.range_constexpr(4):
                                    for h in cutlass.range_constexpr(epw):
                                        fv = self._wf_f32(
                                            self._wf_elem_bits((e0, e1, e2, e3)[j], h)
                                        )
                                        dead = dead | (
                                            cutlass.Int32(fv <= tf_f)
                                            << (uu * self.vec_elems + j * epw + h)
                                        )
                M = (~dead) & valid
                # warp-aggregated reservation: one atomic per warp
                cnt = cutlass.Int32(cute.arch.popc(M))
                inc = warp_inclusive_sum(cnt, lane)
                bpos = cutlass.Int32(0)
                if lane == 31:
                    if inc != 0:
                        bpos = smem_atomic_add(s_count, inc)
                pos = cute.arch.shuffle_sync(bpos, cutlass.Int32(31)) + (inc - cnt)
                # survivor bit-walk (executes ~cnt times, cnt ~ 0-2)
                while M != 0:
                    bp = cutlass.Int32(
                        cute.arch.popc((M & (cutlass.Int32(0) - M)) - cutlass.Int32(1))
                    )
                    M = M & (M - cutlass.Int32(1))
                    fi = vbase + (bp >> cutlass.Int32(lg)) * cutlass.Int32(self.nt)
                    eidx = (
                        start
                        + (fi << cutlass.Int32(lg))
                        + (bp & cutlass.Int32(self.vec_elems - 1))
                    )
                    wbits = self.load_scalar(row_in, eidx)
                    ps = pos
                    if ps > self.lcap:
                        ps = cutlass.Int32(self.lcap)  # trash slot (IMNMX)
                    s_cv[ps] = wbits.bitcast(cutlass.Int32)
                    s_ci[ps] = eidx
                    cute.arch.red(
                        s_h256 + self._wf_bin(self._wf_f32(wbits), tf_f, sc),
                        cutlass.Int32(1),
                        op="add",
                        dtype="s32",
                        sem="relaxed",
                        scope="cta",
                    )
                    pos = pos + cutlass.Int32(1)
                vbase = vbase + cutlass.Int32(4 * self.nt)
                va = va + stride16 * 4
                it = it + cutlass.Int32(1)
            tail_base = lv * cutlass.Int32(self.vec_elems)
            for i in range(tidx, cnt_sl - tail_base, self.nt):
                idx = start + tail_base + i
                self._wf_elem(
                    self.load_scalar(row_in, idx),
                    idx,
                    tf_f,
                    sc,
                    s_count,
                    s_h256,
                    s_cv,
                    s_ci,
                )
        cute.arch.barrier()

    @cute.kernel
    def wf_topk_kernel(
        self,
        input_data: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        mc_state: cute.Tensor,
        page_table: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, sl, _ = cute.arch.block_idx()
        num_rows_dbg, _, _ = cute.arch.grid_dim()
        top_k = cutlass.const_expr(self.top_k)
        n_cols = cutlass.const_expr(input_data.shape[1])
        S = cutlass.const_expr(self.mc_splits)
        # slice-0 test of the short paths: a constant at S == 1 (trace-time
        # ternary), so the single-slice kernels carry neither the ctaid.y read
        # nor the compare on their hot path
        sl_is0 = cutlass.Int32(1) if self.mc_splits == 1 else cutlass.Int32(sl == 0)

        in_ptr = input_data.iterator
        seq_ptr = seqlen.iterator
        oi_ptr = output_indices.iterator
        sl_ptr = slab.iterator
        st_ptr = status.iterator
        mc_ptr = mc_state.iterator
        pt_ptr = page_table.iterator

        smem = SmemAllocator()
        # local candidate stage only (the epilogue reads candidates from
        # the gmem slab directly): LCAP pairs + 1 trash slot each (the
        # branchless overflow sink)
        s_cv = smem.allocate_array(cutlass.Int32, self.lcap + 4, byte_alignment=128)
        s_ci = smem.allocate_array(cutlass.Int32, self.lcap + 4, byte_alignment=128)
        s_h256 = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_tk = smem.allocate_array(cutlass.Uint32, self.tie_cap, byte_alignment=128)
        s_ti = smem.allocate_array(cutlass.Int32, self.tie_cap, byte_alignment=128)
        s_warp_sums = smem.allocate_array(cutlass.Int32, self.nw, byte_alignment=128)
        s_misc = smem.allocate_array(cutlass.Int32, 16, byte_alignment=128)
        # cluster-merged row histogram (the per-CTA s_h256 must stay
        # intact while peers read it, so the merge lands here)
        s_hm = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_count = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)
        # paged output: the staged page-table row and the raw-column stage the
        # finish pass maps out (4-int dummies when unpaged)
        s_pt = smem.allocate_array(
            cutlass.Int32, max(self.wf_pt_cache, 4), byte_alignment=128
        )
        s_stage = smem.allocate_array(
            cutlass.Int32,
            self.top_k if self.wf_page_size > 0 else 4,
            byte_alignment=128,
        )

        row64 = cutlass.Int64(row)
        row_in = in_ptr + row64 * n_cols
        out_idx_row = oi_ptr + row64 * top_k
        # the request's page-table row (paged variants only; request q == row
        # since next_n is 1).  Unpaged launches pass a (1, 4) dummy: address
        # arithmetic only, never dereferenced.  The emit sites receive the
        # triple (smem stage of the leading table entries, global table row,
        # smem raw-column stage); unpaged kernels never unpack it.
        # (unpaged variants never touch the table, not even its shape: every
        # parameter word read here is a constant load in the prologue)
        pt_gmem = (
            pt_ptr + row64 * cutlass.Int64(page_table.shape[1])
            if self.wf_page_size > 0
            else pt_ptr
        )  # trace-time ternary: a bare `if` here would become a DSL region
        pt_row = (s_pt, pt_gmem, s_stage)
        slab_k = sl_ptr + row64 * WF_ROW_INTS
        slab_i = slab_k + GCAP
        slab_h = slab_k + 2 * GCAP  # row survivor histogram: 256 ints
        slab_t = slab_h + 256  # per-CTA publish tables (S > 1 gmem path)
        mc_row = mc_ptr + row64 * 8
        tel_k = cutlass.const_expr(self.wf_telemetry)
        # Sub-phase telemetry uses ONE running mark (ts_m): each mark writes
        # its delta straight into the status buffer instead of holding its
        # own timestamp.  The DSL traces even a const-false `if tel:` region
        # and carries every name assigned inside it as a region result, so
        # each extra Int64 timestamp is a live value in PRODUCTION too --
        # eight of them measured +0.3us at 256K b=1 on the register-bound
        # S=16 kernel.  Blocks: 16 entry->PDL, 17 PDL->ts0, 10-13 sample
        # sub-phases, 19 aim math, 21 tie select.
        ts_m = cutlass.Int64(0)
        if tel_k:
            ts_m = read_clock64()
        # PDL (SM90+, compiled out otherwise): the launch/prologue above may
        # overlap the previous kernel in the stream; wait before the first
        # global read (seqlen, hints, inputs, and this kernel's own self-
        # resetting slab/mc_state from the previous launch).
        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_wait()
        if tel_k:
            if tidx == 0:
                st_ptr[num_rows_dbg * 16 + row] = (read_clock64() - ts_m).to(
                    cutlass.Int32
                )
            ts_m = read_clock64()

        # paged output: load the page-table entries a short-arm row can touch
        # (wf_pt_pre, at most four per thread) BEFORE the length load and store
        # the ones this row needs to the smem stage AFTER it, so both
        # latencies overlap (in-order issue: a store right after its load
        # would stall the warp before the length load goes out).  Every arm
        # places a block barrier before its first emit; identity rows gather
        # from global; rows past small_n stage the rest in the long pipeline.
        pv0 = cutlass.Int32(0)
        pv1 = cutlass.Int32(0)
        pv2 = cutlass.Int32(0)
        pv3 = cutlass.Int32(0)
        # The table may be NARROWER than ceil(N / page_size): a serving engine
        # sizes it for the batch's longest row while the logits buffer is padded
        # wider (SGLang: 129 pages of 64 under 8448 columns).  Never read past
        # its row, and never emit a column it does not cover (the length clamp
        # below).  Trace-time ternary: unpaged variants never touch the table.
        pt_w = (
            cutlass.Int32(page_table.shape[1])
            if self.wf_page_size > 0
            else cutlass.Int32(0)
        )
        if cutlass.const_expr(self.wf_pt_pre > 0):
            n_pre0 = cutlass.Int32(self.wf_pt_pre)
            if n_pre0 > pt_w:
                n_pre0 = pt_w
            if tidx < n_pre0:
                pv0 = pt_gmem[tidx]
            if cutlass.const_expr(self.wf_pt_pre > self.nt):
                j1 = tidx + cutlass.Int32(self.nt)
                if j1 < n_pre0:
                    pv1 = pt_gmem[j1]
            if cutlass.const_expr(self.wf_pt_pre > 2 * self.nt):
                j2 = tidx + cutlass.Int32(2 * self.nt)
                if j2 < n_pre0:
                    pv2 = pt_gmem[j2]
            if cutlass.const_expr(self.wf_pt_pre > 3 * self.nt):
                j3 = tidx + cutlass.Int32(3 * self.nt)
                if j3 < n_pre0:
                    pv3 = pt_gmem[j3]

        length = seq_ptr[row]
        if length < 0:
            length = cutlass.Int32(0)
        if length > n_cols:
            length = cutlass.Int32(n_cols)
        if cutlass.const_expr(self.wf_page_size > 0):
            # a row past the table's coverage is clamped to the covered prefix;
            # a table WIDER than the row can need is counted up to the width's
            # own page count (its full width times the page size need not fit
            # int32; the API bounds the product, this keeps the shift exact)
            pw = pt_w
            if pw > cutlass.Int32(self.wf_pt_pages):
                pw = cutlass.Int32(self.wf_pt_pages)
            cov = pw << cutlass.Int32(self.wf_page_size.bit_length() - 1)
            if length > cov:
                length = cov
        # pages this row can touch (paged variants; trace-time ternary)
        npg = (
            (length + cutlass.Int32(self.wf_page_size - 1))
            >> cutlass.Int32(self.wf_page_size.bit_length() - 1)
            if self.wf_page_size > 0
            else cutlass.Int32(0)
        )

        if top_k >= length:
            if sl_is0 == 1:
                # identity row: every column is a winner.  Paged variants
                # gather the page-table entries straight from the global row
                # (the prefetch above pulled the leading entries into L1): no
                # smem stage and no barrier -- staging cost this arm ~0.14 us
                # (fill + barrier + dependent LDS chain) on the K=2048 identity
                # regime, where the whole row is written in one pass anyway.
                for i in range(tidx, top_k, self.nt):
                    if i < length:
                        self._wf_out_gmem_direct(
                            out_idx_row, pt_row, i, cutlass.Int32(i)
                        )
                    else:
                        out_idx_row[i] = cutlass.Int32(-1)
                if tidx == 0:
                    st_ptr[row] = cutlass.Int32(0)
                    if cutlass.const_expr(self.wf_telemetry):
                        st_ptr[num_rows_dbg + row] = cutlass.Int32(0)
            # exit here (sibling slices too): the kernel's final block sits
            # past the whole long pipeline, a far jump for every identity CTA
            if cutlass.const_expr(self.enable_pdl):
                griddepcontrol_launch_dependents()
            _wf_exit()
        else:
            # stage the prefetched page-table entries THIS ROW needs (pages
            # < npg; only warp 0 stores on decode rows) for the arms and the
            # long pipeline (their first block barrier publishes them; the
            # stage is read only after barriers: _wf_paged_finish, the
            # cluster epilogue)
            if cutlass.const_expr(self.wf_pt_pre > 0):
                n_pre = npg
                if n_pre > cutlass.Int32(self.wf_pt_pre):
                    n_pre = cutlass.Int32(self.wf_pt_pre)
                if tidx < n_pre:
                    s_pt[tidx] = pv0
                if cutlass.const_expr(self.wf_pt_pre > self.nt):
                    j1 = tidx + cutlass.Int32(self.nt)
                    if j1 < n_pre:
                        s_pt[j1] = pv1
                if cutlass.const_expr(self.wf_pt_pre > 2 * self.nt):
                    j2 = tidx + cutlass.Int32(2 * self.nt)
                    if j2 < n_pre:
                        s_pt[j2] = pv2
                if cutlass.const_expr(self.wf_pt_pre > 3 * self.nt):
                    j3 = tidx + cutlass.Int32(3 * self.nt)
                    if j3 < n_pre:
                        s_pt[j3] = pv3
            # ---- SHORT-ROW ARM (per-row runtime gate, gvr_2's varlen
            # idiom: adapt the ALGORITHM per row under a frozen grid).
            # The long pipeline has a ~11us fixed floor (pre-sample,
            # k-derived candidate epilogue, publish/arrival, launch)
            # that does not scale with length: at length N/8 it barely
            # drops while the digit-family cost scales with length
            # (varlen sweep: radix_primitives 5.6us vs our 11.1 vs
            # gvr_2 7.5 on B200 64K-short).  For short rows slice 0
            # solves the row DIRECTLY with the exact MSD radix select
            # (~2 L2-hot passes at these lengths); other slices exit.
            # CTA-uniform (length is per-row), so barriers inside are
            # safe.
            if length <= cutlass.Int32(cutlass.const_expr(self.wf_small_n)):
                if sl_is0 == 1:
                    epi = (st_ptr, row, num_rows_dbg)
                    # census arm FIRST: the decode-row hot path falls from the
                    # prologue straight into its body, and each arm's inline
                    # hot tail exits the CTA in the common case; the register
                    # arm (rows past the census cap) sits behind it.  With the
                    # register arm disabled the census arm takes every row
                    # (its dead body is still compiled).
                    cens_cap = self._wf_census_cap()
                    if length <= cutlass.Int32(cutlass.const_expr(cens_cap)):
                        self._wf_short_row(
                            row_in,
                            length,
                            out_idx_row,
                            pt_row,
                            slab_k,
                            slab_i,
                            s_cv,
                            s_h256,
                            s_tk,
                            s_ti,
                            s_warp_sums,
                            s_misc,
                            s_count,
                            tidx,
                        )
                        self._wf_hot_tail(
                            out_idx_row,
                            pt_row,
                            s_tk,
                            s_ti,
                            s_misc,
                            s_count,
                            tidx,
                            length,
                            epi,
                        )
                    else:
                        self._wf_reg_row(
                            row_in,
                            length,
                            out_idx_row,
                            pt_row,
                            slab_k,
                            slab_i,
                            s_cv,
                            s_h256,
                            s_tk,
                            s_ti,
                            s_warp_sums,
                            s_misc,
                            s_count,
                            tidx,
                            cutlass.const_expr(self.wf_reg_vpt),
                        )
                        self._wf_hot_tail(
                            out_idx_row,
                            pt_row,
                            s_tk,
                            s_ti,
                            s_misc,
                            s_count,
                            tidx,
                            length,
                            epi,
                        )
                    # shared COLD tail for both arms (larger tie sets, floods,
                    # the exact fallback); it ends in the inline epilogue
                    # (slab-histogram reset, status + arm tag, PDL release)
                    # and EXITS the CTA
                    self._wf_short_tail(
                        row_in,
                        length,
                        out_idx_row,
                        pt_row,
                        slab_k,
                        slab_i,
                        s_cv,
                        s_h256,
                        s_tk,
                        s_ti,
                        s_warp_sums,
                        s_misc,
                        s_count,
                        tidx,
                        epi,
                    )
            else:
                # paged: the prologue prefetch covers rows up to small_n; a long
                # row stages the rest of its leading entries here (cold path;
                # the sample phase's barriers publish them before any emit)
                if cutlass.const_expr(self.wf_pt_cache > self.wf_pt_pre):
                    n_st = npg
                    if n_st > cutlass.Int32(self.wf_pt_cache):
                        n_st = cutlass.Int32(self.wf_pt_cache)
                    for j in cutlass.range(
                        tidx + cutlass.Int32(self.wf_pt_pre), n_st, self.nt, unroll=1
                    ):
                        s_pt[j] = pt_gmem[j]
                tel = cutlass.const_expr(self.wf_telemetry)
                # pre-define timestamps: the DSL's staged-control-flow rewriter
                # requires names assigned inside an if region to exist
                # beforehand, even under a const-false condition, and carries
                # them as region results (see ts_m at the kernel top) -- keep
                # this set minimal
                ts0 = cutlass.Int64(0)
                ts1 = cutlass.Int64(0)
                ts2 = cutlass.Int64(0)
                ts3 = cutlass.Int64(0)
                ts3a = cutlass.Int64(0)
                ts3b = cutlass.Int64(0)
                ts4 = cutlass.Int64(0)
                if tel:
                    ts0 = read_clock64()
                    if tidx == 0:  # block 17: PDL released -> ts0 (seqlen + geometry)
                        st_ptr[num_rows_dbg * 17 + row] = (ts0 - ts_m).to(cutlass.Int32)
                    ts_m = ts0
                # slice geometry (walk + sample share it)
                chunk = (
                    (length + cutlass.Int32(S) - 1) // cutlass.Int32(S)
                    + cutlass.Int32(self.vec_elems - 1)
                ) & ~cutlass.Int32(self.vec_elems - 1)
                start = cutlass.Int32(sl) * chunk
                cnt_sl = length - start
                if cnt_sl > chunk:
                    cnt_sl = chunk
                if cnt_sl < 0:
                    cnt_sl = cutlass.Int32(0)
                # ---- 0. threshold from the fused sample (hint-free) ----
                # A previous-step hint could only tighten the sampled
                # threshold (k distinct in-range hints bound the k-th value
                # from below) and measured +1-3 us of gathers per row for no
                # win with realistic hints, so the kernel ignores pre_idx
                # (see get_walkfirst_kernel).
                tf_f = cutlass.Float32(0.0)
                sc = cutlass.Float32(0.0)
                degen = cutlass.Int32(0)
                tf3 = cutlass.Float32(0.0)
                sc3 = cutlass.Float32(0.0)
                # ---- 1. FUSED SAMPLE (gvr_2's streaming-main structure:
                # 3 block barriers total, replica-deterministic, with walk
                # tile 0's latency exposed at walk start).  The
                # earlier separate _pre_scalars paid 5-6 barriers and two
                # serial reduction rounds; the pieces below only work as a
                # WHOLE (measured: priming or the warp fold bolted onto
                # the old structure each regressed).  ----
                lgk = cutlass.const_expr(self.vec_elems.bit_length() - 1)
                epwk = cutlass.const_expr(self.vec_elems // 4)
                n4s = length >> cutlass.Int32(lgk)
                p4 = cutlass.Int32(
                    (cutlass.Int64(tidx) * cutlass.Int64(n4s)) // cutlass.Int64(self.nt)
                )
                if p4 > n4s - 1:
                    p4 = n4s - 1
                if p4 < 0:
                    p4 = cutlass.Int32(0)
                w0, w1, w2, w3 = ld_global_nc_v4_u32(
                    row_in.toint() + cutlass.Int64(p4) * cutlass.Int64(16)
                )
                # register key fold -> warp redux (fkey space) -> per-warp
                # partials (owned slots: no zero-init round).  Non-finite
                # samples stay OUT of the min / max: one sampled +/-inf made
                # the span infinite, which zeroed the survivor scale and sent
                # every such row through the exact fallback (a 5-10x cliff
                # for long rows with sparse -inf masking).  Without them the
                # threshold is finite: -inf never survives the walk, +inf and
                # +NaN survive into the saturated top bin, and a sample with
                # no finite element at all leaves the fold empty (kmin > kmax,
                # span NaN) and takes the degenerate path below.
                kfin_hi = cutlass.Uint32(
                    0xFF7FFFFF
                    if self.is_f32
                    else (0xFBFF if self.dtype == cutlass.Float16 else 0xFF7F)
                )  # key of the largest finite value
                kfin_lo = cutlass.Uint32(
                    0x00800000
                    if self.is_f32
                    else (0x0400 if self.dtype == cutlass.Float16 else 0x0080)
                )  # key of the most negative finite value
                kk = cutlass.Uint32(0)  # identity of the max fold
                nk = cutlass.Uint32(0)  # identity of the max(~key) (min) fold
                for j_ in cutlass.range_constexpr(4):
                    for h_ in cutlass.range_constexpr(epwk):
                        k2 = self.exact_key(
                            self._wf_elem_bits((w0, w1, w2, w3)[j_], h_)
                        )
                        if k2 <= kfin_hi:
                            if k2 > kk:
                                kk = k2
                        if k2 >= kfin_lo:
                            n2 = ~k2
                            if n2 > nk:
                                nk = n2
                kk = warp_max_u32(kk)
                nk = warp_max_u32(nk)
                if tel:  # block 10: sample vector loaded + folded
                    if tidx == 0:
                        st_ptr[num_rows_dbg * 10 + row] = (read_clock64() - ts_m).to(
                            cutlass.Int32
                        )
                    ts_m = read_clock64()
                lane_s = tidx % 32
                warp_s = tidx // 32
                if lane_s == 0:
                    s_warp_sums[warp_s] = kk.bitcast(cutlass.Int32)
                    s_count[warp_s] = nk.bitcast(cutlass.Int32)
                if tidx < 256:
                    s_h256[tidx] = cutlass.Int32(0)
                if tel:
                    if tidx == 0:
                        s_misc[3] = cutlass.Int32(0)  # max per-thread walk time
                        s_misc[5] = cutlass.Int32(0)  # 0x7FFFFFFF - min walk time
                cute.arch.barrier()  # B1: partials + hist zeros
                # cross-warp fold: ONE lane-indexed load + ONE redux per
                # value (a 32-iteration serial loop here was measured slow)
                # lane-indexed partial reads: only nw warps wrote a slot
                # (16 at 512 threads); lanes beyond contribute 0 (neutral
                # for both maxes)
                kmx_p = cutlass.Uint32(0)
                kmn_p = cutlass.Uint32(0)
                if lane_s < cutlass.Int32(self.nw):
                    kmx_p = cutlass.Uint32(s_warp_sums[lane_s])
                    kmn_p = cutlass.Uint32(s_count[lane_s])
                kmax = warp_max_u32(kmx_p)
                kminc = warp_max_u32(kmn_p)
                kmin = ~kminc
                if tel:  # block 11: B1 + cross-warp fold
                    if tidx == 0:
                        st_ptr[num_rows_dbg * 11 + row] = (read_clock64() - ts_m).to(
                            cutlass.Int32
                        )
                    ts_m = read_clock64()
                smin_f = self._wf_val_of_key(kmin)
                smax_f = self._wf_val_of_key(kmax)
                span = smax_f - smin_f
                # degenerate unless the span is a positive finite number: an
                # empty fold gives NaN, finite extremes of opposite sign can
                # overflow to +inf, and a single sampled value gives 0
                degen = cutlass.Int32(1)
                if span > cutlass.Float32(0.0):
                    if span <= cutlass.Float32(3.4028234663852886e38):
                        degen = cutlass.Int32(0)
                if kmin == kmax:
                    degen = cutlass.Int32(1)
                if degen == 0:
                    sc0 = cutlass.Float32(256.0) / span
                    for j_ in cutlass.range_constexpr(4):
                        for h_ in cutlass.range_constexpr(epwk):
                            vj = self._wf_f32(
                                self._wf_elem_bits((w0, w1, w2, w3)[j_], h_)
                            )
                            bs = cutlass.Int32((vj - smin_f) * sc0)
                            if bs < 0:
                                bs = cutlass.Int32(0)
                            if bs > 255:
                                bs = cutlass.Int32(255)
                            smem_atomic_add(s_h256 + bs, 1)
                cute.arch.barrier()  # B2: sample histogram
                if tel:  # block 12: sample histogram + B2
                    if tidx == 0:
                        st_ptr[num_rows_dbg * 12 + row] = (read_clock64() - ts_m).to(
                            cutlass.Int32
                        )
                    ts_m = read_clock64()
                retry_cap = cutlass.const_expr(
                    self.mc_splits == 1 or (self.wf_cluster and self.mc_splits <= 4)
                )
                if cutlass.const_expr(retry_cap):
                    margin = cutlass.Int32(top_k) >> cutlass.Int32(1)
                    lm = length >> cutlass.Int32(8)
                    if lm > margin:
                        margin = lm
                    aim = cutlass.Int32(top_k) + margin
                    # Tail-safe floor for wide batches.  The count above
                    # the sample-derived bar is ~ N(aim, sqrt(aim*len/P)),
                    # so a fixed k/2 margin is only ~2.3 sigma at 256K
                    # k=2048 (r_aim = 48 samples): ~1% of rows undershoot
                    # and take the T3 re-walk (+16us).  With >= 32 rows in
                    # flight the wall time is the SLOWEST row, so most
                    # batches paid that tail (256K b=148 k=2048 measured
                    # bimodal 35 / 51us).  Raise the aim to z = 3.5 sigma:
                    # aim - k >= z*sqrt(aim*len/P)  <=>  sqrt(aim) >=
                    # (z*q + sqrt(z^2 q^2 + 4k)) / 2 with q = sqrt(len/P).
                    # Applied ONLY where the fixed margin is thin (below
                    # 2.5 sigma: k=2048 at 256K-512K, where N/256 < k/2);
                    # elsewhere (z >= 2.8 at 256K k=1024, 4.6 at 64K
                    # k=2048, 3.3 at 1M k=2048) the aim is unchanged, so
                    # no cell outside the band pays anything (a blanket
                    # 3.5-sigma floor measured +2-4% median on the wide
                    # cells it did not need to touch).  Small grids keep
                    # the lean aim (expected retry cost ~1% x 16us).
                    if num_rows_dbg >= cutlass.Int32(32):
                        lenf = cutlass.Float32(length) / cutlass.Float32(self.psamp)
                        mf = cutlass.Float32(aim - cutlass.Int32(top_k))
                        var = cutlass.Float32(aim) * lenf  # sigma^2 of the count
                        if mf * mf < cutlass.Float32(6.25) * var:  # z < 2.5
                            q = cmath.sqrt(lenf)
                            zq = cutlass.Float32(3.5) * q
                            s_ = (
                                zq + cmath.sqrt(zq * zq + cutlass.Float32(4.0 * top_k))
                            ) * cutlass.Float32(0.5)
                            aim_stat = cutlass.Int32(s_ * s_) + cutlass.Int32(1)
                            if aim_stat > aim:
                                aim = aim_stat
                else:
                    aim = cutlass.Int32(top_k) * cutlass.Int32(2)
                    la = length >> cutlass.Int32(7)
                    if la > aim:
                        aim = la
                if aim > cutlass.Int32(3 * SCAP // 4):
                    aim = cutlass.Int32(3 * SCAP // 4)
                # ... and to the per-CTA smem stage: each CTA stages
                # ~aim/S candidates into lcap slots.  Unclamped, N >= 2M
                # aimed above the stage on EVERY row (9216 > 8192 at
                # 2M), turning the whole batch into re-walks (S == 1)
                # or exact fallbacks (S > 1) -- 3-4x wall time.
                aim_cap = cutlass.Int32(
                    cutlass.const_expr(3 * self.lcap * self.mc_splits // 4)
                )
                if aim > aim_cap:
                    aim = aim_cap
                r_aim = cutlass.Int32(
                    (cutlass.Int64(aim) * cutlass.Int64(self.psamp))
                    // cutlass.Int64(length)
                )
                if r_aim < 1:
                    r_aim = cutlass.Int32(1)
                if r_aim > self.psamp:
                    r_aim = cutlass.Int32(self.psamp)
                r3 = r_aim * cutlass.Int32(2)
                if r3 > self.psamp:
                    r3 = cutlass.Int32(self.psamp)
                # ONE warp-0 pass: both ladder crossings + free re-zero of
                # s_h256 for the walk's survivor histogram.  (Two
                # alternatives were built and MEASURED WORSE or equal on
                # B200 at 64K: a 256-thread scan with a named barrier
                # (0.73us, same) and a redundant per-warp scan without the
                # B3 barrier (1.3us: 32 warps issuing the same ~900-cycle
                # dependent chain saturate the 4 schedulers, so idling 31
                # warps behind one barrier is the cheaper shape).)
                self._wf_scan_cross0_2t(s_h256, r_aim, r3, tidx, s_misc)
                cute.arch.barrier()  # B3: scan publish + zeros
                if tel:  # block 13: crossing scan + B3
                    if tidx == 0:
                        st_ptr[num_rows_dbg * 13 + row] = (read_clock64() - ts_m).to(
                            cutlass.Int32
                        )
                    ts_m = read_clock64()
                bkt = s_misc[RES_B]
                bkt3 = s_misc[RES_B2]
                w_bin = span / cutlass.Float32(256.0)
                tf_f = smin_f + cutlass.Float32(bkt) * w_bin
                tf3 = smin_f + cutlass.Float32(bkt3) * w_bin
                if degen == 1:
                    tf_f = smin_f  # unused: the walk is skipped under degen (cand = 0 -> fallback)
                    tf3 = smin_f
                # survivor-hist span 1.5x past the SAMPLE max: the top
                # bin saturates, and the ~N/4096 elements above the
                # sample max (>k on ~2% of fp32 rows) otherwise pile
                # into bin 255 -- when the rank-k crossing lands there
                # and the pile exceeds the tie stage the row takes the
                # exact fallback (measured: one row turned a 53us
                # batch into 320us).  1.5x wider bins cost tens of
                # extra ties in the rank-k bin; no cliff.
                sspan = (smax_f - tf_f) * WF_SPAN_EXT
                sc = cutlass.Float32(0.0)
                if sspan > cutlass.Float32(0.0):
                    sc = cutlass.Float32(255.0) / sspan
                sspan3 = (smax_f - tf3) * WF_SPAN_EXT
                sc3 = cutlass.Float32(0.0)
                if sspan3 > cutlass.Float32(0.0):
                    sc3 = cutlass.Float32(255.0) / sspan3
                if tel:
                    ts1 = read_clock64()
                    if tidx == 0:  # block 19: aim / threshold arithmetic
                        st_ptr[num_rows_dbg * 19 + row] = (ts1 - ts_m).to(cutlass.Int32)
                self._wf_walk_slice(
                    row_in,
                    start,
                    cnt_sl,
                    tf_f,
                    sc,
                    degen,
                    s_count,
                    s_h256,
                    s_cv,
                    s_ci,
                    tidx,
                )
                if tel:
                    ts2 = read_clock64()
                    # per-thread walk-time spread (straggler diagnosis)
                    wdt = cutlass.Uint32((ts2 - ts1).to(cutlass.Int32))
                    smem_red_max_u32(s_misc + 3, wdt)
                    smem_red_max_u32(s_misc + 5, cutlass.Uint32(0x7FFFFFFF) - wdt)

                if cutlass.const_expr(self.wf_cluster and S > 1):
                    # ---- CLUSTER EPILOGUE (the S CTAs of a row = one hw
                    # cluster).  DSMEM replaces the whole gmem coordination
                    # layer: counts and histograms are read from peer smem
                    # (one mapa per rank), classification is DISTRIBUTED --
                    # each CTA classifies the candidates it staged, winners
                    # go straight to the gmem output at a DSMEM cursor on
                    # rank 0, bin ties are DSMEM-staged into rank 0's tie
                    # arrays -- and rank 0 alone runs the key-exact tie
                    # sub-select or the fused fallback.  Phase telemetry
                    # showed the gmem path pays a flat ~1.8us hist gather
                    # + ~1us slab emit + arrival skew after the slowest
                    # CTA; DSMEM removes all three.  No SCAP bound here:
                    # capacity is per-CTA lcap only.  All locals are
                    # cl_-prefixed: a const-eliminated branch still
                    # registers its names, and reusing the gmem branch's
                    # names would break its staged-control-flow joins.
                    cl_local = s_count[0]
                    cl_failo = cutlass.Int32(0)
                    if cl_local > self.lcap:
                        cl_failo = cutlass.Int32(1)
                    if tidx == 0:
                        s_misc[12] = cl_local
                        s_misc[13] = cl_failo
                        s_misc[10] = cutlass.Int32(0)  # winner cursor (rank 0)
                        s_count[4] = cutlass.Int32(0)  # tie cursor (rank 0)
                    _cluster_sync_aligned()
                    if tel:
                        ts3 = read_clock64()
                        ts3a = ts3
                        ts3b = ts3
                    # replicated peer reads: identical result on every CTA,
                    # so cl_ok is cluster-uniform and the barriers inside
                    # the guarded region are safe
                    cl_cand = cutlass.Int32(0)
                    cl_failc = cutlass.Int32(0)
                    for cl_r in cutlass.range_constexpr(S):
                        cl_pm = _mapa_shared_cluster(s_misc, cutlass.Int32(cl_r))
                        cl_cand += _ld_shared_cluster_i32(cl_pm + cutlass.Int32(12 * 4))
                        cl_failc += _ld_shared_cluster_i32(
                            cl_pm + cutlass.Int32(13 * 4)
                        )
                    cl_ok = cutlass.Int32(0)
                    if degen == 0 and cl_failc == 0 and cl_cand >= top_k:
                        cl_ok = cutlass.Int32(1)
                    # ---- T3 retry rung (gvr_2 ladder, cluster form) ----
                    # Undershoot with clean staging: the tight aim missed k.
                    # The verdict is cluster-uniform (merged counts + degen
                    # are identical on every CTA), so every CTA re-walks at
                    # the T3 floor together -- the barriers inside the walk
                    # and the extra cluster sync cannot deadlock.  tf/sc are
                    # rebound so the epilogue's classify matches the
                    # retried histogram (single-binning-function rule).
                    cl_retry = cutlass.Int32(0)
                    if degen == 0 and cl_failc == 0 and cl_cand < top_k:
                        cl_retry = cutlass.Int32(1)
                    if cl_retry == 1:
                        tf_f = tf3
                        sc = sc3
                        self._wf_walk_slice(
                            row_in,
                            start,
                            cnt_sl,
                            tf_f,
                            sc,
                            degen,
                            s_count,
                            s_h256,
                            s_cv,
                            s_ci,
                            tidx,
                        )
                        cl_local = s_count[0]
                        cl_failo = cutlass.Int32(0)
                        if cl_local > self.lcap:
                            cl_failo = cutlass.Int32(1)
                        if tidx == 0:
                            s_misc[12] = cl_local
                            s_misc[13] = cl_failo
                        _cluster_sync_aligned()
                        cl_cand = cutlass.Int32(0)
                        cl_failc = cutlass.Int32(0)
                        for cl_r in cutlass.range_constexpr(S):
                            cl_pm = _mapa_shared_cluster(s_misc, cutlass.Int32(cl_r))
                            cl_cand += _ld_shared_cluster_i32(
                                cl_pm + cutlass.Int32(12 * 4)
                            )
                            cl_failc += _ld_shared_cluster_i32(
                                cl_pm + cutlass.Int32(13 * 4)
                            )
                        if degen == 0 and cl_failc == 0 and cl_cand >= top_k:
                            cl_ok = cutlass.Int32(1)
                    cl_binb = cutlass.Int32(0)
                    cl_above = cutlass.Int32(0)
                    if cl_ok == 1:
                        if tidx < 256:
                            cl_hv = cutlass.Int32(0)
                            for cl_r in cutlass.range_constexpr(S):
                                cl_ph = _mapa_shared_cluster(
                                    s_h256, cutlass.Int32(cl_r)
                                )
                                cl_hv += _ld_shared_cluster_i32(
                                    cl_ph + tidx * cutlass.Int32(4)
                                )
                            s_hm[tidx] = cl_hv
                        cute.arch.barrier()
                        self.scan256_and_find(
                            s_hm,
                            cl_cand,
                            cutlass.Int32(top_k),
                            s_warp_sums,
                            s_misc,
                            tidx,
                        )
                        # no post-scan barrier: scan256_and_find ends
                        # with a full-block barrier and the classify's
                        # smem traffic (DSMEM cursors zeroed before the
                        # first cluster sync, fresh s_tk/s_ti) is disjoint
                        cl_binb = s_misc[0]
                        cl_above = s_misc[1]
                        if tel:
                            ts3a = read_clock64()
                        # distributed classify of this CTA's OWN candidates
                        cl_wcur = _mapa_shared_cluster(
                            s_misc, cutlass.Int32(0)
                        ) + cutlass.Int32(10 * 4)
                        cl_tcur = _mapa_shared_cluster(
                            s_count, cutlass.Int32(0)
                        ) + cutlass.Int32(4 * 4)
                        cl_tk0 = _mapa_shared_cluster(s_tk, cutlass.Int32(0))
                        cl_ti0 = _mapa_shared_cluster(s_ti, cutlass.Int32(0))
                        for cl_t in range(tidx, cl_local, self.nt):
                            cl_wv = cutlass.Uint32(s_cv[cl_t])
                            cl_val = self._wf_f32(cl_wv)
                            cl_b = self._wf_bin(cl_val, tf_f, sc)
                            if cl_b > cl_binb:
                                cl_p = _atom_shared_cluster_add_i32(
                                    cl_wcur, cutlass.Int32(1)
                                )
                                if cl_p < top_k:
                                    # every CTA writes its own winners: no stage
                                    self._wf_out_gmem(
                                        out_idx_row, pt_row, cl_p, s_ci[cl_t]
                                    )
                            else:
                                if cl_b == cl_binb:
                                    cl_e = _atom_shared_cluster_add_i32(
                                        cl_tcur, cutlass.Int32(1)
                                    )
                                    if cl_e < self.tie_cap:
                                        _st_shared_cluster_i32(
                                            cl_tk0 + cl_e * cutlass.Int32(4),
                                            self.exact_key(cl_wv).bitcast(
                                                cutlass.Int32
                                            ),
                                        )
                                        _st_shared_cluster_i32(
                                            cl_ti0 + cl_e * cutlass.Int32(4),
                                            s_ci[cl_t],
                                        )
                    _cluster_sync_aligned()
                    if tel:
                        ts3b = read_clock64()
                    if sl == 0:
                        if cl_ok == 1:  # CTA-uniform on rank 0
                            cl_nb = s_count[4]
                            cl_rem = cutlass.Int32(top_k) - cl_above
                            if cl_rem < 0:
                                cl_ok = cutlass.Int32(0)
                            if cl_rem > cl_nb:
                                cl_ok = cutlass.Int32(0)
                            if cl_nb > self.tie_cap:
                                cl_ok = cutlass.Int32(0)
                            if cl_ok == 1:
                                if cl_rem > 0:
                                    if cl_nb <= 128:
                                        self._wf_tie_select_small(
                                            s_tk,
                                            s_ti,
                                            cl_nb,
                                            cl_above,
                                            cl_rem,
                                            s_cv,  # >= 512 ints scratch
                                            s_warp_sums,
                                            s_misc,
                                            out_idx_row,
                                            pt_row,
                                            tidx,
                                        )
                                    else:
                                        # exact key bounds of the rank-k
                                        # value bin (one-bin float margin,
                                        # as in the gmem epilogue)
                                        cl_bw = cutlass.Float32(1.0)
                                        if sc > cutlass.Float32(0.0):
                                            cl_bw = cutlass.Float32(1.0) / sc
                                        cl_blo = (
                                            tf_f + cutlass.Float32(cl_binb - 1) * cl_bw
                                        )
                                        cl_bhi = cl_blo + cl_bw * (cutlass.Float32(3.0))
                                        cl_klo = self._wf_key_of_f32(cl_blo)
                                        cl_khi = self._wf_key_of_f32(cl_bhi)
                                        if cl_binb == 255:
                                            cl_khi = cutlass.Uint32(
                                                0xFFFFFFFF if self.is_f32 else 0xFFFF
                                            )  # NaN top; 16-bit keys stop at 0xFFFF
                                        if sc <= cutlass.Float32(0.0):
                                            # sc == 0: full key range, no round
                                            # skipped (see the gmem epilogue)
                                            cl_klo = cutlass.Uint32(0)
                                            cl_khi = cutlass.Uint32(
                                                0xFFFFFFFF if self.is_f32 else 0xFFFF
                                            )
                                        self._tie_select_smem_skip_wf(
                                            s_tk,
                                            s_ti,
                                            cl_nb,
                                            cl_above,
                                            cl_rem,
                                            cl_khi,
                                            cl_klo,
                                            s_h256,
                                            s_warp_sums,
                                            s_misc,
                                            out_idx_row,
                                            pt_row,
                                            tidx,
                                        )
                                # paged: the rank-0 tail's ties sit in the stage
                                # (the winners went straight to gmem above)
                                self._wf_paged_finish(
                                    out_idx_row, pt_row, cl_above, cl_rem, tidx
                                )
                        if cl_ok == 0:
                            # inline exact fallback (CTA-uniform cl_ok)
                            self._fallback_row(  # type: ignore[attr-defined]  # Prod MRO
                                row_in,
                                length,
                                out_idx_row,
                                slab_k,
                                slab_i,
                                s_cv,
                                s_h256,
                                s_warp_sums,
                                s_misc,
                                s_count + 8,
                                s_count + 16,
                                tidx,
                            )
                            self._wf_page_fixup(
                                out_idx_row, pt_row, cutlass.Int32(0), top_k, tidx
                            )
                        if tel:
                            ts4 = read_clock64()
                        if tidx == 0:
                            st_ptr[row] = cutlass.Int32(1) - cl_ok
                            if tel:  # blocks 1-3: telemetry builds only
                                st_ptr[num_rows_dbg + row] = cutlass.Int32(6)
                                st_ptr[num_rows_dbg * 2 + row] = cl_cand
                                st_ptr[num_rows_dbg * 3 + row] = cl_failc
                            if tel:
                                st_ptr[num_rows_dbg * 5 + row] = (ts1 - ts0).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 6 + row] = (ts2 - ts1).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 7 + row] = (ts4 - ts3b).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 8 + row] = (ts3a - ts3).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 9 + row] = (ts3b - ts3a).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 14 + row] = (ts3 - ts2).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 15 + row] = s_count[4]  # ties
                else:
                    # bulk publish: one range reservation + copies + 256 red.adds.
                    # S == 1 SHORT-CIRCUIT: the sole CTA is its own last arriver
                    # and s_cv/s_ci/s_h256 already hold the full row's survivors
                    # and histogram, so the slab round trip, the gmem histogram
                    # merge, and the arrival protocol are all skipped (measured
                    # ~1.5-2us of gmem atomics + copies + gather at b >= 128).
                    local = s_count[0]
                    staged = local
                    if staged > self.lcap:
                        staged = cutlass.Int32(self.lcap)
                    if cutlass.const_expr(S > 1):
                        # BIN-MAJOR PUBLISH.  Exclusive prefix of this CTA's
                        # 256-bin survivor histogram (warp 0, 8 bins/lane) ->
                        # s_hm; the per-CTA table [base, staged, prefix[0..256]]
                        # goes to the slab; then the staged candidates are
                        # scattered by bin (s_hm doubles as the per-bin cursor).
                        # The last arriver then copies bins > B as winners and
                        # bin B as ties by RANGE -- the per-candidate classify
                        # it used to run measured 2.5-3us per row (1M b <= 8).
                        if tidx == 0:
                            s_misc[10] = cutlass.Int32(0)
                            if local > 0:
                                gmem_atomic_add(mc_row + 0, local)  # exact count
                                s_misc[10] = cutlass.Int32(
                                    gmem_atomic_add(mc_row + 1, staged)
                                )
                        if tidx < 32:
                            bm_sum = cutlass.Int32(0)
                            for bm_j in cutlass.range_constexpr(8):
                                bm_sum += s_h256[tidx * 8 + bm_j]
                            bm_inc = warp_inclusive_sum(bm_sum, tidx)
                            bm_run = (
                                bm_inc - bm_sum
                            )  # exclusive prefix of this lane's 8 bins
                            for bm_j in cutlass.range_constexpr(8):
                                s_hm[tidx * 8 + bm_j] = bm_run
                                bm_run += s_h256[tidx * 8 + bm_j]
                            if tidx == 31:
                                s_misc[14] = bm_inc  # total survivors (== local)
                        cute.arch.barrier()
                        base = s_misc[10]
                        bm_tbl = slab_t + cutlass.Int32(sl) * WF_TBL
                        if tidx < 256:
                            bm_tbl[2 + tidx] = s_hm[tidx]
                        if tidx == 0:
                            bm_tbl[0] = base
                            bm_tbl[1] = staged
                            bm_tbl[2 + 256] = s_misc[14]
                        # The scatter below advances s_hm (it doubles as the
                        # per-bin cursor); every thread must have read its
                        # prefix entry for the table first.  Without this
                        # barrier a warp that starts scattering early bumps
                        # bins whose table copy is still pending, the
                        # published prefix overstates those bins, the last
                        # arriver copies too few winners, and the unfilled
                        # stage slots feed garbage columns to the paged finish
                        # (illegal smem address in SGLang's V3.2 TP8 decode;
                        # unpaged variants silently dropped winners).
                        cute.arch.barrier()
                        for t in range(tidx, staged, self.nt):
                            wv_t = cutlass.Uint32(s_cv[t])
                            b_t = self._wf_bin(self._wf_f32(wv_t), tf_f, sc)
                            pos_t = smem_atomic_add(s_hm + b_t, 1)
                            p = base + pos_t
                            if p < SCAP:
                                slab_k[p] = s_cv[t]
                                slab_i[p] = s_ci[t]
                        if tidx < 256:
                            hv = s_h256[tidx]
                            if hv > 0:
                                gmem_red_add(slab_h + tidx, hv)

                    # ---- 3. last-arriver epilogue (no spins anywhere) ----
                    cute.arch.barrier()
                    if tel:
                        ts3 = read_clock64()
                    if tidx == 0:
                        s_misc[11] = cutlass.Int32(0)
                        if cutlass.const_expr(S > 1):
                            # acq_rel: release orders this CTA's publish before
                            # its arrival; acquire makes every earlier arriver's
                            # publish visible to the last one -- ONE round trip
                            old = cute.arch.atomic_add(
                                mc_row + 2, cutlass.Int32(1), sem="acq_rel", scope="gpu"
                            )
                            if old == cutlass.Int32(S - 1):
                                s_misc[11] = cutlass.Int32(1)
                        else:
                            s_misc[11] = cutlass.Int32(1)
                    cute.arch.barrier()
                    if s_misc[11] == 1:
                        cand = local
                        slots = staged  # < cand iff s_cv overflowed => fallback
                        if cutlass.const_expr(S == 1):
                            # ---- T3 retry rung (gvr_2 ladder, S==1 form):
                            # CTA-uniform verdict; tf/sc rebound so the
                            # classify matches the retried histogram
                            if degen == 0 and cand < top_k and slots == cand:
                                tf_f = tf3
                                sc = sc3
                                self._wf_walk_slice(
                                    row_in,
                                    start,
                                    cnt_sl,
                                    tf_f,
                                    sc,
                                    degen,
                                    s_count,
                                    s_h256,
                                    s_cv,
                                    s_ci,
                                    tidx,
                                )
                                cand = s_count[0]
                                slots = cand
                                if slots > self.lcap:
                                    slots = cutlass.Int32(self.lcap)
                            # ---- overflow rung (S==1 form): the aim
                            # OVERSHOT the staging capacity (slots < cand).
                            # s_h256 is the complete survivor histogram
                            # (the walk bins every survivor, staged or
                            # not), so the rank-lcap crossing names a
                            # higher threshold with < lcap survivors by
                            # construction; re-walk there when it still
                            # covers k.  Before this rung a ~1.7x sample
                            # overshoot (3/256 fp32 rows at 1M, k=1024)
                            # cost the ~500us fused fallback per row.
                            if degen == 0 and slots < cand:
                                # rank lcap-64, not lcap: the retried walk's
                                # bin-edge rounding can move a few elements,
                                # and a landing at lcap+3 is another overflow
                                # (seen once on L40S at 1M b=256)
                                self.scan256_and_find(
                                    s_h256,
                                    cand,
                                    cutlass.Int32(self.lcap - 64),
                                    s_warp_sums,
                                    s_misc,
                                    tidx,
                                )
                                ob = s_misc[0]
                                oabove = s_misc[1]
                                if oabove >= top_k and ob < 254:
                                    # raise tf to bin ob's upper edge; keep
                                    # the 255-bin scale over [tf', smax]
                                    obw = cutlass.Float32(ob + 1)
                                    tf_f = tf_f + obw / sc
                                    sc = (
                                        sc
                                        * cutlass.Float32(255.0)
                                        / (cutlass.Float32(255.0) - obw)
                                    )
                                    self._wf_walk_slice(
                                        row_in,
                                        start,
                                        cnt_sl,
                                        tf_f,
                                        sc,
                                        degen,
                                        s_count,
                                        s_h256,
                                        s_cv,
                                        s_ci,
                                        tidx,
                                    )
                                    cand = s_count[0]
                                    slots = cand
                                    if slots > self.lcap:
                                        slots = cutlass.Int32(self.lcap)
                        if cutlass.const_expr(S > 1):
                            cand = cutlass.Int32(mc_row[0])
                            slots = cutlass.Int32(mc_row[1])
                            # histogram gather issued with the counts (one
                            # gmem round trip; final under the acquire above).
                            # (Copying all S publish tables into smem here was
                            # measured WORSE: 8 serialized loads per thread,
                            # +2.3us on the last arriver's chain.)
                            if tidx < 256:
                                s_h256[tidx] = cutlass.Int32(slab_h[tidx])
                            cute.arch.barrier()
                        if tel:
                            ts3a = ts3  # epilogue sub-phase marks (telemetry)
                            ts3b = ts3
                        ok = cutlass.Int32(0)
                        if (
                            degen == 0
                            and cand >= top_k
                            and cand <= SCAP
                            and slots == cand
                        ):
                            ok = cutlass.Int32(1)
                            # find rank-k bin over the row survivor histogram
                            # (at S == 1 s_h256 IS the row histogram; at S > 1
                            # it was gathered above)
                            self.scan256_and_find(
                                s_h256,
                                cand,
                                cutlass.Int32(top_k),
                                s_warp_sums,
                                s_misc,
                                tidx,
                            )
                            # no post-scan barrier: scan256_and_find
                            # ends with a full-block barrier; the cursor
                            # zeroing below is fenced by its own barrier
                            binb = s_misc[0]
                            above = s_misc[1]
                            if tel:
                                ts3a = read_clock64()  # hist gather + scan done
                            # emit winners above bin B; stage bin-B ties for the
                            # exact key-space sub-select
                            if tidx == 0:
                                s_misc[10] = cutlass.Int32(0)  # winner cursor
                                s_count[4] = cutlass.Int32(0)  # tie cursor
                            cute.arch.barrier()
                            if cutlass.const_expr(S == 1):
                                # candidates never left shared memory: classify
                                # straight from s_cv/s_ci with one same-address
                                # shared atomic per winner / tie.  Two warp-
                                # aggregated forms were built here and MEASURED
                                # SLOWER on B200 (64K, k=2048, ~3.1k candidates):
                                # a 5-step shuffle scan per warp (1.1 -> 2.0us)
                                # and ballot + popc + lane-0 atomic + shuffle
                                # (1.1 -> 1.5us).  The hardware already
                                # coalesces same-address shared atomics well
                                # enough that the extra collectives and the
                                # uniform-trip-count loop cost more than they
                                # save; keep the plain form.
                                for t in range(tidx, cand, self.nt):
                                    wv = cutlass.Uint32(s_cv[t])
                                    val = self._wf_f32(wv)
                                    b = self._wf_bin(val, tf_f, sc)
                                    if b > binb:
                                        p = smem_atomic_add(s_misc + 10, 1)
                                        if p < top_k:
                                            self._wf_out(
                                                out_idx_row, pt_row, p, s_ci[t]
                                            )
                                    else:
                                        if b == binb:
                                            t2 = smem_atomic_add(s_count + 4, 1)
                                            if t2 < self.tie_cap:
                                                s_tk[t2] = self.exact_key(wv)
                                                s_ti[t2] = s_ci[t]
                            else:
                                # RANGE COPY from the bin-major slab.  Per CTA r:
                                # winners = [base_r + pre_r[B+1], base_r + staged_r),
                                # ties    = [base_r + pre_r[B],   base_r + pre_r[B+1]).
                                # Lane r < S of warp 0 loads its CTA's four table
                                # words (one round trip), a warp scan gives the
                                # output offsets, then every thread copies one
                                # winner index / one tie (key + index) per step.
                                bm_win = cutlass.Int32(0)
                                bm_tie = cutlass.Int32(0)
                                bm_wsrc = cutlass.Int32(0)
                                bm_tsrc = cutlass.Int32(0)
                                if tidx < 32:
                                    if tidx < cutlass.Int32(S):
                                        bm_t = (
                                            slab_t + tidx * WF_TBL
                                        )  # lane r reads CTA r's table
                                        bm_base = bm_t[0]
                                        bm_stg = bm_t[1]
                                        bm_pb = bm_t[2 + binb]
                                        bm_pb1 = bm_t[3 + binb]
                                        bm_wsrc = bm_base + bm_pb1
                                        bm_win = bm_stg - bm_pb1
                                        bm_tsrc = bm_base + bm_pb
                                        bm_tie = bm_pb1 - bm_pb
                                        if bm_win < 0:
                                            bm_win = cutlass.Int32(0)
                                        if bm_tie < 0:
                                            bm_tie = cutlass.Int32(0)
                                    bm_winc = warp_inclusive_sum(bm_win, tidx)
                                    bm_tinc = warp_inclusive_sum(bm_tie, tidx)
                                    # per-CTA descriptors -> smem (s_hm is free here)
                                    s_hm[tidx] = bm_winc - bm_win  # winner out offset
                                    s_hm[32 + tidx] = bm_win
                                    s_hm[64 + tidx] = bm_wsrc
                                    s_hm[96 + tidx] = (
                                        bm_tinc - bm_tie
                                    )  # tie stage offset
                                    s_hm[128 + tidx] = bm_tie
                                    s_hm[160 + tidx] = bm_tsrc
                                    if tidx == 31:
                                        s_hm[192] = bm_winc  # total winners
                                        s_hm[193] = bm_tinc  # total ties
                                cute.arch.barrier()
                                bm_nw = s_hm[192]
                                bm_nt = s_hm[193]
                                # winners: one index per thread per step
                                for g in range(tidx, bm_nw, self.nt):
                                    bm_r = cutlass.Int32(0)
                                    for rr in cutlass.range_constexpr(S):
                                        if g >= s_hm[rr]:
                                            bm_r = cutlass.Int32(rr)
                                    if g < top_k:
                                        self._wf_out(
                                            out_idx_row,
                                            pt_row,
                                            g,
                                            slab_i[s_hm[64 + bm_r] + (g - s_hm[bm_r])],
                                        )
                                # ties: key + index into the tie stage
                                for g in range(tidx, bm_nt, self.nt):
                                    bm_r = cutlass.Int32(0)
                                    for rr in cutlass.range_constexpr(S):
                                        if g >= s_hm[96 + rr]:
                                            bm_r = cutlass.Int32(rr)
                                    if g < self.tie_cap:
                                        bm_src = s_hm[160 + bm_r] + (
                                            g - s_hm[96 + bm_r]
                                        )
                                        s_tk[g] = self.exact_key(
                                            cutlass.Uint32(slab_k[bm_src])
                                        )
                                        s_ti[g] = slab_i[bm_src]
                                if tidx == 0:
                                    s_misc[10] = bm_nw
                                    s_count[4] = bm_nt
                            cute.arch.barrier()
                            if tel:
                                ts3b = read_clock64()  # emit pass done
                            nb = s_count[4]  # bin-B members (== hist[binb] if fit)
                            remaining = top_k - above
                            if tel:
                                ts_m = read_clock64()  # tie select start
                            if remaining < 0 or remaining > nb or nb > self.tie_cap:
                                ok = cutlass.Int32(0)  # bin-B overflow: fallback
                            else:
                                if remaining > 0:
                                    if nb <= 128:
                                        self._wf_tie_select_small(
                                            s_tk,
                                            s_ti,
                                            nb,
                                            above,
                                            remaining,
                                            s_cv,  # >= 512 ints scratch
                                            s_warp_sums,
                                            s_misc,
                                            out_idx_row,
                                            pt_row,
                                            tidx,
                                        )
                                    else:
                                        # bin-B members' VALUES lie in one narrow
                                        # bin, so their KEYS share high bytes
                                        # (exact_key is monotone): pass the bin's
                                        # key bounds so the leading radix rounds
                                        # skip (~0.8us of barriers each)
                                        # one-bin safety margin each side: the
                                        # members were binned by (val-tf)*sc
                                        # truncation, whose rounding can land a
                                        # member marginally outside the
                                        # reconstructed edge floats
                                        binw = cutlass.Float32(1.0)
                                        if sc > cutlass.Float32(0.0):
                                            binw = cutlass.Float32(1.0) / sc
                                        blo_f = tf_f + cutlass.Float32(binb - 1) * binw
                                        bhi_f = blo_f + binw * cutlass.Float32(3.0)
                                        kmin_b = self._wf_key_of_f32(blo_f)
                                        kmax_b = self._wf_key_of_f32(bhi_f)
                                        if binb == 255:
                                            # NaN top; 16-bit keys stop at 0xFFFF
                                            # (a 32-bit sentinel would make the
                                            # two leading byte rounds run)
                                            kmax_b = cutlass.Uint32(
                                                0xFFFFFFFF if self.is_f32 else 0xFFFF
                                            )
                                        if sc <= cutlass.Float32(0.0):
                                            # sc == 0 (the sampled threshold
                                            # rounded up to the sample max):
                                            # every survivor binned to 0, so
                                            # the bin edges say nothing about
                                            # the members' keys -- full range,
                                            # no round skipped (exact, slower)
                                            kmin_b = cutlass.Uint32(0)
                                            kmax_b = cutlass.Uint32(
                                                0xFFFFFFFF if self.is_f32 else 0xFFFF
                                            )
                                        self._tie_select_smem_skip_wf(
                                            s_tk,
                                            s_ti,
                                            nb,
                                            above,
                                            remaining,
                                            kmax_b,
                                            kmin_b,
                                            s_h256,
                                            s_warp_sums,
                                            s_misc,
                                            out_idx_row,
                                            pt_row,
                                            tidx,
                                        )
                                # paged: winners and ties sit in the stage
                                self._wf_paged_finish(
                                    out_idx_row, pt_row, cutlass.Int32(0), top_k, tidx
                                )
                            if tel:  # block 21: tie select alone
                                if tidx == 0:
                                    st_ptr[num_rows_dbg * 21 + row] = (
                                        read_clock64() - ts_m
                                    ).to(cutlass.Int32)
                        if ok == 0:
                            # ---- inline exact fallback (fused; no 2nd launch) ----
                            # CTA-uniform (ok derives from uniform smem/gmem reads),
                            # so the block barriers inside are safe.  smem aliasing:
                            # s_cv (LCAP+4 >= 4096 ints, dead in the failure path)
                            # becomes the 4096-bin key histogram; s_count slots
                            # 8/16 become the gt/eq ticket counters.  The harvest
                            # stays inside slab_k / slab_i (below GCAP), so the
                            # row-histogram tail is untouched.
                            self._fallback_row(  # type: ignore[attr-defined]  # Prod MRO
                                row_in,
                                length,
                                out_idx_row,
                                slab_k,
                                slab_i,
                                s_cv,
                                s_h256,
                                s_warp_sums,
                                s_misc,
                                s_count + 8,
                                s_count + 16,
                                tidx,
                            )
                            self._wf_page_fixup(
                                out_idx_row, pt_row, cutlass.Int32(0), top_k, tidx
                            )
                        if tel:
                            ts4 = read_clock64()
                        if tidx == 0:
                            st_ptr[row] = cutlass.Int32(1) - ok
                            if tel:  # blocks 1-3: telemetry builds only
                                st_ptr[num_rows_dbg + row] = cutlass.Int32(
                                    5 if S == 1 else 3  # family tag (5 = S1 direct)
                                )
                                st_ptr[num_rows_dbg * 2 + row] = cand
                                st_ptr[num_rows_dbg * 3 + row] = slots
                            if tel:
                                # phase telemetry (last-arriver CTA's own view, ns):
                                # pre / walk / select+resets / hist+scan / emit pass
                                st_ptr[num_rows_dbg * 5 + row] = (ts1 - ts0).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 6 + row] = (ts2 - ts1).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 7 + row] = (ts4 - ts3b).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 8 + row] = (ts3a - ts3).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 9 + row] = (ts3b - ts3a).to(
                                    cutlass.Int32
                                )
                                # epilogue wait: walk end -> last-arriver verdict
                                st_ptr[num_rows_dbg * 14 + row] = (ts3 - ts2).to(
                                    cutlass.Int32
                                )
                                st_ptr[num_rows_dbg * 15 + row] = s_count[4]  # ties
                                st_ptr[num_rows_dbg * 18 + row] = (
                                    read_clock64() - ts4
                                ).to(cutlass.Int32)  # select end -> status writes done
                                # per-thread walk time: max / min across the CTA
                                st_ptr[num_rows_dbg * 4 + row] = s_misc[3]
                                st_ptr[num_rows_dbg * 20 + row] = cutlass.Int32(
                                    cutlass.Uint32(0x7FFFFFFF)
                                    - cutlass.Uint32(s_misc[5])
                                )
                            # self-reset: counters + the row histogram (S == 1
                            # never touched mc_row or slab_h -- nothing to reset)
                            if cutlass.const_expr(S > 1):
                                mc_row[0] = cutlass.Int32(0)
                                mc_row[1] = cutlass.Int32(0)
                                mc_row[2] = cutlass.Int32(0)
                        if cutlass.const_expr(S > 1):
                            for t in range(tidx, 256, self.nt):
                                slab_h[t] = cutlass.Int32(0)

        # PDL: release the dependent grid only at the very end.  Releasing
        # after the walk let the next launch's CTAs co-schedule during our
        # epilogue and measured +2-3% at 256K b=64; at the end PDL still hides
        # the next launch's latency (measured -0.4..-0.6us on most cells).
        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_launch_dependents()

    @cute.jit
    def _tie_select_smem_skip_wf(
        self,
        s_tk,
        s_ti,
        eq,
        gt,
        remaining,
        k_hi,
        k_lo,
        s_h256,
        s_warp_sums,
        s_misc,
        out_idx_row,
        pt_row,
        tidx,
    ):
        """Byte-radix rank-select over the smem tie stage.  Every staged key
        lies in [k_lo, k_hi] (callers pass the ties' exact range, bounds with
        a proven margin, or full-range sentinels), so a leading byte round
        on which k_lo and k_hi agree has a single bucket: its histogram and
        scan are skipped and the byte is taken from k_lo.  Identical keys
        (a plateau of exact zeros in the crossing bin, the relu decode case)
        skip all four rounds; the refine's single-key-bin ties skip three.
        Skipping is monotone from the top byte down, so it never runs after
        a real round.  Kept as a separate name from the MC module's select
        so its compiled kernels are not perturbed."""
        prefix = cutlass.Uint32(0)
        pmask = cutlass.Uint32(0)
        need = cutlass.Int32(remaining)
        total = cutlass.Int32(eq)
        diff = cutlass.Uint32(k_lo) ^ cutlass.Uint32(k_hi)
        for r_ in cutlass.range_constexpr(4):
            sh = cutlass.const_expr(24 - 8 * r_)
            if (diff >> cutlass.Uint32(sh)) == cutlass.Uint32(0):
                # one bucket: min and max share every bit from ``sh`` up
                prefix = prefix | (
                    cutlass.Uint32(k_lo) & (cutlass.Uint32(0xFF) << cutlass.Uint32(sh))
                )
                pmask = pmask | (cutlass.Uint32(0xFF) << cutlass.Uint32(sh))
            else:
                if tidx < 256:
                    s_h256[tidx] = cutlass.Int32(0)
                cute.arch.barrier()
                # unroll=1: cold path inlined at several sites; the compiler
                # unrolled these tie loops per copy (see _fallback_row)
                for i in cutlass.range(tidx, eq, self.nt, unroll=1):
                    kk = cutlass.Uint32(s_tk[i])
                    if (kk & pmask) == prefix:
                        smem_atomic_add(
                            s_h256
                            + cutlass.Int32(
                                (kk >> cutlass.Uint32(sh)) & cutlass.Uint32(0xFF)
                            ),
                            1,
                        )
                cute.arch.barrier()
                self.scan256_and_find(s_h256, total, need, s_warp_sums, s_misc, tidx)
                bucket = s_misc[0]
                above = s_misc[1]
                cnt = s_misc[2]
                cute.arch.barrier()
                prefix = prefix | (cutlass.Uint32(bucket) << cutlass.Uint32(sh))
                pmask = pmask | (cutlass.Uint32(0xFF) << cutlass.Uint32(sh))
                need = need - above
                total = cnt
        if tidx == 0:
            s_misc[10] = cutlass.Int32(0)
            s_misc[11] = cutlass.Int32(0)
        cute.arch.barrier()
        nab = remaining - need
        for i in cutlass.range(tidx, eq, self.nt, unroll=1):
            kk = cutlass.Uint32(s_tk[i])
            if kk > prefix:
                p = smem_atomic_add(s_misc + 10, 1)
                if p < nab:  # bounds the store even if a round disagreed
                    self._wf_out(out_idx_row, pt_row, gt + p, s_ti[i])
            else:
                if kk == prefix:
                    e = smem_atomic_add(s_misc + 11, 1)
                    if e < need:
                        self._wf_out(out_idx_row, pt_row, gt + nab + e, s_ti[i])


class ProdWalkFirstTopK(WalkFirstTopK, GatedExactFallback):
    """Walk-first kernel + gated exact fallback, one compiled launcher."""

    @cute.jit
    def launch_prod(
        self,
        input_data: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        slab: cute.Tensor,
        status: cute.Tensor,
        mc_state: cute.Tensor,
        page_table: cute.Tensor,
        stream,
    ):
        num_rows = input_data.shape[0]
        # min_blocks_per_mp=1: __launch_bounds__(1024, 1).  SASS inspection
        # showed that WITHOUT a bound, ptxas' occupancy heuristic squeezed
        # this kernel to REG:32 with a 24-byte SPILL stack (targeting
        # 2 blocks/SM) -- LDL/STL in the walk loop.  Declaring 1 block/SM
        # intended frees the full 64-register budget for the quad-buffered
        # walker.  (min_blocks_per_mp=2 is the same squeeze, explicitly;
        # measured slower.)
        if cutlass.const_expr(self.wf_cluster and self.mc_splits > 1):
            # hw cluster along Y: the S slices of a row share DSMEM
            self.wf_topk_kernel(
                input_data,
                seqlen,
                output_indices,
                slab,
                status,
                mc_state,
                page_table,
            ).launch(
                grid=(num_rows, cutlass.const_expr(self.mc_splits), 1),
                block=(self.nt, 1, 1),
                cluster=(1, cutlass.const_expr(self.mc_splits), 1),
                stream=stream,
                use_pdl=self.enable_pdl,
                min_blocks_per_mp=(1 if self.nt == 1024 else 2),
            )
        else:
            self.wf_topk_kernel(
                input_data,
                seqlen,
                output_indices,
                slab,
                status,
                mc_state,
                page_table,
            ).launch(
                grid=(num_rows, cutlass.const_expr(self.mc_splits), 1),
                block=(self.nt, 1, 1),
                stream=stream,
                use_pdl=self.enable_pdl,
                min_blocks_per_mp=(1 if self.nt == 1024 else 2),
            )
        # No fallback launch: the exact fallback is FUSED into the wf
        # kernel's last-arriver epilogue (a failed row re-solves inline).
        # Per-kernel profiling showed the always-launched gate kernel
        # cost a flat ~1.65us -- the entire b=16 deficit vs gvr_2.


_compiled: dict = {}


def get_walkfirst_kernel(
    top_k: int,
    N: int,
    splits: int,
    telemetry: bool = False,
    dtype=None,
    nt: int | None = None,
    page_size: int = 0,
    device: torch.device | None = None,
):
    """Compile (with on-disk caching) the walk-first kernel + gated
    fallback for a (top_k, N, splits) specialization on ``device`` (the
    current device when ``None``): the architecture decides clusters, PDL,
    the bf16 pair classify, the Ampere tie fold and the stage size, and is
    part of the in-process key; the caller sets the current device so the
    DSL compiles for it too.  mc_state (rows, 8) int32 AND the slab must be
    zero-initialised at first use (the row histogram lives in the slab tail
    and self-resets afterwards).

    page_size > 0 (a power of two) compiles the paged-output variant: the
    launcher's last tensor argument is the ``(rows, max_pages)`` int32 page
    table and every selected column is stored as its physical KV slot (see
    WalkFirstTopK.wf_page_size).  Unpaged launchers still take that argument
    (pass any int32 tensor; it is never read).

    telemetry=True (or FLASHINFER_TOPK_PRIM_TELEMETRY=1, the one debug build
    flag of the primitives backends) compiles the phase-instrumented variant
    (globaltimer reads + status blocks 5-9); production kernels carry no
    instrumentation.  Every other execution-strategy choice below is an
    internal constant with the measurement that picked it in the comment."""
    telemetry = telemetry or os.environ.get("FLASHINFER_TOPK_PRIM_TELEMETRY") == "1"
    # nt=512 was built and MEASURED as a k<=1024 family: no win anywhere
    # (walk phase identical -- the 2-CTAs/SM shape buys no bandwidth --
    # while publish/arrival overhead doubles), and tie_cap = 2*nt = 1024
    # overflows on randn's threshold-adjacent value bins (4/64 rows fell
    # back at 1M).  gvr_2's small-k edge is its leaner fused pipeline,
    # not block shape.  The parameterization stays as an experimentation
    # hook; production uses 1024 threads at every k.
    #
    # Wide batches (b > SMs) are the case where block shape WOULD matter: a
    # 1024-thread CTA at 64 regs fills the register file, so one CTA per SM
    # and a 256-row batch on 148 SMs runs as two waves, while gvr_2 routes
    # those batches to 256/512-thread CTAs and wins 1.3-1.6x at 32K-128K.
    # 512-thread shape (2 CTAs/SM): selected by the dispatcher for batches
    # wider than the SM count (one-CTA-per-row regime), where the 1024-thread
    # CTA's full register file forces two waves.  Requirements met for it:
    # lane-indexed partial reads guarded to nw warps, tie stage
    # sized by k (tie_cap = max(2*nt, k)), radix tie select with tie_cap/nt
    # items per thread, stage kept at 8192 entries.
    if nt is None:
        nt = 1024
    assert nt in (512, 1024)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    cc = torch.cuda.get_device_capability(device)
    cdt = {
        None: cutlass.Float32,
        torch.float32: cutlass.Float32,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }[dtype]
    vec_elems = 4 if cdt == cutlass.Float32 else 8
    dt_tag = {cutlass.Float32: "f32", cutlass.Float16: "f16", cutlass.BFloat16: "bf16"}[
        cdt
    ]
    assert top_k <= max(2 * nt, 2048) and N % vec_elems == 0
    assert splits in (1, 2, 3, 4, 6, 8, 16, 32)
    assert page_size >= 0 and (page_size & (page_size - 1)) == 0, page_size
    # cs=8 was built and MEASURED WORSE (B200 1M b=16: 18.1 -> 28.3us):
    # an 8-CTA cluster must pack into one GPC, and at 1 CTA/SM occupancy
    # the scheduler strands SMs waiting to co-place whole clusters.  cs=2
    # is a clear win (64K b=64 12.98 -> 10.73) and cs=4 neutral-positive;
    # S=8/16 keep the gmem slab + release/acquire arrival path.
    # DSMEM cluster shapes: 2 and 4 (measured on B200), 3 and 6 (one-wave
    # shapes the dispatcher picks for long rows on 208-SM parts; measured on
    # Rubin: 1M b=64 S3 43.2 -> 38.2us, 1M b=32 S6 24.4 -> 22.3us).  8 was
    # built and measured worse everywhere on both parts.
    _cl_sizes: tuple[int, ...] = (2, 3, 4, 6)
    use_cluster = splits in _cl_sizes and cc[0] >= 9
    # short-row arm domain: capped so the register arm holds at most 4 vectors
    # per thread (16K fp32 at 1024 threads; the 512-thread wide-batch shape
    # keeps 8K and sends 8K-16K rows to the walk pipeline).  Crossover sweep
    # (B200, 16 rows, K 512/2048, widths 16K/64K): the census arm wins up to
    # 4096 columns, the register arm (4 vectors per thread) from 5000 up.
    small_n = min(WF_SMALL_N, 4 * nt * vec_elems)
    census_n = min(WF_CENSUS_N, small_n)
    # register-resident arm for census_n < rows <= small_n; the census arm
    # alone measured 7.05 us against 5.41 on full 16K rows
    reg_arm = True
    reg_vpt = max(1, -(-small_n // (nt * vec_elems)))
    # uniform-tie shortcut: ReLU-style rows tie thousands of exact zeros at
    # the boundary; copying the staged prefix measured 6.07 -> 3.80 us
    uniform_tie = True
    # packed-pair 16-bit classify: f16x2 setp exists on every supported
    # arch, bf16x2 setp needs sm_90 (older arches keep the scalar path)
    pair16 = cdt == cutlass.Float16 or (cdt == cutlass.BFloat16 and cc[0] >= 9)
    key = (
        top_k,
        N,
        splits,
        telemetry,
        nt,
        use_cluster,
        dt_tag,
        pair16,
        page_size,
        cc,
    )
    if key in _compiled:
        return _compiled[key]
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    # programmatic dependent launch (griddepcontrol is SM90+): worth 0.39 us
    # per decode launch under graph replay (3.77 -> 3.38 us on the 16K trace)
    use_pdl = cc[0] >= 9
    kern = ProdWalkFirstTopK(
        dtype=cdt,
        top_k=top_k,
        next_n=1,
        compress_ratio=1,
        return_values=False,
        ctas_per_group=1,
        chunk_elems=0,
        num_sms=148,
        min_blocks_per_mp=0,
        boundary_cls=True,
        approx_ties=True,
        enable_pdl=use_pdl,
        warp_agg=False,
        nt=nt,
    )
    kern.mc_splits = splits
    # the tie stage must hold a full rank-k bin regardless of block shape
    # (the base class sizes it 2*nt; the 512-thread shape needs >= k)
    kern.tie_cap = max(2 * nt, top_k)
    kern.wf_telemetry = telemetry
    # Ampere/Ada serialize same-address smem reds: fold the tie key range in
    # registers there (measured on A100/L40S; SM90+ keep the direct reds)
    kern.wf_tie_fold = cc[0] == 8
    kern.wf_cluster = use_cluster
    # (No hint rung: with realistic previous-step hints the value gathers cost
    # 0.6-1.4 us and the tightened threshold saved less; only exact hints won,
    # and a caller cannot know its hint is exact.  The kernel is hint-free and
    # the API's pre_idx is ignored on this backend.)
    kern.wf_small_n = small_n
    kern.wf_census_n = census_n
    kern.wf_reg_arm = reg_arm
    kern.wf_reg_vpt = reg_vpt
    kern.wf_uniform_tie = uniform_tie
    # the arms zero the coarse histogram with 16-byte stores, 4*nt bins per pass
    assert kern.hist_size % (4 * nt) == 0, (kern.hist_size, nt)
    kern.wf_page_size = page_size
    # stage the leading page-table entries in smem (up to four per thread, 8 KB
    # cap); columns past them (beyond 128K at page size 64) gather from global
    kern.wf_pt_pages = -(-N // page_size) if page_size else 0
    kern.wf_pt_cache = min(kern.wf_pt_pages, 4 * nt, WF_PT_CACHE_MAX)
    # prefetched before the length load: the pages of the longest short-arm row
    kern.wf_pt_pre = min(kern.wf_pt_cache, -(-small_n // page_size)) if page_size else 0
    # candidate stage: 16384 entries when the device's opt-in shared memory
    # per block allows (~147KB kernel: B200/H100 228KB, A100 164KB), else 8192
    # (L40S, RTX 5080 ~100KB).
    from ...utils import get_shared_bytes_per_block_optin

    # Only the kernels that can overflow get the big stage: single-worker
    # rows (S == 1) at N >= 1M.  Split-row and short-row kernels keep 8192:
    # they never overflow, and the larger shared-memory carveout costs
    # ~0.4us per row (less L1 for the survivor re-reads) -- 3-4% at 64K.
    lcap_sel = LCAP
    if (
        splits == 1
        and N >= (1 << 20)
        and get_shared_bytes_per_block_optin(device) >= LCAP_BIG_SMEM
    ):
        lcap_sel = LCAP_BIG
    # 512-thread shape: the stage keeps 8192 entries (the aim at k=2048 is
    # ~3k candidates, so 4096 overflowed every row) but not 16384 -- two
    # CTAs per SM must fit: ~64KB stage + 16KB histogram + 16KB ties + misc
    # is ~100KB each on a 228KB carveout
    kern.lcap = lcap_sel if nt == 1024 else min(lcap_sel, LCAP)
    kern.psamp = vec_elems * nt  # one 16B probe per thread
    kern.wf_pair16 = pair16
    sym_rows = cute.sym_int()
    # page table: rows and pages both symbolic (the table may cover more pages
    # than the width needs, and unpaged launches pass a (1, 4) dummy)
    sym_pt_rows = cute.sym_int()
    sym_pages = cute.sym_int()
    f32, i32 = cdt, cutlass.Int32  # f32 alias = logits dtype

    def _fk(dt, shape, align=None):
        so = tuple(range(len(shape) - 1, -1, -1))
        if align is None:
            return cute.runtime.make_fake_compact_tensor(dt, shape, stride_order=so)
        return cute.runtime.make_fake_compact_tensor(
            dt, shape, stride_order=so, assumed_align=align
        )

    def _compile_fn():
        return cute.compile(
            kern.launch_prod,
            _fk(f32, (sym_rows, N), 16),
            _fk(i32, (sym_rows,)),
            _fk(i32, (sym_rows, top_k), 16),
            _fk(i32, (sym_rows, WF_ROW_INTS), 16),
            _fk(i32, (sym_rows,)),
            _fk(i32, (sym_rows, 8), 16),
            _fk(i32, (sym_pt_rows, sym_pages), 16),
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compiled = build_and_load_cute_dsl_kernel(
        "walkfirst_topk_primitives",
        f"wf_v1_{dt_tag}_k{top_k}_N{N}_S{splits}"
        f"{'_nt512' if nt == 512 else ''}{'_tel' if telemetry else ''}"
        f"{'_clu' if use_cluster else ''}"
        f"{'_np' if (cdt != cutlass.Float32 and not pair16) else ''}"
        f"{'_L16k' if kern.lcap == LCAP_BIG else ''}"
        f"{'_pdl' if use_pdl else ''}"
        f"{'_tf' if kern.wf_tie_fold else ''}"
        f"{f'_ps{page_size}' if page_size else ''}",
        _compile_fn,
        extra_key_files=(
            __file__,
            _radix_mod.__file__,
            _fallback_mod.__file__,  # GCAP + the inlined exact fallback
            _gvr2_mod.__file__,  # DSMEM/cluster op set
        ),
    )
    _compiled[key] = compiled
    return compiled
