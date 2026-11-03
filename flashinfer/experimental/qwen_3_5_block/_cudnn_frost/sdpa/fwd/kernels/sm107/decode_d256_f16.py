# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Decode-shaped SDPA forward for the SM107 (Rubin) d256 f16/bf16 flavor: the swap-AB tile.

Why a second template.  The prefill tile (sm100/prefill_d256_f16.py) issues
BMM1 / BMM2 over a 256-row collective Q tile per 128-key KV tile whatever the
number of LIVE Q rows, and its softmax warpgroup exponentiates the full
128 x 128 score tile.  A decode step of a 32/2-head d256 model has 16 live
rows per (batch, KV head), so 15/16 of that work is spent on TMA-OOB zero rows
and the kernel runs at half the HBM rate (67 us for b=32, s_kv=4096 on a B200;
the KV bytes alone take 34 us at 8 TB/s).  Shrinking the Q tile does not help
on this hardware: a 64-row UMMA runs at half rate per SM, so the MMA time per
key is unchanged, and the 64-row TMEM layouts scatter the rows across the
sub-partitions.

This tile transposes the problem instead (the "swap-AB" decode form):

    S^T[128 keys, N_Q] = K[128 keys, d_qk] . Q^T[d_qk, N_Q]      (BMM1, M = keys)
    O^T[d_v, N_Q]     += V^T[d_v, 128 keys] . P^T[128 keys, N_Q]  (BMM2, M = d_v)

The KV tokens are the MMA M axis (M = 128, the standard TMEM lane == row
layout) and the packed Q rows are the N axis (N_Q = 16, 32 or 64, a multiple of
the f16 UMMA N granule), so the tensor-core and exp work scale with the live
rows: 8x less MMA and 8x fewer exponentials per key than the prefill tile at
N_Q = 16.  The online softmax reduces over LANES (a column of S^T is one Q
row's scores for the tile's 128 keys): a 5-step butterfly inside each warp
and, once per KV tile, a shared-memory max across the 4-warp group (red.shared.max
on order-preserving int32 keys into one 16-word slot per group).  A group owns
16 columns; N_Q = 32 runs two groups (8 softmax warps) and N_Q = 64 four (16)
over the same 128 TMEM lanes, so the per-thread exp / shuffle / register load
is the same at every tile width (one 4-warp group over 32 columns was
issue-bound and lost to the prefill tile on the S_q = 2 MTP shape).  On B200
the 32-column tile streams a KV tile 1.6x slower per CTA than the 16-column
one (2.8 vs 1.75 us; 90 us unsplit at b=32 x 2 KV heads x 4096 keys where the
prefill tile takes 66 us), so the Blackwell adapter compiles and tests it at
the template level but does not route it (config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS);
the Rubin adapter ROUTES it (config_sm107.decode_d256_q_tile, measured on cc 10.7
against the prefill body at the 24/2 geometry; the other geometries inside the
route ride the same rule unmeasured): rows in (16, 32] in one unit, and a packed
MTP step of up to four tokens (24/2: 48 rows) as TWO token units of two tokens x
12 heads on the 32-column tile, each streaming the KV range once.
P^T is stored to a small swizzled SMEM tile (the B operand of BMM2 is
MN-major: the N_Q values of one key are contiguous), V is consumed in place
as an MN-major A operand (d_v contiguous), and O^T accumulates in TMEM with
lane = d_v (two 128-lane blocks for d_v = 256).

Roles (6 warps at N_Q = 16, 10 at N_Q = 32, 18 at N_Q = 64; cta_group::1, one CTA per
(KV-head group, batch, split, TOKEN UNIT) unit -- a token unit is one Q box of
Q_BOX_TOKENS = N_Q // HEADS_PER_TILE consecutive tokens, and a graph with more
tokens than one box holds launches ceil(S_q / Q_BOX_TOKENS) units per head
group, each streaming the unit's KV range once (Q_TOKEN_UNITS; the host entry
derives the count from the runtime S_q); W = 4 * N_Q / 16 softmax warps):

  * warps 0..W-1  softmax + epilogue, 4 warps per 16-column group: thread ==
                  key lane (tidx % 128) for S^T / P^T, == d lane for the O^T
                  rescale and the final store;
  * warp W        MMA issue + TMEM alloc/dealloc;
  * warp W+1      TMA loads: Q^T once, then K/V through a 3-slot 64 KiB ring
                  (K(t), V(t), K(t+1), ...), block-table indirection under PAGED_KV.

SMEM: Q^T N_Q x 512 B (8 / 16 / 32 KiB) + 3 x 64 KiB ring + 2 x P^T (4 / 8 /
16 KiB) = 208 / 224 / 256 KiB at N_Q = 16 / 32 / 64 (plus the max-key slots, the
sum scratch and the barriers).  TMEM: 2 S^T slots + 2 O^T blocks = 4 * N_Q
columns (64 / 128 / 256).  The KV loop, per-batch lengths, masks (padding / bottom-right causal
/ SWA / right band), sink, Stats (natural or base-2), dense padded-Q trim and
split-KV partials follow the prefill tile's contract exactly: a split writes
fp32 O / natural-log LSE partials into the split-major workspace that
sm100/split_combine.py reduces, and an empty range yields O := 0 / LSE := -inf
(with a sink, LSE := the sink logit: the row's one live column).
Nothing reaches the host (python/cudnn/AGENTS.md Rule 3): lengths, page
tables and the split chunking are all resolved on device. The host uses the
explicit pointer ABI: extents and Int64 strides are runtime arguments, while
head dimensions, Stats presence and the paged in-page layout specialize compile().

Rubin (SM107) sibling of sm100/decode_d256_f16.py: the SAME body (cta_group::1,
6 / 10 / 18 warps, f16 TILE_K_HW = 16 on both arch lines, 4 x N_Q TMEM columns)
under config_sm107.make_cfg_d256_decode, which builds the shared CfgD256Decode
record over the Rubin width set (config_sm107._D256_DECODE_Q_TILES_SM107: the SM100
pair plus the 64-column tile the Blackwell carveout cannot hold) and re-validates it
against the Rubin facts -- the 320 KiB usable carveout (209 / 226 KiB at N_Q = 16 /
32 fit the STANDARD 227 KiB carveout: no oversized-SMEM launch, no L1 shrink; the
64-column tile's 259 KiB take the oversized carveout, L1 -> 8 kB, which costs this
TMA-fed tile nothing it uses) and the 256 KiB version-0 tcgen05 descriptor window
(every MMA-operand slab starts AND ends inside it -- P^T at N_Q = 64 ends exactly at
the line; DESC_VERSION is DERIVED from the layout, never assumed).

The 64-column tile (four softmax groups, 576 threads) is ONE unit of up to 64
packed rows: an MTP step of four tokens x 12 / 16 heads (48 / 64 rows) streams the
KV range once where the 32-column tile cuts it into two token units.  Its register
budget is what shapes this body: ptxas caps an 18-warp CTA at 96 registers per
thread (16,384 per SM sub-partition over 5 warps x 32 lanes, 8-granular -- the
`.reqntid` the DSL emits, no `.maxnreg`), and the per-thread work is the same 16
columns at every width, so the keyed max exchange, the ftz max intrinsics, the x8
O^T readbacks and the two-half epilogue below exist to keep the sm_107a rendering
spill-free at that cap (MEASURED: REG 93-96, STACK 0, STL + LDL 0 on the admitted
forms; the SM100 twin runs the same source at its roomier caps).  The admitted
forms are a config fact (config_sm107._validate_cfg_d256_decode_sm107: a packed
group of at least two heads, and over a paged cache the causal / window mask); which
steps the route sends to the width is config_sm107.decode_d256_q_tile's decision.  SPIN_RING_WAITS is this kernel's own measured constant (the
sleeping form until its A/B on an exclusive Rubin GPU).  Kept line-for-line with
the SM100 body apart from those declarations (pinned by the twin-diff test in
test_sdpa_fwd_decode_d256_sm107.py): a fix to the tile lands in BOTH files.

Not served here (the adapter keeps these on the prefill tile): THD queries,
graphs past the Rubin route (config_sm107.decode_d256_q_tile: more than two
token units of the 32-column tile for a packed step, an unpacked / MHA step above
32 tokens, a head group wider than the tile), FP8/MXFP8, the fused epilogue gate
and the pre-folded softmax scale (the Rubin d256 prefill kernel carries both);
the 64-column width declines the unpacked and the mask-free paged forms (above).
"""

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from functools import lru_cache
from typing import Callable, NamedTuple, Optional, Tuple

from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver  # noqa: F401

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_arch_requirement_error
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import TemplateParams, d256_decode_desc_version, make_cfg_d256_decode

# The per-graph params are injected as a module global by the loader
# (api._load_kernel_module) before this body executes; a plain import gets a
# padded fp16 decode config for N_Q = 16.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams(decode_q_tile=16, seq_kv_lens_present=True))
CFG, _TMA = make_cfg_d256_decode(PARAMS)

# CuTe DSL version gate (python/cudnn/AGENTS.md Rule 7): sm_107a needs the public 4.8.0 wheel; below it
# every compile on a Rubin box dies inside the DSL (`KeyError: 'sm_107a'`).  The adapter declines first
# (SdpaFwdDslSm100.check_support); this is the typed backstop for a direct template load, placed BEFORE
# the kernel body is defined so the message names the installed version, never a DSL internal.
_DSL_ARCH_ERROR = cutedsl_arch_requirement_error((10, 7))
if _DSL_ARCH_ERROR is not None:
    raise NotImplementedError(f"decode_d256_f16 (sm107): {_DSL_ARCH_ERROR}")

# tcgen05 SMEM-descriptor version for EVERY SmemTile in this module -- ONE decision point, DERIVED from
# the layout by config_sm107.d256_decode_desc_version (0: the highest MMA-operand slab, P^T, starts at
# 200 / 208 KiB, under the 256 KiB version-0 window) and wired into every construction below, never a
# per-tile literal (the sm107 discipline: a re-literalled descriptor version is how the d512 MXFP8 sibling
# shipped NaN on 100 % of cells; test_sm107_decode_every_smem_tile_takes_the_module_desc_version).
DESC_VERSION: int = d256_decode_desc_version(CFG)
# Retry form of the per-KV-iteration RING waits (kv_full / kv_empty, s_full / s_empty, p_full and the
# in-loop bmm2_done): every such site passes the constant as its spin keyword; the one-shot waits
# (q_full, the epilogue's last bmm2_done, tmem_dealloc) keep the default sleeping form.  A MEASURED
# per-kernel fact (frost-tile-dsl.md section 8b: +5.5 % on one sm107 kernel, -5.8 % on another): False =
# the sleeping form the SM100 body runs, until this tile's own A/B on an exclusive Rubin GPU; flipping the
# constant IS the whole experiment (test_sm107_decode_ring_waits_take_the_module_spin_constant).
SPIN_RING_WAITS: bool = False

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance, launch_dependent_grids, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.mma import mma_ss
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import tma_load_tile
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.mask import MASK_CAUSAL, MASK_PADDED, MASK_SWA, compute_kv_loop_bounds
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import fp32_to_fp16
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels._common_blackwell import row_max_for_exp2, sdpa_operand_tensors

if CFG.DTYPE_QKV == 2:
    STORAGE_DTYPE = cutlass.BFloat16
else:
    STORAGE_DTYPE = cutlass.Float16
MMA_KIND = nvvm.Tcgen05MMAKind.F16
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_1

N_Q = CFG.N_Q
TILE_N = CFG.TILE_N
TILE_K = CFG.TILE_K
TILE_O = CFG.TILE_O
BPE = CFG.BPE
STAGES = CFG.STAGES_KV
SPLIT_KV = CFG.SPLIT_KV
_FP32_PARTIALS = SPLIT_KV > 1

# === PackGQA === row r of the Q tile is token r // G, head r % G (the prefill
# tile's convention).  The Q TMA box covers Q_BOX_TOKENS tokens x G heads; a
# group that does not divide N_Q leaves N_Q - Q_BOX_ROWS tail rows, zero-filled
# once at kernel start (their S^T / O^T columns are computed and never stored).
HEADS_PER_TILE = CFG.QH_PER_KH if CFG.PACK_GQA else 1
Q_BOX_TOKENS = N_Q // HEADS_PER_TILE
Q_BOX_ROWS = Q_BOX_TOKENS * HEADS_PER_TILE
if Q_BOX_TOKENS < 1:
    raise ValueError(f"decode_d256_f16: the packed head group ({HEADS_PER_TILE}) does not fit the {N_Q}-row Q tile")
# === Token units === a unit covers Q_BOX_TOKENS consecutive Q tokens of its head group; a graph with more tokens
# than one Q box holds is cut into ceil(S_q / Q_BOX_TOKENS) TOKEN UNITS per (head group, batch, split), each a CTA
# that streams the unit's KV range once (MTP: S_q = 4 at 24/2 = two units of two tokens on the 32-column tile).  The
# host entry derives the unit count from the runtime S_q and widens the grid's x axis by it; the kernel decodes
# (head group, token unit) from blockIdx.x and offsets every token coordinate (the Q box, the causal / window mask,
# the O / LSE rows, the padded-Q trim) by the unit's first token.  Declared here for the binders (prepared.py,
# the direct test driver): a module without the flag covers S_q x HEADS_PER_TILE <= N_Q only.
Q_TOKEN_UNITS: bool = True

# === Paged KV (same contract as prefill_d256_f16.py) ===
PAGED_KV = bool(CFG.PAGED_KV)
PAGE_SIZE = CFG.PAGE_SIZE if PAGED_KV else 0
KV_BOX_ROWS = min(PAGE_SIZE, TILE_N) if PAGED_KV else TILE_N
KV_BOXES = TILE_N // KV_BOX_ROWS

# TMA granule: one 128 B swizzle atom of the head dim (64 half elements).
GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
QK_ITERS = _TMA.QK_ITERS
VO_ITERS = _TMA.VO_ITERS
if _TMA.VO_GRANU_ELEMS != GRANU_ELEMS:
    raise ValueError("decode_d256_f16: Q/K and V/O must share the 128 B swizzle granule")

qBufferElems = N_Q * TILE_K
kvBufferElems = TILE_N * TILE_K
# P^T is stored as packed half pairs: N_Q / 2 words per key row.
pBufferWords = TILE_N * N_Q // 2

Q_TX_BYTES = Q_BOX_ROWS * TILE_K * BPE
KV_TX_BYTES = kvBufferElems * BPE

_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
_SWZ_BITS = {128: 3, 64: 2, 32: 1}
SMEM_LAYOUT_QK = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_P = _SWZ_ENUM[CFG.P_SWZ_BYTES]
# The generic-proxy P^T stores apply the same XOR pattern the UMMA descriptor
# decodes (Swizzle<B, 4, 3>: 16 B chunk index ^= 128 B row-group index).
_P_SMEM_SWIZZLE = cutlass.Swizzle(_SWZ_BITS[CFG.P_SWZ_BYTES], 4, 3)
P_ROW_WORDS = N_Q // 2

# K-major operands (K as A of BMM1, Q^T as B of BMM1): SBO = 8 rows x 128 B.
STRIDE_BYTE_OFFSET_QK = 8 * CFG.K_SWZ_BYTES
# V^T as the MN-major A of BMM2: 128 d columns span two 64-wide swizzle atoms,
# LBO steps between them (TILE_N rows x 128 B each); SBO = 8 key rows.
LEADING_BYTE_OFFSET_VT = TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_VT = 8 * CFG.V_SWZ_BYTES
# P^T as the MN-major B of BMM2: one atom wide (N_Q * 2 B), SBO = 8 key rows.
STRIDE_BYTE_OFFSET_P = 8 * CFG.P_SWZ_BYTES

# TMEM: two S^T slots then the two O^T d-blocks, N_Q fp32 columns each.
S_ACC_OFF = (0, N_Q)
O_OFF = (2 * N_Q, 3 * N_Q)
TMEM_COLS = 32
while TMEM_COLS < 4 * N_Q:
    TMEM_COLS *= 2
O_BLOCKS = TILE_O // 128

# Softmax column groups: a 4-warp group owns COLS = 16 S^T / O^T columns (the
# f16 UMMA N granule), so N_Q = 32 runs two groups that read the same 128 TMEM
# lanes (tcgen05.ld lane access is by warp_id % 4) and the per-thread work is
# independent of the tile width.  Group g = tidx // 128 owns columns
# [16 g, 16 g + 16) and the matching P^T words / O^T columns / LSE rows.
COLS = 16
COL_GROUPS = N_Q // COLS
# This group's slice of a P^T key row: COLS half values = COLS / 2 words.
P_GROUP_WORDS = COLS // 2
P_CHUNKS_PER_GROUP = (COLS * BPE) // 16
SOFTMAX_WARPS = CFG.SOFTMAX_WARPS
SOFTMAX_LANES = SOFTMAX_WARPS * 32
if N_Q % COLS != 0 or SOFTMAX_WARPS != 4 * COL_GROUPS:
    raise ValueError(f"decode_d256_f16: N_Q={N_Q} needs {4 * (N_Q // COLS)} softmax warps (4 per 16 columns); config says {SOFTMAX_WARPS}")
MMA_WARP_ID = CFG.MMA_WARP_ID
TMALDG_WARP_ID = CFG.TMALDG_WARP_ID
# Named barriers: 1 = TMEM base hand-off (softmax warps + MMA warp), 2 = the
# softmax warps' own cross-warp reductions (all groups, they run in lockstep).
_BAR_TMEM = 1
_BAR_SOFTMAX = 2
# Cross-warp reduction scratch.  The per-KV-tile column MAXIMA meet in ONE 16-word Int32 slot per group: each of the
# group's four warps publishes its 16 butterfly maxima with red.shared.max on order-preserving keys (_max_key), and
# after the tile's named barrier every lane reads the 16 group maxima back -- no fold, nothing retained across the
# exchange.  MAX_RED_SLOTS slots rotate per tile and the group's first warp re-arms the slot two tiles ahead, so the
# one barrier per tile also orders every reuse.  The epilogue's column SUMS keep the per-warp fp32 form ([softmax
# warp][COLS], each warp's partial read back by its group).  (Holding the 16 maxima in registers and folding 64
# loaded values was what put the 18-warp N_Q = 64 tile over ptxas's 96-register cap.)
MAX_RED_SLOTS = 3
MAX_RED_WORDS = MAX_RED_SLOTS * COL_GROUPS * COLS  # <= SOFTMAX_LANES by construction (48 per group against 128 lanes), so one lane arms one word
RED_WORDS = SOFTMAX_WARPS * COLS
INT32_MIN = -2147483648  # below every max key: the slot's armed value

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


class DecodeBars(NamedTuple):
    mb_q_full: object
    mb_kv_full: object
    mb_kv_empty: object
    mb_s_full: object
    mb_s_empty: object
    mb_p_full: object
    mb_bmm2_done: object
    mb_tmem_dealloc: object


def make_decode_bars() -> DecodeBars:
    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return DecodeBars(
        mb_q_full=MBarrier(_alloc(1), stages=1, init_count=1, producer=Producer.TMA_LOAD),
        mb_kv_full=MBarrier(_alloc(STAGES), stages=STAGES, init_count=1, producer=Producer.TMA_LOAD),
        mb_kv_empty=MBarrier(_alloc(STAGES), stages=STAGES, init_count=1, producer=Producer.MMA_COMMIT),
        mb_s_full=MBarrier(_alloc(2), stages=2, init_count=1, producer=Producer.MMA_COMMIT),
        mb_s_empty=MBarrier(_alloc(2), stages=2, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_p_full=MBarrier(_alloc(2), stages=2, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_bmm2_done=MBarrier(_alloc(2), stages=2, init_count=1, producer=Producer.MMA_COMMIT),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
    )


@cute.jit
def _select_i32(cond, a, b):
    return cutlass.Int32(arith.select(cond.ir_value(), cutlass.Int32(a).ir_value(), cutlass.Int32(b).ir_value()))


@cute.jit
def _select_f32(cond, a, b):
    return cutlass.Float32(arith.select(cond.ir_value(), a.ir_value(), b.ir_value()))


@cute.jit
def _max_key(x):
    """Order-preserving Int32 key of a non-NaN fp32 (the softmax maxima: finite or -inf): the bit pattern, with the
    magnitude bits inverted for a negative value, so signed-integer order is the float order (-inf -> 0x807FFFFF, the
    smallest key a column can hold, above INT32_MIN).  Its own inverse on the keys it produces (_key_max)."""
    b = x.bitcast(cutlass.Int32)
    return _select_i32(b < cutlass.Int32(0), b ^ cutlass.Int32(0x7FFFFFFF), b)


@cute.jit
def _key_max(k):
    """The fp32 a _max_key key encodes."""
    return _select_i32(k < cutlass.Int32(0), k ^ cutlass.Int32(0x7FFFFFFF), k).bitcast(cutlass.Float32)


@cute.jit
def _unit_bounds(seq_kv_lens_tensor, seq_q_lens_addr, batch_idx, split_idx, seqlen_q, seqlen_kv, tok0):
    """This unit's KV-tile range ``[lo, hi)`` plus the per-batch lengths.

    Every role computes it identically (pure function of the same inputs), so
    the barrier handshakes stay in lockstep.  The masks narrow the range the
    way the prefill tile does (compute_kv_loop_bounds over the unit's token
    span, which starts at ``tok0`` = the token unit's first Q token), the dense
    padded-Q trim collapses a dead batch to an empty range, and the split
    chunking mirrors _common_blackwell's _split_chunk.
    """
    eff_seqlen_kv = seqlen_kv
    if cutlass.const_expr(CFG.SEQ_KV_LENS_PRESENT == 1):
        eff_seqlen_kv = cutlass.Int32(cutlass.make_array_view(seq_kv_lens_tensor)[batch_idx])
    q_len_b = seqlen_q
    if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT == 1):
        _sq_arr = cute.make_tensor(cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24))
        q_len_b = cutlass.Int32(_sq_arr[batch_idx])
    eff_seqlen_q = seqlen_q
    if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT == 1 and CFG.BOTTOM_RIGHT == 1):
        eff_seqlen_q = cute.math.max(cutlass.Int32(0), cute.math.min(q_len_b, seqlen_q))
    bounds = compute_kv_loop_bounds(
        tok0,
        eff_seqlen_q,
        eff_seqlen_kv,
        CFG.WINDOW_LEFT,
        CFG.MASK_FLAGS,
        TILE_N,
        Q_BOX_TOKENS,
        bottom_right=bool(CFG.BOTTOM_RIGHT),
        window_right=int(CFG.WINDOW_RIGHT),
    )
    left = bounds.left
    right = cute.math.max(bounds.right, left)
    if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT == 1):
        right = _select_i32(q_len_b <= cutlass.Int32(0), left, right)
    n_tiles = right - left
    per = n_tiles // cutlass.Int32(SPLIT_KV)
    rem = n_tiles % cutlass.Int32(SPLIT_KV)
    lo = left + split_idx * per + cute.math.min(split_idx, rem)
    hi = lo + per + _select_i32(split_idx < rem, 1, 0)
    return lo, hi, eff_seqlen_kv, eff_seqlen_q, q_len_b


@cute.jit
def _paged_load_tile(
    smem_tile,
    tma,
    block_table_tensor,
    batch_idx,
    kv_head_idx,
    n_pages_b,
    kv_tile,
    mbar,
    box_rows: cutlass.Constexpr[int],
    n_boxes: cutlass.Constexpr[int],
    granu_elems: cutlass.Constexpr[int],
):
    """One paged K or V tile as ``n_boxes`` row boxes through the block table
    (same contract as prefill_d256_f16._paged_load_tile: boxes past the batch's
    live pages take page -1 = TMA-OOB zeros, bytes still credited)."""
    bt = cutlass.make_array_view(block_table_tensor)
    last_live = cute.math.max(n_pages_b - cutlass.Int32(1), cutlass.Int32(0))
    for j in cutlass.range_constexpr(n_boxes):
        g = kv_tile * cutlass.Int32(TILE_N) + cutlass.Int32(j * box_rows)
        slot = g // cutlass.Int32(PAGE_SIZE)
        row_in_page = g % cutlass.Int32(PAGE_SIZE)
        page_live = cutlass.Int32(bt[batch_idx, cute.math.min(slot, last_live)])
        in_range = slot < n_pages_b
        page = _select_i32(in_range, page_live, cutlass.Int32(-1))
        tma_load_tile(
            smem_tile.shifted(j * box_rows * granu_elems),
            tma(cutlass.Int32(0), kv_head_idx, row_in_page, page),
            mbar,
            cta_group=1,
        )


@cute.jit
def _load_kv(smem_tile, tma, block_table_tensor, batch_idx, kv_head_idx, n_pages_b, kv_tile, mbar):
    """One K or V tile (TILE_N keys x TILE_K) into ``smem_tile``: dense TMA box or
    the paged block-table walk."""
    if cutlass.const_expr(PAGED_KV):
        _paged_load_tile(smem_tile, tma, block_table_tensor, batch_idx, kv_head_idx, n_pages_b, kv_tile, mbar, KV_BOX_ROWS, KV_BOXES, GRANU_ELEMS)
    else:
        tma_load_tile(smem_tile, tma(cutlass.Int32(0), kv_head_idx, kv_tile * cutlass.Int32(TILE_N), batch_idx), mbar, cta_group=1)


@cute.jit
def _kv_slot(seq, n_tiles):
    """Ring slot and full-barrier parity of the ``seq``-th load of the unit's
    K(0), K(1), V(0), K(2), V(1), ... sequence (V(n-1) is the 2n-1-th load: no
    K(n) precedes it)."""
    slot = seq % cutlass.Int32(STAGES)
    phase = (seq // cutlass.Int32(STAGES)) & cutlass.Int32(1)
    return slot, phase


@cute.jit
def _tmaldg_warp_group(
    tma_q_desc,
    tma_k_desc,
    tma_v_desc,
    sQ,
    sKV,
    bars,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    seqlen_q,
    seqlen_kv,
    head_idx,
    batch_idx,
    split_idx,
    tok0,
    block_table_tensor=None,
    block_table_v_tensor=None,
    paged_hnd: cutlass.Constexpr[bool] = False,
    use_pdl: cutlass.Constexpr[bool] = False,
):
    tma_q = GmemTileTma(tma_q_desc)
    if cutlass.const_expr(PAGED_KV and paged_hnd):
        # HND page pools: descriptor dims are (D, row, H_kv, page); every load
        # site keeps the (d, head, row, page) vocabulary, the swap lives here.
        _tk, _tv = GmemTileTma(tma_k_desc), GmemTileTma(tma_v_desc)
        tma_k = lambda d, h, r, p: _tk(d, r, h, p)  # noqa: E731
        tma_v = lambda d, h, r, p: _tv(d, r, h, p)  # noqa: E731
    else:
        tma_k = GmemTileTma(tma_k_desc)
        tma_v = GmemTileTma(tma_v_desc)

    if cutlass.const_expr(use_pdl):
        # Programmatic dependent launch: every global read of this role -- the lengths and the table (a serving stack's own
        # earlier kernels write them), the Q box (the slab a projection / norm-RoPE launch wrote), the K / V pages (the
        # write-through filled them) -- sits BELOW this wait; the descriptor prefetches above it are the prologue it overlaps.
        wait_on_dependent_grids()
        # The prologue is behind this CTA: release the dependent grid so it stages its own while the mainloop runs (its wait
        # still blocks on this grid's completion and memory flush, wherever this trigger sits).
        launch_dependent_grids()

    lo, hi, eff_seqlen_kv, _eq, _ql = _unit_bounds(seq_kv_lens_tensor, seq_q_lens_addr, batch_idx, split_idx, seqlen_q, seqlen_kv, tok0)
    n_tiles = hi - lo
    q_head_base = head_idx * cutlass.Int32(HEADS_PER_TILE)
    kv_head_idx = head_idx if cutlass.const_expr(CFG.PACK_GQA) else head_idx // cutlass.Int32(CFG.QH_PER_KH)
    n_pages_b = cutlass.Int32(0)
    if cutlass.const_expr(PAGED_KV):
        n_pages_b = (eff_seqlen_kv + cutlass.Int32(PAGE_SIZE - 1)) // cutlass.Int32(PAGE_SIZE)

    if n_tiles > cutlass.Int32(0):
        bars.mb_q_full.arrive(n_bytes=Q_TX_BYTES, pred=nvvm.elect_sync())
        tma_load_tile(sQ[0], tma_q(cutlass.Int32(0), q_head_base, tok0, batch_idx), bars.mb_q_full.smem_ptr, cta_group=1)

        # Ring order K(0), K(1), V(0), K(2), V(1), ...: K runs one tile ahead of V, so
        # K(t+2) is issued when V(t-1)'s slot frees (end of iteration t-1) and lands
        # a whole iteration before BMM1(t+2) needs it, while V(t+1) is issued when
        # K(t+1)'s slot frees (start of iteration t).  The MMA warp consumes in the
        # same order, so one 3-slot ring serves both operands (see _kv_slot).
        kv_state = PipelineState.start(phase=1)
        bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
        bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES, pred=nvvm.elect_sync())
        _load_kv(sKV[kv_state.idx], tma_k, block_table_tensor, batch_idx, kv_head_idx, n_pages_b, lo, bars.mb_kv_full[kv_state.idx].smem_ptr)
        kv_state = advance(kv_state, STAGES)
        for i in cutlass.range(0, n_tiles, 1, unroll=1):
            kv_tile = lo + i
            if i + cutlass.Int32(1) < n_tiles:
                bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
                bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES, pred=nvvm.elect_sync())
                _load_kv(
                    sKV[kv_state.idx],
                    tma_k,
                    block_table_tensor,
                    batch_idx,
                    kv_head_idx,
                    n_pages_b,
                    kv_tile + cutlass.Int32(1),
                    bars.mb_kv_full[kv_state.idx].smem_ptr,
                )
                kv_state = advance(kv_state, STAGES)
            bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES, pred=nvvm.elect_sync())
            _load_kv(sKV[kv_state.idx], tma_v, block_table_v_tensor, batch_idx, kv_head_idx, n_pages_b, kv_tile, bars.mb_kv_full[kv_state.idx].smem_ptr)
            kv_state = advance(kv_state, STAGES)


@cute.jit
def _mma_warp_group(
    sQ,
    sK,
    sVt,
    sP,
    tmem_ptr_i32,
    bars,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    seqlen_q,
    seqlen_kv,
    batch_idx,
    split_idx,
    tok0,
    use_pdl: cutlass.Constexpr[bool] = False,
):
    tmem_alloc(tmem_ptr_i32, TMEM_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(_BAR_TMEM, 32 * (SOFTMAX_WARPS + 1))

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=N_Q,
        m_dim=TILE_N,
        k_dim=0,
        a_negate=int(PARAMS.negate_scores),
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=N_Q,
        m_dim=128,
        a_major=1,
        b_major=1,
        k_dim=0,
    )
    # BMM1: S^T = K . Q^T -- K [128 keys x TILE_K] K-major, Q^T [N_Q x TILE_K] K-major.
    bmm1_desc = MmaDesc(
        M=TILE_N,
        N=N_Q,
        K=TILE_K,
        bpe_a=BPE,
        bpe_b=BPE,
        tile_k_hw=CFG.TILE_K_HW,
        btranspose=False,
        cta_group=1,
        idesc=idesc_qk,
        kind=MMA_KIND,
    )
    # BMM2: O^T = V^T . P^T -- V^T [128 d x 128 keys] MN-major (d contiguous),
    # P^T [N_Q x 128 keys] MN-major (the N_Q values of one key contiguous).
    bmm2_desc = MmaDesc(
        M=128,
        N=N_Q,
        K=TILE_N,
        bpe_a=BPE,
        bpe_b=BPE,
        tile_k_hw=CFG.TILE_K_HW,
        atranspose=True,
        btranspose=True,
        cta_group=1,
        idesc=idesc_pv,
        kind=MMA_KIND,
    )

    if cutlass.const_expr(use_pdl):
        # PDL: the TMEM allocation and the descriptor builds above are this role's prologue; its first global read (the lengths) is below.
        wait_on_dependent_grids()
    lo, hi, _ek, _eq, _ql = _unit_bounds(seq_kv_lens_tensor, seq_q_lens_addr, batch_idx, split_idx, seqlen_q, seqlen_kv, tok0)
    n_tiles = hi - lo

    # Barrier bookkeeping is arithmetic in the tile index (no loop-carried
    # pipeline state inside a data-dependent branch):
    #   ring load seq (K(0), K(1), V(0), K(2), V(1), ...): slot seq % STAGES, phase (seq // STAGES) & 1;
    #   S slot t % 2: the softmax releases it once per tile that used it, so
    #   BMM1(t) waits completion #(t // 2) of s_empty[t % 2], i.e. parity
    #   ((t // 2) - 1) & 1 -- which is 1 for t < 2 and passes a fresh barrier;
    #   p_full[t % 2] / bmm2_done[t % 2]: parity (t // 2) & 1.
    if n_tiles > cutlass.Int32(0):
        bars.mb_q_full.wait(cutlass.Int32(0))
        desc_Q = sQ[0].desc()

        # Prologue: BMM1(0).
        slot0, phase0 = _kv_slot(cutlass.Int32(0), n_tiles)
        bars.mb_kv_full[slot0].wait(phase0, spin=SPIN_RING_WAITS)
        bars.mb_s_empty[0].wait(cutlass.Int32(1), spin=SPIN_RING_WAITS)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        mma_ss(bmm1_desc, sK[slot0].desc(), desc_Q, tmem_raw.subview(cutlass.Int32(S_ACC_OFF[0])))
        elect_p = nvvm.elect_sync()
        bars.mb_s_full[0].arrive(cta_group=1, pred=elect_p)
        bars.mb_kv_empty[slot0].arrive(cta_group=1, pred=elect_p)

        for i in cutlass.range(0, n_tiles, 1, unroll=1):
            # BMM1(i + 1) ahead of BMM2(i): the softmax of tile i overlaps the next
            # score tile, and the K slot is released as early as possible.
            if i + cutlass.Int32(1) < n_tiles:
                t_next = i + cutlass.Int32(1)
                slot_k, phase_k = _kv_slot(cutlass.Int32(2) * t_next - cutlass.Int32(1), n_tiles)
                par_next = t_next & cutlass.Int32(1)
                phase_s_empty = ((t_next // cutlass.Int32(2)) - cutlass.Int32(1)) & cutlass.Int32(1)
                bars.mb_kv_full[slot_k].wait(phase_k, spin=SPIN_RING_WAITS)
                bars.mb_s_empty[par_next].wait(phase_s_empty, spin=SPIN_RING_WAITS)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
                s_off_next = _select_i32(par_next == cutlass.Int32(0), S_ACC_OFF[0], S_ACC_OFF[1])
                mma_ss(bmm1_desc, sK[slot_k].desc(), desc_Q, tmem_raw.subview(s_off_next))
                elect_p1 = nvvm.elect_sync()
                bars.mb_s_full[par_next].arrive(cta_group=1, pred=elect_p1)
                bars.mb_kv_empty[slot_k].arrive(cta_group=1, pred=elect_p1)

            seq_v = _select_i32(i + cutlass.Int32(1) < n_tiles, cutlass.Int32(2) * i + cutlass.Int32(2), cutlass.Int32(2) * n_tiles - cutlass.Int32(1))
            slot_v, phase_v = _kv_slot(seq_v, n_tiles)
            par_cur = i & cutlass.Int32(1)
            phase_p = (i // cutlass.Int32(2)) & cutlass.Int32(1)
            bars.mb_kv_full[slot_v].wait(phase_v, spin=SPIN_RING_WAITS)
            bars.mb_p_full[par_cur].wait(phase_p, spin=SPIN_RING_WAITS)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
            desc_P = sP[par_cur].desc()
            accum = cutlass.Boolean(i > cutlass.Int32(0))
            for blk in cutlass.range_constexpr(O_BLOCKS):
                mma_ss(
                    bmm2_desc,
                    sVt[slot_v].shifted(blk * 2 * TILE_N * GRANU_ELEMS).desc(),
                    desc_P,
                    tmem_raw.subview(cutlass.Int32(O_OFF[blk])),
                    accumulate=accum,
                )
            elect_p2 = nvvm.elect_sync()
            bars.mb_bmm2_done[par_cur].arrive(cta_group=1, pred=elect_p2)
            bars.mb_kv_empty[slot_v].arrive(cta_group=1, pred=elect_p2)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, TMEM_COLS, CTA_GROUP_KIND)


@cute.jit
def _warp_reduce_max(x):
    # ftz=True (the FMNMX intrinsic, as tile_dsl's row_max_reduction and the sparse sibling spell it): the bare
    # cute.math.max lowers to compare + select -- FSETP.NAN + FSETP.GTU + two FSEL per max, every predicate register busy
    # -- which tipped the 18-warp (N_Q = 64) tile's KV loop over ptxas's 96-register cap; the inline-asm max.f32 form
    # pins its operands around each asm block and spilled more.  Softmax maxima are finite or -inf, so only a score below
    # the fp32 normal range (|s| < 2^-126) could read differently (flushed to 0).
    for off in cutlass.range_constexpr(5):
        x = cute.math.max(x, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, x, 16 >> off, 31, kind=nvvm.Shfl.BFLY)), ftz=True)
    return x


@cute.jit
def _warp_reduce_sum(x):
    for off in cutlass.range_constexpr(5):
        x = x + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, x, 16 >> off, 31, kind=nvvm.Shfl.BFLY))
    return x


@cute.jit
def _softmax_warp_group(
    tmem_ptr_i32,
    bars,
    red_smem,
    mred_smem,
    sP_raw,
    o_tensor,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    seqlen_q,
    seqlen_kv,
    n_batch,
    head_idx,
    batch_idx,
    split_idx,
    tok0,
    scale_log2: cutlass.Float32,
    use_pdl: cutlass.Constexpr[bool] = False,
):
    nvvm.barrier_cta_sync(barrier_id=_BAR_TMEM, thread_count=32 * (SOFTMAX_WARPS + 1))
    tidx = cute.arch.thread_idx()[0]
    sm_warp = tidx // cutlass.Int32(32)  # 0 .. SOFTMAX_WARPS - 1
    if cutlass.const_expr(COL_GROUPS == 1):
        lane = tidx  # key row of S^T / P^T, d row of O^T
        col0 = cutlass.Int32(0)
    else:
        lane = tidx % cutlass.Int32(128)
        # Column group (warp-uniform): the first S^T / O^T column this thread owns.
        col0 = cute.arch.make_warp_uniform(tidx // cutlass.Int32(128)) * cutlass.Int32(COLS)
    G = cutlass.Int32(HEADS_PER_TILE)
    tmem_base = tmem_ptr_i32.load()

    if cutlass.const_expr(use_pdl):
        # PDL: the TMEM handshake above is this role's prologue; its first global read (the lengths) is below.
        wait_on_dependent_grids()
    lo, hi, eff_seqlen_kv, eff_seqlen_q, q_len_b = _unit_bounds(seq_kv_lens_tensor, seq_q_lens_addr, batch_idx, split_idx, seqlen_q, seqlen_kv, tok0)
    n_tiles = hi - lo
    causal_diag = cutlass.Int32(0)
    if cutlass.const_expr(CFG.BOTTOM_RIGHT):
        causal_diag = eff_seqlen_kv - eff_seqlen_q

    NEG_INF = cutlass.Float32(float("-inf"))
    ZERO = cutlass.Float32(0.0)
    ONE = cutlass.Float32(1.0)
    # Running column state: max in the scaled log2 domain (-inf = no live key
    # yet), this lane's partial sum (the lane sum is reduced once, at the end).
    m_vec = cutlass.Vector.from_elements(tuple(NEG_INF for _ in range(COLS)), cutlass.Float32)
    l_vec = cutlass.Vector.from_elements(tuple(ZERO for _ in range(COLS)), cutlass.Float32)

    red_ptr = Pointer(red_smem.data_ptr(), dtype=cutlass.Float32)
    mred_ptr = Pointer(mred_smem.data_ptr(), dtype=cutlass.Int32)
    # The column each thread publishes into its group's max slot (lanes 16..31 repeat 0..15: a duplicate max is idempotent),
    # the group's slot index, and the group's first warp (it re-arms the slot two tiles ahead).
    key_col = (tidx % cutlass.Int32(32)) % cutlass.Int32(COLS)
    grp = col0 // cutlass.Int32(COLS)
    group_lead = (sm_warp % cutlass.Int32(4)) == cutlass.Int32(0)

    for i in cutlass.range(0, n_tiles, 1, unroll=1):
        kv_tile = lo + i
        par = i & cutlass.Int32(1)
        phase_i = (i // cutlass.Int32(2)) & cutlass.Int32(1)
        s_off = _select_i32(par == cutlass.Int32(0), S_ACC_OFF[0], S_ACC_OFF[1])

        bars.mb_s_full[par].wait(phase_i, spin=SPIN_RING_WAITS)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        s_raw = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(tmem_base + s_off + col0, cutlass.Float32), num=COLS)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.mb_s_empty[par].arrive()

        key_abs = kv_tile * cutlass.Int32(TILE_N) + lane
        # Per-column (= per Q row) mask in the transposed orientation: the key
        # index = the thread's lane, the Q token = the unit's first token + the column's row // G.
        s_cols = []
        for j in cutlass.range_constexpr(COLS):
            v = cutlass.Float32(s_raw[j]) * scale_log2
            if cutlass.const_expr(CFG.MASK_FLAGS != 0):
                q_abs = tok0 + (col0 + cutlass.Int32(j)) // G
                masked = None
                if cutlass.const_expr(CFG.MASK_FLAGS & MASK_PADDED):
                    masked = key_abs >= eff_seqlen_kv
                if cutlass.const_expr(CFG.MASK_FLAGS & MASK_CAUSAL):
                    term = key_abs > (q_abs + causal_diag + cutlass.Int32(CFG.WINDOW_RIGHT))
                    masked = term if masked is None else (masked | term)
                if cutlass.const_expr(CFG.MASK_FLAGS & MASK_SWA):
                    term = key_abs < (q_abs + causal_diag - cutlass.Int32(CFG.WINDOW_LEFT))
                    masked = term if masked is None else (masked | term)
                v = _select_f32(masked, NEG_INF, v)
            s_cols.append(v)

        # Column max over the tile's 128 keys: butterfly inside the warp, then the group's four warps meet in the
        # tile's 16-word slot through red.shared.max on order-preserving keys; after the barrier every lane reads the
        # 16 group maxima back (no fold, nothing retained across the exchange).  The slot rotates per tile and the
        # group's first warp re-arms the slot two tiles ahead: its last readers passed this tile's barrier, its next
        # writers wait the next one, so the one barrier per tile orders every reuse.
        mbase = ((i % cutlass.Int32(MAX_RED_SLOTS)) * cutlass.Int32(COL_GROUPS) + grp) * cutlass.Int32(COLS)
        my_key = cutlass.Int32(INT32_MIN)
        for j in cutlass.range_constexpr(COLS):
            my_key = _select_i32(key_col == cutlass.Int32(j), _max_key(_warp_reduce_max(s_cols[j])), my_key)
        nvvm.inline_ptx("red.shared.max.s32 [{$r0}], {$r1};", read_only_args=[mred_smem.subview(mbase + key_col), my_key])
        nvvm.barrier_cta_sync(barrier_id=_BAR_SOFTMAX, thread_count=SOFTMAX_LANES)
        tile_max = []
        for j in cutlass.range_constexpr(COLS):
            tile_max.append(_key_max(cutlass.Int32((mred_ptr + (mbase + cutlass.Int32(j))).load())))
        if group_lead:
            rbase = (((i + cutlass.Int32(2)) % cutlass.Int32(MAX_RED_SLOTS)) * cutlass.Int32(COL_GROUPS) + grp) * cutlass.Int32(COLS)
            (mred_ptr + (rbase + key_col)).store(cutlass.Int32(INT32_MIN))

        # Online update per column (identical in every lane: the max is the
        # tile-wide column max, so alpha is uniform and O^T's rescale is exact).
        m_new = []
        l_new = []
        alpha = []
        all_one = None
        p_cols = []
        for j in cutlass.range_constexpr(COLS):
            m_old_j = cutlass.Float32(m_vec[j])
            m_new_j = cute.math.max(m_old_j, tile_max[j], ftz=True)
            ms_old = row_max_for_exp2(m_old_j)
            ms_new = row_max_for_exp2(m_new_j)
            alpha_j = cute.math.exp2(cute.math.min(ms_old - ms_new, ZERO), fastmath=True)
            p_j = cute.math.exp2(s_cols[j] - ms_new, fastmath=True)
            l_new.append(cutlass.Float32(l_vec[j]) * alpha_j + p_j)
            m_new.append(m_new_j)
            alpha.append(alpha_j)
            p_cols.append(p_j)
            is_one = alpha_j == ONE
            all_one = is_one if all_one is None else (all_one & is_one)
        m_vec = cutlass.Vector.from_elements(tuple(m_new), cutlass.Float32)
        l_vec = cutlass.Vector.from_elements(tuple(l_new), cutlass.Float32)

        # P^T(t) -> SMEM slot t % 2 as packed half pairs, row = this lane's key,
        # this group's COLS-wide slice of it, swizzled the way the BMM2 B
        # descriptor decodes it.
        p_words = []
        for w in cutlass.range_constexpr(P_GROUP_WORDS):
            p_words.append(fp32_to_fp16(p_cols[2 * w], p_cols[2 * w + 1], dtype=STORAGE_DTYPE))
        p_row_base = par * cutlass.Int32(pBufferWords) + lane * cutlass.Int32(P_ROW_WORDS) + (col0 // cutlass.Int32(2))
        for c in cutlass.range_constexpr(P_CHUNKS_PER_GROUP):
            chunk = cutlass.Vector.from_elements(tuple(p_words[4 * c + k] for k in range(4)), cutlass.Int32)
            p_ptr = Pointer(sP_raw.subview(p_row_base + cutlass.Int32(4 * c)).data_ptr(), dtype=cutlass.Int32)
            p_ptr.store_swizzled(chunk, _P_SMEM_SWIZZLE, alignment=16)
        nvvm.fence_proxy("async.shared", space="cta")

        # O^T(t-1) is complete once BMM2(t-1) commits; rescale this group's
        # columns by alpha before BMM2(t) accumulates on top.  Skipped when no
        # column's max moved.
        if i > cutlass.Int32(0):
            t_prev = i - cutlass.Int32(1)
            bars.mb_bmm2_done[t_prev & cutlass.Int32(1)].wait((t_prev // cutlass.Int32(2)) & cutlass.Int32(1), spin=SPIN_RING_WAITS)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
            if ~all_one:
                # Two x8 halves per d block (the register peak of the loop is here: alpha, the running max / sum and the
                # O^T values together; the x16 readback tipped the 18-warp N_Q = 64 tile over ptxas's 96-register cap).
                for blk in cutlass.range_constexpr(O_BLOCKS):
                    for h in cutlass.range_constexpr(2):
                        o_ptr = nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(O_OFF[blk] + h * (COLS // 2)) + col0, cutlass.Float32)
                        o_vals = nvvm.tcgen05_ld("32x32b", o_ptr, num=COLS // 2)
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                        o_scaled = cutlass.Vector.from_elements(
                            tuple(cutlass.Float32(o_vals[jj]) * alpha[h * (COLS // 2) + jj] for jj in range(COLS // 2)), cutlass.Float32
                        )
                        nvvm.tcgen05_st("32x32b", o_ptr, o_scaled)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        # Publish P^T(t) (and the rescaled O^T) to the MMA warp: the tcgen05 stores
        # are ordered before the arrive, and the MMA fences after its wait.
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.mb_p_full[par].arrive()

    # --- epilogue -----------------------------------------------------------
    if n_tiles > cutlass.Int32(0):
        # The last BMM2 must land before O^T is read.
        t_last = n_tiles - cutlass.Int32(1)
        bars.mb_bmm2_done[t_last & cutlass.Int32(1)].wait((t_last // cutlass.Int32(2)) & cutlass.Int32(1))
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)

    # Column sums over the 128 lanes (once per unit), via the epilogue slot.
    red_epi = sm_warp * cutlass.Int32(COLS)
    for j in cutlass.range_constexpr(COLS):
        (red_ptr + (red_epi + cutlass.Int32(j))).store(_warp_reduce_sum(cutlass.Float32(l_vec[j])))
    nvvm.barrier_cta_sync(barrier_id=_BAR_SOFTMAX, thread_count=SOFTMAX_LANES)
    red_epi_read = (sm_warp - (sm_warp % cutlass.Int32(4))) * cutlass.Int32(COLS)

    LN2_F = cutlass.Float32(LN2)
    TINY = cutlass.Float32(1e-30)
    # Under a KV split the fold moves to the combine pass (split_combine), which adds the sink column exactly once; the partials stay sink-free.
    if cutlass.const_expr(CFG.HAS_SINK and SPLIT_KV == 1):
        sinks_arr = cutlass.make_array_view(sinks_tensor)
    # Split partials stack split-major on the workspace batch axis (b + s*B).
    o_batch = batch_idx + split_idx * n_batch
    # O[q, d] = O^T[d, q] / l[q]: this lane is d (and d + 128); the store
    # writes the graph's actual d_v columns only (the tile is an envelope).
    oo = cutlass.make_array_view(o_tensor)
    D_V = cutlass.const_expr(o_tensor.shape[3])
    # The epilogue runs in two 8-column halves: each half derives its columns' inverse sum / dead / live facts, then
    # reads that half of O^T back (the x8 tcgen05.ld shape) and stores it.  The 16-wide per-column lists and the x16
    # O^T readback together were what the 18-warp N_Q = 64 tile spilled at ptxas's 96-register cap; the halves keep
    # the same arithmetic and the same stores (bitwise).  Each thread stores ONE LSE (the column its key lane names, lane == j), kept
    # as a per-lane scalar with a liveness bit; the column's token / head are two integer ops on a warp-uniform row
    # index, recomputed at each store site rather than carried in lists.
    O_HALF = COLS // 2
    my_lse = ZERO
    my_lse_live = lane < cutlass.Int32(0)  # False
    for h in cutlass.range_constexpr(2):
        inv_cols = []
        dead_cols = []
        live_cols = []
        for jj in cutlass.range_constexpr(O_HALF):
            j = h * O_HALF + jj
            jg = col0 + cutlass.Int32(j)  # the packed Q row this column is
            tok_j = tok0 + jg // G
            head_j = head_idx * G + (jg % G)
            # Every warp's partial comes back from SMEM (this warp's included), so
            # the group's four warps see the same total.
            l_tot = ZERO
            for w in cutlass.range_constexpr(4):
                l_tot = l_tot + cutlass.Float32((red_ptr + (red_epi_read + cutlass.Int32(w * COLS + j))).load())
            m_raw = cutlass.Float32(m_vec[j])  # -inf while the row has no live key
            m_nat = row_max_for_exp2(m_raw) * LN2_F
            row_dead = l_tot <= ZERO
            if cutlass.const_expr(CFG.HAS_SINK and SPLIT_KV == 1):
                # The sink folds against the row's ACTUAL max (what the prefill tile
                # does with its published max), -inf on a keyless row: new_max is
                # then the sink logit itself, scale = exp(-inf) = 0 and new_sum = 1,
                # so LSE is the sink logit exactly and O := 0 through inv = 0.  The
                # 0-substitute above is for exp2 arguments only: anchoring the fold
                # on it charged a phantom logit 0 against the sink, and a finite sink
                # far below it (-100) underflowed exp() to 0 -> LSE = -inf.
                m_fold = m_raw * LN2_F
                sink_logit = cutlass.Float32(sinks_arr[head_j])
                new_max = cute.math.max(m_fold, sink_logit)
                scale = cute.math.exp(m_fold - new_max, fastmath=True)
                new_sum = l_tot * scale + cute.math.exp(sink_logit - new_max, fastmath=True)
                lse_j = new_max + cute.math.log(new_sum, fastmath=True)
                inv_j = scale / new_sum
            else:
                lse_j = m_nat + cute.math.log(cute.math.max(l_tot, TINY), fastmath=True)
                inv_j = ONE / cute.math.max(l_tot, TINY)
                lse_j = _select_f32(row_dead, NEG_INF, lse_j)
                inv_j = _select_f32(row_dead, ZERO, inv_j)
            live_j = tok_j < seqlen_q
            if cutlass.const_expr(Q_BOX_ROWS < N_Q):
                # The zero-filled tail rows past the Q box (a head group that does not divide N_Q) decode
                # to the NEXT unit's first token (jg // G == Q_BOX_TOKENS), a live token whenever another
                # unit follows: they are never stored -- without this guard a non-last unit's tail
                # overwrote the next unit's first token (a CTA-order race: 5-14 % of that token's O
                # elements wrong, which unit's store landed last).
                live_j = live_j & (jg < cutlass.Int32(Q_BOX_ROWS))
            if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT):
                # Dense padded-Q trim: rows past seq_len_q[b] write O := 0 / LSE := -inf,
                # applied AFTER the sink fold (a trimmed row is dead even with a sink).
                row_trim = tok_j >= q_len_b
                lse_j = _select_f32(row_trim, NEG_INF, lse_j)
                inv_j = _select_f32(row_trim, ZERO, inv_j)
                row_dead = row_dead | row_trim
            if cutlass.const_expr(CFG.STATS_LOG2):
                lse_j = lse_j * cutlass.Float32(LOG2E)
            is_mine = lane == cutlass.Int32(j)
            my_lse = _select_f32(is_mine, lse_j, my_lse)
            my_lse_live = my_lse_live | (is_mine & live_j)
            inv_cols.append(inv_j)
            dead_cols.append(row_dead)
            live_cols.append(live_j)
        for blk in cutlass.range_constexpr(O_BLOCKS):
            if cutlass.const_expr(blk * 128 < D_V):
                d_idx = lane + cutlass.Int32(blk * 128)
                o_vals = cutlass.Vector.from_elements(tuple(ZERO for _ in range(O_HALF)), cutlass.Float32)
                if n_tiles > cutlass.Int32(0):
                    # Empty range: O^T TMEM was never written, so the load is skipped
                    # rather than multiplied by 0 (NaN * 0 is NaN).
                    o_vals = nvvm.tcgen05_ld(
                        "32x32b", nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(O_OFF[blk] + h * O_HALF) + col0, cutlass.Float32), num=O_HALF
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                if d_idx < cutlass.Int32(D_V):
                    for jj in cutlass.range_constexpr(O_HALF):
                        jg = col0 + cutlass.Int32(h * O_HALF + jj)
                        val = _select_f32(dead_cols[jj], ZERO, cutlass.Float32(o_vals[jj]) * inv_cols[jj])
                        if live_cols[jj]:
                            o_row = oo[o_batch, tok0 + jg // G, head_idx * G + (jg % G), :]
                            o_row[d_idx] = val.to(o_tensor.element_type)

    if cutlass.const_expr(lse_tensor is not None):
        lse_arr = cutlass.make_array_view(lse_tensor)
        if my_lse_live:
            jg = col0 + lane  # lanes 0 .. COLS-1 own one column each (my_lse_live is False past them)
            lse_arr[o_batch, head_idx * G + (jg % G), tok0 + jg // G] = my_lse

    bars.mb_tmem_dealloc.arrive()


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    o_tensor: cute.Tensor,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    n_batch: cutlass.Int32,
    scale_softmax_log2: cutlass.Float32,
    # Dense padded-Q trim: (B,)-int32 per-batch Q lengths by address; 0 (unread)
    # unless CFG.SEQ_Q_LENS_PRESENT.
    seq_q_lens_addr: cutlass.Int64 = 0,
    # Token units per (head group, batch, split): ceil(S_q / Q_BOX_TOKENS), derived by the host entry.
    n_tok_units: cutlass.Int32 = 1,
    # Paged KV: [B, max_pages] int32 page ids for K and V; None unless CFG.PAGED_KV.
    block_table_tensor: Optional[cute.Tensor] = None,
    block_table_v_tensor: Optional[cute.Tensor] = None,
    paged_hnd: cutlass.Constexpr[bool] = False,
    # Programmatic dependent launch (the launch attribute is set by the host entry under the same flag): each role waits on the
    # dependent grid right after its own prologue and before its first global read, and the TMA role releases the dependent grid.
    use_pdl: cutlass.Constexpr[bool] = False,
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()
    # One CTA per unit: x = (KV-head group (packed) or Q head (unpacked)) x token unit, y = batch, z = split.
    # The token unit is the fast index of x, so one group's units are launched together.
    head_idx = cute.arch.block_idx()[0] // n_tok_units
    tok0 = (cute.arch.block_idx()[0] % n_tok_units) * cutlass.Int32(Q_BOX_TOKENS)
    batch_idx = cute.arch.block_idx()[1]
    split_idx = cute.arch.block_idx()[2]

    sQ_raw = cutlass.Array(STORAGE_DTYPE, qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sKV_raw = cutlass.Array(STORAGE_DTYPE, STAGES * kvBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sP_raw = cutlass.Array(cutlass.Int32, 2 * pBufferWords, alignment=1024, space=cutlass.AddressSpace.smem)
    mred_smem = cutlass.Array(cutlass.Int32, MAX_RED_WORDS, alignment=16, space=cutlass.AddressSpace.smem)
    red_smem = cutlass.Array(cutlass.Float32, RED_WORDS, alignment=16, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)

    # Q^T: the K-major B operand of BMM1 (N_Q rows x TILE_K), one TMA sub-tile
    # (N_Q rows x 128 B) per 64-wide head-dim granule.
    sQ = SmemTile(
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=QK_ITERS,
        tma_granu_elems=GRANU_ELEMS,
        tma_subtile_stride_elems=N_Q * GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    # The K/V ring, seen as the K-major A operand of BMM1 (TILE_N rows x TILE_K) ...
    sK = SmemTile(
        base=sKV_raw,
        elems_per_stage=kvBufferElems,
        stages=STAGES,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=QK_ITERS,
        tma_granu_elems=GRANU_ELEMS,
        tma_subtile_stride_elems=TILE_N * GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    # ... and as the MN-major A operand of BMM2 (V^T: d contiguous, LBO steps
    # the 64-wide d atoms).  Same storage, different descriptor.
    sVt = SmemTile(
        base=sKV_raw,
        elems_per_stage=kvBufferElems,
        stages=STAGES,
        leading_byte_offset=LEADING_BYTE_OFFSET_VT,
        stride_byte_offset=STRIDE_BYTE_OFFSET_VT,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=VO_ITERS,
        tma_granu_elems=GRANU_ELEMS,
        tma_subtile_stride_elems=TILE_N * GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    # P^T: the MN-major B operand of BMM2, two buffers of TILE_N key rows.
    sP = SmemTile(
        base=sP_raw,
        elems_per_stage=pBufferWords,
        stages=2,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_P,
        layout=SMEM_LAYOUT_P,
        desc_version=DESC_VERSION,
    )

    bars = make_decode_bars()

    if warp_idx == 0:
        if nvvm.elect_sync():
            bars.mb_q_full.init()
            for s in cutlass.range_constexpr(STAGES):
                bars.mb_kv_full[s].init()
                bars.mb_kv_empty[s].init()
            for p in cutlass.range_constexpr(2):
                bars.mb_s_full[p].init()
                bars.mb_s_empty[p].init()
                bars.mb_p_full[p].init()
                bars.mb_bmm2_done[p].init()
            bars.mb_tmem_dealloc.init()

    if cutlass.const_expr(Q_BOX_ROWS < N_Q):
        # Zero the Q^T tail rows the TMA box does not cover (a head group that
        # does not divide N_Q), so their never-stored columns stay finite.
        if warp_idx < SOFTMAX_WARPS:
            zero4 = cutlass.Vector.from_elements(tuple(cutlass.Int32(0) for _ in range(4)), cutlass.Int32)
            tail_chunks = (N_Q - Q_BOX_ROWS) * 8  # 16 B chunks per sub-tile
            for it in cutlass.range_constexpr((QK_ITERS * tail_chunks + SOFTMAX_LANES - 1) // SOFTMAX_LANES):
                chunk = cutlass.Int32(it * SOFTMAX_LANES) + tidx
                if chunk < cutlass.Int32(QK_ITERS * tail_chunks):
                    sub = chunk // cutlass.Int32(tail_chunks)
                    within = chunk % cutlass.Int32(tail_chunks)
                    elem_off = sub * cutlass.Int32(N_Q * GRANU_ELEMS) + cutlass.Int32(Q_BOX_ROWS * GRANU_ELEMS) + within * cutlass.Int32(8)
                    Pointer(sQ_raw.subview(elem_off).data_ptr(), dtype=cutlass.Int32).store(zero4, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")

    if warp_idx < SOFTMAX_WARPS:
        # Arm every max-key slot (one word per lane): the first MAX_RED_SLOTS tiles find them at the smallest key.
        if tidx < cutlass.Int32(MAX_RED_WORDS):
            (Pointer(mred_smem.data_ptr(), dtype=cutlass.Int32) + tidx).store(cutlass.Int32(INT32_MIN))

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    if warp_idx < SOFTMAX_WARPS:
        _softmax_warp_group(
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            red_smem=red_smem,
            mred_smem=mred_smem,
            sP_raw=sP_raw,
            o_tensor=o_tensor,
            lse_tensor=lse_tensor,
            sinks_tensor=sinks_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            n_batch=n_batch,
            head_idx=head_idx,
            batch_idx=batch_idx,
            split_idx=split_idx,
            tok0=tok0,
            scale_log2=scale_softmax_log2,
            use_pdl=use_pdl,
        )
    elif warp_idx == MMA_WARP_ID:
        _mma_warp_group(
            sQ=sQ,
            sK=sK,
            sVt=sVt,
            sP=sP,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            batch_idx=batch_idx,
            split_idx=split_idx,
            tok0=tok0,
            use_pdl=use_pdl,
        )
    elif warp_idx == TMALDG_WARP_ID:
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        _tmaldg_warp_group(
            tma_q_desc=tma_q_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            sQ=sQ,
            sKV=sK,
            bars=bars,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            head_idx=head_idx,
            batch_idx=batch_idx,
            split_idx=split_idx,
            tok0=tok0,
            block_table_tensor=block_table_tensor,
            block_table_v_tensor=block_table_v_tensor,
            paged_hnd=paged_hnd,
            use_pdl=use_pdl,
        )


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    lse_ptr: Optional[cute.Pointer],
    sinks_ptr: cute.Pointer,
    meta_ptr: cute.Pointer,
    o_desc_ptr: cute.Pointer,
    problem_size: Tuple[int, int, int, int, int, int],
    q_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    k_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    v_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_ext: cutlass.Int64,
    scale_softmax_log2: cutlass.Float32,
    n_thd_units: cutlass.Int32,
    seq_q_lens_addr: cutlass.Int64,
    thd_q_lens_ptr: Optional[cute.Pointer],
    thd_kv_lens_ptr: Optional[cute.Pointer],
    thd_lens_form: Optional[cutlass.Int32],
    o_partial_ptr: Optional[cute.Pointer],
    block_table_ptr: Optional[cute.Pointer],
    block_table_v_ptr: Optional[cute.Pointer],
    table_strides: Tuple[cutlass.Int64, cutlass.Int64],
    n_pages: cutlass.Int32,
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    lse_kind: cutlass.Constexpr[str],
    paged_hnd: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool] = False,
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Bind the shared explicit SDPA ABI and launch the existing decode kernel.

    The private raw entry assumes validated operands: the compiled GQA ratio; any
    SQ (ceil(SQ / Q_BOX_TOKENS) token units per head group, Q_TOKEN_UNITS).  Graph /
    adapter calls bind through prepared.py; direct tests use launch_f16, including
    the 32-row tile where the routing does not select it.
    """
    B, QH, KH, SQ, SKV, _ = problem_size
    (
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        sinks_tensor,
        seq_kv_lens_tensor,
        o_desc_words,
        thd_q_lens_tensor,
        thd_kv_lens_tensor,
        o_partial_f32,
        block_table_tensor,
        block_table_v_tensor,
    ) = sdpa_operand_tensors(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        sinks_ptr,
        meta_ptr,
        o_desc_ptr,
        problem_size,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
        lse_strides,
        lse_ext,
        thd_q_lens_ptr,
        thd_kv_lens_ptr,
        thd_lens_form,
        o_partial_ptr,
        d_qk=d_qk,
        d_v=d_v,
        lse_kind=lse_kind,
        thd=False,
        split_kv=SPLIT_KV,
        tensor_map_qwords=16,
        paged=PAGED_KV,
        page_size=PAGE_SIZE,
        block_table_ptr=block_table_ptr,
        block_table_v_ptr=block_table_v_ptr,
        table_strides=table_strides,
        n_pages=n_pages,
    )

    # Q^T box: Q_BOX_TOKENS tokens x HEADS_PER_TILE heads of one KV group, row =
    # token * G + head (the packed-row convention); rows past S_q zero-fill.
    box_q = (1, Q_BOX_TOKENS, HEADS_PER_TILE, GRANU_ELEMS)
    box_kv = (1, KV_BOX_ROWS, 1, GRANU_ELEMS)
    stride_order = (3, 2, 1, 0)
    kv_stride_order = (3, 1, 2, 0) if paged_hnd else stride_order

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    def _create_tma_desc(tensor: cute.Tensor, box_dims, swizzle, order=stride_order):
        return tmap.create_tensor_map_tiled(
            global_address=tensor.iterator.toint(),
            dtype=tensor.element_type,
            global_dims=tuple(tensor.shape[i] for i in order),
            global_strides=tuple(cutlass.Int64(tensor.stride[i]) * tensor.element_type.width // 128 for i in order[1:]),
            box_dims=tuple(box_dims[i] for i in order),
            swizzle=swizzle,
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )

    tma_q_desc = _create_tma_desc(q_tensor, box_q, _tma_swz(CFG.Q_SWZ_BYTES))
    tma_k_desc = _create_tma_desc(k_tensor, box_kv, _tma_swz(CFG.K_SWZ_BYTES), kv_stride_order)
    tma_v_desc = _create_tma_desc(v_tensor, box_kv, _tma_swz(CFG.V_SWZ_BYTES), kv_stride_order)

    # Token units: ceil(SQ / Q_BOX_TOKENS) CTAs per (head group, batch, split), the unit the fast index of x.
    n_tok_units = (cutlass.Int32(SQ) + cutlass.Int32(Q_BOX_TOKENS - 1)) // cutlass.Int32(Q_BOX_TOKENS)
    grid_shape = ((QH // HEADS_PER_TILE) * n_tok_units, B, SPLIT_KV)

    kernel_args = (
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        o_tensor,
        lse_tensor,
        sinks_tensor,
        seq_kv_lens_tensor,
        cutlass.Int32(SQ),
        cutlass.Int32(SKV),
        cutlass.Int32(B),
        scale_softmax_log2,
        seq_q_lens_addr,
        n_tok_units,
    )

    # `pdl` is an explicit argument of both helpers (like `hnd`): the DSL refuses a closure capture inside a staged `if`.
    def launch(args, k_table, v_table, hnd, pdl, grid, current_stream):
        # pdl: the programmatic-stream-serialization launch attribute, paired with the kernel's griddepcontrol sites.
        _kernel(*args, k_table, v_table, hnd, pdl).launch(
            grid=grid,
            block=[CFG.THREADS_PER_CTA, 1, 1],
            stream=current_stream,
            use_pdl=pdl,
        )

    def launch_tables(args, kt, vt, kp, vp, strides, batch, kv_extent, hnd, pdl, grid, current_stream):
        if cutlass.const_expr(PAGED_KV):
            # Keep the common unit page-column stride static in the device code.
            if strides[1] == cutlass.Int64(1):
                layout = cute.make_layout((batch, kv_extent // cutlass.Int32(PAGE_SIZE)), stride=(strides[0], 1))
                launch(args, cute.make_tensor(kp, layout), cute.make_tensor(vp, layout), hnd, pdl, grid, current_stream)
            else:
                launch(args, kt, vt, hnd, pdl, grid, current_stream)
        else:
            launch(args, kt, vt, hnd, pdl, grid, current_stream)

    compact_inner = o_strides[2] == cutlass.Int64(d_v)
    if cutlass.const_expr(lse_ptr is not None):
        compact_inner = compact_inner & (lse_strides[2] == cutlass.Int64(1))
    if compact_inner:
        output = cute.make_tensor(o_ptr, cute.make_layout((B * SPLIT_KV, SQ, QH, d_v), stride=(o_strides[0], o_strides[1], d_v, 1)))
        stats = None
        if cutlass.const_expr(lse_ptr is not None):
            stats = cute.make_tensor(lse_ptr, cute.make_layout((B * SPLIT_KV, QH, SQ), stride=(lse_strides[0], lse_strides[1], 1)))
        args = kernel_args[:3] + (output, stats) + kernel_args[5:]
        launch_tables(
            args, block_table_tensor, block_table_v_tensor, block_table_ptr, block_table_v_ptr, table_strides, B, SKV, paged_hnd, use_pdl, grid_shape, stream
        )
    else:
        launch_tables(
            kernel_args,
            block_table_tensor,
            block_table_v_tensor,
            block_table_ptr,
            block_table_v_ptr,
            table_strides,
            B,
            SKV,
            paged_hnd,
            use_pdl,
            grid_shape,
            stream,
        )


EXPLICIT_ABI = True  # pointer/int host entry; the adapter builds the argument list itself
LSE_KINDS = ("dense",)  # the decode tile never serves THD


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
    has_lse: bool = True,
    lse_kind: str = "dense",
    paged_hnd: bool = False,
    use_pdl: bool = False,
) -> Callable:
    """Compile the shared pointer ABI for dense or paged decode. Shapes and strides
    are runtime arguments; head-dim envelopes, Stats presence and pool layout
    are specializations. All stride fakes, including page tables, are Int64.
    ``use_pdl`` (appended; in the key) traces the programmatic-dependent-launch form: the launch attribute plus the per-role
    ``griddepcontrol`` wait before the first global read and the release after the prologue -- the same math, bitwise."""
    _cache_key = _template_key(globals(), locals(), "compile")
    if not (0 < d_qk <= CFG.TILE_K and 0 < d_v <= CFG.TILE_O):
        raise ValueError(f"d256 envelope: need 0 < d_qk <= {CFG.TILE_K} and 0 < d_v <= {CFG.TILE_O}; got ({d_qk}, {d_v})")
    if (d_qk * CFG.BPE) % 16 != 0 or (d_v * CFG.BPE_O) % 16 != 0:
        raise ValueError(f"d256 envelope: d_qk*BPE and d_v*BPE must be 16-byte multiples (TMA global-stride rule); got ({d_qk}, {d_v}) at BPE={CFG.BPE}")
    if SPLIT_KV > 1 and not has_lse:
        raise ValueError("d256: split_kv > 1 requires has_lse=True (the per-split LSE drives the combine)")
    if lse_kind not in LSE_KINDS:
        raise ValueError(f"lse_kind must be one of {LSE_KINDS}; got {lse_kind!r}")
    if paged_hnd and not PAGED_KV:
        raise ValueError("paged_hnd is a paged-KV specialization")
    gmem = cute.AddressSpace.gmem

    def P(dtype, align=16):
        return cute.runtime.make_ptr(dtype, 16, gmem, assumed_align=align)  # fake: type only

    i32 = cutlass.Int32(0)
    i64_3 = (cutlass.Int64(0),) * 3  # stride slots: Int64 leaves, see _host
    return _compile_cached(
        _host,
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(cutlass.Float32 if _FP32_PARTIALS else STORAGE_DTYPE),
        P(cutlass.Float32, 4) if has_lse else None,
        P(cutlass.Float32),
        P(cutlass.Int32),
        P(cutlass.Int64),
        (0, 0, 0, 0, 0, 0),
        i64_3,
        i64_3,
        i64_3,
        i64_3,
        i64_3,
        cutlass.Int64(0),
        cutlass.Float32(0.0),
        i32,
        cutlass.Int64(0),
        None,
        None,
        None,
        P(cutlass.Float32) if _FP32_PARTIALS else None,
        P(cutlass.Int32, 4) if PAGED_KV else None,
        P(cutlass.Int32, 4) if PAGED_KV else None,
        (cutlass.Int64(0), cutlass.Int64(0)),
        i32,
        d_qk,
        d_v,
        lse_kind,
        paged_hnd,
        bool(use_pdl),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd_decode",
    )
