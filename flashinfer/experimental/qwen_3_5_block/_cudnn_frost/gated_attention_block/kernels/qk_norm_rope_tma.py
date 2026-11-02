# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stages (2)+(3) of the gated attention block, TMA-staged.

An A/B against ``qk_norm_rope.py``, not a replacement. Same math, same
bit-identical result; a different pipeline. Keep both until one wins on the
perf node, then delete the loser.

**Why this exists, from the measurement rather than from taste.** The LDG
kernel was register-bound, and ``defer_secondary_loads`` fixed that: ncu says
registers/thread 95 -> 64, occupancy limited by registers 5 -> 8 blocks against
a warp limit of 8, achieved occupancy 55% -> 87%. Registers stopped being the
binding constraint, and the WARP limit took over: ~28 warps/SM x 2 rows x 512 B
= ~28 KiB in flight per SM, against the ~35 KiB Little's law wants to saturate
10.8 TB/s at ~700 ns. It cannot buy more, because more rows per thread costs
the registers back and more warps per SM are not available.

**A staged SMEM ring is the one lever that adds in-flight bytes without either.**
``STAGES x TILE_ROWS x 512 B`` per CTA lives in SMEM, not registers, so the
occupancy stays where the defer fix put it.

Three structural wins fall out, and they are the reason to prefer this shape
even at parity:

* **``is_q`` becomes ONE warp-uniform decision per TILE.** A TMA tile is either
  all-Q or all-K by construction, so the per-row branch that picked between two
  base pointers and two strides is gone from the inner loop entirely.
* **The token stride lives in the DESCRIPTOR, not in kernel arithmetic.** That
  is what made the symbolic-stride artifact cost 25% against the compact one:
  the address math, not the traffic. A descriptor is built once on the host and
  costs the kernel nothing, so the strided source is free here.
* **The stores are TILE stores.** The old epilogue's ``rstd`` write was one
  lane emitting 4 scattered bytes per row, which is a 32-byte sector per row;
  here the tile's ``TILE_ROWS`` values are gathered in SMEM and leave as one
  coalesced store, and Q/K leave through TMA.

**The e4m3 epilogue (``fp8_out``; the quantized backward's Q / K rebuild).** With
``q8`` / ``k8`` bound the tile leaves as e4m3 instead of bf16: every lane rounds its 8
normed-and-rotated fp32 to bf16 FIRST (``cvt.rn.bf16x2.f32`` -- the very words the bf16
arm would hand TMA), widens them back, multiplies by the static per-tensor ``scale_q`` /
``scale_k`` (a per-TILE uniform select; the standalone kernel reads each once per thread from a
1-element slot, the quantized backward's fused prologue hands them in as kernel arguments) and
packs through ``cvt.rn.satfinite.e4m3x2.f32`` -- the quantize pass's own multiply and cvt
(``kernels/quantize.py``), so ``q8`` / ``k8`` are BYTE-IDENTICAL to quantizing the bf16
arm's output: the forward's own SDPA operands, recomputed without the bf16 round trip
through HBM.  The 8 bytes per lane leave through one ``st.global.v2`` into the COMPACT
``[T, H, D]`` destination (a warp-row is one contiguous 256-B segment); the output SMEM
ring, its fence, the TMA store and its drain are not traced (``sOut`` is not even
allocated by the fused launch), the input ring is unchanged.  K's last tile may
overshoot ``T``: its padded rows are skipped on the store (the bf16 arm has TMA clip
them).  The body is a ``@cute.jit`` function (``qk_norm_rope_tma_body``) taking a
JOB-RELATIVE block index and the SMEM arrays, so the quantized backward's fused
prologue launch (``fp8_bwd_fused.py``) runs this very code behind a block-range dispatch.

**Not swizzled, deliberately, and the arithmetic is in the SMEM buffer table
below.** Warp-per-row means 16 lanes cover a 256-byte contiguous half-row, so
the access already spreads all 32 banks: 4 wavefronts per instruction, which is
the ideal for 32 lanes x 16 B. TMA's 128 B swizzle would not improve it, and
the thread-per-row alternative it is designed against would be 4x worse (16
wavefronts) because a 512-byte row stride is a multiple of the 128-byte bank
cycle.

**The pool-write arm (``pool_write``: the serving DECODE step's norm + RoPE + V copy +
paged K / V write in ONE launch).**  With the two pools and the slot map bound, the flat
tile space grows a third band: Q tiles, K tiles, then V tiles -- the slab's V band
``[T, H_kv, D]`` through its own load descriptor with the K box, so the V band has exactly
the K band's tile count.  A V tile is loaded and COPIED (its row maths is traced like a K
row's and discarded: the warp collectives stay unconditional, nothing of it is stored); a K
tile is normed and rotated as before.  Every K / V row then goes to ``pool[slot //
page_size, head, slot % page_size, :]`` at ``slot = slot_mapping[token]`` from the SAME
lanes, 32 lanes x 16 B ``st.global.v4`` per 512-B row: the K words are the bf16 pair-rounds
(``cvt.rn.bf16x2.f32``) of the very fp32 this thread hands ``sOut[s]`` -- the vector store's
own rounding, so the pool row is the slab row, bitwise -- and the V words are the 16 B TMA
laid into ``sIn[s]`` (``ld.shared.v4``: a bit copy, no cast).  The predicate is
``cache_write.py``'s, verbatim: the row's token inside ``T`` (TMA zero-filled the band's
overshoot) AND ``0 <= slot < n_slots`` -- a negative slot is the serving stacks' padding
convention, a slot at or past the smaller pool's capacity a caller-contract violation;
neither writes.  The slot loads (one broadcast ``ld.global`` per row; a Q tile reads slot 0
and discards it, so the loads are unconditional) are issued BEFORE the tile's mbarrier wait
and ride in registers across it.  A V tile issues no slab TMA store and writes no rstd.
**Barrier table delta: none** -- no new mbarrier; the pool stores are generic-proxy stores
from registers inside the compute phase, before the fence + barrier that frees ``sIn[s]``;
nothing reads ``sOut[s]`` back, so the drain and the slab store are as before.  **SMEM delta:
none** -- the slots ride in ``rows_per_warp`` Int64 registers per lane, not in an SMEM
array.  The launch attribute and the ``griddepcontrol`` pair are the shipped ones (the wait
before the ring fill, which the slot loads follow; the release right after the wait).
Exclusive with the e4m3 and MX epilogues; refused with the LDG kernel (no arm there).
"""

import hashlib
from pathlib import Path
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.datatypes import _convert_to_cutlass_data_type
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import compile_cached, template_key
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import current_device, multiprocessor_count
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import PipelineState, advance, arrive_expect_tx, launch_dependent_grids, wait, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.handles import GmemTileTma, SmemTile
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import (
    abs_max_tree,
    e8m0_from_amax,
    e8m0_pair,
    f16x2_to_f32,
    fmax_f32,
    fp32_to_fp16,
    fp32_to_fp8x2,
    lane_group_sum,
    opaque_f32_zero,
    pack_u16x2,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.sf_layout import SF_ATOM_BYTES, SF_ATOM_COLS, SF_ATOM_LINE_BYTES, SF_ATOM_LINE_ROWS
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import (
    ld_global,
    ld_global_v2,
    ld_global_v4,
    ld_shared_v4,
    st_global,
    st_global_v2,
    st_global_v4,
    tma_load_tile,
    tma_store_commit,
    tma_store_tile,
    tma_store_wait,
)

from ..cache_write_contract import _SLOT_DTYPES, check_pool, check_slot_mapping
from .cache_write import _fake_pool as _fake_kv_pool
from .cache_write import _fake_slot as _fake_slot_map
from .cache_write import validate_cache_write_shape
from .qk_norm_rope import check_norm_weights_match_recipe
from .quantize import check_scalar_slot, require_fp8_cvt
from .quantize_mxfp8 import SF_BLOCK, SF_TILE_ROWS, n_sf_tiles, sf_bytes, sf_tile_bytes

ELEMS_PER_ACCESS = 8  # bf16 elements in one 16-byte access
WORDS_PER_ACCESS = 4  # 32-bit words in one 16-byte access
BPE = 2
TMA_GRANU_ELEMS = 128  # 256 B per row: the UNSWIZZLED TMA inner-box cap
WARP = 32
MX_TILE_TOKENS = SF_BLOCK  # the MX epilogue's TOKEN tile: one head x 32 consecutive tokens = one columnwise E8M0 block along S
MX_BLOCKS_PER_SF_TILE = SF_TILE_ROWS // MX_TILE_TOKENS  # 4: the 32-token ranges of one 128-row SF tile

DEFAULT_TILE_ROWS = 16
DEFAULT_STAGES = 2
DEFAULT_STAGES_O = 1
"""Output ring depth, INDEPENDENT of the input ring.

They are different resources -- input depth buys load-ahead, output depth buys
store slack -- and SMEM spent on one is SMEM not spent on the other. Measured
cold on the Rubin dev node, S=32768, 16-row tiles, L2-flushed against 4453 GB/s;
each buffer is 8 KiB, so SMEM per CTA is ``(stages + stages_o) * 8 KiB``:

    stages  stages_o  buffers  KiB/CTA    GB/s   % ceiling
      2         1        3        24      4437      99.6    <- default
      2         2        4        32      4439      99.7
      3         1        4        32      4416      99.2
      3         2        5        40      3593      80.7
      4         1        5        40      3533      79.3
      4         2        6        48      3520      79.1

**What decides it is the TOTAL buffer count, not which ring owns them.** Equal
footprints tie to within 0.5%; on the dev node the cliff sits between 4 and 5
buffers, where the resident CTA count steps down. So there is nothing to gain by
deepening EITHER ring, and the whole ``stages_o`` sweep at fixed ``stages`` is
flat: 1, 2, 3 and 4 all land within 0.4% at S=8192 and at S=32768 alike.

**The CLIFF is node-dependent; the flatness is not.** Re-run on the perf node at
S=32768, all four of (stages, stages_o) in {2,3} x {1,2} land within 0.5%
(8310 / 8302 / 8291 / 8273 GB/s) -- no cliff at 5 buffers there at all. So do not
port the cliff position between parts. What survives both is that deepening buys
nothing, which is why the default is the SMALLEST configuration: never worse,
and 25% less SMEM than the coupled one.

The useful consequence runs the other way. ``stages_o=1`` is exactly as fast as
4 and uses **25% less SMEM**, so it is the default: same throughput, 8 KiB per
CTA back. It costs a full ``tma_store_wait(0)`` drain each tile, which measures
free here because the TMA store is never the bottleneck -- if a part ever shows
one, ``stages_o=2`` is the flip and it is equally fast."""
DEFAULT_THREADS = 128
REFILL_LATE = 0
REFILL_AFTER_COMPUTE = 1
REFILL_TOP = 2
DEFAULT_REFILL_POS = REFILL_AFTER_COMPUTE
"""Where the NEXT tile's TMA load is issued.

* ``REFILL_LATE`` -- after the store issue and the rstd write.
* ``REFILL_AFTER_COMPUTE`` -- immediately after the post-compute barrier, so the
  load overlaps this tile's epilogue as well.
* ``REFILL_TOP`` -- at the top of the loop, refilling the stage the PREVIOUS
  iteration finished with.

**``REFILL_AFTER_COMPUTE`` is the earliest CORRECT point.** A stage's input
buffer is provably free only once every warp has finished reading it, which is
exactly what the post-compute barrier establishes. Issuing at the top of the loop
for the CURRENT stage would overwrite the data this iteration is about to read,
which is why ``REFILL_TOP`` refills one stage behind.

**And it does not matter.** All three tie, measured cold on the Rubin dev node,
16-row tiles, ``stages_o=1``, L2-flushed against 4453 GB/s:

    S       refill=0   refill=1   refill=2
    4096      3162       3155       3162
    8192      3822       3830       3839
    32768     4511       4515       4528

That is a 0.6% spread at S=32768 where the kernel is already at 101% of a
measured device-to-device copy, and 0.2% at S=4096 where it is at 71%. Load
placement is simply not what this pipeline is short of. The default stays at the
earliest correct point because it is the least surprising, not because it won.

Getting the load out *before the math* -- which is the thing worth wanting --
needs the row loop split into a read phase and a math phase with a barrier
between, so the refill can go out while the math runs. That holds every row's
operands in registers across the barrier (4 rows x 8 fp32 at the default tile),
i.e. it spends the exact resource the LDG kernel had to stop spending. Untested;
worth trying only on a part where this kernel is not already at the ceiling."""
DEFAULT_FUSED_STORE_WAIT = False
"""Two loop-ordering choices, both measured; the FUSED drain is legal only at ``stages_o >= 2``.

``fused_store_wait`` moves the store drain onto the barrier the epilogue already
runs, deleting a whole CTA barrier from the per-tile critical path.  It shipped
as the default at ``stages_o=1`` and that combination is a WRITE-AFTER-READ RACE
on ``sOut`` (found 2026-09-15): the fused ``tma_store_wait(stages_o - 1)`` runs
AFTER tile i's lanes have written ``sOut[s]``, so with one output stage the lanes
of tile i+1 overwrite ``sOut[0]`` while tile i's bulk store may still be reading
it -- tile i's GMEM rows then carry tile i+1's normed rows.  Reachable only when a
CTA processes >= 2 tiles (> SMs x 8 tiles per launch: h_q=32 at S >= 1024 on
Rubin; the unit test's 640-tile shape never reached it), and it shows up as
non-deterministic Q rows equal to the reference row of token ``tok + n_ctas /
q_tiles_per_token`` -- ~0.1 % of rows at S=16K, which moved a downstream causal
attention's cosine vs a second run to 0.998 and read as a "long-S SDPA bug".
The fused form needs ``tma_store_wait(stages_o - 2)`` (tile i+1's writer must
find tile i+1-stages_o's store retired) and therefore ``stages_o >= 2``;
``compile_qk_norm_rope_tma`` refuses the racy pair.  Its measured gain over the
unfused drain was +0.5 %, inside the control noise, so the default is the
unfused drain.
``refill_pos=REFILL_AFTER_COMPUTE`` issues the next tile's TMA before the store
issue and the rstd write rather than after them, so the load overlaps this
tile's epilogue too.

Measured cold on the Rubin dev node, S=32768, fused layout, L2-flushed against a
4453 GB/s ceiling flushed the same way:

    refill fused    GB/s   % ceiling   vs LDG
      0      0      4367       98.1     +41.8
      1      0      4375       98.2     +42.0
      0      1      4465      100.3     +45.0
      1      1      4487      100.8     +45.7      <- default

**Read that as a floor, not as the size of the win.** At 98-100% of a measured
device-to-device copy the kernel is saturated on this part, so the two orderings
have almost nothing left to buy and the 2.7% spread is compressed against the
ceiling. Both are free and both are structurally right, so both are on; re-measure
on a part with headroom before quoting either number as the effect size.

Final A/B at these defaults, LDG baseline PINNED with ``impl="ldg"``, cold,
fused layout, aarch64 Rubin PERF node (212 SMs, 2376 MHz):

    S        LDG    TMA    gain
    4096    4561   6046   +32.6%
    8192    5013   7130   +42.2%
    32768   5439   8308   +52.8%

**Achieved GB/s only -- do NOT quote a fraction on that node.** Its torch-`copy_`
ceiling probe reads 6791 GB/s while this kernel does 8308, i.e. 122%, which is
impossible; the same probe read 10859 GB/s on a different Rubin perf node at
identical clocks. The kernel number is exact and cold; the probe is what is
wrong there, and `probe_norm_rope_tma.py` now prints ``>CEIL!`` rather than a
plausible percentage.

Pinning matters: `_QkNormRope`'s default `impl="auto"` resolves to THIS kernel on
sm_90+, so an unpinned "baseline" benchmarks TMA against TMA and reports the
whole win as -2.6%."""
"""Ring geometry, set from measurement rather than from the SMEM budget.

Measured cold on the aarch64 Rubin perf node, B=1 bf16 S=32768, the block's real
fused layout, every point L2-flushed against a ceiling flushed the same way
(10859 GB/s). The LDG kernel here is the CURRENT one, i.e. it already carries
the +33% deferred-load fix:

    kernel                       GB/s   % ceiling   vs LDG
    LDG (shipped)                5564       51.2       --
    TMA rows=16 stages=2         8664       79.8    +55.7      <- default
    TMA rows=32 stages=1         7901       72.8    +42.0
    TMA rows=8  stages=2         7839       72.2    +40.3
    TMA rows=32 stages=2         7131       65.7    +28.2
    TMA rows=16 stages=1         6924       63.8    +24.5
    TMA rows=16 stages=3         6408       59.0    +15.2
    TMA rows=32 stages=3         5867       54.0     +5.5

Re-measured on the perf node with the shipped ``stages_o=1``, ``tile_rows`` is
confirmed: 8 -> 8201, **16 -> 8314**, 32 -> 6780 GB/s. 32 rows is an 18% loss,
so the peak is real and it is not an artifact of the dev node.

**Both axes are a genuine peak, and both peaks are the same trade.** ``stages=1``
loses 24 points because nothing overlaps; ``stages=3`` loses 21 because SMEM per
CTA goes 32 -> 48 KiB and the resident CTA count falls. ``rows=32`` doubles the
tile and pays the same way. The default sits at **32 KiB per CTA**
(``stages * tile_rows * 512 B * 2``, separate in and out buffers), which keeps 7
CTAs resident against the 227 KiB opt-in cap -- deliberately UNDER the 327 KiB
Rubin carveout, because reaching it costs L1 down to 8 kB and this kernel still
reads cos/sin and the norm weights through L1.

Re-measure per arch and per node: the two Rubin boxes here differ by 2.4x on the
copy ceiling alone. Reproduce with the internal bandwidth probe script."""

_FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
# Imported normally (not through the template loader), so the module names its own source identity: ``template_key`` joins
# it with a compile site's key to name the persistent object (``compiled_cache``); the cache's environment manifest covers
# every other ``.py`` the kernel is traced from.
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# BARRIER TABLE (cga1, no clusters: every arrive is LOCAL, so P6/P15 drains do
# not apply -- there is no cross-CTA arrive that could outlive a peer).
#
#   mbar          stages  producer          issuing lanes            consumers      init  phase
#   mb_full[s]    STAGES  TMA_LOAD          1 (warp_id==0 AND elect)  all threads   1     start(0)
#                         `elect_sync()` alone is one lane PER WARP = 4 arrives in
#                         a 4-warp CTA. The warp gate is load-bearing, and it must
#                         sit OUTSIDE tma_load_tile, which elects internally.
#                         arrive_expect_tx = TILE_ROWS*512 B; the two subtile
#                         TMA calls both complete into the SAME mbar, so the
#                         byte count is the whole tile, once.
#   There is NO _empty ring, deliberately. In this pipeline the producer and
#   the consumers are the SAME 128 threads, and the compute phase already ends
#   in a nvvm.barrier_cta_sync() -- which proves every thread finished reading
#   sIn[s] strictly before the refill is issued. An mbarrier would re-prove
#   that at the cost of 128 arrives per tile. (It was in the first draft, and
#   it was also MIS-PHASED: arriving and then waiting on the same barrier in
#   one iteration needs phase 0, not the pre-armed 1.)
#
#   SUM(issuing lanes) == init count:  mb_full 1 == 1.
#
#   THE MX ARM (qk_norm_rope_tma_mx_body; the MXFP8 backward's rebuild with the rowwise + columnwise block quantizes):
#   mb_full[s]    STAGES  TMA_LOAD          1 (warp_id==0 AND elect)  all threads   1     start(0)   -- unchanged; expect_tx =
#                         32*D*2 B, ONCE, both subtile TMAs into the SAME mbar (the box is (1, 32, 1, 128) over the 4-D band view)
#   bar.sync #1   1/tile  --                ALL threads (the row loop bound is uniform; the only divergence before it is a
#                         predicated store)                                                       SUM 128 == 128 (threads_per_cta)
#   bar.sync #2   1/tile  --                ALL threads, preceded by EVERY thread's fence.proxy.async.shared::cta: pass 1 WROTE
#                         the bf16-rounded words back into sIn[s] through the generic proxy and the TMA refill overwrites them
#                         through the async proxy -- a WAW the shipped arm never had (it only READS sIn).  The refill of stage s
#                         is issued AFTER this barrier by the elected lane of warp 0 (REFILL_AFTER_COMPUTE, the arm's only
#                         placement).                                                             SUM 128 == 128
#   grid          B * (ceil128(S)/32) * (h_q + h_kv) token tiles, heads INNER (mx_tile_counts); tiles entirely past S are
#                 VISITED: zero-filled load, SF 0x00 on every byte of the unit, NO payload store.  No _empty ring (the shipped
#                 argument: producer == consumers; bar #2 proves sIn[s] free), no cross-CTA arrive, no drain.
#
# SMEM BUFFER TABLE
#
#   name     dtype  elems                    writer          reader            per-lane stride  swizzle
#   sIn      Int32  STAGES*TILE_ROWS*128     TMA             lanes, 16 B each  16 B (contig)    NONE: 16 lanes
#                                                                                               span 256 contiguous
#                                                                                               bytes -> all 32 banks,
#                                                                                               4 wavefronts = ideal
#   sOut     Int32  STAGES*TILE_ROWS*128     lanes, 16 B     TMA               16 B (contig)    NONE, same argument
#            (the bf16 arm only: the e4m3 epilogue stores straight to GMEM and traces no sOut, no fence, no TMA store)
#   sRstd    Fp32   STAGES*TILE_ROWS         1 lane per row  lanes 0..R-1      4 B              n/a (R*4 <= 128 B)
#            (only when want_rstd traces True; absent from the RoPE-only / no-rstd kernel)
#   mb_*     Int64  STAGES each              --              --                --               n/a
#
#   THE MX ARM (36 KiB per CTA at two stages -> 6 CTAs / SM at the 227 KiB opt-in; no sOut, no sRstd):
#   sIn      bf16   STAGES*32*D              TMA (async);    lanes, 16 B each  16 B (contig)    NONE: the shipped argument (16
#                                            lanes WRITE BACK (pass 1: read, write back the bf16 words in place -- each lane  lanes span 256 B =
#                                            16 B in place    exactly the 16 B it read; pass 2: read)                            all 32 banks)
#   sRedCol  Fp32   WARPS*D                  lane: its 8     every lane: the    32 B             NONE: a warp's 32 lanes write
#                                            running |max|   other warps' 8     (2 x 16-B       D contiguous fp32 = 1 KiB = 8
#                                            at warp*D + d0  fp32 at w*D + d0   vectors)        full bank cycles (ideal)
#
#   THE POOL-WRITE ARM (pool_write; module docstring): NO new row in either table.  The paged K / V stores are generic-proxy
#   st.global.v4 from registers inside the compute phase (between the mbarrier wait and the post-compute fence + barrier that
#   frees sIn[s]); the slot values ride in registers (rows_per_warp Int64 per lane); a V tile neither TMA-stores nor writes rstd.
#   SUM(issuing lanes) == init count: unchanged, mb_full 1 == 1.
# ---------------------------------------------------------------------------


def validate_shape(d: int, rope_dim: int, h_q: int, h_kv: int, tile_rows: int, threads: int) -> None:
    """Raise on any geometry this pipeline cannot tile. Never an ``assert``:
    these come from user-facing geometry and must survive ``python -O``."""
    if d % TMA_GRANU_ELEMS != 0:
        raise ValueError(f"d_head must be a multiple of {TMA_GRANU_ELEMS} (the unswizzled TMA inner-box cap), got {d}")
    if d // ELEMS_PER_ACCESS != WARP:
        raise ValueError(
            f"this pipeline is warp-per-row: d_head must be {WARP * ELEMS_PER_ACCESS} so one warp covers a row in one 16-byte access each, got {d}"
        )
    if threads % WARP:
        raise ValueError(f"threads_per_cta must be a whole number of warps, got {threads}")
    warps = threads // WARP
    if tile_rows < 1:
        # resolve_tile_rows() returns 0 when nothing in 16..1 divides h_q, is a multiple of h_kv and of the warps (h_q = 20 MHA,
        # h_q = 6 over h_kv = 2): typed here so `impl="auto"` resolves to the LDG kernel -- the two modulo checks below would raise
        # ZeroDivisionError instead, which no caller catches
        raise ValueError(
            f"tile_rows={tile_rows}: no TMA tile fits h_q={h_q} / h_kv={h_kv} -- a tile is tile_rows consecutive heads of one token, so it must "
            f"divide h_q, be a multiple of h_kv and spread over the {warps} warps of the CTA; the LDG kernel serves this geometry"
        )
    if tile_rows % warps:
        raise ValueError(f"tile_rows={tile_rows} must divide evenly across the {warps} warps of the CTA")
    if h_q % tile_rows:
        raise ValueError(f"tile_rows={tile_rows} must divide h_q={h_q}: a Q tile is tile_rows consecutive heads of ONE token")
    if tile_rows % h_kv:
        raise ValueError(f"tile_rows={tile_rows} must be a multiple of h_kv={h_kv}: a K tile is whole tokens of {h_kv} heads")
    _validate_rope(rope_dim)


def _validate_rope(rope_dim: int) -> None:
    """The RoPE rules every arm of this pipeline shares (a whole-lane butterfly partner inside the first TMA subtile)."""
    if rope_dim:
        if rope_dim % (2 * ELEMS_PER_ACCESS):
            raise ValueError(f"rope_dim must be a multiple of {2 * ELEMS_PER_ACCESS} so the rotate_half partner is a whole-lane shuffle, got {rope_dim}")
        half_lanes = rope_dim // (2 * ELEMS_PER_ACCESS)
        if half_lanes & (half_lanes - 1):
            raise ValueError(f"rope_dim/{2 * ELEMS_PER_ACCESS} must be a power of two for the butterfly shuffle, got {rope_dim}")
        if rope_dim > TMA_GRANU_ELEMS:
            raise ValueError(f"rope_dim={rope_dim} must fit the first TMA subtile ({TMA_GRANU_ELEMS} elements) so the shuffle needs no cross-subtile exchange")


def validate_mx_shape(d: int, rope_dim: int, threads: int) -> None:
    """The MX arm's geometry (``qk_norm_rope_tma_mx_body``): the warp-per-row rule of the shipped pipeline, whole 32-token tiles
    over the warps, and the RoPE rules -- no ``tile_rows`` / head-count rule (the tile is ONE head x 32 tokens for any ``h_q`` /
    ``h_kv``).  Never an ``assert``."""
    if d % TMA_GRANU_ELEMS != 0:
        raise ValueError(f"d_head must be a multiple of {TMA_GRANU_ELEMS} (the unswizzled TMA inner-box cap), got {d}")
    if d // ELEMS_PER_ACCESS != WARP:
        raise ValueError(
            f"this pipeline is warp-per-row: d_head must be {WARP * ELEMS_PER_ACCESS} so one warp covers a row in one 16-byte access each, got {d}"
        )
    if d % SF_BLOCK != 0 or d % SF_TILE_ROWS != 0:
        raise ValueError(f"the MX epilogue needs whole 32-element blocks along D and whole 128-row D-planes: d_head={d}")
    if threads % WARP:
        raise ValueError(f"threads_per_cta must be a whole number of warps, got {threads}")
    warps = threads // WARP
    if MX_TILE_TOKENS % warps:
        raise ValueError(f"the MX epilogue's {MX_TILE_TOKENS}-token tile must spread evenly over the {warps} warps of the CTA (threads_per_cta={threads})")
    _validate_rope(rope_dim)


def _tile_handle(raw, off_elems, tile_elems: int, subtiles: int, subtile_elems: int):
    """A ``SmemTile`` view of one ring stage.

    Module-level and fully explicit on purpose: the DSL rejects a nested
    function that CAPTURES a variable when it is called from inside staged
    control flow (``SCOPE_CLOSURE_CAPTURE``), and every call here sits inside an
    ``if`` or a ``while``.
    """
    return SmemTile(
        base=raw.subview(off_elems),
        elems_per_stage=tile_elems,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=subtiles,
        tma_granu_elems=TMA_GRANU_ELEMS,
        tma_subtile_stride_elems=subtile_elems,
    )


@cute.jit
def _issue_tile_load(
    sIn_raw,
    s_off,
    mb_ptr,
    tile_idx,
    n_q_tiles,
    tma_q,
    tma_k,
    tile_elems: cutlass.Constexpr[int],
    subtiles: cutlass.Constexpr[int],
    subtile_elems: cutlass.Constexpr[int],
    tile_bytes: cutlass.Constexpr[int],
    tile_rows: cutlass.Constexpr[int],
    q_tiles_per_token: cutlass.Constexpr[int],
    toks_per_ktile: cutlass.Constexpr[int],
    n_qk_tiles=None,  # APPENDED (the pool-write arm): the first V tile; None = no V band
    tma_v=None,  # APPENDED: the V band's GmemTileTma (the K box)
):
    """Arm the tile's mbar and fire both subtile TMAs into it.

    **The caller must ALREADY have selected one warp.** `nvvm.elect_sync()`
    elects a lane PER WARP, so in a 4-warp CTA a bare `if elect_sync():` arms
    `expect_tx` four times against `init(1)` and issues the TMA four times.
    `tma_load_tile` elects internally, which is exactly why the warp gate cannot
    live in here -- it has to be outside. Symptom of getting it wrong: a bare
    `unspecified launch failure` that survives every swizzle and box you sweep,
    while a TMA *store* on the same descriptors is fine.

    Both subtile TMAs complete into the SAME mbar, so `expect_tx` is the whole
    tile ONCE, not once per subtile.
    """
    if nvvm.elect_sync():
        arrive_expect_tx(mb_ptr, tile_bytes)
    handle = _tile_handle(sIn_raw, s_off, tile_elems, subtiles, subtile_elems)
    if tile_idx < n_q_tiles:
        tma_load_tile(
            handle,
            tma_q(
                cutlass.Int32(0),
                (tile_idx % cutlass.Int32(q_tiles_per_token)) * cutlass.Int32(tile_rows),
                tile_idx // cutlass.Int32(q_tiles_per_token),
            ),
            mb_ptr,
        )
    else:
        if cutlass.const_expr(tma_v is not None):
            # the pool-write arm: the K band's tiles, then the V band's (the same box, each relative to its band's first tile)
            if tile_idx < n_qk_tiles:
                tma_load_tile(handle, tma_k(cutlass.Int32(0), cutlass.Int32(0), (tile_idx - n_q_tiles) * cutlass.Int32(toks_per_ktile)), mb_ptr)
            else:
                tma_load_tile(handle, tma_v(cutlass.Int32(0), cutlass.Int32(0), (tile_idx - n_qk_tiles) * cutlass.Int32(toks_per_ktile)), mb_ptr)
        else:
            tma_load_tile(handle, tma_k(cutlass.Int32(0), cutlass.Int32(0), (tile_idx - n_q_tiles) * cutlass.Int32(toks_per_ktile)), mb_ptr)


@cute.jit
def qk_norm_rope_tma_body(
    mQ: cute.Tensor,  # [T, H_q, D] -- bound for its element_type only; Q/K data
    mWq: Optional[cute.Tensor],  # [D]  moves entirely through the descriptors
    mWk: Optional[cute.Tensor],  # [D]  None (both): RoPE-only -- no RMSNorm, no weight loads, no rstd
    mCos: cute.Tensor,  # [T, ROPE_DIM]
    mSin: cute.Tensor,  # [T, ROPE_DIM]
    mRstdQ: Optional[cute.Tensor],  # [T, H_q]  fp32, or None
    mRstdK: Optional[cute.Tensor],  # [T, H_kv] fp32, or None
    tma_q,  # GmemTileTma handles, built by the calling kernel from its GridConstant descriptors
    tma_k,
    tma_qo,  # the bf16 OUTPUT descriptors; unused (placeholders) under the e4m3 epilogue
    tma_ko,
    mQ8: Optional[cute.Tensor],  # [T, H_q, D] e4m3 OUT (compact): the e4m3 epilogue; None = the bf16 TMA store
    mK8: Optional[cute.Tensor],  # [T, H_kv, D] e4m3 OUT
    scale_q: cutlass.Float32,  # the static scale_q (e4m3 epilogue only; 1.0, unused, otherwise) -- a VALUE: the standalone kernel reads
    scale_k: cutlass.Float32,  # its slot once per thread, the fused backward prologue hands a kernel argument (its init job writes the slot)
    n_tokens: cutlass.Int32,
    n_q_tiles: cutlass.Int32,
    n_tiles: cutlass.Int32,
    cta: cutlass.Int32,  # JOB-RELATIVE block index in [0, n_ctas): the first tile this CTA takes
    n_ctas: cutlass.Int32,
    eps: cutlass.Float32,
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    tile_rows: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    stages_o: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    refill_pos: cutlass.Constexpr[int],
    fused_store_wait: cutlass.Constexpr[bool],
    sIn_raw,  # io-dtype SMEM Array, stages * tile_elems (the input ring)
    sOut_raw,  # io-dtype SMEM Array, stages_o * tile_elems (the output ring; None under the e4m3 epilogue)
    sRstd,  # fp32 SMEM Array, stages * tile_rows (None without rstd)
    mb_full,  # Int64 SMEM Array, stages (the ring's full barriers)
    # -- APPENDED: the pool-write arm (module docstring).  Every one None / 0 under the shipped arms, so the positional callers read as before. --
    tma_v=None,  # GmemTileTma over the V band [T, H_kv, D] (the K box): the V tiles' loads
    mPoolK=None,  # [P, H_kv, page_size, D], the first three strides symbolic, D contiguous: the K pool
    mPoolV=None,  # the V pool (its own strides)
    mSlot=None,  # [T] int32 / int64: the flat slot of every token
    n_slots=None,  # Int64: P x page_size of the SMALLER pool -- a slot outside [0, n_slots) writes nothing
    page_size: cutlass.Constexpr[int] = 0,
    slot_i64: cutlass.Constexpr[bool] = False,
) -> None:
    """Grid-stride over tiles, ``stages``-deep TMA ring, warp-per-row compute.

    Q tiles come first in the flat tile space, then K tiles, so ``is_q`` is ONE
    comparison per TILE and warp-uniform. The LDG kernel's per-row branch, which
    selected between two base pointers and two token strides inside the hot
    loop, does not exist here.  The SMEM arrays are the CALLER's (allocated once
    per kernel, never per inlined arm); ``cta`` / ``n_ctas`` are the grid-stride
    the caller chose -- the standalone kernel's ``blockIdx.x`` / grid, a fused
    launch's arm offset / arm width.
    """
    warps = cutlass.const_expr(threads_per_cta // WARP)
    rows_per_warp = cutlass.const_expr(tile_rows // warps)
    tile_elems = cutlass.const_expr(tile_rows * d)
    subtile_elems = cutlass.const_expr(tile_rows * TMA_GRANU_ELEMS)
    subtiles = cutlass.const_expr(d // TMA_GRANU_ELEMS)
    q_tiles_per_token = cutlass.const_expr(h_q_ct // tile_rows)
    toks_per_ktile = cutlass.const_expr(tile_rows // h_kv_ct)
    rope_lanes = cutlass.const_expr(rope_dim // ELEMS_PER_ACCESS)
    half_lanes = cutlass.const_expr(rope_lanes // 2)
    tile_bytes = cutlass.const_expr(tile_rows * d * BPE)
    lanes_per_subtile = cutlass.const_expr(TMA_GRANU_ELEMS // ELEMS_PER_ACCESS)  # 16
    # Presence switches, decided at trace time (same idiom for both): None
    # weights trace the RoPE-only kernel (no sum-of-squares, no rsqrt, no
    # weight loads), None rstd traces the store-less one. The host refuses
    # want_rstd without apply_norm, so the conjunction below never hides a
    # requested store -- it only keeps the kernel self-consistent.
    apply_norm = cutlass.const_expr(mWq is not None)
    want_rstd = cutlass.const_expr(apply_norm and mRstdQ is not None)
    # The e4m3 epilogue (module docstring): q8 / k8 bound = no output ring, no TMA store, a direct 8-B store per lane.
    fp8_out = cutlass.const_expr(mQ8 is not None)
    # The pool-write arm (module docstring): the K band's post-RoPE rows and the V band's rows go to the paged pools from this launch.
    pool_write = cutlass.const_expr(mPoolK is not None)
    # Q tiles, K tiles, then -- under the arm -- the V tiles, as many as the K tiles: the K / V boundary is n_q + (n_tiles - n_q) / 2
    # (tile_counts(pool_write=True) builds n_tiles that way); without the arm every tile past the Q band is a K tile.
    n_qk_tiles = ((n_tiles + n_q_tiles) // cutlass.Int32(2)) if cutlass.const_expr(pool_write) else n_tiles

    io_dtype = mQ.element_type  # a trace-time type object, NOT a const_expr candidate

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(WARP)
    warp_id = tidx // cutlass.Int32(WARP)
    sub = lane // cutlass.Int32(lanes_per_subtile)  # which TMA subtile each lane reads
    col = lane % cutlass.Int32(lanes_per_subtile)  # 16-byte slot within it

    # --- P4: ONE warp, ONE lane inits EVERY stage of every ring -------------
    if warp_id == 0:
        if nvvm.elect_sync():
            for s in cutlass.range_constexpr(stages):
                nvvm.mbarrier_init(mb_full.subview(s), 1)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # The static per-tensor scales of the e4m3 epilogue arrive as VALUES (scale_q / scale_k: the caller's reads or arguments).
    my0 = cta

    # --- prologue: fill the ring -------------------------------------------
    for s in cutlass.range_constexpr(stages):
        t0 = my0 + cutlass.Int32(s) * n_ctas
        if t0 < n_tiles and warp_id == 0:  # ONE WARP; _issue_tile_load elects the lane
            _issue_tile_load(
                sIn_raw,
                s * tile_elems,
                mb_full.subview(s),
                t0,
                n_q_tiles,
                tma_q,
                tma_k,
                tile_elems,
                subtiles,
                subtile_elems,
                tile_bytes,
                tile_rows,
                q_tiles_per_token,
                toks_per_ktile,
                n_qk_tiles,
                tma_v,
            )

    full_state = PipelineState.start(phase=0)
    # The OUTPUT ring is independent of the input ring. They are two different
    # resources: the input depth buys load-ahead, the output depth buys store
    # slack, and SMEM spent on one is SMEM not spent on the other. Only .idx is
    # used -- there is no mbarrier on the output side, the drain is the
    # `tma_store_wait` group count.
    out_state = PipelineState.start(phase=0)
    # Loop-carried, for refill_pos == REFILL_TOP only: the stage the PREVIOUS
    # iteration finished with, and the tile it should be refilled with. -1 means
    # "nothing pending". This exists because the top of the loop cannot refill
    # its OWN stage -- that stage still holds the data this iteration is about
    # to read -- so the top placement necessarily refills one stage behind.
    prev_s = cutlass.Int32(0)
    prev_nxt = cutlass.Int32(-1)

    tile = my0
    while tile < n_tiles:
        s = full_state.idx
        so = out_state.idx
        is_q = tile < n_q_tiles  # ONE warp-uniform decision per TILE
        tok_q = tile // cutlass.Int32(q_tiles_per_token)
        head0_q = (tile % cutlass.Int32(q_tiles_per_token)) * cutlass.Int32(tile_rows)
        if cutlass.const_expr(pool_write):
            is_v = tile >= n_qk_tiles  # the V band: ONE warp-uniform decision per TILE, like is_q
            is_kv = tile >= n_q_tiles  # a K or a V tile: the rows that go to the pools
            kv_tile0 = n_qk_tiles if is_v else n_q_tiles
        else:
            kv_tile0 = n_q_tiles
        tok0_k = (tile - kv_tile0) * cutlass.Int32(toks_per_ktile)  # the first token of a K (or V) tile
        if cutlass.const_expr(pool_write):
            # The slots of this tile's rows, loaded BEFORE the TMA wait and kept in registers across it: one broadcast ld.global per
            # row (every lane of the warp reads its row's token; a Q tile reads slot 0 and discards it -- a select keeps the loads
            # unconditional).  int64: two 32-bit halves recombined, so the 64-bit value is what the range check sees and a slot
            # >= 2^31 can never alias an in-range one; int32: one word, sign-extended (cache_write.py's rule, verbatim).
            slots = []
            for u in cutlass.range_constexpr(rows_per_warp):
                r = warp_id + cutlass.Int32(u * warps)
                tok_s = tok0_k + r // cutlass.Int32(h_kv_ct)
                tok_s = tok_s if tok_s < n_tokens else n_tokens - cutlass.Int32(1)  # a padded K / V row: a valid address, its store predicated off
                tok_s = cutlass.Int32(0) if is_q else tok_s
                s_addr = mSlot.iterator.toint() + tok_s.to(cutlass.Int64) * (cutlass.Int64(8) if cutlass.const_expr(slot_i64) else cutlass.Int64(4))
                if cutlass.const_expr(slot_i64):
                    lo, hi = ld_global_v2(s_addr, cutlass.Int32)
                    slots.append((hi.to(cutlass.Int64) * cutlass.Int64(1 << 32)) + (lo.to(cutlass.Int64) & cutlass.Int64(0xFFFFFFFF)))
                else:
                    slots.append(ld_global(s_addr, cutlass.Int32).to(cutlass.Int64))
        wait(mb_full.subview(s), full_state.phase)
        full_state = advance(full_state, stages)
        out_state = advance(out_state, stages_o)

        if cutlass.const_expr(refill_pos == REFILL_TOP):
            if prev_nxt >= cutlass.Int32(0) and warp_id == 0:
                _issue_tile_load(
                    sIn_raw,
                    prev_s * cutlass.Int32(tile_elems),
                    mb_full.subview(prev_s),
                    prev_nxt,
                    n_q_tiles,
                    tma_q,
                    tma_k,
                    tile_elems,
                    subtiles,
                    subtile_elems,
                    tile_bytes,
                    tile_rows,
                    q_tiles_per_token,
                    toks_per_ktile,
                    n_qk_tiles,
                    tma_v,
                )

        # sOut[s]'s PREVIOUS store must have drained before any lane rewrites
        # it. Only the elected lane owns a bulk store group, so it waits and a
        # CTA barrier publishes that to every writer.
        #
        # `fused_store_wait` moves that pair onto the barrier the epilogue
        # ALREADY runs, which deletes a whole CTA barrier from the per-tile
        # critical path. Its drain runs one iteration ahead: the wait sits AFTER
        # tile i's lanes wrote sOut[s], so the buffer it must protect is the one
        # tile i+1 writes next, last read by the store of tile i+1-stages_o.
        # At that point the outstanding groups are tiles <= i-1 (tile i's store
        # is issued after the barrier), so `tma_store_wait(stages_o - 2)` is the
        # bound -- which needs stages_o >= 2.  `stages_o - 1` at stages_o=1 was
        # the WAR race described at DEFAULT_FUSED_STORE_WAIT.
        # The e4m3 epilogue has no output ring: nothing to drain, no barrier here.
        if cutlass.const_expr(not fused_store_wait and not fp8_out):
            if warp_id == 0:
                if nvvm.elect_sync():
                    tma_store_wait(stages_o - 1)
            nvvm.barrier_cta_sync()

        scale_t = scale_q if is_q else scale_k  # the e4m3 epilogue's per-TILE scale (an unused 1.0 otherwise)

        for u in cutlass.range_constexpr(rows_per_warp):
            r = warp_id + cutlass.Int32(u * warps)
            # K's LAST tile may overshoot T (a K tile is `toks_per_ktile` whole
            # tokens; Q never does -- tile_rows divides h_q). TMA clips the
            # overshooting rows on both the load (zero-fill) and the store, and
            # the rstd write below is predicated on `tok < n_tokens` -- but the
            # cos/sin TABLE loads are ordinary `ld.global`, so an unclamped
            # token here reads 16 B per rope lane past the end of a `[T, ROPE]`
            # table (T=3, h_kv=2, tile_rows=8: rows 6..7 are token 3; memcheck
            # on PR #1102). Clamp the padded rows to the last real token: the
            # table row is valid, the result is discarded by the TMA clip, and
            # the warp collectives (lane_group_sum, the RoPE shfl.bfly) stay
            # unconditional -- a `select`, not a branch.
            tok_k_raw = tok0_k + r // cutlass.Int32(h_kv_ct)
            tok_k = tok_k_raw if tok_k_raw < n_tokens else n_tokens - cutlass.Int32(1)
            tok = tok_q if is_q else tok_k
            # subtile-major: TMA laid the tile down as [subtile][row][GRANU].
            # The lane offset is shared, but the STAGE is not -- the input ring
            # and the output ring have independent depths, so a single `off`
            # would write this tile into whichever output buffer the INPUT ring
            # happened to be using. Silent, and wrong only once the two depths
            # differ.
            lane_off = sub * cutlass.Int32(subtile_elems) + r * cutlass.Int32(TMA_GRANU_ELEMS) + col * cutlass.Int32(ELEMS_PER_ACCESS)
            off_in = s * cutlass.Int32(tile_elems) + lane_off
            off_out = so * cutlass.Int32(tile_elems) + lane_off

            xs = sIn_raw.load(off_in, vector_size=ELEMS_PER_ACCESS, alignment=16).to(cutlass.Float32).to_elements()
            # Pre-bound so the names exist on both trace paths; under
            # apply_norm they are replaced by the real rsqrt / normed row, and
            # under RoPE-only `ys IS xs`: the passthrough dims [rope_dim, D)
            # are widened and re-narrowed with no fp32 op in between -> a
            # bit-exact copy (asserted by the tests).
            rstd = cutlass.Float32(1.0)
            ys = list(xs)
            if cutlass.const_expr(apply_norm):
                acc = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    acc = acc + xs[i] * xs[i]
                rstd = cute.math.rsqrt(lane_group_sum(acc, WARP) * cutlass.Float32(1.0 / d) + eps, fastmath=True)

                # Secondary loads at their POINT OF USE, for the same reason as the
                # LDG kernel: both are cache hits by construction, and hoisting them
                # holds 24 live fp32 per row across the reduction.
                w_addr = (mWq.iterator.toint() if is_q else mWk.iterator.toint()) + (
                    sub.to(cutlass.Int64) * cutlass.Int64(TMA_GRANU_ELEMS) + col.to(cutlass.Int64) * cutlass.Int64(ELEMS_PER_ACCESS)
                ) * cutlass.Int64(BPE)
                ws = [v for pair in [f16x2_to_f32(w, dtype=mWq.element_type) for w in ld_global_v4(w_addr, cutlass.Int32)] for v in pair]
                ys = [xs[i] * rstd * ws[i] for i in range(ELEMS_PER_ACCESS)]

            if cutlass.const_expr(rope_dim > 0):
                in_rope = lane < cutlass.Int32(rope_lanes)
                cs = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
                sn = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
                if in_rope:
                    rope_off = lane.to(cutlass.Int64) * cutlass.Int64(ELEMS_PER_ACCESS * BPE)
                    cb = mCos.iterator.toint() + tok.to(cutlass.Int64) * cutlass.Int64(mCos.stride[0]) * cutlass.Int64(BPE) + rope_off
                    sb = mSin.iterator.toint() + tok.to(cutlass.Int64) * cutlass.Int64(mSin.stride[0]) * cutlass.Int64(BPE) + rope_off
                    cs = [v for pair in [f16x2_to_f32(w, dtype=mCos.element_type) for w in ld_global_v4(cb, cutlass.Int32)] for v in pair]
                    sn = [v for pair in [f16x2_to_f32(w, dtype=mSin.element_type) for w in ld_global_v4(sb, cutlass.Int32)] for v in pair]
                # Unconditional: shfl.sync must be reached by every lane.
                rot = []
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    partner = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, ys[i], cutlass.Int32(half_lanes), 31, kind=nvvm.Shfl.BFLY))
                    signed = -partner if lane < cutlass.Int32(half_lanes) else partner
                    rot.append(ys[i] * cs[i] + signed * sn[i])
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    ys[i] = rot[i] if in_rope else ys[i]

            if cutlass.const_expr(fp8_out):
                # The e4m3 epilogue: the bf16 ROUNDING FIRST (cvt.rn.bf16x2.f32 -- the words the bf16 arm hands TMA), widened
                # back, times the tile's static scale, then the quantize pass's cvt.rn.satfinite.e4m3x2.f32 -- byte-identical to
                # quantizing the bf16 arm's output.  Two fp8x2 halves per 32-bit word, low byte first: BIT patterns through
                # st.global.v2, never a value cast through an fp8 view.  A padded K row (TMA would have clipped it) stores nothing.
                words16 = [fp32_to_fp16(ys[2 * i], ys[2 * i + 1], dtype=io_dtype) for i in range(ELEMS_PER_ACCESS // 2)]
                halves = []
                for w in words16:
                    lo, hi = f16x2_to_f32(w, dtype=io_dtype)
                    halves.append(fp32_to_fp8x2(lo * scale_t, hi * scale_t))
                packed8 = [pack_u16x2(halves[0], halves[1]), pack_u16x2(halves[2], halves[3])]
                col_elem = sub.to(cutlass.Int64) * cutlass.Int64(TMA_GRANU_ELEMS) + col.to(cutlass.Int64) * cutlass.Int64(ELEMS_PER_ACCESS)
                q_addr = (
                    mQ8.iterator.toint()
                    + tok_q.to(cutlass.Int64) * cutlass.Int64(mQ8.stride[0])
                    + (head0_q + r).to(cutlass.Int64) * cutlass.Int64(mQ8.stride[1])
                    + col_elem
                )
                head_k = r % cutlass.Int32(h_kv_ct)
                k_addr = (
                    mK8.iterator.toint()
                    + tok_k.to(cutlass.Int64) * cutlass.Int64(mK8.stride[0])
                    + head_k.to(cutlass.Int64) * cutlass.Int64(mK8.stride[1])
                    + col_elem
                )
                dst8 = q_addr if is_q else k_addr
                if is_q | (tok_k_raw < n_tokens):
                    st_global_v2(dst8, packed8, cutlass.Int32)
            else:
                sOut_raw.store(cutlass.Vector.from_elements(tuple(ys), cutlass.Float32).to(io_dtype), off_out, vector_size=ELEMS_PER_ACCESS, alignment=16)
            if cutlass.const_expr(pool_write):
                # The paged write of this row (K and V tiles; a Q tile's row is dead by the predicate): this thread's 16 B to
                # pool[page, head, off, lane*8 : +8] at slot = slot_mapping[token], page = slot // page_size, off = slot % page_size.
                # The K words are the bf16 pair-rounds of the very fp32 this thread just handed sOut[s] (cvt.rn.bf16x2.f32 == the
                # vector store's rounding: the pool row is the slab row, bitwise); the V words are the 16 B TMA laid into sIn[s]
                # (ld.shared.v4: a bit copy).  Everything is formed unconditionally and the band SELECTS -- no assignment inside a
                # dynamic branch (the DSL joins such a name as None on the other path), only the store is predicated, cache_write.py's
                # way: a K / V tile AND the row's token inside T (TMA zero-filled the band's overshoot) AND 0 <= slot < n_slots; a dead
                # row's page / off are garbage its predicated store ignores.
                pw_k = [fp32_to_fp16(ys[2 * i], ys[2 * i + 1], dtype=io_dtype) for i in range(ELEMS_PER_ACCESS // 2)]
                pw_v = ld_shared_v4(sIn_raw.data_ptr(off_in), cutlass.Int32)
                pw_words = [pw_v[i] if is_v else pw_k[i] for i in range(WORDS_PER_ACCESS)]
                pw_slot = slots[u]
                pw_live = is_kv & (tok_k_raw < n_tokens) & (pw_slot >= cutlass.Int64(0)) & (pw_slot < n_slots)
                pw_slot32 = pw_slot.to(cutlass.Int32)  # an in-range slot fits 32 bits; the divide by the constexpr page size strength-reduces
                pw_page = pw_slot32 // cutlass.Int32(page_size)
                pw_off = pw_slot32 - pw_page * cutlass.Int32(page_size)
                pw_head = r % cutlass.Int32(h_kv_ct)
                pw_base = mPoolV.iterator.toint() if is_v else mPoolK.iterator.toint()
                pw_s0 = cutlass.Int64(mPoolV.stride[0]) if is_v else cutlass.Int64(mPoolK.stride[0])
                pw_s1 = cutlass.Int64(mPoolV.stride[1]) if is_v else cutlass.Int64(mPoolK.stride[1])
                pw_s2 = cutlass.Int64(mPoolV.stride[2]) if is_v else cutlass.Int64(mPoolK.stride[2])
                pw_dst = (
                    pw_base
                    + (pw_page.to(cutlass.Int64) * pw_s0 + pw_head.to(cutlass.Int64) * pw_s1 + pw_off.to(cutlass.Int64) * pw_s2) * cutlass.Int64(BPE)
                    + lane.to(cutlass.Int64) * cutlass.Int64(ELEMS_PER_ACCESS * BPE)
                )
                if pw_live:
                    st_global_v4(pw_dst, pw_words, cutlass.Int32)
            if cutlass.const_expr(want_rstd):
                # gathered here, emitted below as ONE coalesced store for the
                # whole tile -- not one 4-byte scattered store per row
                if lane == cutlass.Int32(0):
                    sRstd.subview(s * cutlass.Int32(tile_rows) + r).store(rstd)

        if cutlass.const_expr(not fp8_out):
            nvvm.fence_proxy("async.shared", space="cta")  # lane writes -> TMA (async proxy)
        if cutlass.const_expr(fused_store_wait and not fp8_out):
            # stages_o >= 2 here (compile_qk_norm_rope_tma refuses the pair otherwise):
            # tile i+1 rewrites sOut[(i+1) % stages_o], last read by tile i+1-stages_o's
            # store; with tiles <= i-1 outstanding that leaves stages_o - 2 groups in flight.
            if warp_id == 0:
                if nvvm.elect_sync():
                    tma_store_wait(stages_o - 2)
        nvvm.barrier_cta_sync()  # publishes BOTH the lanes' writes and the drain (and, every arm, proves sIn[s] is free)

        # This stage's sIn is provably free the moment that barrier passes, so
        # the refill can go out BEFORE the store issue and the rstd write rather
        # than after them -- the load then overlaps this tile's epilogue too.
        nxt = tile + n_ctas * cutlass.Int32(stages)
        if cutlass.const_expr(refill_pos == REFILL_AFTER_COMPUTE):
            if nxt < n_tiles and warp_id == 0:
                _issue_tile_load(
                    sIn_raw,
                    s * cutlass.Int32(tile_elems),
                    mb_full.subview(s),
                    nxt,
                    n_q_tiles,
                    tma_q,
                    tma_k,
                    tile_elems,
                    subtiles,
                    subtile_elems,
                    tile_bytes,
                    tile_rows,
                    q_tiles_per_token,
                    toks_per_ktile,
                    n_qk_tiles,
                    tma_v,
                )

        if cutlass.const_expr(not fp8_out):
            if warp_id == 0:
                if nvvm.elect_sync():
                    if is_q:
                        tma_store_tile(
                            _tile_handle(sOut_raw, so * cutlass.Int32(tile_elems), tile_elems, subtiles, subtile_elems),
                            tma_qo(cutlass.Int32(0), head0_q, tok_q),
                        )
                    else:
                        if cutlass.const_expr(pool_write):
                            if tile < n_qk_tiles:  # a V tile leaves nothing in the slab (V was never moved); the K store is as before
                                tma_store_tile(
                                    _tile_handle(sOut_raw, so * cutlass.Int32(tile_elems), tile_elems, subtiles, subtile_elems),
                                    tma_ko(cutlass.Int32(0), cutlass.Int32(0), tok0_k),
                                )
                        else:
                            tma_store_tile(
                                _tile_handle(sOut_raw, so * cutlass.Int32(tile_elems), tile_elems, subtiles, subtile_elems),
                                tma_ko(cutlass.Int32(0), cutlass.Int32(0), tok0_k),
                            )
                    tma_store_commit()

        if cutlass.const_expr(want_rstd):
            if warp_id == 0:
                if lane < cutlass.Int32(tile_rows):
                    tok_l = tok_q if is_q else (tok0_k + lane // cutlass.Int32(h_kv_ct))
                    idx = (tok_q * cutlass.Int32(h_q_ct) + head0_q + lane) if is_q else (tok0_k * cutlass.Int32(h_kv_ct) + lane)
                    rbase = mRstdQ.iterator.toint() if is_q else mRstdK.iterator.toint()
                    val = sRstd.subview(s * cutlass.Int32(tile_rows) + lane).load()
                    rstd_row = tok_l < n_tokens  # K's last tile may overshoot; Q never does
                    if cutlass.const_expr(pool_write):
                        rstd_row = rstd_row & (tile < n_qk_tiles)  # a V tile's row maths is discarded: it has no rstd
                    if rstd_row:
                        st_global(rbase + idx.to(cutlass.Int64) * cutlass.Int64(4), val, cutlass.Float32)

        if cutlass.const_expr(refill_pos == REFILL_LATE):
            if nxt < n_tiles and warp_id == 0:
                _issue_tile_load(
                    sIn_raw,
                    s * cutlass.Int32(tile_elems),
                    mb_full.subview(s),
                    nxt,
                    n_q_tiles,
                    tma_q,
                    tma_k,
                    tile_elems,
                    subtiles,
                    subtile_elems,
                    tile_bytes,
                    tile_rows,
                    q_tiles_per_token,
                    toks_per_ktile,
                    n_qk_tiles,
                    tma_v,
                )
        if cutlass.const_expr(refill_pos == REFILL_TOP):
            prev_s = s
            prev_nxt = nxt if nxt < n_tiles else cutlass.Int32(-1)
        tile = tile + n_ctas

    # Every arrive above is LOCAL, so there is no cross-CTA drain to do (P15
    # does not apply at cga1). The only thing that must not outlive the CTA is
    # the last bulk store (the e4m3 epilogue issues none).
    if cutlass.const_expr(not fp8_out):
        tma_store_wait(0)


@cute.jit
def _issue_token_tile_load(
    sIn_raw,
    s_off,
    mb_ptr,
    tile_idx,
    n_blk,
    tma_q,
    tma_k,
    tile_elems: cutlass.Constexpr[int],
    subtiles: cutlass.Constexpr[int],
    subtile_elems: cutlass.Constexpr[int],
    tile_bytes: cutlass.Constexpr[int],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
):
    """The MX arm's twin of :func:`_issue_tile_load`: arm the tile's mbar and fire both subtile TMAs of ONE head x 32 TOKENS.

    **The caller must ALREADY have selected one warp** (the same rule as :func:`_issue_tile_load`: ``elect_sync`` elects a lane PER
    WARP).  The tile index decodes heads-INNER -- ``tile -> (range, head) = (tile // (h_q + h_kv), tile % (h_q + h_kv))``, the range
    ``-> (b, blk) = (range // n_blk, range % n_blk)`` over the PADDED sequence (``n_blk = ceil128(S) / 32`` ranges per batch) -- so the
    CTAs running at the same time take the heads of one 32-token range (one HBM read of its cos / sin rows, L2 hits for the rest).
    The descriptors are 4-D ``(B, S, H, D)`` views of the bands with box ``(1, 32, 1, 128)``: TMA clips the box at ``S``, so a block
    never straddles a batch and the tail of a ragged ``S`` -- a whole range past ``S`` included -- is zero-filled and tx-counted in
    full (the overshooting-box rule: ``expect_tx`` is the whole tile, once, for both subtiles into the same mbar)."""
    if nvvm.elect_sync():
        arrive_expect_tx(mb_ptr, tile_bytes)
    handle = _tile_handle(sIn_raw, s_off, tile_elems, subtiles, subtile_elems)
    heads_total = cutlass.const_expr(h_q_ct + h_kv_ct)
    rng = tile_idx // cutlass.Int32(heads_total)
    hh = tile_idx % cutlass.Int32(heads_total)
    b = rng // n_blk
    s0 = (rng % n_blk) * cutlass.Int32(MX_TILE_TOKENS)
    if hh < cutlass.Int32(h_q_ct):
        tma_load_tile(handle, tma_q(cutlass.Int32(0), hh, s0, b), mb_ptr)
    else:
        tma_load_tile(handle, tma_k(cutlass.Int32(0), hh - cutlass.Int32(h_q_ct), s0, b), mb_ptr)


def _st_global_b8(addr, value):
    """8-bit global store of the low byte of an ``Int32`` (PTX lets an 8-bit ``st`` take a 32-bit register)."""
    nvvm.inline_ptx("st.global.b8 [$0], $1;", read_only_args=[addr, value])


@cute.jit
def qk_norm_rope_tma_mx_body(
    mQ: cute.Tensor,  # the Q band (any view of it): bound for its element_type only; the data moves through the descriptors
    mWq: Optional[cute.Tensor],  # [D]  None (both): RoPE-only
    mWk: Optional[cute.Tensor],  # [D]
    mCos: cute.Tensor,  # [T, ROPE_DIM]
    mSin: cute.Tensor,  # [T, ROPE_DIM]
    tma_q,  # GmemTileTma over the 4-D (B, S, H_q, D) view of the Q band, box (1, 32, 1, 128): coords (d, h, s, b)
    tma_k,  # ... the (B, S, H_kv, D) view of the K band
    mQ8: cute.Tensor,  # [T, H_q, D] e4m3 compact OUT: the ROWWISE payload (blocks along D)
    mSfQ: cute.Tensor,  # [B*H_q*ceil(S/128)*4*D] u8 OUT: the SDPA rowwise SF tiles
    mQT8: cute.Tensor,  # [T, H_q, D] e4m3 compact OUT: the COLUMNWISE payload (blocks along S; row-major like the rowwise one)
    mSfQT: cute.Tensor,  # u8 OUT: the SDPA columnwise D-plane-major atoms (plane stride B*H_q*ceil(S/128)*512)
    mK8: cute.Tensor,  # the K twins over H_kv
    mSfK: cute.Tensor,
    mKT8: cute.Tensor,
    mSfKT: cute.Tensor,
    seq_len: cutlass.Int32,  # S (one sequence; T = B * S)
    batch: cutlass.Int32,
    n_blk: cutlass.Int32,  # ceil128(S) / 32: the 32-token ranges per batch over the PADDED sequence
    n_sft: cutlass.Int32,  # ceil(S / 128): the SF tiles per (b, h)
    n_tiles: cutlass.Int32,  # B * n_blk * (h_q + h_kv)
    cta: cutlass.Int32,  # JOB-RELATIVE block index in [0, n_ctas)
    n_ctas: cutlass.Int32,
    eps: cutlass.Float32,
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    sIn_raw,  # io-dtype SMEM Array, stages * 32 * d (the input ring; the bf16-ROUNDED rows are written back into it in place)
    sRedCol,  # fp32 SMEM Array, warps * d (the per-warp running column maxima)
    mb_full,  # Int64 SMEM Array, stages
) -> None:
    """The MXFP8 backward's Q / K rebuild with the MX EPILOGUE: norm + RoPE from the slab's PRE-norm bands and, from the SAME tile,
    the ROWWISE (``q8 / sf_q``, ``k8 / sf_k``) and the COLUMNWISE (``q_T8 / sf_q_T``, ``k_T8 / sf_k_T``) MXFP8 quantizations -- the
    four standalone block quantizes of the rebuilt bf16 buffers, byte for byte, without the bf16 round trip through HBM.

    The tile is ONE head x 32 consecutive TOKENS (``MX_TILE_TOKENS``) -- a columnwise E8M0 block is 32 consecutive tokens of one
    ``(h, d)``, which the shipped tile (one token x ``tile_rows`` heads) never holds.  Warp-per-row over the 32 token rows
    (``rows_per_warp = 32 / warps``), each lane's 8 ``d`` as the shipped body.  Per tile, TWO passes:

    * pass 1 (per row): norm + RoPE -> the bf16 ROUNDING FIRST (``cvt.rn.bf16x2.f32``: the very words the bf16 arm hands TMA),
      widened back; a row past ``S`` is zeroed (TMA zero-filled it already; the select keeps the maths of a padded row out of both
      amaxes); (i) the ROWWISE quantize -- the 4-lane block amax (``abs_max_tree`` over each lane's 8, ``shfl.bfly`` 1 and 2),
      ``e8m0_from_amax``, ``x * rcp`` -> ``cvt.rn.satfinite.e4m3x2`` (the standalone's exact arm, per element pair), 8 B per lane
      through ``st.global.v2`` into the compact ``[T, H, D]`` payload PREDICATED on ``s < S``, the block's SF byte gathered from the
      4 block leaders of this (row, subtile) into ONE 4-byte store (the 4 blocks of one atom row are 4 contiguous bytes of the
      per-(b, h, s_tile) 1024-B SDPA tile: ``sub*512 + r*16 + tb*4``) -- UNpredicated, a pad row writes ``0x00``; (ii) the bf16 words
      WRITTEN BACK INTO ``sIn[s]`` in place (each lane overwrites exactly the 16 B it read); (iii) 8 running ``|max|`` per lane over
      the warp's rows (its 8 ``d``);
    * ``bar.sync`` #1; the warps' column maxima combined per ``d`` through ``sRedCol`` (fixed order, ``max`` exact), ``e8m0_pair`` per
      ``d`` pair; warp 0 stores the 256 columnwise SF bytes (8 single bytes per lane at the D-plane-major byte
      ``plane * plane_stride + tile*512 + ((d%128)%32)*16 + ((d%128)//32)*4 + tb``) -- UNpredicated too;
    * pass 2 (per row): each lane re-reads its bf16 words from ``sIn[s]``, scales by its 8 COLUMN rcps, packs and stores 8 B into
      the compact row-major ``[T, H, D]`` columnwise payload PREDICATED on ``s < S`` -- a transposed payload is NOT the transpose of
      the payload (only the scaling axis differs), so it takes the 16-byte-class stores the standalone columnwise arm cannot;
    * EVERY thread's ``fence.proxy.async.shared::cta`` (pass 1 wrote ``sIn[s]`` through the generic proxy; the TMA refill overwrites
      the same bytes through the async proxy -- a WAW the shipped kernel never had), ``bar.sync`` #2, then the elected lane of warp 0
      refills stage ``s`` (``REFILL_AFTER_COMPUTE``, the only placement this arm has).

    Grid: ``B * (ceil128(S) / 32) * (h_q + h_kv)`` token tiles, heads INNER (:func:`_issue_token_tile_load`); the tiles of the last
    128-token SF unit that lie entirely past ``S`` are VISITED (zero-filled load -> SF ``0x00`` on every byte of the unit, no payload
    store) -- the standalone quantizer's tail rule, so every SF byte the SDPA's whole-tile SF TMA reads is written.  BITWISE the
    standalone chain by construction (same helpers, same order, on the bf16-rounded values; the 4-lane block amax equals a lane-pair
    one; ``e8m0_pair``'s bytes are two independent ``cvt.rp`` lanes of the same product).  SMEM: the input ring (32 KiB at two
    stages) + ``sRedCol`` (4 KiB) -- no output ring, no rstd; 6 CTAs per SM at the 227 KiB opt-in.  The barrier and SMEM tables are
    the module header's "MX arm" rows."""
    warps = cutlass.const_expr(threads_per_cta // WARP)
    rows_per_warp = cutlass.const_expr(MX_TILE_TOKENS // warps)
    tile_elems = cutlass.const_expr(MX_TILE_TOKENS * d)
    subtile_elems = cutlass.const_expr(MX_TILE_TOKENS * TMA_GRANU_ELEMS)
    subtiles = cutlass.const_expr(d // TMA_GRANU_ELEMS)
    tile_bytes = cutlass.const_expr(MX_TILE_TOKENS * d * BPE)
    rope_lanes = cutlass.const_expr(rope_dim // ELEMS_PER_ACCESS)
    half_lanes = cutlass.const_expr(rope_lanes // 2)
    lanes_per_subtile = cutlass.const_expr(TMA_GRANU_ELEMS // ELEMS_PER_ACCESS)  # 16
    lanes_per_sf_line = cutlass.const_expr(SF_ATOM_LINE_ROWS // ELEMS_PER_ACCESS)  # 4: one 16-byte SF line covers 32 d = 4 lanes' 8
    fp32_per_access = cutlass.const_expr(ELEMS_PER_ACCESS // 2)  # 4 fp32 in one 16-byte SMEM access: a lane's 8 column maxima move as two
    heads_total = cutlass.const_expr(h_q_ct + h_kv_ct)
    row_tile_bytes = cutlass.const_expr(sf_tile_bytes(d))  # 4*D: the per-(b, h, s_tile) rowwise SF tile
    apply_norm = cutlass.const_expr(mWq is not None)
    io_dtype = mQ.element_type  # a trace-time type object, NOT a const_expr candidate

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(WARP)
    warp_id = tidx // cutlass.Int32(WARP)
    sub = lane // cutlass.Int32(lanes_per_subtile)  # which TMA subtile each lane reads
    col = lane % cutlass.Int32(lanes_per_subtile)  # 16-byte slot within it
    d0 = sub * cutlass.Int32(TMA_GRANU_ELEMS) + col * cutlass.Int32(ELEMS_PER_ACCESS)  # d0: the first d of each lane
    col_leader = col == cutlass.Int32(0)  # gathers the 4 SF bytes of its (row, subtile)
    # the columnwise SF byte of each lane's d = d0 + i: plane = sub, dm = col*8 + i -> the F8_128x4 atom rule (dm % 32) * 16 + (dm // 32) * 4
    # (+ tb), i.e. ((col%4)*8 + i) * SF_ATOM_LINE_BYTES + (col//4) * SF_ATOM_COLS: 8 bytes 16 B apart (`sf_layout.sf_atom_byte`, spelled per lane)
    col_sf_lane = ((col % cutlass.Int32(lanes_per_sf_line)) * cutlass.Int32(ELEMS_PER_ACCESS)) * cutlass.Int32(SF_ATOM_LINE_BYTES) + (
        col // cutlass.Int32(lanes_per_sf_line)
    ) * cutlass.Int32(SF_ATOM_COLS)

    # --- P4: ONE warp, ONE lane inits EVERY stage of the ring -------------
    if warp_id == 0:
        if nvvm.elect_sync():
            for s in cutlass.range_constexpr(stages):
                nvvm.mbarrier_init(mb_full.subview(s), 1)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # --- prologue: fill the ring -------------------------------------------
    for s in cutlass.range_constexpr(stages):
        t0 = cta + cutlass.Int32(s) * n_ctas
        if t0 < n_tiles and warp_id == 0:  # ONE WARP; the issue helper elects one lane
            _issue_token_tile_load(
                sIn_raw, s * tile_elems, mb_full.subview(s), t0, n_blk, tma_q, tma_k, tile_elems, subtiles, subtile_elems, tile_bytes, h_q_ct, h_kv_ct
            )

    full_state = PipelineState.start(phase=0)
    zero = opaque_f32_zero()  # the running maxima's seed: a register, never a float immediate into max.f32
    tile = cta
    while tile < n_tiles:
        s = full_state.idx
        wait(mb_full.subview(s), full_state.phase)
        full_state = advance(full_state, stages)

        # --- the tile: (b, blk) over the padded sequence, the head, its SF tile and payload bases (all warp-uniform) ---
        rng = tile // cutlass.Int32(heads_total)
        hh = tile % cutlass.Int32(heads_total)
        b = rng // n_blk
        blk = rng % n_blk
        s0 = blk * cutlass.Int32(MX_TILE_TOKENS)
        s_tile = blk // cutlass.Int32(MX_BLOCKS_PER_SF_TILE)
        tb = blk % cutlass.Int32(MX_BLOCKS_PER_SF_TILE)
        is_q = hh < cutlass.Int32(h_q_ct)
        head = hh if is_q else hh - cutlass.Int32(h_q_ct)
        tok_base = b * seq_len
        sf_tile_q = (b * cutlass.Int32(h_q_ct) + head) * n_sft + s_tile
        sf_tile_k = (b * cutlass.Int32(h_kv_ct) + head) * n_sft + s_tile
        sf_tile = sf_tile_q if is_q else sf_tile_k
        # the rowwise SF tile (4*D contiguous bytes) and the columnwise atoms (one 512-B atom per D-plane, planes v_sf_groups atoms apart)
        sf_row_base = (mSfQ.iterator.toint() if is_q else mSfK.iterator.toint()) + sf_tile.to(cutlass.Int64) * cutlass.Int64(row_tile_bytes)
        v_sf_groups = (batch * cutlass.Int32(h_q_ct) if is_q else batch * cutlass.Int32(h_kv_ct)) * n_sft
        plane_stride = v_sf_groups.to(cutlass.Int64) * cutlass.Int64(SF_ATOM_BYTES)
        sf_col_base = (mSfQT.iterator.toint() if is_q else mSfKT.iterator.toint()) + sf_tile.to(cutlass.Int64) * cutlass.Int64(SF_ATOM_BYTES)
        pay_tok_stride = cutlass.Int64(mQ8.stride[0]) if is_q else cutlass.Int64(mK8.stride[0])
        pay_base = (
            (mQ8.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(mQ8.stride[1]))
            if is_q
            else (mK8.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(mK8.stride[1]))
        ) + d0.to(cutlass.Int64)
        payt_tok_stride = cutlass.Int64(mQT8.stride[0]) if is_q else cutlass.Int64(mKT8.stride[0])
        payt_base = (
            (mQT8.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(mQT8.stride[1]))
            if is_q
            else (mKT8.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(mKT8.stride[1]))
        ) + d0.to(cutlass.Int64)

        colmax = [zero for _ in range(ELEMS_PER_ACCESS)]  # a comprehension: a statement-level `for range()` in a kernel is STAGED

        # --- pass 1: norm + RoPE -> bf16 -> the rowwise quantize, the column maxima, the write-back -----------------
        for u in cutlass.range_constexpr(rows_per_warp):
            r = warp_id + cutlass.Int32(u * warps)  # token row within the tile
            s_row = s0 + r
            valid = s_row < seq_len
            # the cos / sin TABLE loads are ordinary ld.global: a padded row (TMA zero-filled it) reads the last real token's row
            # instead (a select, not a branch -- the warp collectives below stay unconditional); its result is zeroed anyway
            tok = tok_base + (s_row if valid else seq_len - cutlass.Int32(1))
            off_in = (
                s * cutlass.Int32(tile_elems) + sub * cutlass.Int32(subtile_elems) + r * cutlass.Int32(TMA_GRANU_ELEMS) + col * cutlass.Int32(ELEMS_PER_ACCESS)
            )

            xs = sIn_raw.load(off_in, vector_size=ELEMS_PER_ACCESS, alignment=16).to(cutlass.Float32).to_elements()
            # --- the shipped body's row maths, verbatim (qk_norm_rope_tma_body; kept inline so that body's cubin stays byte-identical) ---
            ys = list(xs)
            if cutlass.const_expr(apply_norm):
                acc = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    acc = acc + xs[i] * xs[i]
                rstd = cute.math.rsqrt(lane_group_sum(acc, WARP) * cutlass.Float32(1.0 / d) + eps, fastmath=True)
                w_addr = (mWq.iterator.toint() if is_q else mWk.iterator.toint()) + (
                    sub.to(cutlass.Int64) * cutlass.Int64(TMA_GRANU_ELEMS) + col.to(cutlass.Int64) * cutlass.Int64(ELEMS_PER_ACCESS)
                ) * cutlass.Int64(BPE)
                ws = [v for pair in [f16x2_to_f32(w, dtype=mWq.element_type) for w in ld_global_v4(w_addr, cutlass.Int32)] for v in pair]
                ys = [xs[i] * rstd * ws[i] for i in range(ELEMS_PER_ACCESS)]
            if cutlass.const_expr(rope_dim > 0):
                in_rope = lane < cutlass.Int32(rope_lanes)
                cs = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
                sn = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
                if in_rope:
                    rope_off = lane.to(cutlass.Int64) * cutlass.Int64(ELEMS_PER_ACCESS * BPE)
                    cb = mCos.iterator.toint() + tok.to(cutlass.Int64) * cutlass.Int64(mCos.stride[0]) * cutlass.Int64(BPE) + rope_off
                    sb = mSin.iterator.toint() + tok.to(cutlass.Int64) * cutlass.Int64(mSin.stride[0]) * cutlass.Int64(BPE) + rope_off
                    cs = [v for pair in [f16x2_to_f32(w, dtype=mCos.element_type) for w in ld_global_v4(cb, cutlass.Int32)] for v in pair]
                    sn = [v for pair in [f16x2_to_f32(w, dtype=mSin.element_type) for w in ld_global_v4(sb, cutlass.Int32)] for v in pair]
                rot = []
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    partner = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, ys[i], cutlass.Int32(half_lanes), 31, kind=nvvm.Shfl.BFLY))
                    signed = -partner if lane < cutlass.Int32(half_lanes) else partner
                    rot.append(ys[i] * cs[i] + signed * sn[i])
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    ys[i] = rot[i] if in_rope else ys[i]
            # --- end of the shipped row maths ---

            # the bf16 ROUNDING FIRST (the words the bf16 arm hands TMA), widened back: the values every quantize below sees
            words16 = [fp32_to_fp16(ys[2 * i], ys[2 * i + 1], dtype=io_dtype) for i in range(ELEMS_PER_ACCESS // 2)]
            y16 = []
            for w in words16:
                lo, hi = f16x2_to_f32(w, dtype=io_dtype)
                y16.append(lo if valid else zero)  # a padded row: zero in both amaxes, SF 0x00, no payload store
                y16.append(hi if valid else zero)
            # (i) the ROWWISE block quantize: lanes 4k..4k+3 of a subtile share one 32-element block along D
            amax = abs_max_tree(y16)
            amax = fmax_f32(amax, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax, cutlass.Int32(1), 31, kind=nvvm.Shfl.BFLY)))
            amax = fmax_f32(amax, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax, cutlass.Int32(2), 31, kind=nvvm.Shfl.BFLY)))
            rcp, sf_byte = e8m0_from_amax(amax)
            halves = []
            for i in cutlass.range_constexpr(ELEMS_PER_ACCESS // 2):
                halves.append(fp32_to_fp8x2(y16[2 * i] * rcp, y16[2 * i + 1] * rcp))
            packed8 = [pack_u16x2(halves[0], halves[1]), pack_u16x2(halves[2], halves[3])]
            if valid:
                st_global_v2(pay_base + tok.to(cutlass.Int64) * pay_tok_stride, packed8, cutlass.Int32)
            # the 4 blocks of this (row, subtile) are 4 contiguous SF bytes: gather them from the block leaders (lanes col 0/4/8/12 of the
            # subtile hold bytes c%4 = 0..3) into ONE 4-byte store by the subtile's first lane -- unpredicated (a pad row stores 0x00).
            # Its byte is the atom rule (r_atom % 32) * SF_ATOM_LINE_BYTES + (r_atom // 32) * SF_ATOM_COLS + c with r_atom = tb * 32 + r: the
            # tile row r is the line, the 32-token block tb within the 128-row SF tile the 4-byte quarter
            b1 = cutlass.Int32(nvvm.shfl_sync(0xFFFFFFFF, sf_byte, lane + cutlass.Int32(4), 31, kind=nvvm.Shfl.IDX))
            b2 = cutlass.Int32(nvvm.shfl_sync(0xFFFFFFFF, sf_byte, lane + cutlass.Int32(8), 31, kind=nvvm.Shfl.IDX))
            b3 = cutlass.Int32(nvvm.shfl_sync(0xFFFFFFFF, sf_byte, lane + cutlass.Int32(12), 31, kind=nvvm.Shfl.IDX))
            sf_word = sf_byte | (b1 << 8) | (b2 << 16) | (b3 << 24)
            if col_leader:
                st_global(
                    sf_row_base
                    + sub.to(cutlass.Int64) * cutlass.Int64(SF_ATOM_BYTES)
                    + (r * cutlass.Int32(SF_ATOM_LINE_BYTES) + tb * cutlass.Int32(SF_ATOM_COLS)).to(cutlass.Int64),
                    sf_word,
                    cutlass.Int32,
                )
            # (iii) the running column maxima of each lane's 8 d
            for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                colmax[i] = fmax_f32(colmax[i], cute.math.abs(y16[i]))
            # (ii) the bf16 words back into the input stage, in place: pass 2 reads them; each lane overwrites exactly the 16 B it read
            sIn_raw.store(cutlass.Vector.from_elements(tuple(y16), cutlass.Float32).to(io_dtype), off_in, vector_size=ELEMS_PER_ACCESS, alignment=16)

        # the warp's column maxima -> sRedCol[warp][d0 .. d0 + 8) (a warp's 32 lanes cover d contiguous fp32 = conflict-free)
        red_off = warp_id * cutlass.Int32(d) + d0
        sRedCol.store(cutlass.Vector.from_elements(tuple(colmax[:fp32_per_access]), cutlass.Float32), red_off, vector_size=fp32_per_access, alignment=16)
        sRedCol.store(
            cutlass.Vector.from_elements(tuple(colmax[fp32_per_access:]), cutlass.Float32),
            red_off + cutlass.Int32(fp32_per_access),
            vector_size=fp32_per_access,
            alignment=16,
        )
        nvvm.barrier_cta_sync()  # bar #1: every warp's maxima are in sRedCol

        # --- the column scales: the warps combined per d (fixed order; max is exact), one e8m0 per d ---------------
        cm = list(colmax)
        for w in cutlass.range_constexpr(warps):
            if w != warp_id:
                v0 = sRedCol.load(cutlass.Int32(w * d) + d0, vector_size=fp32_per_access, alignment=16)
                v1 = sRedCol.load(cutlass.Int32(w * d) + d0 + cutlass.Int32(fp32_per_access), vector_size=fp32_per_access, alignment=16)
                for i in cutlass.range_constexpr(fp32_per_access):
                    cm[i] = fmax_f32(cm[i], v0[i])
                    cm[fp32_per_access + i] = fmax_f32(cm[fp32_per_access + i], v1[i])
        rcps = []
        bytes_c = []
        for i in cutlass.range_constexpr(ELEMS_PER_ACCESS // 2):
            rcp0, rcp1, packed_e = e8m0_pair(cm[2 * i], cm[2 * i + 1])
            rcps.append(rcp0)
            rcps.append(rcp1)
            bytes_c.append(packed_e & cutlass.Int32(0xFF))
            bytes_c.append((packed_e >> 8) & cutlass.Int32(0xFF))
        if warp_id == 0:
            # the columnwise SF bytes of this tile's 32-token block: one per d, by warp 0 (8 single-byte stores per lane, 16 B apart)
            base_c = sf_col_base + sub.to(cutlass.Int64) * plane_stride + (col_sf_lane + tb).to(cutlass.Int64)
            for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                _st_global_b8(base_c + cutlass.Int64(i * SF_ATOM_LINE_BYTES), bytes_c[i])

        # --- pass 2: the COLUMNWISE payload from the bf16 words written back in pass 1 ---------------------------------
        for u in cutlass.range_constexpr(rows_per_warp):
            r = warp_id + cutlass.Int32(u * warps)
            s_row = s0 + r
            valid = s_row < seq_len
            tok = tok_base + (s_row if valid else seq_len - cutlass.Int32(1))
            off_in = (
                s * cutlass.Int32(tile_elems) + sub * cutlass.Int32(subtile_elems) + r * cutlass.Int32(TMA_GRANU_ELEMS) + col * cutlass.Int32(ELEMS_PER_ACCESS)
            )
            x16 = sIn_raw.load(off_in, vector_size=ELEMS_PER_ACCESS, alignment=16).to(cutlass.Float32).to_elements()
            halves_t = []
            for i in cutlass.range_constexpr(ELEMS_PER_ACCESS // 2):
                halves_t.append(fp32_to_fp8x2(x16[2 * i] * rcps[2 * i], x16[2 * i + 1] * rcps[2 * i + 1]))
            packed8_t = [pack_u16x2(halves_t[0], halves_t[1]), pack_u16x2(halves_t[2], halves_t[3])]
            if valid:
                st_global_v2(payt_base + tok.to(cutlass.Int64) * payt_tok_stride, packed8_t, cutlass.Int32)

        # pass 1 wrote sIn[s] through the generic proxy; the refill overwrites it through the async proxy: the WRITERS fence, then the
        # barrier publishes, then the elected lane issues (the shipped order of the bf16 arm's store path)
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.barrier_cta_sync()  # bar #2: sIn[s] is free (read AND written by nobody from here to the refill)
        nxt = tile + n_ctas * cutlass.Int32(stages)
        if nxt < n_tiles and warp_id == 0:
            _issue_token_tile_load(
                sIn_raw,
                s * cutlass.Int32(tile_elems),
                mb_full.subview(s),
                nxt,
                n_blk,
                tma_q,
                tma_k,
                tile_elems,
                subtiles,
                subtile_elems,
                tile_bytes,
                h_q_ct,
                h_kv_ct,
            )
        tile = tile + n_ctas
    # Every arrive above is LOCAL (no cross-CTA drain, P15 does not apply at cga1) and this arm issues no bulk store.


# The MX token-tile arm's RESIDENCY per SM at 128 threads -- the persistent grid's cap, derived from the two facts that bind it and
# MEASURED (Rubin cc 10.7, 212 SMs, locked clocks, CUPTI device time, the 397B geometry at S = 8K, every cap bitwise the cap-8
# bytes): registers REG 90 (norm) / 72 (RoPE-only) on the fused prologue, 96 / 80 standalone (sm_107a cubins; allocation rounds to
# 8) give 65536 / (128 x 96) = 5 and 65536 / (128 x 72..80) = 6..7 CTAs; the static SMEM (the 2-stage 32-token input ring 32 KiB +
# sRedCol 4 KiB + the fused prologue's 1 KiB SF tile + barriers, plus the per-CTA reserve) fits 6 -- the RoPE-only arm's cap sweep
# peaks at 6, the norm arm's at 5 -- so the residency is min(registers, SMEM) = 5 (norm) / 6 (RoPE-only).  A cap above the residency
# runs a grid-stride TAIL (the amax job's SMs x 8 launched 1.6 waves: -8 % on the norm arm, -12 % on the RoPE-only arm vs the right
# cap at S = 8K); a cap below it idles SMs.  A kernel edit that moves REG across an allocation boundary or the SMEM footprint past
# 228 KiB / 6 must re-derive these -- the sm_107a trace-compile pin reads REG off the cubin and checks the register cap against this
# table (test_mxfp8_bwd_fused.py), the cap sweep is the runtime detector.
MX_REBUILD_THREADS = 128
MX_REBUILD_SMEM_CTAS_PER_SM = 6  # the SMEM-bound residency of the MX arm (measured: the RoPE-only cap sweep peaks at 6)
MX_REBUILD_CTAS_PER_SM = {True: 5, False: 6}  # apply_norm -> min(register cap, SMEM cap)


def register_cap_ctas_per_sm(regs: int, threads: int = MX_REBUILD_THREADS, regs_per_sm: int = 65536, alloc_granule: int = 8) -> int:
    """CTAs per SM the register file allows for ``regs`` registers per thread at ``threads`` per CTA (per-thread allocation rounded up
    to ``alloc_granule``): ``65536 // (threads x ceil8(regs))``."""
    alloc = ((int(regs) + alloc_granule - 1) // alloc_granule) * alloc_granule
    return max(1, regs_per_sm // (int(threads) * alloc))


def mx_rebuild_ctas_per_sm(apply_norm: bool) -> int:
    """The MX token-tile arm's resident CTAs per SM (the table above): the persistent cap is ``SMs x`` this, never the amax job's."""
    return MX_REBUILD_CTAS_PER_SM[bool(apply_norm)]


def mx_tile_counts(batch: int, seq_len: int, h_q: int, h_kv: int) -> tuple:
    """``(n_blk, n_sf_tiles, n_tiles)`` of the MX arm: the 32-token ranges per batch over the PADDED sequence ``ceil128(S) / 32``,
    the SF tiles per (b, h) ``ceil(S / 128)``, and the token tiles ``B * n_blk * (h_q + h_kv)`` -- the ``tile_counts`` twin.  Every
    range of the last 128-token unit is a tile, past ``S`` included (its load is zero-filled, its SF bytes ``0x00``, no payload)."""
    n_sft = n_sf_tiles(int(seq_len))
    n_blk = MX_BLOCKS_PER_SF_TILE * n_sft
    return n_blk, n_sft, int(batch) * n_blk * (int(h_q) + int(h_kv))


@cute.kernel
def frost_qk_norm_rope_tma(
    mQ: cute.Tensor,  # [T, H_q, D] -- bound for its element_type only; Q/K data
    mWq: Optional[cute.Tensor],  # [D]  moves entirely through the descriptors
    mWk: Optional[cute.Tensor],  # [D]  None (both): RoPE-only -- no RMSNorm, no weight loads, no rstd
    mCos: cute.Tensor,  # [T, ROPE_DIM]
    mSin: cute.Tensor,  # [T, ROPE_DIM]
    mRstdQ: Optional[cute.Tensor],  # [T, H_q]  fp32, or None
    mRstdK: Optional[cute.Tensor],  # [T, H_kv] fp32, or None
    mQ8: Optional[cute.Tensor],  # [T, H_q, D] e4m3 OUT -- the e4m3 epilogue (module docstring); None = the bf16 TMA store
    mK8: Optional[cute.Tensor],  # [T, H_kv, D] e4m3 OUT
    mScaleQ: Optional[cute.Tensor],  # [1] fp32 static scale_q (e4m3 epilogue)
    mScaleK: Optional[cute.Tensor],  # [1] fp32 static scale_k
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_qo_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_ko_desc: cutlass.GridConstant[tmap.TensorMap],
    n_tokens: cutlass.Int32,
    n_q_tiles: cutlass.Int32,
    n_tiles: cutlass.Int32,
    n_ctas: cutlass.Int32,
    eps: cutlass.Float32,
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    tile_rows: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    stages_o: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    refill_pos: cutlass.Constexpr[int],
    fused_store_wait: cutlass.Constexpr[bool],
    # -- APPENDED: the MX epilogue (qk_norm_rope_tma_mx_body); mSfQ bound = the MX arm, in which mQ8 / mK8 are its rowwise payloads --
    mSfQ: Optional[cute.Tensor],  # the SDPA rowwise SF tiles of q8
    mQT8: Optional[cute.Tensor],  # [T, H_q, D] e4m3: the columnwise payload
    mSfQT: Optional[cute.Tensor],  # the SDPA columnwise (D-plane-major) SF atoms of q_T8
    mSfK: Optional[cute.Tensor],
    mKT8: Optional[cute.Tensor],
    mSfKT: Optional[cute.Tensor],
    seq_len: cutlass.Int32,  # S, B, ceil128(S)/32, ceil(S/128) -- the MX arm's tile algebra (0 under the other arms)
    batch: cutlass.Int32,
    n_blk: cutlass.Int32,
    n_sft: cutlass.Int32,
    # APPENDED: the pool-write arm (module docstring; mPoolK bound = the arm): the two pools, the slot map, the V band's load descriptor
    # (a placeholder -- K's -- under the other arms, never touched), the slot capacity and the two compile keys of the paged write.
    mPoolK: Optional[cute.Tensor],
    mPoolV: Optional[cute.Tensor],
    mSlot: Optional[cute.Tensor],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    n_slots: cutlass.Int64,
    page_size: cutlass.Constexpr[int],
    slot_i64: cutlass.Constexpr[bool],
    # APPENDED: programmatic dependent launch (the launch attribute is set by qk_norm_rope_tma_launch under the same flag): the wait
    # precedes the bodies' ring fill (their first global read: the slab bands a projection launch wrote) and the pool-write arm's slot
    # loads, the release follows it.
    use_pdl: cutlass.Constexpr[bool] = False,
) -> None:
    """The standalone launch shape of :func:`qk_norm_rope_tma_body` (the SMEM rings allocated here -- the SMEM buffer table --,
    ``cta = blockIdx.x`` over a grid of ``n_ctas``) and, with the MX operands bound, of :func:`qk_norm_rope_tma_mx_body` (the
    32-token input ring + ``sRedCol``; ``n_tiles`` is then the MX tile count, ``n_q_tiles`` / ``n_tokens`` unused)."""
    mx_out = cutlass.const_expr(mSfQ is not None)
    tile_elems = cutlass.const_expr((MX_TILE_TOKENS if mx_out else tile_rows) * d)
    apply_norm = cutlass.const_expr(mWq is not None)
    want_rstd = cutlass.const_expr(apply_norm and mRstdQ is not None)
    fp8_out = cutlass.const_expr(mQ8 is not None and not mx_out)
    pool_write = cutlass.const_expr(mPoolK is not None)  # the pool-write arm (bf16 / f16 TMA store only)
    io_dtype = mQ.element_type  # a trace-time type object, NOT a const_expr candidate
    sIn_raw = cutlass.Array(io_dtype, stages * tile_elems, alignment=128, space=cutlass.AddressSpace.smem)
    # The output ring exists only for the bf16 TMA store: the e4m3 and the MX epilogues store straight to GMEM.
    sOut_raw = (
        cutlass.Array(io_dtype, stages_o * tile_elems, alignment=128, space=cutlass.AddressSpace.smem)
        if cutlass.const_expr(not fp8_out and not mx_out)
        else None
    )
    # Allocated only when the rstd store exists: both uses sit under
    # const_expr(want_rstd), so the RoPE-only / no-rstd trace keeps the SMEM too.
    sRstd = cutlass.Array(cutlass.Float32, stages * tile_rows, alignment=16, space=cutlass.AddressSpace.smem) if cutlass.const_expr(want_rstd) else None
    mb_full = cutlass.Array(cutlass.Int64, stages, alignment=16, space=cutlass.AddressSpace.smem)
    if cutlass.const_expr(use_pdl):
        # Every global read of both arms (the TMA ring fill, the weights, the RoPE tables, the scale slots) sits below this wait; the
        # SMEM carve above it is the prologue it overlaps.  The dependent grid is released right away: a streaming pass has nothing
        # a later release would protect (its wait still blocks on this grid's completion and memory flush).
        wait_on_dependent_grids()
        launch_dependent_grids()
    if cutlass.const_expr(mx_out):
        # the MX arm: the per-warp column maxima (warps x D fp32, the SMEM table's sRedCol); the descriptors are the 4-D band views
        sRedCol = cutlass.Array(cutlass.Float32, (threads_per_cta // WARP) * d, alignment=16, space=cutlass.AddressSpace.smem)
        qk_norm_rope_tma_mx_body(
            mQ,
            mWq,
            mWk,
            mCos,
            mSin,
            GmemTileTma(tma_q_desc),
            GmemTileTma(tma_k_desc),
            mQ8,
            mSfQ,
            mQT8,
            mSfQT,
            mK8,
            mSfK,
            mKT8,
            mSfKT,
            seq_len,
            batch,
            n_blk,
            n_sft,
            n_tiles,
            cutlass.Int32(cute.arch.block_idx()[0]),
            n_ctas,
            eps,
            d,
            rope_dim,
            stages,
            threads_per_cta,
            h_q_ct,
            h_kv_ct,
            sIn_raw,
            sRedCol,
            mb_full,
        )
    else:
        # The static per-tensor scales of the e4m3 epilogue, once per thread from their slots (no host readback); 1.0, unused, otherwise.
        scale_q = cutlass.Float32(cutlass.make_array_view(mScaleQ)[0]) if cutlass.const_expr(fp8_out) else cutlass.Float32(1.0)
        scale_k = cutlass.Float32(cutlass.make_array_view(mScaleK)[0]) if cutlass.const_expr(fp8_out) else cutlass.Float32(1.0)
        qk_norm_rope_tma_body(
            mQ,
            mWq,
            mWk,
            mCos,
            mSin,
            mRstdQ,
            mRstdK,
            GmemTileTma(tma_q_desc),
            GmemTileTma(tma_k_desc),
            GmemTileTma(tma_qo_desc),
            GmemTileTma(tma_ko_desc),
            mQ8,
            mK8,
            scale_q,
            scale_k,
            n_tokens,
            n_q_tiles,
            n_tiles,
            cutlass.Int32(cute.arch.block_idx()[0]),
            n_ctas,
            eps,
            d,
            rope_dim,
            tile_rows,
            stages,
            stages_o,
            threads_per_cta,
            h_q_ct,
            h_kv_ct,
            refill_pos,
            fused_store_wait,
            sIn_raw,
            sOut_raw,
            sRstd,
            mb_full,
            GmemTileTma(tma_v_desc) if cutlass.const_expr(pool_write) else None,
            mPoolK,
            mPoolV,
            mSlot,
            n_slots,
            page_size,
            slot_i64,
        )


@cute.jit
def qk_norm_rope_tma_launch(
    q: cute.Tensor,
    k: cute.Tensor,
    q_out: cute.Tensor,
    k_out: cute.Tensor,
    w_q: Optional[cute.Tensor],
    w_k: Optional[cute.Tensor],
    cos: cute.Tensor,
    sin: cute.Tensor,
    rstd_q: Optional[cute.Tensor],
    rstd_k: Optional[cute.Tensor],
    q8: Optional[cute.Tensor],
    k8: Optional[cute.Tensor],
    scale_q: Optional[cute.Tensor],
    scale_k: Optional[cute.Tensor],
    n_tokens: cutlass.Int32,
    n_q_tiles: cutlass.Int32,
    n_tiles: cutlass.Int32,
    n_ctas: cutlass.Int32,
    eps: cutlass.Float32,
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    tile_rows: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    stages_o: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    refill_pos: cutlass.Constexpr[int],
    fused_store_wait: cutlass.Constexpr[bool],
    # -- APPENDED: the MX epilogue's operands (sf_q bound = the MX arm; q / k are then the 4-D (B, S, H, D) views of the bands) --
    sf_q: Optional[cute.Tensor],
    q_T8: Optional[cute.Tensor],
    sf_q_T: Optional[cute.Tensor],
    sf_k: Optional[cute.Tensor],
    k_T8: Optional[cute.Tensor],
    sf_k_T: Optional[cute.Tensor],
    seq_len: cutlass.Int32,
    batch: cutlass.Int32,
    n_blk: cutlass.Int32,
    n_sft: cutlass.Int32,
    # -- APPENDED: the pool-write arm's operands (pool_k bound = the arm): the V band [T, H_kv, D] (its own load descriptor, the K box),
    # the two pools [P, H_kv, page_size, D], the slot map [T], the slot capacity, and the page size / slot width (compile keys) --
    v: Optional[cute.Tensor],
    pool_k: Optional[cute.Tensor],
    pool_v: Optional[cute.Tensor],
    slot: Optional[cute.Tensor],
    n_slots: cutlass.Int64,
    page_size: cutlass.Constexpr[int],
    slot_i64: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    use_pdl: cutlass.Constexpr[bool] = False,  # APPENDED: the programmatic-dependent-launch form (attribute + the kernel's griddepcontrol pair)
):
    """Build the four descriptors, then launch.

    Box shapes, innermost-LAST to match the tensor's ``[T, H, D]`` order:

    * **Q**: ``(1, tile_rows, GRANU)`` -- one token, ``tile_rows`` consecutive
      heads. Legal because ``tile_rows`` divides ``h_q``.
    * **K**: ``(tile_rows // h_kv, h_kv, GRANU)`` -- whole tokens of ``h_kv``
      heads, because ``h_kv`` is small (2 at the 397B geometry) and a tile of
      ``tile_rows`` therefore spans several tokens.

    ``swizzle=none`` caps the innermost box at 256 B, which is why ``d=256``
    bf16 needs ``d // GRANU`` TMA calls per tile. Unswizzled is not a
    compromise here: see the SMEM buffer table at the top -- warp-per-row is
    already conflict-free, and TMA's 128 B swizzle is built for the opposite
    access pattern.

    The per-batch token stride rides in the descriptor, so a strided source
    (Q and K as column slices of the fused projection) costs the KERNEL nothing.
    That is the address-math tax the LDG kernel pays in registers.

    Under the e4m3 epilogue (``q8`` bound) the OUTPUT descriptors are placeholders the
    kernel never touches: the host hands ``q`` / ``k`` themselves as ``q_out`` / ``k_out``.

    Under the MX epilogue (``sf_q`` bound) ``q`` / ``k`` are the 4-D ``(B, S, H, D)`` views of the bands (token
    stride ``N``, batch stride ``S * N``) and BOTH boxes are ``(1, 32, 1, GRANU)`` -- one head x 32 tokens,
    coordinates ``(d, h, s, b)`` -- so TMA clips every tile at ``S`` (a block never straddles a batch) and
    zero-fills the ragged tail; the output descriptors are placeholders as under the e4m3 epilogue.

    Under the pool-write arm (``pool_k`` bound) a FIFTH load descriptor covers the V band with the K box (``v`` is the ``[T, H_kv, D]``
    band at the slab's token stride); under the other arms that slot carries the K descriptor again, a placeholder the kernel never touches.
    """
    mx_out = cutlass.const_expr(sf_q is not None)
    pool_write = cutlass.const_expr(pool_k is not None)
    if cutlass.const_expr(mx_out):
        box_q = (1, MX_TILE_TOKENS, 1, TMA_GRANU_ELEMS)
        box_k = (1, MX_TILE_TOKENS, 1, TMA_GRANU_ELEMS)
        order = (3, 2, 1, 0)
    else:
        box_q = (1, tile_rows, TMA_GRANU_ELEMS)
        box_k = (tile_rows // h_kv_ct, h_kv_ct, TMA_GRANU_ELEMS)
        order = (2, 1, 0)
    mk = lambda t, box: tmap.create_tensor_map_tiled_from_view(
        t,
        box_dims=box,
        stride_order=order,
        swizzle=tmap.TensorMapSwizzle.none,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    frost_qk_norm_rope_tma(
        q,
        w_q,
        w_k,
        cos,
        sin,
        rstd_q,
        rstd_k,
        q8,
        k8,
        scale_q,
        scale_k,
        mk(q, box_q),
        mk(k, box_k),
        mk(q_out, box_q),
        mk(k_out, box_k),
        n_tokens,
        n_q_tiles,
        n_tiles,
        n_ctas,
        eps,
        d,
        rope_dim,
        tile_rows,
        stages,
        stages_o,
        threads_per_cta,
        h_q_ct,
        h_kv_ct,
        refill_pos,
        fused_store_wait,
        sf_q,
        q_T8,
        sf_q_T,
        sf_k,
        k_T8,
        sf_k_T,
        seq_len,
        batch,
        n_blk,
        n_sft,
        pool_k,
        pool_v,
        slot,
        mk(v if cutlass.const_expr(pool_write) else k, box_k),
        n_slots,
        page_size,
        slot_i64,
        use_pdl,
    ).launch(grid=(n_ctas, 1, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl)


compiled_cache = {}


class QkNormRopeTmaRecipe(NamedTuple):
    """Build-time facts of one TMA norm+RoPE launch (a legal compile key, like
    ``QkNormRopeRecipe``). ``apply_norm`` is appended with a default so every
    existing recipe reads the same; ``run_qk_norm_rope_tma`` checks it against
    the bound weights in both directions."""

    compiled: object
    h_q: int
    h_kv: int
    d: int
    eps: float
    tile_rows: int
    stages: int
    stages_o: int
    threads: int
    want_rstd: bool
    ctas_per_sm: int
    apply_norm: bool = True
    # Appended (default = today's artifact): the e4m3 epilogue traced (``q8`` / ``k8`` / ``scale_q`` / ``scale_k`` REQUIRED at
    # execute, ``q_out`` / ``k_out`` refused), ``run_qk_norm_rope_tma`` checks it both ways.
    fp8_out: bool = False
    # Appended (default = today's artifact): the MX epilogue traced (``qk_norm_rope_tma_mx_body``: the 32-token tile, ``q8 / sf_q /
    # q_T8 / sf_q_T / k8 / sf_k / k_T8 / sf_k_T`` + ``batch`` / ``seq_len`` REQUIRED at execute, the bf16 outputs and the per-tensor
    # scales refused); ``tile_rows`` / ``stages_o`` of the recipe are not used by this arm.
    mx_out: bool = False
    # Appended (default = today's artifact): the programmatic-dependent-launch form traced (the launch attribute + the griddepcontrol pair).
    use_pdl: bool = False
    # Appended (default = today's artifact): the pool-write arm traced (module docstring): ``v`` (the V band), ``k_cache`` / ``v_cache``
    # and ``slot_mapping`` REQUIRED at execute (refused otherwise); ``page_size`` and ``slot_dtype`` are its compile keys.
    pool_write: bool = False
    page_size: int = 0
    slot_dtype: object = None


def _fake(dtype, shape, stride_order):
    return cute.runtime.make_fake_compact_tensor(
        dtype=_convert_to_cutlass_data_type(dtype),
        shape=shape,
        stride_order=stride_order,
        assumed_align=16,
    )


def _fake_thd(dtype, tok, h: int, d: int):
    """``[T, H, D]`` with a SYMBOLIC token stride, so one artifact serves both
    the fused-projection slice (stride ``N``) and a compact buffer (stride
    ``h*d``). Unlike the LDG kernel this costs the kernel nothing: the stride
    reaches the hardware through the descriptor, not through address math."""
    return cute.runtime.make_fake_tensor(
        dtype=_convert_to_cutlass_data_type(dtype),
        shape=(tok, h, d),
        stride=(cute.sym_int(), d, 1),
        assumed_align=16,
    )


def _fake_e4m3_thd(tok, h: int, d: int):
    """The e4m3 epilogue's destination: ``[T, H, D]`` ``float8_e4m3fn`` with a symbolic token stride (compact in the block), 16-B rows."""
    return cute.runtime.make_fake_tensor(dtype=cutlass.Float8E4M3FN, shape=(tok, h, d), stride=(cute.sym_int(), d, 1), assumed_align=16)


def _fake_slot():
    """A 1-element fp32 scalar slot at 4-byte alignment (the quantize kernels' slot contract)."""
    return cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), stride_order=(0,), assumed_align=4)


def tile_counts(t: int, h_q: int, h_kv: int, tile_rows: int, pool_write: bool = False) -> tuple[int, int]:
    """(n_q_tiles, n_tiles). A Q tile is ``tile_rows`` heads of one token; a K
    tile is ``tile_rows // h_kv`` whole tokens, so its LAST tile may overshoot
    ``T`` -- TMA clips the store against the descriptor extent and the rstd
    write is predicated.  ``pool_write`` (appended): the V band's tiles follow
    the K band's, as many again (the kernel reads the K / V boundary off that)."""
    n_q = t * (h_q // tile_rows)
    toks_per_ktile = tile_rows // h_kv
    n_k = (t + toks_per_ktile - 1) // toks_per_ktile
    return n_q, n_q + n_k + (n_k if pool_write else 0)


def compile_qk_norm_rope_tma(
    *,
    dtype,
    h_q: int,
    h_kv: int,
    d: int,
    rope_dim: int,
    eps: float,
    want_rstd: bool,
    tile_rows: int = DEFAULT_TILE_ROWS,
    stages: int = DEFAULT_STAGES,
    stages_o: Optional[int] = None,  # None = DEFAULT_STAGES_O
    threads_per_cta: int = DEFAULT_THREADS,
    refill_pos: int = DEFAULT_REFILL_POS,
    fused_store_wait: bool = DEFAULT_FUSED_STORE_WAIT,
    ctas_per_sm: int = 8,
    apply_norm: bool = True,
    fp8_out: bool = False,
    mx_out: bool = False,
    use_pdl: bool = False,
    pool_write: bool = False,
    page_size: int = 0,
    slot_dtype=torch.int32,
) -> QkNormRopeTmaRecipe:
    """Build the TMA artifact from shapes alone. ``apply_norm=False`` traces the
    RoPE-only kernel (weights traced as ``None``, no rstd allowed) and is part
    of the cache key so a norm-on artifact is never reused with ``None`` weights.
    ``fp8_out`` (appended) traces the e4m3 epilogue (module docstring): ``q8`` /
    ``k8`` and the static ``scale_q`` / ``scale_k`` slots become REQUIRED at execute
    and the bf16 outputs do not exist; needs the fp8 ``cvt`` (sm_89+, declined by
    name -- Rule 7).  In the key.  ``mx_out`` (appended) traces the MX epilogue
    (``qk_norm_rope_tma_mx_body``: the 32-token tile, the rowwise AND columnwise
    MXFP8 quantizes of Q and K): ``q8 / sf_q / q_T8 / sf_q_T / k8 / sf_k / k_T8 /
    sf_k_T`` + ``batch`` / ``seq_len`` REQUIRED at execute, the bf16 outputs, the
    per-tensor scales and the rstd refused; needs the fp8 ``cvt`` and the e8m0 one
    (sm_100+); exclusive with ``fp8_out``; ``tile_rows`` is validated for the
    shipped arms only (the MX tile is 32 tokens whatever it says).  In the key.
    ``pool_write`` (appended) traces the pool-write arm (module docstring): the V
    band, ``k_cache`` / ``v_cache`` (``[P, H_kv, page_size, D]``, D contiguous) and
    ``slot_mapping`` (``[T]`` of ``slot_dtype``) become REQUIRED at execute; ``page_size``
    (the in-kernel divide) and ``slot_dtype`` are compile keys; exclusive with the two
    quantizing epilogues.  In the key."""
    if not isinstance(mx_out, bool):
        raise ValueError(f"mx_out must be a bool (whether the MX epilogue is traced), got {mx_out!r}")
    if not isinstance(fp8_out, bool):
        raise ValueError(f"fp8_out must be a bool (whether the e4m3 epilogue is traced), got {fp8_out!r}")
    if mx_out and fp8_out:
        raise ValueError("fp8_out and mx_out are two epilogues of one tile; trace one of them (the per-tensor e4m3 cast or the MXFP8 block quantizes)")
    if mx_out:
        validate_mx_shape(d, rope_dim, threads_per_cta)
        if want_rstd:
            raise ValueError("mx_out=True (the MXFP8 backward's rebuild) emits no rstd (the record carries the forward's); want_rstd must be False")
        if refill_pos != REFILL_AFTER_COMPUTE:
            raise ValueError(f"mx_out=True refills after its second barrier only (REFILL_AFTER_COMPUTE = {REFILL_AFTER_COMPUTE}); got refill_pos={refill_pos}")
        require_fp8_cvt("compile_qk_norm_rope_tma(mx_out=True)")
    else:
        validate_shape(d, rope_dim, h_q, h_kv, tile_rows, threads_per_cta)
    if fp8_out:
        require_fp8_cvt("compile_qk_norm_rope_tma(fp8_out=True)")
    if not isinstance(pool_write, bool):
        raise ValueError(f"pool_write must be a bool (whether the paged K / V write is traced into the launch), got {pool_write!r}")
    if pool_write:
        if fp8_out or mx_out:
            raise ValueError(
                "pool_write=True (the paged K / V write from the norm + RoPE launch) is an arm of the bf16 / f16 TMA store; it is exclusive with fp8_out / mx_out"
            )
        validate_cache_write_shape(d, int(page_size), threads_per_cta)
        if slot_dtype not in _SLOT_DTYPES:
            raise ValueError(f"pool_write=True: slot_dtype must be torch.int32 or torch.int64 (a compile key), got {slot_dtype}")
    else:
        page_size, slot_dtype = 0, None  # not traced: one key for every shipped-arm artifact whatever the caller spelled
    # None means the DEFAULT, not "match stages" -- otherwise changing
    # DEFAULT_STAGES_O silently does nothing for every caller that omits it,
    # which is exactly what happened when it moved 2 -> 1.
    stages_o = int(DEFAULT_STAGES_O if stages_o is None else stages_o)
    if stages_o < 1:
        raise ValueError(f"stages_o must be >= 1, got {stages_o}")
    if fused_store_wait and stages_o < 2:
        # The fused drain waits AFTER this tile's lanes wrote sOut, so it can only
        # protect the NEXT tile's buffer when there is a second one: at stages_o=1
        # the lanes of tile i+1 overwrite sOut[0] under tile i's in-flight bulk
        # store (see DEFAULT_FUSED_STORE_WAIT) -- silent wrong Q/K rows, no error.
        raise ValueError(f"fused_store_wait=True needs stages_o >= 2 (the drain protects the next tile's output stage), got stages_o={stages_o}")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"qk_norm_rope_tma serves bf16/f16 only, got {dtype}")
    if not apply_norm and rope_dim == 0:
        raise ValueError("apply_norm=False with rope_dim=0 is an identity copy of Q/K; drop the stage instead of launching it")
    if want_rstd and not apply_norm:
        raise ValueError("apply_norm=False (RoPE-only Q/K) computes no RMSNorm and therefore emits no rstd; want_rstd must be False")
    # Every Constexpr handed to cute.compile below is in the key. `stages_o`
    # was missing (PR #1102 review): it sizes sOut_raw and bounds the store
    # drain, so the second depth requested in a process was served the FIRST
    # one's artifact under a recipe that reported the second.
    key = (
        str(dtype),
        h_q,
        h_kv,
        d,
        int(rope_dim),
        int(tile_rows),
        int(stages),
        int(stages_o),
        int(threads_per_cta),
        bool(want_rstd),
        int(refill_pos),
        bool(fused_store_wait),
        current_device(),
        bool(apply_norm),
        bool(fp8_out),
        bool(mx_out),
        bool(use_pdl),
        bool(pool_write),
        int(page_size),
        str(slot_dtype),
    )
    if key not in compiled_cache:
        # The dict stays the in-process memo; behind it sits the persistent compiled-plan cache (``cudnn.frost.compiled_cache``),
        # keyed by this file's digest + the dict key's plain values: a second process reloads the object instead of re-tracing.
        cache_key = template_key(globals(), {"key": key}, "compile_qk_norm_rope_tma")
        tok = cute.sym_int()
        if mx_out:
            # the MX arm reads the bands through 4-D (B, S, H, D) views (symbolic batch / token strides: a slab band or a compact
            # buffer); the "output" slots are placeholders the kernel never touches (the same views)
            dense = [_fake_bshd(dtype, h, d) for h in (h_q, h_kv, h_q, h_kv)]
        else:
            dense = [_fake_thd(dtype, tok, h, d) for h in (h_q, h_kv, h_q, h_kv)]
        weights = [_fake(dtype, (d,), (0,)) for _ in range(2)] if apply_norm else [None, None]
        tables = [_fake(dtype, (tok, rope_dim if rope_dim else 1), (1, 0)) for _ in range(2)]
        rstd = [_fake(torch.float32, (tok, h), (1, 0)) for h in (h_q, h_kv)] if want_rstd else [None, None]
        # the e4m3 epilogue's operands, traced ONLY under fp8_out (None otherwise: the ABI keeps the parameters, the kernel folds the arm out);
        # under mx_out the two payload slots carry the MX arm's rowwise payloads and the scale slots stay None
        if fp8_out:
            e4m3 = [_fake_e4m3_thd(tok, h_q, d), _fake_e4m3_thd(tok, h_kv, d), _fake_slot(), _fake_slot()]
        elif mx_out:
            e4m3 = [_fake_e4m3_thd(tok, h_q, d), _fake_e4m3_thd(tok, h_kv, d), None, None]
        else:
            e4m3 = [None] * 4
        # the MX epilogue's six appended operands (the SF blobs flat uint8 with a symbolic length; the columnwise payloads [T, H, D])
        mx = (
            [_fake_u8_flat(), _fake_e4m3_thd(tok, h_q, d), _fake_u8_flat(), _fake_u8_flat(), _fake_e4m3_thd(tok, h_kv, d), _fake_u8_flat()]
            if mx_out
            else [None] * 6
        )
        # the pool-write arm's operands, traced ONLY under pool_write: the V band (a [T, H_kv, D] view at a symbolic token stride), the
        # two pools (cache_write.py's fakes: P and the first three strides symbolic, D contiguous) and the slot map of the keyed width
        pools = (
            [_fake_thd(dtype, tok, h_kv, d), _fake_kv_pool(dtype, h_kv, page_size, d), _fake_kv_pool(dtype, h_kv, page_size, d), _fake_slot_map(slot_dtype)]
            if pool_write
            else [None] * 4
        )
        compiled_cache[key] = compile_cached(
            qk_norm_rope_tma_launch,
            *dense,
            *weights,
            *tables,
            *rstd,
            *e4m3,
            cutlass.Int32(0),  # n_tokens  ) runtime; the zeros pin only the TYPE
            cutlass.Int32(0),  # n_q_tiles )
            cutlass.Int32(0),  # n_tiles   )
            cutlass.Int32(1),  # n_ctas    )
            cutlass.Float32(eps),
            d,
            int(rope_dim),
            int(tile_rows),
            int(stages),
            int(stages_o),
            int(threads_per_cta),
            int(h_q),
            int(h_kv),
            int(refill_pos),
            bool(fused_store_wait),
            *mx,
            cutlass.Int32(0),  # seq_len  ) the MX arm's runtime geometry; the zeros pin only the TYPE
            cutlass.Int32(0),  # batch    )
            cutlass.Int32(0),  # n_blk    )
            cutlass.Int32(0),  # n_sft    )
            *pools,
            cutlass.Int64(0),  # n_slots: runtime (the smaller pool's capacity)
            int(page_size),
            slot_dtype == torch.int64,
            _FAKE_STREAM,
            bool(use_pdl),
            options="--enable-tvm-ffi",
            cache_key=cache_key,
            symbol="cudnn_block_qk_norm_rope_tma",
        )
    return QkNormRopeTmaRecipe(
        compiled=compiled_cache[key],
        h_q=h_q,
        h_kv=h_kv,
        d=d,
        eps=float(eps),
        tile_rows=int(tile_rows),
        stages=int(stages),
        stages_o=int(stages_o),
        threads=int(threads_per_cta),
        want_rstd=bool(want_rstd),
        ctas_per_sm=int(ctas_per_sm),
        apply_norm=bool(apply_norm),
        fp8_out=bool(fp8_out),
        mx_out=bool(mx_out),
        use_pdl=bool(use_pdl),
        pool_write=bool(pool_write),
        page_size=int(page_size),
        slot_dtype=slot_dtype,
    )


def _fake_bshd(dtype, h: int, d: int):
    """The MX arm's 4-D ``(B, S, H, D)`` view of a band: symbolic batch and token strides (``S * N`` and ``N`` for a slab band, ``S * H * D``
    and ``H * D`` for a compact buffer), heads contiguous within a token.  TMA clips its box at ``S`` -- the dimension that makes a ragged
    tail zero-fill instead of running into the next batch."""
    return cute.runtime.make_fake_tensor(
        dtype=_convert_to_cutlass_data_type(dtype),
        shape=(cute.sym_int(), cute.sym_int(), h, d),
        stride=(cute.sym_int(), cute.sym_int(), d, 1),
        assumed_align=16,
    )


def _fake_u8_flat():
    """A flat uint8 SF blob with a symbolic length on a 16-byte-aligned base (the MX epilogue's six blobs)."""
    return cute.runtime.make_fake_tensor(dtype=_convert_to_cutlass_data_type(torch.uint8), shape=(cute.sym_int(),), stride=(1,), assumed_align=16)


def check_mx_sf_blob(name: str, ten, n_bytes: int, dev) -> None:
    """One of the MX epilogue's SF blobs: a contiguous, 16-byte-aligned uint8 CUDA tensor of EXACTLY ``n_bytes`` (the SDPA count
    ``B*H*ceil(S/128)*4*D``, the same for the rowwise tiles and the columnwise atoms) on ``dev``."""
    if not isinstance(ten, torch.Tensor) or ten.dtype != torch.uint8:
        got = f"{ten.dtype}" if isinstance(ten, torch.Tensor) else type(ten).__name__
        raise ValueError(f"{name} must be a torch.uint8 tensor (E8M0 bytes in F8_128x4 order), got {got}")
    if not ten.is_cuda or ten.device != dev:
        raise ValueError(f"{name} must be on {dev} with q, got {ten.device}")
    if not ten.is_contiguous() or ten.data_ptr() % 16:
        raise ValueError(f"{name} must be a contiguous, 16-byte-aligned uint8 tensor")
    if int(ten.numel()) != int(n_bytes):
        raise ValueError(
            f"{name} must hold exactly {n_bytes} bytes (B*H*ceil(S/128)*4*D: every SF byte of every 128-row tile is written), got {int(ten.numel())}"
        )


def _bshd_view(band: torch.Tensor, batch: int, seq_len: int):
    """The ``(B, S, H, D)`` view of a ``[T, H, D]`` band (token stride kept, batch stride ``S * token stride``) -- the MX arm's descriptor view."""
    t, h, d = (int(x) for x in band.shape)
    return torch.as_strided(band, (batch, seq_len, h, d), (seq_len * band.stride(0), band.stride(0), band.stride(1), band.stride(2)), band.storage_offset())


def check_e4m3_out(name: str, ten, t: int, h: int, d: int) -> None:
    """The e4m3 epilogue's destination contract: ``float8_e4m3fn`` ``[T, H, D]`` with heads contiguous within a token and a
    token stride that keeps every row 16-byte aligned (compact ``H*D`` satisfies it) on a 16-B-aligned base -- a lane stores
    8 B per row, so anything else is a misaligned ``st.global.v2`` on odd tokens, not a wrong number."""
    if not isinstance(ten, torch.Tensor) or ten.dtype != torch.float8_e4m3fn:
        got = f"{ten.dtype}" if isinstance(ten, torch.Tensor) else type(ten).__name__
        raise ValueError(f"{name} must be a torch.float8_e4m3fn [T, H, D] tensor (the e4m3 epilogue's destination), got {got}")
    if ten.dim() != 3 or int(ten.shape[0]) != t or int(ten.shape[1]) != h or int(ten.shape[2]) != d:
        raise ValueError(f"{name} must be [T={t}, H={h}, D={d}], got {tuple(ten.shape)}")
    s_t, s_h, s_e = (int(x) for x in ten.stride())
    if s_e != 1 or (h != 1 and s_h != d) or (t != 1 and s_t % 16 != 0) or ten.data_ptr() % 16:
        raise ValueError(
            f"{name} (e4m3) must be a [T, H, D] view with heads contiguous within a token -- strides (N, {d}, 1) with the token stride N a multiple of "
            f"16 elements (16-B rows) -- on a 16-B-aligned base; got strides {(s_t, s_h, s_e)}, base {ten.data_ptr() % 16} B past a 16-B boundary"
        )


def run_qk_norm_rope_tma(
    r,
    q,
    k,
    q_out,
    k_out,
    w_q,
    w_k,
    cos,
    sin,
    rstd_q=None,
    rstd_k=None,
    *,
    stream,
    q8=None,
    k8=None,
    scale_q=None,
    scale_k=None,
    sf_q=None,
    q_T8=None,
    sf_q_T=None,
    sf_k=None,
    k_T8=None,
    sf_k_T=None,
    batch=None,
    seq_len=None,
    v=None,
    k_cache=None,
    v_cache=None,
    slot_mapping=None,
) -> None:
    """The lowered launch. ``w_q``/``w_k`` are both ``None`` for a RoPE-only
    recipe (``r.apply_norm`` False) and both tensors otherwise -- checked, both
    directions, exactly as the LDG runner does.

    Appended, checked BOTH ways against ``r.fp8_out`` (Rule 1): under the e4m3 epilogue ``q8`` / ``k8`` (``float8_e4m3fn``
    ``[T, H, D]``, 16-B rows) and ``scale_q`` / ``scale_k`` (1-element fp32 CUDA slots, read in-kernel) are REQUIRED and the
    bf16 outputs ``q_out`` / ``k_out`` must be ``None`` (they do not exist: the input descriptors stand in for them); without
    it all four must be ``None``.

    Appended, checked BOTH ways against ``r.mx_out``: under the MX epilogue ``q8`` / ``k8`` (the ROWWISE payloads), ``sf_q`` /
    ``sf_k`` (the SDPA rowwise SF tiles), ``q_T8`` / ``k_T8`` (the COLUMNWISE payloads, compact ``[T, H, D]``), ``sf_q_T`` / ``sf_k_T``
    (the D-plane-major atoms) and ``batch`` / ``seq_len`` (``T == batch * seq_len``) are REQUIRED; ``q_out`` / ``k_out``,
    ``scale_q`` / ``scale_k`` and the rstd outputs are refused.  Every SF blob is ``B*H*ceil(S/128)*4*D`` bytes.  Without ``mx_out``
    the eight MX operands must be ``None``.

    Appended, checked BOTH ways against ``r.pool_write``: under the pool-write arm ``v`` (the ``[T, H_kv, D]`` V band at any 16-B-aligned
    token stride), ``k_cache`` / ``v_cache`` (``[P, H_kv, page_size, D]``, the cache-write contract) and ``slot_mapping`` (``[T]`` of the
    recipe's slot dtype) are REQUIRED; the kernel range-checks every slot against the SMALLER pool's capacity.  Without it all four must be
    ``None``."""
    t = int(q.shape[0])
    check_norm_weights_match_recipe(r.apply_norm, w_q, w_k)
    if r.want_rstd and (rstd_q is None or rstd_k is None):
        raise ValueError("this artifact was compiled with rstd outputs; both must be bound at execute (Rule 1: no silent fallback)")
    if not r.want_rstd and (rstd_q is not None or rstd_k is not None):
        raise ValueError("this artifact was compiled WITHOUT rstd outputs (want_rstd=False); rstd_q / rstd_k would be silently ignored -- pass None")
    fp8_out = bool(getattr(r, "fp8_out", False))
    mx_out = bool(getattr(r, "mx_out", False))
    mx_ops = (("sf_q", sf_q), ("q_T8", q_T8), ("sf_q_T", sf_q_T), ("sf_k", sf_k), ("k_T8", k_T8), ("sf_k_T", sf_k_T))
    if mx_out:
        if any(v is None for _, v in mx_ops) or q8 is None or k8 is None or batch is None or seq_len is None:
            raise ValueError(
                "this artifact was compiled WITH the MX epilogue (mx_out=True): q8, sf_q, q_T8, sf_q_T, k8, sf_k, k_T8, sf_k_T and batch / seq_len must all be "
                "bound at execute (Rule 1: no silent fallback)"
            )
        if q_out is not None or k_out is not None or scale_q is not None or scale_k is not None:
            raise ValueError(
                "this artifact was compiled WITH the MX epilogue (mx_out=True): it writes the eight MXFP8 outputs only -- pass q_out=k_out=None and no "
                "per-tensor scale_q / scale_k (the E8M0 block scales are the kernel's own)"
            )
        batch, seq_len = int(batch), int(seq_len)
        if batch < 1 or seq_len < 1 or batch * seq_len != t:
            raise ValueError(f"T must equal batch*seq_len: q has T={t}, got batch={batch} seq_len={seq_len}")
        if q.stride(1) != r.d or k.stride(1) != r.d or q.stride(2) != 1 or k.stride(2) != 1:
            raise ValueError(f"q / k must have head stride D={r.d} and element stride 1 (a slab band or a compact buffer), got {q.stride()} / {k.stride()}")
        check_e4m3_out("q8", q8, t, r.h_q, r.d)
        check_e4m3_out("k8", k8, t, r.h_kv, r.d)
        check_e4m3_out("q_T8", q_T8, t, r.h_q, r.d)
        check_e4m3_out("k_T8", k_T8, t, r.h_kv, r.d)
        for name, ten in (("sf_q", sf_q), ("sf_q_T", sf_q_T)):
            check_mx_sf_blob(name, ten, sf_bytes(batch, r.h_q, seq_len, r.d), q.device)
        for name, ten in (("sf_k", sf_k), ("sf_k_T", sf_k_T)):
            check_mx_sf_blob(name, ten, sf_bytes(batch, r.h_kv, seq_len, r.d), q.device)
        for name, ten in (("q8", q8), ("k8", k8), ("q_T8", q_T8), ("k_T8", k_T8)):
            if ten.device != q.device:
                raise ValueError(f"{name} must be on {q.device} with q, got {ten.device}")
        n_blk, n_sft, n_tiles_mx = mx_tile_counts(batch, seq_len, r.h_q, r.h_kv)
        # the persistent cap is the MX arm's RESIDENCY (5 / 6 CTAs per SM: norm / RoPE-only), never the bf16 tile's 8: a cap above
        # it runs a grid-stride tail (mx_rebuild_ctas_per_sm); r.ctas_per_sm still bounds it from above
        n_ctas = min(n_tiles_mx, multiprocessor_count(current_device()) * min(r.ctas_per_sm, mx_rebuild_ctas_per_sm(r.apply_norm)))
        q4, k4 = _bshd_view(q, batch, seq_len), _bshd_view(k, batch, seq_len)
        r.compiled(
            q4,
            k4,
            q4,  # the output descriptors: placeholders the MX arm never touches
            k4,
            w_q,
            w_k,
            cos,
            sin,
            None,
            None,
            q8,
            k8,
            None,
            None,
            cutlass.Int32(t),
            cutlass.Int32(0),  # n_q_tiles: unused by the MX arm
            cutlass.Int32(n_tiles_mx),
            cutlass.Int32(n_ctas),
            cutlass.Float32(r.eps),
            sf_q.view(-1),
            q_T8,
            sf_q_T.view(-1),
            sf_k.view(-1),
            k_T8,
            sf_k_T.view(-1),
            cutlass.Int32(seq_len),
            cutlass.Int32(batch),
            cutlass.Int32(n_blk),
            cutlass.Int32(n_sft),
            None,  # the pool-write arm's operands: absent from the MX arm (v, k_cache, v_cache, slot_mapping, n_slots)
            None,
            None,
            None,
            cutlass.Int64(0),
            cuda.CUstream(int(stream)),
        )
        return
    if any(v is not None for _, v in mx_ops) or batch is not None or seq_len is not None:
        raise ValueError(
            "this artifact was compiled WITHOUT the MX epilogue (mx_out=False); passing sf_q / q_T8 / sf_q_T / sf_k / k_T8 / sf_k_T / batch / seq_len would "
            "silently ignore them (Rule 1)"
        )
    pool_write = bool(getattr(r, "pool_write", False))
    pool_ops = (("v", v), ("k_cache", k_cache), ("v_cache", v_cache), ("slot_mapping", slot_mapping))
    n_slots = 0
    if pool_write:
        if any(x is None for _, x in pool_ops):
            raise ValueError(
                "this artifact was compiled WITH the pool-write arm (pool_write=True): v (the V band [T, H_kv, D]), k_cache, v_cache and slot_mapping "
                "must all be bound at execute (Rule 1: no silent fallback)"
            )
        if v.dtype != q.dtype or v.dim() != 3 or tuple(int(x) for x in v.shape) != (t, r.h_kv, r.d) or int(v.stride(1)) != r.d or int(v.stride(2)) != 1:
            raise ValueError(
                f"v must be the [T={t}, H_kv={r.h_kv}, D={r.d}] V band of q's dtype with contiguous heads (strides (*, {r.d}, 1)), got {v.dtype} "
                f"{tuple(v.shape)} strides {tuple(v.stride())}"
            )
        if (int(v.stride(0)) * v.element_size()) % 16 or v.data_ptr() % 16 or v.device != q.device:
            raise ValueError(f"v rows must be 16-byte aligned on q's device (token stride {v.stride(0)} elements, data_ptr {v.data_ptr():#x}, {v.device})")
        check_pool(k_cache, "k_cache", h=r.h_kv, page_size=r.page_size, d=r.d, dtype=q.dtype, device=q.device)
        check_pool(v_cache, "v_cache", h=r.h_kv, page_size=r.page_size, d=r.d, dtype=q.dtype, device=q.device)
        check_slot_mapping(slot_mapping, "slot_mapping", t=t, device=q.device)
        if slot_mapping.dtype != r.slot_dtype:
            raise ValueError(f"slot_mapping is {slot_mapping.dtype} but this artifact was compiled for {r.slot_dtype}; the slot dtype is a compile key")
        # the capacity the kernel range-checks every slot against: the SMALLER pool's (cache_write.py's rule)
        n_slots = min(int(k_cache.shape[0]), int(v_cache.shape[0])) * r.page_size
    elif any(x is not None for _, x in pool_ops):
        raise ValueError(
            "this artifact was compiled WITHOUT the pool-write arm (pool_write=False); passing v / k_cache / v_cache / slot_mapping would silently ignore "
            "them (Rule 1)"
        )
    n_q_tiles, n_tiles = tile_counts(t, r.h_q, r.h_kv, r.tile_rows, pool_write=pool_write)
    if fp8_out:
        if q8 is None or k8 is None or scale_q is None or scale_k is None:
            raise ValueError(
                "this artifact was compiled WITH the e4m3 epilogue (fp8_out=True): q8, k8 (float8_e4m3fn [T, H, D]) and scale_q, scale_k (1-element fp32 CUDA "
                "slots) must all be bound at execute (Rule 1: no silent fallback)"
            )
        if q_out is not None or k_out is not None:
            raise ValueError(
                "this artifact was compiled WITH the e4m3 epilogue (fp8_out=True): it writes q8 / k8 only -- pass q_out=k_out=None (they would be silently ignored)"
            )
        check_e4m3_out("q8", q8, t, r.h_q, r.d)
        check_e4m3_out("k8", k8, t, r.h_kv, r.d)
        check_scalar_slot("scale_q", scale_q)
        check_scalar_slot("scale_k", scale_k)
        for name, ten in (("q8", q8), ("k8", k8), ("scale_q", scale_q), ("scale_k", scale_k)):
            if ten.device != q.device:
                raise ValueError(f"{name} must be on {q.device} with q, got {ten.device}")
        q_out, k_out = q, k  # placeholders: the kernel never touches the bf16 output descriptors under the e4m3 epilogue
    elif q8 is not None or k8 is not None or scale_q is not None or scale_k is not None:
        raise ValueError(
            "this artifact was compiled WITHOUT the e4m3 epilogue (fp8_out=False); passing q8 / k8 / scale_q / scale_k would silently ignore them (Rule 1)"
        )
    n_ctas = min(n_tiles, multiprocessor_count(current_device()) * r.ctas_per_sm)
    r.compiled(
        q,
        k,
        q_out,
        k_out,
        w_q,
        w_k,
        cos,
        sin,
        rstd_q,
        rstd_k,
        q8,
        k8,
        scale_q.reshape(1) if scale_q is not None else None,  # a 1-element reshape never copies (Rule 1)
        scale_k.reshape(1) if scale_k is not None else None,
        cutlass.Int32(t),
        cutlass.Int32(n_q_tiles),
        cutlass.Int32(n_tiles),
        cutlass.Int32(n_ctas),
        cutlass.Float32(r.eps),
        None,  # the MX epilogue's operands: absent from this artifact (sf_q, q_T8, sf_q_T, sf_k, k_T8, sf_k_T)
        None,
        None,
        None,
        None,
        None,
        cutlass.Int32(0),  # seq_len / batch / n_blk / n_sft: the MX arm's geometry, unused here
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        v,  # the pool-write arm's operands (None under the shipped arms)
        k_cache,
        v_cache,
        slot_mapping,
        cutlass.Int64(n_slots),
        cuda.CUstream(int(stream)),
    )


frost_qk_norm_rope_tma.set_name_prefix("cudnn", remove_cutlass_symbol=True)
