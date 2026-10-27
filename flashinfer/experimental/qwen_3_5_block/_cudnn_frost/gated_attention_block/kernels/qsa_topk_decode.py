# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decode top-k of the block-sparse indexer: ONE thread block per query row (or, the row-split cluster form, 4 / 8 / 16 CTAs of
one thread-block cluster per row), the row's fp32 block logits held in registers, an exact select by ONE wide (4096-bin) histogram
pass plus an exact rank of the k-th bin's candidates (narrowing passes beyond), the selected block ids written in CANONICAL order (ids
strictly descending, ``-1`` padding last).

Contract (the decode scorer's output is this kernel's input; both modules pin it):

* ``logits [T, n_blocks_max]`` fp32, any row stride, unit column stride: row ``t`` holds one score per complete 4-token block in
  columns ``[0, counts[t])``.  The columns at or past ``counts[t]`` NEVER influence the result; the kernel MAY READ them (it loads the
  row's leading columns together with ``counts[t]`` and discards what lies at or past the count), so they must be addressable -- they
  are, inside the ``[T, n_blocks_max]`` tensor -- but may hold anything (uninitialised memory, NaN, 1e30);
* ``counts [T]`` int32: ``counts[t]`` is the VISIBLE count ``n_vis`` of row ``t`` (clamped in-kernel to ``[0, n_blocks_max]``) --
  NEVER the list length ``min(top_k, n_vis)``: the kernel cannot tell the two spellings apart, and the list-length spelling puts
  every row with ``n_vis > top_k`` in the identity region, so the kernel returns ``top_k - 1 .. 0`` (the OLDEST ``top_k`` blocks,
  not the top-k) bitwise reproducibly and with no error.  The list length is DERIVED from the output: ``min(top_k, counts[t])``
  non-``-1`` slots per row (the test module pins the trap);
* ``block_ids [T, top_k]`` int32 out: the ``n_sel = min(top_k, counts[t])`` selected block ids, STRICTLY DESCENDING in
  ``block_ids[t, :n_sel]``, then ``-1`` in ``block_ids[t, n_sel:]``.  The selected SET is the exact top-``n_sel`` of the fp32
  logits (every selected logit >= every unselected one); a tie at the k-th value keeps the SMALLER block ids; two launches are
  bitwise identical (no atomic decides a slot); a row with ``counts <= top_k`` selects every visible block (the identity
  region: ``counts - 1 .. 0``).  ``NaN`` logits have an unspecified rank (the scorer never writes one; the ordering key below
  sorts them with the infinities of their sign); ``-0.0`` ranks below ``+0.0``.
* ``top_k`` in ``[1, 2048]`` (the block declares a multiple of 4 in ``[4, 512]``; the kernel is generic), a compile-time constant
  read off ``block_ids.shape[1]``.

Why this form (the rework of the one-cluster-per-row kernel this module held first; the measurements are in an internal design record):
the serving stack's own top-k runs ONE 1024-thread CTA per row at <= 16384 columns -- the row in registers, one 4096-bin SMEM histogram,
one block scan, the tie candidates of the k-th bin alone -- and was 1.4-6.5x faster than a cluster doing an 8-bit radix select in four
passes with a release-ordered cluster barrier each (~240 ns per barrier, a GPU-scope membar inside; ~1 us more per node in a captured
chain for any cluster launch on this part).  The same single-CTA shape, with a FINER first pass (the 4096 bins cover the 8 octaves
below the row's maximum, 512 bins per octave -- 11x finer than an fp16 key, so the k-th bin holds a handful of candidates even at
32768 columns) and an exact, deterministic, canonical compaction instead of an atomic-order scatter, is at or under it on every
decode cell of the Qwen 3.8 gate on paper; the ranking on the development node and the exclusive-node gate say whether it is.

Algorithm (``THREADS`` threads, ``EPT = cols_cap / THREADS`` elements per thread; warp ``w`` owns the CONTIGUOUS id range
``[w x EPT x 32, (w + 1) x EPT x 32)`` and its round ``i`` the 32 consecutive ids ``w x EPT x 32 + 32 i + lane`` -- coalesced 128-B
rounds, and an ascending id order that is warp-major, round-major, lane-minor; ``n = clamp(counts[t], 0, n_blocks_max)``):

1. ``counts[t]`` and the row's first ``min(EPT, SPEC_ROUNDS)`` rounds of logits are loaded in the SAME memory flight: ONE memory
   latency instead of two for every row of up to ``SPEC_ROUNDS x 32 x nwarps`` columns.  A row as wide as the compiled capacity
   (``n_blocks_max == cols_cap``: every power-of-two width) has every round inside it: one load per round off a per-thread base, no
   address arithmetic; a narrower row clamps the speculated rounds' address to the last column (a dummy for the tail -- never against
   ``n``, which is not known yet) and the rounds past them, loaded after ``n`` lands, to the last VISIBLE column (a dummy, no
   predicate).  ``n <= top_k`` -> the identity list.
2. Keys: ``u = bits ^ (bits >> 31 ? 0xFFFFFFFF : 0x80000000)`` (a larger float is a larger unsigned key; a REAL key is 0 only for a
   negative NaN, whose rank is unspecified) and every DUMMY (an id at or past ``n``) given key 0 -- the smallest key, below every
   threshold the kernel derives, so a dummy is never selected and never a candidate; a row with every column of the capacity visible
   (``n == cols_cap``, the common decode row) has no dummy and skips that select; kept in REGISTERS for the whole kernel.  The
   valid maximum ``kmax`` (warp ``redux``, one barrier, one in-warp reduce).
3. Pass 1 -- the WINDOWED histogram of EVERY key (no predicate; 4 instructions per element): ``base = max(kmax >> 14, 4095) -
   4095``, digit ``max((key >> 14) - base, 0)`` (never above 4095: no key exceeds the row maximum, whose digit is 4095 or less by the
   choice of the base): 4096 bins over the 8 octaves below the maximum, everything further below saturates into bin 0 (exact,
   merely slower); the dummies all land in digit 0 and their count (``EPT x THREADS - n``) is taken out of bin 0 by the scan.
   Plain shared atomics (no ``match.any``: at 4096 bins a warp seldom holds two equal digits).
4. The scan (``_scan_find``): thread ``t`` owns the 4 (8 at 512 threads) consecutive bins ``4095 - BPT x t - j`` in DESCENDING order;
   its sum -> a branch-free warp inclusive scan (shuffles + arithmetic selects) -> ``s_wsum[warp]`` -> barrier -> every thread scans
   ``s_wsum`` in-warp and takes its own warp's exclusive prefix with ONE index shuffle -> ``above(b)`` per bin -> the ONE thread whose
   bin holds the ``need``-th candidate publishes ``(bin, above, count)`` -> barrier.
5. The verdict is a THRESHOLD PAIR ``(kstar, idstar)``: selected <=> ``key > kstar`` or (``key == kstar`` and ``id <= idstar``).  Bin
   ``B`` of pass 1 spans the keys ``[lo, hi]`` (``lo = (base + B) << 14``, or 0 for the saturated bin; ``hi = lo + 2^14 - 1``) and
   ``need = top_k - above``:
   - the bin taken whole (``count == need``): ``kstar = lo - 1``, ``idstar = n - 1``;
   - THE FAST PATH (``lo >= 2^14`` -- the bin's lower bound above the dummies' key and a negative NaN's -- and the bin holding at
     most ``RANK_CAP`` candidates, taken whole or not; the common case; compiled in from ``FAST_MIN_EPT`` = 8 keys per thread on,
     below which the fallback's two cheap sweeps win): every key AT OR ABOVE
     ``lo`` is either surely selected (a higher bin: fewer than ``top_k`` in the row) or a candidate of the bin, so ONE sweep -- the
     staging sweep -- stages them: per round one compare (``key >= lo``) and one ballot; a round holding any stores, per selected
     lane, ONE word into the warp's region of ``STAGE_CAP = top_k + RANK_CAP`` words at the warp's running count plus the lanes
     below (ascending id order within the warp; the regions concatenate in warp order = ascending id order over the row), the
     candidates of the bin (``key <= hi``) ALSO appended to the rank list (``s_rank_key`` / ``s_rank_id``, ``RANK_CAP`` slots) and
     staged as their slot with bit 31 set.  No region can overflow: a warp stages at most (surely selected < ``top_k``) +
     (candidates <= ``RANK_CAP``); the store's address select carries ``rel < STAGE_CAP`` anyway.  The bin not taken whole: one barrier
     (the rank list), the spread rank -- ``THREADS / RANK_CAP`` consecutive lanes share one candidate, each ranks it against every
     so-many-th other (key descending, id ascending), the partial ranks summed by butterfly shuffles -- names the ``need``-th
     candidate, whose (key, id) IS the pair, one barrier;
   - THE FALLBACK (``lo == 0``: the saturated bin or a row whose maximum is a negative NaN / -inf; a bin wider than ``RANK_CAP``,
     taken whole or not; fewer than ``FAST_MIN_EPT`` keys per thread; a ``top_k`` whose regions exceed ``STAGE_POOL_MAX_BYTES``): the
     bin taken whole needs no list; otherwise the bin's candidates (``key``, ``id``)
     compacted ONCE into the candidate list (one sweep; the append path runs only in the warps that hold a candidate; the id guard
     keeps the dummies out when ``lo == 0``), then narrowing passes over the list (12-bit digits of ``[lo, hi]``; the row's keys
     instead of the list when the bin exceeds ``CAND_CAP``), each 3 barriers, until the bin is taken whole or the key is exact; an
     exact key with only part of its tied ids wanted resolves ``idstar`` by a 2 x 12-bit select over the ids of the keys equal to
     it (ascending ids = descending complemented digits, the same scan).  Four passes at most (the saturated bin: ``[0, 2^32)`` in
     12 + 12 + 8 bits).
6. The canonical compaction.  The fast path works on the staged words alone: each warp walks ITS region (one iteration of 32 words
   for a warp holding up to 32; a flagged word reads its (key, id) from the rank list, a plain word is a selected id) to count its
   selected (``sel = key > kstar | (key == kstar & id <= idstar)``; every staged word when the bin is taken whole, so that walk is
   skipped) -> one barrier -> every thread takes its warp's exclusive prefix over the warp counts -> the same walk writes the
   selected id of ascending rank ``pos`` at slot ``top_k - 1 - pos``.  The fallback sweeps the keys: sweep A counts the selected
   ids per warp (one ballot per round; lane 0 stores the round's ballot word in ``s_bal[warp x EPT + i]``) -> one barrier -> the
   exclusive prefix -> sweep B reads each round's word back (no recomputation; a zero word skips the round) and writes as above.
   Either way: strictly descending ids, every slot of a row with ``n > top_k`` written, no atomic decides a slot, two launches
   bitwise.

Barrier table (REQUIRED pre-development table; no mbarrier, no cluster barrier: every site is ``bar.sync 0`` reached by EVERY thread of
the CTA exactly once per visit, ``SUM(issuing lanes) == THREADS`` at every row; the identity path performs no barrier):

| site | when | what it orders (writes before -> reads after) |
|---|---|---|
| B0 | after the loads, before pass 1 | ``s_hist[0][*] = 0`` (every thread, BPT bins each), ``s_wmax[warp]`` (lane 0 per warp), ``s_ncand = 0`` -> the window base read by every thread and the pass-1 atomics |
| B1_1 | after pass 1's atomics | the atomics into ``s_hist[0]`` -> the scan's reads; ``s_hist[1][*] = 0`` -> a later pass's atomics |
| B2_p / B3_p (``_scan_find``, once per executed histogram pass p and per id-select pass) | the warp sums / the record | ``s_wsum[warp]`` (lane 31 per warp) -> the cross-warp prefix; ``s_found[0..2]`` (ONE thread) -> every thread; ``s_wsum`` free again |
| C1 (fast path, the bin not taken whole) | after the staging sweep | ``s_rank_key`` / ``s_rank_id[slot]`` and ``s_ncand`` (the appending lanes) -> the spread rank; the staging regions are private to their warp (a ``bar.warp.sync`` after the sweep orders the warp's own stores before its walks) |
| R1 (fast path, the bin not taken whole) | after the spread rank | ``s_wmax[0]`` / ``s_found[3]`` = the pair (ONE thread; the fourth word: words 0..2 were read after B3) -> every thread |
| F2 (fast path) | after the counting walk | ``s_wsum[warp]`` = the warp's selected count (lane 0) -> the cross-warp prefix before the writing walk |
| C1 (fallback, the bin not taken whole) | after the candidate sweep | ``s_cand`` / ``s_cand_id[slot]`` and ``s_ncand`` (the appending lanes) -> the later passes (or the all-pairs rank of a kernel without the fast path) |
| R1 (fallback without the fast path) | after the all-pairs rank | ``s_wmax[0]`` / ``s_found[3]`` = the pair (ONE thread) -> every thread |
| B1_p (later passes, p <= 4) | after the pass's atomics | the atomics into ``s_hist[p & 1]`` -> the scan; the other buffer zeroed -> the next pass |
| I0_q / I1_q (the id select, q <= 2) | before / after its atomics | the zeroing of ``s_hist[q & 1]`` -> the atomics; the atomics -> the scan |
| F (fallback) | after sweep A | ``s_wsum[warp]`` = the warp's selected count and ``s_bal[warp x EPT + i]`` = the round's ballot word (lane 0 per warp) -> the cross-warp prefix and sweep B's reads |

The common path (the fast path, the spread rank decides) is 7 sites: B0, B1_1, B2_1, B3_1, C1, R1, F2; a bin taken whole at pass 1 is
5 (B0, B1_1, B2_1, B3_1, F2); the fallback's shortest path is B0, B1_1, B2_1, B3_1, C1, B1_2, B2_2, B3_2, F.

SMEM table (REQUIRED pre-development table; dynamic shared memory through ``SmemAllocator``; no descriptor reads any buffer, so each row
states the per-lane stride that keeps the banks spread):

| name / dtype / elems | written by | read by | per-lane stride, bank argument |
|---|---|---|---|
| ``s_hist`` i32 x ``2 x 4096`` | ``red.shared.add`` at the digit of each candidate (data-dependent); the zeroing ``s_hist[b][j x THREADS + t]`` | the scan: thread ``t`` reads bins ``4095 - BPT x t - j`` (``4 x BPT`` B between lanes: 4 wavefronts per load, BPT loads per pass -- immaterial) | the zeroing is conflict-free; the histogram side is bin-addressed by nature |
| ``s_cand`` u32 x ``CAND_CAP``, ``s_cand_id`` i32 x ``CAND_CAP`` (the fallback's candidate list; views of ``s_pool``) | the appending lanes of a warp at ``base + popc(ballot & lt_mask)`` (consecutive slots: 4 B between lanes) | the later passes (round-strided; the all-pairs rank of a kernel without the fast path: entry ``t`` and every entry) | conflict-free |
| ``s_pool`` i32 x ``max(nwarps x STAGE_CAP + 1, fallback_pool_words(cols_cap) = 2 x CAND_CAP + cols_cap / 32)`` (the fast path's staging regions, ``STAGE_CAP = top_k + RANK_CAP``, 0 words when they would exceed ``STAGE_POOL_MAX_BYTES`` = 96 KB; the fallback's ``s_cand`` / ``s_cand_id`` / ``s_bal`` are VIEWS of the same words -- a row runs one path or the other) | the staging sweep: the selected lanes of a round store CONSECUTIVE words of their warp's region (4 B between them); every other lane stores the last word (one same-address broadcast store) | the two walks: warp ``w`` reads words ``base + lane`` of its region (consecutive) | conflict-free |
| ``s_rank_key`` u32 x ``RANK_CAP``, ``s_rank_id`` i32 x ``RANK_CAP`` | the appending lanes (consecutive slots) | the spread rank: ``THREADS / RANK_CAP`` consecutive lanes read one slot (a broadcast), then slots ``part + j x parts`` (distinct words across the group); the walks: a flagged word's slot (<= 2 words per lane) | conflict-free / broadcast |
| ``s_bal`` u32 x ``cols_cap / 32`` (one word per round per warp; the fallback; a view of ``s_pool`` past the list) | lane 0 of warp ``w`` at word ``w x EPT + i`` in sweep A | sweep B: every lane of warp ``w`` reads word ``w x EPT + i`` (a broadcast) | consecutive words per warp, one lane writes, broadcast reads: conflict-free |
| ``s_wsum`` i32 x 32 | lane 31 (the scans) or lane 0 (sweep A) of each warp | every thread reads slot ``lane`` | 4 B between lanes: conflict-free |
| ``s_wmax`` u32 x 32 | lane 0 of each warp (the warp maxima); slot 0 again by the ranked thread (``kstar``) | every thread reads slot ``lane`` / slot 0 | conflict-free / broadcast |

Dynamic shared memory at 32768 columns, 1024 threads, ``top_k`` 512: 32 KB (``s_hist``) + 80 KB (the pool: the staging regions, 32 x
640 words, over the fallback's 36 KB of list + ballot words) + 1 KB (the rank list) + the small rows ~ 114 KB; 512 threads halve the
regions (40 KB: 73 KB in all -- the 99-KB parts serve 512 threads at ``top_k`` 512 and decline 1024 by the limit gate below).
| ``s_found`` i32 x 4, ``s_ncand`` i32 x 4 | the one finder thread / the appending lanes (atomics) | every thread (broadcast) | trivial |

Degenerate inputs, each with its handling: ``counts[t] == 0`` -> every slot ``-1``, nothing of the row is used; ``counts[t] < 0`` -> clamped
to 0; ``counts[t] > n_blocks_max`` -> clamped (the row is never addressed past its width); ``counts[t] <= top_k`` -> the identity list
``counts - 1 .. 0`` then ``-1``; a row of identical logits -> pass 1 puts them all in one bin wider than the list, the row-key passes keep
them together until the key is exact, the id select names the ``need``-th smallest id; two distinct values / a boundary tie -> the same
id select; negative logits order by value (the key flips every bit of a negative float); garbage past ``counts[t]`` is loaded (within
the row) and discarded.

Programmatic dependent launch (``use_pdl``): ``griddepcontrol.wait`` before the FIRST dependent global read (the counts and the
speculated logits: the scorer writes both), ``griddepcontrol.launch_dependents`` once per CTA right after the ``counts`` read -- before the
identity branch, so the identity path fires it too (the sparse tile that consumes the list waits for THIS grid's completion; the trigger
only lets it be scheduled early) -- traced under the ``use_pdl`` compile key and launched with the attribute under the same flag; the two
forms are bitwise.

The cluster form (``cluster_ctas`` in 4 / 8 / 16; ``frost_qsa_topk_decode_cluster``): the row split into C aligned slices of
``cols_cap / C`` columns, every CTA running the body above on its slice with 1024 threads, and four cluster-wide exchanges by
``st.async.shared::cluster`` completing the receiver's mbarrier (no release-ordered cluster barrier anywhere: a relaxed arrive + wait
publishes the mbarrier inits and another one is the exit drain): the row maximum (C words), the pass-1 histogram as a REDUCE-SCATTER +
ALL-GATHER (``_merge_hist``: 16 KB in and out per CTA per phase), the k-th bin's candidates as one fixed 1 KB message per CTA (a count,
up to ``UNION_CAP`` = 127 (key, id) pairs; a CTA with more sends the merged-pass fallback down the same path in every CTA), and the
per-CTA selected counts (C words) for the canonical prefix.  Every CTA derives the same verdict from the same merged data; the
candidates' rank, the narrowing passes over the union and the id select are CTA-local.  Its barrier table: P0 (``mbarrier.init`` x 5
by thread 0, each with init count ``LOCAL_ARRIVERS`` = 1 -> ``fence.mbarrier_init`` by every thread -> ``bar.sync`` ->
``barrier.cluster.arrive.relaxed`` + ``wait``: the inits before any peer's store), then per exchange ONE local
``mbarrier.arrive.expect_tx`` (thread 0) against exactly the bytes the C senders push (``SUM(stores x bytes) == expect_tx``:
``C x XCHG_WORD_BYTES`` for the maximum and the counts, ``HIST_XCHG_BYTES`` = C x 4096 / C x 4 = 16384 for each merge phase,
``C x UNION_MSG_BYTES`` = C x 1024 for the union), every thread ``mbarrier.try_wait`` on the phase parity (``mb_rs`` / ``mb_ag`` advance
once per merged pass; the others are used once), and the exit drain (``barrier.cluster.arrive.relaxed`` + ``wait`` after the last wait
of every CTA: no CTA leaves while a peer's store into it may be in flight -- a store lands only before the phase it completes, and
every CTA waited for every phase that targets it).  Its CTA-local ``bar.sync`` sites beyond the single-CTA form's: M1 (inside
``_merge_hist``, once per merge phase: the reduced slice ``s_mslice`` before its chunks leave), C2 (the union message ``s_msg``
assembled before its chunks leave), C3 (the union compacted into ``s_cand`` / ``s_cand_id`` before the rank or the passes), R2 (the
id rank over the union list behind the narrowing passes: the finder publishes ``idstar`` in ``s_found[3]`` -- the FOURTH word, read
nowhere else, because every thread read words 0..2 right after the last pass's B3 with no barrier since; a write into word 1 there
raced a reader still behind that read), and F before the counts exchange.  A receive buffer is reused across merged passes without a
second barrier because a peer's pass-p+1 store needs the receiver's pass-p all-gather message, sent only after the receiver consumed
the pass-p data.  SMEM: ``s_rs`` / ``s_merged`` (16 KB each), ``s_mslice`` (4096 / C words), ``s_union`` (C KB), ``s_msg`` (1 KB),
``s_xmax`` / ``s_cnt`` (C words each, the two one-word-per-sender receive buffers), the single-CTA buffers (``s_bal`` sized to the
slice), five mbarriers.

Knobs (performance-only, the same function under every served value -- the tests pin every value bitwise against the others):
``threads_per_cta`` in (512, 1024) for the single-CTA form (``EPT = cols_cap / threads_per_cta`` keys per thread stay in registers: 32 at
1024 threads and 32768 columns, 64 at 512 -- the ``MAX_EPT`` bound; the cluster form runs 1024); ``cluster_ctas`` in (1, 4, 8, 16)
(16 is a non-portable cluster size: the CuTe DSL launch sets ``CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED`` on every kernel it
loads, driver 11.8+).  ``top_k`` sizes the staging regions (``smem_bytes_for(..., top_k)``); above 640 at 1024 threads (1408 at 512)
the regions would exceed ``STAGE_POOL_MAX_BYTES`` and the kernel is compiled as the fallback form alone (correct, slower).  ``cols_cap`` (the compiled row capacity, a power of two, >= ``n_blocks_max``, >= ``cluster_ctas x threads_per_cta``,
<= ``MAX_EPT x cluster_ctas x threads_per_cta``) sizes the register tile; one artifact serves every ``n_blocks_max`` up to it.
Requirements: a CuTe DSL at or above the FROST floor with the device's target (Rule 7, gated before anything else), a cc 9.0+ device
for ``cluster_ctas > 1`` (thread-block clusters, DSMEM) and for ``use_pdl`` (``griddepcontrol``; both exist from Hopper; the single-CTA
form runs on any CuTe DSL target), the public CuTe DSL wheel at or above ``CLUSTER_FORM_MIN_DSL`` = 4.8.0 for ``cluster_ctas > 1`` (its
vector ``st.async`` exchanges: the released 4.7.0 fails that compile -- a typed decline naming the version, the single-CTA form serving
and the default knob chooser picking it there), and the per-CTA dynamic shared memory within the device's opt-in limit (named in the
decline otherwise).  No performance number is claimed here; an internal record carries the ranking and the measurement of record.

Importing THIS module already needs the floor: it imports the CuTe DSL and ``cudnn.frost.tile_dsl`` at module scope, so on a DSL
below the floor the import fails inside the DSL before any gate here can name the version.  The entry a caller hits is the
DSL-free facade ``cudnn.gated_attention_block.qsa_topk`` (Rule 7): it gates the version first and imports this module lazily;
the gates in ``compile_qsa_topk_decode`` / ``qsa_topk_decode`` below are the second line, for a caller that imported this module
on a sufficient DSL.
"""

import hashlib
from pathlib import Path
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.experimental.qwen_3_5_block._cudnn_frost._cutlass_compat import SmemAllocator
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import compile_cached, template_key
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import compute_capability, current_device, shared_memory_per_block_optin
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import ld_global, st_global

__all__ = [
    "FROST_SOURCE_DIGEST",
    "QSA_TOPK_DECODE_MAX_K",
    "MAX_ROWS",
    "MAX_EPT",
    "CLUSTER_SIZES",
    "THREAD_COUNTS",
    "DEFAULT_CLUSTER_CTAS",
    "DEFAULT_THREADS_PER_CTA",
    "default_knobs_for",
    "QsaTopkDecodeRecipe",
    "cols_cap_for",
    "smem_bytes_for",
    "compile_qsa_topk_decode",
    "run_qsa_topk_decode",
    "qsa_topk_decode",
]

QSA_TOPK_DECODE_MAX_K = 2048
"""The widest list this kernel writes (``block_ids.shape[1]``); the block's own declaration caps ``top_k`` at 512."""
CLUSTER_SIZES = (1, 4, 8, 16)
"""The served cluster sizes: 1 = one thread block per row (any thread count); 4 / 8 / 16 = the row split over a thread-block cluster
(1024 threads per CTA; 16 is a non-portable cluster size the CuTe DSL launch enables on every kernel it loads)."""
THREAD_COUNTS = (512, 1024)
DEFAULT_CLUSTER_CTAS = 1
DEFAULT_THREADS_PER_CTA = 1024
BIN_BITS = 12
BINS = 1 << BIN_BITS
WINDOW_SHIFT = 14
"""Pass 1 histograms ``key >> WINDOW_SHIFT`` relative to the row maximum: 4096 bins over ``2^(WINDOW_SHIFT + BIN_BITS - 23) = 8`` octaves."""
MAX_PASSES = 4
CAND_CAP = 4096
RANK_CAP = 128
STAGE_POOL_MAX_BYTES = 96 * 1024  # the staging regions of the fast path (nwarps x (top_k + RANK_CAP) words) must fit this; else the kernel is the fallback form
STAGE_FLAG = -(1 << 31)
FAST_MIN_EPT = 8  # the fast path from 8 keys per thread on (below it the two-sweep fallback is the cheaper form: the ranking of 2048-column rows)  # a staged word with bit 31 set names a rank-list slot (a candidate of the k-th bin), a clear one a plain id
"""Up to this many candidates in the k-th bin, the select ends with an exact all-pairs rank among them (one barrier) instead of the
narrowing passes (three barriers each)."""
"""The k-th bin's candidates compacted into shared memory after pass 1 (keys only); a bin with more candidates than this (a row of
(near-)identical logits) is resolved by the later passes sweeping the row's keys in registers instead of the list (slower, exact)."""
SPEC_ROUNDS = 8
"""The rounds of logits loaded in the same memory flight as ``counts`` (8192 columns at 1024 threads)."""
MAX_EPT = 64
"""Keys per thread kept in registers: the register budget of a 512-thread CTA (128 per thread) carries 64; 1024 threads carry 32."""
MAX_ROWS = 65535
"""The widest row block one launch serves: one CTA per row along ``grid.y``, whose extent the launch grid caps at 65535."""
_FAKE_STREAM = None
compiled_cache = {}
# Imported normally (not through the template loader), so the module names its own source identity: ``template_key`` joins
# it with the compile site's key to name the persistent object (``compiled_cache``); the cache's environment manifest covers
# every other ``.py`` the kernel is traced from.
# read through the module globals by the persistent compiled-plan cache (``template_key(globals(), ...)``: the digest keys every
# artifact of this file) and by the test attestation; exported so the reference is explicit
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def _is_pow2(x: int) -> bool:
    return x >= 1 and (x & (x - 1)) == 0


CLUSTER_FORM_MIN_DSL = (4, 8, 0)
"""The cluster form's exchanges store four words per thread with one tuple-valued ``store_async_dsmem`` (the vector ``st.async``); the
released CuTe DSL 4.7.0 -- admitted by the FROST floor -- evaluates ``Int32(value)`` on that tuple and fails the cluster form's compile
(the single-CTA form compiles there).  The public wheel at or above this version compiles it; an internal build is not judged (the
floor's own rule), and a version that does not parse is not refused."""


def cluster_form_requirement_error(cluster_ctas: int) -> Optional[str]:
    """Message naming the installed DSL version when ``cluster_ctas > 1`` and the public wheel is below ``CLUSTER_FORM_MIN_DSL``, else
    None -- a device-free gate, read before any compile (the single-CTA form is the way out it names)."""
    if int(cluster_ctas) <= 1:
        return None
    installed, version = cutedsl_state()
    if not installed or not version or version[0] != "nvidia-cutlass-dsl":
        return None
    parts = []
    for piece in str(version[1]).split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    if not parts or tuple(parts) >= CLUSTER_FORM_MIN_DSL:
        return None
    need = ".".join(str(x) for x in CLUSTER_FORM_MIN_DSL)
    return (
        f"qsa_topk_decode with cluster_ctas = {int(cluster_ctas)} (the row-split cluster form) needs nvidia-cutlass-dsl >= {need} for its vector "
        f"st.async exchanges; found {version[1]} -- use cluster_ctas = 1 (the single-CTA form) there, or upgrade with: pip install -U 'nvidia-cutlass-dsl>={need}'"
    )


def cluster_form_available() -> bool:
    """Whether the installed DSL compiles the cluster form: the default knob chooser's capability read."""
    return cluster_form_requirement_error(CLUSTER_SIZES[1]) is None


def default_knobs_for(rows: int, n_blocks_max: int, cc=None) -> tuple:
    """The knob pair ``(cluster_ctas, threads_per_cta)`` the convenience entry picks for a row block: the row-split cluster form only
    where it earned its exchanges on the ranking -- rows of more than 8192 blocks (128K tokens and beyond) at up to 32 rows (8 CTAs
    per row up to 16 rows, 4 at 17 .. 32: 32 x 8 CTAs would outnumber the SMs and run in waves); every other block runs one 1024-thread
    CTA per row.  Below cc 9.0 (no clusters) and on a public CuTe DSL wheel below ``CLUSTER_FORM_MIN_DSL`` (the cluster form does not
    compile there) always ``(1, 1024)``.  A caller with its own measurement passes the knobs explicitly."""
    if cc is not None and tuple(cc) < (9, 0):
        return (1, DEFAULT_THREADS_PER_CTA)
    if not cluster_form_available():
        return (1, DEFAULT_THREADS_PER_CTA)
    if n_blocks_max > 8192:
        if rows <= 16:
            return (8, CLUSTER_THREADS)
        if rows <= 32:
            return (4, CLUSTER_THREADS)
    return (1, DEFAULT_THREADS_PER_CTA)


def cols_cap_for(n_blocks_max: int, cluster_ctas: int = DEFAULT_CLUSTER_CTAS, threads_per_cta: int = DEFAULT_THREADS_PER_CTA) -> int:
    """The smallest compiled row capacity that serves ``n_blocks_max`` columns: a power of two, at least one element per thread
    (``threads_per_cta``, so the row is a whole number of rounds).  ``cluster_ctas`` is accepted for the callers of the first form
    (the served value is 1; it does not enter the capacity)."""
    if n_blocks_max < 1:
        raise ValueError(f"n_blocks_max must be >= 1, got {n_blocks_max}")
    cap = 1
    while cap < n_blocks_max:
        cap *= 2
    return max(cap, int(cluster_ctas) * int(threads_per_cta))


def validate_topk_geometry(cols_cap: int, top_k: int, cluster_ctas: int, threads_per_cta: int) -> None:
    """Raise ``ValueError`` on any geometry the kernel cannot address (never an ``assert``: user-facing)."""
    if cluster_ctas not in CLUSTER_SIZES:
        raise ValueError(
            f"cluster_ctas must be one of {CLUSTER_SIZES} (1 = one thread block per row; 4 / 8 / 16 = the row-split cluster form), got {cluster_ctas}"
        )
    if threads_per_cta not in THREAD_COUNTS:
        raise ValueError(f"threads_per_cta must be one of {THREAD_COUNTS}, got {threads_per_cta}")
    if cluster_ctas > 1 and threads_per_cta != CLUSTER_THREADS:
        raise ValueError(f"the cluster form (cluster_ctas = {cluster_ctas}) runs {CLUSTER_THREADS} threads per CTA; got threads_per_cta = {threads_per_cta}")
    if not 1 <= top_k <= QSA_TOPK_DECODE_MAX_K:
        raise ValueError(f"top_k must be in [1, {QSA_TOPK_DECODE_MAX_K}], got {top_k}")
    if not _is_pow2(cols_cap):
        raise ValueError(f"cols_cap must be a power of two, got {cols_cap}")
    if cols_cap % (cluster_ctas * threads_per_cta) != 0:
        raise ValueError(
            f"cols_cap ({cols_cap}) must be a multiple of cluster_ctas x threads_per_cta ({cluster_ctas * threads_per_cta}): every CTA's slice is whole rounds of the thread count"
        )
    if cols_cap // (cluster_ctas * threads_per_cta) > MAX_EPT:
        advice = (
            "use threads_per_cta = 1024"
            if threads_per_cta < THREAD_COUNTS[-1]
            else f"the row is wider than {MAX_EPT * THREAD_COUNTS[-1]} blocks, past this kernel's reach"
        )
        raise ValueError(f"{cols_cap // threads_per_cta} keys per thread overflow the register tile ({MAX_EPT} per thread); {advice}")


def fallback_pool_words(cols_cap: int) -> int:
    """The words the fallback's views need of the pool: the candidate list (``CAND_CAP`` keys + ``CAND_CAP`` ids) and the ballot words
    (one per round per warp = ``cols_cap / 32``)."""
    return 2 * CAND_CAP + cols_cap // 32


def stage_words_for(top_k: int, threads_per_cta: int) -> int:
    """The staging pool of the fast path in words: one region of ``top_k + RANK_CAP`` words per warp plus the scratch word, or 0 when
    the regions would exceed ``STAGE_POOL_MAX_BYTES`` (the kernel then runs its fallback form for every row: ``top_k`` above 640 at
    1024 threads, above 1408 at 512)."""
    words = (threads_per_cta // 32) * (top_k + RANK_CAP) + 1
    return words if words * 4 <= STAGE_POOL_MAX_BYTES else 0


def smem_bytes_for(cols_cap: int, cluster_ctas: int, threads_per_cta: int, top_k: Optional[int] = None) -> int:
    """Dynamic shared memory of one CTA (the SMEM table's rows, 16-B aligned each, plus the allocator's slack).  ``top_k`` sizes the
    fast path's staging pool; without it the pool is its bound (``STAGE_POOL_MAX_BYTES``: the figure a caller plans with)."""
    if cluster_ctas > 1:
        return smem_bytes_for_cluster(cols_cap, cluster_ctas)

    def al(x):
        return (x + 15) // 16 * 16

    stage = STAGE_POOL_MAX_BYTES // 4 if top_k is None else stage_words_for(top_k, threads_per_cta)
    pool = max(stage, fallback_pool_words(cols_cap))  # ONE pool: the staging regions of the fast path, the list + ballot words of the fallback
    total = al(2 * BINS * 4) + al(pool * 4) + 2 * al(RANK_CAP * 4) + al(32 * 4) + al(32 * 4) + al(4 * 4) + al(4 * 4)
    return total + 128


# --------------------------------------------------------------------------------------------------- predicated inline PTX
# Branch-free per-element forms: a predicated instruction instead of an ``scf.if`` region per element (an ``if`` inside a
# ``range_constexpr`` loop is one BSSY/BSYNC pair per iteration; EPT of them per sweep).  The predicate is a 0 / 1 Int32 register
# operand (never an immediate -- the ``inline_ptx(predicate=)`` hazard of frost-tile-dsl.md S 8 is avoided by not using that kwarg).


def _ptr_arg(p):
    if hasattr(p, "llvm_ptr"):
        return p.llvm_ptr
    if hasattr(p, "ir_value") and callable(p.ir_value):
        return p.ir_value()
    return p


@cute.jit
def _warp_scan_incl(v, lane):
    """Branch-free inclusive warp scan of ``v`` (Int32) in lane order: shuffles up by 1, 2, 4, 8, 16 with arithmetic selects."""
    incl = v
    for s in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_up(incl, 1 << s, mask_and_clamp=0)
        incl = incl + (other if lane >= cutlass.Int32(1 << s) else cutlass.Int32(0))
    return incl


@cute.jit
def _cross_warp_excl(s_wsum, lane, warp, nwarps: cutlass.Constexpr[int]):
    """After a barrier that published ``s_wsum[w]`` for every warp ``w < nwarps``: this warp's EXCLUSIVE prefix over the warp sums
    (every thread reads slot ``lane``, scans in-warp, takes slot ``warp`` with one index shuffle) and the total over every warp."""
    idx = lane if lane < cutlass.Int32(nwarps) else cutlass.Int32(0)
    v = s_wsum[idx]
    v = v if lane < cutlass.Int32(nwarps) else cutlass.Int32(0)
    incl = _warp_scan_incl(v, lane)
    excl = incl - v
    mine = cute.arch.shuffle_sync(excl, warp, mask_and_clamp=31)
    total = cute.arch.shuffle_sync(incl, cutlass.Int32(nwarps - 1), mask_and_clamp=31)
    return mine, total


@cute.jit
def _scan_find(s_hist, hbuf: cutlass.Constexpr[int], s_wsum, s_found, need, corr, tidx, lane, warp, threads_per_cta: cutlass.Constexpr[int]):
    """The descending scan over the 4096 bins of ``s_hist[hbuf:]`` (thread ``t`` owns bins ``4095 - BPT x t - j``, ``corr`` taken out of
    bin 0) and the find: the ONE thread whose bin holds the ``need``-th candidate publishes ``(bin, above, count)`` in ``s_found``.
    Two barriers (B2: the warp sums; B3: the record); the caller reads ``s_found`` after it returns."""
    bpt = cutlass.const_expr(BINS // threads_per_cta)
    nwarps = cutlass.const_expr(threads_per_cta // 32)
    zero = cutlass.Int32(0)
    cnts = []
    tot = zero
    for j in cutlass.range_constexpr(bpt):
        c = s_hist[cutlass.Int32(hbuf + BINS - 1) - tidx * cutlass.Int32(bpt) - cutlass.Int32(j)]
        if cutlass.const_expr(j == bpt - 1):
            c = c - corr
        cnts.append(c)
        tot = tot + c
    incl = _warp_scan_incl(tot, lane)
    if lane == cutlass.Int32(31):
        s_wsum[warp] = incl
    cute.arch.barrier()  # B2
    carry, _ = _cross_warp_excl(s_wsum, lane, warp, nwarps)
    above = incl + carry - tot  # the candidates strictly above this thread's first (highest) bin
    for j in cutlass.range_constexpr(bpt):
        c = cnts[j]
        if (above < need) & (need <= above + c):
            s_found[0] = cutlass.Int32(BINS - 1) - tidx * cutlass.Int32(bpt) - cutlass.Int32(j)
            s_found[1] = above
            s_found[2] = c
        above = above + c
    cute.arch.barrier()  # B3


@cute.kernel
def frost_qsa_topk_decode(
    mLogits: cute.Tensor,  # [T, n_blocks_max] the fp32 logits viewed as int32 bits; row stride symbolic, unit column stride
    mCounts: cute.Tensor,  # [T] int32 contiguous
    mOut: cute.Tensor,  # [T, top_k] int32; row stride symbolic, unit column stride
    cols_cap: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """Grid ``(1, T, 1)``: ``block_idx.y`` is the query row; one CTA per row.  The SM issues ~4 warp instructions per cycle for the
    whole CTA, so every instruction of a per-element sweep costs ``EPT x nwarps / 4`` cycles: the body is written for the fewest
    instructions per element (no per-element predicate on the histogram pass, no bitmap words, no per-element branch) and the fewest
    barriers on its chain.  Warp ``w`` owns the CONTIGUOUS id range ``[w x EPT x 32, (w + 1) x EPT x 32)``, round ``i`` of it the 32
    consecutive ids ``w x EPT x 32 + 32 i + lane`` (coalesced 128-B rounds): the ascending id order is then warp-major, round-major,
    lane-minor, and a selected id's canonical rank is (the lower warps' selected count) + (the warp's running count) + (the lanes below)."""
    ept = cutlass.const_expr(cols_cap // threads_per_cta)
    spec = cutlass.const_expr(min(ept, SPEC_ROUNDS))
    nwarps = cutlass.const_expr(threads_per_cta // 32)
    bpt = cutlass.const_expr(BINS // threads_per_cta)  # bins per thread in the scan (4 at 1024 threads, 8 at 512)
    stage_words = cutlass.const_expr(stage_words_for(top_k, threads_per_cta))
    use_fast = cutlass.const_expr(stage_words > 0 and ept >= FAST_MIN_EPT)
    stage_cap = cutlass.const_expr(top_k + RANK_CAP)  # one region per warp
    scratch = cutlass.const_expr(max(stage_words - 1, 0))  # the word every unselected lane's store lands on
    branch_sweep = cutlass.const_expr(ept >= 16)  # at 16+ rounds most rounds stage nothing: skip the body on a zero ballot
    parts = cutlass.const_expr(threads_per_cta // RANK_CAP)  # threads per candidate in the spread rank (8 at 1024, 4 at 512)
    pshift = cutlass.const_expr(parts.bit_length() - 1)

    smem = SmemAllocator()
    pool_words = cutlass.const_expr(max(stage_words, fallback_pool_words(cols_cap)))
    s_hist = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2 * BINS,)), byte_alignment=16)
    s_pool = smem.allocate_tensor(cutlass.Int32, cute.make_layout((pool_words,)), byte_alignment=16)  # the fast path's staging regions
    s_rank_key = smem.allocate_tensor(cutlass.Uint32, cute.make_layout((RANK_CAP,)), byte_alignment=16)
    s_rank_id = smem.allocate_tensor(cutlass.Int32, cute.make_layout((RANK_CAP,)), byte_alignment=16)
    s_wsum = smem.allocate_tensor(cutlass.Int32, cute.make_layout((32,)), byte_alignment=16)
    s_wmax = smem.allocate_tensor(cutlass.Uint32, cute.make_layout((32,)), byte_alignment=16)
    s_found = smem.allocate_tensor(cutlass.Int32, cute.make_layout((4,)), byte_alignment=16)
    s_ncand = smem.allocate_tensor(cutlass.Int32, cute.make_layout((4,)), byte_alignment=16)
    # the fallback's buffers are VIEWS of the pool (a row runs one path or the other, never both): the candidate list's keys and ids,
    # then the ballot words (one per round per warp)
    pool_addr = s_pool.iterator.toint()
    s_cand = cute.make_tensor(cute.make_ptr(cutlass.Uint32, pool_addr, cute.AddressSpace.smem, assumed_align=16), cute.make_layout((CAND_CAP,)))
    s_cand_id = cute.make_tensor(
        cute.make_ptr(cutlass.Int32, pool_addr + cutlass.Int32(CAND_CAP * 4), cute.AddressSpace.smem, assumed_align=16), cute.make_layout((CAND_CAP,))
    )
    s_bal = cute.make_tensor(
        cute.make_ptr(cutlass.Uint32, pool_addr + cutlass.Int32(2 * CAND_CAP * 4), cute.AddressSpace.smem, assumed_align=16),
        cute.make_layout((cols_cap // 32,)),
    )

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = cutlass.Int32(cute.arch.lane_idx())
    warp = cutlass.Int32(cute.arch.make_warp_uniform(tidx // cutlass.Int32(32)))
    row = cutlass.Int32(cute.arch.block_idx()[1])
    my_bit = cutlass.Uint32(1) << lane.to(cutlass.Uint32)
    lt_mask = my_bit - cutlass.Uint32(1)  # the lanes below this one (= %lanemask_lt, derived)
    row64 = row.to(cutlass.Int64)
    four = cutlass.Int64(4)
    one = cutlass.Int32(1)
    zero = cutlass.Int32(0)
    e0 = warp * cutlass.Int32(ept * 32) + lane  # this thread's first id; round i adds 32 i

    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    cols = cutlass.Int32(mLogits.shape[1])
    row_ptr = mLogits.iterator + row64 * cutlass.Int64(mLogits.stride[0])
    out_row = mOut.iterator.toint() + row64 * cutlass.Int64(mOut.stride[0]) * four
    exact = cols == cutlass.Int32(cols_cap)  # the row is as wide as the compiled capacity: every round is inside it, no address clamp
    p0 = row_ptr + e0
    # ---- the counts and the speculated rounds in ONE memory flight (the address clamped to the row width when the row is narrower
    # than the capacity: a dummy for the tail) ----------------------------------------------------------------------------------
    raw = [zero] * spec  # the list's shape is fixed before the branch (the traced join needs one type on both paths)
    if exact:
        for i in cutlass.range_constexpr(spec):
            raw[i] = (p0 + cutlass.Int32(32 * i)).load()
    else:
        for i in cutlass.range_constexpr(spec):
            e = e0 + cutlass.Int32(32 * i)
            e_c = e if e < cols else cols - one
            raw[i] = (row_ptr + e_c).load()
    n = ld_global(mCounts.iterator.toint() + row64 * four, cutlass.Int32)
    n = n if n > zero else zero
    n = n if n < cols else cols

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()

    if n <= cutlass.Int32(top_k):
        # the identity region: every visible block, ids descending, -1 past them
        for j in range(tidx, top_k, threads_per_cta):
            val = n - one - j
            val = val if j < n else cutlass.Int32(-1)
            st_global(out_row + j.to(cutlass.Int64) * four, val, cutlass.Int32)
    else:
        # ---- the rounds past the speculated ones (inside the row by construction, or the address clamped to the last VISIBLE
        # column: a dummy, no predicate) ----------------------------------------------------------------------------------------
        n_m1 = n - one
        tail = [zero] * (ept - spec)
        if exact:
            for i0 in cutlass.range_constexpr(ept - spec):
                tail[i0] = (p0 + cutlass.Int32(32 * (i0 + spec))).load()
        else:
            for i0 in cutlass.range_constexpr(ept - spec):
                e = e0 + cutlass.Int32(32 * (i0 + spec))
                e_c = e if e < n_m1 else n_m1
                tail[i0] = (row_ptr + e_c).load()
        # ---- the keys (registers): every real key >= 1 but a negative NaN's, the dummies' key 0 (never selected: the thresholds
        # below are >= 0 and a dummy's id is past every real id); the row maximum ------------------------------------------------
        keys = [cutlass.Uint32(0)] * ept
        kmax_t = cutlass.Uint32(0)
        full = n == cutlass.Int32(cols_cap)  # every column of the capacity visible: no dummy anywhere (the common decode row)
        if full:
            for i in cutlass.range_constexpr(ept):
                bits = (raw[i] if i < spec else tail[i - spec]).to(cutlass.Uint32)
                sign = bits >> cutlass.Uint32(31)
                flip = (cutlass.Uint32(0) - sign) | cutlass.Uint32(0x80000000)
                key = bits ^ flip
                keys[i] = key
                kmax_t = kmax_t if kmax_t > key else key
        else:
            for i in cutlass.range_constexpr(ept):
                bits = (raw[i] if i < spec else tail[i - spec]).to(cutlass.Uint32)
                sign = bits >> cutlass.Uint32(31)
                flip = (cutlass.Uint32(0) - sign) | cutlass.Uint32(0x80000000)
                key = bits ^ flip
                e = e0 + cutlass.Int32(32 * i)
                key = key if e < n else cutlass.Uint32(0)  # a dummy: the smallest key; a REAL key 0 (a negative NaN) is told apart by its id
                keys[i] = key
                kmax_t = kmax_t if kmax_t > key else key
        kmax_w = cute.arch.warp_redux_sync(kmax_t, "umax")
        if lane == zero:
            s_wmax[warp] = kmax_w
        for j in cutlass.range_constexpr(bpt):
            s_hist[cutlass.Int32(j * threads_per_cta) + tidx] = zero
        if tidx == zero:
            s_ncand[0] = zero
        cute.arch.barrier()  # B0
        idxm = lane if lane < cutlass.Int32(nwarps) else zero
        vm = s_wmax[idxm]
        vm = vm if lane < cutlass.Int32(nwarps) else cutlass.Uint32(0)
        kmax = cute.arch.warp_redux_sync(vm, "umax")
        top = kmax >> cutlass.Uint32(WINDOW_SHIFT)
        base = (top - cutlass.Uint32(BINS - 1)) if top > cutlass.Uint32(BINS - 1) else cutlass.Uint32(0)
        n_dummy = cutlass.Int32(ept * threads_per_cta) - n  # the forced-0 keys, all in digit 0 of pass 1

        # ---- pass 1: the windowed 4096-bin histogram of EVERY key (no predicate; the digit never exceeds 4095: no key is above the
        # row maximum, whose digit is at most 4095 by the choice of the base) ---------------------------------------------------
        wbase = base.to(cutlass.Int32)
        for i in cutlass.range_constexpr(ept):
            d = (keys[i] >> cutlass.Uint32(WINDOW_SHIFT)).to(cutlass.Int32) - wbase
            d = d if d > zero else zero
            cute.arch.atomic_add(s_hist.iterator + d, one, sem="relaxed", scope="cta")
        for j in cutlass.range_constexpr(bpt):
            s_hist[cutlass.Int32(BINS + j * threads_per_cta) + tidx] = zero
        cute.arch.barrier()  # B1_1
        corr = n_dummy if tidx == cutlass.Int32(threads_per_cta - 1) else zero
        _scan_find(s_hist, 0, s_wsum, s_found, cutlass.Int32(top_k), corr, tidx, lane, warp, threads_per_cta)
        b = s_found[0]
        above_b = s_found[1]
        cnt_b = s_found[2]
        need = cutlass.Int32(top_k) - above_b
        bu = (b + wbase).to(cutlass.Uint32)
        width_m1 = cutlass.Uint32((1 << WINDOW_SHIFT) - 1)
        new_lo = bu << cutlass.Uint32(WINDOW_SHIFT)
        saturated = (b == zero) & (wbase > zero)  # the window's bin 0: every key below the window
        lo = cutlass.Uint32(0) if saturated else new_lo
        hi = new_lo + width_m1
        whole = cnt_b == need

        # ---- the verdict as a THRESHOLD PAIR (kstar, idstar): selected <=> key > kstar or (key == kstar and id <= idstar) ----------
        # the bin taken whole: everything at or above lo
        kstar = (lo - cutlass.Uint32(1)) if lo > cutlass.Uint32(0) else cutlass.Uint32(0)
        idstar = n - one
        fast = cutlass.Boolean(False)
        if cutlass.const_expr(use_fast):
            # the fast path: the k-th bin's lower bound is above the dummies' key (lo >= 2^14) and the elements at or above the bin are
            # few enough to stage (every one of them when the bin is taken whole; otherwise at most RANK_CAP candidates on top of the
            # surely-selected ones): ONE sweep stages them per warp in id order, and everything after works on the staged words
            fast = (lo != cutlass.Uint32(0)) & (cnt_b <= cutlass.Int32(RANK_CAP))
        if fast:
            warp_off = warp * cutlass.Int32(stage_cap)
            sh_lt = cutlass.Uint32(32) - lane.to(cutlass.Uint32)  # bal << sh_lt keeps the lanes below (a shift of 32 clears the word)
            run = zero
            for i in cutlass.range_constexpr(ept):
                key = keys[i]
                c = key >= lo
                bal = cutlass.Uint32(cute.arch.vote_ballot_sync(c))
                go = cutlass.Boolean(True)
                if cutlass.const_expr(branch_sweep):
                    go = bal != cutlass.Uint32(0)
                if go:
                    rel = run + cutlass.Int32(cute.arch.popc(bal << sh_lt))
                    is_cand = c & (key <= hi)  # the bin's candidates: lo <= key <= hi
                    w_c = cutlass.Uint32(cute.arch.vote_ballot_sync(is_cand))
                    slot = zero
                    if w_c != cutlass.Uint32(0):
                        wbase_c = zero
                        if lane == zero:
                            wbase_c = cute.arch.atomic_add(s_ncand.iterator, cutlass.Int32(cute.arch.popc(w_c)), sem="relaxed", scope="cta")
                        wbase_c = cute.arch.shuffle_sync(wbase_c, zero, mask_and_clamp=31)
                        slot = wbase_c + cutlass.Int32(cute.arch.popc(w_c << sh_lt))  # < RANK_CAP: the bin holds at most RANK_CAP candidates here
                        if is_cand:
                            s_rank_key[slot] = key
                            s_rank_id[slot] = e0 + cutlass.Int32(32 * i)
                    word = (slot + cutlass.Int32(STAGE_FLAG)) if is_cand else (e0 + cutlass.Int32(32 * i))
                    idx = (warp_off + rel) if (c & (rel < cutlass.Int32(stage_cap))) else cutlass.Int32(scratch)
                    s_pool[idx] = word
                    run = run + cutlass.Int32(cute.arch.popc(bal))
            cnt_w = run  # this warp's staged count (warp-uniform)
            cute.arch.sync_warp()  # the region and the rank-list entries this warp wrote, visible to every lane of it
            if not whole:
                cute.arch.barrier()  # C1: the rank list and its count
                ncand = s_ncand[0]  # <= cnt_b <= RANK_CAP here
                # the spread rank: `parts` consecutive lanes share one candidate, each ranks it against every parts-th other, the
                # partial ranks summed by butterfly shuffles inside the aligned lane group
                cnd = tidx >> cutlass.Int32(pshift)
                part = tidx & cutlass.Int32(parts - 1)
                valid_c = cnd < ncand
                cidx = cnd if valid_c else zero
                key_c = s_rank_key[cidx]
                id_c = s_rank_id[cidx]
                rnk = zero
                for j in range(part, ncand, parts):
                    kj = s_rank_key[j]
                    ij = s_rank_id[j]
                    ahead = (kj > key_c) | ((kj == key_c) & (ij < id_c))
                    rnk = rnk + (one if ahead else zero)
                for sft in cutlass.range_constexpr(pshift):
                    rnk = rnk + cute.arch.shuffle_sync_bfly(rnk, 1 << sft, mask_and_clamp=31)
                if valid_c & (part == zero) & (rnk == need - one):
                    s_wmax[0] = key_c  # s_wmax is free after B0
                    s_found[3] = id_c  # the fourth word: words 0..2 were read after B3 (C1 lies between, the fourth word is read nowhere else)
                cute.arch.barrier()  # R1: the pair
                kstar = s_wmax[0]
                idstar = s_found[3]
            # ---- the canonical compaction over the staged words: this warp's region in id order; pass S1 counts the selected (every
            # staged word is selected when the bin is taken whole: the count is the staged count), F2 publishes the warp counts, pass
            # S2 writes the selected ids at their global descending rank ------------------------------------------------------------
            sel_w = cnt_w
            if not whole:
                sel_w = zero
                for base_r in range(0, cnt_w, 32):
                    r = base_r + lane
                    valid = r < cnt_w
                    rr = r if valid else cnt_w - one
                    word = s_pool[warp_off + rr]
                    flagged = word < zero
                    sl = word & cutlass.Int32(RANK_CAP - 1)
                    kk = s_rank_key[sl]
                    ii = s_rank_id[sl]
                    idv = ii if flagged else word
                    sel = valid & ((~flagged) | (kk > kstar) | ((kk == kstar) & (idv <= idstar)))
                    sel_w = sel_w + cutlass.Int32(cute.arch.popc(cutlass.Uint32(cute.arch.vote_ballot_sync(sel))))
            if lane == zero:
                s_wsum[warp] = sel_w
            cute.arch.barrier()  # F2: the per-warp selected counts
            wlow, _ = _cross_warp_excl(s_wsum, lane, warp, nwarps)
            run2 = wlow
            for base_r in range(0, cnt_w, 32):
                r = base_r + lane
                valid = r < cnt_w
                rr = r if valid else cnt_w - one
                word = s_pool[warp_off + rr]
                flagged = word < zero
                sl = word & cutlass.Int32(RANK_CAP - 1)
                kk = s_rank_key[sl]
                ii = s_rank_id[sl]
                idv = ii if flagged else word
                sel = valid & ((~flagged) | (kk > kstar) | ((kk == kstar) & (idv <= idstar)))
                bal = cutlass.Uint32(cute.arch.vote_ballot_sync(sel))
                if sel:
                    pos = run2 + cutlass.Int32(cute.arch.popc(bal << sh_lt))
                    slot_o = cutlass.Int32(top_k - 1) - pos
                    st_global(out_row + slot_o.to(cutlass.Int64) * four, idv, cutlass.Int32)
                run2 = run2 + cutlass.Int32(cute.arch.popc(bal))
        else:
            # ---- the fallback: the bin's candidates listed (one sweep), the narrowing passes over the list (or the row), the id
            # select for a partial tie, then the two-sweep canonical compaction over the keys ----------------------------------------
            if not whole:
                use_list = cnt_b <= cutlass.Int32(CAND_CAP)
                span1 = hi - lo
                if use_list:
                    for i in cutlass.range_constexpr(ept):
                        key = keys[i]
                        inb = (key - lo) <= span1  # unsigned: lo <= key <= hi in one compare
                        if lo == cutlass.Uint32(0):
                            inb = inb & ((e0 + cutlass.Int32(32 * i)) < n)  # a range from 0 would admit the dummies: keep the real ids only
                        w_c = cutlass.Uint32(cute.arch.vote_ballot_sync(inb))
                        if w_c != cutlass.Uint32(0):
                            wbase_c = zero
                            if lane == zero:
                                wbase_c = cute.arch.atomic_add(s_ncand.iterator, cutlass.Int32(cute.arch.popc(w_c)), sem="relaxed", scope="cta")
                            wbase_c = cute.arch.shuffle_sync(wbase_c, zero, mask_and_clamp=31)
                            slot = wbase_c + cutlass.Int32(cute.arch.popc(w_c & lt_mask))
                            if inb & (slot < cutlass.Int32(CAND_CAP)):
                                s_cand[slot] = key
                                s_cand_id[slot] = e0 + cutlass.Int32(32 * i)
                cute.arch.barrier()  # C1: the list and its count
                ncand = s_ncand[0]
                ncand = ncand if ncand < cutlass.Int32(CAND_CAP) else cutlass.Int32(CAND_CAP)
                if (not use_fast) & use_list & (ncand <= cutlass.Int32(RANK_CAP)):
                    # (only without the fast path: there a bin of <= RANK_CAP candidates never reaches here) few candidates: an exact
                    # all-pairs rank (key descending, id ascending); the need-th candidate IS the threshold pair
                    if tidx < ncand:
                        key_c = s_cand[tidx]
                        id_c = s_cand_id[tidx]
                        rnk = zero
                        for j in range(ncand):
                            kj = s_cand[j]
                            ij = s_cand_id[j]
                            ahead = (kj > key_c) | ((kj == key_c) & (ij < id_c))
                            rnk = rnk + (one if ahead else zero)
                        if rnk == need - one:
                            s_wmax[0] = key_c  # s_wmax is free after B0
                            s_found[3] = id_c
                    cute.arch.barrier()  # R1: the pair
                    kstar = s_wmax[0]
                    idstar = s_found[3]
                else:
                    # the narrowing passes over the list (or, a bin wider than the list, over the row's keys): 12-bit digits of [lo, hi]
                    done = cutlass.Boolean(False)
                    take = zero
                    for p0_ in cutlass.range_constexpr(MAX_PASSES - 1):
                        p = p0_ + 1
                        if not done:
                            hbuf = cutlass.const_expr((p & 1) * BINS)
                            span = hi - lo
                            blen = cutlass.Int32(32) - cute.arch.clz(span).to(cutlass.Int32)
                            nsh = blen - cutlass.Int32(BIN_BITS)
                            sh = (nsh if nsh > zero else zero).to(cutlass.Uint32)
                            span_p = hi - lo
                            if use_list:
                                for c_ in range(tidx, ncand, threads_per_cta):
                                    key = s_cand[c_]
                                    if (key - lo) <= span_p:
                                        d = ((key - lo) >> sh).to(cutlass.Int32)
                                        cute.arch.atomic_add(s_hist.iterator + (cutlass.Int32(hbuf) + d), one, sem="relaxed", scope="cta")
                            else:
                                for i in cutlass.range_constexpr(ept):
                                    key = keys[i]
                                    if ((key - lo) <= span_p) & ((e0 + cutlass.Int32(32 * i)) < n):
                                        d = ((key - lo) >> sh).to(cutlass.Int32)
                                        cute.arch.atomic_add(s_hist.iterator + (cutlass.Int32(hbuf) + d), one, sem="relaxed", scope="cta")
                            if cutlass.const_expr(p + 1 < MAX_PASSES):
                                nbuf = cutlass.const_expr(((p + 1) & 1) * BINS)
                                for j in cutlass.range_constexpr(bpt):
                                    s_hist[cutlass.Int32(nbuf + j * threads_per_cta) + tidx] = zero
                            cute.arch.barrier()  # B1_p
                            _scan_find(s_hist, hbuf, s_wsum, s_found, need, zero, tidx, lane, warp, threads_per_cta)
                            b = s_found[0]
                            above_b = s_found[1]
                            cnt_b = s_found[2]
                            need = need - above_b
                            new_lo = lo + (b.to(cutlass.Uint32) << sh)
                            hi = new_lo + ((cutlass.Uint32(1) << sh) - cutlass.Uint32(1))
                            lo = new_lo
                            whole_p = cnt_b == need
                            exact_k = lo == hi
                            done = whole_p | exact_k
                            take = need
                    # the pass verdict as a pair: a bin taken whole -> everything at or above lo; an exact key with every tied id taken ->
                    # the key with idstar = n - 1; an exact key with a partial take -> the take-th smallest id among the tied ids, found by
                    # a 2 x 12-bit select over the ids of the keys equal to it (ascending ids = descending complemented digits)
                    kstar = (lo - cutlass.Uint32(1)) if lo > cutlass.Uint32(0) else cutlass.Uint32(0)
                    idstar = n - one
                    if (lo == hi) & (take < cnt_b):
                        kstar = lo
                        id_lo = zero  # the id range [id_lo, id_lo + 2^id_sh x 4096) narrowed twice: ids < 2^24
                        for q in cutlass.range_constexpr(2):
                            id_sh = cutlass.const_expr(12 * (1 - q))
                            hbuf = cutlass.const_expr((q & 1) * BINS)
                            for j in cutlass.range_constexpr(bpt):
                                s_hist[cutlass.Int32(hbuf + j * threads_per_cta) + tidx] = zero
                            cute.arch.barrier()  # I0_q
                            for i in cutlass.range_constexpr(ept):
                                e = e0 + cutlass.Int32(32 * i)
                                d = cutlass.Int32(BINS - 1) - (((e - id_lo) >> cutlass.Int32(id_sh)) & cutlass.Int32(BINS - 1))
                                if (keys[i] == kstar) & (e < n) & (e >= id_lo) & (e < id_lo + cutlass.Int32(BINS << id_sh)):
                                    cute.arch.atomic_add(s_hist.iterator + (cutlass.Int32(hbuf) + d), one, sem="relaxed", scope="cta")
                            cute.arch.barrier()  # I1_q
                            _scan_find(s_hist, hbuf, s_wsum, s_found, take, zero, tidx, lane, warp, threads_per_cta)
                            db = cutlass.Int32(BINS - 1) - s_found[0]  # the digit holding the take-th smallest id
                            take = take - s_found[1]
                            id_lo = id_lo + (db << cutlass.Int32(id_sh))
                        idstar = id_lo + take - one  # the exact id: `take` ids remain wanted at the last (unit) digit, and they are one id

            # ---- the canonical compaction: sweep A counts the selected ids per warp, one barrier, sweep B writes --------------------
            wcnt = zero
            for i in cutlass.range_constexpr(ept):
                key = keys[i]
                e = e0 + cutlass.Int32(32 * i)
                sel = (key > kstar) | ((key == kstar) & (e <= idstar))
                bal = cutlass.Uint32(cute.arch.vote_ballot_sync(sel))
                if lane == zero:
                    s_bal[warp * cutlass.Int32(ept) + cutlass.Int32(i)] = bal
                wcnt = wcnt + cutlass.Int32(cute.arch.popc(bal))
            if lane == zero:
                s_wsum[warp] = wcnt
            cute.arch.barrier()  # F: the per-warp selected counts and the words
            wlow, _ = _cross_warp_excl(s_wsum, lane, warp, nwarps)
            run = wlow
            for i in cutlass.range_constexpr(ept):
                bal = s_bal[warp * cutlass.Int32(ept) + cutlass.Int32(i)]
                if bal != cutlass.Uint32(0):
                    if (bal & my_bit) != cutlass.Uint32(0):
                        pos = run + cutlass.Int32(cute.arch.popc(bal & lt_mask))
                        slot = cutlass.Int32(top_k - 1) - pos
                        st_global(out_row + slot.to(cutlass.Int64) * four, e0 + cutlass.Int32(32 * i), cutlass.Int32)
                    run = run + cutlass.Int32(cute.arch.popc(bal))


frost_qsa_topk_decode.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# ------------------------------------------------------------------------------------------------- the row-split CLUSTER form
CLUSTER_THREADS = 1024
"""The cluster form runs 1024 threads per CTA (its merge maps one 16-B chunk of bins per thread)."""
UNION_CAP = 127
"""Candidates of the k-th bin one CTA contributes to the cluster-wide union (one 1 KB message: a count word, a pad word, 127 pairs);
a CTA with more falls back to the merged passes over the row's keys."""
MSG_WORDS = 256
# the exchange table (every mbarrier: init count LOCAL_ARRIVERS; one ``arrive.expect_tx`` by thread 0 per phase against EXACTLY the
# bytes the C senders' ``st.async`` complete; the senders never arrive)
LOCAL_ARRIVERS = 1  # the one local arriver (thread 0) of every mbarrier phase
XCHG_WORD_BYTES = 4  # one Int32 per sender: the row maximum and the selected-count exchanges expect ``C x XCHG_WORD_BYTES``
HIST_XCHG_BYTES = BINS * 4  # each merge phase (reduce-scatter, all-gather) completes the whole histogram: C senders x (4096 / C bins) x 4 B
UNION_MSG_BYTES = MSG_WORDS * 4  # one CTA's union message (1 KB); the union exchange expects ``C x UNION_MSG_BYTES``


@cute.jit
def _merge_hist(s_hist, hbuf: cutlass.Constexpr[int], s_rs, s_mslice, s_merged, mb_rs, mb_ag, parity, rank, C: cutlass.Constexpr[int], tidx):
    """The cluster-wide sum of the C local histograms into every CTA's ``s_merged``: a REDUCE-SCATTER (thread ``t`` pushes its four bins
    ``4t .. 4t + 3`` to the peer that owns that slice of bins, 16 B by ``st.async`` completing the peer's ``mb_rs``; every CTA then sums
    its slice over the C senders) and an ALL-GATHER (every thread pushes one 16-B chunk of the reduced slice to one peer's ``s_merged``,
    completing ``mb_ag``).  Each mbarrier phase expects exactly 16 KB (``C x 4096 / C x 4`` B); ``parity`` is the phase of this call."""
    bins_per_peer = cutlass.const_expr(BINS // C)
    thr_per_peer = cutlass.const_expr(CLUSTER_THREADS // C)
    zero = cutlass.Int32(0)
    q = tidx // cutlass.Int32(thr_per_peer)
    off_in = cutlass.Int32(4) * tidx - q * cutlass.Int32(bins_per_peer)
    v0 = s_hist[cutlass.Int32(hbuf) + cutlass.Int32(4) * tidx]
    v1 = s_hist[cutlass.Int32(hbuf) + cutlass.Int32(4) * tidx + cutlass.Int32(1)]
    v2 = s_hist[cutlass.Int32(hbuf) + cutlass.Int32(4) * tidx + cutlass.Int32(2)]
    v3 = s_hist[cutlass.Int32(hbuf) + cutlass.Int32(4) * tidx + cutlass.Int32(3)]
    dst = cute.make_ptr(
        cutlass.Int32, s_rs.iterator.toint() + (rank * cutlass.Int32(bins_per_peer) + off_in) * cutlass.Int32(4), cute.AddressSpace.smem, assumed_align=16
    )
    cute.arch.store_async_dsmem(dst, (v0, v1, v2, v3), mb_rs, q)
    if tidx == zero:
        cute.arch.mbarrier_arrive_and_expect_tx(mb_rs, HIST_XCHG_BYTES)
    cute.arch.mbarrier_wait(mb_rs, parity)
    if tidx < cutlass.Int32(bins_per_peer):
        acc = zero
        for j in cutlass.range_constexpr(C):
            acc = acc + s_rs[cutlass.Int32(j * bins_per_peer) + tidx]
        s_mslice[tidx] = acc
    cute.arch.barrier()  # M1: the reduced slice before its chunks leave
    c = tidx - q * cutlass.Int32(thr_per_peer)
    w0 = s_mslice[cutlass.Int32(4) * c]
    w1 = s_mslice[cutlass.Int32(4) * c + cutlass.Int32(1)]
    w2 = s_mslice[cutlass.Int32(4) * c + cutlass.Int32(2)]
    w3 = s_mslice[cutlass.Int32(4) * c + cutlass.Int32(3)]
    dst2 = cute.make_ptr(
        cutlass.Int32,
        s_merged.iterator.toint() + (rank * cutlass.Int32(bins_per_peer) + cutlass.Int32(4) * c) * cutlass.Int32(4),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    cute.arch.store_async_dsmem(dst2, (w0, w1, w2, w3), mb_ag, q)
    if tidx == zero:
        cute.arch.mbarrier_arrive_and_expect_tx(mb_ag, HIST_XCHG_BYTES)
    cute.arch.mbarrier_wait(mb_ag, parity)


@cute.kernel
def frost_qsa_topk_decode_cluster(
    mLogits: cute.Tensor,
    mCounts: cute.Tensor,
    mOut: cute.Tensor,
    cols_cap: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    cluster_ctas: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """Grid ``(C, T, 1)``, cluster ``(C, 1, 1)``: CTA ``rank`` owns the aligned slice ``[rank x cols_cap / C, (rank + 1) x cols_cap / C)``
    of the row (the split follows the compiled width, so every round is speculated against the row width and masked by ``n``); the
    body is the single-CTA kernel's with the histogram merged cluster-wide per pass (``_merge_hist``), the row maximum, the k-th bin's
    candidates and the per-CTA selected counts exchanged once each by ``st.async`` + mbarrier, and a relaxed cluster barrier as the
    exit drain."""
    C = cutlass.const_expr(cluster_ctas)
    threads_per_cta = cutlass.const_expr(CLUSTER_THREADS)
    slice_cap = cutlass.const_expr(cols_cap // C)
    ept = cutlass.const_expr(slice_cap // threads_per_cta)
    nwarps = cutlass.const_expr(threads_per_cta // 32)
    bpt = cutlass.const_expr(BINS // threads_per_cta)
    bins_per_peer = cutlass.const_expr(BINS // C)
    msg_chunks = cutlass.const_expr(MSG_WORDS // 4)  # 16-B chunks of the union message (64)

    smem = SmemAllocator()
    s_hist = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2 * BINS,)), byte_alignment=16)
    s_rs = smem.allocate_tensor(cutlass.Int32, cute.make_layout((BINS,)), byte_alignment=16)
    s_merged = smem.allocate_tensor(cutlass.Int32, cute.make_layout((BINS,)), byte_alignment=16)
    s_mslice = smem.allocate_tensor(cutlass.Int32, cute.make_layout((bins_per_peer,)), byte_alignment=16)
    s_union = smem.allocate_tensor(cutlass.Uint32, cute.make_layout((C * MSG_WORDS,)), byte_alignment=16)
    s_msg = smem.allocate_tensor(cutlass.Uint32, cute.make_layout((MSG_WORDS,)), byte_alignment=16)
    s_cand = smem.allocate_tensor(cutlass.Uint32, cute.make_layout((CAND_CAP,)), byte_alignment=16)
    s_cand_id = smem.allocate_tensor(cutlass.Int32, cute.make_layout((CAND_CAP,)), byte_alignment=16)
    s_bal = smem.allocate_tensor(cutlass.Uint32, cute.make_layout((slice_cap // 32,)), byte_alignment=16)
    s_wsum = smem.allocate_tensor(cutlass.Int32, cute.make_layout((32,)), byte_alignment=16)
    s_wmax = smem.allocate_tensor(cutlass.Uint32, cute.make_layout((32,)), byte_alignment=16)
    s_xmax = smem.allocate_tensor(cutlass.Int32, cute.make_layout((16,)), byte_alignment=16)
    s_cnt = smem.allocate_tensor(cutlass.Int32, cute.make_layout((16,)), byte_alignment=16)
    s_found = smem.allocate_tensor(cutlass.Int32, cute.make_layout((4,)), byte_alignment=16)
    s_ncand = smem.allocate_tensor(cutlass.Int32, cute.make_layout((4,)), byte_alignment=16)
    s_mb = smem.allocate_tensor(cutlass.Int64, cute.make_layout((8,)), byte_alignment=16)
    mb_max = s_mb.iterator + 0
    mb_rs = s_mb.iterator + 1
    mb_ag = s_mb.iterator + 2
    mb_un = s_mb.iterator + 3
    mb_cnt = s_mb.iterator + 4

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = cutlass.Int32(cute.arch.lane_idx())
    warp = cutlass.Int32(cute.arch.make_warp_uniform(tidx // cutlass.Int32(32)))
    rank = cutlass.Int32(cute.arch.block_idx()[0])
    row = cutlass.Int32(cute.arch.block_idx()[1])
    my_bit = cutlass.Uint32(1) << lane.to(cutlass.Uint32)
    lt_mask = my_bit - cutlass.Uint32(1)
    row64 = row.to(cutlass.Int64)
    four = cutlass.Int64(4)
    one = cutlass.Int32(1)
    zero = cutlass.Int32(0)
    my_start = rank * cutlass.Int32(slice_cap)
    e0 = my_start + warp * cutlass.Int32(ept * 32) + lane  # this thread's first GLOBAL id; round i adds 32 i

    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()
    cols = cutlass.Int32(mLogits.shape[1])
    row_ptr = mLogits.iterator + row64 * cutlass.Int64(mLogits.stride[0])
    out_row = mOut.iterator.toint() + row64 * cutlass.Int64(mOut.stride[0]) * four
    raw = []
    for i in cutlass.range_constexpr(ept):
        e = e0 + cutlass.Int32(32 * i)
        e_c = e if e < cols else cols - one
        raw.append((row_ptr + e_c).load())
    n = ld_global(mCounts.iterator.toint() + row64 * four, cutlass.Int32)
    n = n if n > zero else zero
    n = n if n < cols else cols
    # the mbarriers: ONE lane initialises, the fence + CTA barrier publish them, the cluster barrier makes them visible to every peer
    # before any remote store; the wait overlaps the loads in flight
    if tidx == zero:
        for k in cutlass.range_constexpr(5):
            cute.arch.mbarrier_init(s_mb.iterator + k, LOCAL_ARRIVERS)
    # the fence by EVERY thread (the order: ONE lane inits -> fence -> CTA barrier -> cluster barrier -> the first remote store); it is
    # a per-thread fence, so for the threads that initialised nothing it orders nothing and costs one instruction
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()  # P0
    cute.arch.cluster_arrive_relaxed(aligned=True)
    cute.arch.cluster_wait()
    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()

    if n <= cutlass.Int32(top_k):
        if rank == zero:
            for j in range(tidx, top_k, threads_per_cta):
                val = n - one - j
                val = val if j < n else cutlass.Int32(-1)
                st_global(out_row + j.to(cutlass.Int64) * four, val, cutlass.Int32)
    else:
        keys = []
        kmax_t = cutlass.Uint32(0)
        for i in cutlass.range_constexpr(ept):
            bits = raw[i].to(cutlass.Uint32)
            sign = bits >> cutlass.Uint32(31)
            flip = (cutlass.Uint32(0) - sign) | cutlass.Uint32(0x80000000)
            key = bits ^ flip
            e = e0 + cutlass.Int32(32 * i)
            key = key if e < n else cutlass.Uint32(0)
            keys.append(key)
            kmax_t = kmax_t if kmax_t > key else key
        kmax_w = cute.arch.warp_redux_sync(kmax_t, "umax")
        if lane == zero:
            s_wmax[warp] = kmax_w
        for j in cutlass.range_constexpr(bpt):
            s_hist[cutlass.Int32(j * threads_per_cta) + tidx] = zero
        if tidx == zero:
            s_ncand[0] = zero
        cute.arch.barrier()  # B0
        idxm = lane if lane < cutlass.Int32(nwarps) else zero
        vm = s_wmax[idxm]
        vm = vm if lane < cutlass.Int32(nwarps) else cutlass.Uint32(0)
        kmax_l = cute.arch.warp_redux_sync(vm, "umax")
        # the cluster-wide maximum: thread t < C pushes the local maximum into peer t's slot `rank`
        if tidx < cutlass.Int32(C):
            cute.arch.store_async_dsmem(s_xmax.iterator + rank, kmax_l.to(cutlass.Int32), mb_max, tidx)
        if tidx == zero:
            cute.arch.mbarrier_arrive_and_expect_tx(mb_max, C * XCHG_WORD_BYTES)
        cute.arch.mbarrier_wait(mb_max, 0)
        kmax = cutlass.Uint32(0)
        for j in cutlass.range_constexpr(C):
            vj = s_xmax[j].to(cutlass.Uint32)
            kmax = kmax if kmax > vj else vj
        top = kmax >> cutlass.Uint32(WINDOW_SHIFT)
        base = (top - cutlass.Uint32(BINS - 1)) if top > cutlass.Uint32(BINS - 1) else cutlass.Uint32(0)
        n_dummy = cutlass.Int32(cols_cap) - n  # every CTA's forced-0 keys together, all in digit 0 of pass 1
        wbase = base.to(cutlass.Int32)
        for i in cutlass.range_constexpr(ept):
            d = (keys[i] >> cutlass.Uint32(WINDOW_SHIFT)).to(cutlass.Int32) - wbase
            d = d if d > zero else zero
            d = d if d < cutlass.Int32(BINS - 1) else cutlass.Int32(BINS - 1)
            cute.arch.atomic_add(s_hist.iterator + d, one, sem="relaxed", scope="cta")
        for j in cutlass.range_constexpr(bpt):
            s_hist[cutlass.Int32(BINS + j * threads_per_cta) + tidx] = zero
        cute.arch.barrier()  # B1_1
        mph = zero  # the merge phase counter (mb_rs / mb_ag parity)
        _merge_hist(s_hist, 0, s_rs, s_mslice, s_merged, mb_rs, mb_ag, mph & one, rank, C, tidx)
        mph = mph + one
        corr = n_dummy if tidx == cutlass.Int32(threads_per_cta - 1) else zero
        _scan_find(s_merged, 0, s_wsum, s_found, cutlass.Int32(top_k), corr, tidx, lane, warp, threads_per_cta)
        b = s_found[0]
        above_b = s_found[1]
        cnt_b = s_found[2]
        need = cutlass.Int32(top_k) - above_b
        bu = (b + wbase).to(cutlass.Uint32)
        width_m1 = cutlass.Uint32((1 << WINDOW_SHIFT) - 1)
        new_lo = bu << cutlass.Uint32(WINDOW_SHIFT)
        saturated = (b == zero) & (wbase > zero)
        lo = cutlass.Uint32(0) if saturated else new_lo
        hi = new_lo + width_m1
        whole = cnt_b == need
        kstar = (lo - cutlass.Uint32(1)) if lo > cutlass.Uint32(0) else cutlass.Uint32(0)
        idstar = n - one
        if not whole:
            # this CTA's candidates of the k-th bin (global ids), the local list
            span1 = hi - lo
            for i in cutlass.range_constexpr(ept):
                key = keys[i]
                inb = (key - lo) <= span1
                if lo == cutlass.Uint32(0):
                    inb = inb & ((e0 + cutlass.Int32(32 * i)) < n)
                w_c = cutlass.Uint32(cute.arch.vote_ballot_sync(inb))
                if w_c != cutlass.Uint32(0):
                    wbase_c = zero
                    if lane == zero:
                        wbase_c = cute.arch.atomic_add(s_ncand.iterator, cutlass.Int32(cute.arch.popc(w_c)), sem="relaxed", scope="cta")
                    wbase_c = cute.arch.shuffle_sync(wbase_c, zero, mask_and_clamp=31)
                    slot = wbase_c + cutlass.Int32(cute.arch.popc(w_c & lt_mask))
                    if inb & (slot < cutlass.Int32(CAND_CAP)):
                        s_cand[slot] = key
                        s_cand_id[slot] = e0 + cutlass.Int32(32 * i)
            cute.arch.barrier()  # C1
            nloc = s_ncand[0]
            # the union message: word 0 = this CTA's TRUE count, words 2 + 2j / 3 + 2j = (key, id) of entry j < min(count, UNION_CAP)
            if tidx < cutlass.Int32(UNION_CAP):
                if tidx < nloc:
                    s_msg[cutlass.Int32(2) + cutlass.Int32(2) * tidx] = s_cand[tidx]
                    s_msg[cutlass.Int32(3) + cutlass.Int32(2) * tidx] = s_cand_id[tidx].to(cutlass.Uint32)
            if tidx == zero:
                s_msg[0] = nloc.to(cutlass.Uint32)
                s_msg[1] = cutlass.Uint32(0)
            cute.arch.barrier()  # C2: the message
            # the all-gather of the union: thread t pushes chunk t % 64 of the message to peer t / 64 (C x 64 threads busy)
            if tidx < cutlass.Int32(C * msg_chunks):
                q = tidx // cutlass.Int32(msg_chunks)
                c = tidx - q * cutlass.Int32(msg_chunks)
                m0 = s_msg[cutlass.Int32(4) * c].to(cutlass.Int32)
                m1 = s_msg[cutlass.Int32(4) * c + cutlass.Int32(1)].to(cutlass.Int32)
                m2 = s_msg[cutlass.Int32(4) * c + cutlass.Int32(2)].to(cutlass.Int32)
                m3 = s_msg[cutlass.Int32(4) * c + cutlass.Int32(3)].to(cutlass.Int32)
                dstu = cute.make_ptr(
                    cutlass.Int32,
                    s_union.iterator.toint() + (rank * cutlass.Int32(MSG_WORDS) + cutlass.Int32(4) * c) * cutlass.Int32(4),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                cute.arch.store_async_dsmem(dstu, (m0, m1, m2, m3), mb_un, q)
            if tidx == zero:
                cute.arch.mbarrier_arrive_and_expect_tx(mb_un, C * UNION_MSG_BYTES)
            cute.arch.mbarrier_wait(mb_un, 0)
            # the union's counts: every CTA derives the same decision
            total = zero
            overflow = cutlass.Boolean(False)
            for j in cutlass.range_constexpr(C):
                cj = s_union[cutlass.Int32(j * MSG_WORDS)].to(cutlass.Int32)
                total = total + cj
                overflow = overflow | (cj > cutlass.Int32(UNION_CAP))
            if not overflow:
                # the union compacted into the local list (identical in every CTA): entry j of sender r at prefix_r + j
                pre = zero
                for r in cutlass.range_constexpr(C):
                    cr = s_union[cutlass.Int32(r * MSG_WORDS)].to(cutlass.Int32)
                    for j in range(tidx, cr, threads_per_cta):
                        s_cand[pre + j] = s_union[cutlass.Int32(r * MSG_WORDS + 2) + cutlass.Int32(2) * j]
                        s_cand_id[pre + j] = s_union[cutlass.Int32(r * MSG_WORDS + 3) + cutlass.Int32(2) * j].to(cutlass.Int32)
                    pre = pre + cr
                cute.arch.barrier()  # C3: the union list
                ncand = total
                if ncand <= cutlass.Int32(RANK_CAP):
                    if tidx < ncand:
                        key_c = s_cand[tidx]
                        id_c = s_cand_id[tidx]
                        rnk = zero
                        for j in range(ncand):
                            kj = s_cand[j]
                            ij = s_cand_id[j]
                            ahead = (kj > key_c) | ((kj == key_c) & (ij < id_c))
                            rnk = rnk + (one if ahead else zero)
                        if rnk == need - one:
                            s_wmax[0] = key_c
                            s_found[1] = id_c
                    cute.arch.barrier()  # R1
                    kstar = s_wmax[0]
                    idstar = s_found[1]
                else:
                    # the narrowing passes over the union list, local to every CTA (identical data -> identical verdict)
                    done = cutlass.Boolean(False)
                    take = zero
                    for p0 in cutlass.range_constexpr(MAX_PASSES - 1):
                        p = p0 + 1
                        if not done:
                            hbuf = cutlass.const_expr((p & 1) * BINS)
                            span = hi - lo
                            blen = cutlass.Int32(32) - cute.arch.clz(span).to(cutlass.Int32)
                            nsh = blen - cutlass.Int32(BIN_BITS)
                            sh = (nsh if nsh > zero else zero).to(cutlass.Uint32)
                            span_p = hi - lo
                            for c in range(tidx, ncand, threads_per_cta):
                                key = s_cand[c]
                                if (key - lo) <= span_p:
                                    d = ((key - lo) >> sh).to(cutlass.Int32)
                                    cute.arch.atomic_add(s_hist.iterator + (cutlass.Int32(hbuf) + d), one, sem="relaxed", scope="cta")
                            if cutlass.const_expr(p + 1 < MAX_PASSES):
                                nbuf = cutlass.const_expr(((p + 1) & 1) * BINS)
                                for j in cutlass.range_constexpr(bpt):
                                    s_hist[cutlass.Int32(nbuf + j * threads_per_cta) + tidx] = zero
                            cute.arch.barrier()  # B1_p
                            _scan_find(s_hist, hbuf, s_wsum, s_found, need, zero, tidx, lane, warp, threads_per_cta)
                            b = s_found[0]
                            above_b = s_found[1]
                            cnt_b = s_found[2]
                            need = need - above_b
                            new_lo = lo + (b.to(cutlass.Uint32) << sh)
                            hi = new_lo + ((cutlass.Uint32(1) << sh) - cutlass.Uint32(1))
                            lo = new_lo
                            done = (cnt_b == need) | (lo == hi)
                            take = need
                    kstar = (lo - cutlass.Uint32(1)) if lo > cutlass.Uint32(0) else cutlass.Uint32(0)
                    idstar = n - one
                    if (lo == hi) & (take < cnt_b):
                        # the take-th smallest id among the union's entries with the exact key: an all-pairs rank over ids (<= CAND_CAP).
                        # The union holds up to C x UNION_CAP = 2032 entries at 16 CTAs -- more than the CTA's threads -- so the owner
                        # loop STRIDES over the whole list: an entry past slot 1023 owned by no thread left the fourth word unwritten and
                        # the compaction read a stale cutoff (every output slot kept its sentinel; the 2026-10-22 review's finding)
                        kstar = lo
                        for c in range(tidx, ncand, threads_per_cta):
                            key_c = s_cand[c]
                            id_c = s_cand_id[c]
                            if key_c == kstar:
                                rnk = zero
                                for j in range(ncand):
                                    rnk = rnk + (one if ((s_cand[j] == kstar) & (s_cand_id[j] < id_c)) else zero)
                                if rnk == take - one:
                                    # the FOURTH word, read nowhere else: words 0..2 were read as (b, above, count) by every thread
                                    # right after the last pass's B3 with no barrier since, so a write into word 1 here would race a
                                    # reader still behind that read (the 2026-10-21 barrier audit's finding)
                                    s_found[3] = id_c
                        cute.arch.barrier()  # R2
                        idstar = s_found[3]
            else:
                # a CTA holds more candidates than the union carries: the merged passes over the row's keys (every CTA, every pass),
                # then -- for an exact key with a partial take -- the merged 2 x 12-bit select over the ids of the keys equal to it
                done = cutlass.Boolean(False)
                take = zero
                for p0 in cutlass.range_constexpr(MAX_PASSES - 1):
                    p = p0 + 1
                    if not done:
                        hbuf = cutlass.const_expr((p & 1) * BINS)
                        span = hi - lo
                        blen = cutlass.Int32(32) - cute.arch.clz(span).to(cutlass.Int32)
                        nsh = blen - cutlass.Int32(BIN_BITS)
                        sh = (nsh if nsh > zero else zero).to(cutlass.Uint32)
                        span_p = hi - lo
                        for i in cutlass.range_constexpr(ept):
                            key = keys[i]
                            if ((key - lo) <= span_p) & ((e0 + cutlass.Int32(32 * i)) < n):
                                d = ((key - lo) >> sh).to(cutlass.Int32)
                                cute.arch.atomic_add(s_hist.iterator + (cutlass.Int32(hbuf) + d), one, sem="relaxed", scope="cta")
                        if cutlass.const_expr(p + 1 < MAX_PASSES):
                            nbuf = cutlass.const_expr(((p + 1) & 1) * BINS)
                            for j in cutlass.range_constexpr(bpt):
                                s_hist[cutlass.Int32(nbuf + j * threads_per_cta) + tidx] = zero
                        cute.arch.barrier()  # B1_p
                        _merge_hist(s_hist, hbuf, s_rs, s_mslice, s_merged, mb_rs, mb_ag, mph & one, rank, C, tidx)
                        mph = mph + one
                        _scan_find(s_merged, 0, s_wsum, s_found, need, zero, tidx, lane, warp, threads_per_cta)
                        b = s_found[0]
                        above_b = s_found[1]
                        cnt_b = s_found[2]
                        need = need - above_b
                        new_lo = lo + (b.to(cutlass.Uint32) << sh)
                        hi = new_lo + ((cutlass.Uint32(1) << sh) - cutlass.Uint32(1))
                        lo = new_lo
                        done = (cnt_b == need) | (lo == hi)
                        take = need
                kstar = (lo - cutlass.Uint32(1)) if lo > cutlass.Uint32(0) else cutlass.Uint32(0)
                idstar = n - one
                if (lo == hi) & (take < cnt_b):
                    kstar = lo
                    id_lo = zero
                    for q in cutlass.range_constexpr(2):
                        id_sh = cutlass.const_expr(12 * (1 - q))
                        hbuf = cutlass.const_expr((q & 1) * BINS)
                        for j in cutlass.range_constexpr(bpt):
                            s_hist[cutlass.Int32(hbuf + j * threads_per_cta) + tidx] = zero
                        cute.arch.barrier()  # I0_q
                        for i in cutlass.range_constexpr(ept):
                            e = e0 + cutlass.Int32(32 * i)
                            d = cutlass.Int32(BINS - 1) - (((e - id_lo) >> cutlass.Int32(id_sh)) & cutlass.Int32(BINS - 1))
                            if (keys[i] == kstar) & (e < n) & (e >= id_lo) & (e < id_lo + cutlass.Int32(BINS << id_sh)):
                                cute.arch.atomic_add(s_hist.iterator + (cutlass.Int32(hbuf) + d), one, sem="relaxed", scope="cta")
                        cute.arch.barrier()  # I1_q
                        _merge_hist(s_hist, hbuf, s_rs, s_mslice, s_merged, mb_rs, mb_ag, mph & one, rank, C, tidx)
                        mph = mph + one
                        _scan_find(s_merged, 0, s_wsum, s_found, take, zero, tidx, lane, warp, threads_per_cta)
                        db = cutlass.Int32(BINS - 1) - s_found[0]
                        take = take - s_found[1]
                        id_lo = id_lo + (db << cutlass.Int32(id_sh))
                    idstar = id_lo + take - one

        # ---- the canonical compaction across the cluster: the per-CTA selected counts exchanged once ---------------------------
        wcnt = zero
        for i in cutlass.range_constexpr(ept):
            key = keys[i]
            e = e0 + cutlass.Int32(32 * i)
            sel = (key > kstar) | ((key == kstar) & (e <= idstar))
            bal = cutlass.Uint32(cute.arch.vote_ballot_sync(sel))
            if lane == zero:
                s_bal[warp * cutlass.Int32(ept) + cutlass.Int32(i)] = bal
            wcnt = wcnt + cutlass.Int32(cute.arch.popc(bal))
        if lane == zero:
            s_wsum[warp] = wcnt
        cute.arch.barrier()  # F
        wlow, ctot = _cross_warp_excl(s_wsum, lane, warp, nwarps)
        if tidx < cutlass.Int32(C):
            cute.arch.store_async_dsmem(s_cnt.iterator + rank, ctot, mb_cnt, tidx)
        if tidx == zero:
            cute.arch.mbarrier_arrive_and_expect_tx(mb_cnt, C * XCHG_WORD_BYTES)
        cute.arch.mbarrier_wait(mb_cnt, 0)
        sel_lt = zero
        for j in cutlass.range_constexpr(C):
            cj = s_cnt[j]
            sel_lt = sel_lt + (cj if cutlass.Int32(j) < rank else zero)
        run = sel_lt + wlow
        for i in cutlass.range_constexpr(ept):
            bal = s_bal[warp * cutlass.Int32(ept) + cutlass.Int32(i)]
            if bal != cutlass.Uint32(0):
                if (bal & my_bit) != cutlass.Uint32(0):
                    pos = run + cutlass.Int32(cute.arch.popc(bal & lt_mask))
                    slot = cutlass.Int32(top_k - 1) - pos
                    st_global(out_row + slot.to(cutlass.Int64) * four, e0 + cutlass.Int32(32 * i), cutlass.Int32)
                run = run + cutlass.Int32(cute.arch.popc(bal))
    # the exit drain: no CTA leaves while a peer's store into it may still be in flight (every peer's stores into THIS CTA completed
    # its own mbarrier phases above; the relaxed barrier orders the exits after every CTA passed its last wait)
    cute.arch.cluster_arrive_relaxed(aligned=True)
    cute.arch.cluster_wait()


frost_qsa_topk_decode_cluster.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def smem_bytes_for_cluster(cols_cap: int, cluster_ctas: int) -> int:
    """Dynamic shared memory of one CTA of the cluster form."""

    def al(x):
        return (x + 15) // 16 * 16

    slice_cap = cols_cap // cluster_ctas
    total = (
        al(2 * BINS * 4)
        + al(BINS * 4)
        + al(BINS * 4)
        + al(BINS // cluster_ctas * 4)
        + al(cluster_ctas * MSG_WORDS * 4)
        + al(MSG_WORDS * 4)
        + 2 * al(CAND_CAP * 4)
        + al(slice_cap // 32 * 4)
        + al(32 * 4)
        + al(32 * 4)
        + al(16 * 4)
        + al(16 * 4)
        + al(4 * 4)
        + al(4 * 4)
        + al(8 * 8)
    )
    return total + 128


@cute.jit
def qsa_topk_decode_launch(
    logits: cute.Tensor,
    counts: cute.Tensor,
    out: cute.Tensor,
    rows: cutlass.Int32,
    cols_cap: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    cluster_ctas: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    smem_bytes: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    if cutlass.const_expr(cluster_ctas > 1):
        kernel = frost_qsa_topk_decode_cluster(logits, counts, out, cols_cap, top_k, cluster_ctas, use_pdl)
        kernel.launch(
            grid=(cluster_ctas, rows, 1), block=(CLUSTER_THREADS, 1, 1), cluster=(cluster_ctas, 1, 1), smem=smem_bytes, stream=stream, use_pdl=use_pdl
        )
    else:
        kernel = frost_qsa_topk_decode(logits, counts, out, cols_cap, top_k, threads_per_cta, use_pdl)
        kernel.launch(grid=(1, rows, 1), block=(threads_per_cta, 1, 1), smem=smem_bytes, stream=stream, use_pdl=use_pdl)


class QsaTopkDecodeRecipe(NamedTuple):
    """Build-time facts of one artifact: everything is a compile-time constant of the kernel (the row count, the row
    strides and the column count enter as runtime tensor shapes -- one artifact serves every ``n_blocks_max <= cols_cap``)."""

    compiled: object
    cols_cap: int
    top_k: int
    cluster_ctas: int
    threads_per_cta: int
    use_pdl: bool
    smem_bytes: int
    device: int


def _fake_rows(dtype):
    """``[rows, cols]`` with a 64-BIT symbolic row stride (the contract takes any row stride and both kernels widen the row arithmetic to
    64 bits; a default-width symbolic stride made the binder refuse one past the int32 range) and a unit column stride, 16-B aligned base."""
    return cute.runtime.make_fake_tensor(dtype=dtype, shape=(cute.sym_int(), cute.sym_int()), stride=(cute.sym_int64(), 1), assumed_align=16)


def compile_qsa_topk_decode(
    *,
    cols_cap: int,
    top_k: int,
    cluster_ctas: int = DEFAULT_CLUSTER_CTAS,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    use_pdl: bool = False,
) -> QsaTopkDecodeRecipe:
    """Build the artifact from the geometry ALONE -- no device allocation, no launch.

    Order of the gates (Rule 7 first): the CuTe DSL floor, the geometry, the DSL's target for the current device, the device's
    cluster / DSMEM and PDL support (cc 9.0+), the cluster form's own DSL floor (``CLUSTER_FORM_MIN_DSL``: a device gate before a
    form's DSL gate, so a part below cc 9.0 is named whatever wheel it runs), the dynamic shared memory against the device's
    opt-in limit.  ``NotImplementedError`` names the version, the target or the limit; ``ValueError`` names the malformed geometry.
    """
    global _FAKE_STREAM
    too_old = cutedsl_requirement_error("qsa_topk_decode")
    if too_old is not None:
        raise NotImplementedError(too_old)
    cols_cap, top_k, cluster_ctas, threads_per_cta = int(cols_cap), int(top_k), int(cluster_ctas), int(threads_per_cta)
    validate_topk_geometry(cols_cap, top_k, cluster_ctas, threads_per_cta)
    device = current_device()
    cc = compute_capability(device)
    arch_problem = cutedsl_arch_requirement_error(cc)
    if arch_problem is not None:
        raise NotImplementedError(arch_problem)
    if cluster_ctas > 1 and cc < (9, 0):
        raise NotImplementedError(
            f"qsa_topk_decode with cluster_ctas = {cluster_ctas} needs thread-block clusters and DSMEM (cc 9.0+); found cc {cc[0]}.{cc[1]} -- use cluster_ctas = 1 there"
        )
    if use_pdl and cc < (9, 0):
        raise NotImplementedError(f"qsa_topk_decode with use_pdl=True needs griddepcontrol (cc 9.0+); found cc {cc[0]}.{cc[1]}")
    cluster_problem = cluster_form_requirement_error(cluster_ctas)  # after the device gates: an older part is named whatever wheel it runs
    if cluster_problem is not None:
        raise NotImplementedError(cluster_problem)
    smem_bytes = smem_bytes_for(cols_cap, cluster_ctas, threads_per_cta, top_k)
    limit = shared_memory_per_block_optin(device)
    if smem_bytes > limit:
        raise NotImplementedError(
            f"qsa_topk_decode at cols_cap = {cols_cap} with threads_per_cta = {threads_per_cta} needs {smem_bytes} B of shared memory per CTA; the device allows {limit} B -- the row is too wide for this device"
        )
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
    key = (cols_cap, top_k, cluster_ctas, threads_per_cta, bool(use_pdl), device)
    if key not in compiled_cache:
        # the in-process memo; behind it the persistent compiled-plan cache keyed by this file's digest + the key's plain values;
        # the compile runs with the plan device current (its context), so the DSL's compile-target auto-detect sees the device the
        # constants above were taken from -- a build_device() scope or the tensors' device of the convenience entry, not whichever
        # GPU the caller left current
        cache_key = template_key(globals(), {"key": key}, "compile_qsa_topk_decode")
        with torch.cuda.device(device):
            compiled_cache[key] = _compile_for(cache_key, cols_cap, top_k, cluster_ctas, threads_per_cta, bool(use_pdl), smem_bytes)
    return QsaTopkDecodeRecipe(compiled_cache[key], cols_cap, top_k, cluster_ctas, threads_per_cta, bool(use_pdl), smem_bytes, device)


def _compile_for(cache_key, cols_cap, top_k, cluster_ctas, threads_per_cta, use_pdl, smem_bytes):
    """The one ``compile_cached`` call of :func:`compile_qsa_topk_decode` (split out so the device scope around it reads as one line)."""
    return compile_cached(
        qsa_topk_decode_launch,
        _fake_rows(cutlass.Int32),  # the fp32 logits enter as their int32 bits (a view at the launch boundary)
        cute.runtime.make_fake_compact_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride_order=(0,), assumed_align=4),
        _fake_rows(cutlass.Int32),
        cutlass.Int32(1),  # rows: runtime, the literal pins the type
        cols_cap,
        top_k,
        cluster_ctas,
        threads_per_cta,
        use_pdl,
        smem_bytes,
        _FAKE_STREAM,
        options="--enable-tvm-ffi",
        cache_key=cache_key,
        symbol="cudnn_qsa_topk_decode",
    )


def _tensor_device_index(x: torch.Tensor) -> int:
    """The CUDA ordinal a tensor lives on (a device without an index = the current one)."""
    return x.device.index if x.device.index is not None else torch.cuda.current_device()


def _check_rows(name: str, x: torch.Tensor, *, dtype, rows: int, device) -> None:
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError(f"{name} must be a 2-D tensor, got {None if not isinstance(x, torch.Tensor) else tuple(x.shape)}")
    if x.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {x.dtype}")
    if not x.is_cuda or x.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on {device}, got {x.device}")
    if int(x.shape[0]) != rows:
        raise ValueError(f"{name} must hold {rows} rows (one per query row), got shape {tuple(x.shape)}")
    if x.stride(1) != 1:
        raise ValueError(f"{name} needs a unit column stride, got strides {tuple(x.stride())}")
    if x.data_ptr() % 16 != 0:
        raise ValueError(f"{name} must be 16-B aligned")


def run_qsa_topk_decode(r: QsaTopkDecodeRecipe, logits: torch.Tensor, counts: torch.Tensor, block_ids: torch.Tensor, *, stream) -> None:
    """The lowered launch: form checks and the launch -- no conversion, no allocation, no torch op (Rules 1 / 3 / 5).

    ``logits [T, n_blocks_max]`` fp32 with ``n_blocks_max <= r.cols_cap``, ``counts [T]`` int32 contiguous, ``block_ids
    [T, r.top_k]`` int32 (unit column stride; any row stride), all on the recipe's device, ``T <= MAX_ROWS`` (one CTA per row
    along ``grid.y``); ``stream`` is the raw handle the kernel launches on (``None`` = torch's current stream of that device).
    """
    if not isinstance(counts, torch.Tensor) or counts.ndim != 1 or counts.dtype != torch.int32 or not counts.is_cuda or not counts.is_contiguous():
        raise ValueError("counts must be a contiguous 1-D int32 CUDA tensor")
    # the recipe's device is the plan device: current for the form checks (a refusal restores the caller's device the same way),
    # the stream selection (``None`` = THAT device's current stream, whose raw handle may be the default-stream sentinel) and the
    # launch (the prepared host enters with the plan device's context, whichever GPU the caller left current)
    with torch.cuda.device(r.device):
        rows = int(counts.shape[0])
        if rows > MAX_ROWS:
            raise ValueError(
                f"counts holds {rows} rows; the launch grid caps the row count at {MAX_ROWS} (one CTA per row along grid.y) -- split the row block"
            )
        device = counts.device
        _check_rows("logits", logits, dtype=torch.float32, rows=rows, device=device)
        _check_rows("block_ids", block_ids, dtype=torch.int32, rows=rows, device=device)
        n_blocks_max = int(logits.shape[1])
        if n_blocks_max < 1 or n_blocks_max > r.cols_cap:
            raise ValueError(f"logits holds {n_blocks_max} columns; this artifact was compiled for cols_cap = {r.cols_cap} (1 <= n_blocks_max <= cols_cap)")
        if int(block_ids.shape[1]) != r.top_k:
            raise ValueError(f"block_ids must be [T, top_k = {r.top_k}] for this artifact, got shape {tuple(block_ids.shape)}")
        dev_index = _tensor_device_index(counts)
        if dev_index != r.device:
            raise ValueError(f"the tensors live on device {dev_index} but the artifact was compiled for device {r.device} (compile per device)")
        if rows == 0:
            return
        handle = int(stream) if stream is not None else torch.cuda.current_stream(r.device).cuda_stream
        r.compiled(logits.view(torch.int32), counts, block_ids, cutlass.Int32(rows), cuda.CUstream(handle))


def qsa_topk_decode(
    logits: torch.Tensor,
    counts: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    cluster_ctas: Optional[int] = None,
    threads_per_cta: Optional[int] = None,
    use_pdl: bool = False,
    stream=None,
) -> torch.Tensor:
    """Compile (cached) and run once -- the convenience form for the standalone selector, the tests and the benchmarks.

    ``block_ids [T, top_k]`` is the CALLER's buffer (nothing is allocated here); it is returned for convenience.  A knob left
    ``None`` follows :func:`default_knobs_for` (the cluster form for rows of more than 8192 blocks at up to 32 rows, one CTA per row
    otherwise); every served value is the same function.
    """
    too_old = cutedsl_requirement_error("qsa_topk_decode")
    if too_old is not None:
        raise NotImplementedError(too_old)
    if not isinstance(block_ids, torch.Tensor) or block_ids.ndim != 2:
        raise ValueError("block_ids must be a [T, top_k] int32 CUDA tensor")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or not logits.is_cuda:
        raise ValueError("logits must be a [T, n_blocks_max] fp32 CUDA tensor")
    # the plan device is the TENSORS' device (``logits``; the lowered launch checks the other two against it), not whichever GPU
    # is current: the knob pick's capability, the compile and the launch all run with it current, the caller's device restored
    # after -- also when a form check refuses
    with torch.cuda.device(_tensor_device_index(logits)):
        if cluster_ctas is None:
            rows = int(counts.shape[0]) if isinstance(counts, torch.Tensor) and counts.ndim == 1 else 0
            auto_c, auto_t = default_knobs_for(rows, int(logits.shape[1]), compute_capability(current_device()))
            # an explicit thread count other than the cluster form's keeps the single-CTA form (the knobs are a pair)
            cluster_ctas = auto_c if (threads_per_cta is None or int(threads_per_cta) == CLUSTER_THREADS) else 1
            threads_per_cta = auto_t if threads_per_cta is None else int(threads_per_cta)
        else:
            cluster_ctas = int(cluster_ctas)
            threads_per_cta = (CLUSTER_THREADS if cluster_ctas > 1 else DEFAULT_THREADS_PER_CTA) if threads_per_cta is None else int(threads_per_cta)
        r = compile_qsa_topk_decode(
            cols_cap=cols_cap_for(int(logits.shape[1]), cluster_ctas, threads_per_cta),
            top_k=int(block_ids.shape[1]),
            cluster_ctas=cluster_ctas,
            threads_per_cta=threads_per_cta,
            use_pdl=use_pdl,
        )
        run_qsa_topk_decode(r, logits, counts, block_ids, stream=stream)
    return block_ids
