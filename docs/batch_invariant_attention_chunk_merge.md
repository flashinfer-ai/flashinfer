<!-- SPDX-License-Identifier: Apache-2.0 -->
# Batch-invariant attention: the one-warpgroup body's chunk-end merge, measured

A design note for a decision. The one-warpgroup row body (the prefill default at D = 128 and, below GQA group 4, at D = 64)
folds every 512-key chunk from the init state and merges the finished chunk into the running state at the chunk end:
`O_m <- fma(O_c, a2, mul(O_m, a1))` per element, in index order, by the correction warpgroup, with `O_m` in TMEM at
D <= 128 (the "om" plan: `S 128 | P 64 | O_c D | O_m D`) and in a per-CTA global scratch at D = 256. An earlier note
estimated this merge at ~3400 cycles per chunk, unpipelined, ~5 % of the body at D = 128, and proposed hiding it. This note
measures it (per-role `clock64` timelines in a probe build, then unprobed CUPTI ablations that bound every merge placement),
finds the estimate wrong in both size and mechanism, and lands the one placement that does pay -- in the epilogue, not the
interior merge. Everything measured is bit-identical to the previous head (same fp32 operations in the same order; the
dumps of the prefill and decode grid compare equal; the suite passes). Rubin (sm_107, bf16), CUPTI kernel sums, cold L2,
30 iterations, two ABAB passes agreeing within 1 %; ratios are previous head / variant, so > 1 is faster.

## 1. The structure (what the code does)

Per tile the MMA warp issues `QK(n+1)` then `PV(n)` (one S stage, P in its own TMEM columns, so the softmax of tile n
overlaps the tensor core). `PV(n)` acquires the single-stage accumulator pipeline, which the correction warpgroup releases
after rescaling `O_c` by tile n's alpha (needs `PV(n-1)` complete). At a chunk end the correction waits for the chunk's
last PV, merges `O_c` into `O_m` and releases; the next chunk's first PV (`ACCUMULATE` off) may only then overwrite `O_c`.
The merge moves `D / CT` pieces of `CT` columns (32 at the 128-register correction of D = 128, 16 at the 96-register one of
D = 64): per piece a `tcgen05.ld` of the `O_c` piece, one of the `O_m` piece, 16 + 16 packed `mul` / `fma`, a `tcgen05.st`.
The SASS shows the two loads of a piece back to back, the next piece's loads after the previous piece's store (ptxas keeps
TMEM loads and stores in program order), and the hardware scoreboard -- not an explicit wait -- ordering each load's
registers: four TMEM round trips of ~250 cycles plus the math, one piece in flight, because the two 32-register buffers hold
exactly one piece of each operand. The softmax has two tiles of slack over the merge: `S(n+1)` is issued before `PV(n)`
waits, so while the merge of chunk c runs the softmax computes tiles 4c + 4 and 4c + 5.

## 2. The critical path at a chunk end (probe build, CTA 0's first unit, 1x8K 32:8 paged, cycles)

D = 128, tiles 7 -> 8 (the chunk end), relative to the softmax's S-full of tile 7:

```
   0  sm[7].S_full
 ...  softmax(7): copy-out, max, exponentials, stats, P store
2511  sm[7].P_commit
2614  mma[7].P_seen            the MMA warp had QK(8) issued (S(8) ready at 2487) and was at its P wait
2738  mma[7].O_free            rescale(7) was released long before
3098  mma[7].PV_issued          (8 MMAs, ~430 cycles of issue at 32 registers)
3347  cor[7].mO_full            PV(7) complete, 249 after the last issue
3705  cor[7].m_piece0           piece 0: two loads, 32 packed ops, one store
4583  cor[7].m_stored           pieces 1-3, ~290 each
4677  cor[7].m_rel              fences + release: the MERGE IS 1293 CYCLES (not 3400)
----
3279  mma[8].top -> 4295 mma[8].QK+1_issued   QK(9) issued under the merge (S stage free since sm[8].S_rel)
4394  sm[8].P_commit            softmax(8) ran under the merge
5105  mma[8].P_seen             the MMA warp saw P(8) 711 after its commit (its own loop, no barrier wait)
5239  mma[8].O_free             134 after P_seen: the SAME wait as every mid-chunk tile (154 vs 153 median)
5671  mma[8].PV_issued
```

The merge never gates the MMA: its release (4677) precedes the MMA's acquire (5239) by more than the MMA's own P wait. The
softmax S-full period by chunk position is 2419 / 2427 / 2380 / 2384 at D = 128 -- no chunk-end excess at all in the first
unit; in a steady-state unit (the third of CTA 0) 3236 / 3164 / 3154 / 3190 with the tile after a chunk end +50..+80, which the
per-phase medians locate in the softmax's exponential phase (1591 vs ~1380 cycles while the merge's 180 instructions per thread
share its SMSP issue slots), not in a wait. D = 64: merge 749 cycles (CT = 16), chunk-position periods 2270 / 2110 / 2107 / 2090 --
the +160 again in the exponential phase of the overlapped tile (1412 vs 1250), the MMA's O wait after a chunk end 152 vs 150.
D = 256 (O_m through the L2 scratch, 8 pieces, 3 scratch loads in flight): merge 7845 cycles, the MMA's O wait after a chunk
end 2560 vs 168 -- that merge IS on the chain, and its 256 KB of L2 traffic per chunk per CTA is a third of the body's L2
traffic (the K / V stream is 512 KB per chunk per CTA).

Probe caveats (they decide nothing below): the stamps slow the probed roles (period 2402 vs ~1520 unprobed at D = 128), and
the inherited probe sent every other CTA's stamps to one global line, which inflated later-wave periods until fixed.

## 3. The ceiling: no-merge ablations (unprobed, wrong bits, timing only)

The interior merge replaced by its waits, fences and release alone (`O_m` never updated), paged 32:8, one-warpgroup body:

| D | 1x1K | 1x8K | 8x2K | 32x1K | 64x1K | geomean |
|---|---|---|---|---|---|---|
| 128 | 1.008 | 1.026 | 1.022 | 1.009 | 1.006 | 1.014 |
| 64 | 1.015 | 1.025 | 1.033 | 1.038 | 1.037 | 1.030 |
| 256 | 0.974 | 1.276 | 1.087 | 1.062 | 1.077 | 1.091 |

Deleting the interior merge outright is worth 1.4 % at D = 128: no placement of it can reach the 2 % bar. The same merge with
its accumulator released BEFORE it runs (its instructions kept, its chain position removed): D = 128 0.998, D = 64 0.999 --
nothing of the 1.4 % / 3.0 % is chain exposure; all of it is the merge's instruction stream on the SMSPs the softmax warps
issue from (one softmax warp and one correction warp per SMSP), ~160 cycles per chunk at D = 128 = ~180 instructions per
thread. The packed `mul` + `fma` per element are the class; the loads, stores and address arithmetic are ~50 of the 180.
At D = 256 the merge is both on the chain and a third of the L2 traffic of a body that is L2-bandwidth bound: 27 % at 1x8K.

## 4. Placements tried

* (a) A second `O_c` buffer in TMEM so the next chunk's PV need not wait for the merge: impossible at D = 128 (448 of 512
  columns used) and pointless at D = 64 (the chain is not where the cost is; the instructions would still issue).
* (b) / (c) Merging or releasing `O_c` piece by piece: every PV instruction (128 x D x 16) writes all D columns, so no column
  of `O_c` can be released before the whole merge has read it; the `O_m` side can move, the `O_c` side cannot.
* (d) Prefetching the `O_m` piece (and its `mul` by a1) before the wait for the chunk's last PV, so one of the four round
  trips leaves the chain; at D = 256 the first three scratch pieces hoisted. Bit-identical, and 10 % SLOWER at D = 128,
  26 % at D = 256: holding one 32-register piece live across the pipeline wait makes ptxas spill the correction's whole tile
  loop (LDL / STL 5 / 6 -> 56 / 165 at 128 registers, 86 / 39 -> 214 / 185 at 208; the D = 256 correction already spills in
  the base). At D = 64 (16-column pieces) it is spill-free and still 1-2 % slower: more instructions in the interfering stream.
  The register file is the binding constraint of every deeper pipelining of this merge, as it was for a third rescale buffer.
* Fewer buffers at D = 256 (3 instead of 4, to stop the base's spills): 94 / 48 LDL / STL -- worse.
* KEPT: two TMEM loads in flight where the correction has both buffers free -- the unit's first merge (`fma(O_c, a2, +0)`:
  no `O_m` operand) and the single-chunk epilogue copy `O_m = O_c` -- two round trips instead of four. Spill-free, bits
  identical. Paged 32:8: D = 128 1x1K 1.023, 1x8K 1.000, 8x2K 1.039, 32x1K 1.051, 64x1K 1.052 (geomean 1.033); D = 64 1.013 /
  1.005 / 1.014 / 1.040 / 1.044 (geomean 1.023). It beats the interior-merge ceiling because its gain is elsewhere: the
  correction's epilogue (merge or copy, release, normalise, stores: 4660 cycles in the boundary probe) is on the unit-boundary
  chain of short units. The boundary probe (8-tile units, CTA 0's third and fourth units) shows the softmax idle 2870 cycles
  between one unit's last P commit and the next unit's first S full, and the next unit's first tile at 5317 cycles against
  3069 for a mid-unit tile while the correction runs the epilogue on the shared SMSPs.
  Over the full grid (paged and ragged at 32:8 and 8:8, decode 8x8K / 64x8K): D = 128 prefill geomean 1.034 paged and 1.034 ragged
  over ten cells each (worst cell 1.001; 32x1K / 64x1K 1.06-1.07), decode 1.004; D = 64 (one-warpgroup body) 1.030 / 1.023, decode
  1.000. Suite 107 passed; mypy clean.
* Measured and dropped: the same two-in-flight first merge for the paired placement's smem `O_m` (the D = 64 prefill default from
  GQA group 4). Paged paired D = 64 1x1K 1.024, 1x8K 1.010, 8x2K 1.022, 32x1K 1.018, 64x1K 1.026 -- but the RAGGED paired kernel,
  whose 96-register correction is at its limit (8-column pieces), takes 13 more spills outside its tile loop and measures 1.015 /
  0.989 / 0.961 / 0.994 / 0.990 (geomean 0.990): a cell 4 % slower, so not kept. At D = 256 the same form measured 0.992-1.000 (an
  L2-bound body gains nothing from the TMEM side). The patch is in the job directory should the ragged compile's budget change.
* Measured and dropped: the epilogue's normalise loop and the mode-1 workspace store with two loads in flight -- within noise of the
  kept placement at D = 128 and D = 64 (the normalise runs after the release; the softmax of the next unit is already under way).

## 5. Recommendation

The interior chunk-end merge of the one-warpgroup body is closed as a lever at D = 128 and D = 64: it is 1216-1325 / 749-844
cycles, hidden by the softmax's two-tile slack, and its visible cost is its instruction count, which the class fixes. The
D = 256 merge is the body's largest single cost at long rows (27 % at 1x8K) and a different problem: the home of `O_m`.
TMEM is full there (`O_c 256 | S0 128 | S1 128`), the smem plan (Q 64 KB + ring 128 KB + payloads) leaves no 128 KB under the
part's capacity, and the footprint rule forbids the growth; a longer C is a class change. Open for the lead.
The unit boundary is the lever these measurements expose: issuing the next unit's `QK(0)` before this unit's last PV (the
S stage is free once the softmax copied S out, Q and K(0) are loaded a unit ahead) would start the next unit's softmax ~2900
cycles earlier (probe build) on every boundary -- a placement (issue order of independent MMAs; the ring order follows), worth
most on the short-unit cells (32x1K, 64x1K, 8x2K), and not touched here.
