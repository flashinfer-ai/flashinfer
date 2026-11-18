<!-- SPDX-License-Identifier: Apache-2.0 -->
# Batch-invariant attention: the one-warpgroup body's unit boundary, measured

A design note for a decision. The one-warpgroup row body (the prefill default at D = 128 and, below GQA group 4, at D = 64)
walks its work units one after the other; on the 1K-row multi-request prefill cells (8x2K, 32x1K, 64x1K) a unit is one to
eight 128-key tiles and the class runs at 0.61-0.68 of the fastest backend. The previous note (the chunk-end merge) left one
lever open: at a unit boundary the softmax warpgroup idled ~2900 cycles between its last P commit and the next unit's first
S, and the next unit's first tile ran at 5317 cycles against 3069 for a steady tile while the correction warpgroup ran the
epilogue. This note instruments the boundary, finds that most of the boundary's cycles were not the idle but the softmax's
own causal mask on the unit's last tile, lands the mask's lowering, builds and measures the overlap of the next unit's first
fold with this unit's last, and rejects it with its mechanism. Everything landed is bit-identical to the previous head (the
same fp32 operations in the same order; the dumps of the prefill and decode grid compare equal; the suite passes). Rubin
(sm_107, bf16), CUPTI kernel sums, cold L2, 30 iterations, two ABAB passes agreeing within 1 %; ratios are previous head /
variant, so > 1 is faster.

## 1. The boundary instrumented (probe build, CTA 0's third and fourth units of 32x1K 32:8 paged, 8 tiles each, D = 128)

Per-role `clock64` stamps (the q2c probe rebased onto the previous head). Cycles relative to the softmax's S-full of unit
A's last tile (A7, the causal diagonal tile of its rows):

```
  -195  mma[A6].QK+1_issued = QK(A7)         0  sm[A7].S_full       295  sm[A7].S_rel        the S stage is FREE from here
  1373  mma[A6].PV_issued                 1996  mma[A7].top        2861  mma[A7].V_land      V(7) in smem; the MMA warp waits for P(7)
  3682  sm[A7].alpha_sent                 <- 3387 cycles between S_rel and alpha on A7, against 310-415 on A1..A6: THE MASK
  5188  sm[A7].stats_sent  5707  sm[A7].P_commit  5809  mma[A7].P_seen  6009  mma[A7].O_free  6593  mma[A7].PV_issued
  5744  cor[A7].epi_top    6848  cor[A7].epi_Ofull  10285  cor[A7].epi_end   the epilogue: copy / merge, release, normalise, stores
  6317  sm[B0].top (idle from here)        8047  mma[B0].top   QK(B0) issued between 6770 and ~7900 (descriptor loads, Q wait,
                                                               K(0) wait, S acquire, eight MMAs at 32 registers)
  8154  sm[B0].S_full    softmax gap A.P_commit -> B.S_full = 2447; B0's period 4651 against 3056 mid-unit
```

The softmax's per-tile phases (cycles, S-full to P-commit): tiles A1..A6 2521-2751; A7 5707, of which the mask-and-maximum
phase 3387 against ~330 unmasked. The same at D = 64: 1974-2209 against 4155, the mask phase 2522. The critical path at the
boundary therefore ran: S(A7) copied out (295) -> the MASK (3387) -> exponentials -> P(A7) committed (5707) -> the MMA warp
sees P, acquires O, issues PV(A7) (6593) -> the unit transition in the MMA warp (descriptor loads, the Q and K(0) waits, the S
acquire: ~1300) -> QK(B0) issued -> S(B0) full (8154). QK(B0) was gated by program order in the MMA warp, behind PV(A7): its
inputs (the free S stage, Q and K(0) of B in smem) were ready from cycle ~900.

The mask's SASS (the previous head, D = 128): the per-key `if (key < lo_row) | (key >= L_row): rF[i] = -inf` over 128
registers lowered to ~800 straight-line instructions -- 256 ISETP on uniform-register key sums (127 UIADD3), 127 predicated
SEL.64 and 292 MOV that pair and copy the scores around the selects -- a predicate-register-bound chain at ~0.27 IPC. On a
causal batch every unit's last tile is masked: one tile in 4.5 on 32x1K / 64x1K (units of 1..8 tiles), one in 8.5 on 8x2K,
one in 32.5 on 1x8K.

## 2. The ceiling of the mask: a no-mask ablation (unprobed, wrong bits, timing only)

The mask skipped unless a window is set (the unit's last tile then computes on unmasked scores), paged 32:8, one-warpgroup body:

| D | 1x1K | 1x8K | 8x2K | 32x1K | 64x1K | geomean |
|---|---|---|---|---|---|---|
| 128 | 1.068 | 1.059 | 1.162 | 1.289 | 1.296 | 1.170 |
| 64 | 1.122 | 1.019 | 1.209 | 1.385 | 1.403 | 1.219 |

The mask was the body's largest single cost on the short-unit cells: 29 % at 32x1K / 64x1K, 16 % at 8x2K.

## 3. KEPT: the mask as per-key selects (`softmax.masked_score`)

The row's visible range is taken in tile-relative columns once per tile (`lim_lo = lo_row - k0`, `lim_hi = L_row - k0`); per
key, two compares of the bounds against the key's immediate column and one `selp` (inline PTX: `setp.gt.s32`, `setp.le.or.s32`,
`selp.f32 -inf`), with no dependence between keys. A select is not an arithmetic op: the masked values are the same -inf and
the unmasked scores pass through, so the row maximum, the exponentials and the row sums see the same operands in the same
order -- the dumps compare bit-identical. SASS (D = 128): the kernel 5920 -> 4856 instructions, the softmax's tile loop
2918 -> 2035 (the mask 800 -> 384; 478 MOV gone: the pairing copies had spread into the exponential stream too), LDL / STL
5 / 6 -> 2 / 6. Probe build: the masked tile's mask phase 3387 -> 998 cycles at D = 128 and 2522 -> 1256 at D = 64; the
UNMASKED tiles 2650 -> 1900-2000 (the exponential phase 1500 -> 965), which is why the long-unit cells gain too.

Paged 32:8, previous head / this placement:

| D | 1x1K | 1x8K | 8x2K | 32x1K | 64x1K | geomean |
|---|---|---|---|---|---|---|
| 128 | 1.061 | 1.075 | 1.130 | 1.217 | 1.229 | 1.140 |
| 64 | 1.093 | 1.072 | 1.190 | 1.284 | 1.285 | 1.181 |

Full grids, base / mask-only (30 iterations, cold L2, CUPTI kernel sums, two passes; a starred cell had one pass perturbed by a
foreign job on the GPU and quotes its clean pass in parentheses):

| D = 128, 1x1K / 1x8K / 8x2K / 32x1K / 64x1K | geomean |
|---|---|
| paged 32:8: 0.935* (1.06) / 1.076 / 1.074* (1.13) / 1.219 / 1.216 | 1.098 |
| ragged 32:8: 1.066 / 0.965* (1.07) / 1.088 / 1.146 / 1.153 | 1.071 |
| paged 8:8: 1.069 / 1.053 / 1.116 / 1.133 / 1.118 | |
| ragged 8:8: 1.053 / 1.045 / 1.084 / 1.071 / 1.053 | |
| decode 8x8K / 64x8K: 1.000 / 1.000 | 1.000 |

| D = 64, 1x1K / 1x8K / 8x2K / 32x1K / 64x1K | geomean |
|---|---|
| paged 32:8 (one-warpgroup body): 1.088 / 1.072 / 1.192 / 1.284 / 1.285 | 1.169 |
| ragged 32:8 (one-warpgroup body): 1.104 / 1.046 / 1.159 / 1.275 / 1.275 | 1.129 |
| paged 8:8: 1.050 / 1.059 / 1.186 / 1.267 / 1.242 | |
| ragged 8:8: 1.036 / 1.027 / 1.102 / 1.155 / 1.139 | |
| paired placement (the default from GQA group 4), paged: 1.093 / 0.980* (1.04) / 1.078 / 1.147 / 1.177 | 1.093 |
| paired placement, ragged: 1.106 / 1.159* (1.08) / 1.058 / 1.129 / 1.377* (1.14) | 1.161 |
| decode 8x8K / 64x8K: 1.011 / 0.936* (0.999) | |

| D = 256 paged 32:8 (the two-S-stage body), 1x1K / 1x8K / 8x2K / 32x1K / 64x1K |
|---|
| 0.963 / 0.987* (1.00) / 1.053* (1.02) / 1.651* (1.065) / 1.155* (1.068) -- the 1x1K cell is a real loss of 3.7 % (both passes agree): open item below |

Bits: the mask-only tree equals the base tree over the full D = 128 grid (22 cells), the full D = 64 grid (22 cells), the paired D = 64
cells (paged and ragged) and the D = 256 paged cells; the suite passes on the committed tree (107).

## 4. BUILT AND REJECTED: the next unit's QK(0) ahead of this unit's last PV

The placement (the commit and its revert are on the branch; the patch is in the job directory): the MMA warp issues the
next unit's QK(0) between this unit's second-to-last PV and its last PV (`mma.qk_head`: the Q stage wait, QK(0), the
stage's release for one-tile units), the load warp puts the next unit's Q and K(0) into the ring before this unit's last V
(`producer.load_head`), the CTA's first unit takes both heads in the roles' prologues, the next unit's PV(0) still acquires
the accumulator stage the epilogue releases after reading it (a barrier dependency, unchanged), the two-S-stage body (D = 256)
and the paired body keep their order. Bit-identical (the prefill and decode grid), spill-free alone (LDL / STL 5 / 6), 432
more instructions (a second copy of the QK issue). Probe build, the same two units:

```
     0  sm[A7].S_full    128  sm[A7].S_rel    545  mma[A7].top    733  mma[A7].nextQ_ready (+188)   830  K(B0) landed (+97)
   913  mma[A7].S_acq   1430  mma[A7].QK(B0)_issued (eight MMAs: 517)    1528/1627  V(A7) wait / landed (99)
  1126  sm[A7].alpha_sent (the mask now 998)   2531  sm[A7].P_commit   2553  mma[A7].P_seen   2906  mma[A7].PV_issued
  2942  sm[B0].top   3017  sm[B0].S_full       softmax gap A.P_commit -> B.S_full 2447 -> 486; PV(A7) never waited for the tail
  3293  cor[A7].epi_Ofull  ..  7419  cor[A7].epi_end          the epilogue now overlaps B0 and B1 instead of the idle gap
  B0: P store 916 (vs ~300)   B1: S copy-out 1109 (vs ~135)   B0's period 3323 against 1900-2000 mid-unit
```

The idle is gone (-1960 cycles per boundary) and the next unit's first two tiles stall for about as much (+600 on B0's P store,
+970 on B1's S copy-out, +80 on B0's exponentials): the epilogue's TMEM traffic (the O_c read, the O_m write, the normalise's
read: twelve 32-column pieces per lane) and its instruction stream now run under the softmax's own TMEM loads and stores and
its exponentials, where before they ran mostly under the softmax's idle. The softmax warp k and the correction warp k share
the lane quadrant k of TMEM and an SMSP, so the two roles' TMEM accesses and issue slots serialise whatever the order; the
overlap only moves where the epilogue's cost lands, and in production, where a softmax tile is shorter than in the probe
build while the epilogue is not, it lands worse. Measured alone (paged 32:8, D = 128): 1x1K 0.981, 1x8K 0.984, 8x2K 0.961,
32x1K 0.965, 64x1K 0.953 (geomean 0.969); on top of the mask placement 0.971 / 0.972 / 0.954 / 0.941 / 0.948 (geomean 0.957). Combined with the mask lowering the
kernel also took spills the parts do not have alone (LDL / STL 16 / 12, one reload per tile in the MMA warp's loop).
Reverted.

## 5. Recommendation and open items

The unit boundary's cost on the short-unit cells was the causal mask's lowering, now landed (+14 % / +19 % geomean on the
D = 128 / 64 paged 32:8 prefill cells, +22-33 % on 32x1K / 64x1K; no cell slower). The overlap of the next unit's first fold
with this unit's last is closed: the softmax's idle at the boundary is not free time but the window in which the correction's
epilogue uses the shared TMEM lane quadrants and SMSPs. What remains at the boundary is the epilogue itself (copy / merge,
normalise, output stores: ~4100 cycles per unit in the probe build, over which the next unit's first softmax tiles stall):
fewer TMEM passes for single-chunk units (normalise straight from O_c, which holds the accumulator stage longer -- the
previous note chose the copy to release it early; the trade is a measurement), or the output rows staged for a TMA store
(smem the footprint rule forbids). The probe tree with the new stamps (the next unit's Q arrival, the tail's QK(0)) is in the
job directory.
