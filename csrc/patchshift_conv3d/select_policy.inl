/*
 * Copyright (c) 2026 by the PatchShift Conv3d contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Shape-policy selection body.
//
// Contract: included exactly once inside the host launcher after opts and
// device_prop have been validated. It declares the launch-policy state consumed
// by runtime.cuh. Keep shape thresholds and dispatch priority in this file;
// kernel implementation files must not make host policy decisions.

// Automatic dispatch: C8/C16 use the K16 activation path; every larger
// supported C uses one C32 activation TMA.
bool use_c16_path = opts.c <= kK;
int c16_groups = patchshift::round_up(opts.c, kK) / kK;
int c32_groups = (c16_groups + kK16GroupsPerStage - 1) / kK16GroupsPerStage;
int supergroups = kT * c32_groups;
int c64_groups = opts.c / 64;

// Merge a small output-channel remainder into a padded M128 tile only when
// the extra arithmetic fits the measured small-grid/wave policy.
int64_t base_ctas = int64_t((opts.h + kMainOutP - 1) / kMainOutP) *
                    int64_t((opts.w + kOutQ - 1) / kOutQ) * int64_t(opts.n) * int64_t(opts.d);
int full_m128_tiles = opts.k / kMainM;
int output_remainder = opts.k % kMainM;
bool merge_m128_tail = false;
if (output_remainder > 0 && base_ctas <= 128) {
  if (output_remainder <= 32) {
    merge_m128_tail = full_m128_tiles == 0 ? opts.c >= 128 : opts.c >= 16;
  } else if (output_remainder <= kTailM && full_m128_tiles > 0 && opts.c >= 88) {
    int64_t sm_count = int64_t(device_prop.multiProcessorCount);
    int64_t full_waves = (int64_t(full_m128_tiles) * base_ctas + sm_count - 1) / sm_count;
    int64_t padded_waves = (int64_t(full_m128_tiles + 1) * base_ctas + sm_count - 1) / sm_count;
    merge_m128_tail = base_ctas >= 96 || padded_waves == full_waves;
  }
}

bool use_m64_tail = !merge_m128_tail && output_remainder > 0 && output_remainder <= kTailM;
int m128_tiles = full_m128_tiles + (output_remainder > (use_m64_tail ? kTailM : 0) ? 1 : 0);
int m64_tiles = use_m64_tail ? 1 : 0;
int m64_output_base = full_m128_tiles * kMainM;
bool has_merged_m128_tail = output_remainder > 0 && m128_tiles > full_m128_tiles;
// Keep a complete M64 remainder independent from preceding M128 tiles on
// small grids so the two disjoint output-channel intervals may execute
// concurrently instead of padding the remainder to another M128 tile.
bool use_exact_m64_tail =
    !use_c16_path && full_m128_tiles > 0 && output_remainder == kTailM&& base_ctas <= 128;
if (use_exact_m64_tail) {
  merge_m128_tail = false;
  use_m64_tail = true;
  m128_tiles = full_m128_tiles;
  m64_tiles = 1;
  m64_output_base = full_m128_tiles * kMainM;
  has_merged_m128_tail = false;
}
// A native M32 tail is admitted only after at least one complete M128 main
// tile. Keep the existing standalone Kout=32 policy separate below.
bool use_exact_m32_tail =
    !use_c16_path && opts.c % 8 == 0 && full_m128_tiles > 0 && output_remainder == kM32P16M;
if (use_exact_m32_tail) {
  merge_m128_tail = false;
  use_m64_tail = true;
  m128_tiles = full_m128_tiles;
  m64_tiles = 1;
  m64_output_base = full_m128_tiles * kMainM;
  has_merged_m128_tail = false;
}
// Kout=192 on the measured N1/D4/P128/Q120 family is faster as one
// logical M256 cluster than as independent M128 + M64 launches.  Rank 0
// owns K[0,128), rank 1 computes a padded M128 tile and stores K[128,192).
// The dedicated kernel epilogue below removes all spatial checks and skips
// TMEM loads for rank 1's padded K[192,256) lanes.
bool use_padded_m256_k160 =
    opts.n == 1 && opts.d == 4 && opts.h == 128 && opts.w == 120 && opts.c == 128 && opts.k == 160;
bool use_padded_m256_k192 =
    opts.n == 1 && opts.d == 4 && opts.h == 128 && opts.w == 120 && opts.c == 128 && opts.k == 192;
bool use_padded_m256 = use_padded_m256_k160 || use_padded_m256_k192;
if (use_padded_m256) {
  use_exact_m32_tail = false;
  merge_m128_tail = true;
  use_m64_tail = false;
  m128_tiles = 2;
  m64_tiles = 0;
  m64_output_base = 0;
  has_merged_m128_tail = true;
}
// A merged M128 output-channel tail always needs the optimized epilogue.
// For a pure spatial tail, switch instances only when at least 10% of the
// P16xQ30 CTA grid is partial.  Use an exact integer comparison so that,
// for example, one Q-tail CTA among 36 columns retains the 112-register
// baseline instance instead of paying the tail instance's register cost in
// all 35 full CTAs as well.
int64_t epilogue_p_tiles = (int64_t(opts.h) + kMainOutP - 1) / kMainOutP;
int64_t epilogue_q_tiles = (int64_t(opts.w) + kOutQ - 1) / kOutQ;
int64_t epilogue_total_ctas = epilogue_p_tiles * epilogue_q_tiles;
int64_t epilogue_full_ctas = int64_t(opts.h / kMainOutP) * int64_t(opts.w / kOutQ);
int64_t epilogue_partial_ctas = epilogue_total_ctas - epilogue_full_ctas;
bool spatial_tail_fraction_at_least_ten_percent = epilogue_partial_ctas * 10 >= epilogue_total_ctas;
// A separate M64 remainder does not make the preceding M128 CTAs partial.
bool use_partial_m128_epilogue = has_merged_m128_tail || spatial_tail_fraction_at_least_ten_percent;

// Kout=32 keeps its existing unconditional exact-M32 policy for C>16.  A
// complete stack of M32 tiles is also profitable for Kout=64/96/128 when
// the whole launch remains at or below the measured 128-CTA small-grid
// boundary.  Larger grids retain their existing M128/M64 dispatch.
int m32_p16_tiles = opts.k / kM32P16M;
int64_t m32_p16_spatial_batch_ctas = int64_t((opts.h + kM32P16OutP - 1) / kM32P16OutP) *
                                     int64_t((opts.w + kOutQ - 1) / kOutQ) * int64_t(opts.n) *
                                     int64_t(opts.d);
int64_t m32_p16_total_ctas = m32_p16_spatial_batch_ctas * int64_t(m32_p16_tiles);
int m64_p16_tiles = (opts.k + kTailM - 1) / kTailM;
int64_t m64_p16_total_ctas = int64_t((opts.h + kM64P16OutP - 1) / kM64P16OutP) *
                             int64_t((opts.w + kOutQ - 1) / kOutQ) * int64_t(opts.n) *
                             int64_t(opts.d) * int64_t(m64_p16_tiles);
// GPU-accepted cluster-A path.  Two adjacent complete spatial tiles share
// one multicast weight stream while retaining independent legal 1SM .ws
// MMA issue.  The gate is deliberately limited to the two measured D4
// launch families and the measured N1D3 case.  The same mapping is also
// profitable for the measured N4D1 and N2D2 flat-batch families. N4D2
// stays on the independent C64/K64 path: with only one valid temporal
// slice per frame, four-rank multicast synchronization is not amortized.
// These families all keep complete
// P16/Q30 tiles and provide enough independent spatial clusters to
// amortize multicast setup.
// D8 remains on the single-CTA path because its gain was not repeatable.
bool cluster_a_batch_depth_family = (opts.d == 4 && opts.n <= 2) || (opts.d == 8 && opts.n == 1) ||
                                    (opts.d == 3 && opts.n == 1) || (opts.d == 2 && opts.n == 2) ||
                                    (opts.d == 1 && opts.n == 4);
// D3 has only 96 coarse M128 spatial/batch CTAs.  Two exact M64 output
// tiles raise the launch to 192 CTAs and are measurably faster despite the
// duplicated activation stream.
bool use_m64_d3_small_grid =
    opts.n == 1 && opts.d == 3 && opts.h == 128 && opts.w == 120 && opts.c == 128 && opts.k == 128;
bool use_split_cluster_a_compact_edges = opts.n == 1 && opts.d == 4 && opts.c == 128 &&
                                         opts.k == 128 && opts.h == 129 && opts.w == 121 &&
                                         m128_tiles == 1 && m64_tiles == 0;
bool use_cluster_a_spatial_c64_k64 =
    !use_m64_d3_small_grid && opts.c == 128 && opts.k == 128 && cluster_a_batch_depth_family &&
    ((opts.h % kMainOutP == 0 && opts.w % kOutQ == 0) || use_split_cluster_a_compact_edges) &&
    m128_tiles == 1 && m64_tiles == 0;
bool use_cluster_a_group4 =
    use_cluster_a_spatial_c64_k64 &&
    ((opts.d == 2 && opts.n == 2) || (opts.d == 4 && opts.n == 1) || (opts.d == 8 && opts.n == 1));
bool use_cluster_a_exact_n2d2 = use_cluster_a_group4 && opts.n == 2 && opts.d == 2 &&
                                opts.h == 128 && opts.w == 120 && opts.c == 128 && opts.k == 128;
bool use_cluster_a_exact_n1d8 = use_cluster_a_spatial_c64_k64 && opts.n == 1 && opts.d == 8 &&
                                opts.h == 128 && opts.w == 120 && opts.c == 128 && opts.k == 128;
bool use_cluster_a_exact_n1d4 = use_cluster_a_spatial_c64_k64 && opts.n == 1 && opts.d == 4 &&
                                opts.h == 128 && opts.w == 120 && opts.c == 128 && opts.k == 128;
bool use_cluster_a_exact_n2d4 = use_cluster_a_spatial_c64_k64 && opts.n == 2 && opts.d == 4 &&
                                opts.h == 128 && opts.w == 120 && opts.c == 128 && opts.k == 128;
bool use_cluster_a_exact_id18 = use_split_cluster_a_compact_edges;
bool use_hybrid_exact_w31 =
    opts.n == 1 && opts.d == 4 && opts.c == 96 && opts.k == 128 && opts.h == 512 && opts.w == 31;
bool use_hybrid_compact_c96 = false;
bool use_m32_p16 = opts.k == kM32P16M && opts.c > kK;
bool use_m64n128_d1_c32_micro =
    opts.n == 1 && opts.d == 1 && opts.h == 64 && opts.w == 64 && opts.c == 32 && opts.k == 64;
bool use_m32_d1_c32_micro = !use_m64n128_d1_c32_micro && opts.n == 1 && opts.d == 1 &&
                            opts.h == 64 && opts.w == 64 && opts.c == 32 && opts.k == 64;
bool use_m32_multi_p16 = !use_cluster_a_spatial_c64_k64 && !use_hybrid_compact_c96 &&
                         !use_m64_d3_small_grid && !use_m64n128_d1_c32_micro && !use_m32_p16 &&
                         !use_exact_m32_tail && opts.c > kK&& opts.k % kM32P16M == 0 &&
                         opts.k <= kMainM&& m32_p16_total_ctas <= 128;
bool use_m32_path = use_m32_p16 || use_exact_m32_tail || use_m32_multi_p16;
// Preserve the M32 tile count selected above while consuming aligned C64
// activation macros.  Restrict this to the multi-tile small-grid family:
// standalone Kout=32 and exact tails keep their separately measured C32
// implementations.
bool use_m32_p16_c64 = use_m32_multi_p16 && opts.c % 64 == 0;
bool use_m32_d1_c128_shallow =
    use_m32_p16_c64 && opts.n == 1 && opts.d == 1 && opts.c == 128 && opts.k == 128;
bool use_m32_d1_c128_shallow_exact =
    use_m32_d1_c128_shallow && opts.h % kM32P16OutP == 0 && opts.w % kOutQ == 0;
bool use_m32_d1_c128_shallow_cluster4 =
    use_m32_d1_c128_shallow_exact && opts.h == 128 && opts.w == 120;
bool use_m64_p16 = !use_padded_m256 && !use_cluster_a_spatial_c64_k64 && !use_hybrid_compact_c96 &&
                   !use_m32_path && !use_c16_path &&
                   (m64_p16_total_ctas <= 128 || use_m64_d3_small_grid);
bool use_m64_p16_c64 = use_m64_p16 && opts.c % 64 == 0;
bool use_m64_p16_c64_exact =
    use_m64_p16_c64 && opts.h % kM64P16OutP == 0 && opts.w % kOutQ == 0 && opts.k % kTailM == 0;
bool use_m64_cluster_b_c64 = use_m64_p16_c64_exact && opts.n == 1 && (opts.d == 2 || opts.d == 3) &&
                             opts.h == 128 && opts.w == 120 && opts.c == 128 && opts.k == 128;
if (use_m32_p16) {
  merge_m128_tail = false;
  use_m64_tail = true;
  m128_tiles = 0;
  m64_tiles = 1;
  m64_output_base = 0;
  has_merged_m128_tail = false;
} else if (use_exact_m32_tail) {
  // Preserve the disjoint M128 main and native M32 tail state established
  // before the epilogue policy was derived.
} else if (use_m32_multi_p16) {
  merge_m128_tail = false;
  use_m64_tail = true;
  m128_tiles = 0;
  m64_tiles = m32_p16_tiles;
  m64_output_base = 0;
  has_merged_m128_tail = false;
} else if (use_m64_p16) {
  merge_m128_tail = false;
  use_m64_tail = true;
  m128_tiles = 0;
  m64_tiles = m64_p16_tiles;
  m64_output_base = 0;
  has_merged_m128_tail = false;
}

// Compact spatial CTAs are legal only when every present edge fits the
// compact P4/Q6 capacities.  The non-empty test implements the requested
// P-tail 1..4 OR Q-tail 1..6 policy, while the 10% threshold avoids routing
// an entire launch through the high-register mixed kernel for a rare edge.
int compact_full_q_tiles = opts.w / kOutQ;
int compact_q_tail = opts.w - compact_full_q_tiles * kOutQ;
int compact_full_p_tiles = opts.h / kMainOutP;
int compact_p_tail = opts.h - compact_full_p_tiles * kMainOutP;
int64_t spatial_total_ctas = int64_t(compact_full_p_tiles + (compact_p_tail > 0)) *
                             int64_t(compact_full_q_tiles + (compact_q_tail > 0));
int64_t spatial_full_ctas = int64_t(compact_full_p_tiles) * int64_t(compact_full_q_tiles);
int64_t spatial_partial_ctas = spatial_total_ctas - spatial_full_ctas;
bool compact_tail_fits = compact_p_tail <= kCompactPOutP && compact_q_tail <= 6 &&
                         (compact_p_tail > 0 || compact_q_tail > 0);
bool compact_tail_fraction_at_least_ten_percent = spatial_partial_ctas * 10 >= spatial_total_ctas;
bool use_compact_spatial = !use_hybrid_compact_c96 && !use_hybrid_exact_w31 && !use_m32_path &&
                           !use_c16_path && opts.k % kMainM == 0 && m128_tiles > 0 &&
                           compact_tail_fits&& compact_tail_fraction_at_least_ten_percent;

// Replace only a one-row P edge whose width has no Q tail.  Both kernels
// consume one CTA/SM because the mixed main storage is 213504 B, so the
// exact launch-level CTA wave count is the relevant admission test.  The
// P1/Q126 variant is enabled only when it removes at least one whole wave
// on the current device; equal-wave shapes retain the mature compact path.
bool hybrid_compact_ptail1_candidate =
    opts.n == 1 && opts.d == 4 && opts.h == 17 && opts.w == 840 && opts.c == 96 && opts.k == 128;
int compact_p1_tail_tasks = int(
    (int64_t(compact_full_q_tiles) * int64_t(kOutQ) + kCompactPTail1OutQ - 1) / kCompactPTail1OutQ);
int ordinary_compact_spatial_tasks =
    compact_full_q_tiles * compact_full_p_tiles + (compact_p_tail > 0 ? compact_full_q_tiles : 0) +
    (compact_q_tail > 0 ? (opts.h + kCompactQOutP - 1) / kCompactQOutP : 0);
int p1_compact_spatial_tasks = compact_full_q_tiles * compact_full_p_tiles + compact_p1_tail_tasks;
int64_t compact_launch_repetitions = int64_t(opts.n) * int64_t(opts.d) * int64_t(m128_tiles);
int64_t sm_count = int64_t(device_prop.multiProcessorCount);
int64_t ordinary_compact_total_ctas =
    int64_t(ordinary_compact_spatial_tasks) * compact_launch_repetitions;
int64_t p1_compact_total_ctas = int64_t(p1_compact_spatial_tasks) * compact_launch_repetitions;
int64_t ordinary_compact_waves = (ordinary_compact_total_ctas + sm_count - 1) / sm_count;
int64_t p1_compact_waves = (p1_compact_total_ctas + sm_count - 1) / sm_count;
bool use_compact_ptail1_wave_path = use_compact_spatial && compact_p_tail == 1 &&
                                    compact_q_tail == 0 && compact_full_q_tiles > 0 &&
                                    ordinary_compact_waves > p1_compact_waves;

// Q1/Q2 reuses the compact M128N128 workset as P32/Q4. If the P tail is
// exactly one row, P1/Q126 owns that complete row, including the corner;
// Q2 then covers only the preceding full-P16 extent. Admission is automatic
// only when the Q-edge CTA count is reduced by the P32 mapping.
bool q2_combines_p1 = compact_p_tail == 1;
int compact_q2_p_extent = q2_combines_p1 ? compact_full_p_tiles * kMainOutP : opts.h;
int compact_q2_tail_tasks = (compact_q2_p_extent + kCompactQ2OutP - 1) / kCompactQ2OutP;
int ordinary_compact_q_tail_tasks = (opts.h + kCompactQOutP - 1) / kCompactQOutP;
int compact_q2_p_tail_tasks =
    compact_p_tail == 0 ? 0
                        : (q2_combines_p1 ? (opts.w + kCompactPTail1OutQ - 1) / kCompactPTail1OutQ
                                          : compact_full_q_tiles);
int compact_q2_spatial_tasks =
    compact_full_q_tiles * compact_full_p_tiles + compact_q2_p_tail_tasks + compact_q2_tail_tasks;
bool use_compact_qtail_q2_single_launch = use_compact_spatial && compact_q_tail >= 1 &&
                                          compact_q_tail <= 2 &&
                                          compact_q2_tail_tasks < ordinary_compact_q_tail_tasks;
bool use_compact_ptail1_single_launch =
    use_compact_ptail1_wave_path || (use_compact_qtail_q2_single_launch && q2_combines_p1);
int hybrid_compact_q1_tail_tasks = (opts.h + kCompactQ1OutP - 1) / kCompactQ1OutP;
int hybrid_compact_q1_spatial_tasks =
    compact_full_q_tiles * compact_full_p_tiles + hybrid_compact_q1_tail_tasks;
bool use_exact_p15_full_q_m128 = !use_compact_spatial && use_partial_m128_epilogue &&
                                 opts.h == 15 && opts.w % kOutQ == 0 && opts.k % kMainM == 0 &&
                                 m128_tiles > 0 && m64_tiles == 0;
int exact_aligned_kout = !use_compact_spatial && opts.h % kMainOutP == 0 && opts.w % kOutQ == 0 &&
                                 m128_tiles == 1 &&
                                 ((m64_tiles == 0 && (opts.k == 96 || opts.k == 120)) ||
                                  (use_exact_m32_tail && opts.k == 160))
                             ? opts.k
                             : 0;
bool use_hybrid_exact_p15 = opts.n == 1 && opts.d == 4 && opts.h == 15 && opts.w % kOutQ == 0 &&
                            opts.c == 96 && opts.k == 128 && m128_tiles == 1 && m64_tiles == 0;
bool use_hybrid_cluster_a4_exact_p15 = use_hybrid_exact_p15;
bool use_hybrid_compact_p1_c96 = use_compact_ptail1_single_launch &&
                                 hybrid_compact_ptail1_candidate && m128_tiles == 1 &&
                                 m64_tiles == 0;

// GPU-accepted full-M128 C96/Kout128 policy.  Compact spatial handling,
// including the P1/Q126 single-launch path above, and the partial M128
// epilogue retain priority.  Existing small-grid M32/M64 choices are not
// overridden by this measured gate.
bool use_hybrid_c64_c32 =
    opts.c == 96 && opts.k == 128 && !use_hybrid_compact_c96 &&
    ((!use_compact_spatial && !use_partial_m128_epilogue) || use_hybrid_exact_p15 ||
     use_hybrid_compact_p1_c96 || use_hybrid_exact_w31) &&
    !use_m32_path && !use_m64_p16 && m128_tiles == 1 && m64_tiles == 0;
bool use_hybrid_exact_spatial =
    use_hybrid_c64_c32 && opts.h % kMainOutP == 0 && opts.w % kOutQ == 0;
bool use_hybrid_exact_h16_w840 =
    use_hybrid_exact_spatial && opts.n == 1 && opts.d == 4 && opts.h == 16 && opts.w == 840;

// Priority is exact M32, M64/P16, compact spatial tails, logical M256
// cluster-B, then the ordinary single-CTA C64/K64 macro.
// Cluster-B requires complete
// pairs of physical M128 output-channel tiles; each rank still executes an
// independent legal 1SM M128N256 MMA workset. The C64/K64 cluster variant
// is retained from the measured 128-CTA gate onward. At exactly 128 base
// CTAs it reduces the Kout256 path by about 2.3% relative to Cluster-B C32;
// larger C64-aligned grids retain the same activation-publication saving.
bool m256_cluster_b_eligible = !use_cluster_a_spatial_c64_k64 && !use_hybrid_c64_c32 &&
                               !use_hybrid_compact_c96 && !use_m32_path && !use_m64_p16 &&
                               !use_compact_spatial && m128_tiles > 0 && opts.c > kK &&
                               (opts.k % 256 == 0 || use_padded_m256);
bool use_m256_cluster_b_c64_k64 =
    m256_cluster_b_eligible && (base_ctas >= 128 || use_padded_m256) && opts.c % 64 == 0;
bool use_m256_cluster_b_c32 = m256_cluster_b_eligible && !use_m256_cluster_b_c64_k64;
bool use_m256_cluster_b_c64_optimized_partial = use_m256_cluster_b_c64_k64 &&
                                                use_partial_m128_epilogue &&
                                                !(opts.n == 1 && opts.d == 4 && opts.h == 180 &&
                                                  opts.w == 320 && opts.c == 64 && opts.k == 256) &&
                                                !(opts.n == 1 && opts.d == 4 && opts.h == 90 &&
                                                  opts.w == 160 && opts.c == 512 && opts.k == 512);
bool use_m256_cluster_b_c64_exact_id40 = use_m256_cluster_b_c64_k64 && opts.n == 1 && opts.d == 4 &&
                                         opts.h == 90 && opts.w == 160 && opts.c == 512 &&
                                         opts.k == 512;
bool use_m256_cluster_b_c64_eight_warp_store = use_m256_cluster_b_c64_k64 && opts.n == 1 &&
                                               opts.d == 4 && opts.k == 256 &&
                                               ((opts.h == 180 && opts.w == 320 && opts.c == 64) ||
                                                (opts.h == 128 && opts.w == 120 && opts.c == 128));
bool use_m256_cluster_b_c64_exact_d4_c128 =
    use_m256_cluster_b_c64_k64 && opts.n == 1 && opts.d == 4 && opts.h == 128 && opts.w == 120 &&
    opts.c == 128 && (opts.k == 160 || opts.k == 192 || opts.k == 256);
// The K64 path retains the ordinary K32 packed-weight layout, but publishes
// one canonical C64 activation tile for two consecutive K32 halves.
bool use_k64_c64_b2a3_k32a = !use_cluster_a_spatial_c64_k64 && !use_hybrid_c64_c32 &&
                             !use_hybrid_compact_c96 && !use_m32_path && !use_m64_p16 &&
                             !use_compact_spatial && !use_m256_cluster_b_c64_k64 &&
                             !use_m256_cluster_b_c32 && m128_tiles > 0 && opts.c % 64 == 0 &&
                             (opts.k % kMainM == 0 || opts.k == 192 || exact_aligned_kout == 96 ||
                              exact_aligned_kout == 120);
bool use_k64_c64_exact_k128 = use_k64_c64_b2a3_k32a && opts.k == kMainM;
int k64_c64_exact_kout =
    use_k64_c64_b2a3_k32a && (exact_aligned_kout == 96 || exact_aligned_kout == 120)
        ? exact_aligned_kout
        : 0;
