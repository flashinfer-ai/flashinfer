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

#pragma once

// Logical M256 cluster-B path with C64/K64 activation multicast.
// Included by the PatchShift kernel umbrella inside its detail namespace.

// Logical M256 cluster-B with one C64 publication per K64 macro
// ----------------------------------------------------------------
// Two CTA ranks execute independent, legal 1-SM M128N256K16 .ws streams for
// adjacent output-channel tiles. Rank 0 TMA-multicasts the shared activation
// operand. A C64 B stage is consumed as two K32 halves, halving activation TMA
// command count relative to two C32 publications without changing bytes or
// arithmetic. We reuse the retained K32 A3 row stream so the complete B2/A3
// object remains below the SM100 per-CTA shared-memory limit.
constexpr int kId40CompactN = 256;
constexpr int kId40CompactPitch = 16;
constexpr int kId40CompactOutPPerWorkset =
    kId40CompactN / kId40CompactPitch;
constexpr int kId40CompactInputP =
    kMainWorksets * kId40CompactOutPPerWorkset + 2;
constexpr int kId40CompactSemanticRows =
    kId40CompactInputP * kId40CompactPitch;
constexpr int kId40PTailPitch = 32;
constexpr int kId40PTailInputP = 12;
constexpr int kId40PTailSemanticRows =
    kId40PTailInputP * kId40PTailPitch;
static_assert(kId40CompactN == 256 &&
              kId40CompactInputP == 34 &&
              kId40CompactSemanticRows == 544 &&
              kId40CompactSemanticRows <= kMainBackingRows);
static_assert(kId40PTailSemanticRows == 384);

__device__ __forceinline__ void issue_id40_ptail_k64_row(
    K64C64B2A3K32ABStage& b_stage,
    DeepIlpARowStage& a_stage,
    int k32_half,
    int filter_row,
    uint32_t tmem_base,
    bool first_row) {
  if (!patchshift::elect_one_sync()) {
    return;
  }
  constexpr uint64_t idesc_n256 = mma_idesc_n<256>();
  constexpr uint64_t idesc_n64 = mma_idesc_n<64>();
#pragma unroll
  for (int kg = 0; kg < kK16GroupsPerStage; ++kg) {
    int k_offset = k32_half * 32 + kg * kK;
    uint64_t desc_b256 = pack_b_c64_desc(
        b_stage.b + swizzled_b_c64_index(
                            filter_row * kId40PTailPitch, k_offset));
    uint64_t desc_b64 = pack_b_c64_desc(
        b_stage.b + swizzled_b_c64_index(
                            256 + filter_row * kId40PTailPitch, k_offset));
#pragma unroll
    for (int kw = 0; kw < 3; ++kw) {
      uint64_t desc_a =
          patchshift::pack_k16_desc(a_stage.a[kw][kg], kMainM);
      bool first = first_row && kg == 0 && kw == 0;
      mma_ws_raw(desc_a, desc_b256, tmem_base,
                 first ? 0u : 1u, idesc_n256,
                 patchshift::shift_desc(kw));
      mma_ws_raw(desc_a, desc_b64, tmem_base + 256,
                 first ? 0u : 1u, idesc_n64,
                 patchshift::shift_desc(kw));
    }
  }
}

__device__ __forceinline__ void issue_id40_qtail_k64_row(
    K64C64B2A3K32ABStage& b_stage,
    DeepIlpARowStage& a_stage,
    int k32_half,
    int filter_row,
    uint32_t tmem_base,
    bool first_row) {
  if (!patchshift::elect_one_sync()) {
    return;
  }
  constexpr uint64_t idesc_n256 = mma_idesc_n<kId40CompactN>();
#pragma unroll
  for (int kg = 0; kg < kK16GroupsPerStage; ++kg) {
    int k_offset = k32_half * 32 + kg * kK;
    uint64_t desc_b0 = pack_b_c64_desc(
        b_stage.b + swizzled_b_c64_index(
                            filter_row * kId40CompactPitch, k_offset));
    uint64_t desc_b1 = pack_b_c64_desc(
        b_stage.b + swizzled_b_c64_index(
                            kId40CompactN +
                                filter_row * kId40CompactPitch,
                            k_offset));
#pragma unroll
    for (int kw = 0; kw < 3; ++kw) {
      uint64_t desc_a =
          patchshift::pack_k16_desc(a_stage.a[kw][kg], kMainM);
      bool first = first_row && kg == 0 && kw == 0;
      mma_ws_raw(desc_a, desc_b0, tmem_base,
                 first ? 0u : 1u, idesc_n256,
                 patchshift::shift_desc(kw));
      mma_ws_raw(desc_a, desc_b1, tmem_base + kId40CompactN,
                 first ? 0u : 1u, idesc_n256,
                 patchshift::shift_desc(kw));
    }
  }
}

struct ClusterM256C64K64SharedStorage {
  K64C64B2A3K32ABStage b_stage[kK64C64B2A3K32ABRing];
  DeepIlpARowStage a_row[3];
  uint32_t tmem_base;
  volatile int tmem_ready;
};

static_assert(sizeof(ClusterM256C64K64SharedStorage) == 226304);
static_assert(offsetof(ClusterM256C64K64SharedStorage, b_stage) == 0);
static_assert(offsetof(ClusterM256C64K64SharedStorage, a_row) % 256 == 0);
static_assert(sizeof(ClusterM256C64K64SharedStorage) <= 232448,
              "cluster M256 C64/K64 pipeline must fit one SM100 CTA");

template <bool OptimizedPartial, int ExactKout = 0,
          bool ExactId40 = false, bool EightWarpStore = false,
          bool ExactD4C128 = false>
__global__ __launch_bounds__(EightWarpStore ? 256 : kClusterBM256Threads, 1)
void general_m256_cluster_b_c64_k64_kernel(
    TensorMap const* input_c64_map,
    TensorMap const* input_id40_ptail_c64_map,
    TensorMap const* input_id40_qtail_c64_map,
    TensorMap const* weight_k32_map,
    Element* output,
    int n_size,
    int d_size,
    int h_size,
    int w_size,
    int c64_groups_per_time,
    int k_size) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  __shared__ ClusterM256C64K64SharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  uint32_t cluster_rank = cute::block_rank_in_cluster();
  constexpr int kClusterSize = ExactId40 ? 4 : 2;

  int spatial_tile = int(blockIdx.x) / kClusterSize;
  // Preserve the rectangular launch's q-fast scheduling order while packing
  // away only its three invalid Q-tail rows: P0..2 have six Q families,
  // P3..5 have five.
  int q_tile = ExactId40
                   ? (spatial_tile < 18 ? spatial_tile % 6
                                        : (spatial_tile - 18) % 5)
                   : spatial_tile;
  int p_tile = ExactId40
                   ? (spatial_tile < 18 ? spatial_tile / 6
                                        : 3 + (spatial_tile - 18) / 5)
                   : int(blockIdx.y);
  bool compact_id40_ptail =
      ExactId40 && q_tile < 5 && p_tile == 5;
  bool compact_id40_qtail = ExactId40 && q_tile == 5;
  // Q=160 leaves a ten-column edge after five complete Q30 tiles. Reinterpret
  // that edge as two N256 worksets with pitch 16: each CTA covers P32.
  int pitch = compact_id40_qtail ? kId40CompactPitch : kPitch;
  int semantic_rows = compact_id40_qtail
                          ? kId40CompactSemanticRows
                          : (compact_id40_ptail
                                 ? kId40PTailSemanticRows
                                 : kMainSemanticRows);
  int q_base = compact_id40_qtail ? 150 : q_tile * kOutQ;
  int p_base = p_tile *
               (compact_id40_qtail
                    ? kMainWorksets * kId40CompactOutPPerWorkset
                    : kMainOutP);
  int flat_batch_count = (ExactId40 || ExactD4C128)
                             ? 4
                             : n_size * d_size;
  int k_pair = (ExactId40 || ExactD4C128)
                   ? 0
                   : int(blockIdx.z) / flat_batch_count;
  int flat_batch = (ExactId40 || ExactD4C128)
                       ? int(blockIdx.z)
                       : int(blockIdx.z) - k_pair * flat_batch_count;
  int n = (ExactId40 || ExactD4C128) ? 0 : flat_batch / d_size;
  int od = (ExactId40 || ExactD4C128)
               ? flat_batch
               : flat_batch - n * d_size;
  int k_tile = ExactId40
                   ? int(cluster_rank)
                   : 2 * k_pair + int(cluster_rank);
  int k_base = k_tile * kMainM;

  int td_begin = od == 0 ? 1 : 0;
  int td_end = od == ((ExactId40 || ExactD4C128) ? 3 : d_size - 1)
                   ? 2
                   : kT;
  int local_td_count = td_end - td_begin;
  int local_macros = local_td_count *
                     (ExactId40 ? 8
                                : (ExactD4C128 ? 2
                                                   : c64_groups_per_time));
  int c32_groups_per_time =
      ExactId40 ? 16
                : (ExactD4C128 ? 4 : c64_groups_per_time * 2);
  int full_k32_supergroups = kT * c32_groups_per_time;

  // Exact ID40 stores only Q30/Q10 logical positions.  For each of its three
  // dedicated maps the largest kh=2/kw=2 address of a stored element remains
  // inside the TMA semantic box; addresses beyond it belong exclusively to
  // discarded Q padding.  The generic path still clears its full guard.
  int guard_rows = ExactId40 ? 0 : kMainBackingRows - semantic_rows;
  int guard_per_stage = guard_rows * 64;
  for (int idx = int(threadIdx.x);
       idx < kK64C64B2A3K32ABRing * guard_per_stage;
       idx += int(blockDim.x)) {
    int slot = idx / guard_per_stage;
    int rest = idx - slot * guard_per_stage;
    int row = semantic_rows + rest / 64;
    int kk = rest % 64;
    shared.b_stage[slot].b[swizzled_b_c64_index(row, kk)] =
        patchshift::element_from_float(0.0f);
  }
  if (threadIdx.x == 0) {
    shared.tmem_base = 0;
    shared.tmem_ready = 0;
#pragma unroll
    for (int slot = 0; slot < kK64C64B2A3K32ABRing; ++slot) {
      patchshift::mbarrier_init(&shared.b_stage[slot].ready, 1);
#pragma unroll
      for (int half = 0; half < kK32HalvesPerK64Macro; ++half) {
#pragma unroll
        for (int row = 0; row < 3; ++row) {
          // A-row storage is private to each rank, so intermediate row
          // lifetimes need only the local CTA's MMA completion.  Only the
          // final row of a C64 macro protects the multicast B stage and must
          // therefore collect one arrival from both cluster ranks.
          patchshift::mbarrier_init(
              &shared.b_stage[slot].half_row_done[half][row],
              (half == kK32HalvesPerK64Macro - 1 && row == 2)
                  ? kClusterSize
                  : 1);
        }
      }
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      patchshift::mbarrier_init(&shared.a_row[row].ready, 1);
    }
  }
  __syncthreads();
  cute::cluster_sync();

  // Rank 0 owns B production. Both local ready barriers are armed before a
  // single P18xQ32xC64 multicast. Slot reuse waits for both MMA engines.
  if (cluster_rank == 0 && wid == 0 && lane == 0) {
    uint32_t b_bytes = uint32_t(semantic_rows * 64 * sizeof(Element));
    constexpr uint16_t cluster_mask = ExactId40 ? 0xfu : 0x3u;
    TensorMap const* selected_input_map =
        compact_id40_qtail
            ? input_id40_qtail_c64_map
            : (compact_id40_ptail ? input_id40_ptail_c64_map
                                  : input_c64_map);
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro % kK64C64B2A3K32ABRing;
      int seq = macro / kK64C64B2A3K32ABRing;
      if (seq > 0) {
        while (!patchshift::mbarrier_try_wait(
            &shared.b_stage[slot].half_row_done[1][2],
            (seq - 1) & 1)) {
        }
      }
      int local_td = macro /
                     (ExactId40 ? 8
                                : (ExactD4C128 ? 2
                                                   : c64_groups_per_time));
      int c64g = macro - local_td *
                             (ExactId40 ? 8
                                        : (ExactD4C128
                                               ? 2
                                               : c64_groups_per_time));
      int td = td_begin + local_td;
      patchshift::mbarrier_arrive_expect_tx(
          &shared.b_stage[slot].ready, b_bytes);
      patchshift::mbarrier_arrive_expect_tx_remote(
          &shared.b_stage[slot].ready, b_bytes, 1);
      if constexpr (ExactId40) {
        patchshift::mbarrier_arrive_expect_tx_remote(
            &shared.b_stage[slot].ready, b_bytes, 2);
        patchshift::mbarrier_arrive_expect_tx_remote(
            &shared.b_stage[slot].ready, b_bytes, 3);
      }
      patchshift::tma_load_5d_multicast(
          selected_input_map, &shared.b_stage[slot].ready, cluster_mask,
          shared.b_stage[slot].b + swizzled_b_c64_index(0, 0),
          c64g * 64, q_base - 1, p_base - 1,
          od + td - 1, n);
    }
  }

  // Each rank streams its own M128 weight rows. Reuse of A[row] waits on the
  // preceding cluster-wide row completion, which also releases shared B.
  if (wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kMainM * kK * sizeof(Element);
    int half_tasks = local_macros * kK32HalvesPerK64Macro;
    for (int half_task = 0; half_task < half_tasks; ++half_task) {
      int macro = half_task / kK32HalvesPerK64Macro;
      int half = half_task % kK32HalvesPerK64Macro;
      int local_td = macro /
                     (ExactId40 ? 8
                                : (ExactD4C128 ? 2
                                                   : c64_groups_per_time));
      int c64g = macro - local_td *
                             (ExactId40 ? 8
                                        : (ExactD4C128
                                               ? 2
                                               : c64_groups_per_time));
      int td = td_begin + local_td;
      int full_sg =
          td * c32_groups_per_time + c64g * 2 + half;
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        if (half_task > 0) {
          int previous_task = half_task - 1;
          int previous_macro =
              previous_task / kK32HalvesPerK64Macro;
          int previous_half =
              previous_task % kK32HalvesPerK64Macro;
          int previous_slot =
              previous_macro % kK64C64B2A3K32ABRing;
          int previous_seq =
              previous_macro / kK64C64B2A3K32ABRing;
          while (!patchshift::mbarrier_try_wait(
              &shared.b_stage[previous_slot]
                   .half_row_done[previous_half][row],
              previous_seq & 1)) {
          }
        }
        patchshift::mbarrier_arrive_expect_tx(
            &shared.a_row[row].ready, a_row_bytes);
        int weight_task =
            (k_tile * full_k32_supergroups + full_sg) * 3 + row;
        patchshift::tma_load_5d(
            weight_k32_map, &shared.a_row[row].ready,
            shared.a_row[row].a[0][0],
            0, 0, 0, 0, weight_task);
      }
    }
  }

  // Every rank owns one independent cta_group::1 M128 stream. One commit per
  // rank arrives at both copies of the count-two completion barrier.
  if (wid == 2) {
    patchshift::tcgen05_alloc(&shared.tmem_base, kMainTmemColumns);
    __syncwarp();
    patchshift::tcgen05_relinquish_alloc_permit();
    if (lane == 0) {
      __threadfence_block();
      shared.tmem_ready = 1;
    }
  }
  if (wid == 2) {
    constexpr uint16_t cluster_mask = ExactId40 ? 0xfu : 0x3u;
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro % kK64C64B2A3K32ABRing;
      int seq = macro / kK64C64B2A3K32ABRing;
      while (!patchshift::mbarrier_try_wait(
          &shared.b_stage[slot].ready, seq & 1)) {
      }
      patchshift::fence_view_async_shared();
#pragma unroll
      for (int half = 0; half < kK32HalvesPerK64Macro; ++half) {
        int half_task = macro * kK32HalvesPerK64Macro + half;
#pragma unroll
        for (int row = 0; row < 3; ++row) {
          while (!patchshift::mbarrier_try_wait(
              &shared.a_row[row].ready, half_task & 1)) {
          }
          patchshift::fence_view_async_shared();
          if (compact_id40_ptail) {
            issue_id40_ptail_k64_row(
                shared.b_stage[slot], shared.a_row[row], half, row,
                shared.tmem_base,
                macro == 0 && half == 0 && row == 0);
          } else if (compact_id40_qtail) {
            issue_id40_qtail_k64_row(
                shared.b_stage[slot], shared.a_row[row], half, row,
                shared.tmem_base,
                macro == 0 && half == 0 && row == 0);
          } else {
            issue_k64_c64_b2a3_k32a_row(
                shared.b_stage[slot], shared.a_row[row], half, row,
                shared.tmem_base, pitch,
                macro == 0 && half == 0 && row == 0);
          }
          if (half == kK32HalvesPerK64Macro - 1 && row == 2) {
            patchshift::tcgen05_commit_multicast(
                &shared.b_stage[slot].half_row_done[half][row],
                cluster_mask);
          } else {
            patchshift::tcgen05_commit(
                &shared.b_stage[slot].half_row_done[half][row]);
          }
        }
      }
    }
  }

  while (shared.tmem_ready == 0) {
  }
  int final_macro = local_macros - 1;
  int final_slot = final_macro % kK64C64B2A3K32ABRing;
  int final_seq = final_macro / kK64C64B2A3K32ABRing;
  while (!patchshift::mbarrier_try_wait(
      &shared.b_stage[final_slot].half_row_done[1][2],
      final_seq & 1)) {
  }
  __syncthreads();
  patchshift::tcgen05_fence_after_thread_sync();

  if constexpr (ExactKout > 0) {
    // These instances are dispatched only for exact spatial shapes with a
    // compile-time Kout in (128, 256).
    // Every spatial tile is complete. Rank 0 stores its complete M128 tile;
    // rank 1 stores only ExactKout-128 logical channels.  With eight store
    // warps, warp groups [0,4) and [4,8) own alternating x32 spatial columns
    // for the same four channel slices; padded rank-1 channel slices skip both
    // their TMEM loads and global stores.
    constexpr int kRank1Valid = ExactKout - kMainM;
    int store_partition = EightWarpStore ? (wid >> 2) : 0;
    int channel_warp = EightWarpStore ? (wid & 3) : wid;
    if (cluster_rank == 0 || channel_warp * 32 < kRank1Valid) {
      int local_k = channel_warp * 32 + lane;
#pragma unroll
      for (int workset = 0; workset < kMainWorksets; ++workset) {
        uint32_t tile_base =
            shared.tmem_base + uint32_t(workset * kMainAccumulatorColumns);
        for (int physical_col = store_partition * 32;
             physical_col < kMainN;
             physical_col += EightWarpStore ? 64 : 32) {
          uint32_t values[32];
          patchshift::tcgen05_load_32dp32b_x32(
              tile_base + physical_col, values);
          patchshift::tcgen05_wait_tmem_load();
          int out_p =
              p_base + workset * kMainOutPPerWorkset + (physical_col >> 5);
          size_t pixel =
              ((size_t(flat_batch) * size_t(h_size) + size_t(out_p)) *
                   size_t(w_size) +
               size_t(q_base));
          Element* out =
              output + pixel * size_t(ExactKout) + size_t(k_base);
#pragma unroll
          for (int q = 0; q < kOutQ; ++q) {
            uint16_t bits = patchshift::element_bits_from_float(
                __uint_as_float(values[q]));
            *reinterpret_cast<uint16_t*>(
                out + size_t(q) * size_t(ExactKout) + size_t(local_k)) = bits;
          }
        }
      }
    }
  } else if constexpr (ExactId40) {
    int store_partition = EightWarpStore ? (wid >> 2) : 0;
    int local_k = (EightWarpStore ? (wid & 3) : wid) * 32 + lane;
    if (compact_id40_ptail) {
      // P=90 leaves exactly ten rows after the five P16 interior tiles.
      // Workset 0 owns rows 80..87 (N256 at pitch 32), workset 1 owns
      // rows 88..89 (N64).  Both intervals are exact, so write Q0..29
      // directly instead of testing all 32 accumulator elements.
#pragma unroll
      for (int workset = 0; workset < 2; ++workset) {
        uint32_t tile_base = shared.tmem_base + uint32_t(workset * 256);
        if (workset == 0) {
          uint32_t values[128];
          patchshift::tcgen05_load_32dp32b_x128(
              tile_base + store_partition * 128, values);
          patchshift::tcgen05_wait_tmem_load();
#pragma unroll
          for (int row = 0; row < 4; ++row) {
            int out_p = 80 + store_partition * 4 + row;
            size_t pixel =
                (size_t(flat_batch) * 90u + size_t(out_p)) * 160u +
                size_t(q_base);
            Element* out =
                output + pixel * 512u + size_t(k_base + local_k);
#pragma unroll
            for (int q = 0; q < kOutQ; ++q) {
              uint16_t bits = patchshift::element_bits_from_float(
                  __uint_as_float(values[row * 32 + q]));
              *reinterpret_cast<uint16_t*>(out + size_t(q) * 512u) = bits;
            }
          }
        } else if (store_partition == 0) {
          uint32_t values[64];
          patchshift::tcgen05_load_32dp32b_x64(tile_base, values);
          patchshift::tcgen05_wait_tmem_load();
#pragma unroll
          for (int row = 0; row < 2; ++row) {
            int out_p = 88 + row;
            size_t pixel =
                (size_t(flat_batch) * 90u + size_t(out_p)) * 160u +
                size_t(q_base);
            Element* out =
                output + pixel * 512u + size_t(k_base + local_k);
#pragma unroll
            for (int q = 0; q < kOutQ; ++q) {
              uint16_t bits = patchshift::element_bits_from_float(
                  __uint_as_float(values[row * 32 + q]));
              *reinterpret_cast<uint16_t*>(out + size_t(q) * 512u) = bits;
            }
          }
        }
      }
    } else if (compact_id40_qtail) {
      // Q=160 leaves ten columns. N256 with pitch16 covers sixteen output
      // rows per workset. Two store partitions each load eight adjacent rows;
      // pitch padding and the final six P rows of the last task are skipped.
#pragma unroll
      for (int workset = 0; workset < kMainWorksets; ++workset) {
        uint32_t tile_base =
            shared.tmem_base + uint32_t(workset * kId40CompactN);
        int workset_p_base =
            p_base + workset * kId40CompactOutPPerWorkset;
        int valid_rows = min(kId40CompactOutPPerWorkset,
                             max(0, 90 - workset_p_base));
        uint32_t values[128];
        patchshift::tcgen05_load_32dp32b_x128(
            tile_base + store_partition * 128, values);
        patchshift::tcgen05_wait_tmem_load();
#pragma unroll
        for (int local_row = 0; local_row < 8; ++local_row) {
          int partition_row = store_partition * 8 + local_row;
          if (partition_row < valid_rows) {
            int out_p = workset_p_base + partition_row;
            size_t pixel =
                (size_t(flat_batch) * 90u + size_t(out_p)) * 160u +
                size_t(q_base);
            Element* out =
                output + pixel * 512u + size_t(k_base + local_k);
#pragma unroll
            for (int q = 0; q < 10; ++q) {
              int value_index = local_row * kId40CompactPitch + q;
              uint16_t bits = patchshift::element_bits_from_float(
                  __uint_as_float(values[value_index]));
              *reinterpret_cast<uint16_t*>(out + size_t(q) * 512u) = bits;
            }
          }
        }
      }
    } else {
      // The 5x5 interior is exact P16xQ30.  Load four adjacent output rows
      // per TMEM instruction; the irregular P/Q tails retain narrower loads.
#pragma unroll
      for (int workset = 0; workset < kMainWorksets; ++workset) {
        uint32_t tile_base =
            shared.tmem_base + uint32_t(workset * kMainAccumulatorColumns);
        for (int physical_col = store_partition * 128;
             physical_col < kMainN;
             physical_col += EightWarpStore ? 256 : 128) {
          uint32_t values[128];
          patchshift::tcgen05_load_32dp32b_x128(
              tile_base + physical_col, values);
          patchshift::tcgen05_wait_tmem_load();
#pragma unroll
          for (int row = 0; row < 4; ++row) {
            int out_p = p_base + workset * kMainOutPPerWorkset +
                        (physical_col >> 5) + row;
            size_t pixel =
                (size_t(flat_batch) * 90u + size_t(out_p)) * 160u +
                size_t(q_base);
            Element* out =
                output + pixel * 512u + size_t(k_base + local_k);
#pragma unroll
            for (int q = 0; q < kOutQ; ++q) {
              uint16_t bits = patchshift::element_bits_from_float(
                  __uint_as_float(values[row * 32 + q]));
              *reinterpret_cast<uint16_t*>(out + size_t(q) * 512u) = bits;
            }
          }
        }
      }
    }
  } else if constexpr (ExactD4C128) {
    static_assert(EightWarpStore && ExactKout == 0 && !ExactId40);
    int store_partition = wid >> 2;
    int local_k = (wid & 3) * 32 + lane;
#pragma unroll
    for (int workset = 0; workset < kMainWorksets; ++workset) {
      uint32_t tile_base =
          shared.tmem_base + uint32_t(workset * kMainAccumulatorColumns);
      for (int physical_col = store_partition * 64;
           physical_col < kMainN; physical_col += 128) {
        uint32_t values[64];
        patchshift::tcgen05_load_32dp32b_x64(
            tile_base + physical_col, values);
        patchshift::tcgen05_wait_tmem_load();
#pragma unroll
        for (int row = 0; row < 2; ++row) {
          int out_p = p_base + workset * kMainOutPPerWorkset +
                      (physical_col >> 5) + row;
          size_t pixel =
              (size_t(flat_batch) * 128u + size_t(out_p)) * 120u +
              size_t(q_base);
          Element* out = output + pixel * 256u + size_t(k_base + local_k);
#pragma unroll
          for (int q = 0; q < kOutQ; ++q) {
            uint16_t bits = patchshift::element_bits_from_float(
                __uint_as_float(values[row * 32 + q]));
            *reinterpret_cast<uint16_t*>(out + size_t(q) * 256u) = bits;
          }
        }
      }
    }
  } else if constexpr (!OptimizedPartial) {
    int store_partition = EightWarpStore ? (wid >> 2) : 0;
    int local_k = (EightWarpStore ? (wid & 3) : wid) * 32 + lane;
    int global_k = k_base + local_k;
    bool full_tile = k_base + kMainM <= k_size &&
                     p_base + kMainOutP <= h_size &&
                     q_base + kOutQ <= w_size;
    if (full_tile) {
#pragma unroll
      for (int workset = 0; workset < kMainWorksets; ++workset) {
        uint32_t tile_base =
            shared.tmem_base + uint32_t(workset * kMainAccumulatorColumns);
        for (int physical_col = store_partition * 64;
             physical_col < kMainN;
             physical_col += EightWarpStore ? 128 : 64) {
          uint32_t values[64];
          patchshift::tcgen05_load_32dp32b_x64(
              tile_base + physical_col, values);
          patchshift::tcgen05_wait_tmem_load();
#pragma unroll
          for (int row = 0; row < 2; ++row) {
            int out_p = p_base + workset * kMainOutPPerWorkset +
                        (physical_col >> 5) + row;
          size_t pixel =
              ((size_t(flat_batch) * size_t(h_size) + size_t(out_p)) *
                   size_t(w_size) +
               size_t(q_base));
          Element* out = output + pixel * size_t(k_size) + size_t(k_base);
#pragma unroll
          for (int q = 0; q < kOutQ; ++q) {
            uint16_t bits = patchshift::element_bits_from_float(
                __uint_as_float(values[row * 32 + q]));
            *reinterpret_cast<uint16_t*>(
                out + size_t(q) * size_t(k_size) + size_t(local_k)) = bits;
          }
          }
        }
      }
    } else {
#pragma unroll
      for (int workset = 0; workset < kMainWorksets; ++workset) {
        uint32_t tile_base =
            shared.tmem_base + uint32_t(workset * kMainAccumulatorColumns);
        for (int physical_col = store_partition * 32;
             physical_col < kMainN;
             physical_col += EightWarpStore ? 64 : 32) {
          uint32_t values[32];
          patchshift::tcgen05_load_32dp32b_x32(
              tile_base + physical_col, values);
          patchshift::tcgen05_wait_tmem_load();
          int out_p = p_base + workset * kMainOutPPerWorkset +
                      (physical_col >> 5);
          if (out_p < h_size && global_k < k_size) {
#pragma unroll
            for (int q = 0; q < kOutQ; ++q) {
              int out_q = q_base + q;
              if (out_q < w_size) {
                size_t pixel =
                    ((size_t(flat_batch) * size_t(h_size) + size_t(out_p)) *
                         size_t(w_size) +
                     size_t(out_q));
                uint16_t bits = patchshift::element_bits_from_float(
                    __uint_as_float(values[q]));
                *reinterpret_cast<uint16_t*>(
                    output + pixel * size_t(k_size) + size_t(global_k)) = bits;
              }
            }
          }
        }
      }
    }
  } else {
    int local_k = wid * 32 + lane;
    int global_k = k_base + local_k;
    int valid_q = min(kOutQ, max(0, w_size - q_base));
    bool full_tile = k_base + kMainM <= k_size &&
                     p_base + kMainOutP <= h_size &&
                     valid_q == kOutQ;
#pragma unroll
    for (int workset = 0; workset < kMainWorksets; ++workset) {
      uint32_t tile_base =
          shared.tmem_base + uint32_t(workset * kMainAccumulatorColumns);
      for (int physical_col = 0; physical_col < kMainN;
           physical_col += 32) {
        uint32_t values[32];
        patchshift::tcgen05_load_32dp32b_x32(
            tile_base + physical_col, values);
        patchshift::tcgen05_wait_tmem_load();
        int out_p =
            p_base + workset * kMainOutPPerWorkset + (physical_col >> 5);
        if (full_tile) {
          size_t pixel =
              ((size_t(flat_batch) * size_t(h_size) + size_t(out_p)) *
                   size_t(w_size) +
               size_t(q_base));
          Element* out = output + pixel * size_t(k_size) + size_t(k_base);
#pragma unroll
          for (int q = 0; q < kOutQ; ++q) {
            uint16_t bits = patchshift::element_bits_from_float(
                __uint_as_float(values[q]));
            *reinterpret_cast<uint16_t*>(
                out + size_t(q) * size_t(k_size) + size_t(local_k)) = bits;
          }
        } else if (out_p < h_size && global_k < k_size) {
          size_t pixel =
              ((size_t(flat_batch) * size_t(h_size) + size_t(out_p)) *
                   size_t(w_size) +
               size_t(q_base));
          Element* out =
              output + pixel * size_t(k_size) + size_t(global_k);
#pragma unroll
          for (int q = 0; q < kOutQ; ++q) {
            if (q < valid_q) {
              uint16_t bits = patchshift::element_bits_from_float(
                  __uint_as_float(values[q]));
              *reinterpret_cast<uint16_t*>(
                  out + size_t(q) * size_t(k_size)) = bits;
            }
          }
        }
      }
    }
  }
  patchshift::tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (wid == kClusterBM256Warps - 1) {
    patchshift::tcgen05_fence_after_thread_sync();
    patchshift::tcgen05_dealloc(shared.tmem_base, kMainTmemColumns);
  }
#else
  (void)input_c64_map;
  (void)weight_k32_map;
  (void)output;
  (void)n_size;
  (void)d_size;
  (void)h_size;
  (void)w_size;
  (void)c64_groups_per_time;
  (void)k_size;
#endif
}
