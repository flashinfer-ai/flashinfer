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

// Two/four-CTA adjacent-spatial cluster-A weight multicast path.
// Included by the PatchShift kernel umbrella inside its detail namespace.

// Cluster-A across adjacent spatial tiles
// ------------------------------------------------
// All ranks compute the same M128 output-channel tile for adjacent
// flattened P16/Q30 spatial tiles.  Activations B are spatially distinct and
// remain CTA-local.  Weights A are identical: rank 0 arms both CTA-local A
// ready barriers and TMA-multicasts one packed K32 row to matching
// shared-memory offsets.  Each rank still issues an independent, legal
// cta_group::1 M128N256K16 .ws stream with bshift 0/1/2.
//
// The per-row completion barriers use the compile-time cluster size. Every
// rank multicasts its completion to all barrier copies, so each copy proves
// that all MMA engines released the multicast A row. In the four-rank path,
// a separate count-1 final-macro barrier releases each CTA-local B slot
// without coupling it to the slowest peer. The two-rank path retains the
// cheaper shared proof because an extra local commit did not improve it.
// An odd final spatial tile is duplicated for the arithmetic/barrier protocol,
// but only its owning rank stores output.
struct ClusterASpatialC64K64SharedStorage {
  K64C64B2A3K32ABStage b_stage[kK64C64B2A3K32ABRing];
  DeepIlpARowStage a_row[3];
  // Activation storage is CTA-local even though A is cluster-multicast.
  // A separate local completion lets each rank recycle its own B slot
  // without waiting for the slowest peer. The cluster-wide per-row barriers
  // remain the ownership proof for the multicast A rows.
  uint64_t local_b_done[kK64C64B2A3K32ABRing];
  uint32_t tmem_base;
  volatile int tmem_ready;
};

static_assert(sizeof(ClusterASpatialC64K64SharedStorage) == 226304);
static_assert(sizeof(ClusterASpatialC64K64SharedStorage) <= 232448,
              "cluster-A spatial C64/K64 pipeline must fit one SM100 CTA");

template <int ClusterSize, bool ExactN2D2 = false,
          bool ExactN1D8 = false, bool ExactN1D4 = false,
          bool ExactN2D4 = false, bool ExactId18 = false>
__global__ __launch_bounds__(kClusterBM256Threads, 1)
void general_m128_cluster_a_spatial_c64_k64_kernel(
    TensorMap const* input_c64_map,
    TensorMap const* weight_k32_map,
    Element* output,
    int n_size,
    int d_size,
    int h_size,
    int w_size,
    int c64_groups_per_time,
    int k_size) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(ClusterSize == 2 || ClusterSize == 4);
  static_assert(!ExactN2D2 || ClusterSize == 4);
  static_assert(!(ExactN2D2 && ExactN1D8));
  static_assert(!ExactN1D8 || ClusterSize == 4);
  static_assert(!ExactN1D4 || ClusterSize == 4);
  static_assert(!ExactN2D4 || ClusterSize == 2);
  static_assert(!ExactId18 || ClusterSize == 4);
  static_assert(int(ExactN2D2) + int(ExactN1D8) + int(ExactN1D4) +
                    int(ExactN2D4) + int(ExactId18) <=
                1);
  __shared__ ClusterASpatialC64K64SharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  uint32_t cluster_rank = cute::block_rank_in_cluster();

  constexpr bool ExactShape =
      ExactN2D2 || ExactN1D8 || ExactN1D4 || ExactN2D4 || ExactId18;
  int effective_d = ExactN2D2 ? 2
                    : ExactN1D8 ? 8
                    : (ExactN1D4 || ExactN2D4 || ExactId18) ? 4
                                                      : d_size;
  int effective_h = ExactId18 ? 129 : ExactShape ? 128 : h_size;
  int effective_w = ExactId18 ? 121 : ExactShape ? 120 : w_size;
  int effective_k = ExactShape ? 128 : k_size;
  int effective_c64_groups = ExactShape ? 2 : c64_groups_per_time;
  int q_tiles = ExactShape ? 4 : effective_w / kOutQ;
  int p_tiles = ExactShape ? 8 : effective_h / kMainOutP;
  int spatial_tiles = p_tiles * q_tiles;
  int spatial_group = int(blockIdx.x) / ClusterSize;
  int requested_spatial =
      spatial_group * ClusterSize + int(cluster_rank);
  bool owns_output = requested_spatial < spatial_tiles;
  // The physical grid always contains complete two-CTA clusters.  If the
  // logical tile count is odd, rank 1 duplicates the final rank-0 tile so it
  // can issue the second real MMA completion arrival without writing output.
  int spatial_tile =
      owns_output ? requested_spatial : spatial_tiles - 1;
  int p_tile = spatial_tile / q_tiles;
  int q_tile = spatial_tile - p_tile * q_tiles;
  int p_base = p_tile * kMainOutP;
  int q_base = q_tile * kOutQ;

  int flat_batch_count = ExactN2D2 ? 4
                         : ExactN1D8 ? 8
                         : ExactN1D4 ? 4
                         : ExactN2D4 ? 8
                         : ExactId18 ? 4
                                     : n_size * effective_d;
  int k_tile = ExactShape ? 0 : int(blockIdx.z) / flat_batch_count;
  int flat_batch = ExactShape
                       ? int(blockIdx.z)
                       : int(blockIdx.z) - k_tile * flat_batch_count;
  int n = flat_batch / effective_d;
  int od = flat_batch - n * effective_d;
  int k_base = k_tile * kMainM;

  int td_begin = od == 0 ? 1 : 0;
  int td_end = od == effective_d - 1 ? 2 : kT;
  int local_td_count = td_end - td_begin;
  int local_macros = local_td_count * effective_c64_groups;
  int c32_groups_per_time = effective_c64_groups * 2;
  int full_k32_supergroups = kT * c32_groups_per_time;

  constexpr int guard_rows = kMainBackingRows - kMainSemanticRows;
  constexpr int guard_per_stage = guard_rows * 64;
  for (int idx = int(threadIdx.x);
       idx < kK64C64B2A3K32ABRing * guard_per_stage;
       idx += int(blockDim.x)) {
    int slot = idx / guard_per_stage;
    int rest = idx - slot * guard_per_stage;
    int row = kMainSemanticRows + rest / 64;
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
      if constexpr (ClusterSize == 4) {
        patchshift::mbarrier_init(&shared.local_b_done[slot], 1);
      }
#pragma unroll
      for (int half = 0; half < kK32HalvesPerK64Macro; ++half) {
#pragma unroll
        for (int row = 0; row < 3; ++row) {
          patchshift::mbarrier_init(
              &shared.b_stage[slot].half_row_done[half][row],
              ClusterSize);
        }
      }
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      patchshift::mbarrier_init(&shared.a_row[row].ready, 1);
    }
  }
  __syncthreads();
  // Remote barriers and multicast destinations are legal only after both
  // CTA copies of the identical shared structure have been initialized.
  cute::cluster_sync();

  // Both ranks load their own spatial activation tile.  B2 slot reuse waits
  // for the cluster-wide completion of the old macro's last half/row.
  if (wid == 0 && lane == 0) {
    patchshift::tma_descriptor_fence_acquire(input_c64_map);
    constexpr uint32_t b_bytes =
        kMainSemanticRows * 64 * sizeof(Element);
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro % kK64C64B2A3K32ABRing;
      int seq = macro / kK64C64B2A3K32ABRing;
      if (seq > 0) {
        if constexpr (ClusterSize == 4) {
          while (!patchshift::mbarrier_try_wait(
              &shared.local_b_done[slot], (seq - 1) & 1)) {
          }
        } else {
          while (!patchshift::mbarrier_try_wait(
              &shared.b_stage[slot].half_row_done[1][2],
              (seq - 1) & 1)) {
          }
        }
      }
      int local_td = macro / effective_c64_groups;
      int c64g = macro - local_td * effective_c64_groups;
      int td = td_begin + local_td;
      patchshift::mbarrier_arrive_expect_tx(
          &shared.b_stage[slot].ready, b_bytes);
      patchshift::tma_load_5d(
          input_c64_map,
          &shared.b_stage[slot].ready,
          shared.b_stage[slot].b + swizzled_b_c64_index(0, 0),
          c64g * 64, q_base - 1, p_base - 1,
          od + td - 1, n);
      patchshift::fence_view_async_shared();
    }
  }

  // Rank 0 alone publishes A.  The local and remote barrier are both armed
  // for the same 24,576-byte row transaction before the cluster multicast.
  if (cluster_rank == 0 && wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kMainM * kK * sizeof(Element);
    constexpr uint16_t cluster_mask =
        uint16_t((1u << ClusterSize) - 1u);
    int half_tasks = local_macros * kK32HalvesPerK64Macro;
    for (int half_task = 0; half_task < half_tasks; ++half_task) {
      int macro = half_task / kK32HalvesPerK64Macro;
      int half = half_task % kK32HalvesPerK64Macro;
      int local_td = macro / effective_c64_groups;
      int c64g = macro - local_td * effective_c64_groups;
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
#pragma unroll
        for (int rank = 1; rank < ClusterSize; ++rank) {
          patchshift::mbarrier_arrive_expect_tx_remote(
              &shared.a_row[row].ready, a_row_bytes, rank);
        }
        int weight_task =
            (k_tile * full_k32_supergroups + full_sg) * 3 + row;
        patchshift::tma_load_5d_multicast(
            weight_k32_map, &shared.a_row[row].ready, cluster_mask,
            shared.a_row[row].a[0][0],
            0, 0, 0, 0, weight_task);
        patchshift::fence_view_async_shared();
      }
    }
  }

  // Both ranks issue independent cta_group::1 MMA streams.  One multicast
  // commit per rank supplies the two arrivals required by both barrier copies.
  if (wid == 2) {
    patchshift::tcgen05_alloc(&shared.tmem_base, kMainTmemColumns);
    __syncwarp();
    patchshift::tcgen05_relinquish_alloc_permit();
    if (lane == 0) {
      __threadfence_block();
      shared.tmem_ready = 1;
    }
    constexpr uint16_t cluster_mask =
        uint16_t((1u << ClusterSize) - 1u);
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
          issue_k64_c64_b2a3_k32a_row(
              shared.b_stage[slot], shared.a_row[row], half, row,
              shared.tmem_base, kPitch,
              macro == 0 && half == 0 && row == 0);
          patchshift::tcgen05_commit_multicast(
              &shared.b_stage[slot].half_row_done[half][row],
              cluster_mask);
          if constexpr (ClusterSize == 4) {
            if (half == kK32HalvesPerK64Macro - 1 && row == 2) {
              patchshift::tcgen05_commit(&shared.local_b_done[slot]);
            }
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
  if constexpr (ClusterSize == 4) {
    while (!patchshift::mbarrier_try_wait(
        &shared.local_b_done[final_slot], final_seq & 1)) {
    }
  } else {
    while (!patchshift::mbarrier_try_wait(
        &shared.b_stage[final_slot].half_row_done[1][2],
        final_seq & 1)) {
    }
  }
  __syncthreads();
  patchshift::tcgen05_fence_after_thread_sync();

  // The duplicate rank in an odd tail cluster skips all output ownership but
  // remains in the MMA and synchronization protocol above and below.
  if (owns_output) {
    int local_k = wid * 32 + lane;
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
            p_base + workset * kMainOutPPerWorkset +
            (physical_col >> 5);
        size_t pixel =
            ((size_t(flat_batch) * size_t(effective_h) + size_t(out_p)) *
                 size_t(effective_w) + size_t(q_base));
#pragma unroll
        for (int q = 0; q < kOutQ; ++q) {
          uint16_t bits = patchshift::element_bits_from_float(
              __uint_as_float(values[q]));
          *reinterpret_cast<uint16_t*>(
              output + (pixel + size_t(q)) * size_t(effective_k) +
                  size_t(k_base + local_k)) = bits;
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


// M64 output-tail policy
// ----------------------
// M64 uses only half as many physical TMEM columns for the same logical N256
// accumulator.  Four independent spatial worksets therefore fill the same 512
// column allocation used by the M128 policy.  The CTA covers P32xQ30xM64 and
// removes the 2x arithmetic waste of representing a 64-channel tail as M128.
constexpr int kTailM = 64;
constexpr int kTailN = 256;
constexpr int kTailWorksets = 4;
constexpr int kTailOutPPerWorkset = kTailN / kPitch;
constexpr int kTailOutP = kTailWorksets * kTailOutPPerWorkset;
constexpr int kTailInputP = kTailOutP + 2;
constexpr int kTailSemanticRows = kTailInputP * kPitch;
constexpr int kTailRequiredRows = kTailWorksets * kTailN + 66;
constexpr int kTailBackingRows = ((kTailRequiredRows + 7) / 8) * 8;
constexpr int kTailAccumulatorColumns = kTailN / 2;
constexpr int kTailTmemColumns = 512;
constexpr int kTailIssueWarps = 4;
constexpr int kTailWarps = 5;
constexpr int kTailThreads = kTailWarps * 32;

static_assert(kTailOutP == 32 && kTailSemanticRows == 1088 &&
              kTailBackingRows == 1096);
