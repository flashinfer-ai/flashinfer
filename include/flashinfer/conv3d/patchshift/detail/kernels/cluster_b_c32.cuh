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

// Logical M256 cluster-B path with C32 activation multicast.
// Included by the PatchShift kernel umbrella inside its detail namespace.

// Logical M256 from a two-CTA cluster: multicast shared activations (B)
// ----------------------------------------------------------------------
// In the transposed PatchShift mapping, MMA.M is the output-channel tile and
// MMA.N is the spatial tile:
//
//   A[M128,K16] = weights for one output-channel tile
//   B[K16,N256] = activations for one spatial tile
//
// Therefore two CTA ranks that compute adjacent M128 output-channel tiles for
// the same spatial coordinates share B, not A.  Each rank keeps a legal 1SM
// M128N256 bshift MMA and loads its own k_tile-specific A row.  Rank 0 arms
// both CTA-local B.ready barriers and multicasts one C32 activation TMA tile.
// A single multicast tcgen05 commit per filter row arrives at both copies of
// B.row_done; arrive_count=2 proves that both independent MMA consumers have
// released the shared activation stage.  The same barrier also gates each
// CTA's local A-row reuse, avoiding any unsafe second/empty commit.
constexpr int kClusterBM256BRing = 3;
constexpr int kClusterBM256ARows = 3;
constexpr int kClusterBM256Warps = 4;
constexpr int kClusterBM256Threads = kClusterBM256Warps * 32;

struct alignas(512) ClusterBM256BStage {
  alignas(512) Element b[kMainBackingRows * 32];
  uint64_t ready;
  uint64_t row_done[3];
};

struct alignas(256) ClusterBM256ARowStage {
  alignas(128) Element a[3][kK16GroupsPerStage][kMainM * kK];
  uint64_t ready;
};

struct ClusterBM256SharedStorage {
  ClusterBM256BStage b_stage[kClusterBM256BRing];
  ClusterBM256ARowStage a_row[kClusterBM256ARows];
  uint32_t tmem_base;
  volatile int tmem_ready;
};

static_assert(sizeof(ClusterBM256BStage) == 37888);
static_assert(sizeof(ClusterBM256ARowStage) == 24832);
static_assert(offsetof(ClusterBM256SharedStorage, b_stage) == 0);
static_assert(offsetof(ClusterBM256BStage, b) == 0);
static_assert(offsetof(ClusterBM256BStage, ready) ==
              kMainBackingRows * 32 * sizeof(Element));
static_assert(offsetof(ClusterBM256SharedStorage, a_row) % 256 == 0);
static_assert(sizeof(ClusterBM256SharedStorage) <= 232448,
              "cluster M256 shared-B pipeline must fit one SM100 CTA");

__device__ __forceinline__ void issue_cluster_b_m256_row(
    ClusterBM256BStage& b_stage,
    ClusterBM256ARowStage& a_stage,
    int filter_row,
    uint32_t tmem_base,
    bool first_row,
    int valid_k16_groups) {
  if (!patchshift::elect_one_sync()) {
    return;
  }
  constexpr uint64_t idesc = main_mma_idesc();
  constexpr uint32_t dst0_offset = 0;
  constexpr uint32_t dst1_offset = kMainAccumulatorColumns;
#pragma unroll
  for (int kg = 0; kg < kK16GroupsPerStage; ++kg) {
    if (kg >= valid_k16_groups) {
      continue;
    }
    int k_offset = kg * kK;
    Element* b0 =
        b_stage.b +
        swizzled_b_c32_index(filter_row * kPitch, k_offset);
    Element* b1 =
        b_stage.b +
        swizzled_b_c32_index(kMainN + filter_row * kPitch, k_offset);
    uint64_t desc_b0 = pack_b_c32_desc(b0);
    uint64_t desc_b1 = pack_b_c32_desc(b1);
#pragma unroll
    for (int kw = 0; kw < 3; ++kw) {
      uint64_t desc_a =
          patchshift::pack_k16_desc(a_stage.a[kw][kg], kMainM);
      bool first = first_row && kg == 0 && kw == 0;
      mma_ws_raw(desc_a, desc_b0, tmem_base + dst0_offset,
                 first ? 0u : 1u, idesc,
                 patchshift::shift_desc(kw));
      mma_ws_raw(desc_a, desc_b1, tmem_base + dst1_offset,
                 first ? 0u : 1u, idesc,
                 patchshift::shift_desc(kw));
    }
  }
}

// Full and partial epilogues are compiled separately.  No compact instance is
// needed because compact spatial tails have higher dispatch priority.
template <bool OptimizedPartial>
__global__ __launch_bounds__(kClusterBM256Threads, 1)
void general_m256_cluster_b_c32_kernel(
    TensorMap const* input_c32_map,
    TensorMap const* weight_map,
    Element* output,
    int n_size,
    int d_size,
    int h_size,
    int w_size,
    int c32_groups_per_time,
    int c16_groups_per_time,
    int k_size) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  __shared__ ClusterBM256SharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  uint32_t cluster_rank = cute::block_rank_in_cluster();

  // clusterDim.x=2.  Two physical blockIdx.x values map to one logical Q30
  // tile; cluster rank selects adjacent M128 output-channel tiles instead.
  int q_tile = int(blockIdx.x) >> 1;
  int q_base = q_tile * kOutQ;
  int p_base = int(blockIdx.y) * kMainOutP;
  int flat_batch_count = n_size * d_size;
  int k_pair = int(blockIdx.z) / flat_batch_count;
  int flat_batch = int(blockIdx.z) - k_pair * flat_batch_count;
  int n = flat_batch / d_size;
  int od = flat_batch - n * d_size;
  int k_tile = 2 * k_pair + int(cluster_rank);
  int k_base = k_tile * kMainM;

  int td_begin = od == 0 ? 1 : 0;
  int td_end = od == d_size - 1 ? 2 : kT;
  int local_td_count = td_end - td_begin;
  int local_supergroups = local_td_count * c32_groups_per_time;
  int full_supergroups = kT * c32_groups_per_time;

  constexpr int guard_rows = kMainBackingRows - kMainSemanticRows;
  constexpr int guard_per_stage = guard_rows * 32;
  for (int idx = int(threadIdx.x);
       idx < kClusterBM256BRing * guard_per_stage;
       idx += int(blockDim.x)) {
    int slot = idx / guard_per_stage;
    int rest = idx - slot * guard_per_stage;
    int row = kMainSemanticRows + rest / 32;
    int kk = rest % 32;
    shared.b_stage[slot].b[swizzled_b_c32_index(row, kk)] =
        patchshift::element_from_float(0.0f);
  }
  if (threadIdx.x == 0) {
    shared.tmem_base = 0;
    shared.tmem_ready = 0;
    for (int slot = 0; slot < kClusterBM256BRing; ++slot) {
      patchshift::mbarrier_init(&shared.b_stage[slot].ready, 1);
      for (int row = 0; row < 3; ++row) {
        patchshift::mbarrier_init(
            &shared.b_stage[slot].row_done[row], 2);
      }
    }
    for (int row = 0; row < kClusterBM256ARows; ++row) {
      patchshift::mbarrier_init(&shared.a_row[row].ready, 1);
    }
  }
  __syncthreads();
  cute::cluster_sync();

  // Rank 0 warp 0 is the only B producer.  Both CTA shared-memory layouts are
  // statically identical, so one local destination offset names both DSM
  // targets selected by mask 0b11.
  if (cluster_rank == 0 && wid == 0 && lane == 0) {
    constexpr uint32_t b_bytes =
        kMainSemanticRows * 32 * sizeof(Element);
    constexpr uint16_t cluster_mask = 0x3u;
    for (int sg = 0; sg < local_supergroups; ++sg) {
      int b_slot = sg % kClusterBM256BRing;
      int b_seq = sg / kClusterBM256BRing;
      if (b_seq > 0) {
        while (!patchshift::mbarrier_try_wait(
            &shared.b_stage[b_slot].row_done[2],
            (b_seq - 1) & 1)) {
        }
      }
      int local_td = sg / c32_groups_per_time;
      int c32g = sg - local_td * c32_groups_per_time;
      int td = td_begin + local_td;
      patchshift::mbarrier_arrive_expect_tx(
          &shared.b_stage[b_slot].ready, b_bytes);
      patchshift::mbarrier_arrive_expect_tx_remote(
          &shared.b_stage[b_slot].ready, b_bytes, 1);
      patchshift::tma_load_5d_multicast(
          input_c32_map, &shared.b_stage[b_slot].ready, cluster_mask,
          shared.b_stage[b_slot].b + swizzled_b_c32_index(0, 0),
          c32g * 32, q_base - 1, p_base - 1,
          od + td - 1, n);
    }
  }

  // Each rank loads its own M128 output-channel weight tile.  Reuse of A[row]
  // waits on the prior supergroup's cluster-wide row completion, which proves
  // both A and multicast B operand reads are finished with one commit.
  if (wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kMainM * kK * sizeof(Element);
    for (int sg = 0; sg < local_supergroups; ++sg) {
      int local_td = sg / c32_groups_per_time;
      int c32g = sg - local_td * c32_groups_per_time;
      int td = td_begin + local_td;
      int full_sg = td * c32_groups_per_time + c32g;
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        if (sg > 0) {
          int old_sg = sg - 1;
          int old_b_slot = old_sg % kClusterBM256BRing;
          int old_b_seq = old_sg / kClusterBM256BRing;
          while (!patchshift::mbarrier_try_wait(
              &shared.b_stage[old_b_slot].row_done[row],
              old_b_seq & 1)) {
          }
        }
        patchshift::mbarrier_arrive_expect_tx(
            &shared.a_row[row].ready, a_row_bytes);
        int weight_task =
            (k_tile * full_supergroups + full_sg) * 3 + row;
        patchshift::tma_load_5d(
            weight_map, &shared.a_row[row].ready,
            shared.a_row[row].a[0][0],
            0, 0, 0, 0, weight_task);
      }
    }
  }

  if (wid == 2) {
    patchshift::tcgen05_alloc(&shared.tmem_base, kMainTmemColumns);
    __syncwarp();
    patchshift::tcgen05_relinquish_alloc_permit();
    if (lane == 0) {
      __threadfence_block();
      shared.tmem_ready = 1;
    }

    constexpr uint16_t cluster_mask = 0x3u;
    for (int sg = 0; sg < local_supergroups; ++sg) {
      int b_slot = sg % kClusterBM256BRing;
      int b_seq = sg / kClusterBM256BRing;
      while (!patchshift::mbarrier_try_wait(
          &shared.b_stage[b_slot].ready, b_seq & 1)) {
      }
      patchshift::fence_view_async_shared();
      int c32g = sg % c32_groups_per_time;
      int valid_k16_groups =
          min(kK16GroupsPerStage,
              c16_groups_per_time -
                  c32g * kK16GroupsPerStage);
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        while (!patchshift::mbarrier_try_wait(
            &shared.a_row[row].ready, sg & 1)) {
        }
        patchshift::fence_view_async_shared();
        issue_cluster_b_m256_row(
            shared.b_stage[b_slot], shared.a_row[row], row,
            shared.tmem_base, sg == 0 && row == 0,
            valid_k16_groups);
        patchshift::tcgen05_commit_multicast(
            &shared.b_stage[b_slot].row_done[row], cluster_mask);
      }
    }
  }

  while (shared.tmem_ready == 0) {
  }
  int final_sg = local_supergroups - 1;
  int final_b_slot = final_sg % kClusterBM256BRing;
  int final_b_seq = final_sg / kClusterBM256BRing;
  while (!patchshift::mbarrier_try_wait(
      &shared.b_stage[final_b_slot].row_done[2],
      final_b_seq & 1)) {
  }
  // A cluster-multicast completion barrier proves that both MMA engines have
  // finished, but it does not by itself form a CTA thread-sync edge for warps
  // that never participated in a producer or issuer role.  Join all four
  // local warps before importing TMEM visibility.  Without this sync, the
  // producer/issuer warps read correct accumulators while idle epilogue-only
  // warps 0/3 can observe stale TMEM values.
  __syncthreads();
  patchshift::tcgen05_fence_after_thread_sync();

  if constexpr (!OptimizedPartial) {
    // Original full-tile instance.  Keep this block byte-for-byte equivalent
    // to the final baseline epilogue so that its resource footprint cannot be
    // inflated by partial-tile row pointers or valid_q state.
    int local_k = wid * 32 + lane;
    int global_k = k_base + local_k;
    bool full_tile = k_base + kMainM <= k_size &&
                     p_base + kMainOutP <= h_size &&
                     q_base + kOutQ <= w_size;
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
  } else {
    // Partial P/Q/M instance.  valid_q is CTA-uniform, while the P/M tests
    // are invariant across the fully unrolled q body for one output row.
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
  cute::cluster_sync();
  if (wid == kClusterBM256Warps - 1) {
    patchshift::tcgen05_fence_after_thread_sync();
    patchshift::tcgen05_dealloc(shared.tmem_base, kMainTmemColumns);
  }
#else
  (void)input_c32_map;
  (void)weight_map;
  (void)output;
  (void)n_size;
  (void)d_size;
  (void)h_size;
  (void)w_size;
  (void)c32_groups_per_time;
  (void)c16_groups_per_time;
  (void)k_size;
#endif
}
