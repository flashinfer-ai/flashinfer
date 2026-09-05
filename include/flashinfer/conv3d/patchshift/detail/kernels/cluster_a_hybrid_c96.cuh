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

// Exact N1/D4/H15/W840/C96/K128 cluster-A path.
//
// Four adjacent Q tiles execute independent cta_group::1 M128N256K16 .ws
// streams.  Activations remain CTA-local, while rank 0 multicasts each packed
// K32 weight row to all four ranks.  C96 is represented without padding as one
// C64 macro followed by one C32 macro for every temporal filter position.

struct HybridClusterA4C96SharedStorage {
  HybridC64C32BStage b_stage[kHybridC64C32BStages];
  HybridC64C32ARowStage a_row[3];
  uint64_t local_b_done[kHybridC64C32BStages];
  uint32_t tmem_base;
};

// P15 needs input rows [-1, 15], i.e. 17 rows.  The generic P16 path loads
// 18 rows because it also computes output row 15.  This exact path never
// stores that row, so the smaller TMA box removes one full 32-column input
// row from every activation publication without changing any valid result.
constexpr int kHybridExactP15InputP = 17;
constexpr int kHybridExactP15SemanticRows =
    kHybridExactP15InputP * kPitch;

static_assert(sizeof(HybridClusterA4C96SharedStorage) <= 232448,
              "hybrid C96 cluster-A4 pipeline must fit one SM100 CTA");
static_assert(offsetof(HybridClusterA4C96SharedStorage, a_row) % 256 == 0);

__global__ __launch_bounds__(256, 1)
void general_hybrid_c96_exact_p15_cluster_a4_kernel(
    TensorMap const* input_c64_map,
    TensorMap const* input_c32_map,
    TensorMap const* weight_k32_map,
    Element* output) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  __shared__ HybridClusterA4C96SharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  uint32_t cluster_rank = cute::block_rank_in_cluster();

  constexpr int kClusterSize = 4;
  constexpr int kQTiles = 28;
  constexpr int kD = 4;
  constexpr int kH = 15;
  constexpr int kW = 840;
  constexpr int kKout = 128;
  constexpr int kC32Groups = 3;
  constexpr int kMacrosPerTime = 2;
  constexpr uint16_t kClusterMask = 0xfu;

  int q_tile = (int(blockIdx.x) / kClusterSize) * kClusterSize +
               int(cluster_rank);
  if (q_tile >= kQTiles) {
    return;
  }
  int q_base = q_tile * kOutQ;
  int od = int(blockIdx.z);
  int td_begin = od == 0 ? 1 : 0;
  int td_end = od == kD - 1 ? 2 : kT;
  int local_td_count = td_end - td_begin;
  int local_macros = local_td_count * kMacrosPerTime;
  int local_half_tasks = local_td_count * kC32Groups;

  if (threadIdx.x == 0) {
    shared.tmem_base = 0;
#pragma unroll
    for (int slot = 0; slot < kHybridC64C32BStages; ++slot) {
      patchshift::mbarrier_init(&shared.b_stage[slot].ready, 1);
      patchshift::mbarrier_init(&shared.local_b_done[slot], 1);
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      patchshift::mbarrier_init(&shared.a_row[row].ready, 1);
      patchshift::mbarrier_init(&shared.a_row[row].done, kClusterSize);
    }
  }
  __syncthreads();
  cute::cluster_sync();

  // Every rank publishes its own spatial activation tile.  The local
  // completion barrier decouples B-slot reuse from slower peers.
  if (wid == 0 && lane == 0) {
    patchshift::tma_descriptor_fence_acquire(input_c64_map);
    patchshift::tma_descriptor_fence_acquire(input_c32_map);
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro % kHybridC64C32BStages;
      int seq = macro / kHybridC64C32BStages;
      if (seq > 0) {
        while (!patchshift::mbarrier_try_wait(
            &shared.local_b_done[slot], (seq - 1) & 1)) {
        }
      }
      int local_td = macro / kMacrosPerTime;
      int macro_in_td = macro - local_td * kMacrosPerTime;
      bool is_c64_macro = macro_in_td == 0;
      int td = td_begin + local_td;
      uint32_t b_bytes = uint32_t(
          kHybridExactP15SemanticRows * (is_c64_macro ? 64 : 32) *
          sizeof(Element));
      patchshift::mbarrier_arrive_expect_tx(
          &shared.b_stage[slot].ready, b_bytes);
      if (is_c64_macro) {
        patchshift::tma_load_5d(
            input_c64_map, &shared.b_stage[slot].ready,
            shared.b_stage[slot].raw + swizzled_b_c64_index(0, 0),
            0, q_base - 1, -1, od + td - 1, 0);
      } else {
        patchshift::tma_load_5d(
            input_c32_map, &shared.b_stage[slot].ready,
            shared.b_stage[slot].raw + kHybridC32RawOffset +
                swizzled_b_c32_index(0, 0),
            64, q_base - 1, -1, od + td - 1, 0);
      }
    }
  }

  // Rank 0 publishes one copy of every packed weight row to all ranks.
  if (cluster_rank == 0 && wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kMainM * kK * sizeof(Element);
    for (int half_task = 0; half_task < local_half_tasks; ++half_task) {
      int local_td = half_task / kC32Groups;
      int sg_in_td = half_task - local_td * kC32Groups;
      int td = td_begin + local_td;
      int full_sg = td * kC32Groups + sg_in_td;
 #pragma unroll
      for (int row = 0; row < 3; ++row) {
        if (half_task > 0) {
          while (!patchshift::mbarrier_try_wait(
              &shared.a_row[row].done, (half_task - 1) & 1)) {
          }
        }
        patchshift::mbarrier_arrive_expect_tx(
            &shared.a_row[row].ready, a_row_bytes);
#pragma unroll
        for (int rank = 1; rank < kClusterSize; ++rank) {
          patchshift::mbarrier_arrive_expect_tx_remote(
              &shared.a_row[row].ready, a_row_bytes, rank);
        }
        int weight_task = full_sg * 3 + row;
        patchshift::tma_load_5d_multicast(
            weight_k32_map, &shared.a_row[row].ready, kClusterMask,
            shared.a_row[row].a[0][0], 0, 0, 0, 0, weight_task);
      }
    }
  }

  if (wid == 2) {
    patchshift::tcgen05_alloc(&shared.tmem_base, kMainTmemColumns);
    __syncwarp();
    patchshift::tcgen05_relinquish_alloc_permit();
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro % kHybridC64C32BStages;
      int seq = macro / kHybridC64C32BStages;
      int local_td = macro / kMacrosPerTime;
      int macro_in_td = macro - local_td * kMacrosPerTime;
      bool is_c64_macro = macro_in_td == 0;
      int valid_halves = is_c64_macro ? 2 : 1;
      int half_task_base = local_td * kC32Groups + macro_in_td * 2;
      while (!patchshift::mbarrier_try_wait(
          &shared.b_stage[slot].ready, seq & 1)) {
      }
      patchshift::fence_view_async_shared();
      for (int half = 0; half < valid_halves; ++half) {
        int half_task = half_task_base + half;
#pragma unroll
        for (int row = 0; row < 3; ++row) {
          while (!patchshift::mbarrier_try_wait(
              &shared.a_row[row].ready, half_task & 1)) {
          }
          patchshift::fence_view_async_shared();
          issue_hybrid_c64_c32_row(
              shared.b_stage[slot], shared.a_row[row],
              is_c64_macro, half, row, shared.tmem_base,
              macro == 0 && half == 0 && row == 0);
          patchshift::tcgen05_commit_multicast(
              &shared.a_row[row].done, kClusterMask);
          if (half == valid_halves - 1 && row == 2) {
            patchshift::tcgen05_commit(&shared.local_b_done[slot]);
          }
        }
      }
    }
  }

  int final_macro = local_macros - 1;
  int final_slot = final_macro % kHybridC64C32BStages;
  int final_seq = final_macro / kHybridC64C32BStages;
  while (!patchshift::mbarrier_try_wait(
      &shared.local_b_done[final_slot], final_seq & 1)) {
  }
  __syncthreads();
  patchshift::tcgen05_fence_after_thread_sync();

  store_hybrid_m128_p16<false, false, true, false, false, false, true>(
      shared.tmem_base, output, wid, lane, od, 0, q_base, 0,
      kH, kW, kKout);
#else
  (void)input_c64_map;
  (void)input_c32_map;
  (void)weight_k32_map;
  (void)output;
#endif
}
