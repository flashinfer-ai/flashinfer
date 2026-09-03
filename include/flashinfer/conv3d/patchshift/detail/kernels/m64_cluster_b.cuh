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

// Logical M128 cluster-B path built from two legal 1-SM M64/P16 CTAs.
// Rank 0 multicasts the common activation tile; each rank loads the weights
// for one adjacent M64 output-channel interval.  The arithmetic instruction
// remains cta_group::1 tcgen05.mma.ws with bshift.

struct M64ClusterBC64SharedStorage {
  K64C64B2A3K32ABStage b_stage[2];
  M64P16ARowStage a_row[2][3];
  // B reuse must prove release by all four M64 worksets in the two-rank
  // cluster. A reuse is CTA-local: one barrier per two-entry A slot collects
  // the two local workset commits after all three filter rows have issued.
  uint64_t b_done[2];
  uint64_t a_done[2];
  uint32_t tmem_base;
  volatile int tmem_ready;
};

static_assert(sizeof(M64ClusterBC64SharedStorage) <= 232448,
              "M64 cluster-B C64 pipeline must fit one SM100 CTA");

template <int ExactD = 0, bool EightWarpStore = false,
          bool WideD2Store = false>
__global__ __launch_bounds__(EightWarpStore ? 256 : kM64P16Threads, 1)
void general_m128_cluster_b_m64_p16_c64_kernel(
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
  static_assert(ExactD == 0 || ExactD == 2 || ExactD == 3);
  constexpr bool kExactDepth = ExactD != 0;
  __shared__ M64ClusterBC64SharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  uint32_t cluster_rank = cute::block_rank_in_cluster();
  int q_tile = int(blockIdx.x) >> 1;
  int q_base = q_tile * kOutQ;
  int p_base = int(blockIdx.y) * kM64P16OutP;
  int flat_batch = int(blockIdx.z);
  int effective_d = kExactDepth ? ExactD : d_size;
  int n = kExactDepth ? 0 : flat_batch / effective_d;
  int od = kExactDepth ? flat_batch
                        : flat_batch - n * effective_d;
  int k_half = int(cluster_rank);
  int k_base = k_half * kTailM;

  int td_begin = od == 0 ? 1 : 0;
  int td_end = od == effective_d - 1 ? 2 : kT;
  int local_td_count = td_end - td_begin;
  int effective_c64_groups = kExactDepth ? 2 : c64_groups_per_time;
  int local_macros = local_td_count * effective_c64_groups;
  int c32_groups_per_time = effective_c64_groups * 2;
  int full_k32_supergroups = kT * c32_groups_per_time;

  if constexpr (!kExactDepth) {
    constexpr int guard_rows = kMainBackingRows - kMainSemanticRows;
    constexpr int guard_per_stage = guard_rows * 64;
    for (int idx = int(threadIdx.x);
         idx < 2 * guard_per_stage;
         idx += int(blockDim.x)) {
      int slot = idx / guard_per_stage;
      int rest = idx - slot * guard_per_stage;
      int row = kMainSemanticRows + rest / 64;
      int kk = rest % 64;
      shared.b_stage[slot].b[swizzled_b_c64_index(row, kk)] =
          patchshift::element_from_float(0.0f);
    }
  }
  if (threadIdx.x == 0) {
    shared.tmem_base = 0;
    shared.tmem_ready = 0;
    for (int slot = 0; slot < 2; ++slot) {
      patchshift::mbarrier_init(&shared.b_stage[slot].ready, 1);
      patchshift::mbarrier_init(
          &shared.b_done[slot], 2 * kM64P16Worksets);
      patchshift::mbarrier_init(&shared.a_done[slot], 2);
    }
    for (int a_slot = 0; a_slot < 2; ++a_slot) {
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        patchshift::mbarrier_init(&shared.a_row[a_slot][row].ready, 1);
      }
    }
  }
  __syncthreads();
  cute::cluster_sync();

  if (cluster_rank == 0 && wid == 0 && lane == 0) {
    constexpr uint32_t b_bytes =
        kMainSemanticRows * 64 * sizeof(Element);
    constexpr uint16_t b_cluster_mask = 0x3u;
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro & 1;
      int seq = macro >> 1;
      if (seq > 0) {
        while (!patchshift::mbarrier_try_wait(
            &shared.b_done[slot], (seq - 1) & 1)) {
        }
      }
      int local_td = macro / effective_c64_groups;
      int c64g = macro - local_td * effective_c64_groups;
      int td = td_begin + local_td;
      int input_d = od + td - 1;
      patchshift::mbarrier_arrive_expect_tx(
          &shared.b_stage[slot].ready, b_bytes);
      patchshift::mbarrier_arrive_expect_tx_remote(
          &shared.b_stage[slot].ready, b_bytes, 1);
      patchshift::tma_load_5d_multicast(
          input_c64_map, &shared.b_stage[slot].ready, b_cluster_mask,
          shared.b_stage[slot].b + swizzled_b_c64_index(0, 0),
          c64g * 64, q_base - 1, p_base - 1,
          input_d, n);
    }
  }

  if (wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kTailM * kK * sizeof(Element);
    int half_tasks = local_macros * kK32HalvesPerK64Macro;
    for (int half_task = 0; half_task < half_tasks; ++half_task) {
      int macro = half_task / kK32HalvesPerK64Macro;
      int half = half_task % kK32HalvesPerK64Macro;
      int local_td = macro / effective_c64_groups;
      int c64g = macro - local_td * effective_c64_groups;
      int td = td_begin + local_td;
      int full_sg = td * c32_groups_per_time + c64g * 2 + half;
      int a_slot = half_task & 1;
      int a_seq = half_task >> 1;
      if (a_seq > 0) {
        while (!patchshift::mbarrier_try_wait(
            &shared.a_done[a_slot], (a_seq - 1) & 1)) {
        }
      }
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        patchshift::mbarrier_arrive_expect_tx(
            &shared.a_row[a_slot][row].ready, a_row_bytes);
        int weight_task =
            (k_half * full_k32_supergroups + full_sg) * 3 + row;
        patchshift::tma_load_5d(
            weight_k32_map, &shared.a_row[a_slot][row].ready,
            shared.a_row[a_slot][row].a[0][0],
            0, 0, 0, 0, weight_task);
      }
    }
  }

  if (wid == 2) {
    patchshift::tcgen05_alloc(&shared.tmem_base, kM64P16TmemColumns);
    __syncwarp();
    patchshift::tcgen05_relinquish_alloc_permit();
    if (lane == 0) {
      __threadfence_block();
      shared.tmem_ready = 1;
    }
  }
  if (wid >= 2 && wid < 2 + kM64P16Worksets) {
    int workset = wid - 2;
    if (wid != 2) {
      while (shared.tmem_ready == 0) {
      }
      __threadfence_block();
    }
    constexpr uint16_t b_cluster_mask = 0x3u;
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro & 1;
      int seq = macro >> 1;
      while (!patchshift::mbarrier_try_wait(
          &shared.b_stage[slot].ready, seq & 1)) {
      }
      patchshift::fence_view_async_shared();
#pragma unroll
      for (int half = 0; half < kK32HalvesPerK64Macro; ++half) {
        int half_task = macro * kK32HalvesPerK64Macro + half;
        int a_slot = half_task & 1;
        int a_seq = half_task >> 1;
#pragma unroll
        for (int row = 0; row < 3; ++row) {
          while (!patchshift::mbarrier_try_wait(
              &shared.a_row[a_slot][row].ready, a_seq & 1)) {
          }
          patchshift::fence_view_async_shared();
          issue_m64_p16_c64_workset_row(
              shared.b_stage[slot], shared.a_row[a_slot][row], half, row,
              workset, shared.tmem_base,
              macro == 0 && half == 0 && row == 0);
          if (half == kK32HalvesPerK64Macro - 1 && row == 2) {
            patchshift::tcgen05_commit_multicast(
                &shared.b_done[slot], b_cluster_mask);
          }
        }
        patchshift::tcgen05_commit(&shared.a_done[a_slot]);
      }
    }
  }

  while (shared.tmem_ready == 0) {
  }
  int final_macro = local_macros - 1;
  int final_slot = final_macro & 1;
  int final_seq = final_macro >> 1;
  while (!patchshift::mbarrier_try_wait(
      &shared.b_done[final_slot], final_seq & 1)) {
  }
  __syncthreads();
  patchshift::tcgen05_fence_after_thread_sync();

  int store_warp = wid & 1;
  int store_group = (wid >> 1) & 1;
  int store_partition = EightWarpStore ? (wid >> 2) : 0;
  int local_k = store_warp * 32 + lane;
  if constexpr (ExactD == 2) {
#pragma unroll
    for (int workset = 0; workset < kM64P16Worksets; ++workset) {
      uint32_t tile_base =
          shared.tmem_base + uint32_t(workset * kTailAccumulatorColumns);
      if constexpr (WideD2Store) {
        uint32_t values[128];
        patchshift::tcgen05_load_32dp32b_x128(tile_base, values);
        patchshift::tcgen05_wait_tmem_load();
#pragma unroll
        for (int row = 0; row < 4; ++row) {
          int logical_col =
              store_group * kTailAccumulatorColumns + row * 32;
          int out_p = p_base + workset * kTailOutPPerWorkset +
                      (logical_col >> 5);
          size_t pixel =
              (size_t(flat_batch) * 128u + size_t(out_p)) * 120u +
              size_t(q_base);
          Element* out = output + pixel * 128u + k_base;
#pragma unroll
          for (int q = 0; q < kOutQ; ++q) {
            uint16_t bits = patchshift::element_bits_from_float(
                __uint_as_float(values[row * 32 + q]));
            *reinterpret_cast<uint16_t*>(
                out + size_t(q) * 128u + local_k) = bits;
          }
        }
      } else {
          for (int physical_col = store_partition * 64;
             physical_col < kTailAccumulatorColumns;
             physical_col += EightWarpStore ? 128 : 64) {
          uint32_t values[64];
          patchshift::tcgen05_load_32dp32b_x64(
              tile_base + physical_col, values);
          patchshift::tcgen05_wait_tmem_load();
#pragma unroll
          for (int row = 0; row < 2; ++row) {
            int logical_col = store_group * kTailAccumulatorColumns +
                              physical_col + row * 32;
            int out_p = p_base + workset * kTailOutPPerWorkset +
                        (logical_col >> 5);
            size_t pixel =
                (size_t(flat_batch) * 128u + size_t(out_p)) * 120u +
                size_t(q_base);
            Element* out = output + pixel * 128u + k_base;
#pragma unroll
            for (int q = 0; q < kOutQ; ++q) {
              uint16_t bits = patchshift::element_bits_from_float(
                  __uint_as_float(values[row * 32 + q]));
              *reinterpret_cast<uint16_t*>(
                  out + size_t(q) * 128u + local_k) = bits;
            }
          }
        }
      }
    }
  } else {
#pragma unroll
    for (int workset = 0; workset < kM64P16Worksets; ++workset) {
      uint32_t tile_base =
          shared.tmem_base + uint32_t(workset * kTailAccumulatorColumns);
      for (int physical_col = store_partition * 32;
           physical_col < kTailAccumulatorColumns;
           physical_col += EightWarpStore ? 64 : 32) {
        uint32_t values[32];
        patchshift::tcgen05_load_32dp32b_x32(
            tile_base + physical_col, values);
        patchshift::tcgen05_wait_tmem_load();
        int logical_col =
            store_group * kTailAccumulatorColumns + physical_col;
        int out_p = p_base + workset * kTailOutPPerWorkset +
                    (logical_col >> 5);
        size_t pixel = kExactDepth
                           ? (size_t(flat_batch) * 128u + size_t(out_p)) *
                                 120u + size_t(q_base)
                           : ((size_t(flat_batch) * size_t(h_size) +
                               size_t(out_p)) * size_t(w_size) +
                              size_t(q_base));
        Element* out = output +
                       pixel * size_t(kExactDepth ? 128 : k_size) +
                       k_base;
#pragma unroll
        for (int q = 0; q < kOutQ; ++q) {
          uint16_t bits = patchshift::element_bits_from_float(
              __uint_as_float(values[q]));
          *reinterpret_cast<uint16_t*>(
              out + size_t(q) *
                        size_t(kExactDepth ? 128 : k_size) +
                  local_k) = bits;
        }
      }
    }
  }
  patchshift::tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (wid == kM64P16Warps - 1) {
    patchshift::tcgen05_fence_after_thread_sync();
    patchshift::tcgen05_dealloc(
        shared.tmem_base, kM64P16TmemColumns);
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
