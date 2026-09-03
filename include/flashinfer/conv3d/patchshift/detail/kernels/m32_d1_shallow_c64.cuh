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

// D1/C128/K128 shallow M32 pipeline. Four K32 half-tasks have private A rows;
// neither A nor the two C64 B macros are recycled inside the CTA.

constexpr int kM32D1C128HalfTasks = 4;

struct alignas(1024) M32D1C128ShallowSharedStorage {
  K64C64B2A3K32ABStage b_stage[2];
  M32P16ARowStage a_stage[kM32D1C128HalfTasks][3];
  alignas(8) uint64_t final_done;
  uint32_t tmem_base;
  volatile int tmem_ready;
};

static_assert(sizeof(M32D1C128ShallowSharedStorage) <= 232448,
              "D1 C128 shallow M32 pipeline must fit one SM100 CTA");

template <bool ExactFull = false, bool ClusterK4 = false>
__global__
void general_m32n256_d1_c128_shallow_c64_kernel(
    TensorMap const* input_c64_map,
    TensorMap const* weight_k32_map,
    Element* output,
    int h_size,
    int w_size) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  __shared__ M32D1C128ShallowSharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  uint32_t cluster_rank =
      ClusterK4 ? cute::block_rank_in_cluster() : 0u;
  int q_tile = ClusterK4 ? (int(blockIdx.x) >> 2) : int(blockIdx.x);
  int q_base = q_tile * kOutQ;
  int p_base = int(blockIdx.y) * kM32P16OutP;
  int m_tile = ClusterK4 ? int(cluster_rank) : int(blockIdx.z);
  int k_base = m_tile * kM32P16M;

  if constexpr (!ExactFull) {
    constexpr int guard_rows = kMainBackingRows - kMainSemanticRows;
    constexpr int guard_per_stage = guard_rows * 64;
    for (int idx = int(threadIdx.x); idx < 2 * guard_per_stage;
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
    patchshift::mbarrier_init(&shared.final_done, kM32P16Worksets);
#pragma unroll
    for (int macro = 0; macro < 2; ++macro) {
      patchshift::mbarrier_init(&shared.b_stage[macro].ready, 1);
    }
#pragma unroll
    for (int half_task = 0; half_task < kM32D1C128HalfTasks;
         ++half_task) {
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        patchshift::mbarrier_init(
            &shared.a_stage[half_task][row].ready, 1);
      }
    }
  }
  __syncthreads();
  if constexpr (ClusterK4) {
    cute::cluster_sync();
  }
  // D1 with pad1 consumes only td=1. C128 is exactly two C64 macros.
  if ((!ClusterK4 || cluster_rank == 0) && wid == 0 && lane == 0) {
    constexpr uint32_t b_bytes =
        kMainSemanticRows * 64 * sizeof(Element);
#pragma unroll
    for (int macro = 0; macro < 2; ++macro) {
      patchshift::mbarrier_arrive_expect_tx(
          &shared.b_stage[macro].ready, b_bytes);
      if constexpr (ClusterK4) {
#pragma unroll
        for (int rank = 1; rank < 4; ++rank) {
          patchshift::mbarrier_arrive_expect_tx_remote(
              &shared.b_stage[macro].ready, b_bytes, rank);
        }
        patchshift::tma_load_5d_multicast(
            input_c64_map, &shared.b_stage[macro].ready, 0x0fu,
            shared.b_stage[macro].b + swizzled_b_c64_index(0, 0),
            macro * 64, q_base - 1, p_base - 1, 0, 0);
      } else {
        patchshift::tma_load_5d(
            input_c64_map, &shared.b_stage[macro].ready,
            shared.b_stage[macro].b + swizzled_b_c64_index(0, 0),
            macro * 64, q_base - 1, p_base - 1, 0, 0);
      }
    }
  }

  if (wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kM32P16M * kK * sizeof(Element);
    constexpr int c32_groups_per_time = 4;
    constexpr int full_supergroups = kT * c32_groups_per_time;
#pragma unroll
    for (int half_task = 0; half_task < kM32D1C128HalfTasks;
         ++half_task) {
      int full_sg = c32_groups_per_time + half_task;
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        patchshift::mbarrier_arrive_expect_tx(
            &shared.a_stage[half_task][row].ready, a_row_bytes);
        int weight_task =
            (m_tile * full_supergroups + full_sg) * 3 + row;
        patchshift::tma_load_5d(
            weight_k32_map,
            &shared.a_stage[half_task][row].ready,
            shared.a_stage[half_task][row].a[0][0],
            0, 0, 0, 0, weight_task);
      }
    }
  }

  if (wid == 2) {
    patchshift::tcgen05_alloc(&shared.tmem_base, kM32P16TmemColumns);
    __syncwarp();
    patchshift::tcgen05_relinquish_alloc_permit();
    if (lane == 0) {
      __threadfence_block();
      shared.tmem_ready = 1;
    }
  }
  if (wid >= 2 && wid < 2 + kM32P16Worksets) {
    int workset = wid - 2;
    if (wid != 2) {
      while (shared.tmem_ready == 0) {
      }
      __threadfence_block();
    }
#pragma unroll
    for (int half_task = 0; half_task < kM32D1C128HalfTasks;
         ++half_task) {
      int macro = half_task >> 1;
      int half = half_task & 1;
      while (!patchshift::mbarrier_try_wait(
          &shared.b_stage[macro].ready, 0)) {
      }
      patchshift::fence_view_async_shared();
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        while (!patchshift::mbarrier_try_wait(
            &shared.a_stage[half_task][row].ready, 0)) {
        }
        patchshift::fence_view_async_shared();
        issue_m32_p16_c64_workset_row(
            shared.b_stage[macro], shared.a_stage[half_task][row],
            half, row, workset, shared.tmem_base,
            half_task == 0 && row == 0);
      }
    }
    patchshift::tcgen05_commit(&shared.final_done);
  }

  while (shared.tmem_ready == 0) {
  }
  while (!patchshift::mbarrier_try_wait(&shared.final_done, 0)) {
  }
  patchshift::tcgen05_fence_after_thread_sync();

  int store_group = wid & 3;
  int local_k = lane;
  int global_k = k_base + local_k;
  bool full_tile = p_base + kM32P16OutP <= h_size &&
                   q_base + kOutQ <= w_size;
#pragma unroll
  for (int workset = 0; workset < kM32P16Worksets; ++workset) {
    uint32_t tile_base =
        shared.tmem_base + uint32_t(workset * kM32P16AccumulatorColumns);
    for (int physical_col = 0;
         physical_col < kM32P16AccumulatorColumns;
         physical_col += 32) {
      uint32_t values[32];
      patchshift::tcgen05_load_32dp32b_x32(
          tile_base + physical_col, values);
      patchshift::tcgen05_wait_tmem_load();
      int logical_col =
          store_group * kM32P16AccumulatorColumns + physical_col;
      int out_p = p_base + workset * kM32P16OutPPerWorkset +
                  (logical_col >> 5);
      if constexpr (ExactFull) {
        size_t pixel =
            (size_t(out_p) * size_t(w_size) + size_t(q_base));
        Element* out = output + pixel * size_t(kMainM) + size_t(k_base);
#pragma unroll
        for (int q = 0; q < kOutQ; ++q) {
          uint16_t bits = patchshift::element_bits_from_float(
              __uint_as_float(values[q]));
          *reinterpret_cast<uint16_t*>(
              out + size_t(q) * size_t(kMainM) + local_k) = bits;
        }
      } else if (full_tile) {
        size_t pixel =
            (size_t(out_p) * size_t(w_size) + size_t(q_base));
        Element* out = output + pixel * size_t(kMainM) + size_t(k_base);
#pragma unroll
        for (int q = 0; q < kOutQ; ++q) {
          uint16_t bits = patchshift::element_bits_from_float(
              __uint_as_float(values[q]));
          *reinterpret_cast<uint16_t*>(
              out + size_t(q) * size_t(kMainM) + local_k) = bits;
        }
      } else if (out_p < h_size && global_k < kMainM) {
#pragma unroll
        for (int q = 0; q < kOutQ; ++q) {
          int out_q = q_base + q;
          if (out_q < w_size) {
            size_t pixel =
                size_t(out_p) * size_t(w_size) + size_t(out_q);
            uint16_t bits = patchshift::element_bits_from_float(
                __uint_as_float(values[q]));
            *reinterpret_cast<uint16_t*>(
                output + pixel * size_t(kMainM) + size_t(global_k)) = bits;
          }
        }
      }
    }
  }
  patchshift::tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (wid == kM32P16Warps - 1) {
    patchshift::tcgen05_fence_after_thread_sync();
    patchshift::tcgen05_dealloc(
        shared.tmem_base, kM32P16TmemColumns);
  }
#else
  (void)input_c64_map;
  (void)weight_k32_map;
  (void)output;
  (void)h_size;
  (void)w_size;
#endif
}
