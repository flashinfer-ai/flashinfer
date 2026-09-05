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

// Native M32/P16 small-grid path with C64 activation macro stages.
// Included after output_tail.cuh and small_grid.cuh.

struct M32P16C64SharedStorage {
  K64C64B2A3K32ABStage b_stage[kK64C64B2A3K32ABRing];
  M32P16ARowStage a_stage[3];
  uint32_t tmem_base;
  volatile int b_published;
  volatile int a_published;
  volatile int commit_issued[kK64C64B2A3K32ABRing]
                            [kK32HalvesPerK64Macro][3];
  volatile int a_release_observed;
  volatile int tmem_ready;
};

static_assert(sizeof(M32P16C64SharedStorage) <= 232448,
              "M32/P16 C64 macro pipeline must fit one SM100 CTA");

__device__ __forceinline__ void issue_m32_p16_c64_workset_row(
    K64C64B2A3K32ABStage& b_stage,
    M32P16ARowStage& a_stage,
    int k32_half,
    int filter_row,
    int workset,
    uint32_t tmem_base,
    bool first_row) {
  if (!patchshift::elect_one_sync()) {
    return;
  }
  constexpr uint64_t idesc = m32_p16_idesc();
  uint32_t dst =
      tmem_base + uint32_t(workset * kM32P16AccumulatorColumns);
#pragma unroll
  for (int kg = 0; kg < kK16GroupsPerStage; ++kg) {
    int k_offset = k32_half * 32 + kg * kK;
    Element* b = b_stage.b + swizzled_b_c64_index(
        workset * kM32P16N + filter_row * kPitch, k_offset);
    uint64_t desc_b = pack_b_c64_desc(b);
#pragma unroll
    for (int kw = 0; kw < 3; ++kw) {
      uint64_t desc_a =
          patchshift::pack_k16_desc(a_stage.a[kw][kg], kM32P16M);
      bool first = first_row && kg == 0 && kw == 0;
      mma_ws_raw(desc_a, desc_b, dst, first ? 0u : 1u, idesc,
                 patchshift::shift_desc(kw));
    }
  }
}

__global__ void general_m32n256_k64_p16_b2a3_c64_kernel(
    TensorMap const* input_c64_map,
    TensorMap const* weight_k32_map,
    Element* output,
    int n_size,
    int d_size,
    int h_size,
    int w_size,
    int c64_groups_per_time,
    int output_pitch_k,
    int output_channel_base) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  __shared__ M32P16C64SharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  int q_base = int(blockIdx.x) * kOutQ;
  int p_base = int(blockIdx.y) * kM32P16OutP;
  int flat_batch_count = n_size * d_size;
  int m_tile = int(blockIdx.z) / flat_batch_count;
  int flat_batch = int(blockIdx.z) - m_tile * flat_batch_count;
  int n = flat_batch / d_size;
  int od = flat_batch - n * d_size;
  int k_base = output_channel_base + m_tile * kM32P16M;

  int td_begin = od == 0 ? 1 : 0;
  int td_end = od == d_size - 1 ? 2 : kT;
  int local_td_count = td_end - td_begin;
  int local_macros = local_td_count * c64_groups_per_time;
  int c32_groups_per_time = c64_groups_per_time * 2;
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
    shared.b_published = 0;
    shared.a_published = 0;
    shared.a_release_observed = 0;
    shared.tmem_ready = 0;
#pragma unroll
    for (int slot = 0; slot < kK64C64B2A3K32ABRing; ++slot) {
      patchshift::mbarrier_init(&shared.b_stage[slot].ready, 1);
#pragma unroll
      for (int half = 0; half < kK32HalvesPerK64Macro; ++half) {
#pragma unroll
        for (int row = 0; row < 3; ++row) {
          shared.commit_issued[slot][half][row] = 0;
          patchshift::mbarrier_init(
              &shared.b_stage[slot].half_row_done[half][row],
              kM32P16Worksets);
        }
      }
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      patchshift::mbarrier_init(&shared.a_stage[row].ready, 1);
    }
  }
  __syncthreads();

  if (wid == 0 && lane == 0) {
    patchshift::tma_descriptor_fence_acquire(input_c64_map);
    constexpr uint32_t b_bytes =
        kMainSemanticRows * 64 * sizeof(Element);
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro % kK64C64B2A3K32ABRing;
      int seq = macro / kK64C64B2A3K32ABRing;
      if (seq > 0) {
        int old_macro = macro - kK64C64B2A3K32ABRing;
        while (shared.a_release_observed < old_macro + 1) {
        }
        while (shared.commit_issued[slot][1][2] < seq) {
        }
        __threadfence_block();
        while (!patchshift::mbarrier_try_wait(
            &shared.b_stage[slot].half_row_done[1][2],
            (seq - 1) & 1)) {
        }
      }
      int local_td = macro / c64_groups_per_time;
      int c64g = macro - local_td * c64_groups_per_time;
      int td = td_begin + local_td;
      patchshift::mbarrier_arrive_expect_tx(
          &shared.b_stage[slot].ready, b_bytes);
      patchshift::tma_load_5d(
          input_c64_map, &shared.b_stage[slot].ready,
          shared.b_stage[slot].b + swizzled_b_c64_index(0, 0),
          c64g * 64, q_base - 1, p_base - 1,
          od + td - 1, n);
      __threadfence_block();
      shared.b_published = macro + 1;
    }
  }

  if (wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kM32P16M * kK * sizeof(Element);
    int half_tasks = local_macros * kK32HalvesPerK64Macro;
    for (int half_task = 0; half_task < half_tasks; ++half_task) {
      int macro = half_task / kK32HalvesPerK64Macro;
      int half = half_task % kK32HalvesPerK64Macro;
      int local_td = macro / c64_groups_per_time;
      int c64g = macro - local_td * c64_groups_per_time;
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
          while (shared.commit_issued[previous_slot]
                                      [previous_half][row] <
                 previous_seq + 1) {
          }
          __threadfence_block();
          while (!patchshift::mbarrier_try_wait(
              &shared.b_stage[previous_slot]
                   .half_row_done[previous_half][row],
              previous_seq & 1)) {
          }
          if (previous_half == 1 && row == 2) {
            __threadfence_block();
            shared.a_release_observed = previous_macro + 1;
          }
        }
        patchshift::mbarrier_arrive_expect_tx(
            &shared.a_stage[row].ready, a_row_bytes);
        int weight_task =
            (m_tile * full_k32_supergroups + full_sg) * 3 + row;
        patchshift::tma_load_5d(
            weight_k32_map, &shared.a_stage[row].ready,
            shared.a_stage[row].a[0][0],
            0, 0, 0, 0, weight_task);
        __threadfence_block();
        shared.a_published = half_task * 3 + row + 1;
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
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro % kK64C64B2A3K32ABRing;
      int seq = macro / kK64C64B2A3K32ABRing;
      while (shared.b_published < macro + 1) {
      }
      while (!patchshift::mbarrier_try_wait(
          &shared.b_stage[slot].ready, seq & 1)) {
      }
      patchshift::fence_view_async_shared();
#pragma unroll
      for (int half = 0; half < kK32HalvesPerK64Macro; ++half) {
        int half_task = macro * kK32HalvesPerK64Macro + half;
#pragma unroll
        for (int row = 0; row < 3; ++row) {
          while (shared.a_published < half_task * 3 + row + 1) {
          }
          while (!patchshift::mbarrier_try_wait(
              &shared.a_stage[row].ready, half_task & 1)) {
          }
          patchshift::fence_view_async_shared();
          issue_m32_p16_c64_workset_row(
              shared.b_stage[slot], shared.a_stage[row], half, row,
              workset, shared.tmem_base,
              macro == 0 && half == 0 && row == 0);
          patchshift::tcgen05_commit(
              &shared.b_stage[slot].half_row_done[half][row]);
          if (workset == 0 && lane == 0) {
            __threadfence_block();
            shared.commit_issued[slot][half][row] = seq + 1;
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
  while (shared.commit_issued[final_slot][1][2] < final_seq + 1) {
  }
  __threadfence_block();
  while (!patchshift::mbarrier_try_wait(
      &shared.b_stage[final_slot].half_row_done[1][2],
      final_seq & 1)) {
  }
  patchshift::tcgen05_fence_after_thread_sync();

  int store_group = wid & 3;
  int local_k = lane;
  int global_k = k_base + local_k;
  bool full_tile = k_base + kM32P16M <= output_pitch_k &&
                   p_base + kM32P16OutP <= h_size &&
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
      if (full_tile) {
        size_t pixel =
            ((size_t(flat_batch) * size_t(h_size) + size_t(out_p)) *
                 size_t(w_size) + size_t(q_base));
        Element* out = output + pixel * size_t(output_pitch_k) + k_base;
#pragma unroll
        for (int q = 0; q < kOutQ; ++q) {
          uint16_t bits = patchshift::element_bits_from_float(
              __uint_as_float(values[q]));
          *reinterpret_cast<uint16_t*>(
              out + size_t(q) * size_t(output_pitch_k) + local_k) = bits;
        }
      } else if (out_p < h_size && global_k < output_pitch_k) {
#pragma unroll
        for (int q = 0; q < kOutQ; ++q) {
          int out_q = q_base + q;
          if (out_q < w_size) {
            size_t pixel =
                ((size_t(flat_batch) * size_t(h_size) + size_t(out_p)) *
                     size_t(w_size) + size_t(out_q));
            uint16_t bits = patchshift::element_bits_from_float(
                __uint_as_float(values[q]));
            *reinterpret_cast<uint16_t*>(
                output + pixel * size_t(output_pitch_k) + global_k) = bits;
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
  (void)n_size;
  (void)d_size;
  (void)h_size;
  (void)w_size;
  (void)c64_groups_per_time;
  (void)output_pitch_k;
  (void)output_channel_base;
#endif
}
