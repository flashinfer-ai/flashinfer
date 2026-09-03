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

// D1/C32 one-supergroup micro path.
// Included after small_grid.cuh by the PatchShift kernel umbrella.
//
// Boundary elimination leaves one K32 supergroup. M32N128 maps one CTA to
// P4xQ30, exposing 96 lightweight CTAs for the 64x64/K64 target while one B,
// three A rows and one completion barrier remove all unused ring state.
constexpr int kM32MicroN = 128;
constexpr int kM32MicroPitch = 32;
constexpr int kM32MicroOutQ = 30;
constexpr int kM32MicroOutP = kM32MicroN / kM32MicroPitch;
constexpr int kM32MicroInputP = kM32MicroOutP + 2;
constexpr int kM32MicroSemanticRows =
    kM32MicroInputP * kM32MicroPitch;
constexpr int kM32MicroRequiredRows =
    kM32MicroN + 2 * kM32MicroPitch + 2;
constexpr int kM32MicroBackingRows =
    ((kM32MicroRequiredRows + 7) / 8) * 8;
constexpr int kM32MicroAccumulatorColumns = kM32MicroN / 4;
constexpr int kM32MicroTmemColumns = 32;
constexpr int kM32MicroWarps = 4;
constexpr int kM32MicroThreads = kM32MicroWarps * 32;
static_assert(kM32MicroOutP == 4 &&
              kM32MicroSemanticRows == 192 &&
              kM32MicroRequiredRows == 194 &&
              kM32MicroBackingRows == 200 &&
              kM32MicroTmemColumns == 32);

struct alignas(512) M32MicroBStage {
  alignas(512) Element b[kM32MicroBackingRows * 32];
  uint64_t ready;
};

struct alignas(128) M32MicroARow {
  alignas(128) Element a[3][kK16GroupsPerStage][kM32P16M * kK];
  uint64_t ready;
};

struct alignas(512) M32MicroSharedStorage {
  M32MicroBStage b_stage;
  M32MicroARow a_row[3];
  uint64_t done;
  uint32_t tmem_base;
};

static_assert(sizeof(M32MicroSharedStorage) <= 49152,
              "one-supergroup micro path must stay lightweight");

__host__ __device__ constexpr uint64_t m32_micro_idesc() {
  constexpr uint32_t c_format_f32 = 1u;
  constexpr uint32_t ab_format_bf16 = 1u;
  constexpr uint32_t n_dim = kM32MicroN >> 3;
  constexpr uint32_t m_dim = kM32P16M >> 4;
  constexpr uint32_t max_shift16 = 2u;
  uint32_t desc = 0;
  desc |= c_format_f32 << 4;
  desc |= ab_format_bf16 << 7;
  desc |= ab_format_bf16 << 10;
  desc |= n_dim << 17;
  desc |= m_dim << 24;
  desc |= max_shift16 << 30;
  return uint64_t(desc) << 32;
}

__device__ __forceinline__ void issue_m32_micro_row(
    M32MicroBStage& b_stage,
    M32MicroARow& a_row,
    int filter_row,
    uint32_t tmem_base,
    bool first_row) {
  if (!patchshift::elect_one_sync()) {
    return;
  }
  constexpr uint64_t idesc = m32_micro_idesc();
#pragma unroll
  for (int kg = 0; kg < kK16GroupsPerStage; ++kg) {
    Element* b = b_stage.b + swizzled_b_c32_index(
        filter_row * kM32MicroPitch, kg * kK);
    uint64_t desc_b = pack_b_c32_desc(b);
#pragma unroll
    for (int kw = 0; kw < 3; ++kw) {
      uint64_t desc_a =
          patchshift::pack_k16_desc(a_row.a[kw][kg], kM32P16M);
      bool first = first_row && kg == 0 && kw == 0;
      mma_ws_raw(desc_a, desc_b, tmem_base,
                 first ? 0u : 1u, idesc,
                 patchshift::shift_desc(kw));
    }
  }
}

__global__ void general_m32n128_d1_c32_micro_kernel(
    TensorMap const* input_c32_map,
    TensorMap const* weight_map,
    Element* output,
    int h_size,
    int w_size,
    int output_pitch_k) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  __shared__ M32MicroSharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  int q_base = int(blockIdx.x) * kM32MicroOutQ;
  int p_base = int(blockIdx.y) * kM32MicroOutP;
  int m32_tile = int(blockIdx.z);
  int k_base = m32_tile * kM32P16M;

  constexpr int guard_rows =
      kM32MicroBackingRows - kM32MicroSemanticRows;
  for (int idx = int(threadIdx.x); idx < guard_rows * 32;
       idx += int(blockDim.x)) {
    int row = kM32MicroSemanticRows + idx / 32;
    int kk = idx % 32;
    shared.b_stage.b[swizzled_b_c32_index(row, kk)] =
        patchshift::element_from_float(0.0f);
  }
  if (threadIdx.x == 0) {
    patchshift::mbarrier_init(&shared.b_stage.ready, 1);
    patchshift::mbarrier_init(&shared.done, 1);
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      patchshift::mbarrier_init(&shared.a_row[row].ready, 1);
    }
  }
  if (wid == 2) {
    patchshift::tcgen05_alloc(
        &shared.tmem_base, kM32MicroTmemColumns);
    __syncwarp();
    patchshift::tcgen05_relinquish_alloc_permit();
  }
  __syncthreads();

  if (wid == 0 && lane == 0) {
    constexpr uint32_t b_bytes =
        kM32MicroSemanticRows * 32 * sizeof(Element);
    patchshift::mbarrier_arrive_expect_tx(
        &shared.b_stage.ready, b_bytes);
    patchshift::tma_load_5d(
        input_c32_map, &shared.b_stage.ready,
        shared.b_stage.b + swizzled_b_c32_index(0, 0),
        0, q_base - 1, p_base - 1, 0, 0);
  }

  if (wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kM32P16M * kK * sizeof(Element);
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      patchshift::mbarrier_arrive_expect_tx(
          &shared.a_row[row].ready, a_row_bytes);
      int weight_task = (m32_tile * kT + 1) * 3 + row;
      patchshift::tma_load_5d(
          weight_map, &shared.a_row[row].ready,
          shared.a_row[row].a[0][0], 0, 0, 0, 0, weight_task);
    }
  }

  if (wid == 2) {
    while (!patchshift::mbarrier_try_wait(&shared.b_stage.ready, 0)) {
    }
    patchshift::fence_view_async_shared();
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      while (!patchshift::mbarrier_try_wait(
          &shared.a_row[row].ready, 0)) {
      }
      patchshift::fence_view_async_shared();
      issue_m32_micro_row(
          shared.b_stage, shared.a_row[row], row,
          shared.tmem_base, row == 0);
    }
    patchshift::tcgen05_commit(&shared.done);
  }

  while (!patchshift::mbarrier_try_wait(&shared.done, 0)) {
  }
  patchshift::tcgen05_fence_after_thread_sync();

  uint32_t values[32];
  patchshift::tcgen05_load_32dp32b_x32(shared.tmem_base, values);
  patchshift::tcgen05_wait_tmem_load();
  int out_p = p_base + (wid & 3);
  bool full_tile = p_base + kM32MicroOutP <= h_size &&
                   q_base + kM32MicroOutQ <= w_size;
  if (full_tile) {
    size_t pixel = size_t(out_p) * size_t(w_size) + size_t(q_base);
    Element* out = output + pixel * size_t(output_pitch_k) + k_base;
#pragma unroll
    for (int q = 0; q < kM32MicroOutQ; ++q) {
      uint16_t bits = patchshift::element_bits_from_float(
          __uint_as_float(values[q]));
      *reinterpret_cast<uint16_t*>(
          out + size_t(q) * size_t(output_pitch_k) + lane) = bits;
    }
  } else if (out_p < h_size) {
#pragma unroll
    for (int q = 0; q < kM32MicroOutQ; ++q) {
      int out_q = q_base + q;
      if (out_q < w_size) {
        size_t pixel = size_t(out_p) * size_t(w_size) + size_t(out_q);
        uint16_t bits = patchshift::element_bits_from_float(
            __uint_as_float(values[q]));
        *reinterpret_cast<uint16_t*>(
            output + pixel * size_t(output_pitch_k) + k_base + lane) = bits;
      }
    }
  }
  patchshift::tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (wid == kM32MicroWarps - 1) {
    patchshift::tcgen05_fence_after_thread_sync();
    patchshift::tcgen05_dealloc(
        shared.tmem_base, kM32MicroTmemColumns);
  }
#else
  (void)input_c32_map;
  (void)weight_map;
  (void)output;
  (void)h_size;
  (void)w_size;
  (void)output_pitch_k;
#endif
}
