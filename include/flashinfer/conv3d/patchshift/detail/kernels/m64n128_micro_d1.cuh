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

// D1/C32/K64 micro path with an exact M64N128 MMA tile.
// P4xQ30 combines both K32 output tiles and both adjacent P2 spatial tiles.
// The Q4 edge is remapped to P16xQ8, reducing the 64x64 target from 48 to
// 36 CTAs. Every CTA executes an independent legal cta_group::1 M64N128 MMA
// and loads its own small weight tile; avoiding cluster setup is faster for
// this launch-sensitive shape.

constexpr int kM64N128MicroM = 64;
constexpr int kM64N128MicroN = 128;
constexpr int kM64N128MicroPitch = 32;
constexpr int kM64N128MicroOutP = kM64N128MicroN / kM64N128MicroPitch;
constexpr int kM64N128MicroOutQ = 30;
constexpr int kM64N128MicroInputP = kM64N128MicroOutP + 2;
constexpr int kM64N128MicroSemanticRows =
    kM64N128MicroInputP * kM64N128MicroPitch;
constexpr int kM64N128MicroRequiredRows =
    kM64N128MicroN + 2 * kM64N128MicroPitch + 2;
constexpr int kM64N128MicroBackingRows =
    ((kM64N128MicroRequiredRows + 7) / 8) * 8;
constexpr int kM64N128MicroTmemColumns = kM64N128MicroN / 2;
constexpr int kM64N128MicroWarps = 4;
constexpr int kM64N128MicroThreads = kM64N128MicroWarps * 32;
constexpr int kM64N128MicroFullQTiles = 2;
constexpr int kM64N128MicroFullPTiles = 16;
constexpr int kM64N128MicroFullTasks =
    kM64N128MicroFullQTiles * kM64N128MicroFullPTiles;
constexpr int kM64N128MicroCompactPitch = 8;
constexpr int kM64N128MicroCompactOutP =
    kM64N128MicroN / kM64N128MicroCompactPitch;
constexpr int kM64N128MicroCompactInputP =
    kM64N128MicroCompactOutP + 2;
constexpr int kM64N128MicroCompactSemanticRows =
    kM64N128MicroCompactInputP * kM64N128MicroCompactPitch;
constexpr int kM64N128MicroCompactTasks = 4;

static_assert(kM64N128MicroOutP == 4 &&
              kM64N128MicroSemanticRows == 192 &&
              kM64N128MicroRequiredRows == 194 &&
              kM64N128MicroBackingRows == 200 &&
              kM64N128MicroTmemColumns == 64 &&
              kM64N128MicroCompactOutP == 16 &&
              kM64N128MicroCompactSemanticRows == 144);

struct alignas(512) M64N128MicroBStage {
  alignas(512) Element b[kM64N128MicroBackingRows * 32];
  uint64_t ready;
};

struct alignas(128) M64N128MicroARow {
  alignas(128) Element
      a[3][kK16GroupsPerStage][kM64N128MicroM * kK];
  uint64_t ready;
};

struct alignas(512) M64N128MicroSharedStorage {
  M64N128MicroBStage b_stage;
  M64N128MicroARow a_row[3];
  uint64_t done;
  uint32_t tmem_base;
};

static_assert(sizeof(M64N128MicroSharedStorage) <= 65536,
              "M64N128 micro path must remain multi-resident");

__host__ __device__ constexpr uint64_t m64n128_micro_idesc() {
  constexpr uint32_t c_format_f32 = 1u;
  constexpr uint32_t ab_format_bf16 = 1u;
  constexpr uint32_t n_dim = kM64N128MicroN >> 3;
  constexpr uint32_t m_dim = kM64N128MicroM >> 4;
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

__device__ __forceinline__ void issue_m64n128_micro_row(
    M64N128MicroBStage& b_stage,
    M64N128MicroARow& a_row,
    int filter_row,
    int pitch,
    uint32_t tmem_base,
    bool first_row) {
  if (!patchshift::elect_one_sync()) {
    return;
  }
  constexpr uint64_t idesc = m64n128_micro_idesc();
#pragma unroll
  for (int kg = 0; kg < kK16GroupsPerStage; ++kg) {
    Element* b = b_stage.b +
                 swizzled_b_c32_index(filter_row * pitch, kg * kK);
    uint64_t desc_b = pack_b_c32_desc(b);
#pragma unroll
    for (int kw = 0; kw < 3; ++kw) {
      uint64_t desc_a =
          patchshift::pack_k16_desc(a_row.a[kw][kg], kM64N128MicroM);
      bool first = first_row && kg == 0 && kw == 0;
      mma_ws_raw(desc_a, desc_b, tmem_base,
                 first ? 0u : 1u, idesc,
                 patchshift::shift_desc(kw));
    }
  }
}

__global__ void general_m64n128_d1_c32_micro_kernel(
    TensorMap const* input_c32_map,
    TensorMap const* input_compact_q4_map,
    TensorMap const* weight_map,
    Element* output,
    int h_size,
    int w_size) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  __shared__ M64N128MicroSharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  int spatial_task = int(blockIdx.x);
  bool compact_q4 = spatial_task >= kM64N128MicroFullTasks;
  int compact_task = spatial_task - kM64N128MicroFullTasks;
  int q_tile = spatial_task / kM64N128MicroFullPTiles;
  int p_tile = spatial_task - q_tile * kM64N128MicroFullPTiles;
  int q_base = compact_q4 ? 60 : q_tile * kM64N128MicroOutQ;
  int p_base = compact_q4 ? compact_task * kM64N128MicroCompactOutP
                          : p_tile * kM64N128MicroOutP;
  int pitch = compact_q4 ? kM64N128MicroCompactPitch
                         : kM64N128MicroPitch;
  int semantic_rows = compact_q4 ? kM64N128MicroCompactSemanticRows
                                 : kM64N128MicroSemanticRows;

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
        &shared.tmem_base, kM64N128MicroTmemColumns);
    __syncwarp();
    patchshift::tcgen05_relinquish_alloc_permit();
  }
  __syncthreads();

  // The bshift address sequence may legally reach the two rows immediately
  // following the semantic TMA box.  They are padding, not input data, so
  // initialize the complete rounded backing interval before issuing MMA.
  constexpr int guard_rows =
      kM64N128MicroRequiredRows - kM64N128MicroSemanticRows;
  for (int idx = int(threadIdx.x); idx < guard_rows * 32;
       idx += int(blockDim.x)) {
    int row = semantic_rows + idx / 32;
    int channel = idx - (idx / 32) * 32;
    shared.b_stage.b[swizzled_b_c32_index(row, channel)] =
        patchshift::element_from_float(0.0f);
  }
  __syncthreads();

  if (wid == 0 && lane == 0) {
    patchshift::tma_descriptor_fence_acquire(compact_q4 ? input_compact_q4_map
                                                        : input_c32_map);
    constexpr uint32_t b_bytes =
        kM64N128MicroSemanticRows * 32 * sizeof(Element);
    uint32_t transfer_bytes =
        compact_q4
            ? kM64N128MicroCompactSemanticRows * 32 * sizeof(Element)
            : b_bytes;
    patchshift::mbarrier_arrive_expect_tx(
        &shared.b_stage.ready, transfer_bytes);
    patchshift::tma_load_5d(
        compact_q4 ? input_compact_q4_map : input_c32_map,
        &shared.b_stage.ready,
        shared.b_stage.b + swizzled_b_c32_index(0, 0),
        0, q_base - 1, p_base - 1, 0, 0);
  }

  if (wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kM64N128MicroM * kK * sizeof(Element);
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      patchshift::mbarrier_arrive_expect_tx(
          &shared.a_row[row].ready, a_row_bytes);
      int weight_task = 3 + row;
      patchshift::tma_load_5d(
          weight_map, &shared.a_row[row].ready,
          shared.a_row[row].a[0][0],
          0, 0, 0, 0, weight_task);
    }
  }

  if (wid == 2) {
    while (!patchshift::mbarrier_try_wait(&shared.b_stage.ready, 0)) {
    }
    patchshift::fence_view_async_shared();
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      while (!patchshift::mbarrier_try_wait(&shared.a_row[row].ready, 0)) {
      }
      patchshift::fence_view_async_shared();
      issue_m64n128_micro_row(
          shared.b_stage, shared.a_row[row], row, pitch,
          shared.tmem_base, row == 0);
    }
    patchshift::tcgen05_commit(&shared.done);
  }

  while (!patchshift::mbarrier_try_wait(&shared.done, 0)) {
  }
  patchshift::tcgen05_fence_after_thread_sync();

  int store_warp = wid & 1;
  int store_group = (wid >> 1) & 1;
  int local_k = store_warp * 32 + lane;
  if (!compact_q4) {
#pragma unroll
    for (int load_phase = 0; load_phase < 2; ++load_phase) {
      uint32_t values[32];
      patchshift::tcgen05_load_32dp32b_x32(
          shared.tmem_base + load_phase * 32, values);
      patchshift::tcgen05_wait_tmem_load();
      int out_p = p_base + store_group * 2 + load_phase;
      size_t pixel = size_t(out_p) * size_t(w_size) + size_t(q_base);
      Element* out = output + pixel * 64 + local_k;
#pragma unroll
      for (int q = 0; q < kM64N128MicroOutQ; ++q) {
        uint16_t bits = patchshift::element_bits_from_float(
            __uint_as_float(values[q]));
        *reinterpret_cast<uint16_t*>(out + size_t(q) * 64) = bits;
      }
    }
  } else {
#pragma unroll
    for (int load_phase = 0; load_phase < 2; ++load_phase) {
      uint32_t values[32];
      patchshift::tcgen05_load_32dp32b_x32(
          shared.tmem_base + load_phase * 32, values);
      patchshift::tcgen05_wait_tmem_load();
      int p_chunk_base = store_group * 8 + load_phase * 4;
#pragma unroll
      for (int local_p = 0; local_p < 4; ++local_p) {
        int out_p = p_base + p_chunk_base + local_p;
        Element* out =
            output +
            (size_t(out_p) * size_t(w_size) + size_t(q_base)) * 64 +
            local_k;
#pragma unroll
        for (int local_q = 0; local_q < 4; ++local_q) {
          uint16_t bits = patchshift::element_bits_from_float(
              __uint_as_float(values[local_p * 8 + local_q]));
          *reinterpret_cast<uint16_t*>(
              out + size_t(local_q) * 64) = bits;
        }
      }
    }
  }
  patchshift::tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (wid == kM64N128MicroWarps - 1) {
    patchshift::tcgen05_fence_after_thread_sync();
    patchshift::tcgen05_dealloc(
        shared.tmem_base, kM64N128MicroTmemColumns);
  }
#else
  (void)input_c32_map;
  (void)input_compact_q4_map;
  (void)weight_map;
  (void)output;
  (void)h_size;
  (void)w_size;
#endif
}
