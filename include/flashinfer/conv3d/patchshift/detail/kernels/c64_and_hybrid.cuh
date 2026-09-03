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

// C64/K64 macro pipeline and exact C96 C64+C32 hybrid.
// Included by the PatchShift kernel umbrella inside its detail namespace.

// M128 x (2 x N256), C64 B2 with K32-granular A3 row streaming
// ----------------------------------------------------------------
// A literal B2/A3 K64 macro cannot fit SM100 shared memory:
//   2 * 75,776 B C64 activation stages
// + 3 * 49,408 B full-K64 packed A rows
// = 299,776 B before metadata, 67,328 B over the 232,448 B CTA limit.
//
// This equivalent variant preserves the useful properties independently:
// two complete C64 B stages reduce activation publication to one TMA per K64
// macro and overlap macro m+1 with MMA on m; three 24,832 B K32 A-row stages
// retain row depth instead of collapsing to A1.  Each resident C64 B tile is
// consumed as two consecutive K32 halves.  The arithmetic remains ordinary
// M128N256K16, and each half executes two K16 groups from the same C64 tile.
constexpr int kK64C64B2A3K32ABRing = 2;
constexpr int kK32HalvesPerK64Macro = 2;
constexpr int kK64C64Warps = 4;
constexpr int kK64C64Threads = kK64C64Warps * 32;

struct alignas(1024) K64C64B2A3K32ABStage {
  alignas(1024) Element b[kMainBackingRows * 64];
  uint64_t ready;
  uint64_t half_row_done[kK32HalvesPerK64Macro][3];
};

struct K64C64B2A3K32ASharedStorage {
  K64C64B2A3K32ABStage b_stage[kK64C64B2A3K32ABRing];
  DeepIlpARowStage a_stage[3];
  uint32_t tmem_base;
  volatile int b_published;
  volatile int a_published;
  volatile int commit_issued[kK64C64B2A3K32ABRing]
                            [kK32HalvesPerK64Macro][3];
  volatile int a_release_observed;
  volatile int tmem_ready;
};

static_assert(sizeof(K64C64B2A3K32ABStage) == 75776);
static_assert(sizeof(DeepIlpARowStage) == 24832);
static_assert(sizeof(K64C64B2A3K32ASharedStorage) == 226304);
static_assert(sizeof(K64C64B2A3K32ASharedStorage) <= 232448,
              "C64 B2 plus K32 A3 must fit one SM100 CTA");
static_assert(alignof(K64C64B2A3K32ASharedStorage) == 1024);
static_assert(offsetof(K64C64B2A3K32ASharedStorage, a_stage) % 256 == 0);
__device__ __forceinline__ void issue_k64_c64_b2a3_k32a_row(
    K64C64B2A3K32ABStage& b_stage,
    DeepIlpARowStage& a_stage,
    int k32_half,
    int filter_row,
    uint32_t tmem_base,
    int pitch,
    bool first_row) {
  if (!patchshift::elect_one_sync()) {
    return;
  }
  constexpr uint64_t idesc = main_mma_idesc();
  constexpr uint32_t dst0_offset = 0;
  constexpr uint32_t dst1_offset = kMainAccumulatorColumns;
#pragma unroll
  for (int kg = 0; kg < kK16GroupsPerStage; ++kg) {
    int k_offset = k32_half * 32 + kg * kK;
    Element* b0 =
        b_stage.b +
        swizzled_b_c64_index(filter_row * pitch, k_offset);
    Element* b1 =
        b_stage.b +
        swizzled_b_c64_index(
            kMainN + filter_row * pitch, k_offset);
    uint64_t desc_b0 = pack_b_c64_desc(b0);
    uint64_t desc_b1 = pack_b_c64_desc(b1);
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

// Compile full and partial epilogues independently so full tiles retain the
// GPU-validated 112-register footprint.  Partial is selected only when the
// existing 10% spatial-tail threshold is crossed.
template <bool OptimizedPartial, bool ExactK128 = false,
          int ExactKout = 0>
__global__ void general_m128n256_k64_c64_b2a3_k32a_kernel(
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
  __shared__ K64C64B2A3K32ASharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  int q_base = int(blockIdx.x) * kOutQ;
  int p_base = int(blockIdx.y) * kMainOutP;
  int flat_batch_count = n_size * d_size;
  int k_tile = int(blockIdx.z) / flat_batch_count;
  int flat_batch = int(blockIdx.z) - k_tile * flat_batch_count;
  int n = flat_batch / d_size;
  int od = flat_batch - n * d_size;
  int k_base = k_tile * kMainM;

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
              &shared.b_stage[slot].half_row_done[half][row], 1);
        }
      }
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      patchshift::mbarrier_init(&shared.a_stage[row].ready, 1);
    }
  }
  __syncthreads();

  // Warp 0: one canonical P18xQ32xC64 transaction per macro.  The second B
  // slot overlaps the next macro without increasing the activation command
  // count back to the two C32 publications used by the retained path.
  if (wid == 0 && lane == 0) {
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

  // Warp 1: three independent K32 A rows.  Each row slot can begin loading
  // the next half as soon as the corresponding prior-row commit completes;
  // A is never collapsed to a single 48 KiB staging slot.
  if (wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kMainM * kK * sizeof(Element);
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
            (k_tile * full_k32_supergroups + full_sg) * 3 + row;
        patchshift::tma_load_5d(
            weight_k32_map, &shared.a_stage[row].ready,
            shared.a_stage[row].a[0][0], 0, 0, 0, 0,
            weight_task);
        __threadfence_block();
        shared.a_published = half_task * 3 + row + 1;
      }
    }
  }

  // Warp 2: two ordinary K32 halves consume the same resident C64 tile.
  // Every instruction is still a legal M128N256K16 MMA; no unsupported K64
  // MMA shape is introduced.
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
          issue_k64_c64_b2a3_k32a_row(
              shared.b_stage[slot], shared.a_stage[row], half, row,
              shared.tmem_base, kPitch,
              macro == 0 && half == 0 && row == 0);
          patchshift::tcgen05_commit(
              &shared.b_stage[slot].half_row_done[half][row]);
          if (lane == 0) {
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

  if constexpr (ExactKout > 0) {
    static_assert(ExactKout <= kMainM);
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
        if (local_k < ExactKout) {
          size_t pixel =
              ((size_t(flat_batch) * size_t(h_size) + size_t(out_p)) *
                   size_t(w_size) +
               size_t(q_base));
          Element* out =
              output + pixel * size_t(ExactKout) + size_t(local_k);
#pragma unroll
          for (int q = 0; q < kOutQ; ++q) {
            uint16_t bits = patchshift::element_bits_from_float(
                __uint_as_float(values[q]));
            *reinterpret_cast<uint16_t*>(
                out + size_t(q) * size_t(ExactKout)) = bits;
          }
        }
      }
    }
  } else if constexpr (ExactK128) {
    int local_k = wid * 32 + lane;
    int valid_q = min(kOutQ, max(0, w_size - q_base));
    bool full_tile = p_base + kMainOutP <= h_size &&
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
        int out_p = p_base + workset * kMainOutPPerWorkset +
                    (physical_col >> 5);
        if (full_tile) {
          size_t pixel =
              ((size_t(flat_batch) * size_t(h_size) + size_t(out_p)) *
                   size_t(w_size) +
               size_t(q_base));
          Element* out =
              output + pixel * size_t(kMainM) + size_t(local_k);
#pragma unroll
          for (int q = 0; q < kOutQ; ++q) {
            uint16_t bits = patchshift::element_bits_from_float(
                __uint_as_float(values[q]));
            *reinterpret_cast<uint16_t*>(
                out + size_t(q) * size_t(kMainM)) = bits;
          }
        } else if (out_p < h_size) {
          size_t pixel =
              ((size_t(flat_batch) * size_t(h_size) + size_t(out_p)) *
                   size_t(w_size) +
               size_t(q_base));
          Element* out =
              output + pixel * size_t(kMainM) + size_t(local_k);
#pragma unroll
          for (int q = 0; q < kOutQ; ++q) {
            if (q < valid_q) {
              uint16_t bits = patchshift::element_bits_from_float(
                  __uint_as_float(values[q]));
              *reinterpret_cast<uint16_t*>(
                  out + size_t(q) * size_t(kMainM)) = bits;
            }
          }
        }
      }
    }
  } else if constexpr (!OptimizedPartial) {
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
  if (wid == kK64C64Warps - 1) {
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


// Automatically selected M128/P16 hybrid C64+C32 activation mainloop
// -------------------------------------------------------------------
// The dispatch below admits only the measured C96/Kout128 full-M128 policy.
// C96 is represented exactly as one C64 macro followed by one C32 macro per
// temporal position.  The existing packed K32 weight tasks are retained
// byte-for-byte: C64 consumes two consecutive K32 tasks and C32 consumes one.
// A raw B2 backing supports both swizzles while three A rows carry their own
// completion barriers.  No channel padding and no padding MMA is introduced.
constexpr int kHybridC64C32BStages = 2;
constexpr int kHybridC64C32Warps = 4;
constexpr int kHybridC64C32Threads = kHybridC64C32Warps * 32;
constexpr int kHybridC32RawOffset = kMainSemanticRows * 32;

static_assert((kHybridC32RawOffset * int(sizeof(Element))) % 512 == 0);
static_assert(kHybridC32RawOffset + kMainSemanticRows * 32 ==
              kMainSemanticRows * 64);
static_assert(kHybridC32RawOffset + kMainBackingRows * 32 <=
              kMainBackingRows * 64);

struct alignas(1024) HybridC64C32BStage {
  alignas(1024) Element raw[kMainBackingRows * 64];
  uint64_t ready;
};

struct alignas(256) HybridC64C32ARowStage {
  alignas(128) Element a[3][kK16GroupsPerStage][kMainM * kK];
  uint64_t ready;
  uint64_t done;
};

struct HybridC64C32SharedStorage {
  HybridC64C32BStage b_stage[kHybridC64C32BStages];
  HybridC64C32ARowStage a_stage[3];
  uint32_t tmem_base;
  volatile int b_published;
  volatile int a_published;
  volatile int commit_issued[3];
  // Number of consecutive K32 half tasks whose three A rows have all been
  // observed complete by the A producer. B2 reuse waits this monotonic value
  // instead of trying to assign a fixed barrier phase to a 2/1-half macro.
  volatile int a_release_observed;
  volatile int tmem_ready;
};

union HybridCompactC64C32SharedStorage {
  HybridC64C32SharedStorage hybrid;
  CompactSpatialSharedStorage compact;
};

static_assert(sizeof(HybridC64C32BStage) == 75776);
static_assert(sizeof(HybridC64C32ARowStage) == 24832);
static_assert(sizeof(HybridC64C32SharedStorage) == 226304);
static_assert(sizeof(HybridC64C32SharedStorage) <= 232448,
              "hybrid C64+C32 B2/A3 must fit one SM100 CTA");
static_assert(sizeof(HybridCompactC64C32SharedStorage) ==
                  sizeof(HybridC64C32SharedStorage));
static_assert(alignof(HybridC64C32SharedStorage) == 1024);
static_assert(offsetof(HybridC64C32SharedStorage, a_stage) % 256 == 0);


__device__ __forceinline__ void issue_hybrid_c64_c32_row(
    HybridC64C32BStage& b_stage,
    HybridC64C32ARowStage& a_stage,
    bool is_c64_macro,
    int k32_half,
    int filter_row,
    uint32_t tmem_base,
    bool first_row) {
  if (!patchshift::elect_one_sync()) {
    return;
  }
  constexpr uint64_t idesc = main_mma_idesc();
  constexpr uint32_t dst0_offset = 0;
  constexpr uint32_t dst1_offset = kMainAccumulatorColumns;
#pragma unroll
  for (int kg = 0; kg < kK16GroupsPerStage; ++kg) {
    int k_offset = (is_c64_macro ? k32_half * 32 : 0) + kg * kK;
    int row0 = filter_row * kPitch;
    int row1 = kMainN + filter_row * kPitch;
    Element* b0 =
        b_stage.raw +
        (is_c64_macro ? swizzled_b_c64_index(row0, k_offset)
                      : kHybridC32RawOffset +
                            swizzled_b_c32_index(row0, k_offset));
    Element* b1 = b_stage.raw +
        (is_c64_macro ? swizzled_b_c64_index(row1, k_offset)
                      : kHybridC32RawOffset +
                            swizzled_b_c32_index(row1, k_offset));
    uint64_t desc_b0 =
        is_c64_macro ? pack_b_c64_desc(b0) : pack_b_c32_desc(b0);
    uint64_t desc_b1 =
        is_c64_macro ? pack_b_c64_desc(b1) : pack_b_c32_desc(b1);
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

__device__ __forceinline__ void issue_hybrid_ptail1_row(
    HybridC64C32BStage& b_stage,
    HybridC64C32ARowStage& a_stage,
    bool is_c64_macro,
    int k32_half,
    int filter_row,
    uint32_t tmem_base,
    bool first_row) {
  if (!patchshift::elect_one_sync()) {
    return;
  }
  constexpr uint64_t idesc = compact_mma_idesc();
#pragma unroll
  for (int kg = 0; kg < kK16GroupsPerStage; ++kg) {
    int k_offset = (is_c64_macro ? k32_half * 32 : 0) + kg * kK;
    int row = filter_row * kCompactPTail1Pitch;
    Element* b = b_stage.raw +
        (is_c64_macro ? swizzled_b_c64_index(row, k_offset)
                      : kHybridC32RawOffset +
                            swizzled_b_c32_index(row, k_offset));
    uint64_t desc_b =
        is_c64_macro ? pack_b_c64_desc(b) : pack_b_c32_desc(b);
#pragma unroll
    for (int kw = 0; kw < 3; ++kw) {
      uint64_t desc_a =
          patchshift::pack_k16_desc(a_stage.a[kw][kg], kMainM);
      bool first = first_row && kg == 0 && kw == 0;
      mma_ws_raw(desc_a, desc_b, tmem_base,
                 first ? 0u : 1u, idesc,
                 patchshift::shift_desc(kw));
    }
  }
}

template <bool ExactH17W840 = false>
__device__ __forceinline__ void run_hybrid_ptail1_cta(
    HybridC64C32SharedStorage& shared,
    TensorMap const* input_c64_map,
    TensorMap const* input_c32_map,
    TensorMap const* weight_k32_map,
    Element* output,
    int n_size,
    int d_size,
    int h_size,
    int w_size,
    int c64_groups_per_time,
    int c32_groups_per_time,
    int k_size,
    int p_base,
    int q_base,
    int valid_q) {
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  int effective_d = ExactH17W840 ? 4 : d_size;
  int effective_h = ExactH17W840 ? 17 : h_size;
  int effective_w = ExactH17W840 ? 840 : w_size;
  int effective_k = ExactH17W840 ? 128 : k_size;
  int effective_c64_groups =
      ExactH17W840 ? 1 : c64_groups_per_time;
  int effective_c32_groups =
      ExactH17W840 ? 3 : c32_groups_per_time;
  int flat_batch_count = ExactH17W840 ? 4 : n_size * effective_d;
  int k_tile = ExactH17W840 ? 0 : int(blockIdx.z) / flat_batch_count;
  int flat_batch = ExactH17W840
                       ? int(blockIdx.z)
                       : int(blockIdx.z) - k_tile * flat_batch_count;
  int n = ExactH17W840 ? 0 : flat_batch / effective_d;
  int od = ExactH17W840 ? flat_batch
                        : flat_batch - n * effective_d;
  int td_begin = od == 0 ? 1 : 0;
  int td_end = od == effective_d - 1 ? 2 : kT;
  int local_td_count = td_end - td_begin;
  int macros_per_time = effective_c64_groups + 1;
  int local_macros = local_td_count * macros_per_time;
  int local_half_tasks = local_td_count * effective_c32_groups;
  int full_k32_supergroups = kT * effective_c32_groups;

  constexpr int guard_rows =
      kCompactPTail1BackingRows - kCompactPTail1SemanticRows;
  constexpr int c64_guard = guard_rows * 64;
  constexpr int c32_guard = guard_rows * 32;
  for (int idx = int(threadIdx.x);
       idx < kHybridC64C32BStages * (c64_guard + c32_guard);
       idx += int(blockDim.x)) {
    int slot = idx / (c64_guard + c32_guard);
    int rest = idx - slot * (c64_guard + c32_guard);
    if (rest < c64_guard) {
      int row = kCompactPTail1SemanticRows + rest / 64;
      int kk = rest % 64;
      shared.b_stage[slot].raw[swizzled_b_c64_index(row, kk)] =
          patchshift::element_from_float(0.0f);
    } else {
      rest -= c64_guard;
      int row = kCompactPTail1SemanticRows + rest / 32;
      int kk = rest % 32;
      shared.b_stage[slot].raw[
          kHybridC32RawOffset + swizzled_b_c32_index(row, kk)] =
          patchshift::element_from_float(0.0f);
    }
  }
  if (threadIdx.x == 0) {
    shared.tmem_base = 0;
    shared.b_published = 0;
    shared.a_published = 0;
    shared.a_release_observed = 0;
    shared.tmem_ready = 0;
#pragma unroll
    for (int slot = 0; slot < kHybridC64C32BStages; ++slot) {
      patchshift::mbarrier_init(&shared.b_stage[slot].ready, 1);
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      shared.commit_issued[row] = 0;
      patchshift::mbarrier_init(&shared.a_stage[row].ready, 1);
      patchshift::mbarrier_init(&shared.a_stage[row].done, 1);
    }
  }
  __syncthreads();

  if (wid == 0 && lane == 0) {
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro % kHybridC64C32BStages;
      int seq = macro / kHybridC64C32BStages;
      if (seq > 0) {
        int old_macro = macro - kHybridC64C32BStages;
        int old_local_td = old_macro / macros_per_time;
        int old_macro_in_td = old_macro - old_local_td * macros_per_time;
        int old_valid_halves =
            old_macro_in_td == effective_c64_groups ? 1 : 2;
        int old_final_half_task =
            old_local_td * effective_c32_groups +
            old_macro_in_td * 2 + old_valid_halves - 1;
        while (shared.a_release_observed < old_final_half_task + 1) {
        }
      }
      int local_td = macro / macros_per_time;
      int macro_in_td = macro - local_td * macros_per_time;
      bool is_c64_macro = macro_in_td < effective_c64_groups;
      int channel_base = macro_in_td * 64;
      int td = td_begin + local_td;
      uint32_t b_bytes = uint32_t(
          kCompactPTail1SemanticRows * (is_c64_macro ? 64 : 32) *
          sizeof(Element));
      patchshift::mbarrier_arrive_expect_tx(
          &shared.b_stage[slot].ready, b_bytes);
      if (is_c64_macro) {
        patchshift::tma_load_5d(
            input_c64_map, &shared.b_stage[slot].ready,
            shared.b_stage[slot].raw + swizzled_b_c64_index(0, 0),
            channel_base, q_base - 1, p_base - 1, od + td - 1, n);
      } else {
        patchshift::tma_load_5d(
            input_c32_map, &shared.b_stage[slot].ready,
            shared.b_stage[slot].raw + kHybridC32RawOffset +
                swizzled_b_c32_index(0, 0),
            channel_base, q_base - 1, p_base - 1, od + td - 1, n);
      }
      __threadfence_block();
      shared.b_published = macro + 1;
    }
  }

  if (wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kMainM * kK * sizeof(Element);
    for (int half_task = 0; half_task < local_half_tasks; ++half_task) {
      int local_td = half_task / effective_c32_groups;
      int sg_in_td = half_task - local_td * effective_c32_groups;
      int td = td_begin + local_td;
      int full_sg = td * effective_c32_groups + sg_in_td;
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        if (half_task > 0) {
          int previous_task = half_task - 1;
          while (shared.commit_issued[row] < previous_task + 1) {
          }
          __threadfence_block();
          while (!patchshift::mbarrier_try_wait(
              &shared.a_stage[row].done, previous_task & 1)) {
          }
          if (row == 2) {
            __threadfence_block();
            shared.a_release_observed = half_task;
          }
        }
        patchshift::mbarrier_arrive_expect_tx(
            &shared.a_stage[row].ready, a_row_bytes);
        int weight_task =
            (k_tile * full_k32_supergroups + full_sg) * 3 + row;
        patchshift::tma_load_5d(
            weight_k32_map, &shared.a_stage[row].ready,
            shared.a_stage[row].a[0][0], 0, 0, 0, 0, weight_task);
        __threadfence_block();
        shared.a_published = half_task * 3 + row + 1;
      }
    }
  }

  if (wid == 2) {
    patchshift::tcgen05_alloc(&shared.tmem_base, kCompactPTail1N);
    __syncwarp();
    patchshift::tcgen05_relinquish_alloc_permit();
    if (lane == 0) {
      __threadfence_block();
      shared.tmem_ready = 1;
    }
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro % kHybridC64C32BStages;
      int seq = macro / kHybridC64C32BStages;
      int local_td = macro / macros_per_time;
      int macro_in_td = macro - local_td * macros_per_time;
      bool is_c64_macro = macro_in_td < effective_c64_groups;
      int valid_halves = is_c64_macro ? 2 : 1;
      int half_task_base =
          local_td * effective_c32_groups + macro_in_td * 2;
      while (shared.b_published < macro + 1) {
      }
      while (!patchshift::mbarrier_try_wait(
          &shared.b_stage[slot].ready, seq & 1)) {
      }
      patchshift::fence_view_async_shared();
      for (int half = 0; half < valid_halves; ++half) {
        int half_task = half_task_base + half;
#pragma unroll
        for (int row = 0; row < 3; ++row) {
          while (shared.a_published < half_task * 3 + row + 1) {
          }
          while (!patchshift::mbarrier_try_wait(
              &shared.a_stage[row].ready, half_task & 1)) {
          }
          patchshift::fence_view_async_shared();
          issue_hybrid_ptail1_row(
              shared.b_stage[slot], shared.a_stage[row],
              is_c64_macro, half, row, shared.tmem_base,
              macro == 0 && half == 0 && row == 0);
          patchshift::tcgen05_commit(&shared.a_stage[row].done);
          if (lane == 0) {
            __threadfence_block();
            shared.commit_issued[row] = half_task + 1;
          }
        }
      }
    }
  }

  while (shared.tmem_ready == 0) {
  }
  int final_half_task = local_half_tasks - 1;
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    while (shared.commit_issued[row] < final_half_task + 1) {
    }
    __threadfence_block();
    while (!patchshift::mbarrier_try_wait(
        &shared.a_stage[row].done, final_half_task & 1)) {
    }
  }
  patchshift::tcgen05_fence_after_thread_sync();

  int local_k = wid * 32 + lane;
  for (int physical_col = 0; physical_col < kCompactPTail1N;
       physical_col += 32) {
    uint32_t values[32];
    patchshift::tcgen05_load_32dp32b_x32(
        shared.tmem_base + physical_col, values);
    patchshift::tcgen05_wait_tmem_load();
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      int local_q = physical_col + i;
      if (local_q < kCompactPTail1OutQ && local_q < valid_q) {
        int out_q = q_base + local_q;
        size_t pixel =
            ((size_t(flat_batch) * size_t(effective_h) + size_t(p_base)) *
                 size_t(effective_w) + size_t(out_q));
        uint16_t bits = patchshift::element_bits_from_float(
            __uint_as_float(values[i]));
        *reinterpret_cast<uint16_t*>(
            output + pixel * size_t(effective_k) + size_t(local_k)) = bits;
      }
    }
  }
  patchshift::tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (wid == kHybridC64C32Warps - 1) {
    patchshift::tcgen05_fence_after_thread_sync();
    patchshift::tcgen05_dealloc(shared.tmem_base, kCompactPTail1N);
  }
}

__global__ void general_hybrid_ptail1_exact_h17_w840_kernel(
    TensorMap const* input_c64_map,
    TensorMap const* input_c32_map,
    TensorMap const* weight_k32_map,
    Element* output) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  __shared__ HybridC64C32SharedStorage shared;
  int q_base = int(blockIdx.x) * kCompactPTail1OutQ;
  int valid_q = min(kCompactPTail1OutQ, 840 - q_base);
  run_hybrid_ptail1_cta<true>(
      shared, input_c64_map, input_c32_map, weight_k32_map, output,
      1, 4, 17, 840, 1, 3, 128, 16, q_base, valid_q);
#else
  (void)input_c64_map;
  (void)input_c32_map;
  (void)weight_k32_map;
  (void)output;
#endif
}

template <bool OptimizedPartial, bool ExactFull = false,
          bool ExactP15 = false, bool ExactH16W840 = false,
          bool ExactH17W840 = false, bool ExactW31 = false,
          bool EightWarpStore = false>
__device__ __forceinline__ void store_hybrid_m128_p16(
    uint32_t tmem_base,
    Element* output,
    int wid,
    int lane,
    int flat_batch,
    int p_base,
    int q_base,
    int k_base,
    int h_size,
    int w_size,
    int k_size) {
  static_assert(!(ExactH16W840 && ExactH17W840));
  static_assert(!ExactW31 || (ExactFull && !ExactP15 &&
                              !ExactH16W840 && !ExactH17W840));
  if constexpr (ExactW31) {
    int local_k = wid * 32 + lane;
#pragma unroll
    for (int workset = 0; workset < kMainWorksets; ++workset) {
      uint32_t tile_base =
          tmem_base + uint32_t(workset * kMainAccumulatorColumns);
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
            (size_t(flat_batch) * size_t(512) + size_t(out_p)) *
            size_t(31);
        Element* out = output + pixel * size_t(128) + size_t(local_k);
#pragma unroll
        for (int q = 0; q < 31; ++q) {
          uint16_t bits = patchshift::element_bits_from_float(
              __uint_as_float(values[q]));
          *reinterpret_cast<uint16_t*>(
              out + size_t(q) * size_t(128)) = bits;
        }
      }
    }
  } else if constexpr (ExactH16W840 || ExactH17W840) {
    static_assert(ExactFull && !ExactP15);
    constexpr int exact_h = ExactH16W840 ? 16 : 17;
    int local_k = wid * 32 + lane;
#pragma unroll
    for (int workset = 0; workset < kMainWorksets; ++workset) {
      uint32_t tile_base =
          tmem_base + uint32_t(workset * kMainAccumulatorColumns);
      for (int physical_col = 0; physical_col < kMainN;
           physical_col += 32) {
        uint32_t values[32];
        patchshift::tcgen05_load_32dp32b_x32(
            tile_base + physical_col, values);
        patchshift::tcgen05_wait_tmem_load();
        int out_p = workset * kMainOutPPerWorkset +
                    (physical_col >> 5);
        size_t pixel = size_t(flat_batch) * size_t(exact_h * 840) +
                       size_t(out_p * 840 + q_base);
        Element* out = output + pixel * size_t(128) + size_t(local_k);
#pragma unroll
        for (int q = 0; q < kOutQ; ++q) {
          uint16_t bits = patchshift::element_bits_from_float(
              __uint_as_float(values[q]));
          *reinterpret_cast<uint16_t*>(
              out + size_t(q * 128)) = bits;
        }
      }
    }
  } else if constexpr (ExactP15) {
    int store_partition = EightWarpStore ? (wid >> 2) : 0;
    int local_k = (EightWarpStore ? (wid & 3) : wid) * 32 + lane;
#pragma unroll
    for (int workset = 0; workset < kMainWorksets; ++workset) {
      uint32_t tile_base =
          tmem_base + uint32_t(workset * kMainAccumulatorColumns);
      int paired_columns = workset == 0 ? kMainN : kMainN - 64;
      for (int physical_col = store_partition * 64;
           physical_col < paired_columns;
           physical_col += EightWarpStore ? 128 : 64) {
        uint32_t values[64];
        patchshift::tcgen05_load_32dp32b_x64(
            tile_base + physical_col, values);
        patchshift::tcgen05_wait_tmem_load();
#pragma unroll
        for (int row = 0; row < 2; ++row) {
          int out_p = workset * kMainOutPPerWorkset +
                      (physical_col >> 5) + row;
          size_t pixel = size_t(flat_batch) * size_t(15 * 840) +
                         size_t(out_p * 840 + q_base);
          Element* out = output + pixel * size_t(128) + size_t(local_k);
#pragma unroll
          for (int q = 0; q < kOutQ; ++q) {
            uint16_t bits = patchshift::element_bits_from_float(
                __uint_as_float(values[row * 32 + q]));
            *reinterpret_cast<uint16_t*>(
                out + size_t(q * 128)) = bits;
          }
        }
      }
      if (workset == 1 && (!EightWarpStore || store_partition == 1)) {
        uint32_t values[32];
        patchshift::tcgen05_load_32dp32b_x32(
            tile_base + 192, values);
        patchshift::tcgen05_wait_tmem_load();
        size_t pixel = size_t(flat_batch) * size_t(15 * 840) +
                       size_t(14 * 840 + q_base);
        Element* out = output + pixel * size_t(128) + size_t(local_k);
#pragma unroll
        for (int q = 0; q < kOutQ; ++q) {
          uint16_t bits = patchshift::element_bits_from_float(
              __uint_as_float(values[q]));
          *reinterpret_cast<uint16_t*>(out + size_t(q * 128)) = bits;
        }
      }
    }
  } else if constexpr (ExactFull) {
    int local_k = wid * 32 + lane;
#pragma unroll
    for (int workset = 0; workset < kMainWorksets; ++workset) {
      uint32_t tile_base =
          tmem_base + uint32_t(workset * kMainAccumulatorColumns);
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
            ((size_t(flat_batch) * size_t(h_size) + size_t(out_p)) *
                 size_t(w_size) +
             size_t(q_base));
        Element* out =
            output + pixel * size_t(k_size) + size_t(k_base);
#pragma unroll
        for (int q = 0; q < kOutQ; ++q) {
          uint16_t bits = patchshift::element_bits_from_float(
              __uint_as_float(values[q]));
          *reinterpret_cast<uint16_t*>(
              out + size_t(q) * size_t(k_size) + size_t(local_k)) = bits;
        }
      }
    }
  } else if constexpr (!OptimizedPartial) {
    int local_k = wid * 32 + lane;
    int global_k = k_base + local_k;
    bool full_tile = k_base + kMainM <= k_size &&
                     p_base + kMainOutP <= h_size &&
                     q_base + kOutQ <= w_size;
#pragma unroll
    for (int workset = 0; workset < kMainWorksets; ++workset) {
      uint32_t tile_base =
          tmem_base + uint32_t(workset * kMainAccumulatorColumns);
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
    int local_k = wid * 32 + lane;
    int global_k = k_base + local_k;
    int valid_q = min(kOutQ, max(0, w_size - q_base));
    bool full_tile = k_base + kMainM <= k_size &&
                     p_base + kMainOutP <= h_size &&
                     valid_q == kOutQ;
#pragma unroll
    for (int workset = 0; workset < kMainWorksets; ++workset) {
      uint32_t tile_base =
          tmem_base + uint32_t(workset * kMainAccumulatorColumns);
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
          Element* out = output + pixel * size_t(k_size) + size_t(global_k);
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
  if (wid == kHybridC64C32Warps - 1) {
    patchshift::tcgen05_fence_after_thread_sync();
    patchshift::tcgen05_dealloc(tmem_base, kMainTmemColumns);
  }
}

// Exact H17/W840 main region.  This is deliberately a four-pointer kernel:
// N1/D4/H17/W840/C96/K128, P0 and the C64+C32 reduction schedule are compile
// time facts.  The separate P1 edge kernel owns P16 concurrently.
__global__ __launch_bounds__(kHybridC64C32Threads, 1)
void general_hybrid_main_exact_h17_w840_kernel(
    TensorMap const* input_c64_map,
    TensorMap const* input_c32_map,
    TensorMap const* weight_k32_map,
    Element* output) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  __shared__ HybridC64C32SharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  int q_base = int(blockIdx.x) * kOutQ;
  int flat_batch = int(blockIdx.z);
  int od = flat_batch;
  int td_begin = od == 0 ? 1 : 0;
  int local_td_count = (od == 0 || od == 3) ? 2 : 3;
  int local_macros = local_td_count * 2;
  int local_half_tasks = local_td_count * 3;

  constexpr int guard_rows = kMainBackingRows - kMainSemanticRows;
  constexpr int guard_per_stage = guard_rows * 64;
  for (int idx = int(threadIdx.x);
       idx < kHybridC64C32BStages * guard_per_stage;
       idx += int(blockDim.x)) {
    int slot = idx / guard_per_stage;
    int rest = idx - slot * guard_per_stage;
    int row = kMainSemanticRows + rest / 64;
    int kk = rest % 64;
    shared.b_stage[slot].raw[swizzled_b_c64_index(row, kk)] =
        patchshift::element_from_float(0.0f);
  }
  if (threadIdx.x == 0) {
    shared.tmem_base = 0;
    shared.b_published = 0;
    shared.a_published = 0;
    shared.a_release_observed = 0;
    shared.tmem_ready = 0;
#pragma unroll
    for (int slot = 0; slot < kHybridC64C32BStages; ++slot) {
      patchshift::mbarrier_init(&shared.b_stage[slot].ready, 1);
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      shared.commit_issued[row] = 0;
      patchshift::mbarrier_init(&shared.a_stage[row].ready, 1);
      patchshift::mbarrier_init(&shared.a_stage[row].done, 1);
    }
  }
  __syncthreads();

  if (wid == 0 && lane == 0) {
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro & 1;
      int seq = macro >> 1;
      if (seq > 0) {
        int old_macro = macro - 2;
        int old_local_td = old_macro >> 1;
        int old_macro_in_td = old_macro & 1;
        int old_final_half_task =
            old_local_td * 3 + old_macro_in_td * 2 +
            (old_macro_in_td == 1 ? 1 : 2) - 1;
        while (shared.a_release_observed < old_final_half_task + 1) {
        }
      }
      int local_td = macro >> 1;
      int macro_in_td = macro & 1;
      bool is_c64_macro = macro_in_td == 0;
      int td = td_begin + local_td;
      uint32_t b_bytes = uint32_t(
          kMainSemanticRows * (is_c64_macro ? 64 : 32) *
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
      __threadfence_block();
      shared.b_published = macro + 1;
    }
  }

  if (wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kMainM * kK * sizeof(Element);
    for (int half_task = 0; half_task < local_half_tasks; ++half_task) {
      int local_td = half_task / 3;
      int sg_in_td = half_task - local_td * 3;
      int full_sg = (td_begin + local_td) * 3 + sg_in_td;
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        if (half_task > 0) {
          int previous_task = half_task - 1;
          while (shared.commit_issued[row] < previous_task + 1) {
          }
          __threadfence_block();
          while (!patchshift::mbarrier_try_wait(
              &shared.a_stage[row].done, previous_task & 1)) {
          }
          if (row == 2) {
            __threadfence_block();
            shared.a_release_observed = half_task;
          }
        }
        patchshift::mbarrier_arrive_expect_tx(
            &shared.a_stage[row].ready, a_row_bytes);
        int weight_task = full_sg * 3 + row;
        patchshift::tma_load_5d(
            weight_k32_map, &shared.a_stage[row].ready,
            shared.a_stage[row].a[0][0], 0, 0, 0, 0, weight_task);
        __threadfence_block();
        shared.a_published = half_task * 3 + row + 1;
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
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro & 1;
      int seq = macro >> 1;
      int local_td = macro >> 1;
      int macro_in_td = macro & 1;
      bool is_c64_macro = macro_in_td == 0;
      int valid_halves = is_c64_macro ? 2 : 1;
      int half_task_base = local_td * 3 + macro_in_td * 2;
      while (shared.b_published < macro + 1) {
      }
      while (!patchshift::mbarrier_try_wait(
          &shared.b_stage[slot].ready, seq & 1)) {
      }
      patchshift::fence_view_async_shared();
      for (int half = 0; half < valid_halves; ++half) {
        int half_task = half_task_base + half;
#pragma unroll
        for (int row = 0; row < 3; ++row) {
          while (shared.a_published < half_task * 3 + row + 1) {
          }
          while (!patchshift::mbarrier_try_wait(
              &shared.a_stage[row].ready, half_task & 1)) {
          }
          patchshift::fence_view_async_shared();
          issue_hybrid_c64_c32_row(
              shared.b_stage[slot], shared.a_stage[row],
              is_c64_macro, half, row, shared.tmem_base,
              macro == 0 && half == 0 && row == 0);
          patchshift::tcgen05_commit(&shared.a_stage[row].done);
          if (lane == 0) {
            __threadfence_block();
            shared.commit_issued[row] = half_task + 1;
          }
        }
      }
    }
  }

  while (shared.tmem_ready == 0) {
  }
  int final_half_task = local_half_tasks - 1;
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    while (shared.commit_issued[row] < final_half_task + 1) {
    }
    __threadfence_block();
    while (!patchshift::mbarrier_try_wait(
        &shared.a_stage[row].done, final_half_task & 1)) {
    }
  }
  patchshift::tcgen05_fence_after_thread_sync();
  store_hybrid_m128_p16<false, true, false, false, true>(
      shared.tmem_base, output, wid, lane, flat_batch,
      0, q_base, 0, 17, 840, 128);
#else
  (void)input_c64_map;
  (void)input_c32_map;
  (void)weight_k32_map;
  (void)output;
#endif
}

template <bool OptimizedPartial, bool ExactFull = false,
          bool MixedCompactQ1 = false, bool ExactP15 = false,
          bool ExactH16W840 = false, bool MixedCompactP1 = false,
          bool ExactH17W840 = false, bool ExactW31 = false>
__global__ void general_m128n256_hybrid_c64_c32_b2a3_kernel(
    TensorMap const* input_c64_map,
    TensorMap const* input_c32_map,
    TensorMap const* weight_k32_map,
    Element* output,
    int n_size,
    int d_size,
    int h_size,
    int w_size,
    int c64_groups_per_time,
    int c32_groups_per_time,
    int k_size,
    TensorMap const* compact_q3_map = nullptr,
    int compact_full_q_tiles = 0,
    int compact_full_p_tiles = 0,
    int compact_q_tail = 0,
    TensorMap const* compact_c64_map = nullptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(!MixedCompactQ1 || ExactFull);
  static_assert(!MixedCompactP1 || ExactFull);
  static_assert(!(MixedCompactQ1 && MixedCompactP1));
  static_assert(!ExactP15 || (!ExactFull && !MixedCompactQ1));
  static_assert(!ExactH16W840 ||
                (ExactFull && !MixedCompactQ1 && !MixedCompactP1));
  static_assert(!ExactH17W840 ||
                (ExactFull && !MixedCompactQ1 && !MixedCompactP1));
  static_assert(!ExactW31 ||
                (ExactFull && !MixedCompactQ1 && !MixedCompactP1 &&
                 !ExactP15 && !ExactH16W840 && !ExactH17W840));
  __shared__ HybridCompactC64C32SharedStorage mixed_shared;
  HybridC64C32SharedStorage& shared = mixed_shared.hybrid;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  int q_base = 0;
  int p_base = 0;
  if constexpr (MixedCompactQ1 || MixedCompactP1) {
    int full_tasks = MixedCompactP1
                         ? 28
                         : compact_full_q_tiles * compact_full_p_tiles;
    int task = int(blockIdx.x);
    if (task < full_tasks) {
      if constexpr (MixedCompactP1) {
        q_base = task * kOutQ;
        p_base = 0;
      } else {
        int q_tile = task / compact_full_p_tiles;
        int p_tile = task - q_tile * compact_full_p_tiles;
        q_base = q_tile * kOutQ;
        p_base = p_tile * kMainOutP;
      }
    } else {
      int edge_task = task - full_tasks;
      if constexpr (MixedCompactQ1) {
        p_base = edge_task * kCompactQ1OutP;
        q_base = compact_full_q_tiles * kOutQ;
        int valid_p = min(kCompactQ1OutP, h_size - p_base);
        run_compact_spatial_cta(
            mixed_shared.compact, compact_q3_map, weight_k32_map,
            output, n_size, d_size, h_size, w_size,
            c32_groups_per_time, c32_groups_per_time * 2, k_size,
            p_base, q_base, kCompactQ1Pitch, valid_p, compact_q_tail);
      } else {
        p_base = 16;
        q_base = edge_task * kCompactPTail1OutQ;
        int valid_q = min(kCompactPTail1OutQ, 840 - q_base);
        run_hybrid_ptail1_cta<true>(
            mixed_shared.hybrid, compact_c64_map, compact_q3_map,
            weight_k32_map, output,
            n_size, d_size, h_size, w_size, c64_groups_per_time,
            c32_groups_per_time, k_size,
            p_base, q_base, valid_q);
      }
      return;
    }
  } else {
    q_base = int(blockIdx.x) * kOutQ;
    p_base = int(blockIdx.y) * kMainOutP;
  }
  constexpr bool ExactKnownC96 =
      ExactP15 || ExactH16W840 || MixedCompactP1 || ExactH17W840 ||
      ExactW31;
  int effective_d = ExactKnownC96 ? 4 : d_size;
  int effective_h = ExactP15
                        ? 15
                    : ExactH16W840
                        ? 16
                    : (MixedCompactP1 || ExactH17W840)
                        ? 17
                    : ExactW31
                        ? 512
                        : h_size;
  int effective_w = ExactW31 ? 31 : (ExactKnownC96 ? 840 : w_size);
  int effective_k = ExactKnownC96 ? 128 : k_size;
  int effective_c64_groups =
      ExactKnownC96 ? 1 : c64_groups_per_time;
  int effective_c32_groups =
      ExactKnownC96 ? 3 : c32_groups_per_time;
  int flat_batch_count = ExactKnownC96 ? 4 : n_size * effective_d;
  int k_tile = ExactKnownC96 ? 0 : int(blockIdx.z) / flat_batch_count;
  int flat_batch = ExactKnownC96
                       ? int(blockIdx.z)
                       : int(blockIdx.z) - k_tile * flat_batch_count;
  int n = ExactKnownC96 ? 0 : flat_batch / effective_d;
  int od = ExactKnownC96 ? flat_batch
                          : flat_batch - n * effective_d;
  int k_base = k_tile * kMainM;

  int td_begin = od == 0 ? 1 : 0;
  int td_end = od == effective_d - 1 ? 2 : kT;
  int local_td_count = td_end - td_begin;
  int macros_per_time = effective_c64_groups + 1;
  int local_macros = local_td_count * macros_per_time;
  int local_half_tasks = local_td_count * effective_c32_groups;
  int full_k32_supergroups = kT * effective_c32_groups;

  if (effective_c32_groups != effective_c64_groups * 2 + 1) {
    return;
  }

  constexpr int guard_rows = kMainBackingRows - kMainSemanticRows;
  constexpr int guard_per_stage = guard_rows * 64;
  for (int idx = int(threadIdx.x);
       idx < kHybridC64C32BStages * guard_per_stage;
       idx += int(blockDim.x)) {
    int slot = idx / guard_per_stage;
    int rest = idx - slot * guard_per_stage;
    int row = kMainSemanticRows + rest / 64;
    int kk = rest % 64;
    shared.b_stage[slot].raw[swizzled_b_c64_index(row, kk)] =
        patchshift::element_from_float(0.0f);
  }
  if (threadIdx.x == 0) {
    shared.tmem_base = 0;
    shared.b_published = 0;
    shared.a_published = 0;
    shared.a_release_observed = 0;
    shared.tmem_ready = 0;
#pragma unroll
    for (int slot = 0; slot < kHybridC64C32BStages; ++slot) {
      patchshift::mbarrier_init(&shared.b_stage[slot].ready, 1);
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      shared.commit_issued[row] = 0;
      patchshift::mbarrier_init(&shared.a_stage[row].ready, 1);
      patchshift::mbarrier_init(&shared.a_stage[row].done, 1);
    }
  }
  __syncthreads();

  // B producer: every temporal position publishes t C64 macros and one C32
  // tail. B2 reuse waits until the A producer has observed the old macro's
  // final K32 half, independent of whether that macro had two halves or one.
  if (wid == 0 && lane == 0) {
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro % kHybridC64C32BStages;
      int seq = macro / kHybridC64C32BStages;
      if (seq > 0) {
        int old_macro = macro - kHybridC64C32BStages;
        int old_local_td = old_macro / macros_per_time;
        int old_macro_in_td = old_macro - old_local_td * macros_per_time;
        int old_valid_halves =
            old_macro_in_td == effective_c64_groups ? 1 : 2;
        int old_final_half_task =
            old_local_td * effective_c32_groups +
            old_macro_in_td * 2 + old_valid_halves - 1;
        while (shared.a_release_observed < old_final_half_task + 1) {
        }
      }
      int local_td = macro / macros_per_time;
      int macro_in_td = macro - local_td * macros_per_time;
      bool is_c64_macro = macro_in_td < effective_c64_groups;
      int channel_base = macro_in_td * 64;
      int td = td_begin + local_td;
      uint32_t b_bytes =
          uint32_t(kMainSemanticRows * (is_c64_macro ? 64 : 32) *
                   sizeof(Element));
      patchshift::mbarrier_arrive_expect_tx(
          &shared.b_stage[slot].ready, b_bytes);
      if (is_c64_macro) {
        patchshift::tma_load_5d(
            input_c64_map, &shared.b_stage[slot].ready,
            shared.b_stage[slot].raw + swizzled_b_c64_index(0, 0),
            channel_base, q_base - 1, p_base - 1,
            od + td - 1, n);
      } else {
        patchshift::tma_load_5d(
            input_c32_map, &shared.b_stage[slot].ready,
            shared.b_stage[slot].raw + kHybridC32RawOffset +
                swizzled_b_c32_index(0, 0),
            channel_base, q_base - 1, p_base - 1,
            od + td - 1, n);
      }
      __threadfence_block();
      shared.b_published = macro + 1;
    }
  }

  // A producer: the packed K32 task stream stays dense. A row is overwritten
  // only after its immediately preceding half-task completion barrier flips.
  if (wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kMainM * kK * sizeof(Element);
    for (int half_task = 0; half_task < local_half_tasks; ++half_task) {
      int local_td = half_task / effective_c32_groups;
      int sg_in_td = half_task - local_td * effective_c32_groups;
      int td = td_begin + local_td;
      int full_sg = td * effective_c32_groups + sg_in_td;
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        if (half_task > 0) {
          int previous_task = half_task - 1;
          while (shared.commit_issued[row] < previous_task + 1) {
          }
          __threadfence_block();
          while (!patchshift::mbarrier_try_wait(
              &shared.a_stage[row].done, previous_task & 1)) {
          }
          if (row == 2) {
            __threadfence_block();
            shared.a_release_observed = half_task;
          }
        }
        patchshift::mbarrier_arrive_expect_tx(
            &shared.a_stage[row].ready, a_row_bytes);
        int weight_task =
            (k_tile * full_k32_supergroups + full_sg) * 3 + row;
        patchshift::tma_load_5d(
            weight_k32_map, &shared.a_stage[row].ready,
            shared.a_stage[row].a[0][0], 0, 0, 0, 0,
            weight_task);
        __threadfence_block();
        shared.a_published = half_task * 3 + row + 1;
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
    for (int macro = 0; macro < local_macros; ++macro) {
      int slot = macro % kHybridC64C32BStages;
      int seq = macro / kHybridC64C32BStages;
      int local_td = macro / macros_per_time;
      int macro_in_td = macro - local_td * macros_per_time;
      bool is_c64_macro = macro_in_td < effective_c64_groups;
      int valid_halves = is_c64_macro ? 2 : 1;
      int half_task_base =
          local_td * effective_c32_groups + macro_in_td * 2;
      while (shared.b_published < macro + 1) {
      }
      while (!patchshift::mbarrier_try_wait(
          &shared.b_stage[slot].ready, seq & 1)) {
      }
      patchshift::fence_view_async_shared();
      for (int half = 0; half < valid_halves; ++half) {
        int half_task = half_task_base + half;
#pragma unroll
        for (int row = 0; row < 3; ++row) {
          while (shared.a_published < half_task * 3 + row + 1) {
          }
          while (!patchshift::mbarrier_try_wait(
              &shared.a_stage[row].ready, half_task & 1)) {
          }
          patchshift::fence_view_async_shared();
          issue_hybrid_c64_c32_row(
              shared.b_stage[slot], shared.a_stage[row],
              is_c64_macro, half, row, shared.tmem_base,
              macro == 0 && half == 0 && row == 0);
          patchshift::tcgen05_commit(&shared.a_stage[row].done);
          if (lane == 0) {
            __threadfence_block();
            shared.commit_issued[row] = half_task + 1;
          }
        }
      }
    }
  }

  while (shared.tmem_ready == 0) {
  }
  int final_half_task = local_half_tasks - 1;
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    while (shared.commit_issued[row] < final_half_task + 1) {
    }
    __threadfence_block();
    while (!patchshift::mbarrier_try_wait(
        &shared.a_stage[row].done, final_half_task & 1)) {
    }
  }
  patchshift::tcgen05_fence_after_thread_sync();

  store_hybrid_m128_p16<
      OptimizedPartial, ExactFull, ExactP15, ExactH16W840,
      ExactH17W840 || MixedCompactP1, ExactW31>(
      shared.tmem_base, output, wid, lane, flat_batch, p_base, q_base,
      k_base, effective_h, effective_w, effective_k);

#else
  (void)input_c64_map;
  (void)input_c32_map;
  (void)weight_k32_map;
  (void)output;
  (void)n_size;
  (void)d_size;
  (void)h_size;
  (void)w_size;
  (void)c64_groups_per_time;
  (void)c32_groups_per_time;
  (void)k_size;
  (void)compact_q3_map;
  (void)compact_full_q_tiles;
  (void)compact_full_p_tiles;
  (void)compact_q_tail;
  (void)compact_c64_map;
#endif
}
