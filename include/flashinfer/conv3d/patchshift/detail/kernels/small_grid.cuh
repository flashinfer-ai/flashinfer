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

// Native M64/M32 P16 small-grid kernel family.
// Included by the PatchShift kernel umbrella inside its detail namespace.

// M64 P16 B2/A3 split pipeline
// -----------------------------
// Two compact P18 activation stages overlap the next C32 TMA transaction with
// the current MMA stream, while the three fixed A slots retain one packed
// filter row each.  Rows 0 and 1 share a prefix completion point; row 2 uses
// the final completion point.  The A-release counter prevents a fast B
// producer from allowing a reused B-slot barrier to enter its next generation
// before the A producer has observed the preceding prefix/final generation.
constexpr int kM64P16B2Stages = 2;

struct alignas(512) M64P16B2Stage {
  alignas(512) Element b[kM64P16BackingRows * 32];
  uint64_t ready;
  uint64_t prefix_done;
  uint64_t done;
};

struct M64P16B2A3SharedStorage {
  M64P16B2Stage b_stage[kM64P16B2Stages];
  M64P16ARowStage a_stage[3];
  uint32_t tmem_base;
  volatile int b_published;
  volatile int a_published;
  volatile int prefix_commit_issued[kM64P16B2Stages];
  volatile int final_commit_issued[kM64P16B2Stages];
  volatile int a_release_observed;
  volatile int tmem_ready;
};

static_assert(sizeof(M64P16B2Stage) == 37888);
static_assert(alignof(M64P16B2A3SharedStorage) == 512);
static_assert(offsetof(M64P16B2A3SharedStorage, a_stage) % 256 == 0);
static_assert(sizeof(M64P16B2A3SharedStorage) <= 116224,
              "M64 P16 B2/A3 must preserve two-CTA shared occupancy");

__global__ void general_m64n256_k32_p16_b2a3_c32_kernel(
    TensorMap const* input_c32_map,
    TensorMap const* weight_map,
    Element* output,
    int n_size,
    int d_size,
    int h_size,
    int w_size,
    int c32_groups_per_time,
    int c16_groups_per_time,
    int output_pitch_k,
    int output_channel_base) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  __shared__ M64P16B2A3SharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  int q_base = int(blockIdx.x) * kOutQ;
  int p_base = int(blockIdx.y) * kM64P16OutP;
  int flat_batch_count = n_size * d_size;
  int m_tile = int(blockIdx.z) / flat_batch_count;
  int flat_batch = int(blockIdx.z) - m_tile * flat_batch_count;
  int n = flat_batch / d_size;
  int od = flat_batch - n * d_size;
  int k_base = output_channel_base + m_tile * kTailM;

  int td_begin = od == 0 ? 1 : 0;
  int td_end = od == d_size - 1 ? 2 : kT;
  int local_td_count = td_end - td_begin;
  int local_supergroups = local_td_count * c32_groups_per_time;
  int full_supergroups = kT * c32_groups_per_time;

  constexpr int guard_rows =
      kM64P16BackingRows - kM64P16SemanticRows;
  constexpr int guard_per_stage = guard_rows * 32;
  for (int idx = int(threadIdx.x);
       idx < kM64P16B2Stages * guard_per_stage;
       idx += int(blockDim.x)) {
    int slot = idx / guard_per_stage;
    int rest = idx - slot * guard_per_stage;
    int row = kM64P16SemanticRows + rest / 32;
    int kk = rest % 32;
    shared.b_stage[slot].b[swizzled_b_c32_index(row, kk)] =
        patchshift::element_from_float(0.0f);
  }
  if (threadIdx.x == 0) {
    shared.tmem_base = 0;
    shared.b_published = 0;
    shared.a_published = 0;
    shared.a_release_observed = 0;
    shared.tmem_ready = 0;
#pragma unroll
    for (int slot = 0; slot < kM64P16B2Stages; ++slot) {
      shared.prefix_commit_issued[slot] = 0;
      shared.final_commit_issued[slot] = 0;
      patchshift::mbarrier_init(&shared.b_stage[slot].ready, 1);
      patchshift::mbarrier_init(&shared.b_stage[slot].prefix_done,
                                kM64P16Worksets);
      patchshift::mbarrier_init(&shared.b_stage[slot].done,
                                kM64P16Worksets);
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      patchshift::mbarrier_init(&shared.a_stage[row].ready, 1);
    }
  }
  __syncthreads();

  // Warp 0: B2 activation producer.  Slot reuse waits for both the final MMA
  // completion and explicit observation of that generation by the A producer.
  if (wid == 0 && lane == 0) {
    patchshift::tma_descriptor_fence_acquire(input_c32_map);
    constexpr uint32_t b_bytes =
        kM64P16SemanticRows * 32 * sizeof(Element);
    for (int sg = 0; sg < local_supergroups; ++sg) {
      int b_slot = sg % kM64P16B2Stages;
      int b_seq = sg / kM64P16B2Stages;
      if (b_seq > 0) {
        int old_sg = sg - kM64P16B2Stages;
        while (shared.a_release_observed < old_sg + 1) {
        }
        while (shared.final_commit_issued[b_slot] < b_seq) {
        }
        __threadfence_block();
        while (!patchshift::mbarrier_try_wait(
            &shared.b_stage[b_slot].done, (b_seq - 1) & 1)) {
        }
      }
      int local_td = sg / c32_groups_per_time;
      int c32g = sg - local_td * c32_groups_per_time;
      int td = td_begin + local_td;
      patchshift::mbarrier_arrive_expect_tx(
          &shared.b_stage[b_slot].ready, b_bytes);
      patchshift::tma_load_5d(
          input_c32_map, &shared.b_stage[b_slot].ready,
          shared.b_stage[b_slot].b + swizzled_b_c32_index(0, 0),
          c32g * 32, q_base - 1, p_base - 1,
          od + td - 1, n);
      __threadfence_block();
      shared.b_published = sg + 1;
    }
  }

  // Warp 1: fixed A-row generation protocol.  Rows 0/1 wait on the prefix of
  // sg-1; row 2 waits on its final completion and publishes the release token
  // needed before the corresponding B slot may cycle again.
  if (wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kTailM * kK * sizeof(Element);
    int total_tasks = local_supergroups * 3;
    for (int task = 0; task < total_tasks; ++task) {
      int sg = task / 3;
      int row = task - sg * 3;
      if (sg > 0) {
        int old_sg = sg - 1;
        int old_b_slot = old_sg % kM64P16B2Stages;
        int old_b_seq = old_sg / kM64P16B2Stages;
        if (row < 2) {
          while (shared.prefix_commit_issued[old_b_slot] <
                 old_b_seq + 1) {
          }
          __threadfence_block();
          while (!patchshift::mbarrier_try_wait(
              &shared.b_stage[old_b_slot].prefix_done,
              old_b_seq & 1)) {
          }
        } else {
          while (shared.final_commit_issued[old_b_slot] <
                 old_b_seq + 1) {
          }
          __threadfence_block();
          while (!patchshift::mbarrier_try_wait(
              &shared.b_stage[old_b_slot].done,
              old_b_seq & 1)) {
          }
          __threadfence_block();
          shared.a_release_observed = old_sg + 1;
        }
      }
      int local_td = sg / c32_groups_per_time;
      int c32g = sg - local_td * c32_groups_per_time;
      int td = td_begin + local_td;
      int full_sg = td * c32_groups_per_time + c32g;
      int weight_task =
          (m_tile * full_supergroups + full_sg) * 3 + row;
      patchshift::mbarrier_arrive_expect_tx(
          &shared.a_stage[row].ready, a_row_bytes);
      patchshift::tma_load_5d(
          weight_map, &shared.a_stage[row].ready,
          shared.a_stage[row].a[0][0], 0, 0, 0, 0, weight_task);
      __threadfence_block();
      shared.a_published = task + 1;
    }
  }

  // Warp 2/3 own independent M64N256 destinations at TMEM columns 0/128.
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
    for (int sg = 0; sg < local_supergroups; ++sg) {
      int b_slot = sg % kM64P16B2Stages;
      int b_seq = sg / kM64P16B2Stages;
      while (shared.b_published < sg + 1) {
      }
      while (!patchshift::mbarrier_try_wait(
          &shared.b_stage[b_slot].ready, b_seq & 1)) {
      }
      patchshift::fence_view_async_shared();
      int c32g = sg % c32_groups_per_time;
      int valid_k16_groups =
          min(kK16GroupsPerStage,
              c16_groups_per_time - c32g * kK16GroupsPerStage);
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        int task = sg * 3 + row;
        while (shared.a_published < task + 1) {
        }
        while (!patchshift::mbarrier_try_wait(
            &shared.a_stage[row].ready, sg & 1)) {
        }
        patchshift::fence_view_async_shared();
        issue_m64_p16_c32_workset_row(
            shared.b_stage[b_slot], shared.a_stage[row], row,
            workset, shared.tmem_base, sg == 0 && row == 0,
            valid_k16_groups);
        if (row == 1) {
          patchshift::tcgen05_commit(
              &shared.b_stage[b_slot].prefix_done);
          if (workset == 0 && lane == 0) {
            __threadfence_block();
            shared.prefix_commit_issued[b_slot] = b_seq + 1;
          }
        } else if (row == 2) {
          patchshift::tcgen05_commit(&shared.b_stage[b_slot].done);
          if (workset == 0 && lane == 0) {
            __threadfence_block();
            shared.final_commit_issued[b_slot] = b_seq + 1;
          }
        }
      }
    }
  }

  while (shared.tmem_ready == 0) {
  }
  int final_sg = local_supergroups - 1;
  int final_slot = final_sg % kM64P16B2Stages;
  int final_seq = final_sg / kM64P16B2Stages;
  while (shared.final_commit_issued[final_slot] < final_seq + 1) {
  }
  __threadfence_block();
  while (!patchshift::mbarrier_try_wait(
      &shared.b_stage[final_slot].done, final_seq & 1)) {
  }
  patchshift::tcgen05_fence_after_thread_sync();

  int store_warp = wid & 1;
  int store_group = (wid >> 1) & 1;
  int local_k = store_warp * 32 + lane;
  int global_k = k_base + local_k;
  bool full_tile = k_base + kTailM <= output_pitch_k &&
                   p_base + kM64P16OutP <= h_size &&
                   q_base + kOutQ <= w_size;
#pragma unroll
  for (int workset = 0; workset < kM64P16Worksets; ++workset) {
    uint32_t tile_base =
        shared.tmem_base + uint32_t(workset * kTailAccumulatorColumns);
    for (int physical_col = 0;
         physical_col < kTailAccumulatorColumns;
         physical_col += 32) {
      uint32_t values[32];
      patchshift::tcgen05_load_32dp32b_x32(
          tile_base + physical_col, values);
      patchshift::tcgen05_wait_tmem_load();
      int logical_col =
          store_group * kTailAccumulatorColumns + physical_col;
      int out_p = p_base + workset * kTailOutPPerWorkset +
                  (logical_col >> 5);
      if (full_tile) {
        size_t pixel =
            ((size_t(flat_batch) * size_t(h_size) + size_t(out_p)) *
                 size_t(w_size) +
             size_t(q_base));
        Element* out = output + pixel * size_t(output_pitch_k) +
                       size_t(k_base);
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
                     size_t(w_size) +
                 size_t(out_q));
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
  if (wid == kM64P16Warps - 1) {
    patchshift::tcgen05_fence_after_thread_sync();
    patchshift::tcgen05_dealloc(shared.tmem_base,
                                kM64P16TmemColumns);
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
  (void)output_pitch_k;
  (void)output_channel_base;
#endif
}

// Exact M32 output-channel path for Kout=32, a 32-channel remainder, or a
// complete small-grid stack of consecutive M32 output-channel tiles.
// --------------------------------------------------------------------------
// tcgen05.mma.ws natively supports M32N256K16.  Layout G distributes one
// logical N256 accumulator over four 64-column TMEM partitions.  Two issuer
// warps therefore cover P16xQ30 with two independent M32N256 accumulators,
// while B2 overlaps the next C32 activation tile and A3 retains one packed
// filter row per slot.  Unlike the M64 tail, no output-channel work is padded.
// output_channel_base selects either standalone/multi-tile output starting at
// channel zero or the exact tail following one or more native M128 tiles.
// blockIdx.z selects both an M32 output-channel tile and flattened (N,D).
constexpr int kM32P16M = 32;
constexpr int kM32P16N = kTailN;
constexpr int kM32P16Worksets = kM64P16Worksets;
constexpr int kM32P16OutPPerWorkset = kM32P16N / kPitch;
constexpr int kM32P16OutP =
    kM32P16Worksets * kM32P16OutPPerWorkset;
constexpr int kM32P16AccumulatorColumns = kM32P16N / 4;
constexpr int kM32P16TmemColumns =
    kM32P16Worksets * kM32P16AccumulatorColumns;
constexpr int kM32P16Warps = 4;
constexpr int kM32P16Threads = kM32P16Warps * 32;

static_assert(kM32P16OutP == kM64P16OutP);
static_assert(kM32P16AccumulatorColumns == 64);
static_assert(kM32P16TmemColumns == 128);

struct alignas(256) M32P16ARowStage {
  alignas(128) Element a[3][kK16GroupsPerStage][kM32P16M * kK];
  uint64_t ready;
};

struct M32P16B2A3SharedStorage {
  M64P16B2Stage b_stage[kM64P16B2Stages];
  M32P16ARowStage a_stage[3];
  uint32_t tmem_base;
  volatile int b_published;
  volatile int a_published;
  volatile int prefix_commit_issued[kM64P16B2Stages];
  volatile int final_commit_issued[kM64P16B2Stages];
  volatile int a_release_observed;
  volatile int tmem_ready;
};

static_assert(sizeof(M32P16ARowStage) == 6400);
static_assert(sizeof(M32P16B2A3SharedStorage) == 95232);
static_assert(alignof(M32P16B2A3SharedStorage) == 512);
static_assert(offsetof(M32P16B2A3SharedStorage, a_stage) % 256 == 0);
static_assert(sizeof(M32P16B2A3SharedStorage) <= 116224,
              "M32 P16 B2/A3 must preserve two CTAs per SM");

__host__ __device__ constexpr uint64_t m32_p16_idesc() {
  constexpr uint32_t c_format_f32 = 1u;
  constexpr uint32_t ab_format_bf16 = 1u;
  constexpr uint32_t n_dim = kM32P16N >> 3;
  constexpr uint32_t m_dim = kM32P16M >> 4;
  // Layout G limits column shift to 16.  This convolution uses shifts 0/1/2.
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

__device__ __forceinline__ void issue_m32_p16_c32_workset_row(
    M64P16B2Stage& b_stage,
    M32P16ARowStage& a_stage,
    int filter_row,
    int workset,
    uint32_t tmem_base,
    bool first_row,
    int valid_k16_groups) {
  if (!patchshift::elect_one_sync()) {
    return;
  }
  constexpr uint64_t idesc = m32_p16_idesc();
  uint32_t dst =
      tmem_base + uint32_t(workset * kM32P16AccumulatorColumns);
#pragma unroll
  for (int kg = 0; kg < kK16GroupsPerStage; ++kg) {
    if (kg >= valid_k16_groups) {
      continue;
    }
    Element* b_base =
        b_stage.b +
        swizzled_b_c32_index(
            workset * kM32P16N + filter_row * kPitch, kg * kK);
    uint64_t desc_b = pack_b_c32_desc(b_base);
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

__global__ void general_m32n256_k32_p16_b2a3_c32_kernel(
    TensorMap const* input_c32_map,
    TensorMap const* weight_map,
    Element* output,
    int n_size,
    int d_size,
    int h_size,
    int w_size,
    int c32_groups_per_time,
    int c16_groups_per_time,
    int output_pitch_k,
    int output_channel_base) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  __shared__ M32P16B2A3SharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  int q_base = int(blockIdx.x) * kOutQ;
  int p_base = int(blockIdx.y) * kM32P16OutP;
  int flat_batch_count = n_size * d_size;
  int m32_tile = int(blockIdx.z) / flat_batch_count;
  int flat_batch = int(blockIdx.z) - m32_tile * flat_batch_count;
  int n = flat_batch / d_size;
  int od = flat_batch - n * d_size;
  int output_channel_tile_base =
      output_channel_base + m32_tile * kM32P16M;

  int td_begin = od == 0 ? 1 : 0;
  int td_end = od == d_size - 1 ? 2 : kT;
  int local_td_count = td_end - td_begin;
  int local_supergroups = local_td_count * c32_groups_per_time;

  constexpr int guard_rows =
      kM64P16BackingRows - kM64P16SemanticRows;
  constexpr int guard_per_stage = guard_rows * 32;
  for (int idx = int(threadIdx.x);
       idx < kM64P16B2Stages * guard_per_stage;
       idx += int(blockDim.x)) {
    int slot = idx / guard_per_stage;
    int rest = idx - slot * guard_per_stage;
    int row = kM64P16SemanticRows + rest / 32;
    int kk = rest % 32;
    shared.b_stage[slot].b[swizzled_b_c32_index(row, kk)] =
        patchshift::element_from_float(0.0f);
  }
  if (threadIdx.x == 0) {
    shared.tmem_base = 0;
    shared.b_published = 0;
    shared.a_published = 0;
    shared.a_release_observed = 0;
    shared.tmem_ready = 0;
#pragma unroll
    for (int slot = 0; slot < kM64P16B2Stages; ++slot) {
      shared.prefix_commit_issued[slot] = 0;
      shared.final_commit_issued[slot] = 0;
      patchshift::mbarrier_init(&shared.b_stage[slot].ready, 1);
      patchshift::mbarrier_init(&shared.b_stage[slot].prefix_done,
                                kM32P16Worksets);
      patchshift::mbarrier_init(&shared.b_stage[slot].done,
                                kM32P16Worksets);
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
      patchshift::mbarrier_init(&shared.a_stage[row].ready, 1);
    }
  }
  __syncthreads();

  // Warp 0 publishes one compact P18xQ32xC32 B tile into a two-stage ring.
  if (wid == 0 && lane == 0) {
    patchshift::tma_descriptor_fence_acquire(input_c32_map);
    constexpr uint32_t b_bytes =
        kM64P16SemanticRows * 32 * sizeof(Element);
    for (int sg = 0; sg < local_supergroups; ++sg) {
      int b_slot = sg % kM64P16B2Stages;
      int b_seq = sg / kM64P16B2Stages;
      if (b_seq > 0) {
        int old_sg = sg - kM64P16B2Stages;
        while (shared.a_release_observed < old_sg + 1) {
        }
        while (shared.final_commit_issued[b_slot] < b_seq) {
        }
        __threadfence_block();
        while (!patchshift::mbarrier_try_wait(
            &shared.b_stage[b_slot].done, (b_seq - 1) & 1)) {
        }
      }
      int local_td = sg / c32_groups_per_time;
      int c32g = sg - local_td * c32_groups_per_time;
      int td = td_begin + local_td;
      patchshift::mbarrier_arrive_expect_tx(
          &shared.b_stage[b_slot].ready, b_bytes);
      patchshift::tma_load_5d(
          input_c32_map, &shared.b_stage[b_slot].ready,
          shared.b_stage[b_slot].b + swizzled_b_c32_index(0, 0),
          c32g * 32, q_base - 1, p_base - 1,
          od + td - 1, n);
      __threadfence_block();
      shared.b_published = sg + 1;
    }
  }

  // Warp 1 retains one packed M32 weight stage per filter row.  Rows 0/1 may
  // cycle after the two issuer prefixes complete; row 2 waits for final use.
  if (wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kM32P16M * kK * sizeof(Element);
    int total_tasks = local_supergroups * 3;
    for (int task = 0; task < total_tasks; ++task) {
      int sg = task / 3;
      int row = task - sg * 3;
      if (sg > 0) {
        int old_sg = sg - 1;
        int old_b_slot = old_sg % kM64P16B2Stages;
        int old_b_seq = old_sg / kM64P16B2Stages;
        if (row < 2) {
          while (shared.prefix_commit_issued[old_b_slot] <
                 old_b_seq + 1) {
          }
          __threadfence_block();
          while (!patchshift::mbarrier_try_wait(
              &shared.b_stage[old_b_slot].prefix_done,
              old_b_seq & 1)) {
          }
        } else {
          while (shared.final_commit_issued[old_b_slot] <
                 old_b_seq + 1) {
          }
          __threadfence_block();
          while (!patchshift::mbarrier_try_wait(
              &shared.b_stage[old_b_slot].done, old_b_seq & 1)) {
          }
          __threadfence_block();
          shared.a_release_observed = old_sg + 1;
        }
      }
      int local_td = sg / c32_groups_per_time;
      int c32g = sg - local_td * c32_groups_per_time;
      int td = td_begin + local_td;
      int full_sg = td * c32_groups_per_time + c32g;
      int supergroups_per_m32_tile = kT * c32_groups_per_time;
      int weight_task =
          (m32_tile * supergroups_per_m32_tile + full_sg) * 3 + row;
      patchshift::mbarrier_arrive_expect_tx(
          &shared.a_stage[row].ready, a_row_bytes);
      patchshift::tma_load_5d(
          weight_map, &shared.a_stage[row].ready,
          shared.a_stage[row].a[0][0], 0, 0, 0, 0, weight_task);
      __threadfence_block();
      shared.a_published = task + 1;
    }
  }

  // Warps 2/3 own independent M32N256 destinations at TMEM columns 0/64.
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
    for (int sg = 0; sg < local_supergroups; ++sg) {
      int b_slot = sg % kM64P16B2Stages;
      int b_seq = sg / kM64P16B2Stages;
      while (shared.b_published < sg + 1) {
      }
      while (!patchshift::mbarrier_try_wait(
          &shared.b_stage[b_slot].ready, b_seq & 1)) {
      }
      patchshift::fence_view_async_shared();
      int c32g = sg % c32_groups_per_time;
      int valid_k16_groups =
          min(kK16GroupsPerStage,
              c16_groups_per_time - c32g * kK16GroupsPerStage);
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        int task = sg * 3 + row;
        while (shared.a_published < task + 1) {
        }
        while (!patchshift::mbarrier_try_wait(
            &shared.a_stage[row].ready, sg & 1)) {
        }
        patchshift::fence_view_async_shared();
        issue_m32_p16_c32_workset_row(
            shared.b_stage[b_slot], shared.a_stage[row], row,
            workset, shared.tmem_base, sg == 0 && row == 0,
            valid_k16_groups);
        if (row == 1) {
          patchshift::tcgen05_commit(
              &shared.b_stage[b_slot].prefix_done);
          if (workset == 0 && lane == 0) {
            __threadfence_block();
            shared.prefix_commit_issued[b_slot] = b_seq + 1;
          }
        } else if (row == 2) {
          patchshift::tcgen05_commit(&shared.b_stage[b_slot].done);
          if (workset == 0 && lane == 0) {
            __threadfence_block();
            shared.final_commit_issued[b_slot] = b_seq + 1;
          }
        }
      }
    }
  }

  while (shared.tmem_ready == 0) {
  }
  int final_sg = local_supergroups - 1;
  int final_slot = final_sg % kM64P16B2Stages;
  int final_seq = final_sg / kM64P16B2Stages;
  while (shared.final_commit_issued[final_slot] < final_seq + 1) {
  }
  __threadfence_block();
  while (!patchshift::mbarrier_try_wait(
      &shared.b_stage[final_slot].done, final_seq & 1)) {
  }
  patchshift::tcgen05_fence_after_thread_sync();

  // Layout G: warp-rank % 4 owns one logical N/4 partition. Each lane owns
  // one channel inside the native M32 tile at output_channel_tile_base.
  int store_group = wid & 3;
  int local_k = lane;
  bool full_tile = p_base + kM32P16OutP <= h_size &&
                   q_base + kOutQ <= w_size;
#pragma unroll
  for (int workset = 0; workset < kM32P16Worksets; ++workset) {
    uint32_t tile_base =
        shared.tmem_base +
        uint32_t(workset * kM32P16AccumulatorColumns);
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
                 size_t(w_size) +
             size_t(q_base));
        Element* out = output + pixel * size_t(output_pitch_k) +
                       size_t(output_channel_tile_base);
#pragma unroll
        for (int q = 0; q < kOutQ; ++q) {
          uint16_t bits = patchshift::element_bits_from_float(
              __uint_as_float(values[q]));
          *reinterpret_cast<uint16_t*>(
              out + size_t(q) * size_t(output_pitch_k) + local_k) = bits;
        }
      } else if (out_p < h_size) {
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
                output + pixel * size_t(output_pitch_k) +
                size_t(output_channel_tile_base) + local_k) = bits;
          }
        }
      }
    }
  }
  patchshift::tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (wid == kM32P16Warps - 1) {
    patchshift::tcgen05_fence_after_thread_sync();
    patchshift::tcgen05_dealloc(shared.tmem_base,
                                kM32P16TmemColumns);
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
  (void)output_pitch_k;
  (void)output_channel_base;
#endif
}
