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

// M64 output-channel tails and multi-issuer tail mainloop.
// Included by the PatchShift kernel umbrella inside its detail namespace.


struct alignas(256) M64K32FullStage {
  alignas(256) Element b[kK16GroupsPerStage][kTailBackingRows * kK];
  alignas(128) Element a[3][3][kK16GroupsPerStage][kTailM * kK];
  uint64_t ready;
  uint64_t done;
};

struct M64K32SharedStorage {
  M64K32FullStage stage[2];
  uint32_t tmem_base;
  volatile int published;
  volatile int commit_issued[2];
  volatile int tmem_ready;
};

static_assert(sizeof(M64K32SharedStorage) <= 232448,
              "M64 K32 double buffer must fit one SM100 CTA");
static_assert(sizeof(M64K32FullStage) == 107264);
static_assert(sizeof(M64K32SharedStorage) == 214784);

__host__ __device__ constexpr uint64_t m64_k32_idesc() {
  constexpr uint32_t c_format_f32 = 1u;
  constexpr uint32_t ab_format_bf16 = 1u;
  constexpr uint32_t n_dim = kTailN >> 3;
  constexpr uint32_t m_dim = kTailM >> 4;
  constexpr uint32_t max_shift32 = 3u;
  uint32_t desc = 0;
  desc |= c_format_f32 << 4;
  desc |= ab_format_bf16 << 7;
  desc |= ab_format_bf16 << 10;
  desc |= n_dim << 17;
  desc |= m_dim << 24;
  desc |= max_shift32 << 30;
  return uint64_t(desc) << 32;
}

__device__ __forceinline__ void issue_m64_k32_full_stage(
    M64K32FullStage& stage,
    int workset,
    uint32_t tmem_base,
    bool first_stage,
    int valid_k16_groups) {
  if (!patchshift::elect_one_sync()) {
    return;
  }
  constexpr uint64_t idesc = m64_k32_idesc();
  uint32_t dst =
      tmem_base + uint32_t(workset * kTailAccumulatorColumns);
#pragma unroll
  for (int filter_row = 0; filter_row < 3; ++filter_row) {
#pragma unroll
    for (int kg = 0; kg < kK16GroupsPerStage; ++kg) {
      if (kg >= valid_k16_groups) {
        continue;
      }
      Element* b_base = stage.b[kg] +
                        swizzled_b_index(workset * kTailN + filter_row * kPitch, 0);
      uint64_t desc_b = pack_b_desc(b_base);
#pragma unroll
      for (int kw = 0; kw < 3; ++kw) {
        uint64_t desc_a =
            patchshift::pack_k16_desc(stage.a[filter_row][kw][kg], kTailM);
        bool first = first_stage && filter_row == 0 && kg == 0 && kw == 0;
        mma_ws_raw(desc_a, desc_b, dst, first ? 0u : 1u, idesc,
                   patchshift::shift_desc(kw));
      }
    }
  }
}

__global__ void general_m64n256_k32_tail_kernel(
    TensorMap const* input_map,
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
  __shared__ M64K32SharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  int q_base = int(blockIdx.x) * kOutQ;
  int p_base = int(blockIdx.y) * kTailOutP;
  int flat_batch_count = n_size * d_size;
  int m_tile = int(blockIdx.z) / flat_batch_count;
  int flat_batch = int(blockIdx.z) - m_tile * flat_batch_count;
  int n = flat_batch / d_size;
  int od = flat_batch - n * d_size;
  int k_base = output_channel_base + m_tile * kTailM;
  // Match the deep-ILP main tile: compress synchronization generations while
  // preserving the host packer's full three-tap weight address space.
  int td_begin = od == 0 ? 1 : 0;
  int td_end = od == d_size - 1 ? 2 : kT;
  int local_td_count = td_end - td_begin;
  int local_supergroups = local_td_count * c32_groups_per_time;
  int full_supergroups = kT * c32_groups_per_time;

  constexpr int guard_rows = kTailBackingRows - kTailSemanticRows;
  constexpr int guard_per_stage = kK16GroupsPerStage * guard_rows * kK;
  for (int idx = int(threadIdx.x); idx < 2 * guard_per_stage;
       idx += int(blockDim.x)) {
    int stage_idx = idx / guard_per_stage;
    int rest = idx - stage_idx * guard_per_stage;
    int kg = rest / (guard_rows * kK);
    rest -= kg * guard_rows * kK;
    int row = kTailSemanticRows + rest / kK;
    int kk = rest % kK;
    shared.stage[stage_idx].b[kg][swizzled_b_index(row, kk)] =
        patchshift::element_from_float(0.0f);
  }
  if (threadIdx.x == 0) {
    shared.tmem_base = 0;
    shared.published = 0;
    shared.tmem_ready = 0;
    for (int s = 0; s < 2; ++s) {
      shared.commit_issued[s] = 0;
      patchshift::mbarrier_init(&shared.stage[s].ready, 1);
      patchshift::mbarrier_init(&shared.stage[s].done, kTailWorksets);
    }
  }
  __syncthreads();

  if (wid == 0 && lane == 0) {
    constexpr uint32_t b_bytes_per_group =
        kTailSemanticRows * kK * sizeof(Element);
    constexpr uint32_t a_bytes =
        3 * 3 * kK16GroupsPerStage * kTailM * kK * sizeof(Element);
    for (int sg = 0; sg < local_supergroups; ++sg) {
      int slot = sg & 1;
      int seq = sg >> 1;
      M64K32FullStage& stage = shared.stage[slot];
      if (seq > 0) {
        while (shared.commit_issued[slot] < seq) {
        }
        __threadfence_block();
        while (!patchshift::mbarrier_try_wait(&stage.done,
                                               (seq - 1) & 1)) {
        }
      }
      int local_td = sg / c32_groups_per_time;
      int c32g = sg - local_td * c32_groups_per_time;
      int td = td_begin + local_td;
      int full_sg = td * c32_groups_per_time + c32g;
      int valid_k16_groups =
          min(kK16GroupsPerStage, c16_groups_per_time - c32g * kK16GroupsPerStage);
      patchshift::mbarrier_arrive_expect_tx(
          &stage.ready, valid_k16_groups * b_bytes_per_group + a_bytes);
#pragma unroll
      for (int kg = 0; kg < kK16GroupsPerStage; ++kg) {
        if (kg >= valid_k16_groups) {
          continue;
        }
        patchshift::tma_load_5d(
            input_map, &stage.ready,
            stage.b[kg] + swizzled_b_index(0, 0),
            c32g * (kK16GroupsPerStage * kK) + kg * kK, q_base - 1, p_base - 1,
            od + td - 1, n);
      }
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        int weight_task =
            (m_tile * full_supergroups + full_sg) * 3 + row;
        patchshift::tma_load_5d(weight_map, &stage.ready, stage.a[row][0],
                                0, 0, 0, 0, weight_task);
      }
      patchshift::fence_view_async_shared();
      __threadfence_block();
      shared.published = sg + 1;
    }
  }

  if (wid == 1) {
    patchshift::tcgen05_alloc(&shared.tmem_base, kTailTmemColumns);
    __syncwarp();
    patchshift::tcgen05_relinquish_alloc_permit();
    if (lane == 0) {
      __threadfence_block();
      shared.tmem_ready = 1;
    }
  }

  if (wid >= 1 && wid < 1 + kTailIssueWarps) {
    int workset = wid - 1;
    if (wid != 1) {
      while (shared.tmem_ready == 0) {
      }
      __threadfence_block();
    }
    for (int sg = 0; sg < local_supergroups; ++sg) {
      int slot = sg & 1;
      int seq = sg >> 1;
      while (shared.published < sg + 1) {
      }
      while (!patchshift::mbarrier_try_wait(&shared.stage[slot].ready,
                                             seq & 1)) {
      }
      patchshift::fence_view_async_shared();
      int c32g = sg % c32_groups_per_time;
      int valid_k16_groups =
          min(kK16GroupsPerStage, c16_groups_per_time - c32g * kK16GroupsPerStage);
      issue_m64_k32_full_stage(shared.stage[slot], workset,
                               shared.tmem_base, sg == 0,
                               valid_k16_groups);
      patchshift::tcgen05_commit(&shared.stage[slot].done);
      if (lane == 0) {
        __threadfence_block();
        shared.commit_issued[slot] = seq + 1;
      }
    }
  }

  while (shared.tmem_ready == 0) {
  }
  int final_stage = local_supergroups - 1;
  int final_slot = final_stage & 1;
  int final_seq = final_stage >> 1;
  while (shared.commit_issued[final_slot] < final_seq + 1) {
  }
  __threadfence_block();
  while (!patchshift::mbarrier_try_wait(&shared.stage[final_slot].done,
                                         final_seq & 1)) {
  }
  patchshift::tcgen05_fence_after_thread_sync();

  if (wid < 4) {
    int store_warp = wid & 1;
    int store_group = (wid >> 1) & 1;
    int local_k = store_warp * 32 + lane;
    int global_k = k_base + local_k;
    bool full_tile = k_base + kTailM <= output_pitch_k &&
                     p_base + kTailOutP <= h_size &&
                     q_base + kOutQ <= w_size;
#pragma unroll
    for (int workset = 0; workset < kTailWorksets; ++workset) {
      uint32_t tile_base =
          shared.tmem_base + uint32_t(workset * kTailAccumulatorColumns);
      for (int physical_col = 0; physical_col < kTailAccumulatorColumns;
           physical_col += 32) {
        uint32_t values[32];
        patchshift::tcgen05_load_32dp32b_x32(tile_base + physical_col,
                                              values);
        patchshift::tcgen05_wait_tmem_load();
        int logical_col = store_group * kTailAccumulatorColumns + physical_col;
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
  }
  patchshift::tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (wid == kTailWarps - 1) {
    patchshift::tcgen05_fence_after_thread_sync();
    patchshift::tcgen05_dealloc(shared.tmem_base, kTailTmemColumns);
  }
#else
  (void)input_map;
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

// Shared B2/A4 stages for the final M64 C32 multi-issuer tail.
constexpr int kM64DeepBC32BRing = 2;
constexpr int kM64DeepBC32ARing = 4;

struct alignas(512) M64DeepBC32BStage {
  alignas(512) Element b[kTailBackingRows * 32];
  uint64_t ready;
  uint64_t prefix_done;
  uint64_t done;
};

struct alignas(256) M64DeepBC32ARowStage {
  alignas(128) Element a[3][kK16GroupsPerStage][kTailM * kK];
  uint64_t ready;
};

static_assert(sizeof(M64DeepBC32BStage) == 70656);
static_assert(sizeof(M64DeepBC32ARowStage) == 12544);

// M64 C32-B tail with four independent MMA issuers
// -------------------------------------------------
// This variant intentionally changes only the consumer topology of the
// isolated M64 C32-B path.  Warp 2..5 own workset 0..3 respectively, so each
// warp carries one M64N256 accumulator dependency chain.  Four commits arrive
// at the same prefix/final barriers; the B2/A4 producer lifetime remains
// identical across all four consumer worksets.
constexpr int kM64DeepBC32MultiIssuerWarps = 6;
constexpr int kM64DeepBC32MultiIssuerThreads =
    kM64DeepBC32MultiIssuerWarps * 32;

struct M64DeepBC32MultiSharedStorage {
  M64DeepBC32BStage b_stage[kM64DeepBC32BRing];
  M64DeepBC32ARowStage a_stage[kM64DeepBC32ARing];
  uint32_t tmem_base;
  volatile int b_published;
  volatile int a_published;
  volatile int prefix_commit_issued[kM64DeepBC32BRing];
  volatile int final_commit_issued[kM64DeepBC32BRing];
  // Consumer 0 is the sole software-counter writer.  The counter only makes
  // the parity generation observable; the four-party mbarrier remains the
  // authoritative proof that every workset's asynchronous MMA has completed.
  volatile int a_release_observed;
  volatile int tmem_ready;
};

static_assert(sizeof(M64DeepBC32MultiSharedStorage) == 192000);
static_assert(alignof(M64DeepBC32MultiSharedStorage) == 512);
static_assert(offsetof(M64DeepBC32MultiSharedStorage, a_stage) % 256 == 0);
static_assert(sizeof(M64DeepBC32MultiSharedStorage) <= 232448,
              "M64 multi-issuer C32-B pipeline must fit one SM100 CTA");

__device__ __forceinline__ void issue_m64_deep_b_c32_workset_row(
    M64DeepBC32BStage& b_stage,
    M64DeepBC32ARowStage& a_stage,
    int filter_row,
    int workset,
    uint32_t tmem_base,
    bool first_row,
    int valid_k16_groups) {
  if (!patchshift::elect_one_sync()) {
    return;
  }
  constexpr uint64_t idesc = m64_k32_idesc();
  uint32_t dst =
      tmem_base + uint32_t(workset * kTailAccumulatorColumns);
#pragma unroll
  for (int kg = 0; kg < kK16GroupsPerStage; ++kg) {
    if (kg >= valid_k16_groups) {
      continue;
    }
    int k_offset = kg * kK;
    Element* b_base =
        b_stage.b +
        swizzled_b_c32_index(
            workset * kTailN + filter_row * kPitch, k_offset);
    uint64_t desc_b = pack_b_c32_desc(b_base);
#pragma unroll
    for (int kw = 0; kw < 3; ++kw) {
      uint64_t desc_a =
          patchshift::pack_k16_desc(a_stage.a[kw][kg], kTailM);
      bool first = first_row && kg == 0 && kw == 0;
      mma_ws_raw(desc_a, desc_b, dst, first ? 0u : 1u, idesc,
                 patchshift::shift_desc(kw));
    }
  }
}

__global__ void general_m64n256_k32_deep_b_c32_multi_issuer_tail_kernel(
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
  __shared__ M64DeepBC32MultiSharedStorage shared;
  int wid = patchshift::warp_id();
  int lane = patchshift::lane_id();
  int q_base = int(blockIdx.x) * kOutQ;
  int p_base = int(blockIdx.y) * kTailOutP;
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

  constexpr int guard_rows = kTailBackingRows - kTailSemanticRows;
  constexpr int guard_per_stage = guard_rows * 32;
  for (int idx = int(threadIdx.x);
       idx < kM64DeepBC32BRing * guard_per_stage;
       idx += int(blockDim.x)) {
    int slot = idx / guard_per_stage;
    int rest = idx - slot * guard_per_stage;
    int row = kTailSemanticRows + rest / 32;
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
    for (int slot = 0; slot < kM64DeepBC32BRing; ++slot) {
      shared.prefix_commit_issued[slot] = 0;
      shared.final_commit_issued[slot] = 0;
      patchshift::mbarrier_init(&shared.b_stage[slot].ready, 1);
      patchshift::mbarrier_init(
          &shared.b_stage[slot].prefix_done, kTailWorksets);
      patchshift::mbarrier_init(
          &shared.b_stage[slot].done, kTailWorksets);
    }
    for (int slot = 0; slot < kM64DeepBC32ARing; ++slot) {
      patchshift::mbarrier_init(&shared.a_stage[slot].ready, 1);
    }
  }
  __syncthreads();

  // Warp 0: unchanged B2 C32 activation producer.
  if (wid == 0 && lane == 0) {
    constexpr uint32_t b_bytes =
        kTailSemanticRows * 32 * sizeof(Element);
    int b_slot = 0;
    int b_seq = 0;
    int local_td = 0;
    int c32g = 0;
    for (int sg = 0; sg < local_supergroups; ++sg) {
      if (b_seq > 0) {
        int old_sg = sg - kM64DeepBC32BRing;
        while (shared.a_release_observed < old_sg + 1) {
        }
        while (shared.final_commit_issued[b_slot] < b_seq) {
        }
        __threadfence_block();
        while (!patchshift::mbarrier_try_wait(
            &shared.b_stage[b_slot].done, (b_seq - 1) & 1)) {
        }
      }
      int td = td_begin + local_td;
      patchshift::mbarrier_arrive_expect_tx(
          &shared.b_stage[b_slot].ready, b_bytes);
      patchshift::tma_load_5d(
          input_c32_map, &shared.b_stage[b_slot].ready,
          shared.b_stage[b_slot].b + swizzled_b_c32_index(0, 0),
          c32g * 32, q_base - 1, p_base - 1,
          od + td - 1, n);
      patchshift::fence_view_async_shared();
      __threadfence_block();
      shared.b_published = sg + 1;
      if (++b_slot == kM64DeepBC32BRing) {
        b_slot = 0;
        ++b_seq;
      }
      if (++c32g == c32_groups_per_time) {
        c32g = 0;
        ++local_td;
      }
    }
  }

  // Warp 1: unchanged A4 packed-row producer and reuse protocol.
  if (wid == 1 && lane == 0) {
    constexpr uint32_t a_row_bytes =
        3 * kK16GroupsPerStage * kTailM * kK * sizeof(Element);
    int4 slot_meta0 = make_int4(-1, 0, 0, 0);
    int4 slot_meta1 = make_int4(-1, 0, 0, 0);
    int4 slot_meta2 = make_int4(-1, 0, 0, 0);
    int4 slot_meta3 = make_int4(-1, 0, 0, 0);
    int sg = 0;
    int row = 0;
    int a_slot = 0;
    int b_slot = 0;
    int b_seq = 0;
    int full_sg = td_begin * c32_groups_per_time;
    int total_tasks = local_supergroups * 3;
    for (int task = 0; task < total_tasks; ++task) {
      int4 current_meta = make_int4(sg, row, b_slot, b_seq);
      int4 old_meta;
      switch (a_slot) {
        case 0:
          old_meta = slot_meta0;
          slot_meta0 = current_meta;
          break;
        case 1:
          old_meta = slot_meta1;
          slot_meta1 = current_meta;
          break;
        case 2:
          old_meta = slot_meta2;
          slot_meta2 = current_meta;
          break;
        default:
          old_meta = slot_meta3;
          slot_meta3 = current_meta;
          break;
      }
      if (old_meta.x >= 0) {
        int old_sg = old_meta.x;
        int old_row = old_meta.y;
        int old_b_slot = old_meta.z;
        int old_b_seq = old_meta.w;
        if (old_row < 2) {
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
      patchshift::mbarrier_arrive_expect_tx(
          &shared.a_stage[a_slot].ready, a_row_bytes);
      int weight_task =
          (m_tile * full_supergroups + full_sg) * 3 + row;
      patchshift::tma_load_5d(
          weight_map, &shared.a_stage[a_slot].ready,
          shared.a_stage[a_slot].a[0][0], 0, 0, 0, 0, weight_task);
      patchshift::fence_view_async_shared();
      __threadfence_block();
      shared.a_published = task + 1;
      if (++a_slot == kM64DeepBC32ARing) {
        a_slot = 0;
      }
      if (++row == 3) {
        row = 0;
        ++sg;
        ++full_sg;
        if (++b_slot == kM64DeepBC32BRing) {
          b_slot = 0;
          ++b_seq;
        }
      }
    }
  }

  // Warp 2 allocates TMEM; warp 2..5 issue exactly one workset each.
  if (wid == 2) {
    patchshift::tcgen05_alloc(&shared.tmem_base, kTailTmemColumns);
    __syncwarp();
    patchshift::tcgen05_relinquish_alloc_permit();
    if (lane == 0) {
      __threadfence_block();
      shared.tmem_ready = 1;
    }
  }
  if (wid >= 2 && wid < 2 + kTailWorksets) {
    int workset = wid - 2;
    if (wid != 2) {
      while (shared.tmem_ready == 0) {
      }
      __threadfence_block();
    }
    int b_slot = 0;
    int b_seq = 0;
    int task = 0;
    int a_slot = 0;
    int a_generation = 0;
    for (int sg = 0; sg < local_supergroups; ++sg) {
      int c32g = sg % c32_groups_per_time;
      int valid_k16_groups =
          min(kK16GroupsPerStage,
              c16_groups_per_time - c32g * kK16GroupsPerStage);
      while (shared.b_published < sg + 1) {
      }
      while (!patchshift::mbarrier_try_wait(
          &shared.b_stage[b_slot].ready, b_seq & 1)) {
      }
      patchshift::fence_view_async_shared();
#pragma unroll
      for (int row = 0; row < 3; ++row) {
        while (shared.a_published < task + 1) {
        }
        while (!patchshift::mbarrier_try_wait(
            &shared.a_stage[a_slot].ready, a_generation & 1)) {
        }
        patchshift::fence_view_async_shared();
        issue_m64_deep_b_c32_workset_row(
            shared.b_stage[b_slot], shared.a_stage[a_slot], row,
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
        ++task;
        if (++a_slot == kM64DeepBC32ARing) {
          a_slot = 0;
          ++a_generation;
        }
      }
      if (++b_slot == kM64DeepBC32BRing) {
        b_slot = 0;
        ++b_seq;
      }
    }
  }

  while (shared.tmem_ready == 0) {
  }
  int final_sg = local_supergroups - 1;
  int final_slot = final_sg % kM64DeepBC32BRing;
  int final_seq = final_sg / kM64DeepBC32BRing;
  while (shared.final_commit_issued[final_slot] < final_seq + 1) {
  }
  __threadfence_block();
  while (!patchshift::mbarrier_try_wait(
      &shared.b_stage[final_slot].done, final_seq & 1)) {
  }
  patchshift::tcgen05_fence_after_thread_sync();

  // Only four warps load/store the M64 epilogue.  The extra two MMA warps join
  // the final CTA barrier but do not duplicate global output stores.
  if (wid < 4) {
    int store_warp = wid & 1;
    int store_group = (wid >> 1) & 1;
    int local_k = store_warp * 32 + lane;
    int global_k = k_base + local_k;
    bool full_tile = k_base + kTailM <= output_pitch_k &&
                     p_base + kTailOutP <= h_size &&
                     q_base + kOutQ <= w_size;
#pragma unroll
    for (int workset = 0; workset < kTailWorksets; ++workset) {
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
  }
  patchshift::tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (wid == kM64DeepBC32MultiIssuerWarps - 1) {
    patchshift::tcgen05_fence_after_thread_sync();
    patchshift::tcgen05_dealloc(shared.tmem_base, kTailTmemColumns);
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

// Automatic small-grid M64 policy: P16 x Q30, two independent issuers
// -----------------------------------------------------------------------
// This specialization deliberately halves the spatial extent of the
// production M64 P32 CTA.  Each CTA owns two legal M64N256 accumulators, one
// per issuer warp, and therefore consumes only 256 TMEM columns.  The B2/A3
// pipeline overlaps the next compact P18 activation TMA with the current MMA
// stream, while three independent packed-weight row slots keep their shortest
// safe lifetime.  Its 113664-byte shared footprint preserves two CTAs/SM.
constexpr int kM64P16Worksets = 2;
constexpr int kM64P16OutP =
    kM64P16Worksets * kTailOutPPerWorkset;
constexpr int kM64P16InputP = kM64P16OutP + 2;
constexpr int kM64P16SemanticRows = kM64P16InputP * kPitch;
constexpr int kM64P16RequiredRows = kM64P16Worksets * kTailN + 66;
constexpr int kM64P16BackingRows =
    ((kM64P16RequiredRows + 7) / 8) * 8;
constexpr int kM64P16TmemColumns =
    kM64P16Worksets * kTailAccumulatorColumns;
constexpr int kM64P16Warps = 4;
constexpr int kM64P16Threads = kM64P16Warps * 32;

static_assert(kM64P16OutP == 16);
static_assert(kM64P16InputP == 18);
static_assert(kM64P16SemanticRows == 576);
static_assert(kM64P16BackingRows == 584);
static_assert(kM64P16TmemColumns == 256);

struct alignas(256) M64P16ARowStage {
  alignas(128) Element a[3][kK16GroupsPerStage][kTailM * kK];
  uint64_t ready;
};
template <class BStage>
__device__ __forceinline__ void issue_m64_p16_c32_workset_row(
    BStage& b_stage,
    M64P16ARowStage& a_stage,
    int filter_row,
    int workset,
    uint32_t tmem_base,
    bool first_row,
    int valid_k16_groups) {
  if (!patchshift::elect_one_sync()) {
    return;
  }
  constexpr uint64_t idesc = m64_k32_idesc();
  uint32_t dst =
      tmem_base + uint32_t(workset * kTailAccumulatorColumns);
#pragma unroll
  for (int kg = 0; kg < kK16GroupsPerStage; ++kg) {
    if (kg >= valid_k16_groups) {
      continue;
    }
    Element* b_base =
        b_stage.b +
        swizzled_b_c32_index(
            workset * kTailN + filter_row * kPitch, kg * kK);
    uint64_t desc_b = pack_b_c32_desc(b_base);
#pragma unroll
    for (int kw = 0; kw < 3; ++kw) {
      uint64_t desc_a =
          patchshift::pack_k16_desc(a_stage.a[kw][kg], kTailM);
      bool first = first_row && kg == 0 && kw == 0;
      mma_ws_raw(desc_a, desc_b, dst, first ? 0u : 1u, idesc,
                 patchshift::shift_desc(kw));
    }
  }
}
