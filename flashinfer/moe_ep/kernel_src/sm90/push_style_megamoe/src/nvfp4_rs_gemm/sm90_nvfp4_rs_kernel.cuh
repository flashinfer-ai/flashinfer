// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#include "decode.cuh"
#include "scheduler.cuh"

#ifndef SM90_NVFP4_RS_N_TACTIC
#define SM90_NVFP4_RS_N_TACTIC 64
#endif

#ifndef SM90_NVFP4_RS_STAGES
#define SM90_NVFP4_RS_STAGES 3
#endif

#ifndef SM90_NVFP4_RS_STAGE_K
#define SM90_NVFP4_RS_STAGE_K 64
#endif

#ifndef SM90_NVFP4_RS_WGMMA_GROUP
#define SM90_NVFP4_RS_WGMMA_GROUP 1
#endif

#ifndef SM90_NVFP4_RS_STATIC_SCHED
#define SM90_NVFP4_RS_STATIC_SCHED 0
#endif

#ifndef SM90_NVFP4_RS_NO_UNION
#define SM90_NVFP4_RS_NO_UNION 0
#endif

namespace flashinfer {
namespace sm90_nvfp4_rs {

constexpr int kStageK = SM90_NVFP4_RS_STAGE_K;
constexpr int kStageSubtiles = kStageK / kBlockK;
constexpr int kWgmmaGroup = SM90_NVFP4_RS_WGMMA_GROUP;
static_assert(kWgmmaGroup == 1 || kWgmmaGroup == 2 || kWgmmaGroup == 4);
static_assert(kStageSubtiles % kWgmmaGroup == 0);

__global__ void validate_offsets_kernel(const int64_t* offsets, int num_experts,
                                        int64_t row_capacity) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int32_t error = 0;
  int64_t previous = offsets[0];
  if (previous != 0) {
    error = 1;
  }
  for (int expert = 1; expert <= num_experts && error == 0; ++expert) {
    int64_t current = offsets[expert];
    if (current < previous) {
      error = 2;
    }
    previous = current;
  }
  if (error == 0 && previous > row_capacity) {
    error = 3;
  }
  if (error != 0) {
    printf("sm90_nvfp4_rs_gemm: invalid offsets, code=%d\n", error);
    asm volatile("trap;");
  }
}

template <int TokenTileN>
__global__ void validate_padded_schedule_kernel(const int64_t* offsets, const int64_t* tile_prefix,
                                                int num_experts, int64_t row_capacity) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int64_t previous_offset = offsets[0];
  int64_t previous_prefix = tile_prefix[0];
  bool invalid = previous_offset != 0 || previous_prefix != 0;
  for (int group = 0; group < num_experts && !invalid; ++group) {
    int64_t current_offset = offsets[group + 1];
    int64_t current_prefix = tile_prefix[group + 1];
    invalid = current_offset < previous_offset || current_offset > row_capacity;
    if (!invalid) {
      int64_t expected =
          previous_prefix + ceil_div_nonnegative(current_offset - previous_offset, TokenTileN);
      invalid = current_prefix != expected;
    }
    previous_offset = current_offset;
    previous_prefix = current_prefix;
  }
  if (invalid) {
    printf("sm90_nvfp4_rs: invalid padded schedule\n");
    asm volatile("trap;");
  }
}

struct alignas(8) TmaBarrier {
  uint64_t value;
};

template <int TokenTileN, int Stages>
struct alignas(256) WgmmaPipelineStorage {
  alignas(256) __nv_bfloat16 activation[Stages][TokenTileN * kStageK];
  alignas(128) uint8_t payload[Stages][kStageSubtiles * kRsThreads * kRsBytesPerThread];
  alignas(128) uint8_t scales[Stages][kStageSubtiles * kTileN];
};

#if SM90_NVFP4_RS_NO_UNION
template <int TokenTileN, int Stages>
struct alignas(256) WgmmaDataStorage {
  WgmmaPipelineStorage<TokenTileN, Stages> pipeline;
  alignas(128) __nv_bfloat16 output[TokenTileN * kBlockM];
};
#else
template <int TokenTileN, int Stages>
union alignas(256) WgmmaDataStorage {
  WgmmaPipelineStorage<TokenTileN, Stages> pipeline;
  alignas(128) __nv_bfloat16 output[TokenTileN * kBlockM];
};
#endif

template <int TokenTileN, int Stages>
struct alignas(256) WgmmaSharedStorage {
  WgmmaDataStorage<TokenTileN, Stages> data;
  alignas(8) TmaBarrier full_barriers[Stages];
  GroupedTask task;
};

template <int TokenTileN, int Stages>
constexpr size_t wgmma_smem_bytes() {
  return sizeof(WgmmaSharedStorage<TokenTileN, Stages>);
}

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void tma_barrier_init(TmaBarrier& barrier) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
               :
               : "r"(shared_address(&barrier))
               : "memory");
}

__device__ __forceinline__ void tma_barrier_invalidate(TmaBarrier& barrier) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" : : "r"(shared_address(&barrier)) : "memory");
}

__device__ __forceinline__ void tma_barrier_arrive_expect_tx(TmaBarrier& barrier, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(shared_address(&barrier)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void tma_barrier_wait(TmaBarrier& barrier, uint32_t phase) {
  uint32_t ready;
  do {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(ready)
        : "r"(shared_address(&barrier)), "r"(phase)
        : "memory");
  } while (ready == 0);
}

__device__ __forceinline__ void tma_load_2d(void* destination, const CUtensorMap& tensor_map,
                                            TmaBarrier& barrier, int32_t coordinate_0,
                                            int32_t coordinate_1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(shared_address(destination)), "l"(reinterpret_cast<uint64_t>(&tensor_map)),
        "r"(coordinate_0), "r"(coordinate_1), "r"(shared_address(&barrier))
      : "memory");
}

__device__ __forceinline__ void prefetch_tma_map(const CUtensorMap& tensor_map) {
  asm volatile("prefetch.tensormap [%0];"
               :
               : "l"(reinterpret_cast<uint64_t>(&tensor_map))
               : "memory");
}

__device__ __forceinline__ uint64_t make_b_descriptor(const __nv_bfloat16* pointer) {
  constexpr uint64_t kStride = (8 * kBlockK * sizeof(__nv_bfloat16)) >> 4;
  const uint64_t start = static_cast<uint64_t>(shared_address(pointer) >> 4);
  return (start & 0x3fffULL) | ((kStride & 0x3fffULL) << 32) | (3ULL << 62);
}

template <int TokenTileN, int Stages>
__device__ __forceinline__ void initialize_barriers(WgmmaSharedStorage<TokenTileN, Stages>& storage,
                                                    int active_stages) {
  if (threadIdx.x == 0) {
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage) {
      if (stage < active_stages) {
        tma_barrier_init(storage.full_barriers[stage]);
      }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
}

template <int TokenTileN, int Stages>
__device__ __forceinline__ void invalidate_barriers(WgmmaSharedStorage<TokenTileN, Stages>& storage,
                                                    int active_stages) {
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage) {
      if (stage < active_stages) {
        tma_barrier_invalidate(storage.full_barriers[stage]);
      }
    }
  }
  __syncthreads();
}

template <bool DirectBf16A, int TokenTileN, int Stages>
__device__ __forceinline__ void prefetch_stage(WgmmaSharedStorage<TokenTileN, Stages>& storage,
                                               int stage, int k_stage, const GroupedTask& task,
                                               int output_tiles, int k_tiles,
                                               const CUtensorMap& activation_map,
                                               const CUtensorMap& payload_map,
                                               const CUtensorMap& scale_map) {
  if (threadIdx.x == 0) {
    TmaBarrier& barrier = storage.full_barriers[stage];
    int first_k_tile = k_stage * kStageSubtiles;
#pragma unroll
    for (int subtile = 0; subtile < kStageSubtiles; ++subtile) {
      tma_load_2d(storage.data.pipeline.activation[stage] + subtile * TokenTileN * kBlockK,
                  activation_map, barrier, (first_k_tile + subtile) * kBlockK,
                  static_cast<int32_t>(task.row_begin));
    }
    if constexpr (!DirectBf16A) {
      const int64_t tile_index =
          (static_cast<int64_t>(task.group) * output_tiles + task.output_tile) * k_tiles +
          first_k_tile;
      tma_load_2d(storage.data.pipeline.payload[stage], payload_map, barrier, 0,
                  static_cast<int32_t>(tile_index * 2));
      tma_load_2d(storage.data.pipeline.scales[stage], scale_map, barrier, 0,
                  static_cast<int32_t>(tile_index));
    }
    constexpr uint32_t kBytes =
        TokenTileN * kStageK * sizeof(__nv_bfloat16) +
        (DirectBf16A ? 0 : kStageSubtiles * (kRsThreads * kRsBytesPerThread + kTileN));
    tma_barrier_arrive_expect_tx(barrier, kBytes);
  }
}

__device__ __forceinline__ void load_a_fragment(uint32_t (&fragment)[4],
                                                const uint8_t* payload_tile,
                                                const uint8_t* scale_tile) {
  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / 32;
  const int lane = thread % 32;
  const int row_0 = warp * 16 + lane / 4;
  const int row_1 = row_0 + 8;
  uint32_t scale_0 = 0;
  uint32_t scale_1 = 0;
  if ((lane & 3) == 0) {
    scale_0 = __bfloat16_as_ushort(__float2bfloat16_rn(decode_e4m3(scale_tile[row_0])));
    scale_1 = __bfloat16_as_ushort(__float2bfloat16_rn(decode_e4m3(scale_tile[row_1])));
  }
  scale_0 = __shfl_sync(0xffffffffU, scale_0, lane & ~3);
  scale_1 = __shfl_sync(0xffffffffU, scale_1, lane & ~3);
  const uint32_t packed =
      *reinterpret_cast<const uint32_t*>(payload_tile + thread * kRsBytesPerThread);
  fragment[0] = decode_pair_scaled_bf16(static_cast<uint8_t>(packed), scale_0);
  fragment[1] = decode_pair_scaled_bf16(static_cast<uint8_t>(packed >> 8), scale_1);
  fragment[2] = decode_pair_scaled_bf16(static_cast<uint8_t>(packed >> 16), scale_0);
  fragment[3] = decode_pair_scaled_bf16(static_cast<uint8_t>(packed >> 24), scale_1);
}

__device__ __forceinline__ void load_bf16_a_fragment(uint32_t (&fragment)[4],
                                                     const __nv_bfloat16* weights,
                                                     const GroupedTask& task, int k_tile,
                                                     int shape_n, int shape_k) {
  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / 32;
  const int lane = thread % 32;
  const int row_0 = warp * 16 + lane / 4;
  const int row_1 = row_0 + 8;
  const int k_0 = (lane % 4) * 2;
  const int64_t tile_base =
      (static_cast<int64_t>(task.group) * shape_n + task.output_tile * kBlockM) * shape_k +
      k_tile * kBlockK;
  const __nv_bfloat16* weight_tile = weights + tile_base;
  fragment[0] = __ldg(reinterpret_cast<const uint32_t*>(weight_tile + row_0 * shape_k + k_0));
  fragment[1] = __ldg(reinterpret_cast<const uint32_t*>(weight_tile + row_1 * shape_k + k_0));
  fragment[2] = __ldg(reinterpret_cast<const uint32_t*>(weight_tile + row_0 * shape_k + k_0 + 8));
  fragment[3] = __ldg(reinterpret_cast<const uint32_t*>(weight_tile + row_1 * shape_k + k_0 + 8));
}

template <int TokenTileN, bool DirectBf16A>
__device__ __forceinline__ void run_scalar_task(const GroupedTask& task, __nv_bfloat16* output,
                                                const __nv_bfloat16* activations,
                                                const uint8_t* payload, const uint8_t* scales,
                                                const __nv_bfloat16* canonical_weights,
                                                const float* alpha, int shape_n, int shape_k) {
  const int thread = static_cast<int>(threadIdx.x);
  const int output_tiles = shape_n / kBlockM;
  const int k_tiles = shape_k / kBlockK;
  for (int element = thread; element < TokenTileN * kBlockM; element += kThreads) {
    const int local_row = element / kBlockM;
    const int local_output = element % kBlockM;
    const int64_t global_row = task.row_begin + local_row;
    const int global_output = task.output_tile * kBlockM + local_output;
    if (global_row < task.row_end) {
      float accumulator = 0.0F;
      for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int k_begin = k_tile * kBlockK;
        const int64_t tile_index =
            (static_cast<int64_t>(task.group) * output_tiles + task.output_tile) * k_tiles + k_tile;
        const uint8_t* payload_tile =
            DirectBf16A ? nullptr : payload + tile_index * kRsThreads * kRsBytesPerThread;
        const float scale =
            DirectBf16A ? 1.0F : decode_e4m3(scales[tile_index * kTileN + local_output]);
#pragma unroll
        for (int local_k = 0; local_k < kBlockK; ++local_k) {
          float weight;
          if constexpr (DirectBf16A) {
            const int64_t index =
                (static_cast<int64_t>(task.group) * shape_n + global_output) * shape_k + k_begin +
                local_k;
            weight = __bfloat162float(canonical_weights[index]);
          } else {
            const int warp = local_output / 16;
            const int row_in_warp = local_output % 16;
            const int row_group = row_in_warp & 7;
            const int pair = local_k / 2;
            const int lane = row_group * 4 + (pair & 3);
            const int fragment_index = (pair >= 4 ? 2 : 0) + (row_in_warp >= 8 ? 1 : 0);
            const int payload_thread = warp * 32 + lane;
            const uint8_t packed =
                payload_tile[payload_thread * kRsBytesPerThread + fragment_index];
            const uint8_t code = local_k & 1 ? static_cast<uint8_t>(packed >> 4)
                                             : static_cast<uint8_t>(packed & 0x0fU);
            weight = __bfloat162float(__float2bfloat16_rn(decode_e2m1(code) * scale));
          }
          const float activation = __bfloat162float(
              activations[global_row * static_cast<int64_t>(shape_k) + k_begin + local_k]);
          accumulator = fmaf(weight, activation, accumulator);
        }
      }
      output[global_row * static_cast<int64_t>(shape_n) + global_output] =
          __float2bfloat16_rn(accumulator * __ldg(alpha + task.group));
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void compiler_fence(uint32_t (&fragment)[4]) {
  asm volatile(""
               : "+r"(fragment[0]), "+r"(fragment[1]), "+r"(fragment[2]), "+r"(fragment[3])
               :
               : "memory");
}

template <int Count>
__device__ __forceinline__ void compiler_fence(float (&accumulator)[Count]) {
#pragma unroll
  for (int index = 0; index < Count; ++index) {
    asm volatile("" : "+f"(accumulator[index]) : : "memory");
  }
}

template <int TokenTileN, int Stages>
__device__ __forceinline__ void store_wgmma_output(
    WgmmaSharedStorage<TokenTileN, Stages>& storage, const GroupedTask& task, __nv_bfloat16* output,
    int32_t shape_n, float alpha,
    float (&accumulator)[WgmmaRsBf16<TokenTileN>::kAccumulatorCount]) {
  constexpr int kAccumulatorCount = WgmmaRsBf16<TokenTileN>::kAccumulatorCount;
  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / 32;
  const int lane = thread % 32;
  int destination_offset;
  if (lane < 8) {
    destination_offset = lane * kBlockM;
  } else if (lane < 16) {
    destination_offset = (lane - 8) * kBlockM + 8;
  } else if (lane < 24) {
    destination_offset = (lane - 8) * kBlockM;
  } else {
    destination_offset = (lane - 16) * kBlockM + 8;
  }
#pragma unroll
  for (int index = 0; index < kAccumulatorCount / 8; ++index) {
    stsm_t_x4(
        float2_to_bf16x2(accumulator[index * 8 + 0] * alpha, accumulator[index * 8 + 1] * alpha),
        float2_to_bf16x2(accumulator[index * 8 + 2] * alpha, accumulator[index * 8 + 3] * alpha),
        float2_to_bf16x2(accumulator[index * 8 + 4] * alpha, accumulator[index * 8 + 5] * alpha),
        float2_to_bf16x2(accumulator[index * 8 + 6] * alpha, accumulator[index * 8 + 7] * alpha),
        storage.data.output + warp * 16 + index * 16 * kBlockM + destination_offset);
  }
  __syncthreads();
  constexpr int kVectorsPerRow = kBlockM * sizeof(__nv_bfloat16) / sizeof(int4);
  constexpr int kVectorCount = TokenTileN * kVectorsPerRow;
  const int4* shared_vectors = reinterpret_cast<const int4*>(storage.data.output);
  for (int index = thread; index < kVectorCount; index += kThreads) {
    const int local_row = index / kVectorsPerRow;
    const int local_vector = index % kVectorsPerRow;
    const int64_t global_row = task.row_begin + local_row;
    if (global_row < task.row_end) {
      int4* global_vectors = reinterpret_cast<int4*>(
          output + global_row * static_cast<int64_t>(shape_n) + task.output_tile * kBlockM);
      global_vectors[local_vector] = shared_vectors[index];
    }
  }
  __syncthreads();
}

template <int TokenTileN, int Stages, bool DirectBf16A>
__global__ __launch_bounds__(kThreads) void grouped_rs_wgmma_kernel(
    __nv_bfloat16* output, const __nv_bfloat16* canonical_weights, const float* alpha,
    const int64_t* offsets, const int64_t* tile_prefix, unsigned long long* task_counter,
    int64_t row_capacity, int32_t shape_n, int32_t shape_k, int32_t num_groups,
    __grid_constant__ const CUtensorMap activation_map,
    __grid_constant__ const CUtensorMap payload_map,
    __grid_constant__ const CUtensorMap scale_map) {
  static_assert(TokenTileN == 16 || TokenTileN == 32 || TokenTileN == 64 || TokenTileN == 96 ||
                TokenTileN == 128);
  static_assert(Stages == 3);
  extern __shared__ __align__(256) uint8_t dynamic_shared[];
  auto& storage = *reinterpret_cast<WgmmaSharedStorage<TokenTileN, Stages>*>(dynamic_shared);
  constexpr int kAccumulatorCount = WgmmaRsBf16<TokenTileN>::kAccumulatorCount;
  const int thread = static_cast<int>(threadIdx.x);
  const int output_tiles = shape_n / kBlockM;
  const int k_tiles = shape_k / kBlockK;
  const int k_stages = shape_k / kStageK;
  if (thread == 0) {
    prefetch_tma_map(activation_map);
    if constexpr (!DirectBf16A) {
      prefetch_tma_map(payload_map);
      prefetch_tma_map(scale_map);
    }
  }
  __syncthreads();

#if SM90_NVFP4_RS_STATIC_SCHED
  uint64_t static_next_task = blockIdx.x;
#endif
  while (true) {
    if (thread == 0) {
#if SM90_NVFP4_RS_STATIC_SCHED
      const uint64_t task_index = static_next_task;
      (void)task_counter;
#else
      const uint64_t task_index = atomicAdd(task_counter, 1ULL);
#endif
      storage.task =
          tile_prefix == nullptr
              ? map_grouped_task<TokenTileN>(task_index, offsets, num_groups, output_tiles,
                                             row_capacity)
              : map_grouped_task_prefix<TokenTileN>(task_index, offsets, tile_prefix, num_groups,
                                                    output_tiles, row_capacity);
    }
#if SM90_NVFP4_RS_STATIC_SCHED
    static_next_task += gridDim.x;
#endif
    __syncthreads();
    if (!storage.task.valid) {
      return;
    }

    float accumulator[kAccumulatorCount] = {};
    uint32_t a_fragments[kWgmmaGroup][4];
    const int active_stages = k_stages < Stages ? k_stages : Stages;
    initialize_barriers(storage, active_stages);
    for (int stage = 0; stage < active_stages; ++stage) {
      prefetch_stage<DirectBf16A>(storage, stage, stage, storage.task, output_tiles, k_tiles,
                                  activation_map, payload_map, scale_map);
    }

    bool accumulate = false;
    for (int k_base = 0; k_base < k_tiles; k_base += kWgmmaGroup) {
#pragma unroll
      for (int member = 0; member < kWgmmaGroup; ++member) {
        const int k_tile = k_base + member;
        const int k_stage = k_tile / kStageSubtiles;
        const int subtile = k_tile % kStageSubtiles;
        const int stage = k_stage % Stages;
        const int phase = (k_stage / Stages) & 1;
        if (subtile == 0) {
          tma_barrier_wait(storage.full_barriers[stage], phase);
        }
        if constexpr (DirectBf16A) {
          load_bf16_a_fragment(a_fragments[member], canonical_weights, storage.task, k_tile,
                               shape_n, shape_k);
        } else {
          load_a_fragment(
              a_fragments[member],
              storage.data.pipeline.payload[stage] + subtile * kRsThreads * kRsBytesPerThread,
              storage.data.pipeline.scales[stage] + subtile * kTileN);
        }
        compiler_fence(a_fragments[member]);
      }
      compiler_fence(accumulator);
      wgmma_fence();
#pragma unroll
      for (int member = 0; member < kWgmmaGroup; ++member) {
        const int k_tile = k_base + member;
        const int k_stage = k_tile / kStageSubtiles;
        const int subtile = k_tile % kStageSubtiles;
        const int stage = k_stage % Stages;
        const uint64_t b_descriptor = make_b_descriptor(storage.data.pipeline.activation[stage] +
                                                        subtile * TokenTileN * kBlockK);
        WgmmaRsBf16<TokenTileN>::mma(a_fragments[member], b_descriptor, accumulator, accumulate);
        accumulate = true;
      }
      wgmma_commit();
      compiler_fence(accumulator);
      // Complete the committed WGMMA group before the next iteration reuses accumulator.
      wgmma_wait<0>();
      compiler_fence(accumulator);
#pragma unroll
      for (int member = 0; member < kWgmmaGroup; ++member) {
        compiler_fence(a_fragments[member]);
      }
      if (k_base > 0) {
#pragma unroll
        for (int member = 0; member < kWgmmaGroup; ++member) {
          const int completed_k_tile = k_base - kWgmmaGroup + member;
          if (completed_k_tile % kStageSubtiles == kStageSubtiles - 1) {
            const int completed_k_stage = completed_k_tile / kStageSubtiles;
            const int completed_stage = completed_k_stage % Stages;
            const int next_k_stage = completed_k_stage + Stages;
            if (next_k_stage < k_stages) {
              prefetch_stage<DirectBf16A>(storage, completed_stage, next_k_stage, storage.task,
                                          output_tiles, k_tiles, activation_map, payload_map,
                                          scale_map);
            }
          }
        }
      }
    }
    wgmma_wait<0>();
    compiler_fence(accumulator);

    store_wgmma_output(storage, storage.task, output, shape_n, __ldg(alpha + storage.task.group),
                       accumulator);
    invalidate_barriers(storage, active_stages);
  }
}

template <int TokenTileN>
__global__ __launch_bounds__(kThreads) void grouped_rs_scalar_kernel(
    __nv_bfloat16* output, const __nv_bfloat16* activations, const uint8_t* payload,
    const uint8_t* scales, const float* alpha, const int64_t* offsets,
    unsigned long long* task_counter, int64_t row_capacity, int32_t shape_n, int32_t shape_k,
    int32_t num_groups) {
  __shared__ GroupedTask task;
  const int thread = static_cast<int>(threadIdx.x);
  const int output_tiles = shape_n / kBlockM;

  while (true) {
    if (thread == 0) {
      const uint64_t task_index = atomicAdd(task_counter, 1ULL);
      task =
          map_grouped_task<TokenTileN>(task_index, offsets, num_groups, output_tiles, row_capacity);
    }
    __syncthreads();
    if (!task.valid) {
      return;
    }
    run_scalar_task<TokenTileN, false>(task, output, activations, payload, scales, nullptr, alpha,
                                       shape_n, shape_k);
  }
}

static_assert(SM90_NVFP4_RS_N_TACTIC == 16 || SM90_NVFP4_RS_N_TACTIC == 32 ||
              SM90_NVFP4_RS_N_TACTIC == 64 || SM90_NVFP4_RS_N_TACTIC == 96 ||
              SM90_NVFP4_RS_N_TACTIC == 128);
static_assert(SM90_NVFP4_RS_STAGES == 3);
static_assert(kStageK == 64 || kStageK == 128);

}  // namespace sm90_nvfp4_rs
}  // namespace flashinfer
