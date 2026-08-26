/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>

#include <cstddef>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <type_traits>

namespace flashinfer::gemm::dual_bf16_weight::split_k {

using Input = cutlass::bfloat16_t;

enum class OutputType {
  kFloat32,
  kBFloat16,
};

struct alignas(8) BFloat16x4 {
  Input values[4];
};

constexpr int kMaximumTokenCount = 256;
constexpr int kTokenTileSwitch = 64;
constexpr int kSmallTokenTile = 8;
constexpr int kLargeTokenTile = 16;
constexpr int kOutputChannelTile = 64;
constexpr int kReductionTile = 128;
constexpr int kBlockSwizzle = 4;
constexpr int kDefaultStageCount = 4;
constexpr int kMinimumStageCount = 3;
constexpr int kMaximumStageCount = 6;
constexpr int kEpilogueThreads = 128;
constexpr int kThreadCount = 32 + 32 + kEpilogueThreads;
// tcgen05 allocation granularity is 32 columns. The two accumulators use at
// most kLargeTokenTile columns each.
constexpr int kTmemColumns = 32;
static_assert(2 * kLargeTokenTile <= kTmemColumns);
constexpr float kLowScale = 1.0f / 256.0f;

struct Arguments {
  void* output;
  float* partial_output;
  int* tile_counters;
  Input const* activation;
  Input const* weight_high;
  Input const* weight_low;
  int token_count;
  int output_channel_count;
  int reduction_size;
  OutputType output_type = OutputType::kFloat32;
};

struct KernelConfig {
  int split_k;
  int token_tile;
  int stage_count;
  int output_tile_count;
  int token_tile_count;
  int base_tile_count;
  int compute_task_count;
  int grid_size;
  int shared_memory_bytes;
};

namespace detail {

using namespace cute;

struct ComputeTask {
  int token_tile;
  int output_tile;
  int k_chunk;
  int base_tile;
};

template <int SplitK>
CUTE_DEVICE ComputeTask decode_compute_task(int task_id, int token_tile_count,
                                            int output_tile_count) {
  int const tasks_per_token_tile = output_tile_count * SplitK;
  int const full_token_groups = token_tile_count / kBlockSwizzle;
  int const swizzled_task_count = full_token_groups * kBlockSwizzle * tasks_per_token_tile;

  int token_tile = 0;
  int output_and_chunk = 0;
  if (task_id < swizzled_task_count) {
    int const tasks_per_group = kBlockSwizzle * tasks_per_token_tile;
    int const token_group = task_id / tasks_per_group;
    int const offset_in_group = task_id - token_group * tasks_per_group;
    int const token_offset = offset_in_group % kBlockSwizzle;
    token_tile = token_group * kBlockSwizzle + token_offset;
    output_and_chunk = offset_in_group / kBlockSwizzle;
  } else {
    int const tail_task = task_id - swizzled_task_count;
    token_tile = full_token_groups * kBlockSwizzle + tail_task / tasks_per_token_tile;
    output_and_chunk = tail_task % tasks_per_token_tile;
  }

  int const k_chunk = output_and_chunk % SplitK;
  int const output_tile = output_and_chunk / SplitK;
  int const base_tile = token_tile * output_tile_count + output_tile;
  return {token_tile, output_tile, k_chunk, base_tile};
}

template <class WeightSmemLayout, class ActivationSmemLayout, class OutputSmemLayout,
          int StageCount>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<Input, cute::cosize_v<WeightSmemLayout>> weight_low;
  alignas(128) cute::ArrayEngine<Input, cute::cosize_v<WeightSmemLayout>> weight_high;
  alignas(128) cute::ArrayEngine<Input, cute::cosize_v<ActivationSmemLayout>> activation;
  alignas(128) cute::ArrayEngine<float, cute::cosize_v<OutputSmemLayout>> partial_output;

  alignas(16) cute::uint64_t low_ready[StageCount];
  alignas(16) cute::uint64_t high_ready[StageCount];
  alignas(16) cute::uint64_t stage_empty[StageCount];

  alignas(16) cute::uint64_t accumulator_ready;
  alignas(16) cute::uint64_t accumulator_empty;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE auto tensor_weight_low() {
    return cute::make_tensor(cute::make_smem_ptr(weight_low.begin()), WeightSmemLayout{});
  }

  CUTE_DEVICE auto tensor_weight_high() {
    return cute::make_tensor(cute::make_smem_ptr(weight_high.begin()), WeightSmemLayout{});
  }

  CUTE_DEVICE auto tensor_activation() {
    return cute::make_tensor(cute::make_smem_ptr(activation.begin()), ActivationSmemLayout{});
  }

  CUTE_DEVICE auto tensor_partial_output() {
    return cute::make_tensor(cute::make_smem_ptr(partial_output.begin()), OutputSmemLayout{});
  }
};

template <int TokenTile, int StageCount>
struct KernelTraits {
  static constexpr int kTokenTile = TokenTile;
  static constexpr int kStageCount = StageCount;
  using MmaAtom = SM100_MMA_F16BF16_SS<Input, Input, float, kOutputChannelTile, TokenTile,
                                       UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(MmaAtom{}));
  using TileShape = Shape<Int<kOutputChannelTile>, Int<TokenTile>, Int<kReductionTile>>;

  using WeightMmaShape = decltype(partition_shape_A(
      TiledMma{}, make_shape(Int<kOutputChannelTile>{}, Int<kReductionTile>{})));
  using ActivationMmaShape =
      decltype(partition_shape_B(TiledMma{}, make_shape(Int<TokenTile>{}, Int<kReductionTile>{})));

  using WeightSmemLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Input>{}, append(WeightMmaShape{}, Int<StageCount>{}),
      Step<_1, _2, _3>{}));
  using ActivationSmemLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Input>{}, append(ActivationMmaShape{}, Int<StageCount>{}),
      Step<_1, _2, _3>{}));
  // The 32B-granularity MN swizzle matches the TMEM-load thread mapping and
  // removes the four-way bank conflicts of a compact channel-major layout.
  // Its FP32 atom is 32x4, so it tiles both 64x8 and 64x16 exactly.
  using OutputSmemLayout =
      decltype(tile_to_shape(UMMA::Layout_MN_SW128_32B_Atom<float>{},
                             make_shape(Int<kOutputChannelTile>{}, Int<TokenTile>{})));

  using Storage =
      SharedStorage<WeightSmemLayout, ActivationSmemLayout, OutputSmemLayout, StageCount>;
  static constexpr int kSharedMemoryBytes = sizeof(Storage);
};

template <class Output, int SplitK, class Traits, class TmaWeight, class TmaActivation,
          class TmaPartialOutput>
__global__ __launch_bounds__(kThreadCount, 1) void kernel(
    CUTE_GRID_CONSTANT TmaWeight const tma_weight_high,
    CUTE_GRID_CONSTANT TmaWeight const tma_weight_low,
    CUTE_GRID_CONSTANT TmaActivation const tma_activation,
    CUTE_GRID_CONSTANT TmaPartialOutput const tma_partial_output, Output* output,
    float* partial_output, int* tile_counters, int token_count, int output_channel_count,
    int reduction_size) {
  using namespace cute;
  using X = Underscore;
  using TiledMma = typename Traits::TiledMma;
  using Storage = typename Traits::Storage;
  constexpr int kTokenTile = Traits::kTokenTile;
  constexpr int kMainloopStageCount = Traits::kStageCount;

  extern __shared__ char shared_memory[];
  Storage& storage = *reinterpret_cast<Storage*>(shared_memory);

  int const warp_index = int(threadIdx.x) / 32;
  bool const elected_lane = cute::elect_one_sync();

  cute::TMEM::Allocator1Sm tmem_allocator;
  if (warp_index == 1) {
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int stage = 0; stage < kMainloopStageCount; ++stage) {
      initialize_barrier(storage.low_ready[stage], 1);
      initialize_barrier(storage.high_ready[stage], 1);
      initialize_barrier(storage.stage_empty[stage], 1);
    }
    initialize_barrier(storage.accumulator_ready, 1);
    initialize_barrier(storage.accumulator_empty, 1);
  }
  __syncthreads();

  auto tile_shape = make_shape(Int<kOutputChannelTile>{}, Int<kTokenTile>{}, Int<kReductionTile>{});
  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(Int<0>{});

  auto shared_weight_low = storage.tensor_weight_low();
  auto shared_weight_high = storage.tensor_weight_high();
  auto shared_activation = storage.tensor_activation();
  auto shared_partial_output = storage.tensor_partial_output();

  int const output_tile_count =
      (output_channel_count + kOutputChannelTile - 1) / kOutputChannelTile;
  int const token_tile_count = (token_count + kTokenTile - 1) / kTokenTile;
  int const padded_output_channel_count = output_tile_count * kOutputChannelTile;
  int const padded_token_count = token_tile_count * kTokenTile;

  auto global_weight_low =
      tma_weight_low.get_tma_tensor(make_shape(output_channel_count, reduction_size));
  auto global_weight_high =
      tma_weight_high.get_tma_tensor(make_shape(output_channel_count, reduction_size));
  auto global_activation = tma_activation.get_tma_tensor(make_shape(token_count, reduction_size));
  auto global_partial_output = tma_partial_output.get_tma_tensor(
      make_shape(padded_output_channel_count, padded_token_count, Int<SplitK>{}));

  auto tiled_weight_low =
      local_tile(global_weight_low, tile_shape, make_coord(_, _, _), Step<_1, X, _1>{});
  auto tiled_weight_high =
      local_tile(global_weight_high, tile_shape, make_coord(_, _, _), Step<_1, X, _1>{});
  auto tiled_activation =
      local_tile(global_activation, tile_shape, make_coord(_, _, _), Step<X, _1, _1>{});

  auto partitioned_weight_low = cta_mma.partition_A(tiled_weight_low);
  auto partitioned_weight_high = cta_mma.partition_A(tiled_weight_high);
  auto partitioned_activation = cta_mma.partition_B(tiled_activation);

  auto partial_output_tma = tma_partial_output.get_slice(0);
  auto tma_shared_partial_output = partial_output_tma.partition_S(shared_partial_output);
  auto tma_global_partial_output = partial_output_tma.partition_D(global_partial_output);

  auto [global_to_tma_weight_low, tma_to_shared_weight_low] =
      tma_partition(tma_weight_low, Int<0>{}, Layout<_1>{}, group_modes<0, 3>(shared_weight_low),
                    group_modes<0, 3>(partitioned_weight_low));
  auto [global_to_tma_weight_high, tma_to_shared_weight_high] =
      tma_partition(tma_weight_high, Int<0>{}, Layout<_1>{}, group_modes<0, 3>(shared_weight_high),
                    group_modes<0, 3>(partitioned_weight_high));
  auto [global_to_tma_activation, tma_to_shared_activation] =
      tma_partition(tma_activation, Int<0>{}, Layout<_1>{}, group_modes<0, 3>(shared_activation),
                    group_modes<0, 3>(partitioned_activation));

  auto descriptor_weight_low = cta_mma.make_fragment_A(shared_weight_low);
  auto descriptor_weight_high = cta_mma.make_fragment_A(shared_weight_high);
  auto descriptor_activation = cta_mma.make_fragment_B(shared_activation);

  // This tensor gives CuTe the output-tile shape used by TMEM fragments. The
  // epilogue writes the selected split-K plane explicitly.
  auto partial_shape_tensor = make_tensor(
      make_gmem_ptr(partial_output), make_shape(padded_output_channel_count, padded_token_count),
      make_stride(Int<1>{}, padded_output_channel_count));
  auto tiled_partial_shape =
      local_tile(partial_shape_tensor, make_shape(Int<kOutputChannelTile>{}, Int<kTokenTile>{}),
                 make_coord(_, _));
  auto partitioned_partial_shape = cta_mma.partition_C(tiled_partial_shape);
  auto partitioned_shared_partial_output = cta_mma.partition_C(shared_partial_output);

  auto accumulator_low =
      cta_mma.make_fragment_C(partitioned_partial_shape(_, _, _, Int<0>{}, Int<0>{}));
  auto accumulator_high =
      cta_mma.make_fragment_C(partitioned_partial_shape(_, _, _, Int<0>{}, Int<0>{}));
  accumulator_low.data() = storage.tmem_base_ptr;
  accumulator_high.data() = storage.tmem_base_ptr + kTokenTile;

  int const reduction_tile_count = reduction_size / kReductionTile;
  int const base_tile_count = output_tile_count * token_tile_count;
  int const compute_task_count = base_tile_count * SplitK;

  constexpr int kActivationBytes = sizeof(Input) * kTokenTile * kReductionTile;
  constexpr int kWeightBytes = sizeof(Input) * kOutputChannelTile * kReductionTile;

  // Persistent task id encodes (token tile, output-channel tile, K chunk).
  if (warp_index == 0) {
    int write_stage = 0;
    int empty_wait_phase = 1;
    int task_id = int(blockIdx.x);

    while (task_id < compute_task_count) {
      ComputeTask const task =
          decode_compute_task<SplitK>(task_id, token_tile_count, output_tile_count);

      for (int reduction_tile = task.k_chunk; reduction_tile < reduction_tile_count;
           reduction_tile += SplitK) {
        if (elected_lane) {
          wait_barrier(storage.stage_empty[write_stage], empty_wait_phase);

          set_barrier_transaction_bytes(storage.low_ready[write_stage],
                                        kActivationBytes + kWeightBytes);
          copy(tma_activation.with(storage.low_ready[write_stage]),
               global_to_tma_activation(_, task.token_tile, reduction_tile),
               tma_to_shared_activation(_, write_stage));
          copy(tma_weight_low.with(storage.low_ready[write_stage]),
               global_to_tma_weight_low(_, task.output_tile, reduction_tile),
               tma_to_shared_weight_low(_, write_stage));

          set_barrier_transaction_bytes(storage.high_ready[write_stage], kWeightBytes);
          copy(tma_weight_high.with(storage.high_ready[write_stage]),
               global_to_tma_weight_high(_, task.output_tile, reduction_tile),
               tma_to_shared_weight_high(_, write_stage));
        }

        ++write_stage;
        if (write_stage == kMainloopStageCount) {
          write_stage = 0;
          empty_wait_phase ^= 1;
        }
      }
      task_id += int(gridDim.x);
    }
  } else if (warp_index == 1) {
    int read_stage = 0;
    int ready_wait_phase = 0;
    int accumulator_empty_wait_phase = 1;
    int task_id = int(blockIdx.x);

    while (task_id < compute_task_count) {
      ComputeTask const task =
          decode_compute_task<SplitK>(task_id, token_tile_count, output_tile_count);
      wait_barrier(storage.accumulator_empty, accumulator_empty_wait_phase);
      accumulator_empty_wait_phase ^= 1;

      TiledMma mma_low;
      TiledMma mma_high;
      mma_low.accumulate_ = UMMA::ScaleOut::Zero;
      mma_high.accumulate_ = UMMA::ScaleOut::Zero;

      int last_stage = read_stage;
      int last_stage_completion_phase = 0;

      for (int reduction_tile = task.k_chunk; reduction_tile < reduction_tile_count;
           reduction_tile += SplitK) {
        wait_barrier(storage.low_ready[read_stage], ready_wait_phase);

#pragma unroll
        for (int reduction_atom = 0; reduction_atom < size<2>(descriptor_activation);
             ++reduction_atom) {
          cute::gemm(mma_low, descriptor_weight_low(_, _, reduction_atom, read_stage),
                     descriptor_activation(_, _, reduction_atom, read_stage), accumulator_low);
          mma_low.accumulate_ = UMMA::ScaleOut::One;
        }

        wait_barrier(storage.high_ready[read_stage], ready_wait_phase);

#pragma unroll
        for (int reduction_atom = 0; reduction_atom < size<2>(descriptor_activation);
             ++reduction_atom) {
          cute::gemm(mma_high, descriptor_weight_high(_, _, reduction_atom, read_stage),
                     descriptor_activation(_, _, reduction_atom, read_stage), accumulator_high);
          mma_high.accumulate_ = UMMA::ScaleOut::One;
        }

        last_stage = read_stage;
        last_stage_completion_phase = ready_wait_phase;
        cutlass::arch::umma_arrive(&storage.stage_empty[read_stage]);

        ++read_stage;
        if (read_stage == kMainloopStageCount) {
          read_stage = 0;
          ready_wait_phase ^= 1;
        }
      }

      wait_barrier(storage.stage_empty[last_stage], last_stage_completion_phase);
      if (elected_lane) {
        arrive_barrier(storage.accumulator_ready);
      }
      task_id += int(gridDim.x);
    }
  } else {
    int const epilogue_thread = int(threadIdx.x) - 64;
    int const tmem_thread = int(threadIdx.x) % kEpilogueThreads;
    auto tmem_load_op =
        TMEM::op_repeater<SM100_TMEM_LOAD_16dp256b1x, kTokenTile * sizeof_bits_v<float>>();
    auto tmem_to_register = make_tmem_copy(tmem_load_op, accumulator_low);
    auto epilogue_slice = tmem_to_register.get_slice(tmem_thread);
    auto tmem_low_for_thread = epilogue_slice.partition_S(accumulator_low);
    auto tmem_high_for_thread = epilogue_slice.partition_S(accumulator_high);

    int accumulator_ready_wait_phase = 0;
    int task_id = int(blockIdx.x);
    bool has_previous_store = false;
    int previous_base_tile = -1;

    while (task_id < compute_task_count) {
      ComputeTask const task =
          decode_compute_task<SplitK>(task_id, token_tile_count, output_tile_count);

      if (has_previous_store) {
        if (epilogue_thread == 0) {
          tma_store_wait<0>();
          atomicAdd(tile_counters + previous_base_tile, 1);
        }
        cutlass::arch::NamedBarrier::sync(kEpilogueThreads,
                                          cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      }

      wait_barrier(storage.accumulator_ready, accumulator_ready_wait_phase);
      accumulator_ready_wait_phase ^= 1;

      auto partial_shape_for_thread = epilogue_slice.partition_D(
          partitioned_partial_shape(_, _, _, task.output_tile, task.token_tile));
      auto shared_partial_for_thread =
          epilogue_slice.partition_D(partitioned_shared_partial_output);
      auto register_low = make_tensor<float>(shape(partial_shape_for_thread));
      auto register_high = make_tensor<float>(shape(partial_shape_for_thread));

      copy(tmem_to_register, tmem_low_for_thread, register_low);
      copy(tmem_to_register, tmem_high_for_thread, register_high);
      cutlass::arch::fence_view_async_tmem_load();

#pragma unroll
      for (int i = 0; i < size(register_low); ++i) {
        register_high(i) = fmaf(register_low(i), kLowScale, register_high(i));
      }
      copy_aligned(register_high, shared_partial_for_thread);

      cutlass::arch::NamedBarrier::sync(kEpilogueThreads,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      tma_store_fence();
      if (epilogue_thread == 0) {
        copy(tma_partial_output, tma_shared_partial_output(_, 0, 0),
             tma_global_partial_output(_, task.output_tile, task.token_tile, task.k_chunk));
        tma_store_arrive();
        arrive_barrier(storage.accumulator_empty);
      }
      has_previous_store = true;
      previous_base_tile = task.base_tile;
      task_id += int(gridDim.x);
    }

    if (has_previous_store) {
      if (epilogue_thread == 0) {
        tma_store_wait<0>();
        __threadfence();
        atomicAdd(tile_counters + previous_base_tile, 1);
      }
      cutlass::arch::NamedBarrier::sync(kEpilogueThreads,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    }
  }

  // A CTA switches to reduction after it exhausts its persistent compute
  // tasks. A tile-local counter is the only dependency between producers and
  // its reducer; no grid-wide barrier or second kernel is used.
  if (warp_index >= 2) {
    int const epilogue_thread = int(threadIdx.x) - 64;
    int reduction_task = int(blockIdx.x);
    while (reduction_task < base_tile_count) {
      if (epilogue_thread == 0) {
        volatile int* counter = tile_counters + reduction_task;
        while (*counter != SplitK) {
          // Grid is capped at one CTA per SM, so producers can make progress.
        }
      }
      cutlass::arch::NamedBarrier::sync(kEpilogueThreads,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

      int const token_tile = reduction_task / output_tile_count;
      int const output_tile = reduction_task - token_tile * output_tile_count;
      constexpr int kVectorsPerToken = kOutputChannelTile / 4;
      constexpr int kVectorsPerTile = kTokenTile * kVectorsPerToken;
      for (int vector_index = epilogue_thread; vector_index < kVectorsPerTile;
           vector_index += kEpilogueThreads) {
        int const local_token = vector_index / kVectorsPerToken;
        int const local_output_vector = vector_index - local_token * kVectorsPerToken;
        int const token = token_tile * kTokenTile + local_token;
        int const output_channel = output_tile * kOutputChannelTile + local_output_vector * 4;
        if (token < token_count && output_channel < output_channel_count) {
          std::size_t const partial_matrix_offset =
              std::size_t(token) * padded_output_channel_count + output_channel;
          float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
          for (int k_chunk = 0; k_chunk < SplitK; ++k_chunk) {
            std::size_t const partial_offset =
                std::size_t(k_chunk) * padded_token_count * padded_output_channel_count +
                partial_matrix_offset;
            float4 const value = *reinterpret_cast<float4 const*>(partial_output + partial_offset);
            sum.x += value.x;
            sum.y += value.y;
            sum.z += value.z;
            sum.w += value.w;
          }

          if constexpr (std::is_same_v<Output, float>) {
            if ((output_channel_count % 4) == 0 && output_channel + 3 < output_channel_count) {
              std::size_t const output_offset =
                  std::size_t(token) * output_channel_count + output_channel;
              *reinterpret_cast<float4*>(output + output_offset) = sum;
            } else {
              float const values[4] = {sum.x, sum.y, sum.z, sum.w};
#pragma unroll
              for (int lane = 0; lane < 4; ++lane) {
                if (output_channel + lane < output_channel_count) {
                  output[std::size_t(token) * output_channel_count + output_channel + lane] =
                      values[lane];
                }
              }
            }
          } else if ((output_channel_count % 4) == 0 && output_channel + 3 < output_channel_count) {
            std::size_t const output_offset =
                std::size_t(token) * output_channel_count + output_channel;
            BFloat16x4 converted{{Input(sum.x), Input(sum.y), Input(sum.z), Input(sum.w)}};
            *reinterpret_cast<BFloat16x4*>(output + output_offset) = converted;
          } else {
            float const values[4] = {sum.x, sum.y, sum.z, sum.w};
#pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
              if (output_channel + lane < output_channel_count) {
                output[std::size_t(token) * output_channel_count + output_channel + lane] =
                    Output(values[lane]);
              }
            }
          }
        }
      }

      cutlass::arch::NamedBarrier::sync(kEpilogueThreads,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      if (epilogue_thread == 0) {
        tile_counters[reduction_task] = 0;
      }
      cutlass::arch::NamedBarrier::sync(kEpilogueThreads,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      reduction_task += int(gridDim.x);
    }
  }

  __syncthreads();
  if (warp_index == 1) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
}

template <class Output, int TokenTile, int StageCount, int SplitK>
cudaError_t launch_impl(Arguments const& args, int multiprocessor_count, cudaStream_t stream) {
  static_assert(std::is_same_v<Output, float> || std::is_same_v<Output, Input>);
  using namespace cute;
  using Traits = KernelTraits<TokenTile, StageCount>;
  using TiledMma = typename Traits::TiledMma;
  using TileShape = typename Traits::TileShape;
  using WeightSmemLayout = typename Traits::WeightSmemLayout;
  using ActivationSmemLayout = typename Traits::ActivationSmemLayout;
  using OutputSmemLayout = typename Traits::OutputSmemLayout;

  static_assert(cosize_v<WeightSmemLayout> == kOutputChannelTile * kReductionTile * StageCount);
  static_assert(cosize_v<ActivationSmemLayout> == TokenTile * kReductionTile * StageCount);
  static_assert(Traits::kSharedMemoryBytes <= cutlass::arch::sm100_smem_capacity_bytes,
                "SM100 shared-memory capacity exceeded");

  auto weight_high = make_tensor(make_gmem_ptr(args.weight_high),
                                 make_shape(args.output_channel_count, args.reduction_size),
                                 make_stride(args.reduction_size, Int<1>{}));
  auto weight_low = make_tensor(make_gmem_ptr(args.weight_low),
                                make_shape(args.output_channel_count, args.reduction_size),
                                make_stride(args.reduction_size, Int<1>{}));
  auto activation =
      make_tensor(make_gmem_ptr(args.activation), make_shape(args.token_count, args.reduction_size),
                  make_stride(args.reduction_size, Int<1>{}));
  int const output_tiles =
      (args.output_channel_count + kOutputChannelTile - 1) / kOutputChannelTile;
  int const token_tiles = (args.token_count + TokenTile - 1) / TokenTile;
  int const padded_output_channel_count = output_tiles * kOutputChannelTile;
  int const padded_token_count = token_tiles * TokenTile;
  auto partial_output =
      make_tensor(make_gmem_ptr(args.partial_output),
                  make_shape(padded_output_channel_count, padded_token_count, Int<SplitK>{}),
                  make_stride(Int<1>{}, padded_output_channel_count,
                              padded_output_channel_count * padded_token_count));

  auto cluster_layout =
      tiled_divide(make_layout(Shape<_1, _1, _1>{}), make_tile(typename TiledMma::AtomThrID{}));

  auto tma_weight_high =
      make_tma_atom_A_sm100(SM90_TMA_LOAD{}, weight_high, WeightSmemLayout{}(_, _, _, Int<0>{}),
                            TileShape{}, TiledMma{}, cluster_layout);
  auto tma_weight_low =
      make_tma_atom_A_sm100(SM90_TMA_LOAD{}, weight_low, WeightSmemLayout{}(_, _, _, Int<0>{}),
                            TileShape{}, TiledMma{}, cluster_layout);
  auto tma_activation =
      make_tma_atom_B_sm100(SM90_TMA_LOAD{}, activation, ActivationSmemLayout{}(_, _, _, Int<0>{}),
                            TileShape{}, TiledMma{}, cluster_layout);
  auto tma_partial_output = make_tma_copy(SM90_TMA_STORE{}, partial_output, OutputSmemLayout{});

  auto kernel_ptr = &kernel<Output, SplitK, Traits, decltype(tma_weight_high),
                            decltype(tma_activation), decltype(tma_partial_output)>;
  cudaError_t status = cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            Traits::kSharedMemoryBytes);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  if (status != cudaSuccess) {
    return status;
  }
  int const compute_tasks = output_tiles * token_tiles * SplitK;
  int const grid_size = compute_tasks < multiprocessor_count ? compute_tasks : multiprocessor_count;
  kernel_ptr<<<grid_size, kThreadCount, Traits::kSharedMemoryBytes, stream>>>(
      tma_weight_high, tma_weight_low, tma_activation, tma_partial_output,
      reinterpret_cast<Output*>(args.output), args.partial_output, args.tile_counters,
      args.token_count, args.output_channel_count, args.reduction_size);
  return cudaGetLastError();
}

}  // namespace detail

inline KernelConfig select_kernel_config(int token_count, int output_channel_count,
                                         int reduction_size, int multiprocessor_count,
                                         int requested_split_k = 0, int requested_stage_count = 0) {
  KernelConfig config{};
  if (token_count <= 0 || token_count > kMaximumTokenCount || output_channel_count <= 0 ||
      reduction_size <= 0 || (reduction_size % kReductionTile) != 0 || multiprocessor_count <= 0) {
    return config;
  }

  config.output_tile_count = (output_channel_count + kOutputChannelTile - 1) / kOutputChannelTile;
  config.token_tile = token_count < kTokenTileSwitch ? kSmallTokenTile : kLargeTokenTile;
  config.token_tile_count = (token_count + config.token_tile - 1) / config.token_tile;
  config.base_tile_count = config.output_tile_count * config.token_tile_count;

  int split_k = requested_split_k;
  if (split_k == 0) {
    int const reduction_tile_count = reduction_size / kReductionTile;
    constexpr int candidates[] = {2, 4, 8};

    // Prefer the largest task grid that approaches the SM count from below.
    for (int candidate : candidates) {
      if (candidate <= reduction_tile_count &&
          config.base_tile_count * candidate <= multiprocessor_count) {
        split_k = candidate;
      }
    }

    // If every valid candidate exceeds the SM count, choose the smallest one.
    if (split_k == 0) {
      for (int candidate : candidates) {
        if (candidate <= reduction_tile_count) {
          split_k = candidate;
          break;
        }
      }
    }
  }

  int const reduction_tile_count = reduction_size / kReductionTile;
  if ((split_k != 2 && split_k != 4 && split_k != 8) || split_k > reduction_tile_count) {
    return KernelConfig{};
  }

  config.split_k = split_k;
  config.stage_count = requested_stage_count == 0 ? kDefaultStageCount : requested_stage_count;
  if (config.stage_count < kMinimumStageCount || config.stage_count > kMaximumStageCount) {
    return KernelConfig{};
  }
  config.compute_task_count = config.base_tile_count * split_k;
  config.grid_size = config.compute_task_count < multiprocessor_count ? config.compute_task_count
                                                                      : multiprocessor_count;
  auto shared_memory_bytes = [&](auto token_tile_tag) {
    constexpr int TokenTile = decltype(token_tile_tag)::value;
    switch (config.stage_count) {
      case 3:
        return detail::KernelTraits<TokenTile, 3>::kSharedMemoryBytes;
      case 4:
        return detail::KernelTraits<TokenTile, 4>::kSharedMemoryBytes;
      case 5:
        return detail::KernelTraits<TokenTile, 5>::kSharedMemoryBytes;
      case 6:
        return detail::KernelTraits<TokenTile, 6>::kSharedMemoryBytes;
      default:
        return 0;
    }
  };
  config.shared_memory_bytes = config.token_tile == kSmallTokenTile
                                   ? shared_memory_bytes(cute::Int<kSmallTokenTile>{})
                                   : shared_memory_bytes(cute::Int<kLargeTokenTile>{});
  return config;
}

inline std::size_t partial_workspace_bytes(KernelConfig const& config, int token_count,
                                           int output_channel_count) {
  int const padded_token_count = config.token_tile_count * config.token_tile;
  int const padded_output_channel_count = config.output_tile_count * kOutputChannelTile;
  return std::size_t(config.split_k) * padded_token_count * padded_output_channel_count *
         sizeof(float);
}

inline std::size_t counter_workspace_bytes(KernelConfig const& config) {
  return std::size_t(config.base_tile_count) * sizeof(int);
}

template <class Output>
cudaError_t launch_typed(Arguments const& args, int requested_split_k, int requested_stage_count,
                         cudaStream_t stream) {
  static_assert(std::is_same_v<Output, float> || std::is_same_v<Output, Input>);
  if (args.output == nullptr || args.activation == nullptr || args.weight_high == nullptr ||
      args.weight_low == nullptr || args.partial_output == nullptr ||
      args.tile_counters == nullptr || args.token_count <= 0 ||
      args.token_count > kMaximumTokenCount || args.output_channel_count <= 0 ||
      args.reduction_size <= 0 || (args.reduction_size % kReductionTile) != 0) {
    return cudaErrorInvalidValue;
  }

  int device = 0;
  int multiprocessor_count = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) {
    return status;
  }

  KernelConfig config =
      select_kernel_config(args.token_count, args.output_channel_count, args.reduction_size,
                           multiprocessor_count, requested_split_k, requested_stage_count);

  auto launch_selected = [&](auto token_tile_tag, auto split_tag) {
    constexpr int TokenTile = decltype(token_tile_tag)::value;
    constexpr int SplitK = decltype(split_tag)::value;
    switch (config.stage_count) {
      case 3:
        return detail::launch_impl<Output, TokenTile, 3, SplitK>(args, multiprocessor_count,
                                                                 stream);
      case 4:
        return detail::launch_impl<Output, TokenTile, 4, SplitK>(args, multiprocessor_count,
                                                                 stream);
      case 5:
        return detail::launch_impl<Output, TokenTile, 5, SplitK>(args, multiprocessor_count,
                                                                 stream);
      case 6:
        return detail::launch_impl<Output, TokenTile, 6, SplitK>(args, multiprocessor_count,
                                                                 stream);
      default:
        return cudaErrorInvalidValue;
    }
  };

  if (config.token_tile == kSmallTokenTile) {
    switch (config.split_k) {
      case 2:
        return launch_selected(cute::Int<kSmallTokenTile>{}, cute::Int<2>{});
      case 4:
        return launch_selected(cute::Int<kSmallTokenTile>{}, cute::Int<4>{});
      case 8:
        return launch_selected(cute::Int<kSmallTokenTile>{}, cute::Int<8>{});
      default:
        return cudaErrorInvalidValue;
    }
  }
  if (config.token_tile == kLargeTokenTile) {
    switch (config.split_k) {
      case 2:
        return launch_selected(cute::Int<kLargeTokenTile>{}, cute::Int<2>{});
      case 4:
        return launch_selected(cute::Int<kLargeTokenTile>{}, cute::Int<4>{});
      case 8:
        return launch_selected(cute::Int<kLargeTokenTile>{}, cute::Int<8>{});
      default:
        return cudaErrorInvalidValue;
    }
  }
  return cudaErrorInvalidValue;
}

inline cudaError_t launch(Arguments const& args, int requested_split_k = 0,
                          int requested_stage_count = 0, cudaStream_t stream = nullptr) {
  if (args.output_type == OutputType::kFloat32) {
    return launch_typed<float>(args, requested_split_k, requested_stage_count, stream);
  }
  if (args.output_type == OutputType::kBFloat16) {
    return launch_typed<Input>(args, requested_split_k, requested_stage_count, stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace flashinfer::gemm::dual_bf16_weight::split_k
