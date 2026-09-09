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

#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <type_traits>

namespace flashinfer::gemm::dual_bf16_weight::one_sm {

using Input = cutlass::bfloat16_t;

enum class OutputType {
  kFloat32,
  kBFloat16,
};

constexpr int kTokenTile = 16;
constexpr int kOutputChannelTile = 64;
constexpr int kReductionTile = 128;
constexpr int kEpilogueThreads = 128;
constexpr int kThreadCount = 32 + 32 + kEpilogueThreads;
constexpr int kTmemColumns = 32;

struct Arguments {
  void* output;
  Input const* activation;
  Input const* weight_high;
  Input const* weight_low;
  int token_count;
  int output_channel_count;
  int reduction_size;
  float low_scale;
  OutputType output_type = OutputType::kFloat32;
};

namespace detail {

template <class WeightSmemLayout, class ActivationSmemLayout, int StageCount>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<Input, cute::cosize_v<WeightSmemLayout>> weight_low;
  alignas(128) cute::ArrayEngine<Input, cute::cosize_v<WeightSmemLayout>> weight_high;
  alignas(128) cute::ArrayEngine<Input, cute::cosize_v<ActivationSmemLayout>> activation;

  // The low-ready barrier tracks activation + low-weight TMA transactions.
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
};

template <class Output, int StageCount, class TiledMma, class TmaWeight, class TmaActivation,
          class WeightSmemLayout, class ActivationSmemLayout>
__global__ __launch_bounds__(kThreadCount,
                             1) void kernel(CUTE_GRID_CONSTANT TmaWeight const tma_weight_high,
                                            CUTE_GRID_CONSTANT TmaWeight const tma_weight_low,
                                            CUTE_GRID_CONSTANT TmaActivation const tma_activation,
                                            Output* output, int token_count,
                                            int output_channel_count, int reduction_size,
                                            float low_scale) {
  using namespace cute;
  using X = Underscore;
  using Storage = SharedStorage<WeightSmemLayout, ActivationSmemLayout, StageCount>;

  extern __shared__ char shared_memory[];
  Storage& storage = *reinterpret_cast<Storage*>(shared_memory);

  int const warp_index = int(threadIdx.x) / 32;
  bool const elected_lane = cute::elect_one_sync();

  // One full warp must participate uniformly in TMEM allocation.
  cute::TMEM::Allocator1Sm tmem_allocator;
  if (warp_index == 1) {
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int stage = 0; stage < StageCount; ++stage) {
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

  auto global_weight_low =
      tma_weight_low.get_tma_tensor(make_shape(output_channel_count, reduction_size));
  auto global_weight_high =
      tma_weight_high.get_tma_tensor(make_shape(output_channel_count, reduction_size));
  auto global_activation = tma_activation.get_tma_tensor(make_shape(token_count, reduction_size));

  auto tiled_weight_low =
      local_tile(global_weight_low, tile_shape, make_coord(_, _, _), Step<_1, X, _1>{});
  auto tiled_weight_high =
      local_tile(global_weight_high, tile_shape, make_coord(_, _, _), Step<_1, X, _1>{});
  auto tiled_activation =
      local_tile(global_activation, tile_shape, make_coord(_, _, _), Step<X, _1, _1>{});

  auto partitioned_weight_low = cta_mma.partition_A(tiled_weight_low);
  auto partitioned_weight_high = cta_mma.partition_A(tiled_weight_high);
  auto partitioned_activation = cta_mma.partition_B(tiled_activation);

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

  auto output_tensor =
      make_tensor(make_gmem_ptr(output), make_shape(output_channel_count, token_count),
                  make_stride(Int<1>{}, output_channel_count));
  auto tiled_output = local_tile(
      output_tensor, make_shape(Int<kOutputChannelTile>{}, Int<kTokenTile>{}), make_coord(_, _));
  auto partitioned_output = cta_mma.partition_C(tiled_output);

  // Partition logical output coordinates exactly like the output tensor.  The
  // epilogue uses the token coordinate to mask stores from the last partial
  // token tile while TMA zero-fills its out-of-bounds activation rows.
  auto output_coordinates = make_identity_tensor(make_shape(output_channel_count, token_count));
  auto tiled_output_coordinates =
      local_tile(output_coordinates, make_shape(Int<kOutputChannelTile>{}, Int<kTokenTile>{}),
                 make_coord(_, _));
  auto partitioned_output_coordinates = cta_mma.partition_C(tiled_output_coordinates);

  auto accumulator_low = cta_mma.make_fragment_C(partitioned_output(_, _, _, Int<0>{}, Int<0>{}));
  auto accumulator_high = cta_mma.make_fragment_C(partitioned_output(_, _, _, Int<0>{}, Int<0>{}));
  accumulator_low.data() = storage.tmem_base_ptr;
  accumulator_high.data() = storage.tmem_base_ptr + kTokenTile;

  int const output_tile_count = output_channel_count / kOutputChannelTile;
  int const token_tile_count = (token_count + kTokenTile - 1) / kTokenTile;
  int const reduction_tile_count = reduction_size / kReductionTile;
  int const total_tile_count = output_tile_count * token_tile_count;

  constexpr int kActivationBytes = sizeof(Input) * kTokenTile * kReductionTile;
  constexpr int kWeightBytes = sizeof(Input) * kOutputChannelTile * kReductionTile;

  if (warp_index == 0) {
    int write_stage = 0;
    int empty_wait_phase = 1;
    int tile_id = int(blockIdx.x);

    while (tile_id < total_tile_count) {
      int const token_tile = tile_id / output_tile_count;
      int const output_tile = tile_id - token_tile * output_tile_count;

      for (int reduction_tile = 0; reduction_tile < reduction_tile_count; ++reduction_tile) {
        if (elected_lane) {
          wait_barrier(storage.stage_empty[write_stage], empty_wait_phase);

          set_barrier_transaction_bytes(storage.low_ready[write_stage],
                                        kActivationBytes + kWeightBytes);
          copy(tma_activation.with(storage.low_ready[write_stage]),
               global_to_tma_activation(_, token_tile, reduction_tile),
               tma_to_shared_activation(_, write_stage));
          copy(tma_weight_low.with(storage.low_ready[write_stage]),
               global_to_tma_weight_low(_, output_tile, reduction_tile),
               tma_to_shared_weight_low(_, write_stage));

          set_barrier_transaction_bytes(storage.high_ready[write_stage], kWeightBytes);
          copy(tma_weight_high.with(storage.high_ready[write_stage]),
               global_to_tma_weight_high(_, output_tile, reduction_tile),
               tma_to_shared_weight_high(_, write_stage));
        }

        ++write_stage;
        if (write_stage == StageCount) {
          write_stage = 0;
          empty_wait_phase ^= 1;
        }
      }
      tile_id += int(gridDim.x);
    }
  } else if (warp_index == 1) {
    int read_stage = 0;
    int ready_wait_phase = 0;
    int accumulator_empty_wait_phase = 1;
    int tile_id = int(blockIdx.x);

    while (tile_id < total_tile_count) {
      wait_barrier(storage.accumulator_empty, accumulator_empty_wait_phase);
      accumulator_empty_wait_phase ^= 1;

      TiledMma mma_low;
      TiledMma mma_high;
      mma_low.accumulate_ = UMMA::ScaleOut::Zero;
      mma_high.accumulate_ = UMMA::ScaleOut::Zero;

      int last_stage = read_stage;
      int last_stage_completion_phase = 0;

      for (int reduction_tile = 0; reduction_tile < reduction_tile_count; ++reduction_tile) {
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
        if (read_stage == StageCount) {
          read_stage = 0;
          ready_wait_phase ^= 1;
        }
      }

      // The completion of the last committed group also orders earlier UMMA groups.
      wait_barrier(storage.stage_empty[last_stage], last_stage_completion_phase);
      if (elected_lane) {
        arrive_barrier(storage.accumulator_ready);
      }

      tile_id += int(gridDim.x);
    }
  } else {
    int const epilogue_thread = int(threadIdx.x) - 64;
    int const tmem_thread = int(threadIdx.x) % kEpilogueThreads;
    // M=64 gives each epilogue warp a 16-datapath TMEM subpartition.  The
    // 16dp256b2x variant covers all 16 FP32 token columns per subpartition.
    auto tmem_load_op =
        TMEM::op_repeater<SM100_TMEM_LOAD_16dp256b1x, kTokenTile * sizeof_bits_v<float>>();
    auto tmem_to_register = make_tmem_copy(tmem_load_op, accumulator_low);
    // TMEM datapath ownership follows the physical warp id modulo four.
    // These epilogue warps are physical warps 2..5, hence subtracting 64
    // would rotate the TMEM subpartitions by two warps.
    auto epilogue_slice = tmem_to_register.get_slice(tmem_thread);
    auto tmem_low_for_thread = epilogue_slice.partition_S(accumulator_low);
    auto tmem_high_for_thread = epilogue_slice.partition_S(accumulator_high);

    int accumulator_ready_wait_phase = 0;
    int tile_id = int(blockIdx.x);

    while (tile_id < total_tile_count) {
      int const token_tile = tile_id / output_tile_count;
      int const output_tile = tile_id - token_tile * output_tile_count;

      wait_barrier(storage.accumulator_ready, accumulator_ready_wait_phase);
      accumulator_ready_wait_phase ^= 1;

      auto output_for_thread =
          epilogue_slice.partition_D(partitioned_output(_, _, _, output_tile, token_tile));
      auto output_coordinates_for_thread = epilogue_slice.partition_D(
          partitioned_output_coordinates(_, _, _, output_tile, token_tile));
      auto register_low = make_tensor<float>(shape(output_for_thread));
      auto register_high = make_tensor<float>(shape(output_for_thread));

      copy(tmem_to_register, tmem_low_for_thread, register_low);
      copy(tmem_to_register, tmem_high_for_thread, register_high);
      cutlass::arch::fence_view_async_tmem_load();

#pragma unroll
      for (int i = 0; i < size(register_low); ++i) {
        register_high(i) = fmaf(register_low(i), low_scale, register_high(i));
        if (get<1>(output_coordinates_for_thread(i)) < token_count) {
          output_for_thread(i) = Output(register_high(i));
        }
      }

      // All four epilogue warps must finish reading the shared TMEM pair
      // before the MMA warp is allowed to reuse it for the next persistent
      // output tile.
      cutlass::arch::NamedBarrier::sync(kEpilogueThreads,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      if (epilogue_thread == 0) {
        arrive_barrier(storage.accumulator_empty);
      }
      tile_id += int(gridDim.x);
    }
  }

  __syncthreads();
  if (warp_index == 1) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
}

template <class Output, int StageCount>
cudaError_t launch_impl(Arguments const& args, cudaStream_t stream) {
  static_assert(std::is_same_v<Output, float> || std::is_same_v<Output, Input>);
  using namespace cute;

  using MmaAtom = SM100_MMA_F16BF16_SS<Input, Input, float, kOutputChannelTile, kTokenTile,
                                       UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(MmaAtom{}));
  using TileShape = Shape<Int<kOutputChannelTile>, Int<kTokenTile>, Int<kReductionTile>>;

  using WeightMmaShape = decltype(partition_shape_A(
      TiledMma{}, make_shape(Int<kOutputChannelTile>{}, Int<kReductionTile>{})));
  using ActivationMmaShape =
      decltype(partition_shape_B(TiledMma{}, make_shape(Int<kTokenTile>{}, Int<kReductionTile>{})));

  using WeightSmemLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Input>{}, append(WeightMmaShape{}, Int<StageCount>{}),
      Step<_1, _2, _3>{}));
  using ActivationSmemLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Input>{}, append(ActivationMmaShape{}, Int<StageCount>{}),
      Step<_1, _2, _3>{}));

  static_assert(cosize_v<WeightSmemLayout> == kOutputChannelTile * kReductionTile * StageCount);
  static_assert(cosize_v<ActivationSmemLayout> == kTokenTile * kReductionTile * StageCount);

  auto weight_high = make_tensor(make_gmem_ptr(args.weight_high),
                                 make_shape(args.output_channel_count, args.reduction_size),
                                 make_stride(args.reduction_size, Int<1>{}));
  auto weight_low = make_tensor(make_gmem_ptr(args.weight_low),
                                make_shape(args.output_channel_count, args.reduction_size),
                                make_stride(args.reduction_size, Int<1>{}));
  auto activation =
      make_tensor(make_gmem_ptr(args.activation), make_shape(args.token_count, args.reduction_size),
                  make_stride(args.reduction_size, Int<1>{}));

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

  using Storage = SharedStorage<WeightSmemLayout, ActivationSmemLayout, StageCount>;
  constexpr int shared_memory_bytes = sizeof(Storage);
  static_assert(shared_memory_bytes <= cutlass::arch::sm100_smem_capacity_bytes,
                "SM100 shared-memory capacity exceeded");

  auto kernel_ptr = &kernel<Output, StageCount, TiledMma, decltype(tma_weight_high),
                            decltype(tma_activation), WeightSmemLayout, ActivationSmemLayout>;

  cudaError_t status = cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            shared_memory_bytes);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  if (status != cudaSuccess) {
    return status;
  }

  int device = 0;
  int multiprocessor_count = 0;
  status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) {
    return status;
  }

  int const output_tiles = args.output_channel_count / kOutputChannelTile;
  int const token_tiles = (args.token_count + kTokenTile - 1) / kTokenTile;
  int const total_tiles = output_tiles * token_tiles;
  int const grid_size = total_tiles < multiprocessor_count ? total_tiles : multiprocessor_count;

  kernel_ptr<<<grid_size, kThreadCount, shared_memory_bytes, stream>>>(
      tma_weight_high, tma_weight_low, tma_activation, reinterpret_cast<Output*>(args.output),
      args.token_count, args.output_channel_count, args.reduction_size, args.low_scale);
  return cudaGetLastError();
}

}  // namespace detail

template <class Output>
cudaError_t launch_typed(Arguments const& args, int stage_count, cudaStream_t stream) {
  static_assert(std::is_same_v<Output, float> || std::is_same_v<Output, Input>);

  switch (stage_count) {
    case 3:
      return detail::launch_impl<Output, 3>(args, stream);
    case 4:
      return detail::launch_impl<Output, 4>(args, stream);
    case 5:
      return detail::launch_impl<Output, 5>(args, stream);
    case 6:
      return detail::launch_impl<Output, 6>(args, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

inline cudaError_t launch(Arguments const& args, int stage_count = 6,
                          cudaStream_t stream = nullptr) {
  if (args.output == nullptr || args.activation == nullptr || args.weight_high == nullptr ||
      args.weight_low == nullptr || args.token_count <= 0 || args.output_channel_count <= 0 ||
      args.reduction_size <= 0) {
    return cudaErrorInvalidValue;
  }
  if ((args.output_channel_count % kOutputChannelTile) != 0 ||
      (args.reduction_size % kReductionTile) != 0) {
    return cudaErrorInvalidValue;
  }

  if (args.output_type == OutputType::kFloat32) {
    return launch_typed<float>(args, stage_count, stream);
  }
  if (args.output_type == OutputType::kBFloat16) {
    return launch_typed<Input>(args, stage_count, stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace flashinfer::gemm::dual_bf16_weight::one_sm
