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
#include <cutlass/cluster_launch.hpp>
#include <type_traits>

namespace flashinfer::gemm::dual_bf16_weight::two_sm {

using Input = cutlass::bfloat16_t;

enum class OutputType {
  kFloat32,
  kBFloat16,
};

constexpr int kReductionTile = 128;
constexpr int kLoadWarpThreads = 32;
constexpr int kUmmaWarpThreads = 32;
constexpr int kEpilogueThreads = 128;
constexpr int kThreadCount = kLoadWarpThreads + kUmmaWarpThreads + kEpilogueThreads;
constexpr float kLowScale = 1.0f / 256.0f;

static_assert(kThreadCount == 192);

struct Arguments {
  void* output;
  Input const* activation;
  Input const* weight_high;
  Input const* weight_low;
  int token_count;
  int output_channel_count;
  int reduction_size;
  OutputType output_type = OutputType::kFloat32;
};

struct KernelConfig {
  int output_channel_tile;
  int token_tile;
  int reduction_tile;
  int stage_count;
  int shared_memory_bytes;
};

namespace detail {

template <class WeightSmemLayout, class ActivationSmemLayout, int StageCount>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<Input, cute::cosize_v<WeightSmemLayout>> weight_low;
  alignas(128) cute::ArrayEngine<Input, cute::cosize_v<WeightSmemLayout>> weight_high;
  alignas(128) cute::ArrayEngine<Input, cute::cosize_v<ActivationSmemLayout>> activation;

  // low_ready covers activation + low weight. high_ready covers high weight.
  // stage_empty is completed by UMMA, except for the final K tile of an
  // output tile, where the epilogue completes it after the TMEM reads.
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

template <int OutputChannelTile, int TokenTile, int StageCount>
struct KernelTraits {
  static constexpr int kOutputChannelTile = OutputChannelTile;
  static constexpr int kTokenTile = TokenTile;
  static constexpr int kStageCount = StageCount;
  static constexpr int kTmemColumns = TokenTile * 2 < 32 ? 32 : TokenTile * 2;

  static_assert(OutputChannelTile == 128 || OutputChannelTile == 256);
  static_assert(TokenTile == 16 || TokenTile == 32 || TokenTile == 64);
  static_assert(kTmemColumns == 32 || kTmemColumns == 64 || kTmemColumns == 128);

  using MmaAtom =
      cute::SM100_MMA_F16BF16_2x1SM_SS<Input, Input, float, OutputChannelTile, TokenTile,
                                       cute::UMMA::Major::K, cute::UMMA::Major::K>;
  using TiledMma = decltype(cute::make_tiled_mma(MmaAtom{}));
  using TileShape =
      cute::Shape<cute::Int<OutputChannelTile>, cute::Int<TokenTile>, cute::Int<kReductionTile>>;

  using WeightMmaShape = decltype(cute::partition_shape_A(
      TiledMma{}, cute::make_shape(cute::Int<OutputChannelTile>{}, cute::Int<kReductionTile>{})));
  using ActivationMmaShape = decltype(cute::partition_shape_B(
      TiledMma{}, cute::make_shape(cute::Int<TokenTile>{}, cute::Int<kReductionTile>{})));

  using WeightSmemLayout = decltype(cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<Input>{},
      cute::append(WeightMmaShape{}, cute::Int<StageCount>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{}));
  using ActivationSmemLayout = decltype(cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<Input>{},
      cute::append(ActivationMmaShape{}, cute::Int<StageCount>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{}));

  using Storage = SharedStorage<WeightSmemLayout, ActivationSmemLayout, StageCount>;

  static constexpr int kWeightElementsPerStage = (OutputChannelTile / 2) * kReductionTile;
  static constexpr int kActivationElementsPerStage = (TokenTile / 2) * kReductionTile;
  static constexpr int kBytesPerStage =
      int(sizeof(Input)) * (2 * kWeightElementsPerStage + kActivationElementsPerStage);
  static constexpr int kSharedMemoryBytes = sizeof(Storage);

  // A cta_group::2 instruction partitions both operands between its two peer
  // CTAs. K=128 therefore appears as exactly eight K=16 UMMA atoms.
  static_assert(cute::cosize_v<WeightSmemLayout> == kWeightElementsPerStage * StageCount);
  static_assert(cute::cosize_v<ActivationSmemLayout> == kActivationElementsPerStage * StageCount);
  static_assert(cute::size<2>(WeightMmaShape{}) == 8);
  static_assert(cute::size<2>(ActivationMmaShape{}) == 8);
};

template <class Output, class Traits, class TmaWeight, class TmaActivation>
__global__ __launch_bounds__(kThreadCount,
                             1) void kernel(CUTE_GRID_CONSTANT TmaWeight const tma_weight_high,
                                            CUTE_GRID_CONSTANT TmaWeight const tma_weight_low,
                                            CUTE_GRID_CONSTANT TmaActivation const tma_activation,
                                            Output* output, int token_count,
                                            int output_channel_count, int reduction_size) {
  using namespace cute;
  using X = Underscore;
  using Storage = typename Traits::Storage;

  constexpr int kOutputChannelTile = Traits::kOutputChannelTile;
  constexpr int kTokenTile = Traits::kTokenTile;
  constexpr int kStageCount = Traits::kStageCount;

  extern __shared__ char shared_memory[];
  Storage& storage = *reinterpret_cast<Storage*>(shared_memory);

  int const warp_index = int(threadIdx.x) / 32;
  bool const elected_lane = cute::elect_one_sync();

  auto cluster_shape = make_shape(_2{}, _1{}, _1{});
  auto cluster_layout_vmnk =
      tiled_divide(make_layout(cluster_shape), make_tile(typename Traits::TiledMma::AtomThrID{}));
  auto cta_in_cluster_coord_vmnk =
      cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
  auto mma_peer = get<0>(cta_in_cluster_coord_vmnk);
  bool const is_leader_cta = int(mma_peer) == 0;
  int const cluster_index = int(blockIdx.x) / 2;
  int const cluster_count = int(gridDim.x) / 2;

  // Allocator2Sm requires the same fully active warp in both peer CTAs.
  cute::TMEM::Allocator2Sm tmem_allocator;
  if (warp_index == 1) {
    tmem_allocator.allocate(Traits::kTmemColumns, &storage.tmem_base_ptr);
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int stage = 0; stage < kStageCount; ++stage) {
      initialize_barrier(storage.low_ready[stage], 1);
      initialize_barrier(storage.high_ready[stage], 1);
      initialize_barrier(storage.stage_empty[stage], 1);
    }
    initialize_barrier(storage.accumulator_ready, 1);
    // Both peer epilogues remotely arrive on the leader CTA's barrier.
    initialize_barrier(storage.accumulator_empty, 2);
    // Publish mbarrier initialization to the peer CTA before either CTA uses
    // the barriers through cluster-scoped TMA/UMMA operations.
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  cute::cluster_sync();

  typename Traits::TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(mma_peer);
  auto tile_shape = typename Traits::TileShape{};

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

  auto [global_to_tma_weight_low, tma_to_shared_weight_low] = tma_partition(
      tma_weight_low, get<2>(cta_in_cluster_coord_vmnk), make_layout(size<2>(cluster_layout_vmnk)),
      group_modes<0, 3>(shared_weight_low), group_modes<0, 3>(partitioned_weight_low));
  auto [global_to_tma_weight_high, tma_to_shared_weight_high] = tma_partition(
      tma_weight_high, get<2>(cta_in_cluster_coord_vmnk), make_layout(size<2>(cluster_layout_vmnk)),
      group_modes<0, 3>(shared_weight_high), group_modes<0, 3>(partitioned_weight_high));
  auto [global_to_tma_activation, tma_to_shared_activation] = tma_partition(
      tma_activation, get<1>(cta_in_cluster_coord_vmnk), make_layout(size<1>(cluster_layout_vmnk)),
      group_modes<0, 3>(shared_activation), group_modes<0, 3>(partitioned_activation));

  auto descriptor_weight_low = cta_mma.make_fragment_A(shared_weight_low);
  auto descriptor_weight_high = cta_mma.make_fragment_A(shared_weight_high);
  auto descriptor_activation = cta_mma.make_fragment_B(shared_activation);

  auto output_tensor =
      make_tensor(make_gmem_ptr(output), make_shape(output_channel_count, token_count),
                  make_stride(Int<1>{}, output_channel_count));
  auto tiled_output = local_tile(
      output_tensor, make_shape(Int<kOutputChannelTile>{}, Int<kTokenTile>{}), make_coord(_, _));
  auto partitioned_output = cta_mma.partition_C(tiled_output);

  auto output_coordinates = make_identity_tensor(make_shape(output_channel_count, token_count));
  auto tiled_output_coordinates =
      local_tile(output_coordinates, make_shape(Int<kOutputChannelTile>{}, Int<kTokenTile>{}),
                 make_coord(_, _));
  auto partitioned_output_coordinates = cta_mma.partition_C(tiled_output_coordinates);

  auto accumulator_low = cta_mma.make_fragment_C(partitioned_output(_, _, _, Int<0>{}, Int<0>{}));
  auto accumulator_high = cta_mma.make_fragment_C(partitioned_output(_, _, _, Int<0>{}, Int<0>{}));
  accumulator_low.data() = storage.tmem_base_ptr;
  accumulator_high.data() = storage.tmem_base_ptr + kTokenTile;

  int const output_tile_count =
      (output_channel_count + kOutputChannelTile - 1) / kOutputChannelTile;
  int const token_tile_count = (token_count + kTokenTile - 1) / kTokenTile;
  int const reduction_tile_count = reduction_size / kReductionTile;
  int const total_tile_count = output_tile_count * token_tile_count;

  constexpr int kPeerCtaCount = 2;
  constexpr int kLowTransactionBytes =
      kPeerCtaCount * int(sizeof(Input)) *
      (Traits::kWeightElementsPerStage + Traits::kActivationElementsPerStage);
  constexpr int kHighTransactionBytes =
      kPeerCtaCount * int(sizeof(Input)) * Traits::kWeightElementsPerStage;

  uint16_t const tma_weight_mask =
      create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  uint16_t const tma_activation_mask =
      create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  uint16_t const umma_accumulator_mask =
      create_tma_multicast_mask<0, 1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk) |
      create_tma_multicast_mask<0, 2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);

  if (warp_index == 0) {
    // Both peer CTAs issue their TMA.2SM halves. Only the leader programs and
    // consumes the transaction barriers.
    int write_stage = 0;
    int empty_wait_phase = 1;
    int tile_id = cluster_index;

    while (tile_id < total_tile_count) {
      int const token_tile = tile_id / output_tile_count;
      int const output_tile = tile_id - token_tile * output_tile_count;

      for (int reduction_tile = 0; reduction_tile < reduction_tile_count; ++reduction_tile) {
        if (elected_lane) {
          wait_barrier(storage.stage_empty[write_stage], empty_wait_phase);

          if (is_leader_cta) {
            set_barrier_transaction_bytes(storage.low_ready[write_stage], kLowTransactionBytes);
          }
          copy(tma_activation.with(storage.low_ready[write_stage], tma_activation_mask),
               global_to_tma_activation(_, token_tile, reduction_tile),
               tma_to_shared_activation(_, write_stage));
          copy(tma_weight_low.with(storage.low_ready[write_stage], tma_weight_mask),
               global_to_tma_weight_low(_, output_tile, reduction_tile),
               tma_to_shared_weight_low(_, write_stage));

          if (is_leader_cta) {
            set_barrier_transaction_bytes(storage.high_ready[write_stage], kHighTransactionBytes);
          }
          copy(tma_weight_high.with(storage.high_ready[write_stage], tma_weight_mask),
               global_to_tma_weight_high(_, output_tile, reduction_tile),
               tma_to_shared_weight_high(_, write_stage));
        }

        ++write_stage;
        if (write_stage == kStageCount) {
          write_stage = 0;
          empty_wait_phase ^= 1;
        }
      }
      tile_id += cluster_count;
    }
  } else if (warp_index == 1) {
    // tcgen05.mma.cta_group::2 is issued only by the leader CTA. The peer's
    // same-index warp is still required for TMEM allocation/deallocation.
    if (is_leader_cta) {
      int read_stage = 0;
      int ready_wait_phase = 0;
      int accumulator_empty_wait_phase = 1;
      int tile_id = cluster_index;

      while (tile_id < total_tile_count) {
        wait_barrier(storage.accumulator_empty, accumulator_empty_wait_phase);
        accumulator_empty_wait_phase ^= 1;

        typename Traits::TiledMma mma_low;
        typename Traits::TiledMma mma_high;
        mma_low.accumulate_ = UMMA::ScaleOut::Zero;
        mma_high.accumulate_ = UMMA::ScaleOut::Zero;

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

          bool const is_last_reduction_tile = reduction_tile + 1 == reduction_tile_count;
          if (is_last_reduction_tile) {
            // This completion event makes both TMEM partitions visible to the
            // epilogue warps in their respective peer CTAs.
            cutlass::arch::umma_arrive_multicast_2x1SM(&storage.accumulator_ready,
                                                       umma_accumulator_mask);
          } else {
            cutlass::arch::umma_arrive_multicast_2x1SM(&storage.stage_empty[read_stage],
                                                       umma_accumulator_mask);
          }

          ++read_stage;
          if (read_stage == kStageCount) {
            read_stage = 0;
            ready_wait_phase ^= 1;
          }
        }
        tile_id += cluster_count;
      }
    }
  } else {
    int const epilogue_thread = int(threadIdx.x) - 64;
    // TMEM datapath ownership follows the physical warp id modulo four.
    int const tmem_thread = int(threadIdx.x) % kEpilogueThreads;

    // The 2SM accumulator's per-CTA C mode keeps the peer-partitioned N mode
    // nested in the TMEM layout. Let make_tmem_copy repeat the basic 32dp atom
    // over that exact layout instead of forcing a flattened repeated atom.
    auto tmem_to_register = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, accumulator_low);
    auto epilogue_slice = tmem_to_register.get_slice(tmem_thread);
    auto tmem_low_for_thread = epilogue_slice.partition_S(accumulator_low);
    auto tmem_high_for_thread = epilogue_slice.partition_S(accumulator_high);

    int accumulator_ready_wait_phase = 0;
    int next_read_stage = 0;
    int tile_id = cluster_index;

    while (tile_id < total_tile_count) {
      int const token_tile = tile_id / output_tile_count;
      int const output_tile = tile_id - token_tile * output_tile_count;
      int const last_stage = (next_read_stage + reduction_tile_count - 1) % kStageCount;
      next_read_stage = (last_stage + 1) % kStageCount;

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
        register_high(i) = fmaf(register_low(i), kLowScale, register_high(i));
        auto coordinate = output_coordinates_for_thread(i);
        if (get<0>(coordinate) < output_channel_count && get<1>(coordinate) < token_count) {
          output_for_thread(i) = Output(register_high(i));
        }
      }

      cutlass::arch::NamedBarrier::sync(kEpilogueThreads,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      if (epilogue_thread == 0) {
        // The final K stage stays occupied until both accumulators have been
        // read. Then each peer releases its local stage and remotely arrives
        // on the leader's two-participant accumulator barrier.
        arrive_barrier(storage.stage_empty[last_stage]);
        cutlass::arch::ClusterBarrier::arrive(&storage.accumulator_empty, 0, 1);
      }
      tile_id += cluster_count;
    }
  }

  __syncthreads();
  cute::cluster_sync();
  if (warp_index == 1) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(storage.tmem_base_ptr, Traits::kTmemColumns);
  }
}

template <class Output, int OutputChannelTile, int TokenTile, int StageCount>
cudaError_t launch_impl(Arguments const& args, cudaStream_t stream) {
  using namespace cute;
  using Traits = KernelTraits<OutputChannelTile, TokenTile, StageCount>;
  using NextStageTraits = KernelTraits<OutputChannelTile, TokenTile, StageCount + 1>;

  static_assert(Traits::kSharedMemoryBytes <= cutlass::arch::sm100_smem_capacity_bytes,
                "selected stage count exceeds SM100 shared memory");
  static_assert(NextStageTraits::kSharedMemoryBytes > cutlass::arch::sm100_smem_capacity_bytes,
                "selected stage count is not the maximum supported by shared memory");

  auto weight_high = make_tensor(make_gmem_ptr(args.weight_high),
                                 make_shape(args.output_channel_count, args.reduction_size),
                                 make_stride(args.reduction_size, Int<1>{}));
  auto weight_low = make_tensor(make_gmem_ptr(args.weight_low),
                                make_shape(args.output_channel_count, args.reduction_size),
                                make_stride(args.reduction_size, Int<1>{}));
  auto activation =
      make_tensor(make_gmem_ptr(args.activation), make_shape(args.token_count, args.reduction_size),
                  make_stride(args.reduction_size, Int<1>{}));

  auto cluster_shape = make_shape(_2{}, _1{}, _1{});
  auto cluster_layout_vmnk =
      tiled_divide(make_layout(cluster_shape), make_tile(typename Traits::TiledMma::AtomThrID{}));

  auto tma_weight_high = make_tma_atom_A_sm100(
      SM100_TMA_2SM_LOAD_MULTICAST{}, weight_high,
      typename Traits::WeightSmemLayout{}(_, _, _, Int<0>{}), typename Traits::TileShape{},
      typename Traits::TiledMma{}, cluster_layout_vmnk);
  auto tma_weight_low = make_tma_atom_A_sm100(
      SM100_TMA_2SM_LOAD_MULTICAST{}, weight_low,
      typename Traits::WeightSmemLayout{}(_, _, _, Int<0>{}), typename Traits::TileShape{},
      typename Traits::TiledMma{}, cluster_layout_vmnk);
  auto tma_activation = make_tma_atom_B_sm100(
      SM100_TMA_2SM_LOAD_MULTICAST{}, activation,
      typename Traits::ActivationSmemLayout{}(_, _, _, Int<0>{}), typename Traits::TileShape{},
      typename Traits::TiledMma{}, cluster_layout_vmnk);

  auto kernel_ptr = &kernel<Output, Traits, decltype(tma_weight_high), decltype(tma_activation)>;
  cudaError_t status = cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            Traits::kSharedMemoryBytes);
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
  if (multiprocessor_count < 2) {
    return cudaErrorInvalidDevice;
  }

  int const output_tiles = (args.output_channel_count + OutputChannelTile - 1) / OutputChannelTile;
  int const token_tiles = (args.token_count + TokenTile - 1) / TokenTile;
  int const total_tiles = output_tiles * token_tiles;
  int const resident_cluster_limit = multiprocessor_count / 2;
  int const cluster_count =
      total_tiles < resident_cluster_limit ? total_tiles : resident_cluster_limit;

  dim3 grid(cluster_count * 2, 1, 1);
  dim3 block(kThreadCount, 1, 1);
  dim3 cluster(2, 1, 1);
  cutlass::ClusterLaunchParams launch_params{grid, block, cluster, Traits::kSharedMemoryBytes,
                                             stream};
  cutlass::Status launch_status = cutlass::launch_kernel_on_cluster(
      launch_params, reinterpret_cast<void const*>(kernel_ptr), tma_weight_high, tma_weight_low,
      tma_activation, reinterpret_cast<Output*>(args.output), args.token_count,
      args.output_channel_count, args.reduction_size);
  if (launch_status != cutlass::Status::kSuccess) {
    return cudaErrorInvalidConfiguration;
  }
  return cudaGetLastError();
}

}  // namespace detail

inline KernelConfig select_kernel_config(int token_count, int output_channel_count) {
  if (output_channel_count <= 128) {
    if (token_count <= 1024) {
      using Traits = detail::KernelTraits<128, 16, 6>;
      return {128, 16, kReductionTile, 6, Traits::kSharedMemoryBytes};
    }
    if (token_count < 4096) {
      using Traits = detail::KernelTraits<128, 32, 6>;
      return {128, 32, kReductionTile, 6, Traits::kSharedMemoryBytes};
    }
    using Traits = detail::KernelTraits<128, 64, 5>;
    return {128, 64, kReductionTile, 5, Traits::kSharedMemoryBytes};
  }

  if (token_count <= 1024) {
    using Traits = detail::KernelTraits<256, 16, 3>;
    return {256, 16, kReductionTile, 3, Traits::kSharedMemoryBytes};
  }
  if (token_count < 4096) {
    using Traits = detail::KernelTraits<256, 32, 3>;
    return {256, 32, kReductionTile, 3, Traits::kSharedMemoryBytes};
  }
  using Traits = detail::KernelTraits<256, 64, 3>;
  return {256, 64, kReductionTile, 3, Traits::kSharedMemoryBytes};
}

template <class Output>
cudaError_t launch_typed(Arguments const& args, cudaStream_t stream) {
  static_assert(std::is_same_v<Output, float> || std::is_same_v<Output, Input>);

  if (args.output_channel_count <= 128) {
    if (args.token_count <= 1024) {
      return detail::launch_impl<Output, 128, 16, 6>(args, stream);
    }
    if (args.token_count < 4096) {
      return detail::launch_impl<Output, 128, 32, 6>(args, stream);
    }
    return detail::launch_impl<Output, 128, 64, 5>(args, stream);
  }

  if (args.token_count <= 1024) {
    return detail::launch_impl<Output, 256, 16, 3>(args, stream);
  }
  if (args.token_count < 4096) {
    return detail::launch_impl<Output, 256, 32, 3>(args, stream);
  }
  return detail::launch_impl<Output, 256, 64, 3>(args, stream);
}

inline cudaError_t launch(Arguments const& args, cudaStream_t stream = nullptr) {
  if (args.output == nullptr || args.activation == nullptr || args.weight_high == nullptr ||
      args.weight_low == nullptr || args.token_count <= 0 || args.output_channel_count <= 0 ||
      args.reduction_size <= 0 || (args.reduction_size % kReductionTile) != 0) {
    return cudaErrorInvalidValue;
  }

  if (args.output_type == OutputType::kFloat32) {
    return launch_typed<float>(args, stream);
  }
  if (args.output_type == OutputType::kBFloat16) {
    return launch_typed<Input>(args, stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace flashinfer::gemm::dual_bf16_weight::two_sm
