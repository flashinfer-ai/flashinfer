/*
 * Copyright (c) 2025 by FlashInfer team.
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

// clang-format off
#include <cstdint>
#include <type_traits>

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/layout.hpp>

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_size.h>

#include "cute_sm120_mxfp8_groupwise/sm120_common/ab_tma_load.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_common/epilogue.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_common/math.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_common/scheduler.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_blockscaled/sf_mxfp8_tma_load.cuh"
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {
namespace sm120_blockscaled {

using namespace cute;

template <int kTileM, int kTileN, bool kUseTmaStore, bool kSwapAB = false>
struct Sm120BlockScaledMMAConfig {
  using MMA_Atom =
      cute::MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<cute::float_e4m3_t, cute::float_e4m3_t,
                                                             float, cute::float_ue8m0_t, 32>>;

  static_assert(kTileN >= 8 && (kTileN % 8) == 0,
                "kTileN must be >= 8 and multiple of MMA atom_N=8");
  static constexpr int kNumMathWarpN = (kTileN >= 32) ? 4 : (kTileN / 8);
  static constexpr int kNumMathWarpM = 8 / kNumMathWarpN;
  static constexpr int kNumMathWarps = kNumMathWarpM * kNumMathWarpN;
  static constexpr int kNumMathThreads = kNumMathWarps * 32;
  static constexpr int kNumMathWG = kNumMathThreads / 128;
  static_assert(kNumMathWarps == 8, "Total math warps must be 8 (256 threads = 2 WG)");
  static_assert(kTileM >= kNumMathWarpM * 16,
                "kTileM must be >= kNumMathWarpM * 16 (ThrLayout M-direction lower bound).");

  using PermMmaTileM = Int<(kUseTmaStore && !kSwapAB) ? ((kTileM < 32) ? kTileM : 32) : kTileM>;
  using PermMmaTileN = Int<kTileN>;

  using TiledMma = TiledMMA<
      MMA_Atom,
      Layout<Shape<Int<kNumMathWarpM>, Int<kNumMathWarpN>, _1>, Stride<_1, Int<kNumMathWarpM>, _0>>,
      Tile<PermMmaTileM, PermMmaTileN, Underscore>>;

  static_assert(size<2>(typename MMA_Atom::Shape_MNK{}) == 32,
                "MMA atom K-dim must be 32 to match SFVecSize");
};

template <int TileM_ = 32, int TileN_ = 128, int TileK_ = 128, int Stages_ = 4, int GranK_ = 128,
          sm120_common::GemmType GemmType_ =
              sm120_common::GemmType::MGroupedContiguousWithPsumLayout,
          bool SwapAB_ = false>
struct SM120BlockScaledBuilder {
  using ElementA = cute::float_e4m3_t;
  using ElementB = cute::float_e4m3_t;
  using ElementAccum = float;
  using ElementD = cute::bfloat16_t;

  static constexpr sm120_common::GemmType kGemmType = GemmType_;
  static constexpr bool kFlat = sm120_common::is_flat_gemm(GemmType_);
  static constexpr bool kSwapAB = SwapAB_;
  static constexpr bool kPerBatchAB = (GemmType_ == sm120_common::GemmType::Batched ||
                                       GemmType_ == sm120_common::GemmType::MGroupedMasked);
  static constexpr bool kUseTmaStore =
      sm120_common::utils::EnableTmaStore<kFlat, kSwapAB, TileN_, kPerBatchAB>();
  static constexpr bool kUnionSmem = !kUseTmaStore;
  static constexpr int AB_Stages = Stages_;

  static constexpr int kGranK = GranK_;

  static constexpr int kTileM = TileM_;
  static constexpr int kTileN = TileN_;
  static constexpr int kTileK = TileK_;
  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using ClusterShape = Shape<_1, _1, _1>;
  using ProblemShape = Shape<int, int, int, int>;

  using SFConfig = Sm120BlockScaledSFConfig<kTileM, kTileN, TileK_, Stages_, GranK_, int32_t,
                                            cute::float_ue8m0_t, kGemmType, kSwapAB>;
  using MMAConfig = Sm120BlockScaledMMAConfig<kTileM, kTileN, kUseTmaStore, kSwapAB>;
  using ABLoadConfig = sm120_common::Sm120BlockScaledABLoadConfig<kTileM, kTileN, kTileK, AB_Stages,
                                                                  ElementA, ElementB>;
  using TmaStoreConfig = std::conditional_t<
      kSwapAB,
      sm120_common::Sm120BlockScaledSwapABTmaStoreConfig<kTileM, kTileN, ElementD, kUseTmaStore>,
      sm120_common::Sm120BlockScaledTmaStoreConfig<kTileM, kTileN, ElementD, kUseTmaStore>>;
  using R2GStoreConfig = std::conditional_t<
      kSwapAB, sm120_common::Sm120BlockScaledSwapABR2GStoreConfig<kTileM, kTileN, ElementD>,
      sm120_common::Sm120BlockScaledR2GStoreConfig<kTileM, kTileN, ElementD>>;

  struct SharedStorageLoad : cute::aligned_struct<128, _0> {
    alignas(1024)
        cute::ArrayEngine<ElementA, cute::cosize_v<typename ABLoadConfig::SmemLayoutA>> smem_A;
    alignas(1024)
        cute::ArrayEngine<ElementB, cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFA>>
        smem_SFA;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFB>>
        smem_SFB;
  };

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;

  static constexpr int kNumStoreMbar = kUseTmaStore ? TmaStoreConfig::StagesD : 1;
  struct BarrierStorage {
    FullBarrier ab_full_mbar[AB_Stages];
    EmptyBarrier ab_empty_mbar[AB_Stages];
    FullBarrier sf_full_mbar[SFConfig::SF_Stages];
    EmptyBarrier sf_empty_mbar[SFConfig::SF_Stages];
    EmptyBarrier store_full_mbar[kNumStoreMbar];
    EmptyBarrier store_empty_mbar[kNumStoreMbar];
  };

  struct TensorStorageSplit {
    SharedStorageLoad load;
    typename TmaStoreConfig::SharedStorageTmaStore store;
  };

  union TensorStorageUnion {
    SharedStorageLoad load;
    typename R2GStoreConfig::SharedStorageR2G store;
  };
};

}  // namespace sm120_blockscaled
}  // namespace flashinfer::gemm::mxfp8_cute_sm120
