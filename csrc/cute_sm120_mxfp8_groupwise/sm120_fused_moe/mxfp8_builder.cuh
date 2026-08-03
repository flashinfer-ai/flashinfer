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

#include "cute_sm120_mxfp8_groupwise/sm120_blockscaled/builder.cuh"
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {
namespace sm120_blockscaled {

using namespace cute;

template <int TileM_ = 128, int TileN_ = 64, int TileK_ = 128, int Stages_ = 2, int GranK_ = 128,
          bool SwapAB_ = false, bool UseStagedR2G_ = false>
struct SM120BlockScaledFusedMoeBuilder
    : SM120BlockScaledBuilder<TileM_, TileN_, TileK_, Stages_, GranK_,
                              sm120_common::GemmType::MGroupedContiguousWithZeroPadding, SwapAB_> {
  using Base =
      SM120BlockScaledBuilder<TileM_, TileN_, TileK_, Stages_, GranK_,
                              sm120_common::GemmType::MGroupedContiguousWithZeroPadding, SwapAB_>;

  static_assert(GranK_ == 32 || GranK_ == 128);

  using ElementA = typename Base::ElementA;
  using ElementB = typename Base::ElementB;
  using ABLoadConfig = typename Base::ABLoadConfig;
  using SFConfig = typename Base::SFConfig;
  using MMAConfig = typename Base::MMAConfig;
  using R2GStoreConfig = typename Base::R2GStoreConfig;
  using FullBarrier = typename Base::FullBarrier;
  using EmptyBarrier = typename Base::EmptyBarrier;

  static constexpr bool kUseStagedR2G = UseStagedR2G_;
  static constexpr bool kUnionSmem = !kUseStagedR2G;
  static constexpr uint32_t LoadRegisterRequirement = kUseStagedR2G ? 120 : 56;
  static constexpr uint32_t MmaRegisterRequirement = kUseStagedR2G ? 192 : 224;

  static_assert(!Base::kUseTmaStore);

  using StagedR2GStoreConfig =
      sm120_common::Sm120BlockScaledStagedR2GStoreConfig<TileM_, TileN_, typename Base::ElementD,
                                                         kUseStagedR2G>;
  static constexpr int kNumStoreMbar = StagedR2GStoreConfig::StagesD;

  struct SharedStorageLoadDefault : cute::aligned_struct<128, _0> {
    alignas(1024)
        cute::ArrayEngine<ElementA, cute::cosize_v<typename ABLoadConfig::SmemLayoutA>> smem_A;
    alignas(1024)
        cute::ArrayEngine<ElementB, cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B_up;
    alignas(1024)
        cute::ArrayEngine<ElementB, cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B_gate;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFA>>
        smem_SFA;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFB>>
        smem_SFB_up;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFB>>
        smem_SFB_gate;
  };

  struct SharedStorageLoadSwapAB : cute::aligned_struct<128, _0> {
    alignas(1024)
        cute::ArrayEngine<ElementA, cute::cosize_v<typename ABLoadConfig::SmemLayoutA>> smem_A_up;
    alignas(1024)
        cute::ArrayEngine<ElementA, cute::cosize_v<typename ABLoadConfig::SmemLayoutA>> smem_A_gate;
    alignas(1024)
        cute::ArrayEngine<ElementB, cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFA>>
        smem_SFA_up;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFA>>
        smem_SFA_gate;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFB>>
        smem_SFB;
  };

  using SharedStorageLoad =
      std::conditional_t<SwapAB_, SharedStorageLoadSwapAB, SharedStorageLoadDefault>;

  struct BarrierStorage {
    FullBarrier ab_full_mbar[Base::AB_Stages];
    EmptyBarrier ab_empty_mbar[Base::AB_Stages];
    FullBarrier sf_full_mbar[Base::SFConfig::SF_Stages];
    EmptyBarrier sf_empty_mbar[Base::SFConfig::SF_Stages];
    EmptyBarrier store_full_mbar[kNumStoreMbar];
    EmptyBarrier store_empty_mbar[kNumStoreMbar];
  };

  union TensorStorageUnionDefault {
    SharedStorageLoad load;
    typename R2GStoreConfig::SharedStorageR2G store;
  };

  struct TensorStorageSplit {
    SharedStorageLoad load;
    typename StagedR2GStoreConfig::SharedStorageStagedR2G store;
  };

  using TensorStorageUnion =
      std::conditional_t<kUseStagedR2G, TensorStorageSplit, TensorStorageUnionDefault>;
};

}  // namespace sm120_blockscaled
}  // namespace flashinfer::gemm::mxfp8_cute_sm120
