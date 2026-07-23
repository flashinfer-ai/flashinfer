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

#include "cute_sm120_mxfp8_groupwise/sm120_blockscaling/builder.cuh"
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {
namespace sm120_blockscaling {

using namespace cute;

template <int TileM_ = 128, int TileN_ = 64, int TileK_ = 128, int Stages_ = 2,
          bool SwapAB_ = false>
struct SM120BlockScalingFusedMoeBuilder
    : SM120BlockScalingBuilder<TileM_, TileN_, TileK_, Stages_, SwapAB_ ? 128 : 1,
                               SwapAB_ ? 1 : 128, 128,
                               sm120_common::GemmType::MGroupedContiguousWithZeroPadding, SwapAB_> {
  using Base =
      SM120BlockScalingBuilder<TileM_, TileN_, TileK_, Stages_, SwapAB_ ? 128 : 1,
                               SwapAB_ ? 1 : 128, 128,
                               sm120_common::GemmType::MGroupedContiguousWithZeroPadding, SwapAB_>;

  using ElementA = typename Base::ElementA;
  using ElementB = typename Base::ElementB;
  using ElementScale = typename Base::ElementScale;
  using ABLoadConfig = typename Base::ABLoadConfig;
  using SFConfig = typename Base::SFConfig;
  using StagedR2GStoreConfig = typename Base::StagedR2GStoreConfig;

  static constexpr uint32_t LoadRegisterRequirement = Base::kUseStagedR2G ? 120 : 40;
  static constexpr uint32_t MmaRegisterRequirement = Base::kUseStagedR2G ? 192 : 232;

  struct SharedStorageLoadDefault : cute::aligned_struct<128, _0> {
    alignas(1024)
        cute::ArrayEngine<ElementA, cute::cosize_v<typename ABLoadConfig::SmemLayoutA>> smem_A;
    alignas(1024)
        cute::ArrayEngine<ElementB, cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B_up;
    alignas(1024)
        cute::ArrayEngine<ElementB, cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B_gate;
    alignas(128)
        cute::ArrayEngine<ElementScale, cute::cosize_v<typename SFConfig::SmemLayoutSFA>> smem_SFA;
    cute::ArrayEngine<ElementScale, cute::cosize_v<typename SFConfig::SmemLayoutSFB>> smem_SFB_up;
    cute::ArrayEngine<ElementScale, cute::cosize_v<typename SFConfig::SmemLayoutSFB>> smem_SFB_gate;
  };

  struct SharedStorageLoadSwapAB : cute::aligned_struct<128, _0> {
    alignas(1024)
        cute::ArrayEngine<ElementA, cute::cosize_v<typename ABLoadConfig::SmemLayoutA>> smem_A_up;
    alignas(1024)
        cute::ArrayEngine<ElementA, cute::cosize_v<typename ABLoadConfig::SmemLayoutA>> smem_A_gate;
    alignas(1024)
        cute::ArrayEngine<ElementB, cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B;
    cute::ArrayEngine<ElementScale, cute::cosize_v<typename SFConfig::SmemLayoutSFA>> smem_SFA_up;
    cute::ArrayEngine<ElementScale, cute::cosize_v<typename SFConfig::SmemLayoutSFA>> smem_SFA_gate;
    cute::ArrayEngine<ElementScale, cute::cosize_v<typename SFConfig::SmemLayoutSFB>> smem_SFB;
  };

  using SharedStorageLoad =
      std::conditional_t<SwapAB_, SharedStorageLoadSwapAB, SharedStorageLoadDefault>;

  union TensorStorageUnion {
    SharedStorageLoad load;
    typename Base::R2GStoreConfig::SharedStorageR2G store;
  };

  struct TensorStorageStagedR2G {
    SharedStorageLoad load;
    typename StagedR2GStoreConfig::SharedStorageStagedR2G store;
  };
};

}  // namespace sm120_blockscaling
}  // namespace flashinfer::gemm::mxfp8_cute_sm120
