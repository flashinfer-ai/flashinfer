/*
 * Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FLASHINFER_MXFP8_GEMM_CUTLASS_TEMPLATE_SM120_H_
#define FLASHINFER_MXFP8_GEMM_CUTLASS_TEMPLATE_SM120_H_

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // #ifndef _WIN32

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "flashinfer/gemm/cutlass_gemm_configs.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif  // #ifndef _WIN32

#include "flashinfer/gemm/mxfp8_gemm_cutlass.h"
#include "flashinfer/gemm/mxfp8_gemm_template_sm120.h"

namespace flashinfer {
namespace gemm {
using namespace cute;

// SM120 MXFP8 dispatch: CTA shape only, cluster shape is always 1x1x1.
template <typename T, bool SwapAB>
size_t dispatchMXFP8xMXFP8GemmCTAShapeSm120(T* D, void const* A, void const* B,
                                            void const* input_sf, void const* weight_sf, int m,
                                            int n, int k, int batch_count,
                                            CutlassGemmConfig gemmConfig, char* workspace,
                                            const size_t workspaceBytes, cudaStream_t stream,
                                            int* occupancy = nullptr) {
  switch (gemmConfig.tile_config_sm120) {
    case CutlassTileConfigSM120::CtaShape128x32x128B:
      return genericMxfp8GemmKernelLauncherSm120<T, cute::Int<128>, cute::Int<32>, cute::Int<128>,
                                                 SwapAB>(D, A, B, input_sf, weight_sf, m, n, k,
                                                         batch_count, gemmConfig, workspace,
                                                         workspaceBytes, stream, occupancy);
    case CutlassTileConfigSM120::CtaShape128x64x128B:
      return genericMxfp8GemmKernelLauncherSm120<T, cute::Int<128>, cute::Int<64>, cute::Int<128>,
                                                 SwapAB>(D, A, B, input_sf, weight_sf, m, n, k,
                                                         batch_count, gemmConfig, workspace,
                                                         workspaceBytes, stream, occupancy);
    case CutlassTileConfigSM120::CtaShape128x128x128B:
      return genericMxfp8GemmKernelLauncherSm120<T, cute::Int<128>, cute::Int<128>, cute::Int<128>,
                                                 SwapAB>(D, A, B, input_sf, weight_sf, m, n, k,
                                                         batch_count, gemmConfig, workspace,
                                                         workspaceBytes, stream, occupancy);
    case CutlassTileConfigSM120::CtaShape256x128x128B:
      return genericMxfp8GemmKernelLauncherSm120<T, cute::Int<256>, cute::Int<128>, cute::Int<128>,
                                                 SwapAB>(D, A, B, input_sf, weight_sf, m, n, k,
                                                         batch_count, gemmConfig, workspace,
                                                         workspaceBytes, stream, occupancy);
    case CutlassTileConfigSM120::CtaShape128x256x128B:
      return genericMxfp8GemmKernelLauncherSm120<T, cute::Int<128>, cute::Int<256>, cute::Int<128>,
                                                 SwapAB>(D, A, B, input_sf, weight_sf, m, n, k,
                                                         batch_count, gemmConfig, workspace,
                                                         workspaceBytes, stream, occupancy);
    case CutlassTileConfigSM120::Undefined:
      throw std::runtime_error(
          "[Error][MXFP8 SM120][dispatch_gemm_cta_shape] Gemm config undefined.");
    case CutlassTileConfigSM120::ChooseWithHeuristic:
      throw std::runtime_error(
          "[Error][MXFP8 SM120][dispatch_gemm_cta_shape] Gemm config should have already been "
          "set by heuristic.");
    default:
      throw std::runtime_error(
          "[Error][MXFP8 SM120][dispatch_gemm_cta_shape] Config is invalid for MXFP8 SM120 GEMM.");
  }
}

// CutlassMxfp8GemmRunnerSm120 — SM120-specific runner, no multi-SM, no cluster shape dispatch.
template <typename T>
class CutlassMxfp8GemmRunnerSm120 : public virtual CutlassMxfp8GemmRunnerInterface {
 public:
  CutlassMxfp8GemmRunnerSm120() = default;
  ~CutlassMxfp8GemmRunnerSm120() = default;

  void gemm(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
            int m, int n, int k, int batch_count, CutlassGemmConfig gemmConfig, char* workspace,
            const size_t workspaceBytes, cudaStream_t stream) override {
    dispatchToArch(reinterpret_cast<T*>(D), A, B, input_sf, weight_sf, m, n, k, batch_count,
                   gemmConfig, workspace, workspaceBytes, stream);
  }

  size_t getWorkspaceSize(int const m, int const n, int const k, int const batch_count) override {
    using MNK = std::tuple<int, int, int, int>;
    struct MNKHash {
      size_t operator()(const MNK& mnk) const {
        auto h1 = std::hash<int>{}(std::get<0>(mnk));
        auto h2 = std::hash<int>{}(std::get<1>(mnk));
        auto h3 = std::hash<int>{}(std::get<2>(mnk));
        auto h4 = std::hash<int>{}(std::get<3>(mnk));
        return h1 ^ h2 ^ h3 ^ h4;
      }
    };
    static std::unordered_map<MNK, size_t, MNKHash> workspace_hashmap;
    if (workspace_hashmap.find(std::make_tuple(m, n, k, batch_count)) == workspace_hashmap.end()) {
      size_t workspace_size = 0;
      for (auto const& gemmConfig : getConfigs()) {
        try {
          size_t curr = dispatchToArch(nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k,
                                       batch_count, gemmConfig, nullptr, 0, nullptr, nullptr);
          workspace_size = std::max(workspace_size, curr);
        } catch (std::runtime_error&) {
          // Some tile configs may fail for certain problem sizes (e.g. exceeding
          // SMEM capacity).  Swallow the error and try the next config.
          continue;
        }
      }
      workspace_hashmap[std::make_tuple(m, n, k, batch_count)] = workspace_size;
    }
    return workspace_hashmap[std::make_tuple(m, n, k, batch_count)];
  }

  std::vector<CutlassGemmConfig> getConfigs() const override {
    // SM120 MXFP8 tile configs.  No cluster shape variants (always 1x1x1).
    static const std::vector<CutlassTileConfigSM120> tiles = {
        CutlassTileConfigSM120::CtaShape128x32x128B,  CutlassTileConfigSM120::CtaShape128x64x128B,
        CutlassTileConfigSM120::CtaShape128x128x128B, CutlassTileConfigSM120::CtaShape256x128x128B,
        CutlassTileConfigSM120::CtaShape128x256x128B,
    };
    std::vector<CutlassGemmConfig> configs;
    configs.reserve(tiles.size());
    for (auto const& tile : tiles) {
      configs.emplace_back(tile, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
                           ClusterShape::ClusterShape_1x1x1, /*swap_ab*/ false);
      configs.emplace_back(tile, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
                           ClusterShape::ClusterShape_1x1x1, /*swap_ab*/ true);
    }
    return configs;
  }

 private:
  size_t dispatchToArch(T* D, void const* A, void const* B, void const* input_sf,
                        void const* weight_sf, int m, int n, int k, int batch_count,
                        CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes,
                        cudaStream_t stream, int* occupancy = nullptr) {
    if (gemmConfig.swap_ab) {
      return dispatchMXFP8xMXFP8GemmCTAShapeSm120<T, true>(D, A, B, input_sf, weight_sf, m, n, k,
                                                           batch_count, gemmConfig, workspace,
                                                           workspaceBytes, stream, occupancy);

    } else {
      return dispatchMXFP8xMXFP8GemmCTAShapeSm120<T, false>(D, A, B, input_sf, weight_sf, m, n, k,
                                                            batch_count, gemmConfig, workspace,
                                                            workspaceBytes, stream, occupancy);
    }
  }
};

}  // namespace gemm
}  // namespace flashinfer
#endif  // FLASHINFER_MXFP8_GEMM_CUTLASS_TEMPLATE_SM120_H_
