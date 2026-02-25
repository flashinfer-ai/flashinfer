/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FLASHINFER_FP4_GEMM_CUTLASS_TEMPLATE_SM120_H_
#define FLASHINFER_FP4_GEMM_CUTLASS_TEMPLATE_SM120_H_

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

#include "flashinfer/gemm/fp4_gemm_cutlass.h"

// Include the SM120-specific template
#define FLASHINFER_ENABLE_SM120
#include "fp4_gemm_template_sm120.h"

namespace flashinfer {
namespace gemm {
using namespace cute;

// UseStreamK: false = DP scheduler (default), true = StreamK scheduler
template <typename T, typename CTA_M_, typename CTA_N_, typename CTA_K_, bool UseStreamK = false>
size_t dispatchNVFP4xNVFP4GemmClusterShapeSm120(T* D, void const* A, void const* B,
                                                void const* input_sf, void const* weight_sf,
                                                float const* global_sf, int m, int n, int k,
                                                int batch_count, CutlassGemmConfig gemmConfig,
                                                char* workspace, const size_t workspaceBytes,
                                                cudaStream_t stream, int* occupancy = nullptr) {
  // For SM120/SM121, only support 1x1x1 cluster shape
  // Always use 1x1x1 cluster shape regardless of gemmConfig.cluster_shape
  if constexpr (UseStreamK) {
    return genericFp4GemmKernelLauncherStreamK<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<1>,
                                               cute::Int<1>, cute::Int<1>, _1SM>(
        D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace,
        workspaceBytes, stream, occupancy);
  } else {
    return genericFp4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<1>, cute::Int<1>,
                                        cute::Int<1>, _1SM>(
        D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace,
        workspaceBytes, stream, occupancy);
  }
}

/*!
 * \brief Dispatch FP4 GEMM operation with CTA shape configuration for SM120/SM121
 * \param D Output matrix pointer
 * \param A Input matrix A pointer (FP4 quantized)
 * \param B Input matrix B pointer (FP4 quantized)
 * \param input_sf Input scale factors
 * \param weight_sf Weight scale factors
 * \param global_sf Global scale factor
 * \param m Number of rows in matrix A and output matrix D
 * \param n Number of columns in matrix B and output matrix D
 * \param k Number of columns in matrix A and rows in matrix B
 * \param batch_count Number of batches for batched GEMM
 * \param gemmConfig GEMM configuration including tile size and cluster shape
 * \param workspace Workspace buffer for temporary storage
 * \param workspaceBytes Size of workspace buffer in bytes
 * \param stream CUDA stream for kernel execution
 * \param occupancy Optional pointer to store kernel occupancy
 * \return Size of workspace required in bytes
 */
// Helper macro to dispatch tile config with scheduler selection
#define DISPATCH_TILE_CONFIG(CTA_M, CTA_N, CTA_K, USE_STREAMK)                              \
  return dispatchNVFP4xNVFP4GemmClusterShapeSm120<T, cute::Int<CTA_M>, cute::Int<CTA_N>,    \
                                                  cute::Int<CTA_K>, USE_STREAMK>(           \
      D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, \
      workspaceBytes, stream, occupancy)

// Helper macro to dispatch with scheduler check
#define DISPATCH_WITH_SCHEDULER(CTA_M, CTA_N, CTA_K)  \
  if (gemmConfig.use_stream_k) {                      \
    DISPATCH_TILE_CONFIG(CTA_M, CTA_N, CTA_K, true);  \
  } else {                                            \
    DISPATCH_TILE_CONFIG(CTA_M, CTA_N, CTA_K, false); \
  }

template <typename T>
size_t dispatchNVFP4xNVFP4GemmCTAShapeSm120(T* D, void const* A, void const* B,
                                            void const* input_sf, void const* weight_sf,
                                            float const* global_sf, int m, int n, int k,
                                            int batch_count, CutlassGemmConfig gemmConfig,
                                            char* workspace, const size_t workspaceBytes,
                                            cudaStream_t stream, int* occupancy = nullptr) {
  // Dispatch based on tile config and scheduler type
  switch (gemmConfig.tile_config_sm120) {
    case CutlassTileConfigSM120::CtaShape128x128x128B:
      DISPATCH_WITH_SCHEDULER(128, 128, 128);
    case CutlassTileConfigSM120::CtaShape128x128x256B:
      DISPATCH_WITH_SCHEDULER(128, 128, 256);
    case CutlassTileConfigSM120::CtaShape256x128x128B:
      DISPATCH_WITH_SCHEDULER(256, 128, 128);
    case CutlassTileConfigSM120::Undefined:
      throw std::runtime_error("[Error][FP4][dispatch_gemm_cta_shape] Gemm config undefined.");
    case CutlassTileConfigSM120::ChooseWithHeuristic:
      throw std::runtime_error(
          "[Error][FP4][dispatch_gemm_cta_shape] Gemm config should have already been set by "
          "heuristic.");
    default:
      DISPATCH_WITH_SCHEDULER(128, 128, 128);  // Fallback
  }
}

#undef DISPATCH_WITH_SCHEDULER
#undef DISPATCH_TILE_CONFIG

template <typename T, FP4GemmType fp4GemmType>
CutlassFp4GemmRunner<T, fp4GemmType>::CutlassFp4GemmRunner() {}

template <typename T, FP4GemmType fp4GemmType>
CutlassFp4GemmRunner<T, fp4GemmType>::~CutlassFp4GemmRunner() {}

template <typename T, FP4GemmType fp4GemmType>
size_t CutlassFp4GemmRunner<T, fp4GemmType>::dispatchToArch(
    T* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
    float const* global_sf, int m, int n, int k, int batch_count, CutlassGemmConfig gemmConfig,
    char* workspace, const size_t workspaceBytes, cudaStream_t stream, int* occupancy) {
  if constexpr (fp4GemmType == FP4GemmType::W4A4_NVFP4_NVFP4) {
    return dispatchNVFP4xNVFP4GemmCTAShapeSm120<T>(D, A, B, input_sf, weight_sf, global_sf, m, n, k,
                                                   batch_count, gemmConfig, workspace,
                                                   workspaceBytes, stream, occupancy);
  } else {
    throw std::runtime_error(
        "[Error][CutlassFp4GemmRunner][GEMM Dispatch] FP4 Gemm type unsupported for "
        "CUTLASS FP4 GEMM");
  }
}

template <typename T, FP4GemmType fp4GemmType>
void CutlassFp4GemmRunner<T, fp4GemmType>::gemm(void* D, void const* A, void const* B,
                                                void const* input_sf, void const* weight_sf,
                                                float const* global_sf, int m, int n, int k,
                                                int batch_count, CutlassGemmConfig gemmConfig,
                                                char* workspace, const size_t workspaceBytes,
                                                cudaStream_t stream) {
  CutlassFp4GemmRunner<T, fp4GemmType>::dispatchToArch(
      reinterpret_cast<T*>(D), A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count,
      gemmConfig, workspace, workspaceBytes, stream);
}

template <typename T, FP4GemmType fp4GemmType>
std::vector<CutlassGemmConfig> CutlassFp4GemmRunner<T, fp4GemmType>::getConfigs() const {
  std::vector<CutlassGemmConfig> candidateConfigs;

  // All supported tile configurations for SM120
  std::vector<CutlassTileConfigSM120> tilesSm120 = {
      CutlassTileConfigSM120::CtaShape128x128x128B,
      CutlassTileConfigSM120::CtaShape128x128x256B,
      CutlassTileConfigSM120::CtaShape256x128x128B,
  };

  // SM120 only supports 1x1x1 cluster shape
  ClusterShape clusterShape = ClusterShape::ClusterShape_1x1x1;

  // Generate configs for both DP and StreamK schedulers
  for (auto const& tile_config : tilesSm120) {
    // Default DP scheduler (use_stream_k = false)
    candidateConfigs.push_back(CutlassGemmConfig(tile_config, MainloopScheduleType::AUTO,
                                                 EpilogueScheduleType::AUTO, clusterShape, false));

    // StreamK scheduler (use_stream_k = true) - better for small M/N, large K
    candidateConfigs.push_back(CutlassGemmConfig(tile_config, MainloopScheduleType::AUTO,
                                                 EpilogueScheduleType::AUTO, clusterShape, true));
  }
  return candidateConfigs;
}

template <typename T, FP4GemmType fp4GemmType>
size_t CutlassFp4GemmRunner<T, fp4GemmType>::getWorkspaceSizeImpl(int const m, int const n,
                                                                  int const k,
                                                                  int const batch_count) {
  size_t workspace_size = 0;
  auto gemmConfigs = CutlassFp4GemmRunner<T, fp4GemmType>{}.getConfigs();
  for (auto const& gemmConfig : gemmConfigs) {
    try {
      size_t curr_workspace_size = CutlassFp4GemmRunner<T, fp4GemmType>::dispatchToArch(
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, batch_count, gemmConfig,
          nullptr, 0, 0);
      workspace_size = std::max(workspace_size, curr_workspace_size);
    } catch (std::runtime_error& e) {
      // Swallow errors when SMEM exceeds maximum allowed
      continue;
    }
  }
  return workspace_size;
}

template <typename T, FP4GemmType fp4GemmType>
size_t CutlassFp4GemmRunner<T, fp4GemmType>::getWorkspaceSize(int const m, int const n, int const k,
                                                              int const batch_count) {
  // Custom hash function for the MNKB type
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

  size_t workspace_size = 0;
  if (workspace_hashmap.find(std::make_tuple(m, n, k, batch_count)) == workspace_hashmap.end()) {
    workspace_size =
        CutlassFp4GemmRunner<T, fp4GemmType>::getWorkspaceSizeImpl(m, n, k, batch_count);
    workspace_hashmap[std::make_tuple(m, n, k, batch_count)] = workspace_size;
  } else {
    workspace_size = workspace_hashmap[std::make_tuple(m, n, k, batch_count)];
  }
  return workspace_size;
}

}  // namespace gemm
}  // namespace flashinfer
#endif  // FLASHINFER_FP4_GEMM_CUTLASS_TEMPLATE_SM120_H_
