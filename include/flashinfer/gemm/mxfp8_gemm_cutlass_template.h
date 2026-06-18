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

#ifndef FLASHINFER_MXFP8_GEMM_CUTLASS_TEMPLATE_H_
#define FLASHINFER_MXFP8_GEMM_CUTLASS_TEMPLATE_H_

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
#include "mxfp8_gemm_template_sm100.h"

namespace flashinfer {
namespace gemm {
using namespace cute;

template <typename T, typename CTA_M_, typename CTA_N_, typename CTA_K_>
size_t dispatchMXFP8xMXFP8GemmClusterShapeSm100(T* D, void const* A, void const* B,
                                                void const* input_sf, void const* weight_sf, int m,
                                                int n, int k, int batch_count,
                                                CutlassGemmConfig gemmConfig, char* workspace,
                                                const size_t workspaceBytes, cudaStream_t stream,
                                                int* occupancy = nullptr) {
  switch (gemmConfig.cluster_shape) {
    case ClusterShape::ClusterShape_1x1x1:
      return genericMxfp8GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<1>, cute::Int<1>,
                                            cute::Int<1>, _1SM>(
          D, A, B, input_sf, weight_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case ClusterShape::ClusterShape_2x1x1:
      return genericMxfp8GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<1>,
                                            cute::Int<1>, _2SM>(
          D, A, B, input_sf, weight_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case ClusterShape::ClusterShape_1x2x1:
      return genericMxfp8GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<1>, cute::Int<2>,
                                            cute::Int<1>, _1SM>(
          D, A, B, input_sf, weight_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case ClusterShape::ClusterShape_2x2x1:
      return genericMxfp8GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<2>,
                                            cute::Int<1>, _2SM>(
          D, A, B, input_sf, weight_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case ClusterShape::ClusterShape_1x4x1:
      return genericMxfp8GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<1>, cute::Int<4>,
                                            cute::Int<1>, _1SM>(
          D, A, B, input_sf, weight_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case ClusterShape::ClusterShape_4x2x1:
      return genericMxfp8GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<4>, cute::Int<2>,
                                            cute::Int<1>, _2SM>(
          D, A, B, input_sf, weight_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case ClusterShape::ClusterShape_2x4x1:
      return genericMxfp8GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<4>,
                                            cute::Int<1>, _2SM>(
          D, A, B, input_sf, weight_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case ClusterShape::ClusterShape_4x4x1:
      return genericMxfp8GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<4>, cute::Int<4>,
                                            cute::Int<1>, _2SM>(
          D, A, B, input_sf, weight_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    default:
      throw std::runtime_error(
          "[Error][MXFP8][dispatch_gemm_cluster_shape] Config is invalid for MXFP8 GEMM.");
      break;
  }
}

template <typename T>
size_t dispatchMXFP8xMXFP8GemmCTAShapeSm100(T* D, void const* A, void const* B,
                                            void const* input_sf, void const* weight_sf, int m,
                                            int n, int k, int batch_count,
                                            CutlassGemmConfig gemmConfig, char* workspace,
                                            const size_t workspaceBytes, cudaStream_t stream,
                                            int* occupancy = nullptr) {
  // TODO: check if true for MXFP8
  // Several constraints:
  // Cta N should be one of 64/128/192/256 for MXFP8 on SM100.
  // M-mode size should be 128 or 256 for 2 CTA cluster MMA;
  // M-mode size should be 128 for 1 CTA cluster OMMA.
  // K256 looks to be better than K128
  switch (gemmConfig.tile_config_sm100) {
    case CutlassTileConfigSM100::CtaShape128x64x128B:
      return dispatchMXFP8xMXFP8GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<64>,
                                                      cute::Int<128>>(
          D, A, B, input_sf, weight_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case CutlassTileConfigSM100::CtaShape128x256x128B:
      return dispatchMXFP8xMXFP8GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<256>,
                                                      cute::Int<128>>(
          D, A, B, input_sf, weight_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case CutlassTileConfigSM100::CtaShape128x128x256B:
      return dispatchMXFP8xMXFP8GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<128>,
                                                      cute::Int<256>>(
          D, A, B, input_sf, weight_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case CutlassTileConfigSM100::CtaShape128x256x256B:
      return dispatchMXFP8xMXFP8GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<256>,
                                                      cute::Int<256>>(
          D, A, B, input_sf, weight_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case CutlassTileConfigSM100::Undefined:
      throw std::runtime_error("[Error][MXFP8][dispatch_gemm_cta_shape] Gemm config undefined.");
      break;
    case CutlassTileConfigSM100::ChooseWithHeuristic:
      throw std::runtime_error(
          "[Error][MXFP8][dispatch_gemm_cta_shape] Gemm config should have already been "
          "set by "
          "heuristic.");
      break;
    default:
      throw std::runtime_error(
          "[Error][MXFP8][dispatch_gemm_cta_shape] Config is invalid for MXFP8 GEMM.");
      break;
  }
}
template <typename T, MXFP8GemmType mxfp8GemmType>
CutlassMxfp8GemmRunner<T, mxfp8GemmType>::CutlassMxfp8GemmRunner() {}

template <typename T, MXFP8GemmType mxfp8GemmType>
CutlassMxfp8GemmRunner<T, mxfp8GemmType>::~CutlassMxfp8GemmRunner() {}

template <typename T, MXFP8GemmType mxfp8GemmType>
size_t CutlassMxfp8GemmRunner<T, mxfp8GemmType>::dispatchToArch(
    T* D, void const* A, void const* B, void const* input_sf, void const* weight_sf, int m, int n,
    int k, int batch_count, CutlassGemmConfig gemmConfig, char* workspace,
    const size_t workspaceBytes, cudaStream_t stream, int* occupancy) {
  if constexpr (mxfp8GemmType == MXFP8GemmType::W8A8_MXFP8_MXFP8) {
    return dispatchMXFP8xMXFP8GemmCTAShapeSm100<T>(D, A, B, input_sf, weight_sf, m, n, k,
                                                   batch_count, gemmConfig, workspace,
                                                   workspaceBytes, stream, occupancy);
  } else {
    throw std::runtime_error(
        "[Error][CutlassMxfp8GemmRunner][GEMM Dispatch] MXFP8 Gemm type unsupported for "
        "CUTLASS MXFP8 GEMM");
  }
}

template <typename T, MXFP8GemmType mxfp8GemmType>
void CutlassMxfp8GemmRunner<T, mxfp8GemmType>::gemm(void* D, void const* A, void const* B,
                                                    void const* input_sf, void const* weight_sf,
                                                    int m, int n, int k, int batch_count,
                                                    CutlassGemmConfig gemmConfig, char* workspace,
                                                    const size_t workspaceBytes,
                                                    cudaStream_t stream) {
  CutlassMxfp8GemmRunner<T, mxfp8GemmType>::dispatchToArch(
      reinterpret_cast<T*>(D), A, B, input_sf, weight_sf, m, n, k, batch_count, gemmConfig,
      workspace, workspaceBytes, stream);
}

template <typename T, MXFP8GemmType mxfp8GemmType>
std::vector<CutlassGemmConfig> CutlassMxfp8GemmRunner<T, mxfp8GemmType>::getConfigs() const {
  std::vector<CutlassGemmConfig> candidateConfigs;

  std::vector<CutlassTileConfigSM100> tilesSm100 = {
      CutlassTileConfigSM100::CtaShape128x64x128B,
      CutlassTileConfigSM100::CtaShape128x256x128B,
      CutlassTileConfigSM100::CtaShape128x128x256B,
      CutlassTileConfigSM100::CtaShape128x256x256B,
  };

  std::vector<ClusterShape> clusterShapes = {
      ClusterShape::ClusterShape_1x1x1, ClusterShape::ClusterShape_1x2x1,
      ClusterShape::ClusterShape_2x1x1, ClusterShape::ClusterShape_2x2x1,
      ClusterShape::ClusterShape_1x4x1, ClusterShape::ClusterShape_2x4x1,
      ClusterShape::ClusterShape_4x2x1, ClusterShape::ClusterShape_4x4x1,
  };
  for (auto const& tile_config : tilesSm100) {
    for (auto const& cluster_config : clusterShapes) {
      CutlassGemmConfig config(tile_config, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
                               cluster_config);
      candidateConfigs.push_back(config);
    }
  }

  // There’s no heuristic yet, so for users without autotuning, we provide an ordering based on
  // performance sweeps from common workloads. Keep it safe if configs are pruned.
  std::vector<int64_t> best_tactics_index = {22, 20, 29, 4, 18};
  std::vector<CutlassGemmConfig> newCandidateConfigs;
  newCandidateConfigs.reserve(candidateConfigs.size());
  for (auto const& tactic_index : best_tactics_index) {
    if (tactic_index >= 0 && tactic_index < static_cast<int64_t>(candidateConfigs.size())) {
      newCandidateConfigs.push_back(candidateConfigs[tactic_index]);
    }
  }
  for (int64_t i = 0; i < static_cast<int64_t>(candidateConfigs.size()); i++) {
    if (std::find(best_tactics_index.begin(), best_tactics_index.end(), i) ==
        best_tactics_index.end()) {
      newCandidateConfigs.push_back(candidateConfigs[i]);
    }
  }
  return newCandidateConfigs;
}

template <typename T, MXFP8GemmType mxfp8GemmType>
size_t CutlassMxfp8GemmRunner<T, mxfp8GemmType>::getWorkspaceSizeImpl(int const m, int const n,
                                                                      int const k,
                                                                      int const batch_count) {
  size_t workspace_size = 0;
  auto gemmConfigs = CutlassMxfp8GemmRunner<T, mxfp8GemmType>{}.getConfigs();
  for (auto const& gemmConfig : gemmConfigs) {
    try {
      size_t curr_workspace_size = CutlassMxfp8GemmRunner<T, mxfp8GemmType>::dispatchToArch(
          nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, batch_count, gemmConfig, nullptr, 0,
          nullptr, nullptr);
      workspace_size = std::max(workspace_size, curr_workspace_size);
    } catch (std::runtime_error& e) {
      // Swallow errors when SMEM exceeds maximum allowed
      continue;
    }
  }
  return workspace_size;
}

template <typename T, MXFP8GemmType mxfp8GemmType>
size_t CutlassMxfp8GemmRunner<T, mxfp8GemmType>::getWorkspaceSize(int const m, int const n,
                                                                  int const k,
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
        CutlassMxfp8GemmRunner<T, mxfp8GemmType>::getWorkspaceSizeImpl(m, n, k, batch_count);
    workspace_hashmap[std::make_tuple(m, n, k, batch_count)] = workspace_size;
  } else {
    workspace_size = workspace_hashmap[std::make_tuple(m, n, k, batch_count)];
  }
  return workspace_size;
}

}  // namespace gemm
}  // namespace flashinfer
#endif  // FLASHINFER_MXFP8_GEMM_CUTLASS_TEMPLATE_H_
