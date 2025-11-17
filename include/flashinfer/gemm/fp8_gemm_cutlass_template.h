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
#ifndef FLASHINFER_FP8_GEMM_CUTLASS_TEMPLATE_H_
#define FLASHINFER_FP8_GEMM_CUTLASS_TEMPLATE_H_

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "flashinfer/arch_condition.h"
#include "flashinfer/cutlass_utils.cuh"

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif  // __GNUC__

#include <cstddef>
#include <stdexcept>

#include "cutlass/float8.h"

namespace flashinfer {
namespace gemm {

struct _1SM {};
struct _2SM {};

// only support fp8 TN gemm
// batch GEMM
// return require_workspace_size
template <typename T, typename arch, int32_t CTA_M_, int32_t CTA_N_, int32_t CTA_K_,
          typename ClusterShape_, typename XSM_>
size_t genericFp8GemmKernelLauncherSm100(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B,
                                         float const* scale_a, float const* scale_b, T* D, int m,
                                         int n, int k, int b, CutlassGemmConfig config,
                                         char* workspacePtr, size_t const workspaceBytes,
                                         cudaStream_t stream);

template <typename T, typename arch, int32_t CTA_M_, int32_t CTA_N_, int32_t CTA_K_>
size_t dispatchGemmClusterShapeSm100(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B,
                                     float const* scale_a, float const* scale_b, T* D, int m, int n,
                                     int k, int b, CutlassGemmConfig gemmConfig, char* workspacePtr,
                                     size_t const workspaceBytes, cudaStream_t stream) {
  using namespace cute;

  switch (gemmConfig.cluster_shape) {
    case ClusterShape::ClusterShape_1x1x1:
      return genericFp8GemmKernelLauncherSm100<T, arch, CTA_M_, CTA_N_, CTA_K_, Shape<_1, _1, _1>,
                                               _1SM>(
          A, B, scale_a, scale_b, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      break;

    case ClusterShape::ClusterShape_2x1x1:
      return genericFp8GemmKernelLauncherSm100<T, arch, CTA_M_, CTA_N_, CTA_K_, Shape<_2, _1, _1>,
                                               _2SM>(
          A, B, scale_a, scale_b, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      break;
    case ClusterShape::ClusterShape_1x2x1:
      return genericFp8GemmKernelLauncherSm100<T, arch, CTA_M_, CTA_N_, CTA_K_, Shape<_1, _2, _1>,
                                               _1SM>(
          A, B, scale_a, scale_b, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      break;
    case ClusterShape::ClusterShape_2x2x1:
      return genericFp8GemmKernelLauncherSm100<T, arch, CTA_M_, CTA_N_, CTA_K_, Shape<_2, _2, _1>,
                                               _2SM>(
          A, B, scale_a, scale_b, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      break;
    case ClusterShape::ClusterShape_1x4x1:
      return genericFp8GemmKernelLauncherSm100<T, arch, CTA_M_, CTA_N_, CTA_K_, Shape<_1, _4, _1>,
                                               _1SM>(
          A, B, scale_a, scale_b, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      break;
    default:
      throw std::runtime_error("invalid config for fp8 gemm");
      break;
  }
}

template <typename T>
size_t dispatchToArch(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, float const* scale_a,
                      float const* scale_b, void* D, int m, int n, int k, int b,
                      CutlassGemmConfig gemmConfig, char* workspacePtr, size_t const workspaceBytes,
                      cudaStream_t stream) {
  using namespace cute;

  using arch = cutlass::arch::Sm100;

  // swap M, N and A,B
  //  change m-> n , n-> m
  // A rowmajor, B colmajor , C, D  rowmajor
  switch (gemmConfig.tile_config_sm100) {
    case CutlassTileConfigSM100::CtaShape64x64x128B:
      return dispatchGemmClusterShapeSm100<T, arch, 64, 64, 128>(
          B, A, scale_a, scale_b, static_cast<T*>(D), n, m, k, b, gemmConfig, workspacePtr,
          workspaceBytes, stream);
      break;
    case CutlassTileConfigSM100::CtaShape64x128x128B:
      return dispatchGemmClusterShapeSm100<T, arch, 64, 128, 128>(
          B, A, scale_a, scale_b, static_cast<T*>(D), n, m, k, b, gemmConfig, workspacePtr,
          workspaceBytes, stream);
      break;
    case CutlassTileConfigSM100::CtaShape64x256x128B:
      return dispatchGemmClusterShapeSm100<T, arch, 64, 256, 128>(
          B, A, scale_a, scale_b, static_cast<T*>(D), n, m, k, b, gemmConfig, workspacePtr,
          workspaceBytes, stream);
      break;
    case CutlassTileConfigSM100::CtaShape128x64x128B:
      return dispatchGemmClusterShapeSm100<T, arch, 128, 64, 128>(
          B, A, scale_a, scale_b, static_cast<T*>(D), n, m, k, b, gemmConfig, workspacePtr,
          workspaceBytes, stream);
      break;
    case CutlassTileConfigSM100::CtaShape128x128x128B:
      return dispatchGemmClusterShapeSm100<T, arch, 128, 128, 128>(
          B, A, scale_a, scale_b, static_cast<T*>(D), n, m, k, b, gemmConfig, workspacePtr,
          workspaceBytes, stream);
      break;
    case CutlassTileConfigSM100::CtaShape128x256x128B:
      return dispatchGemmClusterShapeSm100<T, arch, 128, 256, 128>(
          B, A, scale_a, scale_b, static_cast<T*>(D), n, m, k, b, gemmConfig, workspacePtr,
          workspaceBytes, stream);
      break;

    default:
      throw std::runtime_error("unsupported tile config for fp8 gemm");
      break;
  }
}

template <typename T>
void CutlassFp8GemmRunner<T>::gemm(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B,
                                   float const* scale_a, float const* scale_b, void* D, int m,
                                   int n, int k, int b, CutlassGemmConfig gemmConfig,
                                   char* workspacePtr, size_t const workspaceBytes,
                                   cudaStream_t stream) {
  dispatchToArch<T>(A, B, scale_a, scale_b, reinterpret_cast<T*>(D), m, n, k, b, gemmConfig,
                    workspacePtr, workspaceBytes, stream);
}

template <typename T>
size_t CutlassFp8GemmRunner<T>::getWorkspaceSizeImpl(int m, int n, int k) {
  size_t workspace_size = 0;
  auto gemmConfigs = CutlassFp8GemmRunner<T>{}.getConfigs();
  for (auto const& gemmConfig : gemmConfigs) {
    try {
      size_t curr_workspace_size = dispatchToArch<T>(nullptr, nullptr, nullptr, nullptr, nullptr, m,
                                                     n, k, 1, gemmConfig, nullptr, 0, nullptr);

      workspace_size = std::max(workspace_size, curr_workspace_size);
    } catch (std::runtime_error& e) {
      // Swallow errors when SMEM exceeds maximum allowed
      continue;
    }
  }

  return workspace_size;
}

template <typename T>
size_t CutlassFp8GemmRunner<T>::getWorkspaceSize(int m, int n, int k) {
  // Custom hash function for the MNK type
  using MNK = std::tuple<int, int, int>;

  struct MNKHash {
    size_t operator()(const MNK& mnk) const {
      auto h1 = std::hash<int>{}(std::get<0>(mnk));
      auto h2 = std::hash<int>{}(std::get<1>(mnk));
      auto h3 = std::hash<int>{}(std::get<2>(mnk));
      return h1 ^ h2 ^ h3;
    }
  };

  static std::unordered_map<MNK, size_t, MNKHash> workspace_hashmap;

  size_t workspace_size = 0;
  if (workspace_hashmap.find(std::make_tuple(m, n, k)) == workspace_hashmap.end()) {
    workspace_size = CutlassFp8GemmRunner<T>::getWorkspaceSizeImpl(m, n, k);
    workspace_hashmap[std::make_tuple(m, n, k)] = workspace_size;
  } else {
    workspace_size = workspace_hashmap[std::make_tuple(m, n, k)];
  }
  return workspace_size;
}

template <typename T>
std::vector<CutlassGemmConfig> CutlassFp8GemmRunner<T>::getConfigs() const {
  std::vector<CutlassGemmConfig> candidate_configs;

  std::vector<CutlassGemmConfig> candidateConfigs;

  std::vector<CutlassTileConfigSM100> tilesSm100 = {
      CutlassTileConfigSM100::CtaShape64x64x128B,   CutlassTileConfigSM100::CtaShape64x128x128B,
      CutlassTileConfigSM100::CtaShape64x256x128B,  CutlassTileConfigSM100::CtaShape128x64x128B,
      CutlassTileConfigSM100::CtaShape128x128x128B, CutlassTileConfigSM100::CtaShape128x256x128B,
  };

  std::vector<ClusterShape> clusterShapes = {
      ClusterShape::ClusterShape_1x1x1, ClusterShape::ClusterShape_1x2x1,
      ClusterShape::ClusterShape_1x4x1, ClusterShape::ClusterShape_2x1x1,
      ClusterShape::ClusterShape_2x2x1,
  };
  for (auto const& tile_config : tilesSm100) {
    for (auto const& cluster_config : clusterShapes) {
      CutlassGemmConfig config(tile_config, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
                               cluster_config);
      candidateConfigs.push_back(config);
    }
  }

  return candidateConfigs;
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_FP8_GEMM_CUTLASS_TEMPLATE_H_
