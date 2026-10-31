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
#ifndef FLASHINFER_FP8_GEMM_CUTLASS_TEMPLATE_SM120_H_
#define FLASHINFER_FP8_GEMM_CUTLASS_TEMPLATE_SM120_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "flashinfer/gemm/cutlass_gemm_configs.h"
#include "flashinfer/gemm/fp8_gemm_cutlass.h"
#include "flashinfer/gemm/fp8_gemm_template_sm120.h"

namespace flashinfer {
namespace gemm {

template <typename T>
size_t dispatchFp8GemmCTAShapeSm120(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B,
                                    float const* scale_a, float const* scale_b, T* D, int m, int n,
                                    int k, int b, CutlassGemmConfig gemmConfig, char* workspacePtr,
                                    size_t const workspaceBytes, cudaStream_t stream) {
  switch (gemmConfig.tile_config_sm120) {
    case CutlassTileConfigSM120::CtaShape16x64x128B:
      return genericFp8GemmKernelLauncherSm120<T, 16, 64, 128>(
          A, B, scale_a, scale_b, D, m, n, k, b, workspacePtr, workspaceBytes, stream);
    case CutlassTileConfigSM120::CtaShape32x64x128B:
      return genericFp8GemmKernelLauncherSm120<T, 32, 64, 128>(
          A, B, scale_a, scale_b, D, m, n, k, b, workspacePtr, workspaceBytes, stream);
    case CutlassTileConfigSM120::CtaShape64x64x128B:
      return genericFp8GemmKernelLauncherSm120<T, 64, 64, 128>(
          A, B, scale_a, scale_b, D, m, n, k, b, workspacePtr, workspaceBytes, stream);
    case CutlassTileConfigSM120::CtaShape128x64x128B:
      return genericFp8GemmKernelLauncherSm120<T, 128, 64, 128>(
          A, B, scale_a, scale_b, D, m, n, k, b, workspacePtr, workspaceBytes, stream);
    case CutlassTileConfigSM120::CtaShape128x128x128B:
      return genericFp8GemmKernelLauncherSm120<T, 128, 128, 128>(
          A, B, scale_a, scale_b, D, m, n, k, b, workspacePtr, workspaceBytes, stream);
    case CutlassTileConfigSM120::Undefined:
      throw std::runtime_error("[Fp8 Gemm Runner SM120] Gemm config undefined.");
    case CutlassTileConfigSM120::ChooseWithHeuristic:
      throw std::runtime_error(
          "[Fp8 Gemm Runner SM120] Gemm config should have already been set by heuristic.");
    default:
      throw std::runtime_error("[Fp8 Gemm Runner SM120] Config is invalid for FP8 SM120 GEMM.");
  }
}

// SM120-specific runner: one CTA shape per tactic, cluster shape always 1x1x1.
template <typename T>
class CutlassFp8GemmRunnerSm120 : public virtual CutlassFp8GemmRunnerInterface {
 public:
  CutlassFp8GemmRunnerSm120() = default;
  ~CutlassFp8GemmRunnerSm120() = default;

  void gemm(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, float const* scale_a,
            float const* scale_b, void* D, int m, int n, int k, int b, CutlassGemmConfig gemmConfig,
            char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream) override {
    dispatchFp8GemmCTAShapeSm120<T>(A, B, scale_a, scale_b, reinterpret_cast<T*>(D), m, n, k, b,
                                    gemmConfig, workspacePtr, workspaceBytes, stream);
  }

  size_t getWorkspaceSize(int m, int n, int k) override {
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
    auto key = std::make_tuple(m, n, k);
    if (workspace_hashmap.find(key) == workspace_hashmap.end()) {
      size_t workspace_size = 0;
      for (auto const& gemmConfig : getConfigs()) {
        size_t curr = dispatchFp8GemmCTAShapeSm120<T>(nullptr, nullptr, nullptr, nullptr, nullptr,
                                                      m, n, k, 1, gemmConfig, nullptr, 0, nullptr);
        workspace_size = std::max(workspace_size, curr);
      }
      workspace_hashmap[key] = workspace_size;
    }
    return workspace_hashmap[key];
  }

  std::vector<CutlassGemmConfig> getConfigs() const override {
    // Small-M tiles (Pingpong) first, then the Cooperative 128-row tiles for larger M.
    static const std::vector<CutlassTileConfigSM120> tiles = {
        CutlassTileConfigSM120::CtaShape16x64x128B,   CutlassTileConfigSM120::CtaShape32x64x128B,
        CutlassTileConfigSM120::CtaShape64x64x128B,   CutlassTileConfigSM120::CtaShape128x64x128B,
        CutlassTileConfigSM120::CtaShape128x128x128B,
    };
    std::vector<CutlassGemmConfig> configs;
    configs.reserve(tiles.size());
    for (auto const& tile : tiles) {
      configs.emplace_back(tile, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
                           ClusterShape::ClusterShape_1x1x1, /*swap_ab*/ false);
    }
    return configs;
  }

  // Untuned default (tactic -1): the tile with the lowest (CTA waves x per-wave cost). The costs
  // are relative per-wave times of each tile, fitted on RTX 5090 at M >= 1024 over ten LLM
  // projection shapes; with them the pick was within 10% of the fastest tile on all 130
  // (shape, M) points from M = 1 to 4096 (geomean 0.3%), and within 10% when fitted on half
  // the shapes and checked on the other half. Wave counting is what matters: a 128x128 tile
  // leaves most SMs idle when M * N / (128 * 128) is below the SM count, as at decode M.
  static CutlassTileConfigSM120 heuristicTile(int m, int n, int b, int sm_count) {
    struct Candidate {
      CutlassTileConfigSM120 tile;
      int cta_m, cta_n, wave_cost;
    };
    static constexpr Candidate kCandidates[] = {
        {CutlassTileConfigSM120::CtaShape16x64x128B, 16, 64, 6},
        {CutlassTileConfigSM120::CtaShape32x64x128B, 32, 64, 7},
        {CutlassTileConfigSM120::CtaShape64x64x128B, 64, 64, 10},
        {CutlassTileConfigSM120::CtaShape128x64x128B, 128, 64, 18},
        {CutlassTileConfigSM120::CtaShape128x128x128B, 128, 128, 35},
    };
    sm_count = std::max(sm_count, 1);
    CutlassTileConfigSM120 best_tile = kCandidates[0].tile;
    int64_t best_cost = -1;
    for (auto const& c : kCandidates) {
      int64_t ctas = int64_t(b) * ((m + c.cta_m - 1) / c.cta_m) * ((n + c.cta_n - 1) / c.cta_n);
      int64_t cost = (ctas + sm_count - 1) / sm_count * c.wave_cost;
      if (best_cost < 0 || cost < best_cost) {
        best_cost = cost;
        best_tile = c.tile;
      }
    }
    return best_tile;
  }
};

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_FP8_GEMM_CUTLASS_TEMPLATE_SM120_H_
