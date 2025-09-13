/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FLASHINFER_TGV_GEMM_CONFIG_H_
#define FLASHINFER_TGV_GEMM_CONFIG_H_

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

namespace flashinfer {
namespace gemm {

// TGV GEMM configurations for SM100 (Blackwell) architecture
// Format: CtaShape{CTA_M}x{CTA_N}_DMA{DMA_STAGE}
enum class TGVGemmConfigSM100 {
  // Signals that we should run heuristics to choose a config
  Undefined,

  // Signals that we should run heuristics to choose a config
  ChooseWithHeuristic,

  // CTA_M = 64 configurations
  CtaShape64x8_DMA6,
  CtaShape64x8_DMA8,
  CtaShape64x8_DMA10,
  CtaShape64x8_DMA12,

  CtaShape64x16_DMA6,
  CtaShape64x16_DMA8,
  CtaShape64x16_DMA10,

  CtaShape64x32_DMA6,
  CtaShape64x32_DMA8,

  CtaShape64x64_DMA6,

  // CTA_M = 128 configurations
  CtaShape128x16_DMA6,
};

// TGV GEMM configuration class
struct TGVGemmConfig {
  TGVGemmConfigSM100 tile_config_sm100 = TGVGemmConfigSM100::ChooseWithHeuristic;
  int sm_version = 100;  // Default to SM100 (Blackwell)
  bool enable_cuda_kernel = false;

  TGVGemmConfig() = default;

  TGVGemmConfig(TGVGemmConfigSM100 tile_config, int sm_version = 100)
      : tile_config_sm100(tile_config), sm_version(sm_version) {}

  int getTileConfigAsInt() const { return (int)tile_config_sm100; }

  // Helper function to get CTA_M, CTA_N, and DMA_STAGE from config
  void getTileParams(int& cta_m, int& cta_n, int& dma_stage) const {
    switch (tile_config_sm100) {
      // CTA_M = 64 cases
      case TGVGemmConfigSM100::CtaShape64x8_DMA6:
        cta_m = 64;
        cta_n = 8;
        dma_stage = 6;
        break;
      case TGVGemmConfigSM100::CtaShape64x8_DMA8:
        cta_m = 64;
        cta_n = 8;
        dma_stage = 8;
        break;
      case TGVGemmConfigSM100::CtaShape64x8_DMA10:
        cta_m = 64;
        cta_n = 8;
        dma_stage = 10;
        break;
      case TGVGemmConfigSM100::CtaShape64x8_DMA12:
        cta_m = 64;
        cta_n = 8;
        dma_stage = 12;
        break;

      case TGVGemmConfigSM100::CtaShape64x16_DMA6:
        cta_m = 64;
        cta_n = 16;
        dma_stage = 6;
        break;
      case TGVGemmConfigSM100::CtaShape64x16_DMA8:
        cta_m = 64;
        cta_n = 16;
        dma_stage = 8;
        break;
      case TGVGemmConfigSM100::CtaShape64x16_DMA10:
        cta_m = 64;
        cta_n = 16;
        dma_stage = 10;
        break;

      case TGVGemmConfigSM100::CtaShape64x32_DMA6:
        cta_m = 64;
        cta_n = 32;
        dma_stage = 6;
        break;
      case TGVGemmConfigSM100::CtaShape64x32_DMA8:
        cta_m = 64;
        cta_n = 32;
        dma_stage = 8;
        break;

      case TGVGemmConfigSM100::CtaShape64x64_DMA6:
        cta_m = 64;
        cta_n = 64;
        dma_stage = 6;
        break;

      // CTA_M = 128 cases
      case TGVGemmConfigSM100::CtaShape128x16_DMA6:
        cta_m = 128;
        cta_n = 16;
        dma_stage = 6;
        break;

      default:
        cta_m = -1;
        cta_n = -1;
        dma_stage = -1;
        break;
    }
  }

  std::string toString() const {
    std::stringstream tactic;
    tactic << "TGV GEMM Tactic";
    if (tile_config_sm100 != TGVGemmConfigSM100::ChooseWithHeuristic &&
        tile_config_sm100 != TGVGemmConfigSM100::Undefined) {
      int cta_m, cta_n, dma_stage;
      getTileParams(cta_m, cta_n, dma_stage);
      tactic << "\n\tsm: " << sm_version << "\n\ttile shape: " << cta_m << "x" << cta_n
             << "\n\tDMA stage: " << dma_stage
             << "\n\tenable cuda kernel: " << (enable_cuda_kernel ? "true" : "false");
    } else {
      tactic << "\n\tundefined";
    }
    tactic << "\n";
    return tactic.str();
  }
};

// Helper function to get all available TGV configurations
inline std::vector<TGVGemmConfig> getAllTgvConfigs() {
  return {
      TGVGemmConfig(TGVGemmConfigSM100::CtaShape64x8_DMA6),   // 0
      TGVGemmConfig(TGVGemmConfigSM100::CtaShape64x8_DMA8),   // 1
      TGVGemmConfig(TGVGemmConfigSM100::CtaShape64x8_DMA10),  // 2
      TGVGemmConfig(TGVGemmConfigSM100::CtaShape64x8_DMA12),  // 3

      TGVGemmConfig(TGVGemmConfigSM100::CtaShape64x16_DMA6),   // 4
      TGVGemmConfig(TGVGemmConfigSM100::CtaShape64x16_DMA8),   // 5
      TGVGemmConfig(TGVGemmConfigSM100::CtaShape64x16_DMA10),  // 6

      TGVGemmConfig(TGVGemmConfigSM100::CtaShape64x32_DMA6),  // 7
      TGVGemmConfig(TGVGemmConfigSM100::CtaShape64x32_DMA8),  // 8

      TGVGemmConfig(TGVGemmConfigSM100::CtaShape64x64_DMA6),  // 9

      TGVGemmConfig(TGVGemmConfigSM100::CtaShape128x16_DMA6),  // 10
  };
}

inline std::ostream& operator<<(std::ostream& out, TGVGemmConfig const& config) {
  out << "tile_config_sm100_enum: " << config.getTileConfigAsInt()
      << ", sm_version: " << config.sm_version
      << ", enable_cuda_kernel: " << (config.enable_cuda_kernel ? "true" : "false");
  return out;
}

}  // namespace gemm
}  // namespace flashinfer
#endif  // FLASHINFER_TGV_GEMM_CONFIG_H_
