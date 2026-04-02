#pragma once

// Auto-generated knob selection logic for Sm100RmsNormSiluEngine.
// Generated from optimal knob sweep results on B200 (SM100).
// Knob mapping: knobTileRows {0:1,1:4,2:8,3:32}, knobLoadSize {0:2,1:4,2:8,3:16}

#include <cstdint>

namespace flashinfer {
namespace norm {

enum class RmsNormSiluDtype : uint8_t {
  BF16 = 0,
  FP8 = 1,
  NVFP4 = 2,
};

// Compact knob configuration per (C, tokens, dtype) entry.
// WARPS_N is always 1. All values are the ACTUAL kernel parameters
// (not knob indices).
struct RmsNormSiluKnobs {
  uint8_t warps_m;        // WARPS_M value: 1, 4, 8, or 32
  uint8_t split_cols;     // knobSplitCols: 0 = no split, 4 = estimated CTAS_PER_ROW
  uint8_t kernel_cfg;     // knobKernelCfg: 0, 1, or 2
  uint8_t occupancy;      // DESIRED_OCCUPANCY: 0-16
  uint8_t bytes_per_ldg;  // BYTES_PER_LDG: 2, 4, 8, or 16
};

static constexpr int kSupportedC[] = {64, 128, 160, 256, 320, 512, 640, 1024};
static constexpr int kSupportedTokens[] = {1560, 6240, 24960, 99840, 399360};
static constexpr int kNumC = 8;
static constexpr int kNumTokens = 5;
static constexpr int kNumDtypes = 3;

// Knob LUT indexed as: knob_lut[c_idx][tokens_idx][dtype_idx]
// c_idx:      0=64, 1=128, 2=160, 3=256, 4=320, 5=512, 6=640, 7=1024
// tokens_idx: 0=1560, 1=6240, 2=24960, 3=99840, 4=399360
// dtype_idx:  0=bf16, 1=fp8, 2=nvfp4
static constexpr RmsNormSiluKnobs knob_lut[kNumC][kNumTokens][kNumDtypes] = {
    {
        // C=64
        {{8, 0, 0, 2, 4}, {8, 4, 0, 6, 4}, {8, 0, 2, 1, 4}},     // tokens=1560
        {{32, 4, 0, 2, 4}, {8, 0, 0, 3, 2}, {8, 4, 0, 4, 4}},    // tokens=6240
        {{32, 4, 0, 2, 4}, {8, 0, 0, 7, 4}, {8, 0, 1, 6, 4}},    // tokens=24960
        {{8, 0, 1, 8, 4}, {8, 0, 1, 6, 2}, {32, 4, 1, 2, 4}},    // tokens=99840
        {{4, 0, 1, 16, 4}, {32, 0, 1, 2, 2}, {32, 4, 1, 2, 4}},  // tokens=399360
    },
    {
        // C=128
        {{8, 4, 0, 3, 4}, {8, 0, 0, 3, 4}, {8, 4, 0, 3, 8}},     // tokens=1560
        {{8, 0, 0, 3, 4}, {8, 0, 0, 4, 8}, {8, 0, 0, 5, 8}},     // tokens=6240
        {{8, 0, 0, 6, 4}, {8, 0, 0, 8, 8}, {8, 0, 1, 8, 8}},     // tokens=24960
        {{32, 4, 0, 2, 4}, {32, 0, 0, 2, 8}, {32, 0, 1, 2, 8}},  // tokens=99840
        {{8, 0, 0, 8, 4}, {32, 0, 0, 2, 8}, {32, 0, 1, 2, 8}},   // tokens=399360
    },
    {
        // C=160
        {{8, 0, 0, 4, 2}, {8, 0, 0, 2, 2}, {4, 4, 0, 4, 2}},     // tokens=1560
        {{8, 0, 0, 4, 2}, {8, 0, 0, 4, 2}, {8, 0, 1, 4, 2}},     // tokens=6240
        {{8, 4, 1, 6, 2}, {8, 4, 0, 6, 2}, {8, 4, 1, 8, 2}},     // tokens=24960
        {{32, 4, 1, 2, 2}, {32, 4, 1, 2, 2}, {32, 4, 0, 1, 2}},  // tokens=99840
        {{32, 4, 1, 2, 2}, {32, 4, 1, 2, 2}, {32, 0, 1, 2, 2}},  // tokens=399360
    },
    {
        // C=256
        {{8, 0, 0, 6, 16}, {8, 4, 0, 2, 4}, {8, 0, 2, 1, 16}},      // tokens=1560
        {{8, 0, 0, 4, 4}, {8, 0, 0, 4, 4}, {8, 0, 2, 1, 16}},       // tokens=6240
        {{8, 0, 0, 8, 16}, {8, 4, 0, 8, 16}, {8, 4, 1, 6, 16}},     // tokens=24960
        {{4, 4, 0, 16, 16}, {4, 0, 0, 16, 16}, {32, 0, 1, 1, 16}},  // tokens=99840
        {{4, 0, 0, 16, 16}, {32, 0, 0, 2, 16}, {32, 0, 1, 2, 16}},  // tokens=399360
    },
    {
        // C=320
        {{8, 4, 1, 4, 4}, {8, 0, 0, 2, 4}, {4, 4, 0, 9, 4}},     // tokens=1560
        {{8, 4, 0, 5, 4}, {8, 0, 0, 5, 4}, {4, 0, 0, 9, 4}},     // tokens=6240
        {{8, 0, 0, 5, 4}, {8, 0, 0, 5, 4}, {8, 0, 1, 8, 4}},     // tokens=24960
        {{4, 0, 1, 16, 4}, {32, 0, 1, 2, 4}, {32, 4, 1, 2, 4}},  // tokens=99840
        {{32, 4, 0, 2, 4}, {32, 0, 1, 2, 4}, {32, 4, 1, 2, 4}},  // tokens=399360
    },
    {
        // C=512
        {{8, 0, 0, 2, 16}, {8, 0, 0, 2, 8}, {4, 4, 0, 3, 16}},   // tokens=1560
        {{8, 0, 0, 5, 16}, {8, 0, 0, 4, 8}, {4, 0, 0, 9, 16}},   // tokens=6240
        {{4, 0, 0, 8, 16}, {4, 0, 0, 9, 8}, {4, 0, 2, 1, 16}},   // tokens=24960
        {{4, 0, 2, 1, 8}, {32, 4, 1, 2, 8}, {32, 4, 0, 1, 16}},  // tokens=99840
        {{4, 0, 2, 1, 4}, {32, 4, 1, 2, 8}, {32, 0, 0, 1, 16}},  // tokens=399360
    },
    {
        // C=640
        {{4, 0, 0, 4, 4}, {4, 0, 0, 3, 8}, {4, 4, 0, 5, 8}},    // tokens=1560
        {{4, 0, 0, 5, 4}, {8, 0, 0, 4, 8}, {4, 0, 1, 9, 8}},    // tokens=6240
        {{4, 0, 0, 5, 4}, {8, 0, 0, 4, 8}, {4, 0, 2, 1, 8}},    // tokens=24960
        {{4, 0, 2, 1, 8}, {4, 4, 0, 9, 8}, {32, 0, 1, 1, 8}},   // tokens=99840
        {{4, 0, 2, 1, 8}, {32, 4, 1, 2, 8}, {32, 4, 1, 1, 8}},  // tokens=399360
    },
    {
        // C=1024
        {{4, 4, 0, 3, 16}, {4, 0, 0, 3, 4}, {4, 4, 0, 7, 16}},    // tokens=1560
        {{4, 0, 0, 5, 16}, {4, 0, 0, 5, 8}, {4, 0, 2, 1, 16}},    // tokens=6240
        {{4, 4, 1, 10, 16}, {1, 4, 0, 16, 8}, {4, 0, 2, 1, 16}},  // tokens=24960
        {{8, 0, 2, 1, 16}, {4, 0, 1, 9, 8}, {32, 0, 1, 1, 16}},   // tokens=99840
        {{8, 0, 2, 1, 16}, {32, 4, 1, 1, 8}, {32, 4, 1, 1, 16}},  // tokens=399360
    },
};

// Compute conservative default knobs for arbitrary problem sizes not in the LUT.
// Uses safe defaults (WARPS_M=1, BPL=4, occupancy=1) and validates vectorization
// divisibility constraints before accepting a configuration.
// Returns true if a valid configuration was found, false otherwise.
inline bool compute_default_knobs(int C, int num_tokens, RmsNormSiluDtype dtype,
                                  RmsNormSiluKnobs& out) {
  // Conservative defaults:
  //   CTAS_PER_ROW = 1, WARPS_M = 1, WARPS_N = 1, BPL = 4, occupancy = 1, kernel_cfg = 0
  // For block-scale output (NVFP4): WARPS_M = 32

  int input_size = 2;  // bf16 input always

  // Start with conservative defaults
  int warps_m = (dtype == RmsNormSiluDtype::NVFP4) ? 32 : 1;
  int warps_n = 1;  // always 1 for our engine
  int bpl = 4;      // default bytes per load
  int cpr = 1;      // no column splitting for fallback
  int occ = 1;
  int kcfg = 0;

  // Validation: C must be evenly divisible into vectorized loads.
  // NUM_ELTS = BYTES_PER_LDG / sizeof(input_t)
  // VEC_COLS = C / NUM_ELTS
  // VEC_COLS_PER_LDG = CTAS_PER_ROW * WARPS_N * 32
  // Require: C % NUM_ELTS == 0  AND  VEC_COLS % VEC_COLS_PER_LDG == 0
  // Also: LDGS = VEC_COLS / VEC_COLS_PER_LDG <= 1024 (avoid register spill)

  auto validate = [&](int test_bpl, int test_wm) -> bool {
    int num_elts = test_bpl / input_size;
    if (num_elts <= 0 || C % num_elts != 0) return false;
    int vec_cols = C / num_elts;
    int vec_cols_per_ldg = cpr * warps_n * 32;
    if (vec_cols_per_ldg <= 0 || vec_cols % vec_cols_per_ldg != 0) return false;
    int ldgs = vec_cols / vec_cols_per_ldg;
    if (ldgs > 1024) return false;  // reject extreme LDGS to avoid register spilling
    // Check WARPS_M constraint: if WARPS_M > 1, rows per CTA must divide evenly
    if (test_wm > 1 && num_tokens % test_wm != 0) return false;
    return true;
  };

  // Try default BPL=4, then cascade through {4, 8, 16, 2}
  static constexpr int bpl_candidates[] = {4, 8, 16, 2};
  bool found = false;
  for (int candidate : bpl_candidates) {
    if (validate(candidate, warps_m)) {
      bpl = candidate;
      found = true;
      break;
    }
  }

  // If WARPS_M=1 failed, try bumping to WARPS_M=4 for better row coverage
  if (!found && warps_m == 1 && num_tokens % 4 == 0) {
    warps_m = 4;
    for (int candidate : bpl_candidates) {
      if (validate(candidate, warps_m)) {
        bpl = candidate;
        found = true;
        break;
      }
    }
  }

  if (!found) return false;

  out.warps_m = static_cast<uint8_t>(warps_m);
  out.split_cols = 0;  // no column splitting
  out.kernel_cfg = static_cast<uint8_t>(kcfg);
  out.occupancy = static_cast<uint8_t>(occ);
  out.bytes_per_ldg = static_cast<uint8_t>(bpl);
  return true;
}

// Look up knob configuration for a given (C, num_tokens, output_dtype, sm_version).
// Tier 1: exact LUT match for SM100 VAE problem sizes (optimal, sweep-tuned on B200).
// Tier 2: fallback heuristic for other archs or arbitrary sizes (functional, conservative).
// Returns nullptr only if the problem is fundamentally unsupported.
inline const RmsNormSiluKnobs* lookup_rms_norm_silu_knobs(int C, int num_tokens,
                                                          RmsNormSiluDtype dtype,
                                                          int sm_version = 100) {
  // Tier 1: exact LUT match — only valid for SM100 (swept on B200)
  if (sm_version >= 100) {
    int c_idx = -1, t_idx = -1;
    for (int i = 0; i < kNumC; ++i) {
      if (kSupportedC[i] == C) {
        c_idx = i;
        break;
      }
    }
    for (int i = 0; i < kNumTokens; ++i) {
      if (kSupportedTokens[i] == num_tokens) {
        t_idx = i;
        break;
      }
    }
    if (c_idx >= 0 && t_idx >= 0) {
      return &knob_lut[c_idx][t_idx][static_cast<int>(dtype)];
    }
  }

  // Tier 2: fallback heuristic for non-SM100 archs or non-LUT problem sizes
  static thread_local RmsNormSiluKnobs fallback;
  if (compute_default_knobs(C, num_tokens, dtype, fallback)) {
    return &fallback;
  }

  return nullptr;  // fundamentally unsupported (C not divisible by any valid config)
}

}  // namespace norm
}  // namespace flashinfer
