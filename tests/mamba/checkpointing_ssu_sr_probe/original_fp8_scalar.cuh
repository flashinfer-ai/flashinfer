// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Derived from flashinfer-ai/flashinfer include/flashinfer/mamba/conversion.cuh
// at d84335d8c54e0c8f9f7e22faa15c7e34a58abf94; upstream file has no individual
// copyright notice. Original scalar body and its existing provenance are retained.

// SPDX-License-Identifier: Apache-2.0
// Frozen original FlashInfer scalar e4m3 SR at d84335d8; independent oracle.
// Unchanged original generic converter body, namespaced device oracle.
// clang-format on
namespace flashinfer_test_fp8_original {
__device__ __forceinline__ uint8_t cvt_rs_e4m3_sw(float x, uint32_t rand16) {
  uint32_t bits = __float_as_uint(x);
  uint32_t sign = (bits >> 31) & 1u;
  uint32_t abs_bits = bits & 0x7FFFFFFFu;
  uint32_t f32_exp = (abs_bits >> 23) & 0xFFu;
  uint32_t f32_mant = abs_bits & 0x7FFFFFu;

  // NaN / Inf
  if (f32_exp == 0xFFu) {
    if (f32_mant != 0) {
      return static_cast<uint8_t>(0x7Fu | (sign << 7));  // canonical e4m3 NaN
    } else {
      return static_cast<uint8_t>(0x7Eu | (sign << 7));  // Inf → ±max finite
    }
  }

  // fp32 zero / denormal → e4m3 zero (with sign).
  if (f32_exp == 0u) {
    return static_cast<uint8_t>(sign << 7);
  }

  int unbiased = static_cast<int>(f32_exp) - 127;
  uint64_t mant24 = 0x800000u | f32_mant;  // implicit-1 + mantissa, 24-bit
  int shift_truncate = (unbiased >= -6) ? 20 : (14 - unbiased);
  int rand_shift = shift_truncate - 16;
  uint64_t rand_contrib;
  if (rand_shift < 0) {
    // Defensive: shift_truncate < 16 shouldn't happen for valid normal/subnormal e4m3.
    rand_contrib = static_cast<uint64_t>(rand16 & 0xFFFFu) >> (-rand_shift);
  } else if (rand_shift < 56) {
    rand_contrib = static_cast<uint64_t>(rand16 & 0xFFFFu) << rand_shift;
  } else {
    rand_contrib = 0;
  }
  uint64_t total = mant24 + rand_contrib;
  // Guard the shift: shift_truncate reaches 140 for tiny-normal fp32
  // (unbiased < -49), which is UB for a uint64_t shift.  Mathematically the
  // result is 0 there (the value is far below e4m3's smallest subnormal),
  // so flush to int_part = 0.  The downstream subnormal branch rounds it to ±0.
  uint32_t int_part = (shift_truncate >= 64) ? 0u : static_cast<uint32_t>(total >> shift_truncate);

  if (unbiased >= -6) {
    // Started in normal binade.  int_part ∈ [8, 15] normally; can overflow to 16+ if rand
    // bumped the exponent.
    int e4m3_exp = unbiased + 7;
    while (int_part >= 16u) {
      int_part >>= 1;
      e4m3_exp += 1;
    }
    if (e4m3_exp > 15 || (e4m3_exp == 15 && (int_part & 0x7u) == 7u)) {
      return static_cast<uint8_t>(0x7Eu | (sign << 7));
    }
    return static_cast<uint8_t>((sign << 7) | (e4m3_exp << 3) | (int_part & 0x7u));
  } else {
    // Started in subnormal/underflow.  int_part:
    //   0       → zero
    //   1..7    → subnormal e4m3
    //   8..15   → smallest normal binade (e4m3_exp = 1)
    //   16+     → higher normal binades (rare; only if rand pushed up multiple binades)
    if (int_part == 0u) {
      return static_cast<uint8_t>(sign << 7);
    }
    if (int_part <= 7u) {
      return static_cast<uint8_t>((sign << 7) | int_part);
    }
    int e4m3_exp = 1;
    while (int_part >= 16u) {
      int_part >>= 1;
      e4m3_exp += 1;
    }
    if (e4m3_exp > 15 || (e4m3_exp == 15 && (int_part & 0x7u) == 7u)) {
      return static_cast<uint8_t>(0x7Eu | (sign << 7));
    }
    return static_cast<uint8_t>((sign << 7) | (e4m3_exp << 3) | (int_part & 0x7u));
  }
}

}  // namespace flashinfer_test_fp8_original
