// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#ifndef W4A8_DECODE_VECTOR
#define W4A8_DECODE_VECTOR 1
#endif

#if W4A8_DECODE_VECTOR != 0 && W4A8_DECODE_VECTOR != 1
#error "W4A8_DECODE_VECTOR must be 0 or 1"
#endif

namespace flashinfer {
namespace sm90_w4a8 {

constexpr int kV3PayloadTileK = 32;
constexpr int kV3PayloadTileN = 64;
constexpr int kV3PackedBytesPerRow = kV3PayloadTileK / 2;
constexpr int kV3ResidualBlockK = 16;
constexpr int kV3ResidualsPerPayloadTile = kV3PayloadTileK / kV3ResidualBlockK;

enum class ResidualScheme : uint8_t { kGeneric, kPow2 };

// E2M1 magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6}.  Constructing
// the BF16 encoding avoids an intermediate lookup load in the producer.
__device__ __forceinline__ uint16_t e2m1_bf16_bits(uint8_t code) {
  const uint32_t magnitude = static_cast<uint32_t>(code) & 7U;
  const uint16_t value =
      magnitude == 0U
          ? 0U
          : (magnitude == 1U ? 0x3f00U : static_cast<uint16_t>(0x3f80U + (magnitude - 2U) * 0x40U));
  return static_cast<uint16_t>(value | ((static_cast<uint32_t>(code) & 8U) << 12));
}

__device__ __forceinline__ float decode_e2m1(uint8_t code) {
  return __bfloat162float(__ushort_as_bfloat16(e2m1_bf16_bits(code)));
}

__device__ __forceinline__ uint8_t encode_e4m3fn_rne(float value) {
  // fmaxf/fminf also give the v3 conversion contract for a NaN input:
  // fmaxf(NaN, -448) is -448, which is then representable without overflow.
  const float finite = fminf(fmaxf(value, -448.0F), 448.0F);
  const __nv_fp8_e4m3 encoded(finite);
  return static_cast<uint8_t>(encoded.__x);
}

template <ResidualScheme Scheme>
struct ResidualDecoder;

template <>
struct ResidualDecoder<ResidualScheme::kGeneric> {
  using Storage = __nv_bfloat16;

  __device__ __forceinline__ static uint8_t promote(uint8_t code, Storage residual) {
    // The multiplication is performed in FP32 and rounded exactly once when
    // converted to finite E4M3, matching the v3 generic operand contract.
    return encode_e4m3fn_rne(decode_e2m1(code) * __bfloat162float(residual));
  }
};

template <>
struct ResidualDecoder<ResidualScheme::kPow2> {
  using Storage = int8_t;

  __device__ __forceinline__ static uint8_t promote(uint8_t code, Storage exponent) {
    const uint8_t sign = static_cast<uint8_t>((code & 0x08U) << 4);
    const uint8_t magnitude = code & 0x07U;
    if (exponent == INT8_MIN) {
      // Multiplication by the positive-zero sentinel preserves the E2M1 sign.
      return sign;
    }
    if (magnitude == 0U) {
      return sign;
    }

    // Every nonzero E2M1 value is 1.0 or 1.5 times a power of two.  Its E4M3
    // mantissa is therefore exact whenever the shifted exponent is normal.
    const int base_exponent = static_cast<int>(magnitude >> 1) - 1;
    const uint8_t mantissa = magnitude > 1U && (magnitude & 1U) != 0U ? 4U : 0U;
    const int shifted_exponent = base_exponent + static_cast<int>(exponent);
    if (shifted_exponent >= -6) {
      if (shifted_exponent > 8) {
        return static_cast<uint8_t>(sign | 0x7eU);
      }
      const uint8_t exponent_field = static_cast<uint8_t>(shifted_exponent + 7);
      return static_cast<uint8_t>(sign | (exponent_field << 3) | mantissa);
    }

    // Subnormals are integer multiples of 2^-9.  Round the exact shifted
    // significand to that grid with round-to-nearest, ties-to-even.
    const int shift = -(shifted_exponent + 6);
    if (shift >= 5) {
      return sign;
    }
    const uint8_t significand = static_cast<uint8_t>(8U + mantissa);
    uint8_t rounded = static_cast<uint8_t>(significand >> shift);
    const uint8_t mask = static_cast<uint8_t>((1U << shift) - 1U);
    const uint8_t remainder = significand & mask;
    const uint8_t halfway = static_cast<uint8_t>(1U << (shift - 1));
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U)) {
      ++rounded;
    }
    return static_cast<uint8_t>(sign | rounded);
  }
};

template <ResidualScheme Scheme>
__device__ __forceinline__ void decode_packed_pair(
    uint8_t packed, typename ResidualDecoder<Scheme>::Storage residual, uint8_t& even,
    uint8_t& odd) {
  even = ResidualDecoder<Scheme>::promote(packed & 0x0fU, residual);
  odd = ResidualDecoder<Scheme>::promote((packed >> 4) & 0x0fU, residual);
}

template <ResidualScheme Scheme>
__device__ __forceinline__ uint16_t
decode_packed_pair_u16(uint8_t packed, typename ResidualDecoder<Scheme>::Storage residual) {
  uint8_t even;
  uint8_t odd;
  decode_packed_pair<Scheme>(packed, residual, even, odd);
  return static_cast<uint16_t>(even) | (static_cast<uint16_t>(odd) << 8);
}

template <ResidualScheme Scheme>
__device__ __forceinline__ uint32_t
decode_two_packed_bytes(uint16_t packed, typename ResidualDecoder<Scheme>::Storage residual) {
  const uint16_t decoded0 = decode_packed_pair_u16<Scheme>(static_cast<uint8_t>(packed), residual);
  const uint16_t decoded1 =
      decode_packed_pair_u16<Scheme>(static_cast<uint8_t>(packed >> 8), residual);
  return static_cast<uint32_t>(decoded0) | (static_cast<uint32_t>(decoded1) << 16);
}

template <ResidualScheme Scheme>
__device__ __forceinline__ void run_scalar_task(
    const uint8_t* raw_payload, uint8_t* decoded_weight,
    typename ResidualDecoder<Scheme>::Storage residual) {
#pragma unroll
  for (int pair = 0; pair < kV3ResidualBlockK / 2; ++pair) {
    uint8_t even;
    uint8_t odd;
    decode_packed_pair<Scheme>(raw_payload[pair], residual, even, odd);
    decoded_weight[pair * 2] = even;
    decoded_weight[pair * 2 + 1] = odd;
  }
}

template <ResidualScheme Scheme>
__device__ __forceinline__ void run_vector_task(
    const uint8_t* raw_payload, uint8_t* decoded_weight0, uint8_t* decoded_weight1,
    typename ResidualDecoder<Scheme>::Storage residual0,
    typename ResidualDecoder<Scheme>::Storage residual1) {
  const uint4 packed = *reinterpret_cast<const uint4*>(raw_payload);
  const uint4 decoded0{
      decode_two_packed_bytes<Scheme>(static_cast<uint16_t>(packed.x), residual0),
      decode_two_packed_bytes<Scheme>(static_cast<uint16_t>(packed.x >> 16), residual0),
      decode_two_packed_bytes<Scheme>(static_cast<uint16_t>(packed.y), residual0),
      decode_two_packed_bytes<Scheme>(static_cast<uint16_t>(packed.y >> 16), residual0)};
  const uint4 decoded1{
      decode_two_packed_bytes<Scheme>(static_cast<uint16_t>(packed.z), residual1),
      decode_two_packed_bytes<Scheme>(static_cast<uint16_t>(packed.z >> 16), residual1),
      decode_two_packed_bytes<Scheme>(static_cast<uint16_t>(packed.w), residual1),
      decode_two_packed_bytes<Scheme>(static_cast<uint16_t>(packed.w >> 16), residual1)};
  *reinterpret_cast<uint4*>(decoded_weight0) = decoded0;
  *reinterpret_cast<uint4*>(decoded_weight1) = decoded1;
}

// v3 tensors are standard C-contiguous global-memory arrays.  These helpers
// state the byte/scalar coordinates consumed by both the production decode and
// the operand-byte debug kernel.
__host__ __device__ constexpr int64_t v3_payload_offset(int32_t expert, int32_t k_tile,
                                                        int32_t n_tile, int32_t n_in_tile,
                                                        int32_t packed_k, int32_t k_tiles,
                                                        int32_t n_tiles) {
  return (((static_cast<int64_t>(expert) * k_tiles + k_tile) * n_tiles + n_tile) * kV3PayloadTileN +
          n_in_tile) *
             kV3PackedBytesPerRow +
         packed_k;
}

__host__ __device__ constexpr int64_t v3_residual_offset(int32_t expert, int32_t k_tile,
                                                         int32_t n_tile, int32_t n_in_tile,
                                                         int32_t residual_k, int32_t k_tiles,
                                                         int32_t n_tiles) {
  return (((static_cast<int64_t>(expert) * k_tiles + k_tile) * n_tiles + n_tile) * kV3PayloadTileN +
          n_in_tile) *
             kV3ResidualsPerPayloadTile +
         residual_k;
}

__host__ __device__ constexpr int64_t v3_group_scale_offset(int32_t expert, int32_t k_group,
                                                            int32_t n_tile, int32_t n_in_tile,
                                                            int32_t k_groups, int32_t n_tiles) {
  return (((static_cast<int64_t>(expert) * k_groups + k_group) * n_tiles + n_tile) *
              kV3PayloadTileN +
          n_in_tile);
}

__host__ __device__ constexpr int64_t v3_debug_operand_offset(int32_t expert, int32_t k_tile,
                                                              int32_t n_tile, int32_t n_in_tile,
                                                              int32_t k_in_tile, int32_t k_tiles,
                                                              int32_t n_tiles) {
  return (((static_cast<int64_t>(expert) * k_tiles + k_tile) * n_tiles + n_tile) * kV3PayloadTileN +
          n_in_tile) *
             kV3PayloadTileK +
         k_in_tile;
}

// A WGMMA operand row contains 128 bytes.  The 128B swizzle XORs its 16-byte
// chunk with the low three row bits; the byte position inside a chunk is kept.
__host__ __device__ constexpr int32_t wgmma_swizzle_128b_offset(int32_t row, int32_t k) {
  return row * 128 + (((k >> 4) ^ (row & 7)) << 4) + (k & 15);
}

}  // namespace sm90_w4a8
}  // namespace flashinfer
