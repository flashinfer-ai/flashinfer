#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#ifdef FLASHINFER_ENABLE_BF16
#include <cuda_bf16.h>
#endif
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

namespace flashinfer::mamba::conversion {

inline __device__ float toFloat(float f) { return f; }

inline __device__ float toFloat(__half h) { return __half2float(h); }

#ifdef FLASHINFER_ENABLE_BF16
inline __device__ float toFloat(__nv_bfloat16 val) { return __bfloat162float(val); }
#endif

// No accuracy loss: int16_t range [-32768, 32767] fits exactly in float32
// (24-bit mantissa represents all integers up to 2^24 = 16M exactly).
inline __device__ float toFloat(int16_t val) { return static_cast<float>(val); }

// Packed 2-element conversion: convert a packed pair to float2.
// Uses native packed intrinsics for bf16/fp16 (fewer PRMT/SHF instructions).
inline __device__ float2 toFloat2(float2 packed) { return packed; }

inline __device__ float2 toFloat2(__half2 packed) { return __half22float2(packed); }

// Pointer-based overloads: read two consecutive elements and convert to float2.
// Dispatches to the packed intrinsic for bf16/fp16 via the overloads above.
inline __device__ float2 toFloat2(float const* ptr) { return {ptr[0], ptr[1]}; }

inline __device__ float2 toFloat2(__half const* ptr) {
  return toFloat2(*reinterpret_cast<__half2 const*>(ptr));
}

#ifdef FLASHINFER_ENABLE_BF16
// inline __device__ float2 toFloat2(__nv_bfloat162 packed) { return __bfloat1622float2(packed); }
inline __device__ float2 toFloat2(__nv_bfloat162 packed) {
  // bf16 is the upper 16 bits of f32 — shift/mask is cheaper than PRMT byte permutation.
  // NOTE: this ignores denormals
  uint32_t bits = reinterpret_cast<uint32_t const&>(packed);
  float2 out;
  out.x = __uint_as_float(bits << 16);          // low bf16 → upper 16 bits of f32
  out.y = __uint_as_float(bits & 0xFFFF0000u);  // high bf16 already in upper 16 bits
  return out;
}

inline __device__ float2 toFloat2(__nv_bfloat16 const* ptr) {
  return toFloat2(*reinterpret_cast<__nv_bfloat162 const*>(ptr));
}
#endif

inline __device__ float2 toFloat2(int16_t const* ptr) { return {toFloat(ptr[0]), toFloat(ptr[1])}; }

inline __device__ void convertAndStore(float* output, float input) { *output = input; }

inline __device__ void convertAndStore(__half* output, float input) {
  *output = __float2half(input);
}

#ifdef FLASHINFER_ENABLE_BF16
inline __device__ void convertAndStore(__nv_bfloat16* output, float input) {
  *output = __float2bfloat16(input);
}
#endif

inline __device__ void convertAndStore(int16_t* output, float input) {
  // Symmetric clip: [-max, max] (not [-max-1, max]) so that negation is safe.
  // Matches Triton reference which clips to [-32767, 32767] before storing.
  constexpr float int16_max = static_cast<float>(std::numeric_limits<int16_t>::max());
  input = fminf(fmaxf(input, -int16_max), int16_max);
  *output = static_cast<int16_t>(__float2int_rn(input));
}

// =============================================================================
// Philox-4x32 PRNG (matches Triton's tl.randint)
// =============================================================================

// Generates four pseudorandom uint32s from (seed, offset) using the Philox-4x32 algorithm.
// Produces bit-identical output to Triton's tl.randint4x(seed, offset, n_rounds).
// All four outputs (c0..c3) are independent and uniformly distributed.
template <int n_rounds = 10>
__device__ __forceinline__ void philox_randint4x(int64_t seed, uint32_t offset, uint32_t& r0,
                                                 uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t PHILOX_KEY_A = 0x9E3779B9u;
  constexpr uint32_t PHILOX_KEY_B = 0xBB67AE85u;
  constexpr uint32_t PHILOX_ROUND_A = 0xD2511F53u;
  constexpr uint32_t PHILOX_ROUND_B = 0xCD9E8D57u;

  uint32_t k0 = static_cast<uint32_t>(static_cast<uint64_t>(seed));
  uint32_t k1 = static_cast<uint32_t>(static_cast<uint64_t>(seed) >> 32);
  uint32_t c0 = offset, c1 = 0, c2 = 0, c3 = 0;

#pragma unroll
  for (int i = 0; i < n_rounds; i++) {
    uint32_t _c0 = c0, _c2 = c2;
    c0 = __umulhi(PHILOX_ROUND_B, _c2) ^ c1 ^ k0;
    c2 = __umulhi(PHILOX_ROUND_A, _c0) ^ c3 ^ k1;
    c1 = PHILOX_ROUND_B * _c2;
    c3 = PHILOX_ROUND_A * _c0;
    k0 += PHILOX_KEY_A;
    k1 += PHILOX_KEY_B;
  }
  r0 = c0;
  r1 = c1;
  r2 = c2;
  r3 = c3;
}

// Generates a pseudorandom uint32 from (seed, offset) using the Philox-4x32 algorithm.
// Produces bit-identical output to Triton's tl.randint(seed, offset, n_rounds).
// NOTE: This discards 3 of the 4 Philox outputs. For better throughput, use
// philox_randint4x to get all 4 outputs from a single Philox invocation.
template <int n_rounds = 10>
__device__ __forceinline__ uint32_t philox_randint(int64_t seed, uint32_t offset) {
  uint32_t r0, r1, r2, r3;
  philox_randint4x<n_rounds>(seed, offset, r0, r1, r2, r3);
  return r0;
}

// =============================================================================
// Stochastic rounding: fp32 → fp16
// =============================================================================

// Software stochastic rounding: convert one fp32 value to fp16 using 13 random bits.
// Adds random noise at the sub-fp16-mantissa position, then truncates.
// rand13: 13-bit random value in bits [12:0].
__device__ __forceinline__ uint16_t cvt_rs_f16_sw(float x, uint32_t rand13) {
  uint32_t bits = __float_as_uint(x);
  uint32_t sign = bits & 0x80000000u;
  uint32_t abs_bits = bits & 0x7FFFFFFFu;

  // fp32 has 23 mantissa bits, fp16 has 10. The 13 LSBs are the remainder.
  // Add 13-bit random noise at bits [12:0]. Carry into bit 13 → round up.
  abs_bits += (rand13 & 0x1FFFu);

  // Convert to fp16 by truncation.
  uint32_t f32_exp = (abs_bits >> 23) & 0xFFu;
  uint32_t f32_mantissa = abs_bits & 0x7FFFFFu;

  uint16_t f16_bits;
  if (f32_exp == 0xFF) {
    f16_bits = (f32_mantissa != 0) ? 0x7E00u : 0x7C00u;  // NaN or Inf
  } else if (f32_exp > 142) {                            // 127 + 15 = 142 → overflow to Inf
    f16_bits = 0x7C00u;
  } else if (f32_exp < 113) {  // 127 - 14 = 113 → underflow to zero
    f16_bits = 0;
  } else {
    uint16_t f16_exp = static_cast<uint16_t>(f32_exp - 112);  // rebias: 127→15
    uint16_t f16_mantissa = static_cast<uint16_t>(f32_mantissa >> 13);
    f16_bits = (f16_exp << 10) | f16_mantissa;
  }

  return static_cast<uint16_t>(sign >> 16) | f16_bits;
}

// Forward declaration (defined below, after cvt_rs_f16x2_f32).
__device__ __forceinline__ uint32_t cvt_rs_f16x2_f32(float a, float b, uint32_t rbits);

// Stochastic rounding: convert one fp32 value to fp16 using 13 random bits.
// On sm_100a+: uses PTX cvt.rs.f16x2.f32 with a dummy zero second input.
// On other archs: software emulation.
__device__ __forceinline__ __half cvt_rs_f16_f32(float x, uint32_t rand13) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL)
  // Pack rand13 into rbits[12:0] (for PTX operand b → low half → our x).
  // High half gets zero noise for the dummy input.
  uint32_t rbits = rand13 & 0x1FFFu;
  uint32_t packed = cvt_rs_f16x2_f32(x, 0.0f, rbits);
  return __ushort_as_half(static_cast<uint16_t>(packed & 0xFFFFu));
#else
  return __ushort_as_half(cvt_rs_f16_sw(x, rand13));
#endif
}

// Stochastic rounding: convert two fp32 values to packed fp16x2 using random bits.
// On sm_100a+: uses PTX cvt.rs.f16x2.f32 instruction.
// On other archs: software emulation matching the hardware behavior.
//
// rbits layout (from PTX docs):
//   bits [28:16] = 13 random bits for PTX operand "a" (→ d[31:16], high half)
//   bits [12:0]  = 13 random bits for PTX operand "b" (→ d[15:0], low half)
//   bits [31:29] and [15:13] = unused (zero)
// from: https://docs.nvidia.com/cuda/parallel-thread-execution/#cvt-rs-rbits-layout-f16
//
// Our asm maps: %1→C++ a→PTX b→d[15:0], %2→C++ b→PTX a→d[31:16]
// So: C++ a uses rbits[12:0], C++ b uses rbits[28:16].
__device__ __forceinline__ uint32_t cvt_rs_f16x2_f32(float a, float b, uint32_t rbits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL)
  uint32_t packed;
  asm("cvt.rs.f16x2.f32 %0, %2, %1, %3;"
      : "=r"(packed)
      : "r"(__float_as_uint(a)), "r"(__float_as_uint(b)), "r"(rbits));
  return packed;
#else
  uint32_t rand_a = rbits & 0x1FFFu;          // bits [12:0] → C++ a (PTX b → low half)
  uint32_t rand_b = (rbits >> 16) & 0x1FFFu;  // bits [28:16] → C++ b (PTX a → high half)
  uint16_t a_fp16 = __half_as_ushort(cvt_rs_f16_f32(a, rand_a));
  uint16_t b_fp16 = __half_as_ushort(cvt_rs_f16_f32(b, rand_b));
  return static_cast<uint32_t>(a_fp16) | (static_cast<uint32_t>(b_fp16) << 16);
#endif
}

// Stochastic rounding store: generates Philox random bits and converts fp32 → fp16 in one call.
// PHILOX_ROUNDS: number of Philox rounds (compile-time), must be > 0.
// seed: Philox seed (from params.rand_seed).
// offset: unique per-element offset (e.g. d * DSTATE + i) for deterministic randomness.
template <int PHILOX_ROUNDS>
inline __device__ void convertSRAndStore(__half* output, float input, int64_t seed,
                                         uint32_t offset) {
  uint32_t rand = philox_randint<PHILOX_ROUNDS>(seed, offset);
  *output = cvt_rs_f16_f32(input, rand & 0x1FFFu);
}

}  // namespace flashinfer::mamba::conversion
