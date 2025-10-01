// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpu_iface/mma_types.hpp"
#include "gpu_iface/platform.hpp"

namespace {
using f16 = _Float16;
using f16x4 = f16 __attribute__((ext_vector_type(4)));
using f32x4 = float __attribute__((ext_vector_type(4)));

}  // namespace

namespace flashinfer {
namespace gpu_iface {
namespace mma_impl {
namespace hip {

#define FLASHINFER_RUNTIME_ASSERT(x) assert(0 && x)

__device__ __forceinline__ void transpose_4x4_half_registers(uint32_t* R) {
  // Calculate lane within 4-thread group
  uint32_t lane_id = threadIdx.x % 64;
  uint32_t lane_in_group = lane_id % 4;

  // === ROUND 1: Exchange with neighbor (XOR with 1) ===
  // T0 <-> T1, T2 <-> T3 partial exchange
  uint32_t regid = (lane_in_group >> 1) & 0x1;
  uint32_t exchanged_val = __shfl_xor(R[regid], 0x1);
  uint32_t shift = (lane_in_group & 1) * 16;
  uint32_t keep_mask = 0x0000FFFF << shift;
  int left_shift_amount = 16 * (1 - (lane_in_group & 1));
  int right_shift_amount = 16 * (lane_in_group & 1);
  R[regid] = (R[regid] & keep_mask) | ((exchanged_val >> right_shift_amount) << left_shift_amount);

  // === ROUND 2: Exchange with one hop (XOR with 2) ===
  // T0 <-> T2, T1 <-> T3 exchange R[0] and R[1]
  // Swap entire registers based on thread position
  uint32_t is_top = 1 - regid;
  uint32_t temp0 = __shfl_xor(R[0], 0x2);
  uint32_t temp1 = __shfl_xor(R[1], 0x2);

  // Compute both possibilities and select
  R[0] = R[0] * is_top + temp1 * regid;
  R[1] = temp0 * is_top + R[1] * regid;

  // === ROUND 3: Exchange with neighbor again (XOR with 1) ===
  // T0 <-> T1, T2 <-> T3 exchange remaining parts

  regid = 1 - regid;
  exchanged_val = __shfl_xor(R[regid], 0x1);
  R[regid] = (R[regid] & keep_mask) | ((exchanged_val >> right_shift_amount) << left_shift_amount);
}

// Single unified load function for all fragment types
/// @param R [in] pointer to the register file to load the fragment into
/// @param smem_ptr [in] pointer to the shared memory to load the fragment from
template <typename T>
__device__ __forceinline__ void load_fragment(uint32_t* R, const T* smem_ptr) {
  R[0] = reinterpret_cast<const uint32_t*>(smem_ptr)[0];
  R[1] = reinterpret_cast<const uint32_t*>(smem_ptr)[1];
}

// MMA operation for FP16 inputs with FP32 accumulator
template <typename T, mma::MMAMode mma_mode = mma::MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(float* C, uint32_t* A,
                                                                     uint32_t* B) {
#if defined(__HIP_DEVICE_COMPILE__) && (__gfx90a__ || __gfx908__ || __gfx942__)
  // Ensure T is either __half or __hip_bfloat16
  static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __hip_bfloat16>,
                "T must be __half or __hip_bfloat16");

  // Initialize C if requested
  if constexpr (mma_mode == mma::MMAMode::kInit) {
    C[0] = 0.0f;
    C[1] = 0.0f;
    C[2] = 0.0f;
    C[3] = 0.0f;
  }

  f16x4 B_fp16 = reinterpret_cast<f16x4*>(B)[0];
  f16x4 A_fp16 = reinterpret_cast<f16x4*>(A)[0];
  f32x4 C_fp32 = reinterpret_cast<f32x4*>(C)[0];

  // Perform MMA operation directly with fragments

  if constexpr (std::is_same_v<T, __half>) {
    C_fp32 = __builtin_amdgcn_mfma_f32_16x16x16f16(A_fp16, B_fp16, C_fp32, 0, 0, 0);
  } else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
    C_fp32 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(A_fp16, B_fp16, C_fp32, 0, 0, 0);
  }

  reinterpret_cast<f32x4*>(C)[0] = C_fp32;
#elif defined(__HIP_DEVICE_COMPILE__)
#error "Unsupported GFX platform for MFMA ops."
#endif
}

/// @brief Loads a fragment from LDS to two 32bit registers and then transposes
/// the registers for a group of four consecuitive threads.
///
/// transposes the values in four adjacent threads. The function does the
/// following layout transformation:
/// Original data in registers for Threads 0-3 after fragment load
/// T0 : a b c d
/// T1 : e f g h
/// T2 : i j k l
/// T3 : m n o p
///
/// After transposition:
/// T0 : a e i m
/// T1 : b f j n
/// T2 : c g k o
/// T3 : d h l p
template <typename T>
__device__ __forceinline__ void load_fragment_4x4_half_registers(uint32_t* R, const T* smem_ptr) {
  static_assert(std::is_same_v<T, __half>, "Only half type is supported");
  load_fragment(R, smem_ptr);
  transpose_4x4_half_registers(R);
}

// TODO: Verify correct matrix multiplication order for rowsum on CDNA3
// Current assumption: s_frag × ones_vector = row_sums
// Need to validate:
// 1. How compute_qk stores Q×K^T result in s_frag for CDNA3
// 2. Whether K is pre-transposed or transposed during fragment loading
// 3. If we need s_frag × M1 or M1 × s_frag for correct row sums
//
// Test with known input matrices to verify:
// - s_frag layout matches expected Q×K^T result
// - rowsum produces correct per-row sums
template <typename DType>
__device__ __forceinline__ void m16k16_rowsum_f16f16f32(float* d, DType* s_frag) {
  static_assert(sizeof(DType) == 2, "DType must be 16-bit type");
  transpose_4x4_half_registers(reinterpret_cast<uint32_t*>(s_frag));
  f16x4 a = reinterpret_cast<const f16x4*>(s_frag)[0];
  f16x4 b = {f16(1.0f), f16(1.0f), f16(1.0f), f16(1.0f)};
  f32x4 c = {d[0], d[1], d[2], d[3]};
  f32x4 out = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
  d[0] = out.x;
  d[1] = out.y;
  d[2] = out.z;
  d[3] = out.w;
}

// TODO (rimaddur) : After release 2025.08
// FP8 operations - not implemented for MI300 yet
template <typename T>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_f8f8f32(float* c_frag, T* a_frag,
                                                                   T* b_frag) {
  FLASHINFER_RUNTIME_ASSERT("FP8 MMA not implemented for AMD");
}

template <typename DType>
__device__ __forceinline__ void m16k32_rowsum_f8f8f32(float* d_frag, DType* s_frag) {
  FLASHINFER_RUNTIME_ASSERT("FP8 rowsum not implemented for AMD");
}

}  // namespace hip
}  // namespace mma_impl
}  // namespace gpu_iface
}  // namespace flashinfer
