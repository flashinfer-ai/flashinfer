// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpu_iface/mma_types.hpp"
#include "gpu_iface/platform.hpp"

namespace
{
using f16 = _Float16;
using f16x4 = f16 __attribute__((ext_vector_type(4)));
using f32x4 = float __attribute__((ext_vector_type(4)));

template <typename T>
__device__ __forceinline__ f32x4 mfma_fp32_16x16x16fp16(f32x4 C,
                                                        const f16x4 A,
                                                        const f16x4 B)
{
    if constexpr (std::is_same_v<T, __half>) {
        return __builtin_amdgcn_mfma_f32_16x16x16f16(A, B, C, 0, 0, 0);
    }
    else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
        return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(A, B, C, 0, 0, 0);
    }
    return C;
}

} // namespace

namespace flashinfer
{
namespace gpu_iface
{
namespace mma_impl
{
namespace hip
{

#define FLASHINFER_RUNTIME_ASSERT(x) assert(0 && x)

__device__ __forceinline__ void transpose_4x4_half_registers(uint32_t *R)
{
    // Calculate lane within 4-thread group
    uint32_t lane_id = threadIdx.x % 64;
    uint32_t lane_in_group = lane_id % 4;

    // === ROUND 1: Exchange with neighbor (XOR with 1) ===
    // T0↔T1, T2↔T3 partial exchange
    uint32_t reg_idx = (lane_in_group >> 1) & 0x1;
    uint32_t exchanged_val = __shfl_xor(R[reg_idx], 0x1);
    uint32_t shift = (lane_in_group & 1) * 16;
    uint32_t keep_mask = 0xFFFF0000 >> shift;
    int right_shift_amount = 16 * (1 - (lane_in_group & 1));
    int left_shift_amount = 16 * (lane_in_group & 1);
    R[reg_idx] = (R[reg_idx] & keep_mask) |
                 ((exchanged_val >> right_shift_amount) << left_shift_amount);

    // === ROUND 2: Exchange with one hop (XOR with 2) ===
    // T0↔T2, T1↔T3 exchange R[0] and R[1]
    // Swap entire registers based on thread position
    uint32_t is_top = 1 - reg_idx;
    uint32_t temp0 = __shfl_xor(R[0], 0x2);
    uint32_t temp1 = __shfl_xor(R[1], 0x2);

    // Compute both possibilities and select
    R[0] = R[0] * is_top + temp1 * reg_idx;
    R[1] = temp0 * is_top + R[1] * reg_idx;

    // === ROUND 3: Exchange with neighbor again (XOR with 1) ===
    // T0↔T1, T2↔T3 exchange remaining parts

    reg_idx = 1 - reg_idx;
    exchanged_val = __shfl_xor(R[reg_idx], 0x1);
    R[reg_idx] = (R[reg_idx] & keep_mask) |
                 ((exchanged_val >> right_shift_amount) << left_shift_amount);
}

// Single unified load function for all fragment types
/// @param R [in] pointer to the register file to load the fragment into
/// @param smem_ptr [in] pointer to the shared memory to load the fragment from
template <typename T>
__device__ __forceinline__ void load_fragment(uint32_t *R, const T *smem_ptr)
{
    const uint16_t *v0 = reinterpret_cast<const uint16_t *>(smem_ptr) + 0;
    const uint16_t *v1 = reinterpret_cast<const uint16_t *>(++smem_ptr);
    const uint16_t *v2 = reinterpret_cast<const uint16_t *>(++smem_ptr);
    const uint16_t *v3 = reinterpret_cast<const uint16_t *>(++smem_ptr);

    R[0] = (static_cast<const uint32_t>(*v0) << 16) |
           static_cast<const uint32_t>(*v1);
    R[1] = (static_cast<const uint32_t>(*v2) << 16) |
           static_cast<const uint32_t>(*v3);
}

template <typename T>
__device__ __forceinline__ void
load_fragment_transpose(uint32_t *R, const T *smem_ptr, uint32_t stride)
{
    const uint16_t *v0 = reinterpret_cast<const uint16_t *>(smem_ptr) + 0;
    const uint16_t *v1 =
        reinterpret_cast<const uint16_t *>(smem_ptr + 1 * stride);
    const uint16_t *v2 =
        reinterpret_cast<const uint16_t *>(smem_ptr + 2 * stride);
    const uint16_t *v3 =
        reinterpret_cast<const uint16_t *>(smem_ptr + 3 * stride);

    R[0] = (static_cast<const uint32_t>(*v0) << 16) |
           static_cast<const uint32_t>(*v1);
    R[1] = (static_cast<const uint32_t>(*v2) << 16) |
           static_cast<const uint32_t>(*v3);
}

// MMA operation for FP16 inputs with FP32 accumulator
template <typename T, mma::MMAMode mma_mode = mma::MMAMode::kInplaceUpdate>
__device__ __forceinline__ void
mma_sync_m16n16k16_row_col_f16f16f32(float *C, uint32_t *A, uint32_t *B)
{
    // Ensure T is either __half or __hip_bfloat16
    static_assert(std::is_same_v<T, __half> ||
                      std::is_same_v<T, __hip_bfloat16>,
                  "T must be __half or __hip_bfloat16");

    // Initialize C if requested
    if constexpr (mma_mode == mma::MMAMode::kInit) {
        C[0] = 0.0f;
        C[1] = 0.0f;
        C[2] = 0.0f;
        C[3] = 0.0f;
    }

    f16x4 B_fp16 = reinterpret_cast<f16x4 *>(B)[0];
    f16x4 A_fp16 = reinterpret_cast<f16x4 *>(A)[0];
    f32x4 C_fp32 = reinterpret_cast<f32x4 *>(C)[0];

    // Perform MMA operation directly with fragments
    C_fp32 = mfma_fp32_16x16x16fp16<T>(C_fp32, A_fp16, B_fp16);
    C[0] = C_fp32[0];
    C[1] = C_fp32[1];
    C[2] = C_fp32[2];
    C[3] = C_fp32[3];
}

/// Loads a fragment from LDS to two 32bit registers and then transposes
/// the registers for a group of four consecuitive threads.
template <typename T>
__device__ __forceinline__ void
load_fragment_4x4_half_registers(uint32_t *R, const T *smem_ptr)
{
    static_assert(std::is_same_v<T, __half>(), "Only half type is supported");
    // Each thread loads 4 __half values in two 32b registers.
    load_fragment(R, smem_ptr);
    // transposes the values in four adjacent threads. The function does the
    // following layout transformation:
    // Original data in registers for Threads 0-3 after fragment load
    // T0 : a b c d
    // T1 : e f g h
    // T2 : i j k l
    // T3 : m n o p
    //
    // After transposition:
    // T0 : a e i m
    // T1 : b f j n
    // T2 : c g k o
    // T3 : d h l p

    transpose_4x4_half_registers(R);
}

template <typename DType>
__device__ __forceinline__ void m16k16_rowsum_f16f16f32(float *d, DType *s_frag)
{
    static_assert(sizeof(DType) == 2, "DType must be 16-bit type");
    f16x4 a = reinterpret_cast<const f16x4 *>(s_frag)[0];
    f16x4 b = {f16(1.0f), f16(1.0f), f16(1.0f), f16(1.0f)};
    f32x4 c = {0.f, 0.f, 0.f, 0.f};
    f32x4 out = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
    d[0] = out.x;
    d[1] = out.y;
    d[2] = out.z;
    d[3] = out.w;
}

// TODO (rimaddur) : After release 2025.08
// FP8 operations - not implemented for MI300 yet
template <typename T>
__device__ __forceinline__ void
mma_sync_m16n16k32_row_col_f8f8f32(float *c_frag, T *a_frag, T *b_frag)
{
    FLASHINFER_RUNTIME_ASSERT("FP8 MMA not implemented for AMD");
}

template <typename DType>
__device__ __forceinline__ void m16k32_rowsum_f8f8f32(float *d_frag,
                                                      DType *s_frag)
{
    FLASHINFER_RUNTIME_ASSERT("FP8 rowsum not implemented for AMD");
}

} // namespace hip
} // namespace mma_impl
} // namespace gpu_iface
} // namespace flashinfer
