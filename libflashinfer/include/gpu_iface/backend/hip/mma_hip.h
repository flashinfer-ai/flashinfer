// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpu_iface/mma_types.hpp"
#include "gpu_iface/platform.hpp"

namespace
{
using float16_t = _Float16;
using float16x4 =
    __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

template <typename T>
__device__ __forceinline__ floatx4 mfma_fp32_16x16x16fp16(floatx4 C,
                                                          const float16x4 A,
                                                          const float16x4 B)
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
amdgcn_mfma_fp32_16x16x16fp16(float *C, uint32_t *A, uint32_t *B)
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

    float16x4 A_fp16 = reinterpret_cast<float16x4 *>(A)[0];
    float16x4 B_fp16 = reinterpret_cast<float16x4 *>(B)[0];
    floatx4 C_fp32 = reinterpret_cast<floatx4 *>(C)[0];

    // Perform MMA operation directly with fragments
    C_fp32 = mfma_fp32_16x16x16fp16<T>(C_fp32, A_fp16, B_fp16);
    C[0] = C_fp32[0];
    C[1] = C_fp32[1];
    C[2] = C_fp32[2];
    C[3] = C_fp32[3];
}

// Rowsum operation using MMA
// template <typename DType>
// __device__ __forceinline__ void
// m16k16_rowsum_f16f16f32(accumulator_fragment_m16n16k16<float> &d_frag,
//                         const row_major_fragment_m16n16k16<DType> &s_frag)
// {
//     static_assert(sizeof(DType) == 2, "DType must be 16bit");

//     // Create a ones fragment
//     col_major_fragment_m16n16k16<DType> ones_frag;

//     // Fill with ones
//     if constexpr (std::is_same_v<DType, __half>) {
//         ones_frag.fill(__float2half(1.0f));
//     }
//     else if constexpr (std::is_same_v<DType, hip_bfloat16>) {
//         ones_frag.fill(__float2bfloat16(1.0f));
//     }

//     // Use MMA to compute rowsum
//     mma_sync_m16n16k16_row_col_f16f16f32<DType, MMAMode::kInplaceUpdate>(
//         d_frag, s_frag, ones_frag);
// }

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
