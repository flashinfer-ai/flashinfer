// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache - 2.0

#pragma once
#include "gpu_iface/mma_types.hpp"
#include "gpu_iface/platform.hpp"

// Include platform-specific implementations
#if defined(PLATFORM_CUDA_DEVICE)
#include "backend/cuda/mma.cuh"
namespace mma_detail = flashinfer::gpu_iface::mma_impl::cuda;
#elif defined(PLATFORM_HIP_DEVICE)
#include "backend/hip/mma_hip.h"
namespace mma_detail = flashinfer::gpu_iface::mma_impl::hip;
#endif

namespace flashinfer
{
namespace gpu_iface
{
namespace mma
{

/*!
 * \brief Loads data from shared memory to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
// Call this load fragment
// inside mma there is impl of load

template <typename T>
__device__ __forceinline__ void load_fragment(uint32_t *R, const T *smem_ptr)
{
    mma_detail::load_fragment<T>(R, smem_ptr);
}

template <typename T>
__device__ __forceinline__ void
load_fragment_transpose(uint32_t *R, const T *smem_ptr, uint32_t stride)
{
    mma_detail::load_fragment_transpose<T>(R, smem_ptr, stride);
}

#if defined(PLATFORM_HIP_DEVICE) && defined(__gfx942__)
template <typename T>
__device__ __forceinline__ void
load_fragment_transpose_4x4_half_registers(uint32_t *R, const T *smem_ptr)
{
    static_assert(std::is_same<T, int>::value,
                  "Only __half is supported for the 4x4 register transpose");
    mma_detail::load_fragment_4x4_half_registers<half>(R, smem_ptr);
}
#endif

/*!
 * \brief An m16n16k16 gemm kernel using MMA instructions for CUDA/HIP for row
 * major and column major f16 matrix multiplication, accumulated in f32.
 *
 * \tparam T data type of the fragment
 * \tparam mma_mode whether we are initializing the accumulator or updating it
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <typename T, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void
mma_sync_m16n16k16_row_col_f16f16f32(float *C, uint32_t *A, uint32_t *B)
{
    mma_detail::mma_sync_m16n16k16_row_col_f16f16f32<T, mma_mode>(C, A, B);
}

template <typename DType>
__device__ __forceinline__ void m16k16_rowsum_f16f16f32(float *d, DType *s)
{
    mma_detail::m16k16_rowsum_f16f16f32<DType>(d, s);
}

} // namespace mma
} // namespace gpu_iface
} // namespace flashinfer
