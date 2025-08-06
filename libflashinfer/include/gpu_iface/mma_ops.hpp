// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache - 2.0

#pragma once
#include "gpu_iface/mma_types.hpp"
#include "gpu_iface/platform.hpp"

// Include platform-specific implementations
#if defined(PLATFORM_CUDA_DEVICE)
#include "backend/cuda/mma.cuh"
namespace detail = flashinfer::gpu_iface::mma_impl::cuda;
#elif defined(PLATFORM_HIP_DEVICE)
#include "backend/hip/mma_hip.h"
namespace detail = flashinfer::gpu_iface::mma_impl::hip;
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
    detail::load_fragment<T>(R, smem_ptr);
}

template <typename T>
__device__ __forceinline__ void
load_fragment_transpose(uint32_t *R, const T *smem_ptr, uint32_t stride)
{
    detail::load_fragment_transpose<T>(R, smem_ptr, stride);
}

/*!
 * \brief Wrapper of two mma m16n16k16 instructions for row major and column
 * major f16 matrix multiplication, accumulated in f32.
 * \tparam T data type of the fragment
 * \tparam mma_mode whether we are initializing the accumulator or updating it
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <typename T, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void
amdgcn_mfma_fp32_16x16x16fp16(float *C, uint32_t *A, uint32_t *B)
{
#if defined(PLATFORM_HIP_DEVICE)
    detail::amdgcn_mfma_fp32_16x16x16fp16<T, mma_mode>(C, A, B);
#else
    FLASHINFER_RUNTIME_ASSERT(
        "MMA f16f16f32 not supported on this architecture");
#endif
}

// /*!
//  * \brief Use mma instructions to compute rowsum.
//  */
// template <typename DType>
// __device__ __forceinline__ void
// m16k16_rowsum_f16f16f32(float* d, DType* s)
// {
//     detail::m16k16_rowsum_f16f16f32(d, s);
// }

} // namespace mma
} // namespace gpu_iface
} // namespace flashinfer
