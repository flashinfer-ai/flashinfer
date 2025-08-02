// SPDX - FileCopyrightText : 2023-2035 FlashInfer team.
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#ifndef FLASHINFER_PERMUTED_SMEM_CUH_
#define FLASHINFER_PERMUTED_SMEM_CUH_

#include "gpu_iface/memory_ops.hpp"
#include "gpu_iface/platform.hpp"

#if 0
#include "mma.cuh"
#include <cuda/pipeline>
#endif

namespace gpu_mem = flashinfer::gpu_iface::memory;

namespace flashinfer
{

enum class SwizzleMode
{
    k64B,
    k128B,
    kLinear,
};

// Use 128bit as the granularity to fetch/store data per thread to maximize
// memory bandwidth
using b128_t = uint4;
// 64b type to support 16-bit CDNA3 WMMA ops where each thread in a 64 thread
// wavefront loads a four element fragment.
using b64_t = uint2;

/*!
 * \brief Compute the number of elements that can be stored in a b128_t.
 * \tparam T The data type of the elements.
 */
template <typename T, size_t NumBits = 128>
constexpr __host__ __device__ __forceinline__ uint32_t upcast_size()
{
    static_assert(NumBits == 128 || NumBits == 64,
                  "Only 64 and 128 bits are supported");
    if constexpr (NumBits == 128) {
        return sizeof(b128_t) / sizeof(T);
    }
    else if constexpr (NumBits == 64) {
        return sizeof(b64_t) / sizeof(T);
    }
}

/*!
 * \brief The shared memory wrapper.
 */
template <SwizzleMode swizzle_mode, typename BasePtrTy = b128_t> struct smem_t
{
    // The base pointer.
    BasePtrTy *base;
    __device__ __forceinline__ smem_t() : base(nullptr) {}
    template <typename T>
    __device__ __forceinline__ smem_t(T *base) : base((BasePtrTy *)base)
    {
    }

    /*!
     * \brief Compute the element offset given coordinates in a permuted shared
     * memory.
     * \tparam stride The stride (in terms of b128_t's) in the permuted shared
     * memory.
     * \param i The row index.
     * \param j The column index.
     */
    template <uint32_t stride>
    static __device__ __forceinline__ uint32_t get_permuted_offset(uint32_t i,
                                                                   uint32_t j)
    {
        if constexpr (swizzle_mode == SwizzleMode::k128B) {
            return i * stride + (j ^ (i % 8));
        }
        else if constexpr (swizzle_mode == SwizzleMode::k64B) {
            static_assert(stride == 4);
            return i * stride + (j ^ ((i / 2) % 4));
        }
        else {
            // swizzle_mode == SwizzleMode::kLinear
            return i * stride + j;
        }
    }

    template <uint32_t step_size>
    static __device__ __forceinline__ uint32_t
    advance_offset_by_column(uint32_t offset, uint32_t step_idx)
    {
        if constexpr (swizzle_mode == SwizzleMode::k128B) {
            static_assert(step_size == 2 || step_size == 4 ||
                              step_size % 8 == 0,
                          "Unsupported step size");
            if constexpr (step_size == 2) {
                return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) +
                       (step_idx % 4 == 3) * 8;
            }
            else if constexpr (step_size == 4) {
                return (offset ^ 0x4) + (step_idx % 2 == 1) * 8;
            }
            else {
                // step_size % 8 == 0
                return offset + step_size;
            }
        }
        else if constexpr (swizzle_mode == SwizzleMode::k64B) {
            static_assert(step_size == 2 || step_size == 4,
                          "Unsupported step size");
            return (offset ^ 0x2) + (step_idx % 2 == 1) * 4;
        }
        else {
            // swizzle_mode == SwizzleMode::kLinear
            return offset + step_size;
        }
    }

    template <uint32_t step_size, uint32_t row_stride>
    static __device__ __forceinline__ uint32_t
    advance_offset_by_row(uint32_t offset)
    {
        if constexpr (swizzle_mode == SwizzleMode::k128B) {
            static_assert(step_size == 4 || step_size % 8 == 0,
                          "Unsupported step size");
            if constexpr (step_size == 4) {
                return (offset ^ 0x4) + step_size * row_stride;
            }
            else {
                // step_size % 8 == 0
                return offset + step_size * row_stride;
            }
        }
        else if constexpr (swizzle_mode == SwizzleMode::k64B) {
            static_assert(step_size == 4 || step_size % 8 == 0,
                          "Unsupported step size");
            if constexpr (step_size == 4) {
                return (offset ^ 0x2) + step_size * row_stride;
            }
            else {
                // step_size % 8 == 0
                return offset + step_size * row_stride;
            }
        }
        else {
            // swizzle_mode == SwizzleMode::kLinear
            return offset + step_size * row_stride;
        }
    }

    __device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t offset,
                                                    uint32_t *R)
    {
        // b128_t *smem_ptr = base + offset;
        // mma::ldmatrix_m8n8x4(R, smem_ptr);
    }

    __device__ __forceinline__ void ldmatrix_m8n8x4_left_half(uint32_t offset,
                                                              uint32_t *R)
    {
        // b128_t *smem_ptr = base + offset;
        // mma::ldmatrix_m8n8x4_left_half(R, smem_ptr);
    }

    __device__ __forceinline__ void ldmatrix_m8n8x4_right_half(uint32_t offset,
                                                               uint32_t *R)
    {
        // b128_t *smem_ptr = base + offset;
        // mma::ldmatrix_m8n8x4_right_half(R, smem_ptr);
    }

    __device__ __forceinline__ void stmatrix_m8n8x4(uint32_t offset,
                                                    uint32_t *R)
    {
        // b128_t *smem_ptr = base + offset;
        // mma::stmatrix_m8n8x4(R, smem_ptr);
    }

    __device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t offset,
                                                          uint32_t *R)
    {
        // b128_t *smem_ptr = base + offset;
        // mma::ldmatrix_m8n8x4_trans(R, smem_ptr);
    }

    __device__ __forceinline__ void
    ldmatrix_m8n8x4_trans_left_half(uint32_t offset, uint32_t *R)
    {
        // b128_t *smem_ptr = base + offset;
        // mma::ldmatrix_m8n8x4_trans_left_half(R, smem_ptr);
    }

    __device__ __forceinline__ void
    ldmatrix_m8n8x4_trans_right_half(uint32_t offset, uint32_t *R)
    {
        // b128_t *smem_ptr = base + offset;
        // mma::ldmatrix_m8n8x4_trans_right_half(R, smem_ptr);
    }

    template <gpu_mem::SharedMemFillMode fill_mode, typename T>
    __device__ __forceinline__ void
    load_128b_async(uint32_t offset, const T *gptr, bool predicate)
    {
        b128_t *smem_ptr = base + offset;
        gpu_mem::pred_load_128b<gpu_mem::PrefetchMode::kPrefetch, fill_mode>(
            smem_ptr, reinterpret_cast<const b128_t *>(gptr), predicate);
    }

    template <typename T>
    __device__ __forceinline__ void load_128b_async(uint32_t offset,
                                                    const T *gptr)
    {
        b128_t *smem_ptr = base + offset;
        gpu_mem::load_128b<gpu_mem::PrefetchMode::kPrefetch>(
            smem_ptr, reinterpret_cast<const b128_t *>(gptr));
    }

    template <gpu_mem::SharedMemFillMode fill_mode, typename T>
    __device__ __forceinline__ void
    load_64b_async(uint32_t offset, const T *gptr, bool predicate)
    {
        b64_t *smem_ptr = base + offset;
        gpu_mem::pred_load_64b<gpu_mem::PrefetchMode::kPrefetch, fill_mode>(
            smem_ptr, reinterpret_cast<const b64_t *>(gptr), predicate);
    }

    template <typename T>
    __device__ __forceinline__ void load_64b_async(uint32_t offset,
                                                   const T *gptr)
    {
        b64_t *smem_ptr = base + offset;
        gpu_mem::load_64b<gpu_mem::PrefetchMode::kPrefetch>(
            smem_ptr, reinterpret_cast<const b64_t *>(gptr));
    }

    template <typename T>
    __device__ __forceinline__ void store_128b(uint32_t offset, T *gptr)
    {
        *reinterpret_cast<b128_t *>(gptr) = *(base + offset);
    }
};

} // namespace flashinfer

#endif // FLASHINFER_PERMUTED_SMEM_CUH_
