// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache - 2.0

#pragma once
#include "platform.hpp"

// Include platform-specific implementations
#if defined(PLATFORM_CUDA_DEVICE)
#include "backend/cuda/memory_ops_cuda.cuh"
#elif defined(PLATFORM_HIP_DEVICE)
#include "backend/hip/memory_ops_hip.hip.h"
#endif

namespace flashinfer
{
namespace gpu_iface
{
namespace memory
{

/**
 * @brief Control options for shared memory fill behavior
 */
enum class SharedMemFillMode
{
    kFillZero, // Fill zero to shared memory when predicate is false
    kNoFill    // Do not fill zero to shared memory when predicate is false
};

/**
 * @brief Control options for memory prefetch behavior
 */
enum class PrefetchMode
{
    kNoPrefetch, // Do not fetch additional data from global memory to L2
    kPrefetch    // Fetch additional data from global memory to L2
};
/**
 * @brief Commits pending asynchronous memory operations to a group
 */
__device__ __forceinline__ void commit_group()
{
#if defined(PLATFORM_CUDA_DEVICE)
    detail::cuda::commit_group();
#elif defined(PLATFORM_HIP_DEVICE)
    detail::hip::commit_group();
#endif
}

/**
 * @brief Waits until N most recent groups of async operations are complete
 *
 * @tparam N Number of most recent groups to wait for (0-7)
 */
template <size_t N> __device__ __forceinline__ void wait_group()
{
#if defined(PLATFORM_CUDA_DEVICE)
    detail::cuda::wait_group<N>();
#elif defined(PLATFORM_HIP_DEVICE)
    detail::hip::wait_group<N>();
#endif
}

/**
 * @brief Asynchronously loads 128 bits from global to shared memory
 *
 * @tparam PrefetchOpt Prefetch option
 * @tparam T Data type
 * @param smem_ptr Destination shared memory pointer
 * @param gmem_ptr Source global memory pointer
 */
template <PrefetchMode PrefetchOpt, typename T>
__device__ __forceinline__ void load_128b(T *smem_ptr, const T *gmem_ptr)
{
#if defined(PLATFORM_CUDA_DEVICE)
    detail::cuda::load_128b<PrefetchOpt>(smem_ptr, gmem_ptr);
#elif defined(PLATFORM_HIP_DEVICE)
    detail::hip::load_128b<PrefetchOpt>(smem_ptr, gmem_ptr);
#endif
}

/**
 * @brief Conditionally loads 128 bits from global to shared memory
 *
 * @tparam PrefetchOpt Prefetch option
 * @tparam FillOpt Memory fill option
 * @tparam T Data type
 * @param smem_ptr Destination shared memory pointer
 * @param gmem_ptr Source global memory pointer
 * @param predicate Condition for executing the load
 */
template <PrefetchMode PrefetchOpt, SharedMemFillMode FillOpt, typename T>
__device__ __forceinline__ void
pred_load_128b(T *smem_ptr, const T *gmem_ptr, bool predicate)
{
#if defined(PLATFORM_CUDA_DEVICE)
    detail::cuda::pred_load_128b<PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr,
                                                       predicate);
#elif defined(PLATFORM_HIP_DEVICE)
    detail::hip::pred_load_128b<PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr,
                                                      predicate);
#endif
}

/**
 * @brief Loads N bits (128 or 256) from global to shared memory
 *
 * @tparam NumBits Number of bits to load (128 or 256)
 * @tparam PrefetchOpt Prefetch option
 * @tparam T Data type
 * @param smem_ptr Destination shared memory pointer
 * @param gmem_ptr Source global memory pointer
 */
template <size_t NumBits, PrefetchMode PrefetchOpt, typename T>
__device__ __forceinline__ void load(T *smem_ptr, const T *gmem_ptr)
{
#if defined(PLATFORM_CUDA_DEVICE)
    detail::cuda::load<NumBits, PrefetchOpt>(smem_ptr, gmem_ptr);
#elif defined(PLATFORM_HIP_DEVICE)
    detail::hip::load<NumBits, PrefetchOpt>(smem_ptr, gmem_ptr);
#endif
}

/**
 * @brief Conditionally loads N bits from global to shared memory
 *
 * @tparam NumBits Number of bits to load (128 or 256)
 * @tparam PrefetchOpt Prefetch option
 * @tparam FillOpt Memory fill option
 * @tparam T Data type
 * @param smem_ptr Destination shared memory pointer
 * @param gmem_ptr Source global memory pointer
 * @param predicate Condition for executing the load
 */
template <size_t NumBits,
          PrefetchMode PrefetchOpt,
          SharedMemFillMode FillOpt,
          typename T>
__device__ __forceinline__ void
pred_load(T *smem_ptr, const T *gmem_ptr, bool predicate)
{
#if defined(PLATFORM_CUDA_DEVICE)
    detail::cuda::pred_load<NumBits, PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr,
                                                           predicate);
#elif defined(PLATFORM_HIP_DEVICE)
    detail::hip::pred_load<NumBits, PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr,
                                                          predicate);
#endif
}

} // namespace memory
} // namespace gpu_iface
} // namespace flashinfer
