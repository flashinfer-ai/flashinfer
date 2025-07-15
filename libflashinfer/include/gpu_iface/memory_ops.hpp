// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache - 2.0

#pragma once
#include "platform.hpp"

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

// Include platform-specific implementations
#if defined(PLATFORM_CUDA_DEVICE)
#include "backend/cuda/memory_ops.cuh"
namespace detail = flashinfer::gpu_iface::memory::detail::cuda;
#elif defined(PLATFORM_HIP_DEVICE)
#include "backend/hip/memory_ops_hip.h"
namespace detail = flashinfer::gpu_iface::memory::detail::hip;
#endif

/**
 * @brief Commits pending asynchronous memory operations to a group
 */
__device__ __forceinline__ void commit_group() { detail::commit_group(); }

/**
 * @brief Waits until N most recent groups of async operations are complete
 *
 * @tparam N Number of most recent groups to wait for (0-7)
 */
template <size_t N> __device__ __forceinline__ void wait_group()
{
    detail::wait_group<N>();
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
    detail::load_128b<PrefetchOpt>(smem_ptr, gmem_ptr);
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
    detail::pred_load_128b<PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr, predicate);
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
    detail::load<NumBits, PrefetchOpt>(smem_ptr, gmem_ptr);
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
    detail::pred_load<NumBits, PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr,
                                                     predicate);
}

} // namespace memory
} // namespace gpu_iface
} // namespace flashinfer
