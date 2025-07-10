// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
//
// SPDX - License - Identifier : Apache 2.0

#pragma once

#include "gpu_iface/memory_ops.hpp"
#include <cstdint>
#include <cuda_runtime.h>

#if (__CUDACC_VER_MAJOR__ >= 11)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))
#define FLASHINFER_CP_ASYNC_ENABLED
#endif
#endif

namespace flashinfer
{
namespace gpu_iface
{
namespace detail
{
namespace cuda
{
using namespace flashinfer::gpu_iface;

/*!
 * \brief Wrapper of PTX cp.async.commit_group instruction, commit all prior
 * uncommitted cp.async instructions to a group
 */
__device__ __forceinline__ void commit_group()
{
#ifdef FLASHINFER_CP_ASYNC_ENABLED
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

/*!
 * \brief Wrapper of PTX cp.async.wait_group instruction
 * \tparam n Wait till most recent n groups are committed
 */
template <size_t n> __device__ __forceinline__ void wait_group()
{
#ifdef FLASHINFER_CP_ASYNC_ENABLED
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
#endif
}

/*!
 * \brief Wrapper of PTX cp.async.cg.shared.global instruction, asynchronously
 * copy data from global memory to shared memory
 * \tparam prefetch_mode Whether to fetch additional data from global memory to
 * L2
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 */
template <PrefetchMode prefetch_mode, typename T>
__device__ __forceinline__ void load_128b(T *smem_ptr, const T *gmem_ptr)
{
#ifdef FLASHINFER_CP_ASYNC_ENABLED
    uint32_t smem_int_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
        asm volatile(
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
                smem_int_ptr),
            "l"(gmem_ptr), "n"(16), "r"(16));
    }
    else {
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(
                         smem_int_ptr),
                     "l"(gmem_ptr), "n"(16), "r"(16));
    }
#else
    *((uint4 *)smem_ptr) = *((uint4 *)gmem_ptr);
#endif
}

/*!
 * \brief Wrapper of PTX cp.async.cg.shared.global instruction, asynchronously
 * copy data from global memory to shared memory with predicate.
 * \tparam prefetch_mode Whether to fetch additional data from global memory to
 * L2
 * \tparam fill_mode Whether to fill zero to shared memory when predicate is
 * false
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 * \param predicate Predicate value
 * \note fill zero is slower than not fill zero
 */
template <PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void
pred_load_128b(T *smem_ptr, const T *gmem_ptr, bool predicate)
{
#ifdef FLASHINFER_CP_ASYNC_ENABLED
    uint32_t smem_int_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
        int src_in_bytes = predicate ? 16 : 0;
        if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::
                    "r"(smem_int_ptr),
                "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
        }
        else {
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(
                    smem_int_ptr),
                "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
        }
    }
    else {
        if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
            asm volatile(
                "{\n"
                " .reg .pred p;\n"
                " setp.ne.b32 p, %0, 0;\n"
                " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
                "}\n" ::"r"((int)predicate),
                "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
        }
        else {
            asm volatile("{\n"
                         " .reg .pred p;\n"
                         " setp.ne.b32 p, %0, 0;\n"
                         " @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                         "}\n" ::"r"((int)predicate),
                         "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
        }
    }
#else
    if (predicate) {
        *((uint4 *)smem_ptr) = *((uint4 *)gmem_ptr);
    }
    else {
        if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
            *((uint4 *)smem_ptr) = make_uint4(0, 0, 0, 0);
        }
    }
#endif
}

/*!
 * \brief Load specified number of bits per thread from global memory to shared
 * memory
 * \tparam num_bits Number of bits to load, must be 128 or 256
 * \tparam prefetch_mode Whether to fetch additional data from global memory to
 * L2
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 */
template <size_t num_bits, PrefetchMode prefetch_mode, typename T>
__device__ __forceinline__ void load(T *smem_ptr, const T *gmem_ptr)
{
    static_assert(num_bits == 128 || num_bits == 256,
                  "num_bits must be 128 or 256");
    if constexpr (num_bits == 128) {
        load_128b<prefetch_mode>(smem_ptr, gmem_ptr);
    }
    else {
        load_128b<prefetch_mode>(smem_ptr, gmem_ptr);
        load_128b<prefetch_mode>(smem_ptr + 16 / sizeof(T),
                                 gmem_ptr + 16 / sizeof(T));
    }
}

/*!
 * \brief Load specified number of bits per thread from global memory to shared
 * memory with predicate
 * \tparam num_bits Number of bits to load, must be 128 or 256
 * \tparam prefetch_mode Whether to fetch additional data from global memory to
 * L2
 * \tparam fill_mode Whether to fill zero to shared memory when predicate is
 * false
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 * \param predicate Predicate value
 * \note fill zero is slower than not fill zero
 */
template <size_t num_bits,
          PrefetchMode prefetch_mode,
          SharedMemFillMode fill_mode,
          typename T>
__device__ __forceinline__ void
pred_load(T *smem_ptr, const T *gmem_ptr, bool predicate)
{
    static_assert(num_bits == 128 || num_bits == 256,
                  "num_bits must be 128 or 256");
    if constexpr (num_bits == 128) {
        pred_load_128b<prefetch_mode, fill_mode>(smem_ptr, gmem_ptr, predicate);
    }
    else {
        pred_load_128b<prefetch_mode, fill_mode>(smem_ptr, gmem_ptr, predicate);
        pred_load_128b<prefetch_mode, fill_mode>(
            smem_ptr + 16 / sizeof(T), gmem_ptr + 16 / sizeof(T), predicate);
    }
}

} // namespace cuda
} // namespace detail
} // namespace gpu_iface
} // namespace flashinfer
