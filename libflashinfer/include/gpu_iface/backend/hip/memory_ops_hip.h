#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace flashinfer
{
namespace gpu_iface
{
namespace memory
{
namespace detail
{
namespace hip
{

__device__ __forceinline__ void commit_group()
{
    // Currently a no-op for HIP
}

template <size_t N> __device__ __forceinline__ void wait_group()
{
    // Currently a no-op for HIP
}

/// @brief loads 128 bits from global to shared memory
template <PrefetchMode PrefetchOpt, typename T>
__device__ __forceinline__ void load_128b(T *smem_ptr, const T *gmem_ptr)
{
    *reinterpret_cast<uint4 *>(smem_ptr) =
        *reinterpret_cast<const uint4 *>(gmem_ptr);
}

template <PrefetchMode PrefetchOpt, typename T>
__device__ __forceinline__ void load_64b(T *smem_ptr, const T *gmem_ptr)
{
    *reinterpret_cast<uint2 *>(smem_ptr) =
        *reinterpret_cast<const uint2 *>(gmem_ptr);
}

// Predicated 128-bit load
template <PrefetchMode PrefetchOpt, SharedMemFillMode FillOpt, typename T>
__device__ __forceinline__ void
pred_load_128b(T *smem_ptr, const T *gmem_ptr, bool predicate)
{
    if (predicate) {
        *reinterpret_cast<uint4 *>(smem_ptr) =
            *reinterpret_cast<const uint4 *>(gmem_ptr);
    }
    else {
        if constexpr (FillOpt == SharedMemFillMode::kFillZero) {
            *reinterpret_cast<uint4 *>(smem_ptr) = make_uint4(0, 0, 0, 0);
        }
    }
}

template <PrefetchMode PrefetchOpt, SharedMemFillMode FillOpt, typename T>
__device__ __forceinline__ void
pred_load_64b(T *smem_ptr, const T *gmem_ptr, bool predicate)
{
    if (predicate) {
        *reinterpret_cast<uint2 *>(smem_ptr) =
            *reinterpret_cast<const uint2 *>(gmem_ptr);
    }
    else {
        if constexpr (FillOpt == SharedMemFillMode::kFillZero) {
            *reinterpret_cast<uint2 *>(smem_ptr) = make_uint2(0, 0);
        }
    }
}

// Generic load with NumBits template parameter
template <size_t NumBits, PrefetchMode PrefetchOpt, typename T>
__device__ __forceinline__ void load(T *smem_ptr, const T *gmem_ptr)
{
    static_assert(NumBits == 128 || NumBits == 256,
                  "NumBits must be 128 or 256");
    if constexpr (NumBits == 128) {
        load_128b<PrefetchOpt>(smem_ptr, gmem_ptr);
    }
    else {
        load_128b<PrefetchOpt>(smem_ptr, gmem_ptr);
        load_128b<PrefetchOpt>(smem_ptr + 16 / sizeof(T),
                               gmem_ptr + 16 / sizeof(T));
    }
}

// Generic predicated load with NumBits template parameter
template <size_t NumBits,
          PrefetchMode PrefetchOpt,
          SharedMemFillMode FillOpt,
          typename T>
__device__ __forceinline__ void
pred_load(T *smem_ptr, const T *gmem_ptr, bool predicate)
{
    static_assert(NumBits == 128 || NumBits == 256,
                  "NumBits must be 128 or 256");
    if constexpr (NumBits == 128) {
        pred_load_128b<PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr, predicate);
    }
    else {
        pred_load_128b<PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr, predicate);
        pred_load_128b<PrefetchOpt, FillOpt>(
            smem_ptr + 16 / sizeof(T), gmem_ptr + 16 / sizeof(T), predicate);
    }
}

} // namespace hip
} // namespace detail
} // namespace memory
} // namespace gpu_iface
} // namespace flashinfer
