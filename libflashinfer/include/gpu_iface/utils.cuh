// SPDX - FileCopyrightText : 2023-2035 FlashInfer team.
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#pragma once

#include "gpu_runtime_compat.hpp"

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// macro to turn off fp16 qk reduction to reduce binary
#ifndef FLASHINFER_ALWAYS_DISUSE_FP16_QK_REDUCTION
#define FLASHINFER_ALWAYS_DISUSE_FP16_QK_REDUCTION 0
#endif

namespace flashinfer
{

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y)
{
    return (x + y - 1) / y;
}

#if defined(PLATFORM_CUDA_DEVICE)
inline std::pair<int, int> GetCudaComputeCapability()
{
    int device_id = 0;
    cudaGetDevice(&device_id);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                           device_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                           device_id);
    return std::make_pair(major, minor);
}
#elif defined(PLATFORM_HIP_DEVICE)
inline std::pair<int, int> GetCudaComputeCapability()
{
    int device_id = 0;
    FI_GPU_CALL(hipGetDevice(&device_id));
    int major = 0, minor = 0;
    FI_GPU_CALL(hipDeviceGetAttribute(
        &major, hipDeviceAttributeComputeCapabilityMajor, device_id));
    FI_GPU_CALL(hipDeviceGetAttribute(
        &minor, hipDeviceAttributeComputeCapabilityMinor, device_id));
    return std::make_pair(major, minor);
}
#endif

template <typename T>
inline void
DebugPrintCUDAArray(T *device_ptr, size_t size, std::string prefix = "")
{
    std::vector<T> host_array(size);
    std::cout << prefix;
    gpuMemcpy(host_array.data(), device_ptr, size * sizeof(T),
              gpuMemcpyDeviceToHost);
    for (size_t i = 0; i < size; ++i) {
        std::cout << host_array[i] << " ";
    }
    std::cout << std::endl;
}

inline uint32_t FA2DetermineCtaTileQ(int64_t avg_packed_qo_len,
                                     uint32_t head_dim)
{
#if defined(PLATFORM_CUDA_DEVICE)
    if (avg_packed_qo_len > 64 && head_dim < 256) {
        return 128;
    }
    else {
        auto compute_capacity = GetCudaComputeCapability();
        if (compute_capacity.first >= 8) {
            // Ampere or newer
            if (avg_packed_qo_len > 16) {
                // avg_packed_qo_len <= 64
                return 64;
            }
            else {
                // avg_packed_qo_len <= 16
                return 16;
            }
        }
        else {
            // NOTE(Zihao): not enough shared memory on Turing for 1x4 warp
            // layout
            return 64;
        }
    }
#elif defined(PLATFORM_HIP_DEVICE)
    // Simplified version for HIP
    if (avg_packed_qo_len > 64 && head_dim < 256) {
        return 128;
    }
    else {
        return avg_packed_qo_len <= 16 ? 16 : 64;
    }
#endif
}

/*!
 * \brief Return x - y if x > y, otherwise return 0.
 */
__device__ __forceinline__ uint32_t sub_if_greater_or_zero(uint32_t x,
                                                           uint32_t y)
{
    return (x > y) ? x - y : 0U;
}

__device__ __forceinline__ void swap(uint32_t &a, uint32_t &b)
{
    uint32_t tmp = a;
    a = b;
    b = tmp;
}

__device__ __forceinline__ uint32_t dim2_offset(const uint32_t &dim_a,
                                                const uint32_t &idx_b,
                                                const uint32_t &idx_a)
{
    return idx_b * dim_a + idx_a;
}

__device__ __forceinline__ uint32_t dim3_offset(const uint32_t &dim_b,
                                                const uint32_t &dim_a,
                                                const uint32_t &idx_c,
                                                const uint32_t &idx_b,
                                                const uint32_t &idx_a)
{
    return (idx_c * dim_b + idx_b) * dim_a + idx_a;
}

__device__ __forceinline__ uint32_t dim4_offset(const uint32_t &dim_c,
                                                const uint32_t &dim_b,
                                                const uint32_t &dim_a,
                                                const uint32_t &idx_d,
                                                const uint32_t &idx_c,
                                                const uint32_t &idx_b,
                                                const uint32_t &idx_a)
{
    return ((idx_d * dim_c + idx_c) * dim_b + idx_b) * dim_a + idx_a;
}

#define DEFINE_HAS_MEMBER(member)                                              \
    template <typename T, typename = void>                                     \
    struct has_##member : std::false_type                                      \
    {                                                                          \
    };                                                                         \
    template <typename T>                                                      \
    struct has_##member<T, std::void_t<decltype(std::declval<T>().member)>>    \
        : std::true_type                                                       \
    {                                                                          \
    };                                                                         \
    template <typename T>                                                      \
    inline constexpr bool has_##member##_v = has_##member<T>::value;

} // namespace flashinfer
