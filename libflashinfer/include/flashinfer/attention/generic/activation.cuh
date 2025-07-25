// SPDX - FileCopyrightText : 2023-2035 FlashInfer team.
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#ifndef FLASHINFER_ACTIVATION_CUH_
#define FLASHINFER_ACTIVATION_CUH_

#include "gpu_iface/math_ops.hpp"
#include "gpu_iface/platform.hpp"
#include "gpu_iface/utils.cuh"
#include "gpu_iface/vec_dtypes.hpp"

namespace flashinfer
{
using namespace gpu_iface::vec_dtypes;
namespace activation
{

template <typename T, float (*Activation)(const float &)>
__global__ void act_and_mul_kernel(T *__restrict__ out,
                                   const T *__restrict__ input,
                                   const int d)
{
    constexpr uint32_t vec_size = 16 / sizeof(T);
    const int64_t token_idx = blockIdx.x;
    const int64_t thread_idx = threadIdx.x;
    const int64_t stride = blockDim.x;
    const int64_t offset = token_idx * 2 * d;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) &&                   \
     (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1
    for (uint32_t idx = thread_idx; idx < d / vec_size; idx += stride) {
        vec_t<float, vec_size> x_vec, y_vec, out_vec;
        x_vec.cast_load(input + offset + idx * vec_size);
        y_vec.cast_load(input + offset + d + idx * vec_size);
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
            out_vec[i] = Activation(x_vec[i]) * y_vec[i];
        }
        out_vec.cast_store(out + token_idx * d + idx * vec_size);
    }

    const int64_t remaining_offset = d - d % (stride * vec_size);
    // process the remaining elements
#pragma unroll 1
    for (int64_t idx = thread_idx; idx < d % (stride * vec_size); idx += stride)
    {
        float x = input[offset + remaining_offset + idx],
              y = input[offset + remaining_offset + d + idx];
        out[token_idx * d + remaining_offset + idx] = Activation(x) * y;
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) &&                   \
     (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

} // namespace activation
} // namespace flashinfer

#endif // FLASHINFER_ACTIVATION_CUH_
