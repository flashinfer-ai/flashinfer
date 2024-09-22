/*
Copyright (c) 2024 by LEI WANG
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef FLASHINFER_HIP_CUDA_TYPE_UTILS_H_
#define FLASHINFER_HIP_CUDA_TYPE_UTILS_H_

// namespace flashinfer {

#if defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_common.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

// CUDA DEVICE API Supported : https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Device_API_supported_by_HIP.html

/*! \brief Struct to packet two 16 bit brain floating point numbers. */
using nv_bfloat162 = __hip_bfloat162;
using __nv_bfloat162 = __hip_bfloat162;

/*! \brief Struct to represent a 16 bit brain floating point number. */
using nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat16 = __hip_bfloat16;

using half2 = __half2;

// ROCM FP8 is different from nv FP8 : https://github.com/ROCm/rocBLAS/blob/9b7f692abe3c54b88d1e77e045a7db7f1f188b69/library/include/internal/rocblas_hip_f8_impl.h#L39

// TODO (yiakwy) : FP8 datatype support


// TODO (yiakwy) : FP8 cast, generic cast, vector cast support


// bf16 utils
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 make_bfloat162(const __hip_bfloat16 x, const __hip_bfloat16 y)
{
    __hip_bfloat162 t; t.x = x; t.y = y; return t;
}

// Following math functions included in ROCM6.2 SDK :
// __hmul: bfloat16 -> bfloat16,
// __hmul2: bfloat16 -> bfloat16,
// __floats2bfloat162_rn: (float,float) -> __hip_bfloat162,
// __float22bfloat162_rn: float2 -> __hip_bfloat162,
// __float2bfloat162_rn: float -> __hip_bfloat162,
// __bfloat1622float2: __hip_bfloat162 -> float2

// half utils
// TODO (yiakwy) : add native half2 support implementation
__device__ half2 __hmax2(const half2 a, const half2 b) {
  return half2{
      __float2half(__ocml_fmax_f32(__half2float(a.x), __half2float(b.x))),
      __float2half(__ocml_fmax_f32(__half2float(a.y), __half2float(b.y)))};
}

#endif

// } //  flashinfer

#endif // FLASHINFER_HIP_CUDA_TYPE_UTILS_H_

