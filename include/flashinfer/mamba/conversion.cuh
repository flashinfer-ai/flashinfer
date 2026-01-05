#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

namespace flashinfer::mamba::conversion {

inline __device__ float toFloat(float f) { return f; }

inline __device__ float toFloat(__half h) { return __half2float(h); }
#ifdef ENABLE_BF16
inline __device__ float toFloat(__nv_bfloat16 val) { return __bfloat162float(val); }
#endif

inline __device__ void convertAndStore(float* output, float input) { *output = input; }

inline __device__ void convertAndStore(__half* output, float input) {
  *output = __float2half(input);
}

#ifdef ENABLE_BF16
inline __device__ void convertAndStore(__nv_bfloat16* output, float input) {
  *output = __float2bfloat16(input);
}
#endif

}  // namespace flashinfer::mamba::conversion
