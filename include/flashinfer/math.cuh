#ifndef FLASHINFER_MATH_CUH_
#define FLASHINFER_MATH_CUH_

#include <cuda_runtime.h>

namespace flashinfer {
namespace math {

constexpr float log2e = 1.44269504088896340736f;

__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__forceinline__ __device__ float shfl_xor_sync(float x, int delta) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(y) : "f"(x), "r"(delta));
  return y;
}

}  // namespace math
}  // namespace flashinfer
#endif  // FLASHINFER_MATH_CUH_
