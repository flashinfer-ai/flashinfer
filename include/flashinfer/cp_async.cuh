#ifndef FLASHINFER_CP_ASYNC_CUH_
#define FLASHINFER_CP_ASYNC_CUH_

#include <cuda_runtime.h>

namespace flashinfer {

namespace cp_async {

__device__ __forceinline__ void commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <size_t n>
__device__ __forceinline__ void wait_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
#endif
}

template <bool prefetch, typename T>
__device__ __forceinline__ void load_128(T* smem, const T* gmem) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11)

  uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  if constexpr (prefetch) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_ptr),
                 "l"(gmem), "n"(16), "r"(16));
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr), "l"(gmem), "n"(16));
  }
#else
  *((uint4*)smem) = *((uint4*)gmem);
#endif
}

template <bool prefetch, typename T>
__device__ __forceinline__ void pred_load_128(T* smem, const T* gmem, bool predicate) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11)

  uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  if constexpr (prefetch) {
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, %0, 0;\n"
        " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
        "}\n" ::"r"((int)predicate),
        "r"(smem_ptr), "l"(gmem), "n"(16));
  } else {
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, %0, 0;\n"
        " @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int)predicate),
        "r"(smem_ptr), "l"(gmem), "n"(16));
  }
#else
  if (predicate) {
    *((uint4*)smem) = *((uint4*)gmem);
  }
#endif
}

template <size_t num_bits, bool prefetch, typename T>
__device__ __forceinline__ void load(T* smem, const T* gmem) {
  static_assert(num_bits == 128 || num_bits == 256, "num_bits must be 128 or 256");
  if constexpr (num_bits == 128) {
    load_128<prefetch>(smem, gmem);
  } else {
    load_128<prefetch>(smem, gmem);
    load_128<prefetch>(smem + 16 / sizeof(T), gmem + 16 / sizeof(T));
  }
}

template <size_t num_bits, bool prefetch, typename T>
__device__ __forceinline__ void pred_load(T* smem, const T* gmem, bool predicate) {
  static_assert(num_bits == 128 || num_bits == 256, "num_bits must be 128 or 256");
  if constexpr (num_bits == 128) {
    pred_load_128<prefetch>(smem, gmem, predicate);
  } else {
    pred_load_128<prefetch>(smem, gmem, predicate);
    pred_load_128<prefetch>(smem + 16 / sizeof(T), gmem + 16 / sizeof(T), predicate);
  }
}

}  // namespace cp_async

}  // namespace flashinfer

#endif  // FLASHINFER_CP_ASYNC_CUH_
