#ifndef FLASHINFER_PERMUTED_SMEM_CUH_
#define FLASHINFER_PERMUTED_SMEM_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>

#include "cp_async.cuh"
#include "mma.cuh"

namespace flashinfer {

// Each cell is 4 bytes.
using cell_t = uint4;

template <typename T>
constexpr __host__ __device__ __forceinline__ uint32_t cell_capacity() {
  return sizeof(cell_t) / sizeof(T);
}

template <uint32_t stride, typename T>
struct permuted_smem_t {
  T __align__(16) * base;
  uint32_t offset;
  __device__ __forceinline__ permuted_smem_t() : base(nullptr) {}
  __device__ __forceinline__ permuted_smem_t(T *base) : base(base) {}

  static __device__ __forceinline__ uint32_t get_offset(uint32_t i, uint32_t j) {
    return (i / 2) * stride * 2 + (j / 4) * 8 + (i % 2) * 4 + ((j % 4) ^ ((i / 2) % 4));
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t *R) {
    cell_t *smem_ptr = (cell_t *)base + offset;
    mma::ldmatrix_m8n8x4(R, smem_ptr);
  }
  __device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t *R) {
    cell_t *smem_ptr = (cell_t *)base + offset;
    mma::ldmatrix_m8n8x4_trans(R, smem_ptr);
  }
  __device__ __forceinline__ void load_128b_async(const T *gptr, bool predicate) {
    cell_t *smem_ptr = (cell_t *)base + offset;
    cp_async::pred_load_128b<true>(smem_ptr, reinterpret_cast<const cell_t *>(gptr), predicate);
  }
  __device__ __forceinline__ void load_128b_async(const T *gptr) {
    cell_t *smem_ptr = (cell_t *)base + offset;
    cp_async::load_128b<true>(smem_ptr, reinterpret_cast<const cell_t *>(gptr));
  }
  __device__ __forceinline__ void store_128b(T *gptr) {
    *reinterpret_cast<cell_t *>(gptr) = *((cell_t *)base + offset);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_PERMUTED_SMEM_CUH_
