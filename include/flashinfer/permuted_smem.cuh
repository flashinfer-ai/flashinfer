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

namespace permuted_smem_impl {

/*!
 * \brief Compute the address of the element at (i, j) in the permuted shared memory.
 * \tparam stride The number of cells per row in the permuted shared memory.
 * \param smem_base The base address of the permuted shared memory.
 * \param i The row index.
 * \param j The column (cell) index.
 * \note The permuted shared memory maps 8x4 block in logical space to 4x8 block in physical space.
 * \see GTC 2020: Developing CUDA Kernels to Push Tensor Cores to the Absolute Limit on NVIDIA A100.
 */
template <uint32_t stride, typename T>
__host__ __device__ __forceinline__ cell_t *get_smem_ptr(T *smem_base, uint32_t i, uint32_t j) {
  return ((cell_t *)smem_base) + (i / 2) * stride * 2 + (j / 4) * 8 + (i % 2) * 4 +
         ((j % 4) ^ ((i / 2) % 4));
}

template <uint32_t stride, typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t *R, T *smem_base, uint32_t i, uint32_t j) {
  cell_t *smem_ptr = get_smem_ptr<stride>(smem_base, i, j);
  mma::ldmatrix_m8n8x4(R, smem_ptr);
}

template <uint32_t stride, typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t *R, T *smem_base, uint32_t i,
                                                      uint32_t j) {
  cell_t *smem_ptr = get_smem_ptr<stride>(smem_base, i, j);
  mma::ldmatrix_m8n8x4_trans(R, smem_ptr);
}

template <uint32_t stride, typename T>
__device__ __forceinline__ void stmatrix_m8n8x4(uint32_t *R, T *smem_base, uint32_t i, uint32_t j) {
  cell_t *smem_ptr = get_smem_ptr<stride>(smem_base, i, j);
  mma::stmatrix_m8n8x4(R, smem_ptr);
}

template <uint32_t stride, typename T>
__device__ __forceinline__ void load_128b(T *smem_base, uint32_t i, uint32_t j, const T *gptr) {
  *get_smem_ptr<stride>(smem_base, i, j) = *reinterpret_cast<const cell_t *>(gptr);
}

template <uint32_t stride, typename T>
__device__ __forceinline__ void store_128b(T *smem_base, uint32_t i, uint32_t j, T *gptr) {
  *reinterpret_cast<cell_t *>(gptr) = *get_smem_ptr<stride>(smem_base, i, j);
}

template <uint32_t stride, typename T>
__device__ __forceinline__ void load_128b_async(T *smem_base, uint32_t i, uint32_t j, const T *gptr,
                                                bool predicate) {
  cell_t *smem_ptr = get_smem_ptr<stride>(smem_base, i, j);
  cp_async::pred_load_128b<true>(smem_ptr, reinterpret_cast<const cell_t *>(gptr), predicate);
}

template <uint32_t stride, typename T>
__device__ __forceinline__ void load_128b_async(T *smem_base, uint32_t i, uint32_t j,
                                                const T *gptr) {
  cell_t *smem_ptr = get_smem_ptr<stride>(smem_base, i, j);
  cp_async::load_128b<true>(smem_ptr, reinterpret_cast<const cell_t *>(gptr));
}

}  // namespace permuted_smem_impl

template <uint32_t stride, typename T>
struct permuted_smem_t {
  T __align__(16) * base;
  __device__ __forceinline__ permuted_smem_t() : base(nullptr) {}
  __device__ __forceinline__ permuted_smem_t(T *base) : base(base) {}
  __device__ __forceinline__ cell_t *get_ptr(uint32_t i, uint32_t j) {
    return permuted_smem_impl::get_smem_ptr<stride>(base, i, j);
  }
  __device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t *R, uint32_t i, uint32_t j) {
    permuted_smem_impl::ldmatrix_m8n8x4<stride>(R, base, i, j);
  }
  __device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t *R, uint32_t i, uint32_t j) {
    permuted_smem_impl::ldmatrix_m8n8x4_trans<stride>(R, base, i, j);
  }
  __device__ __forceinline__ void stmatrix_m8n8x4(uint32_t *R, uint32_t i, uint32_t j) {
    permuted_smem_impl::stmatrix_m8n8x4<stride>(R, base, i, j);
  }
  __device__ __forceinline__ void load_128b(uint32_t i, uint32_t j, const T *gptr) {
    permuted_smem_impl::load_128b<stride>(base, i, j, gptr);
  }
  __device__ __forceinline__ void load_128b_async(uint32_t i, uint32_t j, const T *gptr,
                                                  bool predicate) {
    permuted_smem_impl::load_128b_async<stride>(base, i, j, gptr, predicate);
  }
  __device__ __forceinline__ void load_128b_async(uint32_t i, uint32_t j, const T *gptr) {
    permuted_smem_impl::load_128b_async<stride>(base, i, j, gptr);
  }
  __device__ __forceinline__ void store_128b(uint32_t i, uint32_t j, T *gptr) {
    permuted_smem_impl::store_128b<stride>(base, i, j, gptr);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_PERMUTED_SMEM_CUH_
