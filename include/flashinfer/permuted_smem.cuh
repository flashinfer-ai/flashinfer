#ifndef FLASHINFER_PERMUTED_SMEM_CUH_
#define FLASHINFER_PERMUTED_SMEM_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>

#include "cp_async.cuh"
#include "mma.cuh"

namespace flashinfer {

// Each bank is 4 bytes.
using bank_t = uint4;

template <typename T>
constexpr __host__ __device__ __forceinline__ size_t bank_capacity() {
  return sizeof(bank_t) / sizeof(T);
}

namespace permuted_smem_impl {

/*!
 * \brief Compute the address of the element at (i, j) in the permuted shared memory.
 * \tparam stride The number of banks per row in the permuted shared memory.
 * \param smem_base The base address of the permuted shared memory.
 * \param i The row index.
 * \param j The column (bank) index.
 * \note The permuted shared memory maps 8x4 block in logical space to 4x8 block in physical space.
 * \see GTC 2020: Developing CUDA Kernels to Push Tensor Cores to the Absolute Limit on NVIDIA A100.
 */
template <size_t stride, typename T>
__host__ __device__ __forceinline__ bank_t *get_smem_ptr(T *smem_base, size_t i, size_t j) {
  return ((bank_t *)smem_base) + (i / 2) * stride * 2 + (j / 4) * 8 + (i % 2) * 4 +
         ((j % 4) ^ ((i / 2) % 4));
}

template <size_t stride, typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t *R, T *smem_base, size_t i, size_t j) {
  bank_t *smem_ptr = get_smem_ptr<stride>(smem_base, i, j);
  mma::ldmatrix_m8n8x4(R, smem_ptr);
}

template <size_t stride, typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t *R, T *smem_base, size_t i,
                                                      size_t j) {
  bank_t *smem_ptr = get_smem_ptr<stride>(smem_base, i, j);
  mma::ldmatrix_m8n8x4_trans(R, smem_ptr);
}

template <size_t stride, typename T>
__device__ __forceinline__ void stmatrix_m8n8x4(uint32_t *R, T *smem_base, size_t i, size_t j) {
  bank_t *smem_ptr = get_smem_ptr<stride>(smem_base, i, j);
  mma::stmatrix_m8n8x4(R, smem_ptr);
}

template <size_t stride, typename T>
__device__ __forceinline__ void load_bank(T *smem_base, size_t i, size_t j, const T *gptr) {
  *get_smem_ptr<stride>(smem_base, i, j) = *reinterpret_cast<const bank_t *>(gptr);
}

template <size_t stride, typename T>
__device__ __forceinline__ void store_bank(T *smem_base, size_t i, size_t j, T *gptr) {
  *reinterpret_cast<bank_t *>(gptr) = *get_smem_ptr<stride>(smem_base, i, j);
}

template <size_t stride, typename T>
__device__ __forceinline__ void load_bank_async(T *smem_base, size_t i, size_t j, const T *gptr,
                                                bool predicate) {
  bank_t *smem_ptr = get_smem_ptr<stride>(smem_base, i, j);
  cp_async::pred_load_128<true>(smem_ptr, reinterpret_cast<const bank_t *>(gptr), predicate);
}

template <size_t stride, typename T>
__device__ __forceinline__ void load_bank_async(T *smem_base, size_t i, size_t j, const T *gptr) {
  bank_t *smem_ptr = get_smem_ptr<stride>(smem_base, i, j);
  cp_async::load_128<true>(smem_ptr, reinterpret_cast<const bank_t *>(gptr));
}

}  // namespace permuted_smem_impl

template <size_t stride, typename T>
struct permuted_smem_t {
  T __align__(16) * base;
  __device__ __forceinline__ permuted_smem_t() : base(nullptr) {}
  __device__ __forceinline__ permuted_smem_t(T *base) : base(base) {}
  __device__ __forceinline__ bank_t *get_ptr(size_t i, size_t j) {
    return permuted_smem_impl::get_smem_ptr<stride>(base, i, j);
  }
  __device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t *R, size_t i, size_t j) {
    permuted_smem_impl::ldmatrix_m8n8x4<stride>(R, base, i, j);
  }
  __device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t *R, size_t i, size_t j) {
    permuted_smem_impl::ldmatrix_m8n8x4_trans<stride>(R, base, i, j);
  }
  __device__ __forceinline__ void stmatrix_m8n8x4(uint32_t *R, size_t i, size_t j) {
    permuted_smem_impl::stmatrix_m8n8x4<stride>(R, base, i, j);
  }
  __device__ __forceinline__ void load_bank(size_t i, size_t j, const T *gptr) {
    permuted_smem_impl::load_bank<stride>(base, i, j, gptr);
  }
  __device__ __forceinline__ void load_bank_async(size_t i, size_t j, const T *gptr,
                                                  bool predicate) {
    permuted_smem_impl::load_bank_async<stride>(base, i, j, gptr, predicate);
  }
  __device__ __forceinline__ void load_bank_async(size_t i, size_t j, const T *gptr) {
    permuted_smem_impl::load_bank_async<stride>(base, i, j, gptr);
  }
  __device__ __forceinline__ void store_bank(size_t i, size_t j, T *gptr) {
    permuted_smem_impl::store_bank<stride>(base, i, j, gptr);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_PERMUTED_SMEM_CUH_
