/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_CONCAT_MLA_CUH_
#define FLASHINFER_CONCAT_MLA_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "utils.cuh"

namespace flashinfer {

// ======================= Configuration Constants =======================
constexpr int MLA_NUM_LOCAL_HEADS = 128;
constexpr int MLA_QK_NOPE_HEAD_DIM = 128;  // k_nope dimension per head
constexpr int MLA_QK_ROPE_HEAD_DIM = 64;   // k_rope dimension (shared)
constexpr int MLA_K_HEAD_DIM = MLA_QK_NOPE_HEAD_DIM + MLA_QK_ROPE_HEAD_DIM;

constexpr int MLA_HEAD_CHUNK_SIZE = 16;
constexpr int MLA_NUM_HEAD_CHUNKS = MLA_NUM_LOCAL_HEADS / MLA_HEAD_CHUNK_SIZE;

// ======================= Optimized Kernel =======================
/*!
 * \brief Optimized CUDA kernel for concatenating k_nope and k_rope for MLA
 *
 * This kernel efficiently concatenates:
 *   - k_nope: [num_tokens, num_heads, nope_dim]
 *   - k_rope: [num_tokens, 1, rope_dim] (shared across heads)
 * into:
 *   - k: [num_tokens, num_heads, nope_dim + rope_dim]
 *
 * Key optimizations:
 * - Warp-based processing: each warp handles one (token, head_chunk) pair
 * - Vectorized memory access: int2 (8B) for nope, int (4B) for rope
 * - L2 prefetching: prefetch next row while processing current
 * - Register reuse: rope is loaded once and written to all heads in chunk
 *
 * \tparam DType Data type (nv_bfloat16 or nv_half)
 */
// ======================= Vector Type Selection =======================
// Select vector types based on element size:
// - 2-byte types (fp16/bf16): NopeVec=int2 (8B), RopeVec=int (4B)
// - 1-byte types (fp8):       NopeVec=int  (4B), RopeVec=short (2B)
template <typename DType, int ElemSize = sizeof(DType)>
struct ConcatMLAVecTraits;

// Specialization for 2-byte types (fp16/bf16)
template <typename DType>
struct ConcatMLAVecTraits<DType, 2> {
  using NopeVec = int2;
  using RopeVec = int;
  // Number of DType elements per NopeVec: 8 / 2 = 4
  static constexpr int NOPE_ELEMS_PER_VEC = sizeof(NopeVec) / sizeof(DType);
  // Number of DType elements per RopeVec: 4 / 2 = 2
  static constexpr int ROPE_ELEMS_PER_VEC = sizeof(RopeVec) / sizeof(DType);

  static __forceinline__ __device__ NopeVec load_nope(const NopeVec* addr) {
    return ld_na_global_v2(reinterpret_cast<const int2*>(addr));
  }
  static __forceinline__ __device__ void store_nope(NopeVec* addr, NopeVec val) {
    st_na_global_v2(reinterpret_cast<int2*>(addr), val);
  }
  static __forceinline__ __device__ RopeVec load_rope(const RopeVec* addr) {
    return ld_na_global_v1(reinterpret_cast<const int*>(addr));
  }
  static __forceinline__ __device__ void store_rope(RopeVec* addr, RopeVec val) {
    st_na_global_v1(reinterpret_cast<int*>(addr), val);
  }
};

// Specialization for 1-byte types (FP8)
template <typename DType>
struct ConcatMLAVecTraits<DType, 1> {
  using NopeVec = int;
  using RopeVec = short;
  // Number of DType elements per NopeVec: 4 / 1 = 4
  static constexpr int NOPE_ELEMS_PER_VEC = sizeof(NopeVec) / sizeof(DType);
  // Number of DType elements per RopeVec: 2 / 1 = 2
  static constexpr int ROPE_ELEMS_PER_VEC = sizeof(RopeVec) / sizeof(DType);

  static __forceinline__ __device__ NopeVec load_nope(const NopeVec* addr) {
    return ld_na_global_v1(reinterpret_cast<const int*>(addr));
  }
  static __forceinline__ __device__ void store_nope(NopeVec* addr, NopeVec val) {
    st_na_global_v1(reinterpret_cast<int*>(addr), val);
  }
  static __forceinline__ __device__ RopeVec load_rope(const RopeVec* addr) {
    return ld_na_global_v_short(reinterpret_cast<const short*>(addr));
  }
  static __forceinline__ __device__ void store_rope(RopeVec* addr, RopeVec val) {
    st_na_global_v_short(reinterpret_cast<short*>(addr), val);
  }
};

template <typename DType>
__global__ void ConcatMLAKKernel(DType* __restrict__ k, const DType* __restrict__ k_nope,
                                 const DType* __restrict__ k_rope, const int num_tokens,
                                 const int64_t k_stride_0, const int k_stride_1,
                                 const int64_t k_nope_stride_0, const int k_nope_stride_1,
                                 const int64_t k_rope_stride_0) {
  using Traits = ConcatMLAVecTraits<DType>;
  using NopeVec = typename Traits::NopeVec;
  using RopeVec = typename Traits::RopeVec;

  constexpr int NUM_LOCAL_HEADS = MLA_NUM_LOCAL_HEADS;
  constexpr int QK_NOPE_HEAD_DIM = MLA_QK_NOPE_HEAD_DIM;
  constexpr int QK_ROPE_HEAD_DIM = MLA_QK_ROPE_HEAD_DIM;
  constexpr int HEAD_CHUNK_SIZE = MLA_HEAD_CHUNK_SIZE;
  constexpr int NUM_HEAD_CHUNKS = MLA_NUM_HEAD_CHUNKS;

  static_assert(sizeof(NopeVec) * 32 == QK_NOPE_HEAD_DIM * sizeof(DType), "nope vec mismatch");
  static_assert(sizeof(RopeVec) * 32 == QK_ROPE_HEAD_DIM * sizeof(DType), "rope vec mismatch");

  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int token_id = flat_warp_id / NUM_HEAD_CHUNKS;
  const int head_chunk_id = flat_warp_id % NUM_HEAD_CHUNKS;
  const int lane_id = get_lane_id();

  if (token_id >= num_tokens) return;

  const int head_row0 = head_chunk_id * HEAD_CHUNK_SIZE;

  // Source pointer for k_nope (indexed by token and head)
  const NopeVec* __restrict__ nope_src =
      reinterpret_cast<const NopeVec*>(k_nope + token_id * k_nope_stride_0 +
                                       head_row0 * k_nope_stride_1) +
      lane_id;

  // Destination pointers for output k (nope part and rope part)
  NopeVec* __restrict__ nope_dst =
      reinterpret_cast<NopeVec*>(k + token_id * k_stride_0 + head_row0 * k_stride_1) + lane_id;

  RopeVec* __restrict__ rope_dst =
      reinterpret_cast<RopeVec*>(k + token_id * k_stride_0 + head_row0 * k_stride_1 +
                                 QK_NOPE_HEAD_DIM) +
      lane_id;

  // Stride calculations for vector types (in units of the respective vector type)
  constexpr int NOPE_ELEMS_PER_VEC = Traits::NOPE_ELEMS_PER_VEC;
  constexpr int ROPE_ELEMS_PER_VEC = Traits::ROPE_ELEMS_PER_VEC;
  const int nope_src_stride_v = k_nope_stride_1 / NOPE_ELEMS_PER_VEC;
  const int nope_dst_stride_v = k_stride_1 / NOPE_ELEMS_PER_VEC;
  const int rope_dst_stride_v = k_stride_1 / ROPE_ELEMS_PER_VEC;

  // Load rope value once - it's shared across all heads
  const RopeVec* rope_base = reinterpret_cast<const RopeVec*>(k_rope + token_id * k_rope_stride_0);
  const RopeVec rope_val = Traits::load_rope(rope_base + lane_id);

  // Prefetch first nope row and load it
  prefetch_L2(nope_src);
  NopeVec cur = Traits::load_nope(nope_src);

// Process all heads in this chunk with software pipelining
#pragma unroll
  for (int i = 0; i < HEAD_CHUNK_SIZE; ++i) {
    NopeVec next;
    if (i + 1 < HEAD_CHUNK_SIZE) {
      // Prefetch and load next row while processing current
      const NopeVec* next_src = nope_src + nope_src_stride_v;
      prefetch_L2(next_src);
      next = Traits::load_nope(next_src);
    }

    // Write current nope and rope values
    Traits::store_nope(nope_dst, cur);
    Traits::store_rope(rope_dst, rope_val);

    // Advance pointers
    nope_src += nope_src_stride_v;
    nope_dst += nope_dst_stride_v;
    rope_dst += rope_dst_stride_v;

    cur = next;
  }
}

/*!
 * \brief Launch the optimized ConcatMLAK kernel
 *
 * Concatenates k_nope and k_rope for MLA attention.
 *
 * \param k Output tensor [num_tokens, num_heads, nope_dim + rope_dim]
 * \param k_nope Input tensor [num_tokens, num_heads, nope_dim]
 * \param k_rope Input tensor [num_tokens, 1, rope_dim] (broadcast to all heads)
 * \param num_tokens Number of tokens
 * \param k_stride_0 Token stride for k
 * \param k_stride_1 Head stride for k
 * \param k_nope_stride_0 Token stride for k_nope
 * \param k_nope_stride_1 Head stride for k_nope
 * \param k_rope_stride_0 Token stride for k_rope
 * \param stream CUDA stream
 */
template <typename DType>
cudaError_t ConcatMLAK(DType* k, const DType* k_nope, const DType* k_rope, int num_tokens,
                       int64_t k_stride_0, int k_stride_1, int64_t k_nope_stride_0,
                       int k_nope_stride_1, int64_t k_rope_stride_0,
                       cudaStream_t stream = nullptr) {
  if (num_tokens == 0) {
    return cudaSuccess;
  }

  constexpr int NUM_HEAD_CHUNKS = MLA_NUM_HEAD_CHUNKS;
  constexpr int num_warps_per_block = 32;
  const int grid_size = ceil_div(num_tokens * NUM_HEAD_CHUNKS, num_warps_per_block);
  const int block_size = num_warps_per_block * 32;

  ConcatMLAKKernel<DType>
      <<<grid_size, block_size, 0, stream>>>(k, k_nope, k_rope, num_tokens, k_stride_0, k_stride_1,
                                             k_nope_stride_0, k_nope_stride_1, k_rope_stride_0);

  return cudaGetLastError();
}

}  // namespace flashinfer

#endif  // FLASHINFER_CONCAT_MLA_CUH_
