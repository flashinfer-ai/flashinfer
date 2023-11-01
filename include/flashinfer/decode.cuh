/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef FLASHINFER_DECODE_CUH_
#define FLASHINFER_DECODE_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>
#include <iostream>
#include <random>

#include "cp_async.cuh"
#include "layout.cuh"
#include "math.cuh"
#include "page.cuh"
#include "rope.cuh"
#include "state.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

namespace cg = cooperative_groups;

namespace {

/*!
 * \brief Apply RoPE (Rotary Positional Embeddings) to x[0: head_dim],
 *   return thread-local vector
 * \tparam vec_size A template integer indicates the vector size used
 *   in the kernel
 * \tparam bdx A template integer indicates the blockDim.x
 * \tparam T A template type indicates the x data type
 * \param x A pointer to the start of x data
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param offset A integer indicates the offset of the position in RoPE
 */
template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> apply_llama_rope(
    const T* x, const vec_t<float, vec_size>& freq, uint32_t offset) {
  constexpr uint32_t head_dim = vec_size * bdx;
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);
  permuted_vec.cast_load(x + ((threadIdx.x * vec_size < head_dim / 2)
                                  ? threadIdx.x * vec_size + head_dim / 2
                                  : threadIdx.x * vec_size - head_dim / 2));

#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    float embed = float(offset) * freq[i];
    float cos, sin;
    __sincosf(embed, &sin, &cos);
    vec[i] = vec[i] * cos + ((threadIdx.x * vec_size < head_dim / 2)
                                 ? -permuted_vec[i]
                                 : permuted_vec[i]) *
                                sin;
  }
  return vec;
}

/*!
 * \brief Load k tile from smem and compute qk.
 * \tparam rotary_mode The rotary mode used in the kernel
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 * in shared memory of different pipeline stages \param kv_idx A integer
 * indicates the thread-local kv position in kv-cache. \param compute_stage_idx
 * A integer indicates the compute stage index in the pipeline \param sm_scale A
 * float indicates the scale applied to pre-softmax logits \param x A float
 * indicates the thread-local result of qk
 */
template <RotaryMode rotary_mode, uint32_t vec_size, uint32_t bdx, uint32_t bdy,
          typename T>
__device__ __forceinline__ void compute_qk(const T* smem,
                                           const vec_t<float, vec_size>& q_vec,
                                           const vec_t<float, vec_size>& freq,
                                           uint32_t kv_idx_base,
                                           uint32_t compute_stage_idx,
                                           float sm_scale, float* x) {
  uint32_t tx = threadIdx.x, tz = threadIdx.z;
#pragma unroll
  for (uint32_t iy = 0; iy < bdy; ++iy) {
    vec_t<float, vec_size> k_vec;
    if constexpr (rotary_mode == RotaryMode::kLlama) {
      // apply rotary embedding for all rows in k matrix of kv-cache
      k_vec = apply_llama_rope<vec_size, bdx>(smem + iy * bdx * vec_size, freq,
                                              kv_idx_base + tz * bdy + iy);
    } else {
      // do not apply rotary embedding
      k_vec.cast_load(smem + (iy * bdx + tx) * vec_size);
    }
    x[iy] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      x[iy] += q_vec[i] * k_vec[i] * sm_scale;
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      x[iy] += math::shfl_xor_sync(x[iy], offset);
    }
  }
}

/*!
 * \brief Load v tile from shared memory and update partial state.
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam T A template type indicates the input data type
 * \tparam norm_on_the_fly Whether to normalize on the fly or not
 * \param smem A pointer to the start of shared memory
 * \param x A float indicates the pre-softmax logits
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 * in shared memory of different pipeline stages. \param compute_stage_idx A
 * integer indicates the compute stage index in the pipeline \param pred_guard A
 * boolean indicates whether the current thread is in the valid range \param s
 * The flashattention state to be updated
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, typename T,
          bool norm_on_the_fly>
__device__ __forceinline__ void update_partial_state(
    const T* smem, const float* x, uint32_t compute_stage_idx,
    uint32_t kv_idx_base, uint32_t kv_idx_bound,
    state_t<vec_size, norm_on_the_fly>& s) {
  uint32_t tx = threadIdx.x, tz = threadIdx.z;
#pragma unroll
  for (uint32_t iy = 0; iy < bdy; ++iy) {
    vec_t<float, vec_size> v_vec;
    v_vec.cast_load(smem + (iy * bdx + tx) * vec_size);
    if (kv_idx_base + tz * bdy + iy < kv_idx_bound) {
      s.merge(v_vec, x[iy]);
    }
  }
}

/*!
 * \brief Synchronize the state of all warps inside a threadblock.
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam norm_on_the_fly Whether to normalize on the fly or not
 * \param s The warp local state
 * \param smem The pointer to shared memory buffer for o
 * \param smem_md The pointer to shared memory buffer for m/d
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          bool norm_on_the_fly>
__device__ __forceinline__ void sync_state(
    state_t<vec_size, norm_on_the_fly>& s, float* smem, float* smem_md) {
  if constexpr (bdz > 1) {
    constexpr uint32_t head_dim = bdx * vec_size;
    auto block = cg::this_thread_block();
    uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    s.o.store(smem + (tz * bdy + ty) * head_dim + tx * vec_size);
    smem_md[(tz * bdy + ty) * 2] = s.m;
    smem_md[(tz * bdy + ty) * 2 + 1] = s.d;
    block.sync();
    s.init();
#pragma unroll
    for (uint32_t iz = 0; iz < bdz; ++iz) {
      float mz = smem_md[(iz * bdy + ty) * 2],
            dz = smem_md[(iz * bdy + ty) * 2 + 1];
      vec_t<float, vec_size> oz;
      oz.load(smem + (iz * bdy + ty) * head_dim + tx * vec_size);
      s.merge(oz, mz, dz);
    }
  }
}

}  // namespace

/*!
 * \brief FlashAttention decoding cuda kernel with kv-cache for a single
 * sequence, fused with RoPE.
 * \tparam layout The layout of k/v matrices (NHD or HND)
 * \tparam cooperative Whether to use cooperative kernel or not
 * \tparam norm_on_the_fly Whether to normalize on the fly or not
 * \tparam rotary_mode The rotary mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \param q [num_qo_heads, head_dim] The query matrix
 * \param k [seq_len, num_kv_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_kv_heads, head_dim] The value matrix in kv-cache
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param info The tensor info of k/v matrices
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param head_dim A integer indicates the head dimension
 * \param rope_inv_scale A floating number indicate the multiplicative inverse
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_inv_theta A floating number indicate the multiplicative inverse
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 * \param kv_chunk_size A integer indicates the kv-chunk size
 */
template <QKVLayout layout, bool cooperative, bool norm_on_the_fly,
          RotaryMode rotary_mode, uint32_t num_stages_smem, uint32_t vec_size,
          uint32_t bdx, uint32_t bdy, uint32_t bdz, typename DTypeIn,
          typename DTypeOut>
__global__ void SingleDecodeWithKVCacheKernel(
    DTypeIn* __restrict__ q, DTypeIn* __restrict__ k, DTypeIn* __restrict__ v,
    DTypeOut* __restrict__ o, float* __restrict__ tmp,
    tensor_info_t<layout, bdy> info, float sm_scale, float rope_inv_scale,
    float rope_inv_theta, uint32_t kv_chunk_size) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  sm_scale *= math::log2e;

  constexpr uint32_t head_dim = bdx * vec_size;
  uint32_t kv_head_idx = blockIdx.y;
  uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
  uint32_t kv_chunk_idx = blockIdx.x;
  uint32_t num_kv_chunks = gridDim.x;
  uint32_t num_qo_heads = info.get_num_qo_heads();
  uint32_t seq_len = info.kv_len;

  extern __shared__ uint8_t smem[];
  DTypeIn* k_smem = (DTypeIn*)smem;
  DTypeIn* v_smem = (DTypeIn*)(smem + num_stages_smem * bdy * bdz * head_dim *
                                          sizeof(DTypeIn));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * bdy * bdz * head_dim *
                                       sizeof(DTypeIn));

  uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (rotary_mode == RotaryMode::kLlama) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_inv_scale *
                powf(rope_inv_theta,
                     float(2 * ((tx * vec_size + i) % (head_dim / 2))) /
                         float(head_dim));
    }
    // apply rotary embedding to q matrix
    q_vec = apply_llama_rope<vec_size, bdx>(
        q + info.get_qo_elem_offset(0, qo_head_idx, 0), freq, seq_len - 1);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(q + info.get_qo_elem_offset(0, qo_head_idx, tx * vec_size));
  }
  block.sync();

  uint32_t chunk_start = kv_chunk_idx * kv_chunk_size;
  kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);
  uint32_t chunk_end = chunk_start + kv_chunk_size;

  // preload k tiles and v tiles
  uint32_t producer_kv_idx_base = chunk_start;
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    cp_async::pred_load<vec_bits, true, false>(
        k_smem + ((iter * bdz + tz) * bdy + ty) * head_dim + tx * vec_size,
        k + info.get_kv_elem_offset(producer_kv_idx_base + tz * bdy + ty,
                                    kv_head_idx, tx * vec_size),
        producer_kv_idx_base + tz * bdy + ty < chunk_end);
    cp_async::commit_group();
    cp_async::pred_load<vec_bits, true, false>(
        v_smem + ((iter * bdz + tz) * bdy + ty) * head_dim + tx * vec_size,
        v + info.get_kv_elem_offset(producer_kv_idx_base + tz * bdy + ty,
                                    kv_head_idx, tx * vec_size),
        producer_kv_idx_base + tz * bdy + ty < chunk_end);
    cp_async::commit_group();
    producer_kv_idx_base += bdy * bdz;
  }

  // pipelining k/v tiles loading and state updating
  uint32_t consumer_kv_idx_base = chunk_start, stage_idx = 0;
  state_t<vec_size, norm_on_the_fly> s_partial;
  float x[bdy];

#pragma unroll 4
  for (uint32_t iter = 0; iter < (kv_chunk_size + bdy * bdz - 1) / (bdy * bdz);
       ++iter) {
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<rotary_mode, vec_size, bdx, bdy>(
        k_smem + (stage_idx * bdz + tz) * bdy * head_dim, q_vec, freq,
        consumer_kv_idx_base, stage_idx, sm_scale, x);
    block.sync();
    // load k
    cp_async::pred_load<vec_bits, true, false>(
        k_smem + ((stage_idx * bdz + tz) * bdy + ty) * head_dim + tx * vec_size,
        k + info.get_kv_elem_offset(producer_kv_idx_base + tz * bdy + ty,
                                    kv_head_idx, tx * vec_size),
        producer_kv_idx_base + tz * bdy + ty < chunk_end);
    cp_async::commit_group();

    // update m/d/o state
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_partial_state<vec_size, bdx, bdy>(
        v_smem + (stage_idx * bdz + tz) * bdy * head_dim, x, stage_idx,
        consumer_kv_idx_base, chunk_end, s_partial);
    block.sync();

    // load v
    cp_async::pred_load<vec_bits, true, false>(
        v_smem + ((stage_idx * bdz + tz) * bdy + ty) * head_dim + tx * vec_size,
        v + info.get_kv_elem_offset(producer_kv_idx_base + tz * bdy + ty,
                                    kv_head_idx, tx * vec_size),
        producer_kv_idx_base + tz * bdy + ty < chunk_end);
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
    producer_kv_idx_base += bdy * bdz;
    consumer_kv_idx_base += bdy * bdz;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync partial state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(s_partial, reinterpret_cast<float*>(smem),
                                      smem_md);

  if constexpr (cooperative) {
    // update tmp buffer
    s_partial.o.store(tmp +
                      (qo_head_idx * num_kv_chunks + kv_chunk_idx) * head_dim +
                      tx * vec_size);
    float* tmp_md = tmp + num_qo_heads * num_kv_chunks * head_dim;
    *(float2*)&tmp_md[(qo_head_idx * num_kv_chunks + kv_chunk_idx) * 2] =
        make_float2(s_partial.m, s_partial.d);
    grid.sync();

    // sync global states
    if (kv_chunk_idx == 0) {
      state_t<vec_size, norm_on_the_fly> s_global;
#pragma unroll 4
      for (uint32_t iter = 0; iter < (num_kv_chunks + bdz - 1) / bdz; ++iter) {
        uint32_t kv_chunk_idx = iter * bdz + tz;
        if (kv_chunk_idx < num_kv_chunks) {
          float2 md = *(
              float2*)&tmp_md[(qo_head_idx * num_kv_chunks + kv_chunk_idx) * 2];
          s_partial.m = md.x;
          s_partial.d = md.y;
          s_partial.o.load(
              tmp + (qo_head_idx * num_kv_chunks + kv_chunk_idx) * head_dim +
              tx * vec_size);
          s_global.merge(s_partial);
        }
      }
      block.sync();
      // sync partial state of all warps inside a threadblock
      sync_state<vec_size, bdx, bdy, bdz>(
          s_global, reinterpret_cast<float*>(smem), smem_md);
      s_global.normalize();
      s_global.o.cast_store(
          o + info.get_qo_elem_offset(0, qo_head_idx, tx * vec_size));
    }
  } else {
    s_partial.normalize();
    s_partial.o.cast_store(
        o + info.get_qo_elem_offset(0, qo_head_idx, tx * vec_size));
  }
}

template <typename DType, typename IdType>
__forceinline__ __device__ void AdvancePageIterator(
    paged_kv_t<DType, IdType> paged_kv, uint32_t* kv_idx_base,
    uint32_t* valid_page_size, uint32_t& producer_valid_page_size,
    uint32_t& producer_entry_base, uint32_t& producer_page_iter,
    uint32_t& producer_page_idx, uint32_t cur_page_indptr_begin,
    uint32_t cur_page_indptr_end, uint32_t batch_idx, uint32_t stage_idx) {
  if (producer_entry_base >= producer_valid_page_size) {
    producer_entry_base = 0;
    producer_page_iter += 1;
    if (producer_page_iter < cur_page_indptr_end) {
      producer_page_idx = paged_kv.indices[producer_page_iter];
      producer_valid_page_size =
          paged_kv.get_valid_page_size(batch_idx, producer_page_iter);
    } else {
      producer_valid_page_size = 0;
    }
  }
  kv_idx_base[stage_idx] =
      producer_entry_base +
      (producer_page_iter - cur_page_indptr_begin) * paged_kv.page_size;
  valid_page_size[stage_idx] = producer_valid_page_size;
}

/*!
 * \brief FlashAttention decoding cuda kernel with PagedKVCcache for batch
 * requests, fused with RoPE. \tparam cooperative Whether to use cooperative
 * kernel or not \tparam rotary_mode The rotary mode \tparam norm_on_the_fly
 * Whether to normalize on the fly or not \tparam vec_size A template integer
 * indicates the vector size \tparam bdx A template integer indicates the block
 * size in x dimension \tparam bdy A template integer indicates the block size
 * in y dimension \tparam bdz A template integer indicates the block size in z
 * dimension \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param q [batch_size, num_qo_heads, head_dim] The query matrix
 * \param paged_kv The PagedKVCache data structure
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param rope_inv_scale A floating number indicate the multiplicative inverse
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_inv_theta A floating number indicate the multiplicative inverse
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 */
template <bool cooperative, RotaryMode rotary_mode, bool norm_on_the_fly,
          uint32_t num_stages_smem, uint32_t vec_size, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, typename DTypeIn, typename DTypeOut,
          typename IdType>
__global__ void BatchDecodeWithPagedKVCacheKernel(
    DTypeIn* __restrict__ q, paged_kv_t<DTypeIn, IdType> paged_kv,
    DTypeOut* __restrict__ o, float* __restrict__ tmp, float sm_scale,
    float rope_inv_scale, float rope_inv_theta) {
  auto block = cg::this_thread_block();
  sm_scale *= math::log2e;

  constexpr uint32_t head_dim = bdx * vec_size;
  const uint32_t batch_idx = blockIdx.x;
  const uint32_t kv_head_idx = blockIdx.y;
  const uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
  const uint32_t num_qo_heads = gridDim.y * bdy;
  const uint32_t cur_chunk_start =
      cooperative ? paged_kv.chunk_start[batch_idx] : 0U;
  const uint32_t cur_page_indptr_begin = paged_kv.indptr[batch_idx],
                 cur_page_indptr_end = paged_kv.indptr[batch_idx + 1];
  const uint32_t cur_last_page_offset = paged_kv.last_page_offset[batch_idx];
  const uint32_t seq_len =
      cooperative ? paged_kv.seq_lens_before_split[batch_idx]
                  : (cur_page_indptr_end - cur_page_indptr_begin - 1) *
                            paged_kv.page_size +
                        cur_last_page_offset;

  extern __shared__ uint8_t smem[];
  DTypeIn* k_smem = (DTypeIn*)smem;
  DTypeIn* v_smem = (DTypeIn*)(smem + num_stages_smem * bdy * bdz * head_dim *
                                          sizeof(DTypeIn));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * bdy * bdz * head_dim *
                                       sizeof(DTypeIn));

  const uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (rotary_mode == RotaryMode::kLlama) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_inv_scale *
                __powf(rope_inv_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) /
                           float(head_dim));
    }
    // apply rotary embedding to q matrix
    if constexpr (cooperative) {
      q_vec = apply_llama_rope<vec_size, bdx>(
          q + (paged_kv.batch_idx_map[batch_idx] * num_qo_heads + qo_head_idx) *
                  head_dim,
          freq, seq_len - 1);
    } else {
      q_vec = apply_llama_rope<vec_size, bdx>(
          q + (batch_idx * num_qo_heads + qo_head_idx) * head_dim, freq,
          seq_len - 1);
    }
  } else {
    // do not apply rotary embedding to q matrix
    if constexpr (cooperative) {
      q_vec.cast_load(
          q +
          (paged_kv.batch_idx_map[batch_idx] * num_qo_heads + qo_head_idx) *
              head_dim +
          tx * vec_size);
    } else {
      q_vec.cast_load(q + (batch_idx * num_qo_heads + qo_head_idx) * head_dim +
                      tx * vec_size);
    }
  }
  block.sync();

  // preload k/v tiles
  uint32_t producer_entry_base = 0, stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
  uint32_t producer_page_iter = cur_page_indptr_begin;
  uint32_t producer_page_idx = paged_kv.indices[producer_page_iter];
  uint32_t producer_valid_page_size =
      paged_kv.get_valid_page_size(batch_idx, producer_page_iter);
  uint32_t kv_idx_base[num_stages_smem]{0};
  uint32_t valid_page_size[num_stages_smem]{0};
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    AdvancePageIterator(
        paged_kv, kv_idx_base, valid_page_size, producer_valid_page_size,
        producer_entry_base, producer_page_iter, producer_page_idx,
        cur_page_indptr_begin, cur_page_indptr_end, batch_idx, stage_idx);
    bool producer_pred_guard =
        (producer_entry_base + tz * bdy + ty < producer_valid_page_size) &&
        (producer_page_iter < cur_page_indptr_end);
    cp_async::pred_load<vec_bits, true, false>(
        k_smem + ((stage_idx * bdz + tz) * bdy + ty) * head_dim + tx * vec_size,
        paged_kv.data + paged_kv.get_k_elem_offset(
                            producer_page_idx, kv_head_idx,
                            producer_entry_base + tz * bdy + ty, tx * vec_size),
        producer_pred_guard);
    cp_async::commit_group();
    cp_async::pred_load<vec_bits, true, false>(
        v_smem + ((stage_idx * bdz + tz) * bdy + ty) * head_dim + tx * vec_size,
        paged_kv.data + paged_kv.get_v_elem_offset(
                            producer_page_idx, kv_head_idx,
                            producer_entry_base + tz * bdy + ty, tx * vec_size),
        producer_pred_guard);
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
    producer_entry_base += bdy * bdz;
  }

  state_t<vec_size, norm_on_the_fly> s;
  float x[bdy];
  uint32_t consumer_kv_idx_base = 0;

  for (uint32_t consumer_page_iter = cur_page_indptr_begin;
       consumer_page_iter < cur_page_indptr_end; ++consumer_page_iter) {
    uint32_t consumer_valid_page_size = valid_page_size[stage_idx];
#pragma unroll
    for (uint32_t iter = 0;
         iter < (consumer_valid_page_size + (bdy * bdz) - 1) / (bdy * bdz);
         ++iter) {
      consumer_kv_idx_base = kv_idx_base[stage_idx];
      AdvancePageIterator(
          paged_kv, kv_idx_base, valid_page_size, producer_valid_page_size,
          producer_entry_base, producer_page_iter, producer_page_idx,
          cur_page_indptr_begin, cur_page_indptr_end, batch_idx, stage_idx);
      bool producer_pred_guard =
          (producer_entry_base + tz * bdy + ty < producer_valid_page_size) &&
          (producer_page_iter < cur_page_indptr_end);
      // compute qk
      cp_async::wait_group<2 * num_stages_smem - 1>();
      block.sync();
      compute_qk<rotary_mode, vec_size, bdx, bdy>(
          k_smem + (stage_idx * bdz + tz) * bdy * head_dim, q_vec, freq,
          cur_chunk_start + consumer_kv_idx_base, stage_idx, sm_scale, x);
      block.sync();

      // load k tiles
      cp_async::pred_load<vec_bits, true, false>(
          k_smem + ((stage_idx * bdz + tz) * bdy + ty) * head_dim +
              tx * vec_size,
          paged_kv.data +
              paged_kv.get_k_elem_offset(producer_page_idx, kv_head_idx,
                                         producer_entry_base + tz * bdy + ty,
                                         tx * vec_size),
          producer_pred_guard);
      cp_async::commit_group();

      // update m/d/o states
      cp_async::wait_group<2 * num_stages_smem - 1>();
      block.sync();
      update_partial_state<vec_size, bdx, bdy>(
          v_smem + (stage_idx * bdz + tz) * bdy * head_dim, x, stage_idx,
          iter * bdy * bdz, consumer_valid_page_size, s);
      block.sync();

      // load v tiles
      cp_async::pred_load<vec_bits, true, false>(
          v_smem + ((stage_idx * bdz + tz) * bdy + ty) * head_dim +
              tx * vec_size,
          paged_kv.data +
              paged_kv.get_v_elem_offset(producer_page_idx, kv_head_idx,
                                         producer_entry_base + tz * bdy + ty,
                                         tx * vec_size),
          producer_pred_guard);
      cp_async::commit_group();

      stage_idx = (stage_idx + 1) % num_stages_smem;
      producer_entry_base += bdy * bdz;
    }
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync partial state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(s, reinterpret_cast<float*>(smem),
                                      smem_md);

  if constexpr (cooperative) {
    auto grid = cg::this_grid();
    // update tmp buffer
    s.o.store(tmp + (qo_head_idx * paged_kv.batch_size + batch_idx) * head_dim +
              tx * vec_size);
    float* tmp_md = tmp + num_qo_heads * paged_kv.batch_size * head_dim;
    *(float2*)&tmp_md[(qo_head_idx * paged_kv.batch_size + batch_idx) * 2] =
        make_float2(s.m, s.d);
    grid.sync();

    // sync global states
    const uint32_t cooperative_indptr_begin =
                       paged_kv.cooperative_indptr[batch_idx],
                   cooperative_indptr_end =
                       paged_kv.cooperative_indptr[batch_idx + 1];
    if (cooperative_indptr_begin < cooperative_indptr_end) {
      state_t<vec_size, norm_on_the_fly> s_global;
      const uint32_t num_pages =
          cooperative_indptr_end - cooperative_indptr_begin;
#pragma unroll 4
      for (uint32_t iter = 0; iter < (num_pages + bdz - 1) / bdz; ++iter) {
        uint32_t kv_chunk_idx = cooperative_indptr_begin + iter * bdz + tz;
        if (kv_chunk_idx < cooperative_indptr_end) {
          float2 md = *(float2*)&tmp_md[(qo_head_idx * paged_kv.batch_size +
                                         kv_chunk_idx) *
                                        2];
          s.m = md.x;
          s.d = md.y;
          s.o.load(tmp +
                   (qo_head_idx * paged_kv.batch_size + kv_chunk_idx) *
                       head_dim +
                   tx * vec_size);
          s_global.merge(s);
        }
      }
      block.sync();
      // sync partial state of all warps inside a threadblock
      sync_state<vec_size, bdx, bdy, bdz>(
          s_global, reinterpret_cast<float*>(smem), smem_md);
      s_global.normalize();
      s_global.o.cast_store(
          o +
          (paged_kv.batch_idx_map[batch_idx] * num_qo_heads + qo_head_idx) *
              head_dim +
          tx * vec_size);
    }
  } else {
    s.normalize();
    s.o.cast_store(o + (batch_idx * num_qo_heads + qo_head_idx) * head_dim +
                   tx * vec_size);
  }
}

constexpr uint32_t get_heuristic_num_threads(uint32_t group_size,
                                             uint32_t sizeof_dtype) {
  if (group_size == 8U) {
    if (sizeof_dtype == 1U) {
      return 256U;  // not enough registers for 512 threads
    } else {
      return 512U;
    }
  } else {
    return 128U;
  }
}

template <typename DTypeIn, typename DTypeOut>
cudaError_t SingleDecodeWithKVCacheWorkEstimation(
    uint32_t& tmp_size, uint32_t& max_grid_size, uint32_t num_qo_heads,
    uint32_t num_kv_heads, uint32_t seq_len, uint32_t head_dim,
    QKVLayout layout = QKVLayout::kNHD,
    RotaryMode rotary_mode = RotaryMode::kNone, cudaStream_t stream = nullptr) {
  if (seq_len <= 128U) {
    tmp_size = 0;
  } else {
    SWITCH_GQA_GROUP_SIZE(
        num_qo_heads / num_kv_heads, GROUP_SIZE,
        {SWITCH_HEAD_DIM(
            head_dim, HEAD_DIM,
            {SWITCH_ROTARY_MODE(
                rotary_mode, ROTARY_MODE, {SWITCH_LAYOUT(layout, QKV_LAYOUT, {
                  constexpr uint32_t vec_size =
                      std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
                  constexpr uint32_t num_stages_smem = 2U;
                  constexpr uint32_t bdx = HEAD_DIM / vec_size;
                  static_assert(bdx <= 32U);
                  constexpr uint32_t bdy = GROUP_SIZE;
                  constexpr uint32_t num_threads =
                      get_heuristic_num_threads(GROUP_SIZE, sizeof(DTypeIn));
                  constexpr uint32_t bdz = num_threads / (bdx * bdy);
                  constexpr bool norm_on_the_fly = false;
                  const uint32_t smem_size = 2U * num_stages_smem * bdy * bdz *
                                                 head_dim * sizeof(DTypeIn) +
                                             2U * bdy * bdz * sizeof(float);

                  auto kernel = SingleDecodeWithKVCacheKernel<
                      QKV_LAYOUT, true, norm_on_the_fly, ROTARY_MODE,
                      num_stages_smem, vec_size, bdx, bdy, bdz, DTypeIn,
                      DTypeOut>;
                  int num_blocks_per_sm = 0;
                  int num_sm = 0;
                  int dev_id = 0;
                  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
                  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
                      &num_sm, cudaDevAttrMultiProcessorCount, dev_id));
                  FLASHINFER_CUDA_CALL(
                      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                          &num_blocks_per_sm, kernel, num_threads, smem_size));
                  max_grid_size =
                      uint32_t(num_blocks_per_sm) * uint32_t(num_sm);
                  uint32_t max_num_kv_chunks = max_grid_size / num_kv_heads;
                  uint32_t kv_chunk_size = max(
                      (seq_len + max_num_kv_chunks - 1U) / max_num_kv_chunks,
                      min(num_threads, max(num_threads / 8,
                                           seq_len / max(1U, (num_threads /
                                                              num_kv_heads)))));
                  uint32_t num_kv_chunks =
                      (seq_len + kv_chunk_size - 1) / kv_chunk_size;
                  tmp_size = num_qo_heads * num_kv_chunks * (head_dim + 2);
                })})})});
  }
  return cudaSuccess;
}

/*!
 * \brief FlashAttention decoding with kv-cache for a single sequence
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \param q The query matrix, shape: [num_qo_heads, head_dim]
 * \param k The key matrix in kv-cache, shape: [seq_len, num_kv_heads, head_dim]
 *   for NHD layout, [num_kv_heads, head_dim, seq_len] for HND layout
 * \param v The value matrix in kv-cache, shape: [seq_len, num_kv_heads,
 * head_dim] for NHD layout, [num_kv_heads, head_dim, seq_len] for HND layout
 * \param o The output matrix, shape: [num_qo_heads, head_dim]
 * \param tmp Used-allocated temporary buffer
 * \param num_qo_heads A integer indicates the number of heads of query and
 * output \param num_kv_heads A integer indicates the number of heads of key and
 * value \param seq_len A integer indicates the sequence length \param head_dim
 * A integer indicates the head dimension \param layout The layout of q/k/v
 * matrices. \param rotary_mode The rotary mode \param rope_scale A floating
 * point number indicate the scaling ratio used in RoPE Interpolation. \param
 * rope_theta A floating point number indicate the "theta" used in RoPE \param
 * stream The cuda stream to launch the kernel
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t SingleDecodeWithKVCache(
    DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeOut* o, float* tmp,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t seq_len,
    uint32_t head_dim, QKVLayout layout = QKVLayout::kNHD,
    RotaryMode rotary_mode = RotaryMode::kNone, float rope_scale = 1.f,
    float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const float sm_scale = 1.f / std::sqrt(float(head_dim));
  const float rope_inv_scale = 1.f / rope_scale;
  const float rope_inv_theta = 1.f / rope_theta;
  constexpr bool norm_on_the_fly = false;
  assert(num_qo_heads % num_kv_heads == 0);

  SWITCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {SWITCH_HEAD_DIM(
          head_dim, HEAD_DIM,
          {SWITCH_ROTARY_MODE(
              rotary_mode, ROTARY_MODE, {SWITCH_LAYOUT(layout, QKV_LAYOUT, {
                constexpr uint32_t vec_size =
                    std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
                constexpr uint32_t num_stages_smem = 2U;
                constexpr uint32_t bdx = HEAD_DIM / vec_size;
                static_assert(bdx <= 32U);
                constexpr uint32_t bdy = GROUP_SIZE;
                constexpr uint32_t num_threads =
                    get_heuristic_num_threads(GROUP_SIZE, sizeof(DTypeIn));
                constexpr uint32_t bdz = num_threads / (bdx * bdy);
                tensor_info_t<QKV_LAYOUT, GROUP_SIZE> info(
                    1, seq_len, num_kv_heads, head_dim);
                const uint32_t smem_size = 2U * num_stages_smem * bdy * bdz *
                                               head_dim * sizeof(DTypeIn) +
                                           2U * bdy * bdz * sizeof(float);
                if (seq_len <= 128U || tmp == nullptr) {
                  // no need to use cooperative kernel
                  auto kernel = SingleDecodeWithKVCacheKernel<
                      QKV_LAYOUT, false, norm_on_the_fly, ROTARY_MODE,
                      num_stages_smem, vec_size, bdx, bdy, bdz, DTypeIn,
                      DTypeOut>;
                  FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                      smem_size));

                  dim3 nblks = dim3(1, num_kv_heads);
                  dim3 nthrs = dim3(bdx, bdy, bdz);
                  void* args[] = {(void*)&q,
                                  (void*)&k,
                                  (void*)&v,
                                  (void*)&o,
                                  (void*)&tmp,
                                  (void*)&info,
                                  (void*)&sm_scale,
                                  (void*)&rope_inv_scale,
                                  (void*)&rope_inv_theta,
                                  (void*)&seq_len};
                  FLASHINFER_CUDA_CALL(cudaLaunchKernel(
                      (void*)kernel, nblks, nthrs, args, smem_size, stream));
                } else {
                  // use cooperative kernel
                  auto kernel = SingleDecodeWithKVCacheKernel<
                      QKV_LAYOUT, true, norm_on_the_fly, ROTARY_MODE,
                      num_stages_smem, vec_size, bdx, bdy, bdz, DTypeIn,
                      DTypeOut>;
                  FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                      smem_size));

                  int num_blocks_per_sm = 0;
                  int num_sm = 0;
                  int dev_id = 0;
                  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
                  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
                      &num_sm, cudaDevAttrMultiProcessorCount, dev_id));
                  FLASHINFER_CUDA_CALL(
                      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                          &num_blocks_per_sm, kernel, num_threads, smem_size));
                  uint32_t max_grid_size =
                      uint32_t(num_blocks_per_sm) * uint32_t(num_sm);
                  uint32_t max_num_kv_chunks = max_grid_size / num_kv_heads;
                  uint32_t kv_chunk_size = max(
                      (seq_len + max_num_kv_chunks - 1U) / max_num_kv_chunks,
                      min(num_threads, max(num_threads / 8,
                                           seq_len / max(1U, (num_threads /
                                                              num_kv_heads)))));
                  dim3 nblks =
                      dim3((seq_len + kv_chunk_size - 1) / kv_chunk_size,
                           num_kv_heads);
                  assert(nblks.x > 0 && nblks.y > 0);
                  dim3 nthrs = dim3(bdx, bdy, bdz);
                  void* args[] = {(void*)&q,
                                  (void*)&k,
                                  (void*)&v,
                                  (void*)&o,
                                  (void*)&tmp,
                                  (void*)&info,
                                  (void*)&sm_scale,
                                  (void*)&rope_inv_scale,
                                  (void*)&rope_inv_theta,
                                  (void*)&kv_chunk_size};
                  FLASHINFER_CUDA_CALL(cudaLaunchCooperativeKernel(
                      (void*)kernel, nblks, nthrs, args, smem_size, stream));
                }
              })})})});
  return cudaSuccess;
}

template <typename DTypeIn, typename IdType>
cudaError_t SplitPagedKVCache(uint32_t old_batch_size,
                              const IdType* old_page_indptr_h,
                              const IdType* old_last_page_offset_h,
                              uint32_t max_num_pages_per_batch,
                              paged_kv_t<DTypeIn, IdType>* new_paged_kv_d,
                              cudaStream_t stream = nullptr) {
  std::vector<IdType> new_page_indptr_h{0}, new_last_page_offset_h,
      cooperative_indptr_h{0}, batch_idx_map_h, chunk_start_h,
      seq_lens_before_split_h;

  for (uint32_t batch_idx = 0; batch_idx < old_batch_size; batch_idx++) {
    uint32_t cooperative_indptr_delta =
        (old_page_indptr_h[batch_idx + 1] - old_page_indptr_h[batch_idx] +
         max_num_pages_per_batch - 1) /
        max_num_pages_per_batch;
    uint32_t seq_len_before_split =
        (old_page_indptr_h[batch_idx + 1] - old_page_indptr_h[batch_idx] - 1) *
            new_paged_kv_d->page_size +
        old_last_page_offset_h[batch_idx];
    for (uint32_t j = 0; j < cooperative_indptr_delta; ++j) {
      bool is_last = (j + 1) == cooperative_indptr_delta;
      new_page_indptr_h.push_back(
          min(old_page_indptr_h[batch_idx] + (j + 1) * max_num_pages_per_batch,
              old_page_indptr_h[batch_idx + 1]));
      new_last_page_offset_h.push_back(is_last
                                           ? old_last_page_offset_h[batch_idx]
                                           : new_paged_kv_d->page_size);
      batch_idx_map_h.push_back(batch_idx);
      if (j == 0) {
        cooperative_indptr_h.push_back(cooperative_indptr_h.back() +
                                       cooperative_indptr_delta);
      } else {
        cooperative_indptr_h.push_back(cooperative_indptr_h.back());
      }
      chunk_start_h.push_back(j * max_num_pages_per_batch *
                              new_paged_kv_d->page_size);
      seq_lens_before_split_h.push_back(seq_len_before_split);
    }
  }

  FLASHINFER_CUDA_CALL(
      cudaMemcpyAsync(new_paged_kv_d->indptr, new_page_indptr_h.data(),
                      sizeof(IdType) * new_page_indptr_h.size(),
                      cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(
      new_paged_kv_d->last_page_offset, new_last_page_offset_h.data(),
      sizeof(IdType) * new_last_page_offset_h.size(), cudaMemcpyHostToDevice,
      stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(
      new_paged_kv_d->cooperative_indptr, cooperative_indptr_h.data(),
      sizeof(IdType) * cooperative_indptr_h.size(), cudaMemcpyHostToDevice,
      stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(
      new_paged_kv_d->batch_idx_map, batch_idx_map_h.data(),
      sizeof(IdType) * batch_idx_map_h.size(), cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(
      new_paged_kv_d->chunk_start, chunk_start_h.data(),
      sizeof(IdType) * chunk_start_h.size(), cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(
      new_paged_kv_d->seq_lens_before_split, seq_lens_before_split_h.data(),
      sizeof(IdType) * seq_lens_before_split_h.size(), cudaMemcpyHostToDevice,
      stream));
  return cudaSuccess;
}

/*!
 * \brief Compute the maximum number of pages per batch and the new batch size
 * after we split Paged KV-Cache into multiple chunks on KV sequence length
 * dimension. \param max_grid_size The maximum grid size of the kernel \param
 * num_kv_heads The number of KV heads \param num_pages The number of pages per
 * request in the batch \param max_num_pages_per_batch_lb The pre-set lower
 * bound of maximum number of pages per batch, default to 1 \return
 * (max_num_pages_per_batch, new_batch_size) The number of pages per batch and
 * the new batch size after the split.
 */
template <typename IdType>
std::pair<uint32_t, uint32_t> SplitPagedKVCacheBinarySearchMinNumPagePerBatch(
    const uint32_t max_grid_size, const uint32_t num_kv_heads,
    const std::vector<IdType>& num_pages,
    const uint32_t min_num_pages_per_batch = 1) {
  uint32_t low = min_num_pages_per_batch, high = 0;
  for (const IdType& elem : num_pages) {
    high = max(high, elem);
  }
  uint32_t new_batch_size;
  while (low < high) {
    uint32_t mid = (low + high) / 2;
    new_batch_size = 0;
    for (const IdType& elem : num_pages) {
      new_batch_size += (elem + mid - 1) / mid;
    }
    if (new_batch_size * num_kv_heads > max_grid_size) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  new_batch_size = 0;
  for (const IdType& elem : num_pages) {
    new_batch_size += (elem + low - 1) / low;
  }
  return {low, new_batch_size};
}

template <typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheWorkEstimation(
    uint32_t& tmp_size, uint32_t& max_grid_size,
    uint32_t& max_num_pages_per_batch, uint32_t& new_batch_size,
    const paged_kv_t<DTypeIn, IdType>& paged_kv, uint32_t num_qo_heads,
    RotaryMode rotary_mode = RotaryMode::kNone, cudaStream_t stream = nullptr) {
  constexpr bool norm_on_the_fly = false;
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t batch_size = paged_kv.batch_size;
  const uint32_t num_kv_heads = paged_kv.num_heads;
  SWITCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {SWITCH_HEAD_DIM(
          head_dim, HEAD_DIM, {SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
            constexpr uint32_t vec_size =
                std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
            constexpr uint32_t num_stages_smem = 2;
            constexpr uint32_t bdx = HEAD_DIM / vec_size;
            static_assert(bdx <= 32);
            constexpr uint32_t bdy = GROUP_SIZE;
            constexpr uint32_t num_threads =
                get_heuristic_num_threads(GROUP_SIZE, sizeof(DTypeIn));
            constexpr uint32_t bdz = num_threads / (bdx * bdy);
            const uint32_t smem_size =
                2 * num_stages_smem * bdy * bdz * head_dim * sizeof(DTypeIn) +
                2 * bdy * bdz * sizeof(float);

            auto cooperative_kernel = BatchDecodeWithPagedKVCacheKernel<
                true, ROTARY_MODE, norm_on_the_fly, num_stages_smem, vec_size,
                bdx, bdy, bdz, DTypeIn, DTypeOut, IdType>;
            int num_blocks_per_sm = 0;
            int num_sm = 0;
            int dev_id = 0;
            FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
            FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
                &num_sm, cudaDevAttrMultiProcessorCount, dev_id));
            FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &num_blocks_per_sm, cooperative_kernel, num_threads,
                smem_size));
            max_grid_size = num_blocks_per_sm * num_sm;
            const uint32_t num_kv_heads = paged_kv.num_heads;
            if (batch_size * num_kv_heads >= max_grid_size) {
              // do not use cooperative kernel
              tmp_size = 0;
            } else {
              // compute max_num_pages_per_batch and new_batch_size
              std::vector<IdType> page_indptr_h(batch_size + 1),
                  num_pages(batch_size);
              FLASHINFER_CUDA_CALL(
                  cudaMemcpyAsync(page_indptr_h.data(), paged_kv.indptr,
                                  sizeof(IdType) * (batch_size + 1),
                                  cudaMemcpyDeviceToHost, stream));
              FLASHINFER_CUDA_CALL(cudaStreamSynchronize(stream));
              for (uint32_t batch_idx = 0; batch_idx < batch_size;
                   ++batch_idx) {
                num_pages[batch_idx] =
                    page_indptr_h[batch_idx + 1] - page_indptr_h[batch_idx];
              }
              std::tie(max_num_pages_per_batch, new_batch_size) =
                  SplitPagedKVCacheBinarySearchMinNumPagePerBatch(
                      max_grid_size, num_kv_heads, num_pages,
                      128 / paged_kv.page_size);
              tmp_size = num_qo_heads * new_batch_size * (head_dim + 2);
            }
          })})});
  return cudaSuccess;
}

/*!
 * \brief FlashAttention decoding cuda kernel with paged kv-cache for batched
 * requests \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type used in paged
 * kv-cache \param q [batch_size, num_qo_heads, head_dim] The query matrix
 * \param paged_kv The paged kv cache data structure
 * \param o [batch_size, num_qo_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param num_qo_heads A integer indicates the number of heads of query and
 * output \param rotary_mode The rotary mode \param rope_scale A floating point
 * number indicate the scaling ratio used in RoPE Interpolation. \param
 * rope_theta A floating point number indicate the "theta" used in RoPE \param
 * stream The cuda stream to launch the kernel
 */
template <typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCache(
    DTypeIn* q, paged_kv_t<DTypeIn, IdType> paged_kv, DTypeOut* o, float* tmp,
    uint32_t num_qo_heads, RotaryMode rotary_mode = RotaryMode::kNone,
    float rope_scale = 1.f, float rope_theta = 1e4,
    cudaStream_t stream = nullptr) {
  const float sm_scale = 1.f / std::sqrt(float(paged_kv.head_dim));
  const float rope_inv_scale = 1.f / rope_scale;
  const float rope_inv_theta = 1.f / rope_theta;
  constexpr bool norm_on_the_fly = false;
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t batch_size = paged_kv.batch_size;
  assert(num_qo_heads % num_kv_heads == 0);

  SWITCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {SWITCH_HEAD_DIM(
          head_dim, HEAD_DIM, {SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
            constexpr uint32_t vec_size =
                std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
            constexpr uint32_t num_stages_smem = 2;
            constexpr uint32_t bdx = HEAD_DIM / vec_size;
            static_assert(bdx <= 32);
            constexpr uint32_t bdy = GROUP_SIZE;
            constexpr uint32_t num_threads =
                get_heuristic_num_threads(GROUP_SIZE, sizeof(DTypeIn));
            constexpr uint32_t bdz = num_threads / (bdx * bdy);
            const uint32_t smem_size =
                2 * num_stages_smem * bdy * bdz * head_dim * sizeof(DTypeIn) +
                2 * bdy * bdz * sizeof(float);

            if (tmp == nullptr) {
              // do not use cooperative kernel
              dim3 nblks(batch_size, num_kv_heads);
              dim3 nthrs(bdx, bdy, bdz);
              auto kernel = BatchDecodeWithPagedKVCacheKernel<
                  false, ROTARY_MODE, norm_on_the_fly, num_stages_smem,
                  vec_size, bdx, bdy, bdz, DTypeIn, DTypeOut, IdType>;
              FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                  smem_size));
              void* args[] = {(void*)&q,
                              (void*)&paged_kv,
                              (void*)&o,
                              (void*)&tmp,
                              (void*)&sm_scale,
                              (void*)&rope_inv_scale,
                              (void*)&rope_inv_theta};
              FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs,
                                                    args, smem_size, stream));
            } else {
              // use cooperative kernel
              assert(paged_kv.cooperative_indptr != nullptr);
              assert(paged_kv.batch_idx_map != nullptr);
              assert(paged_kv.chunk_start != nullptr);
              assert(paged_kv.seq_lens_before_split != nullptr);
              auto cooperative_kernel = BatchDecodeWithPagedKVCacheKernel<
                  true, ROTARY_MODE, norm_on_the_fly, num_stages_smem, vec_size,
                  bdx, bdy, bdz, DTypeIn, DTypeOut, IdType>;
              FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                  cooperative_kernel,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
              void* args[] = {(void*)&q,
                              (void*)&paged_kv,
                              (void*)&o,
                              (void*)&tmp,
                              (void*)&sm_scale,
                              (void*)&rope_inv_scale,
                              (void*)&rope_inv_theta};
              dim3 nblks(batch_size, num_kv_heads);
              dim3 nthrs(bdx, bdy, bdz);
              FLASHINFER_CUDA_CALL(
                  cudaLaunchCooperativeKernel((void*)cooperative_kernel, nblks,
                                              nthrs, args, smem_size, stream));
            }
          })})});

  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_CUH_
