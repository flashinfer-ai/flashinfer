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
#ifdef FLASHINFER_ENABLE_FP8
#include <cuda_fp8.h>
#endif
#include <cuda_runtime.h>

#include <cuda/pipeline>
#include <iostream>
#include <random>

#include "../cp_async.cuh"
#include "../layout.cuh"
#include "../math.cuh"
#include "../page.cuh"
#include "../rope.cuh"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "cascade.cuh"
#include "state.cuh"

namespace flashinfer {

namespace cg = cooperative_groups;
using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

namespace {

/*!
 * \brief Load k tile from smem and compute qk
 * \tparam rotary_mode The rotary mode used in the kernel
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 *   in shared memory of different pipeline stages
 * \param kv_idx A integer indicates the thread-local kv position in kv-cache
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param s A float indicates the thread-local result of qk
 * \param st The self-attention state to be updated
 */
template <RotaryMode rotary_mode, uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename T>
__device__ __forceinline__ void compute_qk(const T* smem, uint32_t compute_stage_idx,
                                           const vec_t<float, vec_size>& q_vec,
                                           const vec_t<float, vec_size>& freq, uint32_t kv_idx_base,
                                           uint32_t iter_base, uint32_t iter_bound, float* s,
                                           state_t<vec_size>& st) {
  uint32_t tx = threadIdx.x, tz = threadIdx.z;
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> k_vec;
    if constexpr (rotary_mode == RotaryMode::kLlama) {
      // apply rotary embedding for all rows in k matrix of kv-cache
      k_vec = vec_apply_llama_rope<vec_size, bdx>(smem + j * bdx * vec_size, freq,
                                                  kv_idx_base + tz * tile_size + j);
    } else {
      // do not apply rotary embedding
      k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
    }
    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s[j] += q_vec[i] * k_vec[i];
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    s[j] = (iter_base + tz * tile_size + j < iter_bound) ? s[j] : -5e4;
    st.m = max(st.m, s[j]);
  }

  float o_scale = math::ptx_exp2(m_prev - st.m);
  st.d *= o_scale;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    s[j] = math::ptx_exp2(s[j] - st.m);
    st.d += s[j];
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    st.o[i] = st.o[i] * o_scale;
  }
}

/*!
 * \brief Load v tile from shared memory and update local state
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param s A float indicates the pre-softmax attention score
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 * in shared memory of different pipeline stages
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param st The flashattention state to be updated
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename T>
__device__ __forceinline__ void update_local_state(const T* smem, const float* s,
                                                   uint32_t compute_stage_idx,
                                                   state_t<vec_size>& st) {
  uint32_t tx = threadIdx.x;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> v_vec;
    v_vec.cast_load(smem + (j * bdx + tx) * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] + s[j] * v_vec[i];
    }
  }
}

/*!
 * \brief Synchronize the state of all warps inside a threadblock.
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \param st The warp local state
 * \param smem The pointer to shared memory buffer for o
 * \param smem_md The pointer to shared memory buffer for m/d
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz>
__device__ __forceinline__ void sync_state(state_t<vec_size>& st, float* smem, float* smem_md) {
  if constexpr (bdz > 1) {
    constexpr uint32_t head_dim = bdx * vec_size;
    auto block = cg::this_thread_block();
    uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    st.o.store(smem + (tz * bdy + ty) * head_dim + tx * vec_size);
    smem_md[(tz * bdy + ty) * 2] = st.m;
    smem_md[(tz * bdy + ty) * 2 + 1] = st.d;
    block.sync();
    st.init();
#pragma unroll
    for (uint32_t j = 0; j < bdz; ++j) {
      float mz = smem_md[(j * bdy + ty) * 2], dz = smem_md[(j * bdy + ty) * 2 + 1];
      vec_t<float, vec_size> oz;
      oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
      st.merge(oz, mz, dz);
    }
  }
}

}  // namespace

/*!
 * \brief FlashAttention decoding cuda kernel with kv-cache for a single request
 * \tparam kv_layout The layout of k/v matrices (NHD or HND)
 * \tparam partition_kv Whether to partition kv-cache on sequence length dimension or not
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
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 * \param kv_chunk_size A integer indicates the kv-chunk size
 */
template <QKVLayout kv_layout, bool partition_kv, RotaryMode rotary_mode, uint32_t num_stages_smem,
          uint32_t tile_size_per_bdx, uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          typename DTypeIn, typename DTypeOut>
__global__ void SingleDecodeWithKVCacheKernel(DTypeIn* __restrict__ q, DTypeIn* __restrict__ k,
                                              DTypeIn* __restrict__ v, DTypeOut* __restrict__ o,
                                              DTypeOut* __restrict__ tmp,
                                              tensor_info_t<kv_layout, bdy, bdx * vec_size> info,
                                              float sm_scale, float rope_rcp_scale,
                                              float rope_rcp_theta, uint32_t kv_chunk_size) {
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
  DTypeIn* v_smem = (DTypeIn*)(smem + num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
                                          sizeof(DTypeIn));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
                                       sizeof(DTypeIn));

  uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (rotary_mode == RotaryMode::kLlama) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(q + info.get_qo_elem_offset(0, qo_head_idx, 0),
                                                freq, seq_len - 1);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(q + info.get_qo_elem_offset(0, qo_head_idx, tx * vec_size));
  }
  // multiple q_vec by sm_scale
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    q_vec[i] *= sm_scale;
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
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + info.get_kv_elem_offset(
                  producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j, kv_head_idx,
                  tx * vec_size),
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v + info.get_kv_elem_offset(
                  producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j, kv_head_idx,
                  tx * vec_size),
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    producer_kv_idx_base += bdy * bdz * tile_size_per_bdx;
  }

  // pipelining k/v tiles loading and state updating
  uint32_t consumer_kv_idx_base = chunk_start, stage_idx = 0;
  state_t<vec_size> st_local;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(kv_chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<rotary_mode, vec_size, bdx, bdy * tile_size_per_bdx>(
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, stage_idx, q_vec,
        freq, consumer_kv_idx_base, iter * bdy * tile_size_per_bdx * bdz, kv_chunk_size, s,
        st_local);
    block.sync();
    // load k
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + info.get_kv_elem_offset(
                  producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j, kv_head_idx,
                  tx * vec_size),
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    // update m/d/o state
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx,
        st_local);
    block.sync();

    // load v
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v + info.get_kv_elem_offset(
                  producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j, kv_head_idx,
                  tx * vec_size),
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
    producer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
    consumer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync local state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(st_local, reinterpret_cast<float*>(smem), smem_md);
  st_local.normalize();

  if constexpr (partition_kv) {
    // update tmp buffer
    st_local.o.cast_store(tmp + (kv_chunk_idx * num_qo_heads + qo_head_idx) * head_dim +
                          tx * vec_size);
    float* tmp_lse = (float*)(tmp + num_kv_chunks * num_qo_heads * head_dim);
    tmp_lse[kv_chunk_idx * num_qo_heads + qo_head_idx] = st_local.get_lse();
  } else {
    st_local.o.cast_store(o + info.get_qo_elem_offset(0, qo_head_idx, tx * vec_size));
  }
}

template <QKVLayout kv_layout, RotaryMode rotary_mode, uint32_t num_stages_smem, uint32_t vec_size,
          uint32_t bdx, uint32_t bdy, uint32_t bdz, typename DTypeIn, typename DTypeOut>
__global__ void BatchDecodeWithPaddedKVCacheKernel(
    DTypeIn* __restrict__ q, DTypeIn* __restrict__ k, DTypeIn* __restrict__ v,
    DTypeOut* __restrict__ o, float* __restrict__ lse,
    tensor_info_t<kv_layout, bdy, bdx * vec_size> info, float sm_scale, float rope_rcp_scale,
    float rope_rcp_theta) {
  auto block = cg::this_thread_block();
  sm_scale *= math::log2e;

  constexpr uint32_t head_dim = bdx * vec_size;
  uint32_t kv_head_idx = blockIdx.y;
  uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
  uint32_t batch_idx = blockIdx.x;
  uint32_t num_qo_heads = info.get_num_qo_heads();
  uint32_t num_kv_heads = info.get_num_kv_heads();
  uint32_t seq_len = info.kv_len;

  extern __shared__ uint8_t smem[];
  DTypeIn* k_smem = (DTypeIn*)smem;
  DTypeIn* v_smem = (DTypeIn*)(smem + num_stages_smem * bdy * bdz * head_dim * sizeof(DTypeIn));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * bdy * bdz * head_dim * sizeof(DTypeIn));

  uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (rotary_mode == RotaryMode::kLlama) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(
        q + batch_idx * num_qo_heads * head_dim + info.get_qo_elem_offset(0, qo_head_idx, 0), freq,
        seq_len - 1);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(q + batch_idx * num_qo_heads * head_dim +
                    info.get_qo_elem_offset(0, qo_head_idx, tx * vec_size));
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    q_vec[i] *= sm_scale;
  }
  block.sync();

  // preload k tiles and v tiles
  uint32_t producer_kv_idx_base = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
        k_smem + ((iter * bdz + tz) * bdy + ty) * head_dim + tx * vec_size,
        k + batch_idx * seq_len * num_kv_heads * head_dim +
            info.get_kv_elem_offset(producer_kv_idx_base + tz * bdy + ty, kv_head_idx,
                                    tx * vec_size),
        producer_kv_idx_base + tz * bdy + ty < seq_len);
    cp_async::commit_group();
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        v_smem + ((iter * bdz + tz) * bdy + ty) * head_dim + tx * vec_size,
        v + batch_idx * seq_len * num_kv_heads * head_dim +
            info.get_kv_elem_offset(producer_kv_idx_base + tz * bdy + ty, kv_head_idx,
                                    tx * vec_size),
        producer_kv_idx_base + tz * bdy + ty < seq_len);
    cp_async::commit_group();
    producer_kv_idx_base += bdy * bdz;
  }

  // pipelining k/v tiles loading and state updating
  uint32_t consumer_kv_idx_base = 0, stage_idx = 0;
  state_t<vec_size> st_local;
  float s[bdy];

#pragma unroll 4
  for (uint32_t iter = 0; iter < ceil_div(seq_len, bdy * bdz); ++iter) {
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<rotary_mode, vec_size, bdx, bdy>(k_smem + (stage_idx * bdz + tz) * bdy * head_dim,
                                                stage_idx, q_vec, freq, consumer_kv_idx_base,
                                                iter * bdy * bdz, seq_len, s, st_local);
    block.sync();
    // load k
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
        k_smem + ((stage_idx * bdz + tz) * bdy + ty) * head_dim + tx * vec_size,
        k + batch_idx * seq_len * num_kv_heads * head_dim +
            info.get_kv_elem_offset(producer_kv_idx_base + tz * bdy + ty, kv_head_idx,
                                    tx * vec_size),
        producer_kv_idx_base + tz * bdy + ty < seq_len);
    cp_async::commit_group();

    // update m/d/o state
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy>(v_smem + (stage_idx * bdz + tz) * bdy * head_dim, s,
                                           stage_idx, st_local);
    block.sync();

    // load v
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        v_smem + ((stage_idx * bdz + tz) * bdy + ty) * head_dim + tx * vec_size,
        v + batch_idx * seq_len * num_kv_heads * head_dim +
            info.get_kv_elem_offset(producer_kv_idx_base + tz * bdy + ty, kv_head_idx,
                                    tx * vec_size),
        producer_kv_idx_base + tz * bdy + ty < seq_len);
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
    producer_kv_idx_base += bdy * bdz;
    consumer_kv_idx_base += bdy * bdz;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync local state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(st_local, reinterpret_cast<float*>(smem), smem_md);

  st_local.normalize();
  st_local.o.cast_store(o + batch_idx * num_qo_heads * head_dim +
                        info.get_qo_elem_offset(0, qo_head_idx, tx * vec_size));

  // write lse
  if (lse != nullptr) {
    lse[batch_idx * num_qo_heads + qo_head_idx] = st_local.get_lse();
  }
}

/*!
 * \brief FlashAttention decoding cuda kernel with paged kv-cache for multiple requests
 * \tparam partition_kv Whether to partition kv-cache on sequence length dimension or not
 * \tparam rotary_mode The rotary mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam bdz A template integer indicates the block size in z dimension
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param q [batch_size, num_qo_heads, head_dim] The query matrix
 * \param paged_kv The paged kv-cache data structure
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param lse The logsumexp values
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 */
template <bool partition_kv, RotaryMode rotary_mode, uint32_t num_stages_smem,
          uint32_t tile_size_per_bdx, uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
          typename IdType>
__global__ void BatchDecodeWithPagedKVCacheKernel(
    DTypeIn* __restrict__ q, IdType* __restrict__ q_rope_position,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
    kv_partition_info_t<IdType> kv_partition_info, DTypeOut* __restrict__ o,
    DTypeOut* __restrict__ tmp, float* __restrict__ lse, float sm_scale, float rope_rcp_scale,
    float rope_rcp_theta) {
  auto block = cg::this_thread_block();
  sm_scale *= math::log2e;

  constexpr uint32_t head_dim = bdx * vec_size;
  const uint32_t batch_idx = blockIdx.x;
  const uint32_t kv_head_idx = blockIdx.y;
  const uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
  const uint32_t num_qo_heads = gridDim.y * bdy;
  const uint32_t cur_chunk_start = partition_kv ? kv_partition_info.chunk_start_pos[batch_idx] : 0U;
  const uint32_t cur_page_indptr_begin = paged_kv.indptr[batch_idx],
                 cur_page_indptr_end = paged_kv.indptr[batch_idx + 1];
  const uint32_t cur_last_page_len = paged_kv.last_page_len[batch_idx];
  const uint32_t kv_chunk_len =
      cur_page_indptr_begin != cur_page_indptr_end
          ? (cur_page_indptr_end - cur_page_indptr_begin - 1) * paged_kv.page_size +
                cur_last_page_len
          : 0;
  const uint32_t seq_len =
      partition_kv ? kv_partition_info.seq_lens_before_partition[batch_idx] : kv_chunk_len;
  const uint32_t mapped_batch_idx =
      partition_kv ? kv_partition_info.batch_idx_map[batch_idx] : batch_idx;

  extern __shared__ uint8_t smem[];
  DTypeIn* k_smem = (DTypeIn*)smem;
  DTypeIn* v_smem = (DTypeIn*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                          sizeof(DTypeIn));
  DTypeIn** k_ptrs_smem = (DTypeIn**)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                                                 head_dim * sizeof(DTypeIn));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                       sizeof(DTypeIn));

  const uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (rotary_mode == RotaryMode::kLlama) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(
        q + (mapped_batch_idx * num_qo_heads + qo_head_idx) * head_dim, freq,
        q_rope_position == nullptr ? (seq_len - 1) : q_rope_position[mapped_batch_idx]);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(q + (mapped_batch_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    q_vec[i] *= sm_scale;
  }
  block.sync();

  // preload k/v tiles
  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  static_assert(num_stages_smem <= bdx);
#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    k_ptrs_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] = paged_kv.protective_get_k_ptr(
        cur_page_indptr_begin + (((j * bdz + tz) * bdy + ty) * bdx + tx) / paged_kv.page_size,
        kv_head_idx, (((j * bdz + tz) * bdy + ty) * bdx + tx) % paged_kv.page_size, 0, last_indptr);
  }
  block.sync();

  DTypeIn* k_ptrs[tile_size_per_bdx];
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      k_ptrs[j] =
          k_ptrs_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] + tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k_ptrs[j], ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < kv_chunk_len);
    }
    cp_async::commit_group();
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      DTypeIn* v_ptr = k_ptrs[j] + paged_kv.kv_offset_delta();
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v_ptr, ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < kv_chunk_len);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

  state_t<vec_size> st;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(kv_chunk_len, tile_size_per_bdx * bdy * bdz); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        k_ptrs_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] = paged_kv.protective_get_k_ptr(
            cur_page_indptr_begin + ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
                                     ((j * bdz + tz) * bdy + ty) * bdx + tx) /
                                        paged_kv.page_size,
            kv_head_idx,
            ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
             ((j * bdz + tz) * bdy + ty) * bdx + tx) %
                paged_kv.page_size,
            0, last_indptr);
      }
    }
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<rotary_mode, vec_size, bdx, bdy * tile_size_per_bdx>(
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, stage_idx, q_vec,
        freq,
        (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[mapped_batch_idx]) +
            cur_chunk_start + iter * tile_size_per_bdx * bdy * bdz,
        iter * tile_size_per_bdx * bdy * bdz, kv_chunk_len, s, st);
    block.sync();

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      k_ptrs[j] = k_ptrs_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy + ty) *
                                  tile_size_per_bdx +
                              j] +
                  tx * vec_size;
    }
    // load k tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k_ptrs[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j <
              kv_chunk_len);
    }
    cp_async::commit_group();

    // update m/d/o states
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx, st);
    block.sync();

    // load v tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      DTypeIn* v_ptr = k_ptrs[j] + paged_kv.kv_offset_delta();
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v_ptr,
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j <
              kv_chunk_len);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync local state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(st, reinterpret_cast<float*>(smem), smem_md);
  st.normalize();

  if constexpr (partition_kv) {
    st.o.cast_store(tmp + (batch_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
    float* tmp_lse = (float*)(tmp + paged_kv.batch_size * num_qo_heads * head_dim);
    tmp_lse[batch_idx * num_qo_heads + qo_head_idx] = st.get_lse();
  } else {
    st.o.cast_store(o + (batch_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
    // write lse
    if (lse != nullptr) {
      lse[batch_idx * num_qo_heads + qo_head_idx] = st.get_lse();
    }
  }
}

/*!
 * \brief Get the heuristic number of threads per threadblock
 * \param group_size The number of qo heads that maps to the same kv head in GQA.
 * \param sizeof_dtype The size (in terms of bytes) of the input data type
 */
constexpr uint32_t get_heuristic_num_threads(uint32_t group_size, uint32_t sizeof_dtype) {
  if (group_size == 8U) {
    if (sizeof_dtype == 1U) {
      return 256U;  // not enough registers for 512 threads
    } else {
      return 512U;
    }
  } else {
#ifdef FLASHINFER_ENABLE_BF16
    return 128U;
#else
    return 64U;
#endif
  }
}

/*!
 * \brief Esitmate the temporary buffer size and the maximum grid size for the
 *   partiton-kv SingleDecodeWithKVCache kernel
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \param tmp_size The estimated temporary buffer size, return 0 if not use partition-kv kernel
 * \param max_grid_size The maximum grid size that can be used in a partition-kv kernel
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param num_kv_heads A integer indicates the number of heads of key and value
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param kv_layout The layout of k/v matrices
 * \param rotary_mode The rotary mode
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t SingleDecodeWithKVCacheWorkEstimation(uint32_t& tmp_size, uint32_t& max_grid_size,
                                                  uint32_t num_qo_heads, uint32_t num_kv_heads,
                                                  uint32_t seq_len, uint32_t head_dim,
                                                  QKVLayout kv_layout = QKVLayout::kNHD,
                                                  RotaryMode rotary_mode = RotaryMode::kNone,
                                                  cudaStream_t stream = nullptr) {
  const uint32_t GROUP_SIZE = num_qo_heads / num_kv_heads;
  if (seq_len <= 256U) {
    tmp_size = 0;
  } else {
    DISPATCH_GQA_GROUP_SIZE(
        num_qo_heads / num_kv_heads, GROUP_SIZE,
        {DISPATCH_HEAD_DIM(
            head_dim, HEAD_DIM,
            {DISPATCH_ROTARY_MODE(
                rotary_mode, ROTARY_MODE, {DISPATCH_LAYOUT(kv_layout, KV_LAYOUT, {
                  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
                  constexpr uint32_t num_stages_smem = 2U;
                  constexpr uint32_t bdx = HEAD_DIM / vec_size;
                  static_assert(bdx <= 32U);
                  constexpr uint32_t bdy = GROUP_SIZE;
                  constexpr uint32_t num_threads =
                      std::max(get_heuristic_num_threads(GROUP_SIZE, sizeof(DTypeIn)), bdx * bdy);
                  constexpr uint32_t bdz = num_threads / (bdx * bdy);
                  constexpr uint32_t tile_size_per_bdx =
                      GROUP_SIZE == 1 ? (sizeof(DTypeIn) == 1 ? 2U : 8U) : 1U;
                  const uint32_t smem_size = 2U * num_stages_smem * bdy * tile_size_per_bdx * bdz *
                                                 head_dim * sizeof(DTypeIn) +
                                             2U * bdy * bdz * sizeof(float);

                  auto kernel =
                      SingleDecodeWithKVCacheKernel<KV_LAYOUT, /*partition_kv=*/true, ROTARY_MODE,
                                                    num_stages_smem, tile_size_per_bdx, vec_size,
                                                    bdx, bdy, bdz, DTypeIn, DTypeOut>;
                  int num_blocks_per_sm = 0;
                  int num_sm = 0;
                  int dev_id = 0;
                  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
                  FLASHINFER_CUDA_CALL(
                      cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
                  FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                      &num_blocks_per_sm, kernel, num_threads, smem_size));
                  max_grid_size = uint32_t(num_blocks_per_sm) * uint32_t(num_sm);
                  uint32_t max_num_kv_chunks = max_grid_size / num_kv_heads;
                  uint32_t kv_chunk_size = max(ceil_div(seq_len, max_num_kv_chunks), 256);
                  uint32_t num_kv_chunks = ceil_div(seq_len, kv_chunk_size);
                  tmp_size = num_qo_heads * num_kv_chunks *
                             (head_dim * sizeof(DTypeOut) + 2 * sizeof(float));
                })})})});
  }
  return cudaSuccess;
}

/*!
 * \brief FlashAttention decoding with kv-cache for a single request
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \param q The query matrix, shape: [num_qo_heads, head_dim]
 * \param k The key matrix in kv-cache, shape: [seq_len, num_kv_heads, head_dim]
 *   for NHD layout, [num_kv_heads, seq_len, head_dim] for HND layout
 * \param v The value matrix in kv-cache, shape: [seq_len, num_kv_heads,
 *   head_dim] for NHD layout, [num_kv_heads, seq_len, head_dim] for HND layout
 * \param o The output matrix, shape: [num_qo_heads, head_dim]
 * \param tmp Used-allocated temporary buffer
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param num_kv_heads A integer indicates the number of heads of key and value
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param kv_layout The layout of q/k/v matrices
 * \param rotary_mode The rotary mode
 * \param rope_scale The scaling factor used in RoPE Interpolation
 * \param rope_theta The theta used in RoPE
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t SingleDecodeWithKVCache(DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeOut* o, DTypeOut* tmp,
                                    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t seq_len,
                                    uint32_t head_dim, QKVLayout kv_layout = QKVLayout::kNHD,
                                    RotaryMode rotary_mode = RotaryMode::kNone,
                                    float rope_scale = 1.f, float rope_theta = 1e4,
                                    cudaStream_t stream = nullptr) {
  const float sm_scale = 1.f / std::sqrt(float(head_dim));
  const float rope_rcp_scale = 1.f / rope_scale;
  const float rope_rcp_theta = 1.f / rope_theta;
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  DISPATCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {DISPATCH_HEAD_DIM(
          head_dim, HEAD_DIM,
          {DISPATCH_ROTARY_MODE(
              rotary_mode, ROTARY_MODE, {DISPATCH_LAYOUT(kv_layout, KV_LAYOUT, {
                constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
                constexpr uint32_t num_stages_smem = 2U;
                constexpr uint32_t bdx = HEAD_DIM / vec_size;
                static_assert(bdx <= 32U);
                constexpr uint32_t bdy = GROUP_SIZE;
                constexpr uint32_t num_threads =
                    std::max(get_heuristic_num_threads(GROUP_SIZE, sizeof(DTypeIn)), bdx * bdy);
                constexpr uint32_t bdz = num_threads / (bdx * bdy);
                tensor_info_t<KV_LAYOUT, GROUP_SIZE, HEAD_DIM> info(1, seq_len, num_kv_heads);
                constexpr uint32_t tile_size_per_bdx =
                    GROUP_SIZE == 1 ? (sizeof(DTypeIn) == 1 ? 2U : 8U) : 1U;
                const uint32_t smem_size = 2U * num_stages_smem * bdy * tile_size_per_bdx * bdz *
                                               head_dim * sizeof(DTypeIn) +
                                           2U * bdy * bdz * sizeof(float);
                if (seq_len <= 256 || tmp == nullptr) {
                  // no need to use partition-kv kernel
                  auto kernel =
                      SingleDecodeWithKVCacheKernel<KV_LAYOUT, /*partition_kv=*/false, ROTARY_MODE,
                                                    num_stages_smem, tile_size_per_bdx, vec_size,
                                                    bdx, bdy, bdz, DTypeIn, DTypeOut>;
                  FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

                  dim3 nblks = dim3(1, num_kv_heads);
                  dim3 nthrs = dim3(bdx, bdy, bdz);
                  void* args[] = {(void*)&q,
                                  (void*)&k,
                                  (void*)&v,
                                  (void*)&o,
                                  (void*)&tmp,
                                  (void*)&info,
                                  (void*)&sm_scale,
                                  (void*)&rope_rcp_scale,
                                  (void*)&rope_rcp_theta,
                                  (void*)&seq_len};
                  FLASHINFER_CUDA_CALL(
                      cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
                } else {
                  // use partition-kv kernel
                  auto kernel =
                      SingleDecodeWithKVCacheKernel<KV_LAYOUT, /*partition_kv=*/true, ROTARY_MODE,
                                                    num_stages_smem, tile_size_per_bdx, vec_size,
                                                    bdx, bdy, bdz, DTypeIn, DTypeOut>;
                  FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

                  int num_blocks_per_sm = 0;
                  int num_sm = 0;
                  int dev_id = 0;
                  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
                  FLASHINFER_CUDA_CALL(
                      cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
                  FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                      &num_blocks_per_sm, kernel, num_threads, smem_size));
                  uint32_t max_grid_size = uint32_t(num_blocks_per_sm) * uint32_t(num_sm);
                  uint32_t max_num_kv_chunks = max_grid_size / num_kv_heads;
                  uint32_t kv_chunk_size = max(ceil_div(seq_len, max_num_kv_chunks), 256);
                  uint32_t num_chunks = ceil_div(seq_len, kv_chunk_size);
                  dim3 nblks = dim3(num_chunks, num_kv_heads);
                  if (nblks.x == 0 || nblks.y == 0) {
                    std::ostringstream err_msg;
                    err_msg << "Invalid kernel configuration: nblks=(" << nblks.x << "," << nblks.y
                            << ")";
                    throw std::runtime_error(err_msg.str());
                  }
                  dim3 nthrs = dim3(bdx, bdy, bdz);
                  void* args[] = {(void*)&q,
                                  (void*)&k,
                                  (void*)&v,
                                  (void*)&o,
                                  (void*)&tmp,
                                  (void*)&info,
                                  (void*)&sm_scale,
                                  (void*)&rope_rcp_scale,
                                  (void*)&rope_rcp_theta,
                                  (void*)&kv_chunk_size};
                  FLASHINFER_CUDA_CALL(
                      cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
                  FLASHINFER_CUDA_CALL(
                      MergeStates(tmp, (float*)(tmp + num_chunks * num_qo_heads * HEAD_DIM), o,
                                  nullptr, num_chunks, 1, num_qo_heads, HEAD_DIM, stream));
                }
              })})})});
  return cudaSuccess;
}

/*!
 * \brief Partition Paged KV-Cache into multiple chunks on KV sequence length
 * \tparam IdType A template type indicates the index data type
 * \param old_batch_size The batch size of the old Paged KV-Cache
 * \param old_page_indptr_h The host-side page indptr of the old Paged KV-Cache
 * \param old_last_page_len_h The host-side last page offset of the old Paged KV-Cache
 * \param max_num_pages_per_batch The maximum number of pages per batch
 * \param new_paged_kv_d The device-side new Paged KV-Cache
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <typename IdType>
cudaError_t PartitionPagedKVCacheComputeAuxiliaryInfo(
    const uint32_t max_num_pages_per_batch, const uint32_t old_batch_size, const uint32_t page_size,
    IdType* old_indptr, IdType* old_last_page_len, IdType* new_indptr_d,
    IdType* new_last_page_len_d, IdType* chunk_indptr_d, IdType* batch_idx_map_d,
    IdType* chunk_start_d, IdType* seq_lens_before_partition_d, cudaStream_t stream = nullptr) {
  std::vector<IdType> new_page_indptr_h{0}, new_last_page_len_h, chunk_indptr_h{0}, batch_idx_map_h,
      chunk_start_pos_h, seq_lens_before_partition_h;

  std::vector<IdType> old_indptr_h(old_batch_size + 1), old_last_page_len_h(old_batch_size);
  if (is_device_ptr(old_indptr)) {
    FLASHINFER_CUDA_CALL(cudaMemcpyAsync(old_indptr_h.data(), old_indptr,
                                         sizeof(IdType) * (old_batch_size + 1),
                                         cudaMemcpyDeviceToHost, stream));
    FLASHINFER_CUDA_CALL(cudaMemcpyAsync(old_last_page_len_h.data(), old_last_page_len,
                                         sizeof(IdType) * old_batch_size, cudaMemcpyDeviceToHost,
                                         stream));
    FLASHINFER_CUDA_CALL(cudaStreamSynchronize(stream));
  } else {
    old_indptr_h.assign(old_indptr, old_indptr + old_batch_size + 1);
    old_last_page_len_h.assign(old_last_page_len, old_last_page_len + old_batch_size);
  }

  for (uint32_t batch_idx = 0; batch_idx < old_batch_size; batch_idx++) {
    uint32_t num_chunks =
        ceil_div(old_indptr_h[batch_idx + 1] - old_indptr_h[batch_idx], max_num_pages_per_batch);
    chunk_indptr_h.push_back(chunk_indptr_h.back() + num_chunks);
    if (num_chunks == 0) {
      new_page_indptr_h.push_back(old_indptr_h[batch_idx]);
      new_last_page_len_h.push_back(0);
      batch_idx_map_h.push_back(batch_idx);
      chunk_start_pos_h.push_back(0);
      seq_lens_before_partition_h.push_back(0);
    } else {
      uint32_t seq_len_before_partition =
          (old_indptr_h[batch_idx + 1] - old_indptr_h[batch_idx] - 1) * page_size +
          old_last_page_len_h[batch_idx];
      for (uint32_t j = 0; j < num_chunks; ++j) {
        bool is_last = (j + 1) == num_chunks;
        new_page_indptr_h.push_back(min(old_indptr_h[batch_idx] + (j + 1) * max_num_pages_per_batch,
                                        old_indptr_h[batch_idx + 1]));
        new_last_page_len_h.push_back(is_last ? old_last_page_len_h[batch_idx] : page_size);
        batch_idx_map_h.push_back(batch_idx);
        chunk_start_pos_h.push_back(j * max_num_pages_per_batch * page_size);
        seq_lens_before_partition_h.push_back(seq_len_before_partition);
      }
    }
  }

  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(new_indptr_d, new_page_indptr_h.data(),
                                       sizeof(IdType) * new_page_indptr_h.size(),
                                       cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(new_last_page_len_d, new_last_page_len_h.data(),
                                       sizeof(IdType) * new_last_page_len_h.size(),
                                       cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(chunk_indptr_d, chunk_indptr_h.data(),
                                       sizeof(IdType) * chunk_indptr_h.size(),
                                       cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(batch_idx_map_d, batch_idx_map_h.data(),
                                       sizeof(IdType) * batch_idx_map_h.size(),
                                       cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(chunk_start_d, chunk_start_pos_h.data(),
                                       sizeof(IdType) * chunk_start_pos_h.size(),
                                       cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(
      seq_lens_before_partition_d, seq_lens_before_partition_h.data(),
      sizeof(IdType) * seq_lens_before_partition_h.size(), cudaMemcpyHostToDevice, stream));
  return cudaSuccess;
}

/*!
 * \brief Compute the maximum number of pages per batch and the new batch size
 *   after we partition Paged KV-Cache into multiple chunks on KV sequence length
 *   dimension.
 * \tparam IdType A template type indicates the index data type
 * \param max_grid_size The maximum grid size of the kernel
 * \param num_kv_heads The number of KV heads
 * \param num_pages The number of pages per request in the batch
 * \param max_num_pages_per_batch_lb The pre-set lower bound of maximum number of
 *   pages per batch, default to 1
 * \return (max_num_pages_per_batch, new_batch_size) The number of pages per batch and
 *   the new batch size after the partition.
 */
template <typename IdType>
std::pair<uint32_t, uint32_t> PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(
    const uint32_t max_grid_size, const uint32_t num_kv_heads, const std::vector<IdType>& num_pages,
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
      new_batch_size += ceil_div(elem, mid);
    }
    if (new_batch_size * num_kv_heads > max_grid_size) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  new_batch_size = 0;
  for (const IdType& elem : num_pages) {
    new_batch_size += ceil_div(std::max(elem, 1), low);
  }
  return {low, new_batch_size};
}

/*!
 * \brief Estimate the temporary buffer size and the maximum grid size for the
 *   partition-kv BatchDecodeWithPagedKVCache kernel
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param tmp_size The estimated temporary buffer size, return 0 if not use partition-kv kernel
 * \param max_grid_size The maximum grid size that can be used in a partiton-kv kernel
 * \param max_num_pages_per_batch The maximum number of pages per batch
 * \param new_batch_size The new batch size after the partition
 * \param paged_kv The paged kv cache data structure
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param rotary_mode The rotary mode
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheWorkEstimation(
    uint32_t& tmp_size, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t batch_size, IdType* kv_indptr, const uint32_t num_qo_heads,
    const uint32_t num_kv_heads, const uint32_t head_dim, const uint32_t page_size,
    const RotaryMode rotary_mode = RotaryMode::kNone, cudaStream_t stream = nullptr) {
  DISPATCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {DISPATCH_HEAD_DIM(
          head_dim, HEAD_DIM, {DISPATCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
            constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
            constexpr uint32_t num_stages_smem = 2U;
            constexpr uint32_t bdx = HEAD_DIM / vec_size;
            static_assert(bdx <= 32);
            constexpr uint32_t bdy = GROUP_SIZE;
            constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
            constexpr uint32_t bdz = num_threads / (bdx * bdy);
            constexpr uint32_t tile_size_per_bdx =
                GROUP_SIZE == 1 ? (sizeof(DTypeIn) == 1 ? 2U : 4U) : 1U;
            const uint32_t smem_size =
                2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim * sizeof(DTypeIn) +
                std::max(tile_size_per_bdx * num_threads * sizeof(DTypeIn*),
                         2 * bdy * bdz * sizeof(float));

            auto partition_kv_kernel = BatchDecodeWithPagedKVCacheKernel<
                /*partition_kv=*/true, ROTARY_MODE, num_stages_smem, tile_size_per_bdx, vec_size,
                bdx, bdy, bdz, page_storage, kv_layout, DTypeIn, DTypeOut, IdType>;
            int num_blocks_per_sm = 0;
            int num_sm = 0;
            int dev_id = 0;
            FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
            FLASHINFER_CUDA_CALL(
                cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
            FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &num_blocks_per_sm, partition_kv_kernel, num_threads, smem_size));
            max_grid_size = num_blocks_per_sm * num_sm;
            if (batch_size * num_kv_heads >= max_grid_size) {
              // do not use partition-kv kernel
              tmp_size = 0;
              new_batch_size = batch_size;
            } else {
              // compute max_num_pages_per_batch and new_batch_size
              std::vector<IdType> page_indptr_h(batch_size + 1), num_pages(batch_size);
              if (is_device_ptr(kv_indptr)) {
                FLASHINFER_CUDA_CALL(cudaMemcpy(page_indptr_h.data(), kv_indptr,
                                                sizeof(IdType) * (batch_size + 1),
                                                cudaMemcpyDeviceToHost));
              } else {
                page_indptr_h.assign(kv_indptr, kv_indptr + batch_size + 1);
              }
              for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                num_pages[batch_idx] = page_indptr_h[batch_idx + 1] - page_indptr_h[batch_idx];
              }
              std::tie(max_num_pages_per_batch, new_batch_size) =
                  PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(max_grid_size, num_kv_heads,
                                                                      num_pages, 128 / page_size);
              if (new_batch_size == batch_size) {
                // do not use partition-kv kernel for short sequence
                tmp_size = 0;
              } else {
                tmp_size = num_qo_heads * new_batch_size *
                           (head_dim * sizeof(DTypeOut) + 2 * sizeof(float));
              }
            }
          })})});
  return cudaSuccess;
}

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PageStorage page_storage, QKVLayout kv_layout,
          RotaryMode ROTARY_MODE, typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(
    DTypeIn* q, IdType* q_rope_position,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
    kv_partition_info_t<IdType> kv_partition_info, DTypeOut* o, DTypeOut* tmp, float* lse,
    float rope_scale, float rope_theta, cudaStream_t stream) {
  const float sm_scale = 1.f / std::sqrt(float(HEAD_DIM));
  const float rope_rcp_scale = 1.f / rope_scale;
  const float rope_rcp_theta = 1.f / rope_theta;
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t batch_size = paged_kv.batch_size;
  const uint32_t num_qo_heads = num_kv_heads * GROUP_SIZE;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
  constexpr uint32_t num_stages_smem = 2U;
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  constexpr uint32_t bdy = GROUP_SIZE;
  constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
  constexpr uint32_t bdz = num_threads / (bdx * bdy);
  constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeIn) == 1 ? 2U : 4U) : 1U;
  const uint32_t smem_size =
      2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeIn) +
      std::max(tile_size_per_bdx * num_threads * sizeof(DTypeIn*), 2 * bdy * bdz * sizeof(float));

  if (tmp == nullptr) {
    // do not use partition-kv kernel
    dim3 nblks(batch_size, num_kv_heads);
    dim3 nthrs(bdx, bdy, bdz);
    auto kernel =
        BatchDecodeWithPagedKVCacheKernel</*partition_kv=*/false, ROTARY_MODE, num_stages_smem,
                                          tile_size_per_bdx, vec_size, bdx, bdy, bdz, page_storage,
                                          kv_layout, DTypeIn, DTypeOut, IdType>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    void* args[] = {(void*)&q,
                    (void*)&q_rope_position,
                    (void*)&paged_kv,
                    (void*)&kv_partition_info,
                    (void*)&o,
                    (void*)&tmp,
                    (void*)&lse,
                    (void*)&sm_scale,
                    (void*)&rope_rcp_scale,
                    (void*)&rope_rcp_theta};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  } else {
    // use partition-kv kernel
    auto partition_kv_kernel =
        BatchDecodeWithPagedKVCacheKernel</*partition_kv=*/true, ROTARY_MODE, num_stages_smem,
                                          tile_size_per_bdx, vec_size, bdx, bdy, bdz, page_storage,
                                          kv_layout, DTypeIn, DTypeOut, IdType>;
    FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
        partition_kv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    void* args[] = {(void*)&q,
                    (void*)&q_rope_position,
                    (void*)&paged_kv,
                    (void*)&kv_partition_info,
                    (void*)&o,
                    (void*)&tmp,
                    (void*)&lse,
                    (void*)&sm_scale,
                    (void*)&rope_rcp_scale,
                    (void*)&rope_rcp_theta};
    dim3 nblks(batch_size, num_kv_heads);
    dim3 nthrs(bdx, bdy, bdz);
    FLASHINFER_CUDA_CALL(
        cudaLaunchKernel((void*)partition_kv_kernel, nblks, nthrs, args, smem_size, stream));
    FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
        tmp, (float*)(tmp + batch_size * num_qo_heads * HEAD_DIM), kv_partition_info.chunk_indptr,
        o, lse, kv_partition_info.batch_size_before_partition, num_qo_heads, HEAD_DIM, stream));
  }

  return cudaSuccess;
}

/*!
 * \brief FlashAttention decoding cuda kernel with paged kv-cache for batched requests
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type used in paged kv-cache
 * \param q [batch_size, num_qo_heads, head_dim] The query matrix
 * \param paged_kv The paged kv cache data structure
 * \param o [batch_size, num_qo_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param lse The logsumexp values.
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param rotary_mode The rotary mode
 * \param rope_scale The scaling ratio used in RoPE Interpolation.
 * \param rope_theta A floating point number indicate the "theta" used in RoPE
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchDecodeWithPagedKVCache(
    DTypeIn* q, IdType* q_rope_position,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
    kv_partition_info_t<IdType> kv_partition_info, DTypeOut* o, DTypeOut* tmp, float* lse,
    uint32_t num_qo_heads, RotaryMode rotary_mode = RotaryMode::kNone, float rope_scale = 1.f,
    float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t batch_size = paged_kv.batch_size;
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  DISPATCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {DISPATCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
                           return BatchDecodeWithPagedKVCacheDispatched<
                               GROUP_SIZE, HEAD_DIM, page_storage, kv_layout, ROTARY_MODE, DTypeIn,
                               DTypeOut, IdType>(q, q_rope_position, paged_kv, kv_partition_info, o,
                                                 tmp, lse, rope_scale, rope_theta, stream);
                         })})});

  return cudaSuccess;
}

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, QKVLayout KV_LAYOUT, RotaryMode ROTARY_MODE,
          typename DTypeIn, typename DTypeOut>
cudaError_t BatchDecodeWithPaddedKVCacheDispatched(DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeIn* o,
                                                   DTypeOut* tmp, float* lse, uint32_t batch_size,
                                                   uint32_t padded_kv_len, uint32_t num_qo_heads,
                                                   float rope_scale, float rope_theta,
                                                   cudaStream_t stream) {
  const float sm_scale = 1.f / std::sqrt(float(HEAD_DIM));
  const float rope_rcp_scale = 1.f / rope_scale;
  const float rope_rcp_theta = 1.f / rope_theta;
  const uint32_t num_kv_heads = num_qo_heads / GROUP_SIZE;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
  constexpr uint32_t num_stages_smem = 2U;
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  constexpr uint32_t bdy = GROUP_SIZE;
  constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
  constexpr uint32_t bdz = num_threads / (bdx * bdy);

  const uint32_t smem_size =
      2 * num_stages_smem * bdy * bdz * HEAD_DIM * sizeof(DTypeIn) + 2 * bdy * bdz * sizeof(float);

  dim3 nblks(batch_size, num_kv_heads);
  dim3 nthrs(bdx, bdy, bdz);
  auto kernel = BatchDecodeWithPaddedKVCacheKernel<KV_LAYOUT, ROTARY_MODE, num_stages_smem,
                                                   vec_size, bdx, bdy, bdz, DTypeIn, DTypeOut>;
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  tensor_info_t<KV_LAYOUT, GROUP_SIZE, HEAD_DIM> info(1, padded_kv_len, num_kv_heads);
  void* args[] = {(void*)&q,
                  (void*)&k,
                  (void*)&v,
                  (void*)&o,
                  (void*)&lse,
                  (void*)&info,
                  (void*)&sm_scale,
                  (void*)&rope_rcp_scale,
                  (void*)&rope_rcp_theta};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  return cudaSuccess;
}

template <typename DTypeIn, typename DTypeOut>
cudaError_t BatchDecodeWithPaddedKVCache(
    DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeIn* o, DTypeOut* tmp, float* lse, uint32_t batch_size,
    uint32_t padded_kv_len, uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
    QKVLayout kv_layout = QKVLayout::kNHD, RotaryMode rotary_mode = RotaryMode::kNone,
    float rope_scale = 1.f, float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  DISPATCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {DISPATCH_HEAD_DIM(
          head_dim, HEAD_DIM,
          {DISPATCH_ROTARY_MODE(
              rotary_mode, ROTARY_MODE, {DISPATCH_LAYOUT(kv_layout, KV_LAYOUT, {
                return BatchDecodeWithPaddedKVCacheDispatched<GROUP_SIZE, HEAD_DIM, KV_LAYOUT,
                                                              ROTARY_MODE, DTypeIn, DTypeOut>(
                    q, k, v, o, tmp, lse, batch_size, padded_kv_len, num_qo_heads, rope_scale,
                    rope_theta, stream);
              })})})});
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_CUH_
