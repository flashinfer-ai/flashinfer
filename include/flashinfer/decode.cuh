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

#include "layout.cuh"
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
template <size_t vec_size, size_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> apply_llama_rope(
    const T *x, const vec_t<float, vec_size> &freq, size_t offset) {
  constexpr size_t head_dim = vec_size * bdx;
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);
  permuted_vec.cast_load(x + ((threadIdx.x * vec_size < head_dim / 2)
                                  ? threadIdx.x * vec_size + head_dim / 2
                                  : threadIdx.x * vec_size - head_dim / 2));

#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    float embed = float(offset) * freq[i];
    float cos, sin;
    __sincosf(embed, &sin, &cos);
    vec[i] = vec[i] * cos +
             ((threadIdx.x * vec_size < head_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin;
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
 * \param kv_shared_offset An array of size_t indicates the k/v tiles offset in shared
 *   memory of different pipeline stages
 * \param kv_idx A integer indicates the thread-local kv position in kv-cache.
 * \param compute_stage_idx A integer indicates the compute stage index in
 *   the pipeline
 * \param num_heads A integer indicates the number of heads
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param x A float indicates the thread-local result of qk
 */
template <RotaryMode rotary_mode, size_t vec_size, size_t bdx, typename T>
__device__ __forceinline__ void compute_qk(const T *smem, const vec_t<float, vec_size> &q_vec,
                                           const vec_t<float, vec_size> &freq,
                                           const size_t *kv_shared_offset, size_t kv_idx,
                                           size_t compute_stage_idx, size_t num_heads,
                                           float sm_scale, float &x) {
  constexpr size_t head_dim = bdx * vec_size;
  vec_t<float, vec_size> k_vec;
  size_t tx = threadIdx.x, ty = threadIdx.y;
  if constexpr (rotary_mode == RotaryMode::kLlama) {
    // apply rotary embedding for all rows in k matrix of kv-cache
    k_vec = apply_llama_rope<vec_size, bdx>(
        smem + kv_shared_offset[compute_stage_idx] + ty * head_dim, freq, kv_idx);
  } else {
    // do not apply rotary embedding
    k_vec.cast_load(smem + kv_shared_offset[compute_stage_idx] + ty * head_dim + tx * vec_size);
  }
  x = 0.f;
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    x += q_vec[i] * k_vec[i] * sm_scale;
  }
  cg::thread_block_tile g = cg::tiled_partition<bdx>(cg::this_thread_block());
#pragma unroll
  for (size_t offset = bdx / 2; offset > 0; offset /= 2) {
    x += g.shfl_down(x, offset);
  }
  x = g.shfl(x, 0);
}

/*!
 * \brief Load v tile from shared memory and update partial state.
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param x A float indicates the pre-softmax logits
 * \param kv_shared_offset An array of size_t indicates the k/v tiles offset in shared
 *   memory of different pipeline stages.
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param pred_guard A boolean indicates whether the current thread is in the valid range
 * \param s The flashattention state to be updated
 */
template <size_t vec_size, size_t bdx, typename T>
__device__ __forceinline__ void update_partial_state(const T *smem, const float x,
                                                     const size_t *kv_shared_offset,
                                                     size_t compute_stage_idx, bool pred_guard,
                                                     state_t<vec_size> &s) {
  constexpr size_t head_dim = bdx * vec_size;
  vec_t<float, vec_size> v_vec;
  size_t tx = threadIdx.x, ty = threadIdx.y;
  v_vec.cast_load(smem + kv_shared_offset[compute_stage_idx] + ty * head_dim + tx * vec_size);
  if (pred_guard) {
    s.merge(v_vec, x);
  }
}

/*!
 * \brief Synchronize the state of all warps inside a threadblock.
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \param s The warp local state
 * \param smem The pointer to shared memory buffer for o
 * \param smem_md The pointer to shared memory buffer for m/d
 */
template <size_t vec_size, size_t bdx, size_t bdy>
__device__ __forceinline__ void sync_state(state_t<vec_size> &s, float *smem, float *smem_md) {
  constexpr size_t head_dim = bdx * vec_size;
  auto block = cg::this_thread_block();
  size_t tx = threadIdx.x, ty = threadIdx.y;
  s.o.store(smem + ty * head_dim + tx * vec_size);
  smem_md[ty * 2] = s.m;
  smem_md[ty * 2 + 1] = s.d;
  block.sync();
  s.init();
#pragma unroll
  for (size_t j = 0; j < bdy; ++j) {
    float mj = smem_md[j * 2], dj = smem_md[j * 2 + 1];
    vec_t<float, vec_size> oj;
    oj.load(smem + j * head_dim + tx * vec_size);
    s.merge(oj, mj, dj);
  }
}

}  // namespace

/*!
 * \brief FlashAttention decoding cuda kernel with kv-cache for a single
 * sequence, fused with RoPE.
 * \tparam layout The layout of k/v matrices (NHD or HND)
 * \tparam cooperative Whether to use cooperative kernel or not
 * \tparam rotary_mode The rotary mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \param q [num_heads, head_dim] The query matrix
 * \param k [seq_len, num_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_heads, head_dim] The value matrix in kv-cache
 * \param o [num_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param kv_info The tensor info of k/v matrices
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param head_dim A integer indicates the head dimension
 * \param rope_inv_scale A floating number indicate the multiplicative inverse
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_inv_theta A floating number indicate the multiplicative inverse
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 * \param kv_chunk_size A integer indicates the kv-chunk size
 */
template <QKVLayout layout, bool cooperative, RotaryMode rotary_mode, size_t vec_size, size_t bdx,
          size_t bdy, typename DTypeIn, typename DTypeOut>
__global__ void SingleDecodeWithKVCacheKernel(DTypeIn *__restrict__ q, DTypeIn *__restrict__ k,
                                              DTypeIn *__restrict__ v, DTypeOut *__restrict__ o,
                                              float *__restrict__ tmp,
                                              tensor_info_t<layout> kv_info, float sm_scale,
                                              float rope_inv_scale, float rope_inv_theta,
                                              size_t kv_chunk_size) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();

  constexpr size_t stages_count = 4;
  constexpr size_t head_dim = bdx * vec_size;
  size_t head_idx = blockIdx.y;
  size_t kv_chunk_idx = blockIdx.x;
  size_t num_kv_chunks = gridDim.x;
  size_t num_heads = gridDim.y;
  size_t seq_len = kv_info.kv_len;

  static_assert(bdx * bdy == 128);
  static_assert(stages_count >= sizeof(float) / sizeof(DTypeIn));
  __shared__ DTypeIn smem[stages_count * bdy * head_dim];
  __shared__ float smem_md[2 * bdy];

  size_t tx = threadIdx.x, ty = threadIdx.y;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (rotary_mode == RotaryMode::kLlama) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      freq[i] =
          rope_inv_scale *
          powf(rope_inv_theta, float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
    // apply rotary embedding to q matrix
    q_vec = apply_llama_rope<vec_size, bdx>(q + head_idx * head_dim, freq, seq_len - 1);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(q + head_idx * head_dim + tx * vec_size);
  }
  block.sync();

  size_t chunk_start = kv_chunk_idx * kv_chunk_size;
  kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);
  size_t chunk_end = chunk_start + kv_chunk_size;

  // load k tiles and v tiles
  size_t kv_shared_offset[stages_count] = {0U, 1U * bdy * head_dim, 2U * bdy * head_dim,
                                           3U * bdy * head_dim};

  // pipelining k/v tiles loading and state updating
  auto pipeline = cuda::make_pipeline();
  const auto frag_shape = cuda::aligned_size_t<alignof(float4)>(sizeof(DTypeIn) * vec_size);
  size_t producer_kv_idx = chunk_start + ty, consumer_kv_idx;
  bool producer_pred_guard = producer_kv_idx < chunk_end, consumer_pred_guard = true;
  pipeline.producer_acquire();
  if (producer_pred_guard) {
    cuda::memcpy_async(smem + kv_shared_offset[0] + ty * head_dim + tx * vec_size,
                       k + kv_info.get_kv_elem_offset(producer_kv_idx, head_idx, tx * vec_size),
                       frag_shape, pipeline);
  }
  pipeline.producer_commit();
  pipeline.producer_acquire();
  if (producer_pred_guard) {
    cuda::memcpy_async(smem + kv_shared_offset[1] + ty * head_dim + tx * vec_size,
                       v + kv_info.get_kv_elem_offset(producer_kv_idx, head_idx, tx * vec_size),
                       frag_shape, pipeline);
  }
  pipeline.producer_commit();

  state_t<vec_size> s_partial;
  float x = 0.f;
  size_t copy_stage_idx = 2, compute_stage_idx = 0, batch;

#pragma unroll 2
  for (batch = 1; batch < (kv_chunk_size + bdy - 1) / bdy; ++batch) {
    consumer_kv_idx = producer_kv_idx;
    consumer_pred_guard = producer_pred_guard;
    producer_kv_idx = chunk_start + batch * bdy + ty;
    producer_pred_guard = producer_kv_idx < chunk_end;
    // load stage: load k tiles
    pipeline.producer_acquire();
    if (producer_pred_guard) {
      cuda::memcpy_async(smem + kv_shared_offset[copy_stage_idx] + ty * head_dim + tx * vec_size,
                         k + kv_info.get_kv_elem_offset(producer_kv_idx, head_idx, tx * vec_size),
                         frag_shape, pipeline);
    }
    pipeline.producer_commit();
    copy_stage_idx = (copy_stage_idx + 1) % stages_count;

    // compute stage: compute qk
    pipeline.consumer_wait();
    block.sync();
    compute_qk<rotary_mode, vec_size, bdx>(smem, q_vec, freq, kv_shared_offset, consumer_kv_idx,
                                           compute_stage_idx, num_heads, sm_scale, x);
    block.sync();
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;

    // load stage: load v tiles
    pipeline.producer_acquire();
    if (producer_pred_guard) {
      cuda::memcpy_async(smem + kv_shared_offset[copy_stage_idx] + ty * head_dim + tx * vec_size,
                         v + kv_info.get_kv_elem_offset(producer_kv_idx, head_idx, tx * vec_size),
                         frag_shape, pipeline);
    }
    pipeline.producer_commit();
    copy_stage_idx = (copy_stage_idx + 1) % stages_count;

    // compute stage: update partial state
    pipeline.consumer_wait();
    block.sync();
    update_partial_state<vec_size, bdx>(smem, x, kv_shared_offset, compute_stage_idx,
                                        consumer_pred_guard, s_partial);
    block.sync();
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;
  }

  // last two compute stages
  {
    consumer_kv_idx = producer_kv_idx;
    consumer_pred_guard = producer_pred_guard;
    // compute stage: compute qk
    pipeline.consumer_wait();
    block.sync();
    compute_qk<rotary_mode, vec_size, bdx>(smem, q_vec, freq, kv_shared_offset, consumer_kv_idx,
                                           compute_stage_idx, num_heads, sm_scale, x);
    block.sync();
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;
    // compute stage: update partial state
    pipeline.consumer_wait();
    block.sync();
    update_partial_state<vec_size, bdx>(smem, x, kv_shared_offset, compute_stage_idx,
                                        consumer_pred_guard, s_partial);
    block.sync();
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;
  }

  // sync partial state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy>(s_partial, reinterpret_cast<float *>(smem), smem_md);

  if constexpr (cooperative) {
    // update tmp buffer
    s_partial.o.store(tmp + (head_idx * num_kv_chunks + kv_chunk_idx) * head_dim + tx * vec_size);
    float *tmp_md = tmp + num_heads * num_kv_chunks * head_dim;
    tmp_md[(head_idx * num_kv_chunks + kv_chunk_idx) * 2] = s_partial.m;
    tmp_md[(head_idx * num_kv_chunks + kv_chunk_idx) * 2 + 1] = s_partial.d;
    grid.sync();

    // sync global states
    if (kv_chunk_idx == 0) {
      state_t<vec_size> s_global;
#pragma unroll 2
      for (size_t batch = 0; batch < (num_kv_chunks + bdy - 1) / bdy; ++batch) {
        size_t kv_chunk_idx = batch * bdy + ty;
        if (kv_chunk_idx < num_kv_chunks) {
          s_partial.m = tmp_md[(head_idx * num_kv_chunks + kv_chunk_idx) * 2];
          s_partial.d = tmp_md[(head_idx * num_kv_chunks + kv_chunk_idx) * 2 + 1];
          s_partial.o.load(tmp + (head_idx * num_kv_chunks + kv_chunk_idx) * head_dim +
                           tx * vec_size);
          s_global.merge(s_partial);
        }
      }
      block.sync();
      // sync partial state of all warps inside a threadblock
      sync_state<vec_size, bdx, bdy>(s_global, reinterpret_cast<float *>(smem), smem_md);
      s_global.o.cast_store(o + head_idx * head_dim + tx * vec_size);
      tmp[head_idx] = s_global.m;
      tmp[num_heads + head_idx] = s_global.d;
    }
  } else {
    s_partial.o.cast_store(o + head_idx * head_dim + tx * vec_size);
  }
}

/*!
 * \brief FlashAttention decoding cuda kernel with PagedKVCcache for batch requests,
 *   fused with RoPE.
 * \tparam rotary_mode The rotary mode
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param q [batch_size, num_heads, head_dim] The query matrix
 * \param paged_kv The PagedKVCache data structure
 * \param o [num_heads, head_dim] The output matrix
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param rope_inv_scale A floating number indicate the multiplicative inverse
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_inv_theta A floating number indicate the multiplicative inverse
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 */
template <RotaryMode rotary_mode, size_t vec_size, size_t bdx, size_t bdy, typename DTypeIn,
          typename DTypeOut, typename IdType>
__global__ void BatchDecodeWithPagedKVCacheKernel(DTypeIn *__restrict__ q,
                                                  paged_kv_t<DTypeIn, IdType> paged_kv,
                                                  DTypeOut *__restrict__ o, float sm_scale,
                                                  float rope_inv_scale, float rope_inv_theta) {
  auto block = cg::this_thread_block();

  constexpr size_t stages_count = 4;
  constexpr size_t head_dim = bdx * vec_size;
  size_t batch_idx = blockIdx.x;
  size_t head_idx = blockIdx.y;
  size_t num_heads = gridDim.y;
  size_t cur_page_indptr_begin = paged_kv.indptr[batch_idx],
         cur_page_indptr_end = paged_kv.indptr[batch_idx + 1];
  size_t cur_last_page_offset = paged_kv.last_page_offset[batch_idx];
  size_t seq_len =
      (cur_page_indptr_end - cur_page_indptr_begin - 1) * paged_kv.page_size + cur_last_page_offset;

  static_assert(bdx * bdy == 128);
  static_assert(stages_count >= sizeof(float) / sizeof(DTypeIn));
  __shared__ DTypeIn smem[stages_count * bdy * head_dim];
  __shared__ float smem_md[2 * bdy];

  size_t tx = threadIdx.x, ty = threadIdx.y;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (rotary_mode == RotaryMode::kLlama) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_inv_scale *
                __powf(rope_inv_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
    // apply rotary embedding to q matrix
    q_vec = apply_llama_rope<vec_size, bdx>(q + (batch_idx * num_heads + head_idx) * head_dim, freq,
                                            seq_len - 1);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(q + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
  }
  block.sync();

  // load k tiles and v tiles
  size_t kv_shared_offset[stages_count] = {0U, 1U * bdy * head_dim, 2U * bdy * head_dim,
                                           3U * bdy * head_dim};

  // pipelining k/v tiles loading and state updating
  auto pipeline = cuda::make_pipeline();
  const auto frag_shape = cuda::aligned_size_t<alignof(float4)>(sizeof(DTypeIn) * vec_size);
  size_t producer_kv_idx = ty, consumer_kv_idx;
  bool producer_pred_guard = producer_kv_idx < min(seq_len, paged_kv.page_size),
       consumer_pred_guard = true;
  size_t page_idx = paged_kv.indices[cur_page_indptr_begin];
  pipeline.producer_acquire();
  if (producer_pred_guard) {
    cuda::memcpy_async(
        smem + kv_shared_offset[0] + ty * head_dim + tx * vec_size,
        paged_kv.data + paged_kv.get_k_elem_offset(page_idx, head_idx, ty, tx * vec_size),
        frag_shape, pipeline);
  }
  pipeline.producer_commit();
  pipeline.producer_acquire();
  if (producer_pred_guard) {
    cuda::memcpy_async(
        smem + kv_shared_offset[1] + ty * head_dim + tx * vec_size,
        paged_kv.data + paged_kv.get_v_elem_offset(page_idx, head_idx, ty, tx * vec_size),
        frag_shape, pipeline);
  }
  pipeline.producer_commit();

  state_t<vec_size> s;
  float x = 0.f;
  size_t copy_stage_idx = 2, compute_stage_idx = 0, batch;

  for (size_t page_iter = cur_page_indptr_begin; page_iter < cur_page_indptr_end; ++page_iter) {
    page_idx = paged_kv.indices[page_iter];
    size_t valid_page_size =
        (page_iter == cur_page_indptr_end - 1) ? cur_last_page_offset : paged_kv.page_size;

#pragma unroll 2
    for (batch = (page_iter == cur_page_indptr_begin); batch < (valid_page_size + bdy - 1) / bdy;
         ++batch) {
      consumer_kv_idx = producer_kv_idx;
      consumer_pred_guard = producer_pred_guard;
      size_t cur_page_producer_kv_idx = batch * bdy + ty;
      producer_kv_idx =
          cur_page_producer_kv_idx + (page_iter - cur_page_indptr_begin) * paged_kv.page_size;
      producer_pred_guard = cur_page_producer_kv_idx < valid_page_size;
      // load stage: load k tiles
      pipeline.producer_acquire();
      if (producer_pred_guard) {
        cuda::memcpy_async(
            smem + kv_shared_offset[copy_stage_idx] + ty * head_dim + tx * vec_size,
            paged_kv.data + paged_kv.get_k_elem_offset(page_idx, head_idx, cur_page_producer_kv_idx,
                                                       tx * vec_size),
            frag_shape, pipeline);
      }
      pipeline.producer_commit();
      copy_stage_idx = (copy_stage_idx + 1) % stages_count;

      // compute stage: compute qk
      pipeline.consumer_wait();
      block.sync();
      compute_qk<rotary_mode, vec_size, bdx>(smem, q_vec, freq, kv_shared_offset, consumer_kv_idx,
                                             compute_stage_idx, num_heads, sm_scale, x);
      block.sync();
      pipeline.consumer_release();
      compute_stage_idx = (compute_stage_idx + 1) % stages_count;

      // load stage: load v tiles
      pipeline.producer_acquire();
      if (producer_pred_guard) {
        cuda::memcpy_async(
            smem + kv_shared_offset[copy_stage_idx] + ty * head_dim + tx * vec_size,
            paged_kv.data + paged_kv.get_v_elem_offset(page_idx, head_idx, cur_page_producer_kv_idx,
                                                       tx * vec_size),
            frag_shape, pipeline);
      }
      pipeline.producer_commit();
      copy_stage_idx = (copy_stage_idx + 1) % stages_count;

      // compute stage: update partial state
      pipeline.consumer_wait();
      block.sync();
      update_partial_state<vec_size, bdx>(smem, x, kv_shared_offset, compute_stage_idx,
                                          consumer_pred_guard, s);
      block.sync();
      pipeline.consumer_release();
      compute_stage_idx = (compute_stage_idx + 1) % stages_count;
    }
  }

  // last two compute stages
  {
    consumer_kv_idx = producer_kv_idx;
    consumer_pred_guard = producer_pred_guard;
    // compute stage: compute qk
    pipeline.consumer_wait();
    block.sync();
    compute_qk<rotary_mode, vec_size, bdx>(smem, q_vec, freq, kv_shared_offset, consumer_kv_idx,
                                           compute_stage_idx, num_heads, sm_scale, x);
    block.sync();
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;
    // compute stage: update partial state
    pipeline.consumer_wait();
    block.sync();
    update_partial_state<vec_size, bdx>(smem, x, kv_shared_offset, compute_stage_idx,
                                        consumer_pred_guard, s);
    block.sync();
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;
  }

  // sync partial state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy>(s, reinterpret_cast<float *>(smem), smem_md);

  // update global states
  s.o.cast_store(o + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
}

/*!
 * \brief FlashAttention decoding with kv-cache for a single sequence
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \param q The query matrix, shape: [num_heads, head_dim]
 * \param k The key matrix in kv-cache, shape: [seq_len, num_heads, head_dim]
 *   for NHD layout, [num_heads, head_dim, seq_len] for HND layout
 * \param v The value matrix in kv-cache, shape: [seq_len, num_heads, head_dim]
 *   for NHD layout, [num_heads, head_dim, seq_len] for HND layout
 * \param o The output matrix, shape: [num_heads, head_dim]
 * \param tmp Used-allocated temporary buffer
 * \param num_heads A integer indicates the number of heads
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param layout The layout of q/k/v matrices.
 * \param rotary_mode The rotary mode
 * \param rope_scale A floating point number indicate the scaling ratio
 *   used in RoPE Interpolation.
 * \param rope_theta A floating point number indicate the "theta" used in RoPE
 * \param stream The cuda stream to launch the kernel
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t SingleDecodeWithKVCache(DTypeIn *q, DTypeIn *k, DTypeIn *v, DTypeOut *o, float *tmp,
                                    size_t num_heads, size_t seq_len, size_t head_dim,
                                    QKVLayout layout = QKVLayout::kNHD,
                                    RotaryMode rotary_mode = RotaryMode::kNone,
                                    float rope_scale = 1.f, float rope_theta = 1e4,
                                    cudaStream_t stream = nullptr, size_t dev_id = 0) {
  const float sm_scale = 1.f / std::sqrt(float(head_dim));
  const float rope_inv_scale = 1.f / rope_scale;
  const float rope_inv_theta = 1.f / rope_theta;

  FLASHINFER_CUDA_CALL(cudaSetDevice(dev_id));

  SWITCH_HEAD_DIM(
      head_dim, HEAD_DIM,
      {SWITCH_ROTARY_MODE(
          rotary_mode, ROTARY_MODE, {SWITCH_LAYOUT(layout, QKV_LAYOUT, {
            constexpr size_t vec_size = std::max(16 / sizeof(DTypeIn), HEAD_DIM / 32);
            constexpr size_t bdx = HEAD_DIM / vec_size;
            constexpr size_t bdy = 128 / bdx;
            tensor_info_t<QKV_LAYOUT> kv_info(1, seq_len, num_heads, head_dim);
            if (seq_len <= 128) {
              // no need to use cooperative kernel
              auto kernel = SingleDecodeWithKVCacheKernel<QKV_LAYOUT, false, ROTARY_MODE, vec_size,
                                                          bdx, bdy, DTypeIn, DTypeOut>;
              dim3 nblks = dim3(1, num_heads);
              dim3 nthrs = dim3(bdx, bdy);
              void *args[] = {(void *)&q,
                              (void *)&k,
                              (void *)&v,
                              (void *)&o,
                              (void *)&tmp,
                              (void *)&kv_info,
                              (void *)&sm_scale,
                              (void *)&rope_inv_scale,
                              (void *)&rope_inv_theta,
                              (void *)&seq_len};
              FLASHINFER_CUDA_CALL(cudaLaunchKernel((void *)kernel, nblks, nthrs, args, 0, stream));
            } else {
              // use cooperative kernel
              auto kernel = SingleDecodeWithKVCacheKernel<QKV_LAYOUT, true, ROTARY_MODE, vec_size,
                                                          bdx, bdy, DTypeIn, DTypeOut>;
              int num_blocks_per_sm = 0;
              int num_sm = 0;
              FLASHINFER_CUDA_CALL(
                  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
              FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
                                                                                 kernel, 128, 0));
              size_t max_num_blks = size_t(num_blocks_per_sm) * size_t(num_sm);
              size_t max_num_kv_chunks = max_num_blks / num_heads;
              size_t kv_chunk_size =
                  max((seq_len + max_num_kv_chunks - 1UL) / max_num_kv_chunks,
                      min(128UL, max(16UL, seq_len / max(1UL, (128UL / num_heads)))));
              dim3 nblks = dim3((seq_len + kv_chunk_size - 1) / kv_chunk_size, num_heads);
              assert(nblks.x > 0 && nblks.y > 0);
              dim3 nthrs = dim3(bdx, bdy);
              void *args[] = {(void *)&q,
                              (void *)&k,
                              (void *)&v,
                              (void *)&o,
                              (void *)&tmp,
                              (void *)&kv_info,
                              (void *)&sm_scale,
                              (void *)&rope_inv_scale,
                              (void *)&rope_inv_theta,
                              (void *)&kv_chunk_size};
              FLASHINFER_CUDA_CALL(
                  cudaLaunchCooperativeKernel((void *)kernel, nblks, nthrs, args, 0, stream));
            }
          })})});
  return cudaSuccess;
}

/*!
 * \brief FlashAttention decoding cuda kernel with paged kv-cache for batched requests
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type used in paged kv-cache
 * \param q [batch_size, num_heads, head_dim] The query matrix
 * \param paged_kv The paged kv cache data structure
 * \param o [batch_size, num_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param rotary_mode The rotary mode
 * \param rope_scale A floating point number indicate the scaling ratio
 *   used in RoPE Interpolation.
 * \param rope_theta A floating point number indicate the "theta" used in RoPE
 * \param stream The cuda stream to launch the kernel
 * \param dev_id The device id
 */
template <typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCache(DTypeIn *q, paged_kv_t<DTypeIn, IdType> paged_kv,
                                        DTypeOut *o, float *tmp,
                                        RotaryMode rotary_mode = RotaryMode::kNone,
                                        float rope_scale = 1.f, float rope_theta = 1e4,
                                        cudaStream_t stream = nullptr, size_t dev_id = 0) {
  const float sm_scale = 1.f / std::sqrt(float(paged_kv.head_dim));
  const float rope_inv_scale = 1.f / rope_scale;
  const float rope_inv_theta = 1.f / rope_theta;

  FLASHINFER_CUDA_CALL(cudaSetDevice(dev_id));

  SWITCH_HEAD_DIM(
      paged_kv.head_dim, HEAD_DIM, {SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
        constexpr size_t vec_size = std::max(16 / sizeof(DTypeIn), HEAD_DIM / 32);
        constexpr size_t bdx = HEAD_DIM / vec_size;
        constexpr size_t bdy = 128 / bdx;
        dim3 nblks(paged_kv.batch_size, paged_kv.num_heads);
        dim3 nthrs(bdx, bdy);
        auto kernel = BatchDecodeWithPagedKVCacheKernel<ROTARY_MODE, vec_size, bdx, bdy, DTypeIn,
                                                        DTypeOut, IdType>;
        void *args[] = {(void *)&q,        (void *)&paged_kv,       (void *)&o,
                        (void *)&sm_scale, (void *)&rope_inv_scale, (void *)&rope_inv_theta};
        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void *)kernel, nblks, nthrs, args, 0, stream));
      })});

  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_CUH_
