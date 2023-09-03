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

#include "rope.cuh"
#include "state.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

namespace cg = cooperative_groups;

/*!
 * \brief Load k tile from smem and compute qk
 * \tparam rotary_mode The rotary mode used in the kernel
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam bdz A template integer indicates the block size in z dimension
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param rotary_emb A vector of float indicates the thread-local rotary embedding
 * \param kv_shared_offset An array of size_t indicates the k/v tiles offset in shared
 *   memory of different pipeline stages
 * \param offset A integer indicates the warp-local offset of the position in kv-cache.
 * \param compute_stage_idx A integer indicates the compute stage index in
 *   the pipeline
 * \param num_heads A integer indicates the number of heads
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param x A float indicates the thread-local result of qk
 */
template <RotaryMode rotary_mode, size_t head_dim, size_t vec_size, size_t bdx, size_t bdy,
          size_t bdz, typename T>
__device__ __forceinline__ void compute_qk(const T *smem, const vec_t<float, vec_size> &q_vec,
                                           const vec_t<float, vec_size> &rotary_emb,
                                           const size_t *kv_shared_offset, size_t offset,
                                           size_t compute_stage_idx, size_t num_heads,
                                           float sm_scale, float &x) {
  vec_t<float, vec_size> k_vec;
  size_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  if constexpr (rotary_mode == RotaryMode::kApplyRotary) {
    // apply rotary embedding for all rows in k matrix of kv-cache
    k_vec = apply_rotary<head_dim, vec_size>(
        smem + kv_shared_offset[compute_stage_idx] + (tz * bdy + ty) * head_dim, rotary_emb,
        offset + tz);
  } else {
    // do not apply rotary embedding
    k_vec.cast_load(smem + kv_shared_offset[compute_stage_idx] + (tz * bdy + ty) * head_dim +
                    tx * vec_size);
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
 * \brief Load v tile from shared memory and update partial state
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam bdz A template integer indicates the block size in z dimension
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param x A float indicates the pre-softmax logits
 * \param kv_shared_offset An array of size_t indicates the k/v tiles offset in shared
 *   memory of different pipeline stages.
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param offset A integer indicates the warp-local offset of the position in kv-cache
 * \param chunk_end A integer indicates the end position of the kv-chunk in
 *   current threadblock
 * \param s The flashattention state to be updated
 */
template <size_t head_dim, size_t vec_size, size_t bdx, size_t bdy, size_t bdz, typename T>
__device__ __forceinline__ void update_partial_state(const T *smem, const float x,
                                                     const size_t *kv_shared_offset,
                                                     size_t compute_stage_idx, size_t offset,
                                                     size_t chunk_end, state_t<vec_size> &s) {
  vec_t<float, vec_size> v_vec;
  size_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  v_vec.cast_load(smem + kv_shared_offset[compute_stage_idx] + (tz * bdy + ty) * head_dim +
                  tx * vec_size);
  if (offset + tz < chunk_end) {
    s.merge(x, v_vec);
  }
}

/*!
 * \brief Synchronize the state of all warps inside a threadblock
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam bdz A template integer indicates the block size in z dimension
 * \param s The warp local state
 * \param smem The pointer to shared memory buffer for o
 * \param smem_md The pointer to shared memory buffer for m/d
 */
template <size_t head_dim, size_t vec_size, size_t bdx, size_t bdy, size_t bdz>
__device__ __forceinline__ void sync_state(state_t<vec_size> &s, float *smem, float *smem_md) {
  auto block = cg::this_thread_block();
  size_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  s.o.store(smem + (tz * bdy + ty) * head_dim + tx * vec_size);
  smem_md[(tz * bdy + ty) * 2] = s.m;
  smem_md[(tz * bdy + ty) * 2 + 1] = s.d;
  block.sync();
  s.init();
#pragma unroll
  for (size_t j = 0; j < bdz; ++j) {
    float mj = smem_md[(j * bdy + ty) * 2], dj = smem_md[(j * bdy + ty) * 2 + 1];
    vec_t<float, vec_size> oj;
    oj.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
    s.merge(mj, dj, oj);
  }
}

/*!
 * \brief FlashAttention decoding cuda kernel with kv-cache for a single
 * sequence, fused with RoPE.
 * \tparam rotary_mode The rotary mode
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam bdz A template integer indicates the block size in z dimension
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \param q [num_heads, head_dim] The query matrix
 * \param k [seq_len, num_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_heads, head_dim] The value matrix in kv-cache
 * \param o [num_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param rope_inv_scale A floating number indicate the multiplicative inverse
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_inv_theta A floating number indicate the multiplicative inverse
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 * \param kv_chunk_size A integer indicates the kv-chunk size
 */
template <RotaryMode rotary_mode, size_t head_dim, size_t vec_size, size_t bdx, size_t bdy,
          size_t bdz, typename DTypeIn, typename DTypeOut>
__global__ void SingleDecodeWithKVCacheKernel(DTypeIn *__restrict__ q, DTypeIn *__restrict__ k,
                                              DTypeIn *__restrict__ v, DTypeOut *__restrict__ o,
                                              float *__restrict__ tmp, float sm_scale,
                                              size_t seq_len, float rope_inv_scale,
                                              float rope_inv_theta, size_t kv_chunk_size) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();

  constexpr size_t stages_count = 4;
  size_t head_idx_start = blockIdx.y * bdy;
  size_t kv_chunk_idx = blockIdx.x;
  size_t num_kv_chunks = gridDim.x;
  size_t num_heads = gridDim.y * bdy;

  static_assert(bdx * bdy * bdz == 128);
  static_assert(bdx * vec_size == head_dim);
  static_assert(stages_count >= sizeof(float) / sizeof(DTypeIn));
  __shared__ DTypeIn smem[stages_count * bdz * bdy * head_dim];
  __shared__ float smem_md[2 * bdz * bdy];

  size_t tx = threadIdx.x, ty = threadIdx.y, head_idx = head_idx_start + ty, tz = threadIdx.z;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> rotary_emb;
  if constexpr (rotary_mode == RotaryMode::kApplyRotary) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      rotary_emb[i] =
          rope_inv_scale *
          powf(rope_inv_theta, float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
    // apply rotary embedding to q matrix
    q_vec = apply_rotary<head_dim, vec_size>(q + head_idx * head_dim, rotary_emb, seq_len - 1);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(q + head_idx * head_dim + tx * vec_size);
  }
  block.sync();

  size_t chunk_start = kv_chunk_idx * kv_chunk_size;
  kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);
  size_t chunk_end = chunk_start + kv_chunk_size;

  // load k tiles and v tiles
  size_t kv_shared_offset[stages_count] = {0U, 1U * bdz * bdy * head_dim, 2U * bdz * bdy * head_dim,
                                           3U * bdz * bdy * head_dim};

  // pipelining k/v tiles loading and state updating
  auto pipeline = cuda::make_pipeline();
  const auto frag_shape = cuda::aligned_size_t<alignof(float4)>(sizeof(DTypeIn) * vec_size);
  pipeline.producer_acquire();
  if (tz < kv_chunk_size) {
    cuda::memcpy_async(smem + kv_shared_offset[0] + (tz * bdy + ty) * head_dim + tx * vec_size,
                       k + ((chunk_start + tz) * num_heads + head_idx) * head_dim + tx * vec_size,
                       frag_shape, pipeline);
  }
  pipeline.producer_commit();
  pipeline.producer_acquire();
  if (tz < kv_chunk_size) {
    cuda::memcpy_async(smem + kv_shared_offset[1] + (tz * bdy + ty) * head_dim + tx * vec_size,
                       v + ((chunk_start + tz) * num_heads + head_idx) * head_dim + tx * vec_size,
                       frag_shape, pipeline);
  }
  pipeline.producer_commit();

  state_t<vec_size> s_partial;
  float x = 0.f;
  size_t copy_stage_idx = 2, compute_stage_idx = 0, batch;

#pragma unroll 2
  for (batch = 1; batch < (kv_chunk_size + bdz - 1) / bdz; ++batch) {
    size_t kv_idx = chunk_start + batch * bdz + tz;
    // load stage: load k tiles
    pipeline.producer_acquire();
    if (kv_idx < chunk_end) {
      cuda::memcpy_async(
          smem + kv_shared_offset[copy_stage_idx] + (tz * bdy + ty) * head_dim + tx * vec_size,
          k + (kv_idx * num_heads + head_idx) * head_dim + tx * vec_size, frag_shape, pipeline);
    }
    pipeline.producer_commit();
    copy_stage_idx = (copy_stage_idx + 1) % stages_count;

    // compute stage: compute qk
    pipeline.consumer_wait();
    block.sync();
    compute_qk<rotary_mode, head_dim, vec_size, bdx, bdy, bdz>(
        smem, q_vec, rotary_emb, kv_shared_offset, chunk_start + (batch - 1) * bdz,
        compute_stage_idx, num_heads, sm_scale, x);
    block.sync();
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;

    // load stage: load v tiles
    pipeline.producer_acquire();
    if (kv_idx < chunk_end) {
      cuda::memcpy_async(
          smem + kv_shared_offset[copy_stage_idx] + (tz * bdy + ty) * head_dim + tx * vec_size,
          v + (kv_idx * num_heads + head_idx) * head_dim + tx * vec_size, frag_shape, pipeline);
    }
    pipeline.producer_commit();
    copy_stage_idx = (copy_stage_idx + 1) % stages_count;

    // compute stage: update partial state
    pipeline.consumer_wait();
    block.sync();
    update_partial_state<head_dim, vec_size, bdx, bdy, bdz>(
        smem, x, kv_shared_offset, compute_stage_idx, chunk_start + (batch - 1) * bdz, chunk_end,
        s_partial);
    block.sync();
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;
  }

  // last two compute stages
  {
    batch = (kv_chunk_size + bdz - 1) / bdz;
    // compute stage: compute qk
    pipeline.consumer_wait();
    block.sync();
    compute_qk<rotary_mode, head_dim, vec_size, bdx, bdy, bdz>(
        smem, q_vec, rotary_emb, kv_shared_offset, chunk_start + (batch - 1) * bdz,
        compute_stage_idx, num_heads, sm_scale, x);
    block.sync();
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;
    // compute stage: update partial state
    pipeline.consumer_wait();
    block.sync();
    update_partial_state<head_dim, vec_size, bdx, bdy, bdz>(
        smem, x, kv_shared_offset, compute_stage_idx, chunk_start + (batch - 1) * bdz, chunk_end,
        s_partial);
    block.sync();
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;
  }

  // sync partial state of all warps inside a threadblock
  sync_state<head_dim, vec_size, bdx, bdy, bdz>(s_partial, reinterpret_cast<float *>(smem),
                                                smem_md);

  // update tmp buffer
  s_partial.o.store(tmp + (kv_chunk_idx * num_heads + head_idx) * head_dim + tx * vec_size);
  float *tmp_md = tmp + num_kv_chunks * num_heads * head_dim;
  tmp_md[(kv_chunk_idx * num_heads + head_idx) * 2] = s_partial.m;
  tmp_md[(kv_chunk_idx * num_heads + head_idx) * 2 + 1] = s_partial.d;
  grid.sync();

  // sync global states
  if (kv_chunk_idx == 0) {
    state_t<vec_size> s_global;
#pragma unroll 2
    for (size_t batch = 0; batch < (num_kv_chunks + bdz - 1) / bdz; ++batch) {
      size_t kv_chunk_idx = batch * bdz + tz;
      if (kv_chunk_idx < num_kv_chunks) {
        s_partial.m = tmp_md[(kv_chunk_idx * num_heads + head_idx) * 2];
        s_partial.d = tmp_md[(kv_chunk_idx * num_heads + head_idx) * 2 + 1];
        s_partial.o.load(tmp + (kv_chunk_idx * num_heads + head_idx) * head_dim + tx * vec_size);
        s_global.merge(s_partial);
      }
    }
    block.sync();
    // sync partial state of all warps inside a threadblock
    sync_state<head_dim, vec_size, bdx, bdy, bdz>(s_global, reinterpret_cast<float *>(smem),
                                                  smem_md);
    s_global.o.cast_store(o + head_idx * head_dim + tx * vec_size);
    tmp[head_idx] = s_global.m;
    tmp[num_heads + head_idx] = s_global.d;
  }
}

#define SWITCH_HEAD_DIM(head_dim, HEAD_DIM, ...)                      \
  switch (head_dim) {                                                 \
    case 32: {                                                        \
      constexpr size_t HEAD_DIM = 32;                                 \
      __VA_ARGS__                                                     \
      break;                                                          \
    }                                                                 \
    case 64: {                                                        \
      constexpr size_t HEAD_DIM = 64;                                 \
      __VA_ARGS__                                                     \
      break;                                                          \
    }                                                                 \
    case 128: {                                                       \
      constexpr size_t HEAD_DIM = 128;                                \
      __VA_ARGS__                                                     \
      break;                                                          \
    }                                                                 \
    case 256: {                                                       \
      constexpr size_t HEAD_DIM = 256;                                \
      __VA_ARGS__                                                     \
      break;                                                          \
    }                                                                 \
    default: {                                                        \
      std::cerr << "Unsupported head_dim: " << head_dim << std::endl; \
      abort();                                                        \
    }                                                                 \
  }

#define SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, ...)                        \
  switch (rotary_mode) {                                                         \
    case RotaryMode::kNone: {                                                    \
      constexpr RotaryMode ROTARY_MODE = RotaryMode::kNone;                      \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    case RotaryMode::kApplyRotary: {                                             \
      constexpr RotaryMode ROTARY_MODE = RotaryMode::kApplyRotary;               \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    default: {                                                                   \
      std::cerr << "Unsupported rotary_mode: " << int(rotary_mode) << std::endl; \
      abort();                                                                   \
    }                                                                            \
  }

#define CUDA_CALL(func, ...) \
  {                          \
    cudaError_t e = (func);  \
    if (e != cudaSuccess) {  \
      return e;              \
    }                        \
  }

/*!
 * \brief FlashAttention decoding with kv-cache for a single sequence
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \param q [num_heads, head_dim] The query matrix
 * \param k [seq_len, num_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_heads, head_dim] The value matrix in kv-cache
 * \param o [num_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param num_heads A integer indicates the number of heads
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param rotary_mode The rotary mode
 * \param rope_scale A floating point number indicate the scaling ratio
 *   used in RoPE Interpolation.
 * \param rope_theta A floating point number indicate the "theta" used in RoPE
 * \param stream The cuda stream to launch the kernel
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t SingleDecodeWithKVCache(DTypeIn *q, DTypeIn *k, DTypeIn *v, DTypeOut *o, float *tmp,
                                    size_t num_heads, size_t seq_len, size_t head_dim,
                                    RotaryMode rotary_mode = RotaryMode::kNone,
                                    float rope_scale = 1.f, float rope_theta = 1e4,
                                    cudaStream_t stream = nullptr, size_t dev_id = 0) {
  const float sm_scale = 1.f / std::sqrt(float(head_dim));
  const float rope_inv_scale = 1.f / rope_scale;
  const float rope_inv_theta = 1.f / rope_theta;

  SWITCH_HEAD_DIM(
      head_dim, HEAD_DIM, {SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
        constexpr size_t vec_size = std::max(16 / sizeof(DTypeIn), HEAD_DIM / 32);
        constexpr size_t bdx = HEAD_DIM / vec_size;
        constexpr size_t bdy = 32 / bdx;
        constexpr size_t bdz = 4;
        auto kernel = SingleDecodeWithKVCacheKernel<ROTARY_MODE, HEAD_DIM, vec_size, bdx, bdy, bdz,
                                                    DTypeIn, DTypeOut>;
        int num_blocks_per_sm = 0;
        int num_sm = 0;
        CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
        CUDA_CALL(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, 128, 0));
        size_t max_num_blks = size_t(num_blocks_per_sm) * size_t(num_sm);
        size_t max_num_kv_chunks = max_num_blks / (num_heads / bdy);
        size_t kv_chunk_size =
            max((seq_len + max_num_kv_chunks - 1UL) / max_num_kv_chunks,
                min(64UL, max(4UL, seq_len / max(1UL, (64UL * bdy / num_heads)))));
        dim3 nblks = dim3((seq_len + kv_chunk_size - 1) / kv_chunk_size, num_heads / bdy);
        assert(nblks.x > 0 && nblks.y > 0);
        dim3 nthrs = dim3(bdx, bdy, bdz);
        void *args[] = {(void *)&q,
                        (void *)&k,
                        (void *)&v,
                        (void *)&o,
                        (void *)&tmp,
                        (void *)&sm_scale,
                        (void *)&seq_len,
                        (void *)&rope_inv_scale,
                        (void *)&rope_inv_theta,
                        (void *)&kv_chunk_size};
        CUDA_CALL(cudaLaunchCooperativeKernel((void *)kernel, nblks, nthrs, args, 0, stream));
        return cudaSuccess;
      })});
}

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_CUH_
