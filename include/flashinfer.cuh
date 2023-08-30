#ifndef FLASHINFER_CUH_
#define FLASHINFER_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>
#include <iostream>
#include <random>

#include "vec_dtypes.cuh"

namespace flashinfer {

namespace cg = cooperative_groups;

/*!
 * \brief An enumeration class that defines different modes for applying RoPE (Rotary Positional
 * Embeddings).
 */
enum class RotaryMode {
  // No rotary positional embeddings
  kNone = 0U,
  // Apply rotary positional embeddings to q and all rows in k matrix of kv-cache, while keeping the
  // kv-cache unchanged.
  kApplyRotary = 1U,
  // Apply rotary positional embeddings to q and the newly appended rows in k matrix of kv-cache,
  // and write
  // back the updated rows in k matrix of kv-cache.
  kApplyRotaryUpdateLastK = 2U,
};

inline std::string RotaryModeToString(const RotaryMode &rotary_mode) {
  switch (rotary_mode) {
    case RotaryMode::kNone:
      return "None";
    case RotaryMode::kApplyRotary:
      return "ApplyRotary";
    case RotaryMode::kApplyRotaryUpdateLastK:
      return "ApplyRotaryUpdateLastK";
    default:
      return "Unknown";
  }
}

/*!
 * \brief Apply RoPE (Rotary Positional Embeddings) to input[0: head_dim], return
 *   thread-local vector.
 * \tparam vec_size A template integer indicates the vector size used in the kernel.
 * \tparam T A template type indicates the input data type
 * \param input A pointer to the start of input data
 * \param inv_freq A vector of float indicates the multiplicative inverse of frequency
 * \param offset A integer indicates the offset of the position in RoPE (Rotary Positional
 *   Embeddings).
 */
template <size_t vec_size, typename T>
__device__ __forceinline__ vec_t<float, vec_size> apply_rotary(
    const T *input, const vec_t<float, vec_size> &inv_freq, size_t offset, size_t head_dim) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(input + threadIdx.x * vec_size);
  permuted_vec.cast_load(input + ((threadIdx.x * vec_size < head_dim / 2)
                                      ? threadIdx.x * vec_size + head_dim / 2
                                      : threadIdx.x * vec_size - head_dim / 2));

#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    float freq = float(offset) * inv_freq[i];
    float cos, sin;
    __sincosf(freq, &sin, &cos);
    vec[i] = vec[i] * cos +
             ((threadIdx.x * vec_size < head_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin;
  }
  return vec;
}

/*!
 * \brief Load k tile from smem and compute qk
 * \tparam rotary_mode The rotary mode used in the kernel.
 * \tparam T A template type indicates the input data type
 * \tparam vec_size A template integer indicates the vector size used in the kernel.
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param kv_shared_offset An array of size_t indicates the k/v tiles offset in shared memory of
 *   different pipeline stages
 * \param offset A integer indicates the offset of the position in RoPE
 *   (Rotary Positional Embeddings)
 * \param compute_stage_idx A integer indicates the compute stage
 *   index in the pipeline
 * \param seq_len A integer indicates the sequence length
 * \param num_heads A integer indicates the number of heads
 * \param head_dim A integer indicates the head dimension
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param rope_inv_scale A floating number indicate the multiplicative inverse of scaling ratio
 *   used in PI(Position Interpolation) of RoPE (Rotary Positional Embeddings).
 * \param rope_inv_theta A floating number indicate the multiplicative inverse of "theta"
 *   used in RoPE (Rotary Positional Embeddings).
 * \param k_glob A pointer to the start of global memory of k matrix in kv-cache
 * \param mask
 * \param x A float indicates the thread-local result of qk  TODO(Zihao): fix doc
 */
template <RotaryMode rotary_mode, size_t head_dim, size_t vec_size, size_t blockdim_x,
          size_t blockdim_y, size_t blockdim_z, typename T>
__device__ __forceinline__ void compute_qk(const T *smem, const vec_t<float, vec_size> &q_vec,
                                           const vec_t<float, vec_size> &inv_freq,
                                           const size_t *kv_shared_offset, size_t offset,
                                           size_t compute_stage_idx, size_t seq_len,
                                           size_t num_heads, float sm_scale, float rope_inv_scale,
                                           float rope_inv_theta, T *k_glob, bool mask,
                                           float *x_shared) {
  vec_t<float, vec_size> k_vec;
  size_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  size_t head_idx_start = blockIdx.x * blockdim_y, head_idx = head_idx_start + ty;
  size_t kv_idx = offset + tz;
  if constexpr (rotary_mode == RotaryMode::kApplyRotary) {
    // apply rotary embedding for all rows in k matrix of kv-cache
    k_vec = apply_rotary<vec_size>(
        smem + kv_shared_offset[compute_stage_idx] + (tz * blockdim_y + ty) * head_dim, inv_freq,
        offset, head_dim);
  } else if constexpr (rotary_mode == RotaryMode::kApplyRotaryUpdateLastK) {
    // apply rotary embedding for the newly appended rows in k matrix of kv-cache
    if (kv_idx == seq_len - 1) {
      k_vec = apply_rotary<vec_size>(
          smem + kv_shared_offset[compute_stage_idx] + (tz * blockdim_y + ty) * head_dim, inv_freq,
          offset, head_dim);
      k_vec.cast_store(k_glob + (kv_idx * num_heads + head_idx) * head_dim + tx * vec_size);
    } else {
      k_vec.cast_load(smem + kv_shared_offset[compute_stage_idx] +
                      (tz * blockdim_y + ty) * head_dim + tx * vec_size);
    }
  } else {
    // do not apply rotary embedding
    k_vec.cast_load(smem + kv_shared_offset[compute_stage_idx] + (tz * blockdim_y + ty) * head_dim +
                    tx * vec_size);
  }
  float x = 0.f;
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    x += q_vec[i] * k_vec[i] * sm_scale;
  }
  cg::thread_block_tile g = cg::tiled_partition<blockdim_x>(cg::this_thread_block());
#pragma unroll
  for (size_t offset = blockdim_x / 2; offset > 0; offset /= 2) {
    x += g.shfl_down(x, offset);
  }
  x = g.shfl(x, 0);
  x_shared[tz * blockdim_y + ty] = mask ? x : -INFINITY;
}

/*!
 * \brief Load v tile from shared memory and update partial m,d,o status
 * \tparam T A template type indicates the input data type
 * \tparam vec_size A template integer indicates the vector size used in the kernel.
 * \param smem A pointer to the start of shared memory
 * \param x A float indicates the pre-softmax logits TODO(Zihao): fix docstring
 * \param kv_shared_offset An array of size_t indicates the k/v tiles offset in shared memory of
 *   different pipeline stages.
 * \param compute_stage_idx A integer indicates the compute stage index
 *   in the pipeline
 * \param head_dim A integer indicates the head dimension
 * \param m A float indicates the thread-local maximum value of pre-softmax logits
 * \param d A float indicates the thread-local sum of exp(pre-softmax logits - m)
 * \param o A vector of float indicates the thread-local output vector
 */
template <size_t head_dim, size_t vec_size, size_t blockdim_x, size_t blockdim_y, size_t blockdim_z,
          typename T>
__device__ __forceinline__ void update_partial_mdo(const T *smem, const float *x_shared,
                                                   const size_t *kv_shared_offset,
                                                   size_t compute_stage_idx, float &m, float &d,
                                                   vec_t<float, vec_size> &o) {
  vec_t<float, vec_size> v_vec;
  size_t tx = threadIdx.x, ty = threadIdx.y;
#pragma unroll
  for (size_t j = 0; j < blockdim_z; ++j) {
    v_vec.cast_load(smem + kv_shared_offset[compute_stage_idx] + (j * blockdim_y + ty) * head_dim +
                    tx * vec_size);
    float x = x_shared[j * blockdim_y + ty], m_prev = m, d_prev = d;
    m = max(m, x);
    d = d * __expf(m_prev - m) + __expf(x - m);
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] = o[i] * (__expf(m_prev - m) * d_prev / d) + v_vec[i] * (__expf(x - m) / d);
    }
  }
}

/*!
 * \brief FlashAttention decoding cuda kernel with kv-cache for a single sequence, fused with
 *   rotary positional embeddings (if applicable).
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam head_dim The head dimension used in the kernel.
 * \tparam rotary_mode The rotary mode used in the kernel.
 * \param q [num_heads, head_dim] The query matrix
 * \param k [seq_len, num_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_heads, head_dim] The value matrix in kv-cache
 * \param o [num_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer used in the kernel, recommended size is 8MB.
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param rope_inv_scale A floating number indicate the multiplicative inverse of scaling ratio
 *   used in PI(Position Interpolation) for RoPE (Rotary Positional Embeddings).
 * \param rope_inv_theta A floating number indicate the multiplicative inverse of "theta"
 *   used in RoPE (Rotary Positional Embeddings).
 * \param kv_chunk_size A integer indicates the kv-chunk size
 */
template <RotaryMode rotary_mode, size_t head_dim, size_t vec_size, size_t blockdim_x,
          size_t blockdim_y, size_t blockdim_z, typename DTypeIn, typename DTypeOut>
__global__ void SingleDecodeWithKVCacheKernel(DTypeIn *__restrict__ q, DTypeIn *__restrict__ k,
                                              DTypeIn *__restrict__ v, DTypeOut *__restrict__ o,
                                              float *__restrict__ tmp, float sm_scale,
                                              size_t seq_len, float rope_inv_scale,
                                              float rope_inv_theta, size_t kv_chunk_size) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();

  constexpr size_t stages_count = 4;
  // TODO(Zihao): consider switch blockIdx.x and blockIdx.y
  size_t head_idx_start = blockIdx.x * blockdim_y;
  size_t kv_chunk_idx = blockIdx.y;
  size_t num_kv_chunks = gridDim.y;
  size_t num_heads = gridDim.x * blockdim_y;

  __shared__ DTypeIn smem[stages_count * blockdim_z * blockdim_y * head_dim];
  __shared__ float x_shared[blockdim_z * blockdim_y];

  size_t tx = threadIdx.x, ty = threadIdx.y, head_idx = head_idx_start + ty, tz = threadIdx.z;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> inv_freq;
  if constexpr (rotary_mode == RotaryMode::kApplyRotary ||
                rotary_mode == RotaryMode::kApplyRotaryUpdateLastK) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      inv_freq[i] = rope_inv_scale *
                    __powf(rope_inv_theta,
                           float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
    // apply rotary embedding to q matrix
    q_vec = apply_rotary<vec_size>(q + head_idx * head_dim, inv_freq, seq_len - 1, head_dim);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(q + head_idx * head_dim + tx * vec_size);
  }
  block.sync();

  size_t chunk_start = kv_chunk_idx * kv_chunk_size;
  kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);
  size_t chunk_end = chunk_start + kv_chunk_size;

  // load k tiles and v tiles
  size_t kv_shared_offset[stages_count] = {0U, 1U * blockdim_z * blockdim_y * head_dim,
                                           2U * blockdim_z * blockdim_y * head_dim,
                                           3U * blockdim_z * blockdim_y * head_dim};

  // pipelining k/v tiles loading and m/d/o computation
  auto pipeline = cuda::make_pipeline();
  const auto frag_shape = cuda::aligned_size_t<alignof(float4)>(sizeof(DTypeIn) * vec_size);
  pipeline.producer_acquire();
  if (tz < kv_chunk_size) {
    cuda::memcpy_async(
        smem + kv_shared_offset[0] + (tz * blockdim_y + ty) * head_dim + tx * vec_size,
        k + ((chunk_start + tz) * num_heads + head_idx) * head_dim + tx * vec_size, frag_shape,
        pipeline);
  }
  pipeline.producer_commit();
  pipeline.producer_acquire();
  if (tz < kv_chunk_size) {
    cuda::memcpy_async(
        smem + kv_shared_offset[1] + (tz * blockdim_y + ty) * head_dim + tx * vec_size,
        v + ((chunk_start + tz) * num_heads + head_idx) * head_dim + tx * vec_size, frag_shape,
        pipeline);
  }
  pipeline.producer_commit();

  float m = -INFINITY;
  float d = 0.f;
  vec_t<float, vec_size> o_vec;
  o_vec.fill(0.f);

  size_t copy_stage_idx = 2, compute_stage_idx = 0, batch;

#pragma unroll 2
  for (batch = 1; batch < (kv_chunk_size + blockdim_z - 1) / blockdim_z; ++batch) {
    size_t kv_idx = chunk_start + batch * blockdim_z + tz;
    // load stage: load k tiles
    pipeline.producer_acquire();
    if (kv_idx < chunk_end) {
      cuda::memcpy_async(smem + kv_shared_offset[copy_stage_idx] +
                             (tz * blockdim_y + ty) * head_dim + tx * vec_size,
                         k + (kv_idx * num_heads + head_idx) * head_dim + tx * vec_size, frag_shape,
                         pipeline);
    }
    pipeline.producer_commit();
    copy_stage_idx = (copy_stage_idx + 1) % stages_count;

    // compute stage: compute qk
    pipeline.consumer_wait();
    block.sync();
    compute_qk<rotary_mode, head_dim, vec_size, blockdim_x, blockdim_y, blockdim_z>(
        smem, q_vec, inv_freq, kv_shared_offset, chunk_start + (batch - 1) * blockdim_z,
        compute_stage_idx, seq_len, num_heads, sm_scale, rope_inv_scale, rope_inv_theta, k,
        chunk_start + (batch - 1) * blockdim_z + tz < chunk_end, x_shared);
    block.sync();
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;

    // load stage: load v tiles
    pipeline.producer_acquire();
    if (kv_idx < chunk_end) {
      cuda::memcpy_async(smem + kv_shared_offset[copy_stage_idx] +
                             (tz * blockdim_y + ty) * head_dim + tx * vec_size,
                         v + (kv_idx * num_heads + head_idx) * head_dim + tx * vec_size, frag_shape,
                         pipeline);
    }
    pipeline.producer_commit();
    copy_stage_idx = (copy_stage_idx + 1) % stages_count;

    // compute stage: update partial m,d,o status
    pipeline.consumer_wait();
    block.sync();
    update_partial_mdo<head_dim, vec_size, blockdim_x, blockdim_y, blockdim_z>(
        smem, x_shared, kv_shared_offset, compute_stage_idx, m, d, o_vec);
    block.sync();
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;
  }
  // last two compute stages
  // compute stage: compute qk
  pipeline.consumer_wait();
  block.sync();
  compute_qk<rotary_mode, head_dim, vec_size, blockdim_x, blockdim_y, blockdim_z>(
      smem, q_vec, inv_freq, kv_shared_offset, chunk_start + (batch - 1) * blockdim_z,
      compute_stage_idx, seq_len, num_heads, sm_scale, rope_inv_scale, rope_inv_theta, k,
      chunk_start + (batch - 1) * blockdim_z + tz < chunk_end, x_shared);
  block.sync();
  pipeline.consumer_release();
  compute_stage_idx = (compute_stage_idx + 1) % stages_count;
  // compute stage: update partial m,d,o status
  pipeline.consumer_wait();
  block.sync();
  update_partial_mdo<head_dim, vec_size, blockdim_x, blockdim_y, blockdim_z>(
      smem, x_shared, kv_shared_offset, compute_stage_idx, m, d, o_vec);
  block.sync();
  pipeline.consumer_release();
  compute_stage_idx = (compute_stage_idx + 1) % stages_count;

  // update tmp buffer
  o_vec.store(tmp + (kv_chunk_idx * num_heads + head_idx) * head_dim + tx * vec_size);
  float *md_tmp = tmp + num_kv_chunks * num_heads * head_dim;
  if (tz == 0) {
    md_tmp[(kv_chunk_idx * num_heads + head_idx) * 2] = m;
    md_tmp[(kv_chunk_idx * num_heads + head_idx) * 2 + 1] = d;
  }
  grid.sync();

  if (kv_chunk_idx == 0 && tz == 0) {
    float m_prev = -INFINITY, m_now;
    float d_prev = 0.f, d_now;
    vec_t<float, vec_size> o_vec_acc;
    o_vec_acc.fill(0.f);
    for (size_t batch = 0; batch < num_kv_chunks; ++batch) {
      m = md_tmp[(batch * num_heads + head_idx) * 2];
      d = md_tmp[(batch * num_heads + head_idx) * 2 + 1];
      m_now = max(m_prev, m);
      d_now = d_prev * __expf(m_prev - m_now) + d * __expf(m - m_now);
      o_vec.load(tmp + (batch * num_heads + head_idx) * head_dim + tx * vec_size);
#pragma unroll
      for (size_t i = 0; i < vec_size; ++i) {
        o_vec_acc[i] = o_vec_acc[i] * (d_prev / d_now) * __expf(m_prev - m_now) +
                       o_vec[i] * (d / d_now) * __expf(m - m_now);
      }
      m_prev = m_now;
      d_prev = d_now;
    }
    o_vec_acc.cast_store(o + head_idx * head_dim + tx * vec_size);
    tmp[head_idx] = m_prev;
    tmp[num_heads + head_idx] = d_prev;
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
    case RotaryMode::kApplyRotaryUpdateLastK: {                                  \
      constexpr RotaryMode ROTARY_MODE = RotaryMode::kApplyRotaryUpdateLastK;    \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    default: {                                                                   \
      std::cerr << "Unsupported rotary_mode: " << int(rotary_mode) << std::endl; \
      abort();                                                                   \
    }                                                                            \
  }

/*!
 * \brief FlashAttention decoding with kv-cache for a single sequence.
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \param q [num_heads, head_dim] The query matrix
 * \param k [seq_len, num_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_heads, head_dim] The value matrix in kv-cache
 * \param o [num_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer used in the kernel, recommended size is 8MB.
 * \param num_heads A integer indicates the number of heads
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension,
 * \param rotary_mode The rotary mode used in the kernel.
 * \param rope_scale A floating point number indicate the scaling ratio used in RoPE interpolation.
 * \param rope_theta A floating point number indicate the "theta" used in RoPE.
 *   used in PI (Position Interpolation) for RoPE (Rotary Positional Embeddings).
 * \param stream The cuda stream to launch the kernel
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t SingleDecodeWithKVCache(DTypeIn *q, DTypeIn *k, DTypeIn *v, DTypeOut *o, float *tmp,
                                    size_t num_heads, size_t seq_len, size_t head_dim,
                                    RotaryMode rotary_mode = RotaryMode::kNone,
                                    float rope_scale = 1.f, float rope_theta = 1e4,
                                    cudaStream_t stream = nullptr) {
  const float sm_scale = 1.f / std::sqrt(float(head_dim));
  const float rope_inv_scale = 1.f / rope_scale;
  const float rope_inv_theta = 1.f / rope_theta;

  SWITCH_HEAD_DIM(
      head_dim, HEAD_DIM, {SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
        constexpr size_t vec_size = std::max(16 / sizeof(DTypeIn), HEAD_DIM / 32);
        constexpr size_t blockdim_x = HEAD_DIM / vec_size;
        constexpr size_t blockdim_y = 32 / blockdim_x;
        constexpr size_t blockdim_z = 4;
        auto kernel = SingleDecodeWithKVCacheKernel<ROTARY_MODE, HEAD_DIM, vec_size, blockdim_x,
                                                    blockdim_y, blockdim_z, DTypeIn, DTypeOut>;
        int num_blocks_per_sm = 0;
        int num_thrs = 128;
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, num_thrs, 0);
        size_t max_num_blks = size_t(num_blocks_per_sm) * device_prop.multiProcessorCount;
        size_t max_num_kv_chunks = max_num_blks / (num_heads / blockdim_y);
        size_t kv_chunk_size = max((seq_len + max_num_kv_chunks - 1UL) / max_num_kv_chunks,
                                   min(64UL, max(4UL, seq_len / (64 * blockdim_y / num_heads))));
        dim3 nblks = dim3(num_heads / blockdim_y, (seq_len + kv_chunk_size - 1) / kv_chunk_size);
        assert(nblks.x > 0 && nblks.y > 0);
        dim3 nthrs = dim3(blockdim_x, blockdim_y, blockdim_z);
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
        return cudaLaunchCooperativeKernel((void *)kernel, nblks, nthrs, args, 0, stream);
      })});
}

}  // namespace flashinfer

#endif
