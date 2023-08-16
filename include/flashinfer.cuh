#ifndef FLASHINFER_CUH_
#define FLASHINFER_CUH_
#include <cooperative_groups/memcpy_async.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>
#include <iostream>
#include <random>

#include "vec_dtypes.cuh"

namespace flashinfer {

namespace {

template <typename T>
__device__ T warpReduceSum(T val, unsigned int mask = 0xffffffff) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(mask, val, offset);
  }
  val = __shfl_sync(mask, val, 0);
  return val;
}

}  // namespace

/*!
 * \brief An enumeration class that defines different modes for applying ROPE(Rotary Positional
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
 * \brief Apply ROPE(Rotary Positional Embeddings) to input[0: head_dim], return
 *   thread-local result.
 * \tparam T A template type indicates the input data type
 * \tparam inplace_update A boolean indicates whether to update the input data inplace
 * \param input A pointer to the start of input data
 * \param offset A integer indicates the offset of the position in ROPE(Rotary Positional
 *   Embeddings).
 * \param inv_ratio A floating point number indicate the inverse of scaling ratio used
 *   in position interpolation for ROPE(Rotary Positional Embeddings).
 */
template <bool inplace_update, size_t vec_size, typename T>
__device__ __forceinline__ vec_t<T, vec_size> apply_rotary(T *input, size_t offset, size_t head_dim,
                                                           float inv_ratio) {
  vec_t<T, vec_size> permuted_vec, vec;
  vec.load(input + threadIdx.x * vec_size);
  permuted_vec.load(input + ((threadIdx.x < warpSize / 2) ? threadIdx.x * vec_size + head_dim / 2
                                                          : threadIdx.x * vec_size - head_dim / 2));

#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    size_t d = threadIdx.x * vec_size + i;
    float inv_freq =
        (offset * inv_ratio) * powf(1e-4, float(2 * (d % (head_dim / 2))) / float(head_dim));
    float cos = cosf(inv_freq);
    float sin = sinf(inv_freq);
    vec[i] =
        float(vec[i]) * cos +
        ((threadIdx.x < warpSize / 2) ? -float(permuted_vec[i]) : float(permuted_vec[i])) * sin;
  }
  if (inplace_update) {
    vec.store(input + threadIdx.x * vec_size);
  }
  return vec;
}

/*!
 * \brief Update flashattention local states: m, d, o.
 * \tparam apply_rotary_to_k A boolean indicates whether to apply rotary positional embeddings to k
 * \tparam DTypeIn A template type indicates the input data type
 */
template <bool compute_qk, size_t vec_size, bool apply_rotary_to_k, typename DTypeIn>
__device__ __forceinline__ void update_local_states(
    float &m, float &d, vec_t<float, vec_size> &o_vec, float &x, DTypeIn *smem,
    vec_t<DTypeIn, vec_size> &q_vec, float sm_scale, size_t head_dim, size_t compute_stage_idx,
    size_t rotary_offset, float rotary_pi_inv_ratio, const size_t *kv_shared_offset) {
  auto block = cooperative_groups::this_thread_block();

  if (compute_qk) {
    size_t j = threadIdx.y;
    vec_t<DTypeIn, vec_size> k_vec;

    if constexpr (apply_rotary_to_k) {
      k_vec =
          apply_rotary<false, vec_size>(smem + kv_shared_offset[compute_stage_idx] + j * head_dim,
                                        rotary_offset, head_dim, rotary_pi_inv_ratio);
    } else {
      k_vec.load(smem + kv_shared_offset[compute_stage_idx] + j * head_dim +
                 threadIdx.x * vec_size);
    }

    x = 0.f;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      x += float(q_vec[i]) * float(k_vec[i]);
    }
    x = warpReduceSum(x) * sm_scale;
  } else {
    size_t j = threadIdx.y;
    float m_prev = m, d_prev = d;
    vec_t<DTypeIn, vec_size> v_vec;
    v_vec.load(smem + kv_shared_offset[compute_stage_idx] + j * head_dim + threadIdx.x * vec_size);
    m = max(m, x);
    d = d * exp(m_prev - m) + exp(x - m);
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o_vec[i] = o_vec[i] * (exp(m_prev - m) * d_prev / d) + float(v_vec[i]) * (exp(x - m) / d);
    }
  }
  block.sync();
}

/*!
 * \brief FlashAttention decoding cuda kernel with kv-cache for a single sequence, fused with
 *   rotary positional embeddings (if applicable).
 * \tparam num_warps A integer indicates the number of warps used in the kernel
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam rotary_mode The rotary mode used in the kernel.
 * \param q [num_heads, head_dim] The query matrix
 * \param k [seq_len, num_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_heads, head_dim] The value matrix in kv-cache
 * \param o [num_heads, head_dim] The output matrix
 * \param m_global [num_heads, head_dim] The m state used in online-softmax
 * \param d_global [num_heads, head_dim] The d state used in online-softmax
 * \param mutex [num_heads, head_dim] The mutex used to sync m/d/o for all threadblocks
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param rotary_pi_inv_ratio A floating point number indicate the inverse of scaling ratio
 *   used in PI(Position Interpolation) for ROPE (Rotary Positional Embeddings).
 * \param kv_chunk_size A integer indicates the kv-chunk size
 */
template <typename DTypeIn, typename DTypeOut, size_t vec_size, RotaryMode rotary_mode>
__global__ void SingleDecodeWithKVCacheKernel(DTypeIn *__restrict__ q, DTypeIn *__restrict__ k,
                                              DTypeIn *__restrict__ v, DTypeOut *__restrict__ o,
                                              float *__restrict__ m_global,
                                              float *__restrict__ d_global, int *__restrict__ mutex,
                                              float sm_scale, size_t seq_len, size_t head_dim,
                                              float rotary_pi_inv_ratio, size_t kv_chunk_size) {
  auto block = cooperative_groups::this_thread_block();
  auto grid = cooperative_groups::this_grid();

  constexpr size_t h_chunk_size = 4;
  constexpr size_t stages_count = 4;
  size_t head_idx_start = grid.block_index().x * h_chunk_size;
  size_t kv_chunk_idx = grid.block_index().y;
  size_t num_heads = grid.dim_blocks().x * h_chunk_size;

  extern __shared__ DTypeIn smem[];

  // load q tile
  vec_t<DTypeIn, vec_size>::memcpy(
      smem + threadIdx.y * head_dim + threadIdx.x * vec_size,
      q + head_idx_start * head_dim + threadIdx.y * head_dim + threadIdx.x * vec_size);

  // apply rotary to q
  vec_t<DTypeIn, vec_size> q_vec;
  size_t j = threadIdx.y;
  if constexpr (rotary_mode == RotaryMode::kApplyRotary ||
      rotary_mode == RotaryMode::kApplyRotaryUpdateLastK) {
    q_vec = apply_rotary<false, vec_size>(smem + j * head_dim, seq_len - 1, head_dim,
                                          rotary_pi_inv_ratio);
  } else {
    q_vec.load(smem + j * head_dim + threadIdx.x * vec_size);
  }

  size_t chunk_start = kv_chunk_idx * kv_chunk_size;
  DTypeIn *k_glob = k + (chunk_start * num_heads + head_idx_start) * head_dim,
          *v_glob = v + (chunk_start * num_heads + head_idx_start) * head_dim;

  // apply rotary to k and write back the updated k tile
  if constexpr (rotary_mode == RotaryMode::kApplyRotaryUpdateLastK) {
    // for decoding, only do this for the last row in kv-cache
    if (seq_len <= chunk_start + kv_chunk_size) {
      size_t j = threadIdx.y;
      apply_rotary<true, vec_size>(
          k_glob + ((seq_len - 1 - chunk_start) * num_heads + j) * head_dim, seq_len - 1, head_dim,
          rotary_pi_inv_ratio);
    }
  }
  kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);

  // load k tiles and v tiles
  size_t kv_shared_offset[stages_count] = {
      0U, 1U * h_chunk_size * head_dim, 2U * h_chunk_size * head_dim, 3U * h_chunk_size * head_dim};

  // pipelining k/v tiles loading and m/d/o computation
  auto pipeline = cuda::make_pipeline();
  pipeline.producer_acquire();
  cuda::memcpy_async(block, smem + kv_shared_offset[0], k_glob,
                     sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
  pipeline.producer_commit();
  pipeline.producer_acquire();
  cuda::memcpy_async(block, smem + kv_shared_offset[1], v_glob,
                     sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
  pipeline.producer_commit();

  float m = -INFINITY;
  float d = 0.f;
  vec_t<float, vec_size> o_vec;
  o_vec.fill(0.f);
  float x;

  size_t compute_stage_idx = 0, copy_stage_idx = 2;
  size_t batch = 1;
#pragma unroll 2
  for (batch = 1; batch < kv_chunk_size; ++batch) {
    // pipeline stage 0: load k/v tiles
    pipeline.producer_acquire();
    cuda::memcpy_async(block, smem + kv_shared_offset[copy_stage_idx],
                       k_glob + batch * num_heads * head_dim,
                       sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
    pipeline.producer_commit();
    copy_stage_idx = (copy_stage_idx + 1) % stages_count;
    pipeline.producer_acquire();
    cuda::memcpy_async(block, smem + kv_shared_offset[copy_stage_idx],
                       v_glob + batch * num_heads * head_dim,
                       sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
    pipeline.producer_commit();
    copy_stage_idx = (copy_stage_idx + 1) % stages_count;

    // pipeline stage 1: compute m, d, o
    pipeline.consumer_wait();
    update_local_states<true, vec_size, rotary_mode == RotaryMode::kApplyRotary>(
        m, d, o_vec, x, smem, q_vec, sm_scale, head_dim, compute_stage_idx, batch + chunk_start - 1,
        rotary_pi_inv_ratio, kv_shared_offset);
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;
    pipeline.consumer_wait();
    update_local_states<false, vec_size, rotary_mode == RotaryMode::kApplyRotary>(
        m, d, o_vec, x, smem, q_vec, sm_scale, head_dim, compute_stage_idx, batch + chunk_start - 1,
        rotary_pi_inv_ratio, kv_shared_offset);
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;
  }

  pipeline.consumer_wait();
  update_local_states<true, vec_size, rotary_mode == RotaryMode::kApplyRotary>(
      m, d, o_vec, x, smem, q_vec, sm_scale, head_dim, compute_stage_idx, batch + chunk_start - 1,
      rotary_pi_inv_ratio, kv_shared_offset);
  pipeline.consumer_release();
  compute_stage_idx = (compute_stage_idx + 1) % stages_count;
  pipeline.consumer_wait();
  update_local_states<false, vec_size, rotary_mode == RotaryMode::kApplyRotary>(
      m, d, o_vec, x, smem, q_vec, sm_scale, head_dim, compute_stage_idx, batch + chunk_start - 1,
      rotary_pi_inv_ratio, kv_shared_offset);
  pipeline.consumer_release();
  block.sync();

  // size_t j = threadIdx.y;
  int head_idx = head_idx_start + j;
  // critical region to sync m/d/o_smem for all ctas
  // acquire lock
  while (atomicCAS(mutex + head_idx * 32 + threadIdx.x, 0, 1) != 0)
    ;
  float m_prev = m_global[head_idx * 32 + threadIdx.x];
  float d_prev = d_global[head_idx * 32 + threadIdx.x];
  float m_now = max(m_prev, m);
  float d_now = d_prev * exp(m_prev - m_now) + d * exp(m - m_now);
  m_global[head_idx * 32 + threadIdx.x] = m_now;
  d_global[head_idx * 32 + threadIdx.x] = d_now;
  vec_t<DTypeOut, vec_size> o_vec_global;
  o_vec_global.load(o + head_idx * head_dim + threadIdx.x * vec_size);
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    o_vec_global[i] = DTypeOut(float(o_vec_global[i]) * (d_prev / d_now) * exp(m_prev - m_now) +
                               o_vec[i] * (d / d_now) * exp(m - m_now));
  }
  o_vec_global.store(o + head_idx * head_dim + threadIdx.x * vec_size);
  __threadfence();
  // release lock
  atomicExch(mutex + head_idx * 32 + threadIdx.x, 0);
}

#define SWITCH_VEC_SIZE(vec_size, VEC_SIZE, ...)                      \
  switch (vec_size) {                                                 \
    case 1: {                                                         \
      constexpr size_t VEC_SIZE = 1;                                  \
      __VA_ARGS__                                                     \
      break;                                                          \
    }                                                                 \
    case 2: {                                                         \
      constexpr size_t VEC_SIZE = 2;                                  \
      __VA_ARGS__                                                     \
      break;                                                          \
    }                                                                 \
    case 4: {                                                         \
      constexpr size_t VEC_SIZE = 4;                                  \
      __VA_ARGS__                                                     \
      break;                                                          \
    }                                                                 \
    case 8: {                                                         \
      constexpr size_t VEC_SIZE = 8;                                  \
      __VA_ARGS__                                                     \
      break;                                                          \
    }                                                                 \
    default: {                                                        \
      std::cerr << "Unsupported vec_size: " << vec_size << std::endl; \
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

inline int get_heuristic_max_num_threadblocks(int seq_len) {
  if (seq_len <= 128) {
    return 64;
  } else if (seq_len <= 2048) {
    return 128;
  }
  return 256;
}

/*!
 * \brief FlashAttention decoding with kv-cache for a single sequence.
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \param q [num_heads, head_dim] The query matrix
 * \param k [seq_len, num_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_heads, head_dim] The value matrix in kv-cache
 * \param o [num_heads, head_dim] The output matrix
 * \param m_global [num_heads, head_dim] The m state used in online-softmax
 * \param d_global [num_heads, head_dim] The d state used in online-softmax
 * \param mutex [num_heads, head_dim] The mutex used to sync m/d/o for all threadblocks
 * \param num_heads A integer indicates the number of heads
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param rotary_mode The rotary mode used in the kernel.
 * \param rotary_pi_inv_ratio A floating point number indicate the inverse of scaling ratio
 *   used in PI(Position Interpolation) for ROPE (Rotary Positional Embeddings).
 * \param stream The cuda stream to launch the kernel
 */
template <typename DTypeIn, typename DTypeOut>
void SingleDecodeWithKVCache(DTypeIn *q, DTypeIn *k, DTypeIn *v, DTypeOut *o, float *m_global,
                             float *d_global, int *mutex, size_t num_heads, size_t seq_len,
                             size_t head_dim, RotaryMode rotary_mode = RotaryMode::kNone,
                             float rotary_pi_inv_ratio = 1.f, cudaStream_t stream = nullptr) {
  constexpr size_t h_chunk_size = 4;
  constexpr size_t stages_count = 4;
  const float sm_scale = 1.f / sqrtf(float(head_dim));
  assert(head_dim % h_chunk_size == 0);

  const size_t suggested_kv_chunk_size = std::max<size_t>(
      4U, seq_len * num_heads / h_chunk_size / get_heuristic_max_num_threadblocks(seq_len));
  const dim3 nblks = dim3(num_heads / h_chunk_size,
                          (seq_len + suggested_kv_chunk_size - 1) / suggested_kv_chunk_size);
  const dim3 nthrs = dim3(32, 4);
  const size_t shmem_size = sizeof(DTypeIn) * (stages_count * h_chunk_size * head_dim);
  const size_t vec_size = head_dim / 32;

  SWITCH_VEC_SIZE(vec_size, VEC_SIZE, {SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
                    auto kernel =
                        SingleDecodeWithKVCacheKernel<DTypeIn, DTypeOut, VEC_SIZE, ROTARY_MODE>;
                    kernel<<<nblks, nthrs, shmem_size, stream>>>(
                        q, k, v, o, m_global, d_global, mutex, sm_scale, seq_len, head_dim,
                        rotary_pi_inv_ratio, suggested_kv_chunk_size);
                  })});
}

}  // namespace flashinfer

#endif
