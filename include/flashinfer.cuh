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
 * \tparam vec_size A template integer indicates the vector size used in the kernel.
 * \tparam T A template type indicates the input data type
 * \param input A pointer to the start of input data
 * \param offset A integer indicates the offset of the position in ROPE(Rotary Positional
 *   Embeddings).
 * \param inv_ratio A floating point number indicate the inverse of scaling ratio used
 *   in position interpolation for ROPE(Rotary Positional Embeddings).
 */
template <size_t vec_size, typename T>
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
    vec[i] = float(vec[i]) * cos +
             float((threadIdx.x < warpSize / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin;
  }
  return vec;
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
 * \param rotary_pi_inv_ratio A floating point number indicate the inverse of scaling ratio
 *   used in PI(Position Interpolation) for ROPE (Rotary Positional Embeddings).
 * \param kv_chunk_size A integer indicates the kv-chunk size
 */
template <typename DTypeIn, typename DTypeOut, size_t head_dim, RotaryMode rotary_mode>
__global__ void SingleDecodeWithKVCacheKernel(DTypeIn *__restrict__ q, DTypeIn *__restrict__ k,
                                              DTypeIn *__restrict__ v, DTypeOut *__restrict__ o,
                                              float *__restrict__ tmp, float sm_scale,
                                              size_t seq_len, float rotary_pi_inv_ratio,
                                              size_t kv_chunk_size) {
  auto block = cooperative_groups::this_thread_block();
  auto grid = cooperative_groups::this_grid();

  constexpr size_t h_chunk_size = 4;
  constexpr size_t stages_count = 4;
  constexpr size_t vec_size = head_dim / 32;
  size_t head_idx_start = grid.block_index().x * h_chunk_size;
  size_t kv_chunk_idx = grid.block_index().y;
  size_t num_kv_chunks = grid.dim_blocks().y;
  size_t num_heads = grid.dim_blocks().x * h_chunk_size;

  __shared__ DTypeIn smem[stages_count * h_chunk_size * head_dim];

  size_t j = threadIdx.y, head_idx = head_idx_start + j;
  // apply rotary to q
  vec_t<DTypeIn, vec_size> q_vec;
  if constexpr (rotary_mode == RotaryMode::kApplyRotary ||
                rotary_mode == RotaryMode::kApplyRotaryUpdateLastK) {
    // apply rotary embedding to q matrix
    q_vec =
        apply_rotary<vec_size>(q + head_idx * head_dim, seq_len - 1, head_dim, rotary_pi_inv_ratio);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.load(q + head_idx * head_dim + threadIdx.x * vec_size);
  }

  size_t chunk_start = kv_chunk_idx * kv_chunk_size;
  DTypeIn *k_glob = k + head_idx_start * head_dim, *v_glob = v + head_idx_start * head_dim;

  // load k tiles and v tiles
  size_t kv_shared_offset[stages_count] = {
      0U, 1U * h_chunk_size * head_dim, 2U * h_chunk_size * head_dim, 3U * h_chunk_size * head_dim};

  // pipelining k/v tiles loading and m/d/o computation
  auto pipeline = cuda::make_pipeline();
  pipeline.producer_acquire();
  cuda::memcpy_async(block, smem + kv_shared_offset[0], k_glob + chunk_start * num_heads * head_dim,
                     sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
  pipeline.producer_commit();
  pipeline.producer_acquire();
  cuda::memcpy_async(block, smem + kv_shared_offset[1], v_glob + chunk_start * num_heads * head_dim,
                     sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
  pipeline.producer_commit();

  float m = -INFINITY;
  float d = 0.f;
  vec_t<float, vec_size> o_vec;
  o_vec.fill(0.f);
  float x = 0.f;

  size_t head = 0, tail = 2;

#pragma unroll 2
  for (; head < tail; ++head) {
    size_t tail_kv_idx = chunk_start + tail / 2;
    size_t copy_stage_idx = tail % stages_count;
    if (tail_kv_idx < min(chunk_start + kv_chunk_size, seq_len)) {
      pipeline.producer_acquire();
      cuda::memcpy_async(block, smem + kv_shared_offset[copy_stage_idx],
                         k_glob + tail_kv_idx * num_heads * head_dim,
                         sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
      pipeline.producer_commit();
      ++tail;
    }

    pipeline.consumer_wait();
    block.sync();
    size_t head_kv_idx = chunk_start + head / 2;
    const size_t compute_stage_idx = head % stages_count;
    if (head % 2 == 0) {
      // compute qk
      vec_t<DTypeIn, vec_size> k_vec;

      if constexpr (rotary_mode == RotaryMode::kApplyRotary) {
        // apply rotary embedding for all rows in k matrix of kv-cache
        k_vec = apply_rotary<vec_size>(smem + kv_shared_offset[compute_stage_idx] + j * head_dim,
                                       head_kv_idx, head_dim, rotary_pi_inv_ratio);
      } else if constexpr (rotary_mode == RotaryMode::kApplyRotaryUpdateLastK) {
        // apply rotary embedding for the newly appended rows in k matrix of kv-cache
        if (head_kv_idx == seq_len - 1) {
          k_vec = apply_rotary<vec_size>(smem + kv_shared_offset[compute_stage_idx] + j * head_dim,
                                         head_kv_idx, head_dim, rotary_pi_inv_ratio);
          k_vec.store(k_glob + (head_kv_idx * num_heads + j) * head_dim + threadIdx.x * vec_size);
        } else {
          k_vec.load(smem + kv_shared_offset[compute_stage_idx] + j * head_dim +
                     threadIdx.x * vec_size);
        }
      } else {
        // do not apply rotary embedding
        k_vec.load(smem + kv_shared_offset[compute_stage_idx] + j * head_dim +
                   threadIdx.x * vec_size);
      }

      x = 0.f;
#pragma unroll
      for (size_t i = 0; i < vec_size; ++i) {
        x += float(q_vec[i]) * float(k_vec[i]) * sm_scale;
      }
      x = warpReduceSum(x);

    } else {
      // update m,d,o
      float m_prev = m, d_prev = d;
      vec_t<DTypeIn, vec_size> v_vec;
      v_vec.load(smem + kv_shared_offset[compute_stage_idx] + j * head_dim +
                 threadIdx.x * vec_size);
      m = max(m, x);
      d = d * exp(m_prev - m) + exp(x - m);
#pragma unroll
      for (size_t i = 0; i < vec_size; ++i) {
        o_vec[i] = o_vec[i] * (exp(m_prev - m) * d_prev / d) + float(v_vec[i]) * (exp(x - m) / d);
      }
    }
    block.sync();
    pipeline.consumer_release();
  }

  // update tmp buffer
  o_vec.store(tmp + (kv_chunk_idx * num_heads + head_idx) * head_dim + threadIdx.x * vec_size);
  float *md_tmp = tmp + num_kv_chunks * num_heads * head_dim;
  md_tmp[(kv_chunk_idx * num_heads + head_idx) * 2] = m;
  md_tmp[(kv_chunk_idx * num_heads + head_idx) * 2 + 1] = d;
  grid.sync();

  if (kv_chunk_idx == 0) {
    float m_prev = -INFINITY, m_now;
    float d_prev = 0.f, d_now;
    vec_t<float, vec_size> o_vec_acc;
    o_vec_acc.fill(0.f);
    for (size_t batch = 0; batch < num_kv_chunks; ++batch) {
      m = md_tmp[(batch * num_heads + head_idx) * 2];
      d = md_tmp[(batch * num_heads + head_idx) * 2 + 1];
      m_now = max(m_prev, m);
      d_now = d_prev * exp(m_prev - m_now) + d * exp(m - m_now);
      o_vec.load(tmp + (batch * num_heads + head_idx) * head_dim + threadIdx.x * vec_size);
#pragma unroll
      for (size_t i = 0; i < vec_size; ++i) {
        o_vec_acc[i] = o_vec_acc[i] * (d_prev / d_now) * exp(m_prev - m_now) +
                       o_vec[i] * (d / d_now) * exp(m - m_now);
      }
      m_prev = m_now;
      d_prev = d_now;
    }
    vec_t<DTypeOut, vec_size> o_vec_global;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o_vec_global[i] = o_vec_acc[i];
    }
    o_vec_global.store(o + head_idx * head_dim + threadIdx.x * vec_size);
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
 * \param rotary_pi_inv_ratio A floating point number indicate the inverse of scaling ratio
 *   used in PI(Position Interpolation) for ROPE (Rotary Positional Embeddings).
 * \param stream The cuda stream to launch the kernel
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t SingleDecodeWithKVCache(DTypeIn *q, DTypeIn *k, DTypeIn *v, DTypeOut *o, float *tmp,
                                    size_t num_heads, size_t seq_len, size_t head_dim,
                                    RotaryMode rotary_mode = RotaryMode::kNone,
                                    float rotary_pi_inv_ratio = 1.f,
                                    cudaStream_t stream = nullptr) {
  constexpr size_t h_chunk_size = 4;
  constexpr size_t stages_count = 4;
  const float sm_scale = 1.f / std::sqrt(float(head_dim));
  assert(head_dim % h_chunk_size == 0);

  SWITCH_HEAD_DIM(
      head_dim, HEAD_DIM, {SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
        auto kernel = SingleDecodeWithKVCacheKernel<DTypeIn, DTypeOut, HEAD_DIM, ROTARY_MODE>;
        int num_blocks_per_sm = 0;
        int num_thrs = 128;
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, num_thrs, 0);
        size_t max_num_blks = size_t(num_blocks_per_sm) * device_prop.multiProcessorCount;
        size_t max_num_kv_chunks = max_num_blks / (num_heads / h_chunk_size);
        // minimum kv-chunk size is 4
        size_t kv_chunk_size = max((seq_len + max_num_kv_chunks - 1UL) / max_num_kv_chunks, 4UL);
        dim3 nblks = dim3(num_heads / h_chunk_size, (seq_len + kv_chunk_size - 1) / kv_chunk_size);
        assert(nblks.x > 0 && nblks.y > 0);
        dim3 nthrs = dim3(32, 4);
        void *args[] = {(void *)&q,
                        (void *)&k,
                        (void *)&v,
                        (void *)&o,
                        (void *)&tmp,
                        (void *)&sm_scale,
                        (void *)&seq_len,
                        (void *)&rotary_pi_inv_ratio,
                        (void *)&kv_chunk_size};
        return cudaLaunchCooperativeKernel((void *)kernel, nblks, nthrs, args, 0, stream);
      })});
}

}  // namespace flashinfer

#endif
