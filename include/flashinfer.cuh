#ifndef FLASHINFER_CUH_
#define FLASHINFER_CUH_
#include <cooperative_groups/memcpy_async.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>
#include <iostream>
#include <random>

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

template <int num_warps, typename T>
__device__ __forceinline__ T crossWarpReduceSum(volatile T *warp_local) {
  T sum = 0.f;
#pragma unroll
  for (int i = 0; i < num_warps; ++i) {
    sum += warp_local[i];
  }
  return sum;
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
template <bool inplace_update, typename T>
__device__ __forceinline__ float apply_rotary(T *input, int offset, float inv_ratio) {
  auto block = cooperative_groups::this_thread_block();
  unsigned int d = block.thread_rank();
  unsigned int D = block.num_threads();
  float inv_freq = (offset * inv_ratio) * powf(1e-4, float(2 * (d % (D / 2))) / float(D));
  float cos = cosf(inv_freq);
  float sin = sinf(inv_freq);
  T permuted_input =
      (block.thread_index().y < block.dim_threads().y / 2) ? -input[d + D / 2] : input[d - D / 2];
  block.sync();
  float emb = float(input[d]) * cos + float(permuted_input) * sin;
  if (inplace_update) {
    input[d] = T(emb);
  }
  return emb;
}

/*!
 * \brief Update flashattention local states: m, d, o_local.
 * \tparam num_warps A integer indicates the number of warps used in the kernel
 * \tparam apply_rotary_to_k A boolean indicates whether to apply rotary positional embeddings to k
 * \tparam DTypeIn A template type indicates the input data type
 */
template <int num_warps, int h_chunk_size, bool apply_rotary_to_k, typename DTypeIn>
__device__ __forceinline__ void update_local_states(
    float *m, float *d, float *o_local, DTypeIn *kv_smem, float *q_local, float sm_scale,
    size_t head_dim, size_t compute_stage_idx, size_t rotary_offset, float rotary_pi_inv_ratio,
    const size_t *k_shared_offset, const size_t *v_shared_offset, volatile float *x_warp_local) {
  auto block = cooperative_groups::this_thread_block();
#pragma unroll
  for (size_t j = 0; j < h_chunk_size; ++j) {
    float k_local = 0.f;
    if (apply_rotary_to_k) {
      // do not inplace update
      k_local = apply_rotary<false>(kv_smem + k_shared_offset[compute_stage_idx] + j * head_dim,
                                    rotary_offset, rotary_pi_inv_ratio);
    } else {
      k_local =
          float(kv_smem[k_shared_offset[compute_stage_idx] + j * head_dim + block.thread_rank()]);
    }
    float x = k_local * q_local[j];
    int warp_idx = block.thread_index().y;
    x_warp_local[warp_idx] = warpReduceSum(x);
    block.sync();
    x = crossWarpReduceSum<num_warps>(x_warp_local) * sm_scale;
    float m_prev = m[j], d_prev = d[j];
    m[j] = max(m[j], x);
    d[j] = d[j] * exp(m_prev - m[j]) + exp(x - m[j]);
    o_local[j] =
        o_local[j] * (exp(m_prev - m[j]) * d_prev / d[j]) +
        float(kv_smem[v_shared_offset[compute_stage_idx] + j * head_dim + block.thread_rank()]) *
            (exp(x - m[j]) / d[j]);
  }
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
template <int num_warps, int h_chunk_size, typename DTypeIn, typename DTypeOut,
          RotaryMode rotary_mode>
__global__ void SingleDecodeWithKVCacheKernel(DTypeIn *__restrict__ q, DTypeIn *__restrict__ k,
                                              DTypeIn *__restrict__ v, DTypeOut *__restrict__ o,
                                              float *__restrict__ m_global,
                                              float *__restrict__ d_global, int *__restrict__ mutex,
                                              float sm_scale, size_t seq_len, size_t head_dim,
                                              float rotary_pi_inv_ratio, size_t kv_chunk_size) {
  auto block = cooperative_groups::this_thread_block();
  auto grid = cooperative_groups::this_grid();

  size_t head_idx_start = grid.block_index().x * h_chunk_size;
  size_t kv_chunk_idx = grid.block_index().y;
  size_t num_heads = grid.dim_blocks().x * h_chunk_size;
  constexpr size_t stages_count = 2;

  extern __shared__ char smem[];
  DTypeIn *kv_smem = (DTypeIn *)smem;  // 2 * stages_count * h_chunk_size * head_dim
  DTypeIn *q_smem = (DTypeIn *)(smem + (2 * stages_count * h_chunk_size * head_dim) *
                                           sizeof(DTypeIn));  // h_chunk_size * head_dim
  float *x_warp_local =
      (float *)(smem + (2 * stages_count * h_chunk_size * head_dim + h_chunk_size * head_dim) *
                           sizeof(DTypeIn));

  // load q tile
#pragma unroll
  for (size_t j = 0; j < h_chunk_size; ++j) {
    q_smem[j * head_dim + block.thread_rank()] =
        q[(head_idx_start + j) * head_dim + block.thread_rank()];
  }
  block.sync();
  float q_local[h_chunk_size];
#pragma unroll
  for (size_t j = 0; j < h_chunk_size; ++j) {
    if (rotary_mode == RotaryMode::kApplyRotary ||
        rotary_mode == RotaryMode::kApplyRotaryUpdateLastK) {
      q_local[j] = apply_rotary<false>(q_smem + j * head_dim, seq_len - 1, rotary_pi_inv_ratio);
    } else {
      q_local[j] = q_smem[j * head_dim + block.thread_rank()];
    }
  }
  // apply rotary to q

  size_t chunk_start = kv_chunk_idx * kv_chunk_size;
  DTypeIn *k_tile = k + (chunk_start * num_heads + head_idx_start) * head_dim,
          *v_tile = v + (chunk_start * num_heads + head_idx_start) * head_dim;

  // apply rotary to k and write back the updated k tile
  if (rotary_mode == RotaryMode::kApplyRotaryUpdateLastK) {
    // for decoding, only do this for the last row in kv-cache
    if (seq_len <= chunk_start + kv_chunk_size) {
#pragma unroll
      for (size_t j = 0; j < h_chunk_size; ++j) {
        apply_rotary<true>(k_tile + ((seq_len - 1 - chunk_start) * num_heads + j) * head_dim,
                           seq_len - 1, rotary_pi_inv_ratio);
      }
    }
  }
  kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);

  // load k tiles and v tiles
  size_t k_shared_offset[stages_count] = {0U, 2U * h_chunk_size * head_dim};
  size_t v_shared_offset[stages_count] = {h_chunk_size * head_dim, 3U * h_chunk_size * head_dim};

  // pipelining k/v tiles loading and m/d/o_local computation
  auto pipeline = cuda::make_pipeline();
  pipeline.producer_acquire();
  cuda::memcpy_async(block, kv_smem + k_shared_offset[0], k_tile,
                     sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
  cuda::memcpy_async(block, kv_smem + v_shared_offset[0], v_tile,
                     sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
  pipeline.producer_commit();

  float m[h_chunk_size];
  float d[h_chunk_size];
  float o_local[h_chunk_size];

#pragma unroll
  for (size_t j = 0; j < h_chunk_size; ++j) {
    m[j] = -INFINITY;
    d[j] = 0.f;
    o_local[j] = 0.f;
  }

  size_t compute_stage_idx = 0, copy_stage_idx = 1;
  size_t batch = 1;
  for (batch = 1; batch < kv_chunk_size; ++batch) {
    // pipeline stage 0: load k/v tiles
    pipeline.producer_acquire();
    cuda::memcpy_async(block, kv_smem + k_shared_offset[copy_stage_idx],
                       k_tile + batch * num_heads * head_dim,
                       sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
    cuda::memcpy_async(block, kv_smem + v_shared_offset[copy_stage_idx],
                       v_tile + batch * num_heads * head_dim,
                       sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
    pipeline.producer_commit();

    // pipeline stage 1: compute m, d, o_local
    pipeline.consumer_wait();
    update_local_states<num_warps, h_chunk_size, rotary_mode == RotaryMode::kApplyRotary>(
        m, d, o_local, kv_smem, q_local, sm_scale, head_dim, compute_stage_idx,
        batch + chunk_start - 1, rotary_pi_inv_ratio, k_shared_offset, v_shared_offset,
        x_warp_local);
    pipeline.consumer_release();
    compute_stage_idx ^= 1;
    copy_stage_idx ^= 1;
  }

  pipeline.consumer_wait();
  update_local_states<num_warps, h_chunk_size, rotary_mode == RotaryMode::kApplyRotary>(
      m, d, o_local, kv_smem, q_local, sm_scale, head_dim, compute_stage_idx,
      batch + chunk_start - 1, rotary_pi_inv_ratio, k_shared_offset, v_shared_offset, x_warp_local);
  pipeline.consumer_release();
  block.sync();

#pragma unroll
  for (int j_ = 0; j_ < h_chunk_size; ++j_) {
    int j = (j_ + grid.block_rank()) % h_chunk_size;
    int head_idx = head_idx_start + j;
    // critical region to sync m/d/o_smem for all ctas
    // acquire lock
    while (atomicCAS(mutex + head_idx * head_dim + block.thread_rank(), 0, 1) != 0)
      ;
    float m_prev = m_global[head_idx * head_dim + block.thread_rank()];
    float d_prev = d_global[head_idx * head_dim + block.thread_rank()];
    float m_now = max(m_prev, m[j]);
    float d_now = d_prev * exp(m_prev - m_now) + d[j] * exp(m[j] - m_now);
    m_global[head_idx * head_dim + block.thread_rank()] = m_now;
    d_global[head_idx * head_dim + block.thread_rank()] = d_now;
    o[head_idx * head_dim + block.thread_rank()] =
        DTypeOut(float(o[head_idx * head_dim + block.thread_rank()]) * (d_prev / d_now) *
                     exp(m_prev - m_now) +
                 o_local[j] * (d[j] / d_now) * exp(m[j] - m_now));
    __threadfence();
    // release lock
    atomicExch(mutex + head_idx * head_dim + block.thread_rank(), 0);
  }
}

#define SWITCH_NUM_WARPS(num_warps, NUM_WARPS, ...)                     \
  switch (num_warps) {                                                  \
    case 1: {                                                           \
      constexpr int NUM_WARPS = 1;                                      \
      __VA_ARGS__                                                       \
      break;                                                            \
    }                                                                   \
    case 2: {                                                           \
      constexpr int NUM_WARPS = 2;                                      \
      __VA_ARGS__                                                       \
      break;                                                            \
    }                                                                   \
    case 4: {                                                           \
      constexpr int NUM_WARPS = 4;                                      \
      __VA_ARGS__                                                       \
      break;                                                            \
    }                                                                   \
    case 8: {                                                           \
      constexpr int NUM_WARPS = 8;                                      \
      __VA_ARGS__                                                       \
      break;                                                            \
    }                                                                   \
    default: {                                                          \
      std::cerr << "Unsupported num_warps: " << num_warps << std::endl; \
      abort();                                                          \
    }                                                                   \
  }

#define SWITCH_H_CHUNK_SIZE(h_chunk_size, H_CHUNK_SIZE, ...)                  \
  switch (h_chunk_size) {                                                     \
    case 1: {                                                                 \
      constexpr int H_CHUNK_SIZE = 1;                                         \
      __VA_ARGS__                                                             \
      break;                                                                  \
    }                                                                         \
    case 2: {                                                                 \
      constexpr int H_CHUNK_SIZE = 2;                                         \
      __VA_ARGS__                                                             \
      break;                                                                  \
    }                                                                         \
    case 4: {                                                                 \
      constexpr int H_CHUNK_SIZE = 4;                                         \
      __VA_ARGS__                                                             \
      break;                                                                  \
    }                                                                         \
    case 8: {                                                                 \
      constexpr int H_CHUNK_SIZE = 8;                                         \
      __VA_ARGS__                                                             \
      break;                                                                  \
    }                                                                         \
    default: {                                                                \
      std::cerr << "Unsupported h_chunk_size: " << h_chunk_size << std::endl; \
      abort();                                                                \
    }                                                                         \
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
  if (seq_len <= 512) {
    return 128;
  } else if (seq_len <= 2048) {
    return 256;
  } else if (seq_len <= 16384) {
    return 512;
  } else {
    return 1024;
  }
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
                             float *d_global, int *mutex, int num_heads, int seq_len, int head_dim,
                             RotaryMode rotary_mode = RotaryMode::kNone,
                             float rotary_pi_inv_ratio = 1.f, cudaStream_t stream = nullptr) {
  assert(head_dim % 32 == 0);
  assert(head_dim <= 1024);
  float sm_scale = 1.f / sqrtf(float(head_dim));
  int max_num_threadblocks = get_heuristic_max_num_threadblocks(seq_len);
  int h_chunk_size = (seq_len <= 64) ? 1 : ((seq_len <= 2048) ? 2 : 4);
  int suggested_kv_chunk_size = 4;
  while (((seq_len + suggested_kv_chunk_size - 1) / suggested_kv_chunk_size) * num_heads /
             h_chunk_size >
         max_num_threadblocks) {
    suggested_kv_chunk_size *= 2;
  }
  dim3 nblks = dim3(num_heads / h_chunk_size,
                    (seq_len + suggested_kv_chunk_size - 1) / suggested_kv_chunk_size);
  dim3 nthrs = dim3(32, head_dim / 32);
  size_t shmem_size = sizeof(DTypeIn) * (4 * h_chunk_size * head_dim + h_chunk_size * head_dim) +
                      sizeof(float) * nthrs.y;

  SWITCH_NUM_WARPS(
      nthrs.y, NUM_WARPS,
      {SWITCH_ROTARY_MODE(
          rotary_mode, ROTARY_MODE, {SWITCH_H_CHUNK_SIZE(h_chunk_size, H_CHUNK_SIZE, {
            auto kernel = SingleDecodeWithKVCacheKernel<NUM_WARPS, H_CHUNK_SIZE, DTypeIn, DTypeOut,
                                                        ROTARY_MODE>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            kernel<<<nblks, nthrs, shmem_size, stream>>>(
                q, k, v, o, m_global, d_global, mutex, sm_scale, seq_len, head_dim,
                rotary_pi_inv_ratio, suggested_kv_chunk_size);
          })})});
}

}  // namespace flashinfer

#endif
