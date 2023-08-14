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
template <bool compute_qk, int num_warps, int h_chunk_size, bool apply_rotary_to_k,
          typename DTypeIn>
__device__ __forceinline__ void update_local_states(
    float &m, float &d, float4 &o_local, float &x, DTypeIn *kv_smem, DTypeIn *q_smem, float sm_scale,
    size_t head_dim, size_t compute_stage_idx, size_t rotary_offset, float rotary_pi_inv_ratio,
    const size_t *kv_shared_offset) {
  auto block = cooperative_groups::this_thread_block();

  if (compute_qk) {
    size_t j = threadIdx.y;
    uint2 k_local_pack4 = *(uint2 *)(kv_smem + kv_shared_offset[compute_stage_idx] + j * head_dim +
                                      threadIdx.x * sizeof(uint2) / sizeof(DTypeIn));
    uint2 q_local_pack4 = *(uint2 *)(q_smem + j * head_dim + threadIdx.x * sizeof(uint2) / sizeof(DTypeIn));
    x = float(((half2 *)(&q_local_pack4.x))->x) * float(((half2 *)(&k_local_pack4.x))->x) +\
        float(((half2 *)(&q_local_pack4.x))->y) * float(((half2 *)(&k_local_pack4.x))->y) +\
        float(((half2 *)(&q_local_pack4.y))->x) * float(((half2 *)(&k_local_pack4.y))->x) +\
        float(((half2 *)(&q_local_pack4.y))->y) * float(((half2 *)(&k_local_pack4.y))->y);
    x = warpReduceSum(x) * sm_scale;
    block.sync();
  } else {
    size_t j = threadIdx.y;
    float m_prev = m, d_prev = d;
    uint2 v_local_pack4 = *(uint2 *)(kv_smem + kv_shared_offset[compute_stage_idx] + j * head_dim +
                                     threadIdx.x * sizeof(uint2) / sizeof(DTypeIn));
    m = max(m, x);
    d = d * exp(m_prev - m) + exp(x - m);
    o_local.x = o_local.x * (exp(m_prev - m) * d_prev / d) + float(((half2 *)(&v_local_pack4.x))->x) * (exp(x - m) / d);
    o_local.y = o_local.y * (exp(m_prev - m) * d_prev / d) + float(((half2 *)(&v_local_pack4.x))->y) * (exp(x - m) / d);
    o_local.z = o_local.z * (exp(m_prev - m) * d_prev / d) + float(((half2 *)(&v_local_pack4.y))->x) * (exp(x - m) / d);
    o_local.w = o_local.w * (exp(m_prev - m) * d_prev / d) + float(((half2 *)(&v_local_pack4.y))->y) * (exp(x - m) / d);
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
  constexpr size_t stages_count = 4;

  extern __shared__ char smem[];
  DTypeIn *kv_smem = (DTypeIn *)smem;  // stages_count * h_chunk_size * head_dim
  DTypeIn *q_smem = (DTypeIn *)(smem + (stages_count * h_chunk_size * head_dim) *
                                           sizeof(DTypeIn));  // h_chunk_size * head_dim

  // load q tile
#pragma unroll
  for (size_t j = 0; j < h_chunk_size; ++j) {
    q_smem[j * head_dim + block.thread_rank()] =
        q[(head_idx_start + j) * head_dim + block.thread_rank()];
  }
  block.sync();
#pragma unroll
  for (size_t j = 0; j < h_chunk_size; ++j) {
    if (rotary_mode == RotaryMode::kApplyRotary ||
        rotary_mode == RotaryMode::kApplyRotaryUpdateLastK) {
      apply_rotary<true>(q_smem + j * head_dim, seq_len - 1, rotary_pi_inv_ratio);
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
  size_t kv_shared_offset[stages_count] = {
      0U, 1U * h_chunk_size * head_dim, 2U * h_chunk_size * head_dim, 3U * h_chunk_size * head_dim};

  // pipelining k/v tiles loading and m/d/o_local computation
  auto pipeline = cuda::make_pipeline();
  pipeline.producer_acquire();
  cuda::memcpy_async(block, kv_smem + kv_shared_offset[0], k_tile,
                     sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
  pipeline.producer_commit();
  pipeline.producer_acquire();
  cuda::memcpy_async(block, kv_smem + kv_shared_offset[1], v_tile,
                     sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
  pipeline.producer_commit();

  float m = -INFINITY;
  float d = 0.f;
  float4 o_local = make_float4(0.f, 0.f, 0.f, 0.f);
  float x;

  size_t compute_stage_idx = 0, copy_stage_idx = 2;
  size_t batch = 1;
  for (batch = 1; batch < kv_chunk_size; ++batch) {
    // pipeline stage 0: load k/v tiles
    pipeline.producer_acquire();
    cuda::memcpy_async(block, kv_smem + kv_shared_offset[copy_stage_idx],
                       k_tile + batch * num_heads * head_dim,
                       sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
    pipeline.producer_commit();
    copy_stage_idx = (copy_stage_idx + 1) % stages_count;
    pipeline.producer_acquire();
    cuda::memcpy_async(block, kv_smem + kv_shared_offset[copy_stage_idx],
                       v_tile + batch * num_heads * head_dim,
                       sizeof(DTypeIn) * h_chunk_size * head_dim, pipeline);
    pipeline.producer_commit();
    copy_stage_idx = (copy_stage_idx + 1) % stages_count;

    // pipeline stage 1: compute m, d, o_local
    pipeline.consumer_wait();
    update_local_states<true, num_warps, h_chunk_size, rotary_mode == RotaryMode::kApplyRotary>(
        m, d, o_local, x, kv_smem, q_smem, sm_scale, head_dim, compute_stage_idx,
        batch + chunk_start - 1, rotary_pi_inv_ratio, kv_shared_offset);
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;
    pipeline.consumer_wait();
    update_local_states<false, num_warps, h_chunk_size, rotary_mode == RotaryMode::kApplyRotary>(
        m, d, o_local, x, kv_smem, q_smem, sm_scale, head_dim, compute_stage_idx,
        batch + chunk_start - 1, rotary_pi_inv_ratio, kv_shared_offset);
    pipeline.consumer_release();
    compute_stage_idx = (compute_stage_idx + 1) % stages_count;
  }

  pipeline.consumer_wait();
  update_local_states<true, num_warps, h_chunk_size, rotary_mode == RotaryMode::kApplyRotary>(
      m, d, o_local, x, kv_smem, q_smem, sm_scale, head_dim, compute_stage_idx,
      batch + chunk_start - 1, rotary_pi_inv_ratio, kv_shared_offset);
  pipeline.consumer_release();
  compute_stage_idx = (compute_stage_idx + 1) % stages_count;
  pipeline.consumer_wait();
  update_local_states<false, num_warps, h_chunk_size, rotary_mode == RotaryMode::kApplyRotary>(
      m, d, o_local, x, kv_smem, q_smem, sm_scale, head_dim, compute_stage_idx,
      batch + chunk_start - 1, rotary_pi_inv_ratio, kv_shared_offset);
  pipeline.consumer_release();
  block.sync();

#pragma unroll
  size_t j = threadIdx.y;
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
  uint2 o_pack4 = *(uint2 *)(o + head_idx * head_dim + threadIdx.x * sizeof(uint2) / sizeof(DTypeOut));
  ((half2 *)(&o_pack4.x))->x = DTypeOut((float)((half2 *)(&o_pack4.x))->x * (d_prev / d_now) * exp(m_prev - m_now) + o_local.x * (d / d_now) * exp(m - m_now));
  ((half2 *)(&o_pack4.x))->y = DTypeOut((float)((half2 *)(&o_pack4.x))->y * (d_prev / d_now) * exp(m_prev - m_now) + o_local.y * (d / d_now) * exp(m - m_now));
  ((half2 *)(&o_pack4.y))->x = DTypeOut((float)((half2 *)(&o_pack4.y))->x * (d_prev / d_now) * exp(m_prev - m_now) + o_local.z * (d / d_now) * exp(m - m_now));
  ((half2 *)(&o_pack4.y))->y = DTypeOut((float)((half2 *)(&o_pack4.y))->y * (d_prev / d_now) * exp(m_prev - m_now) + o_local.w * (d / d_now) * exp(m - m_now));
  *(uint2 *)(o + head_idx * head_dim + threadIdx.x * sizeof(uint2) / sizeof(DTypeOut)) = o_pack4;
  __threadfence();
  // release lock
  atomicExch(mutex + head_idx * 32 + threadIdx.x, 0);
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
  } else if (seq_len <= 4096) {
    return 512;
  } else if (seq_len <= 8192) {
    return 1024;
  } else {
    return 2048;
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
  int h_chunk_size = 4;
  int suggested_kv_chunk_size = 8;
  while (((seq_len + suggested_kv_chunk_size - 1) / suggested_kv_chunk_size) * num_heads /
             h_chunk_size >
         max_num_threadblocks) {
    suggested_kv_chunk_size *= 2;
  }
  dim3 nblks = dim3(num_heads / h_chunk_size,
                    (seq_len + suggested_kv_chunk_size - 1) / suggested_kv_chunk_size);
  dim3 nthrs = dim3(32, head_dim / 32);
  size_t shmem_size = sizeof(DTypeIn) * (4 * h_chunk_size * head_dim + h_chunk_size * head_dim);

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
