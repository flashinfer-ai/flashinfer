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

template <int num_warps>
__device__ __forceinline__ float crossWarpReduceSum(volatile float *warp_local) {
  return 0.f;
}

template <>
__device__ __forceinline__ float crossWarpReduceSum<1>(volatile float *warp_local) {
  return warp_local[0];
}

template <>
__device__ __forceinline__ float crossWarpReduceSum<2>(volatile float *warp_local) {
  return warp_local[0] + warp_local[1];
}

template <>
__device__ __forceinline__ float crossWarpReduceSum<4>(volatile float *warp_local) {
  return warp_local[0] + warp_local[1] + warp_local[2] + warp_local[3];
}

template <>
__device__ __forceinline__ float crossWarpReduceSum<8>(volatile float *warp_local) {
  return warp_local[0] + warp_local[1] + warp_local[2] + warp_local[3] +\
         warp_local[4] + warp_local[5] + warp_local[6] + warp_local[7];
}

// template <typename T>
// __device__ __forceinline__ T crossWarpReduceSum<1, T>(volatile T *warp_local) {
//   return warp_local[0];
// }

}  // namespace

template <int num_warps, typename DTypeIn>
__device__ __forceinline__ void update_local_states(
    float &m, float &d, float &o_local, DTypeIn *kv_smem, DTypeIn *q_smem, int head_dim,
    int compute_stage_idx, cooperative_groups::thread_block &block, int batch_size,
    const size_t *k_shared_offset, const size_t *v_shared_offset, volatile float *x_warp_local) {
#pragma unroll
  for (int j = 0; j < batch_size; ++j) {
    float x =
        float(kv_smem[k_shared_offset[compute_stage_idx] + j * head_dim + block.thread_rank()]) *
        float(q_smem[block.thread_rank()]);
    x_warp_local[threadIdx.y] = warpReduceSum(x);
    block.sync();
    x = crossWarpReduceSum<num_warps>(x_warp_local);
    float m_prev = m, d_prev = d;
    m = max(m, x);
    d = d * exp(m_prev - m) + exp(x - m);
    o_local =
        o_local * (exp(m_prev - m) * d_prev / d) +
        float(kv_smem[v_shared_offset[compute_stage_idx] + j * head_dim + block.thread_rank()]) *
            (exp(x - m) / d);
  }
}

/*!
 * \brief
 * \param q [num_heads, head_dim] The query matrix
 * \param k [seq_len, num_heads, head_dim] The key matrix
 * \param v [seq_len, num_heads, head_dim] The value matrix
 * \param o [num_heads, head_dim] The output matrix
 * \param m_global [num_heads, head_dim] The m state used in online-softmax
 * \param d_global [num_heads, head_dim] The d state used in online-softmax
 * \param mutex [num_heads, head_dim] The mutex used to sync m/d/o for all threadblocks.
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param kv_chunk_size A integer indicates the kv-chunk size
 */
template <int num_warps, typename DTypeIn, typename DTypeOut>
__global__ void decoding_kernel(DTypeIn *__restrict__ q, DTypeIn *__restrict__ k,
                                DTypeIn *__restrict__ v, DTypeOut *__restrict__ o,
                                float *__restrict__ m_global, float *__restrict__ d_global,
                                int *__restrict__ mutex, int seq_len, int head_dim,
                                int kv_chunk_size) {
  auto block = cooperative_groups::this_thread_block();

  int head_idx = blockIdx.x;
  int kv_chunk_idx = blockIdx.y;
  int num_heads = gridDim.x;
  constexpr int batch_size = 4;
  constexpr size_t stages_count = 2;

  extern __shared__ char smem[];
  DTypeIn *kv_smem = (DTypeIn *)smem;  // 2 * stages_count * batch_size * head_dim
  DTypeIn *q_smem =
      (DTypeIn *)(smem + (2 * stages_count * batch_size * head_dim) * sizeof(DTypeIn));  // head_dim
  float *x_warp_local =
      (float *)(smem + (2 * stages_count * batch_size * head_dim + head_dim) * sizeof(DTypeIn));

  float m = -INFINITY;
  float d = 0;

  // load q tile
  q_smem[block.thread_rank()] = q[head_idx * head_dim + block.thread_rank()];
  block.sync();

  // load k tile and compute qk_smem
  int chunk_start = kv_chunk_idx * kv_chunk_size;
  kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);
  DTypeIn *k_tile = k + (chunk_start * num_heads + head_idx) * head_dim,
          *v_tile = v + (chunk_start * num_heads + head_idx) * head_dim;

  // load k tiles and v tiles
  size_t k_shared_offset[stages_count] = {0, 2 * batch_size * head_dim};
  size_t v_shared_offset[stages_count] = {batch_size * head_dim, 3 * batch_size * head_dim};

  auto pipeline = cuda::make_pipeline();
  pipeline.producer_acquire();
  int effective_batch_size = min(kv_chunk_size, batch_size);
  cuda::memcpy_async(block, kv_smem + k_shared_offset[0], k_tile,
                     sizeof(DTypeIn) * effective_batch_size * head_dim, pipeline);
  cuda::memcpy_async(block, kv_smem + v_shared_offset[0], v_tile,
                     sizeof(DTypeIn) * effective_batch_size * head_dim, pipeline);
  pipeline.producer_commit();

  float o_local = 0.f;

  size_t compute_stage_idx = 0, copy_stage_idx = 1;
  int batch = 1;
  for (batch = 1; batch < (kv_chunk_size + batch_size - 1) / batch_size; ++batch) {
    effective_batch_size = min(kv_chunk_size, batch_size);
    effective_batch_size = batch_size;
    // pipeline stage 0: load k/v tiles
    pipeline.producer_acquire();
    cuda::memcpy_async(block, kv_smem + k_shared_offset[copy_stage_idx],
                       k_tile + batch * batch_size * head_dim,
                       sizeof(DTypeIn) * effective_batch_size * head_dim, pipeline);
    cuda::memcpy_async(block, kv_smem + v_shared_offset[copy_stage_idx],
                       v_tile + batch * batch_size * head_dim,
                       sizeof(DTypeIn) * effective_batch_size * head_dim, pipeline);
    pipeline.producer_commit();

    // pipeline stage 1: compute m, d, o_local
    pipeline.consumer_wait();
    update_local_states<num_warps>(m, d, o_local, kv_smem, q_smem, head_dim, compute_stage_idx,
                                   block, batch_size, k_shared_offset, v_shared_offset,
                                   x_warp_local);
    pipeline.consumer_release();
    compute_stage_idx ^= 1;
    copy_stage_idx ^= 1;
  }

  pipeline.consumer_wait();
  update_local_states<num_warps>(m, d, o_local, kv_smem, q_smem, head_dim, compute_stage_idx, block,
                                 effective_batch_size, k_shared_offset, v_shared_offset,
                                 x_warp_local);
  pipeline.consumer_release();
  block.sync();

  // critical region to sync m/d/o_smem for all ctas
  // acquire lock
  while (atomicCAS(mutex + head_idx * head_dim + block.thread_rank(), 0, 1) != 0)
    ;

  float m_prev = m_global[head_idx * head_dim + block.thread_rank()];
  float d_prev = d_global[head_idx * head_dim + block.thread_rank()];
  float m_now = max(m_prev, m);
  float d_now = d_prev * exp(m_prev - m_now) + d * exp(m - m_now);
  m_global[head_idx * head_dim + block.thread_rank()] = m_now;
  d_global[head_idx * head_dim + block.thread_rank()] = d_now;
  o[head_idx * head_dim + block.thread_rank()] =
      o[head_idx * head_dim + block.thread_rank()] *
          DTypeOut((d_prev / d_now) * exp(m_prev - m_now)) +
      DTypeOut(o_local * (d / d_now) * exp(m - m_now));
  __threadfence();
  // release lock
  atomicExch(mutex + head_idx * head_dim + block.thread_rank(), 0);
}

inline int get_heuristic_max_num_threadblocks(int seq_len) {
  if (seq_len <= 1024) {
    return 256;
  } else if (seq_len <= 4096) {
    return 512;
  } else if (seq_len <= 8192) {
    return 1024;
  } else {
    return 2048;
  }
}

template <typename DTypeIn, typename DTypeOut>
void decoding_dispatch(DTypeIn *q, DTypeIn *k, DTypeIn *v, DTypeOut *o, float *m_global,
                       float *d_global, int *mutex, int num_heads, int seq_len, int head_dim) {
  assert(head_dim % 32 == 0);
  assert(head_dim < 1024);
  int max_num_threadblocks = get_heuristic_max_num_threadblocks(seq_len);
  int suggested_kv_chunk_size = 16;
  while (((seq_len + suggested_kv_chunk_size - 1) / suggested_kv_chunk_size) * num_heads >
         max_num_threadblocks) {
    suggested_kv_chunk_size *= 2;
  }

  dim3 nblks = dim3(num_heads, (seq_len + suggested_kv_chunk_size - 1) / suggested_kv_chunk_size);
  dim3 nthrs = dim3(32, head_dim / 32);
  size_t shmem_size =
      sizeof(DTypeIn) * (4 * 4 * head_dim + head_dim) * sizeof(DTypeIn) + sizeof(float) * nthrs.y;

  if (nthrs.y == 1) {
    auto kernel = decoding_kernel<1, DTypeIn, DTypeOut>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    kernel<<<nblks, nthrs, shmem_size>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim,
                                         suggested_kv_chunk_size);
  } else if (nthrs.y == 2) {
    auto kernel = decoding_kernel<2, DTypeIn, DTypeOut>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    kernel<<<nblks, nthrs, shmem_size>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim,
                                         suggested_kv_chunk_size);

  } else if (nthrs.y == 4) {
    auto kernel = decoding_kernel<4, DTypeIn, DTypeOut>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    kernel<<<nblks, nthrs, shmem_size>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim,
                                         suggested_kv_chunk_size);

  } else if (nthrs.y == 8) {
    auto kernel = decoding_kernel<8, DTypeIn, DTypeOut>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    kernel<<<nblks, nthrs, shmem_size>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim,
                                         suggested_kv_chunk_size);

  } else {
    std::cerr << "Unsupported head_dim: " << head_dim << std::endl;
  }
}

}  // namespace flashinfer

#endif