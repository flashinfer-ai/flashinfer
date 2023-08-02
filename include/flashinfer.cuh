#ifndef FLASHINFER_CUH_
#define FLASHINFER_CUH_
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <random>

namespace flashinfer {

namespace {

template <typename T>
__device__ T warpReduceSum(T val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  val = __shfl_sync(0xffffffff, val, 0);
  return val;
}

template <typename T>
__device__ T warpReduceMax(T val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  val = __shfl_sync(0xffffffff, val, 0);
  return val;
}

}  // namespace

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
template <typename DTypeIn, typename DTypeOut, int kv_chunk_size>
__global__ void decoding_kernel(DTypeIn *__restrict__ q, DTypeIn *__restrict__ k,
                                DTypeIn *__restrict__ v, DTypeOut *__restrict__ o,
                                float *__restrict__ m_global, float *__restrict__ d_global,
                                int *__restrict__ mutex, int seq_len, int head_dim) {
  int warp_size = blockDim.x;  // 32
  int num_warps = blockDim.y;
  int lane_idx = threadIdx.x;
  int warp_idx = threadIdx.y;
  int head_idx = blockIdx.x;
  int kv_chunk_idx = blockIdx.y;
  int num_heads = gridDim.x;

  __shared__ DTypeIn q_smem[128];
  __shared__ float qk_smem[kv_chunk_size];

  float m = -MAXFLOAT;
  float d = 0;

  // load q tile
#pragma unroll
  for (int i = 0; i < head_dim / (num_warps * warp_size); ++i) {
    q_smem[i * num_warps * warp_size + warp_idx * warp_size + lane_idx] =
        q[head_idx * head_dim + i * num_warps * warp_size + warp_idx * warp_size + lane_idx];
  }
  __syncthreads();

  // load k tile and compute qk_smem
  int chunk_start = kv_chunk_idx * kv_chunk_size;
  int chunk_end = min(chunk_start + kv_chunk_size, seq_len);
  int vec_size = 4;
#pragma unroll
  for (int j = 0; j < (kv_chunk_size / num_warps); ++j) {
    int kv_idx = chunk_start + j * num_warps + warp_idx;
    if (kv_idx < chunk_end) {
      float qk = 0;
#pragma unroll
      for (int i = 0; i < head_dim / warp_size / vec_size; ++i) {
        // assert head_dim >= (warp_size * vec_size)
        uint2 q_pack4 = *(uint2 *)(&q_smem[(i * warp_size + lane_idx) * vec_size]);
        uint2 k_pack4 = *(uint2 *)(&k[(kv_idx * num_heads + head_idx) * head_dim +
                                      (i * warp_size + lane_idx) * vec_size]);
        half2 qk_pack2_x = __hmul2(*((half2 *)(&q_pack4.x)), *((half2 *)(&k_pack4.x)));
        half2 qk_pack2_y = __hmul2(*((half2 *)(&q_pack4.y)), *((half2 *)(&k_pack4.y)));
        float local_qk = qk_pack2_x.x + qk_pack2_x.y + qk_pack2_y.x + qk_pack2_y.y;
        qk += warpReduceSum(local_qk);
      }
      qk_smem[j * num_warps + warp_idx] = qk;
    }
  }
  __syncthreads();

  // compute local m
#pragma unroll
  for (int i = 0; i < ((kv_chunk_size + num_warps * warp_size - 1) / num_warps / warp_size); ++i) {
    m = max(m, (i * num_warps * warp_size + warp_idx * warp_size + lane_idx < kv_chunk_size)
                   ? qk_smem[i * num_warps * warp_size + warp_idx * warp_size + lane_idx]
                   : -MAXFLOAT);
  }

  __shared__ float m_warp_local[4];
  m_warp_local[warp_idx] = warpReduceMax(m);
  __syncthreads();
  m = 0;
#pragma unroll
  for (int i = 0; i < num_warps; ++i) {
    m = max(m, m_warp_local[i]);
  }

  // compute qk minus max exp
#pragma unroll
  for (int i = 0; i < ((kv_chunk_size + num_warps * warp_size - 1) / num_warps / warp_size); ++i) {
    if (i * num_warps * warp_size + warp_idx * warp_size + lane_idx < kv_chunk_size) {
      qk_smem[i * num_warps * warp_size + warp_idx * warp_size + lane_idx] =
          expf(qk_smem[i * num_warps * warp_size + warp_idx * warp_size + lane_idx] - m);
    }
  }
  __syncthreads();

  // compute local d
#pragma unroll
  for (int i = 0; i < ((kv_chunk_size + num_warps * warp_size - 1) / num_warps / warp_size); ++i) {
    d += (i * num_warps * warp_size + warp_idx * warp_size + lane_idx < kv_chunk_size)
             ? qk_smem[i * num_warps * warp_size + warp_idx * warp_size + lane_idx]
             : 0.f;
  }
  __shared__ float d_warp_local[4];
  d_warp_local[warp_idx] = warpReduceSum(d);
  __syncthreads();
#pragma unroll
  for (int i = 0; i < num_warps; ++i) {
    d += d_warp_local[i];
  }
  __syncthreads();

  __shared__ float o_smem[128];

  // compute local o
  o_smem[warp_idx * warp_size + lane_idx] = 0;
  for (int j = 0; j < kv_chunk_size; ++j) {
    int kv_idx = chunk_start + j;
    if (kv_idx < chunk_end) {
      o_smem[warp_idx * warp_size + lane_idx] +=
          (qk_smem[j] / d) *
          float(v[(kv_idx * num_heads + head_idx) * head_dim + warp_idx * warp_size + lane_idx]);
    }
  }

  __syncthreads();

  // critical region to sync m/d/o_smem for all ctas
  // acquire lock
  while (atomicCAS(mutex + head_idx * head_dim + warp_idx * warp_size + lane_idx, 0, 1) != 0)
    ;

  float m_prev = m_global[head_idx * head_dim + warp_idx * warp_size + lane_idx];
  float d_prev = d_global[head_idx * head_dim + warp_idx * warp_size + lane_idx];
  float m_now = max(m_prev, m);
  float d_now = d_prev * exp(m_prev - m_now) + d * exp(m - m_now);
  m_global[head_idx * head_dim + warp_idx * warp_size + lane_idx] = m_now;
  d_global[head_idx * head_dim + warp_idx * warp_size + lane_idx] = d_now;
  o[head_idx * head_dim + warp_idx * warp_size + lane_idx] =
      o[head_idx * head_dim + warp_idx * warp_size + lane_idx] *
          DTypeOut((d_prev / d_now) * exp(m_prev - m_now)) +
      DTypeOut(o_smem[warp_idx * warp_size + lane_idx] * (d / d_now) * exp(m - m_now));
  __threadfence();
  // release lock
  atomicExch(mutex + head_idx * head_dim + warp_idx * warp_size + lane_idx, 0);
}

template <typename DTypeIn, typename DTypeOut>
void decoding_dispatch(DTypeIn *q, DTypeIn *k, DTypeIn *v, DTypeOut *o, float *m_global,
                       float *d_global, int *mutex, int num_heads, int seq_len, int head_dim) {
  // skip: shape check
  int suggested_kv_chunk_size = 1;
  const int max_num_threadblocks = 1024;
  while (((seq_len + suggested_kv_chunk_size - 1) / suggested_kv_chunk_size) * num_heads >
         max_num_threadblocks) {
    suggested_kv_chunk_size *= 2;
  }
  // std::cout << "suggested_kv_chunk_size: " << suggested_kv_chunk_size << std::endl;

  dim3 nblks = dim3(num_heads, (seq_len + suggested_kv_chunk_size - 1) / suggested_kv_chunk_size);
  dim3 nthrs = dim3(32, head_dim / 32);
  if (suggested_kv_chunk_size == 1) {
    decoding_kernel<DTypeIn, DTypeOut, 1>
        <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
  } else if (suggested_kv_chunk_size == 2) {
    decoding_kernel<DTypeIn, DTypeOut, 2>
        <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
  } else if (suggested_kv_chunk_size == 4) {
    decoding_kernel<DTypeIn, DTypeOut, 4>
        <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
  } else if (suggested_kv_chunk_size == 8) {
    decoding_kernel<DTypeIn, DTypeOut, 8>
        <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
  } else if (suggested_kv_chunk_size == 16) {
    decoding_kernel<DTypeIn, DTypeOut, 16>
        <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
  } else if (suggested_kv_chunk_size == 32) {
    decoding_kernel<DTypeIn, DTypeOut, 32>
        <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
  } else if (suggested_kv_chunk_size == 64) {
    decoding_kernel<DTypeIn, DTypeOut, 64>
        <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
  } else if (suggested_kv_chunk_size == 128) {
    decoding_kernel<DTypeIn, DTypeOut, 128>
        <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
  } else if (suggested_kv_chunk_size == 256) {
    decoding_kernel<DTypeIn, DTypeOut, 256>
        <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
  } else if (suggested_kv_chunk_size == 512) {
    decoding_kernel<DTypeIn, DTypeOut, 512>
        <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
  } else if (suggested_kv_chunk_size == 1024) {
    decoding_kernel<DTypeIn, DTypeOut, 1024>
        <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
  } else if (suggested_kv_chunk_size == 2048) {
    decoding_kernel<DTypeIn, DTypeOut, 2048>
        <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
  } else if (suggested_kv_chunk_size == 4096) {
    decoding_kernel<DTypeIn, DTypeOut, 4096>
        <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
  } else if (suggested_kv_chunk_size == 8192) {
    decoding_kernel<DTypeIn, DTypeOut, 8192>
        <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
    // } else if (suggested_kv_chunk_size == 16384) {
    //   decoding_kernel<DTypeIn, DTypeOut, 16384>
    //       <<<nblks, nthrs>>>(q, k, v, o, m_global, d_global, mutex, seq_len, head_dim);
  } else {
    std::cerr << "Error: suggested_kv_chunk_size is too large: " << suggested_kv_chunk_size
              << std::endl;
  }
}

}  // namespace flashinfer

#endif