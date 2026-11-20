/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

namespace {

__device__ __forceinline__ float load_scalar(const void* base, long long index, int dtype) {
  if (dtype == 0) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(base)[index]);
  }
  if (dtype == 1) {
    return __half2float(reinterpret_cast<const __half*>(base)[index]);
  }
  return static_cast<float>(reinterpret_cast<const __nv_fp8_e4m3*>(base)[index]);
}

__device__ __forceinline__ void store_scalar(void* base, long long index, int dtype, float value) {
  if (dtype == 0) {
    reinterpret_cast<__nv_bfloat16*>(base)[index] = __float2bfloat16_rn(value);
  } else if (dtype == 1) {
    reinterpret_cast<__half*>(base)[index] = __float2half_rn(value);
  } else {
    reinterpret_cast<__nv_fp8_e4m3*>(base)[index] = __nv_fp8_e4m3(value);
  }
}

__device__ __forceinline__ float e2m1_value(unsigned code) {
  constexpr float levels[16] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
     -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
  return levels[code & 15u];
}

__device__ __forceinline__ unsigned nearest_e2m1(float value) {
  constexpr float levels[16] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
     -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
  unsigned best = 0;
  float best_error = fabsf(value - levels[0]);
  #pragma unroll
  for (unsigned code = 1; code < 16; ++code) {
    const float error = fabsf(value - levels[code]);
    if (error < best_error) {
      best = code;
      best_error = error;
    }
  }
  return best;
}

__device__ __forceinline__ long long output_scale_swizzled_index(
    int row, int column, int rounded_columns) {
  const int row_block = row >> 7;
  const int row_within = row & 127;
  const int row_outer = row_within >> 5;
  const int row_lane = row_within & 31;
  const int column_group = column >> 2;
  const int column_inner = column & 3;
  return (((((static_cast<long long>(row_block) * (rounded_columns >> 2) + column_group) * 32
             + row_lane) * 4 + row_outer) * 4) + column_inner);
}

__device__ __forceinline__ long long kv_index(
    int page, int token, int head, int dim, int layout,
    long long s0, long long s1, long long s2, long long s3) {
  if (layout == 0) {
    return static_cast<long long>(page) * s0 + static_cast<long long>(head) * s1
         + static_cast<long long>(token) * s2 + static_cast<long long>(dim) * s3;
  }
  return static_cast<long long>(page) * s0 + static_cast<long long>(token) * s1
       + static_cast<long long>(head) * s2 + static_cast<long long>(dim) * s3;
}

__device__ __forceinline__ long long v_scale_swizzled_index(
    int page, int token, int head, int block, int layout,
    int page_size, int num_heads, int scale_dim, long long page_stride) {
  const int token_group = token >> 2;
  const int token_inner = token & 3;
  const int scale_groups = scale_dim >> 2;
  const int scale_outer = block / scale_groups;
  const int scale_inner = block % scale_groups;
  long long within;
  if (layout == 0) {
    within = (((((static_cast<long long>(head) * (page_size >> 2) + token_group) * 4
                  + scale_outer) * scale_groups + scale_inner) * 4) + token_inner);
  } else {
    within = (((((static_cast<long long>(token_group) * 4 + scale_outer) * num_heads
                  + head) * scale_groups + scale_inner) * 4) + token_inner);
  }
  return static_cast<long long>(page) * page_stride + within;
}

__device__ __forceinline__ float load_nvfp4(
    const void* packed, const void* scales,
    int page, int token, int head, int dim, int layout, bool value_side,
    int page_size, int num_heads, int head_dim,
    long long p0, long long p1, long long p2, long long p3,
    long long sf0, long long sf1, long long sf2, long long sf3) {
  const long long packed_at = kv_index(page, token, head, dim >> 1, layout, p0, p1, p2, p3);
  const unsigned byte = reinterpret_cast<const unsigned char*>(packed)[packed_at];
  const unsigned code = (dim & 1) ? (byte >> 4) : (byte & 15u);
  const int block = dim >> 4;
  long long sf_at;
  if (value_side) {
    sf_at = v_scale_swizzled_index(
        page, token, head, block, layout, page_size, num_heads, head_dim >> 4, sf0);
  } else {
    sf_at = kv_index(page, token, head, block, layout, sf0, sf1, sf2, sf3);
  }
  const float scale = static_cast<float>(reinterpret_cast<const __nv_fp8_e4m3*>(scales)[sf_at]);
  return e2m1_value(code) * scale;
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
  const int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset != 0; offset >>= 1) {
    if (tid < offset) scratch[tid] += scratch[tid + offset];
    __syncthreads();
  }
  return scratch[0];
}

}  // namespace

extern "C" __global__ __launch_bounds__(256) void kernel_cake_fmha_compat_v1(
    const void* q,
    const void* k,
    const void* v,
    const void* k_scales,
    const void* v_scales,
    void* o,
    void* o_scales,
    float* lse,
    const int* page_table_k,
    const int* page_table_v,
    const int* q_indptr,
    const int* seq_lens_kv,
    const float* sinks,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int kv_layout,
    int q_dtype,
    int kv_dtype,
    int o_dtype,
    int causal,
    int window_left,
    int enable_sink,
    int return_lse,
    float q_scale,
    float k_scale,
    float v_scale,
    float o_scale,
    float sm_scale,
    float o_sf_scale,
    int o_sf_start,
    int o_sf_columns,
    long long q_s0,
    long long q_s1,
    long long k_s0,
    long long k_s1,
    long long k_s2,
    long long k_s3,
    long long v_s0,
    long long v_s1,
    long long v_s2,
    long long v_s3,
    long long ksf_s0,
    long long ksf_s1,
    long long ksf_s2,
    long long ksf_s3,
    long long vsf_s0,
    long long vsf_s1,
    long long vsf_s2,
    long long vsf_s3,
    long long table_k_s0,
    long long table_v_s0,
    long long o_s0,
    long long o_s1) {
  __shared__ float scratch[256];
  __shared__ float update_scale;
  __shared__ float update_weight;
  __shared__ float final_sum;
  __shared__ float final_max;

  const int token_global = blockIdx.x;
  const int q_head = blockIdx.y;
  const int tid = threadIdx.x;
  if (q_head >= num_q_heads || tid >= 256) return;

  int batch = 0;
  while (batch + 1 < batch_size && token_global >= q_indptr[batch + 1]) ++batch;
  const int q_start = q_indptr[batch];
  const int q_end = q_indptr[batch + 1];
  const int q_len = q_end - q_start;
  const int q_pos = token_global - q_start;
  const int kv_len = seq_lens_kv[batch];
  const int kv_head = q_head / (num_q_heads / num_kv_heads);
  const int right = q_pos + kv_len - q_len;
  const int kv_begin = window_left >= 0 ? max(0, right - window_left) : 0;
  const int kv_end = causal ? min(kv_len, right + 1) : kv_len;

  float output_acc = 0.0f;
  float row_max = enable_sink ? sinks[q_head] : -CUDART_INF_F;
  float row_sum = enable_sink ? 1.0f : 0.0f;
  const float q_value = tid < head_dim
      ? load_scalar(q, static_cast<long long>(token_global) * q_s0
                       + static_cast<long long>(q_head) * q_s1 + tid, q_dtype) * q_scale
      : 0.0f;

  for (int kv_pos = kv_begin; kv_pos < kv_end; ++kv_pos) {
    const int page_slot = kv_pos / page_size;
    const int page_offset = kv_pos - page_slot * page_size;
    const int page_k = page_table_k[static_cast<long long>(batch) * table_k_s0 + page_slot];
    const int page_v = page_table_v[static_cast<long long>(batch) * table_v_s0 + page_slot];
    float k_value = 0.0f;
    if (tid < head_dim) {
      if (kv_dtype == 3) {
        k_value = load_nvfp4(
            k, k_scales, page_k, page_offset, kv_head, tid, kv_layout, false,
            page_size, num_kv_heads, head_dim,
            k_s0, k_s1, k_s2, k_s3, ksf_s0, ksf_s1, ksf_s2, ksf_s3);
      } else {
        const long long at = kv_index(
            page_k, page_offset, kv_head, tid, kv_layout, k_s0, k_s1, k_s2, k_s3);
        k_value = load_scalar(k, at, kv_dtype);
      }
      k_value *= k_scale;
    }
    const float dot = block_sum(q_value * k_value, scratch);
    if (tid == 0) {
      const float score = dot * sm_scale;
      const float new_max = fmaxf(row_max, score);
      const float alpha = row_sum == 0.0f ? 0.0f : __expf(row_max - new_max);
      const float weight = __expf(score - new_max);
      row_sum = row_sum * alpha + weight;
      row_max = new_max;
      update_scale = alpha;
      update_weight = weight;
    }
    __syncthreads();
    if (tid < head_dim) {
      float v_value;
      if (kv_dtype == 3) {
        v_value = load_nvfp4(
            v, v_scales, page_v, page_offset, kv_head, tid, kv_layout, true,
            page_size, num_kv_heads, head_dim,
            v_s0, v_s1, v_s2, v_s3, vsf_s0, vsf_s1, vsf_s2, vsf_s3);
      } else {
        const long long at = kv_index(
            page_v, page_offset, kv_head, tid, kv_layout, v_s0, v_s1, v_s2, v_s3);
        v_value = load_scalar(v, at, kv_dtype);
      }
      output_acc = output_acc * update_scale + update_weight * (v_value * v_scale);
    }
    __syncthreads();
  }

  if (tid == 0) {
    final_sum = row_sum;
    final_max = row_max;
  }
  __syncthreads();
  const float normalized = tid < head_dim && final_sum > 0.0f
      ? output_acc / final_sum / o_scale
      : 0.0f;
  if (o_dtype != 3) {
    if (tid < head_dim) {
      store_scalar(
          o,
          static_cast<long long>(token_global) * o_s0 + static_cast<long long>(q_head) * o_s1 + tid,
          o_dtype,
          normalized);
    }
  } else {
    float group_max = fabsf(normalized);
    #pragma unroll
    for (int offset = 8; offset != 0; offset >>= 1) {
      group_max = fmaxf(group_max, __shfl_down_sync(0xffffffffu, group_max, offset, 16));
    }
    const int lane16 = tid & 15;
    const int scale_column = q_head * (head_dim >> 4) + (tid >> 4);
    if (tid < head_dim && lane16 == 0) {
      const float raw_scale = o_sf_scale * group_max / 6.0f;
      const __nv_fp8_e4m3 encoded_scale(raw_scale);
      const long long sf_at = output_scale_swizzled_index(
          token_global + o_sf_start, scale_column, o_sf_columns);
      reinterpret_cast<__nv_fp8_e4m3*>(o_scales)[sf_at] = encoded_scale;
      scratch[tid >> 4] = static_cast<float>(encoded_scale);
    }
    __syncthreads();
    const float peer_value = __shfl_down_sync(0xffffffffu, normalized, 1);
    if (tid < head_dim && (tid & 1) == 0) {
      const float actual_scale = scratch[tid >> 4];
      const float own = actual_scale == 0.0f ? 0.0f : normalized * o_sf_scale / actual_scale;
      const float peer = actual_scale == 0.0f ? 0.0f : peer_value * o_sf_scale / actual_scale;
      const unsigned packed = nearest_e2m1(own) | (nearest_e2m1(peer) << 4);
      reinterpret_cast<unsigned char*>(o)[
          static_cast<long long>(token_global) * o_s0
          + static_cast<long long>(q_head) * o_s1 + (tid >> 1)] = static_cast<unsigned char>(packed);
    }
  }
  if (return_lse && tid == 0) {
    lse[static_cast<long long>(token_global) * num_q_heads + q_head] =
        (logf(final_sum) + final_max) * 1.4426950408889634f;
  }
}
