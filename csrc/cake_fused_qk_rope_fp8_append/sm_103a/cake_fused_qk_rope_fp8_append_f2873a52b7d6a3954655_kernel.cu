/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int int32_t;
typedef short int int16_t;
struct __align__(64) CakeTensorMap64 {
  uint64_t opaque[16];
};
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) {
  uint64_t opaque[16];
} CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
  int result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;" : "=r"(result) : "r"(x));
  return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_META_OFF 0
#define SMEM_META_STAGE_BYTES 16
#define SMEM_META_STRIDE 16
#define SMEM_TOTAL 0
#define THREADS 96

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(96) void kernel_cake_fused_qk_rope_fp8_append_f2873a52b7d6a3954655(
    const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ cos_sin,
    const int* __restrict__ seq_lens, const int* __restrict__ q_indptr,
    const int* __restrict__ page_indices, const float* __restrict__ q_norm_weight,
    const float* __restrict__ k_norm_weight, const float* __restrict__ k_scale,
    const float* __restrict__ v_scale, const float* __restrict__ q_scale_inv,
    uint8_t* __restrict__ out_q, uint8_t* key_cache, uint8_t* value_cache, uint8_t* out_k,
    uint8_t* out_v, float* __restrict__ q_scale, int* __restrict__ split_k_flag, int num_rows,
    int num_requests, int num_q_heads, int num_kv_heads, int page_size, int max_pages_per_request,
    int max_seqlen, int max_seqlen_aligned, int quant_policy, int norm_policy, bool is_prefill,
    bool has_out_kv, float upper_max) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  __shared__ __align__(16) unsigned char smem_static_raw[16];

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  int* meta = reinterpret_cast<int*>(smem_static_raw + 0);
  const int meta_addr = (int)(unsigned long long)__cvta_generic_to_shared(meta);

  // === Task calls (dependency order) ===
  int tid_0 = tid;
  int slot = tid_0 / 8;
  int lane_1 = tid_0 % 8;
  int grp_base = tid_0 % 32 - lane_1;
  unsigned int grp_mask = (unsigned int)(255 << grp_base);
  int kv_head = blockIdx.y;
  int q_per_kv = num_q_heads / num_kv_heads;
  int bid_1 = blockIdx.x;
  bool is_row_cta = bid_1 < num_rows;
  int row = ((is_row_cta) ? bid_1 : 0);
  bool dyn_prefill = quant_policy == 1 && is_prefill;
  int width = (num_q_heads + 2 * num_kv_heads) * 128;
  bool is_q = slot < q_per_kv;
  bool is_k = slot == 8;
  bool is_v = slot == 9;
  bool active = is_q || is_k || is_v;
  int head_in_row =
      ((is_q) ? kv_head * q_per_kv + slot
              : ((is_k) ? num_q_heads + kv_head : num_q_heads + num_kv_heads + kv_head));
  int lane_lo = ((is_v) ? lane_1 * 16 : lane_1 * 8);
  int lane_hi = ((is_v) ? lane_1 * 16 + 8 : 64 + lane_1 * 8);
  unsigned int x_lo_w[4];
  unsigned int x_hi_w[4];
  float w_lo[8];
  float w_hi[8];
  float c_lo[8];
  float c_hi[8];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    x_lo_w[j] = 0;
    x_hi_w[j] = 0;
  }
#pragma unroll
  for (int j_1 = 0; j_1 < 8; j_1++) {
    w_lo[j_1] = 1.0f;
    w_hi[j_1] = 1.0f;
    c_lo[j_1] = 1.0f;
    c_hi[j_1] = 0.0f;
  }
  if (is_row_cta && active) {
    long long src_base = (long long)row * (long long)width + (long long)(head_in_row * 128);
    {
      const uint4* _vptr_0 =
          reinterpret_cast<const uint4*>(qkv + (src_base + (long long)lane_lo) + 0);
      uint4* _vdst_0 = reinterpret_cast<uint4*>(&x_lo_w[0]);
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(_vdst_0[_blk].x), "=r"(_vdst_0[_blk].y), "=r"(_vdst_0[_blk].z),
                       "=r"(_vdst_0[_blk].w)
                     : "l"((const void*)(_vptr_0 + _blk))
                     : "memory");
      }
    }
    {
      const uint4* _vptr_1 =
          reinterpret_cast<const uint4*>(qkv + (src_base + (long long)lane_hi) + 0);
      uint4* _vdst_1 = reinterpret_cast<uint4*>(&x_hi_w[0]);
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(_vdst_1[_blk].x), "=r"(_vdst_1[_blk].y), "=r"(_vdst_1[_blk].z),
                       "=r"(_vdst_1[_blk].w)
                     : "l"((const void*)(_vptr_1 + _blk))
                     : "memory");
      }
    }
    if (norm_policy != 0 && (is_q || is_k)) {
      if (is_q) {
        {
          unsigned _v4_2_0;
          unsigned _v4_2_1;
          unsigned _v4_2_2;
          unsigned _v4_2_3;
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_v4_2_0), "=r"(_v4_2_1), "=r"(_v4_2_2), "=r"(_v4_2_3)
                       : "l"((const void*)(q_norm_weight + (lane_1 * 8) + (0)))
                       : "memory");
          w_lo[0 + 0] = __uint_as_float(_v4_2_0);
          w_lo[0 + 1] = __uint_as_float(_v4_2_1);
          w_lo[0 + 2] = __uint_as_float(_v4_2_2);
          w_lo[0 + 3] = __uint_as_float(_v4_2_3);
        }
        {
          unsigned _v4_3_0;
          unsigned _v4_3_1;
          unsigned _v4_3_2;
          unsigned _v4_3_3;
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_v4_3_0), "=r"(_v4_3_1), "=r"(_v4_3_2), "=r"(_v4_3_3)
                       : "l"((const void*)(q_norm_weight + (lane_1 * 8 + 4) + (0)))
                       : "memory");
          w_lo[4 + 0] = __uint_as_float(_v4_3_0);
          w_lo[4 + 1] = __uint_as_float(_v4_3_1);
          w_lo[4 + 2] = __uint_as_float(_v4_3_2);
          w_lo[4 + 3] = __uint_as_float(_v4_3_3);
        }
        {
          unsigned _v4_4_0;
          unsigned _v4_4_1;
          unsigned _v4_4_2;
          unsigned _v4_4_3;
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_v4_4_0), "=r"(_v4_4_1), "=r"(_v4_4_2), "=r"(_v4_4_3)
                       : "l"((const void*)(q_norm_weight + (64 + lane_1 * 8) + (0)))
                       : "memory");
          w_hi[0 + 0] = __uint_as_float(_v4_4_0);
          w_hi[0 + 1] = __uint_as_float(_v4_4_1);
          w_hi[0 + 2] = __uint_as_float(_v4_4_2);
          w_hi[0 + 3] = __uint_as_float(_v4_4_3);
        }
        {
          unsigned _v4_5_0;
          unsigned _v4_5_1;
          unsigned _v4_5_2;
          unsigned _v4_5_3;
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_v4_5_0), "=r"(_v4_5_1), "=r"(_v4_5_2), "=r"(_v4_5_3)
                       : "l"((const void*)(q_norm_weight + (64 + lane_1 * 8 + 4) + (0)))
                       : "memory");
          w_hi[4 + 0] = __uint_as_float(_v4_5_0);
          w_hi[4 + 1] = __uint_as_float(_v4_5_1);
          w_hi[4 + 2] = __uint_as_float(_v4_5_2);
          w_hi[4 + 3] = __uint_as_float(_v4_5_3);
        }
      } else {
        {
          unsigned _v4_6_0;
          unsigned _v4_6_1;
          unsigned _v4_6_2;
          unsigned _v4_6_3;
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_v4_6_0), "=r"(_v4_6_1), "=r"(_v4_6_2), "=r"(_v4_6_3)
                       : "l"((const void*)(k_norm_weight + (lane_1 * 8) + (0)))
                       : "memory");
          w_lo[0 + 0] = __uint_as_float(_v4_6_0);
          w_lo[0 + 1] = __uint_as_float(_v4_6_1);
          w_lo[0 + 2] = __uint_as_float(_v4_6_2);
          w_lo[0 + 3] = __uint_as_float(_v4_6_3);
        }
        {
          unsigned _v4_7_0;
          unsigned _v4_7_1;
          unsigned _v4_7_2;
          unsigned _v4_7_3;
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_v4_7_0), "=r"(_v4_7_1), "=r"(_v4_7_2), "=r"(_v4_7_3)
                       : "l"((const void*)(k_norm_weight + (lane_1 * 8 + 4) + (0)))
                       : "memory");
          w_lo[4 + 0] = __uint_as_float(_v4_7_0);
          w_lo[4 + 1] = __uint_as_float(_v4_7_1);
          w_lo[4 + 2] = __uint_as_float(_v4_7_2);
          w_lo[4 + 3] = __uint_as_float(_v4_7_3);
        }
        {
          unsigned _v4_8_0;
          unsigned _v4_8_1;
          unsigned _v4_8_2;
          unsigned _v4_8_3;
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_v4_8_0), "=r"(_v4_8_1), "=r"(_v4_8_2), "=r"(_v4_8_3)
                       : "l"((const void*)(k_norm_weight + (64 + lane_1 * 8) + (0)))
                       : "memory");
          w_hi[0 + 0] = __uint_as_float(_v4_8_0);
          w_hi[0 + 1] = __uint_as_float(_v4_8_1);
          w_hi[0 + 2] = __uint_as_float(_v4_8_2);
          w_hi[0 + 3] = __uint_as_float(_v4_8_3);
        }
        {
          unsigned _v4_9_0;
          unsigned _v4_9_1;
          unsigned _v4_9_2;
          unsigned _v4_9_3;
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_v4_9_0), "=r"(_v4_9_1), "=r"(_v4_9_2), "=r"(_v4_9_3)
                       : "l"((const void*)(k_norm_weight + (64 + lane_1 * 8 + 4) + (0)))
                       : "memory");
          w_hi[4 + 0] = __uint_as_float(_v4_9_0);
          w_hi[4 + 1] = __uint_as_float(_v4_9_1);
          w_hi[4 + 2] = __uint_as_float(_v4_9_2);
          w_hi[4 + 3] = __uint_as_float(_v4_9_3);
        }
      }
    }
  }
  float k_scale_v = 1.0f;
  float v_scale_v = 1.0f;
  float q_static = 1.0f;
  if (is_row_cta) {
    if (is_k) {
      k_scale_v = k_scale[0];
    }
    if (is_v) {
      v_scale_v = v_scale[0];
    }
    if (is_q && quant_policy == 2) {
      q_static = q_scale_inv[0];
    }
  }
  int req_id = bid_1 - num_rows;
  int clear_last_pos = -1;
  if (!is_row_cta && req_id < num_requests) {
    clear_last_pos = seq_lens[req_id] - 1;
  }
  bool any_bad = 0;
#pragma unroll 1
  for (int req = tid_0; req < num_requests; req += 96) {
    int begin = q_indptr[req];
    int end = q_indptr[req + 1];
    int seq_len_r = seq_lens[req];
    bool bad = begin < 0 || end < begin || end > num_rows;
    if (dyn_prefill && end - begin > max_seqlen) {
      bad = 1;
    }
    if (bad) {
      any_bad = 1;
    }
    if (is_row_cta) {
      if (begin <= row && row < end) {
        meta[0] = req;
        meta[1] = row + seq_len_r - end;
        meta[2] = row - begin;
      }
    }
  }
  if (tid_0 == 0) {
    int first = q_indptr[0];
    int last_ip = q_indptr[num_requests];
    if (first != 0 || last_ip != num_rows) {
      any_bad = 1;
    }
  }
  uint32_t _cta_or_0 = __syncthreads_or(any_bad);
  unsigned int invalid = _cta_or_0;
  if (is_row_cta) {
    if (invalid == 0 && active) {
      int batch_i = meta[0];
      int pos = meta[1];
      int tok = meta[2];
      long long cache_row = 0;
      if (is_q || is_k) {
        long long cs_base = (long long)pos * 128 + (long long)(lane_1 * 8);
        {
          unsigned _v4_10_0;
          unsigned _v4_10_1;
          unsigned _v4_10_2;
          unsigned _v4_10_3;
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_v4_10_0), "=r"(_v4_10_1), "=r"(_v4_10_2), "=r"(_v4_10_3)
                       : "l"((const void*)(cos_sin + cs_base + (0)))
                       : "memory");
          c_lo[0 + 0] = __uint_as_float(_v4_10_0);
          c_lo[0 + 1] = __uint_as_float(_v4_10_1);
          c_lo[0 + 2] = __uint_as_float(_v4_10_2);
          c_lo[0 + 3] = __uint_as_float(_v4_10_3);
        }
        {
          unsigned _v4_11_0;
          unsigned _v4_11_1;
          unsigned _v4_11_2;
          unsigned _v4_11_3;
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_v4_11_0), "=r"(_v4_11_1), "=r"(_v4_11_2), "=r"(_v4_11_3)
                       : "l"((const void*)(cos_sin + (cs_base + 4) + (0)))
                       : "memory");
          c_lo[4 + 0] = __uint_as_float(_v4_11_0);
          c_lo[4 + 1] = __uint_as_float(_v4_11_1);
          c_lo[4 + 2] = __uint_as_float(_v4_11_2);
          c_lo[4 + 3] = __uint_as_float(_v4_11_3);
        }
        {
          unsigned _v4_12_0;
          unsigned _v4_12_1;
          unsigned _v4_12_2;
          unsigned _v4_12_3;
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_v4_12_0), "=r"(_v4_12_1), "=r"(_v4_12_2), "=r"(_v4_12_3)
                       : "l"((const void*)(cos_sin + (cs_base + 64) + (0)))
                       : "memory");
          c_hi[0 + 0] = __uint_as_float(_v4_12_0);
          c_hi[0 + 1] = __uint_as_float(_v4_12_1);
          c_hi[0 + 2] = __uint_as_float(_v4_12_2);
          c_hi[0 + 3] = __uint_as_float(_v4_12_3);
        }
        {
          unsigned _v4_13_0;
          unsigned _v4_13_1;
          unsigned _v4_13_2;
          unsigned _v4_13_3;
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_v4_13_0), "=r"(_v4_13_1), "=r"(_v4_13_2), "=r"(_v4_13_3)
                       : "l"((const void*)(cos_sin + (cs_base + 64 + 4) + (0)))
                       : "memory");
          c_hi[4 + 0] = __uint_as_float(_v4_13_0);
          c_hi[4 + 1] = __uint_as_float(_v4_13_1);
          c_hi[4 + 2] = __uint_as_float(_v4_13_2);
          c_hi[4 + 3] = __uint_as_float(_v4_13_3);
        }
      }
      if ((is_k || is_v) && !has_out_kv) {
        int page_slot = pos / page_size;
        int row_in_page = pos - page_slot * page_size;
        int phys = page_indices[batch_i * max_pages_per_request + page_slot];
        cache_row = ((long long)phys * (long long)page_size + (long long)row_in_page) *
                        (long long)(num_kv_heads * 128) +
                    (long long)(kv_head * 128);
      }
      float x_lo[8];
      float x_hi[8];
#pragma unroll
      for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&x_lo[_pair * 2])[0]), "=f"((&x_lo[_pair * 2])[1])
            : "r"(x_lo_w[_pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&x_hi[_pair * 2])[0]), "=f"((&x_hi[_pair * 2])[1])
            : "r"(x_hi_w[_pair]));
      }
      if (is_q || is_k) {
        if (norm_policy == 2) {
          float ss = 0.0f;
#pragma unroll
          for (int j_2 = 0; j_2 < 8; j_2++) {
            float _fma_0 = __fmaf_rn(x_lo[j_2], x_lo[j_2], ss);
            ss = _fma_0;
            float _fma_1 = __fmaf_rn(x_hi[j_2], x_hi[j_2], ss);
            ss = _fma_1;
          }
          float _shfl_xor_0 = __shfl_xor_sync(grp_mask, ss, 4);
          ss = ss + _shfl_xor_0;
          float _shfl_xor_1 = __shfl_xor_sync(grp_mask, ss, 2);
          ss = ss + _shfl_xor_1;
          float _shfl_xor_2 = __shfl_xor_sync(grp_mask, ss, 1);
          ss = ss + _shfl_xor_2;
          float _rsqrt_0 = rsqrtf(ss * 0.0078125f + 1e-06f);
          float inv_rms = _rsqrt_0;
#pragma unroll
          for (int j_3 = 0; j_3 < 8; j_3++) {
            x_lo[j_3] = x_lo[j_3] * (inv_rms * w_lo[j_3]);
            x_hi[j_3] = x_hi[j_3] * (inv_rms * w_hi[j_3]);
          }
        }
#pragma unroll
        for (int j_4 = 0; j_4 < 8; j_4++) {
          float y_lo = x_lo[j_4] * c_lo[j_4] - x_hi[j_4] * c_hi[j_4];
          float y_hi = x_hi[j_4] * c_lo[j_4] + x_lo[j_4] * c_hi[j_4];
          x_lo[j_4] = y_lo;
          x_hi[j_4] = y_hi;
        }
        if (norm_policy == 1) {
          float ss1 = 0.0f;
#pragma unroll
          for (int j_5 = 0; j_5 < 8; j_5++) {
            float _fma_2 = __fmaf_rn(x_lo[j_5], x_lo[j_5], ss1);
            ss1 = _fma_2;
            float _fma_3 = __fmaf_rn(x_hi[j_5], x_hi[j_5], ss1);
            ss1 = _fma_3;
          }
          float _shfl_xor_3 = __shfl_xor_sync(grp_mask, ss1, 4);
          ss1 = ss1 + _shfl_xor_3;
          float _shfl_xor_4 = __shfl_xor_sync(grp_mask, ss1, 2);
          ss1 = ss1 + _shfl_xor_4;
          float _shfl_xor_5 = __shfl_xor_sync(grp_mask, ss1, 1);
          ss1 = ss1 + _shfl_xor_5;
          float _rsqrt_1 = rsqrtf(ss1 * 0.0078125f + 1e-06f);
          float inv_rms1 = _rsqrt_1;
#pragma unroll
          for (int j_6 = 0; j_6 < 8; j_6++) {
            x_lo[j_6] = x_lo[j_6] * (inv_rms1 * w_lo[j_6]);
            x_hi[j_6] = x_hi[j_6] * (inv_rms1 * w_hi[j_6]);
          }
        }
      }
      float _rcp_0 = __frcp_rn(v_scale_v);
      float mult = _rcp_0;
      if (is_q) {
        if (quant_policy == 1) {
          float m = 1e-06f;
#pragma unroll
          for (int j_7 = 0; j_7 < 8; j_7++) {
            float _fabs_0 = fabsf(x_lo[j_7]);
            float _fmax_0 = fmaxf(m, _fabs_0);
            m = _fmax_0;
            float _fabs_1 = fabsf(x_hi[j_7]);
            float _fmax_1 = fmaxf(m, _fabs_1);
            m = _fmax_1;
          }
          float _shfl_xor_6 = __shfl_xor_sync(grp_mask, m, 4);
          float _fmax_2 = fmaxf(m, _shfl_xor_6);
          m = _fmax_2;
          float _shfl_xor_7 = __shfl_xor_sync(grp_mask, m, 2);
          float _fmax_3 = fmaxf(m, _shfl_xor_7);
          m = _fmax_3;
          float _shfl_xor_8 = __shfl_xor_sync(grp_mask, m, 1);
          float _fmax_4 = fmaxf(m, _shfl_xor_8);
          m = _fmax_4;
          float _fdiv_rn_0 = __fdiv_rn(m, upper_max);
          float scale_val = _fdiv_rn_0;
          if (lane_1 == 0) {
            if (is_prefill) {
              if (tok < max_seqlen_aligned) {
                long long qs_idx =
                    ((long long)batch_i * (long long)num_q_heads + (long long)head_in_row) *
                        (long long)max_seqlen_aligned +
                    (long long)tok;
                *(reinterpret_cast<float*>(q_scale + qs_idx) + (0)) = scale_val;
              }
            } else {
              long long qs_idx2 = (long long)row * (long long)num_q_heads + (long long)head_in_row;
              *(reinterpret_cast<float*>(q_scale + qs_idx2) + (0)) = scale_val;
            }
          }
          float _rcp_1 = __frcp_rn(scale_val);
          mult = _rcp_1;
        } else {
          mult = q_static;
        }
      } else if (is_k) {
        float _rcp_2 = __frcp_rn(k_scale_v);
        mult = _rcp_2;
      }
#pragma unroll
      for (int j_8 = 0; j_8 < 8; j_8++) {
        x_lo[j_8] = x_lo[j_8] * mult;
        x_hi[j_8] = x_hi[j_8] * mult;
      }
      long long row64 = row;
      if (is_q) {
        long long q_dst = (row64 * (long long)num_q_heads + (long long)head_in_row) * 128;
        {
          unsigned int _fp8_pk[2];
          asm("{\n\t"
              ".reg .b16 _lo, _hi;\n\t"
              "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
              "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
              "mov.b32 %0, {_lo, _hi};\n\t"
              "}\n"
              : "=r"(_fp8_pk[0])
              : "f"(x_lo[0 + 0]), "f"(x_lo[0 + 1]), "f"(x_lo[0 + 2]), "f"(x_lo[0 + 3]));
          asm("{\n\t"
              ".reg .b16 _lo, _hi;\n\t"
              "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
              "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
              "mov.b32 %0, {_lo, _hi};\n\t"
              "}\n"
              : "=r"(_fp8_pk[1])
              : "f"(x_lo[0 + 4]), "f"(x_lo[0 + 5]), "f"(x_lo[0 + 6]), "f"(x_lo[0 + 7]));
          *reinterpret_cast<uint2*>(
              reinterpret_cast<unsigned char*>(out_q + (q_dst + (long long)lane_lo)) + (0)) =
              *reinterpret_cast<uint2*>(_fp8_pk);
        }
        {
          unsigned int _fp8_pk[2];
          asm("{\n\t"
              ".reg .b16 _lo, _hi;\n\t"
              "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
              "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
              "mov.b32 %0, {_lo, _hi};\n\t"
              "}\n"
              : "=r"(_fp8_pk[0])
              : "f"(x_hi[0 + 0]), "f"(x_hi[0 + 1]), "f"(x_hi[0 + 2]), "f"(x_hi[0 + 3]));
          asm("{\n\t"
              ".reg .b16 _lo, _hi;\n\t"
              "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
              "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
              "mov.b32 %0, {_lo, _hi};\n\t"
              "}\n"
              : "=r"(_fp8_pk[1])
              : "f"(x_hi[0 + 4]), "f"(x_hi[0 + 5]), "f"(x_hi[0 + 6]), "f"(x_hi[0 + 7]));
          *reinterpret_cast<uint2*>(
              reinterpret_cast<unsigned char*>(out_q + (q_dst + (long long)lane_hi)) + (0)) =
              *reinterpret_cast<uint2*>(_fp8_pk);
        }
      } else if (is_k) {
        if (has_out_kv) {
          long long k_dst = (row64 * (long long)num_kv_heads + (long long)kv_head) * 128;
          {
            unsigned int _fp8_pk[2];
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[0])
                : "f"(x_lo[0 + 0]), "f"(x_lo[0 + 1]), "f"(x_lo[0 + 2]), "f"(x_lo[0 + 3]));
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[1])
                : "f"(x_lo[0 + 4]), "f"(x_lo[0 + 5]), "f"(x_lo[0 + 6]), "f"(x_lo[0 + 7]));
            *reinterpret_cast<uint2*>(
                reinterpret_cast<unsigned char*>(out_k + (k_dst + (long long)lane_lo)) + (0)) =
                *reinterpret_cast<uint2*>(_fp8_pk);
          }
          {
            unsigned int _fp8_pk[2];
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[0])
                : "f"(x_hi[0 + 0]), "f"(x_hi[0 + 1]), "f"(x_hi[0 + 2]), "f"(x_hi[0 + 3]));
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[1])
                : "f"(x_hi[0 + 4]), "f"(x_hi[0 + 5]), "f"(x_hi[0 + 6]), "f"(x_hi[0 + 7]));
            *reinterpret_cast<uint2*>(
                reinterpret_cast<unsigned char*>(out_k + (k_dst + (long long)lane_hi)) + (0)) =
                *reinterpret_cast<uint2*>(_fp8_pk);
          }
        } else {
          {
            unsigned int _fp8_pk[2];
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[0])
                : "f"(x_lo[0 + 0]), "f"(x_lo[0 + 1]), "f"(x_lo[0 + 2]), "f"(x_lo[0 + 3]));
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[1])
                : "f"(x_lo[0 + 4]), "f"(x_lo[0 + 5]), "f"(x_lo[0 + 6]), "f"(x_lo[0 + 7]));
            *reinterpret_cast<uint2*>(
                reinterpret_cast<unsigned char*>(key_cache + (cache_row + (long long)lane_lo)) +
                (0)) = *reinterpret_cast<uint2*>(_fp8_pk);
          }
          {
            unsigned int _fp8_pk[2];
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[0])
                : "f"(x_hi[0 + 0]), "f"(x_hi[0 + 1]), "f"(x_hi[0 + 2]), "f"(x_hi[0 + 3]));
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[1])
                : "f"(x_hi[0 + 4]), "f"(x_hi[0 + 5]), "f"(x_hi[0 + 6]), "f"(x_hi[0 + 7]));
            *reinterpret_cast<uint2*>(
                reinterpret_cast<unsigned char*>(key_cache + (cache_row + (long long)lane_hi)) +
                (0)) = *reinterpret_cast<uint2*>(_fp8_pk);
          }
        }
      } else {
        if (has_out_kv) {
          long long v_dst = (row64 * (long long)num_kv_heads + (long long)kv_head) * 128;
          {
            unsigned int _fp8_pk[2];
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[0])
                : "f"(x_lo[0 + 0]), "f"(x_lo[0 + 1]), "f"(x_lo[0 + 2]), "f"(x_lo[0 + 3]));
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[1])
                : "f"(x_lo[0 + 4]), "f"(x_lo[0 + 5]), "f"(x_lo[0 + 6]), "f"(x_lo[0 + 7]));
            *reinterpret_cast<uint2*>(
                reinterpret_cast<unsigned char*>(out_v + (v_dst + (long long)lane_lo)) + (0)) =
                *reinterpret_cast<uint2*>(_fp8_pk);
          }
          {
            unsigned int _fp8_pk[2];
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[0])
                : "f"(x_hi[0 + 0]), "f"(x_hi[0 + 1]), "f"(x_hi[0 + 2]), "f"(x_hi[0 + 3]));
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[1])
                : "f"(x_hi[0 + 4]), "f"(x_hi[0 + 5]), "f"(x_hi[0 + 6]), "f"(x_hi[0 + 7]));
            *reinterpret_cast<uint2*>(
                reinterpret_cast<unsigned char*>(out_v + (v_dst + (long long)lane_hi)) + (0)) =
                *reinterpret_cast<uint2*>(_fp8_pk);
          }
        } else {
          {
            unsigned int _fp8_pk[2];
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[0])
                : "f"(x_lo[0 + 0]), "f"(x_lo[0 + 1]), "f"(x_lo[0 + 2]), "f"(x_lo[0 + 3]));
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[1])
                : "f"(x_lo[0 + 4]), "f"(x_lo[0 + 5]), "f"(x_lo[0 + 6]), "f"(x_lo[0 + 7]));
            *reinterpret_cast<uint2*>(
                reinterpret_cast<unsigned char*>(value_cache + (cache_row + (long long)lane_lo)) +
                (0)) = *reinterpret_cast<uint2*>(_fp8_pk);
          }
          {
            unsigned int _fp8_pk[2];
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[0])
                : "f"(x_hi[0 + 0]), "f"(x_hi[0 + 1]), "f"(x_hi[0 + 2]), "f"(x_hi[0 + 3]));
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[1])
                : "f"(x_hi[0 + 4]), "f"(x_hi[0 + 5]), "f"(x_hi[0 + 6]), "f"(x_hi[0 + 7]));
            *reinterpret_cast<uint2*>(
                reinterpret_cast<unsigned char*>(value_cache + (cache_row + (long long)lane_hi)) +
                (0)) = *reinterpret_cast<uint2*>(_fp8_pk);
          }
        }
      }
    }
  } else if (req_id < num_requests) {
    if (tid_0 == 0) {
      int flag_val = ((invalid != 0) ? -1 : 0);
      *(reinterpret_cast<int*>(split_k_flag + (req_id * num_kv_heads + kv_head)) + (0)) = flag_val;
    }
    if (invalid == 0) {
      int last_pos = clear_last_pos;
      if (last_pos >= 0) {
        int last_slot = last_pos / page_size;
        int zero_from = last_pos - last_slot * page_size + 1;
        if (zero_from < page_size) {
          int phys_last = page_indices[req_id * max_pages_per_request + last_slot];
          long long phys_last64 = phys_last;
          unsigned int zero_w[4];
#pragma unroll
          for (int j_9 = 0; j_9 < 4; j_9++) {
            zero_w[j_9] = 0;
          }
          int n_rows = page_size - zero_from;
          int total_items = n_rows * 16;
#pragma unroll 1
          for (int item = tid_0; item < total_items; item += 96) {
            int r = item / 16;
            int sub = item - r * 16;
            int chunk = sub % 8;
            long long base =
                (phys_last64 * (long long)page_size + (long long)zero_from + (long long)r) *
                    (long long)(num_kv_heads * 128) +
                (long long)(kv_head * 128) + (long long)(chunk * 16);
            if (sub < 8) {
              reinterpret_cast<int4*>(key_cache + base)[0] = reinterpret_cast<int4*>(zero_w)[0];
            } else {
              reinterpret_cast<int4*>(value_cache + base)[0] = reinterpret_cast<int4*>(zero_w)[0];
            }
          }
        }
      }
    }
  }
}

}  // extern "C"
