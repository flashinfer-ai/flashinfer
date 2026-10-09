/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#define THREADS 128
#ifndef COEFFICIENT_BF16
#error \
    "COEFFICIENT_BF16 is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef INDEX_I32
#error "INDEX_I32 is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef PREFETCH_ROWS
#error "PREFETCH_ROWS is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef DIM_TILES
#error "DIM_TILES is a downstream specialization of this program; define it on the compile line"
#endif
#define LAUNCH_MIN_BLOCKS 8

#include <math_constants.h>

extern "C" {

__global__
__launch_bounds__(128, LAUNCH_MIN_BLOCKS) void kernel_cake_selective_state_update_f75fba6195c641c8ea09(
    float* __restrict__ state, __nv_bfloat16* __restrict__ x, unsigned long long dt_addr,
    unsigned long long a_addr, __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
    unsigned long long d_addr, __nv_bfloat16* __restrict__ z, unsigned long long dt_bias_addr,
    __nv_bfloat16* __restrict__ output, unsigned long long state_batch_indices_addr,
    unsigned long long dst_state_batch_indices_addr, int nheads, int ngroups,
    unsigned long long state_stride_slot, long long x_batch_stride, long long b_batch_stride,
    long long c_batch_stride, long long out_batch_stride, long long dt_batch_stride,
    long long dt_head_stride, long long a_head_stride, long long d_head_stride,
    long long dt_bias_head_stride, int dt_softplus, int has_z, int disable_state_update,
    long long pad_slot_id) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // === Task calls (dependency order) ===
  int dim_tile = blockIdx.x;
  int head = blockIdx.y;
  int batch = blockIdx.z;
  int heads_per_group = nheads / ngroups;
  int group = head / heads_per_group;
  int rows_per_tile = 64 / DIM_TILES;
  int dim_base = dim_tile * rows_per_tile;
  long long batch_i64 = (long long)batch;
  int index_i32 = INDEX_I32;
  int coefficient_bf16 = COEFFICIENT_BF16;
  long long source_slot = 0;
  long long destination_slot = 0;
  if (index_i32 != 0) {
    int _vec_load_0[1];
    {
      uint32_t _scalar_bits_0;
      asm volatile(
          "ld.global.nc.b32 %0, [%1];"
          : "=r"(_scalar_bits_0)
          : "l"((const void*)(reinterpret_cast<const int*>(state_batch_indices_addr) + (batch_i64)))
          : "memory");
      _vec_load_0[0] = (int32_t)_scalar_bits_0;
    }
    int _vec_load_1[1];
    {
      uint32_t _scalar_bits_1;
      asm volatile("ld.global.nc.b32 %0, [%1];"
                   : "=r"(_scalar_bits_1)
                   : "l"((const void*)(reinterpret_cast<const int*>(dst_state_batch_indices_addr) +
                                       (batch_i64)))
                   : "memory");
      _vec_load_1[0] = (int32_t)_scalar_bits_1;
    }
    source_slot = (long long)_vec_load_0[0];
    destination_slot = (long long)_vec_load_1[0];
  } else {
    long long _vec_load_2[1];
    {
      asm("ld.global.nc.s64 %0, [%1];"
          : "=l"(_vec_load_2[0])
          : "l"((const void*)(reinterpret_cast<const long long*>(state_batch_indices_addr) +
                              (batch_i64))));
    }
    long long _vec_load_3[1];
    {
      asm("ld.global.nc.s64 %0, [%1];"
          : "=l"(_vec_load_3[0])
          : "l"((const void*)(reinterpret_cast<const long long*>(dst_state_batch_indices_addr) +
                              (batch_i64))));
    }
    source_slot = _vec_load_2[0];
    destination_slot = _vec_load_3[0];
  }
  int row_subgroup = lane / 16;
  int row_member = lane % 16;
  int state_col = row_member * 8;
  long long group_i64 = (long long)group;
  long long b_base = batch_i64 * b_batch_stride + group_i64 * 128;
  long long c_base = batch_i64 * c_batch_stride + group_i64 * 128;
  unsigned int b_carriers[4];
  unsigned int c_carriers[4];
  {
    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(B + b_base + (long long)state_col);
    uint4* _vdst_2 = reinterpret_cast<uint4*>(&b_carriers[0]);
#pragma unroll
    for (int _blk = 0; _blk < 1; _blk++) {
      _vdst_2[_blk] = _vptr_2[_blk];
    }
  }
  {
    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(C + c_base + (long long)state_col);
    uint4* _vdst_3 = reinterpret_cast<uint4*>(&c_carriers[0]);
#pragma unroll
    for (int _blk = 0; _blk < 1; _blk++) {
      _vdst_3[_blk] = _vptr_3[_blk];
    }
  }
  float row_values[PREFETCH_ROWS * 8];
  unsigned int row_carriers[PREFETCH_ROWS * 4];
  float x_lanes[PREFETCH_ROWS];
  float z_lanes[PREFETCH_ROWS];
  int live_source = 0;
  if (source_slot != pad_slot_id) {
    live_source = 1;
  }
  int live_store = 0;
  if (disable_state_update == 0) {
    if (source_slot != pad_slot_id) {
      if (destination_slot != pad_slot_id) {
        live_store = 1;
      }
    }
  }
  unsigned long long source_base = (unsigned long long)source_slot * state_stride_slot;
  unsigned long long destination_base = (unsigned long long)destination_slot * state_stride_slot;
  int trip_rows = 8 * PREFETCH_ROWS;
  int first_base = dim_base + warp * 2;
#pragma unroll
  for (int element = 0; element < PREFETCH_ROWS * 8; element++) {
    row_values[element] = 0.0f;
  }
#pragma unroll
  for (int element_1 = 0; element_1 < PREFETCH_ROWS * 4; element_1++) {
    row_carriers[element_1] = 0;
  }
#pragma unroll
  for (int r = 0; r < PREFETCH_ROWS; r++) {
    x_lanes[r] = 0.0f;
    z_lanes[r] = 0.0f;
  }
  if (row_member == 0) {
#pragma unroll
    for (int r_1 = 0; r_1 < PREFETCH_ROWS; r_1++) {
      int dim_index_r = first_base + r_1 * 8 + row_subgroup;
      long long x_index_r = batch_i64 * x_batch_stride + (long long)(head * 64 + dim_index_r);
      x_lanes[r_1] = (float)x[x_index_r];
      if (has_z != 0) {
        z_lanes[r_1] = (float)z[x_index_r];
      }
    }
  }
  if (live_source != 0) {
#pragma unroll
    for (int r_2 = 0; r_2 < PREFETCH_ROWS; r_2++) {
      int dim_index_r_1 = first_base + r_2 * 8 + row_subgroup;
      unsigned long long row_offset_r =
          (unsigned long long)((head * 64 + dim_index_r_1) * 128 + state_col);
      {
        {
          unsigned _ldv8_4_0;
          unsigned _ldv8_4_1;
          unsigned _ldv8_4_2;
          unsigned _ldv8_4_3;
          unsigned _ldv8_4_4;
          unsigned _ldv8_4_5;
          unsigned _ldv8_4_6;
          unsigned _ldv8_4_7;
          asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                       : "=r"(_ldv8_4_0), "=r"(_ldv8_4_1), "=r"(_ldv8_4_2), "=r"(_ldv8_4_3),
                         "=r"(_ldv8_4_4), "=r"(_ldv8_4_5), "=r"(_ldv8_4_6), "=r"(_ldv8_4_7)
                       : "l"((const void*)(state + (source_base + row_offset_r)))
                       : "memory");
          row_values[r_2 * 8 + 0] = __uint_as_float(_ldv8_4_0);
          row_values[r_2 * 8 + 1] = __uint_as_float(_ldv8_4_1);
          row_values[r_2 * 8 + 2] = __uint_as_float(_ldv8_4_2);
          row_values[r_2 * 8 + 3] = __uint_as_float(_ldv8_4_3);
          row_values[r_2 * 8 + 4] = __uint_as_float(_ldv8_4_4);
          row_values[r_2 * 8 + 5] = __uint_as_float(_ldv8_4_5);
          row_values[r_2 * 8 + 6] = __uint_as_float(_ldv8_4_6);
          row_values[r_2 * 8 + 7] = __uint_as_float(_ldv8_4_7);
        }
      }
    }
  }
  float dt_lane = 0.0f;
  float decay_lane = 0.0f;
  float d_lane = 0.0f;
  if (lane == 0) {
    float dt_raw = 0.0f;
    float dt_bias_raw = 0.0f;
    if (coefficient_bf16 != 0) {
      dt_raw = (float)reinterpret_cast<__nv_bfloat16*>(
          dt_addr)[(long long)batch * dt_batch_stride + (long long)head * dt_head_stride];
      dt_bias_raw = (float)reinterpret_cast<__nv_bfloat16*>(
          dt_bias_addr)[(long long)head * dt_bias_head_stride];
      d_lane = (float)reinterpret_cast<__nv_bfloat16*>(d_addr)[(long long)head * d_head_stride];
    } else {
      dt_raw = reinterpret_cast<float*>(
          dt_addr)[(long long)batch * dt_batch_stride + (long long)head * dt_head_stride];
      dt_bias_raw = reinterpret_cast<float*>(dt_bias_addr)[(long long)head * dt_bias_head_stride];
      d_lane = reinterpret_cast<float*>(d_addr)[(long long)head * d_head_stride];
    }
    float a_value = reinterpret_cast<float*>(a_addr)[(long long)head * a_head_stride];
    float dt_value = dt_raw + dt_bias_raw;
    if (dt_softplus != 0) {
      if (dt_value <= 20.0f) {
        {
          float _exp_0 = expf(dt_value);
          float _log1p_0 = log1pf(_exp_0);
          dt_value = _log1p_0;
        }
      }
    }
    dt_lane = dt_value;
    {
      float _exp_1 = expf(a_value * dt_value);
      decay_lane = _exp_1;
    }
  }
  float _shfl_0 = __shfl_sync(0xFFFFFFFF, dt_lane, 0);
  float dt_value_1 = _shfl_0;
  float _shfl_1 = __shfl_sync(0xFFFFFFFF, decay_lane, 0);
  float decay = _shfl_1;
  float _shfl_2 = __shfl_sync(0xFFFFFFFF, d_lane, 0);
  float d_value = _shfl_2;
#pragma unroll 1
  for (int trip = 0; trip < 64 / DIM_TILES / (8 * PREFETCH_ROWS); trip++) {
    int trip_base = first_base + trip * trip_rows;
    if (trip != 0) {
#pragma unroll
      for (int element_2 = 0; element_2 < PREFETCH_ROWS * 8; element_2++) {
        row_values[element_2] = 0.0f;
      }
#pragma unroll
      for (int element_3 = 0; element_3 < PREFETCH_ROWS * 4; element_3++) {
        row_carriers[element_3] = 0;
      }
#pragma unroll
      for (int r_3 = 0; r_3 < PREFETCH_ROWS; r_3++) {
        x_lanes[r_3] = 0.0f;
        z_lanes[r_3] = 0.0f;
      }
      if (row_member == 0) {
#pragma unroll
        for (int r_4 = 0; r_4 < PREFETCH_ROWS; r_4++) {
          int dim_index_r_2 = trip_base + r_4 * 8 + row_subgroup;
          long long x_index_r_1 =
              batch_i64 * x_batch_stride + (long long)(head * 64 + dim_index_r_2);
          x_lanes[r_4] = (float)x[x_index_r_1];
          if (has_z != 0) {
            z_lanes[r_4] = (float)z[x_index_r_1];
          }
        }
      }
      if (live_source != 0) {
#pragma unroll
        for (int r_5 = 0; r_5 < PREFETCH_ROWS; r_5++) {
          int dim_index_r_3 = trip_base + r_5 * 8 + row_subgroup;
          unsigned long long row_offset_r_1 =
              (unsigned long long)((head * 64 + dim_index_r_3) * 128 + state_col);
          {
            {
              unsigned _ldv8_5_0;
              unsigned _ldv8_5_1;
              unsigned _ldv8_5_2;
              unsigned _ldv8_5_3;
              unsigned _ldv8_5_4;
              unsigned _ldv8_5_5;
              unsigned _ldv8_5_6;
              unsigned _ldv8_5_7;
              asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                           : "=r"(_ldv8_5_0), "=r"(_ldv8_5_1), "=r"(_ldv8_5_2), "=r"(_ldv8_5_3),
                             "=r"(_ldv8_5_4), "=r"(_ldv8_5_5), "=r"(_ldv8_5_6), "=r"(_ldv8_5_7)
                           : "l"((const void*)(state + (source_base + row_offset_r_1)))
                           : "memory");
              row_values[r_5 * 8 + 0] = __uint_as_float(_ldv8_5_0);
              row_values[r_5 * 8 + 1] = __uint_as_float(_ldv8_5_1);
              row_values[r_5 * 8 + 2] = __uint_as_float(_ldv8_5_2);
              row_values[r_5 * 8 + 3] = __uint_as_float(_ldv8_5_3);
              row_values[r_5 * 8 + 4] = __uint_as_float(_ldv8_5_4);
              row_values[r_5 * 8 + 5] = __uint_as_float(_ldv8_5_5);
              row_values[r_5 * 8 + 6] = __uint_as_float(_ldv8_5_6);
              row_values[r_5 * 8 + 7] = __uint_as_float(_ldv8_5_7);
            }
          }
        }
      }
    }
#pragma unroll
    for (int r_6 = 0; r_6 < PREFETCH_ROWS; r_6++) {
      int dim_index_r_4 = trip_base + r_6 * 8 + row_subgroup;
      float _shfl_3 = __shfl_sync(0xFFFFFFFF, x_lanes[r_6], row_subgroup * 16);
      float x_value = _shfl_3;
      float partial = 0.0f;
#pragma unroll
      for (int half = 0; half < 1; half++) {
        float b_half[8];
        float c_half[8];
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_half[_pair * 2])[0]), "=f"((&b_half[_pair * 2])[1])
              : "r"((b_carriers + half * 4)[_pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_half[_pair * 2])[0]), "=f"((&c_half[_pair * 2])[1])
              : "r"((c_carriers + half * 4)[_pair]));
        }
#pragma unroll
        for (int element_4 = 0; element_4 < 8; element_4++) {
          float d_b = b_half[element_4] * dt_value_1;
          float decayed_state = row_values[r_6 * 8 + half * 8 + element_4] * decay;
          float new_state = decayed_state + d_b * x_value;
          row_values[r_6 * 8 + half * 8 + element_4] = new_state;
          partial += new_state * c_half[element_4];
        }
      }
      if (live_store != 0) {
        unsigned long long row_offset_r_2 =
            (unsigned long long)((head * 64 + dim_index_r_4) * 128 + state_col);
        {
          unsigned _stv8_6_0 = __float_as_uint(row_values[r_6 * 8 + 0]);
          unsigned _stv8_6_1 = __float_as_uint(row_values[r_6 * 8 + 1]);
          unsigned _stv8_6_2 = __float_as_uint(row_values[r_6 * 8 + 2]);
          unsigned _stv8_6_3 = __float_as_uint(row_values[r_6 * 8 + 3]);
          unsigned _stv8_6_4 = __float_as_uint(row_values[r_6 * 8 + 4]);
          unsigned _stv8_6_5 = __float_as_uint(row_values[r_6 * 8 + 5]);
          unsigned _stv8_6_6 = __float_as_uint(row_values[r_6 * 8 + 6]);
          unsigned _stv8_6_7 = __float_as_uint(row_values[r_6 * 8 + 7]);
          asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(
                           (void*)(state + (destination_base + row_offset_r_2))),
                       "r"(_stv8_6_0), "r"(_stv8_6_1), "r"(_stv8_6_2), "r"(_stv8_6_3),
                       "r"(_stv8_6_4), "r"(_stv8_6_5), "r"(_stv8_6_6), "r"(_stv8_6_7)
                       : "memory");
        }
      }
      float row_sum = partial;
      {
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 8);
        row_sum += _shfl_xor_1;
      }
      float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 4);
      row_sum += _shfl_xor_2;
      float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 2);
      row_sum += _shfl_xor_3;
      float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 1);
      row_sum += _shfl_xor_4;
      if (row_member == 0) {
        float result = row_sum + d_value * x_lanes[r_6];
        if (has_z != 0) {
          float z_lane = z_lanes[r_6];
          float sigmoid_z = 0.0f;
          {
            float _exp_2 = expf(-z_lane);
            sigmoid_z = 1.0f / (1.0f + _exp_2);
          }
          result *= z_lane * sigmoid_z;
        }
        long long out_index_r =
            batch_i64 * out_batch_stride + (long long)(head * 64 + dim_index_r_4);
        output[out_index_r] = result;
      }
    }
  }
}

}  // extern "C"
