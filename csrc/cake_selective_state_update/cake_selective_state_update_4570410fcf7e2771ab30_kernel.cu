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
#define SMEM_SHARED_COEFFICIENTS_OFF 0
#define SMEM_SHARED_COEFFICIENTS_STAGE_BYTES 0
#define SMEM_SHARED_COEFFICIENTS_STRIDE 0
#define SMEM_TOTAL 0
#define THREADS 128
#define COEFFICIENT_BF16 0
#define INDEX_I32 0
#define LAUNCH_MIN_BLOCKS 8

#include <math_constants.h>

extern "C" {

__global__
__launch_bounds__(128, LAUNCH_MIN_BLOCKS) void kernel_cake_selective_state_update_4570410fcf7e2771ab30(
    float* __restrict__ state, __nv_bfloat16* __restrict__ x, unsigned long long dt_addr,
    unsigned long long a_addr, __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
    unsigned long long d_addr, __nv_bfloat16* __restrict__ z, unsigned long long dt_bias_addr,
    __nv_bfloat16* __restrict__ output, unsigned long long state_batch_indices_addr,
    unsigned long long dst_state_batch_indices_addr, int nheads, int ngroups, int dim_tiles,
    unsigned long long state_stride_slot, long long x_batch_stride, long long b_batch_stride,
    long long c_batch_stride, long long out_batch_stride, long long dt_batch_stride,
    long long dt_head_stride, long long a_head_stride, long long d_head_stride,
    long long dt_bias_head_stride, int dt_softplus, int has_z, int disable_state_update,
    long long pad_slot_id) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
#if __CUDA_ARCH__ == 1000
  asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }"
               : "=r"(smem)
               : "l"(smem_raw));
  smem = make_warp_uniform(smem);
#else
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  unsigned int* shared_coefficients = reinterpret_cast<unsigned int*>(smem_raw + 0);
  const int shared_coefficients_addr = smem + 0;

  // === Task calls (dependency order) ===
  int dim_tile = 0;
  int batch_head = bid;
  int head = batch_head % nheads;
  int batch = batch_head / nheads;
  int heads_per_group = nheads / ngroups;
  int group = head / heads_per_group;
  int rows_per_tile = 128;
  int dim_base = 0;
  int dim_end = 128;
  int index_i32 = INDEX_I32;
  int coefficient_bf16 = COEFFICIENT_BF16;
  long long batch_i64 = (long long)batch;
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
    source_slot = (long long)_vec_load_0[0];
    {
      destination_slot = source_slot;
    }
  } else {
    long long _vec_load_2[1];
    {
      asm("ld.global.nc.s64 %0, [%1];"
          : "=l"(_vec_load_2[0])
          : "l"((const void*)(reinterpret_cast<const long long*>(state_batch_indices_addr) +
                              (batch_i64))));
    }
    source_slot = _vec_load_2[0];
    {
      destination_slot = source_slot;
    }
  }
  int row_subgroup = lane / 16;
  int row_member = lane % 16;
  int state_col = row_member * 8;
  long long group_i64 = (long long)group;
  long long b_base = batch_i64 * b_batch_stride + group_i64 * 128;
  long long c_base = batch_i64 * c_batch_stride + group_i64 * 128;
  unsigned int b_carriers[4];
  unsigned int c_carriers[4];
  float b_direct_values[8];
  float c_direct_values[8];
  {
    {
      const uint4* _vptr_1 = reinterpret_cast<const uint4*>(B + b_base + (long long)state_col);
      uint4 _vld_1[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_1[_blk] = _vptr_1[_blk];
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_direct_values[0 + _blk * 8 + _pair * 2])[0]),
                "=f"((&b_direct_values[0 + _blk * 8 + _pair * 2])[1])
              : "r"(_vpairs_1[_pair]));
        }
      }
    }
    {
      const uint4* _vptr_2 = reinterpret_cast<const uint4*>(C + c_base + (long long)state_col);
      uint4 _vld_2[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_2[_blk] = _vptr_2[_blk];
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_direct_values[0 + _blk * 8 + _pair * 2])[0]),
                "=f"((&c_direct_values[0 + _blk * 8 + _pair * 2])[1])
              : "r"(_vpairs_2[_pair]));
        }
      }
    }
  }
  float dt_lane = 0.0f;
  float decay_lane = 0.0f;
  float d_lane = 0.0f;
  if (lane == 0) {
    float dt_value = 0.0f;
    if (coefficient_bf16 != 0) {
      dt_value = (float)reinterpret_cast<__nv_bfloat16*>(
          dt_addr)[(long long)batch * dt_batch_stride + (long long)head * dt_head_stride];
      dt_value += (float)reinterpret_cast<__nv_bfloat16*>(
          dt_bias_addr)[(long long)head * dt_bias_head_stride];
      d_lane = (float)reinterpret_cast<__nv_bfloat16*>(d_addr)[(long long)head * d_head_stride];
    } else {
      dt_value = reinterpret_cast<float*>(
          dt_addr)[(long long)batch * dt_batch_stride + (long long)head * dt_head_stride];
      dt_value += reinterpret_cast<float*>(dt_bias_addr)[(long long)head * dt_bias_head_stride];
      d_lane = reinterpret_cast<float*>(d_addr)[(long long)head * d_head_stride];
    }
    float a_value = reinterpret_cast<float*>(a_addr)[(long long)head * a_head_stride];
    dt_lane = dt_value;
    {
      float _exp_2 = expf(a_value * dt_value);
      decay_lane = _exp_2;
    }
  }
  float _shfl_0 = __shfl_sync(0xFFFFFFFF, dt_lane, 0);
  float dt_value_1 = _shfl_0;
  float _shfl_1 = __shfl_sync(0xFFFFFFFF, decay_lane, 0);
  float decay = _shfl_1;
  float _shfl_2 = __shfl_sync(0xFFFFFFFF, d_lane, 0);
  float d_value = _shfl_2;
  for (int dim_group_base = dim_base + warp * 2; dim_group_base < dim_end; dim_group_base += 8) {
    int dim_index = dim_group_base + row_subgroup;
    long long row_index = (long long)(head * 128 + dim_index);
    long long x_index = batch_i64 * x_batch_stride + row_index;
    long long out_index = batch_i64 * out_batch_stride + row_index;
    float x_lane = 0.0f;
    float z_lane = 0.0f;
    if (row_member == 0) {
      x_lane = (float)x[x_index];
    }
    float _shfl_3 = __shfl_sync(0xFFFFFFFF, x_lane, row_subgroup * 16);
    float x_value = _shfl_3;
    float partial = 0.0f;
    {
      float direct_state_values[8];
#pragma unroll
      for (int element = 0; element < 8; element++) {
        direct_state_values[element] = 0.0f;
      }
      int row_offset_i32 = (head * 128 + dim_index) * 128 + state_col;
      unsigned long long row_offset = (unsigned long long)row_offset_i32;
      unsigned long long source_index =
          (unsigned long long)source_slot * state_stride_slot + row_offset;
      if (source_slot != pad_slot_id) {
        {
          unsigned _ldv8_3_0;
          unsigned _ldv8_3_1;
          unsigned _ldv8_3_2;
          unsigned _ldv8_3_3;
          unsigned _ldv8_3_4;
          unsigned _ldv8_3_5;
          unsigned _ldv8_3_6;
          unsigned _ldv8_3_7;
          asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                       : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3),
                         "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7)
                       : "l"((const void*)(state + (source_index)))
                       : "memory");
          direct_state_values[0 + 0] = __uint_as_float(_ldv8_3_0);
          direct_state_values[0 + 1] = __uint_as_float(_ldv8_3_1);
          direct_state_values[0 + 2] = __uint_as_float(_ldv8_3_2);
          direct_state_values[0 + 3] = __uint_as_float(_ldv8_3_3);
          direct_state_values[0 + 4] = __uint_as_float(_ldv8_3_4);
          direct_state_values[0 + 5] = __uint_as_float(_ldv8_3_5);
          direct_state_values[0 + 6] = __uint_as_float(_ldv8_3_6);
          direct_state_values[0 + 7] = __uint_as_float(_ldv8_3_7);
        }
      }
#pragma unroll
      for (int element_1 = 0; element_1 < 8; element_1++) {
        float d_b = b_direct_values[element_1] * dt_value_1;
        float decayed_state = direct_state_values[element_1] * decay;
        float new_state = decayed_state + d_b * x_value;
        direct_state_values[element_1] = new_state;
        partial += new_state * c_direct_values[element_1];
      }
      if (source_slot != pad_slot_id) {
        unsigned long long destination_index =
            (unsigned long long)source_slot * state_stride_slot + row_offset;
        {
          unsigned _stv8_4_0 = __float_as_uint(direct_state_values[0 + 0]);
          unsigned _stv8_4_1 = __float_as_uint(direct_state_values[0 + 1]);
          unsigned _stv8_4_2 = __float_as_uint(direct_state_values[0 + 2]);
          unsigned _stv8_4_3 = __float_as_uint(direct_state_values[0 + 3]);
          unsigned _stv8_4_4 = __float_as_uint(direct_state_values[0 + 4]);
          unsigned _stv8_4_5 = __float_as_uint(direct_state_values[0 + 5]);
          unsigned _stv8_4_6 = __float_as_uint(direct_state_values[0 + 6]);
          unsigned _stv8_4_7 = __float_as_uint(direct_state_values[0 + 7]);
          asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(
                           (void*)(state + (destination_index))),
                       "r"(_stv8_4_0), "r"(_stv8_4_1), "r"(_stv8_4_2), "r"(_stv8_4_3),
                       "r"(_stv8_4_4), "r"(_stv8_4_5), "r"(_stv8_4_6), "r"(_stv8_4_7)
                       : "memory");
        }
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
      float result = row_sum + d_value * x_lane;
      output[out_index] = result;
    }
  }
}

}  // extern "C"
