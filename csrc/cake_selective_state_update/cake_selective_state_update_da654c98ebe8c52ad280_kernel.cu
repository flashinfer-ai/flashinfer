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
#define SMEM_S_B_OFF 0
#define SMEM_S_B_STAGE_BYTES 1536
#define SMEM_S_B_STRIDE 1536
#define SMEM_S_C_OFF 1536
#define SMEM_S_C_STAGE_BYTES 1536
#define SMEM_S_C_STRIDE 1536
#define SMEM_S_X_OFF 3072
#define SMEM_S_X_STAGE_BYTES 192
#define SMEM_S_X_STRIDE 192
#define SMEM_S_DT_OFF 3264
#define SMEM_S_DT_STAGE_BYTES 24
#define SMEM_S_DT_STRIDE 24
#define SMEM_S_DECAY_OFF 3288
#define SMEM_S_DECAY_STAGE_BYTES 24
#define SMEM_S_DECAY_STRIDE 24
#define SMEM_S_STATE_OFF 3328
#define SMEM_S_STATE_STAGE_BYTES 4096
#define SMEM_S_STATE_STRIDE 4096
#define SMEM_TOTAL 7424
#define THREADS 128
#ifndef COEFFICIENT_BF16
#error \
    "COEFFICIENT_BF16 is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef INDEX_I32
#error "INDEX_I32 is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef NHEADS_STATIC
#error "NHEADS_STATIC is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef NGROUPS_STATIC
#error \
    "NGROUPS_STATIC is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef X_BATCH_STRIDE
#error \
    "X_BATCH_STRIDE is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef X_STEP_STRIDE
#error "X_STEP_STRIDE is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef DT_BATCH_STRIDE
#error \
    "DT_BATCH_STRIDE is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef DT_STEP_STRIDE
#error \
    "DT_STEP_STRIDE is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef B_BATCH_STRIDE
#error \
    "B_BATCH_STRIDE is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef B_STEP_STRIDE
#error "B_STEP_STRIDE is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef C_BATCH_STRIDE
#error \
    "C_BATCH_STRIDE is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef C_STEP_STRIDE
#error "C_STEP_STRIDE is a downstream specialization of this program; define it on the compile line"
#endif
#define LAUNCH_MIN_BLOCKS 7

#include <math_constants.h>

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
  float2 r;
  asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
      : "=l"(*(unsigned long long*)&r)
      : "l"(*(const unsigned long long*)&a), "l"(*(const unsigned long long*)&b),
        "l"(*(const unsigned long long*)&c));
  return r;
}

extern "C" {

__global__
__launch_bounds__(128, LAUNCH_MIN_BLOCKS) void kernel_cake_selective_state_update_da654c98ebe8c52ad280(
    __nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ intermediate_state, unsigned long long dt_addr,
    unsigned long long a_addr, unsigned long long d_addr, unsigned long long dt_bias_addr,
    unsigned long long state_batch_indices_addr, unsigned long long intermediate_state_indices_addr,
    int nheads, int ngroups, unsigned long long state_stride_slot,
    unsigned long long intermediate_stride_slot, long long pad_slot_id) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  __nv_bfloat16* s_B = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
  const int s_B_addr = smem + 0;
  __nv_bfloat16* s_C = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1536);
  const int s_C_addr = smem + 1536;
  __nv_bfloat16* s_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
  const int s_x_addr = smem + 3072;
  float* s_dt = reinterpret_cast<float*>(smem_raw + 3264);
  const int s_dt_addr = smem + 3264;
  float* s_decay = reinterpret_cast<float*>(smem_raw + 3288);
  const int s_decay_addr = smem + 3288;
  __nv_bfloat16* s_state = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3328);
  const int s_state_addr = smem + 3328;

  // === Task calls (dependency order) ===
  int batch = blockIdx.x;
  int head = blockIdx.y;
  int cta_z = blockIdx.z;
  int dim_base = cta_z * 16;
  long long batch_i64 = (long long)batch;
  long long head_i64 = (long long)head;
  int coefficient_bf16 = COEFFICIENT_BF16;
  int index_i32 = INDEX_I32;
  long long source_slot = pad_slot_id;
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
  } else {
    long long _vec_load_1[1];
    {
      asm("ld.global.nc.s64 %0, [%1];"
          : "=l"(_vec_load_1[0])
          : "l"((const void*)(reinterpret_cast<const long long*>(state_batch_indices_addr) +
                              (batch_i64))));
    }
    source_slot = _vec_load_1[0];
  }
  int runtime_nheads = (int)(NHEADS_STATIC == 0);
  int logical_nheads = NHEADS_STATIC + runtime_nheads * nheads;
  int runtime_ngroups = (int)(NGROUPS_STATIC == 0);
  int logical_ngroups = NGROUPS_STATIC + runtime_ngroups * ngroups;
  int heads_per_group = logical_nheads / logical_ngroups;
  int multi_group = 1 - (int)(NGROUPS_STATIC == 1);
  int group = head / heads_per_group * multi_group;
  long long group_i64 = (long long)group;
  long long b_batch_base = batch_i64 * (long long)B_BATCH_STRIDE;
  long long c_batch_base = batch_i64 * (long long)C_BATCH_STRIDE;
  long long x_batch_base = batch_i64 * (long long)X_BATCH_STRIDE;
  long long dt_batch_base = batch_i64 * (long long)DT_BATCH_STRIDE;
  int static_heads = 1 - runtime_nheads;
  unsigned long long source_state_stride =
      (unsigned long long)static_heads * (unsigned long long)(NHEADS_STATIC * 64 * 128) +
      (unsigned long long)runtime_nheads * state_stride_slot;
  unsigned long long intermediate_slot_stride =
      (unsigned long long)static_heads * (unsigned long long)(6 * NHEADS_STATIC * 64 * 128) +
      (unsigned long long)runtime_nheads * intermediate_stride_slot;
#pragma unroll
  for (int pack_turn = 0; pack_turn < 3; pack_turn++) {
    int pack = lane + pack_turn * 32;
    int step = pack / 16;
    int col = pack % 16 * 8;
    long long b_source_index = b_batch_base + (long long)step * (long long)B_STEP_STRIDE +
                               group_i64 * 128 + (long long)col;
    long long c_source_index = c_batch_base + (long long)step * (long long)C_STEP_STRIDE +
                               group_i64 * 128 + (long long)col;
    if (warp == 0) {
      asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                       s_B_addr + (unsigned int)((step * 128 + col) * 2)),
                   "l"(B + b_source_index));
    }
    if (warp == 1) {
      asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                       s_C_addr + (unsigned int)((step * 128 + col) * 2)),
                   "l"(C + c_source_index));
    }
  }
  for (int step_1 = warp; step_1 < 6; step_1 += 4) {
    for (int col_1 = lane * 8; col_1 < 16; col_1 += 256) {
      long long source_index = x_batch_base + (long long)step_1 * (long long)X_STEP_STRIDE +
                               head_i64 * 64 + (long long)(dim_base + col_1);
      asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                       s_x_addr + (unsigned int)((step_1 * 16 + col_1) * 2)),
                   "l"(x + source_index));
    }
  }
#pragma unroll
  for (int pack_turn_1 = 0; pack_turn_1 < 2; pack_turn_1++) {
    int flat_pack = tid + pack_turn_1 * 128;
    int row = flat_pack / 16;
    int pack_in_row = flat_pack % 16;
    int col_2 = pack_in_row * 8;
    int state_row = (head * 64 + dim_base + row) * 128 + col_2;
    if (source_slot != pad_slot_id) {
      unsigned long long state_index =
          (unsigned long long)source_slot * source_state_stride + (unsigned long long)state_row;
      asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                       s_state_addr + (unsigned int)((row * 128 + col_2) * 2)),
                   "l"(state + state_index));
    } else {
#pragma unroll
      for (int element = 0; element < 8; element++) {
        s_state[row * 128 + col_2 + element] = 0.0f;
      }
    }
  }
  float _vec_load_2[1];
  {
    uint32_t _scalar_bits_1;
    asm volatile("ld.global.nc.b32 %0, [%1];"
                 : "=r"(_scalar_bits_1)
                 : "l"((const void*)(reinterpret_cast<const float*>(a_addr) + (head_i64)))
                 : "memory");
    _vec_load_2[0] = __uint_as_float(_scalar_bits_1);
  }
  float a_value = _vec_load_2[0];
  if (tid < 6) {
    int step_2 = tid;
    float dt_value = 0.0f;
    long long dt_index = dt_batch_base + (long long)step_2 * (long long)DT_STEP_STRIDE + head_i64;
    if (coefficient_bf16 != 0) {
      float _vec_load_3[1];
      {
        uint32_t _bf16_bits_2;
        asm volatile(
            "ld.global.nc.u16 %0, [%1];"
            : "=r"(_bf16_bits_2)
            : "l"((const void*)(reinterpret_cast<const __nv_bfloat16*>(dt_addr) + (dt_index)))
            : "memory");
        _vec_load_3[0] = __uint_as_float(_bf16_bits_2 << 16);
      }
      float _vec_load_4[1];
      {
        uint32_t _bf16_bits_3;
        asm volatile(
            "ld.global.nc.u16 %0, [%1];"
            : "=r"(_bf16_bits_3)
            : "l"((const void*)(reinterpret_cast<const __nv_bfloat16*>(dt_bias_addr) + (head_i64)))
            : "memory");
        _vec_load_4[0] = __uint_as_float(_bf16_bits_3 << 16);
      }
      dt_value = _vec_load_3[0] + _vec_load_4[0];
    } else {
      float _vec_load_5[1];
      {
        uint32_t _scalar_bits_4;
        asm volatile("ld.global.nc.b32 %0, [%1];"
                     : "=r"(_scalar_bits_4)
                     : "l"((const void*)(reinterpret_cast<const float*>(dt_addr) + (dt_index)))
                     : "memory");
        _vec_load_5[0] = __uint_as_float(_scalar_bits_4);
      }
      float _vec_load_6[1];
      {
        uint32_t _scalar_bits_5;
        asm volatile("ld.global.nc.b32 %0, [%1];"
                     : "=r"(_scalar_bits_5)
                     : "l"((const void*)(reinterpret_cast<const float*>(dt_bias_addr) + (head_i64)))
                     : "memory");
        _vec_load_6[0] = __uint_as_float(_scalar_bits_5);
      }
      dt_value = _vec_load_5[0] + _vec_load_6[0];
    }
    if (dt_value <= 20.0f) {
      float _expf_0 = __expf(dt_value);
      float _log2_0;
      asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(1.0f + _expf_0));
      dt_value = _log2_0 * 0.6931471805599453f;
    }
    s_dt[step_2] = dt_value;
    {
      float _expf_1 = __expf(a_value * dt_value);
      s_decay[step_2] = _expf_1;
    }
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  int member = lane % 8;
  int subgroup = lane / 8;
  int local_row = warp * 4 + subgroup;
  float d_value = 0.0f;
  if (coefficient_bf16 != 0) {
    float _vec_load_7[1];
    {
      uint32_t _bf16_bits_6;
      asm volatile("ld.global.nc.u16 %0, [%1];"
                   : "=r"(_bf16_bits_6)
                   : "l"((const void*)(reinterpret_cast<const __nv_bfloat16*>(d_addr) + (head_i64)))
                   : "memory");
      _vec_load_7[0] = __uint_as_float(_bf16_bits_6 << 16);
    }
    d_value = _vec_load_7[0];
  } else {
    float _vec_load_8[1];
    {
      uint32_t _scalar_bits_7;
      asm volatile("ld.global.nc.b32 %0, [%1];"
                   : "=r"(_scalar_bits_7)
                   : "l"((const void*)(reinterpret_cast<const float*>(d_addr) + (head_i64)))
                   : "memory");
      _vec_load_8[0] = __uint_as_float(_scalar_bits_7);
    }
    d_value = _vec_load_8[0];
  }
  long long cache_slot = pad_slot_id;
  if (index_i32 != 0) {
    int _vec_load_9[1];
    {
      uint32_t _scalar_bits_8;
      asm volatile(
          "ld.global.nc.b32 %0, [%1];"
          : "=r"(_scalar_bits_8)
          : "l"((const void*)(reinterpret_cast<const int*>(intermediate_state_indices_addr) +
                              (batch_i64)))
          : "memory");
      _vec_load_9[0] = (int32_t)_scalar_bits_8;
    }
    cache_slot = (long long)_vec_load_9[0];
  } else {
    long long _vec_load_10[1];
    {
      asm("ld.global.nc.s64 %0, [%1];"
          : "=l"(_vec_load_10[0])
          : "l"((const void*)(reinterpret_cast<const long long*>(intermediate_state_indices_addr) +
                              (batch_i64))));
    }
    cache_slot = _vec_load_10[0];
  }
#pragma unroll
  for (int dim_pass = 0; dim_pass < 1; dim_pass++) {
    int dim_index = dim_base + dim_pass * 16 + local_row;
    float state_values[16];
    unsigned long long intermediate_step_stride = 0;
    unsigned long long intermediate_row_base = 0;
    {
      intermediate_step_stride = (unsigned long long)(logical_nheads * 64 * 128);
      intermediate_row_base = (unsigned long long)cache_slot * intermediate_slot_stride +
                              (unsigned long long)((head * 64 + dim_index) * 128);
    }
    unsigned int state_carrier[1];
    unsigned int b_carrier[4];
    unsigned int c_carrier[4];
    float state_pair_values[2];
    float b_values[8];
    float c_values[8];
#pragma unroll
    for (int tile = 0; tile < 2; tile++) {
      int state_col = tile * 8 * 8 + member * 8;
      {
#pragma unroll
        for (int pair = 0; pair < 4; pair++) {
          int state_element = local_row * 128 + state_col + pair * 2;
          asm volatile("ld.shared.b32 %0, [%1];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&state_carrier[0]))
                       : "r"(s_state_addr + (unsigned int)(state_element * 2)));
#pragma unroll
          for (int _pair = 0; _pair < 1; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&state_pair_values[_pair * 2])[0]), "=f"((&state_pair_values[_pair * 2])[1])
                : "r"(state_carrier[_pair]));
          }
          state_values[tile * 8 + pair * 2] = state_pair_values[0];
          state_values[tile * 8 + pair * 2 + 1] = state_pair_values[1];
        }
      }
    }
#pragma unroll
    for (int step_3 = 0; step_3 < 6; step_3++) {
      float dt_value_1 = s_dt[step_3];
      float decay = 0.0f;
      {
        decay = s_decay[step_3];
      }
      float x_value = (float)reinterpret_cast<const __nv_bfloat16*>(
          reinterpret_cast<const uint8_t*>(s_x) +
          ((step_3 * 16 + dim_pass * 16 + local_row) * 2))[0];
      float2 _f2_0 = make_float2(decay, decay);
      float2 decay_pair = _f2_0;
      float dtx_value = dt_value_1 * x_value;
      float2 _f2_1 = make_float2(dtx_value, dtx_value);
      float2 dtx_pair = _f2_1;
      float2 _f2_2 = make_float2(0.0f, 0.0f);
      float2 partial_pair = _f2_2;
#pragma unroll
      for (int tile_1 = 0; tile_1 < 2; tile_1++) {
        int state_col_1 = tile_1 * 8 * 8 + member * 8;
        int operand_index = step_3 * 128 + state_col_1;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&b_carrier[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carrier[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carrier[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carrier[(0) + 3]))
                     : "r"(s_B_addr + (unsigned int)(operand_index * 2)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&c_carrier[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carrier[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carrier[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carrier[(0) + 3]))
                     : "r"(s_C_addr + (unsigned int)(operand_index * 2)));
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carrier[_pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carrier[_pair]));
        }
#pragma unroll
        for (int pair_1 = 0; pair_1 < 4; pair_1++) {
          float2 _f2_3 = make_float2(state_values[tile_1 * 8 + pair_1 * 2],
                                     state_values[tile_1 * 8 + pair_1 * 2 + 1]);
          float2 state_pair = _f2_3;
          float2 _f2_4 = make_float2(b_values[pair_1 * 2], b_values[pair_1 * 2 + 1]);
          float2 b_pair = _f2_4;
          float2 _f2_5 = make_float2(c_values[pair_1 * 2], c_values[pair_1 * 2 + 1]);
          float2 c_pair = _f2_5;
          float2 _mul_f32x2_0;
          asm("mul.rn.ftz.f32x2 %0, %1, %2;"
              : "=l"(*(unsigned long long*)&_mul_f32x2_0)
              : "l"(*(const unsigned long long*)&b_pair),
                "l"(*(const unsigned long long*)&dtx_pair));
          float2 dbx_pair = _mul_f32x2_0;
          state_pair = fma_f32x2_rn_ftz(state_pair, decay_pair, dbx_pair);
          partial_pair = fma_f32x2_rn_ftz(state_pair, c_pair, partial_pair);
          state_values[tile_1 * 8 + pair_1 * 2] = state_pair.x;
          state_values[tile_1 * 8 + pair_1 * 2 + 1] = state_pair.y;
        }
      }
      float partial = partial_pair.x + partial_pair.y;
      float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, partial, 4);
      partial += _shfl_xor_0;
      float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, partial, 2);
      partial += _shfl_xor_1;
      float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, partial, 1);
      partial += _shfl_xor_2;
      if (member == 0) {
        int output_index = ((batch * 6 + step_3) * logical_nheads + head) * 64 + dim_index;
        output[output_index] = partial + d_value * x_value;
      }
      {
        if (source_slot != pad_slot_id) {
#pragma unroll
          for (int tile_2 = 0; tile_2 < 2; tile_2++) {
            int state_col_2 = tile_2 * 8 * 8 + member * 8;
            {
              {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(state_values[tile_2 * 8 + 0],
                                               state_values[tile_2 * 8 + 1]);
                _pk[1] = __floats2bfloat162_rn(state_values[tile_2 * 8 + 2],
                                               state_values[tile_2 * 8 + 3]);
                _pk[2] = __floats2bfloat162_rn(state_values[tile_2 * 8 + 4],
                                               state_values[tile_2 * 8 + 5]);
                _pk[3] = __floats2bfloat162_rn(state_values[tile_2 * 8 + 6],
                                               state_values[tile_2 * 8 + 7]);
                uint4 _st_v4_0 = *reinterpret_cast<uint4*>(&_pk[0]);
                asm volatile(
                    "st.global.L1::no_allocate.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(&(
                        (__nv_bfloat16*)(intermediate_state))[intermediate_row_base +
                                                              (unsigned long long)state_col_2 + 0]),
                    "r"(_st_v4_0.x), "r"(_st_v4_0.y), "r"(_st_v4_0.z), "r"(_st_v4_0.w)
                    : "memory");
              }
            }
          }
        }
        intermediate_row_base += intermediate_step_stride;
      }
    }
    if (dim_pass + 1 < 1) {
      __syncthreads();
#pragma unroll
      for (int pack_turn_2 = 0; pack_turn_2 < 2; pack_turn_2++) {
        int flat_pack_1 = tid + pack_turn_2 * 128;
        int row_1 = flat_pack_1 / 16;
        int pack_in_row_1 = flat_pack_1 % 16;
        int col_3 = pack_in_row_1 * 8;
        int next_dim_base = dim_base + (dim_pass + 1) * 16;
        int state_row_1 = (head * 64 + next_dim_base + row_1) * 128 + col_3;
        if (source_slot != pad_slot_id) {
          unsigned long long state_index_1 = (unsigned long long)source_slot * source_state_stride +
                                             (unsigned long long)state_row_1;
          asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                           s_state_addr + (unsigned int)((row_1 * 128 + col_3) * 2)),
                       "l"(state + state_index_1));
        } else {
#pragma unroll
          for (int element_1 = 0; element_1 < 8; element_1++) {
            s_state[row_1 * 128 + col_3 + element_1] = 0.0f;
          }
        }
      }
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 0;");
      __syncthreads();
    }
  }
}

}  // extern "C"
