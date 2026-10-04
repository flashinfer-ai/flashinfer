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
#define SMEM_S_B_STAGE_BYTES 512
#define SMEM_S_B_STRIDE 512
#define SMEM_S_C_OFF 512
#define SMEM_S_C_STAGE_BYTES 512
#define SMEM_S_C_STRIDE 512
#define SMEM_S_X_OFF 1024
#define SMEM_S_X_STAGE_BYTES 512
#define SMEM_S_X_STRIDE 512
#define SMEM_S_DT_OFF 1536
#define SMEM_S_DT_STAGE_BYTES 8
#define SMEM_S_DT_STRIDE 8
#define SMEM_TOTAL 1664
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float max_noftz(float a, float b) {
  float c;
  asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}

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

__global__ __launch_bounds__(128, 8) void kernel_cake_selective_state_update_a623bc5b6863d0333670(
    __nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ x, float* __restrict__ dt,
    float* __restrict__ A, __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
    float* __restrict__ D, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ output,
    long long* __restrict__ state_batch_indices, int batch_size, int nheads, int dim, int dstate,
    int ngroups, int token_steps, unsigned long long state_stride_slot, int dt_softplus,
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
  __nv_bfloat16* s_B = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
  const int s_B_addr = smem + 0;
  __nv_bfloat16* s_C = reinterpret_cast<__nv_bfloat16*>(smem_raw + 512);
  const int s_C_addr = smem + 512;
  __nv_bfloat16* s_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
  const int s_x_addr = smem + 1024;
  float* s_dt = reinterpret_cast<float*>(smem_raw + 1536);
  const int s_dt_addr = smem + 1536;

  // === Task calls (dependency order) ===
  int batch = bid / nheads;
  int head = bid % nheads;
  int heads_per_group = nheads / ngroups;
  int group = head / heads_per_group;
  int token_base = batch * token_steps;
  long long source_slot = state_batch_indices[batch];
  int pack = lane;
  int step = pack / 16;
  int col = pack % 16 * 8;
  int source_step = step;
  if (source_step >= token_steps) {
    source_step = 0;
  }
  int source_index = ((batch * token_steps + source_step) * ngroups + group) * dstate + col;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %0, 0;\n\t"
      "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
      "}" ::"r"((warp == 0) ? 1 : 0),
      "r"(s_B_addr + (unsigned int)((step * 128 + col) * 2)), "l"(B + source_index));
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %0, 0;\n\t"
      "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
      "}" ::"r"((warp == 1) ? 1 : 0),
      "r"(s_C_addr + (unsigned int)((step * 128 + col) * 2)), "l"(C + source_index));
  int step_0 = warp;
  int source_step_1 = step_0;
  if (source_step_1 >= token_steps) {
    source_step_1 = 0;
  }
  int col_2 = lane * 8;
  int source_index_3 = ((token_base + source_step_1) * nheads + head) * dim + col_2;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %0, 0;\n\t"
      "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
      "}" ::"r"((warp < 2 && lane < 16) ? 1 : 0),
      "r"(s_x_addr + (unsigned int)((source_step_1 * 128 + col_2) * 2)), "l"(x + source_index_3));
  asm volatile("cp.async.commit_group;");
  float dt_value = 0.0f;
  if (tid < token_steps) {
    int step_1 = tid;
    dt_value = dt[(token_base + step_1) * nheads + head];
    dt_value += dt_bias[head];
    if (dt_softplus != 0) {
      float _min_0 = fminf(dt_value, 20.0f);
      float _exp_0 = expf(_min_0);
      float _log1p_0 = log1pf(_exp_0);
      float _max_0 = max_noftz(_log1p_0, dt_value);
      dt_value = _max_0;
    }
  }
  asm volatile("cp.async.wait_group 0;");
  asm volatile("fence.proxy.async;");
  if (tid < token_steps) {
    s_dt[tid] = dt_value;
  }
  __syncthreads();
  int member = lane % 4;
  int row_in_warp = lane / 4;
  int local_row = warp * 8 + row_in_warp;
  float a_value = A[head];
  float d_value = D[head];
  float dt_value_0 = s_dt[0];
  float _exp_1 = expf(a_value * dt_value_0);
  float decay_0 = _exp_1;
  float dt_value_1 = 0.0f;
  float decay_1 = 0.0f;
  if (token_steps > 1) {
    dt_value_1 = s_dt[1];
    float _exp_2 = expf(a_value * dt_value_1);
    decay_1 = _exp_2;
  }
  unsigned int b_carriers[8];
  unsigned int c_carriers[8];
  float b_values[2];
  float c_values[2];
#pragma unroll 1
  for (int tile = 0; tile < 4; tile++) {
    int dim_index = tile * 32 + local_row;
    float state_values[32];
#pragma unroll
    for (int i = 0; i < 32; i++) {
      state_values[i] = 0.0f;
    }
    int row_offset_i32 = (head * dim + dim_index) * dstate;
    unsigned long long row_offset = (unsigned long long)row_offset_i32;
    if (source_slot != pad_slot_id) {
#pragma unroll
      for (int state_tile = 0; state_tile < 4; state_tile++) {
        int col_0 = state_tile * 32 + member * 8;
        unsigned long long state_index = (unsigned long long)source_slot * state_stride_slot +
                                         row_offset + (unsigned long long)col_0;
        {
          const uint4* _vptr_0 = reinterpret_cast<const uint4*>(state + state_index);
          uint4 _vld_0[1];
#pragma unroll
          for (int _blk = 0; _blk < 1; _blk++) {
            _vld_0[_blk] = _vptr_0[_blk];
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
#pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
              asm volatile(
                  "{\n\t"
                  "shl.b32 %0, %2, 16;\n\t"
                  "and.b32 %1, %2, 0xffff0000;\n\t"
                  "}\n"
                  : "=f"((&state_values[state_tile * 8 + _blk * 8 + _pair * 2])[0]),
                    "=f"((&state_values[state_tile * 8 + _blk * 8 + _pair * 2])[1])
                  : "r"(_vpairs_0[_pair]));
            }
          }
        }
      }
    }
#pragma unroll
    for (int pair_base = 0; pair_base < 2; pair_base += 2) {
      float x_value = s_x[pair_base * 128 + dim_index];
      float dtx = dt_value_0 * x_value;
      float2 _f2_0 = make_float2(decay_0, decay_0);
      float2 decay_pair = _f2_0;
      float2 _f2_1 = make_float2(dtx, dtx);
      float2 dtx_pair = _f2_1;
      float2 _f2_2 = make_float2(0.0f, 0.0f);
      float2 projection_pair_0 = _f2_2;
      float2 _f2_3 = make_float2(0.0f, 0.0f);
      float2 projection_pair_1 = _f2_3;
      int coefficient_col_0 = member * 8;
      int coefficient_col_1 = 32 + member * 8;
      int coefficient_index_0 = pair_base * 128 + coefficient_col_0;
      int coefficient_index_1 = pair_base * 128 + coefficient_col_1;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 3]))
                   : "r"(s_B_addr + (unsigned int)(coefficient_index_0 * 2)));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 3]))
                   : "r"(s_C_addr + (unsigned int)(coefficient_index_0 * 2)));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[4])),
                     "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 3]))
                   : "r"(s_B_addr + (unsigned int)(coefficient_index_1 * 2)));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[4])),
                     "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 3]))
                   : "r"(s_C_addr + (unsigned int)(coefficient_index_1 * 2)));
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[_pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[_pair]));
      }
      float2 _f2_4 = make_float2(state_values[0], state_values[1]);
      float2 _f2_5 = make_float2(b_values[0], b_values[1]);
      float2 _f2_6 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_0;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_0)
          : "l"(*(const unsigned long long*)&_f2_5), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair = fma_f32x2_rn_ftz(_f2_4, decay_pair, _mul_f32x2_0);
      float2 next_projection_pair = fma_f32x2_rn_ftz(next_state_pair, _f2_6, projection_pair_0);
      state_values[0] = next_state_pair.x;
      state_values[1] = next_state_pair.y;
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[4 + _pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[4 + _pair]));
      }
      float2 _f2_7 = make_float2(state_values[8], state_values[9]);
      float2 _f2_8 = make_float2(b_values[0], b_values[1]);
      float2 _f2_9 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_1;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_1)
          : "l"(*(const unsigned long long*)&_f2_8), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_0 = fma_f32x2_rn_ftz(_f2_7, decay_pair, _mul_f32x2_1);
      float2 next_projection_pair_1 = fma_f32x2_rn_ftz(next_state_pair_0, _f2_9, projection_pair_1);
      state_values[8] = next_state_pair_0.x;
      state_values[9] = next_state_pair_0.y;
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[1 + _pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[1 + _pair]));
      }
      float2 _f2_10 = make_float2(state_values[2], state_values[3]);
      float2 _f2_11 = make_float2(b_values[0], b_values[1]);
      float2 _f2_12 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_2;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_2)
          : "l"(*(const unsigned long long*)&_f2_11), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_2 = fma_f32x2_rn_ftz(_f2_10, decay_pair, _mul_f32x2_2);
      float2 next_projection_pair_3 =
          fma_f32x2_rn_ftz(next_state_pair_2, _f2_12, next_projection_pair);
      state_values[2] = next_state_pair_2.x;
      state_values[3] = next_state_pair_2.y;
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[5 + _pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[5 + _pair]));
      }
      float2 _f2_13 = make_float2(state_values[10], state_values[11]);
      float2 _f2_14 = make_float2(b_values[0], b_values[1]);
      float2 _f2_15 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_3;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_3)
          : "l"(*(const unsigned long long*)&_f2_14), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_4 = fma_f32x2_rn_ftz(_f2_13, decay_pair, _mul_f32x2_3);
      float2 next_projection_pair_5 =
          fma_f32x2_rn_ftz(next_state_pair_4, _f2_15, next_projection_pair_1);
      state_values[10] = next_state_pair_4.x;
      state_values[11] = next_state_pair_4.y;
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[2 + _pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[2 + _pair]));
      }
      float2 _f2_16 = make_float2(state_values[4], state_values[5]);
      float2 _f2_17 = make_float2(b_values[0], b_values[1]);
      float2 _f2_18 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_4;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_4)
          : "l"(*(const unsigned long long*)&_f2_17), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_6 = fma_f32x2_rn_ftz(_f2_16, decay_pair, _mul_f32x2_4);
      float2 next_projection_pair_7 =
          fma_f32x2_rn_ftz(next_state_pair_6, _f2_18, next_projection_pair_3);
      state_values[4] = next_state_pair_6.x;
      state_values[5] = next_state_pair_6.y;
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[6 + _pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[6 + _pair]));
      }
      float2 _f2_19 = make_float2(state_values[12], state_values[13]);
      float2 _f2_20 = make_float2(b_values[0], b_values[1]);
      float2 _f2_21 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_5;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_5)
          : "l"(*(const unsigned long long*)&_f2_20), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_8 = fma_f32x2_rn_ftz(_f2_19, decay_pair, _mul_f32x2_5);
      float2 next_projection_pair_9 =
          fma_f32x2_rn_ftz(next_state_pair_8, _f2_21, next_projection_pair_5);
      state_values[12] = next_state_pair_8.x;
      state_values[13] = next_state_pair_8.y;
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[3 + _pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[3 + _pair]));
      }
      float2 _f2_22 = make_float2(state_values[6], state_values[7]);
      float2 _f2_23 = make_float2(b_values[0], b_values[1]);
      float2 _f2_24 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_6;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_6)
          : "l"(*(const unsigned long long*)&_f2_23), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_10 = fma_f32x2_rn_ftz(_f2_22, decay_pair, _mul_f32x2_6);
      float2 next_projection_pair_11 =
          fma_f32x2_rn_ftz(next_state_pair_10, _f2_24, next_projection_pair_7);
      state_values[6] = next_state_pair_10.x;
      state_values[7] = next_state_pair_10.y;
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[7 + _pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[7 + _pair]));
      }
      float2 _f2_25 = make_float2(state_values[14], state_values[15]);
      float2 _f2_26 = make_float2(b_values[0], b_values[1]);
      float2 _f2_27 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_7;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_7)
          : "l"(*(const unsigned long long*)&_f2_26), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_12 = fma_f32x2_rn_ftz(_f2_25, decay_pair, _mul_f32x2_7);
      float2 next_projection_pair_13 =
          fma_f32x2_rn_ftz(next_state_pair_12, _f2_27, next_projection_pair_9);
      state_values[14] = next_state_pair_12.x;
      state_values[15] = next_state_pair_12.y;
      projection_pair_0 = next_projection_pair_11;
      projection_pair_1 = next_projection_pair_13;
      int coefficient_col_0_14 = 64 + member * 8;
      int coefficient_col_1_15 = 96 + member * 8;
      int coefficient_index_0_16 = pair_base * 128 + coefficient_col_0_14;
      int coefficient_index_1_17 = pair_base * 128 + coefficient_col_1_15;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 3]))
                   : "r"(s_B_addr + (unsigned int)(coefficient_index_0_16 * 2)));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 3]))
                   : "r"(s_C_addr + (unsigned int)(coefficient_index_0_16 * 2)));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[4])),
                     "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 3]))
                   : "r"(s_B_addr + (unsigned int)(coefficient_index_1_17 * 2)));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[4])),
                     "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 3]))
                   : "r"(s_C_addr + (unsigned int)(coefficient_index_1_17 * 2)));
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[_pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[_pair]));
      }
      float2 _f2_28 = make_float2(state_values[16], state_values[17]);
      float2 _f2_29 = make_float2(b_values[0], b_values[1]);
      float2 _f2_30 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_8;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_8)
          : "l"(*(const unsigned long long*)&_f2_29), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_18 = fma_f32x2_rn_ftz(_f2_28, decay_pair, _mul_f32x2_8);
      float2 next_projection_pair_19 =
          fma_f32x2_rn_ftz(next_state_pair_18, _f2_30, projection_pair_0);
      state_values[16] = next_state_pair_18.x;
      state_values[17] = next_state_pair_18.y;
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[4 + _pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[4 + _pair]));
      }
      float2 _f2_31 = make_float2(state_values[24], state_values[25]);
      float2 _f2_32 = make_float2(b_values[0], b_values[1]);
      float2 _f2_33 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_9;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_9)
          : "l"(*(const unsigned long long*)&_f2_32), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_20 = fma_f32x2_rn_ftz(_f2_31, decay_pair, _mul_f32x2_9);
      float2 next_projection_pair_21 =
          fma_f32x2_rn_ftz(next_state_pair_20, _f2_33, projection_pair_1);
      state_values[24] = next_state_pair_20.x;
      state_values[25] = next_state_pair_20.y;
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[1 + _pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[1 + _pair]));
      }
      float2 _f2_34 = make_float2(state_values[18], state_values[19]);
      float2 _f2_35 = make_float2(b_values[0], b_values[1]);
      float2 _f2_36 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_10;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_10)
          : "l"(*(const unsigned long long*)&_f2_35), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_22 = fma_f32x2_rn_ftz(_f2_34, decay_pair, _mul_f32x2_10);
      float2 next_projection_pair_23 =
          fma_f32x2_rn_ftz(next_state_pair_22, _f2_36, next_projection_pair_19);
      state_values[18] = next_state_pair_22.x;
      state_values[19] = next_state_pair_22.y;
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[5 + _pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[5 + _pair]));
      }
      float2 _f2_37 = make_float2(state_values[26], state_values[27]);
      float2 _f2_38 = make_float2(b_values[0], b_values[1]);
      float2 _f2_39 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_11;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_11)
          : "l"(*(const unsigned long long*)&_f2_38), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_24 = fma_f32x2_rn_ftz(_f2_37, decay_pair, _mul_f32x2_11);
      float2 next_projection_pair_25 =
          fma_f32x2_rn_ftz(next_state_pair_24, _f2_39, next_projection_pair_21);
      state_values[26] = next_state_pair_24.x;
      state_values[27] = next_state_pair_24.y;
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[2 + _pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[2 + _pair]));
      }
      float2 _f2_40 = make_float2(state_values[20], state_values[21]);
      float2 _f2_41 = make_float2(b_values[0], b_values[1]);
      float2 _f2_42 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_12;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_12)
          : "l"(*(const unsigned long long*)&_f2_41), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_26 = fma_f32x2_rn_ftz(_f2_40, decay_pair, _mul_f32x2_12);
      float2 next_projection_pair_27 =
          fma_f32x2_rn_ftz(next_state_pair_26, _f2_42, next_projection_pair_23);
      state_values[20] = next_state_pair_26.x;
      state_values[21] = next_state_pair_26.y;
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[6 + _pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[6 + _pair]));
      }
      float2 _f2_43 = make_float2(state_values[28], state_values[29]);
      float2 _f2_44 = make_float2(b_values[0], b_values[1]);
      float2 _f2_45 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_13;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_13)
          : "l"(*(const unsigned long long*)&_f2_44), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_28 = fma_f32x2_rn_ftz(_f2_43, decay_pair, _mul_f32x2_13);
      float2 next_projection_pair_29 =
          fma_f32x2_rn_ftz(next_state_pair_28, _f2_45, next_projection_pair_25);
      state_values[28] = next_state_pair_28.x;
      state_values[29] = next_state_pair_28.y;
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[3 + _pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[3 + _pair]));
      }
      float2 _f2_46 = make_float2(state_values[22], state_values[23]);
      float2 _f2_47 = make_float2(b_values[0], b_values[1]);
      float2 _f2_48 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_14;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_14)
          : "l"(*(const unsigned long long*)&_f2_47), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_30 = fma_f32x2_rn_ftz(_f2_46, decay_pair, _mul_f32x2_14);
      float2 next_projection_pair_31 =
          fma_f32x2_rn_ftz(next_state_pair_30, _f2_48, next_projection_pair_27);
      state_values[22] = next_state_pair_30.x;
      state_values[23] = next_state_pair_30.y;
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[7 + _pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[7 + _pair]));
      }
      float2 _f2_49 = make_float2(state_values[30], state_values[31]);
      float2 _f2_50 = make_float2(b_values[0], b_values[1]);
      float2 _f2_51 = make_float2(c_values[0], c_values[1]);
      float2 _mul_f32x2_15;
      asm("mul.rn.ftz.f32x2 %0, %1, %2;"
          : "=l"(*(unsigned long long*)&_mul_f32x2_15)
          : "l"(*(const unsigned long long*)&_f2_50), "l"(*(const unsigned long long*)&dtx_pair));
      float2 next_state_pair_32 = fma_f32x2_rn_ftz(_f2_49, decay_pair, _mul_f32x2_15);
      float2 next_projection_pair_33 =
          fma_f32x2_rn_ftz(next_state_pair_32, _f2_51, next_projection_pair_29);
      state_values[30] = next_state_pair_32.x;
      state_values[31] = next_state_pair_32.y;
      projection_pair_0 = next_projection_pair_31;
      projection_pair_1 = next_projection_pair_33;
      float virtual_projection_0 = projection_pair_0.x + projection_pair_0.y;
      float virtual_projection_1 = projection_pair_1.x + projection_pair_1.y;
      float out_value = virtual_projection_0 + virtual_projection_1;
      float _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, out_value, 2, 4);
      out_value += _shfl_down_0;
      float _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, out_value, 1, 4);
      out_value += _shfl_down_1;
      if (member == 0) {
        int output_index = ((token_base + pair_base) * nheads + head) * dim + dim_index;
        output[output_index] = out_value + d_value * x_value;
      }
      if (pair_base + 1 >= token_steps) {
        if (source_slot != pad_slot_id) {
          int publication_row_offset_i32 = (head * dim + dim_index) * dstate;
          unsigned long long publication_row_offset =
              (unsigned long long)publication_row_offset_i32;
          int col_0_1 = member * 8;
          unsigned long long destination_index =
              (unsigned long long)source_slot * state_stride_slot + publication_row_offset +
              (unsigned long long)col_0_1;
          {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(state_values[0 + 0], state_values[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(state_values[0 + 2], state_values[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(state_values[0 + 4], state_values[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(state_values[0 + 6], state_values[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[destination_index + 0]) =
                *reinterpret_cast<uint4*>(&_pk[0]);
          }
          int col_1 = 32 + member * 8;
          unsigned long long destination_index_2 =
              (unsigned long long)source_slot * state_stride_slot + publication_row_offset +
              (unsigned long long)col_1;
          {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(state_values[8 + 0], state_values[8 + 1]);
            _pk[1] = __floats2bfloat162_rn(state_values[8 + 2], state_values[8 + 3]);
            _pk[2] = __floats2bfloat162_rn(state_values[8 + 4], state_values[8 + 5]);
            _pk[3] = __floats2bfloat162_rn(state_values[8 + 6], state_values[8 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[destination_index_2 + 0]) =
                *reinterpret_cast<uint4*>(&_pk[0]);
          }
          int col_3 = 64 + member * 8;
          unsigned long long destination_index_4 =
              (unsigned long long)source_slot * state_stride_slot + publication_row_offset +
              (unsigned long long)col_3;
          {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(state_values[16 + 0], state_values[16 + 1]);
            _pk[1] = __floats2bfloat162_rn(state_values[16 + 2], state_values[16 + 3]);
            _pk[2] = __floats2bfloat162_rn(state_values[16 + 4], state_values[16 + 5]);
            _pk[3] = __floats2bfloat162_rn(state_values[16 + 6], state_values[16 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[destination_index_4 + 0]) =
                *reinterpret_cast<uint4*>(&_pk[0]);
          }
          int col_5 = 96 + member * 8;
          unsigned long long destination_index_6 =
              (unsigned long long)source_slot * state_stride_slot + publication_row_offset +
              (unsigned long long)col_5;
          {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(state_values[24 + 0], state_values[24 + 1]);
            _pk[1] = __floats2bfloat162_rn(state_values[24 + 2], state_values[24 + 3]);
            _pk[2] = __floats2bfloat162_rn(state_values[24 + 4], state_values[24 + 5]);
            _pk[3] = __floats2bfloat162_rn(state_values[24 + 6], state_values[24 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[destination_index_6 + 0]) =
                *reinterpret_cast<uint4*>(&_pk[0]);
          }
        }
      }
      if (pair_base + 1 < token_steps) {
        float x_value_0 = s_x[(pair_base + 1) * 128 + dim_index];
        float dtx_1 = dt_value_1 * x_value_0;
        float2 _f2_52 = make_float2(decay_1, decay_1);
        float2 decay_pair_2 = _f2_52;
        float2 _f2_53 = make_float2(dtx_1, dtx_1);
        float2 dtx_pair_3 = _f2_53;
        float2 _f2_54 = make_float2(0.0f, 0.0f);
        float2 projection_pair_0_4 = _f2_54;
        float2 _f2_55 = make_float2(0.0f, 0.0f);
        float2 projection_pair_1_5 = _f2_55;
        int coefficient_col_0_6 = member * 8;
        int coefficient_col_1_7 = 32 + member * 8;
        int coefficient_index_0_8 = (pair_base + 1) * 128 + coefficient_col_0_6;
        int coefficient_index_1_9 = (pair_base + 1) * 128 + coefficient_col_1_7;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 3]))
                     : "r"(s_B_addr + (unsigned int)(coefficient_index_0_8 * 2)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 3]))
                     : "r"(s_C_addr + (unsigned int)(coefficient_index_0_8 * 2)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[4])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 3]))
                     : "r"(s_B_addr + (unsigned int)(coefficient_index_1_9 * 2)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[4])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 3]))
                     : "r"(s_C_addr + (unsigned int)(coefficient_index_1_9 * 2)));
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[_pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[_pair]));
        }
        float2 _f2_56 = make_float2(state_values[0], state_values[1]);
        float2 _f2_57 = make_float2(b_values[0], b_values[1]);
        float2 _f2_58 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_16;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_16)
            : "l"(*(const unsigned long long*)&_f2_57),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_11 = fma_f32x2_rn_ftz(_f2_56, decay_pair_2, _mul_f32x2_16);
        float2 next_projection_pair_12 =
            fma_f32x2_rn_ftz(next_state_pair_11, _f2_58, projection_pair_0_4);
        state_values[0] = next_state_pair_11.x;
        state_values[1] = next_state_pair_11.y;
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[4 + _pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[4 + _pair]));
        }
        float2 _f2_59 = make_float2(state_values[8], state_values[9]);
        float2 _f2_60 = make_float2(b_values[0], b_values[1]);
        float2 _f2_61 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_17;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_17)
            : "l"(*(const unsigned long long*)&_f2_60),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_13 = fma_f32x2_rn_ftz(_f2_59, decay_pair_2, _mul_f32x2_17);
        float2 next_projection_pair_14 =
            fma_f32x2_rn_ftz(next_state_pair_13, _f2_61, projection_pair_1_5);
        state_values[8] = next_state_pair_13.x;
        state_values[9] = next_state_pair_13.y;
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[1 + _pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[1 + _pair]));
        }
        float2 _f2_62 = make_float2(state_values[2], state_values[3]);
        float2 _f2_63 = make_float2(b_values[0], b_values[1]);
        float2 _f2_64 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_18;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_18)
            : "l"(*(const unsigned long long*)&_f2_63),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_15 = fma_f32x2_rn_ftz(_f2_62, decay_pair_2, _mul_f32x2_18);
        float2 next_projection_pair_16 =
            fma_f32x2_rn_ftz(next_state_pair_15, _f2_64, next_projection_pair_12);
        state_values[2] = next_state_pair_15.x;
        state_values[3] = next_state_pair_15.y;
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[5 + _pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[5 + _pair]));
        }
        float2 _f2_65 = make_float2(state_values[10], state_values[11]);
        float2 _f2_66 = make_float2(b_values[0], b_values[1]);
        float2 _f2_67 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_19;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_19)
            : "l"(*(const unsigned long long*)&_f2_66),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_17 = fma_f32x2_rn_ftz(_f2_65, decay_pair_2, _mul_f32x2_19);
        float2 next_projection_pair_18 =
            fma_f32x2_rn_ftz(next_state_pair_17, _f2_67, next_projection_pair_14);
        state_values[10] = next_state_pair_17.x;
        state_values[11] = next_state_pair_17.y;
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[2 + _pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[2 + _pair]));
        }
        float2 _f2_68 = make_float2(state_values[4], state_values[5]);
        float2 _f2_69 = make_float2(b_values[0], b_values[1]);
        float2 _f2_70 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_20;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_20)
            : "l"(*(const unsigned long long*)&_f2_69),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_19 = fma_f32x2_rn_ftz(_f2_68, decay_pair_2, _mul_f32x2_20);
        float2 next_projection_pair_20 =
            fma_f32x2_rn_ftz(next_state_pair_19, _f2_70, next_projection_pair_16);
        state_values[4] = next_state_pair_19.x;
        state_values[5] = next_state_pair_19.y;
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[6 + _pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[6 + _pair]));
        }
        float2 _f2_71 = make_float2(state_values[12], state_values[13]);
        float2 _f2_72 = make_float2(b_values[0], b_values[1]);
        float2 _f2_73 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_21;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_21)
            : "l"(*(const unsigned long long*)&_f2_72),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_21 = fma_f32x2_rn_ftz(_f2_71, decay_pair_2, _mul_f32x2_21);
        float2 next_projection_pair_22 =
            fma_f32x2_rn_ftz(next_state_pair_21, _f2_73, next_projection_pair_18);
        state_values[12] = next_state_pair_21.x;
        state_values[13] = next_state_pair_21.y;
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[3 + _pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[3 + _pair]));
        }
        float2 _f2_74 = make_float2(state_values[6], state_values[7]);
        float2 _f2_75 = make_float2(b_values[0], b_values[1]);
        float2 _f2_76 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_22;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_22)
            : "l"(*(const unsigned long long*)&_f2_75),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_23 = fma_f32x2_rn_ftz(_f2_74, decay_pair_2, _mul_f32x2_22);
        float2 next_projection_pair_24 =
            fma_f32x2_rn_ftz(next_state_pair_23, _f2_76, next_projection_pair_20);
        state_values[6] = next_state_pair_23.x;
        state_values[7] = next_state_pair_23.y;
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[7 + _pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[7 + _pair]));
        }
        float2 _f2_77 = make_float2(state_values[14], state_values[15]);
        float2 _f2_78 = make_float2(b_values[0], b_values[1]);
        float2 _f2_79 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_23;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_23)
            : "l"(*(const unsigned long long*)&_f2_78),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_25 = fma_f32x2_rn_ftz(_f2_77, decay_pair_2, _mul_f32x2_23);
        float2 next_projection_pair_26 =
            fma_f32x2_rn_ftz(next_state_pair_25, _f2_79, next_projection_pair_22);
        state_values[14] = next_state_pair_25.x;
        state_values[15] = next_state_pair_25.y;
        projection_pair_0_4 = next_projection_pair_24;
        projection_pair_1_5 = next_projection_pair_26;
        int coefficient_col_0_27 = 64 + member * 8;
        int coefficient_col_1_28 = 96 + member * 8;
        int coefficient_index_0_29 = (pair_base + 1) * 128 + coefficient_col_0_27;
        int coefficient_index_1_30 = (pair_base + 1) * 128 + coefficient_col_1_28;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 3]))
                     : "r"(s_B_addr + (unsigned int)(coefficient_index_0_29 * 2)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 3]))
                     : "r"(s_C_addr + (unsigned int)(coefficient_index_0_29 * 2)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[4])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 3]))
                     : "r"(s_B_addr + (unsigned int)(coefficient_index_1_30 * 2)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[4])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 3]))
                     : "r"(s_C_addr + (unsigned int)(coefficient_index_1_30 * 2)));
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[_pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[_pair]));
        }
        float2 _f2_80 = make_float2(state_values[16], state_values[17]);
        float2 _f2_81 = make_float2(b_values[0], b_values[1]);
        float2 _f2_82 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_24;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_24)
            : "l"(*(const unsigned long long*)&_f2_81),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_31 = fma_f32x2_rn_ftz(_f2_80, decay_pair_2, _mul_f32x2_24);
        float2 next_projection_pair_32 =
            fma_f32x2_rn_ftz(next_state_pair_31, _f2_82, projection_pair_0_4);
        state_values[16] = next_state_pair_31.x;
        state_values[17] = next_state_pair_31.y;
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[4 + _pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[4 + _pair]));
        }
        float2 _f2_83 = make_float2(state_values[24], state_values[25]);
        float2 _f2_84 = make_float2(b_values[0], b_values[1]);
        float2 _f2_85 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_25;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_25)
            : "l"(*(const unsigned long long*)&_f2_84),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_33 = fma_f32x2_rn_ftz(_f2_83, decay_pair_2, _mul_f32x2_25);
        float2 next_projection_pair_34 =
            fma_f32x2_rn_ftz(next_state_pair_33, _f2_85, projection_pair_1_5);
        state_values[24] = next_state_pair_33.x;
        state_values[25] = next_state_pair_33.y;
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[1 + _pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[1 + _pair]));
        }
        float2 _f2_86 = make_float2(state_values[18], state_values[19]);
        float2 _f2_87 = make_float2(b_values[0], b_values[1]);
        float2 _f2_88 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_26;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_26)
            : "l"(*(const unsigned long long*)&_f2_87),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_35 = fma_f32x2_rn_ftz(_f2_86, decay_pair_2, _mul_f32x2_26);
        float2 next_projection_pair_36 =
            fma_f32x2_rn_ftz(next_state_pair_35, _f2_88, next_projection_pair_32);
        state_values[18] = next_state_pair_35.x;
        state_values[19] = next_state_pair_35.y;
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[5 + _pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[5 + _pair]));
        }
        float2 _f2_89 = make_float2(state_values[26], state_values[27]);
        float2 _f2_90 = make_float2(b_values[0], b_values[1]);
        float2 _f2_91 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_27;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_27)
            : "l"(*(const unsigned long long*)&_f2_90),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_37 = fma_f32x2_rn_ftz(_f2_89, decay_pair_2, _mul_f32x2_27);
        float2 next_projection_pair_38 =
            fma_f32x2_rn_ftz(next_state_pair_37, _f2_91, next_projection_pair_34);
        state_values[26] = next_state_pair_37.x;
        state_values[27] = next_state_pair_37.y;
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[2 + _pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[2 + _pair]));
        }
        float2 _f2_92 = make_float2(state_values[20], state_values[21]);
        float2 _f2_93 = make_float2(b_values[0], b_values[1]);
        float2 _f2_94 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_28;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_28)
            : "l"(*(const unsigned long long*)&_f2_93),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_39 = fma_f32x2_rn_ftz(_f2_92, decay_pair_2, _mul_f32x2_28);
        float2 next_projection_pair_40 =
            fma_f32x2_rn_ftz(next_state_pair_39, _f2_94, next_projection_pair_36);
        state_values[20] = next_state_pair_39.x;
        state_values[21] = next_state_pair_39.y;
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[6 + _pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[6 + _pair]));
        }
        float2 _f2_95 = make_float2(state_values[28], state_values[29]);
        float2 _f2_96 = make_float2(b_values[0], b_values[1]);
        float2 _f2_97 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_29;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_29)
            : "l"(*(const unsigned long long*)&_f2_96),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_41 = fma_f32x2_rn_ftz(_f2_95, decay_pair_2, _mul_f32x2_29);
        float2 next_projection_pair_42 =
            fma_f32x2_rn_ftz(next_state_pair_41, _f2_97, next_projection_pair_38);
        state_values[28] = next_state_pair_41.x;
        state_values[29] = next_state_pair_41.y;
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[3 + _pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[3 + _pair]));
        }
        float2 _f2_98 = make_float2(state_values[22], state_values[23]);
        float2 _f2_99 = make_float2(b_values[0], b_values[1]);
        float2 _f2_100 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_30;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_30)
            : "l"(*(const unsigned long long*)&_f2_99),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_43 = fma_f32x2_rn_ftz(_f2_98, decay_pair_2, _mul_f32x2_30);
        float2 next_projection_pair_44 =
            fma_f32x2_rn_ftz(next_state_pair_43, _f2_100, next_projection_pair_40);
        state_values[22] = next_state_pair_43.x;
        state_values[23] = next_state_pair_43.y;
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
              : "r"(b_carriers[7 + _pair]));
        }
#pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
          asm volatile(
              "{\n\t"
              "shl.b32 %0, %2, 16;\n\t"
              "and.b32 %1, %2, 0xffff0000;\n\t"
              "}\n"
              : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
              : "r"(c_carriers[7 + _pair]));
        }
        float2 _f2_101 = make_float2(state_values[30], state_values[31]);
        float2 _f2_102 = make_float2(b_values[0], b_values[1]);
        float2 _f2_103 = make_float2(c_values[0], c_values[1]);
        float2 _mul_f32x2_31;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&_mul_f32x2_31)
            : "l"(*(const unsigned long long*)&_f2_102),
              "l"(*(const unsigned long long*)&dtx_pair_3));
        float2 next_state_pair_45 = fma_f32x2_rn_ftz(_f2_101, decay_pair_2, _mul_f32x2_31);
        float2 next_projection_pair_46 =
            fma_f32x2_rn_ftz(next_state_pair_45, _f2_103, next_projection_pair_42);
        state_values[30] = next_state_pair_45.x;
        state_values[31] = next_state_pair_45.y;
        projection_pair_0_4 = next_projection_pair_44;
        projection_pair_1_5 = next_projection_pair_46;
        float virtual_projection_0_47 = projection_pair_0_4.x + projection_pair_0_4.y;
        float virtual_projection_1_48 = projection_pair_1_5.x + projection_pair_1_5.y;
        float out_value_49 = virtual_projection_0_47 + virtual_projection_1_48;
        float _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, out_value_49, 2, 4);
        out_value_49 += _shfl_down_2;
        float _shfl_down_3 = __shfl_down_sync(0xFFFFFFFF, out_value_49, 1, 4);
        out_value_49 += _shfl_down_3;
        if (member == 0) {
          int output_index_1 = ((token_base + pair_base + 1) * nheads + head) * dim + dim_index;
          output[output_index_1] = out_value_49 + d_value * x_value_0;
        }
        if (source_slot != pad_slot_id) {
          int publication_row_offset_i32_1 = (head * dim + dim_index) * dstate;
          unsigned long long publication_row_offset_1 =
              (unsigned long long)publication_row_offset_i32_1;
          int col_0_2 = member * 8;
          unsigned long long destination_index_1 =
              (unsigned long long)source_slot * state_stride_slot + publication_row_offset_1 +
              (unsigned long long)col_0_2;
          {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(state_values[0 + 0], state_values[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(state_values[0 + 2], state_values[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(state_values[0 + 4], state_values[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(state_values[0 + 6], state_values[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[destination_index_1 + 0]) =
                *reinterpret_cast<uint4*>(&_pk[0]);
          }
          int col_1_1 = 32 + member * 8;
          unsigned long long destination_index_2_1 =
              (unsigned long long)source_slot * state_stride_slot + publication_row_offset_1 +
              (unsigned long long)col_1_1;
          {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(state_values[8 + 0], state_values[8 + 1]);
            _pk[1] = __floats2bfloat162_rn(state_values[8 + 2], state_values[8 + 3]);
            _pk[2] = __floats2bfloat162_rn(state_values[8 + 4], state_values[8 + 5]);
            _pk[3] = __floats2bfloat162_rn(state_values[8 + 6], state_values[8 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[destination_index_2_1 + 0]) =
                *reinterpret_cast<uint4*>(&_pk[0]);
          }
          int col_3_1 = 64 + member * 8;
          unsigned long long destination_index_4_1 =
              (unsigned long long)source_slot * state_stride_slot + publication_row_offset_1 +
              (unsigned long long)col_3_1;
          {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(state_values[16 + 0], state_values[16 + 1]);
            _pk[1] = __floats2bfloat162_rn(state_values[16 + 2], state_values[16 + 3]);
            _pk[2] = __floats2bfloat162_rn(state_values[16 + 4], state_values[16 + 5]);
            _pk[3] = __floats2bfloat162_rn(state_values[16 + 6], state_values[16 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[destination_index_4_1 + 0]) =
                *reinterpret_cast<uint4*>(&_pk[0]);
          }
          int col_5_1 = 96 + member * 8;
          unsigned long long destination_index_6_1 =
              (unsigned long long)source_slot * state_stride_slot + publication_row_offset_1 +
              (unsigned long long)col_5_1;
          {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(state_values[24 + 0], state_values[24 + 1]);
            _pk[1] = __floats2bfloat162_rn(state_values[24 + 2], state_values[24 + 3]);
            _pk[2] = __floats2bfloat162_rn(state_values[24 + 4], state_values[24 + 5]);
            _pk[3] = __floats2bfloat162_rn(state_values[24 + 6], state_values[24 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[destination_index_6_1 + 0]) =
                *reinterpret_cast<uint4*>(&_pk[0]);
          }
        }
      }
    }
  }
}

}  // extern "C"
