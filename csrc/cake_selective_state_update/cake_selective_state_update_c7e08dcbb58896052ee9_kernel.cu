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
#define NUM_STATE_PIPE_STAGES 4
#define SMEM_S_STATE_OFF 1024
#define SMEM_S_STATE_STAGE_BYTES 8192
#define SMEM_S_STATE_STRIDE 8192
#define SMEM_S_HEAD_SCALARS_OFF 33792
#define SMEM_S_HEAD_SCALARS_STAGE_BYTES 96
#define SMEM_S_HEAD_SCALARS_STRIDE 96
#define SMEM_TOTAL 33920
#define THREADS 288
#ifndef HEADS_PER_GROUP_STATIC
#error \
    "HEADS_PER_GROUP_STATIC is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef DIRECT_UNROLL
#error "DIRECT_UNROLL is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef COEFFICIENT_BF16
#error \
    "COEFFICIENT_BF16 is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef INDEX_I32
#error "INDEX_I32 is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef PAIRED_HEADS
#error "PAIRED_HEADS is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef HEADS_PER_CTA
#error "HEADS_PER_CTA is a downstream specialization of this program; define it on the compile line"
#endif
#define LAUNCH_MIN_BLOCKS 3

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}\n"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr), "r"(count) : "memory");
}

// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
      " P1, [%0], %1;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}\n" ::"r"(mbar_addr),
      "r"(phase)
      : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(mbar_addr),
      "r"(bytes)
      : "memory");
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

__device__ __forceinline__ void tma_4d_gmem2smem(int dst, const void* tmap_ptr, int x, int y, int z,
                                                 int w, int mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst),
      "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(mbar_addr)
      : "memory");
}

__device__ __forceinline__ void tma_store_4d(const void* tmap, int x, int y, int z, int w,
                                             unsigned smem_addr) {
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
      " [%0, {%1, %2, %3, %4}], [%5];" ::"l"(tmap),
      "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr)
      : "memory");
}

extern "C" {

__global__
__launch_bounds__(288, LAUNCH_MIN_BLOCKS) void kernel_cake_selective_state_update_c7e08dcbb58896052ee9(
    const __grid_constant__ CUtensorMap state_tma, __nv_bfloat16* __restrict__ x,
    unsigned long long dt_addr, unsigned long long a_addr, __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C, unsigned long long d_addr, __nv_bfloat16* __restrict__ z,
    unsigned long long dt_bias_addr, __nv_bfloat16* __restrict__ output,
    unsigned long long state_batch_indices_addr, unsigned long long dst_state_batch_indices_addr,
    int nheads, int ngroups, int head_tiles, long long x_batch_stride, long long b_batch_stride,
    long long c_batch_stride, long long out_batch_stride, long long dt_batch_stride,
    int dt_softplus, int has_z, int disable_state_update, long long pad_slot_id) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int mbar_base = smem;
#define state_full_addr (mbar_base + 0)
#define state_updated_addr (mbar_base + 32)

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  __nv_bfloat16* s_state = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
  const int s_state_addr = smem + 1024;
  float* s_head_scalars = reinterpret_cast<float*>(smem_raw + 33792);
  const int s_head_scalars_addr = smem + 33792;

  // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 8 barriers)
  // Mbarriers at smem_raw[0..64)

  if (warp == 0) {
    uint32_t leader = elect_sync();
    if (leader) {
      // --- pipeline 'state_pipe' ---
      // state_full: 4 barriers, init_count=1
      mbarrier_init(smem + 0, 1);
      mbarrier_init(smem + 8, 1);
      mbarrier_init(smem + 16, 1);
      mbarrier_init(smem + 24, 1);
      // state_updated: 4 barriers, init_count=8
      mbarrier_init(smem + 32, 8);
      mbarrier_init(smem + 40, 8);
      mbarrier_init(smem + 48, 8);
      mbarrier_init(smem + 56, 8);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }

  __syncthreads();

  // ---- Role: consumers ----
  if (warp <= 7) {
    {  // consumers_main
      int runtime_ratio = (int)(HEADS_PER_GROUP_STATIC == 0);
      int head_tile_divisor = HEADS_PER_GROUP_STATIC / HEADS_PER_CTA + runtime_ratio * head_tiles;
      int head_tile = bid % head_tile_divisor;
      int batch_group = bid / head_tile_divisor;
      int batch = batch_group / ngroups;
      int group = batch_group % ngroups;
      int heads_per_group = HEADS_PER_GROUP_STATIC + runtime_ratio * (nheads / ngroups);
      int first_local_head = head_tile * HEADS_PER_CTA;
      int work_heads = heads_per_group - first_local_head;
      if (work_heads > HEADS_PER_CTA) {
        work_heads = HEADS_PER_CTA;
      }
      int warp_id_in_role = (warp - 0);
      int local_warp = warp_id_in_role;
      int row_in_stage = local_warp * 4 + (lane >> 3);
      int member = lane & 7;
      int paired = PAIRED_HEADS;
      int coefficient_bf16 = COEFFICIENT_BF16;
      long long batch_i64 = (long long)batch;
      long long group_i64 = (long long)group;
      long long b_row_base = batch_i64 * b_batch_stride + group_i64 * 128;
      long long c_row_base = batch_i64 * c_batch_stride + group_i64 * 128;
      long long member_i64 = (long long)(member * 8);
      int consumer_index_i32 = INDEX_I32;
      long long consumer_slot = 0;
      if (consumer_index_i32 != 0) {
        int _vec_load_4[1];
        {
          uint32_t _scalar_bits_0;
          asm volatile("ld.global.nc.b32 %0, [%1];"
                       : "=r"(_scalar_bits_0)
                       : "l"((const void*)(reinterpret_cast<const int*>(state_batch_indices_addr) +
                                           (batch_i64)))
                       : "memory");
          _vec_load_4[0] = (int32_t)_scalar_bits_0;
        }
        consumer_slot = (long long)_vec_load_4[0];
      } else {
        long long _vec_load_5[1];
        {
          asm("ld.global.nc.s64 %0, [%1];"
              : "=l"(_vec_load_5[0])
              : "l"((const void*)(reinterpret_cast<const long long*>(state_batch_indices_addr) +
                                  (batch_i64))));
        }
        consumer_slot = _vec_load_5[0];
      }
      int pad_tile = 0;
      if (consumer_slot == pad_slot_id) {
        pad_tile = 1;
      }
      unsigned int b_carriers[8];
      unsigned int c_carriers[8];
      {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(B + b_row_base + member_i64);
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&b_carriers[0]);
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_vdst_1[_blk].x), "=r"(_vdst_1[_blk].y), "=r"(_vdst_1[_blk].z),
                         "=r"(_vdst_1[_blk].w)
                       : "l"((const void*)(_vptr_1 + _blk))
                       : "memory");
        }
      }
      {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(B + b_row_base + member_i64 + 64);
        uint4* _vdst_2 = reinterpret_cast<uint4*>(&b_carriers[4]);
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_vdst_2[_blk].x), "=r"(_vdst_2[_blk].y), "=r"(_vdst_2[_blk].z),
                         "=r"(_vdst_2[_blk].w)
                       : "l"((const void*)(_vptr_2 + _blk))
                       : "memory");
        }
      }
      {
        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(C + c_row_base + member_i64);
        uint4* _vdst_3 = reinterpret_cast<uint4*>(&c_carriers[0]);
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_vdst_3[_blk].x), "=r"(_vdst_3[_blk].y), "=r"(_vdst_3[_blk].z),
                         "=r"(_vdst_3[_blk].w)
                       : "l"((const void*)(_vptr_3 + _blk))
                       : "memory");
        }
      }
      {
        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(C + c_row_base + member_i64 + 64);
        uint4* _vdst_4 = reinterpret_cast<uint4*>(&c_carriers[4]);
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(_vdst_4[_blk].x), "=r"(_vdst_4[_blk].y), "=r"(_vdst_4[_blk].z),
                         "=r"(_vdst_4[_blk].w)
                       : "l"((const void*)(_vptr_4 + _blk))
                       : "memory");
        }
      }
      float b_values[16];
      float c_values[16];
#pragma unroll
      for (int _pair = 0; _pair < 8; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
            : "r"(b_carriers[_pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 8; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
            : "r"(c_carriers[_pair]));
      }
      if (local_warp == 2) {
        if (lane < work_heads << paired) {
          int scalar_slot = lane;
          int scalar_head_offset = scalar_slot >> paired;
          int scalar_half = scalar_slot & paired;
          int scalar_local_head = first_local_head + scalar_head_offset;
          int scalar_head = (group * heads_per_group + scalar_local_head << paired) + scalar_half;
          long long scalar_head_i64 = (long long)scalar_head;
          long long dt_index = batch_i64 * dt_batch_stride + scalar_head_i64;
          float scalar_dt = 0.0f;
          float scalar_d = 0.0f;
          if (coefficient_bf16 != 0) {
            float _vec_load_6[1];
            {
              uint32_t _bf16_bits_5;
              asm volatile(
                  "ld.global.nc.u16 %0, [%1];"
                  : "=r"(_bf16_bits_5)
                  : "l"((const void*)(reinterpret_cast<const __nv_bfloat16*>(dt_addr) + (dt_index)))
                  : "memory");
              _vec_load_6[0] = __uint_as_float(_bf16_bits_5 << 16);
            }
            float _vec_load_7[1];
            {
              uint32_t _bf16_bits_6;
              asm volatile(
                  "ld.global.nc.u16 %0, [%1];"
                  : "=r"(_bf16_bits_6)
                  : "l"((const void*)(reinterpret_cast<const __nv_bfloat16*>(dt_bias_addr) +
                                      (scalar_head_i64)))
                  : "memory");
              _vec_load_7[0] = __uint_as_float(_bf16_bits_6 << 16);
            }
            float _vec_load_8[1];
            {
              uint32_t _bf16_bits_7;
              asm volatile("ld.global.nc.u16 %0, [%1];"
                           : "=r"(_bf16_bits_7)
                           : "l"((const void*)(reinterpret_cast<const __nv_bfloat16*>(d_addr) +
                                               (scalar_head_i64)))
                           : "memory");
              _vec_load_8[0] = __uint_as_float(_bf16_bits_7 << 16);
            }
            scalar_dt = _vec_load_6[0] + _vec_load_7[0];
            scalar_d = _vec_load_8[0];
          } else {
            float _vec_load_9[1];
            {
              uint32_t _scalar_bits_8;
              asm volatile(
                  "ld.global.nc.b32 %0, [%1];"
                  : "=r"(_scalar_bits_8)
                  : "l"((const void*)(reinterpret_cast<const float*>(dt_addr) + (dt_index)))
                  : "memory");
              _vec_load_9[0] = __uint_as_float(_scalar_bits_8);
            }
            float _vec_load_10[1];
            {
              uint32_t _scalar_bits_9;
              asm volatile("ld.global.nc.b32 %0, [%1];"
                           : "=r"(_scalar_bits_9)
                           : "l"((const void*)(reinterpret_cast<const float*>(dt_bias_addr) +
                                               (scalar_head_i64)))
                           : "memory");
              _vec_load_10[0] = __uint_as_float(_scalar_bits_9);
            }
            float _vec_load_11[1];
            {
              uint32_t _scalar_bits_10;
              asm volatile(
                  "ld.global.nc.b32 %0, [%1];"
                  : "=r"(_scalar_bits_10)
                  : "l"((const void*)(reinterpret_cast<const float*>(d_addr) + (scalar_head_i64)))
                  : "memory");
              _vec_load_11[0] = __uint_as_float(_scalar_bits_10);
            }
            scalar_dt = _vec_load_9[0] + _vec_load_10[0];
            scalar_d = _vec_load_11[0];
          }
          float _vec_load_12[1];
          {
            uint32_t _scalar_bits_11;
            asm volatile(
                "ld.global.nc.b32 %0, [%1];"
                : "=r"(_scalar_bits_11)
                : "l"((const void*)(reinterpret_cast<const float*>(a_addr) + (scalar_head_i64)))
                : "memory");
            _vec_load_12[0] = __uint_as_float(_scalar_bits_11);
          }
          if (dt_softplus != 0) {
            if (scalar_dt <= 20.0f) {
              float _expf_0 = __expf(scalar_dt);
              float _log2_0;
              asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(1.0f + _expf_0));
              scalar_dt = _log2_0 * 0.6931471805599453f;
            }
          }
          int published_scalar_base = scalar_slot * 3;
          s_head_scalars[published_scalar_base] = scalar_dt;
          float _exp_0 = expf(_vec_load_12[0] * scalar_dt);
          s_head_scalars[published_scalar_base + 1] = _exp_0;
          s_head_scalars[published_scalar_base + 2] = scalar_d;
        }
      }
      asm volatile("barrier.sync 8, 256;" ::: "memory");
      constexpr int head_offset_unroll = DIRECT_UNROLL;
#pragma unroll head_offset_unroll
      for (int head_offset = 0; head_offset < work_heads; head_offset++) {
        int local_head = first_local_head + head_offset;
        int head = group * heads_per_group + local_head;
        unsigned int work_phase = (unsigned int)(head_offset & 1);
        long long row_base = (long long)(head * 128 + row_in_stage);
        float x_values[4];
        float z_values[4];
#pragma unroll
        for (int stage = 0; stage < 4; stage++) {
          long long x_index = batch_i64 * x_batch_stride + row_base + (long long)(stage * 32);
          x_values[stage] = (float)x[x_index];
          z_values[stage] = 0.0f;
          if (has_z != 0) {
            z_values[stage] = (float)z[x_index];
          }
        }
#pragma unroll
        for (int stage_1 = 0; stage_1 < 4; stage_1++) {
          int head_scalar_base = ((head_offset << paired) + stage_1 * 32 / 64) * 3;
          float dt_value = s_head_scalars[head_scalar_base];
          float decay = s_head_scalars[head_scalar_base + 1];
          float d_value = s_head_scalars[head_scalar_base + 2];
          float dtx = dt_value * x_values[stage_1];
          float2 _f2_0 = make_float2(decay, decay);
          float2 decay_pair = _f2_0;
          float2 _f2_1 = make_float2(dtx, dtx);
          float2 dtx_pair = _f2_1;
          float2 _f2_2 = make_float2(0.0f, 0.0f);
          float2 partial_pair = _f2_2;
          unsigned int state_carriers[8];
          float state_values[16];
          mbarrier_wait(state_full_addr + (stage_1) * 8, work_phase);
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          if (pad_tile == 0) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[0])),
                           "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 1])),
                           "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 2])),
                           "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 3]))
                         : "r"(s_state_addr + (unsigned int)(stage_1 * 8192) +
                               (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16)));
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[4])),
                  "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(4) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(4) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(4) + 3]))
                : "r"(s_state_addr + (unsigned int)(stage_1 * 8192) +
                      (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16) + 128));
          } else {
#pragma unroll
            for (int carrier = 0; carrier < 8; carrier++) {
              state_carriers[carrier] = 0;
            }
          }
#pragma unroll
          for (int _pair = 0; _pair < 8; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&state_values[_pair * 2])[0]), "=f"((&state_values[_pair * 2])[1])
                : "r"(state_carriers[_pair]));
          }
#pragma unroll
          for (int pair = 0; pair < 8; pair++) {
            float2 _f2_3 = make_float2(state_values[2 * pair], state_values[2 * pair + 1]);
            float2 state_pair = _f2_3;
            float2 _f2_4 = make_float2(b_values[2 * pair], b_values[2 * pair + 1]);
            float2 b_pair = _f2_4;
            float2 _f2_5 = make_float2(c_values[2 * pair], c_values[2 * pair + 1]);
            float2 c_pair = _f2_5;
            float2 _mul_f32x2_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;"
                : "=l"(*(unsigned long long*)&_mul_f32x2_0)
                : "l"(*(const unsigned long long*)&b_pair),
                  "l"(*(const unsigned long long*)&dtx_pair));
            float2 dbx_pair = _mul_f32x2_0;
            state_pair = fma_f32x2_rn_ftz(state_pair, decay_pair, dbx_pair);
            partial_pair = fma_f32x2_rn_ftz(state_pair, c_pair, partial_pair);
            state_values[2 * pair] = state_pair.x;
            state_values[2 * pair + 1] = state_pair.y;
          }
          uint32_t state_values_bf16[8];
#pragma unroll
          for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(state_values[_lp * 2 + 0], state_values[_lp * 2 + 1 + 0]));
            state_values_bf16[_lp] = *(uint32_t*)&_bf2;
          }
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                           s_state_addr + (unsigned int)(stage_1 * 8192) +
                           (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16)),
                       "r"(*reinterpret_cast<uint32_t*>(&state_values_bf16[0])),
                       "r"(*reinterpret_cast<uint32_t*>(&state_values_bf16[(0) + 1])),
                       "r"(*reinterpret_cast<uint32_t*>(&state_values_bf16[(0) + 2])),
                       "r"(*reinterpret_cast<uint32_t*>(&state_values_bf16[(0) + 3])));
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                           s_state_addr + (unsigned int)(stage_1 * 8192) +
                           (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16) + 128),
                       "r"(*reinterpret_cast<uint32_t*>(&state_values_bf16[4])),
                       "r"(*reinterpret_cast<uint32_t*>(&state_values_bf16[(4) + 1])),
                       "r"(*reinterpret_cast<uint32_t*>(&state_values_bf16[(4) + 2])),
                       "r"(*reinterpret_cast<uint32_t*>(&state_values_bf16[(4) + 3])));
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(state_updated_addr + (stage_1) * 8);
          }
          float partial = partial_pair.x + partial_pair.y;
          float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, partial, 4);
          partial += _shfl_xor_0;
          float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, partial, 2);
          partial += _shfl_xor_1;
          float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, partial, 1);
          partial += _shfl_xor_2;
          if (member == 0) {
            float _fma_0 = __fmaf_rn(d_value, x_values[stage_1], partial);
            float result = _fma_0;
            if (has_z != 0) {
              float _exp_1 = expf(-z_values[stage_1]);
              float sigmoid_z = 1.0f / (1.0f + _exp_1);
              result *= z_values[stage_1] * sigmoid_z;
            }
            long long out_index =
                batch_i64 * out_batch_stride + row_base + (long long)(stage_1 * 32);
            output[out_index] = result;
          }
        }
      }
    }
  }
  // ---- Role: producer ----
  if (warp == 8) {
    {  // producer_main
      int runtime_ratio_1 = (int)(HEADS_PER_GROUP_STATIC == 0);
      int head_tile_divisor_1 =
          HEADS_PER_GROUP_STATIC / HEADS_PER_CTA + runtime_ratio_1 * head_tiles;
      int head_tile_1 = bid % head_tile_divisor_1;
      int batch_group_1 = bid / head_tile_divisor_1;
      int batch_1 = batch_group_1 / ngroups;
      int group_1 = batch_group_1 % ngroups;
      int heads_per_group_1 = HEADS_PER_GROUP_STATIC + runtime_ratio_1 * (nheads / ngroups);
      int first_local_head_1 = head_tile_1 * HEADS_PER_CTA;
      int work_heads_1 = heads_per_group_1 - first_local_head_1;
      if (work_heads_1 > HEADS_PER_CTA) {
        work_heads_1 = HEADS_PER_CTA;
      }
      int index_i32 = INDEX_I32;
      long long batch_i64_1 = (long long)batch_1;
      long long source_slot = 0;
      long long destination_slot = 0;
      if (index_i32 != 0) {
        int _vec_load_0[1];
        {
          uint32_t _scalar_bits_0;
          asm volatile("ld.global.nc.b32 %0, [%1];"
                       : "=r"(_scalar_bits_0)
                       : "l"((const void*)(reinterpret_cast<const int*>(state_batch_indices_addr) +
                                           (batch_i64_1)))
                       : "memory");
          _vec_load_0[0] = (int32_t)_scalar_bits_0;
        }
        int _vec_load_1[1];
        {
          uint32_t _scalar_bits_1;
          asm volatile(
              "ld.global.nc.b32 %0, [%1];"
              : "=r"(_scalar_bits_1)
              : "l"((const void*)(reinterpret_cast<const int*>(dst_state_batch_indices_addr) +
                                  (batch_i64_1)))
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
                                  (batch_i64_1))));
        }
        long long _vec_load_3[1];
        {
          asm("ld.global.nc.s64 %0, [%1];"
              : "=l"(_vec_load_3[0])
              : "l"((const void*)(reinterpret_cast<const long long*>(dst_state_batch_indices_addr) +
                                  (batch_i64_1))));
        }
        source_slot = _vec_load_2[0];
        destination_slot = _vec_load_3[0];
      }
      int load_live = 1;
      int store_live = 1 - disable_state_update;
      if (source_slot == pad_slot_id) {
        load_live = 0;
        store_live = 0;
      }
      if (destination_slot == pad_slot_id) {
        store_live = 0;
      }
      if (elect_sync()) {
        int first_head = group_1 * heads_per_group_1 + first_local_head_1;
#pragma unroll
        for (int stage_2 = 0; stage_2 < 4; stage_2++) {
          if (load_live != 0) {
            tma_4d_gmem2smem(s_state_addr + (unsigned int)(stage_2 * 8192), (&state_tma), 0,
                             stage_2 * 32, first_head, (int)source_slot,
                             state_full_addr + (stage_2) * 8);
            mbarrier_arrive_expect_tx(state_full_addr + (stage_2) * 8, 8192);
          } else {
            mbarrier_arrive(state_full_addr + (stage_2) * 8);
          }
        }
        constexpr int head_offset_1_unroll = DIRECT_UNROLL;
#pragma unroll head_offset_1_unroll
        for (int head_offset_1 = 0; head_offset_1 < work_heads_1; head_offset_1++) {
          int local_head_1 = first_local_head_1 + head_offset_1;
          int head_1 = group_1 * heads_per_group_1 + local_head_1;
          unsigned int work_phase_1 = (unsigned int)(head_offset_1 & 1);
          int next_head_valid = ((work_heads_1 > head_offset_1 + 1) ? 1 : 0);
#pragma unroll
          for (int stage_3 = 0; stage_3 < 4; stage_3++) {
            mbarrier_wait(state_updated_addr + (stage_3) * 8, work_phase_1);
            if (store_live != 0) {
              asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
              tma_store_4d((&state_tma), 0, stage_3 * 32, head_1, (int)destination_slot,
                           s_state_addr + (unsigned int)(stage_3 * 8192));
              asm volatile("cp.async.bulk.commit_group;");
              asm volatile("cp.async.bulk.wait_group.read 0;");
            }
            if (next_head_valid != 0) {
              int next_head = head_1 + 1;
              if (load_live != 0) {
                tma_4d_gmem2smem(s_state_addr + (unsigned int)(stage_3 * 8192), (&state_tma), 0,
                                 stage_3 * 32, next_head, (int)source_slot,
                                 state_full_addr + (stage_3) * 8);
                mbarrier_arrive_expect_tx(state_full_addr + (stage_3) * 8, 8192);
              } else {
                mbarrier_arrive(state_full_addr + (stage_3) * 8);
              }
            }
          }
        }
        if (disable_state_update == 0) {
          asm volatile("cp.async.bulk.wait_group 0;");
        }
      }
    }
  }

  // Cleanup
}

}  // extern "C"
