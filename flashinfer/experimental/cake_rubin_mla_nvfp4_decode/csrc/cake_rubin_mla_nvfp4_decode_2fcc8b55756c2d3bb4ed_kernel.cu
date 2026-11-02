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
#define SMEM_SMEM_W_OFF 0
#define SMEM_SMEM_W_STAGE_BYTES 1024
#define SMEM_SMEM_W_STRIDE 1024
#define SMEM_TOTAL 1024
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float approx_rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float max_noftz(float a, float b) {
  float c;
  asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}

extern "C" {

__global__
__launch_bounds__(128, LAUNCH_MIN_BLOCKS) void kernel_cake_rubin_mla_nvfp4_decode_2fcc8b55756c2d3bb4ed(
    __nv_bfloat16* __restrict__ partial_o, float* __restrict__ partial_lse,
    __nv_bfloat16* __restrict__ O, float* __restrict__ lse, int rows, int num_split, int has_lse) {
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
  float* smem_w = reinterpret_cast<float*>(smem_raw + 0);
  const int smem_w_addr = smem + 0;

  // === Task calls (declaration order) ===
  int part_7 = warp % 4;
  int row_7 = blockIdx.x * 1 + warp / 4;
  int lane_7 = lane;
  int stat_base_7 = row_7 * num_split;
  int w_base_7 = warp * 64;
  int d0_7 = part_7 * 128 + lane_7 * 4;
  int last_7 = num_split - 1;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (row_7 < rows) {
    float l0 = -CUDART_INF_F;
    float l1 = -CUDART_INF_F;
    if (lane_7 < num_split) {
      l0 = partial_lse[stat_base_7 + lane_7];
    }
    if (lane_7 + 32 < num_split) {
      l1 = partial_lse[stat_base_7 + lane_7 + 32];
    }
    unsigned int vec[128];
    int _mn_0 = ((0) < (last_7) ? (0) : (last_7));
    {
      uint2 _vld_0;
      _vld_0 =
          *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_0) * 512 + d0_7) + 0);
      uint2* _vdst_0 = reinterpret_cast<uint2*>(&vec[0]);
      *_vdst_0 = _vld_0;
    }
    int _mn_1 = ((1) < (last_7) ? (1) : (last_7));
    {
      uint2 _vld_1;
      _vld_1 =
          *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_1) * 512 + d0_7) + 0);
      uint2* _vdst_1 = reinterpret_cast<uint2*>(&vec[2]);
      *_vdst_1 = _vld_1;
    }
    int _mn_2 = ((2) < (last_7) ? (2) : (last_7));
    {
      uint2 _vld_2;
      _vld_2 =
          *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_2) * 512 + d0_7) + 0);
      uint2* _vdst_2 = reinterpret_cast<uint2*>(&vec[4]);
      *_vdst_2 = _vld_2;
    }
    int _mn_3 = ((3) < (last_7) ? (3) : (last_7));
    {
      uint2 _vld_3;
      _vld_3 =
          *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_3) * 512 + d0_7) + 0);
      uint2* _vdst_3 = reinterpret_cast<uint2*>(&vec[6]);
      *_vdst_3 = _vld_3;
    }
    int _mn_4 = ((4) < (last_7) ? (4) : (last_7));
    {
      uint2 _vld_4;
      _vld_4 =
          *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_4) * 512 + d0_7) + 0);
      uint2* _vdst_4 = reinterpret_cast<uint2*>(&vec[8]);
      *_vdst_4 = _vld_4;
    }
    int _mn_5 = ((5) < (last_7) ? (5) : (last_7));
    {
      uint2 _vld_5;
      _vld_5 =
          *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_5) * 512 + d0_7) + 0);
      uint2* _vdst_5 = reinterpret_cast<uint2*>(&vec[10]);
      *_vdst_5 = _vld_5;
    }
    int _mn_6 = ((6) < (last_7) ? (6) : (last_7));
    {
      uint2 _vld_6;
      _vld_6 =
          *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_6) * 512 + d0_7) + 0);
      uint2* _vdst_6 = reinterpret_cast<uint2*>(&vec[12]);
      *_vdst_6 = _vld_6;
    }
    int _mn_7 = ((7) < (last_7) ? (7) : (last_7));
    {
      uint2 _vld_7;
      _vld_7 =
          *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_7) * 512 + d0_7) + 0);
      uint2* _vdst_7 = reinterpret_cast<uint2*>(&vec[14]);
      *_vdst_7 = _vld_7;
    }
    if (8 < num_split) {
      int _mn_8 = ((8) < (last_7) ? (8) : (last_7));
      {
        uint2 _vld_8;
        _vld_8 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_8) * 512 + d0_7) + 0);
        uint2* _vdst_8 = reinterpret_cast<uint2*>(&vec[16]);
        *_vdst_8 = _vld_8;
      }
      int _mn_9 = ((9) < (last_7) ? (9) : (last_7));
      {
        uint2 _vld_9;
        _vld_9 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_9) * 512 + d0_7) + 0);
        uint2* _vdst_9 = reinterpret_cast<uint2*>(&vec[18]);
        *_vdst_9 = _vld_9;
      }
      int _mn_10 = ((10) < (last_7) ? (10) : (last_7));
      {
        uint2 _vld_10;
        _vld_10 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_10) * 512 + d0_7) + 0);
        uint2* _vdst_10 = reinterpret_cast<uint2*>(&vec[20]);
        *_vdst_10 = _vld_10;
      }
      int _mn_11 = ((11) < (last_7) ? (11) : (last_7));
      {
        uint2 _vld_11;
        _vld_11 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_11) * 512 + d0_7) + 0);
        uint2* _vdst_11 = reinterpret_cast<uint2*>(&vec[22]);
        *_vdst_11 = _vld_11;
      }
      int _mn_12 = ((12) < (last_7) ? (12) : (last_7));
      {
        uint2 _vld_12;
        _vld_12 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_12) * 512 + d0_7) + 0);
        uint2* _vdst_12 = reinterpret_cast<uint2*>(&vec[24]);
        *_vdst_12 = _vld_12;
      }
      int _mn_13 = ((13) < (last_7) ? (13) : (last_7));
      {
        uint2 _vld_13;
        _vld_13 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_13) * 512 + d0_7) + 0);
        uint2* _vdst_13 = reinterpret_cast<uint2*>(&vec[26]);
        *_vdst_13 = _vld_13;
      }
      int _mn_14 = ((14) < (last_7) ? (14) : (last_7));
      {
        uint2 _vld_14;
        _vld_14 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_14) * 512 + d0_7) + 0);
        uint2* _vdst_14 = reinterpret_cast<uint2*>(&vec[28]);
        *_vdst_14 = _vld_14;
      }
      int _mn_15 = ((15) < (last_7) ? (15) : (last_7));
      {
        uint2 _vld_15;
        _vld_15 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_15) * 512 + d0_7) + 0);
        uint2* _vdst_15 = reinterpret_cast<uint2*>(&vec[30]);
        *_vdst_15 = _vld_15;
      }
    }
    if (16 < num_split) {
      int _mn_16 = ((16) < (last_7) ? (16) : (last_7));
      {
        uint2 _vld_16;
        _vld_16 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_16) * 512 + d0_7) + 0);
        uint2* _vdst_16 = reinterpret_cast<uint2*>(&vec[32]);
        *_vdst_16 = _vld_16;
      }
      int _mn_17 = ((17) < (last_7) ? (17) : (last_7));
      {
        uint2 _vld_17;
        _vld_17 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_17) * 512 + d0_7) + 0);
        uint2* _vdst_17 = reinterpret_cast<uint2*>(&vec[34]);
        *_vdst_17 = _vld_17;
      }
      int _mn_18 = ((18) < (last_7) ? (18) : (last_7));
      {
        uint2 _vld_18;
        _vld_18 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_18) * 512 + d0_7) + 0);
        uint2* _vdst_18 = reinterpret_cast<uint2*>(&vec[36]);
        *_vdst_18 = _vld_18;
      }
      int _mn_19 = ((19) < (last_7) ? (19) : (last_7));
      {
        uint2 _vld_19;
        _vld_19 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_19) * 512 + d0_7) + 0);
        uint2* _vdst_19 = reinterpret_cast<uint2*>(&vec[38]);
        *_vdst_19 = _vld_19;
      }
      int _mn_20 = ((20) < (last_7) ? (20) : (last_7));
      {
        uint2 _vld_20;
        _vld_20 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_20) * 512 + d0_7) + 0);
        uint2* _vdst_20 = reinterpret_cast<uint2*>(&vec[40]);
        *_vdst_20 = _vld_20;
      }
      int _mn_21 = ((21) < (last_7) ? (21) : (last_7));
      {
        uint2 _vld_21;
        _vld_21 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_21) * 512 + d0_7) + 0);
        uint2* _vdst_21 = reinterpret_cast<uint2*>(&vec[42]);
        *_vdst_21 = _vld_21;
      }
      int _mn_22 = ((22) < (last_7) ? (22) : (last_7));
      {
        uint2 _vld_22;
        _vld_22 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_22) * 512 + d0_7) + 0);
        uint2* _vdst_22 = reinterpret_cast<uint2*>(&vec[44]);
        *_vdst_22 = _vld_22;
      }
      int _mn_23 = ((23) < (last_7) ? (23) : (last_7));
      {
        uint2 _vld_23;
        _vld_23 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_23) * 512 + d0_7) + 0);
        uint2* _vdst_23 = reinterpret_cast<uint2*>(&vec[46]);
        *_vdst_23 = _vld_23;
      }
    }
    if (24 < num_split) {
      int _mn_24 = ((24) < (last_7) ? (24) : (last_7));
      {
        uint2 _vld_24;
        _vld_24 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_24) * 512 + d0_7) + 0);
        uint2* _vdst_24 = reinterpret_cast<uint2*>(&vec[48]);
        *_vdst_24 = _vld_24;
      }
      int _mn_25 = ((25) < (last_7) ? (25) : (last_7));
      {
        uint2 _vld_25;
        _vld_25 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_25) * 512 + d0_7) + 0);
        uint2* _vdst_25 = reinterpret_cast<uint2*>(&vec[50]);
        *_vdst_25 = _vld_25;
      }
      int _mn_26 = ((26) < (last_7) ? (26) : (last_7));
      {
        uint2 _vld_26;
        _vld_26 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_26) * 512 + d0_7) + 0);
        uint2* _vdst_26 = reinterpret_cast<uint2*>(&vec[52]);
        *_vdst_26 = _vld_26;
      }
      int _mn_27 = ((27) < (last_7) ? (27) : (last_7));
      {
        uint2 _vld_27;
        _vld_27 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_27) * 512 + d0_7) + 0);
        uint2* _vdst_27 = reinterpret_cast<uint2*>(&vec[54]);
        *_vdst_27 = _vld_27;
      }
      int _mn_28 = ((28) < (last_7) ? (28) : (last_7));
      {
        uint2 _vld_28;
        _vld_28 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_28) * 512 + d0_7) + 0);
        uint2* _vdst_28 = reinterpret_cast<uint2*>(&vec[56]);
        *_vdst_28 = _vld_28;
      }
      int _mn_29 = ((29) < (last_7) ? (29) : (last_7));
      {
        uint2 _vld_29;
        _vld_29 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_29) * 512 + d0_7) + 0);
        uint2* _vdst_29 = reinterpret_cast<uint2*>(&vec[58]);
        *_vdst_29 = _vld_29;
      }
      int _mn_30 = ((30) < (last_7) ? (30) : (last_7));
      {
        uint2 _vld_30;
        _vld_30 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_30) * 512 + d0_7) + 0);
        uint2* _vdst_30 = reinterpret_cast<uint2*>(&vec[60]);
        *_vdst_30 = _vld_30;
      }
      int _mn_31 = ((31) < (last_7) ? (31) : (last_7));
      {
        uint2 _vld_31;
        _vld_31 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_31) * 512 + d0_7) + 0);
        uint2* _vdst_31 = reinterpret_cast<uint2*>(&vec[62]);
        *_vdst_31 = _vld_31;
      }
    }
    if (32 < num_split) {
      int _mn_32 = ((32) < (last_7) ? (32) : (last_7));
      {
        uint2 _vld_32;
        _vld_32 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_32) * 512 + d0_7) + 0);
        uint2* _vdst_32 = reinterpret_cast<uint2*>(&vec[64]);
        *_vdst_32 = _vld_32;
      }
      int _mn_33 = ((33) < (last_7) ? (33) : (last_7));
      {
        uint2 _vld_33;
        _vld_33 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_33) * 512 + d0_7) + 0);
        uint2* _vdst_33 = reinterpret_cast<uint2*>(&vec[66]);
        *_vdst_33 = _vld_33;
      }
      int _mn_34 = ((34) < (last_7) ? (34) : (last_7));
      {
        uint2 _vld_34;
        _vld_34 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_34) * 512 + d0_7) + 0);
        uint2* _vdst_34 = reinterpret_cast<uint2*>(&vec[68]);
        *_vdst_34 = _vld_34;
      }
      int _mn_35 = ((35) < (last_7) ? (35) : (last_7));
      {
        uint2 _vld_35;
        _vld_35 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_35) * 512 + d0_7) + 0);
        uint2* _vdst_35 = reinterpret_cast<uint2*>(&vec[70]);
        *_vdst_35 = _vld_35;
      }
      int _mn_36 = ((36) < (last_7) ? (36) : (last_7));
      {
        uint2 _vld_36;
        _vld_36 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_36) * 512 + d0_7) + 0);
        uint2* _vdst_36 = reinterpret_cast<uint2*>(&vec[72]);
        *_vdst_36 = _vld_36;
      }
      int _mn_37 = ((37) < (last_7) ? (37) : (last_7));
      {
        uint2 _vld_37;
        _vld_37 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_37) * 512 + d0_7) + 0);
        uint2* _vdst_37 = reinterpret_cast<uint2*>(&vec[74]);
        *_vdst_37 = _vld_37;
      }
      int _mn_38 = ((38) < (last_7) ? (38) : (last_7));
      {
        uint2 _vld_38;
        _vld_38 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_38) * 512 + d0_7) + 0);
        uint2* _vdst_38 = reinterpret_cast<uint2*>(&vec[76]);
        *_vdst_38 = _vld_38;
      }
      int _mn_39 = ((39) < (last_7) ? (39) : (last_7));
      {
        uint2 _vld_39;
        _vld_39 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_39) * 512 + d0_7) + 0);
        uint2* _vdst_39 = reinterpret_cast<uint2*>(&vec[78]);
        *_vdst_39 = _vld_39;
      }
    }
    if (40 < num_split) {
      int _mn_40 = ((40) < (last_7) ? (40) : (last_7));
      {
        uint2 _vld_40;
        _vld_40 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_40) * 512 + d0_7) + 0);
        uint2* _vdst_40 = reinterpret_cast<uint2*>(&vec[80]);
        *_vdst_40 = _vld_40;
      }
      int _mn_41 = ((41) < (last_7) ? (41) : (last_7));
      {
        uint2 _vld_41;
        _vld_41 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_41) * 512 + d0_7) + 0);
        uint2* _vdst_41 = reinterpret_cast<uint2*>(&vec[82]);
        *_vdst_41 = _vld_41;
      }
      int _mn_42 = ((42) < (last_7) ? (42) : (last_7));
      {
        uint2 _vld_42;
        _vld_42 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_42) * 512 + d0_7) + 0);
        uint2* _vdst_42 = reinterpret_cast<uint2*>(&vec[84]);
        *_vdst_42 = _vld_42;
      }
      int _mn_43 = ((43) < (last_7) ? (43) : (last_7));
      {
        uint2 _vld_43;
        _vld_43 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_43) * 512 + d0_7) + 0);
        uint2* _vdst_43 = reinterpret_cast<uint2*>(&vec[86]);
        *_vdst_43 = _vld_43;
      }
      int _mn_44 = ((44) < (last_7) ? (44) : (last_7));
      {
        uint2 _vld_44;
        _vld_44 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_44) * 512 + d0_7) + 0);
        uint2* _vdst_44 = reinterpret_cast<uint2*>(&vec[88]);
        *_vdst_44 = _vld_44;
      }
      int _mn_45 = ((45) < (last_7) ? (45) : (last_7));
      {
        uint2 _vld_45;
        _vld_45 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_45) * 512 + d0_7) + 0);
        uint2* _vdst_45 = reinterpret_cast<uint2*>(&vec[90]);
        *_vdst_45 = _vld_45;
      }
      int _mn_46 = ((46) < (last_7) ? (46) : (last_7));
      {
        uint2 _vld_46;
        _vld_46 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_46) * 512 + d0_7) + 0);
        uint2* _vdst_46 = reinterpret_cast<uint2*>(&vec[92]);
        *_vdst_46 = _vld_46;
      }
      int _mn_47 = ((47) < (last_7) ? (47) : (last_7));
      {
        uint2 _vld_47;
        _vld_47 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_47) * 512 + d0_7) + 0);
        uint2* _vdst_47 = reinterpret_cast<uint2*>(&vec[94]);
        *_vdst_47 = _vld_47;
      }
    }
    if (48 < num_split) {
      int _mn_48 = ((48) < (last_7) ? (48) : (last_7));
      {
        uint2 _vld_48;
        _vld_48 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_48) * 512 + d0_7) + 0);
        uint2* _vdst_48 = reinterpret_cast<uint2*>(&vec[96]);
        *_vdst_48 = _vld_48;
      }
      int _mn_49 = ((49) < (last_7) ? (49) : (last_7));
      {
        uint2 _vld_49;
        _vld_49 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_49) * 512 + d0_7) + 0);
        uint2* _vdst_49 = reinterpret_cast<uint2*>(&vec[98]);
        *_vdst_49 = _vld_49;
      }
      int _mn_50 = ((50) < (last_7) ? (50) : (last_7));
      {
        uint2 _vld_50;
        _vld_50 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_50) * 512 + d0_7) + 0);
        uint2* _vdst_50 = reinterpret_cast<uint2*>(&vec[100]);
        *_vdst_50 = _vld_50;
      }
      int _mn_51 = ((51) < (last_7) ? (51) : (last_7));
      {
        uint2 _vld_51;
        _vld_51 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_51) * 512 + d0_7) + 0);
        uint2* _vdst_51 = reinterpret_cast<uint2*>(&vec[102]);
        *_vdst_51 = _vld_51;
      }
      int _mn_52 = ((52) < (last_7) ? (52) : (last_7));
      {
        uint2 _vld_52;
        _vld_52 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_52) * 512 + d0_7) + 0);
        uint2* _vdst_52 = reinterpret_cast<uint2*>(&vec[104]);
        *_vdst_52 = _vld_52;
      }
      int _mn_53 = ((53) < (last_7) ? (53) : (last_7));
      {
        uint2 _vld_53;
        _vld_53 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_53) * 512 + d0_7) + 0);
        uint2* _vdst_53 = reinterpret_cast<uint2*>(&vec[106]);
        *_vdst_53 = _vld_53;
      }
      int _mn_54 = ((54) < (last_7) ? (54) : (last_7));
      {
        uint2 _vld_54;
        _vld_54 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_54) * 512 + d0_7) + 0);
        uint2* _vdst_54 = reinterpret_cast<uint2*>(&vec[108]);
        *_vdst_54 = _vld_54;
      }
      int _mn_55 = ((55) < (last_7) ? (55) : (last_7));
      {
        uint2 _vld_55;
        _vld_55 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_55) * 512 + d0_7) + 0);
        uint2* _vdst_55 = reinterpret_cast<uint2*>(&vec[110]);
        *_vdst_55 = _vld_55;
      }
    }
    if (56 < num_split) {
      int _mn_56 = ((56) < (last_7) ? (56) : (last_7));
      {
        uint2 _vld_56;
        _vld_56 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_56) * 512 + d0_7) + 0);
        uint2* _vdst_56 = reinterpret_cast<uint2*>(&vec[112]);
        *_vdst_56 = _vld_56;
      }
      int _mn_57 = ((57) < (last_7) ? (57) : (last_7));
      {
        uint2 _vld_57;
        _vld_57 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_57) * 512 + d0_7) + 0);
        uint2* _vdst_57 = reinterpret_cast<uint2*>(&vec[114]);
        *_vdst_57 = _vld_57;
      }
      int _mn_58 = ((58) < (last_7) ? (58) : (last_7));
      {
        uint2 _vld_58;
        _vld_58 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_58) * 512 + d0_7) + 0);
        uint2* _vdst_58 = reinterpret_cast<uint2*>(&vec[116]);
        *_vdst_58 = _vld_58;
      }
      int _mn_59 = ((59) < (last_7) ? (59) : (last_7));
      {
        uint2 _vld_59;
        _vld_59 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_59) * 512 + d0_7) + 0);
        uint2* _vdst_59 = reinterpret_cast<uint2*>(&vec[118]);
        *_vdst_59 = _vld_59;
      }
      int _mn_60 = ((60) < (last_7) ? (60) : (last_7));
      {
        uint2 _vld_60;
        _vld_60 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_60) * 512 + d0_7) + 0);
        uint2* _vdst_60 = reinterpret_cast<uint2*>(&vec[120]);
        *_vdst_60 = _vld_60;
      }
      int _mn_61 = ((61) < (last_7) ? (61) : (last_7));
      {
        uint2 _vld_61;
        _vld_61 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_61) * 512 + d0_7) + 0);
        uint2* _vdst_61 = reinterpret_cast<uint2*>(&vec[122]);
        *_vdst_61 = _vld_61;
      }
      int _mn_62 = ((62) < (last_7) ? (62) : (last_7));
      {
        uint2 _vld_62;
        _vld_62 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_62) * 512 + d0_7) + 0);
        uint2* _vdst_62 = reinterpret_cast<uint2*>(&vec[124]);
        *_vdst_62 = _vld_62;
      }
      int _mn_63 = ((63) < (last_7) ? (63) : (last_7));
      {
        uint2 _vld_63;
        _vld_63 =
            *reinterpret_cast<const uint2*>(partial_o + ((stat_base_7 + _mn_63) * 512 + d0_7) + 0);
        uint2* _vdst_63 = reinterpret_cast<uint2*>(&vec[126]);
        *_vdst_63 = _vld_63;
      }
    }
    int merged = 0;
    float _shfl_sync_64 = __shfl_sync(0xFFFFFFFF, l0, 0);
    merged = _shfl_sync_64 == CUDART_INF_F;
    float m = max_noftz(l0, l1);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      m = max_noftz(m, __shfl_xor_sync(0xFFFFFFFF, m, offset));
    float w0 = 0.0f;
    float w1 = 0.0f;
    if (l0 > -CUDART_INF_F) {
      float _exp2_65 = approx_exp2(l0 - m);
      w0 = _exp2_65;
    }
    if (l1 > -CUDART_INF_F) {
      float _exp2_66 = approx_exp2(l1 - m);
      w1 = _exp2_66;
    }
    smem_w[w_base_7 + lane_7] = w0;
    smem_w[w_base_7 + 32 + lane_7] = w1;
    float sum_w = w0 + w1;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      sum_w += __shfl_xor_sync(0xFFFFFFFF, sum_w, offset);
    __syncwarp();
    float inv = 0.0f;
    if (sum_w > 0.0f) {
      float _rcp_67 = approx_rcp(sum_w);
      inv = _rcp_67;
    }
    float acc[4];
    for (int e = 0; e < 4; e++) {
      acc[e] = 0.0f;
    }
    float wts[64];
    wts[0] = smem_w[w_base_7 + 0];
    acc[0] = acc[0] + ((wts[0] > 0.0f) ? wts[0] * __uint_as_float(vec[0] << 16) : 0.0f);
    acc[1] = acc[1] + ((wts[0] > 0.0f) ? wts[0] * __uint_as_float(vec[0] & 4294901760u) : 0.0f);
    acc[2] = acc[2] + ((wts[0] > 0.0f) ? wts[0] * __uint_as_float(vec[1] << 16) : 0.0f);
    acc[3] = acc[3] + ((wts[0] > 0.0f) ? wts[0] * __uint_as_float(vec[1] & 4294901760u) : 0.0f);
    wts[1] = smem_w[w_base_7 + 1];
    acc[0] = acc[0] + ((wts[1] > 0.0f) ? wts[1] * __uint_as_float(vec[2] << 16) : 0.0f);
    acc[1] = acc[1] + ((wts[1] > 0.0f) ? wts[1] * __uint_as_float(vec[2] & 4294901760u) : 0.0f);
    acc[2] = acc[2] + ((wts[1] > 0.0f) ? wts[1] * __uint_as_float(vec[3] << 16) : 0.0f);
    acc[3] = acc[3] + ((wts[1] > 0.0f) ? wts[1] * __uint_as_float(vec[3] & 4294901760u) : 0.0f);
    wts[2] = smem_w[w_base_7 + 2];
    acc[0] = acc[0] + ((wts[2] > 0.0f) ? wts[2] * __uint_as_float(vec[4] << 16) : 0.0f);
    acc[1] = acc[1] + ((wts[2] > 0.0f) ? wts[2] * __uint_as_float(vec[4] & 4294901760u) : 0.0f);
    acc[2] = acc[2] + ((wts[2] > 0.0f) ? wts[2] * __uint_as_float(vec[5] << 16) : 0.0f);
    acc[3] = acc[3] + ((wts[2] > 0.0f) ? wts[2] * __uint_as_float(vec[5] & 4294901760u) : 0.0f);
    wts[3] = smem_w[w_base_7 + 3];
    acc[0] = acc[0] + ((wts[3] > 0.0f) ? wts[3] * __uint_as_float(vec[6] << 16) : 0.0f);
    acc[1] = acc[1] + ((wts[3] > 0.0f) ? wts[3] * __uint_as_float(vec[6] & 4294901760u) : 0.0f);
    acc[2] = acc[2] + ((wts[3] > 0.0f) ? wts[3] * __uint_as_float(vec[7] << 16) : 0.0f);
    acc[3] = acc[3] + ((wts[3] > 0.0f) ? wts[3] * __uint_as_float(vec[7] & 4294901760u) : 0.0f);
    wts[4] = smem_w[w_base_7 + 4];
    acc[0] = acc[0] + ((wts[4] > 0.0f) ? wts[4] * __uint_as_float(vec[8] << 16) : 0.0f);
    acc[1] = acc[1] + ((wts[4] > 0.0f) ? wts[4] * __uint_as_float(vec[8] & 4294901760u) : 0.0f);
    acc[2] = acc[2] + ((wts[4] > 0.0f) ? wts[4] * __uint_as_float(vec[9] << 16) : 0.0f);
    acc[3] = acc[3] + ((wts[4] > 0.0f) ? wts[4] * __uint_as_float(vec[9] & 4294901760u) : 0.0f);
    wts[5] = smem_w[w_base_7 + 5];
    acc[0] = acc[0] + ((wts[5] > 0.0f) ? wts[5] * __uint_as_float(vec[10] << 16) : 0.0f);
    acc[1] = acc[1] + ((wts[5] > 0.0f) ? wts[5] * __uint_as_float(vec[10] & 4294901760u) : 0.0f);
    acc[2] = acc[2] + ((wts[5] > 0.0f) ? wts[5] * __uint_as_float(vec[11] << 16) : 0.0f);
    acc[3] = acc[3] + ((wts[5] > 0.0f) ? wts[5] * __uint_as_float(vec[11] & 4294901760u) : 0.0f);
    wts[6] = smem_w[w_base_7 + 6];
    acc[0] = acc[0] + ((wts[6] > 0.0f) ? wts[6] * __uint_as_float(vec[12] << 16) : 0.0f);
    acc[1] = acc[1] + ((wts[6] > 0.0f) ? wts[6] * __uint_as_float(vec[12] & 4294901760u) : 0.0f);
    acc[2] = acc[2] + ((wts[6] > 0.0f) ? wts[6] * __uint_as_float(vec[13] << 16) : 0.0f);
    acc[3] = acc[3] + ((wts[6] > 0.0f) ? wts[6] * __uint_as_float(vec[13] & 4294901760u) : 0.0f);
    wts[7] = smem_w[w_base_7 + 7];
    acc[0] = acc[0] + ((wts[7] > 0.0f) ? wts[7] * __uint_as_float(vec[14] << 16) : 0.0f);
    acc[1] = acc[1] + ((wts[7] > 0.0f) ? wts[7] * __uint_as_float(vec[14] & 4294901760u) : 0.0f);
    acc[2] = acc[2] + ((wts[7] > 0.0f) ? wts[7] * __uint_as_float(vec[15] << 16) : 0.0f);
    acc[3] = acc[3] + ((wts[7] > 0.0f) ? wts[7] * __uint_as_float(vec[15] & 4294901760u) : 0.0f);
    if (8 < num_split) {
      wts[8] = smem_w[w_base_7 + 8];
      acc[0] = acc[0] + ((wts[8] > 0.0f) ? wts[8] * __uint_as_float(vec[16] << 16) : 0.0f);
      acc[1] = acc[1] + ((wts[8] > 0.0f) ? wts[8] * __uint_as_float(vec[16] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[8] > 0.0f) ? wts[8] * __uint_as_float(vec[17] << 16) : 0.0f);
      acc[3] = acc[3] + ((wts[8] > 0.0f) ? wts[8] * __uint_as_float(vec[17] & 4294901760u) : 0.0f);
      wts[9] = smem_w[w_base_7 + 9];
      acc[0] = acc[0] + ((wts[9] > 0.0f) ? wts[9] * __uint_as_float(vec[18] << 16) : 0.0f);
      acc[1] = acc[1] + ((wts[9] > 0.0f) ? wts[9] * __uint_as_float(vec[18] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[9] > 0.0f) ? wts[9] * __uint_as_float(vec[19] << 16) : 0.0f);
      acc[3] = acc[3] + ((wts[9] > 0.0f) ? wts[9] * __uint_as_float(vec[19] & 4294901760u) : 0.0f);
      wts[10] = smem_w[w_base_7 + 10];
      acc[0] = acc[0] + ((wts[10] > 0.0f) ? wts[10] * __uint_as_float(vec[20] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[10] > 0.0f) ? wts[10] * __uint_as_float(vec[20] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[10] > 0.0f) ? wts[10] * __uint_as_float(vec[21] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[10] > 0.0f) ? wts[10] * __uint_as_float(vec[21] & 4294901760u) : 0.0f);
      wts[11] = smem_w[w_base_7 + 11];
      acc[0] = acc[0] + ((wts[11] > 0.0f) ? wts[11] * __uint_as_float(vec[22] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[11] > 0.0f) ? wts[11] * __uint_as_float(vec[22] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[11] > 0.0f) ? wts[11] * __uint_as_float(vec[23] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[11] > 0.0f) ? wts[11] * __uint_as_float(vec[23] & 4294901760u) : 0.0f);
      wts[12] = smem_w[w_base_7 + 12];
      acc[0] = acc[0] + ((wts[12] > 0.0f) ? wts[12] * __uint_as_float(vec[24] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[12] > 0.0f) ? wts[12] * __uint_as_float(vec[24] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[12] > 0.0f) ? wts[12] * __uint_as_float(vec[25] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[12] > 0.0f) ? wts[12] * __uint_as_float(vec[25] & 4294901760u) : 0.0f);
      wts[13] = smem_w[w_base_7 + 13];
      acc[0] = acc[0] + ((wts[13] > 0.0f) ? wts[13] * __uint_as_float(vec[26] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[13] > 0.0f) ? wts[13] * __uint_as_float(vec[26] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[13] > 0.0f) ? wts[13] * __uint_as_float(vec[27] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[13] > 0.0f) ? wts[13] * __uint_as_float(vec[27] & 4294901760u) : 0.0f);
      wts[14] = smem_w[w_base_7 + 14];
      acc[0] = acc[0] + ((wts[14] > 0.0f) ? wts[14] * __uint_as_float(vec[28] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[14] > 0.0f) ? wts[14] * __uint_as_float(vec[28] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[14] > 0.0f) ? wts[14] * __uint_as_float(vec[29] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[14] > 0.0f) ? wts[14] * __uint_as_float(vec[29] & 4294901760u) : 0.0f);
      wts[15] = smem_w[w_base_7 + 15];
      acc[0] = acc[0] + ((wts[15] > 0.0f) ? wts[15] * __uint_as_float(vec[30] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[15] > 0.0f) ? wts[15] * __uint_as_float(vec[30] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[15] > 0.0f) ? wts[15] * __uint_as_float(vec[31] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[15] > 0.0f) ? wts[15] * __uint_as_float(vec[31] & 4294901760u) : 0.0f);
    }
    if (16 < num_split) {
      wts[16] = smem_w[w_base_7 + 16];
      acc[0] = acc[0] + ((wts[16] > 0.0f) ? wts[16] * __uint_as_float(vec[32] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[16] > 0.0f) ? wts[16] * __uint_as_float(vec[32] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[16] > 0.0f) ? wts[16] * __uint_as_float(vec[33] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[16] > 0.0f) ? wts[16] * __uint_as_float(vec[33] & 4294901760u) : 0.0f);
      wts[17] = smem_w[w_base_7 + 17];
      acc[0] = acc[0] + ((wts[17] > 0.0f) ? wts[17] * __uint_as_float(vec[34] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[17] > 0.0f) ? wts[17] * __uint_as_float(vec[34] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[17] > 0.0f) ? wts[17] * __uint_as_float(vec[35] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[17] > 0.0f) ? wts[17] * __uint_as_float(vec[35] & 4294901760u) : 0.0f);
      wts[18] = smem_w[w_base_7 + 18];
      acc[0] = acc[0] + ((wts[18] > 0.0f) ? wts[18] * __uint_as_float(vec[36] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[18] > 0.0f) ? wts[18] * __uint_as_float(vec[36] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[18] > 0.0f) ? wts[18] * __uint_as_float(vec[37] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[18] > 0.0f) ? wts[18] * __uint_as_float(vec[37] & 4294901760u) : 0.0f);
      wts[19] = smem_w[w_base_7 + 19];
      acc[0] = acc[0] + ((wts[19] > 0.0f) ? wts[19] * __uint_as_float(vec[38] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[19] > 0.0f) ? wts[19] * __uint_as_float(vec[38] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[19] > 0.0f) ? wts[19] * __uint_as_float(vec[39] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[19] > 0.0f) ? wts[19] * __uint_as_float(vec[39] & 4294901760u) : 0.0f);
      wts[20] = smem_w[w_base_7 + 20];
      acc[0] = acc[0] + ((wts[20] > 0.0f) ? wts[20] * __uint_as_float(vec[40] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[20] > 0.0f) ? wts[20] * __uint_as_float(vec[40] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[20] > 0.0f) ? wts[20] * __uint_as_float(vec[41] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[20] > 0.0f) ? wts[20] * __uint_as_float(vec[41] & 4294901760u) : 0.0f);
      wts[21] = smem_w[w_base_7 + 21];
      acc[0] = acc[0] + ((wts[21] > 0.0f) ? wts[21] * __uint_as_float(vec[42] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[21] > 0.0f) ? wts[21] * __uint_as_float(vec[42] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[21] > 0.0f) ? wts[21] * __uint_as_float(vec[43] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[21] > 0.0f) ? wts[21] * __uint_as_float(vec[43] & 4294901760u) : 0.0f);
      wts[22] = smem_w[w_base_7 + 22];
      acc[0] = acc[0] + ((wts[22] > 0.0f) ? wts[22] * __uint_as_float(vec[44] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[22] > 0.0f) ? wts[22] * __uint_as_float(vec[44] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[22] > 0.0f) ? wts[22] * __uint_as_float(vec[45] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[22] > 0.0f) ? wts[22] * __uint_as_float(vec[45] & 4294901760u) : 0.0f);
      wts[23] = smem_w[w_base_7 + 23];
      acc[0] = acc[0] + ((wts[23] > 0.0f) ? wts[23] * __uint_as_float(vec[46] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[23] > 0.0f) ? wts[23] * __uint_as_float(vec[46] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[23] > 0.0f) ? wts[23] * __uint_as_float(vec[47] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[23] > 0.0f) ? wts[23] * __uint_as_float(vec[47] & 4294901760u) : 0.0f);
    }
    if (24 < num_split) {
      wts[24] = smem_w[w_base_7 + 24];
      acc[0] = acc[0] + ((wts[24] > 0.0f) ? wts[24] * __uint_as_float(vec[48] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[24] > 0.0f) ? wts[24] * __uint_as_float(vec[48] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[24] > 0.0f) ? wts[24] * __uint_as_float(vec[49] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[24] > 0.0f) ? wts[24] * __uint_as_float(vec[49] & 4294901760u) : 0.0f);
      wts[25] = smem_w[w_base_7 + 25];
      acc[0] = acc[0] + ((wts[25] > 0.0f) ? wts[25] * __uint_as_float(vec[50] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[25] > 0.0f) ? wts[25] * __uint_as_float(vec[50] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[25] > 0.0f) ? wts[25] * __uint_as_float(vec[51] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[25] > 0.0f) ? wts[25] * __uint_as_float(vec[51] & 4294901760u) : 0.0f);
      wts[26] = smem_w[w_base_7 + 26];
      acc[0] = acc[0] + ((wts[26] > 0.0f) ? wts[26] * __uint_as_float(vec[52] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[26] > 0.0f) ? wts[26] * __uint_as_float(vec[52] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[26] > 0.0f) ? wts[26] * __uint_as_float(vec[53] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[26] > 0.0f) ? wts[26] * __uint_as_float(vec[53] & 4294901760u) : 0.0f);
      wts[27] = smem_w[w_base_7 + 27];
      acc[0] = acc[0] + ((wts[27] > 0.0f) ? wts[27] * __uint_as_float(vec[54] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[27] > 0.0f) ? wts[27] * __uint_as_float(vec[54] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[27] > 0.0f) ? wts[27] * __uint_as_float(vec[55] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[27] > 0.0f) ? wts[27] * __uint_as_float(vec[55] & 4294901760u) : 0.0f);
      wts[28] = smem_w[w_base_7 + 28];
      acc[0] = acc[0] + ((wts[28] > 0.0f) ? wts[28] * __uint_as_float(vec[56] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[28] > 0.0f) ? wts[28] * __uint_as_float(vec[56] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[28] > 0.0f) ? wts[28] * __uint_as_float(vec[57] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[28] > 0.0f) ? wts[28] * __uint_as_float(vec[57] & 4294901760u) : 0.0f);
      wts[29] = smem_w[w_base_7 + 29];
      acc[0] = acc[0] + ((wts[29] > 0.0f) ? wts[29] * __uint_as_float(vec[58] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[29] > 0.0f) ? wts[29] * __uint_as_float(vec[58] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[29] > 0.0f) ? wts[29] * __uint_as_float(vec[59] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[29] > 0.0f) ? wts[29] * __uint_as_float(vec[59] & 4294901760u) : 0.0f);
      wts[30] = smem_w[w_base_7 + 30];
      acc[0] = acc[0] + ((wts[30] > 0.0f) ? wts[30] * __uint_as_float(vec[60] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[30] > 0.0f) ? wts[30] * __uint_as_float(vec[60] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[30] > 0.0f) ? wts[30] * __uint_as_float(vec[61] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[30] > 0.0f) ? wts[30] * __uint_as_float(vec[61] & 4294901760u) : 0.0f);
      wts[31] = smem_w[w_base_7 + 31];
      acc[0] = acc[0] + ((wts[31] > 0.0f) ? wts[31] * __uint_as_float(vec[62] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[31] > 0.0f) ? wts[31] * __uint_as_float(vec[62] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[31] > 0.0f) ? wts[31] * __uint_as_float(vec[63] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[31] > 0.0f) ? wts[31] * __uint_as_float(vec[63] & 4294901760u) : 0.0f);
    }
    if (32 < num_split) {
      wts[32] = smem_w[w_base_7 + 32];
      acc[0] = acc[0] + ((wts[32] > 0.0f) ? wts[32] * __uint_as_float(vec[64] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[32] > 0.0f) ? wts[32] * __uint_as_float(vec[64] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[32] > 0.0f) ? wts[32] * __uint_as_float(vec[65] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[32] > 0.0f) ? wts[32] * __uint_as_float(vec[65] & 4294901760u) : 0.0f);
      wts[33] = smem_w[w_base_7 + 33];
      acc[0] = acc[0] + ((wts[33] > 0.0f) ? wts[33] * __uint_as_float(vec[66] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[33] > 0.0f) ? wts[33] * __uint_as_float(vec[66] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[33] > 0.0f) ? wts[33] * __uint_as_float(vec[67] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[33] > 0.0f) ? wts[33] * __uint_as_float(vec[67] & 4294901760u) : 0.0f);
      wts[34] = smem_w[w_base_7 + 34];
      acc[0] = acc[0] + ((wts[34] > 0.0f) ? wts[34] * __uint_as_float(vec[68] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[34] > 0.0f) ? wts[34] * __uint_as_float(vec[68] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[34] > 0.0f) ? wts[34] * __uint_as_float(vec[69] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[34] > 0.0f) ? wts[34] * __uint_as_float(vec[69] & 4294901760u) : 0.0f);
      wts[35] = smem_w[w_base_7 + 35];
      acc[0] = acc[0] + ((wts[35] > 0.0f) ? wts[35] * __uint_as_float(vec[70] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[35] > 0.0f) ? wts[35] * __uint_as_float(vec[70] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[35] > 0.0f) ? wts[35] * __uint_as_float(vec[71] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[35] > 0.0f) ? wts[35] * __uint_as_float(vec[71] & 4294901760u) : 0.0f);
      wts[36] = smem_w[w_base_7 + 36];
      acc[0] = acc[0] + ((wts[36] > 0.0f) ? wts[36] * __uint_as_float(vec[72] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[36] > 0.0f) ? wts[36] * __uint_as_float(vec[72] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[36] > 0.0f) ? wts[36] * __uint_as_float(vec[73] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[36] > 0.0f) ? wts[36] * __uint_as_float(vec[73] & 4294901760u) : 0.0f);
      wts[37] = smem_w[w_base_7 + 37];
      acc[0] = acc[0] + ((wts[37] > 0.0f) ? wts[37] * __uint_as_float(vec[74] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[37] > 0.0f) ? wts[37] * __uint_as_float(vec[74] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[37] > 0.0f) ? wts[37] * __uint_as_float(vec[75] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[37] > 0.0f) ? wts[37] * __uint_as_float(vec[75] & 4294901760u) : 0.0f);
      wts[38] = smem_w[w_base_7 + 38];
      acc[0] = acc[0] + ((wts[38] > 0.0f) ? wts[38] * __uint_as_float(vec[76] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[38] > 0.0f) ? wts[38] * __uint_as_float(vec[76] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[38] > 0.0f) ? wts[38] * __uint_as_float(vec[77] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[38] > 0.0f) ? wts[38] * __uint_as_float(vec[77] & 4294901760u) : 0.0f);
      wts[39] = smem_w[w_base_7 + 39];
      acc[0] = acc[0] + ((wts[39] > 0.0f) ? wts[39] * __uint_as_float(vec[78] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[39] > 0.0f) ? wts[39] * __uint_as_float(vec[78] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[39] > 0.0f) ? wts[39] * __uint_as_float(vec[79] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[39] > 0.0f) ? wts[39] * __uint_as_float(vec[79] & 4294901760u) : 0.0f);
    }
    if (40 < num_split) {
      wts[40] = smem_w[w_base_7 + 40];
      acc[0] = acc[0] + ((wts[40] > 0.0f) ? wts[40] * __uint_as_float(vec[80] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[40] > 0.0f) ? wts[40] * __uint_as_float(vec[80] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[40] > 0.0f) ? wts[40] * __uint_as_float(vec[81] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[40] > 0.0f) ? wts[40] * __uint_as_float(vec[81] & 4294901760u) : 0.0f);
      wts[41] = smem_w[w_base_7 + 41];
      acc[0] = acc[0] + ((wts[41] > 0.0f) ? wts[41] * __uint_as_float(vec[82] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[41] > 0.0f) ? wts[41] * __uint_as_float(vec[82] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[41] > 0.0f) ? wts[41] * __uint_as_float(vec[83] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[41] > 0.0f) ? wts[41] * __uint_as_float(vec[83] & 4294901760u) : 0.0f);
      wts[42] = smem_w[w_base_7 + 42];
      acc[0] = acc[0] + ((wts[42] > 0.0f) ? wts[42] * __uint_as_float(vec[84] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[42] > 0.0f) ? wts[42] * __uint_as_float(vec[84] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[42] > 0.0f) ? wts[42] * __uint_as_float(vec[85] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[42] > 0.0f) ? wts[42] * __uint_as_float(vec[85] & 4294901760u) : 0.0f);
      wts[43] = smem_w[w_base_7 + 43];
      acc[0] = acc[0] + ((wts[43] > 0.0f) ? wts[43] * __uint_as_float(vec[86] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[43] > 0.0f) ? wts[43] * __uint_as_float(vec[86] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[43] > 0.0f) ? wts[43] * __uint_as_float(vec[87] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[43] > 0.0f) ? wts[43] * __uint_as_float(vec[87] & 4294901760u) : 0.0f);
      wts[44] = smem_w[w_base_7 + 44];
      acc[0] = acc[0] + ((wts[44] > 0.0f) ? wts[44] * __uint_as_float(vec[88] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[44] > 0.0f) ? wts[44] * __uint_as_float(vec[88] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[44] > 0.0f) ? wts[44] * __uint_as_float(vec[89] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[44] > 0.0f) ? wts[44] * __uint_as_float(vec[89] & 4294901760u) : 0.0f);
      wts[45] = smem_w[w_base_7 + 45];
      acc[0] = acc[0] + ((wts[45] > 0.0f) ? wts[45] * __uint_as_float(vec[90] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[45] > 0.0f) ? wts[45] * __uint_as_float(vec[90] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[45] > 0.0f) ? wts[45] * __uint_as_float(vec[91] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[45] > 0.0f) ? wts[45] * __uint_as_float(vec[91] & 4294901760u) : 0.0f);
      wts[46] = smem_w[w_base_7 + 46];
      acc[0] = acc[0] + ((wts[46] > 0.0f) ? wts[46] * __uint_as_float(vec[92] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[46] > 0.0f) ? wts[46] * __uint_as_float(vec[92] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[46] > 0.0f) ? wts[46] * __uint_as_float(vec[93] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[46] > 0.0f) ? wts[46] * __uint_as_float(vec[93] & 4294901760u) : 0.0f);
      wts[47] = smem_w[w_base_7 + 47];
      acc[0] = acc[0] + ((wts[47] > 0.0f) ? wts[47] * __uint_as_float(vec[94] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[47] > 0.0f) ? wts[47] * __uint_as_float(vec[94] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[47] > 0.0f) ? wts[47] * __uint_as_float(vec[95] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[47] > 0.0f) ? wts[47] * __uint_as_float(vec[95] & 4294901760u) : 0.0f);
    }
    if (48 < num_split) {
      wts[48] = smem_w[w_base_7 + 48];
      acc[0] = acc[0] + ((wts[48] > 0.0f) ? wts[48] * __uint_as_float(vec[96] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[48] > 0.0f) ? wts[48] * __uint_as_float(vec[96] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[48] > 0.0f) ? wts[48] * __uint_as_float(vec[97] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[48] > 0.0f) ? wts[48] * __uint_as_float(vec[97] & 4294901760u) : 0.0f);
      wts[49] = smem_w[w_base_7 + 49];
      acc[0] = acc[0] + ((wts[49] > 0.0f) ? wts[49] * __uint_as_float(vec[98] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[49] > 0.0f) ? wts[49] * __uint_as_float(vec[98] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[49] > 0.0f) ? wts[49] * __uint_as_float(vec[99] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[49] > 0.0f) ? wts[49] * __uint_as_float(vec[99] & 4294901760u) : 0.0f);
      wts[50] = smem_w[w_base_7 + 50];
      acc[0] = acc[0] + ((wts[50] > 0.0f) ? wts[50] * __uint_as_float(vec[100] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[50] > 0.0f) ? wts[50] * __uint_as_float(vec[100] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[50] > 0.0f) ? wts[50] * __uint_as_float(vec[101] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[50] > 0.0f) ? wts[50] * __uint_as_float(vec[101] & 4294901760u) : 0.0f);
      wts[51] = smem_w[w_base_7 + 51];
      acc[0] = acc[0] + ((wts[51] > 0.0f) ? wts[51] * __uint_as_float(vec[102] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[51] > 0.0f) ? wts[51] * __uint_as_float(vec[102] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[51] > 0.0f) ? wts[51] * __uint_as_float(vec[103] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[51] > 0.0f) ? wts[51] * __uint_as_float(vec[103] & 4294901760u) : 0.0f);
      wts[52] = smem_w[w_base_7 + 52];
      acc[0] = acc[0] + ((wts[52] > 0.0f) ? wts[52] * __uint_as_float(vec[104] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[52] > 0.0f) ? wts[52] * __uint_as_float(vec[104] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[52] > 0.0f) ? wts[52] * __uint_as_float(vec[105] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[52] > 0.0f) ? wts[52] * __uint_as_float(vec[105] & 4294901760u) : 0.0f);
      wts[53] = smem_w[w_base_7 + 53];
      acc[0] = acc[0] + ((wts[53] > 0.0f) ? wts[53] * __uint_as_float(vec[106] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[53] > 0.0f) ? wts[53] * __uint_as_float(vec[106] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[53] > 0.0f) ? wts[53] * __uint_as_float(vec[107] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[53] > 0.0f) ? wts[53] * __uint_as_float(vec[107] & 4294901760u) : 0.0f);
      wts[54] = smem_w[w_base_7 + 54];
      acc[0] = acc[0] + ((wts[54] > 0.0f) ? wts[54] * __uint_as_float(vec[108] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[54] > 0.0f) ? wts[54] * __uint_as_float(vec[108] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[54] > 0.0f) ? wts[54] * __uint_as_float(vec[109] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[54] > 0.0f) ? wts[54] * __uint_as_float(vec[109] & 4294901760u) : 0.0f);
      wts[55] = smem_w[w_base_7 + 55];
      acc[0] = acc[0] + ((wts[55] > 0.0f) ? wts[55] * __uint_as_float(vec[110] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[55] > 0.0f) ? wts[55] * __uint_as_float(vec[110] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[55] > 0.0f) ? wts[55] * __uint_as_float(vec[111] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[55] > 0.0f) ? wts[55] * __uint_as_float(vec[111] & 4294901760u) : 0.0f);
    }
    if (56 < num_split) {
      wts[56] = smem_w[w_base_7 + 56];
      acc[0] = acc[0] + ((wts[56] > 0.0f) ? wts[56] * __uint_as_float(vec[112] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[56] > 0.0f) ? wts[56] * __uint_as_float(vec[112] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[56] > 0.0f) ? wts[56] * __uint_as_float(vec[113] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[56] > 0.0f) ? wts[56] * __uint_as_float(vec[113] & 4294901760u) : 0.0f);
      wts[57] = smem_w[w_base_7 + 57];
      acc[0] = acc[0] + ((wts[57] > 0.0f) ? wts[57] * __uint_as_float(vec[114] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[57] > 0.0f) ? wts[57] * __uint_as_float(vec[114] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[57] > 0.0f) ? wts[57] * __uint_as_float(vec[115] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[57] > 0.0f) ? wts[57] * __uint_as_float(vec[115] & 4294901760u) : 0.0f);
      wts[58] = smem_w[w_base_7 + 58];
      acc[0] = acc[0] + ((wts[58] > 0.0f) ? wts[58] * __uint_as_float(vec[116] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[58] > 0.0f) ? wts[58] * __uint_as_float(vec[116] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[58] > 0.0f) ? wts[58] * __uint_as_float(vec[117] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[58] > 0.0f) ? wts[58] * __uint_as_float(vec[117] & 4294901760u) : 0.0f);
      wts[59] = smem_w[w_base_7 + 59];
      acc[0] = acc[0] + ((wts[59] > 0.0f) ? wts[59] * __uint_as_float(vec[118] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[59] > 0.0f) ? wts[59] * __uint_as_float(vec[118] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[59] > 0.0f) ? wts[59] * __uint_as_float(vec[119] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[59] > 0.0f) ? wts[59] * __uint_as_float(vec[119] & 4294901760u) : 0.0f);
      wts[60] = smem_w[w_base_7 + 60];
      acc[0] = acc[0] + ((wts[60] > 0.0f) ? wts[60] * __uint_as_float(vec[120] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[60] > 0.0f) ? wts[60] * __uint_as_float(vec[120] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[60] > 0.0f) ? wts[60] * __uint_as_float(vec[121] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[60] > 0.0f) ? wts[60] * __uint_as_float(vec[121] & 4294901760u) : 0.0f);
      wts[61] = smem_w[w_base_7 + 61];
      acc[0] = acc[0] + ((wts[61] > 0.0f) ? wts[61] * __uint_as_float(vec[122] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[61] > 0.0f) ? wts[61] * __uint_as_float(vec[122] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[61] > 0.0f) ? wts[61] * __uint_as_float(vec[123] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[61] > 0.0f) ? wts[61] * __uint_as_float(vec[123] & 4294901760u) : 0.0f);
      wts[62] = smem_w[w_base_7 + 62];
      acc[0] = acc[0] + ((wts[62] > 0.0f) ? wts[62] * __uint_as_float(vec[124] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[62] > 0.0f) ? wts[62] * __uint_as_float(vec[124] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[62] > 0.0f) ? wts[62] * __uint_as_float(vec[125] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[62] > 0.0f) ? wts[62] * __uint_as_float(vec[125] & 4294901760u) : 0.0f);
      wts[63] = smem_w[w_base_7 + 63];
      acc[0] = acc[0] + ((wts[63] > 0.0f) ? wts[63] * __uint_as_float(vec[126] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[63] > 0.0f) ? wts[63] * __uint_as_float(vec[126] & 4294901760u) : 0.0f);
      acc[2] = acc[2] + ((wts[63] > 0.0f) ? wts[63] * __uint_as_float(vec[127] << 16) : 0.0f);
      acc[3] =
          acc[3] + ((wts[63] > 0.0f) ? wts[63] * __uint_as_float(vec[127] & 4294901760u) : 0.0f);
    }
    for (int e_1 = 0; e_1 < 4; e_1++) {
      acc[e_1] = acc[e_1] * inv;
    }
    unsigned int _packed[2];
#pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
      __nv_bfloat162 _bf2 =
          __float22bfloat162_rn(make_float2(acc[_lp * 2 + 0], acc[_lp * 2 + 1 + 0]));
      _packed[_lp] = *(uint32_t*)&_bf2;
    }
#pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
      __nv_bfloat162 _bf2 =
          __float22bfloat162_rn(make_float2(acc[_lp * 2 + 2], acc[_lp * 2 + 1 + 2]));
      _packed[_lp + 1] = *(uint32_t*)&_bf2;
    }
    if (merged == 0) {
      reinterpret_cast<int2*>(O + (row_7 * 512 + d0_7))[0] = reinterpret_cast<int2*>(_packed)[0];
      if (has_lse != 0) {
        if (part_7 == 0) {
          if (lane_7 == 0) {
            float lse_v = -CUDART_INF_F;
            if (sum_w > 0.0f) {
              float _log2_68;
              asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_68) : "f"(sum_w));
              lse_v = (m + _log2_68) * 0.6931472f;
            }
            *(reinterpret_cast<float*>(lse + row_7) + (0)) = lse_v;
          }
        }
      }
    }
  }
}

}  // extern "C"
