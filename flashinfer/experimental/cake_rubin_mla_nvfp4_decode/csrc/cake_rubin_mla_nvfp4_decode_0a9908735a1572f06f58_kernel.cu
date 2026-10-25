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
#define SMEM_SMEM_W_STAGE_BYTES 2048
#define SMEM_SMEM_W_STRIDE 2048
#define SMEM_TOTAL 2048
#define THREADS 256
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
__launch_bounds__(256, LAUNCH_MIN_BLOCKS) void kernel_cake_rubin_mla_nvfp4_decode_0a9908735a1572f06f58(
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
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int part = warp % 8;
  int row = blockIdx.x + warp / 8;
  int merged = 0;
  if (row < rows) {
    merged = partial_lse[row * num_split] == CUDART_INF_F;
  }
  if (row < rows && merged == 0) {
    int lane_0 = lane;
    int stat_base = row * num_split;
    int w_base = warp * 64;
    int d0 = part * 64 + lane_0 * 2;
    int last = num_split - 1;
    if (num_split > 32) {
      int _min_0 = ((0) < (last) ? (0) : (last));
      unsigned int _vec_load_0[2];
      {
        uint32_t _bf16x2_bits_0;
        _bf16x2_bits_0 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_0) * 512 + d0);
        _vec_load_0[0] = _bf16x2_bits_0;
      }
      int _min_1 = ((1) < (last) ? (1) : (last));
      unsigned int _vec_load_1[2];
      {
        uint32_t _bf16x2_bits_1;
        _bf16x2_bits_1 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_1) * 512 + d0);
        _vec_load_1[0] = _bf16x2_bits_1;
      }
      int _min_2 = ((2) < (last) ? (2) : (last));
      unsigned int _vec_load_2[2];
      {
        uint32_t _bf16x2_bits_2;
        _bf16x2_bits_2 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_2) * 512 + d0);
        _vec_load_2[0] = _bf16x2_bits_2;
      }
      int _min_3 = ((3) < (last) ? (3) : (last));
      unsigned int _vec_load_3[2];
      {
        uint32_t _bf16x2_bits_3;
        _bf16x2_bits_3 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_3) * 512 + d0);
        _vec_load_3[0] = _bf16x2_bits_3;
      }
      int _min_4 = ((4) < (last) ? (4) : (last));
      unsigned int _vec_load_4[2];
      {
        uint32_t _bf16x2_bits_4;
        _bf16x2_bits_4 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_4) * 512 + d0);
        _vec_load_4[0] = _bf16x2_bits_4;
      }
      int _min_5 = ((5) < (last) ? (5) : (last));
      unsigned int _vec_load_5[2];
      {
        uint32_t _bf16x2_bits_5;
        _bf16x2_bits_5 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_5) * 512 + d0);
        _vec_load_5[0] = _bf16x2_bits_5;
      }
      int _min_6 = ((6) < (last) ? (6) : (last));
      unsigned int _vec_load_6[2];
      {
        uint32_t _bf16x2_bits_6;
        _bf16x2_bits_6 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_6) * 512 + d0);
        _vec_load_6[0] = _bf16x2_bits_6;
      }
      int _min_7 = ((7) < (last) ? (7) : (last));
      unsigned int _vec_load_7[2];
      {
        uint32_t _bf16x2_bits_7;
        _bf16x2_bits_7 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_7) * 512 + d0);
        _vec_load_7[0] = _bf16x2_bits_7;
      }
      int _min_8 = ((8) < (last) ? (8) : (last));
      unsigned int _vec_load_8[2];
      {
        uint32_t _bf16x2_bits_8;
        _bf16x2_bits_8 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_8) * 512 + d0);
        _vec_load_8[0] = _bf16x2_bits_8;
      }
      int _min_9 = ((9) < (last) ? (9) : (last));
      unsigned int _vec_load_9[2];
      {
        uint32_t _bf16x2_bits_9;
        _bf16x2_bits_9 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_9) * 512 + d0);
        _vec_load_9[0] = _bf16x2_bits_9;
      }
      int _min_10 = ((10) < (last) ? (10) : (last));
      unsigned int _vec_load_10[2];
      {
        uint32_t _bf16x2_bits_10;
        _bf16x2_bits_10 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_10) * 512 + d0);
        _vec_load_10[0] = _bf16x2_bits_10;
      }
      int _min_11 = ((11) < (last) ? (11) : (last));
      unsigned int _vec_load_11[2];
      {
        uint32_t _bf16x2_bits_11;
        _bf16x2_bits_11 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_11) * 512 + d0);
        _vec_load_11[0] = _bf16x2_bits_11;
      }
      int _min_12 = ((12) < (last) ? (12) : (last));
      unsigned int _vec_load_12[2];
      {
        uint32_t _bf16x2_bits_12;
        _bf16x2_bits_12 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_12) * 512 + d0);
        _vec_load_12[0] = _bf16x2_bits_12;
      }
      int _min_13 = ((13) < (last) ? (13) : (last));
      unsigned int _vec_load_13[2];
      {
        uint32_t _bf16x2_bits_13;
        _bf16x2_bits_13 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_13) * 512 + d0);
        _vec_load_13[0] = _bf16x2_bits_13;
      }
      int _min_14 = ((14) < (last) ? (14) : (last));
      unsigned int _vec_load_14[2];
      {
        uint32_t _bf16x2_bits_14;
        _bf16x2_bits_14 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_14) * 512 + d0);
        _vec_load_14[0] = _bf16x2_bits_14;
      }
      int _min_15 = ((15) < (last) ? (15) : (last));
      unsigned int _vec_load_15[2];
      {
        uint32_t _bf16x2_bits_15;
        _bf16x2_bits_15 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_15) * 512 + d0);
        _vec_load_15[0] = _bf16x2_bits_15;
      }
      int _min_16 = ((16) < (last) ? (16) : (last));
      unsigned int _vec_load_16[2];
      {
        uint32_t _bf16x2_bits_16;
        _bf16x2_bits_16 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_16) * 512 + d0);
        _vec_load_16[0] = _bf16x2_bits_16;
      }
      int _min_17 = ((17) < (last) ? (17) : (last));
      unsigned int _vec_load_17[2];
      {
        uint32_t _bf16x2_bits_17;
        _bf16x2_bits_17 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_17) * 512 + d0);
        _vec_load_17[0] = _bf16x2_bits_17;
      }
      int _min_18 = ((18) < (last) ? (18) : (last));
      unsigned int _vec_load_18[2];
      {
        uint32_t _bf16x2_bits_18;
        _bf16x2_bits_18 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_18) * 512 + d0);
        _vec_load_18[0] = _bf16x2_bits_18;
      }
      int _min_19 = ((19) < (last) ? (19) : (last));
      unsigned int _vec_load_19[2];
      {
        uint32_t _bf16x2_bits_19;
        _bf16x2_bits_19 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_19) * 512 + d0);
        _vec_load_19[0] = _bf16x2_bits_19;
      }
      int _min_20 = ((20) < (last) ? (20) : (last));
      unsigned int _vec_load_20[2];
      {
        uint32_t _bf16x2_bits_20;
        _bf16x2_bits_20 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_20) * 512 + d0);
        _vec_load_20[0] = _bf16x2_bits_20;
      }
      int _min_21 = ((21) < (last) ? (21) : (last));
      unsigned int _vec_load_21[2];
      {
        uint32_t _bf16x2_bits_21;
        _bf16x2_bits_21 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_21) * 512 + d0);
        _vec_load_21[0] = _bf16x2_bits_21;
      }
      int _min_22 = ((22) < (last) ? (22) : (last));
      unsigned int _vec_load_22[2];
      {
        uint32_t _bf16x2_bits_22;
        _bf16x2_bits_22 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_22) * 512 + d0);
        _vec_load_22[0] = _bf16x2_bits_22;
      }
      int _min_23 = ((23) < (last) ? (23) : (last));
      unsigned int _vec_load_23[2];
      {
        uint32_t _bf16x2_bits_23;
        _bf16x2_bits_23 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_23) * 512 + d0);
        _vec_load_23[0] = _bf16x2_bits_23;
      }
      int _min_24 = ((24) < (last) ? (24) : (last));
      unsigned int _vec_load_24[2];
      {
        uint32_t _bf16x2_bits_24;
        _bf16x2_bits_24 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_24) * 512 + d0);
        _vec_load_24[0] = _bf16x2_bits_24;
      }
      int _min_25 = ((25) < (last) ? (25) : (last));
      unsigned int _vec_load_25[2];
      {
        uint32_t _bf16x2_bits_25;
        _bf16x2_bits_25 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_25) * 512 + d0);
        _vec_load_25[0] = _bf16x2_bits_25;
      }
      int _min_26 = ((26) < (last) ? (26) : (last));
      unsigned int _vec_load_26[2];
      {
        uint32_t _bf16x2_bits_26;
        _bf16x2_bits_26 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_26) * 512 + d0);
        _vec_load_26[0] = _bf16x2_bits_26;
      }
      int _min_27 = ((27) < (last) ? (27) : (last));
      unsigned int _vec_load_27[2];
      {
        uint32_t _bf16x2_bits_27;
        _bf16x2_bits_27 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_27) * 512 + d0);
        _vec_load_27[0] = _bf16x2_bits_27;
      }
      int _min_28 = ((28) < (last) ? (28) : (last));
      unsigned int _vec_load_28[2];
      {
        uint32_t _bf16x2_bits_28;
        _bf16x2_bits_28 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_28) * 512 + d0);
        _vec_load_28[0] = _bf16x2_bits_28;
      }
      int _min_29 = ((29) < (last) ? (29) : (last));
      unsigned int _vec_load_29[2];
      {
        uint32_t _bf16x2_bits_29;
        _bf16x2_bits_29 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_29) * 512 + d0);
        _vec_load_29[0] = _bf16x2_bits_29;
      }
      int _min_30 = ((30) < (last) ? (30) : (last));
      unsigned int _vec_load_30[2];
      {
        uint32_t _bf16x2_bits_30;
        _bf16x2_bits_30 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_30) * 512 + d0);
        _vec_load_30[0] = _bf16x2_bits_30;
      }
      int _min_31 = ((31) < (last) ? (31) : (last));
      unsigned int _vec_load_31[2];
      {
        uint32_t _bf16x2_bits_31;
        _bf16x2_bits_31 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_31) * 512 + d0);
        _vec_load_31[0] = _bf16x2_bits_31;
      }
      int _min_32 = ((32) < (last) ? (32) : (last));
      unsigned int _vec_load_32[2];
      {
        uint32_t _bf16x2_bits_32;
        _bf16x2_bits_32 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_32) * 512 + d0);
        _vec_load_32[0] = _bf16x2_bits_32;
      }
      int _min_33 = ((33) < (last) ? (33) : (last));
      unsigned int _vec_load_33[2];
      {
        uint32_t _bf16x2_bits_33;
        _bf16x2_bits_33 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_33) * 512 + d0);
        _vec_load_33[0] = _bf16x2_bits_33;
      }
      int _min_34 = ((34) < (last) ? (34) : (last));
      unsigned int _vec_load_34[2];
      {
        uint32_t _bf16x2_bits_34;
        _bf16x2_bits_34 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_34) * 512 + d0);
        _vec_load_34[0] = _bf16x2_bits_34;
      }
      int _min_35 = ((35) < (last) ? (35) : (last));
      unsigned int _vec_load_35[2];
      {
        uint32_t _bf16x2_bits_35;
        _bf16x2_bits_35 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_35) * 512 + d0);
        _vec_load_35[0] = _bf16x2_bits_35;
      }
      int _min_36 = ((36) < (last) ? (36) : (last));
      unsigned int _vec_load_36[2];
      {
        uint32_t _bf16x2_bits_36;
        _bf16x2_bits_36 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_36) * 512 + d0);
        _vec_load_36[0] = _bf16x2_bits_36;
      }
      int _min_37 = ((37) < (last) ? (37) : (last));
      unsigned int _vec_load_37[2];
      {
        uint32_t _bf16x2_bits_37;
        _bf16x2_bits_37 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_37) * 512 + d0);
        _vec_load_37[0] = _bf16x2_bits_37;
      }
      int _min_38 = ((38) < (last) ? (38) : (last));
      unsigned int _vec_load_38[2];
      {
        uint32_t _bf16x2_bits_38;
        _bf16x2_bits_38 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_38) * 512 + d0);
        _vec_load_38[0] = _bf16x2_bits_38;
      }
      int _min_39 = ((39) < (last) ? (39) : (last));
      unsigned int _vec_load_39[2];
      {
        uint32_t _bf16x2_bits_39;
        _bf16x2_bits_39 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_39) * 512 + d0);
        _vec_load_39[0] = _bf16x2_bits_39;
      }
      int _min_40 = ((40) < (last) ? (40) : (last));
      unsigned int _vec_load_40[2];
      {
        uint32_t _bf16x2_bits_40;
        _bf16x2_bits_40 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_40) * 512 + d0);
        _vec_load_40[0] = _bf16x2_bits_40;
      }
      int _min_41 = ((41) < (last) ? (41) : (last));
      unsigned int _vec_load_41[2];
      {
        uint32_t _bf16x2_bits_41;
        _bf16x2_bits_41 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_41) * 512 + d0);
        _vec_load_41[0] = _bf16x2_bits_41;
      }
      int _min_42 = ((42) < (last) ? (42) : (last));
      unsigned int _vec_load_42[2];
      {
        uint32_t _bf16x2_bits_42;
        _bf16x2_bits_42 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_42) * 512 + d0);
        _vec_load_42[0] = _bf16x2_bits_42;
      }
      int _min_43 = ((43) < (last) ? (43) : (last));
      unsigned int _vec_load_43[2];
      {
        uint32_t _bf16x2_bits_43;
        _bf16x2_bits_43 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_43) * 512 + d0);
        _vec_load_43[0] = _bf16x2_bits_43;
      }
      int _min_44 = ((44) < (last) ? (44) : (last));
      unsigned int _vec_load_44[2];
      {
        uint32_t _bf16x2_bits_44;
        _bf16x2_bits_44 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_44) * 512 + d0);
        _vec_load_44[0] = _bf16x2_bits_44;
      }
      int _min_45 = ((45) < (last) ? (45) : (last));
      unsigned int _vec_load_45[2];
      {
        uint32_t _bf16x2_bits_45;
        _bf16x2_bits_45 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_45) * 512 + d0);
        _vec_load_45[0] = _bf16x2_bits_45;
      }
      int _min_46 = ((46) < (last) ? (46) : (last));
      unsigned int _vec_load_46[2];
      {
        uint32_t _bf16x2_bits_46;
        _bf16x2_bits_46 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_46) * 512 + d0);
        _vec_load_46[0] = _bf16x2_bits_46;
      }
      int _min_47 = ((47) < (last) ? (47) : (last));
      unsigned int _vec_load_47[2];
      {
        uint32_t _bf16x2_bits_47;
        _bf16x2_bits_47 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_47) * 512 + d0);
        _vec_load_47[0] = _bf16x2_bits_47;
      }
      int _min_48 = ((48) < (last) ? (48) : (last));
      unsigned int _vec_load_48[2];
      {
        uint32_t _bf16x2_bits_48;
        _bf16x2_bits_48 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_48) * 512 + d0);
        _vec_load_48[0] = _bf16x2_bits_48;
      }
      int _min_49 = ((49) < (last) ? (49) : (last));
      unsigned int _vec_load_49[2];
      {
        uint32_t _bf16x2_bits_49;
        _bf16x2_bits_49 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_49) * 512 + d0);
        _vec_load_49[0] = _bf16x2_bits_49;
      }
      int _min_50 = ((50) < (last) ? (50) : (last));
      unsigned int _vec_load_50[2];
      {
        uint32_t _bf16x2_bits_50;
        _bf16x2_bits_50 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_50) * 512 + d0);
        _vec_load_50[0] = _bf16x2_bits_50;
      }
      int _min_51 = ((51) < (last) ? (51) : (last));
      unsigned int _vec_load_51[2];
      {
        uint32_t _bf16x2_bits_51;
        _bf16x2_bits_51 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_51) * 512 + d0);
        _vec_load_51[0] = _bf16x2_bits_51;
      }
      int _min_52 = ((52) < (last) ? (52) : (last));
      unsigned int _vec_load_52[2];
      {
        uint32_t _bf16x2_bits_52;
        _bf16x2_bits_52 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_52) * 512 + d0);
        _vec_load_52[0] = _bf16x2_bits_52;
      }
      int _min_53 = ((53) < (last) ? (53) : (last));
      unsigned int _vec_load_53[2];
      {
        uint32_t _bf16x2_bits_53;
        _bf16x2_bits_53 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_53) * 512 + d0);
        _vec_load_53[0] = _bf16x2_bits_53;
      }
      int _min_54 = ((54) < (last) ? (54) : (last));
      unsigned int _vec_load_54[2];
      {
        uint32_t _bf16x2_bits_54;
        _bf16x2_bits_54 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_54) * 512 + d0);
        _vec_load_54[0] = _bf16x2_bits_54;
      }
      int _min_55 = ((55) < (last) ? (55) : (last));
      unsigned int _vec_load_55[2];
      {
        uint32_t _bf16x2_bits_55;
        _bf16x2_bits_55 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_55) * 512 + d0);
        _vec_load_55[0] = _bf16x2_bits_55;
      }
      int _min_56 = ((56) < (last) ? (56) : (last));
      unsigned int _vec_load_56[2];
      {
        uint32_t _bf16x2_bits_56;
        _bf16x2_bits_56 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_56) * 512 + d0);
        _vec_load_56[0] = _bf16x2_bits_56;
      }
      int _min_57 = ((57) < (last) ? (57) : (last));
      unsigned int _vec_load_57[2];
      {
        uint32_t _bf16x2_bits_57;
        _bf16x2_bits_57 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_57) * 512 + d0);
        _vec_load_57[0] = _bf16x2_bits_57;
      }
      int _min_58 = ((58) < (last) ? (58) : (last));
      unsigned int _vec_load_58[2];
      {
        uint32_t _bf16x2_bits_58;
        _bf16x2_bits_58 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_58) * 512 + d0);
        _vec_load_58[0] = _bf16x2_bits_58;
      }
      int _min_59 = ((59) < (last) ? (59) : (last));
      unsigned int _vec_load_59[2];
      {
        uint32_t _bf16x2_bits_59;
        _bf16x2_bits_59 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_59) * 512 + d0);
        _vec_load_59[0] = _bf16x2_bits_59;
      }
      int _min_60 = ((60) < (last) ? (60) : (last));
      unsigned int _vec_load_60[2];
      {
        uint32_t _bf16x2_bits_60;
        _bf16x2_bits_60 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_60) * 512 + d0);
        _vec_load_60[0] = _bf16x2_bits_60;
      }
      int _min_61 = ((61) < (last) ? (61) : (last));
      unsigned int _vec_load_61[2];
      {
        uint32_t _bf16x2_bits_61;
        _bf16x2_bits_61 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_61) * 512 + d0);
        _vec_load_61[0] = _bf16x2_bits_61;
      }
      int _min_62 = ((62) < (last) ? (62) : (last));
      unsigned int _vec_load_62[2];
      {
        uint32_t _bf16x2_bits_62;
        _bf16x2_bits_62 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_62) * 512 + d0);
        _vec_load_62[0] = _bf16x2_bits_62;
      }
      int _min_63 = ((63) < (last) ? (63) : (last));
      unsigned int _vec_load_63[2];
      {
        uint32_t _bf16x2_bits_63;
        _bf16x2_bits_63 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_63) * 512 + d0);
        _vec_load_63[0] = _bf16x2_bits_63;
      }
      float l0 = -CUDART_INF_F;
      float l1 = -CUDART_INF_F;
      if (lane_0 < num_split) {
        l0 = partial_lse[stat_base + lane_0];
      }
      if (lane_0 + 32 < num_split) {
        l1 = partial_lse[stat_base + lane_0 + 32];
      }
      float m = max_noftz(l0, l1);
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        m = max_noftz(m, __shfl_xor_sync(0xFFFFFFFF, m, offset));
      float w0 = 0.0f;
      float w1 = 0.0f;
      if (l0 > -CUDART_INF_F) {
        float _exp2_64 = approx_exp2(l0 - m);
        w0 = _exp2_64;
      }
      if (l1 > -CUDART_INF_F) {
        float _exp2_65 = approx_exp2(l1 - m);
        w1 = _exp2_65;
      }
      smem_w[w_base + lane_0] = w0;
      smem_w[w_base + 32 + lane_0] = w1;
      float sum_w = w0 + w1;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        sum_w += __shfl_xor_sync(0xFFFFFFFF, sum_w, offset);
      __syncwarp();
      float inv = 0.0f;
      if (sum_w > 0.0f) {
        float _rcp_66 = approx_rcp(sum_w);
        inv = _rcp_66;
      }
      float acc[2];
      for (int e = 0; e < 2; e++) {
        acc[e] = 0.0f;
      }
      float wts[64];
      wts[0] = smem_w[w_base];
      acc[0] = acc[0] + ((wts[0] > 0.0f) ? wts[0] * __uint_as_float(_vec_load_0[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[0] > 0.0f) ? wts[0] * __uint_as_float(_vec_load_0[0] & 4294901760u) : 0.0f);
      wts[1] = smem_w[w_base + 1];
      acc[0] = acc[0] + ((wts[1] > 0.0f) ? wts[1] * __uint_as_float(_vec_load_1[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[1] > 0.0f) ? wts[1] * __uint_as_float(_vec_load_1[0] & 4294901760u) : 0.0f);
      wts[2] = smem_w[w_base + 2];
      acc[0] = acc[0] + ((wts[2] > 0.0f) ? wts[2] * __uint_as_float(_vec_load_2[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[2] > 0.0f) ? wts[2] * __uint_as_float(_vec_load_2[0] & 4294901760u) : 0.0f);
      wts[3] = smem_w[w_base + 3];
      acc[0] = acc[0] + ((wts[3] > 0.0f) ? wts[3] * __uint_as_float(_vec_load_3[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[3] > 0.0f) ? wts[3] * __uint_as_float(_vec_load_3[0] & 4294901760u) : 0.0f);
      wts[4] = smem_w[w_base + 4];
      acc[0] = acc[0] + ((wts[4] > 0.0f) ? wts[4] * __uint_as_float(_vec_load_4[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[4] > 0.0f) ? wts[4] * __uint_as_float(_vec_load_4[0] & 4294901760u) : 0.0f);
      wts[5] = smem_w[w_base + 5];
      acc[0] = acc[0] + ((wts[5] > 0.0f) ? wts[5] * __uint_as_float(_vec_load_5[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[5] > 0.0f) ? wts[5] * __uint_as_float(_vec_load_5[0] & 4294901760u) : 0.0f);
      wts[6] = smem_w[w_base + 6];
      acc[0] = acc[0] + ((wts[6] > 0.0f) ? wts[6] * __uint_as_float(_vec_load_6[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[6] > 0.0f) ? wts[6] * __uint_as_float(_vec_load_6[0] & 4294901760u) : 0.0f);
      wts[7] = smem_w[w_base + 7];
      acc[0] = acc[0] + ((wts[7] > 0.0f) ? wts[7] * __uint_as_float(_vec_load_7[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[7] > 0.0f) ? wts[7] * __uint_as_float(_vec_load_7[0] & 4294901760u) : 0.0f);
      wts[8] = smem_w[w_base + 8];
      acc[0] = acc[0] + ((wts[8] > 0.0f) ? wts[8] * __uint_as_float(_vec_load_8[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[8] > 0.0f) ? wts[8] * __uint_as_float(_vec_load_8[0] & 4294901760u) : 0.0f);
      wts[9] = smem_w[w_base + 9];
      acc[0] = acc[0] + ((wts[9] > 0.0f) ? wts[9] * __uint_as_float(_vec_load_9[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[9] > 0.0f) ? wts[9] * __uint_as_float(_vec_load_9[0] & 4294901760u) : 0.0f);
      wts[10] = smem_w[w_base + 10];
      acc[0] =
          acc[0] + ((wts[10] > 0.0f) ? wts[10] * __uint_as_float(_vec_load_10[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[10] > 0.0f) ? wts[10] * __uint_as_float(_vec_load_10[0] & 4294901760u) : 0.0f);
      wts[11] = smem_w[w_base + 11];
      acc[0] =
          acc[0] + ((wts[11] > 0.0f) ? wts[11] * __uint_as_float(_vec_load_11[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[11] > 0.0f) ? wts[11] * __uint_as_float(_vec_load_11[0] & 4294901760u) : 0.0f);
      wts[12] = smem_w[w_base + 12];
      acc[0] =
          acc[0] + ((wts[12] > 0.0f) ? wts[12] * __uint_as_float(_vec_load_12[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[12] > 0.0f) ? wts[12] * __uint_as_float(_vec_load_12[0] & 4294901760u) : 0.0f);
      wts[13] = smem_w[w_base + 13];
      acc[0] =
          acc[0] + ((wts[13] > 0.0f) ? wts[13] * __uint_as_float(_vec_load_13[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[13] > 0.0f) ? wts[13] * __uint_as_float(_vec_load_13[0] & 4294901760u) : 0.0f);
      wts[14] = smem_w[w_base + 14];
      acc[0] =
          acc[0] + ((wts[14] > 0.0f) ? wts[14] * __uint_as_float(_vec_load_14[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[14] > 0.0f) ? wts[14] * __uint_as_float(_vec_load_14[0] & 4294901760u) : 0.0f);
      wts[15] = smem_w[w_base + 15];
      acc[0] =
          acc[0] + ((wts[15] > 0.0f) ? wts[15] * __uint_as_float(_vec_load_15[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[15] > 0.0f) ? wts[15] * __uint_as_float(_vec_load_15[0] & 4294901760u) : 0.0f);
      wts[16] = smem_w[w_base + 16];
      acc[0] =
          acc[0] + ((wts[16] > 0.0f) ? wts[16] * __uint_as_float(_vec_load_16[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[16] > 0.0f) ? wts[16] * __uint_as_float(_vec_load_16[0] & 4294901760u) : 0.0f);
      wts[17] = smem_w[w_base + 17];
      acc[0] =
          acc[0] + ((wts[17] > 0.0f) ? wts[17] * __uint_as_float(_vec_load_17[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[17] > 0.0f) ? wts[17] * __uint_as_float(_vec_load_17[0] & 4294901760u) : 0.0f);
      wts[18] = smem_w[w_base + 18];
      acc[0] =
          acc[0] + ((wts[18] > 0.0f) ? wts[18] * __uint_as_float(_vec_load_18[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[18] > 0.0f) ? wts[18] * __uint_as_float(_vec_load_18[0] & 4294901760u) : 0.0f);
      wts[19] = smem_w[w_base + 19];
      acc[0] =
          acc[0] + ((wts[19] > 0.0f) ? wts[19] * __uint_as_float(_vec_load_19[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[19] > 0.0f) ? wts[19] * __uint_as_float(_vec_load_19[0] & 4294901760u) : 0.0f);
      wts[20] = smem_w[w_base + 20];
      acc[0] =
          acc[0] + ((wts[20] > 0.0f) ? wts[20] * __uint_as_float(_vec_load_20[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[20] > 0.0f) ? wts[20] * __uint_as_float(_vec_load_20[0] & 4294901760u) : 0.0f);
      wts[21] = smem_w[w_base + 21];
      acc[0] =
          acc[0] + ((wts[21] > 0.0f) ? wts[21] * __uint_as_float(_vec_load_21[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[21] > 0.0f) ? wts[21] * __uint_as_float(_vec_load_21[0] & 4294901760u) : 0.0f);
      wts[22] = smem_w[w_base + 22];
      acc[0] =
          acc[0] + ((wts[22] > 0.0f) ? wts[22] * __uint_as_float(_vec_load_22[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[22] > 0.0f) ? wts[22] * __uint_as_float(_vec_load_22[0] & 4294901760u) : 0.0f);
      wts[23] = smem_w[w_base + 23];
      acc[0] =
          acc[0] + ((wts[23] > 0.0f) ? wts[23] * __uint_as_float(_vec_load_23[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[23] > 0.0f) ? wts[23] * __uint_as_float(_vec_load_23[0] & 4294901760u) : 0.0f);
      wts[24] = smem_w[w_base + 24];
      acc[0] =
          acc[0] + ((wts[24] > 0.0f) ? wts[24] * __uint_as_float(_vec_load_24[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[24] > 0.0f) ? wts[24] * __uint_as_float(_vec_load_24[0] & 4294901760u) : 0.0f);
      wts[25] = smem_w[w_base + 25];
      acc[0] =
          acc[0] + ((wts[25] > 0.0f) ? wts[25] * __uint_as_float(_vec_load_25[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[25] > 0.0f) ? wts[25] * __uint_as_float(_vec_load_25[0] & 4294901760u) : 0.0f);
      wts[26] = smem_w[w_base + 26];
      acc[0] =
          acc[0] + ((wts[26] > 0.0f) ? wts[26] * __uint_as_float(_vec_load_26[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[26] > 0.0f) ? wts[26] * __uint_as_float(_vec_load_26[0] & 4294901760u) : 0.0f);
      wts[27] = smem_w[w_base + 27];
      acc[0] =
          acc[0] + ((wts[27] > 0.0f) ? wts[27] * __uint_as_float(_vec_load_27[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[27] > 0.0f) ? wts[27] * __uint_as_float(_vec_load_27[0] & 4294901760u) : 0.0f);
      wts[28] = smem_w[w_base + 28];
      acc[0] =
          acc[0] + ((wts[28] > 0.0f) ? wts[28] * __uint_as_float(_vec_load_28[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[28] > 0.0f) ? wts[28] * __uint_as_float(_vec_load_28[0] & 4294901760u) : 0.0f);
      wts[29] = smem_w[w_base + 29];
      acc[0] =
          acc[0] + ((wts[29] > 0.0f) ? wts[29] * __uint_as_float(_vec_load_29[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[29] > 0.0f) ? wts[29] * __uint_as_float(_vec_load_29[0] & 4294901760u) : 0.0f);
      wts[30] = smem_w[w_base + 30];
      acc[0] =
          acc[0] + ((wts[30] > 0.0f) ? wts[30] * __uint_as_float(_vec_load_30[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[30] > 0.0f) ? wts[30] * __uint_as_float(_vec_load_30[0] & 4294901760u) : 0.0f);
      wts[31] = smem_w[w_base + 31];
      acc[0] =
          acc[0] + ((wts[31] > 0.0f) ? wts[31] * __uint_as_float(_vec_load_31[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[31] > 0.0f) ? wts[31] * __uint_as_float(_vec_load_31[0] & 4294901760u) : 0.0f);
      wts[32] = smem_w[w_base + 32];
      acc[0] =
          acc[0] + ((wts[32] > 0.0f) ? wts[32] * __uint_as_float(_vec_load_32[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[32] > 0.0f) ? wts[32] * __uint_as_float(_vec_load_32[0] & 4294901760u) : 0.0f);
      wts[33] = smem_w[w_base + 33];
      acc[0] =
          acc[0] + ((wts[33] > 0.0f) ? wts[33] * __uint_as_float(_vec_load_33[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[33] > 0.0f) ? wts[33] * __uint_as_float(_vec_load_33[0] & 4294901760u) : 0.0f);
      wts[34] = smem_w[w_base + 34];
      acc[0] =
          acc[0] + ((wts[34] > 0.0f) ? wts[34] * __uint_as_float(_vec_load_34[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[34] > 0.0f) ? wts[34] * __uint_as_float(_vec_load_34[0] & 4294901760u) : 0.0f);
      wts[35] = smem_w[w_base + 35];
      acc[0] =
          acc[0] + ((wts[35] > 0.0f) ? wts[35] * __uint_as_float(_vec_load_35[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[35] > 0.0f) ? wts[35] * __uint_as_float(_vec_load_35[0] & 4294901760u) : 0.0f);
      wts[36] = smem_w[w_base + 36];
      acc[0] =
          acc[0] + ((wts[36] > 0.0f) ? wts[36] * __uint_as_float(_vec_load_36[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[36] > 0.0f) ? wts[36] * __uint_as_float(_vec_load_36[0] & 4294901760u) : 0.0f);
      wts[37] = smem_w[w_base + 37];
      acc[0] =
          acc[0] + ((wts[37] > 0.0f) ? wts[37] * __uint_as_float(_vec_load_37[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[37] > 0.0f) ? wts[37] * __uint_as_float(_vec_load_37[0] & 4294901760u) : 0.0f);
      wts[38] = smem_w[w_base + 38];
      acc[0] =
          acc[0] + ((wts[38] > 0.0f) ? wts[38] * __uint_as_float(_vec_load_38[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[38] > 0.0f) ? wts[38] * __uint_as_float(_vec_load_38[0] & 4294901760u) : 0.0f);
      wts[39] = smem_w[w_base + 39];
      acc[0] =
          acc[0] + ((wts[39] > 0.0f) ? wts[39] * __uint_as_float(_vec_load_39[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[39] > 0.0f) ? wts[39] * __uint_as_float(_vec_load_39[0] & 4294901760u) : 0.0f);
      wts[40] = smem_w[w_base + 40];
      acc[0] =
          acc[0] + ((wts[40] > 0.0f) ? wts[40] * __uint_as_float(_vec_load_40[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[40] > 0.0f) ? wts[40] * __uint_as_float(_vec_load_40[0] & 4294901760u) : 0.0f);
      wts[41] = smem_w[w_base + 41];
      acc[0] =
          acc[0] + ((wts[41] > 0.0f) ? wts[41] * __uint_as_float(_vec_load_41[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[41] > 0.0f) ? wts[41] * __uint_as_float(_vec_load_41[0] & 4294901760u) : 0.0f);
      wts[42] = smem_w[w_base + 42];
      acc[0] =
          acc[0] + ((wts[42] > 0.0f) ? wts[42] * __uint_as_float(_vec_load_42[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[42] > 0.0f) ? wts[42] * __uint_as_float(_vec_load_42[0] & 4294901760u) : 0.0f);
      wts[43] = smem_w[w_base + 43];
      acc[0] =
          acc[0] + ((wts[43] > 0.0f) ? wts[43] * __uint_as_float(_vec_load_43[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[43] > 0.0f) ? wts[43] * __uint_as_float(_vec_load_43[0] & 4294901760u) : 0.0f);
      wts[44] = smem_w[w_base + 44];
      acc[0] =
          acc[0] + ((wts[44] > 0.0f) ? wts[44] * __uint_as_float(_vec_load_44[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[44] > 0.0f) ? wts[44] * __uint_as_float(_vec_load_44[0] & 4294901760u) : 0.0f);
      wts[45] = smem_w[w_base + 45];
      acc[0] =
          acc[0] + ((wts[45] > 0.0f) ? wts[45] * __uint_as_float(_vec_load_45[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[45] > 0.0f) ? wts[45] * __uint_as_float(_vec_load_45[0] & 4294901760u) : 0.0f);
      wts[46] = smem_w[w_base + 46];
      acc[0] =
          acc[0] + ((wts[46] > 0.0f) ? wts[46] * __uint_as_float(_vec_load_46[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[46] > 0.0f) ? wts[46] * __uint_as_float(_vec_load_46[0] & 4294901760u) : 0.0f);
      wts[47] = smem_w[w_base + 47];
      acc[0] =
          acc[0] + ((wts[47] > 0.0f) ? wts[47] * __uint_as_float(_vec_load_47[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[47] > 0.0f) ? wts[47] * __uint_as_float(_vec_load_47[0] & 4294901760u) : 0.0f);
      wts[48] = smem_w[w_base + 48];
      acc[0] =
          acc[0] + ((wts[48] > 0.0f) ? wts[48] * __uint_as_float(_vec_load_48[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[48] > 0.0f) ? wts[48] * __uint_as_float(_vec_load_48[0] & 4294901760u) : 0.0f);
      wts[49] = smem_w[w_base + 49];
      acc[0] =
          acc[0] + ((wts[49] > 0.0f) ? wts[49] * __uint_as_float(_vec_load_49[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[49] > 0.0f) ? wts[49] * __uint_as_float(_vec_load_49[0] & 4294901760u) : 0.0f);
      wts[50] = smem_w[w_base + 50];
      acc[0] =
          acc[0] + ((wts[50] > 0.0f) ? wts[50] * __uint_as_float(_vec_load_50[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[50] > 0.0f) ? wts[50] * __uint_as_float(_vec_load_50[0] & 4294901760u) : 0.0f);
      wts[51] = smem_w[w_base + 51];
      acc[0] =
          acc[0] + ((wts[51] > 0.0f) ? wts[51] * __uint_as_float(_vec_load_51[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[51] > 0.0f) ? wts[51] * __uint_as_float(_vec_load_51[0] & 4294901760u) : 0.0f);
      wts[52] = smem_w[w_base + 52];
      acc[0] =
          acc[0] + ((wts[52] > 0.0f) ? wts[52] * __uint_as_float(_vec_load_52[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[52] > 0.0f) ? wts[52] * __uint_as_float(_vec_load_52[0] & 4294901760u) : 0.0f);
      wts[53] = smem_w[w_base + 53];
      acc[0] =
          acc[0] + ((wts[53] > 0.0f) ? wts[53] * __uint_as_float(_vec_load_53[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[53] > 0.0f) ? wts[53] * __uint_as_float(_vec_load_53[0] & 4294901760u) : 0.0f);
      wts[54] = smem_w[w_base + 54];
      acc[0] =
          acc[0] + ((wts[54] > 0.0f) ? wts[54] * __uint_as_float(_vec_load_54[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[54] > 0.0f) ? wts[54] * __uint_as_float(_vec_load_54[0] & 4294901760u) : 0.0f);
      wts[55] = smem_w[w_base + 55];
      acc[0] =
          acc[0] + ((wts[55] > 0.0f) ? wts[55] * __uint_as_float(_vec_load_55[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[55] > 0.0f) ? wts[55] * __uint_as_float(_vec_load_55[0] & 4294901760u) : 0.0f);
      wts[56] = smem_w[w_base + 56];
      acc[0] =
          acc[0] + ((wts[56] > 0.0f) ? wts[56] * __uint_as_float(_vec_load_56[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[56] > 0.0f) ? wts[56] * __uint_as_float(_vec_load_56[0] & 4294901760u) : 0.0f);
      wts[57] = smem_w[w_base + 57];
      acc[0] =
          acc[0] + ((wts[57] > 0.0f) ? wts[57] * __uint_as_float(_vec_load_57[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[57] > 0.0f) ? wts[57] * __uint_as_float(_vec_load_57[0] & 4294901760u) : 0.0f);
      wts[58] = smem_w[w_base + 58];
      acc[0] =
          acc[0] + ((wts[58] > 0.0f) ? wts[58] * __uint_as_float(_vec_load_58[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[58] > 0.0f) ? wts[58] * __uint_as_float(_vec_load_58[0] & 4294901760u) : 0.0f);
      wts[59] = smem_w[w_base + 59];
      acc[0] =
          acc[0] + ((wts[59] > 0.0f) ? wts[59] * __uint_as_float(_vec_load_59[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[59] > 0.0f) ? wts[59] * __uint_as_float(_vec_load_59[0] & 4294901760u) : 0.0f);
      wts[60] = smem_w[w_base + 60];
      acc[0] =
          acc[0] + ((wts[60] > 0.0f) ? wts[60] * __uint_as_float(_vec_load_60[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[60] > 0.0f) ? wts[60] * __uint_as_float(_vec_load_60[0] & 4294901760u) : 0.0f);
      wts[61] = smem_w[w_base + 61];
      acc[0] =
          acc[0] + ((wts[61] > 0.0f) ? wts[61] * __uint_as_float(_vec_load_61[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[61] > 0.0f) ? wts[61] * __uint_as_float(_vec_load_61[0] & 4294901760u) : 0.0f);
      wts[62] = smem_w[w_base + 62];
      acc[0] =
          acc[0] + ((wts[62] > 0.0f) ? wts[62] * __uint_as_float(_vec_load_62[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[62] > 0.0f) ? wts[62] * __uint_as_float(_vec_load_62[0] & 4294901760u) : 0.0f);
      wts[63] = smem_w[w_base + 63];
      acc[0] =
          acc[0] + ((wts[63] > 0.0f) ? wts[63] * __uint_as_float(_vec_load_63[0] << 16) : 0.0f);
      acc[1] = acc[1] +
               ((wts[63] > 0.0f) ? wts[63] * __uint_as_float(_vec_load_63[0] & 4294901760u) : 0.0f);
      for (int e_1 = 0; e_1 < 2; e_1++) {
        acc[e_1] = acc[e_1] * inv;
      }
      unsigned int _packed_0[1];
#pragma unroll
      for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 =
            __float22bfloat162_rn(make_float2(acc[_lp * 2 + 0], acc[_lp * 2 + 1 + 0]));
        _packed_0[_lp] = *(uint32_t*)&_bf2;
      }
      reinterpret_cast<unsigned*>(O + (row * 512 + d0))[0] =
          reinterpret_cast<unsigned*>(_packed_0)[0];
      if (has_lse != 0) {
        if (part == 0) {
          if (lane_0 == 0) {
            float lse_v = -CUDART_INF_F;
            if (sum_w > 0.0f) {
              float _log2_67;
              asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_67) : "f"(sum_w));
              lse_v = (m + _log2_67) * 0.6931472f;
            }
            *(reinterpret_cast<float*>(lse + row) + (0)) = lse_v;
          }
        }
      }
    } else {
      int _min_64 = ((0) < (last) ? (0) : (last));
      unsigned int _vec_load_64[2];
      {
        uint32_t _bf16x2_bits_68;
        _bf16x2_bits_68 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_64) * 512 + d0);
        _vec_load_64[0] = _bf16x2_bits_68;
      }
      int _min_65 = ((1) < (last) ? (1) : (last));
      unsigned int _vec_load_65[2];
      {
        uint32_t _bf16x2_bits_69;
        _bf16x2_bits_69 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_65) * 512 + d0);
        _vec_load_65[0] = _bf16x2_bits_69;
      }
      int _min_66 = ((2) < (last) ? (2) : (last));
      unsigned int _vec_load_66[2];
      {
        uint32_t _bf16x2_bits_70;
        _bf16x2_bits_70 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_66) * 512 + d0);
        _vec_load_66[0] = _bf16x2_bits_70;
      }
      int _min_67 = ((3) < (last) ? (3) : (last));
      unsigned int _vec_load_67[2];
      {
        uint32_t _bf16x2_bits_71;
        _bf16x2_bits_71 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_67) * 512 + d0);
        _vec_load_67[0] = _bf16x2_bits_71;
      }
      int _min_68 = ((4) < (last) ? (4) : (last));
      unsigned int _vec_load_68[2];
      {
        uint32_t _bf16x2_bits_72;
        _bf16x2_bits_72 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_68) * 512 + d0);
        _vec_load_68[0] = _bf16x2_bits_72;
      }
      int _min_69 = ((5) < (last) ? (5) : (last));
      unsigned int _vec_load_69[2];
      {
        uint32_t _bf16x2_bits_73;
        _bf16x2_bits_73 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_69) * 512 + d0);
        _vec_load_69[0] = _bf16x2_bits_73;
      }
      int _min_70 = ((6) < (last) ? (6) : (last));
      unsigned int _vec_load_70[2];
      {
        uint32_t _bf16x2_bits_74;
        _bf16x2_bits_74 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_70) * 512 + d0);
        _vec_load_70[0] = _bf16x2_bits_74;
      }
      int _min_71 = ((7) < (last) ? (7) : (last));
      unsigned int _vec_load_71[2];
      {
        uint32_t _bf16x2_bits_75;
        _bf16x2_bits_75 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_71) * 512 + d0);
        _vec_load_71[0] = _bf16x2_bits_75;
      }
      int _min_72 = ((8) < (last) ? (8) : (last));
      unsigned int _vec_load_72[2];
      {
        uint32_t _bf16x2_bits_76;
        _bf16x2_bits_76 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_72) * 512 + d0);
        _vec_load_72[0] = _bf16x2_bits_76;
      }
      int _min_73 = ((9) < (last) ? (9) : (last));
      unsigned int _vec_load_73[2];
      {
        uint32_t _bf16x2_bits_77;
        _bf16x2_bits_77 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_73) * 512 + d0);
        _vec_load_73[0] = _bf16x2_bits_77;
      }
      int _min_74 = ((10) < (last) ? (10) : (last));
      unsigned int _vec_load_74[2];
      {
        uint32_t _bf16x2_bits_78;
        _bf16x2_bits_78 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_74) * 512 + d0);
        _vec_load_74[0] = _bf16x2_bits_78;
      }
      int _min_75 = ((11) < (last) ? (11) : (last));
      unsigned int _vec_load_75[2];
      {
        uint32_t _bf16x2_bits_79;
        _bf16x2_bits_79 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_75) * 512 + d0);
        _vec_load_75[0] = _bf16x2_bits_79;
      }
      int _min_76 = ((12) < (last) ? (12) : (last));
      unsigned int _vec_load_76[2];
      {
        uint32_t _bf16x2_bits_80;
        _bf16x2_bits_80 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_76) * 512 + d0);
        _vec_load_76[0] = _bf16x2_bits_80;
      }
      int _min_77 = ((13) < (last) ? (13) : (last));
      unsigned int _vec_load_77[2];
      {
        uint32_t _bf16x2_bits_81;
        _bf16x2_bits_81 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_77) * 512 + d0);
        _vec_load_77[0] = _bf16x2_bits_81;
      }
      int _min_78 = ((14) < (last) ? (14) : (last));
      unsigned int _vec_load_78[2];
      {
        uint32_t _bf16x2_bits_82;
        _bf16x2_bits_82 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_78) * 512 + d0);
        _vec_load_78[0] = _bf16x2_bits_82;
      }
      int _min_79 = ((15) < (last) ? (15) : (last));
      unsigned int _vec_load_79[2];
      {
        uint32_t _bf16x2_bits_83;
        _bf16x2_bits_83 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_79) * 512 + d0);
        _vec_load_79[0] = _bf16x2_bits_83;
      }
      int _min_80 = ((16) < (last) ? (16) : (last));
      unsigned int _vec_load_80[2];
      {
        uint32_t _bf16x2_bits_84;
        _bf16x2_bits_84 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_80) * 512 + d0);
        _vec_load_80[0] = _bf16x2_bits_84;
      }
      int _min_81 = ((17) < (last) ? (17) : (last));
      unsigned int _vec_load_81[2];
      {
        uint32_t _bf16x2_bits_85;
        _bf16x2_bits_85 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_81) * 512 + d0);
        _vec_load_81[0] = _bf16x2_bits_85;
      }
      int _min_82 = ((18) < (last) ? (18) : (last));
      unsigned int _vec_load_82[2];
      {
        uint32_t _bf16x2_bits_86;
        _bf16x2_bits_86 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_82) * 512 + d0);
        _vec_load_82[0] = _bf16x2_bits_86;
      }
      int _min_83 = ((19) < (last) ? (19) : (last));
      unsigned int _vec_load_83[2];
      {
        uint32_t _bf16x2_bits_87;
        _bf16x2_bits_87 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_83) * 512 + d0);
        _vec_load_83[0] = _bf16x2_bits_87;
      }
      int _min_84 = ((20) < (last) ? (20) : (last));
      unsigned int _vec_load_84[2];
      {
        uint32_t _bf16x2_bits_88;
        _bf16x2_bits_88 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_84) * 512 + d0);
        _vec_load_84[0] = _bf16x2_bits_88;
      }
      int _min_85 = ((21) < (last) ? (21) : (last));
      unsigned int _vec_load_85[2];
      {
        uint32_t _bf16x2_bits_89;
        _bf16x2_bits_89 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_85) * 512 + d0);
        _vec_load_85[0] = _bf16x2_bits_89;
      }
      int _min_86 = ((22) < (last) ? (22) : (last));
      unsigned int _vec_load_86[2];
      {
        uint32_t _bf16x2_bits_90;
        _bf16x2_bits_90 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_86) * 512 + d0);
        _vec_load_86[0] = _bf16x2_bits_90;
      }
      int _min_87 = ((23) < (last) ? (23) : (last));
      unsigned int _vec_load_87[2];
      {
        uint32_t _bf16x2_bits_91;
        _bf16x2_bits_91 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_87) * 512 + d0);
        _vec_load_87[0] = _bf16x2_bits_91;
      }
      int _min_88 = ((24) < (last) ? (24) : (last));
      unsigned int _vec_load_88[2];
      {
        uint32_t _bf16x2_bits_92;
        _bf16x2_bits_92 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_88) * 512 + d0);
        _vec_load_88[0] = _bf16x2_bits_92;
      }
      int _min_89 = ((25) < (last) ? (25) : (last));
      unsigned int _vec_load_89[2];
      {
        uint32_t _bf16x2_bits_93;
        _bf16x2_bits_93 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_89) * 512 + d0);
        _vec_load_89[0] = _bf16x2_bits_93;
      }
      int _min_90 = ((26) < (last) ? (26) : (last));
      unsigned int _vec_load_90[2];
      {
        uint32_t _bf16x2_bits_94;
        _bf16x2_bits_94 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_90) * 512 + d0);
        _vec_load_90[0] = _bf16x2_bits_94;
      }
      int _min_91 = ((27) < (last) ? (27) : (last));
      unsigned int _vec_load_91[2];
      {
        uint32_t _bf16x2_bits_95;
        _bf16x2_bits_95 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_91) * 512 + d0);
        _vec_load_91[0] = _bf16x2_bits_95;
      }
      int _min_92 = ((28) < (last) ? (28) : (last));
      unsigned int _vec_load_92[2];
      {
        uint32_t _bf16x2_bits_96;
        _bf16x2_bits_96 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_92) * 512 + d0);
        _vec_load_92[0] = _bf16x2_bits_96;
      }
      int _min_93 = ((29) < (last) ? (29) : (last));
      unsigned int _vec_load_93[2];
      {
        uint32_t _bf16x2_bits_97;
        _bf16x2_bits_97 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_93) * 512 + d0);
        _vec_load_93[0] = _bf16x2_bits_97;
      }
      int _min_94 = ((30) < (last) ? (30) : (last));
      unsigned int _vec_load_94[2];
      {
        uint32_t _bf16x2_bits_98;
        _bf16x2_bits_98 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_94) * 512 + d0);
        _vec_load_94[0] = _bf16x2_bits_98;
      }
      int _min_95 = ((31) < (last) ? (31) : (last));
      unsigned int _vec_load_95[2];
      {
        uint32_t _bf16x2_bits_99;
        _bf16x2_bits_99 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base + _min_95) * 512 + d0);
        _vec_load_95[0] = _bf16x2_bits_99;
      }
      float l0_1 = -CUDART_INF_F;
      float l1_1 = -CUDART_INF_F;
      if (lane_0 < num_split) {
        l0_1 = partial_lse[stat_base + lane_0];
      }
      if (lane_0 + 32 < num_split) {
        l1_1 = partial_lse[stat_base + lane_0 + 32];
      }
      float m_1 = max_noftz(l0_1, l1_1);
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        m_1 = max_noftz(m_1, __shfl_xor_sync(0xFFFFFFFF, m_1, offset));
      float w0_1 = 0.0f;
      float w1_1 = 0.0f;
      if (l0_1 > -CUDART_INF_F) {
        float _exp2_100 = approx_exp2(l0_1 - m_1);
        w0_1 = _exp2_100;
      }
      if (l1_1 > -CUDART_INF_F) {
        float _exp2_101 = approx_exp2(l1_1 - m_1);
        w1_1 = _exp2_101;
      }
      smem_w[w_base + lane_0] = w0_1;
      smem_w[w_base + 32 + lane_0] = w1_1;
      float sum_w_1 = w0_1 + w1_1;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        sum_w_1 += __shfl_xor_sync(0xFFFFFFFF, sum_w_1, offset);
      __syncwarp();
      float inv_1 = 0.0f;
      if (sum_w_1 > 0.0f) {
        float _rcp_102 = approx_rcp(sum_w_1);
        inv_1 = _rcp_102;
      }
      float acc_1[2];
      for (int e_2 = 0; e_2 < 2; e_2++) {
        acc_1[e_2] = 0.0f;
      }
      float wts_1[32];
      wts_1[0] = smem_w[w_base];
      acc_1[0] =
          acc_1[0] + ((wts_1[0] > 0.0f) ? wts_1[0] * __uint_as_float(_vec_load_64[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[0] > 0.0f) ? wts_1[0] * __uint_as_float(_vec_load_64[0] & 4294901760u) : 0.0f);
      wts_1[1] = smem_w[w_base + 1];
      acc_1[0] =
          acc_1[0] + ((wts_1[1] > 0.0f) ? wts_1[1] * __uint_as_float(_vec_load_65[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[1] > 0.0f) ? wts_1[1] * __uint_as_float(_vec_load_65[0] & 4294901760u) : 0.0f);
      wts_1[2] = smem_w[w_base + 2];
      acc_1[0] =
          acc_1[0] + ((wts_1[2] > 0.0f) ? wts_1[2] * __uint_as_float(_vec_load_66[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[2] > 0.0f) ? wts_1[2] * __uint_as_float(_vec_load_66[0] & 4294901760u) : 0.0f);
      wts_1[3] = smem_w[w_base + 3];
      acc_1[0] =
          acc_1[0] + ((wts_1[3] > 0.0f) ? wts_1[3] * __uint_as_float(_vec_load_67[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[3] > 0.0f) ? wts_1[3] * __uint_as_float(_vec_load_67[0] & 4294901760u) : 0.0f);
      wts_1[4] = smem_w[w_base + 4];
      acc_1[0] =
          acc_1[0] + ((wts_1[4] > 0.0f) ? wts_1[4] * __uint_as_float(_vec_load_68[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[4] > 0.0f) ? wts_1[4] * __uint_as_float(_vec_load_68[0] & 4294901760u) : 0.0f);
      wts_1[5] = smem_w[w_base + 5];
      acc_1[0] =
          acc_1[0] + ((wts_1[5] > 0.0f) ? wts_1[5] * __uint_as_float(_vec_load_69[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[5] > 0.0f) ? wts_1[5] * __uint_as_float(_vec_load_69[0] & 4294901760u) : 0.0f);
      wts_1[6] = smem_w[w_base + 6];
      acc_1[0] =
          acc_1[0] + ((wts_1[6] > 0.0f) ? wts_1[6] * __uint_as_float(_vec_load_70[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[6] > 0.0f) ? wts_1[6] * __uint_as_float(_vec_load_70[0] & 4294901760u) : 0.0f);
      wts_1[7] = smem_w[w_base + 7];
      acc_1[0] =
          acc_1[0] + ((wts_1[7] > 0.0f) ? wts_1[7] * __uint_as_float(_vec_load_71[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[7] > 0.0f) ? wts_1[7] * __uint_as_float(_vec_load_71[0] & 4294901760u) : 0.0f);
      wts_1[8] = smem_w[w_base + 8];
      acc_1[0] =
          acc_1[0] + ((wts_1[8] > 0.0f) ? wts_1[8] * __uint_as_float(_vec_load_72[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[8] > 0.0f) ? wts_1[8] * __uint_as_float(_vec_load_72[0] & 4294901760u) : 0.0f);
      wts_1[9] = smem_w[w_base + 9];
      acc_1[0] =
          acc_1[0] + ((wts_1[9] > 0.0f) ? wts_1[9] * __uint_as_float(_vec_load_73[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[9] > 0.0f) ? wts_1[9] * __uint_as_float(_vec_load_73[0] & 4294901760u) : 0.0f);
      wts_1[10] = smem_w[w_base + 10];
      acc_1[0] = acc_1[0] +
                 ((wts_1[10] > 0.0f) ? wts_1[10] * __uint_as_float(_vec_load_74[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[10] > 0.0f) ? wts_1[10] * __uint_as_float(_vec_load_74[0] & 4294901760u) : 0.0f);
      wts_1[11] = smem_w[w_base + 11];
      acc_1[0] = acc_1[0] +
                 ((wts_1[11] > 0.0f) ? wts_1[11] * __uint_as_float(_vec_load_75[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[11] > 0.0f) ? wts_1[11] * __uint_as_float(_vec_load_75[0] & 4294901760u) : 0.0f);
      wts_1[12] = smem_w[w_base + 12];
      acc_1[0] = acc_1[0] +
                 ((wts_1[12] > 0.0f) ? wts_1[12] * __uint_as_float(_vec_load_76[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[12] > 0.0f) ? wts_1[12] * __uint_as_float(_vec_load_76[0] & 4294901760u) : 0.0f);
      wts_1[13] = smem_w[w_base + 13];
      acc_1[0] = acc_1[0] +
                 ((wts_1[13] > 0.0f) ? wts_1[13] * __uint_as_float(_vec_load_77[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[13] > 0.0f) ? wts_1[13] * __uint_as_float(_vec_load_77[0] & 4294901760u) : 0.0f);
      wts_1[14] = smem_w[w_base + 14];
      acc_1[0] = acc_1[0] +
                 ((wts_1[14] > 0.0f) ? wts_1[14] * __uint_as_float(_vec_load_78[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[14] > 0.0f) ? wts_1[14] * __uint_as_float(_vec_load_78[0] & 4294901760u) : 0.0f);
      wts_1[15] = smem_w[w_base + 15];
      acc_1[0] = acc_1[0] +
                 ((wts_1[15] > 0.0f) ? wts_1[15] * __uint_as_float(_vec_load_79[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[15] > 0.0f) ? wts_1[15] * __uint_as_float(_vec_load_79[0] & 4294901760u) : 0.0f);
      wts_1[16] = smem_w[w_base + 16];
      acc_1[0] = acc_1[0] +
                 ((wts_1[16] > 0.0f) ? wts_1[16] * __uint_as_float(_vec_load_80[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[16] > 0.0f) ? wts_1[16] * __uint_as_float(_vec_load_80[0] & 4294901760u) : 0.0f);
      wts_1[17] = smem_w[w_base + 17];
      acc_1[0] = acc_1[0] +
                 ((wts_1[17] > 0.0f) ? wts_1[17] * __uint_as_float(_vec_load_81[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[17] > 0.0f) ? wts_1[17] * __uint_as_float(_vec_load_81[0] & 4294901760u) : 0.0f);
      wts_1[18] = smem_w[w_base + 18];
      acc_1[0] = acc_1[0] +
                 ((wts_1[18] > 0.0f) ? wts_1[18] * __uint_as_float(_vec_load_82[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[18] > 0.0f) ? wts_1[18] * __uint_as_float(_vec_load_82[0] & 4294901760u) : 0.0f);
      wts_1[19] = smem_w[w_base + 19];
      acc_1[0] = acc_1[0] +
                 ((wts_1[19] > 0.0f) ? wts_1[19] * __uint_as_float(_vec_load_83[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[19] > 0.0f) ? wts_1[19] * __uint_as_float(_vec_load_83[0] & 4294901760u) : 0.0f);
      wts_1[20] = smem_w[w_base + 20];
      acc_1[0] = acc_1[0] +
                 ((wts_1[20] > 0.0f) ? wts_1[20] * __uint_as_float(_vec_load_84[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[20] > 0.0f) ? wts_1[20] * __uint_as_float(_vec_load_84[0] & 4294901760u) : 0.0f);
      wts_1[21] = smem_w[w_base + 21];
      acc_1[0] = acc_1[0] +
                 ((wts_1[21] > 0.0f) ? wts_1[21] * __uint_as_float(_vec_load_85[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[21] > 0.0f) ? wts_1[21] * __uint_as_float(_vec_load_85[0] & 4294901760u) : 0.0f);
      wts_1[22] = smem_w[w_base + 22];
      acc_1[0] = acc_1[0] +
                 ((wts_1[22] > 0.0f) ? wts_1[22] * __uint_as_float(_vec_load_86[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[22] > 0.0f) ? wts_1[22] * __uint_as_float(_vec_load_86[0] & 4294901760u) : 0.0f);
      wts_1[23] = smem_w[w_base + 23];
      acc_1[0] = acc_1[0] +
                 ((wts_1[23] > 0.0f) ? wts_1[23] * __uint_as_float(_vec_load_87[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[23] > 0.0f) ? wts_1[23] * __uint_as_float(_vec_load_87[0] & 4294901760u) : 0.0f);
      wts_1[24] = smem_w[w_base + 24];
      acc_1[0] = acc_1[0] +
                 ((wts_1[24] > 0.0f) ? wts_1[24] * __uint_as_float(_vec_load_88[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[24] > 0.0f) ? wts_1[24] * __uint_as_float(_vec_load_88[0] & 4294901760u) : 0.0f);
      wts_1[25] = smem_w[w_base + 25];
      acc_1[0] = acc_1[0] +
                 ((wts_1[25] > 0.0f) ? wts_1[25] * __uint_as_float(_vec_load_89[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[25] > 0.0f) ? wts_1[25] * __uint_as_float(_vec_load_89[0] & 4294901760u) : 0.0f);
      wts_1[26] = smem_w[w_base + 26];
      acc_1[0] = acc_1[0] +
                 ((wts_1[26] > 0.0f) ? wts_1[26] * __uint_as_float(_vec_load_90[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[26] > 0.0f) ? wts_1[26] * __uint_as_float(_vec_load_90[0] & 4294901760u) : 0.0f);
      wts_1[27] = smem_w[w_base + 27];
      acc_1[0] = acc_1[0] +
                 ((wts_1[27] > 0.0f) ? wts_1[27] * __uint_as_float(_vec_load_91[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[27] > 0.0f) ? wts_1[27] * __uint_as_float(_vec_load_91[0] & 4294901760u) : 0.0f);
      wts_1[28] = smem_w[w_base + 28];
      acc_1[0] = acc_1[0] +
                 ((wts_1[28] > 0.0f) ? wts_1[28] * __uint_as_float(_vec_load_92[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[28] > 0.0f) ? wts_1[28] * __uint_as_float(_vec_load_92[0] & 4294901760u) : 0.0f);
      wts_1[29] = smem_w[w_base + 29];
      acc_1[0] = acc_1[0] +
                 ((wts_1[29] > 0.0f) ? wts_1[29] * __uint_as_float(_vec_load_93[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[29] > 0.0f) ? wts_1[29] * __uint_as_float(_vec_load_93[0] & 4294901760u) : 0.0f);
      wts_1[30] = smem_w[w_base + 30];
      acc_1[0] = acc_1[0] +
                 ((wts_1[30] > 0.0f) ? wts_1[30] * __uint_as_float(_vec_load_94[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[30] > 0.0f) ? wts_1[30] * __uint_as_float(_vec_load_94[0] & 4294901760u) : 0.0f);
      wts_1[31] = smem_w[w_base + 31];
      acc_1[0] = acc_1[0] +
                 ((wts_1[31] > 0.0f) ? wts_1[31] * __uint_as_float(_vec_load_95[0] << 16) : 0.0f);
      acc_1[1] =
          acc_1[1] +
          ((wts_1[31] > 0.0f) ? wts_1[31] * __uint_as_float(_vec_load_95[0] & 4294901760u) : 0.0f);
      for (int e_3 = 0; e_3 < 2; e_3++) {
        acc_1[e_3] = acc_1[e_3] * inv_1;
      }
      unsigned int _packed_1[1];
#pragma unroll
      for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 =
            __float22bfloat162_rn(make_float2(acc_1[_lp * 2 + 0], acc_1[_lp * 2 + 1 + 0]));
        _packed_1[_lp] = *(uint32_t*)&_bf2;
      }
      reinterpret_cast<unsigned*>(O + (row * 512 + d0))[0] =
          reinterpret_cast<unsigned*>(_packed_1)[0];
      if (has_lse != 0) {
        if (part == 0) {
          if (lane_0 == 0) {
            float lse_v_1 = -CUDART_INF_F;
            if (sum_w_1 > 0.0f) {
              float _log2_103;
              asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_103) : "f"(sum_w_1));
              lse_v_1 = (m_1 + _log2_103) * 0.6931472f;
            }
            *(reinterpret_cast<float*>(lse + row) + (0)) = lse_v_1;
          }
        }
      }
    }
  }
}

}  // extern "C"
