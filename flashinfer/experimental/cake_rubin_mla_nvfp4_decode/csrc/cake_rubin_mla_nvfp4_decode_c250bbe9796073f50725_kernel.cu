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
__launch_bounds__(256, LAUNCH_MIN_BLOCKS) void kernel_cake_rubin_mla_nvfp4_decode_c250bbe9796073f50725(
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
  int part_7 = warp % 8;
  int row_7 = blockIdx.x * 1 + warp / 8;
  int lane_7 = lane;
  int stat_base_7 = row_7 * num_split;
  int w_base_7 = warp * 64;
  int d0_7 = part_7 * 64 + lane_7 * 2;
  int last_7 = num_split - 1;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (row_7 < rows) {
    if (num_split > 32) {
      float l0 = -CUDART_INF_F;
      float l1 = -CUDART_INF_F;
      if (lane_7 < num_split) {
        l0 = partial_lse[stat_base_7 + lane_7];
      }
      if (lane_7 + 32 < num_split) {
        l1 = partial_lse[stat_base_7 + lane_7 + 32];
      }
      unsigned int vec[48];
      int _mn_0 = ((0) < (last_7) ? (0) : (last_7));
      {
        uint32_t _bf16x2_bits_0;
        _bf16x2_bits_0 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_0) * 512 + d0_7);
        vec[0] = _bf16x2_bits_0;
      }
      int _mn_1 = ((1) < (last_7) ? (1) : (last_7));
      {
        uint32_t _bf16x2_bits_1;
        _bf16x2_bits_1 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_1) * 512 + d0_7);
        vec[1] = _bf16x2_bits_1;
      }
      int _mn_2 = ((2) < (last_7) ? (2) : (last_7));
      {
        uint32_t _bf16x2_bits_2;
        _bf16x2_bits_2 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_2) * 512 + d0_7);
        vec[2] = _bf16x2_bits_2;
      }
      int _mn_3 = ((3) < (last_7) ? (3) : (last_7));
      {
        uint32_t _bf16x2_bits_3;
        _bf16x2_bits_3 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_3) * 512 + d0_7);
        vec[3] = _bf16x2_bits_3;
      }
      int _mn_4 = ((4) < (last_7) ? (4) : (last_7));
      {
        uint32_t _bf16x2_bits_4;
        _bf16x2_bits_4 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_4) * 512 + d0_7);
        vec[4] = _bf16x2_bits_4;
      }
      int _mn_5 = ((5) < (last_7) ? (5) : (last_7));
      {
        uint32_t _bf16x2_bits_5;
        _bf16x2_bits_5 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_5) * 512 + d0_7);
        vec[5] = _bf16x2_bits_5;
      }
      int _mn_6 = ((6) < (last_7) ? (6) : (last_7));
      {
        uint32_t _bf16x2_bits_6;
        _bf16x2_bits_6 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_6) * 512 + d0_7);
        vec[6] = _bf16x2_bits_6;
      }
      int _mn_7 = ((7) < (last_7) ? (7) : (last_7));
      {
        uint32_t _bf16x2_bits_7;
        _bf16x2_bits_7 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_7) * 512 + d0_7);
        vec[7] = _bf16x2_bits_7;
      }
      int _mn_8 = ((8) < (last_7) ? (8) : (last_7));
      {
        uint32_t _bf16x2_bits_8;
        _bf16x2_bits_8 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_8) * 512 + d0_7);
        vec[8] = _bf16x2_bits_8;
      }
      int _mn_9 = ((9) < (last_7) ? (9) : (last_7));
      {
        uint32_t _bf16x2_bits_9;
        _bf16x2_bits_9 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_9) * 512 + d0_7);
        vec[9] = _bf16x2_bits_9;
      }
      int _mn_10 = ((10) < (last_7) ? (10) : (last_7));
      {
        uint32_t _bf16x2_bits_10;
        _bf16x2_bits_10 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_10) * 512 + d0_7);
        vec[10] = _bf16x2_bits_10;
      }
      int _mn_11 = ((11) < (last_7) ? (11) : (last_7));
      {
        uint32_t _bf16x2_bits_11;
        _bf16x2_bits_11 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_11) * 512 + d0_7);
        vec[11] = _bf16x2_bits_11;
      }
      int _mn_12 = ((12) < (last_7) ? (12) : (last_7));
      {
        uint32_t _bf16x2_bits_12;
        _bf16x2_bits_12 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_12) * 512 + d0_7);
        vec[12] = _bf16x2_bits_12;
      }
      int _mn_13 = ((13) < (last_7) ? (13) : (last_7));
      {
        uint32_t _bf16x2_bits_13;
        _bf16x2_bits_13 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_13) * 512 + d0_7);
        vec[13] = _bf16x2_bits_13;
      }
      int _mn_14 = ((14) < (last_7) ? (14) : (last_7));
      {
        uint32_t _bf16x2_bits_14;
        _bf16x2_bits_14 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_14) * 512 + d0_7);
        vec[14] = _bf16x2_bits_14;
      }
      int _mn_15 = ((15) < (last_7) ? (15) : (last_7));
      {
        uint32_t _bf16x2_bits_15;
        _bf16x2_bits_15 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_15) * 512 + d0_7);
        vec[15] = _bf16x2_bits_15;
      }
      int _mn_16 = ((16) < (last_7) ? (16) : (last_7));
      {
        uint32_t _bf16x2_bits_16;
        _bf16x2_bits_16 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_16) * 512 + d0_7);
        vec[16] = _bf16x2_bits_16;
      }
      int _mn_17 = ((17) < (last_7) ? (17) : (last_7));
      {
        uint32_t _bf16x2_bits_17;
        _bf16x2_bits_17 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_17) * 512 + d0_7);
        vec[17] = _bf16x2_bits_17;
      }
      int _mn_18 = ((18) < (last_7) ? (18) : (last_7));
      {
        uint32_t _bf16x2_bits_18;
        _bf16x2_bits_18 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_18) * 512 + d0_7);
        vec[18] = _bf16x2_bits_18;
      }
      int _mn_19 = ((19) < (last_7) ? (19) : (last_7));
      {
        uint32_t _bf16x2_bits_19;
        _bf16x2_bits_19 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_19) * 512 + d0_7);
        vec[19] = _bf16x2_bits_19;
      }
      int _mn_20 = ((20) < (last_7) ? (20) : (last_7));
      {
        uint32_t _bf16x2_bits_20;
        _bf16x2_bits_20 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_20) * 512 + d0_7);
        vec[20] = _bf16x2_bits_20;
      }
      int _mn_21 = ((21) < (last_7) ? (21) : (last_7));
      {
        uint32_t _bf16x2_bits_21;
        _bf16x2_bits_21 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_21) * 512 + d0_7);
        vec[21] = _bf16x2_bits_21;
      }
      int _mn_22 = ((22) < (last_7) ? (22) : (last_7));
      {
        uint32_t _bf16x2_bits_22;
        _bf16x2_bits_22 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_22) * 512 + d0_7);
        vec[22] = _bf16x2_bits_22;
      }
      int _mn_23 = ((23) < (last_7) ? (23) : (last_7));
      {
        uint32_t _bf16x2_bits_23;
        _bf16x2_bits_23 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_23) * 512 + d0_7);
        vec[23] = _bf16x2_bits_23;
      }
      int _mn_24 = ((24) < (last_7) ? (24) : (last_7));
      {
        uint32_t _bf16x2_bits_24;
        _bf16x2_bits_24 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_24) * 512 + d0_7);
        vec[24] = _bf16x2_bits_24;
      }
      int _mn_25 = ((25) < (last_7) ? (25) : (last_7));
      {
        uint32_t _bf16x2_bits_25;
        _bf16x2_bits_25 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_25) * 512 + d0_7);
        vec[25] = _bf16x2_bits_25;
      }
      int _mn_26 = ((26) < (last_7) ? (26) : (last_7));
      {
        uint32_t _bf16x2_bits_26;
        _bf16x2_bits_26 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_26) * 512 + d0_7);
        vec[26] = _bf16x2_bits_26;
      }
      int _mn_27 = ((27) < (last_7) ? (27) : (last_7));
      {
        uint32_t _bf16x2_bits_27;
        _bf16x2_bits_27 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_27) * 512 + d0_7);
        vec[27] = _bf16x2_bits_27;
      }
      int _mn_28 = ((28) < (last_7) ? (28) : (last_7));
      {
        uint32_t _bf16x2_bits_28;
        _bf16x2_bits_28 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_28) * 512 + d0_7);
        vec[28] = _bf16x2_bits_28;
      }
      int _mn_29 = ((29) < (last_7) ? (29) : (last_7));
      {
        uint32_t _bf16x2_bits_29;
        _bf16x2_bits_29 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_29) * 512 + d0_7);
        vec[29] = _bf16x2_bits_29;
      }
      int _mn_30 = ((30) < (last_7) ? (30) : (last_7));
      {
        uint32_t _bf16x2_bits_30;
        _bf16x2_bits_30 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_30) * 512 + d0_7);
        vec[30] = _bf16x2_bits_30;
      }
      int _mn_31 = ((31) < (last_7) ? (31) : (last_7));
      {
        uint32_t _bf16x2_bits_31;
        _bf16x2_bits_31 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_31) * 512 + d0_7);
        vec[31] = _bf16x2_bits_31;
      }
      int _mn_32 = ((32) < (last_7) ? (32) : (last_7));
      {
        uint32_t _bf16x2_bits_32;
        _bf16x2_bits_32 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_32) * 512 + d0_7);
        vec[32] = _bf16x2_bits_32;
      }
      int _mn_33 = ((33) < (last_7) ? (33) : (last_7));
      {
        uint32_t _bf16x2_bits_33;
        _bf16x2_bits_33 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_33) * 512 + d0_7);
        vec[33] = _bf16x2_bits_33;
      }
      int _mn_34 = ((34) < (last_7) ? (34) : (last_7));
      {
        uint32_t _bf16x2_bits_34;
        _bf16x2_bits_34 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_34) * 512 + d0_7);
        vec[34] = _bf16x2_bits_34;
      }
      int _mn_35 = ((35) < (last_7) ? (35) : (last_7));
      {
        uint32_t _bf16x2_bits_35;
        _bf16x2_bits_35 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_35) * 512 + d0_7);
        vec[35] = _bf16x2_bits_35;
      }
      int _mn_36 = ((36) < (last_7) ? (36) : (last_7));
      {
        uint32_t _bf16x2_bits_36;
        _bf16x2_bits_36 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_36) * 512 + d0_7);
        vec[36] = _bf16x2_bits_36;
      }
      int _mn_37 = ((37) < (last_7) ? (37) : (last_7));
      {
        uint32_t _bf16x2_bits_37;
        _bf16x2_bits_37 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_37) * 512 + d0_7);
        vec[37] = _bf16x2_bits_37;
      }
      int _mn_38 = ((38) < (last_7) ? (38) : (last_7));
      {
        uint32_t _bf16x2_bits_38;
        _bf16x2_bits_38 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_38) * 512 + d0_7);
        vec[38] = _bf16x2_bits_38;
      }
      int _mn_39 = ((39) < (last_7) ? (39) : (last_7));
      {
        uint32_t _bf16x2_bits_39;
        _bf16x2_bits_39 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_39) * 512 + d0_7);
        vec[39] = _bf16x2_bits_39;
      }
      int _mn_40 = ((40) < (last_7) ? (40) : (last_7));
      {
        uint32_t _bf16x2_bits_40;
        _bf16x2_bits_40 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_40) * 512 + d0_7);
        vec[40] = _bf16x2_bits_40;
      }
      int _mn_41 = ((41) < (last_7) ? (41) : (last_7));
      {
        uint32_t _bf16x2_bits_41;
        _bf16x2_bits_41 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_41) * 512 + d0_7);
        vec[41] = _bf16x2_bits_41;
      }
      int _mn_42 = ((42) < (last_7) ? (42) : (last_7));
      {
        uint32_t _bf16x2_bits_42;
        _bf16x2_bits_42 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_42) * 512 + d0_7);
        vec[42] = _bf16x2_bits_42;
      }
      int _mn_43 = ((43) < (last_7) ? (43) : (last_7));
      {
        uint32_t _bf16x2_bits_43;
        _bf16x2_bits_43 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_43) * 512 + d0_7);
        vec[43] = _bf16x2_bits_43;
      }
      int _mn_44 = ((44) < (last_7) ? (44) : (last_7));
      {
        uint32_t _bf16x2_bits_44;
        _bf16x2_bits_44 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_44) * 512 + d0_7);
        vec[44] = _bf16x2_bits_44;
      }
      int _mn_45 = ((45) < (last_7) ? (45) : (last_7));
      {
        uint32_t _bf16x2_bits_45;
        _bf16x2_bits_45 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_45) * 512 + d0_7);
        vec[45] = _bf16x2_bits_45;
      }
      int _mn_46 = ((46) < (last_7) ? (46) : (last_7));
      {
        uint32_t _bf16x2_bits_46;
        _bf16x2_bits_46 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_46) * 512 + d0_7);
        vec[46] = _bf16x2_bits_46;
      }
      int _mn_47 = ((47) < (last_7) ? (47) : (last_7));
      {
        uint32_t _bf16x2_bits_47;
        _bf16x2_bits_47 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_47) * 512 + d0_7);
        vec[47] = _bf16x2_bits_47;
      }
      int merged = 0;
      float _shfl_sync_48 = __shfl_sync(0xFFFFFFFF, l0, 0);
      merged = _shfl_sync_48 == CUDART_INF_F;
      float m = max_noftz(l0, l1);
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        m = max_noftz(m, __shfl_xor_sync(0xFFFFFFFF, m, offset));
      float w0 = 0.0f;
      float w1 = 0.0f;
      if (l0 > -CUDART_INF_F) {
        float _exp2_49 = approx_exp2(l0 - m);
        w0 = _exp2_49;
      }
      if (l1 > -CUDART_INF_F) {
        float _exp2_50 = approx_exp2(l1 - m);
        w1 = _exp2_50;
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
        float _rcp_51 = approx_rcp(sum_w);
        inv = _rcp_51;
      }
      float acc[2];
      for (int e = 0; e < 2; e++) {
        acc[e] = 0.0f;
      }
      float wts[48];
      wts[0] = smem_w[w_base_7 + 0];
      acc[0] = acc[0] + ((wts[0] > 0.0f) ? wts[0] * __uint_as_float(vec[0] << 16) : 0.0f);
      acc[1] = acc[1] + ((wts[0] > 0.0f) ? wts[0] * __uint_as_float(vec[0] & 4294901760u) : 0.0f);
      wts[1] = smem_w[w_base_7 + 1];
      acc[0] = acc[0] + ((wts[1] > 0.0f) ? wts[1] * __uint_as_float(vec[1] << 16) : 0.0f);
      acc[1] = acc[1] + ((wts[1] > 0.0f) ? wts[1] * __uint_as_float(vec[1] & 4294901760u) : 0.0f);
      wts[2] = smem_w[w_base_7 + 2];
      acc[0] = acc[0] + ((wts[2] > 0.0f) ? wts[2] * __uint_as_float(vec[2] << 16) : 0.0f);
      acc[1] = acc[1] + ((wts[2] > 0.0f) ? wts[2] * __uint_as_float(vec[2] & 4294901760u) : 0.0f);
      wts[3] = smem_w[w_base_7 + 3];
      acc[0] = acc[0] + ((wts[3] > 0.0f) ? wts[3] * __uint_as_float(vec[3] << 16) : 0.0f);
      acc[1] = acc[1] + ((wts[3] > 0.0f) ? wts[3] * __uint_as_float(vec[3] & 4294901760u) : 0.0f);
      wts[4] = smem_w[w_base_7 + 4];
      acc[0] = acc[0] + ((wts[4] > 0.0f) ? wts[4] * __uint_as_float(vec[4] << 16) : 0.0f);
      acc[1] = acc[1] + ((wts[4] > 0.0f) ? wts[4] * __uint_as_float(vec[4] & 4294901760u) : 0.0f);
      wts[5] = smem_w[w_base_7 + 5];
      acc[0] = acc[0] + ((wts[5] > 0.0f) ? wts[5] * __uint_as_float(vec[5] << 16) : 0.0f);
      acc[1] = acc[1] + ((wts[5] > 0.0f) ? wts[5] * __uint_as_float(vec[5] & 4294901760u) : 0.0f);
      wts[6] = smem_w[w_base_7 + 6];
      acc[0] = acc[0] + ((wts[6] > 0.0f) ? wts[6] * __uint_as_float(vec[6] << 16) : 0.0f);
      acc[1] = acc[1] + ((wts[6] > 0.0f) ? wts[6] * __uint_as_float(vec[6] & 4294901760u) : 0.0f);
      wts[7] = smem_w[w_base_7 + 7];
      acc[0] = acc[0] + ((wts[7] > 0.0f) ? wts[7] * __uint_as_float(vec[7] << 16) : 0.0f);
      acc[1] = acc[1] + ((wts[7] > 0.0f) ? wts[7] * __uint_as_float(vec[7] & 4294901760u) : 0.0f);
      wts[8] = smem_w[w_base_7 + 8];
      acc[0] = acc[0] + ((wts[8] > 0.0f) ? wts[8] * __uint_as_float(vec[8] << 16) : 0.0f);
      acc[1] = acc[1] + ((wts[8] > 0.0f) ? wts[8] * __uint_as_float(vec[8] & 4294901760u) : 0.0f);
      wts[9] = smem_w[w_base_7 + 9];
      acc[0] = acc[0] + ((wts[9] > 0.0f) ? wts[9] * __uint_as_float(vec[9] << 16) : 0.0f);
      acc[1] = acc[1] + ((wts[9] > 0.0f) ? wts[9] * __uint_as_float(vec[9] & 4294901760u) : 0.0f);
      wts[10] = smem_w[w_base_7 + 10];
      acc[0] = acc[0] + ((wts[10] > 0.0f) ? wts[10] * __uint_as_float(vec[10] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[10] > 0.0f) ? wts[10] * __uint_as_float(vec[10] & 4294901760u) : 0.0f);
      wts[11] = smem_w[w_base_7 + 11];
      acc[0] = acc[0] + ((wts[11] > 0.0f) ? wts[11] * __uint_as_float(vec[11] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[11] > 0.0f) ? wts[11] * __uint_as_float(vec[11] & 4294901760u) : 0.0f);
      wts[12] = smem_w[w_base_7 + 12];
      acc[0] = acc[0] + ((wts[12] > 0.0f) ? wts[12] * __uint_as_float(vec[12] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[12] > 0.0f) ? wts[12] * __uint_as_float(vec[12] & 4294901760u) : 0.0f);
      wts[13] = smem_w[w_base_7 + 13];
      acc[0] = acc[0] + ((wts[13] > 0.0f) ? wts[13] * __uint_as_float(vec[13] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[13] > 0.0f) ? wts[13] * __uint_as_float(vec[13] & 4294901760u) : 0.0f);
      wts[14] = smem_w[w_base_7 + 14];
      acc[0] = acc[0] + ((wts[14] > 0.0f) ? wts[14] * __uint_as_float(vec[14] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[14] > 0.0f) ? wts[14] * __uint_as_float(vec[14] & 4294901760u) : 0.0f);
      wts[15] = smem_w[w_base_7 + 15];
      acc[0] = acc[0] + ((wts[15] > 0.0f) ? wts[15] * __uint_as_float(vec[15] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[15] > 0.0f) ? wts[15] * __uint_as_float(vec[15] & 4294901760u) : 0.0f);
      wts[16] = smem_w[w_base_7 + 16];
      acc[0] = acc[0] + ((wts[16] > 0.0f) ? wts[16] * __uint_as_float(vec[16] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[16] > 0.0f) ? wts[16] * __uint_as_float(vec[16] & 4294901760u) : 0.0f);
      wts[17] = smem_w[w_base_7 + 17];
      acc[0] = acc[0] + ((wts[17] > 0.0f) ? wts[17] * __uint_as_float(vec[17] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[17] > 0.0f) ? wts[17] * __uint_as_float(vec[17] & 4294901760u) : 0.0f);
      wts[18] = smem_w[w_base_7 + 18];
      acc[0] = acc[0] + ((wts[18] > 0.0f) ? wts[18] * __uint_as_float(vec[18] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[18] > 0.0f) ? wts[18] * __uint_as_float(vec[18] & 4294901760u) : 0.0f);
      wts[19] = smem_w[w_base_7 + 19];
      acc[0] = acc[0] + ((wts[19] > 0.0f) ? wts[19] * __uint_as_float(vec[19] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[19] > 0.0f) ? wts[19] * __uint_as_float(vec[19] & 4294901760u) : 0.0f);
      wts[20] = smem_w[w_base_7 + 20];
      acc[0] = acc[0] + ((wts[20] > 0.0f) ? wts[20] * __uint_as_float(vec[20] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[20] > 0.0f) ? wts[20] * __uint_as_float(vec[20] & 4294901760u) : 0.0f);
      wts[21] = smem_w[w_base_7 + 21];
      acc[0] = acc[0] + ((wts[21] > 0.0f) ? wts[21] * __uint_as_float(vec[21] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[21] > 0.0f) ? wts[21] * __uint_as_float(vec[21] & 4294901760u) : 0.0f);
      wts[22] = smem_w[w_base_7 + 22];
      acc[0] = acc[0] + ((wts[22] > 0.0f) ? wts[22] * __uint_as_float(vec[22] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[22] > 0.0f) ? wts[22] * __uint_as_float(vec[22] & 4294901760u) : 0.0f);
      wts[23] = smem_w[w_base_7 + 23];
      acc[0] = acc[0] + ((wts[23] > 0.0f) ? wts[23] * __uint_as_float(vec[23] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[23] > 0.0f) ? wts[23] * __uint_as_float(vec[23] & 4294901760u) : 0.0f);
      wts[24] = smem_w[w_base_7 + 24];
      acc[0] = acc[0] + ((wts[24] > 0.0f) ? wts[24] * __uint_as_float(vec[24] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[24] > 0.0f) ? wts[24] * __uint_as_float(vec[24] & 4294901760u) : 0.0f);
      wts[25] = smem_w[w_base_7 + 25];
      acc[0] = acc[0] + ((wts[25] > 0.0f) ? wts[25] * __uint_as_float(vec[25] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[25] > 0.0f) ? wts[25] * __uint_as_float(vec[25] & 4294901760u) : 0.0f);
      wts[26] = smem_w[w_base_7 + 26];
      acc[0] = acc[0] + ((wts[26] > 0.0f) ? wts[26] * __uint_as_float(vec[26] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[26] > 0.0f) ? wts[26] * __uint_as_float(vec[26] & 4294901760u) : 0.0f);
      wts[27] = smem_w[w_base_7 + 27];
      acc[0] = acc[0] + ((wts[27] > 0.0f) ? wts[27] * __uint_as_float(vec[27] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[27] > 0.0f) ? wts[27] * __uint_as_float(vec[27] & 4294901760u) : 0.0f);
      wts[28] = smem_w[w_base_7 + 28];
      acc[0] = acc[0] + ((wts[28] > 0.0f) ? wts[28] * __uint_as_float(vec[28] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[28] > 0.0f) ? wts[28] * __uint_as_float(vec[28] & 4294901760u) : 0.0f);
      wts[29] = smem_w[w_base_7 + 29];
      acc[0] = acc[0] + ((wts[29] > 0.0f) ? wts[29] * __uint_as_float(vec[29] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[29] > 0.0f) ? wts[29] * __uint_as_float(vec[29] & 4294901760u) : 0.0f);
      wts[30] = smem_w[w_base_7 + 30];
      acc[0] = acc[0] + ((wts[30] > 0.0f) ? wts[30] * __uint_as_float(vec[30] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[30] > 0.0f) ? wts[30] * __uint_as_float(vec[30] & 4294901760u) : 0.0f);
      wts[31] = smem_w[w_base_7 + 31];
      acc[0] = acc[0] + ((wts[31] > 0.0f) ? wts[31] * __uint_as_float(vec[31] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[31] > 0.0f) ? wts[31] * __uint_as_float(vec[31] & 4294901760u) : 0.0f);
      wts[32] = smem_w[w_base_7 + 32];
      acc[0] = acc[0] + ((wts[32] > 0.0f) ? wts[32] * __uint_as_float(vec[32] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[32] > 0.0f) ? wts[32] * __uint_as_float(vec[32] & 4294901760u) : 0.0f);
      wts[33] = smem_w[w_base_7 + 33];
      acc[0] = acc[0] + ((wts[33] > 0.0f) ? wts[33] * __uint_as_float(vec[33] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[33] > 0.0f) ? wts[33] * __uint_as_float(vec[33] & 4294901760u) : 0.0f);
      wts[34] = smem_w[w_base_7 + 34];
      acc[0] = acc[0] + ((wts[34] > 0.0f) ? wts[34] * __uint_as_float(vec[34] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[34] > 0.0f) ? wts[34] * __uint_as_float(vec[34] & 4294901760u) : 0.0f);
      wts[35] = smem_w[w_base_7 + 35];
      acc[0] = acc[0] + ((wts[35] > 0.0f) ? wts[35] * __uint_as_float(vec[35] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[35] > 0.0f) ? wts[35] * __uint_as_float(vec[35] & 4294901760u) : 0.0f);
      wts[36] = smem_w[w_base_7 + 36];
      acc[0] = acc[0] + ((wts[36] > 0.0f) ? wts[36] * __uint_as_float(vec[36] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[36] > 0.0f) ? wts[36] * __uint_as_float(vec[36] & 4294901760u) : 0.0f);
      wts[37] = smem_w[w_base_7 + 37];
      acc[0] = acc[0] + ((wts[37] > 0.0f) ? wts[37] * __uint_as_float(vec[37] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[37] > 0.0f) ? wts[37] * __uint_as_float(vec[37] & 4294901760u) : 0.0f);
      wts[38] = smem_w[w_base_7 + 38];
      acc[0] = acc[0] + ((wts[38] > 0.0f) ? wts[38] * __uint_as_float(vec[38] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[38] > 0.0f) ? wts[38] * __uint_as_float(vec[38] & 4294901760u) : 0.0f);
      wts[39] = smem_w[w_base_7 + 39];
      acc[0] = acc[0] + ((wts[39] > 0.0f) ? wts[39] * __uint_as_float(vec[39] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[39] > 0.0f) ? wts[39] * __uint_as_float(vec[39] & 4294901760u) : 0.0f);
      wts[40] = smem_w[w_base_7 + 40];
      acc[0] = acc[0] + ((wts[40] > 0.0f) ? wts[40] * __uint_as_float(vec[40] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[40] > 0.0f) ? wts[40] * __uint_as_float(vec[40] & 4294901760u) : 0.0f);
      wts[41] = smem_w[w_base_7 + 41];
      acc[0] = acc[0] + ((wts[41] > 0.0f) ? wts[41] * __uint_as_float(vec[41] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[41] > 0.0f) ? wts[41] * __uint_as_float(vec[41] & 4294901760u) : 0.0f);
      wts[42] = smem_w[w_base_7 + 42];
      acc[0] = acc[0] + ((wts[42] > 0.0f) ? wts[42] * __uint_as_float(vec[42] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[42] > 0.0f) ? wts[42] * __uint_as_float(vec[42] & 4294901760u) : 0.0f);
      wts[43] = smem_w[w_base_7 + 43];
      acc[0] = acc[0] + ((wts[43] > 0.0f) ? wts[43] * __uint_as_float(vec[43] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[43] > 0.0f) ? wts[43] * __uint_as_float(vec[43] & 4294901760u) : 0.0f);
      wts[44] = smem_w[w_base_7 + 44];
      acc[0] = acc[0] + ((wts[44] > 0.0f) ? wts[44] * __uint_as_float(vec[44] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[44] > 0.0f) ? wts[44] * __uint_as_float(vec[44] & 4294901760u) : 0.0f);
      wts[45] = smem_w[w_base_7 + 45];
      acc[0] = acc[0] + ((wts[45] > 0.0f) ? wts[45] * __uint_as_float(vec[45] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[45] > 0.0f) ? wts[45] * __uint_as_float(vec[45] & 4294901760u) : 0.0f);
      wts[46] = smem_w[w_base_7 + 46];
      acc[0] = acc[0] + ((wts[46] > 0.0f) ? wts[46] * __uint_as_float(vec[46] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[46] > 0.0f) ? wts[46] * __uint_as_float(vec[46] & 4294901760u) : 0.0f);
      wts[47] = smem_w[w_base_7 + 47];
      acc[0] = acc[0] + ((wts[47] > 0.0f) ? wts[47] * __uint_as_float(vec[47] << 16) : 0.0f);
      acc[1] =
          acc[1] + ((wts[47] > 0.0f) ? wts[47] * __uint_as_float(vec[47] & 4294901760u) : 0.0f);
      for (int e_1 = 0; e_1 < 2; e_1++) {
        acc[e_1] = acc[e_1] * inv;
      }
      unsigned int _packed[1];
#pragma unroll
      for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 =
            __float22bfloat162_rn(make_float2(acc[_lp * 2 + 0], acc[_lp * 2 + 1 + 0]));
        _packed[_lp] = *(uint32_t*)&_bf2;
      }
      if (merged == 0) {
        reinterpret_cast<unsigned*>(O + (row_7 * 512 + d0_7))[0] =
            reinterpret_cast<unsigned*>(_packed)[0];
        if (has_lse != 0) {
          if (part_7 == 0) {
            if (lane_7 == 0) {
              float lse_v = -CUDART_INF_F;
              if (sum_w > 0.0f) {
                float _log2_52;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_52) : "f"(sum_w));
                lse_v = (m + _log2_52) * 0.6931472f;
              }
              *(reinterpret_cast<float*>(lse + row_7) + (0)) = lse_v;
            }
          }
        }
      }
    } else {
      float l0_1 = -CUDART_INF_F;
      float l1_1 = -CUDART_INF_F;
      if (lane_7 < num_split) {
        l0_1 = partial_lse[stat_base_7 + lane_7];
      }
      if (lane_7 + 32 < num_split) {
        l1_1 = partial_lse[stat_base_7 + lane_7 + 32];
      }
      unsigned int vec_1[32];
      int _mn_48 = ((0) < (last_7) ? (0) : (last_7));
      {
        uint32_t _bf16x2_bits_53;
        _bf16x2_bits_53 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_48) * 512 + d0_7);
        vec_1[0] = _bf16x2_bits_53;
      }
      int _mn_49 = ((1) < (last_7) ? (1) : (last_7));
      {
        uint32_t _bf16x2_bits_54;
        _bf16x2_bits_54 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_49) * 512 + d0_7);
        vec_1[1] = _bf16x2_bits_54;
      }
      int _mn_50 = ((2) < (last_7) ? (2) : (last_7));
      {
        uint32_t _bf16x2_bits_55;
        _bf16x2_bits_55 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_50) * 512 + d0_7);
        vec_1[2] = _bf16x2_bits_55;
      }
      int _mn_51 = ((3) < (last_7) ? (3) : (last_7));
      {
        uint32_t _bf16x2_bits_56;
        _bf16x2_bits_56 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_51) * 512 + d0_7);
        vec_1[3] = _bf16x2_bits_56;
      }
      int _mn_52 = ((4) < (last_7) ? (4) : (last_7));
      {
        uint32_t _bf16x2_bits_57;
        _bf16x2_bits_57 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_52) * 512 + d0_7);
        vec_1[4] = _bf16x2_bits_57;
      }
      int _mn_53 = ((5) < (last_7) ? (5) : (last_7));
      {
        uint32_t _bf16x2_bits_58;
        _bf16x2_bits_58 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_53) * 512 + d0_7);
        vec_1[5] = _bf16x2_bits_58;
      }
      int _mn_54 = ((6) < (last_7) ? (6) : (last_7));
      {
        uint32_t _bf16x2_bits_59;
        _bf16x2_bits_59 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_54) * 512 + d0_7);
        vec_1[6] = _bf16x2_bits_59;
      }
      int _mn_55 = ((7) < (last_7) ? (7) : (last_7));
      {
        uint32_t _bf16x2_bits_60;
        _bf16x2_bits_60 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_55) * 512 + d0_7);
        vec_1[7] = _bf16x2_bits_60;
      }
      int _mn_56 = ((8) < (last_7) ? (8) : (last_7));
      {
        uint32_t _bf16x2_bits_61;
        _bf16x2_bits_61 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_56) * 512 + d0_7);
        vec_1[8] = _bf16x2_bits_61;
      }
      int _mn_57 = ((9) < (last_7) ? (9) : (last_7));
      {
        uint32_t _bf16x2_bits_62;
        _bf16x2_bits_62 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_57) * 512 + d0_7);
        vec_1[9] = _bf16x2_bits_62;
      }
      int _mn_58 = ((10) < (last_7) ? (10) : (last_7));
      {
        uint32_t _bf16x2_bits_63;
        _bf16x2_bits_63 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_58) * 512 + d0_7);
        vec_1[10] = _bf16x2_bits_63;
      }
      int _mn_59 = ((11) < (last_7) ? (11) : (last_7));
      {
        uint32_t _bf16x2_bits_64;
        _bf16x2_bits_64 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_59) * 512 + d0_7);
        vec_1[11] = _bf16x2_bits_64;
      }
      int _mn_60 = ((12) < (last_7) ? (12) : (last_7));
      {
        uint32_t _bf16x2_bits_65;
        _bf16x2_bits_65 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_60) * 512 + d0_7);
        vec_1[12] = _bf16x2_bits_65;
      }
      int _mn_61 = ((13) < (last_7) ? (13) : (last_7));
      {
        uint32_t _bf16x2_bits_66;
        _bf16x2_bits_66 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_61) * 512 + d0_7);
        vec_1[13] = _bf16x2_bits_66;
      }
      int _mn_62 = ((14) < (last_7) ? (14) : (last_7));
      {
        uint32_t _bf16x2_bits_67;
        _bf16x2_bits_67 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_62) * 512 + d0_7);
        vec_1[14] = _bf16x2_bits_67;
      }
      int _mn_63 = ((15) < (last_7) ? (15) : (last_7));
      {
        uint32_t _bf16x2_bits_68;
        _bf16x2_bits_68 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_63) * 512 + d0_7);
        vec_1[15] = _bf16x2_bits_68;
      }
      int _mn_64 = ((16) < (last_7) ? (16) : (last_7));
      {
        uint32_t _bf16x2_bits_69;
        _bf16x2_bits_69 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_64) * 512 + d0_7);
        vec_1[16] = _bf16x2_bits_69;
      }
      int _mn_65 = ((17) < (last_7) ? (17) : (last_7));
      {
        uint32_t _bf16x2_bits_70;
        _bf16x2_bits_70 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_65) * 512 + d0_7);
        vec_1[17] = _bf16x2_bits_70;
      }
      int _mn_66 = ((18) < (last_7) ? (18) : (last_7));
      {
        uint32_t _bf16x2_bits_71;
        _bf16x2_bits_71 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_66) * 512 + d0_7);
        vec_1[18] = _bf16x2_bits_71;
      }
      int _mn_67 = ((19) < (last_7) ? (19) : (last_7));
      {
        uint32_t _bf16x2_bits_72;
        _bf16x2_bits_72 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_67) * 512 + d0_7);
        vec_1[19] = _bf16x2_bits_72;
      }
      int _mn_68 = ((20) < (last_7) ? (20) : (last_7));
      {
        uint32_t _bf16x2_bits_73;
        _bf16x2_bits_73 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_68) * 512 + d0_7);
        vec_1[20] = _bf16x2_bits_73;
      }
      int _mn_69 = ((21) < (last_7) ? (21) : (last_7));
      {
        uint32_t _bf16x2_bits_74;
        _bf16x2_bits_74 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_69) * 512 + d0_7);
        vec_1[21] = _bf16x2_bits_74;
      }
      int _mn_70 = ((22) < (last_7) ? (22) : (last_7));
      {
        uint32_t _bf16x2_bits_75;
        _bf16x2_bits_75 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_70) * 512 + d0_7);
        vec_1[22] = _bf16x2_bits_75;
      }
      int _mn_71 = ((23) < (last_7) ? (23) : (last_7));
      {
        uint32_t _bf16x2_bits_76;
        _bf16x2_bits_76 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_71) * 512 + d0_7);
        vec_1[23] = _bf16x2_bits_76;
      }
      int _mn_72 = ((24) < (last_7) ? (24) : (last_7));
      {
        uint32_t _bf16x2_bits_77;
        _bf16x2_bits_77 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_72) * 512 + d0_7);
        vec_1[24] = _bf16x2_bits_77;
      }
      int _mn_73 = ((25) < (last_7) ? (25) : (last_7));
      {
        uint32_t _bf16x2_bits_78;
        _bf16x2_bits_78 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_73) * 512 + d0_7);
        vec_1[25] = _bf16x2_bits_78;
      }
      int _mn_74 = ((26) < (last_7) ? (26) : (last_7));
      {
        uint32_t _bf16x2_bits_79;
        _bf16x2_bits_79 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_74) * 512 + d0_7);
        vec_1[26] = _bf16x2_bits_79;
      }
      int _mn_75 = ((27) < (last_7) ? (27) : (last_7));
      {
        uint32_t _bf16x2_bits_80;
        _bf16x2_bits_80 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_75) * 512 + d0_7);
        vec_1[27] = _bf16x2_bits_80;
      }
      int _mn_76 = ((28) < (last_7) ? (28) : (last_7));
      {
        uint32_t _bf16x2_bits_81;
        _bf16x2_bits_81 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_76) * 512 + d0_7);
        vec_1[28] = _bf16x2_bits_81;
      }
      int _mn_77 = ((29) < (last_7) ? (29) : (last_7));
      {
        uint32_t _bf16x2_bits_82;
        _bf16x2_bits_82 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_77) * 512 + d0_7);
        vec_1[29] = _bf16x2_bits_82;
      }
      int _mn_78 = ((30) < (last_7) ? (30) : (last_7));
      {
        uint32_t _bf16x2_bits_83;
        _bf16x2_bits_83 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_78) * 512 + d0_7);
        vec_1[30] = _bf16x2_bits_83;
      }
      int _mn_79 = ((31) < (last_7) ? (31) : (last_7));
      {
        uint32_t _bf16x2_bits_84;
        _bf16x2_bits_84 =
            *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_79) * 512 + d0_7);
        vec_1[31] = _bf16x2_bits_84;
      }
      int merged_1 = 0;
      float _shfl_sync_85 = __shfl_sync(0xFFFFFFFF, l0_1, 0);
      merged_1 = _shfl_sync_85 == CUDART_INF_F;
      float m_1 = max_noftz(l0_1, l1_1);
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        m_1 = max_noftz(m_1, __shfl_xor_sync(0xFFFFFFFF, m_1, offset));
      float w0_1 = 0.0f;
      float w1_1 = 0.0f;
      if (l0_1 > -CUDART_INF_F) {
        float _exp2_86 = approx_exp2(l0_1 - m_1);
        w0_1 = _exp2_86;
      }
      if (l1_1 > -CUDART_INF_F) {
        float _exp2_87 = approx_exp2(l1_1 - m_1);
        w1_1 = _exp2_87;
      }
      smem_w[w_base_7 + lane_7] = w0_1;
      smem_w[w_base_7 + 32 + lane_7] = w1_1;
      float sum_w_1 = w0_1 + w1_1;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        sum_w_1 += __shfl_xor_sync(0xFFFFFFFF, sum_w_1, offset);
      __syncwarp();
      float inv_1 = 0.0f;
      if (sum_w_1 > 0.0f) {
        float _rcp_88 = approx_rcp(sum_w_1);
        inv_1 = _rcp_88;
      }
      float acc_1[2];
      for (int e_2 = 0; e_2 < 2; e_2++) {
        acc_1[e_2] = 0.0f;
      }
      float wts_1[32];
      wts_1[0] = smem_w[w_base_7 + 0];
      acc_1[0] = acc_1[0] + ((wts_1[0] > 0.0f) ? wts_1[0] * __uint_as_float(vec_1[0] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[0] > 0.0f) ? wts_1[0] * __uint_as_float(vec_1[0] & 4294901760u) : 0.0f);
      wts_1[1] = smem_w[w_base_7 + 1];
      acc_1[0] = acc_1[0] + ((wts_1[1] > 0.0f) ? wts_1[1] * __uint_as_float(vec_1[1] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[1] > 0.0f) ? wts_1[1] * __uint_as_float(vec_1[1] & 4294901760u) : 0.0f);
      wts_1[2] = smem_w[w_base_7 + 2];
      acc_1[0] = acc_1[0] + ((wts_1[2] > 0.0f) ? wts_1[2] * __uint_as_float(vec_1[2] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[2] > 0.0f) ? wts_1[2] * __uint_as_float(vec_1[2] & 4294901760u) : 0.0f);
      wts_1[3] = smem_w[w_base_7 + 3];
      acc_1[0] = acc_1[0] + ((wts_1[3] > 0.0f) ? wts_1[3] * __uint_as_float(vec_1[3] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[3] > 0.0f) ? wts_1[3] * __uint_as_float(vec_1[3] & 4294901760u) : 0.0f);
      wts_1[4] = smem_w[w_base_7 + 4];
      acc_1[0] = acc_1[0] + ((wts_1[4] > 0.0f) ? wts_1[4] * __uint_as_float(vec_1[4] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[4] > 0.0f) ? wts_1[4] * __uint_as_float(vec_1[4] & 4294901760u) : 0.0f);
      wts_1[5] = smem_w[w_base_7 + 5];
      acc_1[0] = acc_1[0] + ((wts_1[5] > 0.0f) ? wts_1[5] * __uint_as_float(vec_1[5] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[5] > 0.0f) ? wts_1[5] * __uint_as_float(vec_1[5] & 4294901760u) : 0.0f);
      wts_1[6] = smem_w[w_base_7 + 6];
      acc_1[0] = acc_1[0] + ((wts_1[6] > 0.0f) ? wts_1[6] * __uint_as_float(vec_1[6] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[6] > 0.0f) ? wts_1[6] * __uint_as_float(vec_1[6] & 4294901760u) : 0.0f);
      wts_1[7] = smem_w[w_base_7 + 7];
      acc_1[0] = acc_1[0] + ((wts_1[7] > 0.0f) ? wts_1[7] * __uint_as_float(vec_1[7] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[7] > 0.0f) ? wts_1[7] * __uint_as_float(vec_1[7] & 4294901760u) : 0.0f);
      wts_1[8] = smem_w[w_base_7 + 8];
      acc_1[0] = acc_1[0] + ((wts_1[8] > 0.0f) ? wts_1[8] * __uint_as_float(vec_1[8] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[8] > 0.0f) ? wts_1[8] * __uint_as_float(vec_1[8] & 4294901760u) : 0.0f);
      wts_1[9] = smem_w[w_base_7 + 9];
      acc_1[0] = acc_1[0] + ((wts_1[9] > 0.0f) ? wts_1[9] * __uint_as_float(vec_1[9] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[9] > 0.0f) ? wts_1[9] * __uint_as_float(vec_1[9] & 4294901760u) : 0.0f);
      wts_1[10] = smem_w[w_base_7 + 10];
      acc_1[0] =
          acc_1[0] + ((wts_1[10] > 0.0f) ? wts_1[10] * __uint_as_float(vec_1[10] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[10] > 0.0f) ? wts_1[10] * __uint_as_float(vec_1[10] & 4294901760u) : 0.0f);
      wts_1[11] = smem_w[w_base_7 + 11];
      acc_1[0] =
          acc_1[0] + ((wts_1[11] > 0.0f) ? wts_1[11] * __uint_as_float(vec_1[11] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[11] > 0.0f) ? wts_1[11] * __uint_as_float(vec_1[11] & 4294901760u) : 0.0f);
      wts_1[12] = smem_w[w_base_7 + 12];
      acc_1[0] =
          acc_1[0] + ((wts_1[12] > 0.0f) ? wts_1[12] * __uint_as_float(vec_1[12] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[12] > 0.0f) ? wts_1[12] * __uint_as_float(vec_1[12] & 4294901760u) : 0.0f);
      wts_1[13] = smem_w[w_base_7 + 13];
      acc_1[0] =
          acc_1[0] + ((wts_1[13] > 0.0f) ? wts_1[13] * __uint_as_float(vec_1[13] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[13] > 0.0f) ? wts_1[13] * __uint_as_float(vec_1[13] & 4294901760u) : 0.0f);
      wts_1[14] = smem_w[w_base_7 + 14];
      acc_1[0] =
          acc_1[0] + ((wts_1[14] > 0.0f) ? wts_1[14] * __uint_as_float(vec_1[14] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[14] > 0.0f) ? wts_1[14] * __uint_as_float(vec_1[14] & 4294901760u) : 0.0f);
      wts_1[15] = smem_w[w_base_7 + 15];
      acc_1[0] =
          acc_1[0] + ((wts_1[15] > 0.0f) ? wts_1[15] * __uint_as_float(vec_1[15] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[15] > 0.0f) ? wts_1[15] * __uint_as_float(vec_1[15] & 4294901760u) : 0.0f);
      wts_1[16] = smem_w[w_base_7 + 16];
      acc_1[0] =
          acc_1[0] + ((wts_1[16] > 0.0f) ? wts_1[16] * __uint_as_float(vec_1[16] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[16] > 0.0f) ? wts_1[16] * __uint_as_float(vec_1[16] & 4294901760u) : 0.0f);
      wts_1[17] = smem_w[w_base_7 + 17];
      acc_1[0] =
          acc_1[0] + ((wts_1[17] > 0.0f) ? wts_1[17] * __uint_as_float(vec_1[17] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[17] > 0.0f) ? wts_1[17] * __uint_as_float(vec_1[17] & 4294901760u) : 0.0f);
      wts_1[18] = smem_w[w_base_7 + 18];
      acc_1[0] =
          acc_1[0] + ((wts_1[18] > 0.0f) ? wts_1[18] * __uint_as_float(vec_1[18] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[18] > 0.0f) ? wts_1[18] * __uint_as_float(vec_1[18] & 4294901760u) : 0.0f);
      wts_1[19] = smem_w[w_base_7 + 19];
      acc_1[0] =
          acc_1[0] + ((wts_1[19] > 0.0f) ? wts_1[19] * __uint_as_float(vec_1[19] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[19] > 0.0f) ? wts_1[19] * __uint_as_float(vec_1[19] & 4294901760u) : 0.0f);
      wts_1[20] = smem_w[w_base_7 + 20];
      acc_1[0] =
          acc_1[0] + ((wts_1[20] > 0.0f) ? wts_1[20] * __uint_as_float(vec_1[20] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[20] > 0.0f) ? wts_1[20] * __uint_as_float(vec_1[20] & 4294901760u) : 0.0f);
      wts_1[21] = smem_w[w_base_7 + 21];
      acc_1[0] =
          acc_1[0] + ((wts_1[21] > 0.0f) ? wts_1[21] * __uint_as_float(vec_1[21] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[21] > 0.0f) ? wts_1[21] * __uint_as_float(vec_1[21] & 4294901760u) : 0.0f);
      wts_1[22] = smem_w[w_base_7 + 22];
      acc_1[0] =
          acc_1[0] + ((wts_1[22] > 0.0f) ? wts_1[22] * __uint_as_float(vec_1[22] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[22] > 0.0f) ? wts_1[22] * __uint_as_float(vec_1[22] & 4294901760u) : 0.0f);
      wts_1[23] = smem_w[w_base_7 + 23];
      acc_1[0] =
          acc_1[0] + ((wts_1[23] > 0.0f) ? wts_1[23] * __uint_as_float(vec_1[23] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[23] > 0.0f) ? wts_1[23] * __uint_as_float(vec_1[23] & 4294901760u) : 0.0f);
      wts_1[24] = smem_w[w_base_7 + 24];
      acc_1[0] =
          acc_1[0] + ((wts_1[24] > 0.0f) ? wts_1[24] * __uint_as_float(vec_1[24] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[24] > 0.0f) ? wts_1[24] * __uint_as_float(vec_1[24] & 4294901760u) : 0.0f);
      wts_1[25] = smem_w[w_base_7 + 25];
      acc_1[0] =
          acc_1[0] + ((wts_1[25] > 0.0f) ? wts_1[25] * __uint_as_float(vec_1[25] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[25] > 0.0f) ? wts_1[25] * __uint_as_float(vec_1[25] & 4294901760u) : 0.0f);
      wts_1[26] = smem_w[w_base_7 + 26];
      acc_1[0] =
          acc_1[0] + ((wts_1[26] > 0.0f) ? wts_1[26] * __uint_as_float(vec_1[26] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[26] > 0.0f) ? wts_1[26] * __uint_as_float(vec_1[26] & 4294901760u) : 0.0f);
      wts_1[27] = smem_w[w_base_7 + 27];
      acc_1[0] =
          acc_1[0] + ((wts_1[27] > 0.0f) ? wts_1[27] * __uint_as_float(vec_1[27] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[27] > 0.0f) ? wts_1[27] * __uint_as_float(vec_1[27] & 4294901760u) : 0.0f);
      wts_1[28] = smem_w[w_base_7 + 28];
      acc_1[0] =
          acc_1[0] + ((wts_1[28] > 0.0f) ? wts_1[28] * __uint_as_float(vec_1[28] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[28] > 0.0f) ? wts_1[28] * __uint_as_float(vec_1[28] & 4294901760u) : 0.0f);
      wts_1[29] = smem_w[w_base_7 + 29];
      acc_1[0] =
          acc_1[0] + ((wts_1[29] > 0.0f) ? wts_1[29] * __uint_as_float(vec_1[29] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[29] > 0.0f) ? wts_1[29] * __uint_as_float(vec_1[29] & 4294901760u) : 0.0f);
      wts_1[30] = smem_w[w_base_7 + 30];
      acc_1[0] =
          acc_1[0] + ((wts_1[30] > 0.0f) ? wts_1[30] * __uint_as_float(vec_1[30] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[30] > 0.0f) ? wts_1[30] * __uint_as_float(vec_1[30] & 4294901760u) : 0.0f);
      wts_1[31] = smem_w[w_base_7 + 31];
      acc_1[0] =
          acc_1[0] + ((wts_1[31] > 0.0f) ? wts_1[31] * __uint_as_float(vec_1[31] << 16) : 0.0f);
      acc_1[1] = acc_1[1] +
                 ((wts_1[31] > 0.0f) ? wts_1[31] * __uint_as_float(vec_1[31] & 4294901760u) : 0.0f);
      int n_batches = (num_split + 32 - 1) / 32;
#pragma unroll 1
      for (int bt = 1; bt < n_batches; bt++) {
        int k0 = bt * 32;
        int _mn_80 = ((k0 + 0) < (last_7) ? (k0 + 0) : (last_7));
        {
          uint32_t _bf16x2_bits_89;
          _bf16x2_bits_89 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_80) * 512 + d0_7);
          vec_1[0] = _bf16x2_bits_89;
        }
        int _mn_81 = ((k0 + 1) < (last_7) ? (k0 + 1) : (last_7));
        {
          uint32_t _bf16x2_bits_90;
          _bf16x2_bits_90 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_81) * 512 + d0_7);
          vec_1[1] = _bf16x2_bits_90;
        }
        int _mn_82 = ((k0 + 2) < (last_7) ? (k0 + 2) : (last_7));
        {
          uint32_t _bf16x2_bits_91;
          _bf16x2_bits_91 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_82) * 512 + d0_7);
          vec_1[2] = _bf16x2_bits_91;
        }
        int _mn_83 = ((k0 + 3) < (last_7) ? (k0 + 3) : (last_7));
        {
          uint32_t _bf16x2_bits_92;
          _bf16x2_bits_92 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_83) * 512 + d0_7);
          vec_1[3] = _bf16x2_bits_92;
        }
        int _mn_84 = ((k0 + 4) < (last_7) ? (k0 + 4) : (last_7));
        {
          uint32_t _bf16x2_bits_93;
          _bf16x2_bits_93 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_84) * 512 + d0_7);
          vec_1[4] = _bf16x2_bits_93;
        }
        int _mn_85 = ((k0 + 5) < (last_7) ? (k0 + 5) : (last_7));
        {
          uint32_t _bf16x2_bits_94;
          _bf16x2_bits_94 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_85) * 512 + d0_7);
          vec_1[5] = _bf16x2_bits_94;
        }
        int _mn_86 = ((k0 + 6) < (last_7) ? (k0 + 6) : (last_7));
        {
          uint32_t _bf16x2_bits_95;
          _bf16x2_bits_95 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_86) * 512 + d0_7);
          vec_1[6] = _bf16x2_bits_95;
        }
        int _mn_87 = ((k0 + 7) < (last_7) ? (k0 + 7) : (last_7));
        {
          uint32_t _bf16x2_bits_96;
          _bf16x2_bits_96 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_87) * 512 + d0_7);
          vec_1[7] = _bf16x2_bits_96;
        }
        int _mn_88 = ((k0 + 8) < (last_7) ? (k0 + 8) : (last_7));
        {
          uint32_t _bf16x2_bits_97;
          _bf16x2_bits_97 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_88) * 512 + d0_7);
          vec_1[8] = _bf16x2_bits_97;
        }
        int _mn_89 = ((k0 + 9) < (last_7) ? (k0 + 9) : (last_7));
        {
          uint32_t _bf16x2_bits_98;
          _bf16x2_bits_98 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_89) * 512 + d0_7);
          vec_1[9] = _bf16x2_bits_98;
        }
        int _mn_90 = ((k0 + 10) < (last_7) ? (k0 + 10) : (last_7));
        {
          uint32_t _bf16x2_bits_99;
          _bf16x2_bits_99 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_90) * 512 + d0_7);
          vec_1[10] = _bf16x2_bits_99;
        }
        int _mn_91 = ((k0 + 11) < (last_7) ? (k0 + 11) : (last_7));
        {
          uint32_t _bf16x2_bits_100;
          _bf16x2_bits_100 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_91) * 512 + d0_7);
          vec_1[11] = _bf16x2_bits_100;
        }
        int _mn_92 = ((k0 + 12) < (last_7) ? (k0 + 12) : (last_7));
        {
          uint32_t _bf16x2_bits_101;
          _bf16x2_bits_101 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_92) * 512 + d0_7);
          vec_1[12] = _bf16x2_bits_101;
        }
        int _mn_93 = ((k0 + 13) < (last_7) ? (k0 + 13) : (last_7));
        {
          uint32_t _bf16x2_bits_102;
          _bf16x2_bits_102 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_93) * 512 + d0_7);
          vec_1[13] = _bf16x2_bits_102;
        }
        int _mn_94 = ((k0 + 14) < (last_7) ? (k0 + 14) : (last_7));
        {
          uint32_t _bf16x2_bits_103;
          _bf16x2_bits_103 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_94) * 512 + d0_7);
          vec_1[14] = _bf16x2_bits_103;
        }
        int _mn_95 = ((k0 + 15) < (last_7) ? (k0 + 15) : (last_7));
        {
          uint32_t _bf16x2_bits_104;
          _bf16x2_bits_104 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_95) * 512 + d0_7);
          vec_1[15] = _bf16x2_bits_104;
        }
        int _mn_96 = ((k0 + 16) < (last_7) ? (k0 + 16) : (last_7));
        {
          uint32_t _bf16x2_bits_105;
          _bf16x2_bits_105 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_96) * 512 + d0_7);
          vec_1[16] = _bf16x2_bits_105;
        }
        int _mn_97 = ((k0 + 17) < (last_7) ? (k0 + 17) : (last_7));
        {
          uint32_t _bf16x2_bits_106;
          _bf16x2_bits_106 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_97) * 512 + d0_7);
          vec_1[17] = _bf16x2_bits_106;
        }
        int _mn_98 = ((k0 + 18) < (last_7) ? (k0 + 18) : (last_7));
        {
          uint32_t _bf16x2_bits_107;
          _bf16x2_bits_107 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_98) * 512 + d0_7);
          vec_1[18] = _bf16x2_bits_107;
        }
        int _mn_99 = ((k0 + 19) < (last_7) ? (k0 + 19) : (last_7));
        {
          uint32_t _bf16x2_bits_108;
          _bf16x2_bits_108 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_99) * 512 + d0_7);
          vec_1[19] = _bf16x2_bits_108;
        }
        int _mn_100 = ((k0 + 20) < (last_7) ? (k0 + 20) : (last_7));
        {
          uint32_t _bf16x2_bits_109;
          _bf16x2_bits_109 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_100) * 512 + d0_7);
          vec_1[20] = _bf16x2_bits_109;
        }
        int _mn_101 = ((k0 + 21) < (last_7) ? (k0 + 21) : (last_7));
        {
          uint32_t _bf16x2_bits_110;
          _bf16x2_bits_110 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_101) * 512 + d0_7);
          vec_1[21] = _bf16x2_bits_110;
        }
        int _mn_102 = ((k0 + 22) < (last_7) ? (k0 + 22) : (last_7));
        {
          uint32_t _bf16x2_bits_111;
          _bf16x2_bits_111 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_102) * 512 + d0_7);
          vec_1[22] = _bf16x2_bits_111;
        }
        int _mn_103 = ((k0 + 23) < (last_7) ? (k0 + 23) : (last_7));
        {
          uint32_t _bf16x2_bits_112;
          _bf16x2_bits_112 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_103) * 512 + d0_7);
          vec_1[23] = _bf16x2_bits_112;
        }
        int _mn_104 = ((k0 + 24) < (last_7) ? (k0 + 24) : (last_7));
        {
          uint32_t _bf16x2_bits_113;
          _bf16x2_bits_113 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_104) * 512 + d0_7);
          vec_1[24] = _bf16x2_bits_113;
        }
        int _mn_105 = ((k0 + 25) < (last_7) ? (k0 + 25) : (last_7));
        {
          uint32_t _bf16x2_bits_114;
          _bf16x2_bits_114 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_105) * 512 + d0_7);
          vec_1[25] = _bf16x2_bits_114;
        }
        int _mn_106 = ((k0 + 26) < (last_7) ? (k0 + 26) : (last_7));
        {
          uint32_t _bf16x2_bits_115;
          _bf16x2_bits_115 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_106) * 512 + d0_7);
          vec_1[26] = _bf16x2_bits_115;
        }
        int _mn_107 = ((k0 + 27) < (last_7) ? (k0 + 27) : (last_7));
        {
          uint32_t _bf16x2_bits_116;
          _bf16x2_bits_116 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_107) * 512 + d0_7);
          vec_1[27] = _bf16x2_bits_116;
        }
        int _mn_108 = ((k0 + 28) < (last_7) ? (k0 + 28) : (last_7));
        {
          uint32_t _bf16x2_bits_117;
          _bf16x2_bits_117 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_108) * 512 + d0_7);
          vec_1[28] = _bf16x2_bits_117;
        }
        int _mn_109 = ((k0 + 29) < (last_7) ? (k0 + 29) : (last_7));
        {
          uint32_t _bf16x2_bits_118;
          _bf16x2_bits_118 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_109) * 512 + d0_7);
          vec_1[29] = _bf16x2_bits_118;
        }
        int _mn_110 = ((k0 + 30) < (last_7) ? (k0 + 30) : (last_7));
        {
          uint32_t _bf16x2_bits_119;
          _bf16x2_bits_119 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_110) * 512 + d0_7);
          vec_1[30] = _bf16x2_bits_119;
        }
        int _mn_111 = ((k0 + 31) < (last_7) ? (k0 + 31) : (last_7));
        {
          uint32_t _bf16x2_bits_120;
          _bf16x2_bits_120 =
              *reinterpret_cast<const uint32_t*>(partial_o + (stat_base_7 + _mn_111) * 512 + d0_7);
          vec_1[31] = _bf16x2_bits_120;
        }
        wts_1[0] = smem_w[w_base_7 + (k0 + 0)];
        acc_1[0] =
            acc_1[0] + ((wts_1[0] > 0.0f) ? wts_1[0] * __uint_as_float(vec_1[0] << 16) : 0.0f);
        acc_1[1] = acc_1[1] +
                   ((wts_1[0] > 0.0f) ? wts_1[0] * __uint_as_float(vec_1[0] & 4294901760u) : 0.0f);
        wts_1[1] = smem_w[w_base_7 + (k0 + 1)];
        acc_1[0] =
            acc_1[0] + ((wts_1[1] > 0.0f) ? wts_1[1] * __uint_as_float(vec_1[1] << 16) : 0.0f);
        acc_1[1] = acc_1[1] +
                   ((wts_1[1] > 0.0f) ? wts_1[1] * __uint_as_float(vec_1[1] & 4294901760u) : 0.0f);
        wts_1[2] = smem_w[w_base_7 + (k0 + 2)];
        acc_1[0] =
            acc_1[0] + ((wts_1[2] > 0.0f) ? wts_1[2] * __uint_as_float(vec_1[2] << 16) : 0.0f);
        acc_1[1] = acc_1[1] +
                   ((wts_1[2] > 0.0f) ? wts_1[2] * __uint_as_float(vec_1[2] & 4294901760u) : 0.0f);
        wts_1[3] = smem_w[w_base_7 + (k0 + 3)];
        acc_1[0] =
            acc_1[0] + ((wts_1[3] > 0.0f) ? wts_1[3] * __uint_as_float(vec_1[3] << 16) : 0.0f);
        acc_1[1] = acc_1[1] +
                   ((wts_1[3] > 0.0f) ? wts_1[3] * __uint_as_float(vec_1[3] & 4294901760u) : 0.0f);
        wts_1[4] = smem_w[w_base_7 + (k0 + 4)];
        acc_1[0] =
            acc_1[0] + ((wts_1[4] > 0.0f) ? wts_1[4] * __uint_as_float(vec_1[4] << 16) : 0.0f);
        acc_1[1] = acc_1[1] +
                   ((wts_1[4] > 0.0f) ? wts_1[4] * __uint_as_float(vec_1[4] & 4294901760u) : 0.0f);
        wts_1[5] = smem_w[w_base_7 + (k0 + 5)];
        acc_1[0] =
            acc_1[0] + ((wts_1[5] > 0.0f) ? wts_1[5] * __uint_as_float(vec_1[5] << 16) : 0.0f);
        acc_1[1] = acc_1[1] +
                   ((wts_1[5] > 0.0f) ? wts_1[5] * __uint_as_float(vec_1[5] & 4294901760u) : 0.0f);
        wts_1[6] = smem_w[w_base_7 + (k0 + 6)];
        acc_1[0] =
            acc_1[0] + ((wts_1[6] > 0.0f) ? wts_1[6] * __uint_as_float(vec_1[6] << 16) : 0.0f);
        acc_1[1] = acc_1[1] +
                   ((wts_1[6] > 0.0f) ? wts_1[6] * __uint_as_float(vec_1[6] & 4294901760u) : 0.0f);
        wts_1[7] = smem_w[w_base_7 + (k0 + 7)];
        acc_1[0] =
            acc_1[0] + ((wts_1[7] > 0.0f) ? wts_1[7] * __uint_as_float(vec_1[7] << 16) : 0.0f);
        acc_1[1] = acc_1[1] +
                   ((wts_1[7] > 0.0f) ? wts_1[7] * __uint_as_float(vec_1[7] & 4294901760u) : 0.0f);
        wts_1[8] = smem_w[w_base_7 + (k0 + 8)];
        acc_1[0] =
            acc_1[0] + ((wts_1[8] > 0.0f) ? wts_1[8] * __uint_as_float(vec_1[8] << 16) : 0.0f);
        acc_1[1] = acc_1[1] +
                   ((wts_1[8] > 0.0f) ? wts_1[8] * __uint_as_float(vec_1[8] & 4294901760u) : 0.0f);
        wts_1[9] = smem_w[w_base_7 + (k0 + 9)];
        acc_1[0] =
            acc_1[0] + ((wts_1[9] > 0.0f) ? wts_1[9] * __uint_as_float(vec_1[9] << 16) : 0.0f);
        acc_1[1] = acc_1[1] +
                   ((wts_1[9] > 0.0f) ? wts_1[9] * __uint_as_float(vec_1[9] & 4294901760u) : 0.0f);
        wts_1[10] = smem_w[w_base_7 + (k0 + 10)];
        acc_1[0] =
            acc_1[0] + ((wts_1[10] > 0.0f) ? wts_1[10] * __uint_as_float(vec_1[10] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[10] > 0.0f) ? wts_1[10] * __uint_as_float(vec_1[10] & 4294901760u) : 0.0f);
        wts_1[11] = smem_w[w_base_7 + (k0 + 11)];
        acc_1[0] =
            acc_1[0] + ((wts_1[11] > 0.0f) ? wts_1[11] * __uint_as_float(vec_1[11] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[11] > 0.0f) ? wts_1[11] * __uint_as_float(vec_1[11] & 4294901760u) : 0.0f);
        wts_1[12] = smem_w[w_base_7 + (k0 + 12)];
        acc_1[0] =
            acc_1[0] + ((wts_1[12] > 0.0f) ? wts_1[12] * __uint_as_float(vec_1[12] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[12] > 0.0f) ? wts_1[12] * __uint_as_float(vec_1[12] & 4294901760u) : 0.0f);
        wts_1[13] = smem_w[w_base_7 + (k0 + 13)];
        acc_1[0] =
            acc_1[0] + ((wts_1[13] > 0.0f) ? wts_1[13] * __uint_as_float(vec_1[13] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[13] > 0.0f) ? wts_1[13] * __uint_as_float(vec_1[13] & 4294901760u) : 0.0f);
        wts_1[14] = smem_w[w_base_7 + (k0 + 14)];
        acc_1[0] =
            acc_1[0] + ((wts_1[14] > 0.0f) ? wts_1[14] * __uint_as_float(vec_1[14] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[14] > 0.0f) ? wts_1[14] * __uint_as_float(vec_1[14] & 4294901760u) : 0.0f);
        wts_1[15] = smem_w[w_base_7 + (k0 + 15)];
        acc_1[0] =
            acc_1[0] + ((wts_1[15] > 0.0f) ? wts_1[15] * __uint_as_float(vec_1[15] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[15] > 0.0f) ? wts_1[15] * __uint_as_float(vec_1[15] & 4294901760u) : 0.0f);
        wts_1[16] = smem_w[w_base_7 + (k0 + 16)];
        acc_1[0] =
            acc_1[0] + ((wts_1[16] > 0.0f) ? wts_1[16] * __uint_as_float(vec_1[16] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[16] > 0.0f) ? wts_1[16] * __uint_as_float(vec_1[16] & 4294901760u) : 0.0f);
        wts_1[17] = smem_w[w_base_7 + (k0 + 17)];
        acc_1[0] =
            acc_1[0] + ((wts_1[17] > 0.0f) ? wts_1[17] * __uint_as_float(vec_1[17] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[17] > 0.0f) ? wts_1[17] * __uint_as_float(vec_1[17] & 4294901760u) : 0.0f);
        wts_1[18] = smem_w[w_base_7 + (k0 + 18)];
        acc_1[0] =
            acc_1[0] + ((wts_1[18] > 0.0f) ? wts_1[18] * __uint_as_float(vec_1[18] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[18] > 0.0f) ? wts_1[18] * __uint_as_float(vec_1[18] & 4294901760u) : 0.0f);
        wts_1[19] = smem_w[w_base_7 + (k0 + 19)];
        acc_1[0] =
            acc_1[0] + ((wts_1[19] > 0.0f) ? wts_1[19] * __uint_as_float(vec_1[19] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[19] > 0.0f) ? wts_1[19] * __uint_as_float(vec_1[19] & 4294901760u) : 0.0f);
        wts_1[20] = smem_w[w_base_7 + (k0 + 20)];
        acc_1[0] =
            acc_1[0] + ((wts_1[20] > 0.0f) ? wts_1[20] * __uint_as_float(vec_1[20] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[20] > 0.0f) ? wts_1[20] * __uint_as_float(vec_1[20] & 4294901760u) : 0.0f);
        wts_1[21] = smem_w[w_base_7 + (k0 + 21)];
        acc_1[0] =
            acc_1[0] + ((wts_1[21] > 0.0f) ? wts_1[21] * __uint_as_float(vec_1[21] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[21] > 0.0f) ? wts_1[21] * __uint_as_float(vec_1[21] & 4294901760u) : 0.0f);
        wts_1[22] = smem_w[w_base_7 + (k0 + 22)];
        acc_1[0] =
            acc_1[0] + ((wts_1[22] > 0.0f) ? wts_1[22] * __uint_as_float(vec_1[22] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[22] > 0.0f) ? wts_1[22] * __uint_as_float(vec_1[22] & 4294901760u) : 0.0f);
        wts_1[23] = smem_w[w_base_7 + (k0 + 23)];
        acc_1[0] =
            acc_1[0] + ((wts_1[23] > 0.0f) ? wts_1[23] * __uint_as_float(vec_1[23] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[23] > 0.0f) ? wts_1[23] * __uint_as_float(vec_1[23] & 4294901760u) : 0.0f);
        wts_1[24] = smem_w[w_base_7 + (k0 + 24)];
        acc_1[0] =
            acc_1[0] + ((wts_1[24] > 0.0f) ? wts_1[24] * __uint_as_float(vec_1[24] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[24] > 0.0f) ? wts_1[24] * __uint_as_float(vec_1[24] & 4294901760u) : 0.0f);
        wts_1[25] = smem_w[w_base_7 + (k0 + 25)];
        acc_1[0] =
            acc_1[0] + ((wts_1[25] > 0.0f) ? wts_1[25] * __uint_as_float(vec_1[25] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[25] > 0.0f) ? wts_1[25] * __uint_as_float(vec_1[25] & 4294901760u) : 0.0f);
        wts_1[26] = smem_w[w_base_7 + (k0 + 26)];
        acc_1[0] =
            acc_1[0] + ((wts_1[26] > 0.0f) ? wts_1[26] * __uint_as_float(vec_1[26] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[26] > 0.0f) ? wts_1[26] * __uint_as_float(vec_1[26] & 4294901760u) : 0.0f);
        wts_1[27] = smem_w[w_base_7 + (k0 + 27)];
        acc_1[0] =
            acc_1[0] + ((wts_1[27] > 0.0f) ? wts_1[27] * __uint_as_float(vec_1[27] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[27] > 0.0f) ? wts_1[27] * __uint_as_float(vec_1[27] & 4294901760u) : 0.0f);
        wts_1[28] = smem_w[w_base_7 + (k0 + 28)];
        acc_1[0] =
            acc_1[0] + ((wts_1[28] > 0.0f) ? wts_1[28] * __uint_as_float(vec_1[28] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[28] > 0.0f) ? wts_1[28] * __uint_as_float(vec_1[28] & 4294901760u) : 0.0f);
        wts_1[29] = smem_w[w_base_7 + (k0 + 29)];
        acc_1[0] =
            acc_1[0] + ((wts_1[29] > 0.0f) ? wts_1[29] * __uint_as_float(vec_1[29] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[29] > 0.0f) ? wts_1[29] * __uint_as_float(vec_1[29] & 4294901760u) : 0.0f);
        wts_1[30] = smem_w[w_base_7 + (k0 + 30)];
        acc_1[0] =
            acc_1[0] + ((wts_1[30] > 0.0f) ? wts_1[30] * __uint_as_float(vec_1[30] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[30] > 0.0f) ? wts_1[30] * __uint_as_float(vec_1[30] & 4294901760u) : 0.0f);
        wts_1[31] = smem_w[w_base_7 + (k0 + 31)];
        acc_1[0] =
            acc_1[0] + ((wts_1[31] > 0.0f) ? wts_1[31] * __uint_as_float(vec_1[31] << 16) : 0.0f);
        acc_1[1] =
            acc_1[1] +
            ((wts_1[31] > 0.0f) ? wts_1[31] * __uint_as_float(vec_1[31] & 4294901760u) : 0.0f);
      }
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
      if (merged_1 == 0) {
        reinterpret_cast<unsigned*>(O + (row_7 * 512 + d0_7))[0] =
            reinterpret_cast<unsigned*>(_packed_1)[0];
        if (has_lse != 0) {
          if (part_7 == 0) {
            if (lane_7 == 0) {
              float lse_v_1 = -CUDART_INF_F;
              if (sum_w_1 > 0.0f) {
                float _log2_121;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_121) : "f"(sum_w_1));
                lse_v_1 = (m_1 + _log2_121) * 0.6931472f;
              }
              *(reinterpret_cast<float*>(lse + row_7) + (0)) = lse_v_1;
            }
          }
        }
      }
    }
  }
}

}  // extern "C"
