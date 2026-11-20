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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_WARP_DEN_OFF 0
#define SMEM_WARP_DEN_STAGE_BYTES 16
#define SMEM_WARP_DEN_STRIDE 16
#define SMEM_WARP_ACC_OFF 16
#define SMEM_WARP_ACC_STAGE_BYTES 3072
#define SMEM_WARP_ACC_STRIDE 3072
#define SMEM_TOTAL 0
#define THREADS 128

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

__global__ __launch_bounds__(128) void
kernel_cake_qsa_nvfp4_aa2349948b0a711974f9(float* __restrict__ partial_out, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ out, long long out_stride_row, long long out_stride_head, int num_splits, int num_qo_heads)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    __shared__ __align__(16) unsigned char smem_static_raw[3088];

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* warp_den = reinterpret_cast<float*>(smem_static_raw + 0);
    const int warp_den_addr = (int)(unsigned long long)__cvta_generic_to_shared(warp_den);
    float* warp_acc = reinterpret_cast<float*>(smem_static_raw + 16);
    const int warp_acc_addr = (int)(unsigned long long)__cvta_generic_to_shared(warp_acc);

    // === Task calls (declaration order) ===
    int row = blockIdx.x;
    int num_rows = gridDim.x;
    int lane_1 = (int)lane;
    int warp_1 = (int)warp;
    long long dim = (long long)lane_1 * 8;
    long long split_pitch = (long long)num_rows * (long long)num_qo_heads;
    if (num_splits > 4) {
        int head = blockIdx.y;
        long long row_head = (long long)row * (long long)num_qo_heads + (long long)head;
        int chunk = (num_splits + 3) / 4;
        int first = warp_1 * chunk;
        int mine = ((chunk) < (num_splits - first) ? (chunk) : (num_splits - first));
        int last = mine - 1;
        float gate_vals[8];
        if (warp_1 == 0) {
            long long gate_base = row_head * 256 + dim;
            {
                uint2 _vld_0;
                _vld_0 = *reinterpret_cast<const uint2*>(gate + gate_base);
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&gate_vals[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&gate_vals[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1;
                _vld_1 = *reinterpret_cast<const uint2*>(gate + gate_base + 4);
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&gate_vals[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&gate_vals[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        float lane_max = -CUDART_INF_F;
        float lse_g_3_0 = -CUDART_INF_F;
        if (lane_1 < num_splits) {
            lse_g_3_0 = partial_lse[(long long)lane_1 * split_pitch + row_head];
        }
        lane_max = fmaxf(lane_max, lse_g_3_0);
        float lse_g_3_1 = -CUDART_INF_F;
        if (32 + lane_1 < num_splits) {
            lse_g_3_1 = partial_lse[(long long)(32 + lane_1) * split_pitch + row_head];
        }
        lane_max = fmaxf(lane_max, lse_g_3_1);
        float lse_g_3_2 = -CUDART_INF_F;
        if (64 + lane_1 < num_splits) {
            lse_g_3_2 = partial_lse[(long long)(64 + lane_1) * split_pitch + row_head];
        }
        lane_max = fmaxf(lane_max, lse_g_3_2);
        float lse_g_3_3 = -CUDART_INF_F;
        if (96 + lane_1 < num_splits) {
            lse_g_3_3 = partial_lse[(long long)(96 + lane_1) * split_pitch + row_head];
        }
        lane_max = fmaxf(lane_max, lse_g_3_3);
        float lse = -CUDART_INF_F;
        if (lane_1 < mine) {
            lse = partial_lse[(long long)(first + lane_1) * split_pitch + row_head];
        }
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
        float pre[72];
        if (chunk > 4) {
            pre[0] = 0.0f;
            pre[4] = 0.0f;
            pre[1] = 0.0f;
            pre[5] = 0.0f;
            pre[2] = 0.0f;
            pre[6] = 0.0f;
            pre[3] = 0.0f;
            pre[7] = 0.0f;
            pre[8] = 0.0f;
            pre[12] = 0.0f;
            pre[9] = 0.0f;
            pre[13] = 0.0f;
            pre[10] = 0.0f;
            pre[14] = 0.0f;
            pre[11] = 0.0f;
            pre[15] = 0.0f;
            pre[16] = 0.0f;
            pre[20] = 0.0f;
            pre[17] = 0.0f;
            pre[21] = 0.0f;
            pre[18] = 0.0f;
            pre[22] = 0.0f;
            pre[19] = 0.0f;
            pre[23] = 0.0f;
            pre[24] = 0.0f;
            pre[28] = 0.0f;
            pre[25] = 0.0f;
            pre[29] = 0.0f;
            pre[26] = 0.0f;
            pre[30] = 0.0f;
            pre[27] = 0.0f;
            pre[31] = 0.0f;
            pre[32] = 0.0f;
            pre[36] = 0.0f;
            pre[33] = 0.0f;
            pre[37] = 0.0f;
            pre[34] = 0.0f;
            pre[38] = 0.0f;
            pre[35] = 0.0f;
            pre[39] = 0.0f;
            pre[40] = 0.0f;
            pre[44] = 0.0f;
            pre[41] = 0.0f;
            pre[45] = 0.0f;
            pre[42] = 0.0f;
            pre[46] = 0.0f;
            pre[43] = 0.0f;
            pre[47] = 0.0f;
            pre[48] = 0.0f;
            pre[52] = 0.0f;
            pre[49] = 0.0f;
            pre[53] = 0.0f;
            pre[50] = 0.0f;
            pre[54] = 0.0f;
            pre[51] = 0.0f;
            pre[55] = 0.0f;
            pre[56] = 0.0f;
            pre[60] = 0.0f;
            pre[57] = 0.0f;
            pre[61] = 0.0f;
            pre[58] = 0.0f;
            pre[62] = 0.0f;
            pre[59] = 0.0f;
            pre[63] = 0.0f;
            pre[64] = 0.0f;
            pre[68] = 0.0f;
            pre[65] = 0.0f;
            pre[69] = 0.0f;
            pre[66] = 0.0f;
            pre[70] = 0.0f;
            pre[67] = 0.0f;
            pre[71] = 0.0f;
            if (mine > 0) {
                long long pre_row = (long long)first * split_pitch + row_head;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row * 256 + dim);
                    pre[0 + 0] = _v4.x;
                    pre[0 + 1] = _v4.y;
                    pre[0 + 2] = _v4.z;
                    pre[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row * 256 + dim + 4);
                    pre[4 + 0] = _v4.x;
                    pre[4 + 1] = _v4.y;
                    pre[4 + 2] = _v4.z;
                    pre[4 + 3] = _v4.w;
                }
#else
        float pre[32];
        pre[0] = 0.0f;
        pre[4] = 0.0f;
        pre[1] = 0.0f;
        pre[5] = 0.0f;
        pre[2] = 0.0f;
        pre[6] = 0.0f;
        pre[3] = 0.0f;
        pre[7] = 0.0f;
        pre[8] = 0.0f;
        pre[12] = 0.0f;
        pre[9] = 0.0f;
        pre[13] = 0.0f;
        pre[10] = 0.0f;
        pre[14] = 0.0f;
        pre[11] = 0.0f;
        pre[15] = 0.0f;
        pre[16] = 0.0f;
        pre[20] = 0.0f;
        pre[17] = 0.0f;
        pre[21] = 0.0f;
        pre[18] = 0.0f;
        pre[22] = 0.0f;
        pre[19] = 0.0f;
        pre[23] = 0.0f;
        pre[24] = 0.0f;
        pre[28] = 0.0f;
        pre[25] = 0.0f;
        pre[29] = 0.0f;
        pre[26] = 0.0f;
        pre[30] = 0.0f;
        pre[27] = 0.0f;
        pre[31] = 0.0f;
        if (mine > 0) {
            long long pre_row = (long long)first * split_pitch + row_head;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row * 256 + dim);
                pre[0 + 0] = _v4.x;
                pre[0 + 1] = _v4.y;
                pre[0 + 2] = _v4.z;
                pre[0 + 3] = _v4.w;
#endif
            }
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            if (mine > 1) {
                long long pre_row_1 = (long long)(first + 1) * split_pitch + row_head;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_1 * 256 + dim);
                    pre[8 + 0] = _v4.x;
                    pre[8 + 1] = _v4.y;
                    pre[8 + 2] = _v4.z;
                    pre[8 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_1 * 256 + dim + 4);
                    pre[12 + 0] = _v4.x;
                    pre[12 + 1] = _v4.y;
                    pre[12 + 2] = _v4.z;
                    pre[12 + 3] = _v4.w;
                }
            }
            if (mine > 2) {
                long long pre_row_2 = (long long)(first + 2) * split_pitch + row_head;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_2 * 256 + dim);
                    pre[16 + 0] = _v4.x;
                    pre[16 + 1] = _v4.y;
                    pre[16 + 2] = _v4.z;
                    pre[16 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_2 * 256 + dim + 4);
                    pre[20 + 0] = _v4.x;
                    pre[20 + 1] = _v4.y;
                    pre[20 + 2] = _v4.z;
                    pre[20 + 3] = _v4.w;
                }
            }
            if (mine > 3) {
                long long pre_row_3 = (long long)(first + 3) * split_pitch + row_head;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_3 * 256 + dim);
                    pre[24 + 0] = _v4.x;
                    pre[24 + 1] = _v4.y;
                    pre[24 + 2] = _v4.z;
                    pre[24 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_3 * 256 + dim + 4);
                    pre[28 + 0] = _v4.x;
                    pre[28 + 1] = _v4.y;
                    pre[28 + 2] = _v4.z;
                    pre[28 + 3] = _v4.w;
                }
            }
            if (mine > 4) {
                long long pre_row_4 = (long long)(first + 4) * split_pitch + row_head;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_4 * 256 + dim);
                    pre[32 + 0] = _v4.x;
                    pre[32 + 1] = _v4.y;
                    pre[32 + 2] = _v4.z;
                    pre[32 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_4 * 256 + dim + 4);
                    pre[36 + 0] = _v4.x;
                    pre[36 + 1] = _v4.y;
                    pre[36 + 2] = _v4.z;
                    pre[36 + 3] = _v4.w;
                }
            }
            if (mine > 5) {
                long long pre_row_5 = (long long)(first + 5) * split_pitch + row_head;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_5 * 256 + dim);
                    pre[40 + 0] = _v4.x;
                    pre[40 + 1] = _v4.y;
                    pre[40 + 2] = _v4.z;
                    pre[40 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_5 * 256 + dim + 4);
                    pre[44 + 0] = _v4.x;
                    pre[44 + 1] = _v4.y;
                    pre[44 + 2] = _v4.z;
                    pre[44 + 3] = _v4.w;
                }
            }
            if (mine > 6) {
                long long pre_row_6 = (long long)(first + 6) * split_pitch + row_head;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_6 * 256 + dim);
                    pre[48 + 0] = _v4.x;
                    pre[48 + 1] = _v4.y;
                    pre[48 + 2] = _v4.z;
                    pre[48 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_6 * 256 + dim + 4);
                    pre[52 + 0] = _v4.x;
                    pre[52 + 1] = _v4.y;
                    pre[52 + 2] = _v4.z;
                    pre[52 + 3] = _v4.w;
                }
            }
            if (mine > 7) {
                long long pre_row_7 = (long long)(first + 7) * split_pitch + row_head;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_7 * 256 + dim);
                    pre[56 + 0] = _v4.x;
                    pre[56 + 1] = _v4.y;
                    pre[56 + 2] = _v4.z;
                    pre[56 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_7 * 256 + dim + 4);
                    pre[60 + 0] = _v4.x;
                    pre[60 + 1] = _v4.y;
                    pre[60 + 2] = _v4.z;
                    pre[60 + 3] = _v4.w;
                }
            }
            if (mine > 8) {
                long long pre_row_8 = (long long)(first + 8) * split_pitch + row_head;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_8 * 256 + dim);
                    pre[64 + 0] = _v4.x;
                    pre[64 + 1] = _v4.y;
                    pre[64 + 2] = _v4.z;
                    pre[64 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_8 * 256 + dim + 4);
                    pre[68 + 0] = _v4.x;
                    pre[68 + 1] = _v4.y;
                    pre[68 + 2] = _v4.z;
                    pre[68 + 3] = _v4.w;
                }
#else
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row * 256 + dim + 4);
                pre[4 + 0] = _v4.x;
                pre[4 + 1] = _v4.y;
                pre[4 + 2] = _v4.z;
                pre[4 + 3] = _v4.w;
#endif
            }
        }
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
        if (chunk <= 4) {
            pre[0] = 0.0f;
            pre[4] = 0.0f;
            pre[1] = 0.0f;
            pre[5] = 0.0f;
            pre[2] = 0.0f;
            pre[6] = 0.0f;
            pre[3] = 0.0f;
            pre[7] = 0.0f;
            pre[8] = 0.0f;
            pre[12] = 0.0f;
            pre[9] = 0.0f;
            pre[13] = 0.0f;
            pre[10] = 0.0f;
            pre[14] = 0.0f;
            pre[11] = 0.0f;
            pre[15] = 0.0f;
            pre[16] = 0.0f;
            pre[20] = 0.0f;
            pre[17] = 0.0f;
            pre[21] = 0.0f;
            pre[18] = 0.0f;
            pre[22] = 0.0f;
            pre[19] = 0.0f;
            pre[23] = 0.0f;
            pre[24] = 0.0f;
            pre[28] = 0.0f;
            pre[25] = 0.0f;
            pre[29] = 0.0f;
            pre[26] = 0.0f;
            pre[30] = 0.0f;
            pre[27] = 0.0f;
            pre[31] = 0.0f;
            if (mine > 0) {
                long long pre_row_9 = (long long)first * split_pitch + row_head;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_9 * 256 + dim);
                    pre[0 + 0] = _v4.x;
                    pre[0 + 1] = _v4.y;
                    pre[0 + 2] = _v4.z;
                    pre[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_9 * 256 + dim + 4);
                    pre[4 + 0] = _v4.x;
                    pre[4 + 1] = _v4.y;
                    pre[4 + 2] = _v4.z;
                    pre[4 + 3] = _v4.w;
                }
#else
        if (mine > 1) {
            long long pre_row_1 = (long long)(first + 1) * split_pitch + row_head;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_1 * 256 + dim);
                pre[8 + 0] = _v4.x;
                pre[8 + 1] = _v4.y;
                pre[8 + 2] = _v4.z;
                pre[8 + 3] = _v4.w;
#endif
            }
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            if (mine > 1) {
                long long pre_row_10 = (long long)(first + 1) * split_pitch + row_head;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_10 * 256 + dim);
                    pre[8 + 0] = _v4.x;
                    pre[8 + 1] = _v4.y;
                    pre[8 + 2] = _v4.z;
                    pre[8 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_10 * 256 + dim + 4);
                    pre[12 + 0] = _v4.x;
                    pre[12 + 1] = _v4.y;
                    pre[12 + 2] = _v4.z;
                    pre[12 + 3] = _v4.w;
                }
#else
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_1 * 256 + dim + 4);
                pre[12 + 0] = _v4.x;
                pre[12 + 1] = _v4.y;
                pre[12 + 2] = _v4.z;
                pre[12 + 3] = _v4.w;
#endif
            }
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            if (mine > 2) {
                long long pre_row_11 = (long long)(first + 2) * split_pitch + row_head;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_11 * 256 + dim);
                    pre[16 + 0] = _v4.x;
                    pre[16 + 1] = _v4.y;
                    pre[16 + 2] = _v4.z;
                    pre[16 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_11 * 256 + dim + 4);
                    pre[20 + 0] = _v4.x;
                    pre[20 + 1] = _v4.y;
                    pre[20 + 2] = _v4.z;
                    pre[20 + 3] = _v4.w;
                }
#else
        }
        if (mine > 2) {
            long long pre_row_2 = (long long)(first + 2) * split_pitch + row_head;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_2 * 256 + dim);
                pre[16 + 0] = _v4.x;
                pre[16 + 1] = _v4.y;
                pre[16 + 2] = _v4.z;
                pre[16 + 3] = _v4.w;
#endif
            }
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            if (mine > 3) {
                long long pre_row_12 = (long long)(first + 3) * split_pitch + row_head;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_12 * 256 + dim);
                    pre[24 + 0] = _v4.x;
                    pre[24 + 1] = _v4.y;
                    pre[24 + 2] = _v4.z;
                    pre[24 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_12 * 256 + dim + 4);
                    pre[28 + 0] = _v4.x;
                    pre[28 + 1] = _v4.y;
                    pre[28 + 2] = _v4.z;
                    pre[28 + 3] = _v4.w;
                }
#else
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_2 * 256 + dim + 4);
                pre[20 + 0] = _v4.x;
                pre[20 + 1] = _v4.y;
                pre[20 + 2] = _v4.z;
                pre[20 + 3] = _v4.w;
            }
        }
        if (mine > 3) {
            long long pre_row_3 = (long long)(first + 3) * split_pitch + row_head;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_3 * 256 + dim);
                pre[24 + 0] = _v4.x;
                pre[24 + 1] = _v4.y;
                pre[24 + 2] = _v4.z;
                pre[24 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_3 * 256 + dim + 4);
                pre[28 + 0] = _v4.x;
                pre[28 + 1] = _v4.y;
                pre[28 + 2] = _v4.z;
                pre[28 + 3] = _v4.w;
#endif
            }
        }
        float lse_max = lane_max;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            lse_max = max_noftz(lse_max, __shfl_xor_sync(0xFFFFFFFF, lse_max, offset));
        float acc[8];
        acc[0] = 0.0f;
        acc[4] = 0.0f;
        acc[1] = 0.0f;
        acc[5] = 0.0f;
        acc[2] = 0.0f;
        acc[6] = 0.0f;
        acc[3] = 0.0f;
        acc[7] = 0.0f;
        float den = 0.0f;
        if (lse_max > -CUDART_INF_F) {
            float weight_lane = approx_exp2(lse - lse_max);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            if (chunk > 4) {
                if (mine > 0) {
                    float weight = __shfl_sync(0xFFFFFFFF, weight_lane, 0);
                    den += weight;
                    acc[0] = __fmaf_rn(weight, pre[0], acc[0]);
                    acc[4] = __fmaf_rn(weight, pre[4], acc[4]);
                    acc[1] = __fmaf_rn(weight, pre[1], acc[1]);
                    acc[5] = __fmaf_rn(weight, pre[5], acc[5]);
                    acc[2] = __fmaf_rn(weight, pre[2], acc[2]);
                    acc[6] = __fmaf_rn(weight, pre[6], acc[6]);
                    acc[3] = __fmaf_rn(weight, pre[3], acc[3]);
                    acc[7] = __fmaf_rn(weight, pre[7], acc[7]);
                }
                if (mine > 1) {
                    float weight_1 = __shfl_sync(0xFFFFFFFF, weight_lane, 1);
                    den += weight_1;
                    acc[0] = __fmaf_rn(weight_1, pre[8], acc[0]);
                    acc[4] = __fmaf_rn(weight_1, pre[12], acc[4]);
                    acc[1] = __fmaf_rn(weight_1, pre[9], acc[1]);
                    acc[5] = __fmaf_rn(weight_1, pre[13], acc[5]);
                    acc[2] = __fmaf_rn(weight_1, pre[10], acc[2]);
                    acc[6] = __fmaf_rn(weight_1, pre[14], acc[6]);
                    acc[3] = __fmaf_rn(weight_1, pre[11], acc[3]);
                    acc[7] = __fmaf_rn(weight_1, pre[15], acc[7]);
                }
                if (mine > 2) {
                    float weight_2 = __shfl_sync(0xFFFFFFFF, weight_lane, 2);
                    den += weight_2;
                    acc[0] = __fmaf_rn(weight_2, pre[16], acc[0]);
                    acc[4] = __fmaf_rn(weight_2, pre[20], acc[4]);
                    acc[1] = __fmaf_rn(weight_2, pre[17], acc[1]);
                    acc[5] = __fmaf_rn(weight_2, pre[21], acc[5]);
                    acc[2] = __fmaf_rn(weight_2, pre[18], acc[2]);
                    acc[6] = __fmaf_rn(weight_2, pre[22], acc[6]);
                    acc[3] = __fmaf_rn(weight_2, pre[19], acc[3]);
                    acc[7] = __fmaf_rn(weight_2, pre[23], acc[7]);
                }
                if (mine > 3) {
                    float weight_3 = __shfl_sync(0xFFFFFFFF, weight_lane, 3);
                    den += weight_3;
                    acc[0] = __fmaf_rn(weight_3, pre[24], acc[0]);
                    acc[4] = __fmaf_rn(weight_3, pre[28], acc[4]);
                    acc[1] = __fmaf_rn(weight_3, pre[25], acc[1]);
                    acc[5] = __fmaf_rn(weight_3, pre[29], acc[5]);
                    acc[2] = __fmaf_rn(weight_3, pre[26], acc[2]);
                    acc[6] = __fmaf_rn(weight_3, pre[30], acc[6]);
                    acc[3] = __fmaf_rn(weight_3, pre[27], acc[3]);
                    acc[7] = __fmaf_rn(weight_3, pre[31], acc[7]);
                }
                if (mine > 4) {
                    float weight_4 = __shfl_sync(0xFFFFFFFF, weight_lane, 4);
                    den += weight_4;
                    acc[0] = __fmaf_rn(weight_4, pre[32], acc[0]);
                    acc[4] = __fmaf_rn(weight_4, pre[36], acc[4]);
                    acc[1] = __fmaf_rn(weight_4, pre[33], acc[1]);
                    acc[5] = __fmaf_rn(weight_4, pre[37], acc[5]);
                    acc[2] = __fmaf_rn(weight_4, pre[34], acc[2]);
                    acc[6] = __fmaf_rn(weight_4, pre[38], acc[6]);
                    acc[3] = __fmaf_rn(weight_4, pre[35], acc[3]);
                    acc[7] = __fmaf_rn(weight_4, pre[39], acc[7]);
                }
                if (mine > 5) {
                    float weight_5 = __shfl_sync(0xFFFFFFFF, weight_lane, 5);
                    den += weight_5;
                    acc[0] = __fmaf_rn(weight_5, pre[40], acc[0]);
                    acc[4] = __fmaf_rn(weight_5, pre[44], acc[4]);
                    acc[1] = __fmaf_rn(weight_5, pre[41], acc[1]);
                    acc[5] = __fmaf_rn(weight_5, pre[45], acc[5]);
                    acc[2] = __fmaf_rn(weight_5, pre[42], acc[2]);
                    acc[6] = __fmaf_rn(weight_5, pre[46], acc[6]);
                    acc[3] = __fmaf_rn(weight_5, pre[43], acc[3]);
                    acc[7] = __fmaf_rn(weight_5, pre[47], acc[7]);
                }
                if (mine > 6) {
                    float weight_6 = __shfl_sync(0xFFFFFFFF, weight_lane, 6);
                    den += weight_6;
                    acc[0] = __fmaf_rn(weight_6, pre[48], acc[0]);
                    acc[4] = __fmaf_rn(weight_6, pre[52], acc[4]);
                    acc[1] = __fmaf_rn(weight_6, pre[49], acc[1]);
                    acc[5] = __fmaf_rn(weight_6, pre[53], acc[5]);
                    acc[2] = __fmaf_rn(weight_6, pre[50], acc[2]);
                    acc[6] = __fmaf_rn(weight_6, pre[54], acc[6]);
                    acc[3] = __fmaf_rn(weight_6, pre[51], acc[3]);
                    acc[7] = __fmaf_rn(weight_6, pre[55], acc[7]);
                }
                if (mine > 7) {
                    float weight_7 = __shfl_sync(0xFFFFFFFF, weight_lane, 7);
                    den += weight_7;
                    acc[0] = __fmaf_rn(weight_7, pre[56], acc[0]);
                    acc[4] = __fmaf_rn(weight_7, pre[60], acc[4]);
                    acc[1] = __fmaf_rn(weight_7, pre[57], acc[1]);
                    acc[5] = __fmaf_rn(weight_7, pre[61], acc[5]);
                    acc[2] = __fmaf_rn(weight_7, pre[58], acc[2]);
                    acc[6] = __fmaf_rn(weight_7, pre[62], acc[6]);
                    acc[3] = __fmaf_rn(weight_7, pre[59], acc[3]);
                    acc[7] = __fmaf_rn(weight_7, pre[63], acc[7]);
                }
                if (mine > 8) {
                    float weight_8 = __shfl_sync(0xFFFFFFFF, weight_lane, 8);
                    den += weight_8;
                    acc[0] = __fmaf_rn(weight_8, pre[64], acc[0]);
                    acc[4] = __fmaf_rn(weight_8, pre[68], acc[4]);
                    acc[1] = __fmaf_rn(weight_8, pre[65], acc[1]);
                    acc[5] = __fmaf_rn(weight_8, pre[69], acc[5]);
                    acc[2] = __fmaf_rn(weight_8, pre[66], acc[2]);
                    acc[6] = __fmaf_rn(weight_8, pre[70], acc[6]);
                    acc[3] = __fmaf_rn(weight_8, pre[67], acc[3]);
                    acc[7] = __fmaf_rn(weight_8, pre[71], acc[7]);
                }
#else
            int _min_10 = ((0) < (last) ? (0) : (last));
            float weight_7_0 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_10);
            if (last < 0) {
                weight_7_0 = 0.0f;
#endif
            }
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            if (chunk <= 4) {
                int _min_28 = ((0) < (last) ? (0) : (last));
                float weight_8_0 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_28);
                if (last < 0) {
                    weight_8_0 = 0.0f;
                }
                den += weight_8_0;
                acc[0] = __fmaf_rn(weight_8_0, pre[0], acc[0]);
                acc[4] = __fmaf_rn(weight_8_0, pre[4], acc[4]);
                acc[1] = __fmaf_rn(weight_8_0, pre[1], acc[1]);
                acc[5] = __fmaf_rn(weight_8_0, pre[5], acc[5]);
                acc[2] = __fmaf_rn(weight_8_0, pre[2], acc[2]);
                acc[6] = __fmaf_rn(weight_8_0, pre[6], acc[6]);
                acc[3] = __fmaf_rn(weight_8_0, pre[3], acc[3]);
                acc[7] = __fmaf_rn(weight_8_0, pre[7], acc[7]);
                int _min_29 = ((1) < (last) ? (1) : (last));
                float weight_8_1 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_29);
                if (last < 1) {
                    weight_8_1 = 0.0f;
                }
                den += weight_8_1;
                acc[0] = __fmaf_rn(weight_8_1, pre[8], acc[0]);
                acc[4] = __fmaf_rn(weight_8_1, pre[12], acc[4]);
                acc[1] = __fmaf_rn(weight_8_1, pre[9], acc[1]);
                acc[5] = __fmaf_rn(weight_8_1, pre[13], acc[5]);
                acc[2] = __fmaf_rn(weight_8_1, pre[10], acc[2]);
                acc[6] = __fmaf_rn(weight_8_1, pre[14], acc[6]);
                acc[3] = __fmaf_rn(weight_8_1, pre[11], acc[3]);
                acc[7] = __fmaf_rn(weight_8_1, pre[15], acc[7]);
                int _min_30 = ((2) < (last) ? (2) : (last));
                float weight_8_2 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_30);
                if (last < 2) {
                    weight_8_2 = 0.0f;
                }
                den += weight_8_2;
                acc[0] = __fmaf_rn(weight_8_2, pre[16], acc[0]);
                acc[4] = __fmaf_rn(weight_8_2, pre[20], acc[4]);
                acc[1] = __fmaf_rn(weight_8_2, pre[17], acc[1]);
                acc[5] = __fmaf_rn(weight_8_2, pre[21], acc[5]);
                acc[2] = __fmaf_rn(weight_8_2, pre[18], acc[2]);
                acc[6] = __fmaf_rn(weight_8_2, pre[22], acc[6]);
                acc[3] = __fmaf_rn(weight_8_2, pre[19], acc[3]);
                acc[7] = __fmaf_rn(weight_8_2, pre[23], acc[7]);
                int _min_31 = ((3) < (last) ? (3) : (last));
                float weight_8_3 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_31);
                if (last < 3) {
                    weight_8_3 = 0.0f;
                }
                den += weight_8_3;
                acc[0] = __fmaf_rn(weight_8_3, pre[24], acc[0]);
                acc[4] = __fmaf_rn(weight_8_3, pre[28], acc[4]);
                acc[1] = __fmaf_rn(weight_8_3, pre[25], acc[1]);
                acc[5] = __fmaf_rn(weight_8_3, pre[29], acc[5]);
                acc[2] = __fmaf_rn(weight_8_3, pre[26], acc[2]);
                acc[6] = __fmaf_rn(weight_8_3, pre[30], acc[6]);
                acc[3] = __fmaf_rn(weight_8_3, pre[27], acc[3]);
                acc[7] = __fmaf_rn(weight_8_3, pre[31], acc[7]);
#else
            den += weight_7_0;
            acc[0] = __fmaf_rn(weight_7_0, pre[0], acc[0]);
            acc[4] = __fmaf_rn(weight_7_0, pre[4], acc[4]);
            acc[1] = __fmaf_rn(weight_7_0, pre[1], acc[1]);
            acc[5] = __fmaf_rn(weight_7_0, pre[5], acc[5]);
            acc[2] = __fmaf_rn(weight_7_0, pre[2], acc[2]);
            acc[6] = __fmaf_rn(weight_7_0, pre[6], acc[6]);
            acc[3] = __fmaf_rn(weight_7_0, pre[3], acc[3]);
            acc[7] = __fmaf_rn(weight_7_0, pre[7], acc[7]);
            int _min_11 = ((1) < (last) ? (1) : (last));
            float weight_7_1 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_11);
            if (last < 1) {
                weight_7_1 = 0.0f;
#endif
            }
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            for (int split_group = 9; split_group < mine; split_group += 9) {
#else
            den += weight_7_1;
            acc[0] = __fmaf_rn(weight_7_1, pre[8], acc[0]);
            acc[4] = __fmaf_rn(weight_7_1, pre[12], acc[4]);
            acc[1] = __fmaf_rn(weight_7_1, pre[9], acc[1]);
            acc[5] = __fmaf_rn(weight_7_1, pre[13], acc[5]);
            acc[2] = __fmaf_rn(weight_7_1, pre[10], acc[2]);
            acc[6] = __fmaf_rn(weight_7_1, pre[14], acc[6]);
            acc[3] = __fmaf_rn(weight_7_1, pre[11], acc[3]);
            acc[7] = __fmaf_rn(weight_7_1, pre[15], acc[7]);
            int _min_12 = ((2) < (last) ? (2) : (last));
            float weight_7_2 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_12);
            if (last < 2) {
                weight_7_2 = 0.0f;
            }
            den += weight_7_2;
            acc[0] = __fmaf_rn(weight_7_2, pre[16], acc[0]);
            acc[4] = __fmaf_rn(weight_7_2, pre[20], acc[4]);
            acc[1] = __fmaf_rn(weight_7_2, pre[17], acc[1]);
            acc[5] = __fmaf_rn(weight_7_2, pre[21], acc[5]);
            acc[2] = __fmaf_rn(weight_7_2, pre[18], acc[2]);
            acc[6] = __fmaf_rn(weight_7_2, pre[22], acc[6]);
            acc[3] = __fmaf_rn(weight_7_2, pre[19], acc[3]);
            acc[7] = __fmaf_rn(weight_7_2, pre[23], acc[7]);
            int _min_13 = ((3) < (last) ? (3) : (last));
            float weight_7_3 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_13);
            if (last < 3) {
                weight_7_3 = 0.0f;
            }
            den += weight_7_3;
            acc[0] = __fmaf_rn(weight_7_3, pre[24], acc[0]);
            acc[4] = __fmaf_rn(weight_7_3, pre[28], acc[4]);
            acc[1] = __fmaf_rn(weight_7_3, pre[25], acc[1]);
            acc[5] = __fmaf_rn(weight_7_3, pre[29], acc[5]);
            acc[2] = __fmaf_rn(weight_7_3, pre[26], acc[2]);
            acc[6] = __fmaf_rn(weight_7_3, pre[30], acc[6]);
            acc[3] = __fmaf_rn(weight_7_3, pre[27], acc[3]);
            acc[7] = __fmaf_rn(weight_7_3, pre[31], acc[7]);
            for (int split_group = 4; split_group < mine; split_group += 4) {
#endif
                if (mine > split_group) {
                    long long part_row = (long long)(first + split_group) * split_pitch + row_head;
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row * 256 + dim);
                        pre[0 + 0] = _v4.x;
                        pre[0 + 1] = _v4.y;
                        pre[0 + 2] = _v4.z;
                        pre[0 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row * 256 + dim + 4);
                        pre[4 + 0] = _v4.x;
                        pre[4 + 1] = _v4.y;
                        pre[4 + 2] = _v4.z;
                        pre[4 + 3] = _v4.w;
                    }
                }
                if (mine > split_group + 1) {
                    long long part_row_1 = (long long)(first + split_group + 1) * split_pitch + row_head;
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_1 * 256 + dim);
                        pre[8 + 0] = _v4.x;
                        pre[8 + 1] = _v4.y;
                        pre[8 + 2] = _v4.z;
                        pre[8 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_1 * 256 + dim + 4);
                        pre[12 + 0] = _v4.x;
                        pre[12 + 1] = _v4.y;
                        pre[12 + 2] = _v4.z;
                        pre[12 + 3] = _v4.w;
                    }
                }
                if (mine > split_group + 2) {
                    long long part_row_2 = (long long)(first + split_group + 2) * split_pitch + row_head;
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_2 * 256 + dim);
                        pre[16 + 0] = _v4.x;
                        pre[16 + 1] = _v4.y;
                        pre[16 + 2] = _v4.z;
                        pre[16 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_2 * 256 + dim + 4);
                        pre[20 + 0] = _v4.x;
                        pre[20 + 1] = _v4.y;
                        pre[20 + 2] = _v4.z;
                        pre[20 + 3] = _v4.w;
                    }
                }
                if (mine > split_group + 3) {
                    long long part_row_3 = (long long)(first + split_group + 3) * split_pitch + row_head;
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_3 * 256 + dim);
                        pre[24 + 0] = _v4.x;
                        pre[24 + 1] = _v4.y;
                        pre[24 + 2] = _v4.z;
                        pre[24 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_3 * 256 + dim + 4);
                        pre[28 + 0] = _v4.x;
                        pre[28 + 1] = _v4.y;
                        pre[28 + 2] = _v4.z;
                        pre[28 + 3] = _v4.w;
                    }
                }
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                if (mine > split_group + 4) {
                    long long part_row_4 = (long long)(first + split_group + 4) * split_pitch + row_head;
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_4 * 256 + dim);
                        pre[32 + 0] = _v4.x;
                        pre[32 + 1] = _v4.y;
                        pre[32 + 2] = _v4.z;
                        pre[32 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_4 * 256 + dim + 4);
                        pre[36 + 0] = _v4.x;
                        pre[36 + 1] = _v4.y;
                        pre[36 + 2] = _v4.z;
                        pre[36 + 3] = _v4.w;
                    }
                }
                if (mine > split_group + 5) {
                    long long part_row_5 = (long long)(first + split_group + 5) * split_pitch + row_head;
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_5 * 256 + dim);
                        pre[40 + 0] = _v4.x;
                        pre[40 + 1] = _v4.y;
                        pre[40 + 2] = _v4.z;
                        pre[40 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_5 * 256 + dim + 4);
                        pre[44 + 0] = _v4.x;
                        pre[44 + 1] = _v4.y;
                        pre[44 + 2] = _v4.z;
                        pre[44 + 3] = _v4.w;
                    }
                }
                if (mine > split_group + 6) {
                    long long part_row_6 = (long long)(first + split_group + 6) * split_pitch + row_head;
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_6 * 256 + dim);
                        pre[48 + 0] = _v4.x;
                        pre[48 + 1] = _v4.y;
                        pre[48 + 2] = _v4.z;
                        pre[48 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_6 * 256 + dim + 4);
                        pre[52 + 0] = _v4.x;
                        pre[52 + 1] = _v4.y;
                        pre[52 + 2] = _v4.z;
                        pre[52 + 3] = _v4.w;
                    }
                }
                if (mine > split_group + 7) {
                    long long part_row_7 = (long long)(first + split_group + 7) * split_pitch + row_head;
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_7 * 256 + dim);
                        pre[56 + 0] = _v4.x;
                        pre[56 + 1] = _v4.y;
                        pre[56 + 2] = _v4.z;
                        pre[56 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_7 * 256 + dim + 4);
                        pre[60 + 0] = _v4.x;
                        pre[60 + 1] = _v4.y;
                        pre[60 + 2] = _v4.z;
                        pre[60 + 3] = _v4.w;
                    }
                }
                if (mine > split_group + 8) {
                    long long part_row_8 = (long long)(first + split_group + 8) * split_pitch + row_head;
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_8 * 256 + dim);
                        pre[64 + 0] = _v4.x;
                        pre[64 + 1] = _v4.y;
                        pre[64 + 2] = _v4.z;
                        pre[64 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial_out + part_row_8 * 256 + dim + 4);
                        pre[68 + 0] = _v4.x;
                        pre[68 + 1] = _v4.y;
                        pre[68 + 2] = _v4.z;
                        pre[68 + 3] = _v4.w;
                    }
                }
                int _min_50 = ((split_group) < (last) ? (split_group) : (last));
                float weight_8_4 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_50);
#else
                int _min_22 = ((split_group) < (last) ? (split_group) : (last));
                float weight_7_4 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_22);
#endif
                if (last < split_group) {
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                    weight_8_4 = 0.0f;
#else
                    weight_7_4 = 0.0f;
#endif
                }
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                den += weight_8_4;
                acc[0] = __fmaf_rn(weight_8_4, pre[0], acc[0]);
                acc[4] = __fmaf_rn(weight_8_4, pre[4], acc[4]);
                acc[1] = __fmaf_rn(weight_8_4, pre[1], acc[1]);
                acc[5] = __fmaf_rn(weight_8_4, pre[5], acc[5]);
                acc[2] = __fmaf_rn(weight_8_4, pre[2], acc[2]);
                acc[6] = __fmaf_rn(weight_8_4, pre[6], acc[6]);
                acc[3] = __fmaf_rn(weight_8_4, pre[3], acc[3]);
                acc[7] = __fmaf_rn(weight_8_4, pre[7], acc[7]);
                int _min_51 = ((split_group + 1) < (last) ? (split_group + 1) : (last));
                float weight_8_5 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_51);
#else
                den += weight_7_4;
                acc[0] = __fmaf_rn(weight_7_4, pre[0], acc[0]);
                acc[4] = __fmaf_rn(weight_7_4, pre[4], acc[4]);
                acc[1] = __fmaf_rn(weight_7_4, pre[1], acc[1]);
                acc[5] = __fmaf_rn(weight_7_4, pre[5], acc[5]);
                acc[2] = __fmaf_rn(weight_7_4, pre[2], acc[2]);
                acc[6] = __fmaf_rn(weight_7_4, pre[6], acc[6]);
                acc[3] = __fmaf_rn(weight_7_4, pre[3], acc[3]);
                acc[7] = __fmaf_rn(weight_7_4, pre[7], acc[7]);
                int _min_23 = ((split_group + 1) < (last) ? (split_group + 1) : (last));
                float weight_7_5 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_23);
#endif
                if (last < split_group + 1) {
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                    weight_8_5 = 0.0f;
#else
                    weight_7_5 = 0.0f;
#endif
                }
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                den += weight_8_5;
                acc[0] = __fmaf_rn(weight_8_5, pre[8], acc[0]);
                acc[4] = __fmaf_rn(weight_8_5, pre[12], acc[4]);
                acc[1] = __fmaf_rn(weight_8_5, pre[9], acc[1]);
                acc[5] = __fmaf_rn(weight_8_5, pre[13], acc[5]);
                acc[2] = __fmaf_rn(weight_8_5, pre[10], acc[2]);
                acc[6] = __fmaf_rn(weight_8_5, pre[14], acc[6]);
                acc[3] = __fmaf_rn(weight_8_5, pre[11], acc[3]);
                acc[7] = __fmaf_rn(weight_8_5, pre[15], acc[7]);
                int _min_52 = ((split_group + 2) < (last) ? (split_group + 2) : (last));
                float weight_8_6 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_52);
#else
                den += weight_7_5;
                acc[0] = __fmaf_rn(weight_7_5, pre[8], acc[0]);
                acc[4] = __fmaf_rn(weight_7_5, pre[12], acc[4]);
                acc[1] = __fmaf_rn(weight_7_5, pre[9], acc[1]);
                acc[5] = __fmaf_rn(weight_7_5, pre[13], acc[5]);
                acc[2] = __fmaf_rn(weight_7_5, pre[10], acc[2]);
                acc[6] = __fmaf_rn(weight_7_5, pre[14], acc[6]);
                acc[3] = __fmaf_rn(weight_7_5, pre[11], acc[3]);
                acc[7] = __fmaf_rn(weight_7_5, pre[15], acc[7]);
                int _min_24 = ((split_group + 2) < (last) ? (split_group + 2) : (last));
                float weight_7_6 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_24);
#endif
                if (last < split_group + 2) {
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                    weight_8_6 = 0.0f;
#else
                    weight_7_6 = 0.0f;
#endif
                }
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                den += weight_8_6;
                acc[0] = __fmaf_rn(weight_8_6, pre[16], acc[0]);
                acc[4] = __fmaf_rn(weight_8_6, pre[20], acc[4]);
                acc[1] = __fmaf_rn(weight_8_6, pre[17], acc[1]);
                acc[5] = __fmaf_rn(weight_8_6, pre[21], acc[5]);
                acc[2] = __fmaf_rn(weight_8_6, pre[18], acc[2]);
                acc[6] = __fmaf_rn(weight_8_6, pre[22], acc[6]);
                acc[3] = __fmaf_rn(weight_8_6, pre[19], acc[3]);
                acc[7] = __fmaf_rn(weight_8_6, pre[23], acc[7]);
                int _min_53 = ((split_group + 3) < (last) ? (split_group + 3) : (last));
                float weight_8_7 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_53);
#else
                den += weight_7_6;
                acc[0] = __fmaf_rn(weight_7_6, pre[16], acc[0]);
                acc[4] = __fmaf_rn(weight_7_6, pre[20], acc[4]);
                acc[1] = __fmaf_rn(weight_7_6, pre[17], acc[1]);
                acc[5] = __fmaf_rn(weight_7_6, pre[21], acc[5]);
                acc[2] = __fmaf_rn(weight_7_6, pre[18], acc[2]);
                acc[6] = __fmaf_rn(weight_7_6, pre[22], acc[6]);
                acc[3] = __fmaf_rn(weight_7_6, pre[19], acc[3]);
                acc[7] = __fmaf_rn(weight_7_6, pre[23], acc[7]);
                int _min_25 = ((split_group + 3) < (last) ? (split_group + 3) : (last));
                float weight_7_7 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_25);
#endif
                if (last < split_group + 3) {
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                    weight_8_7 = 0.0f;
#else
                    weight_7_7 = 0.0f;
#endif
                }
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                den += weight_8_7;
                acc[0] = __fmaf_rn(weight_8_7, pre[24], acc[0]);
                acc[4] = __fmaf_rn(weight_8_7, pre[28], acc[4]);
                acc[1] = __fmaf_rn(weight_8_7, pre[25], acc[1]);
                acc[5] = __fmaf_rn(weight_8_7, pre[29], acc[5]);
                acc[2] = __fmaf_rn(weight_8_7, pre[26], acc[2]);
                acc[6] = __fmaf_rn(weight_8_7, pre[30], acc[6]);
                acc[3] = __fmaf_rn(weight_8_7, pre[27], acc[3]);
                acc[7] = __fmaf_rn(weight_8_7, pre[31], acc[7]);
                int _min_54 = ((split_group + 4) < (last) ? (split_group + 4) : (last));
                float weight_8_8 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_54);
                if (last < split_group + 4) {
                    weight_8_8 = 0.0f;
                }
                den += weight_8_8;
                acc[0] = __fmaf_rn(weight_8_8, pre[32], acc[0]);
                acc[4] = __fmaf_rn(weight_8_8, pre[36], acc[4]);
                acc[1] = __fmaf_rn(weight_8_8, pre[33], acc[1]);
                acc[5] = __fmaf_rn(weight_8_8, pre[37], acc[5]);
                acc[2] = __fmaf_rn(weight_8_8, pre[34], acc[2]);
                acc[6] = __fmaf_rn(weight_8_8, pre[38], acc[6]);
                acc[3] = __fmaf_rn(weight_8_8, pre[35], acc[3]);
                acc[7] = __fmaf_rn(weight_8_8, pre[39], acc[7]);
                int _min_55 = ((split_group + 5) < (last) ? (split_group + 5) : (last));
                float weight_8_9 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_55);
                if (last < split_group + 5) {
                    weight_8_9 = 0.0f;
                }
                den += weight_8_9;
                acc[0] = __fmaf_rn(weight_8_9, pre[40], acc[0]);
                acc[4] = __fmaf_rn(weight_8_9, pre[44], acc[4]);
                acc[1] = __fmaf_rn(weight_8_9, pre[41], acc[1]);
                acc[5] = __fmaf_rn(weight_8_9, pre[45], acc[5]);
                acc[2] = __fmaf_rn(weight_8_9, pre[42], acc[2]);
                acc[6] = __fmaf_rn(weight_8_9, pre[46], acc[6]);
                acc[3] = __fmaf_rn(weight_8_9, pre[43], acc[3]);
                acc[7] = __fmaf_rn(weight_8_9, pre[47], acc[7]);
                int _min_56 = ((split_group + 6) < (last) ? (split_group + 6) : (last));
                float weight_8_10 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_56);
                if (last < split_group + 6) {
                    weight_8_10 = 0.0f;
                }
                den += weight_8_10;
                acc[0] = __fmaf_rn(weight_8_10, pre[48], acc[0]);
                acc[4] = __fmaf_rn(weight_8_10, pre[52], acc[4]);
                acc[1] = __fmaf_rn(weight_8_10, pre[49], acc[1]);
                acc[5] = __fmaf_rn(weight_8_10, pre[53], acc[5]);
                acc[2] = __fmaf_rn(weight_8_10, pre[50], acc[2]);
                acc[6] = __fmaf_rn(weight_8_10, pre[54], acc[6]);
                acc[3] = __fmaf_rn(weight_8_10, pre[51], acc[3]);
                acc[7] = __fmaf_rn(weight_8_10, pre[55], acc[7]);
                int _min_57 = ((split_group + 7) < (last) ? (split_group + 7) : (last));
                float weight_8_11 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_57);
                if (last < split_group + 7) {
                    weight_8_11 = 0.0f;
                }
                den += weight_8_11;
                acc[0] = __fmaf_rn(weight_8_11, pre[56], acc[0]);
                acc[4] = __fmaf_rn(weight_8_11, pre[60], acc[4]);
                acc[1] = __fmaf_rn(weight_8_11, pre[57], acc[1]);
                acc[5] = __fmaf_rn(weight_8_11, pre[61], acc[5]);
                acc[2] = __fmaf_rn(weight_8_11, pre[58], acc[2]);
                acc[6] = __fmaf_rn(weight_8_11, pre[62], acc[6]);
                acc[3] = __fmaf_rn(weight_8_11, pre[59], acc[3]);
                acc[7] = __fmaf_rn(weight_8_11, pre[63], acc[7]);
                int _min_58 = ((split_group + 8) < (last) ? (split_group + 8) : (last));
                float weight_8_12 = __shfl_sync(0xFFFFFFFF, weight_lane, _min_58);
                if (last < split_group + 8) {
                    weight_8_12 = 0.0f;
                }
                den += weight_8_12;
                acc[0] = __fmaf_rn(weight_8_12, pre[64], acc[0]);
                acc[4] = __fmaf_rn(weight_8_12, pre[68], acc[4]);
                acc[1] = __fmaf_rn(weight_8_12, pre[65], acc[1]);
                acc[5] = __fmaf_rn(weight_8_12, pre[69], acc[5]);
                acc[2] = __fmaf_rn(weight_8_12, pre[66], acc[2]);
                acc[6] = __fmaf_rn(weight_8_12, pre[70], acc[6]);
                acc[3] = __fmaf_rn(weight_8_12, pre[67], acc[3]);
                acc[7] = __fmaf_rn(weight_8_12, pre[71], acc[7]);
#else
                den += weight_7_7;
                acc[0] = __fmaf_rn(weight_7_7, pre[24], acc[0]);
                acc[4] = __fmaf_rn(weight_7_7, pre[28], acc[4]);
                acc[1] = __fmaf_rn(weight_7_7, pre[25], acc[1]);
                acc[5] = __fmaf_rn(weight_7_7, pre[29], acc[5]);
                acc[2] = __fmaf_rn(weight_7_7, pre[26], acc[2]);
                acc[6] = __fmaf_rn(weight_7_7, pre[30], acc[6]);
                acc[3] = __fmaf_rn(weight_7_7, pre[27], acc[3]);
                acc[7] = __fmaf_rn(weight_7_7, pre[31], acc[7]);
#endif
            }
        }
        if (warp_1 > 0) {
            int acc_base = ((warp_1 - 1) * 256 + lane_1 * 8) * 4;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(warp_acc_addr + (unsigned int)acc_base), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(warp_acc_addr + (unsigned int)(acc_base + 16)), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])) : "memory");
            if (lane_1 == 0) {
                warp_den[warp_1] = den;
            }
        }
        __syncthreads();
        if (warp_1 == 0) {
            float other[8];
            int other_base_1_0 = lane_1 * 8 * 4;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&other[0])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 3]))
                : "r"(warp_acc_addr + (unsigned int)other_base_1_0));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&other[4])), "=r"(*reinterpret_cast<uint32_t*>(&other[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&other[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&other[(4) + 3]))
                : "r"(warp_acc_addr + (unsigned int)(other_base_1_0 + 16)));
            float other_den_2_0 = warp_den[1];
            den += other_den_2_0;
            acc[0] = acc[0] + other[0];
            acc[4] = acc[4] + other[4];
            acc[1] = acc[1] + other[1];
            acc[5] = acc[5] + other[5];
            acc[2] = acc[2] + other[2];
            acc[6] = acc[6] + other[6];
            acc[3] = acc[3] + other[3];
            acc[7] = acc[7] + other[7];
            int other_base_1_1 = (256 + lane_1 * 8) * 4;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&other[0])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 3]))
                : "r"(warp_acc_addr + (unsigned int)other_base_1_1));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&other[4])), "=r"(*reinterpret_cast<uint32_t*>(&other[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&other[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&other[(4) + 3]))
                : "r"(warp_acc_addr + (unsigned int)(other_base_1_1 + 16)));
            float other_den_2_1 = warp_den[2];
            den += other_den_2_1;
            acc[0] = acc[0] + other[0];
            acc[4] = acc[4] + other[4];
            acc[1] = acc[1] + other[1];
            acc[5] = acc[5] + other[5];
            acc[2] = acc[2] + other[2];
            acc[6] = acc[6] + other[6];
            acc[3] = acc[3] + other[3];
            acc[7] = acc[7] + other[7];
            int other_base_1_2 = (512 + lane_1 * 8) * 4;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&other[0])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 3]))
                : "r"(warp_acc_addr + (unsigned int)other_base_1_2));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&other[4])), "=r"(*reinterpret_cast<uint32_t*>(&other[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&other[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&other[(4) + 3]))
                : "r"(warp_acc_addr + (unsigned int)(other_base_1_2 + 16)));
            float other_den_2_2 = warp_den[3];
            den += other_den_2_2;
            acc[0] = acc[0] + other[0];
            acc[4] = acc[4] + other[4];
            acc[1] = acc[1] + other[1];
            acc[5] = acc[5] + other[5];
            acc[2] = acc[2] + other[2];
            acc[6] = acc[6] + other[6];
            acc[3] = acc[3] + other[3];
            acc[7] = acc[7] + other[7];
            float inv = 0.0f;
            if (den > 0.0f) {
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                float _rcp_59 = approx_rcp(den);
                inv = _rcp_59;
#else
                float _rcp_26 = approx_rcp(den);
                inv = _rcp_26;
#endif
            }
            float result[8];
            float attn_lo_1_0 = (float)(__nv_bfloat16)(acc[0] * inv);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_61 = approx_exp2((-gate_vals[0]) * 1.442695f);
            float _rcp_60 = approx_rcp(1.0f + _exp2_61);
            result[0] = attn_lo_1_0 * _rcp_60;
#else
            float _exp2_28 = approx_exp2((-gate_vals[0]) * 1.442695f);
            float _rcp_27 = approx_rcp(1.0f + _exp2_28);
            result[0] = attn_lo_1_0 * _rcp_27;
#endif
            float attn_hi_1_0 = (float)(__nv_bfloat16)(acc[4] * inv);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_63 = approx_exp2((-gate_vals[4]) * 1.442695f);
            float _rcp_62 = approx_rcp(1.0f + _exp2_63);
            result[4] = attn_hi_1_0 * _rcp_62;
#else
            float _exp2_30 = approx_exp2((-gate_vals[4]) * 1.442695f);
            float _rcp_29 = approx_rcp(1.0f + _exp2_30);
            result[4] = attn_hi_1_0 * _rcp_29;
#endif
            float attn_lo_1_1 = (float)(__nv_bfloat16)(acc[1] * inv);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_65 = approx_exp2((-gate_vals[1]) * 1.442695f);
            float _rcp_64 = approx_rcp(1.0f + _exp2_65);
            result[1] = attn_lo_1_1 * _rcp_64;
#else
            float _exp2_32 = approx_exp2((-gate_vals[1]) * 1.442695f);
            float _rcp_31 = approx_rcp(1.0f + _exp2_32);
            result[1] = attn_lo_1_1 * _rcp_31;
#endif
            float attn_hi_1_1 = (float)(__nv_bfloat16)(acc[5] * inv);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_67 = approx_exp2((-gate_vals[5]) * 1.442695f);
            float _rcp_66 = approx_rcp(1.0f + _exp2_67);
            result[5] = attn_hi_1_1 * _rcp_66;
#else
            float _exp2_34 = approx_exp2((-gate_vals[5]) * 1.442695f);
            float _rcp_33 = approx_rcp(1.0f + _exp2_34);
            result[5] = attn_hi_1_1 * _rcp_33;
#endif
            float attn_lo_1_2 = (float)(__nv_bfloat16)(acc[2] * inv);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_69 = approx_exp2((-gate_vals[2]) * 1.442695f);
            float _rcp_68 = approx_rcp(1.0f + _exp2_69);
            result[2] = attn_lo_1_2 * _rcp_68;
#else
            float _exp2_36 = approx_exp2((-gate_vals[2]) * 1.442695f);
            float _rcp_35 = approx_rcp(1.0f + _exp2_36);
            result[2] = attn_lo_1_2 * _rcp_35;
#endif
            float attn_hi_1_2 = (float)(__nv_bfloat16)(acc[6] * inv);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_71 = approx_exp2((-gate_vals[6]) * 1.442695f);
            float _rcp_70 = approx_rcp(1.0f + _exp2_71);
            result[6] = attn_hi_1_2 * _rcp_70;
#else
            float _exp2_38 = approx_exp2((-gate_vals[6]) * 1.442695f);
            float _rcp_37 = approx_rcp(1.0f + _exp2_38);
            result[6] = attn_hi_1_2 * _rcp_37;
#endif
            float attn_lo_1_3 = (float)(__nv_bfloat16)(acc[3] * inv);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_73 = approx_exp2((-gate_vals[3]) * 1.442695f);
            float _rcp_72 = approx_rcp(1.0f + _exp2_73);
            result[3] = attn_lo_1_3 * _rcp_72;
#else
            float _exp2_40 = approx_exp2((-gate_vals[3]) * 1.442695f);
            float _rcp_39 = approx_rcp(1.0f + _exp2_40);
            result[3] = attn_lo_1_3 * _rcp_39;
#endif
            float attn_hi_1_3 = (float)(__nv_bfloat16)(acc[7] * inv);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_75 = approx_exp2((-gate_vals[7]) * 1.442695f);
            float _rcp_74 = approx_rcp(1.0f + _exp2_75);
            result[7] = attn_hi_1_3 * _rcp_74;
#else
            float _exp2_42 = approx_exp2((-gate_vals[7]) * 1.442695f);
            float _rcp_41 = approx_rcp(1.0f + _exp2_42);
            result[7] = attn_hi_1_3 * _rcp_41;
#endif
            unsigned int words[4];
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            __nv_bfloat162 _bit_cast_src_76 = __float22bfloat162_rn(make_float2(result[0], result[1]));
            words[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_76)[0];
            __nv_bfloat162 _bit_cast_src_77 = __float22bfloat162_rn(make_float2(result[2], result[3]));
            words[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_77)[0];
            __nv_bfloat162 _bit_cast_src_78 = __float22bfloat162_rn(make_float2(result[4], result[5]));
            words[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_78)[0];
            __nv_bfloat162 _bit_cast_src_79 = __float22bfloat162_rn(make_float2(result[6], result[7]));
            words[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_79)[0];
#else
            __nv_bfloat162 _bit_cast_src_43 = __float22bfloat162_rn(make_float2(result[0], result[1]));
            words[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_43)[0];
            __nv_bfloat162 _bit_cast_src_44 = __float22bfloat162_rn(make_float2(result[2], result[3]));
            words[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_44)[0];
            __nv_bfloat162 _bit_cast_src_45 = __float22bfloat162_rn(make_float2(result[4], result[5]));
            words[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_45)[0];
            __nv_bfloat162 _bit_cast_src_46 = __float22bfloat162_rn(make_float2(result[6], result[7]));
            words[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_46)[0];
#endif
            long long out_offset = (long long)row * out_stride_row + (long long)head * out_stride_head + dim;
            reinterpret_cast<int2*>(out + out_offset)[0] = reinterpret_cast<int2*>(words)[0];
            reinterpret_cast<int2*>(out + (out_offset + 4))[0] = reinterpret_cast<int2*>(words + (2))[0];
        }
    } else {
        int head_b = (int)blockIdx.y * 4 + warp_1;
        if (head_b < num_qo_heads) {
            long long row_head_b = (long long)row * (long long)num_qo_heads + (long long)head_b;
            int last_b = num_splits - 1;
            float gate_b[8];
            long long gate_base_b = row_head_b * 256 + dim;
            {
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                uint2 _vld_80;
                _vld_80 = *reinterpret_cast<const uint2*>(gate + gate_base_b);
                uint32_t* _vpairs_80 = reinterpret_cast<uint32_t*>(&_vld_80);
#else
                uint2 _vld_47;
                _vld_47 = *reinterpret_cast<const uint2*>(gate + gate_base_b);
                uint32_t* _vpairs_47 = reinterpret_cast<uint32_t*>(&_vld_47);
#endif
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                    (&gate_b[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) << 16);
                    (&gate_b[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) & 0xffff0000u);
#else
                    (&gate_b[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) << 16);
                    (&gate_b[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) & 0xffff0000u);
#endif
                }
            }
            {
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                uint2 _vld_81;
                _vld_81 = *reinterpret_cast<const uint2*>(gate + gate_base_b + 4);
                uint32_t* _vpairs_81 = reinterpret_cast<uint32_t*>(&_vld_81);
#else
                uint2 _vld_48;
                _vld_48 = *reinterpret_cast<const uint2*>(gate + gate_base_b + 4);
                uint32_t* _vpairs_48 = reinterpret_cast<uint32_t*>(&_vld_48);
#endif
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                    (&gate_b[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_81[_pair]) << 16);
                    (&gate_b[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_81[_pair]) & 0xffff0000u);
#else
                    (&gate_b[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_48[_pair]) << 16);
                    (&gate_b[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_48[_pair]) & 0xffff0000u);
#endif
                }
            }
            float lse_b = -CUDART_INF_F;
            if (lane_1 < num_splits) {
                lse_b = partial_lse[(long long)lane_1 * split_pitch + row_head_b];
            }
            float pre_b[32];
            pre_b[0] = 0.0f;
            pre_b[4] = 0.0f;
            pre_b[1] = 0.0f;
            pre_b[5] = 0.0f;
            pre_b[2] = 0.0f;
            pre_b[6] = 0.0f;
            pre_b[3] = 0.0f;
            pre_b[7] = 0.0f;
            pre_b[8] = 0.0f;
            pre_b[12] = 0.0f;
            pre_b[9] = 0.0f;
            pre_b[13] = 0.0f;
            pre_b[10] = 0.0f;
            pre_b[14] = 0.0f;
            pre_b[11] = 0.0f;
            pre_b[15] = 0.0f;
            pre_b[16] = 0.0f;
            pre_b[20] = 0.0f;
            pre_b[17] = 0.0f;
            pre_b[21] = 0.0f;
            pre_b[18] = 0.0f;
            pre_b[22] = 0.0f;
            pre_b[19] = 0.0f;
            pre_b[23] = 0.0f;
            pre_b[24] = 0.0f;
            pre_b[28] = 0.0f;
            pre_b[25] = 0.0f;
            pre_b[29] = 0.0f;
            pre_b[26] = 0.0f;
            pre_b[30] = 0.0f;
            pre_b[27] = 0.0f;
            pre_b[31] = 0.0f;
            if (num_splits > 0) {
                long long pre_row_b = row_head_b;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_b * 256 + dim);
                    pre_b[0 + 0] = _v4.x;
                    pre_b[0 + 1] = _v4.y;
                    pre_b[0 + 2] = _v4.z;
                    pre_b[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_b * 256 + dim + 4);
                    pre_b[4 + 0] = _v4.x;
                    pre_b[4 + 1] = _v4.y;
                    pre_b[4 + 2] = _v4.z;
                    pre_b[4 + 3] = _v4.w;
                }
            }
            if (num_splits > 1) {
                long long pre_row_b_1 = split_pitch + row_head_b;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_b_1 * 256 + dim);
                    pre_b[8 + 0] = _v4.x;
                    pre_b[8 + 1] = _v4.y;
                    pre_b[8 + 2] = _v4.z;
                    pre_b[8 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_b_1 * 256 + dim + 4);
                    pre_b[12 + 0] = _v4.x;
                    pre_b[12 + 1] = _v4.y;
                    pre_b[12 + 2] = _v4.z;
                    pre_b[12 + 3] = _v4.w;
                }
            }
            if (num_splits > 2) {
                long long pre_row_b_2 = 2 * split_pitch + row_head_b;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_b_2 * 256 + dim);
                    pre_b[16 + 0] = _v4.x;
                    pre_b[16 + 1] = _v4.y;
                    pre_b[16 + 2] = _v4.z;
                    pre_b[16 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_b_2 * 256 + dim + 4);
                    pre_b[20 + 0] = _v4.x;
                    pre_b[20 + 1] = _v4.y;
                    pre_b[20 + 2] = _v4.z;
                    pre_b[20 + 3] = _v4.w;
                }
            }
            if (num_splits > 3) {
                long long pre_row_b_3 = 3 * split_pitch + row_head_b;
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_b_3 * 256 + dim);
                    pre_b[24 + 0] = _v4.x;
                    pre_b[24 + 1] = _v4.y;
                    pre_b[24 + 2] = _v4.z;
                    pre_b[24 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_out + pre_row_b_3 * 256 + dim + 4);
                    pre_b[28 + 0] = _v4.x;
                    pre_b[28 + 1] = _v4.y;
                    pre_b[28 + 2] = _v4.z;
                    pre_b[28 + 3] = _v4.w;
                }
            }
            float lse_max_b = lse_b;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                lse_max_b = max_noftz(lse_max_b, __shfl_xor_sync(0xFFFFFFFF, lse_max_b, offset));
            float acc_b[8];
            acc_b[0] = 0.0f;
            acc_b[4] = 0.0f;
            acc_b[1] = 0.0f;
            acc_b[5] = 0.0f;
            acc_b[2] = 0.0f;
            acc_b[6] = 0.0f;
            acc_b[3] = 0.0f;
            acc_b[7] = 0.0f;
            float den_b = 0.0f;
            if (lse_max_b > -CUDART_INF_F) {
                float weight_b = approx_exp2(lse_b - lse_max_b);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                int _min_90 = ((0) < (last_b) ? (0) : (last_b));
                float w_b_3_0 = __shfl_sync(0xFFFFFFFF, weight_b, _min_90);
#else
                int _min_57 = ((0) < (last_b) ? (0) : (last_b));
                float w_b_3_0 = __shfl_sync(0xFFFFFFFF, weight_b, _min_57);
#endif
                if (last_b < 0) {
                    w_b_3_0 = 0.0f;
                }
                den_b += w_b_3_0;
                acc_b[0] = __fmaf_rn(w_b_3_0, pre_b[0], acc_b[0]);
                acc_b[4] = __fmaf_rn(w_b_3_0, pre_b[4], acc_b[4]);
                acc_b[1] = __fmaf_rn(w_b_3_0, pre_b[1], acc_b[1]);
                acc_b[5] = __fmaf_rn(w_b_3_0, pre_b[5], acc_b[5]);
                acc_b[2] = __fmaf_rn(w_b_3_0, pre_b[2], acc_b[2]);
                acc_b[6] = __fmaf_rn(w_b_3_0, pre_b[6], acc_b[6]);
                acc_b[3] = __fmaf_rn(w_b_3_0, pre_b[3], acc_b[3]);
                acc_b[7] = __fmaf_rn(w_b_3_0, pre_b[7], acc_b[7]);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                int _min_91 = ((1) < (last_b) ? (1) : (last_b));
                float w_b_3_1 = __shfl_sync(0xFFFFFFFF, weight_b, _min_91);
#else
                int _min_58 = ((1) < (last_b) ? (1) : (last_b));
                float w_b_3_1 = __shfl_sync(0xFFFFFFFF, weight_b, _min_58);
#endif
                if (last_b < 1) {
                    w_b_3_1 = 0.0f;
                }
                den_b += w_b_3_1;
                acc_b[0] = __fmaf_rn(w_b_3_1, pre_b[8], acc_b[0]);
                acc_b[4] = __fmaf_rn(w_b_3_1, pre_b[12], acc_b[4]);
                acc_b[1] = __fmaf_rn(w_b_3_1, pre_b[9], acc_b[1]);
                acc_b[5] = __fmaf_rn(w_b_3_1, pre_b[13], acc_b[5]);
                acc_b[2] = __fmaf_rn(w_b_3_1, pre_b[10], acc_b[2]);
                acc_b[6] = __fmaf_rn(w_b_3_1, pre_b[14], acc_b[6]);
                acc_b[3] = __fmaf_rn(w_b_3_1, pre_b[11], acc_b[3]);
                acc_b[7] = __fmaf_rn(w_b_3_1, pre_b[15], acc_b[7]);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                int _min_92 = ((2) < (last_b) ? (2) : (last_b));
                float w_b_3_2 = __shfl_sync(0xFFFFFFFF, weight_b, _min_92);
#else
                int _min_59 = ((2) < (last_b) ? (2) : (last_b));
                float w_b_3_2 = __shfl_sync(0xFFFFFFFF, weight_b, _min_59);
#endif
                if (last_b < 2) {
                    w_b_3_2 = 0.0f;
                }
                den_b += w_b_3_2;
                acc_b[0] = __fmaf_rn(w_b_3_2, pre_b[16], acc_b[0]);
                acc_b[4] = __fmaf_rn(w_b_3_2, pre_b[20], acc_b[4]);
                acc_b[1] = __fmaf_rn(w_b_3_2, pre_b[17], acc_b[1]);
                acc_b[5] = __fmaf_rn(w_b_3_2, pre_b[21], acc_b[5]);
                acc_b[2] = __fmaf_rn(w_b_3_2, pre_b[18], acc_b[2]);
                acc_b[6] = __fmaf_rn(w_b_3_2, pre_b[22], acc_b[6]);
                acc_b[3] = __fmaf_rn(w_b_3_2, pre_b[19], acc_b[3]);
                acc_b[7] = __fmaf_rn(w_b_3_2, pre_b[23], acc_b[7]);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                int _min_93 = ((3) < (last_b) ? (3) : (last_b));
                float w_b_3_3 = __shfl_sync(0xFFFFFFFF, weight_b, _min_93);
#else
                int _min_60 = ((3) < (last_b) ? (3) : (last_b));
                float w_b_3_3 = __shfl_sync(0xFFFFFFFF, weight_b, _min_60);
#endif
                if (last_b < 3) {
                    w_b_3_3 = 0.0f;
                }
                den_b += w_b_3_3;
                acc_b[0] = __fmaf_rn(w_b_3_3, pre_b[24], acc_b[0]);
                acc_b[4] = __fmaf_rn(w_b_3_3, pre_b[28], acc_b[4]);
                acc_b[1] = __fmaf_rn(w_b_3_3, pre_b[25], acc_b[1]);
                acc_b[5] = __fmaf_rn(w_b_3_3, pre_b[29], acc_b[5]);
                acc_b[2] = __fmaf_rn(w_b_3_3, pre_b[26], acc_b[2]);
                acc_b[6] = __fmaf_rn(w_b_3_3, pre_b[30], acc_b[6]);
                acc_b[3] = __fmaf_rn(w_b_3_3, pre_b[27], acc_b[3]);
                acc_b[7] = __fmaf_rn(w_b_3_3, pre_b[31], acc_b[7]);
            }
            float inv_b = 0.0f;
            if (den_b > 0.0f) {
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
                float _rcp_94 = approx_rcp(den_b);
                inv_b = _rcp_94;
#else
                float _rcp_61 = approx_rcp(den_b);
                inv_b = _rcp_61;
#endif
            }
            float result_b[8];
            float attn_lo_b_1_0 = (float)(__nv_bfloat16)(acc_b[0] * inv_b);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_96 = approx_exp2((-gate_b[0]) * 1.442695f);
            float _rcp_95 = approx_rcp(1.0f + _exp2_96);
            result_b[0] = attn_lo_b_1_0 * _rcp_95;
#else
            float _exp2_63 = approx_exp2((-gate_b[0]) * 1.442695f);
            float _rcp_62 = approx_rcp(1.0f + _exp2_63);
            result_b[0] = attn_lo_b_1_0 * _rcp_62;
#endif
            float attn_hi_b_2_0 = (float)(__nv_bfloat16)(acc_b[4] * inv_b);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_98 = approx_exp2((-gate_b[4]) * 1.442695f);
            float _rcp_97 = approx_rcp(1.0f + _exp2_98);
            result_b[4] = attn_hi_b_2_0 * _rcp_97;
#else
            float _exp2_65 = approx_exp2((-gate_b[4]) * 1.442695f);
            float _rcp_64 = approx_rcp(1.0f + _exp2_65);
            result_b[4] = attn_hi_b_2_0 * _rcp_64;
#endif
            float attn_lo_b_1_1 = (float)(__nv_bfloat16)(acc_b[1] * inv_b);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_100 = approx_exp2((-gate_b[1]) * 1.442695f);
            float _rcp_99 = approx_rcp(1.0f + _exp2_100);
            result_b[1] = attn_lo_b_1_1 * _rcp_99;
#else
            float _exp2_67 = approx_exp2((-gate_b[1]) * 1.442695f);
            float _rcp_66 = approx_rcp(1.0f + _exp2_67);
            result_b[1] = attn_lo_b_1_1 * _rcp_66;
#endif
            float attn_hi_b_2_1 = (float)(__nv_bfloat16)(acc_b[5] * inv_b);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_102 = approx_exp2((-gate_b[5]) * 1.442695f);
            float _rcp_101 = approx_rcp(1.0f + _exp2_102);
            result_b[5] = attn_hi_b_2_1 * _rcp_101;
#else
            float _exp2_69 = approx_exp2((-gate_b[5]) * 1.442695f);
            float _rcp_68 = approx_rcp(1.0f + _exp2_69);
            result_b[5] = attn_hi_b_2_1 * _rcp_68;
#endif
            float attn_lo_b_1_2 = (float)(__nv_bfloat16)(acc_b[2] * inv_b);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_104 = approx_exp2((-gate_b[2]) * 1.442695f);
            float _rcp_103 = approx_rcp(1.0f + _exp2_104);
            result_b[2] = attn_lo_b_1_2 * _rcp_103;
#else
            float _exp2_71 = approx_exp2((-gate_b[2]) * 1.442695f);
            float _rcp_70 = approx_rcp(1.0f + _exp2_71);
            result_b[2] = attn_lo_b_1_2 * _rcp_70;
#endif
            float attn_hi_b_2_2 = (float)(__nv_bfloat16)(acc_b[6] * inv_b);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_106 = approx_exp2((-gate_b[6]) * 1.442695f);
            float _rcp_105 = approx_rcp(1.0f + _exp2_106);
            result_b[6] = attn_hi_b_2_2 * _rcp_105;
#else
            float _exp2_73 = approx_exp2((-gate_b[6]) * 1.442695f);
            float _rcp_72 = approx_rcp(1.0f + _exp2_73);
            result_b[6] = attn_hi_b_2_2 * _rcp_72;
#endif
            float attn_lo_b_1_3 = (float)(__nv_bfloat16)(acc_b[3] * inv_b);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_108 = approx_exp2((-gate_b[3]) * 1.442695f);
            float _rcp_107 = approx_rcp(1.0f + _exp2_108);
            result_b[3] = attn_lo_b_1_3 * _rcp_107;
#else
            float _exp2_75 = approx_exp2((-gate_b[3]) * 1.442695f);
            float _rcp_74 = approx_rcp(1.0f + _exp2_75);
            result_b[3] = attn_lo_b_1_3 * _rcp_74;
#endif
            float attn_hi_b_2_3 = (float)(__nv_bfloat16)(acc_b[7] * inv_b);
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            float _exp2_110 = approx_exp2((-gate_b[7]) * 1.442695f);
            float _rcp_109 = approx_rcp(1.0f + _exp2_110);
            result_b[7] = attn_hi_b_2_3 * _rcp_109;
#else
            float _exp2_77 = approx_exp2((-gate_b[7]) * 1.442695f);
            float _rcp_76 = approx_rcp(1.0f + _exp2_77);
            result_b[7] = attn_hi_b_2_3 * _rcp_76;
#endif
            unsigned int words_b[4];
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070
            __nv_bfloat162 _bit_cast_src_111 = __float22bfloat162_rn(make_float2(result_b[0], result_b[1]));
            words_b[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_111)[0];
            __nv_bfloat162 _bit_cast_src_112 = __float22bfloat162_rn(make_float2(result_b[2], result_b[3]));
            words_b[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_112)[0];
            __nv_bfloat162 _bit_cast_src_113 = __float22bfloat162_rn(make_float2(result_b[4], result_b[5]));
            words_b[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_113)[0];
            __nv_bfloat162 _bit_cast_src_114 = __float22bfloat162_rn(make_float2(result_b[6], result_b[7]));
            words_b[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_114)[0];
#else
            __nv_bfloat162 _bit_cast_src_78 = __float22bfloat162_rn(make_float2(result_b[0], result_b[1]));
            words_b[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_78)[0];
            __nv_bfloat162 _bit_cast_src_79 = __float22bfloat162_rn(make_float2(result_b[2], result_b[3]));
            words_b[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_79)[0];
            __nv_bfloat162 _bit_cast_src_80 = __float22bfloat162_rn(make_float2(result_b[4], result_b[5]));
            words_b[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_80)[0];
            __nv_bfloat162 _bit_cast_src_81 = __float22bfloat162_rn(make_float2(result_b[6], result_b[7]));
            words_b[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_81)[0];
#endif
            long long out_offset_b = (long long)row * out_stride_row + (long long)head_b * out_stride_head + dim;
            reinterpret_cast<int2*>(out + out_offset_b)[0] = reinterpret_cast<int2*>(words_b)[0];
            reinterpret_cast<int2*>(out + (out_offset_b + 4))[0] = reinterpret_cast<int2*>(words_b + (2))[0];
        }
    }
}

} // extern "C"
