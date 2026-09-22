/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_TOTALS_OFF 0
#define SMEM_TOTALS_STAGE_BYTES 4096
#define SMEM_TOTALS_STRIDE 4096
#define SMEM_PART_A_S_OFF 4096
#define SMEM_PART_A_S_STAGE_BYTES 4096
#define SMEM_PART_A_S_STRIDE 4096
#define SMEM_PART_B_S_OFF 8192
#define SMEM_PART_B_S_STAGE_BYTES 4096
#define SMEM_PART_B_S_STRIDE 4096
#define SMEM_TOTAL 12288
#define THREADS 1024

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_cake_kda_chunk_train_0ad4626786c206c7f6d6(float* __restrict__ dg_intra, __nv_bfloat16* __restrict__ g_raw, float* __restrict__ db_total, __nv_bfloat16* __restrict__ beta_raw, float* __restrict__ A_log, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ dg_out, __nv_bfloat16* __restrict__ dbeta, float* __restrict__ dA_part, float* __restrict__ dbias_part, int num_heads, float lower_bound)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* totals = reinterpret_cast<float*>(smem_raw + 0);
    const int totals_addr = smem + 0;
    float* part_a_s = reinterpret_cast<float*>(smem_raw + 4096);
    const int part_a_s_addr = smem + 4096;
    float* part_b_s = reinterpret_cast<float*>(smem_raw + 8192);
    const int part_b_s_addr = smem + 8192;

    // === Task calls (dependency order) ===
    int chunk = blockIdx.x;
    int head = blockIdx.y;
    int dim = tid % 128;
    int tgrp = tid / 128;
    long long row0 = (long long)chunk * 64;
    float _expf_0 = __expf(A_log[head]);
    float rate = _expf_0;
    float bias = dt_bias[head * 128 + dim];
    float dgs[8];
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        long long idx_u = ((row0 + (long long)(tgrp * 8 + u)) * (long long)num_heads + (long long)head) * 128 + (long long)dim;
        dgs[u] = dg_intra[idx_u];
    }
    float suffix = 0.0f;
    #pragma unroll
    for (int uu = 0; uu < 8; uu++) {
        suffix = suffix + dgs[7 - uu];
        dgs[7 - uu] = suffix;
    }
    totals[tgrp * 128 + dim] = suffix;
    __syncthreads();
    float offset = 0.0f;
    #pragma unroll
    for (int gg = 0; gg < 8; gg++) {
        const int g = 7 - gg;
        float tg = totals[g * 128 + dim];
        if (g > tgrp) {
            offset = offset + tg;
        }
    }
    float part_a = 0.0f;
    float part_b = 0.0f;
    #pragma unroll
    for (int u_1 = 0; u_1 < 8; u_1++) {
        long long idx_u_1 = ((row0 + (long long)(tgrp * 8 + u_1)) * (long long)num_heads + (long long)head) * 128 + (long long)dim;
        float dyg = offset + dgs[u_1];
        float x = (float)g_raw[idx_u_1] + bias;
        float _expf_1 = __expf(-(rate * x));
        float _rcp_0 = approx_rcp(1.0f + _expf_1);
        float sig = _rcp_0;
        float dsig = sig * (1.0f - sig);
        float dg_raw = dyg * (lower_bound * dsig) * rate;
        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(dg_raw);
        __nv_bfloat16 dgb = _cvt_bf16_0;
        dg_out[idx_u_1] = dgb;
        float _fma_0 = __fmaf_rn(dg_raw, x, part_a);
        part_a = _fma_0;
        float _cvt_f32_0 = __bfloat162float(dgb);
        part_b = part_b + _cvt_f32_0;
    }
    part_b_s[tgrp * 128 + dim] = part_b;
    part_a_s[tgrp * 128 + dim] = part_a;
    __syncthreads();
    if (tgrp == 0) {
        float sum_a = 0.0f;
        float sum_b = 0.0f;
        #pragma unroll
        for (int g_1 = 0; g_1 < 8; g_1++) {
            sum_a = sum_a + part_a_s[g_1 * 128 + dim];
            sum_b = sum_b + part_b_s[g_1 * 128 + dim];
        }
        long long part_index = ((long long)chunk * (long long)num_heads + (long long)head) * 128 + (long long)dim;
        dA_part[part_index] = sum_a;
        dbias_part[part_index] = sum_b;
    }
    if (tgrp == 1) {
        if (dim < 64) {
            long long beta_index = (row0 + (long long)dim) * (long long)num_heads + (long long)head;
            float _expf_2 = __expf(-(float)beta_raw[beta_index]);
            float _rcp_1 = approx_rcp(1.0f + _expf_2);
            float sb = _rcp_1;
            __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(db_total[beta_index] * sb * (1.0f - sb));
            dbeta[beta_index] = _cvt_bf16_1;
        }
    }
}

} // extern "C"
