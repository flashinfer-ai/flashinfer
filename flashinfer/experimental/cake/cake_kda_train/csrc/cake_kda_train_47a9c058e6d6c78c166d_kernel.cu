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
#define SMEM_TOTAL 4096
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_47a9c058e6d6c78c166d(float* __restrict__ dg_in, float* __restrict__ db_in, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float* __restrict__ dg_out, __nv_bfloat16* __restrict__ dbeta_out, int num_heads)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* totals = reinterpret_cast<float*>(smem_raw + 0);
    const int totals_addr = smem + 0;
    int bx_s = blockIdx.x;
    int by_s = blockIdx.y;
    int nbx_s = (int)gridDim.x;
    int lin_s = by_s * nbx_s + bx_s;
    int chunk_s = lin_s / num_heads;
    int clen_setup = chunk_len[chunk_s];
    if (__syncthreads_and(clen_setup == 0)) return;

    // === Task calls (declaration order) ===
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int nbx = (int)gridDim.x;
    int lin = by * nbx + bx;
    int chunk = lin / num_heads;
    int head = lin - chunk * num_heads;
    int tgrp = tid / 32;
    int quad = tid - tgrp * 32;
    int d0 = quad * 4;
    long long row0 = (long long)chunk * 64;
    long long ext0 = (long long)chunk_bos[chunk];
    int clen = chunk_len[chunk];
    int rs = num_heads * 128;
    int t0 = tgrp * 8;
    long long ibase = ((row0 + (long long)t0) * (long long)num_heads + (long long)head) * 128 + (long long)d0;
    long long ebase = ((ext0 + (long long)t0) * (long long)num_heads + (long long)head) * 128 + (long long)d0;
    int sq = d0 * 4;
    float dgs[32];
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        {
            float4 _v4 = *reinterpret_cast<const float4*>(dg_in + ibase + (long long)(rs * u));
            dgs[u * 4 + 0] = _v4.x;
            dgs[u * 4 + 1] = _v4.y;
            dgs[u * 4 + 2] = _v4.z;
            dgs[u * 4 + 3] = _v4.w;
        }
    }
    int tok_b = tid - 32;
    if (tok_b >= 0) {
        if (tok_b < clen) {
            long long ebeta = (ext0 + (long long)tok_b) * (long long)num_heads + (long long)head;
            long long ibeta = (row0 + (long long)tok_b) * (long long)num_heads + (long long)head;
            dbeta_out[ebeta] = (__nv_bfloat16)db_in[ibeta];
        }
    }
    float suffix[4];
    suffix[0] = 0.0f;
    suffix[1] = 0.0f;
    suffix[2] = 0.0f;
    suffix[3] = 0.0f;
    #pragma unroll
    for (int uu = 0; uu < 8; uu++) {
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            suffix[e] = suffix[e] + dgs[(7 - uu) * 4 + e];
            dgs[(7 - uu) * 4 + e] = suffix[e];
        }
    }
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(totals_addr + (unsigned int)(tgrp * 512) + (unsigned int)sq), "f"(suffix[0]), "f"(suffix[1]), "f"(suffix[2]), "f"(suffix[3]) : "memory");
    __syncthreads();
    float offset[4];
    offset[0] = 0.0f;
    offset[1] = 0.0f;
    offset[2] = 0.0f;
    offset[3] = 0.0f;
    #pragma unroll
    for (int gg = 0; gg < 8; gg++) {
        const int g = 7 - gg;
        float tg[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tg[0])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 3]))
            : "r"(totals_addr + (unsigned int)(g * 512) + (unsigned int)sq));
        if (g > tgrp) {
            #pragma unroll
            for (int e_1 = 0; e_1 < 4; e_1++) {
                offset[e_1] = offset[e_1] + tg[e_1];
            }
        }
    }
    #pragma unroll
    for (int u_1 = 0; u_1 < 8; u_1++) {
        #pragma unroll
        for (int e_2 = 0; e_2 < 4; e_2++) {
            dgs[u_1 * 4 + e_2] = offset[e_2] + dgs[u_1 * 4 + e_2];
        }
    }
    #pragma unroll
    for (int u_2 = 0; u_2 < 8; u_2++) {
        int t_u = t0 + u_2;
        if (t_u < clen) {
            {
                float4 _v4 = make_float4(dgs[u_2 * 4 + 0], dgs[u_2 * 4 + 1], dgs[u_2 * 4 + 2], dgs[u_2 * 4 + 3]);
                *reinterpret_cast<float4*>(dg_out + ebase + (long long)(rs * u_2)) = _v4;
            }
        }
    }
}

} // extern "C"
