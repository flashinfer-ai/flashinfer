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
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_lm_head_det_loss_07366b930f44e3768b7e(__nv_bfloat16* __restrict__ z, float* __restrict__ tau, float* __restrict__ lse, int* __restrict__ offsets, int* __restrict__ lpos, float* __restrict__ teacher, float* __restrict__ g, float* __restrict__ d, float* __restrict__ e, __nv_bfloat16* __restrict__ dz, int T, int M, int upstreams)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int lane_1 = lane;
    bool has_g = (upstreams & 1) != 0;
    bool has_d = (upstreams & 2) != 0;
    bool has_e = (upstreams & 4) != 0;
    int row_stride = gridDim.x * 8;
    for (int t = bid * 8 + warp; t < T; t += row_stride) {
        int r_lo = offsets[t];
        int r_hi = offsets[t + 1];
        int lo = ((r_lo > 0) ? r_lo : 0);
        lo = ((lo < M) ? lo : M);
        int hi = ((r_hi > lo) ? r_hi : lo);
        hi = ((hi < M) ? hi : M);
        int k = hi - lo;
        float tau_t = tau[t];
        float lse_t = lse[t];
        int lp = -1;
        float d_t = 0.0f;
        if (has_d) {
            lp = lpos[t];
            d_t = d[t];
        }
        float neg_e = 0.0f;
        if (has_e) {
            float _neg_0;
            asm("neg.f32 %0, %1;" : "=f"(_neg_0) : "f"(e[t]));
            neg_e = _neg_0;
        }
        bool active = k >= 2 && (has_g || has_e || has_d && lp >= 0);
        float u = 0.0f;
        float usum = 0.0f;
        for (int iu = lane_1; iu < k; iu += 32) {
            u = 0.0f;
            if (has_g) {
                u = g[lo + iu];
            }
            if (has_d) {
                float _add_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(u), "f"(d_t));
                u = ((lo + iu == lp) ? _add_1 : u);
            }
            if (has_e) {
                float _exp_2 = expf(teacher[lo + iu]);
                u = __fmaf_rn(neg_e, _exp_2, u);
            }
            float _add_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(usum), "f"(u));
            usum = _add_3;
        }
        float u_row = usum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            u_row += __shfl_xor_sync(0xFFFFFFFF, u_row, offset);
        for (int iz = lane_1; iz < k; iz += 32) {
            int jz = lo + iz;
            float out = 0.0f;
            if (active) {
                u = 0.0f;
                if (has_g) {
                    u = g[jz];
                }
                if (has_d) {
                    float _add_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(u), "f"(d_t));
                    u = ((jz == lp) ? _add_4 : u);
                }
                if (has_e) {
                    float _exp_5 = expf(teacher[jz]);
                    u = __fmaf_rn(neg_e, _exp_5, u);
                }
                float _fdiv_7 = __fdiv_rn((float)z[jz], tau_t);
                float _sub_6;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(_fdiv_7), "f"(lse_t));
                float p = expf(_sub_6);
                float _neg_9;
                asm("neg.f32 %0, %1;" : "=f"(_neg_9) : "f"(p));
                float _fdiv_8 = __fdiv_rn(__fmaf_rn(_neg_9, u_row, u), tau_t);
                out = _fdiv_8;
            }
            dz[jz] = (__nv_bfloat16)out;
        }
    }
}

} // extern "C"
