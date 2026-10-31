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

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_lm_head_det_loss_728732e30943f09310a4(__nv_bfloat16* __restrict__ z, float* __restrict__ tau, int* __restrict__ offsets, int* __restrict__ lpos, float* __restrict__ teacher, float* __restrict__ lse, float* __restrict__ packed, float* __restrict__ selected, float* __restrict__ kl, int T, int M, int outputs)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int lane_1 = lane;
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
        float m = -CUDART_INF_F;
        for (int im = lane_1; im < k; im += 32) {
            float _fdiv_0 = __fdiv_rn((float)z[lo + im], tau_t);
            m = max_noftz(m, _fdiv_0);
        }
        float row_max = m;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            row_max = max_noftz(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, offset));
        float s = 0.0f;
        for (int i_s = lane_1; i_s < k; i_s += 32) {
            float _fdiv_4 = __fdiv_rn((float)z[lo + i_s], tau_t);
            float _sub_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(_fdiv_4), "f"(row_max));
            float _exp_2 = expf(_sub_3);
            float _add_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(s), "f"(_exp_2));
            s = _add_1;
        }
        float row_sum = s;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, offset);
        float _log_6 = logf(row_sum);
        float _add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(row_max), "f"(_log_6));
        float lse_t = ((k == 0) ? 0.0f : _add_5);
        bool multi = k >= 2;
        if (lane_1 == 0) {
            lse[t] = lse_t;
        }
        if ((outputs & 1) != 0) {
            for (int ip = lane_1; ip < k; ip += 32) {
                float _fdiv_7 = __fdiv_rn((float)z[lo + ip], tau_t);
                float logp_p;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(logp_p) : "f"(_fdiv_7), "f"(lse_t));
                packed[lo + ip] = ((multi) ? logp_p : 0.0f);
            }
        }
        if ((outputs & 2) != 0) {
            if (lane_1 == 0) {
                int lp = lpos[t];
                float sel = 0.0f;
                if (lp >= 0 && multi) {
                    float _fdiv_9 = __fdiv_rn((float)z[((lp < M) ? lp : M - 1)], tau_t);
                    float _sub_8;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(_fdiv_9), "f"(lse_t));
                    sel = _sub_8;
                }
                selected[t] = sel;
            }
        }
        if ((outputs & 4) != 0) {
            float acc = 0.0f;
            for (int ik = lane_1; ik < k; ik += 32) {
                float _fdiv_10 = __fdiv_rn((float)z[lo + ik], tau_t);
                float logp_k;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(logp_k) : "f"(_fdiv_10), "f"(lse_t));
                float lq = teacher[lo + ik];
                float q_t = expf(lq);
                float _sub_11;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(lq), "f"(logp_k));
                acc = ((q_t > 0.0f) ? __fmaf_rn(q_t, _sub_11, acc) : acc);
            }
            float kl_row = acc;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                kl_row += __shfl_xor_sync(0xFFFFFFFF, kl_row, offset);
            if (lane_1 == 0) {
                kl[t] = ((multi) ? kl_row : 0.0f);
            }
        }
    }
}

} // extern "C"
