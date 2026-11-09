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

#include <math_constants.h>

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_lm_head_loss_aee298db0b852a88517a(float* __restrict__ PU, float* __restrict__ PMS, __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ Z, long long* __restrict__ labels, float* __restrict__ DXA, float* __restrict__ ROWS, float* __restrict__ LOGP, float* __restrict__ LOSS, long long* __restrict__ LAB, int parts, int T, int V, int H, int with_u, float loss_div)
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

    // Prologue
    float* ebuf = reinterpret_cast<float*>(smem_raw + 0);
    const int ebuf_addr = smem + 0;
    float* stat = reinterpret_cast<float*>(smem_raw + 4096);
    const int stat_addr = smem + 4096;

    // === Task calls (declaration order) ===
    float d = __fdiv_rn(-1.0f, loss_div);
    if (warp < T) {
        int tw = warp;
        float mx = -CUDART_INF_F;
        #pragma unroll 1
        for (int g = lane; g < parts; g += 32) {
            mx = max_noftz(mx, PMS[(g * T + tw) * 2]);
        }
        float _warp_reduce_0 = mx;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
        mx = _warp_reduce_0;
        float part = 0.0f;
        #pragma unroll 1
        for (int g_1 = lane; g_1 < parts; g_1 += 32) {
            float mg = PMS[(g_1 * T + tw) * 2];
            float e = 0.0f;
            if (mg != -CUDART_INF_F) {
                float _exp_1 = expf(mg - mx);
                e = _exp_1;
            }
            ebuf[tw * 256 + g_1] = e;
            part = __fmaf_rn(PMS[(g_1 * T + tw) * 2 + 1], e, part);
        }
        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, part, 16);
        part = part + _shfl_xor_sync_2;
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, part, 8);
        part = part + _shfl_xor_sync_3;
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, part, 4);
        part = part + _shfl_xor_sync_4;
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, part, 2);
        part = part + _shfl_xor_sync_5;
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, part, 1);
        part = part + _shfl_xor_sync_6;
        if (lane == 0) {
            stat[2 * tw] = mx;
            stat[2 * tw + 1] = part;
        }
    }
    __syncthreads();
    if (with_u != 0) {
        int h = blockIdx.x * 256 + tid;
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            if (t < T) {
                long long y = labels[t];
                float out = 0.0f;
                if (y >= 0ll) {
                    float acc = 0.0f;
                    #pragma unroll 4
                    for (int g_2 = 0; g_2 < parts; g_2++) {
                        acc = __fmaf_rn(PU[(unsigned long long)(g_2 * T + t) * (unsigned long long)H + (unsigned long long)h], ebuf[t * 256 + g_2], acc);
                    }
                    float wy = 0.0f;
                    if (y < (long long)V) {
                        wy = (float)W[(unsigned long long)y * (unsigned long long)H + (unsigned long long)h];
                    }
                    float _fdiv_7 = __fdiv_rn(acc, stat[2 * t + 1]);
                    out = d * (wy - _fdiv_7);
                }
                DXA[(unsigned long long)t * (unsigned long long)H + (unsigned long long)h] = out;
            }
        }
    }
    if (blockIdx.x == 0) {
        if (tid == 0) {
            double terms[4];
            #pragma unroll
            for (int t_1 = 0; t_1 < 4; t_1++) {
                terms[t_1] = (double)0.0f;
                if (t_1 < T) {
                    long long yl = labels[t_1];
                    float _log_8 = logf(stat[2 * t_1 + 1]);
                    float lse = stat[2 * t_1] + _log_8;
                    float logp = 0.0f;
                    float dt = 0.0f;
                    if (yl >= 0ll) {
                        long long yc = yl;
                        if (yc >= (long long)V) {
                            yc = (long long)(V - 1);
                        }
                        logp = (float)Z[(unsigned long long)t_1 * (unsigned long long)V + (unsigned long long)yc] - lse;
                        dt = d;
                    }
                    ROWS[2 * t_1] = lse;
                    ROWS[2 * t_1 + 1] = dt;
                    LOGP[t_1] = logp;
                    LAB[t_1] = yl;
                    terms[t_1] = (double)logp;
                }
            }
            double total = terms[0] + terms[2] + (terms[1] + terms[3]);
            LOSS[0] = (float)(((double)0.0f - total) / (double)loss_div);
        }
    }
}

} // extern "C"
