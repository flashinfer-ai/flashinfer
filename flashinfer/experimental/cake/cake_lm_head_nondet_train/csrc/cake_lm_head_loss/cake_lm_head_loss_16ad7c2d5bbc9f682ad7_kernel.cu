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

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_lm_head_loss_16ad7c2d5bbc9f682ad7(__nv_bfloat16* __restrict__ Z, float* __restrict__ ROWS, long long* __restrict__ LAB, __nv_bfloat16* __restrict__ X, float* __restrict__ DXA, float* __restrict__ grad, __nv_bfloat16* __restrict__ DW, __nv_bfloat16* __restrict__ DX, int V, int T, int ldx, int dw_blocks)
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
    float* dzs = reinterpret_cast<float*>(smem_raw + 0);
    const int dzs_addr = smem + 0;

    // === Task calls (declaration order) ===
    float g = grad[0];
    if (bid < dw_blocks) {
        int v0 = bid * 32;
        if (tid < 128) {
            int r = tid / 4;
            int t = tid % 4;
            float dz = 0.0f;
            if (t < T) {
                if (v0 + r < V) {
                    long long y = LAB[t];
                    if (y >= 0ll) {
                        float lse = ROWS[2 * t];
                        float dt = ROWS[2 * t + 1];
                        float zf = (float)Z[(unsigned long long)t * (unsigned long long)V + (unsigned long long)(v0 + r)];
                        float p = expf(zf - lse);
                        float hot = 0.0f;
                        if ((long long)(v0 + r) == y) {
                            hot = 1.0f;
                        }
                        dz = (float)(__nv_bfloat16)(dt * (hot - p));
                    }
                }
            }
            dzs[tid] = dz;
        }
        __syncthreads();
        int live[4];
        float xf[96];
        #pragma unroll
        for (int t_1 = 0; t_1 < 4; t_1++) {
            live[t_1] = 0;
            if (t_1 < T) {
                if (LAB[t_1] >= 0ll) {
                    live[t_1] = 1;
                }
            }
            if (live[t_1] != 0) {
                #pragma unroll
                for (int j = 0; j < 3; j++) {
                    unsigned int xw[4];
                    {
                        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(X + ((unsigned long long)t_1 * (unsigned long long)ldx + (unsigned long long)(tid + 256 * j) * 8ull) + 0);
                        xw[0 + 0] = _uv4_0.x;
                        xw[0 + 1] = _uv4_0.y;
                        xw[0 + 2] = _uv4_0.z;
                        xw[0 + 3] = _uv4_0.w;
                    }
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        xf[(t_1 * 3 + j) * 8 + 2 * e] = __uint_as_float(xw[e] << 16);
                        xf[(t_1 * 3 + j) * 8 + 2 * e + 1] = __uint_as_float(xw[e] & 4294901760u);
                    }
                }
            }
        }
        #pragma unroll 4
        for (int r_1 = 0; r_1 < 32; r_1++) {
            if (v0 + r_1 < V) {
                #pragma unroll
                for (int j_1 = 0; j_1 < 3; j_1++) {
                    float o[8];
                    #pragma unroll
                    for (int e_1 = 0; e_1 < 8; e_1++) {
                        o[e_1] = 0.0f;
                    }
                    #pragma unroll
                    for (int t_2 = 0; t_2 < 4; t_2++) {
                        if (live[t_2] != 0) {
                            float dzt = dzs[r_1 * 4 + t_2];
                            #pragma unroll
                            for (int e_2 = 0; e_2 < 8; e_2++) {
                                if (t_2 == 0) {
                                    o[e_2] = dzt * xf[(t_2 * 3 + j_1) * 8 + e_2];
                                } else {
                                    o[e_2] = __fmaf_rn(dzt, xf[(t_2 * 3 + j_1) * 8 + e_2], o[e_2]);
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int e_3 = 0; e_3 < 8; e_3++) {
                        o[e_3] = o[e_3] * g;
                    }
                    unsigned int packed[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o[_lp*2 + 0], o[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o[_lp*2 + 2], o[_lp*2+1 + 2]));
                        packed[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o[_lp*2 + 4], o[_lp*2+1 + 4]));
                        packed[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o[_lp*2 + 6], o[_lp*2+1 + 6]));
                        packed[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(DW + ((unsigned long long)(v0 + r_1) * 6144ull + (unsigned long long)(tid + 256 * j_1) * 8ull))[0] = reinterpret_cast<int4*>(packed)[0];
                }
            }
        }
    } else {
        int i = (bid - dw_blocks) * 256 + tid;
        if (i < T * 768) {
            float a0[4];
            float a1[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(DXA + (unsigned long long)i * 8ull + 0);
                a0[0 + 0] = _v4.x;
                a0[0 + 1] = _v4.y;
                a0[0 + 2] = _v4.z;
                a0[0 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(DXA + ((unsigned long long)i * 8ull + 4ull) + 0);
                a1[0 + 0] = _v4.x;
                a1[0 + 1] = _v4.y;
                a1[0 + 2] = _v4.z;
                a1[0 + 3] = _v4.w;
            }
            float o_1[8];
            #pragma unroll
            for (int e_4 = 0; e_4 < 4; e_4++) {
                o_1[e_4] = a0[e_4] * g;
                o_1[e_4 + 4] = a1[e_4] * g;
            }
            unsigned int packed_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_1[_lp*2 + 0], o_1[_lp*2+1 + 0]));
                packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_1[_lp*2 + 2], o_1[_lp*2+1 + 2]));
                packed_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_1[_lp*2 + 4], o_1[_lp*2+1 + 4]));
                packed_1[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_1[_lp*2 + 6], o_1[_lp*2+1 + 6]));
                packed_1[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(DX + (unsigned long long)i * 8ull)[0] = reinterpret_cast<int4*>(packed_1)[0];
        }
    }
}

} // extern "C"
