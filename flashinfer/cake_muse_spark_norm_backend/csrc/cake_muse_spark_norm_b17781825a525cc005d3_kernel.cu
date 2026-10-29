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

extern "C" {

__global__ __launch_bounds__(128) __cluster_dims__(8,1,1) void
kernel_cake_muse_spark_norm_b17781825a525cc005d3(__nv_bfloat16* x, const __nv_bfloat16* __restrict__ gate, int tokens, int heads, long long x_stride, long long gate_stride, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 8;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 8;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // === Task calls (declaration order) ===
    int group = tid >> 3;
    int lane_1 = tid & 7;
    int token_raw = blockIdx.x;
    int token_id = ((token_raw) < (tokens - 1) ? (token_raw) : (tokens - 1));
    int block_id = blockIdx.y;
    int head0 = block_id * 16 + group;
    int last_head = heads - 1;
    long long token = (long long)token_id;
    long long xbase = token * x_stride + (long long)(lane_1 * 8);
    long long gbase = token * gate_stride + (long long)(lane_1 * 8);
    float xs[8];
    float gs[8];
    #pragma unroll
    for (int r = 0; r < 1; r++) {
        int head = head0 + r * 16;
        long long hoff = (long long)(head * 64);
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (xbase + hoff) + 0);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&xs[r * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&xs[r * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (gbase + hoff) + 0);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&gs[r * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&gs[r * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
    }
    #pragma unroll
    for (int r_1 = 0; r_1 < 1; r_1++) {
        int row_head = head0 + r_1 * 16;
        int base = r_1 * 8;
        float sq[4];
        sq[0] = __fmaf_rn(xs[base + 1], xs[base + 1], xs[base] * xs[base]);
        sq[1] = __fmaf_rn(xs[base + 2 + 1], xs[base + 2 + 1], xs[base + 2] * xs[base + 2]);
        sq[2] = __fmaf_rn(xs[base + 4 + 1], xs[base + 4 + 1], xs[base + 4] * xs[base + 4]);
        sq[3] = __fmaf_rn(xs[base + 6 + 1], xs[base + 6 + 1], xs[base + 6] * xs[base + 6]);
        sq[0] = sq[0] + sq[1];
        sq[2] = sq[2] + sq[3];
        sq[0] = sq[0] + sq[2];
        float total = sq[0];
        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
        total = total + _shfl_xor_sync_2;
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
        total = total + _shfl_xor_sync_3;
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
        total = total + _shfl_xor_sync_4;
        float inv;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(__fmaf_rn(total, 0.015625f, eps)));
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float _exp2_5 = approx_exp2(gs[base + j] * -1.442695f);
            float sig = approx_rcp(_exp2_5 + 1.0f);
            xs[base + j] = xs[base + j] * inv * sig;
        }
        long long row_off = (long long)(row_head * 64);
        if (token_raw < tokens) {
            unsigned int _packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((xs + base)[_lp*2 + 0], (xs + base)[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((xs + base)[_lp*2 + 2], (xs + base)[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((xs + base)[_lp*2 + 4], (xs + base)[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((xs + base)[_lp*2 + 6], (xs + base)[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(x + (xbase + row_off))[0] = reinterpret_cast<int4*>(_packed_0)[0];
        }
    }
}

} // extern "C"
