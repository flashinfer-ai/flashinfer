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
#define THREADS 64

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


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

extern "C" {

__global__ __launch_bounds__(64) void
kernel_cake_muse_spark_norm_a44c84445d2eae1b90e1(__nv_bfloat16* x, const __nv_bfloat16* __restrict__ gate, int tokens, int heads, long long x_stride, long long gate_stride, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int group = tid >> 3;
    int lane_1 = tid & 7;
    int token_id = blockIdx.x;
    int block_id = blockIdx.y;
    int head0 = block_id * 32 + group;
    int last_head = heads - 1;
    long long token = (long long)token_id;
    long long xbase = token * x_stride + (long long)(lane_1 * 8);
    long long gbase = token * gate_stride + (long long)(lane_1 * 8);
    float2 neg_pair = make_float2(-1.442695f, -1.442695f);
    float2 one_pair = make_float2(1.0f, 1.0f);
    float2 half_pair = make_float2(0.5f, 0.5f);
    float xs[32];
    float gs[32];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int head = head0 + r * 8;
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
    for (int r_1 = 0; r_1 < 4; r_1++) {
        int row_head = head0 + r_1 * 8;
        int base = r_1 * 8;
        float sq[4];
        float2 xp0_1_0 = make_float2(xs[base], xs[base + 1]);
        float2 sqp0_1_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sqp0_1_0) : "l"(*(const unsigned long long*)&xp0_1_0), "l"(*(const unsigned long long*)&xp0_1_0));
        sq[0] = sqp0_1_0.x + sqp0_1_0.y;
        float2 xp0_1_1 = make_float2(xs[base + 2], xs[base + 2 + 1]);
        float2 sqp0_1_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sqp0_1_1) : "l"(*(const unsigned long long*)&xp0_1_1), "l"(*(const unsigned long long*)&xp0_1_1));
        sq[1] = sqp0_1_1.x + sqp0_1_1.y;
        float2 xp0_1_2 = make_float2(xs[base + 4], xs[base + 4 + 1]);
        float2 sqp0_1_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sqp0_1_2) : "l"(*(const unsigned long long*)&xp0_1_2), "l"(*(const unsigned long long*)&xp0_1_2));
        sq[2] = sqp0_1_2.x + sqp0_1_2.y;
        float2 xp0_1_3 = make_float2(xs[base + 6], xs[base + 6 + 1]);
        float2 sqp0_1_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sqp0_1_3) : "l"(*(const unsigned long long*)&xp0_1_3), "l"(*(const unsigned long long*)&xp0_1_3));
        sq[3] = sqp0_1_3.x + sqp0_1_3.y;
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
        float2 inv_pair = make_float2(inv, inv);
        float2 gp0_1_0 = make_float2(gs[base], gs[base + 1]);
        float2 gl0_1_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gl0_1_0) : "l"(*(const unsigned long long*)&gp0_1_0), "l"(*(const unsigned long long*)&neg_pair));
        float _exp2_5 = approx_exp2(gl0_1_0.x);
        float _exp2_6 = approx_exp2(gl0_1_0.y);
        float2 ex0_1_0 = make_float2(_exp2_5, _exp2_6);
        float2 den0_1_0 = add_f32x2_noftz(ex0_1_0, one_pair);
        float _rcp_7 = approx_rcp(den0_1_0.x);
        float _rcp_8 = approx_rcp(den0_1_0.y);
        float2 sg0_1_0 = make_float2(_rcp_7, _rcp_8);
        float2 xq0_1_0 = make_float2(xs[base], xs[base + 1]);
        float2 xi0_1_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xi0_1_0) : "l"(*(const unsigned long long*)&xq0_1_0), "l"(*(const unsigned long long*)&inv_pair));
        float2 yp0_1_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&yp0_1_0) : "l"(*(const unsigned long long*)&xi0_1_0), "l"(*(const unsigned long long*)&sg0_1_0));
        xs[base] = yp0_1_0.x;
        xs[base + 1] = yp0_1_0.y;
        float2 gp0_1_1 = make_float2(gs[base + 2], gs[base + 2 + 1]);
        float2 gl0_1_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gl0_1_1) : "l"(*(const unsigned long long*)&gp0_1_1), "l"(*(const unsigned long long*)&neg_pair));
        float _exp2_9 = approx_exp2(gl0_1_1.x);
        float _exp2_10 = approx_exp2(gl0_1_1.y);
        float2 ex0_1_1 = make_float2(_exp2_9, _exp2_10);
        float2 den0_1_1 = add_f32x2_noftz(ex0_1_1, one_pair);
        float _rcp_11 = approx_rcp(den0_1_1.x);
        float _rcp_12 = approx_rcp(den0_1_1.y);
        float2 sg0_1_1 = make_float2(_rcp_11, _rcp_12);
        float2 xq0_1_1 = make_float2(xs[base + 2], xs[base + 2 + 1]);
        float2 xi0_1_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xi0_1_1) : "l"(*(const unsigned long long*)&xq0_1_1), "l"(*(const unsigned long long*)&inv_pair));
        float2 yp0_1_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&yp0_1_1) : "l"(*(const unsigned long long*)&xi0_1_1), "l"(*(const unsigned long long*)&sg0_1_1));
        xs[base + 2] = yp0_1_1.x;
        xs[base + 2 + 1] = yp0_1_1.y;
        float2 gp0_1_2 = make_float2(gs[base + 4], gs[base + 4 + 1]);
        float2 gl0_1_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gl0_1_2) : "l"(*(const unsigned long long*)&gp0_1_2), "l"(*(const unsigned long long*)&neg_pair));
        float _exp2_13 = approx_exp2(gl0_1_2.x);
        float _exp2_14 = approx_exp2(gl0_1_2.y);
        float2 ex0_1_2 = make_float2(_exp2_13, _exp2_14);
        float2 den0_1_2 = add_f32x2_noftz(ex0_1_2, one_pair);
        float _rcp_15 = approx_rcp(den0_1_2.x);
        float _rcp_16 = approx_rcp(den0_1_2.y);
        float2 sg0_1_2 = make_float2(_rcp_15, _rcp_16);
        float2 xq0_1_2 = make_float2(xs[base + 4], xs[base + 4 + 1]);
        float2 xi0_1_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xi0_1_2) : "l"(*(const unsigned long long*)&xq0_1_2), "l"(*(const unsigned long long*)&inv_pair));
        float2 yp0_1_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&yp0_1_2) : "l"(*(const unsigned long long*)&xi0_1_2), "l"(*(const unsigned long long*)&sg0_1_2));
        xs[base + 4] = yp0_1_2.x;
        xs[base + 4 + 1] = yp0_1_2.y;
        float2 gp0_1_3 = make_float2(gs[base + 6], gs[base + 6 + 1]);
        float2 gl0_1_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gl0_1_3) : "l"(*(const unsigned long long*)&gp0_1_3), "l"(*(const unsigned long long*)&neg_pair));
        float _exp2_17 = approx_exp2(gl0_1_3.x);
        float _exp2_18 = approx_exp2(gl0_1_3.y);
        float2 ex0_1_3 = make_float2(_exp2_17, _exp2_18);
        float2 den0_1_3 = add_f32x2_noftz(ex0_1_3, one_pair);
        float _rcp_19 = approx_rcp(den0_1_3.x);
        float _rcp_20 = approx_rcp(den0_1_3.y);
        float2 sg0_1_3 = make_float2(_rcp_19, _rcp_20);
        float2 xq0_1_3 = make_float2(xs[base + 6], xs[base + 6 + 1]);
        float2 xi0_1_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xi0_1_3) : "l"(*(const unsigned long long*)&xq0_1_3), "l"(*(const unsigned long long*)&inv_pair));
        float2 yp0_1_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&yp0_1_3) : "l"(*(const unsigned long long*)&xi0_1_3), "l"(*(const unsigned long long*)&sg0_1_3));
        xs[base + 6] = yp0_1_3.x;
        xs[base + 6 + 1] = yp0_1_3.y;
        long long row_off = (long long)(row_head * 64);
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

} // extern "C"
