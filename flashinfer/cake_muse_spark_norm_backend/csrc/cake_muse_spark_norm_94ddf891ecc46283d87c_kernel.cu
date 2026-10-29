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

__global__ __launch_bounds__(128) void
kernel_cake_muse_spark_norm_94ddf891ecc46283d87c(__nv_bfloat16* x, const __nv_bfloat16* __restrict__ gate, int tokens, int heads, long long x_stride, long long gate_stride, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int group = tid >> 2;
    int lane_1 = tid & 3;
    int token_raw = blockIdx.x * 4;
    int token_id = ((token_raw) < (tokens - 1) ? (token_raw) : (tokens - 1));
    int block_id = blockIdx.y;
    int head0 = block_id * 64 + group;
    int last_head = heads - 1;
    long long token = (long long)token_id;
    long long xbase = token * x_stride + (long long)(lane_1 * 16);
    long long gbase = token * gate_stride + (long long)(lane_1 * 16);
    float xs[32];
    float gs[32];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        int fr = group + r * 32;
        int _min_0 = ((token_raw + (fr >> 4)) < (tokens - 1) ? (token_raw + (fr >> 4)) : (tokens - 1));
        long long tok = (long long)_min_0;
        long long moff = (long long)((fr & 15) * 64 + lane_1 * 16);
        {
            const void* _v8p_1 = (const void*)(x + (tok * x_stride + moff) + (0));
            uint32_t _v8_1_0[8];
            asm volatile(
                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_v8_1_0[0]), "=r"(_v8_1_0[1]), "=r"(_v8_1_0[2]), "=r"(_v8_1_0[3]), "=r"(_v8_1_0[4]), "=r"(_v8_1_0[5]), "=r"(_v8_1_0[6]), "=r"(_v8_1_0[7]) : "l"((const char*)_v8p_1 + 0) : "memory");
            (&xs[r * 16 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[0]) << 16);
            (&xs[r * 16 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[0]) & 0xffff0000u);
            (&xs[r * 16 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[1]) << 16);
            (&xs[r * 16 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[1]) & 0xffff0000u);
            (&xs[r * 16 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[2]) << 16);
            (&xs[r * 16 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[2]) & 0xffff0000u);
            (&xs[r * 16 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[3]) << 16);
            (&xs[r * 16 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[3]) & 0xffff0000u);
            (&xs[r * 16 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[4]) << 16);
            (&xs[r * 16 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[4]) & 0xffff0000u);
            (&xs[r * 16 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[5]) << 16);
            (&xs[r * 16 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[5]) & 0xffff0000u);
            (&xs[r * 16 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[6]) << 16);
            (&xs[r * 16 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[6]) & 0xffff0000u);
            (&xs[r * 16 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[7]) << 16);
            (&xs[r * 16 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[7]) & 0xffff0000u);
        }
        {
            const void* _v8p_2 = (const void*)(gate + (tok * gate_stride + moff) + (0));
            uint32_t _v8_2_0[8];
            asm volatile(
                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_v8_2_0[0]), "=r"(_v8_2_0[1]), "=r"(_v8_2_0[2]), "=r"(_v8_2_0[3]), "=r"(_v8_2_0[4]), "=r"(_v8_2_0[5]), "=r"(_v8_2_0[6]), "=r"(_v8_2_0[7]) : "l"((const char*)_v8p_2 + 0) : "memory");
            (&gs[r * 16 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[0]) << 16);
            (&gs[r * 16 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[0]) & 0xffff0000u);
            (&gs[r * 16 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[1]) << 16);
            (&gs[r * 16 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[1]) & 0xffff0000u);
            (&gs[r * 16 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[2]) << 16);
            (&gs[r * 16 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[2]) & 0xffff0000u);
            (&gs[r * 16 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[3]) << 16);
            (&gs[r * 16 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[3]) & 0xffff0000u);
            (&gs[r * 16 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[4]) << 16);
            (&gs[r * 16 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[4]) & 0xffff0000u);
            (&gs[r * 16 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[5]) << 16);
            (&gs[r * 16 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[5]) & 0xffff0000u);
            (&gs[r * 16 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[6]) << 16);
            (&gs[r * 16 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[6]) & 0xffff0000u);
            (&gs[r * 16 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[7]) << 16);
            (&gs[r * 16 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[7]) & 0xffff0000u);
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll
    for (int r_1 = 0; r_1 < 2; r_1++) {
        int row_head = head0 + r_1 * 32;
        int base = r_1 * 16;
        float sq[8];
        sq[0] = __fmaf_rn(xs[base + 1], xs[base + 1], xs[base] * xs[base]);
        sq[1] = __fmaf_rn(xs[base + 2 + 1], xs[base + 2 + 1], xs[base + 2] * xs[base + 2]);
        sq[2] = __fmaf_rn(xs[base + 4 + 1], xs[base + 4 + 1], xs[base + 4] * xs[base + 4]);
        sq[3] = __fmaf_rn(xs[base + 6 + 1], xs[base + 6 + 1], xs[base + 6] * xs[base + 6]);
        sq[4] = __fmaf_rn(xs[base + 8 + 1], xs[base + 8 + 1], xs[base + 8] * xs[base + 8]);
        sq[5] = __fmaf_rn(xs[base + 10 + 1], xs[base + 10 + 1], xs[base + 10] * xs[base + 10]);
        sq[6] = __fmaf_rn(xs[base + 12 + 1], xs[base + 12 + 1], xs[base + 12] * xs[base + 12]);
        sq[7] = __fmaf_rn(xs[base + 14 + 1], xs[base + 14 + 1], xs[base + 14] * xs[base + 14]);
        sq[0] = sq[0] + sq[1];
        sq[2] = sq[2] + sq[3];
        sq[4] = sq[4] + sq[5];
        sq[6] = sq[6] + sq[7];
        sq[0] = sq[0] + sq[2];
        sq[4] = sq[4] + sq[6];
        sq[0] = sq[0] + sq[4];
        float total = sq[0];
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
        total = total + _shfl_xor_sync_3;
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
        total = total + _shfl_xor_sync_4;
        float inv;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(__fmaf_rn(total, 0.015625f, eps)));
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            float _exp2_5 = approx_exp2(gs[base + j] * -1.442695f);
            float sig = approx_rcp(_exp2_5 + 1.0f);
            xs[base + j] = xs[base + j] * inv * sig;
        }
        long long row_off = (long long)(row_head * 64);
        int fr_1 = group + r_1 * 32;
        int tok_u = token_raw + (fr_1 >> 4);
        int _min_6 = ((tok_u) < (tokens - 1) ? (tok_u) : (tokens - 1));
        long long tok_1 = (long long)_min_6;
        long long moff_1 = (long long)((fr_1 & 15) * 64 + lane_1 * 16);
        if (tok_u < tokens) {
            unsigned int _packed_0[8];
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
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((xs + base)[_lp*2 + 8], (xs + base)[_lp*2+1 + 8]));
                _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((xs + base)[_lp*2 + 10], (xs + base)[_lp*2+1 + 10]));
                _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((xs + base)[_lp*2 + 12], (xs + base)[_lp*2+1 + 12]));
                _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((xs + base)[_lp*2 + 14], (xs + base)[_lp*2+1 + 14]));
                _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
            }
            {
                const unsigned* _raw_stv8_7 = reinterpret_cast<const unsigned*>(_packed_0);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(x + (tok_1 * x_stride + moff_1))), "r"(_raw_stv8_7[0]), "r"(_raw_stv8_7[1]), "r"(_raw_stv8_7[2]), "r"(_raw_stv8_7[3]), "r"(_raw_stv8_7[4]), "r"(_raw_stv8_7[5]), "r"(_raw_stv8_7[6]), "r"(_raw_stv8_7[7]) : "memory");
            }
        }
    }
}

} // extern "C"
