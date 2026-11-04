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
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    const unsigned int w = *reinterpret_cast<const unsigned int*>(&v);
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mhc_train_fn_split_atomic__tv128_dep__h5120(float* __restrict__ fn, __nv_bfloat16* __restrict__ fnTp, float* __restrict__ dfn, int accumulate)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int lane_1 = tid % 32;
    int warp_1 = tid / 32;
    int n = lane_1 % 4 * 8;
    int k0 = (bid * 8 + warp_1) * 8 + lane_1 / 4;
    float x[8];
    float p1[8];
    float p2[8];
    float p3[8];
    if (n < 24) {
        long long src = (long long)n * 20480 + (long long)k0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float _vec_load_0[1];
            {
                _vec_load_0[0] = *reinterpret_cast<const float*>(fn + (src + (long long)(j * 20480)) + 0);
            }
            float xv = _vec_load_0[0];
            x[j] = xv;
        }
        if (accumulate == 0) {
            #pragma unroll
            for (int j_1 = 0; j_1 < 8; j_1++) {
                *(reinterpret_cast<float*>(dfn + (src + (long long)(j_1 * 20480))) + (0)) = 0.0f;
            }
        }
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; j_2 += 2) {
            float2 f2 = make_float2(x[j_2], x[j_2 + 1]);
            float2 b1 = cake_bf16x2_to_f32x2(__float22bfloat162_rn(make_float2(x[j_2], x[j_2 + 1])));
            float2 r1 = sub_f32x2_noftz(f2, b1);
            float2 b2 = cake_bf16x2_to_f32x2(__float22bfloat162_rn(make_float2(r1.x, r1.y)));
            float2 r2 = sub_f32x2_noftz(r1, b2);
            float2 b3 = cake_bf16x2_to_f32x2(__float22bfloat162_rn(make_float2(r2.x, r2.y)));
            p1[j_2] = b1.x;
            p1[j_2 + 1] = b1.y;
            p2[j_2] = b2.x;
            p2[j_2 + 1] = b2.y;
            p3[j_2] = b3.x;
            p3[j_2 + 1] = b3.y;
        }
    } else {
        #pragma unroll
        for (int j_3 = 0; j_3 < 8; j_3++) {
            p1[j_3] = 0.0f;
            p2[j_3] = 0.0f;
            p3[j_3] = 0.0f;
        }
    }
    long long dst_0 = (long long)k0 * 32 + (long long)n;
    unsigned int _packed_0[4];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p1[_lp*2 + 0], p1[_lp*2+1 + 0]));
        _packed_0[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p1[_lp*2 + 2], p1[_lp*2+1 + 2]));
        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p1[_lp*2 + 4], p1[_lp*2+1 + 4]));
        _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p1[_lp*2 + 6], p1[_lp*2+1 + 6]));
        _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(fnTp + dst_0)[0] = reinterpret_cast<int4*>(_packed_0)[0];
    long long dst_1 = (20480 + (long long)k0) * 32 + (long long)n;
    unsigned int _packed_1[4];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p2[_lp*2 + 0], p2[_lp*2+1 + 0]));
        _packed_1[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p2[_lp*2 + 2], p2[_lp*2+1 + 2]));
        _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p2[_lp*2 + 4], p2[_lp*2+1 + 4]));
        _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p2[_lp*2 + 6], p2[_lp*2+1 + 6]));
        _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(fnTp + dst_1)[0] = reinterpret_cast<int4*>(_packed_1)[0];
    long long dst_2 = (40960 + (long long)k0) * 32 + (long long)n;
    unsigned int _packed_2[4];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p3[_lp*2 + 0], p3[_lp*2+1 + 0]));
        _packed_2[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p3[_lp*2 + 2], p3[_lp*2+1 + 2]));
        _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p3[_lp*2 + 4], p3[_lp*2+1 + 4]));
        _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p3[_lp*2 + 6], p3[_lp*2+1 + 6]));
        _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(fnTp + dst_2)[0] = reinterpret_cast<int4*>(_packed_2)[0];
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

} // extern "C"
