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
#define THREADS 128
#define LATENT 3584
#define ROWS_PER_CTA 4
#define VECS_PER_LANE 14
#define EARLY_TRIGGER 1

#include <math_constants.h>

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_kimi_k3_latent_moe_912c57b821d966ddadc2(__nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ y_out, int M, int num_partials, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    int row = bid * ROWS_PER_CTA + warp;
    int load_row = ((row) < (M - 1) ? (row) : (M - 1));
    unsigned long long row_base = (unsigned long long)load_row * (unsigned long long)LATENT;
    unsigned int xp[56];
    #pragma unroll
    for (int i = 0; i < VECS_PER_LANE; i++) {
        int kx = (lane + i * 32) * 8;
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(routed + (row_base + (unsigned long long)kx) + 0);
            uint4* _vdst_0 = reinterpret_cast<uint4*>(&xp[i * 4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_0[_blk] = _vptr_0[_blk];
            }
        }
    }
    float psq = 0.0f;
    #pragma unroll
    for (int i_1 = 0; i_1 < VECS_PER_LANE * 4; i_1++) {
        float _bf16x2_add_f32_0[2];
        asm volatile(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(_bf16x2_add_f32_0[0]), "=&f"(_bf16x2_add_f32_0[1]) : "r"(xp[i_1]), "f"(0.0f), "f"(0.0f));
        psq = __fmaf_rn(_bf16x2_add_f32_0[0], _bf16x2_add_f32_0[0], psq);
        psq = __fmaf_rn(_bf16x2_add_f32_0[1], _bf16x2_add_f32_0[1], psq);
    }
    float ptotal = psq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        ptotal += __shfl_xor_sync(0xFFFFFFFF, ptotal, offset);
    float prstd = rsqrtf(ptotal / (float)LATENT + eps);
    if (row < M) {
        unsigned long long out_base = (unsigned long long)row * (unsigned long long)LATENT;
        #pragma unroll
        for (int i_2 = 0; i_2 < VECS_PER_LANE; i_2++) {
            int kw = (lane + i_2 * 32) * 8;
            unsigned int nwp[4];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(norm_weight + kw + 0);
                uint4* _vdst_1 = reinterpret_cast<uint4*>(&nwp[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_1[_blk] = _vptr_1[_blk];
                }
            }
            float _unpack_bf16_0[8];
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((xp + i_2 * 4)[_pair]) << 16);
                (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((xp + i_2 * 4)[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&_unpack_bf16_0[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((xp + i_2 * 4)[1 + _pair]) << 16);
                (&_unpack_bf16_0[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((xp + i_2 * 4)[1 + _pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&_unpack_bf16_0[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((xp + i_2 * 4)[2 + _pair]) << 16);
                (&_unpack_bf16_0[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((xp + i_2 * 4)[2 + _pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&_unpack_bf16_0[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((xp + i_2 * 4)[3 + _pair]) << 16);
                (&_unpack_bf16_0[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((xp + i_2 * 4)[3 + _pair]) & 0xffff0000u);
            }
            const float2 _add2_2 = {0.0f, 0.0f};
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                add_f32x2_inplace(&reinterpret_cast<float2*>(_unpack_bf16_0)[_la], _add2_2);
            float yv[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                yv[j] = (float)(__nv_bfloat16)(_unpack_bf16_0[j] * prstd);
            }
            float _unpack_bf16_1[8];
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(nwp[_pair]) << 16);
                (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(nwp[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&_unpack_bf16_1[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(nwp[1 + _pair]) << 16);
                (&_unpack_bf16_1[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(nwp[1 + _pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&_unpack_bf16_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(nwp[2 + _pair]) << 16);
                (&_unpack_bf16_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(nwp[2 + _pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&_unpack_bf16_1[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(nwp[3 + _pair]) << 16);
                (&_unpack_bf16_1[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(nwp[3 + _pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int _ls = 0; _ls < 4; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(yv)[_ls], reinterpret_cast<const float2*>(_unpack_bf16_1)[_ls]);
            unsigned int _packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv[_lp*2 + 0], yv[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv[_lp*2 + 2], yv[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv[_lp*2 + 4], yv[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv[_lp*2 + 6], yv[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(y_out + (out_base + (unsigned long long)kw))[0] = reinterpret_cast<int4*>(_packed_0)[0];
        }
    }
}

} // extern "C"
