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

__device__ __forceinline__ float approx_exp2_noftz(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(64) void
kernel_cake_dsv4_e1ae8472ce68a65facf8(__nv_bfloat16* partial_O, float* partial_lse, __nv_bfloat16* __restrict__ O, int num_heads, int num_splits)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    unsigned long long heads = (unsigned long long)num_heads;
    unsigned long long head = (unsigned long long)blockIdx.y * 2ull + (unsigned long long)warp;
    unsigned long long row = (unsigned long long)blockIdx.x * 3ull * heads + head;
    unsigned long long chunk = (unsigned long long)lane * 8ull;
    float* lse_row = partial_lse + row;
    __nv_bfloat16* o_chunk = partial_O + (row * 512ull + chunk);
    unsigned long long o_stride = heads * 512ull;
    __nv_bfloat16* out_chunk = O + (((unsigned long long)blockIdx.x * heads + head) * 512ull + chunk);
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float lse[3];
    lse[0] = lse_row[0ull * heads];
    lse[1] = lse_row[1ull * heads];
    lse[2] = lse_row[2ull * heads];
    float m = -CUDART_INF_F;
    m = max_noftz(m, lse[0]);
    m = max_noftz(m, lse[1]);
    m = max_noftz(m, lse[2]);
    if (m != CUDART_INF_F || lse[0] != CUDART_INF_F) {
        float acc[16];
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        acc[8] = 0.0f;
        acc[9] = 0.0f;
        acc[10] = 0.0f;
        acc[11] = 0.0f;
        acc[12] = 0.0f;
        acc[13] = 0.0f;
        acc[14] = 0.0f;
        acc[15] = 0.0f;
        if (m != -CUDART_INF_F) {
            unsigned int words[24];
            words[0] = 0;
            words[1] = 0;
            words[2] = 0;
            words[3] = 0;
            words[4] = 0;
            words[5] = 0;
            words[6] = 0;
            words[7] = 0;
            if (lse[0] != -CUDART_INF_F) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(o_chunk + (0ull * o_stride + 0ull) + 0);
                    uint4* _vdst_0 = reinterpret_cast<uint4*>(&words[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_0[_blk] = _vptr_0[_blk];
                    }
                }
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(o_chunk + (0ull * o_stride + 256ull) + 0);
                    uint4* _vdst_1 = reinterpret_cast<uint4*>(&words[4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_1[_blk] = _vptr_1[_blk];
                    }
                }
            }
            words[8] = 0;
            words[9] = 0;
            words[10] = 0;
            words[11] = 0;
            words[12] = 0;
            words[13] = 0;
            words[14] = 0;
            words[15] = 0;
            if (lse[1] != -CUDART_INF_F) {
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(o_chunk + (1ull * o_stride + 0ull) + 0);
                    uint4* _vdst_2 = reinterpret_cast<uint4*>(&words[8]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_2[_blk] = _vptr_2[_blk];
                    }
                }
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(o_chunk + (1ull * o_stride + 256ull) + 0);
                    uint4* _vdst_3 = reinterpret_cast<uint4*>(&words[12]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_3[_blk] = _vptr_3[_blk];
                    }
                }
            }
            words[16] = 0;
            words[17] = 0;
            words[18] = 0;
            words[19] = 0;
            words[20] = 0;
            words[21] = 0;
            words[22] = 0;
            words[23] = 0;
            if (lse[2] != -CUDART_INF_F) {
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(o_chunk + (2ull * o_stride + 0ull) + 0);
                    uint4* _vdst_4 = reinterpret_cast<uint4*>(&words[16]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_4[_blk] = _vptr_4[_blk];
                    }
                }
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(o_chunk + (2ull * o_stride + 256ull) + 0);
                    uint4* _vdst_5 = reinterpret_cast<uint4*>(&words[20]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_5[_blk] = _vptr_5[_blk];
                    }
                }
            }
            float den = 0.0f;
            float _sub_8;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(lse[0]), "f"(m));
            float _exp2_7 = approx_exp2_noftz(_sub_8);
            float _add_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(den), "f"(_exp2_7));
            den = _add_6;
            float _sub_11;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(lse[1]), "f"(m));
            float _exp2_10 = approx_exp2_noftz(_sub_11);
            float _add_9;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(den), "f"(_exp2_10));
            den = _add_9;
            float _sub_14;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14) : "f"(lse[2]), "f"(m));
            float _exp2_13 = approx_exp2_noftz(_sub_14);
            float _add_12;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(den), "f"(_exp2_13));
            den = _add_12;
            float w[3];
            float _sub_17;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(lse[0]), "f"(m));
            float _exp2_16 = approx_exp2_noftz(_sub_17);
            float _fdiv_15 = __fdiv_rn(_exp2_16, den);
            w[0] = _fdiv_15;
            float _sub_20;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(lse[1]), "f"(m));
            float _exp2_19 = approx_exp2_noftz(_sub_20);
            float _fdiv_18 = __fdiv_rn(_exp2_19, den);
            w[1] = _fdiv_18;
            float _sub_23;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(lse[2]), "f"(m));
            float _exp2_22 = approx_exp2_noftz(_sub_23);
            float _fdiv_21 = __fdiv_rn(_exp2_22, den);
            w[2] = _fdiv_21;
            acc[0] = __fmaf_rn(w[0], __uint_as_float(words[0] << 16), acc[0]);
            acc[1] = __fmaf_rn(w[0], __uint_as_float(words[0] & 4294901760u), acc[1]);
            acc[2] = __fmaf_rn(w[0], __uint_as_float(words[1] << 16), acc[2]);
            acc[3] = __fmaf_rn(w[0], __uint_as_float(words[1] & 4294901760u), acc[3]);
            acc[4] = __fmaf_rn(w[0], __uint_as_float(words[2] << 16), acc[4]);
            acc[5] = __fmaf_rn(w[0], __uint_as_float(words[2] & 4294901760u), acc[5]);
            acc[6] = __fmaf_rn(w[0], __uint_as_float(words[3] << 16), acc[6]);
            acc[7] = __fmaf_rn(w[0], __uint_as_float(words[3] & 4294901760u), acc[7]);
            acc[0] = __fmaf_rn(w[1], __uint_as_float(words[8] << 16), acc[0]);
            acc[1] = __fmaf_rn(w[1], __uint_as_float(words[8] & 4294901760u), acc[1]);
            acc[2] = __fmaf_rn(w[1], __uint_as_float(words[9] << 16), acc[2]);
            acc[3] = __fmaf_rn(w[1], __uint_as_float(words[9] & 4294901760u), acc[3]);
            acc[4] = __fmaf_rn(w[1], __uint_as_float(words[10] << 16), acc[4]);
            acc[5] = __fmaf_rn(w[1], __uint_as_float(words[10] & 4294901760u), acc[5]);
            acc[6] = __fmaf_rn(w[1], __uint_as_float(words[11] << 16), acc[6]);
            acc[7] = __fmaf_rn(w[1], __uint_as_float(words[11] & 4294901760u), acc[7]);
            acc[0] = __fmaf_rn(w[2], __uint_as_float(words[16] << 16), acc[0]);
            acc[1] = __fmaf_rn(w[2], __uint_as_float(words[16] & 4294901760u), acc[1]);
            acc[2] = __fmaf_rn(w[2], __uint_as_float(words[17] << 16), acc[2]);
            acc[3] = __fmaf_rn(w[2], __uint_as_float(words[17] & 4294901760u), acc[3]);
            acc[4] = __fmaf_rn(w[2], __uint_as_float(words[18] << 16), acc[4]);
            acc[5] = __fmaf_rn(w[2], __uint_as_float(words[18] & 4294901760u), acc[5]);
            acc[6] = __fmaf_rn(w[2], __uint_as_float(words[19] << 16), acc[6]);
            acc[7] = __fmaf_rn(w[2], __uint_as_float(words[19] & 4294901760u), acc[7]);
            acc[8] = __fmaf_rn(w[0], __uint_as_float(words[4] << 16), acc[8]);
            acc[9] = __fmaf_rn(w[0], __uint_as_float(words[4] & 4294901760u), acc[9]);
            acc[10] = __fmaf_rn(w[0], __uint_as_float(words[5] << 16), acc[10]);
            acc[11] = __fmaf_rn(w[0], __uint_as_float(words[5] & 4294901760u), acc[11]);
            acc[12] = __fmaf_rn(w[0], __uint_as_float(words[6] << 16), acc[12]);
            acc[13] = __fmaf_rn(w[0], __uint_as_float(words[6] & 4294901760u), acc[13]);
            acc[14] = __fmaf_rn(w[0], __uint_as_float(words[7] << 16), acc[14]);
            acc[15] = __fmaf_rn(w[0], __uint_as_float(words[7] & 4294901760u), acc[15]);
            acc[8] = __fmaf_rn(w[1], __uint_as_float(words[12] << 16), acc[8]);
            acc[9] = __fmaf_rn(w[1], __uint_as_float(words[12] & 4294901760u), acc[9]);
            acc[10] = __fmaf_rn(w[1], __uint_as_float(words[13] << 16), acc[10]);
            acc[11] = __fmaf_rn(w[1], __uint_as_float(words[13] & 4294901760u), acc[11]);
            acc[12] = __fmaf_rn(w[1], __uint_as_float(words[14] << 16), acc[12]);
            acc[13] = __fmaf_rn(w[1], __uint_as_float(words[14] & 4294901760u), acc[13]);
            acc[14] = __fmaf_rn(w[1], __uint_as_float(words[15] << 16), acc[14]);
            acc[15] = __fmaf_rn(w[1], __uint_as_float(words[15] & 4294901760u), acc[15]);
            acc[8] = __fmaf_rn(w[2], __uint_as_float(words[20] << 16), acc[8]);
            acc[9] = __fmaf_rn(w[2], __uint_as_float(words[20] & 4294901760u), acc[9]);
            acc[10] = __fmaf_rn(w[2], __uint_as_float(words[21] << 16), acc[10]);
            acc[11] = __fmaf_rn(w[2], __uint_as_float(words[21] & 4294901760u), acc[11]);
            acc[12] = __fmaf_rn(w[2], __uint_as_float(words[22] << 16), acc[12]);
            acc[13] = __fmaf_rn(w[2], __uint_as_float(words[22] & 4294901760u), acc[13]);
            acc[14] = __fmaf_rn(w[2], __uint_as_float(words[23] << 16), acc[14]);
            acc[15] = __fmaf_rn(w[2], __uint_as_float(words[23] & 4294901760u), acc[15]);
        }
        unsigned int packed[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
            packed[_lp] = *(uint32_t*)&_bf2;
        }
        {
            auto _gst_24 = out_chunk + 0ull;
            __builtin_assume(__isGlobal(_gst_24));
            reinterpret_cast<int4*>(_gst_24)[0] = reinterpret_cast<int4*>(packed)[0];
        }
        {
            auto _gst_25 = out_chunk + 256ull;
            __builtin_assume(__isGlobal(_gst_25));
            reinterpret_cast<int4*>(_gst_25)[0] = reinterpret_cast<int4*>(packed + (4))[0];
        }
    }
}

} // extern "C"
