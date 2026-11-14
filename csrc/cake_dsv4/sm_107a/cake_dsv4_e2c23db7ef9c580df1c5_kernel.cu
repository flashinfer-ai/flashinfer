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
kernel_cake_dsv4_e2c23db7ef9c580df1c5(__nv_bfloat16* partial_O, float* partial_lse, __nv_bfloat16* __restrict__ O, int num_heads, int num_splits)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    unsigned long long heads = (unsigned long long)num_heads;
    unsigned long long row = (unsigned long long)blockIdx.x * (unsigned long long)num_splits * heads + (unsigned long long)blockIdx.y;
    unsigned long long chunk = (unsigned long long)tid * 8;
    float* lse_row = partial_lse + row;
    __nv_bfloat16* o_chunk = partial_O + (row * 512 + chunk);
    unsigned long long o_stride = heads * 512;
    __nv_bfloat16* out_chunk = O + (((unsigned long long)blockIdx.x * heads + (unsigned long long)blockIdx.y) * 512 + chunk);
    if (num_splits == 1) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        float lse[1];
        lse[0] = lse_row[0 * heads];
        float m = -CUDART_INF_F;
        m = max_noftz(m, lse[0]);
        if (m != CUDART_INF_F || lse[0] != CUDART_INF_F) {
            float acc[8];
            acc[0] = 0.0f;
            acc[1] = 0.0f;
            acc[2] = 0.0f;
            acc[3] = 0.0f;
            acc[4] = 0.0f;
            acc[5] = 0.0f;
            acc[6] = 0.0f;
            acc[7] = 0.0f;
            if (m != -CUDART_INF_F) {
                unsigned int words[4];
                words[0] = 0;
                words[1] = 0;
                words[2] = 0;
                words[3] = 0;
                if (lse[0] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(o_chunk + 0 * o_stride + 0);
                        uint4* _vdst_0 = reinterpret_cast<uint4*>(&words[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_0[_blk] = _vptr_0[_blk];
                        }
                    }
                }
                float den = 0.0f;
                float _sub_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(lse[0]), "f"(m));
                float _exp2_2 = approx_exp2_noftz(_sub_3);
                float _add_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(den), "f"(_exp2_2));
                den = _add_1;
                float w[1];
                float _sub_6;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(lse[0]), "f"(m));
                float _exp2_5 = approx_exp2_noftz(_sub_6);
                float _fdiv_4 = __fdiv_rn(_exp2_5, den);
                w[0] = _fdiv_4;
                acc[0] = __fmaf_rn(w[0], __uint_as_float(words[0] << 16), acc[0]);
                acc[1] = __fmaf_rn(w[0], __uint_as_float(words[0] & 4294901760u), acc[1]);
                acc[2] = __fmaf_rn(w[0], __uint_as_float(words[1] << 16), acc[2]);
                acc[3] = __fmaf_rn(w[0], __uint_as_float(words[1] & 4294901760u), acc[3]);
                acc[4] = __fmaf_rn(w[0], __uint_as_float(words[2] << 16), acc[4]);
                acc[5] = __fmaf_rn(w[0], __uint_as_float(words[2] & 4294901760u), acc[5]);
                acc[6] = __fmaf_rn(w[0], __uint_as_float(words[3] << 16), acc[6]);
                acc[7] = __fmaf_rn(w[0], __uint_as_float(words[3] & 4294901760u), acc[7]);
            }
            unsigned int packed[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                packed[_lp] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_7 = out_chunk;
                __builtin_assume(__isGlobal(_gst_7));
                reinterpret_cast<int4*>(_gst_7)[0] = reinterpret_cast<int4*>(packed)[0];
            }
        }
    }
    if (num_splits == 2) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        float lse_1[2];
        lse_1[0] = lse_row[0 * heads];
        lse_1[1] = lse_row[1 * heads];
        float m_1 = -CUDART_INF_F;
        m_1 = max_noftz(m_1, lse_1[0]);
        m_1 = max_noftz(m_1, lse_1[1]);
        if (m_1 != CUDART_INF_F || lse_1[0] != CUDART_INF_F) {
            float acc_1[8];
            acc_1[0] = 0.0f;
            acc_1[1] = 0.0f;
            acc_1[2] = 0.0f;
            acc_1[3] = 0.0f;
            acc_1[4] = 0.0f;
            acc_1[5] = 0.0f;
            acc_1[6] = 0.0f;
            acc_1[7] = 0.0f;
            if (m_1 != -CUDART_INF_F) {
                unsigned int words_1[8];
                words_1[0] = 0;
                words_1[1] = 0;
                words_1[2] = 0;
                words_1[3] = 0;
                if (lse_1[0] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_8 = reinterpret_cast<const uint4*>(o_chunk + 0 * o_stride + 0);
                        uint4* _vdst_8 = reinterpret_cast<uint4*>(&words_1[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_8[_blk] = _vptr_8[_blk];
                        }
                    }
                }
                words_1[4] = 0;
                words_1[5] = 0;
                words_1[6] = 0;
                words_1[7] = 0;
                if (lse_1[1] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_9 = reinterpret_cast<const uint4*>(o_chunk + 1 * o_stride + 0);
                        uint4* _vdst_9 = reinterpret_cast<uint4*>(&words_1[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_9[_blk] = _vptr_9[_blk];
                        }
                    }
                }
                float den_1 = 0.0f;
                float _sub_12;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12) : "f"(lse_1[0]), "f"(m_1));
                float _exp2_11 = approx_exp2_noftz(_sub_12);
                float _add_10;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(den_1), "f"(_exp2_11));
                den_1 = _add_10;
                float _sub_15;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(lse_1[1]), "f"(m_1));
                float _exp2_14 = approx_exp2_noftz(_sub_15);
                float _add_13;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(den_1), "f"(_exp2_14));
                den_1 = _add_13;
                float w_1[2];
                float _sub_18;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(lse_1[0]), "f"(m_1));
                float _exp2_17 = approx_exp2_noftz(_sub_18);
                float _fdiv_16 = __fdiv_rn(_exp2_17, den_1);
                w_1[0] = _fdiv_16;
                float _sub_21;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21) : "f"(lse_1[1]), "f"(m_1));
                float _exp2_20 = approx_exp2_noftz(_sub_21);
                float _fdiv_19 = __fdiv_rn(_exp2_20, den_1);
                w_1[1] = _fdiv_19;
                acc_1[0] = __fmaf_rn(w_1[0], __uint_as_float(words_1[0] << 16), acc_1[0]);
                acc_1[1] = __fmaf_rn(w_1[0], __uint_as_float(words_1[0] & 4294901760u), acc_1[1]);
                acc_1[2] = __fmaf_rn(w_1[0], __uint_as_float(words_1[1] << 16), acc_1[2]);
                acc_1[3] = __fmaf_rn(w_1[0], __uint_as_float(words_1[1] & 4294901760u), acc_1[3]);
                acc_1[4] = __fmaf_rn(w_1[0], __uint_as_float(words_1[2] << 16), acc_1[4]);
                acc_1[5] = __fmaf_rn(w_1[0], __uint_as_float(words_1[2] & 4294901760u), acc_1[5]);
                acc_1[6] = __fmaf_rn(w_1[0], __uint_as_float(words_1[3] << 16), acc_1[6]);
                acc_1[7] = __fmaf_rn(w_1[0], __uint_as_float(words_1[3] & 4294901760u), acc_1[7]);
                acc_1[0] = __fmaf_rn(w_1[1], __uint_as_float(words_1[4] << 16), acc_1[0]);
                acc_1[1] = __fmaf_rn(w_1[1], __uint_as_float(words_1[4] & 4294901760u), acc_1[1]);
                acc_1[2] = __fmaf_rn(w_1[1], __uint_as_float(words_1[5] << 16), acc_1[2]);
                acc_1[3] = __fmaf_rn(w_1[1], __uint_as_float(words_1[5] & 4294901760u), acc_1[3]);
                acc_1[4] = __fmaf_rn(w_1[1], __uint_as_float(words_1[6] << 16), acc_1[4]);
                acc_1[5] = __fmaf_rn(w_1[1], __uint_as_float(words_1[6] & 4294901760u), acc_1[5]);
                acc_1[6] = __fmaf_rn(w_1[1], __uint_as_float(words_1[7] << 16), acc_1[6]);
                acc_1[7] = __fmaf_rn(w_1[1], __uint_as_float(words_1[7] & 4294901760u), acc_1[7]);
            }
            unsigned int packed_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_1[_lp*2 + 0], acc_1[_lp*2+1 + 0]));
                packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_22 = out_chunk;
                __builtin_assume(__isGlobal(_gst_22));
                reinterpret_cast<int4*>(_gst_22)[0] = reinterpret_cast<int4*>(packed_1)[0];
            }
        }
    }
    if (num_splits == 3) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        float lse_2[3];
        lse_2[0] = lse_row[0 * heads];
        lse_2[1] = lse_row[1 * heads];
        lse_2[2] = lse_row[2 * heads];
        float m_2 = -CUDART_INF_F;
        m_2 = max_noftz(m_2, lse_2[0]);
        m_2 = max_noftz(m_2, lse_2[1]);
        m_2 = max_noftz(m_2, lse_2[2]);
        if (m_2 != CUDART_INF_F || lse_2[0] != CUDART_INF_F) {
            float acc_2[8];
            acc_2[0] = 0.0f;
            acc_2[1] = 0.0f;
            acc_2[2] = 0.0f;
            acc_2[3] = 0.0f;
            acc_2[4] = 0.0f;
            acc_2[5] = 0.0f;
            acc_2[6] = 0.0f;
            acc_2[7] = 0.0f;
            if (m_2 != -CUDART_INF_F) {
                unsigned int words_2[12];
                words_2[0] = 0;
                words_2[1] = 0;
                words_2[2] = 0;
                words_2[3] = 0;
                if (lse_2[0] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_23 = reinterpret_cast<const uint4*>(o_chunk + 0 * o_stride + 0);
                        uint4* _vdst_23 = reinterpret_cast<uint4*>(&words_2[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_23[_blk] = _vptr_23[_blk];
                        }
                    }
                }
                words_2[4] = 0;
                words_2[5] = 0;
                words_2[6] = 0;
                words_2[7] = 0;
                if (lse_2[1] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_24 = reinterpret_cast<const uint4*>(o_chunk + 1 * o_stride + 0);
                        uint4* _vdst_24 = reinterpret_cast<uint4*>(&words_2[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_24[_blk] = _vptr_24[_blk];
                        }
                    }
                }
                words_2[8] = 0;
                words_2[9] = 0;
                words_2[10] = 0;
                words_2[11] = 0;
                if (lse_2[2] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_25 = reinterpret_cast<const uint4*>(o_chunk + 2 * o_stride + 0);
                        uint4* _vdst_25 = reinterpret_cast<uint4*>(&words_2[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_25[_blk] = _vptr_25[_blk];
                        }
                    }
                }
                float den_2 = 0.0f;
                float _sub_28;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(lse_2[0]), "f"(m_2));
                float _exp2_27 = approx_exp2_noftz(_sub_28);
                float _add_26;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(den_2), "f"(_exp2_27));
                den_2 = _add_26;
                float _sub_31;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(lse_2[1]), "f"(m_2));
                float _exp2_30 = approx_exp2_noftz(_sub_31);
                float _add_29;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(den_2), "f"(_exp2_30));
                den_2 = _add_29;
                float _sub_34;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(lse_2[2]), "f"(m_2));
                float _exp2_33 = approx_exp2_noftz(_sub_34);
                float _add_32;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(den_2), "f"(_exp2_33));
                den_2 = _add_32;
                float w_2[3];
                float _sub_37;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(lse_2[0]), "f"(m_2));
                float _exp2_36 = approx_exp2_noftz(_sub_37);
                float _fdiv_35 = __fdiv_rn(_exp2_36, den_2);
                w_2[0] = _fdiv_35;
                float _sub_40;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(lse_2[1]), "f"(m_2));
                float _exp2_39 = approx_exp2_noftz(_sub_40);
                float _fdiv_38 = __fdiv_rn(_exp2_39, den_2);
                w_2[1] = _fdiv_38;
                float _sub_43;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(lse_2[2]), "f"(m_2));
                float _exp2_42 = approx_exp2_noftz(_sub_43);
                float _fdiv_41 = __fdiv_rn(_exp2_42, den_2);
                w_2[2] = _fdiv_41;
                acc_2[0] = __fmaf_rn(w_2[0], __uint_as_float(words_2[0] << 16), acc_2[0]);
                acc_2[1] = __fmaf_rn(w_2[0], __uint_as_float(words_2[0] & 4294901760u), acc_2[1]);
                acc_2[2] = __fmaf_rn(w_2[0], __uint_as_float(words_2[1] << 16), acc_2[2]);
                acc_2[3] = __fmaf_rn(w_2[0], __uint_as_float(words_2[1] & 4294901760u), acc_2[3]);
                acc_2[4] = __fmaf_rn(w_2[0], __uint_as_float(words_2[2] << 16), acc_2[4]);
                acc_2[5] = __fmaf_rn(w_2[0], __uint_as_float(words_2[2] & 4294901760u), acc_2[5]);
                acc_2[6] = __fmaf_rn(w_2[0], __uint_as_float(words_2[3] << 16), acc_2[6]);
                acc_2[7] = __fmaf_rn(w_2[0], __uint_as_float(words_2[3] & 4294901760u), acc_2[7]);
                acc_2[0] = __fmaf_rn(w_2[1], __uint_as_float(words_2[4] << 16), acc_2[0]);
                acc_2[1] = __fmaf_rn(w_2[1], __uint_as_float(words_2[4] & 4294901760u), acc_2[1]);
                acc_2[2] = __fmaf_rn(w_2[1], __uint_as_float(words_2[5] << 16), acc_2[2]);
                acc_2[3] = __fmaf_rn(w_2[1], __uint_as_float(words_2[5] & 4294901760u), acc_2[3]);
                acc_2[4] = __fmaf_rn(w_2[1], __uint_as_float(words_2[6] << 16), acc_2[4]);
                acc_2[5] = __fmaf_rn(w_2[1], __uint_as_float(words_2[6] & 4294901760u), acc_2[5]);
                acc_2[6] = __fmaf_rn(w_2[1], __uint_as_float(words_2[7] << 16), acc_2[6]);
                acc_2[7] = __fmaf_rn(w_2[1], __uint_as_float(words_2[7] & 4294901760u), acc_2[7]);
                acc_2[0] = __fmaf_rn(w_2[2], __uint_as_float(words_2[8] << 16), acc_2[0]);
                acc_2[1] = __fmaf_rn(w_2[2], __uint_as_float(words_2[8] & 4294901760u), acc_2[1]);
                acc_2[2] = __fmaf_rn(w_2[2], __uint_as_float(words_2[9] << 16), acc_2[2]);
                acc_2[3] = __fmaf_rn(w_2[2], __uint_as_float(words_2[9] & 4294901760u), acc_2[3]);
                acc_2[4] = __fmaf_rn(w_2[2], __uint_as_float(words_2[10] << 16), acc_2[4]);
                acc_2[5] = __fmaf_rn(w_2[2], __uint_as_float(words_2[10] & 4294901760u), acc_2[5]);
                acc_2[6] = __fmaf_rn(w_2[2], __uint_as_float(words_2[11] << 16), acc_2[6]);
                acc_2[7] = __fmaf_rn(w_2[2], __uint_as_float(words_2[11] & 4294901760u), acc_2[7]);
            }
            unsigned int packed_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_2[_lp*2 + 0], acc_2[_lp*2+1 + 0]));
                packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_44 = out_chunk;
                __builtin_assume(__isGlobal(_gst_44));
                reinterpret_cast<int4*>(_gst_44)[0] = reinterpret_cast<int4*>(packed_2)[0];
            }
        }
    }
    if (num_splits == 4) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        float lse_3[4];
        lse_3[0] = lse_row[0 * heads];
        lse_3[1] = lse_row[1 * heads];
        lse_3[2] = lse_row[2 * heads];
        lse_3[3] = lse_row[3 * heads];
        float m_3 = -CUDART_INF_F;
        m_3 = max_noftz(m_3, lse_3[0]);
        m_3 = max_noftz(m_3, lse_3[1]);
        m_3 = max_noftz(m_3, lse_3[2]);
        m_3 = max_noftz(m_3, lse_3[3]);
        if (m_3 != CUDART_INF_F || lse_3[0] != CUDART_INF_F) {
            float acc_3[8];
            acc_3[0] = 0.0f;
            acc_3[1] = 0.0f;
            acc_3[2] = 0.0f;
            acc_3[3] = 0.0f;
            acc_3[4] = 0.0f;
            acc_3[5] = 0.0f;
            acc_3[6] = 0.0f;
            acc_3[7] = 0.0f;
            if (m_3 != -CUDART_INF_F) {
                unsigned int words_3[16];
                words_3[0] = 0;
                words_3[1] = 0;
                words_3[2] = 0;
                words_3[3] = 0;
                if (lse_3[0] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_45 = reinterpret_cast<const uint4*>(o_chunk + 0 * o_stride + 0);
                        uint4* _vdst_45 = reinterpret_cast<uint4*>(&words_3[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_45[_blk] = _vptr_45[_blk];
                        }
                    }
                }
                words_3[4] = 0;
                words_3[5] = 0;
                words_3[6] = 0;
                words_3[7] = 0;
                if (lse_3[1] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_46 = reinterpret_cast<const uint4*>(o_chunk + 1 * o_stride + 0);
                        uint4* _vdst_46 = reinterpret_cast<uint4*>(&words_3[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_46[_blk] = _vptr_46[_blk];
                        }
                    }
                }
                words_3[8] = 0;
                words_3[9] = 0;
                words_3[10] = 0;
                words_3[11] = 0;
                if (lse_3[2] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_47 = reinterpret_cast<const uint4*>(o_chunk + 2 * o_stride + 0);
                        uint4* _vdst_47 = reinterpret_cast<uint4*>(&words_3[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_47[_blk] = _vptr_47[_blk];
                        }
                    }
                }
                words_3[12] = 0;
                words_3[13] = 0;
                words_3[14] = 0;
                words_3[15] = 0;
                if (lse_3[3] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_48 = reinterpret_cast<const uint4*>(o_chunk + 3 * o_stride + 0);
                        uint4* _vdst_48 = reinterpret_cast<uint4*>(&words_3[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_48[_blk] = _vptr_48[_blk];
                        }
                    }
                }
                float den_3 = 0.0f;
                float _sub_51;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(lse_3[0]), "f"(m_3));
                float _exp2_50 = approx_exp2_noftz(_sub_51);
                float _add_49;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(den_3), "f"(_exp2_50));
                den_3 = _add_49;
                float _sub_54;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(lse_3[1]), "f"(m_3));
                float _exp2_53 = approx_exp2_noftz(_sub_54);
                float _add_52;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(den_3), "f"(_exp2_53));
                den_3 = _add_52;
                float _sub_57;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57) : "f"(lse_3[2]), "f"(m_3));
                float _exp2_56 = approx_exp2_noftz(_sub_57);
                float _add_55;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(den_3), "f"(_exp2_56));
                den_3 = _add_55;
                float _sub_60;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60) : "f"(lse_3[3]), "f"(m_3));
                float _exp2_59 = approx_exp2_noftz(_sub_60);
                float _add_58;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(den_3), "f"(_exp2_59));
                den_3 = _add_58;
                float w_3[4];
                float _sub_63;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(lse_3[0]), "f"(m_3));
                float _exp2_62 = approx_exp2_noftz(_sub_63);
                float _fdiv_61 = __fdiv_rn(_exp2_62, den_3);
                w_3[0] = _fdiv_61;
                float _sub_66;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(lse_3[1]), "f"(m_3));
                float _exp2_65 = approx_exp2_noftz(_sub_66);
                float _fdiv_64 = __fdiv_rn(_exp2_65, den_3);
                w_3[1] = _fdiv_64;
                float _sub_69;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(lse_3[2]), "f"(m_3));
                float _exp2_68 = approx_exp2_noftz(_sub_69);
                float _fdiv_67 = __fdiv_rn(_exp2_68, den_3);
                w_3[2] = _fdiv_67;
                float _sub_72;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_72) : "f"(lse_3[3]), "f"(m_3));
                float _exp2_71 = approx_exp2_noftz(_sub_72);
                float _fdiv_70 = __fdiv_rn(_exp2_71, den_3);
                w_3[3] = _fdiv_70;
                acc_3[0] = __fmaf_rn(w_3[0], __uint_as_float(words_3[0] << 16), acc_3[0]);
                acc_3[1] = __fmaf_rn(w_3[0], __uint_as_float(words_3[0] & 4294901760u), acc_3[1]);
                acc_3[2] = __fmaf_rn(w_3[0], __uint_as_float(words_3[1] << 16), acc_3[2]);
                acc_3[3] = __fmaf_rn(w_3[0], __uint_as_float(words_3[1] & 4294901760u), acc_3[3]);
                acc_3[4] = __fmaf_rn(w_3[0], __uint_as_float(words_3[2] << 16), acc_3[4]);
                acc_3[5] = __fmaf_rn(w_3[0], __uint_as_float(words_3[2] & 4294901760u), acc_3[5]);
                acc_3[6] = __fmaf_rn(w_3[0], __uint_as_float(words_3[3] << 16), acc_3[6]);
                acc_3[7] = __fmaf_rn(w_3[0], __uint_as_float(words_3[3] & 4294901760u), acc_3[7]);
                acc_3[0] = __fmaf_rn(w_3[1], __uint_as_float(words_3[4] << 16), acc_3[0]);
                acc_3[1] = __fmaf_rn(w_3[1], __uint_as_float(words_3[4] & 4294901760u), acc_3[1]);
                acc_3[2] = __fmaf_rn(w_3[1], __uint_as_float(words_3[5] << 16), acc_3[2]);
                acc_3[3] = __fmaf_rn(w_3[1], __uint_as_float(words_3[5] & 4294901760u), acc_3[3]);
                acc_3[4] = __fmaf_rn(w_3[1], __uint_as_float(words_3[6] << 16), acc_3[4]);
                acc_3[5] = __fmaf_rn(w_3[1], __uint_as_float(words_3[6] & 4294901760u), acc_3[5]);
                acc_3[6] = __fmaf_rn(w_3[1], __uint_as_float(words_3[7] << 16), acc_3[6]);
                acc_3[7] = __fmaf_rn(w_3[1], __uint_as_float(words_3[7] & 4294901760u), acc_3[7]);
                acc_3[0] = __fmaf_rn(w_3[2], __uint_as_float(words_3[8] << 16), acc_3[0]);
                acc_3[1] = __fmaf_rn(w_3[2], __uint_as_float(words_3[8] & 4294901760u), acc_3[1]);
                acc_3[2] = __fmaf_rn(w_3[2], __uint_as_float(words_3[9] << 16), acc_3[2]);
                acc_3[3] = __fmaf_rn(w_3[2], __uint_as_float(words_3[9] & 4294901760u), acc_3[3]);
                acc_3[4] = __fmaf_rn(w_3[2], __uint_as_float(words_3[10] << 16), acc_3[4]);
                acc_3[5] = __fmaf_rn(w_3[2], __uint_as_float(words_3[10] & 4294901760u), acc_3[5]);
                acc_3[6] = __fmaf_rn(w_3[2], __uint_as_float(words_3[11] << 16), acc_3[6]);
                acc_3[7] = __fmaf_rn(w_3[2], __uint_as_float(words_3[11] & 4294901760u), acc_3[7]);
                acc_3[0] = __fmaf_rn(w_3[3], __uint_as_float(words_3[12] << 16), acc_3[0]);
                acc_3[1] = __fmaf_rn(w_3[3], __uint_as_float(words_3[12] & 4294901760u), acc_3[1]);
                acc_3[2] = __fmaf_rn(w_3[3], __uint_as_float(words_3[13] << 16), acc_3[2]);
                acc_3[3] = __fmaf_rn(w_3[3], __uint_as_float(words_3[13] & 4294901760u), acc_3[3]);
                acc_3[4] = __fmaf_rn(w_3[3], __uint_as_float(words_3[14] << 16), acc_3[4]);
                acc_3[5] = __fmaf_rn(w_3[3], __uint_as_float(words_3[14] & 4294901760u), acc_3[5]);
                acc_3[6] = __fmaf_rn(w_3[3], __uint_as_float(words_3[15] << 16), acc_3[6]);
                acc_3[7] = __fmaf_rn(w_3[3], __uint_as_float(words_3[15] & 4294901760u), acc_3[7]);
            }
            unsigned int packed_3[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_3[_lp*2 + 0], acc_3[_lp*2+1 + 0]));
                packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_73 = out_chunk;
                __builtin_assume(__isGlobal(_gst_73));
                reinterpret_cast<int4*>(_gst_73)[0] = reinterpret_cast<int4*>(packed_3)[0];
            }
        }
    }
    if (num_splits == 5) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        float lse_4[5];
        lse_4[0] = lse_row[0 * heads];
        lse_4[1] = lse_row[1 * heads];
        lse_4[2] = lse_row[2 * heads];
        lse_4[3] = lse_row[3 * heads];
        lse_4[4] = lse_row[4 * heads];
        float m_4 = -CUDART_INF_F;
        m_4 = max_noftz(m_4, lse_4[0]);
        m_4 = max_noftz(m_4, lse_4[1]);
        m_4 = max_noftz(m_4, lse_4[2]);
        m_4 = max_noftz(m_4, lse_4[3]);
        m_4 = max_noftz(m_4, lse_4[4]);
        if (m_4 != CUDART_INF_F || lse_4[0] != CUDART_INF_F) {
            float acc_4[8];
            acc_4[0] = 0.0f;
            acc_4[1] = 0.0f;
            acc_4[2] = 0.0f;
            acc_4[3] = 0.0f;
            acc_4[4] = 0.0f;
            acc_4[5] = 0.0f;
            acc_4[6] = 0.0f;
            acc_4[7] = 0.0f;
            if (m_4 != -CUDART_INF_F) {
                unsigned int words_4[20];
                words_4[0] = 0;
                words_4[1] = 0;
                words_4[2] = 0;
                words_4[3] = 0;
                if (lse_4[0] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_74 = reinterpret_cast<const uint4*>(o_chunk + 0 * o_stride + 0);
                        uint4* _vdst_74 = reinterpret_cast<uint4*>(&words_4[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_74[_blk] = _vptr_74[_blk];
                        }
                    }
                }
                words_4[4] = 0;
                words_4[5] = 0;
                words_4[6] = 0;
                words_4[7] = 0;
                if (lse_4[1] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_75 = reinterpret_cast<const uint4*>(o_chunk + 1 * o_stride + 0);
                        uint4* _vdst_75 = reinterpret_cast<uint4*>(&words_4[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_75[_blk] = _vptr_75[_blk];
                        }
                    }
                }
                words_4[8] = 0;
                words_4[9] = 0;
                words_4[10] = 0;
                words_4[11] = 0;
                if (lse_4[2] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_76 = reinterpret_cast<const uint4*>(o_chunk + 2 * o_stride + 0);
                        uint4* _vdst_76 = reinterpret_cast<uint4*>(&words_4[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_76[_blk] = _vptr_76[_blk];
                        }
                    }
                }
                words_4[12] = 0;
                words_4[13] = 0;
                words_4[14] = 0;
                words_4[15] = 0;
                if (lse_4[3] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_77 = reinterpret_cast<const uint4*>(o_chunk + 3 * o_stride + 0);
                        uint4* _vdst_77 = reinterpret_cast<uint4*>(&words_4[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_77[_blk] = _vptr_77[_blk];
                        }
                    }
                }
                words_4[16] = 0;
                words_4[17] = 0;
                words_4[18] = 0;
                words_4[19] = 0;
                if (lse_4[4] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_78 = reinterpret_cast<const uint4*>(o_chunk + 4 * o_stride + 0);
                        uint4* _vdst_78 = reinterpret_cast<uint4*>(&words_4[16]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_78[_blk] = _vptr_78[_blk];
                        }
                    }
                }
                float den_4 = 0.0f;
                float _sub_81;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(lse_4[0]), "f"(m_4));
                float _exp2_80 = approx_exp2_noftz(_sub_81);
                float _add_79;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(den_4), "f"(_exp2_80));
                den_4 = _add_79;
                float _sub_84;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(lse_4[1]), "f"(m_4));
                float _exp2_83 = approx_exp2_noftz(_sub_84);
                float _add_82;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(den_4), "f"(_exp2_83));
                den_4 = _add_82;
                float _sub_87;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_87) : "f"(lse_4[2]), "f"(m_4));
                float _exp2_86 = approx_exp2_noftz(_sub_87);
                float _add_85;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(den_4), "f"(_exp2_86));
                den_4 = _add_85;
                float _sub_90;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_90) : "f"(lse_4[3]), "f"(m_4));
                float _exp2_89 = approx_exp2_noftz(_sub_90);
                float _add_88;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(den_4), "f"(_exp2_89));
                den_4 = _add_88;
                float _sub_93;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(lse_4[4]), "f"(m_4));
                float _exp2_92 = approx_exp2_noftz(_sub_93);
                float _add_91;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(den_4), "f"(_exp2_92));
                den_4 = _add_91;
                float w_4[5];
                float _sub_96;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_96) : "f"(lse_4[0]), "f"(m_4));
                float _exp2_95 = approx_exp2_noftz(_sub_96);
                float _fdiv_94 = __fdiv_rn(_exp2_95, den_4);
                w_4[0] = _fdiv_94;
                float _sub_99;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_99) : "f"(lse_4[1]), "f"(m_4));
                float _exp2_98 = approx_exp2_noftz(_sub_99);
                float _fdiv_97 = __fdiv_rn(_exp2_98, den_4);
                w_4[1] = _fdiv_97;
                float _sub_102;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(lse_4[2]), "f"(m_4));
                float _exp2_101 = approx_exp2_noftz(_sub_102);
                float _fdiv_100 = __fdiv_rn(_exp2_101, den_4);
                w_4[2] = _fdiv_100;
                float _sub_105;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_105) : "f"(lse_4[3]), "f"(m_4));
                float _exp2_104 = approx_exp2_noftz(_sub_105);
                float _fdiv_103 = __fdiv_rn(_exp2_104, den_4);
                w_4[3] = _fdiv_103;
                float _sub_108;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_108) : "f"(lse_4[4]), "f"(m_4));
                float _exp2_107 = approx_exp2_noftz(_sub_108);
                float _fdiv_106 = __fdiv_rn(_exp2_107, den_4);
                w_4[4] = _fdiv_106;
                acc_4[0] = __fmaf_rn(w_4[0], __uint_as_float(words_4[0] << 16), acc_4[0]);
                acc_4[1] = __fmaf_rn(w_4[0], __uint_as_float(words_4[0] & 4294901760u), acc_4[1]);
                acc_4[2] = __fmaf_rn(w_4[0], __uint_as_float(words_4[1] << 16), acc_4[2]);
                acc_4[3] = __fmaf_rn(w_4[0], __uint_as_float(words_4[1] & 4294901760u), acc_4[3]);
                acc_4[4] = __fmaf_rn(w_4[0], __uint_as_float(words_4[2] << 16), acc_4[4]);
                acc_4[5] = __fmaf_rn(w_4[0], __uint_as_float(words_4[2] & 4294901760u), acc_4[5]);
                acc_4[6] = __fmaf_rn(w_4[0], __uint_as_float(words_4[3] << 16), acc_4[6]);
                acc_4[7] = __fmaf_rn(w_4[0], __uint_as_float(words_4[3] & 4294901760u), acc_4[7]);
                acc_4[0] = __fmaf_rn(w_4[1], __uint_as_float(words_4[4] << 16), acc_4[0]);
                acc_4[1] = __fmaf_rn(w_4[1], __uint_as_float(words_4[4] & 4294901760u), acc_4[1]);
                acc_4[2] = __fmaf_rn(w_4[1], __uint_as_float(words_4[5] << 16), acc_4[2]);
                acc_4[3] = __fmaf_rn(w_4[1], __uint_as_float(words_4[5] & 4294901760u), acc_4[3]);
                acc_4[4] = __fmaf_rn(w_4[1], __uint_as_float(words_4[6] << 16), acc_4[4]);
                acc_4[5] = __fmaf_rn(w_4[1], __uint_as_float(words_4[6] & 4294901760u), acc_4[5]);
                acc_4[6] = __fmaf_rn(w_4[1], __uint_as_float(words_4[7] << 16), acc_4[6]);
                acc_4[7] = __fmaf_rn(w_4[1], __uint_as_float(words_4[7] & 4294901760u), acc_4[7]);
                acc_4[0] = __fmaf_rn(w_4[2], __uint_as_float(words_4[8] << 16), acc_4[0]);
                acc_4[1] = __fmaf_rn(w_4[2], __uint_as_float(words_4[8] & 4294901760u), acc_4[1]);
                acc_4[2] = __fmaf_rn(w_4[2], __uint_as_float(words_4[9] << 16), acc_4[2]);
                acc_4[3] = __fmaf_rn(w_4[2], __uint_as_float(words_4[9] & 4294901760u), acc_4[3]);
                acc_4[4] = __fmaf_rn(w_4[2], __uint_as_float(words_4[10] << 16), acc_4[4]);
                acc_4[5] = __fmaf_rn(w_4[2], __uint_as_float(words_4[10] & 4294901760u), acc_4[5]);
                acc_4[6] = __fmaf_rn(w_4[2], __uint_as_float(words_4[11] << 16), acc_4[6]);
                acc_4[7] = __fmaf_rn(w_4[2], __uint_as_float(words_4[11] & 4294901760u), acc_4[7]);
                acc_4[0] = __fmaf_rn(w_4[3], __uint_as_float(words_4[12] << 16), acc_4[0]);
                acc_4[1] = __fmaf_rn(w_4[3], __uint_as_float(words_4[12] & 4294901760u), acc_4[1]);
                acc_4[2] = __fmaf_rn(w_4[3], __uint_as_float(words_4[13] << 16), acc_4[2]);
                acc_4[3] = __fmaf_rn(w_4[3], __uint_as_float(words_4[13] & 4294901760u), acc_4[3]);
                acc_4[4] = __fmaf_rn(w_4[3], __uint_as_float(words_4[14] << 16), acc_4[4]);
                acc_4[5] = __fmaf_rn(w_4[3], __uint_as_float(words_4[14] & 4294901760u), acc_4[5]);
                acc_4[6] = __fmaf_rn(w_4[3], __uint_as_float(words_4[15] << 16), acc_4[6]);
                acc_4[7] = __fmaf_rn(w_4[3], __uint_as_float(words_4[15] & 4294901760u), acc_4[7]);
                acc_4[0] = __fmaf_rn(w_4[4], __uint_as_float(words_4[16] << 16), acc_4[0]);
                acc_4[1] = __fmaf_rn(w_4[4], __uint_as_float(words_4[16] & 4294901760u), acc_4[1]);
                acc_4[2] = __fmaf_rn(w_4[4], __uint_as_float(words_4[17] << 16), acc_4[2]);
                acc_4[3] = __fmaf_rn(w_4[4], __uint_as_float(words_4[17] & 4294901760u), acc_4[3]);
                acc_4[4] = __fmaf_rn(w_4[4], __uint_as_float(words_4[18] << 16), acc_4[4]);
                acc_4[5] = __fmaf_rn(w_4[4], __uint_as_float(words_4[18] & 4294901760u), acc_4[5]);
                acc_4[6] = __fmaf_rn(w_4[4], __uint_as_float(words_4[19] << 16), acc_4[6]);
                acc_4[7] = __fmaf_rn(w_4[4], __uint_as_float(words_4[19] & 4294901760u), acc_4[7]);
            }
            unsigned int packed_4[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_4[_lp*2 + 0], acc_4[_lp*2+1 + 0]));
                packed_4[_lp] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_109 = out_chunk;
                __builtin_assume(__isGlobal(_gst_109));
                reinterpret_cast<int4*>(_gst_109)[0] = reinterpret_cast<int4*>(packed_4)[0];
            }
        }
    }
    if (num_splits == 6) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        float lse_5[6];
        lse_5[0] = lse_row[0 * heads];
        lse_5[1] = lse_row[1 * heads];
        lse_5[2] = lse_row[2 * heads];
        lse_5[3] = lse_row[3 * heads];
        lse_5[4] = lse_row[4 * heads];
        lse_5[5] = lse_row[5 * heads];
        float m_5 = -CUDART_INF_F;
        m_5 = max_noftz(m_5, lse_5[0]);
        m_5 = max_noftz(m_5, lse_5[1]);
        m_5 = max_noftz(m_5, lse_5[2]);
        m_5 = max_noftz(m_5, lse_5[3]);
        m_5 = max_noftz(m_5, lse_5[4]);
        m_5 = max_noftz(m_5, lse_5[5]);
        if (m_5 != CUDART_INF_F || lse_5[0] != CUDART_INF_F) {
            float acc_5[8];
            acc_5[0] = 0.0f;
            acc_5[1] = 0.0f;
            acc_5[2] = 0.0f;
            acc_5[3] = 0.0f;
            acc_5[4] = 0.0f;
            acc_5[5] = 0.0f;
            acc_5[6] = 0.0f;
            acc_5[7] = 0.0f;
            if (m_5 != -CUDART_INF_F) {
                unsigned int words_5[24];
                words_5[0] = 0;
                words_5[1] = 0;
                words_5[2] = 0;
                words_5[3] = 0;
                if (lse_5[0] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_110 = reinterpret_cast<const uint4*>(o_chunk + 0 * o_stride + 0);
                        uint4* _vdst_110 = reinterpret_cast<uint4*>(&words_5[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_110[_blk] = _vptr_110[_blk];
                        }
                    }
                }
                words_5[4] = 0;
                words_5[5] = 0;
                words_5[6] = 0;
                words_5[7] = 0;
                if (lse_5[1] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_111 = reinterpret_cast<const uint4*>(o_chunk + 1 * o_stride + 0);
                        uint4* _vdst_111 = reinterpret_cast<uint4*>(&words_5[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_111[_blk] = _vptr_111[_blk];
                        }
                    }
                }
                words_5[8] = 0;
                words_5[9] = 0;
                words_5[10] = 0;
                words_5[11] = 0;
                if (lse_5[2] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_112 = reinterpret_cast<const uint4*>(o_chunk + 2 * o_stride + 0);
                        uint4* _vdst_112 = reinterpret_cast<uint4*>(&words_5[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_112[_blk] = _vptr_112[_blk];
                        }
                    }
                }
                words_5[12] = 0;
                words_5[13] = 0;
                words_5[14] = 0;
                words_5[15] = 0;
                if (lse_5[3] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_113 = reinterpret_cast<const uint4*>(o_chunk + 3 * o_stride + 0);
                        uint4* _vdst_113 = reinterpret_cast<uint4*>(&words_5[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_113[_blk] = _vptr_113[_blk];
                        }
                    }
                }
                words_5[16] = 0;
                words_5[17] = 0;
                words_5[18] = 0;
                words_5[19] = 0;
                if (lse_5[4] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_114 = reinterpret_cast<const uint4*>(o_chunk + 4 * o_stride + 0);
                        uint4* _vdst_114 = reinterpret_cast<uint4*>(&words_5[16]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_114[_blk] = _vptr_114[_blk];
                        }
                    }
                }
                words_5[20] = 0;
                words_5[21] = 0;
                words_5[22] = 0;
                words_5[23] = 0;
                if (lse_5[5] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_115 = reinterpret_cast<const uint4*>(o_chunk + 5 * o_stride + 0);
                        uint4* _vdst_115 = reinterpret_cast<uint4*>(&words_5[20]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_115[_blk] = _vptr_115[_blk];
                        }
                    }
                }
                float den_5 = 0.0f;
                float _sub_118;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_118) : "f"(lse_5[0]), "f"(m_5));
                float _exp2_117 = approx_exp2_noftz(_sub_118);
                float _add_116;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(den_5), "f"(_exp2_117));
                den_5 = _add_116;
                float _sub_121;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_121) : "f"(lse_5[1]), "f"(m_5));
                float _exp2_120 = approx_exp2_noftz(_sub_121);
                float _add_119;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(den_5), "f"(_exp2_120));
                den_5 = _add_119;
                float _sub_124;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_124) : "f"(lse_5[2]), "f"(m_5));
                float _exp2_123 = approx_exp2_noftz(_sub_124);
                float _add_122;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(den_5), "f"(_exp2_123));
                den_5 = _add_122;
                float _sub_127;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_127) : "f"(lse_5[3]), "f"(m_5));
                float _exp2_126 = approx_exp2_noftz(_sub_127);
                float _add_125;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(den_5), "f"(_exp2_126));
                den_5 = _add_125;
                float _sub_130;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_130) : "f"(lse_5[4]), "f"(m_5));
                float _exp2_129 = approx_exp2_noftz(_sub_130);
                float _add_128;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(den_5), "f"(_exp2_129));
                den_5 = _add_128;
                float _sub_133;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_133) : "f"(lse_5[5]), "f"(m_5));
                float _exp2_132 = approx_exp2_noftz(_sub_133);
                float _add_131;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(den_5), "f"(_exp2_132));
                den_5 = _add_131;
                float w_5[6];
                float _sub_136;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_136) : "f"(lse_5[0]), "f"(m_5));
                float _exp2_135 = approx_exp2_noftz(_sub_136);
                float _fdiv_134 = __fdiv_rn(_exp2_135, den_5);
                w_5[0] = _fdiv_134;
                float _sub_139;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_139) : "f"(lse_5[1]), "f"(m_5));
                float _exp2_138 = approx_exp2_noftz(_sub_139);
                float _fdiv_137 = __fdiv_rn(_exp2_138, den_5);
                w_5[1] = _fdiv_137;
                float _sub_142;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_142) : "f"(lse_5[2]), "f"(m_5));
                float _exp2_141 = approx_exp2_noftz(_sub_142);
                float _fdiv_140 = __fdiv_rn(_exp2_141, den_5);
                w_5[2] = _fdiv_140;
                float _sub_145;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_145) : "f"(lse_5[3]), "f"(m_5));
                float _exp2_144 = approx_exp2_noftz(_sub_145);
                float _fdiv_143 = __fdiv_rn(_exp2_144, den_5);
                w_5[3] = _fdiv_143;
                float _sub_148;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_148) : "f"(lse_5[4]), "f"(m_5));
                float _exp2_147 = approx_exp2_noftz(_sub_148);
                float _fdiv_146 = __fdiv_rn(_exp2_147, den_5);
                w_5[4] = _fdiv_146;
                float _sub_151;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_151) : "f"(lse_5[5]), "f"(m_5));
                float _exp2_150 = approx_exp2_noftz(_sub_151);
                float _fdiv_149 = __fdiv_rn(_exp2_150, den_5);
                w_5[5] = _fdiv_149;
                acc_5[0] = __fmaf_rn(w_5[0], __uint_as_float(words_5[0] << 16), acc_5[0]);
                acc_5[1] = __fmaf_rn(w_5[0], __uint_as_float(words_5[0] & 4294901760u), acc_5[1]);
                acc_5[2] = __fmaf_rn(w_5[0], __uint_as_float(words_5[1] << 16), acc_5[2]);
                acc_5[3] = __fmaf_rn(w_5[0], __uint_as_float(words_5[1] & 4294901760u), acc_5[3]);
                acc_5[4] = __fmaf_rn(w_5[0], __uint_as_float(words_5[2] << 16), acc_5[4]);
                acc_5[5] = __fmaf_rn(w_5[0], __uint_as_float(words_5[2] & 4294901760u), acc_5[5]);
                acc_5[6] = __fmaf_rn(w_5[0], __uint_as_float(words_5[3] << 16), acc_5[6]);
                acc_5[7] = __fmaf_rn(w_5[0], __uint_as_float(words_5[3] & 4294901760u), acc_5[7]);
                acc_5[0] = __fmaf_rn(w_5[1], __uint_as_float(words_5[4] << 16), acc_5[0]);
                acc_5[1] = __fmaf_rn(w_5[1], __uint_as_float(words_5[4] & 4294901760u), acc_5[1]);
                acc_5[2] = __fmaf_rn(w_5[1], __uint_as_float(words_5[5] << 16), acc_5[2]);
                acc_5[3] = __fmaf_rn(w_5[1], __uint_as_float(words_5[5] & 4294901760u), acc_5[3]);
                acc_5[4] = __fmaf_rn(w_5[1], __uint_as_float(words_5[6] << 16), acc_5[4]);
                acc_5[5] = __fmaf_rn(w_5[1], __uint_as_float(words_5[6] & 4294901760u), acc_5[5]);
                acc_5[6] = __fmaf_rn(w_5[1], __uint_as_float(words_5[7] << 16), acc_5[6]);
                acc_5[7] = __fmaf_rn(w_5[1], __uint_as_float(words_5[7] & 4294901760u), acc_5[7]);
                acc_5[0] = __fmaf_rn(w_5[2], __uint_as_float(words_5[8] << 16), acc_5[0]);
                acc_5[1] = __fmaf_rn(w_5[2], __uint_as_float(words_5[8] & 4294901760u), acc_5[1]);
                acc_5[2] = __fmaf_rn(w_5[2], __uint_as_float(words_5[9] << 16), acc_5[2]);
                acc_5[3] = __fmaf_rn(w_5[2], __uint_as_float(words_5[9] & 4294901760u), acc_5[3]);
                acc_5[4] = __fmaf_rn(w_5[2], __uint_as_float(words_5[10] << 16), acc_5[4]);
                acc_5[5] = __fmaf_rn(w_5[2], __uint_as_float(words_5[10] & 4294901760u), acc_5[5]);
                acc_5[6] = __fmaf_rn(w_5[2], __uint_as_float(words_5[11] << 16), acc_5[6]);
                acc_5[7] = __fmaf_rn(w_5[2], __uint_as_float(words_5[11] & 4294901760u), acc_5[7]);
                acc_5[0] = __fmaf_rn(w_5[3], __uint_as_float(words_5[12] << 16), acc_5[0]);
                acc_5[1] = __fmaf_rn(w_5[3], __uint_as_float(words_5[12] & 4294901760u), acc_5[1]);
                acc_5[2] = __fmaf_rn(w_5[3], __uint_as_float(words_5[13] << 16), acc_5[2]);
                acc_5[3] = __fmaf_rn(w_5[3], __uint_as_float(words_5[13] & 4294901760u), acc_5[3]);
                acc_5[4] = __fmaf_rn(w_5[3], __uint_as_float(words_5[14] << 16), acc_5[4]);
                acc_5[5] = __fmaf_rn(w_5[3], __uint_as_float(words_5[14] & 4294901760u), acc_5[5]);
                acc_5[6] = __fmaf_rn(w_5[3], __uint_as_float(words_5[15] << 16), acc_5[6]);
                acc_5[7] = __fmaf_rn(w_5[3], __uint_as_float(words_5[15] & 4294901760u), acc_5[7]);
                acc_5[0] = __fmaf_rn(w_5[4], __uint_as_float(words_5[16] << 16), acc_5[0]);
                acc_5[1] = __fmaf_rn(w_5[4], __uint_as_float(words_5[16] & 4294901760u), acc_5[1]);
                acc_5[2] = __fmaf_rn(w_5[4], __uint_as_float(words_5[17] << 16), acc_5[2]);
                acc_5[3] = __fmaf_rn(w_5[4], __uint_as_float(words_5[17] & 4294901760u), acc_5[3]);
                acc_5[4] = __fmaf_rn(w_5[4], __uint_as_float(words_5[18] << 16), acc_5[4]);
                acc_5[5] = __fmaf_rn(w_5[4], __uint_as_float(words_5[18] & 4294901760u), acc_5[5]);
                acc_5[6] = __fmaf_rn(w_5[4], __uint_as_float(words_5[19] << 16), acc_5[6]);
                acc_5[7] = __fmaf_rn(w_5[4], __uint_as_float(words_5[19] & 4294901760u), acc_5[7]);
                acc_5[0] = __fmaf_rn(w_5[5], __uint_as_float(words_5[20] << 16), acc_5[0]);
                acc_5[1] = __fmaf_rn(w_5[5], __uint_as_float(words_5[20] & 4294901760u), acc_5[1]);
                acc_5[2] = __fmaf_rn(w_5[5], __uint_as_float(words_5[21] << 16), acc_5[2]);
                acc_5[3] = __fmaf_rn(w_5[5], __uint_as_float(words_5[21] & 4294901760u), acc_5[3]);
                acc_5[4] = __fmaf_rn(w_5[5], __uint_as_float(words_5[22] << 16), acc_5[4]);
                acc_5[5] = __fmaf_rn(w_5[5], __uint_as_float(words_5[22] & 4294901760u), acc_5[5]);
                acc_5[6] = __fmaf_rn(w_5[5], __uint_as_float(words_5[23] << 16), acc_5[6]);
                acc_5[7] = __fmaf_rn(w_5[5], __uint_as_float(words_5[23] & 4294901760u), acc_5[7]);
            }
            unsigned int packed_5[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_5[_lp*2 + 0], acc_5[_lp*2+1 + 0]));
                packed_5[_lp] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_152 = out_chunk;
                __builtin_assume(__isGlobal(_gst_152));
                reinterpret_cast<int4*>(_gst_152)[0] = reinterpret_cast<int4*>(packed_5)[0];
            }
        }
    }
    if (num_splits == 7) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        float lse_6[7];
        lse_6[0] = lse_row[0 * heads];
        lse_6[1] = lse_row[1 * heads];
        lse_6[2] = lse_row[2 * heads];
        lse_6[3] = lse_row[3 * heads];
        lse_6[4] = lse_row[4 * heads];
        lse_6[5] = lse_row[5 * heads];
        lse_6[6] = lse_row[6 * heads];
        float m_6 = -CUDART_INF_F;
        m_6 = max_noftz(m_6, lse_6[0]);
        m_6 = max_noftz(m_6, lse_6[1]);
        m_6 = max_noftz(m_6, lse_6[2]);
        m_6 = max_noftz(m_6, lse_6[3]);
        m_6 = max_noftz(m_6, lse_6[4]);
        m_6 = max_noftz(m_6, lse_6[5]);
        m_6 = max_noftz(m_6, lse_6[6]);
        if (m_6 != CUDART_INF_F || lse_6[0] != CUDART_INF_F) {
            float acc_6[8];
            acc_6[0] = 0.0f;
            acc_6[1] = 0.0f;
            acc_6[2] = 0.0f;
            acc_6[3] = 0.0f;
            acc_6[4] = 0.0f;
            acc_6[5] = 0.0f;
            acc_6[6] = 0.0f;
            acc_6[7] = 0.0f;
            if (m_6 != -CUDART_INF_F) {
                unsigned int words_6[28];
                words_6[0] = 0;
                words_6[1] = 0;
                words_6[2] = 0;
                words_6[3] = 0;
                if (lse_6[0] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_153 = reinterpret_cast<const uint4*>(o_chunk + 0 * o_stride + 0);
                        uint4* _vdst_153 = reinterpret_cast<uint4*>(&words_6[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_153[_blk] = _vptr_153[_blk];
                        }
                    }
                }
                words_6[4] = 0;
                words_6[5] = 0;
                words_6[6] = 0;
                words_6[7] = 0;
                if (lse_6[1] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_154 = reinterpret_cast<const uint4*>(o_chunk + 1 * o_stride + 0);
                        uint4* _vdst_154 = reinterpret_cast<uint4*>(&words_6[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_154[_blk] = _vptr_154[_blk];
                        }
                    }
                }
                words_6[8] = 0;
                words_6[9] = 0;
                words_6[10] = 0;
                words_6[11] = 0;
                if (lse_6[2] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_155 = reinterpret_cast<const uint4*>(o_chunk + 2 * o_stride + 0);
                        uint4* _vdst_155 = reinterpret_cast<uint4*>(&words_6[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_155[_blk] = _vptr_155[_blk];
                        }
                    }
                }
                words_6[12] = 0;
                words_6[13] = 0;
                words_6[14] = 0;
                words_6[15] = 0;
                if (lse_6[3] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_156 = reinterpret_cast<const uint4*>(o_chunk + 3 * o_stride + 0);
                        uint4* _vdst_156 = reinterpret_cast<uint4*>(&words_6[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_156[_blk] = _vptr_156[_blk];
                        }
                    }
                }
                words_6[16] = 0;
                words_6[17] = 0;
                words_6[18] = 0;
                words_6[19] = 0;
                if (lse_6[4] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_157 = reinterpret_cast<const uint4*>(o_chunk + 4 * o_stride + 0);
                        uint4* _vdst_157 = reinterpret_cast<uint4*>(&words_6[16]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_157[_blk] = _vptr_157[_blk];
                        }
                    }
                }
                words_6[20] = 0;
                words_6[21] = 0;
                words_6[22] = 0;
                words_6[23] = 0;
                if (lse_6[5] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_158 = reinterpret_cast<const uint4*>(o_chunk + 5 * o_stride + 0);
                        uint4* _vdst_158 = reinterpret_cast<uint4*>(&words_6[20]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_158[_blk] = _vptr_158[_blk];
                        }
                    }
                }
                words_6[24] = 0;
                words_6[25] = 0;
                words_6[26] = 0;
                words_6[27] = 0;
                if (lse_6[6] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_159 = reinterpret_cast<const uint4*>(o_chunk + 6 * o_stride + 0);
                        uint4* _vdst_159 = reinterpret_cast<uint4*>(&words_6[24]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_159[_blk] = _vptr_159[_blk];
                        }
                    }
                }
                float den_6 = 0.0f;
                float _sub_162;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_162) : "f"(lse_6[0]), "f"(m_6));
                float _exp2_161 = approx_exp2_noftz(_sub_162);
                float _add_160;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(den_6), "f"(_exp2_161));
                den_6 = _add_160;
                float _sub_165;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_165) : "f"(lse_6[1]), "f"(m_6));
                float _exp2_164 = approx_exp2_noftz(_sub_165);
                float _add_163;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(den_6), "f"(_exp2_164));
                den_6 = _add_163;
                float _sub_168;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_168) : "f"(lse_6[2]), "f"(m_6));
                float _exp2_167 = approx_exp2_noftz(_sub_168);
                float _add_166;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_166) : "f"(den_6), "f"(_exp2_167));
                den_6 = _add_166;
                float _sub_171;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_171) : "f"(lse_6[3]), "f"(m_6));
                float _exp2_170 = approx_exp2_noftz(_sub_171);
                float _add_169;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(den_6), "f"(_exp2_170));
                den_6 = _add_169;
                float _sub_174;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_174) : "f"(lse_6[4]), "f"(m_6));
                float _exp2_173 = approx_exp2_noftz(_sub_174);
                float _add_172;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(den_6), "f"(_exp2_173));
                den_6 = _add_172;
                float _sub_177;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_177) : "f"(lse_6[5]), "f"(m_6));
                float _exp2_176 = approx_exp2_noftz(_sub_177);
                float _add_175;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_175) : "f"(den_6), "f"(_exp2_176));
                den_6 = _add_175;
                float _sub_180;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_180) : "f"(lse_6[6]), "f"(m_6));
                float _exp2_179 = approx_exp2_noftz(_sub_180);
                float _add_178;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_178) : "f"(den_6), "f"(_exp2_179));
                den_6 = _add_178;
                float w_6[7];
                float _sub_183;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_183) : "f"(lse_6[0]), "f"(m_6));
                float _exp2_182 = approx_exp2_noftz(_sub_183);
                float _fdiv_181 = __fdiv_rn(_exp2_182, den_6);
                w_6[0] = _fdiv_181;
                float _sub_186;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_186) : "f"(lse_6[1]), "f"(m_6));
                float _exp2_185 = approx_exp2_noftz(_sub_186);
                float _fdiv_184 = __fdiv_rn(_exp2_185, den_6);
                w_6[1] = _fdiv_184;
                float _sub_189;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_189) : "f"(lse_6[2]), "f"(m_6));
                float _exp2_188 = approx_exp2_noftz(_sub_189);
                float _fdiv_187 = __fdiv_rn(_exp2_188, den_6);
                w_6[2] = _fdiv_187;
                float _sub_192;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_192) : "f"(lse_6[3]), "f"(m_6));
                float _exp2_191 = approx_exp2_noftz(_sub_192);
                float _fdiv_190 = __fdiv_rn(_exp2_191, den_6);
                w_6[3] = _fdiv_190;
                float _sub_195;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_195) : "f"(lse_6[4]), "f"(m_6));
                float _exp2_194 = approx_exp2_noftz(_sub_195);
                float _fdiv_193 = __fdiv_rn(_exp2_194, den_6);
                w_6[4] = _fdiv_193;
                float _sub_198;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_198) : "f"(lse_6[5]), "f"(m_6));
                float _exp2_197 = approx_exp2_noftz(_sub_198);
                float _fdiv_196 = __fdiv_rn(_exp2_197, den_6);
                w_6[5] = _fdiv_196;
                float _sub_201;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_201) : "f"(lse_6[6]), "f"(m_6));
                float _exp2_200 = approx_exp2_noftz(_sub_201);
                float _fdiv_199 = __fdiv_rn(_exp2_200, den_6);
                w_6[6] = _fdiv_199;
                acc_6[0] = __fmaf_rn(w_6[0], __uint_as_float(words_6[0] << 16), acc_6[0]);
                acc_6[1] = __fmaf_rn(w_6[0], __uint_as_float(words_6[0] & 4294901760u), acc_6[1]);
                acc_6[2] = __fmaf_rn(w_6[0], __uint_as_float(words_6[1] << 16), acc_6[2]);
                acc_6[3] = __fmaf_rn(w_6[0], __uint_as_float(words_6[1] & 4294901760u), acc_6[3]);
                acc_6[4] = __fmaf_rn(w_6[0], __uint_as_float(words_6[2] << 16), acc_6[4]);
                acc_6[5] = __fmaf_rn(w_6[0], __uint_as_float(words_6[2] & 4294901760u), acc_6[5]);
                acc_6[6] = __fmaf_rn(w_6[0], __uint_as_float(words_6[3] << 16), acc_6[6]);
                acc_6[7] = __fmaf_rn(w_6[0], __uint_as_float(words_6[3] & 4294901760u), acc_6[7]);
                acc_6[0] = __fmaf_rn(w_6[1], __uint_as_float(words_6[4] << 16), acc_6[0]);
                acc_6[1] = __fmaf_rn(w_6[1], __uint_as_float(words_6[4] & 4294901760u), acc_6[1]);
                acc_6[2] = __fmaf_rn(w_6[1], __uint_as_float(words_6[5] << 16), acc_6[2]);
                acc_6[3] = __fmaf_rn(w_6[1], __uint_as_float(words_6[5] & 4294901760u), acc_6[3]);
                acc_6[4] = __fmaf_rn(w_6[1], __uint_as_float(words_6[6] << 16), acc_6[4]);
                acc_6[5] = __fmaf_rn(w_6[1], __uint_as_float(words_6[6] & 4294901760u), acc_6[5]);
                acc_6[6] = __fmaf_rn(w_6[1], __uint_as_float(words_6[7] << 16), acc_6[6]);
                acc_6[7] = __fmaf_rn(w_6[1], __uint_as_float(words_6[7] & 4294901760u), acc_6[7]);
                acc_6[0] = __fmaf_rn(w_6[2], __uint_as_float(words_6[8] << 16), acc_6[0]);
                acc_6[1] = __fmaf_rn(w_6[2], __uint_as_float(words_6[8] & 4294901760u), acc_6[1]);
                acc_6[2] = __fmaf_rn(w_6[2], __uint_as_float(words_6[9] << 16), acc_6[2]);
                acc_6[3] = __fmaf_rn(w_6[2], __uint_as_float(words_6[9] & 4294901760u), acc_6[3]);
                acc_6[4] = __fmaf_rn(w_6[2], __uint_as_float(words_6[10] << 16), acc_6[4]);
                acc_6[5] = __fmaf_rn(w_6[2], __uint_as_float(words_6[10] & 4294901760u), acc_6[5]);
                acc_6[6] = __fmaf_rn(w_6[2], __uint_as_float(words_6[11] << 16), acc_6[6]);
                acc_6[7] = __fmaf_rn(w_6[2], __uint_as_float(words_6[11] & 4294901760u), acc_6[7]);
                acc_6[0] = __fmaf_rn(w_6[3], __uint_as_float(words_6[12] << 16), acc_6[0]);
                acc_6[1] = __fmaf_rn(w_6[3], __uint_as_float(words_6[12] & 4294901760u), acc_6[1]);
                acc_6[2] = __fmaf_rn(w_6[3], __uint_as_float(words_6[13] << 16), acc_6[2]);
                acc_6[3] = __fmaf_rn(w_6[3], __uint_as_float(words_6[13] & 4294901760u), acc_6[3]);
                acc_6[4] = __fmaf_rn(w_6[3], __uint_as_float(words_6[14] << 16), acc_6[4]);
                acc_6[5] = __fmaf_rn(w_6[3], __uint_as_float(words_6[14] & 4294901760u), acc_6[5]);
                acc_6[6] = __fmaf_rn(w_6[3], __uint_as_float(words_6[15] << 16), acc_6[6]);
                acc_6[7] = __fmaf_rn(w_6[3], __uint_as_float(words_6[15] & 4294901760u), acc_6[7]);
                acc_6[0] = __fmaf_rn(w_6[4], __uint_as_float(words_6[16] << 16), acc_6[0]);
                acc_6[1] = __fmaf_rn(w_6[4], __uint_as_float(words_6[16] & 4294901760u), acc_6[1]);
                acc_6[2] = __fmaf_rn(w_6[4], __uint_as_float(words_6[17] << 16), acc_6[2]);
                acc_6[3] = __fmaf_rn(w_6[4], __uint_as_float(words_6[17] & 4294901760u), acc_6[3]);
                acc_6[4] = __fmaf_rn(w_6[4], __uint_as_float(words_6[18] << 16), acc_6[4]);
                acc_6[5] = __fmaf_rn(w_6[4], __uint_as_float(words_6[18] & 4294901760u), acc_6[5]);
                acc_6[6] = __fmaf_rn(w_6[4], __uint_as_float(words_6[19] << 16), acc_6[6]);
                acc_6[7] = __fmaf_rn(w_6[4], __uint_as_float(words_6[19] & 4294901760u), acc_6[7]);
                acc_6[0] = __fmaf_rn(w_6[5], __uint_as_float(words_6[20] << 16), acc_6[0]);
                acc_6[1] = __fmaf_rn(w_6[5], __uint_as_float(words_6[20] & 4294901760u), acc_6[1]);
                acc_6[2] = __fmaf_rn(w_6[5], __uint_as_float(words_6[21] << 16), acc_6[2]);
                acc_6[3] = __fmaf_rn(w_6[5], __uint_as_float(words_6[21] & 4294901760u), acc_6[3]);
                acc_6[4] = __fmaf_rn(w_6[5], __uint_as_float(words_6[22] << 16), acc_6[4]);
                acc_6[5] = __fmaf_rn(w_6[5], __uint_as_float(words_6[22] & 4294901760u), acc_6[5]);
                acc_6[6] = __fmaf_rn(w_6[5], __uint_as_float(words_6[23] << 16), acc_6[6]);
                acc_6[7] = __fmaf_rn(w_6[5], __uint_as_float(words_6[23] & 4294901760u), acc_6[7]);
                acc_6[0] = __fmaf_rn(w_6[6], __uint_as_float(words_6[24] << 16), acc_6[0]);
                acc_6[1] = __fmaf_rn(w_6[6], __uint_as_float(words_6[24] & 4294901760u), acc_6[1]);
                acc_6[2] = __fmaf_rn(w_6[6], __uint_as_float(words_6[25] << 16), acc_6[2]);
                acc_6[3] = __fmaf_rn(w_6[6], __uint_as_float(words_6[25] & 4294901760u), acc_6[3]);
                acc_6[4] = __fmaf_rn(w_6[6], __uint_as_float(words_6[26] << 16), acc_6[4]);
                acc_6[5] = __fmaf_rn(w_6[6], __uint_as_float(words_6[26] & 4294901760u), acc_6[5]);
                acc_6[6] = __fmaf_rn(w_6[6], __uint_as_float(words_6[27] << 16), acc_6[6]);
                acc_6[7] = __fmaf_rn(w_6[6], __uint_as_float(words_6[27] & 4294901760u), acc_6[7]);
            }
            unsigned int packed_6[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_6[_lp*2 + 0], acc_6[_lp*2+1 + 0]));
                packed_6[_lp] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_202 = out_chunk;
                __builtin_assume(__isGlobal(_gst_202));
                reinterpret_cast<int4*>(_gst_202)[0] = reinterpret_cast<int4*>(packed_6)[0];
            }
        }
    }
    if (num_splits == 8) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        float lse_7[8];
        lse_7[0] = lse_row[0 * heads];
        lse_7[1] = lse_row[1 * heads];
        lse_7[2] = lse_row[2 * heads];
        lse_7[3] = lse_row[3 * heads];
        lse_7[4] = lse_row[4 * heads];
        lse_7[5] = lse_row[5 * heads];
        lse_7[6] = lse_row[6 * heads];
        lse_7[7] = lse_row[7 * heads];
        float m_7 = -CUDART_INF_F;
        m_7 = max_noftz(m_7, lse_7[0]);
        m_7 = max_noftz(m_7, lse_7[1]);
        m_7 = max_noftz(m_7, lse_7[2]);
        m_7 = max_noftz(m_7, lse_7[3]);
        m_7 = max_noftz(m_7, lse_7[4]);
        m_7 = max_noftz(m_7, lse_7[5]);
        m_7 = max_noftz(m_7, lse_7[6]);
        m_7 = max_noftz(m_7, lse_7[7]);
        if (m_7 != CUDART_INF_F || lse_7[0] != CUDART_INF_F) {
            float acc_7[8];
            acc_7[0] = 0.0f;
            acc_7[1] = 0.0f;
            acc_7[2] = 0.0f;
            acc_7[3] = 0.0f;
            acc_7[4] = 0.0f;
            acc_7[5] = 0.0f;
            acc_7[6] = 0.0f;
            acc_7[7] = 0.0f;
            if (m_7 != -CUDART_INF_F) {
                unsigned int words_7[32];
                words_7[0] = 0;
                words_7[1] = 0;
                words_7[2] = 0;
                words_7[3] = 0;
                if (lse_7[0] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_203 = reinterpret_cast<const uint4*>(o_chunk + 0 * o_stride + 0);
                        uint4* _vdst_203 = reinterpret_cast<uint4*>(&words_7[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_203[_blk] = _vptr_203[_blk];
                        }
                    }
                }
                words_7[4] = 0;
                words_7[5] = 0;
                words_7[6] = 0;
                words_7[7] = 0;
                if (lse_7[1] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_204 = reinterpret_cast<const uint4*>(o_chunk + 1 * o_stride + 0);
                        uint4* _vdst_204 = reinterpret_cast<uint4*>(&words_7[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_204[_blk] = _vptr_204[_blk];
                        }
                    }
                }
                words_7[8] = 0;
                words_7[9] = 0;
                words_7[10] = 0;
                words_7[11] = 0;
                if (lse_7[2] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_205 = reinterpret_cast<const uint4*>(o_chunk + 2 * o_stride + 0);
                        uint4* _vdst_205 = reinterpret_cast<uint4*>(&words_7[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_205[_blk] = _vptr_205[_blk];
                        }
                    }
                }
                words_7[12] = 0;
                words_7[13] = 0;
                words_7[14] = 0;
                words_7[15] = 0;
                if (lse_7[3] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_206 = reinterpret_cast<const uint4*>(o_chunk + 3 * o_stride + 0);
                        uint4* _vdst_206 = reinterpret_cast<uint4*>(&words_7[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_206[_blk] = _vptr_206[_blk];
                        }
                    }
                }
                words_7[16] = 0;
                words_7[17] = 0;
                words_7[18] = 0;
                words_7[19] = 0;
                if (lse_7[4] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_207 = reinterpret_cast<const uint4*>(o_chunk + 4 * o_stride + 0);
                        uint4* _vdst_207 = reinterpret_cast<uint4*>(&words_7[16]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_207[_blk] = _vptr_207[_blk];
                        }
                    }
                }
                words_7[20] = 0;
                words_7[21] = 0;
                words_7[22] = 0;
                words_7[23] = 0;
                if (lse_7[5] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_208 = reinterpret_cast<const uint4*>(o_chunk + 5 * o_stride + 0);
                        uint4* _vdst_208 = reinterpret_cast<uint4*>(&words_7[20]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_208[_blk] = _vptr_208[_blk];
                        }
                    }
                }
                words_7[24] = 0;
                words_7[25] = 0;
                words_7[26] = 0;
                words_7[27] = 0;
                if (lse_7[6] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_209 = reinterpret_cast<const uint4*>(o_chunk + 6 * o_stride + 0);
                        uint4* _vdst_209 = reinterpret_cast<uint4*>(&words_7[24]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_209[_blk] = _vptr_209[_blk];
                        }
                    }
                }
                words_7[28] = 0;
                words_7[29] = 0;
                words_7[30] = 0;
                words_7[31] = 0;
                if (lse_7[7] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_210 = reinterpret_cast<const uint4*>(o_chunk + 7 * o_stride + 0);
                        uint4* _vdst_210 = reinterpret_cast<uint4*>(&words_7[28]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_210[_blk] = _vptr_210[_blk];
                        }
                    }
                }
                float den_7 = 0.0f;
                float _sub_213;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_213) : "f"(lse_7[0]), "f"(m_7));
                float _exp2_212 = approx_exp2_noftz(_sub_213);
                float _add_211;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_211) : "f"(den_7), "f"(_exp2_212));
                den_7 = _add_211;
                float _sub_216;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_216) : "f"(lse_7[1]), "f"(m_7));
                float _exp2_215 = approx_exp2_noftz(_sub_216);
                float _add_214;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_214) : "f"(den_7), "f"(_exp2_215));
                den_7 = _add_214;
                float _sub_219;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_219) : "f"(lse_7[2]), "f"(m_7));
                float _exp2_218 = approx_exp2_noftz(_sub_219);
                float _add_217;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_217) : "f"(den_7), "f"(_exp2_218));
                den_7 = _add_217;
                float _sub_222;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_222) : "f"(lse_7[3]), "f"(m_7));
                float _exp2_221 = approx_exp2_noftz(_sub_222);
                float _add_220;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_220) : "f"(den_7), "f"(_exp2_221));
                den_7 = _add_220;
                float _sub_225;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_225) : "f"(lse_7[4]), "f"(m_7));
                float _exp2_224 = approx_exp2_noftz(_sub_225);
                float _add_223;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_223) : "f"(den_7), "f"(_exp2_224));
                den_7 = _add_223;
                float _sub_228;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_228) : "f"(lse_7[5]), "f"(m_7));
                float _exp2_227 = approx_exp2_noftz(_sub_228);
                float _add_226;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_226) : "f"(den_7), "f"(_exp2_227));
                den_7 = _add_226;
                float _sub_231;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_231) : "f"(lse_7[6]), "f"(m_7));
                float _exp2_230 = approx_exp2_noftz(_sub_231);
                float _add_229;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_229) : "f"(den_7), "f"(_exp2_230));
                den_7 = _add_229;
                float _sub_234;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_234) : "f"(lse_7[7]), "f"(m_7));
                float _exp2_233 = approx_exp2_noftz(_sub_234);
                float _add_232;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_232) : "f"(den_7), "f"(_exp2_233));
                den_7 = _add_232;
                float w_7[8];
                float _sub_237;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_237) : "f"(lse_7[0]), "f"(m_7));
                float _exp2_236 = approx_exp2_noftz(_sub_237);
                float _fdiv_235 = __fdiv_rn(_exp2_236, den_7);
                w_7[0] = _fdiv_235;
                float _sub_240;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_240) : "f"(lse_7[1]), "f"(m_7));
                float _exp2_239 = approx_exp2_noftz(_sub_240);
                float _fdiv_238 = __fdiv_rn(_exp2_239, den_7);
                w_7[1] = _fdiv_238;
                float _sub_243;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_243) : "f"(lse_7[2]), "f"(m_7));
                float _exp2_242 = approx_exp2_noftz(_sub_243);
                float _fdiv_241 = __fdiv_rn(_exp2_242, den_7);
                w_7[2] = _fdiv_241;
                float _sub_246;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_246) : "f"(lse_7[3]), "f"(m_7));
                float _exp2_245 = approx_exp2_noftz(_sub_246);
                float _fdiv_244 = __fdiv_rn(_exp2_245, den_7);
                w_7[3] = _fdiv_244;
                float _sub_249;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_249) : "f"(lse_7[4]), "f"(m_7));
                float _exp2_248 = approx_exp2_noftz(_sub_249);
                float _fdiv_247 = __fdiv_rn(_exp2_248, den_7);
                w_7[4] = _fdiv_247;
                float _sub_252;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_252) : "f"(lse_7[5]), "f"(m_7));
                float _exp2_251 = approx_exp2_noftz(_sub_252);
                float _fdiv_250 = __fdiv_rn(_exp2_251, den_7);
                w_7[5] = _fdiv_250;
                float _sub_255;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_255) : "f"(lse_7[6]), "f"(m_7));
                float _exp2_254 = approx_exp2_noftz(_sub_255);
                float _fdiv_253 = __fdiv_rn(_exp2_254, den_7);
                w_7[6] = _fdiv_253;
                float _sub_258;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_258) : "f"(lse_7[7]), "f"(m_7));
                float _exp2_257 = approx_exp2_noftz(_sub_258);
                float _fdiv_256 = __fdiv_rn(_exp2_257, den_7);
                w_7[7] = _fdiv_256;
                acc_7[0] = __fmaf_rn(w_7[0], __uint_as_float(words_7[0] << 16), acc_7[0]);
                acc_7[1] = __fmaf_rn(w_7[0], __uint_as_float(words_7[0] & 4294901760u), acc_7[1]);
                acc_7[2] = __fmaf_rn(w_7[0], __uint_as_float(words_7[1] << 16), acc_7[2]);
                acc_7[3] = __fmaf_rn(w_7[0], __uint_as_float(words_7[1] & 4294901760u), acc_7[3]);
                acc_7[4] = __fmaf_rn(w_7[0], __uint_as_float(words_7[2] << 16), acc_7[4]);
                acc_7[5] = __fmaf_rn(w_7[0], __uint_as_float(words_7[2] & 4294901760u), acc_7[5]);
                acc_7[6] = __fmaf_rn(w_7[0], __uint_as_float(words_7[3] << 16), acc_7[6]);
                acc_7[7] = __fmaf_rn(w_7[0], __uint_as_float(words_7[3] & 4294901760u), acc_7[7]);
                acc_7[0] = __fmaf_rn(w_7[1], __uint_as_float(words_7[4] << 16), acc_7[0]);
                acc_7[1] = __fmaf_rn(w_7[1], __uint_as_float(words_7[4] & 4294901760u), acc_7[1]);
                acc_7[2] = __fmaf_rn(w_7[1], __uint_as_float(words_7[5] << 16), acc_7[2]);
                acc_7[3] = __fmaf_rn(w_7[1], __uint_as_float(words_7[5] & 4294901760u), acc_7[3]);
                acc_7[4] = __fmaf_rn(w_7[1], __uint_as_float(words_7[6] << 16), acc_7[4]);
                acc_7[5] = __fmaf_rn(w_7[1], __uint_as_float(words_7[6] & 4294901760u), acc_7[5]);
                acc_7[6] = __fmaf_rn(w_7[1], __uint_as_float(words_7[7] << 16), acc_7[6]);
                acc_7[7] = __fmaf_rn(w_7[1], __uint_as_float(words_7[7] & 4294901760u), acc_7[7]);
                acc_7[0] = __fmaf_rn(w_7[2], __uint_as_float(words_7[8] << 16), acc_7[0]);
                acc_7[1] = __fmaf_rn(w_7[2], __uint_as_float(words_7[8] & 4294901760u), acc_7[1]);
                acc_7[2] = __fmaf_rn(w_7[2], __uint_as_float(words_7[9] << 16), acc_7[2]);
                acc_7[3] = __fmaf_rn(w_7[2], __uint_as_float(words_7[9] & 4294901760u), acc_7[3]);
                acc_7[4] = __fmaf_rn(w_7[2], __uint_as_float(words_7[10] << 16), acc_7[4]);
                acc_7[5] = __fmaf_rn(w_7[2], __uint_as_float(words_7[10] & 4294901760u), acc_7[5]);
                acc_7[6] = __fmaf_rn(w_7[2], __uint_as_float(words_7[11] << 16), acc_7[6]);
                acc_7[7] = __fmaf_rn(w_7[2], __uint_as_float(words_7[11] & 4294901760u), acc_7[7]);
                acc_7[0] = __fmaf_rn(w_7[3], __uint_as_float(words_7[12] << 16), acc_7[0]);
                acc_7[1] = __fmaf_rn(w_7[3], __uint_as_float(words_7[12] & 4294901760u), acc_7[1]);
                acc_7[2] = __fmaf_rn(w_7[3], __uint_as_float(words_7[13] << 16), acc_7[2]);
                acc_7[3] = __fmaf_rn(w_7[3], __uint_as_float(words_7[13] & 4294901760u), acc_7[3]);
                acc_7[4] = __fmaf_rn(w_7[3], __uint_as_float(words_7[14] << 16), acc_7[4]);
                acc_7[5] = __fmaf_rn(w_7[3], __uint_as_float(words_7[14] & 4294901760u), acc_7[5]);
                acc_7[6] = __fmaf_rn(w_7[3], __uint_as_float(words_7[15] << 16), acc_7[6]);
                acc_7[7] = __fmaf_rn(w_7[3], __uint_as_float(words_7[15] & 4294901760u), acc_7[7]);
                acc_7[0] = __fmaf_rn(w_7[4], __uint_as_float(words_7[16] << 16), acc_7[0]);
                acc_7[1] = __fmaf_rn(w_7[4], __uint_as_float(words_7[16] & 4294901760u), acc_7[1]);
                acc_7[2] = __fmaf_rn(w_7[4], __uint_as_float(words_7[17] << 16), acc_7[2]);
                acc_7[3] = __fmaf_rn(w_7[4], __uint_as_float(words_7[17] & 4294901760u), acc_7[3]);
                acc_7[4] = __fmaf_rn(w_7[4], __uint_as_float(words_7[18] << 16), acc_7[4]);
                acc_7[5] = __fmaf_rn(w_7[4], __uint_as_float(words_7[18] & 4294901760u), acc_7[5]);
                acc_7[6] = __fmaf_rn(w_7[4], __uint_as_float(words_7[19] << 16), acc_7[6]);
                acc_7[7] = __fmaf_rn(w_7[4], __uint_as_float(words_7[19] & 4294901760u), acc_7[7]);
                acc_7[0] = __fmaf_rn(w_7[5], __uint_as_float(words_7[20] << 16), acc_7[0]);
                acc_7[1] = __fmaf_rn(w_7[5], __uint_as_float(words_7[20] & 4294901760u), acc_7[1]);
                acc_7[2] = __fmaf_rn(w_7[5], __uint_as_float(words_7[21] << 16), acc_7[2]);
                acc_7[3] = __fmaf_rn(w_7[5], __uint_as_float(words_7[21] & 4294901760u), acc_7[3]);
                acc_7[4] = __fmaf_rn(w_7[5], __uint_as_float(words_7[22] << 16), acc_7[4]);
                acc_7[5] = __fmaf_rn(w_7[5], __uint_as_float(words_7[22] & 4294901760u), acc_7[5]);
                acc_7[6] = __fmaf_rn(w_7[5], __uint_as_float(words_7[23] << 16), acc_7[6]);
                acc_7[7] = __fmaf_rn(w_7[5], __uint_as_float(words_7[23] & 4294901760u), acc_7[7]);
                acc_7[0] = __fmaf_rn(w_7[6], __uint_as_float(words_7[24] << 16), acc_7[0]);
                acc_7[1] = __fmaf_rn(w_7[6], __uint_as_float(words_7[24] & 4294901760u), acc_7[1]);
                acc_7[2] = __fmaf_rn(w_7[6], __uint_as_float(words_7[25] << 16), acc_7[2]);
                acc_7[3] = __fmaf_rn(w_7[6], __uint_as_float(words_7[25] & 4294901760u), acc_7[3]);
                acc_7[4] = __fmaf_rn(w_7[6], __uint_as_float(words_7[26] << 16), acc_7[4]);
                acc_7[5] = __fmaf_rn(w_7[6], __uint_as_float(words_7[26] & 4294901760u), acc_7[5]);
                acc_7[6] = __fmaf_rn(w_7[6], __uint_as_float(words_7[27] << 16), acc_7[6]);
                acc_7[7] = __fmaf_rn(w_7[6], __uint_as_float(words_7[27] & 4294901760u), acc_7[7]);
                acc_7[0] = __fmaf_rn(w_7[7], __uint_as_float(words_7[28] << 16), acc_7[0]);
                acc_7[1] = __fmaf_rn(w_7[7], __uint_as_float(words_7[28] & 4294901760u), acc_7[1]);
                acc_7[2] = __fmaf_rn(w_7[7], __uint_as_float(words_7[29] << 16), acc_7[2]);
                acc_7[3] = __fmaf_rn(w_7[7], __uint_as_float(words_7[29] & 4294901760u), acc_7[3]);
                acc_7[4] = __fmaf_rn(w_7[7], __uint_as_float(words_7[30] << 16), acc_7[4]);
                acc_7[5] = __fmaf_rn(w_7[7], __uint_as_float(words_7[30] & 4294901760u), acc_7[5]);
                acc_7[6] = __fmaf_rn(w_7[7], __uint_as_float(words_7[31] << 16), acc_7[6]);
                acc_7[7] = __fmaf_rn(w_7[7], __uint_as_float(words_7[31] & 4294901760u), acc_7[7]);
            }
            unsigned int packed_7[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_7[_lp*2 + 0], acc_7[_lp*2+1 + 0]));
                packed_7[_lp] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_259 = out_chunk;
                __builtin_assume(__isGlobal(_gst_259));
                reinterpret_cast<int4*>(_gst_259)[0] = reinterpret_cast<int4*>(packed_7)[0];
            }
        }
    }
    if (num_splits == 9) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        float lse_8[9];
        lse_8[0] = lse_row[0 * heads];
        lse_8[1] = lse_row[1 * heads];
        lse_8[2] = lse_row[2 * heads];
        lse_8[3] = lse_row[3 * heads];
        lse_8[4] = lse_row[4 * heads];
        lse_8[5] = lse_row[5 * heads];
        lse_8[6] = lse_row[6 * heads];
        lse_8[7] = lse_row[7 * heads];
        lse_8[8] = lse_row[8 * heads];
        float m_8 = -CUDART_INF_F;
        m_8 = max_noftz(m_8, lse_8[0]);
        m_8 = max_noftz(m_8, lse_8[1]);
        m_8 = max_noftz(m_8, lse_8[2]);
        m_8 = max_noftz(m_8, lse_8[3]);
        m_8 = max_noftz(m_8, lse_8[4]);
        m_8 = max_noftz(m_8, lse_8[5]);
        m_8 = max_noftz(m_8, lse_8[6]);
        m_8 = max_noftz(m_8, lse_8[7]);
        m_8 = max_noftz(m_8, lse_8[8]);
        if (m_8 != CUDART_INF_F || lse_8[0] != CUDART_INF_F) {
            float acc_8[8];
            acc_8[0] = 0.0f;
            acc_8[1] = 0.0f;
            acc_8[2] = 0.0f;
            acc_8[3] = 0.0f;
            acc_8[4] = 0.0f;
            acc_8[5] = 0.0f;
            acc_8[6] = 0.0f;
            acc_8[7] = 0.0f;
            if (m_8 != -CUDART_INF_F) {
                unsigned int words_8[36];
                words_8[0] = 0;
                words_8[1] = 0;
                words_8[2] = 0;
                words_8[3] = 0;
                if (lse_8[0] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_260 = reinterpret_cast<const uint4*>(o_chunk + 0 * o_stride + 0);
                        uint4* _vdst_260 = reinterpret_cast<uint4*>(&words_8[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_260[_blk] = _vptr_260[_blk];
                        }
                    }
                }
                words_8[4] = 0;
                words_8[5] = 0;
                words_8[6] = 0;
                words_8[7] = 0;
                if (lse_8[1] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_261 = reinterpret_cast<const uint4*>(o_chunk + 1 * o_stride + 0);
                        uint4* _vdst_261 = reinterpret_cast<uint4*>(&words_8[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_261[_blk] = _vptr_261[_blk];
                        }
                    }
                }
                words_8[8] = 0;
                words_8[9] = 0;
                words_8[10] = 0;
                words_8[11] = 0;
                if (lse_8[2] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_262 = reinterpret_cast<const uint4*>(o_chunk + 2 * o_stride + 0);
                        uint4* _vdst_262 = reinterpret_cast<uint4*>(&words_8[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_262[_blk] = _vptr_262[_blk];
                        }
                    }
                }
                words_8[12] = 0;
                words_8[13] = 0;
                words_8[14] = 0;
                words_8[15] = 0;
                if (lse_8[3] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_263 = reinterpret_cast<const uint4*>(o_chunk + 3 * o_stride + 0);
                        uint4* _vdst_263 = reinterpret_cast<uint4*>(&words_8[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_263[_blk] = _vptr_263[_blk];
                        }
                    }
                }
                words_8[16] = 0;
                words_8[17] = 0;
                words_8[18] = 0;
                words_8[19] = 0;
                if (lse_8[4] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_264 = reinterpret_cast<const uint4*>(o_chunk + 4 * o_stride + 0);
                        uint4* _vdst_264 = reinterpret_cast<uint4*>(&words_8[16]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_264[_blk] = _vptr_264[_blk];
                        }
                    }
                }
                words_8[20] = 0;
                words_8[21] = 0;
                words_8[22] = 0;
                words_8[23] = 0;
                if (lse_8[5] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_265 = reinterpret_cast<const uint4*>(o_chunk + 5 * o_stride + 0);
                        uint4* _vdst_265 = reinterpret_cast<uint4*>(&words_8[20]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_265[_blk] = _vptr_265[_blk];
                        }
                    }
                }
                words_8[24] = 0;
                words_8[25] = 0;
                words_8[26] = 0;
                words_8[27] = 0;
                if (lse_8[6] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_266 = reinterpret_cast<const uint4*>(o_chunk + 6 * o_stride + 0);
                        uint4* _vdst_266 = reinterpret_cast<uint4*>(&words_8[24]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_266[_blk] = _vptr_266[_blk];
                        }
                    }
                }
                words_8[28] = 0;
                words_8[29] = 0;
                words_8[30] = 0;
                words_8[31] = 0;
                if (lse_8[7] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_267 = reinterpret_cast<const uint4*>(o_chunk + 7 * o_stride + 0);
                        uint4* _vdst_267 = reinterpret_cast<uint4*>(&words_8[28]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_267[_blk] = _vptr_267[_blk];
                        }
                    }
                }
                words_8[32] = 0;
                words_8[33] = 0;
                words_8[34] = 0;
                words_8[35] = 0;
                if (lse_8[8] != -CUDART_INF_F) {
                    {
                        const uint4* _vptr_268 = reinterpret_cast<const uint4*>(o_chunk + 8 * o_stride + 0);
                        uint4* _vdst_268 = reinterpret_cast<uint4*>(&words_8[32]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_268[_blk] = _vptr_268[_blk];
                        }
                    }
                }
                float den_8 = 0.0f;
                float _sub_271;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_271) : "f"(lse_8[0]), "f"(m_8));
                float _exp2_270 = approx_exp2_noftz(_sub_271);
                float _add_269;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_269) : "f"(den_8), "f"(_exp2_270));
                den_8 = _add_269;
                float _sub_274;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_274) : "f"(lse_8[1]), "f"(m_8));
                float _exp2_273 = approx_exp2_noftz(_sub_274);
                float _add_272;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_272) : "f"(den_8), "f"(_exp2_273));
                den_8 = _add_272;
                float _sub_277;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_277) : "f"(lse_8[2]), "f"(m_8));
                float _exp2_276 = approx_exp2_noftz(_sub_277);
                float _add_275;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_275) : "f"(den_8), "f"(_exp2_276));
                den_8 = _add_275;
                float _sub_280;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_280) : "f"(lse_8[3]), "f"(m_8));
                float _exp2_279 = approx_exp2_noftz(_sub_280);
                float _add_278;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_278) : "f"(den_8), "f"(_exp2_279));
                den_8 = _add_278;
                float _sub_283;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_283) : "f"(lse_8[4]), "f"(m_8));
                float _exp2_282 = approx_exp2_noftz(_sub_283);
                float _add_281;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_281) : "f"(den_8), "f"(_exp2_282));
                den_8 = _add_281;
                float _sub_286;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_286) : "f"(lse_8[5]), "f"(m_8));
                float _exp2_285 = approx_exp2_noftz(_sub_286);
                float _add_284;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_284) : "f"(den_8), "f"(_exp2_285));
                den_8 = _add_284;
                float _sub_289;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_289) : "f"(lse_8[6]), "f"(m_8));
                float _exp2_288 = approx_exp2_noftz(_sub_289);
                float _add_287;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_287) : "f"(den_8), "f"(_exp2_288));
                den_8 = _add_287;
                float _sub_292;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_292) : "f"(lse_8[7]), "f"(m_8));
                float _exp2_291 = approx_exp2_noftz(_sub_292);
                float _add_290;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_290) : "f"(den_8), "f"(_exp2_291));
                den_8 = _add_290;
                float _sub_295;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_295) : "f"(lse_8[8]), "f"(m_8));
                float _exp2_294 = approx_exp2_noftz(_sub_295);
                float _add_293;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_293) : "f"(den_8), "f"(_exp2_294));
                den_8 = _add_293;
                float w_8[9];
                float _sub_298;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_298) : "f"(lse_8[0]), "f"(m_8));
                float _exp2_297 = approx_exp2_noftz(_sub_298);
                float _fdiv_296 = __fdiv_rn(_exp2_297, den_8);
                w_8[0] = _fdiv_296;
                float _sub_301;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_301) : "f"(lse_8[1]), "f"(m_8));
                float _exp2_300 = approx_exp2_noftz(_sub_301);
                float _fdiv_299 = __fdiv_rn(_exp2_300, den_8);
                w_8[1] = _fdiv_299;
                float _sub_304;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_304) : "f"(lse_8[2]), "f"(m_8));
                float _exp2_303 = approx_exp2_noftz(_sub_304);
                float _fdiv_302 = __fdiv_rn(_exp2_303, den_8);
                w_8[2] = _fdiv_302;
                float _sub_307;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_307) : "f"(lse_8[3]), "f"(m_8));
                float _exp2_306 = approx_exp2_noftz(_sub_307);
                float _fdiv_305 = __fdiv_rn(_exp2_306, den_8);
                w_8[3] = _fdiv_305;
                float _sub_310;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_310) : "f"(lse_8[4]), "f"(m_8));
                float _exp2_309 = approx_exp2_noftz(_sub_310);
                float _fdiv_308 = __fdiv_rn(_exp2_309, den_8);
                w_8[4] = _fdiv_308;
                float _sub_313;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_313) : "f"(lse_8[5]), "f"(m_8));
                float _exp2_312 = approx_exp2_noftz(_sub_313);
                float _fdiv_311 = __fdiv_rn(_exp2_312, den_8);
                w_8[5] = _fdiv_311;
                float _sub_316;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_316) : "f"(lse_8[6]), "f"(m_8));
                float _exp2_315 = approx_exp2_noftz(_sub_316);
                float _fdiv_314 = __fdiv_rn(_exp2_315, den_8);
                w_8[6] = _fdiv_314;
                float _sub_319;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_319) : "f"(lse_8[7]), "f"(m_8));
                float _exp2_318 = approx_exp2_noftz(_sub_319);
                float _fdiv_317 = __fdiv_rn(_exp2_318, den_8);
                w_8[7] = _fdiv_317;
                float _sub_322;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_322) : "f"(lse_8[8]), "f"(m_8));
                float _exp2_321 = approx_exp2_noftz(_sub_322);
                float _fdiv_320 = __fdiv_rn(_exp2_321, den_8);
                w_8[8] = _fdiv_320;
                acc_8[0] = __fmaf_rn(w_8[0], __uint_as_float(words_8[0] << 16), acc_8[0]);
                acc_8[1] = __fmaf_rn(w_8[0], __uint_as_float(words_8[0] & 4294901760u), acc_8[1]);
                acc_8[2] = __fmaf_rn(w_8[0], __uint_as_float(words_8[1] << 16), acc_8[2]);
                acc_8[3] = __fmaf_rn(w_8[0], __uint_as_float(words_8[1] & 4294901760u), acc_8[3]);
                acc_8[4] = __fmaf_rn(w_8[0], __uint_as_float(words_8[2] << 16), acc_8[4]);
                acc_8[5] = __fmaf_rn(w_8[0], __uint_as_float(words_8[2] & 4294901760u), acc_8[5]);
                acc_8[6] = __fmaf_rn(w_8[0], __uint_as_float(words_8[3] << 16), acc_8[6]);
                acc_8[7] = __fmaf_rn(w_8[0], __uint_as_float(words_8[3] & 4294901760u), acc_8[7]);
                acc_8[0] = __fmaf_rn(w_8[1], __uint_as_float(words_8[4] << 16), acc_8[0]);
                acc_8[1] = __fmaf_rn(w_8[1], __uint_as_float(words_8[4] & 4294901760u), acc_8[1]);
                acc_8[2] = __fmaf_rn(w_8[1], __uint_as_float(words_8[5] << 16), acc_8[2]);
                acc_8[3] = __fmaf_rn(w_8[1], __uint_as_float(words_8[5] & 4294901760u), acc_8[3]);
                acc_8[4] = __fmaf_rn(w_8[1], __uint_as_float(words_8[6] << 16), acc_8[4]);
                acc_8[5] = __fmaf_rn(w_8[1], __uint_as_float(words_8[6] & 4294901760u), acc_8[5]);
                acc_8[6] = __fmaf_rn(w_8[1], __uint_as_float(words_8[7] << 16), acc_8[6]);
                acc_8[7] = __fmaf_rn(w_8[1], __uint_as_float(words_8[7] & 4294901760u), acc_8[7]);
                acc_8[0] = __fmaf_rn(w_8[2], __uint_as_float(words_8[8] << 16), acc_8[0]);
                acc_8[1] = __fmaf_rn(w_8[2], __uint_as_float(words_8[8] & 4294901760u), acc_8[1]);
                acc_8[2] = __fmaf_rn(w_8[2], __uint_as_float(words_8[9] << 16), acc_8[2]);
                acc_8[3] = __fmaf_rn(w_8[2], __uint_as_float(words_8[9] & 4294901760u), acc_8[3]);
                acc_8[4] = __fmaf_rn(w_8[2], __uint_as_float(words_8[10] << 16), acc_8[4]);
                acc_8[5] = __fmaf_rn(w_8[2], __uint_as_float(words_8[10] & 4294901760u), acc_8[5]);
                acc_8[6] = __fmaf_rn(w_8[2], __uint_as_float(words_8[11] << 16), acc_8[6]);
                acc_8[7] = __fmaf_rn(w_8[2], __uint_as_float(words_8[11] & 4294901760u), acc_8[7]);
                acc_8[0] = __fmaf_rn(w_8[3], __uint_as_float(words_8[12] << 16), acc_8[0]);
                acc_8[1] = __fmaf_rn(w_8[3], __uint_as_float(words_8[12] & 4294901760u), acc_8[1]);
                acc_8[2] = __fmaf_rn(w_8[3], __uint_as_float(words_8[13] << 16), acc_8[2]);
                acc_8[3] = __fmaf_rn(w_8[3], __uint_as_float(words_8[13] & 4294901760u), acc_8[3]);
                acc_8[4] = __fmaf_rn(w_8[3], __uint_as_float(words_8[14] << 16), acc_8[4]);
                acc_8[5] = __fmaf_rn(w_8[3], __uint_as_float(words_8[14] & 4294901760u), acc_8[5]);
                acc_8[6] = __fmaf_rn(w_8[3], __uint_as_float(words_8[15] << 16), acc_8[6]);
                acc_8[7] = __fmaf_rn(w_8[3], __uint_as_float(words_8[15] & 4294901760u), acc_8[7]);
                acc_8[0] = __fmaf_rn(w_8[4], __uint_as_float(words_8[16] << 16), acc_8[0]);
                acc_8[1] = __fmaf_rn(w_8[4], __uint_as_float(words_8[16] & 4294901760u), acc_8[1]);
                acc_8[2] = __fmaf_rn(w_8[4], __uint_as_float(words_8[17] << 16), acc_8[2]);
                acc_8[3] = __fmaf_rn(w_8[4], __uint_as_float(words_8[17] & 4294901760u), acc_8[3]);
                acc_8[4] = __fmaf_rn(w_8[4], __uint_as_float(words_8[18] << 16), acc_8[4]);
                acc_8[5] = __fmaf_rn(w_8[4], __uint_as_float(words_8[18] & 4294901760u), acc_8[5]);
                acc_8[6] = __fmaf_rn(w_8[4], __uint_as_float(words_8[19] << 16), acc_8[6]);
                acc_8[7] = __fmaf_rn(w_8[4], __uint_as_float(words_8[19] & 4294901760u), acc_8[7]);
                acc_8[0] = __fmaf_rn(w_8[5], __uint_as_float(words_8[20] << 16), acc_8[0]);
                acc_8[1] = __fmaf_rn(w_8[5], __uint_as_float(words_8[20] & 4294901760u), acc_8[1]);
                acc_8[2] = __fmaf_rn(w_8[5], __uint_as_float(words_8[21] << 16), acc_8[2]);
                acc_8[3] = __fmaf_rn(w_8[5], __uint_as_float(words_8[21] & 4294901760u), acc_8[3]);
                acc_8[4] = __fmaf_rn(w_8[5], __uint_as_float(words_8[22] << 16), acc_8[4]);
                acc_8[5] = __fmaf_rn(w_8[5], __uint_as_float(words_8[22] & 4294901760u), acc_8[5]);
                acc_8[6] = __fmaf_rn(w_8[5], __uint_as_float(words_8[23] << 16), acc_8[6]);
                acc_8[7] = __fmaf_rn(w_8[5], __uint_as_float(words_8[23] & 4294901760u), acc_8[7]);
                acc_8[0] = __fmaf_rn(w_8[6], __uint_as_float(words_8[24] << 16), acc_8[0]);
                acc_8[1] = __fmaf_rn(w_8[6], __uint_as_float(words_8[24] & 4294901760u), acc_8[1]);
                acc_8[2] = __fmaf_rn(w_8[6], __uint_as_float(words_8[25] << 16), acc_8[2]);
                acc_8[3] = __fmaf_rn(w_8[6], __uint_as_float(words_8[25] & 4294901760u), acc_8[3]);
                acc_8[4] = __fmaf_rn(w_8[6], __uint_as_float(words_8[26] << 16), acc_8[4]);
                acc_8[5] = __fmaf_rn(w_8[6], __uint_as_float(words_8[26] & 4294901760u), acc_8[5]);
                acc_8[6] = __fmaf_rn(w_8[6], __uint_as_float(words_8[27] << 16), acc_8[6]);
                acc_8[7] = __fmaf_rn(w_8[6], __uint_as_float(words_8[27] & 4294901760u), acc_8[7]);
                acc_8[0] = __fmaf_rn(w_8[7], __uint_as_float(words_8[28] << 16), acc_8[0]);
                acc_8[1] = __fmaf_rn(w_8[7], __uint_as_float(words_8[28] & 4294901760u), acc_8[1]);
                acc_8[2] = __fmaf_rn(w_8[7], __uint_as_float(words_8[29] << 16), acc_8[2]);
                acc_8[3] = __fmaf_rn(w_8[7], __uint_as_float(words_8[29] & 4294901760u), acc_8[3]);
                acc_8[4] = __fmaf_rn(w_8[7], __uint_as_float(words_8[30] << 16), acc_8[4]);
                acc_8[5] = __fmaf_rn(w_8[7], __uint_as_float(words_8[30] & 4294901760u), acc_8[5]);
                acc_8[6] = __fmaf_rn(w_8[7], __uint_as_float(words_8[31] << 16), acc_8[6]);
                acc_8[7] = __fmaf_rn(w_8[7], __uint_as_float(words_8[31] & 4294901760u), acc_8[7]);
                acc_8[0] = __fmaf_rn(w_8[8], __uint_as_float(words_8[32] << 16), acc_8[0]);
                acc_8[1] = __fmaf_rn(w_8[8], __uint_as_float(words_8[32] & 4294901760u), acc_8[1]);
                acc_8[2] = __fmaf_rn(w_8[8], __uint_as_float(words_8[33] << 16), acc_8[2]);
                acc_8[3] = __fmaf_rn(w_8[8], __uint_as_float(words_8[33] & 4294901760u), acc_8[3]);
                acc_8[4] = __fmaf_rn(w_8[8], __uint_as_float(words_8[34] << 16), acc_8[4]);
                acc_8[5] = __fmaf_rn(w_8[8], __uint_as_float(words_8[34] & 4294901760u), acc_8[5]);
                acc_8[6] = __fmaf_rn(w_8[8], __uint_as_float(words_8[35] << 16), acc_8[6]);
                acc_8[7] = __fmaf_rn(w_8[8], __uint_as_float(words_8[35] & 4294901760u), acc_8[7]);
            }
            unsigned int packed_8[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_8[_lp*2 + 0], acc_8[_lp*2+1 + 0]));
                packed_8[_lp] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_323 = out_chunk;
                __builtin_assume(__isGlobal(_gst_323));
                reinterpret_cast<int4*>(_gst_323)[0] = reinterpret_cast<int4*>(packed_8)[0];
            }
        }
    }
}

} // extern "C"
