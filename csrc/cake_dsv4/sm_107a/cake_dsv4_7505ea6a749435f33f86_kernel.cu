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
kernel_cake_dsv4_7505ea6a749435f33f86(__nv_bfloat16* partial_O, float* partial_lse, __nv_bfloat16* __restrict__ O, int num_heads, int num_splits)
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
    unsigned long long chunk = (unsigned long long)tid * 8ull;
    float* lse_row = partial_lse + row;
    __nv_bfloat16* o_chunk = partial_O + (row * 512ull + chunk);
    unsigned long long o_stride = heads * 512ull;
    __nv_bfloat16* out_chunk = O + (((unsigned long long)blockIdx.x * heads + (unsigned long long)blockIdx.y) * 512ull + chunk);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float lse[9];
    lse[0] = lse_row[0ull * heads];
    lse[1] = lse_row[1ull * heads];
    lse[2] = lse_row[2ull * heads];
    lse[3] = lse_row[3ull * heads];
    lse[4] = lse_row[4ull * heads];
    lse[5] = lse_row[5ull * heads];
    lse[6] = lse_row[6ull * heads];
    lse[7] = lse_row[7ull * heads];
    lse[8] = lse_row[8ull * heads];
    float m = -CUDART_INF_F;
    m = max_noftz(m, lse[0]);
    m = max_noftz(m, lse[1]);
    m = max_noftz(m, lse[2]);
    m = max_noftz(m, lse[3]);
    m = max_noftz(m, lse[4]);
    m = max_noftz(m, lse[5]);
    m = max_noftz(m, lse[6]);
    m = max_noftz(m, lse[7]);
    m = max_noftz(m, lse[8]);
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
            unsigned int words[36];
            words[0] = 0;
            words[1] = 0;
            words[2] = 0;
            words[3] = 0;
            if (lse[0] != -CUDART_INF_F) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(o_chunk + 0ull * o_stride + 0);
                    uint4* _vdst_0 = reinterpret_cast<uint4*>(&words[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_0[_blk] = _vptr_0[_blk];
                    }
                }
            }
            words[4] = 0;
            words[5] = 0;
            words[6] = 0;
            words[7] = 0;
            if (lse[1] != -CUDART_INF_F) {
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(o_chunk + 1ull * o_stride + 0);
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
            if (lse[2] != -CUDART_INF_F) {
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(o_chunk + 2ull * o_stride + 0);
                    uint4* _vdst_2 = reinterpret_cast<uint4*>(&words[8]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_2[_blk] = _vptr_2[_blk];
                    }
                }
            }
            words[12] = 0;
            words[13] = 0;
            words[14] = 0;
            words[15] = 0;
            if (lse[3] != -CUDART_INF_F) {
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(o_chunk + 3ull * o_stride + 0);
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
            if (lse[4] != -CUDART_INF_F) {
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(o_chunk + 4ull * o_stride + 0);
                    uint4* _vdst_4 = reinterpret_cast<uint4*>(&words[16]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_4[_blk] = _vptr_4[_blk];
                    }
                }
            }
            words[20] = 0;
            words[21] = 0;
            words[22] = 0;
            words[23] = 0;
            if (lse[5] != -CUDART_INF_F) {
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(o_chunk + 5ull * o_stride + 0);
                    uint4* _vdst_5 = reinterpret_cast<uint4*>(&words[20]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_5[_blk] = _vptr_5[_blk];
                    }
                }
            }
            words[24] = 0;
            words[25] = 0;
            words[26] = 0;
            words[27] = 0;
            if (lse[6] != -CUDART_INF_F) {
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(o_chunk + 6ull * o_stride + 0);
                    uint4* _vdst_6 = reinterpret_cast<uint4*>(&words[24]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_6[_blk] = _vptr_6[_blk];
                    }
                }
            }
            words[28] = 0;
            words[29] = 0;
            words[30] = 0;
            words[31] = 0;
            if (lse[7] != -CUDART_INF_F) {
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(o_chunk + 7ull * o_stride + 0);
                    uint4* _vdst_7 = reinterpret_cast<uint4*>(&words[28]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_7[_blk] = _vptr_7[_blk];
                    }
                }
            }
            words[32] = 0;
            words[33] = 0;
            words[34] = 0;
            words[35] = 0;
            if (lse[8] != -CUDART_INF_F) {
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(o_chunk + 8ull * o_stride + 0);
                    uint4* _vdst_8 = reinterpret_cast<uint4*>(&words[32]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_8[_blk] = _vptr_8[_blk];
                    }
                }
            }
            float den = 0.0f;
            float _sub_11;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(lse[0]), "f"(m));
            float _exp2_10 = approx_exp2_noftz(_sub_11);
            float _add_9;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(den), "f"(_exp2_10));
            den = _add_9;
            float _sub_14;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14) : "f"(lse[1]), "f"(m));
            float _exp2_13 = approx_exp2_noftz(_sub_14);
            float _add_12;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(den), "f"(_exp2_13));
            den = _add_12;
            float _sub_17;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(lse[2]), "f"(m));
            float _exp2_16 = approx_exp2_noftz(_sub_17);
            float _add_15;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(den), "f"(_exp2_16));
            den = _add_15;
            float _sub_20;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(lse[3]), "f"(m));
            float _exp2_19 = approx_exp2_noftz(_sub_20);
            float _add_18;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(den), "f"(_exp2_19));
            den = _add_18;
            float _sub_23;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(lse[4]), "f"(m));
            float _exp2_22 = approx_exp2_noftz(_sub_23);
            float _add_21;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(den), "f"(_exp2_22));
            den = _add_21;
            float _sub_26;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26) : "f"(lse[5]), "f"(m));
            float _exp2_25 = approx_exp2_noftz(_sub_26);
            float _add_24;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(den), "f"(_exp2_25));
            den = _add_24;
            float _sub_29;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(lse[6]), "f"(m));
            float _exp2_28 = approx_exp2_noftz(_sub_29);
            float _add_27;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(den), "f"(_exp2_28));
            den = _add_27;
            float _sub_32;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(lse[7]), "f"(m));
            float _exp2_31 = approx_exp2_noftz(_sub_32);
            float _add_30;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(den), "f"(_exp2_31));
            den = _add_30;
            float _sub_35;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(lse[8]), "f"(m));
            float _exp2_34 = approx_exp2_noftz(_sub_35);
            float _add_33;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(den), "f"(_exp2_34));
            den = _add_33;
            float w[9];
            float _sub_38;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(lse[0]), "f"(m));
            float _exp2_37 = approx_exp2_noftz(_sub_38);
            float _fdiv_36 = __fdiv_rn(_exp2_37, den);
            w[0] = _fdiv_36;
            float _sub_41;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(lse[1]), "f"(m));
            float _exp2_40 = approx_exp2_noftz(_sub_41);
            float _fdiv_39 = __fdiv_rn(_exp2_40, den);
            w[1] = _fdiv_39;
            float _sub_44;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(lse[2]), "f"(m));
            float _exp2_43 = approx_exp2_noftz(_sub_44);
            float _fdiv_42 = __fdiv_rn(_exp2_43, den);
            w[2] = _fdiv_42;
            float _sub_47;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(lse[3]), "f"(m));
            float _exp2_46 = approx_exp2_noftz(_sub_47);
            float _fdiv_45 = __fdiv_rn(_exp2_46, den);
            w[3] = _fdiv_45;
            float _sub_50;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(lse[4]), "f"(m));
            float _exp2_49 = approx_exp2_noftz(_sub_50);
            float _fdiv_48 = __fdiv_rn(_exp2_49, den);
            w[4] = _fdiv_48;
            float _sub_53;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(lse[5]), "f"(m));
            float _exp2_52 = approx_exp2_noftz(_sub_53);
            float _fdiv_51 = __fdiv_rn(_exp2_52, den);
            w[5] = _fdiv_51;
            float _sub_56;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56) : "f"(lse[6]), "f"(m));
            float _exp2_55 = approx_exp2_noftz(_sub_56);
            float _fdiv_54 = __fdiv_rn(_exp2_55, den);
            w[6] = _fdiv_54;
            float _sub_59;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_59) : "f"(lse[7]), "f"(m));
            float _exp2_58 = approx_exp2_noftz(_sub_59);
            float _fdiv_57 = __fdiv_rn(_exp2_58, den);
            w[7] = _fdiv_57;
            float _sub_62;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62) : "f"(lse[8]), "f"(m));
            float _exp2_61 = approx_exp2_noftz(_sub_62);
            float _fdiv_60 = __fdiv_rn(_exp2_61, den);
            w[8] = _fdiv_60;
            acc[0] = __fmaf_rn(w[0], __uint_as_float(words[0] << 16), acc[0]);
            acc[1] = __fmaf_rn(w[0], __uint_as_float(words[0] & 4294901760u), acc[1]);
            acc[2] = __fmaf_rn(w[0], __uint_as_float(words[1] << 16), acc[2]);
            acc[3] = __fmaf_rn(w[0], __uint_as_float(words[1] & 4294901760u), acc[3]);
            acc[4] = __fmaf_rn(w[0], __uint_as_float(words[2] << 16), acc[4]);
            acc[5] = __fmaf_rn(w[0], __uint_as_float(words[2] & 4294901760u), acc[5]);
            acc[6] = __fmaf_rn(w[0], __uint_as_float(words[3] << 16), acc[6]);
            acc[7] = __fmaf_rn(w[0], __uint_as_float(words[3] & 4294901760u), acc[7]);
            acc[0] = __fmaf_rn(w[1], __uint_as_float(words[4] << 16), acc[0]);
            acc[1] = __fmaf_rn(w[1], __uint_as_float(words[4] & 4294901760u), acc[1]);
            acc[2] = __fmaf_rn(w[1], __uint_as_float(words[5] << 16), acc[2]);
            acc[3] = __fmaf_rn(w[1], __uint_as_float(words[5] & 4294901760u), acc[3]);
            acc[4] = __fmaf_rn(w[1], __uint_as_float(words[6] << 16), acc[4]);
            acc[5] = __fmaf_rn(w[1], __uint_as_float(words[6] & 4294901760u), acc[5]);
            acc[6] = __fmaf_rn(w[1], __uint_as_float(words[7] << 16), acc[6]);
            acc[7] = __fmaf_rn(w[1], __uint_as_float(words[7] & 4294901760u), acc[7]);
            acc[0] = __fmaf_rn(w[2], __uint_as_float(words[8] << 16), acc[0]);
            acc[1] = __fmaf_rn(w[2], __uint_as_float(words[8] & 4294901760u), acc[1]);
            acc[2] = __fmaf_rn(w[2], __uint_as_float(words[9] << 16), acc[2]);
            acc[3] = __fmaf_rn(w[2], __uint_as_float(words[9] & 4294901760u), acc[3]);
            acc[4] = __fmaf_rn(w[2], __uint_as_float(words[10] << 16), acc[4]);
            acc[5] = __fmaf_rn(w[2], __uint_as_float(words[10] & 4294901760u), acc[5]);
            acc[6] = __fmaf_rn(w[2], __uint_as_float(words[11] << 16), acc[6]);
            acc[7] = __fmaf_rn(w[2], __uint_as_float(words[11] & 4294901760u), acc[7]);
            acc[0] = __fmaf_rn(w[3], __uint_as_float(words[12] << 16), acc[0]);
            acc[1] = __fmaf_rn(w[3], __uint_as_float(words[12] & 4294901760u), acc[1]);
            acc[2] = __fmaf_rn(w[3], __uint_as_float(words[13] << 16), acc[2]);
            acc[3] = __fmaf_rn(w[3], __uint_as_float(words[13] & 4294901760u), acc[3]);
            acc[4] = __fmaf_rn(w[3], __uint_as_float(words[14] << 16), acc[4]);
            acc[5] = __fmaf_rn(w[3], __uint_as_float(words[14] & 4294901760u), acc[5]);
            acc[6] = __fmaf_rn(w[3], __uint_as_float(words[15] << 16), acc[6]);
            acc[7] = __fmaf_rn(w[3], __uint_as_float(words[15] & 4294901760u), acc[7]);
            acc[0] = __fmaf_rn(w[4], __uint_as_float(words[16] << 16), acc[0]);
            acc[1] = __fmaf_rn(w[4], __uint_as_float(words[16] & 4294901760u), acc[1]);
            acc[2] = __fmaf_rn(w[4], __uint_as_float(words[17] << 16), acc[2]);
            acc[3] = __fmaf_rn(w[4], __uint_as_float(words[17] & 4294901760u), acc[3]);
            acc[4] = __fmaf_rn(w[4], __uint_as_float(words[18] << 16), acc[4]);
            acc[5] = __fmaf_rn(w[4], __uint_as_float(words[18] & 4294901760u), acc[5]);
            acc[6] = __fmaf_rn(w[4], __uint_as_float(words[19] << 16), acc[6]);
            acc[7] = __fmaf_rn(w[4], __uint_as_float(words[19] & 4294901760u), acc[7]);
            acc[0] = __fmaf_rn(w[5], __uint_as_float(words[20] << 16), acc[0]);
            acc[1] = __fmaf_rn(w[5], __uint_as_float(words[20] & 4294901760u), acc[1]);
            acc[2] = __fmaf_rn(w[5], __uint_as_float(words[21] << 16), acc[2]);
            acc[3] = __fmaf_rn(w[5], __uint_as_float(words[21] & 4294901760u), acc[3]);
            acc[4] = __fmaf_rn(w[5], __uint_as_float(words[22] << 16), acc[4]);
            acc[5] = __fmaf_rn(w[5], __uint_as_float(words[22] & 4294901760u), acc[5]);
            acc[6] = __fmaf_rn(w[5], __uint_as_float(words[23] << 16), acc[6]);
            acc[7] = __fmaf_rn(w[5], __uint_as_float(words[23] & 4294901760u), acc[7]);
            acc[0] = __fmaf_rn(w[6], __uint_as_float(words[24] << 16), acc[0]);
            acc[1] = __fmaf_rn(w[6], __uint_as_float(words[24] & 4294901760u), acc[1]);
            acc[2] = __fmaf_rn(w[6], __uint_as_float(words[25] << 16), acc[2]);
            acc[3] = __fmaf_rn(w[6], __uint_as_float(words[25] & 4294901760u), acc[3]);
            acc[4] = __fmaf_rn(w[6], __uint_as_float(words[26] << 16), acc[4]);
            acc[5] = __fmaf_rn(w[6], __uint_as_float(words[26] & 4294901760u), acc[5]);
            acc[6] = __fmaf_rn(w[6], __uint_as_float(words[27] << 16), acc[6]);
            acc[7] = __fmaf_rn(w[6], __uint_as_float(words[27] & 4294901760u), acc[7]);
            acc[0] = __fmaf_rn(w[7], __uint_as_float(words[28] << 16), acc[0]);
            acc[1] = __fmaf_rn(w[7], __uint_as_float(words[28] & 4294901760u), acc[1]);
            acc[2] = __fmaf_rn(w[7], __uint_as_float(words[29] << 16), acc[2]);
            acc[3] = __fmaf_rn(w[7], __uint_as_float(words[29] & 4294901760u), acc[3]);
            acc[4] = __fmaf_rn(w[7], __uint_as_float(words[30] << 16), acc[4]);
            acc[5] = __fmaf_rn(w[7], __uint_as_float(words[30] & 4294901760u), acc[5]);
            acc[6] = __fmaf_rn(w[7], __uint_as_float(words[31] << 16), acc[6]);
            acc[7] = __fmaf_rn(w[7], __uint_as_float(words[31] & 4294901760u), acc[7]);
            acc[0] = __fmaf_rn(w[8], __uint_as_float(words[32] << 16), acc[0]);
            acc[1] = __fmaf_rn(w[8], __uint_as_float(words[32] & 4294901760u), acc[1]);
            acc[2] = __fmaf_rn(w[8], __uint_as_float(words[33] << 16), acc[2]);
            acc[3] = __fmaf_rn(w[8], __uint_as_float(words[33] & 4294901760u), acc[3]);
            acc[4] = __fmaf_rn(w[8], __uint_as_float(words[34] << 16), acc[4]);
            acc[5] = __fmaf_rn(w[8], __uint_as_float(words[34] & 4294901760u), acc[5]);
            acc[6] = __fmaf_rn(w[8], __uint_as_float(words[35] << 16), acc[6]);
            acc[7] = __fmaf_rn(w[8], __uint_as_float(words[35] & 4294901760u), acc[7]);
        }
        unsigned int packed[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
            packed[_lp] = *(uint32_t*)&_bf2;
        }
        {
            auto _gst_63 = out_chunk;
            __builtin_assume(__isGlobal(_gst_63));
            reinterpret_cast<int4*>(_gst_63)[0] = reinterpret_cast<int4*>(packed)[0];
        }
    }
}

} // extern "C"
