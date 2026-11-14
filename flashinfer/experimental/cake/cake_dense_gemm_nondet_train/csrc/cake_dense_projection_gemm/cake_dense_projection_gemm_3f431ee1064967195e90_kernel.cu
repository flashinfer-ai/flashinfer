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
kernel_cake_dense_projection_gemm_3f431ee1064967195e90(float* __restrict__ ws, float* __restrict__ out, int splits, int rows, int cols, int ldo)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned long long cols64 = (unsigned long long)cols;
    unsigned long long quads = (cols64 + 3ull) / 4ull;
    unsigned long long q = (unsigned long long)(bid * 256 + tid);
    if (q < (unsigned long long)rows * quads) {
        unsigned long long r = q / quads;
        unsigned long long c0 = (q - r * quads) * 4ull;
        unsigned long long slab = (unsigned long long)rows * cols64;
        int full = ((cols64 >= c0 + 4ull) ? 1 : 0);
        unsigned long long p_bits = (unsigned long long)(ws + (r * cols64 + c0));
        int vec_in = full;
        if ((p_bits & 15ull | (slab & 3ull) << 4ull) != 0ull) {
            vec_in = 0;
        }
        float w[32];
        if (splits > 0) {
            unsigned long long p_s_0 = 0ull * slab + r * cols64 + c0;
            if (vec_in == 1) {
                float _vec_load_0[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(ws + p_s_0 + 0);
                    _vec_load_0[0 + 0] = _v4.x;
                    _vec_load_0[0 + 1] = _v4.y;
                    _vec_load_0[0 + 2] = _v4.z;
                    _vec_load_0[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int j_0 = 0; j_0 < 4; j_0++) {
                    w[j_0] = _vec_load_0[j_0];
                }
            } else {
                #pragma unroll
                for (int j_4 = 0; j_4 < 4; j_4++) {
                    w[j_4] = 0.0f;
                    if (cols64 > c0 + (unsigned long long)j_4) {
                        w[j_4] = ws[p_s_0 + (unsigned long long)j_4];
                    }
                }
            }
        }
        if (splits > 1) {
            unsigned long long p_s_1 = 1ull * slab + r * cols64 + c0;
            if (vec_in == 1) {
                float _vec_load_1[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(ws + p_s_1 + 0);
                    _vec_load_1[0 + 0] = _v4.x;
                    _vec_load_1[0 + 1] = _v4.y;
                    _vec_load_1[0 + 2] = _v4.z;
                    _vec_load_1[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int j_5 = 0; j_5 < 4; j_5++) {
                    w[4 + j_5] = _vec_load_1[j_5];
                }
            } else {
                #pragma unroll
                for (int j_6 = 0; j_6 < 4; j_6++) {
                    w[4 + j_6] = 0.0f;
                    if (cols64 > c0 + (unsigned long long)j_6) {
                        w[4 + j_6] = ws[p_s_1 + (unsigned long long)j_6];
                    }
                }
            }
        }
        if (splits > 2) {
            unsigned long long p_s_2 = 2ull * slab + r * cols64 + c0;
            if (vec_in == 1) {
                float _vec_load_2[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(ws + p_s_2 + 0);
                    _vec_load_2[0 + 0] = _v4.x;
                    _vec_load_2[0 + 1] = _v4.y;
                    _vec_load_2[0 + 2] = _v4.z;
                    _vec_load_2[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int j_7 = 0; j_7 < 4; j_7++) {
                    w[8 + j_7] = _vec_load_2[j_7];
                }
            } else {
                #pragma unroll
                for (int j_8 = 0; j_8 < 4; j_8++) {
                    w[8 + j_8] = 0.0f;
                    if (cols64 > c0 + (unsigned long long)j_8) {
                        w[8 + j_8] = ws[p_s_2 + (unsigned long long)j_8];
                    }
                }
            }
        }
        if (splits > 3) {
            unsigned long long p_s_3 = 3ull * slab + r * cols64 + c0;
            if (vec_in == 1) {
                float _vec_load_3[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(ws + p_s_3 + 0);
                    _vec_load_3[0 + 0] = _v4.x;
                    _vec_load_3[0 + 1] = _v4.y;
                    _vec_load_3[0 + 2] = _v4.z;
                    _vec_load_3[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int j_9 = 0; j_9 < 4; j_9++) {
                    w[12 + j_9] = _vec_load_3[j_9];
                }
            } else {
                #pragma unroll
                for (int j_10 = 0; j_10 < 4; j_10++) {
                    w[12 + j_10] = 0.0f;
                    if (cols64 > c0 + (unsigned long long)j_10) {
                        w[12 + j_10] = ws[p_s_3 + (unsigned long long)j_10];
                    }
                }
            }
        }
        if (splits > 4) {
            unsigned long long p_s_4 = 4ull * slab + r * cols64 + c0;
            if (vec_in == 1) {
                float _vec_load_4[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(ws + p_s_4 + 0);
                    _vec_load_4[0 + 0] = _v4.x;
                    _vec_load_4[0 + 1] = _v4.y;
                    _vec_load_4[0 + 2] = _v4.z;
                    _vec_load_4[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int j_11 = 0; j_11 < 4; j_11++) {
                    w[16 + j_11] = _vec_load_4[j_11];
                }
            } else {
                #pragma unroll
                for (int j_12 = 0; j_12 < 4; j_12++) {
                    w[16 + j_12] = 0.0f;
                    if (cols64 > c0 + (unsigned long long)j_12) {
                        w[16 + j_12] = ws[p_s_4 + (unsigned long long)j_12];
                    }
                }
            }
        }
        if (splits > 5) {
            unsigned long long p_s_5 = 5ull * slab + r * cols64 + c0;
            if (vec_in == 1) {
                float _vec_load_5[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(ws + p_s_5 + 0);
                    _vec_load_5[0 + 0] = _v4.x;
                    _vec_load_5[0 + 1] = _v4.y;
                    _vec_load_5[0 + 2] = _v4.z;
                    _vec_load_5[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int j_13 = 0; j_13 < 4; j_13++) {
                    w[20 + j_13] = _vec_load_5[j_13];
                }
            } else {
                #pragma unroll
                for (int j_14 = 0; j_14 < 4; j_14++) {
                    w[20 + j_14] = 0.0f;
                    if (cols64 > c0 + (unsigned long long)j_14) {
                        w[20 + j_14] = ws[p_s_5 + (unsigned long long)j_14];
                    }
                }
            }
        }
        if (splits > 6) {
            unsigned long long p_s_6 = 6ull * slab + r * cols64 + c0;
            if (vec_in == 1) {
                float _vec_load_6[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(ws + p_s_6 + 0);
                    _vec_load_6[0 + 0] = _v4.x;
                    _vec_load_6[0 + 1] = _v4.y;
                    _vec_load_6[0 + 2] = _v4.z;
                    _vec_load_6[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int j_15 = 0; j_15 < 4; j_15++) {
                    w[24 + j_15] = _vec_load_6[j_15];
                }
            } else {
                #pragma unroll
                for (int j_16 = 0; j_16 < 4; j_16++) {
                    w[24 + j_16] = 0.0f;
                    if (cols64 > c0 + (unsigned long long)j_16) {
                        w[24 + j_16] = ws[p_s_6 + (unsigned long long)j_16];
                    }
                }
            }
        }
        if (splits > 7) {
            unsigned long long p_s_7 = 7ull * slab + r * cols64 + c0;
            if (vec_in == 1) {
                float _vec_load_7[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(ws + p_s_7 + 0);
                    _vec_load_7[0 + 0] = _v4.x;
                    _vec_load_7[0 + 1] = _v4.y;
                    _vec_load_7[0 + 2] = _v4.z;
                    _vec_load_7[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int j_17 = 0; j_17 < 4; j_17++) {
                    w[28 + j_17] = _vec_load_7[j_17];
                }
            } else {
                #pragma unroll
                for (int j_18 = 0; j_18 < 4; j_18++) {
                    w[28 + j_18] = 0.0f;
                    if (cols64 > c0 + (unsigned long long)j_18) {
                        w[28 + j_18] = ws[p_s_7 + (unsigned long long)j_18];
                    }
                }
            }
        }
        float acc[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            acc[i] = 0.0f;
        }
        if (splits > 0) {
            #pragma unroll
            for (int j_1_0 = 0; j_1_0 < 4; j_1_0++) {
                acc[j_1_0] = acc[j_1_0] + w[j_1_0];
            }
        }
        if (splits > 1) {
            #pragma unroll
            for (int j_1_1 = 0; j_1_1 < 4; j_1_1++) {
                acc[4 + j_1_1] = acc[4 + j_1_1] + w[4 + j_1_1];
            }
        }
        if (splits > 2) {
            #pragma unroll
            for (int j_1_2 = 0; j_1_2 < 4; j_1_2++) {
                acc[8 + j_1_2] = acc[8 + j_1_2] + w[8 + j_1_2];
            }
        }
        if (splits > 3) {
            #pragma unroll
            for (int j_1_3 = 0; j_1_3 < 4; j_1_3++) {
                acc[12 + j_1_3] = acc[12 + j_1_3] + w[12 + j_1_3];
            }
        }
        if (splits > 4) {
            #pragma unroll
            for (int j_1_4 = 0; j_1_4 < 4; j_1_4++) {
                acc[j_1_4] = acc[j_1_4] + w[16 + j_1_4];
            }
        }
        if (splits > 5) {
            #pragma unroll
            for (int j_1_5 = 0; j_1_5 < 4; j_1_5++) {
                acc[4 + j_1_5] = acc[4 + j_1_5] + w[20 + j_1_5];
            }
        }
        if (splits > 6) {
            #pragma unroll
            for (int j_1_6 = 0; j_1_6 < 4; j_1_6++) {
                acc[8 + j_1_6] = acc[8 + j_1_6] + w[24 + j_1_6];
            }
        }
        if (splits > 7) {
            #pragma unroll
            for (int j_1_7 = 0; j_1_7 < 4; j_1_7++) {
                acc[12 + j_1_7] = acc[12 + j_1_7] + w[28 + j_1_7];
            }
        }
        float res[4];
        #pragma unroll
        for (int j_2 = 0; j_2 < 4; j_2++) {
            res[j_2] = acc[j_2] + acc[4 + j_2] + acc[8 + j_2] + acc[12 + j_2];
        }
        unsigned long long o_off = r * (unsigned long long)ldo + c0;
        unsigned long long o_bits = (unsigned long long)(out + o_off);
        int vec_out = full;
        if ((o_bits & 15ull) != 0ull) {
            vec_out = 0;
        }
        if (vec_out == 1) {
            {
                float4 _v4 = make_float4(res[0 + 0], res[0 + 1], res[0 + 2], res[0 + 3]);
                *reinterpret_cast<float4*>((out + o_off) + 0) = _v4;
            }
        } else {
            #pragma unroll
            for (int j_3 = 0; j_3 < 4; j_3++) {
                if (cols64 > c0 + (unsigned long long)j_3) {
                    *(reinterpret_cast<float*>(out + (o_off + (unsigned long long)j_3)) + (0)) = res[j_3];
                }
            }
        }
    }
}

} // extern "C"
