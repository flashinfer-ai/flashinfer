/*
 * Copyright (c) 2023 by FlashInfer team.
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
#define THREADS 256
#define H 3584

#include <math_constants.h>

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_df166979b48ef2b54cf3(__nv_bfloat16* __restrict__ x, float* __restrict__ qx, uint8_t* __restrict__ packed, uint8_t* __restrict__ scales, int M)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int token = blockIdx.x;
    int group = tid;
    const int groups_per_row = H / 16;
    if (token < M && group < groups_per_row) {
        unsigned int packed_source[8];
        float source[16];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (token * H + group * 16) + 0);
            uint4* _vdst_0 = reinterpret_cast<uint4*>(&packed_source[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vdst_0[_blk] = _vptr_0[_blk];
            }
        }
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&source[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_source[_pair]) << 16);
            (&source[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_source[_pair]) & 0xffff0000u);
        }
        uint32_t _bf16x2_abs_0;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(packed_source[0]));
        uint32_t _bf16x2_abs_1;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(packed_source[1]));
        uint32_t _bf16x2_max_0;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(_bf16x2_abs_0), "r"(_bf16x2_abs_1));
        uint32_t _bf16x2_abs_2;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(packed_source[2]));
        uint32_t _bf16x2_max_1;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(_bf16x2_max_0), "r"(_bf16x2_abs_2));
        uint32_t _bf16x2_abs_3;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(packed_source[3]));
        uint32_t _bf16x2_max_2;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(_bf16x2_max_1), "r"(_bf16x2_abs_3));
        uint32_t _bf16x2_abs_4;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_4) : "r"(packed_source[4]));
        uint32_t _bf16x2_max_3;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(_bf16x2_max_2), "r"(_bf16x2_abs_4));
        uint32_t _bf16x2_abs_5;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_5) : "r"(packed_source[5]));
        uint32_t _bf16x2_max_4;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(_bf16x2_max_3), "r"(_bf16x2_abs_5));
        uint32_t _bf16x2_abs_6;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_6) : "r"(packed_source[6]));
        uint32_t _bf16x2_max_5;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_5) : "r"(_bf16x2_max_4), "r"(_bf16x2_abs_6));
        uint32_t _bf16x2_abs_7;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_7) : "r"(packed_source[7]));
        uint32_t _bf16x2_max_6;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_6) : "r"(_bf16x2_max_5), "r"(_bf16x2_abs_7));
        uint16_t _bf16_max_0;
        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_0) : "h"((uint16_t)(_bf16x2_max_6 & 65535)), "h"((uint16_t)(_bf16x2_max_6 >> 16)));
        float _cvt_f32_bf16_0;
        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_0) : "h"((uint16_t)(_bf16_max_0)));
        float block_max = _cvt_f32_bf16_0;
        float global_encode = qx[0];
        float scale_value = 0.0f;
        if (block_max != 0.0f) {
            scale_value = block_max * (global_encode * 0.16666666666666666f);
        }
        float _fp8_rt_0;
        uint16_t _e4m3x2_1;
        uint32_t _f16x2_1;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(scale_value));
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
        float rounded_scale = _fp8_rt_0;
        float output_scale = 0.0f;
        if (block_max != 0.0f) {
            float _fdiv_rn_0 = __fdiv_rn(1.0f, global_encode);
            float global_decode = _fdiv_rn_0;
            float _fdiv_rn_1 = __fdiv_rn(1.0f, rounded_scale * global_decode);
            float _min_0 = fminf(_fdiv_rn_1, 3.4028234663852886e+38f);
            output_scale = _min_0;
        }
        #if __CUDA_ARCH__ >= 1000
        const float2 _scale2_2 = {output_scale, output_scale};
        #pragma unroll
        for (int _ls = 0; _ls < 8; _ls++)
            mul_f32x2_inplace(&reinterpret_cast<float2*>(source)[_ls], _scale2_2);
        #else
        #pragma unroll
        for (int _ls = 0; _ls < 16; _ls++) {
            source[_ls] = source[_ls] * output_scale;
        }
        #endif
        uint32_t _fp4_0[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]), "f"(source[4]), "f"(source[5]), "f"(source[6]), "f"(source[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[1]) : "f"(source[8]), "f"(source[9]), "f"(source[10]), "f"(source[11]), "f"(source[12]), "f"(source[13]), "f"(source[14]), "f"(source[15]));
        int packed_byte = token * (H / 2) + group * 8;
        *(reinterpret_cast<int*>(packed + packed_byte) + (0)) = _fp4_0[0];
        *(reinterpret_cast<int*>(packed + (packed_byte + 4)) + (0)) = _fp4_0[1];
        {
            unsigned short _fp8_pair;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale_value));
            *(reinterpret_cast<unsigned char*>(scales + (token * groups_per_row + group)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
        }
    }
}

} // extern "C"
