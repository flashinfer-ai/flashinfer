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


__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_sage_block_sparse_attention_ce646595859ababcf2bd(__nv_bfloat16* __restrict__ v_in, uint8_t* __restrict__ v_out, float* __restrict__ v_amax, float* __restrict__ v_scale, int seqlen_k, int num_heads, int chunks_per_bh)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int head = bid % num_heads;
    int batch_chunk = bid / num_heads;
    int chunk = batch_chunk % chunks_per_bh;
    int batch = batch_chunk / chunks_per_bh;
    int bh = batch * num_heads + head;
    int warp_0 = warp;
    int lane_1 = lane;
    int half = lane_1 / 8;
    int sub = lane_1 % 8;
    int channel0 = sub * 16;
    int token0 = chunk * 64 + warp_0 * 16 + half;
    unsigned int vw[32];
    int v_off[4];
    int v_ok[4];
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        int token = token0 + p * 4;
        int in_range = ((token < seqlen_k) ? 1 : 0);
        int load_token = ((in_range != 0) ? token : seqlen_k - 1);
        int offset = ((batch * seqlen_k + load_token) * num_heads + head) * 128 + channel0;
        v_ok[p] = in_range;
        v_off[p] = offset;
        {
            asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                : "=r"(vw[p * 8 + 0]), "=r"(vw[p * 8 + 1]), "=r"(vw[p * 8 + 2]), "=r"(vw[p * 8 + 3]), "=r"(vw[p * 8 + 4]), "=r"(vw[p * 8 + 5]), "=r"(vw[p * 8 + 6]), "=r"(vw[p * 8 + 7]) : "l"((const void*)((const char*)(v_in + offset + 0) + 0)), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    float amax[16];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        {
            float4 _v4 = *reinterpret_cast<const float4*>(v_amax + (bh * 128 + channel0 + 4 * j) + 0);
            amax[4 * j + 0] = _v4.x;
            amax[4 * j + 1] = _v4.y;
            amax[4 * j + 2] = _v4.z;
            amax[4 * j + 3] = _v4.w;
        }
    }
    float inv[16];
    #pragma unroll
    for (int j_1 = 0; j_1 < 16; j_1++) {
        inv[j_1] = 448.0f / fmaxf(amax[j_1], 1e-06f);
    }
    if (chunk == 0 && tid < 8) {
        #pragma unroll
        for (int j_2 = 0; j_2 < 16; j_2++) {
            v_scale[bh * 128 + channel0 + j_2] = fmaxf(amax[j_2], 1e-06f) / 448.0f;
        }
    }
    #pragma unroll
    for (int p_1 = 0; p_1 < 4; p_1++) {
        float values[16];
        #pragma unroll
        for (int j_3 = 0; j_3 < 8; j_3++) {
            float2 _cvt_f32_0 = __bfloat1622float2(__as_bf16x2(vw[p_1 * 8 + j_3]));
            values[2 * j_3] = _cvt_f32_0.x;
            values[2 * j_3 + 1] = _cvt_f32_0.y;
        }
        #pragma unroll
        for (int j_4 = 0; j_4 < 16; j_4++) {
            values[j_4] = values[j_4] * inv[j_4];
        }
        if (v_ok[p_1] != 0) {
            {
                unsigned int _fp8_pk[4];
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[0]) : "f"(values[0 + 0]), "f"(values[0 + 1]), "f"(values[0 + 2]), "f"(values[0 + 3]));
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[1]) : "f"(values[0 + 4]), "f"(values[0 + 5]), "f"(values[0 + 6]), "f"(values[0 + 7]));
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[2]) : "f"(values[0 + 8]), "f"(values[0 + 9]), "f"(values[0 + 10]), "f"(values[0 + 11]));
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[3]) : "f"(values[0 + 12]), "f"(values[0 + 13]), "f"(values[0 + 14]), "f"(values[0 + 15]));
                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(v_out + v_off[p_1]) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
            }
        }
    }
}

} // extern "C"
