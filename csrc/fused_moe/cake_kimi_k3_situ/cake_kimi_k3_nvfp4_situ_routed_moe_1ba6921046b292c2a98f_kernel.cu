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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
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
#define H 3584

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 4) void
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_1ba6921046b292c2a98f(uint8_t* __restrict__ expert_output, float* __restrict__ partial_scale, __nv_bfloat16* __restrict__ route_weights, int* __restrict__ token_to_permuted, __nv_bfloat16* __restrict__ out, int M)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int token = blockIdx.x;
    if (token < M) {
        unsigned int magic = 1258291200;
        int feature = blockIdx.y * 1024 + tid * 8;
        if (feature < H) {
            float accum[8];
            accum[0] = 0.0f;
            accum[1] = 0.0f;
            accum[2] = 0.0f;
            accum[3] = 0.0f;
            accum[4] = 0.0f;
            accum[5] = 0.0f;
            accum[6] = 0.0f;
            accum[7] = 0.0f;
            int m_tile = feature / 128;
            #pragma unroll
            for (int slot = 0; slot < 16; slot++) {
                int pair = token * 16 + slot;
                int row = token_to_permuted[pair];
                float weight = route_weights[pair];
                float ws = weight * partial_scale[row / 128 * 28 + m_tile];
                unsigned int words[2];
                {
                    uint2 _vld_0;
                    _vld_0 = *reinterpret_cast<const uint2*>(expert_output + (unsigned int)(row * H + feature) + 0);
                    uint2* _vdst_0 = reinterpret_cast<uint2*>(&words[0]);
                    *_vdst_0 = _vld_0;
                }
                uint32_t _prmt_b32_0;
                asm("prmt.b32 %0, %1, %2, 0x7650;" : "=r"(_prmt_b32_0) : "r"(words[0]), "r"(magic));
                float v0 = __uint_as_float(_prmt_b32_0);
                uint32_t _prmt_b32_1;
                asm("prmt.b32 %0, %1, %2, 0x7651;" : "=r"(_prmt_b32_1) : "r"(words[0]), "r"(magic));
                float v1 = __uint_as_float(_prmt_b32_1);
                uint32_t _prmt_b32_2;
                asm("prmt.b32 %0, %1, %2, 0x7652;" : "=r"(_prmt_b32_2) : "r"(words[0]), "r"(magic));
                float v2 = __uint_as_float(_prmt_b32_2);
                uint32_t _prmt_b32_3;
                asm("prmt.b32 %0, %1, %2, 0x7653;" : "=r"(_prmt_b32_3) : "r"(words[0]), "r"(magic));
                float v3 = __uint_as_float(_prmt_b32_3);
                uint32_t _prmt_b32_4;
                asm("prmt.b32 %0, %1, %2, 0x7650;" : "=r"(_prmt_b32_4) : "r"(words[1]), "r"(magic));
                float v4 = __uint_as_float(_prmt_b32_4);
                uint32_t _prmt_b32_5;
                asm("prmt.b32 %0, %1, %2, 0x7651;" : "=r"(_prmt_b32_5) : "r"(words[1]), "r"(magic));
                float v5 = __uint_as_float(_prmt_b32_5);
                uint32_t _prmt_b32_6;
                asm("prmt.b32 %0, %1, %2, 0x7652;" : "=r"(_prmt_b32_6) : "r"(words[1]), "r"(magic));
                float v6 = __uint_as_float(_prmt_b32_6);
                uint32_t _prmt_b32_7;
                asm("prmt.b32 %0, %1, %2, 0x7653;" : "=r"(_prmt_b32_7) : "r"(words[1]), "r"(magic));
                float v7 = __uint_as_float(_prmt_b32_7);
                v0 = v0 - 8388736.0f;
                v1 = v1 - 8388736.0f;
                v2 = v2 - 8388736.0f;
                v3 = v3 - 8388736.0f;
                v4 = v4 - 8388736.0f;
                v5 = v5 - 8388736.0f;
                v6 = v6 - 8388736.0f;
                v7 = v7 - 8388736.0f;
                accum[0] = accum[0] + ws * v0;
                accum[1] = accum[1] + ws * v1;
                accum[2] = accum[2] + ws * v2;
                accum[3] = accum[3] + ws * v3;
                accum[4] = accum[4] + ws * v4;
                accum[5] = accum[5] + ws * v5;
                accum[6] = accum[6] + ws * v6;
                accum[7] = accum[7] + ws * v7;
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(accum[0 + 0], accum[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(accum[0 + 2], accum[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(accum[0 + 4], accum[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(accum[0 + 6], accum[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (token * H + feature)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
}

} // extern "C"
