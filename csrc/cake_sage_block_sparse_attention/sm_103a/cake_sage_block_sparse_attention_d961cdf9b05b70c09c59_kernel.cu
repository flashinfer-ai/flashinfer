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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

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
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_sage_block_sparse_attention_d961cdf9b05b70c09c59(__nv_bfloat16* __restrict__ v_in, uint8_t* __restrict__ v_out, float* __restrict__ v_amax, float* __restrict__ v_scale, int seqlen_k, int num_heads, int chunks_per_bh)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int bh = bid / chunks_per_bh;
    int chunk = bid % chunks_per_bh;
    int batch = bh / num_heads;
    int head = bh % num_heads;
    int warp_0 = warp;
    int lane_1 = lane;
    int token_in_warp = lane_1 / 16;
    int channel0 = lane_1 % 16 * 8;
    int chunk_base = chunk * 64;
    float amax[8];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(v_amax + (bh * 128 + channel0) + 0);
        amax[0 + 0] = _v4.x;
        amax[0 + 1] = _v4.y;
        amax[0 + 2] = _v4.z;
        amax[0 + 3] = _v4.w;
    }
    {
        float4 _v4 = *reinterpret_cast<const float4*>(v_amax + (bh * 128 + channel0 + 4) + 0);
        amax[4 + 0] = _v4.x;
        amax[4 + 1] = _v4.y;
        amax[4 + 2] = _v4.z;
        amax[4 + 3] = _v4.w;
    }
    float inv[8];
    #pragma unroll
    for (int item = 0; item < 8; item++) {
        float _fmax_0 = fmaxf(amax[item], 1e-06f);
        inv[item] = 448.0f / _fmax_0;
    }
    if (chunk == 0 && tid < 16) {
        #pragma unroll
        for (int item_1 = 0; item_1 < 8; item_1++) {
            float _fmax_1 = fmaxf(amax[item_1], 1e-06f);
            v_scale[bh * 128 + channel0 + item_1] = _fmax_1 / 448.0f;
        }
    }
    #pragma unroll
    for (int step = 0; step < 4; step++) {
        int token = chunk_base + step * 16 + warp_0 * 2 + token_in_warp;
        if (token < seqlen_k) {
            int offset = ((batch * seqlen_k + token) * num_heads + head) * 128 + channel0;
            float _vec_load_0[8];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(v_in + offset + 0);
                uint4 _vld_2[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_2[_blk] = _vptr_2[_blk];
                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_2[_pair]));
                    }
                }
            }
            #pragma unroll
            for (int item_2 = 0; item_2 < 8; item_2++) {
                _vec_load_0[item_2] = _vec_load_0[item_2] * inv[item_2];
            }
            {
                unsigned int _fp8_pk[2];
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[0]) : "f"(_vec_load_0[0 + 0]), "f"(_vec_load_0[0 + 1]), "f"(_vec_load_0[0 + 2]), "f"(_vec_load_0[0 + 3]));
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[1]) : "f"(_vec_load_0[0 + 4]), "f"(_vec_load_0[0 + 5]), "f"(_vec_load_0[0 + 6]), "f"(_vec_load_0[0 + 7]));
                *reinterpret_cast<uint2*>(reinterpret_cast<unsigned char*>(v_out + offset) + (0)) = *reinterpret_cast<uint2*>(_fp8_pk);
            }
        }
    }
}

} // extern "C"
