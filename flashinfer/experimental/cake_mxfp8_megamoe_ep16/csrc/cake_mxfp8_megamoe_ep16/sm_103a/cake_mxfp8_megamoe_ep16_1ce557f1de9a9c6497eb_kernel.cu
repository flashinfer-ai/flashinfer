/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
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

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_mxfp8_megamoe_ep16_1ce557f1de9a9c6497eb(__nv_bfloat16* __restrict__ source_hidden, long long* __restrict__ source_ids, float* __restrict__ source_weights, uint8_t* __restrict__ published_hidden, uint8_t* __restrict__ published_scales, int* __restrict__ published_ids, float* __restrict__ published_weights, int tokens, int32_t pg_world, int32_t pg_rank, unsigned* const* __restrict__ pg_flags, unsigned int* __restrict__ input_ready, unsigned int* const* __restrict__ input_ready_peers)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    uint8_t* published_hidden_fp8 = reinterpret_cast<uint8_t*>(published_hidden);
    int token = bid;
    unsigned long long source_base = (unsigned long long)token * 3072;
    unsigned long long scale_base = (unsigned long long)token * 96;
    const int quant_chunks = 192;
    #pragma unroll
    for (int quant_round = 0; quant_round < 4; quant_round++) {
        int quant_chunk = tid + quant_round * 128;
        if (quant_chunk < quant_chunks) {
            int column = quant_chunk * 16;
            float _vec_load_0[16];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(source_hidden + (source_base + (unsigned long long)column) + 0);
                uint4 _vld_0[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_vld_0[_blk].x), "=r"(_vld_0[_blk].y), "=r"(_vld_0[_blk].z), "=r"(_vld_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)) : "memory");
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_0[_pair]));
                    }
                }
            }
            float local_absmax = 0.0f;
            #pragma unroll
            for (int element = 0; element < 16; element++) {
                float _fabs_0 = fabsf(_vec_load_0[element]);
                float _max_0 = max_noftz(local_absmax, _fabs_0);
                local_absmax = _max_0;
            }
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, local_absmax, 1);
            float paired_absmax = _shfl_xor_0;
            float _max_1 = max_noftz(local_absmax, paired_absmax);
            float block_max = _max_1;
            float _max_2 = max_noftz(block_max, 1e-07f);
            float _rcp_0 = approx_rcp(448.0f);
            float xsf = _max_2 * _rcp_0;
            unsigned int xbits = __as_u32(xsf);
            unsigned int scale_code = (xbits >> 23 & 255) + ((xbits & 8388607) + 8388607 >> 23);
            unsigned int _max_3 = ((scale_code) > (1) ? (scale_code) : (1));
            scale_code = _max_3;
            unsigned int _min_0 = ((scale_code) < (254) ? (scale_code) : (254));
            scale_code = _min_0;
            int scale_i = scale_code;
            float _cvt_f32_0 = __bfloat162float(scale_i);
            float _exp2_0 = approx_exp2(127.0f - _cvt_f32_0);
            float inverse_scale = _exp2_0;
            float normalized[16];
            #pragma unroll
            for (int element_1 = 0; element_1 < 16; element_1++) {
                normalized[element_1] = _vec_load_0[element_1] * inverse_scale;
            }
            {
                unsigned int _fp8_pk[4];
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[0]) : "f"(normalized[0 + 0]), "f"(normalized[0 + 1]), "f"(normalized[0 + 2]), "f"(normalized[0 + 3]));
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[1]) : "f"(normalized[0 + 4]), "f"(normalized[0 + 5]), "f"(normalized[0 + 6]), "f"(normalized[0 + 7]));
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[2]) : "f"(normalized[0 + 8]), "f"(normalized[0 + 9]), "f"(normalized[0 + 10]), "f"(normalized[0 + 11]));
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[3]) : "f"(normalized[0 + 12]), "f"(normalized[0 + 13]), "f"(normalized[0 + 14]), "f"(normalized[0 + 15]));
                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(published_hidden_fp8 + (source_base + (unsigned long long)column)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
            }
            if ((quant_chunk & 1) == 0) {
                published_scales[scale_base + (unsigned long long)(quant_chunk / 2)] = scale_i;
            }
        }
    }
    if (tid < 8 && token < tokens) {
        int route = token * 8 + tid;
        published_ids[route] = (int)source_ids[route];
        published_weights[route] = source_weights[route];
    }
    asm volatile("barrier.sync 15, 128;" ::: "memory");
    if (elect_sync()) {
        asm volatile("fence.release.sys;" ::: "memory");
    }
    asm volatile("barrier.sync 15, 128;" ::: "memory");
    if (tid < 16 && token < tokens) {
        asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(input_ready)) + ((unsigned long long)token * 16 + (unsigned long long)tid))), "r"(static_cast<unsigned int>(1)) : "memory");
    }
}

} // extern "C"
