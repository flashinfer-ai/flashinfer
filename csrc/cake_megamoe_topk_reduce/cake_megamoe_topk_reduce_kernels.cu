/*
 * Copyright (c) 2026 by FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
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

#include <math_constants.h>

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_megamoe_workspace_topk_reduce_bfloat16_h4096_k6(__nv_bfloat16* __restrict__ partials, __nv_bfloat16* __restrict__ out)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    unsigned int token_idx = bid / 4;
    unsigned int cta_in_token = bid % 4;
    unsigned int hidden_offset = (cta_in_token * 256 + (unsigned int)tid) * 4;
    unsigned long long partial_offset = (unsigned long long)token_idx * 24576 + (unsigned long long)hidden_offset;
    unsigned long long out_offset = (unsigned long long)token_idx * 4096 + (unsigned long long)hidden_offset;
    float _vec_load_0[4];
    {
        uint2 _vld_0;
        asm volatile("ld.global.L1::no_allocate.v2.b32 {%0, %1}, [%2];"
            : "=r"(_vld_0.x), "=r"(_vld_0.y) : "l"((const void*)(partials + partial_offset + 0)) : "memory");
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_0[0 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _pair * 2])[1])
                : "r"(_vpairs_0[_pair]));
        }
    }
    float2 _f2_0 = make_float2(1.0f, 1.0f);
    float2 one = _f2_0;
    float2 _f2_1 = make_float2(_vec_load_0[0], _vec_load_0[1]);
    float2 acc01 = _f2_1;
    float2 _f2_2 = make_float2(_vec_load_0[2], _vec_load_0[3]);
    float2 acc23 = _f2_2;
    #pragma unroll
    for (int k = 1; k < 6; k++) {
        float _vec_load_1[4];
        {
            uint2 _vld_1;
            asm volatile("ld.global.L1::no_allocate.v2.b32 {%0, %1}, [%2];"
                : "=r"(_vld_1.x), "=r"(_vld_1.y) : "l"((const void*)(partials + (partial_offset + (unsigned long long)(k * 4096)) + 0)) : "memory");
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_1[0 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _pair * 2])[1])
                    : "r"(_vpairs_1[_pair]));
            }
        }
        float2 _f2_3 = make_float2(_vec_load_1[0], _vec_load_1[1]);
        acc01 = fma_f32x2_rn_noftz(_f2_3, one, acc01);
        float2 _f2_4 = make_float2(_vec_load_1[2], _vec_load_1[3]);
        acc23 = fma_f32x2_rn_noftz(_f2_4, one, acc23);
    }
    float result[4];
    result[0] = acc01.x;
    result[1] = acc01.y;
    result[2] = acc23.x;
    result[3] = acc23.y;
    {
        uint2 _pk2;
        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
        _pk[0] = __floats2bfloat162_rn(result[0 + 0], result[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(result[0 + 2], result[0 + 3]);
        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out + out_offset))[0]) = _pk2;
    }
}

} // extern "C"

#undef CAKE_INF
#undef NUM_MAIN_STAGES
#undef THREADS
