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

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_sm100_megamoe_workspace_topk_reduce_tiered_t512x2(__nv_bfloat16* __restrict__ partials, __nv_bfloat16* __restrict__ out)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    unsigned int token_idx = bid / 4;
    unsigned int cta_in_token = bid % 4;
    unsigned int hidden_offset = (cta_in_token * 512 + (unsigned int)tid) * 2;
    unsigned long long partial_offset = (unsigned long long)token_idx * 24576 + (unsigned long long)hidden_offset;
    unsigned long long out_offset = (unsigned long long)token_idx * 4096 + (unsigned long long)hidden_offset;
    float _vec_load_0[2];
    {
        uint32_t _bf16x2_bits_0;
        asm volatile("ld.global.L1::no_allocate.b32 %0, [%1];"
            : "=r"(_bf16x2_bits_0) : "l"((const void*)(partials + partial_offset + (0))) : "memory");
        (&_vec_load_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_0) << 16);
        (&_vec_load_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_0) & 0xffff0000u);
    }
    float2 one = make_float2(1.0f, 1.0f);
    float acc[2];
    acc[0] = _vec_load_0[0];
    acc[1] = _vec_load_0[1];
    #pragma unroll
    for (int k = 1; k < 6; k++) {
        float _vec_load_1[2];
        {
            uint32_t _bf16x2_bits_1;
            asm volatile("ld.global.L1::no_allocate.b32 %0, [%1];"
                : "=r"(_bf16x2_bits_1) : "l"((const void*)(partials + (partial_offset + (unsigned long long)(k * 4096)) + (0))) : "memory");
            (&_vec_load_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_1) << 16);
            (&_vec_load_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_1) & 0xffff0000u);
        }
        float2 total_0 = fma_f32x2_rn_noftz(make_float2(_vec_load_1[0], _vec_load_1[1]), one, make_float2(acc[0], acc[1]));
        acc[0] = total_0.x;
        acc[1] = total_0.y;
    }
    unsigned int _packed_0[1];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
        _packed_0[_lp] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<unsigned*>(out + out_offset)[0] = reinterpret_cast<unsigned*>(_packed_0)[0];
}

__global__ __launch_bounds__(512) void
kernel_flashinfer_sm100_megamoe_workspace_topk_reduce_tiered_t512x4(__nv_bfloat16* __restrict__ partials, __nv_bfloat16* __restrict__ out)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    unsigned int token_idx = bid / 2;
    unsigned int cta_in_token = bid % 2;
    unsigned int hidden_offset = (cta_in_token * 512 + (unsigned int)tid) * 4;
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
            (&_vec_load_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
            (&_vec_load_0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
        }
    }
    float2 one = make_float2(1.0f, 1.0f);
    float acc[4];
    acc[0] = _vec_load_0[0];
    acc[1] = _vec_load_0[1];
    acc[2] = _vec_load_0[2];
    acc[3] = _vec_load_0[3];
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
                (&_vec_load_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&_vec_load_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
        float2 total_0 = fma_f32x2_rn_noftz(make_float2(_vec_load_1[0], _vec_load_1[1]), one, make_float2(acc[0], acc[1]));
        acc[0] = total_0.x;
        acc[1] = total_0.y;
        float2 total_1 = fma_f32x2_rn_noftz(make_float2(_vec_load_1[2], _vec_load_1[3]), one, make_float2(acc[2], acc[3]));
        acc[2] = total_1.x;
        acc[3] = total_1.y;
    }
    unsigned int _packed_0[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
        _packed_0[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 2], acc[_lp*2+1 + 2]));
        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(out + out_offset)[0] = reinterpret_cast<int2*>(_packed_0)[0];
}

__global__ __launch_bounds__(512) void
kernel_flashinfer_sm100_megamoe_workspace_topk_reduce_tiered_t512x4_pf256(__nv_bfloat16* __restrict__ partials, __nv_bfloat16* __restrict__ out)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    unsigned int token_idx = bid / 2;
    unsigned int cta_in_token = bid % 2;
    unsigned int hidden_offset = (cta_in_token * 512 + (unsigned int)tid) * 4;
    unsigned long long partial_offset = (unsigned long long)token_idx * 24576 + (unsigned long long)hidden_offset;
    unsigned long long out_offset = (unsigned long long)token_idx * 4096 + (unsigned long long)hidden_offset;
    float _vec_load_0[4];
    {
        uint2 _vld_0;
        asm volatile("ld.global.L1::no_allocate.L2::256B.v2.b32 {%0, %1}, [%2];"
            : "=r"(_vld_0.x), "=r"(_vld_0.y) : "l"((const void*)(partials + partial_offset + 0)) : "memory");
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&_vec_load_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
            (&_vec_load_0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
        }
    }
    float2 one = make_float2(1.0f, 1.0f);
    float acc[4];
    acc[0] = _vec_load_0[0];
    acc[1] = _vec_load_0[1];
    acc[2] = _vec_load_0[2];
    acc[3] = _vec_load_0[3];
    #pragma unroll
    for (int k = 1; k < 6; k++) {
        float _vec_load_1[4];
        {
            uint2 _vld_1;
            asm volatile("ld.global.L1::no_allocate.L2::256B.v2.b32 {%0, %1}, [%2];"
                : "=r"(_vld_1.x), "=r"(_vld_1.y) : "l"((const void*)(partials + (partial_offset + (unsigned long long)(k * 4096)) + 0)) : "memory");
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&_vec_load_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
        float2 total_0 = fma_f32x2_rn_noftz(make_float2(_vec_load_1[0], _vec_load_1[1]), one, make_float2(acc[0], acc[1]));
        acc[0] = total_0.x;
        acc[1] = total_0.y;
        float2 total_1 = fma_f32x2_rn_noftz(make_float2(_vec_load_1[2], _vec_load_1[3]), one, make_float2(acc[2], acc[3]));
        acc[2] = total_1.x;
        acc[3] = total_1.y;
    }
    unsigned int _packed_0[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
        _packed_0[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 2], acc[_lp*2+1 + 2]));
        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(out + out_offset)[0] = reinterpret_cast<int2*>(_packed_0)[0];
}

__global__ __launch_bounds__(1024) void
kernel_flashinfer_sm100_megamoe_workspace_topk_reduce_tiered_t1024x4_pf256(__nv_bfloat16* __restrict__ partials, __nv_bfloat16* __restrict__ out)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    unsigned int token_idx = bid;
    unsigned int cta_in_token = 0;
    unsigned int hidden_offset = (cta_in_token * 1024 + (unsigned int)tid) * 4;
    unsigned long long partial_offset = (unsigned long long)token_idx * 24576 + (unsigned long long)hidden_offset;
    unsigned long long out_offset = (unsigned long long)token_idx * 4096 + (unsigned long long)hidden_offset;
    float _vec_load_0[4];
    {
        uint2 _vld_0;
        asm volatile("ld.global.L1::no_allocate.L2::256B.v2.b32 {%0, %1}, [%2];"
            : "=r"(_vld_0.x), "=r"(_vld_0.y) : "l"((const void*)(partials + partial_offset + 0)) : "memory");
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&_vec_load_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
            (&_vec_load_0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
        }
    }
    float2 one = make_float2(1.0f, 1.0f);
    float acc[4];
    acc[0] = _vec_load_0[0];
    acc[1] = _vec_load_0[1];
    acc[2] = _vec_load_0[2];
    acc[3] = _vec_load_0[3];
    #pragma unroll
    for (int k = 1; k < 6; k++) {
        float _vec_load_1[4];
        {
            uint2 _vld_1;
            asm volatile("ld.global.L1::no_allocate.L2::256B.v2.b32 {%0, %1}, [%2];"
                : "=r"(_vld_1.x), "=r"(_vld_1.y) : "l"((const void*)(partials + (partial_offset + (unsigned long long)(k * 4096)) + 0)) : "memory");
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&_vec_load_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
        float2 total_0 = fma_f32x2_rn_noftz(make_float2(_vec_load_1[0], _vec_load_1[1]), one, make_float2(acc[0], acc[1]));
        acc[0] = total_0.x;
        acc[1] = total_0.y;
        float2 total_1 = fma_f32x2_rn_noftz(make_float2(_vec_load_1[2], _vec_load_1[3]), one, make_float2(acc[2], acc[3]));
        acc[2] = total_1.x;
        acc[3] = total_1.y;
    }
    unsigned int _packed_0[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
        _packed_0[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 2], acc[_lp*2+1 + 2]));
        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(out + out_offset)[0] = reinterpret_cast<int2*>(_packed_0)[0];
}

} // extern "C"

#undef CAKE_INF
#undef NUM_MAIN_STAGES
