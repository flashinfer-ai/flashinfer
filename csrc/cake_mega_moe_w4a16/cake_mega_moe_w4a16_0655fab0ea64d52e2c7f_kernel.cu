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
template <typename T, int Capacity = 8>
struct __align__(16) CakePeerPointerTable { T* ptrs[Capacity]; };
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

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_mega_moe_w4a16_0655fab0ea64d52e2c7f(__nv_bfloat16* route_terms, int* __restrict__ topk_ids, float* __restrict__ topk_weights, __nv_bfloat16* __restrict__ output, int* tokens_per_rank, unsigned int* epoch, int hidden, int32_t mega_pg_world, int32_t mega_pg_rank, CakePeerPointerTable<unsigned int> mega_pg_flags, unsigned int* completion, CakePeerPointerTable<unsigned int> completion_peers)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int slice_index = blockIdx.x;
    int first_token = blockIdx.y;
    int token_stride = gridDim.y;
    int32_t _sysv_u32_0;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(_sysv_u32_0) : "l"(tokens_per_rank) : "memory");
    int live_tokens = _sysv_u32_0;
    if (first_token < live_tokens) {
        int column = slice_index * 1024 + tid * 8;
        unsigned long long row = (unsigned long long)hidden;
        int ids[8];
        float scores[8];
        unsigned long long first_route = (unsigned long long)first_token * 8;
        {
            const int4* _ivptr_0 = reinterpret_cast<const int4*>(topk_ids + first_route + 0);
            int4 _ivld_0;
            _ivld_0 = *_ivptr_0;
            ids[0 + 0] = _ivld_0.x;
            ids[0 + 1] = _ivld_0.y;
            ids[0 + 2] = _ivld_0.z;
            ids[0 + 3] = _ivld_0.w;
        }
        {
            const int4* _ivptr_1 = reinterpret_cast<const int4*>(topk_ids + (first_route + 4) + 0);
            int4 _ivld_1;
            _ivld_1 = *_ivptr_1;
            ids[4 + 0] = _ivld_1.x;
            ids[4 + 1] = _ivld_1.y;
            ids[4 + 2] = _ivld_1.z;
            ids[4 + 3] = _ivld_1.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(topk_weights + first_route + 0);
            scores[0 + 0] = _v4.x;
            scores[0 + 1] = _v4.y;
            scores[0 + 2] = _v4.z;
            scores[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(topk_weights + (first_route + 4) + 0);
            scores[4 + 0] = _v4.x;
            scores[4 + 1] = _v4.y;
            scores[4 + 2] = _v4.z;
            scores[4 + 3] = _v4.w;
        }
        unsigned int* counter_words = reinterpret_cast<unsigned int*>(completion);
        if (tid == 0) {
            uint32_t _sysv_u32_1;
            asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_1) : "l"(epoch) : "memory");
            unsigned int generation = _sysv_u32_1;
            unsigned int done_target = generation * (unsigned int)mega_pg_world;
            {
            unsigned int _acquire_observed;
            do {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(counter_words) + (0))) : "memory");
            } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(done_target)) >= static_cast<unsigned int>(2147483648u));
            }
            asm volatile("fence.acquire.sys;" ::: "memory");
        }
        asm volatile("barrier.sync 15, 128;" ::: "memory");
        unsigned int terms[32];
        float values[8];
        float accum[8];
        for (int token = first_token; token < live_tokens; token += token_stride) {
            unsigned long long route_offset = (unsigned long long)token * 8;
            unsigned long long route_base = route_offset * row + (unsigned long long)column;
            #pragma unroll
            for (int route_slot = 0; route_slot < 8; route_slot++) {
                {
                    uint32_t _uv4_4_0;
                    uint32_t _uv4_4_1;
                    uint32_t _uv4_4_2;
                    uint32_t _uv4_4_3;
                    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_4_0), "=r"(_uv4_4_1), "=r"(_uv4_4_2), "=r"(_uv4_4_3) : "l"((const void*)(route_terms + (route_base + row * (unsigned long long)route_slot) + (0))) : "memory");
                    terms[route_slot * 4 + 0] = _uv4_4_0;
                    terms[route_slot * 4 + 1] = _uv4_4_1;
                    terms[route_slot * 4 + 2] = _uv4_4_2;
                    terms[route_slot * 4 + 3] = _uv4_4_3;
                }
            }
            accum[0] = 0.0f;
            accum[1] = 0.0f;
            accum[2] = 0.0f;
            accum[3] = 0.0f;
            accum[4] = 0.0f;
            accum[5] = 0.0f;
            accum[6] = 0.0f;
            accum[7] = 0.0f;
            #pragma unroll
            for (int route_slot_1 = 0; route_slot_1 < 8; route_slot_1++) {
                int expert = ids[route_slot_1];
                if (expert >= 0) {
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((terms + route_slot_1 * 4)[_pair]) << 16);
                        (&values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((terms + route_slot_1 * 4)[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&values[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((terms + route_slot_1 * 4)[1 + _pair]) << 16);
                        (&values[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((terms + route_slot_1 * 4)[1 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&values[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((terms + route_slot_1 * 4)[2 + _pair]) << 16);
                        (&values[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((terms + route_slot_1 * 4)[2 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&values[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((terms + route_slot_1 * 4)[3 + _pair]) << 16);
                        (&values[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((terms + route_slot_1 * 4)[3 + _pair]) & 0xffff0000u);
                    }
                    float score = scores[route_slot_1];
                    #pragma unroll
                    for (int element = 0; element < 8; element++) {
                        accum[element] = __fmaf_rn(values[element], score, accum[element]);
                    }
                }
            }
            unsigned int _packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum[_lp*2 + 0], accum[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum[_lp*2 + 2], accum[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum[_lp*2 + 4], accum[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum[_lp*2 + 6], accum[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(output + ((unsigned long long)token * row + (unsigned long long)column))[0] = reinterpret_cast<int4*>(_packed_0)[0];
            int next_token = token + token_stride;
            if (next_token < live_tokens) {
                unsigned long long next_route = (unsigned long long)next_token * 8;
                {
                    const int4* _ivptr_5 = reinterpret_cast<const int4*>(topk_ids + next_route + 0);
                    int4 _ivld_5;
                    _ivld_5 = *_ivptr_5;
                    ids[0 + 0] = _ivld_5.x;
                    ids[0 + 1] = _ivld_5.y;
                    ids[0 + 2] = _ivld_5.z;
                    ids[0 + 3] = _ivld_5.w;
                }
                {
                    const int4* _ivptr_6 = reinterpret_cast<const int4*>(topk_ids + (next_route + 4) + 0);
                    int4 _ivld_6;
                    _ivld_6 = *_ivptr_6;
                    ids[4 + 0] = _ivld_6.x;
                    ids[4 + 1] = _ivld_6.y;
                    ids[4 + 2] = _ivld_6.z;
                    ids[4 + 3] = _ivld_6.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(topk_weights + next_route + 0);
                    scores[0 + 0] = _v4.x;
                    scores[0 + 1] = _v4.y;
                    scores[0 + 2] = _v4.z;
                    scores[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(topk_weights + (next_route + 4) + 0);
                    scores[4 + 0] = _v4.x;
                    scores[4 + 1] = _v4.y;
                    scores[4 + 2] = _v4.z;
                    scores[4 + 3] = _v4.w;
                }
            }
        }
    }
}

} // extern "C"
