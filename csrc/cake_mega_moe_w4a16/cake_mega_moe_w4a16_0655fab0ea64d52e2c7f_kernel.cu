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
    int token = blockIdx.y;
    int32_t _sysv_u32_0;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(_sysv_u32_0) : "l"(tokens_per_rank) : "memory");
    int live_tokens = _sysv_u32_0;
    if (token < live_tokens) {
        int column = slice_index * 1024 + tid * 8;
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
        unsigned long long row = (unsigned long long)hidden;
        unsigned long long route_offset = (unsigned long long)token * 8;
        int ids[8];
        {
            const int4* _ivptr_0 = reinterpret_cast<const int4*>(topk_ids + route_offset + 0);
            int4 _ivld_0;
            _ivld_0 = *_ivptr_0;
            ids[0 + 0] = _ivld_0.x;
            ids[0 + 1] = _ivld_0.y;
            ids[0 + 2] = _ivld_0.z;
            ids[0 + 3] = _ivld_0.w;
        }
        {
            const int4* _ivptr_1 = reinterpret_cast<const int4*>(topk_ids + (route_offset + 4) + 0);
            int4 _ivld_1;
            _ivld_1 = *_ivptr_1;
            ids[4 + 0] = _ivld_1.x;
            ids[4 + 1] = _ivld_1.y;
            ids[4 + 2] = _ivld_1.z;
            ids[4 + 3] = _ivld_1.w;
        }
        float scores[8];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(topk_weights + route_offset + 0);
            scores[0 + 0] = _v4.x;
            scores[0 + 1] = _v4.y;
            scores[0 + 2] = _v4.z;
            scores[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(topk_weights + (route_offset + 4) + 0);
            scores[4 + 0] = _v4.x;
            scores[4 + 1] = _v4.y;
            scores[4 + 2] = _v4.z;
            scores[4 + 3] = _v4.w;
        }
        unsigned long long route_base = route_offset * row + (unsigned long long)column;
        float accum[8];
        accum[0] = 0.0f;
        accum[1] = 0.0f;
        accum[2] = 0.0f;
        accum[3] = 0.0f;
        accum[4] = 0.0f;
        accum[5] = 0.0f;
        accum[6] = 0.0f;
        accum[7] = 0.0f;
        #pragma unroll
        for (int route_slot = 0; route_slot < 8; route_slot++) {
            int expert = ids[route_slot];
            if (expert >= 0) {
                float _vec_load_0[8];
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(route_terms + (route_base + row * (unsigned long long)route_slot) + 0);
                    uint4 _vld_4[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_vld_4[_blk].x), "=r"(_vld_4[_blk].y), "=r"(_vld_4[_blk].z), "=r"(_vld_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)) : "memory");
                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                            (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float score = scores[route_slot];
                #pragma unroll
                for (int element = 0; element < 8; element++) {
                    accum[element] = __fmaf_rn(_vec_load_0[element], score, accum[element]);
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
    }
}

} // extern "C"
