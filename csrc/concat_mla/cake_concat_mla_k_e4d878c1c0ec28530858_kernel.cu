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
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>
extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_concat_mla_k_e4d878c1c0ec28530858(uint8_t* __restrict__ k, uint8_t* __restrict__ k_nope, uint8_t* __restrict__ k_rope, int element_bytes, long long k_stride_0_bytes, long long k_stride_1_bytes, long long k_nope_stride_0_bytes, long long k_nope_stride_1_bytes, long long k_rope_stride_0_bytes, int tokens)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (element_bytes == 1) {
        int staged_1b[24] = {0};
        long long dst_offsets_1b[6] = {0};
        dst_offsets_1b[0] = (long long)(bid * 2) * k_stride_0_bytes + (long long)(tid / 8) * k_stride_1_bytes + (long long)(tid - tid / 8 * 8) * 16;
        if (bid * 2 < tokens) {
            {
                const int4* _ivptr_0 = reinterpret_cast<const int4*>(k_nope + ((long long)(bid * 2) * k_nope_stride_0_bytes + (long long)(tid / 8) * k_nope_stride_1_bytes + (long long)(tid - tid / 8 * 8) * 16) + 0);
                int4 _ivld_0;
                _ivld_0 = *_ivptr_0;
                staged_1b[0 + 0] = _ivld_0.x;
                staged_1b[0 + 1] = _ivld_0.y;
                staged_1b[0 + 2] = _ivld_0.z;
                staged_1b[0 + 3] = _ivld_0.w;
            }
        }
        dst_offsets_1b[1] = (long long)(bid * 2) * k_stride_0_bytes + (long long)((tid + 512) / 8) * k_stride_1_bytes + (long long)(tid + 512 - (tid + 512) / 8 * 8) * 16;
        if (bid * 2 < tokens) {
            {
                const int4* _ivptr_1 = reinterpret_cast<const int4*>(k_nope + ((long long)(bid * 2) * k_nope_stride_0_bytes + (long long)((tid + 512) / 8) * k_nope_stride_1_bytes + (long long)(tid + 512 - (tid + 512) / 8 * 8) * 16) + 0);
                int4 _ivld_1;
                _ivld_1 = *_ivptr_1;
                staged_1b[4 + 0] = _ivld_1.x;
                staged_1b[4 + 1] = _ivld_1.y;
                staged_1b[4 + 2] = _ivld_1.z;
                staged_1b[4 + 3] = _ivld_1.w;
            }
        }
        dst_offsets_1b[2] = (long long)(bid * 2) * k_stride_0_bytes + (long long)(tid / 4) * k_stride_1_bytes + 128 + (long long)(tid - tid / 4 * 4) * 16;
        if (bid * 2 < tokens) {
            {
                const int4* _ivptr_2 = reinterpret_cast<const int4*>(k_rope + ((long long)(bid * 2) * k_rope_stride_0_bytes + (long long)(tid - tid / 4 * 4) * 16) + 0);
                int4 _ivld_2;
                _ivld_2 = *_ivptr_2;
                staged_1b[8 + 0] = _ivld_2.x;
                staged_1b[8 + 1] = _ivld_2.y;
                staged_1b[8 + 2] = _ivld_2.z;
                staged_1b[8 + 3] = _ivld_2.w;
            }
        }
        dst_offsets_1b[3] = (long long)(bid * 2 + 1) * k_stride_0_bytes + (long long)(tid / 8) * k_stride_1_bytes + (long long)(tid - tid / 8 * 8) * 16;
        if (bid * 2 + 1 < tokens) {
            {
                const int4* _ivptr_3 = reinterpret_cast<const int4*>(k_nope + ((long long)(bid * 2 + 1) * k_nope_stride_0_bytes + (long long)(tid / 8) * k_nope_stride_1_bytes + (long long)(tid - tid / 8 * 8) * 16) + 0);
                int4 _ivld_3;
                _ivld_3 = *_ivptr_3;
                staged_1b[12 + 0] = _ivld_3.x;
                staged_1b[12 + 1] = _ivld_3.y;
                staged_1b[12 + 2] = _ivld_3.z;
                staged_1b[12 + 3] = _ivld_3.w;
            }
        }
        dst_offsets_1b[4] = (long long)(bid * 2 + 1) * k_stride_0_bytes + (long long)((tid + 512) / 8) * k_stride_1_bytes + (long long)(tid + 512 - (tid + 512) / 8 * 8) * 16;
        if (bid * 2 + 1 < tokens) {
            {
                const int4* _ivptr_4 = reinterpret_cast<const int4*>(k_nope + ((long long)(bid * 2 + 1) * k_nope_stride_0_bytes + (long long)((tid + 512) / 8) * k_nope_stride_1_bytes + (long long)(tid + 512 - (tid + 512) / 8 * 8) * 16) + 0);
                int4 _ivld_4;
                _ivld_4 = *_ivptr_4;
                staged_1b[16 + 0] = _ivld_4.x;
                staged_1b[16 + 1] = _ivld_4.y;
                staged_1b[16 + 2] = _ivld_4.z;
                staged_1b[16 + 3] = _ivld_4.w;
            }
        }
        dst_offsets_1b[5] = (long long)(bid * 2 + 1) * k_stride_0_bytes + (long long)(tid / 4) * k_stride_1_bytes + 128 + (long long)(tid - tid / 4 * 4) * 16;
        if (bid * 2 + 1 < tokens) {
            {
                const int4* _ivptr_5 = reinterpret_cast<const int4*>(k_rope + ((long long)(bid * 2 + 1) * k_rope_stride_0_bytes + (long long)(tid - tid / 4 * 4) * 16) + 0);
                int4 _ivld_5;
                _ivld_5 = *_ivptr_5;
                staged_1b[20 + 0] = _ivld_5.x;
                staged_1b[20 + 1] = _ivld_5.y;
                staged_1b[20 + 2] = _ivld_5.z;
                staged_1b[20 + 3] = _ivld_5.w;
            }
        }
        if (bid * 2 < tokens) {
            {
                int4 _iv4 = make_int4(staged_1b[0 + 0], staged_1b[0 + 1], staged_1b[0 + 2], staged_1b[0 + 3]);
                *reinterpret_cast<int4*>(k + dst_offsets_1b[0] + 0) = _iv4;
            }
        }
        if (bid * 2 < tokens) {
            {
                int4 _iv4 = make_int4(staged_1b[4 + 0], staged_1b[4 + 1], staged_1b[4 + 2], staged_1b[4 + 3]);
                *reinterpret_cast<int4*>(k + dst_offsets_1b[1] + 0) = _iv4;
            }
        }
        if (bid * 2 < tokens) {
            {
                int4 _iv4 = make_int4(staged_1b[8 + 0], staged_1b[8 + 1], staged_1b[8 + 2], staged_1b[8 + 3]);
                *reinterpret_cast<int4*>(k + dst_offsets_1b[2] + 0) = _iv4;
            }
        }
        if (bid * 2 + 1 < tokens) {
            {
                int4 _iv4 = make_int4(staged_1b[12 + 0], staged_1b[12 + 1], staged_1b[12 + 2], staged_1b[12 + 3]);
                *reinterpret_cast<int4*>(k + dst_offsets_1b[3] + 0) = _iv4;
            }
        }
        if (bid * 2 + 1 < tokens) {
            {
                int4 _iv4 = make_int4(staged_1b[16 + 0], staged_1b[16 + 1], staged_1b[16 + 2], staged_1b[16 + 3]);
                *reinterpret_cast<int4*>(k + dst_offsets_1b[4] + 0) = _iv4;
            }
        }
        if (bid * 2 + 1 < tokens) {
            {
                int4 _iv4 = make_int4(staged_1b[20 + 0], staged_1b[20 + 1], staged_1b[20 + 2], staged_1b[20 + 3]);
                *reinterpret_cast<int4*>(k + dst_offsets_1b[5] + 0) = _iv4;
            }
        }
    } else {
        int staged_2b[24] = {0};
        long long dst_offsets_2b[6] = {0};
        dst_offsets_2b[0] = (long long)bid * k_stride_0_bytes + (long long)(tid / 16) * k_stride_1_bytes + (long long)(tid - tid / 16 * 16) * 16;
        if (bid < tokens) {
            {
                const int4* _ivptr_6 = reinterpret_cast<const int4*>(k_nope + ((long long)bid * k_nope_stride_0_bytes + (long long)(tid / 16) * k_nope_stride_1_bytes + (long long)(tid - tid / 16 * 16) * 16) + 0);
                int4 _ivld_6;
                _ivld_6 = *_ivptr_6;
                staged_2b[0 + 0] = _ivld_6.x;
                staged_2b[0 + 1] = _ivld_6.y;
                staged_2b[0 + 2] = _ivld_6.z;
                staged_2b[0 + 3] = _ivld_6.w;
            }
        }
        dst_offsets_2b[1] = (long long)bid * k_stride_0_bytes + (long long)((tid + 512) / 16) * k_stride_1_bytes + (long long)(tid + 512 - (tid + 512) / 16 * 16) * 16;
        if (bid < tokens) {
            {
                const int4* _ivptr_7 = reinterpret_cast<const int4*>(k_nope + ((long long)bid * k_nope_stride_0_bytes + (long long)((tid + 512) / 16) * k_nope_stride_1_bytes + (long long)(tid + 512 - (tid + 512) / 16 * 16) * 16) + 0);
                int4 _ivld_7;
                _ivld_7 = *_ivptr_7;
                staged_2b[4 + 0] = _ivld_7.x;
                staged_2b[4 + 1] = _ivld_7.y;
                staged_2b[4 + 2] = _ivld_7.z;
                staged_2b[4 + 3] = _ivld_7.w;
            }
        }
        dst_offsets_2b[2] = (long long)bid * k_stride_0_bytes + (long long)((tid + 1024) / 16) * k_stride_1_bytes + (long long)(tid + 1024 - (tid + 1024) / 16 * 16) * 16;
        if (bid < tokens) {
            {
                const int4* _ivptr_8 = reinterpret_cast<const int4*>(k_nope + ((long long)bid * k_nope_stride_0_bytes + (long long)((tid + 1024) / 16) * k_nope_stride_1_bytes + (long long)(tid + 1024 - (tid + 1024) / 16 * 16) * 16) + 0);
                int4 _ivld_8;
                _ivld_8 = *_ivptr_8;
                staged_2b[8 + 0] = _ivld_8.x;
                staged_2b[8 + 1] = _ivld_8.y;
                staged_2b[8 + 2] = _ivld_8.z;
                staged_2b[8 + 3] = _ivld_8.w;
            }
        }
        dst_offsets_2b[3] = (long long)bid * k_stride_0_bytes + (long long)((tid + 1536) / 16) * k_stride_1_bytes + (long long)(tid + 1536 - (tid + 1536) / 16 * 16) * 16;
        if (bid < tokens) {
            {
                const int4* _ivptr_9 = reinterpret_cast<const int4*>(k_nope + ((long long)bid * k_nope_stride_0_bytes + (long long)((tid + 1536) / 16) * k_nope_stride_1_bytes + (long long)(tid + 1536 - (tid + 1536) / 16 * 16) * 16) + 0);
                int4 _ivld_9;
                _ivld_9 = *_ivptr_9;
                staged_2b[12 + 0] = _ivld_9.x;
                staged_2b[12 + 1] = _ivld_9.y;
                staged_2b[12 + 2] = _ivld_9.z;
                staged_2b[12 + 3] = _ivld_9.w;
            }
        }
        dst_offsets_2b[4] = (long long)bid * k_stride_0_bytes + (long long)(tid / 8) * k_stride_1_bytes + 256 + (long long)(tid - tid / 8 * 8) * 16;
        if (bid < tokens) {
            {
                const int4* _ivptr_10 = reinterpret_cast<const int4*>(k_rope + ((long long)bid * k_rope_stride_0_bytes + (long long)(tid - tid / 8 * 8) * 16) + 0);
                int4 _ivld_10;
                _ivld_10 = *_ivptr_10;
                staged_2b[16 + 0] = _ivld_10.x;
                staged_2b[16 + 1] = _ivld_10.y;
                staged_2b[16 + 2] = _ivld_10.z;
                staged_2b[16 + 3] = _ivld_10.w;
            }
        }
        dst_offsets_2b[5] = (long long)bid * k_stride_0_bytes + (long long)((tid + 512) / 8) * k_stride_1_bytes + 256 + (long long)(tid + 512 - (tid + 512) / 8 * 8) * 16;
        if (bid < tokens) {
            {
                const int4* _ivptr_11 = reinterpret_cast<const int4*>(k_rope + ((long long)bid * k_rope_stride_0_bytes + (long long)(tid + 512 - (tid + 512) / 8 * 8) * 16) + 0);
                int4 _ivld_11;
                _ivld_11 = *_ivptr_11;
                staged_2b[20 + 0] = _ivld_11.x;
                staged_2b[20 + 1] = _ivld_11.y;
                staged_2b[20 + 2] = _ivld_11.z;
                staged_2b[20 + 3] = _ivld_11.w;
            }
        }
        if (bid < tokens) {
            {
                int4 _iv4 = make_int4(staged_2b[0 + 0], staged_2b[0 + 1], staged_2b[0 + 2], staged_2b[0 + 3]);
                *reinterpret_cast<int4*>(k + dst_offsets_2b[0] + 0) = _iv4;
            }
        }
        if (bid < tokens) {
            {
                int4 _iv4 = make_int4(staged_2b[4 + 0], staged_2b[4 + 1], staged_2b[4 + 2], staged_2b[4 + 3]);
                *reinterpret_cast<int4*>(k + dst_offsets_2b[1] + 0) = _iv4;
            }
        }
        if (bid < tokens) {
            {
                int4 _iv4 = make_int4(staged_2b[8 + 0], staged_2b[8 + 1], staged_2b[8 + 2], staged_2b[8 + 3]);
                *reinterpret_cast<int4*>(k + dst_offsets_2b[2] + 0) = _iv4;
            }
        }
        if (bid < tokens) {
            {
                int4 _iv4 = make_int4(staged_2b[12 + 0], staged_2b[12 + 1], staged_2b[12 + 2], staged_2b[12 + 3]);
                *reinterpret_cast<int4*>(k + dst_offsets_2b[3] + 0) = _iv4;
            }
        }
        if (bid < tokens) {
            {
                int4 _iv4 = make_int4(staged_2b[16 + 0], staged_2b[16 + 1], staged_2b[16 + 2], staged_2b[16 + 3]);
                *reinterpret_cast<int4*>(k + dst_offsets_2b[4] + 0) = _iv4;
            }
        }
        if (bid < tokens) {
            {
                int4 _iv4 = make_int4(staged_2b[20 + 0], staged_2b[20 + 1], staged_2b[20 + 2], staged_2b[20 + 3]);
                *reinterpret_cast<int4*>(k + dst_offsets_2b[5] + 0) = _iv4;
            }
        }
    }
}

} // extern "C"
