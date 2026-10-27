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
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_warp_decode_6729f7918be8ef4c8c1e(__nv_bfloat16* __restrict__ route_outputs, __nv_bfloat16* __restrict__ route_weights, int* __restrict__ route_slots, __nv_bfloat16* __restrict__ output, int top_k, int num_tokens, int route_stride, int M)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int feature = (blockIdx.x * 128 + tid) * 4;
    int token = blockIdx.y;
    int route_base = token * 10;
    float weights[10];
    if (feature < M && token < num_tokens) {
        weights[0] = (float)route_weights[route_base];
        weights[1] = (float)route_weights[route_base + 1];
        weights[2] = (float)route_weights[route_base + 2];
        weights[3] = (float)route_weights[route_base + 3];
        weights[4] = (float)route_weights[route_base + 4];
        weights[5] = (float)route_weights[route_base + 5];
        weights[6] = (float)route_weights[route_base + 6];
        weights[7] = (float)route_weights[route_base + 7];
        weights[8] = (float)route_weights[route_base + 8];
        weights[9] = (float)route_weights[route_base + 9];
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (feature < M && token < num_tokens) {
        int rows[10];
        rows[0] = route_slots[route_base];
        rows[1] = route_slots[route_base + 1];
        rows[2] = route_slots[route_base + 2];
        rows[3] = route_slots[route_base + 3];
        rows[4] = route_slots[route_base + 4];
        rows[5] = route_slots[route_base + 5];
        rows[6] = route_slots[route_base + 6];
        rows[7] = route_slots[route_base + 7];
        rows[8] = route_slots[route_base + 8];
        rows[9] = route_slots[route_base + 9];
        unsigned int words[20];
        {
            const uint2* _ivptr_0 = reinterpret_cast<const uint2*>(route_outputs + (rows[0] * route_stride + feature) + 0);
            uint2 _ivld_0;
            _ivld_0 = *_ivptr_0;
            words[0 + 0] = _ivld_0.x;
            words[0 + 1] = _ivld_0.y;
        }
        {
            const uint2* _ivptr_1 = reinterpret_cast<const uint2*>(route_outputs + (rows[1] * route_stride + feature) + 0);
            uint2 _ivld_1;
            _ivld_1 = *_ivptr_1;
            words[2 + 0] = _ivld_1.x;
            words[2 + 1] = _ivld_1.y;
        }
        {
            const uint2* _ivptr_2 = reinterpret_cast<const uint2*>(route_outputs + (rows[2] * route_stride + feature) + 0);
            uint2 _ivld_2;
            _ivld_2 = *_ivptr_2;
            words[4 + 0] = _ivld_2.x;
            words[4 + 1] = _ivld_2.y;
        }
        {
            const uint2* _ivptr_3 = reinterpret_cast<const uint2*>(route_outputs + (rows[3] * route_stride + feature) + 0);
            uint2 _ivld_3;
            _ivld_3 = *_ivptr_3;
            words[6 + 0] = _ivld_3.x;
            words[6 + 1] = _ivld_3.y;
        }
        {
            const uint2* _ivptr_4 = reinterpret_cast<const uint2*>(route_outputs + (rows[4] * route_stride + feature) + 0);
            uint2 _ivld_4;
            _ivld_4 = *_ivptr_4;
            words[8 + 0] = _ivld_4.x;
            words[8 + 1] = _ivld_4.y;
        }
        {
            const uint2* _ivptr_5 = reinterpret_cast<const uint2*>(route_outputs + (rows[5] * route_stride + feature) + 0);
            uint2 _ivld_5;
            _ivld_5 = *_ivptr_5;
            words[10 + 0] = _ivld_5.x;
            words[10 + 1] = _ivld_5.y;
        }
        {
            const uint2* _ivptr_6 = reinterpret_cast<const uint2*>(route_outputs + (rows[6] * route_stride + feature) + 0);
            uint2 _ivld_6;
            _ivld_6 = *_ivptr_6;
            words[12 + 0] = _ivld_6.x;
            words[12 + 1] = _ivld_6.y;
        }
        {
            const uint2* _ivptr_7 = reinterpret_cast<const uint2*>(route_outputs + (rows[7] * route_stride + feature) + 0);
            uint2 _ivld_7;
            _ivld_7 = *_ivptr_7;
            words[14 + 0] = _ivld_7.x;
            words[14 + 1] = _ivld_7.y;
        }
        {
            const uint2* _ivptr_8 = reinterpret_cast<const uint2*>(route_outputs + (rows[8] * route_stride + feature) + 0);
            uint2 _ivld_8;
            _ivld_8 = *_ivptr_8;
            words[16 + 0] = _ivld_8.x;
            words[16 + 1] = _ivld_8.y;
        }
        {
            const uint2* _ivptr_9 = reinterpret_cast<const uint2*>(route_outputs + (rows[9] * route_stride + feature) + 0);
            uint2 _ivld_9;
            _ivld_9 = *_ivptr_9;
            words[18 + 0] = _ivld_9.x;
            words[18 + 1] = _ivld_9.y;
        }
        float values[4];
        float partials[4];
        values[0] = 0.0f;
        values[1] = 0.0f;
        values[2] = 0.0f;
        values[3] = 0.0f;
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[_pair]) << 16);
            (&partials[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[1 + _pair]) << 16);
            (&partials[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[1 + _pair]) & 0xffff0000u);
        }
        values[0] = __fmaf_rn(partials[0], weights[0], values[0]);
        values[1] = __fmaf_rn(partials[1], weights[0], values[1]);
        values[2] = __fmaf_rn(partials[2], weights[0], values[2]);
        values[3] = __fmaf_rn(partials[3], weights[0], values[3]);
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[2 + _pair]) << 16);
            (&partials[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[2 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[3 + _pair]) << 16);
            (&partials[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[3 + _pair]) & 0xffff0000u);
        }
        values[0] = __fmaf_rn(partials[0], weights[1], values[0]);
        values[1] = __fmaf_rn(partials[1], weights[1], values[1]);
        values[2] = __fmaf_rn(partials[2], weights[1], values[2]);
        values[3] = __fmaf_rn(partials[3], weights[1], values[3]);
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[4 + _pair]) << 16);
            (&partials[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[4 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[5 + _pair]) << 16);
            (&partials[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[5 + _pair]) & 0xffff0000u);
        }
        values[0] = __fmaf_rn(partials[0], weights[2], values[0]);
        values[1] = __fmaf_rn(partials[1], weights[2], values[1]);
        values[2] = __fmaf_rn(partials[2], weights[2], values[2]);
        values[3] = __fmaf_rn(partials[3], weights[2], values[3]);
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[6 + _pair]) << 16);
            (&partials[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[6 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[7 + _pair]) << 16);
            (&partials[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[7 + _pair]) & 0xffff0000u);
        }
        values[0] = __fmaf_rn(partials[0], weights[3], values[0]);
        values[1] = __fmaf_rn(partials[1], weights[3], values[1]);
        values[2] = __fmaf_rn(partials[2], weights[3], values[2]);
        values[3] = __fmaf_rn(partials[3], weights[3], values[3]);
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[8 + _pair]) << 16);
            (&partials[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[8 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[9 + _pair]) << 16);
            (&partials[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[9 + _pair]) & 0xffff0000u);
        }
        values[0] = __fmaf_rn(partials[0], weights[4], values[0]);
        values[1] = __fmaf_rn(partials[1], weights[4], values[1]);
        values[2] = __fmaf_rn(partials[2], weights[4], values[2]);
        values[3] = __fmaf_rn(partials[3], weights[4], values[3]);
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[10 + _pair]) << 16);
            (&partials[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[10 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[11 + _pair]) << 16);
            (&partials[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[11 + _pair]) & 0xffff0000u);
        }
        values[0] = __fmaf_rn(partials[0], weights[5], values[0]);
        values[1] = __fmaf_rn(partials[1], weights[5], values[1]);
        values[2] = __fmaf_rn(partials[2], weights[5], values[2]);
        values[3] = __fmaf_rn(partials[3], weights[5], values[3]);
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[12 + _pair]) << 16);
            (&partials[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[12 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[13 + _pair]) << 16);
            (&partials[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[13 + _pair]) & 0xffff0000u);
        }
        values[0] = __fmaf_rn(partials[0], weights[6], values[0]);
        values[1] = __fmaf_rn(partials[1], weights[6], values[1]);
        values[2] = __fmaf_rn(partials[2], weights[6], values[2]);
        values[3] = __fmaf_rn(partials[3], weights[6], values[3]);
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[14 + _pair]) << 16);
            (&partials[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[14 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[15 + _pair]) << 16);
            (&partials[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[15 + _pair]) & 0xffff0000u);
        }
        values[0] = __fmaf_rn(partials[0], weights[7], values[0]);
        values[1] = __fmaf_rn(partials[1], weights[7], values[1]);
        values[2] = __fmaf_rn(partials[2], weights[7], values[2]);
        values[3] = __fmaf_rn(partials[3], weights[7], values[3]);
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[16 + _pair]) << 16);
            (&partials[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[16 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[17 + _pair]) << 16);
            (&partials[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[17 + _pair]) & 0xffff0000u);
        }
        values[0] = __fmaf_rn(partials[0], weights[8], values[0]);
        values[1] = __fmaf_rn(partials[1], weights[8], values[1]);
        values[2] = __fmaf_rn(partials[2], weights[8], values[2]);
        values[3] = __fmaf_rn(partials[3], weights[8], values[3]);
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[18 + _pair]) << 16);
            (&partials[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[18 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&partials[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[19 + _pair]) << 16);
            (&partials[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[19 + _pair]) & 0xffff0000u);
        }
        values[0] = __fmaf_rn(partials[0], weights[9], values[0]);
        values[1] = __fmaf_rn(partials[1], weights[9], values[1]);
        values[2] = __fmaf_rn(partials[2], weights[9], values[2]);
        values[3] = __fmaf_rn(partials[3], weights[9], values[3]);
        unsigned int _packed_0[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
            _packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(output + (token * M + feature))[0] = reinterpret_cast<int2*>(_packed_0)[0];
    }
}

} // extern "C"
