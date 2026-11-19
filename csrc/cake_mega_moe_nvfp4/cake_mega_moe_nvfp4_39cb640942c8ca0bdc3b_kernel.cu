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

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(448) void
kernel_cake_mega_moe_nvfp4_39cb640942c8ca0bdc3b(unsigned int* __restrict__ slot_rows, int* __restrict__ topk_idx, unsigned int* __restrict__ out, int world)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int t = bid;
    int live = 0;
    unsigned int held_0[48];
    int vec0_0 = tid;
    unsigned int spec_0[4];
    {
        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(slot_rows + (t * 6 * 3584 + vec0_0 * 4) + 0);
        spec_0[0 + 0] = _uv4_0.x;
        spec_0[0 + 1] = _uv4_0.y;
        spec_0[0 + 2] = _uv4_0.z;
        spec_0[0 + 3] = _uv4_0.w;
    }
    held_0[0] = spec_0[0];
    held_0[1] = spec_0[1];
    held_0[2] = spec_0[2];
    held_0[3] = spec_0[3];
    unsigned int spec_1[4];
    {
        uint4 _uv4_1 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 6 + 1) * 3584 + vec0_0 * 4) + 0);
        spec_1[0 + 0] = _uv4_1.x;
        spec_1[0 + 1] = _uv4_1.y;
        spec_1[0 + 2] = _uv4_1.z;
        spec_1[0 + 3] = _uv4_1.w;
    }
    held_0[4] = spec_1[0];
    held_0[5] = spec_1[1];
    held_0[6] = spec_1[2];
    held_0[7] = spec_1[3];
    unsigned int spec_2[4];
    {
        uint4 _uv4_2 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 6 + 2) * 3584 + vec0_0 * 4) + 0);
        spec_2[0 + 0] = _uv4_2.x;
        spec_2[0 + 1] = _uv4_2.y;
        spec_2[0 + 2] = _uv4_2.z;
        spec_2[0 + 3] = _uv4_2.w;
    }
    held_0[8] = spec_2[0];
    held_0[9] = spec_2[1];
    held_0[10] = spec_2[2];
    held_0[11] = spec_2[3];
    unsigned int spec_3[4];
    {
        uint4 _uv4_3 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 6 + 3) * 3584 + vec0_0 * 4) + 0);
        spec_3[0 + 0] = _uv4_3.x;
        spec_3[0 + 1] = _uv4_3.y;
        spec_3[0 + 2] = _uv4_3.z;
        spec_3[0 + 3] = _uv4_3.w;
    }
    held_0[12] = spec_3[0];
    held_0[13] = spec_3[1];
    held_0[14] = spec_3[2];
    held_0[15] = spec_3[3];
    unsigned int spec_4[4];
    {
        uint4 _uv4_4 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 6 + 4) * 3584 + vec0_0 * 4) + 0);
        spec_4[0 + 0] = _uv4_4.x;
        spec_4[0 + 1] = _uv4_4.y;
        spec_4[0 + 2] = _uv4_4.z;
        spec_4[0 + 3] = _uv4_4.w;
    }
    held_0[16] = spec_4[0];
    held_0[17] = spec_4[1];
    held_0[18] = spec_4[2];
    held_0[19] = spec_4[3];
    unsigned int spec_5[4];
    {
        uint4 _uv4_5 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 6 + 5) * 3584 + vec0_0 * 4) + 0);
        spec_5[0 + 0] = _uv4_5.x;
        spec_5[0 + 1] = _uv4_5.y;
        spec_5[0 + 2] = _uv4_5.z;
        spec_5[0 + 3] = _uv4_5.w;
    }
    held_0[20] = spec_5[0];
    held_0[21] = spec_5[1];
    held_0[22] = spec_5[2];
    held_0[23] = spec_5[3];
    int vec0_1 = 448 + tid;
    unsigned int spec_6[4];
    {
        uint4 _uv4_6 = *reinterpret_cast<const uint4*>(slot_rows + (t * 6 * 3584 + vec0_1 * 4) + 0);
        spec_6[0 + 0] = _uv4_6.x;
        spec_6[0 + 1] = _uv4_6.y;
        spec_6[0 + 2] = _uv4_6.z;
        spec_6[0 + 3] = _uv4_6.w;
    }
    held_0[24] = spec_6[0];
    held_0[25] = spec_6[1];
    held_0[26] = spec_6[2];
    held_0[27] = spec_6[3];
    unsigned int spec_7[4];
    {
        uint4 _uv4_7 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 6 + 1) * 3584 + vec0_1 * 4) + 0);
        spec_7[0 + 0] = _uv4_7.x;
        spec_7[0 + 1] = _uv4_7.y;
        spec_7[0 + 2] = _uv4_7.z;
        spec_7[0 + 3] = _uv4_7.w;
    }
    held_0[28] = spec_7[0];
    held_0[29] = spec_7[1];
    held_0[30] = spec_7[2];
    held_0[31] = spec_7[3];
    unsigned int spec_8[4];
    {
        uint4 _uv4_8 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 6 + 2) * 3584 + vec0_1 * 4) + 0);
        spec_8[0 + 0] = _uv4_8.x;
        spec_8[0 + 1] = _uv4_8.y;
        spec_8[0 + 2] = _uv4_8.z;
        spec_8[0 + 3] = _uv4_8.w;
    }
    held_0[32] = spec_8[0];
    held_0[33] = spec_8[1];
    held_0[34] = spec_8[2];
    held_0[35] = spec_8[3];
    unsigned int spec_9[4];
    {
        uint4 _uv4_9 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 6 + 3) * 3584 + vec0_1 * 4) + 0);
        spec_9[0 + 0] = _uv4_9.x;
        spec_9[0 + 1] = _uv4_9.y;
        spec_9[0 + 2] = _uv4_9.z;
        spec_9[0 + 3] = _uv4_9.w;
    }
    held_0[36] = spec_9[0];
    held_0[37] = spec_9[1];
    held_0[38] = spec_9[2];
    held_0[39] = spec_9[3];
    unsigned int spec_10[4];
    {
        uint4 _uv4_10 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 6 + 4) * 3584 + vec0_1 * 4) + 0);
        spec_10[0 + 0] = _uv4_10.x;
        spec_10[0 + 1] = _uv4_10.y;
        spec_10[0 + 2] = _uv4_10.z;
        spec_10[0 + 3] = _uv4_10.w;
    }
    held_0[40] = spec_10[0];
    held_0[41] = spec_10[1];
    held_0[42] = spec_10[2];
    held_0[43] = spec_10[3];
    unsigned int spec_11[4];
    {
        uint4 _uv4_11 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 6 + 5) * 3584 + vec0_1 * 4) + 0);
        spec_11[0 + 0] = _uv4_11.x;
        spec_11[0 + 1] = _uv4_11.y;
        spec_11[0 + 2] = _uv4_11.z;
        spec_11[0 + 3] = _uv4_11.w;
    }
    held_0[44] = spec_11[0];
    held_0[45] = spec_11[1];
    held_0[46] = spec_11[2];
    held_0[47] = spec_11[3];
    int sid_0 = topk_idx[t * 6];
    if ((unsigned int)sid_0 < (unsigned int)(world * 12)) {
        live = live | 1;
    }
    int sid_1 = topk_idx[t * 6 + 1];
    if ((unsigned int)sid_1 < (unsigned int)(world * 12)) {
        live = live | 2;
    }
    int sid_2 = topk_idx[t * 6 + 2];
    if ((unsigned int)sid_2 < (unsigned int)(world * 12)) {
        live = live | 4;
    }
    int sid_3 = topk_idx[t * 6 + 3];
    if ((unsigned int)sid_3 < (unsigned int)(world * 12)) {
        live = live | 8;
    }
    int sid_4 = topk_idx[t * 6 + 4];
    if ((unsigned int)sid_4 < (unsigned int)(world * 12)) {
        live = live | 16;
    }
    int sid_5 = topk_idx[t * 6 + 5];
    if ((unsigned int)sid_5 < (unsigned int)(world * 12)) {
        live = live | 32;
    }
    int vec_0 = tid;
    float acc_0[8];
    acc_0[0] = 0.0f;
    acc_0[1] = 0.0f;
    acc_0[2] = 0.0f;
    acc_0[3] = 0.0f;
    acc_0[4] = 0.0f;
    acc_0[5] = 0.0f;
    acc_0[6] = 0.0f;
    acc_0[7] = 0.0f;
    bool on_0 = (live & 1) != 0;
    unsigned int words_0[4];
    unsigned int word_0 = ((on_0) ? held_0[0] : 2147516416u);
    words_0[0] = word_0;
    unsigned int word_1 = ((on_0) ? held_0[1] : 2147516416u);
    words_0[1] = word_1;
    unsigned int word_2 = ((on_0) ? held_0[2] : 2147516416u);
    words_0[2] = word_2;
    unsigned int word_3 = ((on_0) ? held_0[3] : 2147516416u);
    words_0[3] = word_3;
    float vals_0[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&vals_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_0[_pair]) << 16);
        (&vals_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_0[_pair]) & 0xffff0000u);
    }
    acc_0[0] = acc_0[0] + vals_0[0];
    acc_0[1] = acc_0[1] + vals_0[1];
    acc_0[2] = acc_0[2] + vals_0[2];
    acc_0[3] = acc_0[3] + vals_0[3];
    acc_0[4] = acc_0[4] + vals_0[4];
    acc_0[5] = acc_0[5] + vals_0[5];
    acc_0[6] = acc_0[6] + vals_0[6];
    acc_0[7] = acc_0[7] + vals_0[7];
    bool on_1 = (live >> 1 & 1) != 0;
    unsigned int words_1[4];
    unsigned int word_4 = ((on_1) ? held_0[4] : 2147516416u);
    words_1[0] = word_4;
    unsigned int word_5 = ((on_1) ? held_0[5] : 2147516416u);
    words_1[1] = word_5;
    unsigned int word_6 = ((on_1) ? held_0[6] : 2147516416u);
    words_1[2] = word_6;
    unsigned int word_7 = ((on_1) ? held_0[7] : 2147516416u);
    words_1[3] = word_7;
    float vals_1[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&vals_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[_pair]) << 16);
        (&vals_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[_pair]) & 0xffff0000u);
    }
    acc_0[0] = acc_0[0] + vals_1[0];
    acc_0[1] = acc_0[1] + vals_1[1];
    acc_0[2] = acc_0[2] + vals_1[2];
    acc_0[3] = acc_0[3] + vals_1[3];
    acc_0[4] = acc_0[4] + vals_1[4];
    acc_0[5] = acc_0[5] + vals_1[5];
    acc_0[6] = acc_0[6] + vals_1[6];
    acc_0[7] = acc_0[7] + vals_1[7];
    bool on_2 = (live >> 2 & 1) != 0;
    unsigned int words_2[4];
    unsigned int word_8 = ((on_2) ? held_0[8] : 2147516416u);
    words_2[0] = word_8;
    unsigned int word_9 = ((on_2) ? held_0[9] : 2147516416u);
    words_2[1] = word_9;
    unsigned int word_10 = ((on_2) ? held_0[10] : 2147516416u);
    words_2[2] = word_10;
    unsigned int word_11 = ((on_2) ? held_0[11] : 2147516416u);
    words_2[3] = word_11;
    float vals_2[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&vals_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_2[_pair]) << 16);
        (&vals_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_2[_pair]) & 0xffff0000u);
    }
    acc_0[0] = acc_0[0] + vals_2[0];
    acc_0[1] = acc_0[1] + vals_2[1];
    acc_0[2] = acc_0[2] + vals_2[2];
    acc_0[3] = acc_0[3] + vals_2[3];
    acc_0[4] = acc_0[4] + vals_2[4];
    acc_0[5] = acc_0[5] + vals_2[5];
    acc_0[6] = acc_0[6] + vals_2[6];
    acc_0[7] = acc_0[7] + vals_2[7];
    bool on_3 = (live >> 3 & 1) != 0;
    unsigned int words_3[4];
    unsigned int word_12 = ((on_3) ? held_0[12] : 2147516416u);
    words_3[0] = word_12;
    unsigned int word_13 = ((on_3) ? held_0[13] : 2147516416u);
    words_3[1] = word_13;
    unsigned int word_14 = ((on_3) ? held_0[14] : 2147516416u);
    words_3[2] = word_14;
    unsigned int word_15 = ((on_3) ? held_0[15] : 2147516416u);
    words_3[3] = word_15;
    float vals_3[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&vals_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_3[_pair]) << 16);
        (&vals_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_3[_pair]) & 0xffff0000u);
    }
    acc_0[0] = acc_0[0] + vals_3[0];
    acc_0[1] = acc_0[1] + vals_3[1];
    acc_0[2] = acc_0[2] + vals_3[2];
    acc_0[3] = acc_0[3] + vals_3[3];
    acc_0[4] = acc_0[4] + vals_3[4];
    acc_0[5] = acc_0[5] + vals_3[5];
    acc_0[6] = acc_0[6] + vals_3[6];
    acc_0[7] = acc_0[7] + vals_3[7];
    bool on_4 = (live >> 4 & 1) != 0;
    unsigned int words_4[4];
    unsigned int word_16 = ((on_4) ? held_0[16] : 2147516416u);
    words_4[0] = word_16;
    unsigned int word_17 = ((on_4) ? held_0[17] : 2147516416u);
    words_4[1] = word_17;
    unsigned int word_18 = ((on_4) ? held_0[18] : 2147516416u);
    words_4[2] = word_18;
    unsigned int word_19 = ((on_4) ? held_0[19] : 2147516416u);
    words_4[3] = word_19;
    float vals_4[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&vals_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_4[_pair]) << 16);
        (&vals_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_4[_pair]) & 0xffff0000u);
    }
    acc_0[0] = acc_0[0] + vals_4[0];
    acc_0[1] = acc_0[1] + vals_4[1];
    acc_0[2] = acc_0[2] + vals_4[2];
    acc_0[3] = acc_0[3] + vals_4[3];
    acc_0[4] = acc_0[4] + vals_4[4];
    acc_0[5] = acc_0[5] + vals_4[5];
    acc_0[6] = acc_0[6] + vals_4[6];
    acc_0[7] = acc_0[7] + vals_4[7];
    bool on_5 = (live >> 5 & 1) != 0;
    unsigned int words_5[4];
    unsigned int word_20 = ((on_5) ? held_0[20] : 2147516416u);
    words_5[0] = word_20;
    unsigned int word_21 = ((on_5) ? held_0[21] : 2147516416u);
    words_5[1] = word_21;
    unsigned int word_22 = ((on_5) ? held_0[22] : 2147516416u);
    words_5[2] = word_22;
    unsigned int word_23 = ((on_5) ? held_0[23] : 2147516416u);
    words_5[3] = word_23;
    float vals_5[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&vals_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_5[_pair]) << 16);
        (&vals_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_5[_pair]) & 0xffff0000u);
    }
    acc_0[0] = acc_0[0] + vals_5[0];
    acc_0[1] = acc_0[1] + vals_5[1];
    acc_0[2] = acc_0[2] + vals_5[2];
    acc_0[3] = acc_0[3] + vals_5[3];
    acc_0[4] = acc_0[4] + vals_5[4];
    acc_0[5] = acc_0[5] + vals_5[5];
    acc_0[6] = acc_0[6] + vals_5[6];
    acc_0[7] = acc_0[7] + vals_5[7];
    unsigned int packed_0[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_0[_lp*2 + 0], acc_0[_lp*2+1 + 0]));
        packed_0[_lp] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(out + (t * 3584 + vec_0 * 4))[0] = reinterpret_cast<int4*>(packed_0)[0];
    int vec_1 = 448 + tid;
    float acc_1[8];
    acc_1[0] = 0.0f;
    acc_1[1] = 0.0f;
    acc_1[2] = 0.0f;
    acc_1[3] = 0.0f;
    acc_1[4] = 0.0f;
    acc_1[5] = 0.0f;
    acc_1[6] = 0.0f;
    acc_1[7] = 0.0f;
    bool on_6 = (live & 1) != 0;
    unsigned int words_6[4];
    unsigned int word_24 = ((on_6) ? held_0[24] : 2147516416u);
    words_6[0] = word_24;
    unsigned int word_25 = ((on_6) ? held_0[25] : 2147516416u);
    words_6[1] = word_25;
    unsigned int word_26 = ((on_6) ? held_0[26] : 2147516416u);
    words_6[2] = word_26;
    unsigned int word_27 = ((on_6) ? held_0[27] : 2147516416u);
    words_6[3] = word_27;
    float vals_6[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&vals_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_6[_pair]) << 16);
        (&vals_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_6[_pair]) & 0xffff0000u);
    }
    acc_1[0] = acc_1[0] + vals_6[0];
    acc_1[1] = acc_1[1] + vals_6[1];
    acc_1[2] = acc_1[2] + vals_6[2];
    acc_1[3] = acc_1[3] + vals_6[3];
    acc_1[4] = acc_1[4] + vals_6[4];
    acc_1[5] = acc_1[5] + vals_6[5];
    acc_1[6] = acc_1[6] + vals_6[6];
    acc_1[7] = acc_1[7] + vals_6[7];
    bool on_7 = (live >> 1 & 1) != 0;
    unsigned int words_7[4];
    unsigned int word_28 = ((on_7) ? held_0[28] : 2147516416u);
    words_7[0] = word_28;
    unsigned int word_29 = ((on_7) ? held_0[29] : 2147516416u);
    words_7[1] = word_29;
    unsigned int word_30 = ((on_7) ? held_0[30] : 2147516416u);
    words_7[2] = word_30;
    unsigned int word_31 = ((on_7) ? held_0[31] : 2147516416u);
    words_7[3] = word_31;
    float vals_7[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&vals_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_7[_pair]) << 16);
        (&vals_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_7[_pair]) & 0xffff0000u);
    }
    acc_1[0] = acc_1[0] + vals_7[0];
    acc_1[1] = acc_1[1] + vals_7[1];
    acc_1[2] = acc_1[2] + vals_7[2];
    acc_1[3] = acc_1[3] + vals_7[3];
    acc_1[4] = acc_1[4] + vals_7[4];
    acc_1[5] = acc_1[5] + vals_7[5];
    acc_1[6] = acc_1[6] + vals_7[6];
    acc_1[7] = acc_1[7] + vals_7[7];
    bool on_8 = (live >> 2 & 1) != 0;
    unsigned int words_8[4];
    unsigned int word_32 = ((on_8) ? held_0[32] : 2147516416u);
    words_8[0] = word_32;
    unsigned int word_33 = ((on_8) ? held_0[33] : 2147516416u);
    words_8[1] = word_33;
    unsigned int word_34 = ((on_8) ? held_0[34] : 2147516416u);
    words_8[2] = word_34;
    unsigned int word_35 = ((on_8) ? held_0[35] : 2147516416u);
    words_8[3] = word_35;
    float vals_8[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&vals_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_8[_pair]) << 16);
        (&vals_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_8[_pair]) & 0xffff0000u);
    }
    acc_1[0] = acc_1[0] + vals_8[0];
    acc_1[1] = acc_1[1] + vals_8[1];
    acc_1[2] = acc_1[2] + vals_8[2];
    acc_1[3] = acc_1[3] + vals_8[3];
    acc_1[4] = acc_1[4] + vals_8[4];
    acc_1[5] = acc_1[5] + vals_8[5];
    acc_1[6] = acc_1[6] + vals_8[6];
    acc_1[7] = acc_1[7] + vals_8[7];
    bool on_9 = (live >> 3 & 1) != 0;
    unsigned int words_9[4];
    unsigned int word_36 = ((on_9) ? held_0[36] : 2147516416u);
    words_9[0] = word_36;
    unsigned int word_37 = ((on_9) ? held_0[37] : 2147516416u);
    words_9[1] = word_37;
    unsigned int word_38 = ((on_9) ? held_0[38] : 2147516416u);
    words_9[2] = word_38;
    unsigned int word_39 = ((on_9) ? held_0[39] : 2147516416u);
    words_9[3] = word_39;
    float vals_9[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&vals_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_9[_pair]) << 16);
        (&vals_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_9[_pair]) & 0xffff0000u);
    }
    acc_1[0] = acc_1[0] + vals_9[0];
    acc_1[1] = acc_1[1] + vals_9[1];
    acc_1[2] = acc_1[2] + vals_9[2];
    acc_1[3] = acc_1[3] + vals_9[3];
    acc_1[4] = acc_1[4] + vals_9[4];
    acc_1[5] = acc_1[5] + vals_9[5];
    acc_1[6] = acc_1[6] + vals_9[6];
    acc_1[7] = acc_1[7] + vals_9[7];
    bool on_10 = (live >> 4 & 1) != 0;
    unsigned int words_10[4];
    unsigned int word_40 = ((on_10) ? held_0[40] : 2147516416u);
    words_10[0] = word_40;
    unsigned int word_41 = ((on_10) ? held_0[41] : 2147516416u);
    words_10[1] = word_41;
    unsigned int word_42 = ((on_10) ? held_0[42] : 2147516416u);
    words_10[2] = word_42;
    unsigned int word_43 = ((on_10) ? held_0[43] : 2147516416u);
    words_10[3] = word_43;
    float vals_10[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&vals_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_10[_pair]) << 16);
        (&vals_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_10[_pair]) & 0xffff0000u);
    }
    acc_1[0] = acc_1[0] + vals_10[0];
    acc_1[1] = acc_1[1] + vals_10[1];
    acc_1[2] = acc_1[2] + vals_10[2];
    acc_1[3] = acc_1[3] + vals_10[3];
    acc_1[4] = acc_1[4] + vals_10[4];
    acc_1[5] = acc_1[5] + vals_10[5];
    acc_1[6] = acc_1[6] + vals_10[6];
    acc_1[7] = acc_1[7] + vals_10[7];
    bool on_11 = (live >> 5 & 1) != 0;
    unsigned int words_11[4];
    unsigned int word_44 = ((on_11) ? held_0[44] : 2147516416u);
    words_11[0] = word_44;
    unsigned int word_45 = ((on_11) ? held_0[45] : 2147516416u);
    words_11[1] = word_45;
    unsigned int word_46 = ((on_11) ? held_0[46] : 2147516416u);
    words_11[2] = word_46;
    unsigned int word_47 = ((on_11) ? held_0[47] : 2147516416u);
    words_11[3] = word_47;
    float vals_11[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&vals_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_11[_pair]) << 16);
        (&vals_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_11[_pair]) & 0xffff0000u);
    }
    acc_1[0] = acc_1[0] + vals_11[0];
    acc_1[1] = acc_1[1] + vals_11[1];
    acc_1[2] = acc_1[2] + vals_11[2];
    acc_1[3] = acc_1[3] + vals_11[3];
    acc_1[4] = acc_1[4] + vals_11[4];
    acc_1[5] = acc_1[5] + vals_11[5];
    acc_1[6] = acc_1[6] + vals_11[6];
    acc_1[7] = acc_1[7] + vals_11[7];
    unsigned int packed_1[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_1[_lp*2 + 0], acc_1[_lp*2+1 + 0]));
        packed_1[_lp] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(out + (t * 3584 + vec_1 * 4))[0] = reinterpret_cast<int4*>(packed_1)[0];
}

} // extern "C"
