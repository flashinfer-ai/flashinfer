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

__global__ __launch_bounds__(128) void
kernel_cake_mega_moe_nvfp4_329085c326b4fc170d5c(unsigned int* __restrict__ slot_rows, int* __restrict__ topk_idx, unsigned int* __restrict__ out, int world)
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
    int eid_0 = topk_idx[t * 6];
    if ((unsigned int)eid_0 < (unsigned int)(world * 12)) {
        live = live | 1 << eid_0 / 12;
    }
    int eid_1 = topk_idx[t * 6 + 1];
    if ((unsigned int)eid_1 < (unsigned int)(world * 12)) {
        live = live | 1 << eid_1 / 12;
    }
    int eid_2 = topk_idx[t * 6 + 2];
    if ((unsigned int)eid_2 < (unsigned int)(world * 12)) {
        live = live | 1 << eid_2 / 12;
    }
    int eid_3 = topk_idx[t * 6 + 3];
    if ((unsigned int)eid_3 < (unsigned int)(world * 12)) {
        live = live | 1 << eid_3 / 12;
    }
    int eid_4 = topk_idx[t * 6 + 4];
    if ((unsigned int)eid_4 < (unsigned int)(world * 12)) {
        live = live | 1 << eid_4 / 12;
    }
    int eid_5 = topk_idx[t * 6 + 5];
    if ((unsigned int)eid_5 < (unsigned int)(world * 12)) {
        live = live | 1 << eid_5 / 12;
    }
    unsigned int held_0[112];
    int vec0_0 = tid;
    held_0[0] = 0;
    held_0[1] = 0;
    held_0[2] = 0;
    held_0[3] = 0;
    if ((live & 1) != 0) {
        unsigned int row_0[4];
        {
            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(slot_rows + (t * 8 * 3584 + vec0_0 * 4) + 0);
            row_0[0 + 0] = _uv4_0.x;
            row_0[0 + 1] = _uv4_0.y;
            row_0[0 + 2] = _uv4_0.z;
            row_0[0 + 3] = _uv4_0.w;
        }
        held_0[0] = row_0[0];
        held_0[1] = row_0[1];
        held_0[2] = row_0[2];
        held_0[3] = row_0[3];
    }
    held_0[4] = 0;
    held_0[5] = 0;
    held_0[6] = 0;
    held_0[7] = 0;
    if ((live >> 1 & 1) != 0) {
        unsigned int row_1[4];
        {
            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 1) * 3584 + vec0_0 * 4) + 0);
            row_1[0 + 0] = _uv4_1.x;
            row_1[0 + 1] = _uv4_1.y;
            row_1[0 + 2] = _uv4_1.z;
            row_1[0 + 3] = _uv4_1.w;
        }
        held_0[4] = row_1[0];
        held_0[5] = row_1[1];
        held_0[6] = row_1[2];
        held_0[7] = row_1[3];
    }
    held_0[8] = 0;
    held_0[9] = 0;
    held_0[10] = 0;
    held_0[11] = 0;
    if ((live >> 2 & 1) != 0) {
        unsigned int row_2[4];
        {
            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 2) * 3584 + vec0_0 * 4) + 0);
            row_2[0 + 0] = _uv4_2.x;
            row_2[0 + 1] = _uv4_2.y;
            row_2[0 + 2] = _uv4_2.z;
            row_2[0 + 3] = _uv4_2.w;
        }
        held_0[8] = row_2[0];
        held_0[9] = row_2[1];
        held_0[10] = row_2[2];
        held_0[11] = row_2[3];
    }
    held_0[12] = 0;
    held_0[13] = 0;
    held_0[14] = 0;
    held_0[15] = 0;
    if ((live >> 3 & 1) != 0) {
        unsigned int row_3[4];
        {
            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 3) * 3584 + vec0_0 * 4) + 0);
            row_3[0 + 0] = _uv4_3.x;
            row_3[0 + 1] = _uv4_3.y;
            row_3[0 + 2] = _uv4_3.z;
            row_3[0 + 3] = _uv4_3.w;
        }
        held_0[12] = row_3[0];
        held_0[13] = row_3[1];
        held_0[14] = row_3[2];
        held_0[15] = row_3[3];
    }
    int vec0_1 = 128 + tid;
    held_0[16] = 0;
    held_0[17] = 0;
    held_0[18] = 0;
    held_0[19] = 0;
    if ((live & 1) != 0) {
        unsigned int row_4[4];
        {
            uint4 _uv4_4 = *reinterpret_cast<const uint4*>(slot_rows + (t * 8 * 3584 + vec0_1 * 4) + 0);
            row_4[0 + 0] = _uv4_4.x;
            row_4[0 + 1] = _uv4_4.y;
            row_4[0 + 2] = _uv4_4.z;
            row_4[0 + 3] = _uv4_4.w;
        }
        held_0[16] = row_4[0];
        held_0[17] = row_4[1];
        held_0[18] = row_4[2];
        held_0[19] = row_4[3];
    }
    held_0[20] = 0;
    held_0[21] = 0;
    held_0[22] = 0;
    held_0[23] = 0;
    if ((live >> 1 & 1) != 0) {
        unsigned int row_5[4];
        {
            uint4 _uv4_5 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 1) * 3584 + vec0_1 * 4) + 0);
            row_5[0 + 0] = _uv4_5.x;
            row_5[0 + 1] = _uv4_5.y;
            row_5[0 + 2] = _uv4_5.z;
            row_5[0 + 3] = _uv4_5.w;
        }
        held_0[20] = row_5[0];
        held_0[21] = row_5[1];
        held_0[22] = row_5[2];
        held_0[23] = row_5[3];
    }
    held_0[24] = 0;
    held_0[25] = 0;
    held_0[26] = 0;
    held_0[27] = 0;
    if ((live >> 2 & 1) != 0) {
        unsigned int row_6[4];
        {
            uint4 _uv4_6 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 2) * 3584 + vec0_1 * 4) + 0);
            row_6[0 + 0] = _uv4_6.x;
            row_6[0 + 1] = _uv4_6.y;
            row_6[0 + 2] = _uv4_6.z;
            row_6[0 + 3] = _uv4_6.w;
        }
        held_0[24] = row_6[0];
        held_0[25] = row_6[1];
        held_0[26] = row_6[2];
        held_0[27] = row_6[3];
    }
    held_0[28] = 0;
    held_0[29] = 0;
    held_0[30] = 0;
    held_0[31] = 0;
    if ((live >> 3 & 1) != 0) {
        unsigned int row_7[4];
        {
            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 3) * 3584 + vec0_1 * 4) + 0);
            row_7[0 + 0] = _uv4_7.x;
            row_7[0 + 1] = _uv4_7.y;
            row_7[0 + 2] = _uv4_7.z;
            row_7[0 + 3] = _uv4_7.w;
        }
        held_0[28] = row_7[0];
        held_0[29] = row_7[1];
        held_0[30] = row_7[2];
        held_0[31] = row_7[3];
    }
    int vec0_2 = 256 + tid;
    held_0[32] = 0;
    held_0[33] = 0;
    held_0[34] = 0;
    held_0[35] = 0;
    if ((live & 1) != 0) {
        unsigned int row_8[4];
        {
            uint4 _uv4_8 = *reinterpret_cast<const uint4*>(slot_rows + (t * 8 * 3584 + vec0_2 * 4) + 0);
            row_8[0 + 0] = _uv4_8.x;
            row_8[0 + 1] = _uv4_8.y;
            row_8[0 + 2] = _uv4_8.z;
            row_8[0 + 3] = _uv4_8.w;
        }
        held_0[32] = row_8[0];
        held_0[33] = row_8[1];
        held_0[34] = row_8[2];
        held_0[35] = row_8[3];
    }
    held_0[36] = 0;
    held_0[37] = 0;
    held_0[38] = 0;
    held_0[39] = 0;
    if ((live >> 1 & 1) != 0) {
        unsigned int row_9[4];
        {
            uint4 _uv4_9 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 1) * 3584 + vec0_2 * 4) + 0);
            row_9[0 + 0] = _uv4_9.x;
            row_9[0 + 1] = _uv4_9.y;
            row_9[0 + 2] = _uv4_9.z;
            row_9[0 + 3] = _uv4_9.w;
        }
        held_0[36] = row_9[0];
        held_0[37] = row_9[1];
        held_0[38] = row_9[2];
        held_0[39] = row_9[3];
    }
    held_0[40] = 0;
    held_0[41] = 0;
    held_0[42] = 0;
    held_0[43] = 0;
    if ((live >> 2 & 1) != 0) {
        unsigned int row_10[4];
        {
            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 2) * 3584 + vec0_2 * 4) + 0);
            row_10[0 + 0] = _uv4_10.x;
            row_10[0 + 1] = _uv4_10.y;
            row_10[0 + 2] = _uv4_10.z;
            row_10[0 + 3] = _uv4_10.w;
        }
        held_0[40] = row_10[0];
        held_0[41] = row_10[1];
        held_0[42] = row_10[2];
        held_0[43] = row_10[3];
    }
    held_0[44] = 0;
    held_0[45] = 0;
    held_0[46] = 0;
    held_0[47] = 0;
    if ((live >> 3 & 1) != 0) {
        unsigned int row_11[4];
        {
            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 3) * 3584 + vec0_2 * 4) + 0);
            row_11[0 + 0] = _uv4_11.x;
            row_11[0 + 1] = _uv4_11.y;
            row_11[0 + 2] = _uv4_11.z;
            row_11[0 + 3] = _uv4_11.w;
        }
        held_0[44] = row_11[0];
        held_0[45] = row_11[1];
        held_0[46] = row_11[2];
        held_0[47] = row_11[3];
    }
    int vec0_3 = 384 + tid;
    held_0[48] = 0;
    held_0[49] = 0;
    held_0[50] = 0;
    held_0[51] = 0;
    if ((live & 1) != 0) {
        unsigned int row_12[4];
        {
            uint4 _uv4_12 = *reinterpret_cast<const uint4*>(slot_rows + (t * 8 * 3584 + vec0_3 * 4) + 0);
            row_12[0 + 0] = _uv4_12.x;
            row_12[0 + 1] = _uv4_12.y;
            row_12[0 + 2] = _uv4_12.z;
            row_12[0 + 3] = _uv4_12.w;
        }
        held_0[48] = row_12[0];
        held_0[49] = row_12[1];
        held_0[50] = row_12[2];
        held_0[51] = row_12[3];
    }
    held_0[52] = 0;
    held_0[53] = 0;
    held_0[54] = 0;
    held_0[55] = 0;
    if ((live >> 1 & 1) != 0) {
        unsigned int row_13[4];
        {
            uint4 _uv4_13 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 1) * 3584 + vec0_3 * 4) + 0);
            row_13[0 + 0] = _uv4_13.x;
            row_13[0 + 1] = _uv4_13.y;
            row_13[0 + 2] = _uv4_13.z;
            row_13[0 + 3] = _uv4_13.w;
        }
        held_0[52] = row_13[0];
        held_0[53] = row_13[1];
        held_0[54] = row_13[2];
        held_0[55] = row_13[3];
    }
    held_0[56] = 0;
    held_0[57] = 0;
    held_0[58] = 0;
    held_0[59] = 0;
    if ((live >> 2 & 1) != 0) {
        unsigned int row_14[4];
        {
            uint4 _uv4_14 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 2) * 3584 + vec0_3 * 4) + 0);
            row_14[0 + 0] = _uv4_14.x;
            row_14[0 + 1] = _uv4_14.y;
            row_14[0 + 2] = _uv4_14.z;
            row_14[0 + 3] = _uv4_14.w;
        }
        held_0[56] = row_14[0];
        held_0[57] = row_14[1];
        held_0[58] = row_14[2];
        held_0[59] = row_14[3];
    }
    held_0[60] = 0;
    held_0[61] = 0;
    held_0[62] = 0;
    held_0[63] = 0;
    if ((live >> 3 & 1) != 0) {
        unsigned int row_15[4];
        {
            uint4 _uv4_15 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 3) * 3584 + vec0_3 * 4) + 0);
            row_15[0 + 0] = _uv4_15.x;
            row_15[0 + 1] = _uv4_15.y;
            row_15[0 + 2] = _uv4_15.z;
            row_15[0 + 3] = _uv4_15.w;
        }
        held_0[60] = row_15[0];
        held_0[61] = row_15[1];
        held_0[62] = row_15[2];
        held_0[63] = row_15[3];
    }
    int vec0_4 = 512 + tid;
    held_0[64] = 0;
    held_0[65] = 0;
    held_0[66] = 0;
    held_0[67] = 0;
    if ((live & 1) != 0) {
        unsigned int row_16[4];
        {
            uint4 _uv4_16 = *reinterpret_cast<const uint4*>(slot_rows + (t * 8 * 3584 + vec0_4 * 4) + 0);
            row_16[0 + 0] = _uv4_16.x;
            row_16[0 + 1] = _uv4_16.y;
            row_16[0 + 2] = _uv4_16.z;
            row_16[0 + 3] = _uv4_16.w;
        }
        held_0[64] = row_16[0];
        held_0[65] = row_16[1];
        held_0[66] = row_16[2];
        held_0[67] = row_16[3];
    }
    held_0[68] = 0;
    held_0[69] = 0;
    held_0[70] = 0;
    held_0[71] = 0;
    if ((live >> 1 & 1) != 0) {
        unsigned int row_17[4];
        {
            uint4 _uv4_17 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 1) * 3584 + vec0_4 * 4) + 0);
            row_17[0 + 0] = _uv4_17.x;
            row_17[0 + 1] = _uv4_17.y;
            row_17[0 + 2] = _uv4_17.z;
            row_17[0 + 3] = _uv4_17.w;
        }
        held_0[68] = row_17[0];
        held_0[69] = row_17[1];
        held_0[70] = row_17[2];
        held_0[71] = row_17[3];
    }
    held_0[72] = 0;
    held_0[73] = 0;
    held_0[74] = 0;
    held_0[75] = 0;
    if ((live >> 2 & 1) != 0) {
        unsigned int row_18[4];
        {
            uint4 _uv4_18 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 2) * 3584 + vec0_4 * 4) + 0);
            row_18[0 + 0] = _uv4_18.x;
            row_18[0 + 1] = _uv4_18.y;
            row_18[0 + 2] = _uv4_18.z;
            row_18[0 + 3] = _uv4_18.w;
        }
        held_0[72] = row_18[0];
        held_0[73] = row_18[1];
        held_0[74] = row_18[2];
        held_0[75] = row_18[3];
    }
    held_0[76] = 0;
    held_0[77] = 0;
    held_0[78] = 0;
    held_0[79] = 0;
    if ((live >> 3 & 1) != 0) {
        unsigned int row_19[4];
        {
            uint4 _uv4_19 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 3) * 3584 + vec0_4 * 4) + 0);
            row_19[0 + 0] = _uv4_19.x;
            row_19[0 + 1] = _uv4_19.y;
            row_19[0 + 2] = _uv4_19.z;
            row_19[0 + 3] = _uv4_19.w;
        }
        held_0[76] = row_19[0];
        held_0[77] = row_19[1];
        held_0[78] = row_19[2];
        held_0[79] = row_19[3];
    }
    int vec0_5 = 640 + tid;
    held_0[80] = 0;
    held_0[81] = 0;
    held_0[82] = 0;
    held_0[83] = 0;
    if ((live & 1) != 0) {
        unsigned int row_20[4];
        {
            uint4 _uv4_20 = *reinterpret_cast<const uint4*>(slot_rows + (t * 8 * 3584 + vec0_5 * 4) + 0);
            row_20[0 + 0] = _uv4_20.x;
            row_20[0 + 1] = _uv4_20.y;
            row_20[0 + 2] = _uv4_20.z;
            row_20[0 + 3] = _uv4_20.w;
        }
        held_0[80] = row_20[0];
        held_0[81] = row_20[1];
        held_0[82] = row_20[2];
        held_0[83] = row_20[3];
    }
    held_0[84] = 0;
    held_0[85] = 0;
    held_0[86] = 0;
    held_0[87] = 0;
    if ((live >> 1 & 1) != 0) {
        unsigned int row_21[4];
        {
            uint4 _uv4_21 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 1) * 3584 + vec0_5 * 4) + 0);
            row_21[0 + 0] = _uv4_21.x;
            row_21[0 + 1] = _uv4_21.y;
            row_21[0 + 2] = _uv4_21.z;
            row_21[0 + 3] = _uv4_21.w;
        }
        held_0[84] = row_21[0];
        held_0[85] = row_21[1];
        held_0[86] = row_21[2];
        held_0[87] = row_21[3];
    }
    held_0[88] = 0;
    held_0[89] = 0;
    held_0[90] = 0;
    held_0[91] = 0;
    if ((live >> 2 & 1) != 0) {
        unsigned int row_22[4];
        {
            uint4 _uv4_22 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 2) * 3584 + vec0_5 * 4) + 0);
            row_22[0 + 0] = _uv4_22.x;
            row_22[0 + 1] = _uv4_22.y;
            row_22[0 + 2] = _uv4_22.z;
            row_22[0 + 3] = _uv4_22.w;
        }
        held_0[88] = row_22[0];
        held_0[89] = row_22[1];
        held_0[90] = row_22[2];
        held_0[91] = row_22[3];
    }
    held_0[92] = 0;
    held_0[93] = 0;
    held_0[94] = 0;
    held_0[95] = 0;
    if ((live >> 3 & 1) != 0) {
        unsigned int row_23[4];
        {
            uint4 _uv4_23 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 3) * 3584 + vec0_5 * 4) + 0);
            row_23[0 + 0] = _uv4_23.x;
            row_23[0 + 1] = _uv4_23.y;
            row_23[0 + 2] = _uv4_23.z;
            row_23[0 + 3] = _uv4_23.w;
        }
        held_0[92] = row_23[0];
        held_0[93] = row_23[1];
        held_0[94] = row_23[2];
        held_0[95] = row_23[3];
    }
    int vec0_6 = 768 + tid;
    held_0[96] = 0;
    held_0[97] = 0;
    held_0[98] = 0;
    held_0[99] = 0;
    if ((live & 1) != 0) {
        unsigned int row_24[4];
        {
            uint4 _uv4_24 = *reinterpret_cast<const uint4*>(slot_rows + (t * 8 * 3584 + vec0_6 * 4) + 0);
            row_24[0 + 0] = _uv4_24.x;
            row_24[0 + 1] = _uv4_24.y;
            row_24[0 + 2] = _uv4_24.z;
            row_24[0 + 3] = _uv4_24.w;
        }
        held_0[96] = row_24[0];
        held_0[97] = row_24[1];
        held_0[98] = row_24[2];
        held_0[99] = row_24[3];
    }
    held_0[100] = 0;
    held_0[101] = 0;
    held_0[102] = 0;
    held_0[103] = 0;
    if ((live >> 1 & 1) != 0) {
        unsigned int row_25[4];
        {
            uint4 _uv4_25 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 1) * 3584 + vec0_6 * 4) + 0);
            row_25[0 + 0] = _uv4_25.x;
            row_25[0 + 1] = _uv4_25.y;
            row_25[0 + 2] = _uv4_25.z;
            row_25[0 + 3] = _uv4_25.w;
        }
        held_0[100] = row_25[0];
        held_0[101] = row_25[1];
        held_0[102] = row_25[2];
        held_0[103] = row_25[3];
    }
    held_0[104] = 0;
    held_0[105] = 0;
    held_0[106] = 0;
    held_0[107] = 0;
    if ((live >> 2 & 1) != 0) {
        unsigned int row_26[4];
        {
            uint4 _uv4_26 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 2) * 3584 + vec0_6 * 4) + 0);
            row_26[0 + 0] = _uv4_26.x;
            row_26[0 + 1] = _uv4_26.y;
            row_26[0 + 2] = _uv4_26.z;
            row_26[0 + 3] = _uv4_26.w;
        }
        held_0[104] = row_26[0];
        held_0[105] = row_26[1];
        held_0[106] = row_26[2];
        held_0[107] = row_26[3];
    }
    held_0[108] = 0;
    held_0[109] = 0;
    held_0[110] = 0;
    held_0[111] = 0;
    if ((live >> 3 & 1) != 0) {
        unsigned int row_27[4];
        {
            uint4 _uv4_27 = *reinterpret_cast<const uint4*>(slot_rows + ((t * 8 + 3) * 3584 + vec0_6 * 4) + 0);
            row_27[0 + 0] = _uv4_27.x;
            row_27[0 + 1] = _uv4_27.y;
            row_27[0 + 2] = _uv4_27.z;
            row_27[0 + 3] = _uv4_27.w;
        }
        held_0[108] = row_27[0];
        held_0[109] = row_27[1];
        held_0[110] = row_27[2];
        held_0[111] = row_27[3];
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
    if ((live & 1) != 0) {
        unsigned int words_0[4];
        words_0[0] = held_0[0];
        words_0[1] = held_0[1];
        words_0[2] = held_0[2];
        words_0[3] = held_0[3];
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
    }
    if ((live >> 1 & 1) != 0) {
        unsigned int words_1[4];
        words_1[0] = held_0[4];
        words_1[1] = held_0[5];
        words_1[2] = held_0[6];
        words_1[3] = held_0[7];
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
    }
    if ((live >> 2 & 1) != 0) {
        unsigned int words_2[4];
        words_2[0] = held_0[8];
        words_2[1] = held_0[9];
        words_2[2] = held_0[10];
        words_2[3] = held_0[11];
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
    }
    if ((live >> 3 & 1) != 0) {
        unsigned int words_3[4];
        words_3[0] = held_0[12];
        words_3[1] = held_0[13];
        words_3[2] = held_0[14];
        words_3[3] = held_0[15];
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
    }
    unsigned int packed_0[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_0[_lp*2 + 0], acc_0[_lp*2+1 + 0]));
        packed_0[_lp] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(out + (t * 3584 + vec_0 * 4))[0] = reinterpret_cast<int4*>(packed_0)[0];
    int vec_1 = 128 + tid;
    float acc_1[8];
    acc_1[0] = 0.0f;
    acc_1[1] = 0.0f;
    acc_1[2] = 0.0f;
    acc_1[3] = 0.0f;
    acc_1[4] = 0.0f;
    acc_1[5] = 0.0f;
    acc_1[6] = 0.0f;
    acc_1[7] = 0.0f;
    if ((live & 1) != 0) {
        unsigned int words_4[4];
        words_4[0] = held_0[16];
        words_4[1] = held_0[17];
        words_4[2] = held_0[18];
        words_4[3] = held_0[19];
        float vals_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_4[_pair]) << 16);
            (&vals_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_4[_pair]) & 0xffff0000u);
        }
        acc_1[0] = acc_1[0] + vals_4[0];
        acc_1[1] = acc_1[1] + vals_4[1];
        acc_1[2] = acc_1[2] + vals_4[2];
        acc_1[3] = acc_1[3] + vals_4[3];
        acc_1[4] = acc_1[4] + vals_4[4];
        acc_1[5] = acc_1[5] + vals_4[5];
        acc_1[6] = acc_1[6] + vals_4[6];
        acc_1[7] = acc_1[7] + vals_4[7];
    }
    if ((live >> 1 & 1) != 0) {
        unsigned int words_5[4];
        words_5[0] = held_0[20];
        words_5[1] = held_0[21];
        words_5[2] = held_0[22];
        words_5[3] = held_0[23];
        float vals_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_5[_pair]) << 16);
            (&vals_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_5[_pair]) & 0xffff0000u);
        }
        acc_1[0] = acc_1[0] + vals_5[0];
        acc_1[1] = acc_1[1] + vals_5[1];
        acc_1[2] = acc_1[2] + vals_5[2];
        acc_1[3] = acc_1[3] + vals_5[3];
        acc_1[4] = acc_1[4] + vals_5[4];
        acc_1[5] = acc_1[5] + vals_5[5];
        acc_1[6] = acc_1[6] + vals_5[6];
        acc_1[7] = acc_1[7] + vals_5[7];
    }
    if ((live >> 2 & 1) != 0) {
        unsigned int words_6[4];
        words_6[0] = held_0[24];
        words_6[1] = held_0[25];
        words_6[2] = held_0[26];
        words_6[3] = held_0[27];
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
    }
    if ((live >> 3 & 1) != 0) {
        unsigned int words_7[4];
        words_7[0] = held_0[28];
        words_7[1] = held_0[29];
        words_7[2] = held_0[30];
        words_7[3] = held_0[31];
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
    }
    unsigned int packed_1[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_1[_lp*2 + 0], acc_1[_lp*2+1 + 0]));
        packed_1[_lp] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(out + (t * 3584 + vec_1 * 4))[0] = reinterpret_cast<int4*>(packed_1)[0];
    int vec_2 = 256 + tid;
    float acc_2[8];
    acc_2[0] = 0.0f;
    acc_2[1] = 0.0f;
    acc_2[2] = 0.0f;
    acc_2[3] = 0.0f;
    acc_2[4] = 0.0f;
    acc_2[5] = 0.0f;
    acc_2[6] = 0.0f;
    acc_2[7] = 0.0f;
    if ((live & 1) != 0) {
        unsigned int words_8[4];
        words_8[0] = held_0[32];
        words_8[1] = held_0[33];
        words_8[2] = held_0[34];
        words_8[3] = held_0[35];
        float vals_8[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_8[_pair]) << 16);
            (&vals_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_8[_pair]) & 0xffff0000u);
        }
        acc_2[0] = acc_2[0] + vals_8[0];
        acc_2[1] = acc_2[1] + vals_8[1];
        acc_2[2] = acc_2[2] + vals_8[2];
        acc_2[3] = acc_2[3] + vals_8[3];
        acc_2[4] = acc_2[4] + vals_8[4];
        acc_2[5] = acc_2[5] + vals_8[5];
        acc_2[6] = acc_2[6] + vals_8[6];
        acc_2[7] = acc_2[7] + vals_8[7];
    }
    if ((live >> 1 & 1) != 0) {
        unsigned int words_9[4];
        words_9[0] = held_0[36];
        words_9[1] = held_0[37];
        words_9[2] = held_0[38];
        words_9[3] = held_0[39];
        float vals_9[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_9[_pair]) << 16);
            (&vals_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_9[_pair]) & 0xffff0000u);
        }
        acc_2[0] = acc_2[0] + vals_9[0];
        acc_2[1] = acc_2[1] + vals_9[1];
        acc_2[2] = acc_2[2] + vals_9[2];
        acc_2[3] = acc_2[3] + vals_9[3];
        acc_2[4] = acc_2[4] + vals_9[4];
        acc_2[5] = acc_2[5] + vals_9[5];
        acc_2[6] = acc_2[6] + vals_9[6];
        acc_2[7] = acc_2[7] + vals_9[7];
    }
    if ((live >> 2 & 1) != 0) {
        unsigned int words_10[4];
        words_10[0] = held_0[40];
        words_10[1] = held_0[41];
        words_10[2] = held_0[42];
        words_10[3] = held_0[43];
        float vals_10[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_10[_pair]) << 16);
            (&vals_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_10[_pair]) & 0xffff0000u);
        }
        acc_2[0] = acc_2[0] + vals_10[0];
        acc_2[1] = acc_2[1] + vals_10[1];
        acc_2[2] = acc_2[2] + vals_10[2];
        acc_2[3] = acc_2[3] + vals_10[3];
        acc_2[4] = acc_2[4] + vals_10[4];
        acc_2[5] = acc_2[5] + vals_10[5];
        acc_2[6] = acc_2[6] + vals_10[6];
        acc_2[7] = acc_2[7] + vals_10[7];
    }
    if ((live >> 3 & 1) != 0) {
        unsigned int words_11[4];
        words_11[0] = held_0[44];
        words_11[1] = held_0[45];
        words_11[2] = held_0[46];
        words_11[3] = held_0[47];
        float vals_11[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_11[_pair]) << 16);
            (&vals_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_11[_pair]) & 0xffff0000u);
        }
        acc_2[0] = acc_2[0] + vals_11[0];
        acc_2[1] = acc_2[1] + vals_11[1];
        acc_2[2] = acc_2[2] + vals_11[2];
        acc_2[3] = acc_2[3] + vals_11[3];
        acc_2[4] = acc_2[4] + vals_11[4];
        acc_2[5] = acc_2[5] + vals_11[5];
        acc_2[6] = acc_2[6] + vals_11[6];
        acc_2[7] = acc_2[7] + vals_11[7];
    }
    unsigned int packed_2[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_2[_lp*2 + 0], acc_2[_lp*2+1 + 0]));
        packed_2[_lp] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(out + (t * 3584 + vec_2 * 4))[0] = reinterpret_cast<int4*>(packed_2)[0];
    int vec_3 = 384 + tid;
    float acc_3[8];
    acc_3[0] = 0.0f;
    acc_3[1] = 0.0f;
    acc_3[2] = 0.0f;
    acc_3[3] = 0.0f;
    acc_3[4] = 0.0f;
    acc_3[5] = 0.0f;
    acc_3[6] = 0.0f;
    acc_3[7] = 0.0f;
    if ((live & 1) != 0) {
        unsigned int words_12[4];
        words_12[0] = held_0[48];
        words_12[1] = held_0[49];
        words_12[2] = held_0[50];
        words_12[3] = held_0[51];
        float vals_12[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_12[_pair]) << 16);
            (&vals_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_12[_pair]) & 0xffff0000u);
        }
        acc_3[0] = acc_3[0] + vals_12[0];
        acc_3[1] = acc_3[1] + vals_12[1];
        acc_3[2] = acc_3[2] + vals_12[2];
        acc_3[3] = acc_3[3] + vals_12[3];
        acc_3[4] = acc_3[4] + vals_12[4];
        acc_3[5] = acc_3[5] + vals_12[5];
        acc_3[6] = acc_3[6] + vals_12[6];
        acc_3[7] = acc_3[7] + vals_12[7];
    }
    if ((live >> 1 & 1) != 0) {
        unsigned int words_13[4];
        words_13[0] = held_0[52];
        words_13[1] = held_0[53];
        words_13[2] = held_0[54];
        words_13[3] = held_0[55];
        float vals_13[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_13[_pair]) << 16);
            (&vals_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_13[_pair]) & 0xffff0000u);
        }
        acc_3[0] = acc_3[0] + vals_13[0];
        acc_3[1] = acc_3[1] + vals_13[1];
        acc_3[2] = acc_3[2] + vals_13[2];
        acc_3[3] = acc_3[3] + vals_13[3];
        acc_3[4] = acc_3[4] + vals_13[4];
        acc_3[5] = acc_3[5] + vals_13[5];
        acc_3[6] = acc_3[6] + vals_13[6];
        acc_3[7] = acc_3[7] + vals_13[7];
    }
    if ((live >> 2 & 1) != 0) {
        unsigned int words_14[4];
        words_14[0] = held_0[56];
        words_14[1] = held_0[57];
        words_14[2] = held_0[58];
        words_14[3] = held_0[59];
        float vals_14[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_14[_pair]) << 16);
            (&vals_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_14[_pair]) & 0xffff0000u);
        }
        acc_3[0] = acc_3[0] + vals_14[0];
        acc_3[1] = acc_3[1] + vals_14[1];
        acc_3[2] = acc_3[2] + vals_14[2];
        acc_3[3] = acc_3[3] + vals_14[3];
        acc_3[4] = acc_3[4] + vals_14[4];
        acc_3[5] = acc_3[5] + vals_14[5];
        acc_3[6] = acc_3[6] + vals_14[6];
        acc_3[7] = acc_3[7] + vals_14[7];
    }
    if ((live >> 3 & 1) != 0) {
        unsigned int words_15[4];
        words_15[0] = held_0[60];
        words_15[1] = held_0[61];
        words_15[2] = held_0[62];
        words_15[3] = held_0[63];
        float vals_15[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_15[_pair]) << 16);
            (&vals_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_15[_pair]) & 0xffff0000u);
        }
        acc_3[0] = acc_3[0] + vals_15[0];
        acc_3[1] = acc_3[1] + vals_15[1];
        acc_3[2] = acc_3[2] + vals_15[2];
        acc_3[3] = acc_3[3] + vals_15[3];
        acc_3[4] = acc_3[4] + vals_15[4];
        acc_3[5] = acc_3[5] + vals_15[5];
        acc_3[6] = acc_3[6] + vals_15[6];
        acc_3[7] = acc_3[7] + vals_15[7];
    }
    unsigned int packed_3[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_3[_lp*2 + 0], acc_3[_lp*2+1 + 0]));
        packed_3[_lp] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(out + (t * 3584 + vec_3 * 4))[0] = reinterpret_cast<int4*>(packed_3)[0];
    int vec_4 = 512 + tid;
    float acc_4[8];
    acc_4[0] = 0.0f;
    acc_4[1] = 0.0f;
    acc_4[2] = 0.0f;
    acc_4[3] = 0.0f;
    acc_4[4] = 0.0f;
    acc_4[5] = 0.0f;
    acc_4[6] = 0.0f;
    acc_4[7] = 0.0f;
    if ((live & 1) != 0) {
        unsigned int words_16[4];
        words_16[0] = held_0[64];
        words_16[1] = held_0[65];
        words_16[2] = held_0[66];
        words_16[3] = held_0[67];
        float vals_16[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_16[_pair]) << 16);
            (&vals_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_16[_pair]) & 0xffff0000u);
        }
        acc_4[0] = acc_4[0] + vals_16[0];
        acc_4[1] = acc_4[1] + vals_16[1];
        acc_4[2] = acc_4[2] + vals_16[2];
        acc_4[3] = acc_4[3] + vals_16[3];
        acc_4[4] = acc_4[4] + vals_16[4];
        acc_4[5] = acc_4[5] + vals_16[5];
        acc_4[6] = acc_4[6] + vals_16[6];
        acc_4[7] = acc_4[7] + vals_16[7];
    }
    if ((live >> 1 & 1) != 0) {
        unsigned int words_17[4];
        words_17[0] = held_0[68];
        words_17[1] = held_0[69];
        words_17[2] = held_0[70];
        words_17[3] = held_0[71];
        float vals_17[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_17[_pair]) << 16);
            (&vals_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_17[_pair]) & 0xffff0000u);
        }
        acc_4[0] = acc_4[0] + vals_17[0];
        acc_4[1] = acc_4[1] + vals_17[1];
        acc_4[2] = acc_4[2] + vals_17[2];
        acc_4[3] = acc_4[3] + vals_17[3];
        acc_4[4] = acc_4[4] + vals_17[4];
        acc_4[5] = acc_4[5] + vals_17[5];
        acc_4[6] = acc_4[6] + vals_17[6];
        acc_4[7] = acc_4[7] + vals_17[7];
    }
    if ((live >> 2 & 1) != 0) {
        unsigned int words_18[4];
        words_18[0] = held_0[72];
        words_18[1] = held_0[73];
        words_18[2] = held_0[74];
        words_18[3] = held_0[75];
        float vals_18[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_18[_pair]) << 16);
            (&vals_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_18[_pair]) & 0xffff0000u);
        }
        acc_4[0] = acc_4[0] + vals_18[0];
        acc_4[1] = acc_4[1] + vals_18[1];
        acc_4[2] = acc_4[2] + vals_18[2];
        acc_4[3] = acc_4[3] + vals_18[3];
        acc_4[4] = acc_4[4] + vals_18[4];
        acc_4[5] = acc_4[5] + vals_18[5];
        acc_4[6] = acc_4[6] + vals_18[6];
        acc_4[7] = acc_4[7] + vals_18[7];
    }
    if ((live >> 3 & 1) != 0) {
        unsigned int words_19[4];
        words_19[0] = held_0[76];
        words_19[1] = held_0[77];
        words_19[2] = held_0[78];
        words_19[3] = held_0[79];
        float vals_19[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_19[_pair]) << 16);
            (&vals_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_19[_pair]) & 0xffff0000u);
        }
        acc_4[0] = acc_4[0] + vals_19[0];
        acc_4[1] = acc_4[1] + vals_19[1];
        acc_4[2] = acc_4[2] + vals_19[2];
        acc_4[3] = acc_4[3] + vals_19[3];
        acc_4[4] = acc_4[4] + vals_19[4];
        acc_4[5] = acc_4[5] + vals_19[5];
        acc_4[6] = acc_4[6] + vals_19[6];
        acc_4[7] = acc_4[7] + vals_19[7];
    }
    unsigned int packed_4[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_4[_lp*2 + 0], acc_4[_lp*2+1 + 0]));
        packed_4[_lp] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(out + (t * 3584 + vec_4 * 4))[0] = reinterpret_cast<int4*>(packed_4)[0];
    int vec_5 = 640 + tid;
    float acc_5[8];
    acc_5[0] = 0.0f;
    acc_5[1] = 0.0f;
    acc_5[2] = 0.0f;
    acc_5[3] = 0.0f;
    acc_5[4] = 0.0f;
    acc_5[5] = 0.0f;
    acc_5[6] = 0.0f;
    acc_5[7] = 0.0f;
    if ((live & 1) != 0) {
        unsigned int words_20[4];
        words_20[0] = held_0[80];
        words_20[1] = held_0[81];
        words_20[2] = held_0[82];
        words_20[3] = held_0[83];
        float vals_20[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_20[_pair]) << 16);
            (&vals_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_20[_pair]) & 0xffff0000u);
        }
        acc_5[0] = acc_5[0] + vals_20[0];
        acc_5[1] = acc_5[1] + vals_20[1];
        acc_5[2] = acc_5[2] + vals_20[2];
        acc_5[3] = acc_5[3] + vals_20[3];
        acc_5[4] = acc_5[4] + vals_20[4];
        acc_5[5] = acc_5[5] + vals_20[5];
        acc_5[6] = acc_5[6] + vals_20[6];
        acc_5[7] = acc_5[7] + vals_20[7];
    }
    if ((live >> 1 & 1) != 0) {
        unsigned int words_21[4];
        words_21[0] = held_0[84];
        words_21[1] = held_0[85];
        words_21[2] = held_0[86];
        words_21[3] = held_0[87];
        float vals_21[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_21[_pair]) << 16);
            (&vals_21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_21[_pair]) & 0xffff0000u);
        }
        acc_5[0] = acc_5[0] + vals_21[0];
        acc_5[1] = acc_5[1] + vals_21[1];
        acc_5[2] = acc_5[2] + vals_21[2];
        acc_5[3] = acc_5[3] + vals_21[3];
        acc_5[4] = acc_5[4] + vals_21[4];
        acc_5[5] = acc_5[5] + vals_21[5];
        acc_5[6] = acc_5[6] + vals_21[6];
        acc_5[7] = acc_5[7] + vals_21[7];
    }
    if ((live >> 2 & 1) != 0) {
        unsigned int words_22[4];
        words_22[0] = held_0[88];
        words_22[1] = held_0[89];
        words_22[2] = held_0[90];
        words_22[3] = held_0[91];
        float vals_22[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_22[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_22[_pair]) << 16);
            (&vals_22[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_22[_pair]) & 0xffff0000u);
        }
        acc_5[0] = acc_5[0] + vals_22[0];
        acc_5[1] = acc_5[1] + vals_22[1];
        acc_5[2] = acc_5[2] + vals_22[2];
        acc_5[3] = acc_5[3] + vals_22[3];
        acc_5[4] = acc_5[4] + vals_22[4];
        acc_5[5] = acc_5[5] + vals_22[5];
        acc_5[6] = acc_5[6] + vals_22[6];
        acc_5[7] = acc_5[7] + vals_22[7];
    }
    if ((live >> 3 & 1) != 0) {
        unsigned int words_23[4];
        words_23[0] = held_0[92];
        words_23[1] = held_0[93];
        words_23[2] = held_0[94];
        words_23[3] = held_0[95];
        float vals_23[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_23[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_23[_pair]) << 16);
            (&vals_23[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_23[_pair]) & 0xffff0000u);
        }
        acc_5[0] = acc_5[0] + vals_23[0];
        acc_5[1] = acc_5[1] + vals_23[1];
        acc_5[2] = acc_5[2] + vals_23[2];
        acc_5[3] = acc_5[3] + vals_23[3];
        acc_5[4] = acc_5[4] + vals_23[4];
        acc_5[5] = acc_5[5] + vals_23[5];
        acc_5[6] = acc_5[6] + vals_23[6];
        acc_5[7] = acc_5[7] + vals_23[7];
    }
    unsigned int packed_5[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_5[_lp*2 + 0], acc_5[_lp*2+1 + 0]));
        packed_5[_lp] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(out + (t * 3584 + vec_5 * 4))[0] = reinterpret_cast<int4*>(packed_5)[0];
    int vec_6 = 768 + tid;
    float acc_6[8];
    acc_6[0] = 0.0f;
    acc_6[1] = 0.0f;
    acc_6[2] = 0.0f;
    acc_6[3] = 0.0f;
    acc_6[4] = 0.0f;
    acc_6[5] = 0.0f;
    acc_6[6] = 0.0f;
    acc_6[7] = 0.0f;
    if ((live & 1) != 0) {
        unsigned int words_24[4];
        words_24[0] = held_0[96];
        words_24[1] = held_0[97];
        words_24[2] = held_0[98];
        words_24[3] = held_0[99];
        float vals_24[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_24[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_24[_pair]) << 16);
            (&vals_24[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_24[_pair]) & 0xffff0000u);
        }
        acc_6[0] = acc_6[0] + vals_24[0];
        acc_6[1] = acc_6[1] + vals_24[1];
        acc_6[2] = acc_6[2] + vals_24[2];
        acc_6[3] = acc_6[3] + vals_24[3];
        acc_6[4] = acc_6[4] + vals_24[4];
        acc_6[5] = acc_6[5] + vals_24[5];
        acc_6[6] = acc_6[6] + vals_24[6];
        acc_6[7] = acc_6[7] + vals_24[7];
    }
    if ((live >> 1 & 1) != 0) {
        unsigned int words_25[4];
        words_25[0] = held_0[100];
        words_25[1] = held_0[101];
        words_25[2] = held_0[102];
        words_25[3] = held_0[103];
        float vals_25[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_25[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_25[_pair]) << 16);
            (&vals_25[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_25[_pair]) & 0xffff0000u);
        }
        acc_6[0] = acc_6[0] + vals_25[0];
        acc_6[1] = acc_6[1] + vals_25[1];
        acc_6[2] = acc_6[2] + vals_25[2];
        acc_6[3] = acc_6[3] + vals_25[3];
        acc_6[4] = acc_6[4] + vals_25[4];
        acc_6[5] = acc_6[5] + vals_25[5];
        acc_6[6] = acc_6[6] + vals_25[6];
        acc_6[7] = acc_6[7] + vals_25[7];
    }
    if ((live >> 2 & 1) != 0) {
        unsigned int words_26[4];
        words_26[0] = held_0[104];
        words_26[1] = held_0[105];
        words_26[2] = held_0[106];
        words_26[3] = held_0[107];
        float vals_26[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_26[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_26[_pair]) << 16);
            (&vals_26[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_26[_pair]) & 0xffff0000u);
        }
        acc_6[0] = acc_6[0] + vals_26[0];
        acc_6[1] = acc_6[1] + vals_26[1];
        acc_6[2] = acc_6[2] + vals_26[2];
        acc_6[3] = acc_6[3] + vals_26[3];
        acc_6[4] = acc_6[4] + vals_26[4];
        acc_6[5] = acc_6[5] + vals_26[5];
        acc_6[6] = acc_6[6] + vals_26[6];
        acc_6[7] = acc_6[7] + vals_26[7];
    }
    if ((live >> 3 & 1) != 0) {
        unsigned int words_27[4];
        words_27[0] = held_0[108];
        words_27[1] = held_0[109];
        words_27[2] = held_0[110];
        words_27[3] = held_0[111];
        float vals_27[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&vals_27[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_27[_pair]) << 16);
            (&vals_27[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_27[_pair]) & 0xffff0000u);
        }
        acc_6[0] = acc_6[0] + vals_27[0];
        acc_6[1] = acc_6[1] + vals_27[1];
        acc_6[2] = acc_6[2] + vals_27[2];
        acc_6[3] = acc_6[3] + vals_27[3];
        acc_6[4] = acc_6[4] + vals_27[4];
        acc_6[5] = acc_6[5] + vals_27[5];
        acc_6[6] = acc_6[6] + vals_27[6];
        acc_6[7] = acc_6[7] + vals_27[7];
    }
    unsigned int packed_6[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_6[_lp*2 + 0], acc_6[_lp*2+1 + 0]));
        packed_6[_lp] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(out + (t * 3584 + vec_6 * 4))[0] = reinterpret_cast<int4*>(packed_6)[0];
}

} // extern "C"
