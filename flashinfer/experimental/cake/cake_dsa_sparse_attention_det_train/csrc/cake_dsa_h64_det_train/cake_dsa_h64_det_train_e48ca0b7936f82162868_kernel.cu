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

#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_det_train_e48ca0b7936f82162868(unsigned long long* __restrict__ cnt64, unsigned int* __restrict__ row_pairs, int* __restrict__ e_lat, int* __restrict__ e_rope, int* dkv_acc, int* dkr_acc, int* __restrict__ redo_mark, int* __restrict__ redo_rows, unsigned int* __restrict__ det_globals, int num_rows, int dkv_stride, int dkr_stride, int dkr_col0, int capacity, int min_bump, int bump_margin, float* dst_packed, int dst_row_stride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int row = blockIdx.x * 8 + warp;
    int lane_1 = lane;
    if (row < num_rows) {
        int word0 = e_lat[row];
        int e = word0 << 16 >> 16;
        int flagged = 0;
        int nonfinite = 0;
        unsigned long long worst = 0ull;
        unsigned long long worst_all = 0ull;
        int fold_bad = 0;
        int e_r = 0;
        int n[20];
        float prev[20];
        if (e != 127) {
            unsigned long long cq_0 = cnt64[(long long)row * 4ll];
            if (cq_0 >> 63ull != 0ull) {
                nonfinite = 1;
            }
            unsigned long long _max_0 = ((worst) > (cq_0 & 9223372036854775807ull) ? (worst) : (cq_0 & 9223372036854775807ull));
            worst = _max_0;
            unsigned long long _max_1 = ((worst_all) > (cq_0) ? (worst_all) : (cq_0));
            worst_all = _max_1;
            unsigned long long cq_1 = cnt64[(long long)row * 4ll + 1ll];
            if (cq_1 >> 63ull != 0ull) {
                nonfinite = 1;
            }
            unsigned long long _max_2 = ((worst) > (cq_1 & 9223372036854775807ull) ? (worst) : (cq_1 & 9223372036854775807ull));
            worst = _max_2;
            unsigned long long _max_3 = ((worst_all) > (cq_1) ? (worst_all) : (cq_1));
            worst_all = _max_3;
            unsigned long long cq_2 = cnt64[(long long)row * 4ll + 2ll];
            if (cq_2 >> 63ull != 0ull) {
                nonfinite = 1;
            }
            unsigned long long _max_4 = ((worst) > (cq_2 & 9223372036854775807ull) ? (worst) : (cq_2 & 9223372036854775807ull));
            worst = _max_4;
            unsigned long long _max_5 = ((worst_all) > (cq_2) ? (worst_all) : (cq_2));
            worst_all = _max_5;
            unsigned long long cq_3 = cnt64[(long long)row * 4ll + 3ll];
            if (cq_3 >> 63ull != 0ull) {
                nonfinite = 1;
            }
            unsigned long long _max_6 = ((worst) > (cq_3 & 9223372036854775807ull) ? (worst) : (cq_3 & 9223372036854775807ull));
            worst = _max_6;
            unsigned long long _max_7 = ((worst_all) > (cq_3) ? (worst_all) : (cq_3));
            worst_all = _max_7;
            unsigned int pairs = row_pairs[row];
            {
                const int4* _ivptr_8 = reinterpret_cast<const int4*>(dkv_acc + ((long long)row * (long long)dkv_stride + (long long)(lane_1 * 4)) + 0);
                int4 _ivld_8;
                _ivld_8 = *_ivptr_8;
                n[0 + 0] = _ivld_8.x;
                n[0 + 1] = _ivld_8.y;
                n[0 + 2] = _ivld_8.z;
                n[0 + 3] = _ivld_8.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)row * (long long)dst_row_stride + (long long)(lane_1 * 4)) + 0);
                prev[0 + 0] = _v4.x;
                prev[0 + 1] = _v4.y;
                prev[0 + 2] = _v4.z;
                prev[0 + 3] = _v4.w;
            }
            {
                const int4* _ivptr_10 = reinterpret_cast<const int4*>(dkv_acc + ((long long)row * (long long)dkv_stride + (long long)(128 + lane_1 * 4)) + 0);
                int4 _ivld_10;
                _ivld_10 = *_ivptr_10;
                n[4 + 0] = _ivld_10.x;
                n[4 + 1] = _ivld_10.y;
                n[4 + 2] = _ivld_10.z;
                n[4 + 3] = _ivld_10.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)row * (long long)dst_row_stride + (long long)(128 + lane_1 * 4)) + 0);
                prev[4 + 0] = _v4.x;
                prev[4 + 1] = _v4.y;
                prev[4 + 2] = _v4.z;
                prev[4 + 3] = _v4.w;
            }
            {
                const int4* _ivptr_12 = reinterpret_cast<const int4*>(dkv_acc + ((long long)row * (long long)dkv_stride + (long long)(256 + lane_1 * 4)) + 0);
                int4 _ivld_12;
                _ivld_12 = *_ivptr_12;
                n[8 + 0] = _ivld_12.x;
                n[8 + 1] = _ivld_12.y;
                n[8 + 2] = _ivld_12.z;
                n[8 + 3] = _ivld_12.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)row * (long long)dst_row_stride + (long long)(256 + lane_1 * 4)) + 0);
                prev[8 + 0] = _v4.x;
                prev[8 + 1] = _v4.y;
                prev[8 + 2] = _v4.z;
                prev[8 + 3] = _v4.w;
            }
            {
                const int4* _ivptr_14 = reinterpret_cast<const int4*>(dkv_acc + ((long long)row * (long long)dkv_stride + (long long)(384 + lane_1 * 4)) + 0);
                int4 _ivld_14;
                _ivld_14 = *_ivptr_14;
                n[12 + 0] = _ivld_14.x;
                n[12 + 1] = _ivld_14.y;
                n[12 + 2] = _ivld_14.z;
                n[12 + 3] = _ivld_14.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)row * (long long)dst_row_stride + (long long)(384 + lane_1 * 4)) + 0);
                prev[12 + 0] = _v4.x;
                prev[12 + 1] = _v4.y;
                prev[12 + 2] = _v4.z;
                prev[12 + 3] = _v4.w;
            }
            if (lane_1 < 16) {
                e_r = e_rope[row];
                {
                    const int4* _ivptr_16 = reinterpret_cast<const int4*>(dkr_acc + ((long long)row * (long long)dkr_stride + (long long)(dkr_col0 + lane_1 * 4)) + 0);
                    int4 _ivld_16;
                    _ivld_16 = *_ivptr_16;
                    n[16 + 0] = _ivld_16.x;
                    n[16 + 1] = _ivld_16.y;
                    n[16 + 2] = _ivld_16.z;
                    n[16 + 3] = _ivld_16.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)row * (long long)dst_row_stride + (long long)(512 + lane_1 * 4)) + 0);
                    prev[16 + 0] = _v4.x;
                    prev[16 + 1] = _v4.y;
                    prev[16 + 2] = _v4.z;
                    prev[16 + 3] = _v4.w;
                }
            }
            unsigned long long slack = (unsigned long long)pairs << (unsigned long long)(word0 >> 16);
            if (worst_all >= 2147483648ull || worst_all + slack >= 2147483648ull) {
                fold_bad = 1;
            }
            worst = worst + slack;
            if (nonfinite == 0 && worst >= 2147483648ull) {
                flagged = 1;
            }
        }
        __syncwarp();
        if (flagged != 0) {
            unsigned int hi = (unsigned int)(worst >> 32ull);
            unsigned int lo = (unsigned int)(worst & 4294967295ull);
            int _count_leading_zeros_18 = __clz(lo);
            int bits = 32 - _count_leading_zeros_18;
            if (hi != 0) {
                int _count_leading_zeros_19 = __clz(hi);
                bits = 64 - _count_leading_zeros_19;
            }
            int bump = ((bits - 30 + bump_margin) > (min_bump) ? (bits - 30 + bump_margin) : (min_bump));
            if (lane_1 == 0) {
                int _min_20 = ((e + bump) < (126) ? (e + bump) : (126));
                e_lat[row] = _min_20 & 65535 | word0 >> 16 << 16;
                int _min_21 = ((e_rope[row] + bump) < (126) ? (e_rope[row] + bump) : (126));
                e_rope[row] = _min_21;
                unsigned int slot = atomicAdd(&det_globals[8], 1);
                if ((int)slot < capacity) {
                    redo_rows[(int)slot] = row;
                }
            }
            int zero4[4];
            zero4[0] = 0;
            zero4[1] = 0;
            zero4[2] = 0;
            zero4[3] = 0;
            long long lbase = (long long)row * (long long)dkv_stride;
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase + (long long)(lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase + (long long)(128 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase + (long long)(256 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase + (long long)(384 + lane_1 * 4)) = _iv4;
            }
            if (lane_1 < 16) {
                long long rbase = (long long)row * (long long)dkr_stride + (long long)dkr_col0;
                {
                    int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                    *reinterpret_cast<int4*>(dkr_acc + rbase + (long long)(lane_1 * 4)) = _iv4;
                }
            }
            if (lane_1 < 4) {
                cnt64[(long long)row * 4ll + (long long)lane_1] = 0ull;
            }
        }
        if (flagged == 0 && e != 127) {
            float nan = __uint_as_float(2143289344);
            float s_lat = __uint_as_float((unsigned int)(e + 127) << 23);
            float v4[4];
            v4[0] = mul_f32_rn_noftz((float)n[0], s_lat);
            if (fold_bad != 0) {
                v4[0] = nan;
            }
            v4[1] = mul_f32_rn_noftz((float)n[1], s_lat);
            if (fold_bad != 0) {
                v4[1] = nan;
            }
            v4[2] = mul_f32_rn_noftz((float)n[2], s_lat);
            if (fold_bad != 0) {
                v4[2] = nan;
            }
            v4[3] = mul_f32_rn_noftz((float)n[3], s_lat);
            if (fold_bad != 0) {
                v4[3] = nan;
            }
            if (n[0] != 0 || fold_bad != 0) {
                float _add_22;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(prev[0]), "f"(v4[0]));
                prev[0] = _add_22;
            }
            if (n[1] != 0 || fold_bad != 0) {
                float _add_23;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(prev[1]), "f"(v4[1]));
                prev[1] = _add_23;
            }
            if (n[2] != 0 || fold_bad != 0) {
                float _add_24;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(prev[2]), "f"(v4[2]));
                prev[2] = _add_24;
            }
            if (n[3] != 0 || fold_bad != 0) {
                float _add_25;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(prev[3]), "f"(v4[3]));
                prev[3] = _add_25;
            }
            {
                float4 _v4 = make_float4(prev[0 + 0], prev[0 + 1], prev[0 + 2], prev[0 + 3]);
                *reinterpret_cast<float4*>(dst_packed + (long long)row * (long long)dst_row_stride + (long long)(lane_1 * 4)) = _v4;
            }
            float v4_0[4];
            v4_0[0] = mul_f32_rn_noftz((float)n[4], s_lat);
            if (fold_bad != 0) {
                v4_0[0] = nan;
            }
            v4_0[1] = mul_f32_rn_noftz((float)n[5], s_lat);
            if (fold_bad != 0) {
                v4_0[1] = nan;
            }
            v4_0[2] = mul_f32_rn_noftz((float)n[6], s_lat);
            if (fold_bad != 0) {
                v4_0[2] = nan;
            }
            v4_0[3] = mul_f32_rn_noftz((float)n[7], s_lat);
            if (fold_bad != 0) {
                v4_0[3] = nan;
            }
            if (n[4] != 0 || fold_bad != 0) {
                float _add_26;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(prev[4]), "f"(v4_0[0]));
                prev[4] = _add_26;
            }
            if (n[5] != 0 || fold_bad != 0) {
                float _add_27;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(prev[5]), "f"(v4_0[1]));
                prev[5] = _add_27;
            }
            if (n[6] != 0 || fold_bad != 0) {
                float _add_28;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(prev[6]), "f"(v4_0[2]));
                prev[6] = _add_28;
            }
            if (n[7] != 0 || fold_bad != 0) {
                float _add_29;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(prev[7]), "f"(v4_0[3]));
                prev[7] = _add_29;
            }
            {
                float4 _v4 = make_float4(prev[4 + 0], prev[4 + 1], prev[4 + 2], prev[4 + 3]);
                *reinterpret_cast<float4*>(dst_packed + (long long)row * (long long)dst_row_stride + (long long)(128 + lane_1 * 4)) = _v4;
            }
            float v4_1[4];
            v4_1[0] = mul_f32_rn_noftz((float)n[8], s_lat);
            if (fold_bad != 0) {
                v4_1[0] = nan;
            }
            v4_1[1] = mul_f32_rn_noftz((float)n[9], s_lat);
            if (fold_bad != 0) {
                v4_1[1] = nan;
            }
            v4_1[2] = mul_f32_rn_noftz((float)n[10], s_lat);
            if (fold_bad != 0) {
                v4_1[2] = nan;
            }
            v4_1[3] = mul_f32_rn_noftz((float)n[11], s_lat);
            if (fold_bad != 0) {
                v4_1[3] = nan;
            }
            if (n[8] != 0 || fold_bad != 0) {
                float _add_30;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(prev[8]), "f"(v4_1[0]));
                prev[8] = _add_30;
            }
            if (n[9] != 0 || fold_bad != 0) {
                float _add_31;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(prev[9]), "f"(v4_1[1]));
                prev[9] = _add_31;
            }
            if (n[10] != 0 || fold_bad != 0) {
                float _add_32;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(prev[10]), "f"(v4_1[2]));
                prev[10] = _add_32;
            }
            if (n[11] != 0 || fold_bad != 0) {
                float _add_33;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(prev[11]), "f"(v4_1[3]));
                prev[11] = _add_33;
            }
            {
                float4 _v4 = make_float4(prev[8 + 0], prev[8 + 1], prev[8 + 2], prev[8 + 3]);
                *reinterpret_cast<float4*>(dst_packed + (long long)row * (long long)dst_row_stride + (long long)(256 + lane_1 * 4)) = _v4;
            }
            float v4_2[4];
            v4_2[0] = mul_f32_rn_noftz((float)n[12], s_lat);
            if (fold_bad != 0) {
                v4_2[0] = nan;
            }
            v4_2[1] = mul_f32_rn_noftz((float)n[13], s_lat);
            if (fold_bad != 0) {
                v4_2[1] = nan;
            }
            v4_2[2] = mul_f32_rn_noftz((float)n[14], s_lat);
            if (fold_bad != 0) {
                v4_2[2] = nan;
            }
            v4_2[3] = mul_f32_rn_noftz((float)n[15], s_lat);
            if (fold_bad != 0) {
                v4_2[3] = nan;
            }
            if (n[12] != 0 || fold_bad != 0) {
                float _add_34;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(prev[12]), "f"(v4_2[0]));
                prev[12] = _add_34;
            }
            if (n[13] != 0 || fold_bad != 0) {
                float _add_35;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(prev[13]), "f"(v4_2[1]));
                prev[13] = _add_35;
            }
            if (n[14] != 0 || fold_bad != 0) {
                float _add_36;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(prev[14]), "f"(v4_2[2]));
                prev[14] = _add_36;
            }
            if (n[15] != 0 || fold_bad != 0) {
                float _add_37;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(prev[15]), "f"(v4_2[3]));
                prev[15] = _add_37;
            }
            {
                float4 _v4 = make_float4(prev[12 + 0], prev[12 + 1], prev[12 + 2], prev[12 + 3]);
                *reinterpret_cast<float4*>(dst_packed + (long long)row * (long long)dst_row_stride + (long long)(384 + lane_1 * 4)) = _v4;
            }
            if (lane_1 < 16) {
                float s_rope = __uint_as_float((unsigned int)(e_r + 127) << 23);
                float v4_3[4];
                v4_3[0] = mul_f32_rn_noftz((float)n[16], s_rope);
                if (fold_bad != 0) {
                    v4_3[0] = nan;
                }
                v4_3[1] = mul_f32_rn_noftz((float)n[17], s_rope);
                if (fold_bad != 0) {
                    v4_3[1] = nan;
                }
                v4_3[2] = mul_f32_rn_noftz((float)n[18], s_rope);
                if (fold_bad != 0) {
                    v4_3[2] = nan;
                }
                v4_3[3] = mul_f32_rn_noftz((float)n[19], s_rope);
                if (fold_bad != 0) {
                    v4_3[3] = nan;
                }
                if (n[16] != 0 || fold_bad != 0) {
                    float _add_38;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(prev[16]), "f"(v4_3[0]));
                    prev[16] = _add_38;
                }
                if (n[17] != 0 || fold_bad != 0) {
                    float _add_39;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(prev[17]), "f"(v4_3[1]));
                    prev[17] = _add_39;
                }
                if (n[18] != 0 || fold_bad != 0) {
                    float _add_40;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(prev[18]), "f"(v4_3[2]));
                    prev[18] = _add_40;
                }
                if (n[19] != 0 || fold_bad != 0) {
                    float _add_41;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(prev[19]), "f"(v4_3[3]));
                    prev[19] = _add_41;
                }
                {
                    float4 _v4 = make_float4(prev[16 + 0], prev[16 + 1], prev[16 + 2], prev[16 + 3]);
                    *reinterpret_cast<float4*>(dst_packed + (long long)row * (long long)dst_row_stride + (long long)(512 + lane_1 * 4)) = _v4;
                }
            }
        }
        if (lane_1 == 0) {
            redo_mark[row] = flagged;
            if (nonfinite != 0) {
                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(det_globals) + (9))), "r"(static_cast<unsigned int>(1)) : "memory");
            }
        }
    }
}

} // extern "C"
