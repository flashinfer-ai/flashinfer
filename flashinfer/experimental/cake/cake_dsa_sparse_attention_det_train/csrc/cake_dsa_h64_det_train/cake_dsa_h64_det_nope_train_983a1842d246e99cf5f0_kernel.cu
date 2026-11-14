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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_det_nope_train_983a1842d246e99cf5f0(unsigned long long* bound, unsigned int* __restrict__ det_globals, int* __restrict__ e_lat, int* dkv_acc, unsigned long long* __restrict__ cnt64, unsigned int* __restrict__ row_pairs, int num_rows, int bound_stride, int part_stride, int dkv_stride, int zero_acc, int shift)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
#if __CUDA_ARCH__ == 1070
#else
    int row = blockIdx.x * 8 + warp;
#endif
    int lane_1 = lane;
#if __CUDA_ARCH__ == 1070
    int first = blockIdx.x * 64 + warp * 8;
    int row = first + lane_1;
    int zero_row = 0;
    if (lane_1 < 8 && row < num_rows) {
#else
    if (row < num_rows) {
#endif
        int exps[1];
        long long bbase = (long long)row * (long long)bound_stride;
        unsigned long long pairs = bound[bbase + (long long)part_stride];
        unsigned long long b0 = bound[bbase];
        int blen0 = 0;
        int use_packed = 0;
        use_packed = (int)(det_globals[10] != 0);
        if (use_packed == 0) {
            unsigned long long _max_0 = ((b0) > (1ull) ? (b0) : (1ull));
            unsigned int hi = (unsigned int)(_max_0 - 1ull >> 32ull);
            unsigned long long _max_1 = ((b0) > (1ull) ? (b0) : (1ull));
            unsigned int lo = (unsigned int)(_max_1 - 1ull & 4294967295ull);
            int _count_leading_zeros_2 = __clz(lo);
            int bits = 32 - _count_leading_zeros_2;
            if (hi != 0) {
                int _count_leading_zeros_3 = __clz(hi);
                bits = 64 - _count_leading_zeros_3;
            }
            blen0 = bits;
        } else {
            unsigned long long fine = pairs;
            pairs = b0 & 67108863ull;
            unsigned long long coarse = b0 >> 26ull;
            unsigned long long b_lo = (coarse << 28ull) + fine;
            unsigned long long b_hi = coarse >> 36ull;
            if (b_lo < fine) {
                b_hi = b_hi + 1ull;
            }
            if (b_hi == 0ull && b_lo == 0ull) {
                b_lo = 1ull;
            }
            if (b_lo == 0ull) {
                b_hi = b_hi - 1ull;
            }
            b_lo = b_lo - 1ull;
            if (b_hi != 0ull) {
                unsigned int hi_1 = (unsigned int)(b_hi >> 32ull);
                unsigned int lo_1 = (unsigned int)(b_hi & 4294967295ull);
                int _count_leading_zeros_4 = __clz(lo_1);
                int bits_1 = 32 - _count_leading_zeros_4;
                if (hi_1 != 0) {
                    int _count_leading_zeros_5 = __clz(hi_1);
                    bits_1 = 64 - _count_leading_zeros_5;
                }
                blen0 = 64 + bits_1;
            } else {
                unsigned int hi_2 = (unsigned int)(b_lo >> 32ull);
                unsigned int lo_2 = (unsigned int)(b_lo & 4294967295ull);
                int _count_leading_zeros_6 = __clz(lo_2);
                int bits_2 = 32 - _count_leading_zeros_6;
                if (hi_2 != 0) {
                    int _count_leading_zeros_7 = __clz(hi_2);
                    bits_2 = 64 - _count_leading_zeros_7;
                }
                blen0 = bits_2;
            }
        }
        int touched = (int)(pairs != 0ull);
        int skip = 0;
        if (touched != 0) {
            unsigned int hi_3 = (unsigned int)(pairs - 1ull >> 32ull);
            unsigned int lo_3 = (unsigned int)(pairs - 1ull & 4294967295ull);
            int _count_leading_zeros_8 = __clz(lo_3);
            int bits_3 = 32 - _count_leading_zeros_8;
            if (hi_3 != 0) {
                int _count_leading_zeros_9 = __clz(hi_3);
                bits_3 = 64 - _count_leading_zeros_9;
            }
            int _max_10 = ((30 - bits_3) > (0) ? (30 - bits_3) : (0));
            skip = _max_10;
        }
        int blen_p_0 = blen0;
        int e_p_0 = 127;
        if (touched != 0) {
            float a_max = __uint_as_float(det_globals[0]);
            float b_max = __uint_as_float(det_globals[1]);
            float k_max = __uint_as_float(det_globals[4]);
            float top = __fmaf_rn(b_max, k_max, a_max);
            int exponent = (int)(__as_u32(top) >> 23 & 255) - 127;
            int _max_12 = ((exponent) > (-86) ? (exponent) : (-86));
            int _min_11 = ((_max_12) < (127) ? (_max_12) : (127));
            exponent = _min_11;
            e_p_0 = blen_p_0 + exponent - 39 - 30 - shift;
            int _max_14 = ((e_p_0) > (-126) ? (e_p_0) : (-126));
            int _min_13 = ((_max_14) < (126) ? (_max_14) : (126));
            e_p_0 = _min_13;
        }
        exps[0] = e_p_0;
#if __CUDA_ARCH__ == 1070
        e_lat[row] = exps[0] & 65535 | skip << 16;
        row_pairs[row] = (unsigned int)pairs;
        if (zero_acc == 2 || zero_acc == 1 && touched != 0) {
            zero_row = 1;
#else
        __syncwarp();
        if (lane_1 == 0) {
            e_lat[row] = exps[0] & 65535 | skip << 16;
            row_pairs[row] = (unsigned int)pairs;
#endif
        }
#if __CUDA_ARCH__ == 1070
    }
    __syncwarp();
    unsigned int zmask = __ballot_sync(0xFFFFFFFF, zero_row != 0);
    if (zmask != 0) {
        int zero4[4];
        zero4[0] = 0;
        zero4[1] = 0;
        zero4[2] = 0;
        zero4[3] = 0;
        if ((zmask & 1) != 0) {
            int zrow_0 = first;
            long long lbase_0 = (long long)zrow_0 * (long long)dkv_stride;
#else
        if (zero_acc == 2 || zero_acc == 1 && touched != 0) {
            int zero4[4];
            zero4[0] = 0;
            zero4[1] = 0;
            zero4[2] = 0;
            zero4[3] = 0;
            long long lbase = (long long)row * (long long)dkv_stride;
#endif
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
#if __CUDA_ARCH__ == 1070
                *reinterpret_cast<int4*>(dkv_acc + lbase_0 + (long long)(lane_1 * 4)) = _iv4;
#else
                *reinterpret_cast<int4*>(dkv_acc + lbase + (long long)(lane_1 * 4)) = _iv4;
#endif
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
#if __CUDA_ARCH__ == 1070
                *reinterpret_cast<int4*>(dkv_acc + lbase_0 + (long long)(128 + lane_1 * 4)) = _iv4;
#else
                *reinterpret_cast<int4*>(dkv_acc + lbase + (long long)(128 + lane_1 * 4)) = _iv4;
#endif
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
#if __CUDA_ARCH__ == 1070
                *reinterpret_cast<int4*>(dkv_acc + lbase_0 + (long long)(256 + lane_1 * 4)) = _iv4;
#else
                *reinterpret_cast<int4*>(dkv_acc + lbase + (long long)(256 + lane_1 * 4)) = _iv4;
#endif
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
#if __CUDA_ARCH__ == 1070
                *reinterpret_cast<int4*>(dkv_acc + lbase_0 + (long long)(384 + lane_1 * 4)) = _iv4;
#else
                *reinterpret_cast<int4*>(dkv_acc + lbase + (long long)(384 + lane_1 * 4)) = _iv4;
#endif
            }
            if (lane_1 < 4) {
#if __CUDA_ARCH__ == 1070
                cnt64[(long long)zrow_0 * 4ll + (long long)lane_1] = 0ull;
            }
        }
        if ((zmask >> 1 & 1) != 0) {
            int zrow_1 = first + 1;
            long long lbase_1 = (long long)zrow_1 * (long long)dkv_stride;
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_1 + (long long)(lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_1 + (long long)(128 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_1 + (long long)(256 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_1 + (long long)(384 + lane_1 * 4)) = _iv4;
            }
            if (lane_1 < 4) {
                cnt64[(long long)zrow_1 * 4ll + (long long)lane_1] = 0ull;
            }
        }
        if ((zmask >> 2 & 1) != 0) {
            int zrow_2 = first + 2;
            long long lbase_2 = (long long)zrow_2 * (long long)dkv_stride;
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_2 + (long long)(lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_2 + (long long)(128 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_2 + (long long)(256 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_2 + (long long)(384 + lane_1 * 4)) = _iv4;
            }
            if (lane_1 < 4) {
                cnt64[(long long)zrow_2 * 4ll + (long long)lane_1] = 0ull;
            }
        }
        if ((zmask >> 3 & 1) != 0) {
            int zrow_3 = first + 3;
            long long lbase_3 = (long long)zrow_3 * (long long)dkv_stride;
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_3 + (long long)(lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_3 + (long long)(128 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_3 + (long long)(256 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_3 + (long long)(384 + lane_1 * 4)) = _iv4;
            }
            if (lane_1 < 4) {
                cnt64[(long long)zrow_3 * 4ll + (long long)lane_1] = 0ull;
            }
        }
        if ((zmask >> 4 & 1) != 0) {
            int zrow_4 = first + 4;
            long long lbase_4 = (long long)zrow_4 * (long long)dkv_stride;
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_4 + (long long)(lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_4 + (long long)(128 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_4 + (long long)(256 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_4 + (long long)(384 + lane_1 * 4)) = _iv4;
            }
            if (lane_1 < 4) {
                cnt64[(long long)zrow_4 * 4ll + (long long)lane_1] = 0ull;
            }
        }
        if ((zmask >> 5 & 1) != 0) {
            int zrow_5 = first + 5;
            long long lbase_5 = (long long)zrow_5 * (long long)dkv_stride;
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_5 + (long long)(lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_5 + (long long)(128 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_5 + (long long)(256 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_5 + (long long)(384 + lane_1 * 4)) = _iv4;
            }
            if (lane_1 < 4) {
                cnt64[(long long)zrow_5 * 4ll + (long long)lane_1] = 0ull;
            }
        }
        if ((zmask >> 6 & 1) != 0) {
            int zrow_6 = first + 6;
            long long lbase_6 = (long long)zrow_6 * (long long)dkv_stride;
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_6 + (long long)(lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_6 + (long long)(128 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_6 + (long long)(256 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_6 + (long long)(384 + lane_1 * 4)) = _iv4;
            }
            if (lane_1 < 4) {
                cnt64[(long long)zrow_6 * 4ll + (long long)lane_1] = 0ull;
            }
        }
        if ((zmask >> 7 & 1) != 0) {
            int zrow_7 = first + 7;
            long long lbase_7 = (long long)zrow_7 * (long long)dkv_stride;
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_7 + (long long)(lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_7 + (long long)(128 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_7 + (long long)(256 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase_7 + (long long)(384 + lane_1 * 4)) = _iv4;
            }
            if (lane_1 < 4) {
                cnt64[(long long)zrow_7 * 4ll + (long long)lane_1] = 0ull;
#else
                cnt64[(long long)row * 4ll + (long long)lane_1] = 0ull;
#endif
            }
        }
    }
}

} // extern "C"
