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
kernel_cake_dsa_h64_det_train_84376fdd68bed4f727cf(__nv_bfloat16* __restrict__ kv_latent, float* __restrict__ knorm, float* __restrict__ qstat, unsigned int* __restrict__ det_globals, unsigned long long* __restrict__ bound, int num_kv, int num_tokens, int zero_rows, int kv_stride, int qstat_stride, int bound_stride, int part_stride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    unsigned int* part = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int part_addr = smem + 0;

    // === Task calls (declaration order) ===
    int warp_1 = warp;
    int lane_1 = lane;
    unsigned int kmax = 0;
    unsigned int smax = 0;
#if __CUDA_ARCH__ == 1070
    unsigned int words[32];
    unsigned int sbits[4];
    int unit_0 = blockIdx.x * 32 + warp_1 * 4;
#else
    int unit_0 = blockIdx.x * 16 + warp_1 * 2;
#endif
    if (unit_0 < num_kv) {
        long long kbase_0 = (long long)unit_0 * (long long)kv_stride + (long long)(lane_1 * 16);
#if __CUDA_ARCH__ == 1070
#else
        float ss_0 = 0.0f;
        float kv8_0[8];
#endif
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(kv_latent + kbase_0 + 0);
#if __CUDA_ARCH__ == 1070
            uint4* _vdst_0 = reinterpret_cast<uint4*>(&words[0]);
#else
            uint4 _vld_0[1];
#endif
            #pragma unroll
#if __CUDA_ARCH__ == 1070
            for (int _blk = 0; _blk < 2; _blk++) {
                _vdst_0[_blk] = _vptr_0[_blk];
#else
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&kv8_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&kv8_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
#endif
            }
        }
#if __CUDA_ARCH__ == 1070
    }
    sbits[0] = 0;
    if (unit_0 < num_tokens && lane_1 < 4) {
        sbits[0] = __as_u32(qstat[(long long)unit_0 * (long long)qstat_stride + (long long)lane_1]);
    }
    int unit_1 = blockIdx.x * 32 + warp_1 * 4 + 1;
    if (unit_1 < num_kv) {
        long long kbase_1 = (long long)unit_1 * (long long)kv_stride + (long long)(lane_1 * 16);
#else
        ss_0 = __fmaf_rn(kv8_0[0], kv8_0[0], ss_0);
        ss_0 = __fmaf_rn(kv8_0[1], kv8_0[1], ss_0);
        ss_0 = __fmaf_rn(kv8_0[2], kv8_0[2], ss_0);
        ss_0 = __fmaf_rn(kv8_0[3], kv8_0[3], ss_0);
        ss_0 = __fmaf_rn(kv8_0[4], kv8_0[4], ss_0);
        ss_0 = __fmaf_rn(kv8_0[5], kv8_0[5], ss_0);
        ss_0 = __fmaf_rn(kv8_0[6], kv8_0[6], ss_0);
        ss_0 = __fmaf_rn(kv8_0[7], kv8_0[7], ss_0);
        float kv8_1[8];
#endif
        {
#if __CUDA_ARCH__ == 1070
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(kv_latent + kbase_1 + 0);
            uint4* _vdst_1 = reinterpret_cast<uint4*>(&words[8]);
#else
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(kv_latent + (kbase_0 + 8ll) + 0);
            uint4 _vld_1[1];
#endif
            #pragma unroll
#if __CUDA_ARCH__ == 1070
            for (int _blk = 0; _blk < 2; _blk++) {
                _vdst_1[_blk] = _vptr_1[_blk];
#else
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&kv8_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&kv8_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
#endif
            }
        }
#if __CUDA_ARCH__ == 1070
    }
    sbits[1] = 0;
    if (unit_1 < num_tokens && lane_1 < 4) {
        sbits[1] = __as_u32(qstat[(long long)unit_1 * (long long)qstat_stride + (long long)lane_1]);
    }
    int unit_2 = blockIdx.x * 32 + warp_1 * 4 + 2;
    if (unit_2 < num_kv) {
        long long kbase_2 = (long long)unit_2 * (long long)kv_stride + (long long)(lane_1 * 16);
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(kv_latent + kbase_2 + 0);
            uint4* _vdst_2 = reinterpret_cast<uint4*>(&words[16]);
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vdst_2[_blk] = _vptr_2[_blk];
            }
        }
    }
    sbits[2] = 0;
    if (unit_2 < num_tokens && lane_1 < 4) {
        sbits[2] = __as_u32(qstat[(long long)unit_2 * (long long)qstat_stride + (long long)lane_1]);
    }
    int unit_3 = blockIdx.x * 32 + warp_1 * 4 + 3;
    if (unit_3 < num_kv) {
        long long kbase_3 = (long long)unit_3 * (long long)kv_stride + (long long)(lane_1 * 16);
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(kv_latent + kbase_3 + 0);
            uint4* _vdst_3 = reinterpret_cast<uint4*>(&words[24]);
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vdst_3[_blk] = _vptr_3[_blk];
            }
        }
    }
    sbits[3] = 0;
    if (unit_3 < num_tokens && lane_1 < 4) {
        sbits[3] = __as_u32(qstat[(long long)unit_3 * (long long)qstat_stride + (long long)lane_1]);
    }
    int unit_u_0 = blockIdx.x * 32 + warp_1 * 4;
    if (unit_u_0 < num_kv) {
        float ss_0 = 0.0f;
        float lo_0 = __uint_as_float(words[0] << 16);
        float hi_0 = __uint_as_float(words[0] & 4294901760u);
        ss_0 = __fmaf_rn(lo_0, lo_0, ss_0);
        ss_0 = __fmaf_rn(hi_0, hi_0, ss_0);
        float lo_1 = __uint_as_float(words[1] << 16);
        float hi_1 = __uint_as_float(words[1] & 4294901760u);
        ss_0 = __fmaf_rn(lo_1, lo_1, ss_0);
        ss_0 = __fmaf_rn(hi_1, hi_1, ss_0);
        float lo_2 = __uint_as_float(words[2] << 16);
        float hi_2 = __uint_as_float(words[2] & 4294901760u);
        ss_0 = __fmaf_rn(lo_2, lo_2, ss_0);
        ss_0 = __fmaf_rn(hi_2, hi_2, ss_0);
        float lo_3 = __uint_as_float(words[3] << 16);
        float hi_3 = __uint_as_float(words[3] & 4294901760u);
        ss_0 = __fmaf_rn(lo_3, lo_3, ss_0);
        ss_0 = __fmaf_rn(hi_3, hi_3, ss_0);
        float lo_4 = __uint_as_float(words[4] << 16);
        float hi_4 = __uint_as_float(words[4] & 4294901760u);
        ss_0 = __fmaf_rn(lo_4, lo_4, ss_0);
        ss_0 = __fmaf_rn(hi_4, hi_4, ss_0);
        float lo_5 = __uint_as_float(words[5] << 16);
        float hi_5 = __uint_as_float(words[5] & 4294901760u);
        ss_0 = __fmaf_rn(lo_5, lo_5, ss_0);
        ss_0 = __fmaf_rn(hi_5, hi_5, ss_0);
        float lo_6 = __uint_as_float(words[6] << 16);
        float hi_6 = __uint_as_float(words[6] & 4294901760u);
        ss_0 = __fmaf_rn(lo_6, lo_6, ss_0);
        ss_0 = __fmaf_rn(hi_6, hi_6, ss_0);
        float lo_7 = __uint_as_float(words[7] << 16);
        float hi_7 = __uint_as_float(words[7] & 4294901760u);
        ss_0 = __fmaf_rn(lo_7, lo_7, ss_0);
        ss_0 = __fmaf_rn(hi_7, hi_7, ss_0);
#else
        ss_0 = __fmaf_rn(kv8_1[0], kv8_1[0], ss_0);
        ss_0 = __fmaf_rn(kv8_1[1], kv8_1[1], ss_0);
        ss_0 = __fmaf_rn(kv8_1[2], kv8_1[2], ss_0);
        ss_0 = __fmaf_rn(kv8_1[3], kv8_1[3], ss_0);
        ss_0 = __fmaf_rn(kv8_1[4], kv8_1[4], ss_0);
        ss_0 = __fmaf_rn(kv8_1[5], kv8_1[5], ss_0);
        ss_0 = __fmaf_rn(kv8_1[6], kv8_1[6], ss_0);
        ss_0 = __fmaf_rn(kv8_1[7], kv8_1[7], ss_0);
#endif
        float total_0 = ss_0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            total_0 += __shfl_xor_sync(0xFFFFFFFF, total_0, offset);
        float kn_0;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(kn_0) : "f"(total_0));
        if (lane_1 == 0) {
#if __CUDA_ARCH__ == 1070
            knorm[unit_u_0] = kn_0;
#else
            knorm[unit_0] = kn_0;
#endif
        }
        unsigned int kbits_0 = __as_u32(kn_0) & 2147483647;
        if (kbits_0 < 2139095040) {
#if __CUDA_ARCH__ == 1070
            unsigned int _max_4 = ((kmax) > (kbits_0) ? (kmax) : (kbits_0));
            kmax = _max_4;
#else
            unsigned int _max_2 = ((kmax) > (kbits_0) ? (kmax) : (kbits_0));
            kmax = _max_2;
#endif
        }
    }
#if __CUDA_ARCH__ == 1070
    if (unit_u_0 < num_tokens && lane_1 < 4) {
        unsigned int term_0 = sbits[0] & 2147483647;
        if (term_0 < 2139095040) {
            unsigned int _max_5 = ((smax) > (term_0) ? (smax) : (term_0));
            smax = _max_5;
#else
    if (unit_0 < num_tokens && lane_1 < 4) {
        unsigned int sbits_0 = __as_u32(qstat[(long long)unit_0 * (long long)qstat_stride + (long long)lane_1]);
        sbits_0 = sbits_0 & 2147483647;
        if (sbits_0 < 2139095040) {
            unsigned int _max_3 = ((smax) > (sbits_0) ? (smax) : (sbits_0));
            smax = _max_3;
#endif
        }
    }
#if __CUDA_ARCH__ == 1070
    if (unit_u_0 < zero_rows && lane_1 <= 2) {
        bound[(long long)unit_u_0 * (long long)bound_stride + (long long)lane_1 * (long long)part_stride] = 0ull;
#else
    if (unit_0 < zero_rows && lane_1 <= 2) {
        bound[(long long)unit_0 * (long long)bound_stride + (long long)lane_1 * (long long)part_stride] = 0ull;
#endif
    }
#if __CUDA_ARCH__ == 1070
    int unit_u_1 = blockIdx.x * 32 + warp_1 * 4 + 1;
    if (unit_u_1 < num_kv) {
#else
    int unit_1 = blockIdx.x * 16 + warp_1 * 2 + 1;
    if (unit_1 < num_kv) {
        long long kbase_1 = (long long)unit_1 * (long long)kv_stride + (long long)(lane_1 * 16);
#endif
        float ss_1 = 0.0f;
#if __CUDA_ARCH__ == 1070
        float lo_8 = __uint_as_float(words[8] << 16);
        float hi_8 = __uint_as_float(words[8] & 4294901760u);
        ss_1 = __fmaf_rn(lo_8, lo_8, ss_1);
        ss_1 = __fmaf_rn(hi_8, hi_8, ss_1);
        float lo_9 = __uint_as_float(words[9] << 16);
        float hi_9 = __uint_as_float(words[9] & 4294901760u);
        ss_1 = __fmaf_rn(lo_9, lo_9, ss_1);
        ss_1 = __fmaf_rn(hi_9, hi_9, ss_1);
        float lo_10 = __uint_as_float(words[10] << 16);
        float hi_10 = __uint_as_float(words[10] & 4294901760u);
        ss_1 = __fmaf_rn(lo_10, lo_10, ss_1);
        ss_1 = __fmaf_rn(hi_10, hi_10, ss_1);
        float lo_11 = __uint_as_float(words[11] << 16);
        float hi_11 = __uint_as_float(words[11] & 4294901760u);
        ss_1 = __fmaf_rn(lo_11, lo_11, ss_1);
        ss_1 = __fmaf_rn(hi_11, hi_11, ss_1);
        float lo_12 = __uint_as_float(words[12] << 16);
        float hi_12 = __uint_as_float(words[12] & 4294901760u);
        ss_1 = __fmaf_rn(lo_12, lo_12, ss_1);
        ss_1 = __fmaf_rn(hi_12, hi_12, ss_1);
        float lo_13 = __uint_as_float(words[13] << 16);
        float hi_13 = __uint_as_float(words[13] & 4294901760u);
        ss_1 = __fmaf_rn(lo_13, lo_13, ss_1);
        ss_1 = __fmaf_rn(hi_13, hi_13, ss_1);
        float lo_14 = __uint_as_float(words[14] << 16);
        float hi_14 = __uint_as_float(words[14] & 4294901760u);
        ss_1 = __fmaf_rn(lo_14, lo_14, ss_1);
        ss_1 = __fmaf_rn(hi_14, hi_14, ss_1);
        float lo_15 = __uint_as_float(words[15] << 16);
        float hi_15 = __uint_as_float(words[15] & 4294901760u);
        ss_1 = __fmaf_rn(lo_15, lo_15, ss_1);
        ss_1 = __fmaf_rn(hi_15, hi_15, ss_1);
#else
        float kv8_2[8];
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(kv_latent + kbase_1 + 0);
            uint4 _vld_4[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_4[_blk] = _vptr_4[_blk];
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&kv8_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                    (&kv8_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                }
            }
        }
        ss_1 = __fmaf_rn(kv8_2[0], kv8_2[0], ss_1);
        ss_1 = __fmaf_rn(kv8_2[1], kv8_2[1], ss_1);
        ss_1 = __fmaf_rn(kv8_2[2], kv8_2[2], ss_1);
        ss_1 = __fmaf_rn(kv8_2[3], kv8_2[3], ss_1);
        ss_1 = __fmaf_rn(kv8_2[4], kv8_2[4], ss_1);
        ss_1 = __fmaf_rn(kv8_2[5], kv8_2[5], ss_1);
        ss_1 = __fmaf_rn(kv8_2[6], kv8_2[6], ss_1);
        ss_1 = __fmaf_rn(kv8_2[7], kv8_2[7], ss_1);
        float kv8_3[8];
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(kv_latent + (kbase_1 + 8ll) + 0);
            uint4 _vld_5[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_5[_blk] = _vptr_5[_blk];
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&kv8_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                    (&kv8_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                }
            }
        }
        ss_1 = __fmaf_rn(kv8_3[0], kv8_3[0], ss_1);
        ss_1 = __fmaf_rn(kv8_3[1], kv8_3[1], ss_1);
        ss_1 = __fmaf_rn(kv8_3[2], kv8_3[2], ss_1);
        ss_1 = __fmaf_rn(kv8_3[3], kv8_3[3], ss_1);
        ss_1 = __fmaf_rn(kv8_3[4], kv8_3[4], ss_1);
        ss_1 = __fmaf_rn(kv8_3[5], kv8_3[5], ss_1);
        ss_1 = __fmaf_rn(kv8_3[6], kv8_3[6], ss_1);
        ss_1 = __fmaf_rn(kv8_3[7], kv8_3[7], ss_1);
#endif
        float total_1 = ss_1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            total_1 += __shfl_xor_sync(0xFFFFFFFF, total_1, offset);
        float kn_1;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(kn_1) : "f"(total_1));
        if (lane_1 == 0) {
#if __CUDA_ARCH__ == 1070
            knorm[unit_u_1] = kn_1;
#else
            knorm[unit_1] = kn_1;
#endif
        }
        unsigned int kbits_1 = __as_u32(kn_1) & 2147483647;
        if (kbits_1 < 2139095040) {
            unsigned int _max_6 = ((kmax) > (kbits_1) ? (kmax) : (kbits_1));
            kmax = _max_6;
        }
    }
#if __CUDA_ARCH__ == 1070
    if (unit_u_1 < num_tokens && lane_1 < 4) {
        unsigned int term_1 = sbits[1] & 2147483647;
        if (term_1 < 2139095040) {
            unsigned int _max_7 = ((smax) > (term_1) ? (smax) : (term_1));
#else
    if (unit_1 < num_tokens && lane_1 < 4) {
        unsigned int sbits_1 = __as_u32(qstat[(long long)unit_1 * (long long)qstat_stride + (long long)lane_1]);
        sbits_1 = sbits_1 & 2147483647;
        if (sbits_1 < 2139095040) {
            unsigned int _max_7 = ((smax) > (sbits_1) ? (smax) : (sbits_1));
#endif
            smax = _max_7;
        }
    }
#if __CUDA_ARCH__ == 1070
    if (unit_u_1 < zero_rows && lane_1 <= 2) {
        bound[(long long)unit_u_1 * (long long)bound_stride + (long long)lane_1 * (long long)part_stride] = 0ull;
    }
    int unit_u_2 = blockIdx.x * 32 + warp_1 * 4 + 2;
    if (unit_u_2 < num_kv) {
        float ss_2 = 0.0f;
        float lo_16 = __uint_as_float(words[16] << 16);
        float hi_16 = __uint_as_float(words[16] & 4294901760u);
        ss_2 = __fmaf_rn(lo_16, lo_16, ss_2);
        ss_2 = __fmaf_rn(hi_16, hi_16, ss_2);
        float lo_17 = __uint_as_float(words[17] << 16);
        float hi_17 = __uint_as_float(words[17] & 4294901760u);
        ss_2 = __fmaf_rn(lo_17, lo_17, ss_2);
        ss_2 = __fmaf_rn(hi_17, hi_17, ss_2);
        float lo_18 = __uint_as_float(words[18] << 16);
        float hi_18 = __uint_as_float(words[18] & 4294901760u);
        ss_2 = __fmaf_rn(lo_18, lo_18, ss_2);
        ss_2 = __fmaf_rn(hi_18, hi_18, ss_2);
        float lo_19 = __uint_as_float(words[19] << 16);
        float hi_19 = __uint_as_float(words[19] & 4294901760u);
        ss_2 = __fmaf_rn(lo_19, lo_19, ss_2);
        ss_2 = __fmaf_rn(hi_19, hi_19, ss_2);
        float lo_20 = __uint_as_float(words[20] << 16);
        float hi_20 = __uint_as_float(words[20] & 4294901760u);
        ss_2 = __fmaf_rn(lo_20, lo_20, ss_2);
        ss_2 = __fmaf_rn(hi_20, hi_20, ss_2);
        float lo_21 = __uint_as_float(words[21] << 16);
        float hi_21 = __uint_as_float(words[21] & 4294901760u);
        ss_2 = __fmaf_rn(lo_21, lo_21, ss_2);
        ss_2 = __fmaf_rn(hi_21, hi_21, ss_2);
        float lo_22 = __uint_as_float(words[22] << 16);
        float hi_22 = __uint_as_float(words[22] & 4294901760u);
        ss_2 = __fmaf_rn(lo_22, lo_22, ss_2);
        ss_2 = __fmaf_rn(hi_22, hi_22, ss_2);
        float lo_23 = __uint_as_float(words[23] << 16);
        float hi_23 = __uint_as_float(words[23] & 4294901760u);
        ss_2 = __fmaf_rn(lo_23, lo_23, ss_2);
        ss_2 = __fmaf_rn(hi_23, hi_23, ss_2);
        float total_2 = ss_2;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            total_2 += __shfl_xor_sync(0xFFFFFFFF, total_2, offset);
        float kn_2;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(kn_2) : "f"(total_2));
        if (lane_1 == 0) {
            knorm[unit_u_2] = kn_2;
        }
        unsigned int kbits_2 = __as_u32(kn_2) & 2147483647;
        if (kbits_2 < 2139095040) {
            unsigned int _max_8 = ((kmax) > (kbits_2) ? (kmax) : (kbits_2));
            kmax = _max_8;
        }
    }
    if (unit_u_2 < num_tokens && lane_1 < 4) {
        unsigned int term_2 = sbits[2] & 2147483647;
        if (term_2 < 2139095040) {
            unsigned int _max_9 = ((smax) > (term_2) ? (smax) : (term_2));
            smax = _max_9;
        }
    }
    if (unit_u_2 < zero_rows && lane_1 <= 2) {
        bound[(long long)unit_u_2 * (long long)bound_stride + (long long)lane_1 * (long long)part_stride] = 0ull;
    }
    int unit_u_3 = blockIdx.x * 32 + warp_1 * 4 + 3;
    if (unit_u_3 < num_kv) {
        float ss_3 = 0.0f;
        float lo_24 = __uint_as_float(words[24] << 16);
        float hi_24 = __uint_as_float(words[24] & 4294901760u);
        ss_3 = __fmaf_rn(lo_24, lo_24, ss_3);
        ss_3 = __fmaf_rn(hi_24, hi_24, ss_3);
        float lo_25 = __uint_as_float(words[25] << 16);
        float hi_25 = __uint_as_float(words[25] & 4294901760u);
        ss_3 = __fmaf_rn(lo_25, lo_25, ss_3);
        ss_3 = __fmaf_rn(hi_25, hi_25, ss_3);
        float lo_26 = __uint_as_float(words[26] << 16);
        float hi_26 = __uint_as_float(words[26] & 4294901760u);
        ss_3 = __fmaf_rn(lo_26, lo_26, ss_3);
        ss_3 = __fmaf_rn(hi_26, hi_26, ss_3);
        float lo_27 = __uint_as_float(words[27] << 16);
        float hi_27 = __uint_as_float(words[27] & 4294901760u);
        ss_3 = __fmaf_rn(lo_27, lo_27, ss_3);
        ss_3 = __fmaf_rn(hi_27, hi_27, ss_3);
        float lo_28 = __uint_as_float(words[28] << 16);
        float hi_28 = __uint_as_float(words[28] & 4294901760u);
        ss_3 = __fmaf_rn(lo_28, lo_28, ss_3);
        ss_3 = __fmaf_rn(hi_28, hi_28, ss_3);
        float lo_29 = __uint_as_float(words[29] << 16);
        float hi_29 = __uint_as_float(words[29] & 4294901760u);
        ss_3 = __fmaf_rn(lo_29, lo_29, ss_3);
        ss_3 = __fmaf_rn(hi_29, hi_29, ss_3);
        float lo_30 = __uint_as_float(words[30] << 16);
        float hi_30 = __uint_as_float(words[30] & 4294901760u);
        ss_3 = __fmaf_rn(lo_30, lo_30, ss_3);
        ss_3 = __fmaf_rn(hi_30, hi_30, ss_3);
        float lo_31 = __uint_as_float(words[31] << 16);
        float hi_31 = __uint_as_float(words[31] & 4294901760u);
        ss_3 = __fmaf_rn(lo_31, lo_31, ss_3);
        ss_3 = __fmaf_rn(hi_31, hi_31, ss_3);
        float total_3 = ss_3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            total_3 += __shfl_xor_sync(0xFFFFFFFF, total_3, offset);
        float kn_3;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(kn_3) : "f"(total_3));
        if (lane_1 == 0) {
            knorm[unit_u_3] = kn_3;
        }
        unsigned int kbits_3 = __as_u32(kn_3) & 2147483647;
        if (kbits_3 < 2139095040) {
            unsigned int _max_10 = ((kmax) > (kbits_3) ? (kmax) : (kbits_3));
            kmax = _max_10;
        }
    }
    if (unit_u_3 < num_tokens && lane_1 < 4) {
        unsigned int term_3 = sbits[3] & 2147483647;
        if (term_3 < 2139095040) {
            unsigned int _max_11 = ((smax) > (term_3) ? (smax) : (term_3));
            smax = _max_11;
        }
    }
    if (unit_u_3 < zero_rows && lane_1 <= 2) {
        bound[(long long)unit_u_3 * (long long)bound_stride + (long long)lane_1 * (long long)part_stride] = 0ull;
#else
    if (unit_1 < zero_rows && lane_1 <= 2) {
        bound[(long long)unit_1 * (long long)bound_stride + (long long)lane_1 * (long long)part_stride] = 0ull;
#endif
    }
    if (lane_1 < 4) {
        part[warp_1 * 8 + lane_1] = smax;
    }
    if (lane_1 == 0) {
        part[warp_1 * 8 + 4] = kmax;
    }
    __syncthreads();
    if (tid < 8 && (tid < 4 || tid == 4)) {
        unsigned int best = part[tid];
#if __CUDA_ARCH__ == 1070
        unsigned int _max_12 = ((best) > (part[8 + tid]) ? (best) : (part[8 + tid]));
#else
        unsigned int _max_8 = ((best) > (part[8 + tid]) ? (best) : (part[8 + tid]));
        best = _max_8;
        unsigned int _max_9 = ((best) > (part[16 + tid]) ? (best) : (part[16 + tid]));
        best = _max_9;
        unsigned int _max_10 = ((best) > (part[24 + tid]) ? (best) : (part[24 + tid]));
        best = _max_10;
        unsigned int _max_11 = ((best) > (part[32 + tid]) ? (best) : (part[32 + tid]));
        best = _max_11;
        unsigned int _max_12 = ((best) > (part[40 + tid]) ? (best) : (part[40 + tid]));
#endif
        best = _max_12;
#if __CUDA_ARCH__ == 1070
        unsigned int _max_13 = ((best) > (part[16 + tid]) ? (best) : (part[16 + tid]));
#else
        unsigned int _max_13 = ((best) > (part[48 + tid]) ? (best) : (part[48 + tid]));
#endif
        best = _max_13;
#if __CUDA_ARCH__ == 1070
        unsigned int _max_14 = ((best) > (part[24 + tid]) ? (best) : (part[24 + tid]));
#else
        unsigned int _max_14 = ((best) > (part[56 + tid]) ? (best) : (part[56 + tid]));
#endif
        best = _max_14;
#if __CUDA_ARCH__ == 1070
        unsigned int _max_15 = ((best) > (part[32 + tid]) ? (best) : (part[32 + tid]));
        best = _max_15;
        unsigned int _max_16 = ((best) > (part[40 + tid]) ? (best) : (part[40 + tid]));
        best = _max_16;
        unsigned int _max_17 = ((best) > (part[48 + tid]) ? (best) : (part[48 + tid]));
        best = _max_17;
        unsigned int _max_18 = ((best) > (part[56 + tid]) ? (best) : (part[56 + tid]));
        best = _max_18;
#else
#endif
        if (best > det_globals[tid]) {
            asm volatile("red.relaxed.gpu.global.max.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(det_globals) + (tid))), "r"(static_cast<unsigned int>(best)) : "memory");
        }
    }
}

} // extern "C"
