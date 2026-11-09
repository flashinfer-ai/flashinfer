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
kernel_cake_dsa_h64_det_nope_train_da63029a854ea164bdbd(__nv_bfloat16* __restrict__ kv_latent, float* __restrict__ knorm, float* __restrict__ qstat, unsigned int* __restrict__ det_globals, unsigned long long* __restrict__ bound, int num_kv, int num_tokens, int zero_rows, int kv_stride, int qstat_stride, int bound_stride, int part_stride)
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
    int unit_0 = blockIdx.x * 16 + warp_1 * 2;
    if (unit_0 < num_kv) {
        long long kbase_0 = (long long)unit_0 * (long long)kv_stride + (long long)(lane_1 * 16);
        float ss_0 = 0.0f;
        float kv8_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(kv_latent + kbase_0 + 0);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&kv8_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&kv8_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        ss_0 = __fmaf_rn(kv8_0[0], kv8_0[0], ss_0);
        ss_0 = __fmaf_rn(kv8_0[1], kv8_0[1], ss_0);
        ss_0 = __fmaf_rn(kv8_0[2], kv8_0[2], ss_0);
        ss_0 = __fmaf_rn(kv8_0[3], kv8_0[3], ss_0);
        ss_0 = __fmaf_rn(kv8_0[4], kv8_0[4], ss_0);
        ss_0 = __fmaf_rn(kv8_0[5], kv8_0[5], ss_0);
        ss_0 = __fmaf_rn(kv8_0[6], kv8_0[6], ss_0);
        ss_0 = __fmaf_rn(kv8_0[7], kv8_0[7], ss_0);
        float kv8_1[8];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(kv_latent + (kbase_0 + 8ll) + 0);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&kv8_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&kv8_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        ss_0 = __fmaf_rn(kv8_1[0], kv8_1[0], ss_0);
        ss_0 = __fmaf_rn(kv8_1[1], kv8_1[1], ss_0);
        ss_0 = __fmaf_rn(kv8_1[2], kv8_1[2], ss_0);
        ss_0 = __fmaf_rn(kv8_1[3], kv8_1[3], ss_0);
        ss_0 = __fmaf_rn(kv8_1[4], kv8_1[4], ss_0);
        ss_0 = __fmaf_rn(kv8_1[5], kv8_1[5], ss_0);
        ss_0 = __fmaf_rn(kv8_1[6], kv8_1[6], ss_0);
        ss_0 = __fmaf_rn(kv8_1[7], kv8_1[7], ss_0);
        float total_0 = ss_0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            total_0 += __shfl_xor_sync(0xFFFFFFFF, total_0, offset);
        float kn_0;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(kn_0) : "f"(total_0));
        if (lane_1 == 0) {
            knorm[unit_0] = kn_0;
        }
        unsigned int kbits_0 = __as_u32(kn_0) & 2147483647;
        if (kbits_0 < 2139095040) {
            unsigned int _max_2 = ((kmax) > (kbits_0) ? (kmax) : (kbits_0));
            kmax = _max_2;
        }
    }
    if (unit_0 < num_tokens && lane_1 < 2) {
        unsigned int sbits_0 = __as_u32(qstat[(long long)unit_0 * (long long)qstat_stride + (long long)lane_1]);
        sbits_0 = sbits_0 & 2147483647;
        if (sbits_0 < 2139095040) {
            unsigned int _max_3 = ((smax) > (sbits_0) ? (smax) : (sbits_0));
            smax = _max_3;
        }
    }
    if (unit_0 < zero_rows && lane_1 <= 1) {
        bound[(long long)unit_0 * (long long)bound_stride + (long long)lane_1 * (long long)part_stride] = 0ull;
    }
    int unit_1 = blockIdx.x * 16 + warp_1 * 2 + 1;
    if (unit_1 < num_kv) {
        long long kbase_1 = (long long)unit_1 * (long long)kv_stride + (long long)(lane_1 * 16);
        float ss_1 = 0.0f;
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
        float total_1 = ss_1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            total_1 += __shfl_xor_sync(0xFFFFFFFF, total_1, offset);
        float kn_1;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(kn_1) : "f"(total_1));
        if (lane_1 == 0) {
            knorm[unit_1] = kn_1;
        }
        unsigned int kbits_1 = __as_u32(kn_1) & 2147483647;
        if (kbits_1 < 2139095040) {
            unsigned int _max_6 = ((kmax) > (kbits_1) ? (kmax) : (kbits_1));
            kmax = _max_6;
        }
    }
    if (unit_1 < num_tokens && lane_1 < 2) {
        unsigned int sbits_1 = __as_u32(qstat[(long long)unit_1 * (long long)qstat_stride + (long long)lane_1]);
        sbits_1 = sbits_1 & 2147483647;
        if (sbits_1 < 2139095040) {
            unsigned int _max_7 = ((smax) > (sbits_1) ? (smax) : (sbits_1));
            smax = _max_7;
        }
    }
    if (unit_1 < zero_rows && lane_1 <= 1) {
        bound[(long long)unit_1 * (long long)bound_stride + (long long)lane_1 * (long long)part_stride] = 0ull;
    }
    if (lane_1 < 2) {
        part[warp_1 * 8 + lane_1] = smax;
    }
    if (lane_1 == 0) {
        part[warp_1 * 8 + 4] = kmax;
    }
    __syncthreads();
    if (tid < 8 && (tid < 2 || tid == 4)) {
        unsigned int best = part[tid];
        unsigned int _max_8 = ((best) > (part[8 + tid]) ? (best) : (part[8 + tid]));
        best = _max_8;
        unsigned int _max_9 = ((best) > (part[16 + tid]) ? (best) : (part[16 + tid]));
        best = _max_9;
        unsigned int _max_10 = ((best) > (part[24 + tid]) ? (best) : (part[24 + tid]));
        best = _max_10;
        unsigned int _max_11 = ((best) > (part[32 + tid]) ? (best) : (part[32 + tid]));
        best = _max_11;
        unsigned int _max_12 = ((best) > (part[40 + tid]) ? (best) : (part[40 + tid]));
        best = _max_12;
        unsigned int _max_13 = ((best) > (part[48 + tid]) ? (best) : (part[48 + tid]));
        best = _max_13;
        unsigned int _max_14 = ((best) > (part[56 + tid]) ? (best) : (part[56 + tid]));
        best = _max_14;
        if (best > det_globals[tid]) {
            asm volatile("red.relaxed.gpu.global.max.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(det_globals) + (tid))), "r"(static_cast<unsigned int>(best)) : "memory");
        }
    }
}

} // extern "C"
