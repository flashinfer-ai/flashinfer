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
#define NUM_MAIN_1_STAGES 1
#define SMEM_RED_OFF 0
#define SMEM_RED_STAGE_BYTES 8192
#define SMEM_RED_STRIDE 8192
#define SMEM_FLAG_OFF 8192
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_TOTAL 8320
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_nvfp4_attention_4899aa3c1a4719209562(__nv_bfloat16* __restrict__ K, float* __restrict__ partials, unsigned int* __restrict__ counters, float* __restrict__ mean_k, int seqlen, long long k_stride_s, long long k_stride_h, long long k_stride_b, int heads, int num_tiles, int total_bh)
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

    // Kernel setup ops
    float* red = reinterpret_cast<float*>(smem_raw + 0);
    const int red_addr = smem + 0;
    int* flag = reinterpret_cast<int*>(smem_raw + 8192);
    const int flag_addr = smem + 8192;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int col = (tid & 15) * 8;
    int rgrp = tid >> 4;
    float seq_f = (float)seqlen;
    float acc[8];
    float vals0[8];
    float vals1[8];
    float vals2[8];
    float vals3[8];
    float vals4[8];
    float vals5[8];
    float vals6[8];
    float vals7[8];
    #pragma unroll 1
    for (int work = bid; work < total_bh * num_tiles; work += num_bids) {
        int bh = work / num_tiles;
        int tile = work - bh * num_tiles;
        int batch = bh / heads;
        int head = bh - batch * heads;
        long long base = (long long)batch * k_stride_b + (long long)head * k_stride_h + (long long)col;
        int rbase = tile * 512 + rgrp;
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll 1
        for (int p0 = 0; p0 < 32; p0 += 8) {
            if (rbase + p0 * 16 < seqlen) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(K + base + (long long)(rbase + p0 * 16) * k_stride_s);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vals0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                            (&vals0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                vals0[0] = 0.0f;
                vals0[1] = 0.0f;
                vals0[2] = 0.0f;
                vals0[3] = 0.0f;
                vals0[4] = 0.0f;
                vals0[5] = 0.0f;
                vals0[6] = 0.0f;
                vals0[7] = 0.0f;
            }
            if (rbase + (p0 + 1) * 16 < seqlen) {
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(K + base + (long long)(rbase + (p0 + 1) * 16) * k_stride_s);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vals1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                            (&vals1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                vals1[0] = 0.0f;
                vals1[1] = 0.0f;
                vals1[2] = 0.0f;
                vals1[3] = 0.0f;
                vals1[4] = 0.0f;
                vals1[5] = 0.0f;
                vals1[6] = 0.0f;
                vals1[7] = 0.0f;
            }
            if (rbase + (p0 + 2) * 16 < seqlen) {
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(K + base + (long long)(rbase + (p0 + 2) * 16) * k_stride_s);
                    uint4 _vld_2[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_2[_blk] = _vptr_2[_blk];
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vals2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                            (&vals2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                vals2[0] = 0.0f;
                vals2[1] = 0.0f;
                vals2[2] = 0.0f;
                vals2[3] = 0.0f;
                vals2[4] = 0.0f;
                vals2[5] = 0.0f;
                vals2[6] = 0.0f;
                vals2[7] = 0.0f;
            }
            if (rbase + (p0 + 3) * 16 < seqlen) {
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(K + base + (long long)(rbase + (p0 + 3) * 16) * k_stride_s);
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vals3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                            (&vals3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                vals3[0] = 0.0f;
                vals3[1] = 0.0f;
                vals3[2] = 0.0f;
                vals3[3] = 0.0f;
                vals3[4] = 0.0f;
                vals3[5] = 0.0f;
                vals3[6] = 0.0f;
                vals3[7] = 0.0f;
            }
            if (rbase + (p0 + 4) * 16 < seqlen) {
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(K + base + (long long)(rbase + (p0 + 4) * 16) * k_stride_s);
                    uint4 _vld_4[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_4[_blk] = _vptr_4[_blk];
                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vals4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                            (&vals4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                vals4[0] = 0.0f;
                vals4[1] = 0.0f;
                vals4[2] = 0.0f;
                vals4[3] = 0.0f;
                vals4[4] = 0.0f;
                vals4[5] = 0.0f;
                vals4[6] = 0.0f;
                vals4[7] = 0.0f;
            }
            if (rbase + (p0 + 5) * 16 < seqlen) {
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(K + base + (long long)(rbase + (p0 + 5) * 16) * k_stride_s);
                    uint4 _vld_5[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_5[_blk] = _vptr_5[_blk];
                        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vals5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                            (&vals5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                vals5[0] = 0.0f;
                vals5[1] = 0.0f;
                vals5[2] = 0.0f;
                vals5[3] = 0.0f;
                vals5[4] = 0.0f;
                vals5[5] = 0.0f;
                vals5[6] = 0.0f;
                vals5[7] = 0.0f;
            }
            if (rbase + (p0 + 6) * 16 < seqlen) {
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(K + base + (long long)(rbase + (p0 + 6) * 16) * k_stride_s);
                    uint4 _vld_6[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_6[_blk] = _vptr_6[_blk];
                        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vals6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                            (&vals6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                vals6[0] = 0.0f;
                vals6[1] = 0.0f;
                vals6[2] = 0.0f;
                vals6[3] = 0.0f;
                vals6[4] = 0.0f;
                vals6[5] = 0.0f;
                vals6[6] = 0.0f;
                vals6[7] = 0.0f;
            }
            if (rbase + (p0 + 7) * 16 < seqlen) {
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(K + base + (long long)(rbase + (p0 + 7) * 16) * k_stride_s);
                    uint4 _vld_7[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_7[_blk] = _vptr_7[_blk];
                        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vals7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                            (&vals7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                vals7[0] = 0.0f;
                vals7[1] = 0.0f;
                vals7[2] = 0.0f;
                vals7[3] = 0.0f;
                vals7[4] = 0.0f;
                vals7[5] = 0.0f;
                vals7[6] = 0.0f;
                vals7[7] = 0.0f;
            }
            acc[0] = acc[0] + vals0[0];
            acc[1] = acc[1] + vals0[1];
            acc[2] = acc[2] + vals0[2];
            acc[3] = acc[3] + vals0[3];
            acc[4] = acc[4] + vals0[4];
            acc[5] = acc[5] + vals0[5];
            acc[6] = acc[6] + vals0[6];
            acc[7] = acc[7] + vals0[7];
            acc[0] = acc[0] + vals1[0];
            acc[1] = acc[1] + vals1[1];
            acc[2] = acc[2] + vals1[2];
            acc[3] = acc[3] + vals1[3];
            acc[4] = acc[4] + vals1[4];
            acc[5] = acc[5] + vals1[5];
            acc[6] = acc[6] + vals1[6];
            acc[7] = acc[7] + vals1[7];
            acc[0] = acc[0] + vals2[0];
            acc[1] = acc[1] + vals2[1];
            acc[2] = acc[2] + vals2[2];
            acc[3] = acc[3] + vals2[3];
            acc[4] = acc[4] + vals2[4];
            acc[5] = acc[5] + vals2[5];
            acc[6] = acc[6] + vals2[6];
            acc[7] = acc[7] + vals2[7];
            acc[0] = acc[0] + vals3[0];
            acc[1] = acc[1] + vals3[1];
            acc[2] = acc[2] + vals3[2];
            acc[3] = acc[3] + vals3[3];
            acc[4] = acc[4] + vals3[4];
            acc[5] = acc[5] + vals3[5];
            acc[6] = acc[6] + vals3[6];
            acc[7] = acc[7] + vals3[7];
            acc[0] = acc[0] + vals4[0];
            acc[1] = acc[1] + vals4[1];
            acc[2] = acc[2] + vals4[2];
            acc[3] = acc[3] + vals4[3];
            acc[4] = acc[4] + vals4[4];
            acc[5] = acc[5] + vals4[5];
            acc[6] = acc[6] + vals4[6];
            acc[7] = acc[7] + vals4[7];
            acc[0] = acc[0] + vals5[0];
            acc[1] = acc[1] + vals5[1];
            acc[2] = acc[2] + vals5[2];
            acc[3] = acc[3] + vals5[3];
            acc[4] = acc[4] + vals5[4];
            acc[5] = acc[5] + vals5[5];
            acc[6] = acc[6] + vals5[6];
            acc[7] = acc[7] + vals5[7];
            acc[0] = acc[0] + vals6[0];
            acc[1] = acc[1] + vals6[1];
            acc[2] = acc[2] + vals6[2];
            acc[3] = acc[3] + vals6[3];
            acc[4] = acc[4] + vals6[4];
            acc[5] = acc[5] + vals6[5];
            acc[6] = acc[6] + vals6[6];
            acc[7] = acc[7] + vals6[7];
            acc[0] = acc[0] + vals7[0];
            acc[1] = acc[1] + vals7[1];
            acc[2] = acc[2] + vals7[2];
            acc[3] = acc[3] + vals7[3];
            acc[4] = acc[4] + vals7[4];
            acc[5] = acc[5] + vals7[5];
            acc[6] = acc[6] + vals7[6];
            acc[7] = acc[7] + vals7[7];
        }
        red[rgrp * 128 + col] = acc[0];
        red[rgrp * 128 + col + 1] = acc[1];
        red[rgrp * 128 + col + 2] = acc[2];
        red[rgrp * 128 + col + 3] = acc[3];
        red[rgrp * 128 + col + 4] = acc[4];
        red[rgrp * 128 + col + 5] = acc[5];
        red[rgrp * 128 + col + 6] = acc[6];
        red[rgrp * 128 + col + 7] = acc[7];
        __syncthreads();
        asm volatile("griddepcontrol.wait;" ::: "memory");
        if (num_tiles == 1) {
            if (tid < 128) {
                float total = 0.0f;
                #pragma unroll
                for (int g = 0; g < 16; g++) {
                    total = total + red[g * 128 + tid];
                }
                float _fdiv_8 = __fdiv_rn(total, seq_f);
                *(reinterpret_cast<float*>(mean_k + (bh * 128 + tid)) + (0)) = _fdiv_8;
            }
        } else {
            if (tid < 128) {
                float tsum = 0.0f;
                #pragma unroll
                for (int g_1 = 0; g_1 < 16; g_1++) {
                    tsum = tsum + red[g_1 * 128 + tid];
                }
                *(reinterpret_cast<float*>(partials + ((bh * num_tiles + tile) * 128 + tid)) + (0)) = tsum;
            }
            __syncthreads();
            if (tid == 0) {
                unsigned int old;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(old) : "l"(&counters[bh]), "r"(static_cast<uint32_t>(1)) : "memory");
                int last = (((int)old == num_tiles - 1) ? 1 : 0);
                flag[0] = last;
                if (last == 1) {
                    *(reinterpret_cast<unsigned int*>(counters + bh) + (0)) = 0;
                }
            }
            __syncthreads();
            if (flag[0] == 1) {
                if (tid < 128) {
                    float fold = 0.0f;
                    #pragma unroll 1
                    for (int t = 0; t < num_tiles; t++) {
                        fold = fold + partials[(bh * num_tiles + t) * 128 + tid];
                    }
                    float _fdiv_9 = __fdiv_rn(fold, seq_f);
                    *(reinterpret_cast<float*>(mean_k + (bh * 128 + tid)) + (0)) = _fdiv_9;
                }
            }
        }
        __syncthreads();
    }
}

} // extern "C"
