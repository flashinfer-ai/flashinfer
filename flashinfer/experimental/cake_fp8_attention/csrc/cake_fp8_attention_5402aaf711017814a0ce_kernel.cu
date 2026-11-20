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
#define SMEM_RED_STAGE_BYTES 16384
#define SMEM_RED_STRIDE 16384
#define SMEM_TOTAL 16384
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_fp8_attention_5402aaf711017814a0ce(__nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, float* __restrict__ partials, int seqlen, int num_heads, int num_tiles, int k_stride_b, int k_stride_s, int k_stride_h, int v_stride_b, int v_stride_s, int v_stride_h)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* red = reinterpret_cast<float*>(smem_raw + 0);
    const int red_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int col = (tid & 15) * 8;
    int rgrp = tid >> 4;
    int tile = bid % num_tiles;
    int bh = bid / num_tiles;
    int batch = bh / num_heads;
    int head = bh % num_heads;
    int row_begin = tile * 128;
    int row_end = ((row_begin + 128 < seqlen) ? row_begin + 128 : seqlen);
    int k_base = batch * k_stride_b + head * k_stride_h + col;
    int v_base = batch * v_stride_b + head * v_stride_h + col;
    float ksum[8];
    float vmax[8];
    for (int e = 0; e < 8; e++) {
        ksum[e] = 0.0f;
        vmax[e] = 0.0f;
    }
    #pragma unroll 1
    for (int row = row_begin + rgrp; row < row_end; row += 16) {
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(K + (k_base + row * k_stride_s) + 0);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        float _vec_load_1[8];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(V + (v_base + row * v_stride_s) + 0);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        for (int e_1 = 0; e_1 < 8; e_1++) {
            float kf = _vec_load_0[e_1];
            float vf = _vec_load_1[e_1];
            ksum[e_1] = ksum[e_1] + kf;
            vmax[e_1] = fmaxf(vmax[e_1], fabsf(vf));
        }
    }
    for (int e_2 = 0; e_2 < 8; e_2++) {
        red[rgrp * 128 + col + e_2] = ksum[e_2];
        red[(16 + rgrp) * 128 + col + e_2] = vmax[e_2];
    }
    __syncthreads();
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (tid < 128) {
        float total_k = 0.0f;
        float vmx = 0.0f;
        for (int g = 0; g < 16; g++) {
            total_k = total_k + red[g * 128 + tid];
            vmx = fmaxf(vmx, red[(16 + g) * 128 + tid]);
        }
        int out = bid * 256 + tid;
        *(reinterpret_cast<float*>(partials + out) + (0)) = total_k;
        *(reinterpret_cast<float*>(partials + (out + 128)) + (0)) = vmx;
    }
}

} // extern "C"
