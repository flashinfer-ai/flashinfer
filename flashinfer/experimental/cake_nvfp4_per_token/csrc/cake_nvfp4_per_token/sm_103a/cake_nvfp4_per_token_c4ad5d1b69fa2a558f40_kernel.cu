/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
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
#define SMEM_REDUCE_SCRATCH_OFF 0
#define SMEM_REDUCE_SCRATCH_STAGE_BYTES 128
#define SMEM_REDUCE_SCRATCH_STRIDE 128
#define SMEM_TOTAL 128
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(128, 5) void
kernel_cake_nvfp4_per_token_c4ad5d1b69fa2a558f40(__nv_bfloat16* __restrict__ x, uint8_t* __restrict__ out_fp4, uint8_t* __restrict__ out_sf, float* __restrict__ per_token_scale, float* __restrict__ global_scale_inv, float* __restrict__ out_scale, int M)
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
    float* reduce_scratch = reinterpret_cast<float*>(smem_raw + 0);
    const int reduce_scratch_addr = smem + 0;

    // === Task calls (dependency order) ===
    int row = bid;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    if (row < M) {
        float gs_inv = global_scale_inv[0];
        float fold_scale = 1.0f;
        fold_scale = out_scale[0];
        unsigned int words[64];
        float block_max[8];
        unsigned long long row_base = (unsigned long long)row * 16384;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int col = tid + j * 128;
            int _min_0 = ((col) < (1023) ? (col) : (1023));
            int col_ld = _min_0;
            unsigned long long src = row_base + (unsigned long long)col_ld * 16;
            {
                asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                    : "=r"(words[j * 8 + 0]), "=r"(words[j * 8 + 1]), "=r"(words[j * 8 + 2]), "=r"(words[j * 8 + 3]), "=r"(words[j * 8 + 4]), "=r"(words[j * 8 + 5]), "=r"(words[j * 8 + 6]), "=r"(words[j * 8 + 7]) : "l"((const void*)((const char*)(x + src) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            }
        }
        float local_amax = 0.0f;
        #pragma unroll
        for (int j_1 = 0; j_1 < 8; j_1++) {
            uint32_t _bf16x2_abs_0;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(words[j_1 * 8]));
            unsigned int pair_max = _bf16x2_abs_0;
            #pragma unroll
            for (int w = 1; w < 8; w++) {
                uint32_t _bf16x2_abs_1;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(words[j_1 * 8 + w]));
                uint32_t _bf16x2_max_0;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(pair_max), "r"(_bf16x2_abs_1));
                pair_max = _bf16x2_max_0;
            }
            uint16_t _bf16_max_0;
            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_0) : "h"((uint16_t)(pair_max & 65535)), "h"((uint16_t)(pair_max >> 16)));
            float _cvt_f32_bf16_0;
            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_0) : "h"((uint16_t)(_bf16_max_0)));
            block_max[j_1] = _cvt_f32_bf16_0;
            float _max_0 = max_noftz(local_amax, block_max[j_1]);
            local_amax = _max_0;
        }
        // BlockReduceF32: threads=128, warps=4, all-thread broadcast
        float _block_reduce_f32_1_accum = local_amax;
        const int _block_reduce_f32_1_lane = threadIdx.x & 31;
        const int _block_reduce_f32_1_warp = threadIdx.x >> 5;
        // Full CTA schedule: one barrier; every warp performs level two.
        #pragma unroll
        for (int _block_reduce_f32_1_offset_level1 = 16; _block_reduce_f32_1_offset_level1 > 0; _block_reduce_f32_1_offset_level1 >>= 1) {
            _block_reduce_f32_1_accum = fmaxf(_block_reduce_f32_1_accum, __shfl_xor_sync(0xffffffffu, _block_reduce_f32_1_accum, _block_reduce_f32_1_offset_level1));
        }
        if (_block_reduce_f32_1_lane == 0) { reduce_scratch[_block_reduce_f32_1_warp] = _block_reduce_f32_1_accum; }
        __syncthreads();
        float _block_reduce_f32_1_level2 = (_block_reduce_f32_1_lane < 4) ? reduce_scratch[_block_reduce_f32_1_lane] : -CUDART_INF_F;
        #pragma unroll
        for (int _block_reduce_f32_1_offset_level2 = 16; _block_reduce_f32_1_offset_level2 > 0; _block_reduce_f32_1_offset_level2 >>= 1) {
            _block_reduce_f32_1_level2 = fmaxf(_block_reduce_f32_1_level2, __shfl_xor_sync(0xffffffffu, _block_reduce_f32_1_level2, _block_reduce_f32_1_offset_level2));
        }
        float _block_reduce_f32_0 = _block_reduce_f32_1_level2;
        float row_amax = _block_reduce_f32_0;
        float token_scale = row_amax * gs_inv;
        float _rcp_0 = approx_rcp(token_scale);
        float encode = _rcp_0;
        if (row_amax == 0.0f) {
            token_scale = 0.0f;
            encode = 3.4028234663852886e+38f;
        }
        float token_scale_out = token_scale;
        token_scale_out = token_scale * fold_scale;
        if (tid == 0) {
            *(reinterpret_cast<float*>(per_token_scale + row) + (0)) = token_scale_out;
        }
        float _rcp_1 = approx_rcp(6.0f);
        float rcp_six = _rcp_1;
        float _rcp_2 = approx_rcp(encode);
        float rcp_encode = _rcp_2;
        unsigned long long out_row_bytes = (unsigned long long)row * 8192;
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; j_2++) {
            int col_1 = tid + j_2 * 128;
            if (col_1 < 1024) {
                float sf_value = encode * (block_max[j_2] * rcp_six);
                float _fp8_rt_0;
                uint16_t _e4m3x2_2;
                uint32_t _f16x2_2;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_2) : "f"(0.0f), "f"(sf_value));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2) : "h"(_e4m3x2_2));
                uint16_t _fp8_h0_2 = (uint16_t)(_f16x2_2 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_2));
                float sf_decoded = _fp8_rt_0;
                float _rcp_3 = approx_rcp(sf_decoded * rcp_encode);
                float output_scale = _rcp_3;
                if (sf_decoded == 0.0f) {
                    output_scale = 0.0f;
                }
                unsigned int packed[2];
                float values[8];
                #pragma unroll
                for (int w_1 = 0; w_1 < 4; w_1++) {
                    unsigned int word = words[j_2 * 8 + w_1];
                    values[2 * w_1] = __uint_as_float(word << 16) * output_scale;
                    values[2 * w_1 + 1] = __uint_as_float(word & 4294901760u) * output_scale;
                }
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(values[0]), "f"(values[1]), "f"(values[2]), "f"(values[3]), "f"(values[4]), "f"(values[5]), "f"(values[6]), "f"(values[7]));
                #pragma unroll
                for (int w_2 = 0; w_2 < 4; w_2++) {
                    unsigned int word_1 = words[j_2 * 8 + 4 + w_2];
                    values[2 * w_2] = __uint_as_float(word_1 << 16) * output_scale;
                    values[2 * w_2 + 1] = __uint_as_float(word_1 & 4294901760u) * output_scale;
                }
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[1]) : "f"(values[0]), "f"(values[1]), "f"(values[2]), "f"(values[3]), "f"(values[4]), "f"(values[5]), "f"(values[6]), "f"(values[7]));
                unsigned long long fp4_offset = out_row_bytes + (unsigned long long)col_1 * 8;
                asm volatile("st.global.v2.b32 [%0], {%1, %2};" :: "l"(out_fp4 + fp4_offset + (0)), "r"(packed[(0) + 0]), "r"(packed[(0) + 1]) : "memory");
                {
                    unsigned short _sf_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(sf_value));
                    *(reinterpret_cast<unsigned char*>(out_sf + (((unsigned int)col_1 & 3) + ((unsigned int)col_1 >> 2 << 9) + (((unsigned int)row & 31) << 4) + (((unsigned int)row >> 5 & 3) << 2) + ((unsigned int)row >> 7) * 131072)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                }
            }
        }
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int pad_col = 1024 + tid + p * 128;
            if (pad_col < 1024) {
                *(reinterpret_cast<unsigned char*>(out_sf + (((unsigned int)pad_col & 3) + ((unsigned int)pad_col >> 2 << 9) + (((unsigned int)row & 31) << 4) + (((unsigned int)row >> 5 & 3) << 2) + ((unsigned int)row >> 7) * 131072)) + (0)) = (unsigned char)((unsigned int)0);
            }
        }
    } else {
        #pragma unroll
        for (int p_1 = 0; p_1 < 8; p_1++) {
            int pad_col_1 = tid + p_1 * 128;
            if (pad_col_1 < 1024) {
                *(reinterpret_cast<unsigned char*>(out_sf + (((unsigned int)pad_col_1 & 3) + ((unsigned int)pad_col_1 >> 2 << 9) + (((unsigned int)row & 31) << 4) + (((unsigned int)row >> 5 & 3) << 2) + ((unsigned int)row >> 7) * 131072)) + (0)) = (unsigned char)((unsigned int)0);
            }
        }
    }
}

} // extern "C"
