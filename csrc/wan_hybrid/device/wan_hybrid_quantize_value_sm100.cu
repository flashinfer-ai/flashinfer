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

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) WanHybridTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) WanHybridTensorMapPack { WanHybridTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define WAN_HYBRID_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_V_SMEM_OFF 0
#define SMEM_V_SMEM_STAGE_BYTES 33280
#define SMEM_V_SMEM_STRIDE 33280
#define SMEM_TOTAL 33280
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max_noftz(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}


__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_wan_hybrid_quantize_value(__nv_bfloat16* __restrict__ v, uint8_t* __restrict__ v_fp4_base_t, uint8_t* __restrict__ v_fp4_residual_t, uint8_t* __restrict__ v_scale_base_lo, uint8_t* __restrict__ v_scale_base_hi, uint8_t* __restrict__ v_scale_residual_lo, uint8_t* __restrict__ v_scale_residual_hi, int heads, int seq_len, int padded_seq_len, int logical_num_blocks, int physical_num_blocks)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* v_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int v_smem_addr = smem + 0;

    // === Task calls (dependency order) ===
    int tile = bid;
    int bh = tile / logical_num_blocks;
    int block = tile - bh * logical_num_blocks;
    int head = bh % heads;
    int batch_idx = bh / heads;
    int token_base = block * 128;
    #pragma unroll
    for (int iteration = 0; iteration < 8; iteration++) {
        int chunk = tid + iteration * 256;
        int row = chunk / 16;
        int row_chunk = chunk - row * 16;
        int dim_base = row_chunk * 8;
        int token = token_base + row;
        long long input_offset = (((long long)batch_idx * (long long)seq_len + (long long)token) * (long long)heads + (long long)head) * 128 + (long long)dim_base;
        float values[8];
        if (token < seq_len) {
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(v + input_offset);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&values[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_0[_pair]));
                    }
                }
            }
        } else {
            values[0] = 0.0f;
            values[1] = 0.0f;
            values[2] = 0.0f;
            values[3] = 0.0f;
            values[4] = 0.0f;
            values[5] = 0.0f;
            values[6] = 0.0f;
            values[7] = 0.0f;
        }
        unsigned int packed_input[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
            packed_input[_lp] = *(uint32_t*)&_bf2;
        }
        int smem_offset = row * 130 + dim_base;
        #pragma unroll
        for (int word = 0; word < 4; word++) {
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(v_smem_addr + (unsigned int)((smem_offset + word * 2) * 2)), "r"((packed_input[word])));
        }
    }
    __syncthreads();
    #pragma unroll
    for (int iteration_1 = 0; iteration_1 < 4; iteration_1++) {
        int vector = tid + iteration_1 * 256;
        int dim = vector / 8;
        int group = vector - dim * 8;
        float values_1[16];
        #pragma unroll
        for (int element = 0; element < 16; element++) {
            int smem_index = (group * 16 + element) * 130 + dim;
            values_1[element] = v_smem[smem_index];
        }
        float values_max = values_1[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            values_max = max_noftz(values_max, values_1[_lr]);
        }
        float value_max = values_max;
        float values_min = values_1[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            values_min = fminf(values_min, values_1[_lr]);
        }
        float value_min = values_min;
        float _max_0 = max_noftz(value_max, -value_min);
        float amax = _max_0;
        float _max_1 = max_noftz(amax * 0.16666666666666666f, 0.001953125f);
        float raw_base_scale = _max_1;
        float _fp8_rt_0;
        uint16_t _e4m3x2_1;
        uint32_t _f16x2_1;
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(raw_base_scale));
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
        float decoded_base_scale = _fp8_rt_0;
        float inv_base_scale = 1.0f / decoded_base_scale;
        float work[16];
        #pragma unroll
        for (int element_1 = 0; element_1 < 16; element_1++) {
            work[element_1] = values_1[element_1] * inv_base_scale;
        }
        uint32_t _fp4_0[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(work[0]), "f"(work[1]), "f"(work[2]), "f"(work[3]), "f"(work[4]), "f"(work[5]), "f"(work[6]), "f"(work[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[1]) : "f"(work[8]), "f"(work[9]), "f"(work[10]), "f"(work[11]), "f"(work[12]), "f"(work[13]), "f"(work[14]), "f"(work[15]));
        {
            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
            uint32_t _fp4_decode_pairs[4];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_0[0]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&work[0])[0]), "=f"((&work[0])[1])
                : "r"(_fp4_decode_pairs[0]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&work[2])[0]), "=f"((&work[2])[1])
                : "r"(_fp4_decode_pairs[1]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&work[4])[0]), "=f"((&work[4])[1])
                : "r"(_fp4_decode_pairs[2]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&work[6])[0]), "=f"((&work[6])[1])
                : "r"(_fp4_decode_pairs[3]));
            #else
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(work[0]), "=f"(work[1]), "=f"(work[2]), "=f"(work[3]), "=f"(work[4]), "=f"(work[5]), "=f"(work[6]), "=f"(work[7]) : "r"(_fp4_0[0]));
            #endif
        }
        {
            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
            uint32_t _fp4_decode_pairs[4];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_0[1]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&work[8])[0]), "=f"((&work[8])[1])
                : "r"(_fp4_decode_pairs[0]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&work[10])[0]), "=f"((&work[10])[1])
                : "r"(_fp4_decode_pairs[1]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&work[12])[0]), "=f"((&work[12])[1])
                : "r"(_fp4_decode_pairs[2]));
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&work[14])[0]), "=f"((&work[14])[1])
                : "r"(_fp4_decode_pairs[3]));
            #else
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(work[8]), "=f"(work[9]), "=f"(work[10]), "=f"(work[11]), "=f"(work[12]), "=f"(work[13]), "=f"(work[14]), "=f"(work[15]) : "r"(_fp4_0[1]));
            #endif
        }
        #pragma unroll
        for (int element_2 = 0; element_2 < 16; element_2++) {
            work[element_2] = values_1[element_2] - work[element_2] * decoded_base_scale;
        }
        float work_max = work[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            work_max = max_noftz(work_max, work[_lr]);
        }
        float residual_max = work_max;
        float work_min = work[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            work_min = fminf(work_min, work[_lr]);
        }
        float residual_min = work_min;
        float _max_2 = max_noftz(residual_max, -residual_min);
        float residual_amax = _max_2;
        float _max_3 = max_noftz(residual_amax * 0.16666666666666666f, 0.001953125f);
        float raw_residual_scale = _max_3;
        float _fp8_rt_1;
        uint16_t _e4m3x2_2;
        uint32_t _f16x2_2;
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_2) : "f"(0.0f), "f"(raw_residual_scale));
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2) : "h"(_e4m3x2_2));
        uint16_t _fp8_h0_2 = (uint16_t)(_f16x2_2 & 0xFFFFu);
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_2));
        float decoded_residual_scale = _fp8_rt_1;
        float inv_residual_scale = 1.0f / decoded_residual_scale;
        #pragma unroll
        for (int element_3 = 0; element_3 < 16; element_3++) {
            work[element_3] = work[element_3] * inv_residual_scale;
        }
        uint32_t _fp4_1[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[0]) : "f"(work[0]), "f"(work[1]), "f"(work[2]), "f"(work[3]), "f"(work[4]), "f"(work[5]), "f"(work[6]), "f"(work[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[1]) : "f"(work[8]), "f"(work[9]), "f"(work[10]), "f"(work[11]), "f"(work[12]), "f"(work[13]), "f"(work[14]), "f"(work[15]));
        long long output_offset = ((long long)bh * 128 + (long long)dim) * (long long)(padded_seq_len / 2) + (long long)(block * 64) + (long long)(group * 8);
        *(reinterpret_cast<int*>(v_fp4_base_t + output_offset) + (0)) = _fp4_0[0];
        *(reinterpret_cast<int*>(v_fp4_base_t + (output_offset + 4)) + (0)) = _fp4_0[1];
        *(reinterpret_cast<int*>(v_fp4_residual_t + output_offset) + (0)) = _fp4_1[0];
        *(reinterpret_cast<int*>(v_fp4_residual_t + (output_offset + 4)) + (0)) = _fp4_1[1];
        int row_outer = dim / 32;
        int row_inner = dim - row_outer * 32;
        int row_quad = row_inner / 8;
        int row_lane = row_inner - row_quad * 8;
        int group_in_half = group - group / 4 * 4;
        int scale_offset = ((row_quad * 8 + row_lane) * 4 + row_outer) * 4 + group_in_half;
        int physical_tile = bh * physical_num_blocks + block;
        long long scale_tile_offset = (long long)physical_tile * 512;
        if (group < 4) {
            {
                unsigned short _sf_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(raw_base_scale));
                *(reinterpret_cast<unsigned char*>(v_scale_base_lo + (scale_tile_offset + (long long)scale_offset)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
            }
            {
                unsigned short _sf_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(raw_residual_scale));
                *(reinterpret_cast<unsigned char*>(v_scale_residual_lo + (scale_tile_offset + (long long)scale_offset)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
            }
        } else {
            {
                unsigned short _sf_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(raw_base_scale));
                *(reinterpret_cast<unsigned char*>(v_scale_base_hi + (scale_tile_offset + (long long)scale_offset)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
            }
            {
                unsigned short _sf_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(raw_residual_scale));
                *(reinterpret_cast<unsigned char*>(v_scale_residual_hi + (scale_tile_offset + (long long)scale_offset)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
            }
        }
    }
}

} // extern "C"
