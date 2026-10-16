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
// clang-format off
// MiniMax-H3 SM120 (GB202) RMSNorm + indexed AdaLN + activation quantization kernels, generated from the
// Cake kernel schedules and shared by the quantized pre-attention and FC1+SwiGLU translation units:
//   norm_adaln_quant_fp8:   RMSNorm(x) * w, shift + n * (1 + scale), per-token E4M3 (scale = amax / 448)
//   norm_adaln_quant_nvfp4: the same normalization, block-16 NVFP4 (FlashInfer fp4_quantize semantics)
// Each kernel lives in its own namespace; its helper ``#define`` block is undefined again after the namespace.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace h3_norm_adaln_quant_fp8_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_QPA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_PARTIALS_OFF 0
#define SMEM_PARTIALS_STAGE_BYTES 32
#define SMEM_PARTIALS_STRIDE 32
#define SMEM_AMAX_PARTIALS_OFF 32
#define SMEM_AMAX_PARTIALS_STAGE_BYTES 32
#define SMEM_AMAX_PARTIALS_STRIDE 32
#define SMEM_TOTAL 128
#define THREADS 128

#include <math_constants.h>


__global__ __launch_bounds__(128, 4) void
kernel_h3_norm_adaln_quant_fp8(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight, __nv_bfloat16* __restrict__ adaln_scale, __nv_bfloat16* __restrict__ adaln_shift, int* __restrict__ adaln_index, unsigned int* __restrict__ act_q, float* __restrict__ act_scale, int M, float eps)
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
    float* partials = reinterpret_cast<float*>(smem_raw + 0);
    const int partials_addr = smem + 0;
    float* amax_partials = reinterpret_cast<float*>(smem_raw + 32);
    const int amax_partials_addr = smem + 32;

    // === Task calls (dependency order) ===
    #pragma unroll 1
    for (int row = bid; row < M; row += num_bids) {
        float x_vals[48];
        float a_vals[48];
        float total = 0.0f;
        for (int i = 0; i < 3; i++) {
            int chunk = tid + i * 128;
            for (int e = 0; e < 16; e++) {
                x_vals[i * 16 + e] = 0.0f;
            }
            if (chunk < 336) {
                int col = chunk * 16;
                float _vec_load_0[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (row * 5376 + col) + 0);
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
                                : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
                float _vec_load_1[8];
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(x + (row * 5376 + col + 8) + 0);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_1[_pair]));
                        }
                    }
                }
                for (int e_1 = 0; e_1 < 8; e_1++) {
                    float xv0 = _vec_load_0[e_1];
                    float xv1 = _vec_load_1[e_1];
                    x_vals[i * 16 + e_1] = xv0;
                    x_vals[i * 16 + 8 + e_1] = xv1;
                    total += xv0 * xv0 + xv1 * xv1;
                }
            }
        }
        for (int stage = 0; stage < 5; stage++) {
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, total, 16 >> stage);
            total += _shfl_xor_0;
        }
        if (lane == 0) {
            partials[warp] = total;
        }
        __syncthreads();
        total = ((lane < 4) ? partials[lane] : 0.0f);
        for (int stage_1 = 0; stage_1 < 2; stage_1++) {
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, total, 2 >> stage_1);
            total += _shfl_xor_1;
        }
        float _shfl_0;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_0) : "f"(total), "r"(0));
        total = _shfl_0;
        float _fdiv_rn_0 = __fdiv_rn(total, 5376.0f);
        float _rsqrt_0 = rsqrtf(_fdiv_rn_0 + eps);
        float rstd = _rsqrt_0;
        int idx = adaln_index[row];
        int valid = ((idx >= 0 && idx < 9) ? 1 : 0);
        int idx_c = ((valid == 1) ? idx : 0);
        float amax = 0.0f;
        for (int i_1 = 0; i_1 < 3; i_1++) {
            int chunk_1 = tid + i_1 * 128;
            for (int e_2 = 0; e_2 < 16; e_2++) {
                a_vals[i_1 * 16 + e_2] = 0.0f;
            }
            if (chunk_1 < 336) {
                int col_1 = chunk_1 * 16;
                for (int h = 0; h < 2; h++) {
                    float _vec_load_2[8];
                    {
                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x_norm_weight + (col_1 + h * 8) + 0);
                        uint4 _vld_2[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_2[_blk] = _vptr_2[_blk];
                            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_2[_pair]));
                            }
                        }
                    }
                    float _vec_load_3[8];
                    {
                        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(adaln_scale + (idx_c * 5376 + col_1 + h * 8) + 0);
                        uint4 _vld_3[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_3[_blk] = _vptr_3[_blk];
                            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_3[_pair]));
                            }
                        }
                    }
                    float _vec_load_4[8];
                    {
                        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(adaln_shift + (idx_c * 5376 + col_1 + h * 8) + 0);
                        uint4 _vld_4[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_4[_blk] = _vptr_4[_blk];
                            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_4[_pair]));
                            }
                        }
                    }
                    float n_raw[8];
                    float sp1_raw[8];
                    for (int e_3 = 0; e_3 < 8; e_3++) {
                        float wv = _vec_load_2[e_3];
                        float sv = _vec_load_3[e_3];
                        n_raw[e_3] = x_vals[i_1 * 16 + h * 8 + e_3] * rstd * wv;
                        sp1_raw[e_3] = sv + 1.0f;
                    }
                    uint32_t n_raw_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(n_raw[_lp*2 + 0], n_raw[_lp*2+1 + 0]));
                        n_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t sp1_raw_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sp1_raw[_lp*2 + 0], sp1_raw[_lp*2+1 + 0]));
                        sp1_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    float n_raw_bf16_f32[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&n_raw_bf16_f32[_pair * 2])[0]), "=f"((&n_raw_bf16_f32[_pair * 2])[1])
                            : "r"(n_raw_bf16[_pair]));
                    }
                    float sp1_raw_bf16_f32[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&sp1_raw_bf16_f32[_pair * 2])[0]), "=f"((&sp1_raw_bf16_f32[_pair * 2])[1])
                            : "r"(sp1_raw_bf16[_pair]));
                    }
                    float a_raw[8];
                    for (int e_4 = 0; e_4 < 8; e_4++) {
                        float bv = _vec_load_4[e_4];
                        a_raw[e_4] = bv + n_raw_bf16_f32[e_4] * sp1_raw_bf16_f32[e_4];
                    }
                    uint32_t a_raw_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(a_raw[_lp*2 + 0], a_raw[_lp*2+1 + 0]));
                        a_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    float a_raw_bf16_f32[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&a_raw_bf16_f32[_pair * 2])[0]), "=f"((&a_raw_bf16_f32[_pair * 2])[1])
                            : "r"(a_raw_bf16[_pair]));
                    }
                    for (int e_5 = 0; e_5 < 8; e_5++) {
                        float av = ((valid == 1) ? a_raw_bf16_f32[e_5] : 0.0f);
                        a_vals[i_1 * 16 + h * 8 + e_5] = av;
                        float _fabs_0 = fabsf(av);
                        float _fmax_0 = fmaxf(amax, _fabs_0);
                        amax = _fmax_0;
                    }
                }
            }
        }
        for (int stage_2 = 0; stage_2 < 5; stage_2++) {
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, amax, 16 >> stage_2);
            float _fmax_1 = fmaxf(amax, _shfl_xor_2);
            amax = _fmax_1;
        }
        if (lane == 0) {
            amax_partials[warp] = amax;
        }
        __syncthreads();
        amax = ((lane < 4) ? amax_partials[lane] : 0.0f);
        for (int stage_3 = 0; stage_3 < 2; stage_3++) {
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, amax, 2 >> stage_3);
            float _fmax_2 = fmaxf(amax, _shfl_xor_3);
            amax = _fmax_2;
        }
        float _shfl_1;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_1) : "f"(amax), "r"(0));
        amax = _shfl_1;
        float _fmax_3 = fmaxf(amax, 1e-12f);
        amax = _fmax_3;
        float _fdiv_rn_1 = __fdiv_rn(amax, 448.0f);
        float scale = _fdiv_rn_1;
        if (tid == 0) {
            *(reinterpret_cast<float*>(act_scale + row) + (0)) = scale;
        }
        for (int i_2 = 0; i_2 < 3; i_2++) {
            int chunk_2 = tid + i_2 * 128;
            if (chunk_2 < 336) {
                unsigned int words[4];
                for (int w = 0; w < 4; w++) {
                    float _fdiv_rn_2 = __fdiv_rn(a_vals[i_2 * 16 + w * 4], scale);
                    float q_a = _fdiv_rn_2;
                    float _fdiv_rn_3 = __fdiv_rn(a_vals[i_2 * 16 + w * 4 + 1], scale);
                    float q_b = _fdiv_rn_3;
                    float _fdiv_rn_4 = __fdiv_rn(a_vals[i_2 * 16 + w * 4 + 2], scale);
                    float q_c = _fdiv_rn_4;
                    float _fdiv_rn_5 = __fdiv_rn(a_vals[i_2 * 16 + w * 4 + 3], scale);
                    float q_d = _fdiv_rn_5;
                    uint16_t _e4m3x2_f32_0;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(q_b), "f"(q_a));
                    uint16_t _e4m3x2_f32_1;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(q_d), "f"(q_c));
                    uint32_t _pack_u16x2_0;
                    asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_0) : "h"(_e4m3x2_f32_0), "h"(_e4m3x2_f32_1));
                    words[w] = _pack_u16x2_0;
                }
                reinterpret_cast<int4*>(act_q + ((row * 5376 + chunk_2 * 16) / 4))[0] = reinterpret_cast<int4*>(words)[0];
            }
        }
        __syncthreads();
    }
}

}  // namespace h3_norm_adaln_quant_fp8_sm120a
#undef H3_QPA_INF
#undef NUM_MAIN_STAGES
#undef SMEM_AMAX_PARTIALS_OFF
#undef SMEM_AMAX_PARTIALS_STAGE_BYTES
#undef SMEM_AMAX_PARTIALS_STRIDE
#undef SMEM_PARTIALS_OFF
#undef SMEM_PARTIALS_STAGE_BYTES
#undef SMEM_PARTIALS_STRIDE
#undef SMEM_TOTAL
#undef THREADS

namespace h3_norm_adaln_quant_nvfp4_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_QPA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_PARTIALS_OFF 0
#define SMEM_PARTIALS_STAGE_BYTES 32
#define SMEM_PARTIALS_STRIDE 32
#define SMEM_TOTAL 128
#define THREADS 128

#include <math_constants.h>


__global__ __launch_bounds__(128, 4) void
kernel_h3_norm_adaln_quant_nvfp4(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight, __nv_bfloat16* __restrict__ adaln_scale, __nv_bfloat16* __restrict__ adaln_shift, int* __restrict__ adaln_index, unsigned int* __restrict__ act_q, uint8_t* __restrict__ act_sf, float* __restrict__ act_global_scale, int M, float eps)
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
    float* partials = reinterpret_cast<float*>(smem_raw + 0);
    const int partials_addr = smem + 0;

    // === Task calls (dependency order) ===
    float global_scale = act_global_scale[0];
    #pragma unroll 1
    for (int row = bid; row < M; row += num_bids) {
        float x_vals[64];
        float total = 0.0f;
        for (int i = 0; i < 2; i++) {
            int slot = tid + i * 128;
            for (int e = 0; e < 32; e++) {
                x_vals[i * 32 + e] = 0.0f;
            }
            if (slot < 168) {
                int col = slot * 32;
                for (int q = 0; q < 4; q++) {
                    float _vec_load_0[8];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (row * 5376 + col + q * 8) + 0);
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
                                    : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_0[_pair]));
                            }
                        }
                    }
                    for (int e_1 = 0; e_1 < 8; e_1++) {
                        float xv = _vec_load_0[e_1];
                        x_vals[i * 32 + q * 8 + e_1] = xv;
                        total += xv * xv;
                    }
                }
            }
        }
        for (int stage = 0; stage < 5; stage++) {
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, total, 16 >> stage);
            total += _shfl_xor_0;
        }
        if (lane == 0) {
            partials[warp] = total;
        }
        __syncthreads();
        total = ((lane < 4) ? partials[lane] : 0.0f);
        for (int stage_1 = 0; stage_1 < 2; stage_1++) {
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, total, 2 >> stage_1);
            total += _shfl_xor_1;
        }
        float _shfl_0;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_0) : "f"(total), "r"(0));
        total = _shfl_0;
        float _fdiv_rn_0 = __fdiv_rn(total, 5376.0f);
        float _rsqrt_0 = rsqrtf(_fdiv_rn_0 + eps);
        float rstd = _rsqrt_0;
        int idx = adaln_index[row];
        int valid = ((idx >= 0 && idx < 9) ? 1 : 0);
        int idx_c = ((valid == 1) ? idx : 0);
        for (int i_1 = 0; i_1 < 2; i_1++) {
            int slot_1 = tid + i_1 * 128;
            if (slot_1 < 168) {
                int col_1 = slot_1 * 32;
                unsigned int words[4];
                for (int blk = 0; blk < 2; blk++) {
                    float a_blk[16];
                    float amax = 0.0f;
                    for (int h = 0; h < 2; h++) {
                        float _vec_load_1[8];
                        {
                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(x_norm_weight + (col_1 + (blk * 16 + h * 8)) + 0);
                            uint4 _vld_1[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_1[_blk] = _vptr_1[_blk];
                                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_1[_pair]));
                                }
                            }
                        }
                        float _vec_load_2[8];
                        {
                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(adaln_scale + (idx_c * 5376 + col_1 + (blk * 16 + h * 8)) + 0);
                            uint4 _vld_2[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_2[_blk] = _vptr_2[_blk];
                                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_2[_pair]));
                                }
                            }
                        }
                        float _vec_load_3[8];
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(adaln_shift + (idx_c * 5376 + col_1 + (blk * 16 + h * 8)) + 0);
                            uint4 _vld_3[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_3[_blk] = _vptr_3[_blk];
                                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_3[_pair]));
                                }
                            }
                        }
                        float n_raw[8];
                        float sp1_raw[8];
                        for (int e_2 = 0; e_2 < 8; e_2++) {
                            float wv = _vec_load_1[e_2];
                            float sv = _vec_load_2[e_2];
                            n_raw[e_2] = x_vals[i_1 * 32 + (blk * 16 + h * 8) + e_2] * rstd * wv;
                            sp1_raw[e_2] = sv + 1.0f;
                        }
                        uint32_t n_raw_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(n_raw[_lp*2 + 0], n_raw[_lp*2+1 + 0]));
                            n_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        float n_raw_bf16_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&n_raw_bf16_f32[_pair * 2])[0]), "=f"((&n_raw_bf16_f32[_pair * 2])[1])
                                : "r"(n_raw_bf16[_pair]));
                        }
                        uint32_t sp1_raw_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sp1_raw[_lp*2 + 0], sp1_raw[_lp*2+1 + 0]));
                            sp1_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        float sp1_raw_bf16_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&sp1_raw_bf16_f32[_pair * 2])[0]), "=f"((&sp1_raw_bf16_f32[_pair * 2])[1])
                                : "r"(sp1_raw_bf16[_pair]));
                        }
                        float a_raw[8];
                        for (int e_3 = 0; e_3 < 8; e_3++) {
                            float bv = _vec_load_3[e_3];
                            a_raw[e_3] = bv + n_raw_bf16_f32[e_3] * sp1_raw_bf16_f32[e_3];
                        }
                        uint32_t a_raw_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(a_raw[_lp*2 + 0], a_raw[_lp*2+1 + 0]));
                            a_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        float a_raw_bf16_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&a_raw_bf16_f32[_pair * 2])[0]), "=f"((&a_raw_bf16_f32[_pair * 2])[1])
                                : "r"(a_raw_bf16[_pair]));
                        }
                        for (int e_4 = 0; e_4 < 8; e_4++) {
                            float av = ((valid == 1) ? a_raw_bf16_f32[e_4] : 0.0f);
                            a_blk[h * 8 + e_4] = av;
                            float _fabs_0 = fabsf(av);
                            float _fmax_0 = fmaxf(amax, _fabs_0);
                            amax = _fmax_0;
                        }
                    }
                    float sf_val = amax * 0.16666666666666666f * global_scale;
                    int block_idx = slot_1 * 2 + blk;
                    {
                        unsigned short _sf_pair;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(sf_val));
                        *(reinterpret_cast<unsigned char*>(act_sf + (row * 336 + block_idx)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                    }
                    uint16_t _e4m3x2_f32_0;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(0.0f), "f"(sf_val));
                    uint16_t _e4m3x2_decode_4 = (uint16_t)((unsigned int)_e4m3x2_f32_0 & 0xFFu);
                    uint32_t _f16x2_decode_4;
                    float _fp8_decode_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_4) : "h"(_e4m3x2_decode_4));
                    uint16_t _f16_decode_4 = (uint16_t)_f16x2_decode_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_0) : "h"(_f16_decode_4));
                    float sf_dec = _fp8_decode_0;
                    float _fdiv_rn_1 = __fdiv_rn(global_scale, sf_dec);
                    float out_scale = ((sf_dec != 0.0f) ? _fdiv_rn_1 : 0.0f);
                    for (int e_5 = 0; e_5 < 16; e_5++) {
                        a_blk[e_5] = a_blk[e_5] * out_scale;
                    }
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(words[(blk * 2) + 0]) : "f"(a_blk[0]), "f"(a_blk[1]), "f"(a_blk[2]), "f"(a_blk[3]), "f"(a_blk[4]), "f"(a_blk[5]), "f"(a_blk[6]), "f"(a_blk[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(words[(blk * 2) + 1]) : "f"(a_blk[8]), "f"(a_blk[9]), "f"(a_blk[10]), "f"(a_blk[11]), "f"(a_blk[12]), "f"(a_blk[13]), "f"(a_blk[14]), "f"(a_blk[15]));
                }
                reinterpret_cast<int4*>(act_q + ((row * 2688 + slot_1 * 16) / 4))[0] = reinterpret_cast<int4*>(words)[0];
            }
        }
        __syncthreads();
    }
}

}  // namespace h3_norm_adaln_quant_nvfp4_sm120a
#undef H3_QPA_INF
#undef NUM_MAIN_STAGES
#undef SMEM_PARTIALS_OFF
#undef SMEM_PARTIALS_STAGE_BYTES
#undef SMEM_PARTIALS_STRIDE
#undef SMEM_TOTAL
#undef THREADS
// clang-format on
