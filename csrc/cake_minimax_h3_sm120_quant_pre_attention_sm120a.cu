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
// MiniMax-H3 quantized fused pre-attention for SM120 (GB202: RTX 5090 / RTX PRO 6000 Blackwell),
// generated from the Cake kernel schedules.  Two launches per call:
//   1. norm_adaln_quant_{fp8,nvfp4}: RMSNorm(x) * w, indexed AdaLN (shift + n * (1 + scale)), then
//      per-token E4M3 (scale = amax / 448) or block-16 NVFP4 (FlashInfer fp4_quantize semantics)
//      activation quantization of the BF16 rows.
//   2. qkv_gemm_fused_{fp8,nvfp4}_{bf16,e4m3,nvfp4}: persistent 128x128x128 GEMM (5376 -> 21504,
//      mma.sync e4m3 / kind::mxf4nvf4 block-scaled, 2-stage TMA ring, one CTA per SM) with the fused
//      dequant -> BF16 -> Q/K RMSNorm -> 3-D RoPE epilogue
//      writing Q/K/V [M, 56, 128] as BF16, E4M3 (+ per-tensor scale) or NVFP4 (+ block-16 scales).
// Device code: TMA, ldmatrix, mma.sync (kind::f8f6f4 / kind::mxf4nvf4), mbarrier pipelines.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

static_assert(sizeof(CUtensorMap) == 128, "CUDA tensor-map ABI size mismatch");
static_assert(alignof(CUtensorMap) >= 64, "CUDA tensor-map ABI requires at least 64-byte alignment");

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

namespace h3_qkv_gemm_fp8_bf16_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_QPA_INF CUDART_INF_F
#define NUM_AB_PIPE_STAGES 2
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 16384
#define SMEM_A_STAGE_STRIDE 16384
#define SMEM_B_STAGE_OFF 33792
#define SMEM_B_STAGE_STAGE_BYTES 16384
#define SMEM_B_STAGE_STRIDE 16384
#define SMEM_STAGING_OFF 66560
#define SMEM_STAGING_STAGE_BYTES 16384
#define SMEM_STAGING_STRIDE 16384
#define SMEM_ROWSTAT_OFF 82944
#define SMEM_ROWSTAT_STAGE_BYTES 512
#define SMEM_ROWSTAT_STRIDE 512
#define SMEM_SFA_STAGE_OFF 83456
#define SMEM_SFA_STAGE_STAGE_BYTES 2048
#define SMEM_SFA_STAGE_STRIDE 2048
#define SMEM_SFB_STAGE_OFF 87552
#define SMEM_SFB_STAGE_STAGE_BYTES 2048
#define SMEM_SFB_STAGE_STRIDE 2048
#define SMEM_TOTAL 91648
#define THREADS 288
#define GROUP_M 16

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}


__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__global__ __launch_bounds__(288, 1) void
kernel_h3_qkv_gemm_fused(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ act_scale, float* __restrict__ w_scale, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, unsigned int* __restrict__ q_out, unsigned int* __restrict__ k_out, unsigned int* __restrict__ v_out, uint8_t* __restrict__ q_sf, uint8_t* __restrict__ k_sf, uint8_t* __restrict__ v_sf, int M, int num_m_tiles, int total_tiles, float eps, float alpha, float out_scale_q, float out_scale_k, float out_scale_v, float sf_mul_q, float sf_mul_k, float sf_mul_v)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int B_stage_addr = smem + 33792;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int staging_addr = smem + 66560;
    float* rowstat = reinterpret_cast<float*>(smem_raw + 82944);
    const int rowstat_addr = smem + 82944;
    unsigned int* SFA_stage = reinterpret_cast<unsigned int*>(smem_raw + 83456);
    const int SFA_stage_addr = smem + 83456;
    unsigned int* SFB_stage = reinterpret_cast<unsigned int*>(smem_raw + 87552);
    const int SFB_stage_addr = smem + 87552;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // ab_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 16, 8);
            mbarrier_init(smem + 24, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: producer ----
    if (warp == 0) {
        { // producer_main
            unsigned int load_stage = 0;
            unsigned int _phase_ab_empty = 1;
            #pragma unroll 1
            for (int tile = bid; tile < total_tiles; tile += num_bids) {
                int tile_m = tile / (GROUP_M * 168) * GROUP_M + (tile - tile / (GROUP_M * 168) * (GROUP_M * 168)) % ((num_m_tiles - tile / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 168) * GROUP_M : GROUP_M);
                int tile_n = (tile - tile / (GROUP_M * 168) * (GROUP_M * 168)) / ((num_m_tiles - tile / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 168) * GROUP_M : GROUP_M);
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int k_tile = 0; k_tile < 42; k_tile++) {
                        mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                        mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 32768);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(A_stage_addr + load_stage * 16384), "l"((&A)), "r"(k_tile * 128), "r"(tile_m * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(B_stage_addr + load_stage * 16384), "l"((&B)), "r"(k_tile * 128), "r"(tile_n * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        load_stage += 1;
                        if (load_stage == 2) { load_stage = 0; _phase_ab_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 1 && warp <= 8) {
        { // mma_main
            unsigned int mma_stage = 0;
            int warp_id_in_role = (warp - 1);
            int warp_m = warp_id_in_role % 4;
            int warp_n = warp_id_in_role / 4;
            int role_tid = warp_id_in_role * 32 + lane;
            float accum[64];
            unsigned int a_frag[8];
            unsigned int b_frag[16];
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 168) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 168) * (GROUP_M * 168)) % ((num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 168) * (GROUP_M * 168)) / ((num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M : GROUP_M);
                accum[0] = 0.0f;
                accum[1] = 0.0f;
                accum[2] = 0.0f;
                accum[3] = 0.0f;
                accum[4] = 0.0f;
                accum[5] = 0.0f;
                accum[6] = 0.0f;
                accum[7] = 0.0f;
                accum[8] = 0.0f;
                accum[9] = 0.0f;
                accum[10] = 0.0f;
                accum[11] = 0.0f;
                accum[12] = 0.0f;
                accum[13] = 0.0f;
                accum[14] = 0.0f;
                accum[15] = 0.0f;
                accum[16] = 0.0f;
                accum[17] = 0.0f;
                accum[18] = 0.0f;
                accum[19] = 0.0f;
                accum[20] = 0.0f;
                accum[21] = 0.0f;
                accum[22] = 0.0f;
                accum[23] = 0.0f;
                accum[24] = 0.0f;
                accum[25] = 0.0f;
                accum[26] = 0.0f;
                accum[27] = 0.0f;
                accum[28] = 0.0f;
                accum[29] = 0.0f;
                accum[30] = 0.0f;
                accum[31] = 0.0f;
                accum[32] = 0.0f;
                accum[33] = 0.0f;
                accum[34] = 0.0f;
                accum[35] = 0.0f;
                accum[36] = 0.0f;
                accum[37] = 0.0f;
                accum[38] = 0.0f;
                accum[39] = 0.0f;
                accum[40] = 0.0f;
                accum[41] = 0.0f;
                accum[42] = 0.0f;
                accum[43] = 0.0f;
                accum[44] = 0.0f;
                accum[45] = 0.0f;
                accum[46] = 0.0f;
                accum[47] = 0.0f;
                accum[48] = 0.0f;
                accum[49] = 0.0f;
                accum[50] = 0.0f;
                accum[51] = 0.0f;
                accum[52] = 0.0f;
                accum[53] = 0.0f;
                accum[54] = 0.0f;
                accum[55] = 0.0f;
                accum[56] = 0.0f;
                accum[57] = 0.0f;
                accum[58] = 0.0f;
                accum[59] = 0.0f;
                accum[60] = 0.0f;
                accum[61] = 0.0f;
                accum[62] = 0.0f;
                accum[63] = 0.0f;
                #pragma unroll 1
                for (int k_tile_1 = 0; k_tile_1 < 42; k_tile_1++) {
                    mbarrier_wait(ab_full_addr + (mma_stage) * 8, _phase_ab_full);
                    for (int k_step = 0; k_step < 4; k_step++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 2) { mma_stage = 0; _phase_ab_full ^= 1; }
                }
                int kind = tile_n_1 % 3;
                int head = tile_n_1 / 3;
                int col_base = tile_n_1 * 128 + warp_n * 64 + (lane & 3) * 2;
                float sw[16];
                float nw[16];
                float out_scale_sel = ((kind == 0) ? out_scale_q : ((kind == 1) ? out_scale_k : out_scale_v));
                float sf_mul_sel = ((kind == 0) ? sf_mul_q : ((kind == 1) ? sf_mul_k : sf_mul_v));
                sw[0] = w_scale[col_base];
                sw[1] = w_scale[col_base + 1];
                sw[2] = w_scale[col_base + 8];
                sw[3] = w_scale[col_base + 8 + 1];
                sw[4] = w_scale[col_base + 16];
                sw[5] = w_scale[col_base + 16 + 1];
                sw[6] = w_scale[col_base + 24];
                sw[7] = w_scale[col_base + 24 + 1];
                sw[8] = w_scale[col_base + 32];
                sw[9] = w_scale[col_base + 32 + 1];
                sw[10] = w_scale[col_base + 40];
                sw[11] = w_scale[col_base + 40 + 1];
                sw[12] = w_scale[col_base + 48];
                sw[13] = w_scale[col_base + 48 + 1];
                sw[14] = w_scale[col_base + 56];
                sw[15] = w_scale[col_base + 56 + 1];
                for (int mma_n = 0; mma_n < 8; mma_n++) {
                    int d_col = warp_n * 64 + mma_n * 8 + (lane & 3) * 2;
                    if (kind == 0) {
                        nw[mma_n * 2] = (float)q_norm_weight[d_col];
                        nw[mma_n * 2 + 1] = (float)q_norm_weight[d_col + 1];
                    } else if (kind == 1) {
                        nw[mma_n * 2] = (float)k_norm_weight[d_col];
                        nw[mma_n * 2 + 1] = (float)k_norm_weight[d_col + 1];
                    } else {
                        nw[mma_n * 2] = 1.0f;
                        nw[mma_n * 2 + 1] = 1.0f;
                    }
                }
                for (int half = 0; half < 2; half++) {
                    int row_lo = tile_m_1 * 128 + half * 64 + warp_m * 16 + (lane >> 2);
                    int row_hi = row_lo + 8;
                    float sa_lo = ((row_lo < M) ? act_scale[row_lo] : 0.0f);
                    float sa_hi = ((row_hi < M) ? act_scale[row_hi] : 0.0f);
                    float rounded[32];
                    float ss_lo = 0.0f;
                    float ss_hi = 0.0f;
                    for (int mma_n_1 = 0; mma_n_1 < 8; mma_n_1++) {
                        float scaled[4];
                        scaled[0] = accum[(half * 8 + mma_n_1) * 4] * sa_lo * sw[mma_n_1 * 2];
                        scaled[1] = accum[(half * 8 + mma_n_1) * 4 + 1] * sa_lo * sw[mma_n_1 * 2 + 1];
                        scaled[2] = accum[(half * 8 + mma_n_1) * 4 + 2] * sa_hi * sw[mma_n_1 * 2];
                        scaled[3] = accum[(half * 8 + mma_n_1) * 4 + 3] * sa_hi * sw[mma_n_1 * 2 + 1];
                        uint32_t scaled_bf16[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                            scaled_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        float scaled_bf16_f32[4];
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&scaled_bf16_f32[_pair * 2])[0]), "=f"((&scaled_bf16_f32[_pair * 2])[1])
                                : "r"(scaled_bf16[_pair]));
                        }
                        for (int e = 0; e < 4; e++) {
                            rounded[mma_n_1 * 4 + e] = scaled_bf16_f32[e];
                        }
                        ss_lo += scaled_bf16_f32[0] * scaled_bf16_f32[0] + scaled_bf16_f32[1] * scaled_bf16_f32[1];
                        ss_hi += scaled_bf16_f32[2] * scaled_bf16_f32[2] + scaled_bf16_f32[3] * scaled_bf16_f32[3];
                    }
                    float rstd_lo = 1.0f;
                    float rstd_hi = 1.0f;
                    if (kind < 2) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, ss_lo, 1);
                        ss_lo += _shfl_xor_0;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo, 2);
                        ss_lo += _shfl_xor_1;
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, ss_hi, 1);
                        ss_hi += _shfl_xor_2;
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, ss_hi, 2);
                        ss_hi += _shfl_xor_3;
                        int srow_lo = warp_m * 16 + (lane >> 2);
                        if ((lane & 3) == 0) {
                            rowstat[warp_n * 64 + srow_lo] = ss_lo;
                            rowstat[warp_n * 64 + srow_lo + 8] = ss_hi;
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        float other_lo = rowstat[(1 - warp_n) * 64 + srow_lo];
                        float other_hi = rowstat[(1 - warp_n) * 64 + srow_lo + 8];
                        float _fdiv_rn_0 = __fdiv_rn(ss_lo + other_lo, 128.0f);
                        float _rsqrt_0 = rsqrtf(_fdiv_rn_0 + eps);
                        rstd_lo = _rsqrt_0;
                        float _fdiv_rn_1 = __fdiv_rn(ss_hi + other_hi, 128.0f);
                        float _rsqrt_1 = rsqrtf(_fdiv_rn_1 + eps);
                        rstd_hi = _rsqrt_1;
                    }
                    int srow_st = warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                    for (int mma_n_2 = 0; mma_n_2 < 8; mma_n_2++) {
                        float normed[4];
                        if (kind < 2) {
                            normed[0] = rounded[mma_n_2 * 4] * rstd_lo * nw[mma_n_2 * 2];
                            normed[1] = rounded[mma_n_2 * 4 + 1] * rstd_lo * nw[mma_n_2 * 2 + 1];
                            normed[2] = rounded[mma_n_2 * 4 + 2] * rstd_hi * nw[mma_n_2 * 2];
                            normed[3] = rounded[mma_n_2 * 4 + 3] * rstd_hi * nw[mma_n_2 * 2 + 1];
                        } else {
                            normed[0] = rounded[mma_n_2 * 4];
                            normed[1] = rounded[mma_n_2 * 4 + 1];
                            normed[2] = rounded[mma_n_2 * 4 + 2];
                            normed[3] = rounded[mma_n_2 * 4 + 3];
                        }
                        uint32_t normed_bf16[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed[_lp*2 + 0], normed[_lp*2+1 + 0]));
                            normed_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + mma_n_2 * 8) * 2 ^ (srow_st & 7) << 4));
                        asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16[1]))
                            : "memory");
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                    for (int it = 0; it < 2; it++) {
                        int task = role_tid + it * 256;
                        if (task < 320) {
                            int srow = task / 5;
                            int c = task % 5;
                            int grow = tile_m_1 * 128 + half * 64 + srow;
                            if (grow < M) {
                                int row_head = grow * 56 + head;
                                unsigned int st_words[4];
                                float st_vals[16];
                                if (c < 3) {
                                    unsigned int x_lo[8];
                                    unsigned int x_hi[8];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 16 * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 16 * 2 + 16 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((c * 16 + 48) * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((c * 16 + 48) * 2 + 16 ^ (srow & 7) << 4)));
                                    if (kind < 2) {
                                        float x_lo_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_lo_f32[_pair * 2])[0]), "=f"((&x_lo_f32[_pair * 2])[1])
                                                : "r"(x_lo[_pair]));
                                        }
                                        float x_hi_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_hi_f32[_pair * 2])[0]), "=f"((&x_hi_f32[_pair * 2])[1])
                                                : "r"(x_hi[_pair]));
                                        }
                                        float _vec_load_0[8];
                                        {
                                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + c * 16) + 0);
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
                                                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_1[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_1[8];
                                        {
                                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + c * 16 + 8) + 0);
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
                                                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_2[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_2[8];
                                        {
                                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + 48 + c * 16) + 0);
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
                                                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_3[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_3[8];
                                        {
                                            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + 48 + c * 16 + 8) + 0);
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
                                                        : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_4[_pair]));
                                                }
                                            }
                                        }
                                        float o_lo[16];
                                        float o_hi[16];
                                        for (int e_1 = 0; e_1 < 8; e_1++) {
                                            float cv0 = _vec_load_0[e_1];
                                            float sv0 = _vec_load_2[e_1];
                                            float cv1 = _vec_load_1[e_1];
                                            float sv1 = _vec_load_3[e_1];
                                            o_lo[e_1] = x_lo_f32[e_1] * cv0 - x_hi_f32[e_1] * sv0;
                                            o_hi[e_1] = x_hi_f32[e_1] * cv0 + x_lo_f32[e_1] * sv0;
                                            o_lo[8 + e_1] = x_lo_f32[8 + e_1] * cv1 - x_hi_f32[8 + e_1] * sv1;
                                            o_hi[8 + e_1] = x_hi_f32[8 + e_1] * cv1 + x_lo_f32[8 + e_1] * sv1;
                                        }
                                        uint32_t o_lo_bf16[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_lo[_lp*2 + 0], o_lo[_lp*2+1 + 0]));
                                            o_lo_bf16[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        uint32_t o_hi_bf16[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_hi[_lp*2 + 0], o_hi[_lp*2+1 + 0]));
                                            o_hi_bf16[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        if (kind == 0) {
                                            reinterpret_cast<int4*>(q_out + (row_head * 64 + (c * 16 >> 1)))[0] = reinterpret_cast<int4*>(o_lo_bf16 + 0)[0];
                                            reinterpret_cast<int4*>(q_out + (row_head * 64 + (c * 16 >> 1) + 4))[0] = reinterpret_cast<int4*>(o_lo_bf16 + 4)[0];
                                            reinterpret_cast<int4*>(q_out + (row_head * 64 + (c * 16 + 48 >> 1)))[0] = reinterpret_cast<int4*>(o_hi_bf16 + 0)[0];
                                            reinterpret_cast<int4*>(q_out + (row_head * 64 + (c * 16 + 48 >> 1) + 4))[0] = reinterpret_cast<int4*>(o_hi_bf16 + 4)[0];
                                        } else {
                                            reinterpret_cast<int4*>(k_out + (row_head * 64 + (c * 16 >> 1)))[0] = reinterpret_cast<int4*>(o_lo_bf16 + 0)[0];
                                            reinterpret_cast<int4*>(k_out + (row_head * 64 + (c * 16 >> 1) + 4))[0] = reinterpret_cast<int4*>(o_lo_bf16 + 4)[0];
                                            reinterpret_cast<int4*>(k_out + (row_head * 64 + (c * 16 + 48 >> 1)))[0] = reinterpret_cast<int4*>(o_hi_bf16 + 0)[0];
                                            reinterpret_cast<int4*>(k_out + (row_head * 64 + (c * 16 + 48 >> 1) + 4))[0] = reinterpret_cast<int4*>(o_hi_bf16 + 4)[0];
                                        }
                                    } else {
                                        reinterpret_cast<int4*>(v_out + (row_head * 64 + (c * 16 >> 1)))[0] = reinterpret_cast<int4*>(x_lo + 0)[0];
                                        reinterpret_cast<int4*>(v_out + (row_head * 64 + (c * 16 >> 1) + 4))[0] = reinterpret_cast<int4*>(x_lo + 4)[0];
                                        reinterpret_cast<int4*>(v_out + (row_head * 64 + (c * 16 + 48 >> 1)))[0] = reinterpret_cast<int4*>(x_hi + 0)[0];
                                        reinterpret_cast<int4*>(v_out + (row_head * 64 + (c * 16 + 48 >> 1) + 4))[0] = reinterpret_cast<int4*>(x_hi + 4)[0];
                                    }
                                } else {
                                    unsigned int x_pt[8];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_pt[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((96 + (c - 3) * 16) * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_pt[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((96 + (c - 3) * 16) * 2 + 16 ^ (srow & 7) << 4)));
                                    if (kind == 0) {
                                        reinterpret_cast<int4*>(q_out + (row_head * 64 + (96 + (c - 3) * 16 >> 1)))[0] = reinterpret_cast<int4*>(x_pt + 0)[0];
                                        reinterpret_cast<int4*>(q_out + (row_head * 64 + (96 + (c - 3) * 16 >> 1) + 4))[0] = reinterpret_cast<int4*>(x_pt + 4)[0];
                                    } else if (kind == 1) {
                                        reinterpret_cast<int4*>(k_out + (row_head * 64 + (96 + (c - 3) * 16 >> 1)))[0] = reinterpret_cast<int4*>(x_pt + 0)[0];
                                        reinterpret_cast<int4*>(k_out + (row_head * 64 + (96 + (c - 3) * 16 >> 1) + 4))[0] = reinterpret_cast<int4*>(x_pt + 4)[0];
                                    } else {
                                        reinterpret_cast<int4*>(v_out + (row_head * 64 + (96 + (c - 3) * 16 >> 1)))[0] = reinterpret_cast<int4*>(x_pt + 0)[0];
                                        reinterpret_cast<int4*>(v_out + (row_head * 64 + (96 + (c - 3) * 16 >> 1) + 4))[0] = reinterpret_cast<int4*>(x_pt + 4)[0];
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                }
            }
        }
    }

    // Cleanup
}

}  // namespace h3_qkv_gemm_fp8_bf16_sm120a
#undef GROUP_M
#undef H3_QPA_INF
#undef NUM_AB_PIPE_STAGES
#undef SMEM_A_STAGE_OFF
#undef SMEM_A_STAGE_STAGE_BYTES
#undef SMEM_A_STAGE_STRIDE
#undef SMEM_B_STAGE_OFF
#undef SMEM_B_STAGE_STAGE_BYTES
#undef SMEM_B_STAGE_STRIDE
#undef SMEM_ROWSTAT_OFF
#undef SMEM_ROWSTAT_STAGE_BYTES
#undef SMEM_ROWSTAT_STRIDE
#undef SMEM_SFA_STAGE_OFF
#undef SMEM_SFA_STAGE_STAGE_BYTES
#undef SMEM_SFA_STAGE_STRIDE
#undef SMEM_SFB_STAGE_OFF
#undef SMEM_SFB_STAGE_STAGE_BYTES
#undef SMEM_SFB_STAGE_STRIDE
#undef SMEM_STAGING_OFF
#undef SMEM_STAGING_STAGE_BYTES
#undef SMEM_STAGING_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef ab_empty_addr
#undef ab_full_addr

namespace h3_qkv_gemm_fp8_e4m3_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_QPA_INF CUDART_INF_F
#define NUM_AB_PIPE_STAGES 2
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 16384
#define SMEM_A_STAGE_STRIDE 16384
#define SMEM_B_STAGE_OFF 33792
#define SMEM_B_STAGE_STAGE_BYTES 16384
#define SMEM_B_STAGE_STRIDE 16384
#define SMEM_STAGING_OFF 66560
#define SMEM_STAGING_STAGE_BYTES 16384
#define SMEM_STAGING_STRIDE 16384
#define SMEM_ROWSTAT_OFF 82944
#define SMEM_ROWSTAT_STAGE_BYTES 512
#define SMEM_ROWSTAT_STRIDE 512
#define SMEM_SFA_STAGE_OFF 83456
#define SMEM_SFA_STAGE_STAGE_BYTES 2048
#define SMEM_SFA_STAGE_STRIDE 2048
#define SMEM_SFB_STAGE_OFF 87552
#define SMEM_SFB_STAGE_STAGE_BYTES 2048
#define SMEM_SFB_STAGE_STRIDE 2048
#define SMEM_TOTAL 91648
#define THREADS 288
#define GROUP_M 16

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}


__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__global__ __launch_bounds__(288, 1) void
kernel_h3_qkv_gemm_fused(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ act_scale, float* __restrict__ w_scale, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, unsigned int* __restrict__ q_out, unsigned int* __restrict__ k_out, unsigned int* __restrict__ v_out, uint8_t* __restrict__ q_sf, uint8_t* __restrict__ k_sf, uint8_t* __restrict__ v_sf, int M, int num_m_tiles, int total_tiles, float eps, float alpha, float out_scale_q, float out_scale_k, float out_scale_v, float sf_mul_q, float sf_mul_k, float sf_mul_v)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int B_stage_addr = smem + 33792;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int staging_addr = smem + 66560;
    float* rowstat = reinterpret_cast<float*>(smem_raw + 82944);
    const int rowstat_addr = smem + 82944;
    unsigned int* SFA_stage = reinterpret_cast<unsigned int*>(smem_raw + 83456);
    const int SFA_stage_addr = smem + 83456;
    unsigned int* SFB_stage = reinterpret_cast<unsigned int*>(smem_raw + 87552);
    const int SFB_stage_addr = smem + 87552;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // ab_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 16, 8);
            mbarrier_init(smem + 24, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: producer ----
    if (warp == 0) {
        { // producer_main
            unsigned int load_stage = 0;
            unsigned int _phase_ab_empty = 1;
            #pragma unroll 1
            for (int tile = bid; tile < total_tiles; tile += num_bids) {
                int tile_m = tile / (GROUP_M * 168) * GROUP_M + (tile - tile / (GROUP_M * 168) * (GROUP_M * 168)) % ((num_m_tiles - tile / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 168) * GROUP_M : GROUP_M);
                int tile_n = (tile - tile / (GROUP_M * 168) * (GROUP_M * 168)) / ((num_m_tiles - tile / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 168) * GROUP_M : GROUP_M);
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int k_tile = 0; k_tile < 42; k_tile++) {
                        mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                        mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 32768);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(A_stage_addr + load_stage * 16384), "l"((&A)), "r"(k_tile * 128), "r"(tile_m * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(B_stage_addr + load_stage * 16384), "l"((&B)), "r"(k_tile * 128), "r"(tile_n * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        load_stage += 1;
                        if (load_stage == 2) { load_stage = 0; _phase_ab_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 1 && warp <= 8) {
        { // mma_main
            unsigned int mma_stage = 0;
            int warp_id_in_role = (warp - 1);
            int warp_m = warp_id_in_role % 4;
            int warp_n = warp_id_in_role / 4;
            int role_tid = warp_id_in_role * 32 + lane;
            float accum[64];
            unsigned int a_frag[8];
            unsigned int b_frag[16];
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 168) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 168) * (GROUP_M * 168)) % ((num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 168) * (GROUP_M * 168)) / ((num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M : GROUP_M);
                accum[0] = 0.0f;
                accum[1] = 0.0f;
                accum[2] = 0.0f;
                accum[3] = 0.0f;
                accum[4] = 0.0f;
                accum[5] = 0.0f;
                accum[6] = 0.0f;
                accum[7] = 0.0f;
                accum[8] = 0.0f;
                accum[9] = 0.0f;
                accum[10] = 0.0f;
                accum[11] = 0.0f;
                accum[12] = 0.0f;
                accum[13] = 0.0f;
                accum[14] = 0.0f;
                accum[15] = 0.0f;
                accum[16] = 0.0f;
                accum[17] = 0.0f;
                accum[18] = 0.0f;
                accum[19] = 0.0f;
                accum[20] = 0.0f;
                accum[21] = 0.0f;
                accum[22] = 0.0f;
                accum[23] = 0.0f;
                accum[24] = 0.0f;
                accum[25] = 0.0f;
                accum[26] = 0.0f;
                accum[27] = 0.0f;
                accum[28] = 0.0f;
                accum[29] = 0.0f;
                accum[30] = 0.0f;
                accum[31] = 0.0f;
                accum[32] = 0.0f;
                accum[33] = 0.0f;
                accum[34] = 0.0f;
                accum[35] = 0.0f;
                accum[36] = 0.0f;
                accum[37] = 0.0f;
                accum[38] = 0.0f;
                accum[39] = 0.0f;
                accum[40] = 0.0f;
                accum[41] = 0.0f;
                accum[42] = 0.0f;
                accum[43] = 0.0f;
                accum[44] = 0.0f;
                accum[45] = 0.0f;
                accum[46] = 0.0f;
                accum[47] = 0.0f;
                accum[48] = 0.0f;
                accum[49] = 0.0f;
                accum[50] = 0.0f;
                accum[51] = 0.0f;
                accum[52] = 0.0f;
                accum[53] = 0.0f;
                accum[54] = 0.0f;
                accum[55] = 0.0f;
                accum[56] = 0.0f;
                accum[57] = 0.0f;
                accum[58] = 0.0f;
                accum[59] = 0.0f;
                accum[60] = 0.0f;
                accum[61] = 0.0f;
                accum[62] = 0.0f;
                accum[63] = 0.0f;
                #pragma unroll 1
                for (int k_tile_1 = 0; k_tile_1 < 42; k_tile_1++) {
                    mbarrier_wait(ab_full_addr + (mma_stage) * 8, _phase_ab_full);
                    for (int k_step = 0; k_step < 4; k_step++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 2) { mma_stage = 0; _phase_ab_full ^= 1; }
                }
                int kind = tile_n_1 % 3;
                int head = tile_n_1 / 3;
                int col_base = tile_n_1 * 128 + warp_n * 64 + (lane & 3) * 2;
                float sw[16];
                float nw[16];
                float out_scale_sel = ((kind == 0) ? out_scale_q : ((kind == 1) ? out_scale_k : out_scale_v));
                float sf_mul_sel = ((kind == 0) ? sf_mul_q : ((kind == 1) ? sf_mul_k : sf_mul_v));
                sw[0] = w_scale[col_base];
                sw[1] = w_scale[col_base + 1];
                sw[2] = w_scale[col_base + 8];
                sw[3] = w_scale[col_base + 8 + 1];
                sw[4] = w_scale[col_base + 16];
                sw[5] = w_scale[col_base + 16 + 1];
                sw[6] = w_scale[col_base + 24];
                sw[7] = w_scale[col_base + 24 + 1];
                sw[8] = w_scale[col_base + 32];
                sw[9] = w_scale[col_base + 32 + 1];
                sw[10] = w_scale[col_base + 40];
                sw[11] = w_scale[col_base + 40 + 1];
                sw[12] = w_scale[col_base + 48];
                sw[13] = w_scale[col_base + 48 + 1];
                sw[14] = w_scale[col_base + 56];
                sw[15] = w_scale[col_base + 56 + 1];
                for (int mma_n = 0; mma_n < 8; mma_n++) {
                    int d_col = warp_n * 64 + mma_n * 8 + (lane & 3) * 2;
                    if (kind == 0) {
                        nw[mma_n * 2] = (float)q_norm_weight[d_col];
                        nw[mma_n * 2 + 1] = (float)q_norm_weight[d_col + 1];
                    } else if (kind == 1) {
                        nw[mma_n * 2] = (float)k_norm_weight[d_col];
                        nw[mma_n * 2 + 1] = (float)k_norm_weight[d_col + 1];
                    } else {
                        nw[mma_n * 2] = 1.0f;
                        nw[mma_n * 2 + 1] = 1.0f;
                    }
                }
                for (int half = 0; half < 2; half++) {
                    int row_lo = tile_m_1 * 128 + half * 64 + warp_m * 16 + (lane >> 2);
                    int row_hi = row_lo + 8;
                    float sa_lo = ((row_lo < M) ? act_scale[row_lo] : 0.0f);
                    float sa_hi = ((row_hi < M) ? act_scale[row_hi] : 0.0f);
                    float rounded[32];
                    float ss_lo = 0.0f;
                    float ss_hi = 0.0f;
                    for (int mma_n_1 = 0; mma_n_1 < 8; mma_n_1++) {
                        float scaled[4];
                        scaled[0] = accum[(half * 8 + mma_n_1) * 4] * sa_lo * sw[mma_n_1 * 2];
                        scaled[1] = accum[(half * 8 + mma_n_1) * 4 + 1] * sa_lo * sw[mma_n_1 * 2 + 1];
                        scaled[2] = accum[(half * 8 + mma_n_1) * 4 + 2] * sa_hi * sw[mma_n_1 * 2];
                        scaled[3] = accum[(half * 8 + mma_n_1) * 4 + 3] * sa_hi * sw[mma_n_1 * 2 + 1];
                        uint32_t scaled_bf16[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                            scaled_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        float scaled_bf16_f32[4];
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&scaled_bf16_f32[_pair * 2])[0]), "=f"((&scaled_bf16_f32[_pair * 2])[1])
                                : "r"(scaled_bf16[_pair]));
                        }
                        for (int e = 0; e < 4; e++) {
                            rounded[mma_n_1 * 4 + e] = scaled_bf16_f32[e];
                        }
                        ss_lo += scaled_bf16_f32[0] * scaled_bf16_f32[0] + scaled_bf16_f32[1] * scaled_bf16_f32[1];
                        ss_hi += scaled_bf16_f32[2] * scaled_bf16_f32[2] + scaled_bf16_f32[3] * scaled_bf16_f32[3];
                    }
                    float rstd_lo = 1.0f;
                    float rstd_hi = 1.0f;
                    if (kind < 2) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, ss_lo, 1);
                        ss_lo += _shfl_xor_0;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo, 2);
                        ss_lo += _shfl_xor_1;
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, ss_hi, 1);
                        ss_hi += _shfl_xor_2;
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, ss_hi, 2);
                        ss_hi += _shfl_xor_3;
                        int srow_lo = warp_m * 16 + (lane >> 2);
                        if ((lane & 3) == 0) {
                            rowstat[warp_n * 64 + srow_lo] = ss_lo;
                            rowstat[warp_n * 64 + srow_lo + 8] = ss_hi;
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        float other_lo = rowstat[(1 - warp_n) * 64 + srow_lo];
                        float other_hi = rowstat[(1 - warp_n) * 64 + srow_lo + 8];
                        float _fdiv_rn_0 = __fdiv_rn(ss_lo + other_lo, 128.0f);
                        float _rsqrt_0 = rsqrtf(_fdiv_rn_0 + eps);
                        rstd_lo = _rsqrt_0;
                        float _fdiv_rn_1 = __fdiv_rn(ss_hi + other_hi, 128.0f);
                        float _rsqrt_1 = rsqrtf(_fdiv_rn_1 + eps);
                        rstd_hi = _rsqrt_1;
                    }
                    int srow_st = warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                    for (int mma_n_2 = 0; mma_n_2 < 8; mma_n_2++) {
                        float normed[4];
                        if (kind < 2) {
                            normed[0] = rounded[mma_n_2 * 4] * rstd_lo * nw[mma_n_2 * 2];
                            normed[1] = rounded[mma_n_2 * 4 + 1] * rstd_lo * nw[mma_n_2 * 2 + 1];
                            normed[2] = rounded[mma_n_2 * 4 + 2] * rstd_hi * nw[mma_n_2 * 2];
                            normed[3] = rounded[mma_n_2 * 4 + 3] * rstd_hi * nw[mma_n_2 * 2 + 1];
                        } else {
                            normed[0] = rounded[mma_n_2 * 4];
                            normed[1] = rounded[mma_n_2 * 4 + 1];
                            normed[2] = rounded[mma_n_2 * 4 + 2];
                            normed[3] = rounded[mma_n_2 * 4 + 3];
                        }
                        uint32_t normed_bf16[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed[_lp*2 + 0], normed[_lp*2+1 + 0]));
                            normed_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + mma_n_2 * 8) * 2 ^ (srow_st & 7) << 4));
                        asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16[1]))
                            : "memory");
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                    for (int it = 0; it < 2; it++) {
                        int task = role_tid + it * 256;
                        if (task < 320) {
                            int srow = task / 5;
                            int c = task % 5;
                            int grow = tile_m_1 * 128 + half * 64 + srow;
                            if (grow < M) {
                                int row_head = grow * 56 + head;
                                unsigned int st_words[4];
                                float st_vals[16];
                                if (c < 3) {
                                    unsigned int x_lo[8];
                                    unsigned int x_hi[8];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 16 * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 16 * 2 + 16 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((c * 16 + 48) * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((c * 16 + 48) * 2 + 16 ^ (srow & 7) << 4)));
                                    if (kind < 2) {
                                        float x_lo_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_lo_f32[_pair * 2])[0]), "=f"((&x_lo_f32[_pair * 2])[1])
                                                : "r"(x_lo[_pair]));
                                        }
                                        float x_hi_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_hi_f32[_pair * 2])[0]), "=f"((&x_hi_f32[_pair * 2])[1])
                                                : "r"(x_hi[_pair]));
                                        }
                                        float _vec_load_0[8];
                                        {
                                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + c * 16) + 0);
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
                                                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_1[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_1[8];
                                        {
                                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + c * 16 + 8) + 0);
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
                                                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_2[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_2[8];
                                        {
                                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + 48 + c * 16) + 0);
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
                                                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_3[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_3[8];
                                        {
                                            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + 48 + c * 16 + 8) + 0);
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
                                                        : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_4[_pair]));
                                                }
                                            }
                                        }
                                        float o_lo[16];
                                        float o_hi[16];
                                        for (int e_1 = 0; e_1 < 8; e_1++) {
                                            float cv0 = _vec_load_0[e_1];
                                            float sv0 = _vec_load_2[e_1];
                                            float cv1 = _vec_load_1[e_1];
                                            float sv1 = _vec_load_3[e_1];
                                            o_lo[e_1] = x_lo_f32[e_1] * cv0 - x_hi_f32[e_1] * sv0;
                                            o_hi[e_1] = x_hi_f32[e_1] * cv0 + x_lo_f32[e_1] * sv0;
                                            o_lo[8 + e_1] = x_lo_f32[8 + e_1] * cv1 - x_hi_f32[8 + e_1] * sv1;
                                            o_hi[8 + e_1] = x_hi_f32[8 + e_1] * cv1 + x_lo_f32[8 + e_1] * sv1;
                                        }
                                        uint32_t o_lo_bf16[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_lo[_lp*2 + 0], o_lo[_lp*2+1 + 0]));
                                            o_lo_bf16[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        uint32_t o_hi_bf16[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_hi[_lp*2 + 0], o_hi[_lp*2+1 + 0]));
                                            o_hi_bf16[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        if (kind == 0) {
                                            float o_lo_bf16_f32[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_lo_bf16_f32[_pair * 2])[0]), "=f"((&o_lo_bf16_f32[_pair * 2])[1])
                                                    : "r"(o_lo_bf16[_pair]));
                                            }
                                            st_vals[0] = o_lo_bf16_f32[0] * out_scale_sel;
                                            st_vals[1] = o_lo_bf16_f32[1] * out_scale_sel;
                                            st_vals[2] = o_lo_bf16_f32[2] * out_scale_sel;
                                            st_vals[3] = o_lo_bf16_f32[3] * out_scale_sel;
                                            st_vals[4] = o_lo_bf16_f32[4] * out_scale_sel;
                                            st_vals[5] = o_lo_bf16_f32[5] * out_scale_sel;
                                            st_vals[6] = o_lo_bf16_f32[6] * out_scale_sel;
                                            st_vals[7] = o_lo_bf16_f32[7] * out_scale_sel;
                                            st_vals[8] = o_lo_bf16_f32[8] * out_scale_sel;
                                            st_vals[9] = o_lo_bf16_f32[9] * out_scale_sel;
                                            st_vals[10] = o_lo_bf16_f32[10] * out_scale_sel;
                                            st_vals[11] = o_lo_bf16_f32[11] * out_scale_sel;
                                            st_vals[12] = o_lo_bf16_f32[12] * out_scale_sel;
                                            st_vals[13] = o_lo_bf16_f32[13] * out_scale_sel;
                                            st_vals[14] = o_lo_bf16_f32[14] * out_scale_sel;
                                            st_vals[15] = o_lo_bf16_f32[15] * out_scale_sel;
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                       "f"(st_vals[2]), "f"(st_vals[3]));
                                                st_words[0] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                       "f"(st_vals[6]), "f"(st_vals[7]));
                                                st_words[1] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                       "f"(st_vals[10]), "f"(st_vals[11]));
                                                st_words[2] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                       "f"(st_vals[14]), "f"(st_vals[15]));
                                                st_words[3] = _packed;
                                            }
                                            reinterpret_cast<int4*>(q_out + (row_head * 32 + (c * 16 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                            float o_hi_bf16_f32[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_hi_bf16_f32[_pair * 2])[0]), "=f"((&o_hi_bf16_f32[_pair * 2])[1])
                                                    : "r"(o_hi_bf16[_pair]));
                                            }
                                            st_vals[0] = o_hi_bf16_f32[0] * out_scale_sel;
                                            st_vals[1] = o_hi_bf16_f32[1] * out_scale_sel;
                                            st_vals[2] = o_hi_bf16_f32[2] * out_scale_sel;
                                            st_vals[3] = o_hi_bf16_f32[3] * out_scale_sel;
                                            st_vals[4] = o_hi_bf16_f32[4] * out_scale_sel;
                                            st_vals[5] = o_hi_bf16_f32[5] * out_scale_sel;
                                            st_vals[6] = o_hi_bf16_f32[6] * out_scale_sel;
                                            st_vals[7] = o_hi_bf16_f32[7] * out_scale_sel;
                                            st_vals[8] = o_hi_bf16_f32[8] * out_scale_sel;
                                            st_vals[9] = o_hi_bf16_f32[9] * out_scale_sel;
                                            st_vals[10] = o_hi_bf16_f32[10] * out_scale_sel;
                                            st_vals[11] = o_hi_bf16_f32[11] * out_scale_sel;
                                            st_vals[12] = o_hi_bf16_f32[12] * out_scale_sel;
                                            st_vals[13] = o_hi_bf16_f32[13] * out_scale_sel;
                                            st_vals[14] = o_hi_bf16_f32[14] * out_scale_sel;
                                            st_vals[15] = o_hi_bf16_f32[15] * out_scale_sel;
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                       "f"(st_vals[2]), "f"(st_vals[3]));
                                                st_words[0] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                       "f"(st_vals[6]), "f"(st_vals[7]));
                                                st_words[1] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                       "f"(st_vals[10]), "f"(st_vals[11]));
                                                st_words[2] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                       "f"(st_vals[14]), "f"(st_vals[15]));
                                                st_words[3] = _packed;
                                            }
                                            reinterpret_cast<int4*>(q_out + (row_head * 32 + (c * 16 + 48 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                        } else {
                                            float o_lo_bf16_f32_1[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_lo_bf16_f32_1[_pair * 2])[0]), "=f"((&o_lo_bf16_f32_1[_pair * 2])[1])
                                                    : "r"(o_lo_bf16[_pair]));
                                            }
                                            st_vals[0] = o_lo_bf16_f32_1[0] * out_scale_sel;
                                            st_vals[1] = o_lo_bf16_f32_1[1] * out_scale_sel;
                                            st_vals[2] = o_lo_bf16_f32_1[2] * out_scale_sel;
                                            st_vals[3] = o_lo_bf16_f32_1[3] * out_scale_sel;
                                            st_vals[4] = o_lo_bf16_f32_1[4] * out_scale_sel;
                                            st_vals[5] = o_lo_bf16_f32_1[5] * out_scale_sel;
                                            st_vals[6] = o_lo_bf16_f32_1[6] * out_scale_sel;
                                            st_vals[7] = o_lo_bf16_f32_1[7] * out_scale_sel;
                                            st_vals[8] = o_lo_bf16_f32_1[8] * out_scale_sel;
                                            st_vals[9] = o_lo_bf16_f32_1[9] * out_scale_sel;
                                            st_vals[10] = o_lo_bf16_f32_1[10] * out_scale_sel;
                                            st_vals[11] = o_lo_bf16_f32_1[11] * out_scale_sel;
                                            st_vals[12] = o_lo_bf16_f32_1[12] * out_scale_sel;
                                            st_vals[13] = o_lo_bf16_f32_1[13] * out_scale_sel;
                                            st_vals[14] = o_lo_bf16_f32_1[14] * out_scale_sel;
                                            st_vals[15] = o_lo_bf16_f32_1[15] * out_scale_sel;
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                       "f"(st_vals[2]), "f"(st_vals[3]));
                                                st_words[0] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                       "f"(st_vals[6]), "f"(st_vals[7]));
                                                st_words[1] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                       "f"(st_vals[10]), "f"(st_vals[11]));
                                                st_words[2] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                       "f"(st_vals[14]), "f"(st_vals[15]));
                                                st_words[3] = _packed;
                                            }
                                            reinterpret_cast<int4*>(k_out + (row_head * 32 + (c * 16 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                            float o_hi_bf16_f32_1[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_hi_bf16_f32_1[_pair * 2])[0]), "=f"((&o_hi_bf16_f32_1[_pair * 2])[1])
                                                    : "r"(o_hi_bf16[_pair]));
                                            }
                                            st_vals[0] = o_hi_bf16_f32_1[0] * out_scale_sel;
                                            st_vals[1] = o_hi_bf16_f32_1[1] * out_scale_sel;
                                            st_vals[2] = o_hi_bf16_f32_1[2] * out_scale_sel;
                                            st_vals[3] = o_hi_bf16_f32_1[3] * out_scale_sel;
                                            st_vals[4] = o_hi_bf16_f32_1[4] * out_scale_sel;
                                            st_vals[5] = o_hi_bf16_f32_1[5] * out_scale_sel;
                                            st_vals[6] = o_hi_bf16_f32_1[6] * out_scale_sel;
                                            st_vals[7] = o_hi_bf16_f32_1[7] * out_scale_sel;
                                            st_vals[8] = o_hi_bf16_f32_1[8] * out_scale_sel;
                                            st_vals[9] = o_hi_bf16_f32_1[9] * out_scale_sel;
                                            st_vals[10] = o_hi_bf16_f32_1[10] * out_scale_sel;
                                            st_vals[11] = o_hi_bf16_f32_1[11] * out_scale_sel;
                                            st_vals[12] = o_hi_bf16_f32_1[12] * out_scale_sel;
                                            st_vals[13] = o_hi_bf16_f32_1[13] * out_scale_sel;
                                            st_vals[14] = o_hi_bf16_f32_1[14] * out_scale_sel;
                                            st_vals[15] = o_hi_bf16_f32_1[15] * out_scale_sel;
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                       "f"(st_vals[2]), "f"(st_vals[3]));
                                                st_words[0] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                       "f"(st_vals[6]), "f"(st_vals[7]));
                                                st_words[1] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                       "f"(st_vals[10]), "f"(st_vals[11]));
                                                st_words[2] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                       "f"(st_vals[14]), "f"(st_vals[15]));
                                                st_words[3] = _packed;
                                            }
                                            reinterpret_cast<int4*>(k_out + (row_head * 32 + (c * 16 + 48 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                        }
                                    } else {
                                        float x_lo_f32_1[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_lo_f32_1[_pair * 2])[0]), "=f"((&x_lo_f32_1[_pair * 2])[1])
                                                : "r"(x_lo[_pair]));
                                        }
                                        st_vals[0] = x_lo_f32_1[0] * out_scale_sel;
                                        st_vals[1] = x_lo_f32_1[1] * out_scale_sel;
                                        st_vals[2] = x_lo_f32_1[2] * out_scale_sel;
                                        st_vals[3] = x_lo_f32_1[3] * out_scale_sel;
                                        st_vals[4] = x_lo_f32_1[4] * out_scale_sel;
                                        st_vals[5] = x_lo_f32_1[5] * out_scale_sel;
                                        st_vals[6] = x_lo_f32_1[6] * out_scale_sel;
                                        st_vals[7] = x_lo_f32_1[7] * out_scale_sel;
                                        st_vals[8] = x_lo_f32_1[8] * out_scale_sel;
                                        st_vals[9] = x_lo_f32_1[9] * out_scale_sel;
                                        st_vals[10] = x_lo_f32_1[10] * out_scale_sel;
                                        st_vals[11] = x_lo_f32_1[11] * out_scale_sel;
                                        st_vals[12] = x_lo_f32_1[12] * out_scale_sel;
                                        st_vals[13] = x_lo_f32_1[13] * out_scale_sel;
                                        st_vals[14] = x_lo_f32_1[14] * out_scale_sel;
                                        st_vals[15] = x_lo_f32_1[15] * out_scale_sel;
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                   "f"(st_vals[2]), "f"(st_vals[3]));
                                            st_words[0] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                   "f"(st_vals[6]), "f"(st_vals[7]));
                                            st_words[1] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                   "f"(st_vals[10]), "f"(st_vals[11]));
                                            st_words[2] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                   "f"(st_vals[14]), "f"(st_vals[15]));
                                            st_words[3] = _packed;
                                        }
                                        reinterpret_cast<int4*>(v_out + (row_head * 32 + (c * 16 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                        float x_hi_f32_1[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_hi_f32_1[_pair * 2])[0]), "=f"((&x_hi_f32_1[_pair * 2])[1])
                                                : "r"(x_hi[_pair]));
                                        }
                                        st_vals[0] = x_hi_f32_1[0] * out_scale_sel;
                                        st_vals[1] = x_hi_f32_1[1] * out_scale_sel;
                                        st_vals[2] = x_hi_f32_1[2] * out_scale_sel;
                                        st_vals[3] = x_hi_f32_1[3] * out_scale_sel;
                                        st_vals[4] = x_hi_f32_1[4] * out_scale_sel;
                                        st_vals[5] = x_hi_f32_1[5] * out_scale_sel;
                                        st_vals[6] = x_hi_f32_1[6] * out_scale_sel;
                                        st_vals[7] = x_hi_f32_1[7] * out_scale_sel;
                                        st_vals[8] = x_hi_f32_1[8] * out_scale_sel;
                                        st_vals[9] = x_hi_f32_1[9] * out_scale_sel;
                                        st_vals[10] = x_hi_f32_1[10] * out_scale_sel;
                                        st_vals[11] = x_hi_f32_1[11] * out_scale_sel;
                                        st_vals[12] = x_hi_f32_1[12] * out_scale_sel;
                                        st_vals[13] = x_hi_f32_1[13] * out_scale_sel;
                                        st_vals[14] = x_hi_f32_1[14] * out_scale_sel;
                                        st_vals[15] = x_hi_f32_1[15] * out_scale_sel;
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                   "f"(st_vals[2]), "f"(st_vals[3]));
                                            st_words[0] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                   "f"(st_vals[6]), "f"(st_vals[7]));
                                            st_words[1] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                   "f"(st_vals[10]), "f"(st_vals[11]));
                                            st_words[2] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                   "f"(st_vals[14]), "f"(st_vals[15]));
                                            st_words[3] = _packed;
                                        }
                                        reinterpret_cast<int4*>(v_out + (row_head * 32 + (c * 16 + 48 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                    }
                                } else {
                                    unsigned int x_pt[8];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_pt[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((96 + (c - 3) * 16) * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_pt[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((96 + (c - 3) * 16) * 2 + 16 ^ (srow & 7) << 4)));
                                    if (kind == 0) {
                                        float x_pt_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_pt_f32[_pair * 2])[0]), "=f"((&x_pt_f32[_pair * 2])[1])
                                                : "r"(x_pt[_pair]));
                                        }
                                        st_vals[0] = x_pt_f32[0] * out_scale_sel;
                                        st_vals[1] = x_pt_f32[1] * out_scale_sel;
                                        st_vals[2] = x_pt_f32[2] * out_scale_sel;
                                        st_vals[3] = x_pt_f32[3] * out_scale_sel;
                                        st_vals[4] = x_pt_f32[4] * out_scale_sel;
                                        st_vals[5] = x_pt_f32[5] * out_scale_sel;
                                        st_vals[6] = x_pt_f32[6] * out_scale_sel;
                                        st_vals[7] = x_pt_f32[7] * out_scale_sel;
                                        st_vals[8] = x_pt_f32[8] * out_scale_sel;
                                        st_vals[9] = x_pt_f32[9] * out_scale_sel;
                                        st_vals[10] = x_pt_f32[10] * out_scale_sel;
                                        st_vals[11] = x_pt_f32[11] * out_scale_sel;
                                        st_vals[12] = x_pt_f32[12] * out_scale_sel;
                                        st_vals[13] = x_pt_f32[13] * out_scale_sel;
                                        st_vals[14] = x_pt_f32[14] * out_scale_sel;
                                        st_vals[15] = x_pt_f32[15] * out_scale_sel;
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                   "f"(st_vals[2]), "f"(st_vals[3]));
                                            st_words[0] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                   "f"(st_vals[6]), "f"(st_vals[7]));
                                            st_words[1] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                   "f"(st_vals[10]), "f"(st_vals[11]));
                                            st_words[2] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                   "f"(st_vals[14]), "f"(st_vals[15]));
                                            st_words[3] = _packed;
                                        }
                                        reinterpret_cast<int4*>(q_out + (row_head * 32 + (96 + (c - 3) * 16 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                    } else if (kind == 1) {
                                        float x_pt_f32_1[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_pt_f32_1[_pair * 2])[0]), "=f"((&x_pt_f32_1[_pair * 2])[1])
                                                : "r"(x_pt[_pair]));
                                        }
                                        st_vals[0] = x_pt_f32_1[0] * out_scale_sel;
                                        st_vals[1] = x_pt_f32_1[1] * out_scale_sel;
                                        st_vals[2] = x_pt_f32_1[2] * out_scale_sel;
                                        st_vals[3] = x_pt_f32_1[3] * out_scale_sel;
                                        st_vals[4] = x_pt_f32_1[4] * out_scale_sel;
                                        st_vals[5] = x_pt_f32_1[5] * out_scale_sel;
                                        st_vals[6] = x_pt_f32_1[6] * out_scale_sel;
                                        st_vals[7] = x_pt_f32_1[7] * out_scale_sel;
                                        st_vals[8] = x_pt_f32_1[8] * out_scale_sel;
                                        st_vals[9] = x_pt_f32_1[9] * out_scale_sel;
                                        st_vals[10] = x_pt_f32_1[10] * out_scale_sel;
                                        st_vals[11] = x_pt_f32_1[11] * out_scale_sel;
                                        st_vals[12] = x_pt_f32_1[12] * out_scale_sel;
                                        st_vals[13] = x_pt_f32_1[13] * out_scale_sel;
                                        st_vals[14] = x_pt_f32_1[14] * out_scale_sel;
                                        st_vals[15] = x_pt_f32_1[15] * out_scale_sel;
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                   "f"(st_vals[2]), "f"(st_vals[3]));
                                            st_words[0] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                   "f"(st_vals[6]), "f"(st_vals[7]));
                                            st_words[1] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                   "f"(st_vals[10]), "f"(st_vals[11]));
                                            st_words[2] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                   "f"(st_vals[14]), "f"(st_vals[15]));
                                            st_words[3] = _packed;
                                        }
                                        reinterpret_cast<int4*>(k_out + (row_head * 32 + (96 + (c - 3) * 16 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                    } else {
                                        float x_pt_f32_2[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_pt_f32_2[_pair * 2])[0]), "=f"((&x_pt_f32_2[_pair * 2])[1])
                                                : "r"(x_pt[_pair]));
                                        }
                                        st_vals[0] = x_pt_f32_2[0] * out_scale_sel;
                                        st_vals[1] = x_pt_f32_2[1] * out_scale_sel;
                                        st_vals[2] = x_pt_f32_2[2] * out_scale_sel;
                                        st_vals[3] = x_pt_f32_2[3] * out_scale_sel;
                                        st_vals[4] = x_pt_f32_2[4] * out_scale_sel;
                                        st_vals[5] = x_pt_f32_2[5] * out_scale_sel;
                                        st_vals[6] = x_pt_f32_2[6] * out_scale_sel;
                                        st_vals[7] = x_pt_f32_2[7] * out_scale_sel;
                                        st_vals[8] = x_pt_f32_2[8] * out_scale_sel;
                                        st_vals[9] = x_pt_f32_2[9] * out_scale_sel;
                                        st_vals[10] = x_pt_f32_2[10] * out_scale_sel;
                                        st_vals[11] = x_pt_f32_2[11] * out_scale_sel;
                                        st_vals[12] = x_pt_f32_2[12] * out_scale_sel;
                                        st_vals[13] = x_pt_f32_2[13] * out_scale_sel;
                                        st_vals[14] = x_pt_f32_2[14] * out_scale_sel;
                                        st_vals[15] = x_pt_f32_2[15] * out_scale_sel;
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                   "f"(st_vals[2]), "f"(st_vals[3]));
                                            st_words[0] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                   "f"(st_vals[6]), "f"(st_vals[7]));
                                            st_words[1] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                   "f"(st_vals[10]), "f"(st_vals[11]));
                                            st_words[2] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                   "f"(st_vals[14]), "f"(st_vals[15]));
                                            st_words[3] = _packed;
                                        }
                                        reinterpret_cast<int4*>(v_out + (row_head * 32 + (96 + (c - 3) * 16 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                }
            }
        }
    }

    // Cleanup
}

}  // namespace h3_qkv_gemm_fp8_e4m3_sm120a
#undef GROUP_M
#undef H3_QPA_INF
#undef NUM_AB_PIPE_STAGES
#undef SMEM_A_STAGE_OFF
#undef SMEM_A_STAGE_STAGE_BYTES
#undef SMEM_A_STAGE_STRIDE
#undef SMEM_B_STAGE_OFF
#undef SMEM_B_STAGE_STAGE_BYTES
#undef SMEM_B_STAGE_STRIDE
#undef SMEM_ROWSTAT_OFF
#undef SMEM_ROWSTAT_STAGE_BYTES
#undef SMEM_ROWSTAT_STRIDE
#undef SMEM_SFA_STAGE_OFF
#undef SMEM_SFA_STAGE_STAGE_BYTES
#undef SMEM_SFA_STAGE_STRIDE
#undef SMEM_SFB_STAGE_OFF
#undef SMEM_SFB_STAGE_STAGE_BYTES
#undef SMEM_SFB_STAGE_STRIDE
#undef SMEM_STAGING_OFF
#undef SMEM_STAGING_STAGE_BYTES
#undef SMEM_STAGING_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef ab_empty_addr
#undef ab_full_addr

namespace h3_qkv_gemm_fp8_nvfp4_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_QPA_INF CUDART_INF_F
#define NUM_AB_PIPE_STAGES 2
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 16384
#define SMEM_A_STAGE_STRIDE 16384
#define SMEM_B_STAGE_OFF 33792
#define SMEM_B_STAGE_STAGE_BYTES 16384
#define SMEM_B_STAGE_STRIDE 16384
#define SMEM_STAGING_OFF 66560
#define SMEM_STAGING_STAGE_BYTES 16384
#define SMEM_STAGING_STRIDE 16384
#define SMEM_ROWSTAT_OFF 82944
#define SMEM_ROWSTAT_STAGE_BYTES 512
#define SMEM_ROWSTAT_STRIDE 512
#define SMEM_SFA_STAGE_OFF 83456
#define SMEM_SFA_STAGE_STAGE_BYTES 2048
#define SMEM_SFA_STAGE_STRIDE 2048
#define SMEM_SFB_STAGE_OFF 87552
#define SMEM_SFB_STAGE_STAGE_BYTES 2048
#define SMEM_SFB_STAGE_STRIDE 2048
#define SMEM_TOTAL 91648
#define THREADS 288
#define GROUP_M 16

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}


__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__global__ __launch_bounds__(288, 1) void
kernel_h3_qkv_gemm_fused(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ act_scale, float* __restrict__ w_scale, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, unsigned int* __restrict__ q_out, unsigned int* __restrict__ k_out, unsigned int* __restrict__ v_out, uint8_t* __restrict__ q_sf, uint8_t* __restrict__ k_sf, uint8_t* __restrict__ v_sf, int M, int num_m_tiles, int total_tiles, float eps, float alpha, float out_scale_q, float out_scale_k, float out_scale_v, float sf_mul_q, float sf_mul_k, float sf_mul_v)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int B_stage_addr = smem + 33792;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int staging_addr = smem + 66560;
    float* rowstat = reinterpret_cast<float*>(smem_raw + 82944);
    const int rowstat_addr = smem + 82944;
    unsigned int* SFA_stage = reinterpret_cast<unsigned int*>(smem_raw + 83456);
    const int SFA_stage_addr = smem + 83456;
    unsigned int* SFB_stage = reinterpret_cast<unsigned int*>(smem_raw + 87552);
    const int SFB_stage_addr = smem + 87552;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // ab_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 16, 8);
            mbarrier_init(smem + 24, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: producer ----
    if (warp == 0) {
        { // producer_main
            unsigned int load_stage = 0;
            unsigned int _phase_ab_empty = 1;
            #pragma unroll 1
            for (int tile = bid; tile < total_tiles; tile += num_bids) {
                int tile_m = tile / (GROUP_M * 168) * GROUP_M + (tile - tile / (GROUP_M * 168) * (GROUP_M * 168)) % ((num_m_tiles - tile / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 168) * GROUP_M : GROUP_M);
                int tile_n = (tile - tile / (GROUP_M * 168) * (GROUP_M * 168)) / ((num_m_tiles - tile / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 168) * GROUP_M : GROUP_M);
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int k_tile = 0; k_tile < 42; k_tile++) {
                        mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                        mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 32768);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(A_stage_addr + load_stage * 16384), "l"((&A)), "r"(k_tile * 128), "r"(tile_m * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(B_stage_addr + load_stage * 16384), "l"((&B)), "r"(k_tile * 128), "r"(tile_n * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        load_stage += 1;
                        if (load_stage == 2) { load_stage = 0; _phase_ab_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 1 && warp <= 8) {
        { // mma_main
            unsigned int mma_stage = 0;
            int warp_id_in_role = (warp - 1);
            int warp_m = warp_id_in_role % 4;
            int warp_n = warp_id_in_role / 4;
            int role_tid = warp_id_in_role * 32 + lane;
            float accum[64];
            unsigned int a_frag[8];
            unsigned int b_frag[16];
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 168) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 168) * (GROUP_M * 168)) % ((num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 168) * (GROUP_M * 168)) / ((num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M : GROUP_M);
                accum[0] = 0.0f;
                accum[1] = 0.0f;
                accum[2] = 0.0f;
                accum[3] = 0.0f;
                accum[4] = 0.0f;
                accum[5] = 0.0f;
                accum[6] = 0.0f;
                accum[7] = 0.0f;
                accum[8] = 0.0f;
                accum[9] = 0.0f;
                accum[10] = 0.0f;
                accum[11] = 0.0f;
                accum[12] = 0.0f;
                accum[13] = 0.0f;
                accum[14] = 0.0f;
                accum[15] = 0.0f;
                accum[16] = 0.0f;
                accum[17] = 0.0f;
                accum[18] = 0.0f;
                accum[19] = 0.0f;
                accum[20] = 0.0f;
                accum[21] = 0.0f;
                accum[22] = 0.0f;
                accum[23] = 0.0f;
                accum[24] = 0.0f;
                accum[25] = 0.0f;
                accum[26] = 0.0f;
                accum[27] = 0.0f;
                accum[28] = 0.0f;
                accum[29] = 0.0f;
                accum[30] = 0.0f;
                accum[31] = 0.0f;
                accum[32] = 0.0f;
                accum[33] = 0.0f;
                accum[34] = 0.0f;
                accum[35] = 0.0f;
                accum[36] = 0.0f;
                accum[37] = 0.0f;
                accum[38] = 0.0f;
                accum[39] = 0.0f;
                accum[40] = 0.0f;
                accum[41] = 0.0f;
                accum[42] = 0.0f;
                accum[43] = 0.0f;
                accum[44] = 0.0f;
                accum[45] = 0.0f;
                accum[46] = 0.0f;
                accum[47] = 0.0f;
                accum[48] = 0.0f;
                accum[49] = 0.0f;
                accum[50] = 0.0f;
                accum[51] = 0.0f;
                accum[52] = 0.0f;
                accum[53] = 0.0f;
                accum[54] = 0.0f;
                accum[55] = 0.0f;
                accum[56] = 0.0f;
                accum[57] = 0.0f;
                accum[58] = 0.0f;
                accum[59] = 0.0f;
                accum[60] = 0.0f;
                accum[61] = 0.0f;
                accum[62] = 0.0f;
                accum[63] = 0.0f;
                #pragma unroll 1
                for (int k_tile_1 = 0; k_tile_1 < 42; k_tile_1++) {
                    mbarrier_wait(ab_full_addr + (mma_stage) * 8, _phase_ab_full);
                    for (int k_step = 0; k_step < 4; k_step++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 2) { mma_stage = 0; _phase_ab_full ^= 1; }
                }
                int kind = tile_n_1 % 3;
                int head = tile_n_1 / 3;
                int col_base = tile_n_1 * 128 + warp_n * 64 + (lane & 3) * 2;
                float sw[16];
                float nw[16];
                float out_scale_sel = ((kind == 0) ? out_scale_q : ((kind == 1) ? out_scale_k : out_scale_v));
                float sf_mul_sel = ((kind == 0) ? sf_mul_q : ((kind == 1) ? sf_mul_k : sf_mul_v));
                sw[0] = w_scale[col_base];
                sw[1] = w_scale[col_base + 1];
                sw[2] = w_scale[col_base + 8];
                sw[3] = w_scale[col_base + 8 + 1];
                sw[4] = w_scale[col_base + 16];
                sw[5] = w_scale[col_base + 16 + 1];
                sw[6] = w_scale[col_base + 24];
                sw[7] = w_scale[col_base + 24 + 1];
                sw[8] = w_scale[col_base + 32];
                sw[9] = w_scale[col_base + 32 + 1];
                sw[10] = w_scale[col_base + 40];
                sw[11] = w_scale[col_base + 40 + 1];
                sw[12] = w_scale[col_base + 48];
                sw[13] = w_scale[col_base + 48 + 1];
                sw[14] = w_scale[col_base + 56];
                sw[15] = w_scale[col_base + 56 + 1];
                for (int mma_n = 0; mma_n < 8; mma_n++) {
                    int d_col = warp_n * 64 + mma_n * 8 + (lane & 3) * 2;
                    if (kind == 0) {
                        nw[mma_n * 2] = (float)q_norm_weight[d_col];
                        nw[mma_n * 2 + 1] = (float)q_norm_weight[d_col + 1];
                    } else if (kind == 1) {
                        nw[mma_n * 2] = (float)k_norm_weight[d_col];
                        nw[mma_n * 2 + 1] = (float)k_norm_weight[d_col + 1];
                    } else {
                        nw[mma_n * 2] = 1.0f;
                        nw[mma_n * 2 + 1] = 1.0f;
                    }
                }
                for (int half = 0; half < 2; half++) {
                    int row_lo = tile_m_1 * 128 + half * 64 + warp_m * 16 + (lane >> 2);
                    int row_hi = row_lo + 8;
                    float sa_lo = ((row_lo < M) ? act_scale[row_lo] : 0.0f);
                    float sa_hi = ((row_hi < M) ? act_scale[row_hi] : 0.0f);
                    float rounded[32];
                    float ss_lo = 0.0f;
                    float ss_hi = 0.0f;
                    for (int mma_n_1 = 0; mma_n_1 < 8; mma_n_1++) {
                        float scaled[4];
                        scaled[0] = accum[(half * 8 + mma_n_1) * 4] * sa_lo * sw[mma_n_1 * 2];
                        scaled[1] = accum[(half * 8 + mma_n_1) * 4 + 1] * sa_lo * sw[mma_n_1 * 2 + 1];
                        scaled[2] = accum[(half * 8 + mma_n_1) * 4 + 2] * sa_hi * sw[mma_n_1 * 2];
                        scaled[3] = accum[(half * 8 + mma_n_1) * 4 + 3] * sa_hi * sw[mma_n_1 * 2 + 1];
                        uint32_t scaled_bf16[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                            scaled_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        float scaled_bf16_f32[4];
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&scaled_bf16_f32[_pair * 2])[0]), "=f"((&scaled_bf16_f32[_pair * 2])[1])
                                : "r"(scaled_bf16[_pair]));
                        }
                        for (int e = 0; e < 4; e++) {
                            rounded[mma_n_1 * 4 + e] = scaled_bf16_f32[e];
                        }
                        ss_lo += scaled_bf16_f32[0] * scaled_bf16_f32[0] + scaled_bf16_f32[1] * scaled_bf16_f32[1];
                        ss_hi += scaled_bf16_f32[2] * scaled_bf16_f32[2] + scaled_bf16_f32[3] * scaled_bf16_f32[3];
                    }
                    float rstd_lo = 1.0f;
                    float rstd_hi = 1.0f;
                    if (kind < 2) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, ss_lo, 1);
                        ss_lo += _shfl_xor_0;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo, 2);
                        ss_lo += _shfl_xor_1;
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, ss_hi, 1);
                        ss_hi += _shfl_xor_2;
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, ss_hi, 2);
                        ss_hi += _shfl_xor_3;
                        int srow_lo = warp_m * 16 + (lane >> 2);
                        if ((lane & 3) == 0) {
                            rowstat[warp_n * 64 + srow_lo] = ss_lo;
                            rowstat[warp_n * 64 + srow_lo + 8] = ss_hi;
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        float other_lo = rowstat[(1 - warp_n) * 64 + srow_lo];
                        float other_hi = rowstat[(1 - warp_n) * 64 + srow_lo + 8];
                        float _fdiv_rn_0 = __fdiv_rn(ss_lo + other_lo, 128.0f);
                        float _rsqrt_0 = rsqrtf(_fdiv_rn_0 + eps);
                        rstd_lo = _rsqrt_0;
                        float _fdiv_rn_1 = __fdiv_rn(ss_hi + other_hi, 128.0f);
                        float _rsqrt_1 = rsqrtf(_fdiv_rn_1 + eps);
                        rstd_hi = _rsqrt_1;
                    }
                    int srow_st = warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                    for (int mma_n_2 = 0; mma_n_2 < 8; mma_n_2++) {
                        float normed[4];
                        if (kind < 2) {
                            normed[0] = rounded[mma_n_2 * 4] * rstd_lo * nw[mma_n_2 * 2];
                            normed[1] = rounded[mma_n_2 * 4 + 1] * rstd_lo * nw[mma_n_2 * 2 + 1];
                            normed[2] = rounded[mma_n_2 * 4 + 2] * rstd_hi * nw[mma_n_2 * 2];
                            normed[3] = rounded[mma_n_2 * 4 + 3] * rstd_hi * nw[mma_n_2 * 2 + 1];
                        } else {
                            normed[0] = rounded[mma_n_2 * 4];
                            normed[1] = rounded[mma_n_2 * 4 + 1];
                            normed[2] = rounded[mma_n_2 * 4 + 2];
                            normed[3] = rounded[mma_n_2 * 4 + 3];
                        }
                        uint32_t normed_bf16[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed[_lp*2 + 0], normed[_lp*2+1 + 0]));
                            normed_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + mma_n_2 * 8) * 2 ^ (srow_st & 7) << 4));
                        asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16[1]))
                            : "memory");
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                    for (int it = 0; it < 2; it++) {
                        int task = role_tid + it * 256;
                        if (task < 320) {
                            int srow = task / 5;
                            int c = task % 5;
                            int grow = tile_m_1 * 128 + half * 64 + srow;
                            if (grow < M) {
                                int row_head = grow * 56 + head;
                                unsigned int st_words[4];
                                float st_vals[16];
                                if (c < 3) {
                                    unsigned int x_lo[8];
                                    unsigned int x_hi[8];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 16 * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 16 * 2 + 16 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((c * 16 + 48) * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((c * 16 + 48) * 2 + 16 ^ (srow & 7) << 4)));
                                    if (kind < 2) {
                                        float x_lo_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_lo_f32[_pair * 2])[0]), "=f"((&x_lo_f32[_pair * 2])[1])
                                                : "r"(x_lo[_pair]));
                                        }
                                        float x_hi_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_hi_f32[_pair * 2])[0]), "=f"((&x_hi_f32[_pair * 2])[1])
                                                : "r"(x_hi[_pair]));
                                        }
                                        float _vec_load_0[8];
                                        {
                                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + c * 16) + 0);
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
                                                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_1[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_1[8];
                                        {
                                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + c * 16 + 8) + 0);
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
                                                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_2[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_2[8];
                                        {
                                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + 48 + c * 16) + 0);
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
                                                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_3[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_3[8];
                                        {
                                            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + 48 + c * 16 + 8) + 0);
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
                                                        : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_4[_pair]));
                                                }
                                            }
                                        }
                                        float o_lo[16];
                                        float o_hi[16];
                                        for (int e_1 = 0; e_1 < 8; e_1++) {
                                            float cv0 = _vec_load_0[e_1];
                                            float sv0 = _vec_load_2[e_1];
                                            float cv1 = _vec_load_1[e_1];
                                            float sv1 = _vec_load_3[e_1];
                                            o_lo[e_1] = x_lo_f32[e_1] * cv0 - x_hi_f32[e_1] * sv0;
                                            o_hi[e_1] = x_hi_f32[e_1] * cv0 + x_lo_f32[e_1] * sv0;
                                            o_lo[8 + e_1] = x_lo_f32[8 + e_1] * cv1 - x_hi_f32[8 + e_1] * sv1;
                                            o_hi[8 + e_1] = x_hi_f32[8 + e_1] * cv1 + x_lo_f32[8 + e_1] * sv1;
                                        }
                                        uint32_t o_lo_bf16[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_lo[_lp*2 + 0], o_lo[_lp*2+1 + 0]));
                                            o_lo_bf16[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        uint32_t o_hi_bf16[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_hi[_lp*2 + 0], o_hi[_lp*2+1 + 0]));
                                            o_hi_bf16[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        if (kind == 0) {
                                            float o_lo_bf16_f32[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_lo_bf16_f32[_pair * 2])[0]), "=f"((&o_lo_bf16_f32[_pair * 2])[1])
                                                    : "r"(o_lo_bf16[_pair]));
                                            }
                                            float _fabs_0 = fabsf(o_lo_bf16_f32[0]);
                                            float _fabs_1 = fabsf(o_lo_bf16_f32[1]);
                                            float _fmax_0 = fmaxf(_fabs_0, _fabs_1);
                                            float _fabs_2 = fabsf(o_lo_bf16_f32[2]);
                                            float _fmax_1 = fmaxf(_fmax_0, _fabs_2);
                                            float _fabs_3 = fabsf(o_lo_bf16_f32[3]);
                                            float _fmax_2 = fmaxf(_fmax_1, _fabs_3);
                                            float _fabs_4 = fabsf(o_lo_bf16_f32[4]);
                                            float _fmax_3 = fmaxf(_fmax_2, _fabs_4);
                                            float _fabs_5 = fabsf(o_lo_bf16_f32[5]);
                                            float _fmax_4 = fmaxf(_fmax_3, _fabs_5);
                                            float _fabs_6 = fabsf(o_lo_bf16_f32[6]);
                                            float _fmax_5 = fmaxf(_fmax_4, _fabs_6);
                                            float _fabs_7 = fabsf(o_lo_bf16_f32[7]);
                                            float _fmax_6 = fmaxf(_fmax_5, _fabs_7);
                                            float _fabs_8 = fabsf(o_lo_bf16_f32[8]);
                                            float _fmax_7 = fmaxf(_fmax_6, _fabs_8);
                                            float _fabs_9 = fabsf(o_lo_bf16_f32[9]);
                                            float _fmax_8 = fmaxf(_fmax_7, _fabs_9);
                                            float _fabs_10 = fabsf(o_lo_bf16_f32[10]);
                                            float _fmax_9 = fmaxf(_fmax_8, _fabs_10);
                                            float _fabs_11 = fabsf(o_lo_bf16_f32[11]);
                                            float _fmax_10 = fmaxf(_fmax_9, _fabs_11);
                                            float _fabs_12 = fabsf(o_lo_bf16_f32[12]);
                                            float _fmax_11 = fmaxf(_fmax_10, _fabs_12);
                                            float _fabs_13 = fabsf(o_lo_bf16_f32[13]);
                                            float _fmax_12 = fmaxf(_fmax_11, _fabs_13);
                                            float _fabs_14 = fabsf(o_lo_bf16_f32[14]);
                                            float _fmax_13 = fmaxf(_fmax_12, _fabs_14);
                                            float _fabs_15 = fabsf(o_lo_bf16_f32[15]);
                                            float _fmax_14 = fmaxf(_fmax_13, _fabs_15);
                                            {
                                                unsigned short _sf_pair;
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_14 * sf_mul_sel));
                                                *(reinterpret_cast<unsigned char*>(q_sf + (row_head * 8 + (c * 16 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                            }
                                            uint16_t _e4m3x2_f32_0;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(0.0f), "f"(_fmax_14 * sf_mul_sel));
                                            uint16_t _e4m3x2_decode_5 = (uint16_t)((unsigned int)_e4m3x2_f32_0 & 0xFFu);
                                            uint32_t _f16x2_decode_5;
                                            float _fp8_decode_0;
                                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_5) : "h"(_e4m3x2_decode_5));
                                            uint16_t _f16_decode_5 = (uint16_t)_f16x2_decode_5;
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_0) : "h"(_f16_decode_5));
                                            float _fdiv_rn_2 = __fdiv_rn(out_scale_sel, _fp8_decode_0);
                                            st_vals[0] = o_lo_bf16_f32[0] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[1] = o_lo_bf16_f32[1] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[2] = o_lo_bf16_f32[2] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[3] = o_lo_bf16_f32[3] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[4] = o_lo_bf16_f32[4] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[5] = o_lo_bf16_f32[5] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[6] = o_lo_bf16_f32[6] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[7] = o_lo_bf16_f32[7] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[8] = o_lo_bf16_f32[8] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[9] = o_lo_bf16_f32[9] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[10] = o_lo_bf16_f32[10] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[11] = o_lo_bf16_f32[11] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[12] = o_lo_bf16_f32[12] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[13] = o_lo_bf16_f32[13] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[14] = o_lo_bf16_f32[14] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[15] = o_lo_bf16_f32[15] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                            *(reinterpret_cast<unsigned int*>(q_out + (row_head * 16 + (c * 16 >> 3))) + (0)) = st_words[0];
                                            *(reinterpret_cast<unsigned int*>(q_out + (row_head * 16 + (c * 16 >> 3) + 1)) + (0)) = st_words[1];
                                            float o_hi_bf16_f32[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_hi_bf16_f32[_pair * 2])[0]), "=f"((&o_hi_bf16_f32[_pair * 2])[1])
                                                    : "r"(o_hi_bf16[_pair]));
                                            }
                                            float _fabs_16 = fabsf(o_hi_bf16_f32[0]);
                                            float _fabs_17 = fabsf(o_hi_bf16_f32[1]);
                                            float _fmax_15 = fmaxf(_fabs_16, _fabs_17);
                                            float _fabs_18 = fabsf(o_hi_bf16_f32[2]);
                                            float _fmax_16 = fmaxf(_fmax_15, _fabs_18);
                                            float _fabs_19 = fabsf(o_hi_bf16_f32[3]);
                                            float _fmax_17 = fmaxf(_fmax_16, _fabs_19);
                                            float _fabs_20 = fabsf(o_hi_bf16_f32[4]);
                                            float _fmax_18 = fmaxf(_fmax_17, _fabs_20);
                                            float _fabs_21 = fabsf(o_hi_bf16_f32[5]);
                                            float _fmax_19 = fmaxf(_fmax_18, _fabs_21);
                                            float _fabs_22 = fabsf(o_hi_bf16_f32[6]);
                                            float _fmax_20 = fmaxf(_fmax_19, _fabs_22);
                                            float _fabs_23 = fabsf(o_hi_bf16_f32[7]);
                                            float _fmax_21 = fmaxf(_fmax_20, _fabs_23);
                                            float _fabs_24 = fabsf(o_hi_bf16_f32[8]);
                                            float _fmax_22 = fmaxf(_fmax_21, _fabs_24);
                                            float _fabs_25 = fabsf(o_hi_bf16_f32[9]);
                                            float _fmax_23 = fmaxf(_fmax_22, _fabs_25);
                                            float _fabs_26 = fabsf(o_hi_bf16_f32[10]);
                                            float _fmax_24 = fmaxf(_fmax_23, _fabs_26);
                                            float _fabs_27 = fabsf(o_hi_bf16_f32[11]);
                                            float _fmax_25 = fmaxf(_fmax_24, _fabs_27);
                                            float _fabs_28 = fabsf(o_hi_bf16_f32[12]);
                                            float _fmax_26 = fmaxf(_fmax_25, _fabs_28);
                                            float _fabs_29 = fabsf(o_hi_bf16_f32[13]);
                                            float _fmax_27 = fmaxf(_fmax_26, _fabs_29);
                                            float _fabs_30 = fabsf(o_hi_bf16_f32[14]);
                                            float _fmax_28 = fmaxf(_fmax_27, _fabs_30);
                                            float _fabs_31 = fabsf(o_hi_bf16_f32[15]);
                                            float _fmax_29 = fmaxf(_fmax_28, _fabs_31);
                                            {
                                                unsigned short _sf_pair;
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_29 * sf_mul_sel));
                                                *(reinterpret_cast<unsigned char*>(q_sf + (row_head * 8 + (c * 16 + 48 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                            }
                                            uint16_t _e4m3x2_f32_1;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(0.0f), "f"(_fmax_29 * sf_mul_sel));
                                            uint16_t _e4m3x2_decode_6 = (uint16_t)((unsigned int)_e4m3x2_f32_1 & 0xFFu);
                                            uint32_t _f16x2_decode_6;
                                            float _fp8_decode_1;
                                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_6) : "h"(_e4m3x2_decode_6));
                                            uint16_t _f16_decode_6 = (uint16_t)_f16x2_decode_6;
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_1) : "h"(_f16_decode_6));
                                            float _fdiv_rn_3 = __fdiv_rn(out_scale_sel, _fp8_decode_1);
                                            st_vals[0] = o_hi_bf16_f32[0] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[1] = o_hi_bf16_f32[1] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[2] = o_hi_bf16_f32[2] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[3] = o_hi_bf16_f32[3] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[4] = o_hi_bf16_f32[4] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[5] = o_hi_bf16_f32[5] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[6] = o_hi_bf16_f32[6] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[7] = o_hi_bf16_f32[7] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[8] = o_hi_bf16_f32[8] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[9] = o_hi_bf16_f32[9] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[10] = o_hi_bf16_f32[10] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[11] = o_hi_bf16_f32[11] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[12] = o_hi_bf16_f32[12] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[13] = o_hi_bf16_f32[13] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[14] = o_hi_bf16_f32[14] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[15] = o_hi_bf16_f32[15] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                            *(reinterpret_cast<unsigned int*>(q_out + (row_head * 16 + (c * 16 + 48 >> 3))) + (0)) = st_words[0];
                                            *(reinterpret_cast<unsigned int*>(q_out + (row_head * 16 + (c * 16 + 48 >> 3) + 1)) + (0)) = st_words[1];
                                        } else {
                                            float o_lo_bf16_f32_1[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_lo_bf16_f32_1[_pair * 2])[0]), "=f"((&o_lo_bf16_f32_1[_pair * 2])[1])
                                                    : "r"(o_lo_bf16[_pair]));
                                            }
                                            float _fabs_32 = fabsf(o_lo_bf16_f32_1[0]);
                                            float _fabs_33 = fabsf(o_lo_bf16_f32_1[1]);
                                            float _fmax_30 = fmaxf(_fabs_32, _fabs_33);
                                            float _fabs_34 = fabsf(o_lo_bf16_f32_1[2]);
                                            float _fmax_31 = fmaxf(_fmax_30, _fabs_34);
                                            float _fabs_35 = fabsf(o_lo_bf16_f32_1[3]);
                                            float _fmax_32 = fmaxf(_fmax_31, _fabs_35);
                                            float _fabs_36 = fabsf(o_lo_bf16_f32_1[4]);
                                            float _fmax_33 = fmaxf(_fmax_32, _fabs_36);
                                            float _fabs_37 = fabsf(o_lo_bf16_f32_1[5]);
                                            float _fmax_34 = fmaxf(_fmax_33, _fabs_37);
                                            float _fabs_38 = fabsf(o_lo_bf16_f32_1[6]);
                                            float _fmax_35 = fmaxf(_fmax_34, _fabs_38);
                                            float _fabs_39 = fabsf(o_lo_bf16_f32_1[7]);
                                            float _fmax_36 = fmaxf(_fmax_35, _fabs_39);
                                            float _fabs_40 = fabsf(o_lo_bf16_f32_1[8]);
                                            float _fmax_37 = fmaxf(_fmax_36, _fabs_40);
                                            float _fabs_41 = fabsf(o_lo_bf16_f32_1[9]);
                                            float _fmax_38 = fmaxf(_fmax_37, _fabs_41);
                                            float _fabs_42 = fabsf(o_lo_bf16_f32_1[10]);
                                            float _fmax_39 = fmaxf(_fmax_38, _fabs_42);
                                            float _fabs_43 = fabsf(o_lo_bf16_f32_1[11]);
                                            float _fmax_40 = fmaxf(_fmax_39, _fabs_43);
                                            float _fabs_44 = fabsf(o_lo_bf16_f32_1[12]);
                                            float _fmax_41 = fmaxf(_fmax_40, _fabs_44);
                                            float _fabs_45 = fabsf(o_lo_bf16_f32_1[13]);
                                            float _fmax_42 = fmaxf(_fmax_41, _fabs_45);
                                            float _fabs_46 = fabsf(o_lo_bf16_f32_1[14]);
                                            float _fmax_43 = fmaxf(_fmax_42, _fabs_46);
                                            float _fabs_47 = fabsf(o_lo_bf16_f32_1[15]);
                                            float _fmax_44 = fmaxf(_fmax_43, _fabs_47);
                                            {
                                                unsigned short _sf_pair;
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_44 * sf_mul_sel));
                                                *(reinterpret_cast<unsigned char*>(k_sf + (row_head * 8 + (c * 16 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                            }
                                            uint16_t _e4m3x2_f32_2;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(0.0f), "f"(_fmax_44 * sf_mul_sel));
                                            uint16_t _e4m3x2_decode_7 = (uint16_t)((unsigned int)_e4m3x2_f32_2 & 0xFFu);
                                            uint32_t _f16x2_decode_7;
                                            float _fp8_decode_2;
                                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_7) : "h"(_e4m3x2_decode_7));
                                            uint16_t _f16_decode_7 = (uint16_t)_f16x2_decode_7;
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_2) : "h"(_f16_decode_7));
                                            float _fdiv_rn_4 = __fdiv_rn(out_scale_sel, _fp8_decode_2);
                                            st_vals[0] = o_lo_bf16_f32_1[0] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[1] = o_lo_bf16_f32_1[1] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[2] = o_lo_bf16_f32_1[2] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[3] = o_lo_bf16_f32_1[3] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[4] = o_lo_bf16_f32_1[4] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[5] = o_lo_bf16_f32_1[5] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[6] = o_lo_bf16_f32_1[6] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[7] = o_lo_bf16_f32_1[7] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[8] = o_lo_bf16_f32_1[8] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[9] = o_lo_bf16_f32_1[9] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[10] = o_lo_bf16_f32_1[10] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[11] = o_lo_bf16_f32_1[11] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[12] = o_lo_bf16_f32_1[12] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[13] = o_lo_bf16_f32_1[13] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[14] = o_lo_bf16_f32_1[14] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[15] = o_lo_bf16_f32_1[15] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                            *(reinterpret_cast<unsigned int*>(k_out + (row_head * 16 + (c * 16 >> 3))) + (0)) = st_words[0];
                                            *(reinterpret_cast<unsigned int*>(k_out + (row_head * 16 + (c * 16 >> 3) + 1)) + (0)) = st_words[1];
                                            float o_hi_bf16_f32_1[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_hi_bf16_f32_1[_pair * 2])[0]), "=f"((&o_hi_bf16_f32_1[_pair * 2])[1])
                                                    : "r"(o_hi_bf16[_pair]));
                                            }
                                            float _fabs_48 = fabsf(o_hi_bf16_f32_1[0]);
                                            float _fabs_49 = fabsf(o_hi_bf16_f32_1[1]);
                                            float _fmax_45 = fmaxf(_fabs_48, _fabs_49);
                                            float _fabs_50 = fabsf(o_hi_bf16_f32_1[2]);
                                            float _fmax_46 = fmaxf(_fmax_45, _fabs_50);
                                            float _fabs_51 = fabsf(o_hi_bf16_f32_1[3]);
                                            float _fmax_47 = fmaxf(_fmax_46, _fabs_51);
                                            float _fabs_52 = fabsf(o_hi_bf16_f32_1[4]);
                                            float _fmax_48 = fmaxf(_fmax_47, _fabs_52);
                                            float _fabs_53 = fabsf(o_hi_bf16_f32_1[5]);
                                            float _fmax_49 = fmaxf(_fmax_48, _fabs_53);
                                            float _fabs_54 = fabsf(o_hi_bf16_f32_1[6]);
                                            float _fmax_50 = fmaxf(_fmax_49, _fabs_54);
                                            float _fabs_55 = fabsf(o_hi_bf16_f32_1[7]);
                                            float _fmax_51 = fmaxf(_fmax_50, _fabs_55);
                                            float _fabs_56 = fabsf(o_hi_bf16_f32_1[8]);
                                            float _fmax_52 = fmaxf(_fmax_51, _fabs_56);
                                            float _fabs_57 = fabsf(o_hi_bf16_f32_1[9]);
                                            float _fmax_53 = fmaxf(_fmax_52, _fabs_57);
                                            float _fabs_58 = fabsf(o_hi_bf16_f32_1[10]);
                                            float _fmax_54 = fmaxf(_fmax_53, _fabs_58);
                                            float _fabs_59 = fabsf(o_hi_bf16_f32_1[11]);
                                            float _fmax_55 = fmaxf(_fmax_54, _fabs_59);
                                            float _fabs_60 = fabsf(o_hi_bf16_f32_1[12]);
                                            float _fmax_56 = fmaxf(_fmax_55, _fabs_60);
                                            float _fabs_61 = fabsf(o_hi_bf16_f32_1[13]);
                                            float _fmax_57 = fmaxf(_fmax_56, _fabs_61);
                                            float _fabs_62 = fabsf(o_hi_bf16_f32_1[14]);
                                            float _fmax_58 = fmaxf(_fmax_57, _fabs_62);
                                            float _fabs_63 = fabsf(o_hi_bf16_f32_1[15]);
                                            float _fmax_59 = fmaxf(_fmax_58, _fabs_63);
                                            {
                                                unsigned short _sf_pair;
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_59 * sf_mul_sel));
                                                *(reinterpret_cast<unsigned char*>(k_sf + (row_head * 8 + (c * 16 + 48 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                            }
                                            uint16_t _e4m3x2_f32_3;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(0.0f), "f"(_fmax_59 * sf_mul_sel));
                                            uint16_t _e4m3x2_decode_8 = (uint16_t)((unsigned int)_e4m3x2_f32_3 & 0xFFu);
                                            uint32_t _f16x2_decode_8;
                                            float _fp8_decode_3;
                                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_8) : "h"(_e4m3x2_decode_8));
                                            uint16_t _f16_decode_8 = (uint16_t)_f16x2_decode_8;
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_3) : "h"(_f16_decode_8));
                                            float _fdiv_rn_5 = __fdiv_rn(out_scale_sel, _fp8_decode_3);
                                            st_vals[0] = o_hi_bf16_f32_1[0] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[1] = o_hi_bf16_f32_1[1] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[2] = o_hi_bf16_f32_1[2] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[3] = o_hi_bf16_f32_1[3] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[4] = o_hi_bf16_f32_1[4] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[5] = o_hi_bf16_f32_1[5] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[6] = o_hi_bf16_f32_1[6] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[7] = o_hi_bf16_f32_1[7] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[8] = o_hi_bf16_f32_1[8] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[9] = o_hi_bf16_f32_1[9] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[10] = o_hi_bf16_f32_1[10] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[11] = o_hi_bf16_f32_1[11] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[12] = o_hi_bf16_f32_1[12] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[13] = o_hi_bf16_f32_1[13] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[14] = o_hi_bf16_f32_1[14] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[15] = o_hi_bf16_f32_1[15] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                            *(reinterpret_cast<unsigned int*>(k_out + (row_head * 16 + (c * 16 + 48 >> 3))) + (0)) = st_words[0];
                                            *(reinterpret_cast<unsigned int*>(k_out + (row_head * 16 + (c * 16 + 48 >> 3) + 1)) + (0)) = st_words[1];
                                        }
                                    } else {
                                        float x_lo_f32_1[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_lo_f32_1[_pair * 2])[0]), "=f"((&x_lo_f32_1[_pair * 2])[1])
                                                : "r"(x_lo[_pair]));
                                        }
                                        float _fabs_64 = fabsf(x_lo_f32_1[0]);
                                        float _fabs_65 = fabsf(x_lo_f32_1[1]);
                                        float _fmax_60 = fmaxf(_fabs_64, _fabs_65);
                                        float _fabs_66 = fabsf(x_lo_f32_1[2]);
                                        float _fmax_61 = fmaxf(_fmax_60, _fabs_66);
                                        float _fabs_67 = fabsf(x_lo_f32_1[3]);
                                        float _fmax_62 = fmaxf(_fmax_61, _fabs_67);
                                        float _fabs_68 = fabsf(x_lo_f32_1[4]);
                                        float _fmax_63 = fmaxf(_fmax_62, _fabs_68);
                                        float _fabs_69 = fabsf(x_lo_f32_1[5]);
                                        float _fmax_64 = fmaxf(_fmax_63, _fabs_69);
                                        float _fabs_70 = fabsf(x_lo_f32_1[6]);
                                        float _fmax_65 = fmaxf(_fmax_64, _fabs_70);
                                        float _fabs_71 = fabsf(x_lo_f32_1[7]);
                                        float _fmax_66 = fmaxf(_fmax_65, _fabs_71);
                                        float _fabs_72 = fabsf(x_lo_f32_1[8]);
                                        float _fmax_67 = fmaxf(_fmax_66, _fabs_72);
                                        float _fabs_73 = fabsf(x_lo_f32_1[9]);
                                        float _fmax_68 = fmaxf(_fmax_67, _fabs_73);
                                        float _fabs_74 = fabsf(x_lo_f32_1[10]);
                                        float _fmax_69 = fmaxf(_fmax_68, _fabs_74);
                                        float _fabs_75 = fabsf(x_lo_f32_1[11]);
                                        float _fmax_70 = fmaxf(_fmax_69, _fabs_75);
                                        float _fabs_76 = fabsf(x_lo_f32_1[12]);
                                        float _fmax_71 = fmaxf(_fmax_70, _fabs_76);
                                        float _fabs_77 = fabsf(x_lo_f32_1[13]);
                                        float _fmax_72 = fmaxf(_fmax_71, _fabs_77);
                                        float _fabs_78 = fabsf(x_lo_f32_1[14]);
                                        float _fmax_73 = fmaxf(_fmax_72, _fabs_78);
                                        float _fabs_79 = fabsf(x_lo_f32_1[15]);
                                        float _fmax_74 = fmaxf(_fmax_73, _fabs_79);
                                        {
                                            unsigned short _sf_pair;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_74 * sf_mul_sel));
                                            *(reinterpret_cast<unsigned char*>(v_sf + (row_head * 8 + (c * 16 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                        }
                                        uint16_t _e4m3x2_f32_4;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_4) : "f"(0.0f), "f"(_fmax_74 * sf_mul_sel));
                                        uint16_t _e4m3x2_decode_9 = (uint16_t)((unsigned int)_e4m3x2_f32_4 & 0xFFu);
                                        uint32_t _f16x2_decode_9;
                                        float _fp8_decode_4;
                                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_9) : "h"(_e4m3x2_decode_9));
                                        uint16_t _f16_decode_9 = (uint16_t)_f16x2_decode_9;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_4) : "h"(_f16_decode_9));
                                        float _fdiv_rn_6 = __fdiv_rn(out_scale_sel, _fp8_decode_4);
                                        st_vals[0] = x_lo_f32_1[0] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[1] = x_lo_f32_1[1] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[2] = x_lo_f32_1[2] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[3] = x_lo_f32_1[3] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[4] = x_lo_f32_1[4] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[5] = x_lo_f32_1[5] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[6] = x_lo_f32_1[6] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[7] = x_lo_f32_1[7] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[8] = x_lo_f32_1[8] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[9] = x_lo_f32_1[9] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[10] = x_lo_f32_1[10] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[11] = x_lo_f32_1[11] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[12] = x_lo_f32_1[12] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[13] = x_lo_f32_1[13] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[14] = x_lo_f32_1[14] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[15] = x_lo_f32_1[15] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                        *(reinterpret_cast<unsigned int*>(v_out + (row_head * 16 + (c * 16 >> 3))) + (0)) = st_words[0];
                                        *(reinterpret_cast<unsigned int*>(v_out + (row_head * 16 + (c * 16 >> 3) + 1)) + (0)) = st_words[1];
                                        float x_hi_f32_1[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_hi_f32_1[_pair * 2])[0]), "=f"((&x_hi_f32_1[_pair * 2])[1])
                                                : "r"(x_hi[_pair]));
                                        }
                                        float _fabs_80 = fabsf(x_hi_f32_1[0]);
                                        float _fabs_81 = fabsf(x_hi_f32_1[1]);
                                        float _fmax_75 = fmaxf(_fabs_80, _fabs_81);
                                        float _fabs_82 = fabsf(x_hi_f32_1[2]);
                                        float _fmax_76 = fmaxf(_fmax_75, _fabs_82);
                                        float _fabs_83 = fabsf(x_hi_f32_1[3]);
                                        float _fmax_77 = fmaxf(_fmax_76, _fabs_83);
                                        float _fabs_84 = fabsf(x_hi_f32_1[4]);
                                        float _fmax_78 = fmaxf(_fmax_77, _fabs_84);
                                        float _fabs_85 = fabsf(x_hi_f32_1[5]);
                                        float _fmax_79 = fmaxf(_fmax_78, _fabs_85);
                                        float _fabs_86 = fabsf(x_hi_f32_1[6]);
                                        float _fmax_80 = fmaxf(_fmax_79, _fabs_86);
                                        float _fabs_87 = fabsf(x_hi_f32_1[7]);
                                        float _fmax_81 = fmaxf(_fmax_80, _fabs_87);
                                        float _fabs_88 = fabsf(x_hi_f32_1[8]);
                                        float _fmax_82 = fmaxf(_fmax_81, _fabs_88);
                                        float _fabs_89 = fabsf(x_hi_f32_1[9]);
                                        float _fmax_83 = fmaxf(_fmax_82, _fabs_89);
                                        float _fabs_90 = fabsf(x_hi_f32_1[10]);
                                        float _fmax_84 = fmaxf(_fmax_83, _fabs_90);
                                        float _fabs_91 = fabsf(x_hi_f32_1[11]);
                                        float _fmax_85 = fmaxf(_fmax_84, _fabs_91);
                                        float _fabs_92 = fabsf(x_hi_f32_1[12]);
                                        float _fmax_86 = fmaxf(_fmax_85, _fabs_92);
                                        float _fabs_93 = fabsf(x_hi_f32_1[13]);
                                        float _fmax_87 = fmaxf(_fmax_86, _fabs_93);
                                        float _fabs_94 = fabsf(x_hi_f32_1[14]);
                                        float _fmax_88 = fmaxf(_fmax_87, _fabs_94);
                                        float _fabs_95 = fabsf(x_hi_f32_1[15]);
                                        float _fmax_89 = fmaxf(_fmax_88, _fabs_95);
                                        {
                                            unsigned short _sf_pair;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_89 * sf_mul_sel));
                                            *(reinterpret_cast<unsigned char*>(v_sf + (row_head * 8 + (c * 16 + 48 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                        }
                                        uint16_t _e4m3x2_f32_5;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_5) : "f"(0.0f), "f"(_fmax_89 * sf_mul_sel));
                                        uint16_t _e4m3x2_decode_10 = (uint16_t)((unsigned int)_e4m3x2_f32_5 & 0xFFu);
                                        uint32_t _f16x2_decode_10;
                                        float _fp8_decode_5;
                                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_10) : "h"(_e4m3x2_decode_10));
                                        uint16_t _f16_decode_10 = (uint16_t)_f16x2_decode_10;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_5) : "h"(_f16_decode_10));
                                        float _fdiv_rn_7 = __fdiv_rn(out_scale_sel, _fp8_decode_5);
                                        st_vals[0] = x_hi_f32_1[0] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[1] = x_hi_f32_1[1] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[2] = x_hi_f32_1[2] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[3] = x_hi_f32_1[3] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[4] = x_hi_f32_1[4] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[5] = x_hi_f32_1[5] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[6] = x_hi_f32_1[6] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[7] = x_hi_f32_1[7] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[8] = x_hi_f32_1[8] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[9] = x_hi_f32_1[9] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[10] = x_hi_f32_1[10] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[11] = x_hi_f32_1[11] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[12] = x_hi_f32_1[12] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[13] = x_hi_f32_1[13] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[14] = x_hi_f32_1[14] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[15] = x_hi_f32_1[15] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                        *(reinterpret_cast<unsigned int*>(v_out + (row_head * 16 + (c * 16 + 48 >> 3))) + (0)) = st_words[0];
                                        *(reinterpret_cast<unsigned int*>(v_out + (row_head * 16 + (c * 16 + 48 >> 3) + 1)) + (0)) = st_words[1];
                                    }
                                } else {
                                    unsigned int x_pt[8];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_pt[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((96 + (c - 3) * 16) * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_pt[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((96 + (c - 3) * 16) * 2 + 16 ^ (srow & 7) << 4)));
                                    if (kind == 0) {
                                        float x_pt_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_pt_f32[_pair * 2])[0]), "=f"((&x_pt_f32[_pair * 2])[1])
                                                : "r"(x_pt[_pair]));
                                        }
                                        float _fabs_96 = fabsf(x_pt_f32[0]);
                                        float _fabs_97 = fabsf(x_pt_f32[1]);
                                        float _fmax_90 = fmaxf(_fabs_96, _fabs_97);
                                        float _fabs_98 = fabsf(x_pt_f32[2]);
                                        float _fmax_91 = fmaxf(_fmax_90, _fabs_98);
                                        float _fabs_99 = fabsf(x_pt_f32[3]);
                                        float _fmax_92 = fmaxf(_fmax_91, _fabs_99);
                                        float _fabs_100 = fabsf(x_pt_f32[4]);
                                        float _fmax_93 = fmaxf(_fmax_92, _fabs_100);
                                        float _fabs_101 = fabsf(x_pt_f32[5]);
                                        float _fmax_94 = fmaxf(_fmax_93, _fabs_101);
                                        float _fabs_102 = fabsf(x_pt_f32[6]);
                                        float _fmax_95 = fmaxf(_fmax_94, _fabs_102);
                                        float _fabs_103 = fabsf(x_pt_f32[7]);
                                        float _fmax_96 = fmaxf(_fmax_95, _fabs_103);
                                        float _fabs_104 = fabsf(x_pt_f32[8]);
                                        float _fmax_97 = fmaxf(_fmax_96, _fabs_104);
                                        float _fabs_105 = fabsf(x_pt_f32[9]);
                                        float _fmax_98 = fmaxf(_fmax_97, _fabs_105);
                                        float _fabs_106 = fabsf(x_pt_f32[10]);
                                        float _fmax_99 = fmaxf(_fmax_98, _fabs_106);
                                        float _fabs_107 = fabsf(x_pt_f32[11]);
                                        float _fmax_100 = fmaxf(_fmax_99, _fabs_107);
                                        float _fabs_108 = fabsf(x_pt_f32[12]);
                                        float _fmax_101 = fmaxf(_fmax_100, _fabs_108);
                                        float _fabs_109 = fabsf(x_pt_f32[13]);
                                        float _fmax_102 = fmaxf(_fmax_101, _fabs_109);
                                        float _fabs_110 = fabsf(x_pt_f32[14]);
                                        float _fmax_103 = fmaxf(_fmax_102, _fabs_110);
                                        float _fabs_111 = fabsf(x_pt_f32[15]);
                                        float _fmax_104 = fmaxf(_fmax_103, _fabs_111);
                                        {
                                            unsigned short _sf_pair;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_104 * sf_mul_sel));
                                            *(reinterpret_cast<unsigned char*>(q_sf + (row_head * 8 + (96 + (c - 3) * 16 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                        }
                                        uint16_t _e4m3x2_f32_6;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_6) : "f"(0.0f), "f"(_fmax_104 * sf_mul_sel));
                                        uint16_t _e4m3x2_decode_11 = (uint16_t)((unsigned int)_e4m3x2_f32_6 & 0xFFu);
                                        uint32_t _f16x2_decode_11;
                                        float _fp8_decode_6;
                                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_11) : "h"(_e4m3x2_decode_11));
                                        uint16_t _f16_decode_11 = (uint16_t)_f16x2_decode_11;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_6) : "h"(_f16_decode_11));
                                        float _fdiv_rn_8 = __fdiv_rn(out_scale_sel, _fp8_decode_6);
                                        st_vals[0] = x_pt_f32[0] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[1] = x_pt_f32[1] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[2] = x_pt_f32[2] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[3] = x_pt_f32[3] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[4] = x_pt_f32[4] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[5] = x_pt_f32[5] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[6] = x_pt_f32[6] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[7] = x_pt_f32[7] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[8] = x_pt_f32[8] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[9] = x_pt_f32[9] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[10] = x_pt_f32[10] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[11] = x_pt_f32[11] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[12] = x_pt_f32[12] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[13] = x_pt_f32[13] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[14] = x_pt_f32[14] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[15] = x_pt_f32[15] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                        *(reinterpret_cast<unsigned int*>(q_out + (row_head * 16 + (96 + (c - 3) * 16 >> 3))) + (0)) = st_words[0];
                                        *(reinterpret_cast<unsigned int*>(q_out + (row_head * 16 + (96 + (c - 3) * 16 >> 3) + 1)) + (0)) = st_words[1];
                                    } else if (kind == 1) {
                                        float x_pt_f32_1[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_pt_f32_1[_pair * 2])[0]), "=f"((&x_pt_f32_1[_pair * 2])[1])
                                                : "r"(x_pt[_pair]));
                                        }
                                        float _fabs_112 = fabsf(x_pt_f32_1[0]);
                                        float _fabs_113 = fabsf(x_pt_f32_1[1]);
                                        float _fmax_105 = fmaxf(_fabs_112, _fabs_113);
                                        float _fabs_114 = fabsf(x_pt_f32_1[2]);
                                        float _fmax_106 = fmaxf(_fmax_105, _fabs_114);
                                        float _fabs_115 = fabsf(x_pt_f32_1[3]);
                                        float _fmax_107 = fmaxf(_fmax_106, _fabs_115);
                                        float _fabs_116 = fabsf(x_pt_f32_1[4]);
                                        float _fmax_108 = fmaxf(_fmax_107, _fabs_116);
                                        float _fabs_117 = fabsf(x_pt_f32_1[5]);
                                        float _fmax_109 = fmaxf(_fmax_108, _fabs_117);
                                        float _fabs_118 = fabsf(x_pt_f32_1[6]);
                                        float _fmax_110 = fmaxf(_fmax_109, _fabs_118);
                                        float _fabs_119 = fabsf(x_pt_f32_1[7]);
                                        float _fmax_111 = fmaxf(_fmax_110, _fabs_119);
                                        float _fabs_120 = fabsf(x_pt_f32_1[8]);
                                        float _fmax_112 = fmaxf(_fmax_111, _fabs_120);
                                        float _fabs_121 = fabsf(x_pt_f32_1[9]);
                                        float _fmax_113 = fmaxf(_fmax_112, _fabs_121);
                                        float _fabs_122 = fabsf(x_pt_f32_1[10]);
                                        float _fmax_114 = fmaxf(_fmax_113, _fabs_122);
                                        float _fabs_123 = fabsf(x_pt_f32_1[11]);
                                        float _fmax_115 = fmaxf(_fmax_114, _fabs_123);
                                        float _fabs_124 = fabsf(x_pt_f32_1[12]);
                                        float _fmax_116 = fmaxf(_fmax_115, _fabs_124);
                                        float _fabs_125 = fabsf(x_pt_f32_1[13]);
                                        float _fmax_117 = fmaxf(_fmax_116, _fabs_125);
                                        float _fabs_126 = fabsf(x_pt_f32_1[14]);
                                        float _fmax_118 = fmaxf(_fmax_117, _fabs_126);
                                        float _fabs_127 = fabsf(x_pt_f32_1[15]);
                                        float _fmax_119 = fmaxf(_fmax_118, _fabs_127);
                                        {
                                            unsigned short _sf_pair;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_119 * sf_mul_sel));
                                            *(reinterpret_cast<unsigned char*>(k_sf + (row_head * 8 + (96 + (c - 3) * 16 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                        }
                                        uint16_t _e4m3x2_f32_7;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_7) : "f"(0.0f), "f"(_fmax_119 * sf_mul_sel));
                                        uint16_t _e4m3x2_decode_12 = (uint16_t)((unsigned int)_e4m3x2_f32_7 & 0xFFu);
                                        uint32_t _f16x2_decode_12;
                                        float _fp8_decode_7;
                                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_12) : "h"(_e4m3x2_decode_12));
                                        uint16_t _f16_decode_12 = (uint16_t)_f16x2_decode_12;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_7) : "h"(_f16_decode_12));
                                        float _fdiv_rn_9 = __fdiv_rn(out_scale_sel, _fp8_decode_7);
                                        st_vals[0] = x_pt_f32_1[0] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[1] = x_pt_f32_1[1] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[2] = x_pt_f32_1[2] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[3] = x_pt_f32_1[3] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[4] = x_pt_f32_1[4] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[5] = x_pt_f32_1[5] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[6] = x_pt_f32_1[6] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[7] = x_pt_f32_1[7] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[8] = x_pt_f32_1[8] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[9] = x_pt_f32_1[9] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[10] = x_pt_f32_1[10] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[11] = x_pt_f32_1[11] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[12] = x_pt_f32_1[12] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[13] = x_pt_f32_1[13] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[14] = x_pt_f32_1[14] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[15] = x_pt_f32_1[15] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                        *(reinterpret_cast<unsigned int*>(k_out + (row_head * 16 + (96 + (c - 3) * 16 >> 3))) + (0)) = st_words[0];
                                        *(reinterpret_cast<unsigned int*>(k_out + (row_head * 16 + (96 + (c - 3) * 16 >> 3) + 1)) + (0)) = st_words[1];
                                    } else {
                                        float x_pt_f32_2[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_pt_f32_2[_pair * 2])[0]), "=f"((&x_pt_f32_2[_pair * 2])[1])
                                                : "r"(x_pt[_pair]));
                                        }
                                        float _fabs_128 = fabsf(x_pt_f32_2[0]);
                                        float _fabs_129 = fabsf(x_pt_f32_2[1]);
                                        float _fmax_120 = fmaxf(_fabs_128, _fabs_129);
                                        float _fabs_130 = fabsf(x_pt_f32_2[2]);
                                        float _fmax_121 = fmaxf(_fmax_120, _fabs_130);
                                        float _fabs_131 = fabsf(x_pt_f32_2[3]);
                                        float _fmax_122 = fmaxf(_fmax_121, _fabs_131);
                                        float _fabs_132 = fabsf(x_pt_f32_2[4]);
                                        float _fmax_123 = fmaxf(_fmax_122, _fabs_132);
                                        float _fabs_133 = fabsf(x_pt_f32_2[5]);
                                        float _fmax_124 = fmaxf(_fmax_123, _fabs_133);
                                        float _fabs_134 = fabsf(x_pt_f32_2[6]);
                                        float _fmax_125 = fmaxf(_fmax_124, _fabs_134);
                                        float _fabs_135 = fabsf(x_pt_f32_2[7]);
                                        float _fmax_126 = fmaxf(_fmax_125, _fabs_135);
                                        float _fabs_136 = fabsf(x_pt_f32_2[8]);
                                        float _fmax_127 = fmaxf(_fmax_126, _fabs_136);
                                        float _fabs_137 = fabsf(x_pt_f32_2[9]);
                                        float _fmax_128 = fmaxf(_fmax_127, _fabs_137);
                                        float _fabs_138 = fabsf(x_pt_f32_2[10]);
                                        float _fmax_129 = fmaxf(_fmax_128, _fabs_138);
                                        float _fabs_139 = fabsf(x_pt_f32_2[11]);
                                        float _fmax_130 = fmaxf(_fmax_129, _fabs_139);
                                        float _fabs_140 = fabsf(x_pt_f32_2[12]);
                                        float _fmax_131 = fmaxf(_fmax_130, _fabs_140);
                                        float _fabs_141 = fabsf(x_pt_f32_2[13]);
                                        float _fmax_132 = fmaxf(_fmax_131, _fabs_141);
                                        float _fabs_142 = fabsf(x_pt_f32_2[14]);
                                        float _fmax_133 = fmaxf(_fmax_132, _fabs_142);
                                        float _fabs_143 = fabsf(x_pt_f32_2[15]);
                                        float _fmax_134 = fmaxf(_fmax_133, _fabs_143);
                                        {
                                            unsigned short _sf_pair;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_134 * sf_mul_sel));
                                            *(reinterpret_cast<unsigned char*>(v_sf + (row_head * 8 + (96 + (c - 3) * 16 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                        }
                                        uint16_t _e4m3x2_f32_8;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_8) : "f"(0.0f), "f"(_fmax_134 * sf_mul_sel));
                                        uint16_t _e4m3x2_decode_13 = (uint16_t)((unsigned int)_e4m3x2_f32_8 & 0xFFu);
                                        uint32_t _f16x2_decode_13;
                                        float _fp8_decode_8;
                                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_decode_13));
                                        uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_8) : "h"(_f16_decode_13));
                                        float _fdiv_rn_10 = __fdiv_rn(out_scale_sel, _fp8_decode_8);
                                        st_vals[0] = x_pt_f32_2[0] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[1] = x_pt_f32_2[1] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[2] = x_pt_f32_2[2] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[3] = x_pt_f32_2[3] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[4] = x_pt_f32_2[4] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[5] = x_pt_f32_2[5] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[6] = x_pt_f32_2[6] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[7] = x_pt_f32_2[7] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[8] = x_pt_f32_2[8] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[9] = x_pt_f32_2[9] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[10] = x_pt_f32_2[10] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[11] = x_pt_f32_2[11] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[12] = x_pt_f32_2[12] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[13] = x_pt_f32_2[13] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[14] = x_pt_f32_2[14] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[15] = x_pt_f32_2[15] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                        *(reinterpret_cast<unsigned int*>(v_out + (row_head * 16 + (96 + (c - 3) * 16 >> 3))) + (0)) = st_words[0];
                                        *(reinterpret_cast<unsigned int*>(v_out + (row_head * 16 + (96 + (c - 3) * 16 >> 3) + 1)) + (0)) = st_words[1];
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                }
            }
        }
    }

    // Cleanup
}

}  // namespace h3_qkv_gemm_fp8_nvfp4_sm120a
#undef GROUP_M
#undef H3_QPA_INF
#undef NUM_AB_PIPE_STAGES
#undef SMEM_A_STAGE_OFF
#undef SMEM_A_STAGE_STAGE_BYTES
#undef SMEM_A_STAGE_STRIDE
#undef SMEM_B_STAGE_OFF
#undef SMEM_B_STAGE_STAGE_BYTES
#undef SMEM_B_STAGE_STRIDE
#undef SMEM_ROWSTAT_OFF
#undef SMEM_ROWSTAT_STAGE_BYTES
#undef SMEM_ROWSTAT_STRIDE
#undef SMEM_SFA_STAGE_OFF
#undef SMEM_SFA_STAGE_STAGE_BYTES
#undef SMEM_SFA_STAGE_STRIDE
#undef SMEM_SFB_STAGE_OFF
#undef SMEM_SFB_STAGE_STAGE_BYTES
#undef SMEM_SFB_STAGE_STRIDE
#undef SMEM_STAGING_OFF
#undef SMEM_STAGING_STAGE_BYTES
#undef SMEM_STAGING_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef ab_empty_addr
#undef ab_full_addr

namespace h3_qkv_gemm_nvfp4_bf16_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_QPA_INF CUDART_INF_F
#define NUM_AB_PIPE_STAGES 2
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 16384
#define SMEM_A_STAGE_STRIDE 16384
#define SMEM_B_STAGE_OFF 33792
#define SMEM_B_STAGE_STAGE_BYTES 16384
#define SMEM_B_STAGE_STRIDE 16384
#define SMEM_STAGING_OFF 66560
#define SMEM_STAGING_STAGE_BYTES 16384
#define SMEM_STAGING_STRIDE 16384
#define SMEM_ROWSTAT_OFF 82944
#define SMEM_ROWSTAT_STAGE_BYTES 512
#define SMEM_ROWSTAT_STRIDE 512
#define SMEM_SFA_STAGE_OFF 83456
#define SMEM_SFA_STAGE_STAGE_BYTES 2048
#define SMEM_SFA_STAGE_STRIDE 2048
#define SMEM_SFB_STAGE_OFF 87552
#define SMEM_SFB_STAGE_STAGE_BYTES 2048
#define SMEM_SFB_STAGE_STRIDE 2048
#define SMEM_TOTAL 91648
#define THREADS 288
#define GROUP_M 16

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}


__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__global__ __launch_bounds__(288, 1) void
kernel_h3_qkv_gemm_fused(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ act_scale, float* __restrict__ w_scale, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, unsigned int* __restrict__ q_out, unsigned int* __restrict__ k_out, unsigned int* __restrict__ v_out, uint8_t* __restrict__ q_sf, uint8_t* __restrict__ k_sf, uint8_t* __restrict__ v_sf, int M, int num_m_tiles, int total_tiles, float eps, float alpha, float out_scale_q, float out_scale_k, float out_scale_v, float sf_mul_q, float sf_mul_k, float sf_mul_v)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int B_stage_addr = smem + 33792;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int staging_addr = smem + 66560;
    float* rowstat = reinterpret_cast<float*>(smem_raw + 82944);
    const int rowstat_addr = smem + 82944;
    unsigned int* SFA_stage = reinterpret_cast<unsigned int*>(smem_raw + 83456);
    const int SFA_stage_addr = smem + 83456;
    unsigned int* SFB_stage = reinterpret_cast<unsigned int*>(smem_raw + 87552);
    const int SFB_stage_addr = smem + 87552;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // ab_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 16, 8);
            mbarrier_init(smem + 24, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: producer ----
    if (warp == 0) {
        { // producer_main
            unsigned int load_stage = 0;
            unsigned int _phase_ab_empty = 1;
            #pragma unroll 1
            for (int tile = bid; tile < total_tiles; tile += num_bids) {
                int tile_m = tile / (GROUP_M * 168) * GROUP_M + (tile - tile / (GROUP_M * 168) * (GROUP_M * 168)) % ((num_m_tiles - tile / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 168) * GROUP_M : GROUP_M);
                int tile_n = (tile - tile / (GROUP_M * 168) * (GROUP_M * 168)) / ((num_m_tiles - tile / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 168) * GROUP_M : GROUP_M);
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int k_tile = 0; k_tile < 21; k_tile++) {
                        mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                        mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 36864);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(A_stage_addr + load_stage * 16384), "l"((&A)), "r"(k_tile * 128), "r"(tile_m * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(B_stage_addr + load_stage * 16384), "l"((&B)), "r"(k_tile * 128), "r"(tile_n * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(SFA_stage_addr + load_stage * 2048), "l"((&SFA)), "r"(k_tile * 16), "r"(tile_m * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(SFB_stage_addr + load_stage * 2048), "l"((&SFB)), "r"(0), "r"(tile_n * 168 + k_tile * 8), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        load_stage += 1;
                        if (load_stage == 2) { load_stage = 0; _phase_ab_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 1 && warp <= 8) {
        { // mma_main
            unsigned int mma_stage = 0;
            int warp_id_in_role = (warp - 1);
            int warp_m = warp_id_in_role % 4;
            int warp_n = warp_id_in_role / 4;
            int role_tid = warp_id_in_role * 32 + lane;
            float accum[64];
            unsigned int a_frag[8];
            unsigned int b_frag[16];
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 168) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 168) * (GROUP_M * 168)) % ((num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 168) * (GROUP_M * 168)) / ((num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M : GROUP_M);
                accum[0] = 0.0f;
                accum[1] = 0.0f;
                accum[2] = 0.0f;
                accum[3] = 0.0f;
                accum[4] = 0.0f;
                accum[5] = 0.0f;
                accum[6] = 0.0f;
                accum[7] = 0.0f;
                accum[8] = 0.0f;
                accum[9] = 0.0f;
                accum[10] = 0.0f;
                accum[11] = 0.0f;
                accum[12] = 0.0f;
                accum[13] = 0.0f;
                accum[14] = 0.0f;
                accum[15] = 0.0f;
                accum[16] = 0.0f;
                accum[17] = 0.0f;
                accum[18] = 0.0f;
                accum[19] = 0.0f;
                accum[20] = 0.0f;
                accum[21] = 0.0f;
                accum[22] = 0.0f;
                accum[23] = 0.0f;
                accum[24] = 0.0f;
                accum[25] = 0.0f;
                accum[26] = 0.0f;
                accum[27] = 0.0f;
                accum[28] = 0.0f;
                accum[29] = 0.0f;
                accum[30] = 0.0f;
                accum[31] = 0.0f;
                accum[32] = 0.0f;
                accum[33] = 0.0f;
                accum[34] = 0.0f;
                accum[35] = 0.0f;
                accum[36] = 0.0f;
                accum[37] = 0.0f;
                accum[38] = 0.0f;
                accum[39] = 0.0f;
                accum[40] = 0.0f;
                accum[41] = 0.0f;
                accum[42] = 0.0f;
                accum[43] = 0.0f;
                accum[44] = 0.0f;
                accum[45] = 0.0f;
                accum[46] = 0.0f;
                accum[47] = 0.0f;
                accum[48] = 0.0f;
                accum[49] = 0.0f;
                accum[50] = 0.0f;
                accum[51] = 0.0f;
                accum[52] = 0.0f;
                accum[53] = 0.0f;
                accum[54] = 0.0f;
                accum[55] = 0.0f;
                accum[56] = 0.0f;
                accum[57] = 0.0f;
                accum[58] = 0.0f;
                accum[59] = 0.0f;
                accum[60] = 0.0f;
                accum[61] = 0.0f;
                accum[62] = 0.0f;
                accum[63] = 0.0f;
                #pragma unroll 1
                for (int k_tile_1 = 0; k_tile_1 < 21; k_tile_1++) {
                    mbarrier_wait(ab_full_addr + (mma_stage) * 8, _phase_ab_full);
                    for (int k_step = 0; k_step < 4; k_step++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        unsigned int _SFA_stage_reg_0[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFA_stage_reg_0[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)((warp_m * 16 + (lane & 1) * 8 + (lane >> 2)) * 4) + (unsigned int)k_step) + _lr];
                        }
                        unsigned int _SFA_stage_reg_1[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFA_stage_reg_1[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)((warp_m * 16 + 64 + (lane & 1) * 8 + (lane >> 2)) * 4) + (unsigned int)k_step) + _lr];
                        }
                        unsigned int _SFB_stage_reg_0[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_0[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_1[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_1[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_2[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_2[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_3[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_3[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_4[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_4[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_5[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_5[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_6[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_6[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_7[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_7[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                        }
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 2) { mma_stage = 0; _phase_ab_full ^= 1; }
                }
                int kind = tile_n_1 % 3;
                int head = tile_n_1 / 3;
                int col_base = tile_n_1 * 128 + warp_n * 64 + (lane & 3) * 2;
                float sw[16];
                float nw[16];
                float out_scale_sel = ((kind == 0) ? out_scale_q : ((kind == 1) ? out_scale_k : out_scale_v));
                float sf_mul_sel = ((kind == 0) ? sf_mul_q : ((kind == 1) ? sf_mul_k : sf_mul_v));
                sw[0] = alpha;
                sw[1] = alpha;
                sw[2] = alpha;
                sw[3] = alpha;
                sw[4] = alpha;
                sw[5] = alpha;
                sw[6] = alpha;
                sw[7] = alpha;
                sw[8] = alpha;
                sw[9] = alpha;
                sw[10] = alpha;
                sw[11] = alpha;
                sw[12] = alpha;
                sw[13] = alpha;
                sw[14] = alpha;
                sw[15] = alpha;
                for (int mma_n = 0; mma_n < 8; mma_n++) {
                    int d_col = warp_n * 64 + mma_n * 8 + (lane & 3) * 2;
                    if (kind == 0) {
                        nw[mma_n * 2] = (float)q_norm_weight[d_col];
                        nw[mma_n * 2 + 1] = (float)q_norm_weight[d_col + 1];
                    } else if (kind == 1) {
                        nw[mma_n * 2] = (float)k_norm_weight[d_col];
                        nw[mma_n * 2 + 1] = (float)k_norm_weight[d_col + 1];
                    } else {
                        nw[mma_n * 2] = 1.0f;
                        nw[mma_n * 2 + 1] = 1.0f;
                    }
                }
                for (int half = 0; half < 2; half++) {
                    int row_lo = tile_m_1 * 128 + half * 64 + warp_m * 16 + (lane >> 2);
                    int row_hi = row_lo + 8;
                    float sa_lo = 1.0f;
                    float sa_hi = 1.0f;
                    float rounded[32];
                    float ss_lo = 0.0f;
                    float ss_hi = 0.0f;
                    for (int mma_n_1 = 0; mma_n_1 < 8; mma_n_1++) {
                        float scaled[4];
                        scaled[0] = accum[(half * 8 + mma_n_1) * 4] * sa_lo * sw[mma_n_1 * 2];
                        scaled[1] = accum[(half * 8 + mma_n_1) * 4 + 1] * sa_lo * sw[mma_n_1 * 2 + 1];
                        scaled[2] = accum[(half * 8 + mma_n_1) * 4 + 2] * sa_hi * sw[mma_n_1 * 2];
                        scaled[3] = accum[(half * 8 + mma_n_1) * 4 + 3] * sa_hi * sw[mma_n_1 * 2 + 1];
                        uint32_t scaled_bf16[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                            scaled_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        float scaled_bf16_f32[4];
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&scaled_bf16_f32[_pair * 2])[0]), "=f"((&scaled_bf16_f32[_pair * 2])[1])
                                : "r"(scaled_bf16[_pair]));
                        }
                        for (int e = 0; e < 4; e++) {
                            rounded[mma_n_1 * 4 + e] = scaled_bf16_f32[e];
                        }
                        ss_lo += scaled_bf16_f32[0] * scaled_bf16_f32[0] + scaled_bf16_f32[1] * scaled_bf16_f32[1];
                        ss_hi += scaled_bf16_f32[2] * scaled_bf16_f32[2] + scaled_bf16_f32[3] * scaled_bf16_f32[3];
                    }
                    float rstd_lo = 1.0f;
                    float rstd_hi = 1.0f;
                    if (kind < 2) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, ss_lo, 1);
                        ss_lo += _shfl_xor_0;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo, 2);
                        ss_lo += _shfl_xor_1;
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, ss_hi, 1);
                        ss_hi += _shfl_xor_2;
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, ss_hi, 2);
                        ss_hi += _shfl_xor_3;
                        int srow_lo = warp_m * 16 + (lane >> 2);
                        if ((lane & 3) == 0) {
                            rowstat[warp_n * 64 + srow_lo] = ss_lo;
                            rowstat[warp_n * 64 + srow_lo + 8] = ss_hi;
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        float other_lo = rowstat[(1 - warp_n) * 64 + srow_lo];
                        float other_hi = rowstat[(1 - warp_n) * 64 + srow_lo + 8];
                        float _fdiv_rn_0 = __fdiv_rn(ss_lo + other_lo, 128.0f);
                        float _rsqrt_0 = rsqrtf(_fdiv_rn_0 + eps);
                        rstd_lo = _rsqrt_0;
                        float _fdiv_rn_1 = __fdiv_rn(ss_hi + other_hi, 128.0f);
                        float _rsqrt_1 = rsqrtf(_fdiv_rn_1 + eps);
                        rstd_hi = _rsqrt_1;
                    }
                    int srow_st = warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                    for (int mma_n_2 = 0; mma_n_2 < 8; mma_n_2++) {
                        float normed[4];
                        if (kind < 2) {
                            normed[0] = rounded[mma_n_2 * 4] * rstd_lo * nw[mma_n_2 * 2];
                            normed[1] = rounded[mma_n_2 * 4 + 1] * rstd_lo * nw[mma_n_2 * 2 + 1];
                            normed[2] = rounded[mma_n_2 * 4 + 2] * rstd_hi * nw[mma_n_2 * 2];
                            normed[3] = rounded[mma_n_2 * 4 + 3] * rstd_hi * nw[mma_n_2 * 2 + 1];
                        } else {
                            normed[0] = rounded[mma_n_2 * 4];
                            normed[1] = rounded[mma_n_2 * 4 + 1];
                            normed[2] = rounded[mma_n_2 * 4 + 2];
                            normed[3] = rounded[mma_n_2 * 4 + 3];
                        }
                        uint32_t normed_bf16[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed[_lp*2 + 0], normed[_lp*2+1 + 0]));
                            normed_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + mma_n_2 * 8) * 2 ^ (srow_st & 7) << 4));
                        asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16[1]))
                            : "memory");
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                    for (int it = 0; it < 2; it++) {
                        int task = role_tid + it * 256;
                        if (task < 320) {
                            int srow = task / 5;
                            int c = task % 5;
                            int grow = tile_m_1 * 128 + half * 64 + srow;
                            if (grow < M) {
                                int row_head = grow * 56 + head;
                                unsigned int st_words[4];
                                float st_vals[16];
                                if (c < 3) {
                                    unsigned int x_lo[8];
                                    unsigned int x_hi[8];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 16 * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 16 * 2 + 16 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((c * 16 + 48) * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((c * 16 + 48) * 2 + 16 ^ (srow & 7) << 4)));
                                    if (kind < 2) {
                                        float x_lo_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_lo_f32[_pair * 2])[0]), "=f"((&x_lo_f32[_pair * 2])[1])
                                                : "r"(x_lo[_pair]));
                                        }
                                        float x_hi_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_hi_f32[_pair * 2])[0]), "=f"((&x_hi_f32[_pair * 2])[1])
                                                : "r"(x_hi[_pair]));
                                        }
                                        float _vec_load_0[8];
                                        {
                                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + c * 16) + 0);
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
                                                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_1[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_1[8];
                                        {
                                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + c * 16 + 8) + 0);
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
                                                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_2[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_2[8];
                                        {
                                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + 48 + c * 16) + 0);
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
                                                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_3[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_3[8];
                                        {
                                            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + 48 + c * 16 + 8) + 0);
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
                                                        : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_4[_pair]));
                                                }
                                            }
                                        }
                                        float o_lo[16];
                                        float o_hi[16];
                                        for (int e_1 = 0; e_1 < 8; e_1++) {
                                            float cv0 = _vec_load_0[e_1];
                                            float sv0 = _vec_load_2[e_1];
                                            float cv1 = _vec_load_1[e_1];
                                            float sv1 = _vec_load_3[e_1];
                                            o_lo[e_1] = x_lo_f32[e_1] * cv0 - x_hi_f32[e_1] * sv0;
                                            o_hi[e_1] = x_hi_f32[e_1] * cv0 + x_lo_f32[e_1] * sv0;
                                            o_lo[8 + e_1] = x_lo_f32[8 + e_1] * cv1 - x_hi_f32[8 + e_1] * sv1;
                                            o_hi[8 + e_1] = x_hi_f32[8 + e_1] * cv1 + x_lo_f32[8 + e_1] * sv1;
                                        }
                                        uint32_t o_lo_bf16[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_lo[_lp*2 + 0], o_lo[_lp*2+1 + 0]));
                                            o_lo_bf16[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        uint32_t o_hi_bf16[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_hi[_lp*2 + 0], o_hi[_lp*2+1 + 0]));
                                            o_hi_bf16[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        if (kind == 0) {
                                            reinterpret_cast<int4*>(q_out + (row_head * 64 + (c * 16 >> 1)))[0] = reinterpret_cast<int4*>(o_lo_bf16 + 0)[0];
                                            reinterpret_cast<int4*>(q_out + (row_head * 64 + (c * 16 >> 1) + 4))[0] = reinterpret_cast<int4*>(o_lo_bf16 + 4)[0];
                                            reinterpret_cast<int4*>(q_out + (row_head * 64 + (c * 16 + 48 >> 1)))[0] = reinterpret_cast<int4*>(o_hi_bf16 + 0)[0];
                                            reinterpret_cast<int4*>(q_out + (row_head * 64 + (c * 16 + 48 >> 1) + 4))[0] = reinterpret_cast<int4*>(o_hi_bf16 + 4)[0];
                                        } else {
                                            reinterpret_cast<int4*>(k_out + (row_head * 64 + (c * 16 >> 1)))[0] = reinterpret_cast<int4*>(o_lo_bf16 + 0)[0];
                                            reinterpret_cast<int4*>(k_out + (row_head * 64 + (c * 16 >> 1) + 4))[0] = reinterpret_cast<int4*>(o_lo_bf16 + 4)[0];
                                            reinterpret_cast<int4*>(k_out + (row_head * 64 + (c * 16 + 48 >> 1)))[0] = reinterpret_cast<int4*>(o_hi_bf16 + 0)[0];
                                            reinterpret_cast<int4*>(k_out + (row_head * 64 + (c * 16 + 48 >> 1) + 4))[0] = reinterpret_cast<int4*>(o_hi_bf16 + 4)[0];
                                        }
                                    } else {
                                        reinterpret_cast<int4*>(v_out + (row_head * 64 + (c * 16 >> 1)))[0] = reinterpret_cast<int4*>(x_lo + 0)[0];
                                        reinterpret_cast<int4*>(v_out + (row_head * 64 + (c * 16 >> 1) + 4))[0] = reinterpret_cast<int4*>(x_lo + 4)[0];
                                        reinterpret_cast<int4*>(v_out + (row_head * 64 + (c * 16 + 48 >> 1)))[0] = reinterpret_cast<int4*>(x_hi + 0)[0];
                                        reinterpret_cast<int4*>(v_out + (row_head * 64 + (c * 16 + 48 >> 1) + 4))[0] = reinterpret_cast<int4*>(x_hi + 4)[0];
                                    }
                                } else {
                                    unsigned int x_pt[8];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_pt[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((96 + (c - 3) * 16) * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_pt[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((96 + (c - 3) * 16) * 2 + 16 ^ (srow & 7) << 4)));
                                    if (kind == 0) {
                                        reinterpret_cast<int4*>(q_out + (row_head * 64 + (96 + (c - 3) * 16 >> 1)))[0] = reinterpret_cast<int4*>(x_pt + 0)[0];
                                        reinterpret_cast<int4*>(q_out + (row_head * 64 + (96 + (c - 3) * 16 >> 1) + 4))[0] = reinterpret_cast<int4*>(x_pt + 4)[0];
                                    } else if (kind == 1) {
                                        reinterpret_cast<int4*>(k_out + (row_head * 64 + (96 + (c - 3) * 16 >> 1)))[0] = reinterpret_cast<int4*>(x_pt + 0)[0];
                                        reinterpret_cast<int4*>(k_out + (row_head * 64 + (96 + (c - 3) * 16 >> 1) + 4))[0] = reinterpret_cast<int4*>(x_pt + 4)[0];
                                    } else {
                                        reinterpret_cast<int4*>(v_out + (row_head * 64 + (96 + (c - 3) * 16 >> 1)))[0] = reinterpret_cast<int4*>(x_pt + 0)[0];
                                        reinterpret_cast<int4*>(v_out + (row_head * 64 + (96 + (c - 3) * 16 >> 1) + 4))[0] = reinterpret_cast<int4*>(x_pt + 4)[0];
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                }
            }
        }
    }

    // Cleanup
}

}  // namespace h3_qkv_gemm_nvfp4_bf16_sm120a
#undef GROUP_M
#undef H3_QPA_INF
#undef NUM_AB_PIPE_STAGES
#undef SMEM_A_STAGE_OFF
#undef SMEM_A_STAGE_STAGE_BYTES
#undef SMEM_A_STAGE_STRIDE
#undef SMEM_B_STAGE_OFF
#undef SMEM_B_STAGE_STAGE_BYTES
#undef SMEM_B_STAGE_STRIDE
#undef SMEM_ROWSTAT_OFF
#undef SMEM_ROWSTAT_STAGE_BYTES
#undef SMEM_ROWSTAT_STRIDE
#undef SMEM_SFA_STAGE_OFF
#undef SMEM_SFA_STAGE_STAGE_BYTES
#undef SMEM_SFA_STAGE_STRIDE
#undef SMEM_SFB_STAGE_OFF
#undef SMEM_SFB_STAGE_STAGE_BYTES
#undef SMEM_SFB_STAGE_STRIDE
#undef SMEM_STAGING_OFF
#undef SMEM_STAGING_STAGE_BYTES
#undef SMEM_STAGING_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef ab_empty_addr
#undef ab_full_addr

namespace h3_qkv_gemm_nvfp4_e4m3_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_QPA_INF CUDART_INF_F
#define NUM_AB_PIPE_STAGES 2
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 16384
#define SMEM_A_STAGE_STRIDE 16384
#define SMEM_B_STAGE_OFF 33792
#define SMEM_B_STAGE_STAGE_BYTES 16384
#define SMEM_B_STAGE_STRIDE 16384
#define SMEM_STAGING_OFF 66560
#define SMEM_STAGING_STAGE_BYTES 16384
#define SMEM_STAGING_STRIDE 16384
#define SMEM_ROWSTAT_OFF 82944
#define SMEM_ROWSTAT_STAGE_BYTES 512
#define SMEM_ROWSTAT_STRIDE 512
#define SMEM_SFA_STAGE_OFF 83456
#define SMEM_SFA_STAGE_STAGE_BYTES 2048
#define SMEM_SFA_STAGE_STRIDE 2048
#define SMEM_SFB_STAGE_OFF 87552
#define SMEM_SFB_STAGE_STAGE_BYTES 2048
#define SMEM_SFB_STAGE_STRIDE 2048
#define SMEM_TOTAL 91648
#define THREADS 288
#define GROUP_M 16

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}


__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__global__ __launch_bounds__(288, 1) void
kernel_h3_qkv_gemm_fused(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ act_scale, float* __restrict__ w_scale, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, unsigned int* __restrict__ q_out, unsigned int* __restrict__ k_out, unsigned int* __restrict__ v_out, uint8_t* __restrict__ q_sf, uint8_t* __restrict__ k_sf, uint8_t* __restrict__ v_sf, int M, int num_m_tiles, int total_tiles, float eps, float alpha, float out_scale_q, float out_scale_k, float out_scale_v, float sf_mul_q, float sf_mul_k, float sf_mul_v)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int B_stage_addr = smem + 33792;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int staging_addr = smem + 66560;
    float* rowstat = reinterpret_cast<float*>(smem_raw + 82944);
    const int rowstat_addr = smem + 82944;
    unsigned int* SFA_stage = reinterpret_cast<unsigned int*>(smem_raw + 83456);
    const int SFA_stage_addr = smem + 83456;
    unsigned int* SFB_stage = reinterpret_cast<unsigned int*>(smem_raw + 87552);
    const int SFB_stage_addr = smem + 87552;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // ab_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 16, 8);
            mbarrier_init(smem + 24, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: producer ----
    if (warp == 0) {
        { // producer_main
            unsigned int load_stage = 0;
            unsigned int _phase_ab_empty = 1;
            #pragma unroll 1
            for (int tile = bid; tile < total_tiles; tile += num_bids) {
                int tile_m = tile / (GROUP_M * 168) * GROUP_M + (tile - tile / (GROUP_M * 168) * (GROUP_M * 168)) % ((num_m_tiles - tile / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 168) * GROUP_M : GROUP_M);
                int tile_n = (tile - tile / (GROUP_M * 168) * (GROUP_M * 168)) / ((num_m_tiles - tile / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 168) * GROUP_M : GROUP_M);
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int k_tile = 0; k_tile < 21; k_tile++) {
                        mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                        mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 36864);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(A_stage_addr + load_stage * 16384), "l"((&A)), "r"(k_tile * 128), "r"(tile_m * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(B_stage_addr + load_stage * 16384), "l"((&B)), "r"(k_tile * 128), "r"(tile_n * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(SFA_stage_addr + load_stage * 2048), "l"((&SFA)), "r"(k_tile * 16), "r"(tile_m * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(SFB_stage_addr + load_stage * 2048), "l"((&SFB)), "r"(0), "r"(tile_n * 168 + k_tile * 8), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        load_stage += 1;
                        if (load_stage == 2) { load_stage = 0; _phase_ab_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 1 && warp <= 8) {
        { // mma_main
            unsigned int mma_stage = 0;
            int warp_id_in_role = (warp - 1);
            int warp_m = warp_id_in_role % 4;
            int warp_n = warp_id_in_role / 4;
            int role_tid = warp_id_in_role * 32 + lane;
            float accum[64];
            unsigned int a_frag[8];
            unsigned int b_frag[16];
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 168) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 168) * (GROUP_M * 168)) % ((num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 168) * (GROUP_M * 168)) / ((num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M : GROUP_M);
                accum[0] = 0.0f;
                accum[1] = 0.0f;
                accum[2] = 0.0f;
                accum[3] = 0.0f;
                accum[4] = 0.0f;
                accum[5] = 0.0f;
                accum[6] = 0.0f;
                accum[7] = 0.0f;
                accum[8] = 0.0f;
                accum[9] = 0.0f;
                accum[10] = 0.0f;
                accum[11] = 0.0f;
                accum[12] = 0.0f;
                accum[13] = 0.0f;
                accum[14] = 0.0f;
                accum[15] = 0.0f;
                accum[16] = 0.0f;
                accum[17] = 0.0f;
                accum[18] = 0.0f;
                accum[19] = 0.0f;
                accum[20] = 0.0f;
                accum[21] = 0.0f;
                accum[22] = 0.0f;
                accum[23] = 0.0f;
                accum[24] = 0.0f;
                accum[25] = 0.0f;
                accum[26] = 0.0f;
                accum[27] = 0.0f;
                accum[28] = 0.0f;
                accum[29] = 0.0f;
                accum[30] = 0.0f;
                accum[31] = 0.0f;
                accum[32] = 0.0f;
                accum[33] = 0.0f;
                accum[34] = 0.0f;
                accum[35] = 0.0f;
                accum[36] = 0.0f;
                accum[37] = 0.0f;
                accum[38] = 0.0f;
                accum[39] = 0.0f;
                accum[40] = 0.0f;
                accum[41] = 0.0f;
                accum[42] = 0.0f;
                accum[43] = 0.0f;
                accum[44] = 0.0f;
                accum[45] = 0.0f;
                accum[46] = 0.0f;
                accum[47] = 0.0f;
                accum[48] = 0.0f;
                accum[49] = 0.0f;
                accum[50] = 0.0f;
                accum[51] = 0.0f;
                accum[52] = 0.0f;
                accum[53] = 0.0f;
                accum[54] = 0.0f;
                accum[55] = 0.0f;
                accum[56] = 0.0f;
                accum[57] = 0.0f;
                accum[58] = 0.0f;
                accum[59] = 0.0f;
                accum[60] = 0.0f;
                accum[61] = 0.0f;
                accum[62] = 0.0f;
                accum[63] = 0.0f;
                #pragma unroll 1
                for (int k_tile_1 = 0; k_tile_1 < 21; k_tile_1++) {
                    mbarrier_wait(ab_full_addr + (mma_stage) * 8, _phase_ab_full);
                    for (int k_step = 0; k_step < 4; k_step++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        unsigned int _SFA_stage_reg_0[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFA_stage_reg_0[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)((warp_m * 16 + (lane & 1) * 8 + (lane >> 2)) * 4) + (unsigned int)k_step) + _lr];
                        }
                        unsigned int _SFA_stage_reg_1[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFA_stage_reg_1[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)((warp_m * 16 + 64 + (lane & 1) * 8 + (lane >> 2)) * 4) + (unsigned int)k_step) + _lr];
                        }
                        unsigned int _SFB_stage_reg_0[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_0[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_1[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_1[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_2[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_2[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_3[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_3[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_4[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_4[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_5[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_5[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_6[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_6[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_7[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_7[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                        }
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 2) { mma_stage = 0; _phase_ab_full ^= 1; }
                }
                int kind = tile_n_1 % 3;
                int head = tile_n_1 / 3;
                int col_base = tile_n_1 * 128 + warp_n * 64 + (lane & 3) * 2;
                float sw[16];
                float nw[16];
                float out_scale_sel = ((kind == 0) ? out_scale_q : ((kind == 1) ? out_scale_k : out_scale_v));
                float sf_mul_sel = ((kind == 0) ? sf_mul_q : ((kind == 1) ? sf_mul_k : sf_mul_v));
                sw[0] = alpha;
                sw[1] = alpha;
                sw[2] = alpha;
                sw[3] = alpha;
                sw[4] = alpha;
                sw[5] = alpha;
                sw[6] = alpha;
                sw[7] = alpha;
                sw[8] = alpha;
                sw[9] = alpha;
                sw[10] = alpha;
                sw[11] = alpha;
                sw[12] = alpha;
                sw[13] = alpha;
                sw[14] = alpha;
                sw[15] = alpha;
                for (int mma_n = 0; mma_n < 8; mma_n++) {
                    int d_col = warp_n * 64 + mma_n * 8 + (lane & 3) * 2;
                    if (kind == 0) {
                        nw[mma_n * 2] = (float)q_norm_weight[d_col];
                        nw[mma_n * 2 + 1] = (float)q_norm_weight[d_col + 1];
                    } else if (kind == 1) {
                        nw[mma_n * 2] = (float)k_norm_weight[d_col];
                        nw[mma_n * 2 + 1] = (float)k_norm_weight[d_col + 1];
                    } else {
                        nw[mma_n * 2] = 1.0f;
                        nw[mma_n * 2 + 1] = 1.0f;
                    }
                }
                for (int half = 0; half < 2; half++) {
                    int row_lo = tile_m_1 * 128 + half * 64 + warp_m * 16 + (lane >> 2);
                    int row_hi = row_lo + 8;
                    float sa_lo = 1.0f;
                    float sa_hi = 1.0f;
                    float rounded[32];
                    float ss_lo = 0.0f;
                    float ss_hi = 0.0f;
                    for (int mma_n_1 = 0; mma_n_1 < 8; mma_n_1++) {
                        float scaled[4];
                        scaled[0] = accum[(half * 8 + mma_n_1) * 4] * sa_lo * sw[mma_n_1 * 2];
                        scaled[1] = accum[(half * 8 + mma_n_1) * 4 + 1] * sa_lo * sw[mma_n_1 * 2 + 1];
                        scaled[2] = accum[(half * 8 + mma_n_1) * 4 + 2] * sa_hi * sw[mma_n_1 * 2];
                        scaled[3] = accum[(half * 8 + mma_n_1) * 4 + 3] * sa_hi * sw[mma_n_1 * 2 + 1];
                        uint32_t scaled_bf16[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                            scaled_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        float scaled_bf16_f32[4];
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&scaled_bf16_f32[_pair * 2])[0]), "=f"((&scaled_bf16_f32[_pair * 2])[1])
                                : "r"(scaled_bf16[_pair]));
                        }
                        for (int e = 0; e < 4; e++) {
                            rounded[mma_n_1 * 4 + e] = scaled_bf16_f32[e];
                        }
                        ss_lo += scaled_bf16_f32[0] * scaled_bf16_f32[0] + scaled_bf16_f32[1] * scaled_bf16_f32[1];
                        ss_hi += scaled_bf16_f32[2] * scaled_bf16_f32[2] + scaled_bf16_f32[3] * scaled_bf16_f32[3];
                    }
                    float rstd_lo = 1.0f;
                    float rstd_hi = 1.0f;
                    if (kind < 2) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, ss_lo, 1);
                        ss_lo += _shfl_xor_0;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo, 2);
                        ss_lo += _shfl_xor_1;
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, ss_hi, 1);
                        ss_hi += _shfl_xor_2;
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, ss_hi, 2);
                        ss_hi += _shfl_xor_3;
                        int srow_lo = warp_m * 16 + (lane >> 2);
                        if ((lane & 3) == 0) {
                            rowstat[warp_n * 64 + srow_lo] = ss_lo;
                            rowstat[warp_n * 64 + srow_lo + 8] = ss_hi;
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        float other_lo = rowstat[(1 - warp_n) * 64 + srow_lo];
                        float other_hi = rowstat[(1 - warp_n) * 64 + srow_lo + 8];
                        float _fdiv_rn_0 = __fdiv_rn(ss_lo + other_lo, 128.0f);
                        float _rsqrt_0 = rsqrtf(_fdiv_rn_0 + eps);
                        rstd_lo = _rsqrt_0;
                        float _fdiv_rn_1 = __fdiv_rn(ss_hi + other_hi, 128.0f);
                        float _rsqrt_1 = rsqrtf(_fdiv_rn_1 + eps);
                        rstd_hi = _rsqrt_1;
                    }
                    int srow_st = warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                    for (int mma_n_2 = 0; mma_n_2 < 8; mma_n_2++) {
                        float normed[4];
                        if (kind < 2) {
                            normed[0] = rounded[mma_n_2 * 4] * rstd_lo * nw[mma_n_2 * 2];
                            normed[1] = rounded[mma_n_2 * 4 + 1] * rstd_lo * nw[mma_n_2 * 2 + 1];
                            normed[2] = rounded[mma_n_2 * 4 + 2] * rstd_hi * nw[mma_n_2 * 2];
                            normed[3] = rounded[mma_n_2 * 4 + 3] * rstd_hi * nw[mma_n_2 * 2 + 1];
                        } else {
                            normed[0] = rounded[mma_n_2 * 4];
                            normed[1] = rounded[mma_n_2 * 4 + 1];
                            normed[2] = rounded[mma_n_2 * 4 + 2];
                            normed[3] = rounded[mma_n_2 * 4 + 3];
                        }
                        uint32_t normed_bf16[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed[_lp*2 + 0], normed[_lp*2+1 + 0]));
                            normed_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + mma_n_2 * 8) * 2 ^ (srow_st & 7) << 4));
                        asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16[1]))
                            : "memory");
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                    for (int it = 0; it < 2; it++) {
                        int task = role_tid + it * 256;
                        if (task < 320) {
                            int srow = task / 5;
                            int c = task % 5;
                            int grow = tile_m_1 * 128 + half * 64 + srow;
                            if (grow < M) {
                                int row_head = grow * 56 + head;
                                unsigned int st_words[4];
                                float st_vals[16];
                                if (c < 3) {
                                    unsigned int x_lo[8];
                                    unsigned int x_hi[8];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 16 * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 16 * 2 + 16 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((c * 16 + 48) * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((c * 16 + 48) * 2 + 16 ^ (srow & 7) << 4)));
                                    if (kind < 2) {
                                        float x_lo_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_lo_f32[_pair * 2])[0]), "=f"((&x_lo_f32[_pair * 2])[1])
                                                : "r"(x_lo[_pair]));
                                        }
                                        float x_hi_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_hi_f32[_pair * 2])[0]), "=f"((&x_hi_f32[_pair * 2])[1])
                                                : "r"(x_hi[_pair]));
                                        }
                                        float _vec_load_0[8];
                                        {
                                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + c * 16) + 0);
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
                                                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_1[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_1[8];
                                        {
                                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + c * 16 + 8) + 0);
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
                                                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_2[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_2[8];
                                        {
                                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + 48 + c * 16) + 0);
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
                                                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_3[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_3[8];
                                        {
                                            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + 48 + c * 16 + 8) + 0);
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
                                                        : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_4[_pair]));
                                                }
                                            }
                                        }
                                        float o_lo[16];
                                        float o_hi[16];
                                        for (int e_1 = 0; e_1 < 8; e_1++) {
                                            float cv0 = _vec_load_0[e_1];
                                            float sv0 = _vec_load_2[e_1];
                                            float cv1 = _vec_load_1[e_1];
                                            float sv1 = _vec_load_3[e_1];
                                            o_lo[e_1] = x_lo_f32[e_1] * cv0 - x_hi_f32[e_1] * sv0;
                                            o_hi[e_1] = x_hi_f32[e_1] * cv0 + x_lo_f32[e_1] * sv0;
                                            o_lo[8 + e_1] = x_lo_f32[8 + e_1] * cv1 - x_hi_f32[8 + e_1] * sv1;
                                            o_hi[8 + e_1] = x_hi_f32[8 + e_1] * cv1 + x_lo_f32[8 + e_1] * sv1;
                                        }
                                        uint32_t o_lo_bf16[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_lo[_lp*2 + 0], o_lo[_lp*2+1 + 0]));
                                            o_lo_bf16[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        uint32_t o_hi_bf16[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_hi[_lp*2 + 0], o_hi[_lp*2+1 + 0]));
                                            o_hi_bf16[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        if (kind == 0) {
                                            float o_lo_bf16_f32[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_lo_bf16_f32[_pair * 2])[0]), "=f"((&o_lo_bf16_f32[_pair * 2])[1])
                                                    : "r"(o_lo_bf16[_pair]));
                                            }
                                            st_vals[0] = o_lo_bf16_f32[0] * out_scale_sel;
                                            st_vals[1] = o_lo_bf16_f32[1] * out_scale_sel;
                                            st_vals[2] = o_lo_bf16_f32[2] * out_scale_sel;
                                            st_vals[3] = o_lo_bf16_f32[3] * out_scale_sel;
                                            st_vals[4] = o_lo_bf16_f32[4] * out_scale_sel;
                                            st_vals[5] = o_lo_bf16_f32[5] * out_scale_sel;
                                            st_vals[6] = o_lo_bf16_f32[6] * out_scale_sel;
                                            st_vals[7] = o_lo_bf16_f32[7] * out_scale_sel;
                                            st_vals[8] = o_lo_bf16_f32[8] * out_scale_sel;
                                            st_vals[9] = o_lo_bf16_f32[9] * out_scale_sel;
                                            st_vals[10] = o_lo_bf16_f32[10] * out_scale_sel;
                                            st_vals[11] = o_lo_bf16_f32[11] * out_scale_sel;
                                            st_vals[12] = o_lo_bf16_f32[12] * out_scale_sel;
                                            st_vals[13] = o_lo_bf16_f32[13] * out_scale_sel;
                                            st_vals[14] = o_lo_bf16_f32[14] * out_scale_sel;
                                            st_vals[15] = o_lo_bf16_f32[15] * out_scale_sel;
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                       "f"(st_vals[2]), "f"(st_vals[3]));
                                                st_words[0] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                       "f"(st_vals[6]), "f"(st_vals[7]));
                                                st_words[1] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                       "f"(st_vals[10]), "f"(st_vals[11]));
                                                st_words[2] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                       "f"(st_vals[14]), "f"(st_vals[15]));
                                                st_words[3] = _packed;
                                            }
                                            reinterpret_cast<int4*>(q_out + (row_head * 32 + (c * 16 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                            float o_hi_bf16_f32[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_hi_bf16_f32[_pair * 2])[0]), "=f"((&o_hi_bf16_f32[_pair * 2])[1])
                                                    : "r"(o_hi_bf16[_pair]));
                                            }
                                            st_vals[0] = o_hi_bf16_f32[0] * out_scale_sel;
                                            st_vals[1] = o_hi_bf16_f32[1] * out_scale_sel;
                                            st_vals[2] = o_hi_bf16_f32[2] * out_scale_sel;
                                            st_vals[3] = o_hi_bf16_f32[3] * out_scale_sel;
                                            st_vals[4] = o_hi_bf16_f32[4] * out_scale_sel;
                                            st_vals[5] = o_hi_bf16_f32[5] * out_scale_sel;
                                            st_vals[6] = o_hi_bf16_f32[6] * out_scale_sel;
                                            st_vals[7] = o_hi_bf16_f32[7] * out_scale_sel;
                                            st_vals[8] = o_hi_bf16_f32[8] * out_scale_sel;
                                            st_vals[9] = o_hi_bf16_f32[9] * out_scale_sel;
                                            st_vals[10] = o_hi_bf16_f32[10] * out_scale_sel;
                                            st_vals[11] = o_hi_bf16_f32[11] * out_scale_sel;
                                            st_vals[12] = o_hi_bf16_f32[12] * out_scale_sel;
                                            st_vals[13] = o_hi_bf16_f32[13] * out_scale_sel;
                                            st_vals[14] = o_hi_bf16_f32[14] * out_scale_sel;
                                            st_vals[15] = o_hi_bf16_f32[15] * out_scale_sel;
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                       "f"(st_vals[2]), "f"(st_vals[3]));
                                                st_words[0] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                       "f"(st_vals[6]), "f"(st_vals[7]));
                                                st_words[1] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                       "f"(st_vals[10]), "f"(st_vals[11]));
                                                st_words[2] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                       "f"(st_vals[14]), "f"(st_vals[15]));
                                                st_words[3] = _packed;
                                            }
                                            reinterpret_cast<int4*>(q_out + (row_head * 32 + (c * 16 + 48 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                        } else {
                                            float o_lo_bf16_f32_1[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_lo_bf16_f32_1[_pair * 2])[0]), "=f"((&o_lo_bf16_f32_1[_pair * 2])[1])
                                                    : "r"(o_lo_bf16[_pair]));
                                            }
                                            st_vals[0] = o_lo_bf16_f32_1[0] * out_scale_sel;
                                            st_vals[1] = o_lo_bf16_f32_1[1] * out_scale_sel;
                                            st_vals[2] = o_lo_bf16_f32_1[2] * out_scale_sel;
                                            st_vals[3] = o_lo_bf16_f32_1[3] * out_scale_sel;
                                            st_vals[4] = o_lo_bf16_f32_1[4] * out_scale_sel;
                                            st_vals[5] = o_lo_bf16_f32_1[5] * out_scale_sel;
                                            st_vals[6] = o_lo_bf16_f32_1[6] * out_scale_sel;
                                            st_vals[7] = o_lo_bf16_f32_1[7] * out_scale_sel;
                                            st_vals[8] = o_lo_bf16_f32_1[8] * out_scale_sel;
                                            st_vals[9] = o_lo_bf16_f32_1[9] * out_scale_sel;
                                            st_vals[10] = o_lo_bf16_f32_1[10] * out_scale_sel;
                                            st_vals[11] = o_lo_bf16_f32_1[11] * out_scale_sel;
                                            st_vals[12] = o_lo_bf16_f32_1[12] * out_scale_sel;
                                            st_vals[13] = o_lo_bf16_f32_1[13] * out_scale_sel;
                                            st_vals[14] = o_lo_bf16_f32_1[14] * out_scale_sel;
                                            st_vals[15] = o_lo_bf16_f32_1[15] * out_scale_sel;
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                       "f"(st_vals[2]), "f"(st_vals[3]));
                                                st_words[0] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                       "f"(st_vals[6]), "f"(st_vals[7]));
                                                st_words[1] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                       "f"(st_vals[10]), "f"(st_vals[11]));
                                                st_words[2] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                       "f"(st_vals[14]), "f"(st_vals[15]));
                                                st_words[3] = _packed;
                                            }
                                            reinterpret_cast<int4*>(k_out + (row_head * 32 + (c * 16 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                            float o_hi_bf16_f32_1[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_hi_bf16_f32_1[_pair * 2])[0]), "=f"((&o_hi_bf16_f32_1[_pair * 2])[1])
                                                    : "r"(o_hi_bf16[_pair]));
                                            }
                                            st_vals[0] = o_hi_bf16_f32_1[0] * out_scale_sel;
                                            st_vals[1] = o_hi_bf16_f32_1[1] * out_scale_sel;
                                            st_vals[2] = o_hi_bf16_f32_1[2] * out_scale_sel;
                                            st_vals[3] = o_hi_bf16_f32_1[3] * out_scale_sel;
                                            st_vals[4] = o_hi_bf16_f32_1[4] * out_scale_sel;
                                            st_vals[5] = o_hi_bf16_f32_1[5] * out_scale_sel;
                                            st_vals[6] = o_hi_bf16_f32_1[6] * out_scale_sel;
                                            st_vals[7] = o_hi_bf16_f32_1[7] * out_scale_sel;
                                            st_vals[8] = o_hi_bf16_f32_1[8] * out_scale_sel;
                                            st_vals[9] = o_hi_bf16_f32_1[9] * out_scale_sel;
                                            st_vals[10] = o_hi_bf16_f32_1[10] * out_scale_sel;
                                            st_vals[11] = o_hi_bf16_f32_1[11] * out_scale_sel;
                                            st_vals[12] = o_hi_bf16_f32_1[12] * out_scale_sel;
                                            st_vals[13] = o_hi_bf16_f32_1[13] * out_scale_sel;
                                            st_vals[14] = o_hi_bf16_f32_1[14] * out_scale_sel;
                                            st_vals[15] = o_hi_bf16_f32_1[15] * out_scale_sel;
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                       "f"(st_vals[2]), "f"(st_vals[3]));
                                                st_words[0] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                       "f"(st_vals[6]), "f"(st_vals[7]));
                                                st_words[1] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                       "f"(st_vals[10]), "f"(st_vals[11]));
                                                st_words[2] = _packed;
                                            }
                                            {
                                                uint32_t _packed;
                                                asm volatile("{\n\t"
                                                    ".reg .b16 _lo;\n\t"
                                                    ".reg .b16 _hi;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                                    "}"
                                                    : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                       "f"(st_vals[14]), "f"(st_vals[15]));
                                                st_words[3] = _packed;
                                            }
                                            reinterpret_cast<int4*>(k_out + (row_head * 32 + (c * 16 + 48 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                        }
                                    } else {
                                        float x_lo_f32_1[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_lo_f32_1[_pair * 2])[0]), "=f"((&x_lo_f32_1[_pair * 2])[1])
                                                : "r"(x_lo[_pair]));
                                        }
                                        st_vals[0] = x_lo_f32_1[0] * out_scale_sel;
                                        st_vals[1] = x_lo_f32_1[1] * out_scale_sel;
                                        st_vals[2] = x_lo_f32_1[2] * out_scale_sel;
                                        st_vals[3] = x_lo_f32_1[3] * out_scale_sel;
                                        st_vals[4] = x_lo_f32_1[4] * out_scale_sel;
                                        st_vals[5] = x_lo_f32_1[5] * out_scale_sel;
                                        st_vals[6] = x_lo_f32_1[6] * out_scale_sel;
                                        st_vals[7] = x_lo_f32_1[7] * out_scale_sel;
                                        st_vals[8] = x_lo_f32_1[8] * out_scale_sel;
                                        st_vals[9] = x_lo_f32_1[9] * out_scale_sel;
                                        st_vals[10] = x_lo_f32_1[10] * out_scale_sel;
                                        st_vals[11] = x_lo_f32_1[11] * out_scale_sel;
                                        st_vals[12] = x_lo_f32_1[12] * out_scale_sel;
                                        st_vals[13] = x_lo_f32_1[13] * out_scale_sel;
                                        st_vals[14] = x_lo_f32_1[14] * out_scale_sel;
                                        st_vals[15] = x_lo_f32_1[15] * out_scale_sel;
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                   "f"(st_vals[2]), "f"(st_vals[3]));
                                            st_words[0] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                   "f"(st_vals[6]), "f"(st_vals[7]));
                                            st_words[1] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                   "f"(st_vals[10]), "f"(st_vals[11]));
                                            st_words[2] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                   "f"(st_vals[14]), "f"(st_vals[15]));
                                            st_words[3] = _packed;
                                        }
                                        reinterpret_cast<int4*>(v_out + (row_head * 32 + (c * 16 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                        float x_hi_f32_1[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_hi_f32_1[_pair * 2])[0]), "=f"((&x_hi_f32_1[_pair * 2])[1])
                                                : "r"(x_hi[_pair]));
                                        }
                                        st_vals[0] = x_hi_f32_1[0] * out_scale_sel;
                                        st_vals[1] = x_hi_f32_1[1] * out_scale_sel;
                                        st_vals[2] = x_hi_f32_1[2] * out_scale_sel;
                                        st_vals[3] = x_hi_f32_1[3] * out_scale_sel;
                                        st_vals[4] = x_hi_f32_1[4] * out_scale_sel;
                                        st_vals[5] = x_hi_f32_1[5] * out_scale_sel;
                                        st_vals[6] = x_hi_f32_1[6] * out_scale_sel;
                                        st_vals[7] = x_hi_f32_1[7] * out_scale_sel;
                                        st_vals[8] = x_hi_f32_1[8] * out_scale_sel;
                                        st_vals[9] = x_hi_f32_1[9] * out_scale_sel;
                                        st_vals[10] = x_hi_f32_1[10] * out_scale_sel;
                                        st_vals[11] = x_hi_f32_1[11] * out_scale_sel;
                                        st_vals[12] = x_hi_f32_1[12] * out_scale_sel;
                                        st_vals[13] = x_hi_f32_1[13] * out_scale_sel;
                                        st_vals[14] = x_hi_f32_1[14] * out_scale_sel;
                                        st_vals[15] = x_hi_f32_1[15] * out_scale_sel;
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                   "f"(st_vals[2]), "f"(st_vals[3]));
                                            st_words[0] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                   "f"(st_vals[6]), "f"(st_vals[7]));
                                            st_words[1] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                   "f"(st_vals[10]), "f"(st_vals[11]));
                                            st_words[2] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                   "f"(st_vals[14]), "f"(st_vals[15]));
                                            st_words[3] = _packed;
                                        }
                                        reinterpret_cast<int4*>(v_out + (row_head * 32 + (c * 16 + 48 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                    }
                                } else {
                                    unsigned int x_pt[8];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_pt[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((96 + (c - 3) * 16) * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_pt[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((96 + (c - 3) * 16) * 2 + 16 ^ (srow & 7) << 4)));
                                    if (kind == 0) {
                                        float x_pt_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_pt_f32[_pair * 2])[0]), "=f"((&x_pt_f32[_pair * 2])[1])
                                                : "r"(x_pt[_pair]));
                                        }
                                        st_vals[0] = x_pt_f32[0] * out_scale_sel;
                                        st_vals[1] = x_pt_f32[1] * out_scale_sel;
                                        st_vals[2] = x_pt_f32[2] * out_scale_sel;
                                        st_vals[3] = x_pt_f32[3] * out_scale_sel;
                                        st_vals[4] = x_pt_f32[4] * out_scale_sel;
                                        st_vals[5] = x_pt_f32[5] * out_scale_sel;
                                        st_vals[6] = x_pt_f32[6] * out_scale_sel;
                                        st_vals[7] = x_pt_f32[7] * out_scale_sel;
                                        st_vals[8] = x_pt_f32[8] * out_scale_sel;
                                        st_vals[9] = x_pt_f32[9] * out_scale_sel;
                                        st_vals[10] = x_pt_f32[10] * out_scale_sel;
                                        st_vals[11] = x_pt_f32[11] * out_scale_sel;
                                        st_vals[12] = x_pt_f32[12] * out_scale_sel;
                                        st_vals[13] = x_pt_f32[13] * out_scale_sel;
                                        st_vals[14] = x_pt_f32[14] * out_scale_sel;
                                        st_vals[15] = x_pt_f32[15] * out_scale_sel;
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                   "f"(st_vals[2]), "f"(st_vals[3]));
                                            st_words[0] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                   "f"(st_vals[6]), "f"(st_vals[7]));
                                            st_words[1] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                   "f"(st_vals[10]), "f"(st_vals[11]));
                                            st_words[2] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                   "f"(st_vals[14]), "f"(st_vals[15]));
                                            st_words[3] = _packed;
                                        }
                                        reinterpret_cast<int4*>(q_out + (row_head * 32 + (96 + (c - 3) * 16 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                    } else if (kind == 1) {
                                        float x_pt_f32_1[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_pt_f32_1[_pair * 2])[0]), "=f"((&x_pt_f32_1[_pair * 2])[1])
                                                : "r"(x_pt[_pair]));
                                        }
                                        st_vals[0] = x_pt_f32_1[0] * out_scale_sel;
                                        st_vals[1] = x_pt_f32_1[1] * out_scale_sel;
                                        st_vals[2] = x_pt_f32_1[2] * out_scale_sel;
                                        st_vals[3] = x_pt_f32_1[3] * out_scale_sel;
                                        st_vals[4] = x_pt_f32_1[4] * out_scale_sel;
                                        st_vals[5] = x_pt_f32_1[5] * out_scale_sel;
                                        st_vals[6] = x_pt_f32_1[6] * out_scale_sel;
                                        st_vals[7] = x_pt_f32_1[7] * out_scale_sel;
                                        st_vals[8] = x_pt_f32_1[8] * out_scale_sel;
                                        st_vals[9] = x_pt_f32_1[9] * out_scale_sel;
                                        st_vals[10] = x_pt_f32_1[10] * out_scale_sel;
                                        st_vals[11] = x_pt_f32_1[11] * out_scale_sel;
                                        st_vals[12] = x_pt_f32_1[12] * out_scale_sel;
                                        st_vals[13] = x_pt_f32_1[13] * out_scale_sel;
                                        st_vals[14] = x_pt_f32_1[14] * out_scale_sel;
                                        st_vals[15] = x_pt_f32_1[15] * out_scale_sel;
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                   "f"(st_vals[2]), "f"(st_vals[3]));
                                            st_words[0] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                   "f"(st_vals[6]), "f"(st_vals[7]));
                                            st_words[1] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                   "f"(st_vals[10]), "f"(st_vals[11]));
                                            st_words[2] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                   "f"(st_vals[14]), "f"(st_vals[15]));
                                            st_words[3] = _packed;
                                        }
                                        reinterpret_cast<int4*>(k_out + (row_head * 32 + (96 + (c - 3) * 16 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                    } else {
                                        float x_pt_f32_2[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_pt_f32_2[_pair * 2])[0]), "=f"((&x_pt_f32_2[_pair * 2])[1])
                                                : "r"(x_pt[_pair]));
                                        }
                                        st_vals[0] = x_pt_f32_2[0] * out_scale_sel;
                                        st_vals[1] = x_pt_f32_2[1] * out_scale_sel;
                                        st_vals[2] = x_pt_f32_2[2] * out_scale_sel;
                                        st_vals[3] = x_pt_f32_2[3] * out_scale_sel;
                                        st_vals[4] = x_pt_f32_2[4] * out_scale_sel;
                                        st_vals[5] = x_pt_f32_2[5] * out_scale_sel;
                                        st_vals[6] = x_pt_f32_2[6] * out_scale_sel;
                                        st_vals[7] = x_pt_f32_2[7] * out_scale_sel;
                                        st_vals[8] = x_pt_f32_2[8] * out_scale_sel;
                                        st_vals[9] = x_pt_f32_2[9] * out_scale_sel;
                                        st_vals[10] = x_pt_f32_2[10] * out_scale_sel;
                                        st_vals[11] = x_pt_f32_2[11] * out_scale_sel;
                                        st_vals[12] = x_pt_f32_2[12] * out_scale_sel;
                                        st_vals[13] = x_pt_f32_2[13] * out_scale_sel;
                                        st_vals[14] = x_pt_f32_2[14] * out_scale_sel;
                                        st_vals[15] = x_pt_f32_2[15] * out_scale_sel;
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[0]), "f"(st_vals[1]),
                                                                   "f"(st_vals[2]), "f"(st_vals[3]));
                                            st_words[0] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[4]), "f"(st_vals[5]),
                                                                   "f"(st_vals[6]), "f"(st_vals[7]));
                                            st_words[1] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[8]), "f"(st_vals[9]),
                                                                   "f"(st_vals[10]), "f"(st_vals[11]));
                                            st_words[2] = _packed;
                                        }
                                        {
                                            uint32_t _packed;
                                            asm volatile("{\n\t"
                                                ".reg .b16 _lo;\n\t"
                                                ".reg .b16 _hi;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                                "mov.b32 %0, {_lo, _hi};\n\t"
                                                "}"
                                                : "=r"(_packed) : "f"(st_vals[12]), "f"(st_vals[13]),
                                                                   "f"(st_vals[14]), "f"(st_vals[15]));
                                            st_words[3] = _packed;
                                        }
                                        reinterpret_cast<int4*>(v_out + (row_head * 32 + (96 + (c - 3) * 16 >> 2)))[0] = reinterpret_cast<int4*>(st_words)[0];
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                }
            }
        }
    }

    // Cleanup
}

}  // namespace h3_qkv_gemm_nvfp4_e4m3_sm120a
#undef GROUP_M
#undef H3_QPA_INF
#undef NUM_AB_PIPE_STAGES
#undef SMEM_A_STAGE_OFF
#undef SMEM_A_STAGE_STAGE_BYTES
#undef SMEM_A_STAGE_STRIDE
#undef SMEM_B_STAGE_OFF
#undef SMEM_B_STAGE_STAGE_BYTES
#undef SMEM_B_STAGE_STRIDE
#undef SMEM_ROWSTAT_OFF
#undef SMEM_ROWSTAT_STAGE_BYTES
#undef SMEM_ROWSTAT_STRIDE
#undef SMEM_SFA_STAGE_OFF
#undef SMEM_SFA_STAGE_STAGE_BYTES
#undef SMEM_SFA_STAGE_STRIDE
#undef SMEM_SFB_STAGE_OFF
#undef SMEM_SFB_STAGE_STAGE_BYTES
#undef SMEM_SFB_STAGE_STRIDE
#undef SMEM_STAGING_OFF
#undef SMEM_STAGING_STAGE_BYTES
#undef SMEM_STAGING_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef ab_empty_addr
#undef ab_full_addr

namespace h3_qkv_gemm_nvfp4_nvfp4_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_QPA_INF CUDART_INF_F
#define NUM_AB_PIPE_STAGES 2
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 16384
#define SMEM_A_STAGE_STRIDE 16384
#define SMEM_B_STAGE_OFF 33792
#define SMEM_B_STAGE_STAGE_BYTES 16384
#define SMEM_B_STAGE_STRIDE 16384
#define SMEM_STAGING_OFF 66560
#define SMEM_STAGING_STAGE_BYTES 16384
#define SMEM_STAGING_STRIDE 16384
#define SMEM_ROWSTAT_OFF 82944
#define SMEM_ROWSTAT_STAGE_BYTES 512
#define SMEM_ROWSTAT_STRIDE 512
#define SMEM_SFA_STAGE_OFF 83456
#define SMEM_SFA_STAGE_STAGE_BYTES 2048
#define SMEM_SFA_STAGE_STRIDE 2048
#define SMEM_SFB_STAGE_OFF 87552
#define SMEM_SFB_STAGE_STAGE_BYTES 2048
#define SMEM_SFB_STAGE_STRIDE 2048
#define SMEM_TOTAL 91648
#define THREADS 288
#define GROUP_M 16

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}


__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__global__ __launch_bounds__(288, 1) void
kernel_h3_qkv_gemm_fused(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ act_scale, float* __restrict__ w_scale, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, unsigned int* __restrict__ q_out, unsigned int* __restrict__ k_out, unsigned int* __restrict__ v_out, uint8_t* __restrict__ q_sf, uint8_t* __restrict__ k_sf, uint8_t* __restrict__ v_sf, int M, int num_m_tiles, int total_tiles, float eps, float alpha, float out_scale_q, float out_scale_k, float out_scale_v, float sf_mul_q, float sf_mul_k, float sf_mul_v)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int B_stage_addr = smem + 33792;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int staging_addr = smem + 66560;
    float* rowstat = reinterpret_cast<float*>(smem_raw + 82944);
    const int rowstat_addr = smem + 82944;
    unsigned int* SFA_stage = reinterpret_cast<unsigned int*>(smem_raw + 83456);
    const int SFA_stage_addr = smem + 83456;
    unsigned int* SFB_stage = reinterpret_cast<unsigned int*>(smem_raw + 87552);
    const int SFB_stage_addr = smem + 87552;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // ab_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 16, 8);
            mbarrier_init(smem + 24, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: producer ----
    if (warp == 0) {
        { // producer_main
            unsigned int load_stage = 0;
            unsigned int _phase_ab_empty = 1;
            #pragma unroll 1
            for (int tile = bid; tile < total_tiles; tile += num_bids) {
                int tile_m = tile / (GROUP_M * 168) * GROUP_M + (tile - tile / (GROUP_M * 168) * (GROUP_M * 168)) % ((num_m_tiles - tile / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 168) * GROUP_M : GROUP_M);
                int tile_n = (tile - tile / (GROUP_M * 168) * (GROUP_M * 168)) / ((num_m_tiles - tile / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 168) * GROUP_M : GROUP_M);
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int k_tile = 0; k_tile < 21; k_tile++) {
                        mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                        mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 36864);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(A_stage_addr + load_stage * 16384), "l"((&A)), "r"(k_tile * 128), "r"(tile_m * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(B_stage_addr + load_stage * 16384), "l"((&B)), "r"(k_tile * 128), "r"(tile_n * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(SFA_stage_addr + load_stage * 2048), "l"((&SFA)), "r"(k_tile * 16), "r"(tile_m * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(SFB_stage_addr + load_stage * 2048), "l"((&SFB)), "r"(0), "r"(tile_n * 168 + k_tile * 8), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        load_stage += 1;
                        if (load_stage == 2) { load_stage = 0; _phase_ab_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 1 && warp <= 8) {
        { // mma_main
            unsigned int mma_stage = 0;
            int warp_id_in_role = (warp - 1);
            int warp_m = warp_id_in_role % 4;
            int warp_n = warp_id_in_role / 4;
            int role_tid = warp_id_in_role * 32 + lane;
            float accum[64];
            unsigned int a_frag[8];
            unsigned int b_frag[16];
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 168) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 168) * (GROUP_M * 168)) % ((num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 168) * (GROUP_M * 168)) / ((num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 168) * GROUP_M : GROUP_M);
                accum[0] = 0.0f;
                accum[1] = 0.0f;
                accum[2] = 0.0f;
                accum[3] = 0.0f;
                accum[4] = 0.0f;
                accum[5] = 0.0f;
                accum[6] = 0.0f;
                accum[7] = 0.0f;
                accum[8] = 0.0f;
                accum[9] = 0.0f;
                accum[10] = 0.0f;
                accum[11] = 0.0f;
                accum[12] = 0.0f;
                accum[13] = 0.0f;
                accum[14] = 0.0f;
                accum[15] = 0.0f;
                accum[16] = 0.0f;
                accum[17] = 0.0f;
                accum[18] = 0.0f;
                accum[19] = 0.0f;
                accum[20] = 0.0f;
                accum[21] = 0.0f;
                accum[22] = 0.0f;
                accum[23] = 0.0f;
                accum[24] = 0.0f;
                accum[25] = 0.0f;
                accum[26] = 0.0f;
                accum[27] = 0.0f;
                accum[28] = 0.0f;
                accum[29] = 0.0f;
                accum[30] = 0.0f;
                accum[31] = 0.0f;
                accum[32] = 0.0f;
                accum[33] = 0.0f;
                accum[34] = 0.0f;
                accum[35] = 0.0f;
                accum[36] = 0.0f;
                accum[37] = 0.0f;
                accum[38] = 0.0f;
                accum[39] = 0.0f;
                accum[40] = 0.0f;
                accum[41] = 0.0f;
                accum[42] = 0.0f;
                accum[43] = 0.0f;
                accum[44] = 0.0f;
                accum[45] = 0.0f;
                accum[46] = 0.0f;
                accum[47] = 0.0f;
                accum[48] = 0.0f;
                accum[49] = 0.0f;
                accum[50] = 0.0f;
                accum[51] = 0.0f;
                accum[52] = 0.0f;
                accum[53] = 0.0f;
                accum[54] = 0.0f;
                accum[55] = 0.0f;
                accum[56] = 0.0f;
                accum[57] = 0.0f;
                accum[58] = 0.0f;
                accum[59] = 0.0f;
                accum[60] = 0.0f;
                accum[61] = 0.0f;
                accum[62] = 0.0f;
                accum[63] = 0.0f;
                #pragma unroll 1
                for (int k_tile_1 = 0; k_tile_1 < 21; k_tile_1++) {
                    mbarrier_wait(ab_full_addr + (mma_stage) * 8, _phase_ab_full);
                    for (int k_step = 0; k_step < 4; k_step++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                            : "memory");
                        unsigned int _SFA_stage_reg_0[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFA_stage_reg_0[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)((warp_m * 16 + (lane & 1) * 8 + (lane >> 2)) * 4) + (unsigned int)k_step) + _lr];
                        }
                        unsigned int _SFA_stage_reg_1[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFA_stage_reg_1[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)((warp_m * 16 + 64 + (lane & 1) * 8 + (lane >> 2)) * 4) + (unsigned int)k_step) + _lr];
                        }
                        unsigned int _SFB_stage_reg_0[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_0[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_1[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_1[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_2[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_2[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_3[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_3[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_4[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_4[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_5[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_5[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_6[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_6[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_7[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_7[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(k_step * 128) + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                        }
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 2) { mma_stage = 0; _phase_ab_full ^= 1; }
                }
                int kind = tile_n_1 % 3;
                int head = tile_n_1 / 3;
                int col_base = tile_n_1 * 128 + warp_n * 64 + (lane & 3) * 2;
                float sw[16];
                float nw[16];
                float out_scale_sel = ((kind == 0) ? out_scale_q : ((kind == 1) ? out_scale_k : out_scale_v));
                float sf_mul_sel = ((kind == 0) ? sf_mul_q : ((kind == 1) ? sf_mul_k : sf_mul_v));
                sw[0] = alpha;
                sw[1] = alpha;
                sw[2] = alpha;
                sw[3] = alpha;
                sw[4] = alpha;
                sw[5] = alpha;
                sw[6] = alpha;
                sw[7] = alpha;
                sw[8] = alpha;
                sw[9] = alpha;
                sw[10] = alpha;
                sw[11] = alpha;
                sw[12] = alpha;
                sw[13] = alpha;
                sw[14] = alpha;
                sw[15] = alpha;
                for (int mma_n = 0; mma_n < 8; mma_n++) {
                    int d_col = warp_n * 64 + mma_n * 8 + (lane & 3) * 2;
                    if (kind == 0) {
                        nw[mma_n * 2] = (float)q_norm_weight[d_col];
                        nw[mma_n * 2 + 1] = (float)q_norm_weight[d_col + 1];
                    } else if (kind == 1) {
                        nw[mma_n * 2] = (float)k_norm_weight[d_col];
                        nw[mma_n * 2 + 1] = (float)k_norm_weight[d_col + 1];
                    } else {
                        nw[mma_n * 2] = 1.0f;
                        nw[mma_n * 2 + 1] = 1.0f;
                    }
                }
                for (int half = 0; half < 2; half++) {
                    int row_lo = tile_m_1 * 128 + half * 64 + warp_m * 16 + (lane >> 2);
                    int row_hi = row_lo + 8;
                    float sa_lo = 1.0f;
                    float sa_hi = 1.0f;
                    float rounded[32];
                    float ss_lo = 0.0f;
                    float ss_hi = 0.0f;
                    for (int mma_n_1 = 0; mma_n_1 < 8; mma_n_1++) {
                        float scaled[4];
                        scaled[0] = accum[(half * 8 + mma_n_1) * 4] * sa_lo * sw[mma_n_1 * 2];
                        scaled[1] = accum[(half * 8 + mma_n_1) * 4 + 1] * sa_lo * sw[mma_n_1 * 2 + 1];
                        scaled[2] = accum[(half * 8 + mma_n_1) * 4 + 2] * sa_hi * sw[mma_n_1 * 2];
                        scaled[3] = accum[(half * 8 + mma_n_1) * 4 + 3] * sa_hi * sw[mma_n_1 * 2 + 1];
                        uint32_t scaled_bf16[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                            scaled_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        float scaled_bf16_f32[4];
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&scaled_bf16_f32[_pair * 2])[0]), "=f"((&scaled_bf16_f32[_pair * 2])[1])
                                : "r"(scaled_bf16[_pair]));
                        }
                        for (int e = 0; e < 4; e++) {
                            rounded[mma_n_1 * 4 + e] = scaled_bf16_f32[e];
                        }
                        ss_lo += scaled_bf16_f32[0] * scaled_bf16_f32[0] + scaled_bf16_f32[1] * scaled_bf16_f32[1];
                        ss_hi += scaled_bf16_f32[2] * scaled_bf16_f32[2] + scaled_bf16_f32[3] * scaled_bf16_f32[3];
                    }
                    float rstd_lo = 1.0f;
                    float rstd_hi = 1.0f;
                    if (kind < 2) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, ss_lo, 1);
                        ss_lo += _shfl_xor_0;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo, 2);
                        ss_lo += _shfl_xor_1;
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, ss_hi, 1);
                        ss_hi += _shfl_xor_2;
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, ss_hi, 2);
                        ss_hi += _shfl_xor_3;
                        int srow_lo = warp_m * 16 + (lane >> 2);
                        if ((lane & 3) == 0) {
                            rowstat[warp_n * 64 + srow_lo] = ss_lo;
                            rowstat[warp_n * 64 + srow_lo + 8] = ss_hi;
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        float other_lo = rowstat[(1 - warp_n) * 64 + srow_lo];
                        float other_hi = rowstat[(1 - warp_n) * 64 + srow_lo + 8];
                        float _fdiv_rn_0 = __fdiv_rn(ss_lo + other_lo, 128.0f);
                        float _rsqrt_0 = rsqrtf(_fdiv_rn_0 + eps);
                        rstd_lo = _rsqrt_0;
                        float _fdiv_rn_1 = __fdiv_rn(ss_hi + other_hi, 128.0f);
                        float _rsqrt_1 = rsqrtf(_fdiv_rn_1 + eps);
                        rstd_hi = _rsqrt_1;
                    }
                    int srow_st = warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                    for (int mma_n_2 = 0; mma_n_2 < 8; mma_n_2++) {
                        float normed[4];
                        if (kind < 2) {
                            normed[0] = rounded[mma_n_2 * 4] * rstd_lo * nw[mma_n_2 * 2];
                            normed[1] = rounded[mma_n_2 * 4 + 1] * rstd_lo * nw[mma_n_2 * 2 + 1];
                            normed[2] = rounded[mma_n_2 * 4 + 2] * rstd_hi * nw[mma_n_2 * 2];
                            normed[3] = rounded[mma_n_2 * 4 + 3] * rstd_hi * nw[mma_n_2 * 2 + 1];
                        } else {
                            normed[0] = rounded[mma_n_2 * 4];
                            normed[1] = rounded[mma_n_2 * 4 + 1];
                            normed[2] = rounded[mma_n_2 * 4 + 2];
                            normed[3] = rounded[mma_n_2 * 4 + 3];
                        }
                        uint32_t normed_bf16[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed[_lp*2 + 0], normed[_lp*2+1 + 0]));
                            normed_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + mma_n_2 * 8) * 2 ^ (srow_st & 7) << 4));
                        asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16[1]))
                            : "memory");
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                    for (int it = 0; it < 2; it++) {
                        int task = role_tid + it * 256;
                        if (task < 320) {
                            int srow = task / 5;
                            int c = task % 5;
                            int grow = tile_m_1 * 128 + half * 64 + srow;
                            if (grow < M) {
                                int row_head = grow * 56 + head;
                                unsigned int st_words[4];
                                float st_vals[16];
                                if (c < 3) {
                                    unsigned int x_lo[8];
                                    unsigned int x_hi[8];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 16 * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 16 * 2 + 16 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((c * 16 + 48) * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((c * 16 + 48) * 2 + 16 ^ (srow & 7) << 4)));
                                    if (kind < 2) {
                                        float x_lo_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_lo_f32[_pair * 2])[0]), "=f"((&x_lo_f32[_pair * 2])[1])
                                                : "r"(x_lo[_pair]));
                                        }
                                        float x_hi_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_hi_f32[_pair * 2])[0]), "=f"((&x_hi_f32[_pair * 2])[1])
                                                : "r"(x_hi[_pair]));
                                        }
                                        float _vec_load_0[8];
                                        {
                                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + c * 16) + 0);
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
                                                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_1[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_1[8];
                                        {
                                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + c * 16 + 8) + 0);
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
                                                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_2[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_2[8];
                                        {
                                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + 48 + c * 16) + 0);
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
                                                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_3[_pair]));
                                                }
                                            }
                                        }
                                        float _vec_load_3[8];
                                        {
                                            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + 48 + c * 16 + 8) + 0);
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
                                                        : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_4[_pair]));
                                                }
                                            }
                                        }
                                        float o_lo[16];
                                        float o_hi[16];
                                        for (int e_1 = 0; e_1 < 8; e_1++) {
                                            float cv0 = _vec_load_0[e_1];
                                            float sv0 = _vec_load_2[e_1];
                                            float cv1 = _vec_load_1[e_1];
                                            float sv1 = _vec_load_3[e_1];
                                            o_lo[e_1] = x_lo_f32[e_1] * cv0 - x_hi_f32[e_1] * sv0;
                                            o_hi[e_1] = x_hi_f32[e_1] * cv0 + x_lo_f32[e_1] * sv0;
                                            o_lo[8 + e_1] = x_lo_f32[8 + e_1] * cv1 - x_hi_f32[8 + e_1] * sv1;
                                            o_hi[8 + e_1] = x_hi_f32[8 + e_1] * cv1 + x_lo_f32[8 + e_1] * sv1;
                                        }
                                        uint32_t o_lo_bf16[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_lo[_lp*2 + 0], o_lo[_lp*2+1 + 0]));
                                            o_lo_bf16[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        uint32_t o_hi_bf16[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_hi[_lp*2 + 0], o_hi[_lp*2+1 + 0]));
                                            o_hi_bf16[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        if (kind == 0) {
                                            float o_lo_bf16_f32[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_lo_bf16_f32[_pair * 2])[0]), "=f"((&o_lo_bf16_f32[_pair * 2])[1])
                                                    : "r"(o_lo_bf16[_pair]));
                                            }
                                            float _fabs_0 = fabsf(o_lo_bf16_f32[0]);
                                            float _fabs_1 = fabsf(o_lo_bf16_f32[1]);
                                            float _fmax_0 = fmaxf(_fabs_0, _fabs_1);
                                            float _fabs_2 = fabsf(o_lo_bf16_f32[2]);
                                            float _fmax_1 = fmaxf(_fmax_0, _fabs_2);
                                            float _fabs_3 = fabsf(o_lo_bf16_f32[3]);
                                            float _fmax_2 = fmaxf(_fmax_1, _fabs_3);
                                            float _fabs_4 = fabsf(o_lo_bf16_f32[4]);
                                            float _fmax_3 = fmaxf(_fmax_2, _fabs_4);
                                            float _fabs_5 = fabsf(o_lo_bf16_f32[5]);
                                            float _fmax_4 = fmaxf(_fmax_3, _fabs_5);
                                            float _fabs_6 = fabsf(o_lo_bf16_f32[6]);
                                            float _fmax_5 = fmaxf(_fmax_4, _fabs_6);
                                            float _fabs_7 = fabsf(o_lo_bf16_f32[7]);
                                            float _fmax_6 = fmaxf(_fmax_5, _fabs_7);
                                            float _fabs_8 = fabsf(o_lo_bf16_f32[8]);
                                            float _fmax_7 = fmaxf(_fmax_6, _fabs_8);
                                            float _fabs_9 = fabsf(o_lo_bf16_f32[9]);
                                            float _fmax_8 = fmaxf(_fmax_7, _fabs_9);
                                            float _fabs_10 = fabsf(o_lo_bf16_f32[10]);
                                            float _fmax_9 = fmaxf(_fmax_8, _fabs_10);
                                            float _fabs_11 = fabsf(o_lo_bf16_f32[11]);
                                            float _fmax_10 = fmaxf(_fmax_9, _fabs_11);
                                            float _fabs_12 = fabsf(o_lo_bf16_f32[12]);
                                            float _fmax_11 = fmaxf(_fmax_10, _fabs_12);
                                            float _fabs_13 = fabsf(o_lo_bf16_f32[13]);
                                            float _fmax_12 = fmaxf(_fmax_11, _fabs_13);
                                            float _fabs_14 = fabsf(o_lo_bf16_f32[14]);
                                            float _fmax_13 = fmaxf(_fmax_12, _fabs_14);
                                            float _fabs_15 = fabsf(o_lo_bf16_f32[15]);
                                            float _fmax_14 = fmaxf(_fmax_13, _fabs_15);
                                            {
                                                unsigned short _sf_pair;
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_14 * sf_mul_sel));
                                                *(reinterpret_cast<unsigned char*>(q_sf + (row_head * 8 + (c * 16 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                            }
                                            uint16_t _e4m3x2_f32_0;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(0.0f), "f"(_fmax_14 * sf_mul_sel));
                                            uint16_t _e4m3x2_decode_5 = (uint16_t)((unsigned int)_e4m3x2_f32_0 & 0xFFu);
                                            uint32_t _f16x2_decode_5;
                                            float _fp8_decode_0;
                                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_5) : "h"(_e4m3x2_decode_5));
                                            uint16_t _f16_decode_5 = (uint16_t)_f16x2_decode_5;
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_0) : "h"(_f16_decode_5));
                                            float _fdiv_rn_2 = __fdiv_rn(out_scale_sel, _fp8_decode_0);
                                            st_vals[0] = o_lo_bf16_f32[0] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[1] = o_lo_bf16_f32[1] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[2] = o_lo_bf16_f32[2] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[3] = o_lo_bf16_f32[3] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[4] = o_lo_bf16_f32[4] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[5] = o_lo_bf16_f32[5] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[6] = o_lo_bf16_f32[6] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[7] = o_lo_bf16_f32[7] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[8] = o_lo_bf16_f32[8] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[9] = o_lo_bf16_f32[9] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[10] = o_lo_bf16_f32[10] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[11] = o_lo_bf16_f32[11] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[12] = o_lo_bf16_f32[12] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[13] = o_lo_bf16_f32[13] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[14] = o_lo_bf16_f32[14] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            st_vals[15] = o_lo_bf16_f32[15] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_2 : 0.0f);
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                            *(reinterpret_cast<unsigned int*>(q_out + (row_head * 16 + (c * 16 >> 3))) + (0)) = st_words[0];
                                            *(reinterpret_cast<unsigned int*>(q_out + (row_head * 16 + (c * 16 >> 3) + 1)) + (0)) = st_words[1];
                                            float o_hi_bf16_f32[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_hi_bf16_f32[_pair * 2])[0]), "=f"((&o_hi_bf16_f32[_pair * 2])[1])
                                                    : "r"(o_hi_bf16[_pair]));
                                            }
                                            float _fabs_16 = fabsf(o_hi_bf16_f32[0]);
                                            float _fabs_17 = fabsf(o_hi_bf16_f32[1]);
                                            float _fmax_15 = fmaxf(_fabs_16, _fabs_17);
                                            float _fabs_18 = fabsf(o_hi_bf16_f32[2]);
                                            float _fmax_16 = fmaxf(_fmax_15, _fabs_18);
                                            float _fabs_19 = fabsf(o_hi_bf16_f32[3]);
                                            float _fmax_17 = fmaxf(_fmax_16, _fabs_19);
                                            float _fabs_20 = fabsf(o_hi_bf16_f32[4]);
                                            float _fmax_18 = fmaxf(_fmax_17, _fabs_20);
                                            float _fabs_21 = fabsf(o_hi_bf16_f32[5]);
                                            float _fmax_19 = fmaxf(_fmax_18, _fabs_21);
                                            float _fabs_22 = fabsf(o_hi_bf16_f32[6]);
                                            float _fmax_20 = fmaxf(_fmax_19, _fabs_22);
                                            float _fabs_23 = fabsf(o_hi_bf16_f32[7]);
                                            float _fmax_21 = fmaxf(_fmax_20, _fabs_23);
                                            float _fabs_24 = fabsf(o_hi_bf16_f32[8]);
                                            float _fmax_22 = fmaxf(_fmax_21, _fabs_24);
                                            float _fabs_25 = fabsf(o_hi_bf16_f32[9]);
                                            float _fmax_23 = fmaxf(_fmax_22, _fabs_25);
                                            float _fabs_26 = fabsf(o_hi_bf16_f32[10]);
                                            float _fmax_24 = fmaxf(_fmax_23, _fabs_26);
                                            float _fabs_27 = fabsf(o_hi_bf16_f32[11]);
                                            float _fmax_25 = fmaxf(_fmax_24, _fabs_27);
                                            float _fabs_28 = fabsf(o_hi_bf16_f32[12]);
                                            float _fmax_26 = fmaxf(_fmax_25, _fabs_28);
                                            float _fabs_29 = fabsf(o_hi_bf16_f32[13]);
                                            float _fmax_27 = fmaxf(_fmax_26, _fabs_29);
                                            float _fabs_30 = fabsf(o_hi_bf16_f32[14]);
                                            float _fmax_28 = fmaxf(_fmax_27, _fabs_30);
                                            float _fabs_31 = fabsf(o_hi_bf16_f32[15]);
                                            float _fmax_29 = fmaxf(_fmax_28, _fabs_31);
                                            {
                                                unsigned short _sf_pair;
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_29 * sf_mul_sel));
                                                *(reinterpret_cast<unsigned char*>(q_sf + (row_head * 8 + (c * 16 + 48 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                            }
                                            uint16_t _e4m3x2_f32_1;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(0.0f), "f"(_fmax_29 * sf_mul_sel));
                                            uint16_t _e4m3x2_decode_6 = (uint16_t)((unsigned int)_e4m3x2_f32_1 & 0xFFu);
                                            uint32_t _f16x2_decode_6;
                                            float _fp8_decode_1;
                                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_6) : "h"(_e4m3x2_decode_6));
                                            uint16_t _f16_decode_6 = (uint16_t)_f16x2_decode_6;
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_1) : "h"(_f16_decode_6));
                                            float _fdiv_rn_3 = __fdiv_rn(out_scale_sel, _fp8_decode_1);
                                            st_vals[0] = o_hi_bf16_f32[0] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[1] = o_hi_bf16_f32[1] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[2] = o_hi_bf16_f32[2] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[3] = o_hi_bf16_f32[3] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[4] = o_hi_bf16_f32[4] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[5] = o_hi_bf16_f32[5] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[6] = o_hi_bf16_f32[6] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[7] = o_hi_bf16_f32[7] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[8] = o_hi_bf16_f32[8] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[9] = o_hi_bf16_f32[9] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[10] = o_hi_bf16_f32[10] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[11] = o_hi_bf16_f32[11] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[12] = o_hi_bf16_f32[12] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[13] = o_hi_bf16_f32[13] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[14] = o_hi_bf16_f32[14] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            st_vals[15] = o_hi_bf16_f32[15] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_3 : 0.0f);
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                            *(reinterpret_cast<unsigned int*>(q_out + (row_head * 16 + (c * 16 + 48 >> 3))) + (0)) = st_words[0];
                                            *(reinterpret_cast<unsigned int*>(q_out + (row_head * 16 + (c * 16 + 48 >> 3) + 1)) + (0)) = st_words[1];
                                        } else {
                                            float o_lo_bf16_f32_1[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_lo_bf16_f32_1[_pair * 2])[0]), "=f"((&o_lo_bf16_f32_1[_pair * 2])[1])
                                                    : "r"(o_lo_bf16[_pair]));
                                            }
                                            float _fabs_32 = fabsf(o_lo_bf16_f32_1[0]);
                                            float _fabs_33 = fabsf(o_lo_bf16_f32_1[1]);
                                            float _fmax_30 = fmaxf(_fabs_32, _fabs_33);
                                            float _fabs_34 = fabsf(o_lo_bf16_f32_1[2]);
                                            float _fmax_31 = fmaxf(_fmax_30, _fabs_34);
                                            float _fabs_35 = fabsf(o_lo_bf16_f32_1[3]);
                                            float _fmax_32 = fmaxf(_fmax_31, _fabs_35);
                                            float _fabs_36 = fabsf(o_lo_bf16_f32_1[4]);
                                            float _fmax_33 = fmaxf(_fmax_32, _fabs_36);
                                            float _fabs_37 = fabsf(o_lo_bf16_f32_1[5]);
                                            float _fmax_34 = fmaxf(_fmax_33, _fabs_37);
                                            float _fabs_38 = fabsf(o_lo_bf16_f32_1[6]);
                                            float _fmax_35 = fmaxf(_fmax_34, _fabs_38);
                                            float _fabs_39 = fabsf(o_lo_bf16_f32_1[7]);
                                            float _fmax_36 = fmaxf(_fmax_35, _fabs_39);
                                            float _fabs_40 = fabsf(o_lo_bf16_f32_1[8]);
                                            float _fmax_37 = fmaxf(_fmax_36, _fabs_40);
                                            float _fabs_41 = fabsf(o_lo_bf16_f32_1[9]);
                                            float _fmax_38 = fmaxf(_fmax_37, _fabs_41);
                                            float _fabs_42 = fabsf(o_lo_bf16_f32_1[10]);
                                            float _fmax_39 = fmaxf(_fmax_38, _fabs_42);
                                            float _fabs_43 = fabsf(o_lo_bf16_f32_1[11]);
                                            float _fmax_40 = fmaxf(_fmax_39, _fabs_43);
                                            float _fabs_44 = fabsf(o_lo_bf16_f32_1[12]);
                                            float _fmax_41 = fmaxf(_fmax_40, _fabs_44);
                                            float _fabs_45 = fabsf(o_lo_bf16_f32_1[13]);
                                            float _fmax_42 = fmaxf(_fmax_41, _fabs_45);
                                            float _fabs_46 = fabsf(o_lo_bf16_f32_1[14]);
                                            float _fmax_43 = fmaxf(_fmax_42, _fabs_46);
                                            float _fabs_47 = fabsf(o_lo_bf16_f32_1[15]);
                                            float _fmax_44 = fmaxf(_fmax_43, _fabs_47);
                                            {
                                                unsigned short _sf_pair;
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_44 * sf_mul_sel));
                                                *(reinterpret_cast<unsigned char*>(k_sf + (row_head * 8 + (c * 16 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                            }
                                            uint16_t _e4m3x2_f32_2;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(0.0f), "f"(_fmax_44 * sf_mul_sel));
                                            uint16_t _e4m3x2_decode_7 = (uint16_t)((unsigned int)_e4m3x2_f32_2 & 0xFFu);
                                            uint32_t _f16x2_decode_7;
                                            float _fp8_decode_2;
                                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_7) : "h"(_e4m3x2_decode_7));
                                            uint16_t _f16_decode_7 = (uint16_t)_f16x2_decode_7;
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_2) : "h"(_f16_decode_7));
                                            float _fdiv_rn_4 = __fdiv_rn(out_scale_sel, _fp8_decode_2);
                                            st_vals[0] = o_lo_bf16_f32_1[0] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[1] = o_lo_bf16_f32_1[1] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[2] = o_lo_bf16_f32_1[2] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[3] = o_lo_bf16_f32_1[3] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[4] = o_lo_bf16_f32_1[4] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[5] = o_lo_bf16_f32_1[5] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[6] = o_lo_bf16_f32_1[6] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[7] = o_lo_bf16_f32_1[7] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[8] = o_lo_bf16_f32_1[8] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[9] = o_lo_bf16_f32_1[9] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[10] = o_lo_bf16_f32_1[10] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[11] = o_lo_bf16_f32_1[11] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[12] = o_lo_bf16_f32_1[12] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[13] = o_lo_bf16_f32_1[13] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[14] = o_lo_bf16_f32_1[14] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            st_vals[15] = o_lo_bf16_f32_1[15] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_4 : 0.0f);
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                            *(reinterpret_cast<unsigned int*>(k_out + (row_head * 16 + (c * 16 >> 3))) + (0)) = st_words[0];
                                            *(reinterpret_cast<unsigned int*>(k_out + (row_head * 16 + (c * 16 >> 3) + 1)) + (0)) = st_words[1];
                                            float o_hi_bf16_f32_1[16];
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 8; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&o_hi_bf16_f32_1[_pair * 2])[0]), "=f"((&o_hi_bf16_f32_1[_pair * 2])[1])
                                                    : "r"(o_hi_bf16[_pair]));
                                            }
                                            float _fabs_48 = fabsf(o_hi_bf16_f32_1[0]);
                                            float _fabs_49 = fabsf(o_hi_bf16_f32_1[1]);
                                            float _fmax_45 = fmaxf(_fabs_48, _fabs_49);
                                            float _fabs_50 = fabsf(o_hi_bf16_f32_1[2]);
                                            float _fmax_46 = fmaxf(_fmax_45, _fabs_50);
                                            float _fabs_51 = fabsf(o_hi_bf16_f32_1[3]);
                                            float _fmax_47 = fmaxf(_fmax_46, _fabs_51);
                                            float _fabs_52 = fabsf(o_hi_bf16_f32_1[4]);
                                            float _fmax_48 = fmaxf(_fmax_47, _fabs_52);
                                            float _fabs_53 = fabsf(o_hi_bf16_f32_1[5]);
                                            float _fmax_49 = fmaxf(_fmax_48, _fabs_53);
                                            float _fabs_54 = fabsf(o_hi_bf16_f32_1[6]);
                                            float _fmax_50 = fmaxf(_fmax_49, _fabs_54);
                                            float _fabs_55 = fabsf(o_hi_bf16_f32_1[7]);
                                            float _fmax_51 = fmaxf(_fmax_50, _fabs_55);
                                            float _fabs_56 = fabsf(o_hi_bf16_f32_1[8]);
                                            float _fmax_52 = fmaxf(_fmax_51, _fabs_56);
                                            float _fabs_57 = fabsf(o_hi_bf16_f32_1[9]);
                                            float _fmax_53 = fmaxf(_fmax_52, _fabs_57);
                                            float _fabs_58 = fabsf(o_hi_bf16_f32_1[10]);
                                            float _fmax_54 = fmaxf(_fmax_53, _fabs_58);
                                            float _fabs_59 = fabsf(o_hi_bf16_f32_1[11]);
                                            float _fmax_55 = fmaxf(_fmax_54, _fabs_59);
                                            float _fabs_60 = fabsf(o_hi_bf16_f32_1[12]);
                                            float _fmax_56 = fmaxf(_fmax_55, _fabs_60);
                                            float _fabs_61 = fabsf(o_hi_bf16_f32_1[13]);
                                            float _fmax_57 = fmaxf(_fmax_56, _fabs_61);
                                            float _fabs_62 = fabsf(o_hi_bf16_f32_1[14]);
                                            float _fmax_58 = fmaxf(_fmax_57, _fabs_62);
                                            float _fabs_63 = fabsf(o_hi_bf16_f32_1[15]);
                                            float _fmax_59 = fmaxf(_fmax_58, _fabs_63);
                                            {
                                                unsigned short _sf_pair;
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_59 * sf_mul_sel));
                                                *(reinterpret_cast<unsigned char*>(k_sf + (row_head * 8 + (c * 16 + 48 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                            }
                                            uint16_t _e4m3x2_f32_3;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(0.0f), "f"(_fmax_59 * sf_mul_sel));
                                            uint16_t _e4m3x2_decode_8 = (uint16_t)((unsigned int)_e4m3x2_f32_3 & 0xFFu);
                                            uint32_t _f16x2_decode_8;
                                            float _fp8_decode_3;
                                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_8) : "h"(_e4m3x2_decode_8));
                                            uint16_t _f16_decode_8 = (uint16_t)_f16x2_decode_8;
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_3) : "h"(_f16_decode_8));
                                            float _fdiv_rn_5 = __fdiv_rn(out_scale_sel, _fp8_decode_3);
                                            st_vals[0] = o_hi_bf16_f32_1[0] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[1] = o_hi_bf16_f32_1[1] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[2] = o_hi_bf16_f32_1[2] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[3] = o_hi_bf16_f32_1[3] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[4] = o_hi_bf16_f32_1[4] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[5] = o_hi_bf16_f32_1[5] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[6] = o_hi_bf16_f32_1[6] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[7] = o_hi_bf16_f32_1[7] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[8] = o_hi_bf16_f32_1[8] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[9] = o_hi_bf16_f32_1[9] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[10] = o_hi_bf16_f32_1[10] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[11] = o_hi_bf16_f32_1[11] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[12] = o_hi_bf16_f32_1[12] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[13] = o_hi_bf16_f32_1[13] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[14] = o_hi_bf16_f32_1[14] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            st_vals[15] = o_hi_bf16_f32_1[15] * ((_fp8_decode_3 != 0.0f) ? _fdiv_rn_5 : 0.0f);
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                            *(reinterpret_cast<unsigned int*>(k_out + (row_head * 16 + (c * 16 + 48 >> 3))) + (0)) = st_words[0];
                                            *(reinterpret_cast<unsigned int*>(k_out + (row_head * 16 + (c * 16 + 48 >> 3) + 1)) + (0)) = st_words[1];
                                        }
                                    } else {
                                        float x_lo_f32_1[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_lo_f32_1[_pair * 2])[0]), "=f"((&x_lo_f32_1[_pair * 2])[1])
                                                : "r"(x_lo[_pair]));
                                        }
                                        float _fabs_64 = fabsf(x_lo_f32_1[0]);
                                        float _fabs_65 = fabsf(x_lo_f32_1[1]);
                                        float _fmax_60 = fmaxf(_fabs_64, _fabs_65);
                                        float _fabs_66 = fabsf(x_lo_f32_1[2]);
                                        float _fmax_61 = fmaxf(_fmax_60, _fabs_66);
                                        float _fabs_67 = fabsf(x_lo_f32_1[3]);
                                        float _fmax_62 = fmaxf(_fmax_61, _fabs_67);
                                        float _fabs_68 = fabsf(x_lo_f32_1[4]);
                                        float _fmax_63 = fmaxf(_fmax_62, _fabs_68);
                                        float _fabs_69 = fabsf(x_lo_f32_1[5]);
                                        float _fmax_64 = fmaxf(_fmax_63, _fabs_69);
                                        float _fabs_70 = fabsf(x_lo_f32_1[6]);
                                        float _fmax_65 = fmaxf(_fmax_64, _fabs_70);
                                        float _fabs_71 = fabsf(x_lo_f32_1[7]);
                                        float _fmax_66 = fmaxf(_fmax_65, _fabs_71);
                                        float _fabs_72 = fabsf(x_lo_f32_1[8]);
                                        float _fmax_67 = fmaxf(_fmax_66, _fabs_72);
                                        float _fabs_73 = fabsf(x_lo_f32_1[9]);
                                        float _fmax_68 = fmaxf(_fmax_67, _fabs_73);
                                        float _fabs_74 = fabsf(x_lo_f32_1[10]);
                                        float _fmax_69 = fmaxf(_fmax_68, _fabs_74);
                                        float _fabs_75 = fabsf(x_lo_f32_1[11]);
                                        float _fmax_70 = fmaxf(_fmax_69, _fabs_75);
                                        float _fabs_76 = fabsf(x_lo_f32_1[12]);
                                        float _fmax_71 = fmaxf(_fmax_70, _fabs_76);
                                        float _fabs_77 = fabsf(x_lo_f32_1[13]);
                                        float _fmax_72 = fmaxf(_fmax_71, _fabs_77);
                                        float _fabs_78 = fabsf(x_lo_f32_1[14]);
                                        float _fmax_73 = fmaxf(_fmax_72, _fabs_78);
                                        float _fabs_79 = fabsf(x_lo_f32_1[15]);
                                        float _fmax_74 = fmaxf(_fmax_73, _fabs_79);
                                        {
                                            unsigned short _sf_pair;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_74 * sf_mul_sel));
                                            *(reinterpret_cast<unsigned char*>(v_sf + (row_head * 8 + (c * 16 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                        }
                                        uint16_t _e4m3x2_f32_4;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_4) : "f"(0.0f), "f"(_fmax_74 * sf_mul_sel));
                                        uint16_t _e4m3x2_decode_9 = (uint16_t)((unsigned int)_e4m3x2_f32_4 & 0xFFu);
                                        uint32_t _f16x2_decode_9;
                                        float _fp8_decode_4;
                                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_9) : "h"(_e4m3x2_decode_9));
                                        uint16_t _f16_decode_9 = (uint16_t)_f16x2_decode_9;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_4) : "h"(_f16_decode_9));
                                        float _fdiv_rn_6 = __fdiv_rn(out_scale_sel, _fp8_decode_4);
                                        st_vals[0] = x_lo_f32_1[0] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[1] = x_lo_f32_1[1] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[2] = x_lo_f32_1[2] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[3] = x_lo_f32_1[3] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[4] = x_lo_f32_1[4] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[5] = x_lo_f32_1[5] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[6] = x_lo_f32_1[6] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[7] = x_lo_f32_1[7] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[8] = x_lo_f32_1[8] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[9] = x_lo_f32_1[9] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[10] = x_lo_f32_1[10] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[11] = x_lo_f32_1[11] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[12] = x_lo_f32_1[12] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[13] = x_lo_f32_1[13] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[14] = x_lo_f32_1[14] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        st_vals[15] = x_lo_f32_1[15] * ((_fp8_decode_4 != 0.0f) ? _fdiv_rn_6 : 0.0f);
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                        *(reinterpret_cast<unsigned int*>(v_out + (row_head * 16 + (c * 16 >> 3))) + (0)) = st_words[0];
                                        *(reinterpret_cast<unsigned int*>(v_out + (row_head * 16 + (c * 16 >> 3) + 1)) + (0)) = st_words[1];
                                        float x_hi_f32_1[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_hi_f32_1[_pair * 2])[0]), "=f"((&x_hi_f32_1[_pair * 2])[1])
                                                : "r"(x_hi[_pair]));
                                        }
                                        float _fabs_80 = fabsf(x_hi_f32_1[0]);
                                        float _fabs_81 = fabsf(x_hi_f32_1[1]);
                                        float _fmax_75 = fmaxf(_fabs_80, _fabs_81);
                                        float _fabs_82 = fabsf(x_hi_f32_1[2]);
                                        float _fmax_76 = fmaxf(_fmax_75, _fabs_82);
                                        float _fabs_83 = fabsf(x_hi_f32_1[3]);
                                        float _fmax_77 = fmaxf(_fmax_76, _fabs_83);
                                        float _fabs_84 = fabsf(x_hi_f32_1[4]);
                                        float _fmax_78 = fmaxf(_fmax_77, _fabs_84);
                                        float _fabs_85 = fabsf(x_hi_f32_1[5]);
                                        float _fmax_79 = fmaxf(_fmax_78, _fabs_85);
                                        float _fabs_86 = fabsf(x_hi_f32_1[6]);
                                        float _fmax_80 = fmaxf(_fmax_79, _fabs_86);
                                        float _fabs_87 = fabsf(x_hi_f32_1[7]);
                                        float _fmax_81 = fmaxf(_fmax_80, _fabs_87);
                                        float _fabs_88 = fabsf(x_hi_f32_1[8]);
                                        float _fmax_82 = fmaxf(_fmax_81, _fabs_88);
                                        float _fabs_89 = fabsf(x_hi_f32_1[9]);
                                        float _fmax_83 = fmaxf(_fmax_82, _fabs_89);
                                        float _fabs_90 = fabsf(x_hi_f32_1[10]);
                                        float _fmax_84 = fmaxf(_fmax_83, _fabs_90);
                                        float _fabs_91 = fabsf(x_hi_f32_1[11]);
                                        float _fmax_85 = fmaxf(_fmax_84, _fabs_91);
                                        float _fabs_92 = fabsf(x_hi_f32_1[12]);
                                        float _fmax_86 = fmaxf(_fmax_85, _fabs_92);
                                        float _fabs_93 = fabsf(x_hi_f32_1[13]);
                                        float _fmax_87 = fmaxf(_fmax_86, _fabs_93);
                                        float _fabs_94 = fabsf(x_hi_f32_1[14]);
                                        float _fmax_88 = fmaxf(_fmax_87, _fabs_94);
                                        float _fabs_95 = fabsf(x_hi_f32_1[15]);
                                        float _fmax_89 = fmaxf(_fmax_88, _fabs_95);
                                        {
                                            unsigned short _sf_pair;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_89 * sf_mul_sel));
                                            *(reinterpret_cast<unsigned char*>(v_sf + (row_head * 8 + (c * 16 + 48 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                        }
                                        uint16_t _e4m3x2_f32_5;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_5) : "f"(0.0f), "f"(_fmax_89 * sf_mul_sel));
                                        uint16_t _e4m3x2_decode_10 = (uint16_t)((unsigned int)_e4m3x2_f32_5 & 0xFFu);
                                        uint32_t _f16x2_decode_10;
                                        float _fp8_decode_5;
                                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_10) : "h"(_e4m3x2_decode_10));
                                        uint16_t _f16_decode_10 = (uint16_t)_f16x2_decode_10;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_5) : "h"(_f16_decode_10));
                                        float _fdiv_rn_7 = __fdiv_rn(out_scale_sel, _fp8_decode_5);
                                        st_vals[0] = x_hi_f32_1[0] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[1] = x_hi_f32_1[1] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[2] = x_hi_f32_1[2] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[3] = x_hi_f32_1[3] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[4] = x_hi_f32_1[4] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[5] = x_hi_f32_1[5] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[6] = x_hi_f32_1[6] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[7] = x_hi_f32_1[7] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[8] = x_hi_f32_1[8] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[9] = x_hi_f32_1[9] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[10] = x_hi_f32_1[10] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[11] = x_hi_f32_1[11] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[12] = x_hi_f32_1[12] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[13] = x_hi_f32_1[13] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[14] = x_hi_f32_1[14] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        st_vals[15] = x_hi_f32_1[15] * ((_fp8_decode_5 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                        *(reinterpret_cast<unsigned int*>(v_out + (row_head * 16 + (c * 16 + 48 >> 3))) + (0)) = st_words[0];
                                        *(reinterpret_cast<unsigned int*>(v_out + (row_head * 16 + (c * 16 + 48 >> 3) + 1)) + (0)) = st_words[1];
                                    }
                                } else {
                                    unsigned int x_pt[8];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_pt[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(0) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((96 + (c - 3) * 16) * 2 ^ (srow & 7) << 4)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_pt[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_pt[(4) + 3]))
                                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)((96 + (c - 3) * 16) * 2 + 16 ^ (srow & 7) << 4)));
                                    if (kind == 0) {
                                        float x_pt_f32[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_pt_f32[_pair * 2])[0]), "=f"((&x_pt_f32[_pair * 2])[1])
                                                : "r"(x_pt[_pair]));
                                        }
                                        float _fabs_96 = fabsf(x_pt_f32[0]);
                                        float _fabs_97 = fabsf(x_pt_f32[1]);
                                        float _fmax_90 = fmaxf(_fabs_96, _fabs_97);
                                        float _fabs_98 = fabsf(x_pt_f32[2]);
                                        float _fmax_91 = fmaxf(_fmax_90, _fabs_98);
                                        float _fabs_99 = fabsf(x_pt_f32[3]);
                                        float _fmax_92 = fmaxf(_fmax_91, _fabs_99);
                                        float _fabs_100 = fabsf(x_pt_f32[4]);
                                        float _fmax_93 = fmaxf(_fmax_92, _fabs_100);
                                        float _fabs_101 = fabsf(x_pt_f32[5]);
                                        float _fmax_94 = fmaxf(_fmax_93, _fabs_101);
                                        float _fabs_102 = fabsf(x_pt_f32[6]);
                                        float _fmax_95 = fmaxf(_fmax_94, _fabs_102);
                                        float _fabs_103 = fabsf(x_pt_f32[7]);
                                        float _fmax_96 = fmaxf(_fmax_95, _fabs_103);
                                        float _fabs_104 = fabsf(x_pt_f32[8]);
                                        float _fmax_97 = fmaxf(_fmax_96, _fabs_104);
                                        float _fabs_105 = fabsf(x_pt_f32[9]);
                                        float _fmax_98 = fmaxf(_fmax_97, _fabs_105);
                                        float _fabs_106 = fabsf(x_pt_f32[10]);
                                        float _fmax_99 = fmaxf(_fmax_98, _fabs_106);
                                        float _fabs_107 = fabsf(x_pt_f32[11]);
                                        float _fmax_100 = fmaxf(_fmax_99, _fabs_107);
                                        float _fabs_108 = fabsf(x_pt_f32[12]);
                                        float _fmax_101 = fmaxf(_fmax_100, _fabs_108);
                                        float _fabs_109 = fabsf(x_pt_f32[13]);
                                        float _fmax_102 = fmaxf(_fmax_101, _fabs_109);
                                        float _fabs_110 = fabsf(x_pt_f32[14]);
                                        float _fmax_103 = fmaxf(_fmax_102, _fabs_110);
                                        float _fabs_111 = fabsf(x_pt_f32[15]);
                                        float _fmax_104 = fmaxf(_fmax_103, _fabs_111);
                                        {
                                            unsigned short _sf_pair;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_104 * sf_mul_sel));
                                            *(reinterpret_cast<unsigned char*>(q_sf + (row_head * 8 + (96 + (c - 3) * 16 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                        }
                                        uint16_t _e4m3x2_f32_6;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_6) : "f"(0.0f), "f"(_fmax_104 * sf_mul_sel));
                                        uint16_t _e4m3x2_decode_11 = (uint16_t)((unsigned int)_e4m3x2_f32_6 & 0xFFu);
                                        uint32_t _f16x2_decode_11;
                                        float _fp8_decode_6;
                                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_11) : "h"(_e4m3x2_decode_11));
                                        uint16_t _f16_decode_11 = (uint16_t)_f16x2_decode_11;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_6) : "h"(_f16_decode_11));
                                        float _fdiv_rn_8 = __fdiv_rn(out_scale_sel, _fp8_decode_6);
                                        st_vals[0] = x_pt_f32[0] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[1] = x_pt_f32[1] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[2] = x_pt_f32[2] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[3] = x_pt_f32[3] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[4] = x_pt_f32[4] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[5] = x_pt_f32[5] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[6] = x_pt_f32[6] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[7] = x_pt_f32[7] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[8] = x_pt_f32[8] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[9] = x_pt_f32[9] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[10] = x_pt_f32[10] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[11] = x_pt_f32[11] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[12] = x_pt_f32[12] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[13] = x_pt_f32[13] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[14] = x_pt_f32[14] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        st_vals[15] = x_pt_f32[15] * ((_fp8_decode_6 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                        *(reinterpret_cast<unsigned int*>(q_out + (row_head * 16 + (96 + (c - 3) * 16 >> 3))) + (0)) = st_words[0];
                                        *(reinterpret_cast<unsigned int*>(q_out + (row_head * 16 + (96 + (c - 3) * 16 >> 3) + 1)) + (0)) = st_words[1];
                                    } else if (kind == 1) {
                                        float x_pt_f32_1[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_pt_f32_1[_pair * 2])[0]), "=f"((&x_pt_f32_1[_pair * 2])[1])
                                                : "r"(x_pt[_pair]));
                                        }
                                        float _fabs_112 = fabsf(x_pt_f32_1[0]);
                                        float _fabs_113 = fabsf(x_pt_f32_1[1]);
                                        float _fmax_105 = fmaxf(_fabs_112, _fabs_113);
                                        float _fabs_114 = fabsf(x_pt_f32_1[2]);
                                        float _fmax_106 = fmaxf(_fmax_105, _fabs_114);
                                        float _fabs_115 = fabsf(x_pt_f32_1[3]);
                                        float _fmax_107 = fmaxf(_fmax_106, _fabs_115);
                                        float _fabs_116 = fabsf(x_pt_f32_1[4]);
                                        float _fmax_108 = fmaxf(_fmax_107, _fabs_116);
                                        float _fabs_117 = fabsf(x_pt_f32_1[5]);
                                        float _fmax_109 = fmaxf(_fmax_108, _fabs_117);
                                        float _fabs_118 = fabsf(x_pt_f32_1[6]);
                                        float _fmax_110 = fmaxf(_fmax_109, _fabs_118);
                                        float _fabs_119 = fabsf(x_pt_f32_1[7]);
                                        float _fmax_111 = fmaxf(_fmax_110, _fabs_119);
                                        float _fabs_120 = fabsf(x_pt_f32_1[8]);
                                        float _fmax_112 = fmaxf(_fmax_111, _fabs_120);
                                        float _fabs_121 = fabsf(x_pt_f32_1[9]);
                                        float _fmax_113 = fmaxf(_fmax_112, _fabs_121);
                                        float _fabs_122 = fabsf(x_pt_f32_1[10]);
                                        float _fmax_114 = fmaxf(_fmax_113, _fabs_122);
                                        float _fabs_123 = fabsf(x_pt_f32_1[11]);
                                        float _fmax_115 = fmaxf(_fmax_114, _fabs_123);
                                        float _fabs_124 = fabsf(x_pt_f32_1[12]);
                                        float _fmax_116 = fmaxf(_fmax_115, _fabs_124);
                                        float _fabs_125 = fabsf(x_pt_f32_1[13]);
                                        float _fmax_117 = fmaxf(_fmax_116, _fabs_125);
                                        float _fabs_126 = fabsf(x_pt_f32_1[14]);
                                        float _fmax_118 = fmaxf(_fmax_117, _fabs_126);
                                        float _fabs_127 = fabsf(x_pt_f32_1[15]);
                                        float _fmax_119 = fmaxf(_fmax_118, _fabs_127);
                                        {
                                            unsigned short _sf_pair;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_119 * sf_mul_sel));
                                            *(reinterpret_cast<unsigned char*>(k_sf + (row_head * 8 + (96 + (c - 3) * 16 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                        }
                                        uint16_t _e4m3x2_f32_7;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_7) : "f"(0.0f), "f"(_fmax_119 * sf_mul_sel));
                                        uint16_t _e4m3x2_decode_12 = (uint16_t)((unsigned int)_e4m3x2_f32_7 & 0xFFu);
                                        uint32_t _f16x2_decode_12;
                                        float _fp8_decode_7;
                                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_12) : "h"(_e4m3x2_decode_12));
                                        uint16_t _f16_decode_12 = (uint16_t)_f16x2_decode_12;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_7) : "h"(_f16_decode_12));
                                        float _fdiv_rn_9 = __fdiv_rn(out_scale_sel, _fp8_decode_7);
                                        st_vals[0] = x_pt_f32_1[0] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[1] = x_pt_f32_1[1] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[2] = x_pt_f32_1[2] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[3] = x_pt_f32_1[3] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[4] = x_pt_f32_1[4] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[5] = x_pt_f32_1[5] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[6] = x_pt_f32_1[6] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[7] = x_pt_f32_1[7] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[8] = x_pt_f32_1[8] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[9] = x_pt_f32_1[9] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[10] = x_pt_f32_1[10] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[11] = x_pt_f32_1[11] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[12] = x_pt_f32_1[12] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[13] = x_pt_f32_1[13] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[14] = x_pt_f32_1[14] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        st_vals[15] = x_pt_f32_1[15] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_9 : 0.0f);
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                        *(reinterpret_cast<unsigned int*>(k_out + (row_head * 16 + (96 + (c - 3) * 16 >> 3))) + (0)) = st_words[0];
                                        *(reinterpret_cast<unsigned int*>(k_out + (row_head * 16 + (96 + (c - 3) * 16 >> 3) + 1)) + (0)) = st_words[1];
                                    } else {
                                        float x_pt_f32_2[16];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 8; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&x_pt_f32_2[_pair * 2])[0]), "=f"((&x_pt_f32_2[_pair * 2])[1])
                                                : "r"(x_pt[_pair]));
                                        }
                                        float _fabs_128 = fabsf(x_pt_f32_2[0]);
                                        float _fabs_129 = fabsf(x_pt_f32_2[1]);
                                        float _fmax_120 = fmaxf(_fabs_128, _fabs_129);
                                        float _fabs_130 = fabsf(x_pt_f32_2[2]);
                                        float _fmax_121 = fmaxf(_fmax_120, _fabs_130);
                                        float _fabs_131 = fabsf(x_pt_f32_2[3]);
                                        float _fmax_122 = fmaxf(_fmax_121, _fabs_131);
                                        float _fabs_132 = fabsf(x_pt_f32_2[4]);
                                        float _fmax_123 = fmaxf(_fmax_122, _fabs_132);
                                        float _fabs_133 = fabsf(x_pt_f32_2[5]);
                                        float _fmax_124 = fmaxf(_fmax_123, _fabs_133);
                                        float _fabs_134 = fabsf(x_pt_f32_2[6]);
                                        float _fmax_125 = fmaxf(_fmax_124, _fabs_134);
                                        float _fabs_135 = fabsf(x_pt_f32_2[7]);
                                        float _fmax_126 = fmaxf(_fmax_125, _fabs_135);
                                        float _fabs_136 = fabsf(x_pt_f32_2[8]);
                                        float _fmax_127 = fmaxf(_fmax_126, _fabs_136);
                                        float _fabs_137 = fabsf(x_pt_f32_2[9]);
                                        float _fmax_128 = fmaxf(_fmax_127, _fabs_137);
                                        float _fabs_138 = fabsf(x_pt_f32_2[10]);
                                        float _fmax_129 = fmaxf(_fmax_128, _fabs_138);
                                        float _fabs_139 = fabsf(x_pt_f32_2[11]);
                                        float _fmax_130 = fmaxf(_fmax_129, _fabs_139);
                                        float _fabs_140 = fabsf(x_pt_f32_2[12]);
                                        float _fmax_131 = fmaxf(_fmax_130, _fabs_140);
                                        float _fabs_141 = fabsf(x_pt_f32_2[13]);
                                        float _fmax_132 = fmaxf(_fmax_131, _fabs_141);
                                        float _fabs_142 = fabsf(x_pt_f32_2[14]);
                                        float _fmax_133 = fmaxf(_fmax_132, _fabs_142);
                                        float _fabs_143 = fabsf(x_pt_f32_2[15]);
                                        float _fmax_134 = fmaxf(_fmax_133, _fabs_143);
                                        {
                                            unsigned short _sf_pair;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(_fmax_134 * sf_mul_sel));
                                            *(reinterpret_cast<unsigned char*>(v_sf + (row_head * 8 + (96 + (c - 3) * 16 >> 4))) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                                        }
                                        uint16_t _e4m3x2_f32_8;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_8) : "f"(0.0f), "f"(_fmax_134 * sf_mul_sel));
                                        uint16_t _e4m3x2_decode_13 = (uint16_t)((unsigned int)_e4m3x2_f32_8 & 0xFFu);
                                        uint32_t _f16x2_decode_13;
                                        float _fp8_decode_8;
                                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_decode_13));
                                        uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_8) : "h"(_f16_decode_13));
                                        float _fdiv_rn_10 = __fdiv_rn(out_scale_sel, _fp8_decode_8);
                                        st_vals[0] = x_pt_f32_2[0] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[1] = x_pt_f32_2[1] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[2] = x_pt_f32_2[2] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[3] = x_pt_f32_2[3] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[4] = x_pt_f32_2[4] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[5] = x_pt_f32_2[5] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[6] = x_pt_f32_2[6] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[7] = x_pt_f32_2[7] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[8] = x_pt_f32_2[8] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[9] = x_pt_f32_2[9] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[10] = x_pt_f32_2[10] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[11] = x_pt_f32_2[11] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[12] = x_pt_f32_2[12] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[13] = x_pt_f32_2[13] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[14] = x_pt_f32_2[14] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        st_vals[15] = x_pt_f32_2[15] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_10 : 0.0f);
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[0]) : "f"(st_vals[0]), "f"(st_vals[1]), "f"(st_vals[2]), "f"(st_vals[3]), "f"(st_vals[4]), "f"(st_vals[5]), "f"(st_vals[6]), "f"(st_vals[7]));
                                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(st_words[1]) : "f"(st_vals[8]), "f"(st_vals[9]), "f"(st_vals[10]), "f"(st_vals[11]), "f"(st_vals[12]), "f"(st_vals[13]), "f"(st_vals[14]), "f"(st_vals[15]));
                                        *(reinterpret_cast<unsigned int*>(v_out + (row_head * 16 + (96 + (c - 3) * 16 >> 3))) + (0)) = st_words[0];
                                        *(reinterpret_cast<unsigned int*>(v_out + (row_head * 16 + (96 + (c - 3) * 16 >> 3) + 1)) + (0)) = st_words[1];
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                }
            }
        }
    }

    // Cleanup
}

}  // namespace h3_qkv_gemm_nvfp4_nvfp4_sm120a
#undef GROUP_M
#undef H3_QPA_INF
#undef NUM_AB_PIPE_STAGES
#undef SMEM_A_STAGE_OFF
#undef SMEM_A_STAGE_STAGE_BYTES
#undef SMEM_A_STAGE_STRIDE
#undef SMEM_B_STAGE_OFF
#undef SMEM_B_STAGE_STAGE_BYTES
#undef SMEM_B_STAGE_STRIDE
#undef SMEM_ROWSTAT_OFF
#undef SMEM_ROWSTAT_STAGE_BYTES
#undef SMEM_ROWSTAT_STRIDE
#undef SMEM_SFA_STAGE_OFF
#undef SMEM_SFA_STAGE_STAGE_BYTES
#undef SMEM_SFA_STAGE_STRIDE
#undef SMEM_SFB_STAGE_OFF
#undef SMEM_SFB_STAGE_STAGE_BYTES
#undef SMEM_SFB_STAGE_STRIDE
#undef SMEM_STAGING_OFF
#undef SMEM_STAGING_STAGE_BYTES
#undef SMEM_STAGING_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef ab_empty_addr
#undef ab_full_addr

#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>
#include <vector>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace {

constexpr int64_t kHidden = 5376;
constexpr int64_t kNumHeads = 56;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kQkvWidth = 21504;
constexpr int64_t kRopeDim = 96;
constexpr int64_t kSfBlock = 16;
constexpr int64_t kHiddenSf = kHidden / kSfBlock;  // 336 UE4M3 scales per activation row
constexpr int64_t kHeadSf = kHeadDim / kSfBlock;   // 8 UE4M3 scales per output head row
// TMA coordinates are 32-bit; the persistent tile counter is int32.
constexpr int64_t kMaxRows = int64_t{1} << 24;
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 128;
constexpr int kNTiles = 168;
constexpr int kSfbRowsPerNTile = 168;
constexpr int kQuantThreads = 128;
// Dynamic shared memory of the two quantization kernels (their SMEM_TOTAL identity macros).
constexpr int kQuantFp8SmemBytes = 128;
constexpr int kQuantNvfp4SmemBytes = 128;
constexpr int kGemmThreads = 288;
constexpr int kOutBf16 = 0;
constexpr int kOutE4m3 = 1;
constexpr int kOutNvfp4 = 2;

using GemmKernel = void (*)(CUtensorMap, CUtensorMap, CUtensorMap, CUtensorMap, float*, float*, __nv_bfloat16*,
                            __nv_bfloat16*, __nv_bfloat16*, unsigned int*, unsigned int*, unsigned int*, uint8_t*,
                            uint8_t*, uint8_t*, int, int, int, float, float, float, float, float, float, float, float);

struct GemmVariant {
  GemmKernel kernel;
  int dynamic_smem_bytes;
};

// [quant (0 = fp8, 1 = nvfp4)][out_mode (0 = bf16, 1 = e4m3, 2 = nvfp4)]
const GemmVariant kGemmVariants[2][3] = {
    {{h3_qkv_gemm_fp8_bf16_sm120a::kernel_h3_qkv_gemm_fused, 91648}, {h3_qkv_gemm_fp8_e4m3_sm120a::kernel_h3_qkv_gemm_fused, 91648}, {h3_qkv_gemm_fp8_nvfp4_sm120a::kernel_h3_qkv_gemm_fused, 91648}},
    {{h3_qkv_gemm_nvfp4_bf16_sm120a::kernel_h3_qkv_gemm_fused, 91648}, {h3_qkv_gemm_nvfp4_e4m3_sm120a::kernel_h3_qkv_gemm_fused, 91648}, {h3_qkv_gemm_nvfp4_nvfp4_sm120a::kernel_h3_qkv_gemm_fused, 91648}},
};

void CheckTensor(const TensorView& tensor, const char* name, DLDevice device, DLDataType dtype,
                 std::initializer_list<int64_t> shape) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError) << name << " must be a CUDA tensor";
  TVM_FFI_CHECK(tensor.device().device_id == device.device_id, ValueError)
      << name << " must be on the same CUDA device as x";
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dtype), ValueError)
      << name << " has the wrong dtype";
  TVM_FFI_CHECK(tensor.ndim() == static_cast<int>(shape.size()), ValueError)
      << name << " must have " << shape.size() << " dimensions";
  int64_t expected_stride = 1;
  int dim = tensor.ndim() - 1;
  for (auto it = std::rbegin(shape); it != std::rend(shape); ++it, --dim) {
    TVM_FFI_CHECK(tensor.size(dim) == *it, ValueError) << name << " has the wrong shape (dimension " << dim << ")";
    TVM_FFI_CHECK(tensor.size(dim) == 1 || tensor.stride(dim) == expected_stride, ValueError)
        << name << " must be contiguous";
    expected_stride *= *it;
  }
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0, ValueError)
      << name << " must be 16-byte aligned";
}

// 2-D byte tile map: rows of ``inner_bytes`` contiguous bytes, box = ``box_rows`` x ``box_inner`` bytes.
// Rows beyond the tensor are zero-filled by TMA (partial M tail tiles).
CUtensorMap EncodeByteTile(const void* base, int64_t inner_bytes, int64_t rows, uint32_t box_inner,
                           uint32_t box_rows, CUtensorMapSwizzle swizzle, const char* name) {
  uint64_t global_dim[2] = {static_cast<uint64_t>(inner_bytes), static_cast<uint64_t>(rows)};
  uint64_t global_strides[1] = {static_cast<uint64_t>(inner_bytes)};
  uint32_t box_dim[2] = {box_inner, box_rows};
  uint32_t element_strides[2] = {1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode the " << name << " tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

int ConfigureKernels() {
  static std::mutex mutex;
  static std::vector<std::pair<int, int>> configured_devices;
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to get the active CUDA device: " << cudaGetErrorString(status);
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto& entry : configured_devices) {
    if (entry.first == device) return entry.second;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to query CUDA device properties: " << cudaGetErrorString(status);
  TVM_FFI_CHECK(properties.major == 12, RuntimeError)
      << "MiniMax-H3 SM120 quantized pre-attention requires compute capability 12.x (GB202); got "
      << properties.major << "." << properties.minor;
  for (const auto& row : kGemmVariants) {
    for (const auto& variant : row) {
      status = cudaFuncSetAttribute(variant.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    variant.dynamic_smem_bytes);
      TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
          << "failed to opt in to dynamic shared memory: " << cudaGetErrorString(status);
    }
  }
  configured_devices.emplace_back(device, properties.multiProcessorCount);
  return properties.multiProcessorCount;
}

struct LaunchPlan {
  int quant_grid;
  int gemm_grid;
  int num_m_tiles;
  int total_tiles;
};

LaunchPlan MakeLaunchPlan(int64_t rows, int num_sms) {
  // Mirrors the Python launch_plan(): one persistent CTA per SM (ctas_per_sm = 1).
  const int64_t num_m_tiles = (rows + kBlockM - 1) / kBlockM;
  const int64_t total_tiles = num_m_tiles * kNTiles;
  LaunchPlan plan{};
  plan.quant_grid = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(rows, int64_t{8} * num_sms)));
  plan.gemm_grid = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(total_tiles, num_sms)));
  plan.num_m_tiles = static_cast<int>(num_m_tiles);
  plan.total_tiles = static_cast<int>(total_tiles);
  return plan;
}

struct CommonArgs {
  int64_t rows;
  DLDevice device;
  cudaStream_t stream;
  int num_sms;
  LaunchPlan plan;
  int out_mode;
  __nv_bfloat16* q_norm_weight;
  __nv_bfloat16* k_norm_weight;
  __nv_bfloat16* rope_cos_sin;
  unsigned int* q_out;
  unsigned int* k_out;
  unsigned int* v_out;
  uint8_t* q_sf;
  uint8_t* k_sf;
  uint8_t* v_sf;
};

// Shared validation of the BF16 operands and the caller-owned Q/K/V outputs for ``out_mode``.
CommonArgs CheckCommon(const TensorView& x, const TensorView& x_norm_weight, const TensorView& adaln_scale,
                       const TensorView& adaln_shift, const TensorView& adaln_index,
                       const TensorView& q_norm_weight, const TensorView& k_norm_weight,
                       const TensorView& rope_cos_sin, const TensorView& q, const TensorView& k,
                       const TensorView& v, const Optional<TensorView>& q_sf, const Optional<TensorView>& k_sf,
                       const Optional<TensorView>& v_sf, int64_t out_mode) {
  TVM_FFI_CHECK(x.ndim() == 2 && x.size(0) >= 1 && x.size(0) <= kMaxRows && x.size(1) == kHidden, ValueError)
      << "x must be [M, 5376] with 1 <= M <= " << kMaxRows;
  TVM_FFI_CHECK(out_mode == kOutBf16 || out_mode == kOutE4m3 || out_mode == kOutNvfp4, ValueError)
      << "out_mode must be 0 (bf16), 1 (e4m3) or 2 (nvfp4)";
  CommonArgs args{};
  args.rows = x.size(0);
  args.device = x.device();
  args.out_mode = static_cast<int>(out_mode);
  CheckTensor(x, "x", args.device, dl_bfloat16, {args.rows, kHidden});
  CheckTensor(x_norm_weight, "x_norm_weight", args.device, dl_bfloat16, {kHidden});
  TVM_FFI_CHECK(adaln_scale.ndim() == 2 && adaln_scale.size(0) >= 1 && adaln_scale.size(1) == kHidden, ValueError)
      << "adaln_scale must be [rows, 5376]";
  const int64_t adaln_rows = adaln_scale.size(0);
  CheckTensor(adaln_scale, "adaln_scale", args.device, dl_bfloat16, {adaln_rows, kHidden});
  CheckTensor(adaln_shift, "adaln_shift", args.device, dl_bfloat16, {adaln_rows, kHidden});
  CheckTensor(adaln_index, "adaln_index", args.device, dl_int32, {args.rows});
  CheckTensor(q_norm_weight, "q_norm_weight", args.device, dl_bfloat16, {kHeadDim});
  CheckTensor(k_norm_weight, "k_norm_weight", args.device, dl_bfloat16, {kHeadDim});
  CheckTensor(rope_cos_sin, "rope_cos_sin", args.device, dl_bfloat16, {args.rows, kRopeDim});
  if (out_mode == kOutBf16) {
    CheckTensor(q, "q", args.device, dl_bfloat16, {args.rows, kNumHeads, kHeadDim});
    CheckTensor(k, "k", args.device, dl_bfloat16, {args.rows, kNumHeads, kHeadDim});
    CheckTensor(v, "v", args.device, dl_bfloat16, {args.rows, kNumHeads, kHeadDim});
  } else if (out_mode == kOutE4m3) {
    CheckTensor(q, "q", args.device, dl_float8_e4m3fn, {args.rows, kNumHeads, kHeadDim});
    CheckTensor(k, "k", args.device, dl_float8_e4m3fn, {args.rows, kNumHeads, kHeadDim});
    CheckTensor(v, "v", args.device, dl_float8_e4m3fn, {args.rows, kNumHeads, kHeadDim});
  } else {
    CheckTensor(q, "q", args.device, dl_uint8, {args.rows, kNumHeads, kHeadDim / 2});
    CheckTensor(k, "k", args.device, dl_uint8, {args.rows, kNumHeads, kHeadDim / 2});
    CheckTensor(v, "v", args.device, dl_uint8, {args.rows, kNumHeads, kHeadDim / 2});
    TVM_FFI_CHECK(q_sf.has_value() && k_sf.has_value() && v_sf.has_value(), ValueError)
        << "q_sf, k_sf and v_sf are required for out_mode=2 (nvfp4)";
    CheckTensor(q_sf.value(), "q_sf", args.device, dl_uint8, {args.rows, kNumHeads, kHeadSf});
    CheckTensor(k_sf.value(), "k_sf", args.device, dl_uint8, {args.rows, kNumHeads, kHeadSf});
    CheckTensor(v_sf.value(), "v_sf", args.device, dl_uint8, {args.rows, kNumHeads, kHeadSf});
    args.q_sf = static_cast<uint8_t*>(q_sf.value().data_ptr());
    args.k_sf = static_cast<uint8_t*>(k_sf.value().data_ptr());
    args.v_sf = static_cast<uint8_t*>(v_sf.value().data_ptr());
  }
  args.q_norm_weight = static_cast<__nv_bfloat16*>(q_norm_weight.data_ptr());
  args.k_norm_weight = static_cast<__nv_bfloat16*>(k_norm_weight.data_ptr());
  args.rope_cos_sin = static_cast<__nv_bfloat16*>(rope_cos_sin.data_ptr());
  args.q_out = static_cast<unsigned int*>(q.data_ptr());
  args.k_out = static_cast<unsigned int*>(k.data_ptr());
  args.v_out = static_cast<unsigned int*>(v.data_ptr());
  args.num_sms = ConfigureKernels();
  args.stream = get_stream(args.device);
  args.plan = MakeLaunchPlan(args.rows, args.num_sms);
  return args;
}

void LaunchGemm(int quant, const CommonArgs& args, const CUtensorMap& a_map, const CUtensorMap& b_map,
                const CUtensorMap& sfa_map, const CUtensorMap& sfb_map, float* act_scale, float* w_scale,
                double eps, double alpha, double out_scale_q, double out_scale_k, double out_scale_v,
                double sf_mul_q, double sf_mul_k, double sf_mul_v, const char* what) {
  const GemmVariant variant = kGemmVariants[quant][args.out_mode];
  variant.kernel<<<dim3(args.plan.gemm_grid), dim3(kGemmThreads), variant.dynamic_smem_bytes, args.stream>>>(
      a_map, b_map, sfa_map, sfb_map, act_scale, w_scale, args.q_norm_weight, args.k_norm_weight,
      args.rope_cos_sin, args.q_out, args.k_out, args.v_out, args.q_sf, args.k_sf, args.v_sf,
      static_cast<int>(args.rows), args.plan.num_m_tiles, args.plan.total_tiles, static_cast<float>(eps),
      static_cast<float>(alpha), static_cast<float>(out_scale_q), static_cast<float>(out_scale_k),
      static_cast<float>(out_scale_v), static_cast<float>(sf_mul_q), static_cast<float>(sf_mul_k),
      static_cast<float>(sf_mul_v));
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << what << " GEMM launch failed: " << cudaGetErrorString(status);
}

}  // namespace

// FP8 W8A8 route.  x [M, 5376] BF16 -> act_q [M, 5376] E4M3 (per-token scale act_scale [M] FP32 =
// RN(amax / 448)) -> Q/K/V = epilogue(act_q @ qkv_weight_q^T * act_scale[m] * qkv_weight_scale[n]).
// qkv_weight_q: E4M3 [21504, 5376] (per-output-channel), qkv_weight_scale: FP32 [21504].
// out_mode 0: q, k, v BF16 [M, 56, 128].  1: E4M3 [M, 56, 128], stored = RN(value * out_scale_*).
// 2: NVFP4 u8 [M, 56, 64] + *_sf UE4M3 u8 [M, 56, 8] with sf = RN(block_amax * sf_mul_*) and
// code = RN(value * out_scale_* / sf) (FlashInfer fp4_quantize semantics; out_scale = global scale,
// sf_mul = global scale / 6).
void minimax_h3_sm120_fp8_pre_attention(TensorView x, TensorView x_norm_weight, TensorView adaln_scale,
                                        TensorView adaln_shift, TensorView adaln_index, TensorView qkv_weight_q,
                                        TensorView qkv_weight_scale, TensorView q_norm_weight,
                                        TensorView k_norm_weight, TensorView rope_cos_sin, TensorView act_q,
                                        TensorView act_scale, TensorView q, TensorView k, TensorView v,
                                        Optional<TensorView> q_sf, Optional<TensorView> k_sf,
                                        Optional<TensorView> v_sf, int64_t out_mode, double eps,
                                        double out_scale_q, double out_scale_k, double out_scale_v,
                                        double sf_mul_q, double sf_mul_k, double sf_mul_v) {
  const CommonArgs args = CheckCommon(x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, q_norm_weight,
                                      k_norm_weight, rope_cos_sin, q, k, v, q_sf, k_sf, v_sf, out_mode);
  CheckTensor(qkv_weight_q, "qkv_weight_q", args.device, dl_float8_e4m3fn, {kQkvWidth, kHidden});
  CheckTensor(qkv_weight_scale, "qkv_weight_scale", args.device, dl_float32, {kQkvWidth});
  CheckTensor(act_q, "act_q", args.device, dl_float8_e4m3fn, {args.rows, kHidden});
  CheckTensor(act_scale, "act_scale", args.device, dl_float32, {args.rows});
  ffi::CUDADeviceGuard device_guard(args.device.device_id);

  h3_norm_adaln_quant_fp8_sm120a::kernel_h3_norm_adaln_quant_fp8<<<dim3(args.plan.quant_grid), dim3(kQuantThreads), kQuantFp8SmemBytes, args.stream>>>(
      static_cast<__nv_bfloat16*>(x.data_ptr()), static_cast<__nv_bfloat16*>(x_norm_weight.data_ptr()),
      static_cast<__nv_bfloat16*>(adaln_scale.data_ptr()), static_cast<__nv_bfloat16*>(adaln_shift.data_ptr()),
      static_cast<int*>(adaln_index.data_ptr()), static_cast<unsigned int*>(act_q.data_ptr()),
      static_cast<float*>(act_scale.data_ptr()), static_cast<int>(args.rows), static_cast<float>(eps));
  cudaError_t status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "MiniMax-H3 FP8 norm/AdaLN quantization launch failed: " << cudaGetErrorString(status);

  const CUtensorMap a_map = EncodeByteTile(act_q.data_ptr(), kHidden, args.rows, kBlockK, kBlockM,
                                           CU_TENSOR_MAP_SWIZZLE_128B, "act_q");
  const CUtensorMap b_map = EncodeByteTile(qkv_weight_q.data_ptr(), kHidden, kQkvWidth, kBlockK, kBlockN,
                                           CU_TENSOR_MAP_SWIZZLE_128B, "qkv_weight_q");
  // The FP8 kernels never issue scale-tile loads; their descriptors only need valid parameter storage.
  const CUtensorMap unused_map{};
  LaunchGemm(0, args, a_map, b_map, unused_map, unused_map, static_cast<float*>(act_scale.data_ptr()),
             static_cast<float*>(qkv_weight_scale.data_ptr()), eps, 1.0, out_scale_q, out_scale_k, out_scale_v,
             sf_mul_q, sf_mul_k, sf_mul_v, "MiniMax-H3 FP8");
}

// NVFP4 route (FlashInfer conventions).  x -> act_q [M, 2688] u8 (E2M1x2) + act_sf [M, 336] UE4M3 u8
// (row-major, block 16) with the caller's activation global scale act_global_scale [1] FP32
// (448 * 6 / amax).  qkv_weight_q: u8 [21504, 2688], qkv_weight_sf: u8 with 21504 * 336 entries in the
// FlashInfer 128x4 swizzled layout (fp4_quantize(..., is_sf_swizzled_layout=True)).
// alpha = 1 / (act_global_scale * weight_global_scale) rescales the block-scaled accumulator.
void minimax_h3_sm120_nvfp4_pre_attention(TensorView x, TensorView x_norm_weight, TensorView adaln_scale,
                                          TensorView adaln_shift, TensorView adaln_index, TensorView qkv_weight_q,
                                          TensorView qkv_weight_sf, TensorView act_global_scale,
                                          TensorView q_norm_weight, TensorView k_norm_weight,
                                          TensorView rope_cos_sin, TensorView act_q, TensorView act_sf,
                                          TensorView q, TensorView k, TensorView v, Optional<TensorView> q_sf,
                                          Optional<TensorView> k_sf, Optional<TensorView> v_sf, int64_t out_mode,
                                          double eps, double alpha, double out_scale_q, double out_scale_k,
                                          double out_scale_v, double sf_mul_q, double sf_mul_k,
                                          double sf_mul_v) {
  const CommonArgs args = CheckCommon(x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, q_norm_weight,
                                      k_norm_weight, rope_cos_sin, q, k, v, q_sf, k_sf, v_sf, out_mode);
  CheckTensor(qkv_weight_q, "qkv_weight_q", args.device, dl_uint8, {kQkvWidth, kHidden / 2});
  TVM_FFI_CHECK(qkv_weight_sf.device().device_type == kDLCUDA &&
                    qkv_weight_sf.device().device_id == args.device.device_id, ValueError)
      << "qkv_weight_sf must be a CUDA tensor on the same device as x";
  TVM_FFI_CHECK(encode_dlpack_dtype(qkv_weight_sf.dtype()) == encode_dlpack_dtype(dl_uint8), ValueError)
      << "qkv_weight_sf must be uint8";
  int64_t sf_numel = 1;
  for (int dim = 0; dim < qkv_weight_sf.ndim(); ++dim) sf_numel *= qkv_weight_sf.size(dim);
  TVM_FFI_CHECK(sf_numel == kQkvWidth * kHiddenSf && qkv_weight_sf.IsContiguous(), ValueError)
      << "qkv_weight_sf must be a contiguous uint8 tensor with 21504 * 336 entries (128x4 swizzled layout)";
  CheckTensor(act_global_scale, "act_global_scale", args.device, dl_float32, {1});
  CheckTensor(act_q, "act_q", args.device, dl_uint8, {args.rows, kHidden / 2});
  CheckTensor(act_sf, "act_sf", args.device, dl_uint8, {args.rows, kHiddenSf});
  ffi::CUDADeviceGuard device_guard(args.device.device_id);

  h3_norm_adaln_quant_nvfp4_sm120a::kernel_h3_norm_adaln_quant_nvfp4<<<dim3(args.plan.quant_grid), dim3(kQuantThreads), kQuantNvfp4SmemBytes, args.stream>>>(
      static_cast<__nv_bfloat16*>(x.data_ptr()), static_cast<__nv_bfloat16*>(x_norm_weight.data_ptr()),
      static_cast<__nv_bfloat16*>(adaln_scale.data_ptr()), static_cast<__nv_bfloat16*>(adaln_shift.data_ptr()),
      static_cast<int*>(adaln_index.data_ptr()), static_cast<unsigned int*>(act_q.data_ptr()),
      static_cast<uint8_t*>(act_sf.data_ptr()), static_cast<float*>(act_global_scale.data_ptr()),
      static_cast<int>(args.rows), static_cast<float>(eps));
  cudaError_t status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "MiniMax-H3 NVFP4 norm/AdaLN quantization launch failed: " << cudaGetErrorString(status);

  const CUtensorMap a_map = EncodeByteTile(act_q.data_ptr(), kHidden / 2, args.rows, kBlockK, kBlockM,
                                           CU_TENSOR_MAP_SWIZZLE_128B, "act_q");
  const CUtensorMap b_map = EncodeByteTile(qkv_weight_q.data_ptr(), kHidden / 2, kQkvWidth, kBlockK, kBlockN,
                                           CU_TENSOR_MAP_SWIZZLE_128B, "qkv_weight_q");
  const CUtensorMap sfa_map = EncodeByteTile(act_sf.data_ptr(), kHiddenSf, args.rows, kBlockK * 2 / kSfBlock,
                                             kBlockM, CU_TENSOR_MAP_SWIZZLE_NONE, "act_sf");
  // The 128x4 swizzled weight scales are addressed as [168 N tiles x 168 rows, 256 bytes].
  const CUtensorMap sfb_map = EncodeByteTile(qkv_weight_sf.data_ptr(), 256, int64_t{kNTiles} * kSfbRowsPerNTile,
                                             256, 8, CU_TENSOR_MAP_SWIZZLE_NONE, "qkv_weight_sf");
  LaunchGemm(1, args, a_map, b_map, sfa_map, sfb_map, nullptr, nullptr, eps, alpha, out_scale_q, out_scale_k,
             out_scale_v, sf_mul_q, sf_mul_k, sf_mul_v, "MiniMax-H3 NVFP4");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_sm120_fp8_pre_attention, minimax_h3_sm120_fp8_pre_attention);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_sm120_nvfp4_pre_attention, minimax_h3_sm120_nvfp4_pre_attention);
// clang-format on
