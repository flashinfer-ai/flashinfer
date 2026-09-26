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
// MiniMax-H3 quantized fused RMSNorm + AdaLN + FC1 + SwiGLU for SM120 (GB202: RTX 5090 / RTX PRO 6000
// Blackwell), generated from the Cake kernel schedules.  Two launches per call:
//   1. norm_adaln_quant_{fp8,nvfp4}: RMSNorm(x) * w, indexed AdaLN (shift + n * (1 + scale)), then
//      per-token E4M3 (scale = amax / 448) or block-16 NVFP4 (FlashInfer fp4_quantize semantics)
//      activation quantization of the BF16 rows.
//   2. fc1_swiglu_gemm_{fp8,nvfp4}: persistent 128 x 128 output tiles of the 5376 -> 14336 SwiGLU
//      (each tile streams 256 prepacked FC1 weight rows: 128 gate rows and the 128 matching up rows
//      interleaved in groups of eight), mma.sync e4m3 / kind::mxf4nvf4 block-scaled, 64-byte-swizzled
//      TMA ring (4 stages FP8 / 3 stages NVFP4), one TMA producer warp + eight MMA warps, one CTA per
//      SM; the epilogue applies the dequant scales (per-token x per-row FP8 or alpha for NVFP4),
//      rounds h to BF16 and writes y = BF16(BF16(silu(h_gate)) * h_up) as BF16 [M, 14336].
// Device code: TMA, ldmatrix, mma.sync (kind::f8f6f4 / kind::mxf4nvf4), mbarrier pipelines.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

static_assert(sizeof(CUtensorMap) == 128, "CUDA tensor-map ABI size mismatch");
static_assert(alignof(CUtensorMap) >= 64, "CUDA tensor-map ABI requires at least 64-byte alignment");

namespace h3_fc1_norm_adaln_quant_fp8_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_FC1_INF CUDART_INF_F
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

}  // namespace h3_fc1_norm_adaln_quant_fp8_sm120a
#undef H3_FC1_INF
#undef NUM_MAIN_STAGES
#undef SMEM_AMAX_PARTIALS_OFF
#undef SMEM_AMAX_PARTIALS_STAGE_BYTES
#undef SMEM_AMAX_PARTIALS_STRIDE
#undef SMEM_PARTIALS_OFF
#undef SMEM_PARTIALS_STAGE_BYTES
#undef SMEM_PARTIALS_STRIDE
#undef SMEM_TOTAL
#undef THREADS

namespace h3_fc1_norm_adaln_quant_nvfp4_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_FC1_INF CUDART_INF_F
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

}  // namespace h3_fc1_norm_adaln_quant_nvfp4_sm120a
#undef H3_FC1_INF
#undef NUM_MAIN_STAGES
#undef SMEM_PARTIALS_OFF
#undef SMEM_PARTIALS_STAGE_BYTES
#undef SMEM_PARTIALS_STRIDE
#undef SMEM_TOTAL
#undef THREADS

namespace h3_fc1_swiglu_gemm_fp8_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_FC1_INF CUDART_INF_F
#define NUM_FULL_PIPE_STAGES 4
#define NUM_EMPTY_PIPE_STAGES 4
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 8192
#define SMEM_A_STAGE_STRIDE 8192
#define SMEM_B_STAGE_OFF 33792
#define SMEM_B_STAGE_STAGE_BYTES 16384
#define SMEM_B_STAGE_STRIDE 16384
#define SMEM_TOTAL 99328
#define THREADS 256
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


__device__ __forceinline__ uint32_t mbarrier_try_wait_plain(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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


__global__ __launch_bounds__(256, 1) void
kernel_h3_fc1_swiglu_gemm(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ act_scale, float* __restrict__ w_scale, unsigned int* __restrict__ out, int M, int num_m_tiles, int total_tiles, float alpha)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 32)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int B_stage_addr = smem + 33792;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'full_pipe' ---
            // ab_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'empty_pipe' ---
            // ab_empty: 4 barriers, init_count=8
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            mbarrier_init(smem + 48, 8);
            mbarrier_init(smem + 56, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: lead ----
    if (warp == 0) {
        { // lead_main
            unsigned int load_stage = 0;
            unsigned int load_g = 0;
            unsigned int mma_stage = 0;
            unsigned int mma_kk = 0;
            int warp_m = 0;
            int warp_n = 0;
            float accum[128];
            unsigned int a_frag[16];
            unsigned int b_frag[16];
            unsigned int _phase_ab_empty = 1;
            if (lane == 0) {
                for (int _prologue = 0; _prologue < 3; _prologue++) {
                    if (bid + (int)(load_g / 84) * num_bids < total_tiles) {
                        mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                        mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 24576);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(A_stage_addr + load_stage * 8192), "l"((&A)), "r"(((int)load_g - (int)(load_g / 84) * 84) * 64), "r"(((bid + (int)(load_g / 84) * num_bids) / (GROUP_M * 112) * GROUP_M + (bid + (int)(load_g / 84) * num_bids - (bid + (int)(load_g / 84) * num_bids) / (GROUP_M * 112) * (GROUP_M * 112)) % ((num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / (GROUP_M * 112) * GROUP_M : GROUP_M)) * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(B_stage_addr + load_stage * 16384), "l"((&B)), "r"(((int)load_g - (int)(load_g / 84) * 84) * 64), "r"((bid + (int)(load_g / 84) * num_bids - (bid + (int)(load_g / 84) * num_bids) / (GROUP_M * 112) * (GROUP_M * 112)) / ((num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / (GROUP_M * 112) * GROUP_M : GROUP_M) * 256), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                    }
                    load_stage += 1;
                    if (load_stage == 4) { load_stage = 0; _phase_ab_empty ^= 1; }
                    load_g = load_g + 1;
                }
            }
            __syncwarp();
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile = bid; tile < total_tiles; tile += num_bids) {
                int tile_m = tile / (GROUP_M * 112) * GROUP_M + (tile - tile / (GROUP_M * 112) * (GROUP_M * 112)) % ((num_m_tiles - tile / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 112) * GROUP_M : GROUP_M);
                int tile_n = (tile - tile / (GROUP_M * 112) * (GROUP_M * 112)) / ((num_m_tiles - tile / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 112) * GROUP_M : GROUP_M);
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
                accum[64] = 0.0f;
                accum[65] = 0.0f;
                accum[66] = 0.0f;
                accum[67] = 0.0f;
                accum[68] = 0.0f;
                accum[69] = 0.0f;
                accum[70] = 0.0f;
                accum[71] = 0.0f;
                accum[72] = 0.0f;
                accum[73] = 0.0f;
                accum[74] = 0.0f;
                accum[75] = 0.0f;
                accum[76] = 0.0f;
                accum[77] = 0.0f;
                accum[78] = 0.0f;
                accum[79] = 0.0f;
                accum[80] = 0.0f;
                accum[81] = 0.0f;
                accum[82] = 0.0f;
                accum[83] = 0.0f;
                accum[84] = 0.0f;
                accum[85] = 0.0f;
                accum[86] = 0.0f;
                accum[87] = 0.0f;
                accum[88] = 0.0f;
                accum[89] = 0.0f;
                accum[90] = 0.0f;
                accum[91] = 0.0f;
                accum[92] = 0.0f;
                accum[93] = 0.0f;
                accum[94] = 0.0f;
                accum[95] = 0.0f;
                accum[96] = 0.0f;
                accum[97] = 0.0f;
                accum[98] = 0.0f;
                accum[99] = 0.0f;
                accum[100] = 0.0f;
                accum[101] = 0.0f;
                accum[102] = 0.0f;
                accum[103] = 0.0f;
                accum[104] = 0.0f;
                accum[105] = 0.0f;
                accum[106] = 0.0f;
                accum[107] = 0.0f;
                accum[108] = 0.0f;
                accum[109] = 0.0f;
                accum[110] = 0.0f;
                accum[111] = 0.0f;
                accum[112] = 0.0f;
                accum[113] = 0.0f;
                accum[114] = 0.0f;
                accum[115] = 0.0f;
                accum[116] = 0.0f;
                accum[117] = 0.0f;
                accum[118] = 0.0f;
                accum[119] = 0.0f;
                accum[120] = 0.0f;
                accum[121] = 0.0f;
                accum[122] = 0.0f;
                accum[123] = 0.0f;
                accum[124] = 0.0f;
                accum[125] = 0.0f;
                accum[126] = 0.0f;
                accum[127] = 0.0f;
                #pragma unroll 1
                for (int k_tile = 0; k_tile < 84; k_tile++) {
                    if (lane == 0) {
                        if (bid + (int)(load_g / 84) * num_bids < total_tiles) {
                            mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 24576);
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(A_stage_addr + load_stage * 8192), "l"((&A)), "r"(((int)load_g - (int)(load_g / 84) * 84) * 64), "r"(((bid + (int)(load_g / 84) * num_bids) / (GROUP_M * 112) * GROUP_M + (bid + (int)(load_g / 84) * num_bids - (bid + (int)(load_g / 84) * num_bids) / (GROUP_M * 112) * (GROUP_M * 112)) % ((num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / (GROUP_M * 112) * GROUP_M : GROUP_M)) * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(B_stage_addr + load_stage * 16384), "l"((&B)), "r"(((int)load_g - (int)(load_g / 84) * 84) * 64), "r"((bid + (int)(load_g / 84) * num_bids - (bid + (int)(load_g / 84) * num_bids) / (GROUP_M * 112) * (GROUP_M * 112)) / ((num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / (GROUP_M * 112) * GROUP_M : GROUP_M) * 256), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        }
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_ab_empty ^= 1; }
                        load_g = load_g + 1;
                    }
                    __syncwarp();
                    mbarrier_wait(ab_full_addr + (mma_stage) * 8, _phase_ab_full);
                    for (int k_step = 0; k_step < 2; k_step++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(A_stage_addr + mma_stage * 8192 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                            : "r"(A_stage_addr + mma_stage * 8192 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                            : "r"(A_stage_addr + mma_stage * 8192 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                            : "r"(A_stage_addr + mma_stage * 8192 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
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
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    }
                    __syncwarp();
                    if (lane == 0) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 4) { mma_stage = 0; _phase_ab_full ^= 1; }
                    mma_kk = mma_kk + 1;
                }
                float sw[16];
                sw[0] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2];
                sw[2] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 8];
                sw[1] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 1];
                sw[3] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 1 + 8];
                sw[4] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 16];
                sw[6] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 16 + 8];
                sw[5] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 16 + 1];
                sw[7] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 16 + 1 + 8];
                sw[8] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 32];
                sw[10] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 32 + 8];
                sw[9] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 32 + 1];
                sw[11] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 32 + 1 + 8];
                sw[12] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 48];
                sw[14] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 48 + 8];
                sw[13] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 48 + 1];
                sw[15] = w_scale[tile_n * 256 + warp_n * 64 + (lane & 3) * 2 + 48 + 1 + 8];
                int row_lo_0 = tile_m * 128 + warp_m * 64 + (lane >> 2);
                int row_hi_0 = row_lo_0 + 8;
                float sa_lo_0 = ((row_lo_0 < M) ? act_scale[row_lo_0] : 0.0f);
                float sa_hi_0 = ((row_hi_0 < M) ? act_scale[row_hi_0] : 0.0f);
                float h_raw_0_0[8];
                h_raw_0_0[0] = accum[0] * sa_lo_0 * sw[0];
                h_raw_0_0[2] = accum[2] * sa_hi_0 * sw[0];
                h_raw_0_0[4] = accum[4] * sa_lo_0 * sw[2];
                h_raw_0_0[6] = accum[6] * sa_hi_0 * sw[2];
                h_raw_0_0[1] = accum[1] * sa_lo_0 * sw[1];
                h_raw_0_0[3] = accum[3] * sa_hi_0 * sw[1];
                h_raw_0_0[5] = accum[5] * sa_lo_0 * sw[3];
                h_raw_0_0[7] = accum[7] * sa_hi_0 * sw[3];
                uint32_t h_raw_0_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_0[_lp*2 + 0], h_raw_0_0[_lp*2+1 + 0]));
                    h_raw_0_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_0_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_0_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_0_0_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_0_0_bf16[_pair]));
                }
                float y_lo_0_0[2];
                float y_hi_0_0[2];
                float _exp2_0 = approx_exp2((-h_raw_0_0_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                float _exp2_1 = approx_exp2((-h_raw_0_0_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(h_raw_0_0_bf16_f32[0] * _rcp_0);
                float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(h_raw_0_0_bf16_f32[2] * _rcp_1);
                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                y_lo_0_0[0] = _cvt_f32_0 * h_raw_0_0_bf16_f32[4];
                y_hi_0_0[0] = _cvt_f32_1 * h_raw_0_0_bf16_f32[6];
                float _exp2_2 = approx_exp2((-h_raw_0_0_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                float _exp2_3 = approx_exp2((-h_raw_0_0_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_3 = approx_rcp(1.0f + _exp2_3);
                __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(h_raw_0_0_bf16_f32[1] * _rcp_2);
                float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(h_raw_0_0_bf16_f32[3] * _rcp_3);
                float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                y_lo_0_0[1] = _cvt_f32_2 * h_raw_0_0_bf16_f32[5];
                y_hi_0_0[1] = _cvt_f32_3 * h_raw_0_0_bf16_f32[7];
                uint32_t y_lo_0_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_0[_lp*2 + 0], y_lo_0_0[_lp*2+1 + 0]));
                    y_lo_0_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_0[_lp*2 + 0], y_hi_0_0[_lp*2+1 + 0]));
                    y_hi_0_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_0_0_bf16[0];
                }
                if (row_hi_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_0_0_bf16[0];
                }
                float h_raw_0_1[8];
                h_raw_0_1[0] = accum[8] * sa_lo_0 * sw[4];
                h_raw_0_1[2] = accum[10] * sa_hi_0 * sw[4];
                h_raw_0_1[4] = accum[12] * sa_lo_0 * sw[6];
                h_raw_0_1[6] = accum[14] * sa_hi_0 * sw[6];
                h_raw_0_1[1] = accum[9] * sa_lo_0 * sw[5];
                h_raw_0_1[3] = accum[11] * sa_hi_0 * sw[5];
                h_raw_0_1[5] = accum[13] * sa_lo_0 * sw[7];
                h_raw_0_1[7] = accum[15] * sa_hi_0 * sw[7];
                uint32_t h_raw_0_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_1[_lp*2 + 0], h_raw_0_1[_lp*2+1 + 0]));
                    h_raw_0_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_1_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_1_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_0_1_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_0_1_bf16[_pair]));
                }
                float y_lo_0_1[2];
                float y_hi_0_1[2];
                float _exp2_4 = approx_exp2((-h_raw_0_1_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                float _exp2_5 = approx_exp2((-h_raw_0_1_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(h_raw_0_1_bf16_f32[0] * _rcp_4);
                float _cvt_f32_4 = __bfloat162float(_cvt_bf16_4);
                __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(h_raw_0_1_bf16_f32[2] * _rcp_5);
                float _cvt_f32_5 = __bfloat162float(_cvt_bf16_5);
                y_lo_0_1[0] = _cvt_f32_4 * h_raw_0_1_bf16_f32[4];
                y_hi_0_1[0] = _cvt_f32_5 * h_raw_0_1_bf16_f32[6];
                float _exp2_6 = approx_exp2((-h_raw_0_1_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                float _exp2_7 = approx_exp2((-h_raw_0_1_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(h_raw_0_1_bf16_f32[1] * _rcp_6);
                float _cvt_f32_6 = __bfloat162float(_cvt_bf16_6);
                __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(h_raw_0_1_bf16_f32[3] * _rcp_7);
                float _cvt_f32_7 = __bfloat162float(_cvt_bf16_7);
                y_lo_0_1[1] = _cvt_f32_6 * h_raw_0_1_bf16_f32[5];
                y_hi_0_1[1] = _cvt_f32_7 * h_raw_0_1_bf16_f32[7];
                uint32_t y_lo_0_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_1[_lp*2 + 0], y_lo_0_1[_lp*2+1 + 0]));
                    y_lo_0_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_1[_lp*2 + 0], y_hi_0_1[_lp*2+1 + 0]));
                    y_hi_0_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_0_1_bf16[0];
                }
                if (row_hi_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_0_1_bf16[0];
                }
                float h_raw_0_2[8];
                h_raw_0_2[0] = accum[16] * sa_lo_0 * sw[8];
                h_raw_0_2[2] = accum[18] * sa_hi_0 * sw[8];
                h_raw_0_2[4] = accum[20] * sa_lo_0 * sw[10];
                h_raw_0_2[6] = accum[22] * sa_hi_0 * sw[10];
                h_raw_0_2[1] = accum[17] * sa_lo_0 * sw[9];
                h_raw_0_2[3] = accum[19] * sa_hi_0 * sw[9];
                h_raw_0_2[5] = accum[21] * sa_lo_0 * sw[11];
                h_raw_0_2[7] = accum[23] * sa_hi_0 * sw[11];
                uint32_t h_raw_0_2_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_2[_lp*2 + 0], h_raw_0_2[_lp*2+1 + 0]));
                    h_raw_0_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_2_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_2_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_0_2_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_0_2_bf16[_pair]));
                }
                float y_lo_0_2[2];
                float y_hi_0_2[2];
                float _exp2_8 = approx_exp2((-h_raw_0_2_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                float _exp2_9 = approx_exp2((-h_raw_0_2_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                __nv_bfloat16 _cvt_bf16_8 = __float2bfloat16(h_raw_0_2_bf16_f32[0] * _rcp_8);
                float _cvt_f32_8 = __bfloat162float(_cvt_bf16_8);
                __nv_bfloat16 _cvt_bf16_9 = __float2bfloat16(h_raw_0_2_bf16_f32[2] * _rcp_9);
                float _cvt_f32_9 = __bfloat162float(_cvt_bf16_9);
                y_lo_0_2[0] = _cvt_f32_8 * h_raw_0_2_bf16_f32[4];
                y_hi_0_2[0] = _cvt_f32_9 * h_raw_0_2_bf16_f32[6];
                float _exp2_10 = approx_exp2((-h_raw_0_2_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                float _exp2_11 = approx_exp2((-h_raw_0_2_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                __nv_bfloat16 _cvt_bf16_10 = __float2bfloat16(h_raw_0_2_bf16_f32[1] * _rcp_10);
                float _cvt_f32_10 = __bfloat162float(_cvt_bf16_10);
                __nv_bfloat16 _cvt_bf16_11 = __float2bfloat16(h_raw_0_2_bf16_f32[3] * _rcp_11);
                float _cvt_f32_11 = __bfloat162float(_cvt_bf16_11);
                y_lo_0_2[1] = _cvt_f32_10 * h_raw_0_2_bf16_f32[5];
                y_hi_0_2[1] = _cvt_f32_11 * h_raw_0_2_bf16_f32[7];
                uint32_t y_lo_0_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_2[_lp*2 + 0], y_lo_0_2[_lp*2+1 + 0]));
                    y_lo_0_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_2[_lp*2 + 0], y_hi_0_2[_lp*2+1 + 0]));
                    y_hi_0_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_0_2_bf16[0];
                }
                if (row_hi_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_0_2_bf16[0];
                }
                float h_raw_0_3[8];
                h_raw_0_3[0] = accum[24] * sa_lo_0 * sw[12];
                h_raw_0_3[2] = accum[26] * sa_hi_0 * sw[12];
                h_raw_0_3[4] = accum[28] * sa_lo_0 * sw[14];
                h_raw_0_3[6] = accum[30] * sa_hi_0 * sw[14];
                h_raw_0_3[1] = accum[25] * sa_lo_0 * sw[13];
                h_raw_0_3[3] = accum[27] * sa_hi_0 * sw[13];
                h_raw_0_3[5] = accum[29] * sa_lo_0 * sw[15];
                h_raw_0_3[7] = accum[31] * sa_hi_0 * sw[15];
                uint32_t h_raw_0_3_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_3[_lp*2 + 0], h_raw_0_3[_lp*2+1 + 0]));
                    h_raw_0_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_3_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_3_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_0_3_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_0_3_bf16[_pair]));
                }
                float y_lo_0_3[2];
                float y_hi_0_3[2];
                float _exp2_12 = approx_exp2((-h_raw_0_3_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                float _exp2_13 = approx_exp2((-h_raw_0_3_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                __nv_bfloat16 _cvt_bf16_12 = __float2bfloat16(h_raw_0_3_bf16_f32[0] * _rcp_12);
                float _cvt_f32_12 = __bfloat162float(_cvt_bf16_12);
                __nv_bfloat16 _cvt_bf16_13 = __float2bfloat16(h_raw_0_3_bf16_f32[2] * _rcp_13);
                float _cvt_f32_13 = __bfloat162float(_cvt_bf16_13);
                y_lo_0_3[0] = _cvt_f32_12 * h_raw_0_3_bf16_f32[4];
                y_hi_0_3[0] = _cvt_f32_13 * h_raw_0_3_bf16_f32[6];
                float _exp2_14 = approx_exp2((-h_raw_0_3_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                float _exp2_15 = approx_exp2((-h_raw_0_3_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                __nv_bfloat16 _cvt_bf16_14 = __float2bfloat16(h_raw_0_3_bf16_f32[1] * _rcp_14);
                float _cvt_f32_14 = __bfloat162float(_cvt_bf16_14);
                __nv_bfloat16 _cvt_bf16_15 = __float2bfloat16(h_raw_0_3_bf16_f32[3] * _rcp_15);
                float _cvt_f32_15 = __bfloat162float(_cvt_bf16_15);
                y_lo_0_3[1] = _cvt_f32_14 * h_raw_0_3_bf16_f32[5];
                y_hi_0_3[1] = _cvt_f32_15 * h_raw_0_3_bf16_f32[7];
                uint32_t y_lo_0_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_3[_lp*2 + 0], y_lo_0_3[_lp*2+1 + 0]));
                    y_lo_0_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_3[_lp*2 + 0], y_hi_0_3[_lp*2+1 + 0]));
                    y_hi_0_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_0_3_bf16[0];
                }
                if (row_hi_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_0_3_bf16[0];
                }
                int row_lo_1 = tile_m * 128 + warp_m * 64 + 16 + (lane >> 2);
                int row_hi_1 = row_lo_1 + 8;
                float sa_lo_1 = ((row_lo_1 < M) ? act_scale[row_lo_1] : 0.0f);
                float sa_hi_1 = ((row_hi_1 < M) ? act_scale[row_hi_1] : 0.0f);
                float h_raw_1_0[8];
                h_raw_1_0[0] = accum[32] * sa_lo_1 * sw[0];
                h_raw_1_0[2] = accum[34] * sa_hi_1 * sw[0];
                h_raw_1_0[4] = accum[36] * sa_lo_1 * sw[2];
                h_raw_1_0[6] = accum[38] * sa_hi_1 * sw[2];
                h_raw_1_0[1] = accum[33] * sa_lo_1 * sw[1];
                h_raw_1_0[3] = accum[35] * sa_hi_1 * sw[1];
                h_raw_1_0[5] = accum[37] * sa_lo_1 * sw[3];
                h_raw_1_0[7] = accum[39] * sa_hi_1 * sw[3];
                uint32_t h_raw_1_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_0[_lp*2 + 0], h_raw_1_0[_lp*2+1 + 0]));
                    h_raw_1_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_0_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_0_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_1_0_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_1_0_bf16[_pair]));
                }
                float y_lo_1_0[2];
                float y_hi_1_0[2];
                float _exp2_16 = approx_exp2((-h_raw_1_0_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_16 = approx_rcp(1.0f + _exp2_16);
                float _exp2_17 = approx_exp2((-h_raw_1_0_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_17 = approx_rcp(1.0f + _exp2_17);
                __nv_bfloat16 _cvt_bf16_16 = __float2bfloat16(h_raw_1_0_bf16_f32[0] * _rcp_16);
                float _cvt_f32_16 = __bfloat162float(_cvt_bf16_16);
                __nv_bfloat16 _cvt_bf16_17 = __float2bfloat16(h_raw_1_0_bf16_f32[2] * _rcp_17);
                float _cvt_f32_17 = __bfloat162float(_cvt_bf16_17);
                y_lo_1_0[0] = _cvt_f32_16 * h_raw_1_0_bf16_f32[4];
                y_hi_1_0[0] = _cvt_f32_17 * h_raw_1_0_bf16_f32[6];
                float _exp2_18 = approx_exp2((-h_raw_1_0_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_18 = approx_rcp(1.0f + _exp2_18);
                float _exp2_19 = approx_exp2((-h_raw_1_0_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_19 = approx_rcp(1.0f + _exp2_19);
                __nv_bfloat16 _cvt_bf16_18 = __float2bfloat16(h_raw_1_0_bf16_f32[1] * _rcp_18);
                float _cvt_f32_18 = __bfloat162float(_cvt_bf16_18);
                __nv_bfloat16 _cvt_bf16_19 = __float2bfloat16(h_raw_1_0_bf16_f32[3] * _rcp_19);
                float _cvt_f32_19 = __bfloat162float(_cvt_bf16_19);
                y_lo_1_0[1] = _cvt_f32_18 * h_raw_1_0_bf16_f32[5];
                y_hi_1_0[1] = _cvt_f32_19 * h_raw_1_0_bf16_f32[7];
                uint32_t y_lo_1_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_0[_lp*2 + 0], y_lo_1_0[_lp*2+1 + 0]));
                    y_lo_1_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_0[_lp*2 + 0], y_hi_1_0[_lp*2+1 + 0]));
                    y_hi_1_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_1_0_bf16[0];
                }
                if (row_hi_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_1_0_bf16[0];
                }
                float h_raw_1_1[8];
                h_raw_1_1[0] = accum[40] * sa_lo_1 * sw[4];
                h_raw_1_1[2] = accum[42] * sa_hi_1 * sw[4];
                h_raw_1_1[4] = accum[44] * sa_lo_1 * sw[6];
                h_raw_1_1[6] = accum[46] * sa_hi_1 * sw[6];
                h_raw_1_1[1] = accum[41] * sa_lo_1 * sw[5];
                h_raw_1_1[3] = accum[43] * sa_hi_1 * sw[5];
                h_raw_1_1[5] = accum[45] * sa_lo_1 * sw[7];
                h_raw_1_1[7] = accum[47] * sa_hi_1 * sw[7];
                uint32_t h_raw_1_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_1[_lp*2 + 0], h_raw_1_1[_lp*2+1 + 0]));
                    h_raw_1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_1_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_1_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_1_1_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_1_1_bf16[_pair]));
                }
                float y_lo_1_1[2];
                float y_hi_1_1[2];
                float _exp2_20 = approx_exp2((-h_raw_1_1_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_20 = approx_rcp(1.0f + _exp2_20);
                float _exp2_21 = approx_exp2((-h_raw_1_1_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_21 = approx_rcp(1.0f + _exp2_21);
                __nv_bfloat16 _cvt_bf16_20 = __float2bfloat16(h_raw_1_1_bf16_f32[0] * _rcp_20);
                float _cvt_f32_20 = __bfloat162float(_cvt_bf16_20);
                __nv_bfloat16 _cvt_bf16_21 = __float2bfloat16(h_raw_1_1_bf16_f32[2] * _rcp_21);
                float _cvt_f32_21 = __bfloat162float(_cvt_bf16_21);
                y_lo_1_1[0] = _cvt_f32_20 * h_raw_1_1_bf16_f32[4];
                y_hi_1_1[0] = _cvt_f32_21 * h_raw_1_1_bf16_f32[6];
                float _exp2_22 = approx_exp2((-h_raw_1_1_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_22 = approx_rcp(1.0f + _exp2_22);
                float _exp2_23 = approx_exp2((-h_raw_1_1_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_23 = approx_rcp(1.0f + _exp2_23);
                __nv_bfloat16 _cvt_bf16_22 = __float2bfloat16(h_raw_1_1_bf16_f32[1] * _rcp_22);
                float _cvt_f32_22 = __bfloat162float(_cvt_bf16_22);
                __nv_bfloat16 _cvt_bf16_23 = __float2bfloat16(h_raw_1_1_bf16_f32[3] * _rcp_23);
                float _cvt_f32_23 = __bfloat162float(_cvt_bf16_23);
                y_lo_1_1[1] = _cvt_f32_22 * h_raw_1_1_bf16_f32[5];
                y_hi_1_1[1] = _cvt_f32_23 * h_raw_1_1_bf16_f32[7];
                uint32_t y_lo_1_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_1[_lp*2 + 0], y_lo_1_1[_lp*2+1 + 0]));
                    y_lo_1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_1[_lp*2 + 0], y_hi_1_1[_lp*2+1 + 0]));
                    y_hi_1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_1_1_bf16[0];
                }
                if (row_hi_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_1_1_bf16[0];
                }
                float h_raw_1_2[8];
                h_raw_1_2[0] = accum[48] * sa_lo_1 * sw[8];
                h_raw_1_2[2] = accum[50] * sa_hi_1 * sw[8];
                h_raw_1_2[4] = accum[52] * sa_lo_1 * sw[10];
                h_raw_1_2[6] = accum[54] * sa_hi_1 * sw[10];
                h_raw_1_2[1] = accum[49] * sa_lo_1 * sw[9];
                h_raw_1_2[3] = accum[51] * sa_hi_1 * sw[9];
                h_raw_1_2[5] = accum[53] * sa_lo_1 * sw[11];
                h_raw_1_2[7] = accum[55] * sa_hi_1 * sw[11];
                uint32_t h_raw_1_2_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_2[_lp*2 + 0], h_raw_1_2[_lp*2+1 + 0]));
                    h_raw_1_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_2_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_2_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_1_2_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_1_2_bf16[_pair]));
                }
                float y_lo_1_2[2];
                float y_hi_1_2[2];
                float _exp2_24 = approx_exp2((-h_raw_1_2_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_24 = approx_rcp(1.0f + _exp2_24);
                float _exp2_25 = approx_exp2((-h_raw_1_2_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_25 = approx_rcp(1.0f + _exp2_25);
                __nv_bfloat16 _cvt_bf16_24 = __float2bfloat16(h_raw_1_2_bf16_f32[0] * _rcp_24);
                float _cvt_f32_24 = __bfloat162float(_cvt_bf16_24);
                __nv_bfloat16 _cvt_bf16_25 = __float2bfloat16(h_raw_1_2_bf16_f32[2] * _rcp_25);
                float _cvt_f32_25 = __bfloat162float(_cvt_bf16_25);
                y_lo_1_2[0] = _cvt_f32_24 * h_raw_1_2_bf16_f32[4];
                y_hi_1_2[0] = _cvt_f32_25 * h_raw_1_2_bf16_f32[6];
                float _exp2_26 = approx_exp2((-h_raw_1_2_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_26 = approx_rcp(1.0f + _exp2_26);
                float _exp2_27 = approx_exp2((-h_raw_1_2_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_27 = approx_rcp(1.0f + _exp2_27);
                __nv_bfloat16 _cvt_bf16_26 = __float2bfloat16(h_raw_1_2_bf16_f32[1] * _rcp_26);
                float _cvt_f32_26 = __bfloat162float(_cvt_bf16_26);
                __nv_bfloat16 _cvt_bf16_27 = __float2bfloat16(h_raw_1_2_bf16_f32[3] * _rcp_27);
                float _cvt_f32_27 = __bfloat162float(_cvt_bf16_27);
                y_lo_1_2[1] = _cvt_f32_26 * h_raw_1_2_bf16_f32[5];
                y_hi_1_2[1] = _cvt_f32_27 * h_raw_1_2_bf16_f32[7];
                uint32_t y_lo_1_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_2[_lp*2 + 0], y_lo_1_2[_lp*2+1 + 0]));
                    y_lo_1_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_2[_lp*2 + 0], y_hi_1_2[_lp*2+1 + 0]));
                    y_hi_1_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_1_2_bf16[0];
                }
                if (row_hi_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_1_2_bf16[0];
                }
                float h_raw_1_3[8];
                h_raw_1_3[0] = accum[56] * sa_lo_1 * sw[12];
                h_raw_1_3[2] = accum[58] * sa_hi_1 * sw[12];
                h_raw_1_3[4] = accum[60] * sa_lo_1 * sw[14];
                h_raw_1_3[6] = accum[62] * sa_hi_1 * sw[14];
                h_raw_1_3[1] = accum[57] * sa_lo_1 * sw[13];
                h_raw_1_3[3] = accum[59] * sa_hi_1 * sw[13];
                h_raw_1_3[5] = accum[61] * sa_lo_1 * sw[15];
                h_raw_1_3[7] = accum[63] * sa_hi_1 * sw[15];
                uint32_t h_raw_1_3_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_3[_lp*2 + 0], h_raw_1_3[_lp*2+1 + 0]));
                    h_raw_1_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_3_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_3_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_1_3_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_1_3_bf16[_pair]));
                }
                float y_lo_1_3[2];
                float y_hi_1_3[2];
                float _exp2_28 = approx_exp2((-h_raw_1_3_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_28 = approx_rcp(1.0f + _exp2_28);
                float _exp2_29 = approx_exp2((-h_raw_1_3_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_29 = approx_rcp(1.0f + _exp2_29);
                __nv_bfloat16 _cvt_bf16_28 = __float2bfloat16(h_raw_1_3_bf16_f32[0] * _rcp_28);
                float _cvt_f32_28 = __bfloat162float(_cvt_bf16_28);
                __nv_bfloat16 _cvt_bf16_29 = __float2bfloat16(h_raw_1_3_bf16_f32[2] * _rcp_29);
                float _cvt_f32_29 = __bfloat162float(_cvt_bf16_29);
                y_lo_1_3[0] = _cvt_f32_28 * h_raw_1_3_bf16_f32[4];
                y_hi_1_3[0] = _cvt_f32_29 * h_raw_1_3_bf16_f32[6];
                float _exp2_30 = approx_exp2((-h_raw_1_3_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_30 = approx_rcp(1.0f + _exp2_30);
                float _exp2_31 = approx_exp2((-h_raw_1_3_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_31 = approx_rcp(1.0f + _exp2_31);
                __nv_bfloat16 _cvt_bf16_30 = __float2bfloat16(h_raw_1_3_bf16_f32[1] * _rcp_30);
                float _cvt_f32_30 = __bfloat162float(_cvt_bf16_30);
                __nv_bfloat16 _cvt_bf16_31 = __float2bfloat16(h_raw_1_3_bf16_f32[3] * _rcp_31);
                float _cvt_f32_31 = __bfloat162float(_cvt_bf16_31);
                y_lo_1_3[1] = _cvt_f32_30 * h_raw_1_3_bf16_f32[5];
                y_hi_1_3[1] = _cvt_f32_31 * h_raw_1_3_bf16_f32[7];
                uint32_t y_lo_1_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_3[_lp*2 + 0], y_lo_1_3[_lp*2+1 + 0]));
                    y_lo_1_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_3[_lp*2 + 0], y_hi_1_3[_lp*2+1 + 0]));
                    y_hi_1_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_1_3_bf16[0];
                }
                if (row_hi_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_1_3_bf16[0];
                }
                int row_lo_2 = tile_m * 128 + warp_m * 64 + 32 + (lane >> 2);
                int row_hi_2 = row_lo_2 + 8;
                float sa_lo_2 = ((row_lo_2 < M) ? act_scale[row_lo_2] : 0.0f);
                float sa_hi_2 = ((row_hi_2 < M) ? act_scale[row_hi_2] : 0.0f);
                float h_raw_2_0[8];
                h_raw_2_0[0] = accum[64] * sa_lo_2 * sw[0];
                h_raw_2_0[2] = accum[66] * sa_hi_2 * sw[0];
                h_raw_2_0[4] = accum[68] * sa_lo_2 * sw[2];
                h_raw_2_0[6] = accum[70] * sa_hi_2 * sw[2];
                h_raw_2_0[1] = accum[65] * sa_lo_2 * sw[1];
                h_raw_2_0[3] = accum[67] * sa_hi_2 * sw[1];
                h_raw_2_0[5] = accum[69] * sa_lo_2 * sw[3];
                h_raw_2_0[7] = accum[71] * sa_hi_2 * sw[3];
                uint32_t h_raw_2_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_0[_lp*2 + 0], h_raw_2_0[_lp*2+1 + 0]));
                    h_raw_2_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_0_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_0_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_2_0_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_2_0_bf16[_pair]));
                }
                float y_lo_2_0[2];
                float y_hi_2_0[2];
                float _exp2_32 = approx_exp2((-h_raw_2_0_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_32 = approx_rcp(1.0f + _exp2_32);
                float _exp2_33 = approx_exp2((-h_raw_2_0_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_33 = approx_rcp(1.0f + _exp2_33);
                __nv_bfloat16 _cvt_bf16_32 = __float2bfloat16(h_raw_2_0_bf16_f32[0] * _rcp_32);
                float _cvt_f32_32 = __bfloat162float(_cvt_bf16_32);
                __nv_bfloat16 _cvt_bf16_33 = __float2bfloat16(h_raw_2_0_bf16_f32[2] * _rcp_33);
                float _cvt_f32_33 = __bfloat162float(_cvt_bf16_33);
                y_lo_2_0[0] = _cvt_f32_32 * h_raw_2_0_bf16_f32[4];
                y_hi_2_0[0] = _cvt_f32_33 * h_raw_2_0_bf16_f32[6];
                float _exp2_34 = approx_exp2((-h_raw_2_0_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_34 = approx_rcp(1.0f + _exp2_34);
                float _exp2_35 = approx_exp2((-h_raw_2_0_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_35 = approx_rcp(1.0f + _exp2_35);
                __nv_bfloat16 _cvt_bf16_34 = __float2bfloat16(h_raw_2_0_bf16_f32[1] * _rcp_34);
                float _cvt_f32_34 = __bfloat162float(_cvt_bf16_34);
                __nv_bfloat16 _cvt_bf16_35 = __float2bfloat16(h_raw_2_0_bf16_f32[3] * _rcp_35);
                float _cvt_f32_35 = __bfloat162float(_cvt_bf16_35);
                y_lo_2_0[1] = _cvt_f32_34 * h_raw_2_0_bf16_f32[5];
                y_hi_2_0[1] = _cvt_f32_35 * h_raw_2_0_bf16_f32[7];
                uint32_t y_lo_2_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_0[_lp*2 + 0], y_lo_2_0[_lp*2+1 + 0]));
                    y_lo_2_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_0[_lp*2 + 0], y_hi_2_0[_lp*2+1 + 0]));
                    y_hi_2_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_2_0_bf16[0];
                }
                if (row_hi_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_2_0_bf16[0];
                }
                float h_raw_2_1[8];
                h_raw_2_1[0] = accum[72] * sa_lo_2 * sw[4];
                h_raw_2_1[2] = accum[74] * sa_hi_2 * sw[4];
                h_raw_2_1[4] = accum[76] * sa_lo_2 * sw[6];
                h_raw_2_1[6] = accum[78] * sa_hi_2 * sw[6];
                h_raw_2_1[1] = accum[73] * sa_lo_2 * sw[5];
                h_raw_2_1[3] = accum[75] * sa_hi_2 * sw[5];
                h_raw_2_1[5] = accum[77] * sa_lo_2 * sw[7];
                h_raw_2_1[7] = accum[79] * sa_hi_2 * sw[7];
                uint32_t h_raw_2_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_1[_lp*2 + 0], h_raw_2_1[_lp*2+1 + 0]));
                    h_raw_2_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_1_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_1_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_2_1_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_2_1_bf16[_pair]));
                }
                float y_lo_2_1[2];
                float y_hi_2_1[2];
                float _exp2_36 = approx_exp2((-h_raw_2_1_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_36 = approx_rcp(1.0f + _exp2_36);
                float _exp2_37 = approx_exp2((-h_raw_2_1_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_37 = approx_rcp(1.0f + _exp2_37);
                __nv_bfloat16 _cvt_bf16_36 = __float2bfloat16(h_raw_2_1_bf16_f32[0] * _rcp_36);
                float _cvt_f32_36 = __bfloat162float(_cvt_bf16_36);
                __nv_bfloat16 _cvt_bf16_37 = __float2bfloat16(h_raw_2_1_bf16_f32[2] * _rcp_37);
                float _cvt_f32_37 = __bfloat162float(_cvt_bf16_37);
                y_lo_2_1[0] = _cvt_f32_36 * h_raw_2_1_bf16_f32[4];
                y_hi_2_1[0] = _cvt_f32_37 * h_raw_2_1_bf16_f32[6];
                float _exp2_38 = approx_exp2((-h_raw_2_1_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_38 = approx_rcp(1.0f + _exp2_38);
                float _exp2_39 = approx_exp2((-h_raw_2_1_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_39 = approx_rcp(1.0f + _exp2_39);
                __nv_bfloat16 _cvt_bf16_38 = __float2bfloat16(h_raw_2_1_bf16_f32[1] * _rcp_38);
                float _cvt_f32_38 = __bfloat162float(_cvt_bf16_38);
                __nv_bfloat16 _cvt_bf16_39 = __float2bfloat16(h_raw_2_1_bf16_f32[3] * _rcp_39);
                float _cvt_f32_39 = __bfloat162float(_cvt_bf16_39);
                y_lo_2_1[1] = _cvt_f32_38 * h_raw_2_1_bf16_f32[5];
                y_hi_2_1[1] = _cvt_f32_39 * h_raw_2_1_bf16_f32[7];
                uint32_t y_lo_2_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_1[_lp*2 + 0], y_lo_2_1[_lp*2+1 + 0]));
                    y_lo_2_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_1[_lp*2 + 0], y_hi_2_1[_lp*2+1 + 0]));
                    y_hi_2_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_2_1_bf16[0];
                }
                if (row_hi_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_2_1_bf16[0];
                }
                float h_raw_2_2[8];
                h_raw_2_2[0] = accum[80] * sa_lo_2 * sw[8];
                h_raw_2_2[2] = accum[82] * sa_hi_2 * sw[8];
                h_raw_2_2[4] = accum[84] * sa_lo_2 * sw[10];
                h_raw_2_2[6] = accum[86] * sa_hi_2 * sw[10];
                h_raw_2_2[1] = accum[81] * sa_lo_2 * sw[9];
                h_raw_2_2[3] = accum[83] * sa_hi_2 * sw[9];
                h_raw_2_2[5] = accum[85] * sa_lo_2 * sw[11];
                h_raw_2_2[7] = accum[87] * sa_hi_2 * sw[11];
                uint32_t h_raw_2_2_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_2[_lp*2 + 0], h_raw_2_2[_lp*2+1 + 0]));
                    h_raw_2_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_2_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_2_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_2_2_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_2_2_bf16[_pair]));
                }
                float y_lo_2_2[2];
                float y_hi_2_2[2];
                float _exp2_40 = approx_exp2((-h_raw_2_2_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_40 = approx_rcp(1.0f + _exp2_40);
                float _exp2_41 = approx_exp2((-h_raw_2_2_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_41 = approx_rcp(1.0f + _exp2_41);
                __nv_bfloat16 _cvt_bf16_40 = __float2bfloat16(h_raw_2_2_bf16_f32[0] * _rcp_40);
                float _cvt_f32_40 = __bfloat162float(_cvt_bf16_40);
                __nv_bfloat16 _cvt_bf16_41 = __float2bfloat16(h_raw_2_2_bf16_f32[2] * _rcp_41);
                float _cvt_f32_41 = __bfloat162float(_cvt_bf16_41);
                y_lo_2_2[0] = _cvt_f32_40 * h_raw_2_2_bf16_f32[4];
                y_hi_2_2[0] = _cvt_f32_41 * h_raw_2_2_bf16_f32[6];
                float _exp2_42 = approx_exp2((-h_raw_2_2_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_42 = approx_rcp(1.0f + _exp2_42);
                float _exp2_43 = approx_exp2((-h_raw_2_2_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_43 = approx_rcp(1.0f + _exp2_43);
                __nv_bfloat16 _cvt_bf16_42 = __float2bfloat16(h_raw_2_2_bf16_f32[1] * _rcp_42);
                float _cvt_f32_42 = __bfloat162float(_cvt_bf16_42);
                __nv_bfloat16 _cvt_bf16_43 = __float2bfloat16(h_raw_2_2_bf16_f32[3] * _rcp_43);
                float _cvt_f32_43 = __bfloat162float(_cvt_bf16_43);
                y_lo_2_2[1] = _cvt_f32_42 * h_raw_2_2_bf16_f32[5];
                y_hi_2_2[1] = _cvt_f32_43 * h_raw_2_2_bf16_f32[7];
                uint32_t y_lo_2_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_2[_lp*2 + 0], y_lo_2_2[_lp*2+1 + 0]));
                    y_lo_2_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_2[_lp*2 + 0], y_hi_2_2[_lp*2+1 + 0]));
                    y_hi_2_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_2_2_bf16[0];
                }
                if (row_hi_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_2_2_bf16[0];
                }
                float h_raw_2_3[8];
                h_raw_2_3[0] = accum[88] * sa_lo_2 * sw[12];
                h_raw_2_3[2] = accum[90] * sa_hi_2 * sw[12];
                h_raw_2_3[4] = accum[92] * sa_lo_2 * sw[14];
                h_raw_2_3[6] = accum[94] * sa_hi_2 * sw[14];
                h_raw_2_3[1] = accum[89] * sa_lo_2 * sw[13];
                h_raw_2_3[3] = accum[91] * sa_hi_2 * sw[13];
                h_raw_2_3[5] = accum[93] * sa_lo_2 * sw[15];
                h_raw_2_3[7] = accum[95] * sa_hi_2 * sw[15];
                uint32_t h_raw_2_3_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_3[_lp*2 + 0], h_raw_2_3[_lp*2+1 + 0]));
                    h_raw_2_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_3_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_3_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_2_3_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_2_3_bf16[_pair]));
                }
                float y_lo_2_3[2];
                float y_hi_2_3[2];
                float _exp2_44 = approx_exp2((-h_raw_2_3_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_44 = approx_rcp(1.0f + _exp2_44);
                float _exp2_45 = approx_exp2((-h_raw_2_3_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_45 = approx_rcp(1.0f + _exp2_45);
                __nv_bfloat16 _cvt_bf16_44 = __float2bfloat16(h_raw_2_3_bf16_f32[0] * _rcp_44);
                float _cvt_f32_44 = __bfloat162float(_cvt_bf16_44);
                __nv_bfloat16 _cvt_bf16_45 = __float2bfloat16(h_raw_2_3_bf16_f32[2] * _rcp_45);
                float _cvt_f32_45 = __bfloat162float(_cvt_bf16_45);
                y_lo_2_3[0] = _cvt_f32_44 * h_raw_2_3_bf16_f32[4];
                y_hi_2_3[0] = _cvt_f32_45 * h_raw_2_3_bf16_f32[6];
                float _exp2_46 = approx_exp2((-h_raw_2_3_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_46 = approx_rcp(1.0f + _exp2_46);
                float _exp2_47 = approx_exp2((-h_raw_2_3_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_47 = approx_rcp(1.0f + _exp2_47);
                __nv_bfloat16 _cvt_bf16_46 = __float2bfloat16(h_raw_2_3_bf16_f32[1] * _rcp_46);
                float _cvt_f32_46 = __bfloat162float(_cvt_bf16_46);
                __nv_bfloat16 _cvt_bf16_47 = __float2bfloat16(h_raw_2_3_bf16_f32[3] * _rcp_47);
                float _cvt_f32_47 = __bfloat162float(_cvt_bf16_47);
                y_lo_2_3[1] = _cvt_f32_46 * h_raw_2_3_bf16_f32[5];
                y_hi_2_3[1] = _cvt_f32_47 * h_raw_2_3_bf16_f32[7];
                uint32_t y_lo_2_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_3[_lp*2 + 0], y_lo_2_3[_lp*2+1 + 0]));
                    y_lo_2_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_3[_lp*2 + 0], y_hi_2_3[_lp*2+1 + 0]));
                    y_hi_2_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_2_3_bf16[0];
                }
                if (row_hi_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_2_3_bf16[0];
                }
                int row_lo_3 = tile_m * 128 + warp_m * 64 + 48 + (lane >> 2);
                int row_hi_3 = row_lo_3 + 8;
                float sa_lo_3 = ((row_lo_3 < M) ? act_scale[row_lo_3] : 0.0f);
                float sa_hi_3 = ((row_hi_3 < M) ? act_scale[row_hi_3] : 0.0f);
                float h_raw_3_0[8];
                h_raw_3_0[0] = accum[96] * sa_lo_3 * sw[0];
                h_raw_3_0[2] = accum[98] * sa_hi_3 * sw[0];
                h_raw_3_0[4] = accum[100] * sa_lo_3 * sw[2];
                h_raw_3_0[6] = accum[102] * sa_hi_3 * sw[2];
                h_raw_3_0[1] = accum[97] * sa_lo_3 * sw[1];
                h_raw_3_0[3] = accum[99] * sa_hi_3 * sw[1];
                h_raw_3_0[5] = accum[101] * sa_lo_3 * sw[3];
                h_raw_3_0[7] = accum[103] * sa_hi_3 * sw[3];
                uint32_t h_raw_3_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_0[_lp*2 + 0], h_raw_3_0[_lp*2+1 + 0]));
                    h_raw_3_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_0_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_0_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_3_0_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_3_0_bf16[_pair]));
                }
                float y_lo_3_0[2];
                float y_hi_3_0[2];
                float _exp2_48 = approx_exp2((-h_raw_3_0_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_48 = approx_rcp(1.0f + _exp2_48);
                float _exp2_49 = approx_exp2((-h_raw_3_0_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_49 = approx_rcp(1.0f + _exp2_49);
                __nv_bfloat16 _cvt_bf16_48 = __float2bfloat16(h_raw_3_0_bf16_f32[0] * _rcp_48);
                float _cvt_f32_48 = __bfloat162float(_cvt_bf16_48);
                __nv_bfloat16 _cvt_bf16_49 = __float2bfloat16(h_raw_3_0_bf16_f32[2] * _rcp_49);
                float _cvt_f32_49 = __bfloat162float(_cvt_bf16_49);
                y_lo_3_0[0] = _cvt_f32_48 * h_raw_3_0_bf16_f32[4];
                y_hi_3_0[0] = _cvt_f32_49 * h_raw_3_0_bf16_f32[6];
                float _exp2_50 = approx_exp2((-h_raw_3_0_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_50 = approx_rcp(1.0f + _exp2_50);
                float _exp2_51 = approx_exp2((-h_raw_3_0_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_51 = approx_rcp(1.0f + _exp2_51);
                __nv_bfloat16 _cvt_bf16_50 = __float2bfloat16(h_raw_3_0_bf16_f32[1] * _rcp_50);
                float _cvt_f32_50 = __bfloat162float(_cvt_bf16_50);
                __nv_bfloat16 _cvt_bf16_51 = __float2bfloat16(h_raw_3_0_bf16_f32[3] * _rcp_51);
                float _cvt_f32_51 = __bfloat162float(_cvt_bf16_51);
                y_lo_3_0[1] = _cvt_f32_50 * h_raw_3_0_bf16_f32[5];
                y_hi_3_0[1] = _cvt_f32_51 * h_raw_3_0_bf16_f32[7];
                uint32_t y_lo_3_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_0[_lp*2 + 0], y_lo_3_0[_lp*2+1 + 0]));
                    y_lo_3_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_0[_lp*2 + 0], y_hi_3_0[_lp*2+1 + 0]));
                    y_hi_3_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_3_0_bf16[0];
                }
                if (row_hi_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_3_0_bf16[0];
                }
                float h_raw_3_1[8];
                h_raw_3_1[0] = accum[104] * sa_lo_3 * sw[4];
                h_raw_3_1[2] = accum[106] * sa_hi_3 * sw[4];
                h_raw_3_1[4] = accum[108] * sa_lo_3 * sw[6];
                h_raw_3_1[6] = accum[110] * sa_hi_3 * sw[6];
                h_raw_3_1[1] = accum[105] * sa_lo_3 * sw[5];
                h_raw_3_1[3] = accum[107] * sa_hi_3 * sw[5];
                h_raw_3_1[5] = accum[109] * sa_lo_3 * sw[7];
                h_raw_3_1[7] = accum[111] * sa_hi_3 * sw[7];
                uint32_t h_raw_3_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_1[_lp*2 + 0], h_raw_3_1[_lp*2+1 + 0]));
                    h_raw_3_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_1_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_1_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_3_1_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_3_1_bf16[_pair]));
                }
                float y_lo_3_1[2];
                float y_hi_3_1[2];
                float _exp2_52 = approx_exp2((-h_raw_3_1_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_52 = approx_rcp(1.0f + _exp2_52);
                float _exp2_53 = approx_exp2((-h_raw_3_1_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_53 = approx_rcp(1.0f + _exp2_53);
                __nv_bfloat16 _cvt_bf16_52 = __float2bfloat16(h_raw_3_1_bf16_f32[0] * _rcp_52);
                float _cvt_f32_52 = __bfloat162float(_cvt_bf16_52);
                __nv_bfloat16 _cvt_bf16_53 = __float2bfloat16(h_raw_3_1_bf16_f32[2] * _rcp_53);
                float _cvt_f32_53 = __bfloat162float(_cvt_bf16_53);
                y_lo_3_1[0] = _cvt_f32_52 * h_raw_3_1_bf16_f32[4];
                y_hi_3_1[0] = _cvt_f32_53 * h_raw_3_1_bf16_f32[6];
                float _exp2_54 = approx_exp2((-h_raw_3_1_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_54 = approx_rcp(1.0f + _exp2_54);
                float _exp2_55 = approx_exp2((-h_raw_3_1_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_55 = approx_rcp(1.0f + _exp2_55);
                __nv_bfloat16 _cvt_bf16_54 = __float2bfloat16(h_raw_3_1_bf16_f32[1] * _rcp_54);
                float _cvt_f32_54 = __bfloat162float(_cvt_bf16_54);
                __nv_bfloat16 _cvt_bf16_55 = __float2bfloat16(h_raw_3_1_bf16_f32[3] * _rcp_55);
                float _cvt_f32_55 = __bfloat162float(_cvt_bf16_55);
                y_lo_3_1[1] = _cvt_f32_54 * h_raw_3_1_bf16_f32[5];
                y_hi_3_1[1] = _cvt_f32_55 * h_raw_3_1_bf16_f32[7];
                uint32_t y_lo_3_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_1[_lp*2 + 0], y_lo_3_1[_lp*2+1 + 0]));
                    y_lo_3_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_1[_lp*2 + 0], y_hi_3_1[_lp*2+1 + 0]));
                    y_hi_3_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_3_1_bf16[0];
                }
                if (row_hi_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_3_1_bf16[0];
                }
                float h_raw_3_2[8];
                h_raw_3_2[0] = accum[112] * sa_lo_3 * sw[8];
                h_raw_3_2[2] = accum[114] * sa_hi_3 * sw[8];
                h_raw_3_2[4] = accum[116] * sa_lo_3 * sw[10];
                h_raw_3_2[6] = accum[118] * sa_hi_3 * sw[10];
                h_raw_3_2[1] = accum[113] * sa_lo_3 * sw[9];
                h_raw_3_2[3] = accum[115] * sa_hi_3 * sw[9];
                h_raw_3_2[5] = accum[117] * sa_lo_3 * sw[11];
                h_raw_3_2[7] = accum[119] * sa_hi_3 * sw[11];
                uint32_t h_raw_3_2_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_2[_lp*2 + 0], h_raw_3_2[_lp*2+1 + 0]));
                    h_raw_3_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_2_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_2_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_3_2_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_3_2_bf16[_pair]));
                }
                float y_lo_3_2[2];
                float y_hi_3_2[2];
                float _exp2_56 = approx_exp2((-h_raw_3_2_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_56 = approx_rcp(1.0f + _exp2_56);
                float _exp2_57 = approx_exp2((-h_raw_3_2_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_57 = approx_rcp(1.0f + _exp2_57);
                __nv_bfloat16 _cvt_bf16_56 = __float2bfloat16(h_raw_3_2_bf16_f32[0] * _rcp_56);
                float _cvt_f32_56 = __bfloat162float(_cvt_bf16_56);
                __nv_bfloat16 _cvt_bf16_57 = __float2bfloat16(h_raw_3_2_bf16_f32[2] * _rcp_57);
                float _cvt_f32_57 = __bfloat162float(_cvt_bf16_57);
                y_lo_3_2[0] = _cvt_f32_56 * h_raw_3_2_bf16_f32[4];
                y_hi_3_2[0] = _cvt_f32_57 * h_raw_3_2_bf16_f32[6];
                float _exp2_58 = approx_exp2((-h_raw_3_2_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_58 = approx_rcp(1.0f + _exp2_58);
                float _exp2_59 = approx_exp2((-h_raw_3_2_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_59 = approx_rcp(1.0f + _exp2_59);
                __nv_bfloat16 _cvt_bf16_58 = __float2bfloat16(h_raw_3_2_bf16_f32[1] * _rcp_58);
                float _cvt_f32_58 = __bfloat162float(_cvt_bf16_58);
                __nv_bfloat16 _cvt_bf16_59 = __float2bfloat16(h_raw_3_2_bf16_f32[3] * _rcp_59);
                float _cvt_f32_59 = __bfloat162float(_cvt_bf16_59);
                y_lo_3_2[1] = _cvt_f32_58 * h_raw_3_2_bf16_f32[5];
                y_hi_3_2[1] = _cvt_f32_59 * h_raw_3_2_bf16_f32[7];
                uint32_t y_lo_3_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_2[_lp*2 + 0], y_lo_3_2[_lp*2+1 + 0]));
                    y_lo_3_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_2[_lp*2 + 0], y_hi_3_2[_lp*2+1 + 0]));
                    y_hi_3_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_3_2_bf16[0];
                }
                if (row_hi_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_3_2_bf16[0];
                }
                float h_raw_3_3[8];
                h_raw_3_3[0] = accum[120] * sa_lo_3 * sw[12];
                h_raw_3_3[2] = accum[122] * sa_hi_3 * sw[12];
                h_raw_3_3[4] = accum[124] * sa_lo_3 * sw[14];
                h_raw_3_3[6] = accum[126] * sa_hi_3 * sw[14];
                h_raw_3_3[1] = accum[121] * sa_lo_3 * sw[13];
                h_raw_3_3[3] = accum[123] * sa_hi_3 * sw[13];
                h_raw_3_3[5] = accum[125] * sa_lo_3 * sw[15];
                h_raw_3_3[7] = accum[127] * sa_hi_3 * sw[15];
                uint32_t h_raw_3_3_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_3[_lp*2 + 0], h_raw_3_3[_lp*2+1 + 0]));
                    h_raw_3_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_3_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_3_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_3_3_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_3_3_bf16[_pair]));
                }
                float y_lo_3_3[2];
                float y_hi_3_3[2];
                float _exp2_60 = approx_exp2((-h_raw_3_3_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_60 = approx_rcp(1.0f + _exp2_60);
                float _exp2_61 = approx_exp2((-h_raw_3_3_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_61 = approx_rcp(1.0f + _exp2_61);
                __nv_bfloat16 _cvt_bf16_60 = __float2bfloat16(h_raw_3_3_bf16_f32[0] * _rcp_60);
                float _cvt_f32_60 = __bfloat162float(_cvt_bf16_60);
                __nv_bfloat16 _cvt_bf16_61 = __float2bfloat16(h_raw_3_3_bf16_f32[2] * _rcp_61);
                float _cvt_f32_61 = __bfloat162float(_cvt_bf16_61);
                y_lo_3_3[0] = _cvt_f32_60 * h_raw_3_3_bf16_f32[4];
                y_hi_3_3[0] = _cvt_f32_61 * h_raw_3_3_bf16_f32[6];
                float _exp2_62 = approx_exp2((-h_raw_3_3_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_62 = approx_rcp(1.0f + _exp2_62);
                float _exp2_63 = approx_exp2((-h_raw_3_3_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_63 = approx_rcp(1.0f + _exp2_63);
                __nv_bfloat16 _cvt_bf16_62 = __float2bfloat16(h_raw_3_3_bf16_f32[1] * _rcp_62);
                float _cvt_f32_62 = __bfloat162float(_cvt_bf16_62);
                __nv_bfloat16 _cvt_bf16_63 = __float2bfloat16(h_raw_3_3_bf16_f32[3] * _rcp_63);
                float _cvt_f32_63 = __bfloat162float(_cvt_bf16_63);
                y_lo_3_3[1] = _cvt_f32_62 * h_raw_3_3_bf16_f32[5];
                y_hi_3_3[1] = _cvt_f32_63 * h_raw_3_3_bf16_f32[7];
                uint32_t y_lo_3_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_3[_lp*2 + 0], y_lo_3_3[_lp*2+1 + 0]));
                    y_lo_3_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_3[_lp*2 + 0], y_hi_3_3[_lp*2+1 + 0]));
                    y_hi_3_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_3_3_bf16[0];
                }
                if (row_hi_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_3_3_bf16[0];
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 1 && warp <= 7) {
        { // mma_main
            unsigned int mma_stage_1 = 0;
            unsigned int mma_kk_1 = 0;
            int warp_m_1 = warp / 4;
            int warp_n_1 = warp % 4;
            float accum_1[128];
            unsigned int a_frag_1[16];
            unsigned int b_frag_1[16];
            unsigned int _phase_ab_full_1 = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 112) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 112) * (GROUP_M * 112)) % ((num_m_tiles - tile_1 / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 112) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 112) * (GROUP_M * 112)) / ((num_m_tiles - tile_1 / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 112) * GROUP_M : GROUP_M);
                accum_1[0] = 0.0f;
                accum_1[1] = 0.0f;
                accum_1[2] = 0.0f;
                accum_1[3] = 0.0f;
                accum_1[4] = 0.0f;
                accum_1[5] = 0.0f;
                accum_1[6] = 0.0f;
                accum_1[7] = 0.0f;
                accum_1[8] = 0.0f;
                accum_1[9] = 0.0f;
                accum_1[10] = 0.0f;
                accum_1[11] = 0.0f;
                accum_1[12] = 0.0f;
                accum_1[13] = 0.0f;
                accum_1[14] = 0.0f;
                accum_1[15] = 0.0f;
                accum_1[16] = 0.0f;
                accum_1[17] = 0.0f;
                accum_1[18] = 0.0f;
                accum_1[19] = 0.0f;
                accum_1[20] = 0.0f;
                accum_1[21] = 0.0f;
                accum_1[22] = 0.0f;
                accum_1[23] = 0.0f;
                accum_1[24] = 0.0f;
                accum_1[25] = 0.0f;
                accum_1[26] = 0.0f;
                accum_1[27] = 0.0f;
                accum_1[28] = 0.0f;
                accum_1[29] = 0.0f;
                accum_1[30] = 0.0f;
                accum_1[31] = 0.0f;
                accum_1[32] = 0.0f;
                accum_1[33] = 0.0f;
                accum_1[34] = 0.0f;
                accum_1[35] = 0.0f;
                accum_1[36] = 0.0f;
                accum_1[37] = 0.0f;
                accum_1[38] = 0.0f;
                accum_1[39] = 0.0f;
                accum_1[40] = 0.0f;
                accum_1[41] = 0.0f;
                accum_1[42] = 0.0f;
                accum_1[43] = 0.0f;
                accum_1[44] = 0.0f;
                accum_1[45] = 0.0f;
                accum_1[46] = 0.0f;
                accum_1[47] = 0.0f;
                accum_1[48] = 0.0f;
                accum_1[49] = 0.0f;
                accum_1[50] = 0.0f;
                accum_1[51] = 0.0f;
                accum_1[52] = 0.0f;
                accum_1[53] = 0.0f;
                accum_1[54] = 0.0f;
                accum_1[55] = 0.0f;
                accum_1[56] = 0.0f;
                accum_1[57] = 0.0f;
                accum_1[58] = 0.0f;
                accum_1[59] = 0.0f;
                accum_1[60] = 0.0f;
                accum_1[61] = 0.0f;
                accum_1[62] = 0.0f;
                accum_1[63] = 0.0f;
                accum_1[64] = 0.0f;
                accum_1[65] = 0.0f;
                accum_1[66] = 0.0f;
                accum_1[67] = 0.0f;
                accum_1[68] = 0.0f;
                accum_1[69] = 0.0f;
                accum_1[70] = 0.0f;
                accum_1[71] = 0.0f;
                accum_1[72] = 0.0f;
                accum_1[73] = 0.0f;
                accum_1[74] = 0.0f;
                accum_1[75] = 0.0f;
                accum_1[76] = 0.0f;
                accum_1[77] = 0.0f;
                accum_1[78] = 0.0f;
                accum_1[79] = 0.0f;
                accum_1[80] = 0.0f;
                accum_1[81] = 0.0f;
                accum_1[82] = 0.0f;
                accum_1[83] = 0.0f;
                accum_1[84] = 0.0f;
                accum_1[85] = 0.0f;
                accum_1[86] = 0.0f;
                accum_1[87] = 0.0f;
                accum_1[88] = 0.0f;
                accum_1[89] = 0.0f;
                accum_1[90] = 0.0f;
                accum_1[91] = 0.0f;
                accum_1[92] = 0.0f;
                accum_1[93] = 0.0f;
                accum_1[94] = 0.0f;
                accum_1[95] = 0.0f;
                accum_1[96] = 0.0f;
                accum_1[97] = 0.0f;
                accum_1[98] = 0.0f;
                accum_1[99] = 0.0f;
                accum_1[100] = 0.0f;
                accum_1[101] = 0.0f;
                accum_1[102] = 0.0f;
                accum_1[103] = 0.0f;
                accum_1[104] = 0.0f;
                accum_1[105] = 0.0f;
                accum_1[106] = 0.0f;
                accum_1[107] = 0.0f;
                accum_1[108] = 0.0f;
                accum_1[109] = 0.0f;
                accum_1[110] = 0.0f;
                accum_1[111] = 0.0f;
                accum_1[112] = 0.0f;
                accum_1[113] = 0.0f;
                accum_1[114] = 0.0f;
                accum_1[115] = 0.0f;
                accum_1[116] = 0.0f;
                accum_1[117] = 0.0f;
                accum_1[118] = 0.0f;
                accum_1[119] = 0.0f;
                accum_1[120] = 0.0f;
                accum_1[121] = 0.0f;
                accum_1[122] = 0.0f;
                accum_1[123] = 0.0f;
                accum_1[124] = 0.0f;
                accum_1[125] = 0.0f;
                accum_1[126] = 0.0f;
                accum_1[127] = 0.0f;
                #pragma unroll 1
                for (int k_tile_1 = 0; k_tile_1 < 84; k_tile_1++) {
                    mbarrier_wait(ab_full_addr + (mma_stage_1) * 8, _phase_ab_full_1);
                    for (int k_step_1 = 0; k_step_1 < 2; k_step_1++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(A_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_m_1 * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[4]), "=r"(a_frag_1[5]), "=r"(a_frag_1[6]), "=r"(a_frag_1[7])
                            : "r"(A_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_m_1 * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[8]), "=r"(a_frag_1[9]), "=r"(a_frag_1[10]), "=r"(a_frag_1[11])
                            : "r"(A_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_m_1 * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[12]), "=r"(a_frag_1[13]), "=r"(a_frag_1[14]), "=r"(a_frag_1[15])
                            : "r"(A_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_m_1 * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1]), "=r"(b_frag_1[2]), "=r"(b_frag_1[3])
                            : "r"(B_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_n_1 * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[4]), "=r"(b_frag_1[5]), "=r"(b_frag_1[6]), "=r"(b_frag_1[7])
                            : "r"(B_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_n_1 * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[8]), "=r"(b_frag_1[9]), "=r"(b_frag_1[10]), "=r"(b_frag_1[11])
                            : "r"(B_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_n_1 * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[12]), "=r"(b_frag_1[13]), "=r"(b_frag_1[14]), "=r"(b_frag_1[15])
                            : "r"(B_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_n_1 * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[0]), "+f"(accum_1[1]), "+f"(accum_1[2]), "+f"(accum_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[4]), "+f"(accum_1[(4) + 1]), "+f"(accum_1[(4) + 2]), "+f"(accum_1[(4) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[8]), "+f"(accum_1[(8) + 1]), "+f"(accum_1[(8) + 2]), "+f"(accum_1[(8) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[12]), "+f"(accum_1[(12) + 1]), "+f"(accum_1[(12) + 2]), "+f"(accum_1[(12) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[16]), "+f"(accum_1[(16) + 1]), "+f"(accum_1[(16) + 2]), "+f"(accum_1[(16) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[20]), "+f"(accum_1[(20) + 1]), "+f"(accum_1[(20) + 2]), "+f"(accum_1[(20) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[24]), "+f"(accum_1[(24) + 1]), "+f"(accum_1[(24) + 2]), "+f"(accum_1[(24) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[28]), "+f"(accum_1[(28) + 1]), "+f"(accum_1[(28) + 2]), "+f"(accum_1[(28) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[32]), "+f"(accum_1[(32) + 1]), "+f"(accum_1[(32) + 2]), "+f"(accum_1[(32) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[36]), "+f"(accum_1[(36) + 1]), "+f"(accum_1[(36) + 2]), "+f"(accum_1[(36) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[40]), "+f"(accum_1[(40) + 1]), "+f"(accum_1[(40) + 2]), "+f"(accum_1[(40) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[44]), "+f"(accum_1[(44) + 1]), "+f"(accum_1[(44) + 2]), "+f"(accum_1[(44) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[48]), "+f"(accum_1[(48) + 1]), "+f"(accum_1[(48) + 2]), "+f"(accum_1[(48) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[52]), "+f"(accum_1[(52) + 1]), "+f"(accum_1[(52) + 2]), "+f"(accum_1[(52) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[56]), "+f"(accum_1[(56) + 1]), "+f"(accum_1[(56) + 2]), "+f"(accum_1[(56) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[60]), "+f"(accum_1[(60) + 1]), "+f"(accum_1[(60) + 2]), "+f"(accum_1[(60) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[64]), "+f"(accum_1[(64) + 1]), "+f"(accum_1[(64) + 2]), "+f"(accum_1[(64) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[68]), "+f"(accum_1[(68) + 1]), "+f"(accum_1[(68) + 2]), "+f"(accum_1[(68) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[72]), "+f"(accum_1[(72) + 1]), "+f"(accum_1[(72) + 2]), "+f"(accum_1[(72) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[76]), "+f"(accum_1[(76) + 1]), "+f"(accum_1[(76) + 2]), "+f"(accum_1[(76) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[80]), "+f"(accum_1[(80) + 1]), "+f"(accum_1[(80) + 2]), "+f"(accum_1[(80) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[84]), "+f"(accum_1[(84) + 1]), "+f"(accum_1[(84) + 2]), "+f"(accum_1[(84) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[88]), "+f"(accum_1[(88) + 1]), "+f"(accum_1[(88) + 2]), "+f"(accum_1[(88) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[92]), "+f"(accum_1[(92) + 1]), "+f"(accum_1[(92) + 2]), "+f"(accum_1[(92) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[96]), "+f"(accum_1[(96) + 1]), "+f"(accum_1[(96) + 2]), "+f"(accum_1[(96) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[100]), "+f"(accum_1[(100) + 1]), "+f"(accum_1[(100) + 2]), "+f"(accum_1[(100) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[104]), "+f"(accum_1[(104) + 1]), "+f"(accum_1[(104) + 2]), "+f"(accum_1[(104) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[108]), "+f"(accum_1[(108) + 1]), "+f"(accum_1[(108) + 2]), "+f"(accum_1[(108) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[112]), "+f"(accum_1[(112) + 1]), "+f"(accum_1[(112) + 2]), "+f"(accum_1[(112) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[116]), "+f"(accum_1[(116) + 1]), "+f"(accum_1[(116) + 2]), "+f"(accum_1[(116) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[120]), "+f"(accum_1[(120) + 1]), "+f"(accum_1[(120) + 2]), "+f"(accum_1[(120) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum_1[124]), "+f"(accum_1[(124) + 1]), "+f"(accum_1[(124) + 2]), "+f"(accum_1[(124) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]));
                    }
                    __syncwarp();
                    if (lane == 0) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage_1) * 8);
                    }
                    mma_stage_1 += 1;
                    if (mma_stage_1 == 4) { mma_stage_1 = 0; _phase_ab_full_1 ^= 1; }
                    mma_kk_1 = mma_kk_1 + 1;
                }
                float sw_1[16];
                sw_1[0] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2];
                sw_1[2] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 8];
                sw_1[1] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 1];
                sw_1[3] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 1 + 8];
                sw_1[4] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 16];
                sw_1[6] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 16 + 8];
                sw_1[5] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 16 + 1];
                sw_1[7] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 16 + 1 + 8];
                sw_1[8] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 32];
                sw_1[10] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 32 + 8];
                sw_1[9] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 32 + 1];
                sw_1[11] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 32 + 1 + 8];
                sw_1[12] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 48];
                sw_1[14] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 48 + 8];
                sw_1[13] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 48 + 1];
                sw_1[15] = w_scale[tile_n_1 * 256 + warp_n_1 * 64 + (lane & 3) * 2 + 48 + 1 + 8];
                int row_lo_0_1 = tile_m_1 * 128 + warp_m_1 * 64 + (lane >> 2);
                int row_hi_0_1 = row_lo_0_1 + 8;
                float sa_lo_0_1 = ((row_lo_0_1 < M) ? act_scale[row_lo_0_1] : 0.0f);
                float sa_hi_0_1 = ((row_hi_0_1 < M) ? act_scale[row_hi_0_1] : 0.0f);
                float h_raw_0_0_1[8];
                h_raw_0_0_1[0] = accum_1[0] * sa_lo_0_1 * sw_1[0];
                h_raw_0_0_1[2] = accum_1[2] * sa_hi_0_1 * sw_1[0];
                h_raw_0_0_1[4] = accum_1[4] * sa_lo_0_1 * sw_1[2];
                h_raw_0_0_1[6] = accum_1[6] * sa_hi_0_1 * sw_1[2];
                h_raw_0_0_1[1] = accum_1[1] * sa_lo_0_1 * sw_1[1];
                h_raw_0_0_1[3] = accum_1[3] * sa_hi_0_1 * sw_1[1];
                h_raw_0_0_1[5] = accum_1[5] * sa_lo_0_1 * sw_1[3];
                h_raw_0_0_1[7] = accum_1[7] * sa_hi_0_1 * sw_1[3];
                uint32_t h_raw_0_0_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_0_1[_lp*2 + 0], h_raw_0_0_1[_lp*2+1 + 0]));
                    h_raw_0_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_0_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_0_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_0_0_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_0_0_bf16_1[_pair]));
                }
                float y_lo_0_0_1[2];
                float y_hi_0_0_1[2];
                float _exp2_64 = approx_exp2((-h_raw_0_0_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_64 = approx_rcp(1.0f + _exp2_64);
                float _exp2_65 = approx_exp2((-h_raw_0_0_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_65 = approx_rcp(1.0f + _exp2_65);
                __nv_bfloat16 _cvt_bf16_64 = __float2bfloat16(h_raw_0_0_bf16_f32_1[0] * _rcp_64);
                float _cvt_f32_64 = __bfloat162float(_cvt_bf16_64);
                __nv_bfloat16 _cvt_bf16_65 = __float2bfloat16(h_raw_0_0_bf16_f32_1[2] * _rcp_65);
                float _cvt_f32_65 = __bfloat162float(_cvt_bf16_65);
                y_lo_0_0_1[0] = _cvt_f32_64 * h_raw_0_0_bf16_f32_1[4];
                y_hi_0_0_1[0] = _cvt_f32_65 * h_raw_0_0_bf16_f32_1[6];
                float _exp2_66 = approx_exp2((-h_raw_0_0_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_66 = approx_rcp(1.0f + _exp2_66);
                float _exp2_67 = approx_exp2((-h_raw_0_0_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_67 = approx_rcp(1.0f + _exp2_67);
                __nv_bfloat16 _cvt_bf16_66 = __float2bfloat16(h_raw_0_0_bf16_f32_1[1] * _rcp_66);
                float _cvt_f32_66 = __bfloat162float(_cvt_bf16_66);
                __nv_bfloat16 _cvt_bf16_67 = __float2bfloat16(h_raw_0_0_bf16_f32_1[3] * _rcp_67);
                float _cvt_f32_67 = __bfloat162float(_cvt_bf16_67);
                y_lo_0_0_1[1] = _cvt_f32_66 * h_raw_0_0_bf16_f32_1[5];
                y_hi_0_0_1[1] = _cvt_f32_67 * h_raw_0_0_bf16_f32_1[7];
                uint32_t y_lo_0_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_0_1[_lp*2 + 0], y_lo_0_0_1[_lp*2+1 + 0]));
                    y_lo_0_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_0_1[_lp*2 + 0], y_hi_0_0_1[_lp*2+1 + 0]));
                    y_hi_0_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_0_0_bf16_1[0];
                }
                if (row_hi_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_0_0_bf16_1[0];
                }
                float h_raw_0_1_1[8];
                h_raw_0_1_1[0] = accum_1[8] * sa_lo_0_1 * sw_1[4];
                h_raw_0_1_1[2] = accum_1[10] * sa_hi_0_1 * sw_1[4];
                h_raw_0_1_1[4] = accum_1[12] * sa_lo_0_1 * sw_1[6];
                h_raw_0_1_1[6] = accum_1[14] * sa_hi_0_1 * sw_1[6];
                h_raw_0_1_1[1] = accum_1[9] * sa_lo_0_1 * sw_1[5];
                h_raw_0_1_1[3] = accum_1[11] * sa_hi_0_1 * sw_1[5];
                h_raw_0_1_1[5] = accum_1[13] * sa_lo_0_1 * sw_1[7];
                h_raw_0_1_1[7] = accum_1[15] * sa_hi_0_1 * sw_1[7];
                uint32_t h_raw_0_1_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_1_1[_lp*2 + 0], h_raw_0_1_1[_lp*2+1 + 0]));
                    h_raw_0_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_1_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_1_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_0_1_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_0_1_bf16_1[_pair]));
                }
                float y_lo_0_1_1[2];
                float y_hi_0_1_1[2];
                float _exp2_68 = approx_exp2((-h_raw_0_1_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_68 = approx_rcp(1.0f + _exp2_68);
                float _exp2_69 = approx_exp2((-h_raw_0_1_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_69 = approx_rcp(1.0f + _exp2_69);
                __nv_bfloat16 _cvt_bf16_68 = __float2bfloat16(h_raw_0_1_bf16_f32_1[0] * _rcp_68);
                float _cvt_f32_68 = __bfloat162float(_cvt_bf16_68);
                __nv_bfloat16 _cvt_bf16_69 = __float2bfloat16(h_raw_0_1_bf16_f32_1[2] * _rcp_69);
                float _cvt_f32_69 = __bfloat162float(_cvt_bf16_69);
                y_lo_0_1_1[0] = _cvt_f32_68 * h_raw_0_1_bf16_f32_1[4];
                y_hi_0_1_1[0] = _cvt_f32_69 * h_raw_0_1_bf16_f32_1[6];
                float _exp2_70 = approx_exp2((-h_raw_0_1_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_70 = approx_rcp(1.0f + _exp2_70);
                float _exp2_71 = approx_exp2((-h_raw_0_1_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_71 = approx_rcp(1.0f + _exp2_71);
                __nv_bfloat16 _cvt_bf16_70 = __float2bfloat16(h_raw_0_1_bf16_f32_1[1] * _rcp_70);
                float _cvt_f32_70 = __bfloat162float(_cvt_bf16_70);
                __nv_bfloat16 _cvt_bf16_71 = __float2bfloat16(h_raw_0_1_bf16_f32_1[3] * _rcp_71);
                float _cvt_f32_71 = __bfloat162float(_cvt_bf16_71);
                y_lo_0_1_1[1] = _cvt_f32_70 * h_raw_0_1_bf16_f32_1[5];
                y_hi_0_1_1[1] = _cvt_f32_71 * h_raw_0_1_bf16_f32_1[7];
                uint32_t y_lo_0_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_1_1[_lp*2 + 0], y_lo_0_1_1[_lp*2+1 + 0]));
                    y_lo_0_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_1_1[_lp*2 + 0], y_hi_0_1_1[_lp*2+1 + 0]));
                    y_hi_0_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_0_1_bf16_1[0];
                }
                if (row_hi_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_0_1_bf16_1[0];
                }
                float h_raw_0_2_1[8];
                h_raw_0_2_1[0] = accum_1[16] * sa_lo_0_1 * sw_1[8];
                h_raw_0_2_1[2] = accum_1[18] * sa_hi_0_1 * sw_1[8];
                h_raw_0_2_1[4] = accum_1[20] * sa_lo_0_1 * sw_1[10];
                h_raw_0_2_1[6] = accum_1[22] * sa_hi_0_1 * sw_1[10];
                h_raw_0_2_1[1] = accum_1[17] * sa_lo_0_1 * sw_1[9];
                h_raw_0_2_1[3] = accum_1[19] * sa_hi_0_1 * sw_1[9];
                h_raw_0_2_1[5] = accum_1[21] * sa_lo_0_1 * sw_1[11];
                h_raw_0_2_1[7] = accum_1[23] * sa_hi_0_1 * sw_1[11];
                uint32_t h_raw_0_2_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_2_1[_lp*2 + 0], h_raw_0_2_1[_lp*2+1 + 0]));
                    h_raw_0_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_2_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_2_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_0_2_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_0_2_bf16_1[_pair]));
                }
                float y_lo_0_2_1[2];
                float y_hi_0_2_1[2];
                float _exp2_72 = approx_exp2((-h_raw_0_2_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_72 = approx_rcp(1.0f + _exp2_72);
                float _exp2_73 = approx_exp2((-h_raw_0_2_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_73 = approx_rcp(1.0f + _exp2_73);
                __nv_bfloat16 _cvt_bf16_72 = __float2bfloat16(h_raw_0_2_bf16_f32_1[0] * _rcp_72);
                float _cvt_f32_72 = __bfloat162float(_cvt_bf16_72);
                __nv_bfloat16 _cvt_bf16_73 = __float2bfloat16(h_raw_0_2_bf16_f32_1[2] * _rcp_73);
                float _cvt_f32_73 = __bfloat162float(_cvt_bf16_73);
                y_lo_0_2_1[0] = _cvt_f32_72 * h_raw_0_2_bf16_f32_1[4];
                y_hi_0_2_1[0] = _cvt_f32_73 * h_raw_0_2_bf16_f32_1[6];
                float _exp2_74 = approx_exp2((-h_raw_0_2_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_74 = approx_rcp(1.0f + _exp2_74);
                float _exp2_75 = approx_exp2((-h_raw_0_2_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_75 = approx_rcp(1.0f + _exp2_75);
                __nv_bfloat16 _cvt_bf16_74 = __float2bfloat16(h_raw_0_2_bf16_f32_1[1] * _rcp_74);
                float _cvt_f32_74 = __bfloat162float(_cvt_bf16_74);
                __nv_bfloat16 _cvt_bf16_75 = __float2bfloat16(h_raw_0_2_bf16_f32_1[3] * _rcp_75);
                float _cvt_f32_75 = __bfloat162float(_cvt_bf16_75);
                y_lo_0_2_1[1] = _cvt_f32_74 * h_raw_0_2_bf16_f32_1[5];
                y_hi_0_2_1[1] = _cvt_f32_75 * h_raw_0_2_bf16_f32_1[7];
                uint32_t y_lo_0_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_2_1[_lp*2 + 0], y_lo_0_2_1[_lp*2+1 + 0]));
                    y_lo_0_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_2_1[_lp*2 + 0], y_hi_0_2_1[_lp*2+1 + 0]));
                    y_hi_0_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_0_2_bf16_1[0];
                }
                if (row_hi_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_0_2_bf16_1[0];
                }
                float h_raw_0_3_1[8];
                h_raw_0_3_1[0] = accum_1[24] * sa_lo_0_1 * sw_1[12];
                h_raw_0_3_1[2] = accum_1[26] * sa_hi_0_1 * sw_1[12];
                h_raw_0_3_1[4] = accum_1[28] * sa_lo_0_1 * sw_1[14];
                h_raw_0_3_1[6] = accum_1[30] * sa_hi_0_1 * sw_1[14];
                h_raw_0_3_1[1] = accum_1[25] * sa_lo_0_1 * sw_1[13];
                h_raw_0_3_1[3] = accum_1[27] * sa_hi_0_1 * sw_1[13];
                h_raw_0_3_1[5] = accum_1[29] * sa_lo_0_1 * sw_1[15];
                h_raw_0_3_1[7] = accum_1[31] * sa_hi_0_1 * sw_1[15];
                uint32_t h_raw_0_3_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_3_1[_lp*2 + 0], h_raw_0_3_1[_lp*2+1 + 0]));
                    h_raw_0_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_3_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_3_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_0_3_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_0_3_bf16_1[_pair]));
                }
                float y_lo_0_3_1[2];
                float y_hi_0_3_1[2];
                float _exp2_76 = approx_exp2((-h_raw_0_3_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_76 = approx_rcp(1.0f + _exp2_76);
                float _exp2_77 = approx_exp2((-h_raw_0_3_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_77 = approx_rcp(1.0f + _exp2_77);
                __nv_bfloat16 _cvt_bf16_76 = __float2bfloat16(h_raw_0_3_bf16_f32_1[0] * _rcp_76);
                float _cvt_f32_76 = __bfloat162float(_cvt_bf16_76);
                __nv_bfloat16 _cvt_bf16_77 = __float2bfloat16(h_raw_0_3_bf16_f32_1[2] * _rcp_77);
                float _cvt_f32_77 = __bfloat162float(_cvt_bf16_77);
                y_lo_0_3_1[0] = _cvt_f32_76 * h_raw_0_3_bf16_f32_1[4];
                y_hi_0_3_1[0] = _cvt_f32_77 * h_raw_0_3_bf16_f32_1[6];
                float _exp2_78 = approx_exp2((-h_raw_0_3_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_78 = approx_rcp(1.0f + _exp2_78);
                float _exp2_79 = approx_exp2((-h_raw_0_3_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_79 = approx_rcp(1.0f + _exp2_79);
                __nv_bfloat16 _cvt_bf16_78 = __float2bfloat16(h_raw_0_3_bf16_f32_1[1] * _rcp_78);
                float _cvt_f32_78 = __bfloat162float(_cvt_bf16_78);
                __nv_bfloat16 _cvt_bf16_79 = __float2bfloat16(h_raw_0_3_bf16_f32_1[3] * _rcp_79);
                float _cvt_f32_79 = __bfloat162float(_cvt_bf16_79);
                y_lo_0_3_1[1] = _cvt_f32_78 * h_raw_0_3_bf16_f32_1[5];
                y_hi_0_3_1[1] = _cvt_f32_79 * h_raw_0_3_bf16_f32_1[7];
                uint32_t y_lo_0_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_3_1[_lp*2 + 0], y_lo_0_3_1[_lp*2+1 + 0]));
                    y_lo_0_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_3_1[_lp*2 + 0], y_hi_0_3_1[_lp*2+1 + 0]));
                    y_hi_0_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_0_3_bf16_1[0];
                }
                if (row_hi_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_0_3_bf16_1[0];
                }
                int row_lo_1_1 = tile_m_1 * 128 + warp_m_1 * 64 + 16 + (lane >> 2);
                int row_hi_1_1 = row_lo_1_1 + 8;
                float sa_lo_1_1 = ((row_lo_1_1 < M) ? act_scale[row_lo_1_1] : 0.0f);
                float sa_hi_1_1 = ((row_hi_1_1 < M) ? act_scale[row_hi_1_1] : 0.0f);
                float h_raw_1_0_1[8];
                h_raw_1_0_1[0] = accum_1[32] * sa_lo_1_1 * sw_1[0];
                h_raw_1_0_1[2] = accum_1[34] * sa_hi_1_1 * sw_1[0];
                h_raw_1_0_1[4] = accum_1[36] * sa_lo_1_1 * sw_1[2];
                h_raw_1_0_1[6] = accum_1[38] * sa_hi_1_1 * sw_1[2];
                h_raw_1_0_1[1] = accum_1[33] * sa_lo_1_1 * sw_1[1];
                h_raw_1_0_1[3] = accum_1[35] * sa_hi_1_1 * sw_1[1];
                h_raw_1_0_1[5] = accum_1[37] * sa_lo_1_1 * sw_1[3];
                h_raw_1_0_1[7] = accum_1[39] * sa_hi_1_1 * sw_1[3];
                uint32_t h_raw_1_0_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_0_1[_lp*2 + 0], h_raw_1_0_1[_lp*2+1 + 0]));
                    h_raw_1_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_0_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_0_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_1_0_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_1_0_bf16_1[_pair]));
                }
                float y_lo_1_0_1[2];
                float y_hi_1_0_1[2];
                float _exp2_80 = approx_exp2((-h_raw_1_0_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_80 = approx_rcp(1.0f + _exp2_80);
                float _exp2_81 = approx_exp2((-h_raw_1_0_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_81 = approx_rcp(1.0f + _exp2_81);
                __nv_bfloat16 _cvt_bf16_80 = __float2bfloat16(h_raw_1_0_bf16_f32_1[0] * _rcp_80);
                float _cvt_f32_80 = __bfloat162float(_cvt_bf16_80);
                __nv_bfloat16 _cvt_bf16_81 = __float2bfloat16(h_raw_1_0_bf16_f32_1[2] * _rcp_81);
                float _cvt_f32_81 = __bfloat162float(_cvt_bf16_81);
                y_lo_1_0_1[0] = _cvt_f32_80 * h_raw_1_0_bf16_f32_1[4];
                y_hi_1_0_1[0] = _cvt_f32_81 * h_raw_1_0_bf16_f32_1[6];
                float _exp2_82 = approx_exp2((-h_raw_1_0_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_82 = approx_rcp(1.0f + _exp2_82);
                float _exp2_83 = approx_exp2((-h_raw_1_0_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_83 = approx_rcp(1.0f + _exp2_83);
                __nv_bfloat16 _cvt_bf16_82 = __float2bfloat16(h_raw_1_0_bf16_f32_1[1] * _rcp_82);
                float _cvt_f32_82 = __bfloat162float(_cvt_bf16_82);
                __nv_bfloat16 _cvt_bf16_83 = __float2bfloat16(h_raw_1_0_bf16_f32_1[3] * _rcp_83);
                float _cvt_f32_83 = __bfloat162float(_cvt_bf16_83);
                y_lo_1_0_1[1] = _cvt_f32_82 * h_raw_1_0_bf16_f32_1[5];
                y_hi_1_0_1[1] = _cvt_f32_83 * h_raw_1_0_bf16_f32_1[7];
                uint32_t y_lo_1_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_0_1[_lp*2 + 0], y_lo_1_0_1[_lp*2+1 + 0]));
                    y_lo_1_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_0_1[_lp*2 + 0], y_hi_1_0_1[_lp*2+1 + 0]));
                    y_hi_1_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_1_0_bf16_1[0];
                }
                if (row_hi_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_1_0_bf16_1[0];
                }
                float h_raw_1_1_1[8];
                h_raw_1_1_1[0] = accum_1[40] * sa_lo_1_1 * sw_1[4];
                h_raw_1_1_1[2] = accum_1[42] * sa_hi_1_1 * sw_1[4];
                h_raw_1_1_1[4] = accum_1[44] * sa_lo_1_1 * sw_1[6];
                h_raw_1_1_1[6] = accum_1[46] * sa_hi_1_1 * sw_1[6];
                h_raw_1_1_1[1] = accum_1[41] * sa_lo_1_1 * sw_1[5];
                h_raw_1_1_1[3] = accum_1[43] * sa_hi_1_1 * sw_1[5];
                h_raw_1_1_1[5] = accum_1[45] * sa_lo_1_1 * sw_1[7];
                h_raw_1_1_1[7] = accum_1[47] * sa_hi_1_1 * sw_1[7];
                uint32_t h_raw_1_1_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_1_1[_lp*2 + 0], h_raw_1_1_1[_lp*2+1 + 0]));
                    h_raw_1_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_1_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_1_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_1_1_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_1_1_bf16_1[_pair]));
                }
                float y_lo_1_1_1[2];
                float y_hi_1_1_1[2];
                float _exp2_84 = approx_exp2((-h_raw_1_1_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_84 = approx_rcp(1.0f + _exp2_84);
                float _exp2_85 = approx_exp2((-h_raw_1_1_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_85 = approx_rcp(1.0f + _exp2_85);
                __nv_bfloat16 _cvt_bf16_84 = __float2bfloat16(h_raw_1_1_bf16_f32_1[0] * _rcp_84);
                float _cvt_f32_84 = __bfloat162float(_cvt_bf16_84);
                __nv_bfloat16 _cvt_bf16_85 = __float2bfloat16(h_raw_1_1_bf16_f32_1[2] * _rcp_85);
                float _cvt_f32_85 = __bfloat162float(_cvt_bf16_85);
                y_lo_1_1_1[0] = _cvt_f32_84 * h_raw_1_1_bf16_f32_1[4];
                y_hi_1_1_1[0] = _cvt_f32_85 * h_raw_1_1_bf16_f32_1[6];
                float _exp2_86 = approx_exp2((-h_raw_1_1_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_86 = approx_rcp(1.0f + _exp2_86);
                float _exp2_87 = approx_exp2((-h_raw_1_1_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_87 = approx_rcp(1.0f + _exp2_87);
                __nv_bfloat16 _cvt_bf16_86 = __float2bfloat16(h_raw_1_1_bf16_f32_1[1] * _rcp_86);
                float _cvt_f32_86 = __bfloat162float(_cvt_bf16_86);
                __nv_bfloat16 _cvt_bf16_87 = __float2bfloat16(h_raw_1_1_bf16_f32_1[3] * _rcp_87);
                float _cvt_f32_87 = __bfloat162float(_cvt_bf16_87);
                y_lo_1_1_1[1] = _cvt_f32_86 * h_raw_1_1_bf16_f32_1[5];
                y_hi_1_1_1[1] = _cvt_f32_87 * h_raw_1_1_bf16_f32_1[7];
                uint32_t y_lo_1_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_1_1[_lp*2 + 0], y_lo_1_1_1[_lp*2+1 + 0]));
                    y_lo_1_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_1_1[_lp*2 + 0], y_hi_1_1_1[_lp*2+1 + 0]));
                    y_hi_1_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_1_1_bf16_1[0];
                }
                if (row_hi_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_1_1_bf16_1[0];
                }
                float h_raw_1_2_1[8];
                h_raw_1_2_1[0] = accum_1[48] * sa_lo_1_1 * sw_1[8];
                h_raw_1_2_1[2] = accum_1[50] * sa_hi_1_1 * sw_1[8];
                h_raw_1_2_1[4] = accum_1[52] * sa_lo_1_1 * sw_1[10];
                h_raw_1_2_1[6] = accum_1[54] * sa_hi_1_1 * sw_1[10];
                h_raw_1_2_1[1] = accum_1[49] * sa_lo_1_1 * sw_1[9];
                h_raw_1_2_1[3] = accum_1[51] * sa_hi_1_1 * sw_1[9];
                h_raw_1_2_1[5] = accum_1[53] * sa_lo_1_1 * sw_1[11];
                h_raw_1_2_1[7] = accum_1[55] * sa_hi_1_1 * sw_1[11];
                uint32_t h_raw_1_2_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_2_1[_lp*2 + 0], h_raw_1_2_1[_lp*2+1 + 0]));
                    h_raw_1_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_2_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_2_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_1_2_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_1_2_bf16_1[_pair]));
                }
                float y_lo_1_2_1[2];
                float y_hi_1_2_1[2];
                float _exp2_88 = approx_exp2((-h_raw_1_2_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_88 = approx_rcp(1.0f + _exp2_88);
                float _exp2_89 = approx_exp2((-h_raw_1_2_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_89 = approx_rcp(1.0f + _exp2_89);
                __nv_bfloat16 _cvt_bf16_88 = __float2bfloat16(h_raw_1_2_bf16_f32_1[0] * _rcp_88);
                float _cvt_f32_88 = __bfloat162float(_cvt_bf16_88);
                __nv_bfloat16 _cvt_bf16_89 = __float2bfloat16(h_raw_1_2_bf16_f32_1[2] * _rcp_89);
                float _cvt_f32_89 = __bfloat162float(_cvt_bf16_89);
                y_lo_1_2_1[0] = _cvt_f32_88 * h_raw_1_2_bf16_f32_1[4];
                y_hi_1_2_1[0] = _cvt_f32_89 * h_raw_1_2_bf16_f32_1[6];
                float _exp2_90 = approx_exp2((-h_raw_1_2_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_90 = approx_rcp(1.0f + _exp2_90);
                float _exp2_91 = approx_exp2((-h_raw_1_2_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_91 = approx_rcp(1.0f + _exp2_91);
                __nv_bfloat16 _cvt_bf16_90 = __float2bfloat16(h_raw_1_2_bf16_f32_1[1] * _rcp_90);
                float _cvt_f32_90 = __bfloat162float(_cvt_bf16_90);
                __nv_bfloat16 _cvt_bf16_91 = __float2bfloat16(h_raw_1_2_bf16_f32_1[3] * _rcp_91);
                float _cvt_f32_91 = __bfloat162float(_cvt_bf16_91);
                y_lo_1_2_1[1] = _cvt_f32_90 * h_raw_1_2_bf16_f32_1[5];
                y_hi_1_2_1[1] = _cvt_f32_91 * h_raw_1_2_bf16_f32_1[7];
                uint32_t y_lo_1_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_2_1[_lp*2 + 0], y_lo_1_2_1[_lp*2+1 + 0]));
                    y_lo_1_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_2_1[_lp*2 + 0], y_hi_1_2_1[_lp*2+1 + 0]));
                    y_hi_1_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_1_2_bf16_1[0];
                }
                if (row_hi_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_1_2_bf16_1[0];
                }
                float h_raw_1_3_1[8];
                h_raw_1_3_1[0] = accum_1[56] * sa_lo_1_1 * sw_1[12];
                h_raw_1_3_1[2] = accum_1[58] * sa_hi_1_1 * sw_1[12];
                h_raw_1_3_1[4] = accum_1[60] * sa_lo_1_1 * sw_1[14];
                h_raw_1_3_1[6] = accum_1[62] * sa_hi_1_1 * sw_1[14];
                h_raw_1_3_1[1] = accum_1[57] * sa_lo_1_1 * sw_1[13];
                h_raw_1_3_1[3] = accum_1[59] * sa_hi_1_1 * sw_1[13];
                h_raw_1_3_1[5] = accum_1[61] * sa_lo_1_1 * sw_1[15];
                h_raw_1_3_1[7] = accum_1[63] * sa_hi_1_1 * sw_1[15];
                uint32_t h_raw_1_3_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_3_1[_lp*2 + 0], h_raw_1_3_1[_lp*2+1 + 0]));
                    h_raw_1_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_3_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_3_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_1_3_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_1_3_bf16_1[_pair]));
                }
                float y_lo_1_3_1[2];
                float y_hi_1_3_1[2];
                float _exp2_92 = approx_exp2((-h_raw_1_3_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_92 = approx_rcp(1.0f + _exp2_92);
                float _exp2_93 = approx_exp2((-h_raw_1_3_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_93 = approx_rcp(1.0f + _exp2_93);
                __nv_bfloat16 _cvt_bf16_92 = __float2bfloat16(h_raw_1_3_bf16_f32_1[0] * _rcp_92);
                float _cvt_f32_92 = __bfloat162float(_cvt_bf16_92);
                __nv_bfloat16 _cvt_bf16_93 = __float2bfloat16(h_raw_1_3_bf16_f32_1[2] * _rcp_93);
                float _cvt_f32_93 = __bfloat162float(_cvt_bf16_93);
                y_lo_1_3_1[0] = _cvt_f32_92 * h_raw_1_3_bf16_f32_1[4];
                y_hi_1_3_1[0] = _cvt_f32_93 * h_raw_1_3_bf16_f32_1[6];
                float _exp2_94 = approx_exp2((-h_raw_1_3_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_94 = approx_rcp(1.0f + _exp2_94);
                float _exp2_95 = approx_exp2((-h_raw_1_3_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_95 = approx_rcp(1.0f + _exp2_95);
                __nv_bfloat16 _cvt_bf16_94 = __float2bfloat16(h_raw_1_3_bf16_f32_1[1] * _rcp_94);
                float _cvt_f32_94 = __bfloat162float(_cvt_bf16_94);
                __nv_bfloat16 _cvt_bf16_95 = __float2bfloat16(h_raw_1_3_bf16_f32_1[3] * _rcp_95);
                float _cvt_f32_95 = __bfloat162float(_cvt_bf16_95);
                y_lo_1_3_1[1] = _cvt_f32_94 * h_raw_1_3_bf16_f32_1[5];
                y_hi_1_3_1[1] = _cvt_f32_95 * h_raw_1_3_bf16_f32_1[7];
                uint32_t y_lo_1_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_3_1[_lp*2 + 0], y_lo_1_3_1[_lp*2+1 + 0]));
                    y_lo_1_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_3_1[_lp*2 + 0], y_hi_1_3_1[_lp*2+1 + 0]));
                    y_hi_1_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_1_3_bf16_1[0];
                }
                if (row_hi_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_1_3_bf16_1[0];
                }
                int row_lo_2_1 = tile_m_1 * 128 + warp_m_1 * 64 + 32 + (lane >> 2);
                int row_hi_2_1 = row_lo_2_1 + 8;
                float sa_lo_2_1 = ((row_lo_2_1 < M) ? act_scale[row_lo_2_1] : 0.0f);
                float sa_hi_2_1 = ((row_hi_2_1 < M) ? act_scale[row_hi_2_1] : 0.0f);
                float h_raw_2_0_1[8];
                h_raw_2_0_1[0] = accum_1[64] * sa_lo_2_1 * sw_1[0];
                h_raw_2_0_1[2] = accum_1[66] * sa_hi_2_1 * sw_1[0];
                h_raw_2_0_1[4] = accum_1[68] * sa_lo_2_1 * sw_1[2];
                h_raw_2_0_1[6] = accum_1[70] * sa_hi_2_1 * sw_1[2];
                h_raw_2_0_1[1] = accum_1[65] * sa_lo_2_1 * sw_1[1];
                h_raw_2_0_1[3] = accum_1[67] * sa_hi_2_1 * sw_1[1];
                h_raw_2_0_1[5] = accum_1[69] * sa_lo_2_1 * sw_1[3];
                h_raw_2_0_1[7] = accum_1[71] * sa_hi_2_1 * sw_1[3];
                uint32_t h_raw_2_0_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_0_1[_lp*2 + 0], h_raw_2_0_1[_lp*2+1 + 0]));
                    h_raw_2_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_0_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_0_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_2_0_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_2_0_bf16_1[_pair]));
                }
                float y_lo_2_0_1[2];
                float y_hi_2_0_1[2];
                float _exp2_96 = approx_exp2((-h_raw_2_0_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_96 = approx_rcp(1.0f + _exp2_96);
                float _exp2_97 = approx_exp2((-h_raw_2_0_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_97 = approx_rcp(1.0f + _exp2_97);
                __nv_bfloat16 _cvt_bf16_96 = __float2bfloat16(h_raw_2_0_bf16_f32_1[0] * _rcp_96);
                float _cvt_f32_96 = __bfloat162float(_cvt_bf16_96);
                __nv_bfloat16 _cvt_bf16_97 = __float2bfloat16(h_raw_2_0_bf16_f32_1[2] * _rcp_97);
                float _cvt_f32_97 = __bfloat162float(_cvt_bf16_97);
                y_lo_2_0_1[0] = _cvt_f32_96 * h_raw_2_0_bf16_f32_1[4];
                y_hi_2_0_1[0] = _cvt_f32_97 * h_raw_2_0_bf16_f32_1[6];
                float _exp2_98 = approx_exp2((-h_raw_2_0_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_98 = approx_rcp(1.0f + _exp2_98);
                float _exp2_99 = approx_exp2((-h_raw_2_0_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_99 = approx_rcp(1.0f + _exp2_99);
                __nv_bfloat16 _cvt_bf16_98 = __float2bfloat16(h_raw_2_0_bf16_f32_1[1] * _rcp_98);
                float _cvt_f32_98 = __bfloat162float(_cvt_bf16_98);
                __nv_bfloat16 _cvt_bf16_99 = __float2bfloat16(h_raw_2_0_bf16_f32_1[3] * _rcp_99);
                float _cvt_f32_99 = __bfloat162float(_cvt_bf16_99);
                y_lo_2_0_1[1] = _cvt_f32_98 * h_raw_2_0_bf16_f32_1[5];
                y_hi_2_0_1[1] = _cvt_f32_99 * h_raw_2_0_bf16_f32_1[7];
                uint32_t y_lo_2_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_0_1[_lp*2 + 0], y_lo_2_0_1[_lp*2+1 + 0]));
                    y_lo_2_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_0_1[_lp*2 + 0], y_hi_2_0_1[_lp*2+1 + 0]));
                    y_hi_2_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_2_0_bf16_1[0];
                }
                if (row_hi_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_2_0_bf16_1[0];
                }
                float h_raw_2_1_1[8];
                h_raw_2_1_1[0] = accum_1[72] * sa_lo_2_1 * sw_1[4];
                h_raw_2_1_1[2] = accum_1[74] * sa_hi_2_1 * sw_1[4];
                h_raw_2_1_1[4] = accum_1[76] * sa_lo_2_1 * sw_1[6];
                h_raw_2_1_1[6] = accum_1[78] * sa_hi_2_1 * sw_1[6];
                h_raw_2_1_1[1] = accum_1[73] * sa_lo_2_1 * sw_1[5];
                h_raw_2_1_1[3] = accum_1[75] * sa_hi_2_1 * sw_1[5];
                h_raw_2_1_1[5] = accum_1[77] * sa_lo_2_1 * sw_1[7];
                h_raw_2_1_1[7] = accum_1[79] * sa_hi_2_1 * sw_1[7];
                uint32_t h_raw_2_1_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_1_1[_lp*2 + 0], h_raw_2_1_1[_lp*2+1 + 0]));
                    h_raw_2_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_1_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_1_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_2_1_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_2_1_bf16_1[_pair]));
                }
                float y_lo_2_1_1[2];
                float y_hi_2_1_1[2];
                float _exp2_100 = approx_exp2((-h_raw_2_1_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_100 = approx_rcp(1.0f + _exp2_100);
                float _exp2_101 = approx_exp2((-h_raw_2_1_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_101 = approx_rcp(1.0f + _exp2_101);
                __nv_bfloat16 _cvt_bf16_100 = __float2bfloat16(h_raw_2_1_bf16_f32_1[0] * _rcp_100);
                float _cvt_f32_100 = __bfloat162float(_cvt_bf16_100);
                __nv_bfloat16 _cvt_bf16_101 = __float2bfloat16(h_raw_2_1_bf16_f32_1[2] * _rcp_101);
                float _cvt_f32_101 = __bfloat162float(_cvt_bf16_101);
                y_lo_2_1_1[0] = _cvt_f32_100 * h_raw_2_1_bf16_f32_1[4];
                y_hi_2_1_1[0] = _cvt_f32_101 * h_raw_2_1_bf16_f32_1[6];
                float _exp2_102 = approx_exp2((-h_raw_2_1_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_102 = approx_rcp(1.0f + _exp2_102);
                float _exp2_103 = approx_exp2((-h_raw_2_1_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_103 = approx_rcp(1.0f + _exp2_103);
                __nv_bfloat16 _cvt_bf16_102 = __float2bfloat16(h_raw_2_1_bf16_f32_1[1] * _rcp_102);
                float _cvt_f32_102 = __bfloat162float(_cvt_bf16_102);
                __nv_bfloat16 _cvt_bf16_103 = __float2bfloat16(h_raw_2_1_bf16_f32_1[3] * _rcp_103);
                float _cvt_f32_103 = __bfloat162float(_cvt_bf16_103);
                y_lo_2_1_1[1] = _cvt_f32_102 * h_raw_2_1_bf16_f32_1[5];
                y_hi_2_1_1[1] = _cvt_f32_103 * h_raw_2_1_bf16_f32_1[7];
                uint32_t y_lo_2_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_1_1[_lp*2 + 0], y_lo_2_1_1[_lp*2+1 + 0]));
                    y_lo_2_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_1_1[_lp*2 + 0], y_hi_2_1_1[_lp*2+1 + 0]));
                    y_hi_2_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_2_1_bf16_1[0];
                }
                if (row_hi_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_2_1_bf16_1[0];
                }
                float h_raw_2_2_1[8];
                h_raw_2_2_1[0] = accum_1[80] * sa_lo_2_1 * sw_1[8];
                h_raw_2_2_1[2] = accum_1[82] * sa_hi_2_1 * sw_1[8];
                h_raw_2_2_1[4] = accum_1[84] * sa_lo_2_1 * sw_1[10];
                h_raw_2_2_1[6] = accum_1[86] * sa_hi_2_1 * sw_1[10];
                h_raw_2_2_1[1] = accum_1[81] * sa_lo_2_1 * sw_1[9];
                h_raw_2_2_1[3] = accum_1[83] * sa_hi_2_1 * sw_1[9];
                h_raw_2_2_1[5] = accum_1[85] * sa_lo_2_1 * sw_1[11];
                h_raw_2_2_1[7] = accum_1[87] * sa_hi_2_1 * sw_1[11];
                uint32_t h_raw_2_2_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_2_1[_lp*2 + 0], h_raw_2_2_1[_lp*2+1 + 0]));
                    h_raw_2_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_2_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_2_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_2_2_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_2_2_bf16_1[_pair]));
                }
                float y_lo_2_2_1[2];
                float y_hi_2_2_1[2];
                float _exp2_104 = approx_exp2((-h_raw_2_2_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_104 = approx_rcp(1.0f + _exp2_104);
                float _exp2_105 = approx_exp2((-h_raw_2_2_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_105 = approx_rcp(1.0f + _exp2_105);
                __nv_bfloat16 _cvt_bf16_104 = __float2bfloat16(h_raw_2_2_bf16_f32_1[0] * _rcp_104);
                float _cvt_f32_104 = __bfloat162float(_cvt_bf16_104);
                __nv_bfloat16 _cvt_bf16_105 = __float2bfloat16(h_raw_2_2_bf16_f32_1[2] * _rcp_105);
                float _cvt_f32_105 = __bfloat162float(_cvt_bf16_105);
                y_lo_2_2_1[0] = _cvt_f32_104 * h_raw_2_2_bf16_f32_1[4];
                y_hi_2_2_1[0] = _cvt_f32_105 * h_raw_2_2_bf16_f32_1[6];
                float _exp2_106 = approx_exp2((-h_raw_2_2_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_106 = approx_rcp(1.0f + _exp2_106);
                float _exp2_107 = approx_exp2((-h_raw_2_2_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_107 = approx_rcp(1.0f + _exp2_107);
                __nv_bfloat16 _cvt_bf16_106 = __float2bfloat16(h_raw_2_2_bf16_f32_1[1] * _rcp_106);
                float _cvt_f32_106 = __bfloat162float(_cvt_bf16_106);
                __nv_bfloat16 _cvt_bf16_107 = __float2bfloat16(h_raw_2_2_bf16_f32_1[3] * _rcp_107);
                float _cvt_f32_107 = __bfloat162float(_cvt_bf16_107);
                y_lo_2_2_1[1] = _cvt_f32_106 * h_raw_2_2_bf16_f32_1[5];
                y_hi_2_2_1[1] = _cvt_f32_107 * h_raw_2_2_bf16_f32_1[7];
                uint32_t y_lo_2_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_2_1[_lp*2 + 0], y_lo_2_2_1[_lp*2+1 + 0]));
                    y_lo_2_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_2_1[_lp*2 + 0], y_hi_2_2_1[_lp*2+1 + 0]));
                    y_hi_2_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_2_2_bf16_1[0];
                }
                if (row_hi_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_2_2_bf16_1[0];
                }
                float h_raw_2_3_1[8];
                h_raw_2_3_1[0] = accum_1[88] * sa_lo_2_1 * sw_1[12];
                h_raw_2_3_1[2] = accum_1[90] * sa_hi_2_1 * sw_1[12];
                h_raw_2_3_1[4] = accum_1[92] * sa_lo_2_1 * sw_1[14];
                h_raw_2_3_1[6] = accum_1[94] * sa_hi_2_1 * sw_1[14];
                h_raw_2_3_1[1] = accum_1[89] * sa_lo_2_1 * sw_1[13];
                h_raw_2_3_1[3] = accum_1[91] * sa_hi_2_1 * sw_1[13];
                h_raw_2_3_1[5] = accum_1[93] * sa_lo_2_1 * sw_1[15];
                h_raw_2_3_1[7] = accum_1[95] * sa_hi_2_1 * sw_1[15];
                uint32_t h_raw_2_3_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_3_1[_lp*2 + 0], h_raw_2_3_1[_lp*2+1 + 0]));
                    h_raw_2_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_3_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_3_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_2_3_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_2_3_bf16_1[_pair]));
                }
                float y_lo_2_3_1[2];
                float y_hi_2_3_1[2];
                float _exp2_108 = approx_exp2((-h_raw_2_3_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_108 = approx_rcp(1.0f + _exp2_108);
                float _exp2_109 = approx_exp2((-h_raw_2_3_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_109 = approx_rcp(1.0f + _exp2_109);
                __nv_bfloat16 _cvt_bf16_108 = __float2bfloat16(h_raw_2_3_bf16_f32_1[0] * _rcp_108);
                float _cvt_f32_108 = __bfloat162float(_cvt_bf16_108);
                __nv_bfloat16 _cvt_bf16_109 = __float2bfloat16(h_raw_2_3_bf16_f32_1[2] * _rcp_109);
                float _cvt_f32_109 = __bfloat162float(_cvt_bf16_109);
                y_lo_2_3_1[0] = _cvt_f32_108 * h_raw_2_3_bf16_f32_1[4];
                y_hi_2_3_1[0] = _cvt_f32_109 * h_raw_2_3_bf16_f32_1[6];
                float _exp2_110 = approx_exp2((-h_raw_2_3_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_110 = approx_rcp(1.0f + _exp2_110);
                float _exp2_111 = approx_exp2((-h_raw_2_3_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_111 = approx_rcp(1.0f + _exp2_111);
                __nv_bfloat16 _cvt_bf16_110 = __float2bfloat16(h_raw_2_3_bf16_f32_1[1] * _rcp_110);
                float _cvt_f32_110 = __bfloat162float(_cvt_bf16_110);
                __nv_bfloat16 _cvt_bf16_111 = __float2bfloat16(h_raw_2_3_bf16_f32_1[3] * _rcp_111);
                float _cvt_f32_111 = __bfloat162float(_cvt_bf16_111);
                y_lo_2_3_1[1] = _cvt_f32_110 * h_raw_2_3_bf16_f32_1[5];
                y_hi_2_3_1[1] = _cvt_f32_111 * h_raw_2_3_bf16_f32_1[7];
                uint32_t y_lo_2_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_3_1[_lp*2 + 0], y_lo_2_3_1[_lp*2+1 + 0]));
                    y_lo_2_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_3_1[_lp*2 + 0], y_hi_2_3_1[_lp*2+1 + 0]));
                    y_hi_2_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_2_3_bf16_1[0];
                }
                if (row_hi_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_2_3_bf16_1[0];
                }
                int row_lo_3_1 = tile_m_1 * 128 + warp_m_1 * 64 + 48 + (lane >> 2);
                int row_hi_3_1 = row_lo_3_1 + 8;
                float sa_lo_3_1 = ((row_lo_3_1 < M) ? act_scale[row_lo_3_1] : 0.0f);
                float sa_hi_3_1 = ((row_hi_3_1 < M) ? act_scale[row_hi_3_1] : 0.0f);
                float h_raw_3_0_1[8];
                h_raw_3_0_1[0] = accum_1[96] * sa_lo_3_1 * sw_1[0];
                h_raw_3_0_1[2] = accum_1[98] * sa_hi_3_1 * sw_1[0];
                h_raw_3_0_1[4] = accum_1[100] * sa_lo_3_1 * sw_1[2];
                h_raw_3_0_1[6] = accum_1[102] * sa_hi_3_1 * sw_1[2];
                h_raw_3_0_1[1] = accum_1[97] * sa_lo_3_1 * sw_1[1];
                h_raw_3_0_1[3] = accum_1[99] * sa_hi_3_1 * sw_1[1];
                h_raw_3_0_1[5] = accum_1[101] * sa_lo_3_1 * sw_1[3];
                h_raw_3_0_1[7] = accum_1[103] * sa_hi_3_1 * sw_1[3];
                uint32_t h_raw_3_0_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_0_1[_lp*2 + 0], h_raw_3_0_1[_lp*2+1 + 0]));
                    h_raw_3_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_0_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_0_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_3_0_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_3_0_bf16_1[_pair]));
                }
                float y_lo_3_0_1[2];
                float y_hi_3_0_1[2];
                float _exp2_112 = approx_exp2((-h_raw_3_0_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_112 = approx_rcp(1.0f + _exp2_112);
                float _exp2_113 = approx_exp2((-h_raw_3_0_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_113 = approx_rcp(1.0f + _exp2_113);
                __nv_bfloat16 _cvt_bf16_112 = __float2bfloat16(h_raw_3_0_bf16_f32_1[0] * _rcp_112);
                float _cvt_f32_112 = __bfloat162float(_cvt_bf16_112);
                __nv_bfloat16 _cvt_bf16_113 = __float2bfloat16(h_raw_3_0_bf16_f32_1[2] * _rcp_113);
                float _cvt_f32_113 = __bfloat162float(_cvt_bf16_113);
                y_lo_3_0_1[0] = _cvt_f32_112 * h_raw_3_0_bf16_f32_1[4];
                y_hi_3_0_1[0] = _cvt_f32_113 * h_raw_3_0_bf16_f32_1[6];
                float _exp2_114 = approx_exp2((-h_raw_3_0_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_114 = approx_rcp(1.0f + _exp2_114);
                float _exp2_115 = approx_exp2((-h_raw_3_0_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_115 = approx_rcp(1.0f + _exp2_115);
                __nv_bfloat16 _cvt_bf16_114 = __float2bfloat16(h_raw_3_0_bf16_f32_1[1] * _rcp_114);
                float _cvt_f32_114 = __bfloat162float(_cvt_bf16_114);
                __nv_bfloat16 _cvt_bf16_115 = __float2bfloat16(h_raw_3_0_bf16_f32_1[3] * _rcp_115);
                float _cvt_f32_115 = __bfloat162float(_cvt_bf16_115);
                y_lo_3_0_1[1] = _cvt_f32_114 * h_raw_3_0_bf16_f32_1[5];
                y_hi_3_0_1[1] = _cvt_f32_115 * h_raw_3_0_bf16_f32_1[7];
                uint32_t y_lo_3_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_0_1[_lp*2 + 0], y_lo_3_0_1[_lp*2+1 + 0]));
                    y_lo_3_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_0_1[_lp*2 + 0], y_hi_3_0_1[_lp*2+1 + 0]));
                    y_hi_3_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_3_0_bf16_1[0];
                }
                if (row_hi_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_3_0_bf16_1[0];
                }
                float h_raw_3_1_1[8];
                h_raw_3_1_1[0] = accum_1[104] * sa_lo_3_1 * sw_1[4];
                h_raw_3_1_1[2] = accum_1[106] * sa_hi_3_1 * sw_1[4];
                h_raw_3_1_1[4] = accum_1[108] * sa_lo_3_1 * sw_1[6];
                h_raw_3_1_1[6] = accum_1[110] * sa_hi_3_1 * sw_1[6];
                h_raw_3_1_1[1] = accum_1[105] * sa_lo_3_1 * sw_1[5];
                h_raw_3_1_1[3] = accum_1[107] * sa_hi_3_1 * sw_1[5];
                h_raw_3_1_1[5] = accum_1[109] * sa_lo_3_1 * sw_1[7];
                h_raw_3_1_1[7] = accum_1[111] * sa_hi_3_1 * sw_1[7];
                uint32_t h_raw_3_1_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_1_1[_lp*2 + 0], h_raw_3_1_1[_lp*2+1 + 0]));
                    h_raw_3_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_1_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_1_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_3_1_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_3_1_bf16_1[_pair]));
                }
                float y_lo_3_1_1[2];
                float y_hi_3_1_1[2];
                float _exp2_116 = approx_exp2((-h_raw_3_1_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_116 = approx_rcp(1.0f + _exp2_116);
                float _exp2_117 = approx_exp2((-h_raw_3_1_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_117 = approx_rcp(1.0f + _exp2_117);
                __nv_bfloat16 _cvt_bf16_116 = __float2bfloat16(h_raw_3_1_bf16_f32_1[0] * _rcp_116);
                float _cvt_f32_116 = __bfloat162float(_cvt_bf16_116);
                __nv_bfloat16 _cvt_bf16_117 = __float2bfloat16(h_raw_3_1_bf16_f32_1[2] * _rcp_117);
                float _cvt_f32_117 = __bfloat162float(_cvt_bf16_117);
                y_lo_3_1_1[0] = _cvt_f32_116 * h_raw_3_1_bf16_f32_1[4];
                y_hi_3_1_1[0] = _cvt_f32_117 * h_raw_3_1_bf16_f32_1[6];
                float _exp2_118 = approx_exp2((-h_raw_3_1_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_118 = approx_rcp(1.0f + _exp2_118);
                float _exp2_119 = approx_exp2((-h_raw_3_1_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_119 = approx_rcp(1.0f + _exp2_119);
                __nv_bfloat16 _cvt_bf16_118 = __float2bfloat16(h_raw_3_1_bf16_f32_1[1] * _rcp_118);
                float _cvt_f32_118 = __bfloat162float(_cvt_bf16_118);
                __nv_bfloat16 _cvt_bf16_119 = __float2bfloat16(h_raw_3_1_bf16_f32_1[3] * _rcp_119);
                float _cvt_f32_119 = __bfloat162float(_cvt_bf16_119);
                y_lo_3_1_1[1] = _cvt_f32_118 * h_raw_3_1_bf16_f32_1[5];
                y_hi_3_1_1[1] = _cvt_f32_119 * h_raw_3_1_bf16_f32_1[7];
                uint32_t y_lo_3_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_1_1[_lp*2 + 0], y_lo_3_1_1[_lp*2+1 + 0]));
                    y_lo_3_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_1_1[_lp*2 + 0], y_hi_3_1_1[_lp*2+1 + 0]));
                    y_hi_3_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_3_1_bf16_1[0];
                }
                if (row_hi_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_3_1_bf16_1[0];
                }
                float h_raw_3_2_1[8];
                h_raw_3_2_1[0] = accum_1[112] * sa_lo_3_1 * sw_1[8];
                h_raw_3_2_1[2] = accum_1[114] * sa_hi_3_1 * sw_1[8];
                h_raw_3_2_1[4] = accum_1[116] * sa_lo_3_1 * sw_1[10];
                h_raw_3_2_1[6] = accum_1[118] * sa_hi_3_1 * sw_1[10];
                h_raw_3_2_1[1] = accum_1[113] * sa_lo_3_1 * sw_1[9];
                h_raw_3_2_1[3] = accum_1[115] * sa_hi_3_1 * sw_1[9];
                h_raw_3_2_1[5] = accum_1[117] * sa_lo_3_1 * sw_1[11];
                h_raw_3_2_1[7] = accum_1[119] * sa_hi_3_1 * sw_1[11];
                uint32_t h_raw_3_2_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_2_1[_lp*2 + 0], h_raw_3_2_1[_lp*2+1 + 0]));
                    h_raw_3_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_2_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_2_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_3_2_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_3_2_bf16_1[_pair]));
                }
                float y_lo_3_2_1[2];
                float y_hi_3_2_1[2];
                float _exp2_120 = approx_exp2((-h_raw_3_2_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_120 = approx_rcp(1.0f + _exp2_120);
                float _exp2_121 = approx_exp2((-h_raw_3_2_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_121 = approx_rcp(1.0f + _exp2_121);
                __nv_bfloat16 _cvt_bf16_120 = __float2bfloat16(h_raw_3_2_bf16_f32_1[0] * _rcp_120);
                float _cvt_f32_120 = __bfloat162float(_cvt_bf16_120);
                __nv_bfloat16 _cvt_bf16_121 = __float2bfloat16(h_raw_3_2_bf16_f32_1[2] * _rcp_121);
                float _cvt_f32_121 = __bfloat162float(_cvt_bf16_121);
                y_lo_3_2_1[0] = _cvt_f32_120 * h_raw_3_2_bf16_f32_1[4];
                y_hi_3_2_1[0] = _cvt_f32_121 * h_raw_3_2_bf16_f32_1[6];
                float _exp2_122 = approx_exp2((-h_raw_3_2_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_122 = approx_rcp(1.0f + _exp2_122);
                float _exp2_123 = approx_exp2((-h_raw_3_2_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_123 = approx_rcp(1.0f + _exp2_123);
                __nv_bfloat16 _cvt_bf16_122 = __float2bfloat16(h_raw_3_2_bf16_f32_1[1] * _rcp_122);
                float _cvt_f32_122 = __bfloat162float(_cvt_bf16_122);
                __nv_bfloat16 _cvt_bf16_123 = __float2bfloat16(h_raw_3_2_bf16_f32_1[3] * _rcp_123);
                float _cvt_f32_123 = __bfloat162float(_cvt_bf16_123);
                y_lo_3_2_1[1] = _cvt_f32_122 * h_raw_3_2_bf16_f32_1[5];
                y_hi_3_2_1[1] = _cvt_f32_123 * h_raw_3_2_bf16_f32_1[7];
                uint32_t y_lo_3_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_2_1[_lp*2 + 0], y_lo_3_2_1[_lp*2+1 + 0]));
                    y_lo_3_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_2_1[_lp*2 + 0], y_hi_3_2_1[_lp*2+1 + 0]));
                    y_hi_3_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_3_2_bf16_1[0];
                }
                if (row_hi_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_3_2_bf16_1[0];
                }
                float h_raw_3_3_1[8];
                h_raw_3_3_1[0] = accum_1[120] * sa_lo_3_1 * sw_1[12];
                h_raw_3_3_1[2] = accum_1[122] * sa_hi_3_1 * sw_1[12];
                h_raw_3_3_1[4] = accum_1[124] * sa_lo_3_1 * sw_1[14];
                h_raw_3_3_1[6] = accum_1[126] * sa_hi_3_1 * sw_1[14];
                h_raw_3_3_1[1] = accum_1[121] * sa_lo_3_1 * sw_1[13];
                h_raw_3_3_1[3] = accum_1[123] * sa_hi_3_1 * sw_1[13];
                h_raw_3_3_1[5] = accum_1[125] * sa_lo_3_1 * sw_1[15];
                h_raw_3_3_1[7] = accum_1[127] * sa_hi_3_1 * sw_1[15];
                uint32_t h_raw_3_3_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_3_1[_lp*2 + 0], h_raw_3_3_1[_lp*2+1 + 0]));
                    h_raw_3_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_3_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_3_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_3_3_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_3_3_bf16_1[_pair]));
                }
                float y_lo_3_3_1[2];
                float y_hi_3_3_1[2];
                float _exp2_124 = approx_exp2((-h_raw_3_3_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_124 = approx_rcp(1.0f + _exp2_124);
                float _exp2_125 = approx_exp2((-h_raw_3_3_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_125 = approx_rcp(1.0f + _exp2_125);
                __nv_bfloat16 _cvt_bf16_124 = __float2bfloat16(h_raw_3_3_bf16_f32_1[0] * _rcp_124);
                float _cvt_f32_124 = __bfloat162float(_cvt_bf16_124);
                __nv_bfloat16 _cvt_bf16_125 = __float2bfloat16(h_raw_3_3_bf16_f32_1[2] * _rcp_125);
                float _cvt_f32_125 = __bfloat162float(_cvt_bf16_125);
                y_lo_3_3_1[0] = _cvt_f32_124 * h_raw_3_3_bf16_f32_1[4];
                y_hi_3_3_1[0] = _cvt_f32_125 * h_raw_3_3_bf16_f32_1[6];
                float _exp2_126 = approx_exp2((-h_raw_3_3_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_126 = approx_rcp(1.0f + _exp2_126);
                float _exp2_127 = approx_exp2((-h_raw_3_3_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_127 = approx_rcp(1.0f + _exp2_127);
                __nv_bfloat16 _cvt_bf16_126 = __float2bfloat16(h_raw_3_3_bf16_f32_1[1] * _rcp_126);
                float _cvt_f32_126 = __bfloat162float(_cvt_bf16_126);
                __nv_bfloat16 _cvt_bf16_127 = __float2bfloat16(h_raw_3_3_bf16_f32_1[3] * _rcp_127);
                float _cvt_f32_127 = __bfloat162float(_cvt_bf16_127);
                y_lo_3_3_1[1] = _cvt_f32_126 * h_raw_3_3_bf16_f32_1[5];
                y_hi_3_3_1[1] = _cvt_f32_127 * h_raw_3_3_bf16_f32_1[7];
                uint32_t y_lo_3_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_3_1[_lp*2 + 0], y_lo_3_3_1[_lp*2+1 + 0]));
                    y_lo_3_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_3_1[_lp*2 + 0], y_hi_3_3_1[_lp*2+1 + 0]));
                    y_hi_3_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_3_3_bf16_1[0];
                }
                if (row_hi_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_3_3_bf16_1[0];
                }
            }
        }
    }

    // Cleanup
}

}  // namespace h3_fc1_swiglu_gemm_fp8_sm120a
#undef GROUP_M
#undef H3_FC1_INF
#undef NUM_EMPTY_PIPE_STAGES
#undef NUM_FULL_PIPE_STAGES
#undef SMEM_A_STAGE_OFF
#undef SMEM_A_STAGE_STAGE_BYTES
#undef SMEM_A_STAGE_STRIDE
#undef SMEM_B_STAGE_OFF
#undef SMEM_B_STAGE_STAGE_BYTES
#undef SMEM_B_STAGE_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef ab_empty_addr
#undef ab_full_addr

namespace h3_fc1_swiglu_gemm_nvfp4_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_FC1_INF CUDART_INF_F
#define NUM_FULL_PIPE_STAGES 3
#define NUM_EMPTY_PIPE_STAGES 3
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 8192
#define SMEM_A_STAGE_STRIDE 8192
#define SMEM_B_STAGE_OFF 25600
#define SMEM_B_STAGE_STAGE_BYTES 16384
#define SMEM_B_STAGE_STRIDE 16384
#define SMEM_SFB_STAGE_OFF 74752
#define SMEM_SFB_STAGE_STAGE_BYTES 2048
#define SMEM_SFB_STAGE_STRIDE 2048
#define SMEM_SFA_PAIRS_OFF 80896
#define SMEM_SFA_PAIRS_STAGE_BYTES 4096
#define SMEM_SFA_PAIRS_STRIDE 4096
#define SMEM_TOTAL 84992
#define THREADS 256
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


__device__ __forceinline__ uint32_t mbarrier_try_wait_plain(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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


__global__ __launch_bounds__(256, 1) void
kernel_h3_fc1_swiglu_gemm(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ act_scale, float* __restrict__ w_scale, unsigned int* __restrict__ out, int M, int num_m_tiles, int total_tiles, float alpha)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 24)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 25600);
    const int B_stage_addr = smem + 25600;
    unsigned int* SFB_stage = reinterpret_cast<unsigned int*>(smem_raw + 74752);
    const int SFB_stage_addr = smem + 74752;
    unsigned int* SFA_pairs = reinterpret_cast<unsigned int*>(smem_raw + 80896);
    const int SFA_pairs_addr = smem + 80896;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'full_pipe' ---
            // ab_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // --- pipeline 'empty_pipe' ---
            // ab_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 24, 8);
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: lead ----
    if (warp == 0) {
        { // lead_main
            unsigned int load_stage = 0;
            unsigned int load_g = 0;
            unsigned int mma_stage = 0;
            unsigned int mma_kk = 0;
            int warp_m = 0;
            int warp_n = 0;
            float accum[128];
            unsigned int a_frag[16];
            unsigned int b_frag[16];
            unsigned int _phase_ab_empty = 1;
            if (lane == 0) {
                for (int _prologue = 0; _prologue < 2; _prologue++) {
                    if (bid + (int)(load_g / 42) * num_bids < total_tiles) {
                        mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                        mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, (((load_g & 1) == 0) ? 28672 : 26624));
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(A_stage_addr + load_stage * 8192), "l"((&A)), "r"(((int)load_g - (int)(load_g / 42) * 42) * 64), "r"(((bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M + (bid + (int)(load_g / 42) * num_bids - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * (GROUP_M * 112)) % ((num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M : GROUP_M)) * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(B_stage_addr + load_stage * 16384), "l"((&B)), "r"(((int)load_g - (int)(load_g / 42) * 42) * 64), "r"((bid + (int)(load_g / 42) * num_bids - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * (GROUP_M * 112)) / ((num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M : GROUP_M) * 256), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(SFB_stage_addr + load_stage * 2048), "l"((&SFB)), "r"(0), "r"((bid + (int)(load_g / 42) * num_bids - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * (GROUP_M * 112)) / ((num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M : GROUP_M) * 2 * 168 + ((int)load_g - (int)(load_g / 42) * 42) * 4), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            :: "r"(SFB_stage_addr + load_stage * 2048 + 1024), "l"((&SFB)), "r"(0), "r"(((bid + (int)(load_g / 42) * num_bids - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * (GROUP_M * 112)) / ((num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M : GROUP_M) * 2 + 1) * 168 + ((int)load_g - (int)(load_g / 42) * 42) * 4), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        if ((load_g & 1) == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(SFA_pairs_addr + (load_g >> 1 & 1) * 2048), "l"((&SFA)), "r"(((int)load_g - (int)(load_g / 42) * 42) * 8), "r"(((bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M + (bid + (int)(load_g / 42) * num_bids - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * (GROUP_M * 112)) % ((num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M : GROUP_M)) * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                        }
                    }
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_ab_empty ^= 1; }
                    load_g = load_g + 1;
                }
            }
            __syncwarp();
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile = bid; tile < total_tiles; tile += num_bids) {
                int tile_m = tile / (GROUP_M * 112) * GROUP_M + (tile - tile / (GROUP_M * 112) * (GROUP_M * 112)) % ((num_m_tiles - tile / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 112) * GROUP_M : GROUP_M);
                int tile_n = (tile - tile / (GROUP_M * 112) * (GROUP_M * 112)) / ((num_m_tiles - tile / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 112) * GROUP_M : GROUP_M);
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
                accum[64] = 0.0f;
                accum[65] = 0.0f;
                accum[66] = 0.0f;
                accum[67] = 0.0f;
                accum[68] = 0.0f;
                accum[69] = 0.0f;
                accum[70] = 0.0f;
                accum[71] = 0.0f;
                accum[72] = 0.0f;
                accum[73] = 0.0f;
                accum[74] = 0.0f;
                accum[75] = 0.0f;
                accum[76] = 0.0f;
                accum[77] = 0.0f;
                accum[78] = 0.0f;
                accum[79] = 0.0f;
                accum[80] = 0.0f;
                accum[81] = 0.0f;
                accum[82] = 0.0f;
                accum[83] = 0.0f;
                accum[84] = 0.0f;
                accum[85] = 0.0f;
                accum[86] = 0.0f;
                accum[87] = 0.0f;
                accum[88] = 0.0f;
                accum[89] = 0.0f;
                accum[90] = 0.0f;
                accum[91] = 0.0f;
                accum[92] = 0.0f;
                accum[93] = 0.0f;
                accum[94] = 0.0f;
                accum[95] = 0.0f;
                accum[96] = 0.0f;
                accum[97] = 0.0f;
                accum[98] = 0.0f;
                accum[99] = 0.0f;
                accum[100] = 0.0f;
                accum[101] = 0.0f;
                accum[102] = 0.0f;
                accum[103] = 0.0f;
                accum[104] = 0.0f;
                accum[105] = 0.0f;
                accum[106] = 0.0f;
                accum[107] = 0.0f;
                accum[108] = 0.0f;
                accum[109] = 0.0f;
                accum[110] = 0.0f;
                accum[111] = 0.0f;
                accum[112] = 0.0f;
                accum[113] = 0.0f;
                accum[114] = 0.0f;
                accum[115] = 0.0f;
                accum[116] = 0.0f;
                accum[117] = 0.0f;
                accum[118] = 0.0f;
                accum[119] = 0.0f;
                accum[120] = 0.0f;
                accum[121] = 0.0f;
                accum[122] = 0.0f;
                accum[123] = 0.0f;
                accum[124] = 0.0f;
                accum[125] = 0.0f;
                accum[126] = 0.0f;
                accum[127] = 0.0f;
                #pragma unroll 1
                for (int k_tile = 0; k_tile < 42; k_tile++) {
                    if (lane == 0) {
                        if (bid + (int)(load_g / 42) * num_bids < total_tiles) {
                            mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, (((load_g & 1) == 0) ? 28672 : 26624));
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(A_stage_addr + load_stage * 8192), "l"((&A)), "r"(((int)load_g - (int)(load_g / 42) * 42) * 64), "r"(((bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M + (bid + (int)(load_g / 42) * num_bids - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * (GROUP_M * 112)) % ((num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M : GROUP_M)) * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(B_stage_addr + load_stage * 16384), "l"((&B)), "r"(((int)load_g - (int)(load_g / 42) * 42) * 64), "r"((bid + (int)(load_g / 42) * num_bids - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * (GROUP_M * 112)) / ((num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M : GROUP_M) * 256), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(SFB_stage_addr + load_stage * 2048), "l"((&SFB)), "r"(0), "r"((bid + (int)(load_g / 42) * num_bids - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * (GROUP_M * 112)) / ((num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M : GROUP_M) * 2 * 168 + ((int)load_g - (int)(load_g / 42) * 42) * 4), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(SFB_stage_addr + load_stage * 2048 + 1024), "l"((&SFB)), "r"(0), "r"(((bid + (int)(load_g / 42) * num_bids - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * (GROUP_M * 112)) / ((num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M : GROUP_M) * 2 + 1) * 168 + ((int)load_g - (int)(load_g / 42) * 42) * 4), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            if ((load_g & 1) == 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3}], [%4];"
                                    :: "r"(SFA_pairs_addr + (load_g >> 1 & 1) * 2048), "l"((&SFA)), "r"(((int)load_g - (int)(load_g / 42) * 42) * 8), "r"(((bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M + (bid + (int)(load_g / 42) * num_bids - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * (GROUP_M * 112)) % ((num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - (bid + (int)(load_g / 42) * num_bids) / (GROUP_M * 112) * GROUP_M : GROUP_M)) * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            }
                        }
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_ab_empty ^= 1; }
                        load_g = load_g + 1;
                    }
                    __syncwarp();
                    mbarrier_wait(ab_full_addr + (mma_stage) * 8, _phase_ab_full);
                    for (int k_step = 0; k_step < 2; k_step++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(A_stage_addr + mma_stage * 8192 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                            : "r"(A_stage_addr + mma_stage * 8192 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                            : "r"(A_stage_addr + mma_stage * 8192 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                            : "r"(A_stage_addr + mma_stage * 8192 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                            : "r"(B_stage_addr + mma_stage * 16384 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        unsigned int _SFA_pairs_reg_0[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_pairs);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFA_pairs_reg_0[_lr] = _smem_ptr[((mma_kk >> 1 & 1) * 512 + (mma_kk & 1) * 2 + (unsigned int)k_step + (unsigned int)((warp_m * 64 + (lane & 1) * 8 + (lane >> 2)) * 4)) + _lr];
                        }
                        unsigned int _SFA_pairs_reg_1[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_pairs);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFA_pairs_reg_1[_lr] = _smem_ptr[((mma_kk >> 1 & 1) * 512 + (mma_kk & 1) * 2 + (unsigned int)k_step + (unsigned int)((warp_m * 64 + 16 + (lane & 1) * 8 + (lane >> 2)) * 4)) + _lr];
                        }
                        unsigned int _SFA_pairs_reg_2[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_pairs);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFA_pairs_reg_2[_lr] = _smem_ptr[((mma_kk >> 1 & 1) * 512 + (mma_kk & 1) * 2 + (unsigned int)k_step + (unsigned int)((warp_m * 64 + 32 + (lane & 1) * 8 + (lane >> 2)) * 4)) + _lr];
                        }
                        unsigned int _SFA_pairs_reg_3[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_pairs);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFA_pairs_reg_3[_lr] = _smem_ptr[((mma_kk >> 1 & 1) * 512 + (mma_kk & 1) * 2 + (unsigned int)k_step + (unsigned int)((warp_m * 64 + 48 + (lane & 1) * 8 + (lane >> 2)) * 4)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_0[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_0[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(((warp_n >> 1) * 2 + k_step) * 128) + (unsigned int)((lane >> 2) * 4) + (unsigned int)((warp_n & 1) * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_1[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_1[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(((warp_n >> 1) * 2 + k_step) * 128) + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)((warp_n & 1) * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_2[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_2[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(((warp_n >> 1) * 2 + k_step) * 128) + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)((warp_n & 1) * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_3[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_3[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(((warp_n >> 1) * 2 + k_step) * 128) + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)((warp_n & 1) * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_4[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_4[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(((warp_n >> 1) * 2 + k_step) * 128) + (unsigned int)((lane >> 2) * 4) + (unsigned int)((warp_n & 1) * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_5[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_5[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(((warp_n >> 1) * 2 + k_step) * 128) + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)((warp_n & 1) * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_6[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_6[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(((warp_n >> 1) * 2 + k_step) * 128) + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)((warp_n & 1) * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_7[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_7[_lr] = _smem_ptr[(mma_stage * 512 + (unsigned int)(((warp_n >> 1) * 2 + k_step) * 128) + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)((warp_n & 1) * 2 + 1)) + _lr];
                        }
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_pairs_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_pairs_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_pairs_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_pairs_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_pairs_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_pairs_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_pairs_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_pairs_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_pairs_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_pairs_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_pairs_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_pairs_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_pairs_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_pairs_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_pairs_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                            : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_pairs_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_pairs_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_pairs_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_pairs_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_pairs_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_pairs_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_pairs_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_pairs_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                            : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_pairs_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_pairs_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_pairs_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_pairs_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_pairs_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_pairs_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_pairs_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_pairs_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                            : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_pairs_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    }
                    __syncwarp();
                    if (lane == 0) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_ab_full ^= 1; }
                    mma_kk = mma_kk + 1;
                }
                float sw[16];
                sw[0] = alpha;
                sw[2] = alpha;
                sw[1] = alpha;
                sw[3] = alpha;
                sw[4] = alpha;
                sw[6] = alpha;
                sw[5] = alpha;
                sw[7] = alpha;
                sw[8] = alpha;
                sw[10] = alpha;
                sw[9] = alpha;
                sw[11] = alpha;
                sw[12] = alpha;
                sw[14] = alpha;
                sw[13] = alpha;
                sw[15] = alpha;
                int row_lo_0 = tile_m * 128 + warp_m * 64 + (lane >> 2);
                int row_hi_0 = row_lo_0 + 8;
                float sa_lo_0 = 1.0f;
                float sa_hi_0 = 1.0f;
                float h_raw_0_0[8];
                h_raw_0_0[0] = accum[0] * sa_lo_0 * sw[0];
                h_raw_0_0[2] = accum[2] * sa_hi_0 * sw[0];
                h_raw_0_0[4] = accum[4] * sa_lo_0 * sw[2];
                h_raw_0_0[6] = accum[6] * sa_hi_0 * sw[2];
                h_raw_0_0[1] = accum[1] * sa_lo_0 * sw[1];
                h_raw_0_0[3] = accum[3] * sa_hi_0 * sw[1];
                h_raw_0_0[5] = accum[5] * sa_lo_0 * sw[3];
                h_raw_0_0[7] = accum[7] * sa_hi_0 * sw[3];
                uint32_t h_raw_0_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_0[_lp*2 + 0], h_raw_0_0[_lp*2+1 + 0]));
                    h_raw_0_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_0_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_0_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_0_0_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_0_0_bf16[_pair]));
                }
                float y_lo_0_0[2];
                float y_hi_0_0[2];
                float _exp2_0 = approx_exp2((-h_raw_0_0_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                float _exp2_1 = approx_exp2((-h_raw_0_0_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(h_raw_0_0_bf16_f32[0] * _rcp_0);
                float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(h_raw_0_0_bf16_f32[2] * _rcp_1);
                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                y_lo_0_0[0] = _cvt_f32_0 * h_raw_0_0_bf16_f32[4];
                y_hi_0_0[0] = _cvt_f32_1 * h_raw_0_0_bf16_f32[6];
                float _exp2_2 = approx_exp2((-h_raw_0_0_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                float _exp2_3 = approx_exp2((-h_raw_0_0_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_3 = approx_rcp(1.0f + _exp2_3);
                __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(h_raw_0_0_bf16_f32[1] * _rcp_2);
                float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(h_raw_0_0_bf16_f32[3] * _rcp_3);
                float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                y_lo_0_0[1] = _cvt_f32_2 * h_raw_0_0_bf16_f32[5];
                y_hi_0_0[1] = _cvt_f32_3 * h_raw_0_0_bf16_f32[7];
                uint32_t y_lo_0_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_0[_lp*2 + 0], y_lo_0_0[_lp*2+1 + 0]));
                    y_lo_0_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_0[_lp*2 + 0], y_hi_0_0[_lp*2+1 + 0]));
                    y_hi_0_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_0_0_bf16[0];
                }
                if (row_hi_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_0_0_bf16[0];
                }
                float h_raw_0_1[8];
                h_raw_0_1[0] = accum[8] * sa_lo_0 * sw[4];
                h_raw_0_1[2] = accum[10] * sa_hi_0 * sw[4];
                h_raw_0_1[4] = accum[12] * sa_lo_0 * sw[6];
                h_raw_0_1[6] = accum[14] * sa_hi_0 * sw[6];
                h_raw_0_1[1] = accum[9] * sa_lo_0 * sw[5];
                h_raw_0_1[3] = accum[11] * sa_hi_0 * sw[5];
                h_raw_0_1[5] = accum[13] * sa_lo_0 * sw[7];
                h_raw_0_1[7] = accum[15] * sa_hi_0 * sw[7];
                uint32_t h_raw_0_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_1[_lp*2 + 0], h_raw_0_1[_lp*2+1 + 0]));
                    h_raw_0_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_1_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_1_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_0_1_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_0_1_bf16[_pair]));
                }
                float y_lo_0_1[2];
                float y_hi_0_1[2];
                float _exp2_4 = approx_exp2((-h_raw_0_1_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                float _exp2_5 = approx_exp2((-h_raw_0_1_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(h_raw_0_1_bf16_f32[0] * _rcp_4);
                float _cvt_f32_4 = __bfloat162float(_cvt_bf16_4);
                __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(h_raw_0_1_bf16_f32[2] * _rcp_5);
                float _cvt_f32_5 = __bfloat162float(_cvt_bf16_5);
                y_lo_0_1[0] = _cvt_f32_4 * h_raw_0_1_bf16_f32[4];
                y_hi_0_1[0] = _cvt_f32_5 * h_raw_0_1_bf16_f32[6];
                float _exp2_6 = approx_exp2((-h_raw_0_1_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                float _exp2_7 = approx_exp2((-h_raw_0_1_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(h_raw_0_1_bf16_f32[1] * _rcp_6);
                float _cvt_f32_6 = __bfloat162float(_cvt_bf16_6);
                __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(h_raw_0_1_bf16_f32[3] * _rcp_7);
                float _cvt_f32_7 = __bfloat162float(_cvt_bf16_7);
                y_lo_0_1[1] = _cvt_f32_6 * h_raw_0_1_bf16_f32[5];
                y_hi_0_1[1] = _cvt_f32_7 * h_raw_0_1_bf16_f32[7];
                uint32_t y_lo_0_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_1[_lp*2 + 0], y_lo_0_1[_lp*2+1 + 0]));
                    y_lo_0_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_1[_lp*2 + 0], y_hi_0_1[_lp*2+1 + 0]));
                    y_hi_0_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_0_1_bf16[0];
                }
                if (row_hi_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_0_1_bf16[0];
                }
                float h_raw_0_2[8];
                h_raw_0_2[0] = accum[16] * sa_lo_0 * sw[8];
                h_raw_0_2[2] = accum[18] * sa_hi_0 * sw[8];
                h_raw_0_2[4] = accum[20] * sa_lo_0 * sw[10];
                h_raw_0_2[6] = accum[22] * sa_hi_0 * sw[10];
                h_raw_0_2[1] = accum[17] * sa_lo_0 * sw[9];
                h_raw_0_2[3] = accum[19] * sa_hi_0 * sw[9];
                h_raw_0_2[5] = accum[21] * sa_lo_0 * sw[11];
                h_raw_0_2[7] = accum[23] * sa_hi_0 * sw[11];
                uint32_t h_raw_0_2_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_2[_lp*2 + 0], h_raw_0_2[_lp*2+1 + 0]));
                    h_raw_0_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_2_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_2_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_0_2_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_0_2_bf16[_pair]));
                }
                float y_lo_0_2[2];
                float y_hi_0_2[2];
                float _exp2_8 = approx_exp2((-h_raw_0_2_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                float _exp2_9 = approx_exp2((-h_raw_0_2_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                __nv_bfloat16 _cvt_bf16_8 = __float2bfloat16(h_raw_0_2_bf16_f32[0] * _rcp_8);
                float _cvt_f32_8 = __bfloat162float(_cvt_bf16_8);
                __nv_bfloat16 _cvt_bf16_9 = __float2bfloat16(h_raw_0_2_bf16_f32[2] * _rcp_9);
                float _cvt_f32_9 = __bfloat162float(_cvt_bf16_9);
                y_lo_0_2[0] = _cvt_f32_8 * h_raw_0_2_bf16_f32[4];
                y_hi_0_2[0] = _cvt_f32_9 * h_raw_0_2_bf16_f32[6];
                float _exp2_10 = approx_exp2((-h_raw_0_2_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                float _exp2_11 = approx_exp2((-h_raw_0_2_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                __nv_bfloat16 _cvt_bf16_10 = __float2bfloat16(h_raw_0_2_bf16_f32[1] * _rcp_10);
                float _cvt_f32_10 = __bfloat162float(_cvt_bf16_10);
                __nv_bfloat16 _cvt_bf16_11 = __float2bfloat16(h_raw_0_2_bf16_f32[3] * _rcp_11);
                float _cvt_f32_11 = __bfloat162float(_cvt_bf16_11);
                y_lo_0_2[1] = _cvt_f32_10 * h_raw_0_2_bf16_f32[5];
                y_hi_0_2[1] = _cvt_f32_11 * h_raw_0_2_bf16_f32[7];
                uint32_t y_lo_0_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_2[_lp*2 + 0], y_lo_0_2[_lp*2+1 + 0]));
                    y_lo_0_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_2[_lp*2 + 0], y_hi_0_2[_lp*2+1 + 0]));
                    y_hi_0_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_0_2_bf16[0];
                }
                if (row_hi_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_0_2_bf16[0];
                }
                float h_raw_0_3[8];
                h_raw_0_3[0] = accum[24] * sa_lo_0 * sw[12];
                h_raw_0_3[2] = accum[26] * sa_hi_0 * sw[12];
                h_raw_0_3[4] = accum[28] * sa_lo_0 * sw[14];
                h_raw_0_3[6] = accum[30] * sa_hi_0 * sw[14];
                h_raw_0_3[1] = accum[25] * sa_lo_0 * sw[13];
                h_raw_0_3[3] = accum[27] * sa_hi_0 * sw[13];
                h_raw_0_3[5] = accum[29] * sa_lo_0 * sw[15];
                h_raw_0_3[7] = accum[31] * sa_hi_0 * sw[15];
                uint32_t h_raw_0_3_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_3[_lp*2 + 0], h_raw_0_3[_lp*2+1 + 0]));
                    h_raw_0_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_3_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_3_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_0_3_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_0_3_bf16[_pair]));
                }
                float y_lo_0_3[2];
                float y_hi_0_3[2];
                float _exp2_12 = approx_exp2((-h_raw_0_3_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                float _exp2_13 = approx_exp2((-h_raw_0_3_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                __nv_bfloat16 _cvt_bf16_12 = __float2bfloat16(h_raw_0_3_bf16_f32[0] * _rcp_12);
                float _cvt_f32_12 = __bfloat162float(_cvt_bf16_12);
                __nv_bfloat16 _cvt_bf16_13 = __float2bfloat16(h_raw_0_3_bf16_f32[2] * _rcp_13);
                float _cvt_f32_13 = __bfloat162float(_cvt_bf16_13);
                y_lo_0_3[0] = _cvt_f32_12 * h_raw_0_3_bf16_f32[4];
                y_hi_0_3[0] = _cvt_f32_13 * h_raw_0_3_bf16_f32[6];
                float _exp2_14 = approx_exp2((-h_raw_0_3_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                float _exp2_15 = approx_exp2((-h_raw_0_3_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                __nv_bfloat16 _cvt_bf16_14 = __float2bfloat16(h_raw_0_3_bf16_f32[1] * _rcp_14);
                float _cvt_f32_14 = __bfloat162float(_cvt_bf16_14);
                __nv_bfloat16 _cvt_bf16_15 = __float2bfloat16(h_raw_0_3_bf16_f32[3] * _rcp_15);
                float _cvt_f32_15 = __bfloat162float(_cvt_bf16_15);
                y_lo_0_3[1] = _cvt_f32_14 * h_raw_0_3_bf16_f32[5];
                y_hi_0_3[1] = _cvt_f32_15 * h_raw_0_3_bf16_f32[7];
                uint32_t y_lo_0_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_3[_lp*2 + 0], y_lo_0_3[_lp*2+1 + 0]));
                    y_lo_0_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_3[_lp*2 + 0], y_hi_0_3[_lp*2+1 + 0]));
                    y_hi_0_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_0_3_bf16[0];
                }
                if (row_hi_0 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_0_3_bf16[0];
                }
                int row_lo_1 = tile_m * 128 + warp_m * 64 + 16 + (lane >> 2);
                int row_hi_1 = row_lo_1 + 8;
                float sa_lo_1 = 1.0f;
                float sa_hi_1 = 1.0f;
                float h_raw_1_0[8];
                h_raw_1_0[0] = accum[32] * sa_lo_1 * sw[0];
                h_raw_1_0[2] = accum[34] * sa_hi_1 * sw[0];
                h_raw_1_0[4] = accum[36] * sa_lo_1 * sw[2];
                h_raw_1_0[6] = accum[38] * sa_hi_1 * sw[2];
                h_raw_1_0[1] = accum[33] * sa_lo_1 * sw[1];
                h_raw_1_0[3] = accum[35] * sa_hi_1 * sw[1];
                h_raw_1_0[5] = accum[37] * sa_lo_1 * sw[3];
                h_raw_1_0[7] = accum[39] * sa_hi_1 * sw[3];
                uint32_t h_raw_1_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_0[_lp*2 + 0], h_raw_1_0[_lp*2+1 + 0]));
                    h_raw_1_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_0_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_0_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_1_0_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_1_0_bf16[_pair]));
                }
                float y_lo_1_0[2];
                float y_hi_1_0[2];
                float _exp2_16 = approx_exp2((-h_raw_1_0_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_16 = approx_rcp(1.0f + _exp2_16);
                float _exp2_17 = approx_exp2((-h_raw_1_0_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_17 = approx_rcp(1.0f + _exp2_17);
                __nv_bfloat16 _cvt_bf16_16 = __float2bfloat16(h_raw_1_0_bf16_f32[0] * _rcp_16);
                float _cvt_f32_16 = __bfloat162float(_cvt_bf16_16);
                __nv_bfloat16 _cvt_bf16_17 = __float2bfloat16(h_raw_1_0_bf16_f32[2] * _rcp_17);
                float _cvt_f32_17 = __bfloat162float(_cvt_bf16_17);
                y_lo_1_0[0] = _cvt_f32_16 * h_raw_1_0_bf16_f32[4];
                y_hi_1_0[0] = _cvt_f32_17 * h_raw_1_0_bf16_f32[6];
                float _exp2_18 = approx_exp2((-h_raw_1_0_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_18 = approx_rcp(1.0f + _exp2_18);
                float _exp2_19 = approx_exp2((-h_raw_1_0_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_19 = approx_rcp(1.0f + _exp2_19);
                __nv_bfloat16 _cvt_bf16_18 = __float2bfloat16(h_raw_1_0_bf16_f32[1] * _rcp_18);
                float _cvt_f32_18 = __bfloat162float(_cvt_bf16_18);
                __nv_bfloat16 _cvt_bf16_19 = __float2bfloat16(h_raw_1_0_bf16_f32[3] * _rcp_19);
                float _cvt_f32_19 = __bfloat162float(_cvt_bf16_19);
                y_lo_1_0[1] = _cvt_f32_18 * h_raw_1_0_bf16_f32[5];
                y_hi_1_0[1] = _cvt_f32_19 * h_raw_1_0_bf16_f32[7];
                uint32_t y_lo_1_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_0[_lp*2 + 0], y_lo_1_0[_lp*2+1 + 0]));
                    y_lo_1_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_0[_lp*2 + 0], y_hi_1_0[_lp*2+1 + 0]));
                    y_hi_1_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_1_0_bf16[0];
                }
                if (row_hi_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_1_0_bf16[0];
                }
                float h_raw_1_1[8];
                h_raw_1_1[0] = accum[40] * sa_lo_1 * sw[4];
                h_raw_1_1[2] = accum[42] * sa_hi_1 * sw[4];
                h_raw_1_1[4] = accum[44] * sa_lo_1 * sw[6];
                h_raw_1_1[6] = accum[46] * sa_hi_1 * sw[6];
                h_raw_1_1[1] = accum[41] * sa_lo_1 * sw[5];
                h_raw_1_1[3] = accum[43] * sa_hi_1 * sw[5];
                h_raw_1_1[5] = accum[45] * sa_lo_1 * sw[7];
                h_raw_1_1[7] = accum[47] * sa_hi_1 * sw[7];
                uint32_t h_raw_1_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_1[_lp*2 + 0], h_raw_1_1[_lp*2+1 + 0]));
                    h_raw_1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_1_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_1_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_1_1_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_1_1_bf16[_pair]));
                }
                float y_lo_1_1[2];
                float y_hi_1_1[2];
                float _exp2_20 = approx_exp2((-h_raw_1_1_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_20 = approx_rcp(1.0f + _exp2_20);
                float _exp2_21 = approx_exp2((-h_raw_1_1_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_21 = approx_rcp(1.0f + _exp2_21);
                __nv_bfloat16 _cvt_bf16_20 = __float2bfloat16(h_raw_1_1_bf16_f32[0] * _rcp_20);
                float _cvt_f32_20 = __bfloat162float(_cvt_bf16_20);
                __nv_bfloat16 _cvt_bf16_21 = __float2bfloat16(h_raw_1_1_bf16_f32[2] * _rcp_21);
                float _cvt_f32_21 = __bfloat162float(_cvt_bf16_21);
                y_lo_1_1[0] = _cvt_f32_20 * h_raw_1_1_bf16_f32[4];
                y_hi_1_1[0] = _cvt_f32_21 * h_raw_1_1_bf16_f32[6];
                float _exp2_22 = approx_exp2((-h_raw_1_1_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_22 = approx_rcp(1.0f + _exp2_22);
                float _exp2_23 = approx_exp2((-h_raw_1_1_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_23 = approx_rcp(1.0f + _exp2_23);
                __nv_bfloat16 _cvt_bf16_22 = __float2bfloat16(h_raw_1_1_bf16_f32[1] * _rcp_22);
                float _cvt_f32_22 = __bfloat162float(_cvt_bf16_22);
                __nv_bfloat16 _cvt_bf16_23 = __float2bfloat16(h_raw_1_1_bf16_f32[3] * _rcp_23);
                float _cvt_f32_23 = __bfloat162float(_cvt_bf16_23);
                y_lo_1_1[1] = _cvt_f32_22 * h_raw_1_1_bf16_f32[5];
                y_hi_1_1[1] = _cvt_f32_23 * h_raw_1_1_bf16_f32[7];
                uint32_t y_lo_1_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_1[_lp*2 + 0], y_lo_1_1[_lp*2+1 + 0]));
                    y_lo_1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_1[_lp*2 + 0], y_hi_1_1[_lp*2+1 + 0]));
                    y_hi_1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_1_1_bf16[0];
                }
                if (row_hi_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_1_1_bf16[0];
                }
                float h_raw_1_2[8];
                h_raw_1_2[0] = accum[48] * sa_lo_1 * sw[8];
                h_raw_1_2[2] = accum[50] * sa_hi_1 * sw[8];
                h_raw_1_2[4] = accum[52] * sa_lo_1 * sw[10];
                h_raw_1_2[6] = accum[54] * sa_hi_1 * sw[10];
                h_raw_1_2[1] = accum[49] * sa_lo_1 * sw[9];
                h_raw_1_2[3] = accum[51] * sa_hi_1 * sw[9];
                h_raw_1_2[5] = accum[53] * sa_lo_1 * sw[11];
                h_raw_1_2[7] = accum[55] * sa_hi_1 * sw[11];
                uint32_t h_raw_1_2_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_2[_lp*2 + 0], h_raw_1_2[_lp*2+1 + 0]));
                    h_raw_1_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_2_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_2_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_1_2_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_1_2_bf16[_pair]));
                }
                float y_lo_1_2[2];
                float y_hi_1_2[2];
                float _exp2_24 = approx_exp2((-h_raw_1_2_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_24 = approx_rcp(1.0f + _exp2_24);
                float _exp2_25 = approx_exp2((-h_raw_1_2_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_25 = approx_rcp(1.0f + _exp2_25);
                __nv_bfloat16 _cvt_bf16_24 = __float2bfloat16(h_raw_1_2_bf16_f32[0] * _rcp_24);
                float _cvt_f32_24 = __bfloat162float(_cvt_bf16_24);
                __nv_bfloat16 _cvt_bf16_25 = __float2bfloat16(h_raw_1_2_bf16_f32[2] * _rcp_25);
                float _cvt_f32_25 = __bfloat162float(_cvt_bf16_25);
                y_lo_1_2[0] = _cvt_f32_24 * h_raw_1_2_bf16_f32[4];
                y_hi_1_2[0] = _cvt_f32_25 * h_raw_1_2_bf16_f32[6];
                float _exp2_26 = approx_exp2((-h_raw_1_2_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_26 = approx_rcp(1.0f + _exp2_26);
                float _exp2_27 = approx_exp2((-h_raw_1_2_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_27 = approx_rcp(1.0f + _exp2_27);
                __nv_bfloat16 _cvt_bf16_26 = __float2bfloat16(h_raw_1_2_bf16_f32[1] * _rcp_26);
                float _cvt_f32_26 = __bfloat162float(_cvt_bf16_26);
                __nv_bfloat16 _cvt_bf16_27 = __float2bfloat16(h_raw_1_2_bf16_f32[3] * _rcp_27);
                float _cvt_f32_27 = __bfloat162float(_cvt_bf16_27);
                y_lo_1_2[1] = _cvt_f32_26 * h_raw_1_2_bf16_f32[5];
                y_hi_1_2[1] = _cvt_f32_27 * h_raw_1_2_bf16_f32[7];
                uint32_t y_lo_1_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_2[_lp*2 + 0], y_lo_1_2[_lp*2+1 + 0]));
                    y_lo_1_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_2[_lp*2 + 0], y_hi_1_2[_lp*2+1 + 0]));
                    y_hi_1_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_1_2_bf16[0];
                }
                if (row_hi_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_1_2_bf16[0];
                }
                float h_raw_1_3[8];
                h_raw_1_3[0] = accum[56] * sa_lo_1 * sw[12];
                h_raw_1_3[2] = accum[58] * sa_hi_1 * sw[12];
                h_raw_1_3[4] = accum[60] * sa_lo_1 * sw[14];
                h_raw_1_3[6] = accum[62] * sa_hi_1 * sw[14];
                h_raw_1_3[1] = accum[57] * sa_lo_1 * sw[13];
                h_raw_1_3[3] = accum[59] * sa_hi_1 * sw[13];
                h_raw_1_3[5] = accum[61] * sa_lo_1 * sw[15];
                h_raw_1_3[7] = accum[63] * sa_hi_1 * sw[15];
                uint32_t h_raw_1_3_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_3[_lp*2 + 0], h_raw_1_3[_lp*2+1 + 0]));
                    h_raw_1_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_3_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_3_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_1_3_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_1_3_bf16[_pair]));
                }
                float y_lo_1_3[2];
                float y_hi_1_3[2];
                float _exp2_28 = approx_exp2((-h_raw_1_3_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_28 = approx_rcp(1.0f + _exp2_28);
                float _exp2_29 = approx_exp2((-h_raw_1_3_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_29 = approx_rcp(1.0f + _exp2_29);
                __nv_bfloat16 _cvt_bf16_28 = __float2bfloat16(h_raw_1_3_bf16_f32[0] * _rcp_28);
                float _cvt_f32_28 = __bfloat162float(_cvt_bf16_28);
                __nv_bfloat16 _cvt_bf16_29 = __float2bfloat16(h_raw_1_3_bf16_f32[2] * _rcp_29);
                float _cvt_f32_29 = __bfloat162float(_cvt_bf16_29);
                y_lo_1_3[0] = _cvt_f32_28 * h_raw_1_3_bf16_f32[4];
                y_hi_1_3[0] = _cvt_f32_29 * h_raw_1_3_bf16_f32[6];
                float _exp2_30 = approx_exp2((-h_raw_1_3_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_30 = approx_rcp(1.0f + _exp2_30);
                float _exp2_31 = approx_exp2((-h_raw_1_3_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_31 = approx_rcp(1.0f + _exp2_31);
                __nv_bfloat16 _cvt_bf16_30 = __float2bfloat16(h_raw_1_3_bf16_f32[1] * _rcp_30);
                float _cvt_f32_30 = __bfloat162float(_cvt_bf16_30);
                __nv_bfloat16 _cvt_bf16_31 = __float2bfloat16(h_raw_1_3_bf16_f32[3] * _rcp_31);
                float _cvt_f32_31 = __bfloat162float(_cvt_bf16_31);
                y_lo_1_3[1] = _cvt_f32_30 * h_raw_1_3_bf16_f32[5];
                y_hi_1_3[1] = _cvt_f32_31 * h_raw_1_3_bf16_f32[7];
                uint32_t y_lo_1_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_3[_lp*2 + 0], y_lo_1_3[_lp*2+1 + 0]));
                    y_lo_1_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_3[_lp*2 + 0], y_hi_1_3[_lp*2+1 + 0]));
                    y_hi_1_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_1_3_bf16[0];
                }
                if (row_hi_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_1_3_bf16[0];
                }
                int row_lo_2 = tile_m * 128 + warp_m * 64 + 32 + (lane >> 2);
                int row_hi_2 = row_lo_2 + 8;
                float sa_lo_2 = 1.0f;
                float sa_hi_2 = 1.0f;
                float h_raw_2_0[8];
                h_raw_2_0[0] = accum[64] * sa_lo_2 * sw[0];
                h_raw_2_0[2] = accum[66] * sa_hi_2 * sw[0];
                h_raw_2_0[4] = accum[68] * sa_lo_2 * sw[2];
                h_raw_2_0[6] = accum[70] * sa_hi_2 * sw[2];
                h_raw_2_0[1] = accum[65] * sa_lo_2 * sw[1];
                h_raw_2_0[3] = accum[67] * sa_hi_2 * sw[1];
                h_raw_2_0[5] = accum[69] * sa_lo_2 * sw[3];
                h_raw_2_0[7] = accum[71] * sa_hi_2 * sw[3];
                uint32_t h_raw_2_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_0[_lp*2 + 0], h_raw_2_0[_lp*2+1 + 0]));
                    h_raw_2_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_0_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_0_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_2_0_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_2_0_bf16[_pair]));
                }
                float y_lo_2_0[2];
                float y_hi_2_0[2];
                float _exp2_32 = approx_exp2((-h_raw_2_0_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_32 = approx_rcp(1.0f + _exp2_32);
                float _exp2_33 = approx_exp2((-h_raw_2_0_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_33 = approx_rcp(1.0f + _exp2_33);
                __nv_bfloat16 _cvt_bf16_32 = __float2bfloat16(h_raw_2_0_bf16_f32[0] * _rcp_32);
                float _cvt_f32_32 = __bfloat162float(_cvt_bf16_32);
                __nv_bfloat16 _cvt_bf16_33 = __float2bfloat16(h_raw_2_0_bf16_f32[2] * _rcp_33);
                float _cvt_f32_33 = __bfloat162float(_cvt_bf16_33);
                y_lo_2_0[0] = _cvt_f32_32 * h_raw_2_0_bf16_f32[4];
                y_hi_2_0[0] = _cvt_f32_33 * h_raw_2_0_bf16_f32[6];
                float _exp2_34 = approx_exp2((-h_raw_2_0_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_34 = approx_rcp(1.0f + _exp2_34);
                float _exp2_35 = approx_exp2((-h_raw_2_0_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_35 = approx_rcp(1.0f + _exp2_35);
                __nv_bfloat16 _cvt_bf16_34 = __float2bfloat16(h_raw_2_0_bf16_f32[1] * _rcp_34);
                float _cvt_f32_34 = __bfloat162float(_cvt_bf16_34);
                __nv_bfloat16 _cvt_bf16_35 = __float2bfloat16(h_raw_2_0_bf16_f32[3] * _rcp_35);
                float _cvt_f32_35 = __bfloat162float(_cvt_bf16_35);
                y_lo_2_0[1] = _cvt_f32_34 * h_raw_2_0_bf16_f32[5];
                y_hi_2_0[1] = _cvt_f32_35 * h_raw_2_0_bf16_f32[7];
                uint32_t y_lo_2_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_0[_lp*2 + 0], y_lo_2_0[_lp*2+1 + 0]));
                    y_lo_2_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_0[_lp*2 + 0], y_hi_2_0[_lp*2+1 + 0]));
                    y_hi_2_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_2_0_bf16[0];
                }
                if (row_hi_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_2_0_bf16[0];
                }
                float h_raw_2_1[8];
                h_raw_2_1[0] = accum[72] * sa_lo_2 * sw[4];
                h_raw_2_1[2] = accum[74] * sa_hi_2 * sw[4];
                h_raw_2_1[4] = accum[76] * sa_lo_2 * sw[6];
                h_raw_2_1[6] = accum[78] * sa_hi_2 * sw[6];
                h_raw_2_1[1] = accum[73] * sa_lo_2 * sw[5];
                h_raw_2_1[3] = accum[75] * sa_hi_2 * sw[5];
                h_raw_2_1[5] = accum[77] * sa_lo_2 * sw[7];
                h_raw_2_1[7] = accum[79] * sa_hi_2 * sw[7];
                uint32_t h_raw_2_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_1[_lp*2 + 0], h_raw_2_1[_lp*2+1 + 0]));
                    h_raw_2_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_1_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_1_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_2_1_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_2_1_bf16[_pair]));
                }
                float y_lo_2_1[2];
                float y_hi_2_1[2];
                float _exp2_36 = approx_exp2((-h_raw_2_1_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_36 = approx_rcp(1.0f + _exp2_36);
                float _exp2_37 = approx_exp2((-h_raw_2_1_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_37 = approx_rcp(1.0f + _exp2_37);
                __nv_bfloat16 _cvt_bf16_36 = __float2bfloat16(h_raw_2_1_bf16_f32[0] * _rcp_36);
                float _cvt_f32_36 = __bfloat162float(_cvt_bf16_36);
                __nv_bfloat16 _cvt_bf16_37 = __float2bfloat16(h_raw_2_1_bf16_f32[2] * _rcp_37);
                float _cvt_f32_37 = __bfloat162float(_cvt_bf16_37);
                y_lo_2_1[0] = _cvt_f32_36 * h_raw_2_1_bf16_f32[4];
                y_hi_2_1[0] = _cvt_f32_37 * h_raw_2_1_bf16_f32[6];
                float _exp2_38 = approx_exp2((-h_raw_2_1_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_38 = approx_rcp(1.0f + _exp2_38);
                float _exp2_39 = approx_exp2((-h_raw_2_1_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_39 = approx_rcp(1.0f + _exp2_39);
                __nv_bfloat16 _cvt_bf16_38 = __float2bfloat16(h_raw_2_1_bf16_f32[1] * _rcp_38);
                float _cvt_f32_38 = __bfloat162float(_cvt_bf16_38);
                __nv_bfloat16 _cvt_bf16_39 = __float2bfloat16(h_raw_2_1_bf16_f32[3] * _rcp_39);
                float _cvt_f32_39 = __bfloat162float(_cvt_bf16_39);
                y_lo_2_1[1] = _cvt_f32_38 * h_raw_2_1_bf16_f32[5];
                y_hi_2_1[1] = _cvt_f32_39 * h_raw_2_1_bf16_f32[7];
                uint32_t y_lo_2_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_1[_lp*2 + 0], y_lo_2_1[_lp*2+1 + 0]));
                    y_lo_2_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_1[_lp*2 + 0], y_hi_2_1[_lp*2+1 + 0]));
                    y_hi_2_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_2_1_bf16[0];
                }
                if (row_hi_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_2_1_bf16[0];
                }
                float h_raw_2_2[8];
                h_raw_2_2[0] = accum[80] * sa_lo_2 * sw[8];
                h_raw_2_2[2] = accum[82] * sa_hi_2 * sw[8];
                h_raw_2_2[4] = accum[84] * sa_lo_2 * sw[10];
                h_raw_2_2[6] = accum[86] * sa_hi_2 * sw[10];
                h_raw_2_2[1] = accum[81] * sa_lo_2 * sw[9];
                h_raw_2_2[3] = accum[83] * sa_hi_2 * sw[9];
                h_raw_2_2[5] = accum[85] * sa_lo_2 * sw[11];
                h_raw_2_2[7] = accum[87] * sa_hi_2 * sw[11];
                uint32_t h_raw_2_2_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_2[_lp*2 + 0], h_raw_2_2[_lp*2+1 + 0]));
                    h_raw_2_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_2_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_2_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_2_2_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_2_2_bf16[_pair]));
                }
                float y_lo_2_2[2];
                float y_hi_2_2[2];
                float _exp2_40 = approx_exp2((-h_raw_2_2_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_40 = approx_rcp(1.0f + _exp2_40);
                float _exp2_41 = approx_exp2((-h_raw_2_2_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_41 = approx_rcp(1.0f + _exp2_41);
                __nv_bfloat16 _cvt_bf16_40 = __float2bfloat16(h_raw_2_2_bf16_f32[0] * _rcp_40);
                float _cvt_f32_40 = __bfloat162float(_cvt_bf16_40);
                __nv_bfloat16 _cvt_bf16_41 = __float2bfloat16(h_raw_2_2_bf16_f32[2] * _rcp_41);
                float _cvt_f32_41 = __bfloat162float(_cvt_bf16_41);
                y_lo_2_2[0] = _cvt_f32_40 * h_raw_2_2_bf16_f32[4];
                y_hi_2_2[0] = _cvt_f32_41 * h_raw_2_2_bf16_f32[6];
                float _exp2_42 = approx_exp2((-h_raw_2_2_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_42 = approx_rcp(1.0f + _exp2_42);
                float _exp2_43 = approx_exp2((-h_raw_2_2_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_43 = approx_rcp(1.0f + _exp2_43);
                __nv_bfloat16 _cvt_bf16_42 = __float2bfloat16(h_raw_2_2_bf16_f32[1] * _rcp_42);
                float _cvt_f32_42 = __bfloat162float(_cvt_bf16_42);
                __nv_bfloat16 _cvt_bf16_43 = __float2bfloat16(h_raw_2_2_bf16_f32[3] * _rcp_43);
                float _cvt_f32_43 = __bfloat162float(_cvt_bf16_43);
                y_lo_2_2[1] = _cvt_f32_42 * h_raw_2_2_bf16_f32[5];
                y_hi_2_2[1] = _cvt_f32_43 * h_raw_2_2_bf16_f32[7];
                uint32_t y_lo_2_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_2[_lp*2 + 0], y_lo_2_2[_lp*2+1 + 0]));
                    y_lo_2_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_2[_lp*2 + 0], y_hi_2_2[_lp*2+1 + 0]));
                    y_hi_2_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_2_2_bf16[0];
                }
                if (row_hi_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_2_2_bf16[0];
                }
                float h_raw_2_3[8];
                h_raw_2_3[0] = accum[88] * sa_lo_2 * sw[12];
                h_raw_2_3[2] = accum[90] * sa_hi_2 * sw[12];
                h_raw_2_3[4] = accum[92] * sa_lo_2 * sw[14];
                h_raw_2_3[6] = accum[94] * sa_hi_2 * sw[14];
                h_raw_2_3[1] = accum[89] * sa_lo_2 * sw[13];
                h_raw_2_3[3] = accum[91] * sa_hi_2 * sw[13];
                h_raw_2_3[5] = accum[93] * sa_lo_2 * sw[15];
                h_raw_2_3[7] = accum[95] * sa_hi_2 * sw[15];
                uint32_t h_raw_2_3_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_3[_lp*2 + 0], h_raw_2_3[_lp*2+1 + 0]));
                    h_raw_2_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_3_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_3_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_2_3_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_2_3_bf16[_pair]));
                }
                float y_lo_2_3[2];
                float y_hi_2_3[2];
                float _exp2_44 = approx_exp2((-h_raw_2_3_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_44 = approx_rcp(1.0f + _exp2_44);
                float _exp2_45 = approx_exp2((-h_raw_2_3_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_45 = approx_rcp(1.0f + _exp2_45);
                __nv_bfloat16 _cvt_bf16_44 = __float2bfloat16(h_raw_2_3_bf16_f32[0] * _rcp_44);
                float _cvt_f32_44 = __bfloat162float(_cvt_bf16_44);
                __nv_bfloat16 _cvt_bf16_45 = __float2bfloat16(h_raw_2_3_bf16_f32[2] * _rcp_45);
                float _cvt_f32_45 = __bfloat162float(_cvt_bf16_45);
                y_lo_2_3[0] = _cvt_f32_44 * h_raw_2_3_bf16_f32[4];
                y_hi_2_3[0] = _cvt_f32_45 * h_raw_2_3_bf16_f32[6];
                float _exp2_46 = approx_exp2((-h_raw_2_3_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_46 = approx_rcp(1.0f + _exp2_46);
                float _exp2_47 = approx_exp2((-h_raw_2_3_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_47 = approx_rcp(1.0f + _exp2_47);
                __nv_bfloat16 _cvt_bf16_46 = __float2bfloat16(h_raw_2_3_bf16_f32[1] * _rcp_46);
                float _cvt_f32_46 = __bfloat162float(_cvt_bf16_46);
                __nv_bfloat16 _cvt_bf16_47 = __float2bfloat16(h_raw_2_3_bf16_f32[3] * _rcp_47);
                float _cvt_f32_47 = __bfloat162float(_cvt_bf16_47);
                y_lo_2_3[1] = _cvt_f32_46 * h_raw_2_3_bf16_f32[5];
                y_hi_2_3[1] = _cvt_f32_47 * h_raw_2_3_bf16_f32[7];
                uint32_t y_lo_2_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_3[_lp*2 + 0], y_lo_2_3[_lp*2+1 + 0]));
                    y_lo_2_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_3[_lp*2 + 0], y_hi_2_3[_lp*2+1 + 0]));
                    y_hi_2_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_2_3_bf16[0];
                }
                if (row_hi_2 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_2_3_bf16[0];
                }
                int row_lo_3 = tile_m * 128 + warp_m * 64 + 48 + (lane >> 2);
                int row_hi_3 = row_lo_3 + 8;
                float sa_lo_3 = 1.0f;
                float sa_hi_3 = 1.0f;
                float h_raw_3_0[8];
                h_raw_3_0[0] = accum[96] * sa_lo_3 * sw[0];
                h_raw_3_0[2] = accum[98] * sa_hi_3 * sw[0];
                h_raw_3_0[4] = accum[100] * sa_lo_3 * sw[2];
                h_raw_3_0[6] = accum[102] * sa_hi_3 * sw[2];
                h_raw_3_0[1] = accum[97] * sa_lo_3 * sw[1];
                h_raw_3_0[3] = accum[99] * sa_hi_3 * sw[1];
                h_raw_3_0[5] = accum[101] * sa_lo_3 * sw[3];
                h_raw_3_0[7] = accum[103] * sa_hi_3 * sw[3];
                uint32_t h_raw_3_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_0[_lp*2 + 0], h_raw_3_0[_lp*2+1 + 0]));
                    h_raw_3_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_0_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_0_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_3_0_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_3_0_bf16[_pair]));
                }
                float y_lo_3_0[2];
                float y_hi_3_0[2];
                float _exp2_48 = approx_exp2((-h_raw_3_0_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_48 = approx_rcp(1.0f + _exp2_48);
                float _exp2_49 = approx_exp2((-h_raw_3_0_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_49 = approx_rcp(1.0f + _exp2_49);
                __nv_bfloat16 _cvt_bf16_48 = __float2bfloat16(h_raw_3_0_bf16_f32[0] * _rcp_48);
                float _cvt_f32_48 = __bfloat162float(_cvt_bf16_48);
                __nv_bfloat16 _cvt_bf16_49 = __float2bfloat16(h_raw_3_0_bf16_f32[2] * _rcp_49);
                float _cvt_f32_49 = __bfloat162float(_cvt_bf16_49);
                y_lo_3_0[0] = _cvt_f32_48 * h_raw_3_0_bf16_f32[4];
                y_hi_3_0[0] = _cvt_f32_49 * h_raw_3_0_bf16_f32[6];
                float _exp2_50 = approx_exp2((-h_raw_3_0_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_50 = approx_rcp(1.0f + _exp2_50);
                float _exp2_51 = approx_exp2((-h_raw_3_0_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_51 = approx_rcp(1.0f + _exp2_51);
                __nv_bfloat16 _cvt_bf16_50 = __float2bfloat16(h_raw_3_0_bf16_f32[1] * _rcp_50);
                float _cvt_f32_50 = __bfloat162float(_cvt_bf16_50);
                __nv_bfloat16 _cvt_bf16_51 = __float2bfloat16(h_raw_3_0_bf16_f32[3] * _rcp_51);
                float _cvt_f32_51 = __bfloat162float(_cvt_bf16_51);
                y_lo_3_0[1] = _cvt_f32_50 * h_raw_3_0_bf16_f32[5];
                y_hi_3_0[1] = _cvt_f32_51 * h_raw_3_0_bf16_f32[7];
                uint32_t y_lo_3_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_0[_lp*2 + 0], y_lo_3_0[_lp*2+1 + 0]));
                    y_lo_3_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_0_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_0[_lp*2 + 0], y_hi_3_0[_lp*2+1 + 0]));
                    y_hi_3_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_3_0_bf16[0];
                }
                if (row_hi_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_3_0_bf16[0];
                }
                float h_raw_3_1[8];
                h_raw_3_1[0] = accum[104] * sa_lo_3 * sw[4];
                h_raw_3_1[2] = accum[106] * sa_hi_3 * sw[4];
                h_raw_3_1[4] = accum[108] * sa_lo_3 * sw[6];
                h_raw_3_1[6] = accum[110] * sa_hi_3 * sw[6];
                h_raw_3_1[1] = accum[105] * sa_lo_3 * sw[5];
                h_raw_3_1[3] = accum[107] * sa_hi_3 * sw[5];
                h_raw_3_1[5] = accum[109] * sa_lo_3 * sw[7];
                h_raw_3_1[7] = accum[111] * sa_hi_3 * sw[7];
                uint32_t h_raw_3_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_1[_lp*2 + 0], h_raw_3_1[_lp*2+1 + 0]));
                    h_raw_3_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_1_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_1_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_3_1_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_3_1_bf16[_pair]));
                }
                float y_lo_3_1[2];
                float y_hi_3_1[2];
                float _exp2_52 = approx_exp2((-h_raw_3_1_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_52 = approx_rcp(1.0f + _exp2_52);
                float _exp2_53 = approx_exp2((-h_raw_3_1_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_53 = approx_rcp(1.0f + _exp2_53);
                __nv_bfloat16 _cvt_bf16_52 = __float2bfloat16(h_raw_3_1_bf16_f32[0] * _rcp_52);
                float _cvt_f32_52 = __bfloat162float(_cvt_bf16_52);
                __nv_bfloat16 _cvt_bf16_53 = __float2bfloat16(h_raw_3_1_bf16_f32[2] * _rcp_53);
                float _cvt_f32_53 = __bfloat162float(_cvt_bf16_53);
                y_lo_3_1[0] = _cvt_f32_52 * h_raw_3_1_bf16_f32[4];
                y_hi_3_1[0] = _cvt_f32_53 * h_raw_3_1_bf16_f32[6];
                float _exp2_54 = approx_exp2((-h_raw_3_1_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_54 = approx_rcp(1.0f + _exp2_54);
                float _exp2_55 = approx_exp2((-h_raw_3_1_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_55 = approx_rcp(1.0f + _exp2_55);
                __nv_bfloat16 _cvt_bf16_54 = __float2bfloat16(h_raw_3_1_bf16_f32[1] * _rcp_54);
                float _cvt_f32_54 = __bfloat162float(_cvt_bf16_54);
                __nv_bfloat16 _cvt_bf16_55 = __float2bfloat16(h_raw_3_1_bf16_f32[3] * _rcp_55);
                float _cvt_f32_55 = __bfloat162float(_cvt_bf16_55);
                y_lo_3_1[1] = _cvt_f32_54 * h_raw_3_1_bf16_f32[5];
                y_hi_3_1[1] = _cvt_f32_55 * h_raw_3_1_bf16_f32[7];
                uint32_t y_lo_3_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_1[_lp*2 + 0], y_lo_3_1[_lp*2+1 + 0]));
                    y_lo_3_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_1_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_1[_lp*2 + 0], y_hi_3_1[_lp*2+1 + 0]));
                    y_hi_3_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_3_1_bf16[0];
                }
                if (row_hi_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_3_1_bf16[0];
                }
                float h_raw_3_2[8];
                h_raw_3_2[0] = accum[112] * sa_lo_3 * sw[8];
                h_raw_3_2[2] = accum[114] * sa_hi_3 * sw[8];
                h_raw_3_2[4] = accum[116] * sa_lo_3 * sw[10];
                h_raw_3_2[6] = accum[118] * sa_hi_3 * sw[10];
                h_raw_3_2[1] = accum[113] * sa_lo_3 * sw[9];
                h_raw_3_2[3] = accum[115] * sa_hi_3 * sw[9];
                h_raw_3_2[5] = accum[117] * sa_lo_3 * sw[11];
                h_raw_3_2[7] = accum[119] * sa_hi_3 * sw[11];
                uint32_t h_raw_3_2_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_2[_lp*2 + 0], h_raw_3_2[_lp*2+1 + 0]));
                    h_raw_3_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_2_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_2_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_3_2_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_3_2_bf16[_pair]));
                }
                float y_lo_3_2[2];
                float y_hi_3_2[2];
                float _exp2_56 = approx_exp2((-h_raw_3_2_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_56 = approx_rcp(1.0f + _exp2_56);
                float _exp2_57 = approx_exp2((-h_raw_3_2_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_57 = approx_rcp(1.0f + _exp2_57);
                __nv_bfloat16 _cvt_bf16_56 = __float2bfloat16(h_raw_3_2_bf16_f32[0] * _rcp_56);
                float _cvt_f32_56 = __bfloat162float(_cvt_bf16_56);
                __nv_bfloat16 _cvt_bf16_57 = __float2bfloat16(h_raw_3_2_bf16_f32[2] * _rcp_57);
                float _cvt_f32_57 = __bfloat162float(_cvt_bf16_57);
                y_lo_3_2[0] = _cvt_f32_56 * h_raw_3_2_bf16_f32[4];
                y_hi_3_2[0] = _cvt_f32_57 * h_raw_3_2_bf16_f32[6];
                float _exp2_58 = approx_exp2((-h_raw_3_2_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_58 = approx_rcp(1.0f + _exp2_58);
                float _exp2_59 = approx_exp2((-h_raw_3_2_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_59 = approx_rcp(1.0f + _exp2_59);
                __nv_bfloat16 _cvt_bf16_58 = __float2bfloat16(h_raw_3_2_bf16_f32[1] * _rcp_58);
                float _cvt_f32_58 = __bfloat162float(_cvt_bf16_58);
                __nv_bfloat16 _cvt_bf16_59 = __float2bfloat16(h_raw_3_2_bf16_f32[3] * _rcp_59);
                float _cvt_f32_59 = __bfloat162float(_cvt_bf16_59);
                y_lo_3_2[1] = _cvt_f32_58 * h_raw_3_2_bf16_f32[5];
                y_hi_3_2[1] = _cvt_f32_59 * h_raw_3_2_bf16_f32[7];
                uint32_t y_lo_3_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_2[_lp*2 + 0], y_lo_3_2[_lp*2+1 + 0]));
                    y_lo_3_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_2_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_2[_lp*2 + 0], y_hi_3_2[_lp*2+1 + 0]));
                    y_hi_3_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_3_2_bf16[0];
                }
                if (row_hi_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_3_2_bf16[0];
                }
                float h_raw_3_3[8];
                h_raw_3_3[0] = accum[120] * sa_lo_3 * sw[12];
                h_raw_3_3[2] = accum[122] * sa_hi_3 * sw[12];
                h_raw_3_3[4] = accum[124] * sa_lo_3 * sw[14];
                h_raw_3_3[6] = accum[126] * sa_hi_3 * sw[14];
                h_raw_3_3[1] = accum[121] * sa_lo_3 * sw[13];
                h_raw_3_3[3] = accum[123] * sa_hi_3 * sw[13];
                h_raw_3_3[5] = accum[125] * sa_lo_3 * sw[15];
                h_raw_3_3[7] = accum[127] * sa_hi_3 * sw[15];
                uint32_t h_raw_3_3_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_3[_lp*2 + 0], h_raw_3_3[_lp*2+1 + 0]));
                    h_raw_3_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_3_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_3_bf16_f32[_pair * 2])[0]), "=f"((&h_raw_3_3_bf16_f32[_pair * 2])[1])
                        : "r"(h_raw_3_3_bf16[_pair]));
                }
                float y_lo_3_3[2];
                float y_hi_3_3[2];
                float _exp2_60 = approx_exp2((-h_raw_3_3_bf16_f32[0]) * 1.4426950408889634f);
                float _rcp_60 = approx_rcp(1.0f + _exp2_60);
                float _exp2_61 = approx_exp2((-h_raw_3_3_bf16_f32[2]) * 1.4426950408889634f);
                float _rcp_61 = approx_rcp(1.0f + _exp2_61);
                __nv_bfloat16 _cvt_bf16_60 = __float2bfloat16(h_raw_3_3_bf16_f32[0] * _rcp_60);
                float _cvt_f32_60 = __bfloat162float(_cvt_bf16_60);
                __nv_bfloat16 _cvt_bf16_61 = __float2bfloat16(h_raw_3_3_bf16_f32[2] * _rcp_61);
                float _cvt_f32_61 = __bfloat162float(_cvt_bf16_61);
                y_lo_3_3[0] = _cvt_f32_60 * h_raw_3_3_bf16_f32[4];
                y_hi_3_3[0] = _cvt_f32_61 * h_raw_3_3_bf16_f32[6];
                float _exp2_62 = approx_exp2((-h_raw_3_3_bf16_f32[1]) * 1.4426950408889634f);
                float _rcp_62 = approx_rcp(1.0f + _exp2_62);
                float _exp2_63 = approx_exp2((-h_raw_3_3_bf16_f32[3]) * 1.4426950408889634f);
                float _rcp_63 = approx_rcp(1.0f + _exp2_63);
                __nv_bfloat16 _cvt_bf16_62 = __float2bfloat16(h_raw_3_3_bf16_f32[1] * _rcp_62);
                float _cvt_f32_62 = __bfloat162float(_cvt_bf16_62);
                __nv_bfloat16 _cvt_bf16_63 = __float2bfloat16(h_raw_3_3_bf16_f32[3] * _rcp_63);
                float _cvt_f32_63 = __bfloat162float(_cvt_bf16_63);
                y_lo_3_3[1] = _cvt_f32_62 * h_raw_3_3_bf16_f32[5];
                y_hi_3_3[1] = _cvt_f32_63 * h_raw_3_3_bf16_f32[7];
                uint32_t y_lo_3_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_3[_lp*2 + 0], y_lo_3_3[_lp*2+1 + 0]));
                    y_lo_3_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_3_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_3[_lp*2 + 0], y_hi_3_3[_lp*2+1 + 0]));
                    y_hi_3_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_3_3_bf16[0];
                }
                if (row_hi_3 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3 * 14336 + (tile_n * 128 + warp_n * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_3_3_bf16[0];
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 1 && warp <= 7) {
        { // mma_main
            unsigned int mma_stage_1 = 0;
            unsigned int mma_kk_1 = 0;
            int warp_m_1 = warp / 4;
            int warp_n_1 = warp % 4;
            float accum_1[128];
            unsigned int a_frag_1[16];
            unsigned int b_frag_1[16];
            unsigned int _phase_ab_full_1 = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 112) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 112) * (GROUP_M * 112)) % ((num_m_tiles - tile_1 / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 112) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 112) * (GROUP_M * 112)) / ((num_m_tiles - tile_1 / (GROUP_M * 112) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 112) * GROUP_M : GROUP_M);
                accum_1[0] = 0.0f;
                accum_1[1] = 0.0f;
                accum_1[2] = 0.0f;
                accum_1[3] = 0.0f;
                accum_1[4] = 0.0f;
                accum_1[5] = 0.0f;
                accum_1[6] = 0.0f;
                accum_1[7] = 0.0f;
                accum_1[8] = 0.0f;
                accum_1[9] = 0.0f;
                accum_1[10] = 0.0f;
                accum_1[11] = 0.0f;
                accum_1[12] = 0.0f;
                accum_1[13] = 0.0f;
                accum_1[14] = 0.0f;
                accum_1[15] = 0.0f;
                accum_1[16] = 0.0f;
                accum_1[17] = 0.0f;
                accum_1[18] = 0.0f;
                accum_1[19] = 0.0f;
                accum_1[20] = 0.0f;
                accum_1[21] = 0.0f;
                accum_1[22] = 0.0f;
                accum_1[23] = 0.0f;
                accum_1[24] = 0.0f;
                accum_1[25] = 0.0f;
                accum_1[26] = 0.0f;
                accum_1[27] = 0.0f;
                accum_1[28] = 0.0f;
                accum_1[29] = 0.0f;
                accum_1[30] = 0.0f;
                accum_1[31] = 0.0f;
                accum_1[32] = 0.0f;
                accum_1[33] = 0.0f;
                accum_1[34] = 0.0f;
                accum_1[35] = 0.0f;
                accum_1[36] = 0.0f;
                accum_1[37] = 0.0f;
                accum_1[38] = 0.0f;
                accum_1[39] = 0.0f;
                accum_1[40] = 0.0f;
                accum_1[41] = 0.0f;
                accum_1[42] = 0.0f;
                accum_1[43] = 0.0f;
                accum_1[44] = 0.0f;
                accum_1[45] = 0.0f;
                accum_1[46] = 0.0f;
                accum_1[47] = 0.0f;
                accum_1[48] = 0.0f;
                accum_1[49] = 0.0f;
                accum_1[50] = 0.0f;
                accum_1[51] = 0.0f;
                accum_1[52] = 0.0f;
                accum_1[53] = 0.0f;
                accum_1[54] = 0.0f;
                accum_1[55] = 0.0f;
                accum_1[56] = 0.0f;
                accum_1[57] = 0.0f;
                accum_1[58] = 0.0f;
                accum_1[59] = 0.0f;
                accum_1[60] = 0.0f;
                accum_1[61] = 0.0f;
                accum_1[62] = 0.0f;
                accum_1[63] = 0.0f;
                accum_1[64] = 0.0f;
                accum_1[65] = 0.0f;
                accum_1[66] = 0.0f;
                accum_1[67] = 0.0f;
                accum_1[68] = 0.0f;
                accum_1[69] = 0.0f;
                accum_1[70] = 0.0f;
                accum_1[71] = 0.0f;
                accum_1[72] = 0.0f;
                accum_1[73] = 0.0f;
                accum_1[74] = 0.0f;
                accum_1[75] = 0.0f;
                accum_1[76] = 0.0f;
                accum_1[77] = 0.0f;
                accum_1[78] = 0.0f;
                accum_1[79] = 0.0f;
                accum_1[80] = 0.0f;
                accum_1[81] = 0.0f;
                accum_1[82] = 0.0f;
                accum_1[83] = 0.0f;
                accum_1[84] = 0.0f;
                accum_1[85] = 0.0f;
                accum_1[86] = 0.0f;
                accum_1[87] = 0.0f;
                accum_1[88] = 0.0f;
                accum_1[89] = 0.0f;
                accum_1[90] = 0.0f;
                accum_1[91] = 0.0f;
                accum_1[92] = 0.0f;
                accum_1[93] = 0.0f;
                accum_1[94] = 0.0f;
                accum_1[95] = 0.0f;
                accum_1[96] = 0.0f;
                accum_1[97] = 0.0f;
                accum_1[98] = 0.0f;
                accum_1[99] = 0.0f;
                accum_1[100] = 0.0f;
                accum_1[101] = 0.0f;
                accum_1[102] = 0.0f;
                accum_1[103] = 0.0f;
                accum_1[104] = 0.0f;
                accum_1[105] = 0.0f;
                accum_1[106] = 0.0f;
                accum_1[107] = 0.0f;
                accum_1[108] = 0.0f;
                accum_1[109] = 0.0f;
                accum_1[110] = 0.0f;
                accum_1[111] = 0.0f;
                accum_1[112] = 0.0f;
                accum_1[113] = 0.0f;
                accum_1[114] = 0.0f;
                accum_1[115] = 0.0f;
                accum_1[116] = 0.0f;
                accum_1[117] = 0.0f;
                accum_1[118] = 0.0f;
                accum_1[119] = 0.0f;
                accum_1[120] = 0.0f;
                accum_1[121] = 0.0f;
                accum_1[122] = 0.0f;
                accum_1[123] = 0.0f;
                accum_1[124] = 0.0f;
                accum_1[125] = 0.0f;
                accum_1[126] = 0.0f;
                accum_1[127] = 0.0f;
                #pragma unroll 1
                for (int k_tile_1 = 0; k_tile_1 < 42; k_tile_1++) {
                    mbarrier_wait(ab_full_addr + (mma_stage_1) * 8, _phase_ab_full_1);
                    for (int k_step_1 = 0; k_step_1 < 2; k_step_1++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(A_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_m_1 * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[4]), "=r"(a_frag_1[5]), "=r"(a_frag_1[6]), "=r"(a_frag_1[7])
                            : "r"(A_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_m_1 * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[8]), "=r"(a_frag_1[9]), "=r"(a_frag_1[10]), "=r"(a_frag_1[11])
                            : "r"(A_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_m_1 * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[12]), "=r"(a_frag_1[13]), "=r"(a_frag_1[14]), "=r"(a_frag_1[15])
                            : "r"(A_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_m_1 * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1]), "=r"(b_frag_1[2]), "=r"(b_frag_1[3])
                            : "r"(B_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_n_1 * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[4]), "=r"(b_frag_1[5]), "=r"(b_frag_1[6]), "=r"(b_frag_1[7])
                            : "r"(B_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_n_1 * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[8]), "=r"(b_frag_1[9]), "=r"(b_frag_1[10]), "=r"(b_frag_1[11])
                            : "r"(B_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_n_1 * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[12]), "=r"(b_frag_1[13]), "=r"(b_frag_1[14]), "=r"(b_frag_1[15])
                            : "r"(B_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_n_1 * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        unsigned int _SFA_pairs_reg_4[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_pairs);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFA_pairs_reg_4[_lr] = _smem_ptr[((mma_kk_1 >> 1 & 1) * 512 + (mma_kk_1 & 1) * 2 + (unsigned int)k_step_1 + (unsigned int)((warp_m_1 * 64 + (lane & 1) * 8 + (lane >> 2)) * 4)) + _lr];
                        }
                        unsigned int _SFA_pairs_reg_5[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_pairs);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFA_pairs_reg_5[_lr] = _smem_ptr[((mma_kk_1 >> 1 & 1) * 512 + (mma_kk_1 & 1) * 2 + (unsigned int)k_step_1 + (unsigned int)((warp_m_1 * 64 + 16 + (lane & 1) * 8 + (lane >> 2)) * 4)) + _lr];
                        }
                        unsigned int _SFA_pairs_reg_6[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_pairs);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFA_pairs_reg_6[_lr] = _smem_ptr[((mma_kk_1 >> 1 & 1) * 512 + (mma_kk_1 & 1) * 2 + (unsigned int)k_step_1 + (unsigned int)((warp_m_1 * 64 + 32 + (lane & 1) * 8 + (lane >> 2)) * 4)) + _lr];
                        }
                        unsigned int _SFA_pairs_reg_7[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_pairs);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFA_pairs_reg_7[_lr] = _smem_ptr[((mma_kk_1 >> 1 & 1) * 512 + (mma_kk_1 & 1) * 2 + (unsigned int)k_step_1 + (unsigned int)((warp_m_1 * 64 + 48 + (lane & 1) * 8 + (lane >> 2)) * 4)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_8[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_8[_lr] = _smem_ptr[(mma_stage_1 * 512 + (unsigned int)(((warp_n_1 >> 1) * 2 + k_step_1) * 128) + (unsigned int)((lane >> 2) * 4) + (unsigned int)((warp_n_1 & 1) * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_9[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_9[_lr] = _smem_ptr[(mma_stage_1 * 512 + (unsigned int)(((warp_n_1 >> 1) * 2 + k_step_1) * 128) + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)((warp_n_1 & 1) * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_10[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_10[_lr] = _smem_ptr[(mma_stage_1 * 512 + (unsigned int)(((warp_n_1 >> 1) * 2 + k_step_1) * 128) + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)((warp_n_1 & 1) * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_11[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_11[_lr] = _smem_ptr[(mma_stage_1 * 512 + (unsigned int)(((warp_n_1 >> 1) * 2 + k_step_1) * 128) + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)((warp_n_1 & 1) * 2)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_12[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_12[_lr] = _smem_ptr[(mma_stage_1 * 512 + (unsigned int)(((warp_n_1 >> 1) * 2 + k_step_1) * 128) + (unsigned int)((lane >> 2) * 4) + (unsigned int)((warp_n_1 & 1) * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_13[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_13[_lr] = _smem_ptr[(mma_stage_1 * 512 + (unsigned int)(((warp_n_1 >> 1) * 2 + k_step_1) * 128) + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)((warp_n_1 & 1) * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_14[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_14[_lr] = _smem_ptr[(mma_stage_1 * 512 + (unsigned int)(((warp_n_1 >> 1) * 2 + k_step_1) * 128) + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)((warp_n_1 & 1) * 2 + 1)) + _lr];
                        }
                        unsigned int _SFB_stage_reg_15[1];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 1; _lr++)
                                _SFB_stage_reg_15[_lr] = _smem_ptr[(mma_stage_1 * 512 + (unsigned int)(((warp_n_1 >> 1) * 2 + k_step_1) * 128) + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)((warp_n_1 & 1) * 2 + 1)) + _lr];
                        }
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[0]), "+f"(accum_1[1]), "+f"(accum_1[2]), "+f"(accum_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(_SFA_pairs_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[4]), "+f"(accum_1[(4) + 1]), "+f"(accum_1[(4) + 2]), "+f"(accum_1[(4) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]), "r"((uint32_t)(_SFA_pairs_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[8]), "+f"(accum_1[(8) + 1]), "+f"(accum_1[(8) + 2]), "+f"(accum_1[(8) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]), "r"((uint32_t)(_SFA_pairs_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[12]), "+f"(accum_1[(12) + 1]), "+f"(accum_1[(12) + 2]), "+f"(accum_1[(12) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]), "r"((uint32_t)(_SFA_pairs_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[16]), "+f"(accum_1[(16) + 1]), "+f"(accum_1[(16) + 2]), "+f"(accum_1[(16) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]), "r"((uint32_t)(_SFA_pairs_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[20]), "+f"(accum_1[(20) + 1]), "+f"(accum_1[(20) + 2]), "+f"(accum_1[(20) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]), "r"((uint32_t)(_SFA_pairs_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[24]), "+f"(accum_1[(24) + 1]), "+f"(accum_1[(24) + 2]), "+f"(accum_1[(24) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]), "r"((uint32_t)(_SFA_pairs_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[28]), "+f"(accum_1[(28) + 1]), "+f"(accum_1[(28) + 2]), "+f"(accum_1[(28) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]), "r"((uint32_t)(_SFA_pairs_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[32]), "+f"(accum_1[(32) + 1]), "+f"(accum_1[(32) + 2]), "+f"(accum_1[(32) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(_SFA_pairs_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[36]), "+f"(accum_1[(36) + 1]), "+f"(accum_1[(36) + 2]), "+f"(accum_1[(36) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]), "r"((uint32_t)(_SFA_pairs_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[40]), "+f"(accum_1[(40) + 1]), "+f"(accum_1[(40) + 2]), "+f"(accum_1[(40) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]), "r"((uint32_t)(_SFA_pairs_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[44]), "+f"(accum_1[(44) + 1]), "+f"(accum_1[(44) + 2]), "+f"(accum_1[(44) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]), "r"((uint32_t)(_SFA_pairs_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[48]), "+f"(accum_1[(48) + 1]), "+f"(accum_1[(48) + 2]), "+f"(accum_1[(48) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]), "r"((uint32_t)(_SFA_pairs_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[52]), "+f"(accum_1[(52) + 1]), "+f"(accum_1[(52) + 2]), "+f"(accum_1[(52) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]), "r"((uint32_t)(_SFA_pairs_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[56]), "+f"(accum_1[(56) + 1]), "+f"(accum_1[(56) + 2]), "+f"(accum_1[(56) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]), "r"((uint32_t)(_SFA_pairs_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[60]), "+f"(accum_1[(60) + 1]), "+f"(accum_1[(60) + 2]), "+f"(accum_1[(60) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]), "r"((uint32_t)(_SFA_pairs_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[64]), "+f"(accum_1[(64) + 1]), "+f"(accum_1[(64) + 2]), "+f"(accum_1[(64) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(_SFA_pairs_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[68]), "+f"(accum_1[(68) + 1]), "+f"(accum_1[(68) + 2]), "+f"(accum_1[(68) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]), "r"((uint32_t)(_SFA_pairs_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[72]), "+f"(accum_1[(72) + 1]), "+f"(accum_1[(72) + 2]), "+f"(accum_1[(72) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]), "r"((uint32_t)(_SFA_pairs_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[76]), "+f"(accum_1[(76) + 1]), "+f"(accum_1[(76) + 2]), "+f"(accum_1[(76) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]), "r"((uint32_t)(_SFA_pairs_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[80]), "+f"(accum_1[(80) + 1]), "+f"(accum_1[(80) + 2]), "+f"(accum_1[(80) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]), "r"((uint32_t)(_SFA_pairs_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[84]), "+f"(accum_1[(84) + 1]), "+f"(accum_1[(84) + 2]), "+f"(accum_1[(84) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]), "r"((uint32_t)(_SFA_pairs_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[88]), "+f"(accum_1[(88) + 1]), "+f"(accum_1[(88) + 2]), "+f"(accum_1[(88) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]), "r"((uint32_t)(_SFA_pairs_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[92]), "+f"(accum_1[(92) + 1]), "+f"(accum_1[(92) + 2]), "+f"(accum_1[(92) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]), "r"((uint32_t)(_SFA_pairs_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[96]), "+f"(accum_1[(96) + 1]), "+f"(accum_1[(96) + 2]), "+f"(accum_1[(96) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(_SFA_pairs_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[100]), "+f"(accum_1[(100) + 1]), "+f"(accum_1[(100) + 2]), "+f"(accum_1[(100) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]), "r"((uint32_t)(_SFA_pairs_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[104]), "+f"(accum_1[(104) + 1]), "+f"(accum_1[(104) + 2]), "+f"(accum_1[(104) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]), "r"((uint32_t)(_SFA_pairs_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[108]), "+f"(accum_1[(108) + 1]), "+f"(accum_1[(108) + 2]), "+f"(accum_1[(108) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]), "r"((uint32_t)(_SFA_pairs_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[112]), "+f"(accum_1[(112) + 1]), "+f"(accum_1[(112) + 2]), "+f"(accum_1[(112) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]), "r"((uint32_t)(_SFA_pairs_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[116]), "+f"(accum_1[(116) + 1]), "+f"(accum_1[(116) + 2]), "+f"(accum_1[(116) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]), "r"((uint32_t)(_SFA_pairs_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[120]), "+f"(accum_1[(120) + 1]), "+f"(accum_1[(120) + 2]), "+f"(accum_1[(120) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]), "r"((uint32_t)(_SFA_pairs_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[124]), "+f"(accum_1[(124) + 1]), "+f"(accum_1[(124) + 2]), "+f"(accum_1[(124) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]), "r"((uint32_t)(_SFA_pairs_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    }
                    __syncwarp();
                    if (lane == 0) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage_1) * 8);
                    }
                    mma_stage_1 += 1;
                    if (mma_stage_1 == 3) { mma_stage_1 = 0; _phase_ab_full_1 ^= 1; }
                    mma_kk_1 = mma_kk_1 + 1;
                }
                float sw_1[16];
                sw_1[0] = alpha;
                sw_1[2] = alpha;
                sw_1[1] = alpha;
                sw_1[3] = alpha;
                sw_1[4] = alpha;
                sw_1[6] = alpha;
                sw_1[5] = alpha;
                sw_1[7] = alpha;
                sw_1[8] = alpha;
                sw_1[10] = alpha;
                sw_1[9] = alpha;
                sw_1[11] = alpha;
                sw_1[12] = alpha;
                sw_1[14] = alpha;
                sw_1[13] = alpha;
                sw_1[15] = alpha;
                int row_lo_0_1 = tile_m_1 * 128 + warp_m_1 * 64 + (lane >> 2);
                int row_hi_0_1 = row_lo_0_1 + 8;
                float sa_lo_0_1 = 1.0f;
                float sa_hi_0_1 = 1.0f;
                float h_raw_0_0_1[8];
                h_raw_0_0_1[0] = accum_1[0] * sa_lo_0_1 * sw_1[0];
                h_raw_0_0_1[2] = accum_1[2] * sa_hi_0_1 * sw_1[0];
                h_raw_0_0_1[4] = accum_1[4] * sa_lo_0_1 * sw_1[2];
                h_raw_0_0_1[6] = accum_1[6] * sa_hi_0_1 * sw_1[2];
                h_raw_0_0_1[1] = accum_1[1] * sa_lo_0_1 * sw_1[1];
                h_raw_0_0_1[3] = accum_1[3] * sa_hi_0_1 * sw_1[1];
                h_raw_0_0_1[5] = accum_1[5] * sa_lo_0_1 * sw_1[3];
                h_raw_0_0_1[7] = accum_1[7] * sa_hi_0_1 * sw_1[3];
                uint32_t h_raw_0_0_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_0_1[_lp*2 + 0], h_raw_0_0_1[_lp*2+1 + 0]));
                    h_raw_0_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_0_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_0_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_0_0_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_0_0_bf16_1[_pair]));
                }
                float y_lo_0_0_1[2];
                float y_hi_0_0_1[2];
                float _exp2_64 = approx_exp2((-h_raw_0_0_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_64 = approx_rcp(1.0f + _exp2_64);
                float _exp2_65 = approx_exp2((-h_raw_0_0_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_65 = approx_rcp(1.0f + _exp2_65);
                __nv_bfloat16 _cvt_bf16_64 = __float2bfloat16(h_raw_0_0_bf16_f32_1[0] * _rcp_64);
                float _cvt_f32_64 = __bfloat162float(_cvt_bf16_64);
                __nv_bfloat16 _cvt_bf16_65 = __float2bfloat16(h_raw_0_0_bf16_f32_1[2] * _rcp_65);
                float _cvt_f32_65 = __bfloat162float(_cvt_bf16_65);
                y_lo_0_0_1[0] = _cvt_f32_64 * h_raw_0_0_bf16_f32_1[4];
                y_hi_0_0_1[0] = _cvt_f32_65 * h_raw_0_0_bf16_f32_1[6];
                float _exp2_66 = approx_exp2((-h_raw_0_0_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_66 = approx_rcp(1.0f + _exp2_66);
                float _exp2_67 = approx_exp2((-h_raw_0_0_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_67 = approx_rcp(1.0f + _exp2_67);
                __nv_bfloat16 _cvt_bf16_66 = __float2bfloat16(h_raw_0_0_bf16_f32_1[1] * _rcp_66);
                float _cvt_f32_66 = __bfloat162float(_cvt_bf16_66);
                __nv_bfloat16 _cvt_bf16_67 = __float2bfloat16(h_raw_0_0_bf16_f32_1[3] * _rcp_67);
                float _cvt_f32_67 = __bfloat162float(_cvt_bf16_67);
                y_lo_0_0_1[1] = _cvt_f32_66 * h_raw_0_0_bf16_f32_1[5];
                y_hi_0_0_1[1] = _cvt_f32_67 * h_raw_0_0_bf16_f32_1[7];
                uint32_t y_lo_0_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_0_1[_lp*2 + 0], y_lo_0_0_1[_lp*2+1 + 0]));
                    y_lo_0_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_0_1[_lp*2 + 0], y_hi_0_0_1[_lp*2+1 + 0]));
                    y_hi_0_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_0_0_bf16_1[0];
                }
                if (row_hi_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_0_0_bf16_1[0];
                }
                float h_raw_0_1_1[8];
                h_raw_0_1_1[0] = accum_1[8] * sa_lo_0_1 * sw_1[4];
                h_raw_0_1_1[2] = accum_1[10] * sa_hi_0_1 * sw_1[4];
                h_raw_0_1_1[4] = accum_1[12] * sa_lo_0_1 * sw_1[6];
                h_raw_0_1_1[6] = accum_1[14] * sa_hi_0_1 * sw_1[6];
                h_raw_0_1_1[1] = accum_1[9] * sa_lo_0_1 * sw_1[5];
                h_raw_0_1_1[3] = accum_1[11] * sa_hi_0_1 * sw_1[5];
                h_raw_0_1_1[5] = accum_1[13] * sa_lo_0_1 * sw_1[7];
                h_raw_0_1_1[7] = accum_1[15] * sa_hi_0_1 * sw_1[7];
                uint32_t h_raw_0_1_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_1_1[_lp*2 + 0], h_raw_0_1_1[_lp*2+1 + 0]));
                    h_raw_0_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_1_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_1_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_0_1_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_0_1_bf16_1[_pair]));
                }
                float y_lo_0_1_1[2];
                float y_hi_0_1_1[2];
                float _exp2_68 = approx_exp2((-h_raw_0_1_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_68 = approx_rcp(1.0f + _exp2_68);
                float _exp2_69 = approx_exp2((-h_raw_0_1_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_69 = approx_rcp(1.0f + _exp2_69);
                __nv_bfloat16 _cvt_bf16_68 = __float2bfloat16(h_raw_0_1_bf16_f32_1[0] * _rcp_68);
                float _cvt_f32_68 = __bfloat162float(_cvt_bf16_68);
                __nv_bfloat16 _cvt_bf16_69 = __float2bfloat16(h_raw_0_1_bf16_f32_1[2] * _rcp_69);
                float _cvt_f32_69 = __bfloat162float(_cvt_bf16_69);
                y_lo_0_1_1[0] = _cvt_f32_68 * h_raw_0_1_bf16_f32_1[4];
                y_hi_0_1_1[0] = _cvt_f32_69 * h_raw_0_1_bf16_f32_1[6];
                float _exp2_70 = approx_exp2((-h_raw_0_1_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_70 = approx_rcp(1.0f + _exp2_70);
                float _exp2_71 = approx_exp2((-h_raw_0_1_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_71 = approx_rcp(1.0f + _exp2_71);
                __nv_bfloat16 _cvt_bf16_70 = __float2bfloat16(h_raw_0_1_bf16_f32_1[1] * _rcp_70);
                float _cvt_f32_70 = __bfloat162float(_cvt_bf16_70);
                __nv_bfloat16 _cvt_bf16_71 = __float2bfloat16(h_raw_0_1_bf16_f32_1[3] * _rcp_71);
                float _cvt_f32_71 = __bfloat162float(_cvt_bf16_71);
                y_lo_0_1_1[1] = _cvt_f32_70 * h_raw_0_1_bf16_f32_1[5];
                y_hi_0_1_1[1] = _cvt_f32_71 * h_raw_0_1_bf16_f32_1[7];
                uint32_t y_lo_0_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_1_1[_lp*2 + 0], y_lo_0_1_1[_lp*2+1 + 0]));
                    y_lo_0_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_1_1[_lp*2 + 0], y_hi_0_1_1[_lp*2+1 + 0]));
                    y_hi_0_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_0_1_bf16_1[0];
                }
                if (row_hi_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_0_1_bf16_1[0];
                }
                float h_raw_0_2_1[8];
                h_raw_0_2_1[0] = accum_1[16] * sa_lo_0_1 * sw_1[8];
                h_raw_0_2_1[2] = accum_1[18] * sa_hi_0_1 * sw_1[8];
                h_raw_0_2_1[4] = accum_1[20] * sa_lo_0_1 * sw_1[10];
                h_raw_0_2_1[6] = accum_1[22] * sa_hi_0_1 * sw_1[10];
                h_raw_0_2_1[1] = accum_1[17] * sa_lo_0_1 * sw_1[9];
                h_raw_0_2_1[3] = accum_1[19] * sa_hi_0_1 * sw_1[9];
                h_raw_0_2_1[5] = accum_1[21] * sa_lo_0_1 * sw_1[11];
                h_raw_0_2_1[7] = accum_1[23] * sa_hi_0_1 * sw_1[11];
                uint32_t h_raw_0_2_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_2_1[_lp*2 + 0], h_raw_0_2_1[_lp*2+1 + 0]));
                    h_raw_0_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_2_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_2_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_0_2_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_0_2_bf16_1[_pair]));
                }
                float y_lo_0_2_1[2];
                float y_hi_0_2_1[2];
                float _exp2_72 = approx_exp2((-h_raw_0_2_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_72 = approx_rcp(1.0f + _exp2_72);
                float _exp2_73 = approx_exp2((-h_raw_0_2_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_73 = approx_rcp(1.0f + _exp2_73);
                __nv_bfloat16 _cvt_bf16_72 = __float2bfloat16(h_raw_0_2_bf16_f32_1[0] * _rcp_72);
                float _cvt_f32_72 = __bfloat162float(_cvt_bf16_72);
                __nv_bfloat16 _cvt_bf16_73 = __float2bfloat16(h_raw_0_2_bf16_f32_1[2] * _rcp_73);
                float _cvt_f32_73 = __bfloat162float(_cvt_bf16_73);
                y_lo_0_2_1[0] = _cvt_f32_72 * h_raw_0_2_bf16_f32_1[4];
                y_hi_0_2_1[0] = _cvt_f32_73 * h_raw_0_2_bf16_f32_1[6];
                float _exp2_74 = approx_exp2((-h_raw_0_2_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_74 = approx_rcp(1.0f + _exp2_74);
                float _exp2_75 = approx_exp2((-h_raw_0_2_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_75 = approx_rcp(1.0f + _exp2_75);
                __nv_bfloat16 _cvt_bf16_74 = __float2bfloat16(h_raw_0_2_bf16_f32_1[1] * _rcp_74);
                float _cvt_f32_74 = __bfloat162float(_cvt_bf16_74);
                __nv_bfloat16 _cvt_bf16_75 = __float2bfloat16(h_raw_0_2_bf16_f32_1[3] * _rcp_75);
                float _cvt_f32_75 = __bfloat162float(_cvt_bf16_75);
                y_lo_0_2_1[1] = _cvt_f32_74 * h_raw_0_2_bf16_f32_1[5];
                y_hi_0_2_1[1] = _cvt_f32_75 * h_raw_0_2_bf16_f32_1[7];
                uint32_t y_lo_0_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_2_1[_lp*2 + 0], y_lo_0_2_1[_lp*2+1 + 0]));
                    y_lo_0_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_2_1[_lp*2 + 0], y_hi_0_2_1[_lp*2+1 + 0]));
                    y_hi_0_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_0_2_bf16_1[0];
                }
                if (row_hi_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_0_2_bf16_1[0];
                }
                float h_raw_0_3_1[8];
                h_raw_0_3_1[0] = accum_1[24] * sa_lo_0_1 * sw_1[12];
                h_raw_0_3_1[2] = accum_1[26] * sa_hi_0_1 * sw_1[12];
                h_raw_0_3_1[4] = accum_1[28] * sa_lo_0_1 * sw_1[14];
                h_raw_0_3_1[6] = accum_1[30] * sa_hi_0_1 * sw_1[14];
                h_raw_0_3_1[1] = accum_1[25] * sa_lo_0_1 * sw_1[13];
                h_raw_0_3_1[3] = accum_1[27] * sa_hi_0_1 * sw_1[13];
                h_raw_0_3_1[5] = accum_1[29] * sa_lo_0_1 * sw_1[15];
                h_raw_0_3_1[7] = accum_1[31] * sa_hi_0_1 * sw_1[15];
                uint32_t h_raw_0_3_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_0_3_1[_lp*2 + 0], h_raw_0_3_1[_lp*2+1 + 0]));
                    h_raw_0_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_0_3_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_0_3_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_0_3_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_0_3_bf16_1[_pair]));
                }
                float y_lo_0_3_1[2];
                float y_hi_0_3_1[2];
                float _exp2_76 = approx_exp2((-h_raw_0_3_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_76 = approx_rcp(1.0f + _exp2_76);
                float _exp2_77 = approx_exp2((-h_raw_0_3_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_77 = approx_rcp(1.0f + _exp2_77);
                __nv_bfloat16 _cvt_bf16_76 = __float2bfloat16(h_raw_0_3_bf16_f32_1[0] * _rcp_76);
                float _cvt_f32_76 = __bfloat162float(_cvt_bf16_76);
                __nv_bfloat16 _cvt_bf16_77 = __float2bfloat16(h_raw_0_3_bf16_f32_1[2] * _rcp_77);
                float _cvt_f32_77 = __bfloat162float(_cvt_bf16_77);
                y_lo_0_3_1[0] = _cvt_f32_76 * h_raw_0_3_bf16_f32_1[4];
                y_hi_0_3_1[0] = _cvt_f32_77 * h_raw_0_3_bf16_f32_1[6];
                float _exp2_78 = approx_exp2((-h_raw_0_3_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_78 = approx_rcp(1.0f + _exp2_78);
                float _exp2_79 = approx_exp2((-h_raw_0_3_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_79 = approx_rcp(1.0f + _exp2_79);
                __nv_bfloat16 _cvt_bf16_78 = __float2bfloat16(h_raw_0_3_bf16_f32_1[1] * _rcp_78);
                float _cvt_f32_78 = __bfloat162float(_cvt_bf16_78);
                __nv_bfloat16 _cvt_bf16_79 = __float2bfloat16(h_raw_0_3_bf16_f32_1[3] * _rcp_79);
                float _cvt_f32_79 = __bfloat162float(_cvt_bf16_79);
                y_lo_0_3_1[1] = _cvt_f32_78 * h_raw_0_3_bf16_f32_1[5];
                y_hi_0_3_1[1] = _cvt_f32_79 * h_raw_0_3_bf16_f32_1[7];
                uint32_t y_lo_0_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_0_3_1[_lp*2 + 0], y_lo_0_3_1[_lp*2+1 + 0]));
                    y_lo_0_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_0_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_0_3_1[_lp*2 + 0], y_hi_0_3_1[_lp*2+1 + 0]));
                    y_hi_0_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_0_3_bf16_1[0];
                }
                if (row_hi_0_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_0_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_0_3_bf16_1[0];
                }
                int row_lo_1_1 = tile_m_1 * 128 + warp_m_1 * 64 + 16 + (lane >> 2);
                int row_hi_1_1 = row_lo_1_1 + 8;
                float sa_lo_1_1 = 1.0f;
                float sa_hi_1_1 = 1.0f;
                float h_raw_1_0_1[8];
                h_raw_1_0_1[0] = accum_1[32] * sa_lo_1_1 * sw_1[0];
                h_raw_1_0_1[2] = accum_1[34] * sa_hi_1_1 * sw_1[0];
                h_raw_1_0_1[4] = accum_1[36] * sa_lo_1_1 * sw_1[2];
                h_raw_1_0_1[6] = accum_1[38] * sa_hi_1_1 * sw_1[2];
                h_raw_1_0_1[1] = accum_1[33] * sa_lo_1_1 * sw_1[1];
                h_raw_1_0_1[3] = accum_1[35] * sa_hi_1_1 * sw_1[1];
                h_raw_1_0_1[5] = accum_1[37] * sa_lo_1_1 * sw_1[3];
                h_raw_1_0_1[7] = accum_1[39] * sa_hi_1_1 * sw_1[3];
                uint32_t h_raw_1_0_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_0_1[_lp*2 + 0], h_raw_1_0_1[_lp*2+1 + 0]));
                    h_raw_1_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_0_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_0_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_1_0_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_1_0_bf16_1[_pair]));
                }
                float y_lo_1_0_1[2];
                float y_hi_1_0_1[2];
                float _exp2_80 = approx_exp2((-h_raw_1_0_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_80 = approx_rcp(1.0f + _exp2_80);
                float _exp2_81 = approx_exp2((-h_raw_1_0_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_81 = approx_rcp(1.0f + _exp2_81);
                __nv_bfloat16 _cvt_bf16_80 = __float2bfloat16(h_raw_1_0_bf16_f32_1[0] * _rcp_80);
                float _cvt_f32_80 = __bfloat162float(_cvt_bf16_80);
                __nv_bfloat16 _cvt_bf16_81 = __float2bfloat16(h_raw_1_0_bf16_f32_1[2] * _rcp_81);
                float _cvt_f32_81 = __bfloat162float(_cvt_bf16_81);
                y_lo_1_0_1[0] = _cvt_f32_80 * h_raw_1_0_bf16_f32_1[4];
                y_hi_1_0_1[0] = _cvt_f32_81 * h_raw_1_0_bf16_f32_1[6];
                float _exp2_82 = approx_exp2((-h_raw_1_0_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_82 = approx_rcp(1.0f + _exp2_82);
                float _exp2_83 = approx_exp2((-h_raw_1_0_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_83 = approx_rcp(1.0f + _exp2_83);
                __nv_bfloat16 _cvt_bf16_82 = __float2bfloat16(h_raw_1_0_bf16_f32_1[1] * _rcp_82);
                float _cvt_f32_82 = __bfloat162float(_cvt_bf16_82);
                __nv_bfloat16 _cvt_bf16_83 = __float2bfloat16(h_raw_1_0_bf16_f32_1[3] * _rcp_83);
                float _cvt_f32_83 = __bfloat162float(_cvt_bf16_83);
                y_lo_1_0_1[1] = _cvt_f32_82 * h_raw_1_0_bf16_f32_1[5];
                y_hi_1_0_1[1] = _cvt_f32_83 * h_raw_1_0_bf16_f32_1[7];
                uint32_t y_lo_1_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_0_1[_lp*2 + 0], y_lo_1_0_1[_lp*2+1 + 0]));
                    y_lo_1_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_0_1[_lp*2 + 0], y_hi_1_0_1[_lp*2+1 + 0]));
                    y_hi_1_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_1_0_bf16_1[0];
                }
                if (row_hi_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_1_0_bf16_1[0];
                }
                float h_raw_1_1_1[8];
                h_raw_1_1_1[0] = accum_1[40] * sa_lo_1_1 * sw_1[4];
                h_raw_1_1_1[2] = accum_1[42] * sa_hi_1_1 * sw_1[4];
                h_raw_1_1_1[4] = accum_1[44] * sa_lo_1_1 * sw_1[6];
                h_raw_1_1_1[6] = accum_1[46] * sa_hi_1_1 * sw_1[6];
                h_raw_1_1_1[1] = accum_1[41] * sa_lo_1_1 * sw_1[5];
                h_raw_1_1_1[3] = accum_1[43] * sa_hi_1_1 * sw_1[5];
                h_raw_1_1_1[5] = accum_1[45] * sa_lo_1_1 * sw_1[7];
                h_raw_1_1_1[7] = accum_1[47] * sa_hi_1_1 * sw_1[7];
                uint32_t h_raw_1_1_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_1_1[_lp*2 + 0], h_raw_1_1_1[_lp*2+1 + 0]));
                    h_raw_1_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_1_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_1_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_1_1_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_1_1_bf16_1[_pair]));
                }
                float y_lo_1_1_1[2];
                float y_hi_1_1_1[2];
                float _exp2_84 = approx_exp2((-h_raw_1_1_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_84 = approx_rcp(1.0f + _exp2_84);
                float _exp2_85 = approx_exp2((-h_raw_1_1_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_85 = approx_rcp(1.0f + _exp2_85);
                __nv_bfloat16 _cvt_bf16_84 = __float2bfloat16(h_raw_1_1_bf16_f32_1[0] * _rcp_84);
                float _cvt_f32_84 = __bfloat162float(_cvt_bf16_84);
                __nv_bfloat16 _cvt_bf16_85 = __float2bfloat16(h_raw_1_1_bf16_f32_1[2] * _rcp_85);
                float _cvt_f32_85 = __bfloat162float(_cvt_bf16_85);
                y_lo_1_1_1[0] = _cvt_f32_84 * h_raw_1_1_bf16_f32_1[4];
                y_hi_1_1_1[0] = _cvt_f32_85 * h_raw_1_1_bf16_f32_1[6];
                float _exp2_86 = approx_exp2((-h_raw_1_1_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_86 = approx_rcp(1.0f + _exp2_86);
                float _exp2_87 = approx_exp2((-h_raw_1_1_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_87 = approx_rcp(1.0f + _exp2_87);
                __nv_bfloat16 _cvt_bf16_86 = __float2bfloat16(h_raw_1_1_bf16_f32_1[1] * _rcp_86);
                float _cvt_f32_86 = __bfloat162float(_cvt_bf16_86);
                __nv_bfloat16 _cvt_bf16_87 = __float2bfloat16(h_raw_1_1_bf16_f32_1[3] * _rcp_87);
                float _cvt_f32_87 = __bfloat162float(_cvt_bf16_87);
                y_lo_1_1_1[1] = _cvt_f32_86 * h_raw_1_1_bf16_f32_1[5];
                y_hi_1_1_1[1] = _cvt_f32_87 * h_raw_1_1_bf16_f32_1[7];
                uint32_t y_lo_1_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_1_1[_lp*2 + 0], y_lo_1_1_1[_lp*2+1 + 0]));
                    y_lo_1_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_1_1[_lp*2 + 0], y_hi_1_1_1[_lp*2+1 + 0]));
                    y_hi_1_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_1_1_bf16_1[0];
                }
                if (row_hi_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_1_1_bf16_1[0];
                }
                float h_raw_1_2_1[8];
                h_raw_1_2_1[0] = accum_1[48] * sa_lo_1_1 * sw_1[8];
                h_raw_1_2_1[2] = accum_1[50] * sa_hi_1_1 * sw_1[8];
                h_raw_1_2_1[4] = accum_1[52] * sa_lo_1_1 * sw_1[10];
                h_raw_1_2_1[6] = accum_1[54] * sa_hi_1_1 * sw_1[10];
                h_raw_1_2_1[1] = accum_1[49] * sa_lo_1_1 * sw_1[9];
                h_raw_1_2_1[3] = accum_1[51] * sa_hi_1_1 * sw_1[9];
                h_raw_1_2_1[5] = accum_1[53] * sa_lo_1_1 * sw_1[11];
                h_raw_1_2_1[7] = accum_1[55] * sa_hi_1_1 * sw_1[11];
                uint32_t h_raw_1_2_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_2_1[_lp*2 + 0], h_raw_1_2_1[_lp*2+1 + 0]));
                    h_raw_1_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_2_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_2_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_1_2_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_1_2_bf16_1[_pair]));
                }
                float y_lo_1_2_1[2];
                float y_hi_1_2_1[2];
                float _exp2_88 = approx_exp2((-h_raw_1_2_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_88 = approx_rcp(1.0f + _exp2_88);
                float _exp2_89 = approx_exp2((-h_raw_1_2_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_89 = approx_rcp(1.0f + _exp2_89);
                __nv_bfloat16 _cvt_bf16_88 = __float2bfloat16(h_raw_1_2_bf16_f32_1[0] * _rcp_88);
                float _cvt_f32_88 = __bfloat162float(_cvt_bf16_88);
                __nv_bfloat16 _cvt_bf16_89 = __float2bfloat16(h_raw_1_2_bf16_f32_1[2] * _rcp_89);
                float _cvt_f32_89 = __bfloat162float(_cvt_bf16_89);
                y_lo_1_2_1[0] = _cvt_f32_88 * h_raw_1_2_bf16_f32_1[4];
                y_hi_1_2_1[0] = _cvt_f32_89 * h_raw_1_2_bf16_f32_1[6];
                float _exp2_90 = approx_exp2((-h_raw_1_2_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_90 = approx_rcp(1.0f + _exp2_90);
                float _exp2_91 = approx_exp2((-h_raw_1_2_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_91 = approx_rcp(1.0f + _exp2_91);
                __nv_bfloat16 _cvt_bf16_90 = __float2bfloat16(h_raw_1_2_bf16_f32_1[1] * _rcp_90);
                float _cvt_f32_90 = __bfloat162float(_cvt_bf16_90);
                __nv_bfloat16 _cvt_bf16_91 = __float2bfloat16(h_raw_1_2_bf16_f32_1[3] * _rcp_91);
                float _cvt_f32_91 = __bfloat162float(_cvt_bf16_91);
                y_lo_1_2_1[1] = _cvt_f32_90 * h_raw_1_2_bf16_f32_1[5];
                y_hi_1_2_1[1] = _cvt_f32_91 * h_raw_1_2_bf16_f32_1[7];
                uint32_t y_lo_1_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_2_1[_lp*2 + 0], y_lo_1_2_1[_lp*2+1 + 0]));
                    y_lo_1_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_2_1[_lp*2 + 0], y_hi_1_2_1[_lp*2+1 + 0]));
                    y_hi_1_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_1_2_bf16_1[0];
                }
                if (row_hi_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_1_2_bf16_1[0];
                }
                float h_raw_1_3_1[8];
                h_raw_1_3_1[0] = accum_1[56] * sa_lo_1_1 * sw_1[12];
                h_raw_1_3_1[2] = accum_1[58] * sa_hi_1_1 * sw_1[12];
                h_raw_1_3_1[4] = accum_1[60] * sa_lo_1_1 * sw_1[14];
                h_raw_1_3_1[6] = accum_1[62] * sa_hi_1_1 * sw_1[14];
                h_raw_1_3_1[1] = accum_1[57] * sa_lo_1_1 * sw_1[13];
                h_raw_1_3_1[3] = accum_1[59] * sa_hi_1_1 * sw_1[13];
                h_raw_1_3_1[5] = accum_1[61] * sa_lo_1_1 * sw_1[15];
                h_raw_1_3_1[7] = accum_1[63] * sa_hi_1_1 * sw_1[15];
                uint32_t h_raw_1_3_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_1_3_1[_lp*2 + 0], h_raw_1_3_1[_lp*2+1 + 0]));
                    h_raw_1_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_1_3_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_1_3_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_1_3_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_1_3_bf16_1[_pair]));
                }
                float y_lo_1_3_1[2];
                float y_hi_1_3_1[2];
                float _exp2_92 = approx_exp2((-h_raw_1_3_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_92 = approx_rcp(1.0f + _exp2_92);
                float _exp2_93 = approx_exp2((-h_raw_1_3_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_93 = approx_rcp(1.0f + _exp2_93);
                __nv_bfloat16 _cvt_bf16_92 = __float2bfloat16(h_raw_1_3_bf16_f32_1[0] * _rcp_92);
                float _cvt_f32_92 = __bfloat162float(_cvt_bf16_92);
                __nv_bfloat16 _cvt_bf16_93 = __float2bfloat16(h_raw_1_3_bf16_f32_1[2] * _rcp_93);
                float _cvt_f32_93 = __bfloat162float(_cvt_bf16_93);
                y_lo_1_3_1[0] = _cvt_f32_92 * h_raw_1_3_bf16_f32_1[4];
                y_hi_1_3_1[0] = _cvt_f32_93 * h_raw_1_3_bf16_f32_1[6];
                float _exp2_94 = approx_exp2((-h_raw_1_3_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_94 = approx_rcp(1.0f + _exp2_94);
                float _exp2_95 = approx_exp2((-h_raw_1_3_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_95 = approx_rcp(1.0f + _exp2_95);
                __nv_bfloat16 _cvt_bf16_94 = __float2bfloat16(h_raw_1_3_bf16_f32_1[1] * _rcp_94);
                float _cvt_f32_94 = __bfloat162float(_cvt_bf16_94);
                __nv_bfloat16 _cvt_bf16_95 = __float2bfloat16(h_raw_1_3_bf16_f32_1[3] * _rcp_95);
                float _cvt_f32_95 = __bfloat162float(_cvt_bf16_95);
                y_lo_1_3_1[1] = _cvt_f32_94 * h_raw_1_3_bf16_f32_1[5];
                y_hi_1_3_1[1] = _cvt_f32_95 * h_raw_1_3_bf16_f32_1[7];
                uint32_t y_lo_1_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_1_3_1[_lp*2 + 0], y_lo_1_3_1[_lp*2+1 + 0]));
                    y_lo_1_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_1_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_1_3_1[_lp*2 + 0], y_hi_1_3_1[_lp*2+1 + 0]));
                    y_hi_1_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_1_3_bf16_1[0];
                }
                if (row_hi_1_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_1_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_1_3_bf16_1[0];
                }
                int row_lo_2_1 = tile_m_1 * 128 + warp_m_1 * 64 + 32 + (lane >> 2);
                int row_hi_2_1 = row_lo_2_1 + 8;
                float sa_lo_2_1 = 1.0f;
                float sa_hi_2_1 = 1.0f;
                float h_raw_2_0_1[8];
                h_raw_2_0_1[0] = accum_1[64] * sa_lo_2_1 * sw_1[0];
                h_raw_2_0_1[2] = accum_1[66] * sa_hi_2_1 * sw_1[0];
                h_raw_2_0_1[4] = accum_1[68] * sa_lo_2_1 * sw_1[2];
                h_raw_2_0_1[6] = accum_1[70] * sa_hi_2_1 * sw_1[2];
                h_raw_2_0_1[1] = accum_1[65] * sa_lo_2_1 * sw_1[1];
                h_raw_2_0_1[3] = accum_1[67] * sa_hi_2_1 * sw_1[1];
                h_raw_2_0_1[5] = accum_1[69] * sa_lo_2_1 * sw_1[3];
                h_raw_2_0_1[7] = accum_1[71] * sa_hi_2_1 * sw_1[3];
                uint32_t h_raw_2_0_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_0_1[_lp*2 + 0], h_raw_2_0_1[_lp*2+1 + 0]));
                    h_raw_2_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_0_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_0_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_2_0_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_2_0_bf16_1[_pair]));
                }
                float y_lo_2_0_1[2];
                float y_hi_2_0_1[2];
                float _exp2_96 = approx_exp2((-h_raw_2_0_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_96 = approx_rcp(1.0f + _exp2_96);
                float _exp2_97 = approx_exp2((-h_raw_2_0_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_97 = approx_rcp(1.0f + _exp2_97);
                __nv_bfloat16 _cvt_bf16_96 = __float2bfloat16(h_raw_2_0_bf16_f32_1[0] * _rcp_96);
                float _cvt_f32_96 = __bfloat162float(_cvt_bf16_96);
                __nv_bfloat16 _cvt_bf16_97 = __float2bfloat16(h_raw_2_0_bf16_f32_1[2] * _rcp_97);
                float _cvt_f32_97 = __bfloat162float(_cvt_bf16_97);
                y_lo_2_0_1[0] = _cvt_f32_96 * h_raw_2_0_bf16_f32_1[4];
                y_hi_2_0_1[0] = _cvt_f32_97 * h_raw_2_0_bf16_f32_1[6];
                float _exp2_98 = approx_exp2((-h_raw_2_0_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_98 = approx_rcp(1.0f + _exp2_98);
                float _exp2_99 = approx_exp2((-h_raw_2_0_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_99 = approx_rcp(1.0f + _exp2_99);
                __nv_bfloat16 _cvt_bf16_98 = __float2bfloat16(h_raw_2_0_bf16_f32_1[1] * _rcp_98);
                float _cvt_f32_98 = __bfloat162float(_cvt_bf16_98);
                __nv_bfloat16 _cvt_bf16_99 = __float2bfloat16(h_raw_2_0_bf16_f32_1[3] * _rcp_99);
                float _cvt_f32_99 = __bfloat162float(_cvt_bf16_99);
                y_lo_2_0_1[1] = _cvt_f32_98 * h_raw_2_0_bf16_f32_1[5];
                y_hi_2_0_1[1] = _cvt_f32_99 * h_raw_2_0_bf16_f32_1[7];
                uint32_t y_lo_2_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_0_1[_lp*2 + 0], y_lo_2_0_1[_lp*2+1 + 0]));
                    y_lo_2_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_0_1[_lp*2 + 0], y_hi_2_0_1[_lp*2+1 + 0]));
                    y_hi_2_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_2_0_bf16_1[0];
                }
                if (row_hi_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_2_0_bf16_1[0];
                }
                float h_raw_2_1_1[8];
                h_raw_2_1_1[0] = accum_1[72] * sa_lo_2_1 * sw_1[4];
                h_raw_2_1_1[2] = accum_1[74] * sa_hi_2_1 * sw_1[4];
                h_raw_2_1_1[4] = accum_1[76] * sa_lo_2_1 * sw_1[6];
                h_raw_2_1_1[6] = accum_1[78] * sa_hi_2_1 * sw_1[6];
                h_raw_2_1_1[1] = accum_1[73] * sa_lo_2_1 * sw_1[5];
                h_raw_2_1_1[3] = accum_1[75] * sa_hi_2_1 * sw_1[5];
                h_raw_2_1_1[5] = accum_1[77] * sa_lo_2_1 * sw_1[7];
                h_raw_2_1_1[7] = accum_1[79] * sa_hi_2_1 * sw_1[7];
                uint32_t h_raw_2_1_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_1_1[_lp*2 + 0], h_raw_2_1_1[_lp*2+1 + 0]));
                    h_raw_2_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_1_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_1_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_2_1_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_2_1_bf16_1[_pair]));
                }
                float y_lo_2_1_1[2];
                float y_hi_2_1_1[2];
                float _exp2_100 = approx_exp2((-h_raw_2_1_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_100 = approx_rcp(1.0f + _exp2_100);
                float _exp2_101 = approx_exp2((-h_raw_2_1_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_101 = approx_rcp(1.0f + _exp2_101);
                __nv_bfloat16 _cvt_bf16_100 = __float2bfloat16(h_raw_2_1_bf16_f32_1[0] * _rcp_100);
                float _cvt_f32_100 = __bfloat162float(_cvt_bf16_100);
                __nv_bfloat16 _cvt_bf16_101 = __float2bfloat16(h_raw_2_1_bf16_f32_1[2] * _rcp_101);
                float _cvt_f32_101 = __bfloat162float(_cvt_bf16_101);
                y_lo_2_1_1[0] = _cvt_f32_100 * h_raw_2_1_bf16_f32_1[4];
                y_hi_2_1_1[0] = _cvt_f32_101 * h_raw_2_1_bf16_f32_1[6];
                float _exp2_102 = approx_exp2((-h_raw_2_1_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_102 = approx_rcp(1.0f + _exp2_102);
                float _exp2_103 = approx_exp2((-h_raw_2_1_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_103 = approx_rcp(1.0f + _exp2_103);
                __nv_bfloat16 _cvt_bf16_102 = __float2bfloat16(h_raw_2_1_bf16_f32_1[1] * _rcp_102);
                float _cvt_f32_102 = __bfloat162float(_cvt_bf16_102);
                __nv_bfloat16 _cvt_bf16_103 = __float2bfloat16(h_raw_2_1_bf16_f32_1[3] * _rcp_103);
                float _cvt_f32_103 = __bfloat162float(_cvt_bf16_103);
                y_lo_2_1_1[1] = _cvt_f32_102 * h_raw_2_1_bf16_f32_1[5];
                y_hi_2_1_1[1] = _cvt_f32_103 * h_raw_2_1_bf16_f32_1[7];
                uint32_t y_lo_2_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_1_1[_lp*2 + 0], y_lo_2_1_1[_lp*2+1 + 0]));
                    y_lo_2_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_1_1[_lp*2 + 0], y_hi_2_1_1[_lp*2+1 + 0]));
                    y_hi_2_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_2_1_bf16_1[0];
                }
                if (row_hi_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_2_1_bf16_1[0];
                }
                float h_raw_2_2_1[8];
                h_raw_2_2_1[0] = accum_1[80] * sa_lo_2_1 * sw_1[8];
                h_raw_2_2_1[2] = accum_1[82] * sa_hi_2_1 * sw_1[8];
                h_raw_2_2_1[4] = accum_1[84] * sa_lo_2_1 * sw_1[10];
                h_raw_2_2_1[6] = accum_1[86] * sa_hi_2_1 * sw_1[10];
                h_raw_2_2_1[1] = accum_1[81] * sa_lo_2_1 * sw_1[9];
                h_raw_2_2_1[3] = accum_1[83] * sa_hi_2_1 * sw_1[9];
                h_raw_2_2_1[5] = accum_1[85] * sa_lo_2_1 * sw_1[11];
                h_raw_2_2_1[7] = accum_1[87] * sa_hi_2_1 * sw_1[11];
                uint32_t h_raw_2_2_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_2_1[_lp*2 + 0], h_raw_2_2_1[_lp*2+1 + 0]));
                    h_raw_2_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_2_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_2_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_2_2_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_2_2_bf16_1[_pair]));
                }
                float y_lo_2_2_1[2];
                float y_hi_2_2_1[2];
                float _exp2_104 = approx_exp2((-h_raw_2_2_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_104 = approx_rcp(1.0f + _exp2_104);
                float _exp2_105 = approx_exp2((-h_raw_2_2_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_105 = approx_rcp(1.0f + _exp2_105);
                __nv_bfloat16 _cvt_bf16_104 = __float2bfloat16(h_raw_2_2_bf16_f32_1[0] * _rcp_104);
                float _cvt_f32_104 = __bfloat162float(_cvt_bf16_104);
                __nv_bfloat16 _cvt_bf16_105 = __float2bfloat16(h_raw_2_2_bf16_f32_1[2] * _rcp_105);
                float _cvt_f32_105 = __bfloat162float(_cvt_bf16_105);
                y_lo_2_2_1[0] = _cvt_f32_104 * h_raw_2_2_bf16_f32_1[4];
                y_hi_2_2_1[0] = _cvt_f32_105 * h_raw_2_2_bf16_f32_1[6];
                float _exp2_106 = approx_exp2((-h_raw_2_2_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_106 = approx_rcp(1.0f + _exp2_106);
                float _exp2_107 = approx_exp2((-h_raw_2_2_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_107 = approx_rcp(1.0f + _exp2_107);
                __nv_bfloat16 _cvt_bf16_106 = __float2bfloat16(h_raw_2_2_bf16_f32_1[1] * _rcp_106);
                float _cvt_f32_106 = __bfloat162float(_cvt_bf16_106);
                __nv_bfloat16 _cvt_bf16_107 = __float2bfloat16(h_raw_2_2_bf16_f32_1[3] * _rcp_107);
                float _cvt_f32_107 = __bfloat162float(_cvt_bf16_107);
                y_lo_2_2_1[1] = _cvt_f32_106 * h_raw_2_2_bf16_f32_1[5];
                y_hi_2_2_1[1] = _cvt_f32_107 * h_raw_2_2_bf16_f32_1[7];
                uint32_t y_lo_2_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_2_1[_lp*2 + 0], y_lo_2_2_1[_lp*2+1 + 0]));
                    y_lo_2_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_2_1[_lp*2 + 0], y_hi_2_2_1[_lp*2+1 + 0]));
                    y_hi_2_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_2_2_bf16_1[0];
                }
                if (row_hi_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_2_2_bf16_1[0];
                }
                float h_raw_2_3_1[8];
                h_raw_2_3_1[0] = accum_1[88] * sa_lo_2_1 * sw_1[12];
                h_raw_2_3_1[2] = accum_1[90] * sa_hi_2_1 * sw_1[12];
                h_raw_2_3_1[4] = accum_1[92] * sa_lo_2_1 * sw_1[14];
                h_raw_2_3_1[6] = accum_1[94] * sa_hi_2_1 * sw_1[14];
                h_raw_2_3_1[1] = accum_1[89] * sa_lo_2_1 * sw_1[13];
                h_raw_2_3_1[3] = accum_1[91] * sa_hi_2_1 * sw_1[13];
                h_raw_2_3_1[5] = accum_1[93] * sa_lo_2_1 * sw_1[15];
                h_raw_2_3_1[7] = accum_1[95] * sa_hi_2_1 * sw_1[15];
                uint32_t h_raw_2_3_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_2_3_1[_lp*2 + 0], h_raw_2_3_1[_lp*2+1 + 0]));
                    h_raw_2_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_2_3_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_2_3_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_2_3_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_2_3_bf16_1[_pair]));
                }
                float y_lo_2_3_1[2];
                float y_hi_2_3_1[2];
                float _exp2_108 = approx_exp2((-h_raw_2_3_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_108 = approx_rcp(1.0f + _exp2_108);
                float _exp2_109 = approx_exp2((-h_raw_2_3_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_109 = approx_rcp(1.0f + _exp2_109);
                __nv_bfloat16 _cvt_bf16_108 = __float2bfloat16(h_raw_2_3_bf16_f32_1[0] * _rcp_108);
                float _cvt_f32_108 = __bfloat162float(_cvt_bf16_108);
                __nv_bfloat16 _cvt_bf16_109 = __float2bfloat16(h_raw_2_3_bf16_f32_1[2] * _rcp_109);
                float _cvt_f32_109 = __bfloat162float(_cvt_bf16_109);
                y_lo_2_3_1[0] = _cvt_f32_108 * h_raw_2_3_bf16_f32_1[4];
                y_hi_2_3_1[0] = _cvt_f32_109 * h_raw_2_3_bf16_f32_1[6];
                float _exp2_110 = approx_exp2((-h_raw_2_3_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_110 = approx_rcp(1.0f + _exp2_110);
                float _exp2_111 = approx_exp2((-h_raw_2_3_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_111 = approx_rcp(1.0f + _exp2_111);
                __nv_bfloat16 _cvt_bf16_110 = __float2bfloat16(h_raw_2_3_bf16_f32_1[1] * _rcp_110);
                float _cvt_f32_110 = __bfloat162float(_cvt_bf16_110);
                __nv_bfloat16 _cvt_bf16_111 = __float2bfloat16(h_raw_2_3_bf16_f32_1[3] * _rcp_111);
                float _cvt_f32_111 = __bfloat162float(_cvt_bf16_111);
                y_lo_2_3_1[1] = _cvt_f32_110 * h_raw_2_3_bf16_f32_1[5];
                y_hi_2_3_1[1] = _cvt_f32_111 * h_raw_2_3_bf16_f32_1[7];
                uint32_t y_lo_2_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_2_3_1[_lp*2 + 0], y_lo_2_3_1[_lp*2+1 + 0]));
                    y_lo_2_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_2_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_2_3_1[_lp*2 + 0], y_hi_2_3_1[_lp*2+1 + 0]));
                    y_hi_2_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_2_3_bf16_1[0];
                }
                if (row_hi_2_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_2_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_2_3_bf16_1[0];
                }
                int row_lo_3_1 = tile_m_1 * 128 + warp_m_1 * 64 + 48 + (lane >> 2);
                int row_hi_3_1 = row_lo_3_1 + 8;
                float sa_lo_3_1 = 1.0f;
                float sa_hi_3_1 = 1.0f;
                float h_raw_3_0_1[8];
                h_raw_3_0_1[0] = accum_1[96] * sa_lo_3_1 * sw_1[0];
                h_raw_3_0_1[2] = accum_1[98] * sa_hi_3_1 * sw_1[0];
                h_raw_3_0_1[4] = accum_1[100] * sa_lo_3_1 * sw_1[2];
                h_raw_3_0_1[6] = accum_1[102] * sa_hi_3_1 * sw_1[2];
                h_raw_3_0_1[1] = accum_1[97] * sa_lo_3_1 * sw_1[1];
                h_raw_3_0_1[3] = accum_1[99] * sa_hi_3_1 * sw_1[1];
                h_raw_3_0_1[5] = accum_1[101] * sa_lo_3_1 * sw_1[3];
                h_raw_3_0_1[7] = accum_1[103] * sa_hi_3_1 * sw_1[3];
                uint32_t h_raw_3_0_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_0_1[_lp*2 + 0], h_raw_3_0_1[_lp*2+1 + 0]));
                    h_raw_3_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_0_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_0_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_3_0_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_3_0_bf16_1[_pair]));
                }
                float y_lo_3_0_1[2];
                float y_hi_3_0_1[2];
                float _exp2_112 = approx_exp2((-h_raw_3_0_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_112 = approx_rcp(1.0f + _exp2_112);
                float _exp2_113 = approx_exp2((-h_raw_3_0_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_113 = approx_rcp(1.0f + _exp2_113);
                __nv_bfloat16 _cvt_bf16_112 = __float2bfloat16(h_raw_3_0_bf16_f32_1[0] * _rcp_112);
                float _cvt_f32_112 = __bfloat162float(_cvt_bf16_112);
                __nv_bfloat16 _cvt_bf16_113 = __float2bfloat16(h_raw_3_0_bf16_f32_1[2] * _rcp_113);
                float _cvt_f32_113 = __bfloat162float(_cvt_bf16_113);
                y_lo_3_0_1[0] = _cvt_f32_112 * h_raw_3_0_bf16_f32_1[4];
                y_hi_3_0_1[0] = _cvt_f32_113 * h_raw_3_0_bf16_f32_1[6];
                float _exp2_114 = approx_exp2((-h_raw_3_0_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_114 = approx_rcp(1.0f + _exp2_114);
                float _exp2_115 = approx_exp2((-h_raw_3_0_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_115 = approx_rcp(1.0f + _exp2_115);
                __nv_bfloat16 _cvt_bf16_114 = __float2bfloat16(h_raw_3_0_bf16_f32_1[1] * _rcp_114);
                float _cvt_f32_114 = __bfloat162float(_cvt_bf16_114);
                __nv_bfloat16 _cvt_bf16_115 = __float2bfloat16(h_raw_3_0_bf16_f32_1[3] * _rcp_115);
                float _cvt_f32_115 = __bfloat162float(_cvt_bf16_115);
                y_lo_3_0_1[1] = _cvt_f32_114 * h_raw_3_0_bf16_f32_1[5];
                y_hi_3_0_1[1] = _cvt_f32_115 * h_raw_3_0_bf16_f32_1[7];
                uint32_t y_lo_3_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_0_1[_lp*2 + 0], y_lo_3_0_1[_lp*2+1 + 0]));
                    y_lo_3_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_0_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_0_1[_lp*2 + 0], y_hi_3_0_1[_lp*2+1 + 0]));
                    y_hi_3_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_lo_3_0_bf16_1[0];
                }
                if (row_hi_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2)) / 2)) + (0)) = y_hi_3_0_bf16_1[0];
                }
                float h_raw_3_1_1[8];
                h_raw_3_1_1[0] = accum_1[104] * sa_lo_3_1 * sw_1[4];
                h_raw_3_1_1[2] = accum_1[106] * sa_hi_3_1 * sw_1[4];
                h_raw_3_1_1[4] = accum_1[108] * sa_lo_3_1 * sw_1[6];
                h_raw_3_1_1[6] = accum_1[110] * sa_hi_3_1 * sw_1[6];
                h_raw_3_1_1[1] = accum_1[105] * sa_lo_3_1 * sw_1[5];
                h_raw_3_1_1[3] = accum_1[107] * sa_hi_3_1 * sw_1[5];
                h_raw_3_1_1[5] = accum_1[109] * sa_lo_3_1 * sw_1[7];
                h_raw_3_1_1[7] = accum_1[111] * sa_hi_3_1 * sw_1[7];
                uint32_t h_raw_3_1_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_1_1[_lp*2 + 0], h_raw_3_1_1[_lp*2+1 + 0]));
                    h_raw_3_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_1_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_1_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_3_1_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_3_1_bf16_1[_pair]));
                }
                float y_lo_3_1_1[2];
                float y_hi_3_1_1[2];
                float _exp2_116 = approx_exp2((-h_raw_3_1_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_116 = approx_rcp(1.0f + _exp2_116);
                float _exp2_117 = approx_exp2((-h_raw_3_1_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_117 = approx_rcp(1.0f + _exp2_117);
                __nv_bfloat16 _cvt_bf16_116 = __float2bfloat16(h_raw_3_1_bf16_f32_1[0] * _rcp_116);
                float _cvt_f32_116 = __bfloat162float(_cvt_bf16_116);
                __nv_bfloat16 _cvt_bf16_117 = __float2bfloat16(h_raw_3_1_bf16_f32_1[2] * _rcp_117);
                float _cvt_f32_117 = __bfloat162float(_cvt_bf16_117);
                y_lo_3_1_1[0] = _cvt_f32_116 * h_raw_3_1_bf16_f32_1[4];
                y_hi_3_1_1[0] = _cvt_f32_117 * h_raw_3_1_bf16_f32_1[6];
                float _exp2_118 = approx_exp2((-h_raw_3_1_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_118 = approx_rcp(1.0f + _exp2_118);
                float _exp2_119 = approx_exp2((-h_raw_3_1_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_119 = approx_rcp(1.0f + _exp2_119);
                __nv_bfloat16 _cvt_bf16_118 = __float2bfloat16(h_raw_3_1_bf16_f32_1[1] * _rcp_118);
                float _cvt_f32_118 = __bfloat162float(_cvt_bf16_118);
                __nv_bfloat16 _cvt_bf16_119 = __float2bfloat16(h_raw_3_1_bf16_f32_1[3] * _rcp_119);
                float _cvt_f32_119 = __bfloat162float(_cvt_bf16_119);
                y_lo_3_1_1[1] = _cvt_f32_118 * h_raw_3_1_bf16_f32_1[5];
                y_hi_3_1_1[1] = _cvt_f32_119 * h_raw_3_1_bf16_f32_1[7];
                uint32_t y_lo_3_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_1_1[_lp*2 + 0], y_lo_3_1_1[_lp*2+1 + 0]));
                    y_lo_3_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_1_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_1_1[_lp*2 + 0], y_hi_3_1_1[_lp*2+1 + 0]));
                    y_hi_3_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_lo_3_1_bf16_1[0];
                }
                if (row_hi_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 8)) / 2)) + (0)) = y_hi_3_1_bf16_1[0];
                }
                float h_raw_3_2_1[8];
                h_raw_3_2_1[0] = accum_1[112] * sa_lo_3_1 * sw_1[8];
                h_raw_3_2_1[2] = accum_1[114] * sa_hi_3_1 * sw_1[8];
                h_raw_3_2_1[4] = accum_1[116] * sa_lo_3_1 * sw_1[10];
                h_raw_3_2_1[6] = accum_1[118] * sa_hi_3_1 * sw_1[10];
                h_raw_3_2_1[1] = accum_1[113] * sa_lo_3_1 * sw_1[9];
                h_raw_3_2_1[3] = accum_1[115] * sa_hi_3_1 * sw_1[9];
                h_raw_3_2_1[5] = accum_1[117] * sa_lo_3_1 * sw_1[11];
                h_raw_3_2_1[7] = accum_1[119] * sa_hi_3_1 * sw_1[11];
                uint32_t h_raw_3_2_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_2_1[_lp*2 + 0], h_raw_3_2_1[_lp*2+1 + 0]));
                    h_raw_3_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_2_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_2_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_3_2_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_3_2_bf16_1[_pair]));
                }
                float y_lo_3_2_1[2];
                float y_hi_3_2_1[2];
                float _exp2_120 = approx_exp2((-h_raw_3_2_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_120 = approx_rcp(1.0f + _exp2_120);
                float _exp2_121 = approx_exp2((-h_raw_3_2_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_121 = approx_rcp(1.0f + _exp2_121);
                __nv_bfloat16 _cvt_bf16_120 = __float2bfloat16(h_raw_3_2_bf16_f32_1[0] * _rcp_120);
                float _cvt_f32_120 = __bfloat162float(_cvt_bf16_120);
                __nv_bfloat16 _cvt_bf16_121 = __float2bfloat16(h_raw_3_2_bf16_f32_1[2] * _rcp_121);
                float _cvt_f32_121 = __bfloat162float(_cvt_bf16_121);
                y_lo_3_2_1[0] = _cvt_f32_120 * h_raw_3_2_bf16_f32_1[4];
                y_hi_3_2_1[0] = _cvt_f32_121 * h_raw_3_2_bf16_f32_1[6];
                float _exp2_122 = approx_exp2((-h_raw_3_2_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_122 = approx_rcp(1.0f + _exp2_122);
                float _exp2_123 = approx_exp2((-h_raw_3_2_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_123 = approx_rcp(1.0f + _exp2_123);
                __nv_bfloat16 _cvt_bf16_122 = __float2bfloat16(h_raw_3_2_bf16_f32_1[1] * _rcp_122);
                float _cvt_f32_122 = __bfloat162float(_cvt_bf16_122);
                __nv_bfloat16 _cvt_bf16_123 = __float2bfloat16(h_raw_3_2_bf16_f32_1[3] * _rcp_123);
                float _cvt_f32_123 = __bfloat162float(_cvt_bf16_123);
                y_lo_3_2_1[1] = _cvt_f32_122 * h_raw_3_2_bf16_f32_1[5];
                y_hi_3_2_1[1] = _cvt_f32_123 * h_raw_3_2_bf16_f32_1[7];
                uint32_t y_lo_3_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_2_1[_lp*2 + 0], y_lo_3_2_1[_lp*2+1 + 0]));
                    y_lo_3_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_2_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_2_1[_lp*2 + 0], y_hi_3_2_1[_lp*2+1 + 0]));
                    y_hi_3_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_lo_3_2_bf16_1[0];
                }
                if (row_hi_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 16)) / 2)) + (0)) = y_hi_3_2_bf16_1[0];
                }
                float h_raw_3_3_1[8];
                h_raw_3_3_1[0] = accum_1[120] * sa_lo_3_1 * sw_1[12];
                h_raw_3_3_1[2] = accum_1[122] * sa_hi_3_1 * sw_1[12];
                h_raw_3_3_1[4] = accum_1[124] * sa_lo_3_1 * sw_1[14];
                h_raw_3_3_1[6] = accum_1[126] * sa_hi_3_1 * sw_1[14];
                h_raw_3_3_1[1] = accum_1[121] * sa_lo_3_1 * sw_1[13];
                h_raw_3_3_1[3] = accum_1[123] * sa_hi_3_1 * sw_1[13];
                h_raw_3_3_1[5] = accum_1[125] * sa_lo_3_1 * sw_1[15];
                h_raw_3_3_1[7] = accum_1[127] * sa_hi_3_1 * sw_1[15];
                uint32_t h_raw_3_3_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_raw_3_3_1[_lp*2 + 0], h_raw_3_3_1[_lp*2+1 + 0]));
                    h_raw_3_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float h_raw_3_3_bf16_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&h_raw_3_3_bf16_f32_1[_pair * 2])[0]), "=f"((&h_raw_3_3_bf16_f32_1[_pair * 2])[1])
                        : "r"(h_raw_3_3_bf16_1[_pair]));
                }
                float y_lo_3_3_1[2];
                float y_hi_3_3_1[2];
                float _exp2_124 = approx_exp2((-h_raw_3_3_bf16_f32_1[0]) * 1.4426950408889634f);
                float _rcp_124 = approx_rcp(1.0f + _exp2_124);
                float _exp2_125 = approx_exp2((-h_raw_3_3_bf16_f32_1[2]) * 1.4426950408889634f);
                float _rcp_125 = approx_rcp(1.0f + _exp2_125);
                __nv_bfloat16 _cvt_bf16_124 = __float2bfloat16(h_raw_3_3_bf16_f32_1[0] * _rcp_124);
                float _cvt_f32_124 = __bfloat162float(_cvt_bf16_124);
                __nv_bfloat16 _cvt_bf16_125 = __float2bfloat16(h_raw_3_3_bf16_f32_1[2] * _rcp_125);
                float _cvt_f32_125 = __bfloat162float(_cvt_bf16_125);
                y_lo_3_3_1[0] = _cvt_f32_124 * h_raw_3_3_bf16_f32_1[4];
                y_hi_3_3_1[0] = _cvt_f32_125 * h_raw_3_3_bf16_f32_1[6];
                float _exp2_126 = approx_exp2((-h_raw_3_3_bf16_f32_1[1]) * 1.4426950408889634f);
                float _rcp_126 = approx_rcp(1.0f + _exp2_126);
                float _exp2_127 = approx_exp2((-h_raw_3_3_bf16_f32_1[3]) * 1.4426950408889634f);
                float _rcp_127 = approx_rcp(1.0f + _exp2_127);
                __nv_bfloat16 _cvt_bf16_126 = __float2bfloat16(h_raw_3_3_bf16_f32_1[1] * _rcp_126);
                float _cvt_f32_126 = __bfloat162float(_cvt_bf16_126);
                __nv_bfloat16 _cvt_bf16_127 = __float2bfloat16(h_raw_3_3_bf16_f32_1[3] * _rcp_127);
                float _cvt_f32_127 = __bfloat162float(_cvt_bf16_127);
                y_lo_3_3_1[1] = _cvt_f32_126 * h_raw_3_3_bf16_f32_1[5];
                y_hi_3_3_1[1] = _cvt_f32_127 * h_raw_3_3_bf16_f32_1[7];
                uint32_t y_lo_3_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_lo_3_3_1[_lp*2 + 0], y_lo_3_3_1[_lp*2+1 + 0]));
                    y_lo_3_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t y_hi_3_3_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_hi_3_3_1[_lp*2 + 0], y_hi_3_3_1[_lp*2+1 + 0]));
                    y_hi_3_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (row_lo_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_lo_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_lo_3_3_bf16_1[0];
                }
                if (row_hi_3_1 < M) {
                    *(reinterpret_cast<unsigned int*>(out + ((row_hi_3_1 * 14336 + (tile_n_1 * 128 + warp_n_1 * 32 + (lane & 3) * 2 + 24)) / 2)) + (0)) = y_hi_3_3_bf16_1[0];
                }
            }
        }
    }

    // Cleanup
}

}  // namespace h3_fc1_swiglu_gemm_nvfp4_sm120a
#undef GROUP_M
#undef H3_FC1_INF
#undef NUM_EMPTY_PIPE_STAGES
#undef NUM_FULL_PIPE_STAGES
#undef SMEM_A_STAGE_OFF
#undef SMEM_A_STAGE_STAGE_BYTES
#undef SMEM_A_STAGE_STRIDE
#undef SMEM_B_STAGE_OFF
#undef SMEM_B_STAGE_STAGE_BYTES
#undef SMEM_B_STAGE_STRIDE
#undef SMEM_SFA_PAIRS_OFF
#undef SMEM_SFA_PAIRS_STAGE_BYTES
#undef SMEM_SFA_PAIRS_STRIDE
#undef SMEM_SFB_STAGE_OFF
#undef SMEM_SFB_STAGE_STAGE_BYTES
#undef SMEM_SFB_STAGE_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef ab_empty_addr
#undef ab_full_addr

#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>
#include <vector>

#include "tvm_ffi_utils.h"

namespace {

constexpr int64_t kHidden = 5376;
constexpr int64_t kFfn = 14336;
constexpr int64_t kFc1Rows = 28672;  // prepacked FC1 weight rows (128 gate + 128 up per output tile)
constexpr int64_t kSfBlock = 16;
constexpr int64_t kHiddenSf = kHidden / kSfBlock;  // 336 UE4M3 scales per activation row
constexpr int64_t kPackedCols = kHidden / 2;      // 2688 E2M1x2 bytes per row
// TMA coordinates are 32-bit; the persistent tile counter is int32.
constexpr int64_t kMaxRows = int64_t{1} << 24;
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockB = 256;    // weight rows per tile
constexpr int kBlockKb = 64;  // operand bytes per row per ring stage
constexpr int kNTiles = 112;
constexpr int kSfaBoxInner = 16;  // activation scale bytes per TMA box (two K tiles)
constexpr int kSfbRowsPerSfTile = 168;  // 256-byte rows per 128-row weight scale tile
constexpr int kSfbTmaRows = 4;
constexpr int64_t kNumSfTiles = 224;  // 128-row weight scale tiles (28672 / 128)
constexpr int kQuantThreads = 128;
// Dynamic shared memory of the two quantization kernels (their SMEM_TOTAL identity macros).
constexpr int kQuantFp8SmemBytes = 128;
constexpr int kQuantNvfp4SmemBytes = 128;
constexpr int kGemmThreads = 256;
constexpr int kGemmFp8SmemBytes = 99328;
constexpr int kGemmNvfp4SmemBytes = 84992;

static_assert(kFc1Rows == 2 * kFfn, "prepacked FC1 rows = gate + up");
static_assert(kNumSfTiles * 128 == kFc1Rows, "weight scale tiles cover the prepacked rows");
static_assert(int64_t{kNTiles} * kBlockN == kFfn, "output tiles cover the FFN width");
static_assert(int64_t{kSfbRowsPerSfTile} * 256 == 128 * kHiddenSf, "weight scale tile bytes");

using GemmKernel = void (*)(CUtensorMap, CUtensorMap, CUtensorMap, CUtensorMap, float*, float*, unsigned int*, int,
                            int, int, float);

struct GemmVariant {
  GemmKernel kernel;
  int dynamic_smem_bytes;
};

// [quant (0 = fp8, 1 = nvfp4)]
const GemmVariant kGemmVariants[2] = {
    {h3_fc1_swiglu_gemm_fp8_sm120a::kernel_h3_fc1_swiglu_gemm, kGemmFp8SmemBytes},
    {h3_fc1_swiglu_gemm_nvfp4_sm120a::kernel_h3_fc1_swiglu_gemm, kGemmNvfp4SmemBytes},
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
      << "MiniMax-H3 SM120 quantized FC1+SwiGLU requires compute capability 12.x (GB202); got "
      << properties.major << "." << properties.minor;
  for (const auto& variant : kGemmVariants) {
    status = cudaFuncSetAttribute(variant.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  variant.dynamic_smem_bytes);
    TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
        << "failed to opt in to dynamic shared memory: " << cudaGetErrorString(status);
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
  // Mirrors the Python launch_plan(): one persistent CTA per SM, 8 quantization CTAs per SM.
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
  unsigned int* out;
};

// Shared validation of the BF16 operands and the caller-owned BF16 [M, 14336] output.
CommonArgs CheckCommon(const TensorView& x, const TensorView& x_norm_weight, const TensorView& adaln_scale,
                       const TensorView& adaln_shift, const TensorView& adaln_index, const TensorView& out) {
  TVM_FFI_CHECK(x.ndim() == 2 && x.size(0) >= 1 && x.size(0) <= kMaxRows && x.size(1) == kHidden, ValueError)
      << "x must be [M, 5376] with 1 <= M <= " << kMaxRows;
  CommonArgs args{};
  args.rows = x.size(0);
  args.device = x.device();
  CheckTensor(x, "x", args.device, dl_bfloat16, {args.rows, kHidden});
  CheckTensor(x_norm_weight, "x_norm_weight", args.device, dl_bfloat16, {kHidden});
  TVM_FFI_CHECK(adaln_scale.ndim() == 2 && adaln_scale.size(0) >= 1 && adaln_scale.size(1) == kHidden, ValueError)
      << "adaln_scale must be [rows, 5376]";
  const int64_t adaln_rows = adaln_scale.size(0);
  CheckTensor(adaln_scale, "adaln_scale", args.device, dl_bfloat16, {adaln_rows, kHidden});
  CheckTensor(adaln_shift, "adaln_shift", args.device, dl_bfloat16, {adaln_rows, kHidden});
  CheckTensor(adaln_index, "adaln_index", args.device, dl_int32, {args.rows});
  CheckTensor(out, "out", args.device, dl_bfloat16, {args.rows, kFfn});
  args.out = static_cast<unsigned int*>(out.data_ptr());
  args.num_sms = ConfigureKernels();
  args.stream = get_stream(args.device);
  args.plan = MakeLaunchPlan(args.rows, args.num_sms);
  return args;
}

void LaunchGemm(int quant, const CommonArgs& args, const CUtensorMap& a_map, const CUtensorMap& b_map,
                const CUtensorMap& sfa_map, const CUtensorMap& sfb_map, float* act_scale, float* w_scale,
                double alpha, const char* what) {
  const GemmVariant variant = kGemmVariants[quant];
  variant.kernel<<<dim3(args.plan.gemm_grid), dim3(kGemmThreads), variant.dynamic_smem_bytes, args.stream>>>(
      a_map, b_map, sfa_map, sfb_map, act_scale, w_scale, args.out, static_cast<int>(args.rows),
      args.plan.num_m_tiles, args.plan.total_tiles, static_cast<float>(alpha));
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << what << " GEMM launch failed: " << cudaGetErrorString(status);
}

}  // namespace

// FP8 W8A8 route.  x [M, 5376] BF16 -> workspace_q [M, 5376] E4M3 (per-token scale workspace_scale [M]
// FP32 = RN(amax / 448)) -> h = BF16(workspace_q @ fc1_weight_q^T * workspace_scale[m] * fc1_weight_scale[n])
// -> out[m, c] = BF16(BF16(silu(h_gate[c])) * h_up[c]) for c in [0, 14336).
// fc1_weight_q: E4M3 [28672, 5376] and fc1_weight_scale: FP32 [28672] in the SM120 prepacked row order:
// packed row 16 * (c / 8) + (c % 8) + 8 * is_up holds output column c (8 gate rows, then the 8 matching
// up rows), i.e. the per-output-channel quantization of the [gate rows; up rows] FC1 weight, interleaved.
void minimax_h3_sm120_fp8_fc1_swiglu(TensorView x, TensorView x_norm_weight, TensorView adaln_scale,
                                     TensorView adaln_shift, TensorView adaln_index, TensorView fc1_weight_q,
                                     TensorView fc1_weight_scale, TensorView workspace_q, TensorView workspace_scale,
                                     TensorView out, double eps) {
  const CommonArgs args = CheckCommon(x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, out);
  CheckTensor(fc1_weight_q, "fc1_weight_q", args.device, dl_float8_e4m3fn, {kFc1Rows, kHidden});
  CheckTensor(fc1_weight_scale, "fc1_weight_scale", args.device, dl_float32, {kFc1Rows});
  CheckTensor(workspace_q, "workspace_q", args.device, dl_float8_e4m3fn, {args.rows, kHidden});
  CheckTensor(workspace_scale, "workspace_scale", args.device, dl_float32, {args.rows});
  ffi::CUDADeviceGuard device_guard(args.device.device_id);

  h3_fc1_norm_adaln_quant_fp8_sm120a::kernel_h3_norm_adaln_quant_fp8<<<dim3(args.plan.quant_grid), dim3(kQuantThreads), kQuantFp8SmemBytes, args.stream>>>(
      static_cast<__nv_bfloat16*>(x.data_ptr()), static_cast<__nv_bfloat16*>(x_norm_weight.data_ptr()),
      static_cast<__nv_bfloat16*>(adaln_scale.data_ptr()), static_cast<__nv_bfloat16*>(adaln_shift.data_ptr()),
      static_cast<int*>(adaln_index.data_ptr()), static_cast<unsigned int*>(workspace_q.data_ptr()),
      static_cast<float*>(workspace_scale.data_ptr()), static_cast<int>(args.rows), static_cast<float>(eps));
  cudaError_t status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "MiniMax-H3 FP8 norm/AdaLN quantization launch failed: " << cudaGetErrorString(status);

  const CUtensorMap a_map = EncodeByteTile(workspace_q.data_ptr(), kHidden, args.rows, kBlockKb, kBlockM,
                                           CU_TENSOR_MAP_SWIZZLE_64B, "workspace_q");
  const CUtensorMap b_map = EncodeByteTile(fc1_weight_q.data_ptr(), kHidden, kFc1Rows, kBlockKb, kBlockB,
                                           CU_TENSOR_MAP_SWIZZLE_64B, "fc1_weight_q");
  // The FP8 kernel never issues scale-tile loads; its descriptors only need valid parameter storage.
  const CUtensorMap unused_map{};
  LaunchGemm(0, args, a_map, b_map, unused_map, unused_map, static_cast<float*>(workspace_scale.data_ptr()),
             static_cast<float*>(fc1_weight_scale.data_ptr()), 1.0, "MiniMax-H3 FP8");
}

// NVFP4 route (FlashInfer conventions).  x -> workspace_q [M, 2688] u8 (E2M1x2) + workspace_sf [M, 336] UE4M3 u8
// (row-major, block 16) with the caller's activation global scale act_global_scale [1] FP32 (448 * 6 / amax).
// fc1_weight_q: u8 [28672, 2688] in the SM120 prepacked row order (see the FP8 route), fc1_scale_tiles: u8 with
// 28672 * 336 entries = the FlashInfer 128x4 swizzled block scales of the prepacked rows
// (fp4_quantize(prepacked_weight, ..., is_sf_swizzled_layout=True)), addressed as [224 x 168 rows, 256 bytes].
// alpha = 1 / (act_global_scale * weight_global_scale) rescales the block-scaled accumulator before the BF16
// round of h; the SwiGLU epilogue is the FP8 route's.
void minimax_h3_sm120_nvfp4_fc1_swiglu(TensorView x, TensorView x_norm_weight, TensorView adaln_scale,
                                       TensorView adaln_shift, TensorView adaln_index, TensorView act_global_scale,
                                       TensorView fc1_weight_q, TensorView fc1_scale_tiles, TensorView workspace_q,
                                       TensorView workspace_sf, TensorView out, double eps, double alpha) {
  const CommonArgs args = CheckCommon(x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, out);
  CheckTensor(act_global_scale, "act_global_scale", args.device, dl_float32, {1});
  CheckTensor(fc1_weight_q, "fc1_weight_q", args.device, dl_uint8, {kFc1Rows, kPackedCols});
  TVM_FFI_CHECK(fc1_scale_tiles.device().device_type == kDLCUDA &&
                    fc1_scale_tiles.device().device_id == args.device.device_id, ValueError)
      << "fc1_scale_tiles must be a CUDA tensor on the same device as x";
  TVM_FFI_CHECK(encode_dlpack_dtype(fc1_scale_tiles.dtype()) == encode_dlpack_dtype(dl_uint8), ValueError)
      << "fc1_scale_tiles must be uint8";
  int64_t sf_numel = 1;
  for (int dim = 0; dim < fc1_scale_tiles.ndim(); ++dim) sf_numel *= fc1_scale_tiles.size(dim);
  TVM_FFI_CHECK(sf_numel == kFc1Rows * kHiddenSf && fc1_scale_tiles.IsContiguous(), ValueError)
      << "fc1_scale_tiles must be a contiguous uint8 tensor with 28672 * 336 entries (128x4 swizzled layout)";
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(fc1_scale_tiles.data_ptr()) % 16 == 0, ValueError)
      << "fc1_scale_tiles must be 16-byte aligned";
  CheckTensor(workspace_q, "workspace_q", args.device, dl_uint8, {args.rows, kPackedCols});
  CheckTensor(workspace_sf, "workspace_sf", args.device, dl_uint8, {args.rows, kHiddenSf});
  ffi::CUDADeviceGuard device_guard(args.device.device_id);

  h3_fc1_norm_adaln_quant_nvfp4_sm120a::kernel_h3_norm_adaln_quant_nvfp4<<<dim3(args.plan.quant_grid), dim3(kQuantThreads), kQuantNvfp4SmemBytes, args.stream>>>(
      static_cast<__nv_bfloat16*>(x.data_ptr()), static_cast<__nv_bfloat16*>(x_norm_weight.data_ptr()),
      static_cast<__nv_bfloat16*>(adaln_scale.data_ptr()), static_cast<__nv_bfloat16*>(adaln_shift.data_ptr()),
      static_cast<int*>(adaln_index.data_ptr()), static_cast<unsigned int*>(workspace_q.data_ptr()),
      static_cast<uint8_t*>(workspace_sf.data_ptr()), static_cast<float*>(act_global_scale.data_ptr()),
      static_cast<int>(args.rows), static_cast<float>(eps));
  cudaError_t status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "MiniMax-H3 NVFP4 norm/AdaLN quantization launch failed: " << cudaGetErrorString(status);

  const CUtensorMap a_map = EncodeByteTile(workspace_q.data_ptr(), kPackedCols, args.rows, kBlockKb, kBlockM,
                                           CU_TENSOR_MAP_SWIZZLE_64B, "workspace_q");
  const CUtensorMap b_map = EncodeByteTile(fc1_weight_q.data_ptr(), kPackedCols, kFc1Rows, kBlockKb, kBlockB,
                                           CU_TENSOR_MAP_SWIZZLE_64B, "fc1_weight_q");
  const CUtensorMap sfa_map = EncodeByteTile(workspace_sf.data_ptr(), kHiddenSf, args.rows, kSfaBoxInner, kBlockM,
                                             CU_TENSOR_MAP_SWIZZLE_NONE, "workspace_sf");
  // The 128x4 swizzled weight scales are addressed as [224 scale tiles x 168 rows, 256 bytes].
  const CUtensorMap sfb_map = EncodeByteTile(fc1_scale_tiles.data_ptr(), 256, kNumSfTiles * kSfbRowsPerSfTile, 256,
                                             kSfbTmaRows, CU_TENSOR_MAP_SWIZZLE_NONE, "fc1_scale_tiles");
  LaunchGemm(1, args, a_map, b_map, sfa_map, sfb_map, nullptr, nullptr, alpha, "MiniMax-H3 NVFP4");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_sm120_fp8_fc1_swiglu, minimax_h3_sm120_fp8_fc1_swiglu);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_sm120_nvfp4_fc1_swiglu, minimax_h3_sm120_nvfp4_fc1_swiglu);
// clang-format on
