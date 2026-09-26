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
// MiniMax-H3 experimental NVFP4 (QK^T) / FP8 (PV) non-causal packed-varlen self-attention for SM120
// (GB202: RTX 5090 / RTX PRO 6000 Blackwell), generated from the Cake kernel schedules.  One operator
// call = three PDL-chained launches over packed THD tensors q, k, v [T, H, 128] BF16 (segments from int32 cu_seqlens):
//   1. kv_stats:           per (segment Q tile, head) K channel sums, V channel maxima, max squared K row norm;
//                          the last tile CTA of a (segment, head) folds the K mean, V channel scale (amax / 448)
//                          and the K prescale bound (device-scope arrival counter, self-resetting),
//   3. quantize_nvfp4:     segment-centred, Hadamard-rotated Q / K rows -> E2M1 with one UE4M3 scale
//                          per 16 channels (per-token Q prescale, per-128-key-block K prescale),
//                          transposed / in-chunk-permuted E4M3 V^T tile (per-channel scale),
//   4. attention_nvfp4:    persistent 256-thread CTA per SM, 8 warps x 16 query rows, 128-key K/V
//                          tiles (+ 1 KiB K-scale tiles) through a two-stage TMA ring, FA3-style
//                          ping-pong between the two warp groups, mma.sync m16n8k64
//                          kind::mxf4nvf4.block_scale.scale_vec::4X for QK^T, mma.sync m16n8k32
//                          kind::f8f6f4 for PV, FP32 online softmax with E4M3 probabilities (2^8
//                          exponent bias), BF16 output rows.
// Device code: TMA, ldmatrix, block-scaled mma.sync, mbarrier pipelines, named barriers.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

static_assert(sizeof(CUtensorMap) == 128, "CUDA tensor-map ABI size mismatch");
static_assert(alignof(CUtensorMap) >= 64, "CUDA tensor-map ABI requires at least 64-byte alignment");

namespace h3_varlen_kv_stats_nvfp4_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_VARLEN_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_RED_SUM_OFF 0
#define SMEM_RED_SUM_STAGE_BYTES 8192
#define SMEM_RED_SUM_STRIDE 8192
#define SMEM_RED_MAX_OFF 8192
#define SMEM_RED_MAX_STAGE_BYTES 8192
#define SMEM_RED_MAX_STRIDE 8192
#define SMEM_RED_NORM_OFF 16384
#define SMEM_RED_NORM_STAGE_BYTES 64
#define SMEM_RED_NORM_STRIDE 64
#define SMEM_RED_OFF 16448
#define SMEM_RED_STAGE_BYTES 32
#define SMEM_RED_STRIDE 32
#define SMEM_FLAG_OFF 16480
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_TOTAL 16512
#define THREADS 256

#include <math_constants.h>


__global__ __launch_bounds__(256, 2) void
kernel_minimax_h3_sm120_varlen_kv_stats_nvfp4(__nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, int* __restrict__ tile_table, float* __restrict__ partials, float* __restrict__ knorm_part, unsigned int* __restrict__ counters, float* __restrict__ mean_k, float* __restrict__ v_scale, float* __restrict__ k_scale, int num_tiles, int num_heads)
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
    float* red_sum = reinterpret_cast<float*>(smem_raw + 0);
    const int red_sum_addr = smem + 0;
    float* red_max = reinterpret_cast<float*>(smem_raw + 8192);
    const int red_max_addr = smem + 8192;
    float* red_norm = reinterpret_cast<float*>(smem_raw + 16384);
    const int red_norm_addr = smem + 16384;
    float* red = reinterpret_cast<float*>(smem_raw + 16448);
    const int red_addr = smem + 16448;
    int* flag = reinterpret_cast<int*>(smem_raw + 16480);
    const int flag_addr = smem + 16480;

    // === Task calls (dependency order) ===
    int col = (tid & 15) * 8;
    int rgrp = tid >> 4;
    int chan = tid & 127;
    int is_max = tid >> 7;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int work = bid; work < num_tiles * num_heads; work += num_bids) {
        int tile = work / num_heads;
        int head = work % num_heads;
        int _vec_load_0[4];
        {
            const int4* _ivptr_0 = reinterpret_cast<const int4*>(tile_table + (8 * tile) + 0);
            int4 _ivld_0;
            _ivld_0 = *_ivptr_0;
            _vec_load_0[0 + 0] = _ivld_0.x;
            _vec_load_0[0 + 1] = _ivld_0.y;
            _vec_load_0[0 + 2] = _ivld_0.z;
            _vec_load_0[0 + 3] = _ivld_0.w;
        }
        int _vec_load_1[4];
        {
            const int4* _ivptr_1 = reinterpret_cast<const int4*>(tile_table + (8 * tile + 4) + 0);
            int4 _ivld_1;
            _ivld_1 = *_ivptr_1;
            _vec_load_1[0 + 0] = _ivld_1.x;
            _vec_load_1[0 + 1] = _ivld_1.y;
            _vec_load_1[0 + 2] = _ivld_1.z;
            _vec_load_1[0 + 3] = _ivld_1.w;
        }
        int seg = _vec_load_0[0];
        int row_begin = _vec_load_0[1];
        int row_end = _vec_load_0[2];
        int t0 = _vec_load_0[3];
        int t1 = _vec_load_1[0];
        int length = _vec_load_1[1];
        float ksum[8];
        float vmax[8];
        for (int e = 0; e < 8; e++) {
            ksum[e] = 0.0f;
            vmax[e] = 0.0f;
        }
        float kf8[8];
        float vf8[8];
        float ksq_a[1];
        float knmax_a[1];
        knmax_a[0] = 0.0f;
        #pragma unroll 1
        for (int batch = 0; batch < (row_end - row_begin + 128 - 1) / 128; batch++) {
            float _vec_load_2[8];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + rgrp) ? row_begin + batch * 128 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + rgrp) ? row_begin + batch * 128 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + 16 + rgrp) ? row_begin + batch * 128 + 16 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
            float _vec_load_5[8];
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + 16 + rgrp) ? row_begin + batch * 128 + 16 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_5[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_5[_blk] = _vptr_5[_blk];
                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_5[_pair]));
                    }
                }
            }
            float _vec_load_6[8];
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + 32 + rgrp) ? row_begin + batch * 128 + 32 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_6[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_6[_blk] = _vptr_6[_blk];
                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_6[_pair]));
                    }
                }
            }
            float _vec_load_7[8];
            {
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + 32 + rgrp) ? row_begin + batch * 128 + 32 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_7[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_7[_blk] = _vptr_7[_blk];
                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_7[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_7[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_7[_pair]));
                    }
                }
            }
            float _vec_load_8[8];
            {
                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + 48 + rgrp) ? row_begin + batch * 128 + 48 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_8[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_8[_blk] = _vptr_8[_blk];
                    uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_8[_pair]));
                    }
                }
            }
            float _vec_load_9[8];
            {
                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + 48 + rgrp) ? row_begin + batch * 128 + 48 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_9[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_9[_blk] = _vptr_9[_blk];
                    uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_9[_pair]));
                    }
                }
            }
            float _vec_load_10[8];
            {
                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + 64 + rgrp) ? row_begin + batch * 128 + 64 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_10[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_10[_blk] = _vptr_10[_blk];
                    uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_10[_pair]));
                    }
                }
            }
            float _vec_load_11[8];
            {
                const uint4* _vptr_11 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + 64 + rgrp) ? row_begin + batch * 128 + 64 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_11[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_11[_blk] = _vptr_11[_blk];
                    uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_11[_pair]));
                    }
                }
            }
            float _vec_load_12[8];
            {
                const uint4* _vptr_12 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + 80 + rgrp) ? row_begin + batch * 128 + 80 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_12[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_12[_blk] = _vptr_12[_blk];
                    uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_12[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_12[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_12[_pair]));
                    }
                }
            }
            float _vec_load_13[8];
            {
                const uint4* _vptr_13 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + 80 + rgrp) ? row_begin + batch * 128 + 80 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_13[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_13[_blk] = _vptr_13[_blk];
                    uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_13[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_13[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_13[_pair]));
                    }
                }
            }
            float _vec_load_14[8];
            {
                const uint4* _vptr_14 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + 96 + rgrp) ? row_begin + batch * 128 + 96 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_14[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_14[_blk] = _vptr_14[_blk];
                    uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_14[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_14[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_14[_pair]));
                    }
                }
            }
            float _vec_load_15[8];
            {
                const uint4* _vptr_15 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + 96 + rgrp) ? row_begin + batch * 128 + 96 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_15[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_15[_blk] = _vptr_15[_blk];
                    uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_15[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_15[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_15[_pair]));
                    }
                }
            }
            float _vec_load_16[8];
            {
                const uint4* _vptr_16 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + 112 + rgrp) ? row_begin + batch * 128 + 112 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_16[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_16[_blk] = _vptr_16[_blk];
                    uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_16[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_16[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_16[_pair]));
                    }
                }
            }
            float _vec_load_17[8];
            {
                const uint4* _vptr_17 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + 112 + rgrp) ? row_begin + batch * 128 + 112 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_17[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_17[_blk] = _vptr_17[_blk];
                    uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_17[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_17[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_17[_pair]));
                    }
                }
            }
            kf8[0] = _vec_load_2[0];
            vf8[0] = _vec_load_3[0];
            kf8[1] = _vec_load_2[1];
            vf8[1] = _vec_load_3[1];
            kf8[2] = _vec_load_2[2];
            vf8[2] = _vec_load_3[2];
            kf8[3] = _vec_load_2[3];
            vf8[3] = _vec_load_3[3];
            kf8[4] = _vec_load_2[4];
            vf8[4] = _vec_load_3[4];
            kf8[5] = _vec_load_2[5];
            vf8[5] = _vec_load_3[5];
            kf8[6] = _vec_load_2[6];
            vf8[6] = _vec_load_3[6];
            kf8[7] = _vec_load_2[7];
            vf8[7] = _vec_load_3[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[0] : 0.0f);
            float _fma_0 = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_0;
            float _fabs_0 = fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[0] : 0.0f));
            float _fmax_0 = fmaxf(vmax[0], _fabs_0);
            vmax[0] = _fmax_0;
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[1] : 0.0f);
            float _fma_1 = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_1;
            float _fabs_1 = fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[1] : 0.0f));
            float _fmax_1 = fmaxf(vmax[1], _fabs_1);
            vmax[1] = _fmax_1;
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[2] : 0.0f);
            float _fma_2 = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_2;
            float _fabs_2 = fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[2] : 0.0f));
            float _fmax_2 = fmaxf(vmax[2], _fabs_2);
            vmax[2] = _fmax_2;
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[3] : 0.0f);
            float _fma_3 = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_3;
            float _fabs_3 = fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[3] : 0.0f));
            float _fmax_3 = fmaxf(vmax[3], _fabs_3);
            vmax[3] = _fmax_3;
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[4] : 0.0f);
            float _fma_4 = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_4;
            float _fabs_4 = fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[4] : 0.0f));
            float _fmax_4 = fmaxf(vmax[4], _fabs_4);
            vmax[4] = _fmax_4;
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[5] : 0.0f);
            float _fma_5 = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_5;
            float _fabs_5 = fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[5] : 0.0f));
            float _fmax_5 = fmaxf(vmax[5], _fabs_5);
            vmax[5] = _fmax_5;
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[6] : 0.0f);
            float _fma_6 = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_6;
            float _fabs_6 = fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[6] : 0.0f));
            float _fmax_6 = fmaxf(vmax[6], _fabs_6);
            vmax[6] = _fmax_6;
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[7] : 0.0f);
            float _fma_7 = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_7;
            float _fabs_7 = fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[7] : 0.0f));
            float _fmax_7 = fmaxf(vmax[7], _fabs_7);
            vmax[7] = _fmax_7;
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_0;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_1;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_2;
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_3;
            float _fmax_8 = fmaxf(knmax_a[0], ksq_a[0]);
            knmax_a[0] = _fmax_8;
            kf8[0] = _vec_load_4[0];
            vf8[0] = _vec_load_5[0];
            kf8[1] = _vec_load_4[1];
            vf8[1] = _vec_load_5[1];
            kf8[2] = _vec_load_4[2];
            vf8[2] = _vec_load_5[2];
            kf8[3] = _vec_load_4[3];
            vf8[3] = _vec_load_5[3];
            kf8[4] = _vec_load_4[4];
            vf8[4] = _vec_load_5[4];
            kf8[5] = _vec_load_4[5];
            vf8[5] = _vec_load_5[5];
            kf8[6] = _vec_load_4[6];
            vf8[6] = _vec_load_5[6];
            kf8[7] = _vec_load_4[7];
            vf8[7] = _vec_load_5[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[0] : 0.0f);
            float _fma_8 = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_8;
            float _fabs_8 = fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[0] : 0.0f));
            float _fmax_9 = fmaxf(vmax[0], _fabs_8);
            vmax[0] = _fmax_9;
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[1] : 0.0f);
            float _fma_9 = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_9;
            float _fabs_9 = fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[1] : 0.0f));
            float _fmax_10 = fmaxf(vmax[1], _fabs_9);
            vmax[1] = _fmax_10;
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[2] : 0.0f);
            float _fma_10 = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_10;
            float _fabs_10 = fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[2] : 0.0f));
            float _fmax_11 = fmaxf(vmax[2], _fabs_10);
            vmax[2] = _fmax_11;
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[3] : 0.0f);
            float _fma_11 = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_11;
            float _fabs_11 = fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[3] : 0.0f));
            float _fmax_12 = fmaxf(vmax[3], _fabs_11);
            vmax[3] = _fmax_12;
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[4] : 0.0f);
            float _fma_12 = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_12;
            float _fabs_12 = fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[4] : 0.0f));
            float _fmax_13 = fmaxf(vmax[4], _fabs_12);
            vmax[4] = _fmax_13;
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[5] : 0.0f);
            float _fma_13 = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_13;
            float _fabs_13 = fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[5] : 0.0f));
            float _fmax_14 = fmaxf(vmax[5], _fabs_13);
            vmax[5] = _fmax_14;
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[6] : 0.0f);
            float _fma_14 = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_14;
            float _fabs_14 = fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[6] : 0.0f));
            float _fmax_15 = fmaxf(vmax[6], _fabs_14);
            vmax[6] = _fmax_15;
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[7] : 0.0f);
            float _fma_15 = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_15;
            float _fabs_15 = fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[7] : 0.0f));
            float _fmax_16 = fmaxf(vmax[7], _fabs_15);
            vmax[7] = _fmax_16;
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_4;
            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_5;
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_6;
            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_7;
            float _fmax_17 = fmaxf(knmax_a[0], ksq_a[0]);
            knmax_a[0] = _fmax_17;
            kf8[0] = _vec_load_6[0];
            vf8[0] = _vec_load_7[0];
            kf8[1] = _vec_load_6[1];
            vf8[1] = _vec_load_7[1];
            kf8[2] = _vec_load_6[2];
            vf8[2] = _vec_load_7[2];
            kf8[3] = _vec_load_6[3];
            vf8[3] = _vec_load_7[3];
            kf8[4] = _vec_load_6[4];
            vf8[4] = _vec_load_7[4];
            kf8[5] = _vec_load_6[5];
            vf8[5] = _vec_load_7[5];
            kf8[6] = _vec_load_6[6];
            vf8[6] = _vec_load_7[6];
            kf8[7] = _vec_load_6[7];
            vf8[7] = _vec_load_7[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[0] : 0.0f);
            float _fma_16 = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_16;
            float _fabs_16 = fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[0] : 0.0f));
            float _fmax_18 = fmaxf(vmax[0], _fabs_16);
            vmax[0] = _fmax_18;
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[1] : 0.0f);
            float _fma_17 = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_17;
            float _fabs_17 = fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[1] : 0.0f));
            float _fmax_19 = fmaxf(vmax[1], _fabs_17);
            vmax[1] = _fmax_19;
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[2] : 0.0f);
            float _fma_18 = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_18;
            float _fabs_18 = fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[2] : 0.0f));
            float _fmax_20 = fmaxf(vmax[2], _fabs_18);
            vmax[2] = _fmax_20;
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[3] : 0.0f);
            float _fma_19 = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_19;
            float _fabs_19 = fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[3] : 0.0f));
            float _fmax_21 = fmaxf(vmax[3], _fabs_19);
            vmax[3] = _fmax_21;
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[4] : 0.0f);
            float _fma_20 = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_20;
            float _fabs_20 = fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[4] : 0.0f));
            float _fmax_22 = fmaxf(vmax[4], _fabs_20);
            vmax[4] = _fmax_22;
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[5] : 0.0f);
            float _fma_21 = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_21;
            float _fabs_21 = fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[5] : 0.0f));
            float _fmax_23 = fmaxf(vmax[5], _fabs_21);
            vmax[5] = _fmax_23;
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[6] : 0.0f);
            float _fma_22 = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_22;
            float _fabs_22 = fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[6] : 0.0f));
            float _fmax_24 = fmaxf(vmax[6], _fabs_22);
            vmax[6] = _fmax_24;
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[7] : 0.0f);
            float _fma_23 = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_23;
            float _fabs_23 = fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[7] : 0.0f));
            float _fmax_25 = fmaxf(vmax[7], _fabs_23);
            vmax[7] = _fmax_25;
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_8;
            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_9;
            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_10;
            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_11;
            float _fmax_26 = fmaxf(knmax_a[0], ksq_a[0]);
            knmax_a[0] = _fmax_26;
            kf8[0] = _vec_load_8[0];
            vf8[0] = _vec_load_9[0];
            kf8[1] = _vec_load_8[1];
            vf8[1] = _vec_load_9[1];
            kf8[2] = _vec_load_8[2];
            vf8[2] = _vec_load_9[2];
            kf8[3] = _vec_load_8[3];
            vf8[3] = _vec_load_9[3];
            kf8[4] = _vec_load_8[4];
            vf8[4] = _vec_load_9[4];
            kf8[5] = _vec_load_8[5];
            vf8[5] = _vec_load_9[5];
            kf8[6] = _vec_load_8[6];
            vf8[6] = _vec_load_9[6];
            kf8[7] = _vec_load_8[7];
            vf8[7] = _vec_load_9[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[0] : 0.0f);
            float _fma_24 = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_24;
            float _fabs_24 = fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[0] : 0.0f));
            float _fmax_27 = fmaxf(vmax[0], _fabs_24);
            vmax[0] = _fmax_27;
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[1] : 0.0f);
            float _fma_25 = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_25;
            float _fabs_25 = fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[1] : 0.0f));
            float _fmax_28 = fmaxf(vmax[1], _fabs_25);
            vmax[1] = _fmax_28;
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[2] : 0.0f);
            float _fma_26 = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_26;
            float _fabs_26 = fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[2] : 0.0f));
            float _fmax_29 = fmaxf(vmax[2], _fabs_26);
            vmax[2] = _fmax_29;
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[3] : 0.0f);
            float _fma_27 = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_27;
            float _fabs_27 = fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[3] : 0.0f));
            float _fmax_30 = fmaxf(vmax[3], _fabs_27);
            vmax[3] = _fmax_30;
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[4] : 0.0f);
            float _fma_28 = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_28;
            float _fabs_28 = fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[4] : 0.0f));
            float _fmax_31 = fmaxf(vmax[4], _fabs_28);
            vmax[4] = _fmax_31;
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[5] : 0.0f);
            float _fma_29 = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_29;
            float _fabs_29 = fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[5] : 0.0f));
            float _fmax_32 = fmaxf(vmax[5], _fabs_29);
            vmax[5] = _fmax_32;
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[6] : 0.0f);
            float _fma_30 = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_30;
            float _fabs_30 = fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[6] : 0.0f));
            float _fmax_33 = fmaxf(vmax[6], _fabs_30);
            vmax[6] = _fmax_33;
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[7] : 0.0f);
            float _fma_31 = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_31;
            float _fabs_31 = fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[7] : 0.0f));
            float _fmax_34 = fmaxf(vmax[7], _fabs_31);
            vmax[7] = _fmax_34;
            float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_12;
            float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_13;
            float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_14;
            float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_15;
            float _fmax_35 = fmaxf(knmax_a[0], ksq_a[0]);
            knmax_a[0] = _fmax_35;
            kf8[0] = _vec_load_10[0];
            vf8[0] = _vec_load_11[0];
            kf8[1] = _vec_load_10[1];
            vf8[1] = _vec_load_11[1];
            kf8[2] = _vec_load_10[2];
            vf8[2] = _vec_load_11[2];
            kf8[3] = _vec_load_10[3];
            vf8[3] = _vec_load_11[3];
            kf8[4] = _vec_load_10[4];
            vf8[4] = _vec_load_11[4];
            kf8[5] = _vec_load_10[5];
            vf8[5] = _vec_load_11[5];
            kf8[6] = _vec_load_10[6];
            vf8[6] = _vec_load_11[6];
            kf8[7] = _vec_load_10[7];
            vf8[7] = _vec_load_11[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[0] : 0.0f);
            float _fma_32 = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_32;
            float _fabs_32 = fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[0] : 0.0f));
            float _fmax_36 = fmaxf(vmax[0], _fabs_32);
            vmax[0] = _fmax_36;
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[1] : 0.0f);
            float _fma_33 = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_33;
            float _fabs_33 = fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[1] : 0.0f));
            float _fmax_37 = fmaxf(vmax[1], _fabs_33);
            vmax[1] = _fmax_37;
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[2] : 0.0f);
            float _fma_34 = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_34;
            float _fabs_34 = fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[2] : 0.0f));
            float _fmax_38 = fmaxf(vmax[2], _fabs_34);
            vmax[2] = _fmax_38;
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[3] : 0.0f);
            float _fma_35 = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_35;
            float _fabs_35 = fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[3] : 0.0f));
            float _fmax_39 = fmaxf(vmax[3], _fabs_35);
            vmax[3] = _fmax_39;
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[4] : 0.0f);
            float _fma_36 = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_36;
            float _fabs_36 = fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[4] : 0.0f));
            float _fmax_40 = fmaxf(vmax[4], _fabs_36);
            vmax[4] = _fmax_40;
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[5] : 0.0f);
            float _fma_37 = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_37;
            float _fabs_37 = fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[5] : 0.0f));
            float _fmax_41 = fmaxf(vmax[5], _fabs_37);
            vmax[5] = _fmax_41;
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[6] : 0.0f);
            float _fma_38 = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_38;
            float _fabs_38 = fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[6] : 0.0f));
            float _fmax_42 = fmaxf(vmax[6], _fabs_38);
            vmax[6] = _fmax_42;
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[7] : 0.0f);
            float _fma_39 = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_39;
            float _fabs_39 = fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[7] : 0.0f));
            float _fmax_43 = fmaxf(vmax[7], _fabs_39);
            vmax[7] = _fmax_43;
            float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_16;
            float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_17;
            float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_18;
            float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_19;
            float _fmax_44 = fmaxf(knmax_a[0], ksq_a[0]);
            knmax_a[0] = _fmax_44;
            kf8[0] = _vec_load_12[0];
            vf8[0] = _vec_load_13[0];
            kf8[1] = _vec_load_12[1];
            vf8[1] = _vec_load_13[1];
            kf8[2] = _vec_load_12[2];
            vf8[2] = _vec_load_13[2];
            kf8[3] = _vec_load_12[3];
            vf8[3] = _vec_load_13[3];
            kf8[4] = _vec_load_12[4];
            vf8[4] = _vec_load_13[4];
            kf8[5] = _vec_load_12[5];
            vf8[5] = _vec_load_13[5];
            kf8[6] = _vec_load_12[6];
            vf8[6] = _vec_load_13[6];
            kf8[7] = _vec_load_12[7];
            vf8[7] = _vec_load_13[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[0] : 0.0f);
            float _fma_40 = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_40;
            float _fabs_40 = fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[0] : 0.0f));
            float _fmax_45 = fmaxf(vmax[0], _fabs_40);
            vmax[0] = _fmax_45;
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[1] : 0.0f);
            float _fma_41 = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_41;
            float _fabs_41 = fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[1] : 0.0f));
            float _fmax_46 = fmaxf(vmax[1], _fabs_41);
            vmax[1] = _fmax_46;
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[2] : 0.0f);
            float _fma_42 = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_42;
            float _fabs_42 = fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[2] : 0.0f));
            float _fmax_47 = fmaxf(vmax[2], _fabs_42);
            vmax[2] = _fmax_47;
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[3] : 0.0f);
            float _fma_43 = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_43;
            float _fabs_43 = fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[3] : 0.0f));
            float _fmax_48 = fmaxf(vmax[3], _fabs_43);
            vmax[3] = _fmax_48;
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[4] : 0.0f);
            float _fma_44 = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_44;
            float _fabs_44 = fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[4] : 0.0f));
            float _fmax_49 = fmaxf(vmax[4], _fabs_44);
            vmax[4] = _fmax_49;
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[5] : 0.0f);
            float _fma_45 = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_45;
            float _fabs_45 = fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[5] : 0.0f));
            float _fmax_50 = fmaxf(vmax[5], _fabs_45);
            vmax[5] = _fmax_50;
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[6] : 0.0f);
            float _fma_46 = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_46;
            float _fabs_46 = fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[6] : 0.0f));
            float _fmax_51 = fmaxf(vmax[6], _fabs_46);
            vmax[6] = _fmax_51;
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[7] : 0.0f);
            float _fma_47 = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_47;
            float _fabs_47 = fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[7] : 0.0f));
            float _fmax_52 = fmaxf(vmax[7], _fabs_47);
            vmax[7] = _fmax_52;
            float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_20;
            float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_21;
            float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_22;
            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_23;
            float _fmax_53 = fmaxf(knmax_a[0], ksq_a[0]);
            knmax_a[0] = _fmax_53;
            kf8[0] = _vec_load_14[0];
            vf8[0] = _vec_load_15[0];
            kf8[1] = _vec_load_14[1];
            vf8[1] = _vec_load_15[1];
            kf8[2] = _vec_load_14[2];
            vf8[2] = _vec_load_15[2];
            kf8[3] = _vec_load_14[3];
            vf8[3] = _vec_load_15[3];
            kf8[4] = _vec_load_14[4];
            vf8[4] = _vec_load_15[4];
            kf8[5] = _vec_load_14[5];
            vf8[5] = _vec_load_15[5];
            kf8[6] = _vec_load_14[6];
            vf8[6] = _vec_load_15[6];
            kf8[7] = _vec_load_14[7];
            vf8[7] = _vec_load_15[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[0] : 0.0f);
            float _fma_48 = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_48;
            float _fabs_48 = fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[0] : 0.0f));
            float _fmax_54 = fmaxf(vmax[0], _fabs_48);
            vmax[0] = _fmax_54;
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[1] : 0.0f);
            float _fma_49 = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_49;
            float _fabs_49 = fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[1] : 0.0f));
            float _fmax_55 = fmaxf(vmax[1], _fabs_49);
            vmax[1] = _fmax_55;
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[2] : 0.0f);
            float _fma_50 = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_50;
            float _fabs_50 = fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[2] : 0.0f));
            float _fmax_56 = fmaxf(vmax[2], _fabs_50);
            vmax[2] = _fmax_56;
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[3] : 0.0f);
            float _fma_51 = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_51;
            float _fabs_51 = fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[3] : 0.0f));
            float _fmax_57 = fmaxf(vmax[3], _fabs_51);
            vmax[3] = _fmax_57;
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[4] : 0.0f);
            float _fma_52 = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_52;
            float _fabs_52 = fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[4] : 0.0f));
            float _fmax_58 = fmaxf(vmax[4], _fabs_52);
            vmax[4] = _fmax_58;
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[5] : 0.0f);
            float _fma_53 = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_53;
            float _fabs_53 = fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[5] : 0.0f));
            float _fmax_59 = fmaxf(vmax[5], _fabs_53);
            vmax[5] = _fmax_59;
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[6] : 0.0f);
            float _fma_54 = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_54;
            float _fabs_54 = fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[6] : 0.0f));
            float _fmax_60 = fmaxf(vmax[6], _fabs_54);
            vmax[6] = _fmax_60;
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[7] : 0.0f);
            float _fma_55 = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_55;
            float _fabs_55 = fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[7] : 0.0f));
            float _fmax_61 = fmaxf(vmax[7], _fabs_55);
            vmax[7] = _fmax_61;
            float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_24;
            float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_25;
            float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_26;
            float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_27;
            float _fmax_62 = fmaxf(knmax_a[0], ksq_a[0]);
            knmax_a[0] = _fmax_62;
            kf8[0] = _vec_load_16[0];
            vf8[0] = _vec_load_17[0];
            kf8[1] = _vec_load_16[1];
            vf8[1] = _vec_load_17[1];
            kf8[2] = _vec_load_16[2];
            vf8[2] = _vec_load_17[2];
            kf8[3] = _vec_load_16[3];
            vf8[3] = _vec_load_17[3];
            kf8[4] = _vec_load_16[4];
            vf8[4] = _vec_load_17[4];
            kf8[5] = _vec_load_16[5];
            vf8[5] = _vec_load_17[5];
            kf8[6] = _vec_load_16[6];
            vf8[6] = _vec_load_17[6];
            kf8[7] = _vec_load_16[7];
            vf8[7] = _vec_load_17[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[0] : 0.0f);
            float _fma_56 = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_56;
            float _fabs_56 = fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[0] : 0.0f));
            float _fmax_63 = fmaxf(vmax[0], _fabs_56);
            vmax[0] = _fmax_63;
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[1] : 0.0f);
            float _fma_57 = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_57;
            float _fabs_57 = fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[1] : 0.0f));
            float _fmax_64 = fmaxf(vmax[1], _fabs_57);
            vmax[1] = _fmax_64;
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[2] : 0.0f);
            float _fma_58 = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_58;
            float _fabs_58 = fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[2] : 0.0f));
            float _fmax_65 = fmaxf(vmax[2], _fabs_58);
            vmax[2] = _fmax_65;
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[3] : 0.0f);
            float _fma_59 = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_59;
            float _fabs_59 = fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[3] : 0.0f));
            float _fmax_66 = fmaxf(vmax[3], _fabs_59);
            vmax[3] = _fmax_66;
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[4] : 0.0f);
            float _fma_60 = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_60;
            float _fabs_60 = fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[4] : 0.0f));
            float _fmax_67 = fmaxf(vmax[4], _fabs_60);
            vmax[4] = _fmax_67;
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[5] : 0.0f);
            float _fma_61 = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_61;
            float _fabs_61 = fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[5] : 0.0f));
            float _fmax_68 = fmaxf(vmax[5], _fabs_61);
            vmax[5] = _fmax_68;
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[6] : 0.0f);
            float _fma_62 = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_62;
            float _fabs_62 = fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[6] : 0.0f));
            float _fmax_69 = fmaxf(vmax[6], _fabs_62);
            vmax[6] = _fmax_69;
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[7] : 0.0f);
            float _fma_63 = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            ksq_a[0] = _fma_63;
            float _fabs_63 = fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[7] : 0.0f));
            float _fmax_70 = fmaxf(vmax[7], _fabs_63);
            vmax[7] = _fmax_70;
            float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_28;
            float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_29;
            float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_30;
            float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_31;
            float _fmax_71 = fmaxf(knmax_a[0], ksq_a[0]);
            knmax_a[0] = _fmax_71;
        }
        for (int e_1 = 0; e_1 < 8; e_1++) {
            red_sum[rgrp * 128 + col + e_1] = ksum[e_1];
            red_max[rgrp * 128 + col + e_1] = vmax[e_1];
        }
        if ((tid & 15) == 0) {
            red_norm[rgrp] = knmax_a[0];
        }
        __syncthreads();
        int direct = ((t1 - t0 == 1) ? 1 : 0);
        if (direct == 0) {
            if (tid < 128) {
                float total = 0.0f;
                float vmx = 0.0f;
                for (int g = 0; g < 16; g++) {
                    total = total + red_sum[g * 128 + tid];
                    float _fmax_72 = fmaxf(vmx, red_max[g * 128 + tid]);
                    vmx = _fmax_72;
                }
                *(reinterpret_cast<float*>(partials + (work * 256 + tid)) + (0)) = total;
                *(reinterpret_cast<float*>(partials + (work * 256 + 128 + tid)) + (0)) = vmx;
            }
            if (tid == 0) {
                float kn = 0.0f;
                for (int g_1 = 0; g_1 < 16; g_1++) {
                    float _fmax_73 = fmaxf(kn, red_norm[g_1]);
                    kn = _fmax_73;
                }
                *(reinterpret_cast<float*>(knorm_part + work) + (0)) = kn;
            }
            __syncthreads();
            if (tid == 0) {
                unsigned int _atomic_old_0;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(_atomic_old_0) : "l"(&counters[seg * num_heads + head]), "r"(static_cast<uint32_t>(1)) : "memory");
                unsigned int old = _atomic_old_0;
                int last = (((int)old == t1 - t0 - 1) ? 1 : 0);
                flag[0] = last;
                if (last == 1) {
                    *(reinterpret_cast<unsigned int*>(counters + (seg * num_heads + head)) + (0)) = (unsigned int)0;
                }
            }
            __syncthreads();
        }
        int is_last = ((direct == 0) ? flag[0] : 1);
        if (is_last == 1) {
            float acc_a[1];
            float knf_a[1];
            float fold_v[4];
            float fold_k[4];
            acc_a[0] = 0.0f;
            knf_a[0] = 0.0f;
            if (direct == 1) {
                for (int g_2 = 0; g_2 < 16; g_2++) {
                    float value = ((is_max == 0) ? red_sum[g_2 * 128 + chan] : red_max[g_2 * 128 + chan]);
                    float _fmax_74 = fmaxf(acc_a[0], value);
                    acc_a[0] = ((is_max == 0) ? acc_a[0] + value : _fmax_74);
                    float _fmax_75 = fmaxf(knf_a[0], red_norm[g_2]);
                    knf_a[0] = _fmax_75;
                }
            } else {
                #pragma unroll 1
                for (int tt = t0; tt < t1; tt += 4) {
                    fold_v[0] = partials[(((t1 > tt) ? tt : t1 - 1) * num_heads + head) * 256 + is_max * 128 + chan];
                    fold_k[0] = knorm_part[((t1 > tt) ? tt : t1 - 1) * num_heads + head];
                    fold_v[1] = partials[(((t1 > tt + 1) ? tt + 1 : t1 - 1) * num_heads + head) * 256 + is_max * 128 + chan];
                    fold_k[1] = knorm_part[((t1 > tt + 1) ? tt + 1 : t1 - 1) * num_heads + head];
                    fold_v[2] = partials[(((t1 > tt + 2) ? tt + 2 : t1 - 1) * num_heads + head) * 256 + is_max * 128 + chan];
                    fold_k[2] = knorm_part[((t1 > tt + 2) ? tt + 2 : t1 - 1) * num_heads + head];
                    fold_v[3] = partials[(((t1 > tt + 3) ? tt + 3 : t1 - 1) * num_heads + head) * 256 + is_max * 128 + chan];
                    fold_k[3] = knorm_part[((t1 > tt + 3) ? tt + 3 : t1 - 1) * num_heads + head];
                    float _fmax_76 = fmaxf(acc_a[0], ((t1 > tt) ? fold_v[0] : 0.0f));
                    acc_a[0] = ((is_max == 0) ? acc_a[0] + ((t1 > tt) ? fold_v[0] : 0.0f) : _fmax_76);
                    float _fmax_77 = fmaxf(knf_a[0], ((t1 > tt) ? fold_k[0] : 0.0f));
                    knf_a[0] = _fmax_77;
                    float _fmax_78 = fmaxf(acc_a[0], ((t1 > tt + 1) ? fold_v[1] : 0.0f));
                    acc_a[0] = ((is_max == 0) ? acc_a[0] + ((t1 > tt + 1) ? fold_v[1] : 0.0f) : _fmax_78);
                    float _fmax_79 = fmaxf(knf_a[0], ((t1 > tt + 1) ? fold_k[1] : 0.0f));
                    knf_a[0] = _fmax_79;
                    float _fmax_80 = fmaxf(acc_a[0], ((t1 > tt + 2) ? fold_v[2] : 0.0f));
                    acc_a[0] = ((is_max == 0) ? acc_a[0] + ((t1 > tt + 2) ? fold_v[2] : 0.0f) : _fmax_80);
                    float _fmax_81 = fmaxf(knf_a[0], ((t1 > tt + 2) ? fold_k[2] : 0.0f));
                    knf_a[0] = _fmax_81;
                    float _fmax_82 = fmaxf(acc_a[0], ((t1 > tt + 3) ? fold_v[3] : 0.0f));
                    acc_a[0] = ((is_max == 0) ? acc_a[0] + ((t1 > tt + 3) ? fold_v[3] : 0.0f) : _fmax_82);
                    float _fmax_83 = fmaxf(knf_a[0], ((t1 > tt + 3) ? fold_k[3] : 0.0f));
                    knf_a[0] = _fmax_83;
                }
            }
            float acc = acc_a[0];
            float knf = knf_a[0];
            float lenf = (float)(((length > 0) ? length : 1));
            float _fdiv_rn_0 = __fdiv_rn(acc, lenf);
            float mean_v = _fdiv_rn_0;
            float msq = ((is_max == 0) ? mean_v * mean_v : 0.0f);
            for (int stage = 0; stage < 5; stage++) {
                float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, msq, 16 >> stage);
                msq = msq + _shfl_xor_32;
            }
            if (lane == 0) {
                red[warp] = msq;
            }
            __syncthreads();
            if (length > 0) {
                int sh = seg * num_heads + head;
                if (is_max == 0) {
                    *(reinterpret_cast<float*>(mean_k + (sh * 128 + chan)) + (0)) = mean_v;
                } else {
                    float _fmax_84 = fmaxf(acc, 1e-12f);
                    float _fdiv_rn_1 = __fdiv_rn(_fmax_84, 448.0f);
                    *(reinterpret_cast<float*>(v_scale + (sh * 128 + chan)) + (0)) = _fdiv_rn_1;
                }
                if (tid == 0) {
                    float mnorm2 = red[0] + red[1] + (red[2] + red[3]);
                    float _sqrt_0;
                    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_0) : "f"(knf));
                    float _sqrt_1;
                    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_1) : "f"(mnorm2));
                    float bound = _sqrt_0 + _sqrt_1;
                    float _fmax_85 = fmaxf(bound, 1e-12f);
                    float _fdiv_rn_2 = __fdiv_rn(_fmax_85, 2688.0f);
                    *(reinterpret_cast<float*>(k_scale + sh) + (0)) = _fdiv_rn_2;
                }
            }
        }
        __syncthreads();
    }
}

}  // namespace h3_varlen_kv_stats_nvfp4_sm120a
#undef H3_VARLEN_INF
#undef NUM_MAIN_STAGES
#undef SMEM_FLAG_OFF
#undef SMEM_FLAG_STAGE_BYTES
#undef SMEM_FLAG_STRIDE
#undef SMEM_RED_MAX_OFF
#undef SMEM_RED_MAX_STAGE_BYTES
#undef SMEM_RED_MAX_STRIDE
#undef SMEM_RED_NORM_OFF
#undef SMEM_RED_NORM_STAGE_BYTES
#undef SMEM_RED_NORM_STRIDE
#undef SMEM_RED_OFF
#undef SMEM_RED_STAGE_BYTES
#undef SMEM_RED_STRIDE
#undef SMEM_RED_SUM_OFF
#undef SMEM_RED_SUM_STAGE_BYTES
#undef SMEM_RED_SUM_STRIDE
#undef SMEM_TOTAL
#undef THREADS

namespace h3_varlen_quantize_nvfp4_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_VARLEN_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

#include <math_constants.h>


__global__ __launch_bounds__(256, 2) void
kernel_minimax_h3_sm120_varlen_quantize_nvfp4(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, int* __restrict__ tok_seg, float* __restrict__ mean_k, float* __restrict__ v_scale, unsigned int* __restrict__ q4, unsigned int* __restrict__ k4, uint8_t* __restrict__ q_sf, uint8_t* __restrict__ k_sf, unsigned int* __restrict__ vt8, float* __restrict__ q_scale, float* __restrict__ k_scale, int total_tokens, int total_padded, int num_heads, int num_kblocks)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int sub = tid & 7;
    int trow = tid >> 3;
    int col16 = sub * 16;
    float sgn_hi = (((105 >> sub & 1) == 1) ? -1.0f : 1.0f);
    int num_tblk = total_padded / 32;
    #pragma unroll 1
    for (int work = bid; work < num_tblk * num_heads; work += num_bids) {
        int tblk = work / num_heads;
        int head = work % num_heads;
        int tok_base = tblk * 32;
        int tok = tok_base + trow;
        int valid = ((tok < total_tokens) ? 1 : 0);
        int tok_c = ((tok < total_tokens) ? tok : total_tokens - 1);
        int base = (tok_c * num_heads + head) * 128 + col16;
        float _vec_load_0[16];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(Q + base + 0);
            uint4 _vld_0[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
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
        float _vec_load_1[16];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(K + base + 0);
            uint4 _vld_1[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
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
        int seg = tok_seg[tok_c];
        int ch = trow * 4;
        int chunk = sub >> 2;
        int t8 = sub & 3;
        int vlive[4];
        int vseg[4];
        vlive[0] = ((tok_base + (chunk * 16 + 2 * t8) < total_tokens) ? 1 : 0);
        vseg[0] = tok_seg[((tok_base + (chunk * 16 + 2 * t8) < total_tokens) ? tok_base + (chunk * 16 + 2 * t8) : total_tokens - 1)];
        float _vec_load_2[4];
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(V + ((((tok_base + (chunk * 16 + 2 * t8) < total_tokens) ? tok_base + (chunk * 16 + 2 * t8) : total_tokens - 1) * num_heads + head) * 128 + ch) + 0);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_2[0 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _pair * 2])[1])
                    : "r"(_vpairs_2[_pair]));
            }
        }
        vlive[1] = ((tok_base + (chunk * 16 + 2 * t8 + 1) < total_tokens) ? 1 : 0);
        vseg[1] = tok_seg[((tok_base + (chunk * 16 + 2 * t8 + 1) < total_tokens) ? tok_base + (chunk * 16 + 2 * t8 + 1) : total_tokens - 1)];
        float _vec_load_3[4];
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(V + ((((tok_base + (chunk * 16 + 2 * t8 + 1) < total_tokens) ? tok_base + (chunk * 16 + 2 * t8 + 1) : total_tokens - 1) * num_heads + head) * 128 + ch) + 0);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_3[0 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _pair * 2])[1])
                    : "r"(_vpairs_3[_pair]));
            }
        }
        vlive[2] = ((tok_base + (chunk * 16 + 8 + 2 * t8) < total_tokens) ? 1 : 0);
        vseg[2] = tok_seg[((tok_base + (chunk * 16 + 8 + 2 * t8) < total_tokens) ? tok_base + (chunk * 16 + 8 + 2 * t8) : total_tokens - 1)];
        float _vec_load_4[4];
        {
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(V + ((((tok_base + (chunk * 16 + 8 + 2 * t8) < total_tokens) ? tok_base + (chunk * 16 + 8 + 2 * t8) : total_tokens - 1) * num_heads + head) * 128 + ch) + 0);
            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_4[0 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _pair * 2])[1])
                    : "r"(_vpairs_4[_pair]));
            }
        }
        vlive[3] = ((tok_base + (chunk * 16 + 8 + 2 * t8 + 1) < total_tokens) ? 1 : 0);
        vseg[3] = tok_seg[((tok_base + (chunk * 16 + 8 + 2 * t8 + 1) < total_tokens) ? tok_base + (chunk * 16 + 8 + 2 * t8 + 1) : total_tokens - 1)];
        float _vec_load_5[4];
        {
            uint2 _vld_5;
            _vld_5 = *reinterpret_cast<const uint2*>(V + ((((tok_base + (chunk * 16 + 8 + 2 * t8 + 1) < total_tokens) ? tok_base + (chunk * 16 + 8 + 2 * t8 + 1) : total_tokens - 1) * num_heads + head) * 128 + ch) + 0);
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_5[0 + _pair * 2])[0]), "=f"((&_vec_load_5[0 + _pair * 2])[1])
                    : "r"(_vpairs_5[_pair]));
            }
        }
        __syncthreads();
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        float tmp[16];
        float qf[16];
        for (int e = 0; e < 16; e++) {
            qf[e] = _vec_load_0[e];
        }
        tmp[0] = qf[0] + qf[1];
        tmp[1] = qf[0] - qf[1];
        tmp[2] = qf[2] + qf[3];
        tmp[3] = qf[2] - qf[3];
        tmp[4] = qf[4] + qf[5];
        tmp[5] = qf[4] - qf[5];
        tmp[6] = qf[6] + qf[7];
        tmp[7] = qf[6] - qf[7];
        tmp[8] = qf[8] + qf[9];
        tmp[9] = qf[8] - qf[9];
        tmp[10] = qf[10] + qf[11];
        tmp[11] = qf[10] - qf[11];
        tmp[12] = qf[12] + qf[13];
        tmp[13] = qf[12] - qf[13];
        tmp[14] = qf[14] + qf[15];
        tmp[15] = qf[14] - qf[15];
        qf[0] = tmp[0];
        qf[1] = tmp[1];
        qf[2] = tmp[2];
        qf[3] = tmp[3];
        qf[4] = tmp[4];
        qf[5] = tmp[5];
        qf[6] = tmp[6];
        qf[7] = tmp[7];
        qf[8] = tmp[8];
        qf[9] = tmp[9];
        qf[10] = tmp[10];
        qf[11] = tmp[11];
        qf[12] = tmp[12];
        qf[13] = tmp[13];
        qf[14] = tmp[14];
        qf[15] = tmp[15];
        tmp[0] = qf[0] + qf[2];
        tmp[1] = qf[1] + qf[3];
        tmp[2] = qf[0] - qf[2];
        tmp[3] = qf[1] - qf[3];
        tmp[4] = qf[4] + qf[6];
        tmp[5] = qf[5] + qf[7];
        tmp[6] = qf[4] - qf[6];
        tmp[7] = qf[5] - qf[7];
        tmp[8] = qf[8] + qf[10];
        tmp[9] = qf[9] + qf[11];
        tmp[10] = qf[8] - qf[10];
        tmp[11] = qf[9] - qf[11];
        tmp[12] = qf[12] + qf[14];
        tmp[13] = qf[13] + qf[15];
        tmp[14] = qf[12] - qf[14];
        tmp[15] = qf[13] - qf[15];
        qf[0] = tmp[0];
        qf[1] = tmp[1];
        qf[2] = tmp[2];
        qf[3] = tmp[3];
        qf[4] = tmp[4];
        qf[5] = tmp[5];
        qf[6] = tmp[6];
        qf[7] = tmp[7];
        qf[8] = tmp[8];
        qf[9] = tmp[9];
        qf[10] = tmp[10];
        qf[11] = tmp[11];
        qf[12] = tmp[12];
        qf[13] = tmp[13];
        qf[14] = tmp[14];
        qf[15] = tmp[15];
        tmp[0] = qf[0] + qf[4];
        tmp[1] = qf[1] + qf[5];
        tmp[2] = qf[2] + qf[6];
        tmp[3] = qf[3] + qf[7];
        tmp[4] = qf[0] - qf[4];
        tmp[5] = qf[1] - qf[5];
        tmp[6] = qf[2] - qf[6];
        tmp[7] = qf[3] - qf[7];
        tmp[8] = qf[8] + qf[12];
        tmp[9] = qf[9] + qf[13];
        tmp[10] = qf[10] + qf[14];
        tmp[11] = qf[11] + qf[15];
        tmp[12] = qf[8] - qf[12];
        tmp[13] = qf[9] - qf[13];
        tmp[14] = qf[10] - qf[14];
        tmp[15] = qf[11] - qf[15];
        qf[0] = tmp[0];
        qf[1] = tmp[1];
        qf[2] = tmp[2];
        qf[3] = tmp[3];
        qf[4] = tmp[4];
        qf[5] = tmp[5];
        qf[6] = tmp[6];
        qf[7] = tmp[7];
        qf[8] = tmp[8];
        qf[9] = tmp[9];
        qf[10] = tmp[10];
        qf[11] = tmp[11];
        qf[12] = tmp[12];
        qf[13] = tmp[13];
        qf[14] = tmp[14];
        qf[15] = tmp[15];
        tmp[0] = qf[0] + qf[8];
        tmp[1] = qf[1] + qf[9];
        tmp[2] = qf[2] + qf[10];
        tmp[3] = qf[3] + qf[11];
        tmp[4] = qf[4] + qf[12];
        tmp[5] = qf[5] + qf[13];
        tmp[6] = qf[6] + qf[14];
        tmp[7] = qf[7] + qf[15];
        tmp[8] = qf[0] - qf[8];
        tmp[9] = qf[1] - qf[9];
        tmp[10] = qf[2] - qf[10];
        tmp[11] = qf[3] - qf[11];
        tmp[12] = qf[4] - qf[12];
        tmp[13] = qf[5] - qf[13];
        tmp[14] = qf[6] - qf[14];
        tmp[15] = qf[7] - qf[15];
        qf[0] = tmp[0];
        qf[1] = tmp[1];
        qf[2] = tmp[2];
        qf[3] = tmp[3];
        qf[4] = tmp[4];
        qf[5] = tmp[5];
        qf[6] = tmp[6];
        qf[7] = tmp[7];
        qf[8] = tmp[8];
        qf[9] = tmp[9];
        qf[10] = tmp[10];
        qf[11] = tmp[11];
        qf[12] = tmp[12];
        qf[13] = tmp[13];
        qf[14] = tmp[14];
        qf[15] = tmp[15];
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, qf[0], 1);
        tmp[0] = _shfl_xor_0;
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, qf[1], 1);
        tmp[1] = _shfl_xor_1;
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, qf[2], 1);
        tmp[2] = _shfl_xor_2;
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, qf[3], 1);
        tmp[3] = _shfl_xor_3;
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, qf[4], 1);
        tmp[4] = _shfl_xor_4;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, qf[5], 1);
        tmp[5] = _shfl_xor_5;
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, qf[6], 1);
        tmp[6] = _shfl_xor_6;
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, qf[7], 1);
        tmp[7] = _shfl_xor_7;
        float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, qf[8], 1);
        tmp[8] = _shfl_xor_8;
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, qf[9], 1);
        tmp[9] = _shfl_xor_9;
        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, qf[10], 1);
        tmp[10] = _shfl_xor_10;
        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, qf[11], 1);
        tmp[11] = _shfl_xor_11;
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, qf[12], 1);
        tmp[12] = _shfl_xor_12;
        float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, qf[13], 1);
        tmp[13] = _shfl_xor_13;
        float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, qf[14], 1);
        tmp[14] = _shfl_xor_14;
        float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, qf[15], 1);
        tmp[15] = _shfl_xor_15;
        qf[0] = (((sub & 1) == 1) ? tmp[0] - qf[0] : qf[0] + tmp[0]);
        qf[1] = (((sub & 1) == 1) ? tmp[1] - qf[1] : qf[1] + tmp[1]);
        qf[2] = (((sub & 1) == 1) ? tmp[2] - qf[2] : qf[2] + tmp[2]);
        qf[3] = (((sub & 1) == 1) ? tmp[3] - qf[3] : qf[3] + tmp[3]);
        qf[4] = (((sub & 1) == 1) ? tmp[4] - qf[4] : qf[4] + tmp[4]);
        qf[5] = (((sub & 1) == 1) ? tmp[5] - qf[5] : qf[5] + tmp[5]);
        qf[6] = (((sub & 1) == 1) ? tmp[6] - qf[6] : qf[6] + tmp[6]);
        qf[7] = (((sub & 1) == 1) ? tmp[7] - qf[7] : qf[7] + tmp[7]);
        qf[8] = (((sub & 1) == 1) ? tmp[8] - qf[8] : qf[8] + tmp[8]);
        qf[9] = (((sub & 1) == 1) ? tmp[9] - qf[9] : qf[9] + tmp[9]);
        qf[10] = (((sub & 1) == 1) ? tmp[10] - qf[10] : qf[10] + tmp[10]);
        qf[11] = (((sub & 1) == 1) ? tmp[11] - qf[11] : qf[11] + tmp[11]);
        qf[12] = (((sub & 1) == 1) ? tmp[12] - qf[12] : qf[12] + tmp[12]);
        qf[13] = (((sub & 1) == 1) ? tmp[13] - qf[13] : qf[13] + tmp[13]);
        qf[14] = (((sub & 1) == 1) ? tmp[14] - qf[14] : qf[14] + tmp[14]);
        qf[15] = (((sub & 1) == 1) ? tmp[15] - qf[15] : qf[15] + tmp[15]);
        float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, qf[0], 2);
        tmp[0] = _shfl_xor_16;
        float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, qf[1], 2);
        tmp[1] = _shfl_xor_17;
        float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, qf[2], 2);
        tmp[2] = _shfl_xor_18;
        float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, qf[3], 2);
        tmp[3] = _shfl_xor_19;
        float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, qf[4], 2);
        tmp[4] = _shfl_xor_20;
        float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, qf[5], 2);
        tmp[5] = _shfl_xor_21;
        float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, qf[6], 2);
        tmp[6] = _shfl_xor_22;
        float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, qf[7], 2);
        tmp[7] = _shfl_xor_23;
        float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, qf[8], 2);
        tmp[8] = _shfl_xor_24;
        float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, qf[9], 2);
        tmp[9] = _shfl_xor_25;
        float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, qf[10], 2);
        tmp[10] = _shfl_xor_26;
        float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, qf[11], 2);
        tmp[11] = _shfl_xor_27;
        float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, qf[12], 2);
        tmp[12] = _shfl_xor_28;
        float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, qf[13], 2);
        tmp[13] = _shfl_xor_29;
        float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, qf[14], 2);
        tmp[14] = _shfl_xor_30;
        float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, qf[15], 2);
        tmp[15] = _shfl_xor_31;
        qf[0] = (((sub >> 1 & 1) == 1) ? tmp[0] - qf[0] : qf[0] + tmp[0]);
        qf[1] = (((sub >> 1 & 1) == 1) ? tmp[1] - qf[1] : qf[1] + tmp[1]);
        qf[2] = (((sub >> 1 & 1) == 1) ? tmp[2] - qf[2] : qf[2] + tmp[2]);
        qf[3] = (((sub >> 1 & 1) == 1) ? tmp[3] - qf[3] : qf[3] + tmp[3]);
        qf[4] = (((sub >> 1 & 1) == 1) ? tmp[4] - qf[4] : qf[4] + tmp[4]);
        qf[5] = (((sub >> 1 & 1) == 1) ? tmp[5] - qf[5] : qf[5] + tmp[5]);
        qf[6] = (((sub >> 1 & 1) == 1) ? tmp[6] - qf[6] : qf[6] + tmp[6]);
        qf[7] = (((sub >> 1 & 1) == 1) ? tmp[7] - qf[7] : qf[7] + tmp[7]);
        qf[8] = (((sub >> 1 & 1) == 1) ? tmp[8] - qf[8] : qf[8] + tmp[8]);
        qf[9] = (((sub >> 1 & 1) == 1) ? tmp[9] - qf[9] : qf[9] + tmp[9]);
        qf[10] = (((sub >> 1 & 1) == 1) ? tmp[10] - qf[10] : qf[10] + tmp[10]);
        qf[11] = (((sub >> 1 & 1) == 1) ? tmp[11] - qf[11] : qf[11] + tmp[11]);
        qf[12] = (((sub >> 1 & 1) == 1) ? tmp[12] - qf[12] : qf[12] + tmp[12]);
        qf[13] = (((sub >> 1 & 1) == 1) ? tmp[13] - qf[13] : qf[13] + tmp[13]);
        qf[14] = (((sub >> 1 & 1) == 1) ? tmp[14] - qf[14] : qf[14] + tmp[14]);
        qf[15] = (((sub >> 1 & 1) == 1) ? tmp[15] - qf[15] : qf[15] + tmp[15]);
        float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, qf[0], 4);
        tmp[0] = _shfl_xor_32;
        float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, qf[1], 4);
        tmp[1] = _shfl_xor_33;
        float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, qf[2], 4);
        tmp[2] = _shfl_xor_34;
        float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, qf[3], 4);
        tmp[3] = _shfl_xor_35;
        float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, qf[4], 4);
        tmp[4] = _shfl_xor_36;
        float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, qf[5], 4);
        tmp[5] = _shfl_xor_37;
        float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, qf[6], 4);
        tmp[6] = _shfl_xor_38;
        float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, qf[7], 4);
        tmp[7] = _shfl_xor_39;
        float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, qf[8], 4);
        tmp[8] = _shfl_xor_40;
        float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, qf[9], 4);
        tmp[9] = _shfl_xor_41;
        float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, qf[10], 4);
        tmp[10] = _shfl_xor_42;
        float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, qf[11], 4);
        tmp[11] = _shfl_xor_43;
        float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, qf[12], 4);
        tmp[12] = _shfl_xor_44;
        float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, qf[13], 4);
        tmp[13] = _shfl_xor_45;
        float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, qf[14], 4);
        tmp[14] = _shfl_xor_46;
        float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, qf[15], 4);
        tmp[15] = _shfl_xor_47;
        qf[0] = (((sub >> 2 & 1) == 1) ? tmp[0] - qf[0] : qf[0] + tmp[0]);
        qf[1] = (((sub >> 2 & 1) == 1) ? tmp[1] - qf[1] : qf[1] + tmp[1]);
        qf[2] = (((sub >> 2 & 1) == 1) ? tmp[2] - qf[2] : qf[2] + tmp[2]);
        qf[3] = (((sub >> 2 & 1) == 1) ? tmp[3] - qf[3] : qf[3] + tmp[3]);
        qf[4] = (((sub >> 2 & 1) == 1) ? tmp[4] - qf[4] : qf[4] + tmp[4]);
        qf[5] = (((sub >> 2 & 1) == 1) ? tmp[5] - qf[5] : qf[5] + tmp[5]);
        qf[6] = (((sub >> 2 & 1) == 1) ? tmp[6] - qf[6] : qf[6] + tmp[6]);
        qf[7] = (((sub >> 2 & 1) == 1) ? tmp[7] - qf[7] : qf[7] + tmp[7]);
        qf[8] = (((sub >> 2 & 1) == 1) ? tmp[8] - qf[8] : qf[8] + tmp[8]);
        qf[9] = (((sub >> 2 & 1) == 1) ? tmp[9] - qf[9] : qf[9] + tmp[9]);
        qf[10] = (((sub >> 2 & 1) == 1) ? tmp[10] - qf[10] : qf[10] + tmp[10]);
        qf[11] = (((sub >> 2 & 1) == 1) ? tmp[11] - qf[11] : qf[11] + tmp[11]);
        qf[12] = (((sub >> 2 & 1) == 1) ? tmp[12] - qf[12] : qf[12] + tmp[12]);
        qf[13] = (((sub >> 2 & 1) == 1) ? tmp[13] - qf[13] : qf[13] + tmp[13]);
        qf[14] = (((sub >> 2 & 1) == 1) ? tmp[14] - qf[14] : qf[14] + tmp[14]);
        qf[15] = (((sub >> 2 & 1) == 1) ? tmp[15] - qf[15] : qf[15] + tmp[15]);
        qf[0] = qf[0] * (sgn_hi * 0.08838834764831843f);
        qf[1] = qf[1] * (sgn_hi * -0.08838834764831843f);
        qf[2] = qf[2] * (sgn_hi * 0.08838834764831843f);
        qf[3] = qf[3] * (sgn_hi * 0.08838834764831843f);
        qf[4] = qf[4] * (sgn_hi * -0.08838834764831843f);
        qf[5] = qf[5] * (sgn_hi * 0.08838834764831843f);
        qf[6] = qf[6] * (sgn_hi * -0.08838834764831843f);
        qf[7] = qf[7] * (sgn_hi * -0.08838834764831843f);
        qf[8] = qf[8] * (sgn_hi * 0.08838834764831843f);
        qf[9] = qf[9] * (sgn_hi * 0.08838834764831843f);
        qf[10] = qf[10] * (sgn_hi * -0.08838834764831843f);
        qf[11] = qf[11] * (sgn_hi * 0.08838834764831843f);
        qf[12] = qf[12] * (sgn_hi * -0.08838834764831843f);
        qf[13] = qf[13] * (sgn_hi * -0.08838834764831843f);
        qf[14] = qf[14] * (sgn_hi * 0.08838834764831843f);
        qf[15] = qf[15] * (sgn_hi * -0.08838834764831843f);
        float qamax = 0.0f;
        float bmax = 0.0f;
        for (int e_1 = 0; e_1 < 16; e_1++) {
            float _fabs_0 = fabsf(qf[e_1]);
            float _fmax_0 = fmaxf(bmax, _fabs_0);
            bmax = _fmax_0;
        }
        float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, bmax, 4);
        float _fmax_1 = fmaxf(bmax, _shfl_xor_48);
        qamax = _fmax_1;
        float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, qamax, 2);
        float _fmax_2 = fmaxf(qamax, _shfl_xor_49);
        qamax = _fmax_2;
        float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, qamax, 1);
        float _fmax_3 = fmaxf(qamax, _shfl_xor_50);
        qamax = _fmax_3;
        float _fmax_4 = fmaxf(qamax, 1e-12f);
        qamax = _fmax_4;
        float _fdiv_rn_0 = __fdiv_rn(qamax, 2688.0f);
        float qs = _fdiv_rn_0;
        float _fdiv_rn_1 = __fdiv_rn(2688.0f, qamax);
        float qg = _fdiv_rn_1;
        float qsf_val = bmax * qg * 0.16666666666666666f;
        uint16_t _e4m3x2_f32_0;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(0.0f), "f"(qsf_val));
        uint16_t _e4m3x2_decode_6 = (uint16_t)((unsigned int)_e4m3x2_f32_0 & 0xFFu);
        uint32_t _f16x2_decode_6;
        float _fp8_decode_0;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_6) : "h"(_e4m3x2_decode_6));
        uint16_t _f16_decode_6 = (uint16_t)_f16x2_decode_6;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_0) : "h"(_f16_decode_6));
        float qsf_dec = _fp8_decode_0;
        float _fdiv_rn_2 = __fdiv_rn(qg, qsf_dec);
        float qout = ((qsf_dec != 0.0f) ? _fdiv_rn_2 : 0.0f);
        float qn[16];
        for (int e_2 = 0; e_2 < 16; e_2++) {
            qn[e_2] = qf[e_2] * qout;
        }
        unsigned int qw[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[0]) : "f"(qn[0]), "f"(qn[1]), "f"(qn[2]), "f"(qn[3]), "f"(qn[4]), "f"(qn[5]), "f"(qn[6]), "f"(qn[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[1]) : "f"(qn[8]), "f"(qn[9]), "f"(qn[10]), "f"(qn[11]), "f"(qn[12]), "f"(qn[13]), "f"(qn[14]), "f"(qn[15]));
        unsigned int qstore[4];
        qstore[0] = qw[0];
        qstore[1] = qw[1];
        unsigned int _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, qw[0], 1);
        qstore[2] = _shfl_xor_51;
        unsigned int _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, qw[1], 1);
        qstore[3] = _shfl_xor_52;
        if (valid == 1) {
            {
                unsigned short _sf_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(qsf_val));
                *(reinterpret_cast<unsigned char*>(q_sf + ((tok * num_heads + head) * 8 + sub)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
            }
            if ((sub & 1) == 0) {
                reinterpret_cast<int4*>(q4 + (((tok * num_heads + head) * 64 + sub * 8) / 4))[0] = reinterpret_cast<int4*>(qstore)[0];
            }
            if (sub == 0) {
                *(reinterpret_cast<float*>(q_scale + (tok * num_heads + head)) + (0)) = qs;
            }
        }
        int mbase = (seg * num_heads + head) * 128 + col16;
        float kf16[16];
        for (int quad = 0; quad < 4; quad++) {
            float _vec_load_6[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(mean_k + (mbase + quad * 4) + 0);
                _vec_load_6[0 + 0] = _v4.x;
                _vec_load_6[0 + 1] = _v4.y;
                _vec_load_6[0 + 2] = _v4.z;
                _vec_load_6[0 + 3] = _v4.w;
            }
            for (int e_3 = 0; e_3 < 4; e_3++) {
                float kf = _vec_load_1[quad * 4 + e_3];
                float kc = kf - _vec_load_6[e_3];
                kf16[quad * 4 + e_3] = ((valid == 1) ? kc : 0.0f);
            }
        }
        tmp[0] = kf16[0] + kf16[1];
        tmp[1] = kf16[0] - kf16[1];
        tmp[2] = kf16[2] + kf16[3];
        tmp[3] = kf16[2] - kf16[3];
        tmp[4] = kf16[4] + kf16[5];
        tmp[5] = kf16[4] - kf16[5];
        tmp[6] = kf16[6] + kf16[7];
        tmp[7] = kf16[6] - kf16[7];
        tmp[8] = kf16[8] + kf16[9];
        tmp[9] = kf16[8] - kf16[9];
        tmp[10] = kf16[10] + kf16[11];
        tmp[11] = kf16[10] - kf16[11];
        tmp[12] = kf16[12] + kf16[13];
        tmp[13] = kf16[12] - kf16[13];
        tmp[14] = kf16[14] + kf16[15];
        tmp[15] = kf16[14] - kf16[15];
        kf16[0] = tmp[0];
        kf16[1] = tmp[1];
        kf16[2] = tmp[2];
        kf16[3] = tmp[3];
        kf16[4] = tmp[4];
        kf16[5] = tmp[5];
        kf16[6] = tmp[6];
        kf16[7] = tmp[7];
        kf16[8] = tmp[8];
        kf16[9] = tmp[9];
        kf16[10] = tmp[10];
        kf16[11] = tmp[11];
        kf16[12] = tmp[12];
        kf16[13] = tmp[13];
        kf16[14] = tmp[14];
        kf16[15] = tmp[15];
        tmp[0] = kf16[0] + kf16[2];
        tmp[1] = kf16[1] + kf16[3];
        tmp[2] = kf16[0] - kf16[2];
        tmp[3] = kf16[1] - kf16[3];
        tmp[4] = kf16[4] + kf16[6];
        tmp[5] = kf16[5] + kf16[7];
        tmp[6] = kf16[4] - kf16[6];
        tmp[7] = kf16[5] - kf16[7];
        tmp[8] = kf16[8] + kf16[10];
        tmp[9] = kf16[9] + kf16[11];
        tmp[10] = kf16[8] - kf16[10];
        tmp[11] = kf16[9] - kf16[11];
        tmp[12] = kf16[12] + kf16[14];
        tmp[13] = kf16[13] + kf16[15];
        tmp[14] = kf16[12] - kf16[14];
        tmp[15] = kf16[13] - kf16[15];
        kf16[0] = tmp[0];
        kf16[1] = tmp[1];
        kf16[2] = tmp[2];
        kf16[3] = tmp[3];
        kf16[4] = tmp[4];
        kf16[5] = tmp[5];
        kf16[6] = tmp[6];
        kf16[7] = tmp[7];
        kf16[8] = tmp[8];
        kf16[9] = tmp[9];
        kf16[10] = tmp[10];
        kf16[11] = tmp[11];
        kf16[12] = tmp[12];
        kf16[13] = tmp[13];
        kf16[14] = tmp[14];
        kf16[15] = tmp[15];
        tmp[0] = kf16[0] + kf16[4];
        tmp[1] = kf16[1] + kf16[5];
        tmp[2] = kf16[2] + kf16[6];
        tmp[3] = kf16[3] + kf16[7];
        tmp[4] = kf16[0] - kf16[4];
        tmp[5] = kf16[1] - kf16[5];
        tmp[6] = kf16[2] - kf16[6];
        tmp[7] = kf16[3] - kf16[7];
        tmp[8] = kf16[8] + kf16[12];
        tmp[9] = kf16[9] + kf16[13];
        tmp[10] = kf16[10] + kf16[14];
        tmp[11] = kf16[11] + kf16[15];
        tmp[12] = kf16[8] - kf16[12];
        tmp[13] = kf16[9] - kf16[13];
        tmp[14] = kf16[10] - kf16[14];
        tmp[15] = kf16[11] - kf16[15];
        kf16[0] = tmp[0];
        kf16[1] = tmp[1];
        kf16[2] = tmp[2];
        kf16[3] = tmp[3];
        kf16[4] = tmp[4];
        kf16[5] = tmp[5];
        kf16[6] = tmp[6];
        kf16[7] = tmp[7];
        kf16[8] = tmp[8];
        kf16[9] = tmp[9];
        kf16[10] = tmp[10];
        kf16[11] = tmp[11];
        kf16[12] = tmp[12];
        kf16[13] = tmp[13];
        kf16[14] = tmp[14];
        kf16[15] = tmp[15];
        tmp[0] = kf16[0] + kf16[8];
        tmp[1] = kf16[1] + kf16[9];
        tmp[2] = kf16[2] + kf16[10];
        tmp[3] = kf16[3] + kf16[11];
        tmp[4] = kf16[4] + kf16[12];
        tmp[5] = kf16[5] + kf16[13];
        tmp[6] = kf16[6] + kf16[14];
        tmp[7] = kf16[7] + kf16[15];
        tmp[8] = kf16[0] - kf16[8];
        tmp[9] = kf16[1] - kf16[9];
        tmp[10] = kf16[2] - kf16[10];
        tmp[11] = kf16[3] - kf16[11];
        tmp[12] = kf16[4] - kf16[12];
        tmp[13] = kf16[5] - kf16[13];
        tmp[14] = kf16[6] - kf16[14];
        tmp[15] = kf16[7] - kf16[15];
        kf16[0] = tmp[0];
        kf16[1] = tmp[1];
        kf16[2] = tmp[2];
        kf16[3] = tmp[3];
        kf16[4] = tmp[4];
        kf16[5] = tmp[5];
        kf16[6] = tmp[6];
        kf16[7] = tmp[7];
        kf16[8] = tmp[8];
        kf16[9] = tmp[9];
        kf16[10] = tmp[10];
        kf16[11] = tmp[11];
        kf16[12] = tmp[12];
        kf16[13] = tmp[13];
        kf16[14] = tmp[14];
        kf16[15] = tmp[15];
        float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, kf16[0], 1);
        tmp[0] = _shfl_xor_53;
        float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, kf16[1], 1);
        tmp[1] = _shfl_xor_54;
        float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, kf16[2], 1);
        tmp[2] = _shfl_xor_55;
        float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, kf16[3], 1);
        tmp[3] = _shfl_xor_56;
        float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, kf16[4], 1);
        tmp[4] = _shfl_xor_57;
        float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, kf16[5], 1);
        tmp[5] = _shfl_xor_58;
        float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, kf16[6], 1);
        tmp[6] = _shfl_xor_59;
        float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, kf16[7], 1);
        tmp[7] = _shfl_xor_60;
        float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, kf16[8], 1);
        tmp[8] = _shfl_xor_61;
        float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, kf16[9], 1);
        tmp[9] = _shfl_xor_62;
        float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, kf16[10], 1);
        tmp[10] = _shfl_xor_63;
        float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, kf16[11], 1);
        tmp[11] = _shfl_xor_64;
        float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, kf16[12], 1);
        tmp[12] = _shfl_xor_65;
        float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, kf16[13], 1);
        tmp[13] = _shfl_xor_66;
        float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, kf16[14], 1);
        tmp[14] = _shfl_xor_67;
        float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, kf16[15], 1);
        tmp[15] = _shfl_xor_68;
        kf16[0] = (((sub & 1) == 1) ? tmp[0] - kf16[0] : kf16[0] + tmp[0]);
        kf16[1] = (((sub & 1) == 1) ? tmp[1] - kf16[1] : kf16[1] + tmp[1]);
        kf16[2] = (((sub & 1) == 1) ? tmp[2] - kf16[2] : kf16[2] + tmp[2]);
        kf16[3] = (((sub & 1) == 1) ? tmp[3] - kf16[3] : kf16[3] + tmp[3]);
        kf16[4] = (((sub & 1) == 1) ? tmp[4] - kf16[4] : kf16[4] + tmp[4]);
        kf16[5] = (((sub & 1) == 1) ? tmp[5] - kf16[5] : kf16[5] + tmp[5]);
        kf16[6] = (((sub & 1) == 1) ? tmp[6] - kf16[6] : kf16[6] + tmp[6]);
        kf16[7] = (((sub & 1) == 1) ? tmp[7] - kf16[7] : kf16[7] + tmp[7]);
        kf16[8] = (((sub & 1) == 1) ? tmp[8] - kf16[8] : kf16[8] + tmp[8]);
        kf16[9] = (((sub & 1) == 1) ? tmp[9] - kf16[9] : kf16[9] + tmp[9]);
        kf16[10] = (((sub & 1) == 1) ? tmp[10] - kf16[10] : kf16[10] + tmp[10]);
        kf16[11] = (((sub & 1) == 1) ? tmp[11] - kf16[11] : kf16[11] + tmp[11]);
        kf16[12] = (((sub & 1) == 1) ? tmp[12] - kf16[12] : kf16[12] + tmp[12]);
        kf16[13] = (((sub & 1) == 1) ? tmp[13] - kf16[13] : kf16[13] + tmp[13]);
        kf16[14] = (((sub & 1) == 1) ? tmp[14] - kf16[14] : kf16[14] + tmp[14]);
        kf16[15] = (((sub & 1) == 1) ? tmp[15] - kf16[15] : kf16[15] + tmp[15]);
        float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, kf16[0], 2);
        tmp[0] = _shfl_xor_69;
        float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, kf16[1], 2);
        tmp[1] = _shfl_xor_70;
        float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, kf16[2], 2);
        tmp[2] = _shfl_xor_71;
        float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, kf16[3], 2);
        tmp[3] = _shfl_xor_72;
        float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, kf16[4], 2);
        tmp[4] = _shfl_xor_73;
        float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, kf16[5], 2);
        tmp[5] = _shfl_xor_74;
        float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, kf16[6], 2);
        tmp[6] = _shfl_xor_75;
        float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, kf16[7], 2);
        tmp[7] = _shfl_xor_76;
        float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, kf16[8], 2);
        tmp[8] = _shfl_xor_77;
        float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, kf16[9], 2);
        tmp[9] = _shfl_xor_78;
        float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, kf16[10], 2);
        tmp[10] = _shfl_xor_79;
        float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, kf16[11], 2);
        tmp[11] = _shfl_xor_80;
        float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, kf16[12], 2);
        tmp[12] = _shfl_xor_81;
        float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, kf16[13], 2);
        tmp[13] = _shfl_xor_82;
        float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, kf16[14], 2);
        tmp[14] = _shfl_xor_83;
        float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, kf16[15], 2);
        tmp[15] = _shfl_xor_84;
        kf16[0] = (((sub >> 1 & 1) == 1) ? tmp[0] - kf16[0] : kf16[0] + tmp[0]);
        kf16[1] = (((sub >> 1 & 1) == 1) ? tmp[1] - kf16[1] : kf16[1] + tmp[1]);
        kf16[2] = (((sub >> 1 & 1) == 1) ? tmp[2] - kf16[2] : kf16[2] + tmp[2]);
        kf16[3] = (((sub >> 1 & 1) == 1) ? tmp[3] - kf16[3] : kf16[3] + tmp[3]);
        kf16[4] = (((sub >> 1 & 1) == 1) ? tmp[4] - kf16[4] : kf16[4] + tmp[4]);
        kf16[5] = (((sub >> 1 & 1) == 1) ? tmp[5] - kf16[5] : kf16[5] + tmp[5]);
        kf16[6] = (((sub >> 1 & 1) == 1) ? tmp[6] - kf16[6] : kf16[6] + tmp[6]);
        kf16[7] = (((sub >> 1 & 1) == 1) ? tmp[7] - kf16[7] : kf16[7] + tmp[7]);
        kf16[8] = (((sub >> 1 & 1) == 1) ? tmp[8] - kf16[8] : kf16[8] + tmp[8]);
        kf16[9] = (((sub >> 1 & 1) == 1) ? tmp[9] - kf16[9] : kf16[9] + tmp[9]);
        kf16[10] = (((sub >> 1 & 1) == 1) ? tmp[10] - kf16[10] : kf16[10] + tmp[10]);
        kf16[11] = (((sub >> 1 & 1) == 1) ? tmp[11] - kf16[11] : kf16[11] + tmp[11]);
        kf16[12] = (((sub >> 1 & 1) == 1) ? tmp[12] - kf16[12] : kf16[12] + tmp[12]);
        kf16[13] = (((sub >> 1 & 1) == 1) ? tmp[13] - kf16[13] : kf16[13] + tmp[13]);
        kf16[14] = (((sub >> 1 & 1) == 1) ? tmp[14] - kf16[14] : kf16[14] + tmp[14]);
        kf16[15] = (((sub >> 1 & 1) == 1) ? tmp[15] - kf16[15] : kf16[15] + tmp[15]);
        float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, kf16[0], 4);
        tmp[0] = _shfl_xor_85;
        float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, kf16[1], 4);
        tmp[1] = _shfl_xor_86;
        float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, kf16[2], 4);
        tmp[2] = _shfl_xor_87;
        float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, kf16[3], 4);
        tmp[3] = _shfl_xor_88;
        float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, kf16[4], 4);
        tmp[4] = _shfl_xor_89;
        float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, kf16[5], 4);
        tmp[5] = _shfl_xor_90;
        float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, kf16[6], 4);
        tmp[6] = _shfl_xor_91;
        float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, kf16[7], 4);
        tmp[7] = _shfl_xor_92;
        float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, kf16[8], 4);
        tmp[8] = _shfl_xor_93;
        float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, kf16[9], 4);
        tmp[9] = _shfl_xor_94;
        float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, kf16[10], 4);
        tmp[10] = _shfl_xor_95;
        float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, kf16[11], 4);
        tmp[11] = _shfl_xor_96;
        float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, kf16[12], 4);
        tmp[12] = _shfl_xor_97;
        float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, kf16[13], 4);
        tmp[13] = _shfl_xor_98;
        float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, kf16[14], 4);
        tmp[14] = _shfl_xor_99;
        float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, kf16[15], 4);
        tmp[15] = _shfl_xor_100;
        kf16[0] = (((sub >> 2 & 1) == 1) ? tmp[0] - kf16[0] : kf16[0] + tmp[0]);
        kf16[1] = (((sub >> 2 & 1) == 1) ? tmp[1] - kf16[1] : kf16[1] + tmp[1]);
        kf16[2] = (((sub >> 2 & 1) == 1) ? tmp[2] - kf16[2] : kf16[2] + tmp[2]);
        kf16[3] = (((sub >> 2 & 1) == 1) ? tmp[3] - kf16[3] : kf16[3] + tmp[3]);
        kf16[4] = (((sub >> 2 & 1) == 1) ? tmp[4] - kf16[4] : kf16[4] + tmp[4]);
        kf16[5] = (((sub >> 2 & 1) == 1) ? tmp[5] - kf16[5] : kf16[5] + tmp[5]);
        kf16[6] = (((sub >> 2 & 1) == 1) ? tmp[6] - kf16[6] : kf16[6] + tmp[6]);
        kf16[7] = (((sub >> 2 & 1) == 1) ? tmp[7] - kf16[7] : kf16[7] + tmp[7]);
        kf16[8] = (((sub >> 2 & 1) == 1) ? tmp[8] - kf16[8] : kf16[8] + tmp[8]);
        kf16[9] = (((sub >> 2 & 1) == 1) ? tmp[9] - kf16[9] : kf16[9] + tmp[9]);
        kf16[10] = (((sub >> 2 & 1) == 1) ? tmp[10] - kf16[10] : kf16[10] + tmp[10]);
        kf16[11] = (((sub >> 2 & 1) == 1) ? tmp[11] - kf16[11] : kf16[11] + tmp[11]);
        kf16[12] = (((sub >> 2 & 1) == 1) ? tmp[12] - kf16[12] : kf16[12] + tmp[12]);
        kf16[13] = (((sub >> 2 & 1) == 1) ? tmp[13] - kf16[13] : kf16[13] + tmp[13]);
        kf16[14] = (((sub >> 2 & 1) == 1) ? tmp[14] - kf16[14] : kf16[14] + tmp[14]);
        kf16[15] = (((sub >> 2 & 1) == 1) ? tmp[15] - kf16[15] : kf16[15] + tmp[15]);
        kf16[0] = kf16[0] * (sgn_hi * 0.08838834764831843f);
        kf16[1] = kf16[1] * (sgn_hi * -0.08838834764831843f);
        kf16[2] = kf16[2] * (sgn_hi * 0.08838834764831843f);
        kf16[3] = kf16[3] * (sgn_hi * 0.08838834764831843f);
        kf16[4] = kf16[4] * (sgn_hi * -0.08838834764831843f);
        kf16[5] = kf16[5] * (sgn_hi * 0.08838834764831843f);
        kf16[6] = kf16[6] * (sgn_hi * -0.08838834764831843f);
        kf16[7] = kf16[7] * (sgn_hi * -0.08838834764831843f);
        kf16[8] = kf16[8] * (sgn_hi * 0.08838834764831843f);
        kf16[9] = kf16[9] * (sgn_hi * 0.08838834764831843f);
        kf16[10] = kf16[10] * (sgn_hi * -0.08838834764831843f);
        kf16[11] = kf16[11] * (sgn_hi * 0.08838834764831843f);
        kf16[12] = kf16[12] * (sgn_hi * -0.08838834764831843f);
        kf16[13] = kf16[13] * (sgn_hi * -0.08838834764831843f);
        kf16[14] = kf16[14] * (sgn_hi * 0.08838834764831843f);
        kf16[15] = kf16[15] * (sgn_hi * -0.08838834764831843f);
        float kbmax = 0.0f;
        for (int e_4 = 0; e_4 < 16; e_4++) {
            float _fabs_1 = fabsf(kf16[e_4]);
            float _fmax_5 = fmaxf(kbmax, _fabs_1);
            kbmax = _fmax_5;
        }
        float ks = k_scale[seg * num_heads + head];
        float _fdiv_rn_3 = __fdiv_rn(1.0f, ks);
        float kg = _fdiv_rn_3;
        float ksf_val = kbmax * kg * 0.16666666666666666f;
        uint16_t _e4m3x2_f32_1;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(0.0f), "f"(ksf_val));
        uint16_t _e4m3x2_decode_8 = (uint16_t)((unsigned int)_e4m3x2_f32_1 & 0xFFu);
        uint32_t _f16x2_decode_8;
        float _fp8_decode_1;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_8) : "h"(_e4m3x2_decode_8));
        uint16_t _f16_decode_8 = (uint16_t)_f16x2_decode_8;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_1) : "h"(_f16_decode_8));
        float ksf_dec = _fp8_decode_1;
        float _fdiv_rn_4 = __fdiv_rn(kg, ksf_dec);
        float kout = ((ksf_dec != 0.0f) ? _fdiv_rn_4 : 0.0f);
        float kn[16];
        for (int e_5 = 0; e_5 < 16; e_5++) {
            kn[e_5] = kf16[e_5] * kout;
        }
        unsigned int kw[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(kw[0]) : "f"(kn[0]), "f"(kn[1]), "f"(kn[2]), "f"(kn[3]), "f"(kn[4]), "f"(kn[5]), "f"(kn[6]), "f"(kn[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(kw[1]) : "f"(kn[8]), "f"(kn[9]), "f"(kn[10]), "f"(kn[11]), "f"(kn[12]), "f"(kn[13]), "f"(kn[14]), "f"(kn[15]));
        unsigned int kstore[4];
        unsigned int _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, kw[0], 1);
        kstore[0] = _shfl_xor_101;
        unsigned int _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, kw[1], 1);
        kstore[1] = _shfl_xor_102;
        kstore[2] = kw[0];
        kstore[3] = kw[1];
        {
            unsigned short _sf_pair;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(ksf_val));
            *(reinterpret_cast<unsigned char*>(k_sf + (((head * num_kblocks + (tok >> 7)) * 128 + (tok & 127)) * 8 + sub)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
        }
        if (valid == 1) {
            if ((sub & 1) == 1) {
                reinterpret_cast<int4*>(k4 + (((tok * num_heads + head) * 64 + (sub - 1) * 8) / 4))[0] = reinterpret_cast<int4*>(kstore)[0];
            }
        }
        float vq[16];
        float vf4[4];
        float _vec_load_7[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(v_scale + ((vseg[0] * num_heads + head) * 128 + ch) + 0);
            _vec_load_7[0 + 0] = _v4.x;
            _vec_load_7[0 + 1] = _v4.y;
            _vec_load_7[0 + 2] = _v4.z;
            _vec_load_7[0 + 3] = _v4.w;
        }
        vf4[0] = _vec_load_2[0];
        vf4[1] = _vec_load_2[1];
        vf4[2] = _vec_load_2[2];
        vf4[3] = _vec_load_2[3];
        float _fdiv_rn_5 = __fdiv_rn(vf4[0], _vec_load_7[0]);
        vq[0] = ((vlive[0] == 1) ? _fdiv_rn_5 : 0.0f);
        float _fdiv_rn_6 = __fdiv_rn(vf4[1], _vec_load_7[1]);
        vq[1] = ((vlive[0] == 1) ? _fdiv_rn_6 : 0.0f);
        float _fdiv_rn_7 = __fdiv_rn(vf4[2], _vec_load_7[2]);
        vq[2] = ((vlive[0] == 1) ? _fdiv_rn_7 : 0.0f);
        float _fdiv_rn_8 = __fdiv_rn(vf4[3], _vec_load_7[3]);
        vq[3] = ((vlive[0] == 1) ? _fdiv_rn_8 : 0.0f);
        float _vec_load_8[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(v_scale + ((vseg[1] * num_heads + head) * 128 + ch) + 0);
            _vec_load_8[0 + 0] = _v4.x;
            _vec_load_8[0 + 1] = _v4.y;
            _vec_load_8[0 + 2] = _v4.z;
            _vec_load_8[0 + 3] = _v4.w;
        }
        vf4[0] = _vec_load_3[0];
        vf4[1] = _vec_load_3[1];
        vf4[2] = _vec_load_3[2];
        vf4[3] = _vec_load_3[3];
        float _fdiv_rn_9 = __fdiv_rn(vf4[0], _vec_load_8[0]);
        vq[4] = ((vlive[1] == 1) ? _fdiv_rn_9 : 0.0f);
        float _fdiv_rn_10 = __fdiv_rn(vf4[1], _vec_load_8[1]);
        vq[5] = ((vlive[1] == 1) ? _fdiv_rn_10 : 0.0f);
        float _fdiv_rn_11 = __fdiv_rn(vf4[2], _vec_load_8[2]);
        vq[6] = ((vlive[1] == 1) ? _fdiv_rn_11 : 0.0f);
        float _fdiv_rn_12 = __fdiv_rn(vf4[3], _vec_load_8[3]);
        vq[7] = ((vlive[1] == 1) ? _fdiv_rn_12 : 0.0f);
        float _vec_load_9[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(v_scale + ((vseg[2] * num_heads + head) * 128 + ch) + 0);
            _vec_load_9[0 + 0] = _v4.x;
            _vec_load_9[0 + 1] = _v4.y;
            _vec_load_9[0 + 2] = _v4.z;
            _vec_load_9[0 + 3] = _v4.w;
        }
        vf4[0] = _vec_load_4[0];
        vf4[1] = _vec_load_4[1];
        vf4[2] = _vec_load_4[2];
        vf4[3] = _vec_load_4[3];
        float _fdiv_rn_13 = __fdiv_rn(vf4[0], _vec_load_9[0]);
        vq[8] = ((vlive[2] == 1) ? _fdiv_rn_13 : 0.0f);
        float _fdiv_rn_14 = __fdiv_rn(vf4[1], _vec_load_9[1]);
        vq[9] = ((vlive[2] == 1) ? _fdiv_rn_14 : 0.0f);
        float _fdiv_rn_15 = __fdiv_rn(vf4[2], _vec_load_9[2]);
        vq[10] = ((vlive[2] == 1) ? _fdiv_rn_15 : 0.0f);
        float _fdiv_rn_16 = __fdiv_rn(vf4[3], _vec_load_9[3]);
        vq[11] = ((vlive[2] == 1) ? _fdiv_rn_16 : 0.0f);
        float _vec_load_10[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(v_scale + ((vseg[3] * num_heads + head) * 128 + ch) + 0);
            _vec_load_10[0 + 0] = _v4.x;
            _vec_load_10[0 + 1] = _v4.y;
            _vec_load_10[0 + 2] = _v4.z;
            _vec_load_10[0 + 3] = _v4.w;
        }
        vf4[0] = _vec_load_5[0];
        vf4[1] = _vec_load_5[1];
        vf4[2] = _vec_load_5[2];
        vf4[3] = _vec_load_5[3];
        float _fdiv_rn_17 = __fdiv_rn(vf4[0], _vec_load_10[0]);
        vq[12] = ((vlive[3] == 1) ? _fdiv_rn_17 : 0.0f);
        float _fdiv_rn_18 = __fdiv_rn(vf4[1], _vec_load_10[1]);
        vq[13] = ((vlive[3] == 1) ? _fdiv_rn_18 : 0.0f);
        float _fdiv_rn_19 = __fdiv_rn(vf4[2], _vec_load_10[2]);
        vq[14] = ((vlive[3] == 1) ? _fdiv_rn_19 : 0.0f);
        float _fdiv_rn_20 = __fdiv_rn(vf4[3], _vec_load_10[3]);
        vq[15] = ((vlive[3] == 1) ? _fdiv_rn_20 : 0.0f);
        int vt_word = (head * 128 + ch) * total_padded + tok_base + chunk * 16 + 4 * t8 >> 2;
        for (int e_6 = 0; e_6 < 4; e_6++) {
            float four[4];
            for (int i = 0; i < 4; i++) {
                four[i] = vq[i * 4 + e_6];
            }
            unsigned int word[1];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(four[0]), "f"(four[1]),
                                       "f"(four[2]), "f"(four[3]));
                word[0] = _packed;
            }
            *(reinterpret_cast<unsigned int*>(vt8 + (vt_word + e_6 * (total_padded >> 2))) + (0)) = word[0];
        }
    }
}

}  // namespace h3_varlen_quantize_nvfp4_sm120a
#undef H3_VARLEN_INF
#undef NUM_MAIN_STAGES
#undef THREADS

namespace h3_varlen_attention_nvfp4_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_VARLEN_INF CUDART_INF_F
#define NUM_KV_STAGES 2
#define SMEM_Q_OFF 1024
#define SMEM_Q_STAGE_BYTES 8192
#define SMEM_Q_STRIDE 8192
#define SMEM_K_STAGE_OFF 9216
#define SMEM_K_STAGE_STAGE_BYTES 8192
#define SMEM_K_STAGE_STRIDE 8192
#define SMEM_KSF_STAGE_OFF 25600
#define SMEM_KSF_STAGE_STAGE_BYTES 1024
#define SMEM_KSF_STAGE_STRIDE 1024
#define SMEM_V_STAGE_OFF 27648
#define SMEM_V_STAGE_STAGE_BYTES 16384
#define SMEM_V_STAGE_STRIDE 16384
#define SMEM_TOTAL 60416
#define THREADS 256

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


// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
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


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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
kernel_minimax_h3_sm120_varlen_attention_nvfp4(const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map, const __grid_constant__ CUtensorMap KSF_map, const __grid_constant__ CUtensorMap V_map, __nv_bfloat16* __restrict__ O, float* __restrict__ q_scale, float* __restrict__ k_scale, float* __restrict__ v_scale, unsigned int* __restrict__ q_sf, int* __restrict__ unit_table, int total_units, int num_heads, int num_kblocks, float softmax_scale_log2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define k_full_addr (mbar_base + 8)
    #define v_full_addr (mbar_base + 24)
    #define k_empty_addr (mbar_base + 40)
    #define v_empty_addr (mbar_base + 56)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* Q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int Q_addr = smem + 1024;
    uint8_t* K_stage = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int K_stage_addr = smem + 9216;
    unsigned int* KSF_stage = reinterpret_cast<unsigned int*>(smem_raw + 25600);
    const int KSF_stage_addr = smem + 25600;
    uint8_t* V_stage = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int V_stage_addr = smem + 27648;

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 9 barriers)
    // Mbarriers at smem_raw[0..72)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'kv' ---
            // k_full: 2 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // k_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 40, 8);
            mbarrier_init(smem + 48, 8);
            // v_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 56, 8);
            mbarrier_init(smem + 64, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    unsigned int kv_stage = 0;
    unsigned int kv_phase = 0;
    if (warp >= 4) {
        asm volatile("barrier.arrive 1, 256;" ::: "memory");
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int _vec_load_0[4];
    {
        const int4* _ivptr_0 = reinterpret_cast<const int4*>(unit_table + (4 * bid) + 0);
        int4 _ivld_0;
        _ivld_0 = *_ivptr_0;
        _vec_load_0[0 + 0] = _ivld_0.x;
        _vec_load_0[0 + 1] = _ivld_0.y;
        _vec_load_0[0 + 2] = _ivld_0.z;
        _vec_load_0[0 + 3] = _ivld_0.w;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(q_full_addr, 8192);
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(Q_addr), "l"((&Q_map)), "r"(0), "r"(_vec_load_0[2] + (_vec_load_0[1] & 65535) * 128), "r"(_vec_load_0[1] >> 16),
                   "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    unsigned int _phase_q_full_0 = 0;
    #pragma unroll 1
    for (int slot = bid; slot < total_units; slot += num_bids) {
        int _vec_load_1[4];
        {
            const int4* _ivptr_1 = reinterpret_cast<const int4*>(unit_table + (4 * slot) + 0);
            int4 _ivld_1;
            _ivld_1 = *_ivptr_1;
            _vec_load_1[0 + 0] = _ivld_1.x;
            _vec_load_1[0 + 1] = _ivld_1.y;
            _vec_load_1[0 + 2] = _ivld_1.z;
            _vec_load_1[0 + 3] = _ivld_1.w;
        }
        int seg = _vec_load_1[0];
        int packed = _vec_load_1[1];
        int head = packed >> 16;
        int q_tile = packed & 65535;
        int seg_begin_v = _vec_load_1[2];
        int seg_end_v = _vec_load_1[3];
        int q_row0 = seg_begin_v + q_tile * 128;
        int kb_first = seg_begin_v / 128;
        int kb_last = (seg_end_v - 1) / 128;
        int next_slot = ((slot + num_bids < total_units) ? slot + num_bids : slot);
        int _vec_load_2[4];
        {
            const int4* _ivptr_2 = reinterpret_cast<const int4*>(unit_table + (4 * next_slot) + 0);
            int4 _ivld_2;
            _ivld_2 = *_ivptr_2;
            _vec_load_2[0 + 0] = _ivld_2.x;
            _vec_load_2[0 + 1] = _ivld_2.y;
            _vec_load_2[0 + 2] = _ivld_2.z;
            _vec_load_2[0 + 3] = _ivld_2.w;
        }
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 9216);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                    :: "r"(K_stage_addr + kv_stage * 8192), "l"((&K_map)), "r"(0), "r"(kb_first * 128), "r"(head),
                       "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3}], [%4], %5;"
                    :: "r"(KSF_stage_addr + kv_stage * 1024), "l"((&KSF_map)), "r"(0), "r"((head * num_kblocks + kb_first) * 64),
                       "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 16384);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                    :: "r"(V_stage_addr + kv_stage * 16384), "l"((&V_map)), "r"(kb_first * 128), "r"(0), "r"(head),
                       "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
            }
        }
        if (kb_last >= kb_first + 1) {
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(k_full_addr + ((kv_stage + 1) % 2) * 8, 9216);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(K_stage_addr + (kv_stage + 1) % 2 * 8192), "l"((&K_map)), "r"(0), "r"((kb_first + 1) * 128), "r"(head),
                           "r"(k_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(KSF_stage_addr + (kv_stage + 1) % 2 * 1024), "l"((&KSF_map)), "r"(0), "r"((head * num_kblocks + (kb_first + 1)) * 64),
                           "r"(k_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    mbarrier_arrive_expect_tx(v_full_addr + ((kv_stage + 1) % 2) * 8, 16384);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(V_stage_addr + (kv_stage + 1) % 2 * 16384), "l"((&V_map)), "r"((kb_first + 1) * 128), "r"(0), "r"(head),
                           "r"(v_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        int row0 = q_row0 + warp * 16 + (lane >> 2);
        int row1 = row0 + 8;
        int row0_c = ((row0 < seg_end_v) ? row0 : seg_end_v - 1);
        int row1_c = ((row1 < seg_end_v) ? row1 : seg_end_v - 1);
        float qs0 = q_scale[row0_c * num_heads + head];
        float qs1 = q_scale[row1_c * num_heads + head];
        float ks_seg = k_scale[seg * num_heads + head];
        float c_row0 = qs0 * softmax_scale_log2 * ks_seg;
        float c_row1 = qs1 * softmax_scale_log2 * ks_seg;
        int sf_row = (((lane & 1) == 1) ? row1_c : row0_c);
        unsigned int qsf[2];
        for (int u = 0; u < 2; u++) {
            qsf[u] = q_sf[(sf_row * num_heads + head) * 2 + u];
        }
        unsigned int q_frags[8];
        unsigned int k_frag[4];
        unsigned int v_frag[4];
        float s_acc[64];
        unsigned int p_frag[16];
        float p_tmp[8];
        unsigned int x16[1];
        float o_acc[64];
        float o_tmp[4];
        float m_state[2];
        float l_state[2];
        o_acc[0] = 0.0f;
        o_acc[1] = 0.0f;
        o_acc[2] = 0.0f;
        o_acc[3] = 0.0f;
        o_acc[4] = 0.0f;
        o_acc[5] = 0.0f;
        o_acc[6] = 0.0f;
        o_acc[7] = 0.0f;
        o_acc[8] = 0.0f;
        o_acc[9] = 0.0f;
        o_acc[10] = 0.0f;
        o_acc[11] = 0.0f;
        o_acc[12] = 0.0f;
        o_acc[13] = 0.0f;
        o_acc[14] = 0.0f;
        o_acc[15] = 0.0f;
        o_acc[16] = 0.0f;
        o_acc[17] = 0.0f;
        o_acc[18] = 0.0f;
        o_acc[19] = 0.0f;
        o_acc[20] = 0.0f;
        o_acc[21] = 0.0f;
        o_acc[22] = 0.0f;
        o_acc[23] = 0.0f;
        o_acc[24] = 0.0f;
        o_acc[25] = 0.0f;
        o_acc[26] = 0.0f;
        o_acc[27] = 0.0f;
        o_acc[28] = 0.0f;
        o_acc[29] = 0.0f;
        o_acc[30] = 0.0f;
        o_acc[31] = 0.0f;
        o_acc[32] = 0.0f;
        o_acc[33] = 0.0f;
        o_acc[34] = 0.0f;
        o_acc[35] = 0.0f;
        o_acc[36] = 0.0f;
        o_acc[37] = 0.0f;
        o_acc[38] = 0.0f;
        o_acc[39] = 0.0f;
        o_acc[40] = 0.0f;
        o_acc[41] = 0.0f;
        o_acc[42] = 0.0f;
        o_acc[43] = 0.0f;
        o_acc[44] = 0.0f;
        o_acc[45] = 0.0f;
        o_acc[46] = 0.0f;
        o_acc[47] = 0.0f;
        o_acc[48] = 0.0f;
        o_acc[49] = 0.0f;
        o_acc[50] = 0.0f;
        o_acc[51] = 0.0f;
        o_acc[52] = 0.0f;
        o_acc[53] = 0.0f;
        o_acc[54] = 0.0f;
        o_acc[55] = 0.0f;
        o_acc[56] = 0.0f;
        o_acc[57] = 0.0f;
        o_acc[58] = 0.0f;
        o_acc[59] = 0.0f;
        o_acc[60] = 0.0f;
        o_acc[61] = 0.0f;
        o_acc[62] = 0.0f;
        o_acc[63] = 0.0f;
        m_state[0] = -H3_VARLEN_INF;
        m_state[1] = -H3_VARLEN_INF;
        l_state[0] = 0.0f;
        l_state[1] = 0.0f;
        mbarrier_wait(q_full_addr, _phase_q_full_0);
        _phase_q_full_0 ^= 1;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[0]), "=r"(q_frags[1]), "=r"(q_frags[2]), "=r"(q_frags[3])
            : "r"(Q_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[4]), "=r"(q_frags[5]), "=r"(q_frags[6]), "=r"(q_frags[7])
            : "r"(Q_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
            : "memory");
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 8192);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                    :: "r"(Q_addr), "l"((&Q_map)), "r"(0), "r"(_vec_load_2[2] + (_vec_load_2[1] & 65535) * 128), "r"(_vec_load_2[1] >> 16),
                       "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            }
        }
        #pragma unroll 1
        for (int kb = kb_first; kb < kb_last + 1; kb++) {
            mbarrier_wait(k_full_addr + (kv_stage) * 8, kv_phase);
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_0[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_0[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((lane >> 2) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[0]), "=f"(s_acc[1]), "=f"(s_acc[2]), "=f"(s_acc[3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_1[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_1[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[4]), "=f"(s_acc[(4) + 1]), "=f"(s_acc[(4) + 2]), "=f"(s_acc[(4) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_2[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_2[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[8]), "=f"(s_acc[(8) + 1]), "=f"(s_acc[(8) + 2]), "=f"(s_acc[(8) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_3[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_3[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[12]), "=f"(s_acc[(12) + 1]), "=f"(s_acc[(12) + 2]), "=f"(s_acc[(12) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_4[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_4[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((32 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[16]), "=f"(s_acc[(16) + 1]), "=f"(s_acc[(16) + 2]), "=f"(s_acc[(16) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_5[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_5[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((40 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[20]), "=f"(s_acc[(20) + 1]), "=f"(s_acc[(20) + 2]), "=f"(s_acc[(20) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_6[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_6[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((48 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[24]), "=f"(s_acc[(24) + 1]), "=f"(s_acc[(24) + 2]), "=f"(s_acc[(24) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_7[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_7[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((56 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[28]), "=f"(s_acc[(28) + 1]), "=f"(s_acc[(28) + 2]), "=f"(s_acc[(28) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_8[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_8[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((64 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[32]), "=f"(s_acc[(32) + 1]), "=f"(s_acc[(32) + 2]), "=f"(s_acc[(32) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_9[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_9[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((72 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[36]), "=f"(s_acc[(36) + 1]), "=f"(s_acc[(36) + 2]), "=f"(s_acc[(36) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_10[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_10[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((80 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[40]), "=f"(s_acc[(40) + 1]), "=f"(s_acc[(40) + 2]), "=f"(s_acc[(40) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_11[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_11[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((88 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[44]), "=f"(s_acc[(44) + 1]), "=f"(s_acc[(44) + 2]), "=f"(s_acc[(44) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_12[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_12[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((96 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[48]), "=f"(s_acc[(48) + 1]), "=f"(s_acc[(48) + 2]), "=f"(s_acc[(48) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_13[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_13[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((104 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[52]), "=f"(s_acc[(52) + 1]), "=f"(s_acc[(52) + 2]), "=f"(s_acc[(52) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_14[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_14[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((112 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[56]), "=f"(s_acc[(56) + 1]), "=f"(s_acc[(56) + 2]), "=f"(s_acc[(56) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_15[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_15[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((120 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[60]), "=f"(s_acc[(60) + 1]), "=f"(s_acc[(60) + 2]), "=f"(s_acc[(60) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_16[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_16[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((lane >> 2) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_17[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_17[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_18[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_18[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[8]), "+f"(s_acc[(8) + 1]), "+f"(s_acc[(8) + 2]), "+f"(s_acc[(8) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_19[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_19[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[12]), "+f"(s_acc[(12) + 1]), "+f"(s_acc[(12) + 2]), "+f"(s_acc[(12) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_20[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_20[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((32 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[16]), "+f"(s_acc[(16) + 1]), "+f"(s_acc[(16) + 2]), "+f"(s_acc[(16) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_21[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_21[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((40 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[20]), "+f"(s_acc[(20) + 1]), "+f"(s_acc[(20) + 2]), "+f"(s_acc[(20) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_22[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_22[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((48 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[24]), "+f"(s_acc[(24) + 1]), "+f"(s_acc[(24) + 2]), "+f"(s_acc[(24) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_23[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_23[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((56 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[28]), "+f"(s_acc[(28) + 1]), "+f"(s_acc[(28) + 2]), "+f"(s_acc[(28) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_23[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_24[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_24[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((64 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[32]), "+f"(s_acc[(32) + 1]), "+f"(s_acc[(32) + 2]), "+f"(s_acc[(32) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_24[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_25[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_25[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((72 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[36]), "+f"(s_acc[(36) + 1]), "+f"(s_acc[(36) + 2]), "+f"(s_acc[(36) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_25[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_26[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_26[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((80 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[40]), "+f"(s_acc[(40) + 1]), "+f"(s_acc[(40) + 2]), "+f"(s_acc[(40) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_26[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_27[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_27[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((88 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[44]), "+f"(s_acc[(44) + 1]), "+f"(s_acc[(44) + 2]), "+f"(s_acc[(44) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_28[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_28[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((96 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[48]), "+f"(s_acc[(48) + 1]), "+f"(s_acc[(48) + 2]), "+f"(s_acc[(48) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_29[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_29[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((104 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[52]), "+f"(s_acc[(52) + 1]), "+f"(s_acc[(52) + 2]), "+f"(s_acc[(52) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_30[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_30[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((112 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[56]), "+f"(s_acc[(56) + 1]), "+f"(s_acc[(56) + 2]), "+f"(s_acc[(56) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_31[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_31[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((120 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[60]), "+f"(s_acc[(60) + 1]), "+f"(s_acc[(60) + 2]), "+f"(s_acc[(60) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_31[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            if (elect_sync()) {
                mbarrier_arrive(k_empty_addr + (kv_stage) * 8);
            }
            if (kb_first == kb || kb_last == kb) {
                s_acc[0] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[0]);
                s_acc[1] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[1]);
                s_acc[2] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[2]);
                s_acc[3] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[3]);
                s_acc[4] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[4]);
                s_acc[5] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[5]);
                s_acc[6] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[6]);
                s_acc[7] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[7]);
                s_acc[8] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[8]);
                s_acc[9] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[9]);
                s_acc[10] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[10]);
                s_acc[11] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[11]);
                s_acc[12] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[12]);
                s_acc[13] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[13]);
                s_acc[14] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[14]);
                s_acc[15] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[15]);
                s_acc[16] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[16]);
                s_acc[17] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[17]);
                s_acc[18] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[18]);
                s_acc[19] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[19]);
                s_acc[20] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[20]);
                s_acc[21] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[21]);
                s_acc[22] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[22]);
                s_acc[23] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[23]);
                s_acc[24] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[24]);
                s_acc[25] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[25]);
                s_acc[26] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[26]);
                s_acc[27] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[27]);
                s_acc[28] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[28]);
                s_acc[29] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[29]);
                s_acc[30] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[30]);
                s_acc[31] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[31]);
                s_acc[32] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[32]);
                s_acc[33] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[33]);
                s_acc[34] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[34]);
                s_acc[35] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[35]);
                s_acc[36] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[36]);
                s_acc[37] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[37]);
                s_acc[38] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[38]);
                s_acc[39] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[39]);
                s_acc[40] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[40]);
                s_acc[41] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[41]);
                s_acc[42] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[42]);
                s_acc[43] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[43]);
                s_acc[44] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[44]);
                s_acc[45] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[45]);
                s_acc[46] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[46]);
                s_acc[47] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[47]);
                s_acc[48] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[48]);
                s_acc[49] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[49]);
                s_acc[50] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[50]);
                s_acc[51] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[51]);
                s_acc[52] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[52]);
                s_acc[53] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[53]);
                s_acc[54] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[54]);
                s_acc[55] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[55]);
                s_acc[56] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[56]);
                s_acc[57] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[57]);
                s_acc[58] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[58]);
                s_acc[59] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[59]);
                s_acc[60] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[60]);
                s_acc[61] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[61]);
                s_acc[62] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[62]);
                s_acc[63] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[63]);
            }
            float _fmax_0 = fmaxf(s_acc[0], s_acc[1]);
            float _fmax_1 = fmaxf(s_acc[2], s_acc[3]);
            float _fmax_2 = fmaxf(s_acc[4], s_acc[5]);
            float _fmax_3 = fmaxf(_fmax_0, _fmax_2);
            float _fmax_4 = fmaxf(s_acc[6], s_acc[7]);
            float _fmax_5 = fmaxf(_fmax_1, _fmax_4);
            float _fmax_6 = fmaxf(s_acc[8], s_acc[9]);
            float _fmax_7 = fmaxf(_fmax_3, _fmax_6);
            float _fmax_8 = fmaxf(s_acc[10], s_acc[11]);
            float _fmax_9 = fmaxf(_fmax_5, _fmax_8);
            float _fmax_10 = fmaxf(s_acc[12], s_acc[13]);
            float _fmax_11 = fmaxf(_fmax_7, _fmax_10);
            float _fmax_12 = fmaxf(s_acc[14], s_acc[15]);
            float _fmax_13 = fmaxf(_fmax_9, _fmax_12);
            float _fmax_14 = fmaxf(s_acc[16], s_acc[17]);
            float _fmax_15 = fmaxf(_fmax_11, _fmax_14);
            float _fmax_16 = fmaxf(s_acc[18], s_acc[19]);
            float _fmax_17 = fmaxf(_fmax_13, _fmax_16);
            float _fmax_18 = fmaxf(s_acc[20], s_acc[21]);
            float _fmax_19 = fmaxf(_fmax_15, _fmax_18);
            float _fmax_20 = fmaxf(s_acc[22], s_acc[23]);
            float _fmax_21 = fmaxf(_fmax_17, _fmax_20);
            float _fmax_22 = fmaxf(s_acc[24], s_acc[25]);
            float _fmax_23 = fmaxf(_fmax_19, _fmax_22);
            float _fmax_24 = fmaxf(s_acc[26], s_acc[27]);
            float _fmax_25 = fmaxf(_fmax_21, _fmax_24);
            float _fmax_26 = fmaxf(s_acc[28], s_acc[29]);
            float _fmax_27 = fmaxf(_fmax_23, _fmax_26);
            float _fmax_28 = fmaxf(s_acc[30], s_acc[31]);
            float _fmax_29 = fmaxf(_fmax_25, _fmax_28);
            float _fmax_30 = fmaxf(s_acc[32], s_acc[33]);
            float _fmax_31 = fmaxf(_fmax_27, _fmax_30);
            float _fmax_32 = fmaxf(s_acc[34], s_acc[35]);
            float _fmax_33 = fmaxf(_fmax_29, _fmax_32);
            float _fmax_34 = fmaxf(s_acc[36], s_acc[37]);
            float _fmax_35 = fmaxf(_fmax_31, _fmax_34);
            float _fmax_36 = fmaxf(s_acc[38], s_acc[39]);
            float _fmax_37 = fmaxf(_fmax_33, _fmax_36);
            float _fmax_38 = fmaxf(s_acc[40], s_acc[41]);
            float _fmax_39 = fmaxf(_fmax_35, _fmax_38);
            float _fmax_40 = fmaxf(s_acc[42], s_acc[43]);
            float _fmax_41 = fmaxf(_fmax_37, _fmax_40);
            float _fmax_42 = fmaxf(s_acc[44], s_acc[45]);
            float _fmax_43 = fmaxf(_fmax_39, _fmax_42);
            float _fmax_44 = fmaxf(s_acc[46], s_acc[47]);
            float _fmax_45 = fmaxf(_fmax_41, _fmax_44);
            float _fmax_46 = fmaxf(s_acc[48], s_acc[49]);
            float _fmax_47 = fmaxf(_fmax_43, _fmax_46);
            float _fmax_48 = fmaxf(s_acc[50], s_acc[51]);
            float _fmax_49 = fmaxf(_fmax_45, _fmax_48);
            float _fmax_50 = fmaxf(s_acc[52], s_acc[53]);
            float _fmax_51 = fmaxf(_fmax_47, _fmax_50);
            float _fmax_52 = fmaxf(s_acc[54], s_acc[55]);
            float _fmax_53 = fmaxf(_fmax_49, _fmax_52);
            float _fmax_54 = fmaxf(s_acc[56], s_acc[57]);
            float _fmax_55 = fmaxf(_fmax_51, _fmax_54);
            float _fmax_56 = fmaxf(s_acc[58], s_acc[59]);
            float _fmax_57 = fmaxf(_fmax_53, _fmax_56);
            float _fmax_58 = fmaxf(s_acc[60], s_acc[61]);
            float _fmax_59 = fmaxf(_fmax_55, _fmax_58);
            float _fmax_60 = fmaxf(s_acc[62], s_acc[63]);
            float _fmax_61 = fmaxf(_fmax_57, _fmax_60);
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, _fmax_59, 1);
            float _fmax_62 = fmaxf(_fmax_59, _shfl_xor_0);
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, _fmax_62, 2);
            float _fmax_63 = fmaxf(_fmax_62, _shfl_xor_1);
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, _fmax_61, 1);
            float _fmax_64 = fmaxf(_fmax_61, _shfl_xor_2);
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, _fmax_64, 2);
            float _fmax_65 = fmaxf(_fmax_64, _shfl_xor_3);
            float _fmax_66 = fmaxf(m_state[0], _fmax_63 * c_row0);
            float _fmax_67 = fmaxf(m_state[1], _fmax_65 * c_row1);
            float _exp2_0 = approx_exp2(m_state[0] - _fmax_66);
            float _exp2_1 = approx_exp2(m_state[1] - _fmax_67);
            int _vote_0 = __any_sync(0xFFFFFFFF, _fmax_66 > m_state[0] || _fmax_67 > m_state[1]);
            m_state[0] = _fmax_66;
            m_state[1] = _fmax_67;
            if (_vote_0 != 0) {
                {
                    float2 _pair_scale_even2_3 = make_float2(_exp2_0, _exp2_0);
                    float2 _pair_scale_odd2_3 = make_float2(_exp2_1, _exp2_1);
                    float2* _pair_scale_src2_3 = reinterpret_cast<float2*>(&o_acc[0]);
                    #if __CUDA_ARCH__ >= 1000
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                    asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                    #else
                    o_acc[0] *= _exp2_0;
                    o_acc[1] *= _exp2_0;
                    o_acc[2] *= _exp2_1;
                    o_acc[3] *= _exp2_1;
                    o_acc[4] *= _exp2_0;
                    o_acc[5] *= _exp2_0;
                    o_acc[6] *= _exp2_1;
                    o_acc[7] *= _exp2_1;
                    o_acc[8] *= _exp2_0;
                    o_acc[9] *= _exp2_0;
                    o_acc[10] *= _exp2_1;
                    o_acc[11] *= _exp2_1;
                    o_acc[12] *= _exp2_0;
                    o_acc[13] *= _exp2_0;
                    o_acc[14] *= _exp2_1;
                    o_acc[15] *= _exp2_1;
                    o_acc[16] *= _exp2_0;
                    o_acc[17] *= _exp2_0;
                    o_acc[18] *= _exp2_1;
                    o_acc[19] *= _exp2_1;
                    o_acc[20] *= _exp2_0;
                    o_acc[21] *= _exp2_0;
                    o_acc[22] *= _exp2_1;
                    o_acc[23] *= _exp2_1;
                    o_acc[24] *= _exp2_0;
                    o_acc[25] *= _exp2_0;
                    o_acc[26] *= _exp2_1;
                    o_acc[27] *= _exp2_1;
                    o_acc[28] *= _exp2_0;
                    o_acc[29] *= _exp2_0;
                    o_acc[30] *= _exp2_1;
                    o_acc[31] *= _exp2_1;
                    o_acc[32] *= _exp2_0;
                    o_acc[33] *= _exp2_0;
                    o_acc[34] *= _exp2_1;
                    o_acc[35] *= _exp2_1;
                    o_acc[36] *= _exp2_0;
                    o_acc[37] *= _exp2_0;
                    o_acc[38] *= _exp2_1;
                    o_acc[39] *= _exp2_1;
                    o_acc[40] *= _exp2_0;
                    o_acc[41] *= _exp2_0;
                    o_acc[42] *= _exp2_1;
                    o_acc[43] *= _exp2_1;
                    o_acc[44] *= _exp2_0;
                    o_acc[45] *= _exp2_0;
                    o_acc[46] *= _exp2_1;
                    o_acc[47] *= _exp2_1;
                    o_acc[48] *= _exp2_0;
                    o_acc[49] *= _exp2_0;
                    o_acc[50] *= _exp2_1;
                    o_acc[51] *= _exp2_1;
                    o_acc[52] *= _exp2_0;
                    o_acc[53] *= _exp2_0;
                    o_acc[54] *= _exp2_1;
                    o_acc[55] *= _exp2_1;
                    o_acc[56] *= _exp2_0;
                    o_acc[57] *= _exp2_0;
                    o_acc[58] *= _exp2_1;
                    o_acc[59] *= _exp2_1;
                    o_acc[60] *= _exp2_0;
                    o_acc[61] *= _exp2_0;
                    o_acc[62] *= _exp2_1;
                    o_acc[63] *= _exp2_1;
                    #endif
                }
            }
            float _fma_0 = __fmaf_rn(s_acc[0], c_row0, 8.0f - _fmax_66);
            float _exp2_2 = approx_exp2(_fma_0);
            s_acc[0] = _exp2_2;
            float _fma_1 = __fmaf_rn(s_acc[1], c_row0, 8.0f - _fmax_66);
            float _exp2_3 = approx_exp2(_fma_1);
            s_acc[1] = _exp2_3;
            float _fma_2 = __fmaf_rn(s_acc[2], c_row1, 8.0f - _fmax_67);
            float _exp2_4 = approx_exp2(_fma_2);
            s_acc[2] = _exp2_4;
            float _fma_3 = __fmaf_rn(s_acc[3], c_row1, 8.0f - _fmax_67);
            float _exp2_5 = approx_exp2(_fma_3);
            s_acc[3] = _exp2_5;
            float _fma_4 = __fmaf_rn(s_acc[4], c_row0, 8.0f - _fmax_66);
            float _exp2_6 = approx_exp2(_fma_4);
            s_acc[4] = _exp2_6;
            float _fma_5 = __fmaf_rn(s_acc[5], c_row0, 8.0f - _fmax_66);
            float _exp2_7 = approx_exp2(_fma_5);
            s_acc[5] = _exp2_7;
            float _fma_6 = __fmaf_rn(s_acc[6], c_row1, 8.0f - _fmax_67);
            float _exp2_8 = approx_exp2(_fma_6);
            s_acc[6] = _exp2_8;
            float _fma_7 = __fmaf_rn(s_acc[7], c_row1, 8.0f - _fmax_67);
            float _exp2_9 = approx_exp2(_fma_7);
            s_acc[7] = _exp2_9;
            float _fma_8 = __fmaf_rn(s_acc[8], c_row0, 8.0f - _fmax_66);
            float _exp2_10 = approx_exp2(_fma_8);
            s_acc[8] = _exp2_10;
            float _fma_9 = __fmaf_rn(s_acc[9], c_row0, 8.0f - _fmax_66);
            float _exp2_11 = approx_exp2(_fma_9);
            s_acc[9] = _exp2_11;
            float _fma_10 = __fmaf_rn(s_acc[10], c_row1, 8.0f - _fmax_67);
            float _exp2_12 = approx_exp2(_fma_10);
            s_acc[10] = _exp2_12;
            float _fma_11 = __fmaf_rn(s_acc[11], c_row1, 8.0f - _fmax_67);
            float _exp2_13 = approx_exp2(_fma_11);
            s_acc[11] = _exp2_13;
            float _fma_12 = __fmaf_rn(s_acc[12], c_row0, 8.0f - _fmax_66);
            float _exp2_14 = approx_exp2(_fma_12);
            s_acc[12] = _exp2_14;
            float _fma_13 = __fmaf_rn(s_acc[13], c_row0, 8.0f - _fmax_66);
            float _exp2_15 = approx_exp2(_fma_13);
            s_acc[13] = _exp2_15;
            float _fma_14 = __fmaf_rn(s_acc[14], c_row1, 8.0f - _fmax_67);
            float _exp2_16 = approx_exp2(_fma_14);
            s_acc[14] = _exp2_16;
            float _fma_15 = __fmaf_rn(s_acc[15], c_row1, 8.0f - _fmax_67);
            float _exp2_17 = approx_exp2(_fma_15);
            s_acc[15] = _exp2_17;
            float _fma_16 = __fmaf_rn(s_acc[16], c_row0, 8.0f - _fmax_66);
            float _exp2_18 = approx_exp2(_fma_16);
            s_acc[16] = _exp2_18;
            float _fma_17 = __fmaf_rn(s_acc[17], c_row0, 8.0f - _fmax_66);
            float _exp2_19 = approx_exp2(_fma_17);
            s_acc[17] = _exp2_19;
            float _fma_18 = __fmaf_rn(s_acc[18], c_row1, 8.0f - _fmax_67);
            float _exp2_20 = approx_exp2(_fma_18);
            s_acc[18] = _exp2_20;
            float _fma_19 = __fmaf_rn(s_acc[19], c_row1, 8.0f - _fmax_67);
            float _exp2_21 = approx_exp2(_fma_19);
            s_acc[19] = _exp2_21;
            float _fma_20 = __fmaf_rn(s_acc[20], c_row0, 8.0f - _fmax_66);
            float _exp2_22 = approx_exp2(_fma_20);
            s_acc[20] = _exp2_22;
            float _fma_21 = __fmaf_rn(s_acc[21], c_row0, 8.0f - _fmax_66);
            float _exp2_23 = approx_exp2(_fma_21);
            s_acc[21] = _exp2_23;
            float _fma_22 = __fmaf_rn(s_acc[22], c_row1, 8.0f - _fmax_67);
            float _exp2_24 = approx_exp2(_fma_22);
            s_acc[22] = _exp2_24;
            float _fma_23 = __fmaf_rn(s_acc[23], c_row1, 8.0f - _fmax_67);
            float _exp2_25 = approx_exp2(_fma_23);
            s_acc[23] = _exp2_25;
            float _fma_24 = __fmaf_rn(s_acc[24], c_row0, 8.0f - _fmax_66);
            float _exp2_26 = approx_exp2(_fma_24);
            s_acc[24] = _exp2_26;
            float _fma_25 = __fmaf_rn(s_acc[25], c_row0, 8.0f - _fmax_66);
            float _exp2_27 = approx_exp2(_fma_25);
            s_acc[25] = _exp2_27;
            float _fma_26 = __fmaf_rn(s_acc[26], c_row1, 8.0f - _fmax_67);
            float _exp2_28 = approx_exp2(_fma_26);
            s_acc[26] = _exp2_28;
            float _fma_27 = __fmaf_rn(s_acc[27], c_row1, 8.0f - _fmax_67);
            float _exp2_29 = approx_exp2(_fma_27);
            s_acc[27] = _exp2_29;
            float _fma_28 = __fmaf_rn(s_acc[28], c_row0, 8.0f - _fmax_66);
            float _exp2_30 = approx_exp2(_fma_28);
            s_acc[28] = _exp2_30;
            float _fma_29 = __fmaf_rn(s_acc[29], c_row0, 8.0f - _fmax_66);
            float _exp2_31 = approx_exp2(_fma_29);
            s_acc[29] = _exp2_31;
            float _fma_30 = __fmaf_rn(s_acc[30], c_row1, 8.0f - _fmax_67);
            float _exp2_32 = approx_exp2(_fma_30);
            s_acc[30] = _exp2_32;
            float _fma_31 = __fmaf_rn(s_acc[31], c_row1, 8.0f - _fmax_67);
            float _exp2_33 = approx_exp2(_fma_31);
            s_acc[31] = _exp2_33;
            float _fma_32 = __fmaf_rn(s_acc[32], c_row0, 8.0f - _fmax_66);
            float _exp2_34 = approx_exp2(_fma_32);
            s_acc[32] = _exp2_34;
            float _fma_33 = __fmaf_rn(s_acc[33], c_row0, 8.0f - _fmax_66);
            float _exp2_35 = approx_exp2(_fma_33);
            s_acc[33] = _exp2_35;
            float _fma_34 = __fmaf_rn(s_acc[34], c_row1, 8.0f - _fmax_67);
            float _exp2_36 = approx_exp2(_fma_34);
            s_acc[34] = _exp2_36;
            float _fma_35 = __fmaf_rn(s_acc[35], c_row1, 8.0f - _fmax_67);
            float _exp2_37 = approx_exp2(_fma_35);
            s_acc[35] = _exp2_37;
            float _fma_36 = __fmaf_rn(s_acc[36], c_row0, 8.0f - _fmax_66);
            float _exp2_38 = approx_exp2(_fma_36);
            s_acc[36] = _exp2_38;
            float _fma_37 = __fmaf_rn(s_acc[37], c_row0, 8.0f - _fmax_66);
            float _exp2_39 = approx_exp2(_fma_37);
            s_acc[37] = _exp2_39;
            float _fma_38 = __fmaf_rn(s_acc[38], c_row1, 8.0f - _fmax_67);
            float _exp2_40 = approx_exp2(_fma_38);
            s_acc[38] = _exp2_40;
            float _fma_39 = __fmaf_rn(s_acc[39], c_row1, 8.0f - _fmax_67);
            float _exp2_41 = approx_exp2(_fma_39);
            s_acc[39] = _exp2_41;
            float _fma_40 = __fmaf_rn(s_acc[40], c_row0, 8.0f - _fmax_66);
            float _exp2_42 = approx_exp2(_fma_40);
            s_acc[40] = _exp2_42;
            float _fma_41 = __fmaf_rn(s_acc[41], c_row0, 8.0f - _fmax_66);
            float _exp2_43 = approx_exp2(_fma_41);
            s_acc[41] = _exp2_43;
            float _fma_42 = __fmaf_rn(s_acc[42], c_row1, 8.0f - _fmax_67);
            float _exp2_44 = approx_exp2(_fma_42);
            s_acc[42] = _exp2_44;
            float _fma_43 = __fmaf_rn(s_acc[43], c_row1, 8.0f - _fmax_67);
            float _exp2_45 = approx_exp2(_fma_43);
            s_acc[43] = _exp2_45;
            float _fma_44 = __fmaf_rn(s_acc[44], c_row0, 8.0f - _fmax_66);
            float _exp2_46 = approx_exp2(_fma_44);
            s_acc[44] = _exp2_46;
            float _fma_45 = __fmaf_rn(s_acc[45], c_row0, 8.0f - _fmax_66);
            float _exp2_47 = approx_exp2(_fma_45);
            s_acc[45] = _exp2_47;
            float _fma_46 = __fmaf_rn(s_acc[46], c_row1, 8.0f - _fmax_67);
            float _exp2_48 = approx_exp2(_fma_46);
            s_acc[46] = _exp2_48;
            float _fma_47 = __fmaf_rn(s_acc[47], c_row1, 8.0f - _fmax_67);
            float _exp2_49 = approx_exp2(_fma_47);
            s_acc[47] = _exp2_49;
            float _fma_48 = __fmaf_rn(s_acc[48], c_row0, 8.0f - _fmax_66);
            float _exp2_50 = approx_exp2(_fma_48);
            s_acc[48] = _exp2_50;
            float _fma_49 = __fmaf_rn(s_acc[49], c_row0, 8.0f - _fmax_66);
            float _exp2_51 = approx_exp2(_fma_49);
            s_acc[49] = _exp2_51;
            float _fma_50 = __fmaf_rn(s_acc[50], c_row1, 8.0f - _fmax_67);
            float _exp2_52 = approx_exp2(_fma_50);
            s_acc[50] = _exp2_52;
            float _fma_51 = __fmaf_rn(s_acc[51], c_row1, 8.0f - _fmax_67);
            float _exp2_53 = approx_exp2(_fma_51);
            s_acc[51] = _exp2_53;
            float _fma_52 = __fmaf_rn(s_acc[52], c_row0, 8.0f - _fmax_66);
            float _exp2_54 = approx_exp2(_fma_52);
            s_acc[52] = _exp2_54;
            float _fma_53 = __fmaf_rn(s_acc[53], c_row0, 8.0f - _fmax_66);
            float _exp2_55 = approx_exp2(_fma_53);
            s_acc[53] = _exp2_55;
            float _fma_54 = __fmaf_rn(s_acc[54], c_row1, 8.0f - _fmax_67);
            float _exp2_56 = approx_exp2(_fma_54);
            s_acc[54] = _exp2_56;
            float _fma_55 = __fmaf_rn(s_acc[55], c_row1, 8.0f - _fmax_67);
            float _exp2_57 = approx_exp2(_fma_55);
            s_acc[55] = _exp2_57;
            float _fma_56 = __fmaf_rn(s_acc[56], c_row0, 8.0f - _fmax_66);
            float _exp2_58 = approx_exp2(_fma_56);
            s_acc[56] = _exp2_58;
            float _fma_57 = __fmaf_rn(s_acc[57], c_row0, 8.0f - _fmax_66);
            float _exp2_59 = approx_exp2(_fma_57);
            s_acc[57] = _exp2_59;
            float _fma_58 = __fmaf_rn(s_acc[58], c_row1, 8.0f - _fmax_67);
            float _exp2_60 = approx_exp2(_fma_58);
            s_acc[58] = _exp2_60;
            float _fma_59 = __fmaf_rn(s_acc[59], c_row1, 8.0f - _fmax_67);
            float _exp2_61 = approx_exp2(_fma_59);
            s_acc[59] = _exp2_61;
            float _fma_60 = __fmaf_rn(s_acc[60], c_row0, 8.0f - _fmax_66);
            float _exp2_62 = approx_exp2(_fma_60);
            s_acc[60] = _exp2_62;
            float _fma_61 = __fmaf_rn(s_acc[61], c_row0, 8.0f - _fmax_66);
            float _exp2_63 = approx_exp2(_fma_61);
            s_acc[61] = _exp2_63;
            float _fma_62 = __fmaf_rn(s_acc[62], c_row1, 8.0f - _fmax_67);
            float _exp2_64 = approx_exp2(_fma_62);
            s_acc[62] = _exp2_64;
            float _fma_63 = __fmaf_rn(s_acc[63], c_row1, 8.0f - _fmax_67);
            float _exp2_65 = approx_exp2(_fma_63);
            s_acc[63] = _exp2_65;
            float _fma_64 = __fmaf_rn(l_state[0], _exp2_0, _exp2_2 + _exp2_3 + _exp2_6 + _exp2_7 + _exp2_10 + _exp2_11 + _exp2_14 + _exp2_15 + _exp2_18 + _exp2_19 + _exp2_22 + _exp2_23 + _exp2_26 + _exp2_27 + _exp2_30 + _exp2_31 + _exp2_34 + _exp2_35 + _exp2_38 + _exp2_39 + _exp2_42 + _exp2_43 + _exp2_46 + _exp2_47 + _exp2_50 + _exp2_51 + _exp2_54 + _exp2_55 + _exp2_58 + _exp2_59 + _exp2_62 + _exp2_63);
            l_state[0] = _fma_64;
            float _fma_65 = __fmaf_rn(l_state[1], _exp2_1, _exp2_4 + _exp2_5 + _exp2_8 + _exp2_9 + _exp2_12 + _exp2_13 + _exp2_16 + _exp2_17 + _exp2_20 + _exp2_21 + _exp2_24 + _exp2_25 + _exp2_28 + _exp2_29 + _exp2_32 + _exp2_33 + _exp2_36 + _exp2_37 + _exp2_40 + _exp2_41 + _exp2_44 + _exp2_45 + _exp2_48 + _exp2_49 + _exp2_52 + _exp2_53 + _exp2_56 + _exp2_57 + _exp2_60 + _exp2_61 + _exp2_64 + _exp2_65);
            l_state[1] = _fma_65;
            p_tmp[0] = s_acc[0];
            p_tmp[1] = s_acc[1];
            p_tmp[2] = s_acc[4];
            p_tmp[3] = s_acc[5];
            p_tmp[4] = s_acc[2];
            p_tmp[5] = s_acc[3];
            p_tmp[6] = s_acc[6];
            p_tmp[7] = s_acc[7];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[0] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[1] = _packed;
            }
            p_tmp[0] = s_acc[8];
            p_tmp[1] = s_acc[9];
            p_tmp[2] = s_acc[12];
            p_tmp[3] = s_acc[13];
            p_tmp[4] = s_acc[10];
            p_tmp[5] = s_acc[11];
            p_tmp[6] = s_acc[14];
            p_tmp[7] = s_acc[15];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[2] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[3] = _packed;
            }
            p_tmp[0] = s_acc[16];
            p_tmp[1] = s_acc[17];
            p_tmp[2] = s_acc[20];
            p_tmp[3] = s_acc[21];
            p_tmp[4] = s_acc[18];
            p_tmp[5] = s_acc[19];
            p_tmp[6] = s_acc[22];
            p_tmp[7] = s_acc[23];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[4] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[5] = _packed;
            }
            p_tmp[0] = s_acc[24];
            p_tmp[1] = s_acc[25];
            p_tmp[2] = s_acc[28];
            p_tmp[3] = s_acc[29];
            p_tmp[4] = s_acc[26];
            p_tmp[5] = s_acc[27];
            p_tmp[6] = s_acc[30];
            p_tmp[7] = s_acc[31];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[6] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[7] = _packed;
            }
            p_tmp[0] = s_acc[32];
            p_tmp[1] = s_acc[33];
            p_tmp[2] = s_acc[36];
            p_tmp[3] = s_acc[37];
            p_tmp[4] = s_acc[34];
            p_tmp[5] = s_acc[35];
            p_tmp[6] = s_acc[38];
            p_tmp[7] = s_acc[39];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[8] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[9] = _packed;
            }
            p_tmp[0] = s_acc[40];
            p_tmp[1] = s_acc[41];
            p_tmp[2] = s_acc[44];
            p_tmp[3] = s_acc[45];
            p_tmp[4] = s_acc[42];
            p_tmp[5] = s_acc[43];
            p_tmp[6] = s_acc[46];
            p_tmp[7] = s_acc[47];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[10] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[11] = _packed;
            }
            p_tmp[0] = s_acc[48];
            p_tmp[1] = s_acc[49];
            p_tmp[2] = s_acc[52];
            p_tmp[3] = s_acc[53];
            p_tmp[4] = s_acc[50];
            p_tmp[5] = s_acc[51];
            p_tmp[6] = s_acc[54];
            p_tmp[7] = s_acc[55];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[12] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[13] = _packed;
            }
            p_tmp[0] = s_acc[56];
            p_tmp[1] = s_acc[57];
            p_tmp[2] = s_acc[60];
            p_tmp[3] = s_acc[61];
            p_tmp[4] = s_acc[58];
            p_tmp[5] = s_acc[59];
            p_tmp[6] = s_acc[62];
            p_tmp[7] = s_acc[63];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[14] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[15] = _packed;
            }
            mbarrier_wait(v_full_addr + (kv_stage) * 8, kv_phase);
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            if (elect_sync()) {
                mbarrier_arrive(v_empty_addr + (kv_stage) * 8);
            }
            if (warp == 0) {
                mbarrier_wait(k_empty_addr + (kv_stage) * 8, kv_phase);
            }
            if (kb_last >= kb + 2) {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 9216);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(K_stage_addr + kv_stage * 8192), "l"((&K_map)), "r"(0), "r"((kb + 2) * 128), "r"(head),
                               "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(KSF_stage_addr + kv_stage * 1024), "l"((&KSF_map)), "r"(0), "r"((head * num_kblocks + (kb + 2)) * 64),
                               "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (warp == 4) {
                mbarrier_wait(v_empty_addr + (kv_stage) * 8, kv_phase);
            }
            if (kb_last >= kb + 2) {
                if (warp == 4) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(V_stage_addr + kv_stage * 16384), "l"((&V_map)), "r"((kb + 2) * 128), "r"(0), "r"(head),
                               "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            kv_stage += 1;
            if (kv_stage == 2) { kv_stage = 0; kv_phase ^= 1; }
        }
        float l0 = l_state[0];
        float l1 = l_state[1];
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, l0, 1);
        l0 = l0 + _shfl_xor_4;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, l0, 2);
        l0 = l0 + _shfl_xor_5;
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, l1, 1);
        l1 = l1 + _shfl_xor_6;
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, l1, 2);
        l1 = l1 + _shfl_xor_7;
        float _rcp_0 = approx_rcp(l0);
        float inv0 = _rcp_0;
        float _rcp_1 = approx_rcp(l1);
        float inv1 = _rcp_1;
        float _vec_load_3[2];
        {
            float2 _v2_4 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + (lane & 3) * 2) + 0);
            _vec_load_3[0] = _v2_4.x;
            _vec_load_3[0 + 1] = _v2_4.y;
        }
        o_tmp[0] = o_acc[0] * inv0 * _vec_load_3[0];
        o_tmp[1] = o_acc[1] * inv0 * _vec_load_3[1];
        o_tmp[2] = o_acc[2] * inv1 * _vec_load_3[0];
        o_tmp[3] = o_acc[3] * inv1 * _vec_load_3[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2)))[0]) = _pk;
            }
        }
        float _vec_load_4[2];
        {
            float2 _v2_5 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 8 + (lane & 3) * 2) + 0);
            _vec_load_4[0] = _v2_5.x;
            _vec_load_4[0 + 1] = _v2_5.y;
        }
        o_tmp[0] = o_acc[4] * inv0 * _vec_load_4[0];
        o_tmp[1] = o_acc[5] * inv0 * _vec_load_4[1];
        o_tmp[2] = o_acc[6] * inv1 * _vec_load_4[0];
        o_tmp[3] = o_acc[7] * inv1 * _vec_load_4[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 8)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 8)))[0]) = _pk;
            }
        }
        float _vec_load_5[2];
        {
            float2 _v2_6 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 16 + (lane & 3) * 2) + 0);
            _vec_load_5[0] = _v2_6.x;
            _vec_load_5[0 + 1] = _v2_6.y;
        }
        o_tmp[0] = o_acc[8] * inv0 * _vec_load_5[0];
        o_tmp[1] = o_acc[9] * inv0 * _vec_load_5[1];
        o_tmp[2] = o_acc[10] * inv1 * _vec_load_5[0];
        o_tmp[3] = o_acc[11] * inv1 * _vec_load_5[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 16)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 16)))[0]) = _pk;
            }
        }
        float _vec_load_6[2];
        {
            float2 _v2_7 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 24 + (lane & 3) * 2) + 0);
            _vec_load_6[0] = _v2_7.x;
            _vec_load_6[0 + 1] = _v2_7.y;
        }
        o_tmp[0] = o_acc[12] * inv0 * _vec_load_6[0];
        o_tmp[1] = o_acc[13] * inv0 * _vec_load_6[1];
        o_tmp[2] = o_acc[14] * inv1 * _vec_load_6[0];
        o_tmp[3] = o_acc[15] * inv1 * _vec_load_6[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 24)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 24)))[0]) = _pk;
            }
        }
        float _vec_load_7[2];
        {
            float2 _v2_8 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 32 + (lane & 3) * 2) + 0);
            _vec_load_7[0] = _v2_8.x;
            _vec_load_7[0 + 1] = _v2_8.y;
        }
        o_tmp[0] = o_acc[16] * inv0 * _vec_load_7[0];
        o_tmp[1] = o_acc[17] * inv0 * _vec_load_7[1];
        o_tmp[2] = o_acc[18] * inv1 * _vec_load_7[0];
        o_tmp[3] = o_acc[19] * inv1 * _vec_load_7[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 32)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 32)))[0]) = _pk;
            }
        }
        float _vec_load_8[2];
        {
            float2 _v2_9 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 40 + (lane & 3) * 2) + 0);
            _vec_load_8[0] = _v2_9.x;
            _vec_load_8[0 + 1] = _v2_9.y;
        }
        o_tmp[0] = o_acc[20] * inv0 * _vec_load_8[0];
        o_tmp[1] = o_acc[21] * inv0 * _vec_load_8[1];
        o_tmp[2] = o_acc[22] * inv1 * _vec_load_8[0];
        o_tmp[3] = o_acc[23] * inv1 * _vec_load_8[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 40)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 40)))[0]) = _pk;
            }
        }
        float _vec_load_9[2];
        {
            float2 _v2_10 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 48 + (lane & 3) * 2) + 0);
            _vec_load_9[0] = _v2_10.x;
            _vec_load_9[0 + 1] = _v2_10.y;
        }
        o_tmp[0] = o_acc[24] * inv0 * _vec_load_9[0];
        o_tmp[1] = o_acc[25] * inv0 * _vec_load_9[1];
        o_tmp[2] = o_acc[26] * inv1 * _vec_load_9[0];
        o_tmp[3] = o_acc[27] * inv1 * _vec_load_9[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 48)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 48)))[0]) = _pk;
            }
        }
        float _vec_load_10[2];
        {
            float2 _v2_11 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 56 + (lane & 3) * 2) + 0);
            _vec_load_10[0] = _v2_11.x;
            _vec_load_10[0 + 1] = _v2_11.y;
        }
        o_tmp[0] = o_acc[28] * inv0 * _vec_load_10[0];
        o_tmp[1] = o_acc[29] * inv0 * _vec_load_10[1];
        o_tmp[2] = o_acc[30] * inv1 * _vec_load_10[0];
        o_tmp[3] = o_acc[31] * inv1 * _vec_load_10[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 56)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 56)))[0]) = _pk;
            }
        }
        float _vec_load_11[2];
        {
            float2 _v2_12 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 64 + (lane & 3) * 2) + 0);
            _vec_load_11[0] = _v2_12.x;
            _vec_load_11[0 + 1] = _v2_12.y;
        }
        o_tmp[0] = o_acc[32] * inv0 * _vec_load_11[0];
        o_tmp[1] = o_acc[33] * inv0 * _vec_load_11[1];
        o_tmp[2] = o_acc[34] * inv1 * _vec_load_11[0];
        o_tmp[3] = o_acc[35] * inv1 * _vec_load_11[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 64)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 64)))[0]) = _pk;
            }
        }
        float _vec_load_12[2];
        {
            float2 _v2_13 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 72 + (lane & 3) * 2) + 0);
            _vec_load_12[0] = _v2_13.x;
            _vec_load_12[0 + 1] = _v2_13.y;
        }
        o_tmp[0] = o_acc[36] * inv0 * _vec_load_12[0];
        o_tmp[1] = o_acc[37] * inv0 * _vec_load_12[1];
        o_tmp[2] = o_acc[38] * inv1 * _vec_load_12[0];
        o_tmp[3] = o_acc[39] * inv1 * _vec_load_12[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 72)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 72)))[0]) = _pk;
            }
        }
        float _vec_load_13[2];
        {
            float2 _v2_14 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 80 + (lane & 3) * 2) + 0);
            _vec_load_13[0] = _v2_14.x;
            _vec_load_13[0 + 1] = _v2_14.y;
        }
        o_tmp[0] = o_acc[40] * inv0 * _vec_load_13[0];
        o_tmp[1] = o_acc[41] * inv0 * _vec_load_13[1];
        o_tmp[2] = o_acc[42] * inv1 * _vec_load_13[0];
        o_tmp[3] = o_acc[43] * inv1 * _vec_load_13[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 80)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 80)))[0]) = _pk;
            }
        }
        float _vec_load_14[2];
        {
            float2 _v2_15 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 88 + (lane & 3) * 2) + 0);
            _vec_load_14[0] = _v2_15.x;
            _vec_load_14[0 + 1] = _v2_15.y;
        }
        o_tmp[0] = o_acc[44] * inv0 * _vec_load_14[0];
        o_tmp[1] = o_acc[45] * inv0 * _vec_load_14[1];
        o_tmp[2] = o_acc[46] * inv1 * _vec_load_14[0];
        o_tmp[3] = o_acc[47] * inv1 * _vec_load_14[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 88)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 88)))[0]) = _pk;
            }
        }
        float _vec_load_15[2];
        {
            float2 _v2_16 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 96 + (lane & 3) * 2) + 0);
            _vec_load_15[0] = _v2_16.x;
            _vec_load_15[0 + 1] = _v2_16.y;
        }
        o_tmp[0] = o_acc[48] * inv0 * _vec_load_15[0];
        o_tmp[1] = o_acc[49] * inv0 * _vec_load_15[1];
        o_tmp[2] = o_acc[50] * inv1 * _vec_load_15[0];
        o_tmp[3] = o_acc[51] * inv1 * _vec_load_15[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 96)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 96)))[0]) = _pk;
            }
        }
        float _vec_load_16[2];
        {
            float2 _v2_17 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 104 + (lane & 3) * 2) + 0);
            _vec_load_16[0] = _v2_17.x;
            _vec_load_16[0 + 1] = _v2_17.y;
        }
        o_tmp[0] = o_acc[52] * inv0 * _vec_load_16[0];
        o_tmp[1] = o_acc[53] * inv0 * _vec_load_16[1];
        o_tmp[2] = o_acc[54] * inv1 * _vec_load_16[0];
        o_tmp[3] = o_acc[55] * inv1 * _vec_load_16[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 104)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 104)))[0]) = _pk;
            }
        }
        float _vec_load_17[2];
        {
            float2 _v2_18 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 112 + (lane & 3) * 2) + 0);
            _vec_load_17[0] = _v2_18.x;
            _vec_load_17[0 + 1] = _v2_18.y;
        }
        o_tmp[0] = o_acc[56] * inv0 * _vec_load_17[0];
        o_tmp[1] = o_acc[57] * inv0 * _vec_load_17[1];
        o_tmp[2] = o_acc[58] * inv1 * _vec_load_17[0];
        o_tmp[3] = o_acc[59] * inv1 * _vec_load_17[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 112)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 112)))[0]) = _pk;
            }
        }
        float _vec_load_18[2];
        {
            float2 _v2_19 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 120 + (lane & 3) * 2) + 0);
            _vec_load_18[0] = _v2_19.x;
            _vec_load_18[0 + 1] = _v2_19.y;
        }
        o_tmp[0] = o_acc[60] * inv0 * _vec_load_18[0];
        o_tmp[1] = o_acc[61] * inv0 * _vec_load_18[1];
        o_tmp[2] = o_acc[62] * inv1 * _vec_load_18[0];
        o_tmp[3] = o_acc[63] * inv1 * _vec_load_18[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 120)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 120)))[0]) = _pk;
            }
        }
        __syncthreads();
    }
    mbarrier_wait(q_full_addr, _phase_q_full_0);
    _phase_q_full_0 ^= 1;

    // Cleanup
    __syncthreads();
}

}  // namespace h3_varlen_attention_nvfp4_sm120a
#undef H3_VARLEN_INF
#undef NUM_KV_STAGES
#undef SMEM_KSF_STAGE_OFF
#undef SMEM_KSF_STAGE_STAGE_BYTES
#undef SMEM_KSF_STAGE_STRIDE
#undef SMEM_K_STAGE_OFF
#undef SMEM_K_STAGE_STAGE_BYTES
#undef SMEM_K_STAGE_STRIDE
#undef SMEM_Q_OFF
#undef SMEM_Q_STAGE_BYTES
#undef SMEM_Q_STRIDE
#undef SMEM_TOTAL
#undef SMEM_V_STAGE_OFF
#undef SMEM_V_STAGE_STAGE_BYTES
#undef SMEM_V_STAGE_STRIDE
#undef THREADS
#undef k_empty_addr
#undef k_full_addr
#undef q_full_addr
#undef v_empty_addr
#undef v_full_addr

namespace h3_varlen_attention_nvfp4_short_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_VARLEN_INF CUDART_INF_F
#define NUM_KV_STAGES 2
#define SMEM_Q_OFF 1024
#define SMEM_Q_STAGE_BYTES 8192
#define SMEM_Q_STRIDE 8192
#define SMEM_K_STAGE_OFF 9216
#define SMEM_K_STAGE_STAGE_BYTES 8192
#define SMEM_K_STAGE_STRIDE 8192
#define SMEM_KSF_STAGE_OFF 25600
#define SMEM_KSF_STAGE_STAGE_BYTES 1024
#define SMEM_KSF_STAGE_STRIDE 1024
#define SMEM_V_STAGE_OFF 27648
#define SMEM_V_STAGE_STAGE_BYTES 16384
#define SMEM_V_STAGE_STRIDE 16384
#define SMEM_TOTAL 60416
#define THREADS 256

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


// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
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


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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
kernel_minimax_h3_sm120_varlen_attention_nvfp4_short(const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map, const __grid_constant__ CUtensorMap KSF_map, const __grid_constant__ CUtensorMap V_map, __nv_bfloat16* __restrict__ O, float* __restrict__ q_scale, float* __restrict__ k_scale, float* __restrict__ v_scale, unsigned int* __restrict__ q_sf, int* __restrict__ unit_table, int total_units, int num_heads, int num_kblocks, float softmax_scale_log2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define k_full_addr (mbar_base + 8)
    #define v_full_addr (mbar_base + 24)
    #define k_empty_addr (mbar_base + 40)
    #define v_empty_addr (mbar_base + 56)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* Q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int Q_addr = smem + 1024;
    uint8_t* K_stage = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int K_stage_addr = smem + 9216;
    unsigned int* KSF_stage = reinterpret_cast<unsigned int*>(smem_raw + 25600);
    const int KSF_stage_addr = smem + 25600;
    uint8_t* V_stage = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int V_stage_addr = smem + 27648;

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 9 barriers)
    // Mbarriers at smem_raw[0..72)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'kv' ---
            // k_full: 2 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // k_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 40, 8);
            mbarrier_init(smem + 48, 8);
            // v_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 56, 8);
            mbarrier_init(smem + 64, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    unsigned int kv_stage = 0;
    unsigned int kv_phase = 0;
    if (warp >= 4) {
        asm volatile("barrier.arrive 1, 256;" ::: "memory");
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int _vec_load_0[4];
    {
        const int4* _ivptr_0 = reinterpret_cast<const int4*>(unit_table + (4 * bid) + 0);
        int4 _ivld_0;
        _ivld_0 = *_ivptr_0;
        _vec_load_0[0 + 0] = _ivld_0.x;
        _vec_load_0[0 + 1] = _ivld_0.y;
        _vec_load_0[0 + 2] = _ivld_0.z;
        _vec_load_0[0 + 3] = _ivld_0.w;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(q_full_addr, 8192);
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(Q_addr), "l"((&Q_map)), "r"(0), "r"(_vec_load_0[2] + (_vec_load_0[1] & 65535) * 128), "r"(_vec_load_0[1] >> 16),
                   "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    unsigned int _phase_q_full_0 = 0;
    #pragma unroll 1
    for (int slot = bid; slot < total_units; slot += num_bids) {
        int _vec_load_1[4];
        {
            const int4* _ivptr_1 = reinterpret_cast<const int4*>(unit_table + (4 * slot) + 0);
            int4 _ivld_1;
            _ivld_1 = *_ivptr_1;
            _vec_load_1[0 + 0] = _ivld_1.x;
            _vec_load_1[0 + 1] = _ivld_1.y;
            _vec_load_1[0 + 2] = _ivld_1.z;
            _vec_load_1[0 + 3] = _ivld_1.w;
        }
        int seg = _vec_load_1[0];
        int packed = _vec_load_1[1];
        int head = packed >> 16;
        int q_tile = packed & 65535;
        int seg_begin_v = _vec_load_1[2];
        int seg_end_v = _vec_load_1[3];
        int q_row0 = seg_begin_v + q_tile * 128;
        int kb_first = seg_begin_v / 128;
        int kb_last = (seg_end_v - 1) / 128;
        int next_slot = ((slot + num_bids < total_units) ? slot + num_bids : slot);
        int _vec_load_2[4];
        {
            const int4* _ivptr_2 = reinterpret_cast<const int4*>(unit_table + (4 * next_slot) + 0);
            int4 _ivld_2;
            _ivld_2 = *_ivptr_2;
            _vec_load_2[0 + 0] = _ivld_2.x;
            _vec_load_2[0 + 1] = _ivld_2.y;
            _vec_load_2[0 + 2] = _ivld_2.z;
            _vec_load_2[0 + 3] = _ivld_2.w;
        }
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 9216);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                    :: "r"(K_stage_addr + kv_stage * 8192), "l"((&K_map)), "r"(0), "r"(kb_first * 128), "r"(head),
                       "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3}], [%4], %5;"
                    :: "r"(KSF_stage_addr + kv_stage * 1024), "l"((&KSF_map)), "r"(0), "r"((head * num_kblocks + kb_first) * 64),
                       "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 16384);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                    :: "r"(V_stage_addr + kv_stage * 16384), "l"((&V_map)), "r"(kb_first * 128), "r"(0), "r"(head),
                       "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
            }
        }
        if (kb_last >= kb_first + 1) {
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(k_full_addr + ((kv_stage + 1) % 2) * 8, 9216);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(K_stage_addr + (kv_stage + 1) % 2 * 8192), "l"((&K_map)), "r"(0), "r"((kb_first + 1) * 128), "r"(head),
                           "r"(k_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(KSF_stage_addr + (kv_stage + 1) % 2 * 1024), "l"((&KSF_map)), "r"(0), "r"((head * num_kblocks + (kb_first + 1)) * 64),
                           "r"(k_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    mbarrier_arrive_expect_tx(v_full_addr + ((kv_stage + 1) % 2) * 8, 16384);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(V_stage_addr + (kv_stage + 1) % 2 * 16384), "l"((&V_map)), "r"((kb_first + 1) * 128), "r"(0), "r"(head),
                           "r"(v_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        int row0 = q_row0 + warp * 16 + (lane >> 2);
        int row1 = row0 + 8;
        int row0_c = ((row0 < seg_end_v) ? row0 : seg_end_v - 1);
        int row1_c = ((row1 < seg_end_v) ? row1 : seg_end_v - 1);
        float qs0 = q_scale[row0_c * num_heads + head];
        float qs1 = q_scale[row1_c * num_heads + head];
        float ks_seg = k_scale[seg * num_heads + head];
        float c_row0 = qs0 * softmax_scale_log2 * ks_seg;
        float c_row1 = qs1 * softmax_scale_log2 * ks_seg;
        int sf_row = (((lane & 1) == 1) ? row1_c : row0_c);
        unsigned int qsf[2];
        for (int u = 0; u < 2; u++) {
            qsf[u] = q_sf[(sf_row * num_heads + head) * 2 + u];
        }
        unsigned int q_frags[8];
        unsigned int k_frag[4];
        unsigned int v_frag[4];
        float s_acc[64];
        unsigned int p_frag[16];
        float p_tmp[8];
        unsigned int x16[1];
        float o_acc[64];
        float o_tmp[4];
        float m_state[2];
        float l_state[2];
        o_acc[0] = 0.0f;
        o_acc[1] = 0.0f;
        o_acc[2] = 0.0f;
        o_acc[3] = 0.0f;
        o_acc[4] = 0.0f;
        o_acc[5] = 0.0f;
        o_acc[6] = 0.0f;
        o_acc[7] = 0.0f;
        o_acc[8] = 0.0f;
        o_acc[9] = 0.0f;
        o_acc[10] = 0.0f;
        o_acc[11] = 0.0f;
        o_acc[12] = 0.0f;
        o_acc[13] = 0.0f;
        o_acc[14] = 0.0f;
        o_acc[15] = 0.0f;
        o_acc[16] = 0.0f;
        o_acc[17] = 0.0f;
        o_acc[18] = 0.0f;
        o_acc[19] = 0.0f;
        o_acc[20] = 0.0f;
        o_acc[21] = 0.0f;
        o_acc[22] = 0.0f;
        o_acc[23] = 0.0f;
        o_acc[24] = 0.0f;
        o_acc[25] = 0.0f;
        o_acc[26] = 0.0f;
        o_acc[27] = 0.0f;
        o_acc[28] = 0.0f;
        o_acc[29] = 0.0f;
        o_acc[30] = 0.0f;
        o_acc[31] = 0.0f;
        o_acc[32] = 0.0f;
        o_acc[33] = 0.0f;
        o_acc[34] = 0.0f;
        o_acc[35] = 0.0f;
        o_acc[36] = 0.0f;
        o_acc[37] = 0.0f;
        o_acc[38] = 0.0f;
        o_acc[39] = 0.0f;
        o_acc[40] = 0.0f;
        o_acc[41] = 0.0f;
        o_acc[42] = 0.0f;
        o_acc[43] = 0.0f;
        o_acc[44] = 0.0f;
        o_acc[45] = 0.0f;
        o_acc[46] = 0.0f;
        o_acc[47] = 0.0f;
        o_acc[48] = 0.0f;
        o_acc[49] = 0.0f;
        o_acc[50] = 0.0f;
        o_acc[51] = 0.0f;
        o_acc[52] = 0.0f;
        o_acc[53] = 0.0f;
        o_acc[54] = 0.0f;
        o_acc[55] = 0.0f;
        o_acc[56] = 0.0f;
        o_acc[57] = 0.0f;
        o_acc[58] = 0.0f;
        o_acc[59] = 0.0f;
        o_acc[60] = 0.0f;
        o_acc[61] = 0.0f;
        o_acc[62] = 0.0f;
        o_acc[63] = 0.0f;
        m_state[0] = -H3_VARLEN_INF;
        m_state[1] = -H3_VARLEN_INF;
        l_state[0] = 0.0f;
        l_state[1] = 0.0f;
        mbarrier_wait(q_full_addr, _phase_q_full_0);
        _phase_q_full_0 ^= 1;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[0]), "=r"(q_frags[1]), "=r"(q_frags[2]), "=r"(q_frags[3])
            : "r"(Q_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[4]), "=r"(q_frags[5]), "=r"(q_frags[6]), "=r"(q_frags[7])
            : "r"(Q_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
            : "memory");
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 8192);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                    :: "r"(Q_addr), "l"((&Q_map)), "r"(0), "r"(_vec_load_2[2] + (_vec_load_2[1] & 65535) * 128), "r"(_vec_load_2[1] >> 16),
                       "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            }
        }
        #pragma unroll 1
        for (int kb = kb_first; kb < kb_last + 1; kb++) {
            mbarrier_wait(k_full_addr + (kv_stage) * 8, kv_phase);
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_0[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_0[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((lane >> 2) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[0]), "=f"(s_acc[1]), "=f"(s_acc[2]), "=f"(s_acc[3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_1[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_1[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[4]), "=f"(s_acc[(4) + 1]), "=f"(s_acc[(4) + 2]), "=f"(s_acc[(4) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_2[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_2[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[8]), "=f"(s_acc[(8) + 1]), "=f"(s_acc[(8) + 2]), "=f"(s_acc[(8) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_3[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_3[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[12]), "=f"(s_acc[(12) + 1]), "=f"(s_acc[(12) + 2]), "=f"(s_acc[(12) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_4[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_4[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((32 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[16]), "=f"(s_acc[(16) + 1]), "=f"(s_acc[(16) + 2]), "=f"(s_acc[(16) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_5[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_5[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((40 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[20]), "=f"(s_acc[(20) + 1]), "=f"(s_acc[(20) + 2]), "=f"(s_acc[(20) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_6[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_6[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((48 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[24]), "=f"(s_acc[(24) + 1]), "=f"(s_acc[(24) + 2]), "=f"(s_acc[(24) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_7[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_7[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((56 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[28]), "=f"(s_acc[(28) + 1]), "=f"(s_acc[(28) + 2]), "=f"(s_acc[(28) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_8[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_8[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((64 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[32]), "=f"(s_acc[(32) + 1]), "=f"(s_acc[(32) + 2]), "=f"(s_acc[(32) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_9[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_9[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((72 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[36]), "=f"(s_acc[(36) + 1]), "=f"(s_acc[(36) + 2]), "=f"(s_acc[(36) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_10[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_10[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((80 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[40]), "=f"(s_acc[(40) + 1]), "=f"(s_acc[(40) + 2]), "=f"(s_acc[(40) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_11[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_11[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((88 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[44]), "=f"(s_acc[(44) + 1]), "=f"(s_acc[(44) + 2]), "=f"(s_acc[(44) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_12[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_12[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((96 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[48]), "=f"(s_acc[(48) + 1]), "=f"(s_acc[(48) + 2]), "=f"(s_acc[(48) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_13[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_13[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((104 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[52]), "=f"(s_acc[(52) + 1]), "=f"(s_acc[(52) + 2]), "=f"(s_acc[(52) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_14[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_14[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((112 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[56]), "=f"(s_acc[(56) + 1]), "=f"(s_acc[(56) + 2]), "=f"(s_acc[(56) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_15[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_15[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((120 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[60]), "=f"(s_acc[(60) + 1]), "=f"(s_acc[(60) + 2]), "=f"(s_acc[(60) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_16[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_16[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((lane >> 2) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_17[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_17[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_18[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_18[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[8]), "+f"(s_acc[(8) + 1]), "+f"(s_acc[(8) + 2]), "+f"(s_acc[(8) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_19[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_19[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[12]), "+f"(s_acc[(12) + 1]), "+f"(s_acc[(12) + 2]), "+f"(s_acc[(12) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_20[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_20[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((32 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[16]), "+f"(s_acc[(16) + 1]), "+f"(s_acc[(16) + 2]), "+f"(s_acc[(16) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_21[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_21[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((40 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[20]), "+f"(s_acc[(20) + 1]), "+f"(s_acc[(20) + 2]), "+f"(s_acc[(20) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_22[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_22[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((48 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[24]), "+f"(s_acc[(24) + 1]), "+f"(s_acc[(24) + 2]), "+f"(s_acc[(24) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_23[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_23[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((56 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[28]), "+f"(s_acc[(28) + 1]), "+f"(s_acc[(28) + 2]), "+f"(s_acc[(28) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_23[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_24[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_24[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((64 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[32]), "+f"(s_acc[(32) + 1]), "+f"(s_acc[(32) + 2]), "+f"(s_acc[(32) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_24[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_25[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_25[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((72 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[36]), "+f"(s_acc[(36) + 1]), "+f"(s_acc[(36) + 2]), "+f"(s_acc[(36) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_25[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_26[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_26[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((80 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[40]), "+f"(s_acc[(40) + 1]), "+f"(s_acc[(40) + 2]), "+f"(s_acc[(40) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_26[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_27[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_27[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((88 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[44]), "+f"(s_acc[(44) + 1]), "+f"(s_acc[(44) + 2]), "+f"(s_acc[(44) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_28[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_28[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((96 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[48]), "+f"(s_acc[(48) + 1]), "+f"(s_acc[(48) + 2]), "+f"(s_acc[(48) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_29[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_29[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((104 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[52]), "+f"(s_acc[(52) + 1]), "+f"(s_acc[(52) + 2]), "+f"(s_acc[(52) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_30[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_30[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((112 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[56]), "+f"(s_acc[(56) + 1]), "+f"(s_acc[(56) + 2]), "+f"(s_acc[(56) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_31[1];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _KSF_stage_reg_31[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((120 + (lane >> 2)) * 2) + 1) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[60]), "+f"(s_acc[(60) + 1]), "+f"(s_acc[(60) + 2]), "+f"(s_acc[(60) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_31[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            if (elect_sync()) {
                mbarrier_arrive(k_empty_addr + (kv_stage) * 8);
            }
            if (kb_first == kb || kb_last == kb) {
                s_acc[0] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[0]);
                s_acc[1] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[1]);
                s_acc[2] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[2]);
                s_acc[3] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[3]);
                s_acc[4] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[4]);
                s_acc[5] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[5]);
                s_acc[6] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[6]);
                s_acc[7] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[7]);
                s_acc[8] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[8]);
                s_acc[9] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[9]);
                s_acc[10] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[10]);
                s_acc[11] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[11]);
                s_acc[12] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[12]);
                s_acc[13] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[13]);
                s_acc[14] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[14]);
                s_acc[15] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[15]);
                s_acc[16] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[16]);
                s_acc[17] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[17]);
                s_acc[18] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[18]);
                s_acc[19] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[19]);
                s_acc[20] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[20]);
                s_acc[21] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[21]);
                s_acc[22] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[22]);
                s_acc[23] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[23]);
                s_acc[24] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[24]);
                s_acc[25] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[25]);
                s_acc[26] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[26]);
                s_acc[27] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[27]);
                s_acc[28] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[28]);
                s_acc[29] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[29]);
                s_acc[30] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[30]);
                s_acc[31] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[31]);
                s_acc[32] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[32]);
                s_acc[33] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[33]);
                s_acc[34] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[34]);
                s_acc[35] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[35]);
                s_acc[36] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[36]);
                s_acc[37] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[37]);
                s_acc[38] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[38]);
                s_acc[39] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[39]);
                s_acc[40] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[40]);
                s_acc[41] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[41]);
                s_acc[42] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[42]);
                s_acc[43] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[43]);
                s_acc[44] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[44]);
                s_acc[45] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[45]);
                s_acc[46] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[46]);
                s_acc[47] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[47]);
                s_acc[48] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[48]);
                s_acc[49] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[49]);
                s_acc[50] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[50]);
                s_acc[51] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[51]);
                s_acc[52] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[52]);
                s_acc[53] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[53]);
                s_acc[54] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[54]);
                s_acc[55] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[55]);
                s_acc[56] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[56]);
                s_acc[57] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[57]);
                s_acc[58] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[58]);
                s_acc[59] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[59]);
                s_acc[60] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[60]);
                s_acc[61] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[61]);
                s_acc[62] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3)) ? -H3_VARLEN_INF : s_acc[62]);
                s_acc[63] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3) + 1) ? -H3_VARLEN_INF : s_acc[63]);
            }
            float _fmax_0 = fmaxf(s_acc[0], s_acc[1]);
            float _fmax_1 = fmaxf(s_acc[2], s_acc[3]);
            float _fmax_2 = fmaxf(s_acc[4], s_acc[5]);
            float _fmax_3 = fmaxf(_fmax_0, _fmax_2);
            float _fmax_4 = fmaxf(s_acc[6], s_acc[7]);
            float _fmax_5 = fmaxf(_fmax_1, _fmax_4);
            float _fmax_6 = fmaxf(s_acc[8], s_acc[9]);
            float _fmax_7 = fmaxf(_fmax_3, _fmax_6);
            float _fmax_8 = fmaxf(s_acc[10], s_acc[11]);
            float _fmax_9 = fmaxf(_fmax_5, _fmax_8);
            float _fmax_10 = fmaxf(s_acc[12], s_acc[13]);
            float _fmax_11 = fmaxf(_fmax_7, _fmax_10);
            float _fmax_12 = fmaxf(s_acc[14], s_acc[15]);
            float _fmax_13 = fmaxf(_fmax_9, _fmax_12);
            float _fmax_14 = fmaxf(s_acc[16], s_acc[17]);
            float _fmax_15 = fmaxf(_fmax_11, _fmax_14);
            float _fmax_16 = fmaxf(s_acc[18], s_acc[19]);
            float _fmax_17 = fmaxf(_fmax_13, _fmax_16);
            float _fmax_18 = fmaxf(s_acc[20], s_acc[21]);
            float _fmax_19 = fmaxf(_fmax_15, _fmax_18);
            float _fmax_20 = fmaxf(s_acc[22], s_acc[23]);
            float _fmax_21 = fmaxf(_fmax_17, _fmax_20);
            float _fmax_22 = fmaxf(s_acc[24], s_acc[25]);
            float _fmax_23 = fmaxf(_fmax_19, _fmax_22);
            float _fmax_24 = fmaxf(s_acc[26], s_acc[27]);
            float _fmax_25 = fmaxf(_fmax_21, _fmax_24);
            float _fmax_26 = fmaxf(s_acc[28], s_acc[29]);
            float _fmax_27 = fmaxf(_fmax_23, _fmax_26);
            float _fmax_28 = fmaxf(s_acc[30], s_acc[31]);
            float _fmax_29 = fmaxf(_fmax_25, _fmax_28);
            float _fmax_30 = fmaxf(s_acc[32], s_acc[33]);
            float _fmax_31 = fmaxf(_fmax_27, _fmax_30);
            float _fmax_32 = fmaxf(s_acc[34], s_acc[35]);
            float _fmax_33 = fmaxf(_fmax_29, _fmax_32);
            float _fmax_34 = fmaxf(s_acc[36], s_acc[37]);
            float _fmax_35 = fmaxf(_fmax_31, _fmax_34);
            float _fmax_36 = fmaxf(s_acc[38], s_acc[39]);
            float _fmax_37 = fmaxf(_fmax_33, _fmax_36);
            float _fmax_38 = fmaxf(s_acc[40], s_acc[41]);
            float _fmax_39 = fmaxf(_fmax_35, _fmax_38);
            float _fmax_40 = fmaxf(s_acc[42], s_acc[43]);
            float _fmax_41 = fmaxf(_fmax_37, _fmax_40);
            float _fmax_42 = fmaxf(s_acc[44], s_acc[45]);
            float _fmax_43 = fmaxf(_fmax_39, _fmax_42);
            float _fmax_44 = fmaxf(s_acc[46], s_acc[47]);
            float _fmax_45 = fmaxf(_fmax_41, _fmax_44);
            float _fmax_46 = fmaxf(s_acc[48], s_acc[49]);
            float _fmax_47 = fmaxf(_fmax_43, _fmax_46);
            float _fmax_48 = fmaxf(s_acc[50], s_acc[51]);
            float _fmax_49 = fmaxf(_fmax_45, _fmax_48);
            float _fmax_50 = fmaxf(s_acc[52], s_acc[53]);
            float _fmax_51 = fmaxf(_fmax_47, _fmax_50);
            float _fmax_52 = fmaxf(s_acc[54], s_acc[55]);
            float _fmax_53 = fmaxf(_fmax_49, _fmax_52);
            float _fmax_54 = fmaxf(s_acc[56], s_acc[57]);
            float _fmax_55 = fmaxf(_fmax_51, _fmax_54);
            float _fmax_56 = fmaxf(s_acc[58], s_acc[59]);
            float _fmax_57 = fmaxf(_fmax_53, _fmax_56);
            float _fmax_58 = fmaxf(s_acc[60], s_acc[61]);
            float _fmax_59 = fmaxf(_fmax_55, _fmax_58);
            float _fmax_60 = fmaxf(s_acc[62], s_acc[63]);
            float _fmax_61 = fmaxf(_fmax_57, _fmax_60);
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, _fmax_59, 1);
            float _fmax_62 = fmaxf(_fmax_59, _shfl_xor_0);
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, _fmax_62, 2);
            float _fmax_63 = fmaxf(_fmax_62, _shfl_xor_1);
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, _fmax_61, 1);
            float _fmax_64 = fmaxf(_fmax_61, _shfl_xor_2);
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, _fmax_64, 2);
            float _fmax_65 = fmaxf(_fmax_64, _shfl_xor_3);
            float _fmax_66 = fmaxf(m_state[0], _fmax_63 * c_row0);
            float _fmax_67 = fmaxf(m_state[1], _fmax_65 * c_row1);
            float _exp2_0 = approx_exp2(m_state[0] - _fmax_66);
            float _exp2_1 = approx_exp2(m_state[1] - _fmax_67);
            m_state[0] = _fmax_66;
            m_state[1] = _fmax_67;
            {
                float2 _pair_scale_even2_3 = make_float2(_exp2_0, _exp2_0);
                float2 _pair_scale_odd2_3 = make_float2(_exp2_1, _exp2_1);
                float2* _pair_scale_src2_3 = reinterpret_cast<float2*>(&o_acc[0]);
                #if __CUDA_ARCH__ >= 1000
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                #else
                o_acc[0] *= _exp2_0;
                o_acc[1] *= _exp2_0;
                o_acc[2] *= _exp2_1;
                o_acc[3] *= _exp2_1;
                o_acc[4] *= _exp2_0;
                o_acc[5] *= _exp2_0;
                o_acc[6] *= _exp2_1;
                o_acc[7] *= _exp2_1;
                o_acc[8] *= _exp2_0;
                o_acc[9] *= _exp2_0;
                o_acc[10] *= _exp2_1;
                o_acc[11] *= _exp2_1;
                o_acc[12] *= _exp2_0;
                o_acc[13] *= _exp2_0;
                o_acc[14] *= _exp2_1;
                o_acc[15] *= _exp2_1;
                o_acc[16] *= _exp2_0;
                o_acc[17] *= _exp2_0;
                o_acc[18] *= _exp2_1;
                o_acc[19] *= _exp2_1;
                o_acc[20] *= _exp2_0;
                o_acc[21] *= _exp2_0;
                o_acc[22] *= _exp2_1;
                o_acc[23] *= _exp2_1;
                o_acc[24] *= _exp2_0;
                o_acc[25] *= _exp2_0;
                o_acc[26] *= _exp2_1;
                o_acc[27] *= _exp2_1;
                o_acc[28] *= _exp2_0;
                o_acc[29] *= _exp2_0;
                o_acc[30] *= _exp2_1;
                o_acc[31] *= _exp2_1;
                o_acc[32] *= _exp2_0;
                o_acc[33] *= _exp2_0;
                o_acc[34] *= _exp2_1;
                o_acc[35] *= _exp2_1;
                o_acc[36] *= _exp2_0;
                o_acc[37] *= _exp2_0;
                o_acc[38] *= _exp2_1;
                o_acc[39] *= _exp2_1;
                o_acc[40] *= _exp2_0;
                o_acc[41] *= _exp2_0;
                o_acc[42] *= _exp2_1;
                o_acc[43] *= _exp2_1;
                o_acc[44] *= _exp2_0;
                o_acc[45] *= _exp2_0;
                o_acc[46] *= _exp2_1;
                o_acc[47] *= _exp2_1;
                o_acc[48] *= _exp2_0;
                o_acc[49] *= _exp2_0;
                o_acc[50] *= _exp2_1;
                o_acc[51] *= _exp2_1;
                o_acc[52] *= _exp2_0;
                o_acc[53] *= _exp2_0;
                o_acc[54] *= _exp2_1;
                o_acc[55] *= _exp2_1;
                o_acc[56] *= _exp2_0;
                o_acc[57] *= _exp2_0;
                o_acc[58] *= _exp2_1;
                o_acc[59] *= _exp2_1;
                o_acc[60] *= _exp2_0;
                o_acc[61] *= _exp2_0;
                o_acc[62] *= _exp2_1;
                o_acc[63] *= _exp2_1;
                #endif
            }
            float _fma_0 = __fmaf_rn(s_acc[0], c_row0, 8.0f - _fmax_66);
            float _exp2_2 = approx_exp2(_fma_0);
            s_acc[0] = _exp2_2;
            float _fma_1 = __fmaf_rn(s_acc[1], c_row0, 8.0f - _fmax_66);
            float _exp2_3 = approx_exp2(_fma_1);
            s_acc[1] = _exp2_3;
            float _fma_2 = __fmaf_rn(s_acc[2], c_row1, 8.0f - _fmax_67);
            float _exp2_4 = approx_exp2(_fma_2);
            s_acc[2] = _exp2_4;
            float _fma_3 = __fmaf_rn(s_acc[3], c_row1, 8.0f - _fmax_67);
            float _exp2_5 = approx_exp2(_fma_3);
            s_acc[3] = _exp2_5;
            float _fma_4 = __fmaf_rn(s_acc[4], c_row0, 8.0f - _fmax_66);
            float _exp2_6 = approx_exp2(_fma_4);
            s_acc[4] = _exp2_6;
            float _fma_5 = __fmaf_rn(s_acc[5], c_row0, 8.0f - _fmax_66);
            float _exp2_7 = approx_exp2(_fma_5);
            s_acc[5] = _exp2_7;
            float _fma_6 = __fmaf_rn(s_acc[6], c_row1, 8.0f - _fmax_67);
            float _exp2_8 = approx_exp2(_fma_6);
            s_acc[6] = _exp2_8;
            float _fma_7 = __fmaf_rn(s_acc[7], c_row1, 8.0f - _fmax_67);
            float _exp2_9 = approx_exp2(_fma_7);
            s_acc[7] = _exp2_9;
            float _fma_8 = __fmaf_rn(s_acc[8], c_row0, 8.0f - _fmax_66);
            float _exp2_10 = approx_exp2(_fma_8);
            s_acc[8] = _exp2_10;
            float _fma_9 = __fmaf_rn(s_acc[9], c_row0, 8.0f - _fmax_66);
            float _exp2_11 = approx_exp2(_fma_9);
            s_acc[9] = _exp2_11;
            float _fma_10 = __fmaf_rn(s_acc[10], c_row1, 8.0f - _fmax_67);
            float _exp2_12 = approx_exp2(_fma_10);
            s_acc[10] = _exp2_12;
            float _fma_11 = __fmaf_rn(s_acc[11], c_row1, 8.0f - _fmax_67);
            float _exp2_13 = approx_exp2(_fma_11);
            s_acc[11] = _exp2_13;
            float _fma_12 = __fmaf_rn(s_acc[12], c_row0, 8.0f - _fmax_66);
            float _exp2_14 = approx_exp2(_fma_12);
            s_acc[12] = _exp2_14;
            float _fma_13 = __fmaf_rn(s_acc[13], c_row0, 8.0f - _fmax_66);
            float _exp2_15 = approx_exp2(_fma_13);
            s_acc[13] = _exp2_15;
            float _fma_14 = __fmaf_rn(s_acc[14], c_row1, 8.0f - _fmax_67);
            float _exp2_16 = approx_exp2(_fma_14);
            s_acc[14] = _exp2_16;
            float _fma_15 = __fmaf_rn(s_acc[15], c_row1, 8.0f - _fmax_67);
            float _exp2_17 = approx_exp2(_fma_15);
            s_acc[15] = _exp2_17;
            float _fma_16 = __fmaf_rn(s_acc[16], c_row0, 8.0f - _fmax_66);
            float _exp2_18 = approx_exp2(_fma_16);
            s_acc[16] = _exp2_18;
            float _fma_17 = __fmaf_rn(s_acc[17], c_row0, 8.0f - _fmax_66);
            float _exp2_19 = approx_exp2(_fma_17);
            s_acc[17] = _exp2_19;
            float _fma_18 = __fmaf_rn(s_acc[18], c_row1, 8.0f - _fmax_67);
            float _exp2_20 = approx_exp2(_fma_18);
            s_acc[18] = _exp2_20;
            float _fma_19 = __fmaf_rn(s_acc[19], c_row1, 8.0f - _fmax_67);
            float _exp2_21 = approx_exp2(_fma_19);
            s_acc[19] = _exp2_21;
            float _fma_20 = __fmaf_rn(s_acc[20], c_row0, 8.0f - _fmax_66);
            float _exp2_22 = approx_exp2(_fma_20);
            s_acc[20] = _exp2_22;
            float _fma_21 = __fmaf_rn(s_acc[21], c_row0, 8.0f - _fmax_66);
            float _exp2_23 = approx_exp2(_fma_21);
            s_acc[21] = _exp2_23;
            float _fma_22 = __fmaf_rn(s_acc[22], c_row1, 8.0f - _fmax_67);
            float _exp2_24 = approx_exp2(_fma_22);
            s_acc[22] = _exp2_24;
            float _fma_23 = __fmaf_rn(s_acc[23], c_row1, 8.0f - _fmax_67);
            float _exp2_25 = approx_exp2(_fma_23);
            s_acc[23] = _exp2_25;
            float _fma_24 = __fmaf_rn(s_acc[24], c_row0, 8.0f - _fmax_66);
            float _exp2_26 = approx_exp2(_fma_24);
            s_acc[24] = _exp2_26;
            float _fma_25 = __fmaf_rn(s_acc[25], c_row0, 8.0f - _fmax_66);
            float _exp2_27 = approx_exp2(_fma_25);
            s_acc[25] = _exp2_27;
            float _fma_26 = __fmaf_rn(s_acc[26], c_row1, 8.0f - _fmax_67);
            float _exp2_28 = approx_exp2(_fma_26);
            s_acc[26] = _exp2_28;
            float _fma_27 = __fmaf_rn(s_acc[27], c_row1, 8.0f - _fmax_67);
            float _exp2_29 = approx_exp2(_fma_27);
            s_acc[27] = _exp2_29;
            float _fma_28 = __fmaf_rn(s_acc[28], c_row0, 8.0f - _fmax_66);
            float _exp2_30 = approx_exp2(_fma_28);
            s_acc[28] = _exp2_30;
            float _fma_29 = __fmaf_rn(s_acc[29], c_row0, 8.0f - _fmax_66);
            float _exp2_31 = approx_exp2(_fma_29);
            s_acc[29] = _exp2_31;
            float _fma_30 = __fmaf_rn(s_acc[30], c_row1, 8.0f - _fmax_67);
            float _exp2_32 = approx_exp2(_fma_30);
            s_acc[30] = _exp2_32;
            float _fma_31 = __fmaf_rn(s_acc[31], c_row1, 8.0f - _fmax_67);
            float _exp2_33 = approx_exp2(_fma_31);
            s_acc[31] = _exp2_33;
            float _fma_32 = __fmaf_rn(s_acc[32], c_row0, 8.0f - _fmax_66);
            float _exp2_34 = approx_exp2(_fma_32);
            s_acc[32] = _exp2_34;
            float _fma_33 = __fmaf_rn(s_acc[33], c_row0, 8.0f - _fmax_66);
            float _exp2_35 = approx_exp2(_fma_33);
            s_acc[33] = _exp2_35;
            float _fma_34 = __fmaf_rn(s_acc[34], c_row1, 8.0f - _fmax_67);
            float _exp2_36 = approx_exp2(_fma_34);
            s_acc[34] = _exp2_36;
            float _fma_35 = __fmaf_rn(s_acc[35], c_row1, 8.0f - _fmax_67);
            float _exp2_37 = approx_exp2(_fma_35);
            s_acc[35] = _exp2_37;
            float _fma_36 = __fmaf_rn(s_acc[36], c_row0, 8.0f - _fmax_66);
            float _exp2_38 = approx_exp2(_fma_36);
            s_acc[36] = _exp2_38;
            float _fma_37 = __fmaf_rn(s_acc[37], c_row0, 8.0f - _fmax_66);
            float _exp2_39 = approx_exp2(_fma_37);
            s_acc[37] = _exp2_39;
            float _fma_38 = __fmaf_rn(s_acc[38], c_row1, 8.0f - _fmax_67);
            float _exp2_40 = approx_exp2(_fma_38);
            s_acc[38] = _exp2_40;
            float _fma_39 = __fmaf_rn(s_acc[39], c_row1, 8.0f - _fmax_67);
            float _exp2_41 = approx_exp2(_fma_39);
            s_acc[39] = _exp2_41;
            float _fma_40 = __fmaf_rn(s_acc[40], c_row0, 8.0f - _fmax_66);
            float _exp2_42 = approx_exp2(_fma_40);
            s_acc[40] = _exp2_42;
            float _fma_41 = __fmaf_rn(s_acc[41], c_row0, 8.0f - _fmax_66);
            float _exp2_43 = approx_exp2(_fma_41);
            s_acc[41] = _exp2_43;
            float _fma_42 = __fmaf_rn(s_acc[42], c_row1, 8.0f - _fmax_67);
            float _exp2_44 = approx_exp2(_fma_42);
            s_acc[42] = _exp2_44;
            float _fma_43 = __fmaf_rn(s_acc[43], c_row1, 8.0f - _fmax_67);
            float _exp2_45 = approx_exp2(_fma_43);
            s_acc[43] = _exp2_45;
            float _fma_44 = __fmaf_rn(s_acc[44], c_row0, 8.0f - _fmax_66);
            float _exp2_46 = approx_exp2(_fma_44);
            s_acc[44] = _exp2_46;
            float _fma_45 = __fmaf_rn(s_acc[45], c_row0, 8.0f - _fmax_66);
            float _exp2_47 = approx_exp2(_fma_45);
            s_acc[45] = _exp2_47;
            float _fma_46 = __fmaf_rn(s_acc[46], c_row1, 8.0f - _fmax_67);
            float _exp2_48 = approx_exp2(_fma_46);
            s_acc[46] = _exp2_48;
            float _fma_47 = __fmaf_rn(s_acc[47], c_row1, 8.0f - _fmax_67);
            float _exp2_49 = approx_exp2(_fma_47);
            s_acc[47] = _exp2_49;
            float _fma_48 = __fmaf_rn(s_acc[48], c_row0, 8.0f - _fmax_66);
            float _exp2_50 = approx_exp2(_fma_48);
            s_acc[48] = _exp2_50;
            float _fma_49 = __fmaf_rn(s_acc[49], c_row0, 8.0f - _fmax_66);
            float _exp2_51 = approx_exp2(_fma_49);
            s_acc[49] = _exp2_51;
            float _fma_50 = __fmaf_rn(s_acc[50], c_row1, 8.0f - _fmax_67);
            float _exp2_52 = approx_exp2(_fma_50);
            s_acc[50] = _exp2_52;
            float _fma_51 = __fmaf_rn(s_acc[51], c_row1, 8.0f - _fmax_67);
            float _exp2_53 = approx_exp2(_fma_51);
            s_acc[51] = _exp2_53;
            float _fma_52 = __fmaf_rn(s_acc[52], c_row0, 8.0f - _fmax_66);
            float _exp2_54 = approx_exp2(_fma_52);
            s_acc[52] = _exp2_54;
            float _fma_53 = __fmaf_rn(s_acc[53], c_row0, 8.0f - _fmax_66);
            float _exp2_55 = approx_exp2(_fma_53);
            s_acc[53] = _exp2_55;
            float _fma_54 = __fmaf_rn(s_acc[54], c_row1, 8.0f - _fmax_67);
            float _exp2_56 = approx_exp2(_fma_54);
            s_acc[54] = _exp2_56;
            float _fma_55 = __fmaf_rn(s_acc[55], c_row1, 8.0f - _fmax_67);
            float _exp2_57 = approx_exp2(_fma_55);
            s_acc[55] = _exp2_57;
            float _fma_56 = __fmaf_rn(s_acc[56], c_row0, 8.0f - _fmax_66);
            float _exp2_58 = approx_exp2(_fma_56);
            s_acc[56] = _exp2_58;
            float _fma_57 = __fmaf_rn(s_acc[57], c_row0, 8.0f - _fmax_66);
            float _exp2_59 = approx_exp2(_fma_57);
            s_acc[57] = _exp2_59;
            float _fma_58 = __fmaf_rn(s_acc[58], c_row1, 8.0f - _fmax_67);
            float _exp2_60 = approx_exp2(_fma_58);
            s_acc[58] = _exp2_60;
            float _fma_59 = __fmaf_rn(s_acc[59], c_row1, 8.0f - _fmax_67);
            float _exp2_61 = approx_exp2(_fma_59);
            s_acc[59] = _exp2_61;
            float _fma_60 = __fmaf_rn(s_acc[60], c_row0, 8.0f - _fmax_66);
            float _exp2_62 = approx_exp2(_fma_60);
            s_acc[60] = _exp2_62;
            float _fma_61 = __fmaf_rn(s_acc[61], c_row0, 8.0f - _fmax_66);
            float _exp2_63 = approx_exp2(_fma_61);
            s_acc[61] = _exp2_63;
            float _fma_62 = __fmaf_rn(s_acc[62], c_row1, 8.0f - _fmax_67);
            float _exp2_64 = approx_exp2(_fma_62);
            s_acc[62] = _exp2_64;
            float _fma_63 = __fmaf_rn(s_acc[63], c_row1, 8.0f - _fmax_67);
            float _exp2_65 = approx_exp2(_fma_63);
            s_acc[63] = _exp2_65;
            float _fma_64 = __fmaf_rn(l_state[0], _exp2_0, _exp2_2 + _exp2_3 + _exp2_6 + _exp2_7 + _exp2_10 + _exp2_11 + _exp2_14 + _exp2_15 + _exp2_18 + _exp2_19 + _exp2_22 + _exp2_23 + _exp2_26 + _exp2_27 + _exp2_30 + _exp2_31 + _exp2_34 + _exp2_35 + _exp2_38 + _exp2_39 + _exp2_42 + _exp2_43 + _exp2_46 + _exp2_47 + _exp2_50 + _exp2_51 + _exp2_54 + _exp2_55 + _exp2_58 + _exp2_59 + _exp2_62 + _exp2_63);
            l_state[0] = _fma_64;
            float _fma_65 = __fmaf_rn(l_state[1], _exp2_1, _exp2_4 + _exp2_5 + _exp2_8 + _exp2_9 + _exp2_12 + _exp2_13 + _exp2_16 + _exp2_17 + _exp2_20 + _exp2_21 + _exp2_24 + _exp2_25 + _exp2_28 + _exp2_29 + _exp2_32 + _exp2_33 + _exp2_36 + _exp2_37 + _exp2_40 + _exp2_41 + _exp2_44 + _exp2_45 + _exp2_48 + _exp2_49 + _exp2_52 + _exp2_53 + _exp2_56 + _exp2_57 + _exp2_60 + _exp2_61 + _exp2_64 + _exp2_65);
            l_state[1] = _fma_65;
            p_tmp[0] = s_acc[0];
            p_tmp[1] = s_acc[1];
            p_tmp[2] = s_acc[4];
            p_tmp[3] = s_acc[5];
            p_tmp[4] = s_acc[2];
            p_tmp[5] = s_acc[3];
            p_tmp[6] = s_acc[6];
            p_tmp[7] = s_acc[7];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[0] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[1] = _packed;
            }
            p_tmp[0] = s_acc[8];
            p_tmp[1] = s_acc[9];
            p_tmp[2] = s_acc[12];
            p_tmp[3] = s_acc[13];
            p_tmp[4] = s_acc[10];
            p_tmp[5] = s_acc[11];
            p_tmp[6] = s_acc[14];
            p_tmp[7] = s_acc[15];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[2] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[3] = _packed;
            }
            p_tmp[0] = s_acc[16];
            p_tmp[1] = s_acc[17];
            p_tmp[2] = s_acc[20];
            p_tmp[3] = s_acc[21];
            p_tmp[4] = s_acc[18];
            p_tmp[5] = s_acc[19];
            p_tmp[6] = s_acc[22];
            p_tmp[7] = s_acc[23];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[4] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[5] = _packed;
            }
            p_tmp[0] = s_acc[24];
            p_tmp[1] = s_acc[25];
            p_tmp[2] = s_acc[28];
            p_tmp[3] = s_acc[29];
            p_tmp[4] = s_acc[26];
            p_tmp[5] = s_acc[27];
            p_tmp[6] = s_acc[30];
            p_tmp[7] = s_acc[31];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[6] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[7] = _packed;
            }
            p_tmp[0] = s_acc[32];
            p_tmp[1] = s_acc[33];
            p_tmp[2] = s_acc[36];
            p_tmp[3] = s_acc[37];
            p_tmp[4] = s_acc[34];
            p_tmp[5] = s_acc[35];
            p_tmp[6] = s_acc[38];
            p_tmp[7] = s_acc[39];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[8] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[9] = _packed;
            }
            p_tmp[0] = s_acc[40];
            p_tmp[1] = s_acc[41];
            p_tmp[2] = s_acc[44];
            p_tmp[3] = s_acc[45];
            p_tmp[4] = s_acc[42];
            p_tmp[5] = s_acc[43];
            p_tmp[6] = s_acc[46];
            p_tmp[7] = s_acc[47];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[10] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[11] = _packed;
            }
            p_tmp[0] = s_acc[48];
            p_tmp[1] = s_acc[49];
            p_tmp[2] = s_acc[52];
            p_tmp[3] = s_acc[53];
            p_tmp[4] = s_acc[50];
            p_tmp[5] = s_acc[51];
            p_tmp[6] = s_acc[54];
            p_tmp[7] = s_acc[55];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[12] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[13] = _packed;
            }
            p_tmp[0] = s_acc[56];
            p_tmp[1] = s_acc[57];
            p_tmp[2] = s_acc[60];
            p_tmp[3] = s_acc[61];
            p_tmp[4] = s_acc[58];
            p_tmp[5] = s_acc[59];
            p_tmp[6] = s_acc[62];
            p_tmp[7] = s_acc[63];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[14] = _packed;
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
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[15] = _packed;
            }
            mbarrier_wait(v_full_addr + (kv_stage) * 8, kv_phase);
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            if (elect_sync()) {
                mbarrier_arrive(v_empty_addr + (kv_stage) * 8);
            }
            if (warp == 0) {
                mbarrier_wait(k_empty_addr + (kv_stage) * 8, kv_phase);
            }
            if (kb_last >= kb + 2) {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 9216);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(K_stage_addr + kv_stage * 8192), "l"((&K_map)), "r"(0), "r"((kb + 2) * 128), "r"(head),
                               "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(KSF_stage_addr + kv_stage * 1024), "l"((&KSF_map)), "r"(0), "r"((head * num_kblocks + (kb + 2)) * 64),
                               "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (warp == 4) {
                mbarrier_wait(v_empty_addr + (kv_stage) * 8, kv_phase);
            }
            if (kb_last >= kb + 2) {
                if (warp == 4) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(V_stage_addr + kv_stage * 16384), "l"((&V_map)), "r"((kb + 2) * 128), "r"(0), "r"(head),
                               "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            kv_stage += 1;
            if (kv_stage == 2) { kv_stage = 0; kv_phase ^= 1; }
        }
        float l0 = l_state[0];
        float l1 = l_state[1];
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, l0, 1);
        l0 = l0 + _shfl_xor_4;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, l0, 2);
        l0 = l0 + _shfl_xor_5;
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, l1, 1);
        l1 = l1 + _shfl_xor_6;
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, l1, 2);
        l1 = l1 + _shfl_xor_7;
        float _rcp_0 = approx_rcp(l0);
        float inv0 = _rcp_0;
        float _rcp_1 = approx_rcp(l1);
        float inv1 = _rcp_1;
        float _vec_load_3[2];
        {
            float2 _v2_4 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + (lane & 3) * 2) + 0);
            _vec_load_3[0] = _v2_4.x;
            _vec_load_3[0 + 1] = _v2_4.y;
        }
        o_tmp[0] = o_acc[0] * inv0 * _vec_load_3[0];
        o_tmp[1] = o_acc[1] * inv0 * _vec_load_3[1];
        o_tmp[2] = o_acc[2] * inv1 * _vec_load_3[0];
        o_tmp[3] = o_acc[3] * inv1 * _vec_load_3[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2)))[0]) = _pk;
            }
        }
        float _vec_load_4[2];
        {
            float2 _v2_5 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 8 + (lane & 3) * 2) + 0);
            _vec_load_4[0] = _v2_5.x;
            _vec_load_4[0 + 1] = _v2_5.y;
        }
        o_tmp[0] = o_acc[4] * inv0 * _vec_load_4[0];
        o_tmp[1] = o_acc[5] * inv0 * _vec_load_4[1];
        o_tmp[2] = o_acc[6] * inv1 * _vec_load_4[0];
        o_tmp[3] = o_acc[7] * inv1 * _vec_load_4[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 8)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 8)))[0]) = _pk;
            }
        }
        float _vec_load_5[2];
        {
            float2 _v2_6 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 16 + (lane & 3) * 2) + 0);
            _vec_load_5[0] = _v2_6.x;
            _vec_load_5[0 + 1] = _v2_6.y;
        }
        o_tmp[0] = o_acc[8] * inv0 * _vec_load_5[0];
        o_tmp[1] = o_acc[9] * inv0 * _vec_load_5[1];
        o_tmp[2] = o_acc[10] * inv1 * _vec_load_5[0];
        o_tmp[3] = o_acc[11] * inv1 * _vec_load_5[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 16)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 16)))[0]) = _pk;
            }
        }
        float _vec_load_6[2];
        {
            float2 _v2_7 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 24 + (lane & 3) * 2) + 0);
            _vec_load_6[0] = _v2_7.x;
            _vec_load_6[0 + 1] = _v2_7.y;
        }
        o_tmp[0] = o_acc[12] * inv0 * _vec_load_6[0];
        o_tmp[1] = o_acc[13] * inv0 * _vec_load_6[1];
        o_tmp[2] = o_acc[14] * inv1 * _vec_load_6[0];
        o_tmp[3] = o_acc[15] * inv1 * _vec_load_6[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 24)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 24)))[0]) = _pk;
            }
        }
        float _vec_load_7[2];
        {
            float2 _v2_8 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 32 + (lane & 3) * 2) + 0);
            _vec_load_7[0] = _v2_8.x;
            _vec_load_7[0 + 1] = _v2_8.y;
        }
        o_tmp[0] = o_acc[16] * inv0 * _vec_load_7[0];
        o_tmp[1] = o_acc[17] * inv0 * _vec_load_7[1];
        o_tmp[2] = o_acc[18] * inv1 * _vec_load_7[0];
        o_tmp[3] = o_acc[19] * inv1 * _vec_load_7[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 32)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 32)))[0]) = _pk;
            }
        }
        float _vec_load_8[2];
        {
            float2 _v2_9 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 40 + (lane & 3) * 2) + 0);
            _vec_load_8[0] = _v2_9.x;
            _vec_load_8[0 + 1] = _v2_9.y;
        }
        o_tmp[0] = o_acc[20] * inv0 * _vec_load_8[0];
        o_tmp[1] = o_acc[21] * inv0 * _vec_load_8[1];
        o_tmp[2] = o_acc[22] * inv1 * _vec_load_8[0];
        o_tmp[3] = o_acc[23] * inv1 * _vec_load_8[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 40)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 40)))[0]) = _pk;
            }
        }
        float _vec_load_9[2];
        {
            float2 _v2_10 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 48 + (lane & 3) * 2) + 0);
            _vec_load_9[0] = _v2_10.x;
            _vec_load_9[0 + 1] = _v2_10.y;
        }
        o_tmp[0] = o_acc[24] * inv0 * _vec_load_9[0];
        o_tmp[1] = o_acc[25] * inv0 * _vec_load_9[1];
        o_tmp[2] = o_acc[26] * inv1 * _vec_load_9[0];
        o_tmp[3] = o_acc[27] * inv1 * _vec_load_9[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 48)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 48)))[0]) = _pk;
            }
        }
        float _vec_load_10[2];
        {
            float2 _v2_11 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 56 + (lane & 3) * 2) + 0);
            _vec_load_10[0] = _v2_11.x;
            _vec_load_10[0 + 1] = _v2_11.y;
        }
        o_tmp[0] = o_acc[28] * inv0 * _vec_load_10[0];
        o_tmp[1] = o_acc[29] * inv0 * _vec_load_10[1];
        o_tmp[2] = o_acc[30] * inv1 * _vec_load_10[0];
        o_tmp[3] = o_acc[31] * inv1 * _vec_load_10[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 56)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 56)))[0]) = _pk;
            }
        }
        float _vec_load_11[2];
        {
            float2 _v2_12 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 64 + (lane & 3) * 2) + 0);
            _vec_load_11[0] = _v2_12.x;
            _vec_load_11[0 + 1] = _v2_12.y;
        }
        o_tmp[0] = o_acc[32] * inv0 * _vec_load_11[0];
        o_tmp[1] = o_acc[33] * inv0 * _vec_load_11[1];
        o_tmp[2] = o_acc[34] * inv1 * _vec_load_11[0];
        o_tmp[3] = o_acc[35] * inv1 * _vec_load_11[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 64)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 64)))[0]) = _pk;
            }
        }
        float _vec_load_12[2];
        {
            float2 _v2_13 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 72 + (lane & 3) * 2) + 0);
            _vec_load_12[0] = _v2_13.x;
            _vec_load_12[0 + 1] = _v2_13.y;
        }
        o_tmp[0] = o_acc[36] * inv0 * _vec_load_12[0];
        o_tmp[1] = o_acc[37] * inv0 * _vec_load_12[1];
        o_tmp[2] = o_acc[38] * inv1 * _vec_load_12[0];
        o_tmp[3] = o_acc[39] * inv1 * _vec_load_12[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 72)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 72)))[0]) = _pk;
            }
        }
        float _vec_load_13[2];
        {
            float2 _v2_14 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 80 + (lane & 3) * 2) + 0);
            _vec_load_13[0] = _v2_14.x;
            _vec_load_13[0 + 1] = _v2_14.y;
        }
        o_tmp[0] = o_acc[40] * inv0 * _vec_load_13[0];
        o_tmp[1] = o_acc[41] * inv0 * _vec_load_13[1];
        o_tmp[2] = o_acc[42] * inv1 * _vec_load_13[0];
        o_tmp[3] = o_acc[43] * inv1 * _vec_load_13[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 80)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 80)))[0]) = _pk;
            }
        }
        float _vec_load_14[2];
        {
            float2 _v2_15 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 88 + (lane & 3) * 2) + 0);
            _vec_load_14[0] = _v2_15.x;
            _vec_load_14[0 + 1] = _v2_15.y;
        }
        o_tmp[0] = o_acc[44] * inv0 * _vec_load_14[0];
        o_tmp[1] = o_acc[45] * inv0 * _vec_load_14[1];
        o_tmp[2] = o_acc[46] * inv1 * _vec_load_14[0];
        o_tmp[3] = o_acc[47] * inv1 * _vec_load_14[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 88)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 88)))[0]) = _pk;
            }
        }
        float _vec_load_15[2];
        {
            float2 _v2_16 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 96 + (lane & 3) * 2) + 0);
            _vec_load_15[0] = _v2_16.x;
            _vec_load_15[0 + 1] = _v2_16.y;
        }
        o_tmp[0] = o_acc[48] * inv0 * _vec_load_15[0];
        o_tmp[1] = o_acc[49] * inv0 * _vec_load_15[1];
        o_tmp[2] = o_acc[50] * inv1 * _vec_load_15[0];
        o_tmp[3] = o_acc[51] * inv1 * _vec_load_15[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 96)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 96)))[0]) = _pk;
            }
        }
        float _vec_load_16[2];
        {
            float2 _v2_17 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 104 + (lane & 3) * 2) + 0);
            _vec_load_16[0] = _v2_17.x;
            _vec_load_16[0 + 1] = _v2_17.y;
        }
        o_tmp[0] = o_acc[52] * inv0 * _vec_load_16[0];
        o_tmp[1] = o_acc[53] * inv0 * _vec_load_16[1];
        o_tmp[2] = o_acc[54] * inv1 * _vec_load_16[0];
        o_tmp[3] = o_acc[55] * inv1 * _vec_load_16[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 104)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 104)))[0]) = _pk;
            }
        }
        float _vec_load_17[2];
        {
            float2 _v2_18 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 112 + (lane & 3) * 2) + 0);
            _vec_load_17[0] = _v2_18.x;
            _vec_load_17[0 + 1] = _v2_18.y;
        }
        o_tmp[0] = o_acc[56] * inv0 * _vec_load_17[0];
        o_tmp[1] = o_acc[57] * inv0 * _vec_load_17[1];
        o_tmp[2] = o_acc[58] * inv1 * _vec_load_17[0];
        o_tmp[3] = o_acc[59] * inv1 * _vec_load_17[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 112)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 112)))[0]) = _pk;
            }
        }
        float _vec_load_18[2];
        {
            float2 _v2_19 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 120 + (lane & 3) * 2) + 0);
            _vec_load_18[0] = _v2_19.x;
            _vec_load_18[0 + 1] = _v2_19.y;
        }
        o_tmp[0] = o_acc[60] * inv0 * _vec_load_18[0];
        o_tmp[1] = o_acc[61] * inv0 * _vec_load_18[1];
        o_tmp[2] = o_acc[62] * inv1 * _vec_load_18[0];
        o_tmp[3] = o_acc[63] * inv1 * _vec_load_18[1];
        if (row0 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[0 + 0], o_tmp[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 120)))[0]) = _pk;
            }
        }
        if (row1 < seg_end_v) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(o_tmp[2 + 0], o_tmp[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 120)))[0]) = _pk;
            }
        }
        __syncthreads();
    }
    mbarrier_wait(q_full_addr, _phase_q_full_0);
    _phase_q_full_0 ^= 1;

    // Cleanup
    __syncthreads();
}

}  // namespace h3_varlen_attention_nvfp4_short_sm120a
#undef H3_VARLEN_INF
#undef NUM_KV_STAGES
#undef SMEM_KSF_STAGE_OFF
#undef SMEM_KSF_STAGE_STAGE_BYTES
#undef SMEM_KSF_STAGE_STRIDE
#undef SMEM_K_STAGE_OFF
#undef SMEM_K_STAGE_STAGE_BYTES
#undef SMEM_K_STAGE_STRIDE
#undef SMEM_Q_OFF
#undef SMEM_Q_STAGE_BYTES
#undef SMEM_Q_STRIDE
#undef SMEM_TOTAL
#undef SMEM_V_STAGE_OFF
#undef SMEM_V_STAGE_STAGE_BYTES
#undef SMEM_V_STAGE_STRIDE
#undef THREADS
#undef k_empty_addr
#undef k_full_addr
#undef q_full_addr
#undef v_empty_addr
#undef v_full_addr

#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>
#include <utility>
#include <vector>

#include "tvm_ffi_utils.h"

namespace {

// Every launch carries the programmatic-stream-serialization attribute: the kernels execute
// griddepcontrol.wait before touching a predecessor's outputs and griddepcontrol.launch_dependents
// once they have, so the next grid's launch latency overlaps the current grid's tail.
template <typename... KArgs, typename... Args>
void LaunchPdl(void (*kernel)(KArgs...), int grid, int block, size_t smem, cudaStream_t stream, Args&&... args) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid));
  config.blockDim = dim3(static_cast<unsigned>(block));
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  const cudaError_t status = cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "MiniMax-H3 SM120 NVFP4 varlen attention launch failed: " << cudaGetErrorString(status);
}

constexpr int64_t kHeadDim = 128;
constexpr int64_t kTileRowInts = 8;
constexpr int64_t kUnitRowInts = 4;
constexpr int64_t kAttnShortMaxTokens = 4096;
constexpr int64_t kBlockM = 128;
constexpr int64_t kBlockN = 128;
constexpr int64_t kMaxHeads = 32768;
constexpr int kStatsThreads = 256;
constexpr int kQuantThreads = 256;
constexpr int kAttentionThreads = 256;
constexpr int kStatsSmemBytes = 16512;
constexpr int kQuantSmemBytes = 0;
constexpr int64_t kQuantTokens = 32;      // tokens per quantizer CTA
constexpr int kAttentionSmemBytes = 60416;  // Q tile + two-stage K / K-scale / V^T ring + mbarriers
constexpr int64_t kRowBytes4 = kHeadDim / 2;          // packed E2M1 bytes per (token, head) row
constexpr int64_t kBlocksPerRow = kHeadDim / 16;      // UE4M3 block scales per row
constexpr int64_t kKsfTileBytes = kBlockN * kBlocksPerRow;  // 1 KiB K-scale tile per (head, key block)
constexpr uint32_t kKsfTileRows16 = static_cast<uint32_t>(kKsfTileBytes / 16);
// Statistics partials: per (Q tile, head) 128 K channel sums followed by 128 V channel maxima.
constexpr int64_t kPartialFloats = 2 * kHeadDim;

struct DeviceInfo {
  int num_sms;
};

DeviceInfo ConfigureKernels() {
  static std::mutex mutex;
  static std::vector<std::pair<int, DeviceInfo>> configured;
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to get the active CUDA device: " << cudaGetErrorString(status);
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto& entry : configured) {
    if (entry.first == device) return entry.second;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to query CUDA device properties: " << cudaGetErrorString(status);
  TVM_FFI_CHECK(properties.major == 12, RuntimeError)
      << "MiniMax-H3 SM120 NVFP4 varlen attention requires compute capability 12.x (GB202)";
  status = cudaFuncSetAttribute(h3_varlen_attention_nvfp4_sm120a::kernel_minimax_h3_sm120_varlen_attention_nvfp4,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, kAttentionSmemBytes);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to opt in to dynamic shared memory: " << cudaGetErrorString(status);
  status = cudaFuncSetAttribute(
      h3_varlen_attention_nvfp4_short_sm120a::kernel_minimax_h3_sm120_varlen_attention_nvfp4_short,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kAttentionSmemBytes);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to opt in to dynamic shared memory (short variant): " << cudaGetErrorString(status);
  DeviceInfo info{properties.multiProcessorCount};
  configured.emplace_back(device, info);
  return info;
}

void CheckDevice(const TensorView& tensor, const char* name, DLDevice device) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError) << name << " must be a CUDA tensor";
  TVM_FFI_CHECK(tensor.device().device_id == device.device_id, ValueError)
      << name << " must be on the same CUDA device as q";
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0, ValueError)
      << name << " must be 16-byte aligned";
}

void CheckThd(const TensorView& tensor, const char* name, int64_t tokens, int64_t heads, DLDevice device) {
  CheckDevice(tensor, name, device);
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dl_bfloat16), ValueError)
      << name << " must be bfloat16";
  TVM_FFI_CHECK(tensor.ndim() == 3 && tensor.size(0) == tokens && tensor.size(1) == heads &&
                    tensor.size(2) == kHeadDim,
                ValueError)
      << name << " must have shape [tokens, heads, 128]";
  TVM_FFI_CHECK(tensor.IsContiguous(), ValueError) << name << " must be contiguous";
}

void CheckFlat(const TensorView& tensor, const char* name, DLDataType dtype, const char* dtype_name,
               int64_t min_numel, DLDevice device) {
  CheckDevice(tensor, name, device);
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dtype), ValueError)
      << name << " must be " << dtype_name;
  TVM_FFI_CHECK(tensor.ndim() == 1 && tensor.IsContiguous() && tensor.size(0) >= min_numel, ValueError)
      << name << " must be a contiguous 1-D " << dtype_name << " tensor with at least " << min_numel
      << " elements";
}

// One head's [64 packed-E2M1 bytes x box_rows tokens] tile of a [tokens, heads, 64] byte tensor,
// addressed as the 3-D view (64 bytes, tokens, heads) with a 64-byte swizzle.  The token box may run
// past the tensor: TMA zero-fills those rows, the kernel masks those keys / never stores those query
// rows.
CUtensorMap EncodeRowsTile(const TensorView& rows, int64_t tokens, int64_t heads, uint32_t box_rows,
                           const char* name) {
  uint64_t global_dim[3] = {static_cast<uint64_t>(kRowBytes4), static_cast<uint64_t>(tokens),
                            static_cast<uint64_t>(heads)};
  uint64_t global_strides[2] = {static_cast<uint64_t>(heads * kRowBytes4), static_cast<uint64_t>(kRowBytes4)};
  uint32_t box_dim[3] = {static_cast<uint32_t>(kRowBytes4), box_rows, 1};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, rows.data_ptr(), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode the " << name << " tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

// The K block-scale tiles: one contiguous 1 KiB tile (128 keys x 8 UE4M3) per (head, key block),
// addressed as the 2-D view (16 bytes, heads * num_kblocks * 64 rows) without swizzle.
CUtensorMap EncodeScaleTiles(const TensorView& k_sf, int64_t heads, int64_t num_kblocks, const char* name) {
  uint64_t global_dim[2] = {16, static_cast<uint64_t>(heads * num_kblocks * kKsfTileRows16)};
  uint64_t global_strides[1] = {16};
  uint32_t box_dim[2] = {16, kKsfTileRows16};
  uint32_t element_strides[2] = {1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, k_sf.data_ptr(), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode the " << name << " tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

// One head's [128 channel rows x 128 keys] tile of the transposed V^T byte tensor [heads, 128,
// padded_tokens]: 3-D view (padded tokens, channels, heads), 128-byte swizzle.
CUtensorMap EncodeTransposedTile(const TensorView& vt, int64_t padded_tokens, int64_t heads, const char* name) {
  uint64_t global_dim[3] = {static_cast<uint64_t>(padded_tokens), static_cast<uint64_t>(kHeadDim),
                            static_cast<uint64_t>(heads)};
  uint64_t global_strides[2] = {static_cast<uint64_t>(padded_tokens),
                                static_cast<uint64_t>(padded_tokens * kHeadDim)};
  uint32_t box_dim[3] = {static_cast<uint32_t>(kBlockN), static_cast<uint32_t>(kHeadDim), 1};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, vt.data_ptr(), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode the " << name << " tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

}  // namespace

// out[a:b, h] = softmax(q[a:b, h] k[a:b, h]^T * softmax_scale) v[a:b, h] for every segment [a, b) of
// cu_seqlens and every head h, with NVFP4 (E2M1 + UE4M3 block scales) Q / K, E4M3 P / V operands and
// an FP32 softmax.
//   q, k, v, out: contiguous BF16 [tokens, heads, 128].
//   Plan (built on the host from cu_seqlens): tok_seg int32 [padded_tokens] (segment of every token, padded
//   tokens repeat the last), tile_table int32 [8 * num_tiles] (segment, row_begin, row_end, first tile of
//   the segment, end tile, segment length, 0, 0), unit_table int32 [4 * num_units] (segment,
//   head << 16 | q_tile, segment begin, segment end) in persistent-grid slot order.
//   Workspaces: q4, k4 uint8 [>= tokens * heads * 64], q_sf uint8 [>= tokens * heads * 8], k_sf uint8
//   [>= heads * num_kblocks * 1024], vt8 uint8 [>= heads * 128 * padded_tokens], q_scale f32
//   [>= tokens * heads], k_scale f32 [>= segments * heads], v_scale and mean_k f32
//   [>= segments * heads * 128], partials f32 [>= num_tiles * heads * 256], knorm_part f32 [>= num_tiles * heads],
//   counters uint32 [>= segments * heads] (zero before the first call; the kernel resets them).
void minimax_h3_sm120_varlen_attention_nvfp4(TensorView q, TensorView k, TensorView v, TensorView tok_seg,
                                             TensorView out, TensorView tile_table, TensorView unit_table,
                                             TensorView q4, TensorView k4, TensorView q_sf, TensorView k_sf,
                                             TensorView vt8, TensorView q_scale, TensorView k_scale,
                                             TensorView v_scale, TensorView mean_k, TensorView partials,
                                             TensorView knorm_part, TensorView counters, int64_t num_segments, int64_t num_tiles, int64_t num_units,
                                             int64_t attention_grid, int64_t longest_segment, double softmax_scale) {
  TVM_FFI_CHECK(q.ndim() == 3, ValueError) << "q must be [tokens, heads, 128]";
  const int64_t tokens = q.size(0);
  const int64_t heads = q.size(1);
  TVM_FFI_CHECK(tokens >= 0 && heads >= 1 && heads < kMaxHeads, ValueError)
      << "q must be [tokens, heads, 128] with 1 <= heads < " << kMaxHeads;
  const DLDevice device = q.device();
  CheckThd(q, "q", tokens, heads, device);
  CheckThd(k, "k", tokens, heads, device);
  CheckThd(v, "v", tokens, heads, device);
  CheckThd(out, "out", tokens, heads, device);
  TVM_FFI_CHECK(num_segments >= 1 && num_tiles >= 0 && num_units >= 0 && attention_grid >= 1 && longest_segment >= 0,
                ValueError)
      << "invalid segment plan";
  const int64_t num_kblocks = std::max<int64_t>(1, (tokens + kBlockN - 1) / kBlockN);
  const int64_t padded_tokens = num_kblocks * kBlockN;
  CheckFlat(tok_seg, "tok_seg", dl_int32, "int32", std::max<int64_t>(1, padded_tokens), device);
  CheckFlat(tile_table, "tile_table", dl_int32, "int32", std::max<int64_t>(kTileRowInts, kTileRowInts * num_tiles), device);
  CheckFlat(unit_table, "unit_table", dl_int32, "int32", std::max<int64_t>(kUnitRowInts, kUnitRowInts * num_units), device);
  CheckFlat(q4, "q4", dl_uint8, "uint8", tokens * heads * kRowBytes4, device);
  CheckFlat(k4, "k4", dl_uint8, "uint8", tokens * heads * kRowBytes4, device);
  CheckFlat(q_sf, "q_sf", dl_uint8, "uint8", tokens * heads * kBlocksPerRow, device);
  CheckFlat(k_sf, "k_sf", dl_uint8, "uint8", heads * num_kblocks * kKsfTileBytes, device);
  CheckFlat(vt8, "vt8", dl_uint8, "uint8", heads * kHeadDim * padded_tokens, device);
  CheckFlat(q_scale, "q_scale", dl_float32, "float32", tokens * heads, device);
  CheckFlat(k_scale, "k_scale", dl_float32, "float32", std::max<int64_t>(1, num_segments) * heads, device);
  CheckFlat(v_scale, "v_scale", dl_float32, "float32", num_segments * heads * kHeadDim, device);
  CheckFlat(mean_k, "mean_k", dl_float32, "float32", num_segments * heads * kHeadDim, device);
  CheckFlat(partials, "partials", dl_float32, "float32", std::max<int64_t>(1, num_tiles * heads * kPartialFloats), device);
  CheckFlat(knorm_part, "knorm_part", dl_float32, "float32", std::max<int64_t>(1, num_tiles * heads), device);
  CheckFlat(counters, "counters", dl_uint32, "uint32", std::max<int64_t>(1, num_segments) * heads, device);

  ffi::CUDADeviceGuard device_guard(device.device_id);
  const cudaStream_t stream = get_stream(device);
  const DeviceInfo info = ConfigureKernels();
  if (tokens == 0 || num_units == 0) return;  // no rows to write (all segments empty)
  const int cta_cap = 4 * info.num_sms;
  const int heads_i = static_cast<int>(heads);

  if (num_tiles > 0) {
    const int stats_grid = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(num_tiles * heads, cta_cap)));
    LaunchPdl(&h3_varlen_kv_stats_nvfp4_sm120a::kernel_minimax_h3_sm120_varlen_kv_stats_nvfp4, stats_grid, kStatsThreads,
        kStatsSmemBytes, stream,
        static_cast<__nv_bfloat16*>(k.data_ptr()), static_cast<__nv_bfloat16*>(v.data_ptr()),
        static_cast<int*>(tile_table.data_ptr()), static_cast<float*>(partials.data_ptr()), static_cast<float*>(knorm_part.data_ptr()),
        static_cast<unsigned int*>(counters.data_ptr()), static_cast<float*>(mean_k.data_ptr()),
        static_cast<float*>(v_scale.data_ptr()), static_cast<float*>(k_scale.data_ptr()), static_cast<int>(num_tiles),
        heads_i);
  }
  const int quant_grid = static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>((padded_tokens / kQuantTokens) * heads, 8 * static_cast<int64_t>(info.num_sms))));
  LaunchPdl(&h3_varlen_quantize_nvfp4_sm120a::kernel_minimax_h3_sm120_varlen_quantize_nvfp4, quant_grid, kQuantThreads,
      kQuantSmemBytes, stream,
      static_cast<__nv_bfloat16*>(q.data_ptr()), static_cast<__nv_bfloat16*>(k.data_ptr()),
      static_cast<__nv_bfloat16*>(v.data_ptr()), static_cast<int*>(tok_seg.data_ptr()),
      static_cast<float*>(mean_k.data_ptr()), static_cast<float*>(v_scale.data_ptr()),
      static_cast<unsigned int*>(q4.data_ptr()), static_cast<unsigned int*>(k4.data_ptr()),
      static_cast<uint8_t*>(q_sf.data_ptr()), static_cast<uint8_t*>(k_sf.data_ptr()),
      static_cast<unsigned int*>(vt8.data_ptr()), static_cast<float*>(q_scale.data_ptr()),
      static_cast<float*>(k_scale.data_ptr()), static_cast<int>(tokens), static_cast<int>(padded_tokens), heads_i,
      static_cast<int>(num_kblocks));

  const CUtensorMap q_map = EncodeRowsTile(q4, tokens, heads, static_cast<uint32_t>(kBlockM), "q4");
  const CUtensorMap k_map = EncodeRowsTile(k4, tokens, heads, static_cast<uint32_t>(kBlockN), "k4");
  const CUtensorMap ksf_map = EncodeScaleTiles(k_sf, heads, num_kblocks, "k_sf");
  const CUtensorMap v_map = EncodeTransposedTile(vt8, padded_tokens, heads, "vt8");
  const int grid = static_cast<int>(std::min<int64_t>(attention_grid, std::max<int64_t>(1, num_units)));
  const float softmax_scale_log2 = static_cast<float>(softmax_scale * 1.4426950408889634);
  // Plans whose longest segment fits kAttnShortMaxTokens run the variant without the lazy O rescale.
  auto* attention = longest_segment <= kAttnShortMaxTokens
                        ? &h3_varlen_attention_nvfp4_short_sm120a::kernel_minimax_h3_sm120_varlen_attention_nvfp4_short
                        : &h3_varlen_attention_nvfp4_sm120a::kernel_minimax_h3_sm120_varlen_attention_nvfp4;
  LaunchPdl(attention, grid, kAttentionThreads,
      kAttentionSmemBytes, stream,
      q_map, k_map, ksf_map, v_map, static_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<float*>(q_scale.data_ptr()), static_cast<float*>(k_scale.data_ptr()),
      static_cast<float*>(v_scale.data_ptr()), static_cast<unsigned int*>(q_sf.data_ptr()),
      static_cast<int*>(unit_table.data_ptr()), static_cast<int>(num_units), heads_i, static_cast<int>(num_kblocks),
      softmax_scale_log2);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_sm120_varlen_attention_nvfp4, minimax_h3_sm120_varlen_attention_nvfp4);
// clang-format on
