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
// MiniMax-H3 experimental NVFP4 non-causal packed-varlen self-attention for SM120 (GB202: RTX 5090 /
// RTX PRO 6000 Blackwell) following the SageAttention3 FP4 recipe, generated from the Cake kernel schedules.
// One operator call = three PDL-chained launches over packed THD tensors q, k, v [T, H, 128] BF16 (segments
// from int32 cu_seqlens):
//   1. stats:      per (segment, head) K channel mean (fixed-order fold over the segment's tiles, device-scope
//                  arrival counter, self-resetting) and per 128-row Q block channel mean (q_mean rows),
//   2. quantize:   K rows centred by the segment mean and Q rows by their block mean -> E2M1 codes with one
//                  UE4M3 scale per 16 channels (block amax / 6, round-to-nearest E4M3, x * (1 / decoded));
//                  the quantized block-mean row per (Q block, head); V transposed into [H, 128, T_pad / 2]
//                  E2M1 tiles with one UE4M3 scale per (channel, 16 keys) in the FA3 key permutation,
//   3. attention:  persistent 256-thread CTA per SM, 8 warps x 16 query rows in two ping-pong warp groups,
//                  128-key K / K-scale / V^T / V-scale tiles through a two-stage TMA ring; QK^T and PV both run
//                  mma.sync m16n8k64 kind::mxf4nvf4.block_scale.scale_vec::4X; the block-mean compensation
//                  row (qm K^T) is one extra block-scaled MMA row per tile, shared by the CTA and seeded into
//                  the score accumulator; FP32 online softmax in the log2 domain with the row maximum brought
//                  to 448 * 6, P quantized per 16 keys (UE4M3 block scale, E2M1 codes); BF16 output rows.
// Device code: TMA, ldmatrix, block-scaled mma.sync, mbarrier pipelines, named barriers.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

static_assert(sizeof(CUtensorMap) == 128, "CUDA tensor-map ABI size mismatch");
static_assert(alignof(CUtensorMap) >= 64, "CUDA tensor-map ABI requires at least 64-byte alignment");

namespace h3_varlen_stats_sage_sm120a {

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
#define SMEM_RED_Q_OFF 8192
#define SMEM_RED_Q_STAGE_BYTES 8192
#define SMEM_RED_Q_STRIDE 8192
#define SMEM_FLAG_OFF 16384
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_TOTAL 16512
#define THREADS 256

#include <math_constants.h>


__global__ __launch_bounds__(256, 2) void
kernel_minimax_h3_sm120_varlen_stats_sage(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, int* __restrict__ tile_table, int* __restrict__ seg_tile_begin, float* __restrict__ partials, unsigned int* __restrict__ counters, float* __restrict__ mean_k, float* __restrict__ q_mean, int num_tiles, int num_heads)
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
    float* red_q = reinterpret_cast<float*>(smem_raw + 8192);
    const int red_q_addr = smem + 8192;
    int* flag = reinterpret_cast<int*>(smem_raw + 16384);
    const int flag_addr = smem + 16384;

    // === Task calls (dependency order) ===
    int col = (tid & 15) * 8;
    int rgrp = tid >> 4;
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
        int qt_base = seg_tile_begin[seg] + (tile - t0);
        float ksum[8];
        for (int e = 0; e < 8; e++) {
            ksum[e] = 0.0f;
        }
        float qsum[8];
        float qf8[8];
        float kf8[8];
        #pragma unroll 1
        for (int batch = 0; batch < (row_end - row_begin + 128 - 1) / 128; batch++) {
            for (int e_1 = 0; e_1 < 8; e_1++) {
                qsum[e_1] = 0.0f;
            }
            int brow = row_begin + batch * 128;
            float _vec_load_2[8];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + rgrp) ? brow + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + rgrp) ? brow + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 16 + rgrp) ? brow + 16 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + 16 + rgrp) ? brow + 16 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 32 + rgrp) ? brow + 32 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + 32 + rgrp) ? brow + 32 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 48 + rgrp) ? brow + 48 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + 48 + rgrp) ? brow + 48 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 64 + rgrp) ? brow + 64 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_11 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + 64 + rgrp) ? brow + 64 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_12 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 80 + rgrp) ? brow + 80 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_13 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + 80 + rgrp) ? brow + 80 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_14 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 96 + rgrp) ? brow + 96 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_15 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + 96 + rgrp) ? brow + 96 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_16 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 112 + rgrp) ? brow + 112 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
                const uint4* _vptr_17 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + 112 + rgrp) ? brow + 112 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
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
            qf8[0] = _vec_load_2[0];
            kf8[0] = _vec_load_3[0];
            qf8[1] = _vec_load_2[1];
            kf8[1] = _vec_load_3[1];
            qf8[2] = _vec_load_2[2];
            kf8[2] = _vec_load_3[2];
            qf8[3] = _vec_load_2[3];
            kf8[3] = _vec_load_3[3];
            qf8[4] = _vec_load_2[4];
            kf8[4] = _vec_load_3[4];
            qf8[5] = _vec_load_2[5];
            kf8[5] = _vec_load_3[5];
            qf8[6] = _vec_load_2[6];
            kf8[6] = _vec_load_3[6];
            qf8[7] = _vec_load_2[7];
            kf8[7] = _vec_load_3[7];
            ksum[0] = ksum[0] + ((row_end > brow + rgrp) ? kf8[0] : 0.0f);
            qsum[0] = qsum[0] + ((row_end > brow + rgrp) ? qf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end > brow + rgrp) ? kf8[1] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + rgrp) ? qf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end > brow + rgrp) ? kf8[2] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + rgrp) ? qf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end > brow + rgrp) ? kf8[3] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + rgrp) ? qf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end > brow + rgrp) ? kf8[4] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + rgrp) ? qf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end > brow + rgrp) ? kf8[5] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + rgrp) ? qf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end > brow + rgrp) ? kf8[6] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + rgrp) ? qf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end > brow + rgrp) ? kf8[7] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + rgrp) ? qf8[7] : 0.0f);
            qf8[0] = _vec_load_4[0];
            kf8[0] = _vec_load_5[0];
            qf8[1] = _vec_load_4[1];
            kf8[1] = _vec_load_5[1];
            qf8[2] = _vec_load_4[2];
            kf8[2] = _vec_load_5[2];
            qf8[3] = _vec_load_4[3];
            kf8[3] = _vec_load_5[3];
            qf8[4] = _vec_load_4[4];
            kf8[4] = _vec_load_5[4];
            qf8[5] = _vec_load_4[5];
            kf8[5] = _vec_load_5[5];
            qf8[6] = _vec_load_4[6];
            kf8[6] = _vec_load_5[6];
            qf8[7] = _vec_load_4[7];
            kf8[7] = _vec_load_5[7];
            ksum[0] = ksum[0] + ((row_end > brow + 16 + rgrp) ? kf8[0] : 0.0f);
            qsum[0] = qsum[0] + ((row_end > brow + 16 + rgrp) ? qf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end > brow + 16 + rgrp) ? kf8[1] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + 16 + rgrp) ? qf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end > brow + 16 + rgrp) ? kf8[2] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + 16 + rgrp) ? qf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end > brow + 16 + rgrp) ? kf8[3] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + 16 + rgrp) ? qf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end > brow + 16 + rgrp) ? kf8[4] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + 16 + rgrp) ? qf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end > brow + 16 + rgrp) ? kf8[5] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + 16 + rgrp) ? qf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end > brow + 16 + rgrp) ? kf8[6] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + 16 + rgrp) ? qf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end > brow + 16 + rgrp) ? kf8[7] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + 16 + rgrp) ? qf8[7] : 0.0f);
            qf8[0] = _vec_load_6[0];
            kf8[0] = _vec_load_7[0];
            qf8[1] = _vec_load_6[1];
            kf8[1] = _vec_load_7[1];
            qf8[2] = _vec_load_6[2];
            kf8[2] = _vec_load_7[2];
            qf8[3] = _vec_load_6[3];
            kf8[3] = _vec_load_7[3];
            qf8[4] = _vec_load_6[4];
            kf8[4] = _vec_load_7[4];
            qf8[5] = _vec_load_6[5];
            kf8[5] = _vec_load_7[5];
            qf8[6] = _vec_load_6[6];
            kf8[6] = _vec_load_7[6];
            qf8[7] = _vec_load_6[7];
            kf8[7] = _vec_load_7[7];
            ksum[0] = ksum[0] + ((row_end > brow + 32 + rgrp) ? kf8[0] : 0.0f);
            qsum[0] = qsum[0] + ((row_end > brow + 32 + rgrp) ? qf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end > brow + 32 + rgrp) ? kf8[1] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + 32 + rgrp) ? qf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end > brow + 32 + rgrp) ? kf8[2] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + 32 + rgrp) ? qf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end > brow + 32 + rgrp) ? kf8[3] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + 32 + rgrp) ? qf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end > brow + 32 + rgrp) ? kf8[4] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + 32 + rgrp) ? qf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end > brow + 32 + rgrp) ? kf8[5] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + 32 + rgrp) ? qf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end > brow + 32 + rgrp) ? kf8[6] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + 32 + rgrp) ? qf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end > brow + 32 + rgrp) ? kf8[7] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + 32 + rgrp) ? qf8[7] : 0.0f);
            qf8[0] = _vec_load_8[0];
            kf8[0] = _vec_load_9[0];
            qf8[1] = _vec_load_8[1];
            kf8[1] = _vec_load_9[1];
            qf8[2] = _vec_load_8[2];
            kf8[2] = _vec_load_9[2];
            qf8[3] = _vec_load_8[3];
            kf8[3] = _vec_load_9[3];
            qf8[4] = _vec_load_8[4];
            kf8[4] = _vec_load_9[4];
            qf8[5] = _vec_load_8[5];
            kf8[5] = _vec_load_9[5];
            qf8[6] = _vec_load_8[6];
            kf8[6] = _vec_load_9[6];
            qf8[7] = _vec_load_8[7];
            kf8[7] = _vec_load_9[7];
            ksum[0] = ksum[0] + ((row_end > brow + 48 + rgrp) ? kf8[0] : 0.0f);
            qsum[0] = qsum[0] + ((row_end > brow + 48 + rgrp) ? qf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end > brow + 48 + rgrp) ? kf8[1] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + 48 + rgrp) ? qf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end > brow + 48 + rgrp) ? kf8[2] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + 48 + rgrp) ? qf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end > brow + 48 + rgrp) ? kf8[3] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + 48 + rgrp) ? qf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end > brow + 48 + rgrp) ? kf8[4] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + 48 + rgrp) ? qf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end > brow + 48 + rgrp) ? kf8[5] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + 48 + rgrp) ? qf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end > brow + 48 + rgrp) ? kf8[6] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + 48 + rgrp) ? qf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end > brow + 48 + rgrp) ? kf8[7] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + 48 + rgrp) ? qf8[7] : 0.0f);
            qf8[0] = _vec_load_10[0];
            kf8[0] = _vec_load_11[0];
            qf8[1] = _vec_load_10[1];
            kf8[1] = _vec_load_11[1];
            qf8[2] = _vec_load_10[2];
            kf8[2] = _vec_load_11[2];
            qf8[3] = _vec_load_10[3];
            kf8[3] = _vec_load_11[3];
            qf8[4] = _vec_load_10[4];
            kf8[4] = _vec_load_11[4];
            qf8[5] = _vec_load_10[5];
            kf8[5] = _vec_load_11[5];
            qf8[6] = _vec_load_10[6];
            kf8[6] = _vec_load_11[6];
            qf8[7] = _vec_load_10[7];
            kf8[7] = _vec_load_11[7];
            ksum[0] = ksum[0] + ((row_end > brow + 64 + rgrp) ? kf8[0] : 0.0f);
            qsum[0] = qsum[0] + ((row_end > brow + 64 + rgrp) ? qf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end > brow + 64 + rgrp) ? kf8[1] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + 64 + rgrp) ? qf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end > brow + 64 + rgrp) ? kf8[2] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + 64 + rgrp) ? qf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end > brow + 64 + rgrp) ? kf8[3] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + 64 + rgrp) ? qf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end > brow + 64 + rgrp) ? kf8[4] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + 64 + rgrp) ? qf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end > brow + 64 + rgrp) ? kf8[5] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + 64 + rgrp) ? qf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end > brow + 64 + rgrp) ? kf8[6] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + 64 + rgrp) ? qf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end > brow + 64 + rgrp) ? kf8[7] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + 64 + rgrp) ? qf8[7] : 0.0f);
            qf8[0] = _vec_load_12[0];
            kf8[0] = _vec_load_13[0];
            qf8[1] = _vec_load_12[1];
            kf8[1] = _vec_load_13[1];
            qf8[2] = _vec_load_12[2];
            kf8[2] = _vec_load_13[2];
            qf8[3] = _vec_load_12[3];
            kf8[3] = _vec_load_13[3];
            qf8[4] = _vec_load_12[4];
            kf8[4] = _vec_load_13[4];
            qf8[5] = _vec_load_12[5];
            kf8[5] = _vec_load_13[5];
            qf8[6] = _vec_load_12[6];
            kf8[6] = _vec_load_13[6];
            qf8[7] = _vec_load_12[7];
            kf8[7] = _vec_load_13[7];
            ksum[0] = ksum[0] + ((row_end > brow + 80 + rgrp) ? kf8[0] : 0.0f);
            qsum[0] = qsum[0] + ((row_end > brow + 80 + rgrp) ? qf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end > brow + 80 + rgrp) ? kf8[1] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + 80 + rgrp) ? qf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end > brow + 80 + rgrp) ? kf8[2] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + 80 + rgrp) ? qf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end > brow + 80 + rgrp) ? kf8[3] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + 80 + rgrp) ? qf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end > brow + 80 + rgrp) ? kf8[4] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + 80 + rgrp) ? qf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end > brow + 80 + rgrp) ? kf8[5] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + 80 + rgrp) ? qf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end > brow + 80 + rgrp) ? kf8[6] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + 80 + rgrp) ? qf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end > brow + 80 + rgrp) ? kf8[7] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + 80 + rgrp) ? qf8[7] : 0.0f);
            qf8[0] = _vec_load_14[0];
            kf8[0] = _vec_load_15[0];
            qf8[1] = _vec_load_14[1];
            kf8[1] = _vec_load_15[1];
            qf8[2] = _vec_load_14[2];
            kf8[2] = _vec_load_15[2];
            qf8[3] = _vec_load_14[3];
            kf8[3] = _vec_load_15[3];
            qf8[4] = _vec_load_14[4];
            kf8[4] = _vec_load_15[4];
            qf8[5] = _vec_load_14[5];
            kf8[5] = _vec_load_15[5];
            qf8[6] = _vec_load_14[6];
            kf8[6] = _vec_load_15[6];
            qf8[7] = _vec_load_14[7];
            kf8[7] = _vec_load_15[7];
            ksum[0] = ksum[0] + ((row_end > brow + 96 + rgrp) ? kf8[0] : 0.0f);
            qsum[0] = qsum[0] + ((row_end > brow + 96 + rgrp) ? qf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end > brow + 96 + rgrp) ? kf8[1] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + 96 + rgrp) ? qf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end > brow + 96 + rgrp) ? kf8[2] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + 96 + rgrp) ? qf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end > brow + 96 + rgrp) ? kf8[3] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + 96 + rgrp) ? qf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end > brow + 96 + rgrp) ? kf8[4] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + 96 + rgrp) ? qf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end > brow + 96 + rgrp) ? kf8[5] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + 96 + rgrp) ? qf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end > brow + 96 + rgrp) ? kf8[6] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + 96 + rgrp) ? qf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end > brow + 96 + rgrp) ? kf8[7] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + 96 + rgrp) ? qf8[7] : 0.0f);
            qf8[0] = _vec_load_16[0];
            kf8[0] = _vec_load_17[0];
            qf8[1] = _vec_load_16[1];
            kf8[1] = _vec_load_17[1];
            qf8[2] = _vec_load_16[2];
            kf8[2] = _vec_load_17[2];
            qf8[3] = _vec_load_16[3];
            kf8[3] = _vec_load_17[3];
            qf8[4] = _vec_load_16[4];
            kf8[4] = _vec_load_17[4];
            qf8[5] = _vec_load_16[5];
            kf8[5] = _vec_load_17[5];
            qf8[6] = _vec_load_16[6];
            kf8[6] = _vec_load_17[6];
            qf8[7] = _vec_load_16[7];
            kf8[7] = _vec_load_17[7];
            ksum[0] = ksum[0] + ((row_end > brow + 112 + rgrp) ? kf8[0] : 0.0f);
            qsum[0] = qsum[0] + ((row_end > brow + 112 + rgrp) ? qf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end > brow + 112 + rgrp) ? kf8[1] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + 112 + rgrp) ? qf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end > brow + 112 + rgrp) ? kf8[2] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + 112 + rgrp) ? qf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end > brow + 112 + rgrp) ? kf8[3] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + 112 + rgrp) ? qf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end > brow + 112 + rgrp) ? kf8[4] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + 112 + rgrp) ? qf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end > brow + 112 + rgrp) ? kf8[5] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + 112 + rgrp) ? qf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end > brow + 112 + rgrp) ? kf8[6] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + 112 + rgrp) ? qf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end > brow + 112 + rgrp) ? kf8[7] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + 112 + rgrp) ? qf8[7] : 0.0f);
            for (int e_2 = 0; e_2 < 8; e_2++) {
                red_q[rgrp * 128 + col + e_2] = qsum[e_2];
            }
            __syncthreads();
            if (tid < 128) {
                float qtot = 0.0f;
                for (int g = 0; g < 16; g++) {
                    qtot = qtot + red_q[g * 128 + tid];
                }
                int brows = ((row_end - brow < 128) ? row_end - brow : 128);
                float _fdiv_rn_0 = __fdiv_rn(qtot, (float)brows);
                *(reinterpret_cast<float*>(q_mean + (((qt_base + batch) * num_heads + head) * 128 + tid)) + (0)) = _fdiv_rn_0;
            }
            __syncthreads();
        }
        for (int e_3 = 0; e_3 < 8; e_3++) {
            red_sum[rgrp * 128 + col + e_3] = ksum[e_3];
        }
        __syncthreads();
        int direct = ((t1 - t0 == 1) ? 1 : 0);
        if (direct == 0) {
            if (tid < 128) {
                float total = 0.0f;
                for (int g_1 = 0; g_1 < 16; g_1++) {
                    total = total + red_sum[g_1 * 128 + tid];
                }
                *(reinterpret_cast<float*>(partials + (work * 128 + tid)) + (0)) = total;
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
            if (tid < 128) {
                float acc_a[1];
                float fold_v[4];
                acc_a[0] = 0.0f;
                if (direct == 1) {
                    for (int g_2 = 0; g_2 < 16; g_2++) {
                        acc_a[0] = acc_a[0] + red_sum[g_2 * 128 + tid];
                    }
                } else {
                    #pragma unroll 1
                    for (int tt = t0; tt < t1; tt += 4) {
                        for (int j = 0; j < 4; j++) {
                            int ttj = ((t1 > tt + j) ? tt + j : t1 - 1);
                            fold_v[j] = partials[(ttj * num_heads + head) * 128 + tid];
                        }
                        for (int j_1 = 0; j_1 < 4; j_1++) {
                            float fv = ((t1 > tt + j_1) ? fold_v[j_1] : 0.0f);
                            acc_a[0] = acc_a[0] + fv;
                        }
                    }
                }
                if (length > 0) {
                    float lenf = (float)length;
                    float _fdiv_rn_1 = __fdiv_rn(acc_a[0], lenf);
                    *(reinterpret_cast<float*>(mean_k + ((seg * num_heads + head) * 128 + tid)) + (0)) = _fdiv_rn_1;
                }
            }
        }
        __syncthreads();
    }
}

}  // namespace h3_varlen_stats_sage_sm120a
#undef H3_VARLEN_INF
#undef NUM_MAIN_STAGES
#undef SMEM_FLAG_OFF
#undef SMEM_FLAG_STAGE_BYTES
#undef SMEM_FLAG_STRIDE
#undef SMEM_RED_Q_OFF
#undef SMEM_RED_Q_STAGE_BYTES
#undef SMEM_RED_Q_STRIDE
#undef SMEM_RED_SUM_OFF
#undef SMEM_RED_SUM_STAGE_BYTES
#undef SMEM_RED_SUM_STRIDE
#undef SMEM_TOTAL
#undef THREADS

namespace h3_varlen_quantize_sage_sm120a {

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
kernel_minimax_h3_sm120_varlen_quantize_sage(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, int* __restrict__ tok_seg, int* __restrict__ cu_seqlens, int* __restrict__ seg_tile_begin, float* __restrict__ mean_k, float* __restrict__ q_mean, unsigned int* __restrict__ q4, unsigned int* __restrict__ k4, uint8_t* __restrict__ q_sf, uint8_t* __restrict__ k_sf, unsigned int* __restrict__ qm4, uint8_t* __restrict__ qm_sf, unsigned int* __restrict__ vt4, uint8_t* __restrict__ v_sf, int total_tokens, int total_padded, int num_heads, int num_kblocks)
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
    int num_tblk = total_padded / 32;
    int vwarp = tid >> 5;
    int vlane = tid & 31;
    int vblk = vwarp & 1;
    int vdp = (vwarp >> 1) * 16 + (vlane & 15);
    int vhalf = vlane >> 4;
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
        int seg_begin = cu_seqlens[seg];
        int qt_first = seg_tile_begin[seg];
        int vlive[8];
        vlive[0] = ((tok_base + (vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
        float _vec_load_2[2];
        {
            uint32_t _bf16x2_bits_2;
            _bf16x2_bits_2 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (vblk * 4 + vhalf * 2) < total_tokens) ? tok_base + (vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_2[0])[0]), "=f"((&_vec_load_2[0])[1])
                : "r"(_bf16x2_bits_2));
        }
        vlive[1] = ((tok_base + (vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
        float _vec_load_3[2];
        {
            uint32_t _bf16x2_bits_3;
            _bf16x2_bits_3 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (vblk * 4 + vhalf * 2 + 1) < total_tokens) ? tok_base + (vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_3[0])[0]), "=f"((&_vec_load_3[0])[1])
                : "r"(_bf16x2_bits_3));
        }
        vlive[2] = ((tok_base + (8 + vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
        float _vec_load_4[2];
        {
            uint32_t _bf16x2_bits_4;
            _bf16x2_bits_4 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (8 + vblk * 4 + vhalf * 2) < total_tokens) ? tok_base + (8 + vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_4[0])[0]), "=f"((&_vec_load_4[0])[1])
                : "r"(_bf16x2_bits_4));
        }
        vlive[3] = ((tok_base + (8 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
        float _vec_load_5[2];
        {
            uint32_t _bf16x2_bits_5;
            _bf16x2_bits_5 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (8 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? tok_base + (8 + vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_5[0])[0]), "=f"((&_vec_load_5[0])[1])
                : "r"(_bf16x2_bits_5));
        }
        vlive[4] = ((tok_base + (16 + vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
        float _vec_load_6[2];
        {
            uint32_t _bf16x2_bits_6;
            _bf16x2_bits_6 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (16 + vblk * 4 + vhalf * 2) < total_tokens) ? tok_base + (16 + vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_6[0])[0]), "=f"((&_vec_load_6[0])[1])
                : "r"(_bf16x2_bits_6));
        }
        vlive[5] = ((tok_base + (16 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
        float _vec_load_7[2];
        {
            uint32_t _bf16x2_bits_7;
            _bf16x2_bits_7 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (16 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? tok_base + (16 + vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_7[0])[0]), "=f"((&_vec_load_7[0])[1])
                : "r"(_bf16x2_bits_7));
        }
        vlive[6] = ((tok_base + (24 + vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
        float _vec_load_8[2];
        {
            uint32_t _bf16x2_bits_8;
            _bf16x2_bits_8 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (24 + vblk * 4 + vhalf * 2) < total_tokens) ? tok_base + (24 + vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_8[0])[0]), "=f"((&_vec_load_8[0])[1])
                : "r"(_bf16x2_bits_8));
        }
        vlive[7] = ((tok_base + (24 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
        float _vec_load_9[2];
        {
            uint32_t _bf16x2_bits_9;
            _bf16x2_bits_9 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (24 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? tok_base + (24 + vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_9[0])[0]), "=f"((&_vec_load_9[0])[1])
                : "r"(_bf16x2_bits_9));
        }
        __syncthreads();
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        int qoff = tok_c - seg_begin;
        int qt = qt_first + (qoff >> 7);
        int is_first = ((valid == 1 && (qoff & 127) == 0) ? 1 : 0);
        int qmbase = (qt * num_heads + head) * 128 + col16;
        float qmv[16];
        float qc[16];
        for (int quad = 0; quad < 4; quad++) {
            float _vec_load_10[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(q_mean + (qmbase + quad * 4) + 0);
                _vec_load_10[0 + 0] = _v4.x;
                _vec_load_10[0 + 1] = _v4.y;
                _vec_load_10[0 + 2] = _v4.z;
                _vec_load_10[0 + 3] = _v4.w;
            }
            for (int e = 0; e < 4; e++) {
                qmv[quad * 4 + e] = _vec_load_10[e];
                float qf = _vec_load_0[quad * 4 + e];
                qc[quad * 4 + e] = ((valid == 1) ? qf - _vec_load_10[e] : 0.0f);
            }
        }
        unsigned int qw[2];
        float qsf_a[1];
        float _fabs_0 = fabsf(qc[0]);
        float _fabs_1 = fabsf(qc[1]);
        float _fmax_0 = fmaxf(_fabs_0, _fabs_1);
        float _fabs_2 = fabsf(qc[2]);
        float _fmax_1 = fmaxf(_fmax_0, _fabs_2);
        float _fabs_3 = fabsf(qc[3]);
        float _fmax_2 = fmaxf(_fmax_1, _fabs_3);
        float _fabs_4 = fabsf(qc[4]);
        float _fmax_3 = fmaxf(_fmax_2, _fabs_4);
        float _fabs_5 = fabsf(qc[5]);
        float _fmax_4 = fmaxf(_fmax_3, _fabs_5);
        float _fabs_6 = fabsf(qc[6]);
        float _fmax_5 = fmaxf(_fmax_4, _fabs_6);
        float _fabs_7 = fabsf(qc[7]);
        float _fmax_6 = fmaxf(_fmax_5, _fabs_7);
        float _fabs_8 = fabsf(qc[8]);
        float _fmax_7 = fmaxf(_fmax_6, _fabs_8);
        float _fabs_9 = fabsf(qc[9]);
        float _fmax_8 = fmaxf(_fmax_7, _fabs_9);
        float _fabs_10 = fabsf(qc[10]);
        float _fmax_9 = fmaxf(_fmax_8, _fabs_10);
        float _fabs_11 = fabsf(qc[11]);
        float _fmax_10 = fmaxf(_fmax_9, _fabs_11);
        float _fabs_12 = fabsf(qc[12]);
        float _fmax_11 = fmaxf(_fmax_10, _fabs_12);
        float _fabs_13 = fabsf(qc[13]);
        float _fmax_12 = fmaxf(_fmax_11, _fabs_13);
        float _fabs_14 = fabsf(qc[14]);
        float _fmax_13 = fmaxf(_fmax_12, _fabs_14);
        float _fabs_15 = fabsf(qc[15]);
        float _fmax_14 = fmaxf(_fmax_13, _fabs_15);
        uint16_t _e4m3x2_f32_0;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(0.0f), "f"(_fmax_14 * 0.16666666666666666f));
        uint16_t _e4m3x2_decode_11 = (uint16_t)((unsigned int)_e4m3x2_f32_0 & 0xFFu);
        uint32_t _f16x2_decode_11;
        float _fp8_decode_0;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_11) : "h"(_e4m3x2_decode_11));
        uint16_t _f16_decode_11 = (uint16_t)_f16x2_decode_11;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_0) : "h"(_f16_decode_11));
        float _fdiv_rn_0 = __fdiv_rn(1.0f, _fp8_decode_0);
        qc[0] = qc[0] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[1] = qc[1] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[2] = qc[2] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[3] = qc[3] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[4] = qc[4] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[5] = qc[5] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[6] = qc[6] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[7] = qc[7] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[8] = qc[8] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[9] = qc[9] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[10] = qc[10] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[11] = qc[11] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[12] = qc[12] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[13] = qc[13] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[14] = qc[14] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        qc[15] = qc[15] * ((_fp8_decode_0 != 0.0f) ? _fdiv_rn_0 : 0.0f);
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[0]) : "f"(qc[0]), "f"(qc[1]), "f"(qc[2]), "f"(qc[3]), "f"(qc[4]), "f"(qc[5]), "f"(qc[6]), "f"(qc[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[1]) : "f"(qc[8]), "f"(qc[9]), "f"(qc[10]), "f"(qc[11]), "f"(qc[12]), "f"(qc[13]), "f"(qc[14]), "f"(qc[15]));
        qsf_a[0] = _fmax_14 * 0.16666666666666666f;
        unsigned int qstore[4];
        qstore[0] = qw[0];
        qstore[1] = qw[1];
        unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, qw[0], 1);
        qstore[2] = _shfl_xor_0;
        unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, qw[1], 1);
        qstore[3] = _shfl_xor_1;
        if (valid == 1) {
            {
                unsigned short _sf_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(qsf_a[0]));
                *(reinterpret_cast<unsigned char*>(q_sf + ((tok * num_heads + head) * 8 + sub)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
            }
            if ((sub & 1) == 0) {
                reinterpret_cast<int4*>(q4 + (((tok * num_heads + head) * 64 + sub * 8) / 4))[0] = reinterpret_cast<int4*>(qstore)[0];
            }
        }
        if (is_first == 1) {
            unsigned int mw[2];
            float msf_a[1];
            float _fabs_16 = fabsf(qmv[0]);
            float _fabs_17 = fabsf(qmv[1]);
            float _fmax_15 = fmaxf(_fabs_16, _fabs_17);
            float _fabs_18 = fabsf(qmv[2]);
            float _fmax_16 = fmaxf(_fmax_15, _fabs_18);
            float _fabs_19 = fabsf(qmv[3]);
            float _fmax_17 = fmaxf(_fmax_16, _fabs_19);
            float _fabs_20 = fabsf(qmv[4]);
            float _fmax_18 = fmaxf(_fmax_17, _fabs_20);
            float _fabs_21 = fabsf(qmv[5]);
            float _fmax_19 = fmaxf(_fmax_18, _fabs_21);
            float _fabs_22 = fabsf(qmv[6]);
            float _fmax_20 = fmaxf(_fmax_19, _fabs_22);
            float _fabs_23 = fabsf(qmv[7]);
            float _fmax_21 = fmaxf(_fmax_20, _fabs_23);
            float _fabs_24 = fabsf(qmv[8]);
            float _fmax_22 = fmaxf(_fmax_21, _fabs_24);
            float _fabs_25 = fabsf(qmv[9]);
            float _fmax_23 = fmaxf(_fmax_22, _fabs_25);
            float _fabs_26 = fabsf(qmv[10]);
            float _fmax_24 = fmaxf(_fmax_23, _fabs_26);
            float _fabs_27 = fabsf(qmv[11]);
            float _fmax_25 = fmaxf(_fmax_24, _fabs_27);
            float _fabs_28 = fabsf(qmv[12]);
            float _fmax_26 = fmaxf(_fmax_25, _fabs_28);
            float _fabs_29 = fabsf(qmv[13]);
            float _fmax_27 = fmaxf(_fmax_26, _fabs_29);
            float _fabs_30 = fabsf(qmv[14]);
            float _fmax_28 = fmaxf(_fmax_27, _fabs_30);
            float _fabs_31 = fabsf(qmv[15]);
            float _fmax_29 = fmaxf(_fmax_28, _fabs_31);
            uint16_t _e4m3x2_f32_1;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(0.0f), "f"(_fmax_29 * 0.16666666666666666f));
            uint16_t _e4m3x2_decode_12 = (uint16_t)((unsigned int)_e4m3x2_f32_1 & 0xFFu);
            uint32_t _f16x2_decode_12;
            float _fp8_decode_1;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_12) : "h"(_e4m3x2_decode_12));
            uint16_t _f16_decode_12 = (uint16_t)_f16x2_decode_12;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_1) : "h"(_f16_decode_12));
            float _fdiv_rn_1 = __fdiv_rn(1.0f, _fp8_decode_1);
            qmv[0] = qmv[0] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[1] = qmv[1] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[2] = qmv[2] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[3] = qmv[3] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[4] = qmv[4] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[5] = qmv[5] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[6] = qmv[6] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[7] = qmv[7] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[8] = qmv[8] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[9] = qmv[9] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[10] = qmv[10] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[11] = qmv[11] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[12] = qmv[12] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[13] = qmv[13] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[14] = qmv[14] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            qmv[15] = qmv[15] * ((_fp8_decode_1 != 0.0f) ? _fdiv_rn_1 : 0.0f);
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(mw[0]) : "f"(qmv[0]), "f"(qmv[1]), "f"(qmv[2]), "f"(qmv[3]), "f"(qmv[4]), "f"(qmv[5]), "f"(qmv[6]), "f"(qmv[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(mw[1]) : "f"(qmv[8]), "f"(qmv[9]), "f"(qmv[10]), "f"(qmv[11]), "f"(qmv[12]), "f"(qmv[13]), "f"(qmv[14]), "f"(qmv[15]));
            msf_a[0] = _fmax_29 * 0.16666666666666666f;
            int qmrow = qt * num_heads + head;
            *(reinterpret_cast<unsigned int*>(qm4 + (qmrow * 16 + sub * 2)) + (0)) = mw[0];
            *(reinterpret_cast<unsigned int*>(qm4 + (qmrow * 16 + sub * 2 + 1)) + (0)) = mw[1];
            {
                unsigned short _sf_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(msf_a[0]));
                *(reinterpret_cast<unsigned char*>(qm_sf + (qmrow * 8 + sub)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
            }
        }
        int mbase = (seg * num_heads + head) * 128 + col16;
        float kc[16];
        for (int quad_1 = 0; quad_1 < 4; quad_1++) {
            float _vec_load_11[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(mean_k + (mbase + quad_1 * 4) + 0);
                _vec_load_11[0 + 0] = _v4.x;
                _vec_load_11[0 + 1] = _v4.y;
                _vec_load_11[0 + 2] = _v4.z;
                _vec_load_11[0 + 3] = _v4.w;
            }
            for (int e_1 = 0; e_1 < 4; e_1++) {
                float kf = _vec_load_1[quad_1 * 4 + e_1];
                kc[quad_1 * 4 + e_1] = ((valid == 1) ? kf - _vec_load_11[e_1] : 0.0f);
            }
        }
        unsigned int kw[2];
        float ksf_a[1];
        float _fabs_32 = fabsf(kc[0]);
        float _fabs_33 = fabsf(kc[1]);
        float _fmax_30 = fmaxf(_fabs_32, _fabs_33);
        float _fabs_34 = fabsf(kc[2]);
        float _fmax_31 = fmaxf(_fmax_30, _fabs_34);
        float _fabs_35 = fabsf(kc[3]);
        float _fmax_32 = fmaxf(_fmax_31, _fabs_35);
        float _fabs_36 = fabsf(kc[4]);
        float _fmax_33 = fmaxf(_fmax_32, _fabs_36);
        float _fabs_37 = fabsf(kc[5]);
        float _fmax_34 = fmaxf(_fmax_33, _fabs_37);
        float _fabs_38 = fabsf(kc[6]);
        float _fmax_35 = fmaxf(_fmax_34, _fabs_38);
        float _fabs_39 = fabsf(kc[7]);
        float _fmax_36 = fmaxf(_fmax_35, _fabs_39);
        float _fabs_40 = fabsf(kc[8]);
        float _fmax_37 = fmaxf(_fmax_36, _fabs_40);
        float _fabs_41 = fabsf(kc[9]);
        float _fmax_38 = fmaxf(_fmax_37, _fabs_41);
        float _fabs_42 = fabsf(kc[10]);
        float _fmax_39 = fmaxf(_fmax_38, _fabs_42);
        float _fabs_43 = fabsf(kc[11]);
        float _fmax_40 = fmaxf(_fmax_39, _fabs_43);
        float _fabs_44 = fabsf(kc[12]);
        float _fmax_41 = fmaxf(_fmax_40, _fabs_44);
        float _fabs_45 = fabsf(kc[13]);
        float _fmax_42 = fmaxf(_fmax_41, _fabs_45);
        float _fabs_46 = fabsf(kc[14]);
        float _fmax_43 = fmaxf(_fmax_42, _fabs_46);
        float _fabs_47 = fabsf(kc[15]);
        float _fmax_44 = fmaxf(_fmax_43, _fabs_47);
        uint16_t _e4m3x2_f32_2;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(0.0f), "f"(_fmax_44 * 0.16666666666666666f));
        uint16_t _e4m3x2_decode_14 = (uint16_t)((unsigned int)_e4m3x2_f32_2 & 0xFFu);
        uint32_t _f16x2_decode_14;
        float _fp8_decode_2;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_14) : "h"(_e4m3x2_decode_14));
        uint16_t _f16_decode_14 = (uint16_t)_f16x2_decode_14;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_2) : "h"(_f16_decode_14));
        float _fdiv_rn_2 = __fdiv_rn(1.0f, _fp8_decode_2);
        kc[0] = kc[0] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[1] = kc[1] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[2] = kc[2] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[3] = kc[3] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[4] = kc[4] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[5] = kc[5] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[6] = kc[6] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[7] = kc[7] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[8] = kc[8] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[9] = kc[9] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[10] = kc[10] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[11] = kc[11] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[12] = kc[12] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[13] = kc[13] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[14] = kc[14] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        kc[15] = kc[15] * ((_fp8_decode_2 != 0.0f) ? _fdiv_rn_2 : 0.0f);
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(kw[0]) : "f"(kc[0]), "f"(kc[1]), "f"(kc[2]), "f"(kc[3]), "f"(kc[4]), "f"(kc[5]), "f"(kc[6]), "f"(kc[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(kw[1]) : "f"(kc[8]), "f"(kc[9]), "f"(kc[10]), "f"(kc[11]), "f"(kc[12]), "f"(kc[13]), "f"(kc[14]), "f"(kc[15]));
        ksf_a[0] = _fmax_44 * 0.16666666666666666f;
        unsigned int kstore[4];
        unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, kw[0], 1);
        kstore[0] = _shfl_xor_2;
        unsigned int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, kw[1], 1);
        kstore[1] = _shfl_xor_3;
        kstore[2] = kw[0];
        kstore[3] = kw[1];
        {
            unsigned short _sf_pair;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(ksf_a[0]));
            *(reinterpret_cast<unsigned char*>(k_sf + (((head * num_kblocks + (tok >> 7)) * 128 + (tok & 127)) * 8 + sub)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
        }
        if (valid == 1) {
            if ((sub & 1) == 1) {
                reinterpret_cast<int4*>(k4 + (((tok * num_heads + head) * 64 + (sub - 1) * 8) / 4))[0] = reinterpret_cast<int4*>(kstore)[0];
            }
        }
        float va[8];
        float vb[8];
        va[0] = _vec_load_2[0];
        vb[0] = _vec_load_2[1];
        va[1] = _vec_load_3[0];
        vb[1] = _vec_load_3[1];
        va[2] = _vec_load_4[0];
        vb[2] = _vec_load_4[1];
        va[3] = _vec_load_5[0];
        vb[3] = _vec_load_5[1];
        va[4] = _vec_load_6[0];
        vb[4] = _vec_load_6[1];
        va[5] = _vec_load_7[0];
        vb[5] = _vec_load_7[1];
        va[6] = _vec_load_8[0];
        vb[6] = _vec_load_8[1];
        va[7] = _vec_load_9[0];
        vb[7] = _vec_load_9[1];
        va[0] = ((vlive[0] == 1) ? va[0] : 0.0f);
        vb[0] = ((vlive[0] == 1) ? vb[0] : 0.0f);
        va[1] = ((vlive[1] == 1) ? va[1] : 0.0f);
        vb[1] = ((vlive[1] == 1) ? vb[1] : 0.0f);
        va[2] = ((vlive[2] == 1) ? va[2] : 0.0f);
        vb[2] = ((vlive[2] == 1) ? vb[2] : 0.0f);
        va[3] = ((vlive[3] == 1) ? va[3] : 0.0f);
        vb[3] = ((vlive[3] == 1) ? vb[3] : 0.0f);
        va[4] = ((vlive[4] == 1) ? va[4] : 0.0f);
        vb[4] = ((vlive[4] == 1) ? vb[4] : 0.0f);
        va[5] = ((vlive[5] == 1) ? va[5] : 0.0f);
        vb[5] = ((vlive[5] == 1) ? vb[5] : 0.0f);
        va[6] = ((vlive[6] == 1) ? va[6] : 0.0f);
        vb[6] = ((vlive[6] == 1) ? vb[6] : 0.0f);
        va[7] = ((vlive[7] == 1) ? va[7] : 0.0f);
        vb[7] = ((vlive[7] == 1) ? vb[7] : 0.0f);
        float amax_a = 0.0f;
        float amax_b = 0.0f;
        for (int ii = 0; ii < 8; ii++) {
            float _fabs_48 = fabsf(va[ii]);
            float _fmax_45 = fmaxf(amax_a, _fabs_48);
            amax_a = _fmax_45;
            float _fabs_49 = fabsf(vb[ii]);
            float _fmax_46 = fmaxf(amax_b, _fabs_49);
            amax_b = _fmax_46;
        }
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, amax_a, 16);
        float _fmax_47 = fmaxf(amax_a, _shfl_xor_4);
        amax_a = _fmax_47;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, amax_b, 16);
        float _fmax_48 = fmaxf(amax_b, _shfl_xor_5);
        amax_b = _fmax_48;
        float vsf_a = amax_a * 0.16666666666666666f;
        float vsf_b = amax_b * 0.16666666666666666f;
        uint16_t _e4m3x2_f32_3;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(vsf_b), "f"(vsf_a));
        uint16_t _e4m3x2_decode_15 = (uint16_t)((unsigned int)_e4m3x2_f32_3 & 0xFFu);
        uint32_t _f16x2_decode_15;
        float _fp8_decode_3;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_15) : "h"(_e4m3x2_decode_15));
        uint16_t _f16_decode_15 = (uint16_t)_f16x2_decode_15;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_3) : "h"(_f16_decode_15));
        float vdec_a = _fp8_decode_3;
        uint16_t _e4m3x2_decode_16 = (uint16_t)((unsigned int)_e4m3x2_f32_3 >> 8 & 0xFFu);
        uint32_t _f16x2_decode_16;
        float _fp8_decode_4;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_16) : "h"(_e4m3x2_decode_16));
        uint16_t _f16_decode_16 = (uint16_t)_f16x2_decode_16;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_4) : "h"(_f16_decode_16));
        float vdec_b = _fp8_decode_4;
        float _fdiv_rn_3 = __fdiv_rn(1.0f, vdec_a);
        float vinv_a = ((vdec_a != 0.0f) ? _fdiv_rn_3 : 0.0f);
        float _fdiv_rn_4 = __fdiv_rn(1.0f, vdec_b);
        float vinv_b = ((vdec_b != 0.0f) ? _fdiv_rn_4 : 0.0f);
        for (int ii_1 = 0; ii_1 < 8; ii_1++) {
            va[ii_1] = va[ii_1] * vinv_a;
            vb[ii_1] = vb[ii_1] * vinv_b;
        }
        unsigned int vwa[1];
        unsigned int vwb[1];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(vwa[0]) : "f"(va[0]), "f"(va[1]), "f"(va[2]), "f"(va[3]), "f"(va[4]), "f"(va[5]), "f"(va[6]), "f"(va[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(vwb[0]) : "f"(vb[0]), "f"(vb[1]), "f"(vb[2]), "f"(vb[3]), "f"(vb[4]), "f"(vb[5]), "f"(vb[6]), "f"(vb[7]));
        int row_words = total_padded >> 3;
        int vt_word = (head * 128 + vdp * 2) * row_words + (tok_base >> 3) + vblk * 2 + vhalf;
        *(reinterpret_cast<unsigned int*>(vt4 + vt_word) + (0)) = vwa[0];
        *(reinterpret_cast<unsigned int*>(vt4 + (vt_word + row_words)) + (0)) = vwb[0];
        if (vhalf == 0) {
            int vsf_base = ((head * num_kblocks + (tok_base >> 7)) * 128 + vdp * 2) * 8 + (((tok_base & 127) >> 4) + vblk);
            {
                unsigned short _sf_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(vsf_a));
                *(reinterpret_cast<unsigned char*>(v_sf + vsf_base) + (0)) = (unsigned char)(_sf_pair & 0x7F);
            }
            {
                unsigned short _sf_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(vsf_b));
                *(reinterpret_cast<unsigned char*>(v_sf + (vsf_base + 8)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
            }
        }
    }
}

}  // namespace h3_varlen_quantize_sage_sm120a
#undef H3_VARLEN_INF
#undef NUM_MAIN_STAGES
#undef THREADS

namespace h3_varlen_attention_sage_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_VARLEN_INF CUDART_INF_F
#define NUM_KV_STAGES 2
#define NUM_DELTA_STAGES 2
#define SMEM_Q_OFF 1024
#define SMEM_Q_STAGE_BYTES 8192
#define SMEM_Q_STRIDE 8192
#define SMEM_K_STAGE_OFF 9216
#define SMEM_K_STAGE_STAGE_BYTES 8192
#define SMEM_K_STAGE_STRIDE 8192
#define SMEM_KSF_STAGE_OFF 25600
#define SMEM_KSF_STAGE_STAGE_BYTES 1024
#define SMEM_KSF_STAGE_STRIDE 1024
#define SMEM_VT_STAGE_OFF 27648
#define SMEM_VT_STAGE_STAGE_BYTES 8192
#define SMEM_VT_STAGE_STRIDE 8192
#define SMEM_VSF_STAGE_OFF 44032
#define SMEM_VSF_STAGE_STAGE_BYTES 1024
#define SMEM_VSF_STAGE_STRIDE 1024
#define SMEM_DELTA_OFF 46080
#define SMEM_DELTA_STAGE_BYTES 2048
#define SMEM_DELTA_STRIDE 2048
#define SMEM_TOTAL 48128
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
kernel_minimax_h3_sm120_varlen_attention_sage(const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map, const __grid_constant__ CUtensorMap KSF_map, const __grid_constant__ CUtensorMap VT_map, const __grid_constant__ CUtensorMap VSF_map, __nv_bfloat16* __restrict__ O, unsigned int* __restrict__ q_sf, unsigned int* __restrict__ qm4, unsigned int* __restrict__ qm_sf, int* __restrict__ seg_tile_begin, int* __restrict__ unit_table, int total_units, int num_heads, int num_kblocks, float softmax_scale_log2)
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
    #define delta_ready_addr (mbar_base + 72)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* Q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int Q_addr = smem + 1024;
    uint8_t* K_stage = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int K_stage_addr = smem + 9216;
    unsigned int* KSF_stage = reinterpret_cast<unsigned int*>(smem_raw + 25600);
    const int KSF_stage_addr = smem + 25600;
    uint8_t* VT_stage = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int VT_stage_addr = smem + 27648;
    unsigned int* VSF_stage = reinterpret_cast<unsigned int*>(smem_raw + 44032);
    const int VSF_stage_addr = smem + 44032;
    float* DELTA = reinterpret_cast<float*>(smem_raw + 46080);
    const int DELTA_addr = smem + 46080;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 11 barriers)
    // Mbarriers at smem_raw[0..88)

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
            // --- pipeline 'delta' ---
            // delta_ready: 2 barriers, init_count=8
            mbarrier_init(smem + 72, 8);
            mbarrier_init(smem + 80, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    unsigned int kv_stage = 0;
    unsigned int kv_phase = 0;
    unsigned int d_stage = 0;
    unsigned int d_phase = 0;
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
                mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 9216);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                    :: "r"(VT_stage_addr + kv_stage * 8192), "l"((&VT_map)), "r"(kb_first * 64), "r"(0), "r"(head),
                       "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3}], [%4], %5;"
                    :: "r"(VSF_stage_addr + kv_stage * 1024), "l"((&VSF_map)), "r"(0), "r"((head * num_kblocks + kb_first) * 64),
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
                    mbarrier_arrive_expect_tx(v_full_addr + ((kv_stage + 1) % 2) * 8, 9216);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(VT_stage_addr + (kv_stage + 1) % 2 * 8192), "l"((&VT_map)), "r"((kb_first + 1) * 64), "r"(0), "r"(head),
                           "r"(v_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(VSF_stage_addr + (kv_stage + 1) % 2 * 1024), "l"((&VSF_map)), "r"(0), "r"((head * num_kblocks + (kb_first + 1)) * 64),
                           "r"(v_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        int row0 = q_row0 + warp * 16 + (lane >> 2);
        int row1 = row0 + 8;
        int row0_c = ((row0 < seg_end_v) ? row0 : seg_end_v - 1);
        int row1_c = ((row1 < seg_end_v) ? row1 : seg_end_v - 1);
        int sf_row = (((lane & 1) == 1) ? row1_c : row0_c);
        unsigned int qsf[2];
        for (int u = 0; u < 2; u++) {
            qsf[u] = q_sf[(sf_row * num_heads + head) * 2 + u];
        }
        int qmrow = (seg_tile_begin[seg] + q_tile) * num_heads + head;
        unsigned int qm_frag0[4];
        unsigned int qm_frag1[4];
        unsigned int qmsf[2];
        unsigned int qmb0[2];
        unsigned int qmb1[2];
        qm_frag0[0] = ((lane >> 2 == 0) ? qm4[qmrow * 16 + (lane & 3)] : (unsigned int)0);
        qm_frag0[1] = (unsigned int)0;
        qm_frag0[2] = ((lane >> 2 == 0) ? qm4[qmrow * 16 + 4 + (lane & 3)] : (unsigned int)0);
        qm_frag0[3] = (unsigned int)0;
        qmsf[0] = qm_sf[qmrow * 2];
        qm_frag1[0] = ((lane >> 2 == 0) ? qm4[qmrow * 16 + 8 + (lane & 3)] : (unsigned int)0);
        qm_frag1[1] = (unsigned int)0;
        qm_frag1[2] = ((lane >> 2 == 0) ? qm4[qmrow * 16 + 8 + 4 + (lane & 3)] : (unsigned int)0);
        qm_frag1[3] = (unsigned int)0;
        qmsf[1] = qm_sf[qmrow * 2 + 1];
        unsigned int ka0[4];
        unsigned int ka1[4];
        unsigned int q_frags[8];
        unsigned int k_frag[8];
        unsigned int v_frag[8];
        float s_acc[64];
        float d_tmp[8];
        unsigned int p_frag[8];
        float p_tmp[32];
        unsigned int psf[2];
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
        {
            mbarrier_wait(k_full_addr + (kv_stage) * 8, kv_phase);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_4[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_4[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((2 * warp * 8 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(d_tmp[0]), "=f"(d_tmp[1]), "=f"(d_tmp[2]), "=f"(d_tmp[3])
                : "r"(qm_frag0[0]), "r"(qm_frag0[1]), "r"(qm_frag0[2]), "r"(qm_frag0[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qmsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(d_tmp[0]), "+f"(d_tmp[1]), "+f"(d_tmp[2]), "+f"(d_tmp[3])
                : "r"(qm_frag1[0]), "r"(qm_frag1[1]), "r"(qm_frag1[2]), "r"(qm_frag1[3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qmsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_4[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_5[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_5[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)(((2 * warp + 1) * 8 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(d_tmp[4]), "=f"(d_tmp[(4) + 1]), "=f"(d_tmp[(4) + 2]), "=f"(d_tmp[(4) + 3])
                : "r"(qm_frag0[0]), "r"(qm_frag0[1]), "r"(qm_frag0[2]), "r"(qm_frag0[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qmsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(d_tmp[4]), "+f"(d_tmp[(4) + 1]), "+f"(d_tmp[(4) + 2]), "+f"(d_tmp[(4) + 3])
                : "r"(qm_frag1[0]), "r"(qm_frag1[1]), "r"(qm_frag1[2]), "r"(qm_frag1[3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qmsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_5[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            if (lane < 4) {
                DELTA[d_stage * 128 + (unsigned int)(2 * warp * 8) + (unsigned int)((lane & 3) * 2)] = d_tmp[0];
                DELTA[d_stage * 128 + (unsigned int)(2 * warp * 8) + (unsigned int)((lane & 3) * 2) + 1] = d_tmp[1];
                DELTA[d_stage * 128 + (unsigned int)((2 * warp + 1) * 8) + (unsigned int)((lane & 3) * 2)] = d_tmp[4];
                DELTA[d_stage * 128 + (unsigned int)((2 * warp + 1) * 8) + (unsigned int)((lane & 3) * 2) + 1] = d_tmp[5];
            }
            if (elect_sync()) {
                mbarrier_arrive(delta_ready_addr + (d_stage) * 8);
            }
        }
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
            mbarrier_wait(delta_ready_addr + (d_stage) * 8, d_phase);
            float _DELTA_reg_0[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_0[_lr] = _smem_ptr[(d_stage * 128 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[0] = _DELTA_reg_0[0];
            s_acc[1] = _DELTA_reg_0[1];
            s_acc[2] = _DELTA_reg_0[0];
            s_acc[3] = _DELTA_reg_0[1];
            float _DELTA_reg_1[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_1[_lr] = _smem_ptr[(d_stage * 128 + 8 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[4] = _DELTA_reg_1[0];
            s_acc[5] = _DELTA_reg_1[1];
            s_acc[6] = _DELTA_reg_1[0];
            s_acc[7] = _DELTA_reg_1[1];
            float _DELTA_reg_2[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_2[_lr] = _smem_ptr[(d_stage * 128 + 16 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[8] = _DELTA_reg_2[0];
            s_acc[9] = _DELTA_reg_2[1];
            s_acc[10] = _DELTA_reg_2[0];
            s_acc[11] = _DELTA_reg_2[1];
            float _DELTA_reg_3[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_3[_lr] = _smem_ptr[(d_stage * 128 + 24 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[12] = _DELTA_reg_3[0];
            s_acc[13] = _DELTA_reg_3[1];
            s_acc[14] = _DELTA_reg_3[0];
            s_acc[15] = _DELTA_reg_3[1];
            float _DELTA_reg_4[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_4[_lr] = _smem_ptr[(d_stage * 128 + 32 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[16] = _DELTA_reg_4[0];
            s_acc[17] = _DELTA_reg_4[1];
            s_acc[18] = _DELTA_reg_4[0];
            s_acc[19] = _DELTA_reg_4[1];
            float _DELTA_reg_5[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_5[_lr] = _smem_ptr[(d_stage * 128 + 40 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[20] = _DELTA_reg_5[0];
            s_acc[21] = _DELTA_reg_5[1];
            s_acc[22] = _DELTA_reg_5[0];
            s_acc[23] = _DELTA_reg_5[1];
            float _DELTA_reg_6[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_6[_lr] = _smem_ptr[(d_stage * 128 + 48 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[24] = _DELTA_reg_6[0];
            s_acc[25] = _DELTA_reg_6[1];
            s_acc[26] = _DELTA_reg_6[0];
            s_acc[27] = _DELTA_reg_6[1];
            float _DELTA_reg_7[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_7[_lr] = _smem_ptr[(d_stage * 128 + 56 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[28] = _DELTA_reg_7[0];
            s_acc[29] = _DELTA_reg_7[1];
            s_acc[30] = _DELTA_reg_7[0];
            s_acc[31] = _DELTA_reg_7[1];
            float _DELTA_reg_8[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_8[_lr] = _smem_ptr[(d_stage * 128 + 64 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[32] = _DELTA_reg_8[0];
            s_acc[33] = _DELTA_reg_8[1];
            s_acc[34] = _DELTA_reg_8[0];
            s_acc[35] = _DELTA_reg_8[1];
            float _DELTA_reg_9[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_9[_lr] = _smem_ptr[(d_stage * 128 + 72 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[36] = _DELTA_reg_9[0];
            s_acc[37] = _DELTA_reg_9[1];
            s_acc[38] = _DELTA_reg_9[0];
            s_acc[39] = _DELTA_reg_9[1];
            float _DELTA_reg_10[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_10[_lr] = _smem_ptr[(d_stage * 128 + 80 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[40] = _DELTA_reg_10[0];
            s_acc[41] = _DELTA_reg_10[1];
            s_acc[42] = _DELTA_reg_10[0];
            s_acc[43] = _DELTA_reg_10[1];
            float _DELTA_reg_11[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_11[_lr] = _smem_ptr[(d_stage * 128 + 88 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[44] = _DELTA_reg_11[0];
            s_acc[45] = _DELTA_reg_11[1];
            s_acc[46] = _DELTA_reg_11[0];
            s_acc[47] = _DELTA_reg_11[1];
            float _DELTA_reg_12[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_12[_lr] = _smem_ptr[(d_stage * 128 + 96 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[48] = _DELTA_reg_12[0];
            s_acc[49] = _DELTA_reg_12[1];
            s_acc[50] = _DELTA_reg_12[0];
            s_acc[51] = _DELTA_reg_12[1];
            float _DELTA_reg_13[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_13[_lr] = _smem_ptr[(d_stage * 128 + 104 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[52] = _DELTA_reg_13[0];
            s_acc[53] = _DELTA_reg_13[1];
            s_acc[54] = _DELTA_reg_13[0];
            s_acc[55] = _DELTA_reg_13[1];
            float _DELTA_reg_14[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_14[_lr] = _smem_ptr[(d_stage * 128 + 112 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[56] = _DELTA_reg_14[0];
            s_acc[57] = _DELTA_reg_14[1];
            s_acc[58] = _DELTA_reg_14[0];
            s_acc[59] = _DELTA_reg_14[1];
            float _DELTA_reg_15[2];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(DELTA);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _DELTA_reg_15[_lr] = _smem_ptr[(d_stage * 128 + 120 + (unsigned int)((lane & 3) * 2)) + _lr];
            }
            s_acc[60] = _DELTA_reg_15[0];
            s_acc[61] = _DELTA_reg_15[1];
            s_acc[62] = _DELTA_reg_15[0];
            s_acc[63] = _DELTA_reg_15[1];
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_7[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_7[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((lane >> 2) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_7[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_8[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_8[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_8[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_9[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_9[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[8]), "+f"(s_acc[(8) + 1]), "+f"(s_acc[(8) + 2]), "+f"(s_acc[(8) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[8]), "+f"(s_acc[(8) + 1]), "+f"(s_acc[(8) + 2]), "+f"(s_acc[(8) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_9[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_10[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_10[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[12]), "+f"(s_acc[(12) + 1]), "+f"(s_acc[(12) + 2]), "+f"(s_acc[(12) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[12]), "+f"(s_acc[(12) + 1]), "+f"(s_acc[(12) + 2]), "+f"(s_acc[(12) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_10[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_11[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_11[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((32 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[16]), "+f"(s_acc[(16) + 1]), "+f"(s_acc[(16) + 2]), "+f"(s_acc[(16) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[16]), "+f"(s_acc[(16) + 1]), "+f"(s_acc[(16) + 2]), "+f"(s_acc[(16) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_11[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_12[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_12[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((40 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[20]), "+f"(s_acc[(20) + 1]), "+f"(s_acc[(20) + 2]), "+f"(s_acc[(20) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[20]), "+f"(s_acc[(20) + 1]), "+f"(s_acc[(20) + 2]), "+f"(s_acc[(20) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_12[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_13[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_13[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((48 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[24]), "+f"(s_acc[(24) + 1]), "+f"(s_acc[(24) + 2]), "+f"(s_acc[(24) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[24]), "+f"(s_acc[(24) + 1]), "+f"(s_acc[(24) + 2]), "+f"(s_acc[(24) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_13[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_14[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_14[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((56 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[28]), "+f"(s_acc[(28) + 1]), "+f"(s_acc[(28) + 2]), "+f"(s_acc[(28) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[28]), "+f"(s_acc[(28) + 1]), "+f"(s_acc[(28) + 2]), "+f"(s_acc[(28) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_14[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_15[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_15[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((64 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[32]), "+f"(s_acc[(32) + 1]), "+f"(s_acc[(32) + 2]), "+f"(s_acc[(32) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[32]), "+f"(s_acc[(32) + 1]), "+f"(s_acc[(32) + 2]), "+f"(s_acc[(32) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_15[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_16[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_16[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((72 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[36]), "+f"(s_acc[(36) + 1]), "+f"(s_acc[(36) + 2]), "+f"(s_acc[(36) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[36]), "+f"(s_acc[(36) + 1]), "+f"(s_acc[(36) + 2]), "+f"(s_acc[(36) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_16[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_17[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_17[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((80 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[40]), "+f"(s_acc[(40) + 1]), "+f"(s_acc[(40) + 2]), "+f"(s_acc[(40) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[40]), "+f"(s_acc[(40) + 1]), "+f"(s_acc[(40) + 2]), "+f"(s_acc[(40) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_17[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_18[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_18[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((88 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[44]), "+f"(s_acc[(44) + 1]), "+f"(s_acc[(44) + 2]), "+f"(s_acc[(44) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[44]), "+f"(s_acc[(44) + 1]), "+f"(s_acc[(44) + 2]), "+f"(s_acc[(44) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_18[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_19[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_19[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((96 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[48]), "+f"(s_acc[(48) + 1]), "+f"(s_acc[(48) + 2]), "+f"(s_acc[(48) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[48]), "+f"(s_acc[(48) + 1]), "+f"(s_acc[(48) + 2]), "+f"(s_acc[(48) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_19[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_20[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_20[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((104 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[52]), "+f"(s_acc[(52) + 1]), "+f"(s_acc[(52) + 2]), "+f"(s_acc[(52) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[52]), "+f"(s_acc[(52) + 1]), "+f"(s_acc[(52) + 2]), "+f"(s_acc[(52) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_20[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _KSF_stage_reg_21[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_21[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((112 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[56]), "+f"(s_acc[(56) + 1]), "+f"(s_acc[(56) + 2]), "+f"(s_acc[(56) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[56]), "+f"(s_acc[(56) + 1]), "+f"(s_acc[(56) + 2]), "+f"(s_acc[(56) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_21[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _KSF_stage_reg_22[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _KSF_stage_reg_22[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((120 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[60]), "+f"(s_acc[(60) + 1]), "+f"(s_acc[(60) + 2]), "+f"(s_acc[(60) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[60]), "+f"(s_acc[(60) + 1]), "+f"(s_acc[(60) + 2]), "+f"(s_acc[(60) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_22[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            if (elect_sync()) {
                mbarrier_arrive(k_empty_addr + (kv_stage) * 8);
            }
            if (kb_last >= kb + 1) {
                mbarrier_wait(k_full_addr + (((kv_stage + 1 < 2) ? kv_stage + 1 : (unsigned int)0)) * 8, ((kv_stage + 1 < 2) ? kv_phase : kv_phase ^ 1));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                    : "r"(K_stage_addr + ((kv_stage + 1 < 2) ? kv_stage + 1 : (unsigned int)0) * 8192 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                    : "r"(K_stage_addr + ((kv_stage + 1 < 2) ? kv_stage + 1 : (unsigned int)0) * 8192 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                    : "memory");
                unsigned int _KSF_stage_reg_23[2];
                {
                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                    #pragma unroll
                    for (int _lr = 0; _lr < 2; _lr++)
                        _KSF_stage_reg_23[_lr] = _smem_ptr[(((kv_stage + 1 < 2) ? kv_stage + 1 : (unsigned int)0) * 256 + (unsigned int)((2 * warp * 8 + (lane >> 2)) * 2)) + _lr];
                }
                asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                    : "=f"(d_tmp[0]), "=f"(d_tmp[1]), "=f"(d_tmp[2]), "=f"(d_tmp[3])
                    : "r"(qm_frag0[0]), "r"(qm_frag0[1]), "r"(qm_frag0[2]), "r"(qm_frag0[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qmsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_23[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                    : "+f"(d_tmp[0]), "+f"(d_tmp[1]), "+f"(d_tmp[2]), "+f"(d_tmp[3])
                    : "r"(qm_frag1[0]), "r"(qm_frag1[1]), "r"(qm_frag1[2]), "r"(qm_frag1[3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qmsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_23[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                unsigned int _KSF_stage_reg_24[2];
                {
                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(KSF_stage);
                    #pragma unroll
                    for (int _lr = 0; _lr < 2; _lr++)
                        _KSF_stage_reg_24[_lr] = _smem_ptr[(((kv_stage + 1 < 2) ? kv_stage + 1 : (unsigned int)0) * 256 + (unsigned int)(((2 * warp + 1) * 8 + (lane >> 2)) * 2)) + _lr];
                }
                asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                    : "=f"(d_tmp[4]), "=f"(d_tmp[(4) + 1]), "=f"(d_tmp[(4) + 2]), "=f"(d_tmp[(4) + 3])
                    : "r"(qm_frag0[0]), "r"(qm_frag0[1]), "r"(qm_frag0[2]), "r"(qm_frag0[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qmsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_24[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                    : "+f"(d_tmp[4]), "+f"(d_tmp[(4) + 1]), "+f"(d_tmp[(4) + 2]), "+f"(d_tmp[(4) + 3])
                    : "r"(qm_frag1[0]), "r"(qm_frag1[1]), "r"(qm_frag1[2]), "r"(qm_frag1[3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qmsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_KSF_stage_reg_24[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                if (lane < 4) {
                    DELTA[((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0) * 128 + (unsigned int)(2 * warp * 8) + (unsigned int)((lane & 3) * 2)] = d_tmp[0];
                    DELTA[((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0) * 128 + (unsigned int)(2 * warp * 8) + (unsigned int)((lane & 3) * 2) + 1] = d_tmp[1];
                    DELTA[((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0) * 128 + (unsigned int)((2 * warp + 1) * 8) + (unsigned int)((lane & 3) * 2)] = d_tmp[4];
                    DELTA[((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0) * 128 + (unsigned int)((2 * warp + 1) * 8) + (unsigned int)((lane & 3) * 2) + 1] = d_tmp[5];
                }
                if (elect_sync()) {
                    mbarrier_arrive(delta_ready_addr + (((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0)) * 8);
                }
            }
            d_stage += 1;
            if (d_stage == 2) { d_stage = 0; d_phase ^= 1; }
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
            float _fmax_66 = fmaxf(m_state[0], _fmax_63 * softmax_scale_log2);
            float _fmax_67 = fmaxf(m_state[1], _fmax_65 * softmax_scale_log2);
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
            float _fma_0 = __fmaf_rn(s_acc[0], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_2 = approx_exp2(_fma_0);
            s_acc[0] = _exp2_2;
            float _fma_1 = __fmaf_rn(s_acc[1], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_3 = approx_exp2(_fma_1);
            s_acc[1] = _exp2_3;
            float _fma_2 = __fmaf_rn(s_acc[2], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_4 = approx_exp2(_fma_2);
            s_acc[2] = _exp2_4;
            float _fma_3 = __fmaf_rn(s_acc[3], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_5 = approx_exp2(_fma_3);
            s_acc[3] = _exp2_5;
            float _fma_4 = __fmaf_rn(s_acc[4], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_6 = approx_exp2(_fma_4);
            s_acc[4] = _exp2_6;
            float _fma_5 = __fmaf_rn(s_acc[5], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_7 = approx_exp2(_fma_5);
            s_acc[5] = _exp2_7;
            float _fma_6 = __fmaf_rn(s_acc[6], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_8 = approx_exp2(_fma_6);
            s_acc[6] = _exp2_8;
            float _fma_7 = __fmaf_rn(s_acc[7], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_9 = approx_exp2(_fma_7);
            s_acc[7] = _exp2_9;
            float _fma_8 = __fmaf_rn(s_acc[8], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_10 = approx_exp2(_fma_8);
            s_acc[8] = _exp2_10;
            float _fma_9 = __fmaf_rn(s_acc[9], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_11 = approx_exp2(_fma_9);
            s_acc[9] = _exp2_11;
            float _fma_10 = __fmaf_rn(s_acc[10], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_12 = approx_exp2(_fma_10);
            s_acc[10] = _exp2_12;
            float _fma_11 = __fmaf_rn(s_acc[11], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_13 = approx_exp2(_fma_11);
            s_acc[11] = _exp2_13;
            float _fma_12 = __fmaf_rn(s_acc[12], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_14 = approx_exp2(_fma_12);
            s_acc[12] = _exp2_14;
            float _fma_13 = __fmaf_rn(s_acc[13], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_15 = approx_exp2(_fma_13);
            s_acc[13] = _exp2_15;
            float _fma_14 = __fmaf_rn(s_acc[14], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_16 = approx_exp2(_fma_14);
            s_acc[14] = _exp2_16;
            float _fma_15 = __fmaf_rn(s_acc[15], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_17 = approx_exp2(_fma_15);
            s_acc[15] = _exp2_17;
            float _fma_16 = __fmaf_rn(s_acc[16], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_18 = approx_exp2(_fma_16);
            s_acc[16] = _exp2_18;
            float _fma_17 = __fmaf_rn(s_acc[17], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_19 = approx_exp2(_fma_17);
            s_acc[17] = _exp2_19;
            float _fma_18 = __fmaf_rn(s_acc[18], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_20 = approx_exp2(_fma_18);
            s_acc[18] = _exp2_20;
            float _fma_19 = __fmaf_rn(s_acc[19], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_21 = approx_exp2(_fma_19);
            s_acc[19] = _exp2_21;
            float _fma_20 = __fmaf_rn(s_acc[20], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_22 = approx_exp2(_fma_20);
            s_acc[20] = _exp2_22;
            float _fma_21 = __fmaf_rn(s_acc[21], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_23 = approx_exp2(_fma_21);
            s_acc[21] = _exp2_23;
            float _fma_22 = __fmaf_rn(s_acc[22], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_24 = approx_exp2(_fma_22);
            s_acc[22] = _exp2_24;
            float _fma_23 = __fmaf_rn(s_acc[23], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_25 = approx_exp2(_fma_23);
            s_acc[23] = _exp2_25;
            float _fma_24 = __fmaf_rn(s_acc[24], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_26 = approx_exp2(_fma_24);
            s_acc[24] = _exp2_26;
            float _fma_25 = __fmaf_rn(s_acc[25], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_27 = approx_exp2(_fma_25);
            s_acc[25] = _exp2_27;
            float _fma_26 = __fmaf_rn(s_acc[26], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_28 = approx_exp2(_fma_26);
            s_acc[26] = _exp2_28;
            float _fma_27 = __fmaf_rn(s_acc[27], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_29 = approx_exp2(_fma_27);
            s_acc[27] = _exp2_29;
            float _fma_28 = __fmaf_rn(s_acc[28], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_30 = approx_exp2(_fma_28);
            s_acc[28] = _exp2_30;
            float _fma_29 = __fmaf_rn(s_acc[29], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_31 = approx_exp2(_fma_29);
            s_acc[29] = _exp2_31;
            float _fma_30 = __fmaf_rn(s_acc[30], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_32 = approx_exp2(_fma_30);
            s_acc[30] = _exp2_32;
            float _fma_31 = __fmaf_rn(s_acc[31], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_33 = approx_exp2(_fma_31);
            s_acc[31] = _exp2_33;
            float _fma_32 = __fmaf_rn(s_acc[32], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_34 = approx_exp2(_fma_32);
            s_acc[32] = _exp2_34;
            float _fma_33 = __fmaf_rn(s_acc[33], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_35 = approx_exp2(_fma_33);
            s_acc[33] = _exp2_35;
            float _fma_34 = __fmaf_rn(s_acc[34], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_36 = approx_exp2(_fma_34);
            s_acc[34] = _exp2_36;
            float _fma_35 = __fmaf_rn(s_acc[35], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_37 = approx_exp2(_fma_35);
            s_acc[35] = _exp2_37;
            float _fma_36 = __fmaf_rn(s_acc[36], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_38 = approx_exp2(_fma_36);
            s_acc[36] = _exp2_38;
            float _fma_37 = __fmaf_rn(s_acc[37], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_39 = approx_exp2(_fma_37);
            s_acc[37] = _exp2_39;
            float _fma_38 = __fmaf_rn(s_acc[38], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_40 = approx_exp2(_fma_38);
            s_acc[38] = _exp2_40;
            float _fma_39 = __fmaf_rn(s_acc[39], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_41 = approx_exp2(_fma_39);
            s_acc[39] = _exp2_41;
            float _fma_40 = __fmaf_rn(s_acc[40], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_42 = approx_exp2(_fma_40);
            s_acc[40] = _exp2_42;
            float _fma_41 = __fmaf_rn(s_acc[41], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_43 = approx_exp2(_fma_41);
            s_acc[41] = _exp2_43;
            float _fma_42 = __fmaf_rn(s_acc[42], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_44 = approx_exp2(_fma_42);
            s_acc[42] = _exp2_44;
            float _fma_43 = __fmaf_rn(s_acc[43], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_45 = approx_exp2(_fma_43);
            s_acc[43] = _exp2_45;
            float _fma_44 = __fmaf_rn(s_acc[44], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_46 = approx_exp2(_fma_44);
            s_acc[44] = _exp2_46;
            float _fma_45 = __fmaf_rn(s_acc[45], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_47 = approx_exp2(_fma_45);
            s_acc[45] = _exp2_47;
            float _fma_46 = __fmaf_rn(s_acc[46], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_48 = approx_exp2(_fma_46);
            s_acc[46] = _exp2_48;
            float _fma_47 = __fmaf_rn(s_acc[47], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_49 = approx_exp2(_fma_47);
            s_acc[47] = _exp2_49;
            float _fma_48 = __fmaf_rn(s_acc[48], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_50 = approx_exp2(_fma_48);
            s_acc[48] = _exp2_50;
            float _fma_49 = __fmaf_rn(s_acc[49], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_51 = approx_exp2(_fma_49);
            s_acc[49] = _exp2_51;
            float _fma_50 = __fmaf_rn(s_acc[50], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_52 = approx_exp2(_fma_50);
            s_acc[50] = _exp2_52;
            float _fma_51 = __fmaf_rn(s_acc[51], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_53 = approx_exp2(_fma_51);
            s_acc[51] = _exp2_53;
            float _fma_52 = __fmaf_rn(s_acc[52], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_54 = approx_exp2(_fma_52);
            s_acc[52] = _exp2_54;
            float _fma_53 = __fmaf_rn(s_acc[53], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_55 = approx_exp2(_fma_53);
            s_acc[53] = _exp2_55;
            float _fma_54 = __fmaf_rn(s_acc[54], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_56 = approx_exp2(_fma_54);
            s_acc[54] = _exp2_56;
            float _fma_55 = __fmaf_rn(s_acc[55], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_57 = approx_exp2(_fma_55);
            s_acc[55] = _exp2_57;
            float _fma_56 = __fmaf_rn(s_acc[56], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_58 = approx_exp2(_fma_56);
            s_acc[56] = _exp2_58;
            float _fma_57 = __fmaf_rn(s_acc[57], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_59 = approx_exp2(_fma_57);
            s_acc[57] = _exp2_59;
            float _fma_58 = __fmaf_rn(s_acc[58], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_60 = approx_exp2(_fma_58);
            s_acc[58] = _exp2_60;
            float _fma_59 = __fmaf_rn(s_acc[59], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_61 = approx_exp2(_fma_59);
            s_acc[59] = _exp2_61;
            float _fma_60 = __fmaf_rn(s_acc[60], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_62 = approx_exp2(_fma_60);
            s_acc[60] = _exp2_62;
            float _fma_61 = __fmaf_rn(s_acc[61], softmax_scale_log2, 11.39231742277876f - _fmax_66);
            float _exp2_63 = approx_exp2(_fma_61);
            s_acc[61] = _exp2_63;
            float _fma_62 = __fmaf_rn(s_acc[62], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_64 = approx_exp2(_fma_62);
            s_acc[62] = _exp2_64;
            float _fma_63 = __fmaf_rn(s_acc[63], softmax_scale_log2, 11.39231742277876f - _fmax_67);
            float _exp2_65 = approx_exp2(_fma_63);
            s_acc[63] = _exp2_65;
            float _fma_64 = __fmaf_rn(l_state[0], _exp2_0, _exp2_2 + _exp2_3 + _exp2_6 + _exp2_7 + _exp2_10 + _exp2_11 + _exp2_14 + _exp2_15 + _exp2_18 + _exp2_19 + _exp2_22 + _exp2_23 + _exp2_26 + _exp2_27 + _exp2_30 + _exp2_31 + _exp2_34 + _exp2_35 + _exp2_38 + _exp2_39 + _exp2_42 + _exp2_43 + _exp2_46 + _exp2_47 + _exp2_50 + _exp2_51 + _exp2_54 + _exp2_55 + _exp2_58 + _exp2_59 + _exp2_62 + _exp2_63);
            l_state[0] = _fma_64;
            float _fma_65 = __fmaf_rn(l_state[1], _exp2_1, _exp2_4 + _exp2_5 + _exp2_8 + _exp2_9 + _exp2_12 + _exp2_13 + _exp2_16 + _exp2_17 + _exp2_20 + _exp2_21 + _exp2_24 + _exp2_25 + _exp2_28 + _exp2_29 + _exp2_32 + _exp2_33 + _exp2_36 + _exp2_37 + _exp2_40 + _exp2_41 + _exp2_44 + _exp2_45 + _exp2_48 + _exp2_49 + _exp2_52 + _exp2_53 + _exp2_56 + _exp2_57 + _exp2_60 + _exp2_61 + _exp2_64 + _exp2_65);
            l_state[1] = _fma_65;
            float _fmax_68 = fmaxf(s_acc[0], s_acc[1]);
            float _fmax_69 = fmaxf(s_acc[2], s_acc[3]);
            float _fmax_70 = fmaxf(s_acc[4], s_acc[5]);
            float _fmax_71 = fmaxf(_fmax_68, _fmax_70);
            float _fmax_72 = fmaxf(s_acc[6], s_acc[7]);
            float _fmax_73 = fmaxf(_fmax_69, _fmax_72);
            float _fmax_74 = fmaxf(s_acc[8], s_acc[9]);
            float _fmax_75 = fmaxf(_fmax_71, _fmax_74);
            float _fmax_76 = fmaxf(s_acc[10], s_acc[11]);
            float _fmax_77 = fmaxf(_fmax_73, _fmax_76);
            float _fmax_78 = fmaxf(s_acc[12], s_acc[13]);
            float _fmax_79 = fmaxf(_fmax_75, _fmax_78);
            float _fmax_80 = fmaxf(s_acc[14], s_acc[15]);
            float _fmax_81 = fmaxf(_fmax_77, _fmax_80);
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, _fmax_79, 1);
            float _fmax_82 = fmaxf(_fmax_79, _shfl_xor_4);
            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, _fmax_81, 1);
            float _fmax_83 = fmaxf(_fmax_81, _shfl_xor_5);
            uint16_t _e4m3x2_f32_0;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(_fmax_83 * 0.16666666666666666f), "f"(_fmax_82 * 0.16666666666666666f));
            float _rcp_0 = approx_rcp(_fmax_82);
            float _rcp_1 = approx_rcp(_fmax_83);
            s_acc[0] = s_acc[0] * ((_fmax_82 != 0.0f) ? _rcp_0 * 6.0f : 0.0f);
            s_acc[1] = s_acc[1] * ((_fmax_82 != 0.0f) ? _rcp_0 * 6.0f : 0.0f);
            s_acc[2] = s_acc[2] * ((_fmax_83 != 0.0f) ? _rcp_1 * 6.0f : 0.0f);
            s_acc[3] = s_acc[3] * ((_fmax_83 != 0.0f) ? _rcp_1 * 6.0f : 0.0f);
            s_acc[4] = s_acc[4] * ((_fmax_82 != 0.0f) ? _rcp_0 * 6.0f : 0.0f);
            s_acc[5] = s_acc[5] * ((_fmax_82 != 0.0f) ? _rcp_0 * 6.0f : 0.0f);
            s_acc[6] = s_acc[6] * ((_fmax_83 != 0.0f) ? _rcp_1 * 6.0f : 0.0f);
            s_acc[7] = s_acc[7] * ((_fmax_83 != 0.0f) ? _rcp_1 * 6.0f : 0.0f);
            s_acc[8] = s_acc[8] * ((_fmax_82 != 0.0f) ? _rcp_0 * 6.0f : 0.0f);
            s_acc[9] = s_acc[9] * ((_fmax_82 != 0.0f) ? _rcp_0 * 6.0f : 0.0f);
            s_acc[10] = s_acc[10] * ((_fmax_83 != 0.0f) ? _rcp_1 * 6.0f : 0.0f);
            s_acc[11] = s_acc[11] * ((_fmax_83 != 0.0f) ? _rcp_1 * 6.0f : 0.0f);
            s_acc[12] = s_acc[12] * ((_fmax_82 != 0.0f) ? _rcp_0 * 6.0f : 0.0f);
            s_acc[13] = s_acc[13] * ((_fmax_82 != 0.0f) ? _rcp_0 * 6.0f : 0.0f);
            s_acc[14] = s_acc[14] * ((_fmax_83 != 0.0f) ? _rcp_1 * 6.0f : 0.0f);
            s_acc[15] = s_acc[15] * ((_fmax_83 != 0.0f) ? _rcp_1 * 6.0f : 0.0f);
            float _fmax_84 = fmaxf(s_acc[16], s_acc[17]);
            float _fmax_85 = fmaxf(s_acc[18], s_acc[19]);
            float _fmax_86 = fmaxf(s_acc[20], s_acc[21]);
            float _fmax_87 = fmaxf(_fmax_84, _fmax_86);
            float _fmax_88 = fmaxf(s_acc[22], s_acc[23]);
            float _fmax_89 = fmaxf(_fmax_85, _fmax_88);
            float _fmax_90 = fmaxf(s_acc[24], s_acc[25]);
            float _fmax_91 = fmaxf(_fmax_87, _fmax_90);
            float _fmax_92 = fmaxf(s_acc[26], s_acc[27]);
            float _fmax_93 = fmaxf(_fmax_89, _fmax_92);
            float _fmax_94 = fmaxf(s_acc[28], s_acc[29]);
            float _fmax_95 = fmaxf(_fmax_91, _fmax_94);
            float _fmax_96 = fmaxf(s_acc[30], s_acc[31]);
            float _fmax_97 = fmaxf(_fmax_93, _fmax_96);
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, _fmax_95, 1);
            float _fmax_98 = fmaxf(_fmax_95, _shfl_xor_6);
            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, _fmax_97, 1);
            float _fmax_99 = fmaxf(_fmax_97, _shfl_xor_7);
            uint16_t _e4m3x2_f32_1;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(_fmax_99 * 0.16666666666666666f), "f"(_fmax_98 * 0.16666666666666666f));
            float _rcp_2 = approx_rcp(_fmax_98);
            float _rcp_3 = approx_rcp(_fmax_99);
            s_acc[16] = s_acc[16] * ((_fmax_98 != 0.0f) ? _rcp_2 * 6.0f : 0.0f);
            s_acc[17] = s_acc[17] * ((_fmax_98 != 0.0f) ? _rcp_2 * 6.0f : 0.0f);
            s_acc[18] = s_acc[18] * ((_fmax_99 != 0.0f) ? _rcp_3 * 6.0f : 0.0f);
            s_acc[19] = s_acc[19] * ((_fmax_99 != 0.0f) ? _rcp_3 * 6.0f : 0.0f);
            s_acc[20] = s_acc[20] * ((_fmax_98 != 0.0f) ? _rcp_2 * 6.0f : 0.0f);
            s_acc[21] = s_acc[21] * ((_fmax_98 != 0.0f) ? _rcp_2 * 6.0f : 0.0f);
            s_acc[22] = s_acc[22] * ((_fmax_99 != 0.0f) ? _rcp_3 * 6.0f : 0.0f);
            s_acc[23] = s_acc[23] * ((_fmax_99 != 0.0f) ? _rcp_3 * 6.0f : 0.0f);
            s_acc[24] = s_acc[24] * ((_fmax_98 != 0.0f) ? _rcp_2 * 6.0f : 0.0f);
            s_acc[25] = s_acc[25] * ((_fmax_98 != 0.0f) ? _rcp_2 * 6.0f : 0.0f);
            s_acc[26] = s_acc[26] * ((_fmax_99 != 0.0f) ? _rcp_3 * 6.0f : 0.0f);
            s_acc[27] = s_acc[27] * ((_fmax_99 != 0.0f) ? _rcp_3 * 6.0f : 0.0f);
            s_acc[28] = s_acc[28] * ((_fmax_98 != 0.0f) ? _rcp_2 * 6.0f : 0.0f);
            s_acc[29] = s_acc[29] * ((_fmax_98 != 0.0f) ? _rcp_2 * 6.0f : 0.0f);
            s_acc[30] = s_acc[30] * ((_fmax_99 != 0.0f) ? _rcp_3 * 6.0f : 0.0f);
            s_acc[31] = s_acc[31] * ((_fmax_99 != 0.0f) ? _rcp_3 * 6.0f : 0.0f);
            unsigned int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 16), 2);
            psf[0] = (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 16) | _shfl_xor_8;
            p_tmp[0] = s_acc[0];
            p_tmp[1] = s_acc[1];
            p_tmp[2] = s_acc[4];
            p_tmp[3] = s_acc[5];
            p_tmp[4] = s_acc[8];
            p_tmp[5] = s_acc[9];
            p_tmp[6] = s_acc[12];
            p_tmp[7] = s_acc[13];
            p_tmp[8] = s_acc[2];
            p_tmp[9] = s_acc[3];
            p_tmp[10] = s_acc[6];
            p_tmp[11] = s_acc[7];
            p_tmp[12] = s_acc[10];
            p_tmp[13] = s_acc[11];
            p_tmp[14] = s_acc[14];
            p_tmp[15] = s_acc[15];
            p_tmp[16] = s_acc[16];
            p_tmp[17] = s_acc[17];
            p_tmp[18] = s_acc[20];
            p_tmp[19] = s_acc[21];
            p_tmp[20] = s_acc[24];
            p_tmp[21] = s_acc[25];
            p_tmp[22] = s_acc[28];
            p_tmp[23] = s_acc[29];
            p_tmp[24] = s_acc[18];
            p_tmp[25] = s_acc[19];
            p_tmp[26] = s_acc[22];
            p_tmp[27] = s_acc[23];
            p_tmp[28] = s_acc[26];
            p_tmp[29] = s_acc[27];
            p_tmp[30] = s_acc[30];
            p_tmp[31] = s_acc[31];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[0]) : "f"(p_tmp[0]), "f"(p_tmp[1]), "f"(p_tmp[2]), "f"(p_tmp[3]), "f"(p_tmp[4]), "f"(p_tmp[5]), "f"(p_tmp[6]), "f"(p_tmp[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[1]) : "f"(p_tmp[8]), "f"(p_tmp[9]), "f"(p_tmp[10]), "f"(p_tmp[11]), "f"(p_tmp[12]), "f"(p_tmp[13]), "f"(p_tmp[14]), "f"(p_tmp[15]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[2]) : "f"(p_tmp[16]), "f"(p_tmp[17]), "f"(p_tmp[18]), "f"(p_tmp[19]), "f"(p_tmp[20]), "f"(p_tmp[21]), "f"(p_tmp[22]), "f"(p_tmp[23]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[3]) : "f"(p_tmp[24]), "f"(p_tmp[25]), "f"(p_tmp[26]), "f"(p_tmp[27]), "f"(p_tmp[28]), "f"(p_tmp[29]), "f"(p_tmp[30]), "f"(p_tmp[31]));
            float _fmax_100 = fmaxf(s_acc[32], s_acc[33]);
            float _fmax_101 = fmaxf(s_acc[34], s_acc[35]);
            float _fmax_102 = fmaxf(s_acc[36], s_acc[37]);
            float _fmax_103 = fmaxf(_fmax_100, _fmax_102);
            float _fmax_104 = fmaxf(s_acc[38], s_acc[39]);
            float _fmax_105 = fmaxf(_fmax_101, _fmax_104);
            float _fmax_106 = fmaxf(s_acc[40], s_acc[41]);
            float _fmax_107 = fmaxf(_fmax_103, _fmax_106);
            float _fmax_108 = fmaxf(s_acc[42], s_acc[43]);
            float _fmax_109 = fmaxf(_fmax_105, _fmax_108);
            float _fmax_110 = fmaxf(s_acc[44], s_acc[45]);
            float _fmax_111 = fmaxf(_fmax_107, _fmax_110);
            float _fmax_112 = fmaxf(s_acc[46], s_acc[47]);
            float _fmax_113 = fmaxf(_fmax_109, _fmax_112);
            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, _fmax_111, 1);
            float _fmax_114 = fmaxf(_fmax_111, _shfl_xor_9);
            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, _fmax_113, 1);
            float _fmax_115 = fmaxf(_fmax_113, _shfl_xor_10);
            uint16_t _e4m3x2_f32_2;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(_fmax_115 * 0.16666666666666666f), "f"(_fmax_114 * 0.16666666666666666f));
            float _rcp_4 = approx_rcp(_fmax_114);
            float _rcp_5 = approx_rcp(_fmax_115);
            s_acc[32] = s_acc[32] * ((_fmax_114 != 0.0f) ? _rcp_4 * 6.0f : 0.0f);
            s_acc[33] = s_acc[33] * ((_fmax_114 != 0.0f) ? _rcp_4 * 6.0f : 0.0f);
            s_acc[34] = s_acc[34] * ((_fmax_115 != 0.0f) ? _rcp_5 * 6.0f : 0.0f);
            s_acc[35] = s_acc[35] * ((_fmax_115 != 0.0f) ? _rcp_5 * 6.0f : 0.0f);
            s_acc[36] = s_acc[36] * ((_fmax_114 != 0.0f) ? _rcp_4 * 6.0f : 0.0f);
            s_acc[37] = s_acc[37] * ((_fmax_114 != 0.0f) ? _rcp_4 * 6.0f : 0.0f);
            s_acc[38] = s_acc[38] * ((_fmax_115 != 0.0f) ? _rcp_5 * 6.0f : 0.0f);
            s_acc[39] = s_acc[39] * ((_fmax_115 != 0.0f) ? _rcp_5 * 6.0f : 0.0f);
            s_acc[40] = s_acc[40] * ((_fmax_114 != 0.0f) ? _rcp_4 * 6.0f : 0.0f);
            s_acc[41] = s_acc[41] * ((_fmax_114 != 0.0f) ? _rcp_4 * 6.0f : 0.0f);
            s_acc[42] = s_acc[42] * ((_fmax_115 != 0.0f) ? _rcp_5 * 6.0f : 0.0f);
            s_acc[43] = s_acc[43] * ((_fmax_115 != 0.0f) ? _rcp_5 * 6.0f : 0.0f);
            s_acc[44] = s_acc[44] * ((_fmax_114 != 0.0f) ? _rcp_4 * 6.0f : 0.0f);
            s_acc[45] = s_acc[45] * ((_fmax_114 != 0.0f) ? _rcp_4 * 6.0f : 0.0f);
            s_acc[46] = s_acc[46] * ((_fmax_115 != 0.0f) ? _rcp_5 * 6.0f : 0.0f);
            s_acc[47] = s_acc[47] * ((_fmax_115 != 0.0f) ? _rcp_5 * 6.0f : 0.0f);
            float _fmax_116 = fmaxf(s_acc[48], s_acc[49]);
            float _fmax_117 = fmaxf(s_acc[50], s_acc[51]);
            float _fmax_118 = fmaxf(s_acc[52], s_acc[53]);
            float _fmax_119 = fmaxf(_fmax_116, _fmax_118);
            float _fmax_120 = fmaxf(s_acc[54], s_acc[55]);
            float _fmax_121 = fmaxf(_fmax_117, _fmax_120);
            float _fmax_122 = fmaxf(s_acc[56], s_acc[57]);
            float _fmax_123 = fmaxf(_fmax_119, _fmax_122);
            float _fmax_124 = fmaxf(s_acc[58], s_acc[59]);
            float _fmax_125 = fmaxf(_fmax_121, _fmax_124);
            float _fmax_126 = fmaxf(s_acc[60], s_acc[61]);
            float _fmax_127 = fmaxf(_fmax_123, _fmax_126);
            float _fmax_128 = fmaxf(s_acc[62], s_acc[63]);
            float _fmax_129 = fmaxf(_fmax_125, _fmax_128);
            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, _fmax_127, 1);
            float _fmax_130 = fmaxf(_fmax_127, _shfl_xor_11);
            float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, _fmax_129, 1);
            float _fmax_131 = fmaxf(_fmax_129, _shfl_xor_12);
            uint16_t _e4m3x2_f32_3;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(_fmax_131 * 0.16666666666666666f), "f"(_fmax_130 * 0.16666666666666666f));
            float _rcp_6 = approx_rcp(_fmax_130);
            float _rcp_7 = approx_rcp(_fmax_131);
            s_acc[48] = s_acc[48] * ((_fmax_130 != 0.0f) ? _rcp_6 * 6.0f : 0.0f);
            s_acc[49] = s_acc[49] * ((_fmax_130 != 0.0f) ? _rcp_6 * 6.0f : 0.0f);
            s_acc[50] = s_acc[50] * ((_fmax_131 != 0.0f) ? _rcp_7 * 6.0f : 0.0f);
            s_acc[51] = s_acc[51] * ((_fmax_131 != 0.0f) ? _rcp_7 * 6.0f : 0.0f);
            s_acc[52] = s_acc[52] * ((_fmax_130 != 0.0f) ? _rcp_6 * 6.0f : 0.0f);
            s_acc[53] = s_acc[53] * ((_fmax_130 != 0.0f) ? _rcp_6 * 6.0f : 0.0f);
            s_acc[54] = s_acc[54] * ((_fmax_131 != 0.0f) ? _rcp_7 * 6.0f : 0.0f);
            s_acc[55] = s_acc[55] * ((_fmax_131 != 0.0f) ? _rcp_7 * 6.0f : 0.0f);
            s_acc[56] = s_acc[56] * ((_fmax_130 != 0.0f) ? _rcp_6 * 6.0f : 0.0f);
            s_acc[57] = s_acc[57] * ((_fmax_130 != 0.0f) ? _rcp_6 * 6.0f : 0.0f);
            s_acc[58] = s_acc[58] * ((_fmax_131 != 0.0f) ? _rcp_7 * 6.0f : 0.0f);
            s_acc[59] = s_acc[59] * ((_fmax_131 != 0.0f) ? _rcp_7 * 6.0f : 0.0f);
            s_acc[60] = s_acc[60] * ((_fmax_130 != 0.0f) ? _rcp_6 * 6.0f : 0.0f);
            s_acc[61] = s_acc[61] * ((_fmax_130 != 0.0f) ? _rcp_6 * 6.0f : 0.0f);
            s_acc[62] = s_acc[62] * ((_fmax_131 != 0.0f) ? _rcp_7 * 6.0f : 0.0f);
            s_acc[63] = s_acc[63] * ((_fmax_131 != 0.0f) ? _rcp_7 * 6.0f : 0.0f);
            unsigned int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_2 >> 8 : (unsigned int)_e4m3x2_f32_2) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_2 >> 8 : (unsigned int)_e4m3x2_f32_2) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_3 >> 8 : (unsigned int)_e4m3x2_f32_3) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_3 >> 8 : (unsigned int)_e4m3x2_f32_3) & 255) << 16), 2);
            psf[1] = (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_2 >> 8 : (unsigned int)_e4m3x2_f32_2) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_2 >> 8 : (unsigned int)_e4m3x2_f32_2) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_3 >> 8 : (unsigned int)_e4m3x2_f32_3) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_3 >> 8 : (unsigned int)_e4m3x2_f32_3) & 255) << 16) | _shfl_xor_13;
            p_tmp[0] = s_acc[32];
            p_tmp[1] = s_acc[33];
            p_tmp[2] = s_acc[36];
            p_tmp[3] = s_acc[37];
            p_tmp[4] = s_acc[40];
            p_tmp[5] = s_acc[41];
            p_tmp[6] = s_acc[44];
            p_tmp[7] = s_acc[45];
            p_tmp[8] = s_acc[34];
            p_tmp[9] = s_acc[35];
            p_tmp[10] = s_acc[38];
            p_tmp[11] = s_acc[39];
            p_tmp[12] = s_acc[42];
            p_tmp[13] = s_acc[43];
            p_tmp[14] = s_acc[46];
            p_tmp[15] = s_acc[47];
            p_tmp[16] = s_acc[48];
            p_tmp[17] = s_acc[49];
            p_tmp[18] = s_acc[52];
            p_tmp[19] = s_acc[53];
            p_tmp[20] = s_acc[56];
            p_tmp[21] = s_acc[57];
            p_tmp[22] = s_acc[60];
            p_tmp[23] = s_acc[61];
            p_tmp[24] = s_acc[50];
            p_tmp[25] = s_acc[51];
            p_tmp[26] = s_acc[54];
            p_tmp[27] = s_acc[55];
            p_tmp[28] = s_acc[58];
            p_tmp[29] = s_acc[59];
            p_tmp[30] = s_acc[62];
            p_tmp[31] = s_acc[63];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[4]) : "f"(p_tmp[0]), "f"(p_tmp[1]), "f"(p_tmp[2]), "f"(p_tmp[3]), "f"(p_tmp[4]), "f"(p_tmp[5]), "f"(p_tmp[6]), "f"(p_tmp[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[5]) : "f"(p_tmp[8]), "f"(p_tmp[9]), "f"(p_tmp[10]), "f"(p_tmp[11]), "f"(p_tmp[12]), "f"(p_tmp[13]), "f"(p_tmp[14]), "f"(p_tmp[15]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[6]) : "f"(p_tmp[16]), "f"(p_tmp[17]), "f"(p_tmp[18]), "f"(p_tmp[19]), "f"(p_tmp[20]), "f"(p_tmp[21]), "f"(p_tmp[22]), "f"(p_tmp[23]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[7]) : "f"(p_tmp[24]), "f"(p_tmp[25]), "f"(p_tmp[26]), "f"(p_tmp[27]), "f"(p_tmp[28]), "f"(p_tmp[29]), "f"(p_tmp[30]), "f"(p_tmp[31]));
            mbarrier_wait(v_full_addr + (kv_stage) * 8, kv_phase);
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _VSF_stage_reg_0[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_0[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((lane >> 2) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_0[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _VSF_stage_reg_1[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_1[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _VSF_stage_reg_2[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_2[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_2[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _VSF_stage_reg_3[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_3[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_3[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _VSF_stage_reg_4[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_4[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((32 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_4[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _VSF_stage_reg_5[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_5[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((40 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_5[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _VSF_stage_reg_6[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_6[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((48 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_6[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _VSF_stage_reg_7[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_7[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((56 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_7[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _VSF_stage_reg_8[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_8[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((64 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_8[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _VSF_stage_reg_9[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_9[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((72 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_9[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _VSF_stage_reg_10[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_10[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((80 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_10[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _VSF_stage_reg_11[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_11[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((88 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_11[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _VSF_stage_reg_12[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_12[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((96 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_12[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _VSF_stage_reg_13[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_13[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((104 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_13[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int _VSF_stage_reg_14[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_14[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((112 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_14[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int _VSF_stage_reg_15[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(VSF_stage);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _VSF_stage_reg_15[_lr] = _smem_ptr[(kv_stage * 256 + (unsigned int)((120 + (lane >> 2)) * 2)) + _lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_VSF_stage_reg_15[1])), "h"((uint16_t)0), "h"((uint16_t)0));
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
                        mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 9216);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(VT_stage_addr + kv_stage * 8192), "l"((&VT_map)), "r"((kb + 2) * 64), "r"(0), "r"(head),
                               "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(VSF_stage_addr + kv_stage * 1024), "l"((&VSF_map)), "r"(0), "r"((head * num_kblocks + (kb + 2)) * 64),
                               "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            kv_stage += 1;
            if (kv_stage == 2) { kv_stage = 0; kv_phase ^= 1; }
        }
        float l0 = l_state[0];
        float l1 = l_state[1];
        float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, l0, 1);
        l0 = l0 + _shfl_xor_14;
        float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, l0, 2);
        l0 = l0 + _shfl_xor_15;
        float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, l1, 1);
        l1 = l1 + _shfl_xor_16;
        float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, l1, 2);
        l1 = l1 + _shfl_xor_17;
        float _rcp_8 = approx_rcp(l0);
        float inv0 = _rcp_8;
        float _rcp_9 = approx_rcp(l1);
        float inv1 = _rcp_9;
        o_tmp[0] = o_acc[0] * inv0;
        o_tmp[1] = o_acc[1] * inv0;
        o_tmp[2] = o_acc[2] * inv1;
        o_tmp[3] = o_acc[3] * inv1;
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
        o_tmp[0] = o_acc[4] * inv0;
        o_tmp[1] = o_acc[5] * inv0;
        o_tmp[2] = o_acc[6] * inv1;
        o_tmp[3] = o_acc[7] * inv1;
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
        o_tmp[0] = o_acc[8] * inv0;
        o_tmp[1] = o_acc[9] * inv0;
        o_tmp[2] = o_acc[10] * inv1;
        o_tmp[3] = o_acc[11] * inv1;
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
        o_tmp[0] = o_acc[12] * inv0;
        o_tmp[1] = o_acc[13] * inv0;
        o_tmp[2] = o_acc[14] * inv1;
        o_tmp[3] = o_acc[15] * inv1;
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
        o_tmp[0] = o_acc[16] * inv0;
        o_tmp[1] = o_acc[17] * inv0;
        o_tmp[2] = o_acc[18] * inv1;
        o_tmp[3] = o_acc[19] * inv1;
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
        o_tmp[0] = o_acc[20] * inv0;
        o_tmp[1] = o_acc[21] * inv0;
        o_tmp[2] = o_acc[22] * inv1;
        o_tmp[3] = o_acc[23] * inv1;
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
        o_tmp[0] = o_acc[24] * inv0;
        o_tmp[1] = o_acc[25] * inv0;
        o_tmp[2] = o_acc[26] * inv1;
        o_tmp[3] = o_acc[27] * inv1;
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
        o_tmp[0] = o_acc[28] * inv0;
        o_tmp[1] = o_acc[29] * inv0;
        o_tmp[2] = o_acc[30] * inv1;
        o_tmp[3] = o_acc[31] * inv1;
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
        o_tmp[0] = o_acc[32] * inv0;
        o_tmp[1] = o_acc[33] * inv0;
        o_tmp[2] = o_acc[34] * inv1;
        o_tmp[3] = o_acc[35] * inv1;
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
        o_tmp[0] = o_acc[36] * inv0;
        o_tmp[1] = o_acc[37] * inv0;
        o_tmp[2] = o_acc[38] * inv1;
        o_tmp[3] = o_acc[39] * inv1;
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
        o_tmp[0] = o_acc[40] * inv0;
        o_tmp[1] = o_acc[41] * inv0;
        o_tmp[2] = o_acc[42] * inv1;
        o_tmp[3] = o_acc[43] * inv1;
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
        o_tmp[0] = o_acc[44] * inv0;
        o_tmp[1] = o_acc[45] * inv0;
        o_tmp[2] = o_acc[46] * inv1;
        o_tmp[3] = o_acc[47] * inv1;
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
        o_tmp[0] = o_acc[48] * inv0;
        o_tmp[1] = o_acc[49] * inv0;
        o_tmp[2] = o_acc[50] * inv1;
        o_tmp[3] = o_acc[51] * inv1;
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
        o_tmp[0] = o_acc[52] * inv0;
        o_tmp[1] = o_acc[53] * inv0;
        o_tmp[2] = o_acc[54] * inv1;
        o_tmp[3] = o_acc[55] * inv1;
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
        o_tmp[0] = o_acc[56] * inv0;
        o_tmp[1] = o_acc[57] * inv0;
        o_tmp[2] = o_acc[58] * inv1;
        o_tmp[3] = o_acc[59] * inv1;
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
        o_tmp[0] = o_acc[60] * inv0;
        o_tmp[1] = o_acc[61] * inv0;
        o_tmp[2] = o_acc[62] * inv1;
        o_tmp[3] = o_acc[63] * inv1;
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

}  // namespace h3_varlen_attention_sage_sm120a
#undef H3_VARLEN_INF
#undef NUM_DELTA_STAGES
#undef NUM_KV_STAGES
#undef SMEM_DELTA_OFF
#undef SMEM_DELTA_STAGE_BYTES
#undef SMEM_DELTA_STRIDE
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
#undef SMEM_VSF_STAGE_OFF
#undef SMEM_VSF_STAGE_STAGE_BYTES
#undef SMEM_VSF_STAGE_STRIDE
#undef SMEM_VT_STAGE_OFF
#undef SMEM_VT_STAGE_STAGE_BYTES
#undef SMEM_VT_STAGE_STRIDE
#undef THREADS
#undef delta_ready_addr
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
constexpr int64_t kBlockM = 128;
constexpr int64_t kBlockN = 128;
constexpr int64_t kMaxHeads = 32768;
constexpr int kStatsThreads = 256;
constexpr int kQuantThreads = 256;
constexpr int kAttentionThreads = 256;
constexpr int kStatsSmemBytes = 16512;
constexpr int kQuantSmemBytes = 0;
constexpr int64_t kQuantTokens = 32;      // tokens per quantizer CTA
constexpr int kAttentionSmemBytes = 48128;  // Q tile + two-stage K / K-scale / V^T / V-scale ring + mbarriers
constexpr int64_t kStatsCtasPerSm = 4;
constexpr int64_t kQuantCtasPerSm = 8;
constexpr int64_t kRowBytes4 = kHeadDim / 2;          // packed E2M1 bytes per (token, head) row
constexpr int64_t kBlocksPerRow = kHeadDim / 16;      // UE4M3 block scales per row
constexpr int64_t kScaleTileBytes = kBlockN * kBlocksPerRow;  // 1 KiB K- or V-scale tile per (head, key block)
constexpr uint32_t kScaleTileRows16 = static_cast<uint32_t>(kScaleTileBytes / 16);
// Statistics partials: per (statistics tile, head) 128 K channel sums.
constexpr int64_t kPartialFloats = kHeadDim;

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
  status = cudaFuncSetAttribute(h3_varlen_attention_sage_sm120a::kernel_minimax_h3_sm120_varlen_attention_sage,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, kAttentionSmemBytes);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to opt in to dynamic shared memory: " << cudaGetErrorString(status);
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

// The K / V block-scale tiles: one contiguous 1 KiB tile (128 keys x 8 UE4M3, or 128 channels x 8 UE4M3)
// per (head, key block), addressed as the 2-D view (16 bytes, heads * num_kblocks * 64 rows) without swizzle.
CUtensorMap EncodeScaleTiles(const TensorView& sf, int64_t heads, int64_t num_kblocks, const char* name) {
  uint64_t global_dim[2] = {16, static_cast<uint64_t>(heads * num_kblocks * kScaleTileRows16)};
  uint64_t global_strides[1] = {16};
  uint32_t box_dim[2] = {16, kScaleTileRows16};
  uint32_t element_strides[2] = {1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, sf.data_ptr(), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode the " << name << " tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

// One head's [128 channel rows x 64 packed-E2M1 bytes (128 keys)] tile of the transposed V^T byte tensor
// [heads, 128, padded_tokens / 2]: 3-D view (padded_tokens / 2 bytes, channels, heads), 64-byte swizzle.
CUtensorMap EncodeTransposedTile(const TensorView& vt, int64_t padded_tokens, int64_t heads, const char* name) {
  const int64_t row_bytes = padded_tokens / 2;
  uint64_t global_dim[3] = {static_cast<uint64_t>(row_bytes), static_cast<uint64_t>(kHeadDim),
                            static_cast<uint64_t>(heads)};
  uint64_t global_strides[2] = {static_cast<uint64_t>(row_bytes), static_cast<uint64_t>(row_bytes * kHeadDim)};
  uint32_t box_dim[3] = {static_cast<uint32_t>(kRowBytes4), static_cast<uint32_t>(kHeadDim), 1};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, vt.data_ptr(), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode the " << name << " tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

}  // namespace

// out[a:b, h] = softmax(q[a:b, h] k[a:b, h]^T * softmax_scale) v[a:b, h] for every segment [a, b) of
// cu_seqlens and every head h, with the SageAttention3 FP4 recipe: E2M1 Q / K / V / P operands with UE4M3
// block scales (16 elements), per-segment K mean and per-128-row Q block mean removal with in-kernel
// compensation, and an FP32 softmax.
//   q, k, v, out: contiguous BF16 [tokens, heads, 128].
//   Plan (built on the host from cu_seqlens): tok_seg int32 [padded_tokens] (segment of every token, padded
//   tokens repeat the last), cu_seqlens int32 [segments + 1], seg_tile_begin int32 [segments + 1] (first Q
//   block of every segment, cumulative), tile_table int32 [8 * num_tiles] (segment, row_begin, row_end,
//   first tile of the segment, end tile, segment length, 0, 0), unit_table int32 [4 * num_units] (segment,
//   head << 16 | q_tile, segment begin, segment end) in persistent-grid slot order.
//   Workspaces: q4, k4 uint8 [>= tokens * heads * 64], q_sf uint8 [>= tokens * heads * 8], k_sf and v_sf uint8
//   [>= heads * num_kblocks * 1024], qm4 uint8 [>= num_qtiles * heads * 64], qm_sf uint8 [>= num_qtiles * heads * 8],
//   vt4 uint8 [>= heads * 128 * padded_tokens / 2], q_mean f32 [>= num_qtiles * heads * 128], mean_k f32
//   [>= segments * heads * 128], partials f32 [>= num_tiles * heads * 128], counters uint32 [>= segments * heads]
//   (zero before the first call; the kernel resets them).
void minimax_h3_sm120_varlen_attention_nvfp4(TensorView q, TensorView k, TensorView v, TensorView tok_seg,
                                             TensorView cu_seqlens, TensorView seg_tile_begin, TensorView out,
                                             TensorView tile_table, TensorView unit_table, TensorView q4,
                                             TensorView k4, TensorView q_sf, TensorView k_sf, TensorView qm4,
                                             TensorView qm_sf, TensorView vt4, TensorView v_sf, TensorView q_mean,
                                             TensorView mean_k, TensorView partials, TensorView counters,
                                             int64_t num_segments, int64_t num_tiles, int64_t num_qtiles,
                                             int64_t num_units, int64_t attention_grid, double softmax_scale) {
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
  TVM_FFI_CHECK(num_segments >= 1 && num_tiles >= 0 && num_qtiles >= 0 && num_units >= 0 && attention_grid >= 1,
                ValueError)
      << "invalid segment plan";
  const int64_t num_kblocks = std::max<int64_t>(1, (tokens + kBlockN - 1) / kBlockN);
  const int64_t padded_tokens = num_kblocks * kBlockN;
  const int64_t qtiles = std::max<int64_t>(1, num_qtiles);
  CheckFlat(tok_seg, "tok_seg", dl_int32, "int32", std::max<int64_t>(1, padded_tokens), device);
  CheckFlat(cu_seqlens, "cu_seqlens", dl_int32, "int32", num_segments + 1, device);
  CheckFlat(seg_tile_begin, "seg_tile_begin", dl_int32, "int32", num_segments + 1, device);
  CheckFlat(tile_table, "tile_table", dl_int32, "int32", std::max<int64_t>(kTileRowInts, kTileRowInts * num_tiles), device);
  CheckFlat(unit_table, "unit_table", dl_int32, "int32", std::max<int64_t>(kUnitRowInts, kUnitRowInts * num_units), device);
  CheckFlat(q4, "q4", dl_uint8, "uint8", tokens * heads * kRowBytes4, device);
  CheckFlat(k4, "k4", dl_uint8, "uint8", tokens * heads * kRowBytes4, device);
  CheckFlat(q_sf, "q_sf", dl_uint8, "uint8", tokens * heads * kBlocksPerRow, device);
  CheckFlat(k_sf, "k_sf", dl_uint8, "uint8", heads * num_kblocks * kScaleTileBytes, device);
  CheckFlat(qm4, "qm4", dl_uint8, "uint8", qtiles * heads * kRowBytes4, device);
  CheckFlat(qm_sf, "qm_sf", dl_uint8, "uint8", qtiles * heads * kBlocksPerRow, device);
  CheckFlat(vt4, "vt4", dl_uint8, "uint8", heads * kHeadDim * (padded_tokens / 2), device);
  CheckFlat(v_sf, "v_sf", dl_uint8, "uint8", heads * num_kblocks * kScaleTileBytes, device);
  CheckFlat(q_mean, "q_mean", dl_float32, "float32", qtiles * heads * kHeadDim, device);
  CheckFlat(mean_k, "mean_k", dl_float32, "float32", num_segments * heads * kHeadDim, device);
  CheckFlat(partials, "partials", dl_float32, "float32", std::max<int64_t>(1, num_tiles * heads * kPartialFloats), device);
  CheckFlat(counters, "counters", dl_uint32, "uint32", std::max<int64_t>(1, num_segments) * heads, device);

  ffi::CUDADeviceGuard device_guard(device.device_id);
  const cudaStream_t stream = get_stream(device);
  const DeviceInfo info = ConfigureKernels();
  if (tokens == 0 || num_units == 0) return;  // no rows to write (all segments empty)
  const int heads_i = static_cast<int>(heads);

  if (num_tiles > 0) {
    const int stats_grid = static_cast<int>(
        std::max<int64_t>(1, std::min<int64_t>(num_tiles * heads, kStatsCtasPerSm * static_cast<int64_t>(info.num_sms))));
    LaunchPdl(&h3_varlen_stats_sage_sm120a::kernel_minimax_h3_sm120_varlen_stats_sage, stats_grid, kStatsThreads,
        kStatsSmemBytes, stream,
        static_cast<__nv_bfloat16*>(q.data_ptr()), static_cast<__nv_bfloat16*>(k.data_ptr()),
        static_cast<int*>(tile_table.data_ptr()), static_cast<int*>(seg_tile_begin.data_ptr()),
        static_cast<float*>(partials.data_ptr()), static_cast<unsigned int*>(counters.data_ptr()),
        static_cast<float*>(mean_k.data_ptr()), static_cast<float*>(q_mean.data_ptr()), static_cast<int>(num_tiles),
        heads_i);
  }
  const int quant_grid = static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>((padded_tokens / kQuantTokens) * heads, kQuantCtasPerSm * static_cast<int64_t>(info.num_sms))));
  LaunchPdl(&h3_varlen_quantize_sage_sm120a::kernel_minimax_h3_sm120_varlen_quantize_sage, quant_grid, kQuantThreads,
      kQuantSmemBytes, stream,
      static_cast<__nv_bfloat16*>(q.data_ptr()), static_cast<__nv_bfloat16*>(k.data_ptr()),
      static_cast<__nv_bfloat16*>(v.data_ptr()), static_cast<int*>(tok_seg.data_ptr()),
      static_cast<int*>(cu_seqlens.data_ptr()), static_cast<int*>(seg_tile_begin.data_ptr()),
      static_cast<float*>(mean_k.data_ptr()), static_cast<float*>(q_mean.data_ptr()),
      static_cast<unsigned int*>(q4.data_ptr()), static_cast<unsigned int*>(k4.data_ptr()),
      static_cast<uint8_t*>(q_sf.data_ptr()), static_cast<uint8_t*>(k_sf.data_ptr()),
      static_cast<unsigned int*>(qm4.data_ptr()), static_cast<uint8_t*>(qm_sf.data_ptr()),
      static_cast<unsigned int*>(vt4.data_ptr()), static_cast<uint8_t*>(v_sf.data_ptr()),
      static_cast<int>(tokens), static_cast<int>(padded_tokens), heads_i, static_cast<int>(num_kblocks));

  const CUtensorMap q_map = EncodeRowsTile(q4, tokens, heads, static_cast<uint32_t>(kBlockM), "q4");
  const CUtensorMap k_map = EncodeRowsTile(k4, tokens, heads, static_cast<uint32_t>(kBlockN), "k4");
  const CUtensorMap ksf_map = EncodeScaleTiles(k_sf, heads, num_kblocks, "k_sf");
  const CUtensorMap vt_map = EncodeTransposedTile(vt4, padded_tokens, heads, "vt4");
  const CUtensorMap vsf_map = EncodeScaleTiles(v_sf, heads, num_kblocks, "v_sf");
  const int grid = static_cast<int>(std::min<int64_t>(attention_grid, std::max<int64_t>(1, num_units)));
  const float softmax_scale_log2 = static_cast<float>(softmax_scale * 1.4426950408889634);
  LaunchPdl(&h3_varlen_attention_sage_sm120a::kernel_minimax_h3_sm120_varlen_attention_sage, grid, kAttentionThreads,
      kAttentionSmemBytes, stream,
      q_map, k_map, ksf_map, vt_map, vsf_map, static_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<unsigned int*>(q_sf.data_ptr()), static_cast<unsigned int*>(qm4.data_ptr()),
      static_cast<unsigned int*>(qm_sf.data_ptr()), static_cast<int*>(seg_tile_begin.data_ptr()),
      static_cast<int*>(unit_table.data_ptr()), static_cast<int>(num_units), heads_i, static_cast<int>(num_kblocks),
      softmax_scale_log2);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_sm120_varlen_attention_nvfp4, minimax_h3_sm120_varlen_attention_nvfp4);
// clang-format on
