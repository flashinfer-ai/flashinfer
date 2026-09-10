/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// clang-format off
// Generated from a recurrent-KDA Loom schedule.
// Raw generated body SHA256: 2307b896466dd58ff1daba770763b0a7142451e73225e940e9e9461a21bb9452
// Normalized generated SHA256: 1de157a38002ebc51ab603e79545eeea92d6fc13d53ed307e595ee29d04a8a02
// BEGIN FROZEN GENERATED BODY
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) LoomTensorMapPack { LoomTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SSTATE0_OFF 0
#define SMEM_SSTATE0_STAGE_BYTES 2048
#define SMEM_SSTATE0_STRIDE 2048
#define SMEM_SSTATE1_OFF 2048
#define SMEM_SSTATE1_STAGE_BYTES 2048
#define SMEM_SSTATE1_STRIDE 2048
#define SMEM_SVEC_OFF 4096
#define SMEM_SVEC_STAGE_BYTES 4096
#define SMEM_SVEC_STRIDE 4096
#define SMEM_SK_OFF 8192
#define SMEM_SK_STAGE_BYTES 2560
#define SMEM_SK_STRIDE 2560
#define SMEM_SD_OFF 10752
#define SMEM_SD_STAGE_BYTES 2560
#define SMEM_SD_STRIDE 2560
#define SMEM_SBETA_OFF 13312
#define SMEM_SBETA_STAGE_BYTES 20
#define SMEM_SBETA_STRIDE 20
#define SMEM_SSLOT_OFF 13332
#define SMEM_SSLOT_STAGE_BYTES 20
#define SMEM_SSLOT_STRIDE 20
#define SMEM_STOKEN_OFF 13352
#define SMEM_STOKEN_STAGE_BYTES 20
#define SMEM_STOKEN_STRIDE 20
#define SMEM_SINIT_OFF 13372
#define SMEM_SINIT_STAGE_BYTES 4
#define SMEM_SINIT_STRIDE 4
#define SMEM_SL_OFF 13388
#define SMEM_SL_STAGE_BYTES 100
#define SMEM_SL_STRIDE 100
#define SMEM_SR_OFF 13488
#define SMEM_SR_STAGE_BYTES 100
#define SMEM_SR_STRIDE 100
#define SMEM_SU_OFF 13588
#define SMEM_SU_STAGE_BYTES 320
#define SMEM_SU_STRIDE 320
#define SMEM_SGRAMA0_OFF 13952
#define SMEM_SGRAMA0_STAGE_BYTES 2048
#define SMEM_SGRAMA0_STRIDE 2048
#define SMEM_SGRAMA1_OFF 16000
#define SMEM_SGRAMA1_STAGE_BYTES 2048
#define SMEM_SGRAMA1_STRIDE 2048
#define SMEM_TOTAL 18048
#define THREADS 256
#define GATE_KIND 0
#define DIRECT_PREFIX_CHECKPOINT 0
#define BLOCK_CHECKPOINT_MMA 0

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_recurrent_kda_wy_vtile_short(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ beta, float* __restrict__ A_log, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ out, int* __restrict__ cu_seqlens, int* __restrict__ ssm_state_indices, int* __restrict__ num_accepted_tokens, float scale, float lower_bound, int g_stride_token, int state_stride_slot, int H, int HV, int head_ratio)
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
    __nv_bfloat16* sState0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int sState0_addr = smem + 0;
    __nv_bfloat16* sState1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int sState1_addr = smem + 2048;
    __nv_bfloat16* sVec = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4096);
    const int sVec_addr = smem + 4096;
    float* sK = reinterpret_cast<float*>(smem_raw + 8192);
    const int sK_addr = smem + 8192;
    float* sD = reinterpret_cast<float*>(smem_raw + 10752);
    const int sD_addr = smem + 10752;
    float* sBeta = reinterpret_cast<float*>(smem_raw + 13312);
    const int sBeta_addr = smem + 13312;
    int* sSlot = reinterpret_cast<int*>(smem_raw + 13332);
    const int sSlot_addr = smem + 13332;
    int* sToken = reinterpret_cast<int*>(smem_raw + 13352);
    const int sToken_addr = smem + 13352;
    int* sInit = reinterpret_cast<int*>(smem_raw + 13372);
    const int sInit_addr = smem + 13372;
    float* sL = reinterpret_cast<float*>(smem_raw + 13388);
    const int sL_addr = smem + 13388;
    float* sR = reinterpret_cast<float*>(smem_raw + 13488);
    const int sR_addr = smem + 13488;
    float* sU = reinterpret_cast<float*>(smem_raw + 13588);
    const int sU_addr = smem + 13588;
    __nv_bfloat16* sGramA0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 13952);
    const int sGramA0_addr = smem + 13952;
    __nv_bfloat16* sGramA1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16000);
    const int sGramA1_addr = smem + 16000;

    // === Task calls (dependency order) ===
    int work = blockIdx.x;
    int value_tile = work & 7;
    int hv = work / 8;
    int n = blockIdx.y;
    int query_head = hv / head_ratio;
    int token_base = cu_seqlens[n];
    int seq_len = cu_seqlens[n + 1] - token_base;
    int warp_0 = warp;
    int lane_1 = lane;
    int lane_quad = lane_1 & 3;
    int quad_base = lane_1 - lane_quad;
    int frag_row = (lane_1 >> 2 & 1) + (lane_1 >> 3 & 1) * 2;
    frag_row = frag_row + (lane_1 >> 4 & 1) * 4;
    int group = tid / 16;
    int lane_group = tid - group * 16;
    const int k_per_lane = 8;
    int k_start = lane_group * k_per_lane;
    int tile_row_base = value_tile * 16;
    int owned_row_base = group * 2;
    float r_q[4];
    float r_k[4];
    float r_d[4];
    float ratio_scan[4];
    float r_state[8];
    float hist[16];
    unsigned int state_pack[4];
    unsigned int state_frag[4];
    unsigned int vec_frag[4];
    float mma_acc[4];
    float mma_acc_c[4];
    float ha_lo[5];
    float ha_hi[5];
    float hc_lo[5];
    float hc_hi[5];
    float u_lo[5];
    float u_hi[5];
    int token = warp_0;
    int elem_start = lane_1 * 4;
    if (warp_0 < 5) {
        bool active_token = token < seq_len;
        int token_pos = token_base + token;
        if (!active_token) {
            token_pos = 0;
        }
        int qk_base = (token_pos * H + query_head) * 128 + elem_start;
        int gate_base = token_pos * g_stride_token + hv * 128 + elem_start;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            r_q[i] = 0.0f;
            r_k[i] = 0.0f;
            r_d[i] = 0.0f;
        }
        if (elem_start < 128) {
            {
                uint2 _vld_0 = *reinterpret_cast<const uint2*>(q + qk_base);
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&r_q[0 + _pair * 2])[0]), "=f"((&r_q[0 + _pair * 2])[1])
                        : "r"(_vpairs_0[_pair]));
                }
            }
            {
                uint2 _vld_1 = *reinterpret_cast<const uint2*>(k + qk_base);
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&r_k[0 + _pair * 2])[0]), "=f"((&r_k[0 + _pair * 2])[1])
                        : "r"(_vpairs_1[_pair]));
                }
            }
            {
                uint2 _vld_2 = *reinterpret_cast<const uint2*>(g + gate_base);
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&r_d[0 + _pair * 2])[0]), "=f"((&r_d[0 + _pair * 2])[1])
                        : "r"(_vpairs_2[_pair]));
                }
            }
        }
        float q_sq = 0.0f;
        float k_sq = 0.0f;
        #pragma unroll
        for (int i_1 = 0; i_1 < 4; i_1++) {
            q_sq += r_q[i_1] * r_q[i_1];
            k_sq += r_k[i_1] * r_k[i_1];
        }
        float _warp_reduce_0 = q_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        q_sq = _warp_reduce_0;
        float _warp_reduce_1 = k_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        k_sq = _warp_reduce_1;
        float _rsqrt_0 = rsqrtf(q_sq + 1e-06f);
        float q_norm = _rsqrt_0 * scale;
        float _rsqrt_1 = rsqrtf(k_sq + 1e-06f);
        float k_norm = _rsqrt_1;
        float gate_a = 1.0f;
        if (elem_start < 128) {
            #pragma unroll
            for (int i_2 = 0; i_2 < 4; i_2++) {
                int k_idx = elem_start + i_2;
                r_q[i_2] = r_q[i_2] * q_norm;
                r_k[i_2] = r_k[i_2] * k_norm;
                float log_gate = r_d[i_2];
                float _expf_2 = __expf(log_gate);
                r_d[i_2] = _expf_2;
                sK[token * 128 + k_idx] = r_k[i_2];
                sD[token * 128 + k_idx] = r_d[i_2];
            }
        }
        if (lane_1 == 0) {
            int raw_slot = ssm_state_indices[n * 5 + token];
            sSlot[token] = ((active_token) ? raw_slot : -1);
            sToken[token] = token_pos;
            sBeta[token] = (float)beta[token_pos * HV + hv];
            if (token == 0) {
                int accepted = num_accepted_tokens[n] - 1;
                if (accepted < 0) {
                    accepted = 0;
                }
                if (accepted >= 5) {
                    accepted = 4;
                }
                int initial_slot = ssm_state_indices[n * 5 + accepted];
                sInit[0] = ((initial_slot < 0) ? 0 : initial_slot);
            }
        }
    }
    __syncthreads();
    int initial_head_base = sInit[0] * state_stride_slot + hv * 128 * 128;
    if (warp_0 < 5) {
        if (elem_start < 128) {
            #pragma unroll
            for (int i_3 = 0; i_3 < 4; i_3++) {
                int k_idx_1 = elem_start + i_3;
                float prefix = 1.0f;
                #pragma unroll
                for (int prefix_token = 0; prefix_token < 5; prefix_token++) {
                    if (token >= prefix_token) {
                        prefix *= sD[prefix_token * 128 + k_idx_1];
                    }
                }
                {
                    __nv_bfloat16 _bval_972216480 = __float2bfloat16_rn(prefix * r_k[i_3]);
                    uint16_t _bits_972216480 = *(uint16_t*)&_bval_972216480;
                    uint32_t _addr_972216480 = static_cast<uint32_t>((sVec_addr + (unsigned int)(k_idx_1 * 32 + token * 2 ^ (k_idx_1 * 32 + token * 2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_972216480), "h"(_bits_972216480) : "memory");
                }
                int c_col = 4 + token;
                {
                    c_col = 8 + token;
                }
                {
                    __nv_bfloat16 _bval_972219984 = __float2bfloat16_rn(prefix * r_q[i_3]);
                    uint16_t _bits_972219984 = *(uint16_t*)&_bval_972219984;
                    uint32_t _addr_972219984 = static_cast<uint32_t>((sVec_addr + (unsigned int)(k_idx_1 * 32 + c_col * 2 ^ (k_idx_1 * 32 + c_col * 2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_972219984), "h"(_bits_972219984) : "memory");
                }
                {
                    if (k_idx_1 < 64) {
                        {
                            __nv_bfloat16 _bval_972213648 = __float2bfloat16_rn(r_k[i_3] / prefix);
                            uint16_t _bits_972213648 = *(uint16_t*)&_bval_972213648;
                            uint32_t _addr_972213648 = static_cast<uint32_t>((sGramA0_addr + (unsigned int)(token * 128 + k_idx_1 * 2 ^ (token * 128 + k_idx_1 * 2 >> 7 & 7) << 4)));
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_972213648), "h"(_bits_972213648) : "memory");
                        }
                    } else {
                        {
                            __nv_bfloat16 _bval_972213648 = __float2bfloat16_rn(r_k[i_3] / prefix);
                            uint16_t _bits_972213648 = *(uint16_t*)&_bval_972213648;
                            uint32_t _addr_972213648 = static_cast<uint32_t>((sGramA1_addr + (unsigned int)(token * 128 + (k_idx_1 - 64) * 2 ^ (token * 128 + (k_idx_1 - 64) * 2 >> 7 & 7) << 4)));
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_972213648), "h"(_bits_972213648) : "memory");
                        }
                    }
                }
            }
        }
        {
            if (warp_0 == ((1) ? 4 : 0)) {
                {
                    asm volatile("barrier.sync 1, 160;" ::: "memory");
                }
                unsigned int gram_a_frag[4];
                unsigned int gram_b_frag[4];
                float gram_k_acc[4];
                float gram_q_acc[4];
                #pragma unroll
                for (int gram_half = 0; gram_half < 2; gram_half++) {
                    #pragma unroll
                    for (int gram_k = 0; gram_k < 64; gram_k += 16) {
                        int global_gram_k = gram_half * 64 + gram_k;
                        int logical = (lane_1 % 16 * 64 + (gram_k + lane_1 / 16 * 8)) * 2;
                        unsigned int gram_a_addr = sGramA0_addr + (unsigned int)(logical ^ (logical >> 7 & 7) << 4);
                        if (gram_half != 0) {
                            int logical_0 = (lane_1 % 16 * 64 + (gram_k + lane_1 / 16 * 8)) * 2;
                            gram_a_addr = sGramA1_addr + (unsigned int)(logical_0 ^ (logical_0 >> 7 & 7) << 4);
                        }
                        int logical_0_1 = ((global_gram_k + lane_1 % 16) * 16 + lane_1 / 16 * 8) * 2;
                        unsigned int gram_b_addr = sVec_addr + (unsigned int)(logical_0_1 ^ (logical_0_1 >> 7 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(gram_a_frag[0]), "=r"(gram_a_frag[1]), "=r"(gram_a_frag[2]), "=r"(gram_a_frag[3])
                            : "r"(gram_a_addr)
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(gram_b_frag[0]), "=r"(gram_b_frag[1]), "=r"(gram_b_frag[2]), "=r"(gram_b_frag[3])
                            : "r"(gram_b_addr)
                            : "memory");
                        if (gram_half == 0 && gram_k == 0) {
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                                : "=f"(gram_k_acc[0]), "=f"(gram_k_acc[1]), "=f"(gram_k_acc[2]), "=f"(gram_k_acc[3])
                                : "r"(gram_a_frag[0]), "r"(gram_a_frag[1]), "r"(gram_a_frag[2]), "r"(gram_a_frag[3]), "r"(gram_b_frag[0]), "r"(gram_b_frag[1]));
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                                : "=f"(gram_q_acc[0]), "=f"(gram_q_acc[1]), "=f"(gram_q_acc[2]), "=f"(gram_q_acc[3])
                                : "r"(gram_a_frag[0]), "r"(gram_a_frag[1]), "r"(gram_a_frag[2]), "r"(gram_a_frag[3]), "r"(gram_b_frag[2]), "r"(gram_b_frag[(2) + 1]));
                        } else {
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(gram_k_acc[0]), "+f"(gram_k_acc[1]), "+f"(gram_k_acc[2]), "+f"(gram_k_acc[3])
                                : "r"(gram_a_frag[0]), "r"(gram_a_frag[1]), "r"(gram_a_frag[2]), "r"(gram_a_frag[3]), "r"(gram_b_frag[0]), "r"(gram_b_frag[1]));
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(gram_q_acc[0]), "+f"(gram_q_acc[1]), "+f"(gram_q_acc[2]), "+f"(gram_q_acc[3])
                                : "r"(gram_a_frag[0]), "r"(gram_a_frag[1]), "r"(gram_a_frag[2]), "r"(gram_a_frag[3]), "r"(gram_b_frag[2]), "r"(gram_b_frag[(2) + 1]));
                        }
                    }
                }
                int source_token = frag_row;
                int target0 = lane_quad * 2;
                int target1 = target0 + 1;
                if (source_token < 5) {
                    float beta_source = sBeta[source_token];
                    if (source_token < target0 && target0 < 5) {
                        sL[target0 * 5 + source_token] = beta_source * gram_k_acc[0];
                    }
                    if (source_token < target1 && target1 < 5) {
                        sL[target1 * 5 + source_token] = beta_source * gram_k_acc[1];
                    }
                    if (source_token <= target0 && target0 < 5) {
                        sR[target0 * 5 + source_token] = beta_source * gram_q_acc[0];
                    }
                    if (source_token <= target1 && target1 < 5) {
                        sR[target1 * 5 + source_token] = beta_source * gram_q_acc[1];
                    }
                }
            } else if (1) {
                asm volatile("barrier.arrive 1, 160;" ::: "memory");
            }
        }
    }
    {
        if (group < 8) {
            #pragma unroll
            for (int row_local = 0; row_local < 2; row_local++) {
                int value_row_local = owned_row_base + row_local;
                int value_row = tile_row_base + value_row_local;
                int state_base = initial_head_base + value_row * 128 + k_start;
                {
                    {
                        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(state + state_base);
                        uint4* _vdst_3 = reinterpret_cast<uint4*>(&state_pack[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_3[_blk] = _vptr_3[_blk];
                        }
                    }
                    {
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&hist[row_local * 8 + _pair * 2])[0]), "=f"((&hist[row_local * 8 + _pair * 2])[1])
                                : "r"(state_pack[_pair]));
                        }
                    }
                    if (lane_group < 8) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sState0_addr + (unsigned int)(value_row_local * 128 + k_start * 2 ^ (value_row_local * 128 + k_start * 2 >> 7 & 7) << 4))), "r"(state_pack[0]), "r"(state_pack[1]), "r"(state_pack[2]), "r"(state_pack[3]) : "memory");
                    } else {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sState1_addr + (unsigned int)(value_row_local * 128 + (k_start - 64) * 2 ^ (value_row_local * 128 + (k_start - 64) * 2 >> 7 & 7) << 4))), "r"(state_pack[0]), "r"(state_pack[1]), "r"(state_pack[2]), "r"(state_pack[3]) : "memory");
                    }
                }
            }
        }
    }
    __syncthreads();
    if (warp_0 < 1) {
        #pragma unroll
        for (int state_half = 0; state_half < 2; state_half++) {
            #pragma unroll
            for (int mma_k = 0; mma_k < 64; mma_k += 16) {
                int global_k = state_half * 64 + mma_k;
                int logical_1 = ((global_k + lane_1 % 16) * 16 + lane_1 / 16 * 8) * 2;
                unsigned int vec_addr = sVec_addr + (unsigned int)(logical_1 ^ (logical_1 >> 7 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vec_frag[0]), "=r"(vec_frag[1]), "=r"(vec_frag[2]), "=r"(vec_frag[3])
                    : "r"(vec_addr)
                    : "memory");
                int logical_0_2 = ((warp_0 * 16 + lane_1 % 16) * 64 + (mma_k + lane_1 / 16 * 8)) * 2;
                unsigned int state_addr = sState0_addr + (unsigned int)(logical_0_2 ^ (logical_0_2 >> 7 & 7) << 4);
                if (state_half != 0) {
                    int logical_1_1 = ((warp_0 * 16 + lane_1 % 16) * 64 + (mma_k + lane_1 / 16 * 8)) * 2;
                    state_addr = sState1_addr + (unsigned int)(logical_1_1 ^ (logical_1_1 >> 7 & 7) << 4);
                }
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(state_frag[0]), "=r"(state_frag[1]), "=r"(state_frag[2]), "=r"(state_frag[3])
                    : "r"(state_addr)
                    : "memory");
                if (state_half == 0 && mma_k == 0) {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                        : "=f"(mma_acc[0]), "=f"(mma_acc[1]), "=f"(mma_acc[2]), "=f"(mma_acc[3])
                        : "r"(state_frag[0]), "r"(state_frag[1]), "r"(state_frag[2]), "r"(state_frag[3]), "r"(vec_frag[0]), "r"(vec_frag[1]));
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(mma_acc_c[0]), "=f"(mma_acc_c[1]), "=f"(mma_acc_c[2]), "=f"(mma_acc_c[3])
                            : "r"(state_frag[0]), "r"(state_frag[1]), "r"(state_frag[2]), "r"(state_frag[3]), "r"(vec_frag[2]), "r"(vec_frag[(2) + 1]));
                    }
                } else {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(mma_acc[0]), "+f"(mma_acc[1]), "+f"(mma_acc[2]), "+f"(mma_acc[3])
                        : "r"(state_frag[0]), "r"(state_frag[1]), "r"(state_frag[2]), "r"(state_frag[3]), "r"(vec_frag[0]), "r"(vec_frag[1]));
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(mma_acc_c[0]), "+f"(mma_acc_c[1]), "+f"(mma_acc_c[2]), "+f"(mma_acc_c[3])
                            : "r"(state_frag[0]), "r"(state_frag[1]), "r"(state_frag[2]), "r"(state_frag[3]), "r"(vec_frag[2]), "r"(vec_frag[(2) + 1]));
                    }
                }
            }
        }
        float _shfl_0 = __shfl_sync(0xFFFFFFFF, mma_acc[0], quad_base);
        ha_lo[0] = _shfl_0;
        float _shfl_1 = __shfl_sync(0xFFFFFFFF, mma_acc[1], quad_base);
        ha_lo[1] = _shfl_1;
        float _shfl_2 = __shfl_sync(0xFFFFFFFF, mma_acc[0], quad_base + 1);
        ha_lo[2] = _shfl_2;
        float _shfl_3 = __shfl_sync(0xFFFFFFFF, mma_acc[1], quad_base + 1);
        ha_lo[3] = _shfl_3;
        float _shfl_4 = __shfl_sync(0xFFFFFFFF, mma_acc[2], quad_base);
        ha_hi[0] = _shfl_4;
        float _shfl_5 = __shfl_sync(0xFFFFFFFF, mma_acc[3], quad_base);
        ha_hi[1] = _shfl_5;
        float _shfl_6 = __shfl_sync(0xFFFFFFFF, mma_acc[2], quad_base + 1);
        ha_hi[2] = _shfl_6;
        float _shfl_7 = __shfl_sync(0xFFFFFFFF, mma_acc[3], quad_base + 1);
        ha_hi[3] = _shfl_7;
        {
            float _shfl_8 = __shfl_sync(0xFFFFFFFF, mma_acc[0], quad_base + 2);
            ha_lo[4] = _shfl_8;
            float _shfl_9 = __shfl_sync(0xFFFFFFFF, mma_acc[2], quad_base + 2);
            ha_hi[4] = _shfl_9;
            float _shfl_10 = __shfl_sync(0xFFFFFFFF, mma_acc_c[0], quad_base);
            hc_lo[0] = _shfl_10;
            float _shfl_11 = __shfl_sync(0xFFFFFFFF, mma_acc_c[1], quad_base);
            hc_lo[1] = _shfl_11;
            float _shfl_12 = __shfl_sync(0xFFFFFFFF, mma_acc_c[0], quad_base + 1);
            hc_lo[2] = _shfl_12;
            float _shfl_13 = __shfl_sync(0xFFFFFFFF, mma_acc_c[1], quad_base + 1);
            hc_lo[3] = _shfl_13;
            float _shfl_14 = __shfl_sync(0xFFFFFFFF, mma_acc_c[0], quad_base + 2);
            hc_lo[4] = _shfl_14;
            float _shfl_15 = __shfl_sync(0xFFFFFFFF, mma_acc_c[2], quad_base);
            hc_hi[0] = _shfl_15;
            float _shfl_16 = __shfl_sync(0xFFFFFFFF, mma_acc_c[3], quad_base);
            hc_hi[1] = _shfl_16;
            float _shfl_17 = __shfl_sync(0xFFFFFFFF, mma_acc_c[2], quad_base + 1);
            hc_hi[2] = _shfl_17;
            float _shfl_18 = __shfl_sync(0xFFFFFFFF, mma_acc_c[3], quad_base + 1);
            hc_hi[3] = _shfl_18;
            float _shfl_19 = __shfl_sync(0xFFFFFFFF, mma_acc_c[2], quad_base + 2);
            hc_hi[4] = _shfl_19;
        }
        if (lane_quad == 2) {
            int u_row_lo = warp_0 * 16 + frag_row;
            int u_row_hi = u_row_lo + 8;
            int value_row_lo = tile_row_base + u_row_lo;
            int value_row_hi = tile_row_base + u_row_hi;
            #pragma unroll
            for (int solve_token = 0; solve_token < 5; solve_token++) {
                int value_base = (sToken[solve_token] * HV + hv) * 128;
                float solved_lo = (float)v[value_base + value_row_lo] - ha_lo[solve_token];
                float solved_hi = (float)v[value_base + value_row_hi] - ha_hi[solve_token];
                #pragma unroll
                for (int previous = 0; previous < 5; previous++) {
                    if (previous < solve_token) {
                        solved_lo -= sL[solve_token * 5 + previous] * u_lo[previous];
                        solved_hi -= sL[solve_token * 5 + previous] * u_hi[previous];
                    }
                }
                u_lo[solve_token] = solved_lo;
                u_hi[solve_token] = solved_hi;
            }
        }
        #pragma unroll
        for (int solve_token_1 = 0; solve_token_1 < 5; solve_token_1++) {
            float _shfl_20 = __shfl_sync(0xFFFFFFFF, u_lo[solve_token_1], quad_base + 2);
            u_lo[solve_token_1] = _shfl_20;
            float _shfl_21 = __shfl_sync(0xFFFFFFFF, u_hi[solve_token_1], quad_base + 2);
            u_hi[solve_token_1] = _shfl_21;
        }
    }
    if (warp_0 < 1 && lane_quad >= 2) {
        int token0 = (lane_quad - 2) * 2;
        int token1 = token0 + 1;
        int value_row_lo_1 = warp_0 * 16 + frag_row;
        int value_row_hi_1 = value_row_lo_1 + 8;
        float out0_lo = mma_acc[0];
        float out1_lo = mma_acc[1];
        float out0_hi = mma_acc[2];
        float out1_hi = mma_acc[3];
        {
            out0_lo = hc_lo[0];
            out1_lo = hc_lo[1];
            out0_hi = hc_hi[0];
            out1_hi = hc_hi[1];
            if (lane_quad == 3) {
                out0_lo = hc_lo[2];
                out1_lo = hc_lo[3];
                out0_hi = hc_hi[2];
                out1_hi = hc_hi[3];
            }
        }
        #pragma unroll
        for (int source_token_1 = 0; source_token_1 < 5; source_token_1++) {
            float residual_lo = u_lo[source_token_1];
            float residual_hi = u_hi[source_token_1];
            float coef0 = 0.0f;
            float coef1 = 0.0f;
            if (token0 >= source_token_1) {
                coef0 = sR[token0 * 5 + source_token_1];
            }
            if (token1 >= source_token_1) {
                coef1 = sR[token1 * 5 + source_token_1];
            }
            out0_lo += coef0 * residual_lo;
            out1_lo += coef1 * residual_lo;
            out0_hi += coef0 * residual_hi;
            out1_hi += coef1 * residual_hi;
        }
        int global_row_lo = tile_row_base + value_row_lo_1;
        int global_row_hi = tile_row_base + value_row_hi_1;
        int slot0 = sSlot[token0];
        int slot1 = sSlot[token1];
        if (token0 < 5) {
            if (slot0 >= 0) {
                out[(sToken[token0] * HV + hv) * 128 + global_row_lo] = out0_lo;
                out[(sToken[token0] * HV + hv) * 128 + global_row_hi] = out0_hi;
            } else {
                out[(sToken[token0] * HV + hv) * 128 + global_row_lo] = 0.0f;
                out[(sToken[token0] * HV + hv) * 128 + global_row_hi] = 0.0f;
            }
        }
        if (token1 < 5) {
            if (slot1 >= 0) {
                out[(sToken[token1] * HV + hv) * 128 + global_row_lo] = out1_lo;
                out[(sToken[token1] * HV + hv) * 128 + global_row_hi] = out1_hi;
            } else {
                out[(sToken[token1] * HV + hv) * 128 + global_row_lo] = 0.0f;
                out[(sToken[token1] * HV + hv) * 128 + global_row_hi] = 0.0f;
            }
        }
        if (lane_quad == 2) {
            float out4_lo = hc_lo[4];
            float out4_hi = hc_hi[4];
            #pragma unroll
            for (int source_token_2 = 0; source_token_2 < 5; source_token_2++) {
                float coef4 = sR[20 + source_token_2];
                out4_lo += coef4 * u_lo[source_token_2];
                out4_hi += coef4 * u_hi[source_token_2];
            }
            int slot4 = sSlot[4];
            if (slot4 >= 0) {
                out[(sToken[4] * HV + hv) * 128 + global_row_lo] = out4_lo;
                out[(sToken[4] * HV + hv) * 128 + global_row_hi] = out4_hi;
            } else {
                out[(sToken[4] * HV + hv) * 128 + global_row_lo] = 0.0f;
                out[(sToken[4] * HV + hv) * 128 + global_row_hi] = 0.0f;
            }
        }
    }
    if (warp_0 < 1) {
        if (lane_quad == 2) {
            int u_row_lo_1 = warp_0 * 16 + frag_row;
            int u_row_hi_1 = u_row_lo_1 + 8;
            #pragma unroll
            for (int solved_token = 0; solved_token < 5; solved_token++) {
                sU[solved_token * 16 + u_row_lo_1] = u_lo[solved_token];
                sU[solved_token * 16 + u_row_hi_1] = u_hi[solved_token];
            }
        }
    }
    {
        __syncthreads();
    }
    if (BLOCK_CHECKPOINT_MMA != 0 && warp_0 < 1) {
        int mma_lane_group = lane_1 / 4;
        int mma_lane_in_group = lane_1 & 3;
        int source0 = mma_lane_in_group * 2;
        int source1 = source0 + 1;
        int checkpoint_row_lo = warp_0 * 16 + frag_row;
        int checkpoint_row_hi = checkpoint_row_lo + 8;
        int checkpoint_global_row_lo = tile_row_base + checkpoint_row_lo;
        int checkpoint_global_row_hi = tile_row_base + checkpoint_row_hi;
        float u_mma_vals[8];
        unsigned int u_mma_frag[4];
        float coefficient_vals[4];
        unsigned int coefficient_frag[2];
        float checkpoint_acc[4];
        float checkpoint_lo[2];
        float checkpoint_hi[2];
        #pragma unroll
        for (int i_4 = 0; i_4 < 8; i_4++) {
            u_mma_vals[i_4] = 0.0f;
        }
        if (source0 < 5) {
            u_mma_vals[0] = u_lo[source0];
            u_mma_vals[2] = u_hi[source0];
        }
        if (source1 < 5) {
            u_mma_vals[1] = u_lo[source1];
            u_mma_vals[3] = u_hi[source1];
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(u_mma_vals[_lp*2 + 0], u_mma_vals[_lp*2+1 + 0]));
            u_mma_frag[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int key_tile = 0; key_tile < 128; key_tile += 8) {
            int coefficient_key = key_tile + mma_lane_group;
            int checkpoint_key0 = key_tile + lane_quad * 2;
            int checkpoint_key1 = checkpoint_key0 + 1;
            float h0_lo0 = 0.0f;
            float h0_lo1 = 0.0f;
            float h0_hi0 = 0.0f;
            float h0_hi1 = 0.0f;
            if (checkpoint_key0 < 64) {
                int logical_2 = (checkpoint_row_lo * 64 + checkpoint_key0) * 2;
                h0_lo0 = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(sState0) + (logical_2 ^ (logical_2 >> 7 & 7) << 4))[0];
                int logical_0_3 = (checkpoint_row_lo * 64 + checkpoint_key1) * 2;
                h0_lo1 = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(sState0) + (logical_0_3 ^ (logical_0_3 >> 7 & 7) << 4))[0];
                int logical_1_2 = (checkpoint_row_hi * 64 + checkpoint_key0) * 2;
                h0_hi0 = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(sState0) + (logical_1_2 ^ (logical_1_2 >> 7 & 7) << 4))[0];
                int logical_2_1 = (checkpoint_row_hi * 64 + checkpoint_key1) * 2;
                h0_hi1 = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(sState0) + (logical_2_1 ^ (logical_2_1 >> 7 & 7) << 4))[0];
            } else {
                int logical_3 = (checkpoint_row_lo * 64 + (checkpoint_key0 - 64)) * 2;
                h0_lo0 = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(sState1) + (logical_3 ^ (logical_3 >> 7 & 7) << 4))[0];
                int logical_0_4 = (checkpoint_row_lo * 64 + (checkpoint_key1 - 64)) * 2;
                h0_lo1 = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(sState1) + (logical_0_4 ^ (logical_0_4 >> 7 & 7) << 4))[0];
                int logical_1_3 = (checkpoint_row_hi * 64 + (checkpoint_key0 - 64)) * 2;
                h0_hi0 = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(sState1) + (logical_1_3 ^ (logical_1_3 >> 7 & 7) << 4))[0];
                int logical_2_2 = (checkpoint_row_hi * 64 + (checkpoint_key1 - 64)) * 2;
                h0_hi1 = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(sState1) + (logical_2_2 ^ (logical_2_2 >> 7 & 7) << 4))[0];
            }
            float initial_prefix0 = 1.0f;
            float initial_prefix1 = 1.0f;
            float coefficient0 = 0.0f;
            float coefficient1 = 0.0f;
            #pragma unroll
            for (int checkpoint_token = 0; checkpoint_token < 5; checkpoint_token++) {
                initial_prefix0 *= sD[checkpoint_token * 128 + checkpoint_key0];
                initial_prefix1 *= sD[checkpoint_token * 128 + checkpoint_key1];
                if (checkpoint_token == source0) {
                    coefficient0 = sBeta[source0] * sK[source0 * 128 + coefficient_key];
                }
                if (source0 < checkpoint_token) {
                    coefficient0 *= sD[checkpoint_token * 128 + coefficient_key];
                }
                if (checkpoint_token == source1) {
                    coefficient1 = sBeta[source1] * sK[source1 * 128 + coefficient_key];
                }
                if (source1 < checkpoint_token) {
                    coefficient1 *= sD[checkpoint_token * 128 + coefficient_key];
                }
                coefficient_vals[0] = coefficient0;
                coefficient_vals[1] = coefficient1;
                coefficient_vals[2] = 0.0f;
                coefficient_vals[3] = 0.0f;
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(coefficient_vals[_lp*2 + 0], coefficient_vals[_lp*2+1 + 0]));
                    coefficient_frag[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                    : "=f"(checkpoint_acc[0]), "=f"(checkpoint_acc[1]), "=f"(checkpoint_acc[2]), "=f"(checkpoint_acc[3])
                    : "r"(u_mma_frag[0]), "r"(u_mma_frag[1]), "r"(u_mma_frag[2]), "r"(u_mma_frag[3]), "r"(coefficient_frag[0]), "r"(coefficient_frag[1]));
                checkpoint_lo[0] = checkpoint_acc[0] + h0_lo0 * initial_prefix0;
                checkpoint_lo[1] = checkpoint_acc[1] + h0_lo1 * initial_prefix1;
                checkpoint_hi[0] = checkpoint_acc[2] + h0_hi0 * initial_prefix0;
                checkpoint_hi[1] = checkpoint_acc[3] + h0_hi1 * initial_prefix1;
                int checkpoint_slot = sSlot[checkpoint_token];
                if (checkpoint_slot >= 0) {
                    int checkpoint_base_lo = checkpoint_slot * state_stride_slot + hv * 128 * 128 + checkpoint_global_row_lo * 128 + checkpoint_key0;
                    int checkpoint_base_hi = checkpoint_slot * state_stride_slot + hv * 128 * 128 + checkpoint_global_row_hi * 128 + checkpoint_key0;
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(checkpoint_lo[0 + 0], checkpoint_lo[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(state + checkpoint_base_lo))[0]) = _pk;
                    }
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(checkpoint_hi[0 + 0], checkpoint_hi[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(state + checkpoint_base_hi))[0]) = _pk;
                    }
                }
            }
        }
    }
    if (BLOCK_CHECKPOINT_MMA == 0 && group < 8) {
        #pragma unroll
        for (int checkpoint_token_1 = 0; checkpoint_token_1 < 5; checkpoint_token_1++) {
            int slot_t = sSlot[checkpoint_token_1];
            #pragma unroll
            for (int row_local_1 = 0; row_local_1 < 2; row_local_1++) {
                int value_row_local_1 = owned_row_base + row_local_1;
                {
                    float beta_t = sBeta[checkpoint_token_1];
                    float update = sU[checkpoint_token_1 * 16 + value_row_local_1] * beta_t;
                    {
                        #pragma unroll
                        for (int i_5 = 0; i_5 < 8; i_5++) {
                            int k_idx_2 = k_start + i_5;
                            float state_value = hist[row_local_1 * 8 + i_5] * sD[checkpoint_token_1 * 128 + k_idx_2] + update * sK[checkpoint_token_1 * 128 + k_idx_2];
                            hist[row_local_1 * 8 + i_5] = state_value;
                            r_state[i_5] = state_value;
                        }
                    }
                }
                if (slot_t >= 0) {
                    int value_row_1 = tile_row_base + value_row_local_1;
                    int checkpoint_base = slot_t * state_stride_slot + hv * 128 * 128 + value_row_1 * 128 + k_start;
                    {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(r_state[0 + 0], r_state[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(r_state[0 + 2], r_state[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(r_state[0 + 4], r_state[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(r_state[0 + 6], r_state[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[checkpoint_base + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

// END FROZEN GENERATED BODY
// clang-format on
