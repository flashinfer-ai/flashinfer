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
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
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

#define LAUNCH_MIN_BLOCKS 1

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





__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_sm120_quant_pre_attention_0f96e947ae73cae3b0fc(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ act_scale, float* __restrict__ w_scale, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, unsigned int* __restrict__ q_out, unsigned int* __restrict__ k_out, unsigned int* __restrict__ v_out, uint8_t* __restrict__ q_sf, uint8_t* __restrict__ k_sf, uint8_t* __restrict__ v_sf, int M, int num_m_tiles, int total_tiles, float eps, float alpha, float out_scale_q, float out_scale_k, float out_scale_v, float sf_mul_q, float sf_mul_k, float sf_mul_v)
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

    const int cta_rank = 0;

    // Prologue
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int B_stage_addr = smem + 66560;
    float* rowstat = reinterpret_cast<float*>(smem_raw + 99328);
    const int rowstat_addr = smem + 99328;

    // Mbarrier init (8 barriers at smem_raw[0..64))

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
            unsigned int load_stage_0 = 0;
            unsigned int load_g_0 = 0;
            unsigned int mma_stage_0 = 0;
            unsigned int mma_kk_0 = 0;
            int warp_m_0 = 0;
            int warp_n_0 = 0;
            int role_tid_0 = lane;
            float accum_0[128];
            unsigned int a_frag_0[16];
            unsigned int b_frag_0[16];
            unsigned int fp8_unit_word[1];
            fp8_unit_word[0] = 2139062143;
            uint8_t fp8_unit_sfa[1];
            uint8_t fp8_unit_sfb[1];
            fp8_unit_sfa[0] = fp8_unit_word[0];
            fp8_unit_sfb[0] = fp8_unit_word[0];
            float sw_0[16];
            float nw_0[16];
            unsigned int _phase_ab_empty = 1;
            if (lane == 0) {
                for (int _prologue_0 = 0; _prologue_0 < 3; _prologue_0++) {
                    if (bid + (int)(load_g_0 / 84) * num_bids < total_tiles) {
                        mbarrier_wait_hint(ab_empty_addr + (load_stage_0) * 8, _phase_ab_empty, 10000000);
                        mbarrier_arrive_expect_tx(ab_full_addr + (load_stage_0) * 8, 24576);
                        tma_2d_gmem2smem(A_stage_addr + load_stage_0 * 16384, (&A), ((int)load_g_0 - (int)(load_g_0 / 84) * 84) * 64, ((bid + (int)(load_g_0 / 84) * num_bids) / 5376 * 32 + (bid + (int)(load_g_0 / 84) * num_bids - (bid + (int)(load_g_0 / 84) * num_bids) / 5376 * 5376) % ((num_m_tiles - (bid + (int)(load_g_0 / 84) * num_bids) / 5376 * 32 < 32) ? num_m_tiles - (bid + (int)(load_g_0 / 84) * num_bids) / 5376 * 32 : 32)) * 256, ab_full_addr + (load_stage_0) * 8);
                        tma_2d_gmem2smem(B_stage_addr + load_stage_0 * 8192, (&B), ((int)load_g_0 - (int)(load_g_0 / 84) * 84) * 64, (bid + (int)(load_g_0 / 84) * num_bids - (bid + (int)(load_g_0 / 84) * num_bids) / 5376 * 5376) / ((num_m_tiles - (bid + (int)(load_g_0 / 84) * num_bids) / 5376 * 32 < 32) ? num_m_tiles - (bid + (int)(load_g_0 / 84) * num_bids) / 5376 * 32 : 32) * 128, ab_full_addr + (load_stage_0) * 8);
                    }
                    load_stage_0 += 1;
                    if (load_stage_0 == 4) { load_stage_0 = 0; _phase_ab_empty ^= 1; }
                    load_g_0 = load_g_0 + 1;
                }
            }
            __syncwarp();
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile_0 = bid; tile_0 < total_tiles; tile_0 += num_bids) {
                int tile_m_0 = tile_0 / 5376 * 32 + (tile_0 - tile_0 / 5376 * 5376) % ((num_m_tiles - tile_0 / 5376 * 32 < 32) ? num_m_tiles - tile_0 / 5376 * 32 : 32);
                int tile_n_0 = (tile_0 - tile_0 / 5376 * 5376) / ((num_m_tiles - tile_0 / 5376 * 32 < 32) ? num_m_tiles - tile_0 / 5376 * 32 : 32);
                accum_0[0] = 0.0f;
                accum_0[1] = 0.0f;
                accum_0[2] = 0.0f;
                accum_0[3] = 0.0f;
                accum_0[4] = 0.0f;
                accum_0[5] = 0.0f;
                accum_0[6] = 0.0f;
                accum_0[7] = 0.0f;
                accum_0[8] = 0.0f;
                accum_0[9] = 0.0f;
                accum_0[10] = 0.0f;
                accum_0[11] = 0.0f;
                accum_0[12] = 0.0f;
                accum_0[13] = 0.0f;
                accum_0[14] = 0.0f;
                accum_0[15] = 0.0f;
                accum_0[16] = 0.0f;
                accum_0[17] = 0.0f;
                accum_0[18] = 0.0f;
                accum_0[19] = 0.0f;
                accum_0[20] = 0.0f;
                accum_0[21] = 0.0f;
                accum_0[22] = 0.0f;
                accum_0[23] = 0.0f;
                accum_0[24] = 0.0f;
                accum_0[25] = 0.0f;
                accum_0[26] = 0.0f;
                accum_0[27] = 0.0f;
                accum_0[28] = 0.0f;
                accum_0[29] = 0.0f;
                accum_0[30] = 0.0f;
                accum_0[31] = 0.0f;
                accum_0[32] = 0.0f;
                accum_0[33] = 0.0f;
                accum_0[34] = 0.0f;
                accum_0[35] = 0.0f;
                accum_0[36] = 0.0f;
                accum_0[37] = 0.0f;
                accum_0[38] = 0.0f;
                accum_0[39] = 0.0f;
                accum_0[40] = 0.0f;
                accum_0[41] = 0.0f;
                accum_0[42] = 0.0f;
                accum_0[43] = 0.0f;
                accum_0[44] = 0.0f;
                accum_0[45] = 0.0f;
                accum_0[46] = 0.0f;
                accum_0[47] = 0.0f;
                accum_0[48] = 0.0f;
                accum_0[49] = 0.0f;
                accum_0[50] = 0.0f;
                accum_0[51] = 0.0f;
                accum_0[52] = 0.0f;
                accum_0[53] = 0.0f;
                accum_0[54] = 0.0f;
                accum_0[55] = 0.0f;
                accum_0[56] = 0.0f;
                accum_0[57] = 0.0f;
                accum_0[58] = 0.0f;
                accum_0[59] = 0.0f;
                accum_0[60] = 0.0f;
                accum_0[61] = 0.0f;
                accum_0[62] = 0.0f;
                accum_0[63] = 0.0f;
                accum_0[64] = 0.0f;
                accum_0[65] = 0.0f;
                accum_0[66] = 0.0f;
                accum_0[67] = 0.0f;
                accum_0[68] = 0.0f;
                accum_0[69] = 0.0f;
                accum_0[70] = 0.0f;
                accum_0[71] = 0.0f;
                accum_0[72] = 0.0f;
                accum_0[73] = 0.0f;
                accum_0[74] = 0.0f;
                accum_0[75] = 0.0f;
                accum_0[76] = 0.0f;
                accum_0[77] = 0.0f;
                accum_0[78] = 0.0f;
                accum_0[79] = 0.0f;
                accum_0[80] = 0.0f;
                accum_0[81] = 0.0f;
                accum_0[82] = 0.0f;
                accum_0[83] = 0.0f;
                accum_0[84] = 0.0f;
                accum_0[85] = 0.0f;
                accum_0[86] = 0.0f;
                accum_0[87] = 0.0f;
                accum_0[88] = 0.0f;
                accum_0[89] = 0.0f;
                accum_0[90] = 0.0f;
                accum_0[91] = 0.0f;
                accum_0[92] = 0.0f;
                accum_0[93] = 0.0f;
                accum_0[94] = 0.0f;
                accum_0[95] = 0.0f;
                accum_0[96] = 0.0f;
                accum_0[97] = 0.0f;
                accum_0[98] = 0.0f;
                accum_0[99] = 0.0f;
                accum_0[100] = 0.0f;
                accum_0[101] = 0.0f;
                accum_0[102] = 0.0f;
                accum_0[103] = 0.0f;
                accum_0[104] = 0.0f;
                accum_0[105] = 0.0f;
                accum_0[106] = 0.0f;
                accum_0[107] = 0.0f;
                accum_0[108] = 0.0f;
                accum_0[109] = 0.0f;
                accum_0[110] = 0.0f;
                accum_0[111] = 0.0f;
                accum_0[112] = 0.0f;
                accum_0[113] = 0.0f;
                accum_0[114] = 0.0f;
                accum_0[115] = 0.0f;
                accum_0[116] = 0.0f;
                accum_0[117] = 0.0f;
                accum_0[118] = 0.0f;
                accum_0[119] = 0.0f;
                accum_0[120] = 0.0f;
                accum_0[121] = 0.0f;
                accum_0[122] = 0.0f;
                accum_0[123] = 0.0f;
                accum_0[124] = 0.0f;
                accum_0[125] = 0.0f;
                accum_0[126] = 0.0f;
                accum_0[127] = 0.0f;
                int c_kind_0 = tile_n_0 % 3;
                int c_col_base_0 = tile_n_0 * 128 + warp_n_0 * 64 + (lane & 3) * 2;
                sw_0[0] = w_scale[c_col_base_0];
                sw_0[1] = w_scale[c_col_base_0 + 1];
                sw_0[2] = w_scale[c_col_base_0 + 8];
                sw_0[3] = w_scale[c_col_base_0 + 8 + 1];
                sw_0[4] = w_scale[c_col_base_0 + 16];
                sw_0[5] = w_scale[c_col_base_0 + 16 + 1];
                sw_0[6] = w_scale[c_col_base_0 + 24];
                sw_0[7] = w_scale[c_col_base_0 + 24 + 1];
                sw_0[8] = w_scale[c_col_base_0 + 32];
                sw_0[9] = w_scale[c_col_base_0 + 32 + 1];
                sw_0[10] = w_scale[c_col_base_0 + 40];
                sw_0[11] = w_scale[c_col_base_0 + 40 + 1];
                sw_0[12] = w_scale[c_col_base_0 + 48];
                sw_0[13] = w_scale[c_col_base_0 + 48 + 1];
                sw_0[14] = w_scale[c_col_base_0 + 56];
                sw_0[15] = w_scale[c_col_base_0 + 56 + 1];
                int c_d_col_0 = warp_n_0 * 64 + (lane & 3) * 2;
                if (c_kind_0 == 0) {
                    nw_0[0] = (float)q_norm_weight[c_d_col_0];
                    nw_0[1] = (float)q_norm_weight[c_d_col_0 + 1];
                } else if (c_kind_0 == 1) {
                    nw_0[0] = (float)k_norm_weight[c_d_col_0];
                    nw_0[1] = (float)k_norm_weight[c_d_col_0 + 1];
                } else {
                    nw_0[0] = 1.0f;
                    nw_0[1] = 1.0f;
                }
                int c_d_col_1 = warp_n_0 * 64 + 8 + (lane & 3) * 2;
                if (c_kind_0 == 0) {
                    nw_0[2] = (float)q_norm_weight[c_d_col_1];
                    nw_0[3] = (float)q_norm_weight[c_d_col_1 + 1];
                } else if (c_kind_0 == 1) {
                    nw_0[2] = (float)k_norm_weight[c_d_col_1];
                    nw_0[3] = (float)k_norm_weight[c_d_col_1 + 1];
                } else {
                    nw_0[2] = 1.0f;
                    nw_0[3] = 1.0f;
                }
                int c_d_col_2 = warp_n_0 * 64 + 16 + (lane & 3) * 2;
                if (c_kind_0 == 0) {
                    nw_0[4] = (float)q_norm_weight[c_d_col_2];
                    nw_0[5] = (float)q_norm_weight[c_d_col_2 + 1];
                } else if (c_kind_0 == 1) {
                    nw_0[4] = (float)k_norm_weight[c_d_col_2];
                    nw_0[5] = (float)k_norm_weight[c_d_col_2 + 1];
                } else {
                    nw_0[4] = 1.0f;
                    nw_0[5] = 1.0f;
                }
                int c_d_col_3 = warp_n_0 * 64 + 24 + (lane & 3) * 2;
                if (c_kind_0 == 0) {
                    nw_0[6] = (float)q_norm_weight[c_d_col_3];
                    nw_0[7] = (float)q_norm_weight[c_d_col_3 + 1];
                } else if (c_kind_0 == 1) {
                    nw_0[6] = (float)k_norm_weight[c_d_col_3];
                    nw_0[7] = (float)k_norm_weight[c_d_col_3 + 1];
                } else {
                    nw_0[6] = 1.0f;
                    nw_0[7] = 1.0f;
                }
                int c_d_col_4 = warp_n_0 * 64 + 32 + (lane & 3) * 2;
                if (c_kind_0 == 0) {
                    nw_0[8] = (float)q_norm_weight[c_d_col_4];
                    nw_0[9] = (float)q_norm_weight[c_d_col_4 + 1];
                } else if (c_kind_0 == 1) {
                    nw_0[8] = (float)k_norm_weight[c_d_col_4];
                    nw_0[9] = (float)k_norm_weight[c_d_col_4 + 1];
                } else {
                    nw_0[8] = 1.0f;
                    nw_0[9] = 1.0f;
                }
                int c_d_col_5 = warp_n_0 * 64 + 40 + (lane & 3) * 2;
                if (c_kind_0 == 0) {
                    nw_0[10] = (float)q_norm_weight[c_d_col_5];
                    nw_0[11] = (float)q_norm_weight[c_d_col_5 + 1];
                } else if (c_kind_0 == 1) {
                    nw_0[10] = (float)k_norm_weight[c_d_col_5];
                    nw_0[11] = (float)k_norm_weight[c_d_col_5 + 1];
                } else {
                    nw_0[10] = 1.0f;
                    nw_0[11] = 1.0f;
                }
                int c_d_col_6 = warp_n_0 * 64 + 48 + (lane & 3) * 2;
                if (c_kind_0 == 0) {
                    nw_0[12] = (float)q_norm_weight[c_d_col_6];
                    nw_0[13] = (float)q_norm_weight[c_d_col_6 + 1];
                } else if (c_kind_0 == 1) {
                    nw_0[12] = (float)k_norm_weight[c_d_col_6];
                    nw_0[13] = (float)k_norm_weight[c_d_col_6 + 1];
                } else {
                    nw_0[12] = 1.0f;
                    nw_0[13] = 1.0f;
                }
                int c_d_col_7 = warp_n_0 * 64 + 56 + (lane & 3) * 2;
                if (c_kind_0 == 0) {
                    nw_0[14] = (float)q_norm_weight[c_d_col_7];
                    nw_0[15] = (float)q_norm_weight[c_d_col_7 + 1];
                } else if (c_kind_0 == 1) {
                    nw_0[14] = (float)k_norm_weight[c_d_col_7];
                    nw_0[15] = (float)k_norm_weight[c_d_col_7 + 1];
                } else {
                    nw_0[14] = 1.0f;
                    nw_0[15] = 1.0f;
                }
                if (lane == 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                #pragma unroll 1
                for (int k_tile_0 = 0; k_tile_0 < 84; k_tile_0++) {
                    if (lane == 0) {
                        if (bid + (int)(load_g_0 / 84) * num_bids < total_tiles) {
                            mbarrier_wait_hint(ab_empty_addr + (load_stage_0) * 8, _phase_ab_empty, 10000000);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage_0) * 8, 24576);
                            tma_2d_gmem2smem(A_stage_addr + load_stage_0 * 16384, (&A), ((int)load_g_0 - (int)(load_g_0 / 84) * 84) * 64, ((bid + (int)(load_g_0 / 84) * num_bids) / 5376 * 32 + (bid + (int)(load_g_0 / 84) * num_bids - (bid + (int)(load_g_0 / 84) * num_bids) / 5376 * 5376) % ((num_m_tiles - (bid + (int)(load_g_0 / 84) * num_bids) / 5376 * 32 < 32) ? num_m_tiles - (bid + (int)(load_g_0 / 84) * num_bids) / 5376 * 32 : 32)) * 256, ab_full_addr + (load_stage_0) * 8);
                            tma_2d_gmem2smem(B_stage_addr + load_stage_0 * 8192, (&B), ((int)load_g_0 - (int)(load_g_0 / 84) * 84) * 64, (bid + (int)(load_g_0 / 84) * num_bids - (bid + (int)(load_g_0 / 84) * num_bids) / 5376 * 5376) / ((num_m_tiles - (bid + (int)(load_g_0 / 84) * num_bids) / 5376 * 32 < 32) ? num_m_tiles - (bid + (int)(load_g_0 / 84) * num_bids) / 5376 * 32 : 32) * 128, ab_full_addr + (load_stage_0) * 8);
                        }
                        load_stage_0 += 1;
                        if (load_stage_0 == 4) { load_stage_0 = 0; _phase_ab_empty ^= 1; }
                        load_g_0 = load_g_0 + 1;
                    }
                    __syncwarp();
                    mbarrier_wait_hint(ab_full_addr + (mma_stage_0) * 8, _phase_ab_full, 10000000);
                    for (int k_step_0 = 0; k_step_0 < 2; k_step_0++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(A_stage_addr + mma_stage_0 * 16384 + (unsigned int)((warp_m_0 * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_0 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_0 * 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[4]), "=r"(a_frag_0[5]), "=r"(a_frag_0[6]), "=r"(a_frag_0[7])
                            : "r"(A_stage_addr + mma_stage_0 * 16384 + (unsigned int)((warp_m_0 * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_0 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_0 * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[8]), "=r"(a_frag_0[9]), "=r"(a_frag_0[10]), "=r"(a_frag_0[11])
                            : "r"(A_stage_addr + mma_stage_0 * 16384 + (unsigned int)((warp_m_0 * 16 + 128 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_0 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_0 * 16 + 128 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[12]), "=r"(a_frag_0[13]), "=r"(a_frag_0[14]), "=r"(a_frag_0[15])
                            : "r"(A_stage_addr + mma_stage_0 * 16384 + (unsigned int)((warp_m_0 * 16 + 192 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_0 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_0 * 16 + 192 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1]), "=r"(b_frag_0[2]), "=r"(b_frag_0[3])
                            : "r"(B_stage_addr + mma_stage_0 * 8192 + (unsigned int)((warp_n_0 * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_0 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_0 * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_0[4]), "=r"(b_frag_0[5]), "=r"(b_frag_0[6]), "=r"(b_frag_0[7])
                            : "r"(B_stage_addr + mma_stage_0 * 8192 + (unsigned int)((warp_n_0 * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_0 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_0 * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_0[8]), "=r"(b_frag_0[9]), "=r"(b_frag_0[10]), "=r"(b_frag_0[11])
                            : "r"(B_stage_addr + mma_stage_0 * 8192 + (unsigned int)((warp_n_0 * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_0 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_0 * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_0[12]), "=r"(b_frag_0[13]), "=r"(b_frag_0[14]), "=r"(b_frag_0[15])
                            : "r"(B_stage_addr + mma_stage_0 * 8192 + (unsigned int)((warp_n_0 * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_0 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_0 * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[0]), "+f"(accum_0[1]), "+f"(accum_0[2]), "+f"(accum_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[4]), "+f"(accum_0[(4) + 1]), "+f"(accum_0[(4) + 2]), "+f"(accum_0[(4) + 3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[2]), "r"(b_frag_0[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[8]), "+f"(accum_0[(8) + 1]), "+f"(accum_0[(8) + 2]), "+f"(accum_0[(8) + 3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[4]), "r"(b_frag_0[(4) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[12]), "+f"(accum_0[(12) + 1]), "+f"(accum_0[(12) + 2]), "+f"(accum_0[(12) + 3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[6]), "r"(b_frag_0[(6) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[16]), "+f"(accum_0[(16) + 1]), "+f"(accum_0[(16) + 2]), "+f"(accum_0[(16) + 3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[8]), "r"(b_frag_0[(8) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[20]), "+f"(accum_0[(20) + 1]), "+f"(accum_0[(20) + 2]), "+f"(accum_0[(20) + 3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[10]), "r"(b_frag_0[(10) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[24]), "+f"(accum_0[(24) + 1]), "+f"(accum_0[(24) + 2]), "+f"(accum_0[(24) + 3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[12]), "r"(b_frag_0[(12) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[28]), "+f"(accum_0[(28) + 1]), "+f"(accum_0[(28) + 2]), "+f"(accum_0[(28) + 3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[14]), "r"(b_frag_0[(14) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[32]), "+f"(accum_0[(32) + 1]), "+f"(accum_0[(32) + 2]), "+f"(accum_0[(32) + 3])
                            : "r"(a_frag_0[4]), "r"(a_frag_0[(4) + 1]), "r"(a_frag_0[(4) + 2]), "r"(a_frag_0[(4) + 3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[36]), "+f"(accum_0[(36) + 1]), "+f"(accum_0[(36) + 2]), "+f"(accum_0[(36) + 3])
                            : "r"(a_frag_0[4]), "r"(a_frag_0[(4) + 1]), "r"(a_frag_0[(4) + 2]), "r"(a_frag_0[(4) + 3]), "r"(b_frag_0[2]), "r"(b_frag_0[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[40]), "+f"(accum_0[(40) + 1]), "+f"(accum_0[(40) + 2]), "+f"(accum_0[(40) + 3])
                            : "r"(a_frag_0[4]), "r"(a_frag_0[(4) + 1]), "r"(a_frag_0[(4) + 2]), "r"(a_frag_0[(4) + 3]), "r"(b_frag_0[4]), "r"(b_frag_0[(4) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[44]), "+f"(accum_0[(44) + 1]), "+f"(accum_0[(44) + 2]), "+f"(accum_0[(44) + 3])
                            : "r"(a_frag_0[4]), "r"(a_frag_0[(4) + 1]), "r"(a_frag_0[(4) + 2]), "r"(a_frag_0[(4) + 3]), "r"(b_frag_0[6]), "r"(b_frag_0[(6) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[48]), "+f"(accum_0[(48) + 1]), "+f"(accum_0[(48) + 2]), "+f"(accum_0[(48) + 3])
                            : "r"(a_frag_0[4]), "r"(a_frag_0[(4) + 1]), "r"(a_frag_0[(4) + 2]), "r"(a_frag_0[(4) + 3]), "r"(b_frag_0[8]), "r"(b_frag_0[(8) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[52]), "+f"(accum_0[(52) + 1]), "+f"(accum_0[(52) + 2]), "+f"(accum_0[(52) + 3])
                            : "r"(a_frag_0[4]), "r"(a_frag_0[(4) + 1]), "r"(a_frag_0[(4) + 2]), "r"(a_frag_0[(4) + 3]), "r"(b_frag_0[10]), "r"(b_frag_0[(10) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[56]), "+f"(accum_0[(56) + 1]), "+f"(accum_0[(56) + 2]), "+f"(accum_0[(56) + 3])
                            : "r"(a_frag_0[4]), "r"(a_frag_0[(4) + 1]), "r"(a_frag_0[(4) + 2]), "r"(a_frag_0[(4) + 3]), "r"(b_frag_0[12]), "r"(b_frag_0[(12) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[60]), "+f"(accum_0[(60) + 1]), "+f"(accum_0[(60) + 2]), "+f"(accum_0[(60) + 3])
                            : "r"(a_frag_0[4]), "r"(a_frag_0[(4) + 1]), "r"(a_frag_0[(4) + 2]), "r"(a_frag_0[(4) + 3]), "r"(b_frag_0[14]), "r"(b_frag_0[(14) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[64]), "+f"(accum_0[(64) + 1]), "+f"(accum_0[(64) + 2]), "+f"(accum_0[(64) + 3])
                            : "r"(a_frag_0[8]), "r"(a_frag_0[(8) + 1]), "r"(a_frag_0[(8) + 2]), "r"(a_frag_0[(8) + 3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[68]), "+f"(accum_0[(68) + 1]), "+f"(accum_0[(68) + 2]), "+f"(accum_0[(68) + 3])
                            : "r"(a_frag_0[8]), "r"(a_frag_0[(8) + 1]), "r"(a_frag_0[(8) + 2]), "r"(a_frag_0[(8) + 3]), "r"(b_frag_0[2]), "r"(b_frag_0[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[72]), "+f"(accum_0[(72) + 1]), "+f"(accum_0[(72) + 2]), "+f"(accum_0[(72) + 3])
                            : "r"(a_frag_0[8]), "r"(a_frag_0[(8) + 1]), "r"(a_frag_0[(8) + 2]), "r"(a_frag_0[(8) + 3]), "r"(b_frag_0[4]), "r"(b_frag_0[(4) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[76]), "+f"(accum_0[(76) + 1]), "+f"(accum_0[(76) + 2]), "+f"(accum_0[(76) + 3])
                            : "r"(a_frag_0[8]), "r"(a_frag_0[(8) + 1]), "r"(a_frag_0[(8) + 2]), "r"(a_frag_0[(8) + 3]), "r"(b_frag_0[6]), "r"(b_frag_0[(6) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[80]), "+f"(accum_0[(80) + 1]), "+f"(accum_0[(80) + 2]), "+f"(accum_0[(80) + 3])
                            : "r"(a_frag_0[8]), "r"(a_frag_0[(8) + 1]), "r"(a_frag_0[(8) + 2]), "r"(a_frag_0[(8) + 3]), "r"(b_frag_0[8]), "r"(b_frag_0[(8) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[84]), "+f"(accum_0[(84) + 1]), "+f"(accum_0[(84) + 2]), "+f"(accum_0[(84) + 3])
                            : "r"(a_frag_0[8]), "r"(a_frag_0[(8) + 1]), "r"(a_frag_0[(8) + 2]), "r"(a_frag_0[(8) + 3]), "r"(b_frag_0[10]), "r"(b_frag_0[(10) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[88]), "+f"(accum_0[(88) + 1]), "+f"(accum_0[(88) + 2]), "+f"(accum_0[(88) + 3])
                            : "r"(a_frag_0[8]), "r"(a_frag_0[(8) + 1]), "r"(a_frag_0[(8) + 2]), "r"(a_frag_0[(8) + 3]), "r"(b_frag_0[12]), "r"(b_frag_0[(12) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[92]), "+f"(accum_0[(92) + 1]), "+f"(accum_0[(92) + 2]), "+f"(accum_0[(92) + 3])
                            : "r"(a_frag_0[8]), "r"(a_frag_0[(8) + 1]), "r"(a_frag_0[(8) + 2]), "r"(a_frag_0[(8) + 3]), "r"(b_frag_0[14]), "r"(b_frag_0[(14) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[96]), "+f"(accum_0[(96) + 1]), "+f"(accum_0[(96) + 2]), "+f"(accum_0[(96) + 3])
                            : "r"(a_frag_0[12]), "r"(a_frag_0[(12) + 1]), "r"(a_frag_0[(12) + 2]), "r"(a_frag_0[(12) + 3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[100]), "+f"(accum_0[(100) + 1]), "+f"(accum_0[(100) + 2]), "+f"(accum_0[(100) + 3])
                            : "r"(a_frag_0[12]), "r"(a_frag_0[(12) + 1]), "r"(a_frag_0[(12) + 2]), "r"(a_frag_0[(12) + 3]), "r"(b_frag_0[2]), "r"(b_frag_0[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[104]), "+f"(accum_0[(104) + 1]), "+f"(accum_0[(104) + 2]), "+f"(accum_0[(104) + 3])
                            : "r"(a_frag_0[12]), "r"(a_frag_0[(12) + 1]), "r"(a_frag_0[(12) + 2]), "r"(a_frag_0[(12) + 3]), "r"(b_frag_0[4]), "r"(b_frag_0[(4) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[108]), "+f"(accum_0[(108) + 1]), "+f"(accum_0[(108) + 2]), "+f"(accum_0[(108) + 3])
                            : "r"(a_frag_0[12]), "r"(a_frag_0[(12) + 1]), "r"(a_frag_0[(12) + 2]), "r"(a_frag_0[(12) + 3]), "r"(b_frag_0[6]), "r"(b_frag_0[(6) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[112]), "+f"(accum_0[(112) + 1]), "+f"(accum_0[(112) + 2]), "+f"(accum_0[(112) + 3])
                            : "r"(a_frag_0[12]), "r"(a_frag_0[(12) + 1]), "r"(a_frag_0[(12) + 2]), "r"(a_frag_0[(12) + 3]), "r"(b_frag_0[8]), "r"(b_frag_0[(8) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[116]), "+f"(accum_0[(116) + 1]), "+f"(accum_0[(116) + 2]), "+f"(accum_0[(116) + 3])
                            : "r"(a_frag_0[12]), "r"(a_frag_0[(12) + 1]), "r"(a_frag_0[(12) + 2]), "r"(a_frag_0[(12) + 3]), "r"(b_frag_0[10]), "r"(b_frag_0[(10) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[120]), "+f"(accum_0[(120) + 1]), "+f"(accum_0[(120) + 2]), "+f"(accum_0[(120) + 3])
                            : "r"(a_frag_0[12]), "r"(a_frag_0[(12) + 1]), "r"(a_frag_0[(12) + 2]), "r"(a_frag_0[(12) + 3]), "r"(b_frag_0[12]), "r"(b_frag_0[(12) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_0[124]), "+f"(accum_0[(124) + 1]), "+f"(accum_0[(124) + 2]), "+f"(accum_0[(124) + 3])
                            : "r"(a_frag_0[12]), "r"(a_frag_0[(12) + 1]), "r"(a_frag_0[(12) + 2]), "r"(a_frag_0[(12) + 3]), "r"(b_frag_0[14]), "r"(b_frag_0[(14) + 1]), "r"((uint32_t)(fp8_unit_sfa[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane == 0) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage_0) * 8);
                    }
                    mma_stage_0 += 1;
                    if (mma_stage_0 == 4) { mma_stage_0 = 0; _phase_ab_full ^= 1; }
                    mma_kk_0 = mma_kk_0 + 1;
                }
                unsigned int free_stage_0 = ((mma_stage_0 == 0) ? 3 : mma_stage_0 - 1);
                int w_kind_0 = tile_n_0 % 3;
                int w_head_0 = tile_n_0 / 3;
                float out_scale_sel_0 = ((w_kind_0 == 0) ? out_scale_q : ((w_kind_0 == 1) ? out_scale_k : out_scale_v));
                float sf_mul_sel_0 = ((w_kind_0 == 0) ? sf_mul_q : ((w_kind_0 == 1) ? sf_mul_k : sf_mul_v));
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int row_lo = tile_m_0 * 256 + warp_m_0 * 16 + (lane >> 2);
                int row_hi = row_lo + 8;
                float sa_lo = ((row_lo < M) ? act_scale[row_lo] : 0.0f);
                float sa_hi = ((row_hi < M) ? act_scale[row_hi] : 0.0f);
                float rounded[32];
                float ss_lo = 0.0f;
                float ss_hi = 0.0f;
                float scaled_0[4];
                scaled_0[0] = accum_0[0] * sa_lo * sw_0[0];
                scaled_0[1] = accum_0[1] * sa_lo * sw_0[1];
                scaled_0[2] = accum_0[2] * sa_hi * sw_0[0];
                scaled_0[3] = accum_0[3] * sa_hi * sw_0[1];
                unsigned int scaled_bf16_0[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_0[_lp*2 + 0], scaled_0[_lp*2+1 + 0]));
                    scaled_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_0[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_0[_pair]) << 16);
                    (&unpacked_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_0[_pair]) & 0xffff0000u);
                }
                rounded[0] = unpacked_0[0];
                rounded[1] = unpacked_0[1];
                rounded[2] = unpacked_0[2];
                rounded[3] = unpacked_0[3];
                ss_lo += __fmaf_rn(unpacked_0[0], unpacked_0[0], unpacked_0[1] * unpacked_0[1]);
                ss_hi += __fmaf_rn(unpacked_0[2], unpacked_0[2], unpacked_0[3] * unpacked_0[3]);
                float scaled_1[4];
                scaled_1[0] = accum_0[4] * sa_lo * sw_0[2];
                scaled_1[1] = accum_0[5] * sa_lo * sw_0[3];
                scaled_1[2] = accum_0[6] * sa_hi * sw_0[2];
                scaled_1[3] = accum_0[7] * sa_hi * sw_0[3];
                unsigned int scaled_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_1[_lp*2 + 0], scaled_1[_lp*2+1 + 0]));
                    scaled_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_1[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_1[_pair]) << 16);
                    (&unpacked_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_1[_pair]) & 0xffff0000u);
                }
                rounded[4] = unpacked_1[0];
                rounded[5] = unpacked_1[1];
                rounded[6] = unpacked_1[2];
                rounded[7] = unpacked_1[3];
                ss_lo += __fmaf_rn(unpacked_1[0], unpacked_1[0], unpacked_1[1] * unpacked_1[1]);
                ss_hi += __fmaf_rn(unpacked_1[2], unpacked_1[2], unpacked_1[3] * unpacked_1[3]);
                float scaled_2[4];
                scaled_2[0] = accum_0[8] * sa_lo * sw_0[4];
                scaled_2[1] = accum_0[9] * sa_lo * sw_0[5];
                scaled_2[2] = accum_0[10] * sa_hi * sw_0[4];
                scaled_2[3] = accum_0[11] * sa_hi * sw_0[5];
                unsigned int scaled_bf16_2[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_2[_lp*2 + 0], scaled_2[_lp*2+1 + 0]));
                    scaled_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_2[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_2[_pair]) << 16);
                    (&unpacked_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_2[_pair]) & 0xffff0000u);
                }
                rounded[8] = unpacked_2[0];
                rounded[9] = unpacked_2[1];
                rounded[10] = unpacked_2[2];
                rounded[11] = unpacked_2[3];
                ss_lo += __fmaf_rn(unpacked_2[0], unpacked_2[0], unpacked_2[1] * unpacked_2[1]);
                ss_hi += __fmaf_rn(unpacked_2[2], unpacked_2[2], unpacked_2[3] * unpacked_2[3]);
                float scaled_3[4];
                scaled_3[0] = accum_0[12] * sa_lo * sw_0[6];
                scaled_3[1] = accum_0[13] * sa_lo * sw_0[7];
                scaled_3[2] = accum_0[14] * sa_hi * sw_0[6];
                scaled_3[3] = accum_0[15] * sa_hi * sw_0[7];
                unsigned int scaled_bf16_3[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_3[_lp*2 + 0], scaled_3[_lp*2+1 + 0]));
                    scaled_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_3[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_3[_pair]) << 16);
                    (&unpacked_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_3[_pair]) & 0xffff0000u);
                }
                rounded[12] = unpacked_3[0];
                rounded[13] = unpacked_3[1];
                rounded[14] = unpacked_3[2];
                rounded[15] = unpacked_3[3];
                ss_lo += __fmaf_rn(unpacked_3[0], unpacked_3[0], unpacked_3[1] * unpacked_3[1]);
                ss_hi += __fmaf_rn(unpacked_3[2], unpacked_3[2], unpacked_3[3] * unpacked_3[3]);
                float scaled_4[4];
                scaled_4[0] = accum_0[16] * sa_lo * sw_0[8];
                scaled_4[1] = accum_0[17] * sa_lo * sw_0[9];
                scaled_4[2] = accum_0[18] * sa_hi * sw_0[8];
                scaled_4[3] = accum_0[19] * sa_hi * sw_0[9];
                unsigned int scaled_bf16_4[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_4[_lp*2 + 0], scaled_4[_lp*2+1 + 0]));
                    scaled_bf16_4[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_4[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_4[_pair]) << 16);
                    (&unpacked_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_4[_pair]) & 0xffff0000u);
                }
                rounded[16] = unpacked_4[0];
                rounded[17] = unpacked_4[1];
                rounded[18] = unpacked_4[2];
                rounded[19] = unpacked_4[3];
                ss_lo += __fmaf_rn(unpacked_4[0], unpacked_4[0], unpacked_4[1] * unpacked_4[1]);
                ss_hi += __fmaf_rn(unpacked_4[2], unpacked_4[2], unpacked_4[3] * unpacked_4[3]);
                float scaled_5[4];
                scaled_5[0] = accum_0[20] * sa_lo * sw_0[10];
                scaled_5[1] = accum_0[21] * sa_lo * sw_0[11];
                scaled_5[2] = accum_0[22] * sa_hi * sw_0[10];
                scaled_5[3] = accum_0[23] * sa_hi * sw_0[11];
                unsigned int scaled_bf16_5[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_5[_lp*2 + 0], scaled_5[_lp*2+1 + 0]));
                    scaled_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_5[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_5[_pair]) << 16);
                    (&unpacked_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_5[_pair]) & 0xffff0000u);
                }
                rounded[20] = unpacked_5[0];
                rounded[21] = unpacked_5[1];
                rounded[22] = unpacked_5[2];
                rounded[23] = unpacked_5[3];
                ss_lo += __fmaf_rn(unpacked_5[0], unpacked_5[0], unpacked_5[1] * unpacked_5[1]);
                ss_hi += __fmaf_rn(unpacked_5[2], unpacked_5[2], unpacked_5[3] * unpacked_5[3]);
                float scaled_6[4];
                scaled_6[0] = accum_0[24] * sa_lo * sw_0[12];
                scaled_6[1] = accum_0[25] * sa_lo * sw_0[13];
                scaled_6[2] = accum_0[26] * sa_hi * sw_0[12];
                scaled_6[3] = accum_0[27] * sa_hi * sw_0[13];
                unsigned int scaled_bf16_6[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_6[_lp*2 + 0], scaled_6[_lp*2+1 + 0]));
                    scaled_bf16_6[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_6[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_6[_pair]) << 16);
                    (&unpacked_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_6[_pair]) & 0xffff0000u);
                }
                rounded[24] = unpacked_6[0];
                rounded[25] = unpacked_6[1];
                rounded[26] = unpacked_6[2];
                rounded[27] = unpacked_6[3];
                ss_lo += __fmaf_rn(unpacked_6[0], unpacked_6[0], unpacked_6[1] * unpacked_6[1]);
                ss_hi += __fmaf_rn(unpacked_6[2], unpacked_6[2], unpacked_6[3] * unpacked_6[3]);
                float scaled_7[4];
                scaled_7[0] = accum_0[28] * sa_lo * sw_0[14];
                scaled_7[1] = accum_0[29] * sa_lo * sw_0[15];
                scaled_7[2] = accum_0[30] * sa_hi * sw_0[14];
                scaled_7[3] = accum_0[31] * sa_hi * sw_0[15];
                unsigned int scaled_bf16_7[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_7[_lp*2 + 0], scaled_7[_lp*2+1 + 0]));
                    scaled_bf16_7[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_7[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_7[_pair]) << 16);
                    (&unpacked_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_7[_pair]) & 0xffff0000u);
                }
                rounded[28] = unpacked_7[0];
                rounded[29] = unpacked_7[1];
                rounded[30] = unpacked_7[2];
                rounded[31] = unpacked_7[3];
                ss_lo += __fmaf_rn(unpacked_7[0], unpacked_7[0], unpacked_7[1] * unpacked_7[1]);
                ss_hi += __fmaf_rn(unpacked_7[2], unpacked_7[2], unpacked_7[3] * unpacked_7[3]);
                float rstd_lo = 1.0f;
                float rstd_hi = 1.0f;
                if (w_kind_0 < 2) {
                    float _shfl_xor_sync_0 = __shfl_xor_sync(0xFFFFFFFF, ss_lo, 1);
                    ss_lo += _shfl_xor_sync_0;
                    float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo, 2);
                    ss_lo += _shfl_xor_sync_1;
                    float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, ss_hi, 1);
                    ss_hi += _shfl_xor_sync_2;
                    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, ss_hi, 2);
                    ss_hi += _shfl_xor_sync_3;
                    int srow_lo = warp_m_0 * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n_0 * 64 + srow_lo] = ss_lo;
                        rowstat[warp_n_0 * 64 + srow_lo + 8] = ss_hi;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float other_lo = rowstat[(1 - warp_n_0) * 64 + srow_lo];
                    float other_hi = rowstat[(1 - warp_n_0) * 64 + srow_lo + 8];
                    float _fdiv_5 = __fdiv_rn(ss_lo + other_lo, 128.0f);
                    float _rsqrt_4 = rsqrtf(_fdiv_5 + eps);
                    rstd_lo = _rsqrt_4;
                    float _fdiv_7 = __fdiv_rn(ss_hi + other_hi, 128.0f);
                    float _rsqrt_6 = rsqrtf(_fdiv_7 + eps);
                    rstd_hi = _rsqrt_6;
                }
                int srow_st = warp_m_0 * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float normed_0[4];
                if (w_kind_0 < 2) {
                    normed_0[0] = rounded[0] * rstd_lo * nw_0[0];
                    normed_0[1] = rounded[1] * rstd_lo * nw_0[1];
                    normed_0[2] = rounded[2] * rstd_hi * nw_0[0];
                    normed_0[3] = rounded[3] * rstd_hi * nw_0[1];
                } else {
                    normed_0[0] = rounded[0];
                    normed_0[1] = rounded[1];
                    normed_0[2] = rounded[2];
                    normed_0[3] = rounded[3];
                }
                unsigned int normed_bf16_0[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_0[_lp*2 + 0], normed_0[_lp*2+1 + 0]));
                    normed_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st * 256) + (unsigned int)(warp_n_0 * 64 * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_0[1]))
                    : "memory");
                float normed_1[4];
                if (w_kind_0 < 2) {
                    normed_1[0] = rounded[4] * rstd_lo * nw_0[2];
                    normed_1[1] = rounded[5] * rstd_lo * nw_0[3];
                    normed_1[2] = rounded[6] * rstd_hi * nw_0[2];
                    normed_1[3] = rounded[7] * rstd_hi * nw_0[3];
                } else {
                    normed_1[0] = rounded[4];
                    normed_1[1] = rounded[5];
                    normed_1[2] = rounded[6];
                    normed_1[3] = rounded[7];
                }
                unsigned int normed_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_1[_lp*2 + 0], normed_1[_lp*2+1 + 0]));
                    normed_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n_0 * 64 + 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_1[1]))
                    : "memory");
                float normed_2[4];
                if (w_kind_0 < 2) {
                    normed_2[0] = rounded[8] * rstd_lo * nw_0[4];
                    normed_2[1] = rounded[9] * rstd_lo * nw_0[5];
                    normed_2[2] = rounded[10] * rstd_hi * nw_0[4];
                    normed_2[3] = rounded[11] * rstd_hi * nw_0[5];
                } else {
                    normed_2[0] = rounded[8];
                    normed_2[1] = rounded[9];
                    normed_2[2] = rounded[10];
                    normed_2[3] = rounded[11];
                }
                unsigned int normed_bf16_2[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_2[_lp*2 + 0], normed_2[_lp*2+1 + 0]));
                    normed_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n_0 * 64 + 16) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_2[1]))
                    : "memory");
                float normed_3[4];
                if (w_kind_0 < 2) {
                    normed_3[0] = rounded[12] * rstd_lo * nw_0[6];
                    normed_3[1] = rounded[13] * rstd_lo * nw_0[7];
                    normed_3[2] = rounded[14] * rstd_hi * nw_0[6];
                    normed_3[3] = rounded[15] * rstd_hi * nw_0[7];
                } else {
                    normed_3[0] = rounded[12];
                    normed_3[1] = rounded[13];
                    normed_3[2] = rounded[14];
                    normed_3[3] = rounded[15];
                }
                unsigned int normed_bf16_3[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_3[_lp*2 + 0], normed_3[_lp*2+1 + 0]));
                    normed_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n_0 * 64 + 24) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_3[1]))
                    : "memory");
                float normed_4[4];
                if (w_kind_0 < 2) {
                    normed_4[0] = rounded[16] * rstd_lo * nw_0[8];
                    normed_4[1] = rounded[17] * rstd_lo * nw_0[9];
                    normed_4[2] = rounded[18] * rstd_hi * nw_0[8];
                    normed_4[3] = rounded[19] * rstd_hi * nw_0[9];
                } else {
                    normed_4[0] = rounded[16];
                    normed_4[1] = rounded[17];
                    normed_4[2] = rounded[18];
                    normed_4[3] = rounded[19];
                }
                unsigned int normed_bf16_4[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_4[_lp*2 + 0], normed_4[_lp*2+1 + 0]));
                    normed_bf16_4[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n_0 * 64 + 32) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_4[1]))
                    : "memory");
                float normed_5[4];
                if (w_kind_0 < 2) {
                    normed_5[0] = rounded[20] * rstd_lo * nw_0[10];
                    normed_5[1] = rounded[21] * rstd_lo * nw_0[11];
                    normed_5[2] = rounded[22] * rstd_hi * nw_0[10];
                    normed_5[3] = rounded[23] * rstd_hi * nw_0[11];
                } else {
                    normed_5[0] = rounded[20];
                    normed_5[1] = rounded[21];
                    normed_5[2] = rounded[22];
                    normed_5[3] = rounded[23];
                }
                unsigned int normed_bf16_5[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_5[_lp*2 + 0], normed_5[_lp*2+1 + 0]));
                    normed_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n_0 * 64 + 40) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_5[1]))
                    : "memory");
                float normed_6[4];
                if (w_kind_0 < 2) {
                    normed_6[0] = rounded[24] * rstd_lo * nw_0[12];
                    normed_6[1] = rounded[25] * rstd_lo * nw_0[13];
                    normed_6[2] = rounded[26] * rstd_hi * nw_0[12];
                    normed_6[3] = rounded[27] * rstd_hi * nw_0[13];
                } else {
                    normed_6[0] = rounded[24];
                    normed_6[1] = rounded[25];
                    normed_6[2] = rounded[26];
                    normed_6[3] = rounded[27];
                }
                unsigned int normed_bf16_6[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_6[_lp*2 + 0], normed_6[_lp*2+1 + 0]));
                    normed_bf16_6[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n_0 * 64 + 48) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_6[1]))
                    : "memory");
                float normed_7[4];
                if (w_kind_0 < 2) {
                    normed_7[0] = rounded[28] * rstd_lo * nw_0[14];
                    normed_7[1] = rounded[29] * rstd_lo * nw_0[15];
                    normed_7[2] = rounded[30] * rstd_hi * nw_0[14];
                    normed_7[3] = rounded[31] * rstd_hi * nw_0[15];
                } else {
                    normed_7[0] = rounded[28];
                    normed_7[1] = rounded[29];
                    normed_7[2] = rounded[30];
                    normed_7[3] = rounded[31];
                }
                unsigned int normed_bf16_7[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_7[_lp*2 + 0], normed_7[_lp*2+1 + 0]));
                    normed_bf16_7[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n_0 * 64 + 56) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_7[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int srow = role_tid_0 >> 2;
                int c = role_tid_0 & 3;
                int grow = tile_m_0 * 256 + srow;
                int chunk_lo = ((c < 3) ? c * 16 : 96);
                int chunk_hi = ((c < 3) ? chunk_lo + 48 : 112);
                if (grow < M) {
                    unsigned int x_lo[8];
                    unsigned int x_hi[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow * 256) + (unsigned int)(chunk_lo * 2 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow * 256) + (unsigned int)(chunk_lo * 2 + 16 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow * 256) + (unsigned int)(chunk_hi * 2 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow * 256) + (unsigned int)(chunk_hi * 2 + 16 ^ (srow & 7) << 4)));
                    if (c < 3) {
                        if (w_kind_0 < 2) {
                            float x_lo_f[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_lo_f[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo[_pair]) << 16);
                                (&x_lo_f[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo[_pair]) & 0xffff0000u);
                            }
                            float x_hi_f[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_hi_f[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi[_pair]) << 16);
                                (&x_hi_f[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi[_pair]) & 0xffff0000u);
                            }
                            float v_0[8];
                            {
                                const uint4* _vptr_16 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + chunk_lo) + 0);
                                uint4 _vld_16[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_16[_blk] = _vptr_16[_blk];
                                    uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                                        (&v_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_1[8];
                            {
                                const uint4* _vptr_17 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + chunk_lo + 8) + 0);
                                uint4 _vld_17[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_17[_blk] = _vptr_17[_blk];
                                    uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
                                        (&v_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_2[8];
                            {
                                const uint4* _vptr_18 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + 48 + chunk_lo) + 0);
                                uint4 _vld_18[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_18[_blk] = _vptr_18[_blk];
                                    uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                                        (&v_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_3[8];
                            {
                                const uint4* _vptr_19 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow * 96 + 48 + chunk_lo + 8) + 0);
                                uint4 _vld_19[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_19[_blk] = _vptr_19[_blk];
                                    uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                                        (&v_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float o_lo[16];
                            float o_hi[16];
                            float cv0 = v_0[0];
                            float sv0 = v_2[0];
                            float cv1 = v_1[0];
                            float sv1 = v_3[0];
                            o_lo[0] = __fmaf_rn(x_lo_f[0], cv0, -(x_hi_f[0] * sv0));
                            o_hi[0] = __fmaf_rn(x_hi_f[0], cv0, x_lo_f[0] * sv0);
                            o_lo[8] = __fmaf_rn(x_lo_f[8], cv1, -(x_hi_f[8] * sv1));
                            o_hi[8] = __fmaf_rn(x_hi_f[8], cv1, x_lo_f[8] * sv1);
                            float cv0_0 = v_0[1];
                            float sv0_0 = v_2[1];
                            float cv1_0 = v_1[1];
                            float sv1_0 = v_3[1];
                            o_lo[1] = __fmaf_rn(x_lo_f[1], cv0_0, -(x_hi_f[1] * sv0_0));
                            o_hi[1] = __fmaf_rn(x_hi_f[1], cv0_0, x_lo_f[1] * sv0_0);
                            o_lo[9] = __fmaf_rn(x_lo_f[9], cv1_0, -(x_hi_f[9] * sv1_0));
                            o_hi[9] = __fmaf_rn(x_hi_f[9], cv1_0, x_lo_f[9] * sv1_0);
                            float cv0_1 = v_0[2];
                            float sv0_1 = v_2[2];
                            float cv1_1 = v_1[2];
                            float sv1_1 = v_3[2];
                            o_lo[2] = __fmaf_rn(x_lo_f[2], cv0_1, -(x_hi_f[2] * sv0_1));
                            o_hi[2] = __fmaf_rn(x_hi_f[2], cv0_1, x_lo_f[2] * sv0_1);
                            o_lo[10] = __fmaf_rn(x_lo_f[10], cv1_1, -(x_hi_f[10] * sv1_1));
                            o_hi[10] = __fmaf_rn(x_hi_f[10], cv1_1, x_lo_f[10] * sv1_1);
                            float cv0_2 = v_0[3];
                            float sv0_2 = v_2[3];
                            float cv1_2 = v_1[3];
                            float sv1_2 = v_3[3];
                            o_lo[3] = __fmaf_rn(x_lo_f[3], cv0_2, -(x_hi_f[3] * sv0_2));
                            o_hi[3] = __fmaf_rn(x_hi_f[3], cv0_2, x_lo_f[3] * sv0_2);
                            o_lo[11] = __fmaf_rn(x_lo_f[11], cv1_2, -(x_hi_f[11] * sv1_2));
                            o_hi[11] = __fmaf_rn(x_hi_f[11], cv1_2, x_lo_f[11] * sv1_2);
                            float cv0_3 = v_0[4];
                            float sv0_3 = v_2[4];
                            float cv1_3 = v_1[4];
                            float sv1_3 = v_3[4];
                            o_lo[4] = __fmaf_rn(x_lo_f[4], cv0_3, -(x_hi_f[4] * sv0_3));
                            o_hi[4] = __fmaf_rn(x_hi_f[4], cv0_3, x_lo_f[4] * sv0_3);
                            o_lo[12] = __fmaf_rn(x_lo_f[12], cv1_3, -(x_hi_f[12] * sv1_3));
                            o_hi[12] = __fmaf_rn(x_hi_f[12], cv1_3, x_lo_f[12] * sv1_3);
                            float cv0_4 = v_0[5];
                            float sv0_4 = v_2[5];
                            float cv1_4 = v_1[5];
                            float sv1_4 = v_3[5];
                            o_lo[5] = __fmaf_rn(x_lo_f[5], cv0_4, -(x_hi_f[5] * sv0_4));
                            o_hi[5] = __fmaf_rn(x_hi_f[5], cv0_4, x_lo_f[5] * sv0_4);
                            o_lo[13] = __fmaf_rn(x_lo_f[13], cv1_4, -(x_hi_f[13] * sv1_4));
                            o_hi[13] = __fmaf_rn(x_hi_f[13], cv1_4, x_lo_f[13] * sv1_4);
                            float cv0_5 = v_0[6];
                            float sv0_5 = v_2[6];
                            float cv1_5 = v_1[6];
                            float sv1_5 = v_3[6];
                            o_lo[6] = __fmaf_rn(x_lo_f[6], cv0_5, -(x_hi_f[6] * sv0_5));
                            o_hi[6] = __fmaf_rn(x_hi_f[6], cv0_5, x_lo_f[6] * sv0_5);
                            o_lo[14] = __fmaf_rn(x_lo_f[14], cv1_5, -(x_hi_f[14] * sv1_5));
                            o_hi[14] = __fmaf_rn(x_hi_f[14], cv1_5, x_lo_f[14] * sv1_5);
                            float cv0_6 = v_0[7];
                            float sv0_6 = v_2[7];
                            float cv1_6 = v_1[7];
                            float sv1_6 = v_3[7];
                            o_lo[7] = __fmaf_rn(x_lo_f[7], cv0_6, -(x_hi_f[7] * sv0_6));
                            o_hi[7] = __fmaf_rn(x_hi_f[7], cv0_6, x_lo_f[7] * sv0_6);
                            o_lo[15] = __fmaf_rn(x_lo_f[15], cv1_6, -(x_hi_f[15] * sv1_6));
                            o_hi[15] = __fmaf_rn(x_hi_f[15], cv1_6, x_lo_f[15] * sv1_6);
                            unsigned int o_lo_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_lo[_lp*2 + 0], o_lo[_lp*2+1 + 0]));
                                o_lo_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int o_hi_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_hi[_lp*2 + 0], o_hi[_lp*2+1 + 0]));
                                o_hi_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            x_lo[0] = o_lo_bf16[0];
                            x_hi[0] = o_hi_bf16[0];
                            x_lo[1] = o_lo_bf16[1];
                            x_hi[1] = o_hi_bf16[1];
                            x_lo[2] = o_lo_bf16[2];
                            x_hi[2] = o_hi_bf16[2];
                            x_lo[3] = o_lo_bf16[3];
                            x_hi[3] = o_hi_bf16[3];
                            x_lo[4] = o_lo_bf16[4];
                            x_hi[4] = o_hi_bf16[4];
                            x_lo[5] = o_lo_bf16[5];
                            x_hi[5] = o_hi_bf16[5];
                            x_lo[6] = o_lo_bf16[6];
                            x_hi[6] = o_hi_bf16[6];
                            x_lo[7] = o_lo_bf16[7];
                            x_hi[7] = o_hi_bf16[7];
                        }
                    }
                    unsigned int q_words[4];
                    float q_vals[16];
                    float f_0[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo[_pair]) << 16);
                        (&f_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_0;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_0) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_0[0]), fabsf(f_0[1])), fabsf(f_0[2])), fabsf(f_0[3])), fabsf(f_0[4])), fabsf(f_0[5])), fabsf(f_0[6])), fabsf(f_0[7])), fabsf(f_0[8])), fabsf(f_0[9])), fabsf(f_0[10])), fabsf(f_0[11])), fabsf(f_0[12])), fabsf(f_0[13])), fabsf(f_0[14])), fabsf(f_0[15])) * sf_mul_sel_0));
                    uint16_t _e4m3x2_decode_20 = (uint16_t)((unsigned int)sf_pair_0 & 0xFFu);
                    uint32_t _f16x2_decode_20;
                    float sf_dec_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_20) : "h"(_e4m3x2_decode_20));
                    uint16_t _f16_decode_20 = (uint16_t)_f16x2_decode_20;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_0) : "h"(_f16_decode_20));
                    float out_s_0 = __fdiv_rn(out_scale_sel_0, sf_dec_0);
                    q_vals[0] = f_0[0] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[1] = f_0[1] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[2] = f_0[2] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[3] = f_0[3] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[4] = f_0[4] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[5] = f_0[5] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[6] = f_0[6] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[7] = f_0[7] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[8] = f_0[8] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[9] = f_0[9] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[10] = f_0[10] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[11] = f_0[11] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[12] = f_0[12] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[13] = f_0[13] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[14] = f_0[14] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    q_vals[15] = f_0[15] * ((sf_dec_0 != 0.0f) ? out_s_0 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words[0]) : "f"(q_vals[0]), "f"(q_vals[1]), "f"(q_vals[2]), "f"(q_vals[3]), "f"(q_vals[4]), "f"(q_vals[5]), "f"(q_vals[6]), "f"(q_vals[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words[1]) : "f"(q_vals[8]), "f"(q_vals[9]), "f"(q_vals[10]), "f"(q_vals[11]), "f"(q_vals[12]), "f"(q_vals[13]), "f"(q_vals[14]), "f"(q_vals[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow * 128) + (unsigned int)(chunk_lo >> 1 ^ (srow & 7) << 4)), "r"(q_words[0]), "r"(q_words[1]) : "memory");
                    {
                        uint32_t _byte_21 = static_cast<uint32_t>((unsigned int)sf_pair_0 & 127) & 0xFFu;
                        uint32_t _addr_21 = static_cast<uint32_t>(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow * 128) + (unsigned int)(64 + (chunk_lo >> 4) ^ (srow & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_21), "r"(_byte_21) : "memory");
                    }
                    float f_1[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi[_pair]) << 16);
                        (&f_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_1;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_1) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_1[0]), fabsf(f_1[1])), fabsf(f_1[2])), fabsf(f_1[3])), fabsf(f_1[4])), fabsf(f_1[5])), fabsf(f_1[6])), fabsf(f_1[7])), fabsf(f_1[8])), fabsf(f_1[9])), fabsf(f_1[10])), fabsf(f_1[11])), fabsf(f_1[12])), fabsf(f_1[13])), fabsf(f_1[14])), fabsf(f_1[15])) * sf_mul_sel_0));
                    uint16_t _e4m3x2_decode_22 = (uint16_t)((unsigned int)sf_pair_1 & 0xFFu);
                    uint32_t _f16x2_decode_22;
                    float sf_dec_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_22) : "h"(_e4m3x2_decode_22));
                    uint16_t _f16_decode_22 = (uint16_t)_f16x2_decode_22;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_1) : "h"(_f16_decode_22));
                    float out_s_1 = __fdiv_rn(out_scale_sel_0, sf_dec_1);
                    q_vals[0] = f_1[0] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[1] = f_1[1] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[2] = f_1[2] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[3] = f_1[3] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[4] = f_1[4] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[5] = f_1[5] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[6] = f_1[6] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[7] = f_1[7] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[8] = f_1[8] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[9] = f_1[9] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[10] = f_1[10] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[11] = f_1[11] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[12] = f_1[12] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[13] = f_1[13] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[14] = f_1[14] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    q_vals[15] = f_1[15] * ((sf_dec_1 != 0.0f) ? out_s_1 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words[0]) : "f"(q_vals[0]), "f"(q_vals[1]), "f"(q_vals[2]), "f"(q_vals[3]), "f"(q_vals[4]), "f"(q_vals[5]), "f"(q_vals[6]), "f"(q_vals[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words[1]) : "f"(q_vals[8]), "f"(q_vals[9]), "f"(q_vals[10]), "f"(q_vals[11]), "f"(q_vals[12]), "f"(q_vals[13]), "f"(q_vals[14]), "f"(q_vals[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow * 128) + (unsigned int)(chunk_hi >> 1 ^ (srow & 7) << 4)), "r"(q_words[0]), "r"(q_words[1]) : "memory");
                    {
                        uint32_t _byte_23 = static_cast<uint32_t>((unsigned int)sf_pair_1 & 127) & 0xFFu;
                        uint32_t _addr_23 = static_cast<uint32_t>(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow * 128) + (unsigned int)(64 + (chunk_hi >> 4) ^ (srow & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_23), "r"(_byte_23) : "memory");
                    }
                }
                __syncwarp();
                int s_row_0 = (role_tid_0 >> 5) * 8 + lane / 4;
                int s_piece_0 = lane % 4;
                unsigned int s_words_0[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(s_row_0 * 128) + (unsigned int)(s_piece_0 * 16 ^ (s_row_0 & 7) << 4)));
                int s_dst_0 = ((tile_m_0 * 256 + s_row_0) * 56 + w_head_0) * 16 + s_piece_0 * 4;
                if (tile_m_0 * 256 + s_row_0 < M) {
                    if (w_kind_0 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_0)[0] = reinterpret_cast<int4*>(s_words_0)[0];
                    } else if (w_kind_0 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_0)[0] = reinterpret_cast<int4*>(s_words_0)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_0)[0] = reinterpret_cast<int4*>(s_words_0)[0];
                    }
                }
                int s_sf_row = (role_tid_0 >> 5) * 8 + (lane & 7);
                unsigned int s_sf_words[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words[(0) + 1]))
                    : "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(s_sf_row * 128) + (unsigned int)(64 ^ (s_sf_row & 7) << 4)));
                int s_sf_dst = ((tile_m_0 * 256 + s_sf_row) * 56 + w_head_0) * 8;
                if (lane < 8) {
                    if (tile_m_0 * 256 + s_sf_row < M) {
                        if (w_kind_0 == 0) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(q_sf + s_sf_dst))[0] = reinterpret_cast<int2*>(s_sf_words)[0];
                        } else if (w_kind_0 == 1) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(k_sf + s_sf_dst))[0] = reinterpret_cast<int2*>(s_sf_words)[0];
                        } else {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(v_sf + s_sf_dst))[0] = reinterpret_cast<int2*>(s_sf_words)[0];
                        }
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int row_lo_0 = tile_m_0 * 256 + 64 + warp_m_0 * 16 + (lane >> 2);
                int row_hi_0 = row_lo_0 + 8;
                float sa_lo_0 = ((row_lo_0 < M) ? act_scale[row_lo_0] : 0.0f);
                float sa_hi_0 = ((row_hi_0 < M) ? act_scale[row_hi_0] : 0.0f);
                float rounded_0[32];
                float ss_lo_0 = 0.0f;
                float ss_hi_0 = 0.0f;
                float scaled_8[4];
                scaled_8[0] = accum_0[32] * sa_lo_0 * sw_0[0];
                scaled_8[1] = accum_0[33] * sa_lo_0 * sw_0[1];
                scaled_8[2] = accum_0[34] * sa_hi_0 * sw_0[0];
                scaled_8[3] = accum_0[35] * sa_hi_0 * sw_0[1];
                unsigned int scaled_bf16_8[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_8[_lp*2 + 0], scaled_8[_lp*2+1 + 0]));
                    scaled_bf16_8[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_8[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_8[_pair]) << 16);
                    (&unpacked_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_8[_pair]) & 0xffff0000u);
                }
                rounded_0[0] = unpacked_8[0];
                rounded_0[1] = unpacked_8[1];
                rounded_0[2] = unpacked_8[2];
                rounded_0[3] = unpacked_8[3];
                ss_lo_0 += __fmaf_rn(unpacked_8[0], unpacked_8[0], unpacked_8[1] * unpacked_8[1]);
                ss_hi_0 += __fmaf_rn(unpacked_8[2], unpacked_8[2], unpacked_8[3] * unpacked_8[3]);
                float scaled_9[4];
                scaled_9[0] = accum_0[36] * sa_lo_0 * sw_0[2];
                scaled_9[1] = accum_0[37] * sa_lo_0 * sw_0[3];
                scaled_9[2] = accum_0[38] * sa_hi_0 * sw_0[2];
                scaled_9[3] = accum_0[39] * sa_hi_0 * sw_0[3];
                unsigned int scaled_bf16_9[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_9[_lp*2 + 0], scaled_9[_lp*2+1 + 0]));
                    scaled_bf16_9[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_9[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_9[_pair]) << 16);
                    (&unpacked_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_9[_pair]) & 0xffff0000u);
                }
                rounded_0[4] = unpacked_9[0];
                rounded_0[5] = unpacked_9[1];
                rounded_0[6] = unpacked_9[2];
                rounded_0[7] = unpacked_9[3];
                ss_lo_0 += __fmaf_rn(unpacked_9[0], unpacked_9[0], unpacked_9[1] * unpacked_9[1]);
                ss_hi_0 += __fmaf_rn(unpacked_9[2], unpacked_9[2], unpacked_9[3] * unpacked_9[3]);
                float scaled_10[4];
                scaled_10[0] = accum_0[40] * sa_lo_0 * sw_0[4];
                scaled_10[1] = accum_0[41] * sa_lo_0 * sw_0[5];
                scaled_10[2] = accum_0[42] * sa_hi_0 * sw_0[4];
                scaled_10[3] = accum_0[43] * sa_hi_0 * sw_0[5];
                unsigned int scaled_bf16_10[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_10[_lp*2 + 0], scaled_10[_lp*2+1 + 0]));
                    scaled_bf16_10[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_10[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_10[_pair]) << 16);
                    (&unpacked_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_10[_pair]) & 0xffff0000u);
                }
                rounded_0[8] = unpacked_10[0];
                rounded_0[9] = unpacked_10[1];
                rounded_0[10] = unpacked_10[2];
                rounded_0[11] = unpacked_10[3];
                ss_lo_0 += __fmaf_rn(unpacked_10[0], unpacked_10[0], unpacked_10[1] * unpacked_10[1]);
                ss_hi_0 += __fmaf_rn(unpacked_10[2], unpacked_10[2], unpacked_10[3] * unpacked_10[3]);
                float scaled_11[4];
                scaled_11[0] = accum_0[44] * sa_lo_0 * sw_0[6];
                scaled_11[1] = accum_0[45] * sa_lo_0 * sw_0[7];
                scaled_11[2] = accum_0[46] * sa_hi_0 * sw_0[6];
                scaled_11[3] = accum_0[47] * sa_hi_0 * sw_0[7];
                unsigned int scaled_bf16_11[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_11[_lp*2 + 0], scaled_11[_lp*2+1 + 0]));
                    scaled_bf16_11[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_11[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_11[_pair]) << 16);
                    (&unpacked_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_11[_pair]) & 0xffff0000u);
                }
                rounded_0[12] = unpacked_11[0];
                rounded_0[13] = unpacked_11[1];
                rounded_0[14] = unpacked_11[2];
                rounded_0[15] = unpacked_11[3];
                ss_lo_0 += __fmaf_rn(unpacked_11[0], unpacked_11[0], unpacked_11[1] * unpacked_11[1]);
                ss_hi_0 += __fmaf_rn(unpacked_11[2], unpacked_11[2], unpacked_11[3] * unpacked_11[3]);
                float scaled_12[4];
                scaled_12[0] = accum_0[48] * sa_lo_0 * sw_0[8];
                scaled_12[1] = accum_0[49] * sa_lo_0 * sw_0[9];
                scaled_12[2] = accum_0[50] * sa_hi_0 * sw_0[8];
                scaled_12[3] = accum_0[51] * sa_hi_0 * sw_0[9];
                unsigned int scaled_bf16_12[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_12[_lp*2 + 0], scaled_12[_lp*2+1 + 0]));
                    scaled_bf16_12[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_12[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_12[_pair]) << 16);
                    (&unpacked_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_12[_pair]) & 0xffff0000u);
                }
                rounded_0[16] = unpacked_12[0];
                rounded_0[17] = unpacked_12[1];
                rounded_0[18] = unpacked_12[2];
                rounded_0[19] = unpacked_12[3];
                ss_lo_0 += __fmaf_rn(unpacked_12[0], unpacked_12[0], unpacked_12[1] * unpacked_12[1]);
                ss_hi_0 += __fmaf_rn(unpacked_12[2], unpacked_12[2], unpacked_12[3] * unpacked_12[3]);
                float scaled_13[4];
                scaled_13[0] = accum_0[52] * sa_lo_0 * sw_0[10];
                scaled_13[1] = accum_0[53] * sa_lo_0 * sw_0[11];
                scaled_13[2] = accum_0[54] * sa_hi_0 * sw_0[10];
                scaled_13[3] = accum_0[55] * sa_hi_0 * sw_0[11];
                unsigned int scaled_bf16_13[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_13[_lp*2 + 0], scaled_13[_lp*2+1 + 0]));
                    scaled_bf16_13[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_13[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_13[_pair]) << 16);
                    (&unpacked_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_13[_pair]) & 0xffff0000u);
                }
                rounded_0[20] = unpacked_13[0];
                rounded_0[21] = unpacked_13[1];
                rounded_0[22] = unpacked_13[2];
                rounded_0[23] = unpacked_13[3];
                ss_lo_0 += __fmaf_rn(unpacked_13[0], unpacked_13[0], unpacked_13[1] * unpacked_13[1]);
                ss_hi_0 += __fmaf_rn(unpacked_13[2], unpacked_13[2], unpacked_13[3] * unpacked_13[3]);
                float scaled_14[4];
                scaled_14[0] = accum_0[56] * sa_lo_0 * sw_0[12];
                scaled_14[1] = accum_0[57] * sa_lo_0 * sw_0[13];
                scaled_14[2] = accum_0[58] * sa_hi_0 * sw_0[12];
                scaled_14[3] = accum_0[59] * sa_hi_0 * sw_0[13];
                unsigned int scaled_bf16_14[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_14[_lp*2 + 0], scaled_14[_lp*2+1 + 0]));
                    scaled_bf16_14[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_14[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_14[_pair]) << 16);
                    (&unpacked_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_14[_pair]) & 0xffff0000u);
                }
                rounded_0[24] = unpacked_14[0];
                rounded_0[25] = unpacked_14[1];
                rounded_0[26] = unpacked_14[2];
                rounded_0[27] = unpacked_14[3];
                ss_lo_0 += __fmaf_rn(unpacked_14[0], unpacked_14[0], unpacked_14[1] * unpacked_14[1]);
                ss_hi_0 += __fmaf_rn(unpacked_14[2], unpacked_14[2], unpacked_14[3] * unpacked_14[3]);
                float scaled_15[4];
                scaled_15[0] = accum_0[60] * sa_lo_0 * sw_0[14];
                scaled_15[1] = accum_0[61] * sa_lo_0 * sw_0[15];
                scaled_15[2] = accum_0[62] * sa_hi_0 * sw_0[14];
                scaled_15[3] = accum_0[63] * sa_hi_0 * sw_0[15];
                unsigned int scaled_bf16_15[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_15[_lp*2 + 0], scaled_15[_lp*2+1 + 0]));
                    scaled_bf16_15[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_15[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_15[_pair]) << 16);
                    (&unpacked_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_15[_pair]) & 0xffff0000u);
                }
                rounded_0[28] = unpacked_15[0];
                rounded_0[29] = unpacked_15[1];
                rounded_0[30] = unpacked_15[2];
                rounded_0[31] = unpacked_15[3];
                ss_lo_0 += __fmaf_rn(unpacked_15[0], unpacked_15[0], unpacked_15[1] * unpacked_15[1]);
                ss_hi_0 += __fmaf_rn(unpacked_15[2], unpacked_15[2], unpacked_15[3] * unpacked_15[3]);
                float rstd_lo_0 = 1.0f;
                float rstd_hi_0 = 1.0f;
                if (w_kind_0 < 2) {
                    float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, ss_lo_0, 1);
                    ss_lo_0 += _shfl_xor_sync_24;
                    float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, ss_lo_0, 2);
                    ss_lo_0 += _shfl_xor_sync_25;
                    float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, ss_hi_0, 1);
                    ss_hi_0 += _shfl_xor_sync_26;
                    float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, ss_hi_0, 2);
                    ss_hi_0 += _shfl_xor_sync_27;
                    int srow_lo_1 = warp_m_0 * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n_0 * 64 + srow_lo_1] = ss_lo_0;
                        rowstat[warp_n_0 * 64 + srow_lo_1 + 8] = ss_hi_0;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float other_lo_1 = rowstat[(1 - warp_n_0) * 64 + srow_lo_1];
                    float other_hi_1 = rowstat[(1 - warp_n_0) * 64 + srow_lo_1 + 8];
                    float _fdiv_29 = __fdiv_rn(ss_lo_0 + other_lo_1, 128.0f);
                    float _rsqrt_28 = rsqrtf(_fdiv_29 + eps);
                    rstd_lo_0 = _rsqrt_28;
                    float _fdiv_31 = __fdiv_rn(ss_hi_0 + other_hi_1, 128.0f);
                    float _rsqrt_30 = rsqrtf(_fdiv_31 + eps);
                    rstd_hi_0 = _rsqrt_30;
                }
                int srow_st_0 = warp_m_0 * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float normed_8[4];
                if (w_kind_0 < 2) {
                    normed_8[0] = rounded_0[0] * rstd_lo_0 * nw_0[0];
                    normed_8[1] = rounded_0[1] * rstd_lo_0 * nw_0[1];
                    normed_8[2] = rounded_0[2] * rstd_hi_0 * nw_0[0];
                    normed_8[3] = rounded_0[3] * rstd_hi_0 * nw_0[1];
                } else {
                    normed_8[0] = rounded_0[0];
                    normed_8[1] = rounded_0[1];
                    normed_8[2] = rounded_0[2];
                    normed_8[3] = rounded_0[3];
                }
                unsigned int normed_bf16_8[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_8[_lp*2 + 0], normed_8[_lp*2+1 + 0]));
                    normed_bf16_8[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_32 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_0 * 256) + (unsigned int)(warp_n_0 * 64 * 2 ^ (srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_32), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_8[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_8[1]))
                    : "memory");
                float normed_9[4];
                if (w_kind_0 < 2) {
                    normed_9[0] = rounded_0[4] * rstd_lo_0 * nw_0[2];
                    normed_9[1] = rounded_0[5] * rstd_lo_0 * nw_0[3];
                    normed_9[2] = rounded_0[6] * rstd_hi_0 * nw_0[2];
                    normed_9[3] = rounded_0[7] * rstd_hi_0 * nw_0[3];
                } else {
                    normed_9[0] = rounded_0[4];
                    normed_9[1] = rounded_0[5];
                    normed_9[2] = rounded_0[6];
                    normed_9[3] = rounded_0[7];
                }
                unsigned int normed_bf16_9[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_9[_lp*2 + 0], normed_9[_lp*2+1 + 0]));
                    normed_bf16_9[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_33 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_0 * 256) + (unsigned int)((warp_n_0 * 64 + 8) * 2 ^ (srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_33), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_9[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_9[1]))
                    : "memory");
                float normed_10[4];
                if (w_kind_0 < 2) {
                    normed_10[0] = rounded_0[8] * rstd_lo_0 * nw_0[4];
                    normed_10[1] = rounded_0[9] * rstd_lo_0 * nw_0[5];
                    normed_10[2] = rounded_0[10] * rstd_hi_0 * nw_0[4];
                    normed_10[3] = rounded_0[11] * rstd_hi_0 * nw_0[5];
                } else {
                    normed_10[0] = rounded_0[8];
                    normed_10[1] = rounded_0[9];
                    normed_10[2] = rounded_0[10];
                    normed_10[3] = rounded_0[11];
                }
                unsigned int normed_bf16_10[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_10[_lp*2 + 0], normed_10[_lp*2+1 + 0]));
                    normed_bf16_10[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_0 * 256) + (unsigned int)((warp_n_0 * 64 + 16) * 2 ^ (srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_10[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_10[1]))
                    : "memory");
                float normed_11[4];
                if (w_kind_0 < 2) {
                    normed_11[0] = rounded_0[12] * rstd_lo_0 * nw_0[6];
                    normed_11[1] = rounded_0[13] * rstd_lo_0 * nw_0[7];
                    normed_11[2] = rounded_0[14] * rstd_hi_0 * nw_0[6];
                    normed_11[3] = rounded_0[15] * rstd_hi_0 * nw_0[7];
                } else {
                    normed_11[0] = rounded_0[12];
                    normed_11[1] = rounded_0[13];
                    normed_11[2] = rounded_0[14];
                    normed_11[3] = rounded_0[15];
                }
                unsigned int normed_bf16_11[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_11[_lp*2 + 0], normed_11[_lp*2+1 + 0]));
                    normed_bf16_11[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_0 * 256) + (unsigned int)((warp_n_0 * 64 + 24) * 2 ^ (srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_11[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_11[1]))
                    : "memory");
                float normed_12[4];
                if (w_kind_0 < 2) {
                    normed_12[0] = rounded_0[16] * rstd_lo_0 * nw_0[8];
                    normed_12[1] = rounded_0[17] * rstd_lo_0 * nw_0[9];
                    normed_12[2] = rounded_0[18] * rstd_hi_0 * nw_0[8];
                    normed_12[3] = rounded_0[19] * rstd_hi_0 * nw_0[9];
                } else {
                    normed_12[0] = rounded_0[16];
                    normed_12[1] = rounded_0[17];
                    normed_12[2] = rounded_0[18];
                    normed_12[3] = rounded_0[19];
                }
                unsigned int normed_bf16_12[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_12[_lp*2 + 0], normed_12[_lp*2+1 + 0]));
                    normed_bf16_12[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_36 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_0 * 256) + (unsigned int)((warp_n_0 * 64 + 32) * 2 ^ (srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_36), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_12[1]))
                    : "memory");
                float normed_13[4];
                if (w_kind_0 < 2) {
                    normed_13[0] = rounded_0[20] * rstd_lo_0 * nw_0[10];
                    normed_13[1] = rounded_0[21] * rstd_lo_0 * nw_0[11];
                    normed_13[2] = rounded_0[22] * rstd_hi_0 * nw_0[10];
                    normed_13[3] = rounded_0[23] * rstd_hi_0 * nw_0[11];
                } else {
                    normed_13[0] = rounded_0[20];
                    normed_13[1] = rounded_0[21];
                    normed_13[2] = rounded_0[22];
                    normed_13[3] = rounded_0[23];
                }
                unsigned int normed_bf16_13[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_13[_lp*2 + 0], normed_13[_lp*2+1 + 0]));
                    normed_bf16_13[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_37 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_0 * 256) + (unsigned int)((warp_n_0 * 64 + 40) * 2 ^ (srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_37), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_13[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_13[1]))
                    : "memory");
                float normed_14[4];
                if (w_kind_0 < 2) {
                    normed_14[0] = rounded_0[24] * rstd_lo_0 * nw_0[12];
                    normed_14[1] = rounded_0[25] * rstd_lo_0 * nw_0[13];
                    normed_14[2] = rounded_0[26] * rstd_hi_0 * nw_0[12];
                    normed_14[3] = rounded_0[27] * rstd_hi_0 * nw_0[13];
                } else {
                    normed_14[0] = rounded_0[24];
                    normed_14[1] = rounded_0[25];
                    normed_14[2] = rounded_0[26];
                    normed_14[3] = rounded_0[27];
                }
                unsigned int normed_bf16_14[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_14[_lp*2 + 0], normed_14[_lp*2+1 + 0]));
                    normed_bf16_14[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_38 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_0 * 256) + (unsigned int)((warp_n_0 * 64 + 48) * 2 ^ (srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_38), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_14[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_14[1]))
                    : "memory");
                float normed_15[4];
                if (w_kind_0 < 2) {
                    normed_15[0] = rounded_0[28] * rstd_lo_0 * nw_0[14];
                    normed_15[1] = rounded_0[29] * rstd_lo_0 * nw_0[15];
                    normed_15[2] = rounded_0[30] * rstd_hi_0 * nw_0[14];
                    normed_15[3] = rounded_0[31] * rstd_hi_0 * nw_0[15];
                } else {
                    normed_15[0] = rounded_0[28];
                    normed_15[1] = rounded_0[29];
                    normed_15[2] = rounded_0[30];
                    normed_15[3] = rounded_0[31];
                }
                unsigned int normed_bf16_15[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_15[_lp*2 + 0], normed_15[_lp*2+1 + 0]));
                    normed_bf16_15[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_39 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_0 * 256) + (unsigned int)((warp_n_0 * 64 + 56) * 2 ^ (srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_39), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_15[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_15[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int srow_0 = role_tid_0 >> 2;
                int c_0 = role_tid_0 & 3;
                int grow_0 = tile_m_0 * 256 + 64 + srow_0;
                int chunk_lo_0 = ((c_0 < 3) ? c_0 * 16 : 96);
                int chunk_hi_0 = ((c_0 < 3) ? chunk_lo_0 + 48 : 112);
                if (grow_0 < M) {
                    unsigned int x_lo_1[8];
                    unsigned int x_hi_1[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_1[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_0 * 256) + (unsigned int)(chunk_lo_0 * 2 ^ (srow_0 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_1[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_0 * 256) + (unsigned int)(chunk_lo_0 * 2 + 16 ^ (srow_0 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_1[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_0 * 256) + (unsigned int)(chunk_hi_0 * 2 ^ (srow_0 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_1[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_0 * 256) + (unsigned int)(chunk_hi_0 * 2 + 16 ^ (srow_0 & 7) << 4)));
                    if (c_0 < 3) {
                        if (w_kind_0 < 2) {
                            float x_lo_f_1[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_lo_f_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_1[_pair]) << 16);
                                (&x_lo_f_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_1[_pair]) & 0xffff0000u);
                            }
                            float x_hi_f_1[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_hi_f_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_1[_pair]) << 16);
                                (&x_hi_f_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_1[_pair]) & 0xffff0000u);
                            }
                            float v_4[8];
                            {
                                const uint4* _vptr_40 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_0 * 96 + chunk_lo_0) + 0);
                                uint4 _vld_40[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_40[_blk] = _vptr_40[_blk];
                                    uint32_t* _vpairs_40 = reinterpret_cast<uint32_t*>(&_vld_40[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) << 16);
                                        (&v_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_5[8];
                            {
                                const uint4* _vptr_41 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_0 * 96 + chunk_lo_0 + 8) + 0);
                                uint4 _vld_41[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_41[_blk] = _vptr_41[_blk];
                                    uint32_t* _vpairs_41 = reinterpret_cast<uint32_t*>(&_vld_41[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) << 16);
                                        (&v_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_6[8];
                            {
                                const uint4* _vptr_42 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_0 * 96 + 48 + chunk_lo_0) + 0);
                                uint4 _vld_42[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_42[_blk] = _vptr_42[_blk];
                                    uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
                                        (&v_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_7[8];
                            {
                                const uint4* _vptr_43 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_0 * 96 + 48 + chunk_lo_0 + 8) + 0);
                                uint4 _vld_43[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_43[_blk] = _vptr_43[_blk];
                                    uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
                                        (&v_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float o_lo_1[16];
                            float o_hi_1[16];
                            float cv0_7 = v_4[0];
                            float sv0_7 = v_6[0];
                            float cv1_7 = v_5[0];
                            float sv1_7 = v_7[0];
                            o_lo_1[0] = __fmaf_rn(x_lo_f_1[0], cv0_7, -(x_hi_f_1[0] * sv0_7));
                            o_hi_1[0] = __fmaf_rn(x_hi_f_1[0], cv0_7, x_lo_f_1[0] * sv0_7);
                            o_lo_1[8] = __fmaf_rn(x_lo_f_1[8], cv1_7, -(x_hi_f_1[8] * sv1_7));
                            o_hi_1[8] = __fmaf_rn(x_hi_f_1[8], cv1_7, x_lo_f_1[8] * sv1_7);
                            float cv0_7_1 = v_4[1];
                            float sv0_7_1 = v_6[1];
                            float cv1_7_1 = v_5[1];
                            float sv1_7_1 = v_7[1];
                            o_lo_1[1] = __fmaf_rn(x_lo_f_1[1], cv0_7_1, -(x_hi_f_1[1] * sv0_7_1));
                            o_hi_1[1] = __fmaf_rn(x_hi_f_1[1], cv0_7_1, x_lo_f_1[1] * sv0_7_1);
                            o_lo_1[9] = __fmaf_rn(x_lo_f_1[9], cv1_7_1, -(x_hi_f_1[9] * sv1_7_1));
                            o_hi_1[9] = __fmaf_rn(x_hi_f_1[9], cv1_7_1, x_lo_f_1[9] * sv1_7_1);
                            float cv0_8 = v_4[2];
                            float sv0_8 = v_6[2];
                            float cv1_8 = v_5[2];
                            float sv1_8 = v_7[2];
                            o_lo_1[2] = __fmaf_rn(x_lo_f_1[2], cv0_8, -(x_hi_f_1[2] * sv0_8));
                            o_hi_1[2] = __fmaf_rn(x_hi_f_1[2], cv0_8, x_lo_f_1[2] * sv0_8);
                            o_lo_1[10] = __fmaf_rn(x_lo_f_1[10], cv1_8, -(x_hi_f_1[10] * sv1_8));
                            o_hi_1[10] = __fmaf_rn(x_hi_f_1[10], cv1_8, x_lo_f_1[10] * sv1_8);
                            float cv0_9 = v_4[3];
                            float sv0_9 = v_6[3];
                            float cv1_9 = v_5[3];
                            float sv1_9 = v_7[3];
                            o_lo_1[3] = __fmaf_rn(x_lo_f_1[3], cv0_9, -(x_hi_f_1[3] * sv0_9));
                            o_hi_1[3] = __fmaf_rn(x_hi_f_1[3], cv0_9, x_lo_f_1[3] * sv0_9);
                            o_lo_1[11] = __fmaf_rn(x_lo_f_1[11], cv1_9, -(x_hi_f_1[11] * sv1_9));
                            o_hi_1[11] = __fmaf_rn(x_hi_f_1[11], cv1_9, x_lo_f_1[11] * sv1_9);
                            float cv0_10 = v_4[4];
                            float sv0_10 = v_6[4];
                            float cv1_10 = v_5[4];
                            float sv1_10 = v_7[4];
                            o_lo_1[4] = __fmaf_rn(x_lo_f_1[4], cv0_10, -(x_hi_f_1[4] * sv0_10));
                            o_hi_1[4] = __fmaf_rn(x_hi_f_1[4], cv0_10, x_lo_f_1[4] * sv0_10);
                            o_lo_1[12] = __fmaf_rn(x_lo_f_1[12], cv1_10, -(x_hi_f_1[12] * sv1_10));
                            o_hi_1[12] = __fmaf_rn(x_hi_f_1[12], cv1_10, x_lo_f_1[12] * sv1_10);
                            float cv0_11 = v_4[5];
                            float sv0_11 = v_6[5];
                            float cv1_11 = v_5[5];
                            float sv1_11 = v_7[5];
                            o_lo_1[5] = __fmaf_rn(x_lo_f_1[5], cv0_11, -(x_hi_f_1[5] * sv0_11));
                            o_hi_1[5] = __fmaf_rn(x_hi_f_1[5], cv0_11, x_lo_f_1[5] * sv0_11);
                            o_lo_1[13] = __fmaf_rn(x_lo_f_1[13], cv1_11, -(x_hi_f_1[13] * sv1_11));
                            o_hi_1[13] = __fmaf_rn(x_hi_f_1[13], cv1_11, x_lo_f_1[13] * sv1_11);
                            float cv0_12 = v_4[6];
                            float sv0_12 = v_6[6];
                            float cv1_12 = v_5[6];
                            float sv1_12 = v_7[6];
                            o_lo_1[6] = __fmaf_rn(x_lo_f_1[6], cv0_12, -(x_hi_f_1[6] * sv0_12));
                            o_hi_1[6] = __fmaf_rn(x_hi_f_1[6], cv0_12, x_lo_f_1[6] * sv0_12);
                            o_lo_1[14] = __fmaf_rn(x_lo_f_1[14], cv1_12, -(x_hi_f_1[14] * sv1_12));
                            o_hi_1[14] = __fmaf_rn(x_hi_f_1[14], cv1_12, x_lo_f_1[14] * sv1_12);
                            float cv0_13 = v_4[7];
                            float sv0_13 = v_6[7];
                            float cv1_13 = v_5[7];
                            float sv1_13 = v_7[7];
                            o_lo_1[7] = __fmaf_rn(x_lo_f_1[7], cv0_13, -(x_hi_f_1[7] * sv0_13));
                            o_hi_1[7] = __fmaf_rn(x_hi_f_1[7], cv0_13, x_lo_f_1[7] * sv0_13);
                            o_lo_1[15] = __fmaf_rn(x_lo_f_1[15], cv1_13, -(x_hi_f_1[15] * sv1_13));
                            o_hi_1[15] = __fmaf_rn(x_hi_f_1[15], cv1_13, x_lo_f_1[15] * sv1_13);
                            unsigned int o_lo_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_lo_1[_lp*2 + 0], o_lo_1[_lp*2+1 + 0]));
                                o_lo_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int o_hi_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_hi_1[_lp*2 + 0], o_hi_1[_lp*2+1 + 0]));
                                o_hi_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            x_lo_1[0] = o_lo_bf16_1[0];
                            x_hi_1[0] = o_hi_bf16_1[0];
                            x_lo_1[1] = o_lo_bf16_1[1];
                            x_hi_1[1] = o_hi_bf16_1[1];
                            x_lo_1[2] = o_lo_bf16_1[2];
                            x_hi_1[2] = o_hi_bf16_1[2];
                            x_lo_1[3] = o_lo_bf16_1[3];
                            x_hi_1[3] = o_hi_bf16_1[3];
                            x_lo_1[4] = o_lo_bf16_1[4];
                            x_hi_1[4] = o_hi_bf16_1[4];
                            x_lo_1[5] = o_lo_bf16_1[5];
                            x_hi_1[5] = o_hi_bf16_1[5];
                            x_lo_1[6] = o_lo_bf16_1[6];
                            x_hi_1[6] = o_hi_bf16_1[6];
                            x_lo_1[7] = o_lo_bf16_1[7];
                            x_hi_1[7] = o_hi_bf16_1[7];
                        }
                    }
                    unsigned int q_words_1[4];
                    float q_vals_1[16];
                    float f_2[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_1[_pair]) << 16);
                        (&f_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_1[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_2;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_2) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_2[0]), fabsf(f_2[1])), fabsf(f_2[2])), fabsf(f_2[3])), fabsf(f_2[4])), fabsf(f_2[5])), fabsf(f_2[6])), fabsf(f_2[7])), fabsf(f_2[8])), fabsf(f_2[9])), fabsf(f_2[10])), fabsf(f_2[11])), fabsf(f_2[12])), fabsf(f_2[13])), fabsf(f_2[14])), fabsf(f_2[15])) * sf_mul_sel_0));
                    uint16_t _e4m3x2_decode_44 = (uint16_t)((unsigned int)sf_pair_2 & 0xFFu);
                    uint32_t _f16x2_decode_44;
                    float sf_dec_2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_44) : "h"(_e4m3x2_decode_44));
                    uint16_t _f16_decode_44 = (uint16_t)_f16x2_decode_44;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_2) : "h"(_f16_decode_44));
                    float out_s_2 = __fdiv_rn(out_scale_sel_0, sf_dec_2);
                    q_vals_1[0] = f_2[0] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[1] = f_2[1] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[2] = f_2[2] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[3] = f_2[3] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[4] = f_2[4] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[5] = f_2[5] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[6] = f_2[6] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[7] = f_2[7] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[8] = f_2[8] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[9] = f_2[9] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[10] = f_2[10] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[11] = f_2[11] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[12] = f_2[12] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[13] = f_2[13] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[14] = f_2[14] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    q_vals_1[15] = f_2[15] * ((sf_dec_2 != 0.0f) ? out_s_2 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_1[0]) : "f"(q_vals_1[0]), "f"(q_vals_1[1]), "f"(q_vals_1[2]), "f"(q_vals_1[3]), "f"(q_vals_1[4]), "f"(q_vals_1[5]), "f"(q_vals_1[6]), "f"(q_vals_1[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_1[1]) : "f"(q_vals_1[8]), "f"(q_vals_1[9]), "f"(q_vals_1[10]), "f"(q_vals_1[11]), "f"(q_vals_1[12]), "f"(q_vals_1[13]), "f"(q_vals_1[14]), "f"(q_vals_1[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow_0 * 128) + (unsigned int)(chunk_lo_0 >> 1 ^ (srow_0 & 7) << 4)), "r"(q_words_1[0]), "r"(q_words_1[1]) : "memory");
                    {
                        uint32_t _byte_45 = static_cast<uint32_t>((unsigned int)sf_pair_2 & 127) & 0xFFu;
                        uint32_t _addr_45 = static_cast<uint32_t>(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow_0 * 128) + (unsigned int)(64 + (chunk_lo_0 >> 4) ^ (srow_0 & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_45), "r"(_byte_45) : "memory");
                    }
                    float f_3[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_1[_pair]) << 16);
                        (&f_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_1[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_3;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_3) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_3[0]), fabsf(f_3[1])), fabsf(f_3[2])), fabsf(f_3[3])), fabsf(f_3[4])), fabsf(f_3[5])), fabsf(f_3[6])), fabsf(f_3[7])), fabsf(f_3[8])), fabsf(f_3[9])), fabsf(f_3[10])), fabsf(f_3[11])), fabsf(f_3[12])), fabsf(f_3[13])), fabsf(f_3[14])), fabsf(f_3[15])) * sf_mul_sel_0));
                    uint16_t _e4m3x2_decode_46 = (uint16_t)((unsigned int)sf_pair_3 & 0xFFu);
                    uint32_t _f16x2_decode_46;
                    float sf_dec_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_46) : "h"(_e4m3x2_decode_46));
                    uint16_t _f16_decode_46 = (uint16_t)_f16x2_decode_46;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_3) : "h"(_f16_decode_46));
                    float out_s_3 = __fdiv_rn(out_scale_sel_0, sf_dec_3);
                    q_vals_1[0] = f_3[0] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[1] = f_3[1] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[2] = f_3[2] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[3] = f_3[3] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[4] = f_3[4] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[5] = f_3[5] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[6] = f_3[6] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[7] = f_3[7] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[8] = f_3[8] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[9] = f_3[9] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[10] = f_3[10] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[11] = f_3[11] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[12] = f_3[12] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[13] = f_3[13] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[14] = f_3[14] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    q_vals_1[15] = f_3[15] * ((sf_dec_3 != 0.0f) ? out_s_3 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_1[0]) : "f"(q_vals_1[0]), "f"(q_vals_1[1]), "f"(q_vals_1[2]), "f"(q_vals_1[3]), "f"(q_vals_1[4]), "f"(q_vals_1[5]), "f"(q_vals_1[6]), "f"(q_vals_1[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_1[1]) : "f"(q_vals_1[8]), "f"(q_vals_1[9]), "f"(q_vals_1[10]), "f"(q_vals_1[11]), "f"(q_vals_1[12]), "f"(q_vals_1[13]), "f"(q_vals_1[14]), "f"(q_vals_1[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow_0 * 128) + (unsigned int)(chunk_hi_0 >> 1 ^ (srow_0 & 7) << 4)), "r"(q_words_1[0]), "r"(q_words_1[1]) : "memory");
                    {
                        uint32_t _byte_47 = static_cast<uint32_t>((unsigned int)sf_pair_3 & 127) & 0xFFu;
                        uint32_t _addr_47 = static_cast<uint32_t>(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow_0 * 128) + (unsigned int)(64 + (chunk_hi_0 >> 4) ^ (srow_0 & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_47), "r"(_byte_47) : "memory");
                    }
                }
                __syncwarp();
                int s_row_1 = (role_tid_0 >> 5) * 8 + lane / 4;
                int s_piece_1 = lane % 4;
                unsigned int s_words_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(s_row_1 * 128) + (unsigned int)(s_piece_1 * 16 ^ (s_row_1 & 7) << 4)));
                int s_dst_1 = ((tile_m_0 * 256 + 64 + s_row_1) * 56 + w_head_0) * 16 + s_piece_1 * 4;
                if (tile_m_0 * 256 + 64 + s_row_1 < M) {
                    if (w_kind_0 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_1)[0] = reinterpret_cast<int4*>(s_words_1)[0];
                    } else if (w_kind_0 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_1)[0] = reinterpret_cast<int4*>(s_words_1)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_1)[0] = reinterpret_cast<int4*>(s_words_1)[0];
                    }
                }
                int s_sf_row_0 = (role_tid_0 >> 5) * 8 + (lane & 7);
                unsigned int s_sf_words_0[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words_0[(0) + 1]))
                    : "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(s_sf_row_0 * 128) + (unsigned int)(64 ^ (s_sf_row_0 & 7) << 4)));
                int s_sf_dst_0 = ((tile_m_0 * 256 + 64 + s_sf_row_0) * 56 + w_head_0) * 8;
                if (lane < 8) {
                    if (tile_m_0 * 256 + 64 + s_sf_row_0 < M) {
                        if (w_kind_0 == 0) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(q_sf + s_sf_dst_0))[0] = reinterpret_cast<int2*>(s_sf_words_0)[0];
                        } else if (w_kind_0 == 1) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(k_sf + s_sf_dst_0))[0] = reinterpret_cast<int2*>(s_sf_words_0)[0];
                        } else {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(v_sf + s_sf_dst_0))[0] = reinterpret_cast<int2*>(s_sf_words_0)[0];
                        }
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int row_lo_1 = tile_m_0 * 256 + 128 + warp_m_0 * 16 + (lane >> 2);
                int row_hi_1 = row_lo_1 + 8;
                float sa_lo_1 = ((row_lo_1 < M) ? act_scale[row_lo_1] : 0.0f);
                float sa_hi_1 = ((row_hi_1 < M) ? act_scale[row_hi_1] : 0.0f);
                float rounded_1[32];
                float ss_lo_1 = 0.0f;
                float ss_hi_1 = 0.0f;
                float scaled_16[4];
                scaled_16[0] = accum_0[64] * sa_lo_1 * sw_0[0];
                scaled_16[1] = accum_0[65] * sa_lo_1 * sw_0[1];
                scaled_16[2] = accum_0[66] * sa_hi_1 * sw_0[0];
                scaled_16[3] = accum_0[67] * sa_hi_1 * sw_0[1];
                unsigned int scaled_bf16_16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_16[_lp*2 + 0], scaled_16[_lp*2+1 + 0]));
                    scaled_bf16_16[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_16[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_16[_pair]) << 16);
                    (&unpacked_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_16[_pair]) & 0xffff0000u);
                }
                rounded_1[0] = unpacked_16[0];
                rounded_1[1] = unpacked_16[1];
                rounded_1[2] = unpacked_16[2];
                rounded_1[3] = unpacked_16[3];
                ss_lo_1 += __fmaf_rn(unpacked_16[0], unpacked_16[0], unpacked_16[1] * unpacked_16[1]);
                ss_hi_1 += __fmaf_rn(unpacked_16[2], unpacked_16[2], unpacked_16[3] * unpacked_16[3]);
                float scaled_17[4];
                scaled_17[0] = accum_0[68] * sa_lo_1 * sw_0[2];
                scaled_17[1] = accum_0[69] * sa_lo_1 * sw_0[3];
                scaled_17[2] = accum_0[70] * sa_hi_1 * sw_0[2];
                scaled_17[3] = accum_0[71] * sa_hi_1 * sw_0[3];
                unsigned int scaled_bf16_17[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_17[_lp*2 + 0], scaled_17[_lp*2+1 + 0]));
                    scaled_bf16_17[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_17[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_17[_pair]) << 16);
                    (&unpacked_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_17[_pair]) & 0xffff0000u);
                }
                rounded_1[4] = unpacked_17[0];
                rounded_1[5] = unpacked_17[1];
                rounded_1[6] = unpacked_17[2];
                rounded_1[7] = unpacked_17[3];
                ss_lo_1 += __fmaf_rn(unpacked_17[0], unpacked_17[0], unpacked_17[1] * unpacked_17[1]);
                ss_hi_1 += __fmaf_rn(unpacked_17[2], unpacked_17[2], unpacked_17[3] * unpacked_17[3]);
                float scaled_18[4];
                scaled_18[0] = accum_0[72] * sa_lo_1 * sw_0[4];
                scaled_18[1] = accum_0[73] * sa_lo_1 * sw_0[5];
                scaled_18[2] = accum_0[74] * sa_hi_1 * sw_0[4];
                scaled_18[3] = accum_0[75] * sa_hi_1 * sw_0[5];
                unsigned int scaled_bf16_18[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_18[_lp*2 + 0], scaled_18[_lp*2+1 + 0]));
                    scaled_bf16_18[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_18[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_18[_pair]) << 16);
                    (&unpacked_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_18[_pair]) & 0xffff0000u);
                }
                rounded_1[8] = unpacked_18[0];
                rounded_1[9] = unpacked_18[1];
                rounded_1[10] = unpacked_18[2];
                rounded_1[11] = unpacked_18[3];
                ss_lo_1 += __fmaf_rn(unpacked_18[0], unpacked_18[0], unpacked_18[1] * unpacked_18[1]);
                ss_hi_1 += __fmaf_rn(unpacked_18[2], unpacked_18[2], unpacked_18[3] * unpacked_18[3]);
                float scaled_19[4];
                scaled_19[0] = accum_0[76] * sa_lo_1 * sw_0[6];
                scaled_19[1] = accum_0[77] * sa_lo_1 * sw_0[7];
                scaled_19[2] = accum_0[78] * sa_hi_1 * sw_0[6];
                scaled_19[3] = accum_0[79] * sa_hi_1 * sw_0[7];
                unsigned int scaled_bf16_19[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_19[_lp*2 + 0], scaled_19[_lp*2+1 + 0]));
                    scaled_bf16_19[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_19[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_19[_pair]) << 16);
                    (&unpacked_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_19[_pair]) & 0xffff0000u);
                }
                rounded_1[12] = unpacked_19[0];
                rounded_1[13] = unpacked_19[1];
                rounded_1[14] = unpacked_19[2];
                rounded_1[15] = unpacked_19[3];
                ss_lo_1 += __fmaf_rn(unpacked_19[0], unpacked_19[0], unpacked_19[1] * unpacked_19[1]);
                ss_hi_1 += __fmaf_rn(unpacked_19[2], unpacked_19[2], unpacked_19[3] * unpacked_19[3]);
                float scaled_20[4];
                scaled_20[0] = accum_0[80] * sa_lo_1 * sw_0[8];
                scaled_20[1] = accum_0[81] * sa_lo_1 * sw_0[9];
                scaled_20[2] = accum_0[82] * sa_hi_1 * sw_0[8];
                scaled_20[3] = accum_0[83] * sa_hi_1 * sw_0[9];
                unsigned int scaled_bf16_20[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_20[_lp*2 + 0], scaled_20[_lp*2+1 + 0]));
                    scaled_bf16_20[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_20[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_20[_pair]) << 16);
                    (&unpacked_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_20[_pair]) & 0xffff0000u);
                }
                rounded_1[16] = unpacked_20[0];
                rounded_1[17] = unpacked_20[1];
                rounded_1[18] = unpacked_20[2];
                rounded_1[19] = unpacked_20[3];
                ss_lo_1 += __fmaf_rn(unpacked_20[0], unpacked_20[0], unpacked_20[1] * unpacked_20[1]);
                ss_hi_1 += __fmaf_rn(unpacked_20[2], unpacked_20[2], unpacked_20[3] * unpacked_20[3]);
                float scaled_21[4];
                scaled_21[0] = accum_0[84] * sa_lo_1 * sw_0[10];
                scaled_21[1] = accum_0[85] * sa_lo_1 * sw_0[11];
                scaled_21[2] = accum_0[86] * sa_hi_1 * sw_0[10];
                scaled_21[3] = accum_0[87] * sa_hi_1 * sw_0[11];
                unsigned int scaled_bf16_21[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_21[_lp*2 + 0], scaled_21[_lp*2+1 + 0]));
                    scaled_bf16_21[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_21[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_21[_pair]) << 16);
                    (&unpacked_21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_21[_pair]) & 0xffff0000u);
                }
                rounded_1[20] = unpacked_21[0];
                rounded_1[21] = unpacked_21[1];
                rounded_1[22] = unpacked_21[2];
                rounded_1[23] = unpacked_21[3];
                ss_lo_1 += __fmaf_rn(unpacked_21[0], unpacked_21[0], unpacked_21[1] * unpacked_21[1]);
                ss_hi_1 += __fmaf_rn(unpacked_21[2], unpacked_21[2], unpacked_21[3] * unpacked_21[3]);
                float scaled_22[4];
                scaled_22[0] = accum_0[88] * sa_lo_1 * sw_0[12];
                scaled_22[1] = accum_0[89] * sa_lo_1 * sw_0[13];
                scaled_22[2] = accum_0[90] * sa_hi_1 * sw_0[12];
                scaled_22[3] = accum_0[91] * sa_hi_1 * sw_0[13];
                unsigned int scaled_bf16_22[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_22[_lp*2 + 0], scaled_22[_lp*2+1 + 0]));
                    scaled_bf16_22[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_22[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_22[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_22[_pair]) << 16);
                    (&unpacked_22[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_22[_pair]) & 0xffff0000u);
                }
                rounded_1[24] = unpacked_22[0];
                rounded_1[25] = unpacked_22[1];
                rounded_1[26] = unpacked_22[2];
                rounded_1[27] = unpacked_22[3];
                ss_lo_1 += __fmaf_rn(unpacked_22[0], unpacked_22[0], unpacked_22[1] * unpacked_22[1]);
                ss_hi_1 += __fmaf_rn(unpacked_22[2], unpacked_22[2], unpacked_22[3] * unpacked_22[3]);
                float scaled_23[4];
                scaled_23[0] = accum_0[92] * sa_lo_1 * sw_0[14];
                scaled_23[1] = accum_0[93] * sa_lo_1 * sw_0[15];
                scaled_23[2] = accum_0[94] * sa_hi_1 * sw_0[14];
                scaled_23[3] = accum_0[95] * sa_hi_1 * sw_0[15];
                unsigned int scaled_bf16_23[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_23[_lp*2 + 0], scaled_23[_lp*2+1 + 0]));
                    scaled_bf16_23[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_23[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_23[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_23[_pair]) << 16);
                    (&unpacked_23[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_23[_pair]) & 0xffff0000u);
                }
                rounded_1[28] = unpacked_23[0];
                rounded_1[29] = unpacked_23[1];
                rounded_1[30] = unpacked_23[2];
                rounded_1[31] = unpacked_23[3];
                ss_lo_1 += __fmaf_rn(unpacked_23[0], unpacked_23[0], unpacked_23[1] * unpacked_23[1]);
                ss_hi_1 += __fmaf_rn(unpacked_23[2], unpacked_23[2], unpacked_23[3] * unpacked_23[3]);
                float rstd_lo_1 = 1.0f;
                float rstd_hi_1 = 1.0f;
                if (w_kind_0 < 2) {
                    float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, ss_lo_1, 1);
                    ss_lo_1 += _shfl_xor_sync_48;
                    float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, ss_lo_1, 2);
                    ss_lo_1 += _shfl_xor_sync_49;
                    float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, ss_hi_1, 1);
                    ss_hi_1 += _shfl_xor_sync_50;
                    float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, ss_hi_1, 2);
                    ss_hi_1 += _shfl_xor_sync_51;
                    int srow_lo_2 = warp_m_0 * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n_0 * 64 + srow_lo_2] = ss_lo_1;
                        rowstat[warp_n_0 * 64 + srow_lo_2 + 8] = ss_hi_1;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float other_lo_2 = rowstat[(1 - warp_n_0) * 64 + srow_lo_2];
                    float other_hi_2 = rowstat[(1 - warp_n_0) * 64 + srow_lo_2 + 8];
                    float _fdiv_53 = __fdiv_rn(ss_lo_1 + other_lo_2, 128.0f);
                    float _rsqrt_52 = rsqrtf(_fdiv_53 + eps);
                    rstd_lo_1 = _rsqrt_52;
                    float _fdiv_55 = __fdiv_rn(ss_hi_1 + other_hi_2, 128.0f);
                    float _rsqrt_54 = rsqrtf(_fdiv_55 + eps);
                    rstd_hi_1 = _rsqrt_54;
                }
                int srow_st_1 = warp_m_0 * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float normed_16[4];
                if (w_kind_0 < 2) {
                    normed_16[0] = rounded_1[0] * rstd_lo_1 * nw_0[0];
                    normed_16[1] = rounded_1[1] * rstd_lo_1 * nw_0[1];
                    normed_16[2] = rounded_1[2] * rstd_hi_1 * nw_0[0];
                    normed_16[3] = rounded_1[3] * rstd_hi_1 * nw_0[1];
                } else {
                    normed_16[0] = rounded_1[0];
                    normed_16[1] = rounded_1[1];
                    normed_16[2] = rounded_1[2];
                    normed_16[3] = rounded_1[3];
                }
                unsigned int normed_bf16_16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_16[_lp*2 + 0], normed_16[_lp*2+1 + 0]));
                    normed_bf16_16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_56 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_1 * 256) + (unsigned int)(warp_n_0 * 64 * 2 ^ (srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_56), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_16[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_16[1]))
                    : "memory");
                float normed_17[4];
                if (w_kind_0 < 2) {
                    normed_17[0] = rounded_1[4] * rstd_lo_1 * nw_0[2];
                    normed_17[1] = rounded_1[5] * rstd_lo_1 * nw_0[3];
                    normed_17[2] = rounded_1[6] * rstd_hi_1 * nw_0[2];
                    normed_17[3] = rounded_1[7] * rstd_hi_1 * nw_0[3];
                } else {
                    normed_17[0] = rounded_1[4];
                    normed_17[1] = rounded_1[5];
                    normed_17[2] = rounded_1[6];
                    normed_17[3] = rounded_1[7];
                }
                unsigned int normed_bf16_17[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_17[_lp*2 + 0], normed_17[_lp*2+1 + 0]));
                    normed_bf16_17[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_57 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_1 * 256) + (unsigned int)((warp_n_0 * 64 + 8) * 2 ^ (srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_57), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_17[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_17[1]))
                    : "memory");
                float normed_18[4];
                if (w_kind_0 < 2) {
                    normed_18[0] = rounded_1[8] * rstd_lo_1 * nw_0[4];
                    normed_18[1] = rounded_1[9] * rstd_lo_1 * nw_0[5];
                    normed_18[2] = rounded_1[10] * rstd_hi_1 * nw_0[4];
                    normed_18[3] = rounded_1[11] * rstd_hi_1 * nw_0[5];
                } else {
                    normed_18[0] = rounded_1[8];
                    normed_18[1] = rounded_1[9];
                    normed_18[2] = rounded_1[10];
                    normed_18[3] = rounded_1[11];
                }
                unsigned int normed_bf16_18[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_18[_lp*2 + 0], normed_18[_lp*2+1 + 0]));
                    normed_bf16_18[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_58 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_1 * 256) + (unsigned int)((warp_n_0 * 64 + 16) * 2 ^ (srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_58), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_18[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_18[1]))
                    : "memory");
                float normed_19[4];
                if (w_kind_0 < 2) {
                    normed_19[0] = rounded_1[12] * rstd_lo_1 * nw_0[6];
                    normed_19[1] = rounded_1[13] * rstd_lo_1 * nw_0[7];
                    normed_19[2] = rounded_1[14] * rstd_hi_1 * nw_0[6];
                    normed_19[3] = rounded_1[15] * rstd_hi_1 * nw_0[7];
                } else {
                    normed_19[0] = rounded_1[12];
                    normed_19[1] = rounded_1[13];
                    normed_19[2] = rounded_1[14];
                    normed_19[3] = rounded_1[15];
                }
                unsigned int normed_bf16_19[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_19[_lp*2 + 0], normed_19[_lp*2+1 + 0]));
                    normed_bf16_19[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_59 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_1 * 256) + (unsigned int)((warp_n_0 * 64 + 24) * 2 ^ (srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_59), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_19[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_19[1]))
                    : "memory");
                float normed_20[4];
                if (w_kind_0 < 2) {
                    normed_20[0] = rounded_1[16] * rstd_lo_1 * nw_0[8];
                    normed_20[1] = rounded_1[17] * rstd_lo_1 * nw_0[9];
                    normed_20[2] = rounded_1[18] * rstd_hi_1 * nw_0[8];
                    normed_20[3] = rounded_1[19] * rstd_hi_1 * nw_0[9];
                } else {
                    normed_20[0] = rounded_1[16];
                    normed_20[1] = rounded_1[17];
                    normed_20[2] = rounded_1[18];
                    normed_20[3] = rounded_1[19];
                }
                unsigned int normed_bf16_20[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_20[_lp*2 + 0], normed_20[_lp*2+1 + 0]));
                    normed_bf16_20[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_60 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_1 * 256) + (unsigned int)((warp_n_0 * 64 + 32) * 2 ^ (srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_60), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_20[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_20[1]))
                    : "memory");
                float normed_21[4];
                if (w_kind_0 < 2) {
                    normed_21[0] = rounded_1[20] * rstd_lo_1 * nw_0[10];
                    normed_21[1] = rounded_1[21] * rstd_lo_1 * nw_0[11];
                    normed_21[2] = rounded_1[22] * rstd_hi_1 * nw_0[10];
                    normed_21[3] = rounded_1[23] * rstd_hi_1 * nw_0[11];
                } else {
                    normed_21[0] = rounded_1[20];
                    normed_21[1] = rounded_1[21];
                    normed_21[2] = rounded_1[22];
                    normed_21[3] = rounded_1[23];
                }
                unsigned int normed_bf16_21[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_21[_lp*2 + 0], normed_21[_lp*2+1 + 0]));
                    normed_bf16_21[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_61 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_1 * 256) + (unsigned int)((warp_n_0 * 64 + 40) * 2 ^ (srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_61), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_21[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_21[1]))
                    : "memory");
                float normed_22[4];
                if (w_kind_0 < 2) {
                    normed_22[0] = rounded_1[24] * rstd_lo_1 * nw_0[12];
                    normed_22[1] = rounded_1[25] * rstd_lo_1 * nw_0[13];
                    normed_22[2] = rounded_1[26] * rstd_hi_1 * nw_0[12];
                    normed_22[3] = rounded_1[27] * rstd_hi_1 * nw_0[13];
                } else {
                    normed_22[0] = rounded_1[24];
                    normed_22[1] = rounded_1[25];
                    normed_22[2] = rounded_1[26];
                    normed_22[3] = rounded_1[27];
                }
                unsigned int normed_bf16_22[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_22[_lp*2 + 0], normed_22[_lp*2+1 + 0]));
                    normed_bf16_22[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_62 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_1 * 256) + (unsigned int)((warp_n_0 * 64 + 48) * 2 ^ (srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_62), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_22[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_22[1]))
                    : "memory");
                float normed_23[4];
                if (w_kind_0 < 2) {
                    normed_23[0] = rounded_1[28] * rstd_lo_1 * nw_0[14];
                    normed_23[1] = rounded_1[29] * rstd_lo_1 * nw_0[15];
                    normed_23[2] = rounded_1[30] * rstd_hi_1 * nw_0[14];
                    normed_23[3] = rounded_1[31] * rstd_hi_1 * nw_0[15];
                } else {
                    normed_23[0] = rounded_1[28];
                    normed_23[1] = rounded_1[29];
                    normed_23[2] = rounded_1[30];
                    normed_23[3] = rounded_1[31];
                }
                unsigned int normed_bf16_23[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_23[_lp*2 + 0], normed_23[_lp*2+1 + 0]));
                    normed_bf16_23[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_63 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_1 * 256) + (unsigned int)((warp_n_0 * 64 + 56) * 2 ^ (srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_63), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_23[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_23[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int srow_1 = role_tid_0 >> 2;
                int c_1 = role_tid_0 & 3;
                int grow_1 = tile_m_0 * 256 + 128 + srow_1;
                int chunk_lo_1 = ((c_1 < 3) ? c_1 * 16 : 96);
                int chunk_hi_1 = ((c_1 < 3) ? chunk_lo_1 + 48 : 112);
                if (grow_1 < M) {
                    unsigned int x_lo_2[8];
                    unsigned int x_hi_2[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_2[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_1 * 256) + (unsigned int)(chunk_lo_1 * 2 ^ (srow_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_2[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_1 * 256) + (unsigned int)(chunk_lo_1 * 2 + 16 ^ (srow_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_2[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_1 * 256) + (unsigned int)(chunk_hi_1 * 2 ^ (srow_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_2[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_1 * 256) + (unsigned int)(chunk_hi_1 * 2 + 16 ^ (srow_1 & 7) << 4)));
                    if (c_1 < 3) {
                        if (w_kind_0 < 2) {
                            float x_lo_f_2[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_lo_f_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_2[_pair]) << 16);
                                (&x_lo_f_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_2[_pair]) & 0xffff0000u);
                            }
                            float x_hi_f_2[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_hi_f_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_2[_pair]) << 16);
                                (&x_hi_f_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_2[_pair]) & 0xffff0000u);
                            }
                            float v_8[8];
                            {
                                const uint4* _vptr_64 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_1 * 96 + chunk_lo_1) + 0);
                                uint4 _vld_64[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_64[_blk] = _vptr_64[_blk];
                                    uint32_t* _vpairs_64 = reinterpret_cast<uint32_t*>(&_vld_64[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) << 16);
                                        (&v_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_9[8];
                            {
                                const uint4* _vptr_65 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_1 * 96 + chunk_lo_1 + 8) + 0);
                                uint4 _vld_65[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_65[_blk] = _vptr_65[_blk];
                                    uint32_t* _vpairs_65 = reinterpret_cast<uint32_t*>(&_vld_65[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) << 16);
                                        (&v_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_10[8];
                            {
                                const uint4* _vptr_66 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_1 * 96 + 48 + chunk_lo_1) + 0);
                                uint4 _vld_66[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_66[_blk] = _vptr_66[_blk];
                                    uint32_t* _vpairs_66 = reinterpret_cast<uint32_t*>(&_vld_66[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) << 16);
                                        (&v_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_11[8];
                            {
                                const uint4* _vptr_67 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_1 * 96 + 48 + chunk_lo_1 + 8) + 0);
                                uint4 _vld_67[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_67[_blk] = _vptr_67[_blk];
                                    uint32_t* _vpairs_67 = reinterpret_cast<uint32_t*>(&_vld_67[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_67[_pair]) << 16);
                                        (&v_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_67[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float o_lo_2[16];
                            float o_hi_2[16];
                            float cv0_14 = v_8[0];
                            float sv0_14 = v_10[0];
                            float cv1_14 = v_9[0];
                            float sv1_14 = v_11[0];
                            o_lo_2[0] = __fmaf_rn(x_lo_f_2[0], cv0_14, -(x_hi_f_2[0] * sv0_14));
                            o_hi_2[0] = __fmaf_rn(x_hi_f_2[0], cv0_14, x_lo_f_2[0] * sv0_14);
                            o_lo_2[8] = __fmaf_rn(x_lo_f_2[8], cv1_14, -(x_hi_f_2[8] * sv1_14));
                            o_hi_2[8] = __fmaf_rn(x_hi_f_2[8], cv1_14, x_lo_f_2[8] * sv1_14);
                            float cv0_14_1 = v_8[1];
                            float sv0_14_1 = v_10[1];
                            float cv1_14_1 = v_9[1];
                            float sv1_14_1 = v_11[1];
                            o_lo_2[1] = __fmaf_rn(x_lo_f_2[1], cv0_14_1, -(x_hi_f_2[1] * sv0_14_1));
                            o_hi_2[1] = __fmaf_rn(x_hi_f_2[1], cv0_14_1, x_lo_f_2[1] * sv0_14_1);
                            o_lo_2[9] = __fmaf_rn(x_lo_f_2[9], cv1_14_1, -(x_hi_f_2[9] * sv1_14_1));
                            o_hi_2[9] = __fmaf_rn(x_hi_f_2[9], cv1_14_1, x_lo_f_2[9] * sv1_14_1);
                            float cv0_15 = v_8[2];
                            float sv0_15 = v_10[2];
                            float cv1_15 = v_9[2];
                            float sv1_15 = v_11[2];
                            o_lo_2[2] = __fmaf_rn(x_lo_f_2[2], cv0_15, -(x_hi_f_2[2] * sv0_15));
                            o_hi_2[2] = __fmaf_rn(x_hi_f_2[2], cv0_15, x_lo_f_2[2] * sv0_15);
                            o_lo_2[10] = __fmaf_rn(x_lo_f_2[10], cv1_15, -(x_hi_f_2[10] * sv1_15));
                            o_hi_2[10] = __fmaf_rn(x_hi_f_2[10], cv1_15, x_lo_f_2[10] * sv1_15);
                            float cv0_16 = v_8[3];
                            float sv0_16 = v_10[3];
                            float cv1_16 = v_9[3];
                            float sv1_16 = v_11[3];
                            o_lo_2[3] = __fmaf_rn(x_lo_f_2[3], cv0_16, -(x_hi_f_2[3] * sv0_16));
                            o_hi_2[3] = __fmaf_rn(x_hi_f_2[3], cv0_16, x_lo_f_2[3] * sv0_16);
                            o_lo_2[11] = __fmaf_rn(x_lo_f_2[11], cv1_16, -(x_hi_f_2[11] * sv1_16));
                            o_hi_2[11] = __fmaf_rn(x_hi_f_2[11], cv1_16, x_lo_f_2[11] * sv1_16);
                            float cv0_17 = v_8[4];
                            float sv0_17 = v_10[4];
                            float cv1_17 = v_9[4];
                            float sv1_17 = v_11[4];
                            o_lo_2[4] = __fmaf_rn(x_lo_f_2[4], cv0_17, -(x_hi_f_2[4] * sv0_17));
                            o_hi_2[4] = __fmaf_rn(x_hi_f_2[4], cv0_17, x_lo_f_2[4] * sv0_17);
                            o_lo_2[12] = __fmaf_rn(x_lo_f_2[12], cv1_17, -(x_hi_f_2[12] * sv1_17));
                            o_hi_2[12] = __fmaf_rn(x_hi_f_2[12], cv1_17, x_lo_f_2[12] * sv1_17);
                            float cv0_18 = v_8[5];
                            float sv0_18 = v_10[5];
                            float cv1_18 = v_9[5];
                            float sv1_18 = v_11[5];
                            o_lo_2[5] = __fmaf_rn(x_lo_f_2[5], cv0_18, -(x_hi_f_2[5] * sv0_18));
                            o_hi_2[5] = __fmaf_rn(x_hi_f_2[5], cv0_18, x_lo_f_2[5] * sv0_18);
                            o_lo_2[13] = __fmaf_rn(x_lo_f_2[13], cv1_18, -(x_hi_f_2[13] * sv1_18));
                            o_hi_2[13] = __fmaf_rn(x_hi_f_2[13], cv1_18, x_lo_f_2[13] * sv1_18);
                            float cv0_19 = v_8[6];
                            float sv0_19 = v_10[6];
                            float cv1_19 = v_9[6];
                            float sv1_19 = v_11[6];
                            o_lo_2[6] = __fmaf_rn(x_lo_f_2[6], cv0_19, -(x_hi_f_2[6] * sv0_19));
                            o_hi_2[6] = __fmaf_rn(x_hi_f_2[6], cv0_19, x_lo_f_2[6] * sv0_19);
                            o_lo_2[14] = __fmaf_rn(x_lo_f_2[14], cv1_19, -(x_hi_f_2[14] * sv1_19));
                            o_hi_2[14] = __fmaf_rn(x_hi_f_2[14], cv1_19, x_lo_f_2[14] * sv1_19);
                            float cv0_20 = v_8[7];
                            float sv0_20 = v_10[7];
                            float cv1_20 = v_9[7];
                            float sv1_20 = v_11[7];
                            o_lo_2[7] = __fmaf_rn(x_lo_f_2[7], cv0_20, -(x_hi_f_2[7] * sv0_20));
                            o_hi_2[7] = __fmaf_rn(x_hi_f_2[7], cv0_20, x_lo_f_2[7] * sv0_20);
                            o_lo_2[15] = __fmaf_rn(x_lo_f_2[15], cv1_20, -(x_hi_f_2[15] * sv1_20));
                            o_hi_2[15] = __fmaf_rn(x_hi_f_2[15], cv1_20, x_lo_f_2[15] * sv1_20);
                            unsigned int o_lo_bf16_2[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_lo_2[_lp*2 + 0], o_lo_2[_lp*2+1 + 0]));
                                o_lo_bf16_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int o_hi_bf16_2[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_hi_2[_lp*2 + 0], o_hi_2[_lp*2+1 + 0]));
                                o_hi_bf16_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            x_lo_2[0] = o_lo_bf16_2[0];
                            x_hi_2[0] = o_hi_bf16_2[0];
                            x_lo_2[1] = o_lo_bf16_2[1];
                            x_hi_2[1] = o_hi_bf16_2[1];
                            x_lo_2[2] = o_lo_bf16_2[2];
                            x_hi_2[2] = o_hi_bf16_2[2];
                            x_lo_2[3] = o_lo_bf16_2[3];
                            x_hi_2[3] = o_hi_bf16_2[3];
                            x_lo_2[4] = o_lo_bf16_2[4];
                            x_hi_2[4] = o_hi_bf16_2[4];
                            x_lo_2[5] = o_lo_bf16_2[5];
                            x_hi_2[5] = o_hi_bf16_2[5];
                            x_lo_2[6] = o_lo_bf16_2[6];
                            x_hi_2[6] = o_hi_bf16_2[6];
                            x_lo_2[7] = o_lo_bf16_2[7];
                            x_hi_2[7] = o_hi_bf16_2[7];
                        }
                    }
                    unsigned int q_words_2[4];
                    float q_vals_2[16];
                    float f_4[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_2[_pair]) << 16);
                        (&f_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_2[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_4;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_4) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_4[0]), fabsf(f_4[1])), fabsf(f_4[2])), fabsf(f_4[3])), fabsf(f_4[4])), fabsf(f_4[5])), fabsf(f_4[6])), fabsf(f_4[7])), fabsf(f_4[8])), fabsf(f_4[9])), fabsf(f_4[10])), fabsf(f_4[11])), fabsf(f_4[12])), fabsf(f_4[13])), fabsf(f_4[14])), fabsf(f_4[15])) * sf_mul_sel_0));
                    uint16_t _e4m3x2_decode_68 = (uint16_t)((unsigned int)sf_pair_4 & 0xFFu);
                    uint32_t _f16x2_decode_68;
                    float sf_dec_4;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_68) : "h"(_e4m3x2_decode_68));
                    uint16_t _f16_decode_68 = (uint16_t)_f16x2_decode_68;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_4) : "h"(_f16_decode_68));
                    float out_s_4 = __fdiv_rn(out_scale_sel_0, sf_dec_4);
                    q_vals_2[0] = f_4[0] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[1] = f_4[1] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[2] = f_4[2] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[3] = f_4[3] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[4] = f_4[4] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[5] = f_4[5] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[6] = f_4[6] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[7] = f_4[7] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[8] = f_4[8] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[9] = f_4[9] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[10] = f_4[10] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[11] = f_4[11] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[12] = f_4[12] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[13] = f_4[13] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[14] = f_4[14] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    q_vals_2[15] = f_4[15] * ((sf_dec_4 != 0.0f) ? out_s_4 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_2[0]) : "f"(q_vals_2[0]), "f"(q_vals_2[1]), "f"(q_vals_2[2]), "f"(q_vals_2[3]), "f"(q_vals_2[4]), "f"(q_vals_2[5]), "f"(q_vals_2[6]), "f"(q_vals_2[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_2[1]) : "f"(q_vals_2[8]), "f"(q_vals_2[9]), "f"(q_vals_2[10]), "f"(q_vals_2[11]), "f"(q_vals_2[12]), "f"(q_vals_2[13]), "f"(q_vals_2[14]), "f"(q_vals_2[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow_1 * 128) + (unsigned int)(chunk_lo_1 >> 1 ^ (srow_1 & 7) << 4)), "r"(q_words_2[0]), "r"(q_words_2[1]) : "memory");
                    {
                        uint32_t _byte_69 = static_cast<uint32_t>((unsigned int)sf_pair_4 & 127) & 0xFFu;
                        uint32_t _addr_69 = static_cast<uint32_t>(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow_1 * 128) + (unsigned int)(64 + (chunk_lo_1 >> 4) ^ (srow_1 & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_69), "r"(_byte_69) : "memory");
                    }
                    float f_5[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_2[_pair]) << 16);
                        (&f_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_2[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_5;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_5) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_5[0]), fabsf(f_5[1])), fabsf(f_5[2])), fabsf(f_5[3])), fabsf(f_5[4])), fabsf(f_5[5])), fabsf(f_5[6])), fabsf(f_5[7])), fabsf(f_5[8])), fabsf(f_5[9])), fabsf(f_5[10])), fabsf(f_5[11])), fabsf(f_5[12])), fabsf(f_5[13])), fabsf(f_5[14])), fabsf(f_5[15])) * sf_mul_sel_0));
                    uint16_t _e4m3x2_decode_70 = (uint16_t)((unsigned int)sf_pair_5 & 0xFFu);
                    uint32_t _f16x2_decode_70;
                    float sf_dec_5;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_70) : "h"(_e4m3x2_decode_70));
                    uint16_t _f16_decode_70 = (uint16_t)_f16x2_decode_70;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_5) : "h"(_f16_decode_70));
                    float out_s_5 = __fdiv_rn(out_scale_sel_0, sf_dec_5);
                    q_vals_2[0] = f_5[0] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[1] = f_5[1] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[2] = f_5[2] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[3] = f_5[3] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[4] = f_5[4] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[5] = f_5[5] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[6] = f_5[6] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[7] = f_5[7] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[8] = f_5[8] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[9] = f_5[9] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[10] = f_5[10] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[11] = f_5[11] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[12] = f_5[12] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[13] = f_5[13] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[14] = f_5[14] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    q_vals_2[15] = f_5[15] * ((sf_dec_5 != 0.0f) ? out_s_5 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_2[0]) : "f"(q_vals_2[0]), "f"(q_vals_2[1]), "f"(q_vals_2[2]), "f"(q_vals_2[3]), "f"(q_vals_2[4]), "f"(q_vals_2[5]), "f"(q_vals_2[6]), "f"(q_vals_2[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_2[1]) : "f"(q_vals_2[8]), "f"(q_vals_2[9]), "f"(q_vals_2[10]), "f"(q_vals_2[11]), "f"(q_vals_2[12]), "f"(q_vals_2[13]), "f"(q_vals_2[14]), "f"(q_vals_2[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow_1 * 128) + (unsigned int)(chunk_hi_1 >> 1 ^ (srow_1 & 7) << 4)), "r"(q_words_2[0]), "r"(q_words_2[1]) : "memory");
                    {
                        uint32_t _byte_71 = static_cast<uint32_t>((unsigned int)sf_pair_5 & 127) & 0xFFu;
                        uint32_t _addr_71 = static_cast<uint32_t>(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow_1 * 128) + (unsigned int)(64 + (chunk_hi_1 >> 4) ^ (srow_1 & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_71), "r"(_byte_71) : "memory");
                    }
                }
                __syncwarp();
                int s_row_2 = (role_tid_0 >> 5) * 8 + lane / 4;
                int s_piece_2 = lane % 4;
                unsigned int s_words_2[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(s_row_2 * 128) + (unsigned int)(s_piece_2 * 16 ^ (s_row_2 & 7) << 4)));
                int s_dst_2 = ((tile_m_0 * 256 + 128 + s_row_2) * 56 + w_head_0) * 16 + s_piece_2 * 4;
                if (tile_m_0 * 256 + 128 + s_row_2 < M) {
                    if (w_kind_0 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_2)[0] = reinterpret_cast<int4*>(s_words_2)[0];
                    } else if (w_kind_0 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_2)[0] = reinterpret_cast<int4*>(s_words_2)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_2)[0] = reinterpret_cast<int4*>(s_words_2)[0];
                    }
                }
                int s_sf_row_1 = (role_tid_0 >> 5) * 8 + (lane & 7);
                unsigned int s_sf_words_1[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words_1[(0) + 1]))
                    : "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(s_sf_row_1 * 128) + (unsigned int)(64 ^ (s_sf_row_1 & 7) << 4)));
                int s_sf_dst_1 = ((tile_m_0 * 256 + 128 + s_sf_row_1) * 56 + w_head_0) * 8;
                if (lane < 8) {
                    if (tile_m_0 * 256 + 128 + s_sf_row_1 < M) {
                        if (w_kind_0 == 0) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(q_sf + s_sf_dst_1))[0] = reinterpret_cast<int2*>(s_sf_words_1)[0];
                        } else if (w_kind_0 == 1) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(k_sf + s_sf_dst_1))[0] = reinterpret_cast<int2*>(s_sf_words_1)[0];
                        } else {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(v_sf + s_sf_dst_1))[0] = reinterpret_cast<int2*>(s_sf_words_1)[0];
                        }
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int row_lo_2 = tile_m_0 * 256 + 192 + warp_m_0 * 16 + (lane >> 2);
                int row_hi_2 = row_lo_2 + 8;
                float sa_lo_2 = ((row_lo_2 < M) ? act_scale[row_lo_2] : 0.0f);
                float sa_hi_2 = ((row_hi_2 < M) ? act_scale[row_hi_2] : 0.0f);
                float rounded_2[32];
                float ss_lo_2 = 0.0f;
                float ss_hi_2 = 0.0f;
                float scaled_24[4];
                scaled_24[0] = accum_0[96] * sa_lo_2 * sw_0[0];
                scaled_24[1] = accum_0[97] * sa_lo_2 * sw_0[1];
                scaled_24[2] = accum_0[98] * sa_hi_2 * sw_0[0];
                scaled_24[3] = accum_0[99] * sa_hi_2 * sw_0[1];
                unsigned int scaled_bf16_24[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_24[_lp*2 + 0], scaled_24[_lp*2+1 + 0]));
                    scaled_bf16_24[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_24[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_24[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_24[_pair]) << 16);
                    (&unpacked_24[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_24[_pair]) & 0xffff0000u);
                }
                rounded_2[0] = unpacked_24[0];
                rounded_2[1] = unpacked_24[1];
                rounded_2[2] = unpacked_24[2];
                rounded_2[3] = unpacked_24[3];
                ss_lo_2 += __fmaf_rn(unpacked_24[0], unpacked_24[0], unpacked_24[1] * unpacked_24[1]);
                ss_hi_2 += __fmaf_rn(unpacked_24[2], unpacked_24[2], unpacked_24[3] * unpacked_24[3]);
                float scaled_25[4];
                scaled_25[0] = accum_0[100] * sa_lo_2 * sw_0[2];
                scaled_25[1] = accum_0[101] * sa_lo_2 * sw_0[3];
                scaled_25[2] = accum_0[102] * sa_hi_2 * sw_0[2];
                scaled_25[3] = accum_0[103] * sa_hi_2 * sw_0[3];
                unsigned int scaled_bf16_25[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_25[_lp*2 + 0], scaled_25[_lp*2+1 + 0]));
                    scaled_bf16_25[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_25[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_25[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_25[_pair]) << 16);
                    (&unpacked_25[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_25[_pair]) & 0xffff0000u);
                }
                rounded_2[4] = unpacked_25[0];
                rounded_2[5] = unpacked_25[1];
                rounded_2[6] = unpacked_25[2];
                rounded_2[7] = unpacked_25[3];
                ss_lo_2 += __fmaf_rn(unpacked_25[0], unpacked_25[0], unpacked_25[1] * unpacked_25[1]);
                ss_hi_2 += __fmaf_rn(unpacked_25[2], unpacked_25[2], unpacked_25[3] * unpacked_25[3]);
                float scaled_26[4];
                scaled_26[0] = accum_0[104] * sa_lo_2 * sw_0[4];
                scaled_26[1] = accum_0[105] * sa_lo_2 * sw_0[5];
                scaled_26[2] = accum_0[106] * sa_hi_2 * sw_0[4];
                scaled_26[3] = accum_0[107] * sa_hi_2 * sw_0[5];
                unsigned int scaled_bf16_26[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_26[_lp*2 + 0], scaled_26[_lp*2+1 + 0]));
                    scaled_bf16_26[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_26[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_26[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_26[_pair]) << 16);
                    (&unpacked_26[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_26[_pair]) & 0xffff0000u);
                }
                rounded_2[8] = unpacked_26[0];
                rounded_2[9] = unpacked_26[1];
                rounded_2[10] = unpacked_26[2];
                rounded_2[11] = unpacked_26[3];
                ss_lo_2 += __fmaf_rn(unpacked_26[0], unpacked_26[0], unpacked_26[1] * unpacked_26[1]);
                ss_hi_2 += __fmaf_rn(unpacked_26[2], unpacked_26[2], unpacked_26[3] * unpacked_26[3]);
                float scaled_27[4];
                scaled_27[0] = accum_0[108] * sa_lo_2 * sw_0[6];
                scaled_27[1] = accum_0[109] * sa_lo_2 * sw_0[7];
                scaled_27[2] = accum_0[110] * sa_hi_2 * sw_0[6];
                scaled_27[3] = accum_0[111] * sa_hi_2 * sw_0[7];
                unsigned int scaled_bf16_27[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_27[_lp*2 + 0], scaled_27[_lp*2+1 + 0]));
                    scaled_bf16_27[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_27[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_27[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_27[_pair]) << 16);
                    (&unpacked_27[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_27[_pair]) & 0xffff0000u);
                }
                rounded_2[12] = unpacked_27[0];
                rounded_2[13] = unpacked_27[1];
                rounded_2[14] = unpacked_27[2];
                rounded_2[15] = unpacked_27[3];
                ss_lo_2 += __fmaf_rn(unpacked_27[0], unpacked_27[0], unpacked_27[1] * unpacked_27[1]);
                ss_hi_2 += __fmaf_rn(unpacked_27[2], unpacked_27[2], unpacked_27[3] * unpacked_27[3]);
                float scaled_28[4];
                scaled_28[0] = accum_0[112] * sa_lo_2 * sw_0[8];
                scaled_28[1] = accum_0[113] * sa_lo_2 * sw_0[9];
                scaled_28[2] = accum_0[114] * sa_hi_2 * sw_0[8];
                scaled_28[3] = accum_0[115] * sa_hi_2 * sw_0[9];
                unsigned int scaled_bf16_28[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_28[_lp*2 + 0], scaled_28[_lp*2+1 + 0]));
                    scaled_bf16_28[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_28[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_28[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_28[_pair]) << 16);
                    (&unpacked_28[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_28[_pair]) & 0xffff0000u);
                }
                rounded_2[16] = unpacked_28[0];
                rounded_2[17] = unpacked_28[1];
                rounded_2[18] = unpacked_28[2];
                rounded_2[19] = unpacked_28[3];
                ss_lo_2 += __fmaf_rn(unpacked_28[0], unpacked_28[0], unpacked_28[1] * unpacked_28[1]);
                ss_hi_2 += __fmaf_rn(unpacked_28[2], unpacked_28[2], unpacked_28[3] * unpacked_28[3]);
                float scaled_29[4];
                scaled_29[0] = accum_0[116] * sa_lo_2 * sw_0[10];
                scaled_29[1] = accum_0[117] * sa_lo_2 * sw_0[11];
                scaled_29[2] = accum_0[118] * sa_hi_2 * sw_0[10];
                scaled_29[3] = accum_0[119] * sa_hi_2 * sw_0[11];
                unsigned int scaled_bf16_29[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_29[_lp*2 + 0], scaled_29[_lp*2+1 + 0]));
                    scaled_bf16_29[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_29[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_29[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_29[_pair]) << 16);
                    (&unpacked_29[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_29[_pair]) & 0xffff0000u);
                }
                rounded_2[20] = unpacked_29[0];
                rounded_2[21] = unpacked_29[1];
                rounded_2[22] = unpacked_29[2];
                rounded_2[23] = unpacked_29[3];
                ss_lo_2 += __fmaf_rn(unpacked_29[0], unpacked_29[0], unpacked_29[1] * unpacked_29[1]);
                ss_hi_2 += __fmaf_rn(unpacked_29[2], unpacked_29[2], unpacked_29[3] * unpacked_29[3]);
                float scaled_30[4];
                scaled_30[0] = accum_0[120] * sa_lo_2 * sw_0[12];
                scaled_30[1] = accum_0[121] * sa_lo_2 * sw_0[13];
                scaled_30[2] = accum_0[122] * sa_hi_2 * sw_0[12];
                scaled_30[3] = accum_0[123] * sa_hi_2 * sw_0[13];
                unsigned int scaled_bf16_30[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_30[_lp*2 + 0], scaled_30[_lp*2+1 + 0]));
                    scaled_bf16_30[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_30[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_30[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_30[_pair]) << 16);
                    (&unpacked_30[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_30[_pair]) & 0xffff0000u);
                }
                rounded_2[24] = unpacked_30[0];
                rounded_2[25] = unpacked_30[1];
                rounded_2[26] = unpacked_30[2];
                rounded_2[27] = unpacked_30[3];
                ss_lo_2 += __fmaf_rn(unpacked_30[0], unpacked_30[0], unpacked_30[1] * unpacked_30[1]);
                ss_hi_2 += __fmaf_rn(unpacked_30[2], unpacked_30[2], unpacked_30[3] * unpacked_30[3]);
                float scaled_31[4];
                scaled_31[0] = accum_0[124] * sa_lo_2 * sw_0[14];
                scaled_31[1] = accum_0[125] * sa_lo_2 * sw_0[15];
                scaled_31[2] = accum_0[126] * sa_hi_2 * sw_0[14];
                scaled_31[3] = accum_0[127] * sa_hi_2 * sw_0[15];
                unsigned int scaled_bf16_31[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_31[_lp*2 + 0], scaled_31[_lp*2+1 + 0]));
                    scaled_bf16_31[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_31[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_31[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_31[_pair]) << 16);
                    (&unpacked_31[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_31[_pair]) & 0xffff0000u);
                }
                rounded_2[28] = unpacked_31[0];
                rounded_2[29] = unpacked_31[1];
                rounded_2[30] = unpacked_31[2];
                rounded_2[31] = unpacked_31[3];
                ss_lo_2 += __fmaf_rn(unpacked_31[0], unpacked_31[0], unpacked_31[1] * unpacked_31[1]);
                ss_hi_2 += __fmaf_rn(unpacked_31[2], unpacked_31[2], unpacked_31[3] * unpacked_31[3]);
                float rstd_lo_2 = 1.0f;
                float rstd_hi_2 = 1.0f;
                if (w_kind_0 < 2) {
                    float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, ss_lo_2, 1);
                    ss_lo_2 += _shfl_xor_sync_72;
                    float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, ss_lo_2, 2);
                    ss_lo_2 += _shfl_xor_sync_73;
                    float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, ss_hi_2, 1);
                    ss_hi_2 += _shfl_xor_sync_74;
                    float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, ss_hi_2, 2);
                    ss_hi_2 += _shfl_xor_sync_75;
                    int srow_lo_3 = warp_m_0 * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n_0 * 64 + srow_lo_3] = ss_lo_2;
                        rowstat[warp_n_0 * 64 + srow_lo_3 + 8] = ss_hi_2;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float other_lo_3 = rowstat[(1 - warp_n_0) * 64 + srow_lo_3];
                    float other_hi_3 = rowstat[(1 - warp_n_0) * 64 + srow_lo_3 + 8];
                    float _fdiv_77 = __fdiv_rn(ss_lo_2 + other_lo_3, 128.0f);
                    float _rsqrt_76 = rsqrtf(_fdiv_77 + eps);
                    rstd_lo_2 = _rsqrt_76;
                    float _fdiv_79 = __fdiv_rn(ss_hi_2 + other_hi_3, 128.0f);
                    float _rsqrt_78 = rsqrtf(_fdiv_79 + eps);
                    rstd_hi_2 = _rsqrt_78;
                }
                int srow_st_2 = warp_m_0 * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float normed_24[4];
                if (w_kind_0 < 2) {
                    normed_24[0] = rounded_2[0] * rstd_lo_2 * nw_0[0];
                    normed_24[1] = rounded_2[1] * rstd_lo_2 * nw_0[1];
                    normed_24[2] = rounded_2[2] * rstd_hi_2 * nw_0[0];
                    normed_24[3] = rounded_2[3] * rstd_hi_2 * nw_0[1];
                } else {
                    normed_24[0] = rounded_2[0];
                    normed_24[1] = rounded_2[1];
                    normed_24[2] = rounded_2[2];
                    normed_24[3] = rounded_2[3];
                }
                unsigned int normed_bf16_24[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_24[_lp*2 + 0], normed_24[_lp*2+1 + 0]));
                    normed_bf16_24[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_80 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_2 * 256) + (unsigned int)(warp_n_0 * 64 * 2 ^ (srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_80), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_24[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_24[1]))
                    : "memory");
                float normed_25[4];
                if (w_kind_0 < 2) {
                    normed_25[0] = rounded_2[4] * rstd_lo_2 * nw_0[2];
                    normed_25[1] = rounded_2[5] * rstd_lo_2 * nw_0[3];
                    normed_25[2] = rounded_2[6] * rstd_hi_2 * nw_0[2];
                    normed_25[3] = rounded_2[7] * rstd_hi_2 * nw_0[3];
                } else {
                    normed_25[0] = rounded_2[4];
                    normed_25[1] = rounded_2[5];
                    normed_25[2] = rounded_2[6];
                    normed_25[3] = rounded_2[7];
                }
                unsigned int normed_bf16_25[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_25[_lp*2 + 0], normed_25[_lp*2+1 + 0]));
                    normed_bf16_25[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_81 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_2 * 256) + (unsigned int)((warp_n_0 * 64 + 8) * 2 ^ (srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_81), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_25[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_25[1]))
                    : "memory");
                float normed_26[4];
                if (w_kind_0 < 2) {
                    normed_26[0] = rounded_2[8] * rstd_lo_2 * nw_0[4];
                    normed_26[1] = rounded_2[9] * rstd_lo_2 * nw_0[5];
                    normed_26[2] = rounded_2[10] * rstd_hi_2 * nw_0[4];
                    normed_26[3] = rounded_2[11] * rstd_hi_2 * nw_0[5];
                } else {
                    normed_26[0] = rounded_2[8];
                    normed_26[1] = rounded_2[9];
                    normed_26[2] = rounded_2[10];
                    normed_26[3] = rounded_2[11];
                }
                unsigned int normed_bf16_26[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_26[_lp*2 + 0], normed_26[_lp*2+1 + 0]));
                    normed_bf16_26[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_82 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_2 * 256) + (unsigned int)((warp_n_0 * 64 + 16) * 2 ^ (srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_82), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_26[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_26[1]))
                    : "memory");
                float normed_27[4];
                if (w_kind_0 < 2) {
                    normed_27[0] = rounded_2[12] * rstd_lo_2 * nw_0[6];
                    normed_27[1] = rounded_2[13] * rstd_lo_2 * nw_0[7];
                    normed_27[2] = rounded_2[14] * rstd_hi_2 * nw_0[6];
                    normed_27[3] = rounded_2[15] * rstd_hi_2 * nw_0[7];
                } else {
                    normed_27[0] = rounded_2[12];
                    normed_27[1] = rounded_2[13];
                    normed_27[2] = rounded_2[14];
                    normed_27[3] = rounded_2[15];
                }
                unsigned int normed_bf16_27[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_27[_lp*2 + 0], normed_27[_lp*2+1 + 0]));
                    normed_bf16_27[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_83 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_2 * 256) + (unsigned int)((warp_n_0 * 64 + 24) * 2 ^ (srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_83), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_27[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_27[1]))
                    : "memory");
                float normed_28[4];
                if (w_kind_0 < 2) {
                    normed_28[0] = rounded_2[16] * rstd_lo_2 * nw_0[8];
                    normed_28[1] = rounded_2[17] * rstd_lo_2 * nw_0[9];
                    normed_28[2] = rounded_2[18] * rstd_hi_2 * nw_0[8];
                    normed_28[3] = rounded_2[19] * rstd_hi_2 * nw_0[9];
                } else {
                    normed_28[0] = rounded_2[16];
                    normed_28[1] = rounded_2[17];
                    normed_28[2] = rounded_2[18];
                    normed_28[3] = rounded_2[19];
                }
                unsigned int normed_bf16_28[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_28[_lp*2 + 0], normed_28[_lp*2+1 + 0]));
                    normed_bf16_28[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_84 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_2 * 256) + (unsigned int)((warp_n_0 * 64 + 32) * 2 ^ (srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_84), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_28[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_28[1]))
                    : "memory");
                float normed_29[4];
                if (w_kind_0 < 2) {
                    normed_29[0] = rounded_2[20] * rstd_lo_2 * nw_0[10];
                    normed_29[1] = rounded_2[21] * rstd_lo_2 * nw_0[11];
                    normed_29[2] = rounded_2[22] * rstd_hi_2 * nw_0[10];
                    normed_29[3] = rounded_2[23] * rstd_hi_2 * nw_0[11];
                } else {
                    normed_29[0] = rounded_2[20];
                    normed_29[1] = rounded_2[21];
                    normed_29[2] = rounded_2[22];
                    normed_29[3] = rounded_2[23];
                }
                unsigned int normed_bf16_29[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_29[_lp*2 + 0], normed_29[_lp*2+1 + 0]));
                    normed_bf16_29[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_85 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_2 * 256) + (unsigned int)((warp_n_0 * 64 + 40) * 2 ^ (srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_85), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_29[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_29[1]))
                    : "memory");
                float normed_30[4];
                if (w_kind_0 < 2) {
                    normed_30[0] = rounded_2[24] * rstd_lo_2 * nw_0[12];
                    normed_30[1] = rounded_2[25] * rstd_lo_2 * nw_0[13];
                    normed_30[2] = rounded_2[26] * rstd_hi_2 * nw_0[12];
                    normed_30[3] = rounded_2[27] * rstd_hi_2 * nw_0[13];
                } else {
                    normed_30[0] = rounded_2[24];
                    normed_30[1] = rounded_2[25];
                    normed_30[2] = rounded_2[26];
                    normed_30[3] = rounded_2[27];
                }
                unsigned int normed_bf16_30[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_30[_lp*2 + 0], normed_30[_lp*2+1 + 0]));
                    normed_bf16_30[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_86 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_2 * 256) + (unsigned int)((warp_n_0 * 64 + 48) * 2 ^ (srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_86), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_30[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_30[1]))
                    : "memory");
                float normed_31[4];
                if (w_kind_0 < 2) {
                    normed_31[0] = rounded_2[28] * rstd_lo_2 * nw_0[14];
                    normed_31[1] = rounded_2[29] * rstd_lo_2 * nw_0[15];
                    normed_31[2] = rounded_2[30] * rstd_hi_2 * nw_0[14];
                    normed_31[3] = rounded_2[31] * rstd_hi_2 * nw_0[15];
                } else {
                    normed_31[0] = rounded_2[28];
                    normed_31[1] = rounded_2[29];
                    normed_31[2] = rounded_2[30];
                    normed_31[3] = rounded_2[31];
                }
                unsigned int normed_bf16_31[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_31[_lp*2 + 0], normed_31[_lp*2+1 + 0]));
                    normed_bf16_31[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_87 = static_cast<uint32_t>(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_st_2 * 256) + (unsigned int)((warp_n_0 * 64 + 56) * 2 ^ (srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_87), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_31[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_31[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int srow_2 = role_tid_0 >> 2;
                int c_2 = role_tid_0 & 3;
                int grow_2 = tile_m_0 * 256 + 192 + srow_2;
                int chunk_lo_2 = ((c_2 < 3) ? c_2 * 16 : 96);
                int chunk_hi_2 = ((c_2 < 3) ? chunk_lo_2 + 48 : 112);
                if (grow_2 < M) {
                    unsigned int x_lo_3[8];
                    unsigned int x_hi_3[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_3[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_2 * 256) + (unsigned int)(chunk_lo_2 * 2 ^ (srow_2 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_3[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_2 * 256) + (unsigned int)(chunk_lo_2 * 2 + 16 ^ (srow_2 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_3[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_2 * 256) + (unsigned int)(chunk_hi_2 * 2 ^ (srow_2 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_3[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_0 * 16384 + (unsigned int)(srow_2 * 256) + (unsigned int)(chunk_hi_2 * 2 + 16 ^ (srow_2 & 7) << 4)));
                    if (c_2 < 3) {
                        if (w_kind_0 < 2) {
                            float x_lo_f_3[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_lo_f_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_3[_pair]) << 16);
                                (&x_lo_f_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_3[_pair]) & 0xffff0000u);
                            }
                            float x_hi_f_3[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_hi_f_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_3[_pair]) << 16);
                                (&x_hi_f_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_3[_pair]) & 0xffff0000u);
                            }
                            float v_12[8];
                            {
                                const uint4* _vptr_88 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_2 * 96 + chunk_lo_2) + 0);
                                uint4 _vld_88[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_88[_blk] = _vptr_88[_blk];
                                    uint32_t* _vpairs_88 = reinterpret_cast<uint32_t*>(&_vld_88[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_12[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_88[_pair]) << 16);
                                        (&v_12[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_88[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_13[8];
                            {
                                const uint4* _vptr_89 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_2 * 96 + chunk_lo_2 + 8) + 0);
                                uint4 _vld_89[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_89[_blk] = _vptr_89[_blk];
                                    uint32_t* _vpairs_89 = reinterpret_cast<uint32_t*>(&_vld_89[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_13[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_89[_pair]) << 16);
                                        (&v_13[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_89[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_14[8];
                            {
                                const uint4* _vptr_90 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_2 * 96 + 48 + chunk_lo_2) + 0);
                                uint4 _vld_90[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_90[_blk] = _vptr_90[_blk];
                                    uint32_t* _vpairs_90 = reinterpret_cast<uint32_t*>(&_vld_90[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_14[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_90[_pair]) << 16);
                                        (&v_14[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_90[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_15[8];
                            {
                                const uint4* _vptr_91 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_2 * 96 + 48 + chunk_lo_2 + 8) + 0);
                                uint4 _vld_91[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_91[_blk] = _vptr_91[_blk];
                                    uint32_t* _vpairs_91 = reinterpret_cast<uint32_t*>(&_vld_91[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_15[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_91[_pair]) << 16);
                                        (&v_15[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_91[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float o_lo_3[16];
                            float o_hi_3[16];
                            float cv0_21 = v_12[0];
                            float sv0_21 = v_14[0];
                            float cv1_21 = v_13[0];
                            float sv1_21 = v_15[0];
                            o_lo_3[0] = __fmaf_rn(x_lo_f_3[0], cv0_21, -(x_hi_f_3[0] * sv0_21));
                            o_hi_3[0] = __fmaf_rn(x_hi_f_3[0], cv0_21, x_lo_f_3[0] * sv0_21);
                            o_lo_3[8] = __fmaf_rn(x_lo_f_3[8], cv1_21, -(x_hi_f_3[8] * sv1_21));
                            o_hi_3[8] = __fmaf_rn(x_hi_f_3[8], cv1_21, x_lo_f_3[8] * sv1_21);
                            float cv0_21_1 = v_12[1];
                            float sv0_21_1 = v_14[1];
                            float cv1_21_1 = v_13[1];
                            float sv1_21_1 = v_15[1];
                            o_lo_3[1] = __fmaf_rn(x_lo_f_3[1], cv0_21_1, -(x_hi_f_3[1] * sv0_21_1));
                            o_hi_3[1] = __fmaf_rn(x_hi_f_3[1], cv0_21_1, x_lo_f_3[1] * sv0_21_1);
                            o_lo_3[9] = __fmaf_rn(x_lo_f_3[9], cv1_21_1, -(x_hi_f_3[9] * sv1_21_1));
                            o_hi_3[9] = __fmaf_rn(x_hi_f_3[9], cv1_21_1, x_lo_f_3[9] * sv1_21_1);
                            float cv0_22 = v_12[2];
                            float sv0_22 = v_14[2];
                            float cv1_22 = v_13[2];
                            float sv1_22 = v_15[2];
                            o_lo_3[2] = __fmaf_rn(x_lo_f_3[2], cv0_22, -(x_hi_f_3[2] * sv0_22));
                            o_hi_3[2] = __fmaf_rn(x_hi_f_3[2], cv0_22, x_lo_f_3[2] * sv0_22);
                            o_lo_3[10] = __fmaf_rn(x_lo_f_3[10], cv1_22, -(x_hi_f_3[10] * sv1_22));
                            o_hi_3[10] = __fmaf_rn(x_hi_f_3[10], cv1_22, x_lo_f_3[10] * sv1_22);
                            float cv0_23 = v_12[3];
                            float sv0_23 = v_14[3];
                            float cv1_23 = v_13[3];
                            float sv1_23 = v_15[3];
                            o_lo_3[3] = __fmaf_rn(x_lo_f_3[3], cv0_23, -(x_hi_f_3[3] * sv0_23));
                            o_hi_3[3] = __fmaf_rn(x_hi_f_3[3], cv0_23, x_lo_f_3[3] * sv0_23);
                            o_lo_3[11] = __fmaf_rn(x_lo_f_3[11], cv1_23, -(x_hi_f_3[11] * sv1_23));
                            o_hi_3[11] = __fmaf_rn(x_hi_f_3[11], cv1_23, x_lo_f_3[11] * sv1_23);
                            float cv0_24 = v_12[4];
                            float sv0_24 = v_14[4];
                            float cv1_24 = v_13[4];
                            float sv1_24 = v_15[4];
                            o_lo_3[4] = __fmaf_rn(x_lo_f_3[4], cv0_24, -(x_hi_f_3[4] * sv0_24));
                            o_hi_3[4] = __fmaf_rn(x_hi_f_3[4], cv0_24, x_lo_f_3[4] * sv0_24);
                            o_lo_3[12] = __fmaf_rn(x_lo_f_3[12], cv1_24, -(x_hi_f_3[12] * sv1_24));
                            o_hi_3[12] = __fmaf_rn(x_hi_f_3[12], cv1_24, x_lo_f_3[12] * sv1_24);
                            float cv0_25 = v_12[5];
                            float sv0_25 = v_14[5];
                            float cv1_25 = v_13[5];
                            float sv1_25 = v_15[5];
                            o_lo_3[5] = __fmaf_rn(x_lo_f_3[5], cv0_25, -(x_hi_f_3[5] * sv0_25));
                            o_hi_3[5] = __fmaf_rn(x_hi_f_3[5], cv0_25, x_lo_f_3[5] * sv0_25);
                            o_lo_3[13] = __fmaf_rn(x_lo_f_3[13], cv1_25, -(x_hi_f_3[13] * sv1_25));
                            o_hi_3[13] = __fmaf_rn(x_hi_f_3[13], cv1_25, x_lo_f_3[13] * sv1_25);
                            float cv0_26 = v_12[6];
                            float sv0_26 = v_14[6];
                            float cv1_26 = v_13[6];
                            float sv1_26 = v_15[6];
                            o_lo_3[6] = __fmaf_rn(x_lo_f_3[6], cv0_26, -(x_hi_f_3[6] * sv0_26));
                            o_hi_3[6] = __fmaf_rn(x_hi_f_3[6], cv0_26, x_lo_f_3[6] * sv0_26);
                            o_lo_3[14] = __fmaf_rn(x_lo_f_3[14], cv1_26, -(x_hi_f_3[14] * sv1_26));
                            o_hi_3[14] = __fmaf_rn(x_hi_f_3[14], cv1_26, x_lo_f_3[14] * sv1_26);
                            float cv0_27 = v_12[7];
                            float sv0_27 = v_14[7];
                            float cv1_27 = v_13[7];
                            float sv1_27 = v_15[7];
                            o_lo_3[7] = __fmaf_rn(x_lo_f_3[7], cv0_27, -(x_hi_f_3[7] * sv0_27));
                            o_hi_3[7] = __fmaf_rn(x_hi_f_3[7], cv0_27, x_lo_f_3[7] * sv0_27);
                            o_lo_3[15] = __fmaf_rn(x_lo_f_3[15], cv1_27, -(x_hi_f_3[15] * sv1_27));
                            o_hi_3[15] = __fmaf_rn(x_hi_f_3[15], cv1_27, x_lo_f_3[15] * sv1_27);
                            unsigned int o_lo_bf16_3[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_lo_3[_lp*2 + 0], o_lo_3[_lp*2+1 + 0]));
                                o_lo_bf16_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int o_hi_bf16_3[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_hi_3[_lp*2 + 0], o_hi_3[_lp*2+1 + 0]));
                                o_hi_bf16_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            x_lo_3[0] = o_lo_bf16_3[0];
                            x_hi_3[0] = o_hi_bf16_3[0];
                            x_lo_3[1] = o_lo_bf16_3[1];
                            x_hi_3[1] = o_hi_bf16_3[1];
                            x_lo_3[2] = o_lo_bf16_3[2];
                            x_hi_3[2] = o_hi_bf16_3[2];
                            x_lo_3[3] = o_lo_bf16_3[3];
                            x_hi_3[3] = o_hi_bf16_3[3];
                            x_lo_3[4] = o_lo_bf16_3[4];
                            x_hi_3[4] = o_hi_bf16_3[4];
                            x_lo_3[5] = o_lo_bf16_3[5];
                            x_hi_3[5] = o_hi_bf16_3[5];
                            x_lo_3[6] = o_lo_bf16_3[6];
                            x_hi_3[6] = o_hi_bf16_3[6];
                            x_lo_3[7] = o_lo_bf16_3[7];
                            x_hi_3[7] = o_hi_bf16_3[7];
                        }
                    }
                    unsigned int q_words_3[4];
                    float q_vals_3[16];
                    float f_6[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_3[_pair]) << 16);
                        (&f_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_3[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_6;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_6) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_6[0]), fabsf(f_6[1])), fabsf(f_6[2])), fabsf(f_6[3])), fabsf(f_6[4])), fabsf(f_6[5])), fabsf(f_6[6])), fabsf(f_6[7])), fabsf(f_6[8])), fabsf(f_6[9])), fabsf(f_6[10])), fabsf(f_6[11])), fabsf(f_6[12])), fabsf(f_6[13])), fabsf(f_6[14])), fabsf(f_6[15])) * sf_mul_sel_0));
                    uint16_t _e4m3x2_decode_92 = (uint16_t)((unsigned int)sf_pair_6 & 0xFFu);
                    uint32_t _f16x2_decode_92;
                    float sf_dec_6;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_92) : "h"(_e4m3x2_decode_92));
                    uint16_t _f16_decode_92 = (uint16_t)_f16x2_decode_92;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_6) : "h"(_f16_decode_92));
                    float out_s_6 = __fdiv_rn(out_scale_sel_0, sf_dec_6);
                    q_vals_3[0] = f_6[0] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[1] = f_6[1] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[2] = f_6[2] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[3] = f_6[3] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[4] = f_6[4] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[5] = f_6[5] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[6] = f_6[6] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[7] = f_6[7] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[8] = f_6[8] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[9] = f_6[9] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[10] = f_6[10] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[11] = f_6[11] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[12] = f_6[12] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[13] = f_6[13] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[14] = f_6[14] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    q_vals_3[15] = f_6[15] * ((sf_dec_6 != 0.0f) ? out_s_6 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_3[0]) : "f"(q_vals_3[0]), "f"(q_vals_3[1]), "f"(q_vals_3[2]), "f"(q_vals_3[3]), "f"(q_vals_3[4]), "f"(q_vals_3[5]), "f"(q_vals_3[6]), "f"(q_vals_3[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_3[1]) : "f"(q_vals_3[8]), "f"(q_vals_3[9]), "f"(q_vals_3[10]), "f"(q_vals_3[11]), "f"(q_vals_3[12]), "f"(q_vals_3[13]), "f"(q_vals_3[14]), "f"(q_vals_3[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow_2 * 128) + (unsigned int)(chunk_lo_2 >> 1 ^ (srow_2 & 7) << 4)), "r"(q_words_3[0]), "r"(q_words_3[1]) : "memory");
                    {
                        uint32_t _byte_93 = static_cast<uint32_t>((unsigned int)sf_pair_6 & 127) & 0xFFu;
                        uint32_t _addr_93 = static_cast<uint32_t>(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow_2 * 128) + (unsigned int)(64 + (chunk_lo_2 >> 4) ^ (srow_2 & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_93), "r"(_byte_93) : "memory");
                    }
                    float f_7[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_3[_pair]) << 16);
                        (&f_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_3[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_7;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_7) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_7[0]), fabsf(f_7[1])), fabsf(f_7[2])), fabsf(f_7[3])), fabsf(f_7[4])), fabsf(f_7[5])), fabsf(f_7[6])), fabsf(f_7[7])), fabsf(f_7[8])), fabsf(f_7[9])), fabsf(f_7[10])), fabsf(f_7[11])), fabsf(f_7[12])), fabsf(f_7[13])), fabsf(f_7[14])), fabsf(f_7[15])) * sf_mul_sel_0));
                    uint16_t _e4m3x2_decode_94 = (uint16_t)((unsigned int)sf_pair_7 & 0xFFu);
                    uint32_t _f16x2_decode_94;
                    float sf_dec_7;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_94) : "h"(_e4m3x2_decode_94));
                    uint16_t _f16_decode_94 = (uint16_t)_f16x2_decode_94;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_7) : "h"(_f16_decode_94));
                    float out_s_7 = __fdiv_rn(out_scale_sel_0, sf_dec_7);
                    q_vals_3[0] = f_7[0] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[1] = f_7[1] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[2] = f_7[2] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[3] = f_7[3] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[4] = f_7[4] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[5] = f_7[5] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[6] = f_7[6] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[7] = f_7[7] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[8] = f_7[8] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[9] = f_7[9] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[10] = f_7[10] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[11] = f_7[11] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[12] = f_7[12] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[13] = f_7[13] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[14] = f_7[14] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    q_vals_3[15] = f_7[15] * ((sf_dec_7 != 0.0f) ? out_s_7 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_3[0]) : "f"(q_vals_3[0]), "f"(q_vals_3[1]), "f"(q_vals_3[2]), "f"(q_vals_3[3]), "f"(q_vals_3[4]), "f"(q_vals_3[5]), "f"(q_vals_3[6]), "f"(q_vals_3[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_3[1]) : "f"(q_vals_3[8]), "f"(q_vals_3[9]), "f"(q_vals_3[10]), "f"(q_vals_3[11]), "f"(q_vals_3[12]), "f"(q_vals_3[13]), "f"(q_vals_3[14]), "f"(q_vals_3[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow_2 * 128) + (unsigned int)(chunk_hi_2 >> 1 ^ (srow_2 & 7) << 4)), "r"(q_words_3[0]), "r"(q_words_3[1]) : "memory");
                    {
                        uint32_t _byte_95 = static_cast<uint32_t>((unsigned int)sf_pair_7 & 127) & 0xFFu;
                        uint32_t _addr_95 = static_cast<uint32_t>(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(srow_2 * 128) + (unsigned int)(64 + (chunk_hi_2 >> 4) ^ (srow_2 & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_95), "r"(_byte_95) : "memory");
                    }
                }
                __syncwarp();
                int s_row_3 = (role_tid_0 >> 5) * 8 + lane / 4;
                int s_piece_3 = lane % 4;
                unsigned int s_words_3[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(s_row_3 * 128) + (unsigned int)(s_piece_3 * 16 ^ (s_row_3 & 7) << 4)));
                int s_dst_3 = ((tile_m_0 * 256 + 192 + s_row_3) * 56 + w_head_0) * 16 + s_piece_3 * 4;
                if (tile_m_0 * 256 + 192 + s_row_3 < M) {
                    if (w_kind_0 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_3)[0] = reinterpret_cast<int4*>(s_words_3)[0];
                    } else if (w_kind_0 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_3)[0] = reinterpret_cast<int4*>(s_words_3)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_3)[0] = reinterpret_cast<int4*>(s_words_3)[0];
                    }
                }
                int s_sf_row_2 = (role_tid_0 >> 5) * 8 + (lane & 7);
                unsigned int s_sf_words_2[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words_2[(0) + 1]))
                    : "r"(B_stage_addr + free_stage_0 * 8192 + (unsigned int)(s_sf_row_2 * 128) + (unsigned int)(64 ^ (s_sf_row_2 & 7) << 4)));
                int s_sf_dst_2 = ((tile_m_0 * 256 + 192 + s_sf_row_2) * 56 + w_head_0) * 8;
                if (lane < 8) {
                    if (tile_m_0 * 256 + 192 + s_sf_row_2 < M) {
                        if (w_kind_0 == 0) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(q_sf + s_sf_dst_2))[0] = reinterpret_cast<int2*>(s_sf_words_2)[0];
                        } else if (w_kind_0 == 1) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(k_sf + s_sf_dst_2))[0] = reinterpret_cast<int2*>(s_sf_words_2)[0];
                        } else {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(v_sf + s_sf_dst_2))[0] = reinterpret_cast<int2*>(s_sf_words_2)[0];
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 1 && warp <= 7) {
        { // mma_main
            unsigned int mma_stage_1 = 0;
            unsigned int mma_kk_1 = 0;
            int warp_m_1 = warp % 4;
            int warp_n_1 = warp / 4;
            int role_tid_1 = warp * 32 + lane;
            float accum_1[128];
            unsigned int a_frag_1[16];
            unsigned int b_frag_1[16];
            unsigned int fp8_unit_word_1[1];
            fp8_unit_word_1[0] = 2139062143;
            uint8_t fp8_unit_sfa_1[1];
            uint8_t fp8_unit_sfb_1[1];
            fp8_unit_sfa_1[0] = fp8_unit_word_1[0];
            fp8_unit_sfb_1[0] = fp8_unit_word_1[0];
            float sw_1[16];
            float nw_1[16];
            unsigned int _phase_ab_full_1 = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / 5376 * 32 + (tile_1 - tile_1 / 5376 * 5376) % ((num_m_tiles - tile_1 / 5376 * 32 < 32) ? num_m_tiles - tile_1 / 5376 * 32 : 32);
                int tile_n_1 = (tile_1 - tile_1 / 5376 * 5376) / ((num_m_tiles - tile_1 / 5376 * 32 < 32) ? num_m_tiles - tile_1 / 5376 * 32 : 32);
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
                int c_kind_1 = tile_n_1 % 3;
                int c_col_base_1 = tile_n_1 * 128 + warp_n_1 * 64 + (lane & 3) * 2;
                sw_1[0] = w_scale[c_col_base_1];
                sw_1[1] = w_scale[c_col_base_1 + 1];
                sw_1[2] = w_scale[c_col_base_1 + 8];
                sw_1[3] = w_scale[c_col_base_1 + 8 + 1];
                sw_1[4] = w_scale[c_col_base_1 + 16];
                sw_1[5] = w_scale[c_col_base_1 + 16 + 1];
                sw_1[6] = w_scale[c_col_base_1 + 24];
                sw_1[7] = w_scale[c_col_base_1 + 24 + 1];
                sw_1[8] = w_scale[c_col_base_1 + 32];
                sw_1[9] = w_scale[c_col_base_1 + 32 + 1];
                sw_1[10] = w_scale[c_col_base_1 + 40];
                sw_1[11] = w_scale[c_col_base_1 + 40 + 1];
                sw_1[12] = w_scale[c_col_base_1 + 48];
                sw_1[13] = w_scale[c_col_base_1 + 48 + 1];
                sw_1[14] = w_scale[c_col_base_1 + 56];
                sw_1[15] = w_scale[c_col_base_1 + 56 + 1];
                int c_d_col_8 = warp_n_1 * 64 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[0] = (float)q_norm_weight[c_d_col_8];
                    nw_1[1] = (float)q_norm_weight[c_d_col_8 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[0] = (float)k_norm_weight[c_d_col_8];
                    nw_1[1] = (float)k_norm_weight[c_d_col_8 + 1];
                } else {
                    nw_1[0] = 1.0f;
                    nw_1[1] = 1.0f;
                }
                int c_d_col_9 = warp_n_1 * 64 + 8 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[2] = (float)q_norm_weight[c_d_col_9];
                    nw_1[3] = (float)q_norm_weight[c_d_col_9 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[2] = (float)k_norm_weight[c_d_col_9];
                    nw_1[3] = (float)k_norm_weight[c_d_col_9 + 1];
                } else {
                    nw_1[2] = 1.0f;
                    nw_1[3] = 1.0f;
                }
                int c_d_col_10 = warp_n_1 * 64 + 16 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[4] = (float)q_norm_weight[c_d_col_10];
                    nw_1[5] = (float)q_norm_weight[c_d_col_10 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[4] = (float)k_norm_weight[c_d_col_10];
                    nw_1[5] = (float)k_norm_weight[c_d_col_10 + 1];
                } else {
                    nw_1[4] = 1.0f;
                    nw_1[5] = 1.0f;
                }
                int c_d_col_11 = warp_n_1 * 64 + 24 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[6] = (float)q_norm_weight[c_d_col_11];
                    nw_1[7] = (float)q_norm_weight[c_d_col_11 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[6] = (float)k_norm_weight[c_d_col_11];
                    nw_1[7] = (float)k_norm_weight[c_d_col_11 + 1];
                } else {
                    nw_1[6] = 1.0f;
                    nw_1[7] = 1.0f;
                }
                int c_d_col_12 = warp_n_1 * 64 + 32 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[8] = (float)q_norm_weight[c_d_col_12];
                    nw_1[9] = (float)q_norm_weight[c_d_col_12 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[8] = (float)k_norm_weight[c_d_col_12];
                    nw_1[9] = (float)k_norm_weight[c_d_col_12 + 1];
                } else {
                    nw_1[8] = 1.0f;
                    nw_1[9] = 1.0f;
                }
                int c_d_col_13 = warp_n_1 * 64 + 40 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[10] = (float)q_norm_weight[c_d_col_13];
                    nw_1[11] = (float)q_norm_weight[c_d_col_13 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[10] = (float)k_norm_weight[c_d_col_13];
                    nw_1[11] = (float)k_norm_weight[c_d_col_13 + 1];
                } else {
                    nw_1[10] = 1.0f;
                    nw_1[11] = 1.0f;
                }
                int c_d_col_14 = warp_n_1 * 64 + 48 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[12] = (float)q_norm_weight[c_d_col_14];
                    nw_1[13] = (float)q_norm_weight[c_d_col_14 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[12] = (float)k_norm_weight[c_d_col_14];
                    nw_1[13] = (float)k_norm_weight[c_d_col_14 + 1];
                } else {
                    nw_1[12] = 1.0f;
                    nw_1[13] = 1.0f;
                }
                int c_d_col_15 = warp_n_1 * 64 + 56 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[14] = (float)q_norm_weight[c_d_col_15];
                    nw_1[15] = (float)q_norm_weight[c_d_col_15 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[14] = (float)k_norm_weight[c_d_col_15];
                    nw_1[15] = (float)k_norm_weight[c_d_col_15 + 1];
                } else {
                    nw_1[14] = 1.0f;
                    nw_1[15] = 1.0f;
                }
                #pragma unroll 1
                for (int k_tile_1 = 0; k_tile_1 < 84; k_tile_1++) {
                    mbarrier_wait_hint(ab_full_addr + (mma_stage_1) * 8, _phase_ab_full_1, 10000000);
                    for (int k_step_1 = 0; k_step_1 < 2; k_step_1++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(A_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_m_1 * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[4]), "=r"(a_frag_1[5]), "=r"(a_frag_1[6]), "=r"(a_frag_1[7])
                            : "r"(A_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_m_1 * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[8]), "=r"(a_frag_1[9]), "=r"(a_frag_1[10]), "=r"(a_frag_1[11])
                            : "r"(A_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_m_1 * 16 + 128 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 16 + 128 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[12]), "=r"(a_frag_1[13]), "=r"(a_frag_1[14]), "=r"(a_frag_1[15])
                            : "r"(A_stage_addr + mma_stage_1 * 16384 + (unsigned int)((warp_m_1 * 16 + 192 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m_1 * 16 + 192 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1]), "=r"(b_frag_1[2]), "=r"(b_frag_1[3])
                            : "r"(B_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_n_1 * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[4]), "=r"(b_frag_1[5]), "=r"(b_frag_1[6]), "=r"(b_frag_1[7])
                            : "r"(B_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_n_1 * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[8]), "=r"(b_frag_1[9]), "=r"(b_frag_1[10]), "=r"(b_frag_1[11])
                            : "r"(B_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_n_1 * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_1[12]), "=r"(b_frag_1[13]), "=r"(b_frag_1[14]), "=r"(b_frag_1[15])
                            : "r"(B_stage_addr + mma_stage_1 * 8192 + (unsigned int)((warp_n_1 * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step_1 * 32 + (lane >> 3 & 1) * 16 ^ (warp_n_1 * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[0]), "+f"(accum_1[1]), "+f"(accum_1[2]), "+f"(accum_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[4]), "+f"(accum_1[(4) + 1]), "+f"(accum_1[(4) + 2]), "+f"(accum_1[(4) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[8]), "+f"(accum_1[(8) + 1]), "+f"(accum_1[(8) + 2]), "+f"(accum_1[(8) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[12]), "+f"(accum_1[(12) + 1]), "+f"(accum_1[(12) + 2]), "+f"(accum_1[(12) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[16]), "+f"(accum_1[(16) + 1]), "+f"(accum_1[(16) + 2]), "+f"(accum_1[(16) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[20]), "+f"(accum_1[(20) + 1]), "+f"(accum_1[(20) + 2]), "+f"(accum_1[(20) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[24]), "+f"(accum_1[(24) + 1]), "+f"(accum_1[(24) + 2]), "+f"(accum_1[(24) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[28]), "+f"(accum_1[(28) + 1]), "+f"(accum_1[(28) + 2]), "+f"(accum_1[(28) + 3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[32]), "+f"(accum_1[(32) + 1]), "+f"(accum_1[(32) + 2]), "+f"(accum_1[(32) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[36]), "+f"(accum_1[(36) + 1]), "+f"(accum_1[(36) + 2]), "+f"(accum_1[(36) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[40]), "+f"(accum_1[(40) + 1]), "+f"(accum_1[(40) + 2]), "+f"(accum_1[(40) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[44]), "+f"(accum_1[(44) + 1]), "+f"(accum_1[(44) + 2]), "+f"(accum_1[(44) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[48]), "+f"(accum_1[(48) + 1]), "+f"(accum_1[(48) + 2]), "+f"(accum_1[(48) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[52]), "+f"(accum_1[(52) + 1]), "+f"(accum_1[(52) + 2]), "+f"(accum_1[(52) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[56]), "+f"(accum_1[(56) + 1]), "+f"(accum_1[(56) + 2]), "+f"(accum_1[(56) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[60]), "+f"(accum_1[(60) + 1]), "+f"(accum_1[(60) + 2]), "+f"(accum_1[(60) + 3])
                            : "r"(a_frag_1[4]), "r"(a_frag_1[(4) + 1]), "r"(a_frag_1[(4) + 2]), "r"(a_frag_1[(4) + 3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[64]), "+f"(accum_1[(64) + 1]), "+f"(accum_1[(64) + 2]), "+f"(accum_1[(64) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[68]), "+f"(accum_1[(68) + 1]), "+f"(accum_1[(68) + 2]), "+f"(accum_1[(68) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[72]), "+f"(accum_1[(72) + 1]), "+f"(accum_1[(72) + 2]), "+f"(accum_1[(72) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[76]), "+f"(accum_1[(76) + 1]), "+f"(accum_1[(76) + 2]), "+f"(accum_1[(76) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[80]), "+f"(accum_1[(80) + 1]), "+f"(accum_1[(80) + 2]), "+f"(accum_1[(80) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[84]), "+f"(accum_1[(84) + 1]), "+f"(accum_1[(84) + 2]), "+f"(accum_1[(84) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[88]), "+f"(accum_1[(88) + 1]), "+f"(accum_1[(88) + 2]), "+f"(accum_1[(88) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[92]), "+f"(accum_1[(92) + 1]), "+f"(accum_1[(92) + 2]), "+f"(accum_1[(92) + 3])
                            : "r"(a_frag_1[8]), "r"(a_frag_1[(8) + 1]), "r"(a_frag_1[(8) + 2]), "r"(a_frag_1[(8) + 3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[96]), "+f"(accum_1[(96) + 1]), "+f"(accum_1[(96) + 2]), "+f"(accum_1[(96) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[100]), "+f"(accum_1[(100) + 1]), "+f"(accum_1[(100) + 2]), "+f"(accum_1[(100) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[104]), "+f"(accum_1[(104) + 1]), "+f"(accum_1[(104) + 2]), "+f"(accum_1[(104) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[4]), "r"(b_frag_1[(4) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[108]), "+f"(accum_1[(108) + 1]), "+f"(accum_1[(108) + 2]), "+f"(accum_1[(108) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[6]), "r"(b_frag_1[(6) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[112]), "+f"(accum_1[(112) + 1]), "+f"(accum_1[(112) + 2]), "+f"(accum_1[(112) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[8]), "r"(b_frag_1[(8) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[116]), "+f"(accum_1[(116) + 1]), "+f"(accum_1[(116) + 2]), "+f"(accum_1[(116) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[10]), "r"(b_frag_1[(10) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[120]), "+f"(accum_1[(120) + 1]), "+f"(accum_1[(120) + 2]), "+f"(accum_1[(120) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[12]), "r"(b_frag_1[(12) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accum_1[124]), "+f"(accum_1[(124) + 1]), "+f"(accum_1[(124) + 2]), "+f"(accum_1[(124) + 3])
                            : "r"(a_frag_1[12]), "r"(a_frag_1[(12) + 1]), "r"(a_frag_1[(12) + 2]), "r"(a_frag_1[(12) + 3]), "r"(b_frag_1[14]), "r"(b_frag_1[(14) + 1]), "r"((uint32_t)(fp8_unit_sfa_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane == 0) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage_1) * 8);
                    }
                    mma_stage_1 += 1;
                    if (mma_stage_1 == 4) { mma_stage_1 = 0; _phase_ab_full_1 ^= 1; }
                    mma_kk_1 = mma_kk_1 + 1;
                }
                unsigned int free_stage_1 = ((mma_stage_1 == 0) ? 3 : mma_stage_1 - 1);
                int w_kind_1 = tile_n_1 % 3;
                int w_head_1 = tile_n_1 / 3;
                float out_scale_sel_1 = ((w_kind_1 == 0) ? out_scale_q : ((w_kind_1 == 1) ? out_scale_k : out_scale_v));
                float sf_mul_sel_1 = ((w_kind_1 == 0) ? sf_mul_q : ((w_kind_1 == 1) ? sf_mul_k : sf_mul_v));
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int row_lo_3 = tile_m_1 * 256 + warp_m_1 * 16 + (lane >> 2);
                int row_hi_3 = row_lo_3 + 8;
                float sa_lo_3 = ((row_lo_3 < M) ? act_scale[row_lo_3] : 0.0f);
                float sa_hi_3 = ((row_hi_3 < M) ? act_scale[row_hi_3] : 0.0f);
                float rounded_3[32];
                float ss_lo_3 = 0.0f;
                float ss_hi_3 = 0.0f;
                float scaled_32[4];
                scaled_32[0] = accum_1[0] * sa_lo_3 * sw_1[0];
                scaled_32[1] = accum_1[1] * sa_lo_3 * sw_1[1];
                scaled_32[2] = accum_1[2] * sa_hi_3 * sw_1[0];
                scaled_32[3] = accum_1[3] * sa_hi_3 * sw_1[1];
                unsigned int scaled_bf16_32[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_32[_lp*2 + 0], scaled_32[_lp*2+1 + 0]));
                    scaled_bf16_32[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_32[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_32[_pair]) << 16);
                    (&unpacked_32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_32[_pair]) & 0xffff0000u);
                }
                rounded_3[0] = unpacked_32[0];
                rounded_3[1] = unpacked_32[1];
                rounded_3[2] = unpacked_32[2];
                rounded_3[3] = unpacked_32[3];
                ss_lo_3 += __fmaf_rn(unpacked_32[0], unpacked_32[0], unpacked_32[1] * unpacked_32[1]);
                ss_hi_3 += __fmaf_rn(unpacked_32[2], unpacked_32[2], unpacked_32[3] * unpacked_32[3]);
                float scaled_33[4];
                scaled_33[0] = accum_1[4] * sa_lo_3 * sw_1[2];
                scaled_33[1] = accum_1[5] * sa_lo_3 * sw_1[3];
                scaled_33[2] = accum_1[6] * sa_hi_3 * sw_1[2];
                scaled_33[3] = accum_1[7] * sa_hi_3 * sw_1[3];
                unsigned int scaled_bf16_33[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_33[_lp*2 + 0], scaled_33[_lp*2+1 + 0]));
                    scaled_bf16_33[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_33[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_33[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_33[_pair]) << 16);
                    (&unpacked_33[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_33[_pair]) & 0xffff0000u);
                }
                rounded_3[4] = unpacked_33[0];
                rounded_3[5] = unpacked_33[1];
                rounded_3[6] = unpacked_33[2];
                rounded_3[7] = unpacked_33[3];
                ss_lo_3 += __fmaf_rn(unpacked_33[0], unpacked_33[0], unpacked_33[1] * unpacked_33[1]);
                ss_hi_3 += __fmaf_rn(unpacked_33[2], unpacked_33[2], unpacked_33[3] * unpacked_33[3]);
                float scaled_34[4];
                scaled_34[0] = accum_1[8] * sa_lo_3 * sw_1[4];
                scaled_34[1] = accum_1[9] * sa_lo_3 * sw_1[5];
                scaled_34[2] = accum_1[10] * sa_hi_3 * sw_1[4];
                scaled_34[3] = accum_1[11] * sa_hi_3 * sw_1[5];
                unsigned int scaled_bf16_34[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_34[_lp*2 + 0], scaled_34[_lp*2+1 + 0]));
                    scaled_bf16_34[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_34[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_34[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_34[_pair]) << 16);
                    (&unpacked_34[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_34[_pair]) & 0xffff0000u);
                }
                rounded_3[8] = unpacked_34[0];
                rounded_3[9] = unpacked_34[1];
                rounded_3[10] = unpacked_34[2];
                rounded_3[11] = unpacked_34[3];
                ss_lo_3 += __fmaf_rn(unpacked_34[0], unpacked_34[0], unpacked_34[1] * unpacked_34[1]);
                ss_hi_3 += __fmaf_rn(unpacked_34[2], unpacked_34[2], unpacked_34[3] * unpacked_34[3]);
                float scaled_35[4];
                scaled_35[0] = accum_1[12] * sa_lo_3 * sw_1[6];
                scaled_35[1] = accum_1[13] * sa_lo_3 * sw_1[7];
                scaled_35[2] = accum_1[14] * sa_hi_3 * sw_1[6];
                scaled_35[3] = accum_1[15] * sa_hi_3 * sw_1[7];
                unsigned int scaled_bf16_35[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_35[_lp*2 + 0], scaled_35[_lp*2+1 + 0]));
                    scaled_bf16_35[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_35[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_35[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_35[_pair]) << 16);
                    (&unpacked_35[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_35[_pair]) & 0xffff0000u);
                }
                rounded_3[12] = unpacked_35[0];
                rounded_3[13] = unpacked_35[1];
                rounded_3[14] = unpacked_35[2];
                rounded_3[15] = unpacked_35[3];
                ss_lo_3 += __fmaf_rn(unpacked_35[0], unpacked_35[0], unpacked_35[1] * unpacked_35[1]);
                ss_hi_3 += __fmaf_rn(unpacked_35[2], unpacked_35[2], unpacked_35[3] * unpacked_35[3]);
                float scaled_36[4];
                scaled_36[0] = accum_1[16] * sa_lo_3 * sw_1[8];
                scaled_36[1] = accum_1[17] * sa_lo_3 * sw_1[9];
                scaled_36[2] = accum_1[18] * sa_hi_3 * sw_1[8];
                scaled_36[3] = accum_1[19] * sa_hi_3 * sw_1[9];
                unsigned int scaled_bf16_36[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_36[_lp*2 + 0], scaled_36[_lp*2+1 + 0]));
                    scaled_bf16_36[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_36[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_36[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_36[_pair]) << 16);
                    (&unpacked_36[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_36[_pair]) & 0xffff0000u);
                }
                rounded_3[16] = unpacked_36[0];
                rounded_3[17] = unpacked_36[1];
                rounded_3[18] = unpacked_36[2];
                rounded_3[19] = unpacked_36[3];
                ss_lo_3 += __fmaf_rn(unpacked_36[0], unpacked_36[0], unpacked_36[1] * unpacked_36[1]);
                ss_hi_3 += __fmaf_rn(unpacked_36[2], unpacked_36[2], unpacked_36[3] * unpacked_36[3]);
                float scaled_37[4];
                scaled_37[0] = accum_1[20] * sa_lo_3 * sw_1[10];
                scaled_37[1] = accum_1[21] * sa_lo_3 * sw_1[11];
                scaled_37[2] = accum_1[22] * sa_hi_3 * sw_1[10];
                scaled_37[3] = accum_1[23] * sa_hi_3 * sw_1[11];
                unsigned int scaled_bf16_37[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_37[_lp*2 + 0], scaled_37[_lp*2+1 + 0]));
                    scaled_bf16_37[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_37[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_37[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_37[_pair]) << 16);
                    (&unpacked_37[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_37[_pair]) & 0xffff0000u);
                }
                rounded_3[20] = unpacked_37[0];
                rounded_3[21] = unpacked_37[1];
                rounded_3[22] = unpacked_37[2];
                rounded_3[23] = unpacked_37[3];
                ss_lo_3 += __fmaf_rn(unpacked_37[0], unpacked_37[0], unpacked_37[1] * unpacked_37[1]);
                ss_hi_3 += __fmaf_rn(unpacked_37[2], unpacked_37[2], unpacked_37[3] * unpacked_37[3]);
                float scaled_38[4];
                scaled_38[0] = accum_1[24] * sa_lo_3 * sw_1[12];
                scaled_38[1] = accum_1[25] * sa_lo_3 * sw_1[13];
                scaled_38[2] = accum_1[26] * sa_hi_3 * sw_1[12];
                scaled_38[3] = accum_1[27] * sa_hi_3 * sw_1[13];
                unsigned int scaled_bf16_38[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_38[_lp*2 + 0], scaled_38[_lp*2+1 + 0]));
                    scaled_bf16_38[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_38[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_38[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_38[_pair]) << 16);
                    (&unpacked_38[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_38[_pair]) & 0xffff0000u);
                }
                rounded_3[24] = unpacked_38[0];
                rounded_3[25] = unpacked_38[1];
                rounded_3[26] = unpacked_38[2];
                rounded_3[27] = unpacked_38[3];
                ss_lo_3 += __fmaf_rn(unpacked_38[0], unpacked_38[0], unpacked_38[1] * unpacked_38[1]);
                ss_hi_3 += __fmaf_rn(unpacked_38[2], unpacked_38[2], unpacked_38[3] * unpacked_38[3]);
                float scaled_39[4];
                scaled_39[0] = accum_1[28] * sa_lo_3 * sw_1[14];
                scaled_39[1] = accum_1[29] * sa_lo_3 * sw_1[15];
                scaled_39[2] = accum_1[30] * sa_hi_3 * sw_1[14];
                scaled_39[3] = accum_1[31] * sa_hi_3 * sw_1[15];
                unsigned int scaled_bf16_39[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_39[_lp*2 + 0], scaled_39[_lp*2+1 + 0]));
                    scaled_bf16_39[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_39[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_39[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_39[_pair]) << 16);
                    (&unpacked_39[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_39[_pair]) & 0xffff0000u);
                }
                rounded_3[28] = unpacked_39[0];
                rounded_3[29] = unpacked_39[1];
                rounded_3[30] = unpacked_39[2];
                rounded_3[31] = unpacked_39[3];
                ss_lo_3 += __fmaf_rn(unpacked_39[0], unpacked_39[0], unpacked_39[1] * unpacked_39[1]);
                ss_hi_3 += __fmaf_rn(unpacked_39[2], unpacked_39[2], unpacked_39[3] * unpacked_39[3]);
                float rstd_lo_3 = 1.0f;
                float rstd_hi_3 = 1.0f;
                if (w_kind_1 < 2) {
                    float _shfl_xor_sync_0_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo_3, 1);
                    ss_lo_3 += _shfl_xor_sync_0_1;
                    float _shfl_xor_sync_1_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo_3, 2);
                    ss_lo_3 += _shfl_xor_sync_1_1;
                    float _shfl_xor_sync_2_1 = __shfl_xor_sync(0xFFFFFFFF, ss_hi_3, 1);
                    ss_hi_3 += _shfl_xor_sync_2_1;
                    float _shfl_xor_sync_3_1 = __shfl_xor_sync(0xFFFFFFFF, ss_hi_3, 2);
                    ss_hi_3 += _shfl_xor_sync_3_1;
                    int srow_lo_4 = warp_m_1 * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n_1 * 64 + srow_lo_4] = ss_lo_3;
                        rowstat[warp_n_1 * 64 + srow_lo_4 + 8] = ss_hi_3;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float other_lo_4 = rowstat[(1 - warp_n_1) * 64 + srow_lo_4];
                    float other_hi_4 = rowstat[(1 - warp_n_1) * 64 + srow_lo_4 + 8];
                    float _fdiv_5_1 = __fdiv_rn(ss_lo_3 + other_lo_4, 128.0f);
                    float _rsqrt_4_1 = rsqrtf(_fdiv_5_1 + eps);
                    rstd_lo_3 = _rsqrt_4_1;
                    float _fdiv_7_1 = __fdiv_rn(ss_hi_3 + other_hi_4, 128.0f);
                    float _rsqrt_6_1 = rsqrtf(_fdiv_7_1 + eps);
                    rstd_hi_3 = _rsqrt_6_1;
                }
                int srow_st_3 = warp_m_1 * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float normed_32[4];
                if (w_kind_1 < 2) {
                    normed_32[0] = rounded_3[0] * rstd_lo_3 * nw_1[0];
                    normed_32[1] = rounded_3[1] * rstd_lo_3 * nw_1[1];
                    normed_32[2] = rounded_3[2] * rstd_hi_3 * nw_1[0];
                    normed_32[3] = rounded_3[3] * rstd_hi_3 * nw_1[1];
                } else {
                    normed_32[0] = rounded_3[0];
                    normed_32[1] = rounded_3[1];
                    normed_32[2] = rounded_3[2];
                    normed_32[3] = rounded_3[3];
                }
                unsigned int normed_bf16_32[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_32[_lp*2 + 0], normed_32[_lp*2+1 + 0]));
                    normed_bf16_32[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3 * 256) + (unsigned int)(warp_n_1 * 64 * 2 ^ (srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_32[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_32[1]))
                    : "memory");
                float normed_33[4];
                if (w_kind_1 < 2) {
                    normed_33[0] = rounded_3[4] * rstd_lo_3 * nw_1[2];
                    normed_33[1] = rounded_3[5] * rstd_lo_3 * nw_1[3];
                    normed_33[2] = rounded_3[6] * rstd_hi_3 * nw_1[2];
                    normed_33[3] = rounded_3[7] * rstd_hi_3 * nw_1[3];
                } else {
                    normed_33[0] = rounded_3[4];
                    normed_33[1] = rounded_3[5];
                    normed_33[2] = rounded_3[6];
                    normed_33[3] = rounded_3[7];
                }
                unsigned int normed_bf16_33[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_33[_lp*2 + 0], normed_33[_lp*2+1 + 0]));
                    normed_bf16_33[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3 * 256) + (unsigned int)((warp_n_1 * 64 + 8) * 2 ^ (srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_33[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_33[1]))
                    : "memory");
                float normed_34[4];
                if (w_kind_1 < 2) {
                    normed_34[0] = rounded_3[8] * rstd_lo_3 * nw_1[4];
                    normed_34[1] = rounded_3[9] * rstd_lo_3 * nw_1[5];
                    normed_34[2] = rounded_3[10] * rstd_hi_3 * nw_1[4];
                    normed_34[3] = rounded_3[11] * rstd_hi_3 * nw_1[5];
                } else {
                    normed_34[0] = rounded_3[8];
                    normed_34[1] = rounded_3[9];
                    normed_34[2] = rounded_3[10];
                    normed_34[3] = rounded_3[11];
                }
                unsigned int normed_bf16_34[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_34[_lp*2 + 0], normed_34[_lp*2+1 + 0]));
                    normed_bf16_34[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3 * 256) + (unsigned int)((warp_n_1 * 64 + 16) * 2 ^ (srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_34[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_34[1]))
                    : "memory");
                float normed_35[4];
                if (w_kind_1 < 2) {
                    normed_35[0] = rounded_3[12] * rstd_lo_3 * nw_1[6];
                    normed_35[1] = rounded_3[13] * rstd_lo_3 * nw_1[7];
                    normed_35[2] = rounded_3[14] * rstd_hi_3 * nw_1[6];
                    normed_35[3] = rounded_3[15] * rstd_hi_3 * nw_1[7];
                } else {
                    normed_35[0] = rounded_3[12];
                    normed_35[1] = rounded_3[13];
                    normed_35[2] = rounded_3[14];
                    normed_35[3] = rounded_3[15];
                }
                unsigned int normed_bf16_35[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_35[_lp*2 + 0], normed_35[_lp*2+1 + 0]));
                    normed_bf16_35[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3 * 256) + (unsigned int)((warp_n_1 * 64 + 24) * 2 ^ (srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_35[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_35[1]))
                    : "memory");
                float normed_36[4];
                if (w_kind_1 < 2) {
                    normed_36[0] = rounded_3[16] * rstd_lo_3 * nw_1[8];
                    normed_36[1] = rounded_3[17] * rstd_lo_3 * nw_1[9];
                    normed_36[2] = rounded_3[18] * rstd_hi_3 * nw_1[8];
                    normed_36[3] = rounded_3[19] * rstd_hi_3 * nw_1[9];
                } else {
                    normed_36[0] = rounded_3[16];
                    normed_36[1] = rounded_3[17];
                    normed_36[2] = rounded_3[18];
                    normed_36[3] = rounded_3[19];
                }
                unsigned int normed_bf16_36[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_36[_lp*2 + 0], normed_36[_lp*2+1 + 0]));
                    normed_bf16_36[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3 * 256) + (unsigned int)((warp_n_1 * 64 + 32) * 2 ^ (srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_36[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_36[1]))
                    : "memory");
                float normed_37[4];
                if (w_kind_1 < 2) {
                    normed_37[0] = rounded_3[20] * rstd_lo_3 * nw_1[10];
                    normed_37[1] = rounded_3[21] * rstd_lo_3 * nw_1[11];
                    normed_37[2] = rounded_3[22] * rstd_hi_3 * nw_1[10];
                    normed_37[3] = rounded_3[23] * rstd_hi_3 * nw_1[11];
                } else {
                    normed_37[0] = rounded_3[20];
                    normed_37[1] = rounded_3[21];
                    normed_37[2] = rounded_3[22];
                    normed_37[3] = rounded_3[23];
                }
                unsigned int normed_bf16_37[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_37[_lp*2 + 0], normed_37[_lp*2+1 + 0]));
                    normed_bf16_37[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3 * 256) + (unsigned int)((warp_n_1 * 64 + 40) * 2 ^ (srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_37[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_37[1]))
                    : "memory");
                float normed_38[4];
                if (w_kind_1 < 2) {
                    normed_38[0] = rounded_3[24] * rstd_lo_3 * nw_1[12];
                    normed_38[1] = rounded_3[25] * rstd_lo_3 * nw_1[13];
                    normed_38[2] = rounded_3[26] * rstd_hi_3 * nw_1[12];
                    normed_38[3] = rounded_3[27] * rstd_hi_3 * nw_1[13];
                } else {
                    normed_38[0] = rounded_3[24];
                    normed_38[1] = rounded_3[25];
                    normed_38[2] = rounded_3[26];
                    normed_38[3] = rounded_3[27];
                }
                unsigned int normed_bf16_38[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_38[_lp*2 + 0], normed_38[_lp*2+1 + 0]));
                    normed_bf16_38[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3 * 256) + (unsigned int)((warp_n_1 * 64 + 48) * 2 ^ (srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_38[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_38[1]))
                    : "memory");
                float normed_39[4];
                if (w_kind_1 < 2) {
                    normed_39[0] = rounded_3[28] * rstd_lo_3 * nw_1[14];
                    normed_39[1] = rounded_3[29] * rstd_lo_3 * nw_1[15];
                    normed_39[2] = rounded_3[30] * rstd_hi_3 * nw_1[14];
                    normed_39[3] = rounded_3[31] * rstd_hi_3 * nw_1[15];
                } else {
                    normed_39[0] = rounded_3[28];
                    normed_39[1] = rounded_3[29];
                    normed_39[2] = rounded_3[30];
                    normed_39[3] = rounded_3[31];
                }
                unsigned int normed_bf16_39[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_39[_lp*2 + 0], normed_39[_lp*2+1 + 0]));
                    normed_bf16_39[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3 * 256) + (unsigned int)((warp_n_1 * 64 + 56) * 2 ^ (srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_39[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_39[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int srow_3 = role_tid_1 >> 2;
                int c_3 = role_tid_1 & 3;
                int grow_3 = tile_m_1 * 256 + srow_3;
                int chunk_lo_3 = ((c_3 < 3) ? c_3 * 16 : 96);
                int chunk_hi_3 = ((c_3 < 3) ? chunk_lo_3 + 48 : 112);
                if (grow_3 < M) {
                    unsigned int x_lo_4[8];
                    unsigned int x_hi_4[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_4[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_3 * 256) + (unsigned int)(chunk_lo_3 * 2 ^ (srow_3 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_4[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_3 * 256) + (unsigned int)(chunk_lo_3 * 2 + 16 ^ (srow_3 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_4[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_3 * 256) + (unsigned int)(chunk_hi_3 * 2 ^ (srow_3 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_4[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_3 * 256) + (unsigned int)(chunk_hi_3 * 2 + 16 ^ (srow_3 & 7) << 4)));
                    if (c_3 < 3) {
                        if (w_kind_1 < 2) {
                            float x_lo_f_4[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_lo_f_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_4[_pair]) << 16);
                                (&x_lo_f_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_4[_pair]) & 0xffff0000u);
                            }
                            float x_hi_f_4[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_hi_f_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_4[_pair]) << 16);
                                (&x_hi_f_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_4[_pair]) & 0xffff0000u);
                            }
                            float v_16[8];
                            {
                                const uint4* _vptr_16 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_3 * 96 + chunk_lo_3) + 0);
                                uint4 _vld_16[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_16[_blk] = _vptr_16[_blk];
                                    uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_16[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                                        (&v_16[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_17[8];
                            {
                                const uint4* _vptr_17 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_3 * 96 + chunk_lo_3 + 8) + 0);
                                uint4 _vld_17[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_17[_blk] = _vptr_17[_blk];
                                    uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_17[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
                                        (&v_17[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_18[8];
                            {
                                const uint4* _vptr_18 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_3 * 96 + 48 + chunk_lo_3) + 0);
                                uint4 _vld_18[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_18[_blk] = _vptr_18[_blk];
                                    uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_18[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                                        (&v_18[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_19[8];
                            {
                                const uint4* _vptr_19 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_3 * 96 + 48 + chunk_lo_3 + 8) + 0);
                                uint4 _vld_19[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_19[_blk] = _vptr_19[_blk];
                                    uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_19[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                                        (&v_19[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float o_lo_4[16];
                            float o_hi_4[16];
                            float cv0_28 = v_16[0];
                            float sv0_28 = v_18[0];
                            float cv1_28 = v_17[0];
                            float sv1_28 = v_19[0];
                            o_lo_4[0] = __fmaf_rn(x_lo_f_4[0], cv0_28, -(x_hi_f_4[0] * sv0_28));
                            o_hi_4[0] = __fmaf_rn(x_hi_f_4[0], cv0_28, x_lo_f_4[0] * sv0_28);
                            o_lo_4[8] = __fmaf_rn(x_lo_f_4[8], cv1_28, -(x_hi_f_4[8] * sv1_28));
                            o_hi_4[8] = __fmaf_rn(x_hi_f_4[8], cv1_28, x_lo_f_4[8] * sv1_28);
                            float cv0_28_1 = v_16[1];
                            float sv0_28_1 = v_18[1];
                            float cv1_28_1 = v_17[1];
                            float sv1_28_1 = v_19[1];
                            o_lo_4[1] = __fmaf_rn(x_lo_f_4[1], cv0_28_1, -(x_hi_f_4[1] * sv0_28_1));
                            o_hi_4[1] = __fmaf_rn(x_hi_f_4[1], cv0_28_1, x_lo_f_4[1] * sv0_28_1);
                            o_lo_4[9] = __fmaf_rn(x_lo_f_4[9], cv1_28_1, -(x_hi_f_4[9] * sv1_28_1));
                            o_hi_4[9] = __fmaf_rn(x_hi_f_4[9], cv1_28_1, x_lo_f_4[9] * sv1_28_1);
                            float cv0_29 = v_16[2];
                            float sv0_29 = v_18[2];
                            float cv1_29 = v_17[2];
                            float sv1_29 = v_19[2];
                            o_lo_4[2] = __fmaf_rn(x_lo_f_4[2], cv0_29, -(x_hi_f_4[2] * sv0_29));
                            o_hi_4[2] = __fmaf_rn(x_hi_f_4[2], cv0_29, x_lo_f_4[2] * sv0_29);
                            o_lo_4[10] = __fmaf_rn(x_lo_f_4[10], cv1_29, -(x_hi_f_4[10] * sv1_29));
                            o_hi_4[10] = __fmaf_rn(x_hi_f_4[10], cv1_29, x_lo_f_4[10] * sv1_29);
                            float cv0_30 = v_16[3];
                            float sv0_30 = v_18[3];
                            float cv1_30 = v_17[3];
                            float sv1_30 = v_19[3];
                            o_lo_4[3] = __fmaf_rn(x_lo_f_4[3], cv0_30, -(x_hi_f_4[3] * sv0_30));
                            o_hi_4[3] = __fmaf_rn(x_hi_f_4[3], cv0_30, x_lo_f_4[3] * sv0_30);
                            o_lo_4[11] = __fmaf_rn(x_lo_f_4[11], cv1_30, -(x_hi_f_4[11] * sv1_30));
                            o_hi_4[11] = __fmaf_rn(x_hi_f_4[11], cv1_30, x_lo_f_4[11] * sv1_30);
                            float cv0_31 = v_16[4];
                            float sv0_31 = v_18[4];
                            float cv1_31 = v_17[4];
                            float sv1_31 = v_19[4];
                            o_lo_4[4] = __fmaf_rn(x_lo_f_4[4], cv0_31, -(x_hi_f_4[4] * sv0_31));
                            o_hi_4[4] = __fmaf_rn(x_hi_f_4[4], cv0_31, x_lo_f_4[4] * sv0_31);
                            o_lo_4[12] = __fmaf_rn(x_lo_f_4[12], cv1_31, -(x_hi_f_4[12] * sv1_31));
                            o_hi_4[12] = __fmaf_rn(x_hi_f_4[12], cv1_31, x_lo_f_4[12] * sv1_31);
                            float cv0_32 = v_16[5];
                            float sv0_32 = v_18[5];
                            float cv1_32 = v_17[5];
                            float sv1_32 = v_19[5];
                            o_lo_4[5] = __fmaf_rn(x_lo_f_4[5], cv0_32, -(x_hi_f_4[5] * sv0_32));
                            o_hi_4[5] = __fmaf_rn(x_hi_f_4[5], cv0_32, x_lo_f_4[5] * sv0_32);
                            o_lo_4[13] = __fmaf_rn(x_lo_f_4[13], cv1_32, -(x_hi_f_4[13] * sv1_32));
                            o_hi_4[13] = __fmaf_rn(x_hi_f_4[13], cv1_32, x_lo_f_4[13] * sv1_32);
                            float cv0_33 = v_16[6];
                            float sv0_33 = v_18[6];
                            float cv1_33 = v_17[6];
                            float sv1_33 = v_19[6];
                            o_lo_4[6] = __fmaf_rn(x_lo_f_4[6], cv0_33, -(x_hi_f_4[6] * sv0_33));
                            o_hi_4[6] = __fmaf_rn(x_hi_f_4[6], cv0_33, x_lo_f_4[6] * sv0_33);
                            o_lo_4[14] = __fmaf_rn(x_lo_f_4[14], cv1_33, -(x_hi_f_4[14] * sv1_33));
                            o_hi_4[14] = __fmaf_rn(x_hi_f_4[14], cv1_33, x_lo_f_4[14] * sv1_33);
                            float cv0_34 = v_16[7];
                            float sv0_34 = v_18[7];
                            float cv1_34 = v_17[7];
                            float sv1_34 = v_19[7];
                            o_lo_4[7] = __fmaf_rn(x_lo_f_4[7], cv0_34, -(x_hi_f_4[7] * sv0_34));
                            o_hi_4[7] = __fmaf_rn(x_hi_f_4[7], cv0_34, x_lo_f_4[7] * sv0_34);
                            o_lo_4[15] = __fmaf_rn(x_lo_f_4[15], cv1_34, -(x_hi_f_4[15] * sv1_34));
                            o_hi_4[15] = __fmaf_rn(x_hi_f_4[15], cv1_34, x_lo_f_4[15] * sv1_34);
                            unsigned int o_lo_bf16_4[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_lo_4[_lp*2 + 0], o_lo_4[_lp*2+1 + 0]));
                                o_lo_bf16_4[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int o_hi_bf16_4[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_hi_4[_lp*2 + 0], o_hi_4[_lp*2+1 + 0]));
                                o_hi_bf16_4[_lp] = *(uint32_t*)&_bf2;
                            }
                            x_lo_4[0] = o_lo_bf16_4[0];
                            x_hi_4[0] = o_hi_bf16_4[0];
                            x_lo_4[1] = o_lo_bf16_4[1];
                            x_hi_4[1] = o_hi_bf16_4[1];
                            x_lo_4[2] = o_lo_bf16_4[2];
                            x_hi_4[2] = o_hi_bf16_4[2];
                            x_lo_4[3] = o_lo_bf16_4[3];
                            x_hi_4[3] = o_hi_bf16_4[3];
                            x_lo_4[4] = o_lo_bf16_4[4];
                            x_hi_4[4] = o_hi_bf16_4[4];
                            x_lo_4[5] = o_lo_bf16_4[5];
                            x_hi_4[5] = o_hi_bf16_4[5];
                            x_lo_4[6] = o_lo_bf16_4[6];
                            x_hi_4[6] = o_hi_bf16_4[6];
                            x_lo_4[7] = o_lo_bf16_4[7];
                            x_hi_4[7] = o_hi_bf16_4[7];
                        }
                    }
                    unsigned int q_words_4[4];
                    float q_vals_4[16];
                    float f_8[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_4[_pair]) << 16);
                        (&f_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_4[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_8;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_8) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_8[0]), fabsf(f_8[1])), fabsf(f_8[2])), fabsf(f_8[3])), fabsf(f_8[4])), fabsf(f_8[5])), fabsf(f_8[6])), fabsf(f_8[7])), fabsf(f_8[8])), fabsf(f_8[9])), fabsf(f_8[10])), fabsf(f_8[11])), fabsf(f_8[12])), fabsf(f_8[13])), fabsf(f_8[14])), fabsf(f_8[15])) * sf_mul_sel_1));
                    uint16_t _e4m3x2_decode_20 = (uint16_t)((unsigned int)sf_pair_8 & 0xFFu);
                    uint32_t _f16x2_decode_20;
                    float sf_dec_8;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_20) : "h"(_e4m3x2_decode_20));
                    uint16_t _f16_decode_20 = (uint16_t)_f16x2_decode_20;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_8) : "h"(_f16_decode_20));
                    float out_s_8 = __fdiv_rn(out_scale_sel_1, sf_dec_8);
                    q_vals_4[0] = f_8[0] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[1] = f_8[1] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[2] = f_8[2] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[3] = f_8[3] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[4] = f_8[4] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[5] = f_8[5] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[6] = f_8[6] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[7] = f_8[7] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[8] = f_8[8] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[9] = f_8[9] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[10] = f_8[10] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[11] = f_8[11] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[12] = f_8[12] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[13] = f_8[13] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[14] = f_8[14] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    q_vals_4[15] = f_8[15] * ((sf_dec_8 != 0.0f) ? out_s_8 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_4[0]) : "f"(q_vals_4[0]), "f"(q_vals_4[1]), "f"(q_vals_4[2]), "f"(q_vals_4[3]), "f"(q_vals_4[4]), "f"(q_vals_4[5]), "f"(q_vals_4[6]), "f"(q_vals_4[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_4[1]) : "f"(q_vals_4[8]), "f"(q_vals_4[9]), "f"(q_vals_4[10]), "f"(q_vals_4[11]), "f"(q_vals_4[12]), "f"(q_vals_4[13]), "f"(q_vals_4[14]), "f"(q_vals_4[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_3 * 128) + (unsigned int)(chunk_lo_3 >> 1 ^ (srow_3 & 7) << 4)), "r"(q_words_4[0]), "r"(q_words_4[1]) : "memory");
                    {
                        uint32_t _byte_21 = static_cast<uint32_t>((unsigned int)sf_pair_8 & 127) & 0xFFu;
                        uint32_t _addr_21 = static_cast<uint32_t>(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_3 * 128) + (unsigned int)(64 + (chunk_lo_3 >> 4) ^ (srow_3 & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_21), "r"(_byte_21) : "memory");
                    }
                    float f_9[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_4[_pair]) << 16);
                        (&f_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_4[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_9;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_9) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_9[0]), fabsf(f_9[1])), fabsf(f_9[2])), fabsf(f_9[3])), fabsf(f_9[4])), fabsf(f_9[5])), fabsf(f_9[6])), fabsf(f_9[7])), fabsf(f_9[8])), fabsf(f_9[9])), fabsf(f_9[10])), fabsf(f_9[11])), fabsf(f_9[12])), fabsf(f_9[13])), fabsf(f_9[14])), fabsf(f_9[15])) * sf_mul_sel_1));
                    uint16_t _e4m3x2_decode_22 = (uint16_t)((unsigned int)sf_pair_9 & 0xFFu);
                    uint32_t _f16x2_decode_22;
                    float sf_dec_9;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_22) : "h"(_e4m3x2_decode_22));
                    uint16_t _f16_decode_22 = (uint16_t)_f16x2_decode_22;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_9) : "h"(_f16_decode_22));
                    float out_s_9 = __fdiv_rn(out_scale_sel_1, sf_dec_9);
                    q_vals_4[0] = f_9[0] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[1] = f_9[1] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[2] = f_9[2] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[3] = f_9[3] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[4] = f_9[4] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[5] = f_9[5] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[6] = f_9[6] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[7] = f_9[7] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[8] = f_9[8] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[9] = f_9[9] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[10] = f_9[10] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[11] = f_9[11] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[12] = f_9[12] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[13] = f_9[13] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[14] = f_9[14] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    q_vals_4[15] = f_9[15] * ((sf_dec_9 != 0.0f) ? out_s_9 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_4[0]) : "f"(q_vals_4[0]), "f"(q_vals_4[1]), "f"(q_vals_4[2]), "f"(q_vals_4[3]), "f"(q_vals_4[4]), "f"(q_vals_4[5]), "f"(q_vals_4[6]), "f"(q_vals_4[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_4[1]) : "f"(q_vals_4[8]), "f"(q_vals_4[9]), "f"(q_vals_4[10]), "f"(q_vals_4[11]), "f"(q_vals_4[12]), "f"(q_vals_4[13]), "f"(q_vals_4[14]), "f"(q_vals_4[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_3 * 128) + (unsigned int)(chunk_hi_3 >> 1 ^ (srow_3 & 7) << 4)), "r"(q_words_4[0]), "r"(q_words_4[1]) : "memory");
                    {
                        uint32_t _byte_23 = static_cast<uint32_t>((unsigned int)sf_pair_9 & 127) & 0xFFu;
                        uint32_t _addr_23 = static_cast<uint32_t>(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_3 * 128) + (unsigned int)(64 + (chunk_hi_3 >> 4) ^ (srow_3 & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_23), "r"(_byte_23) : "memory");
                    }
                }
                __syncwarp();
                int s_row_4 = (role_tid_1 >> 5) * 8 + lane / 4;
                int s_piece_4 = lane % 4;
                unsigned int s_words_4[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_4[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_row_4 * 128) + (unsigned int)(s_piece_4 * 16 ^ (s_row_4 & 7) << 4)));
                int s_dst_4 = ((tile_m_1 * 256 + s_row_4) * 56 + w_head_1) * 16 + s_piece_4 * 4;
                if (tile_m_1 * 256 + s_row_4 < M) {
                    if (w_kind_1 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_4)[0] = reinterpret_cast<int4*>(s_words_4)[0];
                    } else if (w_kind_1 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_4)[0] = reinterpret_cast<int4*>(s_words_4)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_4)[0] = reinterpret_cast<int4*>(s_words_4)[0];
                    }
                }
                int s_sf_row_3 = (role_tid_1 >> 5) * 8 + (lane & 7);
                unsigned int s_sf_words_3[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words_3[(0) + 1]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_sf_row_3 * 128) + (unsigned int)(64 ^ (s_sf_row_3 & 7) << 4)));
                int s_sf_dst_3 = ((tile_m_1 * 256 + s_sf_row_3) * 56 + w_head_1) * 8;
                if (lane < 8) {
                    if (tile_m_1 * 256 + s_sf_row_3 < M) {
                        if (w_kind_1 == 0) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(q_sf + s_sf_dst_3))[0] = reinterpret_cast<int2*>(s_sf_words_3)[0];
                        } else if (w_kind_1 == 1) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(k_sf + s_sf_dst_3))[0] = reinterpret_cast<int2*>(s_sf_words_3)[0];
                        } else {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(v_sf + s_sf_dst_3))[0] = reinterpret_cast<int2*>(s_sf_words_3)[0];
                        }
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int row_lo_3_1 = tile_m_1 * 256 + 64 + warp_m_1 * 16 + (lane >> 2);
                int row_hi_3_1 = row_lo_3_1 + 8;
                float sa_lo_3_1 = ((row_lo_3_1 < M) ? act_scale[row_lo_3_1] : 0.0f);
                float sa_hi_3_1 = ((row_hi_3_1 < M) ? act_scale[row_hi_3_1] : 0.0f);
                float rounded_3_1[32];
                float ss_lo_3_1 = 0.0f;
                float ss_hi_3_1 = 0.0f;
                float scaled_40[4];
                scaled_40[0] = accum_1[32] * sa_lo_3_1 * sw_1[0];
                scaled_40[1] = accum_1[33] * sa_lo_3_1 * sw_1[1];
                scaled_40[2] = accum_1[34] * sa_hi_3_1 * sw_1[0];
                scaled_40[3] = accum_1[35] * sa_hi_3_1 * sw_1[1];
                unsigned int scaled_bf16_40[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_40[_lp*2 + 0], scaled_40[_lp*2+1 + 0]));
                    scaled_bf16_40[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_40[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_40[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_40[_pair]) << 16);
                    (&unpacked_40[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_40[_pair]) & 0xffff0000u);
                }
                rounded_3_1[0] = unpacked_40[0];
                rounded_3_1[1] = unpacked_40[1];
                rounded_3_1[2] = unpacked_40[2];
                rounded_3_1[3] = unpacked_40[3];
                ss_lo_3_1 += __fmaf_rn(unpacked_40[0], unpacked_40[0], unpacked_40[1] * unpacked_40[1]);
                ss_hi_3_1 += __fmaf_rn(unpacked_40[2], unpacked_40[2], unpacked_40[3] * unpacked_40[3]);
                float scaled_41[4];
                scaled_41[0] = accum_1[36] * sa_lo_3_1 * sw_1[2];
                scaled_41[1] = accum_1[37] * sa_lo_3_1 * sw_1[3];
                scaled_41[2] = accum_1[38] * sa_hi_3_1 * sw_1[2];
                scaled_41[3] = accum_1[39] * sa_hi_3_1 * sw_1[3];
                unsigned int scaled_bf16_41[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_41[_lp*2 + 0], scaled_41[_lp*2+1 + 0]));
                    scaled_bf16_41[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_41[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_41[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_41[_pair]) << 16);
                    (&unpacked_41[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_41[_pair]) & 0xffff0000u);
                }
                rounded_3_1[4] = unpacked_41[0];
                rounded_3_1[5] = unpacked_41[1];
                rounded_3_1[6] = unpacked_41[2];
                rounded_3_1[7] = unpacked_41[3];
                ss_lo_3_1 += __fmaf_rn(unpacked_41[0], unpacked_41[0], unpacked_41[1] * unpacked_41[1]);
                ss_hi_3_1 += __fmaf_rn(unpacked_41[2], unpacked_41[2], unpacked_41[3] * unpacked_41[3]);
                float scaled_42[4];
                scaled_42[0] = accum_1[40] * sa_lo_3_1 * sw_1[4];
                scaled_42[1] = accum_1[41] * sa_lo_3_1 * sw_1[5];
                scaled_42[2] = accum_1[42] * sa_hi_3_1 * sw_1[4];
                scaled_42[3] = accum_1[43] * sa_hi_3_1 * sw_1[5];
                unsigned int scaled_bf16_42[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_42[_lp*2 + 0], scaled_42[_lp*2+1 + 0]));
                    scaled_bf16_42[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_42[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_42[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_42[_pair]) << 16);
                    (&unpacked_42[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_42[_pair]) & 0xffff0000u);
                }
                rounded_3_1[8] = unpacked_42[0];
                rounded_3_1[9] = unpacked_42[1];
                rounded_3_1[10] = unpacked_42[2];
                rounded_3_1[11] = unpacked_42[3];
                ss_lo_3_1 += __fmaf_rn(unpacked_42[0], unpacked_42[0], unpacked_42[1] * unpacked_42[1]);
                ss_hi_3_1 += __fmaf_rn(unpacked_42[2], unpacked_42[2], unpacked_42[3] * unpacked_42[3]);
                float scaled_43[4];
                scaled_43[0] = accum_1[44] * sa_lo_3_1 * sw_1[6];
                scaled_43[1] = accum_1[45] * sa_lo_3_1 * sw_1[7];
                scaled_43[2] = accum_1[46] * sa_hi_3_1 * sw_1[6];
                scaled_43[3] = accum_1[47] * sa_hi_3_1 * sw_1[7];
                unsigned int scaled_bf16_43[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_43[_lp*2 + 0], scaled_43[_lp*2+1 + 0]));
                    scaled_bf16_43[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_43[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_43[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_43[_pair]) << 16);
                    (&unpacked_43[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_43[_pair]) & 0xffff0000u);
                }
                rounded_3_1[12] = unpacked_43[0];
                rounded_3_1[13] = unpacked_43[1];
                rounded_3_1[14] = unpacked_43[2];
                rounded_3_1[15] = unpacked_43[3];
                ss_lo_3_1 += __fmaf_rn(unpacked_43[0], unpacked_43[0], unpacked_43[1] * unpacked_43[1]);
                ss_hi_3_1 += __fmaf_rn(unpacked_43[2], unpacked_43[2], unpacked_43[3] * unpacked_43[3]);
                float scaled_44[4];
                scaled_44[0] = accum_1[48] * sa_lo_3_1 * sw_1[8];
                scaled_44[1] = accum_1[49] * sa_lo_3_1 * sw_1[9];
                scaled_44[2] = accum_1[50] * sa_hi_3_1 * sw_1[8];
                scaled_44[3] = accum_1[51] * sa_hi_3_1 * sw_1[9];
                unsigned int scaled_bf16_44[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_44[_lp*2 + 0], scaled_44[_lp*2+1 + 0]));
                    scaled_bf16_44[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_44[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_44[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_44[_pair]) << 16);
                    (&unpacked_44[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_44[_pair]) & 0xffff0000u);
                }
                rounded_3_1[16] = unpacked_44[0];
                rounded_3_1[17] = unpacked_44[1];
                rounded_3_1[18] = unpacked_44[2];
                rounded_3_1[19] = unpacked_44[3];
                ss_lo_3_1 += __fmaf_rn(unpacked_44[0], unpacked_44[0], unpacked_44[1] * unpacked_44[1]);
                ss_hi_3_1 += __fmaf_rn(unpacked_44[2], unpacked_44[2], unpacked_44[3] * unpacked_44[3]);
                float scaled_45[4];
                scaled_45[0] = accum_1[52] * sa_lo_3_1 * sw_1[10];
                scaled_45[1] = accum_1[53] * sa_lo_3_1 * sw_1[11];
                scaled_45[2] = accum_1[54] * sa_hi_3_1 * sw_1[10];
                scaled_45[3] = accum_1[55] * sa_hi_3_1 * sw_1[11];
                unsigned int scaled_bf16_45[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_45[_lp*2 + 0], scaled_45[_lp*2+1 + 0]));
                    scaled_bf16_45[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_45[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_45[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_45[_pair]) << 16);
                    (&unpacked_45[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_45[_pair]) & 0xffff0000u);
                }
                rounded_3_1[20] = unpacked_45[0];
                rounded_3_1[21] = unpacked_45[1];
                rounded_3_1[22] = unpacked_45[2];
                rounded_3_1[23] = unpacked_45[3];
                ss_lo_3_1 += __fmaf_rn(unpacked_45[0], unpacked_45[0], unpacked_45[1] * unpacked_45[1]);
                ss_hi_3_1 += __fmaf_rn(unpacked_45[2], unpacked_45[2], unpacked_45[3] * unpacked_45[3]);
                float scaled_46[4];
                scaled_46[0] = accum_1[56] * sa_lo_3_1 * sw_1[12];
                scaled_46[1] = accum_1[57] * sa_lo_3_1 * sw_1[13];
                scaled_46[2] = accum_1[58] * sa_hi_3_1 * sw_1[12];
                scaled_46[3] = accum_1[59] * sa_hi_3_1 * sw_1[13];
                unsigned int scaled_bf16_46[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_46[_lp*2 + 0], scaled_46[_lp*2+1 + 0]));
                    scaled_bf16_46[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_46[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_46[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_46[_pair]) << 16);
                    (&unpacked_46[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_46[_pair]) & 0xffff0000u);
                }
                rounded_3_1[24] = unpacked_46[0];
                rounded_3_1[25] = unpacked_46[1];
                rounded_3_1[26] = unpacked_46[2];
                rounded_3_1[27] = unpacked_46[3];
                ss_lo_3_1 += __fmaf_rn(unpacked_46[0], unpacked_46[0], unpacked_46[1] * unpacked_46[1]);
                ss_hi_3_1 += __fmaf_rn(unpacked_46[2], unpacked_46[2], unpacked_46[3] * unpacked_46[3]);
                float scaled_47[4];
                scaled_47[0] = accum_1[60] * sa_lo_3_1 * sw_1[14];
                scaled_47[1] = accum_1[61] * sa_lo_3_1 * sw_1[15];
                scaled_47[2] = accum_1[62] * sa_hi_3_1 * sw_1[14];
                scaled_47[3] = accum_1[63] * sa_hi_3_1 * sw_1[15];
                unsigned int scaled_bf16_47[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_47[_lp*2 + 0], scaled_47[_lp*2+1 + 0]));
                    scaled_bf16_47[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_47[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_47[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_47[_pair]) << 16);
                    (&unpacked_47[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_47[_pair]) & 0xffff0000u);
                }
                rounded_3_1[28] = unpacked_47[0];
                rounded_3_1[29] = unpacked_47[1];
                rounded_3_1[30] = unpacked_47[2];
                rounded_3_1[31] = unpacked_47[3];
                ss_lo_3_1 += __fmaf_rn(unpacked_47[0], unpacked_47[0], unpacked_47[1] * unpacked_47[1]);
                ss_hi_3_1 += __fmaf_rn(unpacked_47[2], unpacked_47[2], unpacked_47[3] * unpacked_47[3]);
                float rstd_lo_3_1 = 1.0f;
                float rstd_hi_3_1 = 1.0f;
                if (w_kind_1 < 2) {
                    float _shfl_xor_sync_24_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo_3_1, 1);
                    ss_lo_3_1 += _shfl_xor_sync_24_1;
                    float _shfl_xor_sync_25_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo_3_1, 2);
                    ss_lo_3_1 += _shfl_xor_sync_25_1;
                    float _shfl_xor_sync_26_1 = __shfl_xor_sync(0xFFFFFFFF, ss_hi_3_1, 1);
                    ss_hi_3_1 += _shfl_xor_sync_26_1;
                    float _shfl_xor_sync_27_1 = __shfl_xor_sync(0xFFFFFFFF, ss_hi_3_1, 2);
                    ss_hi_3_1 += _shfl_xor_sync_27_1;
                    int srow_lo_5 = warp_m_1 * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n_1 * 64 + srow_lo_5] = ss_lo_3_1;
                        rowstat[warp_n_1 * 64 + srow_lo_5 + 8] = ss_hi_3_1;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float other_lo_5 = rowstat[(1 - warp_n_1) * 64 + srow_lo_5];
                    float other_hi_5 = rowstat[(1 - warp_n_1) * 64 + srow_lo_5 + 8];
                    float _fdiv_29_1 = __fdiv_rn(ss_lo_3_1 + other_lo_5, 128.0f);
                    float _rsqrt_28_1 = rsqrtf(_fdiv_29_1 + eps);
                    rstd_lo_3_1 = _rsqrt_28_1;
                    float _fdiv_31_1 = __fdiv_rn(ss_hi_3_1 + other_hi_5, 128.0f);
                    float _rsqrt_30_1 = rsqrtf(_fdiv_31_1 + eps);
                    rstd_hi_3_1 = _rsqrt_30_1;
                }
                int srow_st_3_1 = warp_m_1 * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float normed_40[4];
                if (w_kind_1 < 2) {
                    normed_40[0] = rounded_3_1[0] * rstd_lo_3_1 * nw_1[0];
                    normed_40[1] = rounded_3_1[1] * rstd_lo_3_1 * nw_1[1];
                    normed_40[2] = rounded_3_1[2] * rstd_hi_3_1 * nw_1[0];
                    normed_40[3] = rounded_3_1[3] * rstd_hi_3_1 * nw_1[1];
                } else {
                    normed_40[0] = rounded_3_1[0];
                    normed_40[1] = rounded_3_1[1];
                    normed_40[2] = rounded_3_1[2];
                    normed_40[3] = rounded_3_1[3];
                }
                unsigned int normed_bf16_40[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_40[_lp*2 + 0], normed_40[_lp*2+1 + 0]));
                    normed_bf16_40[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_32 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3_1 * 256) + (unsigned int)(warp_n_1 * 64 * 2 ^ (srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_32), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_40[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_40[1]))
                    : "memory");
                float normed_41[4];
                if (w_kind_1 < 2) {
                    normed_41[0] = rounded_3_1[4] * rstd_lo_3_1 * nw_1[2];
                    normed_41[1] = rounded_3_1[5] * rstd_lo_3_1 * nw_1[3];
                    normed_41[2] = rounded_3_1[6] * rstd_hi_3_1 * nw_1[2];
                    normed_41[3] = rounded_3_1[7] * rstd_hi_3_1 * nw_1[3];
                } else {
                    normed_41[0] = rounded_3_1[4];
                    normed_41[1] = rounded_3_1[5];
                    normed_41[2] = rounded_3_1[6];
                    normed_41[3] = rounded_3_1[7];
                }
                unsigned int normed_bf16_41[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_41[_lp*2 + 0], normed_41[_lp*2+1 + 0]));
                    normed_bf16_41[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_33 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3_1 * 256) + (unsigned int)((warp_n_1 * 64 + 8) * 2 ^ (srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_33), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_41[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_41[1]))
                    : "memory");
                float normed_42[4];
                if (w_kind_1 < 2) {
                    normed_42[0] = rounded_3_1[8] * rstd_lo_3_1 * nw_1[4];
                    normed_42[1] = rounded_3_1[9] * rstd_lo_3_1 * nw_1[5];
                    normed_42[2] = rounded_3_1[10] * rstd_hi_3_1 * nw_1[4];
                    normed_42[3] = rounded_3_1[11] * rstd_hi_3_1 * nw_1[5];
                } else {
                    normed_42[0] = rounded_3_1[8];
                    normed_42[1] = rounded_3_1[9];
                    normed_42[2] = rounded_3_1[10];
                    normed_42[3] = rounded_3_1[11];
                }
                unsigned int normed_bf16_42[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_42[_lp*2 + 0], normed_42[_lp*2+1 + 0]));
                    normed_bf16_42[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3_1 * 256) + (unsigned int)((warp_n_1 * 64 + 16) * 2 ^ (srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_42[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_42[1]))
                    : "memory");
                float normed_43[4];
                if (w_kind_1 < 2) {
                    normed_43[0] = rounded_3_1[12] * rstd_lo_3_1 * nw_1[6];
                    normed_43[1] = rounded_3_1[13] * rstd_lo_3_1 * nw_1[7];
                    normed_43[2] = rounded_3_1[14] * rstd_hi_3_1 * nw_1[6];
                    normed_43[3] = rounded_3_1[15] * rstd_hi_3_1 * nw_1[7];
                } else {
                    normed_43[0] = rounded_3_1[12];
                    normed_43[1] = rounded_3_1[13];
                    normed_43[2] = rounded_3_1[14];
                    normed_43[3] = rounded_3_1[15];
                }
                unsigned int normed_bf16_43[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_43[_lp*2 + 0], normed_43[_lp*2+1 + 0]));
                    normed_bf16_43[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3_1 * 256) + (unsigned int)((warp_n_1 * 64 + 24) * 2 ^ (srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_43[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_43[1]))
                    : "memory");
                float normed_44[4];
                if (w_kind_1 < 2) {
                    normed_44[0] = rounded_3_1[16] * rstd_lo_3_1 * nw_1[8];
                    normed_44[1] = rounded_3_1[17] * rstd_lo_3_1 * nw_1[9];
                    normed_44[2] = rounded_3_1[18] * rstd_hi_3_1 * nw_1[8];
                    normed_44[3] = rounded_3_1[19] * rstd_hi_3_1 * nw_1[9];
                } else {
                    normed_44[0] = rounded_3_1[16];
                    normed_44[1] = rounded_3_1[17];
                    normed_44[2] = rounded_3_1[18];
                    normed_44[3] = rounded_3_1[19];
                }
                unsigned int normed_bf16_44[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_44[_lp*2 + 0], normed_44[_lp*2+1 + 0]));
                    normed_bf16_44[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_36 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3_1 * 256) + (unsigned int)((warp_n_1 * 64 + 32) * 2 ^ (srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_36), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_44[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_44[1]))
                    : "memory");
                float normed_45[4];
                if (w_kind_1 < 2) {
                    normed_45[0] = rounded_3_1[20] * rstd_lo_3_1 * nw_1[10];
                    normed_45[1] = rounded_3_1[21] * rstd_lo_3_1 * nw_1[11];
                    normed_45[2] = rounded_3_1[22] * rstd_hi_3_1 * nw_1[10];
                    normed_45[3] = rounded_3_1[23] * rstd_hi_3_1 * nw_1[11];
                } else {
                    normed_45[0] = rounded_3_1[20];
                    normed_45[1] = rounded_3_1[21];
                    normed_45[2] = rounded_3_1[22];
                    normed_45[3] = rounded_3_1[23];
                }
                unsigned int normed_bf16_45[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_45[_lp*2 + 0], normed_45[_lp*2+1 + 0]));
                    normed_bf16_45[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_37 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3_1 * 256) + (unsigned int)((warp_n_1 * 64 + 40) * 2 ^ (srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_37), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_45[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_45[1]))
                    : "memory");
                float normed_46[4];
                if (w_kind_1 < 2) {
                    normed_46[0] = rounded_3_1[24] * rstd_lo_3_1 * nw_1[12];
                    normed_46[1] = rounded_3_1[25] * rstd_lo_3_1 * nw_1[13];
                    normed_46[2] = rounded_3_1[26] * rstd_hi_3_1 * nw_1[12];
                    normed_46[3] = rounded_3_1[27] * rstd_hi_3_1 * nw_1[13];
                } else {
                    normed_46[0] = rounded_3_1[24];
                    normed_46[1] = rounded_3_1[25];
                    normed_46[2] = rounded_3_1[26];
                    normed_46[3] = rounded_3_1[27];
                }
                unsigned int normed_bf16_46[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_46[_lp*2 + 0], normed_46[_lp*2+1 + 0]));
                    normed_bf16_46[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_38 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3_1 * 256) + (unsigned int)((warp_n_1 * 64 + 48) * 2 ^ (srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_38), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_46[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_46[1]))
                    : "memory");
                float normed_47[4];
                if (w_kind_1 < 2) {
                    normed_47[0] = rounded_3_1[28] * rstd_lo_3_1 * nw_1[14];
                    normed_47[1] = rounded_3_1[29] * rstd_lo_3_1 * nw_1[15];
                    normed_47[2] = rounded_3_1[30] * rstd_hi_3_1 * nw_1[14];
                    normed_47[3] = rounded_3_1[31] * rstd_hi_3_1 * nw_1[15];
                } else {
                    normed_47[0] = rounded_3_1[28];
                    normed_47[1] = rounded_3_1[29];
                    normed_47[2] = rounded_3_1[30];
                    normed_47[3] = rounded_3_1[31];
                }
                unsigned int normed_bf16_47[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_47[_lp*2 + 0], normed_47[_lp*2+1 + 0]));
                    normed_bf16_47[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_39 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_3_1 * 256) + (unsigned int)((warp_n_1 * 64 + 56) * 2 ^ (srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_39), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_47[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_47[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int srow_3_1 = role_tid_1 >> 2;
                int c_3_1 = role_tid_1 & 3;
                int grow_3_1 = tile_m_1 * 256 + 64 + srow_3_1;
                int chunk_lo_3_1 = ((c_3_1 < 3) ? c_3_1 * 16 : 96);
                int chunk_hi_3_1 = ((c_3_1 < 3) ? chunk_lo_3_1 + 48 : 112);
                if (grow_3_1 < M) {
                    unsigned int x_lo_5[8];
                    unsigned int x_hi_5[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_5[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_3_1 * 256) + (unsigned int)(chunk_lo_3_1 * 2 ^ (srow_3_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_5[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_3_1 * 256) + (unsigned int)(chunk_lo_3_1 * 2 + 16 ^ (srow_3_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_5[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_3_1 * 256) + (unsigned int)(chunk_hi_3_1 * 2 ^ (srow_3_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_5[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_3_1 * 256) + (unsigned int)(chunk_hi_3_1 * 2 + 16 ^ (srow_3_1 & 7) << 4)));
                    if (c_3_1 < 3) {
                        if (w_kind_1 < 2) {
                            float x_lo_f_5[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_lo_f_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_5[_pair]) << 16);
                                (&x_lo_f_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_5[_pair]) & 0xffff0000u);
                            }
                            float x_hi_f_5[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_hi_f_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_5[_pair]) << 16);
                                (&x_hi_f_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_5[_pair]) & 0xffff0000u);
                            }
                            float v_20[8];
                            {
                                const uint4* _vptr_40 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_3_1 * 96 + chunk_lo_3_1) + 0);
                                uint4 _vld_40[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_40[_blk] = _vptr_40[_blk];
                                    uint32_t* _vpairs_40 = reinterpret_cast<uint32_t*>(&_vld_40[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_20[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) << 16);
                                        (&v_20[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_21[8];
                            {
                                const uint4* _vptr_41 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_3_1 * 96 + chunk_lo_3_1 + 8) + 0);
                                uint4 _vld_41[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_41[_blk] = _vptr_41[_blk];
                                    uint32_t* _vpairs_41 = reinterpret_cast<uint32_t*>(&_vld_41[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_21[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) << 16);
                                        (&v_21[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_22[8];
                            {
                                const uint4* _vptr_42 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_3_1 * 96 + 48 + chunk_lo_3_1) + 0);
                                uint4 _vld_42[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_42[_blk] = _vptr_42[_blk];
                                    uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_22[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
                                        (&v_22[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_23[8];
                            {
                                const uint4* _vptr_43 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_3_1 * 96 + 48 + chunk_lo_3_1 + 8) + 0);
                                uint4 _vld_43[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_43[_blk] = _vptr_43[_blk];
                                    uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_23[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
                                        (&v_23[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float o_lo_5[16];
                            float o_hi_5[16];
                            float cv0_35 = v_20[0];
                            float sv0_35 = v_22[0];
                            float cv1_35 = v_21[0];
                            float sv1_35 = v_23[0];
                            o_lo_5[0] = __fmaf_rn(x_lo_f_5[0], cv0_35, -(x_hi_f_5[0] * sv0_35));
                            o_hi_5[0] = __fmaf_rn(x_hi_f_5[0], cv0_35, x_lo_f_5[0] * sv0_35);
                            o_lo_5[8] = __fmaf_rn(x_lo_f_5[8], cv1_35, -(x_hi_f_5[8] * sv1_35));
                            o_hi_5[8] = __fmaf_rn(x_hi_f_5[8], cv1_35, x_lo_f_5[8] * sv1_35);
                            float cv0_35_1 = v_20[1];
                            float sv0_35_1 = v_22[1];
                            float cv1_35_1 = v_21[1];
                            float sv1_35_1 = v_23[1];
                            o_lo_5[1] = __fmaf_rn(x_lo_f_5[1], cv0_35_1, -(x_hi_f_5[1] * sv0_35_1));
                            o_hi_5[1] = __fmaf_rn(x_hi_f_5[1], cv0_35_1, x_lo_f_5[1] * sv0_35_1);
                            o_lo_5[9] = __fmaf_rn(x_lo_f_5[9], cv1_35_1, -(x_hi_f_5[9] * sv1_35_1));
                            o_hi_5[9] = __fmaf_rn(x_hi_f_5[9], cv1_35_1, x_lo_f_5[9] * sv1_35_1);
                            float cv0_36 = v_20[2];
                            float sv0_36 = v_22[2];
                            float cv1_36 = v_21[2];
                            float sv1_36 = v_23[2];
                            o_lo_5[2] = __fmaf_rn(x_lo_f_5[2], cv0_36, -(x_hi_f_5[2] * sv0_36));
                            o_hi_5[2] = __fmaf_rn(x_hi_f_5[2], cv0_36, x_lo_f_5[2] * sv0_36);
                            o_lo_5[10] = __fmaf_rn(x_lo_f_5[10], cv1_36, -(x_hi_f_5[10] * sv1_36));
                            o_hi_5[10] = __fmaf_rn(x_hi_f_5[10], cv1_36, x_lo_f_5[10] * sv1_36);
                            float cv0_37 = v_20[3];
                            float sv0_37 = v_22[3];
                            float cv1_37 = v_21[3];
                            float sv1_37 = v_23[3];
                            o_lo_5[3] = __fmaf_rn(x_lo_f_5[3], cv0_37, -(x_hi_f_5[3] * sv0_37));
                            o_hi_5[3] = __fmaf_rn(x_hi_f_5[3], cv0_37, x_lo_f_5[3] * sv0_37);
                            o_lo_5[11] = __fmaf_rn(x_lo_f_5[11], cv1_37, -(x_hi_f_5[11] * sv1_37));
                            o_hi_5[11] = __fmaf_rn(x_hi_f_5[11], cv1_37, x_lo_f_5[11] * sv1_37);
                            float cv0_38 = v_20[4];
                            float sv0_38 = v_22[4];
                            float cv1_38 = v_21[4];
                            float sv1_38 = v_23[4];
                            o_lo_5[4] = __fmaf_rn(x_lo_f_5[4], cv0_38, -(x_hi_f_5[4] * sv0_38));
                            o_hi_5[4] = __fmaf_rn(x_hi_f_5[4], cv0_38, x_lo_f_5[4] * sv0_38);
                            o_lo_5[12] = __fmaf_rn(x_lo_f_5[12], cv1_38, -(x_hi_f_5[12] * sv1_38));
                            o_hi_5[12] = __fmaf_rn(x_hi_f_5[12], cv1_38, x_lo_f_5[12] * sv1_38);
                            float cv0_39 = v_20[5];
                            float sv0_39 = v_22[5];
                            float cv1_39 = v_21[5];
                            float sv1_39 = v_23[5];
                            o_lo_5[5] = __fmaf_rn(x_lo_f_5[5], cv0_39, -(x_hi_f_5[5] * sv0_39));
                            o_hi_5[5] = __fmaf_rn(x_hi_f_5[5], cv0_39, x_lo_f_5[5] * sv0_39);
                            o_lo_5[13] = __fmaf_rn(x_lo_f_5[13], cv1_39, -(x_hi_f_5[13] * sv1_39));
                            o_hi_5[13] = __fmaf_rn(x_hi_f_5[13], cv1_39, x_lo_f_5[13] * sv1_39);
                            float cv0_40 = v_20[6];
                            float sv0_40 = v_22[6];
                            float cv1_40 = v_21[6];
                            float sv1_40 = v_23[6];
                            o_lo_5[6] = __fmaf_rn(x_lo_f_5[6], cv0_40, -(x_hi_f_5[6] * sv0_40));
                            o_hi_5[6] = __fmaf_rn(x_hi_f_5[6], cv0_40, x_lo_f_5[6] * sv0_40);
                            o_lo_5[14] = __fmaf_rn(x_lo_f_5[14], cv1_40, -(x_hi_f_5[14] * sv1_40));
                            o_hi_5[14] = __fmaf_rn(x_hi_f_5[14], cv1_40, x_lo_f_5[14] * sv1_40);
                            float cv0_41 = v_20[7];
                            float sv0_41 = v_22[7];
                            float cv1_41 = v_21[7];
                            float sv1_41 = v_23[7];
                            o_lo_5[7] = __fmaf_rn(x_lo_f_5[7], cv0_41, -(x_hi_f_5[7] * sv0_41));
                            o_hi_5[7] = __fmaf_rn(x_hi_f_5[7], cv0_41, x_lo_f_5[7] * sv0_41);
                            o_lo_5[15] = __fmaf_rn(x_lo_f_5[15], cv1_41, -(x_hi_f_5[15] * sv1_41));
                            o_hi_5[15] = __fmaf_rn(x_hi_f_5[15], cv1_41, x_lo_f_5[15] * sv1_41);
                            unsigned int o_lo_bf16_5[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_lo_5[_lp*2 + 0], o_lo_5[_lp*2+1 + 0]));
                                o_lo_bf16_5[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int o_hi_bf16_5[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_hi_5[_lp*2 + 0], o_hi_5[_lp*2+1 + 0]));
                                o_hi_bf16_5[_lp] = *(uint32_t*)&_bf2;
                            }
                            x_lo_5[0] = o_lo_bf16_5[0];
                            x_hi_5[0] = o_hi_bf16_5[0];
                            x_lo_5[1] = o_lo_bf16_5[1];
                            x_hi_5[1] = o_hi_bf16_5[1];
                            x_lo_5[2] = o_lo_bf16_5[2];
                            x_hi_5[2] = o_hi_bf16_5[2];
                            x_lo_5[3] = o_lo_bf16_5[3];
                            x_hi_5[3] = o_hi_bf16_5[3];
                            x_lo_5[4] = o_lo_bf16_5[4];
                            x_hi_5[4] = o_hi_bf16_5[4];
                            x_lo_5[5] = o_lo_bf16_5[5];
                            x_hi_5[5] = o_hi_bf16_5[5];
                            x_lo_5[6] = o_lo_bf16_5[6];
                            x_hi_5[6] = o_hi_bf16_5[6];
                            x_lo_5[7] = o_lo_bf16_5[7];
                            x_hi_5[7] = o_hi_bf16_5[7];
                        }
                    }
                    unsigned int q_words_5[4];
                    float q_vals_5[16];
                    float f_10[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_5[_pair]) << 16);
                        (&f_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_5[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_10;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_10) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_10[0]), fabsf(f_10[1])), fabsf(f_10[2])), fabsf(f_10[3])), fabsf(f_10[4])), fabsf(f_10[5])), fabsf(f_10[6])), fabsf(f_10[7])), fabsf(f_10[8])), fabsf(f_10[9])), fabsf(f_10[10])), fabsf(f_10[11])), fabsf(f_10[12])), fabsf(f_10[13])), fabsf(f_10[14])), fabsf(f_10[15])) * sf_mul_sel_1));
                    uint16_t _e4m3x2_decode_44 = (uint16_t)((unsigned int)sf_pair_10 & 0xFFu);
                    uint32_t _f16x2_decode_44;
                    float sf_dec_10;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_44) : "h"(_e4m3x2_decode_44));
                    uint16_t _f16_decode_44 = (uint16_t)_f16x2_decode_44;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_10) : "h"(_f16_decode_44));
                    float out_s_10 = __fdiv_rn(out_scale_sel_1, sf_dec_10);
                    q_vals_5[0] = f_10[0] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[1] = f_10[1] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[2] = f_10[2] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[3] = f_10[3] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[4] = f_10[4] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[5] = f_10[5] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[6] = f_10[6] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[7] = f_10[7] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[8] = f_10[8] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[9] = f_10[9] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[10] = f_10[10] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[11] = f_10[11] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[12] = f_10[12] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[13] = f_10[13] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[14] = f_10[14] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    q_vals_5[15] = f_10[15] * ((sf_dec_10 != 0.0f) ? out_s_10 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_5[0]) : "f"(q_vals_5[0]), "f"(q_vals_5[1]), "f"(q_vals_5[2]), "f"(q_vals_5[3]), "f"(q_vals_5[4]), "f"(q_vals_5[5]), "f"(q_vals_5[6]), "f"(q_vals_5[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_5[1]) : "f"(q_vals_5[8]), "f"(q_vals_5[9]), "f"(q_vals_5[10]), "f"(q_vals_5[11]), "f"(q_vals_5[12]), "f"(q_vals_5[13]), "f"(q_vals_5[14]), "f"(q_vals_5[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_3_1 * 128) + (unsigned int)(chunk_lo_3_1 >> 1 ^ (srow_3_1 & 7) << 4)), "r"(q_words_5[0]), "r"(q_words_5[1]) : "memory");
                    {
                        uint32_t _byte_45 = static_cast<uint32_t>((unsigned int)sf_pair_10 & 127) & 0xFFu;
                        uint32_t _addr_45 = static_cast<uint32_t>(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_3_1 * 128) + (unsigned int)(64 + (chunk_lo_3_1 >> 4) ^ (srow_3_1 & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_45), "r"(_byte_45) : "memory");
                    }
                    float f_11[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_5[_pair]) << 16);
                        (&f_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_5[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_11;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_11) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_11[0]), fabsf(f_11[1])), fabsf(f_11[2])), fabsf(f_11[3])), fabsf(f_11[4])), fabsf(f_11[5])), fabsf(f_11[6])), fabsf(f_11[7])), fabsf(f_11[8])), fabsf(f_11[9])), fabsf(f_11[10])), fabsf(f_11[11])), fabsf(f_11[12])), fabsf(f_11[13])), fabsf(f_11[14])), fabsf(f_11[15])) * sf_mul_sel_1));
                    uint16_t _e4m3x2_decode_46 = (uint16_t)((unsigned int)sf_pair_11 & 0xFFu);
                    uint32_t _f16x2_decode_46;
                    float sf_dec_11;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_46) : "h"(_e4m3x2_decode_46));
                    uint16_t _f16_decode_46 = (uint16_t)_f16x2_decode_46;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_11) : "h"(_f16_decode_46));
                    float out_s_11 = __fdiv_rn(out_scale_sel_1, sf_dec_11);
                    q_vals_5[0] = f_11[0] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[1] = f_11[1] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[2] = f_11[2] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[3] = f_11[3] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[4] = f_11[4] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[5] = f_11[5] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[6] = f_11[6] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[7] = f_11[7] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[8] = f_11[8] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[9] = f_11[9] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[10] = f_11[10] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[11] = f_11[11] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[12] = f_11[12] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[13] = f_11[13] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[14] = f_11[14] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    q_vals_5[15] = f_11[15] * ((sf_dec_11 != 0.0f) ? out_s_11 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_5[0]) : "f"(q_vals_5[0]), "f"(q_vals_5[1]), "f"(q_vals_5[2]), "f"(q_vals_5[3]), "f"(q_vals_5[4]), "f"(q_vals_5[5]), "f"(q_vals_5[6]), "f"(q_vals_5[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_5[1]) : "f"(q_vals_5[8]), "f"(q_vals_5[9]), "f"(q_vals_5[10]), "f"(q_vals_5[11]), "f"(q_vals_5[12]), "f"(q_vals_5[13]), "f"(q_vals_5[14]), "f"(q_vals_5[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_3_1 * 128) + (unsigned int)(chunk_hi_3_1 >> 1 ^ (srow_3_1 & 7) << 4)), "r"(q_words_5[0]), "r"(q_words_5[1]) : "memory");
                    {
                        uint32_t _byte_47 = static_cast<uint32_t>((unsigned int)sf_pair_11 & 127) & 0xFFu;
                        uint32_t _addr_47 = static_cast<uint32_t>(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_3_1 * 128) + (unsigned int)(64 + (chunk_hi_3_1 >> 4) ^ (srow_3_1 & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_47), "r"(_byte_47) : "memory");
                    }
                }
                __syncwarp();
                int s_row_5 = (role_tid_1 >> 5) * 8 + lane / 4;
                int s_piece_5 = lane % 4;
                unsigned int s_words_5[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_5[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_row_5 * 128) + (unsigned int)(s_piece_5 * 16 ^ (s_row_5 & 7) << 4)));
                int s_dst_5 = ((tile_m_1 * 256 + 64 + s_row_5) * 56 + w_head_1) * 16 + s_piece_5 * 4;
                if (tile_m_1 * 256 + 64 + s_row_5 < M) {
                    if (w_kind_1 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_5)[0] = reinterpret_cast<int4*>(s_words_5)[0];
                    } else if (w_kind_1 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_5)[0] = reinterpret_cast<int4*>(s_words_5)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_5)[0] = reinterpret_cast<int4*>(s_words_5)[0];
                    }
                }
                int s_sf_row_3_1 = (role_tid_1 >> 5) * 8 + (lane & 7);
                unsigned int s_sf_words_3_1[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words_3_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words_3_1[(0) + 1]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_sf_row_3_1 * 128) + (unsigned int)(64 ^ (s_sf_row_3_1 & 7) << 4)));
                int s_sf_dst_3_1 = ((tile_m_1 * 256 + 64 + s_sf_row_3_1) * 56 + w_head_1) * 8;
                if (lane < 8) {
                    if (tile_m_1 * 256 + 64 + s_sf_row_3_1 < M) {
                        if (w_kind_1 == 0) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(q_sf + s_sf_dst_3_1))[0] = reinterpret_cast<int2*>(s_sf_words_3_1)[0];
                        } else if (w_kind_1 == 1) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(k_sf + s_sf_dst_3_1))[0] = reinterpret_cast<int2*>(s_sf_words_3_1)[0];
                        } else {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(v_sf + s_sf_dst_3_1))[0] = reinterpret_cast<int2*>(s_sf_words_3_1)[0];
                        }
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int row_lo_4 = tile_m_1 * 256 + 128 + warp_m_1 * 16 + (lane >> 2);
                int row_hi_4 = row_lo_4 + 8;
                float sa_lo_4 = ((row_lo_4 < M) ? act_scale[row_lo_4] : 0.0f);
                float sa_hi_4 = ((row_hi_4 < M) ? act_scale[row_hi_4] : 0.0f);
                float rounded_4[32];
                float ss_lo_4 = 0.0f;
                float ss_hi_4 = 0.0f;
                float scaled_48[4];
                scaled_48[0] = accum_1[64] * sa_lo_4 * sw_1[0];
                scaled_48[1] = accum_1[65] * sa_lo_4 * sw_1[1];
                scaled_48[2] = accum_1[66] * sa_hi_4 * sw_1[0];
                scaled_48[3] = accum_1[67] * sa_hi_4 * sw_1[1];
                unsigned int scaled_bf16_48[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_48[_lp*2 + 0], scaled_48[_lp*2+1 + 0]));
                    scaled_bf16_48[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_48[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_48[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_48[_pair]) << 16);
                    (&unpacked_48[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_48[_pair]) & 0xffff0000u);
                }
                rounded_4[0] = unpacked_48[0];
                rounded_4[1] = unpacked_48[1];
                rounded_4[2] = unpacked_48[2];
                rounded_4[3] = unpacked_48[3];
                ss_lo_4 += __fmaf_rn(unpacked_48[0], unpacked_48[0], unpacked_48[1] * unpacked_48[1]);
                ss_hi_4 += __fmaf_rn(unpacked_48[2], unpacked_48[2], unpacked_48[3] * unpacked_48[3]);
                float scaled_49[4];
                scaled_49[0] = accum_1[68] * sa_lo_4 * sw_1[2];
                scaled_49[1] = accum_1[69] * sa_lo_4 * sw_1[3];
                scaled_49[2] = accum_1[70] * sa_hi_4 * sw_1[2];
                scaled_49[3] = accum_1[71] * sa_hi_4 * sw_1[3];
                unsigned int scaled_bf16_49[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_49[_lp*2 + 0], scaled_49[_lp*2+1 + 0]));
                    scaled_bf16_49[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_49[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_49[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_49[_pair]) << 16);
                    (&unpacked_49[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_49[_pair]) & 0xffff0000u);
                }
                rounded_4[4] = unpacked_49[0];
                rounded_4[5] = unpacked_49[1];
                rounded_4[6] = unpacked_49[2];
                rounded_4[7] = unpacked_49[3];
                ss_lo_4 += __fmaf_rn(unpacked_49[0], unpacked_49[0], unpacked_49[1] * unpacked_49[1]);
                ss_hi_4 += __fmaf_rn(unpacked_49[2], unpacked_49[2], unpacked_49[3] * unpacked_49[3]);
                float scaled_50[4];
                scaled_50[0] = accum_1[72] * sa_lo_4 * sw_1[4];
                scaled_50[1] = accum_1[73] * sa_lo_4 * sw_1[5];
                scaled_50[2] = accum_1[74] * sa_hi_4 * sw_1[4];
                scaled_50[3] = accum_1[75] * sa_hi_4 * sw_1[5];
                unsigned int scaled_bf16_50[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_50[_lp*2 + 0], scaled_50[_lp*2+1 + 0]));
                    scaled_bf16_50[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_50[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_50[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_50[_pair]) << 16);
                    (&unpacked_50[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_50[_pair]) & 0xffff0000u);
                }
                rounded_4[8] = unpacked_50[0];
                rounded_4[9] = unpacked_50[1];
                rounded_4[10] = unpacked_50[2];
                rounded_4[11] = unpacked_50[3];
                ss_lo_4 += __fmaf_rn(unpacked_50[0], unpacked_50[0], unpacked_50[1] * unpacked_50[1]);
                ss_hi_4 += __fmaf_rn(unpacked_50[2], unpacked_50[2], unpacked_50[3] * unpacked_50[3]);
                float scaled_51[4];
                scaled_51[0] = accum_1[76] * sa_lo_4 * sw_1[6];
                scaled_51[1] = accum_1[77] * sa_lo_4 * sw_1[7];
                scaled_51[2] = accum_1[78] * sa_hi_4 * sw_1[6];
                scaled_51[3] = accum_1[79] * sa_hi_4 * sw_1[7];
                unsigned int scaled_bf16_51[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_51[_lp*2 + 0], scaled_51[_lp*2+1 + 0]));
                    scaled_bf16_51[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_51[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_51[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_51[_pair]) << 16);
                    (&unpacked_51[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_51[_pair]) & 0xffff0000u);
                }
                rounded_4[12] = unpacked_51[0];
                rounded_4[13] = unpacked_51[1];
                rounded_4[14] = unpacked_51[2];
                rounded_4[15] = unpacked_51[3];
                ss_lo_4 += __fmaf_rn(unpacked_51[0], unpacked_51[0], unpacked_51[1] * unpacked_51[1]);
                ss_hi_4 += __fmaf_rn(unpacked_51[2], unpacked_51[2], unpacked_51[3] * unpacked_51[3]);
                float scaled_52[4];
                scaled_52[0] = accum_1[80] * sa_lo_4 * sw_1[8];
                scaled_52[1] = accum_1[81] * sa_lo_4 * sw_1[9];
                scaled_52[2] = accum_1[82] * sa_hi_4 * sw_1[8];
                scaled_52[3] = accum_1[83] * sa_hi_4 * sw_1[9];
                unsigned int scaled_bf16_52[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_52[_lp*2 + 0], scaled_52[_lp*2+1 + 0]));
                    scaled_bf16_52[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_52[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_52[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_52[_pair]) << 16);
                    (&unpacked_52[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_52[_pair]) & 0xffff0000u);
                }
                rounded_4[16] = unpacked_52[0];
                rounded_4[17] = unpacked_52[1];
                rounded_4[18] = unpacked_52[2];
                rounded_4[19] = unpacked_52[3];
                ss_lo_4 += __fmaf_rn(unpacked_52[0], unpacked_52[0], unpacked_52[1] * unpacked_52[1]);
                ss_hi_4 += __fmaf_rn(unpacked_52[2], unpacked_52[2], unpacked_52[3] * unpacked_52[3]);
                float scaled_53[4];
                scaled_53[0] = accum_1[84] * sa_lo_4 * sw_1[10];
                scaled_53[1] = accum_1[85] * sa_lo_4 * sw_1[11];
                scaled_53[2] = accum_1[86] * sa_hi_4 * sw_1[10];
                scaled_53[3] = accum_1[87] * sa_hi_4 * sw_1[11];
                unsigned int scaled_bf16_53[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_53[_lp*2 + 0], scaled_53[_lp*2+1 + 0]));
                    scaled_bf16_53[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_53[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_53[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_53[_pair]) << 16);
                    (&unpacked_53[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_53[_pair]) & 0xffff0000u);
                }
                rounded_4[20] = unpacked_53[0];
                rounded_4[21] = unpacked_53[1];
                rounded_4[22] = unpacked_53[2];
                rounded_4[23] = unpacked_53[3];
                ss_lo_4 += __fmaf_rn(unpacked_53[0], unpacked_53[0], unpacked_53[1] * unpacked_53[1]);
                ss_hi_4 += __fmaf_rn(unpacked_53[2], unpacked_53[2], unpacked_53[3] * unpacked_53[3]);
                float scaled_54[4];
                scaled_54[0] = accum_1[88] * sa_lo_4 * sw_1[12];
                scaled_54[1] = accum_1[89] * sa_lo_4 * sw_1[13];
                scaled_54[2] = accum_1[90] * sa_hi_4 * sw_1[12];
                scaled_54[3] = accum_1[91] * sa_hi_4 * sw_1[13];
                unsigned int scaled_bf16_54[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_54[_lp*2 + 0], scaled_54[_lp*2+1 + 0]));
                    scaled_bf16_54[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_54[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_54[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_54[_pair]) << 16);
                    (&unpacked_54[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_54[_pair]) & 0xffff0000u);
                }
                rounded_4[24] = unpacked_54[0];
                rounded_4[25] = unpacked_54[1];
                rounded_4[26] = unpacked_54[2];
                rounded_4[27] = unpacked_54[3];
                ss_lo_4 += __fmaf_rn(unpacked_54[0], unpacked_54[0], unpacked_54[1] * unpacked_54[1]);
                ss_hi_4 += __fmaf_rn(unpacked_54[2], unpacked_54[2], unpacked_54[3] * unpacked_54[3]);
                float scaled_55[4];
                scaled_55[0] = accum_1[92] * sa_lo_4 * sw_1[14];
                scaled_55[1] = accum_1[93] * sa_lo_4 * sw_1[15];
                scaled_55[2] = accum_1[94] * sa_hi_4 * sw_1[14];
                scaled_55[3] = accum_1[95] * sa_hi_4 * sw_1[15];
                unsigned int scaled_bf16_55[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_55[_lp*2 + 0], scaled_55[_lp*2+1 + 0]));
                    scaled_bf16_55[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_55[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_55[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_55[_pair]) << 16);
                    (&unpacked_55[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_55[_pair]) & 0xffff0000u);
                }
                rounded_4[28] = unpacked_55[0];
                rounded_4[29] = unpacked_55[1];
                rounded_4[30] = unpacked_55[2];
                rounded_4[31] = unpacked_55[3];
                ss_lo_4 += __fmaf_rn(unpacked_55[0], unpacked_55[0], unpacked_55[1] * unpacked_55[1]);
                ss_hi_4 += __fmaf_rn(unpacked_55[2], unpacked_55[2], unpacked_55[3] * unpacked_55[3]);
                float rstd_lo_4 = 1.0f;
                float rstd_hi_4 = 1.0f;
                if (w_kind_1 < 2) {
                    float _shfl_xor_sync_48_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo_4, 1);
                    ss_lo_4 += _shfl_xor_sync_48_1;
                    float _shfl_xor_sync_49_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo_4, 2);
                    ss_lo_4 += _shfl_xor_sync_49_1;
                    float _shfl_xor_sync_50_1 = __shfl_xor_sync(0xFFFFFFFF, ss_hi_4, 1);
                    ss_hi_4 += _shfl_xor_sync_50_1;
                    float _shfl_xor_sync_51_1 = __shfl_xor_sync(0xFFFFFFFF, ss_hi_4, 2);
                    ss_hi_4 += _shfl_xor_sync_51_1;
                    int srow_lo_6 = warp_m_1 * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n_1 * 64 + srow_lo_6] = ss_lo_4;
                        rowstat[warp_n_1 * 64 + srow_lo_6 + 8] = ss_hi_4;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float other_lo_6 = rowstat[(1 - warp_n_1) * 64 + srow_lo_6];
                    float other_hi_6 = rowstat[(1 - warp_n_1) * 64 + srow_lo_6 + 8];
                    float _fdiv_53_1 = __fdiv_rn(ss_lo_4 + other_lo_6, 128.0f);
                    float _rsqrt_52_1 = rsqrtf(_fdiv_53_1 + eps);
                    rstd_lo_4 = _rsqrt_52_1;
                    float _fdiv_55_1 = __fdiv_rn(ss_hi_4 + other_hi_6, 128.0f);
                    float _rsqrt_54_1 = rsqrtf(_fdiv_55_1 + eps);
                    rstd_hi_4 = _rsqrt_54_1;
                }
                int srow_st_4 = warp_m_1 * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float normed_48[4];
                if (w_kind_1 < 2) {
                    normed_48[0] = rounded_4[0] * rstd_lo_4 * nw_1[0];
                    normed_48[1] = rounded_4[1] * rstd_lo_4 * nw_1[1];
                    normed_48[2] = rounded_4[2] * rstd_hi_4 * nw_1[0];
                    normed_48[3] = rounded_4[3] * rstd_hi_4 * nw_1[1];
                } else {
                    normed_48[0] = rounded_4[0];
                    normed_48[1] = rounded_4[1];
                    normed_48[2] = rounded_4[2];
                    normed_48[3] = rounded_4[3];
                }
                unsigned int normed_bf16_48[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_48[_lp*2 + 0], normed_48[_lp*2+1 + 0]));
                    normed_bf16_48[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_56 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_4 * 256) + (unsigned int)(warp_n_1 * 64 * 2 ^ (srow_st_4 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_56), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_48[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_48[1]))
                    : "memory");
                float normed_49[4];
                if (w_kind_1 < 2) {
                    normed_49[0] = rounded_4[4] * rstd_lo_4 * nw_1[2];
                    normed_49[1] = rounded_4[5] * rstd_lo_4 * nw_1[3];
                    normed_49[2] = rounded_4[6] * rstd_hi_4 * nw_1[2];
                    normed_49[3] = rounded_4[7] * rstd_hi_4 * nw_1[3];
                } else {
                    normed_49[0] = rounded_4[4];
                    normed_49[1] = rounded_4[5];
                    normed_49[2] = rounded_4[6];
                    normed_49[3] = rounded_4[7];
                }
                unsigned int normed_bf16_49[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_49[_lp*2 + 0], normed_49[_lp*2+1 + 0]));
                    normed_bf16_49[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_57 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_4 * 256) + (unsigned int)((warp_n_1 * 64 + 8) * 2 ^ (srow_st_4 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_57), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_49[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_49[1]))
                    : "memory");
                float normed_50[4];
                if (w_kind_1 < 2) {
                    normed_50[0] = rounded_4[8] * rstd_lo_4 * nw_1[4];
                    normed_50[1] = rounded_4[9] * rstd_lo_4 * nw_1[5];
                    normed_50[2] = rounded_4[10] * rstd_hi_4 * nw_1[4];
                    normed_50[3] = rounded_4[11] * rstd_hi_4 * nw_1[5];
                } else {
                    normed_50[0] = rounded_4[8];
                    normed_50[1] = rounded_4[9];
                    normed_50[2] = rounded_4[10];
                    normed_50[3] = rounded_4[11];
                }
                unsigned int normed_bf16_50[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_50[_lp*2 + 0], normed_50[_lp*2+1 + 0]));
                    normed_bf16_50[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_58 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_4 * 256) + (unsigned int)((warp_n_1 * 64 + 16) * 2 ^ (srow_st_4 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_58), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_50[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_50[1]))
                    : "memory");
                float normed_51[4];
                if (w_kind_1 < 2) {
                    normed_51[0] = rounded_4[12] * rstd_lo_4 * nw_1[6];
                    normed_51[1] = rounded_4[13] * rstd_lo_4 * nw_1[7];
                    normed_51[2] = rounded_4[14] * rstd_hi_4 * nw_1[6];
                    normed_51[3] = rounded_4[15] * rstd_hi_4 * nw_1[7];
                } else {
                    normed_51[0] = rounded_4[12];
                    normed_51[1] = rounded_4[13];
                    normed_51[2] = rounded_4[14];
                    normed_51[3] = rounded_4[15];
                }
                unsigned int normed_bf16_51[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_51[_lp*2 + 0], normed_51[_lp*2+1 + 0]));
                    normed_bf16_51[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_59 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_4 * 256) + (unsigned int)((warp_n_1 * 64 + 24) * 2 ^ (srow_st_4 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_59), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_51[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_51[1]))
                    : "memory");
                float normed_52[4];
                if (w_kind_1 < 2) {
                    normed_52[0] = rounded_4[16] * rstd_lo_4 * nw_1[8];
                    normed_52[1] = rounded_4[17] * rstd_lo_4 * nw_1[9];
                    normed_52[2] = rounded_4[18] * rstd_hi_4 * nw_1[8];
                    normed_52[3] = rounded_4[19] * rstd_hi_4 * nw_1[9];
                } else {
                    normed_52[0] = rounded_4[16];
                    normed_52[1] = rounded_4[17];
                    normed_52[2] = rounded_4[18];
                    normed_52[3] = rounded_4[19];
                }
                unsigned int normed_bf16_52[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_52[_lp*2 + 0], normed_52[_lp*2+1 + 0]));
                    normed_bf16_52[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_60 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_4 * 256) + (unsigned int)((warp_n_1 * 64 + 32) * 2 ^ (srow_st_4 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_60), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_52[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_52[1]))
                    : "memory");
                float normed_53[4];
                if (w_kind_1 < 2) {
                    normed_53[0] = rounded_4[20] * rstd_lo_4 * nw_1[10];
                    normed_53[1] = rounded_4[21] * rstd_lo_4 * nw_1[11];
                    normed_53[2] = rounded_4[22] * rstd_hi_4 * nw_1[10];
                    normed_53[3] = rounded_4[23] * rstd_hi_4 * nw_1[11];
                } else {
                    normed_53[0] = rounded_4[20];
                    normed_53[1] = rounded_4[21];
                    normed_53[2] = rounded_4[22];
                    normed_53[3] = rounded_4[23];
                }
                unsigned int normed_bf16_53[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_53[_lp*2 + 0], normed_53[_lp*2+1 + 0]));
                    normed_bf16_53[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_61 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_4 * 256) + (unsigned int)((warp_n_1 * 64 + 40) * 2 ^ (srow_st_4 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_61), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_53[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_53[1]))
                    : "memory");
                float normed_54[4];
                if (w_kind_1 < 2) {
                    normed_54[0] = rounded_4[24] * rstd_lo_4 * nw_1[12];
                    normed_54[1] = rounded_4[25] * rstd_lo_4 * nw_1[13];
                    normed_54[2] = rounded_4[26] * rstd_hi_4 * nw_1[12];
                    normed_54[3] = rounded_4[27] * rstd_hi_4 * nw_1[13];
                } else {
                    normed_54[0] = rounded_4[24];
                    normed_54[1] = rounded_4[25];
                    normed_54[2] = rounded_4[26];
                    normed_54[3] = rounded_4[27];
                }
                unsigned int normed_bf16_54[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_54[_lp*2 + 0], normed_54[_lp*2+1 + 0]));
                    normed_bf16_54[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_62 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_4 * 256) + (unsigned int)((warp_n_1 * 64 + 48) * 2 ^ (srow_st_4 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_62), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_54[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_54[1]))
                    : "memory");
                float normed_55[4];
                if (w_kind_1 < 2) {
                    normed_55[0] = rounded_4[28] * rstd_lo_4 * nw_1[14];
                    normed_55[1] = rounded_4[29] * rstd_lo_4 * nw_1[15];
                    normed_55[2] = rounded_4[30] * rstd_hi_4 * nw_1[14];
                    normed_55[3] = rounded_4[31] * rstd_hi_4 * nw_1[15];
                } else {
                    normed_55[0] = rounded_4[28];
                    normed_55[1] = rounded_4[29];
                    normed_55[2] = rounded_4[30];
                    normed_55[3] = rounded_4[31];
                }
                unsigned int normed_bf16_55[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_55[_lp*2 + 0], normed_55[_lp*2+1 + 0]));
                    normed_bf16_55[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_63 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_4 * 256) + (unsigned int)((warp_n_1 * 64 + 56) * 2 ^ (srow_st_4 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_63), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_55[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_55[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int srow_4 = role_tid_1 >> 2;
                int c_4 = role_tid_1 & 3;
                int grow_4 = tile_m_1 * 256 + 128 + srow_4;
                int chunk_lo_4 = ((c_4 < 3) ? c_4 * 16 : 96);
                int chunk_hi_4 = ((c_4 < 3) ? chunk_lo_4 + 48 : 112);
                if (grow_4 < M) {
                    unsigned int x_lo_6[8];
                    unsigned int x_hi_6[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_6[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_4 * 256) + (unsigned int)(chunk_lo_4 * 2 ^ (srow_4 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_6[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_4 * 256) + (unsigned int)(chunk_lo_4 * 2 + 16 ^ (srow_4 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_6[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_4 * 256) + (unsigned int)(chunk_hi_4 * 2 ^ (srow_4 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_6[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_4 * 256) + (unsigned int)(chunk_hi_4 * 2 + 16 ^ (srow_4 & 7) << 4)));
                    if (c_4 < 3) {
                        if (w_kind_1 < 2) {
                            float x_lo_f_6[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_lo_f_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_6[_pair]) << 16);
                                (&x_lo_f_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_6[_pair]) & 0xffff0000u);
                            }
                            float x_hi_f_6[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_hi_f_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_6[_pair]) << 16);
                                (&x_hi_f_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_6[_pair]) & 0xffff0000u);
                            }
                            float v_24[8];
                            {
                                const uint4* _vptr_64 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_4 * 96 + chunk_lo_4) + 0);
                                uint4 _vld_64[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_64[_blk] = _vptr_64[_blk];
                                    uint32_t* _vpairs_64 = reinterpret_cast<uint32_t*>(&_vld_64[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_24[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) << 16);
                                        (&v_24[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_25[8];
                            {
                                const uint4* _vptr_65 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_4 * 96 + chunk_lo_4 + 8) + 0);
                                uint4 _vld_65[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_65[_blk] = _vptr_65[_blk];
                                    uint32_t* _vpairs_65 = reinterpret_cast<uint32_t*>(&_vld_65[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_25[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) << 16);
                                        (&v_25[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_26[8];
                            {
                                const uint4* _vptr_66 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_4 * 96 + 48 + chunk_lo_4) + 0);
                                uint4 _vld_66[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_66[_blk] = _vptr_66[_blk];
                                    uint32_t* _vpairs_66 = reinterpret_cast<uint32_t*>(&_vld_66[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_26[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) << 16);
                                        (&v_26[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_27[8];
                            {
                                const uint4* _vptr_67 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_4 * 96 + 48 + chunk_lo_4 + 8) + 0);
                                uint4 _vld_67[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_67[_blk] = _vptr_67[_blk];
                                    uint32_t* _vpairs_67 = reinterpret_cast<uint32_t*>(&_vld_67[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_27[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_67[_pair]) << 16);
                                        (&v_27[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_67[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float o_lo_6[16];
                            float o_hi_6[16];
                            float cv0_42 = v_24[0];
                            float sv0_42 = v_26[0];
                            float cv1_42 = v_25[0];
                            float sv1_42 = v_27[0];
                            o_lo_6[0] = __fmaf_rn(x_lo_f_6[0], cv0_42, -(x_hi_f_6[0] * sv0_42));
                            o_hi_6[0] = __fmaf_rn(x_hi_f_6[0], cv0_42, x_lo_f_6[0] * sv0_42);
                            o_lo_6[8] = __fmaf_rn(x_lo_f_6[8], cv1_42, -(x_hi_f_6[8] * sv1_42));
                            o_hi_6[8] = __fmaf_rn(x_hi_f_6[8], cv1_42, x_lo_f_6[8] * sv1_42);
                            float cv0_42_1 = v_24[1];
                            float sv0_42_1 = v_26[1];
                            float cv1_42_1 = v_25[1];
                            float sv1_42_1 = v_27[1];
                            o_lo_6[1] = __fmaf_rn(x_lo_f_6[1], cv0_42_1, -(x_hi_f_6[1] * sv0_42_1));
                            o_hi_6[1] = __fmaf_rn(x_hi_f_6[1], cv0_42_1, x_lo_f_6[1] * sv0_42_1);
                            o_lo_6[9] = __fmaf_rn(x_lo_f_6[9], cv1_42_1, -(x_hi_f_6[9] * sv1_42_1));
                            o_hi_6[9] = __fmaf_rn(x_hi_f_6[9], cv1_42_1, x_lo_f_6[9] * sv1_42_1);
                            float cv0_43 = v_24[2];
                            float sv0_43 = v_26[2];
                            float cv1_43 = v_25[2];
                            float sv1_43 = v_27[2];
                            o_lo_6[2] = __fmaf_rn(x_lo_f_6[2], cv0_43, -(x_hi_f_6[2] * sv0_43));
                            o_hi_6[2] = __fmaf_rn(x_hi_f_6[2], cv0_43, x_lo_f_6[2] * sv0_43);
                            o_lo_6[10] = __fmaf_rn(x_lo_f_6[10], cv1_43, -(x_hi_f_6[10] * sv1_43));
                            o_hi_6[10] = __fmaf_rn(x_hi_f_6[10], cv1_43, x_lo_f_6[10] * sv1_43);
                            float cv0_44 = v_24[3];
                            float sv0_44 = v_26[3];
                            float cv1_44 = v_25[3];
                            float sv1_44 = v_27[3];
                            o_lo_6[3] = __fmaf_rn(x_lo_f_6[3], cv0_44, -(x_hi_f_6[3] * sv0_44));
                            o_hi_6[3] = __fmaf_rn(x_hi_f_6[3], cv0_44, x_lo_f_6[3] * sv0_44);
                            o_lo_6[11] = __fmaf_rn(x_lo_f_6[11], cv1_44, -(x_hi_f_6[11] * sv1_44));
                            o_hi_6[11] = __fmaf_rn(x_hi_f_6[11], cv1_44, x_lo_f_6[11] * sv1_44);
                            float cv0_45 = v_24[4];
                            float sv0_45 = v_26[4];
                            float cv1_45 = v_25[4];
                            float sv1_45 = v_27[4];
                            o_lo_6[4] = __fmaf_rn(x_lo_f_6[4], cv0_45, -(x_hi_f_6[4] * sv0_45));
                            o_hi_6[4] = __fmaf_rn(x_hi_f_6[4], cv0_45, x_lo_f_6[4] * sv0_45);
                            o_lo_6[12] = __fmaf_rn(x_lo_f_6[12], cv1_45, -(x_hi_f_6[12] * sv1_45));
                            o_hi_6[12] = __fmaf_rn(x_hi_f_6[12], cv1_45, x_lo_f_6[12] * sv1_45);
                            float cv0_46 = v_24[5];
                            float sv0_46 = v_26[5];
                            float cv1_46 = v_25[5];
                            float sv1_46 = v_27[5];
                            o_lo_6[5] = __fmaf_rn(x_lo_f_6[5], cv0_46, -(x_hi_f_6[5] * sv0_46));
                            o_hi_6[5] = __fmaf_rn(x_hi_f_6[5], cv0_46, x_lo_f_6[5] * sv0_46);
                            o_lo_6[13] = __fmaf_rn(x_lo_f_6[13], cv1_46, -(x_hi_f_6[13] * sv1_46));
                            o_hi_6[13] = __fmaf_rn(x_hi_f_6[13], cv1_46, x_lo_f_6[13] * sv1_46);
                            float cv0_47 = v_24[6];
                            float sv0_47 = v_26[6];
                            float cv1_47 = v_25[6];
                            float sv1_47 = v_27[6];
                            o_lo_6[6] = __fmaf_rn(x_lo_f_6[6], cv0_47, -(x_hi_f_6[6] * sv0_47));
                            o_hi_6[6] = __fmaf_rn(x_hi_f_6[6], cv0_47, x_lo_f_6[6] * sv0_47);
                            o_lo_6[14] = __fmaf_rn(x_lo_f_6[14], cv1_47, -(x_hi_f_6[14] * sv1_47));
                            o_hi_6[14] = __fmaf_rn(x_hi_f_6[14], cv1_47, x_lo_f_6[14] * sv1_47);
                            float cv0_48 = v_24[7];
                            float sv0_48 = v_26[7];
                            float cv1_48 = v_25[7];
                            float sv1_48 = v_27[7];
                            o_lo_6[7] = __fmaf_rn(x_lo_f_6[7], cv0_48, -(x_hi_f_6[7] * sv0_48));
                            o_hi_6[7] = __fmaf_rn(x_hi_f_6[7], cv0_48, x_lo_f_6[7] * sv0_48);
                            o_lo_6[15] = __fmaf_rn(x_lo_f_6[15], cv1_48, -(x_hi_f_6[15] * sv1_48));
                            o_hi_6[15] = __fmaf_rn(x_hi_f_6[15], cv1_48, x_lo_f_6[15] * sv1_48);
                            unsigned int o_lo_bf16_6[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_lo_6[_lp*2 + 0], o_lo_6[_lp*2+1 + 0]));
                                o_lo_bf16_6[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int o_hi_bf16_6[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_hi_6[_lp*2 + 0], o_hi_6[_lp*2+1 + 0]));
                                o_hi_bf16_6[_lp] = *(uint32_t*)&_bf2;
                            }
                            x_lo_6[0] = o_lo_bf16_6[0];
                            x_hi_6[0] = o_hi_bf16_6[0];
                            x_lo_6[1] = o_lo_bf16_6[1];
                            x_hi_6[1] = o_hi_bf16_6[1];
                            x_lo_6[2] = o_lo_bf16_6[2];
                            x_hi_6[2] = o_hi_bf16_6[2];
                            x_lo_6[3] = o_lo_bf16_6[3];
                            x_hi_6[3] = o_hi_bf16_6[3];
                            x_lo_6[4] = o_lo_bf16_6[4];
                            x_hi_6[4] = o_hi_bf16_6[4];
                            x_lo_6[5] = o_lo_bf16_6[5];
                            x_hi_6[5] = o_hi_bf16_6[5];
                            x_lo_6[6] = o_lo_bf16_6[6];
                            x_hi_6[6] = o_hi_bf16_6[6];
                            x_lo_6[7] = o_lo_bf16_6[7];
                            x_hi_6[7] = o_hi_bf16_6[7];
                        }
                    }
                    unsigned int q_words_6[4];
                    float q_vals_6[16];
                    float f_12[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_6[_pair]) << 16);
                        (&f_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_6[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_12;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_12) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_12[0]), fabsf(f_12[1])), fabsf(f_12[2])), fabsf(f_12[3])), fabsf(f_12[4])), fabsf(f_12[5])), fabsf(f_12[6])), fabsf(f_12[7])), fabsf(f_12[8])), fabsf(f_12[9])), fabsf(f_12[10])), fabsf(f_12[11])), fabsf(f_12[12])), fabsf(f_12[13])), fabsf(f_12[14])), fabsf(f_12[15])) * sf_mul_sel_1));
                    uint16_t _e4m3x2_decode_68 = (uint16_t)((unsigned int)sf_pair_12 & 0xFFu);
                    uint32_t _f16x2_decode_68;
                    float sf_dec_12;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_68) : "h"(_e4m3x2_decode_68));
                    uint16_t _f16_decode_68 = (uint16_t)_f16x2_decode_68;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_12) : "h"(_f16_decode_68));
                    float out_s_12 = __fdiv_rn(out_scale_sel_1, sf_dec_12);
                    q_vals_6[0] = f_12[0] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[1] = f_12[1] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[2] = f_12[2] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[3] = f_12[3] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[4] = f_12[4] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[5] = f_12[5] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[6] = f_12[6] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[7] = f_12[7] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[8] = f_12[8] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[9] = f_12[9] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[10] = f_12[10] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[11] = f_12[11] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[12] = f_12[12] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[13] = f_12[13] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[14] = f_12[14] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    q_vals_6[15] = f_12[15] * ((sf_dec_12 != 0.0f) ? out_s_12 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_6[0]) : "f"(q_vals_6[0]), "f"(q_vals_6[1]), "f"(q_vals_6[2]), "f"(q_vals_6[3]), "f"(q_vals_6[4]), "f"(q_vals_6[5]), "f"(q_vals_6[6]), "f"(q_vals_6[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_6[1]) : "f"(q_vals_6[8]), "f"(q_vals_6[9]), "f"(q_vals_6[10]), "f"(q_vals_6[11]), "f"(q_vals_6[12]), "f"(q_vals_6[13]), "f"(q_vals_6[14]), "f"(q_vals_6[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_4 * 128) + (unsigned int)(chunk_lo_4 >> 1 ^ (srow_4 & 7) << 4)), "r"(q_words_6[0]), "r"(q_words_6[1]) : "memory");
                    {
                        uint32_t _byte_69 = static_cast<uint32_t>((unsigned int)sf_pair_12 & 127) & 0xFFu;
                        uint32_t _addr_69 = static_cast<uint32_t>(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_4 * 128) + (unsigned int)(64 + (chunk_lo_4 >> 4) ^ (srow_4 & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_69), "r"(_byte_69) : "memory");
                    }
                    float f_13[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_6[_pair]) << 16);
                        (&f_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_6[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_13;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_13) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_13[0]), fabsf(f_13[1])), fabsf(f_13[2])), fabsf(f_13[3])), fabsf(f_13[4])), fabsf(f_13[5])), fabsf(f_13[6])), fabsf(f_13[7])), fabsf(f_13[8])), fabsf(f_13[9])), fabsf(f_13[10])), fabsf(f_13[11])), fabsf(f_13[12])), fabsf(f_13[13])), fabsf(f_13[14])), fabsf(f_13[15])) * sf_mul_sel_1));
                    uint16_t _e4m3x2_decode_70 = (uint16_t)((unsigned int)sf_pair_13 & 0xFFu);
                    uint32_t _f16x2_decode_70;
                    float sf_dec_13;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_70) : "h"(_e4m3x2_decode_70));
                    uint16_t _f16_decode_70 = (uint16_t)_f16x2_decode_70;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_13) : "h"(_f16_decode_70));
                    float out_s_13 = __fdiv_rn(out_scale_sel_1, sf_dec_13);
                    q_vals_6[0] = f_13[0] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[1] = f_13[1] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[2] = f_13[2] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[3] = f_13[3] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[4] = f_13[4] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[5] = f_13[5] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[6] = f_13[6] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[7] = f_13[7] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[8] = f_13[8] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[9] = f_13[9] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[10] = f_13[10] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[11] = f_13[11] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[12] = f_13[12] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[13] = f_13[13] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[14] = f_13[14] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    q_vals_6[15] = f_13[15] * ((sf_dec_13 != 0.0f) ? out_s_13 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_6[0]) : "f"(q_vals_6[0]), "f"(q_vals_6[1]), "f"(q_vals_6[2]), "f"(q_vals_6[3]), "f"(q_vals_6[4]), "f"(q_vals_6[5]), "f"(q_vals_6[6]), "f"(q_vals_6[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_6[1]) : "f"(q_vals_6[8]), "f"(q_vals_6[9]), "f"(q_vals_6[10]), "f"(q_vals_6[11]), "f"(q_vals_6[12]), "f"(q_vals_6[13]), "f"(q_vals_6[14]), "f"(q_vals_6[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_4 * 128) + (unsigned int)(chunk_hi_4 >> 1 ^ (srow_4 & 7) << 4)), "r"(q_words_6[0]), "r"(q_words_6[1]) : "memory");
                    {
                        uint32_t _byte_71 = static_cast<uint32_t>((unsigned int)sf_pair_13 & 127) & 0xFFu;
                        uint32_t _addr_71 = static_cast<uint32_t>(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_4 * 128) + (unsigned int)(64 + (chunk_hi_4 >> 4) ^ (srow_4 & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_71), "r"(_byte_71) : "memory");
                    }
                }
                __syncwarp();
                int s_row_6 = (role_tid_1 >> 5) * 8 + lane / 4;
                int s_piece_6 = lane % 4;
                unsigned int s_words_6[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_6[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_row_6 * 128) + (unsigned int)(s_piece_6 * 16 ^ (s_row_6 & 7) << 4)));
                int s_dst_6 = ((tile_m_1 * 256 + 128 + s_row_6) * 56 + w_head_1) * 16 + s_piece_6 * 4;
                if (tile_m_1 * 256 + 128 + s_row_6 < M) {
                    if (w_kind_1 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_6)[0] = reinterpret_cast<int4*>(s_words_6)[0];
                    } else if (w_kind_1 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_6)[0] = reinterpret_cast<int4*>(s_words_6)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_6)[0] = reinterpret_cast<int4*>(s_words_6)[0];
                    }
                }
                int s_sf_row_4 = (role_tid_1 >> 5) * 8 + (lane & 7);
                unsigned int s_sf_words_4[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words_4[(0) + 1]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_sf_row_4 * 128) + (unsigned int)(64 ^ (s_sf_row_4 & 7) << 4)));
                int s_sf_dst_4 = ((tile_m_1 * 256 + 128 + s_sf_row_4) * 56 + w_head_1) * 8;
                if (lane < 8) {
                    if (tile_m_1 * 256 + 128 + s_sf_row_4 < M) {
                        if (w_kind_1 == 0) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(q_sf + s_sf_dst_4))[0] = reinterpret_cast<int2*>(s_sf_words_4)[0];
                        } else if (w_kind_1 == 1) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(k_sf + s_sf_dst_4))[0] = reinterpret_cast<int2*>(s_sf_words_4)[0];
                        } else {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(v_sf + s_sf_dst_4))[0] = reinterpret_cast<int2*>(s_sf_words_4)[0];
                        }
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int row_lo_5 = tile_m_1 * 256 + 192 + warp_m_1 * 16 + (lane >> 2);
                int row_hi_5 = row_lo_5 + 8;
                float sa_lo_5 = ((row_lo_5 < M) ? act_scale[row_lo_5] : 0.0f);
                float sa_hi_5 = ((row_hi_5 < M) ? act_scale[row_hi_5] : 0.0f);
                float rounded_5[32];
                float ss_lo_5 = 0.0f;
                float ss_hi_5 = 0.0f;
                float scaled_56[4];
                scaled_56[0] = accum_1[96] * sa_lo_5 * sw_1[0];
                scaled_56[1] = accum_1[97] * sa_lo_5 * sw_1[1];
                scaled_56[2] = accum_1[98] * sa_hi_5 * sw_1[0];
                scaled_56[3] = accum_1[99] * sa_hi_5 * sw_1[1];
                unsigned int scaled_bf16_56[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_56[_lp*2 + 0], scaled_56[_lp*2+1 + 0]));
                    scaled_bf16_56[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_56[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_56[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_56[_pair]) << 16);
                    (&unpacked_56[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_56[_pair]) & 0xffff0000u);
                }
                rounded_5[0] = unpacked_56[0];
                rounded_5[1] = unpacked_56[1];
                rounded_5[2] = unpacked_56[2];
                rounded_5[3] = unpacked_56[3];
                ss_lo_5 += __fmaf_rn(unpacked_56[0], unpacked_56[0], unpacked_56[1] * unpacked_56[1]);
                ss_hi_5 += __fmaf_rn(unpacked_56[2], unpacked_56[2], unpacked_56[3] * unpacked_56[3]);
                float scaled_57[4];
                scaled_57[0] = accum_1[100] * sa_lo_5 * sw_1[2];
                scaled_57[1] = accum_1[101] * sa_lo_5 * sw_1[3];
                scaled_57[2] = accum_1[102] * sa_hi_5 * sw_1[2];
                scaled_57[3] = accum_1[103] * sa_hi_5 * sw_1[3];
                unsigned int scaled_bf16_57[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_57[_lp*2 + 0], scaled_57[_lp*2+1 + 0]));
                    scaled_bf16_57[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_57[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_57[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_57[_pair]) << 16);
                    (&unpacked_57[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_57[_pair]) & 0xffff0000u);
                }
                rounded_5[4] = unpacked_57[0];
                rounded_5[5] = unpacked_57[1];
                rounded_5[6] = unpacked_57[2];
                rounded_5[7] = unpacked_57[3];
                ss_lo_5 += __fmaf_rn(unpacked_57[0], unpacked_57[0], unpacked_57[1] * unpacked_57[1]);
                ss_hi_5 += __fmaf_rn(unpacked_57[2], unpacked_57[2], unpacked_57[3] * unpacked_57[3]);
                float scaled_58[4];
                scaled_58[0] = accum_1[104] * sa_lo_5 * sw_1[4];
                scaled_58[1] = accum_1[105] * sa_lo_5 * sw_1[5];
                scaled_58[2] = accum_1[106] * sa_hi_5 * sw_1[4];
                scaled_58[3] = accum_1[107] * sa_hi_5 * sw_1[5];
                unsigned int scaled_bf16_58[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_58[_lp*2 + 0], scaled_58[_lp*2+1 + 0]));
                    scaled_bf16_58[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_58[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_58[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_58[_pair]) << 16);
                    (&unpacked_58[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_58[_pair]) & 0xffff0000u);
                }
                rounded_5[8] = unpacked_58[0];
                rounded_5[9] = unpacked_58[1];
                rounded_5[10] = unpacked_58[2];
                rounded_5[11] = unpacked_58[3];
                ss_lo_5 += __fmaf_rn(unpacked_58[0], unpacked_58[0], unpacked_58[1] * unpacked_58[1]);
                ss_hi_5 += __fmaf_rn(unpacked_58[2], unpacked_58[2], unpacked_58[3] * unpacked_58[3]);
                float scaled_59[4];
                scaled_59[0] = accum_1[108] * sa_lo_5 * sw_1[6];
                scaled_59[1] = accum_1[109] * sa_lo_5 * sw_1[7];
                scaled_59[2] = accum_1[110] * sa_hi_5 * sw_1[6];
                scaled_59[3] = accum_1[111] * sa_hi_5 * sw_1[7];
                unsigned int scaled_bf16_59[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_59[_lp*2 + 0], scaled_59[_lp*2+1 + 0]));
                    scaled_bf16_59[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_59[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_59[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_59[_pair]) << 16);
                    (&unpacked_59[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_59[_pair]) & 0xffff0000u);
                }
                rounded_5[12] = unpacked_59[0];
                rounded_5[13] = unpacked_59[1];
                rounded_5[14] = unpacked_59[2];
                rounded_5[15] = unpacked_59[3];
                ss_lo_5 += __fmaf_rn(unpacked_59[0], unpacked_59[0], unpacked_59[1] * unpacked_59[1]);
                ss_hi_5 += __fmaf_rn(unpacked_59[2], unpacked_59[2], unpacked_59[3] * unpacked_59[3]);
                float scaled_60[4];
                scaled_60[0] = accum_1[112] * sa_lo_5 * sw_1[8];
                scaled_60[1] = accum_1[113] * sa_lo_5 * sw_1[9];
                scaled_60[2] = accum_1[114] * sa_hi_5 * sw_1[8];
                scaled_60[3] = accum_1[115] * sa_hi_5 * sw_1[9];
                unsigned int scaled_bf16_60[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_60[_lp*2 + 0], scaled_60[_lp*2+1 + 0]));
                    scaled_bf16_60[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_60[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_60[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_60[_pair]) << 16);
                    (&unpacked_60[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_60[_pair]) & 0xffff0000u);
                }
                rounded_5[16] = unpacked_60[0];
                rounded_5[17] = unpacked_60[1];
                rounded_5[18] = unpacked_60[2];
                rounded_5[19] = unpacked_60[3];
                ss_lo_5 += __fmaf_rn(unpacked_60[0], unpacked_60[0], unpacked_60[1] * unpacked_60[1]);
                ss_hi_5 += __fmaf_rn(unpacked_60[2], unpacked_60[2], unpacked_60[3] * unpacked_60[3]);
                float scaled_61[4];
                scaled_61[0] = accum_1[116] * sa_lo_5 * sw_1[10];
                scaled_61[1] = accum_1[117] * sa_lo_5 * sw_1[11];
                scaled_61[2] = accum_1[118] * sa_hi_5 * sw_1[10];
                scaled_61[3] = accum_1[119] * sa_hi_5 * sw_1[11];
                unsigned int scaled_bf16_61[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_61[_lp*2 + 0], scaled_61[_lp*2+1 + 0]));
                    scaled_bf16_61[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_61[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_61[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_61[_pair]) << 16);
                    (&unpacked_61[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_61[_pair]) & 0xffff0000u);
                }
                rounded_5[20] = unpacked_61[0];
                rounded_5[21] = unpacked_61[1];
                rounded_5[22] = unpacked_61[2];
                rounded_5[23] = unpacked_61[3];
                ss_lo_5 += __fmaf_rn(unpacked_61[0], unpacked_61[0], unpacked_61[1] * unpacked_61[1]);
                ss_hi_5 += __fmaf_rn(unpacked_61[2], unpacked_61[2], unpacked_61[3] * unpacked_61[3]);
                float scaled_62[4];
                scaled_62[0] = accum_1[120] * sa_lo_5 * sw_1[12];
                scaled_62[1] = accum_1[121] * sa_lo_5 * sw_1[13];
                scaled_62[2] = accum_1[122] * sa_hi_5 * sw_1[12];
                scaled_62[3] = accum_1[123] * sa_hi_5 * sw_1[13];
                unsigned int scaled_bf16_62[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_62[_lp*2 + 0], scaled_62[_lp*2+1 + 0]));
                    scaled_bf16_62[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_62[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_62[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_62[_pair]) << 16);
                    (&unpacked_62[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_62[_pair]) & 0xffff0000u);
                }
                rounded_5[24] = unpacked_62[0];
                rounded_5[25] = unpacked_62[1];
                rounded_5[26] = unpacked_62[2];
                rounded_5[27] = unpacked_62[3];
                ss_lo_5 += __fmaf_rn(unpacked_62[0], unpacked_62[0], unpacked_62[1] * unpacked_62[1]);
                ss_hi_5 += __fmaf_rn(unpacked_62[2], unpacked_62[2], unpacked_62[3] * unpacked_62[3]);
                float scaled_63[4];
                scaled_63[0] = accum_1[124] * sa_lo_5 * sw_1[14];
                scaled_63[1] = accum_1[125] * sa_lo_5 * sw_1[15];
                scaled_63[2] = accum_1[126] * sa_hi_5 * sw_1[14];
                scaled_63[3] = accum_1[127] * sa_hi_5 * sw_1[15];
                unsigned int scaled_bf16_63[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_63[_lp*2 + 0], scaled_63[_lp*2+1 + 0]));
                    scaled_bf16_63[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_63[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&unpacked_63[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_63[_pair]) << 16);
                    (&unpacked_63[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_63[_pair]) & 0xffff0000u);
                }
                rounded_5[28] = unpacked_63[0];
                rounded_5[29] = unpacked_63[1];
                rounded_5[30] = unpacked_63[2];
                rounded_5[31] = unpacked_63[3];
                ss_lo_5 += __fmaf_rn(unpacked_63[0], unpacked_63[0], unpacked_63[1] * unpacked_63[1]);
                ss_hi_5 += __fmaf_rn(unpacked_63[2], unpacked_63[2], unpacked_63[3] * unpacked_63[3]);
                float rstd_lo_5 = 1.0f;
                float rstd_hi_5 = 1.0f;
                if (w_kind_1 < 2) {
                    float _shfl_xor_sync_72_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo_5, 1);
                    ss_lo_5 += _shfl_xor_sync_72_1;
                    float _shfl_xor_sync_73_1 = __shfl_xor_sync(0xFFFFFFFF, ss_lo_5, 2);
                    ss_lo_5 += _shfl_xor_sync_73_1;
                    float _shfl_xor_sync_74_1 = __shfl_xor_sync(0xFFFFFFFF, ss_hi_5, 1);
                    ss_hi_5 += _shfl_xor_sync_74_1;
                    float _shfl_xor_sync_75_1 = __shfl_xor_sync(0xFFFFFFFF, ss_hi_5, 2);
                    ss_hi_5 += _shfl_xor_sync_75_1;
                    int srow_lo_7 = warp_m_1 * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n_1 * 64 + srow_lo_7] = ss_lo_5;
                        rowstat[warp_n_1 * 64 + srow_lo_7 + 8] = ss_hi_5;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float other_lo_7 = rowstat[(1 - warp_n_1) * 64 + srow_lo_7];
                    float other_hi_7 = rowstat[(1 - warp_n_1) * 64 + srow_lo_7 + 8];
                    float _fdiv_77_1 = __fdiv_rn(ss_lo_5 + other_lo_7, 128.0f);
                    float _rsqrt_76_1 = rsqrtf(_fdiv_77_1 + eps);
                    rstd_lo_5 = _rsqrt_76_1;
                    float _fdiv_79_1 = __fdiv_rn(ss_hi_5 + other_hi_7, 128.0f);
                    float _rsqrt_78_1 = rsqrtf(_fdiv_79_1 + eps);
                    rstd_hi_5 = _rsqrt_78_1;
                }
                int srow_st_5 = warp_m_1 * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float normed_56[4];
                if (w_kind_1 < 2) {
                    normed_56[0] = rounded_5[0] * rstd_lo_5 * nw_1[0];
                    normed_56[1] = rounded_5[1] * rstd_lo_5 * nw_1[1];
                    normed_56[2] = rounded_5[2] * rstd_hi_5 * nw_1[0];
                    normed_56[3] = rounded_5[3] * rstd_hi_5 * nw_1[1];
                } else {
                    normed_56[0] = rounded_5[0];
                    normed_56[1] = rounded_5[1];
                    normed_56[2] = rounded_5[2];
                    normed_56[3] = rounded_5[3];
                }
                unsigned int normed_bf16_56[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_56[_lp*2 + 0], normed_56[_lp*2+1 + 0]));
                    normed_bf16_56[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_80 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_5 * 256) + (unsigned int)(warp_n_1 * 64 * 2 ^ (srow_st_5 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_80), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_56[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_56[1]))
                    : "memory");
                float normed_57[4];
                if (w_kind_1 < 2) {
                    normed_57[0] = rounded_5[4] * rstd_lo_5 * nw_1[2];
                    normed_57[1] = rounded_5[5] * rstd_lo_5 * nw_1[3];
                    normed_57[2] = rounded_5[6] * rstd_hi_5 * nw_1[2];
                    normed_57[3] = rounded_5[7] * rstd_hi_5 * nw_1[3];
                } else {
                    normed_57[0] = rounded_5[4];
                    normed_57[1] = rounded_5[5];
                    normed_57[2] = rounded_5[6];
                    normed_57[3] = rounded_5[7];
                }
                unsigned int normed_bf16_57[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_57[_lp*2 + 0], normed_57[_lp*2+1 + 0]));
                    normed_bf16_57[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_81 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_5 * 256) + (unsigned int)((warp_n_1 * 64 + 8) * 2 ^ (srow_st_5 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_81), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_57[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_57[1]))
                    : "memory");
                float normed_58[4];
                if (w_kind_1 < 2) {
                    normed_58[0] = rounded_5[8] * rstd_lo_5 * nw_1[4];
                    normed_58[1] = rounded_5[9] * rstd_lo_5 * nw_1[5];
                    normed_58[2] = rounded_5[10] * rstd_hi_5 * nw_1[4];
                    normed_58[3] = rounded_5[11] * rstd_hi_5 * nw_1[5];
                } else {
                    normed_58[0] = rounded_5[8];
                    normed_58[1] = rounded_5[9];
                    normed_58[2] = rounded_5[10];
                    normed_58[3] = rounded_5[11];
                }
                unsigned int normed_bf16_58[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_58[_lp*2 + 0], normed_58[_lp*2+1 + 0]));
                    normed_bf16_58[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_82 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_5 * 256) + (unsigned int)((warp_n_1 * 64 + 16) * 2 ^ (srow_st_5 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_82), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_58[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_58[1]))
                    : "memory");
                float normed_59[4];
                if (w_kind_1 < 2) {
                    normed_59[0] = rounded_5[12] * rstd_lo_5 * nw_1[6];
                    normed_59[1] = rounded_5[13] * rstd_lo_5 * nw_1[7];
                    normed_59[2] = rounded_5[14] * rstd_hi_5 * nw_1[6];
                    normed_59[3] = rounded_5[15] * rstd_hi_5 * nw_1[7];
                } else {
                    normed_59[0] = rounded_5[12];
                    normed_59[1] = rounded_5[13];
                    normed_59[2] = rounded_5[14];
                    normed_59[3] = rounded_5[15];
                }
                unsigned int normed_bf16_59[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_59[_lp*2 + 0], normed_59[_lp*2+1 + 0]));
                    normed_bf16_59[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_83 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_5 * 256) + (unsigned int)((warp_n_1 * 64 + 24) * 2 ^ (srow_st_5 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_83), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_59[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_59[1]))
                    : "memory");
                float normed_60[4];
                if (w_kind_1 < 2) {
                    normed_60[0] = rounded_5[16] * rstd_lo_5 * nw_1[8];
                    normed_60[1] = rounded_5[17] * rstd_lo_5 * nw_1[9];
                    normed_60[2] = rounded_5[18] * rstd_hi_5 * nw_1[8];
                    normed_60[3] = rounded_5[19] * rstd_hi_5 * nw_1[9];
                } else {
                    normed_60[0] = rounded_5[16];
                    normed_60[1] = rounded_5[17];
                    normed_60[2] = rounded_5[18];
                    normed_60[3] = rounded_5[19];
                }
                unsigned int normed_bf16_60[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_60[_lp*2 + 0], normed_60[_lp*2+1 + 0]));
                    normed_bf16_60[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_84 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_5 * 256) + (unsigned int)((warp_n_1 * 64 + 32) * 2 ^ (srow_st_5 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_84), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_60[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_60[1]))
                    : "memory");
                float normed_61[4];
                if (w_kind_1 < 2) {
                    normed_61[0] = rounded_5[20] * rstd_lo_5 * nw_1[10];
                    normed_61[1] = rounded_5[21] * rstd_lo_5 * nw_1[11];
                    normed_61[2] = rounded_5[22] * rstd_hi_5 * nw_1[10];
                    normed_61[3] = rounded_5[23] * rstd_hi_5 * nw_1[11];
                } else {
                    normed_61[0] = rounded_5[20];
                    normed_61[1] = rounded_5[21];
                    normed_61[2] = rounded_5[22];
                    normed_61[3] = rounded_5[23];
                }
                unsigned int normed_bf16_61[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_61[_lp*2 + 0], normed_61[_lp*2+1 + 0]));
                    normed_bf16_61[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_85 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_5 * 256) + (unsigned int)((warp_n_1 * 64 + 40) * 2 ^ (srow_st_5 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_85), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_61[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_61[1]))
                    : "memory");
                float normed_62[4];
                if (w_kind_1 < 2) {
                    normed_62[0] = rounded_5[24] * rstd_lo_5 * nw_1[12];
                    normed_62[1] = rounded_5[25] * rstd_lo_5 * nw_1[13];
                    normed_62[2] = rounded_5[26] * rstd_hi_5 * nw_1[12];
                    normed_62[3] = rounded_5[27] * rstd_hi_5 * nw_1[13];
                } else {
                    normed_62[0] = rounded_5[24];
                    normed_62[1] = rounded_5[25];
                    normed_62[2] = rounded_5[26];
                    normed_62[3] = rounded_5[27];
                }
                unsigned int normed_bf16_62[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_62[_lp*2 + 0], normed_62[_lp*2+1 + 0]));
                    normed_bf16_62[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_86 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_5 * 256) + (unsigned int)((warp_n_1 * 64 + 48) * 2 ^ (srow_st_5 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_86), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_62[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_62[1]))
                    : "memory");
                float normed_63[4];
                if (w_kind_1 < 2) {
                    normed_63[0] = rounded_5[28] * rstd_lo_5 * nw_1[14];
                    normed_63[1] = rounded_5[29] * rstd_lo_5 * nw_1[15];
                    normed_63[2] = rounded_5[30] * rstd_hi_5 * nw_1[14];
                    normed_63[3] = rounded_5[31] * rstd_hi_5 * nw_1[15];
                } else {
                    normed_63[0] = rounded_5[28];
                    normed_63[1] = rounded_5[29];
                    normed_63[2] = rounded_5[30];
                    normed_63[3] = rounded_5[31];
                }
                unsigned int normed_bf16_63[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normed_63[_lp*2 + 0], normed_63[_lp*2+1 + 0]));
                    normed_bf16_63[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_87 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_st_5 * 256) + (unsigned int)((warp_n_1 * 64 + 56) * 2 ^ (srow_st_5 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_87), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_63[0])), "r"(*reinterpret_cast<const uint32_t*>(&normed_bf16_63[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int srow_5 = role_tid_1 >> 2;
                int c_5 = role_tid_1 & 3;
                int grow_5 = tile_m_1 * 256 + 192 + srow_5;
                int chunk_lo_5 = ((c_5 < 3) ? c_5 * 16 : 96);
                int chunk_hi_5 = ((c_5 < 3) ? chunk_lo_5 + 48 : 112);
                if (grow_5 < M) {
                    unsigned int x_lo_7[8];
                    unsigned int x_hi_7[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_7[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_5 * 256) + (unsigned int)(chunk_lo_5 * 2 ^ (srow_5 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_lo_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_lo_7[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_5 * 256) + (unsigned int)(chunk_lo_5 * 2 + 16 ^ (srow_5 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_7[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_5 * 256) + (unsigned int)(chunk_hi_5 * 2 ^ (srow_5 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&x_hi_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_hi_7[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(srow_5 * 256) + (unsigned int)(chunk_hi_5 * 2 + 16 ^ (srow_5 & 7) << 4)));
                    if (c_5 < 3) {
                        if (w_kind_1 < 2) {
                            float x_lo_f_7[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_lo_f_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_7[_pair]) << 16);
                                (&x_lo_f_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_7[_pair]) & 0xffff0000u);
                            }
                            float x_hi_f_7[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&x_hi_f_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_7[_pair]) << 16);
                                (&x_hi_f_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_7[_pair]) & 0xffff0000u);
                            }
                            float v_28[8];
                            {
                                const uint4* _vptr_88 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_5 * 96 + chunk_lo_5) + 0);
                                uint4 _vld_88[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_88[_blk] = _vptr_88[_blk];
                                    uint32_t* _vpairs_88 = reinterpret_cast<uint32_t*>(&_vld_88[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_28[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_88[_pair]) << 16);
                                        (&v_28[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_88[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_29[8];
                            {
                                const uint4* _vptr_89 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_5 * 96 + chunk_lo_5 + 8) + 0);
                                uint4 _vld_89[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_89[_blk] = _vptr_89[_blk];
                                    uint32_t* _vpairs_89 = reinterpret_cast<uint32_t*>(&_vld_89[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_29[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_89[_pair]) << 16);
                                        (&v_29[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_89[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_30[8];
                            {
                                const uint4* _vptr_90 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_5 * 96 + 48 + chunk_lo_5) + 0);
                                uint4 _vld_90[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_90[_blk] = _vptr_90[_blk];
                                    uint32_t* _vpairs_90 = reinterpret_cast<uint32_t*>(&_vld_90[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_30[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_90[_pair]) << 16);
                                        (&v_30[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_90[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float v_31[8];
                            {
                                const uint4* _vptr_91 = reinterpret_cast<const uint4*>(rope_cos_sin + (grow_5 * 96 + 48 + chunk_lo_5 + 8) + 0);
                                uint4 _vld_91[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_91[_blk] = _vptr_91[_blk];
                                    uint32_t* _vpairs_91 = reinterpret_cast<uint32_t*>(&_vld_91[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&v_31[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_91[_pair]) << 16);
                                        (&v_31[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_91[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float o_lo_7[16];
                            float o_hi_7[16];
                            float cv0_49 = v_28[0];
                            float sv0_49 = v_30[0];
                            float cv1_49 = v_29[0];
                            float sv1_49 = v_31[0];
                            o_lo_7[0] = __fmaf_rn(x_lo_f_7[0], cv0_49, -(x_hi_f_7[0] * sv0_49));
                            o_hi_7[0] = __fmaf_rn(x_hi_f_7[0], cv0_49, x_lo_f_7[0] * sv0_49);
                            o_lo_7[8] = __fmaf_rn(x_lo_f_7[8], cv1_49, -(x_hi_f_7[8] * sv1_49));
                            o_hi_7[8] = __fmaf_rn(x_hi_f_7[8], cv1_49, x_lo_f_7[8] * sv1_49);
                            float cv0_49_1 = v_28[1];
                            float sv0_49_1 = v_30[1];
                            float cv1_49_1 = v_29[1];
                            float sv1_49_1 = v_31[1];
                            o_lo_7[1] = __fmaf_rn(x_lo_f_7[1], cv0_49_1, -(x_hi_f_7[1] * sv0_49_1));
                            o_hi_7[1] = __fmaf_rn(x_hi_f_7[1], cv0_49_1, x_lo_f_7[1] * sv0_49_1);
                            o_lo_7[9] = __fmaf_rn(x_lo_f_7[9], cv1_49_1, -(x_hi_f_7[9] * sv1_49_1));
                            o_hi_7[9] = __fmaf_rn(x_hi_f_7[9], cv1_49_1, x_lo_f_7[9] * sv1_49_1);
                            float cv0_50 = v_28[2];
                            float sv0_50 = v_30[2];
                            float cv1_50 = v_29[2];
                            float sv1_50 = v_31[2];
                            o_lo_7[2] = __fmaf_rn(x_lo_f_7[2], cv0_50, -(x_hi_f_7[2] * sv0_50));
                            o_hi_7[2] = __fmaf_rn(x_hi_f_7[2], cv0_50, x_lo_f_7[2] * sv0_50);
                            o_lo_7[10] = __fmaf_rn(x_lo_f_7[10], cv1_50, -(x_hi_f_7[10] * sv1_50));
                            o_hi_7[10] = __fmaf_rn(x_hi_f_7[10], cv1_50, x_lo_f_7[10] * sv1_50);
                            float cv0_51 = v_28[3];
                            float sv0_51 = v_30[3];
                            float cv1_51 = v_29[3];
                            float sv1_51 = v_31[3];
                            o_lo_7[3] = __fmaf_rn(x_lo_f_7[3], cv0_51, -(x_hi_f_7[3] * sv0_51));
                            o_hi_7[3] = __fmaf_rn(x_hi_f_7[3], cv0_51, x_lo_f_7[3] * sv0_51);
                            o_lo_7[11] = __fmaf_rn(x_lo_f_7[11], cv1_51, -(x_hi_f_7[11] * sv1_51));
                            o_hi_7[11] = __fmaf_rn(x_hi_f_7[11], cv1_51, x_lo_f_7[11] * sv1_51);
                            float cv0_52 = v_28[4];
                            float sv0_52 = v_30[4];
                            float cv1_52 = v_29[4];
                            float sv1_52 = v_31[4];
                            o_lo_7[4] = __fmaf_rn(x_lo_f_7[4], cv0_52, -(x_hi_f_7[4] * sv0_52));
                            o_hi_7[4] = __fmaf_rn(x_hi_f_7[4], cv0_52, x_lo_f_7[4] * sv0_52);
                            o_lo_7[12] = __fmaf_rn(x_lo_f_7[12], cv1_52, -(x_hi_f_7[12] * sv1_52));
                            o_hi_7[12] = __fmaf_rn(x_hi_f_7[12], cv1_52, x_lo_f_7[12] * sv1_52);
                            float cv0_53 = v_28[5];
                            float sv0_53 = v_30[5];
                            float cv1_53 = v_29[5];
                            float sv1_53 = v_31[5];
                            o_lo_7[5] = __fmaf_rn(x_lo_f_7[5], cv0_53, -(x_hi_f_7[5] * sv0_53));
                            o_hi_7[5] = __fmaf_rn(x_hi_f_7[5], cv0_53, x_lo_f_7[5] * sv0_53);
                            o_lo_7[13] = __fmaf_rn(x_lo_f_7[13], cv1_53, -(x_hi_f_7[13] * sv1_53));
                            o_hi_7[13] = __fmaf_rn(x_hi_f_7[13], cv1_53, x_lo_f_7[13] * sv1_53);
                            float cv0_54 = v_28[6];
                            float sv0_54 = v_30[6];
                            float cv1_54 = v_29[6];
                            float sv1_54 = v_31[6];
                            o_lo_7[6] = __fmaf_rn(x_lo_f_7[6], cv0_54, -(x_hi_f_7[6] * sv0_54));
                            o_hi_7[6] = __fmaf_rn(x_hi_f_7[6], cv0_54, x_lo_f_7[6] * sv0_54);
                            o_lo_7[14] = __fmaf_rn(x_lo_f_7[14], cv1_54, -(x_hi_f_7[14] * sv1_54));
                            o_hi_7[14] = __fmaf_rn(x_hi_f_7[14], cv1_54, x_lo_f_7[14] * sv1_54);
                            float cv0_55 = v_28[7];
                            float sv0_55 = v_30[7];
                            float cv1_55 = v_29[7];
                            float sv1_55 = v_31[7];
                            o_lo_7[7] = __fmaf_rn(x_lo_f_7[7], cv0_55, -(x_hi_f_7[7] * sv0_55));
                            o_hi_7[7] = __fmaf_rn(x_hi_f_7[7], cv0_55, x_lo_f_7[7] * sv0_55);
                            o_lo_7[15] = __fmaf_rn(x_lo_f_7[15], cv1_55, -(x_hi_f_7[15] * sv1_55));
                            o_hi_7[15] = __fmaf_rn(x_hi_f_7[15], cv1_55, x_lo_f_7[15] * sv1_55);
                            unsigned int o_lo_bf16_7[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_lo_7[_lp*2 + 0], o_lo_7[_lp*2+1 + 0]));
                                o_lo_bf16_7[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int o_hi_bf16_7[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_hi_7[_lp*2 + 0], o_hi_7[_lp*2+1 + 0]));
                                o_hi_bf16_7[_lp] = *(uint32_t*)&_bf2;
                            }
                            x_lo_7[0] = o_lo_bf16_7[0];
                            x_hi_7[0] = o_hi_bf16_7[0];
                            x_lo_7[1] = o_lo_bf16_7[1];
                            x_hi_7[1] = o_hi_bf16_7[1];
                            x_lo_7[2] = o_lo_bf16_7[2];
                            x_hi_7[2] = o_hi_bf16_7[2];
                            x_lo_7[3] = o_lo_bf16_7[3];
                            x_hi_7[3] = o_hi_bf16_7[3];
                            x_lo_7[4] = o_lo_bf16_7[4];
                            x_hi_7[4] = o_hi_bf16_7[4];
                            x_lo_7[5] = o_lo_bf16_7[5];
                            x_hi_7[5] = o_hi_bf16_7[5];
                            x_lo_7[6] = o_lo_bf16_7[6];
                            x_hi_7[6] = o_hi_bf16_7[6];
                            x_lo_7[7] = o_lo_bf16_7[7];
                            x_hi_7[7] = o_hi_bf16_7[7];
                        }
                    }
                    unsigned int q_words_7[4];
                    float q_vals_7[16];
                    float f_14[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_lo_7[_pair]) << 16);
                        (&f_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_lo_7[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_14;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_14) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_14[0]), fabsf(f_14[1])), fabsf(f_14[2])), fabsf(f_14[3])), fabsf(f_14[4])), fabsf(f_14[5])), fabsf(f_14[6])), fabsf(f_14[7])), fabsf(f_14[8])), fabsf(f_14[9])), fabsf(f_14[10])), fabsf(f_14[11])), fabsf(f_14[12])), fabsf(f_14[13])), fabsf(f_14[14])), fabsf(f_14[15])) * sf_mul_sel_1));
                    uint16_t _e4m3x2_decode_92 = (uint16_t)((unsigned int)sf_pair_14 & 0xFFu);
                    uint32_t _f16x2_decode_92;
                    float sf_dec_14;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_92) : "h"(_e4m3x2_decode_92));
                    uint16_t _f16_decode_92 = (uint16_t)_f16x2_decode_92;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_14) : "h"(_f16_decode_92));
                    float out_s_14 = __fdiv_rn(out_scale_sel_1, sf_dec_14);
                    q_vals_7[0] = f_14[0] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[1] = f_14[1] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[2] = f_14[2] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[3] = f_14[3] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[4] = f_14[4] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[5] = f_14[5] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[6] = f_14[6] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[7] = f_14[7] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[8] = f_14[8] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[9] = f_14[9] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[10] = f_14[10] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[11] = f_14[11] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[12] = f_14[12] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[13] = f_14[13] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[14] = f_14[14] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    q_vals_7[15] = f_14[15] * ((sf_dec_14 != 0.0f) ? out_s_14 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_7[0]) : "f"(q_vals_7[0]), "f"(q_vals_7[1]), "f"(q_vals_7[2]), "f"(q_vals_7[3]), "f"(q_vals_7[4]), "f"(q_vals_7[5]), "f"(q_vals_7[6]), "f"(q_vals_7[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_7[1]) : "f"(q_vals_7[8]), "f"(q_vals_7[9]), "f"(q_vals_7[10]), "f"(q_vals_7[11]), "f"(q_vals_7[12]), "f"(q_vals_7[13]), "f"(q_vals_7[14]), "f"(q_vals_7[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_5 * 128) + (unsigned int)(chunk_lo_5 >> 1 ^ (srow_5 & 7) << 4)), "r"(q_words_7[0]), "r"(q_words_7[1]) : "memory");
                    {
                        uint32_t _byte_93 = static_cast<uint32_t>((unsigned int)sf_pair_14 & 127) & 0xFFu;
                        uint32_t _addr_93 = static_cast<uint32_t>(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_5 * 128) + (unsigned int)(64 + (chunk_lo_5 >> 4) ^ (srow_5 & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_93), "r"(_byte_93) : "memory");
                    }
                    float f_15[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&f_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_hi_7[_pair]) << 16);
                        (&f_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_hi_7[_pair]) & 0xffff0000u);
                    }
                    uint16_t sf_pair_15;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_15) : "f"(0.0f), "f"(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(f_15[0]), fabsf(f_15[1])), fabsf(f_15[2])), fabsf(f_15[3])), fabsf(f_15[4])), fabsf(f_15[5])), fabsf(f_15[6])), fabsf(f_15[7])), fabsf(f_15[8])), fabsf(f_15[9])), fabsf(f_15[10])), fabsf(f_15[11])), fabsf(f_15[12])), fabsf(f_15[13])), fabsf(f_15[14])), fabsf(f_15[15])) * sf_mul_sel_1));
                    uint16_t _e4m3x2_decode_94 = (uint16_t)((unsigned int)sf_pair_15 & 0xFFu);
                    uint32_t _f16x2_decode_94;
                    float sf_dec_15;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_94) : "h"(_e4m3x2_decode_94));
                    uint16_t _f16_decode_94 = (uint16_t)_f16x2_decode_94;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec_15) : "h"(_f16_decode_94));
                    float out_s_15 = __fdiv_rn(out_scale_sel_1, sf_dec_15);
                    q_vals_7[0] = f_15[0] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[1] = f_15[1] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[2] = f_15[2] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[3] = f_15[3] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[4] = f_15[4] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[5] = f_15[5] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[6] = f_15[6] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[7] = f_15[7] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[8] = f_15[8] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[9] = f_15[9] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[10] = f_15[10] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[11] = f_15[11] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[12] = f_15[12] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[13] = f_15[13] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[14] = f_15[14] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    q_vals_7[15] = f_15[15] * ((sf_dec_15 != 0.0f) ? out_s_15 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_7[0]) : "f"(q_vals_7[0]), "f"(q_vals_7[1]), "f"(q_vals_7[2]), "f"(q_vals_7[3]), "f"(q_vals_7[4]), "f"(q_vals_7[5]), "f"(q_vals_7[6]), "f"(q_vals_7[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_words_7[1]) : "f"(q_vals_7[8]), "f"(q_vals_7[9]), "f"(q_vals_7[10]), "f"(q_vals_7[11]), "f"(q_vals_7[12]), "f"(q_vals_7[13]), "f"(q_vals_7[14]), "f"(q_vals_7[15]));
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_5 * 128) + (unsigned int)(chunk_hi_5 >> 1 ^ (srow_5 & 7) << 4)), "r"(q_words_7[0]), "r"(q_words_7[1]) : "memory");
                    {
                        uint32_t _byte_95 = static_cast<uint32_t>((unsigned int)sf_pair_15 & 127) & 0xFFu;
                        uint32_t _addr_95 = static_cast<uint32_t>(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(srow_5 * 128) + (unsigned int)(64 + (chunk_hi_5 >> 4) ^ (srow_5 & 7) << 4));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_95), "r"(_byte_95) : "memory");
                    }
                }
                __syncwarp();
                int s_row_7 = (role_tid_1 >> 5) * 8 + lane / 4;
                int s_piece_7 = lane % 4;
                unsigned int s_words_7[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_7[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_row_7 * 128) + (unsigned int)(s_piece_7 * 16 ^ (s_row_7 & 7) << 4)));
                int s_dst_7 = ((tile_m_1 * 256 + 192 + s_row_7) * 56 + w_head_1) * 16 + s_piece_7 * 4;
                if (tile_m_1 * 256 + 192 + s_row_7 < M) {
                    if (w_kind_1 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_7)[0] = reinterpret_cast<int4*>(s_words_7)[0];
                    } else if (w_kind_1 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_7)[0] = reinterpret_cast<int4*>(s_words_7)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_7)[0] = reinterpret_cast<int4*>(s_words_7)[0];
                    }
                }
                int s_sf_row_5 = (role_tid_1 >> 5) * 8 + (lane & 7);
                unsigned int s_sf_words_5[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_sf_words_5[(0) + 1]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_sf_row_5 * 128) + (unsigned int)(64 ^ (s_sf_row_5 & 7) << 4)));
                int s_sf_dst_5 = ((tile_m_1 * 256 + 192 + s_sf_row_5) * 56 + w_head_1) * 8;
                if (lane < 8) {
                    if (tile_m_1 * 256 + 192 + s_sf_row_5 < M) {
                        if (w_kind_1 == 0) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(q_sf + s_sf_dst_5))[0] = reinterpret_cast<int2*>(s_sf_words_5)[0];
                        } else if (w_kind_1 == 1) {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(k_sf + s_sf_dst_5))[0] = reinterpret_cast<int2*>(s_sf_words_5)[0];
                        } else {
                            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(v_sf + s_sf_dst_5))[0] = reinterpret_cast<int2*>(s_sf_words_5)[0];
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
            }
        }
    }

    // Cleanup
}

} // extern "C"
