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
#define NUM_FULL_PIPE_STAGES 4
#define NUM_EMPTY_PIPE_STAGES 4
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 16384
#define SMEM_A_STAGE_STRIDE 16384
#define SMEM_B_STAGE_OFF 66560
#define SMEM_B_STAGE_STAGE_BYTES 8192
#define SMEM_B_STAGE_STRIDE 8192
#define SMEM_ROWSTAT_OFF 99328
#define SMEM_ROWSTAT_STAGE_BYTES 512
#define SMEM_ROWSTAT_STRIDE 512
#define SMEM_TOTAL 99840
#define THREADS 256
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
kernel_cake_minimax_h3_sm120_quant_pre_attention_7903f989d4c353f03045(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ act_scale, float* __restrict__ w_scale, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, unsigned int* __restrict__ q_out, unsigned int* __restrict__ k_out, unsigned int* __restrict__ v_out, uint8_t* __restrict__ q_sf, uint8_t* __restrict__ k_sf, uint8_t* __restrict__ v_sf, int M, int num_m_tiles, int total_tiles, float eps, float alpha, float out_scale_q, float out_scale_k, float out_scale_v, float sf_mul_q, float sf_mul_k, float sf_mul_v)
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

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int B_stage_addr = smem + 66560;
    float* rowstat = reinterpret_cast<float*>(smem_raw + 99328);
    const int rowstat_addr = smem + 99328;

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
            int role_tid = lane;
            float accum[128];
            unsigned int a_frag[16];
            unsigned int b_frag[16];
            float sw[16];
            float nw[16];
            unsigned int _phase_ab_empty = 1;
            if (lane == 0) {
                for (int _prologue = 0; _prologue < 3; _prologue++) {
                    if (bid + (int)(load_g / 84) * num_bids < total_tiles) {
                        mbarrier_wait_hint(ab_empty_addr + (load_stage) * 8, _phase_ab_empty, 10000000);
                        mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 24576);
                        tma_2d_gmem2smem(A_stage_addr + load_stage * 16384, (&A), ((int)load_g - (int)(load_g / 84) * 84) * 64, ((bid + (int)(load_g / 84) * num_bids) / 2688 * 16 + (bid + (int)(load_g / 84) * num_bids - (bid + (int)(load_g / 84) * num_bids) / 2688 * 2688) % ((num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / 2688 * 16 < 16) ? num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / 2688 * 16 : 16)) * 256, ab_full_addr + (load_stage) * 8);
                        tma_2d_gmem2smem(B_stage_addr + load_stage * 8192, (&B), ((int)load_g - (int)(load_g / 84) * 84) * 64, (bid + (int)(load_g / 84) * num_bids - (bid + (int)(load_g / 84) * num_bids) / 2688 * 2688) / ((num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / 2688 * 16 < 16) ? num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / 2688 * 16 : 16) * 128, ab_full_addr + (load_stage) * 8);
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
                int tile_m = tile / 2688 * 16 + (tile - tile / 2688 * 2688) % ((num_m_tiles - tile / 2688 * 16 < 16) ? num_m_tiles - tile / 2688 * 16 : 16);
                int tile_n = (tile - tile / 2688 * 2688) / ((num_m_tiles - tile / 2688 * 16 < 16) ? num_m_tiles - tile / 2688 * 16 : 16);
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
                int c_kind = tile_n % 3;
                int c_col_base = tile_n * 128 + warp_n * 64 + (lane & 3) * 2;
                sw[0] = w_scale[c_col_base];
                sw[1] = w_scale[c_col_base + 1];
                sw[2] = w_scale[c_col_base + 8];
                sw[3] = w_scale[c_col_base + 8 + 1];
                sw[4] = w_scale[c_col_base + 16];
                sw[5] = w_scale[c_col_base + 16 + 1];
                sw[6] = w_scale[c_col_base + 24];
                sw[7] = w_scale[c_col_base + 24 + 1];
                sw[8] = w_scale[c_col_base + 32];
                sw[9] = w_scale[c_col_base + 32 + 1];
                sw[10] = w_scale[c_col_base + 40];
                sw[11] = w_scale[c_col_base + 40 + 1];
                sw[12] = w_scale[c_col_base + 48];
                sw[13] = w_scale[c_col_base + 48 + 1];
                sw[14] = w_scale[c_col_base + 56];
                sw[15] = w_scale[c_col_base + 56 + 1];
                int c_d_col_0 = warp_n * 64 + (lane & 3) * 2;
                if (c_kind == 0) {
                    nw[0] = (float)q_norm_weight[c_d_col_0];
                    nw[1] = (float)q_norm_weight[c_d_col_0 + 1];
                } else if (c_kind == 1) {
                    nw[0] = (float)k_norm_weight[c_d_col_0];
                    nw[1] = (float)k_norm_weight[c_d_col_0 + 1];
                } else {
                    nw[0] = 1.0f;
                    nw[1] = 1.0f;
                }
                int c_d_col_1 = warp_n * 64 + 8 + (lane & 3) * 2;
                if (c_kind == 0) {
                    nw[2] = (float)q_norm_weight[c_d_col_1];
                    nw[3] = (float)q_norm_weight[c_d_col_1 + 1];
                } else if (c_kind == 1) {
                    nw[2] = (float)k_norm_weight[c_d_col_1];
                    nw[3] = (float)k_norm_weight[c_d_col_1 + 1];
                } else {
                    nw[2] = 1.0f;
                    nw[3] = 1.0f;
                }
                int c_d_col_2 = warp_n * 64 + 16 + (lane & 3) * 2;
                if (c_kind == 0) {
                    nw[4] = (float)q_norm_weight[c_d_col_2];
                    nw[5] = (float)q_norm_weight[c_d_col_2 + 1];
                } else if (c_kind == 1) {
                    nw[4] = (float)k_norm_weight[c_d_col_2];
                    nw[5] = (float)k_norm_weight[c_d_col_2 + 1];
                } else {
                    nw[4] = 1.0f;
                    nw[5] = 1.0f;
                }
                int c_d_col_3 = warp_n * 64 + 24 + (lane & 3) * 2;
                if (c_kind == 0) {
                    nw[6] = (float)q_norm_weight[c_d_col_3];
                    nw[7] = (float)q_norm_weight[c_d_col_3 + 1];
                } else if (c_kind == 1) {
                    nw[6] = (float)k_norm_weight[c_d_col_3];
                    nw[7] = (float)k_norm_weight[c_d_col_3 + 1];
                } else {
                    nw[6] = 1.0f;
                    nw[7] = 1.0f;
                }
                int c_d_col_4 = warp_n * 64 + 32 + (lane & 3) * 2;
                if (c_kind == 0) {
                    nw[8] = (float)q_norm_weight[c_d_col_4];
                    nw[9] = (float)q_norm_weight[c_d_col_4 + 1];
                } else if (c_kind == 1) {
                    nw[8] = (float)k_norm_weight[c_d_col_4];
                    nw[9] = (float)k_norm_weight[c_d_col_4 + 1];
                } else {
                    nw[8] = 1.0f;
                    nw[9] = 1.0f;
                }
                int c_d_col_5 = warp_n * 64 + 40 + (lane & 3) * 2;
                if (c_kind == 0) {
                    nw[10] = (float)q_norm_weight[c_d_col_5];
                    nw[11] = (float)q_norm_weight[c_d_col_5 + 1];
                } else if (c_kind == 1) {
                    nw[10] = (float)k_norm_weight[c_d_col_5];
                    nw[11] = (float)k_norm_weight[c_d_col_5 + 1];
                } else {
                    nw[10] = 1.0f;
                    nw[11] = 1.0f;
                }
                int c_d_col_6 = warp_n * 64 + 48 + (lane & 3) * 2;
                if (c_kind == 0) {
                    nw[12] = (float)q_norm_weight[c_d_col_6];
                    nw[13] = (float)q_norm_weight[c_d_col_6 + 1];
                } else if (c_kind == 1) {
                    nw[12] = (float)k_norm_weight[c_d_col_6];
                    nw[13] = (float)k_norm_weight[c_d_col_6 + 1];
                } else {
                    nw[12] = 1.0f;
                    nw[13] = 1.0f;
                }
                int c_d_col_7 = warp_n * 64 + 56 + (lane & 3) * 2;
                if (c_kind == 0) {
                    nw[14] = (float)q_norm_weight[c_d_col_7];
                    nw[15] = (float)q_norm_weight[c_d_col_7 + 1];
                } else if (c_kind == 1) {
                    nw[14] = (float)k_norm_weight[c_d_col_7];
                    nw[15] = (float)k_norm_weight[c_d_col_7 + 1];
                } else {
                    nw[14] = 1.0f;
                    nw[15] = 1.0f;
                }
                if (lane == 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                #pragma unroll 1
                for (int k_tile = 0; k_tile < 84; k_tile++) {
                    if (lane == 0) {
                        if (bid + (int)(load_g / 84) * num_bids < total_tiles) {
                            mbarrier_wait_hint(ab_empty_addr + (load_stage) * 8, _phase_ab_empty, 10000000);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 24576);
                            tma_2d_gmem2smem(A_stage_addr + load_stage * 16384, (&A), ((int)load_g - (int)(load_g / 84) * 84) * 64, ((bid + (int)(load_g / 84) * num_bids) / 2688 * 16 + (bid + (int)(load_g / 84) * num_bids - (bid + (int)(load_g / 84) * num_bids) / 2688 * 2688) % ((num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / 2688 * 16 < 16) ? num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / 2688 * 16 : 16)) * 256, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(B_stage_addr + load_stage * 8192, (&B), ((int)load_g - (int)(load_g / 84) * 84) * 64, (bid + (int)(load_g / 84) * num_bids - (bid + (int)(load_g / 84) * num_bids) / 2688 * 2688) / ((num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / 2688 * 16 < 16) ? num_m_tiles - (bid + (int)(load_g / 84) * num_bids) / 2688 * 16 : 16) * 128, ab_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_ab_empty ^= 1; }
                        load_g = load_g + 1;
                    }
                    __syncwarp();
                    mbarrier_wait_hint(ab_full_addr + (mma_stage) * 8, _phase_ab_full, 10000000);
                    for (int k_step = 0; k_step < 2; k_step++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + 128 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + 128 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                            : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 16 + 192 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 16 + 192 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                            : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                            : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                            : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(k_step * 32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
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
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane == 0) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 4) { mma_stage = 0; _phase_ab_full ^= 1; }
                    mma_kk = mma_kk + 1;
                }
                unsigned int free_stage = ((mma_stage == 0) ? 3 : mma_stage - 1);
                int w_kind = tile_n % 3;
                int w_head = tile_n / 3;
                float w_out_scale_sel = ((w_kind == 0) ? out_scale_q : ((w_kind == 1) ? out_scale_k : out_scale_v));
                float w_sf_mul_sel = ((w_kind == 0) ? sf_mul_q : ((w_kind == 1) ? sf_mul_k : sf_mul_v));
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_row_lo_0 = tile_m * 256 + warp_m * 16 + (lane >> 2);
                int w_row_hi_0 = w_row_lo_0 + 8;
                float w_sa_lo_0 = ((w_row_lo_0 < M) ? act_scale[w_row_lo_0] : 0.0f);
                float w_sa_hi_0 = ((w_row_hi_0 < M) ? act_scale[w_row_hi_0] : 0.0f);
                float w_rounded_0[32];
                float w_ss_lo_0 = 0.0f;
                float w_ss_hi_0 = 0.0f;
                float w_scaled_0_0[4];
                w_scaled_0_0[0] = accum[0] * w_sa_lo_0 * sw[0];
                w_scaled_0_0[1] = accum[1] * w_sa_lo_0 * sw[1];
                w_scaled_0_0[2] = accum[2] * w_sa_hi_0 * sw[0];
                w_scaled_0_0[3] = accum[3] * w_sa_hi_0 * sw[1];
                unsigned int w_scaled_0_0_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_0[_lp*2 + 0], w_scaled_0_0[_lp*2+1 + 0]));
                    w_scaled_0_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_0[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_0_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_0[0] = _unpack_bf16_0[0];
                w_rounded_0[1] = _unpack_bf16_0[1];
                w_rounded_0[2] = _unpack_bf16_0[2];
                w_rounded_0[3] = _unpack_bf16_0[3];
                w_ss_lo_0 += __fmaf_rn(_unpack_bf16_0[0], _unpack_bf16_0[0], _unpack_bf16_0[1] * _unpack_bf16_0[1]);
                w_ss_hi_0 += __fmaf_rn(_unpack_bf16_0[2], _unpack_bf16_0[2], _unpack_bf16_0[3] * _unpack_bf16_0[3]);
                float w_scaled_0_1[4];
                w_scaled_0_1[0] = accum[4] * w_sa_lo_0 * sw[2];
                w_scaled_0_1[1] = accum[5] * w_sa_lo_0 * sw[3];
                w_scaled_0_1[2] = accum[6] * w_sa_hi_0 * sw[2];
                w_scaled_0_1[3] = accum[7] * w_sa_hi_0 * sw[3];
                unsigned int w_scaled_0_1_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_1[_lp*2 + 0], w_scaled_0_1[_lp*2+1 + 0]));
                    w_scaled_0_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_1[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_1_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_0[4] = _unpack_bf16_1[0];
                w_rounded_0[5] = _unpack_bf16_1[1];
                w_rounded_0[6] = _unpack_bf16_1[2];
                w_rounded_0[7] = _unpack_bf16_1[3];
                w_ss_lo_0 += __fmaf_rn(_unpack_bf16_1[0], _unpack_bf16_1[0], _unpack_bf16_1[1] * _unpack_bf16_1[1]);
                w_ss_hi_0 += __fmaf_rn(_unpack_bf16_1[2], _unpack_bf16_1[2], _unpack_bf16_1[3] * _unpack_bf16_1[3]);
                float w_scaled_0_2[4];
                w_scaled_0_2[0] = accum[8] * w_sa_lo_0 * sw[4];
                w_scaled_0_2[1] = accum[9] * w_sa_lo_0 * sw[5];
                w_scaled_0_2[2] = accum[10] * w_sa_hi_0 * sw[4];
                w_scaled_0_2[3] = accum[11] * w_sa_hi_0 * sw[5];
                unsigned int w_scaled_0_2_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_2[_lp*2 + 0], w_scaled_0_2[_lp*2+1 + 0]));
                    w_scaled_0_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_2[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_2_bf16[_pair]) << 16);
                    (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_2_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_0[8] = _unpack_bf16_2[0];
                w_rounded_0[9] = _unpack_bf16_2[1];
                w_rounded_0[10] = _unpack_bf16_2[2];
                w_rounded_0[11] = _unpack_bf16_2[3];
                w_ss_lo_0 += __fmaf_rn(_unpack_bf16_2[0], _unpack_bf16_2[0], _unpack_bf16_2[1] * _unpack_bf16_2[1]);
                w_ss_hi_0 += __fmaf_rn(_unpack_bf16_2[2], _unpack_bf16_2[2], _unpack_bf16_2[3] * _unpack_bf16_2[3]);
                float w_scaled_0_3[4];
                w_scaled_0_3[0] = accum[12] * w_sa_lo_0 * sw[6];
                w_scaled_0_3[1] = accum[13] * w_sa_lo_0 * sw[7];
                w_scaled_0_3[2] = accum[14] * w_sa_hi_0 * sw[6];
                w_scaled_0_3[3] = accum[15] * w_sa_hi_0 * sw[7];
                unsigned int w_scaled_0_3_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_3[_lp*2 + 0], w_scaled_0_3[_lp*2+1 + 0]));
                    w_scaled_0_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_3[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_3_bf16[_pair]) << 16);
                    (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_3_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_0[12] = _unpack_bf16_3[0];
                w_rounded_0[13] = _unpack_bf16_3[1];
                w_rounded_0[14] = _unpack_bf16_3[2];
                w_rounded_0[15] = _unpack_bf16_3[3];
                w_ss_lo_0 += __fmaf_rn(_unpack_bf16_3[0], _unpack_bf16_3[0], _unpack_bf16_3[1] * _unpack_bf16_3[1]);
                w_ss_hi_0 += __fmaf_rn(_unpack_bf16_3[2], _unpack_bf16_3[2], _unpack_bf16_3[3] * _unpack_bf16_3[3]);
                float w_scaled_0_4[4];
                w_scaled_0_4[0] = accum[16] * w_sa_lo_0 * sw[8];
                w_scaled_0_4[1] = accum[17] * w_sa_lo_0 * sw[9];
                w_scaled_0_4[2] = accum[18] * w_sa_hi_0 * sw[8];
                w_scaled_0_4[3] = accum[19] * w_sa_hi_0 * sw[9];
                unsigned int w_scaled_0_4_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_4[_lp*2 + 0], w_scaled_0_4[_lp*2+1 + 0]));
                    w_scaled_0_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_4[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_4_bf16[_pair]) << 16);
                    (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_4_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_0[16] = _unpack_bf16_4[0];
                w_rounded_0[17] = _unpack_bf16_4[1];
                w_rounded_0[18] = _unpack_bf16_4[2];
                w_rounded_0[19] = _unpack_bf16_4[3];
                w_ss_lo_0 += __fmaf_rn(_unpack_bf16_4[0], _unpack_bf16_4[0], _unpack_bf16_4[1] * _unpack_bf16_4[1]);
                w_ss_hi_0 += __fmaf_rn(_unpack_bf16_4[2], _unpack_bf16_4[2], _unpack_bf16_4[3] * _unpack_bf16_4[3]);
                float w_scaled_0_5[4];
                w_scaled_0_5[0] = accum[20] * w_sa_lo_0 * sw[10];
                w_scaled_0_5[1] = accum[21] * w_sa_lo_0 * sw[11];
                w_scaled_0_5[2] = accum[22] * w_sa_hi_0 * sw[10];
                w_scaled_0_5[3] = accum[23] * w_sa_hi_0 * sw[11];
                unsigned int w_scaled_0_5_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_5[_lp*2 + 0], w_scaled_0_5[_lp*2+1 + 0]));
                    w_scaled_0_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_5[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_5_bf16[_pair]) << 16);
                    (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_5_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_0[20] = _unpack_bf16_5[0];
                w_rounded_0[21] = _unpack_bf16_5[1];
                w_rounded_0[22] = _unpack_bf16_5[2];
                w_rounded_0[23] = _unpack_bf16_5[3];
                w_ss_lo_0 += __fmaf_rn(_unpack_bf16_5[0], _unpack_bf16_5[0], _unpack_bf16_5[1] * _unpack_bf16_5[1]);
                w_ss_hi_0 += __fmaf_rn(_unpack_bf16_5[2], _unpack_bf16_5[2], _unpack_bf16_5[3] * _unpack_bf16_5[3]);
                float w_scaled_0_6[4];
                w_scaled_0_6[0] = accum[24] * w_sa_lo_0 * sw[12];
                w_scaled_0_6[1] = accum[25] * w_sa_lo_0 * sw[13];
                w_scaled_0_6[2] = accum[26] * w_sa_hi_0 * sw[12];
                w_scaled_0_6[3] = accum[27] * w_sa_hi_0 * sw[13];
                unsigned int w_scaled_0_6_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_6[_lp*2 + 0], w_scaled_0_6[_lp*2+1 + 0]));
                    w_scaled_0_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_6[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_6_bf16[_pair]) << 16);
                    (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_6_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_0[24] = _unpack_bf16_6[0];
                w_rounded_0[25] = _unpack_bf16_6[1];
                w_rounded_0[26] = _unpack_bf16_6[2];
                w_rounded_0[27] = _unpack_bf16_6[3];
                w_ss_lo_0 += __fmaf_rn(_unpack_bf16_6[0], _unpack_bf16_6[0], _unpack_bf16_6[1] * _unpack_bf16_6[1]);
                w_ss_hi_0 += __fmaf_rn(_unpack_bf16_6[2], _unpack_bf16_6[2], _unpack_bf16_6[3] * _unpack_bf16_6[3]);
                float w_scaled_0_7[4];
                w_scaled_0_7[0] = accum[28] * w_sa_lo_0 * sw[14];
                w_scaled_0_7[1] = accum[29] * w_sa_lo_0 * sw[15];
                w_scaled_0_7[2] = accum[30] * w_sa_hi_0 * sw[14];
                w_scaled_0_7[3] = accum[31] * w_sa_hi_0 * sw[15];
                unsigned int w_scaled_0_7_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_7[_lp*2 + 0], w_scaled_0_7[_lp*2+1 + 0]));
                    w_scaled_0_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_7[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_7_bf16[_pair]) << 16);
                    (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_7_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_0[28] = _unpack_bf16_7[0];
                w_rounded_0[29] = _unpack_bf16_7[1];
                w_rounded_0[30] = _unpack_bf16_7[2];
                w_rounded_0[31] = _unpack_bf16_7[3];
                w_ss_lo_0 += __fmaf_rn(_unpack_bf16_7[0], _unpack_bf16_7[0], _unpack_bf16_7[1] * _unpack_bf16_7[1]);
                w_ss_hi_0 += __fmaf_rn(_unpack_bf16_7[2], _unpack_bf16_7[2], _unpack_bf16_7[3] * _unpack_bf16_7[3]);
                float w_rstd_lo_0 = 1.0f;
                float w_rstd_hi_0 = 1.0f;
                if (w_kind < 2) {
                    float _shfl_xor_sync_0 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_0, 1);
                    w_ss_lo_0 += _shfl_xor_sync_0;
                    float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_0, 2);
                    w_ss_lo_0 += _shfl_xor_sync_1;
                    float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_0, 1);
                    w_ss_hi_0 += _shfl_xor_sync_2;
                    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_0, 2);
                    w_ss_hi_0 += _shfl_xor_sync_3;
                    int w_srow_lo_0 = warp_m * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n * 64 + w_srow_lo_0] = w_ss_lo_0;
                        rowstat[warp_n * 64 + w_srow_lo_0 + 8] = w_ss_hi_0;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float w_other_lo_0 = rowstat[(1 - warp_n) * 64 + w_srow_lo_0];
                    float w_other_hi_0 = rowstat[(1 - warp_n) * 64 + w_srow_lo_0 + 8];
                    float _fdiv_5 = __fdiv_rn(w_ss_lo_0 + w_other_lo_0, 128.0f);
                    float _rsqrt_4 = rsqrtf(_fdiv_5 + eps);
                    w_rstd_lo_0 = _rsqrt_4;
                    float _fdiv_7 = __fdiv_rn(w_ss_hi_0 + w_other_hi_0, 128.0f);
                    float _rsqrt_6 = rsqrtf(_fdiv_7 + eps);
                    w_rstd_hi_0 = _rsqrt_6;
                }
                int w_srow_st_0 = warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float w_normed_0_0[4];
                if (w_kind < 2) {
                    w_normed_0_0[0] = w_rounded_0[0] * w_rstd_lo_0 * nw[0];
                    w_normed_0_0[1] = w_rounded_0[1] * w_rstd_lo_0 * nw[1];
                    w_normed_0_0[2] = w_rounded_0[2] * w_rstd_hi_0 * nw[0];
                    w_normed_0_0[3] = w_rounded_0[3] * w_rstd_hi_0 * nw[1];
                } else {
                    w_normed_0_0[0] = w_rounded_0[0];
                    w_normed_0_0[1] = w_rounded_0[1];
                    w_normed_0_0[2] = w_rounded_0[2];
                    w_normed_0_0[3] = w_rounded_0[3];
                }
                unsigned int w_normed_0_0_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_0[_lp*2 + 0], w_normed_0_0[_lp*2+1 + 0]));
                    w_normed_0_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_0 * 256) + (unsigned int)(warp_n * 64 * 2 ^ (w_srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_0_bf16[1]))
                    : "memory");
                float w_normed_0_1[4];
                if (w_kind < 2) {
                    w_normed_0_1[0] = w_rounded_0[4] * w_rstd_lo_0 * nw[2];
                    w_normed_0_1[1] = w_rounded_0[5] * w_rstd_lo_0 * nw[3];
                    w_normed_0_1[2] = w_rounded_0[6] * w_rstd_hi_0 * nw[2];
                    w_normed_0_1[3] = w_rounded_0[7] * w_rstd_hi_0 * nw[3];
                } else {
                    w_normed_0_1[0] = w_rounded_0[4];
                    w_normed_0_1[1] = w_rounded_0[5];
                    w_normed_0_1[2] = w_rounded_0[6];
                    w_normed_0_1[3] = w_rounded_0[7];
                }
                unsigned int w_normed_0_1_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_1[_lp*2 + 0], w_normed_0_1[_lp*2+1 + 0]));
                    w_normed_0_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_0 * 256) + (unsigned int)((warp_n * 64 + 8) * 2 ^ (w_srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_1_bf16[1]))
                    : "memory");
                float w_normed_0_2[4];
                if (w_kind < 2) {
                    w_normed_0_2[0] = w_rounded_0[8] * w_rstd_lo_0 * nw[4];
                    w_normed_0_2[1] = w_rounded_0[9] * w_rstd_lo_0 * nw[5];
                    w_normed_0_2[2] = w_rounded_0[10] * w_rstd_hi_0 * nw[4];
                    w_normed_0_2[3] = w_rounded_0[11] * w_rstd_hi_0 * nw[5];
                } else {
                    w_normed_0_2[0] = w_rounded_0[8];
                    w_normed_0_2[1] = w_rounded_0[9];
                    w_normed_0_2[2] = w_rounded_0[10];
                    w_normed_0_2[3] = w_rounded_0[11];
                }
                unsigned int w_normed_0_2_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_2[_lp*2 + 0], w_normed_0_2[_lp*2+1 + 0]));
                    w_normed_0_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_0 * 256) + (unsigned int)((warp_n * 64 + 16) * 2 ^ (w_srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_2_bf16[1]))
                    : "memory");
                float w_normed_0_3[4];
                if (w_kind < 2) {
                    w_normed_0_3[0] = w_rounded_0[12] * w_rstd_lo_0 * nw[6];
                    w_normed_0_3[1] = w_rounded_0[13] * w_rstd_lo_0 * nw[7];
                    w_normed_0_3[2] = w_rounded_0[14] * w_rstd_hi_0 * nw[6];
                    w_normed_0_3[3] = w_rounded_0[15] * w_rstd_hi_0 * nw[7];
                } else {
                    w_normed_0_3[0] = w_rounded_0[12];
                    w_normed_0_3[1] = w_rounded_0[13];
                    w_normed_0_3[2] = w_rounded_0[14];
                    w_normed_0_3[3] = w_rounded_0[15];
                }
                unsigned int w_normed_0_3_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_3[_lp*2 + 0], w_normed_0_3[_lp*2+1 + 0]));
                    w_normed_0_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_0 * 256) + (unsigned int)((warp_n * 64 + 24) * 2 ^ (w_srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_3_bf16[1]))
                    : "memory");
                float w_normed_0_4[4];
                if (w_kind < 2) {
                    w_normed_0_4[0] = w_rounded_0[16] * w_rstd_lo_0 * nw[8];
                    w_normed_0_4[1] = w_rounded_0[17] * w_rstd_lo_0 * nw[9];
                    w_normed_0_4[2] = w_rounded_0[18] * w_rstd_hi_0 * nw[8];
                    w_normed_0_4[3] = w_rounded_0[19] * w_rstd_hi_0 * nw[9];
                } else {
                    w_normed_0_4[0] = w_rounded_0[16];
                    w_normed_0_4[1] = w_rounded_0[17];
                    w_normed_0_4[2] = w_rounded_0[18];
                    w_normed_0_4[3] = w_rounded_0[19];
                }
                unsigned int w_normed_0_4_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_4[_lp*2 + 0], w_normed_0_4[_lp*2+1 + 0]));
                    w_normed_0_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_0 * 256) + (unsigned int)((warp_n * 64 + 32) * 2 ^ (w_srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_4_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_4_bf16[1]))
                    : "memory");
                float w_normed_0_5[4];
                if (w_kind < 2) {
                    w_normed_0_5[0] = w_rounded_0[20] * w_rstd_lo_0 * nw[10];
                    w_normed_0_5[1] = w_rounded_0[21] * w_rstd_lo_0 * nw[11];
                    w_normed_0_5[2] = w_rounded_0[22] * w_rstd_hi_0 * nw[10];
                    w_normed_0_5[3] = w_rounded_0[23] * w_rstd_hi_0 * nw[11];
                } else {
                    w_normed_0_5[0] = w_rounded_0[20];
                    w_normed_0_5[1] = w_rounded_0[21];
                    w_normed_0_5[2] = w_rounded_0[22];
                    w_normed_0_5[3] = w_rounded_0[23];
                }
                unsigned int w_normed_0_5_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_5[_lp*2 + 0], w_normed_0_5[_lp*2+1 + 0]));
                    w_normed_0_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_0 * 256) + (unsigned int)((warp_n * 64 + 40) * 2 ^ (w_srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_5_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_5_bf16[1]))
                    : "memory");
                float w_normed_0_6[4];
                if (w_kind < 2) {
                    w_normed_0_6[0] = w_rounded_0[24] * w_rstd_lo_0 * nw[12];
                    w_normed_0_6[1] = w_rounded_0[25] * w_rstd_lo_0 * nw[13];
                    w_normed_0_6[2] = w_rounded_0[26] * w_rstd_hi_0 * nw[12];
                    w_normed_0_6[3] = w_rounded_0[27] * w_rstd_hi_0 * nw[13];
                } else {
                    w_normed_0_6[0] = w_rounded_0[24];
                    w_normed_0_6[1] = w_rounded_0[25];
                    w_normed_0_6[2] = w_rounded_0[26];
                    w_normed_0_6[3] = w_rounded_0[27];
                }
                unsigned int w_normed_0_6_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_6[_lp*2 + 0], w_normed_0_6[_lp*2+1 + 0]));
                    w_normed_0_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_0 * 256) + (unsigned int)((warp_n * 64 + 48) * 2 ^ (w_srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_6_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_6_bf16[1]))
                    : "memory");
                float w_normed_0_7[4];
                if (w_kind < 2) {
                    w_normed_0_7[0] = w_rounded_0[28] * w_rstd_lo_0 * nw[14];
                    w_normed_0_7[1] = w_rounded_0[29] * w_rstd_lo_0 * nw[15];
                    w_normed_0_7[2] = w_rounded_0[30] * w_rstd_hi_0 * nw[14];
                    w_normed_0_7[3] = w_rounded_0[31] * w_rstd_hi_0 * nw[15];
                } else {
                    w_normed_0_7[0] = w_rounded_0[28];
                    w_normed_0_7[1] = w_rounded_0[29];
                    w_normed_0_7[2] = w_rounded_0[30];
                    w_normed_0_7[3] = w_rounded_0[31];
                }
                unsigned int w_normed_0_7_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_7[_lp*2 + 0], w_normed_0_7[_lp*2+1 + 0]));
                    w_normed_0_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_0 * 256) + (unsigned int)((warp_n * 64 + 56) * 2 ^ (w_srow_st_0 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_7_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_7_bf16[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_srow_0 = role_tid >> 2;
                int w_c_0 = role_tid & 3;
                int w_grow_0 = tile_m * 256 + w_srow_0;
                int w_chunk_lo_0 = ((w_c_0 < 3) ? w_c_0 * 16 : 96);
                int w_chunk_hi_0 = ((w_c_0 < 3) ? w_chunk_lo_0 + 48 : 112);
                if (w_grow_0 < M) {
                    unsigned int w_x_lo_0[8];
                    unsigned int w_x_hi_0[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0[(0) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_0 * 256) + (unsigned int)(w_chunk_lo_0 * 2 ^ (w_srow_0 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0[(4) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_0 * 256) + (unsigned int)(w_chunk_lo_0 * 2 + 16 ^ (w_srow_0 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0[(0) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_0 * 256) + (unsigned int)(w_chunk_hi_0 * 2 ^ (w_srow_0 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0[(4) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_0 * 256) + (unsigned int)(w_chunk_hi_0 * 2 + 16 ^ (w_srow_0 & 7) << 4)));
                    if (w_c_0 < 3) {
                        if (w_kind < 2) {
                            float _unpack_bf16_8[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_0[_pair]) << 16);
                                (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_0[_pair]) & 0xffff0000u);
                            }
                            float _unpack_bf16_9[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_0[_pair]) << 16);
                                (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_0[_pair]) & 0xffff0000u);
                            }
                            float _vec_load_0[8];
                            {
                                const uint4* _vptr_16 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_0 * 96 + w_chunk_lo_0) + 0);
                                uint4 _vld_16[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_16[_blk] = _vptr_16[_blk];
                                    uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_1[8];
                            {
                                const uint4* _vptr_17 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_0 * 96 + w_chunk_lo_0 + 8) + 0);
                                uint4 _vld_17[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_17[_blk] = _vptr_17[_blk];
                                    uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
                                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_2[8];
                            {
                                const uint4* _vptr_18 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_0 * 96 + 48 + w_chunk_lo_0) + 0);
                                uint4 _vld_18[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_18[_blk] = _vptr_18[_blk];
                                    uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                                        (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_3[8];
                            {
                                const uint4* _vptr_19 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_0 * 96 + 48 + w_chunk_lo_0 + 8) + 0);
                                uint4 _vld_19[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_19[_blk] = _vptr_19[_blk];
                                    uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float w_o_lo_0[16];
                            float w_o_hi_0[16];
                            float w_cv0_0_0 = _vec_load_0[0];
                            float w_sv0_0_0 = _vec_load_2[0];
                            float w_cv1_0_0 = _vec_load_1[0];
                            float w_sv1_0_0 = _vec_load_3[0];
                            w_o_lo_0[0] = __fmaf_rn(_unpack_bf16_8[0], w_cv0_0_0, -(_unpack_bf16_9[0] * w_sv0_0_0));
                            w_o_hi_0[0] = __fmaf_rn(_unpack_bf16_9[0], w_cv0_0_0, _unpack_bf16_8[0] * w_sv0_0_0);
                            w_o_lo_0[8] = __fmaf_rn(_unpack_bf16_8[8], w_cv1_0_0, -(_unpack_bf16_9[8] * w_sv1_0_0));
                            w_o_hi_0[8] = __fmaf_rn(_unpack_bf16_9[8], w_cv1_0_0, _unpack_bf16_8[8] * w_sv1_0_0);
                            float w_cv0_0_1 = _vec_load_0[1];
                            float w_sv0_0_1 = _vec_load_2[1];
                            float w_cv1_0_1 = _vec_load_1[1];
                            float w_sv1_0_1 = _vec_load_3[1];
                            w_o_lo_0[1] = __fmaf_rn(_unpack_bf16_8[1], w_cv0_0_1, -(_unpack_bf16_9[1] * w_sv0_0_1));
                            w_o_hi_0[1] = __fmaf_rn(_unpack_bf16_9[1], w_cv0_0_1, _unpack_bf16_8[1] * w_sv0_0_1);
                            w_o_lo_0[9] = __fmaf_rn(_unpack_bf16_8[9], w_cv1_0_1, -(_unpack_bf16_9[9] * w_sv1_0_1));
                            w_o_hi_0[9] = __fmaf_rn(_unpack_bf16_9[9], w_cv1_0_1, _unpack_bf16_8[9] * w_sv1_0_1);
                            float w_cv0_0_2 = _vec_load_0[2];
                            float w_sv0_0_2 = _vec_load_2[2];
                            float w_cv1_0_2 = _vec_load_1[2];
                            float w_sv1_0_2 = _vec_load_3[2];
                            w_o_lo_0[2] = __fmaf_rn(_unpack_bf16_8[2], w_cv0_0_2, -(_unpack_bf16_9[2] * w_sv0_0_2));
                            w_o_hi_0[2] = __fmaf_rn(_unpack_bf16_9[2], w_cv0_0_2, _unpack_bf16_8[2] * w_sv0_0_2);
                            w_o_lo_0[10] = __fmaf_rn(_unpack_bf16_8[10], w_cv1_0_2, -(_unpack_bf16_9[10] * w_sv1_0_2));
                            w_o_hi_0[10] = __fmaf_rn(_unpack_bf16_9[10], w_cv1_0_2, _unpack_bf16_8[10] * w_sv1_0_2);
                            float w_cv0_0_3 = _vec_load_0[3];
                            float w_sv0_0_3 = _vec_load_2[3];
                            float w_cv1_0_3 = _vec_load_1[3];
                            float w_sv1_0_3 = _vec_load_3[3];
                            w_o_lo_0[3] = __fmaf_rn(_unpack_bf16_8[3], w_cv0_0_3, -(_unpack_bf16_9[3] * w_sv0_0_3));
                            w_o_hi_0[3] = __fmaf_rn(_unpack_bf16_9[3], w_cv0_0_3, _unpack_bf16_8[3] * w_sv0_0_3);
                            w_o_lo_0[11] = __fmaf_rn(_unpack_bf16_8[11], w_cv1_0_3, -(_unpack_bf16_9[11] * w_sv1_0_3));
                            w_o_hi_0[11] = __fmaf_rn(_unpack_bf16_9[11], w_cv1_0_3, _unpack_bf16_8[11] * w_sv1_0_3);
                            float w_cv0_0_4 = _vec_load_0[4];
                            float w_sv0_0_4 = _vec_load_2[4];
                            float w_cv1_0_4 = _vec_load_1[4];
                            float w_sv1_0_4 = _vec_load_3[4];
                            w_o_lo_0[4] = __fmaf_rn(_unpack_bf16_8[4], w_cv0_0_4, -(_unpack_bf16_9[4] * w_sv0_0_4));
                            w_o_hi_0[4] = __fmaf_rn(_unpack_bf16_9[4], w_cv0_0_4, _unpack_bf16_8[4] * w_sv0_0_4);
                            w_o_lo_0[12] = __fmaf_rn(_unpack_bf16_8[12], w_cv1_0_4, -(_unpack_bf16_9[12] * w_sv1_0_4));
                            w_o_hi_0[12] = __fmaf_rn(_unpack_bf16_9[12], w_cv1_0_4, _unpack_bf16_8[12] * w_sv1_0_4);
                            float w_cv0_0_5 = _vec_load_0[5];
                            float w_sv0_0_5 = _vec_load_2[5];
                            float w_cv1_0_5 = _vec_load_1[5];
                            float w_sv1_0_5 = _vec_load_3[5];
                            w_o_lo_0[5] = __fmaf_rn(_unpack_bf16_8[5], w_cv0_0_5, -(_unpack_bf16_9[5] * w_sv0_0_5));
                            w_o_hi_0[5] = __fmaf_rn(_unpack_bf16_9[5], w_cv0_0_5, _unpack_bf16_8[5] * w_sv0_0_5);
                            w_o_lo_0[13] = __fmaf_rn(_unpack_bf16_8[13], w_cv1_0_5, -(_unpack_bf16_9[13] * w_sv1_0_5));
                            w_o_hi_0[13] = __fmaf_rn(_unpack_bf16_9[13], w_cv1_0_5, _unpack_bf16_8[13] * w_sv1_0_5);
                            float w_cv0_0_6 = _vec_load_0[6];
                            float w_sv0_0_6 = _vec_load_2[6];
                            float w_cv1_0_6 = _vec_load_1[6];
                            float w_sv1_0_6 = _vec_load_3[6];
                            w_o_lo_0[6] = __fmaf_rn(_unpack_bf16_8[6], w_cv0_0_6, -(_unpack_bf16_9[6] * w_sv0_0_6));
                            w_o_hi_0[6] = __fmaf_rn(_unpack_bf16_9[6], w_cv0_0_6, _unpack_bf16_8[6] * w_sv0_0_6);
                            w_o_lo_0[14] = __fmaf_rn(_unpack_bf16_8[14], w_cv1_0_6, -(_unpack_bf16_9[14] * w_sv1_0_6));
                            w_o_hi_0[14] = __fmaf_rn(_unpack_bf16_9[14], w_cv1_0_6, _unpack_bf16_8[14] * w_sv1_0_6);
                            float w_cv0_0_7 = _vec_load_0[7];
                            float w_sv0_0_7 = _vec_load_2[7];
                            float w_cv1_0_7 = _vec_load_1[7];
                            float w_sv1_0_7 = _vec_load_3[7];
                            w_o_lo_0[7] = __fmaf_rn(_unpack_bf16_8[7], w_cv0_0_7, -(_unpack_bf16_9[7] * w_sv0_0_7));
                            w_o_hi_0[7] = __fmaf_rn(_unpack_bf16_9[7], w_cv0_0_7, _unpack_bf16_8[7] * w_sv0_0_7);
                            w_o_lo_0[15] = __fmaf_rn(_unpack_bf16_8[15], w_cv1_0_7, -(_unpack_bf16_9[15] * w_sv1_0_7));
                            w_o_hi_0[15] = __fmaf_rn(_unpack_bf16_9[15], w_cv1_0_7, _unpack_bf16_8[15] * w_sv1_0_7);
                            unsigned int w_o_lo_0_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_lo_0[_lp*2 + 0], w_o_lo_0[_lp*2+1 + 0]));
                                w_o_lo_0_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int w_o_hi_0_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_hi_0[_lp*2 + 0], w_o_hi_0[_lp*2+1 + 0]));
                                w_o_hi_0_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            w_x_lo_0[0] = w_o_lo_0_bf16[0];
                            w_x_hi_0[0] = w_o_hi_0_bf16[0];
                            w_x_lo_0[1] = w_o_lo_0_bf16[1];
                            w_x_hi_0[1] = w_o_hi_0_bf16[1];
                            w_x_lo_0[2] = w_o_lo_0_bf16[2];
                            w_x_hi_0[2] = w_o_hi_0_bf16[2];
                            w_x_lo_0[3] = w_o_lo_0_bf16[3];
                            w_x_hi_0[3] = w_o_hi_0_bf16[3];
                            w_x_lo_0[4] = w_o_lo_0_bf16[4];
                            w_x_hi_0[4] = w_o_hi_0_bf16[4];
                            w_x_lo_0[5] = w_o_lo_0_bf16[5];
                            w_x_hi_0[5] = w_o_hi_0_bf16[5];
                            w_x_lo_0[6] = w_o_lo_0_bf16[6];
                            w_x_hi_0[6] = w_o_hi_0_bf16[6];
                            w_x_lo_0[7] = w_o_lo_0_bf16[7];
                            w_x_hi_0[7] = w_o_hi_0_bf16[7];
                        }
                    }
                    unsigned int w_q_words_0[4];
                    float w_q_vals_0[16];
                    float _unpack_bf16_10[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_0[_pair]) << 16);
                        (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_0[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_0[0] = _unpack_bf16_10[0] * w_out_scale_sel;
                    w_q_vals_0[1] = _unpack_bf16_10[1] * w_out_scale_sel;
                    w_q_vals_0[2] = _unpack_bf16_10[2] * w_out_scale_sel;
                    w_q_vals_0[3] = _unpack_bf16_10[3] * w_out_scale_sel;
                    w_q_vals_0[4] = _unpack_bf16_10[4] * w_out_scale_sel;
                    w_q_vals_0[5] = _unpack_bf16_10[5] * w_out_scale_sel;
                    w_q_vals_0[6] = _unpack_bf16_10[6] * w_out_scale_sel;
                    w_q_vals_0[7] = _unpack_bf16_10[7] * w_out_scale_sel;
                    w_q_vals_0[8] = _unpack_bf16_10[8] * w_out_scale_sel;
                    w_q_vals_0[9] = _unpack_bf16_10[9] * w_out_scale_sel;
                    w_q_vals_0[10] = _unpack_bf16_10[10] * w_out_scale_sel;
                    w_q_vals_0[11] = _unpack_bf16_10[11] * w_out_scale_sel;
                    w_q_vals_0[12] = _unpack_bf16_10[12] * w_out_scale_sel;
                    w_q_vals_0[13] = _unpack_bf16_10[13] * w_out_scale_sel;
                    w_q_vals_0[14] = _unpack_bf16_10[14] * w_out_scale_sel;
                    w_q_vals_0[15] = _unpack_bf16_10[15] * w_out_scale_sel;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_0[0]), "f"(w_q_vals_0[1]),
                                               "f"(w_q_vals_0[2]), "f"(w_q_vals_0[3]));
                        w_q_words_0[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_0[4]), "f"(w_q_vals_0[5]),
                                               "f"(w_q_vals_0[6]), "f"(w_q_vals_0[7]));
                        w_q_words_0[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_0[8]), "f"(w_q_vals_0[9]),
                                               "f"(w_q_vals_0[10]), "f"(w_q_vals_0[11]));
                        w_q_words_0[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_0[12]), "f"(w_q_vals_0[13]),
                                               "f"(w_q_vals_0[14]), "f"(w_q_vals_0[15]));
                        w_q_words_0[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(w_srow_0 * 128) + (unsigned int)(w_chunk_lo_0 ^ (w_srow_0 & 7) << 4)), "r"(w_q_words_0[0]), "r"(w_q_words_0[1]), "r"(w_q_words_0[2]), "r"(w_q_words_0[3]) : "memory");
                    float _unpack_bf16_11[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_0[_pair]) << 16);
                        (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_0[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_0[0] = _unpack_bf16_11[0] * w_out_scale_sel;
                    w_q_vals_0[1] = _unpack_bf16_11[1] * w_out_scale_sel;
                    w_q_vals_0[2] = _unpack_bf16_11[2] * w_out_scale_sel;
                    w_q_vals_0[3] = _unpack_bf16_11[3] * w_out_scale_sel;
                    w_q_vals_0[4] = _unpack_bf16_11[4] * w_out_scale_sel;
                    w_q_vals_0[5] = _unpack_bf16_11[5] * w_out_scale_sel;
                    w_q_vals_0[6] = _unpack_bf16_11[6] * w_out_scale_sel;
                    w_q_vals_0[7] = _unpack_bf16_11[7] * w_out_scale_sel;
                    w_q_vals_0[8] = _unpack_bf16_11[8] * w_out_scale_sel;
                    w_q_vals_0[9] = _unpack_bf16_11[9] * w_out_scale_sel;
                    w_q_vals_0[10] = _unpack_bf16_11[10] * w_out_scale_sel;
                    w_q_vals_0[11] = _unpack_bf16_11[11] * w_out_scale_sel;
                    w_q_vals_0[12] = _unpack_bf16_11[12] * w_out_scale_sel;
                    w_q_vals_0[13] = _unpack_bf16_11[13] * w_out_scale_sel;
                    w_q_vals_0[14] = _unpack_bf16_11[14] * w_out_scale_sel;
                    w_q_vals_0[15] = _unpack_bf16_11[15] * w_out_scale_sel;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_0[0]), "f"(w_q_vals_0[1]),
                                               "f"(w_q_vals_0[2]), "f"(w_q_vals_0[3]));
                        w_q_words_0[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_0[4]), "f"(w_q_vals_0[5]),
                                               "f"(w_q_vals_0[6]), "f"(w_q_vals_0[7]));
                        w_q_words_0[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_0[8]), "f"(w_q_vals_0[9]),
                                               "f"(w_q_vals_0[10]), "f"(w_q_vals_0[11]));
                        w_q_words_0[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_0[12]), "f"(w_q_vals_0[13]),
                                               "f"(w_q_vals_0[14]), "f"(w_q_vals_0[15]));
                        w_q_words_0[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(w_srow_0 * 128) + (unsigned int)(w_chunk_hi_0 ^ (w_srow_0 & 7) << 4)), "r"(w_q_words_0[0]), "r"(w_q_words_0[1]), "r"(w_q_words_0[2]), "r"(w_q_words_0[3]) : "memory");
                }
                __syncwarp();
                int s_row_0_0 = (role_tid >> 5) * 8 + lane / 8;
                int s_piece_0_0 = lane % 8;
                unsigned int s_words_0_0[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_0[(0) + 3]))
                    : "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(s_row_0_0 * 128) + (unsigned int)(s_piece_0_0 * 16 ^ (s_row_0_0 & 7) << 4)));
                int s_dst_0_0 = ((tile_m * 256 + s_row_0_0) * 56 + w_head) * 32 + s_piece_0_0 * 4;
                if (tile_m * 256 + s_row_0_0 < M) {
                    if (w_kind == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_0_0)[0] = reinterpret_cast<int4*>(s_words_0_0)[0];
                    } else if (w_kind == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_0_0)[0] = reinterpret_cast<int4*>(s_words_0_0)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_0_0)[0] = reinterpret_cast<int4*>(s_words_0_0)[0];
                    }
                }
                int s_row_0_1 = (role_tid >> 5) * 8 + 4 + lane / 8;
                int s_piece_0_1 = lane % 8;
                unsigned int s_words_0_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_1[(0) + 3]))
                    : "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(s_row_0_1 * 128) + (unsigned int)(s_piece_0_1 * 16 ^ (s_row_0_1 & 7) << 4)));
                int s_dst_0_1 = ((tile_m * 256 + s_row_0_1) * 56 + w_head) * 32 + s_piece_0_1 * 4;
                if (tile_m * 256 + s_row_0_1 < M) {
                    if (w_kind == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_0_1)[0] = reinterpret_cast<int4*>(s_words_0_1)[0];
                    } else if (w_kind == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_0_1)[0] = reinterpret_cast<int4*>(s_words_0_1)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_0_1)[0] = reinterpret_cast<int4*>(s_words_0_1)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_row_lo_1 = tile_m * 256 + 64 + warp_m * 16 + (lane >> 2);
                int w_row_hi_1 = w_row_lo_1 + 8;
                float w_sa_lo_1 = ((w_row_lo_1 < M) ? act_scale[w_row_lo_1] : 0.0f);
                float w_sa_hi_1 = ((w_row_hi_1 < M) ? act_scale[w_row_hi_1] : 0.0f);
                float w_rounded_1[32];
                float w_ss_lo_1 = 0.0f;
                float w_ss_hi_1 = 0.0f;
                float w_scaled_1_0[4];
                w_scaled_1_0[0] = accum[32] * w_sa_lo_1 * sw[0];
                w_scaled_1_0[1] = accum[33] * w_sa_lo_1 * sw[1];
                w_scaled_1_0[2] = accum[34] * w_sa_hi_1 * sw[0];
                w_scaled_1_0[3] = accum[35] * w_sa_hi_1 * sw[1];
                unsigned int w_scaled_1_0_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_0[_lp*2 + 0], w_scaled_1_0[_lp*2+1 + 0]));
                    w_scaled_1_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_12[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_0_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_1[0] = _unpack_bf16_12[0];
                w_rounded_1[1] = _unpack_bf16_12[1];
                w_rounded_1[2] = _unpack_bf16_12[2];
                w_rounded_1[3] = _unpack_bf16_12[3];
                w_ss_lo_1 += __fmaf_rn(_unpack_bf16_12[0], _unpack_bf16_12[0], _unpack_bf16_12[1] * _unpack_bf16_12[1]);
                w_ss_hi_1 += __fmaf_rn(_unpack_bf16_12[2], _unpack_bf16_12[2], _unpack_bf16_12[3] * _unpack_bf16_12[3]);
                float w_scaled_1_1[4];
                w_scaled_1_1[0] = accum[36] * w_sa_lo_1 * sw[2];
                w_scaled_1_1[1] = accum[37] * w_sa_lo_1 * sw[3];
                w_scaled_1_1[2] = accum[38] * w_sa_hi_1 * sw[2];
                w_scaled_1_1[3] = accum[39] * w_sa_hi_1 * sw[3];
                unsigned int w_scaled_1_1_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_1[_lp*2 + 0], w_scaled_1_1[_lp*2+1 + 0]));
                    w_scaled_1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_13[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_1_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_1[4] = _unpack_bf16_13[0];
                w_rounded_1[5] = _unpack_bf16_13[1];
                w_rounded_1[6] = _unpack_bf16_13[2];
                w_rounded_1[7] = _unpack_bf16_13[3];
                w_ss_lo_1 += __fmaf_rn(_unpack_bf16_13[0], _unpack_bf16_13[0], _unpack_bf16_13[1] * _unpack_bf16_13[1]);
                w_ss_hi_1 += __fmaf_rn(_unpack_bf16_13[2], _unpack_bf16_13[2], _unpack_bf16_13[3] * _unpack_bf16_13[3]);
                float w_scaled_1_2[4];
                w_scaled_1_2[0] = accum[40] * w_sa_lo_1 * sw[4];
                w_scaled_1_2[1] = accum[41] * w_sa_lo_1 * sw[5];
                w_scaled_1_2[2] = accum[42] * w_sa_hi_1 * sw[4];
                w_scaled_1_2[3] = accum[43] * w_sa_hi_1 * sw[5];
                unsigned int w_scaled_1_2_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_2[_lp*2 + 0], w_scaled_1_2[_lp*2+1 + 0]));
                    w_scaled_1_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_14[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_2_bf16[_pair]) << 16);
                    (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_2_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_1[8] = _unpack_bf16_14[0];
                w_rounded_1[9] = _unpack_bf16_14[1];
                w_rounded_1[10] = _unpack_bf16_14[2];
                w_rounded_1[11] = _unpack_bf16_14[3];
                w_ss_lo_1 += __fmaf_rn(_unpack_bf16_14[0], _unpack_bf16_14[0], _unpack_bf16_14[1] * _unpack_bf16_14[1]);
                w_ss_hi_1 += __fmaf_rn(_unpack_bf16_14[2], _unpack_bf16_14[2], _unpack_bf16_14[3] * _unpack_bf16_14[3]);
                float w_scaled_1_3[4];
                w_scaled_1_3[0] = accum[44] * w_sa_lo_1 * sw[6];
                w_scaled_1_3[1] = accum[45] * w_sa_lo_1 * sw[7];
                w_scaled_1_3[2] = accum[46] * w_sa_hi_1 * sw[6];
                w_scaled_1_3[3] = accum[47] * w_sa_hi_1 * sw[7];
                unsigned int w_scaled_1_3_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_3[_lp*2 + 0], w_scaled_1_3[_lp*2+1 + 0]));
                    w_scaled_1_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_15[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_3_bf16[_pair]) << 16);
                    (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_3_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_1[12] = _unpack_bf16_15[0];
                w_rounded_1[13] = _unpack_bf16_15[1];
                w_rounded_1[14] = _unpack_bf16_15[2];
                w_rounded_1[15] = _unpack_bf16_15[3];
                w_ss_lo_1 += __fmaf_rn(_unpack_bf16_15[0], _unpack_bf16_15[0], _unpack_bf16_15[1] * _unpack_bf16_15[1]);
                w_ss_hi_1 += __fmaf_rn(_unpack_bf16_15[2], _unpack_bf16_15[2], _unpack_bf16_15[3] * _unpack_bf16_15[3]);
                float w_scaled_1_4[4];
                w_scaled_1_4[0] = accum[48] * w_sa_lo_1 * sw[8];
                w_scaled_1_4[1] = accum[49] * w_sa_lo_1 * sw[9];
                w_scaled_1_4[2] = accum[50] * w_sa_hi_1 * sw[8];
                w_scaled_1_4[3] = accum[51] * w_sa_hi_1 * sw[9];
                unsigned int w_scaled_1_4_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_4[_lp*2 + 0], w_scaled_1_4[_lp*2+1 + 0]));
                    w_scaled_1_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_16[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_4_bf16[_pair]) << 16);
                    (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_4_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_1[16] = _unpack_bf16_16[0];
                w_rounded_1[17] = _unpack_bf16_16[1];
                w_rounded_1[18] = _unpack_bf16_16[2];
                w_rounded_1[19] = _unpack_bf16_16[3];
                w_ss_lo_1 += __fmaf_rn(_unpack_bf16_16[0], _unpack_bf16_16[0], _unpack_bf16_16[1] * _unpack_bf16_16[1]);
                w_ss_hi_1 += __fmaf_rn(_unpack_bf16_16[2], _unpack_bf16_16[2], _unpack_bf16_16[3] * _unpack_bf16_16[3]);
                float w_scaled_1_5[4];
                w_scaled_1_5[0] = accum[52] * w_sa_lo_1 * sw[10];
                w_scaled_1_5[1] = accum[53] * w_sa_lo_1 * sw[11];
                w_scaled_1_5[2] = accum[54] * w_sa_hi_1 * sw[10];
                w_scaled_1_5[3] = accum[55] * w_sa_hi_1 * sw[11];
                unsigned int w_scaled_1_5_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_5[_lp*2 + 0], w_scaled_1_5[_lp*2+1 + 0]));
                    w_scaled_1_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_17[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_5_bf16[_pair]) << 16);
                    (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_5_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_1[20] = _unpack_bf16_17[0];
                w_rounded_1[21] = _unpack_bf16_17[1];
                w_rounded_1[22] = _unpack_bf16_17[2];
                w_rounded_1[23] = _unpack_bf16_17[3];
                w_ss_lo_1 += __fmaf_rn(_unpack_bf16_17[0], _unpack_bf16_17[0], _unpack_bf16_17[1] * _unpack_bf16_17[1]);
                w_ss_hi_1 += __fmaf_rn(_unpack_bf16_17[2], _unpack_bf16_17[2], _unpack_bf16_17[3] * _unpack_bf16_17[3]);
                float w_scaled_1_6[4];
                w_scaled_1_6[0] = accum[56] * w_sa_lo_1 * sw[12];
                w_scaled_1_6[1] = accum[57] * w_sa_lo_1 * sw[13];
                w_scaled_1_6[2] = accum[58] * w_sa_hi_1 * sw[12];
                w_scaled_1_6[3] = accum[59] * w_sa_hi_1 * sw[13];
                unsigned int w_scaled_1_6_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_6[_lp*2 + 0], w_scaled_1_6[_lp*2+1 + 0]));
                    w_scaled_1_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_18[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_6_bf16[_pair]) << 16);
                    (&_unpack_bf16_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_6_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_1[24] = _unpack_bf16_18[0];
                w_rounded_1[25] = _unpack_bf16_18[1];
                w_rounded_1[26] = _unpack_bf16_18[2];
                w_rounded_1[27] = _unpack_bf16_18[3];
                w_ss_lo_1 += __fmaf_rn(_unpack_bf16_18[0], _unpack_bf16_18[0], _unpack_bf16_18[1] * _unpack_bf16_18[1]);
                w_ss_hi_1 += __fmaf_rn(_unpack_bf16_18[2], _unpack_bf16_18[2], _unpack_bf16_18[3] * _unpack_bf16_18[3]);
                float w_scaled_1_7[4];
                w_scaled_1_7[0] = accum[60] * w_sa_lo_1 * sw[14];
                w_scaled_1_7[1] = accum[61] * w_sa_lo_1 * sw[15];
                w_scaled_1_7[2] = accum[62] * w_sa_hi_1 * sw[14];
                w_scaled_1_7[3] = accum[63] * w_sa_hi_1 * sw[15];
                unsigned int w_scaled_1_7_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_7[_lp*2 + 0], w_scaled_1_7[_lp*2+1 + 0]));
                    w_scaled_1_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_19[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_7_bf16[_pair]) << 16);
                    (&_unpack_bf16_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_7_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_1[28] = _unpack_bf16_19[0];
                w_rounded_1[29] = _unpack_bf16_19[1];
                w_rounded_1[30] = _unpack_bf16_19[2];
                w_rounded_1[31] = _unpack_bf16_19[3];
                w_ss_lo_1 += __fmaf_rn(_unpack_bf16_19[0], _unpack_bf16_19[0], _unpack_bf16_19[1] * _unpack_bf16_19[1]);
                w_ss_hi_1 += __fmaf_rn(_unpack_bf16_19[2], _unpack_bf16_19[2], _unpack_bf16_19[3] * _unpack_bf16_19[3]);
                float w_rstd_lo_1 = 1.0f;
                float w_rstd_hi_1 = 1.0f;
                if (w_kind < 2) {
                    float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_1, 1);
                    w_ss_lo_1 += _shfl_xor_sync_20;
                    float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_1, 2);
                    w_ss_lo_1 += _shfl_xor_sync_21;
                    float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_1, 1);
                    w_ss_hi_1 += _shfl_xor_sync_22;
                    float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_1, 2);
                    w_ss_hi_1 += _shfl_xor_sync_23;
                    int w_srow_lo_1 = warp_m * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n * 64 + w_srow_lo_1] = w_ss_lo_1;
                        rowstat[warp_n * 64 + w_srow_lo_1 + 8] = w_ss_hi_1;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float w_other_lo_1 = rowstat[(1 - warp_n) * 64 + w_srow_lo_1];
                    float w_other_hi_1 = rowstat[(1 - warp_n) * 64 + w_srow_lo_1 + 8];
                    float _fdiv_25 = __fdiv_rn(w_ss_lo_1 + w_other_lo_1, 128.0f);
                    float _rsqrt_24 = rsqrtf(_fdiv_25 + eps);
                    w_rstd_lo_1 = _rsqrt_24;
                    float _fdiv_27 = __fdiv_rn(w_ss_hi_1 + w_other_hi_1, 128.0f);
                    float _rsqrt_26 = rsqrtf(_fdiv_27 + eps);
                    w_rstd_hi_1 = _rsqrt_26;
                }
                int w_srow_st_1 = warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float w_normed_1_0[4];
                if (w_kind < 2) {
                    w_normed_1_0[0] = w_rounded_1[0] * w_rstd_lo_1 * nw[0];
                    w_normed_1_0[1] = w_rounded_1[1] * w_rstd_lo_1 * nw[1];
                    w_normed_1_0[2] = w_rounded_1[2] * w_rstd_hi_1 * nw[0];
                    w_normed_1_0[3] = w_rounded_1[3] * w_rstd_hi_1 * nw[1];
                } else {
                    w_normed_1_0[0] = w_rounded_1[0];
                    w_normed_1_0[1] = w_rounded_1[1];
                    w_normed_1_0[2] = w_rounded_1[2];
                    w_normed_1_0[3] = w_rounded_1[3];
                }
                unsigned int w_normed_1_0_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_0[_lp*2 + 0], w_normed_1_0[_lp*2+1 + 0]));
                    w_normed_1_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_28 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_1 * 256) + (unsigned int)(warp_n * 64 * 2 ^ (w_srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_0_bf16[1]))
                    : "memory");
                float w_normed_1_1[4];
                if (w_kind < 2) {
                    w_normed_1_1[0] = w_rounded_1[4] * w_rstd_lo_1 * nw[2];
                    w_normed_1_1[1] = w_rounded_1[5] * w_rstd_lo_1 * nw[3];
                    w_normed_1_1[2] = w_rounded_1[6] * w_rstd_hi_1 * nw[2];
                    w_normed_1_1[3] = w_rounded_1[7] * w_rstd_hi_1 * nw[3];
                } else {
                    w_normed_1_1[0] = w_rounded_1[4];
                    w_normed_1_1[1] = w_rounded_1[5];
                    w_normed_1_1[2] = w_rounded_1[6];
                    w_normed_1_1[3] = w_rounded_1[7];
                }
                unsigned int w_normed_1_1_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_1[_lp*2 + 0], w_normed_1_1[_lp*2+1 + 0]));
                    w_normed_1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_29 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_1 * 256) + (unsigned int)((warp_n * 64 + 8) * 2 ^ (w_srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_1_bf16[1]))
                    : "memory");
                float w_normed_1_2[4];
                if (w_kind < 2) {
                    w_normed_1_2[0] = w_rounded_1[8] * w_rstd_lo_1 * nw[4];
                    w_normed_1_2[1] = w_rounded_1[9] * w_rstd_lo_1 * nw[5];
                    w_normed_1_2[2] = w_rounded_1[10] * w_rstd_hi_1 * nw[4];
                    w_normed_1_2[3] = w_rounded_1[11] * w_rstd_hi_1 * nw[5];
                } else {
                    w_normed_1_2[0] = w_rounded_1[8];
                    w_normed_1_2[1] = w_rounded_1[9];
                    w_normed_1_2[2] = w_rounded_1[10];
                    w_normed_1_2[3] = w_rounded_1[11];
                }
                unsigned int w_normed_1_2_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_2[_lp*2 + 0], w_normed_1_2[_lp*2+1 + 0]));
                    w_normed_1_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_30 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_1 * 256) + (unsigned int)((warp_n * 64 + 16) * 2 ^ (w_srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_30), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_2_bf16[1]))
                    : "memory");
                float w_normed_1_3[4];
                if (w_kind < 2) {
                    w_normed_1_3[0] = w_rounded_1[12] * w_rstd_lo_1 * nw[6];
                    w_normed_1_3[1] = w_rounded_1[13] * w_rstd_lo_1 * nw[7];
                    w_normed_1_3[2] = w_rounded_1[14] * w_rstd_hi_1 * nw[6];
                    w_normed_1_3[3] = w_rounded_1[15] * w_rstd_hi_1 * nw[7];
                } else {
                    w_normed_1_3[0] = w_rounded_1[12];
                    w_normed_1_3[1] = w_rounded_1[13];
                    w_normed_1_3[2] = w_rounded_1[14];
                    w_normed_1_3[3] = w_rounded_1[15];
                }
                unsigned int w_normed_1_3_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_3[_lp*2 + 0], w_normed_1_3[_lp*2+1 + 0]));
                    w_normed_1_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_31 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_1 * 256) + (unsigned int)((warp_n * 64 + 24) * 2 ^ (w_srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_31), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_3_bf16[1]))
                    : "memory");
                float w_normed_1_4[4];
                if (w_kind < 2) {
                    w_normed_1_4[0] = w_rounded_1[16] * w_rstd_lo_1 * nw[8];
                    w_normed_1_4[1] = w_rounded_1[17] * w_rstd_lo_1 * nw[9];
                    w_normed_1_4[2] = w_rounded_1[18] * w_rstd_hi_1 * nw[8];
                    w_normed_1_4[3] = w_rounded_1[19] * w_rstd_hi_1 * nw[9];
                } else {
                    w_normed_1_4[0] = w_rounded_1[16];
                    w_normed_1_4[1] = w_rounded_1[17];
                    w_normed_1_4[2] = w_rounded_1[18];
                    w_normed_1_4[3] = w_rounded_1[19];
                }
                unsigned int w_normed_1_4_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_4[_lp*2 + 0], w_normed_1_4[_lp*2+1 + 0]));
                    w_normed_1_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_32 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_1 * 256) + (unsigned int)((warp_n * 64 + 32) * 2 ^ (w_srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_32), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_4_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_4_bf16[1]))
                    : "memory");
                float w_normed_1_5[4];
                if (w_kind < 2) {
                    w_normed_1_5[0] = w_rounded_1[20] * w_rstd_lo_1 * nw[10];
                    w_normed_1_5[1] = w_rounded_1[21] * w_rstd_lo_1 * nw[11];
                    w_normed_1_5[2] = w_rounded_1[22] * w_rstd_hi_1 * nw[10];
                    w_normed_1_5[3] = w_rounded_1[23] * w_rstd_hi_1 * nw[11];
                } else {
                    w_normed_1_5[0] = w_rounded_1[20];
                    w_normed_1_5[1] = w_rounded_1[21];
                    w_normed_1_5[2] = w_rounded_1[22];
                    w_normed_1_5[3] = w_rounded_1[23];
                }
                unsigned int w_normed_1_5_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_5[_lp*2 + 0], w_normed_1_5[_lp*2+1 + 0]));
                    w_normed_1_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_33 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_1 * 256) + (unsigned int)((warp_n * 64 + 40) * 2 ^ (w_srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_33), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_5_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_5_bf16[1]))
                    : "memory");
                float w_normed_1_6[4];
                if (w_kind < 2) {
                    w_normed_1_6[0] = w_rounded_1[24] * w_rstd_lo_1 * nw[12];
                    w_normed_1_6[1] = w_rounded_1[25] * w_rstd_lo_1 * nw[13];
                    w_normed_1_6[2] = w_rounded_1[26] * w_rstd_hi_1 * nw[12];
                    w_normed_1_6[3] = w_rounded_1[27] * w_rstd_hi_1 * nw[13];
                } else {
                    w_normed_1_6[0] = w_rounded_1[24];
                    w_normed_1_6[1] = w_rounded_1[25];
                    w_normed_1_6[2] = w_rounded_1[26];
                    w_normed_1_6[3] = w_rounded_1[27];
                }
                unsigned int w_normed_1_6_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_6[_lp*2 + 0], w_normed_1_6[_lp*2+1 + 0]));
                    w_normed_1_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_1 * 256) + (unsigned int)((warp_n * 64 + 48) * 2 ^ (w_srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_6_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_6_bf16[1]))
                    : "memory");
                float w_normed_1_7[4];
                if (w_kind < 2) {
                    w_normed_1_7[0] = w_rounded_1[28] * w_rstd_lo_1 * nw[14];
                    w_normed_1_7[1] = w_rounded_1[29] * w_rstd_lo_1 * nw[15];
                    w_normed_1_7[2] = w_rounded_1[30] * w_rstd_hi_1 * nw[14];
                    w_normed_1_7[3] = w_rounded_1[31] * w_rstd_hi_1 * nw[15];
                } else {
                    w_normed_1_7[0] = w_rounded_1[28];
                    w_normed_1_7[1] = w_rounded_1[29];
                    w_normed_1_7[2] = w_rounded_1[30];
                    w_normed_1_7[3] = w_rounded_1[31];
                }
                unsigned int w_normed_1_7_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_7[_lp*2 + 0], w_normed_1_7[_lp*2+1 + 0]));
                    w_normed_1_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_1 * 256) + (unsigned int)((warp_n * 64 + 56) * 2 ^ (w_srow_st_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_7_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_7_bf16[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_srow_1 = role_tid >> 2;
                int w_c_1 = role_tid & 3;
                int w_grow_1 = tile_m * 256 + 64 + w_srow_1;
                int w_chunk_lo_1 = ((w_c_1 < 3) ? w_c_1 * 16 : 96);
                int w_chunk_hi_1 = ((w_c_1 < 3) ? w_chunk_lo_1 + 48 : 112);
                if (w_grow_1 < M) {
                    unsigned int w_x_lo_1[8];
                    unsigned int w_x_hi_1[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1[(0) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_1 * 256) + (unsigned int)(w_chunk_lo_1 * 2 ^ (w_srow_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1[(4) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_1 * 256) + (unsigned int)(w_chunk_lo_1 * 2 + 16 ^ (w_srow_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1[(0) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_1 * 256) + (unsigned int)(w_chunk_hi_1 * 2 ^ (w_srow_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1[(4) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_1 * 256) + (unsigned int)(w_chunk_hi_1 * 2 + 16 ^ (w_srow_1 & 7) << 4)));
                    if (w_c_1 < 3) {
                        if (w_kind < 2) {
                            float _unpack_bf16_20[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_1[_pair]) << 16);
                                (&_unpack_bf16_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_1[_pair]) & 0xffff0000u);
                            }
                            float _unpack_bf16_21[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_1[_pair]) << 16);
                                (&_unpack_bf16_21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_1[_pair]) & 0xffff0000u);
                            }
                            float _vec_load_4[8];
                            {
                                const uint4* _vptr_36 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_1 * 96 + w_chunk_lo_1) + 0);
                                uint4 _vld_36[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_36[_blk] = _vptr_36[_blk];
                                    uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) << 16);
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_5[8];
                            {
                                const uint4* _vptr_37 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_1 * 96 + w_chunk_lo_1 + 8) + 0);
                                uint4 _vld_37[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_37[_blk] = _vptr_37[_blk];
                                    uint32_t* _vpairs_37 = reinterpret_cast<uint32_t*>(&_vld_37[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) << 16);
                                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_6[8];
                            {
                                const uint4* _vptr_38 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_1 * 96 + 48 + w_chunk_lo_1) + 0);
                                uint4 _vld_38[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_38[_blk] = _vptr_38[_blk];
                                    uint32_t* _vpairs_38 = reinterpret_cast<uint32_t*>(&_vld_38[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) << 16);
                                        (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_7[8];
                            {
                                const uint4* _vptr_39 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_1 * 96 + 48 + w_chunk_lo_1 + 8) + 0);
                                uint4 _vld_39[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_39[_blk] = _vptr_39[_blk];
                                    uint32_t* _vpairs_39 = reinterpret_cast<uint32_t*>(&_vld_39[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) << 16);
                                        (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float w_o_lo_1[16];
                            float w_o_hi_1[16];
                            float w_cv0_1_0 = _vec_load_4[0];
                            float w_sv0_1_0 = _vec_load_6[0];
                            float w_cv1_1_0 = _vec_load_5[0];
                            float w_sv1_1_0 = _vec_load_7[0];
                            w_o_lo_1[0] = __fmaf_rn(_unpack_bf16_20[0], w_cv0_1_0, -(_unpack_bf16_21[0] * w_sv0_1_0));
                            w_o_hi_1[0] = __fmaf_rn(_unpack_bf16_21[0], w_cv0_1_0, _unpack_bf16_20[0] * w_sv0_1_0);
                            w_o_lo_1[8] = __fmaf_rn(_unpack_bf16_20[8], w_cv1_1_0, -(_unpack_bf16_21[8] * w_sv1_1_0));
                            w_o_hi_1[8] = __fmaf_rn(_unpack_bf16_21[8], w_cv1_1_0, _unpack_bf16_20[8] * w_sv1_1_0);
                            float w_cv0_1_1 = _vec_load_4[1];
                            float w_sv0_1_1 = _vec_load_6[1];
                            float w_cv1_1_1 = _vec_load_5[1];
                            float w_sv1_1_1 = _vec_load_7[1];
                            w_o_lo_1[1] = __fmaf_rn(_unpack_bf16_20[1], w_cv0_1_1, -(_unpack_bf16_21[1] * w_sv0_1_1));
                            w_o_hi_1[1] = __fmaf_rn(_unpack_bf16_21[1], w_cv0_1_1, _unpack_bf16_20[1] * w_sv0_1_1);
                            w_o_lo_1[9] = __fmaf_rn(_unpack_bf16_20[9], w_cv1_1_1, -(_unpack_bf16_21[9] * w_sv1_1_1));
                            w_o_hi_1[9] = __fmaf_rn(_unpack_bf16_21[9], w_cv1_1_1, _unpack_bf16_20[9] * w_sv1_1_1);
                            float w_cv0_1_2 = _vec_load_4[2];
                            float w_sv0_1_2 = _vec_load_6[2];
                            float w_cv1_1_2 = _vec_load_5[2];
                            float w_sv1_1_2 = _vec_load_7[2];
                            w_o_lo_1[2] = __fmaf_rn(_unpack_bf16_20[2], w_cv0_1_2, -(_unpack_bf16_21[2] * w_sv0_1_2));
                            w_o_hi_1[2] = __fmaf_rn(_unpack_bf16_21[2], w_cv0_1_2, _unpack_bf16_20[2] * w_sv0_1_2);
                            w_o_lo_1[10] = __fmaf_rn(_unpack_bf16_20[10], w_cv1_1_2, -(_unpack_bf16_21[10] * w_sv1_1_2));
                            w_o_hi_1[10] = __fmaf_rn(_unpack_bf16_21[10], w_cv1_1_2, _unpack_bf16_20[10] * w_sv1_1_2);
                            float w_cv0_1_3 = _vec_load_4[3];
                            float w_sv0_1_3 = _vec_load_6[3];
                            float w_cv1_1_3 = _vec_load_5[3];
                            float w_sv1_1_3 = _vec_load_7[3];
                            w_o_lo_1[3] = __fmaf_rn(_unpack_bf16_20[3], w_cv0_1_3, -(_unpack_bf16_21[3] * w_sv0_1_3));
                            w_o_hi_1[3] = __fmaf_rn(_unpack_bf16_21[3], w_cv0_1_3, _unpack_bf16_20[3] * w_sv0_1_3);
                            w_o_lo_1[11] = __fmaf_rn(_unpack_bf16_20[11], w_cv1_1_3, -(_unpack_bf16_21[11] * w_sv1_1_3));
                            w_o_hi_1[11] = __fmaf_rn(_unpack_bf16_21[11], w_cv1_1_3, _unpack_bf16_20[11] * w_sv1_1_3);
                            float w_cv0_1_4 = _vec_load_4[4];
                            float w_sv0_1_4 = _vec_load_6[4];
                            float w_cv1_1_4 = _vec_load_5[4];
                            float w_sv1_1_4 = _vec_load_7[4];
                            w_o_lo_1[4] = __fmaf_rn(_unpack_bf16_20[4], w_cv0_1_4, -(_unpack_bf16_21[4] * w_sv0_1_4));
                            w_o_hi_1[4] = __fmaf_rn(_unpack_bf16_21[4], w_cv0_1_4, _unpack_bf16_20[4] * w_sv0_1_4);
                            w_o_lo_1[12] = __fmaf_rn(_unpack_bf16_20[12], w_cv1_1_4, -(_unpack_bf16_21[12] * w_sv1_1_4));
                            w_o_hi_1[12] = __fmaf_rn(_unpack_bf16_21[12], w_cv1_1_4, _unpack_bf16_20[12] * w_sv1_1_4);
                            float w_cv0_1_5 = _vec_load_4[5];
                            float w_sv0_1_5 = _vec_load_6[5];
                            float w_cv1_1_5 = _vec_load_5[5];
                            float w_sv1_1_5 = _vec_load_7[5];
                            w_o_lo_1[5] = __fmaf_rn(_unpack_bf16_20[5], w_cv0_1_5, -(_unpack_bf16_21[5] * w_sv0_1_5));
                            w_o_hi_1[5] = __fmaf_rn(_unpack_bf16_21[5], w_cv0_1_5, _unpack_bf16_20[5] * w_sv0_1_5);
                            w_o_lo_1[13] = __fmaf_rn(_unpack_bf16_20[13], w_cv1_1_5, -(_unpack_bf16_21[13] * w_sv1_1_5));
                            w_o_hi_1[13] = __fmaf_rn(_unpack_bf16_21[13], w_cv1_1_5, _unpack_bf16_20[13] * w_sv1_1_5);
                            float w_cv0_1_6 = _vec_load_4[6];
                            float w_sv0_1_6 = _vec_load_6[6];
                            float w_cv1_1_6 = _vec_load_5[6];
                            float w_sv1_1_6 = _vec_load_7[6];
                            w_o_lo_1[6] = __fmaf_rn(_unpack_bf16_20[6], w_cv0_1_6, -(_unpack_bf16_21[6] * w_sv0_1_6));
                            w_o_hi_1[6] = __fmaf_rn(_unpack_bf16_21[6], w_cv0_1_6, _unpack_bf16_20[6] * w_sv0_1_6);
                            w_o_lo_1[14] = __fmaf_rn(_unpack_bf16_20[14], w_cv1_1_6, -(_unpack_bf16_21[14] * w_sv1_1_6));
                            w_o_hi_1[14] = __fmaf_rn(_unpack_bf16_21[14], w_cv1_1_6, _unpack_bf16_20[14] * w_sv1_1_6);
                            float w_cv0_1_7 = _vec_load_4[7];
                            float w_sv0_1_7 = _vec_load_6[7];
                            float w_cv1_1_7 = _vec_load_5[7];
                            float w_sv1_1_7 = _vec_load_7[7];
                            w_o_lo_1[7] = __fmaf_rn(_unpack_bf16_20[7], w_cv0_1_7, -(_unpack_bf16_21[7] * w_sv0_1_7));
                            w_o_hi_1[7] = __fmaf_rn(_unpack_bf16_21[7], w_cv0_1_7, _unpack_bf16_20[7] * w_sv0_1_7);
                            w_o_lo_1[15] = __fmaf_rn(_unpack_bf16_20[15], w_cv1_1_7, -(_unpack_bf16_21[15] * w_sv1_1_7));
                            w_o_hi_1[15] = __fmaf_rn(_unpack_bf16_21[15], w_cv1_1_7, _unpack_bf16_20[15] * w_sv1_1_7);
                            unsigned int w_o_lo_1_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_lo_1[_lp*2 + 0], w_o_lo_1[_lp*2+1 + 0]));
                                w_o_lo_1_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int w_o_hi_1_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_hi_1[_lp*2 + 0], w_o_hi_1[_lp*2+1 + 0]));
                                w_o_hi_1_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            w_x_lo_1[0] = w_o_lo_1_bf16[0];
                            w_x_hi_1[0] = w_o_hi_1_bf16[0];
                            w_x_lo_1[1] = w_o_lo_1_bf16[1];
                            w_x_hi_1[1] = w_o_hi_1_bf16[1];
                            w_x_lo_1[2] = w_o_lo_1_bf16[2];
                            w_x_hi_1[2] = w_o_hi_1_bf16[2];
                            w_x_lo_1[3] = w_o_lo_1_bf16[3];
                            w_x_hi_1[3] = w_o_hi_1_bf16[3];
                            w_x_lo_1[4] = w_o_lo_1_bf16[4];
                            w_x_hi_1[4] = w_o_hi_1_bf16[4];
                            w_x_lo_1[5] = w_o_lo_1_bf16[5];
                            w_x_hi_1[5] = w_o_hi_1_bf16[5];
                            w_x_lo_1[6] = w_o_lo_1_bf16[6];
                            w_x_hi_1[6] = w_o_hi_1_bf16[6];
                            w_x_lo_1[7] = w_o_lo_1_bf16[7];
                            w_x_hi_1[7] = w_o_hi_1_bf16[7];
                        }
                    }
                    unsigned int w_q_words_1[4];
                    float w_q_vals_1[16];
                    float _unpack_bf16_22[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_22[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_1[_pair]) << 16);
                        (&_unpack_bf16_22[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_1[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_1[0] = _unpack_bf16_22[0] * w_out_scale_sel;
                    w_q_vals_1[1] = _unpack_bf16_22[1] * w_out_scale_sel;
                    w_q_vals_1[2] = _unpack_bf16_22[2] * w_out_scale_sel;
                    w_q_vals_1[3] = _unpack_bf16_22[3] * w_out_scale_sel;
                    w_q_vals_1[4] = _unpack_bf16_22[4] * w_out_scale_sel;
                    w_q_vals_1[5] = _unpack_bf16_22[5] * w_out_scale_sel;
                    w_q_vals_1[6] = _unpack_bf16_22[6] * w_out_scale_sel;
                    w_q_vals_1[7] = _unpack_bf16_22[7] * w_out_scale_sel;
                    w_q_vals_1[8] = _unpack_bf16_22[8] * w_out_scale_sel;
                    w_q_vals_1[9] = _unpack_bf16_22[9] * w_out_scale_sel;
                    w_q_vals_1[10] = _unpack_bf16_22[10] * w_out_scale_sel;
                    w_q_vals_1[11] = _unpack_bf16_22[11] * w_out_scale_sel;
                    w_q_vals_1[12] = _unpack_bf16_22[12] * w_out_scale_sel;
                    w_q_vals_1[13] = _unpack_bf16_22[13] * w_out_scale_sel;
                    w_q_vals_1[14] = _unpack_bf16_22[14] * w_out_scale_sel;
                    w_q_vals_1[15] = _unpack_bf16_22[15] * w_out_scale_sel;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_1[0]), "f"(w_q_vals_1[1]),
                                               "f"(w_q_vals_1[2]), "f"(w_q_vals_1[3]));
                        w_q_words_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_1[4]), "f"(w_q_vals_1[5]),
                                               "f"(w_q_vals_1[6]), "f"(w_q_vals_1[7]));
                        w_q_words_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_1[8]), "f"(w_q_vals_1[9]),
                                               "f"(w_q_vals_1[10]), "f"(w_q_vals_1[11]));
                        w_q_words_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_1[12]), "f"(w_q_vals_1[13]),
                                               "f"(w_q_vals_1[14]), "f"(w_q_vals_1[15]));
                        w_q_words_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(w_srow_1 * 128) + (unsigned int)(w_chunk_lo_1 ^ (w_srow_1 & 7) << 4)), "r"(w_q_words_1[0]), "r"(w_q_words_1[1]), "r"(w_q_words_1[2]), "r"(w_q_words_1[3]) : "memory");
                    float _unpack_bf16_23[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_23[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_1[_pair]) << 16);
                        (&_unpack_bf16_23[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_1[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_1[0] = _unpack_bf16_23[0] * w_out_scale_sel;
                    w_q_vals_1[1] = _unpack_bf16_23[1] * w_out_scale_sel;
                    w_q_vals_1[2] = _unpack_bf16_23[2] * w_out_scale_sel;
                    w_q_vals_1[3] = _unpack_bf16_23[3] * w_out_scale_sel;
                    w_q_vals_1[4] = _unpack_bf16_23[4] * w_out_scale_sel;
                    w_q_vals_1[5] = _unpack_bf16_23[5] * w_out_scale_sel;
                    w_q_vals_1[6] = _unpack_bf16_23[6] * w_out_scale_sel;
                    w_q_vals_1[7] = _unpack_bf16_23[7] * w_out_scale_sel;
                    w_q_vals_1[8] = _unpack_bf16_23[8] * w_out_scale_sel;
                    w_q_vals_1[9] = _unpack_bf16_23[9] * w_out_scale_sel;
                    w_q_vals_1[10] = _unpack_bf16_23[10] * w_out_scale_sel;
                    w_q_vals_1[11] = _unpack_bf16_23[11] * w_out_scale_sel;
                    w_q_vals_1[12] = _unpack_bf16_23[12] * w_out_scale_sel;
                    w_q_vals_1[13] = _unpack_bf16_23[13] * w_out_scale_sel;
                    w_q_vals_1[14] = _unpack_bf16_23[14] * w_out_scale_sel;
                    w_q_vals_1[15] = _unpack_bf16_23[15] * w_out_scale_sel;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_1[0]), "f"(w_q_vals_1[1]),
                                               "f"(w_q_vals_1[2]), "f"(w_q_vals_1[3]));
                        w_q_words_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_1[4]), "f"(w_q_vals_1[5]),
                                               "f"(w_q_vals_1[6]), "f"(w_q_vals_1[7]));
                        w_q_words_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_1[8]), "f"(w_q_vals_1[9]),
                                               "f"(w_q_vals_1[10]), "f"(w_q_vals_1[11]));
                        w_q_words_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_1[12]), "f"(w_q_vals_1[13]),
                                               "f"(w_q_vals_1[14]), "f"(w_q_vals_1[15]));
                        w_q_words_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(w_srow_1 * 128) + (unsigned int)(w_chunk_hi_1 ^ (w_srow_1 & 7) << 4)), "r"(w_q_words_1[0]), "r"(w_q_words_1[1]), "r"(w_q_words_1[2]), "r"(w_q_words_1[3]) : "memory");
                }
                __syncwarp();
                int s_row_1_0 = (role_tid >> 5) * 8 + lane / 8;
                int s_piece_1_0 = lane % 8;
                unsigned int s_words_1_0[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_0[(0) + 3]))
                    : "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(s_row_1_0 * 128) + (unsigned int)(s_piece_1_0 * 16 ^ (s_row_1_0 & 7) << 4)));
                int s_dst_1_0 = ((tile_m * 256 + 64 + s_row_1_0) * 56 + w_head) * 32 + s_piece_1_0 * 4;
                if (tile_m * 256 + 64 + s_row_1_0 < M) {
                    if (w_kind == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_1_0)[0] = reinterpret_cast<int4*>(s_words_1_0)[0];
                    } else if (w_kind == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_1_0)[0] = reinterpret_cast<int4*>(s_words_1_0)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_1_0)[0] = reinterpret_cast<int4*>(s_words_1_0)[0];
                    }
                }
                int s_row_1_1 = (role_tid >> 5) * 8 + 4 + lane / 8;
                int s_piece_1_1 = lane % 8;
                unsigned int s_words_1_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_1[(0) + 3]))
                    : "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(s_row_1_1 * 128) + (unsigned int)(s_piece_1_1 * 16 ^ (s_row_1_1 & 7) << 4)));
                int s_dst_1_1 = ((tile_m * 256 + 64 + s_row_1_1) * 56 + w_head) * 32 + s_piece_1_1 * 4;
                if (tile_m * 256 + 64 + s_row_1_1 < M) {
                    if (w_kind == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_1_1)[0] = reinterpret_cast<int4*>(s_words_1_1)[0];
                    } else if (w_kind == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_1_1)[0] = reinterpret_cast<int4*>(s_words_1_1)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_1_1)[0] = reinterpret_cast<int4*>(s_words_1_1)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_row_lo_2 = tile_m * 256 + 128 + warp_m * 16 + (lane >> 2);
                int w_row_hi_2 = w_row_lo_2 + 8;
                float w_sa_lo_2 = ((w_row_lo_2 < M) ? act_scale[w_row_lo_2] : 0.0f);
                float w_sa_hi_2 = ((w_row_hi_2 < M) ? act_scale[w_row_hi_2] : 0.0f);
                float w_rounded_2[32];
                float w_ss_lo_2 = 0.0f;
                float w_ss_hi_2 = 0.0f;
                float w_scaled_2_0[4];
                w_scaled_2_0[0] = accum[64] * w_sa_lo_2 * sw[0];
                w_scaled_2_0[1] = accum[65] * w_sa_lo_2 * sw[1];
                w_scaled_2_0[2] = accum[66] * w_sa_hi_2 * sw[0];
                w_scaled_2_0[3] = accum[67] * w_sa_hi_2 * sw[1];
                unsigned int w_scaled_2_0_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_0[_lp*2 + 0], w_scaled_2_0[_lp*2+1 + 0]));
                    w_scaled_2_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_24[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_24[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_24[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_0_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_2[0] = _unpack_bf16_24[0];
                w_rounded_2[1] = _unpack_bf16_24[1];
                w_rounded_2[2] = _unpack_bf16_24[2];
                w_rounded_2[3] = _unpack_bf16_24[3];
                w_ss_lo_2 += __fmaf_rn(_unpack_bf16_24[0], _unpack_bf16_24[0], _unpack_bf16_24[1] * _unpack_bf16_24[1]);
                w_ss_hi_2 += __fmaf_rn(_unpack_bf16_24[2], _unpack_bf16_24[2], _unpack_bf16_24[3] * _unpack_bf16_24[3]);
                float w_scaled_2_1[4];
                w_scaled_2_1[0] = accum[68] * w_sa_lo_2 * sw[2];
                w_scaled_2_1[1] = accum[69] * w_sa_lo_2 * sw[3];
                w_scaled_2_1[2] = accum[70] * w_sa_hi_2 * sw[2];
                w_scaled_2_1[3] = accum[71] * w_sa_hi_2 * sw[3];
                unsigned int w_scaled_2_1_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_1[_lp*2 + 0], w_scaled_2_1[_lp*2+1 + 0]));
                    w_scaled_2_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_25[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_25[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_25[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_1_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_2[4] = _unpack_bf16_25[0];
                w_rounded_2[5] = _unpack_bf16_25[1];
                w_rounded_2[6] = _unpack_bf16_25[2];
                w_rounded_2[7] = _unpack_bf16_25[3];
                w_ss_lo_2 += __fmaf_rn(_unpack_bf16_25[0], _unpack_bf16_25[0], _unpack_bf16_25[1] * _unpack_bf16_25[1]);
                w_ss_hi_2 += __fmaf_rn(_unpack_bf16_25[2], _unpack_bf16_25[2], _unpack_bf16_25[3] * _unpack_bf16_25[3]);
                float w_scaled_2_2[4];
                w_scaled_2_2[0] = accum[72] * w_sa_lo_2 * sw[4];
                w_scaled_2_2[1] = accum[73] * w_sa_lo_2 * sw[5];
                w_scaled_2_2[2] = accum[74] * w_sa_hi_2 * sw[4];
                w_scaled_2_2[3] = accum[75] * w_sa_hi_2 * sw[5];
                unsigned int w_scaled_2_2_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_2[_lp*2 + 0], w_scaled_2_2[_lp*2+1 + 0]));
                    w_scaled_2_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_26[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_26[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_2_bf16[_pair]) << 16);
                    (&_unpack_bf16_26[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_2_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_2[8] = _unpack_bf16_26[0];
                w_rounded_2[9] = _unpack_bf16_26[1];
                w_rounded_2[10] = _unpack_bf16_26[2];
                w_rounded_2[11] = _unpack_bf16_26[3];
                w_ss_lo_2 += __fmaf_rn(_unpack_bf16_26[0], _unpack_bf16_26[0], _unpack_bf16_26[1] * _unpack_bf16_26[1]);
                w_ss_hi_2 += __fmaf_rn(_unpack_bf16_26[2], _unpack_bf16_26[2], _unpack_bf16_26[3] * _unpack_bf16_26[3]);
                float w_scaled_2_3[4];
                w_scaled_2_3[0] = accum[76] * w_sa_lo_2 * sw[6];
                w_scaled_2_3[1] = accum[77] * w_sa_lo_2 * sw[7];
                w_scaled_2_3[2] = accum[78] * w_sa_hi_2 * sw[6];
                w_scaled_2_3[3] = accum[79] * w_sa_hi_2 * sw[7];
                unsigned int w_scaled_2_3_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_3[_lp*2 + 0], w_scaled_2_3[_lp*2+1 + 0]));
                    w_scaled_2_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_27[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_27[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_3_bf16[_pair]) << 16);
                    (&_unpack_bf16_27[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_3_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_2[12] = _unpack_bf16_27[0];
                w_rounded_2[13] = _unpack_bf16_27[1];
                w_rounded_2[14] = _unpack_bf16_27[2];
                w_rounded_2[15] = _unpack_bf16_27[3];
                w_ss_lo_2 += __fmaf_rn(_unpack_bf16_27[0], _unpack_bf16_27[0], _unpack_bf16_27[1] * _unpack_bf16_27[1]);
                w_ss_hi_2 += __fmaf_rn(_unpack_bf16_27[2], _unpack_bf16_27[2], _unpack_bf16_27[3] * _unpack_bf16_27[3]);
                float w_scaled_2_4[4];
                w_scaled_2_4[0] = accum[80] * w_sa_lo_2 * sw[8];
                w_scaled_2_4[1] = accum[81] * w_sa_lo_2 * sw[9];
                w_scaled_2_4[2] = accum[82] * w_sa_hi_2 * sw[8];
                w_scaled_2_4[3] = accum[83] * w_sa_hi_2 * sw[9];
                unsigned int w_scaled_2_4_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_4[_lp*2 + 0], w_scaled_2_4[_lp*2+1 + 0]));
                    w_scaled_2_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_28[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_28[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_4_bf16[_pair]) << 16);
                    (&_unpack_bf16_28[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_4_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_2[16] = _unpack_bf16_28[0];
                w_rounded_2[17] = _unpack_bf16_28[1];
                w_rounded_2[18] = _unpack_bf16_28[2];
                w_rounded_2[19] = _unpack_bf16_28[3];
                w_ss_lo_2 += __fmaf_rn(_unpack_bf16_28[0], _unpack_bf16_28[0], _unpack_bf16_28[1] * _unpack_bf16_28[1]);
                w_ss_hi_2 += __fmaf_rn(_unpack_bf16_28[2], _unpack_bf16_28[2], _unpack_bf16_28[3] * _unpack_bf16_28[3]);
                float w_scaled_2_5[4];
                w_scaled_2_5[0] = accum[84] * w_sa_lo_2 * sw[10];
                w_scaled_2_5[1] = accum[85] * w_sa_lo_2 * sw[11];
                w_scaled_2_5[2] = accum[86] * w_sa_hi_2 * sw[10];
                w_scaled_2_5[3] = accum[87] * w_sa_hi_2 * sw[11];
                unsigned int w_scaled_2_5_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_5[_lp*2 + 0], w_scaled_2_5[_lp*2+1 + 0]));
                    w_scaled_2_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_29[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_29[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_5_bf16[_pair]) << 16);
                    (&_unpack_bf16_29[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_5_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_2[20] = _unpack_bf16_29[0];
                w_rounded_2[21] = _unpack_bf16_29[1];
                w_rounded_2[22] = _unpack_bf16_29[2];
                w_rounded_2[23] = _unpack_bf16_29[3];
                w_ss_lo_2 += __fmaf_rn(_unpack_bf16_29[0], _unpack_bf16_29[0], _unpack_bf16_29[1] * _unpack_bf16_29[1]);
                w_ss_hi_2 += __fmaf_rn(_unpack_bf16_29[2], _unpack_bf16_29[2], _unpack_bf16_29[3] * _unpack_bf16_29[3]);
                float w_scaled_2_6[4];
                w_scaled_2_6[0] = accum[88] * w_sa_lo_2 * sw[12];
                w_scaled_2_6[1] = accum[89] * w_sa_lo_2 * sw[13];
                w_scaled_2_6[2] = accum[90] * w_sa_hi_2 * sw[12];
                w_scaled_2_6[3] = accum[91] * w_sa_hi_2 * sw[13];
                unsigned int w_scaled_2_6_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_6[_lp*2 + 0], w_scaled_2_6[_lp*2+1 + 0]));
                    w_scaled_2_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_30[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_30[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_6_bf16[_pair]) << 16);
                    (&_unpack_bf16_30[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_6_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_2[24] = _unpack_bf16_30[0];
                w_rounded_2[25] = _unpack_bf16_30[1];
                w_rounded_2[26] = _unpack_bf16_30[2];
                w_rounded_2[27] = _unpack_bf16_30[3];
                w_ss_lo_2 += __fmaf_rn(_unpack_bf16_30[0], _unpack_bf16_30[0], _unpack_bf16_30[1] * _unpack_bf16_30[1]);
                w_ss_hi_2 += __fmaf_rn(_unpack_bf16_30[2], _unpack_bf16_30[2], _unpack_bf16_30[3] * _unpack_bf16_30[3]);
                float w_scaled_2_7[4];
                w_scaled_2_7[0] = accum[92] * w_sa_lo_2 * sw[14];
                w_scaled_2_7[1] = accum[93] * w_sa_lo_2 * sw[15];
                w_scaled_2_7[2] = accum[94] * w_sa_hi_2 * sw[14];
                w_scaled_2_7[3] = accum[95] * w_sa_hi_2 * sw[15];
                unsigned int w_scaled_2_7_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_7[_lp*2 + 0], w_scaled_2_7[_lp*2+1 + 0]));
                    w_scaled_2_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_31[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_31[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_7_bf16[_pair]) << 16);
                    (&_unpack_bf16_31[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_7_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_2[28] = _unpack_bf16_31[0];
                w_rounded_2[29] = _unpack_bf16_31[1];
                w_rounded_2[30] = _unpack_bf16_31[2];
                w_rounded_2[31] = _unpack_bf16_31[3];
                w_ss_lo_2 += __fmaf_rn(_unpack_bf16_31[0], _unpack_bf16_31[0], _unpack_bf16_31[1] * _unpack_bf16_31[1]);
                w_ss_hi_2 += __fmaf_rn(_unpack_bf16_31[2], _unpack_bf16_31[2], _unpack_bf16_31[3] * _unpack_bf16_31[3]);
                float w_rstd_lo_2 = 1.0f;
                float w_rstd_hi_2 = 1.0f;
                if (w_kind < 2) {
                    float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_2, 1);
                    w_ss_lo_2 += _shfl_xor_sync_40;
                    float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_2, 2);
                    w_ss_lo_2 += _shfl_xor_sync_41;
                    float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_2, 1);
                    w_ss_hi_2 += _shfl_xor_sync_42;
                    float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_2, 2);
                    w_ss_hi_2 += _shfl_xor_sync_43;
                    int w_srow_lo_2 = warp_m * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n * 64 + w_srow_lo_2] = w_ss_lo_2;
                        rowstat[warp_n * 64 + w_srow_lo_2 + 8] = w_ss_hi_2;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float w_other_lo_2 = rowstat[(1 - warp_n) * 64 + w_srow_lo_2];
                    float w_other_hi_2 = rowstat[(1 - warp_n) * 64 + w_srow_lo_2 + 8];
                    float _fdiv_45 = __fdiv_rn(w_ss_lo_2 + w_other_lo_2, 128.0f);
                    float _rsqrt_44 = rsqrtf(_fdiv_45 + eps);
                    w_rstd_lo_2 = _rsqrt_44;
                    float _fdiv_47 = __fdiv_rn(w_ss_hi_2 + w_other_hi_2, 128.0f);
                    float _rsqrt_46 = rsqrtf(_fdiv_47 + eps);
                    w_rstd_hi_2 = _rsqrt_46;
                }
                int w_srow_st_2 = warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float w_normed_2_0[4];
                if (w_kind < 2) {
                    w_normed_2_0[0] = w_rounded_2[0] * w_rstd_lo_2 * nw[0];
                    w_normed_2_0[1] = w_rounded_2[1] * w_rstd_lo_2 * nw[1];
                    w_normed_2_0[2] = w_rounded_2[2] * w_rstd_hi_2 * nw[0];
                    w_normed_2_0[3] = w_rounded_2[3] * w_rstd_hi_2 * nw[1];
                } else {
                    w_normed_2_0[0] = w_rounded_2[0];
                    w_normed_2_0[1] = w_rounded_2[1];
                    w_normed_2_0[2] = w_rounded_2[2];
                    w_normed_2_0[3] = w_rounded_2[3];
                }
                unsigned int w_normed_2_0_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_0[_lp*2 + 0], w_normed_2_0[_lp*2+1 + 0]));
                    w_normed_2_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_48 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_2 * 256) + (unsigned int)(warp_n * 64 * 2 ^ (w_srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_48), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_0_bf16[1]))
                    : "memory");
                float w_normed_2_1[4];
                if (w_kind < 2) {
                    w_normed_2_1[0] = w_rounded_2[4] * w_rstd_lo_2 * nw[2];
                    w_normed_2_1[1] = w_rounded_2[5] * w_rstd_lo_2 * nw[3];
                    w_normed_2_1[2] = w_rounded_2[6] * w_rstd_hi_2 * nw[2];
                    w_normed_2_1[3] = w_rounded_2[7] * w_rstd_hi_2 * nw[3];
                } else {
                    w_normed_2_1[0] = w_rounded_2[4];
                    w_normed_2_1[1] = w_rounded_2[5];
                    w_normed_2_1[2] = w_rounded_2[6];
                    w_normed_2_1[3] = w_rounded_2[7];
                }
                unsigned int w_normed_2_1_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_1[_lp*2 + 0], w_normed_2_1[_lp*2+1 + 0]));
                    w_normed_2_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_49 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_2 * 256) + (unsigned int)((warp_n * 64 + 8) * 2 ^ (w_srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_49), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_1_bf16[1]))
                    : "memory");
                float w_normed_2_2[4];
                if (w_kind < 2) {
                    w_normed_2_2[0] = w_rounded_2[8] * w_rstd_lo_2 * nw[4];
                    w_normed_2_2[1] = w_rounded_2[9] * w_rstd_lo_2 * nw[5];
                    w_normed_2_2[2] = w_rounded_2[10] * w_rstd_hi_2 * nw[4];
                    w_normed_2_2[3] = w_rounded_2[11] * w_rstd_hi_2 * nw[5];
                } else {
                    w_normed_2_2[0] = w_rounded_2[8];
                    w_normed_2_2[1] = w_rounded_2[9];
                    w_normed_2_2[2] = w_rounded_2[10];
                    w_normed_2_2[3] = w_rounded_2[11];
                }
                unsigned int w_normed_2_2_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_2[_lp*2 + 0], w_normed_2_2[_lp*2+1 + 0]));
                    w_normed_2_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_50 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_2 * 256) + (unsigned int)((warp_n * 64 + 16) * 2 ^ (w_srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_50), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_2_bf16[1]))
                    : "memory");
                float w_normed_2_3[4];
                if (w_kind < 2) {
                    w_normed_2_3[0] = w_rounded_2[12] * w_rstd_lo_2 * nw[6];
                    w_normed_2_3[1] = w_rounded_2[13] * w_rstd_lo_2 * nw[7];
                    w_normed_2_3[2] = w_rounded_2[14] * w_rstd_hi_2 * nw[6];
                    w_normed_2_3[3] = w_rounded_2[15] * w_rstd_hi_2 * nw[7];
                } else {
                    w_normed_2_3[0] = w_rounded_2[12];
                    w_normed_2_3[1] = w_rounded_2[13];
                    w_normed_2_3[2] = w_rounded_2[14];
                    w_normed_2_3[3] = w_rounded_2[15];
                }
                unsigned int w_normed_2_3_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_3[_lp*2 + 0], w_normed_2_3[_lp*2+1 + 0]));
                    w_normed_2_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_51 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_2 * 256) + (unsigned int)((warp_n * 64 + 24) * 2 ^ (w_srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_51), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_3_bf16[1]))
                    : "memory");
                float w_normed_2_4[4];
                if (w_kind < 2) {
                    w_normed_2_4[0] = w_rounded_2[16] * w_rstd_lo_2 * nw[8];
                    w_normed_2_4[1] = w_rounded_2[17] * w_rstd_lo_2 * nw[9];
                    w_normed_2_4[2] = w_rounded_2[18] * w_rstd_hi_2 * nw[8];
                    w_normed_2_4[3] = w_rounded_2[19] * w_rstd_hi_2 * nw[9];
                } else {
                    w_normed_2_4[0] = w_rounded_2[16];
                    w_normed_2_4[1] = w_rounded_2[17];
                    w_normed_2_4[2] = w_rounded_2[18];
                    w_normed_2_4[3] = w_rounded_2[19];
                }
                unsigned int w_normed_2_4_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_4[_lp*2 + 0], w_normed_2_4[_lp*2+1 + 0]));
                    w_normed_2_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_52 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_2 * 256) + (unsigned int)((warp_n * 64 + 32) * 2 ^ (w_srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_52), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_4_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_4_bf16[1]))
                    : "memory");
                float w_normed_2_5[4];
                if (w_kind < 2) {
                    w_normed_2_5[0] = w_rounded_2[20] * w_rstd_lo_2 * nw[10];
                    w_normed_2_5[1] = w_rounded_2[21] * w_rstd_lo_2 * nw[11];
                    w_normed_2_5[2] = w_rounded_2[22] * w_rstd_hi_2 * nw[10];
                    w_normed_2_5[3] = w_rounded_2[23] * w_rstd_hi_2 * nw[11];
                } else {
                    w_normed_2_5[0] = w_rounded_2[20];
                    w_normed_2_5[1] = w_rounded_2[21];
                    w_normed_2_5[2] = w_rounded_2[22];
                    w_normed_2_5[3] = w_rounded_2[23];
                }
                unsigned int w_normed_2_5_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_5[_lp*2 + 0], w_normed_2_5[_lp*2+1 + 0]));
                    w_normed_2_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_53 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_2 * 256) + (unsigned int)((warp_n * 64 + 40) * 2 ^ (w_srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_53), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_5_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_5_bf16[1]))
                    : "memory");
                float w_normed_2_6[4];
                if (w_kind < 2) {
                    w_normed_2_6[0] = w_rounded_2[24] * w_rstd_lo_2 * nw[12];
                    w_normed_2_6[1] = w_rounded_2[25] * w_rstd_lo_2 * nw[13];
                    w_normed_2_6[2] = w_rounded_2[26] * w_rstd_hi_2 * nw[12];
                    w_normed_2_6[3] = w_rounded_2[27] * w_rstd_hi_2 * nw[13];
                } else {
                    w_normed_2_6[0] = w_rounded_2[24];
                    w_normed_2_6[1] = w_rounded_2[25];
                    w_normed_2_6[2] = w_rounded_2[26];
                    w_normed_2_6[3] = w_rounded_2[27];
                }
                unsigned int w_normed_2_6_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_6[_lp*2 + 0], w_normed_2_6[_lp*2+1 + 0]));
                    w_normed_2_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_54 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_2 * 256) + (unsigned int)((warp_n * 64 + 48) * 2 ^ (w_srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_54), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_6_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_6_bf16[1]))
                    : "memory");
                float w_normed_2_7[4];
                if (w_kind < 2) {
                    w_normed_2_7[0] = w_rounded_2[28] * w_rstd_lo_2 * nw[14];
                    w_normed_2_7[1] = w_rounded_2[29] * w_rstd_lo_2 * nw[15];
                    w_normed_2_7[2] = w_rounded_2[30] * w_rstd_hi_2 * nw[14];
                    w_normed_2_7[3] = w_rounded_2[31] * w_rstd_hi_2 * nw[15];
                } else {
                    w_normed_2_7[0] = w_rounded_2[28];
                    w_normed_2_7[1] = w_rounded_2[29];
                    w_normed_2_7[2] = w_rounded_2[30];
                    w_normed_2_7[3] = w_rounded_2[31];
                }
                unsigned int w_normed_2_7_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_7[_lp*2 + 0], w_normed_2_7[_lp*2+1 + 0]));
                    w_normed_2_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_55 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_2 * 256) + (unsigned int)((warp_n * 64 + 56) * 2 ^ (w_srow_st_2 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_55), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_7_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_7_bf16[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_srow_2 = role_tid >> 2;
                int w_c_2 = role_tid & 3;
                int w_grow_2 = tile_m * 256 + 128 + w_srow_2;
                int w_chunk_lo_2 = ((w_c_2 < 3) ? w_c_2 * 16 : 96);
                int w_chunk_hi_2 = ((w_c_2 < 3) ? w_chunk_lo_2 + 48 : 112);
                if (w_grow_2 < M) {
                    unsigned int w_x_lo_2[8];
                    unsigned int w_x_hi_2[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2[(0) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_2 * 256) + (unsigned int)(w_chunk_lo_2 * 2 ^ (w_srow_2 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2[(4) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_2 * 256) + (unsigned int)(w_chunk_lo_2 * 2 + 16 ^ (w_srow_2 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2[(0) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_2 * 256) + (unsigned int)(w_chunk_hi_2 * 2 ^ (w_srow_2 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2[(4) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_2 * 256) + (unsigned int)(w_chunk_hi_2 * 2 + 16 ^ (w_srow_2 & 7) << 4)));
                    if (w_c_2 < 3) {
                        if (w_kind < 2) {
                            float _unpack_bf16_32[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_2[_pair]) << 16);
                                (&_unpack_bf16_32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_2[_pair]) & 0xffff0000u);
                            }
                            float _unpack_bf16_33[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_33[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_2[_pair]) << 16);
                                (&_unpack_bf16_33[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_2[_pair]) & 0xffff0000u);
                            }
                            float _vec_load_8[8];
                            {
                                const uint4* _vptr_56 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_2 * 96 + w_chunk_lo_2) + 0);
                                uint4 _vld_56[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_56[_blk] = _vptr_56[_blk];
                                    uint32_t* _vpairs_56 = reinterpret_cast<uint32_t*>(&_vld_56[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) << 16);
                                        (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_9[8];
                            {
                                const uint4* _vptr_57 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_2 * 96 + w_chunk_lo_2 + 8) + 0);
                                uint4 _vld_57[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_57[_blk] = _vptr_57[_blk];
                                    uint32_t* _vpairs_57 = reinterpret_cast<uint32_t*>(&_vld_57[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) << 16);
                                        (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_10[8];
                            {
                                const uint4* _vptr_58 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_2 * 96 + 48 + w_chunk_lo_2) + 0);
                                uint4 _vld_58[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_58[_blk] = _vptr_58[_blk];
                                    uint32_t* _vpairs_58 = reinterpret_cast<uint32_t*>(&_vld_58[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) << 16);
                                        (&_vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_11[8];
                            {
                                const uint4* _vptr_59 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_2 * 96 + 48 + w_chunk_lo_2 + 8) + 0);
                                uint4 _vld_59[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_59[_blk] = _vptr_59[_blk];
                                    uint32_t* _vpairs_59 = reinterpret_cast<uint32_t*>(&_vld_59[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) << 16);
                                        (&_vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float w_o_lo_2[16];
                            float w_o_hi_2[16];
                            float w_cv0_2_0 = _vec_load_8[0];
                            float w_sv0_2_0 = _vec_load_10[0];
                            float w_cv1_2_0 = _vec_load_9[0];
                            float w_sv1_2_0 = _vec_load_11[0];
                            w_o_lo_2[0] = __fmaf_rn(_unpack_bf16_32[0], w_cv0_2_0, -(_unpack_bf16_33[0] * w_sv0_2_0));
                            w_o_hi_2[0] = __fmaf_rn(_unpack_bf16_33[0], w_cv0_2_0, _unpack_bf16_32[0] * w_sv0_2_0);
                            w_o_lo_2[8] = __fmaf_rn(_unpack_bf16_32[8], w_cv1_2_0, -(_unpack_bf16_33[8] * w_sv1_2_0));
                            w_o_hi_2[8] = __fmaf_rn(_unpack_bf16_33[8], w_cv1_2_0, _unpack_bf16_32[8] * w_sv1_2_0);
                            float w_cv0_2_1 = _vec_load_8[1];
                            float w_sv0_2_1 = _vec_load_10[1];
                            float w_cv1_2_1 = _vec_load_9[1];
                            float w_sv1_2_1 = _vec_load_11[1];
                            w_o_lo_2[1] = __fmaf_rn(_unpack_bf16_32[1], w_cv0_2_1, -(_unpack_bf16_33[1] * w_sv0_2_1));
                            w_o_hi_2[1] = __fmaf_rn(_unpack_bf16_33[1], w_cv0_2_1, _unpack_bf16_32[1] * w_sv0_2_1);
                            w_o_lo_2[9] = __fmaf_rn(_unpack_bf16_32[9], w_cv1_2_1, -(_unpack_bf16_33[9] * w_sv1_2_1));
                            w_o_hi_2[9] = __fmaf_rn(_unpack_bf16_33[9], w_cv1_2_1, _unpack_bf16_32[9] * w_sv1_2_1);
                            float w_cv0_2_2 = _vec_load_8[2];
                            float w_sv0_2_2 = _vec_load_10[2];
                            float w_cv1_2_2 = _vec_load_9[2];
                            float w_sv1_2_2 = _vec_load_11[2];
                            w_o_lo_2[2] = __fmaf_rn(_unpack_bf16_32[2], w_cv0_2_2, -(_unpack_bf16_33[2] * w_sv0_2_2));
                            w_o_hi_2[2] = __fmaf_rn(_unpack_bf16_33[2], w_cv0_2_2, _unpack_bf16_32[2] * w_sv0_2_2);
                            w_o_lo_2[10] = __fmaf_rn(_unpack_bf16_32[10], w_cv1_2_2, -(_unpack_bf16_33[10] * w_sv1_2_2));
                            w_o_hi_2[10] = __fmaf_rn(_unpack_bf16_33[10], w_cv1_2_2, _unpack_bf16_32[10] * w_sv1_2_2);
                            float w_cv0_2_3 = _vec_load_8[3];
                            float w_sv0_2_3 = _vec_load_10[3];
                            float w_cv1_2_3 = _vec_load_9[3];
                            float w_sv1_2_3 = _vec_load_11[3];
                            w_o_lo_2[3] = __fmaf_rn(_unpack_bf16_32[3], w_cv0_2_3, -(_unpack_bf16_33[3] * w_sv0_2_3));
                            w_o_hi_2[3] = __fmaf_rn(_unpack_bf16_33[3], w_cv0_2_3, _unpack_bf16_32[3] * w_sv0_2_3);
                            w_o_lo_2[11] = __fmaf_rn(_unpack_bf16_32[11], w_cv1_2_3, -(_unpack_bf16_33[11] * w_sv1_2_3));
                            w_o_hi_2[11] = __fmaf_rn(_unpack_bf16_33[11], w_cv1_2_3, _unpack_bf16_32[11] * w_sv1_2_3);
                            float w_cv0_2_4 = _vec_load_8[4];
                            float w_sv0_2_4 = _vec_load_10[4];
                            float w_cv1_2_4 = _vec_load_9[4];
                            float w_sv1_2_4 = _vec_load_11[4];
                            w_o_lo_2[4] = __fmaf_rn(_unpack_bf16_32[4], w_cv0_2_4, -(_unpack_bf16_33[4] * w_sv0_2_4));
                            w_o_hi_2[4] = __fmaf_rn(_unpack_bf16_33[4], w_cv0_2_4, _unpack_bf16_32[4] * w_sv0_2_4);
                            w_o_lo_2[12] = __fmaf_rn(_unpack_bf16_32[12], w_cv1_2_4, -(_unpack_bf16_33[12] * w_sv1_2_4));
                            w_o_hi_2[12] = __fmaf_rn(_unpack_bf16_33[12], w_cv1_2_4, _unpack_bf16_32[12] * w_sv1_2_4);
                            float w_cv0_2_5 = _vec_load_8[5];
                            float w_sv0_2_5 = _vec_load_10[5];
                            float w_cv1_2_5 = _vec_load_9[5];
                            float w_sv1_2_5 = _vec_load_11[5];
                            w_o_lo_2[5] = __fmaf_rn(_unpack_bf16_32[5], w_cv0_2_5, -(_unpack_bf16_33[5] * w_sv0_2_5));
                            w_o_hi_2[5] = __fmaf_rn(_unpack_bf16_33[5], w_cv0_2_5, _unpack_bf16_32[5] * w_sv0_2_5);
                            w_o_lo_2[13] = __fmaf_rn(_unpack_bf16_32[13], w_cv1_2_5, -(_unpack_bf16_33[13] * w_sv1_2_5));
                            w_o_hi_2[13] = __fmaf_rn(_unpack_bf16_33[13], w_cv1_2_5, _unpack_bf16_32[13] * w_sv1_2_5);
                            float w_cv0_2_6 = _vec_load_8[6];
                            float w_sv0_2_6 = _vec_load_10[6];
                            float w_cv1_2_6 = _vec_load_9[6];
                            float w_sv1_2_6 = _vec_load_11[6];
                            w_o_lo_2[6] = __fmaf_rn(_unpack_bf16_32[6], w_cv0_2_6, -(_unpack_bf16_33[6] * w_sv0_2_6));
                            w_o_hi_2[6] = __fmaf_rn(_unpack_bf16_33[6], w_cv0_2_6, _unpack_bf16_32[6] * w_sv0_2_6);
                            w_o_lo_2[14] = __fmaf_rn(_unpack_bf16_32[14], w_cv1_2_6, -(_unpack_bf16_33[14] * w_sv1_2_6));
                            w_o_hi_2[14] = __fmaf_rn(_unpack_bf16_33[14], w_cv1_2_6, _unpack_bf16_32[14] * w_sv1_2_6);
                            float w_cv0_2_7 = _vec_load_8[7];
                            float w_sv0_2_7 = _vec_load_10[7];
                            float w_cv1_2_7 = _vec_load_9[7];
                            float w_sv1_2_7 = _vec_load_11[7];
                            w_o_lo_2[7] = __fmaf_rn(_unpack_bf16_32[7], w_cv0_2_7, -(_unpack_bf16_33[7] * w_sv0_2_7));
                            w_o_hi_2[7] = __fmaf_rn(_unpack_bf16_33[7], w_cv0_2_7, _unpack_bf16_32[7] * w_sv0_2_7);
                            w_o_lo_2[15] = __fmaf_rn(_unpack_bf16_32[15], w_cv1_2_7, -(_unpack_bf16_33[15] * w_sv1_2_7));
                            w_o_hi_2[15] = __fmaf_rn(_unpack_bf16_33[15], w_cv1_2_7, _unpack_bf16_32[15] * w_sv1_2_7);
                            unsigned int w_o_lo_2_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_lo_2[_lp*2 + 0], w_o_lo_2[_lp*2+1 + 0]));
                                w_o_lo_2_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int w_o_hi_2_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_hi_2[_lp*2 + 0], w_o_hi_2[_lp*2+1 + 0]));
                                w_o_hi_2_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            w_x_lo_2[0] = w_o_lo_2_bf16[0];
                            w_x_hi_2[0] = w_o_hi_2_bf16[0];
                            w_x_lo_2[1] = w_o_lo_2_bf16[1];
                            w_x_hi_2[1] = w_o_hi_2_bf16[1];
                            w_x_lo_2[2] = w_o_lo_2_bf16[2];
                            w_x_hi_2[2] = w_o_hi_2_bf16[2];
                            w_x_lo_2[3] = w_o_lo_2_bf16[3];
                            w_x_hi_2[3] = w_o_hi_2_bf16[3];
                            w_x_lo_2[4] = w_o_lo_2_bf16[4];
                            w_x_hi_2[4] = w_o_hi_2_bf16[4];
                            w_x_lo_2[5] = w_o_lo_2_bf16[5];
                            w_x_hi_2[5] = w_o_hi_2_bf16[5];
                            w_x_lo_2[6] = w_o_lo_2_bf16[6];
                            w_x_hi_2[6] = w_o_hi_2_bf16[6];
                            w_x_lo_2[7] = w_o_lo_2_bf16[7];
                            w_x_hi_2[7] = w_o_hi_2_bf16[7];
                        }
                    }
                    unsigned int w_q_words_2[4];
                    float w_q_vals_2[16];
                    float _unpack_bf16_34[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_34[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_2[_pair]) << 16);
                        (&_unpack_bf16_34[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_2[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_2[0] = _unpack_bf16_34[0] * w_out_scale_sel;
                    w_q_vals_2[1] = _unpack_bf16_34[1] * w_out_scale_sel;
                    w_q_vals_2[2] = _unpack_bf16_34[2] * w_out_scale_sel;
                    w_q_vals_2[3] = _unpack_bf16_34[3] * w_out_scale_sel;
                    w_q_vals_2[4] = _unpack_bf16_34[4] * w_out_scale_sel;
                    w_q_vals_2[5] = _unpack_bf16_34[5] * w_out_scale_sel;
                    w_q_vals_2[6] = _unpack_bf16_34[6] * w_out_scale_sel;
                    w_q_vals_2[7] = _unpack_bf16_34[7] * w_out_scale_sel;
                    w_q_vals_2[8] = _unpack_bf16_34[8] * w_out_scale_sel;
                    w_q_vals_2[9] = _unpack_bf16_34[9] * w_out_scale_sel;
                    w_q_vals_2[10] = _unpack_bf16_34[10] * w_out_scale_sel;
                    w_q_vals_2[11] = _unpack_bf16_34[11] * w_out_scale_sel;
                    w_q_vals_2[12] = _unpack_bf16_34[12] * w_out_scale_sel;
                    w_q_vals_2[13] = _unpack_bf16_34[13] * w_out_scale_sel;
                    w_q_vals_2[14] = _unpack_bf16_34[14] * w_out_scale_sel;
                    w_q_vals_2[15] = _unpack_bf16_34[15] * w_out_scale_sel;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_2[0]), "f"(w_q_vals_2[1]),
                                               "f"(w_q_vals_2[2]), "f"(w_q_vals_2[3]));
                        w_q_words_2[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_2[4]), "f"(w_q_vals_2[5]),
                                               "f"(w_q_vals_2[6]), "f"(w_q_vals_2[7]));
                        w_q_words_2[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_2[8]), "f"(w_q_vals_2[9]),
                                               "f"(w_q_vals_2[10]), "f"(w_q_vals_2[11]));
                        w_q_words_2[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_2[12]), "f"(w_q_vals_2[13]),
                                               "f"(w_q_vals_2[14]), "f"(w_q_vals_2[15]));
                        w_q_words_2[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(w_srow_2 * 128) + (unsigned int)(w_chunk_lo_2 ^ (w_srow_2 & 7) << 4)), "r"(w_q_words_2[0]), "r"(w_q_words_2[1]), "r"(w_q_words_2[2]), "r"(w_q_words_2[3]) : "memory");
                    float _unpack_bf16_35[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_35[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_2[_pair]) << 16);
                        (&_unpack_bf16_35[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_2[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_2[0] = _unpack_bf16_35[0] * w_out_scale_sel;
                    w_q_vals_2[1] = _unpack_bf16_35[1] * w_out_scale_sel;
                    w_q_vals_2[2] = _unpack_bf16_35[2] * w_out_scale_sel;
                    w_q_vals_2[3] = _unpack_bf16_35[3] * w_out_scale_sel;
                    w_q_vals_2[4] = _unpack_bf16_35[4] * w_out_scale_sel;
                    w_q_vals_2[5] = _unpack_bf16_35[5] * w_out_scale_sel;
                    w_q_vals_2[6] = _unpack_bf16_35[6] * w_out_scale_sel;
                    w_q_vals_2[7] = _unpack_bf16_35[7] * w_out_scale_sel;
                    w_q_vals_2[8] = _unpack_bf16_35[8] * w_out_scale_sel;
                    w_q_vals_2[9] = _unpack_bf16_35[9] * w_out_scale_sel;
                    w_q_vals_2[10] = _unpack_bf16_35[10] * w_out_scale_sel;
                    w_q_vals_2[11] = _unpack_bf16_35[11] * w_out_scale_sel;
                    w_q_vals_2[12] = _unpack_bf16_35[12] * w_out_scale_sel;
                    w_q_vals_2[13] = _unpack_bf16_35[13] * w_out_scale_sel;
                    w_q_vals_2[14] = _unpack_bf16_35[14] * w_out_scale_sel;
                    w_q_vals_2[15] = _unpack_bf16_35[15] * w_out_scale_sel;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_2[0]), "f"(w_q_vals_2[1]),
                                               "f"(w_q_vals_2[2]), "f"(w_q_vals_2[3]));
                        w_q_words_2[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_2[4]), "f"(w_q_vals_2[5]),
                                               "f"(w_q_vals_2[6]), "f"(w_q_vals_2[7]));
                        w_q_words_2[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_2[8]), "f"(w_q_vals_2[9]),
                                               "f"(w_q_vals_2[10]), "f"(w_q_vals_2[11]));
                        w_q_words_2[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_2[12]), "f"(w_q_vals_2[13]),
                                               "f"(w_q_vals_2[14]), "f"(w_q_vals_2[15]));
                        w_q_words_2[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(w_srow_2 * 128) + (unsigned int)(w_chunk_hi_2 ^ (w_srow_2 & 7) << 4)), "r"(w_q_words_2[0]), "r"(w_q_words_2[1]), "r"(w_q_words_2[2]), "r"(w_q_words_2[3]) : "memory");
                }
                __syncwarp();
                int s_row_2_0 = (role_tid >> 5) * 8 + lane / 8;
                int s_piece_2_0 = lane % 8;
                unsigned int s_words_2_0[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_0[(0) + 3]))
                    : "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(s_row_2_0 * 128) + (unsigned int)(s_piece_2_0 * 16 ^ (s_row_2_0 & 7) << 4)));
                int s_dst_2_0 = ((tile_m * 256 + 128 + s_row_2_0) * 56 + w_head) * 32 + s_piece_2_0 * 4;
                if (tile_m * 256 + 128 + s_row_2_0 < M) {
                    if (w_kind == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_2_0)[0] = reinterpret_cast<int4*>(s_words_2_0)[0];
                    } else if (w_kind == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_2_0)[0] = reinterpret_cast<int4*>(s_words_2_0)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_2_0)[0] = reinterpret_cast<int4*>(s_words_2_0)[0];
                    }
                }
                int s_row_2_1 = (role_tid >> 5) * 8 + 4 + lane / 8;
                int s_piece_2_1 = lane % 8;
                unsigned int s_words_2_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_1[(0) + 3]))
                    : "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(s_row_2_1 * 128) + (unsigned int)(s_piece_2_1 * 16 ^ (s_row_2_1 & 7) << 4)));
                int s_dst_2_1 = ((tile_m * 256 + 128 + s_row_2_1) * 56 + w_head) * 32 + s_piece_2_1 * 4;
                if (tile_m * 256 + 128 + s_row_2_1 < M) {
                    if (w_kind == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_2_1)[0] = reinterpret_cast<int4*>(s_words_2_1)[0];
                    } else if (w_kind == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_2_1)[0] = reinterpret_cast<int4*>(s_words_2_1)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_2_1)[0] = reinterpret_cast<int4*>(s_words_2_1)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_row_lo_3 = tile_m * 256 + 192 + warp_m * 16 + (lane >> 2);
                int w_row_hi_3 = w_row_lo_3 + 8;
                float w_sa_lo_3 = ((w_row_lo_3 < M) ? act_scale[w_row_lo_3] : 0.0f);
                float w_sa_hi_3 = ((w_row_hi_3 < M) ? act_scale[w_row_hi_3] : 0.0f);
                float w_rounded_3[32];
                float w_ss_lo_3 = 0.0f;
                float w_ss_hi_3 = 0.0f;
                float w_scaled_3_0[4];
                w_scaled_3_0[0] = accum[96] * w_sa_lo_3 * sw[0];
                w_scaled_3_0[1] = accum[97] * w_sa_lo_3 * sw[1];
                w_scaled_3_0[2] = accum[98] * w_sa_hi_3 * sw[0];
                w_scaled_3_0[3] = accum[99] * w_sa_hi_3 * sw[1];
                unsigned int w_scaled_3_0_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_0[_lp*2 + 0], w_scaled_3_0[_lp*2+1 + 0]));
                    w_scaled_3_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_36[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_36[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_36[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_0_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_3[0] = _unpack_bf16_36[0];
                w_rounded_3[1] = _unpack_bf16_36[1];
                w_rounded_3[2] = _unpack_bf16_36[2];
                w_rounded_3[3] = _unpack_bf16_36[3];
                w_ss_lo_3 += __fmaf_rn(_unpack_bf16_36[0], _unpack_bf16_36[0], _unpack_bf16_36[1] * _unpack_bf16_36[1]);
                w_ss_hi_3 += __fmaf_rn(_unpack_bf16_36[2], _unpack_bf16_36[2], _unpack_bf16_36[3] * _unpack_bf16_36[3]);
                float w_scaled_3_1[4];
                w_scaled_3_1[0] = accum[100] * w_sa_lo_3 * sw[2];
                w_scaled_3_1[1] = accum[101] * w_sa_lo_3 * sw[3];
                w_scaled_3_1[2] = accum[102] * w_sa_hi_3 * sw[2];
                w_scaled_3_1[3] = accum[103] * w_sa_hi_3 * sw[3];
                unsigned int w_scaled_3_1_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_1[_lp*2 + 0], w_scaled_3_1[_lp*2+1 + 0]));
                    w_scaled_3_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_37[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_37[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_37[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_1_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_3[4] = _unpack_bf16_37[0];
                w_rounded_3[5] = _unpack_bf16_37[1];
                w_rounded_3[6] = _unpack_bf16_37[2];
                w_rounded_3[7] = _unpack_bf16_37[3];
                w_ss_lo_3 += __fmaf_rn(_unpack_bf16_37[0], _unpack_bf16_37[0], _unpack_bf16_37[1] * _unpack_bf16_37[1]);
                w_ss_hi_3 += __fmaf_rn(_unpack_bf16_37[2], _unpack_bf16_37[2], _unpack_bf16_37[3] * _unpack_bf16_37[3]);
                float w_scaled_3_2[4];
                w_scaled_3_2[0] = accum[104] * w_sa_lo_3 * sw[4];
                w_scaled_3_2[1] = accum[105] * w_sa_lo_3 * sw[5];
                w_scaled_3_2[2] = accum[106] * w_sa_hi_3 * sw[4];
                w_scaled_3_2[3] = accum[107] * w_sa_hi_3 * sw[5];
                unsigned int w_scaled_3_2_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_2[_lp*2 + 0], w_scaled_3_2[_lp*2+1 + 0]));
                    w_scaled_3_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_38[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_38[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_2_bf16[_pair]) << 16);
                    (&_unpack_bf16_38[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_2_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_3[8] = _unpack_bf16_38[0];
                w_rounded_3[9] = _unpack_bf16_38[1];
                w_rounded_3[10] = _unpack_bf16_38[2];
                w_rounded_3[11] = _unpack_bf16_38[3];
                w_ss_lo_3 += __fmaf_rn(_unpack_bf16_38[0], _unpack_bf16_38[0], _unpack_bf16_38[1] * _unpack_bf16_38[1]);
                w_ss_hi_3 += __fmaf_rn(_unpack_bf16_38[2], _unpack_bf16_38[2], _unpack_bf16_38[3] * _unpack_bf16_38[3]);
                float w_scaled_3_3[4];
                w_scaled_3_3[0] = accum[108] * w_sa_lo_3 * sw[6];
                w_scaled_3_3[1] = accum[109] * w_sa_lo_3 * sw[7];
                w_scaled_3_3[2] = accum[110] * w_sa_hi_3 * sw[6];
                w_scaled_3_3[3] = accum[111] * w_sa_hi_3 * sw[7];
                unsigned int w_scaled_3_3_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_3[_lp*2 + 0], w_scaled_3_3[_lp*2+1 + 0]));
                    w_scaled_3_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_39[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_39[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_3_bf16[_pair]) << 16);
                    (&_unpack_bf16_39[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_3_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_3[12] = _unpack_bf16_39[0];
                w_rounded_3[13] = _unpack_bf16_39[1];
                w_rounded_3[14] = _unpack_bf16_39[2];
                w_rounded_3[15] = _unpack_bf16_39[3];
                w_ss_lo_3 += __fmaf_rn(_unpack_bf16_39[0], _unpack_bf16_39[0], _unpack_bf16_39[1] * _unpack_bf16_39[1]);
                w_ss_hi_3 += __fmaf_rn(_unpack_bf16_39[2], _unpack_bf16_39[2], _unpack_bf16_39[3] * _unpack_bf16_39[3]);
                float w_scaled_3_4[4];
                w_scaled_3_4[0] = accum[112] * w_sa_lo_3 * sw[8];
                w_scaled_3_4[1] = accum[113] * w_sa_lo_3 * sw[9];
                w_scaled_3_4[2] = accum[114] * w_sa_hi_3 * sw[8];
                w_scaled_3_4[3] = accum[115] * w_sa_hi_3 * sw[9];
                unsigned int w_scaled_3_4_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_4[_lp*2 + 0], w_scaled_3_4[_lp*2+1 + 0]));
                    w_scaled_3_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_40[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_40[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_4_bf16[_pair]) << 16);
                    (&_unpack_bf16_40[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_4_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_3[16] = _unpack_bf16_40[0];
                w_rounded_3[17] = _unpack_bf16_40[1];
                w_rounded_3[18] = _unpack_bf16_40[2];
                w_rounded_3[19] = _unpack_bf16_40[3];
                w_ss_lo_3 += __fmaf_rn(_unpack_bf16_40[0], _unpack_bf16_40[0], _unpack_bf16_40[1] * _unpack_bf16_40[1]);
                w_ss_hi_3 += __fmaf_rn(_unpack_bf16_40[2], _unpack_bf16_40[2], _unpack_bf16_40[3] * _unpack_bf16_40[3]);
                float w_scaled_3_5[4];
                w_scaled_3_5[0] = accum[116] * w_sa_lo_3 * sw[10];
                w_scaled_3_5[1] = accum[117] * w_sa_lo_3 * sw[11];
                w_scaled_3_5[2] = accum[118] * w_sa_hi_3 * sw[10];
                w_scaled_3_5[3] = accum[119] * w_sa_hi_3 * sw[11];
                unsigned int w_scaled_3_5_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_5[_lp*2 + 0], w_scaled_3_5[_lp*2+1 + 0]));
                    w_scaled_3_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_41[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_41[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_5_bf16[_pair]) << 16);
                    (&_unpack_bf16_41[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_5_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_3[20] = _unpack_bf16_41[0];
                w_rounded_3[21] = _unpack_bf16_41[1];
                w_rounded_3[22] = _unpack_bf16_41[2];
                w_rounded_3[23] = _unpack_bf16_41[3];
                w_ss_lo_3 += __fmaf_rn(_unpack_bf16_41[0], _unpack_bf16_41[0], _unpack_bf16_41[1] * _unpack_bf16_41[1]);
                w_ss_hi_3 += __fmaf_rn(_unpack_bf16_41[2], _unpack_bf16_41[2], _unpack_bf16_41[3] * _unpack_bf16_41[3]);
                float w_scaled_3_6[4];
                w_scaled_3_6[0] = accum[120] * w_sa_lo_3 * sw[12];
                w_scaled_3_6[1] = accum[121] * w_sa_lo_3 * sw[13];
                w_scaled_3_6[2] = accum[122] * w_sa_hi_3 * sw[12];
                w_scaled_3_6[3] = accum[123] * w_sa_hi_3 * sw[13];
                unsigned int w_scaled_3_6_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_6[_lp*2 + 0], w_scaled_3_6[_lp*2+1 + 0]));
                    w_scaled_3_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_42[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_42[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_6_bf16[_pair]) << 16);
                    (&_unpack_bf16_42[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_6_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_3[24] = _unpack_bf16_42[0];
                w_rounded_3[25] = _unpack_bf16_42[1];
                w_rounded_3[26] = _unpack_bf16_42[2];
                w_rounded_3[27] = _unpack_bf16_42[3];
                w_ss_lo_3 += __fmaf_rn(_unpack_bf16_42[0], _unpack_bf16_42[0], _unpack_bf16_42[1] * _unpack_bf16_42[1]);
                w_ss_hi_3 += __fmaf_rn(_unpack_bf16_42[2], _unpack_bf16_42[2], _unpack_bf16_42[3] * _unpack_bf16_42[3]);
                float w_scaled_3_7[4];
                w_scaled_3_7[0] = accum[124] * w_sa_lo_3 * sw[14];
                w_scaled_3_7[1] = accum[125] * w_sa_lo_3 * sw[15];
                w_scaled_3_7[2] = accum[126] * w_sa_hi_3 * sw[14];
                w_scaled_3_7[3] = accum[127] * w_sa_hi_3 * sw[15];
                unsigned int w_scaled_3_7_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_7[_lp*2 + 0], w_scaled_3_7[_lp*2+1 + 0]));
                    w_scaled_3_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_43[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_43[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_7_bf16[_pair]) << 16);
                    (&_unpack_bf16_43[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_7_bf16[_pair]) & 0xffff0000u);
                }
                w_rounded_3[28] = _unpack_bf16_43[0];
                w_rounded_3[29] = _unpack_bf16_43[1];
                w_rounded_3[30] = _unpack_bf16_43[2];
                w_rounded_3[31] = _unpack_bf16_43[3];
                w_ss_lo_3 += __fmaf_rn(_unpack_bf16_43[0], _unpack_bf16_43[0], _unpack_bf16_43[1] * _unpack_bf16_43[1]);
                w_ss_hi_3 += __fmaf_rn(_unpack_bf16_43[2], _unpack_bf16_43[2], _unpack_bf16_43[3] * _unpack_bf16_43[3]);
                float w_rstd_lo_3 = 1.0f;
                float w_rstd_hi_3 = 1.0f;
                if (w_kind < 2) {
                    float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_3, 1);
                    w_ss_lo_3 += _shfl_xor_sync_60;
                    float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_3, 2);
                    w_ss_lo_3 += _shfl_xor_sync_61;
                    float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_3, 1);
                    w_ss_hi_3 += _shfl_xor_sync_62;
                    float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_3, 2);
                    w_ss_hi_3 += _shfl_xor_sync_63;
                    int w_srow_lo_3 = warp_m * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n * 64 + w_srow_lo_3] = w_ss_lo_3;
                        rowstat[warp_n * 64 + w_srow_lo_3 + 8] = w_ss_hi_3;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float w_other_lo_3 = rowstat[(1 - warp_n) * 64 + w_srow_lo_3];
                    float w_other_hi_3 = rowstat[(1 - warp_n) * 64 + w_srow_lo_3 + 8];
                    float _fdiv_65 = __fdiv_rn(w_ss_lo_3 + w_other_lo_3, 128.0f);
                    float _rsqrt_64 = rsqrtf(_fdiv_65 + eps);
                    w_rstd_lo_3 = _rsqrt_64;
                    float _fdiv_67 = __fdiv_rn(w_ss_hi_3 + w_other_hi_3, 128.0f);
                    float _rsqrt_66 = rsqrtf(_fdiv_67 + eps);
                    w_rstd_hi_3 = _rsqrt_66;
                }
                int w_srow_st_3 = warp_m * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float w_normed_3_0[4];
                if (w_kind < 2) {
                    w_normed_3_0[0] = w_rounded_3[0] * w_rstd_lo_3 * nw[0];
                    w_normed_3_0[1] = w_rounded_3[1] * w_rstd_lo_3 * nw[1];
                    w_normed_3_0[2] = w_rounded_3[2] * w_rstd_hi_3 * nw[0];
                    w_normed_3_0[3] = w_rounded_3[3] * w_rstd_hi_3 * nw[1];
                } else {
                    w_normed_3_0[0] = w_rounded_3[0];
                    w_normed_3_0[1] = w_rounded_3[1];
                    w_normed_3_0[2] = w_rounded_3[2];
                    w_normed_3_0[3] = w_rounded_3[3];
                }
                unsigned int w_normed_3_0_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_0[_lp*2 + 0], w_normed_3_0[_lp*2+1 + 0]));
                    w_normed_3_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_68 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_3 * 256) + (unsigned int)(warp_n * 64 * 2 ^ (w_srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_68), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_0_bf16[1]))
                    : "memory");
                float w_normed_3_1[4];
                if (w_kind < 2) {
                    w_normed_3_1[0] = w_rounded_3[4] * w_rstd_lo_3 * nw[2];
                    w_normed_3_1[1] = w_rounded_3[5] * w_rstd_lo_3 * nw[3];
                    w_normed_3_1[2] = w_rounded_3[6] * w_rstd_hi_3 * nw[2];
                    w_normed_3_1[3] = w_rounded_3[7] * w_rstd_hi_3 * nw[3];
                } else {
                    w_normed_3_1[0] = w_rounded_3[4];
                    w_normed_3_1[1] = w_rounded_3[5];
                    w_normed_3_1[2] = w_rounded_3[6];
                    w_normed_3_1[3] = w_rounded_3[7];
                }
                unsigned int w_normed_3_1_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_1[_lp*2 + 0], w_normed_3_1[_lp*2+1 + 0]));
                    w_normed_3_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_69 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_3 * 256) + (unsigned int)((warp_n * 64 + 8) * 2 ^ (w_srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_69), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_1_bf16[1]))
                    : "memory");
                float w_normed_3_2[4];
                if (w_kind < 2) {
                    w_normed_3_2[0] = w_rounded_3[8] * w_rstd_lo_3 * nw[4];
                    w_normed_3_2[1] = w_rounded_3[9] * w_rstd_lo_3 * nw[5];
                    w_normed_3_2[2] = w_rounded_3[10] * w_rstd_hi_3 * nw[4];
                    w_normed_3_2[3] = w_rounded_3[11] * w_rstd_hi_3 * nw[5];
                } else {
                    w_normed_3_2[0] = w_rounded_3[8];
                    w_normed_3_2[1] = w_rounded_3[9];
                    w_normed_3_2[2] = w_rounded_3[10];
                    w_normed_3_2[3] = w_rounded_3[11];
                }
                unsigned int w_normed_3_2_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_2[_lp*2 + 0], w_normed_3_2[_lp*2+1 + 0]));
                    w_normed_3_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_70 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_3 * 256) + (unsigned int)((warp_n * 64 + 16) * 2 ^ (w_srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_70), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_2_bf16[1]))
                    : "memory");
                float w_normed_3_3[4];
                if (w_kind < 2) {
                    w_normed_3_3[0] = w_rounded_3[12] * w_rstd_lo_3 * nw[6];
                    w_normed_3_3[1] = w_rounded_3[13] * w_rstd_lo_3 * nw[7];
                    w_normed_3_3[2] = w_rounded_3[14] * w_rstd_hi_3 * nw[6];
                    w_normed_3_3[3] = w_rounded_3[15] * w_rstd_hi_3 * nw[7];
                } else {
                    w_normed_3_3[0] = w_rounded_3[12];
                    w_normed_3_3[1] = w_rounded_3[13];
                    w_normed_3_3[2] = w_rounded_3[14];
                    w_normed_3_3[3] = w_rounded_3[15];
                }
                unsigned int w_normed_3_3_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_3[_lp*2 + 0], w_normed_3_3[_lp*2+1 + 0]));
                    w_normed_3_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_71 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_3 * 256) + (unsigned int)((warp_n * 64 + 24) * 2 ^ (w_srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_71), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_3_bf16[1]))
                    : "memory");
                float w_normed_3_4[4];
                if (w_kind < 2) {
                    w_normed_3_4[0] = w_rounded_3[16] * w_rstd_lo_3 * nw[8];
                    w_normed_3_4[1] = w_rounded_3[17] * w_rstd_lo_3 * nw[9];
                    w_normed_3_4[2] = w_rounded_3[18] * w_rstd_hi_3 * nw[8];
                    w_normed_3_4[3] = w_rounded_3[19] * w_rstd_hi_3 * nw[9];
                } else {
                    w_normed_3_4[0] = w_rounded_3[16];
                    w_normed_3_4[1] = w_rounded_3[17];
                    w_normed_3_4[2] = w_rounded_3[18];
                    w_normed_3_4[3] = w_rounded_3[19];
                }
                unsigned int w_normed_3_4_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_4[_lp*2 + 0], w_normed_3_4[_lp*2+1 + 0]));
                    w_normed_3_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_72 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_3 * 256) + (unsigned int)((warp_n * 64 + 32) * 2 ^ (w_srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_72), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_4_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_4_bf16[1]))
                    : "memory");
                float w_normed_3_5[4];
                if (w_kind < 2) {
                    w_normed_3_5[0] = w_rounded_3[20] * w_rstd_lo_3 * nw[10];
                    w_normed_3_5[1] = w_rounded_3[21] * w_rstd_lo_3 * nw[11];
                    w_normed_3_5[2] = w_rounded_3[22] * w_rstd_hi_3 * nw[10];
                    w_normed_3_5[3] = w_rounded_3[23] * w_rstd_hi_3 * nw[11];
                } else {
                    w_normed_3_5[0] = w_rounded_3[20];
                    w_normed_3_5[1] = w_rounded_3[21];
                    w_normed_3_5[2] = w_rounded_3[22];
                    w_normed_3_5[3] = w_rounded_3[23];
                }
                unsigned int w_normed_3_5_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_5[_lp*2 + 0], w_normed_3_5[_lp*2+1 + 0]));
                    w_normed_3_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_73 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_3 * 256) + (unsigned int)((warp_n * 64 + 40) * 2 ^ (w_srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_73), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_5_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_5_bf16[1]))
                    : "memory");
                float w_normed_3_6[4];
                if (w_kind < 2) {
                    w_normed_3_6[0] = w_rounded_3[24] * w_rstd_lo_3 * nw[12];
                    w_normed_3_6[1] = w_rounded_3[25] * w_rstd_lo_3 * nw[13];
                    w_normed_3_6[2] = w_rounded_3[26] * w_rstd_hi_3 * nw[12];
                    w_normed_3_6[3] = w_rounded_3[27] * w_rstd_hi_3 * nw[13];
                } else {
                    w_normed_3_6[0] = w_rounded_3[24];
                    w_normed_3_6[1] = w_rounded_3[25];
                    w_normed_3_6[2] = w_rounded_3[26];
                    w_normed_3_6[3] = w_rounded_3[27];
                }
                unsigned int w_normed_3_6_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_6[_lp*2 + 0], w_normed_3_6[_lp*2+1 + 0]));
                    w_normed_3_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_74 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_3 * 256) + (unsigned int)((warp_n * 64 + 48) * 2 ^ (w_srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_74), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_6_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_6_bf16[1]))
                    : "memory");
                float w_normed_3_7[4];
                if (w_kind < 2) {
                    w_normed_3_7[0] = w_rounded_3[28] * w_rstd_lo_3 * nw[14];
                    w_normed_3_7[1] = w_rounded_3[29] * w_rstd_lo_3 * nw[15];
                    w_normed_3_7[2] = w_rounded_3[30] * w_rstd_hi_3 * nw[14];
                    w_normed_3_7[3] = w_rounded_3[31] * w_rstd_hi_3 * nw[15];
                } else {
                    w_normed_3_7[0] = w_rounded_3[28];
                    w_normed_3_7[1] = w_rounded_3[29];
                    w_normed_3_7[2] = w_rounded_3[30];
                    w_normed_3_7[3] = w_rounded_3[31];
                }
                unsigned int w_normed_3_7_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_7[_lp*2 + 0], w_normed_3_7[_lp*2+1 + 0]));
                    w_normed_3_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_75 = static_cast<uint32_t>(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_st_3 * 256) + (unsigned int)((warp_n * 64 + 56) * 2 ^ (w_srow_st_3 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_75), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_7_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_7_bf16[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_srow_3 = role_tid >> 2;
                int w_c_3 = role_tid & 3;
                int w_grow_3 = tile_m * 256 + 192 + w_srow_3;
                int w_chunk_lo_3 = ((w_c_3 < 3) ? w_c_3 * 16 : 96);
                int w_chunk_hi_3 = ((w_c_3 < 3) ? w_chunk_lo_3 + 48 : 112);
                if (w_grow_3 < M) {
                    unsigned int w_x_lo_3[8];
                    unsigned int w_x_hi_3[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3[(0) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_3 * 256) + (unsigned int)(w_chunk_lo_3 * 2 ^ (w_srow_3 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3[(4) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_3 * 256) + (unsigned int)(w_chunk_lo_3 * 2 + 16 ^ (w_srow_3 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3[(0) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_3 * 256) + (unsigned int)(w_chunk_hi_3 * 2 ^ (w_srow_3 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3[(4) + 3]))
                        : "r"(A_stage_addr + free_stage * 16384 + (unsigned int)(w_srow_3 * 256) + (unsigned int)(w_chunk_hi_3 * 2 + 16 ^ (w_srow_3 & 7) << 4)));
                    if (w_c_3 < 3) {
                        if (w_kind < 2) {
                            float _unpack_bf16_44[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_44[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_3[_pair]) << 16);
                                (&_unpack_bf16_44[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_3[_pair]) & 0xffff0000u);
                            }
                            float _unpack_bf16_45[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_45[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_3[_pair]) << 16);
                                (&_unpack_bf16_45[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_3[_pair]) & 0xffff0000u);
                            }
                            float _vec_load_12[8];
                            {
                                const uint4* _vptr_76 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_3 * 96 + w_chunk_lo_3) + 0);
                                uint4 _vld_76[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_76[_blk] = _vptr_76[_blk];
                                    uint32_t* _vpairs_76 = reinterpret_cast<uint32_t*>(&_vld_76[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_12[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_76[_pair]) << 16);
                                        (&_vec_load_12[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_76[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_13[8];
                            {
                                const uint4* _vptr_77 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_3 * 96 + w_chunk_lo_3 + 8) + 0);
                                uint4 _vld_77[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_77[_blk] = _vptr_77[_blk];
                                    uint32_t* _vpairs_77 = reinterpret_cast<uint32_t*>(&_vld_77[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_13[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_77[_pair]) << 16);
                                        (&_vec_load_13[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_77[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_14[8];
                            {
                                const uint4* _vptr_78 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_3 * 96 + 48 + w_chunk_lo_3) + 0);
                                uint4 _vld_78[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_78[_blk] = _vptr_78[_blk];
                                    uint32_t* _vpairs_78 = reinterpret_cast<uint32_t*>(&_vld_78[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_14[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) << 16);
                                        (&_vec_load_14[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_15[8];
                            {
                                const uint4* _vptr_79 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_3 * 96 + 48 + w_chunk_lo_3 + 8) + 0);
                                uint4 _vld_79[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_79[_blk] = _vptr_79[_blk];
                                    uint32_t* _vpairs_79 = reinterpret_cast<uint32_t*>(&_vld_79[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_15[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_79[_pair]) << 16);
                                        (&_vec_load_15[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_79[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float w_o_lo_3[16];
                            float w_o_hi_3[16];
                            float w_cv0_3_0 = _vec_load_12[0];
                            float w_sv0_3_0 = _vec_load_14[0];
                            float w_cv1_3_0 = _vec_load_13[0];
                            float w_sv1_3_0 = _vec_load_15[0];
                            w_o_lo_3[0] = __fmaf_rn(_unpack_bf16_44[0], w_cv0_3_0, -(_unpack_bf16_45[0] * w_sv0_3_0));
                            w_o_hi_3[0] = __fmaf_rn(_unpack_bf16_45[0], w_cv0_3_0, _unpack_bf16_44[0] * w_sv0_3_0);
                            w_o_lo_3[8] = __fmaf_rn(_unpack_bf16_44[8], w_cv1_3_0, -(_unpack_bf16_45[8] * w_sv1_3_0));
                            w_o_hi_3[8] = __fmaf_rn(_unpack_bf16_45[8], w_cv1_3_0, _unpack_bf16_44[8] * w_sv1_3_0);
                            float w_cv0_3_1 = _vec_load_12[1];
                            float w_sv0_3_1 = _vec_load_14[1];
                            float w_cv1_3_1 = _vec_load_13[1];
                            float w_sv1_3_1 = _vec_load_15[1];
                            w_o_lo_3[1] = __fmaf_rn(_unpack_bf16_44[1], w_cv0_3_1, -(_unpack_bf16_45[1] * w_sv0_3_1));
                            w_o_hi_3[1] = __fmaf_rn(_unpack_bf16_45[1], w_cv0_3_1, _unpack_bf16_44[1] * w_sv0_3_1);
                            w_o_lo_3[9] = __fmaf_rn(_unpack_bf16_44[9], w_cv1_3_1, -(_unpack_bf16_45[9] * w_sv1_3_1));
                            w_o_hi_3[9] = __fmaf_rn(_unpack_bf16_45[9], w_cv1_3_1, _unpack_bf16_44[9] * w_sv1_3_1);
                            float w_cv0_3_2 = _vec_load_12[2];
                            float w_sv0_3_2 = _vec_load_14[2];
                            float w_cv1_3_2 = _vec_load_13[2];
                            float w_sv1_3_2 = _vec_load_15[2];
                            w_o_lo_3[2] = __fmaf_rn(_unpack_bf16_44[2], w_cv0_3_2, -(_unpack_bf16_45[2] * w_sv0_3_2));
                            w_o_hi_3[2] = __fmaf_rn(_unpack_bf16_45[2], w_cv0_3_2, _unpack_bf16_44[2] * w_sv0_3_2);
                            w_o_lo_3[10] = __fmaf_rn(_unpack_bf16_44[10], w_cv1_3_2, -(_unpack_bf16_45[10] * w_sv1_3_2));
                            w_o_hi_3[10] = __fmaf_rn(_unpack_bf16_45[10], w_cv1_3_2, _unpack_bf16_44[10] * w_sv1_3_2);
                            float w_cv0_3_3 = _vec_load_12[3];
                            float w_sv0_3_3 = _vec_load_14[3];
                            float w_cv1_3_3 = _vec_load_13[3];
                            float w_sv1_3_3 = _vec_load_15[3];
                            w_o_lo_3[3] = __fmaf_rn(_unpack_bf16_44[3], w_cv0_3_3, -(_unpack_bf16_45[3] * w_sv0_3_3));
                            w_o_hi_3[3] = __fmaf_rn(_unpack_bf16_45[3], w_cv0_3_3, _unpack_bf16_44[3] * w_sv0_3_3);
                            w_o_lo_3[11] = __fmaf_rn(_unpack_bf16_44[11], w_cv1_3_3, -(_unpack_bf16_45[11] * w_sv1_3_3));
                            w_o_hi_3[11] = __fmaf_rn(_unpack_bf16_45[11], w_cv1_3_3, _unpack_bf16_44[11] * w_sv1_3_3);
                            float w_cv0_3_4 = _vec_load_12[4];
                            float w_sv0_3_4 = _vec_load_14[4];
                            float w_cv1_3_4 = _vec_load_13[4];
                            float w_sv1_3_4 = _vec_load_15[4];
                            w_o_lo_3[4] = __fmaf_rn(_unpack_bf16_44[4], w_cv0_3_4, -(_unpack_bf16_45[4] * w_sv0_3_4));
                            w_o_hi_3[4] = __fmaf_rn(_unpack_bf16_45[4], w_cv0_3_4, _unpack_bf16_44[4] * w_sv0_3_4);
                            w_o_lo_3[12] = __fmaf_rn(_unpack_bf16_44[12], w_cv1_3_4, -(_unpack_bf16_45[12] * w_sv1_3_4));
                            w_o_hi_3[12] = __fmaf_rn(_unpack_bf16_45[12], w_cv1_3_4, _unpack_bf16_44[12] * w_sv1_3_4);
                            float w_cv0_3_5 = _vec_load_12[5];
                            float w_sv0_3_5 = _vec_load_14[5];
                            float w_cv1_3_5 = _vec_load_13[5];
                            float w_sv1_3_5 = _vec_load_15[5];
                            w_o_lo_3[5] = __fmaf_rn(_unpack_bf16_44[5], w_cv0_3_5, -(_unpack_bf16_45[5] * w_sv0_3_5));
                            w_o_hi_3[5] = __fmaf_rn(_unpack_bf16_45[5], w_cv0_3_5, _unpack_bf16_44[5] * w_sv0_3_5);
                            w_o_lo_3[13] = __fmaf_rn(_unpack_bf16_44[13], w_cv1_3_5, -(_unpack_bf16_45[13] * w_sv1_3_5));
                            w_o_hi_3[13] = __fmaf_rn(_unpack_bf16_45[13], w_cv1_3_5, _unpack_bf16_44[13] * w_sv1_3_5);
                            float w_cv0_3_6 = _vec_load_12[6];
                            float w_sv0_3_6 = _vec_load_14[6];
                            float w_cv1_3_6 = _vec_load_13[6];
                            float w_sv1_3_6 = _vec_load_15[6];
                            w_o_lo_3[6] = __fmaf_rn(_unpack_bf16_44[6], w_cv0_3_6, -(_unpack_bf16_45[6] * w_sv0_3_6));
                            w_o_hi_3[6] = __fmaf_rn(_unpack_bf16_45[6], w_cv0_3_6, _unpack_bf16_44[6] * w_sv0_3_6);
                            w_o_lo_3[14] = __fmaf_rn(_unpack_bf16_44[14], w_cv1_3_6, -(_unpack_bf16_45[14] * w_sv1_3_6));
                            w_o_hi_3[14] = __fmaf_rn(_unpack_bf16_45[14], w_cv1_3_6, _unpack_bf16_44[14] * w_sv1_3_6);
                            float w_cv0_3_7 = _vec_load_12[7];
                            float w_sv0_3_7 = _vec_load_14[7];
                            float w_cv1_3_7 = _vec_load_13[7];
                            float w_sv1_3_7 = _vec_load_15[7];
                            w_o_lo_3[7] = __fmaf_rn(_unpack_bf16_44[7], w_cv0_3_7, -(_unpack_bf16_45[7] * w_sv0_3_7));
                            w_o_hi_3[7] = __fmaf_rn(_unpack_bf16_45[7], w_cv0_3_7, _unpack_bf16_44[7] * w_sv0_3_7);
                            w_o_lo_3[15] = __fmaf_rn(_unpack_bf16_44[15], w_cv1_3_7, -(_unpack_bf16_45[15] * w_sv1_3_7));
                            w_o_hi_3[15] = __fmaf_rn(_unpack_bf16_45[15], w_cv1_3_7, _unpack_bf16_44[15] * w_sv1_3_7);
                            unsigned int w_o_lo_3_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_lo_3[_lp*2 + 0], w_o_lo_3[_lp*2+1 + 0]));
                                w_o_lo_3_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int w_o_hi_3_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_hi_3[_lp*2 + 0], w_o_hi_3[_lp*2+1 + 0]));
                                w_o_hi_3_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            w_x_lo_3[0] = w_o_lo_3_bf16[0];
                            w_x_hi_3[0] = w_o_hi_3_bf16[0];
                            w_x_lo_3[1] = w_o_lo_3_bf16[1];
                            w_x_hi_3[1] = w_o_hi_3_bf16[1];
                            w_x_lo_3[2] = w_o_lo_3_bf16[2];
                            w_x_hi_3[2] = w_o_hi_3_bf16[2];
                            w_x_lo_3[3] = w_o_lo_3_bf16[3];
                            w_x_hi_3[3] = w_o_hi_3_bf16[3];
                            w_x_lo_3[4] = w_o_lo_3_bf16[4];
                            w_x_hi_3[4] = w_o_hi_3_bf16[4];
                            w_x_lo_3[5] = w_o_lo_3_bf16[5];
                            w_x_hi_3[5] = w_o_hi_3_bf16[5];
                            w_x_lo_3[6] = w_o_lo_3_bf16[6];
                            w_x_hi_3[6] = w_o_hi_3_bf16[6];
                            w_x_lo_3[7] = w_o_lo_3_bf16[7];
                            w_x_hi_3[7] = w_o_hi_3_bf16[7];
                        }
                    }
                    unsigned int w_q_words_3[4];
                    float w_q_vals_3[16];
                    float _unpack_bf16_46[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_46[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_3[_pair]) << 16);
                        (&_unpack_bf16_46[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_3[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_3[0] = _unpack_bf16_46[0] * w_out_scale_sel;
                    w_q_vals_3[1] = _unpack_bf16_46[1] * w_out_scale_sel;
                    w_q_vals_3[2] = _unpack_bf16_46[2] * w_out_scale_sel;
                    w_q_vals_3[3] = _unpack_bf16_46[3] * w_out_scale_sel;
                    w_q_vals_3[4] = _unpack_bf16_46[4] * w_out_scale_sel;
                    w_q_vals_3[5] = _unpack_bf16_46[5] * w_out_scale_sel;
                    w_q_vals_3[6] = _unpack_bf16_46[6] * w_out_scale_sel;
                    w_q_vals_3[7] = _unpack_bf16_46[7] * w_out_scale_sel;
                    w_q_vals_3[8] = _unpack_bf16_46[8] * w_out_scale_sel;
                    w_q_vals_3[9] = _unpack_bf16_46[9] * w_out_scale_sel;
                    w_q_vals_3[10] = _unpack_bf16_46[10] * w_out_scale_sel;
                    w_q_vals_3[11] = _unpack_bf16_46[11] * w_out_scale_sel;
                    w_q_vals_3[12] = _unpack_bf16_46[12] * w_out_scale_sel;
                    w_q_vals_3[13] = _unpack_bf16_46[13] * w_out_scale_sel;
                    w_q_vals_3[14] = _unpack_bf16_46[14] * w_out_scale_sel;
                    w_q_vals_3[15] = _unpack_bf16_46[15] * w_out_scale_sel;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_3[0]), "f"(w_q_vals_3[1]),
                                               "f"(w_q_vals_3[2]), "f"(w_q_vals_3[3]));
                        w_q_words_3[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_3[4]), "f"(w_q_vals_3[5]),
                                               "f"(w_q_vals_3[6]), "f"(w_q_vals_3[7]));
                        w_q_words_3[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_3[8]), "f"(w_q_vals_3[9]),
                                               "f"(w_q_vals_3[10]), "f"(w_q_vals_3[11]));
                        w_q_words_3[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_3[12]), "f"(w_q_vals_3[13]),
                                               "f"(w_q_vals_3[14]), "f"(w_q_vals_3[15]));
                        w_q_words_3[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(w_srow_3 * 128) + (unsigned int)(w_chunk_lo_3 ^ (w_srow_3 & 7) << 4)), "r"(w_q_words_3[0]), "r"(w_q_words_3[1]), "r"(w_q_words_3[2]), "r"(w_q_words_3[3]) : "memory");
                    float _unpack_bf16_47[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_47[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_3[_pair]) << 16);
                        (&_unpack_bf16_47[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_3[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_3[0] = _unpack_bf16_47[0] * w_out_scale_sel;
                    w_q_vals_3[1] = _unpack_bf16_47[1] * w_out_scale_sel;
                    w_q_vals_3[2] = _unpack_bf16_47[2] * w_out_scale_sel;
                    w_q_vals_3[3] = _unpack_bf16_47[3] * w_out_scale_sel;
                    w_q_vals_3[4] = _unpack_bf16_47[4] * w_out_scale_sel;
                    w_q_vals_3[5] = _unpack_bf16_47[5] * w_out_scale_sel;
                    w_q_vals_3[6] = _unpack_bf16_47[6] * w_out_scale_sel;
                    w_q_vals_3[7] = _unpack_bf16_47[7] * w_out_scale_sel;
                    w_q_vals_3[8] = _unpack_bf16_47[8] * w_out_scale_sel;
                    w_q_vals_3[9] = _unpack_bf16_47[9] * w_out_scale_sel;
                    w_q_vals_3[10] = _unpack_bf16_47[10] * w_out_scale_sel;
                    w_q_vals_3[11] = _unpack_bf16_47[11] * w_out_scale_sel;
                    w_q_vals_3[12] = _unpack_bf16_47[12] * w_out_scale_sel;
                    w_q_vals_3[13] = _unpack_bf16_47[13] * w_out_scale_sel;
                    w_q_vals_3[14] = _unpack_bf16_47[14] * w_out_scale_sel;
                    w_q_vals_3[15] = _unpack_bf16_47[15] * w_out_scale_sel;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_3[0]), "f"(w_q_vals_3[1]),
                                               "f"(w_q_vals_3[2]), "f"(w_q_vals_3[3]));
                        w_q_words_3[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_3[4]), "f"(w_q_vals_3[5]),
                                               "f"(w_q_vals_3[6]), "f"(w_q_vals_3[7]));
                        w_q_words_3[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_3[8]), "f"(w_q_vals_3[9]),
                                               "f"(w_q_vals_3[10]), "f"(w_q_vals_3[11]));
                        w_q_words_3[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_3[12]), "f"(w_q_vals_3[13]),
                                               "f"(w_q_vals_3[14]), "f"(w_q_vals_3[15]));
                        w_q_words_3[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(w_srow_3 * 128) + (unsigned int)(w_chunk_hi_3 ^ (w_srow_3 & 7) << 4)), "r"(w_q_words_3[0]), "r"(w_q_words_3[1]), "r"(w_q_words_3[2]), "r"(w_q_words_3[3]) : "memory");
                }
                __syncwarp();
                int s_row_3_0 = (role_tid >> 5) * 8 + lane / 8;
                int s_piece_3_0 = lane % 8;
                unsigned int s_words_3_0[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_0[(0) + 3]))
                    : "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(s_row_3_0 * 128) + (unsigned int)(s_piece_3_0 * 16 ^ (s_row_3_0 & 7) << 4)));
                int s_dst_3_0 = ((tile_m * 256 + 192 + s_row_3_0) * 56 + w_head) * 32 + s_piece_3_0 * 4;
                if (tile_m * 256 + 192 + s_row_3_0 < M) {
                    if (w_kind == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_3_0)[0] = reinterpret_cast<int4*>(s_words_3_0)[0];
                    } else if (w_kind == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_3_0)[0] = reinterpret_cast<int4*>(s_words_3_0)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_3_0)[0] = reinterpret_cast<int4*>(s_words_3_0)[0];
                    }
                }
                int s_row_3_1 = (role_tid >> 5) * 8 + 4 + lane / 8;
                int s_piece_3_1 = lane % 8;
                unsigned int s_words_3_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_1[(0) + 3]))
                    : "r"(B_stage_addr + free_stage * 8192 + (unsigned int)(s_row_3_1 * 128) + (unsigned int)(s_piece_3_1 * 16 ^ (s_row_3_1 & 7) << 4)));
                int s_dst_3_1 = ((tile_m * 256 + 192 + s_row_3_1) * 56 + w_head) * 32 + s_piece_3_1 * 4;
                if (tile_m * 256 + 192 + s_row_3_1 < M) {
                    if (w_kind == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_3_1)[0] = reinterpret_cast<int4*>(s_words_3_1)[0];
                    } else if (w_kind == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_3_1)[0] = reinterpret_cast<int4*>(s_words_3_1)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_3_1)[0] = reinterpret_cast<int4*>(s_words_3_1)[0];
                    }
                }
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
            float sw_1[16];
            float nw_1[16];
            unsigned int _phase_ab_full_1 = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / 2688 * 16 + (tile_1 - tile_1 / 2688 * 2688) % ((num_m_tiles - tile_1 / 2688 * 16 < 16) ? num_m_tiles - tile_1 / 2688 * 16 : 16);
                int tile_n_1 = (tile_1 - tile_1 / 2688 * 2688) / ((num_m_tiles - tile_1 / 2688 * 16 < 16) ? num_m_tiles - tile_1 / 2688 * 16 : 16);
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
                int c_d_col_0_1 = warp_n_1 * 64 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[0] = (float)q_norm_weight[c_d_col_0_1];
                    nw_1[1] = (float)q_norm_weight[c_d_col_0_1 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[0] = (float)k_norm_weight[c_d_col_0_1];
                    nw_1[1] = (float)k_norm_weight[c_d_col_0_1 + 1];
                } else {
                    nw_1[0] = 1.0f;
                    nw_1[1] = 1.0f;
                }
                int c_d_col_1_1 = warp_n_1 * 64 + 8 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[2] = (float)q_norm_weight[c_d_col_1_1];
                    nw_1[3] = (float)q_norm_weight[c_d_col_1_1 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[2] = (float)k_norm_weight[c_d_col_1_1];
                    nw_1[3] = (float)k_norm_weight[c_d_col_1_1 + 1];
                } else {
                    nw_1[2] = 1.0f;
                    nw_1[3] = 1.0f;
                }
                int c_d_col_2_1 = warp_n_1 * 64 + 16 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[4] = (float)q_norm_weight[c_d_col_2_1];
                    nw_1[5] = (float)q_norm_weight[c_d_col_2_1 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[4] = (float)k_norm_weight[c_d_col_2_1];
                    nw_1[5] = (float)k_norm_weight[c_d_col_2_1 + 1];
                } else {
                    nw_1[4] = 1.0f;
                    nw_1[5] = 1.0f;
                }
                int c_d_col_3_1 = warp_n_1 * 64 + 24 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[6] = (float)q_norm_weight[c_d_col_3_1];
                    nw_1[7] = (float)q_norm_weight[c_d_col_3_1 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[6] = (float)k_norm_weight[c_d_col_3_1];
                    nw_1[7] = (float)k_norm_weight[c_d_col_3_1 + 1];
                } else {
                    nw_1[6] = 1.0f;
                    nw_1[7] = 1.0f;
                }
                int c_d_col_4_1 = warp_n_1 * 64 + 32 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[8] = (float)q_norm_weight[c_d_col_4_1];
                    nw_1[9] = (float)q_norm_weight[c_d_col_4_1 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[8] = (float)k_norm_weight[c_d_col_4_1];
                    nw_1[9] = (float)k_norm_weight[c_d_col_4_1 + 1];
                } else {
                    nw_1[8] = 1.0f;
                    nw_1[9] = 1.0f;
                }
                int c_d_col_5_1 = warp_n_1 * 64 + 40 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[10] = (float)q_norm_weight[c_d_col_5_1];
                    nw_1[11] = (float)q_norm_weight[c_d_col_5_1 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[10] = (float)k_norm_weight[c_d_col_5_1];
                    nw_1[11] = (float)k_norm_weight[c_d_col_5_1 + 1];
                } else {
                    nw_1[10] = 1.0f;
                    nw_1[11] = 1.0f;
                }
                int c_d_col_6_1 = warp_n_1 * 64 + 48 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[12] = (float)q_norm_weight[c_d_col_6_1];
                    nw_1[13] = (float)q_norm_weight[c_d_col_6_1 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[12] = (float)k_norm_weight[c_d_col_6_1];
                    nw_1[13] = (float)k_norm_weight[c_d_col_6_1 + 1];
                } else {
                    nw_1[12] = 1.0f;
                    nw_1[13] = 1.0f;
                }
                int c_d_col_7_1 = warp_n_1 * 64 + 56 + (lane & 3) * 2;
                if (c_kind_1 == 0) {
                    nw_1[14] = (float)q_norm_weight[c_d_col_7_1];
                    nw_1[15] = (float)q_norm_weight[c_d_col_7_1 + 1];
                } else if (c_kind_1 == 1) {
                    nw_1[14] = (float)k_norm_weight[c_d_col_7_1];
                    nw_1[15] = (float)k_norm_weight[c_d_col_7_1 + 1];
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
                float w_out_scale_sel_1 = ((w_kind_1 == 0) ? out_scale_q : ((w_kind_1 == 1) ? out_scale_k : out_scale_v));
                float w_sf_mul_sel_1 = ((w_kind_1 == 0) ? sf_mul_q : ((w_kind_1 == 1) ? sf_mul_k : sf_mul_v));
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_row_lo_0_1 = tile_m_1 * 256 + warp_m_1 * 16 + (lane >> 2);
                int w_row_hi_0_1 = w_row_lo_0_1 + 8;
                float w_sa_lo_0_1 = ((w_row_lo_0_1 < M) ? act_scale[w_row_lo_0_1] : 0.0f);
                float w_sa_hi_0_1 = ((w_row_hi_0_1 < M) ? act_scale[w_row_hi_0_1] : 0.0f);
                float w_rounded_0_1[32];
                float w_ss_lo_0_1 = 0.0f;
                float w_ss_hi_0_1 = 0.0f;
                float w_scaled_0_0_1[4];
                w_scaled_0_0_1[0] = accum_1[0] * w_sa_lo_0_1 * sw_1[0];
                w_scaled_0_0_1[1] = accum_1[1] * w_sa_lo_0_1 * sw_1[1];
                w_scaled_0_0_1[2] = accum_1[2] * w_sa_hi_0_1 * sw_1[0];
                w_scaled_0_0_1[3] = accum_1[3] * w_sa_hi_0_1 * sw_1[1];
                unsigned int w_scaled_0_0_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_0_1[_lp*2 + 0], w_scaled_0_0_1[_lp*2+1 + 0]));
                    w_scaled_0_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_48[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_48[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_0_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_48[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_0_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_0_1[0] = _unpack_bf16_48[0];
                w_rounded_0_1[1] = _unpack_bf16_48[1];
                w_rounded_0_1[2] = _unpack_bf16_48[2];
                w_rounded_0_1[3] = _unpack_bf16_48[3];
                w_ss_lo_0_1 += __fmaf_rn(_unpack_bf16_48[0], _unpack_bf16_48[0], _unpack_bf16_48[1] * _unpack_bf16_48[1]);
                w_ss_hi_0_1 += __fmaf_rn(_unpack_bf16_48[2], _unpack_bf16_48[2], _unpack_bf16_48[3] * _unpack_bf16_48[3]);
                float w_scaled_0_1_1[4];
                w_scaled_0_1_1[0] = accum_1[4] * w_sa_lo_0_1 * sw_1[2];
                w_scaled_0_1_1[1] = accum_1[5] * w_sa_lo_0_1 * sw_1[3];
                w_scaled_0_1_1[2] = accum_1[6] * w_sa_hi_0_1 * sw_1[2];
                w_scaled_0_1_1[3] = accum_1[7] * w_sa_hi_0_1 * sw_1[3];
                unsigned int w_scaled_0_1_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_1_1[_lp*2 + 0], w_scaled_0_1_1[_lp*2+1 + 0]));
                    w_scaled_0_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_49[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_49[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_1_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_49[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_1_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_0_1[4] = _unpack_bf16_49[0];
                w_rounded_0_1[5] = _unpack_bf16_49[1];
                w_rounded_0_1[6] = _unpack_bf16_49[2];
                w_rounded_0_1[7] = _unpack_bf16_49[3];
                w_ss_lo_0_1 += __fmaf_rn(_unpack_bf16_49[0], _unpack_bf16_49[0], _unpack_bf16_49[1] * _unpack_bf16_49[1]);
                w_ss_hi_0_1 += __fmaf_rn(_unpack_bf16_49[2], _unpack_bf16_49[2], _unpack_bf16_49[3] * _unpack_bf16_49[3]);
                float w_scaled_0_2_1[4];
                w_scaled_0_2_1[0] = accum_1[8] * w_sa_lo_0_1 * sw_1[4];
                w_scaled_0_2_1[1] = accum_1[9] * w_sa_lo_0_1 * sw_1[5];
                w_scaled_0_2_1[2] = accum_1[10] * w_sa_hi_0_1 * sw_1[4];
                w_scaled_0_2_1[3] = accum_1[11] * w_sa_hi_0_1 * sw_1[5];
                unsigned int w_scaled_0_2_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_2_1[_lp*2 + 0], w_scaled_0_2_1[_lp*2+1 + 0]));
                    w_scaled_0_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_50[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_50[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_2_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_50[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_2_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_0_1[8] = _unpack_bf16_50[0];
                w_rounded_0_1[9] = _unpack_bf16_50[1];
                w_rounded_0_1[10] = _unpack_bf16_50[2];
                w_rounded_0_1[11] = _unpack_bf16_50[3];
                w_ss_lo_0_1 += __fmaf_rn(_unpack_bf16_50[0], _unpack_bf16_50[0], _unpack_bf16_50[1] * _unpack_bf16_50[1]);
                w_ss_hi_0_1 += __fmaf_rn(_unpack_bf16_50[2], _unpack_bf16_50[2], _unpack_bf16_50[3] * _unpack_bf16_50[3]);
                float w_scaled_0_3_1[4];
                w_scaled_0_3_1[0] = accum_1[12] * w_sa_lo_0_1 * sw_1[6];
                w_scaled_0_3_1[1] = accum_1[13] * w_sa_lo_0_1 * sw_1[7];
                w_scaled_0_3_1[2] = accum_1[14] * w_sa_hi_0_1 * sw_1[6];
                w_scaled_0_3_1[3] = accum_1[15] * w_sa_hi_0_1 * sw_1[7];
                unsigned int w_scaled_0_3_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_3_1[_lp*2 + 0], w_scaled_0_3_1[_lp*2+1 + 0]));
                    w_scaled_0_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_51[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_51[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_3_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_51[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_3_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_0_1[12] = _unpack_bf16_51[0];
                w_rounded_0_1[13] = _unpack_bf16_51[1];
                w_rounded_0_1[14] = _unpack_bf16_51[2];
                w_rounded_0_1[15] = _unpack_bf16_51[3];
                w_ss_lo_0_1 += __fmaf_rn(_unpack_bf16_51[0], _unpack_bf16_51[0], _unpack_bf16_51[1] * _unpack_bf16_51[1]);
                w_ss_hi_0_1 += __fmaf_rn(_unpack_bf16_51[2], _unpack_bf16_51[2], _unpack_bf16_51[3] * _unpack_bf16_51[3]);
                float w_scaled_0_4_1[4];
                w_scaled_0_4_1[0] = accum_1[16] * w_sa_lo_0_1 * sw_1[8];
                w_scaled_0_4_1[1] = accum_1[17] * w_sa_lo_0_1 * sw_1[9];
                w_scaled_0_4_1[2] = accum_1[18] * w_sa_hi_0_1 * sw_1[8];
                w_scaled_0_4_1[3] = accum_1[19] * w_sa_hi_0_1 * sw_1[9];
                unsigned int w_scaled_0_4_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_4_1[_lp*2 + 0], w_scaled_0_4_1[_lp*2+1 + 0]));
                    w_scaled_0_4_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_52[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_52[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_4_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_52[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_4_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_0_1[16] = _unpack_bf16_52[0];
                w_rounded_0_1[17] = _unpack_bf16_52[1];
                w_rounded_0_1[18] = _unpack_bf16_52[2];
                w_rounded_0_1[19] = _unpack_bf16_52[3];
                w_ss_lo_0_1 += __fmaf_rn(_unpack_bf16_52[0], _unpack_bf16_52[0], _unpack_bf16_52[1] * _unpack_bf16_52[1]);
                w_ss_hi_0_1 += __fmaf_rn(_unpack_bf16_52[2], _unpack_bf16_52[2], _unpack_bf16_52[3] * _unpack_bf16_52[3]);
                float w_scaled_0_5_1[4];
                w_scaled_0_5_1[0] = accum_1[20] * w_sa_lo_0_1 * sw_1[10];
                w_scaled_0_5_1[1] = accum_1[21] * w_sa_lo_0_1 * sw_1[11];
                w_scaled_0_5_1[2] = accum_1[22] * w_sa_hi_0_1 * sw_1[10];
                w_scaled_0_5_1[3] = accum_1[23] * w_sa_hi_0_1 * sw_1[11];
                unsigned int w_scaled_0_5_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_5_1[_lp*2 + 0], w_scaled_0_5_1[_lp*2+1 + 0]));
                    w_scaled_0_5_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_53[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_53[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_5_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_53[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_5_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_0_1[20] = _unpack_bf16_53[0];
                w_rounded_0_1[21] = _unpack_bf16_53[1];
                w_rounded_0_1[22] = _unpack_bf16_53[2];
                w_rounded_0_1[23] = _unpack_bf16_53[3];
                w_ss_lo_0_1 += __fmaf_rn(_unpack_bf16_53[0], _unpack_bf16_53[0], _unpack_bf16_53[1] * _unpack_bf16_53[1]);
                w_ss_hi_0_1 += __fmaf_rn(_unpack_bf16_53[2], _unpack_bf16_53[2], _unpack_bf16_53[3] * _unpack_bf16_53[3]);
                float w_scaled_0_6_1[4];
                w_scaled_0_6_1[0] = accum_1[24] * w_sa_lo_0_1 * sw_1[12];
                w_scaled_0_6_1[1] = accum_1[25] * w_sa_lo_0_1 * sw_1[13];
                w_scaled_0_6_1[2] = accum_1[26] * w_sa_hi_0_1 * sw_1[12];
                w_scaled_0_6_1[3] = accum_1[27] * w_sa_hi_0_1 * sw_1[13];
                unsigned int w_scaled_0_6_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_6_1[_lp*2 + 0], w_scaled_0_6_1[_lp*2+1 + 0]));
                    w_scaled_0_6_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_54[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_54[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_6_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_54[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_6_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_0_1[24] = _unpack_bf16_54[0];
                w_rounded_0_1[25] = _unpack_bf16_54[1];
                w_rounded_0_1[26] = _unpack_bf16_54[2];
                w_rounded_0_1[27] = _unpack_bf16_54[3];
                w_ss_lo_0_1 += __fmaf_rn(_unpack_bf16_54[0], _unpack_bf16_54[0], _unpack_bf16_54[1] * _unpack_bf16_54[1]);
                w_ss_hi_0_1 += __fmaf_rn(_unpack_bf16_54[2], _unpack_bf16_54[2], _unpack_bf16_54[3] * _unpack_bf16_54[3]);
                float w_scaled_0_7_1[4];
                w_scaled_0_7_1[0] = accum_1[28] * w_sa_lo_0_1 * sw_1[14];
                w_scaled_0_7_1[1] = accum_1[29] * w_sa_lo_0_1 * sw_1[15];
                w_scaled_0_7_1[2] = accum_1[30] * w_sa_hi_0_1 * sw_1[14];
                w_scaled_0_7_1[3] = accum_1[31] * w_sa_hi_0_1 * sw_1[15];
                unsigned int w_scaled_0_7_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_0_7_1[_lp*2 + 0], w_scaled_0_7_1[_lp*2+1 + 0]));
                    w_scaled_0_7_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_55[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_55[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_7_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_55[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_0_7_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_0_1[28] = _unpack_bf16_55[0];
                w_rounded_0_1[29] = _unpack_bf16_55[1];
                w_rounded_0_1[30] = _unpack_bf16_55[2];
                w_rounded_0_1[31] = _unpack_bf16_55[3];
                w_ss_lo_0_1 += __fmaf_rn(_unpack_bf16_55[0], _unpack_bf16_55[0], _unpack_bf16_55[1] * _unpack_bf16_55[1]);
                w_ss_hi_0_1 += __fmaf_rn(_unpack_bf16_55[2], _unpack_bf16_55[2], _unpack_bf16_55[3] * _unpack_bf16_55[3]);
                float w_rstd_lo_0_1 = 1.0f;
                float w_rstd_hi_0_1 = 1.0f;
                if (w_kind_1 < 2) {
                    float _shfl_xor_sync_0_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_0_1, 1);
                    w_ss_lo_0_1 += _shfl_xor_sync_0_1;
                    float _shfl_xor_sync_1_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_0_1, 2);
                    w_ss_lo_0_1 += _shfl_xor_sync_1_1;
                    float _shfl_xor_sync_2_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_0_1, 1);
                    w_ss_hi_0_1 += _shfl_xor_sync_2_1;
                    float _shfl_xor_sync_3_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_0_1, 2);
                    w_ss_hi_0_1 += _shfl_xor_sync_3_1;
                    int w_srow_lo_0_1 = warp_m_1 * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n_1 * 64 + w_srow_lo_0_1] = w_ss_lo_0_1;
                        rowstat[warp_n_1 * 64 + w_srow_lo_0_1 + 8] = w_ss_hi_0_1;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float w_other_lo_0_1 = rowstat[(1 - warp_n_1) * 64 + w_srow_lo_0_1];
                    float w_other_hi_0_1 = rowstat[(1 - warp_n_1) * 64 + w_srow_lo_0_1 + 8];
                    float _fdiv_5_1 = __fdiv_rn(w_ss_lo_0_1 + w_other_lo_0_1, 128.0f);
                    float _rsqrt_4_1 = rsqrtf(_fdiv_5_1 + eps);
                    w_rstd_lo_0_1 = _rsqrt_4_1;
                    float _fdiv_7_1 = __fdiv_rn(w_ss_hi_0_1 + w_other_hi_0_1, 128.0f);
                    float _rsqrt_6_1 = rsqrtf(_fdiv_7_1 + eps);
                    w_rstd_hi_0_1 = _rsqrt_6_1;
                }
                int w_srow_st_0_1 = warp_m_1 * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float w_normed_0_0_1[4];
                if (w_kind_1 < 2) {
                    w_normed_0_0_1[0] = w_rounded_0_1[0] * w_rstd_lo_0_1 * nw_1[0];
                    w_normed_0_0_1[1] = w_rounded_0_1[1] * w_rstd_lo_0_1 * nw_1[1];
                    w_normed_0_0_1[2] = w_rounded_0_1[2] * w_rstd_hi_0_1 * nw_1[0];
                    w_normed_0_0_1[3] = w_rounded_0_1[3] * w_rstd_hi_0_1 * nw_1[1];
                } else {
                    w_normed_0_0_1[0] = w_rounded_0_1[0];
                    w_normed_0_0_1[1] = w_rounded_0_1[1];
                    w_normed_0_0_1[2] = w_rounded_0_1[2];
                    w_normed_0_0_1[3] = w_rounded_0_1[3];
                }
                unsigned int w_normed_0_0_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_0_1[_lp*2 + 0], w_normed_0_0_1[_lp*2+1 + 0]));
                    w_normed_0_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_0_1 * 256) + (unsigned int)(warp_n_1 * 64 * 2 ^ (w_srow_st_0_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_0_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_0_bf16_1[1]))
                    : "memory");
                float w_normed_0_1_1[4];
                if (w_kind_1 < 2) {
                    w_normed_0_1_1[0] = w_rounded_0_1[4] * w_rstd_lo_0_1 * nw_1[2];
                    w_normed_0_1_1[1] = w_rounded_0_1[5] * w_rstd_lo_0_1 * nw_1[3];
                    w_normed_0_1_1[2] = w_rounded_0_1[6] * w_rstd_hi_0_1 * nw_1[2];
                    w_normed_0_1_1[3] = w_rounded_0_1[7] * w_rstd_hi_0_1 * nw_1[3];
                } else {
                    w_normed_0_1_1[0] = w_rounded_0_1[4];
                    w_normed_0_1_1[1] = w_rounded_0_1[5];
                    w_normed_0_1_1[2] = w_rounded_0_1[6];
                    w_normed_0_1_1[3] = w_rounded_0_1[7];
                }
                unsigned int w_normed_0_1_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_1_1[_lp*2 + 0], w_normed_0_1_1[_lp*2+1 + 0]));
                    w_normed_0_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_0_1 * 256) + (unsigned int)((warp_n_1 * 64 + 8) * 2 ^ (w_srow_st_0_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_1_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_1_bf16_1[1]))
                    : "memory");
                float w_normed_0_2_1[4];
                if (w_kind_1 < 2) {
                    w_normed_0_2_1[0] = w_rounded_0_1[8] * w_rstd_lo_0_1 * nw_1[4];
                    w_normed_0_2_1[1] = w_rounded_0_1[9] * w_rstd_lo_0_1 * nw_1[5];
                    w_normed_0_2_1[2] = w_rounded_0_1[10] * w_rstd_hi_0_1 * nw_1[4];
                    w_normed_0_2_1[3] = w_rounded_0_1[11] * w_rstd_hi_0_1 * nw_1[5];
                } else {
                    w_normed_0_2_1[0] = w_rounded_0_1[8];
                    w_normed_0_2_1[1] = w_rounded_0_1[9];
                    w_normed_0_2_1[2] = w_rounded_0_1[10];
                    w_normed_0_2_1[3] = w_rounded_0_1[11];
                }
                unsigned int w_normed_0_2_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_2_1[_lp*2 + 0], w_normed_0_2_1[_lp*2+1 + 0]));
                    w_normed_0_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_0_1 * 256) + (unsigned int)((warp_n_1 * 64 + 16) * 2 ^ (w_srow_st_0_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_2_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_2_bf16_1[1]))
                    : "memory");
                float w_normed_0_3_1[4];
                if (w_kind_1 < 2) {
                    w_normed_0_3_1[0] = w_rounded_0_1[12] * w_rstd_lo_0_1 * nw_1[6];
                    w_normed_0_3_1[1] = w_rounded_0_1[13] * w_rstd_lo_0_1 * nw_1[7];
                    w_normed_0_3_1[2] = w_rounded_0_1[14] * w_rstd_hi_0_1 * nw_1[6];
                    w_normed_0_3_1[3] = w_rounded_0_1[15] * w_rstd_hi_0_1 * nw_1[7];
                } else {
                    w_normed_0_3_1[0] = w_rounded_0_1[12];
                    w_normed_0_3_1[1] = w_rounded_0_1[13];
                    w_normed_0_3_1[2] = w_rounded_0_1[14];
                    w_normed_0_3_1[3] = w_rounded_0_1[15];
                }
                unsigned int w_normed_0_3_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_3_1[_lp*2 + 0], w_normed_0_3_1[_lp*2+1 + 0]));
                    w_normed_0_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_0_1 * 256) + (unsigned int)((warp_n_1 * 64 + 24) * 2 ^ (w_srow_st_0_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_3_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_3_bf16_1[1]))
                    : "memory");
                float w_normed_0_4_1[4];
                if (w_kind_1 < 2) {
                    w_normed_0_4_1[0] = w_rounded_0_1[16] * w_rstd_lo_0_1 * nw_1[8];
                    w_normed_0_4_1[1] = w_rounded_0_1[17] * w_rstd_lo_0_1 * nw_1[9];
                    w_normed_0_4_1[2] = w_rounded_0_1[18] * w_rstd_hi_0_1 * nw_1[8];
                    w_normed_0_4_1[3] = w_rounded_0_1[19] * w_rstd_hi_0_1 * nw_1[9];
                } else {
                    w_normed_0_4_1[0] = w_rounded_0_1[16];
                    w_normed_0_4_1[1] = w_rounded_0_1[17];
                    w_normed_0_4_1[2] = w_rounded_0_1[18];
                    w_normed_0_4_1[3] = w_rounded_0_1[19];
                }
                unsigned int w_normed_0_4_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_4_1[_lp*2 + 0], w_normed_0_4_1[_lp*2+1 + 0]));
                    w_normed_0_4_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_0_1 * 256) + (unsigned int)((warp_n_1 * 64 + 32) * 2 ^ (w_srow_st_0_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_4_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_4_bf16_1[1]))
                    : "memory");
                float w_normed_0_5_1[4];
                if (w_kind_1 < 2) {
                    w_normed_0_5_1[0] = w_rounded_0_1[20] * w_rstd_lo_0_1 * nw_1[10];
                    w_normed_0_5_1[1] = w_rounded_0_1[21] * w_rstd_lo_0_1 * nw_1[11];
                    w_normed_0_5_1[2] = w_rounded_0_1[22] * w_rstd_hi_0_1 * nw_1[10];
                    w_normed_0_5_1[3] = w_rounded_0_1[23] * w_rstd_hi_0_1 * nw_1[11];
                } else {
                    w_normed_0_5_1[0] = w_rounded_0_1[20];
                    w_normed_0_5_1[1] = w_rounded_0_1[21];
                    w_normed_0_5_1[2] = w_rounded_0_1[22];
                    w_normed_0_5_1[3] = w_rounded_0_1[23];
                }
                unsigned int w_normed_0_5_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_5_1[_lp*2 + 0], w_normed_0_5_1[_lp*2+1 + 0]));
                    w_normed_0_5_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_0_1 * 256) + (unsigned int)((warp_n_1 * 64 + 40) * 2 ^ (w_srow_st_0_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_5_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_5_bf16_1[1]))
                    : "memory");
                float w_normed_0_6_1[4];
                if (w_kind_1 < 2) {
                    w_normed_0_6_1[0] = w_rounded_0_1[24] * w_rstd_lo_0_1 * nw_1[12];
                    w_normed_0_6_1[1] = w_rounded_0_1[25] * w_rstd_lo_0_1 * nw_1[13];
                    w_normed_0_6_1[2] = w_rounded_0_1[26] * w_rstd_hi_0_1 * nw_1[12];
                    w_normed_0_6_1[3] = w_rounded_0_1[27] * w_rstd_hi_0_1 * nw_1[13];
                } else {
                    w_normed_0_6_1[0] = w_rounded_0_1[24];
                    w_normed_0_6_1[1] = w_rounded_0_1[25];
                    w_normed_0_6_1[2] = w_rounded_0_1[26];
                    w_normed_0_6_1[3] = w_rounded_0_1[27];
                }
                unsigned int w_normed_0_6_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_6_1[_lp*2 + 0], w_normed_0_6_1[_lp*2+1 + 0]));
                    w_normed_0_6_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_0_1 * 256) + (unsigned int)((warp_n_1 * 64 + 48) * 2 ^ (w_srow_st_0_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_6_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_6_bf16_1[1]))
                    : "memory");
                float w_normed_0_7_1[4];
                if (w_kind_1 < 2) {
                    w_normed_0_7_1[0] = w_rounded_0_1[28] * w_rstd_lo_0_1 * nw_1[14];
                    w_normed_0_7_1[1] = w_rounded_0_1[29] * w_rstd_lo_0_1 * nw_1[15];
                    w_normed_0_7_1[2] = w_rounded_0_1[30] * w_rstd_hi_0_1 * nw_1[14];
                    w_normed_0_7_1[3] = w_rounded_0_1[31] * w_rstd_hi_0_1 * nw_1[15];
                } else {
                    w_normed_0_7_1[0] = w_rounded_0_1[28];
                    w_normed_0_7_1[1] = w_rounded_0_1[29];
                    w_normed_0_7_1[2] = w_rounded_0_1[30];
                    w_normed_0_7_1[3] = w_rounded_0_1[31];
                }
                unsigned int w_normed_0_7_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_0_7_1[_lp*2 + 0], w_normed_0_7_1[_lp*2+1 + 0]));
                    w_normed_0_7_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_0_1 * 256) + (unsigned int)((warp_n_1 * 64 + 56) * 2 ^ (w_srow_st_0_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_7_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_0_7_bf16_1[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_srow_0_1 = role_tid_1 >> 2;
                int w_c_0_1 = role_tid_1 & 3;
                int w_grow_0_1 = tile_m_1 * 256 + w_srow_0_1;
                int w_chunk_lo_0_1 = ((w_c_0_1 < 3) ? w_c_0_1 * 16 : 96);
                int w_chunk_hi_0_1 = ((w_c_0_1 < 3) ? w_chunk_lo_0_1 + 48 : 112);
                if (w_grow_0_1 < M) {
                    unsigned int w_x_lo_0_1[8];
                    unsigned int w_x_hi_0_1[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0_1[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_0_1 * 256) + (unsigned int)(w_chunk_lo_0_1 * 2 ^ (w_srow_0_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_0_1[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_0_1 * 256) + (unsigned int)(w_chunk_lo_0_1 * 2 + 16 ^ (w_srow_0_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0_1[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_0_1 * 256) + (unsigned int)(w_chunk_hi_0_1 * 2 ^ (w_srow_0_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_0_1[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_0_1 * 256) + (unsigned int)(w_chunk_hi_0_1 * 2 + 16 ^ (w_srow_0_1 & 7) << 4)));
                    if (w_c_0_1 < 3) {
                        if (w_kind_1 < 2) {
                            float _unpack_bf16_56[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_56[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_0_1[_pair]) << 16);
                                (&_unpack_bf16_56[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_0_1[_pair]) & 0xffff0000u);
                            }
                            float _unpack_bf16_57[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_57[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_0_1[_pair]) << 16);
                                (&_unpack_bf16_57[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_0_1[_pair]) & 0xffff0000u);
                            }
                            float _vec_load_16[8];
                            {
                                const uint4* _vptr_16 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_0_1 * 96 + w_chunk_lo_0_1) + 0);
                                uint4 _vld_16[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_16[_blk] = _vptr_16[_blk];
                                    uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_16[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                                        (&_vec_load_16[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_17[8];
                            {
                                const uint4* _vptr_17 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_0_1 * 96 + w_chunk_lo_0_1 + 8) + 0);
                                uint4 _vld_17[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_17[_blk] = _vptr_17[_blk];
                                    uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_17[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
                                        (&_vec_load_17[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_18[8];
                            {
                                const uint4* _vptr_18 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_0_1 * 96 + 48 + w_chunk_lo_0_1) + 0);
                                uint4 _vld_18[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_18[_blk] = _vptr_18[_blk];
                                    uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_18[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                                        (&_vec_load_18[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_19[8];
                            {
                                const uint4* _vptr_19 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_0_1 * 96 + 48 + w_chunk_lo_0_1 + 8) + 0);
                                uint4 _vld_19[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_19[_blk] = _vptr_19[_blk];
                                    uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_19[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                                        (&_vec_load_19[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float w_o_lo_0_1[16];
                            float w_o_hi_0_1[16];
                            float w_cv0_0_0_1 = _vec_load_16[0];
                            float w_sv0_0_0_1 = _vec_load_18[0];
                            float w_cv1_0_0_1 = _vec_load_17[0];
                            float w_sv1_0_0_1 = _vec_load_19[0];
                            w_o_lo_0_1[0] = __fmaf_rn(_unpack_bf16_56[0], w_cv0_0_0_1, -(_unpack_bf16_57[0] * w_sv0_0_0_1));
                            w_o_hi_0_1[0] = __fmaf_rn(_unpack_bf16_57[0], w_cv0_0_0_1, _unpack_bf16_56[0] * w_sv0_0_0_1);
                            w_o_lo_0_1[8] = __fmaf_rn(_unpack_bf16_56[8], w_cv1_0_0_1, -(_unpack_bf16_57[8] * w_sv1_0_0_1));
                            w_o_hi_0_1[8] = __fmaf_rn(_unpack_bf16_57[8], w_cv1_0_0_1, _unpack_bf16_56[8] * w_sv1_0_0_1);
                            float w_cv0_0_1_1 = _vec_load_16[1];
                            float w_sv0_0_1_1 = _vec_load_18[1];
                            float w_cv1_0_1_1 = _vec_load_17[1];
                            float w_sv1_0_1_1 = _vec_load_19[1];
                            w_o_lo_0_1[1] = __fmaf_rn(_unpack_bf16_56[1], w_cv0_0_1_1, -(_unpack_bf16_57[1] * w_sv0_0_1_1));
                            w_o_hi_0_1[1] = __fmaf_rn(_unpack_bf16_57[1], w_cv0_0_1_1, _unpack_bf16_56[1] * w_sv0_0_1_1);
                            w_o_lo_0_1[9] = __fmaf_rn(_unpack_bf16_56[9], w_cv1_0_1_1, -(_unpack_bf16_57[9] * w_sv1_0_1_1));
                            w_o_hi_0_1[9] = __fmaf_rn(_unpack_bf16_57[9], w_cv1_0_1_1, _unpack_bf16_56[9] * w_sv1_0_1_1);
                            float w_cv0_0_2_1 = _vec_load_16[2];
                            float w_sv0_0_2_1 = _vec_load_18[2];
                            float w_cv1_0_2_1 = _vec_load_17[2];
                            float w_sv1_0_2_1 = _vec_load_19[2];
                            w_o_lo_0_1[2] = __fmaf_rn(_unpack_bf16_56[2], w_cv0_0_2_1, -(_unpack_bf16_57[2] * w_sv0_0_2_1));
                            w_o_hi_0_1[2] = __fmaf_rn(_unpack_bf16_57[2], w_cv0_0_2_1, _unpack_bf16_56[2] * w_sv0_0_2_1);
                            w_o_lo_0_1[10] = __fmaf_rn(_unpack_bf16_56[10], w_cv1_0_2_1, -(_unpack_bf16_57[10] * w_sv1_0_2_1));
                            w_o_hi_0_1[10] = __fmaf_rn(_unpack_bf16_57[10], w_cv1_0_2_1, _unpack_bf16_56[10] * w_sv1_0_2_1);
                            float w_cv0_0_3_1 = _vec_load_16[3];
                            float w_sv0_0_3_1 = _vec_load_18[3];
                            float w_cv1_0_3_1 = _vec_load_17[3];
                            float w_sv1_0_3_1 = _vec_load_19[3];
                            w_o_lo_0_1[3] = __fmaf_rn(_unpack_bf16_56[3], w_cv0_0_3_1, -(_unpack_bf16_57[3] * w_sv0_0_3_1));
                            w_o_hi_0_1[3] = __fmaf_rn(_unpack_bf16_57[3], w_cv0_0_3_1, _unpack_bf16_56[3] * w_sv0_0_3_1);
                            w_o_lo_0_1[11] = __fmaf_rn(_unpack_bf16_56[11], w_cv1_0_3_1, -(_unpack_bf16_57[11] * w_sv1_0_3_1));
                            w_o_hi_0_1[11] = __fmaf_rn(_unpack_bf16_57[11], w_cv1_0_3_1, _unpack_bf16_56[11] * w_sv1_0_3_1);
                            float w_cv0_0_4_1 = _vec_load_16[4];
                            float w_sv0_0_4_1 = _vec_load_18[4];
                            float w_cv1_0_4_1 = _vec_load_17[4];
                            float w_sv1_0_4_1 = _vec_load_19[4];
                            w_o_lo_0_1[4] = __fmaf_rn(_unpack_bf16_56[4], w_cv0_0_4_1, -(_unpack_bf16_57[4] * w_sv0_0_4_1));
                            w_o_hi_0_1[4] = __fmaf_rn(_unpack_bf16_57[4], w_cv0_0_4_1, _unpack_bf16_56[4] * w_sv0_0_4_1);
                            w_o_lo_0_1[12] = __fmaf_rn(_unpack_bf16_56[12], w_cv1_0_4_1, -(_unpack_bf16_57[12] * w_sv1_0_4_1));
                            w_o_hi_0_1[12] = __fmaf_rn(_unpack_bf16_57[12], w_cv1_0_4_1, _unpack_bf16_56[12] * w_sv1_0_4_1);
                            float w_cv0_0_5_1 = _vec_load_16[5];
                            float w_sv0_0_5_1 = _vec_load_18[5];
                            float w_cv1_0_5_1 = _vec_load_17[5];
                            float w_sv1_0_5_1 = _vec_load_19[5];
                            w_o_lo_0_1[5] = __fmaf_rn(_unpack_bf16_56[5], w_cv0_0_5_1, -(_unpack_bf16_57[5] * w_sv0_0_5_1));
                            w_o_hi_0_1[5] = __fmaf_rn(_unpack_bf16_57[5], w_cv0_0_5_1, _unpack_bf16_56[5] * w_sv0_0_5_1);
                            w_o_lo_0_1[13] = __fmaf_rn(_unpack_bf16_56[13], w_cv1_0_5_1, -(_unpack_bf16_57[13] * w_sv1_0_5_1));
                            w_o_hi_0_1[13] = __fmaf_rn(_unpack_bf16_57[13], w_cv1_0_5_1, _unpack_bf16_56[13] * w_sv1_0_5_1);
                            float w_cv0_0_6_1 = _vec_load_16[6];
                            float w_sv0_0_6_1 = _vec_load_18[6];
                            float w_cv1_0_6_1 = _vec_load_17[6];
                            float w_sv1_0_6_1 = _vec_load_19[6];
                            w_o_lo_0_1[6] = __fmaf_rn(_unpack_bf16_56[6], w_cv0_0_6_1, -(_unpack_bf16_57[6] * w_sv0_0_6_1));
                            w_o_hi_0_1[6] = __fmaf_rn(_unpack_bf16_57[6], w_cv0_0_6_1, _unpack_bf16_56[6] * w_sv0_0_6_1);
                            w_o_lo_0_1[14] = __fmaf_rn(_unpack_bf16_56[14], w_cv1_0_6_1, -(_unpack_bf16_57[14] * w_sv1_0_6_1));
                            w_o_hi_0_1[14] = __fmaf_rn(_unpack_bf16_57[14], w_cv1_0_6_1, _unpack_bf16_56[14] * w_sv1_0_6_1);
                            float w_cv0_0_7_1 = _vec_load_16[7];
                            float w_sv0_0_7_1 = _vec_load_18[7];
                            float w_cv1_0_7_1 = _vec_load_17[7];
                            float w_sv1_0_7_1 = _vec_load_19[7];
                            w_o_lo_0_1[7] = __fmaf_rn(_unpack_bf16_56[7], w_cv0_0_7_1, -(_unpack_bf16_57[7] * w_sv0_0_7_1));
                            w_o_hi_0_1[7] = __fmaf_rn(_unpack_bf16_57[7], w_cv0_0_7_1, _unpack_bf16_56[7] * w_sv0_0_7_1);
                            w_o_lo_0_1[15] = __fmaf_rn(_unpack_bf16_56[15], w_cv1_0_7_1, -(_unpack_bf16_57[15] * w_sv1_0_7_1));
                            w_o_hi_0_1[15] = __fmaf_rn(_unpack_bf16_57[15], w_cv1_0_7_1, _unpack_bf16_56[15] * w_sv1_0_7_1);
                            unsigned int w_o_lo_0_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_lo_0_1[_lp*2 + 0], w_o_lo_0_1[_lp*2+1 + 0]));
                                w_o_lo_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int w_o_hi_0_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_hi_0_1[_lp*2 + 0], w_o_hi_0_1[_lp*2+1 + 0]));
                                w_o_hi_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            w_x_lo_0_1[0] = w_o_lo_0_bf16_1[0];
                            w_x_hi_0_1[0] = w_o_hi_0_bf16_1[0];
                            w_x_lo_0_1[1] = w_o_lo_0_bf16_1[1];
                            w_x_hi_0_1[1] = w_o_hi_0_bf16_1[1];
                            w_x_lo_0_1[2] = w_o_lo_0_bf16_1[2];
                            w_x_hi_0_1[2] = w_o_hi_0_bf16_1[2];
                            w_x_lo_0_1[3] = w_o_lo_0_bf16_1[3];
                            w_x_hi_0_1[3] = w_o_hi_0_bf16_1[3];
                            w_x_lo_0_1[4] = w_o_lo_0_bf16_1[4];
                            w_x_hi_0_1[4] = w_o_hi_0_bf16_1[4];
                            w_x_lo_0_1[5] = w_o_lo_0_bf16_1[5];
                            w_x_hi_0_1[5] = w_o_hi_0_bf16_1[5];
                            w_x_lo_0_1[6] = w_o_lo_0_bf16_1[6];
                            w_x_hi_0_1[6] = w_o_hi_0_bf16_1[6];
                            w_x_lo_0_1[7] = w_o_lo_0_bf16_1[7];
                            w_x_hi_0_1[7] = w_o_hi_0_bf16_1[7];
                        }
                    }
                    unsigned int w_q_words_0_1[4];
                    float w_q_vals_0_1[16];
                    float _unpack_bf16_58[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_58[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_0_1[_pair]) << 16);
                        (&_unpack_bf16_58[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_0_1[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_0_1[0] = _unpack_bf16_58[0] * w_out_scale_sel_1;
                    w_q_vals_0_1[1] = _unpack_bf16_58[1] * w_out_scale_sel_1;
                    w_q_vals_0_1[2] = _unpack_bf16_58[2] * w_out_scale_sel_1;
                    w_q_vals_0_1[3] = _unpack_bf16_58[3] * w_out_scale_sel_1;
                    w_q_vals_0_1[4] = _unpack_bf16_58[4] * w_out_scale_sel_1;
                    w_q_vals_0_1[5] = _unpack_bf16_58[5] * w_out_scale_sel_1;
                    w_q_vals_0_1[6] = _unpack_bf16_58[6] * w_out_scale_sel_1;
                    w_q_vals_0_1[7] = _unpack_bf16_58[7] * w_out_scale_sel_1;
                    w_q_vals_0_1[8] = _unpack_bf16_58[8] * w_out_scale_sel_1;
                    w_q_vals_0_1[9] = _unpack_bf16_58[9] * w_out_scale_sel_1;
                    w_q_vals_0_1[10] = _unpack_bf16_58[10] * w_out_scale_sel_1;
                    w_q_vals_0_1[11] = _unpack_bf16_58[11] * w_out_scale_sel_1;
                    w_q_vals_0_1[12] = _unpack_bf16_58[12] * w_out_scale_sel_1;
                    w_q_vals_0_1[13] = _unpack_bf16_58[13] * w_out_scale_sel_1;
                    w_q_vals_0_1[14] = _unpack_bf16_58[14] * w_out_scale_sel_1;
                    w_q_vals_0_1[15] = _unpack_bf16_58[15] * w_out_scale_sel_1;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_0_1[0]), "f"(w_q_vals_0_1[1]),
                                               "f"(w_q_vals_0_1[2]), "f"(w_q_vals_0_1[3]));
                        w_q_words_0_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_0_1[4]), "f"(w_q_vals_0_1[5]),
                                               "f"(w_q_vals_0_1[6]), "f"(w_q_vals_0_1[7]));
                        w_q_words_0_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_0_1[8]), "f"(w_q_vals_0_1[9]),
                                               "f"(w_q_vals_0_1[10]), "f"(w_q_vals_0_1[11]));
                        w_q_words_0_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_0_1[12]), "f"(w_q_vals_0_1[13]),
                                               "f"(w_q_vals_0_1[14]), "f"(w_q_vals_0_1[15]));
                        w_q_words_0_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(w_srow_0_1 * 128) + (unsigned int)(w_chunk_lo_0_1 ^ (w_srow_0_1 & 7) << 4)), "r"(w_q_words_0_1[0]), "r"(w_q_words_0_1[1]), "r"(w_q_words_0_1[2]), "r"(w_q_words_0_1[3]) : "memory");
                    float _unpack_bf16_59[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_59[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_0_1[_pair]) << 16);
                        (&_unpack_bf16_59[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_0_1[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_0_1[0] = _unpack_bf16_59[0] * w_out_scale_sel_1;
                    w_q_vals_0_1[1] = _unpack_bf16_59[1] * w_out_scale_sel_1;
                    w_q_vals_0_1[2] = _unpack_bf16_59[2] * w_out_scale_sel_1;
                    w_q_vals_0_1[3] = _unpack_bf16_59[3] * w_out_scale_sel_1;
                    w_q_vals_0_1[4] = _unpack_bf16_59[4] * w_out_scale_sel_1;
                    w_q_vals_0_1[5] = _unpack_bf16_59[5] * w_out_scale_sel_1;
                    w_q_vals_0_1[6] = _unpack_bf16_59[6] * w_out_scale_sel_1;
                    w_q_vals_0_1[7] = _unpack_bf16_59[7] * w_out_scale_sel_1;
                    w_q_vals_0_1[8] = _unpack_bf16_59[8] * w_out_scale_sel_1;
                    w_q_vals_0_1[9] = _unpack_bf16_59[9] * w_out_scale_sel_1;
                    w_q_vals_0_1[10] = _unpack_bf16_59[10] * w_out_scale_sel_1;
                    w_q_vals_0_1[11] = _unpack_bf16_59[11] * w_out_scale_sel_1;
                    w_q_vals_0_1[12] = _unpack_bf16_59[12] * w_out_scale_sel_1;
                    w_q_vals_0_1[13] = _unpack_bf16_59[13] * w_out_scale_sel_1;
                    w_q_vals_0_1[14] = _unpack_bf16_59[14] * w_out_scale_sel_1;
                    w_q_vals_0_1[15] = _unpack_bf16_59[15] * w_out_scale_sel_1;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_0_1[0]), "f"(w_q_vals_0_1[1]),
                                               "f"(w_q_vals_0_1[2]), "f"(w_q_vals_0_1[3]));
                        w_q_words_0_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_0_1[4]), "f"(w_q_vals_0_1[5]),
                                               "f"(w_q_vals_0_1[6]), "f"(w_q_vals_0_1[7]));
                        w_q_words_0_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_0_1[8]), "f"(w_q_vals_0_1[9]),
                                               "f"(w_q_vals_0_1[10]), "f"(w_q_vals_0_1[11]));
                        w_q_words_0_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_0_1[12]), "f"(w_q_vals_0_1[13]),
                                               "f"(w_q_vals_0_1[14]), "f"(w_q_vals_0_1[15]));
                        w_q_words_0_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(w_srow_0_1 * 128) + (unsigned int)(w_chunk_hi_0_1 ^ (w_srow_0_1 & 7) << 4)), "r"(w_q_words_0_1[0]), "r"(w_q_words_0_1[1]), "r"(w_q_words_0_1[2]), "r"(w_q_words_0_1[3]) : "memory");
                }
                __syncwarp();
                int s_row_0_0_1 = (role_tid_1 >> 5) * 8 + lane / 8;
                int s_piece_0_0_1 = lane % 8;
                unsigned int s_words_0_0_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_0_1[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_row_0_0_1 * 128) + (unsigned int)(s_piece_0_0_1 * 16 ^ (s_row_0_0_1 & 7) << 4)));
                int s_dst_0_0_1 = ((tile_m_1 * 256 + s_row_0_0_1) * 56 + w_head_1) * 32 + s_piece_0_0_1 * 4;
                if (tile_m_1 * 256 + s_row_0_0_1 < M) {
                    if (w_kind_1 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_0_0_1)[0] = reinterpret_cast<int4*>(s_words_0_0_1)[0];
                    } else if (w_kind_1 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_0_0_1)[0] = reinterpret_cast<int4*>(s_words_0_0_1)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_0_0_1)[0] = reinterpret_cast<int4*>(s_words_0_0_1)[0];
                    }
                }
                int s_row_0_1_1 = (role_tid_1 >> 5) * 8 + 4 + lane / 8;
                int s_piece_0_1_1 = lane % 8;
                unsigned int s_words_0_1_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_0_1_1[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_row_0_1_1 * 128) + (unsigned int)(s_piece_0_1_1 * 16 ^ (s_row_0_1_1 & 7) << 4)));
                int s_dst_0_1_1 = ((tile_m_1 * 256 + s_row_0_1_1) * 56 + w_head_1) * 32 + s_piece_0_1_1 * 4;
                if (tile_m_1 * 256 + s_row_0_1_1 < M) {
                    if (w_kind_1 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_0_1_1)[0] = reinterpret_cast<int4*>(s_words_0_1_1)[0];
                    } else if (w_kind_1 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_0_1_1)[0] = reinterpret_cast<int4*>(s_words_0_1_1)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_0_1_1)[0] = reinterpret_cast<int4*>(s_words_0_1_1)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_row_lo_1_1 = tile_m_1 * 256 + 64 + warp_m_1 * 16 + (lane >> 2);
                int w_row_hi_1_1 = w_row_lo_1_1 + 8;
                float w_sa_lo_1_1 = ((w_row_lo_1_1 < M) ? act_scale[w_row_lo_1_1] : 0.0f);
                float w_sa_hi_1_1 = ((w_row_hi_1_1 < M) ? act_scale[w_row_hi_1_1] : 0.0f);
                float w_rounded_1_1[32];
                float w_ss_lo_1_1 = 0.0f;
                float w_ss_hi_1_1 = 0.0f;
                float w_scaled_1_0_1[4];
                w_scaled_1_0_1[0] = accum_1[32] * w_sa_lo_1_1 * sw_1[0];
                w_scaled_1_0_1[1] = accum_1[33] * w_sa_lo_1_1 * sw_1[1];
                w_scaled_1_0_1[2] = accum_1[34] * w_sa_hi_1_1 * sw_1[0];
                w_scaled_1_0_1[3] = accum_1[35] * w_sa_hi_1_1 * sw_1[1];
                unsigned int w_scaled_1_0_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_0_1[_lp*2 + 0], w_scaled_1_0_1[_lp*2+1 + 0]));
                    w_scaled_1_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_60[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_60[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_0_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_60[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_0_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_1_1[0] = _unpack_bf16_60[0];
                w_rounded_1_1[1] = _unpack_bf16_60[1];
                w_rounded_1_1[2] = _unpack_bf16_60[2];
                w_rounded_1_1[3] = _unpack_bf16_60[3];
                w_ss_lo_1_1 += __fmaf_rn(_unpack_bf16_60[0], _unpack_bf16_60[0], _unpack_bf16_60[1] * _unpack_bf16_60[1]);
                w_ss_hi_1_1 += __fmaf_rn(_unpack_bf16_60[2], _unpack_bf16_60[2], _unpack_bf16_60[3] * _unpack_bf16_60[3]);
                float w_scaled_1_1_1[4];
                w_scaled_1_1_1[0] = accum_1[36] * w_sa_lo_1_1 * sw_1[2];
                w_scaled_1_1_1[1] = accum_1[37] * w_sa_lo_1_1 * sw_1[3];
                w_scaled_1_1_1[2] = accum_1[38] * w_sa_hi_1_1 * sw_1[2];
                w_scaled_1_1_1[3] = accum_1[39] * w_sa_hi_1_1 * sw_1[3];
                unsigned int w_scaled_1_1_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_1_1[_lp*2 + 0], w_scaled_1_1_1[_lp*2+1 + 0]));
                    w_scaled_1_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_61[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_61[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_1_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_61[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_1_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_1_1[4] = _unpack_bf16_61[0];
                w_rounded_1_1[5] = _unpack_bf16_61[1];
                w_rounded_1_1[6] = _unpack_bf16_61[2];
                w_rounded_1_1[7] = _unpack_bf16_61[3];
                w_ss_lo_1_1 += __fmaf_rn(_unpack_bf16_61[0], _unpack_bf16_61[0], _unpack_bf16_61[1] * _unpack_bf16_61[1]);
                w_ss_hi_1_1 += __fmaf_rn(_unpack_bf16_61[2], _unpack_bf16_61[2], _unpack_bf16_61[3] * _unpack_bf16_61[3]);
                float w_scaled_1_2_1[4];
                w_scaled_1_2_1[0] = accum_1[40] * w_sa_lo_1_1 * sw_1[4];
                w_scaled_1_2_1[1] = accum_1[41] * w_sa_lo_1_1 * sw_1[5];
                w_scaled_1_2_1[2] = accum_1[42] * w_sa_hi_1_1 * sw_1[4];
                w_scaled_1_2_1[3] = accum_1[43] * w_sa_hi_1_1 * sw_1[5];
                unsigned int w_scaled_1_2_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_2_1[_lp*2 + 0], w_scaled_1_2_1[_lp*2+1 + 0]));
                    w_scaled_1_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_62[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_62[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_2_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_62[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_2_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_1_1[8] = _unpack_bf16_62[0];
                w_rounded_1_1[9] = _unpack_bf16_62[1];
                w_rounded_1_1[10] = _unpack_bf16_62[2];
                w_rounded_1_1[11] = _unpack_bf16_62[3];
                w_ss_lo_1_1 += __fmaf_rn(_unpack_bf16_62[0], _unpack_bf16_62[0], _unpack_bf16_62[1] * _unpack_bf16_62[1]);
                w_ss_hi_1_1 += __fmaf_rn(_unpack_bf16_62[2], _unpack_bf16_62[2], _unpack_bf16_62[3] * _unpack_bf16_62[3]);
                float w_scaled_1_3_1[4];
                w_scaled_1_3_1[0] = accum_1[44] * w_sa_lo_1_1 * sw_1[6];
                w_scaled_1_3_1[1] = accum_1[45] * w_sa_lo_1_1 * sw_1[7];
                w_scaled_1_3_1[2] = accum_1[46] * w_sa_hi_1_1 * sw_1[6];
                w_scaled_1_3_1[3] = accum_1[47] * w_sa_hi_1_1 * sw_1[7];
                unsigned int w_scaled_1_3_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_3_1[_lp*2 + 0], w_scaled_1_3_1[_lp*2+1 + 0]));
                    w_scaled_1_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_63[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_63[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_3_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_63[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_3_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_1_1[12] = _unpack_bf16_63[0];
                w_rounded_1_1[13] = _unpack_bf16_63[1];
                w_rounded_1_1[14] = _unpack_bf16_63[2];
                w_rounded_1_1[15] = _unpack_bf16_63[3];
                w_ss_lo_1_1 += __fmaf_rn(_unpack_bf16_63[0], _unpack_bf16_63[0], _unpack_bf16_63[1] * _unpack_bf16_63[1]);
                w_ss_hi_1_1 += __fmaf_rn(_unpack_bf16_63[2], _unpack_bf16_63[2], _unpack_bf16_63[3] * _unpack_bf16_63[3]);
                float w_scaled_1_4_1[4];
                w_scaled_1_4_1[0] = accum_1[48] * w_sa_lo_1_1 * sw_1[8];
                w_scaled_1_4_1[1] = accum_1[49] * w_sa_lo_1_1 * sw_1[9];
                w_scaled_1_4_1[2] = accum_1[50] * w_sa_hi_1_1 * sw_1[8];
                w_scaled_1_4_1[3] = accum_1[51] * w_sa_hi_1_1 * sw_1[9];
                unsigned int w_scaled_1_4_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_4_1[_lp*2 + 0], w_scaled_1_4_1[_lp*2+1 + 0]));
                    w_scaled_1_4_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_64[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_64[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_4_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_64[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_4_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_1_1[16] = _unpack_bf16_64[0];
                w_rounded_1_1[17] = _unpack_bf16_64[1];
                w_rounded_1_1[18] = _unpack_bf16_64[2];
                w_rounded_1_1[19] = _unpack_bf16_64[3];
                w_ss_lo_1_1 += __fmaf_rn(_unpack_bf16_64[0], _unpack_bf16_64[0], _unpack_bf16_64[1] * _unpack_bf16_64[1]);
                w_ss_hi_1_1 += __fmaf_rn(_unpack_bf16_64[2], _unpack_bf16_64[2], _unpack_bf16_64[3] * _unpack_bf16_64[3]);
                float w_scaled_1_5_1[4];
                w_scaled_1_5_1[0] = accum_1[52] * w_sa_lo_1_1 * sw_1[10];
                w_scaled_1_5_1[1] = accum_1[53] * w_sa_lo_1_1 * sw_1[11];
                w_scaled_1_5_1[2] = accum_1[54] * w_sa_hi_1_1 * sw_1[10];
                w_scaled_1_5_1[3] = accum_1[55] * w_sa_hi_1_1 * sw_1[11];
                unsigned int w_scaled_1_5_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_5_1[_lp*2 + 0], w_scaled_1_5_1[_lp*2+1 + 0]));
                    w_scaled_1_5_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_65[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_65[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_5_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_65[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_5_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_1_1[20] = _unpack_bf16_65[0];
                w_rounded_1_1[21] = _unpack_bf16_65[1];
                w_rounded_1_1[22] = _unpack_bf16_65[2];
                w_rounded_1_1[23] = _unpack_bf16_65[3];
                w_ss_lo_1_1 += __fmaf_rn(_unpack_bf16_65[0], _unpack_bf16_65[0], _unpack_bf16_65[1] * _unpack_bf16_65[1]);
                w_ss_hi_1_1 += __fmaf_rn(_unpack_bf16_65[2], _unpack_bf16_65[2], _unpack_bf16_65[3] * _unpack_bf16_65[3]);
                float w_scaled_1_6_1[4];
                w_scaled_1_6_1[0] = accum_1[56] * w_sa_lo_1_1 * sw_1[12];
                w_scaled_1_6_1[1] = accum_1[57] * w_sa_lo_1_1 * sw_1[13];
                w_scaled_1_6_1[2] = accum_1[58] * w_sa_hi_1_1 * sw_1[12];
                w_scaled_1_6_1[3] = accum_1[59] * w_sa_hi_1_1 * sw_1[13];
                unsigned int w_scaled_1_6_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_6_1[_lp*2 + 0], w_scaled_1_6_1[_lp*2+1 + 0]));
                    w_scaled_1_6_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_66[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_66[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_6_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_66[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_6_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_1_1[24] = _unpack_bf16_66[0];
                w_rounded_1_1[25] = _unpack_bf16_66[1];
                w_rounded_1_1[26] = _unpack_bf16_66[2];
                w_rounded_1_1[27] = _unpack_bf16_66[3];
                w_ss_lo_1_1 += __fmaf_rn(_unpack_bf16_66[0], _unpack_bf16_66[0], _unpack_bf16_66[1] * _unpack_bf16_66[1]);
                w_ss_hi_1_1 += __fmaf_rn(_unpack_bf16_66[2], _unpack_bf16_66[2], _unpack_bf16_66[3] * _unpack_bf16_66[3]);
                float w_scaled_1_7_1[4];
                w_scaled_1_7_1[0] = accum_1[60] * w_sa_lo_1_1 * sw_1[14];
                w_scaled_1_7_1[1] = accum_1[61] * w_sa_lo_1_1 * sw_1[15];
                w_scaled_1_7_1[2] = accum_1[62] * w_sa_hi_1_1 * sw_1[14];
                w_scaled_1_7_1[3] = accum_1[63] * w_sa_hi_1_1 * sw_1[15];
                unsigned int w_scaled_1_7_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_1_7_1[_lp*2 + 0], w_scaled_1_7_1[_lp*2+1 + 0]));
                    w_scaled_1_7_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_67[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_67[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_7_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_67[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_1_7_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_1_1[28] = _unpack_bf16_67[0];
                w_rounded_1_1[29] = _unpack_bf16_67[1];
                w_rounded_1_1[30] = _unpack_bf16_67[2];
                w_rounded_1_1[31] = _unpack_bf16_67[3];
                w_ss_lo_1_1 += __fmaf_rn(_unpack_bf16_67[0], _unpack_bf16_67[0], _unpack_bf16_67[1] * _unpack_bf16_67[1]);
                w_ss_hi_1_1 += __fmaf_rn(_unpack_bf16_67[2], _unpack_bf16_67[2], _unpack_bf16_67[3] * _unpack_bf16_67[3]);
                float w_rstd_lo_1_1 = 1.0f;
                float w_rstd_hi_1_1 = 1.0f;
                if (w_kind_1 < 2) {
                    float _shfl_xor_sync_20_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_1_1, 1);
                    w_ss_lo_1_1 += _shfl_xor_sync_20_1;
                    float _shfl_xor_sync_21_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_1_1, 2);
                    w_ss_lo_1_1 += _shfl_xor_sync_21_1;
                    float _shfl_xor_sync_22_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_1_1, 1);
                    w_ss_hi_1_1 += _shfl_xor_sync_22_1;
                    float _shfl_xor_sync_23_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_1_1, 2);
                    w_ss_hi_1_1 += _shfl_xor_sync_23_1;
                    int w_srow_lo_1_1 = warp_m_1 * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n_1 * 64 + w_srow_lo_1_1] = w_ss_lo_1_1;
                        rowstat[warp_n_1 * 64 + w_srow_lo_1_1 + 8] = w_ss_hi_1_1;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float w_other_lo_1_1 = rowstat[(1 - warp_n_1) * 64 + w_srow_lo_1_1];
                    float w_other_hi_1_1 = rowstat[(1 - warp_n_1) * 64 + w_srow_lo_1_1 + 8];
                    float _fdiv_25_1 = __fdiv_rn(w_ss_lo_1_1 + w_other_lo_1_1, 128.0f);
                    float _rsqrt_24_1 = rsqrtf(_fdiv_25_1 + eps);
                    w_rstd_lo_1_1 = _rsqrt_24_1;
                    float _fdiv_27_1 = __fdiv_rn(w_ss_hi_1_1 + w_other_hi_1_1, 128.0f);
                    float _rsqrt_26_1 = rsqrtf(_fdiv_27_1 + eps);
                    w_rstd_hi_1_1 = _rsqrt_26_1;
                }
                int w_srow_st_1_1 = warp_m_1 * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float w_normed_1_0_1[4];
                if (w_kind_1 < 2) {
                    w_normed_1_0_1[0] = w_rounded_1_1[0] * w_rstd_lo_1_1 * nw_1[0];
                    w_normed_1_0_1[1] = w_rounded_1_1[1] * w_rstd_lo_1_1 * nw_1[1];
                    w_normed_1_0_1[2] = w_rounded_1_1[2] * w_rstd_hi_1_1 * nw_1[0];
                    w_normed_1_0_1[3] = w_rounded_1_1[3] * w_rstd_hi_1_1 * nw_1[1];
                } else {
                    w_normed_1_0_1[0] = w_rounded_1_1[0];
                    w_normed_1_0_1[1] = w_rounded_1_1[1];
                    w_normed_1_0_1[2] = w_rounded_1_1[2];
                    w_normed_1_0_1[3] = w_rounded_1_1[3];
                }
                unsigned int w_normed_1_0_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_0_1[_lp*2 + 0], w_normed_1_0_1[_lp*2+1 + 0]));
                    w_normed_1_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_28 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_1_1 * 256) + (unsigned int)(warp_n_1 * 64 * 2 ^ (w_srow_st_1_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_0_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_0_bf16_1[1]))
                    : "memory");
                float w_normed_1_1_1[4];
                if (w_kind_1 < 2) {
                    w_normed_1_1_1[0] = w_rounded_1_1[4] * w_rstd_lo_1_1 * nw_1[2];
                    w_normed_1_1_1[1] = w_rounded_1_1[5] * w_rstd_lo_1_1 * nw_1[3];
                    w_normed_1_1_1[2] = w_rounded_1_1[6] * w_rstd_hi_1_1 * nw_1[2];
                    w_normed_1_1_1[3] = w_rounded_1_1[7] * w_rstd_hi_1_1 * nw_1[3];
                } else {
                    w_normed_1_1_1[0] = w_rounded_1_1[4];
                    w_normed_1_1_1[1] = w_rounded_1_1[5];
                    w_normed_1_1_1[2] = w_rounded_1_1[6];
                    w_normed_1_1_1[3] = w_rounded_1_1[7];
                }
                unsigned int w_normed_1_1_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_1_1[_lp*2 + 0], w_normed_1_1_1[_lp*2+1 + 0]));
                    w_normed_1_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_29 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_1_1 * 256) + (unsigned int)((warp_n_1 * 64 + 8) * 2 ^ (w_srow_st_1_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_1_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_1_bf16_1[1]))
                    : "memory");
                float w_normed_1_2_1[4];
                if (w_kind_1 < 2) {
                    w_normed_1_2_1[0] = w_rounded_1_1[8] * w_rstd_lo_1_1 * nw_1[4];
                    w_normed_1_2_1[1] = w_rounded_1_1[9] * w_rstd_lo_1_1 * nw_1[5];
                    w_normed_1_2_1[2] = w_rounded_1_1[10] * w_rstd_hi_1_1 * nw_1[4];
                    w_normed_1_2_1[3] = w_rounded_1_1[11] * w_rstd_hi_1_1 * nw_1[5];
                } else {
                    w_normed_1_2_1[0] = w_rounded_1_1[8];
                    w_normed_1_2_1[1] = w_rounded_1_1[9];
                    w_normed_1_2_1[2] = w_rounded_1_1[10];
                    w_normed_1_2_1[3] = w_rounded_1_1[11];
                }
                unsigned int w_normed_1_2_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_2_1[_lp*2 + 0], w_normed_1_2_1[_lp*2+1 + 0]));
                    w_normed_1_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_30 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_1_1 * 256) + (unsigned int)((warp_n_1 * 64 + 16) * 2 ^ (w_srow_st_1_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_30), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_2_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_2_bf16_1[1]))
                    : "memory");
                float w_normed_1_3_1[4];
                if (w_kind_1 < 2) {
                    w_normed_1_3_1[0] = w_rounded_1_1[12] * w_rstd_lo_1_1 * nw_1[6];
                    w_normed_1_3_1[1] = w_rounded_1_1[13] * w_rstd_lo_1_1 * nw_1[7];
                    w_normed_1_3_1[2] = w_rounded_1_1[14] * w_rstd_hi_1_1 * nw_1[6];
                    w_normed_1_3_1[3] = w_rounded_1_1[15] * w_rstd_hi_1_1 * nw_1[7];
                } else {
                    w_normed_1_3_1[0] = w_rounded_1_1[12];
                    w_normed_1_3_1[1] = w_rounded_1_1[13];
                    w_normed_1_3_1[2] = w_rounded_1_1[14];
                    w_normed_1_3_1[3] = w_rounded_1_1[15];
                }
                unsigned int w_normed_1_3_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_3_1[_lp*2 + 0], w_normed_1_3_1[_lp*2+1 + 0]));
                    w_normed_1_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_31 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_1_1 * 256) + (unsigned int)((warp_n_1 * 64 + 24) * 2 ^ (w_srow_st_1_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_31), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_3_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_3_bf16_1[1]))
                    : "memory");
                float w_normed_1_4_1[4];
                if (w_kind_1 < 2) {
                    w_normed_1_4_1[0] = w_rounded_1_1[16] * w_rstd_lo_1_1 * nw_1[8];
                    w_normed_1_4_1[1] = w_rounded_1_1[17] * w_rstd_lo_1_1 * nw_1[9];
                    w_normed_1_4_1[2] = w_rounded_1_1[18] * w_rstd_hi_1_1 * nw_1[8];
                    w_normed_1_4_1[3] = w_rounded_1_1[19] * w_rstd_hi_1_1 * nw_1[9];
                } else {
                    w_normed_1_4_1[0] = w_rounded_1_1[16];
                    w_normed_1_4_1[1] = w_rounded_1_1[17];
                    w_normed_1_4_1[2] = w_rounded_1_1[18];
                    w_normed_1_4_1[3] = w_rounded_1_1[19];
                }
                unsigned int w_normed_1_4_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_4_1[_lp*2 + 0], w_normed_1_4_1[_lp*2+1 + 0]));
                    w_normed_1_4_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_32 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_1_1 * 256) + (unsigned int)((warp_n_1 * 64 + 32) * 2 ^ (w_srow_st_1_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_32), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_4_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_4_bf16_1[1]))
                    : "memory");
                float w_normed_1_5_1[4];
                if (w_kind_1 < 2) {
                    w_normed_1_5_1[0] = w_rounded_1_1[20] * w_rstd_lo_1_1 * nw_1[10];
                    w_normed_1_5_1[1] = w_rounded_1_1[21] * w_rstd_lo_1_1 * nw_1[11];
                    w_normed_1_5_1[2] = w_rounded_1_1[22] * w_rstd_hi_1_1 * nw_1[10];
                    w_normed_1_5_1[3] = w_rounded_1_1[23] * w_rstd_hi_1_1 * nw_1[11];
                } else {
                    w_normed_1_5_1[0] = w_rounded_1_1[20];
                    w_normed_1_5_1[1] = w_rounded_1_1[21];
                    w_normed_1_5_1[2] = w_rounded_1_1[22];
                    w_normed_1_5_1[3] = w_rounded_1_1[23];
                }
                unsigned int w_normed_1_5_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_5_1[_lp*2 + 0], w_normed_1_5_1[_lp*2+1 + 0]));
                    w_normed_1_5_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_33 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_1_1 * 256) + (unsigned int)((warp_n_1 * 64 + 40) * 2 ^ (w_srow_st_1_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_33), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_5_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_5_bf16_1[1]))
                    : "memory");
                float w_normed_1_6_1[4];
                if (w_kind_1 < 2) {
                    w_normed_1_6_1[0] = w_rounded_1_1[24] * w_rstd_lo_1_1 * nw_1[12];
                    w_normed_1_6_1[1] = w_rounded_1_1[25] * w_rstd_lo_1_1 * nw_1[13];
                    w_normed_1_6_1[2] = w_rounded_1_1[26] * w_rstd_hi_1_1 * nw_1[12];
                    w_normed_1_6_1[3] = w_rounded_1_1[27] * w_rstd_hi_1_1 * nw_1[13];
                } else {
                    w_normed_1_6_1[0] = w_rounded_1_1[24];
                    w_normed_1_6_1[1] = w_rounded_1_1[25];
                    w_normed_1_6_1[2] = w_rounded_1_1[26];
                    w_normed_1_6_1[3] = w_rounded_1_1[27];
                }
                unsigned int w_normed_1_6_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_6_1[_lp*2 + 0], w_normed_1_6_1[_lp*2+1 + 0]));
                    w_normed_1_6_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_1_1 * 256) + (unsigned int)((warp_n_1 * 64 + 48) * 2 ^ (w_srow_st_1_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_6_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_6_bf16_1[1]))
                    : "memory");
                float w_normed_1_7_1[4];
                if (w_kind_1 < 2) {
                    w_normed_1_7_1[0] = w_rounded_1_1[28] * w_rstd_lo_1_1 * nw_1[14];
                    w_normed_1_7_1[1] = w_rounded_1_1[29] * w_rstd_lo_1_1 * nw_1[15];
                    w_normed_1_7_1[2] = w_rounded_1_1[30] * w_rstd_hi_1_1 * nw_1[14];
                    w_normed_1_7_1[3] = w_rounded_1_1[31] * w_rstd_hi_1_1 * nw_1[15];
                } else {
                    w_normed_1_7_1[0] = w_rounded_1_1[28];
                    w_normed_1_7_1[1] = w_rounded_1_1[29];
                    w_normed_1_7_1[2] = w_rounded_1_1[30];
                    w_normed_1_7_1[3] = w_rounded_1_1[31];
                }
                unsigned int w_normed_1_7_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_1_7_1[_lp*2 + 0], w_normed_1_7_1[_lp*2+1 + 0]));
                    w_normed_1_7_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_1_1 * 256) + (unsigned int)((warp_n_1 * 64 + 56) * 2 ^ (w_srow_st_1_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_7_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_1_7_bf16_1[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_srow_1_1 = role_tid_1 >> 2;
                int w_c_1_1 = role_tid_1 & 3;
                int w_grow_1_1 = tile_m_1 * 256 + 64 + w_srow_1_1;
                int w_chunk_lo_1_1 = ((w_c_1_1 < 3) ? w_c_1_1 * 16 : 96);
                int w_chunk_hi_1_1 = ((w_c_1_1 < 3) ? w_chunk_lo_1_1 + 48 : 112);
                if (w_grow_1_1 < M) {
                    unsigned int w_x_lo_1_1[8];
                    unsigned int w_x_hi_1_1[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1_1[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_1_1 * 256) + (unsigned int)(w_chunk_lo_1_1 * 2 ^ (w_srow_1_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_1_1[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_1_1 * 256) + (unsigned int)(w_chunk_lo_1_1 * 2 + 16 ^ (w_srow_1_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1_1[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_1_1 * 256) + (unsigned int)(w_chunk_hi_1_1 * 2 ^ (w_srow_1_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_1_1[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_1_1 * 256) + (unsigned int)(w_chunk_hi_1_1 * 2 + 16 ^ (w_srow_1_1 & 7) << 4)));
                    if (w_c_1_1 < 3) {
                        if (w_kind_1 < 2) {
                            float _unpack_bf16_68[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_68[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_1_1[_pair]) << 16);
                                (&_unpack_bf16_68[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_1_1[_pair]) & 0xffff0000u);
                            }
                            float _unpack_bf16_69[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_69[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_1_1[_pair]) << 16);
                                (&_unpack_bf16_69[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_1_1[_pair]) & 0xffff0000u);
                            }
                            float _vec_load_20[8];
                            {
                                const uint4* _vptr_36 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_1_1 * 96 + w_chunk_lo_1_1) + 0);
                                uint4 _vld_36[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_36[_blk] = _vptr_36[_blk];
                                    uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_20[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) << 16);
                                        (&_vec_load_20[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_21[8];
                            {
                                const uint4* _vptr_37 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_1_1 * 96 + w_chunk_lo_1_1 + 8) + 0);
                                uint4 _vld_37[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_37[_blk] = _vptr_37[_blk];
                                    uint32_t* _vpairs_37 = reinterpret_cast<uint32_t*>(&_vld_37[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_21[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) << 16);
                                        (&_vec_load_21[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_22[8];
                            {
                                const uint4* _vptr_38 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_1_1 * 96 + 48 + w_chunk_lo_1_1) + 0);
                                uint4 _vld_38[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_38[_blk] = _vptr_38[_blk];
                                    uint32_t* _vpairs_38 = reinterpret_cast<uint32_t*>(&_vld_38[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_22[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) << 16);
                                        (&_vec_load_22[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_23[8];
                            {
                                const uint4* _vptr_39 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_1_1 * 96 + 48 + w_chunk_lo_1_1 + 8) + 0);
                                uint4 _vld_39[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_39[_blk] = _vptr_39[_blk];
                                    uint32_t* _vpairs_39 = reinterpret_cast<uint32_t*>(&_vld_39[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_23[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) << 16);
                                        (&_vec_load_23[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float w_o_lo_1_1[16];
                            float w_o_hi_1_1[16];
                            float w_cv0_1_0_1 = _vec_load_20[0];
                            float w_sv0_1_0_1 = _vec_load_22[0];
                            float w_cv1_1_0_1 = _vec_load_21[0];
                            float w_sv1_1_0_1 = _vec_load_23[0];
                            w_o_lo_1_1[0] = __fmaf_rn(_unpack_bf16_68[0], w_cv0_1_0_1, -(_unpack_bf16_69[0] * w_sv0_1_0_1));
                            w_o_hi_1_1[0] = __fmaf_rn(_unpack_bf16_69[0], w_cv0_1_0_1, _unpack_bf16_68[0] * w_sv0_1_0_1);
                            w_o_lo_1_1[8] = __fmaf_rn(_unpack_bf16_68[8], w_cv1_1_0_1, -(_unpack_bf16_69[8] * w_sv1_1_0_1));
                            w_o_hi_1_1[8] = __fmaf_rn(_unpack_bf16_69[8], w_cv1_1_0_1, _unpack_bf16_68[8] * w_sv1_1_0_1);
                            float w_cv0_1_1_1 = _vec_load_20[1];
                            float w_sv0_1_1_1 = _vec_load_22[1];
                            float w_cv1_1_1_1 = _vec_load_21[1];
                            float w_sv1_1_1_1 = _vec_load_23[1];
                            w_o_lo_1_1[1] = __fmaf_rn(_unpack_bf16_68[1], w_cv0_1_1_1, -(_unpack_bf16_69[1] * w_sv0_1_1_1));
                            w_o_hi_1_1[1] = __fmaf_rn(_unpack_bf16_69[1], w_cv0_1_1_1, _unpack_bf16_68[1] * w_sv0_1_1_1);
                            w_o_lo_1_1[9] = __fmaf_rn(_unpack_bf16_68[9], w_cv1_1_1_1, -(_unpack_bf16_69[9] * w_sv1_1_1_1));
                            w_o_hi_1_1[9] = __fmaf_rn(_unpack_bf16_69[9], w_cv1_1_1_1, _unpack_bf16_68[9] * w_sv1_1_1_1);
                            float w_cv0_1_2_1 = _vec_load_20[2];
                            float w_sv0_1_2_1 = _vec_load_22[2];
                            float w_cv1_1_2_1 = _vec_load_21[2];
                            float w_sv1_1_2_1 = _vec_load_23[2];
                            w_o_lo_1_1[2] = __fmaf_rn(_unpack_bf16_68[2], w_cv0_1_2_1, -(_unpack_bf16_69[2] * w_sv0_1_2_1));
                            w_o_hi_1_1[2] = __fmaf_rn(_unpack_bf16_69[2], w_cv0_1_2_1, _unpack_bf16_68[2] * w_sv0_1_2_1);
                            w_o_lo_1_1[10] = __fmaf_rn(_unpack_bf16_68[10], w_cv1_1_2_1, -(_unpack_bf16_69[10] * w_sv1_1_2_1));
                            w_o_hi_1_1[10] = __fmaf_rn(_unpack_bf16_69[10], w_cv1_1_2_1, _unpack_bf16_68[10] * w_sv1_1_2_1);
                            float w_cv0_1_3_1 = _vec_load_20[3];
                            float w_sv0_1_3_1 = _vec_load_22[3];
                            float w_cv1_1_3_1 = _vec_load_21[3];
                            float w_sv1_1_3_1 = _vec_load_23[3];
                            w_o_lo_1_1[3] = __fmaf_rn(_unpack_bf16_68[3], w_cv0_1_3_1, -(_unpack_bf16_69[3] * w_sv0_1_3_1));
                            w_o_hi_1_1[3] = __fmaf_rn(_unpack_bf16_69[3], w_cv0_1_3_1, _unpack_bf16_68[3] * w_sv0_1_3_1);
                            w_o_lo_1_1[11] = __fmaf_rn(_unpack_bf16_68[11], w_cv1_1_3_1, -(_unpack_bf16_69[11] * w_sv1_1_3_1));
                            w_o_hi_1_1[11] = __fmaf_rn(_unpack_bf16_69[11], w_cv1_1_3_1, _unpack_bf16_68[11] * w_sv1_1_3_1);
                            float w_cv0_1_4_1 = _vec_load_20[4];
                            float w_sv0_1_4_1 = _vec_load_22[4];
                            float w_cv1_1_4_1 = _vec_load_21[4];
                            float w_sv1_1_4_1 = _vec_load_23[4];
                            w_o_lo_1_1[4] = __fmaf_rn(_unpack_bf16_68[4], w_cv0_1_4_1, -(_unpack_bf16_69[4] * w_sv0_1_4_1));
                            w_o_hi_1_1[4] = __fmaf_rn(_unpack_bf16_69[4], w_cv0_1_4_1, _unpack_bf16_68[4] * w_sv0_1_4_1);
                            w_o_lo_1_1[12] = __fmaf_rn(_unpack_bf16_68[12], w_cv1_1_4_1, -(_unpack_bf16_69[12] * w_sv1_1_4_1));
                            w_o_hi_1_1[12] = __fmaf_rn(_unpack_bf16_69[12], w_cv1_1_4_1, _unpack_bf16_68[12] * w_sv1_1_4_1);
                            float w_cv0_1_5_1 = _vec_load_20[5];
                            float w_sv0_1_5_1 = _vec_load_22[5];
                            float w_cv1_1_5_1 = _vec_load_21[5];
                            float w_sv1_1_5_1 = _vec_load_23[5];
                            w_o_lo_1_1[5] = __fmaf_rn(_unpack_bf16_68[5], w_cv0_1_5_1, -(_unpack_bf16_69[5] * w_sv0_1_5_1));
                            w_o_hi_1_1[5] = __fmaf_rn(_unpack_bf16_69[5], w_cv0_1_5_1, _unpack_bf16_68[5] * w_sv0_1_5_1);
                            w_o_lo_1_1[13] = __fmaf_rn(_unpack_bf16_68[13], w_cv1_1_5_1, -(_unpack_bf16_69[13] * w_sv1_1_5_1));
                            w_o_hi_1_1[13] = __fmaf_rn(_unpack_bf16_69[13], w_cv1_1_5_1, _unpack_bf16_68[13] * w_sv1_1_5_1);
                            float w_cv0_1_6_1 = _vec_load_20[6];
                            float w_sv0_1_6_1 = _vec_load_22[6];
                            float w_cv1_1_6_1 = _vec_load_21[6];
                            float w_sv1_1_6_1 = _vec_load_23[6];
                            w_o_lo_1_1[6] = __fmaf_rn(_unpack_bf16_68[6], w_cv0_1_6_1, -(_unpack_bf16_69[6] * w_sv0_1_6_1));
                            w_o_hi_1_1[6] = __fmaf_rn(_unpack_bf16_69[6], w_cv0_1_6_1, _unpack_bf16_68[6] * w_sv0_1_6_1);
                            w_o_lo_1_1[14] = __fmaf_rn(_unpack_bf16_68[14], w_cv1_1_6_1, -(_unpack_bf16_69[14] * w_sv1_1_6_1));
                            w_o_hi_1_1[14] = __fmaf_rn(_unpack_bf16_69[14], w_cv1_1_6_1, _unpack_bf16_68[14] * w_sv1_1_6_1);
                            float w_cv0_1_7_1 = _vec_load_20[7];
                            float w_sv0_1_7_1 = _vec_load_22[7];
                            float w_cv1_1_7_1 = _vec_load_21[7];
                            float w_sv1_1_7_1 = _vec_load_23[7];
                            w_o_lo_1_1[7] = __fmaf_rn(_unpack_bf16_68[7], w_cv0_1_7_1, -(_unpack_bf16_69[7] * w_sv0_1_7_1));
                            w_o_hi_1_1[7] = __fmaf_rn(_unpack_bf16_69[7], w_cv0_1_7_1, _unpack_bf16_68[7] * w_sv0_1_7_1);
                            w_o_lo_1_1[15] = __fmaf_rn(_unpack_bf16_68[15], w_cv1_1_7_1, -(_unpack_bf16_69[15] * w_sv1_1_7_1));
                            w_o_hi_1_1[15] = __fmaf_rn(_unpack_bf16_69[15], w_cv1_1_7_1, _unpack_bf16_68[15] * w_sv1_1_7_1);
                            unsigned int w_o_lo_1_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_lo_1_1[_lp*2 + 0], w_o_lo_1_1[_lp*2+1 + 0]));
                                w_o_lo_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int w_o_hi_1_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_hi_1_1[_lp*2 + 0], w_o_hi_1_1[_lp*2+1 + 0]));
                                w_o_hi_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            w_x_lo_1_1[0] = w_o_lo_1_bf16_1[0];
                            w_x_hi_1_1[0] = w_o_hi_1_bf16_1[0];
                            w_x_lo_1_1[1] = w_o_lo_1_bf16_1[1];
                            w_x_hi_1_1[1] = w_o_hi_1_bf16_1[1];
                            w_x_lo_1_1[2] = w_o_lo_1_bf16_1[2];
                            w_x_hi_1_1[2] = w_o_hi_1_bf16_1[2];
                            w_x_lo_1_1[3] = w_o_lo_1_bf16_1[3];
                            w_x_hi_1_1[3] = w_o_hi_1_bf16_1[3];
                            w_x_lo_1_1[4] = w_o_lo_1_bf16_1[4];
                            w_x_hi_1_1[4] = w_o_hi_1_bf16_1[4];
                            w_x_lo_1_1[5] = w_o_lo_1_bf16_1[5];
                            w_x_hi_1_1[5] = w_o_hi_1_bf16_1[5];
                            w_x_lo_1_1[6] = w_o_lo_1_bf16_1[6];
                            w_x_hi_1_1[6] = w_o_hi_1_bf16_1[6];
                            w_x_lo_1_1[7] = w_o_lo_1_bf16_1[7];
                            w_x_hi_1_1[7] = w_o_hi_1_bf16_1[7];
                        }
                    }
                    unsigned int w_q_words_1_1[4];
                    float w_q_vals_1_1[16];
                    float _unpack_bf16_70[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_70[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_1_1[_pair]) << 16);
                        (&_unpack_bf16_70[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_1_1[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_1_1[0] = _unpack_bf16_70[0] * w_out_scale_sel_1;
                    w_q_vals_1_1[1] = _unpack_bf16_70[1] * w_out_scale_sel_1;
                    w_q_vals_1_1[2] = _unpack_bf16_70[2] * w_out_scale_sel_1;
                    w_q_vals_1_1[3] = _unpack_bf16_70[3] * w_out_scale_sel_1;
                    w_q_vals_1_1[4] = _unpack_bf16_70[4] * w_out_scale_sel_1;
                    w_q_vals_1_1[5] = _unpack_bf16_70[5] * w_out_scale_sel_1;
                    w_q_vals_1_1[6] = _unpack_bf16_70[6] * w_out_scale_sel_1;
                    w_q_vals_1_1[7] = _unpack_bf16_70[7] * w_out_scale_sel_1;
                    w_q_vals_1_1[8] = _unpack_bf16_70[8] * w_out_scale_sel_1;
                    w_q_vals_1_1[9] = _unpack_bf16_70[9] * w_out_scale_sel_1;
                    w_q_vals_1_1[10] = _unpack_bf16_70[10] * w_out_scale_sel_1;
                    w_q_vals_1_1[11] = _unpack_bf16_70[11] * w_out_scale_sel_1;
                    w_q_vals_1_1[12] = _unpack_bf16_70[12] * w_out_scale_sel_1;
                    w_q_vals_1_1[13] = _unpack_bf16_70[13] * w_out_scale_sel_1;
                    w_q_vals_1_1[14] = _unpack_bf16_70[14] * w_out_scale_sel_1;
                    w_q_vals_1_1[15] = _unpack_bf16_70[15] * w_out_scale_sel_1;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_1_1[0]), "f"(w_q_vals_1_1[1]),
                                               "f"(w_q_vals_1_1[2]), "f"(w_q_vals_1_1[3]));
                        w_q_words_1_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_1_1[4]), "f"(w_q_vals_1_1[5]),
                                               "f"(w_q_vals_1_1[6]), "f"(w_q_vals_1_1[7]));
                        w_q_words_1_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_1_1[8]), "f"(w_q_vals_1_1[9]),
                                               "f"(w_q_vals_1_1[10]), "f"(w_q_vals_1_1[11]));
                        w_q_words_1_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_1_1[12]), "f"(w_q_vals_1_1[13]),
                                               "f"(w_q_vals_1_1[14]), "f"(w_q_vals_1_1[15]));
                        w_q_words_1_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(w_srow_1_1 * 128) + (unsigned int)(w_chunk_lo_1_1 ^ (w_srow_1_1 & 7) << 4)), "r"(w_q_words_1_1[0]), "r"(w_q_words_1_1[1]), "r"(w_q_words_1_1[2]), "r"(w_q_words_1_1[3]) : "memory");
                    float _unpack_bf16_71[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_71[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_1_1[_pair]) << 16);
                        (&_unpack_bf16_71[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_1_1[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_1_1[0] = _unpack_bf16_71[0] * w_out_scale_sel_1;
                    w_q_vals_1_1[1] = _unpack_bf16_71[1] * w_out_scale_sel_1;
                    w_q_vals_1_1[2] = _unpack_bf16_71[2] * w_out_scale_sel_1;
                    w_q_vals_1_1[3] = _unpack_bf16_71[3] * w_out_scale_sel_1;
                    w_q_vals_1_1[4] = _unpack_bf16_71[4] * w_out_scale_sel_1;
                    w_q_vals_1_1[5] = _unpack_bf16_71[5] * w_out_scale_sel_1;
                    w_q_vals_1_1[6] = _unpack_bf16_71[6] * w_out_scale_sel_1;
                    w_q_vals_1_1[7] = _unpack_bf16_71[7] * w_out_scale_sel_1;
                    w_q_vals_1_1[8] = _unpack_bf16_71[8] * w_out_scale_sel_1;
                    w_q_vals_1_1[9] = _unpack_bf16_71[9] * w_out_scale_sel_1;
                    w_q_vals_1_1[10] = _unpack_bf16_71[10] * w_out_scale_sel_1;
                    w_q_vals_1_1[11] = _unpack_bf16_71[11] * w_out_scale_sel_1;
                    w_q_vals_1_1[12] = _unpack_bf16_71[12] * w_out_scale_sel_1;
                    w_q_vals_1_1[13] = _unpack_bf16_71[13] * w_out_scale_sel_1;
                    w_q_vals_1_1[14] = _unpack_bf16_71[14] * w_out_scale_sel_1;
                    w_q_vals_1_1[15] = _unpack_bf16_71[15] * w_out_scale_sel_1;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_1_1[0]), "f"(w_q_vals_1_1[1]),
                                               "f"(w_q_vals_1_1[2]), "f"(w_q_vals_1_1[3]));
                        w_q_words_1_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_1_1[4]), "f"(w_q_vals_1_1[5]),
                                               "f"(w_q_vals_1_1[6]), "f"(w_q_vals_1_1[7]));
                        w_q_words_1_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_1_1[8]), "f"(w_q_vals_1_1[9]),
                                               "f"(w_q_vals_1_1[10]), "f"(w_q_vals_1_1[11]));
                        w_q_words_1_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_1_1[12]), "f"(w_q_vals_1_1[13]),
                                               "f"(w_q_vals_1_1[14]), "f"(w_q_vals_1_1[15]));
                        w_q_words_1_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(w_srow_1_1 * 128) + (unsigned int)(w_chunk_hi_1_1 ^ (w_srow_1_1 & 7) << 4)), "r"(w_q_words_1_1[0]), "r"(w_q_words_1_1[1]), "r"(w_q_words_1_1[2]), "r"(w_q_words_1_1[3]) : "memory");
                }
                __syncwarp();
                int s_row_1_0_1 = (role_tid_1 >> 5) * 8 + lane / 8;
                int s_piece_1_0_1 = lane % 8;
                unsigned int s_words_1_0_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_0_1[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_row_1_0_1 * 128) + (unsigned int)(s_piece_1_0_1 * 16 ^ (s_row_1_0_1 & 7) << 4)));
                int s_dst_1_0_1 = ((tile_m_1 * 256 + 64 + s_row_1_0_1) * 56 + w_head_1) * 32 + s_piece_1_0_1 * 4;
                if (tile_m_1 * 256 + 64 + s_row_1_0_1 < M) {
                    if (w_kind_1 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_1_0_1)[0] = reinterpret_cast<int4*>(s_words_1_0_1)[0];
                    } else if (w_kind_1 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_1_0_1)[0] = reinterpret_cast<int4*>(s_words_1_0_1)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_1_0_1)[0] = reinterpret_cast<int4*>(s_words_1_0_1)[0];
                    }
                }
                int s_row_1_1_1 = (role_tid_1 >> 5) * 8 + 4 + lane / 8;
                int s_piece_1_1_1 = lane % 8;
                unsigned int s_words_1_1_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_1_1_1[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_row_1_1_1 * 128) + (unsigned int)(s_piece_1_1_1 * 16 ^ (s_row_1_1_1 & 7) << 4)));
                int s_dst_1_1_1 = ((tile_m_1 * 256 + 64 + s_row_1_1_1) * 56 + w_head_1) * 32 + s_piece_1_1_1 * 4;
                if (tile_m_1 * 256 + 64 + s_row_1_1_1 < M) {
                    if (w_kind_1 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_1_1_1)[0] = reinterpret_cast<int4*>(s_words_1_1_1)[0];
                    } else if (w_kind_1 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_1_1_1)[0] = reinterpret_cast<int4*>(s_words_1_1_1)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_1_1_1)[0] = reinterpret_cast<int4*>(s_words_1_1_1)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_row_lo_2_1 = tile_m_1 * 256 + 128 + warp_m_1 * 16 + (lane >> 2);
                int w_row_hi_2_1 = w_row_lo_2_1 + 8;
                float w_sa_lo_2_1 = ((w_row_lo_2_1 < M) ? act_scale[w_row_lo_2_1] : 0.0f);
                float w_sa_hi_2_1 = ((w_row_hi_2_1 < M) ? act_scale[w_row_hi_2_1] : 0.0f);
                float w_rounded_2_1[32];
                float w_ss_lo_2_1 = 0.0f;
                float w_ss_hi_2_1 = 0.0f;
                float w_scaled_2_0_1[4];
                w_scaled_2_0_1[0] = accum_1[64] * w_sa_lo_2_1 * sw_1[0];
                w_scaled_2_0_1[1] = accum_1[65] * w_sa_lo_2_1 * sw_1[1];
                w_scaled_2_0_1[2] = accum_1[66] * w_sa_hi_2_1 * sw_1[0];
                w_scaled_2_0_1[3] = accum_1[67] * w_sa_hi_2_1 * sw_1[1];
                unsigned int w_scaled_2_0_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_0_1[_lp*2 + 0], w_scaled_2_0_1[_lp*2+1 + 0]));
                    w_scaled_2_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_72[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_72[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_0_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_72[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_0_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_2_1[0] = _unpack_bf16_72[0];
                w_rounded_2_1[1] = _unpack_bf16_72[1];
                w_rounded_2_1[2] = _unpack_bf16_72[2];
                w_rounded_2_1[3] = _unpack_bf16_72[3];
                w_ss_lo_2_1 += __fmaf_rn(_unpack_bf16_72[0], _unpack_bf16_72[0], _unpack_bf16_72[1] * _unpack_bf16_72[1]);
                w_ss_hi_2_1 += __fmaf_rn(_unpack_bf16_72[2], _unpack_bf16_72[2], _unpack_bf16_72[3] * _unpack_bf16_72[3]);
                float w_scaled_2_1_1[4];
                w_scaled_2_1_1[0] = accum_1[68] * w_sa_lo_2_1 * sw_1[2];
                w_scaled_2_1_1[1] = accum_1[69] * w_sa_lo_2_1 * sw_1[3];
                w_scaled_2_1_1[2] = accum_1[70] * w_sa_hi_2_1 * sw_1[2];
                w_scaled_2_1_1[3] = accum_1[71] * w_sa_hi_2_1 * sw_1[3];
                unsigned int w_scaled_2_1_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_1_1[_lp*2 + 0], w_scaled_2_1_1[_lp*2+1 + 0]));
                    w_scaled_2_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_73[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_73[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_1_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_73[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_1_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_2_1[4] = _unpack_bf16_73[0];
                w_rounded_2_1[5] = _unpack_bf16_73[1];
                w_rounded_2_1[6] = _unpack_bf16_73[2];
                w_rounded_2_1[7] = _unpack_bf16_73[3];
                w_ss_lo_2_1 += __fmaf_rn(_unpack_bf16_73[0], _unpack_bf16_73[0], _unpack_bf16_73[1] * _unpack_bf16_73[1]);
                w_ss_hi_2_1 += __fmaf_rn(_unpack_bf16_73[2], _unpack_bf16_73[2], _unpack_bf16_73[3] * _unpack_bf16_73[3]);
                float w_scaled_2_2_1[4];
                w_scaled_2_2_1[0] = accum_1[72] * w_sa_lo_2_1 * sw_1[4];
                w_scaled_2_2_1[1] = accum_1[73] * w_sa_lo_2_1 * sw_1[5];
                w_scaled_2_2_1[2] = accum_1[74] * w_sa_hi_2_1 * sw_1[4];
                w_scaled_2_2_1[3] = accum_1[75] * w_sa_hi_2_1 * sw_1[5];
                unsigned int w_scaled_2_2_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_2_1[_lp*2 + 0], w_scaled_2_2_1[_lp*2+1 + 0]));
                    w_scaled_2_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_74[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_74[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_2_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_74[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_2_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_2_1[8] = _unpack_bf16_74[0];
                w_rounded_2_1[9] = _unpack_bf16_74[1];
                w_rounded_2_1[10] = _unpack_bf16_74[2];
                w_rounded_2_1[11] = _unpack_bf16_74[3];
                w_ss_lo_2_1 += __fmaf_rn(_unpack_bf16_74[0], _unpack_bf16_74[0], _unpack_bf16_74[1] * _unpack_bf16_74[1]);
                w_ss_hi_2_1 += __fmaf_rn(_unpack_bf16_74[2], _unpack_bf16_74[2], _unpack_bf16_74[3] * _unpack_bf16_74[3]);
                float w_scaled_2_3_1[4];
                w_scaled_2_3_1[0] = accum_1[76] * w_sa_lo_2_1 * sw_1[6];
                w_scaled_2_3_1[1] = accum_1[77] * w_sa_lo_2_1 * sw_1[7];
                w_scaled_2_3_1[2] = accum_1[78] * w_sa_hi_2_1 * sw_1[6];
                w_scaled_2_3_1[3] = accum_1[79] * w_sa_hi_2_1 * sw_1[7];
                unsigned int w_scaled_2_3_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_3_1[_lp*2 + 0], w_scaled_2_3_1[_lp*2+1 + 0]));
                    w_scaled_2_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_75[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_75[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_3_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_75[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_3_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_2_1[12] = _unpack_bf16_75[0];
                w_rounded_2_1[13] = _unpack_bf16_75[1];
                w_rounded_2_1[14] = _unpack_bf16_75[2];
                w_rounded_2_1[15] = _unpack_bf16_75[3];
                w_ss_lo_2_1 += __fmaf_rn(_unpack_bf16_75[0], _unpack_bf16_75[0], _unpack_bf16_75[1] * _unpack_bf16_75[1]);
                w_ss_hi_2_1 += __fmaf_rn(_unpack_bf16_75[2], _unpack_bf16_75[2], _unpack_bf16_75[3] * _unpack_bf16_75[3]);
                float w_scaled_2_4_1[4];
                w_scaled_2_4_1[0] = accum_1[80] * w_sa_lo_2_1 * sw_1[8];
                w_scaled_2_4_1[1] = accum_1[81] * w_sa_lo_2_1 * sw_1[9];
                w_scaled_2_4_1[2] = accum_1[82] * w_sa_hi_2_1 * sw_1[8];
                w_scaled_2_4_1[3] = accum_1[83] * w_sa_hi_2_1 * sw_1[9];
                unsigned int w_scaled_2_4_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_4_1[_lp*2 + 0], w_scaled_2_4_1[_lp*2+1 + 0]));
                    w_scaled_2_4_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_76[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_76[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_4_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_76[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_4_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_2_1[16] = _unpack_bf16_76[0];
                w_rounded_2_1[17] = _unpack_bf16_76[1];
                w_rounded_2_1[18] = _unpack_bf16_76[2];
                w_rounded_2_1[19] = _unpack_bf16_76[3];
                w_ss_lo_2_1 += __fmaf_rn(_unpack_bf16_76[0], _unpack_bf16_76[0], _unpack_bf16_76[1] * _unpack_bf16_76[1]);
                w_ss_hi_2_1 += __fmaf_rn(_unpack_bf16_76[2], _unpack_bf16_76[2], _unpack_bf16_76[3] * _unpack_bf16_76[3]);
                float w_scaled_2_5_1[4];
                w_scaled_2_5_1[0] = accum_1[84] * w_sa_lo_2_1 * sw_1[10];
                w_scaled_2_5_1[1] = accum_1[85] * w_sa_lo_2_1 * sw_1[11];
                w_scaled_2_5_1[2] = accum_1[86] * w_sa_hi_2_1 * sw_1[10];
                w_scaled_2_5_1[3] = accum_1[87] * w_sa_hi_2_1 * sw_1[11];
                unsigned int w_scaled_2_5_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_5_1[_lp*2 + 0], w_scaled_2_5_1[_lp*2+1 + 0]));
                    w_scaled_2_5_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_77[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_77[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_5_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_77[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_5_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_2_1[20] = _unpack_bf16_77[0];
                w_rounded_2_1[21] = _unpack_bf16_77[1];
                w_rounded_2_1[22] = _unpack_bf16_77[2];
                w_rounded_2_1[23] = _unpack_bf16_77[3];
                w_ss_lo_2_1 += __fmaf_rn(_unpack_bf16_77[0], _unpack_bf16_77[0], _unpack_bf16_77[1] * _unpack_bf16_77[1]);
                w_ss_hi_2_1 += __fmaf_rn(_unpack_bf16_77[2], _unpack_bf16_77[2], _unpack_bf16_77[3] * _unpack_bf16_77[3]);
                float w_scaled_2_6_1[4];
                w_scaled_2_6_1[0] = accum_1[88] * w_sa_lo_2_1 * sw_1[12];
                w_scaled_2_6_1[1] = accum_1[89] * w_sa_lo_2_1 * sw_1[13];
                w_scaled_2_6_1[2] = accum_1[90] * w_sa_hi_2_1 * sw_1[12];
                w_scaled_2_6_1[3] = accum_1[91] * w_sa_hi_2_1 * sw_1[13];
                unsigned int w_scaled_2_6_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_6_1[_lp*2 + 0], w_scaled_2_6_1[_lp*2+1 + 0]));
                    w_scaled_2_6_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_78[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_78[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_6_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_78[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_6_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_2_1[24] = _unpack_bf16_78[0];
                w_rounded_2_1[25] = _unpack_bf16_78[1];
                w_rounded_2_1[26] = _unpack_bf16_78[2];
                w_rounded_2_1[27] = _unpack_bf16_78[3];
                w_ss_lo_2_1 += __fmaf_rn(_unpack_bf16_78[0], _unpack_bf16_78[0], _unpack_bf16_78[1] * _unpack_bf16_78[1]);
                w_ss_hi_2_1 += __fmaf_rn(_unpack_bf16_78[2], _unpack_bf16_78[2], _unpack_bf16_78[3] * _unpack_bf16_78[3]);
                float w_scaled_2_7_1[4];
                w_scaled_2_7_1[0] = accum_1[92] * w_sa_lo_2_1 * sw_1[14];
                w_scaled_2_7_1[1] = accum_1[93] * w_sa_lo_2_1 * sw_1[15];
                w_scaled_2_7_1[2] = accum_1[94] * w_sa_hi_2_1 * sw_1[14];
                w_scaled_2_7_1[3] = accum_1[95] * w_sa_hi_2_1 * sw_1[15];
                unsigned int w_scaled_2_7_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_2_7_1[_lp*2 + 0], w_scaled_2_7_1[_lp*2+1 + 0]));
                    w_scaled_2_7_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_79[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_79[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_7_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_79[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_2_7_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_2_1[28] = _unpack_bf16_79[0];
                w_rounded_2_1[29] = _unpack_bf16_79[1];
                w_rounded_2_1[30] = _unpack_bf16_79[2];
                w_rounded_2_1[31] = _unpack_bf16_79[3];
                w_ss_lo_2_1 += __fmaf_rn(_unpack_bf16_79[0], _unpack_bf16_79[0], _unpack_bf16_79[1] * _unpack_bf16_79[1]);
                w_ss_hi_2_1 += __fmaf_rn(_unpack_bf16_79[2], _unpack_bf16_79[2], _unpack_bf16_79[3] * _unpack_bf16_79[3]);
                float w_rstd_lo_2_1 = 1.0f;
                float w_rstd_hi_2_1 = 1.0f;
                if (w_kind_1 < 2) {
                    float _shfl_xor_sync_40_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_2_1, 1);
                    w_ss_lo_2_1 += _shfl_xor_sync_40_1;
                    float _shfl_xor_sync_41_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_2_1, 2);
                    w_ss_lo_2_1 += _shfl_xor_sync_41_1;
                    float _shfl_xor_sync_42_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_2_1, 1);
                    w_ss_hi_2_1 += _shfl_xor_sync_42_1;
                    float _shfl_xor_sync_43_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_2_1, 2);
                    w_ss_hi_2_1 += _shfl_xor_sync_43_1;
                    int w_srow_lo_2_1 = warp_m_1 * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n_1 * 64 + w_srow_lo_2_1] = w_ss_lo_2_1;
                        rowstat[warp_n_1 * 64 + w_srow_lo_2_1 + 8] = w_ss_hi_2_1;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float w_other_lo_2_1 = rowstat[(1 - warp_n_1) * 64 + w_srow_lo_2_1];
                    float w_other_hi_2_1 = rowstat[(1 - warp_n_1) * 64 + w_srow_lo_2_1 + 8];
                    float _fdiv_45_1 = __fdiv_rn(w_ss_lo_2_1 + w_other_lo_2_1, 128.0f);
                    float _rsqrt_44_1 = rsqrtf(_fdiv_45_1 + eps);
                    w_rstd_lo_2_1 = _rsqrt_44_1;
                    float _fdiv_47_1 = __fdiv_rn(w_ss_hi_2_1 + w_other_hi_2_1, 128.0f);
                    float _rsqrt_46_1 = rsqrtf(_fdiv_47_1 + eps);
                    w_rstd_hi_2_1 = _rsqrt_46_1;
                }
                int w_srow_st_2_1 = warp_m_1 * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float w_normed_2_0_1[4];
                if (w_kind_1 < 2) {
                    w_normed_2_0_1[0] = w_rounded_2_1[0] * w_rstd_lo_2_1 * nw_1[0];
                    w_normed_2_0_1[1] = w_rounded_2_1[1] * w_rstd_lo_2_1 * nw_1[1];
                    w_normed_2_0_1[2] = w_rounded_2_1[2] * w_rstd_hi_2_1 * nw_1[0];
                    w_normed_2_0_1[3] = w_rounded_2_1[3] * w_rstd_hi_2_1 * nw_1[1];
                } else {
                    w_normed_2_0_1[0] = w_rounded_2_1[0];
                    w_normed_2_0_1[1] = w_rounded_2_1[1];
                    w_normed_2_0_1[2] = w_rounded_2_1[2];
                    w_normed_2_0_1[3] = w_rounded_2_1[3];
                }
                unsigned int w_normed_2_0_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_0_1[_lp*2 + 0], w_normed_2_0_1[_lp*2+1 + 0]));
                    w_normed_2_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_48 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_2_1 * 256) + (unsigned int)(warp_n_1 * 64 * 2 ^ (w_srow_st_2_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_48), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_0_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_0_bf16_1[1]))
                    : "memory");
                float w_normed_2_1_1[4];
                if (w_kind_1 < 2) {
                    w_normed_2_1_1[0] = w_rounded_2_1[4] * w_rstd_lo_2_1 * nw_1[2];
                    w_normed_2_1_1[1] = w_rounded_2_1[5] * w_rstd_lo_2_1 * nw_1[3];
                    w_normed_2_1_1[2] = w_rounded_2_1[6] * w_rstd_hi_2_1 * nw_1[2];
                    w_normed_2_1_1[3] = w_rounded_2_1[7] * w_rstd_hi_2_1 * nw_1[3];
                } else {
                    w_normed_2_1_1[0] = w_rounded_2_1[4];
                    w_normed_2_1_1[1] = w_rounded_2_1[5];
                    w_normed_2_1_1[2] = w_rounded_2_1[6];
                    w_normed_2_1_1[3] = w_rounded_2_1[7];
                }
                unsigned int w_normed_2_1_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_1_1[_lp*2 + 0], w_normed_2_1_1[_lp*2+1 + 0]));
                    w_normed_2_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_49 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_2_1 * 256) + (unsigned int)((warp_n_1 * 64 + 8) * 2 ^ (w_srow_st_2_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_49), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_1_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_1_bf16_1[1]))
                    : "memory");
                float w_normed_2_2_1[4];
                if (w_kind_1 < 2) {
                    w_normed_2_2_1[0] = w_rounded_2_1[8] * w_rstd_lo_2_1 * nw_1[4];
                    w_normed_2_2_1[1] = w_rounded_2_1[9] * w_rstd_lo_2_1 * nw_1[5];
                    w_normed_2_2_1[2] = w_rounded_2_1[10] * w_rstd_hi_2_1 * nw_1[4];
                    w_normed_2_2_1[3] = w_rounded_2_1[11] * w_rstd_hi_2_1 * nw_1[5];
                } else {
                    w_normed_2_2_1[0] = w_rounded_2_1[8];
                    w_normed_2_2_1[1] = w_rounded_2_1[9];
                    w_normed_2_2_1[2] = w_rounded_2_1[10];
                    w_normed_2_2_1[3] = w_rounded_2_1[11];
                }
                unsigned int w_normed_2_2_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_2_1[_lp*2 + 0], w_normed_2_2_1[_lp*2+1 + 0]));
                    w_normed_2_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_50 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_2_1 * 256) + (unsigned int)((warp_n_1 * 64 + 16) * 2 ^ (w_srow_st_2_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_50), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_2_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_2_bf16_1[1]))
                    : "memory");
                float w_normed_2_3_1[4];
                if (w_kind_1 < 2) {
                    w_normed_2_3_1[0] = w_rounded_2_1[12] * w_rstd_lo_2_1 * nw_1[6];
                    w_normed_2_3_1[1] = w_rounded_2_1[13] * w_rstd_lo_2_1 * nw_1[7];
                    w_normed_2_3_1[2] = w_rounded_2_1[14] * w_rstd_hi_2_1 * nw_1[6];
                    w_normed_2_3_1[3] = w_rounded_2_1[15] * w_rstd_hi_2_1 * nw_1[7];
                } else {
                    w_normed_2_3_1[0] = w_rounded_2_1[12];
                    w_normed_2_3_1[1] = w_rounded_2_1[13];
                    w_normed_2_3_1[2] = w_rounded_2_1[14];
                    w_normed_2_3_1[3] = w_rounded_2_1[15];
                }
                unsigned int w_normed_2_3_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_3_1[_lp*2 + 0], w_normed_2_3_1[_lp*2+1 + 0]));
                    w_normed_2_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_51 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_2_1 * 256) + (unsigned int)((warp_n_1 * 64 + 24) * 2 ^ (w_srow_st_2_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_51), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_3_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_3_bf16_1[1]))
                    : "memory");
                float w_normed_2_4_1[4];
                if (w_kind_1 < 2) {
                    w_normed_2_4_1[0] = w_rounded_2_1[16] * w_rstd_lo_2_1 * nw_1[8];
                    w_normed_2_4_1[1] = w_rounded_2_1[17] * w_rstd_lo_2_1 * nw_1[9];
                    w_normed_2_4_1[2] = w_rounded_2_1[18] * w_rstd_hi_2_1 * nw_1[8];
                    w_normed_2_4_1[3] = w_rounded_2_1[19] * w_rstd_hi_2_1 * nw_1[9];
                } else {
                    w_normed_2_4_1[0] = w_rounded_2_1[16];
                    w_normed_2_4_1[1] = w_rounded_2_1[17];
                    w_normed_2_4_1[2] = w_rounded_2_1[18];
                    w_normed_2_4_1[3] = w_rounded_2_1[19];
                }
                unsigned int w_normed_2_4_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_4_1[_lp*2 + 0], w_normed_2_4_1[_lp*2+1 + 0]));
                    w_normed_2_4_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_52 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_2_1 * 256) + (unsigned int)((warp_n_1 * 64 + 32) * 2 ^ (w_srow_st_2_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_52), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_4_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_4_bf16_1[1]))
                    : "memory");
                float w_normed_2_5_1[4];
                if (w_kind_1 < 2) {
                    w_normed_2_5_1[0] = w_rounded_2_1[20] * w_rstd_lo_2_1 * nw_1[10];
                    w_normed_2_5_1[1] = w_rounded_2_1[21] * w_rstd_lo_2_1 * nw_1[11];
                    w_normed_2_5_1[2] = w_rounded_2_1[22] * w_rstd_hi_2_1 * nw_1[10];
                    w_normed_2_5_1[3] = w_rounded_2_1[23] * w_rstd_hi_2_1 * nw_1[11];
                } else {
                    w_normed_2_5_1[0] = w_rounded_2_1[20];
                    w_normed_2_5_1[1] = w_rounded_2_1[21];
                    w_normed_2_5_1[2] = w_rounded_2_1[22];
                    w_normed_2_5_1[3] = w_rounded_2_1[23];
                }
                unsigned int w_normed_2_5_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_5_1[_lp*2 + 0], w_normed_2_5_1[_lp*2+1 + 0]));
                    w_normed_2_5_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_53 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_2_1 * 256) + (unsigned int)((warp_n_1 * 64 + 40) * 2 ^ (w_srow_st_2_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_53), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_5_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_5_bf16_1[1]))
                    : "memory");
                float w_normed_2_6_1[4];
                if (w_kind_1 < 2) {
                    w_normed_2_6_1[0] = w_rounded_2_1[24] * w_rstd_lo_2_1 * nw_1[12];
                    w_normed_2_6_1[1] = w_rounded_2_1[25] * w_rstd_lo_2_1 * nw_1[13];
                    w_normed_2_6_1[2] = w_rounded_2_1[26] * w_rstd_hi_2_1 * nw_1[12];
                    w_normed_2_6_1[3] = w_rounded_2_1[27] * w_rstd_hi_2_1 * nw_1[13];
                } else {
                    w_normed_2_6_1[0] = w_rounded_2_1[24];
                    w_normed_2_6_1[1] = w_rounded_2_1[25];
                    w_normed_2_6_1[2] = w_rounded_2_1[26];
                    w_normed_2_6_1[3] = w_rounded_2_1[27];
                }
                unsigned int w_normed_2_6_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_6_1[_lp*2 + 0], w_normed_2_6_1[_lp*2+1 + 0]));
                    w_normed_2_6_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_54 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_2_1 * 256) + (unsigned int)((warp_n_1 * 64 + 48) * 2 ^ (w_srow_st_2_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_54), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_6_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_6_bf16_1[1]))
                    : "memory");
                float w_normed_2_7_1[4];
                if (w_kind_1 < 2) {
                    w_normed_2_7_1[0] = w_rounded_2_1[28] * w_rstd_lo_2_1 * nw_1[14];
                    w_normed_2_7_1[1] = w_rounded_2_1[29] * w_rstd_lo_2_1 * nw_1[15];
                    w_normed_2_7_1[2] = w_rounded_2_1[30] * w_rstd_hi_2_1 * nw_1[14];
                    w_normed_2_7_1[3] = w_rounded_2_1[31] * w_rstd_hi_2_1 * nw_1[15];
                } else {
                    w_normed_2_7_1[0] = w_rounded_2_1[28];
                    w_normed_2_7_1[1] = w_rounded_2_1[29];
                    w_normed_2_7_1[2] = w_rounded_2_1[30];
                    w_normed_2_7_1[3] = w_rounded_2_1[31];
                }
                unsigned int w_normed_2_7_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_2_7_1[_lp*2 + 0], w_normed_2_7_1[_lp*2+1 + 0]));
                    w_normed_2_7_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_55 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_2_1 * 256) + (unsigned int)((warp_n_1 * 64 + 56) * 2 ^ (w_srow_st_2_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_55), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_7_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_2_7_bf16_1[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_srow_2_1 = role_tid_1 >> 2;
                int w_c_2_1 = role_tid_1 & 3;
                int w_grow_2_1 = tile_m_1 * 256 + 128 + w_srow_2_1;
                int w_chunk_lo_2_1 = ((w_c_2_1 < 3) ? w_c_2_1 * 16 : 96);
                int w_chunk_hi_2_1 = ((w_c_2_1 < 3) ? w_chunk_lo_2_1 + 48 : 112);
                if (w_grow_2_1 < M) {
                    unsigned int w_x_lo_2_1[8];
                    unsigned int w_x_hi_2_1[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2_1[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_2_1 * 256) + (unsigned int)(w_chunk_lo_2_1 * 2 ^ (w_srow_2_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_2_1[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_2_1 * 256) + (unsigned int)(w_chunk_lo_2_1 * 2 + 16 ^ (w_srow_2_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2_1[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_2_1 * 256) + (unsigned int)(w_chunk_hi_2_1 * 2 ^ (w_srow_2_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_2_1[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_2_1 * 256) + (unsigned int)(w_chunk_hi_2_1 * 2 + 16 ^ (w_srow_2_1 & 7) << 4)));
                    if (w_c_2_1 < 3) {
                        if (w_kind_1 < 2) {
                            float _unpack_bf16_80[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_80[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_2_1[_pair]) << 16);
                                (&_unpack_bf16_80[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_2_1[_pair]) & 0xffff0000u);
                            }
                            float _unpack_bf16_81[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_81[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_2_1[_pair]) << 16);
                                (&_unpack_bf16_81[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_2_1[_pair]) & 0xffff0000u);
                            }
                            float _vec_load_24[8];
                            {
                                const uint4* _vptr_56 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_2_1 * 96 + w_chunk_lo_2_1) + 0);
                                uint4 _vld_56[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_56[_blk] = _vptr_56[_blk];
                                    uint32_t* _vpairs_56 = reinterpret_cast<uint32_t*>(&_vld_56[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_24[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) << 16);
                                        (&_vec_load_24[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_25[8];
                            {
                                const uint4* _vptr_57 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_2_1 * 96 + w_chunk_lo_2_1 + 8) + 0);
                                uint4 _vld_57[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_57[_blk] = _vptr_57[_blk];
                                    uint32_t* _vpairs_57 = reinterpret_cast<uint32_t*>(&_vld_57[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_25[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) << 16);
                                        (&_vec_load_25[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_26[8];
                            {
                                const uint4* _vptr_58 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_2_1 * 96 + 48 + w_chunk_lo_2_1) + 0);
                                uint4 _vld_58[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_58[_blk] = _vptr_58[_blk];
                                    uint32_t* _vpairs_58 = reinterpret_cast<uint32_t*>(&_vld_58[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_26[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) << 16);
                                        (&_vec_load_26[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_27[8];
                            {
                                const uint4* _vptr_59 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_2_1 * 96 + 48 + w_chunk_lo_2_1 + 8) + 0);
                                uint4 _vld_59[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_59[_blk] = _vptr_59[_blk];
                                    uint32_t* _vpairs_59 = reinterpret_cast<uint32_t*>(&_vld_59[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_27[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) << 16);
                                        (&_vec_load_27[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float w_o_lo_2_1[16];
                            float w_o_hi_2_1[16];
                            float w_cv0_2_0_1 = _vec_load_24[0];
                            float w_sv0_2_0_1 = _vec_load_26[0];
                            float w_cv1_2_0_1 = _vec_load_25[0];
                            float w_sv1_2_0_1 = _vec_load_27[0];
                            w_o_lo_2_1[0] = __fmaf_rn(_unpack_bf16_80[0], w_cv0_2_0_1, -(_unpack_bf16_81[0] * w_sv0_2_0_1));
                            w_o_hi_2_1[0] = __fmaf_rn(_unpack_bf16_81[0], w_cv0_2_0_1, _unpack_bf16_80[0] * w_sv0_2_0_1);
                            w_o_lo_2_1[8] = __fmaf_rn(_unpack_bf16_80[8], w_cv1_2_0_1, -(_unpack_bf16_81[8] * w_sv1_2_0_1));
                            w_o_hi_2_1[8] = __fmaf_rn(_unpack_bf16_81[8], w_cv1_2_0_1, _unpack_bf16_80[8] * w_sv1_2_0_1);
                            float w_cv0_2_1_1 = _vec_load_24[1];
                            float w_sv0_2_1_1 = _vec_load_26[1];
                            float w_cv1_2_1_1 = _vec_load_25[1];
                            float w_sv1_2_1_1 = _vec_load_27[1];
                            w_o_lo_2_1[1] = __fmaf_rn(_unpack_bf16_80[1], w_cv0_2_1_1, -(_unpack_bf16_81[1] * w_sv0_2_1_1));
                            w_o_hi_2_1[1] = __fmaf_rn(_unpack_bf16_81[1], w_cv0_2_1_1, _unpack_bf16_80[1] * w_sv0_2_1_1);
                            w_o_lo_2_1[9] = __fmaf_rn(_unpack_bf16_80[9], w_cv1_2_1_1, -(_unpack_bf16_81[9] * w_sv1_2_1_1));
                            w_o_hi_2_1[9] = __fmaf_rn(_unpack_bf16_81[9], w_cv1_2_1_1, _unpack_bf16_80[9] * w_sv1_2_1_1);
                            float w_cv0_2_2_1 = _vec_load_24[2];
                            float w_sv0_2_2_1 = _vec_load_26[2];
                            float w_cv1_2_2_1 = _vec_load_25[2];
                            float w_sv1_2_2_1 = _vec_load_27[2];
                            w_o_lo_2_1[2] = __fmaf_rn(_unpack_bf16_80[2], w_cv0_2_2_1, -(_unpack_bf16_81[2] * w_sv0_2_2_1));
                            w_o_hi_2_1[2] = __fmaf_rn(_unpack_bf16_81[2], w_cv0_2_2_1, _unpack_bf16_80[2] * w_sv0_2_2_1);
                            w_o_lo_2_1[10] = __fmaf_rn(_unpack_bf16_80[10], w_cv1_2_2_1, -(_unpack_bf16_81[10] * w_sv1_2_2_1));
                            w_o_hi_2_1[10] = __fmaf_rn(_unpack_bf16_81[10], w_cv1_2_2_1, _unpack_bf16_80[10] * w_sv1_2_2_1);
                            float w_cv0_2_3_1 = _vec_load_24[3];
                            float w_sv0_2_3_1 = _vec_load_26[3];
                            float w_cv1_2_3_1 = _vec_load_25[3];
                            float w_sv1_2_3_1 = _vec_load_27[3];
                            w_o_lo_2_1[3] = __fmaf_rn(_unpack_bf16_80[3], w_cv0_2_3_1, -(_unpack_bf16_81[3] * w_sv0_2_3_1));
                            w_o_hi_2_1[3] = __fmaf_rn(_unpack_bf16_81[3], w_cv0_2_3_1, _unpack_bf16_80[3] * w_sv0_2_3_1);
                            w_o_lo_2_1[11] = __fmaf_rn(_unpack_bf16_80[11], w_cv1_2_3_1, -(_unpack_bf16_81[11] * w_sv1_2_3_1));
                            w_o_hi_2_1[11] = __fmaf_rn(_unpack_bf16_81[11], w_cv1_2_3_1, _unpack_bf16_80[11] * w_sv1_2_3_1);
                            float w_cv0_2_4_1 = _vec_load_24[4];
                            float w_sv0_2_4_1 = _vec_load_26[4];
                            float w_cv1_2_4_1 = _vec_load_25[4];
                            float w_sv1_2_4_1 = _vec_load_27[4];
                            w_o_lo_2_1[4] = __fmaf_rn(_unpack_bf16_80[4], w_cv0_2_4_1, -(_unpack_bf16_81[4] * w_sv0_2_4_1));
                            w_o_hi_2_1[4] = __fmaf_rn(_unpack_bf16_81[4], w_cv0_2_4_1, _unpack_bf16_80[4] * w_sv0_2_4_1);
                            w_o_lo_2_1[12] = __fmaf_rn(_unpack_bf16_80[12], w_cv1_2_4_1, -(_unpack_bf16_81[12] * w_sv1_2_4_1));
                            w_o_hi_2_1[12] = __fmaf_rn(_unpack_bf16_81[12], w_cv1_2_4_1, _unpack_bf16_80[12] * w_sv1_2_4_1);
                            float w_cv0_2_5_1 = _vec_load_24[5];
                            float w_sv0_2_5_1 = _vec_load_26[5];
                            float w_cv1_2_5_1 = _vec_load_25[5];
                            float w_sv1_2_5_1 = _vec_load_27[5];
                            w_o_lo_2_1[5] = __fmaf_rn(_unpack_bf16_80[5], w_cv0_2_5_1, -(_unpack_bf16_81[5] * w_sv0_2_5_1));
                            w_o_hi_2_1[5] = __fmaf_rn(_unpack_bf16_81[5], w_cv0_2_5_1, _unpack_bf16_80[5] * w_sv0_2_5_1);
                            w_o_lo_2_1[13] = __fmaf_rn(_unpack_bf16_80[13], w_cv1_2_5_1, -(_unpack_bf16_81[13] * w_sv1_2_5_1));
                            w_o_hi_2_1[13] = __fmaf_rn(_unpack_bf16_81[13], w_cv1_2_5_1, _unpack_bf16_80[13] * w_sv1_2_5_1);
                            float w_cv0_2_6_1 = _vec_load_24[6];
                            float w_sv0_2_6_1 = _vec_load_26[6];
                            float w_cv1_2_6_1 = _vec_load_25[6];
                            float w_sv1_2_6_1 = _vec_load_27[6];
                            w_o_lo_2_1[6] = __fmaf_rn(_unpack_bf16_80[6], w_cv0_2_6_1, -(_unpack_bf16_81[6] * w_sv0_2_6_1));
                            w_o_hi_2_1[6] = __fmaf_rn(_unpack_bf16_81[6], w_cv0_2_6_1, _unpack_bf16_80[6] * w_sv0_2_6_1);
                            w_o_lo_2_1[14] = __fmaf_rn(_unpack_bf16_80[14], w_cv1_2_6_1, -(_unpack_bf16_81[14] * w_sv1_2_6_1));
                            w_o_hi_2_1[14] = __fmaf_rn(_unpack_bf16_81[14], w_cv1_2_6_1, _unpack_bf16_80[14] * w_sv1_2_6_1);
                            float w_cv0_2_7_1 = _vec_load_24[7];
                            float w_sv0_2_7_1 = _vec_load_26[7];
                            float w_cv1_2_7_1 = _vec_load_25[7];
                            float w_sv1_2_7_1 = _vec_load_27[7];
                            w_o_lo_2_1[7] = __fmaf_rn(_unpack_bf16_80[7], w_cv0_2_7_1, -(_unpack_bf16_81[7] * w_sv0_2_7_1));
                            w_o_hi_2_1[7] = __fmaf_rn(_unpack_bf16_81[7], w_cv0_2_7_1, _unpack_bf16_80[7] * w_sv0_2_7_1);
                            w_o_lo_2_1[15] = __fmaf_rn(_unpack_bf16_80[15], w_cv1_2_7_1, -(_unpack_bf16_81[15] * w_sv1_2_7_1));
                            w_o_hi_2_1[15] = __fmaf_rn(_unpack_bf16_81[15], w_cv1_2_7_1, _unpack_bf16_80[15] * w_sv1_2_7_1);
                            unsigned int w_o_lo_2_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_lo_2_1[_lp*2 + 0], w_o_lo_2_1[_lp*2+1 + 0]));
                                w_o_lo_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int w_o_hi_2_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_hi_2_1[_lp*2 + 0], w_o_hi_2_1[_lp*2+1 + 0]));
                                w_o_hi_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            w_x_lo_2_1[0] = w_o_lo_2_bf16_1[0];
                            w_x_hi_2_1[0] = w_o_hi_2_bf16_1[0];
                            w_x_lo_2_1[1] = w_o_lo_2_bf16_1[1];
                            w_x_hi_2_1[1] = w_o_hi_2_bf16_1[1];
                            w_x_lo_2_1[2] = w_o_lo_2_bf16_1[2];
                            w_x_hi_2_1[2] = w_o_hi_2_bf16_1[2];
                            w_x_lo_2_1[3] = w_o_lo_2_bf16_1[3];
                            w_x_hi_2_1[3] = w_o_hi_2_bf16_1[3];
                            w_x_lo_2_1[4] = w_o_lo_2_bf16_1[4];
                            w_x_hi_2_1[4] = w_o_hi_2_bf16_1[4];
                            w_x_lo_2_1[5] = w_o_lo_2_bf16_1[5];
                            w_x_hi_2_1[5] = w_o_hi_2_bf16_1[5];
                            w_x_lo_2_1[6] = w_o_lo_2_bf16_1[6];
                            w_x_hi_2_1[6] = w_o_hi_2_bf16_1[6];
                            w_x_lo_2_1[7] = w_o_lo_2_bf16_1[7];
                            w_x_hi_2_1[7] = w_o_hi_2_bf16_1[7];
                        }
                    }
                    unsigned int w_q_words_2_1[4];
                    float w_q_vals_2_1[16];
                    float _unpack_bf16_82[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_82[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_2_1[_pair]) << 16);
                        (&_unpack_bf16_82[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_2_1[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_2_1[0] = _unpack_bf16_82[0] * w_out_scale_sel_1;
                    w_q_vals_2_1[1] = _unpack_bf16_82[1] * w_out_scale_sel_1;
                    w_q_vals_2_1[2] = _unpack_bf16_82[2] * w_out_scale_sel_1;
                    w_q_vals_2_1[3] = _unpack_bf16_82[3] * w_out_scale_sel_1;
                    w_q_vals_2_1[4] = _unpack_bf16_82[4] * w_out_scale_sel_1;
                    w_q_vals_2_1[5] = _unpack_bf16_82[5] * w_out_scale_sel_1;
                    w_q_vals_2_1[6] = _unpack_bf16_82[6] * w_out_scale_sel_1;
                    w_q_vals_2_1[7] = _unpack_bf16_82[7] * w_out_scale_sel_1;
                    w_q_vals_2_1[8] = _unpack_bf16_82[8] * w_out_scale_sel_1;
                    w_q_vals_2_1[9] = _unpack_bf16_82[9] * w_out_scale_sel_1;
                    w_q_vals_2_1[10] = _unpack_bf16_82[10] * w_out_scale_sel_1;
                    w_q_vals_2_1[11] = _unpack_bf16_82[11] * w_out_scale_sel_1;
                    w_q_vals_2_1[12] = _unpack_bf16_82[12] * w_out_scale_sel_1;
                    w_q_vals_2_1[13] = _unpack_bf16_82[13] * w_out_scale_sel_1;
                    w_q_vals_2_1[14] = _unpack_bf16_82[14] * w_out_scale_sel_1;
                    w_q_vals_2_1[15] = _unpack_bf16_82[15] * w_out_scale_sel_1;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_2_1[0]), "f"(w_q_vals_2_1[1]),
                                               "f"(w_q_vals_2_1[2]), "f"(w_q_vals_2_1[3]));
                        w_q_words_2_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_2_1[4]), "f"(w_q_vals_2_1[5]),
                                               "f"(w_q_vals_2_1[6]), "f"(w_q_vals_2_1[7]));
                        w_q_words_2_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_2_1[8]), "f"(w_q_vals_2_1[9]),
                                               "f"(w_q_vals_2_1[10]), "f"(w_q_vals_2_1[11]));
                        w_q_words_2_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_2_1[12]), "f"(w_q_vals_2_1[13]),
                                               "f"(w_q_vals_2_1[14]), "f"(w_q_vals_2_1[15]));
                        w_q_words_2_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(w_srow_2_1 * 128) + (unsigned int)(w_chunk_lo_2_1 ^ (w_srow_2_1 & 7) << 4)), "r"(w_q_words_2_1[0]), "r"(w_q_words_2_1[1]), "r"(w_q_words_2_1[2]), "r"(w_q_words_2_1[3]) : "memory");
                    float _unpack_bf16_83[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_83[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_2_1[_pair]) << 16);
                        (&_unpack_bf16_83[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_2_1[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_2_1[0] = _unpack_bf16_83[0] * w_out_scale_sel_1;
                    w_q_vals_2_1[1] = _unpack_bf16_83[1] * w_out_scale_sel_1;
                    w_q_vals_2_1[2] = _unpack_bf16_83[2] * w_out_scale_sel_1;
                    w_q_vals_2_1[3] = _unpack_bf16_83[3] * w_out_scale_sel_1;
                    w_q_vals_2_1[4] = _unpack_bf16_83[4] * w_out_scale_sel_1;
                    w_q_vals_2_1[5] = _unpack_bf16_83[5] * w_out_scale_sel_1;
                    w_q_vals_2_1[6] = _unpack_bf16_83[6] * w_out_scale_sel_1;
                    w_q_vals_2_1[7] = _unpack_bf16_83[7] * w_out_scale_sel_1;
                    w_q_vals_2_1[8] = _unpack_bf16_83[8] * w_out_scale_sel_1;
                    w_q_vals_2_1[9] = _unpack_bf16_83[9] * w_out_scale_sel_1;
                    w_q_vals_2_1[10] = _unpack_bf16_83[10] * w_out_scale_sel_1;
                    w_q_vals_2_1[11] = _unpack_bf16_83[11] * w_out_scale_sel_1;
                    w_q_vals_2_1[12] = _unpack_bf16_83[12] * w_out_scale_sel_1;
                    w_q_vals_2_1[13] = _unpack_bf16_83[13] * w_out_scale_sel_1;
                    w_q_vals_2_1[14] = _unpack_bf16_83[14] * w_out_scale_sel_1;
                    w_q_vals_2_1[15] = _unpack_bf16_83[15] * w_out_scale_sel_1;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_2_1[0]), "f"(w_q_vals_2_1[1]),
                                               "f"(w_q_vals_2_1[2]), "f"(w_q_vals_2_1[3]));
                        w_q_words_2_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_2_1[4]), "f"(w_q_vals_2_1[5]),
                                               "f"(w_q_vals_2_1[6]), "f"(w_q_vals_2_1[7]));
                        w_q_words_2_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_2_1[8]), "f"(w_q_vals_2_1[9]),
                                               "f"(w_q_vals_2_1[10]), "f"(w_q_vals_2_1[11]));
                        w_q_words_2_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_2_1[12]), "f"(w_q_vals_2_1[13]),
                                               "f"(w_q_vals_2_1[14]), "f"(w_q_vals_2_1[15]));
                        w_q_words_2_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(w_srow_2_1 * 128) + (unsigned int)(w_chunk_hi_2_1 ^ (w_srow_2_1 & 7) << 4)), "r"(w_q_words_2_1[0]), "r"(w_q_words_2_1[1]), "r"(w_q_words_2_1[2]), "r"(w_q_words_2_1[3]) : "memory");
                }
                __syncwarp();
                int s_row_2_0_1 = (role_tid_1 >> 5) * 8 + lane / 8;
                int s_piece_2_0_1 = lane % 8;
                unsigned int s_words_2_0_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_0_1[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_row_2_0_1 * 128) + (unsigned int)(s_piece_2_0_1 * 16 ^ (s_row_2_0_1 & 7) << 4)));
                int s_dst_2_0_1 = ((tile_m_1 * 256 + 128 + s_row_2_0_1) * 56 + w_head_1) * 32 + s_piece_2_0_1 * 4;
                if (tile_m_1 * 256 + 128 + s_row_2_0_1 < M) {
                    if (w_kind_1 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_2_0_1)[0] = reinterpret_cast<int4*>(s_words_2_0_1)[0];
                    } else if (w_kind_1 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_2_0_1)[0] = reinterpret_cast<int4*>(s_words_2_0_1)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_2_0_1)[0] = reinterpret_cast<int4*>(s_words_2_0_1)[0];
                    }
                }
                int s_row_2_1_1 = (role_tid_1 >> 5) * 8 + 4 + lane / 8;
                int s_piece_2_1_1 = lane % 8;
                unsigned int s_words_2_1_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_2_1_1[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_row_2_1_1 * 128) + (unsigned int)(s_piece_2_1_1 * 16 ^ (s_row_2_1_1 & 7) << 4)));
                int s_dst_2_1_1 = ((tile_m_1 * 256 + 128 + s_row_2_1_1) * 56 + w_head_1) * 32 + s_piece_2_1_1 * 4;
                if (tile_m_1 * 256 + 128 + s_row_2_1_1 < M) {
                    if (w_kind_1 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_2_1_1)[0] = reinterpret_cast<int4*>(s_words_2_1_1)[0];
                    } else if (w_kind_1 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_2_1_1)[0] = reinterpret_cast<int4*>(s_words_2_1_1)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_2_1_1)[0] = reinterpret_cast<int4*>(s_words_2_1_1)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_row_lo_3_1 = tile_m_1 * 256 + 192 + warp_m_1 * 16 + (lane >> 2);
                int w_row_hi_3_1 = w_row_lo_3_1 + 8;
                float w_sa_lo_3_1 = ((w_row_lo_3_1 < M) ? act_scale[w_row_lo_3_1] : 0.0f);
                float w_sa_hi_3_1 = ((w_row_hi_3_1 < M) ? act_scale[w_row_hi_3_1] : 0.0f);
                float w_rounded_3_1[32];
                float w_ss_lo_3_1 = 0.0f;
                float w_ss_hi_3_1 = 0.0f;
                float w_scaled_3_0_1[4];
                w_scaled_3_0_1[0] = accum_1[96] * w_sa_lo_3_1 * sw_1[0];
                w_scaled_3_0_1[1] = accum_1[97] * w_sa_lo_3_1 * sw_1[1];
                w_scaled_3_0_1[2] = accum_1[98] * w_sa_hi_3_1 * sw_1[0];
                w_scaled_3_0_1[3] = accum_1[99] * w_sa_hi_3_1 * sw_1[1];
                unsigned int w_scaled_3_0_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_0_1[_lp*2 + 0], w_scaled_3_0_1[_lp*2+1 + 0]));
                    w_scaled_3_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_84[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_84[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_0_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_84[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_0_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_3_1[0] = _unpack_bf16_84[0];
                w_rounded_3_1[1] = _unpack_bf16_84[1];
                w_rounded_3_1[2] = _unpack_bf16_84[2];
                w_rounded_3_1[3] = _unpack_bf16_84[3];
                w_ss_lo_3_1 += __fmaf_rn(_unpack_bf16_84[0], _unpack_bf16_84[0], _unpack_bf16_84[1] * _unpack_bf16_84[1]);
                w_ss_hi_3_1 += __fmaf_rn(_unpack_bf16_84[2], _unpack_bf16_84[2], _unpack_bf16_84[3] * _unpack_bf16_84[3]);
                float w_scaled_3_1_1[4];
                w_scaled_3_1_1[0] = accum_1[100] * w_sa_lo_3_1 * sw_1[2];
                w_scaled_3_1_1[1] = accum_1[101] * w_sa_lo_3_1 * sw_1[3];
                w_scaled_3_1_1[2] = accum_1[102] * w_sa_hi_3_1 * sw_1[2];
                w_scaled_3_1_1[3] = accum_1[103] * w_sa_hi_3_1 * sw_1[3];
                unsigned int w_scaled_3_1_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_1_1[_lp*2 + 0], w_scaled_3_1_1[_lp*2+1 + 0]));
                    w_scaled_3_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_85[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_85[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_1_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_85[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_1_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_3_1[4] = _unpack_bf16_85[0];
                w_rounded_3_1[5] = _unpack_bf16_85[1];
                w_rounded_3_1[6] = _unpack_bf16_85[2];
                w_rounded_3_1[7] = _unpack_bf16_85[3];
                w_ss_lo_3_1 += __fmaf_rn(_unpack_bf16_85[0], _unpack_bf16_85[0], _unpack_bf16_85[1] * _unpack_bf16_85[1]);
                w_ss_hi_3_1 += __fmaf_rn(_unpack_bf16_85[2], _unpack_bf16_85[2], _unpack_bf16_85[3] * _unpack_bf16_85[3]);
                float w_scaled_3_2_1[4];
                w_scaled_3_2_1[0] = accum_1[104] * w_sa_lo_3_1 * sw_1[4];
                w_scaled_3_2_1[1] = accum_1[105] * w_sa_lo_3_1 * sw_1[5];
                w_scaled_3_2_1[2] = accum_1[106] * w_sa_hi_3_1 * sw_1[4];
                w_scaled_3_2_1[3] = accum_1[107] * w_sa_hi_3_1 * sw_1[5];
                unsigned int w_scaled_3_2_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_2_1[_lp*2 + 0], w_scaled_3_2_1[_lp*2+1 + 0]));
                    w_scaled_3_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_86[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_86[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_2_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_86[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_2_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_3_1[8] = _unpack_bf16_86[0];
                w_rounded_3_1[9] = _unpack_bf16_86[1];
                w_rounded_3_1[10] = _unpack_bf16_86[2];
                w_rounded_3_1[11] = _unpack_bf16_86[3];
                w_ss_lo_3_1 += __fmaf_rn(_unpack_bf16_86[0], _unpack_bf16_86[0], _unpack_bf16_86[1] * _unpack_bf16_86[1]);
                w_ss_hi_3_1 += __fmaf_rn(_unpack_bf16_86[2], _unpack_bf16_86[2], _unpack_bf16_86[3] * _unpack_bf16_86[3]);
                float w_scaled_3_3_1[4];
                w_scaled_3_3_1[0] = accum_1[108] * w_sa_lo_3_1 * sw_1[6];
                w_scaled_3_3_1[1] = accum_1[109] * w_sa_lo_3_1 * sw_1[7];
                w_scaled_3_3_1[2] = accum_1[110] * w_sa_hi_3_1 * sw_1[6];
                w_scaled_3_3_1[3] = accum_1[111] * w_sa_hi_3_1 * sw_1[7];
                unsigned int w_scaled_3_3_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_3_1[_lp*2 + 0], w_scaled_3_3_1[_lp*2+1 + 0]));
                    w_scaled_3_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_87[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_87[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_3_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_87[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_3_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_3_1[12] = _unpack_bf16_87[0];
                w_rounded_3_1[13] = _unpack_bf16_87[1];
                w_rounded_3_1[14] = _unpack_bf16_87[2];
                w_rounded_3_1[15] = _unpack_bf16_87[3];
                w_ss_lo_3_1 += __fmaf_rn(_unpack_bf16_87[0], _unpack_bf16_87[0], _unpack_bf16_87[1] * _unpack_bf16_87[1]);
                w_ss_hi_3_1 += __fmaf_rn(_unpack_bf16_87[2], _unpack_bf16_87[2], _unpack_bf16_87[3] * _unpack_bf16_87[3]);
                float w_scaled_3_4_1[4];
                w_scaled_3_4_1[0] = accum_1[112] * w_sa_lo_3_1 * sw_1[8];
                w_scaled_3_4_1[1] = accum_1[113] * w_sa_lo_3_1 * sw_1[9];
                w_scaled_3_4_1[2] = accum_1[114] * w_sa_hi_3_1 * sw_1[8];
                w_scaled_3_4_1[3] = accum_1[115] * w_sa_hi_3_1 * sw_1[9];
                unsigned int w_scaled_3_4_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_4_1[_lp*2 + 0], w_scaled_3_4_1[_lp*2+1 + 0]));
                    w_scaled_3_4_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_88[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_88[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_4_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_88[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_4_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_3_1[16] = _unpack_bf16_88[0];
                w_rounded_3_1[17] = _unpack_bf16_88[1];
                w_rounded_3_1[18] = _unpack_bf16_88[2];
                w_rounded_3_1[19] = _unpack_bf16_88[3];
                w_ss_lo_3_1 += __fmaf_rn(_unpack_bf16_88[0], _unpack_bf16_88[0], _unpack_bf16_88[1] * _unpack_bf16_88[1]);
                w_ss_hi_3_1 += __fmaf_rn(_unpack_bf16_88[2], _unpack_bf16_88[2], _unpack_bf16_88[3] * _unpack_bf16_88[3]);
                float w_scaled_3_5_1[4];
                w_scaled_3_5_1[0] = accum_1[116] * w_sa_lo_3_1 * sw_1[10];
                w_scaled_3_5_1[1] = accum_1[117] * w_sa_lo_3_1 * sw_1[11];
                w_scaled_3_5_1[2] = accum_1[118] * w_sa_hi_3_1 * sw_1[10];
                w_scaled_3_5_1[3] = accum_1[119] * w_sa_hi_3_1 * sw_1[11];
                unsigned int w_scaled_3_5_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_5_1[_lp*2 + 0], w_scaled_3_5_1[_lp*2+1 + 0]));
                    w_scaled_3_5_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_89[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_89[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_5_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_89[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_5_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_3_1[20] = _unpack_bf16_89[0];
                w_rounded_3_1[21] = _unpack_bf16_89[1];
                w_rounded_3_1[22] = _unpack_bf16_89[2];
                w_rounded_3_1[23] = _unpack_bf16_89[3];
                w_ss_lo_3_1 += __fmaf_rn(_unpack_bf16_89[0], _unpack_bf16_89[0], _unpack_bf16_89[1] * _unpack_bf16_89[1]);
                w_ss_hi_3_1 += __fmaf_rn(_unpack_bf16_89[2], _unpack_bf16_89[2], _unpack_bf16_89[3] * _unpack_bf16_89[3]);
                float w_scaled_3_6_1[4];
                w_scaled_3_6_1[0] = accum_1[120] * w_sa_lo_3_1 * sw_1[12];
                w_scaled_3_6_1[1] = accum_1[121] * w_sa_lo_3_1 * sw_1[13];
                w_scaled_3_6_1[2] = accum_1[122] * w_sa_hi_3_1 * sw_1[12];
                w_scaled_3_6_1[3] = accum_1[123] * w_sa_hi_3_1 * sw_1[13];
                unsigned int w_scaled_3_6_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_6_1[_lp*2 + 0], w_scaled_3_6_1[_lp*2+1 + 0]));
                    w_scaled_3_6_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_90[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_90[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_6_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_90[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_6_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_3_1[24] = _unpack_bf16_90[0];
                w_rounded_3_1[25] = _unpack_bf16_90[1];
                w_rounded_3_1[26] = _unpack_bf16_90[2];
                w_rounded_3_1[27] = _unpack_bf16_90[3];
                w_ss_lo_3_1 += __fmaf_rn(_unpack_bf16_90[0], _unpack_bf16_90[0], _unpack_bf16_90[1] * _unpack_bf16_90[1]);
                w_ss_hi_3_1 += __fmaf_rn(_unpack_bf16_90[2], _unpack_bf16_90[2], _unpack_bf16_90[3] * _unpack_bf16_90[3]);
                float w_scaled_3_7_1[4];
                w_scaled_3_7_1[0] = accum_1[124] * w_sa_lo_3_1 * sw_1[14];
                w_scaled_3_7_1[1] = accum_1[125] * w_sa_lo_3_1 * sw_1[15];
                w_scaled_3_7_1[2] = accum_1[126] * w_sa_hi_3_1 * sw_1[14];
                w_scaled_3_7_1[3] = accum_1[127] * w_sa_hi_3_1 * sw_1[15];
                unsigned int w_scaled_3_7_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_scaled_3_7_1[_lp*2 + 0], w_scaled_3_7_1[_lp*2+1 + 0]));
                    w_scaled_3_7_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_91[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_91[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_7_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_91[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_scaled_3_7_bf16_1[_pair]) & 0xffff0000u);
                }
                w_rounded_3_1[28] = _unpack_bf16_91[0];
                w_rounded_3_1[29] = _unpack_bf16_91[1];
                w_rounded_3_1[30] = _unpack_bf16_91[2];
                w_rounded_3_1[31] = _unpack_bf16_91[3];
                w_ss_lo_3_1 += __fmaf_rn(_unpack_bf16_91[0], _unpack_bf16_91[0], _unpack_bf16_91[1] * _unpack_bf16_91[1]);
                w_ss_hi_3_1 += __fmaf_rn(_unpack_bf16_91[2], _unpack_bf16_91[2], _unpack_bf16_91[3] * _unpack_bf16_91[3]);
                float w_rstd_lo_3_1 = 1.0f;
                float w_rstd_hi_3_1 = 1.0f;
                if (w_kind_1 < 2) {
                    float _shfl_xor_sync_60_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_3_1, 1);
                    w_ss_lo_3_1 += _shfl_xor_sync_60_1;
                    float _shfl_xor_sync_61_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_lo_3_1, 2);
                    w_ss_lo_3_1 += _shfl_xor_sync_61_1;
                    float _shfl_xor_sync_62_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_3_1, 1);
                    w_ss_hi_3_1 += _shfl_xor_sync_62_1;
                    float _shfl_xor_sync_63_1 = __shfl_xor_sync(0xFFFFFFFF, w_ss_hi_3_1, 2);
                    w_ss_hi_3_1 += _shfl_xor_sync_63_1;
                    int w_srow_lo_3_1 = warp_m_1 * 16 + (lane >> 2);
                    if ((lane & 3) == 0) {
                        rowstat[warp_n_1 * 64 + w_srow_lo_3_1] = w_ss_lo_3_1;
                        rowstat[warp_n_1 * 64 + w_srow_lo_3_1 + 8] = w_ss_hi_3_1;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float w_other_lo_3_1 = rowstat[(1 - warp_n_1) * 64 + w_srow_lo_3_1];
                    float w_other_hi_3_1 = rowstat[(1 - warp_n_1) * 64 + w_srow_lo_3_1 + 8];
                    float _fdiv_65_1 = __fdiv_rn(w_ss_lo_3_1 + w_other_lo_3_1, 128.0f);
                    float _rsqrt_64_1 = rsqrtf(_fdiv_65_1 + eps);
                    w_rstd_lo_3_1 = _rsqrt_64_1;
                    float _fdiv_67_1 = __fdiv_rn(w_ss_hi_3_1 + w_other_hi_3_1, 128.0f);
                    float _rsqrt_66_1 = rsqrtf(_fdiv_67_1 + eps);
                    w_rstd_hi_3_1 = _rsqrt_66_1;
                }
                int w_srow_st_3_1 = warp_m_1 * 16 + (lane >> 3 & 1) * 8 + (lane & 7);
                float w_normed_3_0_1[4];
                if (w_kind_1 < 2) {
                    w_normed_3_0_1[0] = w_rounded_3_1[0] * w_rstd_lo_3_1 * nw_1[0];
                    w_normed_3_0_1[1] = w_rounded_3_1[1] * w_rstd_lo_3_1 * nw_1[1];
                    w_normed_3_0_1[2] = w_rounded_3_1[2] * w_rstd_hi_3_1 * nw_1[0];
                    w_normed_3_0_1[3] = w_rounded_3_1[3] * w_rstd_hi_3_1 * nw_1[1];
                } else {
                    w_normed_3_0_1[0] = w_rounded_3_1[0];
                    w_normed_3_0_1[1] = w_rounded_3_1[1];
                    w_normed_3_0_1[2] = w_rounded_3_1[2];
                    w_normed_3_0_1[3] = w_rounded_3_1[3];
                }
                unsigned int w_normed_3_0_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_0_1[_lp*2 + 0], w_normed_3_0_1[_lp*2+1 + 0]));
                    w_normed_3_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_68 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_3_1 * 256) + (unsigned int)(warp_n_1 * 64 * 2 ^ (w_srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_68), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_0_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_0_bf16_1[1]))
                    : "memory");
                float w_normed_3_1_1[4];
                if (w_kind_1 < 2) {
                    w_normed_3_1_1[0] = w_rounded_3_1[4] * w_rstd_lo_3_1 * nw_1[2];
                    w_normed_3_1_1[1] = w_rounded_3_1[5] * w_rstd_lo_3_1 * nw_1[3];
                    w_normed_3_1_1[2] = w_rounded_3_1[6] * w_rstd_hi_3_1 * nw_1[2];
                    w_normed_3_1_1[3] = w_rounded_3_1[7] * w_rstd_hi_3_1 * nw_1[3];
                } else {
                    w_normed_3_1_1[0] = w_rounded_3_1[4];
                    w_normed_3_1_1[1] = w_rounded_3_1[5];
                    w_normed_3_1_1[2] = w_rounded_3_1[6];
                    w_normed_3_1_1[3] = w_rounded_3_1[7];
                }
                unsigned int w_normed_3_1_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_1_1[_lp*2 + 0], w_normed_3_1_1[_lp*2+1 + 0]));
                    w_normed_3_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_69 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_3_1 * 256) + (unsigned int)((warp_n_1 * 64 + 8) * 2 ^ (w_srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_69), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_1_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_1_bf16_1[1]))
                    : "memory");
                float w_normed_3_2_1[4];
                if (w_kind_1 < 2) {
                    w_normed_3_2_1[0] = w_rounded_3_1[8] * w_rstd_lo_3_1 * nw_1[4];
                    w_normed_3_2_1[1] = w_rounded_3_1[9] * w_rstd_lo_3_1 * nw_1[5];
                    w_normed_3_2_1[2] = w_rounded_3_1[10] * w_rstd_hi_3_1 * nw_1[4];
                    w_normed_3_2_1[3] = w_rounded_3_1[11] * w_rstd_hi_3_1 * nw_1[5];
                } else {
                    w_normed_3_2_1[0] = w_rounded_3_1[8];
                    w_normed_3_2_1[1] = w_rounded_3_1[9];
                    w_normed_3_2_1[2] = w_rounded_3_1[10];
                    w_normed_3_2_1[3] = w_rounded_3_1[11];
                }
                unsigned int w_normed_3_2_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_2_1[_lp*2 + 0], w_normed_3_2_1[_lp*2+1 + 0]));
                    w_normed_3_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_70 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_3_1 * 256) + (unsigned int)((warp_n_1 * 64 + 16) * 2 ^ (w_srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_70), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_2_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_2_bf16_1[1]))
                    : "memory");
                float w_normed_3_3_1[4];
                if (w_kind_1 < 2) {
                    w_normed_3_3_1[0] = w_rounded_3_1[12] * w_rstd_lo_3_1 * nw_1[6];
                    w_normed_3_3_1[1] = w_rounded_3_1[13] * w_rstd_lo_3_1 * nw_1[7];
                    w_normed_3_3_1[2] = w_rounded_3_1[14] * w_rstd_hi_3_1 * nw_1[6];
                    w_normed_3_3_1[3] = w_rounded_3_1[15] * w_rstd_hi_3_1 * nw_1[7];
                } else {
                    w_normed_3_3_1[0] = w_rounded_3_1[12];
                    w_normed_3_3_1[1] = w_rounded_3_1[13];
                    w_normed_3_3_1[2] = w_rounded_3_1[14];
                    w_normed_3_3_1[3] = w_rounded_3_1[15];
                }
                unsigned int w_normed_3_3_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_3_1[_lp*2 + 0], w_normed_3_3_1[_lp*2+1 + 0]));
                    w_normed_3_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_71 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_3_1 * 256) + (unsigned int)((warp_n_1 * 64 + 24) * 2 ^ (w_srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_71), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_3_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_3_bf16_1[1]))
                    : "memory");
                float w_normed_3_4_1[4];
                if (w_kind_1 < 2) {
                    w_normed_3_4_1[0] = w_rounded_3_1[16] * w_rstd_lo_3_1 * nw_1[8];
                    w_normed_3_4_1[1] = w_rounded_3_1[17] * w_rstd_lo_3_1 * nw_1[9];
                    w_normed_3_4_1[2] = w_rounded_3_1[18] * w_rstd_hi_3_1 * nw_1[8];
                    w_normed_3_4_1[3] = w_rounded_3_1[19] * w_rstd_hi_3_1 * nw_1[9];
                } else {
                    w_normed_3_4_1[0] = w_rounded_3_1[16];
                    w_normed_3_4_1[1] = w_rounded_3_1[17];
                    w_normed_3_4_1[2] = w_rounded_3_1[18];
                    w_normed_3_4_1[3] = w_rounded_3_1[19];
                }
                unsigned int w_normed_3_4_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_4_1[_lp*2 + 0], w_normed_3_4_1[_lp*2+1 + 0]));
                    w_normed_3_4_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_72 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_3_1 * 256) + (unsigned int)((warp_n_1 * 64 + 32) * 2 ^ (w_srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_72), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_4_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_4_bf16_1[1]))
                    : "memory");
                float w_normed_3_5_1[4];
                if (w_kind_1 < 2) {
                    w_normed_3_5_1[0] = w_rounded_3_1[20] * w_rstd_lo_3_1 * nw_1[10];
                    w_normed_3_5_1[1] = w_rounded_3_1[21] * w_rstd_lo_3_1 * nw_1[11];
                    w_normed_3_5_1[2] = w_rounded_3_1[22] * w_rstd_hi_3_1 * nw_1[10];
                    w_normed_3_5_1[3] = w_rounded_3_1[23] * w_rstd_hi_3_1 * nw_1[11];
                } else {
                    w_normed_3_5_1[0] = w_rounded_3_1[20];
                    w_normed_3_5_1[1] = w_rounded_3_1[21];
                    w_normed_3_5_1[2] = w_rounded_3_1[22];
                    w_normed_3_5_1[3] = w_rounded_3_1[23];
                }
                unsigned int w_normed_3_5_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_5_1[_lp*2 + 0], w_normed_3_5_1[_lp*2+1 + 0]));
                    w_normed_3_5_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_73 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_3_1 * 256) + (unsigned int)((warp_n_1 * 64 + 40) * 2 ^ (w_srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_73), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_5_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_5_bf16_1[1]))
                    : "memory");
                float w_normed_3_6_1[4];
                if (w_kind_1 < 2) {
                    w_normed_3_6_1[0] = w_rounded_3_1[24] * w_rstd_lo_3_1 * nw_1[12];
                    w_normed_3_6_1[1] = w_rounded_3_1[25] * w_rstd_lo_3_1 * nw_1[13];
                    w_normed_3_6_1[2] = w_rounded_3_1[26] * w_rstd_hi_3_1 * nw_1[12];
                    w_normed_3_6_1[3] = w_rounded_3_1[27] * w_rstd_hi_3_1 * nw_1[13];
                } else {
                    w_normed_3_6_1[0] = w_rounded_3_1[24];
                    w_normed_3_6_1[1] = w_rounded_3_1[25];
                    w_normed_3_6_1[2] = w_rounded_3_1[26];
                    w_normed_3_6_1[3] = w_rounded_3_1[27];
                }
                unsigned int w_normed_3_6_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_6_1[_lp*2 + 0], w_normed_3_6_1[_lp*2+1 + 0]));
                    w_normed_3_6_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_74 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_3_1 * 256) + (unsigned int)((warp_n_1 * 64 + 48) * 2 ^ (w_srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_74), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_6_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_6_bf16_1[1]))
                    : "memory");
                float w_normed_3_7_1[4];
                if (w_kind_1 < 2) {
                    w_normed_3_7_1[0] = w_rounded_3_1[28] * w_rstd_lo_3_1 * nw_1[14];
                    w_normed_3_7_1[1] = w_rounded_3_1[29] * w_rstd_lo_3_1 * nw_1[15];
                    w_normed_3_7_1[2] = w_rounded_3_1[30] * w_rstd_hi_3_1 * nw_1[14];
                    w_normed_3_7_1[3] = w_rounded_3_1[31] * w_rstd_hi_3_1 * nw_1[15];
                } else {
                    w_normed_3_7_1[0] = w_rounded_3_1[28];
                    w_normed_3_7_1[1] = w_rounded_3_1[29];
                    w_normed_3_7_1[2] = w_rounded_3_1[30];
                    w_normed_3_7_1[3] = w_rounded_3_1[31];
                }
                unsigned int w_normed_3_7_bf16_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_normed_3_7_1[_lp*2 + 0], w_normed_3_7_1[_lp*2+1 + 0]));
                    w_normed_3_7_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_75 = static_cast<uint32_t>(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_st_3_1 * 256) + (unsigned int)((warp_n_1 * 64 + 56) * 2 ^ (w_srow_st_3_1 & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_75), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_7_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_normed_3_7_bf16_1[1]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int w_srow_3_1 = role_tid_1 >> 2;
                int w_c_3_1 = role_tid_1 & 3;
                int w_grow_3_1 = tile_m_1 * 256 + 192 + w_srow_3_1;
                int w_chunk_lo_3_1 = ((w_c_3_1 < 3) ? w_c_3_1 * 16 : 96);
                int w_chunk_hi_3_1 = ((w_c_3_1 < 3) ? w_chunk_lo_3_1 + 48 : 112);
                if (w_grow_3_1 < M) {
                    unsigned int w_x_lo_3_1[8];
                    unsigned int w_x_hi_3_1[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3_1[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_3_1 * 256) + (unsigned int)(w_chunk_lo_3_1 * 2 ^ (w_srow_3_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_lo_3_1[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_3_1 * 256) + (unsigned int)(w_chunk_lo_3_1 * 2 + 16 ^ (w_srow_3_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3_1[(0) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_3_1 * 256) + (unsigned int)(w_chunk_hi_3_1 * 2 ^ (w_srow_3_1 & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_x_hi_3_1[(4) + 3]))
                        : "r"(A_stage_addr + free_stage_1 * 16384 + (unsigned int)(w_srow_3_1 * 256) + (unsigned int)(w_chunk_hi_3_1 * 2 + 16 ^ (w_srow_3_1 & 7) << 4)));
                    if (w_c_3_1 < 3) {
                        if (w_kind_1 < 2) {
                            float _unpack_bf16_92[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_92[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_3_1[_pair]) << 16);
                                (&_unpack_bf16_92[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_3_1[_pair]) & 0xffff0000u);
                            }
                            float _unpack_bf16_93[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&_unpack_bf16_93[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_3_1[_pair]) << 16);
                                (&_unpack_bf16_93[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_3_1[_pair]) & 0xffff0000u);
                            }
                            float _vec_load_28[8];
                            {
                                const uint4* _vptr_76 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_3_1 * 96 + w_chunk_lo_3_1) + 0);
                                uint4 _vld_76[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_76[_blk] = _vptr_76[_blk];
                                    uint32_t* _vpairs_76 = reinterpret_cast<uint32_t*>(&_vld_76[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_28[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_76[_pair]) << 16);
                                        (&_vec_load_28[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_76[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_29[8];
                            {
                                const uint4* _vptr_77 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_3_1 * 96 + w_chunk_lo_3_1 + 8) + 0);
                                uint4 _vld_77[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_77[_blk] = _vptr_77[_blk];
                                    uint32_t* _vpairs_77 = reinterpret_cast<uint32_t*>(&_vld_77[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_29[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_77[_pair]) << 16);
                                        (&_vec_load_29[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_77[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_30[8];
                            {
                                const uint4* _vptr_78 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_3_1 * 96 + 48 + w_chunk_lo_3_1) + 0);
                                uint4 _vld_78[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_78[_blk] = _vptr_78[_blk];
                                    uint32_t* _vpairs_78 = reinterpret_cast<uint32_t*>(&_vld_78[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_30[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) << 16);
                                        (&_vec_load_30[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_31[8];
                            {
                                const uint4* _vptr_79 = reinterpret_cast<const uint4*>(rope_cos_sin + (w_grow_3_1 * 96 + 48 + w_chunk_lo_3_1 + 8) + 0);
                                uint4 _vld_79[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_79[_blk] = _vptr_79[_blk];
                                    uint32_t* _vpairs_79 = reinterpret_cast<uint32_t*>(&_vld_79[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_31[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_79[_pair]) << 16);
                                        (&_vec_load_31[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_79[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float w_o_lo_3_1[16];
                            float w_o_hi_3_1[16];
                            float w_cv0_3_0_1 = _vec_load_28[0];
                            float w_sv0_3_0_1 = _vec_load_30[0];
                            float w_cv1_3_0_1 = _vec_load_29[0];
                            float w_sv1_3_0_1 = _vec_load_31[0];
                            w_o_lo_3_1[0] = __fmaf_rn(_unpack_bf16_92[0], w_cv0_3_0_1, -(_unpack_bf16_93[0] * w_sv0_3_0_1));
                            w_o_hi_3_1[0] = __fmaf_rn(_unpack_bf16_93[0], w_cv0_3_0_1, _unpack_bf16_92[0] * w_sv0_3_0_1);
                            w_o_lo_3_1[8] = __fmaf_rn(_unpack_bf16_92[8], w_cv1_3_0_1, -(_unpack_bf16_93[8] * w_sv1_3_0_1));
                            w_o_hi_3_1[8] = __fmaf_rn(_unpack_bf16_93[8], w_cv1_3_0_1, _unpack_bf16_92[8] * w_sv1_3_0_1);
                            float w_cv0_3_1_1 = _vec_load_28[1];
                            float w_sv0_3_1_1 = _vec_load_30[1];
                            float w_cv1_3_1_1 = _vec_load_29[1];
                            float w_sv1_3_1_1 = _vec_load_31[1];
                            w_o_lo_3_1[1] = __fmaf_rn(_unpack_bf16_92[1], w_cv0_3_1_1, -(_unpack_bf16_93[1] * w_sv0_3_1_1));
                            w_o_hi_3_1[1] = __fmaf_rn(_unpack_bf16_93[1], w_cv0_3_1_1, _unpack_bf16_92[1] * w_sv0_3_1_1);
                            w_o_lo_3_1[9] = __fmaf_rn(_unpack_bf16_92[9], w_cv1_3_1_1, -(_unpack_bf16_93[9] * w_sv1_3_1_1));
                            w_o_hi_3_1[9] = __fmaf_rn(_unpack_bf16_93[9], w_cv1_3_1_1, _unpack_bf16_92[9] * w_sv1_3_1_1);
                            float w_cv0_3_2_1 = _vec_load_28[2];
                            float w_sv0_3_2_1 = _vec_load_30[2];
                            float w_cv1_3_2_1 = _vec_load_29[2];
                            float w_sv1_3_2_1 = _vec_load_31[2];
                            w_o_lo_3_1[2] = __fmaf_rn(_unpack_bf16_92[2], w_cv0_3_2_1, -(_unpack_bf16_93[2] * w_sv0_3_2_1));
                            w_o_hi_3_1[2] = __fmaf_rn(_unpack_bf16_93[2], w_cv0_3_2_1, _unpack_bf16_92[2] * w_sv0_3_2_1);
                            w_o_lo_3_1[10] = __fmaf_rn(_unpack_bf16_92[10], w_cv1_3_2_1, -(_unpack_bf16_93[10] * w_sv1_3_2_1));
                            w_o_hi_3_1[10] = __fmaf_rn(_unpack_bf16_93[10], w_cv1_3_2_1, _unpack_bf16_92[10] * w_sv1_3_2_1);
                            float w_cv0_3_3_1 = _vec_load_28[3];
                            float w_sv0_3_3_1 = _vec_load_30[3];
                            float w_cv1_3_3_1 = _vec_load_29[3];
                            float w_sv1_3_3_1 = _vec_load_31[3];
                            w_o_lo_3_1[3] = __fmaf_rn(_unpack_bf16_92[3], w_cv0_3_3_1, -(_unpack_bf16_93[3] * w_sv0_3_3_1));
                            w_o_hi_3_1[3] = __fmaf_rn(_unpack_bf16_93[3], w_cv0_3_3_1, _unpack_bf16_92[3] * w_sv0_3_3_1);
                            w_o_lo_3_1[11] = __fmaf_rn(_unpack_bf16_92[11], w_cv1_3_3_1, -(_unpack_bf16_93[11] * w_sv1_3_3_1));
                            w_o_hi_3_1[11] = __fmaf_rn(_unpack_bf16_93[11], w_cv1_3_3_1, _unpack_bf16_92[11] * w_sv1_3_3_1);
                            float w_cv0_3_4_1 = _vec_load_28[4];
                            float w_sv0_3_4_1 = _vec_load_30[4];
                            float w_cv1_3_4_1 = _vec_load_29[4];
                            float w_sv1_3_4_1 = _vec_load_31[4];
                            w_o_lo_3_1[4] = __fmaf_rn(_unpack_bf16_92[4], w_cv0_3_4_1, -(_unpack_bf16_93[4] * w_sv0_3_4_1));
                            w_o_hi_3_1[4] = __fmaf_rn(_unpack_bf16_93[4], w_cv0_3_4_1, _unpack_bf16_92[4] * w_sv0_3_4_1);
                            w_o_lo_3_1[12] = __fmaf_rn(_unpack_bf16_92[12], w_cv1_3_4_1, -(_unpack_bf16_93[12] * w_sv1_3_4_1));
                            w_o_hi_3_1[12] = __fmaf_rn(_unpack_bf16_93[12], w_cv1_3_4_1, _unpack_bf16_92[12] * w_sv1_3_4_1);
                            float w_cv0_3_5_1 = _vec_load_28[5];
                            float w_sv0_3_5_1 = _vec_load_30[5];
                            float w_cv1_3_5_1 = _vec_load_29[5];
                            float w_sv1_3_5_1 = _vec_load_31[5];
                            w_o_lo_3_1[5] = __fmaf_rn(_unpack_bf16_92[5], w_cv0_3_5_1, -(_unpack_bf16_93[5] * w_sv0_3_5_1));
                            w_o_hi_3_1[5] = __fmaf_rn(_unpack_bf16_93[5], w_cv0_3_5_1, _unpack_bf16_92[5] * w_sv0_3_5_1);
                            w_o_lo_3_1[13] = __fmaf_rn(_unpack_bf16_92[13], w_cv1_3_5_1, -(_unpack_bf16_93[13] * w_sv1_3_5_1));
                            w_o_hi_3_1[13] = __fmaf_rn(_unpack_bf16_93[13], w_cv1_3_5_1, _unpack_bf16_92[13] * w_sv1_3_5_1);
                            float w_cv0_3_6_1 = _vec_load_28[6];
                            float w_sv0_3_6_1 = _vec_load_30[6];
                            float w_cv1_3_6_1 = _vec_load_29[6];
                            float w_sv1_3_6_1 = _vec_load_31[6];
                            w_o_lo_3_1[6] = __fmaf_rn(_unpack_bf16_92[6], w_cv0_3_6_1, -(_unpack_bf16_93[6] * w_sv0_3_6_1));
                            w_o_hi_3_1[6] = __fmaf_rn(_unpack_bf16_93[6], w_cv0_3_6_1, _unpack_bf16_92[6] * w_sv0_3_6_1);
                            w_o_lo_3_1[14] = __fmaf_rn(_unpack_bf16_92[14], w_cv1_3_6_1, -(_unpack_bf16_93[14] * w_sv1_3_6_1));
                            w_o_hi_3_1[14] = __fmaf_rn(_unpack_bf16_93[14], w_cv1_3_6_1, _unpack_bf16_92[14] * w_sv1_3_6_1);
                            float w_cv0_3_7_1 = _vec_load_28[7];
                            float w_sv0_3_7_1 = _vec_load_30[7];
                            float w_cv1_3_7_1 = _vec_load_29[7];
                            float w_sv1_3_7_1 = _vec_load_31[7];
                            w_o_lo_3_1[7] = __fmaf_rn(_unpack_bf16_92[7], w_cv0_3_7_1, -(_unpack_bf16_93[7] * w_sv0_3_7_1));
                            w_o_hi_3_1[7] = __fmaf_rn(_unpack_bf16_93[7], w_cv0_3_7_1, _unpack_bf16_92[7] * w_sv0_3_7_1);
                            w_o_lo_3_1[15] = __fmaf_rn(_unpack_bf16_92[15], w_cv1_3_7_1, -(_unpack_bf16_93[15] * w_sv1_3_7_1));
                            w_o_hi_3_1[15] = __fmaf_rn(_unpack_bf16_93[15], w_cv1_3_7_1, _unpack_bf16_92[15] * w_sv1_3_7_1);
                            unsigned int w_o_lo_3_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_lo_3_1[_lp*2 + 0], w_o_lo_3_1[_lp*2+1 + 0]));
                                w_o_lo_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int w_o_hi_3_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_o_hi_3_1[_lp*2 + 0], w_o_hi_3_1[_lp*2+1 + 0]));
                                w_o_hi_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            w_x_lo_3_1[0] = w_o_lo_3_bf16_1[0];
                            w_x_hi_3_1[0] = w_o_hi_3_bf16_1[0];
                            w_x_lo_3_1[1] = w_o_lo_3_bf16_1[1];
                            w_x_hi_3_1[1] = w_o_hi_3_bf16_1[1];
                            w_x_lo_3_1[2] = w_o_lo_3_bf16_1[2];
                            w_x_hi_3_1[2] = w_o_hi_3_bf16_1[2];
                            w_x_lo_3_1[3] = w_o_lo_3_bf16_1[3];
                            w_x_hi_3_1[3] = w_o_hi_3_bf16_1[3];
                            w_x_lo_3_1[4] = w_o_lo_3_bf16_1[4];
                            w_x_hi_3_1[4] = w_o_hi_3_bf16_1[4];
                            w_x_lo_3_1[5] = w_o_lo_3_bf16_1[5];
                            w_x_hi_3_1[5] = w_o_hi_3_bf16_1[5];
                            w_x_lo_3_1[6] = w_o_lo_3_bf16_1[6];
                            w_x_hi_3_1[6] = w_o_hi_3_bf16_1[6];
                            w_x_lo_3_1[7] = w_o_lo_3_bf16_1[7];
                            w_x_hi_3_1[7] = w_o_hi_3_bf16_1[7];
                        }
                    }
                    unsigned int w_q_words_3_1[4];
                    float w_q_vals_3_1[16];
                    float _unpack_bf16_94[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_94[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_lo_3_1[_pair]) << 16);
                        (&_unpack_bf16_94[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_lo_3_1[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_3_1[0] = _unpack_bf16_94[0] * w_out_scale_sel_1;
                    w_q_vals_3_1[1] = _unpack_bf16_94[1] * w_out_scale_sel_1;
                    w_q_vals_3_1[2] = _unpack_bf16_94[2] * w_out_scale_sel_1;
                    w_q_vals_3_1[3] = _unpack_bf16_94[3] * w_out_scale_sel_1;
                    w_q_vals_3_1[4] = _unpack_bf16_94[4] * w_out_scale_sel_1;
                    w_q_vals_3_1[5] = _unpack_bf16_94[5] * w_out_scale_sel_1;
                    w_q_vals_3_1[6] = _unpack_bf16_94[6] * w_out_scale_sel_1;
                    w_q_vals_3_1[7] = _unpack_bf16_94[7] * w_out_scale_sel_1;
                    w_q_vals_3_1[8] = _unpack_bf16_94[8] * w_out_scale_sel_1;
                    w_q_vals_3_1[9] = _unpack_bf16_94[9] * w_out_scale_sel_1;
                    w_q_vals_3_1[10] = _unpack_bf16_94[10] * w_out_scale_sel_1;
                    w_q_vals_3_1[11] = _unpack_bf16_94[11] * w_out_scale_sel_1;
                    w_q_vals_3_1[12] = _unpack_bf16_94[12] * w_out_scale_sel_1;
                    w_q_vals_3_1[13] = _unpack_bf16_94[13] * w_out_scale_sel_1;
                    w_q_vals_3_1[14] = _unpack_bf16_94[14] * w_out_scale_sel_1;
                    w_q_vals_3_1[15] = _unpack_bf16_94[15] * w_out_scale_sel_1;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_3_1[0]), "f"(w_q_vals_3_1[1]),
                                               "f"(w_q_vals_3_1[2]), "f"(w_q_vals_3_1[3]));
                        w_q_words_3_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_3_1[4]), "f"(w_q_vals_3_1[5]),
                                               "f"(w_q_vals_3_1[6]), "f"(w_q_vals_3_1[7]));
                        w_q_words_3_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_3_1[8]), "f"(w_q_vals_3_1[9]),
                                               "f"(w_q_vals_3_1[10]), "f"(w_q_vals_3_1[11]));
                        w_q_words_3_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_3_1[12]), "f"(w_q_vals_3_1[13]),
                                               "f"(w_q_vals_3_1[14]), "f"(w_q_vals_3_1[15]));
                        w_q_words_3_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(w_srow_3_1 * 128) + (unsigned int)(w_chunk_lo_3_1 ^ (w_srow_3_1 & 7) << 4)), "r"(w_q_words_3_1[0]), "r"(w_q_words_3_1[1]), "r"(w_q_words_3_1[2]), "r"(w_q_words_3_1[3]) : "memory");
                    float _unpack_bf16_95[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_95[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_x_hi_3_1[_pair]) << 16);
                        (&_unpack_bf16_95[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_x_hi_3_1[_pair]) & 0xffff0000u);
                    }
                    w_q_vals_3_1[0] = _unpack_bf16_95[0] * w_out_scale_sel_1;
                    w_q_vals_3_1[1] = _unpack_bf16_95[1] * w_out_scale_sel_1;
                    w_q_vals_3_1[2] = _unpack_bf16_95[2] * w_out_scale_sel_1;
                    w_q_vals_3_1[3] = _unpack_bf16_95[3] * w_out_scale_sel_1;
                    w_q_vals_3_1[4] = _unpack_bf16_95[4] * w_out_scale_sel_1;
                    w_q_vals_3_1[5] = _unpack_bf16_95[5] * w_out_scale_sel_1;
                    w_q_vals_3_1[6] = _unpack_bf16_95[6] * w_out_scale_sel_1;
                    w_q_vals_3_1[7] = _unpack_bf16_95[7] * w_out_scale_sel_1;
                    w_q_vals_3_1[8] = _unpack_bf16_95[8] * w_out_scale_sel_1;
                    w_q_vals_3_1[9] = _unpack_bf16_95[9] * w_out_scale_sel_1;
                    w_q_vals_3_1[10] = _unpack_bf16_95[10] * w_out_scale_sel_1;
                    w_q_vals_3_1[11] = _unpack_bf16_95[11] * w_out_scale_sel_1;
                    w_q_vals_3_1[12] = _unpack_bf16_95[12] * w_out_scale_sel_1;
                    w_q_vals_3_1[13] = _unpack_bf16_95[13] * w_out_scale_sel_1;
                    w_q_vals_3_1[14] = _unpack_bf16_95[14] * w_out_scale_sel_1;
                    w_q_vals_3_1[15] = _unpack_bf16_95[15] * w_out_scale_sel_1;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(w_q_vals_3_1[0]), "f"(w_q_vals_3_1[1]),
                                               "f"(w_q_vals_3_1[2]), "f"(w_q_vals_3_1[3]));
                        w_q_words_3_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_3_1[4]), "f"(w_q_vals_3_1[5]),
                                               "f"(w_q_vals_3_1[6]), "f"(w_q_vals_3_1[7]));
                        w_q_words_3_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_3_1[8]), "f"(w_q_vals_3_1[9]),
                                               "f"(w_q_vals_3_1[10]), "f"(w_q_vals_3_1[11]));
                        w_q_words_3_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(w_q_vals_3_1[12]), "f"(w_q_vals_3_1[13]),
                                               "f"(w_q_vals_3_1[14]), "f"(w_q_vals_3_1[15]));
                        w_q_words_3_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(w_srow_3_1 * 128) + (unsigned int)(w_chunk_hi_3_1 ^ (w_srow_3_1 & 7) << 4)), "r"(w_q_words_3_1[0]), "r"(w_q_words_3_1[1]), "r"(w_q_words_3_1[2]), "r"(w_q_words_3_1[3]) : "memory");
                }
                __syncwarp();
                int s_row_3_0_1 = (role_tid_1 >> 5) * 8 + lane / 8;
                int s_piece_3_0_1 = lane % 8;
                unsigned int s_words_3_0_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_0_1[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_row_3_0_1 * 128) + (unsigned int)(s_piece_3_0_1 * 16 ^ (s_row_3_0_1 & 7) << 4)));
                int s_dst_3_0_1 = ((tile_m_1 * 256 + 192 + s_row_3_0_1) * 56 + w_head_1) * 32 + s_piece_3_0_1 * 4;
                if (tile_m_1 * 256 + 192 + s_row_3_0_1 < M) {
                    if (w_kind_1 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_3_0_1)[0] = reinterpret_cast<int4*>(s_words_3_0_1)[0];
                    } else if (w_kind_1 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_3_0_1)[0] = reinterpret_cast<int4*>(s_words_3_0_1)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_3_0_1)[0] = reinterpret_cast<int4*>(s_words_3_0_1)[0];
                    }
                }
                int s_row_3_1_1 = (role_tid_1 >> 5) * 8 + 4 + lane / 8;
                int s_piece_3_1_1 = lane % 8;
                unsigned int s_words_3_1_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s_words_3_1_1[(0) + 3]))
                    : "r"(B_stage_addr + free_stage_1 * 8192 + (unsigned int)(s_row_3_1_1 * 128) + (unsigned int)(s_piece_3_1_1 * 16 ^ (s_row_3_1_1 & 7) << 4)));
                int s_dst_3_1_1 = ((tile_m_1 * 256 + 192 + s_row_3_1_1) * 56 + w_head_1) * 32 + s_piece_3_1_1 * 4;
                if (tile_m_1 * 256 + 192 + s_row_3_1_1 < M) {
                    if (w_kind_1 == 0) {
                        reinterpret_cast<int4*>(q_out + s_dst_3_1_1)[0] = reinterpret_cast<int4*>(s_words_3_1_1)[0];
                    } else if (w_kind_1 == 1) {
                        reinterpret_cast<int4*>(k_out + s_dst_3_1_1)[0] = reinterpret_cast<int4*>(s_words_3_1_1)[0];
                    } else {
                        reinterpret_cast<int4*>(v_out + s_dst_3_1_1)[0] = reinterpret_cast<int4*>(s_words_3_1_1)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
            }
        }
    }

    // Cleanup
}

} // extern "C"
