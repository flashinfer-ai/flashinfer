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
// Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek.
// DeepGEMM portions are licensed under MIT; see DEEPGEMM_NOTICE.txt in this directory.

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
#define NUM_Q_PIPE_STAGES 2
#define NUM_KV_PIPE_STAGES 3
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_W_OFF 33792
#define SMEM_SMEM_W_STAGE_BYTES 512
#define SMEM_SMEM_W_STRIDE 1024
#define SMEM_SMEM_KV_G0_OFF 35840
#define SMEM_SMEM_KV_G0_STAGE_BYTES 8192
#define SMEM_SMEM_KV_G0_STRIDE 8192
#define SMEM_SMEM_KV_G1_OFF 60416
#define SMEM_SMEM_KV_G1_STAGE_BYTES 8192
#define SMEM_SMEM_KV_G1_STRIDE 8192
#define SMEM_SMEM_SC_G0_OFF 84992
#define SMEM_SMEM_SC_G0_STAGE_BYTES 256
#define SMEM_SMEM_SC_G0_STRIDE 1024
#define SMEM_SMEM_SC_G1_OFF 88064
#define SMEM_SMEM_SC_G1_STAGE_BYTES 256
#define SMEM_SMEM_SC_G1_STRIDE 1024
#define SMEM_TOTAL 91136
#define THREADS 384
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

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_deepgemm_sm120_paged_mqa_logits_ae634aa73adb6c9a3bfe(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KV_scales, const __grid_constant__ CUtensorMap Weights, float* __restrict__ Logits, int* __restrict__ context_lens, int* __restrict__ block_table, int* __restrict__ schedule_meta, int logits_stride, int block_table_stride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define kv_full_g0_addr (mbar_base + 32)
    #define kv_full_g1_addr (mbar_base + 56)
    #define kv_empty_g0_addr (mbar_base + 80)
    #define kv_empty_g1_addr (mbar_base + 104)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    float* smem_w = reinterpret_cast<float*>(smem_raw + 33792);
    const int smem_w_addr = smem + 33792;
    uint8_t* smem_kv_g0 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_kv_g0_addr = smem + 35840;
    uint8_t* smem_kv_g1 = reinterpret_cast<uint8_t*>(smem_raw + 60416);
    const int smem_kv_g1_addr = smem + 60416;
    float* smem_sc_g0 = reinterpret_cast<float*>(smem_raw + 84992);
    const int smem_sc_g0_addr = smem + 84992;
    float* smem_sc_g1 = reinterpret_cast<float*>(smem_raw + 88064);
    const int smem_sc_g1_addr = smem + 88064;
    if (tid == 0) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(schedule_meta + (bid * 2)))); }
    if (tid == 0) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(schedule_meta + (bid * 2 + 3)))); }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // q_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 16, 256);
            mbarrier_init(smem + 24, 256);
            // --- pipeline 'kv_pipe' ---
            // kv_full_g0: 3 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // kv_full_g1: 3 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // kv_empty_g0: 3 barriers, init_count=128
            mbarrier_init(smem + 80, 128);
            mbarrier_init(smem + 88, 128);
            mbarrier_init(smem + 96, 128);
            // kv_empty_g1: 3 barriers, init_count=128
            mbarrier_init(smem + 104, 128);
            mbarrier_init(smem + 112, 128);
            mbarrier_init(smem + 120, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    }

    // ---- Role: math_g0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // math_g0_main
            int w_in_g = warp;
            int lane_0 = lane;
            int quad = lane_0 / 4;
            int tq = lane_0 % 4;
            int a_row = w_in_g * 16 + (lane_0 & 7) + (lane_0 >> 3 & 1) * 8;
            int a_col = (lane_0 >> 4) * 16;
            int b_row_lane = lane_0 & 7;
            int b_col = (lane_0 >> 3 & 1) * 16;
            unsigned int q_stage_m = 0;
            unsigned int kv_stage_m = 0;
            unsigned int a_frag[4];
            unsigned int b_frag[2];
            unsigned int a_tile[16];
            float acc[4];
            unsigned int b_atom[128];
            int sm = bid;
            int start_q = schedule_meta[sm * 2];
            int start_kv = schedule_meta[sm * 2 + 1] * 2;
            int end_q = schedule_meta[sm * 2 + 2];
            int end_kv = schedule_meta[sm * 2 + 3] * 2;
            int q_stop = ((end_kv > 0) ? end_q + 1 : end_q);
            unsigned int _phase_q_full = 0;
            unsigned int _phase_kv_full_g0 = 0;
            #pragma unroll 1
            for (int q_atom = start_q; q_atom < q_stop; q_atom++) {
                int q_idx = q_atom / 2;
                int atom_in_q = q_atom % 2;
                int tok = q_idx * 4 + atom_in_q * 2;
                int ctx_last = context_lens[q_idx * 4 + 4 - 1];
                int num_kv = (ctx_last + 64 - 1) / 64;
                int lo = ((q_atom == start_q) ? start_kv : 0);
                int hi = ((q_atom == end_q) ? end_kv : num_kv);
                mbarrier_wait(q_full_addr + (q_stage_m) * 8, _phase_q_full);
                int b_row = b_row_lane;
                int b_cb = b_col;
                unsigned int b_addr = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row * 128) + (unsigned int)(b_cb ^ (b_row & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[0]), "=r"(b_atom[1])
                    : "r"(b_addr)
                    : "memory");
                int b_row_0 = b_row_lane;
                int b_cb_0 = b_col + 32;
                unsigned int b_addr_0 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_0 * 128) + (unsigned int)(b_cb_0 ^ (b_row_0 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[2]), "=r"(b_atom[3])
                    : "r"(b_addr_0)
                    : "memory");
                int b_row_1 = b_row_lane;
                int b_cb_1 = b_col + 64;
                unsigned int b_addr_1 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_1 * 128) + (unsigned int)(b_cb_1 ^ (b_row_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[4]), "=r"(b_atom[5])
                    : "r"(b_addr_1)
                    : "memory");
                int b_row_2 = b_row_lane;
                int b_cb_2 = b_col + 96;
                unsigned int b_addr_2 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_2 * 128) + (unsigned int)(b_cb_2 ^ (b_row_2 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[6]), "=r"(b_atom[7])
                    : "r"(b_addr_2)
                    : "memory");
                int b_row_3 = 8 + b_row_lane;
                int b_cb_3 = b_col;
                unsigned int b_addr_3 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_3 * 128) + (unsigned int)(b_cb_3 ^ (b_row_3 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[8]), "=r"(b_atom[9])
                    : "r"(b_addr_3)
                    : "memory");
                int b_row_4 = 8 + b_row_lane;
                int b_cb_4 = b_col + 32;
                unsigned int b_addr_4 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_4 * 128) + (unsigned int)(b_cb_4 ^ (b_row_4 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[10]), "=r"(b_atom[11])
                    : "r"(b_addr_4)
                    : "memory");
                int b_row_5 = 8 + b_row_lane;
                int b_cb_5 = b_col + 64;
                unsigned int b_addr_5 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_5 * 128) + (unsigned int)(b_cb_5 ^ (b_row_5 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[12]), "=r"(b_atom[13])
                    : "r"(b_addr_5)
                    : "memory");
                int b_row_6 = 8 + b_row_lane;
                int b_cb_6 = b_col + 96;
                unsigned int b_addr_6 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_6 * 128) + (unsigned int)(b_cb_6 ^ (b_row_6 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[14]), "=r"(b_atom[15])
                    : "r"(b_addr_6)
                    : "memory");
                int b_row_7 = 16 + b_row_lane;
                int b_cb_7 = b_col;
                unsigned int b_addr_7 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_7 * 128) + (unsigned int)(b_cb_7 ^ (b_row_7 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[16]), "=r"(b_atom[17])
                    : "r"(b_addr_7)
                    : "memory");
                int b_row_8 = 16 + b_row_lane;
                int b_cb_8 = b_col + 32;
                unsigned int b_addr_8 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_8 * 128) + (unsigned int)(b_cb_8 ^ (b_row_8 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[18]), "=r"(b_atom[19])
                    : "r"(b_addr_8)
                    : "memory");
                int b_row_9 = 16 + b_row_lane;
                int b_cb_9 = b_col + 64;
                unsigned int b_addr_9 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_9 * 128) + (unsigned int)(b_cb_9 ^ (b_row_9 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[20]), "=r"(b_atom[21])
                    : "r"(b_addr_9)
                    : "memory");
                int b_row_10 = 16 + b_row_lane;
                int b_cb_10 = b_col + 96;
                unsigned int b_addr_10 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_10 * 128) + (unsigned int)(b_cb_10 ^ (b_row_10 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[22]), "=r"(b_atom[23])
                    : "r"(b_addr_10)
                    : "memory");
                int b_row_11 = 24 + b_row_lane;
                int b_cb_11 = b_col;
                unsigned int b_addr_11 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_11 * 128) + (unsigned int)(b_cb_11 ^ (b_row_11 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[24]), "=r"(b_atom[25])
                    : "r"(b_addr_11)
                    : "memory");
                int b_row_12 = 24 + b_row_lane;
                int b_cb_12 = b_col + 32;
                unsigned int b_addr_12 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_12 * 128) + (unsigned int)(b_cb_12 ^ (b_row_12 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[26]), "=r"(b_atom[27])
                    : "r"(b_addr_12)
                    : "memory");
                int b_row_13 = 24 + b_row_lane;
                int b_cb_13 = b_col + 64;
                unsigned int b_addr_13 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_13 * 128) + (unsigned int)(b_cb_13 ^ (b_row_13 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[28]), "=r"(b_atom[29])
                    : "r"(b_addr_13)
                    : "memory");
                int b_row_14 = 24 + b_row_lane;
                int b_cb_14 = b_col + 96;
                unsigned int b_addr_14 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_14 * 128) + (unsigned int)(b_cb_14 ^ (b_row_14 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[30]), "=r"(b_atom[31])
                    : "r"(b_addr_14)
                    : "memory");
                int b_row_15 = 32 + b_row_lane;
                int b_cb_15 = b_col;
                unsigned int b_addr_15 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_15 * 128) + (unsigned int)(b_cb_15 ^ (b_row_15 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[32]), "=r"(b_atom[33])
                    : "r"(b_addr_15)
                    : "memory");
                int b_row_16 = 32 + b_row_lane;
                int b_cb_16 = b_col + 32;
                unsigned int b_addr_16 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_16 * 128) + (unsigned int)(b_cb_16 ^ (b_row_16 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[34]), "=r"(b_atom[35])
                    : "r"(b_addr_16)
                    : "memory");
                int b_row_17 = 32 + b_row_lane;
                int b_cb_17 = b_col + 64;
                unsigned int b_addr_17 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_17 * 128) + (unsigned int)(b_cb_17 ^ (b_row_17 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[36]), "=r"(b_atom[37])
                    : "r"(b_addr_17)
                    : "memory");
                int b_row_18 = 32 + b_row_lane;
                int b_cb_18 = b_col + 96;
                unsigned int b_addr_18 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_18 * 128) + (unsigned int)(b_cb_18 ^ (b_row_18 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[38]), "=r"(b_atom[39])
                    : "r"(b_addr_18)
                    : "memory");
                int b_row_19 = 40 + b_row_lane;
                int b_cb_19 = b_col;
                unsigned int b_addr_19 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_19 * 128) + (unsigned int)(b_cb_19 ^ (b_row_19 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[40]), "=r"(b_atom[41])
                    : "r"(b_addr_19)
                    : "memory");
                int b_row_20 = 40 + b_row_lane;
                int b_cb_20 = b_col + 32;
                unsigned int b_addr_20 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_20 * 128) + (unsigned int)(b_cb_20 ^ (b_row_20 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[42]), "=r"(b_atom[43])
                    : "r"(b_addr_20)
                    : "memory");
                int b_row_21 = 40 + b_row_lane;
                int b_cb_21 = b_col + 64;
                unsigned int b_addr_21 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_21 * 128) + (unsigned int)(b_cb_21 ^ (b_row_21 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[44]), "=r"(b_atom[45])
                    : "r"(b_addr_21)
                    : "memory");
                int b_row_22 = 40 + b_row_lane;
                int b_cb_22 = b_col + 96;
                unsigned int b_addr_22 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_22 * 128) + (unsigned int)(b_cb_22 ^ (b_row_22 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[46]), "=r"(b_atom[47])
                    : "r"(b_addr_22)
                    : "memory");
                int b_row_23 = 48 + b_row_lane;
                int b_cb_23 = b_col;
                unsigned int b_addr_23 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_23 * 128) + (unsigned int)(b_cb_23 ^ (b_row_23 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[48]), "=r"(b_atom[49])
                    : "r"(b_addr_23)
                    : "memory");
                int b_row_24 = 48 + b_row_lane;
                int b_cb_24 = b_col + 32;
                unsigned int b_addr_24 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_24 * 128) + (unsigned int)(b_cb_24 ^ (b_row_24 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[50]), "=r"(b_atom[51])
                    : "r"(b_addr_24)
                    : "memory");
                int b_row_25 = 48 + b_row_lane;
                int b_cb_25 = b_col + 64;
                unsigned int b_addr_25 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_25 * 128) + (unsigned int)(b_cb_25 ^ (b_row_25 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[52]), "=r"(b_atom[53])
                    : "r"(b_addr_25)
                    : "memory");
                int b_row_26 = 48 + b_row_lane;
                int b_cb_26 = b_col + 96;
                unsigned int b_addr_26 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_26 * 128) + (unsigned int)(b_cb_26 ^ (b_row_26 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[54]), "=r"(b_atom[55])
                    : "r"(b_addr_26)
                    : "memory");
                int b_row_27 = 56 + b_row_lane;
                int b_cb_27 = b_col;
                unsigned int b_addr_27 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_27 * 128) + (unsigned int)(b_cb_27 ^ (b_row_27 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[56]), "=r"(b_atom[57])
                    : "r"(b_addr_27)
                    : "memory");
                int b_row_28 = 56 + b_row_lane;
                int b_cb_28 = b_col + 32;
                unsigned int b_addr_28 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_28 * 128) + (unsigned int)(b_cb_28 ^ (b_row_28 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[58]), "=r"(b_atom[59])
                    : "r"(b_addr_28)
                    : "memory");
                int b_row_29 = 56 + b_row_lane;
                int b_cb_29 = b_col + 64;
                unsigned int b_addr_29 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_29 * 128) + (unsigned int)(b_cb_29 ^ (b_row_29 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[60]), "=r"(b_atom[61])
                    : "r"(b_addr_29)
                    : "memory");
                int b_row_30 = 56 + b_row_lane;
                int b_cb_30 = b_col + 96;
                unsigned int b_addr_30 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_30 * 128) + (unsigned int)(b_cb_30 ^ (b_row_30 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[62]), "=r"(b_atom[63])
                    : "r"(b_addr_30)
                    : "memory");
                int b_row_31 = 64 + b_row_lane;
                int b_cb_31 = b_col;
                unsigned int b_addr_31 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_31 * 128) + (unsigned int)(b_cb_31 ^ (b_row_31 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[64]), "=r"(b_atom[65])
                    : "r"(b_addr_31)
                    : "memory");
                int b_row_32 = 64 + b_row_lane;
                int b_cb_32 = b_col + 32;
                unsigned int b_addr_32 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_32 * 128) + (unsigned int)(b_cb_32 ^ (b_row_32 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[66]), "=r"(b_atom[67])
                    : "r"(b_addr_32)
                    : "memory");
                int b_row_33 = 64 + b_row_lane;
                int b_cb_33 = b_col + 64;
                unsigned int b_addr_33 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_33 * 128) + (unsigned int)(b_cb_33 ^ (b_row_33 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[68]), "=r"(b_atom[69])
                    : "r"(b_addr_33)
                    : "memory");
                int b_row_34 = 64 + b_row_lane;
                int b_cb_34 = b_col + 96;
                unsigned int b_addr_34 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_34 * 128) + (unsigned int)(b_cb_34 ^ (b_row_34 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[70]), "=r"(b_atom[71])
                    : "r"(b_addr_34)
                    : "memory");
                int b_row_35 = 72 + b_row_lane;
                int b_cb_35 = b_col;
                unsigned int b_addr_35 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_35 * 128) + (unsigned int)(b_cb_35 ^ (b_row_35 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[72]), "=r"(b_atom[73])
                    : "r"(b_addr_35)
                    : "memory");
                int b_row_36 = 72 + b_row_lane;
                int b_cb_36 = b_col + 32;
                unsigned int b_addr_36 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_36 * 128) + (unsigned int)(b_cb_36 ^ (b_row_36 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[74]), "=r"(b_atom[75])
                    : "r"(b_addr_36)
                    : "memory");
                int b_row_37 = 72 + b_row_lane;
                int b_cb_37 = b_col + 64;
                unsigned int b_addr_37 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_37 * 128) + (unsigned int)(b_cb_37 ^ (b_row_37 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[76]), "=r"(b_atom[77])
                    : "r"(b_addr_37)
                    : "memory");
                int b_row_38 = 72 + b_row_lane;
                int b_cb_38 = b_col + 96;
                unsigned int b_addr_38 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_38 * 128) + (unsigned int)(b_cb_38 ^ (b_row_38 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[78]), "=r"(b_atom[79])
                    : "r"(b_addr_38)
                    : "memory");
                int b_row_39 = 80 + b_row_lane;
                int b_cb_39 = b_col;
                unsigned int b_addr_39 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_39 * 128) + (unsigned int)(b_cb_39 ^ (b_row_39 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[80]), "=r"(b_atom[81])
                    : "r"(b_addr_39)
                    : "memory");
                int b_row_40 = 80 + b_row_lane;
                int b_cb_40 = b_col + 32;
                unsigned int b_addr_40 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_40 * 128) + (unsigned int)(b_cb_40 ^ (b_row_40 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[82]), "=r"(b_atom[83])
                    : "r"(b_addr_40)
                    : "memory");
                int b_row_41 = 80 + b_row_lane;
                int b_cb_41 = b_col + 64;
                unsigned int b_addr_41 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_41 * 128) + (unsigned int)(b_cb_41 ^ (b_row_41 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[84]), "=r"(b_atom[85])
                    : "r"(b_addr_41)
                    : "memory");
                int b_row_42 = 80 + b_row_lane;
                int b_cb_42 = b_col + 96;
                unsigned int b_addr_42 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_42 * 128) + (unsigned int)(b_cb_42 ^ (b_row_42 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[86]), "=r"(b_atom[87])
                    : "r"(b_addr_42)
                    : "memory");
                int b_row_43 = 88 + b_row_lane;
                int b_cb_43 = b_col;
                unsigned int b_addr_43 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_43 * 128) + (unsigned int)(b_cb_43 ^ (b_row_43 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[88]), "=r"(b_atom[89])
                    : "r"(b_addr_43)
                    : "memory");
                int b_row_44 = 88 + b_row_lane;
                int b_cb_44 = b_col + 32;
                unsigned int b_addr_44 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_44 * 128) + (unsigned int)(b_cb_44 ^ (b_row_44 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[90]), "=r"(b_atom[91])
                    : "r"(b_addr_44)
                    : "memory");
                int b_row_45 = 88 + b_row_lane;
                int b_cb_45 = b_col + 64;
                unsigned int b_addr_45 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_45 * 128) + (unsigned int)(b_cb_45 ^ (b_row_45 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[92]), "=r"(b_atom[93])
                    : "r"(b_addr_45)
                    : "memory");
                int b_row_46 = 88 + b_row_lane;
                int b_cb_46 = b_col + 96;
                unsigned int b_addr_46 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_46 * 128) + (unsigned int)(b_cb_46 ^ (b_row_46 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[94]), "=r"(b_atom[95])
                    : "r"(b_addr_46)
                    : "memory");
                int b_row_47 = 96 + b_row_lane;
                int b_cb_47 = b_col;
                unsigned int b_addr_47 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_47 * 128) + (unsigned int)(b_cb_47 ^ (b_row_47 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[96]), "=r"(b_atom[97])
                    : "r"(b_addr_47)
                    : "memory");
                int b_row_48 = 96 + b_row_lane;
                int b_cb_48 = b_col + 32;
                unsigned int b_addr_48 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_48 * 128) + (unsigned int)(b_cb_48 ^ (b_row_48 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[98]), "=r"(b_atom[99])
                    : "r"(b_addr_48)
                    : "memory");
                int b_row_49 = 96 + b_row_lane;
                int b_cb_49 = b_col + 64;
                unsigned int b_addr_49 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_49 * 128) + (unsigned int)(b_cb_49 ^ (b_row_49 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[100]), "=r"(b_atom[101])
                    : "r"(b_addr_49)
                    : "memory");
                int b_row_50 = 96 + b_row_lane;
                int b_cb_50 = b_col + 96;
                unsigned int b_addr_50 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_50 * 128) + (unsigned int)(b_cb_50 ^ (b_row_50 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[102]), "=r"(b_atom[103])
                    : "r"(b_addr_50)
                    : "memory");
                int b_row_51 = 104 + b_row_lane;
                int b_cb_51 = b_col;
                unsigned int b_addr_51 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_51 * 128) + (unsigned int)(b_cb_51 ^ (b_row_51 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[104]), "=r"(b_atom[105])
                    : "r"(b_addr_51)
                    : "memory");
                int b_row_52 = 104 + b_row_lane;
                int b_cb_52 = b_col + 32;
                unsigned int b_addr_52 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_52 * 128) + (unsigned int)(b_cb_52 ^ (b_row_52 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[106]), "=r"(b_atom[107])
                    : "r"(b_addr_52)
                    : "memory");
                int b_row_53 = 104 + b_row_lane;
                int b_cb_53 = b_col + 64;
                unsigned int b_addr_53 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_53 * 128) + (unsigned int)(b_cb_53 ^ (b_row_53 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[108]), "=r"(b_atom[109])
                    : "r"(b_addr_53)
                    : "memory");
                int b_row_54 = 104 + b_row_lane;
                int b_cb_54 = b_col + 96;
                unsigned int b_addr_54 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_54 * 128) + (unsigned int)(b_cb_54 ^ (b_row_54 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[110]), "=r"(b_atom[111])
                    : "r"(b_addr_54)
                    : "memory");
                int b_row_55 = 112 + b_row_lane;
                int b_cb_55 = b_col;
                unsigned int b_addr_55 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_55 * 128) + (unsigned int)(b_cb_55 ^ (b_row_55 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[112]), "=r"(b_atom[113])
                    : "r"(b_addr_55)
                    : "memory");
                int b_row_56 = 112 + b_row_lane;
                int b_cb_56 = b_col + 32;
                unsigned int b_addr_56 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_56 * 128) + (unsigned int)(b_cb_56 ^ (b_row_56 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[114]), "=r"(b_atom[115])
                    : "r"(b_addr_56)
                    : "memory");
                int b_row_57 = 112 + b_row_lane;
                int b_cb_57 = b_col + 64;
                unsigned int b_addr_57 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_57 * 128) + (unsigned int)(b_cb_57 ^ (b_row_57 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[116]), "=r"(b_atom[117])
                    : "r"(b_addr_57)
                    : "memory");
                int b_row_58 = 112 + b_row_lane;
                int b_cb_58 = b_col + 96;
                unsigned int b_addr_58 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_58 * 128) + (unsigned int)(b_cb_58 ^ (b_row_58 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[118]), "=r"(b_atom[119])
                    : "r"(b_addr_58)
                    : "memory");
                int b_row_59 = 120 + b_row_lane;
                int b_cb_59 = b_col;
                unsigned int b_addr_59 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_59 * 128) + (unsigned int)(b_cb_59 ^ (b_row_59 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[120]), "=r"(b_atom[121])
                    : "r"(b_addr_59)
                    : "memory");
                int b_row_60 = 120 + b_row_lane;
                int b_cb_60 = b_col + 32;
                unsigned int b_addr_60 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_60 * 128) + (unsigned int)(b_cb_60 ^ (b_row_60 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[122]), "=r"(b_atom[123])
                    : "r"(b_addr_60)
                    : "memory");
                int b_row_61 = 120 + b_row_lane;
                int b_cb_61 = b_col + 64;
                unsigned int b_addr_61 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_61 * 128) + (unsigned int)(b_cb_61 ^ (b_row_61 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[124]), "=r"(b_atom[125])
                    : "r"(b_addr_61)
                    : "memory");
                int b_row_62 = 120 + b_row_lane;
                int b_cb_62 = b_col + 96;
                unsigned int b_addr_62 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_62 * 128) + (unsigned int)(b_cb_62 ^ (b_row_62 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[126]), "=r"(b_atom[127])
                    : "r"(b_addr_62)
                    : "memory");
                int n_tok_m = 2;
                #pragma unroll 1
                for (int kv_idx = lo; kv_idx < hi; kv_idx += 2) {
                    mbarrier_wait(kv_full_g0_addr + (kv_stage_m) * 8, _phase_kv_full_g0);
                    int sc_idx = kv_stage_m * 256 + (unsigned int)(w_in_g * 16) + (unsigned int)quad;
                    float scale0 = smem_sc_g0[sc_idx];
                    float scale1 = smem_sc_g0[sc_idx + 8];
                    int kv_offset = tok * logits_stride + kv_idx * 64 + w_in_g * 16;
                    int a_cb = a_col;
                    unsigned int a_addr = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb ^ (a_row & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile[0]), "=r"(a_tile[1]), "=r"(a_tile[2]), "=r"(a_tile[3])
                        : "r"(a_addr)
                        : "memory");
                    int a_cb_0 = a_col + 32;
                    unsigned int a_addr_0 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_0 ^ (a_row & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile[4]), "=r"(a_tile[5]), "=r"(a_tile[6]), "=r"(a_tile[7])
                        : "r"(a_addr_0)
                        : "memory");
                    int a_cb_1 = a_col + 64;
                    unsigned int a_addr_1 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_1 ^ (a_row & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile[8]), "=r"(a_tile[9]), "=r"(a_tile[10]), "=r"(a_tile[11])
                        : "r"(a_addr_1)
                        : "memory");
                    int a_cb_2 = a_col + 96;
                    unsigned int a_addr_2 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_2 ^ (a_row & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile[12]), "=r"(a_tile[13]), "=r"(a_tile[14]), "=r"(a_tile[15])
                        : "r"(a_addr_2)
                        : "memory");
                    float partial0 = 0.0f;
                    float partial1 = 0.0f;
                    #pragma unroll
                    for (int nt = 0; nt < 8; nt++) {
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_tile[0]), "r"(a_tile[1]), "r"(a_tile[2]), "r"(a_tile[3]), "r"((b_atom + nt * 4 * 2)[0]), "r"((b_atom + nt * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[4]), "r"(a_tile[(4) + 1]), "r"(a_tile[(4) + 2]), "r"(a_tile[(4) + 3]), "r"((b_atom + (nt * 4 + 1) * 2)[0]), "r"((b_atom + (nt * 4 + 1) * 2)[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[8]), "r"(a_tile[(8) + 1]), "r"(a_tile[(8) + 2]), "r"(a_tile[(8) + 3]), "r"((b_atom + (nt * 4 + 2) * 2)[0]), "r"((b_atom + (nt * 4 + 2) * 2)[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[12]), "r"(a_tile[(12) + 1]), "r"(a_tile[(12) + 2]), "r"(a_tile[(12) + 3]), "r"((b_atom + (nt * 4 + 3) * 2)[0]), "r"((b_atom + (nt * 4 + 3) * 2)[1]));
                        int w_idx = q_stage_m * 256 + (unsigned int)(nt * 8) + (unsigned int)(tq * 2);
                        float w0 = smem_w[w_idx];
                        float w1 = smem_w[w_idx + 1];
                        float r0 = fmaxf(acc[0], 0.0f);
                        float r1 = fmaxf(acc[1], 0.0f);
                        float r2 = fmaxf(acc[2], 0.0f);
                        float r3 = fmaxf(acc[3], 0.0f);
                        partial0 += __fmaf_rn(r0, w0, r1 * w1);
                        partial1 += __fmaf_rn(r2, w0, r3 * w1);
                    }
                    float v0 = partial0 * scale0;
                    float v1 = partial1 * scale1;
                    float _shfl_xor_sync_0 = __shfl_xor_sync(0xFFFFFFFF, v0, 1);
                    v0 = v0 + _shfl_xor_sync_0;
                    float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, v1, 1);
                    v1 = v1 + _shfl_xor_sync_1;
                    float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, v0, 2);
                    v0 = v0 + _shfl_xor_sync_2;
                    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, v1, 2);
                    v1 = v1 + _shfl_xor_sync_3;
                    int out0 = kv_offset + quad;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0)), "f"(v0), "r"((unsigned int)(tq == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0 + 8))), "f"(v1), "r"((unsigned int)(tq == 0)));
                    float partial0_6 = 0.0f;
                    float partial1_7 = 0.0f;
                    #pragma unroll
                    for (int nt_1 = 0; nt_1 < 8; nt_1++) {
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_tile[0]), "r"(a_tile[1]), "r"(a_tile[2]), "r"(a_tile[3]), "r"((b_atom + (8 + nt_1) * 4 * 2)[0]), "r"((b_atom + (8 + nt_1) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[4]), "r"(a_tile[(4) + 1]), "r"(a_tile[(4) + 2]), "r"(a_tile[(4) + 3]), "r"((b_atom + ((8 + nt_1) * 4 + 1) * 2)[0]), "r"((b_atom + ((8 + nt_1) * 4 + 1) * 2)[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[8]), "r"(a_tile[(8) + 1]), "r"(a_tile[(8) + 2]), "r"(a_tile[(8) + 3]), "r"((b_atom + ((8 + nt_1) * 4 + 2) * 2)[0]), "r"((b_atom + ((8 + nt_1) * 4 + 2) * 2)[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[12]), "r"(a_tile[(12) + 1]), "r"(a_tile[(12) + 2]), "r"(a_tile[(12) + 3]), "r"((b_atom + ((8 + nt_1) * 4 + 3) * 2)[0]), "r"((b_atom + ((8 + nt_1) * 4 + 3) * 2)[1]));
                        int w_idx_1 = q_stage_m * 256 + 64 + (unsigned int)(nt_1 * 8) + (unsigned int)(tq * 2);
                        float w0_1 = smem_w[w_idx_1];
                        float w1_1 = smem_w[w_idx_1 + 1];
                        float r0_1 = fmaxf(acc[0], 0.0f);
                        float r1_1 = fmaxf(acc[1], 0.0f);
                        float r2_1 = fmaxf(acc[2], 0.0f);
                        float r3_1 = fmaxf(acc[3], 0.0f);
                        partial0_6 += __fmaf_rn(r0_1, w0_1, r1_1 * w1_1);
                        partial1_7 += __fmaf_rn(r2_1, w0_1, r3_1 * w1_1);
                    }
                    float v0_8 = partial0_6 * scale0;
                    float v1_9 = partial1_7 * scale1;
                    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, v0_8, 1);
                    v0_8 = v0_8 + _shfl_xor_sync_4;
                    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, v1_9, 1);
                    v1_9 = v1_9 + _shfl_xor_sync_5;
                    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, v0_8, 2);
                    v0_8 = v0_8 + _shfl_xor_sync_6;
                    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, v1_9, 2);
                    v1_9 = v1_9 + _shfl_xor_sync_7;
                    int out0_10 = kv_offset + logits_stride + quad;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_10)), "f"(v0_8), "r"((unsigned int)(tq == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_10 + 8))), "f"(v1_9), "r"((unsigned int)(tq == 0)));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_empty_g0_addr + (kv_stage_m) * 8);
                    kv_stage_m += 1;
                    if (kv_stage_m == 3) { kv_stage_m = 0; _phase_kv_full_g0 ^= 1; }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(q_empty_addr + (q_stage_m) * 8);
                q_stage_m += 1;
                if (q_stage_m == 2) { q_stage_m = 0; _phase_q_full ^= 1; }
            }
        }
    }
    // ---- Role: math_g1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // math_g1_main
            int w_in_g_1 = warp - 4;
            int lane_0_1 = lane;
            int quad_1 = lane_0_1 / 4;
            int tq_1 = lane_0_1 % 4;
            int a_row_1 = w_in_g_1 * 16 + (lane_0_1 & 7) + (lane_0_1 >> 3 & 1) * 8;
            int a_col_1 = (lane_0_1 >> 4) * 16;
            int b_row_lane_1 = lane_0_1 & 7;
            int b_col_1 = (lane_0_1 >> 3 & 1) * 16;
            unsigned int q_stage_m_1 = 0;
            unsigned int kv_stage_m_1 = 0;
            unsigned int a_frag_1[4];
            unsigned int b_frag_1[2];
            unsigned int a_tile_1[16];
            float acc_1[4];
            unsigned int b_atom_1[128];
            int sm_1 = bid;
            int start_q_1 = schedule_meta[sm_1 * 2];
            int start_kv_1 = schedule_meta[sm_1 * 2 + 1] * 2;
            int end_q_1 = schedule_meta[sm_1 * 2 + 2];
            int end_kv_1 = schedule_meta[sm_1 * 2 + 3] * 2;
            int q_stop_1 = ((end_kv_1 > 0) ? end_q_1 + 1 : end_q_1);
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_kv_full_g1 = 0;
            #pragma unroll 1
            for (int q_atom_1 = start_q_1; q_atom_1 < q_stop_1; q_atom_1++) {
                int q_idx_1 = q_atom_1 / 2;
                int atom_in_q_1 = q_atom_1 % 2;
                int tok_1 = q_idx_1 * 4 + atom_in_q_1 * 2;
                int ctx_last_1 = context_lens[q_idx_1 * 4 + 4 - 1];
                int num_kv_1 = (ctx_last_1 + 64 - 1) / 64;
                int lo_1 = ((q_atom_1 == start_q_1) ? start_kv_1 : 0);
                int hi_1 = ((q_atom_1 == end_q_1) ? end_kv_1 : num_kv_1);
                mbarrier_wait(q_full_addr + (q_stage_m_1) * 8, _phase_q_full_1);
                int b_row_63 = b_row_lane_1;
                int b_cb_63 = b_col_1;
                unsigned int b_addr_63 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_63 * 128) + (unsigned int)(b_cb_63 ^ (b_row_63 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[0]), "=r"(b_atom_1[1])
                    : "r"(b_addr_63)
                    : "memory");
                int b_row_63_1 = b_row_lane_1;
                int b_cb_63_1 = b_col_1 + 32;
                unsigned int b_addr_63_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_63_1 * 128) + (unsigned int)(b_cb_63_1 ^ (b_row_63_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[2]), "=r"(b_atom_1[3])
                    : "r"(b_addr_63_1)
                    : "memory");
                int b_row_64 = b_row_lane_1;
                int b_cb_64 = b_col_1 + 64;
                unsigned int b_addr_64 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_64 * 128) + (unsigned int)(b_cb_64 ^ (b_row_64 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[4]), "=r"(b_atom_1[5])
                    : "r"(b_addr_64)
                    : "memory");
                int b_row_65 = b_row_lane_1;
                int b_cb_65 = b_col_1 + 96;
                unsigned int b_addr_65 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_65 * 128) + (unsigned int)(b_cb_65 ^ (b_row_65 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[6]), "=r"(b_atom_1[7])
                    : "r"(b_addr_65)
                    : "memory");
                int b_row_66 = 8 + b_row_lane_1;
                int b_cb_66 = b_col_1;
                unsigned int b_addr_66 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_66 * 128) + (unsigned int)(b_cb_66 ^ (b_row_66 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[8]), "=r"(b_atom_1[9])
                    : "r"(b_addr_66)
                    : "memory");
                int b_row_67 = 8 + b_row_lane_1;
                int b_cb_67 = b_col_1 + 32;
                unsigned int b_addr_67 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_67 * 128) + (unsigned int)(b_cb_67 ^ (b_row_67 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[10]), "=r"(b_atom_1[11])
                    : "r"(b_addr_67)
                    : "memory");
                int b_row_68 = 8 + b_row_lane_1;
                int b_cb_68 = b_col_1 + 64;
                unsigned int b_addr_68 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_68 * 128) + (unsigned int)(b_cb_68 ^ (b_row_68 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[12]), "=r"(b_atom_1[13])
                    : "r"(b_addr_68)
                    : "memory");
                int b_row_69 = 8 + b_row_lane_1;
                int b_cb_69 = b_col_1 + 96;
                unsigned int b_addr_69 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_69 * 128) + (unsigned int)(b_cb_69 ^ (b_row_69 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[14]), "=r"(b_atom_1[15])
                    : "r"(b_addr_69)
                    : "memory");
                int b_row_70 = 16 + b_row_lane_1;
                int b_cb_70 = b_col_1;
                unsigned int b_addr_70 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_70 * 128) + (unsigned int)(b_cb_70 ^ (b_row_70 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[16]), "=r"(b_atom_1[17])
                    : "r"(b_addr_70)
                    : "memory");
                int b_row_71 = 16 + b_row_lane_1;
                int b_cb_71 = b_col_1 + 32;
                unsigned int b_addr_71 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_71 * 128) + (unsigned int)(b_cb_71 ^ (b_row_71 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[18]), "=r"(b_atom_1[19])
                    : "r"(b_addr_71)
                    : "memory");
                int b_row_72 = 16 + b_row_lane_1;
                int b_cb_72 = b_col_1 + 64;
                unsigned int b_addr_72 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_72 * 128) + (unsigned int)(b_cb_72 ^ (b_row_72 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[20]), "=r"(b_atom_1[21])
                    : "r"(b_addr_72)
                    : "memory");
                int b_row_73 = 16 + b_row_lane_1;
                int b_cb_73 = b_col_1 + 96;
                unsigned int b_addr_73 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_73 * 128) + (unsigned int)(b_cb_73 ^ (b_row_73 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[22]), "=r"(b_atom_1[23])
                    : "r"(b_addr_73)
                    : "memory");
                int b_row_74 = 24 + b_row_lane_1;
                int b_cb_74 = b_col_1;
                unsigned int b_addr_74 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_74 * 128) + (unsigned int)(b_cb_74 ^ (b_row_74 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[24]), "=r"(b_atom_1[25])
                    : "r"(b_addr_74)
                    : "memory");
                int b_row_75 = 24 + b_row_lane_1;
                int b_cb_75 = b_col_1 + 32;
                unsigned int b_addr_75 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_75 * 128) + (unsigned int)(b_cb_75 ^ (b_row_75 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[26]), "=r"(b_atom_1[27])
                    : "r"(b_addr_75)
                    : "memory");
                int b_row_76 = 24 + b_row_lane_1;
                int b_cb_76 = b_col_1 + 64;
                unsigned int b_addr_76 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_76 * 128) + (unsigned int)(b_cb_76 ^ (b_row_76 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[28]), "=r"(b_atom_1[29])
                    : "r"(b_addr_76)
                    : "memory");
                int b_row_77 = 24 + b_row_lane_1;
                int b_cb_77 = b_col_1 + 96;
                unsigned int b_addr_77 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_77 * 128) + (unsigned int)(b_cb_77 ^ (b_row_77 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[30]), "=r"(b_atom_1[31])
                    : "r"(b_addr_77)
                    : "memory");
                int b_row_78 = 32 + b_row_lane_1;
                int b_cb_78 = b_col_1;
                unsigned int b_addr_78 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_78 * 128) + (unsigned int)(b_cb_78 ^ (b_row_78 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[32]), "=r"(b_atom_1[33])
                    : "r"(b_addr_78)
                    : "memory");
                int b_row_79 = 32 + b_row_lane_1;
                int b_cb_79 = b_col_1 + 32;
                unsigned int b_addr_79 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_79 * 128) + (unsigned int)(b_cb_79 ^ (b_row_79 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[34]), "=r"(b_atom_1[35])
                    : "r"(b_addr_79)
                    : "memory");
                int b_row_80 = 32 + b_row_lane_1;
                int b_cb_80 = b_col_1 + 64;
                unsigned int b_addr_80 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_80 * 128) + (unsigned int)(b_cb_80 ^ (b_row_80 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[36]), "=r"(b_atom_1[37])
                    : "r"(b_addr_80)
                    : "memory");
                int b_row_81 = 32 + b_row_lane_1;
                int b_cb_81 = b_col_1 + 96;
                unsigned int b_addr_81 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_81 * 128) + (unsigned int)(b_cb_81 ^ (b_row_81 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[38]), "=r"(b_atom_1[39])
                    : "r"(b_addr_81)
                    : "memory");
                int b_row_82 = 40 + b_row_lane_1;
                int b_cb_82 = b_col_1;
                unsigned int b_addr_82 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_82 * 128) + (unsigned int)(b_cb_82 ^ (b_row_82 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[40]), "=r"(b_atom_1[41])
                    : "r"(b_addr_82)
                    : "memory");
                int b_row_83 = 40 + b_row_lane_1;
                int b_cb_83 = b_col_1 + 32;
                unsigned int b_addr_83 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_83 * 128) + (unsigned int)(b_cb_83 ^ (b_row_83 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[42]), "=r"(b_atom_1[43])
                    : "r"(b_addr_83)
                    : "memory");
                int b_row_84 = 40 + b_row_lane_1;
                int b_cb_84 = b_col_1 + 64;
                unsigned int b_addr_84 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_84 * 128) + (unsigned int)(b_cb_84 ^ (b_row_84 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[44]), "=r"(b_atom_1[45])
                    : "r"(b_addr_84)
                    : "memory");
                int b_row_85 = 40 + b_row_lane_1;
                int b_cb_85 = b_col_1 + 96;
                unsigned int b_addr_85 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_85 * 128) + (unsigned int)(b_cb_85 ^ (b_row_85 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[46]), "=r"(b_atom_1[47])
                    : "r"(b_addr_85)
                    : "memory");
                int b_row_86 = 48 + b_row_lane_1;
                int b_cb_86 = b_col_1;
                unsigned int b_addr_86 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_86 * 128) + (unsigned int)(b_cb_86 ^ (b_row_86 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[48]), "=r"(b_atom_1[49])
                    : "r"(b_addr_86)
                    : "memory");
                int b_row_87 = 48 + b_row_lane_1;
                int b_cb_87 = b_col_1 + 32;
                unsigned int b_addr_87 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_87 * 128) + (unsigned int)(b_cb_87 ^ (b_row_87 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[50]), "=r"(b_atom_1[51])
                    : "r"(b_addr_87)
                    : "memory");
                int b_row_88 = 48 + b_row_lane_1;
                int b_cb_88 = b_col_1 + 64;
                unsigned int b_addr_88 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_88 * 128) + (unsigned int)(b_cb_88 ^ (b_row_88 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[52]), "=r"(b_atom_1[53])
                    : "r"(b_addr_88)
                    : "memory");
                int b_row_89 = 48 + b_row_lane_1;
                int b_cb_89 = b_col_1 + 96;
                unsigned int b_addr_89 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_89 * 128) + (unsigned int)(b_cb_89 ^ (b_row_89 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[54]), "=r"(b_atom_1[55])
                    : "r"(b_addr_89)
                    : "memory");
                int b_row_90 = 56 + b_row_lane_1;
                int b_cb_90 = b_col_1;
                unsigned int b_addr_90 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_90 * 128) + (unsigned int)(b_cb_90 ^ (b_row_90 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[56]), "=r"(b_atom_1[57])
                    : "r"(b_addr_90)
                    : "memory");
                int b_row_91 = 56 + b_row_lane_1;
                int b_cb_91 = b_col_1 + 32;
                unsigned int b_addr_91 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_91 * 128) + (unsigned int)(b_cb_91 ^ (b_row_91 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[58]), "=r"(b_atom_1[59])
                    : "r"(b_addr_91)
                    : "memory");
                int b_row_92 = 56 + b_row_lane_1;
                int b_cb_92 = b_col_1 + 64;
                unsigned int b_addr_92 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_92 * 128) + (unsigned int)(b_cb_92 ^ (b_row_92 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[60]), "=r"(b_atom_1[61])
                    : "r"(b_addr_92)
                    : "memory");
                int b_row_93 = 56 + b_row_lane_1;
                int b_cb_93 = b_col_1 + 96;
                unsigned int b_addr_93 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_93 * 128) + (unsigned int)(b_cb_93 ^ (b_row_93 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[62]), "=r"(b_atom_1[63])
                    : "r"(b_addr_93)
                    : "memory");
                int b_row_94 = 64 + b_row_lane_1;
                int b_cb_94 = b_col_1;
                unsigned int b_addr_94 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_94 * 128) + (unsigned int)(b_cb_94 ^ (b_row_94 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[64]), "=r"(b_atom_1[65])
                    : "r"(b_addr_94)
                    : "memory");
                int b_row_95 = 64 + b_row_lane_1;
                int b_cb_95 = b_col_1 + 32;
                unsigned int b_addr_95 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_95 * 128) + (unsigned int)(b_cb_95 ^ (b_row_95 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[66]), "=r"(b_atom_1[67])
                    : "r"(b_addr_95)
                    : "memory");
                int b_row_96 = 64 + b_row_lane_1;
                int b_cb_96 = b_col_1 + 64;
                unsigned int b_addr_96 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_96 * 128) + (unsigned int)(b_cb_96 ^ (b_row_96 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[68]), "=r"(b_atom_1[69])
                    : "r"(b_addr_96)
                    : "memory");
                int b_row_97 = 64 + b_row_lane_1;
                int b_cb_97 = b_col_1 + 96;
                unsigned int b_addr_97 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_97 * 128) + (unsigned int)(b_cb_97 ^ (b_row_97 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[70]), "=r"(b_atom_1[71])
                    : "r"(b_addr_97)
                    : "memory");
                int b_row_98 = 72 + b_row_lane_1;
                int b_cb_98 = b_col_1;
                unsigned int b_addr_98 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_98 * 128) + (unsigned int)(b_cb_98 ^ (b_row_98 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[72]), "=r"(b_atom_1[73])
                    : "r"(b_addr_98)
                    : "memory");
                int b_row_99 = 72 + b_row_lane_1;
                int b_cb_99 = b_col_1 + 32;
                unsigned int b_addr_99 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_99 * 128) + (unsigned int)(b_cb_99 ^ (b_row_99 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[74]), "=r"(b_atom_1[75])
                    : "r"(b_addr_99)
                    : "memory");
                int b_row_100 = 72 + b_row_lane_1;
                int b_cb_100 = b_col_1 + 64;
                unsigned int b_addr_100 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_100 * 128) + (unsigned int)(b_cb_100 ^ (b_row_100 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[76]), "=r"(b_atom_1[77])
                    : "r"(b_addr_100)
                    : "memory");
                int b_row_101 = 72 + b_row_lane_1;
                int b_cb_101 = b_col_1 + 96;
                unsigned int b_addr_101 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_101 * 128) + (unsigned int)(b_cb_101 ^ (b_row_101 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[78]), "=r"(b_atom_1[79])
                    : "r"(b_addr_101)
                    : "memory");
                int b_row_102 = 80 + b_row_lane_1;
                int b_cb_102 = b_col_1;
                unsigned int b_addr_102 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_102 * 128) + (unsigned int)(b_cb_102 ^ (b_row_102 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[80]), "=r"(b_atom_1[81])
                    : "r"(b_addr_102)
                    : "memory");
                int b_row_103 = 80 + b_row_lane_1;
                int b_cb_103 = b_col_1 + 32;
                unsigned int b_addr_103 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_103 * 128) + (unsigned int)(b_cb_103 ^ (b_row_103 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[82]), "=r"(b_atom_1[83])
                    : "r"(b_addr_103)
                    : "memory");
                int b_row_104 = 80 + b_row_lane_1;
                int b_cb_104 = b_col_1 + 64;
                unsigned int b_addr_104 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_104 * 128) + (unsigned int)(b_cb_104 ^ (b_row_104 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[84]), "=r"(b_atom_1[85])
                    : "r"(b_addr_104)
                    : "memory");
                int b_row_105 = 80 + b_row_lane_1;
                int b_cb_105 = b_col_1 + 96;
                unsigned int b_addr_105 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_105 * 128) + (unsigned int)(b_cb_105 ^ (b_row_105 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[86]), "=r"(b_atom_1[87])
                    : "r"(b_addr_105)
                    : "memory");
                int b_row_106 = 88 + b_row_lane_1;
                int b_cb_106 = b_col_1;
                unsigned int b_addr_106 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_106 * 128) + (unsigned int)(b_cb_106 ^ (b_row_106 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[88]), "=r"(b_atom_1[89])
                    : "r"(b_addr_106)
                    : "memory");
                int b_row_107 = 88 + b_row_lane_1;
                int b_cb_107 = b_col_1 + 32;
                unsigned int b_addr_107 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_107 * 128) + (unsigned int)(b_cb_107 ^ (b_row_107 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[90]), "=r"(b_atom_1[91])
                    : "r"(b_addr_107)
                    : "memory");
                int b_row_108 = 88 + b_row_lane_1;
                int b_cb_108 = b_col_1 + 64;
                unsigned int b_addr_108 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_108 * 128) + (unsigned int)(b_cb_108 ^ (b_row_108 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[92]), "=r"(b_atom_1[93])
                    : "r"(b_addr_108)
                    : "memory");
                int b_row_109 = 88 + b_row_lane_1;
                int b_cb_109 = b_col_1 + 96;
                unsigned int b_addr_109 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_109 * 128) + (unsigned int)(b_cb_109 ^ (b_row_109 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[94]), "=r"(b_atom_1[95])
                    : "r"(b_addr_109)
                    : "memory");
                int b_row_110 = 96 + b_row_lane_1;
                int b_cb_110 = b_col_1;
                unsigned int b_addr_110 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_110 * 128) + (unsigned int)(b_cb_110 ^ (b_row_110 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[96]), "=r"(b_atom_1[97])
                    : "r"(b_addr_110)
                    : "memory");
                int b_row_111 = 96 + b_row_lane_1;
                int b_cb_111 = b_col_1 + 32;
                unsigned int b_addr_111 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_111 * 128) + (unsigned int)(b_cb_111 ^ (b_row_111 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[98]), "=r"(b_atom_1[99])
                    : "r"(b_addr_111)
                    : "memory");
                int b_row_112 = 96 + b_row_lane_1;
                int b_cb_112 = b_col_1 + 64;
                unsigned int b_addr_112 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_112 * 128) + (unsigned int)(b_cb_112 ^ (b_row_112 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[100]), "=r"(b_atom_1[101])
                    : "r"(b_addr_112)
                    : "memory");
                int b_row_113 = 96 + b_row_lane_1;
                int b_cb_113 = b_col_1 + 96;
                unsigned int b_addr_113 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_113 * 128) + (unsigned int)(b_cb_113 ^ (b_row_113 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[102]), "=r"(b_atom_1[103])
                    : "r"(b_addr_113)
                    : "memory");
                int b_row_114 = 104 + b_row_lane_1;
                int b_cb_114 = b_col_1;
                unsigned int b_addr_114 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_114 * 128) + (unsigned int)(b_cb_114 ^ (b_row_114 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[104]), "=r"(b_atom_1[105])
                    : "r"(b_addr_114)
                    : "memory");
                int b_row_115 = 104 + b_row_lane_1;
                int b_cb_115 = b_col_1 + 32;
                unsigned int b_addr_115 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_115 * 128) + (unsigned int)(b_cb_115 ^ (b_row_115 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[106]), "=r"(b_atom_1[107])
                    : "r"(b_addr_115)
                    : "memory");
                int b_row_116 = 104 + b_row_lane_1;
                int b_cb_116 = b_col_1 + 64;
                unsigned int b_addr_116 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_116 * 128) + (unsigned int)(b_cb_116 ^ (b_row_116 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[108]), "=r"(b_atom_1[109])
                    : "r"(b_addr_116)
                    : "memory");
                int b_row_117 = 104 + b_row_lane_1;
                int b_cb_117 = b_col_1 + 96;
                unsigned int b_addr_117 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_117 * 128) + (unsigned int)(b_cb_117 ^ (b_row_117 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[110]), "=r"(b_atom_1[111])
                    : "r"(b_addr_117)
                    : "memory");
                int b_row_118 = 112 + b_row_lane_1;
                int b_cb_118 = b_col_1;
                unsigned int b_addr_118 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_118 * 128) + (unsigned int)(b_cb_118 ^ (b_row_118 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[112]), "=r"(b_atom_1[113])
                    : "r"(b_addr_118)
                    : "memory");
                int b_row_119 = 112 + b_row_lane_1;
                int b_cb_119 = b_col_1 + 32;
                unsigned int b_addr_119 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_119 * 128) + (unsigned int)(b_cb_119 ^ (b_row_119 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[114]), "=r"(b_atom_1[115])
                    : "r"(b_addr_119)
                    : "memory");
                int b_row_120 = 112 + b_row_lane_1;
                int b_cb_120 = b_col_1 + 64;
                unsigned int b_addr_120 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_120 * 128) + (unsigned int)(b_cb_120 ^ (b_row_120 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[116]), "=r"(b_atom_1[117])
                    : "r"(b_addr_120)
                    : "memory");
                int b_row_121 = 112 + b_row_lane_1;
                int b_cb_121 = b_col_1 + 96;
                unsigned int b_addr_121 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_121 * 128) + (unsigned int)(b_cb_121 ^ (b_row_121 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[118]), "=r"(b_atom_1[119])
                    : "r"(b_addr_121)
                    : "memory");
                int b_row_122 = 120 + b_row_lane_1;
                int b_cb_122 = b_col_1;
                unsigned int b_addr_122 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_122 * 128) + (unsigned int)(b_cb_122 ^ (b_row_122 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[120]), "=r"(b_atom_1[121])
                    : "r"(b_addr_122)
                    : "memory");
                int b_row_123 = 120 + b_row_lane_1;
                int b_cb_123 = b_col_1 + 32;
                unsigned int b_addr_123 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_123 * 128) + (unsigned int)(b_cb_123 ^ (b_row_123 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[122]), "=r"(b_atom_1[123])
                    : "r"(b_addr_123)
                    : "memory");
                int b_row_124 = 120 + b_row_lane_1;
                int b_cb_124 = b_col_1 + 64;
                unsigned int b_addr_124 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_124 * 128) + (unsigned int)(b_cb_124 ^ (b_row_124 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[124]), "=r"(b_atom_1[125])
                    : "r"(b_addr_124)
                    : "memory");
                int b_row_125 = 120 + b_row_lane_1;
                int b_cb_125 = b_col_1 + 96;
                unsigned int b_addr_125 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_125 * 128) + (unsigned int)(b_cb_125 ^ (b_row_125 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[126]), "=r"(b_atom_1[127])
                    : "r"(b_addr_125)
                    : "memory");
                int n_tok_m_1 = 2;
                #pragma unroll 1
                for (int kv_idx_1 = lo_1; kv_idx_1 < hi_1; kv_idx_1 += 2) {
                    mbarrier_wait(kv_full_g1_addr + (kv_stage_m_1) * 8, _phase_kv_full_g1);
                    int sc_idx_1 = kv_stage_m_1 * 256 + (unsigned int)(w_in_g_1 * 16) + (unsigned int)quad_1;
                    float scale0_1 = smem_sc_g1[sc_idx_1];
                    float scale1_1 = smem_sc_g1[sc_idx_1 + 8];
                    int kv_offset_1 = tok_1 * logits_stride + (kv_idx_1 + 1) * 64 + w_in_g_1 * 16;
                    int a_cb_3 = a_col_1;
                    unsigned int a_addr_3 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_3 ^ (a_row_1 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile_1[0]), "=r"(a_tile_1[1]), "=r"(a_tile_1[2]), "=r"(a_tile_1[3])
                        : "r"(a_addr_3)
                        : "memory");
                    int a_cb_3_1 = a_col_1 + 32;
                    unsigned int a_addr_3_1 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_3_1 ^ (a_row_1 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile_1[4]), "=r"(a_tile_1[5]), "=r"(a_tile_1[6]), "=r"(a_tile_1[7])
                        : "r"(a_addr_3_1)
                        : "memory");
                    int a_cb_4 = a_col_1 + 64;
                    unsigned int a_addr_4 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_4 ^ (a_row_1 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile_1[8]), "=r"(a_tile_1[9]), "=r"(a_tile_1[10]), "=r"(a_tile_1[11])
                        : "r"(a_addr_4)
                        : "memory");
                    int a_cb_5 = a_col_1 + 96;
                    unsigned int a_addr_5 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_5 ^ (a_row_1 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile_1[12]), "=r"(a_tile_1[13]), "=r"(a_tile_1[14]), "=r"(a_tile_1[15])
                        : "r"(a_addr_5)
                        : "memory");
                    float partial0_1 = 0.0f;
                    float partial1_1 = 0.0f;
                    #pragma unroll
                    for (int nt_2 = 0; nt_2 < 8; nt_2++) {
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_tile_1[0]), "r"(a_tile_1[1]), "r"(a_tile_1[2]), "r"(a_tile_1[3]), "r"((b_atom_1 + nt_2 * 4 * 2)[0]), "r"((b_atom_1 + nt_2 * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[4]), "r"(a_tile_1[(4) + 1]), "r"(a_tile_1[(4) + 2]), "r"(a_tile_1[(4) + 3]), "r"((b_atom_1 + (nt_2 * 4 + 1) * 2)[0]), "r"((b_atom_1 + (nt_2 * 4 + 1) * 2)[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[8]), "r"(a_tile_1[(8) + 1]), "r"(a_tile_1[(8) + 2]), "r"(a_tile_1[(8) + 3]), "r"((b_atom_1 + (nt_2 * 4 + 2) * 2)[0]), "r"((b_atom_1 + (nt_2 * 4 + 2) * 2)[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[12]), "r"(a_tile_1[(12) + 1]), "r"(a_tile_1[(12) + 2]), "r"(a_tile_1[(12) + 3]), "r"((b_atom_1 + (nt_2 * 4 + 3) * 2)[0]), "r"((b_atom_1 + (nt_2 * 4 + 3) * 2)[1]));
                        int w_idx_2 = q_stage_m_1 * 256 + (unsigned int)(nt_2 * 8) + (unsigned int)(tq_1 * 2);
                        float w0_2 = smem_w[w_idx_2];
                        float w1_2 = smem_w[w_idx_2 + 1];
                        float r0_2 = fmaxf(acc_1[0], 0.0f);
                        float r1_2 = fmaxf(acc_1[1], 0.0f);
                        float r2_2 = fmaxf(acc_1[2], 0.0f);
                        float r3_2 = fmaxf(acc_1[3], 0.0f);
                        partial0_1 += __fmaf_rn(r0_2, w0_2, r1_2 * w1_2);
                        partial1_1 += __fmaf_rn(r2_2, w0_2, r3_2 * w1_2);
                    }
                    float v0_1 = partial0_1 * scale0_1;
                    float v1_1 = partial1_1 * scale1_1;
                    float _shfl_xor_sync_0_1 = __shfl_xor_sync(0xFFFFFFFF, v0_1, 1);
                    v0_1 = v0_1 + _shfl_xor_sync_0_1;
                    float _shfl_xor_sync_1_1 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 1);
                    v1_1 = v1_1 + _shfl_xor_sync_1_1;
                    float _shfl_xor_sync_2_1 = __shfl_xor_sync(0xFFFFFFFF, v0_1, 2);
                    v0_1 = v0_1 + _shfl_xor_sync_2_1;
                    float _shfl_xor_sync_3_1 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 2);
                    v1_1 = v1_1 + _shfl_xor_sync_3_1;
                    int out0_1 = kv_offset_1 + quad_1;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_1)), "f"(v0_1), "r"((unsigned int)(tq_1 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_1 + 8))), "f"(v1_1), "r"((unsigned int)(tq_1 == 0)));
                    float partial0_6_1 = 0.0f;
                    float partial1_7_1 = 0.0f;
                    #pragma unroll
                    for (int nt_3 = 0; nt_3 < 8; nt_3++) {
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_tile_1[0]), "r"(a_tile_1[1]), "r"(a_tile_1[2]), "r"(a_tile_1[3]), "r"((b_atom_1 + (8 + nt_3) * 4 * 2)[0]), "r"((b_atom_1 + (8 + nt_3) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[4]), "r"(a_tile_1[(4) + 1]), "r"(a_tile_1[(4) + 2]), "r"(a_tile_1[(4) + 3]), "r"((b_atom_1 + ((8 + nt_3) * 4 + 1) * 2)[0]), "r"((b_atom_1 + ((8 + nt_3) * 4 + 1) * 2)[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[8]), "r"(a_tile_1[(8) + 1]), "r"(a_tile_1[(8) + 2]), "r"(a_tile_1[(8) + 3]), "r"((b_atom_1 + ((8 + nt_3) * 4 + 2) * 2)[0]), "r"((b_atom_1 + ((8 + nt_3) * 4 + 2) * 2)[1]));
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[12]), "r"(a_tile_1[(12) + 1]), "r"(a_tile_1[(12) + 2]), "r"(a_tile_1[(12) + 3]), "r"((b_atom_1 + ((8 + nt_3) * 4 + 3) * 2)[0]), "r"((b_atom_1 + ((8 + nt_3) * 4 + 3) * 2)[1]));
                        int w_idx_3 = q_stage_m_1 * 256 + 64 + (unsigned int)(nt_3 * 8) + (unsigned int)(tq_1 * 2);
                        float w0_3 = smem_w[w_idx_3];
                        float w1_3 = smem_w[w_idx_3 + 1];
                        float r0_3 = fmaxf(acc_1[0], 0.0f);
                        float r1_3 = fmaxf(acc_1[1], 0.0f);
                        float r2_3 = fmaxf(acc_1[2], 0.0f);
                        float r3_3 = fmaxf(acc_1[3], 0.0f);
                        partial0_6_1 += __fmaf_rn(r0_3, w0_3, r1_3 * w1_3);
                        partial1_7_1 += __fmaf_rn(r2_3, w0_3, r3_3 * w1_3);
                    }
                    float v0_8_1 = partial0_6_1 * scale0_1;
                    float v1_9_1 = partial1_7_1 * scale1_1;
                    float _shfl_xor_sync_4_1 = __shfl_xor_sync(0xFFFFFFFF, v0_8_1, 1);
                    v0_8_1 = v0_8_1 + _shfl_xor_sync_4_1;
                    float _shfl_xor_sync_5_1 = __shfl_xor_sync(0xFFFFFFFF, v1_9_1, 1);
                    v1_9_1 = v1_9_1 + _shfl_xor_sync_5_1;
                    float _shfl_xor_sync_6_1 = __shfl_xor_sync(0xFFFFFFFF, v0_8_1, 2);
                    v0_8_1 = v0_8_1 + _shfl_xor_sync_6_1;
                    float _shfl_xor_sync_7_1 = __shfl_xor_sync(0xFFFFFFFF, v1_9_1, 2);
                    v1_9_1 = v1_9_1 + _shfl_xor_sync_7_1;
                    int out0_10_1 = kv_offset_1 + logits_stride + quad_1;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_10_1)), "f"(v0_8_1), "r"((unsigned int)(tq_1 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_10_1 + 8))), "f"(v1_9_1), "r"((unsigned int)(tq_1 == 0)));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_empty_g1_addr + (kv_stage_m_1) * 8);
                    kv_stage_m_1 += 1;
                    if (kv_stage_m_1 == 3) { kv_stage_m_1 = 0; _phase_kv_full_g1 ^= 1; }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(q_empty_addr + (q_stage_m_1) * 8);
                q_stage_m_1 += 1;
                if (q_stage_m_1 == 2) { q_stage_m_1 = 0; _phase_q_full_1 ^= 1; }
            }
        }
    }
    // ---- Role: tma_g0 ----
    if (warp == 8) {
        { // tma_g0_main
            unsigned int q_stage = 0;
            unsigned int q_phase = 1;
            unsigned int kv_stage = 0;
            unsigned int kv_phase = 1;
            int sm_2 = bid;
            int start_q_2 = schedule_meta[sm_2 * 2];
            int start_kv_2 = schedule_meta[sm_2 * 2 + 1] * 2;
            int end_q_2 = schedule_meta[sm_2 * 2 + 2];
            int end_kv_2 = schedule_meta[sm_2 * 2 + 3] * 2;
            int q_stop_2 = ((end_kv_2 > 0) ? end_q_2 + 1 : end_q_2);
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_kv_empty_g0 = 1;
            #pragma unroll 1
            for (int q_atom_2 = start_q_2; q_atom_2 < q_stop_2; q_atom_2++) {
                int q_idx_2 = q_atom_2 / 2;
                int atom_in_q_2 = q_atom_2 % 2;
                int tok_2 = q_idx_2 * 4 + atom_in_q_2 * 2;
                int ctx_last_2 = context_lens[q_idx_2 * 4 + 4 - 1];
                int kv_lo = ((q_atom_2 == start_q_2) ? start_kv_2 : 0);
                int bt_lane0 = lane;
                int bt_col0 = kv_lo + bt_lane0 * 2;
                int bt_raw = 0;
                if (bt_col0 < block_table_stride) {
                    bt_raw = block_table[q_idx_2 * block_table_stride + bt_col0];
                }
                mbarrier_wait(q_empty_addr + (q_stage) * 8, _phase_q_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + (q_stage) * 8, 16896);
                    tma_2d_gmem2smem(smem_q_addr + q_stage * 16384, (&Q), 0, tok_2 * 64, q_full_addr + (q_stage) * 8);
                    tma_2d_gmem2smem(smem_w_addr + q_stage * 1024, (&Weights), 0, tok_2, q_full_addr + (q_stage) * 8);
                }
                q_stage += 1;
                if (q_stage == 2) { q_stage = 0; _phase_q_empty ^= 1; }
                int num_kv_2 = (ctx_last_2 + 64 - 1) / 64;
                int kv_hi = ((q_atom_2 == end_q_2) ? end_kv_2 : num_kv_2);
                int bt_row = q_idx_2 * block_table_stride;
                int bt_lane = lane;
                int bt_ptr = 0;
                int bt_cache = bt_raw;
                if (num_kv_2 <= kv_lo + bt_lane * 2) {
                    bt_cache = 0;
                }
                #pragma unroll 1
                for (int kv_idx_2 = kv_lo; kv_idx_2 < kv_hi; kv_idx_2 += 2) {
                    int tile = kv_idx_2;
                    if (bt_ptr == 32) {
                        bt_ptr = 0;
                        int bt_tile = tile + bt_lane * 2;
                        bt_cache = 0;
                        if (bt_tile < num_kv_2) {
                            bt_cache = block_table[bt_row + bt_tile];
                        }
                    }
                    int kv_block = __shfl_sync(0xFFFFFFFF, bt_cache, bt_ptr);
                    bt_ptr += 1;
                    int page_off = 0;
                    mbarrier_wait(kv_empty_g0_addr + (kv_stage) * 8, _phase_kv_empty_g0);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_g0_addr + (kv_stage) * 8, 8448);
                        tma_3d_gmem2smem(smem_kv_g0_addr + kv_stage * 8192, (&KV), 0, page_off, kv_block, kv_full_g0_addr + (kv_stage) * 8);
                        tma_2d_gmem2smem(smem_sc_g0_addr + kv_stage * 1024, (&KV_scales), 2048 + page_off, kv_block, kv_full_g0_addr + (kv_stage) * 8);
                    }
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; _phase_kv_empty_g0 ^= 1; }
                }
            }
        }
    }
    // ---- Role: tma_g1 ----
    if (warp == 9) {
        { // tma_g1_main
            unsigned int q_stage_1 = 0;
            unsigned int q_phase_1 = 1;
            unsigned int kv_stage_1 = 0;
            unsigned int kv_phase_1 = 1;
            int sm_3 = bid;
            int start_q_3 = schedule_meta[sm_3 * 2];
            int start_kv_3 = schedule_meta[sm_3 * 2 + 1] * 2;
            int end_q_3 = schedule_meta[sm_3 * 2 + 2];
            int end_kv_3 = schedule_meta[sm_3 * 2 + 3] * 2;
            int q_stop_3 = ((end_kv_3 > 0) ? end_q_3 + 1 : end_q_3);
            unsigned int _phase_kv_empty_g1 = 1;
            #pragma unroll 1
            for (int q_atom_3 = start_q_3; q_atom_3 < q_stop_3; q_atom_3++) {
                int q_idx_3 = q_atom_3 / 2;
                int atom_in_q_3 = q_atom_3 % 2;
                int tok_3 = q_idx_3 * 4 + atom_in_q_3 * 2;
                int ctx_last_3 = context_lens[q_idx_3 * 4 + 4 - 1];
                int kv_lo_1 = ((q_atom_3 == start_q_3) ? start_kv_3 : 0);
                int bt_lane0_1 = lane;
                int bt_col0_1 = kv_lo_1 + 1 + bt_lane0_1 * 2;
                int bt_raw_1 = 0;
                if (bt_col0_1 < block_table_stride) {
                    bt_raw_1 = block_table[q_idx_3 * block_table_stride + bt_col0_1];
                }
                int num_kv_3 = (ctx_last_3 + 64 - 1) / 64;
                int kv_hi_1 = ((q_atom_3 == end_q_3) ? end_kv_3 : num_kv_3);
                int bt_row_1 = q_idx_3 * block_table_stride;
                int bt_lane_1 = lane;
                int bt_ptr_1 = 0;
                int bt_cache_1 = bt_raw_1;
                if (num_kv_3 <= kv_lo_1 + 1 + bt_lane_1 * 2) {
                    bt_cache_1 = 0;
                }
                #pragma unroll 1
                for (int kv_idx_3 = kv_lo_1; kv_idx_3 < kv_hi_1; kv_idx_3 += 2) {
                    int tile_1 = kv_idx_3 + 1;
                    if (bt_ptr_1 == 32) {
                        bt_ptr_1 = 0;
                        int bt_tile_1 = tile_1 + bt_lane_1 * 2;
                        bt_cache_1 = 0;
                        if (bt_tile_1 < num_kv_3) {
                            bt_cache_1 = block_table[bt_row_1 + bt_tile_1];
                        }
                    }
                    int kv_block_1 = __shfl_sync(0xFFFFFFFF, bt_cache_1, bt_ptr_1);
                    bt_ptr_1 += 1;
                    int page_off_1 = 0;
                    mbarrier_wait(kv_empty_g1_addr + (kv_stage_1) * 8, _phase_kv_empty_g1);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_g1_addr + (kv_stage_1) * 8, 8448);
                        tma_3d_gmem2smem(smem_kv_g1_addr + kv_stage_1 * 8192, (&KV), 0, page_off_1, kv_block_1, kv_full_g1_addr + (kv_stage_1) * 8);
                        tma_2d_gmem2smem(smem_sc_g1_addr + kv_stage_1 * 1024, (&KV_scales), 2048 + page_off_1, kv_block_1, kv_full_g1_addr + (kv_stage_1) * 8);
                    }
                    kv_stage_1 += 1;
                    if (kv_stage_1 == 3) { kv_stage_1 = 0; _phase_kv_empty_g1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp >= 10 && warp <= 11) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
