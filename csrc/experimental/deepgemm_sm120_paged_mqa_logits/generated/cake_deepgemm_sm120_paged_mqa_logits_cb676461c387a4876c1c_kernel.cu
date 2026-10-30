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
#define NUM_IT_PIPE_STAGES 8
#define NUM_CG_PIPE_STAGES 4
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 12288
#define SMEM_SMEM_Q_STRIDE 12288
#define SMEM_SMEM_W_OFF 25600
#define SMEM_SMEM_W_STAGE_BYTES 384
#define SMEM_SMEM_W_STRIDE 1024
#define SMEM_SMEM_KV_G0_OFF 27648
#define SMEM_SMEM_KV_G0_STAGE_BYTES 8192
#define SMEM_SMEM_KV_G0_STRIDE 8192
#define SMEM_SMEM_KV_G1_OFF 52224
#define SMEM_SMEM_KV_G1_STAGE_BYTES 8192
#define SMEM_SMEM_KV_G1_STRIDE 8192
#define SMEM_SMEM_SC_G0_OFF 76800
#define SMEM_SMEM_SC_G0_STAGE_BYTES 256
#define SMEM_SMEM_SC_G0_STRIDE 1024
#define SMEM_SMEM_SC_G1_OFF 79872
#define SMEM_SMEM_SC_G1_STAGE_BYTES 256
#define SMEM_SMEM_SC_G1_STRIDE 1024
#define SMEM_SMEM_IT_OFF 82944
#define SMEM_SMEM_IT_STAGE_BYTES 256
#define SMEM_SMEM_IT_STRIDE 256
#define SMEM_TOTAL 83200
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
kernel_cake_deepgemm_sm120_paged_mqa_logits_cb676461c387a4876c1c(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KV_scales, const __grid_constant__ CUtensorMap Weights, float* __restrict__ Logits, int* __restrict__ context_lens, int* __restrict__ block_table, int* __restrict__ schedule_meta, int logits_stride, int block_table_stride)
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
    #define it_full_addr (mbar_base + 128)
    #define it_empty_addr (mbar_base + 192)
    #define claim_go_addr (mbar_base + 256)
    #define claim_free_addr (mbar_base + 288)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    float* smem_w = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_w_addr = smem + 25600;
    uint8_t* smem_kv_g0 = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int smem_kv_g0_addr = smem + 27648;
    uint8_t* smem_kv_g1 = reinterpret_cast<uint8_t*>(smem_raw + 52224);
    const int smem_kv_g1_addr = smem + 52224;
    float* smem_sc_g0 = reinterpret_cast<float*>(smem_raw + 76800);
    const int smem_sc_g0_addr = smem + 76800;
    float* smem_sc_g1 = reinterpret_cast<float*>(smem_raw + 79872);
    const int smem_sc_g1_addr = smem + 79872;
    int* smem_it = reinterpret_cast<int*>(smem_raw + 82944);
    const int smem_it_addr = smem + 82944;
    if (tid == 0) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(schedule_meta + (bid * 2)))); }
    if (tid == 0) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(schedule_meta + (bid * 2 + 3)))); }

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 40 barriers)
    // Mbarriers at smem_raw[0..320)

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
            // --- pipeline 'it_pipe' ---
            // it_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // it_empty: 8 barriers, init_count=321
            mbarrier_init(smem + 192, 321);
            mbarrier_init(smem + 200, 321);
            mbarrier_init(smem + 208, 321);
            mbarrier_init(smem + 216, 321);
            mbarrier_init(smem + 224, 321);
            mbarrier_init(smem + 232, 321);
            mbarrier_init(smem + 240, 321);
            mbarrier_init(smem + 248, 321);
            // --- pipeline 'cg_pipe' ---
            // claim_go: 4 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            // claim_free: 4 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
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
            unsigned int b_atom[96];
            unsigned int mi_stage = 0;
            unsigned int _phase_it_full = 0;
            mbarrier_wait(it_full_addr + (mi_stage) * 8, _phase_it_full);
            int e_start_q = smem_it[mi_stage * 8];
            int e_start_kv = smem_it[mi_stage * 8 + 1];
            int e_end_q = smem_it[mi_stage * 8 + 2];
            int e_end_kv = smem_it[mi_stage * 8 + 3];
            int e_ctx = -1;
            e_ctx = smem_it[mi_stage * 8 + 4];
            int e_claim_q = smem_it[mi_stage * 8 + 5];
            int e_claim_kv = smem_it[mi_stage * 8 + 6];
            int e_claim_stop = smem_it[mi_stage * 8 + 7];
            mbarrier_arrive(it_empty_addr + (mi_stage) * 8);
            mi_stage += 1;
            if (mi_stage == 8) { mi_stage = 0; _phase_it_full ^= 1; }
            unsigned int _phase_q_full = 0;
            unsigned int _phase_kv_full_g0 = 0;
            #pragma unroll 1
            for (int _e_trip = 0; _e_trip < gridDim.x * 8 + 1; _e_trip++) {
                int e_q_stop = ((e_end_kv > 0) ? e_end_q + 1 : e_end_q);
                int e_claim_ctx = -1;
                e_claim_ctx = ((e_claim_q == e_start_q) ? e_ctx : -1);
                #pragma unroll 1
                for (int q_atom_4 = e_start_q; q_atom_4 < e_claim_stop; q_atom_4++) {
                    int q_idx = q_atom_4;
                    int atom_in_q = 0;
                    int tok = q_idx * 3 + atom_in_q * 3;
                    int c = -1;
                    if (q_atom_4 == e_start_q) {
                        c = e_ctx;
                    }
                    if (c < 0) {
                        c = context_lens[q_idx * 3 + 3 - 1];
                    }
                    int num_kv = (c + 64 - 1) / 64;
                    int lo = ((q_atom_4 == e_start_q) ? e_start_kv : 0);
                    int hi = ((q_atom_4 == e_claim_q) ? e_claim_kv : num_kv);
                    mbarrier_wait(q_full_addr + (q_stage_m) * 8, _phase_q_full);
                    int b_row = b_row_lane;
                    int b_cb = b_col;
                    unsigned int b_addr = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row * 128) + (unsigned int)(b_cb ^ (b_row & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[0]), "=r"(b_atom[1])
                        : "r"(b_addr)
                        : "memory");
                    int b_row_0 = b_row_lane;
                    int b_cb_0 = b_col + 32;
                    unsigned int b_addr_0 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_0 * 128) + (unsigned int)(b_cb_0 ^ (b_row_0 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[2]), "=r"(b_atom[3])
                        : "r"(b_addr_0)
                        : "memory");
                    int b_row_1 = b_row_lane;
                    int b_cb_1 = b_col + 64;
                    unsigned int b_addr_1 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_1 * 128) + (unsigned int)(b_cb_1 ^ (b_row_1 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[4]), "=r"(b_atom[5])
                        : "r"(b_addr_1)
                        : "memory");
                    int b_row_2 = b_row_lane;
                    int b_cb_2 = b_col + 96;
                    unsigned int b_addr_2 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_2 * 128) + (unsigned int)(b_cb_2 ^ (b_row_2 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[6]), "=r"(b_atom[7])
                        : "r"(b_addr_2)
                        : "memory");
                    int b_row_3 = 8 + b_row_lane;
                    int b_cb_3 = b_col;
                    unsigned int b_addr_3 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_3 * 128) + (unsigned int)(b_cb_3 ^ (b_row_3 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[8]), "=r"(b_atom[9])
                        : "r"(b_addr_3)
                        : "memory");
                    int b_row_4 = 8 + b_row_lane;
                    int b_cb_4 = b_col + 32;
                    unsigned int b_addr_4 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_4 * 128) + (unsigned int)(b_cb_4 ^ (b_row_4 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[10]), "=r"(b_atom[11])
                        : "r"(b_addr_4)
                        : "memory");
                    int b_row_5 = 8 + b_row_lane;
                    int b_cb_5 = b_col + 64;
                    unsigned int b_addr_5 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_5 * 128) + (unsigned int)(b_cb_5 ^ (b_row_5 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[12]), "=r"(b_atom[13])
                        : "r"(b_addr_5)
                        : "memory");
                    int b_row_6 = 8 + b_row_lane;
                    int b_cb_6 = b_col + 96;
                    unsigned int b_addr_6 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_6 * 128) + (unsigned int)(b_cb_6 ^ (b_row_6 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[14]), "=r"(b_atom[15])
                        : "r"(b_addr_6)
                        : "memory");
                    int b_row_7 = 16 + b_row_lane;
                    int b_cb_7 = b_col;
                    unsigned int b_addr_7 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_7 * 128) + (unsigned int)(b_cb_7 ^ (b_row_7 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[16]), "=r"(b_atom[17])
                        : "r"(b_addr_7)
                        : "memory");
                    int b_row_8 = 16 + b_row_lane;
                    int b_cb_8 = b_col + 32;
                    unsigned int b_addr_8 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_8 * 128) + (unsigned int)(b_cb_8 ^ (b_row_8 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[18]), "=r"(b_atom[19])
                        : "r"(b_addr_8)
                        : "memory");
                    int b_row_9 = 16 + b_row_lane;
                    int b_cb_9 = b_col + 64;
                    unsigned int b_addr_9 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_9 * 128) + (unsigned int)(b_cb_9 ^ (b_row_9 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[20]), "=r"(b_atom[21])
                        : "r"(b_addr_9)
                        : "memory");
                    int b_row_10 = 16 + b_row_lane;
                    int b_cb_10 = b_col + 96;
                    unsigned int b_addr_10 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_10 * 128) + (unsigned int)(b_cb_10 ^ (b_row_10 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[22]), "=r"(b_atom[23])
                        : "r"(b_addr_10)
                        : "memory");
                    int b_row_11 = 24 + b_row_lane;
                    int b_cb_11 = b_col;
                    unsigned int b_addr_11 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_11 * 128) + (unsigned int)(b_cb_11 ^ (b_row_11 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[24]), "=r"(b_atom[25])
                        : "r"(b_addr_11)
                        : "memory");
                    int b_row_12 = 24 + b_row_lane;
                    int b_cb_12 = b_col + 32;
                    unsigned int b_addr_12 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_12 * 128) + (unsigned int)(b_cb_12 ^ (b_row_12 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[26]), "=r"(b_atom[27])
                        : "r"(b_addr_12)
                        : "memory");
                    int b_row_13 = 24 + b_row_lane;
                    int b_cb_13 = b_col + 64;
                    unsigned int b_addr_13 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_13 * 128) + (unsigned int)(b_cb_13 ^ (b_row_13 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[28]), "=r"(b_atom[29])
                        : "r"(b_addr_13)
                        : "memory");
                    int b_row_14 = 24 + b_row_lane;
                    int b_cb_14 = b_col + 96;
                    unsigned int b_addr_14 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_14 * 128) + (unsigned int)(b_cb_14 ^ (b_row_14 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[30]), "=r"(b_atom[31])
                        : "r"(b_addr_14)
                        : "memory");
                    int b_row_15 = 32 + b_row_lane;
                    int b_cb_15 = b_col;
                    unsigned int b_addr_15 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_15 * 128) + (unsigned int)(b_cb_15 ^ (b_row_15 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[32]), "=r"(b_atom[33])
                        : "r"(b_addr_15)
                        : "memory");
                    int b_row_16 = 32 + b_row_lane;
                    int b_cb_16 = b_col + 32;
                    unsigned int b_addr_16 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_16 * 128) + (unsigned int)(b_cb_16 ^ (b_row_16 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[34]), "=r"(b_atom[35])
                        : "r"(b_addr_16)
                        : "memory");
                    int b_row_17 = 32 + b_row_lane;
                    int b_cb_17 = b_col + 64;
                    unsigned int b_addr_17 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_17 * 128) + (unsigned int)(b_cb_17 ^ (b_row_17 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[36]), "=r"(b_atom[37])
                        : "r"(b_addr_17)
                        : "memory");
                    int b_row_18 = 32 + b_row_lane;
                    int b_cb_18 = b_col + 96;
                    unsigned int b_addr_18 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_18 * 128) + (unsigned int)(b_cb_18 ^ (b_row_18 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[38]), "=r"(b_atom[39])
                        : "r"(b_addr_18)
                        : "memory");
                    int b_row_19 = 40 + b_row_lane;
                    int b_cb_19 = b_col;
                    unsigned int b_addr_19 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_19 * 128) + (unsigned int)(b_cb_19 ^ (b_row_19 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[40]), "=r"(b_atom[41])
                        : "r"(b_addr_19)
                        : "memory");
                    int b_row_20 = 40 + b_row_lane;
                    int b_cb_20 = b_col + 32;
                    unsigned int b_addr_20 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_20 * 128) + (unsigned int)(b_cb_20 ^ (b_row_20 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[42]), "=r"(b_atom[43])
                        : "r"(b_addr_20)
                        : "memory");
                    int b_row_21 = 40 + b_row_lane;
                    int b_cb_21 = b_col + 64;
                    unsigned int b_addr_21 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_21 * 128) + (unsigned int)(b_cb_21 ^ (b_row_21 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[44]), "=r"(b_atom[45])
                        : "r"(b_addr_21)
                        : "memory");
                    int b_row_22 = 40 + b_row_lane;
                    int b_cb_22 = b_col + 96;
                    unsigned int b_addr_22 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_22 * 128) + (unsigned int)(b_cb_22 ^ (b_row_22 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[46]), "=r"(b_atom[47])
                        : "r"(b_addr_22)
                        : "memory");
                    int b_row_23 = 48 + b_row_lane;
                    int b_cb_23 = b_col;
                    unsigned int b_addr_23 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_23 * 128) + (unsigned int)(b_cb_23 ^ (b_row_23 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[48]), "=r"(b_atom[49])
                        : "r"(b_addr_23)
                        : "memory");
                    int b_row_24 = 48 + b_row_lane;
                    int b_cb_24 = b_col + 32;
                    unsigned int b_addr_24 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_24 * 128) + (unsigned int)(b_cb_24 ^ (b_row_24 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[50]), "=r"(b_atom[51])
                        : "r"(b_addr_24)
                        : "memory");
                    int b_row_25 = 48 + b_row_lane;
                    int b_cb_25 = b_col + 64;
                    unsigned int b_addr_25 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_25 * 128) + (unsigned int)(b_cb_25 ^ (b_row_25 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[52]), "=r"(b_atom[53])
                        : "r"(b_addr_25)
                        : "memory");
                    int b_row_26 = 48 + b_row_lane;
                    int b_cb_26 = b_col + 96;
                    unsigned int b_addr_26 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_26 * 128) + (unsigned int)(b_cb_26 ^ (b_row_26 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[54]), "=r"(b_atom[55])
                        : "r"(b_addr_26)
                        : "memory");
                    int b_row_27 = 56 + b_row_lane;
                    int b_cb_27 = b_col;
                    unsigned int b_addr_27 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_27 * 128) + (unsigned int)(b_cb_27 ^ (b_row_27 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[56]), "=r"(b_atom[57])
                        : "r"(b_addr_27)
                        : "memory");
                    int b_row_28 = 56 + b_row_lane;
                    int b_cb_28 = b_col + 32;
                    unsigned int b_addr_28 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_28 * 128) + (unsigned int)(b_cb_28 ^ (b_row_28 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[58]), "=r"(b_atom[59])
                        : "r"(b_addr_28)
                        : "memory");
                    int b_row_29 = 56 + b_row_lane;
                    int b_cb_29 = b_col + 64;
                    unsigned int b_addr_29 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_29 * 128) + (unsigned int)(b_cb_29 ^ (b_row_29 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[60]), "=r"(b_atom[61])
                        : "r"(b_addr_29)
                        : "memory");
                    int b_row_30 = 56 + b_row_lane;
                    int b_cb_30 = b_col + 96;
                    unsigned int b_addr_30 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_30 * 128) + (unsigned int)(b_cb_30 ^ (b_row_30 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[62]), "=r"(b_atom[63])
                        : "r"(b_addr_30)
                        : "memory");
                    int b_row_31 = 64 + b_row_lane;
                    int b_cb_31 = b_col;
                    unsigned int b_addr_31 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_31 * 128) + (unsigned int)(b_cb_31 ^ (b_row_31 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[64]), "=r"(b_atom[65])
                        : "r"(b_addr_31)
                        : "memory");
                    int b_row_32 = 64 + b_row_lane;
                    int b_cb_32 = b_col + 32;
                    unsigned int b_addr_32 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_32 * 128) + (unsigned int)(b_cb_32 ^ (b_row_32 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[66]), "=r"(b_atom[67])
                        : "r"(b_addr_32)
                        : "memory");
                    int b_row_33 = 64 + b_row_lane;
                    int b_cb_33 = b_col + 64;
                    unsigned int b_addr_33 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_33 * 128) + (unsigned int)(b_cb_33 ^ (b_row_33 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[68]), "=r"(b_atom[69])
                        : "r"(b_addr_33)
                        : "memory");
                    int b_row_34 = 64 + b_row_lane;
                    int b_cb_34 = b_col + 96;
                    unsigned int b_addr_34 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_34 * 128) + (unsigned int)(b_cb_34 ^ (b_row_34 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[70]), "=r"(b_atom[71])
                        : "r"(b_addr_34)
                        : "memory");
                    int b_row_35 = 72 + b_row_lane;
                    int b_cb_35 = b_col;
                    unsigned int b_addr_35 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_35 * 128) + (unsigned int)(b_cb_35 ^ (b_row_35 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[72]), "=r"(b_atom[73])
                        : "r"(b_addr_35)
                        : "memory");
                    int b_row_36 = 72 + b_row_lane;
                    int b_cb_36 = b_col + 32;
                    unsigned int b_addr_36 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_36 * 128) + (unsigned int)(b_cb_36 ^ (b_row_36 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[74]), "=r"(b_atom[75])
                        : "r"(b_addr_36)
                        : "memory");
                    int b_row_37 = 72 + b_row_lane;
                    int b_cb_37 = b_col + 64;
                    unsigned int b_addr_37 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_37 * 128) + (unsigned int)(b_cb_37 ^ (b_row_37 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[76]), "=r"(b_atom[77])
                        : "r"(b_addr_37)
                        : "memory");
                    int b_row_38 = 72 + b_row_lane;
                    int b_cb_38 = b_col + 96;
                    unsigned int b_addr_38 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_38 * 128) + (unsigned int)(b_cb_38 ^ (b_row_38 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[78]), "=r"(b_atom[79])
                        : "r"(b_addr_38)
                        : "memory");
                    int b_row_39 = 80 + b_row_lane;
                    int b_cb_39 = b_col;
                    unsigned int b_addr_39 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_39 * 128) + (unsigned int)(b_cb_39 ^ (b_row_39 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[80]), "=r"(b_atom[81])
                        : "r"(b_addr_39)
                        : "memory");
                    int b_row_40 = 80 + b_row_lane;
                    int b_cb_40 = b_col + 32;
                    unsigned int b_addr_40 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_40 * 128) + (unsigned int)(b_cb_40 ^ (b_row_40 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[82]), "=r"(b_atom[83])
                        : "r"(b_addr_40)
                        : "memory");
                    int b_row_41 = 80 + b_row_lane;
                    int b_cb_41 = b_col + 64;
                    unsigned int b_addr_41 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_41 * 128) + (unsigned int)(b_cb_41 ^ (b_row_41 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[84]), "=r"(b_atom[85])
                        : "r"(b_addr_41)
                        : "memory");
                    int b_row_42 = 80 + b_row_lane;
                    int b_cb_42 = b_col + 96;
                    unsigned int b_addr_42 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_42 * 128) + (unsigned int)(b_cb_42 ^ (b_row_42 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[86]), "=r"(b_atom[87])
                        : "r"(b_addr_42)
                        : "memory");
                    int b_row_43 = 88 + b_row_lane;
                    int b_cb_43 = b_col;
                    unsigned int b_addr_43 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_43 * 128) + (unsigned int)(b_cb_43 ^ (b_row_43 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[88]), "=r"(b_atom[89])
                        : "r"(b_addr_43)
                        : "memory");
                    int b_row_44 = 88 + b_row_lane;
                    int b_cb_44 = b_col + 32;
                    unsigned int b_addr_44 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_44 * 128) + (unsigned int)(b_cb_44 ^ (b_row_44 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[90]), "=r"(b_atom[91])
                        : "r"(b_addr_44)
                        : "memory");
                    int b_row_45 = 88 + b_row_lane;
                    int b_cb_45 = b_col + 64;
                    unsigned int b_addr_45 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_45 * 128) + (unsigned int)(b_cb_45 ^ (b_row_45 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[92]), "=r"(b_atom[93])
                        : "r"(b_addr_45)
                        : "memory");
                    int b_row_46 = 88 + b_row_lane;
                    int b_cb_46 = b_col + 96;
                    unsigned int b_addr_46 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_46 * 128) + (unsigned int)(b_cb_46 ^ (b_row_46 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[94]), "=r"(b_atom[95])
                        : "r"(b_addr_46)
                        : "memory");
                    int n_tok_m = 3;
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
                        for (int nt = 0; nt < 4; nt++) {
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
                        float partial0_0 = 0.0f;
                        float partial1_0 = 0.0f;
                        #pragma unroll
                        for (int nt_1 = 0; nt_1 < 4; nt_1++) {
                            acc[0] = 0.0f;
                            acc[1] = 0.0f;
                            acc[2] = 0.0f;
                            acc[3] = 0.0f;
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                                : "r"(a_tile[0]), "r"(a_tile[1]), "r"(a_tile[2]), "r"(a_tile[3]), "r"((b_atom + (4 + nt_1) * 4 * 2)[0]), "r"((b_atom + (4 + nt_1) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[4]), "r"(a_tile[(4) + 1]), "r"(a_tile[(4) + 2]), "r"(a_tile[(4) + 3]), "r"((b_atom + ((4 + nt_1) * 4 + 1) * 2)[0]), "r"((b_atom + ((4 + nt_1) * 4 + 1) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[8]), "r"(a_tile[(8) + 1]), "r"(a_tile[(8) + 2]), "r"(a_tile[(8) + 3]), "r"((b_atom + ((4 + nt_1) * 4 + 2) * 2)[0]), "r"((b_atom + ((4 + nt_1) * 4 + 2) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[12]), "r"(a_tile[(12) + 1]), "r"(a_tile[(12) + 2]), "r"(a_tile[(12) + 3]), "r"((b_atom + ((4 + nt_1) * 4 + 3) * 2)[0]), "r"((b_atom + ((4 + nt_1) * 4 + 3) * 2)[1]));
                            int w_idx_1 = q_stage_m * 256 + 32 + (unsigned int)(nt_1 * 8) + (unsigned int)(tq * 2);
                            float w0_1 = smem_w[w_idx_1];
                            float w1_1 = smem_w[w_idx_1 + 1];
                            float r0_1 = fmaxf(acc[0], 0.0f);
                            float r1_1 = fmaxf(acc[1], 0.0f);
                            float r2_1 = fmaxf(acc[2], 0.0f);
                            float r3_1 = fmaxf(acc[3], 0.0f);
                            partial0_0 += __fmaf_rn(r0_1, w0_1, r1_1 * w1_1);
                            partial1_0 += __fmaf_rn(r2_1, w0_1, r3_1 * w1_1);
                        }
                        float v0_0 = partial0_0 * scale0;
                        float v1_0 = partial1_0 * scale1;
                        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, v0_0, 1);
                        v0_0 = v0_0 + _shfl_xor_sync_4;
                        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, v1_0, 1);
                        v1_0 = v1_0 + _shfl_xor_sync_5;
                        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, v0_0, 2);
                        v0_0 = v0_0 + _shfl_xor_sync_6;
                        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, v1_0, 2);
                        v1_0 = v1_0 + _shfl_xor_sync_7;
                        int out0_0 = kv_offset + logits_stride + quad;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_0)), "f"(v0_0), "r"((unsigned int)(tq == 0)));
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_0 + 8))), "f"(v1_0), "r"((unsigned int)(tq == 0)));
                        float partial0_1 = 0.0f;
                        float partial1_1 = 0.0f;
                        #pragma unroll
                        for (int nt_2 = 0; nt_2 < 4; nt_2++) {
                            acc[0] = 0.0f;
                            acc[1] = 0.0f;
                            acc[2] = 0.0f;
                            acc[3] = 0.0f;
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                                : "r"(a_tile[0]), "r"(a_tile[1]), "r"(a_tile[2]), "r"(a_tile[3]), "r"((b_atom + (8 + nt_2) * 4 * 2)[0]), "r"((b_atom + (8 + nt_2) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[4]), "r"(a_tile[(4) + 1]), "r"(a_tile[(4) + 2]), "r"(a_tile[(4) + 3]), "r"((b_atom + ((8 + nt_2) * 4 + 1) * 2)[0]), "r"((b_atom + ((8 + nt_2) * 4 + 1) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[8]), "r"(a_tile[(8) + 1]), "r"(a_tile[(8) + 2]), "r"(a_tile[(8) + 3]), "r"((b_atom + ((8 + nt_2) * 4 + 2) * 2)[0]), "r"((b_atom + ((8 + nt_2) * 4 + 2) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[12]), "r"(a_tile[(12) + 1]), "r"(a_tile[(12) + 2]), "r"(a_tile[(12) + 3]), "r"((b_atom + ((8 + nt_2) * 4 + 3) * 2)[0]), "r"((b_atom + ((8 + nt_2) * 4 + 3) * 2)[1]));
                            int w_idx_2 = q_stage_m * 256 + 64 + (unsigned int)(nt_2 * 8) + (unsigned int)(tq * 2);
                            float w0_2 = smem_w[w_idx_2];
                            float w1_2 = smem_w[w_idx_2 + 1];
                            float r0_2 = fmaxf(acc[0], 0.0f);
                            float r1_2 = fmaxf(acc[1], 0.0f);
                            float r2_2 = fmaxf(acc[2], 0.0f);
                            float r3_2 = fmaxf(acc[3], 0.0f);
                            partial0_1 += __fmaf_rn(r0_2, w0_2, r1_2 * w1_2);
                            partial1_1 += __fmaf_rn(r2_2, w0_2, r3_2 * w1_2);
                        }
                        float v0_1 = partial0_1 * scale0;
                        float v1_1 = partial1_1 * scale1;
                        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, v0_1, 1);
                        v0_1 = v0_1 + _shfl_xor_sync_8;
                        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 1);
                        v1_1 = v1_1 + _shfl_xor_sync_9;
                        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, v0_1, 2);
                        v0_1 = v0_1 + _shfl_xor_sync_10;
                        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 2);
                        v1_1 = v1_1 + _shfl_xor_sync_11;
                        int out0_1 = kv_offset + 2 * logits_stride + quad;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_1)), "f"(v0_1), "r"((unsigned int)(tq == 0)));
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_1 + 8))), "f"(v1_1), "r"((unsigned int)(tq == 0)));
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
                #pragma unroll 1
                for (int q_atom_5 = e_claim_q; q_atom_5 < e_q_stop; q_atom_5++) {
                    int q_idx_1 = q_atom_5;
                    int atom_in_q_1 = 0;
                    int tok_1 = q_idx_1 * 3 + atom_in_q_1 * 3;
                    int c_1 = -1;
                    if (q_atom_5 == e_claim_q) {
                        c_1 = e_claim_ctx;
                    }
                    if (c_1 < 0) {
                        c_1 = context_lens[q_idx_1 * 3 + 3 - 1];
                    }
                    int num_kv_1 = (c_1 + 64 - 1) / 64;
                    int lo_1 = ((q_atom_5 == e_claim_q) ? e_claim_kv : 0);
                    int hi_1 = ((q_atom_5 == e_end_q) ? e_end_kv : num_kv_1);
                    mbarrier_wait(q_full_addr + (q_stage_m) * 8, _phase_q_full);
                    int b_row_47 = b_row_lane;
                    int b_cb_47 = b_col;
                    unsigned int b_addr_47 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_47 * 128) + (unsigned int)(b_cb_47 ^ (b_row_47 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[0]), "=r"(b_atom[1])
                        : "r"(b_addr_47)
                        : "memory");
                    int b_row_47_1 = b_row_lane;
                    int b_cb_47_1 = b_col + 32;
                    unsigned int b_addr_47_1 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_47_1 * 128) + (unsigned int)(b_cb_47_1 ^ (b_row_47_1 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[2]), "=r"(b_atom[3])
                        : "r"(b_addr_47_1)
                        : "memory");
                    int b_row_48 = b_row_lane;
                    int b_cb_48 = b_col + 64;
                    unsigned int b_addr_48 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_48 * 128) + (unsigned int)(b_cb_48 ^ (b_row_48 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[4]), "=r"(b_atom[5])
                        : "r"(b_addr_48)
                        : "memory");
                    int b_row_49 = b_row_lane;
                    int b_cb_49 = b_col + 96;
                    unsigned int b_addr_49 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_49 * 128) + (unsigned int)(b_cb_49 ^ (b_row_49 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[6]), "=r"(b_atom[7])
                        : "r"(b_addr_49)
                        : "memory");
                    int b_row_50 = 8 + b_row_lane;
                    int b_cb_50 = b_col;
                    unsigned int b_addr_50 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_50 * 128) + (unsigned int)(b_cb_50 ^ (b_row_50 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[8]), "=r"(b_atom[9])
                        : "r"(b_addr_50)
                        : "memory");
                    int b_row_51 = 8 + b_row_lane;
                    int b_cb_51 = b_col + 32;
                    unsigned int b_addr_51 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_51 * 128) + (unsigned int)(b_cb_51 ^ (b_row_51 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[10]), "=r"(b_atom[11])
                        : "r"(b_addr_51)
                        : "memory");
                    int b_row_52 = 8 + b_row_lane;
                    int b_cb_52 = b_col + 64;
                    unsigned int b_addr_52 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_52 * 128) + (unsigned int)(b_cb_52 ^ (b_row_52 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[12]), "=r"(b_atom[13])
                        : "r"(b_addr_52)
                        : "memory");
                    int b_row_53 = 8 + b_row_lane;
                    int b_cb_53 = b_col + 96;
                    unsigned int b_addr_53 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_53 * 128) + (unsigned int)(b_cb_53 ^ (b_row_53 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[14]), "=r"(b_atom[15])
                        : "r"(b_addr_53)
                        : "memory");
                    int b_row_54 = 16 + b_row_lane;
                    int b_cb_54 = b_col;
                    unsigned int b_addr_54 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_54 * 128) + (unsigned int)(b_cb_54 ^ (b_row_54 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[16]), "=r"(b_atom[17])
                        : "r"(b_addr_54)
                        : "memory");
                    int b_row_55 = 16 + b_row_lane;
                    int b_cb_55 = b_col + 32;
                    unsigned int b_addr_55 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_55 * 128) + (unsigned int)(b_cb_55 ^ (b_row_55 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[18]), "=r"(b_atom[19])
                        : "r"(b_addr_55)
                        : "memory");
                    int b_row_56 = 16 + b_row_lane;
                    int b_cb_56 = b_col + 64;
                    unsigned int b_addr_56 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_56 * 128) + (unsigned int)(b_cb_56 ^ (b_row_56 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[20]), "=r"(b_atom[21])
                        : "r"(b_addr_56)
                        : "memory");
                    int b_row_57 = 16 + b_row_lane;
                    int b_cb_57 = b_col + 96;
                    unsigned int b_addr_57 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_57 * 128) + (unsigned int)(b_cb_57 ^ (b_row_57 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[22]), "=r"(b_atom[23])
                        : "r"(b_addr_57)
                        : "memory");
                    int b_row_58 = 24 + b_row_lane;
                    int b_cb_58 = b_col;
                    unsigned int b_addr_58 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_58 * 128) + (unsigned int)(b_cb_58 ^ (b_row_58 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[24]), "=r"(b_atom[25])
                        : "r"(b_addr_58)
                        : "memory");
                    int b_row_59 = 24 + b_row_lane;
                    int b_cb_59 = b_col + 32;
                    unsigned int b_addr_59 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_59 * 128) + (unsigned int)(b_cb_59 ^ (b_row_59 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[26]), "=r"(b_atom[27])
                        : "r"(b_addr_59)
                        : "memory");
                    int b_row_60 = 24 + b_row_lane;
                    int b_cb_60 = b_col + 64;
                    unsigned int b_addr_60 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_60 * 128) + (unsigned int)(b_cb_60 ^ (b_row_60 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[28]), "=r"(b_atom[29])
                        : "r"(b_addr_60)
                        : "memory");
                    int b_row_61 = 24 + b_row_lane;
                    int b_cb_61 = b_col + 96;
                    unsigned int b_addr_61 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_61 * 128) + (unsigned int)(b_cb_61 ^ (b_row_61 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[30]), "=r"(b_atom[31])
                        : "r"(b_addr_61)
                        : "memory");
                    int b_row_62 = 32 + b_row_lane;
                    int b_cb_62 = b_col;
                    unsigned int b_addr_62 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_62 * 128) + (unsigned int)(b_cb_62 ^ (b_row_62 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[32]), "=r"(b_atom[33])
                        : "r"(b_addr_62)
                        : "memory");
                    int b_row_63 = 32 + b_row_lane;
                    int b_cb_63 = b_col + 32;
                    unsigned int b_addr_63 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_63 * 128) + (unsigned int)(b_cb_63 ^ (b_row_63 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[34]), "=r"(b_atom[35])
                        : "r"(b_addr_63)
                        : "memory");
                    int b_row_64 = 32 + b_row_lane;
                    int b_cb_64 = b_col + 64;
                    unsigned int b_addr_64 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_64 * 128) + (unsigned int)(b_cb_64 ^ (b_row_64 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[36]), "=r"(b_atom[37])
                        : "r"(b_addr_64)
                        : "memory");
                    int b_row_65 = 32 + b_row_lane;
                    int b_cb_65 = b_col + 96;
                    unsigned int b_addr_65 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_65 * 128) + (unsigned int)(b_cb_65 ^ (b_row_65 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[38]), "=r"(b_atom[39])
                        : "r"(b_addr_65)
                        : "memory");
                    int b_row_66 = 40 + b_row_lane;
                    int b_cb_66 = b_col;
                    unsigned int b_addr_66 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_66 * 128) + (unsigned int)(b_cb_66 ^ (b_row_66 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[40]), "=r"(b_atom[41])
                        : "r"(b_addr_66)
                        : "memory");
                    int b_row_67 = 40 + b_row_lane;
                    int b_cb_67 = b_col + 32;
                    unsigned int b_addr_67 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_67 * 128) + (unsigned int)(b_cb_67 ^ (b_row_67 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[42]), "=r"(b_atom[43])
                        : "r"(b_addr_67)
                        : "memory");
                    int b_row_68 = 40 + b_row_lane;
                    int b_cb_68 = b_col + 64;
                    unsigned int b_addr_68 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_68 * 128) + (unsigned int)(b_cb_68 ^ (b_row_68 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[44]), "=r"(b_atom[45])
                        : "r"(b_addr_68)
                        : "memory");
                    int b_row_69 = 40 + b_row_lane;
                    int b_cb_69 = b_col + 96;
                    unsigned int b_addr_69 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_69 * 128) + (unsigned int)(b_cb_69 ^ (b_row_69 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[46]), "=r"(b_atom[47])
                        : "r"(b_addr_69)
                        : "memory");
                    int b_row_70 = 48 + b_row_lane;
                    int b_cb_70 = b_col;
                    unsigned int b_addr_70 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_70 * 128) + (unsigned int)(b_cb_70 ^ (b_row_70 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[48]), "=r"(b_atom[49])
                        : "r"(b_addr_70)
                        : "memory");
                    int b_row_71 = 48 + b_row_lane;
                    int b_cb_71 = b_col + 32;
                    unsigned int b_addr_71 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_71 * 128) + (unsigned int)(b_cb_71 ^ (b_row_71 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[50]), "=r"(b_atom[51])
                        : "r"(b_addr_71)
                        : "memory");
                    int b_row_72 = 48 + b_row_lane;
                    int b_cb_72 = b_col + 64;
                    unsigned int b_addr_72 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_72 * 128) + (unsigned int)(b_cb_72 ^ (b_row_72 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[52]), "=r"(b_atom[53])
                        : "r"(b_addr_72)
                        : "memory");
                    int b_row_73 = 48 + b_row_lane;
                    int b_cb_73 = b_col + 96;
                    unsigned int b_addr_73 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_73 * 128) + (unsigned int)(b_cb_73 ^ (b_row_73 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[54]), "=r"(b_atom[55])
                        : "r"(b_addr_73)
                        : "memory");
                    int b_row_74 = 56 + b_row_lane;
                    int b_cb_74 = b_col;
                    unsigned int b_addr_74 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_74 * 128) + (unsigned int)(b_cb_74 ^ (b_row_74 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[56]), "=r"(b_atom[57])
                        : "r"(b_addr_74)
                        : "memory");
                    int b_row_75 = 56 + b_row_lane;
                    int b_cb_75 = b_col + 32;
                    unsigned int b_addr_75 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_75 * 128) + (unsigned int)(b_cb_75 ^ (b_row_75 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[58]), "=r"(b_atom[59])
                        : "r"(b_addr_75)
                        : "memory");
                    int b_row_76 = 56 + b_row_lane;
                    int b_cb_76 = b_col + 64;
                    unsigned int b_addr_76 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_76 * 128) + (unsigned int)(b_cb_76 ^ (b_row_76 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[60]), "=r"(b_atom[61])
                        : "r"(b_addr_76)
                        : "memory");
                    int b_row_77 = 56 + b_row_lane;
                    int b_cb_77 = b_col + 96;
                    unsigned int b_addr_77 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_77 * 128) + (unsigned int)(b_cb_77 ^ (b_row_77 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[62]), "=r"(b_atom[63])
                        : "r"(b_addr_77)
                        : "memory");
                    int b_row_78 = 64 + b_row_lane;
                    int b_cb_78 = b_col;
                    unsigned int b_addr_78 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_78 * 128) + (unsigned int)(b_cb_78 ^ (b_row_78 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[64]), "=r"(b_atom[65])
                        : "r"(b_addr_78)
                        : "memory");
                    int b_row_79 = 64 + b_row_lane;
                    int b_cb_79 = b_col + 32;
                    unsigned int b_addr_79 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_79 * 128) + (unsigned int)(b_cb_79 ^ (b_row_79 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[66]), "=r"(b_atom[67])
                        : "r"(b_addr_79)
                        : "memory");
                    int b_row_80 = 64 + b_row_lane;
                    int b_cb_80 = b_col + 64;
                    unsigned int b_addr_80 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_80 * 128) + (unsigned int)(b_cb_80 ^ (b_row_80 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[68]), "=r"(b_atom[69])
                        : "r"(b_addr_80)
                        : "memory");
                    int b_row_81 = 64 + b_row_lane;
                    int b_cb_81 = b_col + 96;
                    unsigned int b_addr_81 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_81 * 128) + (unsigned int)(b_cb_81 ^ (b_row_81 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[70]), "=r"(b_atom[71])
                        : "r"(b_addr_81)
                        : "memory");
                    int b_row_82 = 72 + b_row_lane;
                    int b_cb_82 = b_col;
                    unsigned int b_addr_82 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_82 * 128) + (unsigned int)(b_cb_82 ^ (b_row_82 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[72]), "=r"(b_atom[73])
                        : "r"(b_addr_82)
                        : "memory");
                    int b_row_83 = 72 + b_row_lane;
                    int b_cb_83 = b_col + 32;
                    unsigned int b_addr_83 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_83 * 128) + (unsigned int)(b_cb_83 ^ (b_row_83 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[74]), "=r"(b_atom[75])
                        : "r"(b_addr_83)
                        : "memory");
                    int b_row_84 = 72 + b_row_lane;
                    int b_cb_84 = b_col + 64;
                    unsigned int b_addr_84 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_84 * 128) + (unsigned int)(b_cb_84 ^ (b_row_84 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[76]), "=r"(b_atom[77])
                        : "r"(b_addr_84)
                        : "memory");
                    int b_row_85 = 72 + b_row_lane;
                    int b_cb_85 = b_col + 96;
                    unsigned int b_addr_85 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_85 * 128) + (unsigned int)(b_cb_85 ^ (b_row_85 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[78]), "=r"(b_atom[79])
                        : "r"(b_addr_85)
                        : "memory");
                    int b_row_86 = 80 + b_row_lane;
                    int b_cb_86 = b_col;
                    unsigned int b_addr_86 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_86 * 128) + (unsigned int)(b_cb_86 ^ (b_row_86 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[80]), "=r"(b_atom[81])
                        : "r"(b_addr_86)
                        : "memory");
                    int b_row_87 = 80 + b_row_lane;
                    int b_cb_87 = b_col + 32;
                    unsigned int b_addr_87 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_87 * 128) + (unsigned int)(b_cb_87 ^ (b_row_87 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[82]), "=r"(b_atom[83])
                        : "r"(b_addr_87)
                        : "memory");
                    int b_row_88 = 80 + b_row_lane;
                    int b_cb_88 = b_col + 64;
                    unsigned int b_addr_88 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_88 * 128) + (unsigned int)(b_cb_88 ^ (b_row_88 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[84]), "=r"(b_atom[85])
                        : "r"(b_addr_88)
                        : "memory");
                    int b_row_89 = 80 + b_row_lane;
                    int b_cb_89 = b_col + 96;
                    unsigned int b_addr_89 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_89 * 128) + (unsigned int)(b_cb_89 ^ (b_row_89 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[86]), "=r"(b_atom[87])
                        : "r"(b_addr_89)
                        : "memory");
                    int b_row_90 = 88 + b_row_lane;
                    int b_cb_90 = b_col;
                    unsigned int b_addr_90 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_90 * 128) + (unsigned int)(b_cb_90 ^ (b_row_90 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[88]), "=r"(b_atom[89])
                        : "r"(b_addr_90)
                        : "memory");
                    int b_row_91 = 88 + b_row_lane;
                    int b_cb_91 = b_col + 32;
                    unsigned int b_addr_91 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_91 * 128) + (unsigned int)(b_cb_91 ^ (b_row_91 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[90]), "=r"(b_atom[91])
                        : "r"(b_addr_91)
                        : "memory");
                    int b_row_92 = 88 + b_row_lane;
                    int b_cb_92 = b_col + 64;
                    unsigned int b_addr_92 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_92 * 128) + (unsigned int)(b_cb_92 ^ (b_row_92 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[92]), "=r"(b_atom[93])
                        : "r"(b_addr_92)
                        : "memory");
                    int b_row_93 = 88 + b_row_lane;
                    int b_cb_93 = b_col + 96;
                    unsigned int b_addr_93 = smem_q_addr + q_stage_m * 12288 + (unsigned int)(b_row_93 * 128) + (unsigned int)(b_cb_93 ^ (b_row_93 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom[94]), "=r"(b_atom[95])
                        : "r"(b_addr_93)
                        : "memory");
                    int n_tok_m_1 = 3;
                    #pragma unroll 1
                    for (int kv_idx_1 = lo_1; kv_idx_1 < hi_1; kv_idx_1 += 2) {
                        mbarrier_wait(kv_full_g0_addr + (kv_stage_m) * 8, _phase_kv_full_g0);
                        int sc_idx_1 = kv_stage_m * 256 + (unsigned int)(w_in_g * 16) + (unsigned int)quad;
                        float scale0_1 = smem_sc_g0[sc_idx_1];
                        float scale1_1 = smem_sc_g0[sc_idx_1 + 8];
                        int kv_offset_1 = tok_1 * logits_stride + kv_idx_1 * 64 + w_in_g * 16;
                        int a_cb_3 = a_col;
                        unsigned int a_addr_3 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_3 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_tile[0]), "=r"(a_tile[1]), "=r"(a_tile[2]), "=r"(a_tile[3])
                            : "r"(a_addr_3)
                            : "memory");
                        int a_cb_3_1 = a_col + 32;
                        unsigned int a_addr_3_1 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_3_1 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_tile[4]), "=r"(a_tile[5]), "=r"(a_tile[6]), "=r"(a_tile[7])
                            : "r"(a_addr_3_1)
                            : "memory");
                        int a_cb_4 = a_col + 64;
                        unsigned int a_addr_4 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_4 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_tile[8]), "=r"(a_tile[9]), "=r"(a_tile[10]), "=r"(a_tile[11])
                            : "r"(a_addr_4)
                            : "memory");
                        int a_cb_5 = a_col + 96;
                        unsigned int a_addr_5 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_5 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_tile[12]), "=r"(a_tile[13]), "=r"(a_tile[14]), "=r"(a_tile[15])
                            : "r"(a_addr_5)
                            : "memory");
                        float partial0_2 = 0.0f;
                        float partial1_2 = 0.0f;
                        #pragma unroll
                        for (int nt_3 = 0; nt_3 < 4; nt_3++) {
                            acc[0] = 0.0f;
                            acc[1] = 0.0f;
                            acc[2] = 0.0f;
                            acc[3] = 0.0f;
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                                : "r"(a_tile[0]), "r"(a_tile[1]), "r"(a_tile[2]), "r"(a_tile[3]), "r"((b_atom + nt_3 * 4 * 2)[0]), "r"((b_atom + nt_3 * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[4]), "r"(a_tile[(4) + 1]), "r"(a_tile[(4) + 2]), "r"(a_tile[(4) + 3]), "r"((b_atom + (nt_3 * 4 + 1) * 2)[0]), "r"((b_atom + (nt_3 * 4 + 1) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[8]), "r"(a_tile[(8) + 1]), "r"(a_tile[(8) + 2]), "r"(a_tile[(8) + 3]), "r"((b_atom + (nt_3 * 4 + 2) * 2)[0]), "r"((b_atom + (nt_3 * 4 + 2) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[12]), "r"(a_tile[(12) + 1]), "r"(a_tile[(12) + 2]), "r"(a_tile[(12) + 3]), "r"((b_atom + (nt_3 * 4 + 3) * 2)[0]), "r"((b_atom + (nt_3 * 4 + 3) * 2)[1]));
                            int w_idx_3 = q_stage_m * 256 + (unsigned int)(nt_3 * 8) + (unsigned int)(tq * 2);
                            float w0_3 = smem_w[w_idx_3];
                            float w1_3 = smem_w[w_idx_3 + 1];
                            float r0_3 = fmaxf(acc[0], 0.0f);
                            float r1_3 = fmaxf(acc[1], 0.0f);
                            float r2_3 = fmaxf(acc[2], 0.0f);
                            float r3_3 = fmaxf(acc[3], 0.0f);
                            partial0_2 += __fmaf_rn(r0_3, w0_3, r1_3 * w1_3);
                            partial1_2 += __fmaf_rn(r2_3, w0_3, r3_3 * w1_3);
                        }
                        float v0_2 = partial0_2 * scale0_1;
                        float v1_2 = partial1_2 * scale1_1;
                        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, v0_2, 1);
                        v0_2 = v0_2 + _shfl_xor_sync_12;
                        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, v1_2, 1);
                        v1_2 = v1_2 + _shfl_xor_sync_13;
                        float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, v0_2, 2);
                        v0_2 = v0_2 + _shfl_xor_sync_14;
                        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, v1_2, 2);
                        v1_2 = v1_2 + _shfl_xor_sync_15;
                        int out0_2 = kv_offset_1 + quad;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_2)), "f"(v0_2), "r"((unsigned int)(tq == 0)));
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_2 + 8))), "f"(v1_2), "r"((unsigned int)(tq == 0)));
                        float partial0_2_1 = 0.0f;
                        float partial1_2_1 = 0.0f;
                        #pragma unroll
                        for (int nt_4 = 0; nt_4 < 4; nt_4++) {
                            acc[0] = 0.0f;
                            acc[1] = 0.0f;
                            acc[2] = 0.0f;
                            acc[3] = 0.0f;
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                                : "r"(a_tile[0]), "r"(a_tile[1]), "r"(a_tile[2]), "r"(a_tile[3]), "r"((b_atom + (4 + nt_4) * 4 * 2)[0]), "r"((b_atom + (4 + nt_4) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[4]), "r"(a_tile[(4) + 1]), "r"(a_tile[(4) + 2]), "r"(a_tile[(4) + 3]), "r"((b_atom + ((4 + nt_4) * 4 + 1) * 2)[0]), "r"((b_atom + ((4 + nt_4) * 4 + 1) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[8]), "r"(a_tile[(8) + 1]), "r"(a_tile[(8) + 2]), "r"(a_tile[(8) + 3]), "r"((b_atom + ((4 + nt_4) * 4 + 2) * 2)[0]), "r"((b_atom + ((4 + nt_4) * 4 + 2) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[12]), "r"(a_tile[(12) + 1]), "r"(a_tile[(12) + 2]), "r"(a_tile[(12) + 3]), "r"((b_atom + ((4 + nt_4) * 4 + 3) * 2)[0]), "r"((b_atom + ((4 + nt_4) * 4 + 3) * 2)[1]));
                            int w_idx_4 = q_stage_m * 256 + 32 + (unsigned int)(nt_4 * 8) + (unsigned int)(tq * 2);
                            float w0_4 = smem_w[w_idx_4];
                            float w1_4 = smem_w[w_idx_4 + 1];
                            float r0_4 = fmaxf(acc[0], 0.0f);
                            float r1_4 = fmaxf(acc[1], 0.0f);
                            float r2_4 = fmaxf(acc[2], 0.0f);
                            float r3_4 = fmaxf(acc[3], 0.0f);
                            partial0_2_1 += __fmaf_rn(r0_4, w0_4, r1_4 * w1_4);
                            partial1_2_1 += __fmaf_rn(r2_4, w0_4, r3_4 * w1_4);
                        }
                        float v0_2_1 = partial0_2_1 * scale0_1;
                        float v1_2_1 = partial1_2_1 * scale1_1;
                        float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, v0_2_1, 1);
                        v0_2_1 = v0_2_1 + _shfl_xor_sync_16;
                        float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, v1_2_1, 1);
                        v1_2_1 = v1_2_1 + _shfl_xor_sync_17;
                        float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, v0_2_1, 2);
                        v0_2_1 = v0_2_1 + _shfl_xor_sync_18;
                        float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, v1_2_1, 2);
                        v1_2_1 = v1_2_1 + _shfl_xor_sync_19;
                        int out0_2_1 = kv_offset_1 + logits_stride + quad;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_2_1)), "f"(v0_2_1), "r"((unsigned int)(tq == 0)));
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_2_1 + 8))), "f"(v1_2_1), "r"((unsigned int)(tq == 0)));
                        float partial0_3 = 0.0f;
                        float partial1_3 = 0.0f;
                        #pragma unroll
                        for (int nt_5 = 0; nt_5 < 4; nt_5++) {
                            acc[0] = 0.0f;
                            acc[1] = 0.0f;
                            acc[2] = 0.0f;
                            acc[3] = 0.0f;
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                                : "r"(a_tile[0]), "r"(a_tile[1]), "r"(a_tile[2]), "r"(a_tile[3]), "r"((b_atom + (8 + nt_5) * 4 * 2)[0]), "r"((b_atom + (8 + nt_5) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[4]), "r"(a_tile[(4) + 1]), "r"(a_tile[(4) + 2]), "r"(a_tile[(4) + 3]), "r"((b_atom + ((8 + nt_5) * 4 + 1) * 2)[0]), "r"((b_atom + ((8 + nt_5) * 4 + 1) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[8]), "r"(a_tile[(8) + 1]), "r"(a_tile[(8) + 2]), "r"(a_tile[(8) + 3]), "r"((b_atom + ((8 + nt_5) * 4 + 2) * 2)[0]), "r"((b_atom + ((8 + nt_5) * 4 + 2) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                                : "r"(a_tile[12]), "r"(a_tile[(12) + 1]), "r"(a_tile[(12) + 2]), "r"(a_tile[(12) + 3]), "r"((b_atom + ((8 + nt_5) * 4 + 3) * 2)[0]), "r"((b_atom + ((8 + nt_5) * 4 + 3) * 2)[1]));
                            int w_idx_5 = q_stage_m * 256 + 64 + (unsigned int)(nt_5 * 8) + (unsigned int)(tq * 2);
                            float w0_5 = smem_w[w_idx_5];
                            float w1_5 = smem_w[w_idx_5 + 1];
                            float r0_5 = fmaxf(acc[0], 0.0f);
                            float r1_5 = fmaxf(acc[1], 0.0f);
                            float r2_5 = fmaxf(acc[2], 0.0f);
                            float r3_5 = fmaxf(acc[3], 0.0f);
                            partial0_3 += __fmaf_rn(r0_5, w0_5, r1_5 * w1_5);
                            partial1_3 += __fmaf_rn(r2_5, w0_5, r3_5 * w1_5);
                        }
                        float v0_3 = partial0_3 * scale0_1;
                        float v1_3 = partial1_3 * scale1_1;
                        float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, v0_3, 1);
                        v0_3 = v0_3 + _shfl_xor_sync_20;
                        float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, v1_3, 1);
                        v1_3 = v1_3 + _shfl_xor_sync_21;
                        float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, v0_3, 2);
                        v0_3 = v0_3 + _shfl_xor_sync_22;
                        float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, v1_3, 2);
                        v1_3 = v1_3 + _shfl_xor_sync_23;
                        int out0_3 = kv_offset_1 + 2 * logits_stride + quad;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_3)), "f"(v0_3), "r"((unsigned int)(tq == 0)));
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_3 + 8))), "f"(v1_3), "r"((unsigned int)(tq == 0)));
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
                mbarrier_wait(it_full_addr + (mi_stage) * 8, _phase_it_full);
                e_start_q = smem_it[mi_stage * 8];
                e_start_kv = smem_it[mi_stage * 8 + 1];
                e_end_q = smem_it[mi_stage * 8 + 2];
                e_end_kv = smem_it[mi_stage * 8 + 3];
                e_ctx = smem_it[mi_stage * 8 + 4];
                e_claim_q = smem_it[mi_stage * 8 + 5];
                e_claim_kv = smem_it[mi_stage * 8 + 6];
                e_claim_stop = smem_it[mi_stage * 8 + 7];
                mbarrier_arrive(it_empty_addr + (mi_stage) * 8);
                mi_stage += 1;
                if (mi_stage == 8) { mi_stage = 0; _phase_it_full ^= 1; }
                if (e_start_q < 0) {
                    break;
                }
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
            unsigned int b_atom_1[96];
            unsigned int mi_stage_1 = 0;
            unsigned int _phase_it_full_1 = 0;
            mbarrier_wait(it_full_addr + (mi_stage_1) * 8, _phase_it_full_1);
            int e_start_q_1 = smem_it[mi_stage_1 * 8];
            int e_start_kv_1 = smem_it[mi_stage_1 * 8 + 1];
            int e_end_q_1 = smem_it[mi_stage_1 * 8 + 2];
            int e_end_kv_1 = smem_it[mi_stage_1 * 8 + 3];
            int e_ctx_1 = -1;
            e_ctx_1 = smem_it[mi_stage_1 * 8 + 4];
            int e_claim_q_1 = smem_it[mi_stage_1 * 8 + 5];
            int e_claim_kv_1 = smem_it[mi_stage_1 * 8 + 6];
            int e_claim_stop_1 = smem_it[mi_stage_1 * 8 + 7];
            mbarrier_arrive(it_empty_addr + (mi_stage_1) * 8);
            mi_stage_1 += 1;
            if (mi_stage_1 == 8) { mi_stage_1 = 0; _phase_it_full_1 ^= 1; }
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_kv_full_g1 = 0;
            #pragma unroll 1
            for (int _e_trip_1 = 0; _e_trip_1 < gridDim.x * 8 + 1; _e_trip_1++) {
                int e_q_stop_1 = ((e_end_kv_1 > 0) ? e_end_q_1 + 1 : e_end_q_1);
                int e_claim_ctx_1 = -1;
                e_claim_ctx_1 = ((e_claim_q_1 == e_start_q_1) ? e_ctx_1 : -1);
                #pragma unroll 1
                for (int q_atom_6 = e_start_q_1; q_atom_6 < e_claim_stop_1; q_atom_6++) {
                    int q_idx_2 = q_atom_6;
                    int atom_in_q_2 = 0;
                    int tok_2 = q_idx_2 * 3 + atom_in_q_2 * 3;
                    int c_2 = -1;
                    if (q_atom_6 == e_start_q_1) {
                        c_2 = e_ctx_1;
                    }
                    if (c_2 < 0) {
                        c_2 = context_lens[q_idx_2 * 3 + 3 - 1];
                    }
                    int num_kv_2 = (c_2 + 64 - 1) / 64;
                    int lo_2 = ((q_atom_6 == e_start_q_1) ? e_start_kv_1 : 0);
                    int hi_2 = ((q_atom_6 == e_claim_q_1) ? e_claim_kv_1 : num_kv_2);
                    mbarrier_wait(q_full_addr + (q_stage_m_1) * 8, _phase_q_full_1);
                    int b_row_94 = b_row_lane_1;
                    int b_cb_94 = b_col_1;
                    unsigned int b_addr_94 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_94 * 128) + (unsigned int)(b_cb_94 ^ (b_row_94 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[0]), "=r"(b_atom_1[1])
                        : "r"(b_addr_94)
                        : "memory");
                    int b_row_94_1 = b_row_lane_1;
                    int b_cb_94_1 = b_col_1 + 32;
                    unsigned int b_addr_94_1 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_94_1 * 128) + (unsigned int)(b_cb_94_1 ^ (b_row_94_1 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[2]), "=r"(b_atom_1[3])
                        : "r"(b_addr_94_1)
                        : "memory");
                    int b_row_95 = b_row_lane_1;
                    int b_cb_95 = b_col_1 + 64;
                    unsigned int b_addr_95 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_95 * 128) + (unsigned int)(b_cb_95 ^ (b_row_95 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[4]), "=r"(b_atom_1[5])
                        : "r"(b_addr_95)
                        : "memory");
                    int b_row_96 = b_row_lane_1;
                    int b_cb_96 = b_col_1 + 96;
                    unsigned int b_addr_96 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_96 * 128) + (unsigned int)(b_cb_96 ^ (b_row_96 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[6]), "=r"(b_atom_1[7])
                        : "r"(b_addr_96)
                        : "memory");
                    int b_row_97 = 8 + b_row_lane_1;
                    int b_cb_97 = b_col_1;
                    unsigned int b_addr_97 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_97 * 128) + (unsigned int)(b_cb_97 ^ (b_row_97 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[8]), "=r"(b_atom_1[9])
                        : "r"(b_addr_97)
                        : "memory");
                    int b_row_98 = 8 + b_row_lane_1;
                    int b_cb_98 = b_col_1 + 32;
                    unsigned int b_addr_98 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_98 * 128) + (unsigned int)(b_cb_98 ^ (b_row_98 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[10]), "=r"(b_atom_1[11])
                        : "r"(b_addr_98)
                        : "memory");
                    int b_row_99 = 8 + b_row_lane_1;
                    int b_cb_99 = b_col_1 + 64;
                    unsigned int b_addr_99 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_99 * 128) + (unsigned int)(b_cb_99 ^ (b_row_99 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[12]), "=r"(b_atom_1[13])
                        : "r"(b_addr_99)
                        : "memory");
                    int b_row_100 = 8 + b_row_lane_1;
                    int b_cb_100 = b_col_1 + 96;
                    unsigned int b_addr_100 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_100 * 128) + (unsigned int)(b_cb_100 ^ (b_row_100 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[14]), "=r"(b_atom_1[15])
                        : "r"(b_addr_100)
                        : "memory");
                    int b_row_101 = 16 + b_row_lane_1;
                    int b_cb_101 = b_col_1;
                    unsigned int b_addr_101 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_101 * 128) + (unsigned int)(b_cb_101 ^ (b_row_101 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[16]), "=r"(b_atom_1[17])
                        : "r"(b_addr_101)
                        : "memory");
                    int b_row_102 = 16 + b_row_lane_1;
                    int b_cb_102 = b_col_1 + 32;
                    unsigned int b_addr_102 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_102 * 128) + (unsigned int)(b_cb_102 ^ (b_row_102 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[18]), "=r"(b_atom_1[19])
                        : "r"(b_addr_102)
                        : "memory");
                    int b_row_103 = 16 + b_row_lane_1;
                    int b_cb_103 = b_col_1 + 64;
                    unsigned int b_addr_103 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_103 * 128) + (unsigned int)(b_cb_103 ^ (b_row_103 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[20]), "=r"(b_atom_1[21])
                        : "r"(b_addr_103)
                        : "memory");
                    int b_row_104 = 16 + b_row_lane_1;
                    int b_cb_104 = b_col_1 + 96;
                    unsigned int b_addr_104 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_104 * 128) + (unsigned int)(b_cb_104 ^ (b_row_104 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[22]), "=r"(b_atom_1[23])
                        : "r"(b_addr_104)
                        : "memory");
                    int b_row_105 = 24 + b_row_lane_1;
                    int b_cb_105 = b_col_1;
                    unsigned int b_addr_105 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_105 * 128) + (unsigned int)(b_cb_105 ^ (b_row_105 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[24]), "=r"(b_atom_1[25])
                        : "r"(b_addr_105)
                        : "memory");
                    int b_row_106 = 24 + b_row_lane_1;
                    int b_cb_106 = b_col_1 + 32;
                    unsigned int b_addr_106 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_106 * 128) + (unsigned int)(b_cb_106 ^ (b_row_106 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[26]), "=r"(b_atom_1[27])
                        : "r"(b_addr_106)
                        : "memory");
                    int b_row_107 = 24 + b_row_lane_1;
                    int b_cb_107 = b_col_1 + 64;
                    unsigned int b_addr_107 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_107 * 128) + (unsigned int)(b_cb_107 ^ (b_row_107 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[28]), "=r"(b_atom_1[29])
                        : "r"(b_addr_107)
                        : "memory");
                    int b_row_108 = 24 + b_row_lane_1;
                    int b_cb_108 = b_col_1 + 96;
                    unsigned int b_addr_108 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_108 * 128) + (unsigned int)(b_cb_108 ^ (b_row_108 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[30]), "=r"(b_atom_1[31])
                        : "r"(b_addr_108)
                        : "memory");
                    int b_row_109 = 32 + b_row_lane_1;
                    int b_cb_109 = b_col_1;
                    unsigned int b_addr_109 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_109 * 128) + (unsigned int)(b_cb_109 ^ (b_row_109 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[32]), "=r"(b_atom_1[33])
                        : "r"(b_addr_109)
                        : "memory");
                    int b_row_110 = 32 + b_row_lane_1;
                    int b_cb_110 = b_col_1 + 32;
                    unsigned int b_addr_110 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_110 * 128) + (unsigned int)(b_cb_110 ^ (b_row_110 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[34]), "=r"(b_atom_1[35])
                        : "r"(b_addr_110)
                        : "memory");
                    int b_row_111 = 32 + b_row_lane_1;
                    int b_cb_111 = b_col_1 + 64;
                    unsigned int b_addr_111 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_111 * 128) + (unsigned int)(b_cb_111 ^ (b_row_111 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[36]), "=r"(b_atom_1[37])
                        : "r"(b_addr_111)
                        : "memory");
                    int b_row_112 = 32 + b_row_lane_1;
                    int b_cb_112 = b_col_1 + 96;
                    unsigned int b_addr_112 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_112 * 128) + (unsigned int)(b_cb_112 ^ (b_row_112 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[38]), "=r"(b_atom_1[39])
                        : "r"(b_addr_112)
                        : "memory");
                    int b_row_113 = 40 + b_row_lane_1;
                    int b_cb_113 = b_col_1;
                    unsigned int b_addr_113 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_113 * 128) + (unsigned int)(b_cb_113 ^ (b_row_113 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[40]), "=r"(b_atom_1[41])
                        : "r"(b_addr_113)
                        : "memory");
                    int b_row_114 = 40 + b_row_lane_1;
                    int b_cb_114 = b_col_1 + 32;
                    unsigned int b_addr_114 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_114 * 128) + (unsigned int)(b_cb_114 ^ (b_row_114 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[42]), "=r"(b_atom_1[43])
                        : "r"(b_addr_114)
                        : "memory");
                    int b_row_115 = 40 + b_row_lane_1;
                    int b_cb_115 = b_col_1 + 64;
                    unsigned int b_addr_115 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_115 * 128) + (unsigned int)(b_cb_115 ^ (b_row_115 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[44]), "=r"(b_atom_1[45])
                        : "r"(b_addr_115)
                        : "memory");
                    int b_row_116 = 40 + b_row_lane_1;
                    int b_cb_116 = b_col_1 + 96;
                    unsigned int b_addr_116 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_116 * 128) + (unsigned int)(b_cb_116 ^ (b_row_116 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[46]), "=r"(b_atom_1[47])
                        : "r"(b_addr_116)
                        : "memory");
                    int b_row_117 = 48 + b_row_lane_1;
                    int b_cb_117 = b_col_1;
                    unsigned int b_addr_117 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_117 * 128) + (unsigned int)(b_cb_117 ^ (b_row_117 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[48]), "=r"(b_atom_1[49])
                        : "r"(b_addr_117)
                        : "memory");
                    int b_row_118 = 48 + b_row_lane_1;
                    int b_cb_118 = b_col_1 + 32;
                    unsigned int b_addr_118 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_118 * 128) + (unsigned int)(b_cb_118 ^ (b_row_118 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[50]), "=r"(b_atom_1[51])
                        : "r"(b_addr_118)
                        : "memory");
                    int b_row_119 = 48 + b_row_lane_1;
                    int b_cb_119 = b_col_1 + 64;
                    unsigned int b_addr_119 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_119 * 128) + (unsigned int)(b_cb_119 ^ (b_row_119 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[52]), "=r"(b_atom_1[53])
                        : "r"(b_addr_119)
                        : "memory");
                    int b_row_120 = 48 + b_row_lane_1;
                    int b_cb_120 = b_col_1 + 96;
                    unsigned int b_addr_120 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_120 * 128) + (unsigned int)(b_cb_120 ^ (b_row_120 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[54]), "=r"(b_atom_1[55])
                        : "r"(b_addr_120)
                        : "memory");
                    int b_row_121 = 56 + b_row_lane_1;
                    int b_cb_121 = b_col_1;
                    unsigned int b_addr_121 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_121 * 128) + (unsigned int)(b_cb_121 ^ (b_row_121 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[56]), "=r"(b_atom_1[57])
                        : "r"(b_addr_121)
                        : "memory");
                    int b_row_122 = 56 + b_row_lane_1;
                    int b_cb_122 = b_col_1 + 32;
                    unsigned int b_addr_122 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_122 * 128) + (unsigned int)(b_cb_122 ^ (b_row_122 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[58]), "=r"(b_atom_1[59])
                        : "r"(b_addr_122)
                        : "memory");
                    int b_row_123 = 56 + b_row_lane_1;
                    int b_cb_123 = b_col_1 + 64;
                    unsigned int b_addr_123 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_123 * 128) + (unsigned int)(b_cb_123 ^ (b_row_123 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[60]), "=r"(b_atom_1[61])
                        : "r"(b_addr_123)
                        : "memory");
                    int b_row_124 = 56 + b_row_lane_1;
                    int b_cb_124 = b_col_1 + 96;
                    unsigned int b_addr_124 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_124 * 128) + (unsigned int)(b_cb_124 ^ (b_row_124 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[62]), "=r"(b_atom_1[63])
                        : "r"(b_addr_124)
                        : "memory");
                    int b_row_125 = 64 + b_row_lane_1;
                    int b_cb_125 = b_col_1;
                    unsigned int b_addr_125 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_125 * 128) + (unsigned int)(b_cb_125 ^ (b_row_125 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[64]), "=r"(b_atom_1[65])
                        : "r"(b_addr_125)
                        : "memory");
                    int b_row_126 = 64 + b_row_lane_1;
                    int b_cb_126 = b_col_1 + 32;
                    unsigned int b_addr_126 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_126 * 128) + (unsigned int)(b_cb_126 ^ (b_row_126 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[66]), "=r"(b_atom_1[67])
                        : "r"(b_addr_126)
                        : "memory");
                    int b_row_127 = 64 + b_row_lane_1;
                    int b_cb_127 = b_col_1 + 64;
                    unsigned int b_addr_127 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_127 * 128) + (unsigned int)(b_cb_127 ^ (b_row_127 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[68]), "=r"(b_atom_1[69])
                        : "r"(b_addr_127)
                        : "memory");
                    int b_row_128 = 64 + b_row_lane_1;
                    int b_cb_128 = b_col_1 + 96;
                    unsigned int b_addr_128 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_128 * 128) + (unsigned int)(b_cb_128 ^ (b_row_128 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[70]), "=r"(b_atom_1[71])
                        : "r"(b_addr_128)
                        : "memory");
                    int b_row_129 = 72 + b_row_lane_1;
                    int b_cb_129 = b_col_1;
                    unsigned int b_addr_129 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_129 * 128) + (unsigned int)(b_cb_129 ^ (b_row_129 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[72]), "=r"(b_atom_1[73])
                        : "r"(b_addr_129)
                        : "memory");
                    int b_row_130 = 72 + b_row_lane_1;
                    int b_cb_130 = b_col_1 + 32;
                    unsigned int b_addr_130 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_130 * 128) + (unsigned int)(b_cb_130 ^ (b_row_130 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[74]), "=r"(b_atom_1[75])
                        : "r"(b_addr_130)
                        : "memory");
                    int b_row_131 = 72 + b_row_lane_1;
                    int b_cb_131 = b_col_1 + 64;
                    unsigned int b_addr_131 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_131 * 128) + (unsigned int)(b_cb_131 ^ (b_row_131 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[76]), "=r"(b_atom_1[77])
                        : "r"(b_addr_131)
                        : "memory");
                    int b_row_132 = 72 + b_row_lane_1;
                    int b_cb_132 = b_col_1 + 96;
                    unsigned int b_addr_132 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_132 * 128) + (unsigned int)(b_cb_132 ^ (b_row_132 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[78]), "=r"(b_atom_1[79])
                        : "r"(b_addr_132)
                        : "memory");
                    int b_row_133 = 80 + b_row_lane_1;
                    int b_cb_133 = b_col_1;
                    unsigned int b_addr_133 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_133 * 128) + (unsigned int)(b_cb_133 ^ (b_row_133 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[80]), "=r"(b_atom_1[81])
                        : "r"(b_addr_133)
                        : "memory");
                    int b_row_134 = 80 + b_row_lane_1;
                    int b_cb_134 = b_col_1 + 32;
                    unsigned int b_addr_134 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_134 * 128) + (unsigned int)(b_cb_134 ^ (b_row_134 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[82]), "=r"(b_atom_1[83])
                        : "r"(b_addr_134)
                        : "memory");
                    int b_row_135 = 80 + b_row_lane_1;
                    int b_cb_135 = b_col_1 + 64;
                    unsigned int b_addr_135 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_135 * 128) + (unsigned int)(b_cb_135 ^ (b_row_135 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[84]), "=r"(b_atom_1[85])
                        : "r"(b_addr_135)
                        : "memory");
                    int b_row_136 = 80 + b_row_lane_1;
                    int b_cb_136 = b_col_1 + 96;
                    unsigned int b_addr_136 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_136 * 128) + (unsigned int)(b_cb_136 ^ (b_row_136 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[86]), "=r"(b_atom_1[87])
                        : "r"(b_addr_136)
                        : "memory");
                    int b_row_137 = 88 + b_row_lane_1;
                    int b_cb_137 = b_col_1;
                    unsigned int b_addr_137 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_137 * 128) + (unsigned int)(b_cb_137 ^ (b_row_137 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[88]), "=r"(b_atom_1[89])
                        : "r"(b_addr_137)
                        : "memory");
                    int b_row_138 = 88 + b_row_lane_1;
                    int b_cb_138 = b_col_1 + 32;
                    unsigned int b_addr_138 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_138 * 128) + (unsigned int)(b_cb_138 ^ (b_row_138 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[90]), "=r"(b_atom_1[91])
                        : "r"(b_addr_138)
                        : "memory");
                    int b_row_139 = 88 + b_row_lane_1;
                    int b_cb_139 = b_col_1 + 64;
                    unsigned int b_addr_139 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_139 * 128) + (unsigned int)(b_cb_139 ^ (b_row_139 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[92]), "=r"(b_atom_1[93])
                        : "r"(b_addr_139)
                        : "memory");
                    int b_row_140 = 88 + b_row_lane_1;
                    int b_cb_140 = b_col_1 + 96;
                    unsigned int b_addr_140 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_140 * 128) + (unsigned int)(b_cb_140 ^ (b_row_140 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[94]), "=r"(b_atom_1[95])
                        : "r"(b_addr_140)
                        : "memory");
                    int n_tok_m_2 = 3;
                    #pragma unroll 1
                    for (int kv_idx_2 = lo_2; kv_idx_2 < hi_2; kv_idx_2 += 2) {
                        mbarrier_wait(kv_full_g1_addr + (kv_stage_m_1) * 8, _phase_kv_full_g1);
                        int sc_idx_2 = kv_stage_m_1 * 256 + (unsigned int)(w_in_g_1 * 16) + (unsigned int)quad_1;
                        float scale0_2 = smem_sc_g1[sc_idx_2];
                        float scale1_2 = smem_sc_g1[sc_idx_2 + 8];
                        int kv_offset_2 = tok_2 * logits_stride + (kv_idx_2 + 1) * 64 + w_in_g_1 * 16;
                        int a_cb_6 = a_col_1;
                        unsigned int a_addr_6 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_6 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_tile_1[0]), "=r"(a_tile_1[1]), "=r"(a_tile_1[2]), "=r"(a_tile_1[3])
                            : "r"(a_addr_6)
                            : "memory");
                        int a_cb_6_1 = a_col_1 + 32;
                        unsigned int a_addr_6_1 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_6_1 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_tile_1[4]), "=r"(a_tile_1[5]), "=r"(a_tile_1[6]), "=r"(a_tile_1[7])
                            : "r"(a_addr_6_1)
                            : "memory");
                        int a_cb_7 = a_col_1 + 64;
                        unsigned int a_addr_7 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_7 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_tile_1[8]), "=r"(a_tile_1[9]), "=r"(a_tile_1[10]), "=r"(a_tile_1[11])
                            : "r"(a_addr_7)
                            : "memory");
                        int a_cb_8 = a_col_1 + 96;
                        unsigned int a_addr_8 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_8 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_tile_1[12]), "=r"(a_tile_1[13]), "=r"(a_tile_1[14]), "=r"(a_tile_1[15])
                            : "r"(a_addr_8)
                            : "memory");
                        float partial0_4 = 0.0f;
                        float partial1_4 = 0.0f;
                        #pragma unroll
                        for (int nt_6 = 0; nt_6 < 4; nt_6++) {
                            acc_1[0] = 0.0f;
                            acc_1[1] = 0.0f;
                            acc_1[2] = 0.0f;
                            acc_1[3] = 0.0f;
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                                : "r"(a_tile_1[0]), "r"(a_tile_1[1]), "r"(a_tile_1[2]), "r"(a_tile_1[3]), "r"((b_atom_1 + nt_6 * 4 * 2)[0]), "r"((b_atom_1 + nt_6 * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[4]), "r"(a_tile_1[(4) + 1]), "r"(a_tile_1[(4) + 2]), "r"(a_tile_1[(4) + 3]), "r"((b_atom_1 + (nt_6 * 4 + 1) * 2)[0]), "r"((b_atom_1 + (nt_6 * 4 + 1) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[8]), "r"(a_tile_1[(8) + 1]), "r"(a_tile_1[(8) + 2]), "r"(a_tile_1[(8) + 3]), "r"((b_atom_1 + (nt_6 * 4 + 2) * 2)[0]), "r"((b_atom_1 + (nt_6 * 4 + 2) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[12]), "r"(a_tile_1[(12) + 1]), "r"(a_tile_1[(12) + 2]), "r"(a_tile_1[(12) + 3]), "r"((b_atom_1 + (nt_6 * 4 + 3) * 2)[0]), "r"((b_atom_1 + (nt_6 * 4 + 3) * 2)[1]));
                            int w_idx_6 = q_stage_m_1 * 256 + (unsigned int)(nt_6 * 8) + (unsigned int)(tq_1 * 2);
                            float w0_6 = smem_w[w_idx_6];
                            float w1_6 = smem_w[w_idx_6 + 1];
                            float r0_6 = fmaxf(acc_1[0], 0.0f);
                            float r1_6 = fmaxf(acc_1[1], 0.0f);
                            float r2_6 = fmaxf(acc_1[2], 0.0f);
                            float r3_6 = fmaxf(acc_1[3], 0.0f);
                            partial0_4 += __fmaf_rn(r0_6, w0_6, r1_6 * w1_6);
                            partial1_4 += __fmaf_rn(r2_6, w0_6, r3_6 * w1_6);
                        }
                        float v0_4 = partial0_4 * scale0_2;
                        float v1_4 = partial1_4 * scale1_2;
                        float _shfl_xor_sync_0_1 = __shfl_xor_sync(0xFFFFFFFF, v0_4, 1);
                        v0_4 = v0_4 + _shfl_xor_sync_0_1;
                        float _shfl_xor_sync_1_1 = __shfl_xor_sync(0xFFFFFFFF, v1_4, 1);
                        v1_4 = v1_4 + _shfl_xor_sync_1_1;
                        float _shfl_xor_sync_2_1 = __shfl_xor_sync(0xFFFFFFFF, v0_4, 2);
                        v0_4 = v0_4 + _shfl_xor_sync_2_1;
                        float _shfl_xor_sync_3_1 = __shfl_xor_sync(0xFFFFFFFF, v1_4, 2);
                        v1_4 = v1_4 + _shfl_xor_sync_3_1;
                        int out0_4 = kv_offset_2 + quad_1;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_4)), "f"(v0_4), "r"((unsigned int)(tq_1 == 0)));
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_4 + 8))), "f"(v1_4), "r"((unsigned int)(tq_1 == 0)));
                        float partial0_4_1 = 0.0f;
                        float partial1_4_1 = 0.0f;
                        #pragma unroll
                        for (int nt_7 = 0; nt_7 < 4; nt_7++) {
                            acc_1[0] = 0.0f;
                            acc_1[1] = 0.0f;
                            acc_1[2] = 0.0f;
                            acc_1[3] = 0.0f;
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                                : "r"(a_tile_1[0]), "r"(a_tile_1[1]), "r"(a_tile_1[2]), "r"(a_tile_1[3]), "r"((b_atom_1 + (4 + nt_7) * 4 * 2)[0]), "r"((b_atom_1 + (4 + nt_7) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[4]), "r"(a_tile_1[(4) + 1]), "r"(a_tile_1[(4) + 2]), "r"(a_tile_1[(4) + 3]), "r"((b_atom_1 + ((4 + nt_7) * 4 + 1) * 2)[0]), "r"((b_atom_1 + ((4 + nt_7) * 4 + 1) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[8]), "r"(a_tile_1[(8) + 1]), "r"(a_tile_1[(8) + 2]), "r"(a_tile_1[(8) + 3]), "r"((b_atom_1 + ((4 + nt_7) * 4 + 2) * 2)[0]), "r"((b_atom_1 + ((4 + nt_7) * 4 + 2) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[12]), "r"(a_tile_1[(12) + 1]), "r"(a_tile_1[(12) + 2]), "r"(a_tile_1[(12) + 3]), "r"((b_atom_1 + ((4 + nt_7) * 4 + 3) * 2)[0]), "r"((b_atom_1 + ((4 + nt_7) * 4 + 3) * 2)[1]));
                            int w_idx_7 = q_stage_m_1 * 256 + 32 + (unsigned int)(nt_7 * 8) + (unsigned int)(tq_1 * 2);
                            float w0_7 = smem_w[w_idx_7];
                            float w1_7 = smem_w[w_idx_7 + 1];
                            float r0_7 = fmaxf(acc_1[0], 0.0f);
                            float r1_7 = fmaxf(acc_1[1], 0.0f);
                            float r2_7 = fmaxf(acc_1[2], 0.0f);
                            float r3_7 = fmaxf(acc_1[3], 0.0f);
                            partial0_4_1 += __fmaf_rn(r0_7, w0_7, r1_7 * w1_7);
                            partial1_4_1 += __fmaf_rn(r2_7, w0_7, r3_7 * w1_7);
                        }
                        float v0_4_1 = partial0_4_1 * scale0_2;
                        float v1_4_1 = partial1_4_1 * scale1_2;
                        float _shfl_xor_sync_4_1 = __shfl_xor_sync(0xFFFFFFFF, v0_4_1, 1);
                        v0_4_1 = v0_4_1 + _shfl_xor_sync_4_1;
                        float _shfl_xor_sync_5_1 = __shfl_xor_sync(0xFFFFFFFF, v1_4_1, 1);
                        v1_4_1 = v1_4_1 + _shfl_xor_sync_5_1;
                        float _shfl_xor_sync_6_1 = __shfl_xor_sync(0xFFFFFFFF, v0_4_1, 2);
                        v0_4_1 = v0_4_1 + _shfl_xor_sync_6_1;
                        float _shfl_xor_sync_7_1 = __shfl_xor_sync(0xFFFFFFFF, v1_4_1, 2);
                        v1_4_1 = v1_4_1 + _shfl_xor_sync_7_1;
                        int out0_4_1 = kv_offset_2 + logits_stride + quad_1;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_4_1)), "f"(v0_4_1), "r"((unsigned int)(tq_1 == 0)));
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_4_1 + 8))), "f"(v1_4_1), "r"((unsigned int)(tq_1 == 0)));
                        float partial0_5 = 0.0f;
                        float partial1_5 = 0.0f;
                        #pragma unroll
                        for (int nt_8 = 0; nt_8 < 4; nt_8++) {
                            acc_1[0] = 0.0f;
                            acc_1[1] = 0.0f;
                            acc_1[2] = 0.0f;
                            acc_1[3] = 0.0f;
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                                : "r"(a_tile_1[0]), "r"(a_tile_1[1]), "r"(a_tile_1[2]), "r"(a_tile_1[3]), "r"((b_atom_1 + (8 + nt_8) * 4 * 2)[0]), "r"((b_atom_1 + (8 + nt_8) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[4]), "r"(a_tile_1[(4) + 1]), "r"(a_tile_1[(4) + 2]), "r"(a_tile_1[(4) + 3]), "r"((b_atom_1 + ((8 + nt_8) * 4 + 1) * 2)[0]), "r"((b_atom_1 + ((8 + nt_8) * 4 + 1) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[8]), "r"(a_tile_1[(8) + 1]), "r"(a_tile_1[(8) + 2]), "r"(a_tile_1[(8) + 3]), "r"((b_atom_1 + ((8 + nt_8) * 4 + 2) * 2)[0]), "r"((b_atom_1 + ((8 + nt_8) * 4 + 2) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[12]), "r"(a_tile_1[(12) + 1]), "r"(a_tile_1[(12) + 2]), "r"(a_tile_1[(12) + 3]), "r"((b_atom_1 + ((8 + nt_8) * 4 + 3) * 2)[0]), "r"((b_atom_1 + ((8 + nt_8) * 4 + 3) * 2)[1]));
                            int w_idx_8 = q_stage_m_1 * 256 + 64 + (unsigned int)(nt_8 * 8) + (unsigned int)(tq_1 * 2);
                            float w0_8 = smem_w[w_idx_8];
                            float w1_8 = smem_w[w_idx_8 + 1];
                            float r0_8 = fmaxf(acc_1[0], 0.0f);
                            float r1_8 = fmaxf(acc_1[1], 0.0f);
                            float r2_8 = fmaxf(acc_1[2], 0.0f);
                            float r3_8 = fmaxf(acc_1[3], 0.0f);
                            partial0_5 += __fmaf_rn(r0_8, w0_8, r1_8 * w1_8);
                            partial1_5 += __fmaf_rn(r2_8, w0_8, r3_8 * w1_8);
                        }
                        float v0_5 = partial0_5 * scale0_2;
                        float v1_5 = partial1_5 * scale1_2;
                        float _shfl_xor_sync_8_1 = __shfl_xor_sync(0xFFFFFFFF, v0_5, 1);
                        v0_5 = v0_5 + _shfl_xor_sync_8_1;
                        float _shfl_xor_sync_9_1 = __shfl_xor_sync(0xFFFFFFFF, v1_5, 1);
                        v1_5 = v1_5 + _shfl_xor_sync_9_1;
                        float _shfl_xor_sync_10_1 = __shfl_xor_sync(0xFFFFFFFF, v0_5, 2);
                        v0_5 = v0_5 + _shfl_xor_sync_10_1;
                        float _shfl_xor_sync_11_1 = __shfl_xor_sync(0xFFFFFFFF, v1_5, 2);
                        v1_5 = v1_5 + _shfl_xor_sync_11_1;
                        int out0_5 = kv_offset_2 + 2 * logits_stride + quad_1;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_5)), "f"(v0_5), "r"((unsigned int)(tq_1 == 0)));
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_5 + 8))), "f"(v1_5), "r"((unsigned int)(tq_1 == 0)));
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
                #pragma unroll 1
                for (int q_atom_7 = e_claim_q_1; q_atom_7 < e_q_stop_1; q_atom_7++) {
                    int q_idx_3 = q_atom_7;
                    int atom_in_q_3 = 0;
                    int tok_3 = q_idx_3 * 3 + atom_in_q_3 * 3;
                    int c_3 = -1;
                    if (q_atom_7 == e_claim_q_1) {
                        c_3 = e_claim_ctx_1;
                    }
                    if (c_3 < 0) {
                        c_3 = context_lens[q_idx_3 * 3 + 3 - 1];
                    }
                    int num_kv_3 = (c_3 + 64 - 1) / 64;
                    int lo_3 = ((q_atom_7 == e_claim_q_1) ? e_claim_kv_1 : 0);
                    int hi_3 = ((q_atom_7 == e_end_q_1) ? e_end_kv_1 : num_kv_3);
                    mbarrier_wait(q_full_addr + (q_stage_m_1) * 8, _phase_q_full_1);
                    int b_row_141 = b_row_lane_1;
                    int b_cb_141 = b_col_1;
                    unsigned int b_addr_141 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_141 * 128) + (unsigned int)(b_cb_141 ^ (b_row_141 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[0]), "=r"(b_atom_1[1])
                        : "r"(b_addr_141)
                        : "memory");
                    int b_row_141_1 = b_row_lane_1;
                    int b_cb_141_1 = b_col_1 + 32;
                    unsigned int b_addr_141_1 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_141_1 * 128) + (unsigned int)(b_cb_141_1 ^ (b_row_141_1 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[2]), "=r"(b_atom_1[3])
                        : "r"(b_addr_141_1)
                        : "memory");
                    int b_row_142 = b_row_lane_1;
                    int b_cb_142 = b_col_1 + 64;
                    unsigned int b_addr_142 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_142 * 128) + (unsigned int)(b_cb_142 ^ (b_row_142 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[4]), "=r"(b_atom_1[5])
                        : "r"(b_addr_142)
                        : "memory");
                    int b_row_143 = b_row_lane_1;
                    int b_cb_143 = b_col_1 + 96;
                    unsigned int b_addr_143 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_143 * 128) + (unsigned int)(b_cb_143 ^ (b_row_143 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[6]), "=r"(b_atom_1[7])
                        : "r"(b_addr_143)
                        : "memory");
                    int b_row_144 = 8 + b_row_lane_1;
                    int b_cb_144 = b_col_1;
                    unsigned int b_addr_144 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_144 * 128) + (unsigned int)(b_cb_144 ^ (b_row_144 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[8]), "=r"(b_atom_1[9])
                        : "r"(b_addr_144)
                        : "memory");
                    int b_row_145 = 8 + b_row_lane_1;
                    int b_cb_145 = b_col_1 + 32;
                    unsigned int b_addr_145 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_145 * 128) + (unsigned int)(b_cb_145 ^ (b_row_145 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[10]), "=r"(b_atom_1[11])
                        : "r"(b_addr_145)
                        : "memory");
                    int b_row_146 = 8 + b_row_lane_1;
                    int b_cb_146 = b_col_1 + 64;
                    unsigned int b_addr_146 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_146 * 128) + (unsigned int)(b_cb_146 ^ (b_row_146 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[12]), "=r"(b_atom_1[13])
                        : "r"(b_addr_146)
                        : "memory");
                    int b_row_147 = 8 + b_row_lane_1;
                    int b_cb_147 = b_col_1 + 96;
                    unsigned int b_addr_147 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_147 * 128) + (unsigned int)(b_cb_147 ^ (b_row_147 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[14]), "=r"(b_atom_1[15])
                        : "r"(b_addr_147)
                        : "memory");
                    int b_row_148 = 16 + b_row_lane_1;
                    int b_cb_148 = b_col_1;
                    unsigned int b_addr_148 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_148 * 128) + (unsigned int)(b_cb_148 ^ (b_row_148 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[16]), "=r"(b_atom_1[17])
                        : "r"(b_addr_148)
                        : "memory");
                    int b_row_149 = 16 + b_row_lane_1;
                    int b_cb_149 = b_col_1 + 32;
                    unsigned int b_addr_149 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_149 * 128) + (unsigned int)(b_cb_149 ^ (b_row_149 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[18]), "=r"(b_atom_1[19])
                        : "r"(b_addr_149)
                        : "memory");
                    int b_row_150 = 16 + b_row_lane_1;
                    int b_cb_150 = b_col_1 + 64;
                    unsigned int b_addr_150 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_150 * 128) + (unsigned int)(b_cb_150 ^ (b_row_150 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[20]), "=r"(b_atom_1[21])
                        : "r"(b_addr_150)
                        : "memory");
                    int b_row_151 = 16 + b_row_lane_1;
                    int b_cb_151 = b_col_1 + 96;
                    unsigned int b_addr_151 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_151 * 128) + (unsigned int)(b_cb_151 ^ (b_row_151 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[22]), "=r"(b_atom_1[23])
                        : "r"(b_addr_151)
                        : "memory");
                    int b_row_152 = 24 + b_row_lane_1;
                    int b_cb_152 = b_col_1;
                    unsigned int b_addr_152 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_152 * 128) + (unsigned int)(b_cb_152 ^ (b_row_152 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[24]), "=r"(b_atom_1[25])
                        : "r"(b_addr_152)
                        : "memory");
                    int b_row_153 = 24 + b_row_lane_1;
                    int b_cb_153 = b_col_1 + 32;
                    unsigned int b_addr_153 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_153 * 128) + (unsigned int)(b_cb_153 ^ (b_row_153 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[26]), "=r"(b_atom_1[27])
                        : "r"(b_addr_153)
                        : "memory");
                    int b_row_154 = 24 + b_row_lane_1;
                    int b_cb_154 = b_col_1 + 64;
                    unsigned int b_addr_154 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_154 * 128) + (unsigned int)(b_cb_154 ^ (b_row_154 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[28]), "=r"(b_atom_1[29])
                        : "r"(b_addr_154)
                        : "memory");
                    int b_row_155 = 24 + b_row_lane_1;
                    int b_cb_155 = b_col_1 + 96;
                    unsigned int b_addr_155 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_155 * 128) + (unsigned int)(b_cb_155 ^ (b_row_155 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[30]), "=r"(b_atom_1[31])
                        : "r"(b_addr_155)
                        : "memory");
                    int b_row_156 = 32 + b_row_lane_1;
                    int b_cb_156 = b_col_1;
                    unsigned int b_addr_156 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_156 * 128) + (unsigned int)(b_cb_156 ^ (b_row_156 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[32]), "=r"(b_atom_1[33])
                        : "r"(b_addr_156)
                        : "memory");
                    int b_row_157 = 32 + b_row_lane_1;
                    int b_cb_157 = b_col_1 + 32;
                    unsigned int b_addr_157 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_157 * 128) + (unsigned int)(b_cb_157 ^ (b_row_157 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[34]), "=r"(b_atom_1[35])
                        : "r"(b_addr_157)
                        : "memory");
                    int b_row_158 = 32 + b_row_lane_1;
                    int b_cb_158 = b_col_1 + 64;
                    unsigned int b_addr_158 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_158 * 128) + (unsigned int)(b_cb_158 ^ (b_row_158 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[36]), "=r"(b_atom_1[37])
                        : "r"(b_addr_158)
                        : "memory");
                    int b_row_159 = 32 + b_row_lane_1;
                    int b_cb_159 = b_col_1 + 96;
                    unsigned int b_addr_159 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_159 * 128) + (unsigned int)(b_cb_159 ^ (b_row_159 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[38]), "=r"(b_atom_1[39])
                        : "r"(b_addr_159)
                        : "memory");
                    int b_row_160 = 40 + b_row_lane_1;
                    int b_cb_160 = b_col_1;
                    unsigned int b_addr_160 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_160 * 128) + (unsigned int)(b_cb_160 ^ (b_row_160 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[40]), "=r"(b_atom_1[41])
                        : "r"(b_addr_160)
                        : "memory");
                    int b_row_161 = 40 + b_row_lane_1;
                    int b_cb_161 = b_col_1 + 32;
                    unsigned int b_addr_161 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_161 * 128) + (unsigned int)(b_cb_161 ^ (b_row_161 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[42]), "=r"(b_atom_1[43])
                        : "r"(b_addr_161)
                        : "memory");
                    int b_row_162 = 40 + b_row_lane_1;
                    int b_cb_162 = b_col_1 + 64;
                    unsigned int b_addr_162 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_162 * 128) + (unsigned int)(b_cb_162 ^ (b_row_162 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[44]), "=r"(b_atom_1[45])
                        : "r"(b_addr_162)
                        : "memory");
                    int b_row_163 = 40 + b_row_lane_1;
                    int b_cb_163 = b_col_1 + 96;
                    unsigned int b_addr_163 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_163 * 128) + (unsigned int)(b_cb_163 ^ (b_row_163 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[46]), "=r"(b_atom_1[47])
                        : "r"(b_addr_163)
                        : "memory");
                    int b_row_164 = 48 + b_row_lane_1;
                    int b_cb_164 = b_col_1;
                    unsigned int b_addr_164 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_164 * 128) + (unsigned int)(b_cb_164 ^ (b_row_164 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[48]), "=r"(b_atom_1[49])
                        : "r"(b_addr_164)
                        : "memory");
                    int b_row_165 = 48 + b_row_lane_1;
                    int b_cb_165 = b_col_1 + 32;
                    unsigned int b_addr_165 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_165 * 128) + (unsigned int)(b_cb_165 ^ (b_row_165 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[50]), "=r"(b_atom_1[51])
                        : "r"(b_addr_165)
                        : "memory");
                    int b_row_166 = 48 + b_row_lane_1;
                    int b_cb_166 = b_col_1 + 64;
                    unsigned int b_addr_166 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_166 * 128) + (unsigned int)(b_cb_166 ^ (b_row_166 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[52]), "=r"(b_atom_1[53])
                        : "r"(b_addr_166)
                        : "memory");
                    int b_row_167 = 48 + b_row_lane_1;
                    int b_cb_167 = b_col_1 + 96;
                    unsigned int b_addr_167 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_167 * 128) + (unsigned int)(b_cb_167 ^ (b_row_167 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[54]), "=r"(b_atom_1[55])
                        : "r"(b_addr_167)
                        : "memory");
                    int b_row_168 = 56 + b_row_lane_1;
                    int b_cb_168 = b_col_1;
                    unsigned int b_addr_168 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_168 * 128) + (unsigned int)(b_cb_168 ^ (b_row_168 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[56]), "=r"(b_atom_1[57])
                        : "r"(b_addr_168)
                        : "memory");
                    int b_row_169 = 56 + b_row_lane_1;
                    int b_cb_169 = b_col_1 + 32;
                    unsigned int b_addr_169 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_169 * 128) + (unsigned int)(b_cb_169 ^ (b_row_169 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[58]), "=r"(b_atom_1[59])
                        : "r"(b_addr_169)
                        : "memory");
                    int b_row_170 = 56 + b_row_lane_1;
                    int b_cb_170 = b_col_1 + 64;
                    unsigned int b_addr_170 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_170 * 128) + (unsigned int)(b_cb_170 ^ (b_row_170 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[60]), "=r"(b_atom_1[61])
                        : "r"(b_addr_170)
                        : "memory");
                    int b_row_171 = 56 + b_row_lane_1;
                    int b_cb_171 = b_col_1 + 96;
                    unsigned int b_addr_171 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_171 * 128) + (unsigned int)(b_cb_171 ^ (b_row_171 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[62]), "=r"(b_atom_1[63])
                        : "r"(b_addr_171)
                        : "memory");
                    int b_row_172 = 64 + b_row_lane_1;
                    int b_cb_172 = b_col_1;
                    unsigned int b_addr_172 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_172 * 128) + (unsigned int)(b_cb_172 ^ (b_row_172 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[64]), "=r"(b_atom_1[65])
                        : "r"(b_addr_172)
                        : "memory");
                    int b_row_173 = 64 + b_row_lane_1;
                    int b_cb_173 = b_col_1 + 32;
                    unsigned int b_addr_173 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_173 * 128) + (unsigned int)(b_cb_173 ^ (b_row_173 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[66]), "=r"(b_atom_1[67])
                        : "r"(b_addr_173)
                        : "memory");
                    int b_row_174 = 64 + b_row_lane_1;
                    int b_cb_174 = b_col_1 + 64;
                    unsigned int b_addr_174 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_174 * 128) + (unsigned int)(b_cb_174 ^ (b_row_174 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[68]), "=r"(b_atom_1[69])
                        : "r"(b_addr_174)
                        : "memory");
                    int b_row_175 = 64 + b_row_lane_1;
                    int b_cb_175 = b_col_1 + 96;
                    unsigned int b_addr_175 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_175 * 128) + (unsigned int)(b_cb_175 ^ (b_row_175 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[70]), "=r"(b_atom_1[71])
                        : "r"(b_addr_175)
                        : "memory");
                    int b_row_176 = 72 + b_row_lane_1;
                    int b_cb_176 = b_col_1;
                    unsigned int b_addr_176 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_176 * 128) + (unsigned int)(b_cb_176 ^ (b_row_176 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[72]), "=r"(b_atom_1[73])
                        : "r"(b_addr_176)
                        : "memory");
                    int b_row_177 = 72 + b_row_lane_1;
                    int b_cb_177 = b_col_1 + 32;
                    unsigned int b_addr_177 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_177 * 128) + (unsigned int)(b_cb_177 ^ (b_row_177 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[74]), "=r"(b_atom_1[75])
                        : "r"(b_addr_177)
                        : "memory");
                    int b_row_178 = 72 + b_row_lane_1;
                    int b_cb_178 = b_col_1 + 64;
                    unsigned int b_addr_178 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_178 * 128) + (unsigned int)(b_cb_178 ^ (b_row_178 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[76]), "=r"(b_atom_1[77])
                        : "r"(b_addr_178)
                        : "memory");
                    int b_row_179 = 72 + b_row_lane_1;
                    int b_cb_179 = b_col_1 + 96;
                    unsigned int b_addr_179 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_179 * 128) + (unsigned int)(b_cb_179 ^ (b_row_179 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[78]), "=r"(b_atom_1[79])
                        : "r"(b_addr_179)
                        : "memory");
                    int b_row_180 = 80 + b_row_lane_1;
                    int b_cb_180 = b_col_1;
                    unsigned int b_addr_180 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_180 * 128) + (unsigned int)(b_cb_180 ^ (b_row_180 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[80]), "=r"(b_atom_1[81])
                        : "r"(b_addr_180)
                        : "memory");
                    int b_row_181 = 80 + b_row_lane_1;
                    int b_cb_181 = b_col_1 + 32;
                    unsigned int b_addr_181 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_181 * 128) + (unsigned int)(b_cb_181 ^ (b_row_181 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[82]), "=r"(b_atom_1[83])
                        : "r"(b_addr_181)
                        : "memory");
                    int b_row_182 = 80 + b_row_lane_1;
                    int b_cb_182 = b_col_1 + 64;
                    unsigned int b_addr_182 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_182 * 128) + (unsigned int)(b_cb_182 ^ (b_row_182 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[84]), "=r"(b_atom_1[85])
                        : "r"(b_addr_182)
                        : "memory");
                    int b_row_183 = 80 + b_row_lane_1;
                    int b_cb_183 = b_col_1 + 96;
                    unsigned int b_addr_183 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_183 * 128) + (unsigned int)(b_cb_183 ^ (b_row_183 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[86]), "=r"(b_atom_1[87])
                        : "r"(b_addr_183)
                        : "memory");
                    int b_row_184 = 88 + b_row_lane_1;
                    int b_cb_184 = b_col_1;
                    unsigned int b_addr_184 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_184 * 128) + (unsigned int)(b_cb_184 ^ (b_row_184 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[88]), "=r"(b_atom_1[89])
                        : "r"(b_addr_184)
                        : "memory");
                    int b_row_185 = 88 + b_row_lane_1;
                    int b_cb_185 = b_col_1 + 32;
                    unsigned int b_addr_185 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_185 * 128) + (unsigned int)(b_cb_185 ^ (b_row_185 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[90]), "=r"(b_atom_1[91])
                        : "r"(b_addr_185)
                        : "memory");
                    int b_row_186 = 88 + b_row_lane_1;
                    int b_cb_186 = b_col_1 + 64;
                    unsigned int b_addr_186 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_186 * 128) + (unsigned int)(b_cb_186 ^ (b_row_186 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[92]), "=r"(b_atom_1[93])
                        : "r"(b_addr_186)
                        : "memory");
                    int b_row_187 = 88 + b_row_lane_1;
                    int b_cb_187 = b_col_1 + 96;
                    unsigned int b_addr_187 = smem_q_addr + q_stage_m_1 * 12288 + (unsigned int)(b_row_187 * 128) + (unsigned int)(b_cb_187 ^ (b_row_187 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_atom_1[94]), "=r"(b_atom_1[95])
                        : "r"(b_addr_187)
                        : "memory");
                    int n_tok_m_3 = 3;
                    #pragma unroll 1
                    for (int kv_idx_3 = lo_3; kv_idx_3 < hi_3; kv_idx_3 += 2) {
                        mbarrier_wait(kv_full_g1_addr + (kv_stage_m_1) * 8, _phase_kv_full_g1);
                        int sc_idx_3 = kv_stage_m_1 * 256 + (unsigned int)(w_in_g_1 * 16) + (unsigned int)quad_1;
                        float scale0_3 = smem_sc_g1[sc_idx_3];
                        float scale1_3 = smem_sc_g1[sc_idx_3 + 8];
                        int kv_offset_3 = tok_3 * logits_stride + (kv_idx_3 + 1) * 64 + w_in_g_1 * 16;
                        int a_cb_9 = a_col_1;
                        unsigned int a_addr_9 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_9 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_tile_1[0]), "=r"(a_tile_1[1]), "=r"(a_tile_1[2]), "=r"(a_tile_1[3])
                            : "r"(a_addr_9)
                            : "memory");
                        int a_cb_9_1 = a_col_1 + 32;
                        unsigned int a_addr_9_1 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_9_1 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_tile_1[4]), "=r"(a_tile_1[5]), "=r"(a_tile_1[6]), "=r"(a_tile_1[7])
                            : "r"(a_addr_9_1)
                            : "memory");
                        int a_cb_10 = a_col_1 + 64;
                        unsigned int a_addr_10 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_10 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_tile_1[8]), "=r"(a_tile_1[9]), "=r"(a_tile_1[10]), "=r"(a_tile_1[11])
                            : "r"(a_addr_10)
                            : "memory");
                        int a_cb_11 = a_col_1 + 96;
                        unsigned int a_addr_11 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_11 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_tile_1[12]), "=r"(a_tile_1[13]), "=r"(a_tile_1[14]), "=r"(a_tile_1[15])
                            : "r"(a_addr_11)
                            : "memory");
                        float partial0_6 = 0.0f;
                        float partial1_6 = 0.0f;
                        #pragma unroll
                        for (int nt_9 = 0; nt_9 < 4; nt_9++) {
                            acc_1[0] = 0.0f;
                            acc_1[1] = 0.0f;
                            acc_1[2] = 0.0f;
                            acc_1[3] = 0.0f;
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                                : "r"(a_tile_1[0]), "r"(a_tile_1[1]), "r"(a_tile_1[2]), "r"(a_tile_1[3]), "r"((b_atom_1 + nt_9 * 4 * 2)[0]), "r"((b_atom_1 + nt_9 * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[4]), "r"(a_tile_1[(4) + 1]), "r"(a_tile_1[(4) + 2]), "r"(a_tile_1[(4) + 3]), "r"((b_atom_1 + (nt_9 * 4 + 1) * 2)[0]), "r"((b_atom_1 + (nt_9 * 4 + 1) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[8]), "r"(a_tile_1[(8) + 1]), "r"(a_tile_1[(8) + 2]), "r"(a_tile_1[(8) + 3]), "r"((b_atom_1 + (nt_9 * 4 + 2) * 2)[0]), "r"((b_atom_1 + (nt_9 * 4 + 2) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[12]), "r"(a_tile_1[(12) + 1]), "r"(a_tile_1[(12) + 2]), "r"(a_tile_1[(12) + 3]), "r"((b_atom_1 + (nt_9 * 4 + 3) * 2)[0]), "r"((b_atom_1 + (nt_9 * 4 + 3) * 2)[1]));
                            int w_idx_9 = q_stage_m_1 * 256 + (unsigned int)(nt_9 * 8) + (unsigned int)(tq_1 * 2);
                            float w0_9 = smem_w[w_idx_9];
                            float w1_9 = smem_w[w_idx_9 + 1];
                            float r0_9 = fmaxf(acc_1[0], 0.0f);
                            float r1_9 = fmaxf(acc_1[1], 0.0f);
                            float r2_9 = fmaxf(acc_1[2], 0.0f);
                            float r3_9 = fmaxf(acc_1[3], 0.0f);
                            partial0_6 += __fmaf_rn(r0_9, w0_9, r1_9 * w1_9);
                            partial1_6 += __fmaf_rn(r2_9, w0_9, r3_9 * w1_9);
                        }
                        float v0_6 = partial0_6 * scale0_3;
                        float v1_6 = partial1_6 * scale1_3;
                        float _shfl_xor_sync_12_1 = __shfl_xor_sync(0xFFFFFFFF, v0_6, 1);
                        v0_6 = v0_6 + _shfl_xor_sync_12_1;
                        float _shfl_xor_sync_13_1 = __shfl_xor_sync(0xFFFFFFFF, v1_6, 1);
                        v1_6 = v1_6 + _shfl_xor_sync_13_1;
                        float _shfl_xor_sync_14_1 = __shfl_xor_sync(0xFFFFFFFF, v0_6, 2);
                        v0_6 = v0_6 + _shfl_xor_sync_14_1;
                        float _shfl_xor_sync_15_1 = __shfl_xor_sync(0xFFFFFFFF, v1_6, 2);
                        v1_6 = v1_6 + _shfl_xor_sync_15_1;
                        int out0_6 = kv_offset_3 + quad_1;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_6)), "f"(v0_6), "r"((unsigned int)(tq_1 == 0)));
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_6 + 8))), "f"(v1_6), "r"((unsigned int)(tq_1 == 0)));
                        float partial0_6_1 = 0.0f;
                        float partial1_6_1 = 0.0f;
                        #pragma unroll
                        for (int nt_10 = 0; nt_10 < 4; nt_10++) {
                            acc_1[0] = 0.0f;
                            acc_1[1] = 0.0f;
                            acc_1[2] = 0.0f;
                            acc_1[3] = 0.0f;
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                                : "r"(a_tile_1[0]), "r"(a_tile_1[1]), "r"(a_tile_1[2]), "r"(a_tile_1[3]), "r"((b_atom_1 + (4 + nt_10) * 4 * 2)[0]), "r"((b_atom_1 + (4 + nt_10) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[4]), "r"(a_tile_1[(4) + 1]), "r"(a_tile_1[(4) + 2]), "r"(a_tile_1[(4) + 3]), "r"((b_atom_1 + ((4 + nt_10) * 4 + 1) * 2)[0]), "r"((b_atom_1 + ((4 + nt_10) * 4 + 1) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[8]), "r"(a_tile_1[(8) + 1]), "r"(a_tile_1[(8) + 2]), "r"(a_tile_1[(8) + 3]), "r"((b_atom_1 + ((4 + nt_10) * 4 + 2) * 2)[0]), "r"((b_atom_1 + ((4 + nt_10) * 4 + 2) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[12]), "r"(a_tile_1[(12) + 1]), "r"(a_tile_1[(12) + 2]), "r"(a_tile_1[(12) + 3]), "r"((b_atom_1 + ((4 + nt_10) * 4 + 3) * 2)[0]), "r"((b_atom_1 + ((4 + nt_10) * 4 + 3) * 2)[1]));
                            int w_idx_10 = q_stage_m_1 * 256 + 32 + (unsigned int)(nt_10 * 8) + (unsigned int)(tq_1 * 2);
                            float w0_10 = smem_w[w_idx_10];
                            float w1_10 = smem_w[w_idx_10 + 1];
                            float r0_10 = fmaxf(acc_1[0], 0.0f);
                            float r1_10 = fmaxf(acc_1[1], 0.0f);
                            float r2_10 = fmaxf(acc_1[2], 0.0f);
                            float r3_10 = fmaxf(acc_1[3], 0.0f);
                            partial0_6_1 += __fmaf_rn(r0_10, w0_10, r1_10 * w1_10);
                            partial1_6_1 += __fmaf_rn(r2_10, w0_10, r3_10 * w1_10);
                        }
                        float v0_6_1 = partial0_6_1 * scale0_3;
                        float v1_6_1 = partial1_6_1 * scale1_3;
                        float _shfl_xor_sync_16_1 = __shfl_xor_sync(0xFFFFFFFF, v0_6_1, 1);
                        v0_6_1 = v0_6_1 + _shfl_xor_sync_16_1;
                        float _shfl_xor_sync_17_1 = __shfl_xor_sync(0xFFFFFFFF, v1_6_1, 1);
                        v1_6_1 = v1_6_1 + _shfl_xor_sync_17_1;
                        float _shfl_xor_sync_18_1 = __shfl_xor_sync(0xFFFFFFFF, v0_6_1, 2);
                        v0_6_1 = v0_6_1 + _shfl_xor_sync_18_1;
                        float _shfl_xor_sync_19_1 = __shfl_xor_sync(0xFFFFFFFF, v1_6_1, 2);
                        v1_6_1 = v1_6_1 + _shfl_xor_sync_19_1;
                        int out0_6_1 = kv_offset_3 + logits_stride + quad_1;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_6_1)), "f"(v0_6_1), "r"((unsigned int)(tq_1 == 0)));
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_6_1 + 8))), "f"(v1_6_1), "r"((unsigned int)(tq_1 == 0)));
                        float partial0_7 = 0.0f;
                        float partial1_7 = 0.0f;
                        #pragma unroll
                        for (int nt_11 = 0; nt_11 < 4; nt_11++) {
                            acc_1[0] = 0.0f;
                            acc_1[1] = 0.0f;
                            acc_1[2] = 0.0f;
                            acc_1[3] = 0.0f;
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                                : "r"(a_tile_1[0]), "r"(a_tile_1[1]), "r"(a_tile_1[2]), "r"(a_tile_1[3]), "r"((b_atom_1 + (8 + nt_11) * 4 * 2)[0]), "r"((b_atom_1 + (8 + nt_11) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[4]), "r"(a_tile_1[(4) + 1]), "r"(a_tile_1[(4) + 2]), "r"(a_tile_1[(4) + 3]), "r"((b_atom_1 + ((8 + nt_11) * 4 + 1) * 2)[0]), "r"((b_atom_1 + ((8 + nt_11) * 4 + 1) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[8]), "r"(a_tile_1[(8) + 1]), "r"(a_tile_1[(8) + 2]), "r"(a_tile_1[(8) + 3]), "r"((b_atom_1 + ((8 + nt_11) * 4 + 2) * 2)[0]), "r"((b_atom_1 + ((8 + nt_11) * 4 + 2) * 2)[1]));
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                                : "r"(a_tile_1[12]), "r"(a_tile_1[(12) + 1]), "r"(a_tile_1[(12) + 2]), "r"(a_tile_1[(12) + 3]), "r"((b_atom_1 + ((8 + nt_11) * 4 + 3) * 2)[0]), "r"((b_atom_1 + ((8 + nt_11) * 4 + 3) * 2)[1]));
                            int w_idx_11 = q_stage_m_1 * 256 + 64 + (unsigned int)(nt_11 * 8) + (unsigned int)(tq_1 * 2);
                            float w0_11 = smem_w[w_idx_11];
                            float w1_11 = smem_w[w_idx_11 + 1];
                            float r0_11 = fmaxf(acc_1[0], 0.0f);
                            float r1_11 = fmaxf(acc_1[1], 0.0f);
                            float r2_11 = fmaxf(acc_1[2], 0.0f);
                            float r3_11 = fmaxf(acc_1[3], 0.0f);
                            partial0_7 += __fmaf_rn(r0_11, w0_11, r1_11 * w1_11);
                            partial1_7 += __fmaf_rn(r2_11, w0_11, r3_11 * w1_11);
                        }
                        float v0_7 = partial0_7 * scale0_3;
                        float v1_7 = partial1_7 * scale1_3;
                        float _shfl_xor_sync_20_1 = __shfl_xor_sync(0xFFFFFFFF, v0_7, 1);
                        v0_7 = v0_7 + _shfl_xor_sync_20_1;
                        float _shfl_xor_sync_21_1 = __shfl_xor_sync(0xFFFFFFFF, v1_7, 1);
                        v1_7 = v1_7 + _shfl_xor_sync_21_1;
                        float _shfl_xor_sync_22_1 = __shfl_xor_sync(0xFFFFFFFF, v0_7, 2);
                        v0_7 = v0_7 + _shfl_xor_sync_22_1;
                        float _shfl_xor_sync_23_1 = __shfl_xor_sync(0xFFFFFFFF, v1_7, 2);
                        v1_7 = v1_7 + _shfl_xor_sync_23_1;
                        int out0_7 = kv_offset_3 + 2 * logits_stride + quad_1;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_7)), "f"(v0_7), "r"((unsigned int)(tq_1 == 0)));
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_7 + 8))), "f"(v1_7), "r"((unsigned int)(tq_1 == 0)));
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
                mbarrier_wait(it_full_addr + (mi_stage_1) * 8, _phase_it_full_1);
                e_start_q_1 = smem_it[mi_stage_1 * 8];
                e_start_kv_1 = smem_it[mi_stage_1 * 8 + 1];
                e_end_q_1 = smem_it[mi_stage_1 * 8 + 2];
                e_end_kv_1 = smem_it[mi_stage_1 * 8 + 3];
                e_ctx_1 = smem_it[mi_stage_1 * 8 + 4];
                e_claim_q_1 = smem_it[mi_stage_1 * 8 + 5];
                e_claim_kv_1 = smem_it[mi_stage_1 * 8 + 6];
                e_claim_stop_1 = smem_it[mi_stage_1 * 8 + 7];
                mbarrier_arrive(it_empty_addr + (mi_stage_1) * 8);
                mi_stage_1 += 1;
                if (mi_stage_1 == 8) { mi_stage_1 = 0; _phase_it_full_1 ^= 1; }
                if (e_start_q_1 < 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: tma_g0 ----
    if (warp == 8) {
        { // tma_g0_main
            unsigned int cg_cur = 0;
            unsigned int q_stage = 0;
            unsigned int q_phase = 1;
            unsigned int kv_stage = 0;
            unsigned int kv_phase = 1;
            unsigned int i_stage = 0;
            unsigned int _phase_it_full_2 = 0;
            mbarrier_wait(it_full_addr + (i_stage) * 8, _phase_it_full_2);
            int d_start_q = smem_it[i_stage * 8];
            int d_start_kv = smem_it[i_stage * 8 + 1];
            int d_end_q = smem_it[i_stage * 8 + 2];
            int d_end_kv = smem_it[i_stage * 8 + 3];
            int d_ctx = -1;
            d_ctx = smem_it[i_stage * 8 + 4];
            int d_claim_q = smem_it[i_stage * 8 + 5];
            int d_claim_kv = smem_it[i_stage * 8 + 6];
            int d_claim_stop = smem_it[i_stage * 8 + 7];
            mbarrier_arrive(it_empty_addr + (i_stage) * 8);
            i_stage += 1;
            if (i_stage == 8) { i_stage = 0; _phase_it_full_2 ^= 1; }
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_kv_empty_g0 = 1;
            unsigned int _phase_claim_free = 1;
            #pragma unroll 1
            for (int _d_trip = 0; _d_trip < gridDim.x * 8 + 1; _d_trip++) {
                int d_q_stop = ((d_end_kv > 0) ? d_end_q + 1 : d_end_q);
                int d_claim_ctx = -1;
                d_claim_ctx = ((d_claim_q == d_start_q) ? d_ctx : -1);
                #pragma unroll 1
                for (int q_atom_0 = d_start_q; q_atom_0 < d_claim_stop; q_atom_0++) {
                    int q_idx_4 = q_atom_0;
                    int atom_in_q_4 = 0;
                    int tok_4 = q_idx_4 * 3 + atom_in_q_4 * 3;
                    int c_4 = -1;
                    if (q_atom_0 == d_start_q) {
                        c_4 = d_ctx;
                    }
                    if (c_4 < 0) {
                        c_4 = context_lens[q_idx_4 * 3 + 3 - 1];
                    }
                    int kv_lo = ((q_atom_0 == d_start_q) ? d_start_kv : 0);
                    int bt_lane0 = lane;
                    int bt_col0 = kv_lo + bt_lane0 * 2;
                    int bt_raw = 0;
                    if (bt_col0 < block_table_stride) {
                        bt_raw = block_table[q_idx_4 * block_table_stride + bt_col0];
                    }
                    mbarrier_wait(q_empty_addr + (q_stage) * 8, _phase_q_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_stage) * 8, 12672);
                        tma_2d_gmem2smem(smem_q_addr + q_stage * 12288, (&Q), 0, tok_4 * 32, q_full_addr + (q_stage) * 8);
                        tma_2d_gmem2smem(smem_w_addr + q_stage * 1024, (&Weights), 0, tok_4, q_full_addr + (q_stage) * 8);
                    }
                    q_stage += 1;
                    if (q_stage == 2) { q_stage = 0; _phase_q_empty ^= 1; }
                    int num_kv_4 = (c_4 + 64 - 1) / 64;
                    int kv_hi = ((q_atom_0 == d_claim_q) ? d_claim_kv : num_kv_4);
                    int bt_row = q_idx_4 * block_table_stride;
                    int bt_lane = lane;
                    int bt_ptr = 0;
                    int bt_cache = bt_raw;
                    if (num_kv_4 <= kv_lo + bt_lane * 2) {
                        bt_cache = 0;
                    }
                    #pragma unroll 1
                    for (int kv_idx_4 = kv_lo; kv_idx_4 < kv_hi; kv_idx_4 += 2) {
                        int tile = kv_idx_4;
                        if (bt_ptr == 32) {
                            bt_ptr = 0;
                            int bt_tile = tile + bt_lane * 2;
                            bt_cache = 0;
                            if (bt_tile < num_kv_4) {
                                bt_cache = block_table[bt_row + bt_tile];
                            }
                        }
                        int kv_block = __shfl_sync(0xFFFFFFFF, bt_cache, bt_ptr);
                        bt_ptr += 1;
                        int page_off = 0;
                        mbarrier_wait(kv_empty_g0_addr + (kv_stage) * 8, _phase_kv_empty_g0);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_g0_addr + (kv_stage) * 8, 8448);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_kv_g0_addr + kv_stage * 8192), "l"((&KV)), "r"(0), "r"(page_off), "r"(kv_block),
                                   "r"(kv_full_g0_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_sc_g0_addr + kv_stage * 1024), "l"((&KV_scales)), "r"(2048 + page_off), "r"(kv_block),
                                   "r"(kv_full_g0_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                        kv_stage += 1;
                        if (kv_stage == 3) { kv_stage = 0; _phase_kv_empty_g0 ^= 1; }
                    }
                }
                mbarrier_wait(claim_free_addr + (cg_cur) * 8, _phase_claim_free);
                if (elect_sync()) {
                    mbarrier_arrive(claim_go_addr + (cg_cur) * 8);
                }
                cg_cur += 1;
                if (cg_cur == 4) { cg_cur = 0; _phase_claim_free ^= 1; }
                #pragma unroll 1
                for (int q_atom_1 = d_claim_q; q_atom_1 < d_q_stop; q_atom_1++) {
                    int q_idx_5 = q_atom_1;
                    int atom_in_q_5 = 0;
                    int tok_5 = q_idx_5 * 3 + atom_in_q_5 * 3;
                    int c_5 = -1;
                    if (q_atom_1 == d_claim_q) {
                        c_5 = d_claim_ctx;
                    }
                    if (c_5 < 0) {
                        c_5 = context_lens[q_idx_5 * 3 + 3 - 1];
                    }
                    int kv_lo_1 = ((q_atom_1 == d_claim_q) ? d_claim_kv : 0);
                    int bt_lane0_1 = lane;
                    int bt_col0_1 = kv_lo_1 + bt_lane0_1 * 2;
                    int bt_raw_1 = 0;
                    if (bt_col0_1 < block_table_stride) {
                        bt_raw_1 = block_table[q_idx_5 * block_table_stride + bt_col0_1];
                    }
                    mbarrier_wait(q_empty_addr + (q_stage) * 8, _phase_q_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_stage) * 8, 12672);
                        tma_2d_gmem2smem(smem_q_addr + q_stage * 12288, (&Q), 0, tok_5 * 32, q_full_addr + (q_stage) * 8);
                        tma_2d_gmem2smem(smem_w_addr + q_stage * 1024, (&Weights), 0, tok_5, q_full_addr + (q_stage) * 8);
                    }
                    q_stage += 1;
                    if (q_stage == 2) { q_stage = 0; _phase_q_empty ^= 1; }
                    int num_kv_5 = (c_5 + 64 - 1) / 64;
                    int kv_hi_1 = ((q_atom_1 == d_end_q) ? d_end_kv : num_kv_5);
                    int bt_row_1 = q_idx_5 * block_table_stride;
                    int bt_lane_1 = lane;
                    int bt_ptr_1 = 0;
                    int bt_cache_1 = bt_raw_1;
                    if (num_kv_5 <= kv_lo_1 + bt_lane_1 * 2) {
                        bt_cache_1 = 0;
                    }
                    #pragma unroll 1
                    for (int kv_idx_5 = kv_lo_1; kv_idx_5 < kv_hi_1; kv_idx_5 += 2) {
                        int tile_1 = kv_idx_5;
                        if (bt_ptr_1 == 32) {
                            bt_ptr_1 = 0;
                            int bt_tile_1 = tile_1 + bt_lane_1 * 2;
                            bt_cache_1 = 0;
                            if (bt_tile_1 < num_kv_5) {
                                bt_cache_1 = block_table[bt_row_1 + bt_tile_1];
                            }
                        }
                        int kv_block_1 = __shfl_sync(0xFFFFFFFF, bt_cache_1, bt_ptr_1);
                        bt_ptr_1 += 1;
                        int page_off_1 = 0;
                        mbarrier_wait(kv_empty_g0_addr + (kv_stage) * 8, _phase_kv_empty_g0);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_g0_addr + (kv_stage) * 8, 8448);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_kv_g0_addr + kv_stage * 8192), "l"((&KV)), "r"(0), "r"(page_off_1), "r"(kv_block_1),
                                   "r"(kv_full_g0_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_sc_g0_addr + kv_stage * 1024), "l"((&KV_scales)), "r"(2048 + page_off_1), "r"(kv_block_1),
                                   "r"(kv_full_g0_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                        kv_stage += 1;
                        if (kv_stage == 3) { kv_stage = 0; _phase_kv_empty_g0 ^= 1; }
                    }
                }
                mbarrier_wait(it_full_addr + (i_stage) * 8, _phase_it_full_2);
                d_start_q = smem_it[i_stage * 8];
                d_start_kv = smem_it[i_stage * 8 + 1];
                d_end_q = smem_it[i_stage * 8 + 2];
                d_end_kv = smem_it[i_stage * 8 + 3];
                d_ctx = smem_it[i_stage * 8 + 4];
                d_claim_q = smem_it[i_stage * 8 + 5];
                d_claim_kv = smem_it[i_stage * 8 + 6];
                d_claim_stop = smem_it[i_stage * 8 + 7];
                mbarrier_arrive(it_empty_addr + (i_stage) * 8);
                i_stage += 1;
                if (i_stage == 8) { i_stage = 0; _phase_it_full_2 ^= 1; }
                if (d_start_q < 0) {
                    break;
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
            unsigned int i_stage_1 = 0;
            unsigned int _phase_it_full_3 = 0;
            mbarrier_wait(it_full_addr + (i_stage_1) * 8, _phase_it_full_3);
            int d_start_q_1 = smem_it[i_stage_1 * 8];
            int d_start_kv_1 = smem_it[i_stage_1 * 8 + 1];
            int d_end_q_1 = smem_it[i_stage_1 * 8 + 2];
            int d_end_kv_1 = smem_it[i_stage_1 * 8 + 3];
            int d_ctx_1 = -1;
            d_ctx_1 = smem_it[i_stage_1 * 8 + 4];
            int d_claim_q_1 = smem_it[i_stage_1 * 8 + 5];
            int d_claim_kv_1 = smem_it[i_stage_1 * 8 + 6];
            int d_claim_stop_1 = smem_it[i_stage_1 * 8 + 7];
            mbarrier_arrive(it_empty_addr + (i_stage_1) * 8);
            i_stage_1 += 1;
            if (i_stage_1 == 8) { i_stage_1 = 0; _phase_it_full_3 ^= 1; }
            unsigned int _phase_kv_empty_g1 = 1;
            #pragma unroll 1
            for (int _d_trip_1 = 0; _d_trip_1 < gridDim.x * 8 + 1; _d_trip_1++) {
                int d_q_stop_1 = ((d_end_kv_1 > 0) ? d_end_q_1 + 1 : d_end_q_1);
                int d_claim_ctx_1 = -1;
                d_claim_ctx_1 = ((d_claim_q_1 == d_start_q_1) ? d_ctx_1 : -1);
                #pragma unroll 1
                for (int q_atom_2 = d_start_q_1; q_atom_2 < d_claim_stop_1; q_atom_2++) {
                    int q_idx_6 = q_atom_2;
                    int atom_in_q_6 = 0;
                    int tok_6 = q_idx_6 * 3 + atom_in_q_6 * 3;
                    int c_6 = -1;
                    if (q_atom_2 == d_start_q_1) {
                        c_6 = d_ctx_1;
                    }
                    if (c_6 < 0) {
                        c_6 = context_lens[q_idx_6 * 3 + 3 - 1];
                    }
                    int kv_lo_2 = ((q_atom_2 == d_start_q_1) ? d_start_kv_1 : 0);
                    int bt_lane0_2 = lane;
                    int bt_col0_2 = kv_lo_2 + 1 + bt_lane0_2 * 2;
                    int bt_raw_2 = 0;
                    if (bt_col0_2 < block_table_stride) {
                        bt_raw_2 = block_table[q_idx_6 * block_table_stride + bt_col0_2];
                    }
                    int num_kv_6 = (c_6 + 64 - 1) / 64;
                    int kv_hi_2 = ((q_atom_2 == d_claim_q_1) ? d_claim_kv_1 : num_kv_6);
                    int bt_row_2 = q_idx_6 * block_table_stride;
                    int bt_lane_2 = lane;
                    int bt_ptr_2 = 0;
                    int bt_cache_2 = bt_raw_2;
                    if (num_kv_6 <= kv_lo_2 + 1 + bt_lane_2 * 2) {
                        bt_cache_2 = 0;
                    }
                    #pragma unroll 1
                    for (int kv_idx_6 = kv_lo_2; kv_idx_6 < kv_hi_2; kv_idx_6 += 2) {
                        int tile_2 = kv_idx_6 + 1;
                        if (bt_ptr_2 == 32) {
                            bt_ptr_2 = 0;
                            int bt_tile_2 = tile_2 + bt_lane_2 * 2;
                            bt_cache_2 = 0;
                            if (bt_tile_2 < num_kv_6) {
                                bt_cache_2 = block_table[bt_row_2 + bt_tile_2];
                            }
                        }
                        int kv_block_2 = __shfl_sync(0xFFFFFFFF, bt_cache_2, bt_ptr_2);
                        bt_ptr_2 += 1;
                        int page_off_2 = 0;
                        mbarrier_wait(kv_empty_g1_addr + (kv_stage_1) * 8, _phase_kv_empty_g1);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_g1_addr + (kv_stage_1) * 8, 8448);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_kv_g1_addr + kv_stage_1 * 8192), "l"((&KV)), "r"(0), "r"(page_off_2), "r"(kv_block_2),
                                   "r"(kv_full_g1_addr + (kv_stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_sc_g1_addr + kv_stage_1 * 1024), "l"((&KV_scales)), "r"(2048 + page_off_2), "r"(kv_block_2),
                                   "r"(kv_full_g1_addr + (kv_stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                        kv_stage_1 += 1;
                        if (kv_stage_1 == 3) { kv_stage_1 = 0; _phase_kv_empty_g1 ^= 1; }
                    }
                }
                #pragma unroll 1
                for (int q_atom_3 = d_claim_q_1; q_atom_3 < d_q_stop_1; q_atom_3++) {
                    int q_idx_7 = q_atom_3;
                    int atom_in_q_7 = 0;
                    int tok_7 = q_idx_7 * 3 + atom_in_q_7 * 3;
                    int c_7 = -1;
                    if (q_atom_3 == d_claim_q_1) {
                        c_7 = d_claim_ctx_1;
                    }
                    if (c_7 < 0) {
                        c_7 = context_lens[q_idx_7 * 3 + 3 - 1];
                    }
                    int kv_lo_3 = ((q_atom_3 == d_claim_q_1) ? d_claim_kv_1 : 0);
                    int bt_lane0_3 = lane;
                    int bt_col0_3 = kv_lo_3 + 1 + bt_lane0_3 * 2;
                    int bt_raw_3 = 0;
                    if (bt_col0_3 < block_table_stride) {
                        bt_raw_3 = block_table[q_idx_7 * block_table_stride + bt_col0_3];
                    }
                    int num_kv_7 = (c_7 + 64 - 1) / 64;
                    int kv_hi_3 = ((q_atom_3 == d_end_q_1) ? d_end_kv_1 : num_kv_7);
                    int bt_row_3 = q_idx_7 * block_table_stride;
                    int bt_lane_3 = lane;
                    int bt_ptr_3 = 0;
                    int bt_cache_3 = bt_raw_3;
                    if (num_kv_7 <= kv_lo_3 + 1 + bt_lane_3 * 2) {
                        bt_cache_3 = 0;
                    }
                    #pragma unroll 1
                    for (int kv_idx_7 = kv_lo_3; kv_idx_7 < kv_hi_3; kv_idx_7 += 2) {
                        int tile_3 = kv_idx_7 + 1;
                        if (bt_ptr_3 == 32) {
                            bt_ptr_3 = 0;
                            int bt_tile_3 = tile_3 + bt_lane_3 * 2;
                            bt_cache_3 = 0;
                            if (bt_tile_3 < num_kv_7) {
                                bt_cache_3 = block_table[bt_row_3 + bt_tile_3];
                            }
                        }
                        int kv_block_3 = __shfl_sync(0xFFFFFFFF, bt_cache_3, bt_ptr_3);
                        bt_ptr_3 += 1;
                        int page_off_3 = 0;
                        mbarrier_wait(kv_empty_g1_addr + (kv_stage_1) * 8, _phase_kv_empty_g1);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_g1_addr + (kv_stage_1) * 8, 8448);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_kv_g1_addr + kv_stage_1 * 8192), "l"((&KV)), "r"(0), "r"(page_off_3), "r"(kv_block_3),
                                   "r"(kv_full_g1_addr + (kv_stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_sc_g1_addr + kv_stage_1 * 1024), "l"((&KV_scales)), "r"(2048 + page_off_3), "r"(kv_block_3),
                                   "r"(kv_full_g1_addr + (kv_stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                        kv_stage_1 += 1;
                        if (kv_stage_1 == 3) { kv_stage_1 = 0; _phase_kv_empty_g1 ^= 1; }
                    }
                }
                mbarrier_wait(it_full_addr + (i_stage_1) * 8, _phase_it_full_3);
                d_start_q_1 = smem_it[i_stage_1 * 8];
                d_start_kv_1 = smem_it[i_stage_1 * 8 + 1];
                d_end_q_1 = smem_it[i_stage_1 * 8 + 2];
                d_end_kv_1 = smem_it[i_stage_1 * 8 + 3];
                d_ctx_1 = smem_it[i_stage_1 * 8 + 4];
                d_claim_q_1 = smem_it[i_stage_1 * 8 + 5];
                d_claim_kv_1 = smem_it[i_stage_1 * 8 + 6];
                d_claim_stop_1 = smem_it[i_stage_1 * 8 + 7];
                mbarrier_arrive(it_empty_addr + (i_stage_1) * 8);
                i_stage_1 += 1;
                if (i_stage_1 == 8) { i_stage_1 = 0; _phase_it_full_3 ^= 1; }
                if (d_start_q_1 < 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: dyn_disp ----
    if (warp == 10) {
        { // dyn_disp_main
            unsigned int d_stage = 0;
            unsigned int go_stage = 0;
            int n_grid = gridDim.x;
            int cnt_idx = (n_grid * 9 + 1) * 2;
            int n_pool = 0;
            int b_sq = -1;
            int b_skv = 0;
            int b_eq = -1;
            int b_ekv = 0;
            int b_ctx = -1;
            int b_pg0 = -1;
            int b_pg1 = -1;
            int b_cq = -1;
            int b_ckv = 0;
            int b_cstop = -1;
            unsigned int _phase_it_empty = 1;
            mbarrier_wait(it_empty_addr + (d_stage) * 8, _phase_it_empty);
            if (lane == 0) {
                n_pool = schedule_meta[cnt_idx + 1];
                b_sq = schedule_meta[bid * 2];
                b_skv = schedule_meta[bid * 2 + 1] * 2;
                b_eq = schedule_meta[bid * 2 + 2];
                b_ekv = schedule_meta[bid * 2 + 3] * 2;
                b_cq = b_sq;
                b_ckv = b_skv;
                b_cstop = b_sq;
                int s_q1 = ((b_ekv > 0) ? b_eq + 1 : b_eq);
                if (s_q1 > b_sq) {
                    b_ctx = context_lens[b_sq * 3 + 3 - 1];
                    b_cq = s_q1 - 1;
                    int s_lo = ((b_cq == b_sq) ? b_skv : 0);
                    int s_hi = b_ekv;
                    if (b_cq != b_eq) {
                        s_hi = (context_lens[b_cq * 3 + 3 - 1] + 64 - 1) / 64;
                    }
                    int s_steps = (s_hi - s_lo + 2 - 1) / 2;
                    b_ckv = s_lo;
                    if (s_steps > 2) {
                        b_ckv = s_lo + (s_steps - 2) * 2;
                    }
                    b_cstop = ((b_ckv > s_lo) ? b_cq + 1 : b_cq);
                }
                smem_it[d_stage * 8] = b_sq;
                smem_it[d_stage * 8 + 1] = b_skv;
                smem_it[d_stage * 8 + 2] = b_eq;
                smem_it[d_stage * 8 + 3] = b_ekv;
                smem_it[d_stage * 8 + 4] = b_ctx;
                smem_it[d_stage * 8 + 5] = b_cq;
                smem_it[d_stage * 8 + 6] = b_ckv;
                smem_it[d_stage * 8 + 7] = b_cstop;
                mbarrier_arrive(it_full_addr + (d_stage) * 8);
            }
            unsigned int _phase_it_full_4 = 0;
            mbarrier_wait(it_full_addr + (d_stage) * 8, _phase_it_full_4);
            __syncwarp();
            if (lane == 0) {
                mbarrier_arrive(it_empty_addr + (d_stage) * 8);
            }
            d_stage += 1;
            if (d_stage == 8) { d_stage = 0; _phase_it_empty ^= 1; _phase_it_full_4 ^= 1; }
            int p_sq = 0;
            unsigned int _phase_claim_go = 0;
            #pragma unroll 1
            for (int p_trip = 0; p_trip < gridDim.x * 8 + 1; p_trip++) {
                mbarrier_wait(claim_go_addr + (go_stage) * 8, _phase_claim_go);
                __syncwarp();
                if (lane == 0) {
                    mbarrier_arrive(claim_free_addr + (go_stage) * 8);
                }
                go_stage += 1;
                if (go_stage == 4) { go_stage = 0; _phase_claim_go ^= 1; }
                if (p_trip == 0) {
                    if (lane == 0) {
                        int f_tk = atomicAdd(&schedule_meta[cnt_idx], 1);
                        if (f_tk == n_pool + n_grid - 1) {
                            schedule_meta[cnt_idx] = 0;
                        }
                        b_sq = -1;
                        b_skv = 0;
                        b_eq = -1;
                        b_ekv = 0;
                        b_cq = -1;
                        b_ckv = 0;
                        b_cstop = -1;
                        b_ctx = -1;
                        b_pg0 = -1;
                        b_pg1 = -1;
                        if (f_tk < n_pool) {
                            int f_it = n_grid + f_tk;
                            b_sq = schedule_meta[f_it * 2];
                            b_skv = schedule_meta[f_it * 2 + 1] * 2;
                            b_eq = schedule_meta[f_it * 2 + 2];
                            b_ekv = schedule_meta[f_it * 2 + 3] * 2;
                            b_cq = b_sq;
                            b_ckv = b_skv;
                            b_cstop = b_sq;
                            b_ctx = context_lens[b_sq * 3 + 3 - 1];
                            int f_nkv = (b_ctx + 64 - 1) / 64;
                            if (b_skv < f_nkv) {
                                b_pg0 = block_table[b_sq * block_table_stride + b_skv];
                            }
                            if (f_nkv > b_skv + 1) {
                                b_pg1 = block_table[b_sq * block_table_stride + (b_skv + 1)];
                            }
                        }
                    }
                }
                mbarrier_wait(it_empty_addr + (d_stage) * 8, _phase_it_empty);
                if (lane == 0) {
                    smem_it[d_stage * 8] = b_sq;
                    smem_it[d_stage * 8 + 1] = b_skv;
                    smem_it[d_stage * 8 + 2] = b_eq;
                    smem_it[d_stage * 8 + 3] = b_ekv;
                    smem_it[d_stage * 8 + 4] = b_ctx;
                    smem_it[d_stage * 8 + 5] = b_cq;
                    smem_it[d_stage * 8 + 6] = b_ckv;
                    smem_it[d_stage * 8 + 7] = b_cstop;
                    mbarrier_arrive(it_full_addr + (d_stage) * 8);
                    if (b_sq >= 0) {
                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&Q))), "r"((int)(0)), "r"((int)(b_sq * 96)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&Weights))), "r"((int)(0)), "r"((int)(b_sq * 3)) : "memory");
                        if (b_pg0 >= 0) {
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&KV))), "r"((int)(0)), "r"((int)(0)), "r"((int)(b_pg0)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&KV_scales))), "r"((int)(2048)), "r"((int)(b_pg0)) : "memory");
                        }
                        if (b_pg1 >= 0) {
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&KV))), "r"((int)(0)), "r"((int)(0)), "r"((int)(b_pg1)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&KV_scales))), "r"((int)(2048)), "r"((int)(b_pg1)) : "memory");
                        }
                    }
                    if (b_sq >= 0) {
                        int h_tk = atomicAdd(&schedule_meta[cnt_idx], 1);
                        if (h_tk == n_pool + n_grid - 1) {
                            schedule_meta[cnt_idx] = 0;
                        }
                        b_sq = -1;
                        b_skv = 0;
                        b_eq = -1;
                        b_ekv = 0;
                        b_cq = -1;
                        b_ckv = 0;
                        b_cstop = -1;
                        b_ctx = -1;
                        b_pg0 = -1;
                        b_pg1 = -1;
                        if (h_tk < n_pool) {
                            int h_it = n_grid + h_tk;
                            b_sq = schedule_meta[h_it * 2];
                            b_skv = schedule_meta[h_it * 2 + 1] * 2;
                            b_eq = schedule_meta[h_it * 2 + 2];
                            b_ekv = schedule_meta[h_it * 2 + 3] * 2;
                            b_cq = b_sq;
                            b_ckv = b_skv;
                            b_cstop = b_sq;
                            b_ctx = context_lens[b_sq * 3 + 3 - 1];
                            int h_nkv = (b_ctx + 64 - 1) / 64;
                            if (b_skv < h_nkv) {
                                b_pg0 = block_table[b_sq * block_table_stride + b_skv];
                            }
                            if (h_nkv > b_skv + 1) {
                                b_pg1 = block_table[b_sq * block_table_stride + (b_skv + 1)];
                            }
                        }
                    }
                }
                mbarrier_wait(it_full_addr + (d_stage) * 8, _phase_it_full_4);
                p_sq = smem_it[d_stage * 8];
                __syncwarp();
                if (lane == 0) {
                    mbarrier_arrive(it_empty_addr + (d_stage) * 8);
                }
                d_stage += 1;
                if (d_stage == 8) { d_stage = 0; _phase_it_empty ^= 1; _phase_it_full_4 ^= 1; }
                if (p_sq < 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 11) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
