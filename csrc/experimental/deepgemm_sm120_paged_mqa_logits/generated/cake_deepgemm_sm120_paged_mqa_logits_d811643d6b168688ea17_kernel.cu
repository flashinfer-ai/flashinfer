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
#define SMEM_SMEM_Q_STAGE_BYTES 20480
#define SMEM_SMEM_Q_STRIDE 20480
#define SMEM_SMEM_W_OFF 41984
#define SMEM_SMEM_W_STAGE_BYTES 640
#define SMEM_SMEM_W_STRIDE 1024
#define SMEM_SMEM_KV_0_OFF 44032
#define SMEM_SMEM_KV_0_STAGE_BYTES 8192
#define SMEM_SMEM_KV_0_STRIDE 8192
#define SMEM_SMEM_KV_1_OFF 68608
#define SMEM_SMEM_KV_1_STAGE_BYTES 8192
#define SMEM_SMEM_KV_1_STRIDE 8192
#define SMEM_SMEM_SC_0_OFF 93184
#define SMEM_SMEM_SC_0_STAGE_BYTES 256
#define SMEM_SMEM_SC_0_STRIDE 1024
#define SMEM_SMEM_SC_1_OFF 96256
#define SMEM_SMEM_SC_1_STAGE_BYTES 256
#define SMEM_SMEM_SC_1_STRIDE 1024
#define SMEM_TOTAL 99328
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
kernel_cake_deepgemm_sm120_paged_mqa_logits_d811643d6b168688ea17(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KV_scales, const __grid_constant__ CUtensorMap Weights, float* __restrict__ Logits, int* __restrict__ context_lens, int* __restrict__ block_table, int* __restrict__ schedule_meta, int logits_stride, int block_table_stride)
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
    #define kv_full_0_addr (mbar_base + 32)
    #define kv_full_1_addr (mbar_base + 56)
    #define kv_empty_0_addr (mbar_base + 80)
    #define kv_empty_1_addr (mbar_base + 104)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    float* smem_w = reinterpret_cast<float*>(smem_raw + 41984);
    const int smem_w_addr = smem + 41984;
    uint8_t* smem_kv_0 = reinterpret_cast<uint8_t*>(smem_raw + 44032);
    const int smem_kv_addr_0 = smem + 44032;
    uint8_t* smem_kv_1 = reinterpret_cast<uint8_t*>(smem_raw + 68608);
    const int smem_kv_addr_1 = smem + 68608;
    float* smem_sc_0 = reinterpret_cast<float*>(smem_raw + 93184);
    const int smem_sc_addr_0 = smem + 93184;
    float* smem_sc_1 = reinterpret_cast<float*>(smem_raw + 96256);
    const int smem_sc_addr_1 = smem + 96256;

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
            // kv_full_0: 3 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // kv_full_1: 3 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // kv_empty_0: 3 barriers, init_count=128
            mbarrier_init(smem + 80, 128);
            mbarrier_init(smem + 88, 128);
            mbarrier_init(smem + 96, 128);
            // kv_empty_1: 3 barriers, init_count=128
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

    // ---- Role: math_0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // math_0_main
            int w_in_g_0 = warp;
            int lane_0_0 = lane;
            int quad_0 = lane_0_0 / 4;
            int tq_0 = lane_0_0 % 4;
            int a_row_0 = w_in_g_0 * 16 + (lane_0_0 & 7) + (lane_0_0 >> 3 & 1) * 8;
            int a_col_0 = (lane_0_0 >> 4) * 16;
            int b_row_lane_0 = lane_0_0 & 7;
            int b_col_0 = (lane_0_0 >> 3 & 1) * 16;
            unsigned int q_stage_m_0 = 0;
            unsigned int kv_stage_m_0 = 0;
            unsigned int a_frag_0[4];
            unsigned int b_frag_0[2];
            float acc_0[4];
            int sm = bid;
            int start_q = schedule_meta[sm * 2];
            int start_kv = schedule_meta[sm * 2 + 1] * 2;
            int end_q = schedule_meta[sm * 2 + 2];
            int end_kv = schedule_meta[sm * 2 + 3] * 2;
            int q_stop = ((end_kv > 0) ? end_q + 1 : end_q);
            unsigned int _phase_q_full = 0;
            unsigned int _phase_kv_full_0 = 0;
            #pragma unroll 1
            for (int q_atom_2 = start_q; q_atom_2 < q_stop; q_atom_2++) {
                int q_idx = q_atom_2;
                int atom_in_q = 0;
                int tok = q_idx * 5 + atom_in_q * 5;
                int ctx_last = context_lens[q_idx * 5 + 5 - 1];
                int num_kv = (ctx_last + 64 - 1) / 64;
                int lo = ((q_atom_2 == start_q) ? start_kv : 0);
                int hi = ((q_atom_2 == end_q) ? end_kv : num_kv);
                mbarrier_wait(q_full_addr + (q_stage_m_0) * 8, _phase_q_full);
                int n_tok_m_0 = 5;
                #pragma unroll 1
                for (int kv_idx_2 = lo; kv_idx_2 < hi; kv_idx_2 += 2) {
                    mbarrier_wait(kv_full_0_addr + (kv_stage_m_0) * 8, _phase_kv_full_0);
                    int sc_idx_0 = kv_stage_m_0 * 256 + (unsigned int)(w_in_g_0 * 16) + (unsigned int)quad_0;
                    float scale0_0 = smem_sc_0[sc_idx_0];
                    float scale1_0 = smem_sc_0[sc_idx_0 + 8];
                    int kv_offset_0 = tok * logits_stride + kv_idx_2 * 64 + w_in_g_0 * 16;
                    float partial0 = 0.0f;
                    float partial1 = 0.0f;
                    #pragma unroll
                    for (int nt = 0; nt < 4; nt++) {
                        acc_0[0] = 0.0f;
                        acc_0[1] = 0.0f;
                        acc_0[2] = 0.0f;
                        acc_0[3] = 0.0f;
                        int a_cb_0 = a_col_0;
                        unsigned int a_addr_0 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_0 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_0)
                            : "memory");
                        int b_row = nt * 8 + b_row_lane_0;
                        int b_cb = b_col_0;
                        unsigned int b_addr = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row * 128) + (unsigned int)(b_cb ^ (b_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_0[0]), "=f"(acc_0[1]), "=f"(acc_0[2]), "=f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_1 = a_col_0 + 32;
                        unsigned int a_addr_1 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_1 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_1)
                            : "memory");
                        int b_row_0 = nt * 8 + b_row_lane_0;
                        int b_cb_0 = b_col_0 + 32;
                        unsigned int b_addr_0 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_0 * 128) + (unsigned int)(b_cb_0 ^ (b_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_0)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int a_cb_2 = a_col_0 + 64;
                        unsigned int a_addr_2 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_2 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_2)
                            : "memory");
                        int b_row_1 = nt * 8 + b_row_lane_0;
                        int b_cb_1 = b_col_0 + 64;
                        unsigned int b_addr_1 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_1 * 128) + (unsigned int)(b_cb_1 ^ (b_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int a_cb_3 = a_col_0 + 96;
                        unsigned int a_addr_3 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_3 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_3)
                            : "memory");
                        int b_row_2 = nt * 8 + b_row_lane_0;
                        int b_cb_2 = b_col_0 + 96;
                        unsigned int b_addr_2 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_2 * 128) + (unsigned int)(b_cb_2 ^ (b_row_2 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_2)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int w_idx = q_stage_m_0 * 256 + (unsigned int)(nt * 8) + (unsigned int)(tq_0 * 2);
                        float w0 = smem_w[w_idx];
                        float w1 = smem_w[w_idx + 1];
                        float r0 = fmaxf(acc_0[0], 0.0f);
                        float r1 = fmaxf(acc_0[1], 0.0f);
                        float r2 = fmaxf(acc_0[2], 0.0f);
                        float r3 = fmaxf(acc_0[3], 0.0f);
                        partial0 += __fmaf_rn(r0, w0, r1 * w1);
                        partial1 += __fmaf_rn(r2, w0, r3 * w1);
                    }
                    float v0 = partial0 * scale0_0;
                    float v1 = partial1 * scale1_0;
                    float _shfl_xor_sync_0 = __shfl_xor_sync(0xFFFFFFFF, v0, 1);
                    v0 = v0 + _shfl_xor_sync_0;
                    float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, v1, 1);
                    v1 = v1 + _shfl_xor_sync_1;
                    float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, v0, 2);
                    v0 = v0 + _shfl_xor_sync_2;
                    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, v1, 2);
                    v1 = v1 + _shfl_xor_sync_3;
                    int out0 = kv_offset_0 + quad_0;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0)), "f"(v0), "r"((unsigned int)(tq_0 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0 + 8))), "f"(v1), "r"((unsigned int)(tq_0 == 0)));
                    float partial0_0 = 0.0f;
                    float partial1_0 = 0.0f;
                    #pragma unroll
                    for (int nt_1 = 0; nt_1 < 4; nt_1++) {
                        acc_0[0] = 0.0f;
                        acc_0[1] = 0.0f;
                        acc_0[2] = 0.0f;
                        acc_0[3] = 0.0f;
                        int a_cb_4 = a_col_0;
                        unsigned int a_addr_4 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_4 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_4)
                            : "memory");
                        int b_row_3 = 32 + nt_1 * 8 + b_row_lane_0;
                        int b_cb_3 = b_col_0;
                        unsigned int b_addr_3 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_3 * 128) + (unsigned int)(b_cb_3 ^ (b_row_3 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_3)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_0[0]), "=f"(acc_0[1]), "=f"(acc_0[2]), "=f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_5 = a_col_0 + 32;
                        unsigned int a_addr_5 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_5 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_5)
                            : "memory");
                        int b_row_3_1 = 32 + nt_1 * 8 + b_row_lane_0;
                        int b_cb_3_1 = b_col_0 + 32;
                        unsigned int b_addr_3_1 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_3_1 * 128) + (unsigned int)(b_cb_3_1 ^ (b_row_3_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_3_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int a_cb_6 = a_col_0 + 64;
                        unsigned int a_addr_6 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_6 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_6)
                            : "memory");
                        int b_row_4 = 32 + nt_1 * 8 + b_row_lane_0;
                        int b_cb_4 = b_col_0 + 64;
                        unsigned int b_addr_4 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_4 * 128) + (unsigned int)(b_cb_4 ^ (b_row_4 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_4)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int a_cb_7 = a_col_0 + 96;
                        unsigned int a_addr_7 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_7 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_7)
                            : "memory");
                        int b_row_5 = 32 + nt_1 * 8 + b_row_lane_0;
                        int b_cb_5 = b_col_0 + 96;
                        unsigned int b_addr_5 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_5 * 128) + (unsigned int)(b_cb_5 ^ (b_row_5 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_5)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int w_idx_1 = q_stage_m_0 * 256 + 32 + (unsigned int)(nt_1 * 8) + (unsigned int)(tq_0 * 2);
                        float w0_1 = smem_w[w_idx_1];
                        float w1_1 = smem_w[w_idx_1 + 1];
                        float r0_1 = fmaxf(acc_0[0], 0.0f);
                        float r1_1 = fmaxf(acc_0[1], 0.0f);
                        float r2_1 = fmaxf(acc_0[2], 0.0f);
                        float r3_1 = fmaxf(acc_0[3], 0.0f);
                        partial0_0 += __fmaf_rn(r0_1, w0_1, r1_1 * w1_1);
                        partial1_0 += __fmaf_rn(r2_1, w0_1, r3_1 * w1_1);
                    }
                    float v0_0 = partial0_0 * scale0_0;
                    float v1_0 = partial1_0 * scale1_0;
                    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, v0_0, 1);
                    v0_0 = v0_0 + _shfl_xor_sync_4;
                    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, v1_0, 1);
                    v1_0 = v1_0 + _shfl_xor_sync_5;
                    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, v0_0, 2);
                    v0_0 = v0_0 + _shfl_xor_sync_6;
                    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, v1_0, 2);
                    v1_0 = v1_0 + _shfl_xor_sync_7;
                    int out0_0 = kv_offset_0 + logits_stride + quad_0;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_0)), "f"(v0_0), "r"((unsigned int)(tq_0 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_0 + 8))), "f"(v1_0), "r"((unsigned int)(tq_0 == 0)));
                    float partial0_1 = 0.0f;
                    float partial1_1 = 0.0f;
                    #pragma unroll
                    for (int nt_2 = 0; nt_2 < 4; nt_2++) {
                        acc_0[0] = 0.0f;
                        acc_0[1] = 0.0f;
                        acc_0[2] = 0.0f;
                        acc_0[3] = 0.0f;
                        int a_cb_8 = a_col_0;
                        unsigned int a_addr_8 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_8 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_8)
                            : "memory");
                        int b_row_6 = 64 + nt_2 * 8 + b_row_lane_0;
                        int b_cb_6 = b_col_0;
                        unsigned int b_addr_6 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_6 * 128) + (unsigned int)(b_cb_6 ^ (b_row_6 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_6)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_0[0]), "=f"(acc_0[1]), "=f"(acc_0[2]), "=f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_9 = a_col_0 + 32;
                        unsigned int a_addr_9 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_9 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_9)
                            : "memory");
                        int b_row_6_1 = 64 + nt_2 * 8 + b_row_lane_0;
                        int b_cb_6_1 = b_col_0 + 32;
                        unsigned int b_addr_6_1 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_6_1 * 128) + (unsigned int)(b_cb_6_1 ^ (b_row_6_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_6_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int a_cb_10 = a_col_0 + 64;
                        unsigned int a_addr_10 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_10 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_10)
                            : "memory");
                        int b_row_7 = 64 + nt_2 * 8 + b_row_lane_0;
                        int b_cb_7 = b_col_0 + 64;
                        unsigned int b_addr_7 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_7 * 128) + (unsigned int)(b_cb_7 ^ (b_row_7 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_7)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int a_cb_11 = a_col_0 + 96;
                        unsigned int a_addr_11 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_11 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_11)
                            : "memory");
                        int b_row_8 = 64 + nt_2 * 8 + b_row_lane_0;
                        int b_cb_8 = b_col_0 + 96;
                        unsigned int b_addr_8 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_8 * 128) + (unsigned int)(b_cb_8 ^ (b_row_8 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_8)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int w_idx_2 = q_stage_m_0 * 256 + 64 + (unsigned int)(nt_2 * 8) + (unsigned int)(tq_0 * 2);
                        float w0_2 = smem_w[w_idx_2];
                        float w1_2 = smem_w[w_idx_2 + 1];
                        float r0_2 = fmaxf(acc_0[0], 0.0f);
                        float r1_2 = fmaxf(acc_0[1], 0.0f);
                        float r2_2 = fmaxf(acc_0[2], 0.0f);
                        float r3_2 = fmaxf(acc_0[3], 0.0f);
                        partial0_1 += __fmaf_rn(r0_2, w0_2, r1_2 * w1_2);
                        partial1_1 += __fmaf_rn(r2_2, w0_2, r3_2 * w1_2);
                    }
                    float v0_1 = partial0_1 * scale0_0;
                    float v1_1 = partial1_1 * scale1_0;
                    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, v0_1, 1);
                    v0_1 = v0_1 + _shfl_xor_sync_8;
                    float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 1);
                    v1_1 = v1_1 + _shfl_xor_sync_9;
                    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, v0_1, 2);
                    v0_1 = v0_1 + _shfl_xor_sync_10;
                    float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 2);
                    v1_1 = v1_1 + _shfl_xor_sync_11;
                    int out0_1 = kv_offset_0 + 2 * logits_stride + quad_0;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_1)), "f"(v0_1), "r"((unsigned int)(tq_0 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_1 + 8))), "f"(v1_1), "r"((unsigned int)(tq_0 == 0)));
                    float partial0_2 = 0.0f;
                    float partial1_2 = 0.0f;
                    #pragma unroll
                    for (int nt_3 = 0; nt_3 < 4; nt_3++) {
                        acc_0[0] = 0.0f;
                        acc_0[1] = 0.0f;
                        acc_0[2] = 0.0f;
                        acc_0[3] = 0.0f;
                        int a_cb_12 = a_col_0;
                        unsigned int a_addr_12 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_12 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_12)
                            : "memory");
                        int b_row_9 = 96 + nt_3 * 8 + b_row_lane_0;
                        int b_cb_9 = b_col_0;
                        unsigned int b_addr_9 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_9 * 128) + (unsigned int)(b_cb_9 ^ (b_row_9 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_9)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_0[0]), "=f"(acc_0[1]), "=f"(acc_0[2]), "=f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_13 = a_col_0 + 32;
                        unsigned int a_addr_13 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_13 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_13)
                            : "memory");
                        int b_row_9_1 = 96 + nt_3 * 8 + b_row_lane_0;
                        int b_cb_9_1 = b_col_0 + 32;
                        unsigned int b_addr_9_1 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_9_1 * 128) + (unsigned int)(b_cb_9_1 ^ (b_row_9_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_9_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int a_cb_14 = a_col_0 + 64;
                        unsigned int a_addr_14 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_14 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_14)
                            : "memory");
                        int b_row_10 = 96 + nt_3 * 8 + b_row_lane_0;
                        int b_cb_10 = b_col_0 + 64;
                        unsigned int b_addr_10 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_10 * 128) + (unsigned int)(b_cb_10 ^ (b_row_10 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_10)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int a_cb_15 = a_col_0 + 96;
                        unsigned int a_addr_15 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_15 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_15)
                            : "memory");
                        int b_row_11 = 96 + nt_3 * 8 + b_row_lane_0;
                        int b_cb_11 = b_col_0 + 96;
                        unsigned int b_addr_11 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_11 * 128) + (unsigned int)(b_cb_11 ^ (b_row_11 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_11)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int w_idx_3 = q_stage_m_0 * 256 + 96 + (unsigned int)(nt_3 * 8) + (unsigned int)(tq_0 * 2);
                        float w0_3 = smem_w[w_idx_3];
                        float w1_3 = smem_w[w_idx_3 + 1];
                        float r0_3 = fmaxf(acc_0[0], 0.0f);
                        float r1_3 = fmaxf(acc_0[1], 0.0f);
                        float r2_3 = fmaxf(acc_0[2], 0.0f);
                        float r3_3 = fmaxf(acc_0[3], 0.0f);
                        partial0_2 += __fmaf_rn(r0_3, w0_3, r1_3 * w1_3);
                        partial1_2 += __fmaf_rn(r2_3, w0_3, r3_3 * w1_3);
                    }
                    float v0_2 = partial0_2 * scale0_0;
                    float v1_2 = partial1_2 * scale1_0;
                    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, v0_2, 1);
                    v0_2 = v0_2 + _shfl_xor_sync_12;
                    float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, v1_2, 1);
                    v1_2 = v1_2 + _shfl_xor_sync_13;
                    float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, v0_2, 2);
                    v0_2 = v0_2 + _shfl_xor_sync_14;
                    float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, v1_2, 2);
                    v1_2 = v1_2 + _shfl_xor_sync_15;
                    int out0_2 = kv_offset_0 + 3 * logits_stride + quad_0;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_2)), "f"(v0_2), "r"((unsigned int)(tq_0 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_2 + 8))), "f"(v1_2), "r"((unsigned int)(tq_0 == 0)));
                    float partial0_3 = 0.0f;
                    float partial1_3 = 0.0f;
                    #pragma unroll
                    for (int nt_4 = 0; nt_4 < 4; nt_4++) {
                        acc_0[0] = 0.0f;
                        acc_0[1] = 0.0f;
                        acc_0[2] = 0.0f;
                        acc_0[3] = 0.0f;
                        int a_cb_16 = a_col_0;
                        unsigned int a_addr_16 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_16 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_16)
                            : "memory");
                        int b_row_12 = 128 + nt_4 * 8 + b_row_lane_0;
                        int b_cb_12 = b_col_0;
                        unsigned int b_addr_12 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_12 * 128) + (unsigned int)(b_cb_12 ^ (b_row_12 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_12)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_0[0]), "=f"(acc_0[1]), "=f"(acc_0[2]), "=f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_17 = a_col_0 + 32;
                        unsigned int a_addr_17 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_17 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_17)
                            : "memory");
                        int b_row_12_1 = 128 + nt_4 * 8 + b_row_lane_0;
                        int b_cb_12_1 = b_col_0 + 32;
                        unsigned int b_addr_12_1 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_12_1 * 128) + (unsigned int)(b_cb_12_1 ^ (b_row_12_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_12_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int a_cb_18 = a_col_0 + 64;
                        unsigned int a_addr_18 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_18 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_18)
                            : "memory");
                        int b_row_13 = 128 + nt_4 * 8 + b_row_lane_0;
                        int b_cb_13 = b_col_0 + 64;
                        unsigned int b_addr_13 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_13 * 128) + (unsigned int)(b_cb_13 ^ (b_row_13 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_13)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int a_cb_19 = a_col_0 + 96;
                        unsigned int a_addr_19 = smem_kv_addr_0 + kv_stage_m_0 * 8192 + (unsigned int)(a_row_0 * 128) + (unsigned int)(a_cb_19 ^ (a_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_0[0]), "=r"(a_frag_0[1]), "=r"(a_frag_0[2]), "=r"(a_frag_0[3])
                            : "r"(a_addr_19)
                            : "memory");
                        int b_row_14 = 128 + nt_4 * 8 + b_row_lane_0;
                        int b_cb_14 = b_col_0 + 96;
                        unsigned int b_addr_14 = smem_q_addr + q_stage_m_0 * 20480 + (unsigned int)(b_row_14 * 128) + (unsigned int)(b_cb_14 ^ (b_row_14 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_0[0]), "=r"(b_frag_0[1])
                            : "r"(b_addr_14)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                            : "r"(a_frag_0[0]), "r"(a_frag_0[1]), "r"(a_frag_0[2]), "r"(a_frag_0[3]), "r"(b_frag_0[0]), "r"(b_frag_0[1]));
                        int w_idx_4 = q_stage_m_0 * 256 + 128 + (unsigned int)(nt_4 * 8) + (unsigned int)(tq_0 * 2);
                        float w0_4 = smem_w[w_idx_4];
                        float w1_4 = smem_w[w_idx_4 + 1];
                        float r0_4 = fmaxf(acc_0[0], 0.0f);
                        float r1_4 = fmaxf(acc_0[1], 0.0f);
                        float r2_4 = fmaxf(acc_0[2], 0.0f);
                        float r3_4 = fmaxf(acc_0[3], 0.0f);
                        partial0_3 += __fmaf_rn(r0_4, w0_4, r1_4 * w1_4);
                        partial1_3 += __fmaf_rn(r2_4, w0_4, r3_4 * w1_4);
                    }
                    float v0_3 = partial0_3 * scale0_0;
                    float v1_3 = partial1_3 * scale1_0;
                    float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, v0_3, 1);
                    v0_3 = v0_3 + _shfl_xor_sync_16;
                    float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, v1_3, 1);
                    v1_3 = v1_3 + _shfl_xor_sync_17;
                    float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, v0_3, 2);
                    v0_3 = v0_3 + _shfl_xor_sync_18;
                    float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, v1_3, 2);
                    v1_3 = v1_3 + _shfl_xor_sync_19;
                    int out0_3 = kv_offset_0 + 4 * logits_stride + quad_0;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_3)), "f"(v0_3), "r"((unsigned int)(tq_0 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_3 + 8))), "f"(v1_3), "r"((unsigned int)(tq_0 == 0)));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_empty_0_addr + (kv_stage_m_0) * 8);
                    kv_stage_m_0 += 1;
                    if (kv_stage_m_0 == 3) { kv_stage_m_0 = 0; _phase_kv_full_0 ^= 1; }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(q_empty_addr + (q_stage_m_0) * 8);
                q_stage_m_0 += 1;
                if (q_stage_m_0 == 2) { q_stage_m_0 = 0; _phase_q_full ^= 1; }
            }
        }
    }
    // ---- Role: math_1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // math_1_main
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
            float acc_1[4];
            int sm_1 = bid;
            int start_q_1 = schedule_meta[sm_1 * 2];
            int start_kv_1 = schedule_meta[sm_1 * 2 + 1] * 2;
            int end_q_1 = schedule_meta[sm_1 * 2 + 2];
            int end_kv_1 = schedule_meta[sm_1 * 2 + 3] * 2;
            int q_stop_1 = ((end_kv_1 > 0) ? end_q_1 + 1 : end_q_1);
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_kv_full_1 = 0;
            #pragma unroll 1
            for (int q_atom_3 = start_q_1; q_atom_3 < q_stop_1; q_atom_3++) {
                int q_idx_1 = q_atom_3;
                int atom_in_q_1 = 0;
                int tok_1 = q_idx_1 * 5 + atom_in_q_1 * 5;
                int ctx_last_1 = context_lens[q_idx_1 * 5 + 5 - 1];
                int num_kv_1 = (ctx_last_1 + 64 - 1) / 64;
                int lo_1 = ((q_atom_3 == start_q_1) ? start_kv_1 : 0);
                int hi_1 = ((q_atom_3 == end_q_1) ? end_kv_1 : num_kv_1);
                mbarrier_wait(q_full_addr + (q_stage_m_1) * 8, _phase_q_full_1);
                int n_tok_m_1 = 5;
                #pragma unroll 1
                for (int kv_idx_3 = lo_1; kv_idx_3 < hi_1; kv_idx_3 += 2) {
                    mbarrier_wait(kv_full_1_addr + (kv_stage_m_1) * 8, _phase_kv_full_1);
                    int sc_idx_1 = kv_stage_m_1 * 256 + (unsigned int)(w_in_g_1 * 16) + (unsigned int)quad_1;
                    float scale0_1 = smem_sc_1[sc_idx_1];
                    float scale1_1 = smem_sc_1[sc_idx_1 + 8];
                    int kv_offset_1 = tok_1 * logits_stride + (kv_idx_3 + 1) * 64 + w_in_g_1 * 16;
                    float partial0_4 = 0.0f;
                    float partial1_4 = 0.0f;
                    #pragma unroll
                    for (int nt_5 = 0; nt_5 < 4; nt_5++) {
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        int a_cb_20 = a_col_1;
                        unsigned int a_addr_20 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_20 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_20)
                            : "memory");
                        int b_row_15 = nt_5 * 8 + b_row_lane_1;
                        int b_cb_15 = b_col_1;
                        unsigned int b_addr_15 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_15 * 128) + (unsigned int)(b_cb_15 ^ (b_row_15 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_15)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_21 = a_col_1 + 32;
                        unsigned int a_addr_21 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_21 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_21)
                            : "memory");
                        int b_row_15_1 = nt_5 * 8 + b_row_lane_1;
                        int b_cb_15_1 = b_col_1 + 32;
                        unsigned int b_addr_15_1 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_15_1 * 128) + (unsigned int)(b_cb_15_1 ^ (b_row_15_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_15_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int a_cb_22 = a_col_1 + 64;
                        unsigned int a_addr_22 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_22 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_22)
                            : "memory");
                        int b_row_16 = nt_5 * 8 + b_row_lane_1;
                        int b_cb_16 = b_col_1 + 64;
                        unsigned int b_addr_16 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_16 * 128) + (unsigned int)(b_cb_16 ^ (b_row_16 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_16)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int a_cb_23 = a_col_1 + 96;
                        unsigned int a_addr_23 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_23 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_23)
                            : "memory");
                        int b_row_17 = nt_5 * 8 + b_row_lane_1;
                        int b_cb_17 = b_col_1 + 96;
                        unsigned int b_addr_17 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_17 * 128) + (unsigned int)(b_cb_17 ^ (b_row_17 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_17)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int w_idx_5 = q_stage_m_1 * 256 + (unsigned int)(nt_5 * 8) + (unsigned int)(tq_1 * 2);
                        float w0_5 = smem_w[w_idx_5];
                        float w1_5 = smem_w[w_idx_5 + 1];
                        float r0_5 = fmaxf(acc_1[0], 0.0f);
                        float r1_5 = fmaxf(acc_1[1], 0.0f);
                        float r2_5 = fmaxf(acc_1[2], 0.0f);
                        float r3_5 = fmaxf(acc_1[3], 0.0f);
                        partial0_4 += __fmaf_rn(r0_5, w0_5, r1_5 * w1_5);
                        partial1_4 += __fmaf_rn(r2_5, w0_5, r3_5 * w1_5);
                    }
                    float v0_4 = partial0_4 * scale0_1;
                    float v1_4 = partial1_4 * scale1_1;
                    float _shfl_xor_sync_0_1 = __shfl_xor_sync(0xFFFFFFFF, v0_4, 1);
                    v0_4 = v0_4 + _shfl_xor_sync_0_1;
                    float _shfl_xor_sync_1_1 = __shfl_xor_sync(0xFFFFFFFF, v1_4, 1);
                    v1_4 = v1_4 + _shfl_xor_sync_1_1;
                    float _shfl_xor_sync_2_1 = __shfl_xor_sync(0xFFFFFFFF, v0_4, 2);
                    v0_4 = v0_4 + _shfl_xor_sync_2_1;
                    float _shfl_xor_sync_3_1 = __shfl_xor_sync(0xFFFFFFFF, v1_4, 2);
                    v1_4 = v1_4 + _shfl_xor_sync_3_1;
                    int out0_4 = kv_offset_1 + quad_1;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_4)), "f"(v0_4), "r"((unsigned int)(tq_1 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_4 + 8))), "f"(v1_4), "r"((unsigned int)(tq_1 == 0)));
                    float partial0_4_1 = 0.0f;
                    float partial1_4_1 = 0.0f;
                    #pragma unroll
                    for (int nt_6 = 0; nt_6 < 4; nt_6++) {
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        int a_cb_24 = a_col_1;
                        unsigned int a_addr_24 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_24 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_24)
                            : "memory");
                        int b_row_18 = 32 + nt_6 * 8 + b_row_lane_1;
                        int b_cb_18 = b_col_1;
                        unsigned int b_addr_18 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_18 * 128) + (unsigned int)(b_cb_18 ^ (b_row_18 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_18)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_25 = a_col_1 + 32;
                        unsigned int a_addr_25 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_25 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_25)
                            : "memory");
                        int b_row_18_1 = 32 + nt_6 * 8 + b_row_lane_1;
                        int b_cb_18_1 = b_col_1 + 32;
                        unsigned int b_addr_18_1 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_18_1 * 128) + (unsigned int)(b_cb_18_1 ^ (b_row_18_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_18_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int a_cb_26 = a_col_1 + 64;
                        unsigned int a_addr_26 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_26 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_26)
                            : "memory");
                        int b_row_19 = 32 + nt_6 * 8 + b_row_lane_1;
                        int b_cb_19 = b_col_1 + 64;
                        unsigned int b_addr_19 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_19 * 128) + (unsigned int)(b_cb_19 ^ (b_row_19 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_19)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int a_cb_27 = a_col_1 + 96;
                        unsigned int a_addr_27 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_27 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_27)
                            : "memory");
                        int b_row_20 = 32 + nt_6 * 8 + b_row_lane_1;
                        int b_cb_20 = b_col_1 + 96;
                        unsigned int b_addr_20 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_20 * 128) + (unsigned int)(b_cb_20 ^ (b_row_20 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_20)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int w_idx_6 = q_stage_m_1 * 256 + 32 + (unsigned int)(nt_6 * 8) + (unsigned int)(tq_1 * 2);
                        float w0_6 = smem_w[w_idx_6];
                        float w1_6 = smem_w[w_idx_6 + 1];
                        float r0_6 = fmaxf(acc_1[0], 0.0f);
                        float r1_6 = fmaxf(acc_1[1], 0.0f);
                        float r2_6 = fmaxf(acc_1[2], 0.0f);
                        float r3_6 = fmaxf(acc_1[3], 0.0f);
                        partial0_4_1 += __fmaf_rn(r0_6, w0_6, r1_6 * w1_6);
                        partial1_4_1 += __fmaf_rn(r2_6, w0_6, r3_6 * w1_6);
                    }
                    float v0_4_1 = partial0_4_1 * scale0_1;
                    float v1_4_1 = partial1_4_1 * scale1_1;
                    float _shfl_xor_sync_4_1 = __shfl_xor_sync(0xFFFFFFFF, v0_4_1, 1);
                    v0_4_1 = v0_4_1 + _shfl_xor_sync_4_1;
                    float _shfl_xor_sync_5_1 = __shfl_xor_sync(0xFFFFFFFF, v1_4_1, 1);
                    v1_4_1 = v1_4_1 + _shfl_xor_sync_5_1;
                    float _shfl_xor_sync_6_1 = __shfl_xor_sync(0xFFFFFFFF, v0_4_1, 2);
                    v0_4_1 = v0_4_1 + _shfl_xor_sync_6_1;
                    float _shfl_xor_sync_7_1 = __shfl_xor_sync(0xFFFFFFFF, v1_4_1, 2);
                    v1_4_1 = v1_4_1 + _shfl_xor_sync_7_1;
                    int out0_4_1 = kv_offset_1 + logits_stride + quad_1;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_4_1)), "f"(v0_4_1), "r"((unsigned int)(tq_1 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_4_1 + 8))), "f"(v1_4_1), "r"((unsigned int)(tq_1 == 0)));
                    float partial0_5 = 0.0f;
                    float partial1_5 = 0.0f;
                    #pragma unroll
                    for (int nt_7 = 0; nt_7 < 4; nt_7++) {
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        int a_cb_28 = a_col_1;
                        unsigned int a_addr_28 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_28 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_28)
                            : "memory");
                        int b_row_21 = 64 + nt_7 * 8 + b_row_lane_1;
                        int b_cb_21 = b_col_1;
                        unsigned int b_addr_21 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_21 * 128) + (unsigned int)(b_cb_21 ^ (b_row_21 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_21)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_29 = a_col_1 + 32;
                        unsigned int a_addr_29 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_29 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_29)
                            : "memory");
                        int b_row_21_1 = 64 + nt_7 * 8 + b_row_lane_1;
                        int b_cb_21_1 = b_col_1 + 32;
                        unsigned int b_addr_21_1 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_21_1 * 128) + (unsigned int)(b_cb_21_1 ^ (b_row_21_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_21_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int a_cb_30 = a_col_1 + 64;
                        unsigned int a_addr_30 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_30 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_30)
                            : "memory");
                        int b_row_22 = 64 + nt_7 * 8 + b_row_lane_1;
                        int b_cb_22 = b_col_1 + 64;
                        unsigned int b_addr_22 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_22 * 128) + (unsigned int)(b_cb_22 ^ (b_row_22 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_22)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int a_cb_31 = a_col_1 + 96;
                        unsigned int a_addr_31 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_31 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_31)
                            : "memory");
                        int b_row_23 = 64 + nt_7 * 8 + b_row_lane_1;
                        int b_cb_23 = b_col_1 + 96;
                        unsigned int b_addr_23 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_23 * 128) + (unsigned int)(b_cb_23 ^ (b_row_23 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_23)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int w_idx_7 = q_stage_m_1 * 256 + 64 + (unsigned int)(nt_7 * 8) + (unsigned int)(tq_1 * 2);
                        float w0_7 = smem_w[w_idx_7];
                        float w1_7 = smem_w[w_idx_7 + 1];
                        float r0_7 = fmaxf(acc_1[0], 0.0f);
                        float r1_7 = fmaxf(acc_1[1], 0.0f);
                        float r2_7 = fmaxf(acc_1[2], 0.0f);
                        float r3_7 = fmaxf(acc_1[3], 0.0f);
                        partial0_5 += __fmaf_rn(r0_7, w0_7, r1_7 * w1_7);
                        partial1_5 += __fmaf_rn(r2_7, w0_7, r3_7 * w1_7);
                    }
                    float v0_5 = partial0_5 * scale0_1;
                    float v1_5 = partial1_5 * scale1_1;
                    float _shfl_xor_sync_8_1 = __shfl_xor_sync(0xFFFFFFFF, v0_5, 1);
                    v0_5 = v0_5 + _shfl_xor_sync_8_1;
                    float _shfl_xor_sync_9_1 = __shfl_xor_sync(0xFFFFFFFF, v1_5, 1);
                    v1_5 = v1_5 + _shfl_xor_sync_9_1;
                    float _shfl_xor_sync_10_1 = __shfl_xor_sync(0xFFFFFFFF, v0_5, 2);
                    v0_5 = v0_5 + _shfl_xor_sync_10_1;
                    float _shfl_xor_sync_11_1 = __shfl_xor_sync(0xFFFFFFFF, v1_5, 2);
                    v1_5 = v1_5 + _shfl_xor_sync_11_1;
                    int out0_5 = kv_offset_1 + 2 * logits_stride + quad_1;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_5)), "f"(v0_5), "r"((unsigned int)(tq_1 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_5 + 8))), "f"(v1_5), "r"((unsigned int)(tq_1 == 0)));
                    float partial0_6 = 0.0f;
                    float partial1_6 = 0.0f;
                    #pragma unroll
                    for (int nt_8 = 0; nt_8 < 4; nt_8++) {
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        int a_cb_32 = a_col_1;
                        unsigned int a_addr_32 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_32 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_32)
                            : "memory");
                        int b_row_24 = 96 + nt_8 * 8 + b_row_lane_1;
                        int b_cb_24 = b_col_1;
                        unsigned int b_addr_24 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_24 * 128) + (unsigned int)(b_cb_24 ^ (b_row_24 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_24)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_33 = a_col_1 + 32;
                        unsigned int a_addr_33 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_33 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_33)
                            : "memory");
                        int b_row_24_1 = 96 + nt_8 * 8 + b_row_lane_1;
                        int b_cb_24_1 = b_col_1 + 32;
                        unsigned int b_addr_24_1 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_24_1 * 128) + (unsigned int)(b_cb_24_1 ^ (b_row_24_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_24_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int a_cb_34 = a_col_1 + 64;
                        unsigned int a_addr_34 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_34 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_34)
                            : "memory");
                        int b_row_25 = 96 + nt_8 * 8 + b_row_lane_1;
                        int b_cb_25 = b_col_1 + 64;
                        unsigned int b_addr_25 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_25 * 128) + (unsigned int)(b_cb_25 ^ (b_row_25 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_25)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int a_cb_35 = a_col_1 + 96;
                        unsigned int a_addr_35 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_35 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_35)
                            : "memory");
                        int b_row_26 = 96 + nt_8 * 8 + b_row_lane_1;
                        int b_cb_26 = b_col_1 + 96;
                        unsigned int b_addr_26 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_26 * 128) + (unsigned int)(b_cb_26 ^ (b_row_26 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_26)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int w_idx_8 = q_stage_m_1 * 256 + 96 + (unsigned int)(nt_8 * 8) + (unsigned int)(tq_1 * 2);
                        float w0_8 = smem_w[w_idx_8];
                        float w1_8 = smem_w[w_idx_8 + 1];
                        float r0_8 = fmaxf(acc_1[0], 0.0f);
                        float r1_8 = fmaxf(acc_1[1], 0.0f);
                        float r2_8 = fmaxf(acc_1[2], 0.0f);
                        float r3_8 = fmaxf(acc_1[3], 0.0f);
                        partial0_6 += __fmaf_rn(r0_8, w0_8, r1_8 * w1_8);
                        partial1_6 += __fmaf_rn(r2_8, w0_8, r3_8 * w1_8);
                    }
                    float v0_6 = partial0_6 * scale0_1;
                    float v1_6 = partial1_6 * scale1_1;
                    float _shfl_xor_sync_12_1 = __shfl_xor_sync(0xFFFFFFFF, v0_6, 1);
                    v0_6 = v0_6 + _shfl_xor_sync_12_1;
                    float _shfl_xor_sync_13_1 = __shfl_xor_sync(0xFFFFFFFF, v1_6, 1);
                    v1_6 = v1_6 + _shfl_xor_sync_13_1;
                    float _shfl_xor_sync_14_1 = __shfl_xor_sync(0xFFFFFFFF, v0_6, 2);
                    v0_6 = v0_6 + _shfl_xor_sync_14_1;
                    float _shfl_xor_sync_15_1 = __shfl_xor_sync(0xFFFFFFFF, v1_6, 2);
                    v1_6 = v1_6 + _shfl_xor_sync_15_1;
                    int out0_6 = kv_offset_1 + 3 * logits_stride + quad_1;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_6)), "f"(v0_6), "r"((unsigned int)(tq_1 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_6 + 8))), "f"(v1_6), "r"((unsigned int)(tq_1 == 0)));
                    float partial0_7 = 0.0f;
                    float partial1_7 = 0.0f;
                    #pragma unroll
                    for (int nt_9 = 0; nt_9 < 4; nt_9++) {
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        int a_cb_36 = a_col_1;
                        unsigned int a_addr_36 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_36 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_36)
                            : "memory");
                        int b_row_27 = 128 + nt_9 * 8 + b_row_lane_1;
                        int b_cb_27 = b_col_1;
                        unsigned int b_addr_27 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_27 * 128) + (unsigned int)(b_cb_27 ^ (b_row_27 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_27)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_37 = a_col_1 + 32;
                        unsigned int a_addr_37 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_37 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_37)
                            : "memory");
                        int b_row_27_1 = 128 + nt_9 * 8 + b_row_lane_1;
                        int b_cb_27_1 = b_col_1 + 32;
                        unsigned int b_addr_27_1 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_27_1 * 128) + (unsigned int)(b_cb_27_1 ^ (b_row_27_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_27_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int a_cb_38 = a_col_1 + 64;
                        unsigned int a_addr_38 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_38 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_38)
                            : "memory");
                        int b_row_28 = 128 + nt_9 * 8 + b_row_lane_1;
                        int b_cb_28 = b_col_1 + 64;
                        unsigned int b_addr_28 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_28 * 128) + (unsigned int)(b_cb_28 ^ (b_row_28 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_28)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int a_cb_39 = a_col_1 + 96;
                        unsigned int a_addr_39 = smem_kv_addr_1 + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_39 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_39)
                            : "memory");
                        int b_row_29 = 128 + nt_9 * 8 + b_row_lane_1;
                        int b_cb_29 = b_col_1 + 96;
                        unsigned int b_addr_29 = smem_q_addr + q_stage_m_1 * 20480 + (unsigned int)(b_row_29 * 128) + (unsigned int)(b_cb_29 ^ (b_row_29 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_29)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]));
                        int w_idx_9 = q_stage_m_1 * 256 + 128 + (unsigned int)(nt_9 * 8) + (unsigned int)(tq_1 * 2);
                        float w0_9 = smem_w[w_idx_9];
                        float w1_9 = smem_w[w_idx_9 + 1];
                        float r0_9 = fmaxf(acc_1[0], 0.0f);
                        float r1_9 = fmaxf(acc_1[1], 0.0f);
                        float r2_9 = fmaxf(acc_1[2], 0.0f);
                        float r3_9 = fmaxf(acc_1[3], 0.0f);
                        partial0_7 += __fmaf_rn(r0_9, w0_9, r1_9 * w1_9);
                        partial1_7 += __fmaf_rn(r2_9, w0_9, r3_9 * w1_9);
                    }
                    float v0_7 = partial0_7 * scale0_1;
                    float v1_7 = partial1_7 * scale1_1;
                    float _shfl_xor_sync_16_1 = __shfl_xor_sync(0xFFFFFFFF, v0_7, 1);
                    v0_7 = v0_7 + _shfl_xor_sync_16_1;
                    float _shfl_xor_sync_17_1 = __shfl_xor_sync(0xFFFFFFFF, v1_7, 1);
                    v1_7 = v1_7 + _shfl_xor_sync_17_1;
                    float _shfl_xor_sync_18_1 = __shfl_xor_sync(0xFFFFFFFF, v0_7, 2);
                    v0_7 = v0_7 + _shfl_xor_sync_18_1;
                    float _shfl_xor_sync_19_1 = __shfl_xor_sync(0xFFFFFFFF, v1_7, 2);
                    v1_7 = v1_7 + _shfl_xor_sync_19_1;
                    int out0_7 = kv_offset_1 + 4 * logits_stride + quad_1;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_7)), "f"(v0_7), "r"((unsigned int)(tq_1 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_7 + 8))), "f"(v1_7), "r"((unsigned int)(tq_1 == 0)));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_empty_1_addr + (kv_stage_m_1) * 8);
                    kv_stage_m_1 += 1;
                    if (kv_stage_m_1 == 3) { kv_stage_m_1 = 0; _phase_kv_full_1 ^= 1; }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(q_empty_addr + (q_stage_m_1) * 8);
                q_stage_m_1 += 1;
                if (q_stage_m_1 == 2) { q_stage_m_1 = 0; _phase_q_full_1 ^= 1; }
            }
        }
    }
    // ---- Role: tma_0 ----
    if (warp == 8) {
        { // tma_0_main
            unsigned int q_stage_0 = 0;
            unsigned int q_phase_0 = 1;
            unsigned int kv_stage_0 = 0;
            unsigned int kv_phase_0 = 1;
            int sm_2 = bid;
            int start_q_2 = schedule_meta[sm_2 * 2];
            int start_kv_2 = schedule_meta[sm_2 * 2 + 1] * 2;
            int end_q_2 = schedule_meta[sm_2 * 2 + 2];
            int end_kv_2 = schedule_meta[sm_2 * 2 + 3] * 2;
            int q_stop_2 = ((end_kv_2 > 0) ? end_q_2 + 1 : end_q_2);
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_kv_empty_0 = 1;
            #pragma unroll 1
            for (int q_atom_0 = start_q_2; q_atom_0 < q_stop_2; q_atom_0++) {
                int q_idx_2 = q_atom_0;
                int atom_in_q_2 = 0;
                int tok_2 = q_idx_2 * 5 + atom_in_q_2 * 5;
                int ctx_last_2 = context_lens[q_idx_2 * 5 + 5 - 1];
                int num_kv_2 = (ctx_last_2 + 64 - 1) / 64;
                int lo_2 = ((q_atom_0 == start_q_2) ? start_kv_2 : 0);
                int hi_2 = ((q_atom_0 == end_q_2) ? end_kv_2 : num_kv_2);
                mbarrier_wait(q_empty_addr + (q_stage_0) * 8, _phase_q_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + (q_stage_0) * 8, 21120);
                    tma_2d_gmem2smem(smem_q_addr + q_stage_0 * 20480, (&Q), 0, tok_2 * 32, q_full_addr + (q_stage_0) * 8);
                    tma_2d_gmem2smem(smem_w_addr + q_stage_0 * 1024, (&Weights), 0, tok_2, q_full_addr + (q_stage_0) * 8);
                }
                q_stage_0 += 1;
                if (q_stage_0 == 2) { q_stage_0 = 0; _phase_q_empty ^= 1; }
                int bt_row_0 = q_idx_2 * block_table_stride;
                int bt_lane_0 = lane;
                int bt_ptr_0 = 32;
                int bt_cache_0 = 0;
                #pragma unroll 1
                for (int kv_idx_0 = lo_2; kv_idx_0 < hi_2; kv_idx_0 += 2) {
                    int tile_0 = kv_idx_0;
                    if (bt_ptr_0 == 32) {
                        bt_ptr_0 = 0;
                        int bt_tile_0 = tile_0 + bt_lane_0 * 2;
                        bt_cache_0 = 0;
                        if (bt_tile_0 < num_kv_2) {
                            bt_cache_0 = block_table[bt_row_0 + bt_tile_0 / 2];
                        }
                    }
                    int kv_block_0 = __shfl_sync(0xFFFFFFFF, bt_cache_0, bt_ptr_0);
                    bt_ptr_0 += 1;
                    int page_off_0 = tile_0 % 2 * 64;
                    mbarrier_wait(kv_empty_0_addr + (kv_stage_0) * 8, _phase_kv_empty_0);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_0_addr + (kv_stage_0) * 8, 8448);
                        tma_3d_gmem2smem(smem_kv_addr_0 + kv_stage_0 * 8192, (&KV), 0, page_off_0, kv_block_0, kv_full_0_addr + (kv_stage_0) * 8);
                        tma_2d_gmem2smem(smem_sc_addr_0 + kv_stage_0 * 1024, (&KV_scales), 4096 + page_off_0, kv_block_0, kv_full_0_addr + (kv_stage_0) * 8);
                    }
                    kv_stage_0 += 1;
                    if (kv_stage_0 == 3) { kv_stage_0 = 0; _phase_kv_empty_0 ^= 1; }
                }
            }
        }
    }
    // ---- Role: tma_1 ----
    if (warp == 9) {
        { // tma_1_main
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
            unsigned int _phase_kv_empty_1 = 1;
            #pragma unroll 1
            for (int q_atom_1 = start_q_3; q_atom_1 < q_stop_3; q_atom_1++) {
                int q_idx_3 = q_atom_1;
                int atom_in_q_3 = 0;
                int tok_3 = q_idx_3 * 5 + atom_in_q_3 * 5;
                int ctx_last_3 = context_lens[q_idx_3 * 5 + 5 - 1];
                int num_kv_3 = (ctx_last_3 + 64 - 1) / 64;
                int lo_3 = ((q_atom_1 == start_q_3) ? start_kv_3 : 0);
                int hi_3 = ((q_atom_1 == end_q_3) ? end_kv_3 : num_kv_3);
                int bt_row_1 = q_idx_3 * block_table_stride;
                int bt_lane_1 = lane;
                int bt_ptr_1 = 32;
                int bt_cache_1 = 0;
                #pragma unroll 1
                for (int kv_idx_1 = lo_3; kv_idx_1 < hi_3; kv_idx_1 += 2) {
                    int tile_1 = kv_idx_1 + 1;
                    if (bt_ptr_1 == 32) {
                        bt_ptr_1 = 0;
                        int bt_tile_1 = tile_1 + bt_lane_1 * 2;
                        bt_cache_1 = 0;
                        if (bt_tile_1 < num_kv_3) {
                            bt_cache_1 = block_table[bt_row_1 + bt_tile_1 / 2];
                        }
                    }
                    int kv_block_1 = __shfl_sync(0xFFFFFFFF, bt_cache_1, bt_ptr_1);
                    bt_ptr_1 += 1;
                    int page_off_1 = tile_1 % 2 * 64;
                    mbarrier_wait(kv_empty_1_addr + (kv_stage_1) * 8, _phase_kv_empty_1);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_1_addr + (kv_stage_1) * 8, 8448);
                        tma_3d_gmem2smem(smem_kv_addr_1 + kv_stage_1 * 8192, (&KV), 0, page_off_1, kv_block_1, kv_full_1_addr + (kv_stage_1) * 8);
                        tma_2d_gmem2smem(smem_sc_addr_1 + kv_stage_1 * 1024, (&KV_scales), 4096 + page_off_1, kv_block_1, kv_full_1_addr + (kv_stage_1) * 8);
                    }
                    kv_stage_1 += 1;
                    if (kv_stage_1 == 3) { kv_stage_1 = 0; _phase_kv_empty_1 ^= 1; }
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
