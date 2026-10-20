/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_deepgemm_dense_mqa_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 192
#define TMEM_TMEM_ACC_OFFSET 0
#define NUM_Q_PIPE_STAGES 3
#define NUM_KV_PIPE_STAGES 5
#define NUM_TMEM_PIPE_STAGES 3
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_KV_OFF 25600
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_KV_SCALES_OFF 189440
#define SMEM_SMEM_KV_SCALES_STAGE_BYTES 1024
#define SMEM_SMEM_KV_SCALES_STRIDE 1024
#define SMEM_SMEM_WEIGHTS_OFF 194560
#define SMEM_SMEM_WEIGHTS_STAGE_BYTES 256
#define SMEM_SMEM_WEIGHTS_STRIDE 256
#define SMEM_SCHED_PREFIX_OFF 25600
#define SMEM_SCHED_PREFIX_STAGE_BYTES 16384
#define SMEM_SCHED_PREFIX_STRIDE 16384
#define SMEM_SCHED_WARP_SUMS_OFF 41984
#define SMEM_SCHED_WARP_SUMS_STAGE_BYTES 128
#define SMEM_SCHED_WARP_SUMS_STRIDE 128
#define SMEM_SCHED_BOUNDS_OFF 195328
#define SMEM_SCHED_BOUNDS_STAGE_BYTES 16
#define SMEM_SCHED_BOUNDS_STRIDE 16
#define SMEM_TOTAL 195456
#define K_NEXT_N 1
#define K_NEXT_N_ATOM 1
#define K_NUM_NEXT_N_ATOMS 1
#define K_CHUNK_SPLITS 1048576
#define K_NUM_HEADS 64
#define K_UMMA_N 64
#define K_Q_TILE_ROWS 64
#define K_Q_TX_BYTES 8448
#define K_BLOCK_KV 64
#define K_NUM_BLOCKS_PER_SPLIT 4
#define K_SCALE_TAIL_F32 2048
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_deepgemm_dense_mqa_56340b07d9523ccfb400(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KV_scales, const __grid_constant__ CUtensorMap Weights, float* __restrict__ Logits, int* __restrict__ context_lens, int* __restrict__ block_table, int block_table_stride, int batch_size, int num_sms, int stride_logits, unsigned int* __restrict__ sched_counters)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 24)
    #define kv_full_addr (mbar_base + 48)
    #define kv_empty_addr (mbar_base + 88)
    #define umma_full_addr (mbar_base + 128)
    #define umma_empty_addr (mbar_base + 152)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_Q_OFF);
    const int smem_q_addr = smem + SMEM_SMEM_Q_OFF;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_KV_OFF);
    const int smem_kv_addr = smem + SMEM_SMEM_KV_OFF;
    float* smem_kv_scales = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_KV_SCALES_OFF);
    const int smem_kv_scales_addr = smem + SMEM_SMEM_KV_SCALES_OFF;
    float* smem_weights = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_WEIGHTS_OFF);
    const int smem_weights_addr = smem + SMEM_SMEM_WEIGHTS_OFF;
    int* sched_prefix = reinterpret_cast<int*>(smem_raw + SMEM_SCHED_PREFIX_OFF);
    const int sched_prefix_addr = smem + SMEM_SCHED_PREFIX_OFF;
    int* sched_warp_sums = reinterpret_cast<int*>(smem_raw + SMEM_SCHED_WARP_SUMS_OFF);
    const int sched_warp_sums_addr = smem + SMEM_SCHED_WARP_SUMS_OFF;
    int* sched_bounds = reinterpret_cast<int*>(smem_raw + SMEM_SCHED_BOUNDS_OFF);
    const int sched_bounds_addr = smem + SMEM_SCHED_BOUNDS_OFF;
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Weights))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV_scales))) : "memory"); }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 22 barriers)
    // Mbarriers at smem_raw[0..176)

    if (warp == 9) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // q_empty: 3 barriers, init_count=288
            mbarrier_init(smem + 24, 288);
            mbarrier_init(smem + 32, 288);
            mbarrier_init(smem + 40, 288);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 5 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // kv_empty: 5 barriers, init_count=256
            mbarrier_init(smem + 88, 256);
            mbarrier_init(smem + 96, 256);
            mbarrier_init(smem + 104, 256);
            mbarrier_init(smem + 112, 256);
            mbarrier_init(smem + 120, 256);
            // --- pipeline 'tmem_pipe' ---
            // umma_full: 3 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // umma_empty: 3 barriers, init_count=128
            mbarrier_init(smem + 152, 128);
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 176);
    if (warp == 10) {
        int _tmem_hold = smem + 176;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: math ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // math_main
            unsigned int wg_idx = make_warp_uniform(warp / 4);
            int math_thread_idx = wg_idx * 128 + (unsigned int)(warp % 4 * 32 + lane);
            unsigned int mt_q_stage = 0;
            unsigned int mt_kv_stage = 0;
            unsigned int mt_tmem_stage = wg_idx;
            int rg_smax = (block_table_stride * K_BLOCK_KV + 255) / 256;
            int rg_units = batch_size * rg_smax;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_umma_full = 0;
            #pragma unroll 1
            for (int u = bid; u < rg_units; u += num_bids) {
                int ru_u = u;
                int ru_req = ru_u / rg_smax;
                int ru_s = ru_u % rg_smax;
                int ctx_last = context_lens[ru_req * K_NEXT_N + (K_NEXT_N - 1)];
                int num_kv_splits = (ctx_last + 256 - 1) / 256;
                if (ru_s < num_kv_splits) {
                    int tok_base = ru_req * K_NEXT_N;
                    mbarrier_wait(q_full_addr + (mt_q_stage) * 8, _phase_q_full);
                    float weights_reg[K_Q_TILE_ROWS];
                    int w_base = mt_q_stage * (unsigned int)K_Q_TILE_ROWS;
                    #pragma unroll
                    for (int wi = 0; wi < K_Q_TILE_ROWS; wi++) {
                        weights_reg[wi] = smem_weights[w_base + wi];
                    }
                    int ctx_qi[K_NEXT_N_ATOM];
                    #pragma unroll
                    for (int qp = 0; qp < K_NEXT_N_ATOM; qp++) {
                        ctx_qi[qp] = context_lens[tok_base + qp];
                    }
                    mbarrier_wait(kv_full_addr + (mt_kv_stage) * 8, _phase_kv_full);
                    int sc_base = mt_kv_stage * 256;
                    float scale_kv = smem_kv_scales[sc_base + math_thread_idx];
                    mbarrier_wait(umma_full_addr + (mt_tmem_stage) * 8, _phase_umma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_empty_addr + (mt_kv_stage) * 8);
                    mt_kv_stage += 1;
                    if (mt_kv_stage == 5) { mt_kv_stage = 0; _phase_kv_full ^= 1; }
                    int kv_pos = ru_s * 256 + math_thread_idx;
                    #pragma unroll
                    for (int qi = 0; qi < K_NEXT_N_ATOM; qi++) {
                        float _tmem_load_0[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(taddr + mt_tmem_stage * (unsigned int)K_UMMA_N + (unsigned int)(qi * K_NUM_HEADS) + (unsigned int)(warp % 4 * 32 << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                            : "r"(taddr + mt_tmem_stage * (unsigned int)K_UMMA_N + (unsigned int)(qi * K_NUM_HEADS) + (unsigned int)(warp % 4 * 32 << 16) + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (qi == K_NEXT_N_ATOM - 1) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(umma_empty_addr + (mt_tmem_stage) * 8);
                        }
                        float _relu_wsum_0;
                        {
                            float2 _sum0 = make_float2(0.0f, 0.0f);
                            float2 _sum1 = make_float2(0.0f, 0.0f);
                            #pragma unroll
                            for (int _j = 0; _j < 64; _j += 4) {
                                asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                    "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                    "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                    "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                    "mov.b64 _pb, {%3, %4};\n\t"
                                    "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                    : "+l"(*(unsigned long long*)&_sum0) : "f"(_tmem_load_0[0 + _j + 0]), "f"(_tmem_load_0[0 + _j + 1]), "f"(weights_reg[qi * K_NUM_HEADS + _j + 0]), "f"(weights_reg[qi * K_NUM_HEADS + _j + 1]));
                                asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                    "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                    "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                    "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                    "mov.b64 _pb, {%3, %4};\n\t"
                                    "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                    : "+l"(*(unsigned long long*)&_sum1) : "f"(_tmem_load_0[0 + _j + 2]), "f"(_tmem_load_0[0 + _j + 3]), "f"(weights_reg[qi * K_NUM_HEADS + _j + 2]), "f"(weights_reg[qi * K_NUM_HEADS + _j + 3]));
                            }
                            float2 _sum;
                            asm("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_sum) : "l"(*(const unsigned long long*)&_sum0), "l"(*(const unsigned long long*)&_sum1));
                            _relu_wsum_0 = (_sum.x + _sum.y) * 0.5f;
                        }
                        float scaled = _relu_wsum_0 * scale_kv;
                        float result = ((kv_pos < ctx_qi[qi]) ? scaled : -CAKE_INF);
                        int out_elem = (tok_base + qi) * stride_logits + kv_pos;
                        *(reinterpret_cast<float*>(Logits + out_elem) + (0)) = result;
                    }
                    mt_tmem_stage += 1;
                    if (mt_tmem_stage == 3) { mt_tmem_stage = 0; _phase_umma_full ^= 1; }
                    mt_tmem_stage += 1;
                    if (mt_tmem_stage == 3) { mt_tmem_stage = 0; _phase_umma_full ^= 1; }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(q_empty_addr + (mt_q_stage) * 8);
                    mt_q_stage += 1;
                    if (mt_q_stage == 3) { mt_q_stage = 0; _phase_q_full ^= 1; }
                }
            }
        }
    // ---- Role: load_q ----
    } else if (warp == 8) {
        { // load_q_main
            unsigned int lq_stage = 0;
            int rg_smax_1 = (block_table_stride * K_BLOCK_KV + 255) / 256;
            int rg_units_1 = batch_size * rg_smax_1;
            unsigned int _phase_q_empty = 1;
            #pragma unroll 1
            for (int u_1 = bid; u_1 < rg_units_1; u_1 += num_bids) {
                int ru_u_1 = u_1;
                int ru_req_1 = ru_u_1 / rg_smax_1;
                int ru_s_1 = ru_u_1 % rg_smax_1;
                int ctx_last_1 = context_lens[ru_req_1 * K_NEXT_N + (K_NEXT_N - 1)];
                int num_kv_splits_1 = (ctx_last_1 + 256 - 1) / 256;
                if (ru_s_1 < num_kv_splits_1) {
                    int tok_base_1 = ru_req_1 * K_NEXT_N;
                    mbarrier_wait(q_empty_addr + (lq_stage) * 8, _phase_q_empty);
                    if (elect_sync()) {
                        tma_2d_gmem2smem(smem_q_addr + lq_stage * 8192, (&Q), 0, tok_base_1 * K_NUM_HEADS, q_full_addr + (lq_stage) * 8);
                        tma_2d_gmem2smem(smem_weights_addr + lq_stage * 256, (&Weights), 0, tok_base_1, q_full_addr + (lq_stage) * 8);
                        mbarrier_arrive_expect_tx(q_full_addr + (lq_stage) * 8, K_Q_TX_BYTES);
                    }
                    lq_stage += 1;
                    if (lq_stage == 3) { lq_stage = 0; _phase_q_empty ^= 1; }
                }
            }
        }
    // ---- Role: load_kv ----
    } else if (warp == 9) {
        { // load_kv_main
            unsigned int lk_stage = 0;
            int rg_smax_2 = (block_table_stride * K_BLOCK_KV + 255) / 256;
            int rg_units_2 = batch_size * rg_smax_2;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (int u_2 = bid; u_2 < rg_units_2; u_2 += num_bids) {
                int ru_u_2 = u_2;
                int ru_req_2 = ru_u_2 / rg_smax_2;
                int ru_s_2 = ru_u_2 % rg_smax_2;
                int ctx_last_2 = context_lens[ru_req_2 * K_NEXT_N + (K_NEXT_N - 1)];
                int num_kv_splits_2 = (ctx_last_2 + 256 - 1) / 256;
                int lk_page_off = ru_s_2 * K_NUM_BLOCKS_PER_SPLIT + lane;
                int lk_word = 0;
                if (lane < K_NUM_BLOCKS_PER_SPLIT) {
                    if (lk_page_off < block_table_stride) {
                        lk_word = block_table[ru_req_2 * block_table_stride + lk_page_off];
                    }
                }
                if (ru_s_2 < num_kv_splits_2) {
                    int lk_num_pages = (ctx_last_2 + K_BLOCK_KV - 1) / K_BLOCK_KV;
                    int lk_coord = ((lk_page_off < lk_num_pages) ? lk_word : 0);
                    int page_idx_reg[K_NUM_BLOCKS_PER_SPLIT];
                    #pragma unroll
                    for (int i = 0; i < K_NUM_BLOCKS_PER_SPLIT; i++) {
                        int _shfl_0 = __shfl_sync(0xFFFFFFFF, lk_coord, i);
                        page_idx_reg[i] = _shfl_0;
                    }
                    mbarrier_wait(kv_empty_addr + (lk_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        #pragma unroll
                        for (int i_1 = 0; i_1 < K_NUM_BLOCKS_PER_SPLIT; i_1++) {
                            tma_3d_gmem2smem(smem_kv_addr + lk_stage * 32768 + (unsigned int)(i_1 * (K_BLOCK_KV * 128)), (&KV), 0, 0, page_idx_reg[i_1], kv_full_addr + (lk_stage) * 8);
                            tma_2d_gmem2smem(smem_kv_scales_addr + lk_stage * 1024 + (unsigned int)(i_1 * (K_BLOCK_KV * 4)), (&KV_scales), K_SCALE_TAIL_F32, page_idx_reg[i_1], kv_full_addr + (lk_stage) * 8);
                        }
                        mbarrier_arrive_expect_tx(kv_full_addr + (lk_stage) * 8, 33792);
                    }
                    lk_stage += 1;
                    if (lk_stage == 5) { lk_stage = 0; _phase_kv_empty ^= 1; }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 10) {
        { // mma_main
            unsigned int mm_q_stage = 0;
            unsigned int mm_kv_stage = 0;
            unsigned int mm_tmem_stage = 0;
            int rg_smax_3 = (block_table_stride * K_BLOCK_KV + 255) / 256;
            int rg_units_3 = batch_size * rg_smax_3;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_kv_full_1 = 0;
            unsigned int _phase_umma_empty = 1;
            #pragma unroll 1
            for (int u_3 = bid; u_3 < rg_units_3; u_3 += num_bids) {
                int ru_u_3 = u_3;
                int ru_req_3 = ru_u_3 / rg_smax_3;
                int ru_s_3 = ru_u_3 % rg_smax_3;
                int ctx_last_3 = context_lens[ru_req_3 * K_NEXT_N + (K_NEXT_N - 1)];
                int num_kv_splits_3 = (ctx_last_3 + 256 - 1) / 256;
                if (ru_s_3 < num_kv_splits_3) {
                    mbarrier_wait(q_full_addr + (mm_q_stage) * 8, _phase_q_full_1);
                    mbarrier_wait(kv_full_addr + (mm_kv_stage) * 8, _phase_kv_full_1);
                    if (elect_sync()) {
                        mbarrier_wait(umma_empty_addr + (mm_tmem_stage) * 8, _phase_umma_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = (((smem_kv_addr) >> 4) & 0x3FFF) + (mm_kv_stage) * 2048;
                        int _mma_b_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (mm_q_stage) * 512;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135266320;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_acc + (mm_tmem_stage * 64))), "r"(0));
                        tcgen05_commit(umma_full_addr + (mm_tmem_stage) * 8);
                        mm_tmem_stage += 1;
                        if (mm_tmem_stage == 3) { mm_tmem_stage = 0; _phase_umma_empty ^= 1; }
                        mbarrier_wait(umma_empty_addr + (mm_tmem_stage) * 8, _phase_umma_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_1 = (((smem_kv_addr + 16384) >> 4) & 0x3FFF) + (mm_kv_stage) * 2048;
                        int _mma_b_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (mm_q_stage) * 512;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135266320;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_acc + (mm_tmem_stage * 64))), "r"(0));
                        tcgen05_commit(umma_full_addr + (mm_tmem_stage) * 8);
                        mm_tmem_stage += 1;
                        if (mm_tmem_stage == 3) { mm_tmem_stage = 0; _phase_umma_empty ^= 1; }
                    }
                    __syncwarp();
                    mm_kv_stage += 1;
                    if (mm_kv_stage == 5) { mm_kv_stage = 0; _phase_kv_full_1 ^= 1; }
                    mbarrier_arrive(q_empty_addr + (mm_q_stage) * 8);
                    mm_q_stage += 1;
                    if (mm_q_stage == 3) { mm_q_stage = 0; _phase_q_full_1 ^= 1; }
                }
            }
        }
    // ---- Role: idle ----
    } else if (warp == 11) {
        // idle — no tasks assigned
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 10) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
