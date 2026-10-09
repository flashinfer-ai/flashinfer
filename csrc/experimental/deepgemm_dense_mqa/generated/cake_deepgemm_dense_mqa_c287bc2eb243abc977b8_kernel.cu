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
#define TMEM_NCOLS 396
#define TMEM_TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SF_Q_OFFSET 384
#define TMEM_TMEM_SF_KV_OFFSET 388
#define NUM_Q_PIPE_STAGES 3
#define NUM_KV_PIPE_STAGES 10
#define NUM_TMEM_PIPE_STAGES 3
#define SMEM_NEGINF_SCRATCH_OFF 201728
#define SMEM_NEGINF_SCRATCH_STAGE_BYTES 4096
#define SMEM_NEGINF_SCRATCH_STRIDE 4096
#define SMEM_SMEM_Q_FP4_OFF 0
#define SMEM_SMEM_Q_FP4_STAGE_BYTES 8192
#define SMEM_SMEM_Q_FP4_STRIDE 8192
#define SMEM_SMEM_KV_FP4_OFF 24576
#define SMEM_SMEM_KV_FP4_STAGE_BYTES 16384
#define SMEM_SMEM_KV_FP4_STRIDE 16384
#define SMEM_SMEM_SF_Q_OFF 188416
#define SMEM_SMEM_SF_Q_STAGE_BYTES 512
#define SMEM_SMEM_SF_Q_STRIDE 512
#define SMEM_SMEM_SF_KV_OFF 189952
#define SMEM_SMEM_SF_KV_STAGE_BYTES 1024
#define SMEM_SMEM_SF_KV_STRIDE 1024
#define SMEM_SMEM_WEIGHTS_OFF 200192
#define SMEM_SMEM_WEIGHTS_STAGE_BYTES 512
#define SMEM_SMEM_WEIGHTS_STRIDE 512
#define SMEM_TOTAL 206336
#ifndef SM_COUNT
#error "SM_COUNT is a downstream specialization of this program; define it on the compile line"
#endif
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_deepgemm_dense_mqa_c287bc2eb243abc977b8(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap Weights, const __grid_constant__ CUtensorMap SF_Q, const __grid_constant__ CUtensorMap SF_KV, float* __restrict__ Logits, int* __restrict__ cu_seq_len_k_start, int* __restrict__ cu_seq_len_k_end, int seq_len, int seq_len_kv, int stride_logits, int num_q_blocks, unsigned int* __restrict__ ScheduleMeta)
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

    const int mbar_base = smem + 205824;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 24)
    #define kv_full_addr (mbar_base + 48)
    #define kv_empty_addr (mbar_base + 128)
    #define umma_full_addr (mbar_base + 208)
    #define umma_empty_addr (mbar_base + 232)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* neginf_scratch = reinterpret_cast<float*>(smem_raw + SMEM_NEGINF_SCRATCH_OFF);
    const int neginf_scratch_addr = smem + SMEM_NEGINF_SCRATCH_OFF;
    uint8_t* smem_q_fp4 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_Q_FP4_OFF);
    const int smem_q_fp4_addr = smem + SMEM_SMEM_Q_FP4_OFF;
    uint8_t* smem_kv_fp4 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_KV_FP4_OFF);
    const int smem_kv_fp4_addr = smem + SMEM_SMEM_KV_FP4_OFF;
    uint8_t* smem_sf_q = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SF_Q_OFF);
    const int smem_sf_q_addr = smem + SMEM_SMEM_SF_Q_OFF;
    uint8_t* smem_sf_kv = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SF_KV_OFF);
    const int smem_sf_kv_addr = smem + SMEM_SMEM_SF_KV_OFF;
    uint8_t* smem_weights = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_WEIGHTS_OFF);
    const int smem_weights_addr = smem + SMEM_SMEM_WEIGHTS_OFF;
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SF_Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SF_KV))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Weights))) : "memory"); }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int _min_0 = (((unsigned int)cu_seq_len_k_start[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : ((unsigned int)seq_len_kv));
    unsigned int start = _min_0;
    unsigned int _min_1 = (((unsigned int)cu_seq_len_k_end[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : ((unsigned int)seq_len_kv));
    unsigned int end = _min_1;
    unsigned int base = start / 4 * 4;
    unsigned int work = (end - base + 255) / 256;
    unsigned int empty_work = work;
    if (empty_work == 0) {
        if (tid == 0) {
            ScheduleMeta[2 * bid] = 1;
            ScheduleMeta[2 * bid + 1] = 0;
            ScheduleMeta[2 * SM_COUNT + bid] = 0;
            if (bid == 0) {
                ScheduleMeta[(3 * SM_COUNT + 1) / 2 * 2] = base;
                ScheduleMeta[(3 * SM_COUNT + 1) / 2 * 2 + 1] = 0;
            }
        }
        float empty_fill[4];
        #pragma unroll
        for (int component = 0; component < 4; component++) {
            empty_fill[component] = -CUDART_INF_F;
        }
        for (unsigned int col = (bid * 384 + tid) * 4; col < (unsigned int)stride_logits * 4; col += SM_COUNT * 384 * 4) {
            reinterpret_cast<int4*>(Logits + col)[0] = reinterpret_cast<int4*>(empty_fill)[0];
        }
    }
    if (__syncthreads_and(empty_work == 0)) return;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 32 barriers)
    // Mbarriers at smem_raw[205824..206080)

    if (warp == 9) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 3 barriers, init_count=1
            mbarrier_init(smem + 205824, 1);
            mbarrier_init(smem + 205832, 1);
            mbarrier_init(smem + 205840, 1);
            // q_empty: 3 barriers, init_count=288
            mbarrier_init(smem + 205848, 288);
            mbarrier_init(smem + 205856, 288);
            mbarrier_init(smem + 205864, 288);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 10 barriers, init_count=1
            mbarrier_init(smem + 205872, 1);
            mbarrier_init(smem + 205880, 1);
            mbarrier_init(smem + 205888, 1);
            mbarrier_init(smem + 205896, 1);
            mbarrier_init(smem + 205904, 1);
            mbarrier_init(smem + 205912, 1);
            mbarrier_init(smem + 205920, 1);
            mbarrier_init(smem + 205928, 1);
            mbarrier_init(smem + 205936, 1);
            mbarrier_init(smem + 205944, 1);
            // kv_empty: 10 barriers, init_count=1
            mbarrier_init(smem + 205952, 1);
            mbarrier_init(smem + 205960, 1);
            mbarrier_init(smem + 205968, 1);
            mbarrier_init(smem + 205976, 1);
            mbarrier_init(smem + 205984, 1);
            mbarrier_init(smem + 205992, 1);
            mbarrier_init(smem + 206000, 1);
            mbarrier_init(smem + 206008, 1);
            mbarrier_init(smem + 206016, 1);
            mbarrier_init(smem + 206024, 1);
            // --- pipeline 'tmem_pipe' ---
            // umma_full: 3 barriers, init_count=1
            mbarrier_init(smem + 206032, 1);
            mbarrier_init(smem + 206040, 1);
            mbarrier_init(smem + 206048, 1);
            // umma_empty: 3 barriers, init_count=128
            mbarrier_init(smem + 206056, 128);
            mbarrier_init(smem + 206064, 128);
            mbarrier_init(smem + 206072, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 396 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 206080);
    if (warp == 10) {
        int _tmem_hold = smem + 206080;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    const int tmem_tmem_sf_q = taddr + 384;
    const int tmem_tmem_sf_kv = taddr + 388;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (tid == 0) {
        unsigned int _min_2 = (((unsigned int)cu_seq_len_k_start[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : ((unsigned int)seq_len_kv));
        unsigned int start_0 = _min_2;
        unsigned int _min_3 = (((unsigned int)cu_seq_len_k_end[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : ((unsigned int)seq_len_kv));
        unsigned int end_1 = _min_3;
        unsigned int base_2 = start_0 / 4 * 4;
        unsigned int work_3 = (end_1 - base_2 + 255) / 256;
        unsigned int total_work = work_3;
        unsigned int total_cost = total_work;
        if (total_work != 0) {
            total_cost += 1;
        }
        unsigned int quotient = total_cost / (unsigned int)SM_COUNT;
        unsigned int remainder = total_cost % (unsigned int)SM_COUNT;
        int _min_4 = ((bid) < (remainder) ? (bid) : (remainder));
        unsigned int begin = (unsigned int)bid * quotient + (unsigned int)_min_4;
        int _min_5 = ((bid + 1) < (remainder) ? (bid + 1) : (remainder));
        unsigned int end_4 = (unsigned int)(bid + 1) * quotient + (unsigned int)_min_5;
        unsigned int first_q = 1;
        unsigned int first_split = 0;
        unsigned int begin_coord = total_work;
        unsigned int end_coord = total_work;
        if (begin != total_cost) {
            first_q = 0;
            unsigned int _max_0 = ((begin) > (1) ? (begin) : (1));
            unsigned int _min_6 = ((_max_0 - 1) < (total_work - 1) ? (_max_0 - 1) : (total_work - 1));
            first_split = _min_6;
            begin_coord = first_split;
        }
        if (end_4 != total_cost) {
            unsigned int _max_1 = ((end_4) > (1) ? (end_4) : (1));
            unsigned int _min_7 = ((_max_1 - 1) < (total_work - 1) ? (_max_1 - 1) : (total_work - 1));
            end_coord = _min_7;
        }
        unsigned int remaining = end_coord - begin_coord;
        ScheduleMeta[2 * bid] = first_q;
        ScheduleMeta[2 * bid + 1] = first_split;
        ScheduleMeta[2 * SM_COUNT + bid] = remaining;
        if (bid == 0) {
            unsigned int _min_8 = (((unsigned int)cu_seq_len_k_start[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : ((unsigned int)seq_len_kv));
            unsigned int start_1 = _min_8;
            unsigned int _min_9 = (((unsigned int)cu_seq_len_k_end[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : ((unsigned int)seq_len_kv));
            unsigned int end_2 = _min_9;
            unsigned int base_3 = start_1 / 4 * 4;
            unsigned int work_4 = (end_2 - base_3 + 255) / 256;
            ScheduleMeta[(3 * SM_COUNT + 1) / 2 * 2] = base_3;
            ScheduleMeta[(3 * SM_COUNT + 1) / 2 * 2 + 1] = work_4;
        }
    }

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
            unsigned int math_q_stage = 0;
            unsigned int math_tmem_stage = wg_idx;
            unsigned int first_q_1 = bid;
            unsigned int q_step = num_bids;
            unsigned int split_offset = 0;
            unsigned int remaining_1 = 0;
            unsigned int _min_41 = (((unsigned int)cu_seq_len_k_start[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : ((unsigned int)seq_len_kv));
            unsigned int start_0_1 = _min_41;
            unsigned int _min_42 = (((unsigned int)cu_seq_len_k_end[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : ((unsigned int)seq_len_kv));
            unsigned int end_1_1 = _min_42;
            unsigned int base_2_1 = start_0_1 / 4 * 4;
            unsigned int work_3_1 = (end_1_1 - base_2_1 + 255) / 256;
            unsigned int total_work_1 = work_3_1;
            unsigned int total_cost_1 = total_work_1;
            if (total_work_1 != 0) {
                total_cost_1 += 1;
            }
            unsigned int quotient_1 = total_cost_1 / (unsigned int)SM_COUNT;
            unsigned int remainder_1 = total_cost_1 % (unsigned int)SM_COUNT;
            int _min_43 = ((bid) < (remainder_1) ? (bid) : (remainder_1));
            unsigned int begin_1 = (unsigned int)bid * quotient_1 + (unsigned int)_min_43;
            int _min_44 = ((bid + 1) < (remainder_1) ? (bid + 1) : (remainder_1));
            unsigned int end_4_1 = (unsigned int)(bid + 1) * quotient_1 + (unsigned int)_min_44;
            unsigned int first_q_5 = 1;
            unsigned int first_split_1 = 0;
            unsigned int begin_coord_1 = total_work_1;
            unsigned int end_coord_1 = total_work_1;
            if (begin_1 != total_cost_1) {
                first_q_5 = 0;
                unsigned int _max_9 = ((begin_1) > (1) ? (begin_1) : (1));
                unsigned int _min_45 = ((_max_9 - 1) < (total_work_1 - 1) ? (_max_9 - 1) : (total_work_1 - 1));
                first_split_1 = _min_45;
                begin_coord_1 = first_split_1;
            }
            if (end_4_1 != total_cost_1) {
                unsigned int _max_10 = ((end_4_1) > (1) ? (end_4_1) : (1));
                unsigned int _min_46 = ((_max_10 - 1) < (total_work_1 - 1) ? (_max_10 - 1) : (total_work_1 - 1));
                end_coord_1 = _min_46;
            }
            unsigned int remaining_6 = end_coord_1 - begin_coord_1;
            first_q_1 = first_q_5;
            split_offset = first_split_1;
            remaining_1 = remaining_6;
            q_step = 1;
            unsigned int first_q_7 = first_q_1;
            unsigned int q_step_8 = q_step;
            unsigned int split_offset_9 = split_offset;
            unsigned int remaining_10 = remaining_1;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_umma_full = 0;
            #pragma unroll 1
            for (unsigned int q_block_idx = first_q_7; q_block_idx < num_q_blocks; q_block_idx += q_step_8) {
                if (remaining_10 == 0) {
                    break;
                }
                int q_start = q_block_idx * 4;
                int kv_start = 0;
                unsigned int num_kv_blocks = 0;
                unsigned int _min_47 = (((unsigned int)cu_seq_len_k_start[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : ((unsigned int)seq_len_kv));
                unsigned int start_1_1 = _min_47;
                unsigned int _min_48 = (((unsigned int)cu_seq_len_k_end[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : ((unsigned int)seq_len_kv));
                unsigned int end_2_1 = _min_48;
                unsigned int base_3_1 = start_1_1 / 4 * 4;
                unsigned int work_4_1 = (end_2_1 - base_3_1 + 255) / 256;
                kv_start = base_3_1 + split_offset_9 * 256;
                unsigned int span_splits = work_4_1;
                if (split_offset_9 < span_splits) {
                    unsigned int _min_49 = ((span_splits - split_offset_9) < (remaining_10) ? (span_splits - split_offset_9) : (remaining_10));
                    num_kv_blocks = _min_49;
                }
                remaining_10 -= num_kv_blocks;
                split_offset_9 = 0;
                if (num_kv_blocks > 0) {
                    mbarrier_wait(q_full_addr + (math_q_stage) * 8, _phase_q_full);
                    float weights_reg[128];
                    int wsmem = smem_weights_addr + math_q_stage * 512;
                    #pragma unroll
                    for (int wi = 0; wi < 32; wi++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[wi * 4])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(wi * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(wi * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(wi * 4) + 3]))
                            : "r"(wsmem + wi * 16));
                    }
                    int seq_start[4];
                    int seq_end[4];
                    {
                        #pragma unroll
                        for (int qi_bound = 0; qi_bound < 4; qi_bound++) {
                            int bound_row = ((q_start + qi_bound < seq_len) ? q_start + qi_bound : seq_len - 1);
                            int raw_start = cu_seq_len_k_start[bound_row];
                            int raw_end = cu_seq_len_k_end[bound_row];
                            seq_start[qi_bound] = ((raw_start < seq_len_kv) ? raw_start : seq_len_kv);
                            seq_end[qi_bound] = ((raw_end < seq_len_kv) ? raw_end : seq_len_kv);
                        }
                    }
                    #pragma unroll 1
                    for (unsigned int kv_iter = 0; kv_iter < num_kv_blocks; kv_iter++) {
                        mbarrier_wait(umma_full_addr + (math_tmem_stage) * 8, _phase_umma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int kv_pos = (unsigned int)kv_start + kv_iter * 256 + wg_idx * 128 + (unsigned int)(warp % 4 * 32 + lane);
                        float _tmem_load_0[32];
                        tmem_ld_x16(&_tmem_load_0[0], taddr + math_tmem_stage * 128 + (unsigned int)(warp % 4 * 32 << 16));
                        tmem_ld_x16(&_tmem_load_0[16], taddr + math_tmem_stage * 128 + (unsigned int)(warp % 4 * 32 << 16) + 16);
                        float _tmem_load_1[32];
                        tmem_ld_x16(&_tmem_load_1[0], taddr + math_tmem_stage * 128 + 32 + (unsigned int)(warp % 4 * 32 << 16));
                        tmem_ld_x16(&_tmem_load_1[16], taddr + math_tmem_stage * 128 + 32 + (unsigned int)(warp % 4 * 32 << 16) + 16);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float _tmem_load_2[32];
                        tmem_ld_x16(&_tmem_load_2[0], taddr + math_tmem_stage * 128 + 64 + (unsigned int)(warp % 4 * 32 << 16));
                        tmem_ld_x16(&_tmem_load_2[16], taddr + math_tmem_stage * 128 + 64 + (unsigned int)(warp % 4 * 32 << 16) + 16);
                        float _tmem_load_3[32];
                        tmem_ld_x16(&_tmem_load_3[0], taddr + math_tmem_stage * 128 + 96 + (unsigned int)(warp % 4 * 32 << 16));
                        tmem_ld_x16(&_tmem_load_3[16], taddr + math_tmem_stage * 128 + 96 + (unsigned int)(warp % 4 * 32 << 16) + 16);
                        {
                            float _relu_wsum_0;
                            {
                                float2 _sum0 = make_float2(0.0f, 0.0f);
                                float2 _sum1 = make_float2(0.0f, 0.0f);
                                #pragma unroll
                                for (int _j = 0; _j < 32; _j += 4) {
                                    float2 _a0_raw = make_float2(_tmem_load_0[0 + _j], _tmem_load_0[0 + _j + 1]);
                                    float2 _a0_abs = make_float2(fabsf(_tmem_load_0[0 + _j]), fabsf(_tmem_load_0[0 + _j + 1]));
                                    float2 _a0;
                                    asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a0) : "l"(*(const unsigned long long*)&_a0_raw), "l"(*(const unsigned long long*)&_a0_abs));
                                    float2 _b0 = make_float2(weights_reg[0 + _j], weights_reg[0 + _j + 1]);
                                    asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum0) : "l"(*(const unsigned long long*)&_a0), "l"(*(const unsigned long long*)&_b0));
                                    float2 _a1_raw = make_float2(_tmem_load_0[0 + _j + 2], _tmem_load_0[0 + _j + 3]);
                                    float2 _a1_abs = make_float2(fabsf(_tmem_load_0[0 + _j + 2]), fabsf(_tmem_load_0[0 + _j + 3]));
                                    float2 _a1;
                                    asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a1) : "l"(*(const unsigned long long*)&_a1_raw), "l"(*(const unsigned long long*)&_a1_abs));
                                    float2 _b1 = make_float2(weights_reg[0 + _j + 2], weights_reg[0 + _j + 3]);
                                    asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum1) : "l"(*(const unsigned long long*)&_a1), "l"(*(const unsigned long long*)&_b1));
                                }
                                float2 _sum;
                                asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_sum) : "l"(*(const unsigned long long*)&_sum0), "l"(*(const unsigned long long*)&_sum1));
                                _relu_wsum_0 = (_sum.x + _sum.y) * 0.5f;
                            }
                            float weighted_sum = _relu_wsum_0;
                            int q_row = q_start;
                            unsigned long long q_offset = (unsigned long long)q_row * (unsigned long long)stride_logits;
                            unsigned long long out_elem = q_offset + (unsigned long long)kv_pos;
                            {
                                weighted_sum = ((kv_pos >= seq_start[0] && kv_pos < seq_end[0]) ? weighted_sum : -CUDART_INF_F);
                            }
                            *(reinterpret_cast<float*>(Logits + out_elem) + (0)) = weighted_sum;
                        }
                        {
                            float _relu_wsum_1;
                            {
                                float2 _sum0 = make_float2(0.0f, 0.0f);
                                float2 _sum1 = make_float2(0.0f, 0.0f);
                                #pragma unroll
                                for (int _j = 0; _j < 32; _j += 4) {
                                    float2 _a0_raw = make_float2(_tmem_load_1[0 + _j], _tmem_load_1[0 + _j + 1]);
                                    float2 _a0_abs = make_float2(fabsf(_tmem_load_1[0 + _j]), fabsf(_tmem_load_1[0 + _j + 1]));
                                    float2 _a0;
                                    asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a0) : "l"(*(const unsigned long long*)&_a0_raw), "l"(*(const unsigned long long*)&_a0_abs));
                                    float2 _b0 = make_float2(weights_reg[32 + _j], weights_reg[32 + _j + 1]);
                                    asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum0) : "l"(*(const unsigned long long*)&_a0), "l"(*(const unsigned long long*)&_b0));
                                    float2 _a1_raw = make_float2(_tmem_load_1[0 + _j + 2], _tmem_load_1[0 + _j + 3]);
                                    float2 _a1_abs = make_float2(fabsf(_tmem_load_1[0 + _j + 2]), fabsf(_tmem_load_1[0 + _j + 3]));
                                    float2 _a1;
                                    asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a1) : "l"(*(const unsigned long long*)&_a1_raw), "l"(*(const unsigned long long*)&_a1_abs));
                                    float2 _b1 = make_float2(weights_reg[32 + _j + 2], weights_reg[32 + _j + 3]);
                                    asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum1) : "l"(*(const unsigned long long*)&_a1), "l"(*(const unsigned long long*)&_b1));
                                }
                                float2 _sum;
                                asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_sum) : "l"(*(const unsigned long long*)&_sum0), "l"(*(const unsigned long long*)&_sum1));
                                _relu_wsum_1 = (_sum.x + _sum.y) * 0.5f;
                            }
                            float weighted_sum_1 = _relu_wsum_1;
                            int q_row_1 = q_start + 1;
                            unsigned long long q_offset_1 = (unsigned long long)q_row_1 * (unsigned long long)stride_logits;
                            unsigned long long out_elem_1 = q_offset_1 + (unsigned long long)kv_pos;
                            {
                                weighted_sum_1 = ((kv_pos >= seq_start[1] && kv_pos < seq_end[1]) ? weighted_sum_1 : -CUDART_INF_F);
                            }
                            *(reinterpret_cast<float*>(Logits + out_elem_1) + (0)) = weighted_sum_1;
                        }
                        {
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(umma_empty_addr + (math_tmem_stage) * 8);
                            float _relu_wsum_2;
                            {
                                float2 _sum0 = make_float2(0.0f, 0.0f);
                                float2 _sum1 = make_float2(0.0f, 0.0f);
                                #pragma unroll
                                for (int _j = 0; _j < 32; _j += 4) {
                                    float2 _a0_raw = make_float2(_tmem_load_2[0 + _j], _tmem_load_2[0 + _j + 1]);
                                    float2 _a0_abs = make_float2(fabsf(_tmem_load_2[0 + _j]), fabsf(_tmem_load_2[0 + _j + 1]));
                                    float2 _a0;
                                    asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a0) : "l"(*(const unsigned long long*)&_a0_raw), "l"(*(const unsigned long long*)&_a0_abs));
                                    float2 _b0 = make_float2(weights_reg[64 + _j], weights_reg[64 + _j + 1]);
                                    asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum0) : "l"(*(const unsigned long long*)&_a0), "l"(*(const unsigned long long*)&_b0));
                                    float2 _a1_raw = make_float2(_tmem_load_2[0 + _j + 2], _tmem_load_2[0 + _j + 3]);
                                    float2 _a1_abs = make_float2(fabsf(_tmem_load_2[0 + _j + 2]), fabsf(_tmem_load_2[0 + _j + 3]));
                                    float2 _a1;
                                    asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a1) : "l"(*(const unsigned long long*)&_a1_raw), "l"(*(const unsigned long long*)&_a1_abs));
                                    float2 _b1 = make_float2(weights_reg[64 + _j + 2], weights_reg[64 + _j + 3]);
                                    asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum1) : "l"(*(const unsigned long long*)&_a1), "l"(*(const unsigned long long*)&_b1));
                                }
                                float2 _sum;
                                asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_sum) : "l"(*(const unsigned long long*)&_sum0), "l"(*(const unsigned long long*)&_sum1));
                                _relu_wsum_2 = (_sum.x + _sum.y) * 0.5f;
                            }
                            float weighted_sum_2 = _relu_wsum_2;
                            int q_row_2 = q_start + 2;
                            unsigned long long q_offset_2 = (unsigned long long)q_row_2 * (unsigned long long)stride_logits;
                            unsigned long long out_elem_2 = q_offset_2 + (unsigned long long)kv_pos;
                            {
                                weighted_sum_2 = ((kv_pos >= seq_start[2] && kv_pos < seq_end[2]) ? weighted_sum_2 : -CUDART_INF_F);
                            }
                            *(reinterpret_cast<float*>(Logits + out_elem_2) + (0)) = weighted_sum_2;
                        }
                        {
                            float _relu_wsum_3;
                            {
                                float2 _sum0 = make_float2(0.0f, 0.0f);
                                float2 _sum1 = make_float2(0.0f, 0.0f);
                                #pragma unroll
                                for (int _j = 0; _j < 32; _j += 4) {
                                    float2 _a0_raw = make_float2(_tmem_load_3[0 + _j], _tmem_load_3[0 + _j + 1]);
                                    float2 _a0_abs = make_float2(fabsf(_tmem_load_3[0 + _j]), fabsf(_tmem_load_3[0 + _j + 1]));
                                    float2 _a0;
                                    asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a0) : "l"(*(const unsigned long long*)&_a0_raw), "l"(*(const unsigned long long*)&_a0_abs));
                                    float2 _b0 = make_float2(weights_reg[96 + _j], weights_reg[96 + _j + 1]);
                                    asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum0) : "l"(*(const unsigned long long*)&_a0), "l"(*(const unsigned long long*)&_b0));
                                    float2 _a1_raw = make_float2(_tmem_load_3[0 + _j + 2], _tmem_load_3[0 + _j + 3]);
                                    float2 _a1_abs = make_float2(fabsf(_tmem_load_3[0 + _j + 2]), fabsf(_tmem_load_3[0 + _j + 3]));
                                    float2 _a1;
                                    asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a1) : "l"(*(const unsigned long long*)&_a1_raw), "l"(*(const unsigned long long*)&_a1_abs));
                                    float2 _b1 = make_float2(weights_reg[96 + _j + 2], weights_reg[96 + _j + 3]);
                                    asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum1) : "l"(*(const unsigned long long*)&_a1), "l"(*(const unsigned long long*)&_b1));
                                }
                                float2 _sum;
                                asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_sum) : "l"(*(const unsigned long long*)&_sum0), "l"(*(const unsigned long long*)&_sum1));
                                _relu_wsum_3 = (_sum.x + _sum.y) * 0.5f;
                            }
                            float weighted_sum_3 = _relu_wsum_3;
                            int q_row_3 = q_start + 3;
                            unsigned long long q_offset_3 = (unsigned long long)q_row_3 * (unsigned long long)stride_logits;
                            unsigned long long out_elem_3 = q_offset_3 + (unsigned long long)kv_pos;
                            {
                                weighted_sum_3 = ((kv_pos >= seq_start[3] && kv_pos < seq_end[3]) ? weighted_sum_3 : -CUDART_INF_F);
                            }
                            *(reinterpret_cast<float*>(Logits + out_elem_3) + (0)) = weighted_sum_3;
                        }
                        math_tmem_stage += 1;
                        if (math_tmem_stage == 3) { math_tmem_stage = 0; _phase_umma_full ^= 1; }
                        math_tmem_stage += 1;
                        if (math_tmem_stage == 3) { math_tmem_stage = 0; _phase_umma_full ^= 1; }
                    }
                    {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    mbarrier_arrive(q_empty_addr + (math_q_stage) * 8);
                    math_q_stage += 1;
                    if (math_q_stage == 3) { math_q_stage = 0; _phase_q_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_q ----
    if (warp == 8) {
        { // load_q_main
            unsigned int load_q_stage = 0;
            unsigned int _phase_q_empty = 1;
            if (elect_sync()) {
                unsigned int first_q_2 = bid;
                unsigned int q_step_1 = num_bids;
                unsigned int split_offset_1 = 0;
                unsigned int remaining_2 = 0;
                unsigned int _min_14 = (((unsigned int)cu_seq_len_k_start[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : ((unsigned int)seq_len_kv));
                unsigned int start_0_2 = _min_14;
                unsigned int _min_15 = (((unsigned int)cu_seq_len_k_end[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : ((unsigned int)seq_len_kv));
                unsigned int end_1_2 = _min_15;
                unsigned int base_2_2 = start_0_2 / 4 * 4;
                unsigned int work_3_2 = (end_1_2 - base_2_2 + 255) / 256;
                unsigned int total_work_2 = work_3_2;
                unsigned int total_cost_2 = total_work_2;
                if (total_work_2 != 0) {
                    total_cost_2 += 1;
                }
                unsigned int quotient_2 = total_cost_2 / (unsigned int)SM_COUNT;
                unsigned int remainder_2 = total_cost_2 % (unsigned int)SM_COUNT;
                int _min_16 = ((bid) < (remainder_2) ? (bid) : (remainder_2));
                unsigned int begin_2 = (unsigned int)bid * quotient_2 + (unsigned int)_min_16;
                int _min_17 = ((bid + 1) < (remainder_2) ? (bid + 1) : (remainder_2));
                unsigned int end_4_2 = (unsigned int)(bid + 1) * quotient_2 + (unsigned int)_min_17;
                unsigned int first_q_5_1 = 1;
                unsigned int first_split_2 = 0;
                unsigned int begin_coord_2 = total_work_2;
                unsigned int end_coord_2 = total_work_2;
                if (begin_2 != total_cost_2) {
                    first_q_5_1 = 0;
                    unsigned int _max_3 = ((begin_2) > (1) ? (begin_2) : (1));
                    unsigned int _min_18 = ((_max_3 - 1) < (total_work_2 - 1) ? (_max_3 - 1) : (total_work_2 - 1));
                    first_split_2 = _min_18;
                    begin_coord_2 = first_split_2;
                }
                if (end_4_2 != total_cost_2) {
                    unsigned int _max_4 = ((end_4_2) > (1) ? (end_4_2) : (1));
                    unsigned int _min_19 = ((_max_4 - 1) < (total_work_2 - 1) ? (_max_4 - 1) : (total_work_2 - 1));
                    end_coord_2 = _min_19;
                }
                unsigned int remaining_6_1 = end_coord_2 - begin_coord_2;
                first_q_2 = first_q_5_1;
                split_offset_1 = first_split_2;
                remaining_2 = remaining_6_1;
                q_step_1 = 1;
                unsigned int first_q_7_1 = first_q_2;
                unsigned int q_step_8_1 = q_step_1;
                unsigned int split_offset_9_1 = split_offset_1;
                unsigned int remaining_10_1 = remaining_2;
                #pragma unroll 1
                for (unsigned int q_block_idx_1 = first_q_7_1; q_block_idx_1 < num_q_blocks; q_block_idx_1 += q_step_8_1) {
                    if (remaining_10_1 == 0) {
                        break;
                    }
                    int q_start_1 = q_block_idx_1 * 4;
                    int kv_start_1 = 0;
                    unsigned int num_kv_blocks_1 = 0;
                    unsigned int _min_20 = (((unsigned int)cu_seq_len_k_start[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : ((unsigned int)seq_len_kv));
                    unsigned int start_1_2 = _min_20;
                    unsigned int _min_21 = (((unsigned int)cu_seq_len_k_end[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : ((unsigned int)seq_len_kv));
                    unsigned int end_2_2 = _min_21;
                    unsigned int base_3_2 = start_1_2 / 4 * 4;
                    unsigned int work_4_2 = (end_2_2 - base_3_2 + 255) / 256;
                    kv_start_1 = base_3_2 + split_offset_9_1 * 256;
                    unsigned int span_splits_1 = work_4_2;
                    if (split_offset_9_1 < span_splits_1) {
                        unsigned int _min_22 = ((span_splits_1 - split_offset_9_1) < (remaining_10_1) ? (span_splits_1 - split_offset_9_1) : (remaining_10_1));
                        num_kv_blocks_1 = _min_22;
                    }
                    remaining_10_1 -= num_kv_blocks_1;
                    split_offset_9_1 = 0;
                    if (num_kv_blocks_1 > 0) {
                        mbarrier_wait(q_empty_addr + (load_q_stage) * 8, _phase_q_empty);
                        tma_2d_gmem2smem(smem_q_fp4_addr + load_q_stage * 8192, (&Q), 0, q_block_idx_1 * 128, q_full_addr + (load_q_stage) * 8);
                        tma_2d_gmem2smem(smem_sf_q_addr + load_q_stage * 512, (&SF_Q), 0, q_block_idx_1 * 32, q_full_addr + (load_q_stage) * 8);
                        tma_2d_gmem2smem(smem_weights_addr + load_q_stage * 512, (&Weights), 0, q_block_idx_1 * 4, q_full_addr + (load_q_stage) * 8);
                        mbarrier_arrive_expect_tx(q_full_addr + (load_q_stage) * 8, 9216);
                        load_q_stage += 1;
                        if (load_q_stage == 3) { load_q_stage = 0; _phase_q_empty ^= 1; }
                    }
                }
            }
            __syncwarp();
        }
    }
    // ---- Role: load_kv ----
    if (warp == 9) {
        { // load_kv_main
            unsigned int load_kv_stage = 0;
            unsigned int first_q_3 = bid;
            unsigned int q_step_2 = num_bids;
            unsigned int split_offset_2 = 0;
            unsigned int remaining_3 = 0;
            unsigned int _min_23 = (((unsigned int)cu_seq_len_k_start[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : ((unsigned int)seq_len_kv));
            unsigned int start_0_3 = _min_23;
            unsigned int _min_24 = (((unsigned int)cu_seq_len_k_end[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : ((unsigned int)seq_len_kv));
            unsigned int end_1_3 = _min_24;
            unsigned int base_2_3 = start_0_3 / 4 * 4;
            unsigned int work_3_3 = (end_1_3 - base_2_3 + 255) / 256;
            unsigned int total_work_3 = work_3_3;
            unsigned int total_cost_3 = total_work_3;
            if (total_work_3 != 0) {
                total_cost_3 += 1;
            }
            unsigned int quotient_3 = total_cost_3 / (unsigned int)SM_COUNT;
            unsigned int remainder_3 = total_cost_3 % (unsigned int)SM_COUNT;
            int _min_25 = ((bid) < (remainder_3) ? (bid) : (remainder_3));
            unsigned int begin_3 = (unsigned int)bid * quotient_3 + (unsigned int)_min_25;
            int _min_26 = ((bid + 1) < (remainder_3) ? (bid + 1) : (remainder_3));
            unsigned int end_4_3 = (unsigned int)(bid + 1) * quotient_3 + (unsigned int)_min_26;
            unsigned int first_q_5_2 = 1;
            unsigned int first_split_3 = 0;
            unsigned int begin_coord_3 = total_work_3;
            unsigned int end_coord_3 = total_work_3;
            if (begin_3 != total_cost_3) {
                first_q_5_2 = 0;
                unsigned int _max_5 = ((begin_3) > (1) ? (begin_3) : (1));
                unsigned int _min_27 = ((_max_5 - 1) < (total_work_3 - 1) ? (_max_5 - 1) : (total_work_3 - 1));
                first_split_3 = _min_27;
                begin_coord_3 = first_split_3;
            }
            if (end_4_3 != total_cost_3) {
                unsigned int _max_6 = ((end_4_3) > (1) ? (end_4_3) : (1));
                unsigned int _min_28 = ((_max_6 - 1) < (total_work_3 - 1) ? (_max_6 - 1) : (total_work_3 - 1));
                end_coord_3 = _min_28;
            }
            unsigned int remaining_6_2 = end_coord_3 - begin_coord_3;
            first_q_3 = first_q_5_2;
            split_offset_2 = first_split_3;
            remaining_3 = remaining_6_2;
            q_step_2 = 1;
            unsigned int first_q_7_2 = first_q_3;
            unsigned int q_step_8_2 = q_step_2;
            unsigned int split_offset_9_2 = split_offset_2;
            unsigned int remaining_10_2 = remaining_3;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int q_block_idx_2 = first_q_7_2; q_block_idx_2 < num_q_blocks; q_block_idx_2 += q_step_8_2) {
                if (remaining_10_2 == 0) {
                    break;
                }
                int q_start_2 = q_block_idx_2 * 4;
                int kv_start_2 = 0;
                unsigned int num_kv_blocks_2 = 0;
                unsigned int _min_29 = (((unsigned int)cu_seq_len_k_start[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : ((unsigned int)seq_len_kv));
                unsigned int start_1_3 = _min_29;
                unsigned int _min_30 = (((unsigned int)cu_seq_len_k_end[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : ((unsigned int)seq_len_kv));
                unsigned int end_2_3 = _min_30;
                unsigned int base_3_3 = start_1_3 / 4 * 4;
                unsigned int work_4_3 = (end_2_3 - base_3_3 + 255) / 256;
                kv_start_2 = base_3_3 + split_offset_9_2 * 256;
                unsigned int span_splits_2 = work_4_3;
                if (split_offset_9_2 < span_splits_2) {
                    unsigned int _min_31 = ((span_splits_2 - split_offset_9_2) < (remaining_10_2) ? (span_splits_2 - split_offset_9_2) : (remaining_10_2));
                    num_kv_blocks_2 = _min_31;
                }
                remaining_10_2 -= num_kv_blocks_2;
                split_offset_9_2 = 0;
                if (num_kv_blocks_2 > 0) {
                    #pragma unroll 1
                    for (unsigned int kv_iter_1 = 0; kv_iter_1 < num_kv_blocks_2; kv_iter_1++) {
                        int kv_row = (unsigned int)kv_start_2 + kv_iter_1 * 256;
                        if (elect_sync()) {
                            mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                            tma_2d_gmem2smem(smem_kv_fp4_addr + load_kv_stage * 16384, (&KV), 0, kv_row, kv_full_addr + (load_kv_stage) * 8);
                            tma_2d_gmem2smem(smem_sf_kv_addr + load_kv_stage * 1024, (&SF_KV), 0, kv_row / 4, kv_full_addr + (load_kv_stage) * 8);
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 17408);
                            load_kv_stage += 1;
                            if (load_kv_stage == 10) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                        }
                        __syncwarp();
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 10) {
        { // mma_main
            unsigned int mma_q_stage = 0;
            unsigned int mma_kv_stage = 0;
            unsigned int mma_tmem_stage = 0;
            unsigned int first_q_4 = bid;
            unsigned int q_step_3 = num_bids;
            unsigned int split_offset_3 = 0;
            unsigned int remaining_4 = 0;
            unsigned int _min_32 = (((unsigned int)cu_seq_len_k_start[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : ((unsigned int)seq_len_kv));
            unsigned int start_0_4 = _min_32;
            unsigned int _min_33 = (((unsigned int)cu_seq_len_k_end[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : ((unsigned int)seq_len_kv));
            unsigned int end_1_4 = _min_33;
            unsigned int base_2_4 = start_0_4 / 4 * 4;
            unsigned int work_3_4 = (end_1_4 - base_2_4 + 255) / 256;
            unsigned int total_work_4 = work_3_4;
            unsigned int total_cost_4 = total_work_4;
            if (total_work_4 != 0) {
                total_cost_4 += 1;
            }
            unsigned int quotient_4 = total_cost_4 / (unsigned int)SM_COUNT;
            unsigned int remainder_4 = total_cost_4 % (unsigned int)SM_COUNT;
            int _min_34 = ((bid) < (remainder_4) ? (bid) : (remainder_4));
            unsigned int begin_4 = (unsigned int)bid * quotient_4 + (unsigned int)_min_34;
            int _min_35 = ((bid + 1) < (remainder_4) ? (bid + 1) : (remainder_4));
            unsigned int end_4_4 = (unsigned int)(bid + 1) * quotient_4 + (unsigned int)_min_35;
            unsigned int first_q_5_3 = 1;
            unsigned int first_split_4 = 0;
            unsigned int begin_coord_4 = total_work_4;
            unsigned int end_coord_4 = total_work_4;
            if (begin_4 != total_cost_4) {
                first_q_5_3 = 0;
                unsigned int _max_7 = ((begin_4) > (1) ? (begin_4) : (1));
                unsigned int _min_36 = ((_max_7 - 1) < (total_work_4 - 1) ? (_max_7 - 1) : (total_work_4 - 1));
                first_split_4 = _min_36;
                begin_coord_4 = first_split_4;
            }
            if (end_4_4 != total_cost_4) {
                unsigned int _max_8 = ((end_4_4) > (1) ? (end_4_4) : (1));
                unsigned int _min_37 = ((_max_8 - 1) < (total_work_4 - 1) ? (_max_8 - 1) : (total_work_4 - 1));
                end_coord_4 = _min_37;
            }
            unsigned int remaining_6_3 = end_coord_4 - begin_coord_4;
            first_q_4 = first_q_5_3;
            split_offset_3 = first_split_4;
            remaining_4 = remaining_6_3;
            q_step_3 = 1;
            unsigned int first_q_7_3 = first_q_4;
            unsigned int q_step_8_3 = q_step_3;
            unsigned int split_offset_9_3 = split_offset_3;
            unsigned int remaining_10_3 = remaining_4;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_umma_empty = 1;
            #pragma unroll 1
            for (unsigned int q_block_idx_3 = first_q_7_3; q_block_idx_3 < num_q_blocks; q_block_idx_3 += q_step_8_3) {
                if (remaining_10_3 == 0) {
                    break;
                }
                int q_start_3 = q_block_idx_3 * 4;
                int kv_start_3 = 0;
                unsigned int num_kv_blocks_3 = 0;
                unsigned int _min_38 = (((unsigned int)cu_seq_len_k_start[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : ((unsigned int)seq_len_kv));
                unsigned int start_1_4 = _min_38;
                unsigned int _min_39 = (((unsigned int)cu_seq_len_k_end[0]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : ((unsigned int)seq_len_kv));
                unsigned int end_2_4 = _min_39;
                unsigned int base_3_4 = start_1_4 / 4 * 4;
                unsigned int work_4_4 = (end_2_4 - base_3_4 + 255) / 256;
                kv_start_3 = base_3_4 + split_offset_9_3 * 256;
                unsigned int span_splits_3 = work_4_4;
                if (split_offset_9_3 < span_splits_3) {
                    unsigned int _min_40 = ((span_splits_3 - split_offset_9_3) < (remaining_10_3) ? (span_splits_3 - split_offset_9_3) : (remaining_10_3));
                    num_kv_blocks_3 = _min_40;
                }
                remaining_10_3 -= num_kv_blocks_3;
                split_offset_9_3 = 0;
                if (num_kv_blocks_3 > 0) {
                    mbarrier_wait(q_full_addr + (mma_q_stage) * 8, _phase_q_full_1);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    unsigned int _sf_v[4];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[0])) : "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)(((lane >> 3 ^ 0) * 32 + lane) * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[1])) : "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)(((lane >> 3 ^ 1) * 32 + lane) * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[2])) : "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)(((lane >> 3 ^ 2) * 32 + lane) * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[3])) : "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)(((lane >> 3 ^ 3) * 32 + lane) * 4)));
                    __syncwarp();
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 0)) * 4)), "r"((_sf_v[0])));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 1)) * 4)), "r"((_sf_v[1])));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 2)) * 4)), "r"((_sf_v[2])));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 3)) * 4)), "r"((_sf_v[3])));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sf_q, make_sf_cp_desc_lo_sbo128((((smem_sf_q_addr) >> 4) + (mma_q_stage) * 32)));
                    }
                    __syncwarp();
                    #pragma unroll 1
                    for (unsigned int kv_iter_2 = 0; kv_iter_2 < num_kv_blocks_3; kv_iter_2++) {
                        mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                        int sfkv_smem = smem_sf_kv_addr + mma_kv_stage * 1024;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        unsigned int _sf_v_0[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[0])) : "r"(sfkv_smem + ((lane >> 3 ^ 0) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[1])) : "r"(sfkv_smem + ((lane >> 3 ^ 1) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[2])) : "r"(sfkv_smem + ((lane >> 3 ^ 2) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[3])) : "r"(sfkv_smem + ((lane >> 3 ^ 3) * 32 + lane) * 4));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + (lane * 4 + (lane >> 3 ^ 0)) * 4), "r"((_sf_v_0[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + (lane * 4 + (lane >> 3 ^ 1)) * 4), "r"((_sf_v_0[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + (lane * 4 + (lane >> 3 ^ 2)) * 4), "r"((_sf_v_0[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + (lane * 4 + (lane >> 3 ^ 3)) * 4), "r"((_sf_v_0[3])));
                        unsigned int _sf_v_1[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[0])) : "r"(sfkv_smem + 512 + ((lane >> 3 ^ 0) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[1])) : "r"(sfkv_smem + 512 + ((lane >> 3 ^ 1) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[2])) : "r"(sfkv_smem + 512 + ((lane >> 3 ^ 2) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[3])) : "r"(sfkv_smem + 512 + ((lane >> 3 ^ 3) * 32 + lane) * 4));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + 512 + (lane * 4 + (lane >> 3 ^ 0)) * 4), "r"((_sf_v_1[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + 512 + (lane * 4 + (lane >> 3 ^ 1)) * 4), "r"((_sf_v_1[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + 512 + (lane * 4 + (lane >> 3 ^ 2)) * 4), "r"((_sf_v_1[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + 512 + (lane * 4 + (lane >> 3 ^ 3)) * 4), "r"((_sf_v_1[3])));
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4(tmem_tmem_sf_kv, make_sf_cp_desc_lo_sbo128((((smem_sf_kv_addr) >> 4) + (mma_kv_stage) * 64)));
                            tcgen05_cp_32x128b_warpx4((tmem_tmem_sf_kv + 4), make_sf_cp_desc_lo_sbo128((((smem_sf_kv_addr) >> 4) + (mma_kv_stage) * 64 + 32)));
                            mbarrier_wait(umma_empty_addr + (mma_tmem_stage) * 8, _phase_umma_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_0 = (((smem_kv_fp4_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024;
                            int _mma_b_lo_0 = (((smem_q_fp4_addr) >> 4) & 0x3FFF) + (mma_q_stage) * 512;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mma_tmem_stage * 128)), a_desc + 0, b_desc + 0,
                                    0x8a00480U, tmem_tmem_sf_kv + 0, tmem_tmem_sf_q + 0, 0);
                                tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mma_tmem_stage * 128)), a_desc + 2, b_desc + 2,
                                    0x48a004a0U, tmem_tmem_sf_kv + 0, tmem_tmem_sf_q + 0, 1);
                            }
                            tcgen05_commit(umma_full_addr + (mma_tmem_stage) * 8);
                            mma_tmem_stage += 1;
                            if (mma_tmem_stage == 3) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
                            mbarrier_wait(umma_empty_addr + (mma_tmem_stage) * 8, _phase_umma_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_1 = (((smem_kv_fp4_addr + 8192) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024;
                            int _mma_b_lo_1 = (((smem_q_fp4_addr) >> 4) & 0x3FFF) + (mma_q_stage) * 512;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mma_tmem_stage * 128)), a_desc + 0, b_desc + 0,
                                    0x8a00480U, tmem_tmem_sf_kv + 4 + 0, tmem_tmem_sf_q + 0, 0);
                                tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mma_tmem_stage * 128)), a_desc + 2, b_desc + 2,
                                    0x48a004a0U, tmem_tmem_sf_kv + 4 + 0, tmem_tmem_sf_q + 0, 1);
                            }
                            tcgen05_commit(umma_full_addr + (mma_tmem_stage) * 8);
                            mma_tmem_stage += 1;
                            if (mma_tmem_stage == 3) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
                        }
                        __syncwarp();
                        elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                        mma_kv_stage += 1;
                        if (mma_kv_stage == 10) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                    }
                    mbarrier_arrive(q_empty_addr + (mma_q_stage) * 8);
                    mma_q_stage += 1;
                    if (mma_q_stage == 3) { mma_q_stage = 0; _phase_q_full_1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 11) {
        { // idle_main
            {
                float neg_inf[4];
                #pragma unroll
                for (int component_1 = 0; component_1 < 4; component_1++) {
                    neg_inf[component_1] = -CUDART_INF_F;
                }
                for (int scratch_j = lane; scratch_j < 1024; scratch_j += 32) {
                    neginf_scratch[scratch_j] = -CUDART_INF_F;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                unsigned int clean_tiles_per_row = (stride_logits + 1023) / 1024;
                unsigned int clean_tasks = (unsigned int)num_q_blocks * clean_tiles_per_row;
                #pragma unroll 1
                for (unsigned int clean_task = bid; clean_task < clean_tasks; clean_task += SM_COUNT) {
                    unsigned int q_block_idx_4 = clean_task / clean_tiles_per_row;
                    int clean_begin = clean_task % clean_tiles_per_row * 1024;
                    int clean_end_raw = clean_begin + 1024;
                    int _min_10 = ((clean_end_raw) < (stride_logits) ? (clean_end_raw) : (stride_logits));
                    int clean_end = _min_10;
                    int q_start_4 = q_block_idx_4 * 4;
                    int ks0 = cu_seq_len_k_start[q_start_4];
                    int ke0 = cu_seq_len_k_end[q_start_4];
                    int kv_start_acc = ((ks0 < seq_len_kv) ? ks0 : seq_len_kv);
                    int kv_end_acc = ((ke0 < seq_len_kv) ? ke0 : seq_len_kv);
                    int row_idx = ((q_start_4 + 1 < seq_len) ? q_start_4 + 1 : seq_len - 1);
                    int ks = cu_seq_len_k_start[row_idx];
                    int ke = cu_seq_len_k_end[row_idx];
                    int sk = ((ks < seq_len_kv) ? ks : seq_len_kv);
                    int se = ((ke < seq_len_kv) ? ke : seq_len_kv);
                    kv_start_acc = ((sk < kv_start_acc) ? sk : kv_start_acc);
                    kv_end_acc = ((se > kv_end_acc) ? se : kv_end_acc);
                    int row_idx_0 = ((q_start_4 + 2 < seq_len) ? q_start_4 + 2 : seq_len - 1);
                    int ks_1 = cu_seq_len_k_start[row_idx_0];
                    int ke_2 = cu_seq_len_k_end[row_idx_0];
                    int sk_3 = ((ks_1 < seq_len_kv) ? ks_1 : seq_len_kv);
                    int se_4 = ((ke_2 < seq_len_kv) ? ke_2 : seq_len_kv);
                    kv_start_acc = ((sk_3 < kv_start_acc) ? sk_3 : kv_start_acc);
                    kv_end_acc = ((se_4 > kv_end_acc) ? se_4 : kv_end_acc);
                    int row_idx_5 = ((q_start_4 + 3 < seq_len) ? q_start_4 + 3 : seq_len - 1);
                    int ks_6 = cu_seq_len_k_start[row_idx_5];
                    int ke_7 = cu_seq_len_k_end[row_idx_5];
                    int sk_8 = ((ks_6 < seq_len_kv) ? ks_6 : seq_len_kv);
                    int se_9 = ((ke_7 < seq_len_kv) ? ke_7 : seq_len_kv);
                    kv_start_acc = ((sk_8 < kv_start_acc) ? sk_8 : kv_start_acc);
                    kv_end_acc = ((se_9 > kv_end_acc) ? se_9 : kv_end_acc);
                    int kv_start_4 = kv_start_acc / 4 * 4;
                    unsigned int num_kv_blocks_4 = (kv_end_acc - kv_start_4 + 256 - 1) / 256;
                    int raw_end_1 = (unsigned int)kv_start_4 + num_kv_blocks_4 * 256;
                    int coverage_end = ((raw_end_1 < stride_logits) ? raw_end_1 : stride_logits);
                    #pragma unroll 1
                    for (int qi = 0; qi < 4; qi++) {
                        unsigned long long row_base = (unsigned long long)(q_start_4 + qi) * (unsigned long long)stride_logits;
                        int _min_11 = ((kv_start_4) < (clean_end) ? (kv_start_4) : (clean_end));
                        int aligned_start = (clean_begin + 3) / 4 * 4;
                        int aligned_end = _min_11 / 4 * 4;
                        if (aligned_start >= aligned_end) {
                            for (int j = clean_begin + lane; j < _min_11; j += 32) {
                                *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j)) + (0)) = -CUDART_INF_F;
                            }
                            __syncwarp();
                        } else {
                            for (int j_1 = clean_begin + lane; j_1 < aligned_start; j_1 += 32) {
                                *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_1)) + (0)) = -CUDART_INF_F;
                            }
                            for (int j_2 = aligned_end + lane; j_2 < _min_11; j_2 += 32) {
                                *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_2)) + (0)) = -CUDART_INF_F;
                            }
                            __syncwarp();
                            if (elect_sync()) {
                                for (int j_3 = aligned_start; j_3 < aligned_end; j_3 += 1024) {
                                    int _min_12 = ((aligned_end - j_3) < (1024) ? (aligned_end - j_3) : (1024));
                                    int bulk_elems = _min_12;
                                    {
                                        void* _cpbulk_dst_0 = reinterpret_cast<void*>(Logits + (row_base + (unsigned long long)j_3));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_0), "r"(neginf_scratch_addr), "r"((uint32_t)(bulk_elems * 4))
                                            : "memory");
                                    }
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            __syncwarp();
                        }
                        int _max_2 = ((coverage_end) > (clean_begin) ? (coverage_end) : (clean_begin));
                        int aligned_start_0 = (_max_2 + 3) / 4 * 4;
                        int aligned_end_1 = clean_end / 4 * 4;
                        if (aligned_start_0 >= aligned_end_1) {
                            for (int j_4 = _max_2 + lane; j_4 < clean_end; j_4 += 32) {
                                *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_4)) + (0)) = -CUDART_INF_F;
                            }
                            __syncwarp();
                        } else {
                            for (int j_5 = _max_2 + lane; j_5 < aligned_start_0; j_5 += 32) {
                                *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_5)) + (0)) = -CUDART_INF_F;
                            }
                            for (int j_6 = aligned_end_1 + lane; j_6 < clean_end; j_6 += 32) {
                                *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_6)) + (0)) = -CUDART_INF_F;
                            }
                            __syncwarp();
                            if (elect_sync()) {
                                for (int j_7 = aligned_start_0; j_7 < aligned_end_1; j_7 += 1024) {
                                    int _min_13 = ((aligned_end_1 - j_7) < (1024) ? (aligned_end_1 - j_7) : (1024));
                                    int bulk_elems_1 = _min_13;
                                    {
                                        void* _cpbulk_dst_1 = reinterpret_cast<void*>(Logits + (row_base + (unsigned long long)j_7));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_1), "r"(neginf_scratch_addr), "r"((uint32_t)(bulk_elems_1 * 4))
                                            : "memory");
                                    }
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            __syncwarp();
                        }
                    }
                }
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                __syncwarp();
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 10) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
