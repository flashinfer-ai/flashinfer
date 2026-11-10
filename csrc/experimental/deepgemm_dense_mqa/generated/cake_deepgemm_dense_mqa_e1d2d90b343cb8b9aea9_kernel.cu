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
#define TMEM_NCOLS 256
#define TMEM_TMEM_ACC_OFFSET 0
#define NUM_Q_PIPE_STAGES 3
#define NUM_KV_PIPE_STAGES 5
#define NUM_TMEM_PIPE_STAGES 2
#define SMEM_NEGINF_SCRATCH_OFF 220672
#define SMEM_NEGINF_SCRATCH_STAGE_BYTES 4096
#define SMEM_NEGINF_SCRATCH_STRIDE 4096
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_WEIGHTS_OFF 50176
#define SMEM_SMEM_WEIGHTS_STAGE_BYTES 512
#define SMEM_SMEM_WEIGHTS_STRIDE 512
#define SMEM_SMEM_KV_OFF 51712
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 33792
#define SMEM_SMEM_KV_SCALES_OFF 84480
#define SMEM_SMEM_KV_SCALES_STAGE_BYTES 1024
#define SMEM_SMEM_KV_SCALES_STRIDE 33792
#define SMEM_CANDIDATE_QUEUE_VALUES_OFF 153088
#define SMEM_CANDIDATE_QUEUE_VALUES_STAGE_BYTES 8192
#define SMEM_CANDIDATE_QUEUE_VALUES_STRIDE 8192
#define SMEM_CANDIDATE_QUEUE_INDICES_OFF 161280
#define SMEM_CANDIDATE_QUEUE_INDICES_STAGE_BYTES 8192
#define SMEM_CANDIDATE_QUEUE_INDICES_STRIDE 8192
#define SMEM_TOTAL 224768
#define CANDIDATE_MODE 0
#define FULL_Q_BLOCKS 0
#ifndef SM_COUNT
#error "SM_COUNT is a downstream specialization of this program; define it on the compile line"
#endif
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_deepgemm_dense_mqa_e1d2d90b343cb8b9aea9(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap Q_scales_alias, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KV_scales, const __grid_constant__ CUtensorMap Weights, float* __restrict__ Logits, unsigned int* __restrict__ ScheduleMeta, float* __restrict__ CandidateValues, int* __restrict__ CandidateIndices, int* __restrict__ CandidateCounts, float* __restrict__ ScoreThresholds, int* __restrict__ cu_seq_len_k_start, int* __restrict__ cu_seq_len_k_end, unsigned int seq_len, unsigned int seq_len_kv, unsigned int stride_logits, unsigned int num_q_blocks, unsigned int num_kv_splits, unsigned int candidate_capacity)
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
    #define umma_empty_addr (mbar_base + 144)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* neginf_scratch = reinterpret_cast<float*>(smem_raw + SMEM_NEGINF_SCRATCH_OFF);
    const int neginf_scratch_addr = smem + SMEM_NEGINF_SCRATCH_OFF;
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_Q_OFF);
    const int smem_q_addr = smem + SMEM_SMEM_Q_OFF;
    float* smem_weights = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_WEIGHTS_OFF);
    const int smem_weights_addr = smem + SMEM_SMEM_WEIGHTS_OFF;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_KV_OFF);
    const int smem_kv_addr = smem + SMEM_SMEM_KV_OFF;
    float* smem_kv_scales = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_KV_SCALES_OFF);
    const int smem_kv_scales_addr = smem + SMEM_SMEM_KV_SCALES_OFF;
    float* candidate_queue_values = reinterpret_cast<float*>(smem_raw + SMEM_CANDIDATE_QUEUE_VALUES_OFF);
    const int candidate_queue_values_addr = smem + SMEM_CANDIDATE_QUEUE_VALUES_OFF;
    int* candidate_queue_indices = reinterpret_cast<int*>(smem_raw + SMEM_CANDIDATE_QUEUE_INDICES_OFF);
    const int candidate_queue_indices_addr = smem + SMEM_CANDIDATE_QUEUE_INDICES_OFF;
    if (tid == 0) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        unsigned int pf_row0 = bid * 2;
        unsigned int pf_row1 = pf_row0 + 1;
        if (pf_row0 < seq_len) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(cu_seq_len_k_start + pf_row0))); }
        if (pf_row0 < seq_len) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(cu_seq_len_k_end + pf_row0))); }
        if (pf_row1 < seq_len) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(cu_seq_len_k_start + pf_row1))); }
        if (pf_row1 < seq_len) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(cu_seq_len_k_end + pf_row1))); }
    }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q_scales_alias))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Weights))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV_scales))) : "memory"); }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

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
            // umma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // umma_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 144, 128);
            mbarrier_init(smem + 152, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 10) {
        int _tmem_hold = smem + 160;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (tid == 0) {
        unsigned int ft_row0 = bid * 2;
        if (ft_row0 < seq_len) {
            int ft_q_row = bid * 128;
            int ft_w_row = ft_row0;
            asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&Q))), "r"((int)(0)), "r"((int)(ft_q_row)) : "memory");
            asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&Weights))), "r"((int)(0)), "r"((int)(ft_w_row)) : "memory");
            if (bid == 0) {
                unsigned int ft_row1 = ft_row0 + 1;
                unsigned int ft_row1c = ((ft_row1 < seq_len) ? ft_row1 : ft_row0);
                unsigned int _min_0 = (((unsigned int)cu_seq_len_k_start[ft_row0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[ft_row0]) : (seq_len_kv));
                unsigned int ft_ks0 = _min_0;
                unsigned int _min_1 = (((unsigned int)cu_seq_len_k_start[ft_row1c]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[ft_row1c]) : (seq_len_kv));
                unsigned int ft_ks1 = _min_1;
                unsigned int _min_2 = ((ft_ks0) < (ft_ks1) ? (ft_ks0) : (ft_ks1));
                unsigned int ft_ks = _min_2;
                int ft_kv_row = (ft_ks >> 2) * 4;
                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&KV))), "r"((int)(0)), "r"((int)(ft_kv_row)) : "memory");
                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&KV_scales))), "r"((int)(ft_kv_row)), "r"((int)(0)) : "memory");
            }
        }
    }
    if (tid == 0) {
        unsigned int _min_3 = (((unsigned int)cu_seq_len_k_start[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : (seq_len_kv));
        unsigned int start = _min_3;
        unsigned int _min_4 = (((unsigned int)cu_seq_len_k_end[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : (seq_len_kv));
        unsigned int end = _min_4;
        unsigned int base = start / 4 * 4;
        unsigned int work = (end - base + 255) / 256;
        unsigned int total_work = work;
        unsigned int total_cost = total_work;
        if (total_work != 0) {
            total_cost += 1;
        }
        unsigned int quotient = total_cost / (unsigned int)SM_COUNT;
        unsigned int remainder = total_cost % (unsigned int)SM_COUNT;
        int _min_5 = ((bid) < (remainder) ? (bid) : (remainder));
        unsigned int begin = (unsigned int)bid * quotient + (unsigned int)_min_5;
        int _min_6 = ((bid + 1) < (remainder) ? (bid + 1) : (remainder));
        unsigned int end_0 = (unsigned int)(bid + 1) * quotient + (unsigned int)_min_6;
        unsigned int first_q = 1;
        unsigned int first_split = 0;
        unsigned int begin_coord = total_work;
        unsigned int end_coord = total_work;
        if (begin != total_cost) {
            first_q = 0;
            unsigned int _max_0 = ((begin) > (1) ? (begin) : (1));
            unsigned int _min_7 = ((_max_0 - 1) < (total_work - 1) ? (_max_0 - 1) : (total_work - 1));
            first_split = _min_7;
            begin_coord = first_split;
        }
        if (end_0 != total_cost) {
            unsigned int _max_1 = ((end_0) > (1) ? (end_0) : (1));
            unsigned int _min_8 = ((_max_1 - 1) < (total_work - 1) ? (_max_1 - 1) : (total_work - 1));
            end_coord = _min_8;
        }
        unsigned int remaining = end_coord - begin_coord;
        ScheduleMeta[2 * bid] = first_q;
        ScheduleMeta[2 * bid + 1] = first_split;
        ScheduleMeta[2 * SM_COUNT + bid] = remaining;
        if (bid == 0) {
            unsigned int _min_9 = (((unsigned int)cu_seq_len_k_start[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : (seq_len_kv));
            unsigned int start_0 = _min_9;
            unsigned int _min_10 = (((unsigned int)cu_seq_len_k_end[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : (seq_len_kv));
            unsigned int end_1 = _min_10;
            unsigned int base_2 = start_0 / 4 * 4;
            unsigned int work_3 = (end_1 - base_2 + 255) / 256;
            ScheduleMeta[(3 * SM_COUNT + 1) / 2 * 2] = base_2;
            ScheduleMeta[(3 * SM_COUNT + 1) / 2 * 2 + 1] = work_3;
        }
    }

    // ---- Role: load_q ----
    if (warp == 8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // load_q_main
            unsigned int load_q_stage = 0;
            unsigned int load_q_num_blocks = (seq_len + 2 - 1) / 2;
            unsigned int _phase_q_empty = 1;
            if (elect_sync()) {
                unsigned int first_q_1 = bid;
                unsigned int first_split_1 = 0;
                unsigned int remaining_1 = 0;
                {
                    unsigned int _min_18 = (((unsigned int)cu_seq_len_k_start[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : (seq_len_kv));
                    unsigned int start_1 = _min_18;
                    unsigned int _min_19 = (((unsigned int)cu_seq_len_k_end[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : (seq_len_kv));
                    unsigned int end_2 = _min_19;
                    unsigned int base_1 = start_1 / 4 * 4;
                    unsigned int work_1 = (end_2 - base_1 + 255) / 256;
                    unsigned int total_work_1 = work_1;
                    unsigned int total_cost_1 = total_work_1;
                    if (total_work_1 != 0) {
                        total_cost_1 += 1;
                    }
                    unsigned int quotient_1 = total_cost_1 / (unsigned int)SM_COUNT;
                    unsigned int remainder_1 = total_cost_1 % (unsigned int)SM_COUNT;
                    int _min_20 = ((bid) < (remainder_1) ? (bid) : (remainder_1));
                    unsigned int begin_1 = (unsigned int)bid * quotient_1 + (unsigned int)_min_20;
                    int _min_21 = ((bid + 1) < (remainder_1) ? (bid + 1) : (remainder_1));
                    unsigned int end_0_1 = (unsigned int)(bid + 1) * quotient_1 + (unsigned int)_min_21;
                    unsigned int first_q_1_1 = 1;
                    unsigned int first_split_2 = 0;
                    unsigned int begin_coord_1 = total_work_1;
                    unsigned int end_coord_1 = total_work_1;
                    if (begin_1 != total_cost_1) {
                        first_q_1_1 = 0;
                        unsigned int _max_4 = ((begin_1) > (1) ? (begin_1) : (1));
                        unsigned int _min_22 = ((_max_4 - 1) < (total_work_1 - 1) ? (_max_4 - 1) : (total_work_1 - 1));
                        first_split_2 = _min_22;
                        begin_coord_1 = first_split_2;
                    }
                    if (end_0_1 != total_cost_1) {
                        unsigned int _max_5 = ((end_0_1) > (1) ? (end_0_1) : (1));
                        unsigned int _min_23 = ((_max_5 - 1) < (total_work_1 - 1) ? (_max_5 - 1) : (total_work_1 - 1));
                        end_coord_1 = _min_23;
                    }
                    unsigned int remaining_3 = end_coord_1 - begin_coord_1;
                    first_q_1 = first_q_1_1;
                    first_split_1 = first_split_2;
                    remaining_1 = remaining_3;
                }
                unsigned int first_q_0 = first_q_1;
                unsigned int first_split_1_1 = first_split_1;
                unsigned int remaining_2 = remaining_1;
                #pragma unroll 1
                for (unsigned int q_block_idx = first_q_0; q_block_idx < load_q_num_blocks; q_block_idx++) {
                    unsigned int scheduled_kv_start = 0;
                    unsigned int scheduled_num_splits = 1;
                    {
                        if (remaining_2 == 0) {
                            break;
                        }
                        unsigned int span_word = (unsigned int)((3 * SM_COUNT + 1) / 2 * 2) + q_block_idx * 2;
                        unsigned int base_3 = 0;
                        unsigned int span_splits = 0;
                        unsigned int _min_24 = (((unsigned int)cu_seq_len_k_start[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : (seq_len_kv));
                        unsigned int start_2 = _min_24;
                        unsigned int _min_25 = (((unsigned int)cu_seq_len_k_end[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : (seq_len_kv));
                        unsigned int end_3 = _min_25;
                        unsigned int base_0 = start_2 / 4 * 4;
                        unsigned int work_2 = (end_3 - base_0 + 255) / 256;
                        base_3 = base_0;
                        span_splits = work_2;
                        unsigned int splits = 0;
                        if (first_split_1_1 < span_splits) {
                            unsigned int available = span_splits - first_split_1_1;
                            splits = ((available < remaining_2) ? available : remaining_2);
                        }
                        scheduled_kv_start = base_3 + first_split_1_1 * 256;
                        scheduled_num_splits = splits;
                        remaining_2 -= scheduled_num_splits;
                        first_split_1_1 = 0;
                    }
                    if (scheduled_num_splits > 0) {
                        mbarrier_wait(q_empty_addr + (load_q_stage) * 8, _phase_q_empty);
                        tma_2d_gmem2smem(smem_q_addr + load_q_stage * 16384, (&Q), 0, q_block_idx * 128, q_full_addr + (load_q_stage) * 8);
                        tma_2d_gmem2smem(smem_weights_addr + load_q_stage * 512, (&Weights), 0, q_block_idx * 2, q_full_addr + (load_q_stage) * 8);
                        mbarrier_arrive_expect_tx(q_full_addr + (load_q_stage) * 8, 16896);
                        load_q_stage += 1;
                        if (load_q_stage == 3) { load_q_stage = 0; _phase_q_empty ^= 1; }
                    }
                }
            }
            __syncwarp();
        }
    // ---- Role: load_kv ----
    } else if (warp == 9) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // load_kv_main
            unsigned int load_kv_stage = 0;
            unsigned int load_kv_num_blocks = (seq_len + 2 - 1) / 2;
            unsigned int first_q_2 = bid;
            unsigned int first_split_3 = 0;
            unsigned int remaining_4 = 0;
            {
                unsigned int _min_26 = (((unsigned int)cu_seq_len_k_start[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : (seq_len_kv));
                unsigned int start_3 = _min_26;
                unsigned int _min_27 = (((unsigned int)cu_seq_len_k_end[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : (seq_len_kv));
                unsigned int end_4 = _min_27;
                unsigned int base_4 = start_3 / 4 * 4;
                unsigned int work_4 = (end_4 - base_4 + 255) / 256;
                unsigned int total_work_2 = work_4;
                unsigned int total_cost_2 = total_work_2;
                if (total_work_2 != 0) {
                    total_cost_2 += 1;
                }
                unsigned int quotient_2 = total_cost_2 / (unsigned int)SM_COUNT;
                unsigned int remainder_2 = total_cost_2 % (unsigned int)SM_COUNT;
                int _min_28 = ((bid) < (remainder_2) ? (bid) : (remainder_2));
                unsigned int begin_2 = (unsigned int)bid * quotient_2 + (unsigned int)_min_28;
                int _min_29 = ((bid + 1) < (remainder_2) ? (bid + 1) : (remainder_2));
                unsigned int end_0_2 = (unsigned int)(bid + 1) * quotient_2 + (unsigned int)_min_29;
                unsigned int first_q_1_2 = 1;
                unsigned int first_split_2_1 = 0;
                unsigned int begin_coord_2 = total_work_2;
                unsigned int end_coord_2 = total_work_2;
                if (begin_2 != total_cost_2) {
                    first_q_1_2 = 0;
                    unsigned int _max_6 = ((begin_2) > (1) ? (begin_2) : (1));
                    unsigned int _min_30 = ((_max_6 - 1) < (total_work_2 - 1) ? (_max_6 - 1) : (total_work_2 - 1));
                    first_split_2_1 = _min_30;
                    begin_coord_2 = first_split_2_1;
                }
                if (end_0_2 != total_cost_2) {
                    unsigned int _max_7 = ((end_0_2) > (1) ? (end_0_2) : (1));
                    unsigned int _min_31 = ((_max_7 - 1) < (total_work_2 - 1) ? (_max_7 - 1) : (total_work_2 - 1));
                    end_coord_2 = _min_31;
                }
                unsigned int remaining_3_1 = end_coord_2 - begin_coord_2;
                first_q_2 = first_q_1_2;
                first_split_3 = first_split_2_1;
                remaining_4 = remaining_3_1;
            }
            unsigned int first_q_0_1 = first_q_2;
            unsigned int first_split_1_2 = first_split_3;
            unsigned int remaining_2_1 = remaining_4;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int q_block_idx_1 = first_q_0_1; q_block_idx_1 < load_kv_num_blocks; q_block_idx_1++) {
                unsigned int scheduled_kv_start_1 = 0;
                unsigned int scheduled_num_splits_1 = 1;
                {
                    if (remaining_2_1 == 0) {
                        break;
                    }
                    unsigned int span_word_1 = (unsigned int)((3 * SM_COUNT + 1) / 2 * 2) + q_block_idx_1 * 2;
                    unsigned int base_5 = 0;
                    unsigned int span_splits_1 = 0;
                    unsigned int _min_32 = (((unsigned int)cu_seq_len_k_start[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : (seq_len_kv));
                    unsigned int start_4 = _min_32;
                    unsigned int _min_33 = (((unsigned int)cu_seq_len_k_end[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : (seq_len_kv));
                    unsigned int end_5 = _min_33;
                    unsigned int base_0_1 = start_4 / 4 * 4;
                    unsigned int work_5 = (end_5 - base_0_1 + 255) / 256;
                    base_5 = base_0_1;
                    span_splits_1 = work_5;
                    unsigned int splits_1 = 0;
                    if (first_split_1_2 < span_splits_1) {
                        unsigned int available_1 = span_splits_1 - first_split_1_2;
                        splits_1 = ((available_1 < remaining_2_1) ? available_1 : remaining_2_1);
                    }
                    scheduled_kv_start_1 = base_5 + first_split_1_2 * 256;
                    scheduled_num_splits_1 = splits_1;
                    remaining_2_1 -= scheduled_num_splits_1;
                    first_split_1_2 = 0;
                }
                if (scheduled_num_splits_1 > 0) {
                    unsigned int kv_start = scheduled_kv_start_1;
                    unsigned int num_kv_blocks = scheduled_num_splits_1;
                    #pragma unroll 1
                    for (unsigned int kv_iter = 0; kv_iter < num_kv_blocks; kv_iter++) {
                        int kv_row = kv_start + kv_iter * 256;
                        if (elect_sync()) {
                            mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                            tma_2d_gmem2smem(smem_kv_addr + load_kv_stage * 33792, (&KV), 0, kv_row, kv_full_addr + (load_kv_stage) * 8);
                            tma_2d_gmem2smem(smem_kv_scales_addr + load_kv_stage * 33792, (&KV_scales), kv_row, 0, kv_full_addr + (load_kv_stage) * 8);
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 33792);
                            load_kv_stage += 1;
                            if (load_kv_stage == 5) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                        }
                        __syncwarp();
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 10) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // mma_main
            unsigned int mma_q_stage = 0;
            unsigned int mma_kv_stage = 0;
            unsigned int mma_tmem_stage = 0;
            unsigned int mma_num_blocks = (seq_len + 2 - 1) / 2;
            unsigned int first_q_3 = bid;
            unsigned int first_split_4 = 0;
            unsigned int remaining_5 = 0;
            {
                unsigned int _min_34 = (((unsigned int)cu_seq_len_k_start[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : (seq_len_kv));
                unsigned int start_5 = _min_34;
                unsigned int _min_35 = (((unsigned int)cu_seq_len_k_end[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : (seq_len_kv));
                unsigned int end_6 = _min_35;
                unsigned int base_6 = start_5 / 4 * 4;
                unsigned int work_6 = (end_6 - base_6 + 255) / 256;
                unsigned int total_work_3 = work_6;
                unsigned int total_cost_3 = total_work_3;
                if (total_work_3 != 0) {
                    total_cost_3 += 1;
                }
                unsigned int quotient_3 = total_cost_3 / (unsigned int)SM_COUNT;
                unsigned int remainder_3 = total_cost_3 % (unsigned int)SM_COUNT;
                int _min_36 = ((bid) < (remainder_3) ? (bid) : (remainder_3));
                unsigned int begin_3 = (unsigned int)bid * quotient_3 + (unsigned int)_min_36;
                int _min_37 = ((bid + 1) < (remainder_3) ? (bid + 1) : (remainder_3));
                unsigned int end_0_3 = (unsigned int)(bid + 1) * quotient_3 + (unsigned int)_min_37;
                unsigned int first_q_1_3 = 1;
                unsigned int first_split_2_2 = 0;
                unsigned int begin_coord_3 = total_work_3;
                unsigned int end_coord_3 = total_work_3;
                if (begin_3 != total_cost_3) {
                    first_q_1_3 = 0;
                    unsigned int _max_8 = ((begin_3) > (1) ? (begin_3) : (1));
                    unsigned int _min_38 = ((_max_8 - 1) < (total_work_3 - 1) ? (_max_8 - 1) : (total_work_3 - 1));
                    first_split_2_2 = _min_38;
                    begin_coord_3 = first_split_2_2;
                }
                if (end_0_3 != total_cost_3) {
                    unsigned int _max_9 = ((end_0_3) > (1) ? (end_0_3) : (1));
                    unsigned int _min_39 = ((_max_9 - 1) < (total_work_3 - 1) ? (_max_9 - 1) : (total_work_3 - 1));
                    end_coord_3 = _min_39;
                }
                unsigned int remaining_3_2 = end_coord_3 - begin_coord_3;
                first_q_3 = first_q_1_3;
                first_split_4 = first_split_2_2;
                remaining_5 = remaining_3_2;
            }
            unsigned int first_q_0_2 = first_q_3;
            unsigned int first_split_1_3 = first_split_4;
            unsigned int remaining_2_2 = remaining_5;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_umma_empty = 1;
            #pragma unroll 1
            for (unsigned int q_block_idx_2 = first_q_0_2; q_block_idx_2 < mma_num_blocks; q_block_idx_2++) {
                unsigned int scheduled_kv_start_2 = 0;
                unsigned int scheduled_num_splits_2 = 1;
                {
                    if (remaining_2_2 == 0) {
                        break;
                    }
                    unsigned int span_word_2 = (unsigned int)((3 * SM_COUNT + 1) / 2 * 2) + q_block_idx_2 * 2;
                    unsigned int base_7 = 0;
                    unsigned int span_splits_2 = 0;
                    unsigned int _min_40 = (((unsigned int)cu_seq_len_k_start[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : (seq_len_kv));
                    unsigned int start_6 = _min_40;
                    unsigned int _min_41 = (((unsigned int)cu_seq_len_k_end[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : (seq_len_kv));
                    unsigned int end_7 = _min_41;
                    unsigned int base_0_2 = start_6 / 4 * 4;
                    unsigned int work_7 = (end_7 - base_0_2 + 255) / 256;
                    base_7 = base_0_2;
                    span_splits_2 = work_7;
                    unsigned int splits_2 = 0;
                    if (first_split_1_3 < span_splits_2) {
                        unsigned int available_2 = span_splits_2 - first_split_1_3;
                        splits_2 = ((available_2 < remaining_2_2) ? available_2 : remaining_2_2);
                    }
                    scheduled_kv_start_2 = base_7 + first_split_1_3 * 256;
                    scheduled_num_splits_2 = splits_2;
                    remaining_2_2 -= scheduled_num_splits_2;
                    first_split_1_3 = 0;
                }
                if (scheduled_num_splits_2 > 0) {
                    unsigned int kv_start_1 = scheduled_kv_start_2;
                    unsigned int num_kv_blocks_1 = scheduled_num_splits_2;
                    mbarrier_wait(q_full_addr + (mma_q_stage) * 8, _phase_q_full);
                    #pragma unroll 1
                    for (unsigned int kv_iter_1 = 0; kv_iter_1 < num_kv_blocks_1; kv_iter_1++) {
                        mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                        if (elect_sync()) {
                            mbarrier_wait(umma_empty_addr + (mma_tmem_stage) * 8, _phase_umma_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_0 = (((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 2112;
                            int _mma_b_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (mma_q_stage) * 1024;
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
                    "mov.b32 id, 136314896;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_acc + (mma_tmem_stage * 128))), "r"(0));
                            tcgen05_commit(umma_full_addr + (mma_tmem_stage) * 8);
                            mma_tmem_stage += 1;
                            if (mma_tmem_stage == 2) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
                            mbarrier_wait(umma_empty_addr + (mma_tmem_stage) * 8, _phase_umma_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_1 = (((smem_kv_addr + 16384) >> 4) & 0x3FFF) + (mma_kv_stage) * 2112;
                            int _mma_b_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (mma_q_stage) * 1024;
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
                    "mov.b32 id, 136314896;\n\t"
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_acc + (mma_tmem_stage * 128))), "r"(0));
                            tcgen05_commit(umma_full_addr + (mma_tmem_stage) * 8);
                            mma_tmem_stage += 1;
                            if (mma_tmem_stage == 2) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
                        }
                        __syncwarp();
                        mma_kv_stage += 1;
                        if (mma_kv_stage == 5) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                    }
                    mbarrier_arrive(q_empty_addr + (mma_q_stage) * 8);
                    mma_q_stage += 1;
                    if (mma_q_stage == 3) { mma_q_stage = 0; _phase_q_full ^= 1; }
                }
            }
        }
    // ---- Role: clean ----
    } else if (warp == 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // clean_main

        }
    // ---- Role: math ----
    } else if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // math_main
            int local_thread_idx = warp % 4 * 32 + lane;
            unsigned int wg_idx = make_warp_uniform(warp / 4);
            int math_thread_idx = wg_idx * 128 + (unsigned int)local_thread_idx;
            unsigned int math_q_stage = 0;
            unsigned int math_kv_stage = 0;
            unsigned int math_tmem_stage = wg_idx;
            unsigned int math_tmem_phase = 0;
            unsigned int math_num_blocks = (seq_len + 2 - 1) / 2;
            unsigned int first_q_4 = bid;
            unsigned int first_split_5 = 0;
            unsigned int remaining_6 = 0;
            {
                unsigned int _min_42 = (((unsigned int)cu_seq_len_k_start[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : (seq_len_kv));
                unsigned int start_7 = _min_42;
                unsigned int _min_43 = (((unsigned int)cu_seq_len_k_end[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : (seq_len_kv));
                unsigned int end_8 = _min_43;
                unsigned int base_8 = start_7 / 4 * 4;
                unsigned int work_8 = (end_8 - base_8 + 255) / 256;
                unsigned int total_work_4 = work_8;
                unsigned int total_cost_4 = total_work_4;
                if (total_work_4 != 0) {
                    total_cost_4 += 1;
                }
                unsigned int quotient_4 = total_cost_4 / (unsigned int)SM_COUNT;
                unsigned int remainder_4 = total_cost_4 % (unsigned int)SM_COUNT;
                int _min_44 = ((bid) < (remainder_4) ? (bid) : (remainder_4));
                unsigned int begin_4 = (unsigned int)bid * quotient_4 + (unsigned int)_min_44;
                int _min_45 = ((bid + 1) < (remainder_4) ? (bid + 1) : (remainder_4));
                unsigned int end_0_4 = (unsigned int)(bid + 1) * quotient_4 + (unsigned int)_min_45;
                unsigned int first_q_1_4 = 1;
                unsigned int first_split_2_3 = 0;
                unsigned int begin_coord_4 = total_work_4;
                unsigned int end_coord_4 = total_work_4;
                if (begin_4 != total_cost_4) {
                    first_q_1_4 = 0;
                    unsigned int _max_10 = ((begin_4) > (1) ? (begin_4) : (1));
                    unsigned int _min_46 = ((_max_10 - 1) < (total_work_4 - 1) ? (_max_10 - 1) : (total_work_4 - 1));
                    first_split_2_3 = _min_46;
                    begin_coord_4 = first_split_2_3;
                }
                if (end_0_4 != total_cost_4) {
                    unsigned int _max_11 = ((end_0_4) > (1) ? (end_0_4) : (1));
                    unsigned int _min_47 = ((_max_11 - 1) < (total_work_4 - 1) ? (_max_11 - 1) : (total_work_4 - 1));
                    end_coord_4 = _min_47;
                }
                unsigned int remaining_3_3 = end_coord_4 - begin_coord_4;
                first_q_4 = first_q_1_4;
                first_split_5 = first_split_2_3;
                remaining_6 = remaining_3_3;
            }
            unsigned int first_q_0_3 = first_q_4;
            unsigned int first_split_1_4 = first_split_5;
            unsigned int remaining_2_3 = remaining_6;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_kv_full_1 = 0;
            #pragma unroll 1
            for (unsigned int q_block_idx_3 = first_q_0_3; q_block_idx_3 < math_num_blocks; q_block_idx_3++) {
                unsigned int scheduled_kv_start_3 = 0;
                unsigned int scheduled_num_splits_3 = 1;
                {
                    if (remaining_2_3 == 0) {
                        break;
                    }
                    unsigned int span_word_3 = (unsigned int)((3 * SM_COUNT + 1) / 2 * 2) + q_block_idx_3 * 2;
                    unsigned int base_9 = 0;
                    unsigned int span_splits_3 = 0;
                    unsigned int _min_48 = (((unsigned int)cu_seq_len_k_start[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[0]) : (seq_len_kv));
                    unsigned int start_8 = _min_48;
                    unsigned int _min_49 = (((unsigned int)cu_seq_len_k_end[0]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[0]) : (seq_len_kv));
                    unsigned int end_9 = _min_49;
                    unsigned int base_0_3 = start_8 / 4 * 4;
                    unsigned int work_9 = (end_9 - base_0_3 + 255) / 256;
                    base_9 = base_0_3;
                    span_splits_3 = work_9;
                    unsigned int splits_3 = 0;
                    if (first_split_1_4 < span_splits_3) {
                        unsigned int available_3 = span_splits_3 - first_split_1_4;
                        splits_3 = ((available_3 < remaining_2_3) ? available_3 : remaining_2_3);
                    }
                    scheduled_kv_start_3 = base_9 + first_split_1_4 * 256;
                    scheduled_num_splits_3 = splits_3;
                    remaining_2_3 -= scheduled_num_splits_3;
                    first_split_1_4 = 0;
                }
                if (scheduled_num_splits_3 > 0) {
                    unsigned int q_start = q_block_idx_3 * 2;
                    unsigned int last = seq_len - 1;
                    unsigned int q0 = ((last > q_start) ? q_start : last);
                    unsigned int q1 = ((last > q_start + 1) ? q_start + 1 : last);
                    unsigned int ks_l0 = cu_seq_len_k_start[q0];
                    unsigned int ks_l1 = cu_seq_len_k_start[q1];
                    unsigned int ks0 = ((ks_l0 < seq_len_kv) ? ks_l0 : seq_len_kv);
                    unsigned int ks1 = ((ks_l1 < seq_len_kv) ? ks_l1 : seq_len_kv);
                    unsigned int ke_l0 = cu_seq_len_k_end[q0];
                    unsigned int ke_l1 = cu_seq_len_k_end[q1];
                    unsigned int ke0 = ((ke_l0 < seq_len_kv) ? ke_l0 : seq_len_kv);
                    unsigned int ke1 = ((ke_l1 < seq_len_kv) ? ke_l1 : seq_len_kv);
                    unsigned int start_v = ((ks0 < ks1) ? ks0 : ks1);
                    unsigned int end_v = ((ke0 > ke1) ? ke0 : ke1);
                    unsigned int aligned_start = start_v / 4 * 4;
                    unsigned int kv_start_2 = aligned_start;
                    unsigned int kv_end = end_v;
                    unsigned int kv_span = kv_end - kv_start_2;
                    unsigned int num_kv_blocks_2 = (kv_span + 256 - 1) / 256;
                    unsigned int q_start_0 = q_start;
                    unsigned int kv_start_1_1 = kv_start_2;
                    unsigned int num_kv_blocks_2_1 = num_kv_blocks_2;
                    {
                        kv_start_1_1 = scheduled_kv_start_3;
                        num_kv_blocks_2_1 = scheduled_num_splits_3;
                    }
                    unsigned int dense_start[2];
                    unsigned int dense_end[2];
                    mbarrier_wait(q_full_addr + (math_q_stage) * 8, _phase_q_full_1);
                    float weights_reg[128];
                    int weight_stage_base = math_q_stage * 128;
                    #pragma unroll
                    for (int wi = 0; wi < 64; wi++) {
                        weights_reg[wi] = smem_weights[weight_stage_base + wi];
                    }
                    int q_row_valid[2];
                    int ks_qi[2];
                    int ke_qi[2];
                    float threshold_qi[2];
                    int queue_counts[2];
                    #pragma unroll
                    for (int qi_pre = 0; qi_pre < 1; qi_pre++) {
                        int q_row_pre = q_start_0 + (unsigned int)qi_pre;
                        {
                            int in_range = (((unsigned int)q_row_pre < seq_len) ? 1 : 0);
                            q_row_valid[qi_pre] = in_range;
                        }
                    }
                    unsigned long long block_row_base64 = (unsigned long long)q_start_0 * (unsigned long long)stride_logits;
                    float* block_base_ptr = Logits + block_row_base64;
                    #pragma unroll 1
                    for (unsigned int kv_iter_2 = 0; kv_iter_2 < num_kv_blocks_2_1; kv_iter_2++) {
                        mbarrier_wait(kv_full_addr + (math_kv_stage) * 8, _phase_kv_full_1);
                        int kv_scale_stage_base = math_kv_stage * 8448;
                        float scale_kv = smem_kv_scales[kv_scale_stage_base + math_thread_idx];
                        mbarrier_wait(umma_full_addr + (math_tmem_stage) * 8, math_tmem_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(kv_empty_addr + (math_kv_stage) * 8);
                        math_kv_stage += 1;
                        if (math_kv_stage == 5) { math_kv_stage = 0; _phase_kv_full_1 ^= 1; }
                        int kv_pos = kv_start_1_1 + kv_iter_2 * 256 + (unsigned int)math_thread_idx;
                        {
                            float _tmem_load_0[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                                : "r"(taddr + math_tmem_stage * 128 + (unsigned int)(warp % 4 * 32 << 16)));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                                : "r"(taddr + math_tmem_stage * 128 + (unsigned int)(warp % 4 * 32 << 16) + 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(umma_empty_addr + (math_tmem_stage) * 8);
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
                                        : "+l"(*(unsigned long long*)&_sum0) : "f"(_tmem_load_0[0 + _j + 0]), "f"(_tmem_load_0[0 + _j + 1]), "f"(weights_reg[0 + _j + 0]), "f"(weights_reg[0 + _j + 1]));
                                    asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                        "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                        "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                        "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                        "mov.b64 _pb, {%3, %4};\n\t"
                                        "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                        : "+l"(*(unsigned long long*)&_sum1) : "f"(_tmem_load_0[0 + _j + 2]), "f"(_tmem_load_0[0 + _j + 3]), "f"(weights_reg[0 + _j + 2]), "f"(weights_reg[0 + _j + 3]));
                                }
                                float2 _sum;
                                asm("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_sum) : "l"(*(const unsigned long long*)&_sum0), "l"(*(const unsigned long long*)&_sum1));
                                _relu_wsum_0 = (_sum.x + _sum.y) * 0.5f;
                            }
                            int q_row = q_start_0;
                            float in_range_result = scale_kv * _relu_wsum_0;
                            {
                                float materialized_result = in_range_result;
                                int q_rel = (unsigned int)q_row - q_start_0;
                                if (q_row_valid[0] != 0) {
                                    int out_elem = (unsigned int)q_rel * stride_logits + (unsigned int)kv_pos;
                                    *(reinterpret_cast<float*>(block_base_ptr + out_elem) + (0)) = materialized_result;
                                }
                            }
                        }
                        math_tmem_stage += 2;
                        if (math_tmem_stage >= 2) {
                            math_tmem_stage -= 2;
                            math_tmem_phase ^= 1;
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(q_empty_addr + (math_q_stage) * 8);
                    math_q_stage += 1;
                    if (math_q_stage == 3) { math_q_stage = 0; _phase_q_full_1 ^= 1; }
                }
            }
            asm volatile("barrier.sync 8, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(256));
            }
        }
    }

    // Cleanup
}

} // extern "C"
