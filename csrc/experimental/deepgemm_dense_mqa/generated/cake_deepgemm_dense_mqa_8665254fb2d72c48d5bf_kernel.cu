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
#define TMEM_NCOLS 384
#define TMEM_TMEM_ACC_OFFSET 0
#define NUM_Q_PIPE_STAGES 3
#define NUM_KV_PIPE_STAGES 5
#define NUM_TMEM_PIPE_STAGES 3
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
#define FULL_Q_BLOCKS 1
#ifndef SM_COUNT
#error "SM_COUNT is a downstream specialization of this program; define it on the compile line"
#endif
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_deepgemm_dense_mqa_8665254fb2d72c48d5bf(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap Q_scales_alias, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KV_scales, const __grid_constant__ CUtensorMap Weights, float* __restrict__ Logits, unsigned int* __restrict__ ScheduleMeta, float* __restrict__ CandidateValues, int* __restrict__ CandidateIndices, int* __restrict__ CandidateCounts, float* __restrict__ ScoreThresholds, int* __restrict__ cu_seq_len_k_start, int* __restrict__ cu_seq_len_k_end, unsigned int seq_len, unsigned int seq_len_kv, unsigned int stride_logits, unsigned int num_q_blocks, unsigned int num_kv_splits, unsigned int candidate_capacity)
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
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q_scales_alias))) : "memory"); }
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

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 176);
    if (warp == 10) {
        int _tmem_hold = smem + 176;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: load_q ----
    if (warp == 8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // load_q_main
            unsigned int load_q_stage = 0;
            unsigned int load_q_num_blocks = (seq_len + 4 - 1) / 4;
            unsigned int _phase_q_empty = 1;
            if (elect_sync()) {
                unsigned int first_q = bid;
                unsigned int first_split = 0;
                unsigned int remaining = 0;
                {
                    first_q = ScheduleMeta[bid * 2];
                    first_split = ScheduleMeta[bid * 2 + 1];
                    remaining = ScheduleMeta[2 * SM_COUNT + bid];
                }
                unsigned int first_q_0 = first_q;
                unsigned int first_split_1 = first_split;
                unsigned int remaining_2 = remaining;
                #pragma unroll 1
                for (unsigned int q_block_idx = first_q_0; q_block_idx < load_q_num_blocks; q_block_idx++) {
                    unsigned int scheduled_kv_start = 0;
                    unsigned int scheduled_num_splits = 1;
                    {
                        if (remaining_2 == 0) {
                            break;
                        }
                        unsigned int span_word = (unsigned int)((3 * SM_COUNT + 1) / 2 * 2) + q_block_idx * 2;
                        unsigned int base = ScheduleMeta[span_word];
                        unsigned int span_splits = ScheduleMeta[span_word + 1];
                        unsigned int splits = 0;
                        if (first_split_1 < span_splits) {
                            unsigned int available = span_splits - first_split_1;
                            splits = ((available < remaining_2) ? available : remaining_2);
                        }
                        scheduled_kv_start = base + first_split_1 * 256;
                        scheduled_num_splits = splits;
                        remaining_2 -= scheduled_num_splits;
                        first_split_1 = 0;
                    }
                    if (scheduled_num_splits > 0) {
                        mbarrier_wait(q_empty_addr + (load_q_stage) * 8, _phase_q_empty);
                        tma_2d_gmem2smem(smem_q_addr + load_q_stage * 16384, (&Q), 0, q_block_idx * 128, q_full_addr + (load_q_stage) * 8);
                        tma_2d_gmem2smem(smem_weights_addr + load_q_stage * 512, (&Weights), 0, q_block_idx * 4, q_full_addr + (load_q_stage) * 8);
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
            unsigned int load_kv_num_blocks = (seq_len + 4 - 1) / 4;
            unsigned int first_q_1 = bid;
            unsigned int first_split_2 = 0;
            unsigned int remaining_1 = 0;
            {
                first_q_1 = ScheduleMeta[bid * 2];
                first_split_2 = ScheduleMeta[bid * 2 + 1];
                remaining_1 = ScheduleMeta[2 * SM_COUNT + bid];
            }
            unsigned int first_q_0_1 = first_q_1;
            unsigned int first_split_1_1 = first_split_2;
            unsigned int remaining_2_1 = remaining_1;
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
                    unsigned int base_1 = ScheduleMeta[span_word_1];
                    unsigned int span_splits_1 = ScheduleMeta[span_word_1 + 1];
                    unsigned int splits_1 = 0;
                    if (first_split_1_1 < span_splits_1) {
                        unsigned int available_1 = span_splits_1 - first_split_1_1;
                        splits_1 = ((available_1 < remaining_2_1) ? available_1 : remaining_2_1);
                    }
                    scheduled_kv_start_1 = base_1 + first_split_1_1 * 256;
                    scheduled_num_splits_1 = splits_1;
                    remaining_2_1 -= scheduled_num_splits_1;
                    first_split_1_1 = 0;
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
            unsigned int mma_num_blocks = (seq_len + 4 - 1) / 4;
            unsigned int first_q_2 = bid;
            unsigned int first_split_3 = 0;
            unsigned int remaining_3 = 0;
            {
                first_q_2 = ScheduleMeta[bid * 2];
                first_split_3 = ScheduleMeta[bid * 2 + 1];
                remaining_3 = ScheduleMeta[2 * SM_COUNT + bid];
            }
            unsigned int first_q_0_2 = first_q_2;
            unsigned int first_split_1_2 = first_split_3;
            unsigned int remaining_2_2 = remaining_3;
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
                    unsigned int base_2 = ScheduleMeta[span_word_2];
                    unsigned int span_splits_2 = ScheduleMeta[span_word_2 + 1];
                    unsigned int splits_2 = 0;
                    if (first_split_1_2 < span_splits_2) {
                        unsigned int available_2 = span_splits_2 - first_split_1_2;
                        splits_2 = ((available_2 < remaining_2_2) ? available_2 : remaining_2_2);
                    }
                    scheduled_kv_start_2 = base_2 + first_split_1_2 * 256;
                    scheduled_num_splits_2 = splits_2;
                    remaining_2_2 -= scheduled_num_splits_2;
                    first_split_1_2 = 0;
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
                            if (mma_tmem_stage == 3) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
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
                            if (mma_tmem_stage == 3) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
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
            {
                float neg_inf[4];
                #pragma unroll
                for (int component = 0; component < 4; component++) {
                    neg_inf[component] = -CUDART_INF_F;
                }
                for (unsigned int scratch_j = (unsigned int)lane; scratch_j < 1024; scratch_j += 32) {
                    neginf_scratch[scratch_j] = -CUDART_INF_F;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                unsigned int clean_num_q_blocks = (seq_len + 4 - 1) / 4;
                unsigned int clean_tiles_per_row = (stride_logits + 4095) / 4096;
                unsigned int clean_tasks = clean_num_q_blocks * clean_tiles_per_row;
                #pragma unroll 1
                for (unsigned int clean_task = bid; clean_task < clean_tasks; clean_task += SM_COUNT) {
                    unsigned int q_block_idx_3 = clean_task / clean_tiles_per_row;
                    unsigned int clean_begin = clean_task % clean_tiles_per_row * 4096;
                    unsigned int clean_end_raw = clean_begin + 4096;
                    unsigned int _min_0 = ((clean_end_raw) < (stride_logits) ? (clean_end_raw) : (stride_logits));
                    unsigned int clean_end = _min_0;
                    unsigned int q_start = q_block_idx_3 * 4;
                    unsigned int start_v = 4294967295;
                    unsigned int end_v = 0;
                    #pragma unroll 8
                    for (unsigned int token_idx = 0; token_idx < 4; token_idx++) {
                        unsigned int row_unclamped = q_start + token_idx;
                        unsigned int row_idx = ((row_unclamped < seq_len - 1) ? row_unclamped : seq_len - 1);
                        unsigned int k_start_raw = cu_seq_len_k_start[row_idx];
                        unsigned int k_end_raw = cu_seq_len_k_end[row_idx];
                        unsigned int k_start = ((k_start_raw < seq_len_kv) ? k_start_raw : seq_len_kv);
                        unsigned int k_end = ((k_end_raw < seq_len_kv) ? k_end_raw : seq_len_kv);
                        start_v = ((start_v < k_start) ? start_v : k_start);
                        end_v = ((end_v > k_end) ? end_v : k_end);
                    }
                    unsigned int kv_start_2 = start_v / 4 * 4;
                    unsigned int num_kv_blocks_2 = (end_v - kv_start_2 + 256 - 1) / 256;
                    unsigned int raw_end = kv_start_2 + num_kv_blocks_2 * 256;
                    unsigned int coverage_end = ((raw_end < stride_logits) ? raw_end : stride_logits);
                    #pragma unroll 1
                    for (unsigned int qi = 0; qi < 4; qi++) {
                        unsigned long long row_base = (unsigned long long)(q_start + qi) * (unsigned long long)stride_logits;
                        {
                            float* row_ptr = Logits + row_base;
                            unsigned int _min_1 = ((kv_start_2) < (clean_end) ? (kv_start_2) : (clean_end));
                            unsigned int aligned_start = (clean_begin + 3) / 4 * 4;
                            unsigned int aligned_end = _min_1 / 4 * 4;
                            if (aligned_start >= aligned_end) {
                                for (unsigned int j = clean_begin + (unsigned int)lane; j < _min_1; j += 32) {
                                    *(reinterpret_cast<float*>(row_ptr + j) + (0)) = -CUDART_INF_F;
                                }
                                __syncwarp();
                            } else {
                                for (unsigned int j_1 = clean_begin + (unsigned int)lane; j_1 < aligned_start; j_1 += 32) {
                                    *(reinterpret_cast<float*>(row_ptr + j_1) + (0)) = -CUDART_INF_F;
                                }
                                for (unsigned int j_2 = aligned_end + (unsigned int)lane; j_2 < _min_1; j_2 += 32) {
                                    *(reinterpret_cast<float*>(row_ptr + j_2) + (0)) = -CUDART_INF_F;
                                }
                                __syncwarp();
                                if (elect_sync()) {
                                    for (unsigned int j_3 = aligned_start; j_3 < aligned_end; j_3 += 1024) {
                                        unsigned int _min_2 = ((aligned_end - j_3) < (1024) ? (aligned_end - j_3) : (1024));
                                        unsigned int bulk_elems = _min_2;
                                        {
                                            void* _cpbulk_dst_0 = reinterpret_cast<void*>(row_ptr + j_3);
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
                            unsigned int _max_0 = ((coverage_end) > (clean_begin) ? (coverage_end) : (clean_begin));
                            unsigned int aligned_start_0 = (_max_0 + 3) / 4 * 4;
                            unsigned int aligned_end_1 = clean_end / 4 * 4;
                            if (aligned_start_0 >= aligned_end_1) {
                                for (unsigned int j_4 = _max_0 + (unsigned int)lane; j_4 < clean_end; j_4 += 32) {
                                    *(reinterpret_cast<float*>(row_ptr + j_4) + (0)) = -CUDART_INF_F;
                                }
                                __syncwarp();
                            } else {
                                for (unsigned int j_5 = _max_0 + (unsigned int)lane; j_5 < aligned_start_0; j_5 += 32) {
                                    *(reinterpret_cast<float*>(row_ptr + j_5) + (0)) = -CUDART_INF_F;
                                }
                                for (unsigned int j_6 = aligned_end_1 + (unsigned int)lane; j_6 < clean_end; j_6 += 32) {
                                    *(reinterpret_cast<float*>(row_ptr + j_6) + (0)) = -CUDART_INF_F;
                                }
                                __syncwarp();
                                if (elect_sync()) {
                                    for (unsigned int j_7 = aligned_start_0; j_7 < aligned_end_1; j_7 += 1024) {
                                        unsigned int _min_3 = ((aligned_end_1 - j_7) < (1024) ? (aligned_end_1 - j_7) : (1024));
                                        unsigned int bulk_elems_1 = _min_3;
                                        {
                                            void* _cpbulk_dst_1 = reinterpret_cast<void*>(row_ptr + j_7);
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
                }
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
                __syncwarp();
            }
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
            unsigned int math_num_blocks = (seq_len + 4 - 1) / 4;
            unsigned int first_q_3 = bid;
            unsigned int first_split_4 = 0;
            unsigned int remaining_4 = 0;
            {
                first_q_3 = ScheduleMeta[bid * 2];
                first_split_4 = ScheduleMeta[bid * 2 + 1];
                remaining_4 = ScheduleMeta[2 * SM_COUNT + bid];
            }
            unsigned int first_q_0_3 = first_q_3;
            unsigned int first_split_1_3 = first_split_4;
            unsigned int remaining_2_3 = remaining_4;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_kv_full_1 = 0;
            #pragma unroll 1
            for (unsigned int q_block_idx_4 = first_q_0_3; q_block_idx_4 < math_num_blocks; q_block_idx_4++) {
                unsigned int scheduled_kv_start_3 = 0;
                unsigned int scheduled_num_splits_3 = 1;
                {
                    if (remaining_2_3 == 0) {
                        break;
                    }
                    unsigned int span_word_3 = (unsigned int)((3 * SM_COUNT + 1) / 2 * 2) + q_block_idx_4 * 2;
                    unsigned int base_3 = ScheduleMeta[span_word_3];
                    unsigned int span_splits_3 = ScheduleMeta[span_word_3 + 1];
                    unsigned int splits_3 = 0;
                    if (first_split_1_3 < span_splits_3) {
                        unsigned int available_3 = span_splits_3 - first_split_1_3;
                        splits_3 = ((available_3 < remaining_2_3) ? available_3 : remaining_2_3);
                    }
                    scheduled_kv_start_3 = base_3 + first_split_1_3 * 256;
                    scheduled_num_splits_3 = splits_3;
                    remaining_2_3 -= scheduled_num_splits_3;
                    first_split_1_3 = 0;
                }
                if (scheduled_num_splits_3 > 0) {
                    unsigned int q_start_1 = q_block_idx_4 * 4;
                    unsigned int last = seq_len - 1;
                    unsigned int q0 = ((last > q_start_1) ? q_start_1 : last);
                    unsigned int q1 = ((last > q_start_1 + 1) ? q_start_1 + 1 : last);
                    unsigned int q2 = ((last > q_start_1 + 2) ? q_start_1 + 2 : last);
                    unsigned int q3 = ((last > q_start_1 + 3) ? q_start_1 + 3 : last);
                    unsigned int ks_l0 = cu_seq_len_k_start[q0];
                    unsigned int ks_l1 = cu_seq_len_k_start[q1];
                    unsigned int ks_l2 = cu_seq_len_k_start[q2];
                    unsigned int ks_l3 = cu_seq_len_k_start[q3];
                    unsigned int ks0 = ((ks_l0 < seq_len_kv) ? ks_l0 : seq_len_kv);
                    unsigned int ks1 = ((ks_l1 < seq_len_kv) ? ks_l1 : seq_len_kv);
                    unsigned int ks2 = ((ks_l2 < seq_len_kv) ? ks_l2 : seq_len_kv);
                    unsigned int ks3 = ((ks_l3 < seq_len_kv) ? ks_l3 : seq_len_kv);
                    unsigned int ke_l0 = cu_seq_len_k_end[q0];
                    unsigned int ke_l1 = cu_seq_len_k_end[q1];
                    unsigned int ke_l2 = cu_seq_len_k_end[q2];
                    unsigned int ke_l3 = cu_seq_len_k_end[q3];
                    unsigned int ke0 = ((ke_l0 < seq_len_kv) ? ke_l0 : seq_len_kv);
                    unsigned int ke1 = ((ke_l1 < seq_len_kv) ? ke_l1 : seq_len_kv);
                    unsigned int ke2 = ((ke_l2 < seq_len_kv) ? ke_l2 : seq_len_kv);
                    unsigned int ke3 = ((ke_l3 < seq_len_kv) ? ke_l3 : seq_len_kv);
                    unsigned int start_01 = ((ks0 < ks1) ? ks0 : ks1);
                    unsigned int start_012 = ((start_01 < ks2) ? start_01 : ks2);
                    unsigned int start_v_1 = ((start_012 < ks3) ? start_012 : ks3);
                    unsigned int end_01 = ((ke0 > ke1) ? ke0 : ke1);
                    unsigned int end_012 = ((end_01 > ke2) ? end_01 : ke2);
                    unsigned int end_v_1 = ((end_012 > ke3) ? end_012 : ke3);
                    unsigned int aligned_start_1 = start_v_1 / 4 * 4;
                    unsigned int kv_start_3 = aligned_start_1;
                    unsigned int kv_end = end_v_1;
                    unsigned int kv_span = kv_end - kv_start_3;
                    unsigned int num_kv_blocks_3 = (kv_span + 256 - 1) / 256;
                    unsigned int q_start_0 = q_start_1;
                    unsigned int kv_start_1_1 = kv_start_3;
                    unsigned int num_kv_blocks_2_1 = num_kv_blocks_3;
                    {
                        kv_start_1_1 = scheduled_kv_start_3;
                        num_kv_blocks_2_1 = scheduled_num_splits_3;
                    }
                    unsigned int dense_start[4];
                    unsigned int dense_end[4];
                    {
                        dense_start[0] = ks0;
                        dense_end[0] = ke0;
                        dense_start[1] = ks1;
                        dense_end[1] = ke1;
                        dense_start[2] = ks2;
                        dense_end[2] = ke2;
                        dense_start[3] = ks3;
                        dense_end[3] = ke3;
                    }
                    mbarrier_wait(q_full_addr + (math_q_stage) * 8, _phase_q_full_1);
                    float weights_reg[128];
                    int weight_stage_base = math_q_stage * 128;
                    #pragma unroll
                    for (int wi = 0; wi < 128; wi++) {
                        weights_reg[wi] = smem_weights[weight_stage_base + wi];
                    }
                    int q_row_valid[4];
                    int ks_qi[4];
                    int ke_qi[4];
                    float threshold_qi[4];
                    int queue_counts[4];
                    #pragma unroll
                    for (int qi_pre = 0; qi_pre < 4; qi_pre++) {
                        int q_row_pre = q_start_0 + (unsigned int)qi_pre;
                        {
                            q_row_valid[qi_pre] = 1;
                        }
                    }
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
                            int q_row = q_start_0;
                            float in_range_result = scale_kv * _relu_wsum_0;
                            {
                                float materialized_result = in_range_result;
                                {
                                    unsigned int rel_kv = (unsigned int)kv_pos - dense_start[0];
                                    unsigned int row_len = dense_end[0] - dense_start[0];
                                    materialized_result = ((rel_kv < row_len) ? in_range_result : -CUDART_INF_F);
                                }
                                {
                                    int out_elem = (unsigned int)q_row * stride_logits + (unsigned int)kv_pos;
                                    *(reinterpret_cast<float*>(Logits + out_elem) + (0)) = materialized_result;
                                }
                            }
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
                            int q_row_1 = q_start_0 + 1;
                            float in_range_result_1 = scale_kv * _relu_wsum_1;
                            {
                                float materialized_result_1 = in_range_result_1;
                                {
                                    unsigned int rel_kv_1 = (unsigned int)kv_pos - dense_start[1];
                                    unsigned int row_len_1 = dense_end[1] - dense_start[1];
                                    materialized_result_1 = ((rel_kv_1 < row_len_1) ? in_range_result_1 : -CUDART_INF_F);
                                }
                                {
                                    int out_elem_1 = (unsigned int)q_row_1 * stride_logits + (unsigned int)kv_pos;
                                    *(reinterpret_cast<float*>(Logits + out_elem_1) + (0)) = materialized_result_1;
                                }
                            }
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
                            int q_row_2 = q_start_0 + 2;
                            float in_range_result_2 = scale_kv * _relu_wsum_2;
                            {
                                float materialized_result_2 = in_range_result_2;
                                {
                                    unsigned int rel_kv_2 = (unsigned int)kv_pos - dense_start[2];
                                    unsigned int row_len_2 = dense_end[2] - dense_start[2];
                                    materialized_result_2 = ((rel_kv_2 < row_len_2) ? in_range_result_2 : -CUDART_INF_F);
                                }
                                {
                                    int out_elem_2 = (unsigned int)q_row_2 * stride_logits + (unsigned int)kv_pos;
                                    *(reinterpret_cast<float*>(Logits + out_elem_2) + (0)) = materialized_result_2;
                                }
                            }
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
                            int q_row_3 = q_start_0 + 3;
                            float in_range_result_3 = scale_kv * _relu_wsum_3;
                            {
                                float materialized_result_3 = in_range_result_3;
                                {
                                    unsigned int rel_kv_3 = (unsigned int)kv_pos - dense_start[3];
                                    unsigned int row_len_3 = dense_end[3] - dense_start[3];
                                    materialized_result_3 = ((rel_kv_3 < row_len_3) ? in_range_result_3 : -CUDART_INF_F);
                                }
                                {
                                    int out_elem_3 = (unsigned int)q_row_3 * stride_logits + (unsigned int)kv_pos;
                                    *(reinterpret_cast<float*>(Logits + out_elem_3) + (0)) = materialized_result_3;
                                }
                            }
                        }
                        math_tmem_stage += 2;
                        if (math_tmem_stage >= 3) {
                            math_tmem_stage -= 3;
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
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
            }
        }
    }

    // Cleanup
}

} // extern "C"
