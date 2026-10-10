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
#define NUM_MAIN_STAGES 1
#define THREADS 256
#ifndef SM_COUNT
#error "SM_COUNT is a downstream specialization of this program; define it on the compile line"
#endif
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_deepgemm_dense_mqa_2c9f274eb5a2741c7895(unsigned int* __restrict__ Starts, unsigned int* __restrict__ Ends, unsigned int* __restrict__ Metadata, unsigned int num_q_tokens, unsigned int num_kv_tokens)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // === Task calls (dependency order) ===
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    unsigned int span_offset = (3 * SM_COUNT + 1) / 2 * 2;
    unsigned int _min_0 = ((Starts[0]) < (num_kv_tokens) ? (Starts[0]) : (num_kv_tokens));
    unsigned int start = _min_0;
    unsigned int _min_1 = ((Ends[0]) < (num_kv_tokens) ? (Ends[0]) : (num_kv_tokens));
    unsigned int end = _min_1;
    unsigned int span_base = start / 4 * 4;
    unsigned int total_work = (end - span_base + 255) / 256;
    unsigned int total_cost = total_work;
    if (total_work != 0) {
        total_cost += 1;
    }
    if (tid == 0) {
        Metadata[span_offset] = span_base;
        Metadata[span_offset + 1] = total_work;
    }
    if (tid < SM_COUNT) {
        unsigned int base = total_cost / (unsigned int)SM_COUNT;
        unsigned int remainder = total_cost % (unsigned int)SM_COUNT;
        int _min_2 = ((tid) < (remainder) ? (tid) : (remainder));
        unsigned int target = (unsigned int)tid * base + (unsigned int)_min_2;
        unsigned int qblock = 1;
        unsigned int split = 0;
        unsigned int coordinate = total_work;
        if (target != total_cost) {
            qblock = 0;
            unsigned int _max_0 = ((target) > (1) ? (target) : (1));
            unsigned int _min_3 = ((_max_0 - 1) < (total_work - 1) ? (_max_0 - 1) : (total_work - 1));
            split = _min_3;
            coordinate = split;
        }
        int _min_4 = ((tid + 1) < (remainder) ? (tid + 1) : (remainder));
        unsigned int target_0 = (unsigned int)(tid + 1) * base + (unsigned int)_min_4;
        unsigned int qblock_1 = 1;
        unsigned int split_2 = 0;
        unsigned int coordinate_3 = total_work;
        if (target_0 != total_cost) {
            qblock_1 = 0;
            unsigned int _max_1 = ((target_0) > (1) ? (target_0) : (1));
            unsigned int _min_5 = ((_max_1 - 1) < (total_work - 1) ? (_max_1 - 1) : (total_work - 1));
            split_2 = _min_5;
            coordinate_3 = split_2;
        }
        Metadata[2 * tid] = qblock;
        Metadata[2 * tid + 1] = split;
        Metadata[2 * SM_COUNT + tid] = coordinate_3 - coordinate;
    }
}

} // extern "C"
