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
#define NUM_MAIN_STAGES 1
#define SMEM_WORK_PREFIX_OFF 0
#define SMEM_WORK_PREFIX_STAGE_BYTES 16
#define SMEM_WORK_PREFIX_STRIDE 16
#define SMEM_COST_PREFIX_OFF 16
#define SMEM_COST_PREFIX_STAGE_BYTES 16
#define SMEM_COST_PREFIX_STRIDE 16
#define SMEM_WARP_SUMS_OFF 32
#define SMEM_WARP_SUMS_STAGE_BYTES 64
#define SMEM_WARP_SUMS_STRIDE 64
#define SMEM_CARRY_OFF 96
#define SMEM_CARRY_STAGE_BYTES 8
#define SMEM_CARRY_STRIDE 8
#define SMEM_TOTAL 128
#define THREADS 256
#ifndef SM_COUNT
#error "SM_COUNT is a downstream specialization of this program; define it on the compile line"
#endif
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

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_deepgemm_dense_mqa_ed3fb6920c34d71fb495(unsigned int* __restrict__ Starts, unsigned int* __restrict__ Ends, unsigned int* __restrict__ Metadata, unsigned int num_q_tokens, unsigned int num_kv_tokens)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(128) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    unsigned int* work_prefix = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int work_prefix_addr = smem + 0;
    unsigned int* cost_prefix = reinterpret_cast<unsigned int*>(smem_raw + 16);
    const int cost_prefix_addr = smem + 16;
    unsigned long long* warp_sums = reinterpret_cast<unsigned long long*>(smem_raw + 32);
    const int warp_sums_addr = smem + 32;
    unsigned long long* carry = reinterpret_cast<unsigned long long*>(smem_raw + 96);
    const int carry_addr = smem + 96;

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // === Task calls (dependency order) ===
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    unsigned int num_q_blocks = (num_q_tokens + 3) / 4;
    unsigned int span_offset = (3 * SM_COUNT + 1) / 2 * 2;
    #pragma unroll 1
    for (unsigned int qidx = tid; qidx < 4; qidx += 256) {
        if (num_q_blocks > qidx) {
            unsigned int start = 4294967295;
            unsigned int end = 0;
            #pragma unroll
            for (int qi = 0; qi < 4; qi++) {
                unsigned int _min_0 = ((qidx * 4 + (unsigned int)qi) < (num_q_tokens - 1) ? (qidx * 4 + (unsigned int)qi) : (num_q_tokens - 1));
                unsigned int row = _min_0;
                unsigned int _min_1 = ((Starts[row]) < (num_kv_tokens) ? (Starts[row]) : (num_kv_tokens));
                unsigned int _min_2 = ((start) < (_min_1) ? (start) : (_min_1));
                start = _min_2;
                unsigned int _min_3 = ((Ends[row]) < (num_kv_tokens) ? (Ends[row]) : (num_kv_tokens));
                unsigned int _max_0 = ((end) > (_min_3) ? (end) : (_min_3));
                end = _max_0;
            }
            unsigned int base = start / 4 * 4;
            unsigned int splits = (end - base + 255) / 256;
            Metadata[span_offset + qidx * 2] = base;
            Metadata[span_offset + qidx * 2 + 1] = splits;
            work_prefix[qidx] = splits;
            cost_prefix[qidx] = splits + (unsigned int)(((splits == 0) ? 0 : 1));
        }
    }
    __syncthreads();
    unsigned long long value = 0;
    if (num_q_blocks > (unsigned int)tid) {
        value = ((unsigned long long)work_prefix[tid] << 32) + (unsigned long long)cost_prefix[tid];
    }
    unsigned long long scanned = value;
    unsigned long long _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, scanned, 1, 32);
    unsigned long long synced = _shfl_up_0;
    if (lane >= 1) {
        scanned += synced;
    }
    unsigned long long _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, scanned, 2, 32);
    unsigned long long synced_0 = _shfl_up_1;
    if (lane >= 2) {
        scanned += synced_0;
    }
    unsigned long long _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, scanned, 4, 32);
    unsigned long long synced_1 = _shfl_up_2;
    if (lane >= 4) {
        scanned += synced_1;
    }
    unsigned long long _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, scanned, 8, 32);
    unsigned long long synced_2 = _shfl_up_3;
    if (lane >= 8) {
        scanned += synced_2;
    }
    unsigned long long _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, scanned, 16, 32);
    unsigned long long synced_3 = _shfl_up_4;
    if (lane >= 16) {
        scanned += synced_3;
    }
    if (num_q_blocks > (unsigned int)tid) {
        work_prefix[tid] = (unsigned int)(scanned >> 32);
        cost_prefix[tid] = (unsigned int)scanned;
    }
    __syncthreads();
    unsigned int total_work = work_prefix[num_q_blocks - 1];
    unsigned int total_cost = cost_prefix[num_q_blocks - 1];
    if (tid < SM_COUNT) {
        unsigned int base_1 = total_cost / (unsigned int)SM_COUNT;
        unsigned int remainder = total_cost % (unsigned int)SM_COUNT;
        int _min_4 = ((tid) < (remainder) ? (tid) : (remainder));
        unsigned int target = (unsigned int)tid * base_1 + (unsigned int)_min_4;
        unsigned int block = num_q_blocks;
        unsigned int split = 0;
        unsigned int coordinate = total_work;
        if (target != total_cost) {
            block = 0;
            unsigned int candidate = block + 4;
            if (candidate <= num_q_blocks) {
                if (target >= cost_prefix[candidate - 1]) {
                    block = candidate;
                }
            }
            unsigned int candidate_0 = block + 2;
            if (candidate_0 <= num_q_blocks) {
                if (target >= cost_prefix[candidate_0 - 1]) {
                    block = candidate_0;
                }
            }
            unsigned int candidate_1 = block + 1;
            if (candidate_1 <= num_q_blocks) {
                if (target >= cost_prefix[candidate_1 - 1]) {
                    block = candidate_1;
                }
            }
            unsigned int cost_before = 0;
            unsigned int work_before = 0;
            if (block > 0) {
                cost_before = cost_prefix[block - 1];
                work_before = work_prefix[block - 1];
            }
            unsigned int _max_1 = ((target - cost_before) > (1) ? (target - cost_before) : (1));
            unsigned int _min_5 = ((_max_1 - 1) < (work_prefix[block] - work_before - 1) ? (_max_1 - 1) : (work_prefix[block] - work_before - 1));
            split = _min_5;
            coordinate = work_before + split;
        }
        int _min_6 = ((tid + 1) < (remainder) ? (tid + 1) : (remainder));
        unsigned int target_0 = (unsigned int)(tid + 1) * base_1 + (unsigned int)_min_6;
        unsigned int block_1 = num_q_blocks;
        unsigned int split_2 = 0;
        unsigned int coordinate_3 = total_work;
        if (target_0 != total_cost) {
            block_1 = 0;
            unsigned int candidate_2 = block_1 + 4;
            if (candidate_2 <= num_q_blocks) {
                if (target_0 >= cost_prefix[candidate_2 - 1]) {
                    block_1 = candidate_2;
                }
            }
            unsigned int candidate_0_1 = block_1 + 2;
            if (candidate_0_1 <= num_q_blocks) {
                if (target_0 >= cost_prefix[candidate_0_1 - 1]) {
                    block_1 = candidate_0_1;
                }
            }
            unsigned int candidate_1_1 = block_1 + 1;
            if (candidate_1_1 <= num_q_blocks) {
                if (target_0 >= cost_prefix[candidate_1_1 - 1]) {
                    block_1 = candidate_1_1;
                }
            }
            unsigned int cost_before_1 = 0;
            unsigned int work_before_1 = 0;
            if (block_1 > 0) {
                cost_before_1 = cost_prefix[block_1 - 1];
                work_before_1 = work_prefix[block_1 - 1];
            }
            unsigned int _max_2 = ((target_0 - cost_before_1) > (1) ? (target_0 - cost_before_1) : (1));
            unsigned int _min_7 = ((_max_2 - 1) < (work_prefix[block_1] - work_before_1 - 1) ? (_max_2 - 1) : (work_prefix[block_1] - work_before_1 - 1));
            split_2 = _min_7;
            coordinate_3 = work_before_1 + split_2;
        }
        Metadata[tid * 2] = block;
        Metadata[tid * 2 + 1] = split;
        Metadata[2 * SM_COUNT + tid] = coordinate_3 - coordinate;
    }
}

} // extern "C"
