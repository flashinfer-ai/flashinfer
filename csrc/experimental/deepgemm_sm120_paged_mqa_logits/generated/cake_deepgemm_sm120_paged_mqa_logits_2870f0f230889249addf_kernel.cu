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
#define SMEM_PREFIX_OFF 0
#define SMEM_PREFIX_STAGE_BYTES 16384
#define SMEM_PREFIX_STRIDE 16384
#define SMEM_WARP_SUMS_OFF 16384
#define SMEM_WARP_SUMS_STAGE_BYTES 128
#define SMEM_WARP_SUMS_STRIDE 128
#define SMEM_TOTAL 16512
#define THREADS 1024

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_cake_deepgemm_sm120_paged_mqa_logits_2870f0f230889249addf(int* __restrict__ context_lens, int* __restrict__ schedule_meta, int batch_size, int next_n, int num_next_n_atoms, int split_kv, int num_sms, int dyn_slots, int dyn_pct, int dyn_seg)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int* prefix = reinterpret_cast<int*>(smem_raw + 0);
    const int prefix_addr = smem + 0;
    int* warp_sums = reinterpret_cast<int*>(smem_raw + 16384);
    const int warp_sums_addr = smem + 16384;

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // === Task calls (declaration order) ===
    int tid_1 = tid;
    int carry = 0;
    #pragma unroll 1
    for (int base = 0; base < batch_size; base += 1024) {
        int qb = base + tid_1;
        int qr = ((qb < batch_size) ? qb : batch_size - 1);
        int ctx = context_lens[qr * next_n + next_n - 1];
        int segs = ((qb < batch_size) ? (ctx + split_kv - 1) / split_kv : 0);
        // CtaScanSum: threads=1024, warps=32
        int incl;
        int chunk_total;
        {
            const uint32_t incl_own = (uint32_t)(segs);
            uint32_t incl_ws = incl_own;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_ws, 1, 32); if (lane >= 1) { incl_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_ws, 2, 32); if (lane >= 2) { incl_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_ws, 4, 32); if (lane >= 4) { incl_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_ws, 8, 32); if (lane >= 8) { incl_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_ws, 16, 32); if (lane >= 16) { incl_ws += peer; } }
            if (lane == 31) { warp_sums[warp] = (int)incl_ws; }
            __syncthreads();
            const uint32_t incl_wt = (uint32_t)warp_sums[lane];
            uint32_t incl_wsum = incl_wt;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_wsum, 1, 32); if (lane >= 1) { incl_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_wsum, 2, 32); if (lane >= 2) { incl_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_wsum, 4, 32); if (lane >= 4) { incl_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_wsum, 8, 32); if (lane >= 8) { incl_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_wsum, 16, 32); if (lane >= 16) { incl_wsum += peer; } }
            chunk_total = (int)__shfl_sync(0xFFFFFFFF, incl_wsum, 31);
            const uint32_t incl_before = __shfl_sync(0xFFFFFFFF, incl_wsum - incl_wt, warp);
            incl = (int)(incl_ws + incl_before);
        }
        if (qb < batch_size) {
            prefix[qb] = carry + incl;
        }
        carry += chunk_total;
        __syncthreads();
    }
    int total = carry * num_next_n_atoms;
    int sentinel = batch_size * num_next_n_atoms;
    int slots = dyn_slots * num_sms;
    int walk_rows = num_sms + slots;
    long long per_item = (long long)dyn_seg * 100;
    long long want = ((long long)total * (long long)dyn_pct + per_item - 1) / per_item;
    long long cap = ((long long)total + (long long)dyn_seg - 1) / (long long)dyn_seg;
    long long _min_2 = (((long long)slots) < (want) ? ((long long)slots) : (want));
    long long _min_1 = ((_min_2) < (cap) ? (_min_2) : (cap));
    long long _max_0 = ((0) > (_min_1) ? (0) : (_min_1));
    int n = (int)_max_0;
    long long _max_3 = ((0) > ((long long)total - (long long)n * (long long)dyn_seg) ? (0) : ((long long)total - (long long)n * (long long)dyn_seg));
    int static_segs = (int)_max_3;
    int qd = static_segs / num_sms;
    int rd = static_segs % num_sms;
    int pivot = num_sms - rd;
    int _count_leading_zeros_4 = __clz(batch_size);
    int steps = 32 - _count_leading_zeros_4;
    #pragma unroll 1
    for (int row = tid_1; row < walk_rows; row += 1024) {
        int extra = ((pivot < row) ? row - pivot : 0);
        int slot = row - num_sms;
        int _min_5 = ((total) < (static_segs + slot * dyn_seg) ? (total) : (static_segs + slot * dyn_seg));
        int pool_start = ((slot < n) ? _min_5 : total);
        int seg_starts = ((row < num_sms) ? row * qd + extra : pool_start);
        int q_atom_idx = sentinel;
        int kv_split_idx = 0;
        if (seg_starts < total) {
            int lo = 0;
            int hi = batch_size;
            #pragma unroll 1
            for (int _ = 0; _ < steps; _++) {
                int mid = (lo + hi) / 2;
                int midr = ((mid < batch_size) ? mid : batch_size - 1);
                int le = ((seg_starts >= prefix[midr] * num_next_n_atoms) ? 1 : 0);
                int inb = ((mid < batch_size) ? 1 : 0);
                int go = le * inb;
                lo = ((go != 0) ? mid + 1 : lo);
                hi = ((go == 0) ? mid : hi);
            }
            int q_idx = ((lo < batch_size) ? lo : batch_size - 1);
            int prev_cum = ((q_idx > 0) ? prefix[q_idx - 1] : 0);
            int cur_cum = prefix[q_idx];
            int offset_in_q = seg_starts - prev_cum * num_next_n_atoms;
            int num_segs_q = cur_cum - prev_cum;
            int atom_idx = ((num_segs_q > 0) ? offset_in_q / num_segs_q : 0);
            q_atom_idx = q_idx * num_next_n_atoms + atom_idx;
            kv_split_idx = ((num_segs_q > 0) ? offset_in_q % num_segs_q : 0);
        }
        *(reinterpret_cast<int*>(schedule_meta + (row * 2)) + (0)) = q_atom_idx;
        *(reinterpret_cast<int*>(schedule_meta + (row * 2 + 1)) + (0)) = kv_split_idx;
    }
    if (tid_1 == 0) {
        *(reinterpret_cast<int*>(schedule_meta + (walk_rows * 2)) + (0)) = sentinel;
        *(reinterpret_cast<int*>(schedule_meta + (walk_rows * 2 + 1)) + (0)) = 0;
        *(reinterpret_cast<int*>(schedule_meta + (walk_rows * 2 + 2)) + (0)) = 0;
        *(reinterpret_cast<int*>(schedule_meta + (walk_rows * 2 + 3)) + (0)) = n;
    }
}

} // extern "C"
