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
#define SMEM_WARP_SUMS_STAGE_BYTES 32
#define SMEM_WARP_SUMS_STRIDE 32
#define SMEM_TOTAL 16512
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_deepgemm_sm120_paged_mqa_logits_f91d5cd744783557f500(int* __restrict__ context_lens, int* __restrict__ schedule_meta, int batch_size, int next_n, int num_next_n_atoms, int split_kv, int num_sms)
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
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int tid_1 = tid;
    int row_base = tid_1 * 16;
    int ctx[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int qb = row_base + i;
        int qr = ((qb < batch_size) ? qb : batch_size - 1);
        ctx[i] = context_lens[qr * next_n + next_n - 1];
    }
    unsigned int divisor = (unsigned int)split_kv;
    unsigned int inv = 4294967295u / divisor;
    int local_incl[16];
    int running = 0;
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; i_1++) {
        unsigned int num = (unsigned int)(ctx[i_1] + split_kv - 1);
        unsigned int q0 = (unsigned int)((unsigned long long)num * (unsigned long long)inv >> 32);
        int quo = (int)(((num - q0 * divisor >= divisor) ? q0 + 1 : q0));
        int segs = ((row_base + i_1 < batch_size) ? quo : 0);
        running += segs;
        local_incl[i_1] = running;
    }
    // CtaScanSum: threads=256, warps=8
    int before;
    int total_segs;
    {
        const uint32_t before_own = (uint32_t)(running);
        uint32_t before_ws = before_own;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_ws, 1, 32); if (lane >= 1) { before_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_ws, 2, 32); if (lane >= 2) { before_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_ws, 4, 32); if (lane >= 4) { before_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_ws, 8, 32); if (lane >= 8) { before_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_ws, 16, 32); if (lane >= 16) { before_ws += peer; } }
        if (lane == 31) { warp_sums[warp] = (int)before_ws; }
        __syncthreads();
        const uint32_t before_wt = (lane < 8) ? (uint32_t)warp_sums[lane] : 0;
        uint32_t before_wsum = before_wt;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_wsum, 1, 32); if (lane >= 1) { before_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_wsum, 2, 32); if (lane >= 2) { before_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_wsum, 4, 32); if (lane >= 4) { before_wsum += peer; } }
        total_segs = (int)__shfl_sync(0xFFFFFFFF, before_wsum, 7);
        const uint32_t before_before = __shfl_sync(0xFFFFFFFF, before_wsum - before_wt, warp);
        before = (int)(before_ws - before_own + before_before);
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; i_2++) {
        int q_out = row_base + i_2;
        if (q_out < batch_size) {
            prefix[q_out] = before + local_incl[i_2];
        }
    }
    __syncthreads();
    int total = total_segs * num_next_n_atoms;
    int sentinel = batch_size * num_next_n_atoms;
    if (total == 0) {
        #pragma unroll 1
        for (int sm0 = tid_1; sm0 < num_sms + 1; sm0 += 256) {
            *(reinterpret_cast<int*>(schedule_meta + (sm0 * 2)) + (0)) = sentinel;
            *(reinterpret_cast<int*>(schedule_meta + (sm0 * 2 + 1)) + (0)) = 0;
        }
    } else {
        int qd = total / num_sms;
        int rd = total % num_sms;
        int pivot = num_sms - rd;
        int _count_leading_zeros_0 = __clz(batch_size);
        int steps = 32 - _count_leading_zeros_0;
        #pragma unroll 1
        for (int sm = tid_1; sm < num_sms; sm += 256) {
            int extra = ((pivot < sm) ? sm - pivot : 0);
            int seg_starts = sm * qd + extra;
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
            int kv_split_idx = ((num_segs_q > 0) ? offset_in_q % num_segs_q : 0);
            int q_atom_idx = q_idx * num_next_n_atoms + atom_idx;
            *(reinterpret_cast<int*>(schedule_meta + (sm * 2)) + (0)) = q_atom_idx;
            *(reinterpret_cast<int*>(schedule_meta + (sm * 2 + 1)) + (0)) = kv_split_idx;
        }
        if (tid_1 == 0) {
            *(reinterpret_cast<int*>(schedule_meta + (num_sms * 2)) + (0)) = sentinel;
            *(reinterpret_cast<int*>(schedule_meta + (num_sms * 2 + 1)) + (0)) = 0;
        }
    }
}

} // extern "C"
