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
#define SMEM_ITEM_SUMS_OFF 0
#define SMEM_ITEM_SUMS_STAGE_BYTES 128
#define SMEM_ITEM_SUMS_STRIDE 128
#define SMEM_COL_SUMS_OFF 128
#define SMEM_COL_SUMS_STAGE_BYTES 128
#define SMEM_COL_SUMS_STRIDE 128
#define SMEM_TOTAL 256
#define THREADS 1024

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_cake_msa_icp_indexer_eb804215a3d74733ee47(int* __restrict__ qo_lens, int* __restrict__ kv_lens, int* __restrict__ qo_offset, long long* __restrict__ work_range, long long* __restrict__ work_info, int batch, int num_ctas)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int* item_sums = reinterpret_cast<int*>(smem_raw + 0);
    const int item_sums_addr = smem + 0;
    int* col_sums = reinterpret_cast<int*>(smem_raw + 128);
    const int col_sums_addr = smem + 128;

    // === Task calls (declaration order) ===
    int tid_0 = tid;
    int per_thread = (batch + 1024 - 1) / 1024;
    int b_lo = ((tid_0 * per_thread) < (batch) ? (tid_0 * per_thread) : (batch));
    int b_hi = ((b_lo + per_thread) < (batch) ? (b_lo + per_thread) : (batch));
    int n_mine = 0;
    int c_mine = 0;
    #pragma unroll 1
    for (int b = b_lo; b < b_hi; b++) {
        int q = qo_lens[b];
        int kv = kv_lens[b];
        int qoff = qo_offset[b];
        int _max_0 = ((q) > (0) ? (q) : (0));
        int n_b = (_max_0 + 64 - 1) / 64;
        n_mine += n_b;
        #pragma unroll 1
        for (int i = 0; i < n_b; i++) {
            int _min_3 = (((i + 1) * 64) < (q) ? ((i + 1) * 64) : (q));
            int _min_2 = ((kv) < (qoff + _min_3) ? (kv) : (qoff + _min_3));
            int _max_1 = ((_min_2) > (0) ? (_min_2) : (0));
            c_mine += (_max_1 + 127) / 128;
        }
    }
    // CtaScanSum: threads=1024, warps=32
    int item_base;
    int n_items;
    {
        const uint32_t item_base_own = (uint32_t)(n_mine);
        uint32_t item_base_ws = item_base_own;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, item_base_ws, 1, 32); if (lane >= 1) { item_base_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, item_base_ws, 2, 32); if (lane >= 2) { item_base_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, item_base_ws, 4, 32); if (lane >= 4) { item_base_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, item_base_ws, 8, 32); if (lane >= 8) { item_base_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, item_base_ws, 16, 32); if (lane >= 16) { item_base_ws += peer; } }
        if (lane == 31) { item_sums[warp] = (int)item_base_ws; }
        __syncthreads();
        const uint32_t item_base_wt = (uint32_t)item_sums[lane];
        uint32_t item_base_wsum = item_base_wt;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, item_base_wsum, 1, 32); if (lane >= 1) { item_base_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, item_base_wsum, 2, 32); if (lane >= 2) { item_base_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, item_base_wsum, 4, 32); if (lane >= 4) { item_base_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, item_base_wsum, 8, 32); if (lane >= 8) { item_base_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, item_base_wsum, 16, 32); if (lane >= 16) { item_base_wsum += peer; } }
        n_items = (int)__shfl_sync(0xFFFFFFFF, item_base_wsum, 31);
        const uint32_t item_base_before = __shfl_sync(0xFFFFFFFF, item_base_wsum - item_base_wt, warp);
        item_base = (int)(item_base_ws - item_base_own + item_base_before);
    }
    // CtaScanSum: threads=1024, warps=32
    int col_base;
    int n_cols;
    {
        const uint32_t col_base_own = (uint32_t)(c_mine);
        uint32_t col_base_ws = col_base_own;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, col_base_ws, 1, 32); if (lane >= 1) { col_base_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, col_base_ws, 2, 32); if (lane >= 2) { col_base_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, col_base_ws, 4, 32); if (lane >= 4) { col_base_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, col_base_ws, 8, 32); if (lane >= 8) { col_base_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, col_base_ws, 16, 32); if (lane >= 16) { col_base_ws += peer; } }
        if (lane == 31) { col_sums[warp] = (int)col_base_ws; }
        __syncthreads();
        const uint32_t col_base_wt = (uint32_t)col_sums[lane];
        uint32_t col_base_wsum = col_base_wt;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, col_base_wsum, 1, 32); if (lane >= 1) { col_base_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, col_base_wsum, 2, 32); if (lane >= 2) { col_base_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, col_base_wsum, 4, 32); if (lane >= 4) { col_base_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, col_base_wsum, 8, 32); if (lane >= 8) { col_base_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, col_base_wsum, 16, 32); if (lane >= 16) { col_base_wsum += peer; } }
        n_cols = (int)__shfl_sync(0xFFFFFFFF, col_base_wsum, 31);
        const uint32_t col_base_before = __shfl_sync(0xFFFFFFFF, col_base_wsum - col_base_wt, warp);
        col_base = (int)(col_base_ws - col_base_own + col_base_before);
    }
    int j = item_base;
    int prefix = col_base;
    #pragma unroll 1
    for (int b_1 = b_lo; b_1 < b_hi; b_1++) {
        int q_1 = qo_lens[b_1];
        int kv_1 = kv_lens[b_1];
        int qoff_1 = qo_offset[b_1];
        int _max_4 = ((q_1) > (0) ? (q_1) : (0));
        int n_b_1 = (_max_4 + 64 - 1) / 64;
        #pragma unroll 1
        for (int i_1 = 0; i_1 < n_b_1; i_1++) {
            int _min_7 = (((i_1 + 1) * 64) < (q_1) ? ((i_1 + 1) * 64) : (q_1));
            int _min_6 = ((kv_1) < (qoff_1 + _min_7) ? (kv_1) : (qoff_1 + _min_7));
            int _max_5 = ((_min_6) > (0) ? (_min_6) : (0));
            prefix += (_max_5 + 127) / 128;
            work_info[j] = (long long)b_1 | (long long)i_1 << 16 | (long long)prefix << 32;
            j += 1;
        }
    }
    __syncthreads();
    long long total = (long long)n_cols + (long long)n_items * 4;
    #pragma unroll 1
    for (int s = tid_0; s < num_ctas; s += 1024) {
        int lo = 0;
        int hi = n_items;
        #pragma unroll
        for (int _ = 0; _ < 17; _++) {
            int mid = (lo + hi) / 2;
            if (lo < hi) {
                if ((work_info[mid] >> 32) + (long long)(mid + 1) * 4 > (long long)s * total / (long long)num_ctas) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
        }
        long long col = 0;
        if (lo < n_items) {
            long long begin = 0;
            if (lo > 0) {
                begin = (work_info[lo - 1] >> 32) + (long long)(lo - 1 + 1) * 4;
            }
            col = (long long)s * total / (long long)num_ctas - begin - 4;
            if (col < 0) {
                col = 0;
            }
            col = col - col % 8;
        }
        int lo_0 = 0;
        int hi_0 = n_items;
        #pragma unroll
        for (int __1 = 0; __1 < 17; __1++) {
            int mid_1 = (lo_0 + hi_0) / 2;
            if (lo_0 < hi_0) {
                if ((work_info[mid_1] >> 32) + (long long)(mid_1 + 1) * 4 > (long long)(s + 1) * total / (long long)num_ctas) {
                    hi_0 = mid_1;
                } else {
                    lo_0 = mid_1 + 1;
                }
            }
        }
        long long col_0 = 0;
        if (lo_0 < n_items) {
            long long begin_1 = 0;
            if (lo_0 > 0) {
                begin_1 = (work_info[lo_0 - 1] >> 32) + (long long)(lo_0 - 1 + 1) * 4;
            }
            col_0 = (long long)(s + 1) * total / (long long)num_ctas - begin_1 - 4;
            if (col_0 < 0) {
                col_0 = 0;
            }
            col_0 = col_0 - col_0 % 8;
        }
        work_range[s] = (long long)lo | (long long)lo_0 << 16 | col << 32 | col_0 << 48;
    }
}

} // extern "C"
