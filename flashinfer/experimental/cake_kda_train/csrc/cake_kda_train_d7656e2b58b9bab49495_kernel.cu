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
#define NUM_MAIN_STAGES 1
#define SMEM_WARP_TOT_OFF 0
#define SMEM_WARP_TOT_STAGE_BYTES 32
#define SMEM_WARP_TOT_STRIDE 32
#define SMEM_CNT_TILE_OFF 64
#define SMEM_CNT_TILE_STAGE_BYTES 1024
#define SMEM_CNT_TILE_STRIDE 1024
#define SMEM_TOTAL 1152
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_kda_train_d7656e2b58b9bab49495(int* __restrict__ cu_seqlens, int num_seqs, int num_chunks_max, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, int* __restrict__ seq_chunk_start, int* __restrict__ seq_order)
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
    int* warp_tot = reinterpret_cast<int*>(smem_raw + 0);
    const int warp_tot_addr = smem + 0;
    int* cnt_tile = reinterpret_cast<int*>(smem_raw + 64);
    const int cnt_tile_addr = smem + 64;

    // === Task calls (declaration order) ===
    int tid_0 = tid;
    int warp_1 = warp;
    int carry = 0;
    for (int base = 0; base < num_seqs; base += 256) {
        int i = base + tid_0;
        int n_i = 0;
        if (i < num_seqs) {
            n_i = (cu_seqlens[i + 1] - cu_seqlens[i] + 63) / 64;
        }
        // CtaScanSum: threads=256, warps=8
        int _cta_scan_sum_prefix_0;
        int _cta_scan_sum_total_0;
        {
            const uint32_t _cta_scan_sum_prefix_0_own = (uint32_t)(n_i);
            uint32_t _cta_scan_sum_prefix_0_ws = _cta_scan_sum_prefix_0_own;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_ws, 1, 32); if (lane >= 1) { _cta_scan_sum_prefix_0_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_ws, 2, 32); if (lane >= 2) { _cta_scan_sum_prefix_0_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_ws, 4, 32); if (lane >= 4) { _cta_scan_sum_prefix_0_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_ws, 8, 32); if (lane >= 8) { _cta_scan_sum_prefix_0_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_ws, 16, 32); if (lane >= 16) { _cta_scan_sum_prefix_0_ws += peer; } }
            if (lane == 31) { warp_tot[warp] = (int)_cta_scan_sum_prefix_0_ws; }
            __syncthreads();
            const uint32_t _cta_scan_sum_prefix_0_wt = (lane < 8) ? (uint32_t)warp_tot[lane] : 0;
            uint32_t _cta_scan_sum_prefix_0_wsum = _cta_scan_sum_prefix_0_wt;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_wsum, 1, 32); if (lane >= 1) { _cta_scan_sum_prefix_0_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_wsum, 2, 32); if (lane >= 2) { _cta_scan_sum_prefix_0_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_wsum, 4, 32); if (lane >= 4) { _cta_scan_sum_prefix_0_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_wsum, 8, 32); if (lane >= 8) { _cta_scan_sum_prefix_0_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_wsum, 16, 32); if (lane >= 16) { _cta_scan_sum_prefix_0_wsum += peer; } }
            _cta_scan_sum_total_0 = (int)__shfl_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_wsum, 7);
            const uint32_t _cta_scan_sum_prefix_0_before = __shfl_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_wsum - _cta_scan_sum_prefix_0_wt, warp);
            _cta_scan_sum_prefix_0 = (int)(_cta_scan_sum_prefix_0_ws - _cta_scan_sum_prefix_0_own + _cta_scan_sum_prefix_0_before);
        }
        if (i < num_seqs) {
            seq_chunk_start[i] = carry + _cta_scan_sum_prefix_0;
        }
        carry = carry + _cta_scan_sum_total_0;
        __syncthreads();
    }
    if (tid_0 == 0) {
        seq_chunk_start[num_seqs] = carry;
    }
    __syncthreads();
    int total = carry;
    for (int c = tid_0; c < num_chunks_max; c += 256) {
        int bos = 0;
        int ln = 0;
        if (total > c) {
            int lo = 0;
            int hi = num_seqs - 1;
            #pragma unroll
            for (int _step = 0; _step < 16; _step++) {
                int mid = (lo + hi + 1) / 2;
                if (seq_chunk_start[mid] <= c) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            int s0 = cu_seqlens[lo];
            int length = cu_seqlens[lo + 1] - s0;
            int i_c = c - seq_chunk_start[lo];
            bos = s0 + i_c * 64;
            int _min_0 = ((length - i_c * 64) < (64) ? (length - i_c * 64) : (64));
            ln = _min_0;
        }
        chunk_bos[c] = bos;
        chunk_len[c] = ln;
    }
    for (int ibase = 0; ibase < num_seqs; ibase += 256) {
        int i_s = ibase + tid_0;
        int n_me = 0;
        if (i_s < num_seqs) {
            n_me = seq_chunk_start[i_s + 1] - seq_chunk_start[i_s];
        }
        int rank = 0;
        for (int jbase = 0; jbase < num_seqs; jbase += 256) {
            int j_s = jbase + tid_0;
            int n_j = 0;
            if (j_s < num_seqs) {
                n_j = seq_chunk_start[j_s + 1] - seq_chunk_start[j_s];
            }
            cnt_tile[tid_0] = n_j;
            __syncthreads();
            if (i_s < num_seqs) {
                int jn = ((num_seqs - jbase) < (256) ? (num_seqs - jbase) : (256));
                for (int jj = 0; jj < jn; jj++) {
                    int n_o = cnt_tile[jj];
                    if (n_o > n_me) {
                        rank = rank + 1;
                    }
                    if (n_o == n_me) {
                        if (i_s > jbase + jj) {
                            rank = rank + 1;
                        }
                    }
                }
            }
            __syncthreads();
        }
        if (i_s < num_seqs) {
            seq_order[rank] = i_s;
        }
    }
}

} // extern "C"
