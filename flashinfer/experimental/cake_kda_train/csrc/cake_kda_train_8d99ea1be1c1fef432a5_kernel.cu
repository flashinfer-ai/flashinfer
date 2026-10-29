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
#define SMEM_STARTS_OFF 64
#define SMEM_STARTS_STAGE_BYTES 1040
#define SMEM_STARTS_STRIDE 1040
#define SMEM_TOTAL 1152
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_kda_train_8d99ea1be1c1fef432a5(int* __restrict__ cu_seqlens, int num_seqs, int num_chunks_max, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, int* __restrict__ seq_chunk_start, int* __restrict__ seq_order)
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
    int* starts = reinterpret_cast<int*>(smem_raw + 64);
    const int starts_addr = smem + 64;

    // === Task calls (declaration order) ===
    int tid_0 = tid;
    int lane_1 = lane;
    int warp_2 = warp;
    int cta = blockIdx.x;
    int n_i = 0;
    if (tid_0 < num_seqs) {
        n_i = (cu_seqlens[tid_0 + 1] - cu_seqlens[tid_0] + 63) / 64;
    }
    int incl = n_i;
    {
        uint32_t incl_scan = (uint32_t)incl;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_scan, 1, 32); if (lane >= 1) { incl_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_scan, 2, 32); if (lane >= 2) { incl_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_scan, 4, 32); if (lane >= 4) { incl_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_scan, 8, 32); if (lane >= 8) { incl_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_scan, 16, 32); if (lane >= 16) { incl_scan += peer; } }
        incl = (int)incl_scan;
    }
    if (lane_1 == 31) {
        warp_tot[warp_2] = incl;
    }
    __syncthreads();
    if (warp_2 == 0) {
        int wt = 0;
        if (lane_1 < 8) {
            wt = warp_tot[lane_1];
        }
        int wincl = wt;
        int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, wincl, 1, 32);
        if (lane_1 >= 1) {
            wincl = wincl + _shfl_up_0;
        }
        int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, wincl, 2, 32);
        if (lane_1 >= 2) {
            wincl = wincl + _shfl_up_1;
        }
        int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, wincl, 4, 32);
        if (lane_1 >= 4) {
            wincl = wincl + _shfl_up_2;
        }
        if (lane_1 < 8) {
            warp_tot[lane_1] = wincl - wt;
        }
        if (lane_1 == 7) {
            starts[num_seqs] = wincl;
        }
    }
    __syncthreads();
    if (tid_0 < num_seqs) {
        starts[tid_0] = warp_tot[warp_2] + incl - n_i;
    }
    __syncthreads();
    if (cta == 0) {
        for (int i = tid_0; i < num_seqs + 1; i += 256) {
            seq_chunk_start[i] = starts[i];
        }
    }
    int total = starts[num_seqs];
    int c = cta * 256 + tid_0;
    if (c < num_chunks_max) {
        int bos = 0;
        int ln = 0;
        if (c < total) {
            int lo = 0;
            int hi = num_seqs - 1;
            #pragma unroll
            for (int _step = 0; _step < 8; _step++) {
                int mid = (lo + hi + 1) / 2;
                if (c >= starts[mid]) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            int s0 = cu_seqlens[lo];
            int length = cu_seqlens[lo + 1] - s0;
            int i_c = c - starts[lo];
            bos = s0 + i_c * 64;
            int _min_0 = ((length - i_c * 64) < (64) ? (length - i_c * 64) : (64));
            ln = _min_0;
        }
        chunk_bos[c] = bos;
        chunk_len[c] = ln;
    }
    if (cta == 0) {
        if (tid_0 < num_seqs) {
            int n_me = starts[tid_0 + 1] - starts[tid_0];
            int rank = 0;
            for (int j = 0; j < num_seqs; j++) {
                int n_o = starts[j + 1] - starts[j];
                if (n_o > n_me) {
                    rank = rank + 1;
                }
                if (n_o == n_me) {
                    if (tid_0 > j) {
                        rank = rank + 1;
                    }
                }
            }
            seq_order[rank] = tid_0;
        }
    }
}

} // extern "C"
