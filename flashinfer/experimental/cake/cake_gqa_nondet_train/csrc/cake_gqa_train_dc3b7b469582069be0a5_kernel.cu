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

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_cake_gqa_train_dc3b7b469582069be0a5(int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int* __restrict__ tile_base, int* __restrict__ pair_base, int* __restrict__ pair_order, int num_segments, int key_blocks, int pair_bound)
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

    // Prologue
    int* warp_tot = reinterpret_cast<int*>(smem_raw + 0);
    const int warp_tot_addr = smem + 0;
    int* blk = reinterpret_cast<int*>(smem_raw + 128);
    const int blk_addr = smem + 128;
    int* seen_t = reinterpret_cast<int*>(smem_raw + 256);
    const int seen_t_addr = smem + 256;

    // === Task calls (declaration order) ===
    int tid_1 = tid;
    int lane_1 = lane;
    int warp_1 = warp;
    int entries = (key_blocks + 1) * num_segments;
    int carry = 0;
    #pragma unroll 1
    for (int start = 0; start < entries; start += 4096) {
        int i0 = start + tid_1 * 4;
        int c = 0;
        if (i0 < entries) {
            int n = i0 / num_segments;
            int b = i0 % num_segments;
            int q_len = cu_seqlens_q[b + 1] - cu_seqlens_q[b];
            int k_len = cu_seqlens_k[b + 1] - cu_seqlens_k[b];
            if (n * 128 < k_len) {
                int first = n * 128 - (k_len - q_len);
                int m_min = ((first > 0) ? first : 0) >> 6;
                c = (q_len + 63 >> 6) - m_min;
                if (c < 0) {
                    c = 0;
                }
            }
        }
        int c0 = c;
        int c_0 = 0;
        if (i0 + 1 < entries) {
            int n_1 = (i0 + 1) / num_segments;
            int b_1 = (i0 + 1) % num_segments;
            int q_len_1 = cu_seqlens_q[b_1 + 1] - cu_seqlens_q[b_1];
            int k_len_1 = cu_seqlens_k[b_1 + 1] - cu_seqlens_k[b_1];
            if (n_1 * 128 < k_len_1) {
                int first_1 = n_1 * 128 - (k_len_1 - q_len_1);
                int m_min_1 = ((first_1 > 0) ? first_1 : 0) >> 6;
                c_0 = (q_len_1 + 63 >> 6) - m_min_1;
                if (c_0 < 0) {
                    c_0 = 0;
                }
            }
        }
        int c1 = c_0;
        int c_1 = 0;
        if (i0 + 2 < entries) {
            int n_2 = (i0 + 2) / num_segments;
            int b_2 = (i0 + 2) % num_segments;
            int q_len_2 = cu_seqlens_q[b_2 + 1] - cu_seqlens_q[b_2];
            int k_len_2 = cu_seqlens_k[b_2 + 1] - cu_seqlens_k[b_2];
            if (n_2 * 128 < k_len_2) {
                int first_2 = n_2 * 128 - (k_len_2 - q_len_2);
                int m_min_2 = ((first_2 > 0) ? first_2 : 0) >> 6;
                c_1 = (q_len_2 + 63 >> 6) - m_min_2;
                if (c_1 < 0) {
                    c_1 = 0;
                }
            }
        }
        int c2 = c_1;
        int c_2 = 0;
        if (i0 + 3 < entries) {
            int n_3 = (i0 + 3) / num_segments;
            int b_3 = (i0 + 3) % num_segments;
            int q_len_3 = cu_seqlens_q[b_3 + 1] - cu_seqlens_q[b_3];
            int k_len_3 = cu_seqlens_k[b_3 + 1] - cu_seqlens_k[b_3];
            if (n_3 * 128 < k_len_3) {
                int first_3 = n_3 * 128 - (k_len_3 - q_len_3);
                int m_min_3 = ((first_3 > 0) ? first_3 : 0) >> 6;
                c_2 = (q_len_3 + 63 >> 6) - m_min_3;
                if (c_2 < 0) {
                    c_2 = 0;
                }
            }
        }
        int c3 = c_2;
        __syncthreads();
        int incl = c0 + c1 + c2 + c3;
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
            warp_tot[warp_1] = incl;
        }
        __syncthreads();
        if (warp_1 == 0) {
            int wt = warp_tot[lane_1];
            int wincl = wt;
            {
                uint32_t wincl_scan = (uint32_t)wincl;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, wincl_scan, 1, 32); if (lane >= 1) { wincl_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, wincl_scan, 2, 32); if (lane >= 2) { wincl_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, wincl_scan, 4, 32); if (lane >= 4) { wincl_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, wincl_scan, 8, 32); if (lane >= 8) { wincl_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, wincl_scan, 16, 32); if (lane >= 16) { wincl_scan += peer; } }
                wincl = (int)wincl_scan;
            }
            warp_tot[lane_1] = wincl - wt;
            if (lane_1 == 31) {
                blk[0] = wincl;
            }
        }
        __syncthreads();
        int excl = warp_tot[warp_1] + incl - (c0 + c1 + c2 + c3);
        int total = blk[0];
        int e = carry + excl;
        if (i0 < entries) {
            tile_base[i0] = e;
        }
        if (i0 + 1 < entries) {
            tile_base[i0 + 1] = e + c0;
        }
        if (i0 + 2 < entries) {
            tile_base[i0 + 2] = e + c0 + c1;
        }
        if (i0 + 3 < entries) {
            tile_base[i0 + 3] = e + c0 + c1 + c2;
        }
        carry = carry + total;
    }
    int pairs = 0;
    #pragma unroll 1
    for (int seg0 = 0; seg0 < num_segments; seg0 += 1024) {
        int sb = seg0 + tid_1;
        int np = 0;
        if (sb < num_segments) {
            np = cu_seqlens_q[sb + 1] - cu_seqlens_q[sb] + 127 >> 7;
        }
        __syncthreads();
        int incl_1 = np;
        {
            uint32_t incl_1_scan = (uint32_t)incl_1;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_1_scan, 1, 32); if (lane >= 1) { incl_1_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_1_scan, 2, 32); if (lane >= 2) { incl_1_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_1_scan, 4, 32); if (lane >= 4) { incl_1_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_1_scan, 8, 32); if (lane >= 8) { incl_1_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_1_scan, 16, 32); if (lane >= 16) { incl_1_scan += peer; } }
            incl_1 = (int)incl_1_scan;
        }
        if (lane_1 == 31) {
            warp_tot[warp_1] = incl_1;
        }
        __syncthreads();
        if (warp_1 == 0) {
            int wt_1 = warp_tot[lane_1];
            int wincl_1 = wt_1;
            {
                uint32_t wincl_1_scan = (uint32_t)wincl_1;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, wincl_1_scan, 1, 32); if (lane >= 1) { wincl_1_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, wincl_1_scan, 2, 32); if (lane >= 2) { wincl_1_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, wincl_1_scan, 4, 32); if (lane >= 4) { wincl_1_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, wincl_1_scan, 8, 32); if (lane >= 8) { wincl_1_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, wincl_1_scan, 16, 32); if (lane >= 16) { wincl_1_scan += peer; } }
                wincl_1 = (int)wincl_1_scan;
            }
            warp_tot[lane_1] = wincl_1 - wt_1;
            if (lane_1 == 31) {
                blk[0] = wincl_1;
            }
        }
        __syncthreads();
        int excl_1 = warp_tot[warp_1] + incl_1 - np;
        int total_1 = blk[0];
        if (sb < num_segments) {
            pair_base[sb] = pairs + excl_1;
        }
        pairs = pairs + total_1;
    }
    if (tid_1 == 0) {
        pair_base[num_segments] = pairs;
    }
    int live = ((pairs < pair_bound) ? pairs : pair_bound);
    __syncthreads();
    #pragma unroll 1
    for (int p = tid_1; p < pair_bound; p += 1024) {
        int s = -1;
        if (p < live) {
            int slo = 0;
            int shi = num_segments;
            #pragma unroll 1
            for (int _ = 0; _ < 31; _++) {
                if (shi - slo > 1) {
                    int smid = slo + shi >> 1;
                    if (pair_base[smid] <= p) {
                        slo = smid;
                    } else {
                        shi = smid;
                    }
                }
            }
            int sq = cu_seqlens_q[slo + 1] - cu_seqlens_q[slo];
            int sk = cu_seqlens_k[slo + 1] - cu_seqlens_k[slo];
            int last = 2 * (p - pair_base[slo]) + 2;
            if (last > sq + 63 >> 6) {
                last = sq + 63 >> 6;
            }
            s = 64 * last + (sk - sq) + 127 >> 7;
            if (s < 0) {
                s = 0;
            }
            if (s > sk + 127 >> 7) {
                s = sk + 127 >> 7;
            }
        }
        seen_t[p] = s;
    }
    __syncthreads();
    #pragma unroll 1
    for (int p2 = tid_1; p2 < pair_bound; p2 += 1024) {
        int rank = p2;
        if (p2 < live) {
            int mine = seen_t[p2];
            rank = 0;
            #pragma unroll 1
            for (int j = 0; j < live; j++) {
                int other = seen_t[j];
                if (other > mine) {
                    rank = rank + 1;
                }
                if (other == mine) {
                    if (j < p2) {
                        rank = rank + 1;
                    }
                }
            }
        }
        pair_order[rank] = p2;
    }
}

} // extern "C"
