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
#define SMEM_BLK_OFF 32
#define SMEM_BLK_STAGE_BYTES 4
#define SMEM_BLK_STRIDE 4
#define SMEM_STARTS_OFF 64
#define SMEM_STARTS_STAGE_BYTES 1040
#define SMEM_STARTS_STRIDE 1040
#define SMEM_ORDER_T_OFF 1104
#define SMEM_ORDER_T_STAGE_BYTES 1040
#define SMEM_ORDER_T_STRIDE 1040
#define SMEM_NSEG_T_OFF 2144
#define SMEM_NSEG_T_STAGE_BYTES 1040
#define SMEM_NSEG_T_STRIDE 1040
#define SMEM_REM_T_OFF 3184
#define SMEM_REM_T_STAGE_BYTES 1040
#define SMEM_REM_T_STRIDE 1040
#define SMEM_CS_T_OFF 4224
#define SMEM_CS_T_STAGE_BYTES 1040
#define SMEM_CS_T_STRIDE 1040
#define SMEM_TOTAL 5376
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_kda_train_9f69ca35a631ae5e1b4c(int* __restrict__ cu_seqlens, int num_seqs, int num_chunks_max, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, int* __restrict__ seq_chunk_start, int* __restrict__ seq_order, int segments, int proportional, int* __restrict__ seg_table)
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
    int* blk = reinterpret_cast<int*>(smem_raw + 32);
    const int blk_addr = smem + 32;
    int* starts = reinterpret_cast<int*>(smem_raw + 64);
    const int starts_addr = smem + 64;
    int* order_t = reinterpret_cast<int*>(smem_raw + 1104);
    const int order_t_addr = smem + 1104;
    int* nseg_t = reinterpret_cast<int*>(smem_raw + 2144);
    const int nseg_t_addr = smem + 2144;
    int* rem_t = reinterpret_cast<int*>(smem_raw + 3184);
    const int rem_t_addr = smem + 3184;
    int* cs_t = reinterpret_cast<int*>(smem_raw + 4224);
    const int cs_t_addr = smem + 4224;

    // === Task calls (declaration order) ===
    int tid_0 = tid;
    int lane_1 = lane;
    int warp_2 = warp;
    int cta = blockIdx.x;
    int n_i = 0;
    if (tid_0 < num_seqs) {
        n_i = (cu_seqlens[tid_0 + 1] - cu_seqlens[tid_0] + 63) / 64;
    }
    __syncthreads();
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
            blk[0] = wincl;
        }
    }
    __syncthreads();
    int excl = warp_tot[warp_2] + incl - n_i;
    int tile_total = blk[0];
    if (tid_0 < num_seqs) {
        starts[tid_0] = excl;
    }
    if (tid_0 == 0) {
        starts[num_seqs] = tile_total;
    }
    __syncthreads();
    if (cta == 0) {
        for (int i = tid_0; i < num_seqs + 1; i += 256) {
            seq_chunk_start[i] = starts[i];
        }
    }
    int c = cta * 256 + tid_0;
    if (c < num_chunks_max) {
        int bos = 0;
        int ln = 0;
        if (c < tile_total) {
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
            order_t[rank] = tid_0;
        }
        if (segments > 0) {
            __syncthreads();
            int nseg = num_seqs * segments;
            int base = num_seqs + 1;
            int prop = 0;
            if (proportional != 0) {
                if (num_seqs <= 256) {
                    prop = 1;
                }
            }
            if (prop == 1) {
                int q = segments;
                int rem = 0;
                int base_r = 0;
                if (tid_0 < num_seqs) {
                    int d_a = order_t[tid_0];
                    int n_a = starts[d_a + 1] - starts[d_a];
                    if (tile_total > 0) {
                        long long prod = (long long)nseg * (long long)n_a;
                        long long q64 = prod / (long long)tile_total;
                        q = (int)q64;
                        rem = (int)(prod - q64 * (long long)tile_total);
                    }
                    base_r = q;
                    if (q < 1) {
                        base_r = 1;
                    }
                }
                nseg_t[tid_0] = base_r;
                rem_t[tid_0] = rem;
                __syncthreads();
                int incl_0 = base_r;
                {
                    uint32_t incl_0_scan = (uint32_t)incl_0;
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_0_scan, 1, 32); if (lane >= 1) { incl_0_scan += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_0_scan, 2, 32); if (lane >= 2) { incl_0_scan += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_0_scan, 4, 32); if (lane >= 4) { incl_0_scan += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_0_scan, 8, 32); if (lane >= 8) { incl_0_scan += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_0_scan, 16, 32); if (lane >= 16) { incl_0_scan += peer; } }
                    incl_0 = (int)incl_0_scan;
                }
                if (lane_1 == 31) {
                    warp_tot[warp_2] = incl_0;
                }
                __syncthreads();
                if (warp_2 == 0) {
                    int wt_1 = 0;
                    if (lane_1 < 8) {
                        wt_1 = warp_tot[lane_1];
                    }
                    int wincl_1 = wt_1;
                    int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, wincl_1, 1, 32);
                    if (lane_1 >= 1) {
                        wincl_1 = wincl_1 + _shfl_up_3;
                    }
                    int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, wincl_1, 2, 32);
                    if (lane_1 >= 2) {
                        wincl_1 = wincl_1 + _shfl_up_4;
                    }
                    int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, wincl_1, 4, 32);
                    if (lane_1 >= 4) {
                        wincl_1 = wincl_1 + _shfl_up_5;
                    }
                    if (lane_1 < 8) {
                        warp_tot[lane_1] = wincl_1 - wt_1;
                    }
                    if (lane_1 == 7) {
                        blk[0] = wincl_1;
                    }
                }
                __syncthreads();
                int excl_1 = warp_tot[warp_2] + incl_0 - base_r;
                int tile_total_2 = blk[0];
                int dfix = nseg - tile_total_2;
                if (dfix > 0) {
                    if (tid_0 < num_seqs) {
                        int rank_r = 0;
                        for (int j_r = 0; j_r < num_seqs; j_r++) {
                            int r_o = rem_t[j_r];
                            if (r_o > rem) {
                                rank_r = rank_r + 1;
                            }
                            if (r_o == rem) {
                                if (tid_0 > j_r) {
                                    rank_r = rank_r + 1;
                                }
                            }
                        }
                        if (rank_r < dfix) {
                            nseg_t[tid_0] = base_r + 1;
                        }
                    }
                }
                if (dfix < 0) {
                    if (tid_0 == 0) {
                        int ndec = -dfix;
                        for (int _it = 0; _it < ndec; _it++) {
                            int best = 0;
                            int bestv = 1;
                            for (int j_d = 0; j_d < num_seqs; j_d++) {
                                int v = nseg_t[j_d];
                                if (v > bestv) {
                                    bestv = v;
                                    best = j_d;
                                }
                            }
                            nseg_t[best] = bestv - 1;
                        }
                    }
                }
                __syncthreads();
                int my = nseg_t[tid_0];
                __syncthreads();
                int incl_3 = my;
                {
                    uint32_t incl_3_scan = (uint32_t)incl_3;
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_3_scan, 1, 32); if (lane >= 1) { incl_3_scan += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_3_scan, 2, 32); if (lane >= 2) { incl_3_scan += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_3_scan, 4, 32); if (lane >= 4) { incl_3_scan += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_3_scan, 8, 32); if (lane >= 8) { incl_3_scan += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_3_scan, 16, 32); if (lane >= 16) { incl_3_scan += peer; } }
                    incl_3 = (int)incl_3_scan;
                }
                if (lane_1 == 31) {
                    warp_tot[warp_2] = incl_3;
                }
                __syncthreads();
                if (warp_2 == 0) {
                    int wt_2 = 0;
                    if (lane_1 < 8) {
                        wt_2 = warp_tot[lane_1];
                    }
                    int wincl_2 = wt_2;
                    int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, wincl_2, 1, 32);
                    if (lane_1 >= 1) {
                        wincl_2 = wincl_2 + _shfl_up_6;
                    }
                    int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, wincl_2, 2, 32);
                    if (lane_1 >= 2) {
                        wincl_2 = wincl_2 + _shfl_up_7;
                    }
                    int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, wincl_2, 4, 32);
                    if (lane_1 >= 4) {
                        wincl_2 = wincl_2 + _shfl_up_8;
                    }
                    if (lane_1 < 8) {
                        warp_tot[lane_1] = wincl_2 - wt_2;
                    }
                    if (lane_1 == 7) {
                        blk[0] = wincl_2;
                    }
                }
                __syncthreads();
                int excl_4 = warp_tot[warp_2] + incl_3 - my;
                int tile_total_5 = blk[0];
                if (tid_0 < num_seqs) {
                    cs_t[tid_0] = excl_4;
                    seg_table[tid_0] = excl_4;
                }
                if (tid_0 == 0) {
                    cs_t[num_seqs] = tile_total_5;
                    seg_table[num_seqs] = tile_total_5;
                }
                __syncthreads();
                for (int i_s = tid_0; i_s < nseg; i_s += 256) {
                    int lo_r = 0;
                    int hi_r = num_seqs - 1;
                    #pragma unroll
                    for (int _step_r = 0; _step_r < 8; _step_r++) {
                        int mid_r = (lo_r + hi_r + 1) / 2;
                        if (cs_t[mid_r] <= i_s) {
                            lo_r = mid_r;
                        } else {
                            hi_r = mid_r - 1;
                        }
                    }
                    int p = i_s - cs_t[lo_r];
                    int k = nseg_t[lo_r];
                    int d = order_t[lo_r];
                    int c0 = starts[d];
                    int n_d = starts[d + 1] - c0;
                    long long num0 = (long long)p * (long long)n_d;
                    int off0 = (int)(num0 / (long long)k);
                    seg_table[base + i_s + lo_r] = c0 + off0;
                    if (p == k - 1) {
                        seg_table[base + i_s + lo_r + 1] = c0 + n_d;
                    }
                }
            } else {
                for (int r = tid_0; r < num_seqs + 1; r += 256) {
                    seg_table[r] = r * segments;
                }
                for (int i_u = tid_0; i_u < nseg; i_u += 256) {
                    int ru = i_u / segments;
                    int pu = i_u - ru * segments;
                    int du = order_t[ru];
                    int c0u = starts[du];
                    int n_u = starts[du + 1] - c0u;
                    seg_table[base + i_u + ru] = c0u + pu * n_u / segments;
                    if (pu == segments - 1) {
                        seg_table[base + i_u + ru + 1] = c0u + n_u;
                    }
                }
            }
        }
    }
}

} // extern "C"
