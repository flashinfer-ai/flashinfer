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
#define SMEM_BITMAP_OFF 0
#define SMEM_BITMAP_STAGE_BYTES 32768
#define SMEM_BITMAP_STRIDE 32768
#define SMEM_TOTAL 32768
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_lm_head_det_loss_412aa4df26fa3d57c8f0(long long* __restrict__ ids, int* __restrict__ offsets, long long* __restrict__ labels, long long* __restrict__ word, int* __restrict__ lpos, int* __restrict__ offsets32, int* __restrict__ ids32, int T, int M, int V, int k_cap, int has_labels)
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
    unsigned int* bitmap = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int bitmap_addr = smem + 0;

    // === Task calls (declaration order) ===
    int lane_1 = lane;
    int wbase = warp * 1024;
    for (int w0 = lane_1; w0 < 1024; w0 += 32) {
        bitmap[wbase + w0] = 0;
    }
    __syncwarp();
    long long m64 = (long long)M;
    int num_windows = (V + 32767) / 32768;
    long long run = 9223372036854775807;
    int row_stride = gridDim.x * 8;
    if (bid == 0) {
        if (tid == 0) {
            long long o_end = (long long)offsets[T];
            long long e_lo = ((o_end > 0) ? o_end : 0);
            offsets32[T] = (int)(((e_lo < m64) ? e_lo : m64));
        }
    }
    for (int t = bid * 8 + warp; t < T; t += row_stride) {
        long long o_lo = (long long)offsets[t];
        long long o_hi = (long long)offsets[t + 1];
        long long t64 = (long long)t;
        long long cand = 9223372036854775807;
        cand = ((t == 0 && o_lo != 0) ? 1 : cand);
        long long w_dec = t64 << 8 | 2;
        cand = ((o_hi < o_lo && w_dec < cand) ? w_dec : cand);
        long long w_last = (long long)T << 8 | 3;
        cand = ((t == T - 1 && o_hi != m64 && w_last < cand) ? w_last : cand);
        long long c_lo = ((o_lo > 0) ? o_lo : 0);
        c_lo = ((c_lo < m64) ? c_lo : m64);
        long long c_hi = ((o_hi > c_lo) ? o_hi : c_lo);
        c_hi = ((c_hi < m64) ? c_hi : m64);
        bool over = o_hi - o_lo > (long long)k_cap;
        int lo = (int)c_lo;
        int k = ((over) ? 0 : (int)(c_hi - c_lo));
        if (lane_1 == 0) {
            offsets32[t] = lo;
        }
        long long lab = -100;
        if (has_labels != 0) {
            lab = labels[t];
        }
        bool need = lab != -100;
        int bad = 0;
        int dup = 0;
        int first = 2147483647;
        long long idv = 0;
        int hit = 0;
        if (k <= 32) {
            int key = -1 - lane_1;
            if (lane_1 < k) {
                idv = (long long)ids[lo + lane_1];
                long long v64 = (long long)V;
                bad = ((idv < 0 || idv >= v64) ? 1 : bad);
                hit = ((idv == lab) ? 1 : 0);
                long long lo_id = ((idv > 0) ? idv : 0);
                ids32[lo + lane_1] = (int)(((lo_id < v64) ? lo_id : v64 - 1));
                key = ((idv >= 0 && idv < (long long)V) ? (int)idv : key);
                first = ((hit != 0) ? lane_1 : first);
            }
            if (k >= 2) {
                int other_0 = __shfl_xor_sync(0xFFFFFFFF, key, 1);
                dup = ((other_0 == key && key >= 0) ? 1 : dup);
                int other_1 = __shfl_xor_sync(0xFFFFFFFF, key, 2);
                dup = ((other_1 == key && key >= 0) ? 1 : dup);
                int other_2 = __shfl_xor_sync(0xFFFFFFFF, key, 3);
                dup = ((other_2 == key && key >= 0) ? 1 : dup);
                int other_3 = __shfl_xor_sync(0xFFFFFFFF, key, 4);
                dup = ((other_3 == key && key >= 0) ? 1 : dup);
                int other_4 = __shfl_xor_sync(0xFFFFFFFF, key, 5);
                dup = ((other_4 == key && key >= 0) ? 1 : dup);
                int other_5 = __shfl_xor_sync(0xFFFFFFFF, key, 6);
                dup = ((other_5 == key && key >= 0) ? 1 : dup);
                int other_6 = __shfl_xor_sync(0xFFFFFFFF, key, 7);
                dup = ((other_6 == key && key >= 0) ? 1 : dup);
                int other_7 = __shfl_xor_sync(0xFFFFFFFF, key, 8);
                dup = ((other_7 == key && key >= 0) ? 1 : dup);
                int other_8 = __shfl_xor_sync(0xFFFFFFFF, key, 9);
                dup = ((other_8 == key && key >= 0) ? 1 : dup);
                int other_9 = __shfl_xor_sync(0xFFFFFFFF, key, 10);
                dup = ((other_9 == key && key >= 0) ? 1 : dup);
                int other_10 = __shfl_xor_sync(0xFFFFFFFF, key, 11);
                dup = ((other_10 == key && key >= 0) ? 1 : dup);
                int other_11 = __shfl_xor_sync(0xFFFFFFFF, key, 12);
                dup = ((other_11 == key && key >= 0) ? 1 : dup);
                int other_12 = __shfl_xor_sync(0xFFFFFFFF, key, 13);
                dup = ((other_12 == key && key >= 0) ? 1 : dup);
                int other_13 = __shfl_xor_sync(0xFFFFFFFF, key, 14);
                dup = ((other_13 == key && key >= 0) ? 1 : dup);
                int other_14 = __shfl_xor_sync(0xFFFFFFFF, key, 15);
                dup = ((other_14 == key && key >= 0) ? 1 : dup);
                int other_15 = __shfl_xor_sync(0xFFFFFFFF, key, 16);
                dup = ((other_15 == key && key >= 0) ? 1 : dup);
                int other_16 = __shfl_xor_sync(0xFFFFFFFF, key, 17);
                dup = ((other_16 == key && key >= 0) ? 1 : dup);
                int other_17 = __shfl_xor_sync(0xFFFFFFFF, key, 18);
                dup = ((other_17 == key && key >= 0) ? 1 : dup);
                int other_18 = __shfl_xor_sync(0xFFFFFFFF, key, 19);
                dup = ((other_18 == key && key >= 0) ? 1 : dup);
                int other_19 = __shfl_xor_sync(0xFFFFFFFF, key, 20);
                dup = ((other_19 == key && key >= 0) ? 1 : dup);
                int other_20 = __shfl_xor_sync(0xFFFFFFFF, key, 21);
                dup = ((other_20 == key && key >= 0) ? 1 : dup);
                int other_21 = __shfl_xor_sync(0xFFFFFFFF, key, 22);
                dup = ((other_21 == key && key >= 0) ? 1 : dup);
                int other_22 = __shfl_xor_sync(0xFFFFFFFF, key, 23);
                dup = ((other_22 == key && key >= 0) ? 1 : dup);
                int other_23 = __shfl_xor_sync(0xFFFFFFFF, key, 24);
                dup = ((other_23 == key && key >= 0) ? 1 : dup);
                int other_24 = __shfl_xor_sync(0xFFFFFFFF, key, 25);
                dup = ((other_24 == key && key >= 0) ? 1 : dup);
                int other_25 = __shfl_xor_sync(0xFFFFFFFF, key, 26);
                dup = ((other_25 == key && key >= 0) ? 1 : dup);
                int other_26 = __shfl_xor_sync(0xFFFFFFFF, key, 27);
                dup = ((other_26 == key && key >= 0) ? 1 : dup);
                int other_27 = __shfl_xor_sync(0xFFFFFFFF, key, 28);
                dup = ((other_27 == key && key >= 0) ? 1 : dup);
                int other_28 = __shfl_xor_sync(0xFFFFFFFF, key, 29);
                dup = ((other_28 == key && key >= 0) ? 1 : dup);
                int other_29 = __shfl_xor_sync(0xFFFFFFFF, key, 30);
                dup = ((other_29 == key && key >= 0) ? 1 : dup);
                int other_30 = __shfl_xor_sync(0xFFFFFFFF, key, 31);
                dup = ((other_30 == key && key >= 0) ? 1 : dup);
            }
        } else {
            for (int i = lane_1; i < k; i += 32) {
                idv = (long long)ids[lo + i];
                long long v64_1 = (long long)V;
                bad = ((idv < 0 || idv >= v64_1) ? 1 : bad);
                hit = ((idv == lab) ? 1 : 0);
                long long lo_id_1 = ((idv > 0) ? idv : 0);
                ids32[lo + i] = (int)(((lo_id_1 < v64_1) ? lo_id_1 : v64_1 - 1));
                first = ((hit != 0 && i < first) ? i : first);
            }
            for (int win = 0; win < num_windows; win++) {
                long long base = (long long)win * 32768;
                for (int im = lane_1; im < k; im += 32) {
                    long long id_m = (long long)ids[lo + im];
                    long long rel_m = id_m - base;
                    if (rel_m >= 0 && rel_m < 32768 && id_m < (long long)V) {
                        int r_m = (int)rel_m;
                        unsigned int bit = 1 << (unsigned int)(r_m & 31);
                        unsigned int old = atomicAdd(&bitmap[wbase + (r_m >> 5)], bit);
                        dup = (((old & bit) != 0) ? 1 : dup);
                    }
                }
                __syncwarp();
                for (int ic = lane_1; ic < k; ic += 32) {
                    long long id_c = (long long)ids[lo + ic];
                    long long rel_c = id_c - base;
                    if (rel_c >= 0 && rel_c < 32768 && id_c < (long long)V) {
                        atomicAnd(&bitmap[wbase + ((int)rel_c >> 5)], 0);
                    }
                }
                __syncwarp();
            }
        }
        int _warp_vote_0 = __any_sync(0xFFFFFFFF, bad != 0);
        bool any_bad = _warp_vote_0;
        int _warp_vote_1 = __any_sync(0xFFFFFFFF, dup != 0);
        bool any_dup = _warp_vote_1;
        int first_row;
        asm volatile("redux.sync.min.s32 %0, %1, 0xffffffff;" : "=r"(first_row) : "r"(first));
        bool found = first_row < 2147483647;
        int code = 0;
        code = ((need && !found) ? 20 : code);
        code = ((need && k == 0) ? 19 : code);
        code = ((any_dup) ? 18 : code);
        code = ((any_bad) ? 17 : code);
        code = ((over) ? 16 : code);
        long long w_row = 2305843009213693952 | t64 << 8 | (long long)code;
        cand = ((code != 0 && w_row < cand) ? w_row : cand);
        run = ((cand < run) ? cand : run);
        if (has_labels != 0) {
            if (lane_1 == 0) {
                lpos[t] = ((need && found) ? lo + first_row : -1);
            }
        }
    }
    if (run != 9223372036854775807) {
        if (lane_1 == 0) {
            atomicMin(&word[0], run);
        }
    }
}

} // extern "C"
