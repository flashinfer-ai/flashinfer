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

__global__ __launch_bounds__(128) void
kernel_cake_msa_icp_indexer_a5534981e50f1cd79a8a(int* __restrict__ recv, int* __restrict__ num_valid_pages, int* __restrict__ output, int* __restrict__ errors, int* __restrict__ expected, int recv_stride_src, int recv_stride_h, int recv_stride_t, int out_stride_t, int out_stride_h, int capacity, int num_tokens, int world, int tagged, int prefill, int init_blocks, int local_blocks, int spin_limit)
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
    unsigned int* ranked = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int ranked_addr = smem + 0;

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // === Task calls (declaration order) ===
    int lane_1 = (int)lane;
    int t = (int)blockIdx.x * 4 + (int)warp;
    int h = (int)blockIdx.y;
    int mode = 0;
    if (mode == 2) {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    if (t < num_tokens) {
        if (world == 2) {
            long long row = (long long)h * (long long)recv_stride_h + (long long)t * (long long)recv_stride_t;
            unsigned int gen = 0;
            long long counter = (long long)h * (long long)capacity + (long long)t;
            if (mode == 2) {
                gen = (unsigned int)expected[counter] + 1;
                if (gen == 0) {
                    gen = 1;
                }
            }
            unsigned int sb[1];
            int ib[1];
            int nv = 0;
            int early = (int)(mode != 1);
            if (early != 0) {
                nv = num_valid_pages[t];
            }
            int late = 0;
            unsigned long long* recv64 = reinterpret_cast<unsigned long long*>(recv);
            long long src_0 = (long long)(lane_1 >> 4);
            long long base_0 = row + src_0 * (long long)recv_stride_src;
            long long word_0 = base_0 + (long long)(lane_1 & 15);
            long long word64_0 = (base_0 >> 1) + (long long)(lane_1 & 15);
            sb[0] = (unsigned int)recv[word_0];
            ib[0] = recv[word_0 + 16];
            int pending = late;
            unsigned int stalled = __ballot_sync(0xFFFFFFFF, pending != 0);
            if (stalled != 0) {
                if (lane_1 == 0) {
                    atomicAdd(&errors[0], 1);
                    __threadfence();
                }
                __syncwarp();
                asm volatile("trap;" ::: "memory");
            }
            if (early == 0) {
                nv = num_valid_pages[t];
            }
            int n = ((nv < 8192) ? nv : 8192);
            int tail = n - local_blocks;
            tail = ((tail > 0) ? tail : 0);
            unsigned int ak[1];
            int ai[1];
            int cid_0 = ib[0];
            int valid_0 = (int)(cid_0 >= 0 && cid_0 < n);
            unsigned int ok_canon = sb[0];
            if ((sb[0] & 2147483647) > 2139095040) {
                ok_canon = 4286578688u;
            }
            if (sb[0] == 2147483648u) {
                ok_canon = 0;
            }
            unsigned int ok_key = ((ok_canon >> 31 != 0) ? ~ok_canon : ok_canon | 2147483648u);
            unsigned int key_0 = ok_key;
            if (cid_0 < init_blocks || cid_0 >= tail) {
                key_0 = 4286578688u;
            }
            ak[0] = ((valid_0 != 0) ? 2147483647 - (unsigned int)cid_0 : 0);
            ai[0] = ((valid_0 != 0) ? (int)(key_0 ^ 2147483647) : 2147483647);
            int bad = 0;
            unsigned int nb_0 = __shfl_down_sync(0xFFFFFFFF, ak[0], 1, 32);
            int nc_0 = __shfl_down_sync(0xFFFFFFFF, ai[0], 1, 32);
            int r = (int)(nb_0 > ak[0] || nb_0 == ak[0] && nc_0 < ai[0]);
            bad = bad | r & (int)((lane_1 & 15) != 15);
            unsigned int unsorted = __ballot_sync(0xFFFFFFFF, bad != 0);
            int ok = (int)(unsorted == 0);
            if (ok != 0) {
                unsigned int pb = __shfl_xor_sync(0xFFFFFFFF, ak[0], 31);
                int pc = __shfl_xor_sync(0xFFFFFFFF, ai[0], 31);
                int r_0 = (int)(pb > ak[0] || pb == ak[0] && pc < ai[0]);
                int po = r_0;
                int lower = (int)((lane_1 & 16) == 0);
                int tk = ((lower != 0) ? po : 1 - po);
                ak[0] = ((tk != 0) ? pb : ak[0]);
                ai[0] = ((tk != 0) ? pc : ai[0]);
                unsigned int pb_0 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 8);
                int pc_0 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 8);
                int r_1 = (int)(pb_0 > ak[0] || pb_0 == ak[0] && pc_0 < ai[0]);
                int po_0 = r_1;
                int lower_0 = (int)((lane_1 & 8) == 0);
                int tk_0 = ((lower_0 != 0) ? po_0 : 1 - po_0);
                ak[0] = ((tk_0 != 0) ? pb_0 : ak[0]);
                ai[0] = ((tk_0 != 0) ? pc_0 : ai[0]);
                unsigned int pb_1 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 4);
                int pc_1 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 4);
                int r_2 = (int)(pb_1 > ak[0] || pb_1 == ak[0] && pc_1 < ai[0]);
                int po_1 = r_2;
                int lower_1 = (int)((lane_1 & 4) == 0);
                int tk_1 = ((lower_1 != 0) ? po_1 : 1 - po_1);
                ak[0] = ((tk_1 != 0) ? pb_1 : ak[0]);
                ai[0] = ((tk_1 != 0) ? pc_1 : ai[0]);
                unsigned int pb_2 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 2);
                int pc_2 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 2);
                int r_3 = (int)(pb_2 > ak[0] || pb_2 == ak[0] && pc_2 < ai[0]);
                int po_2 = r_3;
                int lower_2 = (int)((lane_1 & 2) == 0);
                int tk_2 = ((lower_2 != 0) ? po_2 : 1 - po_2);
                ak[0] = ((tk_2 != 0) ? pb_2 : ak[0]);
                ai[0] = ((tk_2 != 0) ? pc_2 : ai[0]);
                unsigned int pb_3 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 1);
                int pc_3 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 1);
                int r_4 = (int)(pb_3 > ak[0] || pb_3 == ak[0] && pc_3 < ai[0]);
                int po_3 = r_4;
                int lower_3 = (int)((lane_1 & 1) == 0);
                int tk_3 = ((lower_3 != 0) ? po_3 : 1 - po_3);
                ak[0] = ((tk_3 != 0) ? pb_3 : ak[0]);
                ai[0] = ((tk_3 != 0) ? pc_3 : ai[0]);
            } else {
                unsigned int pb_4 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 1);
                int pc_4 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 1);
                int r_5 = (int)(pb_4 > ak[0] || pb_4 == ak[0] && pc_4 < ai[0]);
                int po_4 = r_5;
                int lower_4 = (int)((lane_1 & 1) == 0);
                int desc_0 = (int)((lane_1 & 2) == 0);
                int wmax_0 = desc_0 * lower_4 + (1 - desc_0) * (1 - lower_4);
                int tk_4 = ((wmax_0 != 0) ? po_4 : 1 - po_4);
                ak[0] = ((tk_4 != 0) ? pb_4 : ak[0]);
                ai[0] = ((tk_4 != 0) ? pc_4 : ai[0]);
                unsigned int pb_5 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 2);
                int pc_5 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 2);
                int r_6 = (int)(pb_5 > ak[0] || pb_5 == ak[0] && pc_5 < ai[0]);
                int po_5 = r_6;
                int lower_5 = (int)((lane_1 & 2) == 0);
                int desc_1 = (int)((lane_1 & 4) == 0);
                int wmax_1 = desc_1 * lower_5 + (1 - desc_1) * (1 - lower_5);
                int tk_5 = ((wmax_1 != 0) ? po_5 : 1 - po_5);
                ak[0] = ((tk_5 != 0) ? pb_5 : ak[0]);
                ai[0] = ((tk_5 != 0) ? pc_5 : ai[0]);
                unsigned int pb_6 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 1);
                int pc_6 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 1);
                int r_7 = (int)(pb_6 > ak[0] || pb_6 == ak[0] && pc_6 < ai[0]);
                int po_6 = r_7;
                int lower_6 = (int)((lane_1 & 1) == 0);
                int desc_2 = (int)((lane_1 & 4) == 0);
                int wmax_2 = desc_2 * lower_6 + (1 - desc_2) * (1 - lower_6);
                int tk_6 = ((wmax_2 != 0) ? po_6 : 1 - po_6);
                ak[0] = ((tk_6 != 0) ? pb_6 : ak[0]);
                ai[0] = ((tk_6 != 0) ? pc_6 : ai[0]);
                unsigned int pb_7 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 4);
                int pc_7 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 4);
                int r_8 = (int)(pb_7 > ak[0] || pb_7 == ak[0] && pc_7 < ai[0]);
                int po_7 = r_8;
                int lower_7 = (int)((lane_1 & 4) == 0);
                int desc_3 = (int)((lane_1 & 8) == 0);
                int wmax_3 = desc_3 * lower_7 + (1 - desc_3) * (1 - lower_7);
                int tk_7 = ((wmax_3 != 0) ? po_7 : 1 - po_7);
                ak[0] = ((tk_7 != 0) ? pb_7 : ak[0]);
                ai[0] = ((tk_7 != 0) ? pc_7 : ai[0]);
                unsigned int pb_8 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 2);
                int pc_8 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 2);
                int r_9 = (int)(pb_8 > ak[0] || pb_8 == ak[0] && pc_8 < ai[0]);
                int po_8 = r_9;
                int lower_8 = (int)((lane_1 & 2) == 0);
                int desc_4 = (int)((lane_1 & 8) == 0);
                int wmax_4 = desc_4 * lower_8 + (1 - desc_4) * (1 - lower_8);
                int tk_8 = ((wmax_4 != 0) ? po_8 : 1 - po_8);
                ak[0] = ((tk_8 != 0) ? pb_8 : ak[0]);
                ai[0] = ((tk_8 != 0) ? pc_8 : ai[0]);
                unsigned int pb_9 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 1);
                int pc_9 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 1);
                int r_10 = (int)(pb_9 > ak[0] || pb_9 == ak[0] && pc_9 < ai[0]);
                int po_9 = r_10;
                int lower_9 = (int)((lane_1 & 1) == 0);
                int desc_5 = (int)((lane_1 & 8) == 0);
                int wmax_5 = desc_5 * lower_9 + (1 - desc_5) * (1 - lower_9);
                int tk_9 = ((wmax_5 != 0) ? po_9 : 1 - po_9);
                ak[0] = ((tk_9 != 0) ? pb_9 : ak[0]);
                ai[0] = ((tk_9 != 0) ? pc_9 : ai[0]);
                unsigned int pb_10 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 8);
                int pc_10 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 8);
                int r_11 = (int)(pb_10 > ak[0] || pb_10 == ak[0] && pc_10 < ai[0]);
                int po_10 = r_11;
                int lower_10 = (int)((lane_1 & 8) == 0);
                int desc_6 = (int)((lane_1 & 16) == 0);
                int wmax_6 = desc_6 * lower_10 + (1 - desc_6) * (1 - lower_10);
                int tk_10 = ((wmax_6 != 0) ? po_10 : 1 - po_10);
                ak[0] = ((tk_10 != 0) ? pb_10 : ak[0]);
                ai[0] = ((tk_10 != 0) ? pc_10 : ai[0]);
                unsigned int pb_11 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 4);
                int pc_11 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 4);
                int r_12 = (int)(pb_11 > ak[0] || pb_11 == ak[0] && pc_11 < ai[0]);
                int po_11 = r_12;
                int lower_11 = (int)((lane_1 & 4) == 0);
                int desc_7 = (int)((lane_1 & 16) == 0);
                int wmax_7 = desc_7 * lower_11 + (1 - desc_7) * (1 - lower_11);
                int tk_11 = ((wmax_7 != 0) ? po_11 : 1 - po_11);
                ak[0] = ((tk_11 != 0) ? pb_11 : ak[0]);
                ai[0] = ((tk_11 != 0) ? pc_11 : ai[0]);
                unsigned int pb_12 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 2);
                int pc_12 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 2);
                int r_13 = (int)(pb_12 > ak[0] || pb_12 == ak[0] && pc_12 < ai[0]);
                int po_12 = r_13;
                int lower_12 = (int)((lane_1 & 2) == 0);
                int desc_8 = (int)((lane_1 & 16) == 0);
                int wmax_8 = desc_8 * lower_12 + (1 - desc_8) * (1 - lower_12);
                int tk_12 = ((wmax_8 != 0) ? po_12 : 1 - po_12);
                ak[0] = ((tk_12 != 0) ? pb_12 : ak[0]);
                ai[0] = ((tk_12 != 0) ? pc_12 : ai[0]);
                unsigned int pb_13 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 1);
                int pc_13 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 1);
                int r_14 = (int)(pb_13 > ak[0] || pb_13 == ak[0] && pc_13 < ai[0]);
                int po_13 = r_14;
                int lower_13 = (int)((lane_1 & 1) == 0);
                int desc_9 = (int)((lane_1 & 16) == 0);
                int wmax_9 = desc_9 * lower_13 + (1 - desc_9) * (1 - lower_13);
                int tk_13 = ((wmax_9 != 0) ? po_13 : 1 - po_13);
                ak[0] = ((tk_13 != 0) ? pb_13 : ak[0]);
                ai[0] = ((tk_13 != 0) ? pc_13 : ai[0]);
                unsigned int pb_14 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 16);
                int pc_14 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 16);
                int r_15 = (int)(pb_14 > ak[0] || pb_14 == ak[0] && pc_14 < ai[0]);
                int po_14 = r_15;
                int lower_14 = (int)((lane_1 & 16) == 0);
                int desc_10 = (int)((lane_1 & 0) == 0);
                int wmax_10 = desc_10 * lower_14 + (1 - desc_10) * (1 - lower_14);
                int tk_14 = ((wmax_10 != 0) ? po_14 : 1 - po_14);
                ak[0] = ((tk_14 != 0) ? pb_14 : ak[0]);
                ai[0] = ((tk_14 != 0) ? pc_14 : ai[0]);
                unsigned int pb_15 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 8);
                int pc_15 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 8);
                int r_16 = (int)(pb_15 > ak[0] || pb_15 == ak[0] && pc_15 < ai[0]);
                int po_15 = r_16;
                int lower_15 = (int)((lane_1 & 8) == 0);
                int desc_11 = (int)((lane_1 & 0) == 0);
                int wmax_11 = desc_11 * lower_15 + (1 - desc_11) * (1 - lower_15);
                int tk_15 = ((wmax_11 != 0) ? po_15 : 1 - po_15);
                ak[0] = ((tk_15 != 0) ? pb_15 : ak[0]);
                ai[0] = ((tk_15 != 0) ? pc_15 : ai[0]);
                unsigned int pb_16 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 4);
                int pc_16 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 4);
                int r_17 = (int)(pb_16 > ak[0] || pb_16 == ak[0] && pc_16 < ai[0]);
                int po_16 = r_17;
                int lower_16 = (int)((lane_1 & 4) == 0);
                int desc_12 = (int)((lane_1 & 0) == 0);
                int wmax_12 = desc_12 * lower_16 + (1 - desc_12) * (1 - lower_16);
                int tk_16 = ((wmax_12 != 0) ? po_16 : 1 - po_16);
                ak[0] = ((tk_16 != 0) ? pb_16 : ak[0]);
                ai[0] = ((tk_16 != 0) ? pc_16 : ai[0]);
                unsigned int pb_17 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 2);
                int pc_17 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 2);
                int r_18 = (int)(pb_17 > ak[0] || pb_17 == ak[0] && pc_17 < ai[0]);
                int po_17 = r_18;
                int lower_17 = (int)((lane_1 & 2) == 0);
                int desc_13 = (int)((lane_1 & 0) == 0);
                int wmax_13 = desc_13 * lower_17 + (1 - desc_13) * (1 - lower_17);
                int tk_17 = ((wmax_13 != 0) ? po_17 : 1 - po_17);
                ak[0] = ((tk_17 != 0) ? pb_17 : ak[0]);
                ai[0] = ((tk_17 != 0) ? pc_17 : ai[0]);
                unsigned int pb_18 = __shfl_xor_sync(0xFFFFFFFF, ak[0], 1);
                int pc_18 = __shfl_xor_sync(0xFFFFFFFF, ai[0], 1);
                int r_19 = (int)(pb_18 > ak[0] || pb_18 == ak[0] && pc_18 < ai[0]);
                int po_18 = r_19;
                int lower_18 = (int)((lane_1 & 1) == 0);
                int desc_14 = (int)((lane_1 & 0) == 0);
                int wmax_14 = desc_14 * lower_18 + (1 - desc_14) * (1 - lower_18);
                int tk_18 = ((wmax_14 != 0) ? po_18 : 1 - po_18);
                ak[0] = ((tk_18 != 0) ? pb_18 : ak[0]);
                ai[0] = ((tk_18 != 0) ? pc_18 : ai[0]);
            }
            unsigned int lk[1];
            int li[1];
            int lead[1];
            unsigned int prev_0 = __shfl_up_sync(0xFFFFFFFF, ak[0], 1, 32);
            int starts_0 = (int)(lane_1 == 0);
            int is_lead_0 = (int)(ak[0] != 0 && (starts_0 != 0 || prev_0 != ak[0]));
            lead[0] = is_lead_0;
            lk[0] = ((is_lead_0 != 0) ? (unsigned int)ai[0] ^ 2147483647 : 0);
            li[0] = ((is_lead_0 != 0) ? (int)(2147483647 - ak[0]) : 2147483647);
            int rk[1];
            rk[0] = 0;
            unsigned int bj_0 = __shfl_sync(0xFFFFFFFF, lk[0], 0);
            rk[0] = rk[0] + (int)(bj_0 > lk[0] || bj_0 == lk[0] && lane_1 > 0);
            unsigned int bj_1 = __shfl_sync(0xFFFFFFFF, lk[0], 1);
            rk[0] = rk[0] + (int)(bj_1 > lk[0] || bj_1 == lk[0] && lane_1 > 1);
            unsigned int bj_2 = __shfl_sync(0xFFFFFFFF, lk[0], 2);
            rk[0] = rk[0] + (int)(bj_2 > lk[0] || bj_2 == lk[0] && lane_1 > 2);
            unsigned int bj_3 = __shfl_sync(0xFFFFFFFF, lk[0], 3);
            rk[0] = rk[0] + (int)(bj_3 > lk[0] || bj_3 == lk[0] && lane_1 > 3);
            unsigned int bj_4 = __shfl_sync(0xFFFFFFFF, lk[0], 4);
            rk[0] = rk[0] + (int)(bj_4 > lk[0] || bj_4 == lk[0] && lane_1 > 4);
            unsigned int bj_5 = __shfl_sync(0xFFFFFFFF, lk[0], 5);
            rk[0] = rk[0] + (int)(bj_5 > lk[0] || bj_5 == lk[0] && lane_1 > 5);
            unsigned int bj_6 = __shfl_sync(0xFFFFFFFF, lk[0], 6);
            rk[0] = rk[0] + (int)(bj_6 > lk[0] || bj_6 == lk[0] && lane_1 > 6);
            unsigned int bj_7 = __shfl_sync(0xFFFFFFFF, lk[0], 7);
            rk[0] = rk[0] + (int)(bj_7 > lk[0] || bj_7 == lk[0] && lane_1 > 7);
            unsigned int bj_8 = __shfl_sync(0xFFFFFFFF, lk[0], 8);
            rk[0] = rk[0] + (int)(bj_8 > lk[0] || bj_8 == lk[0] && lane_1 > 8);
            unsigned int bj_9 = __shfl_sync(0xFFFFFFFF, lk[0], 9);
            rk[0] = rk[0] + (int)(bj_9 > lk[0] || bj_9 == lk[0] && lane_1 > 9);
            unsigned int bj_10 = __shfl_sync(0xFFFFFFFF, lk[0], 10);
            rk[0] = rk[0] + (int)(bj_10 > lk[0] || bj_10 == lk[0] && lane_1 > 10);
            unsigned int bj_11 = __shfl_sync(0xFFFFFFFF, lk[0], 11);
            rk[0] = rk[0] + (int)(bj_11 > lk[0] || bj_11 == lk[0] && lane_1 > 11);
            unsigned int bj_12 = __shfl_sync(0xFFFFFFFF, lk[0], 12);
            rk[0] = rk[0] + (int)(bj_12 > lk[0] || bj_12 == lk[0] && lane_1 > 12);
            unsigned int bj_13 = __shfl_sync(0xFFFFFFFF, lk[0], 13);
            rk[0] = rk[0] + (int)(bj_13 > lk[0] || bj_13 == lk[0] && lane_1 > 13);
            unsigned int bj_14 = __shfl_sync(0xFFFFFFFF, lk[0], 14);
            rk[0] = rk[0] + (int)(bj_14 > lk[0] || bj_14 == lk[0] && lane_1 > 14);
            unsigned int bj_15 = __shfl_sync(0xFFFFFFFF, lk[0], 15);
            rk[0] = rk[0] + (int)(bj_15 > lk[0] || bj_15 == lk[0] && lane_1 > 15);
            unsigned int bj_16 = __shfl_sync(0xFFFFFFFF, lk[0], 16);
            rk[0] = rk[0] + (int)(bj_16 > lk[0] || bj_16 == lk[0] && lane_1 > 16);
            unsigned int bj_17 = __shfl_sync(0xFFFFFFFF, lk[0], 17);
            rk[0] = rk[0] + (int)(bj_17 > lk[0] || bj_17 == lk[0] && lane_1 > 17);
            unsigned int bj_18 = __shfl_sync(0xFFFFFFFF, lk[0], 18);
            rk[0] = rk[0] + (int)(bj_18 > lk[0] || bj_18 == lk[0] && lane_1 > 18);
            unsigned int bj_19 = __shfl_sync(0xFFFFFFFF, lk[0], 19);
            rk[0] = rk[0] + (int)(bj_19 > lk[0] || bj_19 == lk[0] && lane_1 > 19);
            unsigned int bj_20 = __shfl_sync(0xFFFFFFFF, lk[0], 20);
            rk[0] = rk[0] + (int)(bj_20 > lk[0] || bj_20 == lk[0] && lane_1 > 20);
            unsigned int bj_21 = __shfl_sync(0xFFFFFFFF, lk[0], 21);
            rk[0] = rk[0] + (int)(bj_21 > lk[0] || bj_21 == lk[0] && lane_1 > 21);
            unsigned int bj_22 = __shfl_sync(0xFFFFFFFF, lk[0], 22);
            rk[0] = rk[0] + (int)(bj_22 > lk[0] || bj_22 == lk[0] && lane_1 > 22);
            unsigned int bj_23 = __shfl_sync(0xFFFFFFFF, lk[0], 23);
            rk[0] = rk[0] + (int)(bj_23 > lk[0] || bj_23 == lk[0] && lane_1 > 23);
            unsigned int bj_24 = __shfl_sync(0xFFFFFFFF, lk[0], 24);
            rk[0] = rk[0] + (int)(bj_24 > lk[0] || bj_24 == lk[0] && lane_1 > 24);
            unsigned int bj_25 = __shfl_sync(0xFFFFFFFF, lk[0], 25);
            rk[0] = rk[0] + (int)(bj_25 > lk[0] || bj_25 == lk[0] && lane_1 > 25);
            unsigned int bj_26 = __shfl_sync(0xFFFFFFFF, lk[0], 26);
            rk[0] = rk[0] + (int)(bj_26 > lk[0] || bj_26 == lk[0] && lane_1 > 26);
            unsigned int bj_27 = __shfl_sync(0xFFFFFFFF, lk[0], 27);
            rk[0] = rk[0] + (int)(bj_27 > lk[0] || bj_27 == lk[0] && lane_1 > 27);
            unsigned int bj_28 = __shfl_sync(0xFFFFFFFF, lk[0], 28);
            rk[0] = rk[0] + (int)(bj_28 > lk[0] || bj_28 == lk[0] && lane_1 > 28);
            unsigned int bj_29 = __shfl_sync(0xFFFFFFFF, lk[0], 29);
            rk[0] = rk[0] + (int)(bj_29 > lk[0] || bj_29 == lk[0] && lane_1 > 29);
            unsigned int bj_30 = __shfl_sync(0xFFFFFFFF, lk[0], 30);
            rk[0] = rk[0] + (int)(bj_30 > lk[0] || bj_30 == lk[0] && lane_1 > 30);
            unsigned int bj_31 = __shfl_sync(0xFFFFFFFF, lk[0], 31);
            rk[0] = rk[0] + (int)(bj_31 > lk[0] || bj_31 == lk[0] && lane_1 > 31);
            int pick[1];
            pick[0] = (int)(lead[0] != 0 && rk[0] < 16);
            long long dst = (long long)t * (long long)out_stride_t + (long long)h * (long long)out_stride_h;
            int taken = 0;
            int sel_0 = pick[0];
            unsigned int chosen_0 = __ballot_sync(0xFFFFFFFF, sel_0 != 0);
            int _rank_in_mask_0 = (int)__popc(chosen_0 & ((1u << (unsigned int)lane) - 1u));
            int pos_0 = taken + (int)_rank_in_mask_0;
            if (sel_0 != 0) {
                output[dst + (long long)pos_0] = li[0];
            }
            taken = taken + (int)(unsigned int)__popc(chosen_0);
            if (lane_1 >= taken && lane_1 < 16) {
                int pad = -1;
                if (mode == 0 && nv == 0 && lane_1 == 0) {
                    pad = 0;
                }
                output[dst + (long long)lane_1] = pad;
            }
            if (mode == 1) {
                __syncwarp();
                if (lane_1 < world) {
                    asm volatile("red.release.sys.global.and.b32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(recv + (row + (long long)lane_1 * (long long)recv_stride_src + 32)))), "r"(static_cast<unsigned int>(0)) : "memory");
                }
            }
            if (mode == 2) {
                __syncwarp();
                if (lane_1 == 0) {
                    expected[counter] = (int)gen;
                }
            }
        } else {
            long long row_1 = (long long)h * (long long)recv_stride_h + (long long)t * (long long)recv_stride_t;
            unsigned int gen_1 = 0;
            long long counter_1 = (long long)h * (long long)capacity + (long long)t;
            if (mode == 2) {
                gen_1 = (unsigned int)expected[counter_1] + 1;
                if (gen_1 == 0) {
                    gen_1 = 1;
                }
            }
            unsigned int sb_1[2];
            int ib_1[2];
            int nv_1 = 0;
            int early_1 = (int)(mode != 1);
            if (early_1 != 0) {
                nv_1 = num_valid_pages[t];
            }
            int late_1 = 0;
            unsigned long long* recv64_1 = reinterpret_cast<unsigned long long*>(recv);
            long long src_1 = (long long)(lane_1 >> 4);
            long long base_1 = row_1 + src_1 * (long long)recv_stride_src;
            long long word_1 = base_1 + (long long)(lane_1 & 15);
            long long word64_1 = (base_1 >> 1) + (long long)(lane_1 & 15);
            sb_1[0] = (unsigned int)recv[word_1];
            ib_1[0] = recv[word_1 + 16];
            long long src_2 = (long long)(2 + (lane_1 >> 4));
            long long base_2 = row_1 + src_2 * (long long)recv_stride_src;
            long long word_2 = base_2 + (long long)(lane_1 & 15);
            long long word64_2 = (base_2 >> 1) + (long long)(lane_1 & 15);
            sb_1[1] = (unsigned int)recv[word_2];
            ib_1[1] = recv[word_2 + 16];
            int pending_1 = late_1;
            unsigned int stalled_1 = __ballot_sync(0xFFFFFFFF, pending_1 != 0);
            if (stalled_1 != 0) {
                if (lane_1 == 0) {
                    atomicAdd(&errors[0], 1);
                    __threadfence();
                }
                __syncwarp();
                asm volatile("trap;" ::: "memory");
            }
            if (early_1 == 0) {
                nv_1 = num_valid_pages[t];
            }
            int n_1 = ((nv_1 < 8192) ? nv_1 : 8192);
            int tail_1 = n_1 - local_blocks;
            tail_1 = ((tail_1 > 0) ? tail_1 : 0);
            unsigned int ak_1[2];
            int ai_1[2];
            int cid_1 = ib_1[0];
            int valid_1 = (int)(cid_1 >= 0 && cid_1 < n_1);
            unsigned int ok_canon_1 = sb_1[0];
            if ((sb_1[0] & 2147483647) > 2139095040) {
                ok_canon_1 = 4286578688u;
            }
            if (sb_1[0] == 2147483648u) {
                ok_canon_1 = 0;
            }
            unsigned int ok_key_1 = ((ok_canon_1 >> 31 != 0) ? ~ok_canon_1 : ok_canon_1 | 2147483648u);
            unsigned int key_1 = ok_key_1;
            if (cid_1 < init_blocks || cid_1 >= tail_1) {
                key_1 = 4286578688u;
            }
            ak_1[0] = ((valid_1 != 0) ? 2147483647 - (unsigned int)cid_1 : 0);
            ai_1[0] = ((valid_1 != 0) ? (int)(key_1 ^ 2147483647) : 2147483647);
            int cid_2 = ib_1[1];
            int valid_2 = (int)(cid_2 >= 0 && cid_2 < n_1);
            unsigned int ok_canon_0 = sb_1[1];
            if ((sb_1[1] & 2147483647) > 2139095040) {
                ok_canon_0 = 4286578688u;
            }
            if (sb_1[1] == 2147483648u) {
                ok_canon_0 = 0;
            }
            unsigned int ok_key_0 = ((ok_canon_0 >> 31 != 0) ? ~ok_canon_0 : ok_canon_0 | 2147483648u);
            unsigned int key_2 = ok_key_0;
            if (cid_2 < init_blocks || cid_2 >= tail_1) {
                key_2 = 4286578688u;
            }
            ak_1[1] = ((valid_2 != 0) ? 2147483647 - (unsigned int)cid_2 : 0);
            ai_1[1] = ((valid_2 != 0) ? (int)(key_2 ^ 2147483647) : 2147483647);
            int bad_1 = 0;
            unsigned int nb_1 = __shfl_down_sync(0xFFFFFFFF, ak_1[0], 1, 32);
            int nc_1 = __shfl_down_sync(0xFFFFFFFF, ai_1[0], 1, 32);
            int r_20 = (int)(nb_1 > ak_1[0] || nb_1 == ak_1[0] && nc_1 < ai_1[0]);
            bad_1 = bad_1 | r_20 & (int)((lane_1 & 15) != 15);
            unsigned int nb_2 = __shfl_down_sync(0xFFFFFFFF, ak_1[1], 1, 32);
            int nc_2 = __shfl_down_sync(0xFFFFFFFF, ai_1[1], 1, 32);
            int r_20_1 = (int)(nb_2 > ak_1[1] || nb_2 == ak_1[1] && nc_2 < ai_1[1]);
            bad_1 = bad_1 | r_20_1 & (int)((lane_1 & 15) != 15);
            unsigned int unsorted_1 = __ballot_sync(0xFFFFFFFF, bad_1 != 0);
            int ok_1 = (int)(unsorted_1 == 0);
            if (ok_1 != 0) {
                unsigned int pb_19 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 31);
                int pc_19 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 31);
                int r_21 = (int)(pb_19 > ak_1[0] || pb_19 == ak_1[0] && pc_19 < ai_1[0]);
                int po_19 = r_21;
                int lower_19 = (int)((lane_1 & 16) == 0);
                int tk_19 = ((lower_19 != 0) ? po_19 : 1 - po_19);
                ak_1[0] = ((tk_19 != 0) ? pb_19 : ak_1[0]);
                ai_1[0] = ((tk_19 != 0) ? pc_19 : ai_1[0]);
                unsigned int pb_19_1 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 31);
                int pc_19_1 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 31);
                int r_22 = (int)(pb_19_1 > ak_1[1] || pb_19_1 == ak_1[1] && pc_19_1 < ai_1[1]);
                int po_19_1 = r_22;
                int lower_19_1 = (int)((lane_1 & 16) == 0);
                int tk_19_1 = ((lower_19_1 != 0) ? po_19_1 : 1 - po_19_1);
                ak_1[1] = ((tk_19_1 != 0) ? pb_19_1 : ak_1[1]);
                ai_1[1] = ((tk_19_1 != 0) ? pc_19_1 : ai_1[1]);
                unsigned int pb_20 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 8);
                int pc_20 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 8);
                int r_23 = (int)(pb_20 > ak_1[0] || pb_20 == ak_1[0] && pc_20 < ai_1[0]);
                int po_20 = r_23;
                int lower_20 = (int)((lane_1 & 8) == 0);
                int tk_20 = ((lower_20 != 0) ? po_20 : 1 - po_20);
                ak_1[0] = ((tk_20 != 0) ? pb_20 : ak_1[0]);
                ai_1[0] = ((tk_20 != 0) ? pc_20 : ai_1[0]);
                unsigned int pb_21 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 8);
                int pc_21 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 8);
                int r_24 = (int)(pb_21 > ak_1[1] || pb_21 == ak_1[1] && pc_21 < ai_1[1]);
                int po_21 = r_24;
                int lower_21 = (int)((lane_1 & 8) == 0);
                int tk_21 = ((lower_21 != 0) ? po_21 : 1 - po_21);
                ak_1[1] = ((tk_21 != 0) ? pb_21 : ak_1[1]);
                ai_1[1] = ((tk_21 != 0) ? pc_21 : ai_1[1]);
                unsigned int pb_22 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 4);
                int pc_22 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 4);
                int r_25 = (int)(pb_22 > ak_1[0] || pb_22 == ak_1[0] && pc_22 < ai_1[0]);
                int po_22 = r_25;
                int lower_22 = (int)((lane_1 & 4) == 0);
                int tk_22 = ((lower_22 != 0) ? po_22 : 1 - po_22);
                ak_1[0] = ((tk_22 != 0) ? pb_22 : ak_1[0]);
                ai_1[0] = ((tk_22 != 0) ? pc_22 : ai_1[0]);
                unsigned int pb_23 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 4);
                int pc_23 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 4);
                int r_26 = (int)(pb_23 > ak_1[1] || pb_23 == ak_1[1] && pc_23 < ai_1[1]);
                int po_23 = r_26;
                int lower_23 = (int)((lane_1 & 4) == 0);
                int tk_23 = ((lower_23 != 0) ? po_23 : 1 - po_23);
                ak_1[1] = ((tk_23 != 0) ? pb_23 : ak_1[1]);
                ai_1[1] = ((tk_23 != 0) ? pc_23 : ai_1[1]);
                unsigned int pb_24 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 2);
                int pc_24 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 2);
                int r_27 = (int)(pb_24 > ak_1[0] || pb_24 == ak_1[0] && pc_24 < ai_1[0]);
                int po_24 = r_27;
                int lower_24 = (int)((lane_1 & 2) == 0);
                int tk_24 = ((lower_24 != 0) ? po_24 : 1 - po_24);
                ak_1[0] = ((tk_24 != 0) ? pb_24 : ak_1[0]);
                ai_1[0] = ((tk_24 != 0) ? pc_24 : ai_1[0]);
                unsigned int pb_25 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 2);
                int pc_25 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 2);
                int r_28 = (int)(pb_25 > ak_1[1] || pb_25 == ak_1[1] && pc_25 < ai_1[1]);
                int po_25 = r_28;
                int lower_25 = (int)((lane_1 & 2) == 0);
                int tk_25 = ((lower_25 != 0) ? po_25 : 1 - po_25);
                ak_1[1] = ((tk_25 != 0) ? pb_25 : ak_1[1]);
                ai_1[1] = ((tk_25 != 0) ? pc_25 : ai_1[1]);
                unsigned int pb_26 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 1);
                int pc_26 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 1);
                int r_29 = (int)(pb_26 > ak_1[0] || pb_26 == ak_1[0] && pc_26 < ai_1[0]);
                int po_26 = r_29;
                int lower_26 = (int)((lane_1 & 1) == 0);
                int tk_26 = ((lower_26 != 0) ? po_26 : 1 - po_26);
                ak_1[0] = ((tk_26 != 0) ? pb_26 : ak_1[0]);
                ai_1[0] = ((tk_26 != 0) ? pc_26 : ai_1[0]);
                unsigned int pb_27 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 1);
                int pc_27 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 1);
                int r_30 = (int)(pb_27 > ak_1[1] || pb_27 == ak_1[1] && pc_27 < ai_1[1]);
                int po_27 = r_30;
                int lower_27 = (int)((lane_1 & 1) == 0);
                int tk_27 = ((lower_27 != 0) ? po_27 : 1 - po_27);
                ak_1[1] = ((tk_27 != 0) ? pb_27 : ak_1[1]);
                ai_1[1] = ((tk_27 != 0) ? pc_27 : ai_1[1]);
                unsigned int x0b = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 31);
                int x0c = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 31);
                unsigned int x1b = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 31);
                int x1c = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 31);
                int r_31 = (int)(x0b > ak_1[0] || x0b == ak_1[0] && x0c < ai_1[0]);
                int t0 = r_31;
                int r_32 = (int)(x1b > ak_1[1] || x1b == ak_1[1] && x1c < ai_1[1]);
                int t1 = 1 - r_32;
                ak_1[0] = ((t0 != 0) ? x0b : ak_1[0]);
                ai_1[0] = ((t0 != 0) ? x0c : ai_1[0]);
                ak_1[1] = ((t1 != 0) ? x1b : ak_1[1]);
                ai_1[1] = ((t1 != 0) ? x1c : ai_1[1]);
                unsigned int pb_28 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 16);
                int pc_28 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 16);
                int r_33 = (int)(pb_28 > ak_1[0] || pb_28 == ak_1[0] && pc_28 < ai_1[0]);
                int po_28 = r_33;
                int lower_28 = (int)((lane_1 & 16) == 0);
                int tk_28 = ((lower_28 != 0) ? po_28 : 1 - po_28);
                ak_1[0] = ((tk_28 != 0) ? pb_28 : ak_1[0]);
                ai_1[0] = ((tk_28 != 0) ? pc_28 : ai_1[0]);
                unsigned int pb_29 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 16);
                int pc_29 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 16);
                int r_34 = (int)(pb_29 > ak_1[1] || pb_29 == ak_1[1] && pc_29 < ai_1[1]);
                int po_29 = r_34;
                int lower_29 = (int)((lane_1 & 16) == 0);
                int tk_29 = ((lower_29 != 0) ? po_29 : 1 - po_29);
                ak_1[1] = ((tk_29 != 0) ? pb_29 : ak_1[1]);
                ai_1[1] = ((tk_29 != 0) ? pc_29 : ai_1[1]);
                unsigned int pb_30 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 8);
                int pc_30 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 8);
                int r_35 = (int)(pb_30 > ak_1[0] || pb_30 == ak_1[0] && pc_30 < ai_1[0]);
                int po_30 = r_35;
                int lower_30 = (int)((lane_1 & 8) == 0);
                int tk_30 = ((lower_30 != 0) ? po_30 : 1 - po_30);
                ak_1[0] = ((tk_30 != 0) ? pb_30 : ak_1[0]);
                ai_1[0] = ((tk_30 != 0) ? pc_30 : ai_1[0]);
                unsigned int pb_31 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 8);
                int pc_31 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 8);
                int r_36 = (int)(pb_31 > ak_1[1] || pb_31 == ak_1[1] && pc_31 < ai_1[1]);
                int po_31 = r_36;
                int lower_31 = (int)((lane_1 & 8) == 0);
                int tk_31 = ((lower_31 != 0) ? po_31 : 1 - po_31);
                ak_1[1] = ((tk_31 != 0) ? pb_31 : ak_1[1]);
                ai_1[1] = ((tk_31 != 0) ? pc_31 : ai_1[1]);
                unsigned int pb_32 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 4);
                int pc_32 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 4);
                int r_37 = (int)(pb_32 > ak_1[0] || pb_32 == ak_1[0] && pc_32 < ai_1[0]);
                int po_32 = r_37;
                int lower_32 = (int)((lane_1 & 4) == 0);
                int tk_32 = ((lower_32 != 0) ? po_32 : 1 - po_32);
                ak_1[0] = ((tk_32 != 0) ? pb_32 : ak_1[0]);
                ai_1[0] = ((tk_32 != 0) ? pc_32 : ai_1[0]);
                unsigned int pb_33 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 4);
                int pc_33 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 4);
                int r_38 = (int)(pb_33 > ak_1[1] || pb_33 == ak_1[1] && pc_33 < ai_1[1]);
                int po_33 = r_38;
                int lower_33 = (int)((lane_1 & 4) == 0);
                int tk_33 = ((lower_33 != 0) ? po_33 : 1 - po_33);
                ak_1[1] = ((tk_33 != 0) ? pb_33 : ak_1[1]);
                ai_1[1] = ((tk_33 != 0) ? pc_33 : ai_1[1]);
                unsigned int pb_34 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 2);
                int pc_34 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 2);
                int r_39 = (int)(pb_34 > ak_1[0] || pb_34 == ak_1[0] && pc_34 < ai_1[0]);
                int po_34 = r_39;
                int lower_34 = (int)((lane_1 & 2) == 0);
                int tk_34 = ((lower_34 != 0) ? po_34 : 1 - po_34);
                ak_1[0] = ((tk_34 != 0) ? pb_34 : ak_1[0]);
                ai_1[0] = ((tk_34 != 0) ? pc_34 : ai_1[0]);
                unsigned int pb_35 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 2);
                int pc_35 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 2);
                int r_40 = (int)(pb_35 > ak_1[1] || pb_35 == ak_1[1] && pc_35 < ai_1[1]);
                int po_35 = r_40;
                int lower_35 = (int)((lane_1 & 2) == 0);
                int tk_35 = ((lower_35 != 0) ? po_35 : 1 - po_35);
                ak_1[1] = ((tk_35 != 0) ? pb_35 : ak_1[1]);
                ai_1[1] = ((tk_35 != 0) ? pc_35 : ai_1[1]);
                unsigned int pb_36 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 1);
                int pc_36 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 1);
                int r_41 = (int)(pb_36 > ak_1[0] || pb_36 == ak_1[0] && pc_36 < ai_1[0]);
                int po_36 = r_41;
                int lower_36 = (int)((lane_1 & 1) == 0);
                int tk_36 = ((lower_36 != 0) ? po_36 : 1 - po_36);
                ak_1[0] = ((tk_36 != 0) ? pb_36 : ak_1[0]);
                ai_1[0] = ((tk_36 != 0) ? pc_36 : ai_1[0]);
                unsigned int pb_37 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 1);
                int pc_37 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 1);
                int r_42 = (int)(pb_37 > ak_1[1] || pb_37 == ak_1[1] && pc_37 < ai_1[1]);
                int po_37 = r_42;
                int lower_37 = (int)((lane_1 & 1) == 0);
                int tk_37 = ((lower_37 != 0) ? po_37 : 1 - po_37);
                ak_1[1] = ((tk_37 != 0) ? pb_37 : ak_1[1]);
                ai_1[1] = ((tk_37 != 0) ? pc_37 : ai_1[1]);
            } else {
                unsigned int pb_38 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 1);
                int pc_38 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 1);
                int r_43 = (int)(pb_38 > ak_1[0] || pb_38 == ak_1[0] && pc_38 < ai_1[0]);
                int po_38 = r_43;
                int lower_38 = (int)((lane_1 & 1) == 0);
                int desc_15 = (int)((lane_1 & 2) == 0);
                int wmax_15 = desc_15 * lower_38 + (1 - desc_15) * (1 - lower_38);
                int tk_38 = ((wmax_15 != 0) ? po_38 : 1 - po_38);
                ak_1[0] = ((tk_38 != 0) ? pb_38 : ak_1[0]);
                ai_1[0] = ((tk_38 != 0) ? pc_38 : ai_1[0]);
                unsigned int pb_39 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 1);
                int pc_39 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 1);
                int r_44 = (int)(pb_39 > ak_1[1] || pb_39 == ak_1[1] && pc_39 < ai_1[1]);
                int po_39 = r_44;
                int lower_39 = (int)((lane_1 & 1) == 0);
                int desc_16 = (int)((lane_1 & 2) == 0);
                int wmax_16 = desc_16 * lower_39 + (1 - desc_16) * (1 - lower_39);
                int tk_39 = ((wmax_16 != 0) ? po_39 : 1 - po_39);
                ak_1[1] = ((tk_39 != 0) ? pb_39 : ak_1[1]);
                ai_1[1] = ((tk_39 != 0) ? pc_39 : ai_1[1]);
                unsigned int pb_40 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 2);
                int pc_40 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 2);
                int r_45 = (int)(pb_40 > ak_1[0] || pb_40 == ak_1[0] && pc_40 < ai_1[0]);
                int po_40 = r_45;
                int lower_40 = (int)((lane_1 & 2) == 0);
                int desc_17 = (int)((lane_1 & 4) == 0);
                int wmax_17 = desc_17 * lower_40 + (1 - desc_17) * (1 - lower_40);
                int tk_40 = ((wmax_17 != 0) ? po_40 : 1 - po_40);
                ak_1[0] = ((tk_40 != 0) ? pb_40 : ak_1[0]);
                ai_1[0] = ((tk_40 != 0) ? pc_40 : ai_1[0]);
                unsigned int pb_41 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 2);
                int pc_41 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 2);
                int r_46 = (int)(pb_41 > ak_1[1] || pb_41 == ak_1[1] && pc_41 < ai_1[1]);
                int po_41 = r_46;
                int lower_41 = (int)((lane_1 & 2) == 0);
                int desc_18 = (int)((lane_1 & 4) == 0);
                int wmax_18 = desc_18 * lower_41 + (1 - desc_18) * (1 - lower_41);
                int tk_41 = ((wmax_18 != 0) ? po_41 : 1 - po_41);
                ak_1[1] = ((tk_41 != 0) ? pb_41 : ak_1[1]);
                ai_1[1] = ((tk_41 != 0) ? pc_41 : ai_1[1]);
                unsigned int pb_42 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 1);
                int pc_42 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 1);
                int r_47 = (int)(pb_42 > ak_1[0] || pb_42 == ak_1[0] && pc_42 < ai_1[0]);
                int po_42 = r_47;
                int lower_42 = (int)((lane_1 & 1) == 0);
                int desc_19 = (int)((lane_1 & 4) == 0);
                int wmax_19 = desc_19 * lower_42 + (1 - desc_19) * (1 - lower_42);
                int tk_42 = ((wmax_19 != 0) ? po_42 : 1 - po_42);
                ak_1[0] = ((tk_42 != 0) ? pb_42 : ak_1[0]);
                ai_1[0] = ((tk_42 != 0) ? pc_42 : ai_1[0]);
                unsigned int pb_43 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 1);
                int pc_43 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 1);
                int r_48 = (int)(pb_43 > ak_1[1] || pb_43 == ak_1[1] && pc_43 < ai_1[1]);
                int po_43 = r_48;
                int lower_43 = (int)((lane_1 & 1) == 0);
                int desc_20 = (int)((lane_1 & 4) == 0);
                int wmax_20 = desc_20 * lower_43 + (1 - desc_20) * (1 - lower_43);
                int tk_43 = ((wmax_20 != 0) ? po_43 : 1 - po_43);
                ak_1[1] = ((tk_43 != 0) ? pb_43 : ak_1[1]);
                ai_1[1] = ((tk_43 != 0) ? pc_43 : ai_1[1]);
                unsigned int pb_44 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 4);
                int pc_44 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 4);
                int r_49 = (int)(pb_44 > ak_1[0] || pb_44 == ak_1[0] && pc_44 < ai_1[0]);
                int po_44 = r_49;
                int lower_44 = (int)((lane_1 & 4) == 0);
                int desc_21 = (int)((lane_1 & 8) == 0);
                int wmax_21 = desc_21 * lower_44 + (1 - desc_21) * (1 - lower_44);
                int tk_44 = ((wmax_21 != 0) ? po_44 : 1 - po_44);
                ak_1[0] = ((tk_44 != 0) ? pb_44 : ak_1[0]);
                ai_1[0] = ((tk_44 != 0) ? pc_44 : ai_1[0]);
                unsigned int pb_45 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 4);
                int pc_45 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 4);
                int r_50 = (int)(pb_45 > ak_1[1] || pb_45 == ak_1[1] && pc_45 < ai_1[1]);
                int po_45 = r_50;
                int lower_45 = (int)((lane_1 & 4) == 0);
                int desc_22 = (int)((lane_1 & 8) == 0);
                int wmax_22 = desc_22 * lower_45 + (1 - desc_22) * (1 - lower_45);
                int tk_45 = ((wmax_22 != 0) ? po_45 : 1 - po_45);
                ak_1[1] = ((tk_45 != 0) ? pb_45 : ak_1[1]);
                ai_1[1] = ((tk_45 != 0) ? pc_45 : ai_1[1]);
                unsigned int pb_46 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 2);
                int pc_46 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 2);
                int r_51 = (int)(pb_46 > ak_1[0] || pb_46 == ak_1[0] && pc_46 < ai_1[0]);
                int po_46 = r_51;
                int lower_46 = (int)((lane_1 & 2) == 0);
                int desc_23 = (int)((lane_1 & 8) == 0);
                int wmax_23 = desc_23 * lower_46 + (1 - desc_23) * (1 - lower_46);
                int tk_46 = ((wmax_23 != 0) ? po_46 : 1 - po_46);
                ak_1[0] = ((tk_46 != 0) ? pb_46 : ak_1[0]);
                ai_1[0] = ((tk_46 != 0) ? pc_46 : ai_1[0]);
                unsigned int pb_47 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 2);
                int pc_47 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 2);
                int r_52 = (int)(pb_47 > ak_1[1] || pb_47 == ak_1[1] && pc_47 < ai_1[1]);
                int po_47 = r_52;
                int lower_47 = (int)((lane_1 & 2) == 0);
                int desc_24 = (int)((lane_1 & 8) == 0);
                int wmax_24 = desc_24 * lower_47 + (1 - desc_24) * (1 - lower_47);
                int tk_47 = ((wmax_24 != 0) ? po_47 : 1 - po_47);
                ak_1[1] = ((tk_47 != 0) ? pb_47 : ak_1[1]);
                ai_1[1] = ((tk_47 != 0) ? pc_47 : ai_1[1]);
                unsigned int pb_48 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 1);
                int pc_48 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 1);
                int r_53 = (int)(pb_48 > ak_1[0] || pb_48 == ak_1[0] && pc_48 < ai_1[0]);
                int po_48 = r_53;
                int lower_48 = (int)((lane_1 & 1) == 0);
                int desc_25 = (int)((lane_1 & 8) == 0);
                int wmax_25 = desc_25 * lower_48 + (1 - desc_25) * (1 - lower_48);
                int tk_48 = ((wmax_25 != 0) ? po_48 : 1 - po_48);
                ak_1[0] = ((tk_48 != 0) ? pb_48 : ak_1[0]);
                ai_1[0] = ((tk_48 != 0) ? pc_48 : ai_1[0]);
                unsigned int pb_49 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 1);
                int pc_49 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 1);
                int r_54 = (int)(pb_49 > ak_1[1] || pb_49 == ak_1[1] && pc_49 < ai_1[1]);
                int po_49 = r_54;
                int lower_49 = (int)((lane_1 & 1) == 0);
                int desc_26 = (int)((lane_1 & 8) == 0);
                int wmax_26 = desc_26 * lower_49 + (1 - desc_26) * (1 - lower_49);
                int tk_49 = ((wmax_26 != 0) ? po_49 : 1 - po_49);
                ak_1[1] = ((tk_49 != 0) ? pb_49 : ak_1[1]);
                ai_1[1] = ((tk_49 != 0) ? pc_49 : ai_1[1]);
                unsigned int pb_50 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 8);
                int pc_50 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 8);
                int r_55 = (int)(pb_50 > ak_1[0] || pb_50 == ak_1[0] && pc_50 < ai_1[0]);
                int po_50 = r_55;
                int lower_50 = (int)((lane_1 & 8) == 0);
                int desc_27 = (int)((lane_1 & 16) == 0);
                int wmax_27 = desc_27 * lower_50 + (1 - desc_27) * (1 - lower_50);
                int tk_50 = ((wmax_27 != 0) ? po_50 : 1 - po_50);
                ak_1[0] = ((tk_50 != 0) ? pb_50 : ak_1[0]);
                ai_1[0] = ((tk_50 != 0) ? pc_50 : ai_1[0]);
                unsigned int pb_51 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 8);
                int pc_51 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 8);
                int r_56 = (int)(pb_51 > ak_1[1] || pb_51 == ak_1[1] && pc_51 < ai_1[1]);
                int po_51 = r_56;
                int lower_51 = (int)((lane_1 & 8) == 0);
                int desc_28 = (int)((lane_1 & 16) == 0);
                int wmax_28 = desc_28 * lower_51 + (1 - desc_28) * (1 - lower_51);
                int tk_51 = ((wmax_28 != 0) ? po_51 : 1 - po_51);
                ak_1[1] = ((tk_51 != 0) ? pb_51 : ak_1[1]);
                ai_1[1] = ((tk_51 != 0) ? pc_51 : ai_1[1]);
                unsigned int pb_52 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 4);
                int pc_52 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 4);
                int r_57 = (int)(pb_52 > ak_1[0] || pb_52 == ak_1[0] && pc_52 < ai_1[0]);
                int po_52 = r_57;
                int lower_52 = (int)((lane_1 & 4) == 0);
                int desc_29 = (int)((lane_1 & 16) == 0);
                int wmax_29 = desc_29 * lower_52 + (1 - desc_29) * (1 - lower_52);
                int tk_52 = ((wmax_29 != 0) ? po_52 : 1 - po_52);
                ak_1[0] = ((tk_52 != 0) ? pb_52 : ak_1[0]);
                ai_1[0] = ((tk_52 != 0) ? pc_52 : ai_1[0]);
                unsigned int pb_53 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 4);
                int pc_53 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 4);
                int r_58 = (int)(pb_53 > ak_1[1] || pb_53 == ak_1[1] && pc_53 < ai_1[1]);
                int po_53 = r_58;
                int lower_53 = (int)((lane_1 & 4) == 0);
                int desc_30 = (int)((lane_1 & 16) == 0);
                int wmax_30 = desc_30 * lower_53 + (1 - desc_30) * (1 - lower_53);
                int tk_53 = ((wmax_30 != 0) ? po_53 : 1 - po_53);
                ak_1[1] = ((tk_53 != 0) ? pb_53 : ak_1[1]);
                ai_1[1] = ((tk_53 != 0) ? pc_53 : ai_1[1]);
                unsigned int pb_54 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 2);
                int pc_54 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 2);
                int r_59 = (int)(pb_54 > ak_1[0] || pb_54 == ak_1[0] && pc_54 < ai_1[0]);
                int po_54 = r_59;
                int lower_54 = (int)((lane_1 & 2) == 0);
                int desc_31 = (int)((lane_1 & 16) == 0);
                int wmax_31 = desc_31 * lower_54 + (1 - desc_31) * (1 - lower_54);
                int tk_54 = ((wmax_31 != 0) ? po_54 : 1 - po_54);
                ak_1[0] = ((tk_54 != 0) ? pb_54 : ak_1[0]);
                ai_1[0] = ((tk_54 != 0) ? pc_54 : ai_1[0]);
                unsigned int pb_55 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 2);
                int pc_55 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 2);
                int r_60 = (int)(pb_55 > ak_1[1] || pb_55 == ak_1[1] && pc_55 < ai_1[1]);
                int po_55 = r_60;
                int lower_55 = (int)((lane_1 & 2) == 0);
                int desc_32 = (int)((lane_1 & 16) == 0);
                int wmax_32 = desc_32 * lower_55 + (1 - desc_32) * (1 - lower_55);
                int tk_55 = ((wmax_32 != 0) ? po_55 : 1 - po_55);
                ak_1[1] = ((tk_55 != 0) ? pb_55 : ak_1[1]);
                ai_1[1] = ((tk_55 != 0) ? pc_55 : ai_1[1]);
                unsigned int pb_56 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 1);
                int pc_56 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 1);
                int r_61 = (int)(pb_56 > ak_1[0] || pb_56 == ak_1[0] && pc_56 < ai_1[0]);
                int po_56 = r_61;
                int lower_56 = (int)((lane_1 & 1) == 0);
                int desc_33 = (int)((lane_1 & 16) == 0);
                int wmax_33 = desc_33 * lower_56 + (1 - desc_33) * (1 - lower_56);
                int tk_56 = ((wmax_33 != 0) ? po_56 : 1 - po_56);
                ak_1[0] = ((tk_56 != 0) ? pb_56 : ak_1[0]);
                ai_1[0] = ((tk_56 != 0) ? pc_56 : ai_1[0]);
                unsigned int pb_57 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 1);
                int pc_57 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 1);
                int r_62 = (int)(pb_57 > ak_1[1] || pb_57 == ak_1[1] && pc_57 < ai_1[1]);
                int po_57 = r_62;
                int lower_57 = (int)((lane_1 & 1) == 0);
                int desc_34 = (int)((lane_1 & 16) == 0);
                int wmax_34 = desc_34 * lower_57 + (1 - desc_34) * (1 - lower_57);
                int tk_57 = ((wmax_34 != 0) ? po_57 : 1 - po_57);
                ak_1[1] = ((tk_57 != 0) ? pb_57 : ak_1[1]);
                ai_1[1] = ((tk_57 != 0) ? pc_57 : ai_1[1]);
                unsigned int pb_58 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 16);
                int pc_58 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 16);
                int r_63 = (int)(pb_58 > ak_1[0] || pb_58 == ak_1[0] && pc_58 < ai_1[0]);
                int po_58 = r_63;
                int lower_58 = (int)((lane_1 & 16) == 0);
                int desc_35 = (int)((lane_1 & 0) == 0);
                int wmax_35 = desc_35 * lower_58 + (1 - desc_35) * (1 - lower_58);
                int tk_58 = ((wmax_35 != 0) ? po_58 : 1 - po_58);
                ak_1[0] = ((tk_58 != 0) ? pb_58 : ak_1[0]);
                ai_1[0] = ((tk_58 != 0) ? pc_58 : ai_1[0]);
                unsigned int pb_59 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 16);
                int pc_59 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 16);
                int r_64 = (int)(pb_59 > ak_1[1] || pb_59 == ak_1[1] && pc_59 < ai_1[1]);
                int po_59 = r_64;
                int lower_59 = (int)((lane_1 & 16) == 0);
                int desc_36 = 0;
                int wmax_36 = desc_36 * lower_59 + (1 - desc_36) * (1 - lower_59);
                int tk_59 = ((wmax_36 != 0) ? po_59 : 1 - po_59);
                ak_1[1] = ((tk_59 != 0) ? pb_59 : ak_1[1]);
                ai_1[1] = ((tk_59 != 0) ? pc_59 : ai_1[1]);
                unsigned int pb_60 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 8);
                int pc_60 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 8);
                int r_65 = (int)(pb_60 > ak_1[0] || pb_60 == ak_1[0] && pc_60 < ai_1[0]);
                int po_60 = r_65;
                int lower_60 = (int)((lane_1 & 8) == 0);
                int desc_37 = (int)((lane_1 & 0) == 0);
                int wmax_37 = desc_37 * lower_60 + (1 - desc_37) * (1 - lower_60);
                int tk_60 = ((wmax_37 != 0) ? po_60 : 1 - po_60);
                ak_1[0] = ((tk_60 != 0) ? pb_60 : ak_1[0]);
                ai_1[0] = ((tk_60 != 0) ? pc_60 : ai_1[0]);
                unsigned int pb_61 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 8);
                int pc_61 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 8);
                int r_66 = (int)(pb_61 > ak_1[1] || pb_61 == ak_1[1] && pc_61 < ai_1[1]);
                int po_61 = r_66;
                int lower_61 = (int)((lane_1 & 8) == 0);
                int desc_38 = 0;
                int wmax_38 = desc_38 * lower_61 + (1 - desc_38) * (1 - lower_61);
                int tk_61 = ((wmax_38 != 0) ? po_61 : 1 - po_61);
                ak_1[1] = ((tk_61 != 0) ? pb_61 : ak_1[1]);
                ai_1[1] = ((tk_61 != 0) ? pc_61 : ai_1[1]);
                unsigned int pb_62 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 4);
                int pc_62 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 4);
                int r_67 = (int)(pb_62 > ak_1[0] || pb_62 == ak_1[0] && pc_62 < ai_1[0]);
                int po_62 = r_67;
                int lower_62 = (int)((lane_1 & 4) == 0);
                int desc_39 = (int)((lane_1 & 0) == 0);
                int wmax_39 = desc_39 * lower_62 + (1 - desc_39) * (1 - lower_62);
                int tk_62 = ((wmax_39 != 0) ? po_62 : 1 - po_62);
                ak_1[0] = ((tk_62 != 0) ? pb_62 : ak_1[0]);
                ai_1[0] = ((tk_62 != 0) ? pc_62 : ai_1[0]);
                unsigned int pb_63 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 4);
                int pc_63 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 4);
                int r_68 = (int)(pb_63 > ak_1[1] || pb_63 == ak_1[1] && pc_63 < ai_1[1]);
                int po_63 = r_68;
                int lower_63 = (int)((lane_1 & 4) == 0);
                int desc_40 = 0;
                int wmax_40 = desc_40 * lower_63 + (1 - desc_40) * (1 - lower_63);
                int tk_63 = ((wmax_40 != 0) ? po_63 : 1 - po_63);
                ak_1[1] = ((tk_63 != 0) ? pb_63 : ak_1[1]);
                ai_1[1] = ((tk_63 != 0) ? pc_63 : ai_1[1]);
                unsigned int pb_64 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 2);
                int pc_64 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 2);
                int r_69 = (int)(pb_64 > ak_1[0] || pb_64 == ak_1[0] && pc_64 < ai_1[0]);
                int po_64 = r_69;
                int lower_64 = (int)((lane_1 & 2) == 0);
                int desc_41 = (int)((lane_1 & 0) == 0);
                int wmax_41 = desc_41 * lower_64 + (1 - desc_41) * (1 - lower_64);
                int tk_64 = ((wmax_41 != 0) ? po_64 : 1 - po_64);
                ak_1[0] = ((tk_64 != 0) ? pb_64 : ak_1[0]);
                ai_1[0] = ((tk_64 != 0) ? pc_64 : ai_1[0]);
                unsigned int pb_65 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 2);
                int pc_65 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 2);
                int r_70 = (int)(pb_65 > ak_1[1] || pb_65 == ak_1[1] && pc_65 < ai_1[1]);
                int po_65 = r_70;
                int lower_65 = (int)((lane_1 & 2) == 0);
                int desc_42 = 0;
                int wmax_42 = desc_42 * lower_65 + (1 - desc_42) * (1 - lower_65);
                int tk_65 = ((wmax_42 != 0) ? po_65 : 1 - po_65);
                ak_1[1] = ((tk_65 != 0) ? pb_65 : ak_1[1]);
                ai_1[1] = ((tk_65 != 0) ? pc_65 : ai_1[1]);
                unsigned int pb_66 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 1);
                int pc_66 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 1);
                int r_71 = (int)(pb_66 > ak_1[0] || pb_66 == ak_1[0] && pc_66 < ai_1[0]);
                int po_66 = r_71;
                int lower_66 = (int)((lane_1 & 1) == 0);
                int desc_43 = (int)((lane_1 & 0) == 0);
                int wmax_43 = desc_43 * lower_66 + (1 - desc_43) * (1 - lower_66);
                int tk_66 = ((wmax_43 != 0) ? po_66 : 1 - po_66);
                ak_1[0] = ((tk_66 != 0) ? pb_66 : ak_1[0]);
                ai_1[0] = ((tk_66 != 0) ? pc_66 : ai_1[0]);
                unsigned int pb_67 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 1);
                int pc_67 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 1);
                int r_72 = (int)(pb_67 > ak_1[1] || pb_67 == ak_1[1] && pc_67 < ai_1[1]);
                int po_67 = r_72;
                int lower_67 = (int)((lane_1 & 1) == 0);
                int desc_44 = 0;
                int wmax_44 = desc_44 * lower_67 + (1 - desc_44) * (1 - lower_67);
                int tk_67 = ((wmax_44 != 0) ? po_67 : 1 - po_67);
                ak_1[1] = ((tk_67 != 0) ? pb_67 : ak_1[1]);
                ai_1[1] = ((tk_67 != 0) ? pc_67 : ai_1[1]);
                unsigned int xk = ak_1[0];
                int xi = ai_1[0];
                unsigned int yk = ak_1[1];
                int yi = ai_1[1];
                int take = (int)(yk > xk || yk == xk && yi < xi);
                ak_1[0] = ((take != 0) ? yk : xk);
                ai_1[0] = ((take != 0) ? yi : xi);
                ak_1[1] = ((take != 0) ? xk : yk);
                ai_1[1] = ((take != 0) ? xi : yi);
                unsigned int pb_68 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 16);
                int pc_68 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 16);
                int r_73 = (int)(pb_68 > ak_1[0] || pb_68 == ak_1[0] && pc_68 < ai_1[0]);
                int po_68 = r_73;
                int lower_68 = (int)((lane_1 & 16) == 0);
                int desc_45 = (int)((lane_1 & 0) == 0);
                int wmax_45 = desc_45 * lower_68 + (1 - desc_45) * (1 - lower_68);
                int tk_68 = ((wmax_45 != 0) ? po_68 : 1 - po_68);
                ak_1[0] = ((tk_68 != 0) ? pb_68 : ak_1[0]);
                ai_1[0] = ((tk_68 != 0) ? pc_68 : ai_1[0]);
                unsigned int pb_69 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 16);
                int pc_69 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 16);
                int r_74 = (int)(pb_69 > ak_1[1] || pb_69 == ak_1[1] && pc_69 < ai_1[1]);
                int po_69 = r_74;
                int lower_69 = (int)((lane_1 & 16) == 0);
                int desc_46 = (int)((lane_1 & 0) == 0);
                int wmax_46 = desc_46 * lower_69 + (1 - desc_46) * (1 - lower_69);
                int tk_69 = ((wmax_46 != 0) ? po_69 : 1 - po_69);
                ak_1[1] = ((tk_69 != 0) ? pb_69 : ak_1[1]);
                ai_1[1] = ((tk_69 != 0) ? pc_69 : ai_1[1]);
                unsigned int pb_70 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 8);
                int pc_70 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 8);
                int r_75 = (int)(pb_70 > ak_1[0] || pb_70 == ak_1[0] && pc_70 < ai_1[0]);
                int po_70 = r_75;
                int lower_70 = (int)((lane_1 & 8) == 0);
                int desc_47 = (int)((lane_1 & 0) == 0);
                int wmax_47 = desc_47 * lower_70 + (1 - desc_47) * (1 - lower_70);
                int tk_70 = ((wmax_47 != 0) ? po_70 : 1 - po_70);
                ak_1[0] = ((tk_70 != 0) ? pb_70 : ak_1[0]);
                ai_1[0] = ((tk_70 != 0) ? pc_70 : ai_1[0]);
                unsigned int pb_71 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 8);
                int pc_71 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 8);
                int r_76 = (int)(pb_71 > ak_1[1] || pb_71 == ak_1[1] && pc_71 < ai_1[1]);
                int po_71 = r_76;
                int lower_71 = (int)((lane_1 & 8) == 0);
                int desc_48 = (int)((lane_1 & 0) == 0);
                int wmax_48 = desc_48 * lower_71 + (1 - desc_48) * (1 - lower_71);
                int tk_71 = ((wmax_48 != 0) ? po_71 : 1 - po_71);
                ak_1[1] = ((tk_71 != 0) ? pb_71 : ak_1[1]);
                ai_1[1] = ((tk_71 != 0) ? pc_71 : ai_1[1]);
                unsigned int pb_72 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 4);
                int pc_72 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 4);
                int r_77 = (int)(pb_72 > ak_1[0] || pb_72 == ak_1[0] && pc_72 < ai_1[0]);
                int po_72 = r_77;
                int lower_72 = (int)((lane_1 & 4) == 0);
                int desc_49 = (int)((lane_1 & 0) == 0);
                int wmax_49 = desc_49 * lower_72 + (1 - desc_49) * (1 - lower_72);
                int tk_72 = ((wmax_49 != 0) ? po_72 : 1 - po_72);
                ak_1[0] = ((tk_72 != 0) ? pb_72 : ak_1[0]);
                ai_1[0] = ((tk_72 != 0) ? pc_72 : ai_1[0]);
                unsigned int pb_73 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 4);
                int pc_73 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 4);
                int r_78 = (int)(pb_73 > ak_1[1] || pb_73 == ak_1[1] && pc_73 < ai_1[1]);
                int po_73 = r_78;
                int lower_73 = (int)((lane_1 & 4) == 0);
                int desc_50 = (int)((lane_1 & 0) == 0);
                int wmax_50 = desc_50 * lower_73 + (1 - desc_50) * (1 - lower_73);
                int tk_73 = ((wmax_50 != 0) ? po_73 : 1 - po_73);
                ak_1[1] = ((tk_73 != 0) ? pb_73 : ak_1[1]);
                ai_1[1] = ((tk_73 != 0) ? pc_73 : ai_1[1]);
                unsigned int pb_74 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 2);
                int pc_74 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 2);
                int r_79 = (int)(pb_74 > ak_1[0] || pb_74 == ak_1[0] && pc_74 < ai_1[0]);
                int po_74 = r_79;
                int lower_74 = (int)((lane_1 & 2) == 0);
                int desc_51 = (int)((lane_1 & 0) == 0);
                int wmax_51 = desc_51 * lower_74 + (1 - desc_51) * (1 - lower_74);
                int tk_74 = ((wmax_51 != 0) ? po_74 : 1 - po_74);
                ak_1[0] = ((tk_74 != 0) ? pb_74 : ak_1[0]);
                ai_1[0] = ((tk_74 != 0) ? pc_74 : ai_1[0]);
                unsigned int pb_75 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 2);
                int pc_75 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 2);
                int r_80 = (int)(pb_75 > ak_1[1] || pb_75 == ak_1[1] && pc_75 < ai_1[1]);
                int po_75 = r_80;
                int lower_75 = (int)((lane_1 & 2) == 0);
                int desc_52 = (int)((lane_1 & 0) == 0);
                int wmax_52 = desc_52 * lower_75 + (1 - desc_52) * (1 - lower_75);
                int tk_75 = ((wmax_52 != 0) ? po_75 : 1 - po_75);
                ak_1[1] = ((tk_75 != 0) ? pb_75 : ak_1[1]);
                ai_1[1] = ((tk_75 != 0) ? pc_75 : ai_1[1]);
                unsigned int pb_76 = __shfl_xor_sync(0xFFFFFFFF, ak_1[0], 1);
                int pc_76 = __shfl_xor_sync(0xFFFFFFFF, ai_1[0], 1);
                int r_81 = (int)(pb_76 > ak_1[0] || pb_76 == ak_1[0] && pc_76 < ai_1[0]);
                int po_76 = r_81;
                int lower_76 = (int)((lane_1 & 1) == 0);
                int desc_53 = (int)((lane_1 & 0) == 0);
                int wmax_53 = desc_53 * lower_76 + (1 - desc_53) * (1 - lower_76);
                int tk_76 = ((wmax_53 != 0) ? po_76 : 1 - po_76);
                ak_1[0] = ((tk_76 != 0) ? pb_76 : ak_1[0]);
                ai_1[0] = ((tk_76 != 0) ? pc_76 : ai_1[0]);
                unsigned int pb_77 = __shfl_xor_sync(0xFFFFFFFF, ak_1[1], 1);
                int pc_77 = __shfl_xor_sync(0xFFFFFFFF, ai_1[1], 1);
                int r_82 = (int)(pb_77 > ak_1[1] || pb_77 == ak_1[1] && pc_77 < ai_1[1]);
                int po_77 = r_82;
                int lower_77 = (int)((lane_1 & 1) == 0);
                int desc_54 = (int)((lane_1 & 0) == 0);
                int wmax_54 = desc_54 * lower_77 + (1 - desc_54) * (1 - lower_77);
                int tk_77 = ((wmax_54 != 0) ? po_77 : 1 - po_77);
                ak_1[1] = ((tk_77 != 0) ? pb_77 : ak_1[1]);
                ai_1[1] = ((tk_77 != 0) ? pc_77 : ai_1[1]);
            }
            unsigned int lk_1[2];
            int li_1[2];
            int lead_1[2];
            unsigned int prev_1 = __shfl_up_sync(0xFFFFFFFF, ak_1[0], 1, 32);
            int starts_1 = (int)(lane_1 == 0);
            int is_lead_1 = (int)(ak_1[0] != 0 && (starts_1 != 0 || prev_1 != ak_1[0]));
            lead_1[0] = is_lead_1;
            lk_1[0] = ((is_lead_1 != 0) ? (unsigned int)ai_1[0] ^ 2147483647 : 0);
            li_1[0] = ((is_lead_1 != 0) ? (int)(2147483647 - ak_1[0]) : 2147483647);
            unsigned int prev_2 = __shfl_up_sync(0xFFFFFFFF, ak_1[1], 1, 32);
            unsigned int wrap_0 = __shfl_sync(0xFFFFFFFF, ak_1[0], 31);
            prev_2 = ((lane_1 == 0) ? wrap_0 : prev_2);
            int starts_2 = 0;
            int is_lead_2 = (int)(ak_1[1] != 0 && (starts_2 != 0 || prev_2 != ak_1[1]));
            lead_1[1] = is_lead_2;
            lk_1[1] = ((is_lead_2 != 0) ? (unsigned int)ai_1[1] ^ 2147483647 : 0);
            li_1[1] = ((is_lead_2 != 0) ? (int)(2147483647 - ak_1[1]) : 2147483647);
            int rk_1[2];
            rk_1[0] = 0;
            rk_1[1] = 0;
            unsigned int bj_32 = __shfl_sync(0xFFFFFFFF, lk_1[0], 0);
            rk_1[0] = rk_1[0] + (int)(bj_32 > lk_1[0] || bj_32 == lk_1[0] && lane_1 > 0);
            unsigned int bj_33 = __shfl_sync(0xFFFFFFFF, lk_1[1], 0);
            rk_1[1] = rk_1[1] + (int)(bj_33 > lk_1[1] || bj_33 == lk_1[1] && lane_1 > 0);
            unsigned int bj_34 = __shfl_sync(0xFFFFFFFF, lk_1[0], 1);
            rk_1[0] = rk_1[0] + (int)(bj_34 > lk_1[0] || bj_34 == lk_1[0] && lane_1 > 1);
            unsigned int bj_35 = __shfl_sync(0xFFFFFFFF, lk_1[1], 1);
            rk_1[1] = rk_1[1] + (int)(bj_35 > lk_1[1] || bj_35 == lk_1[1] && lane_1 > 1);
            unsigned int bj_36 = __shfl_sync(0xFFFFFFFF, lk_1[0], 2);
            rk_1[0] = rk_1[0] + (int)(bj_36 > lk_1[0] || bj_36 == lk_1[0] && lane_1 > 2);
            unsigned int bj_37 = __shfl_sync(0xFFFFFFFF, lk_1[1], 2);
            rk_1[1] = rk_1[1] + (int)(bj_37 > lk_1[1] || bj_37 == lk_1[1] && lane_1 > 2);
            unsigned int bj_38 = __shfl_sync(0xFFFFFFFF, lk_1[0], 3);
            rk_1[0] = rk_1[0] + (int)(bj_38 > lk_1[0] || bj_38 == lk_1[0] && lane_1 > 3);
            unsigned int bj_39 = __shfl_sync(0xFFFFFFFF, lk_1[1], 3);
            rk_1[1] = rk_1[1] + (int)(bj_39 > lk_1[1] || bj_39 == lk_1[1] && lane_1 > 3);
            unsigned int bj_40 = __shfl_sync(0xFFFFFFFF, lk_1[0], 4);
            rk_1[0] = rk_1[0] + (int)(bj_40 > lk_1[0] || bj_40 == lk_1[0] && lane_1 > 4);
            unsigned int bj_41 = __shfl_sync(0xFFFFFFFF, lk_1[1], 4);
            rk_1[1] = rk_1[1] + (int)(bj_41 > lk_1[1] || bj_41 == lk_1[1] && lane_1 > 4);
            unsigned int bj_42 = __shfl_sync(0xFFFFFFFF, lk_1[0], 5);
            rk_1[0] = rk_1[0] + (int)(bj_42 > lk_1[0] || bj_42 == lk_1[0] && lane_1 > 5);
            unsigned int bj_43 = __shfl_sync(0xFFFFFFFF, lk_1[1], 5);
            rk_1[1] = rk_1[1] + (int)(bj_43 > lk_1[1] || bj_43 == lk_1[1] && lane_1 > 5);
            unsigned int bj_44 = __shfl_sync(0xFFFFFFFF, lk_1[0], 6);
            rk_1[0] = rk_1[0] + (int)(bj_44 > lk_1[0] || bj_44 == lk_1[0] && lane_1 > 6);
            unsigned int bj_45 = __shfl_sync(0xFFFFFFFF, lk_1[1], 6);
            rk_1[1] = rk_1[1] + (int)(bj_45 > lk_1[1] || bj_45 == lk_1[1] && lane_1 > 6);
            unsigned int bj_46 = __shfl_sync(0xFFFFFFFF, lk_1[0], 7);
            rk_1[0] = rk_1[0] + (int)(bj_46 > lk_1[0] || bj_46 == lk_1[0] && lane_1 > 7);
            unsigned int bj_47 = __shfl_sync(0xFFFFFFFF, lk_1[1], 7);
            rk_1[1] = rk_1[1] + (int)(bj_47 > lk_1[1] || bj_47 == lk_1[1] && lane_1 > 7);
            unsigned int bj_48 = __shfl_sync(0xFFFFFFFF, lk_1[0], 8);
            rk_1[0] = rk_1[0] + (int)(bj_48 > lk_1[0] || bj_48 == lk_1[0] && lane_1 > 8);
            unsigned int bj_49 = __shfl_sync(0xFFFFFFFF, lk_1[1], 8);
            rk_1[1] = rk_1[1] + (int)(bj_49 > lk_1[1] || bj_49 == lk_1[1] && lane_1 > 8);
            unsigned int bj_50 = __shfl_sync(0xFFFFFFFF, lk_1[0], 9);
            rk_1[0] = rk_1[0] + (int)(bj_50 > lk_1[0] || bj_50 == lk_1[0] && lane_1 > 9);
            unsigned int bj_51 = __shfl_sync(0xFFFFFFFF, lk_1[1], 9);
            rk_1[1] = rk_1[1] + (int)(bj_51 > lk_1[1] || bj_51 == lk_1[1] && lane_1 > 9);
            unsigned int bj_52 = __shfl_sync(0xFFFFFFFF, lk_1[0], 10);
            rk_1[0] = rk_1[0] + (int)(bj_52 > lk_1[0] || bj_52 == lk_1[0] && lane_1 > 10);
            unsigned int bj_53 = __shfl_sync(0xFFFFFFFF, lk_1[1], 10);
            rk_1[1] = rk_1[1] + (int)(bj_53 > lk_1[1] || bj_53 == lk_1[1] && lane_1 > 10);
            unsigned int bj_54 = __shfl_sync(0xFFFFFFFF, lk_1[0], 11);
            rk_1[0] = rk_1[0] + (int)(bj_54 > lk_1[0] || bj_54 == lk_1[0] && lane_1 > 11);
            unsigned int bj_55 = __shfl_sync(0xFFFFFFFF, lk_1[1], 11);
            rk_1[1] = rk_1[1] + (int)(bj_55 > lk_1[1] || bj_55 == lk_1[1] && lane_1 > 11);
            unsigned int bj_56 = __shfl_sync(0xFFFFFFFF, lk_1[0], 12);
            rk_1[0] = rk_1[0] + (int)(bj_56 > lk_1[0] || bj_56 == lk_1[0] && lane_1 > 12);
            unsigned int bj_57 = __shfl_sync(0xFFFFFFFF, lk_1[1], 12);
            rk_1[1] = rk_1[1] + (int)(bj_57 > lk_1[1] || bj_57 == lk_1[1] && lane_1 > 12);
            unsigned int bj_58 = __shfl_sync(0xFFFFFFFF, lk_1[0], 13);
            rk_1[0] = rk_1[0] + (int)(bj_58 > lk_1[0] || bj_58 == lk_1[0] && lane_1 > 13);
            unsigned int bj_59 = __shfl_sync(0xFFFFFFFF, lk_1[1], 13);
            rk_1[1] = rk_1[1] + (int)(bj_59 > lk_1[1] || bj_59 == lk_1[1] && lane_1 > 13);
            unsigned int bj_60 = __shfl_sync(0xFFFFFFFF, lk_1[0], 14);
            rk_1[0] = rk_1[0] + (int)(bj_60 > lk_1[0] || bj_60 == lk_1[0] && lane_1 > 14);
            unsigned int bj_61 = __shfl_sync(0xFFFFFFFF, lk_1[1], 14);
            rk_1[1] = rk_1[1] + (int)(bj_61 > lk_1[1] || bj_61 == lk_1[1] && lane_1 > 14);
            unsigned int bj_62 = __shfl_sync(0xFFFFFFFF, lk_1[0], 15);
            rk_1[0] = rk_1[0] + (int)(bj_62 > lk_1[0] || bj_62 == lk_1[0] && lane_1 > 15);
            unsigned int bj_63 = __shfl_sync(0xFFFFFFFF, lk_1[1], 15);
            rk_1[1] = rk_1[1] + (int)(bj_63 > lk_1[1] || bj_63 == lk_1[1] && lane_1 > 15);
            unsigned int bj_64 = __shfl_sync(0xFFFFFFFF, lk_1[0], 16);
            rk_1[0] = rk_1[0] + (int)(bj_64 > lk_1[0] || bj_64 == lk_1[0] && lane_1 > 16);
            unsigned int bj_65 = __shfl_sync(0xFFFFFFFF, lk_1[1], 16);
            rk_1[1] = rk_1[1] + (int)(bj_65 > lk_1[1] || bj_65 == lk_1[1] && lane_1 > 16);
            unsigned int bj_66 = __shfl_sync(0xFFFFFFFF, lk_1[0], 17);
            rk_1[0] = rk_1[0] + (int)(bj_66 > lk_1[0] || bj_66 == lk_1[0] && lane_1 > 17);
            unsigned int bj_67 = __shfl_sync(0xFFFFFFFF, lk_1[1], 17);
            rk_1[1] = rk_1[1] + (int)(bj_67 > lk_1[1] || bj_67 == lk_1[1] && lane_1 > 17);
            unsigned int bj_68 = __shfl_sync(0xFFFFFFFF, lk_1[0], 18);
            rk_1[0] = rk_1[0] + (int)(bj_68 > lk_1[0] || bj_68 == lk_1[0] && lane_1 > 18);
            unsigned int bj_69 = __shfl_sync(0xFFFFFFFF, lk_1[1], 18);
            rk_1[1] = rk_1[1] + (int)(bj_69 > lk_1[1] || bj_69 == lk_1[1] && lane_1 > 18);
            unsigned int bj_70 = __shfl_sync(0xFFFFFFFF, lk_1[0], 19);
            rk_1[0] = rk_1[0] + (int)(bj_70 > lk_1[0] || bj_70 == lk_1[0] && lane_1 > 19);
            unsigned int bj_71 = __shfl_sync(0xFFFFFFFF, lk_1[1], 19);
            rk_1[1] = rk_1[1] + (int)(bj_71 > lk_1[1] || bj_71 == lk_1[1] && lane_1 > 19);
            unsigned int bj_72 = __shfl_sync(0xFFFFFFFF, lk_1[0], 20);
            rk_1[0] = rk_1[0] + (int)(bj_72 > lk_1[0] || bj_72 == lk_1[0] && lane_1 > 20);
            unsigned int bj_73 = __shfl_sync(0xFFFFFFFF, lk_1[1], 20);
            rk_1[1] = rk_1[1] + (int)(bj_73 > lk_1[1] || bj_73 == lk_1[1] && lane_1 > 20);
            unsigned int bj_74 = __shfl_sync(0xFFFFFFFF, lk_1[0], 21);
            rk_1[0] = rk_1[0] + (int)(bj_74 > lk_1[0] || bj_74 == lk_1[0] && lane_1 > 21);
            unsigned int bj_75 = __shfl_sync(0xFFFFFFFF, lk_1[1], 21);
            rk_1[1] = rk_1[1] + (int)(bj_75 > lk_1[1] || bj_75 == lk_1[1] && lane_1 > 21);
            unsigned int bj_76 = __shfl_sync(0xFFFFFFFF, lk_1[0], 22);
            rk_1[0] = rk_1[0] + (int)(bj_76 > lk_1[0] || bj_76 == lk_1[0] && lane_1 > 22);
            unsigned int bj_77 = __shfl_sync(0xFFFFFFFF, lk_1[1], 22);
            rk_1[1] = rk_1[1] + (int)(bj_77 > lk_1[1] || bj_77 == lk_1[1] && lane_1 > 22);
            unsigned int bj_78 = __shfl_sync(0xFFFFFFFF, lk_1[0], 23);
            rk_1[0] = rk_1[0] + (int)(bj_78 > lk_1[0] || bj_78 == lk_1[0] && lane_1 > 23);
            unsigned int bj_79 = __shfl_sync(0xFFFFFFFF, lk_1[1], 23);
            rk_1[1] = rk_1[1] + (int)(bj_79 > lk_1[1] || bj_79 == lk_1[1] && lane_1 > 23);
            unsigned int bj_80 = __shfl_sync(0xFFFFFFFF, lk_1[0], 24);
            rk_1[0] = rk_1[0] + (int)(bj_80 > lk_1[0] || bj_80 == lk_1[0] && lane_1 > 24);
            unsigned int bj_81 = __shfl_sync(0xFFFFFFFF, lk_1[1], 24);
            rk_1[1] = rk_1[1] + (int)(bj_81 > lk_1[1] || bj_81 == lk_1[1] && lane_1 > 24);
            unsigned int bj_82 = __shfl_sync(0xFFFFFFFF, lk_1[0], 25);
            rk_1[0] = rk_1[0] + (int)(bj_82 > lk_1[0] || bj_82 == lk_1[0] && lane_1 > 25);
            unsigned int bj_83 = __shfl_sync(0xFFFFFFFF, lk_1[1], 25);
            rk_1[1] = rk_1[1] + (int)(bj_83 > lk_1[1] || bj_83 == lk_1[1] && lane_1 > 25);
            unsigned int bj_84 = __shfl_sync(0xFFFFFFFF, lk_1[0], 26);
            rk_1[0] = rk_1[0] + (int)(bj_84 > lk_1[0] || bj_84 == lk_1[0] && lane_1 > 26);
            unsigned int bj_85 = __shfl_sync(0xFFFFFFFF, lk_1[1], 26);
            rk_1[1] = rk_1[1] + (int)(bj_85 > lk_1[1] || bj_85 == lk_1[1] && lane_1 > 26);
            unsigned int bj_86 = __shfl_sync(0xFFFFFFFF, lk_1[0], 27);
            rk_1[0] = rk_1[0] + (int)(bj_86 > lk_1[0] || bj_86 == lk_1[0] && lane_1 > 27);
            unsigned int bj_87 = __shfl_sync(0xFFFFFFFF, lk_1[1], 27);
            rk_1[1] = rk_1[1] + (int)(bj_87 > lk_1[1] || bj_87 == lk_1[1] && lane_1 > 27);
            unsigned int bj_88 = __shfl_sync(0xFFFFFFFF, lk_1[0], 28);
            rk_1[0] = rk_1[0] + (int)(bj_88 > lk_1[0] || bj_88 == lk_1[0] && lane_1 > 28);
            unsigned int bj_89 = __shfl_sync(0xFFFFFFFF, lk_1[1], 28);
            rk_1[1] = rk_1[1] + (int)(bj_89 > lk_1[1] || bj_89 == lk_1[1] && lane_1 > 28);
            unsigned int bj_90 = __shfl_sync(0xFFFFFFFF, lk_1[0], 29);
            rk_1[0] = rk_1[0] + (int)(bj_90 > lk_1[0] || bj_90 == lk_1[0] && lane_1 > 29);
            unsigned int bj_91 = __shfl_sync(0xFFFFFFFF, lk_1[1], 29);
            rk_1[1] = rk_1[1] + (int)(bj_91 > lk_1[1] || bj_91 == lk_1[1] && lane_1 > 29);
            unsigned int bj_92 = __shfl_sync(0xFFFFFFFF, lk_1[0], 30);
            rk_1[0] = rk_1[0] + (int)(bj_92 > lk_1[0] || bj_92 == lk_1[0] && lane_1 > 30);
            unsigned int bj_93 = __shfl_sync(0xFFFFFFFF, lk_1[1], 30);
            rk_1[1] = rk_1[1] + (int)(bj_93 > lk_1[1] || bj_93 == lk_1[1] && lane_1 > 30);
            unsigned int bj_94 = __shfl_sync(0xFFFFFFFF, lk_1[0], 31);
            rk_1[0] = rk_1[0] + (int)(bj_94 > lk_1[0] || bj_94 == lk_1[0] && lane_1 > 31);
            unsigned int bj_95 = __shfl_sync(0xFFFFFFFF, lk_1[1], 31);
            rk_1[1] = rk_1[1] + (int)(bj_95 > lk_1[1] || bj_95 == lk_1[1] && lane_1 > 31);
            int pick_1[2];
            int wbase = (int)warp * 64;
            ranked[wbase + (rk_1[0] & 31)] = lk_1[0];
            ranked[wbase + 32 + (rk_1[1] & 31)] = lk_1[1];
            __syncwarp();
            int at_0 = 15 - rk_1[0];
            at_0 = ((at_0 > 0) ? at_0 : 0);
            unsigned int other_0 = ranked[wbase + 32 + at_0];
            int beats_0 = (int)(other_0 <= lk_1[0]);
            pick_1[0] = (int)(lead_1[0] != 0 && rk_1[0] < 16 && beats_0 != 0);
            int at_1 = 15 - rk_1[1];
            at_1 = ((at_1 > 0) ? at_1 : 0);
            unsigned int other_1 = ranked[wbase + at_1];
            int beats_1 = (int)(other_1 < lk_1[1]);
            pick_1[1] = (int)(lead_1[1] != 0 && rk_1[1] < 16 && beats_1 != 0);
            long long dst_1 = (long long)t * (long long)out_stride_t + (long long)h * (long long)out_stride_h;
            int taken_1 = 0;
            int sel_1 = pick_1[0];
            unsigned int chosen_1 = __ballot_sync(0xFFFFFFFF, sel_1 != 0);
            int _rank_in_mask_1 = (int)__popc(chosen_1 & ((1u << (unsigned int)lane) - 1u));
            int pos_1 = taken_1 + (int)_rank_in_mask_1;
            if (sel_1 != 0) {
                output[dst_1 + (long long)pos_1] = li_1[0];
            }
            taken_1 = taken_1 + (int)(unsigned int)__popc(chosen_1);
            int sel_2 = pick_1[1];
            unsigned int chosen_2 = __ballot_sync(0xFFFFFFFF, sel_2 != 0);
            int _rank_in_mask_2 = (int)__popc(chosen_2 & ((1u << (unsigned int)lane) - 1u));
            int pos_2 = taken_1 + (int)_rank_in_mask_2;
            if (sel_2 != 0) {
                output[dst_1 + (long long)pos_2] = li_1[1];
            }
            taken_1 = taken_1 + (int)(unsigned int)__popc(chosen_2);
            if (lane_1 >= taken_1 && lane_1 < 16) {
                int pad_1 = -1;
                if (mode == 0 && nv_1 == 0 && lane_1 == 0) {
                    pad_1 = 0;
                }
                output[dst_1 + (long long)lane_1] = pad_1;
            }
            if (mode == 1) {
                __syncwarp();
                if (lane_1 < world) {
                    asm volatile("red.release.sys.global.and.b32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(recv + (row_1 + (long long)lane_1 * (long long)recv_stride_src + 32)))), "r"(static_cast<unsigned int>(0)) : "memory");
                }
            }
            if (mode == 2) {
                __syncwarp();
                if (lane_1 == 0) {
                    expected[counter_1] = (int)gen_1;
                }
            }
        }
    }
}

} // extern "C"
