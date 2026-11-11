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

#define LAUNCH_MIN_BLOCKS 20

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(32, LAUNCH_MIN_BLOCKS) void
kernel_cake_msa_icp_indexer_d2b97d127516aab42ec9(unsigned long long* __restrict__ pool, int* __restrict__ qo_indptr, long long* __restrict__ work_info, long long* __restrict__ work_range, unsigned long long* __restrict__ recv_0, unsigned long long* __restrict__ recv_1, unsigned long long* __restrict__ recv_2, unsigned long long* __restrict__ recv_3, int* __restrict__ generations, int total_q, int batch, int num_ctas, long long pool_piece_stride, int t_base, int slot, int rank, int world, int heads_per_rank, int capacity)
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
    int* stage = reinterpret_cast<int*>(smem_raw + 0);
    const int stage_addr = smem + 0;

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // === Task calls (declaration order) ===
    int lane_1 = (int)lane;
    int h = lane_1 >> 3;
    int sl = lane_1 & 7;
    int t = (int)blockIdx.x * 4 + (int)blockIdx.y + (int)warp;
    if (t < total_q) {
        unsigned long long R[2];
        R[0] = 0;
        R[1] = 0;
        int small = (int)(batch < 32 && num_ctas <= 256);
        int qv = 0;
        long long WR[8];
        WR[0] = 0;
        WR[1] = 0;
        WR[2] = 0;
        WR[3] = 0;
        WR[4] = 0;
        WR[5] = 0;
        WR[6] = 0;
        WR[7] = 0;
        if (small != 0) {
            if (lane_1 <= batch) {
                qv = qo_indptr[lane_1];
            }
            if (lane_1 < num_ctas) {
                WR[0] = work_range[lane_1];
            }
            if (32 + lane_1 < num_ctas) {
                WR[1] = work_range[32 + lane_1];
            }
            if (64 + lane_1 < num_ctas) {
                WR[2] = work_range[64 + lane_1];
            }
            if (96 + lane_1 < num_ctas) {
                WR[3] = work_range[96 + lane_1];
            }
            if (128 + lane_1 < num_ctas) {
                WR[4] = work_range[128 + lane_1];
            }
            if (160 + lane_1 < num_ctas) {
                WR[5] = work_range[160 + lane_1];
            }
            if (192 + lane_1 < num_ctas) {
                WR[6] = work_range[192 + lane_1];
            }
            if (224 + lane_1 < num_ctas) {
                WR[7] = work_range[224 + lane_1];
            }
        }
        int n_le = 0;
        if (small != 0) {
            unsigned int _warp_vote_0 = __ballot_sync(0xFFFFFFFF, lane_1 <= batch && qv <= t);
            n_le = (int)(unsigned int)__popc(_warp_vote_0);
        } else {
            int cl_rng[2];
            cl_rng[0] = 0;
            cl_rng[1] = batch + 1;
            if (cl_rng[0] < cl_rng[1]) {
                int cl_step_0 = (cl_rng[1] - cl_rng[0] + 31) / 32;
                int cl_p_0 = cl_rng[0] + lane_1 * cl_step_0;
                int cl_ok_0 = 0;
                if (cl_p_0 < cl_rng[1]) {
                    cl_ok_0 = (int)(qo_indptr[cl_p_0] <= t);
                }
                unsigned int nw_vote = __ballot_sync(0xFFFFFFFF, cl_ok_0 != 0);
                int nw_cnt = (int)(unsigned int)__popc(nw_vote);
                int nw_lo = ((nw_cnt > 0) ? cl_rng[0] + (nw_cnt - 1) * cl_step_0 + 1 : cl_rng[0]);
                int _min_1 = ((cl_rng[0] + nw_cnt * cl_step_0) < (cl_rng[1]) ? (cl_rng[0] + nw_cnt * cl_step_0) : (cl_rng[1]));
                int nw_hi = ((nw_cnt > 0) ? _min_1 : cl_rng[0]);
                cl_rng[0] = nw_lo;
                cl_rng[1] = nw_hi;
            }
            if (cl_rng[0] < cl_rng[1]) {
                int cl_step_1 = (cl_rng[1] - cl_rng[0] + 31) / 32;
                int cl_p_1 = cl_rng[0] + lane_1 * cl_step_1;
                int cl_ok_1 = 0;
                if (cl_p_1 < cl_rng[1]) {
                    cl_ok_1 = (int)(qo_indptr[cl_p_1] <= t);
                }
                unsigned int nw_vote_1 = __ballot_sync(0xFFFFFFFF, cl_ok_1 != 0);
                int nw_cnt_1 = (int)(unsigned int)__popc(nw_vote_1);
                int nw_lo_1 = ((nw_cnt_1 > 0) ? cl_rng[0] + (nw_cnt_1 - 1) * cl_step_1 + 1 : cl_rng[0]);
                int _min_2 = ((cl_rng[0] + nw_cnt_1 * cl_step_1) < (cl_rng[1]) ? (cl_rng[0] + nw_cnt_1 * cl_step_1) : (cl_rng[1]));
                int nw_hi_1 = ((nw_cnt_1 > 0) ? _min_2 : cl_rng[0]);
                cl_rng[0] = nw_lo_1;
                cl_rng[1] = nw_hi_1;
            }
            if (cl_rng[0] < cl_rng[1]) {
                int cl_step_2 = (cl_rng[1] - cl_rng[0] + 31) / 32;
                int cl_p_2 = cl_rng[0] + lane_1 * cl_step_2;
                int cl_ok_2 = 0;
                if (cl_p_2 < cl_rng[1]) {
                    cl_ok_2 = (int)(qo_indptr[cl_p_2] <= t);
                }
                unsigned int nw_vote_2 = __ballot_sync(0xFFFFFFFF, cl_ok_2 != 0);
                int nw_cnt_2 = (int)(unsigned int)__popc(nw_vote_2);
                int nw_lo_2 = ((nw_cnt_2 > 0) ? cl_rng[0] + (nw_cnt_2 - 1) * cl_step_2 + 1 : cl_rng[0]);
                int _min_3 = ((cl_rng[0] + nw_cnt_2 * cl_step_2) < (cl_rng[1]) ? (cl_rng[0] + nw_cnt_2 * cl_step_2) : (cl_rng[1]));
                int nw_hi_2 = ((nw_cnt_2 > 0) ? _min_3 : cl_rng[0]);
                cl_rng[0] = nw_lo_2;
                cl_rng[1] = nw_hi_2;
            }
            if (cl_rng[0] < cl_rng[1]) {
                int cl_step_3 = (cl_rng[1] - cl_rng[0] + 31) / 32;
                int cl_p_3 = cl_rng[0] + lane_1 * cl_step_3;
                int cl_ok_3 = 0;
                if (cl_p_3 < cl_rng[1]) {
                    cl_ok_3 = (int)(qo_indptr[cl_p_3] <= t);
                }
                unsigned int nw_vote_3 = __ballot_sync(0xFFFFFFFF, cl_ok_3 != 0);
                int nw_cnt_3 = (int)(unsigned int)__popc(nw_vote_3);
                int nw_lo_3 = ((nw_cnt_3 > 0) ? cl_rng[0] + (nw_cnt_3 - 1) * cl_step_3 + 1 : cl_rng[0]);
                int _min_4 = ((cl_rng[0] + nw_cnt_3 * cl_step_3) < (cl_rng[1]) ? (cl_rng[0] + nw_cnt_3 * cl_step_3) : (cl_rng[1]));
                int nw_hi_3 = ((nw_cnt_3 > 0) ? _min_4 : cl_rng[0]);
                cl_rng[0] = nw_lo_3;
                cl_rng[1] = nw_hi_3;
            }
            int cl_res = cl_rng[0];
            n_le = cl_res;
        }
        if (n_le > 0 && n_le <= batch) {
            int b = n_le - 1;
            int seg0 = 0;
            int seg_b = 0;
            if (small != 0) {
                int _shfl_sync_5 = __shfl_sync(0xFFFFFFFF, qv, 0);
                seg0 = _shfl_sync_5;
                int _shfl_sync_6 = __shfl_sync(0xFFFFFFFF, qv, b);
                seg_b = _shfl_sync_6;
            } else {
                seg0 = *qo_indptr;
                seg_b = qo_indptr[b];
            }
            int rs = t - seg_b;
            int it = rs >> 6;
            int r = rs & 63;
            long long last_wr = work_range[num_ctas - 1];
            int n_items = (int)(last_wr >> 16 & 65535);
            int qb = seg_b - seg0;
            int jlo = (qb + 64 - 1 >> 6) + it;
            int jhi = (qb + 63 * b >> 6) + it;
            jhi = ((jhi < n_items - 1) ? jhi : n_items - 1);
            int cp_rng[2];
            cp_rng[0] = jlo;
            cp_rng[1] = jlo + (jhi + 1 - jlo);
            if (cp_rng[0] < cp_rng[1]) {
                int cp_step_0 = (cp_rng[1] - cp_rng[0] + 31) / 32;
                int cp_p_0 = cp_rng[0] + lane_1 * cp_step_0;
                int cp_ok_0 = 0;
                if (cp_p_0 < cp_rng[1]) {
                    long long cp_v_0 = work_info[cp_p_0];
                    cp_ok_0 = (int)(((cp_v_0 & 65535) << 16 | cp_v_0 >> 16 & 65535) <= ((long long)b << 16 | (long long)it));
                }
                unsigned int nw_vote_4 = __ballot_sync(0xFFFFFFFF, cp_ok_0 != 0);
                int nw_cnt_4 = (int)(unsigned int)__popc(nw_vote_4);
                int nw_lo_4 = ((nw_cnt_4 > 0) ? cp_rng[0] + (nw_cnt_4 - 1) * cp_step_0 + 1 : cp_rng[0]);
                int _min_7 = ((cp_rng[0] + nw_cnt_4 * cp_step_0) < (cp_rng[1]) ? (cp_rng[0] + nw_cnt_4 * cp_step_0) : (cp_rng[1]));
                int nw_hi_4 = ((nw_cnt_4 > 0) ? _min_7 : cp_rng[0]);
                cp_rng[0] = nw_lo_4;
                cp_rng[1] = nw_hi_4;
            }
            if (cp_rng[0] < cp_rng[1]) {
                int cp_step_1 = (cp_rng[1] - cp_rng[0] + 31) / 32;
                int cp_p_1 = cp_rng[0] + lane_1 * cp_step_1;
                int cp_ok_1 = 0;
                if (cp_p_1 < cp_rng[1]) {
                    long long cp_v_1 = work_info[cp_p_1];
                    cp_ok_1 = (int)(((cp_v_1 & 65535) << 16 | cp_v_1 >> 16 & 65535) <= ((long long)b << 16 | (long long)it));
                }
                unsigned int nw_vote_5 = __ballot_sync(0xFFFFFFFF, cp_ok_1 != 0);
                int nw_cnt_5 = (int)(unsigned int)__popc(nw_vote_5);
                int nw_lo_5 = ((nw_cnt_5 > 0) ? cp_rng[0] + (nw_cnt_5 - 1) * cp_step_1 + 1 : cp_rng[0]);
                int _min_8 = ((cp_rng[0] + nw_cnt_5 * cp_step_1) < (cp_rng[1]) ? (cp_rng[0] + nw_cnt_5 * cp_step_1) : (cp_rng[1]));
                int nw_hi_5 = ((nw_cnt_5 > 0) ? _min_8 : cp_rng[0]);
                cp_rng[0] = nw_lo_5;
                cp_rng[1] = nw_hi_5;
            }
            if (cp_rng[0] < cp_rng[1]) {
                int cp_step_2 = (cp_rng[1] - cp_rng[0] + 31) / 32;
                int cp_p_2 = cp_rng[0] + lane_1 * cp_step_2;
                int cp_ok_2 = 0;
                if (cp_p_2 < cp_rng[1]) {
                    long long cp_v_2 = work_info[cp_p_2];
                    cp_ok_2 = (int)(((cp_v_2 & 65535) << 16 | cp_v_2 >> 16 & 65535) <= ((long long)b << 16 | (long long)it));
                }
                unsigned int nw_vote_6 = __ballot_sync(0xFFFFFFFF, cp_ok_2 != 0);
                int nw_cnt_6 = (int)(unsigned int)__popc(nw_vote_6);
                int nw_lo_6 = ((nw_cnt_6 > 0) ? cp_rng[0] + (nw_cnt_6 - 1) * cp_step_2 + 1 : cp_rng[0]);
                int _min_9 = ((cp_rng[0] + nw_cnt_6 * cp_step_2) < (cp_rng[1]) ? (cp_rng[0] + nw_cnt_6 * cp_step_2) : (cp_rng[1]));
                int nw_hi_6 = ((nw_cnt_6 > 0) ? _min_9 : cp_rng[0]);
                cp_rng[0] = nw_lo_6;
                cp_rng[1] = nw_hi_6;
            }
            if (cp_rng[0] < cp_rng[1]) {
                int cp_step_3 = (cp_rng[1] - cp_rng[0] + 31) / 32;
                int cp_p_3 = cp_rng[0] + lane_1 * cp_step_3;
                int cp_ok_3 = 0;
                if (cp_p_3 < cp_rng[1]) {
                    long long cp_v_3 = work_info[cp_p_3];
                    cp_ok_3 = (int)(((cp_v_3 & 65535) << 16 | cp_v_3 >> 16 & 65535) <= ((long long)b << 16 | (long long)it));
                }
                unsigned int nw_vote_7 = __ballot_sync(0xFFFFFFFF, cp_ok_3 != 0);
                int nw_cnt_7 = (int)(unsigned int)__popc(nw_vote_7);
                int nw_lo_7 = ((nw_cnt_7 > 0) ? cp_rng[0] + (nw_cnt_7 - 1) * cp_step_3 + 1 : cp_rng[0]);
                int _min_10 = ((cp_rng[0] + nw_cnt_7 * cp_step_3) < (cp_rng[1]) ? (cp_rng[0] + nw_cnt_7 * cp_step_3) : (cp_rng[1]));
                int nw_hi_7 = ((nw_cnt_7 > 0) ? _min_10 : cp_rng[0]);
                cp_rng[0] = nw_lo_7;
                cp_rng[1] = nw_hi_7;
            }
            int cp_res = cp_rng[0];
            int j = cp_res - 1;
            long long wi = work_info[j];
            long long prev = 0;
            if (j > 0) {
                prev = work_info[j - 1] >> 32;
            }
            int cj = (int)((wi >> 32) - prev);
            long long before_next = (long long)j << 16 | 65535;
            int first_count = 0;
            int last_count = 0;
            if (small != 0) {
                int wr_in_0 = (int)(lane_1 < num_ctas);
                long long wr_p_0 = (WR[0] & 65535) << 16 | WR[0] >> 32 & 65535;
                unsigned int _warp_vote_11 = __ballot_sync(0xFFFFFFFF, wr_in_0 != 0 && wr_p_0 <= (long long)j << 16);
                first_count = first_count + (int)(unsigned int)__popc(_warp_vote_11);
                unsigned int _warp_vote_12 = __ballot_sync(0xFFFFFFFF, wr_in_0 != 0 && wr_p_0 <= before_next);
                last_count = last_count + (int)(unsigned int)__popc(_warp_vote_12);
                int wr_in_1 = (int)(32 + lane_1 < num_ctas);
                long long wr_p_1 = (WR[1] & 65535) << 16 | WR[1] >> 32 & 65535;
                unsigned int _warp_vote_13 = __ballot_sync(0xFFFFFFFF, wr_in_1 != 0 && wr_p_1 <= (long long)j << 16);
                first_count = first_count + (int)(unsigned int)__popc(_warp_vote_13);
                unsigned int _warp_vote_14 = __ballot_sync(0xFFFFFFFF, wr_in_1 != 0 && wr_p_1 <= before_next);
                last_count = last_count + (int)(unsigned int)__popc(_warp_vote_14);
                int wr_in_2 = (int)(64 + lane_1 < num_ctas);
                long long wr_p_2 = (WR[2] & 65535) << 16 | WR[2] >> 32 & 65535;
                unsigned int _warp_vote_15 = __ballot_sync(0xFFFFFFFF, wr_in_2 != 0 && wr_p_2 <= (long long)j << 16);
                first_count = first_count + (int)(unsigned int)__popc(_warp_vote_15);
                unsigned int _warp_vote_16 = __ballot_sync(0xFFFFFFFF, wr_in_2 != 0 && wr_p_2 <= before_next);
                last_count = last_count + (int)(unsigned int)__popc(_warp_vote_16);
                int wr_in_3 = (int)(96 + lane_1 < num_ctas);
                long long wr_p_3 = (WR[3] & 65535) << 16 | WR[3] >> 32 & 65535;
                unsigned int _warp_vote_17 = __ballot_sync(0xFFFFFFFF, wr_in_3 != 0 && wr_p_3 <= (long long)j << 16);
                first_count = first_count + (int)(unsigned int)__popc(_warp_vote_17);
                unsigned int _warp_vote_18 = __ballot_sync(0xFFFFFFFF, wr_in_3 != 0 && wr_p_3 <= before_next);
                last_count = last_count + (int)(unsigned int)__popc(_warp_vote_18);
                int wr_in_4 = (int)(128 + lane_1 < num_ctas);
                long long wr_p_4 = (WR[4] & 65535) << 16 | WR[4] >> 32 & 65535;
                unsigned int _warp_vote_19 = __ballot_sync(0xFFFFFFFF, wr_in_4 != 0 && wr_p_4 <= (long long)j << 16);
                first_count = first_count + (int)(unsigned int)__popc(_warp_vote_19);
                unsigned int _warp_vote_20 = __ballot_sync(0xFFFFFFFF, wr_in_4 != 0 && wr_p_4 <= before_next);
                last_count = last_count + (int)(unsigned int)__popc(_warp_vote_20);
                int wr_in_5 = (int)(160 + lane_1 < num_ctas);
                long long wr_p_5 = (WR[5] & 65535) << 16 | WR[5] >> 32 & 65535;
                unsigned int _warp_vote_21 = __ballot_sync(0xFFFFFFFF, wr_in_5 != 0 && wr_p_5 <= (long long)j << 16);
                first_count = first_count + (int)(unsigned int)__popc(_warp_vote_21);
                unsigned int _warp_vote_22 = __ballot_sync(0xFFFFFFFF, wr_in_5 != 0 && wr_p_5 <= before_next);
                last_count = last_count + (int)(unsigned int)__popc(_warp_vote_22);
                int wr_in_6 = (int)(192 + lane_1 < num_ctas);
                long long wr_p_6 = (WR[6] & 65535) << 16 | WR[6] >> 32 & 65535;
                unsigned int _warp_vote_23 = __ballot_sync(0xFFFFFFFF, wr_in_6 != 0 && wr_p_6 <= (long long)j << 16);
                first_count = first_count + (int)(unsigned int)__popc(_warp_vote_23);
                unsigned int _warp_vote_24 = __ballot_sync(0xFFFFFFFF, wr_in_6 != 0 && wr_p_6 <= before_next);
                last_count = last_count + (int)(unsigned int)__popc(_warp_vote_24);
                int wr_in_7 = (int)(224 + lane_1 < num_ctas);
                long long wr_p_7 = (WR[7] & 65535) << 16 | WR[7] >> 32 & 65535;
                unsigned int _warp_vote_25 = __ballot_sync(0xFFFFFFFF, wr_in_7 != 0 && wr_p_7 <= (long long)j << 16);
                first_count = first_count + (int)(unsigned int)__popc(_warp_vote_25);
                unsigned int _warp_vote_26 = __ballot_sync(0xFFFFFFFF, wr_in_7 != 0 && wr_p_7 <= before_next);
                last_count = last_count + (int)(unsigned int)__popc(_warp_vote_26);
            } else {
                int cp_rng_0[2];
                cp_rng_0[0] = 0;
                cp_rng_0[1] = num_ctas;
                if (cp_rng_0[0] < cp_rng_0[1]) {
                    int cp_step_4 = (cp_rng_0[1] - cp_rng_0[0] + 31) / 32;
                    int cp_p_4 = cp_rng_0[0] + lane_1 * cp_step_4;
                    int cp_ok_4 = 0;
                    if (cp_p_4 < cp_rng_0[1]) {
                        long long cp_v_4 = work_range[cp_p_4];
                        cp_ok_4 = (int)(((cp_v_4 & 65535) << 16 | cp_v_4 >> 32 & 65535) <= (long long)j << 16);
                    }
                    unsigned int nw_vote_8 = __ballot_sync(0xFFFFFFFF, cp_ok_4 != 0);
                    int nw_cnt_8 = (int)(unsigned int)__popc(nw_vote_8);
                    int nw_lo_8 = ((nw_cnt_8 > 0) ? cp_rng_0[0] + (nw_cnt_8 - 1) * cp_step_4 + 1 : cp_rng_0[0]);
                    int _min_27 = ((cp_rng_0[0] + nw_cnt_8 * cp_step_4) < (cp_rng_0[1]) ? (cp_rng_0[0] + nw_cnt_8 * cp_step_4) : (cp_rng_0[1]));
                    int nw_hi_8 = ((nw_cnt_8 > 0) ? _min_27 : cp_rng_0[0]);
                    cp_rng_0[0] = nw_lo_8;
                    cp_rng_0[1] = nw_hi_8;
                }
                if (cp_rng_0[0] < cp_rng_0[1]) {
                    int cp_step_5 = (cp_rng_0[1] - cp_rng_0[0] + 31) / 32;
                    int cp_p_5 = cp_rng_0[0] + lane_1 * cp_step_5;
                    int cp_ok_5 = 0;
                    if (cp_p_5 < cp_rng_0[1]) {
                        long long cp_v_5 = work_range[cp_p_5];
                        cp_ok_5 = (int)(((cp_v_5 & 65535) << 16 | cp_v_5 >> 32 & 65535) <= (long long)j << 16);
                    }
                    unsigned int nw_vote_9 = __ballot_sync(0xFFFFFFFF, cp_ok_5 != 0);
                    int nw_cnt_9 = (int)(unsigned int)__popc(nw_vote_9);
                    int nw_lo_9 = ((nw_cnt_9 > 0) ? cp_rng_0[0] + (nw_cnt_9 - 1) * cp_step_5 + 1 : cp_rng_0[0]);
                    int _min_28 = ((cp_rng_0[0] + nw_cnt_9 * cp_step_5) < (cp_rng_0[1]) ? (cp_rng_0[0] + nw_cnt_9 * cp_step_5) : (cp_rng_0[1]));
                    int nw_hi_9 = ((nw_cnt_9 > 0) ? _min_28 : cp_rng_0[0]);
                    cp_rng_0[0] = nw_lo_9;
                    cp_rng_0[1] = nw_hi_9;
                }
                if (cp_rng_0[0] < cp_rng_0[1]) {
                    int cp_step_6 = (cp_rng_0[1] - cp_rng_0[0] + 31) / 32;
                    int cp_p_6 = cp_rng_0[0] + lane_1 * cp_step_6;
                    int cp_ok_6 = 0;
                    if (cp_p_6 < cp_rng_0[1]) {
                        long long cp_v_6 = work_range[cp_p_6];
                        cp_ok_6 = (int)(((cp_v_6 & 65535) << 16 | cp_v_6 >> 32 & 65535) <= (long long)j << 16);
                    }
                    unsigned int nw_vote_10 = __ballot_sync(0xFFFFFFFF, cp_ok_6 != 0);
                    int nw_cnt_10 = (int)(unsigned int)__popc(nw_vote_10);
                    int nw_lo_10 = ((nw_cnt_10 > 0) ? cp_rng_0[0] + (nw_cnt_10 - 1) * cp_step_6 + 1 : cp_rng_0[0]);
                    int _min_29 = ((cp_rng_0[0] + nw_cnt_10 * cp_step_6) < (cp_rng_0[1]) ? (cp_rng_0[0] + nw_cnt_10 * cp_step_6) : (cp_rng_0[1]));
                    int nw_hi_10 = ((nw_cnt_10 > 0) ? _min_29 : cp_rng_0[0]);
                    cp_rng_0[0] = nw_lo_10;
                    cp_rng_0[1] = nw_hi_10;
                }
                if (cp_rng_0[0] < cp_rng_0[1]) {
                    int cp_step_7 = (cp_rng_0[1] - cp_rng_0[0] + 31) / 32;
                    int cp_p_7 = cp_rng_0[0] + lane_1 * cp_step_7;
                    int cp_ok_7 = 0;
                    if (cp_p_7 < cp_rng_0[1]) {
                        long long cp_v_7 = work_range[cp_p_7];
                        cp_ok_7 = (int)(((cp_v_7 & 65535) << 16 | cp_v_7 >> 32 & 65535) <= (long long)j << 16);
                    }
                    unsigned int nw_vote_11 = __ballot_sync(0xFFFFFFFF, cp_ok_7 != 0);
                    int nw_cnt_11 = (int)(unsigned int)__popc(nw_vote_11);
                    int nw_lo_11 = ((nw_cnt_11 > 0) ? cp_rng_0[0] + (nw_cnt_11 - 1) * cp_step_7 + 1 : cp_rng_0[0]);
                    int _min_30 = ((cp_rng_0[0] + nw_cnt_11 * cp_step_7) < (cp_rng_0[1]) ? (cp_rng_0[0] + nw_cnt_11 * cp_step_7) : (cp_rng_0[1]));
                    int nw_hi_11 = ((nw_cnt_11 > 0) ? _min_30 : cp_rng_0[0]);
                    cp_rng_0[0] = nw_lo_11;
                    cp_rng_0[1] = nw_hi_11;
                }
                int cp_res_0 = cp_rng_0[0];
                first_count = cp_res_0;
                int cp_rng_1[2];
                cp_rng_1[0] = cp_res_0 - 1;
                cp_rng_1[1] = cp_res_0 - 1 + (num_ctas - (cp_res_0 - 1));
                if (cp_rng_1[0] < cp_rng_1[1]) {
                    int cp_step_8 = (cp_rng_1[1] - cp_rng_1[0] + 31) / 32;
                    int cp_p_8 = cp_rng_1[0] + lane_1 * cp_step_8;
                    int cp_ok_8 = 0;
                    if (cp_p_8 < cp_rng_1[1]) {
                        long long cp_v_8 = work_range[cp_p_8];
                        cp_ok_8 = (int)(((cp_v_8 & 65535) << 16 | cp_v_8 >> 32 & 65535) <= before_next);
                    }
                    unsigned int nw_vote_12 = __ballot_sync(0xFFFFFFFF, cp_ok_8 != 0);
                    int nw_cnt_12 = (int)(unsigned int)__popc(nw_vote_12);
                    int nw_lo_12 = ((nw_cnt_12 > 0) ? cp_rng_1[0] + (nw_cnt_12 - 1) * cp_step_8 + 1 : cp_rng_1[0]);
                    int _min_31 = ((cp_rng_1[0] + nw_cnt_12 * cp_step_8) < (cp_rng_1[1]) ? (cp_rng_1[0] + nw_cnt_12 * cp_step_8) : (cp_rng_1[1]));
                    int nw_hi_12 = ((nw_cnt_12 > 0) ? _min_31 : cp_rng_1[0]);
                    cp_rng_1[0] = nw_lo_12;
                    cp_rng_1[1] = nw_hi_12;
                }
                if (cp_rng_1[0] < cp_rng_1[1]) {
                    int cp_step_9 = (cp_rng_1[1] - cp_rng_1[0] + 31) / 32;
                    int cp_p_9 = cp_rng_1[0] + lane_1 * cp_step_9;
                    int cp_ok_9 = 0;
                    if (cp_p_9 < cp_rng_1[1]) {
                        long long cp_v_9 = work_range[cp_p_9];
                        cp_ok_9 = (int)(((cp_v_9 & 65535) << 16 | cp_v_9 >> 32 & 65535) <= before_next);
                    }
                    unsigned int nw_vote_13 = __ballot_sync(0xFFFFFFFF, cp_ok_9 != 0);
                    int nw_cnt_13 = (int)(unsigned int)__popc(nw_vote_13);
                    int nw_lo_13 = ((nw_cnt_13 > 0) ? cp_rng_1[0] + (nw_cnt_13 - 1) * cp_step_9 + 1 : cp_rng_1[0]);
                    int _min_32 = ((cp_rng_1[0] + nw_cnt_13 * cp_step_9) < (cp_rng_1[1]) ? (cp_rng_1[0] + nw_cnt_13 * cp_step_9) : (cp_rng_1[1]));
                    int nw_hi_13 = ((nw_cnt_13 > 0) ? _min_32 : cp_rng_1[0]);
                    cp_rng_1[0] = nw_lo_13;
                    cp_rng_1[1] = nw_hi_13;
                }
                if (cp_rng_1[0] < cp_rng_1[1]) {
                    int cp_step_10 = (cp_rng_1[1] - cp_rng_1[0] + 31) / 32;
                    int cp_p_10 = cp_rng_1[0] + lane_1 * cp_step_10;
                    int cp_ok_10 = 0;
                    if (cp_p_10 < cp_rng_1[1]) {
                        long long cp_v_10 = work_range[cp_p_10];
                        cp_ok_10 = (int)(((cp_v_10 & 65535) << 16 | cp_v_10 >> 32 & 65535) <= before_next);
                    }
                    unsigned int nw_vote_14 = __ballot_sync(0xFFFFFFFF, cp_ok_10 != 0);
                    int nw_cnt_14 = (int)(unsigned int)__popc(nw_vote_14);
                    int nw_lo_14 = ((nw_cnt_14 > 0) ? cp_rng_1[0] + (nw_cnt_14 - 1) * cp_step_10 + 1 : cp_rng_1[0]);
                    int _min_33 = ((cp_rng_1[0] + nw_cnt_14 * cp_step_10) < (cp_rng_1[1]) ? (cp_rng_1[0] + nw_cnt_14 * cp_step_10) : (cp_rng_1[1]));
                    int nw_hi_14 = ((nw_cnt_14 > 0) ? _min_33 : cp_rng_1[0]);
                    cp_rng_1[0] = nw_lo_14;
                    cp_rng_1[1] = nw_hi_14;
                }
                if (cp_rng_1[0] < cp_rng_1[1]) {
                    int cp_step_11 = (cp_rng_1[1] - cp_rng_1[0] + 31) / 32;
                    int cp_p_11 = cp_rng_1[0] + lane_1 * cp_step_11;
                    int cp_ok_11 = 0;
                    if (cp_p_11 < cp_rng_1[1]) {
                        long long cp_v_11 = work_range[cp_p_11];
                        cp_ok_11 = (int)(((cp_v_11 & 65535) << 16 | cp_v_11 >> 32 & 65535) <= before_next);
                    }
                    unsigned int nw_vote_15 = __ballot_sync(0xFFFFFFFF, cp_ok_11 != 0);
                    int nw_cnt_15 = (int)(unsigned int)__popc(nw_vote_15);
                    int nw_lo_15 = ((nw_cnt_15 > 0) ? cp_rng_1[0] + (nw_cnt_15 - 1) * cp_step_11 + 1 : cp_rng_1[0]);
                    int _min_34 = ((cp_rng_1[0] + nw_cnt_15 * cp_step_11) < (cp_rng_1[1]) ? (cp_rng_1[0] + nw_cnt_15 * cp_step_11) : (cp_rng_1[1]));
                    int nw_hi_15 = ((nw_cnt_15 > 0) ? _min_34 : cp_rng_1[0]);
                    cp_rng_1[0] = nw_lo_15;
                    cp_rng_1[1] = nw_hi_15;
                }
                int cp_res_1 = cp_rng_1[0];
                last_count = cp_res_1;
            }
            int s_first = first_count - 1;
            int s_last = last_count - 1;
            int npieces = s_last + 1 - s_first;
            long long row_dir = (long long)(r * 4 + h) * 16 + (long long)(2 * sl);
            long long row_rev = (long long)(r * 4 + h) * 16 + (long long)(14 - 2 * sl);
            if (npieces > 16) {
                unsigned long long RL[16];
                unsigned long long BL[16];
                unsigned long long BN[16];
                RL[0] = 0;
                RL[1] = 0;
                RL[2] = 0;
                RL[3] = 0;
                RL[4] = 0;
                RL[5] = 0;
                RL[6] = 0;
                RL[7] = 0;
                RL[8] = 0;
                RL[9] = 0;
                RL[10] = 0;
                RL[11] = 0;
                RL[12] = 0;
                RL[13] = 0;
                RL[14] = 0;
                RL[15] = 0;
                long long row_all = (long long)(r * 4 + h) * 16;
                unsigned long long top1 = 0;
                unsigned long long top2 = 0;
                #pragma unroll 1
                for (int g0 = 0; g0 < npieces; g0 += 256) {
                    unsigned int mine = 0;
                    int ws_0 = s_first + g0 + lane_1;
                    long long ww_0 = work_range[((ws_0 <= s_last) ? ws_0 : s_last)];
                    int w_cb_0 = (((int)(ww_0 & 65535) == j) ? (int)(ww_0 >> 32 & 65535) : 0);
                    int w_hi_0 = (int)(ww_0 >> 16 & 65535);
                    int w_ce_0 = ((w_hi_0 == j) ? (int)(ww_0 >> 48 & 65535) : cj);
                    unsigned int w_m_0 = __ballot_sync(0xFFFFFFFF, ws_0 <= s_last && w_ce_0 > w_cb_0);
                    mine = mine | (w_m_0 >> (unsigned int)sl & 1) << 0;
                    mine = mine | (w_m_0 >> (unsigned int)(sl + 8) & 1) << 1;
                    mine = mine | (w_m_0 >> (unsigned int)(sl + 16) & 1) << 2;
                    mine = mine | (w_m_0 >> (unsigned int)(sl + 24) & 1) << 3;
                    int ws_1 = s_first + g0 + 32 + lane_1;
                    long long ww_1 = work_range[((ws_1 <= s_last) ? ws_1 : s_last)];
                    int w_cb_1 = (((int)(ww_1 & 65535) == j) ? (int)(ww_1 >> 32 & 65535) : 0);
                    int w_hi_1 = (int)(ww_1 >> 16 & 65535);
                    int w_ce_1 = ((w_hi_1 == j) ? (int)(ww_1 >> 48 & 65535) : cj);
                    unsigned int w_m_1 = __ballot_sync(0xFFFFFFFF, ws_1 <= s_last && w_ce_1 > w_cb_1);
                    mine = mine | (w_m_1 >> (unsigned int)sl & 1) << 4;
                    mine = mine | (w_m_1 >> (unsigned int)(sl + 8) & 1) << 5;
                    mine = mine | (w_m_1 >> (unsigned int)(sl + 16) & 1) << 6;
                    mine = mine | (w_m_1 >> (unsigned int)(sl + 24) & 1) << 7;
                    int ws_2 = s_first + g0 + 64 + lane_1;
                    long long ww_2 = work_range[((ws_2 <= s_last) ? ws_2 : s_last)];
                    int w_cb_2 = (((int)(ww_2 & 65535) == j) ? (int)(ww_2 >> 32 & 65535) : 0);
                    int w_hi_2 = (int)(ww_2 >> 16 & 65535);
                    int w_ce_2 = ((w_hi_2 == j) ? (int)(ww_2 >> 48 & 65535) : cj);
                    unsigned int w_m_2 = __ballot_sync(0xFFFFFFFF, ws_2 <= s_last && w_ce_2 > w_cb_2);
                    mine = mine | (w_m_2 >> (unsigned int)sl & 1) << 8;
                    mine = mine | (w_m_2 >> (unsigned int)(sl + 8) & 1) << 9;
                    mine = mine | (w_m_2 >> (unsigned int)(sl + 16) & 1) << 10;
                    mine = mine | (w_m_2 >> (unsigned int)(sl + 24) & 1) << 11;
                    int ws_3 = s_first + g0 + 96 + lane_1;
                    long long ww_3 = work_range[((ws_3 <= s_last) ? ws_3 : s_last)];
                    int w_cb_3 = (((int)(ww_3 & 65535) == j) ? (int)(ww_3 >> 32 & 65535) : 0);
                    int w_hi_3 = (int)(ww_3 >> 16 & 65535);
                    int w_ce_3 = ((w_hi_3 == j) ? (int)(ww_3 >> 48 & 65535) : cj);
                    unsigned int w_m_3 = __ballot_sync(0xFFFFFFFF, ws_3 <= s_last && w_ce_3 > w_cb_3);
                    mine = mine | (w_m_3 >> (unsigned int)sl & 1) << 12;
                    mine = mine | (w_m_3 >> (unsigned int)(sl + 8) & 1) << 13;
                    mine = mine | (w_m_3 >> (unsigned int)(sl + 16) & 1) << 14;
                    mine = mine | (w_m_3 >> (unsigned int)(sl + 24) & 1) << 15;
                    int ws_4 = s_first + g0 + 128 + lane_1;
                    long long ww_4 = work_range[((ws_4 <= s_last) ? ws_4 : s_last)];
                    int w_cb_4 = (((int)(ww_4 & 65535) == j) ? (int)(ww_4 >> 32 & 65535) : 0);
                    int w_hi_4 = (int)(ww_4 >> 16 & 65535);
                    int w_ce_4 = ((w_hi_4 == j) ? (int)(ww_4 >> 48 & 65535) : cj);
                    unsigned int w_m_4 = __ballot_sync(0xFFFFFFFF, ws_4 <= s_last && w_ce_4 > w_cb_4);
                    mine = mine | (w_m_4 >> (unsigned int)sl & 1) << 16;
                    mine = mine | (w_m_4 >> (unsigned int)(sl + 8) & 1) << 17;
                    mine = mine | (w_m_4 >> (unsigned int)(sl + 16) & 1) << 18;
                    mine = mine | (w_m_4 >> (unsigned int)(sl + 24) & 1) << 19;
                    int ws_5 = s_first + g0 + 160 + lane_1;
                    long long ww_5 = work_range[((ws_5 <= s_last) ? ws_5 : s_last)];
                    int w_cb_5 = (((int)(ww_5 & 65535) == j) ? (int)(ww_5 >> 32 & 65535) : 0);
                    int w_hi_5 = (int)(ww_5 >> 16 & 65535);
                    int w_ce_5 = ((w_hi_5 == j) ? (int)(ww_5 >> 48 & 65535) : cj);
                    unsigned int w_m_5 = __ballot_sync(0xFFFFFFFF, ws_5 <= s_last && w_ce_5 > w_cb_5);
                    mine = mine | (w_m_5 >> (unsigned int)sl & 1) << 20;
                    mine = mine | (w_m_5 >> (unsigned int)(sl + 8) & 1) << 21;
                    mine = mine | (w_m_5 >> (unsigned int)(sl + 16) & 1) << 22;
                    mine = mine | (w_m_5 >> (unsigned int)(sl + 24) & 1) << 23;
                    int ws_6 = s_first + g0 + 192 + lane_1;
                    long long ww_6 = work_range[((ws_6 <= s_last) ? ws_6 : s_last)];
                    int w_cb_6 = (((int)(ww_6 & 65535) == j) ? (int)(ww_6 >> 32 & 65535) : 0);
                    int w_hi_6 = (int)(ww_6 >> 16 & 65535);
                    int w_ce_6 = ((w_hi_6 == j) ? (int)(ww_6 >> 48 & 65535) : cj);
                    unsigned int w_m_6 = __ballot_sync(0xFFFFFFFF, ws_6 <= s_last && w_ce_6 > w_cb_6);
                    mine = mine | (w_m_6 >> (unsigned int)sl & 1) << 24;
                    mine = mine | (w_m_6 >> (unsigned int)(sl + 8) & 1) << 25;
                    mine = mine | (w_m_6 >> (unsigned int)(sl + 16) & 1) << 26;
                    mine = mine | (w_m_6 >> (unsigned int)(sl + 24) & 1) << 27;
                    int ws_7 = s_first + g0 + 224 + lane_1;
                    long long ww_7 = work_range[((ws_7 <= s_last) ? ws_7 : s_last)];
                    int w_cb_7 = (((int)(ww_7 & 65535) == j) ? (int)(ww_7 >> 32 & 65535) : 0);
                    int w_hi_7 = (int)(ww_7 >> 16 & 65535);
                    int w_ce_7 = ((w_hi_7 == j) ? (int)(ww_7 >> 48 & 65535) : cj);
                    unsigned int w_m_7 = __ballot_sync(0xFFFFFFFF, ws_7 <= s_last && w_ce_7 > w_cb_7);
                    mine = mine | (w_m_7 >> (unsigned int)sl & 1) << 28;
                    mine = mine | (w_m_7 >> (unsigned int)(sl + 8) & 1) << 29;
                    mine = mine | (w_m_7 >> (unsigned int)(sl + 16) & 1) << 30;
                    mine = mine | (w_m_7 >> (unsigned int)(sl + 24) & 1) << 31;
                    unsigned int pf = mine;
                    int n_pf = (int)(unsigned int)__popc(pf);
                    #pragma unroll 1
                    for (int _pi = 0; _pi < n_pf; _pi++) {
                        int pb;
                        asm volatile("bfind.u32 %0, %1;" : "=r"(pb) : "r"(pf));
                        pf = pf & ~(1 << (unsigned int)pb);
                        int p_s = s_first + g0 + 32 * (pb >> 2) + 8 * (pb & 3) + sl;
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(pool + ((long long)(p_s + j) * pool_piece_stride + row_all))));
                    }
                    if (g0 < npieces) {
                        if ((mine >> 0 & 1) != 0) {
                            int k_s_0 = s_first + g0 + sl;
                            unsigned long long k_0_0 = pool[(long long)(k_s_0 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_0 = ((top1 < k_0_0) ? top1 : k_0_0);
                            top2 = ((top2 > k_lo_0) ? top2 : k_lo_0);
                            top1 = ((top1 > k_0_0) ? top1 : k_0_0);
                        }
                        if ((mine >> 1 & 1) != 0) {
                            int k_s_1 = s_first + g0 + 8 + sl;
                            unsigned long long k_0_1 = pool[(long long)(k_s_1 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_1 = ((top1 < k_0_1) ? top1 : k_0_1);
                            top2 = ((top2 > k_lo_1) ? top2 : k_lo_1);
                            top1 = ((top1 > k_0_1) ? top1 : k_0_1);
                        }
                        if ((mine >> 2 & 1) != 0) {
                            int k_s_2 = s_first + g0 + 16 + sl;
                            unsigned long long k_0_2 = pool[(long long)(k_s_2 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_2 = ((top1 < k_0_2) ? top1 : k_0_2);
                            top2 = ((top2 > k_lo_2) ? top2 : k_lo_2);
                            top1 = ((top1 > k_0_2) ? top1 : k_0_2);
                        }
                        if ((mine >> 3 & 1) != 0) {
                            int k_s_3 = s_first + g0 + 24 + sl;
                            unsigned long long k_0_3 = pool[(long long)(k_s_3 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_3 = ((top1 < k_0_3) ? top1 : k_0_3);
                            top2 = ((top2 > k_lo_3) ? top2 : k_lo_3);
                            top1 = ((top1 > k_0_3) ? top1 : k_0_3);
                        }
                    }
                    if (g0 + 32 < npieces) {
                        if ((mine >> 4 & 1) != 0) {
                            int k_s_4 = s_first + g0 + 32 + sl;
                            unsigned long long k_0_4 = pool[(long long)(k_s_4 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_4 = ((top1 < k_0_4) ? top1 : k_0_4);
                            top2 = ((top2 > k_lo_4) ? top2 : k_lo_4);
                            top1 = ((top1 > k_0_4) ? top1 : k_0_4);
                        }
                        if ((mine >> 5 & 1) != 0) {
                            int k_s_5 = s_first + g0 + 32 + 8 + sl;
                            unsigned long long k_0_5 = pool[(long long)(k_s_5 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_5 = ((top1 < k_0_5) ? top1 : k_0_5);
                            top2 = ((top2 > k_lo_5) ? top2 : k_lo_5);
                            top1 = ((top1 > k_0_5) ? top1 : k_0_5);
                        }
                        if ((mine >> 6 & 1) != 0) {
                            int k_s_6 = s_first + g0 + 32 + 16 + sl;
                            unsigned long long k_0_6 = pool[(long long)(k_s_6 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_6 = ((top1 < k_0_6) ? top1 : k_0_6);
                            top2 = ((top2 > k_lo_6) ? top2 : k_lo_6);
                            top1 = ((top1 > k_0_6) ? top1 : k_0_6);
                        }
                        if ((mine >> 7 & 1) != 0) {
                            int k_s_7 = s_first + g0 + 32 + 24 + sl;
                            unsigned long long k_0_7 = pool[(long long)(k_s_7 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_7 = ((top1 < k_0_7) ? top1 : k_0_7);
                            top2 = ((top2 > k_lo_7) ? top2 : k_lo_7);
                            top1 = ((top1 > k_0_7) ? top1 : k_0_7);
                        }
                    }
                    if (g0 + 64 < npieces) {
                        if ((mine >> 8 & 1) != 0) {
                            int k_s_8 = s_first + g0 + 64 + sl;
                            unsigned long long k_0_8 = pool[(long long)(k_s_8 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_8 = ((top1 < k_0_8) ? top1 : k_0_8);
                            top2 = ((top2 > k_lo_8) ? top2 : k_lo_8);
                            top1 = ((top1 > k_0_8) ? top1 : k_0_8);
                        }
                        if ((mine >> 9 & 1) != 0) {
                            int k_s_9 = s_first + g0 + 64 + 8 + sl;
                            unsigned long long k_0_9 = pool[(long long)(k_s_9 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_9 = ((top1 < k_0_9) ? top1 : k_0_9);
                            top2 = ((top2 > k_lo_9) ? top2 : k_lo_9);
                            top1 = ((top1 > k_0_9) ? top1 : k_0_9);
                        }
                        if ((mine >> 10 & 1) != 0) {
                            int k_s_10 = s_first + g0 + 64 + 16 + sl;
                            unsigned long long k_0_10 = pool[(long long)(k_s_10 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_10 = ((top1 < k_0_10) ? top1 : k_0_10);
                            top2 = ((top2 > k_lo_10) ? top2 : k_lo_10);
                            top1 = ((top1 > k_0_10) ? top1 : k_0_10);
                        }
                        if ((mine >> 11 & 1) != 0) {
                            int k_s_11 = s_first + g0 + 64 + 24 + sl;
                            unsigned long long k_0_11 = pool[(long long)(k_s_11 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_11 = ((top1 < k_0_11) ? top1 : k_0_11);
                            top2 = ((top2 > k_lo_11) ? top2 : k_lo_11);
                            top1 = ((top1 > k_0_11) ? top1 : k_0_11);
                        }
                    }
                    if (g0 + 96 < npieces) {
                        if ((mine >> 12 & 1) != 0) {
                            int k_s_12 = s_first + g0 + 96 + sl;
                            unsigned long long k_0_12 = pool[(long long)(k_s_12 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_12 = ((top1 < k_0_12) ? top1 : k_0_12);
                            top2 = ((top2 > k_lo_12) ? top2 : k_lo_12);
                            top1 = ((top1 > k_0_12) ? top1 : k_0_12);
                        }
                        if ((mine >> 13 & 1) != 0) {
                            int k_s_13 = s_first + g0 + 96 + 8 + sl;
                            unsigned long long k_0_13 = pool[(long long)(k_s_13 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_13 = ((top1 < k_0_13) ? top1 : k_0_13);
                            top2 = ((top2 > k_lo_13) ? top2 : k_lo_13);
                            top1 = ((top1 > k_0_13) ? top1 : k_0_13);
                        }
                        if ((mine >> 14 & 1) != 0) {
                            int k_s_14 = s_first + g0 + 96 + 16 + sl;
                            unsigned long long k_0_14 = pool[(long long)(k_s_14 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_14 = ((top1 < k_0_14) ? top1 : k_0_14);
                            top2 = ((top2 > k_lo_14) ? top2 : k_lo_14);
                            top1 = ((top1 > k_0_14) ? top1 : k_0_14);
                        }
                        if ((mine >> 15 & 1) != 0) {
                            int k_s_15 = s_first + g0 + 96 + 24 + sl;
                            unsigned long long k_0_15 = pool[(long long)(k_s_15 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_15 = ((top1 < k_0_15) ? top1 : k_0_15);
                            top2 = ((top2 > k_lo_15) ? top2 : k_lo_15);
                            top1 = ((top1 > k_0_15) ? top1 : k_0_15);
                        }
                    }
                    if (g0 + 128 < npieces) {
                        if ((mine >> 16 & 1) != 0) {
                            int k_s_16 = s_first + g0 + 128 + sl;
                            unsigned long long k_0_16 = pool[(long long)(k_s_16 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_16 = ((top1 < k_0_16) ? top1 : k_0_16);
                            top2 = ((top2 > k_lo_16) ? top2 : k_lo_16);
                            top1 = ((top1 > k_0_16) ? top1 : k_0_16);
                        }
                        if ((mine >> 17 & 1) != 0) {
                            int k_s_17 = s_first + g0 + 128 + 8 + sl;
                            unsigned long long k_0_17 = pool[(long long)(k_s_17 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_17 = ((top1 < k_0_17) ? top1 : k_0_17);
                            top2 = ((top2 > k_lo_17) ? top2 : k_lo_17);
                            top1 = ((top1 > k_0_17) ? top1 : k_0_17);
                        }
                        if ((mine >> 18 & 1) != 0) {
                            int k_s_18 = s_first + g0 + 128 + 16 + sl;
                            unsigned long long k_0_18 = pool[(long long)(k_s_18 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_18 = ((top1 < k_0_18) ? top1 : k_0_18);
                            top2 = ((top2 > k_lo_18) ? top2 : k_lo_18);
                            top1 = ((top1 > k_0_18) ? top1 : k_0_18);
                        }
                        if ((mine >> 19 & 1) != 0) {
                            int k_s_19 = s_first + g0 + 128 + 24 + sl;
                            unsigned long long k_0_19 = pool[(long long)(k_s_19 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_19 = ((top1 < k_0_19) ? top1 : k_0_19);
                            top2 = ((top2 > k_lo_19) ? top2 : k_lo_19);
                            top1 = ((top1 > k_0_19) ? top1 : k_0_19);
                        }
                    }
                    if (g0 + 160 < npieces) {
                        if ((mine >> 20 & 1) != 0) {
                            int k_s_20 = s_first + g0 + 160 + sl;
                            unsigned long long k_0_20 = pool[(long long)(k_s_20 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_20 = ((top1 < k_0_20) ? top1 : k_0_20);
                            top2 = ((top2 > k_lo_20) ? top2 : k_lo_20);
                            top1 = ((top1 > k_0_20) ? top1 : k_0_20);
                        }
                        if ((mine >> 21 & 1) != 0) {
                            int k_s_21 = s_first + g0 + 160 + 8 + sl;
                            unsigned long long k_0_21 = pool[(long long)(k_s_21 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_21 = ((top1 < k_0_21) ? top1 : k_0_21);
                            top2 = ((top2 > k_lo_21) ? top2 : k_lo_21);
                            top1 = ((top1 > k_0_21) ? top1 : k_0_21);
                        }
                        if ((mine >> 22 & 1) != 0) {
                            int k_s_22 = s_first + g0 + 160 + 16 + sl;
                            unsigned long long k_0_22 = pool[(long long)(k_s_22 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_22 = ((top1 < k_0_22) ? top1 : k_0_22);
                            top2 = ((top2 > k_lo_22) ? top2 : k_lo_22);
                            top1 = ((top1 > k_0_22) ? top1 : k_0_22);
                        }
                        if ((mine >> 23 & 1) != 0) {
                            int k_s_23 = s_first + g0 + 160 + 24 + sl;
                            unsigned long long k_0_23 = pool[(long long)(k_s_23 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_23 = ((top1 < k_0_23) ? top1 : k_0_23);
                            top2 = ((top2 > k_lo_23) ? top2 : k_lo_23);
                            top1 = ((top1 > k_0_23) ? top1 : k_0_23);
                        }
                    }
                    if (g0 + 192 < npieces) {
                        if ((mine >> 24 & 1) != 0) {
                            int k_s_24 = s_first + g0 + 192 + sl;
                            unsigned long long k_0_24 = pool[(long long)(k_s_24 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_24 = ((top1 < k_0_24) ? top1 : k_0_24);
                            top2 = ((top2 > k_lo_24) ? top2 : k_lo_24);
                            top1 = ((top1 > k_0_24) ? top1 : k_0_24);
                        }
                        if ((mine >> 25 & 1) != 0) {
                            int k_s_25 = s_first + g0 + 192 + 8 + sl;
                            unsigned long long k_0_25 = pool[(long long)(k_s_25 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_25 = ((top1 < k_0_25) ? top1 : k_0_25);
                            top2 = ((top2 > k_lo_25) ? top2 : k_lo_25);
                            top1 = ((top1 > k_0_25) ? top1 : k_0_25);
                        }
                        if ((mine >> 26 & 1) != 0) {
                            int k_s_26 = s_first + g0 + 192 + 16 + sl;
                            unsigned long long k_0_26 = pool[(long long)(k_s_26 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_26 = ((top1 < k_0_26) ? top1 : k_0_26);
                            top2 = ((top2 > k_lo_26) ? top2 : k_lo_26);
                            top1 = ((top1 > k_0_26) ? top1 : k_0_26);
                        }
                        if ((mine >> 27 & 1) != 0) {
                            int k_s_27 = s_first + g0 + 192 + 24 + sl;
                            unsigned long long k_0_27 = pool[(long long)(k_s_27 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_27 = ((top1 < k_0_27) ? top1 : k_0_27);
                            top2 = ((top2 > k_lo_27) ? top2 : k_lo_27);
                            top1 = ((top1 > k_0_27) ? top1 : k_0_27);
                        }
                    }
                    if (g0 + 224 < npieces) {
                        if ((mine >> 28 & 1) != 0) {
                            int k_s_28 = s_first + g0 + 224 + sl;
                            unsigned long long k_0_28 = pool[(long long)(k_s_28 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_28 = ((top1 < k_0_28) ? top1 : k_0_28);
                            top2 = ((top2 > k_lo_28) ? top2 : k_lo_28);
                            top1 = ((top1 > k_0_28) ? top1 : k_0_28);
                        }
                        if ((mine >> 29 & 1) != 0) {
                            int k_s_29 = s_first + g0 + 224 + 8 + sl;
                            unsigned long long k_0_29 = pool[(long long)(k_s_29 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_29 = ((top1 < k_0_29) ? top1 : k_0_29);
                            top2 = ((top2 > k_lo_29) ? top2 : k_lo_29);
                            top1 = ((top1 > k_0_29) ? top1 : k_0_29);
                        }
                        if ((mine >> 30 & 1) != 0) {
                            int k_s_30 = s_first + g0 + 224 + 16 + sl;
                            unsigned long long k_0_30 = pool[(long long)(k_s_30 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_30 = ((top1 < k_0_30) ? top1 : k_0_30);
                            top2 = ((top2 > k_lo_30) ? top2 : k_lo_30);
                            top1 = ((top1 > k_0_30) ? top1 : k_0_30);
                        }
                        if ((mine >> 31 & 1) != 0) {
                            int k_s_31 = s_first + g0 + 224 + 24 + sl;
                            unsigned long long k_0_31 = pool[(long long)(k_s_31 + j) * pool_piece_stride + row_all];
                            unsigned long long k_lo_31 = ((top1 < k_0_31) ? top1 : k_0_31);
                            top2 = ((top2 > k_lo_31) ? top2 : k_lo_31);
                            top1 = ((top1 > k_0_31) ? top1 : k_0_31);
                        }
                    }
                    unsigned long long tau = top2;
                    unsigned int t_lo_0 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)tau, 1);
                    unsigned int t_hi_0 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(tau >> 32), 1);
                    unsigned long long t_o_0 = (unsigned long long)t_hi_0 << 32 | (unsigned long long)t_lo_0;
                    tau = ((tau < t_o_0) ? tau : t_o_0);
                    unsigned int t_lo_1 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)tau, 2);
                    unsigned int t_hi_1 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(tau >> 32), 2);
                    unsigned long long t_o_1 = (unsigned long long)t_hi_1 << 32 | (unsigned long long)t_lo_1;
                    tau = ((tau < t_o_1) ? tau : t_o_1);
                    unsigned int t_lo_2 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)tau, 4);
                    unsigned int t_hi_2 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(tau >> 32), 4);
                    unsigned long long t_o_2 = (unsigned long long)t_hi_2 << 32 | (unsigned long long)t_lo_2;
                    tau = ((tau < t_o_2) ? tau : t_o_2);
                    unsigned int keep = 0;
                    if (g0 < npieces) {
                        if ((mine >> 0 & 1) != 0) {
                            int c_s_0 = s_first + g0 + sl;
                            unsigned long long c_0_0 = pool[(long long)(c_s_0 + j) * pool_piece_stride + row_all];
                            if (c_0_0 >= tau) {
                                keep = keep | 1 << 0;
                            }
                        }
                        if ((mine >> 1 & 1) != 0) {
                            int c_s_1 = s_first + g0 + 8 + sl;
                            unsigned long long c_0_1 = pool[(long long)(c_s_1 + j) * pool_piece_stride + row_all];
                            if (c_0_1 >= tau) {
                                keep = keep | 1 << 1;
                            }
                        }
                        if ((mine >> 2 & 1) != 0) {
                            int c_s_2 = s_first + g0 + 16 + sl;
                            unsigned long long c_0_2 = pool[(long long)(c_s_2 + j) * pool_piece_stride + row_all];
                            if (c_0_2 >= tau) {
                                keep = keep | 1 << 2;
                            }
                        }
                        if ((mine >> 3 & 1) != 0) {
                            int c_s_3 = s_first + g0 + 24 + sl;
                            unsigned long long c_0_3 = pool[(long long)(c_s_3 + j) * pool_piece_stride + row_all];
                            if (c_0_3 >= tau) {
                                keep = keep | 1 << 3;
                            }
                        }
                    }
                    if (g0 + 32 < npieces) {
                        if ((mine >> 4 & 1) != 0) {
                            int c_s_4 = s_first + g0 + 32 + sl;
                            unsigned long long c_0_4 = pool[(long long)(c_s_4 + j) * pool_piece_stride + row_all];
                            if (c_0_4 >= tau) {
                                keep = keep | 1 << 4;
                            }
                        }
                        if ((mine >> 5 & 1) != 0) {
                            int c_s_5 = s_first + g0 + 32 + 8 + sl;
                            unsigned long long c_0_5 = pool[(long long)(c_s_5 + j) * pool_piece_stride + row_all];
                            if (c_0_5 >= tau) {
                                keep = keep | 1 << 5;
                            }
                        }
                        if ((mine >> 6 & 1) != 0) {
                            int c_s_6 = s_first + g0 + 32 + 16 + sl;
                            unsigned long long c_0_6 = pool[(long long)(c_s_6 + j) * pool_piece_stride + row_all];
                            if (c_0_6 >= tau) {
                                keep = keep | 1 << 6;
                            }
                        }
                        if ((mine >> 7 & 1) != 0) {
                            int c_s_7 = s_first + g0 + 32 + 24 + sl;
                            unsigned long long c_0_7 = pool[(long long)(c_s_7 + j) * pool_piece_stride + row_all];
                            if (c_0_7 >= tau) {
                                keep = keep | 1 << 7;
                            }
                        }
                    }
                    if (g0 + 64 < npieces) {
                        if ((mine >> 8 & 1) != 0) {
                            int c_s_8 = s_first + g0 + 64 + sl;
                            unsigned long long c_0_8 = pool[(long long)(c_s_8 + j) * pool_piece_stride + row_all];
                            if (c_0_8 >= tau) {
                                keep = keep | 1 << 8;
                            }
                        }
                        if ((mine >> 9 & 1) != 0) {
                            int c_s_9 = s_first + g0 + 64 + 8 + sl;
                            unsigned long long c_0_9 = pool[(long long)(c_s_9 + j) * pool_piece_stride + row_all];
                            if (c_0_9 >= tau) {
                                keep = keep | 1 << 9;
                            }
                        }
                        if ((mine >> 10 & 1) != 0) {
                            int c_s_10 = s_first + g0 + 64 + 16 + sl;
                            unsigned long long c_0_10 = pool[(long long)(c_s_10 + j) * pool_piece_stride + row_all];
                            if (c_0_10 >= tau) {
                                keep = keep | 1 << 10;
                            }
                        }
                        if ((mine >> 11 & 1) != 0) {
                            int c_s_11 = s_first + g0 + 64 + 24 + sl;
                            unsigned long long c_0_11 = pool[(long long)(c_s_11 + j) * pool_piece_stride + row_all];
                            if (c_0_11 >= tau) {
                                keep = keep | 1 << 11;
                            }
                        }
                    }
                    if (g0 + 96 < npieces) {
                        if ((mine >> 12 & 1) != 0) {
                            int c_s_12 = s_first + g0 + 96 + sl;
                            unsigned long long c_0_12 = pool[(long long)(c_s_12 + j) * pool_piece_stride + row_all];
                            if (c_0_12 >= tau) {
                                keep = keep | 1 << 12;
                            }
                        }
                        if ((mine >> 13 & 1) != 0) {
                            int c_s_13 = s_first + g0 + 96 + 8 + sl;
                            unsigned long long c_0_13 = pool[(long long)(c_s_13 + j) * pool_piece_stride + row_all];
                            if (c_0_13 >= tau) {
                                keep = keep | 1 << 13;
                            }
                        }
                        if ((mine >> 14 & 1) != 0) {
                            int c_s_14 = s_first + g0 + 96 + 16 + sl;
                            unsigned long long c_0_14 = pool[(long long)(c_s_14 + j) * pool_piece_stride + row_all];
                            if (c_0_14 >= tau) {
                                keep = keep | 1 << 14;
                            }
                        }
                        if ((mine >> 15 & 1) != 0) {
                            int c_s_15 = s_first + g0 + 96 + 24 + sl;
                            unsigned long long c_0_15 = pool[(long long)(c_s_15 + j) * pool_piece_stride + row_all];
                            if (c_0_15 >= tau) {
                                keep = keep | 1 << 15;
                            }
                        }
                    }
                    if (g0 + 128 < npieces) {
                        if ((mine >> 16 & 1) != 0) {
                            int c_s_16 = s_first + g0 + 128 + sl;
                            unsigned long long c_0_16 = pool[(long long)(c_s_16 + j) * pool_piece_stride + row_all];
                            if (c_0_16 >= tau) {
                                keep = keep | 1 << 16;
                            }
                        }
                        if ((mine >> 17 & 1) != 0) {
                            int c_s_17 = s_first + g0 + 128 + 8 + sl;
                            unsigned long long c_0_17 = pool[(long long)(c_s_17 + j) * pool_piece_stride + row_all];
                            if (c_0_17 >= tau) {
                                keep = keep | 1 << 17;
                            }
                        }
                        if ((mine >> 18 & 1) != 0) {
                            int c_s_18 = s_first + g0 + 128 + 16 + sl;
                            unsigned long long c_0_18 = pool[(long long)(c_s_18 + j) * pool_piece_stride + row_all];
                            if (c_0_18 >= tau) {
                                keep = keep | 1 << 18;
                            }
                        }
                        if ((mine >> 19 & 1) != 0) {
                            int c_s_19 = s_first + g0 + 128 + 24 + sl;
                            unsigned long long c_0_19 = pool[(long long)(c_s_19 + j) * pool_piece_stride + row_all];
                            if (c_0_19 >= tau) {
                                keep = keep | 1 << 19;
                            }
                        }
                    }
                    if (g0 + 160 < npieces) {
                        if ((mine >> 20 & 1) != 0) {
                            int c_s_20 = s_first + g0 + 160 + sl;
                            unsigned long long c_0_20 = pool[(long long)(c_s_20 + j) * pool_piece_stride + row_all];
                            if (c_0_20 >= tau) {
                                keep = keep | 1 << 20;
                            }
                        }
                        if ((mine >> 21 & 1) != 0) {
                            int c_s_21 = s_first + g0 + 160 + 8 + sl;
                            unsigned long long c_0_21 = pool[(long long)(c_s_21 + j) * pool_piece_stride + row_all];
                            if (c_0_21 >= tau) {
                                keep = keep | 1 << 21;
                            }
                        }
                        if ((mine >> 22 & 1) != 0) {
                            int c_s_22 = s_first + g0 + 160 + 16 + sl;
                            unsigned long long c_0_22 = pool[(long long)(c_s_22 + j) * pool_piece_stride + row_all];
                            if (c_0_22 >= tau) {
                                keep = keep | 1 << 22;
                            }
                        }
                        if ((mine >> 23 & 1) != 0) {
                            int c_s_23 = s_first + g0 + 160 + 24 + sl;
                            unsigned long long c_0_23 = pool[(long long)(c_s_23 + j) * pool_piece_stride + row_all];
                            if (c_0_23 >= tau) {
                                keep = keep | 1 << 23;
                            }
                        }
                    }
                    if (g0 + 192 < npieces) {
                        if ((mine >> 24 & 1) != 0) {
                            int c_s_24 = s_first + g0 + 192 + sl;
                            unsigned long long c_0_24 = pool[(long long)(c_s_24 + j) * pool_piece_stride + row_all];
                            if (c_0_24 >= tau) {
                                keep = keep | 1 << 24;
                            }
                        }
                        if ((mine >> 25 & 1) != 0) {
                            int c_s_25 = s_first + g0 + 192 + 8 + sl;
                            unsigned long long c_0_25 = pool[(long long)(c_s_25 + j) * pool_piece_stride + row_all];
                            if (c_0_25 >= tau) {
                                keep = keep | 1 << 25;
                            }
                        }
                        if ((mine >> 26 & 1) != 0) {
                            int c_s_26 = s_first + g0 + 192 + 16 + sl;
                            unsigned long long c_0_26 = pool[(long long)(c_s_26 + j) * pool_piece_stride + row_all];
                            if (c_0_26 >= tau) {
                                keep = keep | 1 << 26;
                            }
                        }
                        if ((mine >> 27 & 1) != 0) {
                            int c_s_27 = s_first + g0 + 192 + 24 + sl;
                            unsigned long long c_0_27 = pool[(long long)(c_s_27 + j) * pool_piece_stride + row_all];
                            if (c_0_27 >= tau) {
                                keep = keep | 1 << 27;
                            }
                        }
                    }
                    if (g0 + 224 < npieces) {
                        if ((mine >> 28 & 1) != 0) {
                            int c_s_28 = s_first + g0 + 224 + sl;
                            unsigned long long c_0_28 = pool[(long long)(c_s_28 + j) * pool_piece_stride + row_all];
                            if (c_0_28 >= tau) {
                                keep = keep | 1 << 28;
                            }
                        }
                        if ((mine >> 29 & 1) != 0) {
                            int c_s_29 = s_first + g0 + 224 + 8 + sl;
                            unsigned long long c_0_29 = pool[(long long)(c_s_29 + j) * pool_piece_stride + row_all];
                            if (c_0_29 >= tau) {
                                keep = keep | 1 << 29;
                            }
                        }
                        if ((mine >> 30 & 1) != 0) {
                            int c_s_30 = s_first + g0 + 224 + 16 + sl;
                            unsigned long long c_0_30 = pool[(long long)(c_s_30 + j) * pool_piece_stride + row_all];
                            if (c_0_30 >= tau) {
                                keep = keep | 1 << 30;
                            }
                        }
                        if ((mine >> 31 & 1) != 0) {
                            int c_s_31 = s_first + g0 + 224 + 24 + sl;
                            unsigned long long c_0_31 = pool[(long long)(c_s_31 + j) * pool_piece_stride + row_all];
                            if (c_0_31 >= tau) {
                                keep = keep | 1 << 31;
                            }
                        }
                    }
                    mine = keep;
                    int n_cells = (int)(unsigned int)__popc(mine);
                    if (n_cells > 0 && RL[0] == 0) {
                        int fb;
                        asm volatile("bfind.u32 %0, %1;" : "=r"(fb) : "r"(mine));
                        mine = mine & ~(1 << (unsigned int)fb);
                        float* ll_src = reinterpret_cast<float*>(pool + ((long long)(s_first + g0 + 32 * (fb >> 2) + 8 * (fb & 3) + sl + j) * pool_piece_stride + row_all));
                        float ll_f[32];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src + 0);
                            ll_f[0 + 0] = _v4.x;
                            ll_f[0 + 1] = _v4.y;
                            ll_f[0 + 2] = _v4.z;
                            ll_f[0 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src + 4);
                            ll_f[4 + 0] = _v4.x;
                            ll_f[4 + 1] = _v4.y;
                            ll_f[4 + 2] = _v4.z;
                            ll_f[4 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src + 8);
                            ll_f[8 + 0] = _v4.x;
                            ll_f[8 + 1] = _v4.y;
                            ll_f[8 + 2] = _v4.z;
                            ll_f[8 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src + 12);
                            ll_f[12 + 0] = _v4.x;
                            ll_f[12 + 1] = _v4.y;
                            ll_f[12 + 2] = _v4.z;
                            ll_f[12 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src + 16);
                            ll_f[16 + 0] = _v4.x;
                            ll_f[16 + 1] = _v4.y;
                            ll_f[16 + 2] = _v4.z;
                            ll_f[16 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src + 20);
                            ll_f[20 + 0] = _v4.x;
                            ll_f[20 + 1] = _v4.y;
                            ll_f[20 + 2] = _v4.z;
                            ll_f[20 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src + 24);
                            ll_f[24 + 0] = _v4.x;
                            ll_f[24 + 1] = _v4.y;
                            ll_f[24 + 2] = _v4.z;
                            ll_f[24 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src + 28);
                            ll_f[28 + 0] = _v4.x;
                            ll_f[28 + 1] = _v4.y;
                            ll_f[28 + 2] = _v4.z;
                            ll_f[28 + 3] = _v4.w;
                        }
                        unsigned int ll_lo_0 = 0;
                        unsigned int ll_hi_0 = 0;
                        float _bit_cast_src_43 = ll_f[0];
                        ll_lo_0 = reinterpret_cast<unsigned int*>(&_bit_cast_src_43)[0];
                        float _bit_cast_src_44 = ll_f[1];
                        ll_hi_0 = reinterpret_cast<unsigned int*>(&_bit_cast_src_44)[0];
                        RL[0] = (unsigned long long)ll_hi_0 << 32 | (unsigned long long)ll_lo_0;
                        unsigned int ll_lo_1 = 0;
                        unsigned int ll_hi_1 = 0;
                        float _bit_cast_src_45 = ll_f[2];
                        ll_lo_1 = reinterpret_cast<unsigned int*>(&_bit_cast_src_45)[0];
                        float _bit_cast_src_46 = ll_f[3];
                        ll_hi_1 = reinterpret_cast<unsigned int*>(&_bit_cast_src_46)[0];
                        RL[1] = (unsigned long long)ll_hi_1 << 32 | (unsigned long long)ll_lo_1;
                        unsigned int ll_lo_2 = 0;
                        unsigned int ll_hi_2 = 0;
                        float _bit_cast_src_47 = ll_f[4];
                        ll_lo_2 = reinterpret_cast<unsigned int*>(&_bit_cast_src_47)[0];
                        float _bit_cast_src_48 = ll_f[5];
                        ll_hi_2 = reinterpret_cast<unsigned int*>(&_bit_cast_src_48)[0];
                        RL[2] = (unsigned long long)ll_hi_2 << 32 | (unsigned long long)ll_lo_2;
                        unsigned int ll_lo_3 = 0;
                        unsigned int ll_hi_3 = 0;
                        float _bit_cast_src_49 = ll_f[6];
                        ll_lo_3 = reinterpret_cast<unsigned int*>(&_bit_cast_src_49)[0];
                        float _bit_cast_src_50 = ll_f[7];
                        ll_hi_3 = reinterpret_cast<unsigned int*>(&_bit_cast_src_50)[0];
                        RL[3] = (unsigned long long)ll_hi_3 << 32 | (unsigned long long)ll_lo_3;
                        unsigned int ll_lo_4 = 0;
                        unsigned int ll_hi_4 = 0;
                        float _bit_cast_src_51 = ll_f[8];
                        ll_lo_4 = reinterpret_cast<unsigned int*>(&_bit_cast_src_51)[0];
                        float _bit_cast_src_52 = ll_f[9];
                        ll_hi_4 = reinterpret_cast<unsigned int*>(&_bit_cast_src_52)[0];
                        RL[4] = (unsigned long long)ll_hi_4 << 32 | (unsigned long long)ll_lo_4;
                        unsigned int ll_lo_5 = 0;
                        unsigned int ll_hi_5 = 0;
                        float _bit_cast_src_53 = ll_f[10];
                        ll_lo_5 = reinterpret_cast<unsigned int*>(&_bit_cast_src_53)[0];
                        float _bit_cast_src_54 = ll_f[11];
                        ll_hi_5 = reinterpret_cast<unsigned int*>(&_bit_cast_src_54)[0];
                        RL[5] = (unsigned long long)ll_hi_5 << 32 | (unsigned long long)ll_lo_5;
                        unsigned int ll_lo_6 = 0;
                        unsigned int ll_hi_6 = 0;
                        float _bit_cast_src_55 = ll_f[12];
                        ll_lo_6 = reinterpret_cast<unsigned int*>(&_bit_cast_src_55)[0];
                        float _bit_cast_src_56 = ll_f[13];
                        ll_hi_6 = reinterpret_cast<unsigned int*>(&_bit_cast_src_56)[0];
                        RL[6] = (unsigned long long)ll_hi_6 << 32 | (unsigned long long)ll_lo_6;
                        unsigned int ll_lo_7 = 0;
                        unsigned int ll_hi_7 = 0;
                        float _bit_cast_src_57 = ll_f[14];
                        ll_lo_7 = reinterpret_cast<unsigned int*>(&_bit_cast_src_57)[0];
                        float _bit_cast_src_58 = ll_f[15];
                        ll_hi_7 = reinterpret_cast<unsigned int*>(&_bit_cast_src_58)[0];
                        RL[7] = (unsigned long long)ll_hi_7 << 32 | (unsigned long long)ll_lo_7;
                        unsigned int ll_lo_8 = 0;
                        unsigned int ll_hi_8 = 0;
                        float _bit_cast_src_59 = ll_f[16];
                        ll_lo_8 = reinterpret_cast<unsigned int*>(&_bit_cast_src_59)[0];
                        float _bit_cast_src_60 = ll_f[17];
                        ll_hi_8 = reinterpret_cast<unsigned int*>(&_bit_cast_src_60)[0];
                        RL[8] = (unsigned long long)ll_hi_8 << 32 | (unsigned long long)ll_lo_8;
                        unsigned int ll_lo_9 = 0;
                        unsigned int ll_hi_9 = 0;
                        float _bit_cast_src_61 = ll_f[18];
                        ll_lo_9 = reinterpret_cast<unsigned int*>(&_bit_cast_src_61)[0];
                        float _bit_cast_src_62 = ll_f[19];
                        ll_hi_9 = reinterpret_cast<unsigned int*>(&_bit_cast_src_62)[0];
                        RL[9] = (unsigned long long)ll_hi_9 << 32 | (unsigned long long)ll_lo_9;
                        unsigned int ll_lo_10 = 0;
                        unsigned int ll_hi_10 = 0;
                        float _bit_cast_src_63 = ll_f[20];
                        ll_lo_10 = reinterpret_cast<unsigned int*>(&_bit_cast_src_63)[0];
                        float _bit_cast_src_64 = ll_f[21];
                        ll_hi_10 = reinterpret_cast<unsigned int*>(&_bit_cast_src_64)[0];
                        RL[10] = (unsigned long long)ll_hi_10 << 32 | (unsigned long long)ll_lo_10;
                        unsigned int ll_lo_11 = 0;
                        unsigned int ll_hi_11 = 0;
                        float _bit_cast_src_65 = ll_f[22];
                        ll_lo_11 = reinterpret_cast<unsigned int*>(&_bit_cast_src_65)[0];
                        float _bit_cast_src_66 = ll_f[23];
                        ll_hi_11 = reinterpret_cast<unsigned int*>(&_bit_cast_src_66)[0];
                        RL[11] = (unsigned long long)ll_hi_11 << 32 | (unsigned long long)ll_lo_11;
                        unsigned int ll_lo_12 = 0;
                        unsigned int ll_hi_12 = 0;
                        float _bit_cast_src_67 = ll_f[24];
                        ll_lo_12 = reinterpret_cast<unsigned int*>(&_bit_cast_src_67)[0];
                        float _bit_cast_src_68 = ll_f[25];
                        ll_hi_12 = reinterpret_cast<unsigned int*>(&_bit_cast_src_68)[0];
                        RL[12] = (unsigned long long)ll_hi_12 << 32 | (unsigned long long)ll_lo_12;
                        unsigned int ll_lo_13 = 0;
                        unsigned int ll_hi_13 = 0;
                        float _bit_cast_src_69 = ll_f[26];
                        ll_lo_13 = reinterpret_cast<unsigned int*>(&_bit_cast_src_69)[0];
                        float _bit_cast_src_70 = ll_f[27];
                        ll_hi_13 = reinterpret_cast<unsigned int*>(&_bit_cast_src_70)[0];
                        RL[13] = (unsigned long long)ll_hi_13 << 32 | (unsigned long long)ll_lo_13;
                        unsigned int ll_lo_14 = 0;
                        unsigned int ll_hi_14 = 0;
                        float _bit_cast_src_71 = ll_f[28];
                        ll_lo_14 = reinterpret_cast<unsigned int*>(&_bit_cast_src_71)[0];
                        float _bit_cast_src_72 = ll_f[29];
                        ll_hi_14 = reinterpret_cast<unsigned int*>(&_bit_cast_src_72)[0];
                        RL[14] = (unsigned long long)ll_hi_14 << 32 | (unsigned long long)ll_lo_14;
                        unsigned int ll_lo_15 = 0;
                        unsigned int ll_hi_15 = 0;
                        float _bit_cast_src_73 = ll_f[30];
                        ll_lo_15 = reinterpret_cast<unsigned int*>(&_bit_cast_src_73)[0];
                        float _bit_cast_src_74 = ll_f[31];
                        ll_hi_15 = reinterpret_cast<unsigned int*>(&_bit_cast_src_74)[0];
                        RL[15] = (unsigned long long)ll_hi_15 << 32 | (unsigned long long)ll_lo_15;
                        n_cells = n_cells - 1;
                    }
                    if (n_cells > 0) {
                        int nb;
                        asm volatile("bfind.u32 %0, %1;" : "=r"(nb) : "r"(mine));
                        mine = mine & ~(1 << (unsigned int)nb);
                        float* ll_src_1 = reinterpret_cast<float*>(pool + ((long long)(s_first + g0 + 32 * (nb >> 2) + 8 * (nb & 3) + sl + j) * pool_piece_stride + row_all));
                        float ll_f_1[32];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src_1 + 0);
                            ll_f_1[0 + 0] = _v4.x;
                            ll_f_1[0 + 1] = _v4.y;
                            ll_f_1[0 + 2] = _v4.z;
                            ll_f_1[0 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src_1 + 4);
                            ll_f_1[4 + 0] = _v4.x;
                            ll_f_1[4 + 1] = _v4.y;
                            ll_f_1[4 + 2] = _v4.z;
                            ll_f_1[4 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src_1 + 8);
                            ll_f_1[8 + 0] = _v4.x;
                            ll_f_1[8 + 1] = _v4.y;
                            ll_f_1[8 + 2] = _v4.z;
                            ll_f_1[8 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src_1 + 12);
                            ll_f_1[12 + 0] = _v4.x;
                            ll_f_1[12 + 1] = _v4.y;
                            ll_f_1[12 + 2] = _v4.z;
                            ll_f_1[12 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src_1 + 16);
                            ll_f_1[16 + 0] = _v4.x;
                            ll_f_1[16 + 1] = _v4.y;
                            ll_f_1[16 + 2] = _v4.z;
                            ll_f_1[16 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src_1 + 20);
                            ll_f_1[20 + 0] = _v4.x;
                            ll_f_1[20 + 1] = _v4.y;
                            ll_f_1[20 + 2] = _v4.z;
                            ll_f_1[20 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src_1 + 24);
                            ll_f_1[24 + 0] = _v4.x;
                            ll_f_1[24 + 1] = _v4.y;
                            ll_f_1[24 + 2] = _v4.z;
                            ll_f_1[24 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(ll_src_1 + 28);
                            ll_f_1[28 + 0] = _v4.x;
                            ll_f_1[28 + 1] = _v4.y;
                            ll_f_1[28 + 2] = _v4.z;
                            ll_f_1[28 + 3] = _v4.w;
                        }
                        unsigned int ll_lo_16 = 0;
                        unsigned int ll_hi_16 = 0;
                        float _bit_cast_src_83 = ll_f_1[0];
                        ll_lo_16 = reinterpret_cast<unsigned int*>(&_bit_cast_src_83)[0];
                        float _bit_cast_src_84 = ll_f_1[1];
                        ll_hi_16 = reinterpret_cast<unsigned int*>(&_bit_cast_src_84)[0];
                        BN[0] = (unsigned long long)ll_hi_16 << 32 | (unsigned long long)ll_lo_16;
                        unsigned int ll_lo_17 = 0;
                        unsigned int ll_hi_17 = 0;
                        float _bit_cast_src_85 = ll_f_1[2];
                        ll_lo_17 = reinterpret_cast<unsigned int*>(&_bit_cast_src_85)[0];
                        float _bit_cast_src_86 = ll_f_1[3];
                        ll_hi_17 = reinterpret_cast<unsigned int*>(&_bit_cast_src_86)[0];
                        BN[1] = (unsigned long long)ll_hi_17 << 32 | (unsigned long long)ll_lo_17;
                        unsigned int ll_lo_18 = 0;
                        unsigned int ll_hi_18 = 0;
                        float _bit_cast_src_87 = ll_f_1[4];
                        ll_lo_18 = reinterpret_cast<unsigned int*>(&_bit_cast_src_87)[0];
                        float _bit_cast_src_88 = ll_f_1[5];
                        ll_hi_18 = reinterpret_cast<unsigned int*>(&_bit_cast_src_88)[0];
                        BN[2] = (unsigned long long)ll_hi_18 << 32 | (unsigned long long)ll_lo_18;
                        unsigned int ll_lo_19 = 0;
                        unsigned int ll_hi_19 = 0;
                        float _bit_cast_src_89 = ll_f_1[6];
                        ll_lo_19 = reinterpret_cast<unsigned int*>(&_bit_cast_src_89)[0];
                        float _bit_cast_src_90 = ll_f_1[7];
                        ll_hi_19 = reinterpret_cast<unsigned int*>(&_bit_cast_src_90)[0];
                        BN[3] = (unsigned long long)ll_hi_19 << 32 | (unsigned long long)ll_lo_19;
                        unsigned int ll_lo_20 = 0;
                        unsigned int ll_hi_20 = 0;
                        float _bit_cast_src_91 = ll_f_1[8];
                        ll_lo_20 = reinterpret_cast<unsigned int*>(&_bit_cast_src_91)[0];
                        float _bit_cast_src_92 = ll_f_1[9];
                        ll_hi_20 = reinterpret_cast<unsigned int*>(&_bit_cast_src_92)[0];
                        BN[4] = (unsigned long long)ll_hi_20 << 32 | (unsigned long long)ll_lo_20;
                        unsigned int ll_lo_21 = 0;
                        unsigned int ll_hi_21 = 0;
                        float _bit_cast_src_93 = ll_f_1[10];
                        ll_lo_21 = reinterpret_cast<unsigned int*>(&_bit_cast_src_93)[0];
                        float _bit_cast_src_94 = ll_f_1[11];
                        ll_hi_21 = reinterpret_cast<unsigned int*>(&_bit_cast_src_94)[0];
                        BN[5] = (unsigned long long)ll_hi_21 << 32 | (unsigned long long)ll_lo_21;
                        unsigned int ll_lo_22 = 0;
                        unsigned int ll_hi_22 = 0;
                        float _bit_cast_src_95 = ll_f_1[12];
                        ll_lo_22 = reinterpret_cast<unsigned int*>(&_bit_cast_src_95)[0];
                        float _bit_cast_src_96 = ll_f_1[13];
                        ll_hi_22 = reinterpret_cast<unsigned int*>(&_bit_cast_src_96)[0];
                        BN[6] = (unsigned long long)ll_hi_22 << 32 | (unsigned long long)ll_lo_22;
                        unsigned int ll_lo_23 = 0;
                        unsigned int ll_hi_23 = 0;
                        float _bit_cast_src_97 = ll_f_1[14];
                        ll_lo_23 = reinterpret_cast<unsigned int*>(&_bit_cast_src_97)[0];
                        float _bit_cast_src_98 = ll_f_1[15];
                        ll_hi_23 = reinterpret_cast<unsigned int*>(&_bit_cast_src_98)[0];
                        BN[7] = (unsigned long long)ll_hi_23 << 32 | (unsigned long long)ll_lo_23;
                        unsigned int ll_lo_24 = 0;
                        unsigned int ll_hi_24 = 0;
                        float _bit_cast_src_99 = ll_f_1[16];
                        ll_lo_24 = reinterpret_cast<unsigned int*>(&_bit_cast_src_99)[0];
                        float _bit_cast_src_100 = ll_f_1[17];
                        ll_hi_24 = reinterpret_cast<unsigned int*>(&_bit_cast_src_100)[0];
                        BN[8] = (unsigned long long)ll_hi_24 << 32 | (unsigned long long)ll_lo_24;
                        unsigned int ll_lo_25 = 0;
                        unsigned int ll_hi_25 = 0;
                        float _bit_cast_src_101 = ll_f_1[18];
                        ll_lo_25 = reinterpret_cast<unsigned int*>(&_bit_cast_src_101)[0];
                        float _bit_cast_src_102 = ll_f_1[19];
                        ll_hi_25 = reinterpret_cast<unsigned int*>(&_bit_cast_src_102)[0];
                        BN[9] = (unsigned long long)ll_hi_25 << 32 | (unsigned long long)ll_lo_25;
                        unsigned int ll_lo_26 = 0;
                        unsigned int ll_hi_26 = 0;
                        float _bit_cast_src_103 = ll_f_1[20];
                        ll_lo_26 = reinterpret_cast<unsigned int*>(&_bit_cast_src_103)[0];
                        float _bit_cast_src_104 = ll_f_1[21];
                        ll_hi_26 = reinterpret_cast<unsigned int*>(&_bit_cast_src_104)[0];
                        BN[10] = (unsigned long long)ll_hi_26 << 32 | (unsigned long long)ll_lo_26;
                        unsigned int ll_lo_27 = 0;
                        unsigned int ll_hi_27 = 0;
                        float _bit_cast_src_105 = ll_f_1[22];
                        ll_lo_27 = reinterpret_cast<unsigned int*>(&_bit_cast_src_105)[0];
                        float _bit_cast_src_106 = ll_f_1[23];
                        ll_hi_27 = reinterpret_cast<unsigned int*>(&_bit_cast_src_106)[0];
                        BN[11] = (unsigned long long)ll_hi_27 << 32 | (unsigned long long)ll_lo_27;
                        unsigned int ll_lo_28 = 0;
                        unsigned int ll_hi_28 = 0;
                        float _bit_cast_src_107 = ll_f_1[24];
                        ll_lo_28 = reinterpret_cast<unsigned int*>(&_bit_cast_src_107)[0];
                        float _bit_cast_src_108 = ll_f_1[25];
                        ll_hi_28 = reinterpret_cast<unsigned int*>(&_bit_cast_src_108)[0];
                        BN[12] = (unsigned long long)ll_hi_28 << 32 | (unsigned long long)ll_lo_28;
                        unsigned int ll_lo_29 = 0;
                        unsigned int ll_hi_29 = 0;
                        float _bit_cast_src_109 = ll_f_1[26];
                        ll_lo_29 = reinterpret_cast<unsigned int*>(&_bit_cast_src_109)[0];
                        float _bit_cast_src_110 = ll_f_1[27];
                        ll_hi_29 = reinterpret_cast<unsigned int*>(&_bit_cast_src_110)[0];
                        BN[13] = (unsigned long long)ll_hi_29 << 32 | (unsigned long long)ll_lo_29;
                        unsigned int ll_lo_30 = 0;
                        unsigned int ll_hi_30 = 0;
                        float _bit_cast_src_111 = ll_f_1[28];
                        ll_lo_30 = reinterpret_cast<unsigned int*>(&_bit_cast_src_111)[0];
                        float _bit_cast_src_112 = ll_f_1[29];
                        ll_hi_30 = reinterpret_cast<unsigned int*>(&_bit_cast_src_112)[0];
                        BN[14] = (unsigned long long)ll_hi_30 << 32 | (unsigned long long)ll_lo_30;
                        unsigned int ll_lo_31 = 0;
                        unsigned int ll_hi_31 = 0;
                        float _bit_cast_src_113 = ll_f_1[30];
                        ll_lo_31 = reinterpret_cast<unsigned int*>(&_bit_cast_src_113)[0];
                        float _bit_cast_src_114 = ll_f_1[31];
                        ll_hi_31 = reinterpret_cast<unsigned int*>(&_bit_cast_src_114)[0];
                        BN[15] = (unsigned long long)ll_hi_31 << 32 | (unsigned long long)ll_lo_31;
                        #pragma unroll 1
                        for (int mi = 0; mi < n_cells; mi++) {
                            BL[0] = BN[0];
                            BL[1] = BN[1];
                            BL[2] = BN[2];
                            BL[3] = BN[3];
                            BL[4] = BN[4];
                            BL[5] = BN[5];
                            BL[6] = BN[6];
                            BL[7] = BN[7];
                            BL[8] = BN[8];
                            BL[9] = BN[9];
                            BL[10] = BN[10];
                            BL[11] = BN[11];
                            BL[12] = BN[12];
                            BL[13] = BN[13];
                            BL[14] = BN[14];
                            BL[15] = BN[15];
                            if (mi + 1 < n_cells) {
                                int mb;
                                asm volatile("bfind.u32 %0, %1;" : "=r"(mb) : "r"(mine));
                                mine = mine & ~(1 << (unsigned int)mb);
                                float* ll_src_0 = reinterpret_cast<float*>(pool + ((long long)(s_first + g0 + 32 * (mb >> 2) + 8 * (mb & 3) + sl + j) * pool_piece_stride + row_all));
                                float ll_f_0[32];
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ll_src_0 + 0);
                                    ll_f_0[0 + 0] = _v4.x;
                                    ll_f_0[0 + 1] = _v4.y;
                                    ll_f_0[0 + 2] = _v4.z;
                                    ll_f_0[0 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ll_src_0 + 4);
                                    ll_f_0[4 + 0] = _v4.x;
                                    ll_f_0[4 + 1] = _v4.y;
                                    ll_f_0[4 + 2] = _v4.z;
                                    ll_f_0[4 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ll_src_0 + 8);
                                    ll_f_0[8 + 0] = _v4.x;
                                    ll_f_0[8 + 1] = _v4.y;
                                    ll_f_0[8 + 2] = _v4.z;
                                    ll_f_0[8 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ll_src_0 + 12);
                                    ll_f_0[12 + 0] = _v4.x;
                                    ll_f_0[12 + 1] = _v4.y;
                                    ll_f_0[12 + 2] = _v4.z;
                                    ll_f_0[12 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ll_src_0 + 16);
                                    ll_f_0[16 + 0] = _v4.x;
                                    ll_f_0[16 + 1] = _v4.y;
                                    ll_f_0[16 + 2] = _v4.z;
                                    ll_f_0[16 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ll_src_0 + 20);
                                    ll_f_0[20 + 0] = _v4.x;
                                    ll_f_0[20 + 1] = _v4.y;
                                    ll_f_0[20 + 2] = _v4.z;
                                    ll_f_0[20 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ll_src_0 + 24);
                                    ll_f_0[24 + 0] = _v4.x;
                                    ll_f_0[24 + 1] = _v4.y;
                                    ll_f_0[24 + 2] = _v4.z;
                                    ll_f_0[24 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ll_src_0 + 28);
                                    ll_f_0[28 + 0] = _v4.x;
                                    ll_f_0[28 + 1] = _v4.y;
                                    ll_f_0[28 + 2] = _v4.z;
                                    ll_f_0[28 + 3] = _v4.w;
                                }
                                unsigned int ll_lo_32 = 0;
                                unsigned int ll_hi_32 = 0;
                                float _bit_cast_src_123 = ll_f_0[0];
                                ll_lo_32 = reinterpret_cast<unsigned int*>(&_bit_cast_src_123)[0];
                                float _bit_cast_src_124 = ll_f_0[1];
                                ll_hi_32 = reinterpret_cast<unsigned int*>(&_bit_cast_src_124)[0];
                                BN[0] = (unsigned long long)ll_hi_32 << 32 | (unsigned long long)ll_lo_32;
                                unsigned int ll_lo_33 = 0;
                                unsigned int ll_hi_33 = 0;
                                float _bit_cast_src_125 = ll_f_0[2];
                                ll_lo_33 = reinterpret_cast<unsigned int*>(&_bit_cast_src_125)[0];
                                float _bit_cast_src_126 = ll_f_0[3];
                                ll_hi_33 = reinterpret_cast<unsigned int*>(&_bit_cast_src_126)[0];
                                BN[1] = (unsigned long long)ll_hi_33 << 32 | (unsigned long long)ll_lo_33;
                                unsigned int ll_lo_34 = 0;
                                unsigned int ll_hi_34 = 0;
                                float _bit_cast_src_127 = ll_f_0[4];
                                ll_lo_34 = reinterpret_cast<unsigned int*>(&_bit_cast_src_127)[0];
                                float _bit_cast_src_128 = ll_f_0[5];
                                ll_hi_34 = reinterpret_cast<unsigned int*>(&_bit_cast_src_128)[0];
                                BN[2] = (unsigned long long)ll_hi_34 << 32 | (unsigned long long)ll_lo_34;
                                unsigned int ll_lo_35 = 0;
                                unsigned int ll_hi_35 = 0;
                                float _bit_cast_src_129 = ll_f_0[6];
                                ll_lo_35 = reinterpret_cast<unsigned int*>(&_bit_cast_src_129)[0];
                                float _bit_cast_src_130 = ll_f_0[7];
                                ll_hi_35 = reinterpret_cast<unsigned int*>(&_bit_cast_src_130)[0];
                                BN[3] = (unsigned long long)ll_hi_35 << 32 | (unsigned long long)ll_lo_35;
                                unsigned int ll_lo_36 = 0;
                                unsigned int ll_hi_36 = 0;
                                float _bit_cast_src_131 = ll_f_0[8];
                                ll_lo_36 = reinterpret_cast<unsigned int*>(&_bit_cast_src_131)[0];
                                float _bit_cast_src_132 = ll_f_0[9];
                                ll_hi_36 = reinterpret_cast<unsigned int*>(&_bit_cast_src_132)[0];
                                BN[4] = (unsigned long long)ll_hi_36 << 32 | (unsigned long long)ll_lo_36;
                                unsigned int ll_lo_37 = 0;
                                unsigned int ll_hi_37 = 0;
                                float _bit_cast_src_133 = ll_f_0[10];
                                ll_lo_37 = reinterpret_cast<unsigned int*>(&_bit_cast_src_133)[0];
                                float _bit_cast_src_134 = ll_f_0[11];
                                ll_hi_37 = reinterpret_cast<unsigned int*>(&_bit_cast_src_134)[0];
                                BN[5] = (unsigned long long)ll_hi_37 << 32 | (unsigned long long)ll_lo_37;
                                unsigned int ll_lo_38 = 0;
                                unsigned int ll_hi_38 = 0;
                                float _bit_cast_src_135 = ll_f_0[12];
                                ll_lo_38 = reinterpret_cast<unsigned int*>(&_bit_cast_src_135)[0];
                                float _bit_cast_src_136 = ll_f_0[13];
                                ll_hi_38 = reinterpret_cast<unsigned int*>(&_bit_cast_src_136)[0];
                                BN[6] = (unsigned long long)ll_hi_38 << 32 | (unsigned long long)ll_lo_38;
                                unsigned int ll_lo_39 = 0;
                                unsigned int ll_hi_39 = 0;
                                float _bit_cast_src_137 = ll_f_0[14];
                                ll_lo_39 = reinterpret_cast<unsigned int*>(&_bit_cast_src_137)[0];
                                float _bit_cast_src_138 = ll_f_0[15];
                                ll_hi_39 = reinterpret_cast<unsigned int*>(&_bit_cast_src_138)[0];
                                BN[7] = (unsigned long long)ll_hi_39 << 32 | (unsigned long long)ll_lo_39;
                                unsigned int ll_lo_40 = 0;
                                unsigned int ll_hi_40 = 0;
                                float _bit_cast_src_139 = ll_f_0[16];
                                ll_lo_40 = reinterpret_cast<unsigned int*>(&_bit_cast_src_139)[0];
                                float _bit_cast_src_140 = ll_f_0[17];
                                ll_hi_40 = reinterpret_cast<unsigned int*>(&_bit_cast_src_140)[0];
                                BN[8] = (unsigned long long)ll_hi_40 << 32 | (unsigned long long)ll_lo_40;
                                unsigned int ll_lo_41 = 0;
                                unsigned int ll_hi_41 = 0;
                                float _bit_cast_src_141 = ll_f_0[18];
                                ll_lo_41 = reinterpret_cast<unsigned int*>(&_bit_cast_src_141)[0];
                                float _bit_cast_src_142 = ll_f_0[19];
                                ll_hi_41 = reinterpret_cast<unsigned int*>(&_bit_cast_src_142)[0];
                                BN[9] = (unsigned long long)ll_hi_41 << 32 | (unsigned long long)ll_lo_41;
                                unsigned int ll_lo_42 = 0;
                                unsigned int ll_hi_42 = 0;
                                float _bit_cast_src_143 = ll_f_0[20];
                                ll_lo_42 = reinterpret_cast<unsigned int*>(&_bit_cast_src_143)[0];
                                float _bit_cast_src_144 = ll_f_0[21];
                                ll_hi_42 = reinterpret_cast<unsigned int*>(&_bit_cast_src_144)[0];
                                BN[10] = (unsigned long long)ll_hi_42 << 32 | (unsigned long long)ll_lo_42;
                                unsigned int ll_lo_43 = 0;
                                unsigned int ll_hi_43 = 0;
                                float _bit_cast_src_145 = ll_f_0[22];
                                ll_lo_43 = reinterpret_cast<unsigned int*>(&_bit_cast_src_145)[0];
                                float _bit_cast_src_146 = ll_f_0[23];
                                ll_hi_43 = reinterpret_cast<unsigned int*>(&_bit_cast_src_146)[0];
                                BN[11] = (unsigned long long)ll_hi_43 << 32 | (unsigned long long)ll_lo_43;
                                unsigned int ll_lo_44 = 0;
                                unsigned int ll_hi_44 = 0;
                                float _bit_cast_src_147 = ll_f_0[24];
                                ll_lo_44 = reinterpret_cast<unsigned int*>(&_bit_cast_src_147)[0];
                                float _bit_cast_src_148 = ll_f_0[25];
                                ll_hi_44 = reinterpret_cast<unsigned int*>(&_bit_cast_src_148)[0];
                                BN[12] = (unsigned long long)ll_hi_44 << 32 | (unsigned long long)ll_lo_44;
                                unsigned int ll_lo_45 = 0;
                                unsigned int ll_hi_45 = 0;
                                float _bit_cast_src_149 = ll_f_0[26];
                                ll_lo_45 = reinterpret_cast<unsigned int*>(&_bit_cast_src_149)[0];
                                float _bit_cast_src_150 = ll_f_0[27];
                                ll_hi_45 = reinterpret_cast<unsigned int*>(&_bit_cast_src_150)[0];
                                BN[13] = (unsigned long long)ll_hi_45 << 32 | (unsigned long long)ll_lo_45;
                                unsigned int ll_lo_46 = 0;
                                unsigned int ll_hi_46 = 0;
                                float _bit_cast_src_151 = ll_f_0[28];
                                ll_lo_46 = reinterpret_cast<unsigned int*>(&_bit_cast_src_151)[0];
                                float _bit_cast_src_152 = ll_f_0[29];
                                ll_hi_46 = reinterpret_cast<unsigned int*>(&_bit_cast_src_152)[0];
                                BN[14] = (unsigned long long)ll_hi_46 << 32 | (unsigned long long)ll_lo_46;
                                unsigned int ll_lo_47 = 0;
                                unsigned int ll_hi_47 = 0;
                                float _bit_cast_src_153 = ll_f_0[30];
                                ll_lo_47 = reinterpret_cast<unsigned int*>(&_bit_cast_src_153)[0];
                                float _bit_cast_src_154 = ll_f_0[31];
                                ll_hi_47 = reinterpret_cast<unsigned int*>(&_bit_cast_src_154)[0];
                                BN[15] = (unsigned long long)ll_hi_47 << 32 | (unsigned long long)ll_lo_47;
                            }
                            unsigned long long l16_m_0 = BL[15];
                            RL[0] = ((RL[0] > l16_m_0) ? RL[0] : l16_m_0);
                            unsigned long long l16_m_1 = BL[14];
                            RL[1] = ((RL[1] > l16_m_1) ? RL[1] : l16_m_1);
                            unsigned long long l16_m_2 = BL[13];
                            RL[2] = ((RL[2] > l16_m_2) ? RL[2] : l16_m_2);
                            unsigned long long l16_m_3 = BL[12];
                            RL[3] = ((RL[3] > l16_m_3) ? RL[3] : l16_m_3);
                            unsigned long long l16_m_4 = BL[11];
                            RL[4] = ((RL[4] > l16_m_4) ? RL[4] : l16_m_4);
                            unsigned long long l16_m_5 = BL[10];
                            RL[5] = ((RL[5] > l16_m_5) ? RL[5] : l16_m_5);
                            unsigned long long l16_m_6 = BL[9];
                            RL[6] = ((RL[6] > l16_m_6) ? RL[6] : l16_m_6);
                            unsigned long long l16_m_7 = BL[8];
                            RL[7] = ((RL[7] > l16_m_7) ? RL[7] : l16_m_7);
                            unsigned long long l16_m_8 = BL[7];
                            RL[8] = ((RL[8] > l16_m_8) ? RL[8] : l16_m_8);
                            unsigned long long l16_m_9 = BL[6];
                            RL[9] = ((RL[9] > l16_m_9) ? RL[9] : l16_m_9);
                            unsigned long long l16_m_10 = BL[5];
                            RL[10] = ((RL[10] > l16_m_10) ? RL[10] : l16_m_10);
                            unsigned long long l16_m_11 = BL[4];
                            RL[11] = ((RL[11] > l16_m_11) ? RL[11] : l16_m_11);
                            unsigned long long l16_m_12 = BL[3];
                            RL[12] = ((RL[12] > l16_m_12) ? RL[12] : l16_m_12);
                            unsigned long long l16_m_13 = BL[2];
                            RL[13] = ((RL[13] > l16_m_13) ? RL[13] : l16_m_13);
                            unsigned long long l16_m_14 = BL[1];
                            RL[14] = ((RL[14] > l16_m_14) ? RL[14] : l16_m_14);
                            unsigned long long l16_m_15 = BL[0];
                            RL[15] = ((RL[15] > l16_m_15) ? RL[15] : l16_m_15);
                            unsigned long long l16_a_0 = RL[0];
                            unsigned long long l16_b_0 = RL[8];
                            RL[0] = ((l16_a_0 > l16_b_0) ? l16_a_0 : l16_b_0);
                            RL[8] = ((l16_a_0 > l16_b_0) ? l16_b_0 : l16_a_0);
                            unsigned long long l16_a_1 = RL[1];
                            unsigned long long l16_b_1 = RL[9];
                            RL[1] = ((l16_a_1 > l16_b_1) ? l16_a_1 : l16_b_1);
                            RL[9] = ((l16_a_1 > l16_b_1) ? l16_b_1 : l16_a_1);
                            unsigned long long l16_a_2 = RL[2];
                            unsigned long long l16_b_2 = RL[10];
                            RL[2] = ((l16_a_2 > l16_b_2) ? l16_a_2 : l16_b_2);
                            RL[10] = ((l16_a_2 > l16_b_2) ? l16_b_2 : l16_a_2);
                            unsigned long long l16_a_3 = RL[3];
                            unsigned long long l16_b_3 = RL[11];
                            RL[3] = ((l16_a_3 > l16_b_3) ? l16_a_3 : l16_b_3);
                            RL[11] = ((l16_a_3 > l16_b_3) ? l16_b_3 : l16_a_3);
                            unsigned long long l16_a_4 = RL[4];
                            unsigned long long l16_b_4 = RL[12];
                            RL[4] = ((l16_a_4 > l16_b_4) ? l16_a_4 : l16_b_4);
                            RL[12] = ((l16_a_4 > l16_b_4) ? l16_b_4 : l16_a_4);
                            unsigned long long l16_a_5 = RL[5];
                            unsigned long long l16_b_5 = RL[13];
                            RL[5] = ((l16_a_5 > l16_b_5) ? l16_a_5 : l16_b_5);
                            RL[13] = ((l16_a_5 > l16_b_5) ? l16_b_5 : l16_a_5);
                            unsigned long long l16_a_6 = RL[6];
                            unsigned long long l16_b_6 = RL[14];
                            RL[6] = ((l16_a_6 > l16_b_6) ? l16_a_6 : l16_b_6);
                            RL[14] = ((l16_a_6 > l16_b_6) ? l16_b_6 : l16_a_6);
                            unsigned long long l16_a_7 = RL[7];
                            unsigned long long l16_b_7 = RL[15];
                            RL[7] = ((l16_a_7 > l16_b_7) ? l16_a_7 : l16_b_7);
                            RL[15] = ((l16_a_7 > l16_b_7) ? l16_b_7 : l16_a_7);
                            unsigned long long l16_a_8 = RL[0];
                            unsigned long long l16_b_8 = RL[4];
                            RL[0] = ((l16_a_8 > l16_b_8) ? l16_a_8 : l16_b_8);
                            RL[4] = ((l16_a_8 > l16_b_8) ? l16_b_8 : l16_a_8);
                            unsigned long long l16_a_9 = RL[1];
                            unsigned long long l16_b_9 = RL[5];
                            RL[1] = ((l16_a_9 > l16_b_9) ? l16_a_9 : l16_b_9);
                            RL[5] = ((l16_a_9 > l16_b_9) ? l16_b_9 : l16_a_9);
                            unsigned long long l16_a_10 = RL[2];
                            unsigned long long l16_b_10 = RL[6];
                            RL[2] = ((l16_a_10 > l16_b_10) ? l16_a_10 : l16_b_10);
                            RL[6] = ((l16_a_10 > l16_b_10) ? l16_b_10 : l16_a_10);
                            unsigned long long l16_a_11 = RL[3];
                            unsigned long long l16_b_11 = RL[7];
                            RL[3] = ((l16_a_11 > l16_b_11) ? l16_a_11 : l16_b_11);
                            RL[7] = ((l16_a_11 > l16_b_11) ? l16_b_11 : l16_a_11);
                            unsigned long long l16_a_12 = RL[8];
                            unsigned long long l16_b_12 = RL[12];
                            RL[8] = ((l16_a_12 > l16_b_12) ? l16_a_12 : l16_b_12);
                            RL[12] = ((l16_a_12 > l16_b_12) ? l16_b_12 : l16_a_12);
                            unsigned long long l16_a_13 = RL[9];
                            unsigned long long l16_b_13 = RL[13];
                            RL[9] = ((l16_a_13 > l16_b_13) ? l16_a_13 : l16_b_13);
                            RL[13] = ((l16_a_13 > l16_b_13) ? l16_b_13 : l16_a_13);
                            unsigned long long l16_a_14 = RL[10];
                            unsigned long long l16_b_14 = RL[14];
                            RL[10] = ((l16_a_14 > l16_b_14) ? l16_a_14 : l16_b_14);
                            RL[14] = ((l16_a_14 > l16_b_14) ? l16_b_14 : l16_a_14);
                            unsigned long long l16_a_15 = RL[11];
                            unsigned long long l16_b_15 = RL[15];
                            RL[11] = ((l16_a_15 > l16_b_15) ? l16_a_15 : l16_b_15);
                            RL[15] = ((l16_a_15 > l16_b_15) ? l16_b_15 : l16_a_15);
                            unsigned long long l16_a_16 = RL[0];
                            unsigned long long l16_b_16 = RL[2];
                            RL[0] = ((l16_a_16 > l16_b_16) ? l16_a_16 : l16_b_16);
                            RL[2] = ((l16_a_16 > l16_b_16) ? l16_b_16 : l16_a_16);
                            unsigned long long l16_a_17 = RL[1];
                            unsigned long long l16_b_17 = RL[3];
                            RL[1] = ((l16_a_17 > l16_b_17) ? l16_a_17 : l16_b_17);
                            RL[3] = ((l16_a_17 > l16_b_17) ? l16_b_17 : l16_a_17);
                            unsigned long long l16_a_18 = RL[4];
                            unsigned long long l16_b_18 = RL[6];
                            RL[4] = ((l16_a_18 > l16_b_18) ? l16_a_18 : l16_b_18);
                            RL[6] = ((l16_a_18 > l16_b_18) ? l16_b_18 : l16_a_18);
                            unsigned long long l16_a_19 = RL[5];
                            unsigned long long l16_b_19 = RL[7];
                            RL[5] = ((l16_a_19 > l16_b_19) ? l16_a_19 : l16_b_19);
                            RL[7] = ((l16_a_19 > l16_b_19) ? l16_b_19 : l16_a_19);
                            unsigned long long l16_a_20 = RL[8];
                            unsigned long long l16_b_20 = RL[10];
                            RL[8] = ((l16_a_20 > l16_b_20) ? l16_a_20 : l16_b_20);
                            RL[10] = ((l16_a_20 > l16_b_20) ? l16_b_20 : l16_a_20);
                            unsigned long long l16_a_21 = RL[9];
                            unsigned long long l16_b_21 = RL[11];
                            RL[9] = ((l16_a_21 > l16_b_21) ? l16_a_21 : l16_b_21);
                            RL[11] = ((l16_a_21 > l16_b_21) ? l16_b_21 : l16_a_21);
                            unsigned long long l16_a_22 = RL[12];
                            unsigned long long l16_b_22 = RL[14];
                            RL[12] = ((l16_a_22 > l16_b_22) ? l16_a_22 : l16_b_22);
                            RL[14] = ((l16_a_22 > l16_b_22) ? l16_b_22 : l16_a_22);
                            unsigned long long l16_a_23 = RL[13];
                            unsigned long long l16_b_23 = RL[15];
                            RL[13] = ((l16_a_23 > l16_b_23) ? l16_a_23 : l16_b_23);
                            RL[15] = ((l16_a_23 > l16_b_23) ? l16_b_23 : l16_a_23);
                            unsigned long long l16_a_24 = RL[0];
                            unsigned long long l16_b_24 = RL[1];
                            RL[0] = ((l16_a_24 > l16_b_24) ? l16_a_24 : l16_b_24);
                            RL[1] = ((l16_a_24 > l16_b_24) ? l16_b_24 : l16_a_24);
                            unsigned long long l16_a_25 = RL[2];
                            unsigned long long l16_b_25 = RL[3];
                            RL[2] = ((l16_a_25 > l16_b_25) ? l16_a_25 : l16_b_25);
                            RL[3] = ((l16_a_25 > l16_b_25) ? l16_b_25 : l16_a_25);
                            unsigned long long l16_a_26 = RL[4];
                            unsigned long long l16_b_26 = RL[5];
                            RL[4] = ((l16_a_26 > l16_b_26) ? l16_a_26 : l16_b_26);
                            RL[5] = ((l16_a_26 > l16_b_26) ? l16_b_26 : l16_a_26);
                            unsigned long long l16_a_27 = RL[6];
                            unsigned long long l16_b_27 = RL[7];
                            RL[6] = ((l16_a_27 > l16_b_27) ? l16_a_27 : l16_b_27);
                            RL[7] = ((l16_a_27 > l16_b_27) ? l16_b_27 : l16_a_27);
                            unsigned long long l16_a_28 = RL[8];
                            unsigned long long l16_b_28 = RL[9];
                            RL[8] = ((l16_a_28 > l16_b_28) ? l16_a_28 : l16_b_28);
                            RL[9] = ((l16_a_28 > l16_b_28) ? l16_b_28 : l16_a_28);
                            unsigned long long l16_a_29 = RL[10];
                            unsigned long long l16_b_29 = RL[11];
                            RL[10] = ((l16_a_29 > l16_b_29) ? l16_a_29 : l16_b_29);
                            RL[11] = ((l16_a_29 > l16_b_29) ? l16_b_29 : l16_a_29);
                            unsigned long long l16_a_30 = RL[12];
                            unsigned long long l16_b_30 = RL[13];
                            RL[12] = ((l16_a_30 > l16_b_30) ? l16_a_30 : l16_b_30);
                            RL[13] = ((l16_a_30 > l16_b_30) ? l16_b_30 : l16_a_30);
                            unsigned long long l16_a_31 = RL[14];
                            unsigned long long l16_b_31 = RL[15];
                            RL[14] = ((l16_a_31 > l16_b_31) ? l16_a_31 : l16_b_31);
                            RL[15] = ((l16_a_31 > l16_b_31) ? l16_b_31 : l16_a_31);
                        }
                    }
                }
                int odd = sl & 1;
                unsigned long long x_v_0 = RL[0];
                unsigned int x_lo_0 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_0, 1);
                unsigned int x_hi_0 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_0 >> 32), 1);
                BL[0] = (unsigned long long)x_hi_0 << 32 | (unsigned long long)x_lo_0;
                unsigned long long x_v_1 = RL[1];
                unsigned int x_lo_1 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_1, 1);
                unsigned int x_hi_1 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_1 >> 32), 1);
                BL[1] = (unsigned long long)x_hi_1 << 32 | (unsigned long long)x_lo_1;
                unsigned long long x_v_2 = RL[2];
                unsigned int x_lo_2 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_2, 1);
                unsigned int x_hi_2 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_2 >> 32), 1);
                BL[2] = (unsigned long long)x_hi_2 << 32 | (unsigned long long)x_lo_2;
                unsigned long long x_v_3 = RL[3];
                unsigned int x_lo_3 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_3, 1);
                unsigned int x_hi_3 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_3 >> 32), 1);
                BL[3] = (unsigned long long)x_hi_3 << 32 | (unsigned long long)x_lo_3;
                unsigned long long x_v_4 = RL[4];
                unsigned int x_lo_4 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_4, 1);
                unsigned int x_hi_4 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_4 >> 32), 1);
                BL[4] = (unsigned long long)x_hi_4 << 32 | (unsigned long long)x_lo_4;
                unsigned long long x_v_5 = RL[5];
                unsigned int x_lo_5 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_5, 1);
                unsigned int x_hi_5 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_5 >> 32), 1);
                BL[5] = (unsigned long long)x_hi_5 << 32 | (unsigned long long)x_lo_5;
                unsigned long long x_v_6 = RL[6];
                unsigned int x_lo_6 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_6, 1);
                unsigned int x_hi_6 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_6 >> 32), 1);
                BL[6] = (unsigned long long)x_hi_6 << 32 | (unsigned long long)x_lo_6;
                unsigned long long x_v_7 = RL[7];
                unsigned int x_lo_7 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_7, 1);
                unsigned int x_hi_7 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_7 >> 32), 1);
                BL[7] = (unsigned long long)x_hi_7 << 32 | (unsigned long long)x_lo_7;
                unsigned long long x_v_8 = RL[8];
                unsigned int x_lo_8 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_8, 1);
                unsigned int x_hi_8 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_8 >> 32), 1);
                BL[8] = (unsigned long long)x_hi_8 << 32 | (unsigned long long)x_lo_8;
                unsigned long long x_v_9 = RL[9];
                unsigned int x_lo_9 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_9, 1);
                unsigned int x_hi_9 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_9 >> 32), 1);
                BL[9] = (unsigned long long)x_hi_9 << 32 | (unsigned long long)x_lo_9;
                unsigned long long x_v_10 = RL[10];
                unsigned int x_lo_10 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_10, 1);
                unsigned int x_hi_10 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_10 >> 32), 1);
                BL[10] = (unsigned long long)x_hi_10 << 32 | (unsigned long long)x_lo_10;
                unsigned long long x_v_11 = RL[11];
                unsigned int x_lo_11 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_11, 1);
                unsigned int x_hi_11 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_11 >> 32), 1);
                BL[11] = (unsigned long long)x_hi_11 << 32 | (unsigned long long)x_lo_11;
                unsigned long long x_v_12 = RL[12];
                unsigned int x_lo_12 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_12, 1);
                unsigned int x_hi_12 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_12 >> 32), 1);
                BL[12] = (unsigned long long)x_hi_12 << 32 | (unsigned long long)x_lo_12;
                unsigned long long x_v_13 = RL[13];
                unsigned int x_lo_13 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_13, 1);
                unsigned int x_hi_13 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_13 >> 32), 1);
                BL[13] = (unsigned long long)x_hi_13 << 32 | (unsigned long long)x_lo_13;
                unsigned long long x_v_14 = RL[14];
                unsigned int x_lo_14 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_14, 1);
                unsigned int x_hi_14 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_14 >> 32), 1);
                BL[14] = (unsigned long long)x_hi_14 << 32 | (unsigned long long)x_lo_14;
                unsigned long long x_v_15 = RL[15];
                unsigned int x_lo_15 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)x_v_15, 1);
                unsigned int x_hi_15 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(x_v_15 >> 32), 1);
                BL[15] = (unsigned long long)x_hi_15 << 32 | (unsigned long long)x_lo_15;
                unsigned long long x_b_0 = BL[15];
                RL[0] = ((RL[0] > x_b_0) ? RL[0] : x_b_0);
                unsigned long long x_b_1 = BL[14];
                RL[1] = ((RL[1] > x_b_1) ? RL[1] : x_b_1);
                unsigned long long x_b_2 = BL[13];
                RL[2] = ((RL[2] > x_b_2) ? RL[2] : x_b_2);
                unsigned long long x_b_3 = BL[12];
                RL[3] = ((RL[3] > x_b_3) ? RL[3] : x_b_3);
                unsigned long long x_b_4 = BL[11];
                RL[4] = ((RL[4] > x_b_4) ? RL[4] : x_b_4);
                unsigned long long x_b_5 = BL[10];
                RL[5] = ((RL[5] > x_b_5) ? RL[5] : x_b_5);
                unsigned long long x_b_6 = BL[9];
                RL[6] = ((RL[6] > x_b_6) ? RL[6] : x_b_6);
                unsigned long long x_b_7 = BL[8];
                RL[7] = ((RL[7] > x_b_7) ? RL[7] : x_b_7);
                unsigned long long x_b_8 = BL[7];
                RL[8] = ((RL[8] > x_b_8) ? RL[8] : x_b_8);
                unsigned long long x_b_9 = BL[6];
                RL[9] = ((RL[9] > x_b_9) ? RL[9] : x_b_9);
                unsigned long long x_b_10 = BL[5];
                RL[10] = ((RL[10] > x_b_10) ? RL[10] : x_b_10);
                unsigned long long x_b_11 = BL[4];
                RL[11] = ((RL[11] > x_b_11) ? RL[11] : x_b_11);
                unsigned long long x_b_12 = BL[3];
                RL[12] = ((RL[12] > x_b_12) ? RL[12] : x_b_12);
                unsigned long long x_b_13 = BL[2];
                RL[13] = ((RL[13] > x_b_13) ? RL[13] : x_b_13);
                unsigned long long x_b_14 = BL[1];
                RL[14] = ((RL[14] > x_b_14) ? RL[14] : x_b_14);
                unsigned long long x_b_15 = BL[0];
                RL[15] = ((RL[15] > x_b_15) ? RL[15] : x_b_15);
                unsigned long long hk_a_0 = RL[0];
                unsigned long long hk_b_0 = RL[8];
                unsigned long long hk_hi_0 = ((hk_a_0 > hk_b_0) ? hk_a_0 : hk_b_0);
                unsigned long long hk_lo_0 = ((hk_a_0 > hk_b_0) ? hk_b_0 : hk_a_0);
                RL[0] = ((odd != 0) ? hk_lo_0 : hk_hi_0);
                unsigned long long hk_a_1 = RL[1];
                unsigned long long hk_b_1 = RL[9];
                unsigned long long hk_hi_1 = ((hk_a_1 > hk_b_1) ? hk_a_1 : hk_b_1);
                unsigned long long hk_lo_1 = ((hk_a_1 > hk_b_1) ? hk_b_1 : hk_a_1);
                RL[1] = ((odd != 0) ? hk_lo_1 : hk_hi_1);
                unsigned long long hk_a_2 = RL[2];
                unsigned long long hk_b_2 = RL[10];
                unsigned long long hk_hi_2 = ((hk_a_2 > hk_b_2) ? hk_a_2 : hk_b_2);
                unsigned long long hk_lo_2 = ((hk_a_2 > hk_b_2) ? hk_b_2 : hk_a_2);
                RL[2] = ((odd != 0) ? hk_lo_2 : hk_hi_2);
                unsigned long long hk_a_3 = RL[3];
                unsigned long long hk_b_3 = RL[11];
                unsigned long long hk_hi_3 = ((hk_a_3 > hk_b_3) ? hk_a_3 : hk_b_3);
                unsigned long long hk_lo_3 = ((hk_a_3 > hk_b_3) ? hk_b_3 : hk_a_3);
                RL[3] = ((odd != 0) ? hk_lo_3 : hk_hi_3);
                unsigned long long hk_a_4 = RL[4];
                unsigned long long hk_b_4 = RL[12];
                unsigned long long hk_hi_4 = ((hk_a_4 > hk_b_4) ? hk_a_4 : hk_b_4);
                unsigned long long hk_lo_4 = ((hk_a_4 > hk_b_4) ? hk_b_4 : hk_a_4);
                RL[4] = ((odd != 0) ? hk_lo_4 : hk_hi_4);
                unsigned long long hk_a_5 = RL[5];
                unsigned long long hk_b_5 = RL[13];
                unsigned long long hk_hi_5 = ((hk_a_5 > hk_b_5) ? hk_a_5 : hk_b_5);
                unsigned long long hk_lo_5 = ((hk_a_5 > hk_b_5) ? hk_b_5 : hk_a_5);
                RL[5] = ((odd != 0) ? hk_lo_5 : hk_hi_5);
                unsigned long long hk_a_6 = RL[6];
                unsigned long long hk_b_6 = RL[14];
                unsigned long long hk_hi_6 = ((hk_a_6 > hk_b_6) ? hk_a_6 : hk_b_6);
                unsigned long long hk_lo_6 = ((hk_a_6 > hk_b_6) ? hk_b_6 : hk_a_6);
                RL[6] = ((odd != 0) ? hk_lo_6 : hk_hi_6);
                unsigned long long hk_a_7 = RL[7];
                unsigned long long hk_b_7 = RL[15];
                unsigned long long hk_hi_7 = ((hk_a_7 > hk_b_7) ? hk_a_7 : hk_b_7);
                unsigned long long hk_lo_7 = ((hk_a_7 > hk_b_7) ? hk_b_7 : hk_a_7);
                RL[7] = ((odd != 0) ? hk_lo_7 : hk_hi_7);
                unsigned long long l8_a_0 = RL[0];
                unsigned long long l8_b_0 = RL[4];
                RL[0] = ((l8_a_0 > l8_b_0) ? l8_a_0 : l8_b_0);
                RL[4] = ((l8_a_0 > l8_b_0) ? l8_b_0 : l8_a_0);
                unsigned long long l8_a_1 = RL[1];
                unsigned long long l8_b_1 = RL[5];
                RL[1] = ((l8_a_1 > l8_b_1) ? l8_a_1 : l8_b_1);
                RL[5] = ((l8_a_1 > l8_b_1) ? l8_b_1 : l8_a_1);
                unsigned long long l8_a_2 = RL[2];
                unsigned long long l8_b_2 = RL[6];
                RL[2] = ((l8_a_2 > l8_b_2) ? l8_a_2 : l8_b_2);
                RL[6] = ((l8_a_2 > l8_b_2) ? l8_b_2 : l8_a_2);
                unsigned long long l8_a_3 = RL[3];
                unsigned long long l8_b_3 = RL[7];
                RL[3] = ((l8_a_3 > l8_b_3) ? l8_a_3 : l8_b_3);
                RL[7] = ((l8_a_3 > l8_b_3) ? l8_b_3 : l8_a_3);
                unsigned long long l8_a_4 = RL[0];
                unsigned long long l8_b_4 = RL[2];
                RL[0] = ((l8_a_4 > l8_b_4) ? l8_a_4 : l8_b_4);
                RL[2] = ((l8_a_4 > l8_b_4) ? l8_b_4 : l8_a_4);
                unsigned long long l8_a_5 = RL[1];
                unsigned long long l8_b_5 = RL[3];
                RL[1] = ((l8_a_5 > l8_b_5) ? l8_a_5 : l8_b_5);
                RL[3] = ((l8_a_5 > l8_b_5) ? l8_b_5 : l8_a_5);
                unsigned long long l8_a_6 = RL[4];
                unsigned long long l8_b_6 = RL[6];
                RL[4] = ((l8_a_6 > l8_b_6) ? l8_a_6 : l8_b_6);
                RL[6] = ((l8_a_6 > l8_b_6) ? l8_b_6 : l8_a_6);
                unsigned long long l8_a_7 = RL[5];
                unsigned long long l8_b_7 = RL[7];
                RL[5] = ((l8_a_7 > l8_b_7) ? l8_a_7 : l8_b_7);
                RL[7] = ((l8_a_7 > l8_b_7) ? l8_b_7 : l8_a_7);
                unsigned long long l8_a_8 = RL[0];
                unsigned long long l8_b_8 = RL[1];
                RL[0] = ((l8_a_8 > l8_b_8) ? l8_a_8 : l8_b_8);
                RL[1] = ((l8_a_8 > l8_b_8) ? l8_b_8 : l8_a_8);
                unsigned long long l8_a_9 = RL[2];
                unsigned long long l8_b_9 = RL[3];
                RL[2] = ((l8_a_9 > l8_b_9) ? l8_a_9 : l8_b_9);
                RL[3] = ((l8_a_9 > l8_b_9) ? l8_b_9 : l8_a_9);
                unsigned long long l8_a_10 = RL[4];
                unsigned long long l8_b_10 = RL[5];
                RL[4] = ((l8_a_10 > l8_b_10) ? l8_a_10 : l8_b_10);
                RL[5] = ((l8_a_10 > l8_b_10) ? l8_b_10 : l8_a_10);
                unsigned long long l8_a_11 = RL[6];
                unsigned long long l8_b_11 = RL[7];
                RL[6] = ((l8_a_11 > l8_b_11) ? l8_a_11 : l8_b_11);
                RL[7] = ((l8_a_11 > l8_b_11) ? l8_b_11 : l8_a_11);
                unsigned long long y_v_0 = RL[7];
                unsigned int y_lo_0 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)y_v_0, 3);
                unsigned int y_hi_0 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(y_v_0 >> 32), 3);
                BL[0] = (unsigned long long)y_hi_0 << 32 | (unsigned long long)y_lo_0;
                unsigned long long y_v_1 = RL[6];
                unsigned int y_lo_1 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)y_v_1, 3);
                unsigned int y_hi_1 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(y_v_1 >> 32), 3);
                BL[1] = (unsigned long long)y_hi_1 << 32 | (unsigned long long)y_lo_1;
                unsigned long long y_v_2 = RL[5];
                unsigned int y_lo_2 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)y_v_2, 3);
                unsigned int y_hi_2 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(y_v_2 >> 32), 3);
                BL[2] = (unsigned long long)y_hi_2 << 32 | (unsigned long long)y_lo_2;
                unsigned long long y_v_3 = RL[4];
                unsigned int y_lo_3 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)y_v_3, 3);
                unsigned int y_hi_3 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(y_v_3 >> 32), 3);
                BL[3] = (unsigned long long)y_hi_3 << 32 | (unsigned long long)y_lo_3;
                unsigned long long y_v_4 = RL[3];
                unsigned int y_lo_4 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)y_v_4, 3);
                unsigned int y_hi_4 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(y_v_4 >> 32), 3);
                BL[4] = (unsigned long long)y_hi_4 << 32 | (unsigned long long)y_lo_4;
                unsigned long long y_v_5 = RL[2];
                unsigned int y_lo_5 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)y_v_5, 3);
                unsigned int y_hi_5 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(y_v_5 >> 32), 3);
                BL[5] = (unsigned long long)y_hi_5 << 32 | (unsigned long long)y_lo_5;
                unsigned long long y_v_6 = RL[1];
                unsigned int y_lo_6 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)y_v_6, 3);
                unsigned int y_hi_6 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(y_v_6 >> 32), 3);
                BL[6] = (unsigned long long)y_hi_6 << 32 | (unsigned long long)y_lo_6;
                unsigned long long y_v_7 = RL[0];
                unsigned int y_lo_7 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)y_v_7, 3);
                unsigned int y_hi_7 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(y_v_7 >> 32), 3);
                BL[7] = (unsigned long long)y_hi_7 << 32 | (unsigned long long)y_lo_7;
                unsigned long long y_o_0 = BL[0];
                RL[0] = ((RL[0] > y_o_0) ? RL[0] : y_o_0);
                unsigned long long y_o_1 = BL[1];
                RL[1] = ((RL[1] > y_o_1) ? RL[1] : y_o_1);
                unsigned long long y_o_2 = BL[2];
                RL[2] = ((RL[2] > y_o_2) ? RL[2] : y_o_2);
                unsigned long long y_o_3 = BL[3];
                RL[3] = ((RL[3] > y_o_3) ? RL[3] : y_o_3);
                unsigned long long y_o_4 = BL[4];
                RL[4] = ((RL[4] > y_o_4) ? RL[4] : y_o_4);
                unsigned long long y_o_5 = BL[5];
                RL[5] = ((RL[5] > y_o_5) ? RL[5] : y_o_5);
                unsigned long long y_o_6 = BL[6];
                RL[6] = ((RL[6] > y_o_6) ? RL[6] : y_o_6);
                unsigned long long y_o_7 = BL[7];
                RL[7] = ((RL[7] > y_o_7) ? RL[7] : y_o_7);
                unsigned long long z_v_0 = RL[0];
                unsigned int z_lo_0 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)z_v_0, 1);
                unsigned int z_hi_0 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(z_v_0 >> 32), 1);
                unsigned long long z_o_0 = (unsigned long long)z_hi_0 << 32 | (unsigned long long)z_lo_0;
                unsigned long long z_hi64_0 = ((z_v_0 > z_o_0) ? z_v_0 : z_o_0);
                unsigned long long z_lo64_0 = ((z_v_0 > z_o_0) ? z_o_0 : z_v_0);
                RL[0] = ((odd != 0) ? z_lo64_0 : z_hi64_0);
                unsigned long long z_v_1 = RL[1];
                unsigned int z_lo_1 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)z_v_1, 1);
                unsigned int z_hi_1 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(z_v_1 >> 32), 1);
                unsigned long long z_o_1 = (unsigned long long)z_hi_1 << 32 | (unsigned long long)z_lo_1;
                unsigned long long z_hi64_1 = ((z_v_1 > z_o_1) ? z_v_1 : z_o_1);
                unsigned long long z_lo64_1 = ((z_v_1 > z_o_1) ? z_o_1 : z_v_1);
                RL[1] = ((odd != 0) ? z_lo64_1 : z_hi64_1);
                unsigned long long z_v_2 = RL[2];
                unsigned int z_lo_2 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)z_v_2, 1);
                unsigned int z_hi_2 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(z_v_2 >> 32), 1);
                unsigned long long z_o_2 = (unsigned long long)z_hi_2 << 32 | (unsigned long long)z_lo_2;
                unsigned long long z_hi64_2 = ((z_v_2 > z_o_2) ? z_v_2 : z_o_2);
                unsigned long long z_lo64_2 = ((z_v_2 > z_o_2) ? z_o_2 : z_v_2);
                RL[2] = ((odd != 0) ? z_lo64_2 : z_hi64_2);
                unsigned long long z_v_3 = RL[3];
                unsigned int z_lo_3 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)z_v_3, 1);
                unsigned int z_hi_3 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(z_v_3 >> 32), 1);
                unsigned long long z_o_3 = (unsigned long long)z_hi_3 << 32 | (unsigned long long)z_lo_3;
                unsigned long long z_hi64_3 = ((z_v_3 > z_o_3) ? z_v_3 : z_o_3);
                unsigned long long z_lo64_3 = ((z_v_3 > z_o_3) ? z_o_3 : z_v_3);
                RL[3] = ((odd != 0) ? z_lo64_3 : z_hi64_3);
                unsigned long long z_v_4 = RL[4];
                unsigned int z_lo_4 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)z_v_4, 1);
                unsigned int z_hi_4 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(z_v_4 >> 32), 1);
                unsigned long long z_o_4 = (unsigned long long)z_hi_4 << 32 | (unsigned long long)z_lo_4;
                unsigned long long z_hi64_4 = ((z_v_4 > z_o_4) ? z_v_4 : z_o_4);
                unsigned long long z_lo64_4 = ((z_v_4 > z_o_4) ? z_o_4 : z_v_4);
                RL[4] = ((odd != 0) ? z_lo64_4 : z_hi64_4);
                unsigned long long z_v_5 = RL[5];
                unsigned int z_lo_5 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)z_v_5, 1);
                unsigned int z_hi_5 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(z_v_5 >> 32), 1);
                unsigned long long z_o_5 = (unsigned long long)z_hi_5 << 32 | (unsigned long long)z_lo_5;
                unsigned long long z_hi64_5 = ((z_v_5 > z_o_5) ? z_v_5 : z_o_5);
                unsigned long long z_lo64_5 = ((z_v_5 > z_o_5) ? z_o_5 : z_v_5);
                RL[5] = ((odd != 0) ? z_lo64_5 : z_hi64_5);
                unsigned long long z_v_6 = RL[6];
                unsigned int z_lo_6 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)z_v_6, 1);
                unsigned int z_hi_6 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(z_v_6 >> 32), 1);
                unsigned long long z_o_6 = (unsigned long long)z_hi_6 << 32 | (unsigned long long)z_lo_6;
                unsigned long long z_hi64_6 = ((z_v_6 > z_o_6) ? z_v_6 : z_o_6);
                unsigned long long z_lo64_6 = ((z_v_6 > z_o_6) ? z_o_6 : z_v_6);
                RL[6] = ((odd != 0) ? z_lo64_6 : z_hi64_6);
                unsigned long long z_v_7 = RL[7];
                unsigned int z_lo_7 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)z_v_7, 1);
                unsigned int z_hi_7 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(z_v_7 >> 32), 1);
                unsigned long long z_o_7 = (unsigned long long)z_hi_7 << 32 | (unsigned long long)z_lo_7;
                unsigned long long z_hi64_7 = ((z_v_7 > z_o_7) ? z_v_7 : z_o_7);
                unsigned long long z_lo64_7 = ((z_v_7 > z_o_7) ? z_o_7 : z_v_7);
                RL[7] = ((odd != 0) ? z_lo64_7 : z_hi64_7);
                unsigned long long hk_a_8 = RL[0];
                unsigned long long hk_b_8 = RL[4];
                unsigned long long hk_hi_8 = ((hk_a_8 > hk_b_8) ? hk_a_8 : hk_b_8);
                unsigned long long hk_lo_8 = ((hk_a_8 > hk_b_8) ? hk_b_8 : hk_a_8);
                RL[0] = (((sl >> 1 & 1) != 0) ? hk_lo_8 : hk_hi_8);
                unsigned long long hk_a_9 = RL[1];
                unsigned long long hk_b_9 = RL[5];
                unsigned long long hk_hi_9 = ((hk_a_9 > hk_b_9) ? hk_a_9 : hk_b_9);
                unsigned long long hk_lo_9 = ((hk_a_9 > hk_b_9) ? hk_b_9 : hk_a_9);
                RL[1] = (((sl >> 1 & 1) != 0) ? hk_lo_9 : hk_hi_9);
                unsigned long long hk_a_10 = RL[2];
                unsigned long long hk_b_10 = RL[6];
                unsigned long long hk_hi_10 = ((hk_a_10 > hk_b_10) ? hk_a_10 : hk_b_10);
                unsigned long long hk_lo_10 = ((hk_a_10 > hk_b_10) ? hk_b_10 : hk_a_10);
                RL[2] = (((sl >> 1 & 1) != 0) ? hk_lo_10 : hk_hi_10);
                unsigned long long hk_a_11 = RL[3];
                unsigned long long hk_b_11 = RL[7];
                unsigned long long hk_hi_11 = ((hk_a_11 > hk_b_11) ? hk_a_11 : hk_b_11);
                unsigned long long hk_lo_11 = ((hk_a_11 > hk_b_11) ? hk_b_11 : hk_a_11);
                RL[3] = (((sl >> 1 & 1) != 0) ? hk_lo_11 : hk_hi_11);
                unsigned long long l4_a_0 = RL[0];
                unsigned long long l4_b_0 = RL[2];
                RL[0] = ((l4_a_0 > l4_b_0) ? l4_a_0 : l4_b_0);
                RL[2] = ((l4_a_0 > l4_b_0) ? l4_b_0 : l4_a_0);
                unsigned long long l4_a_1 = RL[1];
                unsigned long long l4_b_1 = RL[3];
                RL[1] = ((l4_a_1 > l4_b_1) ? l4_a_1 : l4_b_1);
                RL[3] = ((l4_a_1 > l4_b_1) ? l4_b_1 : l4_a_1);
                unsigned long long l4_a_2 = RL[0];
                unsigned long long l4_b_2 = RL[1];
                RL[0] = ((l4_a_2 > l4_b_2) ? l4_a_2 : l4_b_2);
                RL[1] = ((l4_a_2 > l4_b_2) ? l4_b_2 : l4_a_2);
                unsigned long long l4_a_3 = RL[2];
                unsigned long long l4_b_3 = RL[3];
                RL[2] = ((l4_a_3 > l4_b_3) ? l4_a_3 : l4_b_3);
                RL[3] = ((l4_a_3 > l4_b_3) ? l4_b_3 : l4_a_3);
                unsigned long long f_v_0 = RL[3];
                unsigned int f_lo_0 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)f_v_0, 7);
                unsigned int f_hi_0 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(f_v_0 >> 32), 7);
                unsigned long long f_o_0 = (unsigned long long)f_hi_0 << 32 | (unsigned long long)f_lo_0;
                R[0] = ((RL[0] > f_o_0) ? RL[0] : f_o_0);
                unsigned long long f_v_1 = RL[2];
                unsigned int f_lo_1 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)f_v_1, 7);
                unsigned int f_hi_1 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(f_v_1 >> 32), 7);
                unsigned long long f_o_1 = (unsigned long long)f_hi_1 << 32 | (unsigned long long)f_lo_1;
                R[1] = ((RL[1] > f_o_1) ? RL[1] : f_o_1);
            } else {
                unsigned long long A1[2];
                unsigned long long A2[2];
                unsigned long long A3[2];
                unsigned long long L0[2];
                unsigned long long L1[2];
                unsigned long long L2[2];
                unsigned long long L3[2];
                int C[4];
                A1[0] = 0;
                A2[0] = 0;
                A3[0] = 0;
                A1[1] = 0;
                A2[1] = 0;
                A3[1] = 0;
                int used = 0;
                #pragma unroll 1
                for (int c0 = 0; c0 < npieces; c0 += 32) {
                    int cs = s_first + c0 + lane_1;
                    long long cw = work_range[((cs <= s_last) ? cs : s_last)];
                    int c_cb = (((int)(cw & 65535) == j) ? (int)(cw >> 32 & 65535) : 0);
                    int c_ce = (((int)(cw >> 16 & 65535) == j) ? (int)(cw >> 48 & 65535) : cj);
                    unsigned int mask = __ballot_sync(0xFFFFFFFF, cs <= s_last && c_ce > c_cb);
                    int n_steps = ((int)(unsigned int)__popc(mask) + 4 - 1 >> 2);
                    #pragma unroll 1
                    for (int _st = 0; _st < n_steps; _st++) {
                        int taken = 0;
                        C[0] = s_first;
                        if (mask != 0) {
                            int bt_0;
                            asm volatile("bfind.u32 %0, %1;" : "=r"(bt_0) : "r"(mask));
                            C[0] = s_first + c0 + bt_0;
                            mask = mask & ~(1 << (unsigned int)bt_0);
                            taken = taken + 1;
                        }
                        C[1] = s_first;
                        if (mask != 0) {
                            int bt_1;
                            asm volatile("bfind.u32 %0, %1;" : "=r"(bt_1) : "r"(mask));
                            C[1] = s_first + c0 + bt_1;
                            mask = mask & ~(1 << (unsigned int)bt_1);
                            taken = taken + 1;
                        }
                        C[2] = s_first;
                        if (mask != 0) {
                            int bt_2;
                            asm volatile("bfind.u32 %0, %1;" : "=r"(bt_2) : "r"(mask));
                            C[2] = s_first + c0 + bt_2;
                            mask = mask & ~(1 << (unsigned int)bt_2);
                            taken = taken + 1;
                        }
                        C[3] = s_first;
                        if (mask != 0) {
                            int bt_3;
                            asm volatile("bfind.u32 %0, %1;" : "=r"(bt_3) : "r"(mask));
                            C[3] = s_first + c0 + bt_3;
                            mask = mask & ~(1 << (unsigned int)bt_3);
                            taken = taken + 1;
                        }
                        if (0 < taken) {
                            if (0 < used) {
                                float* lk_src = reinterpret_cast<float*>(pool + ((long long)(C[0] + j) * pool_piece_stride + row_rev));
                                float lk_f[4];
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(lk_src + 0);
                                    lk_f[0 + 0] = _v4.x;
                                    lk_f[0 + 1] = _v4.y;
                                    lk_f[0 + 2] = _v4.z;
                                    lk_f[0 + 3] = _v4.w;
                                }
                                unsigned int lk_lo_0 = 0;
                                unsigned int lk_hi_0 = 0;
                                float _bit_cast_src_156 = lk_f[0];
                                lk_lo_0 = reinterpret_cast<unsigned int*>(&_bit_cast_src_156)[0];
                                float _bit_cast_src_157 = lk_f[1];
                                lk_hi_0 = reinterpret_cast<unsigned int*>(&_bit_cast_src_157)[0];
                                L0[0] = (unsigned long long)lk_hi_0 << 32 | (unsigned long long)lk_lo_0;
                                unsigned int lk_lo_1 = 0;
                                unsigned int lk_hi_1 = 0;
                                float _bit_cast_src_158 = lk_f[2];
                                lk_lo_1 = reinterpret_cast<unsigned int*>(&_bit_cast_src_158)[0];
                                float _bit_cast_src_159 = lk_f[3];
                                lk_hi_1 = reinterpret_cast<unsigned int*>(&_bit_cast_src_159)[0];
                                L0[1] = (unsigned long long)lk_hi_1 << 32 | (unsigned long long)lk_lo_1;
                            } else {
                                float* lk_src_1 = reinterpret_cast<float*>(pool + ((long long)(C[0] + j) * pool_piece_stride + row_dir));
                                float lk_f_1[4];
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(lk_src_1 + 0);
                                    lk_f_1[0 + 0] = _v4.x;
                                    lk_f_1[0 + 1] = _v4.y;
                                    lk_f_1[0 + 2] = _v4.z;
                                    lk_f_1[0 + 3] = _v4.w;
                                }
                                unsigned int lk_lo_2 = 0;
                                unsigned int lk_hi_2 = 0;
                                float _bit_cast_src_161 = lk_f_1[0];
                                lk_lo_2 = reinterpret_cast<unsigned int*>(&_bit_cast_src_161)[0];
                                float _bit_cast_src_162 = lk_f_1[1];
                                lk_hi_2 = reinterpret_cast<unsigned int*>(&_bit_cast_src_162)[0];
                                R[0] = (unsigned long long)lk_hi_2 << 32 | (unsigned long long)lk_lo_2;
                                unsigned int lk_lo_3 = 0;
                                unsigned int lk_hi_3 = 0;
                                float _bit_cast_src_163 = lk_f_1[2];
                                lk_lo_3 = reinterpret_cast<unsigned int*>(&_bit_cast_src_163)[0];
                                float _bit_cast_src_164 = lk_f_1[3];
                                lk_hi_3 = reinterpret_cast<unsigned int*>(&_bit_cast_src_164)[0];
                                R[1] = (unsigned long long)lk_hi_3 << 32 | (unsigned long long)lk_lo_3;
                            }
                        }
                        if (1 < taken) {
                            if (1 < used) {
                                float* lk_src_2 = reinterpret_cast<float*>(pool + ((long long)(C[1] + j) * pool_piece_stride + row_rev));
                                float lk_f_2[4];
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(lk_src_2 + 0);
                                    lk_f_2[0 + 0] = _v4.x;
                                    lk_f_2[0 + 1] = _v4.y;
                                    lk_f_2[0 + 2] = _v4.z;
                                    lk_f_2[0 + 3] = _v4.w;
                                }
                                unsigned int lk_lo_4 = 0;
                                unsigned int lk_hi_4 = 0;
                                float _bit_cast_src_166 = lk_f_2[0];
                                lk_lo_4 = reinterpret_cast<unsigned int*>(&_bit_cast_src_166)[0];
                                float _bit_cast_src_167 = lk_f_2[1];
                                lk_hi_4 = reinterpret_cast<unsigned int*>(&_bit_cast_src_167)[0];
                                L1[0] = (unsigned long long)lk_hi_4 << 32 | (unsigned long long)lk_lo_4;
                                unsigned int lk_lo_5 = 0;
                                unsigned int lk_hi_5 = 0;
                                float _bit_cast_src_168 = lk_f_2[2];
                                lk_lo_5 = reinterpret_cast<unsigned int*>(&_bit_cast_src_168)[0];
                                float _bit_cast_src_169 = lk_f_2[3];
                                lk_hi_5 = reinterpret_cast<unsigned int*>(&_bit_cast_src_169)[0];
                                L1[1] = (unsigned long long)lk_hi_5 << 32 | (unsigned long long)lk_lo_5;
                            } else {
                                float* lk_src_3 = reinterpret_cast<float*>(pool + ((long long)(C[1] + j) * pool_piece_stride + row_dir));
                                float lk_f_3[4];
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(lk_src_3 + 0);
                                    lk_f_3[0 + 0] = _v4.x;
                                    lk_f_3[0 + 1] = _v4.y;
                                    lk_f_3[0 + 2] = _v4.z;
                                    lk_f_3[0 + 3] = _v4.w;
                                }
                                unsigned int lk_lo_6 = 0;
                                unsigned int lk_hi_6 = 0;
                                float _bit_cast_src_171 = lk_f_3[0];
                                lk_lo_6 = reinterpret_cast<unsigned int*>(&_bit_cast_src_171)[0];
                                float _bit_cast_src_172 = lk_f_3[1];
                                lk_hi_6 = reinterpret_cast<unsigned int*>(&_bit_cast_src_172)[0];
                                A1[0] = (unsigned long long)lk_hi_6 << 32 | (unsigned long long)lk_lo_6;
                                unsigned int lk_lo_7 = 0;
                                unsigned int lk_hi_7 = 0;
                                float _bit_cast_src_173 = lk_f_3[2];
                                lk_lo_7 = reinterpret_cast<unsigned int*>(&_bit_cast_src_173)[0];
                                float _bit_cast_src_174 = lk_f_3[3];
                                lk_hi_7 = reinterpret_cast<unsigned int*>(&_bit_cast_src_174)[0];
                                A1[1] = (unsigned long long)lk_hi_7 << 32 | (unsigned long long)lk_lo_7;
                            }
                        }
                        if (2 < taken) {
                            if (2 < used) {
                                float* lk_src_4 = reinterpret_cast<float*>(pool + ((long long)(C[2] + j) * pool_piece_stride + row_rev));
                                float lk_f_4[4];
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(lk_src_4 + 0);
                                    lk_f_4[0 + 0] = _v4.x;
                                    lk_f_4[0 + 1] = _v4.y;
                                    lk_f_4[0 + 2] = _v4.z;
                                    lk_f_4[0 + 3] = _v4.w;
                                }
                                unsigned int lk_lo_8 = 0;
                                unsigned int lk_hi_8 = 0;
                                float _bit_cast_src_176 = lk_f_4[0];
                                lk_lo_8 = reinterpret_cast<unsigned int*>(&_bit_cast_src_176)[0];
                                float _bit_cast_src_177 = lk_f_4[1];
                                lk_hi_8 = reinterpret_cast<unsigned int*>(&_bit_cast_src_177)[0];
                                L2[0] = (unsigned long long)lk_hi_8 << 32 | (unsigned long long)lk_lo_8;
                                unsigned int lk_lo_9 = 0;
                                unsigned int lk_hi_9 = 0;
                                float _bit_cast_src_178 = lk_f_4[2];
                                lk_lo_9 = reinterpret_cast<unsigned int*>(&_bit_cast_src_178)[0];
                                float _bit_cast_src_179 = lk_f_4[3];
                                lk_hi_9 = reinterpret_cast<unsigned int*>(&_bit_cast_src_179)[0];
                                L2[1] = (unsigned long long)lk_hi_9 << 32 | (unsigned long long)lk_lo_9;
                            } else {
                                float* lk_src_5 = reinterpret_cast<float*>(pool + ((long long)(C[2] + j) * pool_piece_stride + row_dir));
                                float lk_f_5[4];
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(lk_src_5 + 0);
                                    lk_f_5[0 + 0] = _v4.x;
                                    lk_f_5[0 + 1] = _v4.y;
                                    lk_f_5[0 + 2] = _v4.z;
                                    lk_f_5[0 + 3] = _v4.w;
                                }
                                unsigned int lk_lo_10 = 0;
                                unsigned int lk_hi_10 = 0;
                                float _bit_cast_src_181 = lk_f_5[0];
                                lk_lo_10 = reinterpret_cast<unsigned int*>(&_bit_cast_src_181)[0];
                                float _bit_cast_src_182 = lk_f_5[1];
                                lk_hi_10 = reinterpret_cast<unsigned int*>(&_bit_cast_src_182)[0];
                                A2[0] = (unsigned long long)lk_hi_10 << 32 | (unsigned long long)lk_lo_10;
                                unsigned int lk_lo_11 = 0;
                                unsigned int lk_hi_11 = 0;
                                float _bit_cast_src_183 = lk_f_5[2];
                                lk_lo_11 = reinterpret_cast<unsigned int*>(&_bit_cast_src_183)[0];
                                float _bit_cast_src_184 = lk_f_5[3];
                                lk_hi_11 = reinterpret_cast<unsigned int*>(&_bit_cast_src_184)[0];
                                A2[1] = (unsigned long long)lk_hi_11 << 32 | (unsigned long long)lk_lo_11;
                            }
                        }
                        if (3 < taken) {
                            if (3 < used) {
                                float* lk_src_6 = reinterpret_cast<float*>(pool + ((long long)(C[3] + j) * pool_piece_stride + row_rev));
                                float lk_f_6[4];
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(lk_src_6 + 0);
                                    lk_f_6[0 + 0] = _v4.x;
                                    lk_f_6[0 + 1] = _v4.y;
                                    lk_f_6[0 + 2] = _v4.z;
                                    lk_f_6[0 + 3] = _v4.w;
                                }
                                unsigned int lk_lo_12 = 0;
                                unsigned int lk_hi_12 = 0;
                                float _bit_cast_src_186 = lk_f_6[0];
                                lk_lo_12 = reinterpret_cast<unsigned int*>(&_bit_cast_src_186)[0];
                                float _bit_cast_src_187 = lk_f_6[1];
                                lk_hi_12 = reinterpret_cast<unsigned int*>(&_bit_cast_src_187)[0];
                                L3[0] = (unsigned long long)lk_hi_12 << 32 | (unsigned long long)lk_lo_12;
                                unsigned int lk_lo_13 = 0;
                                unsigned int lk_hi_13 = 0;
                                float _bit_cast_src_188 = lk_f_6[2];
                                lk_lo_13 = reinterpret_cast<unsigned int*>(&_bit_cast_src_188)[0];
                                float _bit_cast_src_189 = lk_f_6[3];
                                lk_hi_13 = reinterpret_cast<unsigned int*>(&_bit_cast_src_189)[0];
                                L3[1] = (unsigned long long)lk_hi_13 << 32 | (unsigned long long)lk_lo_13;
                            } else {
                                float* lk_src_7 = reinterpret_cast<float*>(pool + ((long long)(C[3] + j) * pool_piece_stride + row_dir));
                                float lk_f_7[4];
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(lk_src_7 + 0);
                                    lk_f_7[0 + 0] = _v4.x;
                                    lk_f_7[0 + 1] = _v4.y;
                                    lk_f_7[0 + 2] = _v4.z;
                                    lk_f_7[0 + 3] = _v4.w;
                                }
                                unsigned int lk_lo_14 = 0;
                                unsigned int lk_hi_14 = 0;
                                float _bit_cast_src_191 = lk_f_7[0];
                                lk_lo_14 = reinterpret_cast<unsigned int*>(&_bit_cast_src_191)[0];
                                float _bit_cast_src_192 = lk_f_7[1];
                                lk_hi_14 = reinterpret_cast<unsigned int*>(&_bit_cast_src_192)[0];
                                A3[0] = (unsigned long long)lk_hi_14 << 32 | (unsigned long long)lk_lo_14;
                                unsigned int lk_lo_15 = 0;
                                unsigned int lk_hi_15 = 0;
                                float _bit_cast_src_193 = lk_f_7[2];
                                lk_lo_15 = reinterpret_cast<unsigned int*>(&_bit_cast_src_193)[0];
                                float _bit_cast_src_194 = lk_f_7[3];
                                lk_hi_15 = reinterpret_cast<unsigned int*>(&_bit_cast_src_194)[0];
                                A3[1] = (unsigned long long)lk_hi_15 << 32 | (unsigned long long)lk_lo_15;
                            }
                        }
                        if (0 < taken && 0 < used) {
                            R[0] = ((R[0] > L0[1]) ? R[0] : L0[1]);
                            R[1] = ((R[1] > L0[0]) ? R[1] : L0[0]);
                            unsigned long long hm_x_0 = R[0];
                            unsigned int hm_lo_0 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_0, 4);
                            unsigned int hm_hi_0 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_0 >> 32), 4);
                            unsigned long long hm_o_0 = (unsigned long long)hm_hi_0 << 32 | (unsigned long long)hm_lo_0;
                            unsigned long long hm_max_0 = ((hm_x_0 > hm_o_0) ? hm_x_0 : hm_o_0);
                            unsigned long long hm_min_0 = ((hm_x_0 > hm_o_0) ? hm_o_0 : hm_x_0);
                            R[0] = (((sl & 4) == 0) ? hm_max_0 : hm_min_0);
                            unsigned long long hm_x_1 = R[1];
                            unsigned int hm_lo_1 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_1, 4);
                            unsigned int hm_hi_1 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_1 >> 32), 4);
                            unsigned long long hm_o_1 = (unsigned long long)hm_hi_1 << 32 | (unsigned long long)hm_lo_1;
                            unsigned long long hm_max_1 = ((hm_x_1 > hm_o_1) ? hm_x_1 : hm_o_1);
                            unsigned long long hm_min_1 = ((hm_x_1 > hm_o_1) ? hm_o_1 : hm_x_1);
                            R[1] = (((sl & 4) == 0) ? hm_max_1 : hm_min_1);
                            unsigned long long hm_x_2 = R[0];
                            unsigned int hm_lo_2 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_2, 2);
                            unsigned int hm_hi_2 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_2 >> 32), 2);
                            unsigned long long hm_o_2 = (unsigned long long)hm_hi_2 << 32 | (unsigned long long)hm_lo_2;
                            unsigned long long hm_max_2 = ((hm_x_2 > hm_o_2) ? hm_x_2 : hm_o_2);
                            unsigned long long hm_min_2 = ((hm_x_2 > hm_o_2) ? hm_o_2 : hm_x_2);
                            R[0] = (((sl & 2) == 0) ? hm_max_2 : hm_min_2);
                            unsigned long long hm_x_3 = R[1];
                            unsigned int hm_lo_3 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_3, 2);
                            unsigned int hm_hi_3 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_3 >> 32), 2);
                            unsigned long long hm_o_3 = (unsigned long long)hm_hi_3 << 32 | (unsigned long long)hm_lo_3;
                            unsigned long long hm_max_3 = ((hm_x_3 > hm_o_3) ? hm_x_3 : hm_o_3);
                            unsigned long long hm_min_3 = ((hm_x_3 > hm_o_3) ? hm_o_3 : hm_x_3);
                            R[1] = (((sl & 2) == 0) ? hm_max_3 : hm_min_3);
                            unsigned long long hm_x_4 = R[0];
                            unsigned int hm_lo_4 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_4, 1);
                            unsigned int hm_hi_4 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_4 >> 32), 1);
                            unsigned long long hm_o_4 = (unsigned long long)hm_hi_4 << 32 | (unsigned long long)hm_lo_4;
                            unsigned long long hm_max_4 = ((hm_x_4 > hm_o_4) ? hm_x_4 : hm_o_4);
                            unsigned long long hm_min_4 = ((hm_x_4 > hm_o_4) ? hm_o_4 : hm_x_4);
                            R[0] = (((sl & 1) == 0) ? hm_max_4 : hm_min_4);
                            unsigned long long hm_x_5 = R[1];
                            unsigned int hm_lo_5 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_5, 1);
                            unsigned int hm_hi_5 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_5 >> 32), 1);
                            unsigned long long hm_o_5 = (unsigned long long)hm_hi_5 << 32 | (unsigned long long)hm_lo_5;
                            unsigned long long hm_max_5 = ((hm_x_5 > hm_o_5) ? hm_x_5 : hm_o_5);
                            unsigned long long hm_min_5 = ((hm_x_5 > hm_o_5) ? hm_o_5 : hm_x_5);
                            R[1] = (((sl & 1) == 0) ? hm_max_5 : hm_min_5);
                            unsigned long long hm_a = R[0];
                            unsigned long long hm_b = R[1];
                            R[0] = ((hm_a > hm_b) ? hm_a : hm_b);
                            R[1] = ((hm_a > hm_b) ? hm_b : hm_a);
                        }
                        if (1 < taken && 1 < used) {
                            A1[0] = ((A1[0] > L1[1]) ? A1[0] : L1[1]);
                            A1[1] = ((A1[1] > L1[0]) ? A1[1] : L1[0]);
                            unsigned long long hm_x_6 = A1[0];
                            unsigned int hm_lo_6 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_6, 4);
                            unsigned int hm_hi_6 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_6 >> 32), 4);
                            unsigned long long hm_o_6 = (unsigned long long)hm_hi_6 << 32 | (unsigned long long)hm_lo_6;
                            unsigned long long hm_max_6 = ((hm_x_6 > hm_o_6) ? hm_x_6 : hm_o_6);
                            unsigned long long hm_min_6 = ((hm_x_6 > hm_o_6) ? hm_o_6 : hm_x_6);
                            A1[0] = (((sl & 4) == 0) ? hm_max_6 : hm_min_6);
                            unsigned long long hm_x_7 = A1[1];
                            unsigned int hm_lo_7 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_7, 4);
                            unsigned int hm_hi_7 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_7 >> 32), 4);
                            unsigned long long hm_o_7 = (unsigned long long)hm_hi_7 << 32 | (unsigned long long)hm_lo_7;
                            unsigned long long hm_max_7 = ((hm_x_7 > hm_o_7) ? hm_x_7 : hm_o_7);
                            unsigned long long hm_min_7 = ((hm_x_7 > hm_o_7) ? hm_o_7 : hm_x_7);
                            A1[1] = (((sl & 4) == 0) ? hm_max_7 : hm_min_7);
                            unsigned long long hm_x_8 = A1[0];
                            unsigned int hm_lo_8 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_8, 2);
                            unsigned int hm_hi_8 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_8 >> 32), 2);
                            unsigned long long hm_o_8 = (unsigned long long)hm_hi_8 << 32 | (unsigned long long)hm_lo_8;
                            unsigned long long hm_max_8 = ((hm_x_8 > hm_o_8) ? hm_x_8 : hm_o_8);
                            unsigned long long hm_min_8 = ((hm_x_8 > hm_o_8) ? hm_o_8 : hm_x_8);
                            A1[0] = (((sl & 2) == 0) ? hm_max_8 : hm_min_8);
                            unsigned long long hm_x_9 = A1[1];
                            unsigned int hm_lo_9 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_9, 2);
                            unsigned int hm_hi_9 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_9 >> 32), 2);
                            unsigned long long hm_o_9 = (unsigned long long)hm_hi_9 << 32 | (unsigned long long)hm_lo_9;
                            unsigned long long hm_max_9 = ((hm_x_9 > hm_o_9) ? hm_x_9 : hm_o_9);
                            unsigned long long hm_min_9 = ((hm_x_9 > hm_o_9) ? hm_o_9 : hm_x_9);
                            A1[1] = (((sl & 2) == 0) ? hm_max_9 : hm_min_9);
                            unsigned long long hm_x_10 = A1[0];
                            unsigned int hm_lo_10 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_10, 1);
                            unsigned int hm_hi_10 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_10 >> 32), 1);
                            unsigned long long hm_o_10 = (unsigned long long)hm_hi_10 << 32 | (unsigned long long)hm_lo_10;
                            unsigned long long hm_max_10 = ((hm_x_10 > hm_o_10) ? hm_x_10 : hm_o_10);
                            unsigned long long hm_min_10 = ((hm_x_10 > hm_o_10) ? hm_o_10 : hm_x_10);
                            A1[0] = (((sl & 1) == 0) ? hm_max_10 : hm_min_10);
                            unsigned long long hm_x_11 = A1[1];
                            unsigned int hm_lo_11 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_11, 1);
                            unsigned int hm_hi_11 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_11 >> 32), 1);
                            unsigned long long hm_o_11 = (unsigned long long)hm_hi_11 << 32 | (unsigned long long)hm_lo_11;
                            unsigned long long hm_max_11 = ((hm_x_11 > hm_o_11) ? hm_x_11 : hm_o_11);
                            unsigned long long hm_min_11 = ((hm_x_11 > hm_o_11) ? hm_o_11 : hm_x_11);
                            A1[1] = (((sl & 1) == 0) ? hm_max_11 : hm_min_11);
                            unsigned long long hm_a_1 = A1[0];
                            unsigned long long hm_b_1 = A1[1];
                            A1[0] = ((hm_a_1 > hm_b_1) ? hm_a_1 : hm_b_1);
                            A1[1] = ((hm_a_1 > hm_b_1) ? hm_b_1 : hm_a_1);
                        }
                        if (2 < taken && 2 < used) {
                            A2[0] = ((A2[0] > L2[1]) ? A2[0] : L2[1]);
                            A2[1] = ((A2[1] > L2[0]) ? A2[1] : L2[0]);
                            unsigned long long hm_x_12 = A2[0];
                            unsigned int hm_lo_12 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_12, 4);
                            unsigned int hm_hi_12 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_12 >> 32), 4);
                            unsigned long long hm_o_12 = (unsigned long long)hm_hi_12 << 32 | (unsigned long long)hm_lo_12;
                            unsigned long long hm_max_12 = ((hm_x_12 > hm_o_12) ? hm_x_12 : hm_o_12);
                            unsigned long long hm_min_12 = ((hm_x_12 > hm_o_12) ? hm_o_12 : hm_x_12);
                            A2[0] = (((sl & 4) == 0) ? hm_max_12 : hm_min_12);
                            unsigned long long hm_x_13 = A2[1];
                            unsigned int hm_lo_13 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_13, 4);
                            unsigned int hm_hi_13 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_13 >> 32), 4);
                            unsigned long long hm_o_13 = (unsigned long long)hm_hi_13 << 32 | (unsigned long long)hm_lo_13;
                            unsigned long long hm_max_13 = ((hm_x_13 > hm_o_13) ? hm_x_13 : hm_o_13);
                            unsigned long long hm_min_13 = ((hm_x_13 > hm_o_13) ? hm_o_13 : hm_x_13);
                            A2[1] = (((sl & 4) == 0) ? hm_max_13 : hm_min_13);
                            unsigned long long hm_x_14 = A2[0];
                            unsigned int hm_lo_14 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_14, 2);
                            unsigned int hm_hi_14 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_14 >> 32), 2);
                            unsigned long long hm_o_14 = (unsigned long long)hm_hi_14 << 32 | (unsigned long long)hm_lo_14;
                            unsigned long long hm_max_14 = ((hm_x_14 > hm_o_14) ? hm_x_14 : hm_o_14);
                            unsigned long long hm_min_14 = ((hm_x_14 > hm_o_14) ? hm_o_14 : hm_x_14);
                            A2[0] = (((sl & 2) == 0) ? hm_max_14 : hm_min_14);
                            unsigned long long hm_x_15 = A2[1];
                            unsigned int hm_lo_15 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_15, 2);
                            unsigned int hm_hi_15 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_15 >> 32), 2);
                            unsigned long long hm_o_15 = (unsigned long long)hm_hi_15 << 32 | (unsigned long long)hm_lo_15;
                            unsigned long long hm_max_15 = ((hm_x_15 > hm_o_15) ? hm_x_15 : hm_o_15);
                            unsigned long long hm_min_15 = ((hm_x_15 > hm_o_15) ? hm_o_15 : hm_x_15);
                            A2[1] = (((sl & 2) == 0) ? hm_max_15 : hm_min_15);
                            unsigned long long hm_x_16 = A2[0];
                            unsigned int hm_lo_16 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_16, 1);
                            unsigned int hm_hi_16 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_16 >> 32), 1);
                            unsigned long long hm_o_16 = (unsigned long long)hm_hi_16 << 32 | (unsigned long long)hm_lo_16;
                            unsigned long long hm_max_16 = ((hm_x_16 > hm_o_16) ? hm_x_16 : hm_o_16);
                            unsigned long long hm_min_16 = ((hm_x_16 > hm_o_16) ? hm_o_16 : hm_x_16);
                            A2[0] = (((sl & 1) == 0) ? hm_max_16 : hm_min_16);
                            unsigned long long hm_x_17 = A2[1];
                            unsigned int hm_lo_17 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_17, 1);
                            unsigned int hm_hi_17 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_17 >> 32), 1);
                            unsigned long long hm_o_17 = (unsigned long long)hm_hi_17 << 32 | (unsigned long long)hm_lo_17;
                            unsigned long long hm_max_17 = ((hm_x_17 > hm_o_17) ? hm_x_17 : hm_o_17);
                            unsigned long long hm_min_17 = ((hm_x_17 > hm_o_17) ? hm_o_17 : hm_x_17);
                            A2[1] = (((sl & 1) == 0) ? hm_max_17 : hm_min_17);
                            unsigned long long hm_a_2 = A2[0];
                            unsigned long long hm_b_2 = A2[1];
                            A2[0] = ((hm_a_2 > hm_b_2) ? hm_a_2 : hm_b_2);
                            A2[1] = ((hm_a_2 > hm_b_2) ? hm_b_2 : hm_a_2);
                        }
                        if (3 < taken && 3 < used) {
                            A3[0] = ((A3[0] > L3[1]) ? A3[0] : L3[1]);
                            A3[1] = ((A3[1] > L3[0]) ? A3[1] : L3[0]);
                            unsigned long long hm_x_18 = A3[0];
                            unsigned int hm_lo_18 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_18, 4);
                            unsigned int hm_hi_18 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_18 >> 32), 4);
                            unsigned long long hm_o_18 = (unsigned long long)hm_hi_18 << 32 | (unsigned long long)hm_lo_18;
                            unsigned long long hm_max_18 = ((hm_x_18 > hm_o_18) ? hm_x_18 : hm_o_18);
                            unsigned long long hm_min_18 = ((hm_x_18 > hm_o_18) ? hm_o_18 : hm_x_18);
                            A3[0] = (((sl & 4) == 0) ? hm_max_18 : hm_min_18);
                            unsigned long long hm_x_19 = A3[1];
                            unsigned int hm_lo_19 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_19, 4);
                            unsigned int hm_hi_19 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_19 >> 32), 4);
                            unsigned long long hm_o_19 = (unsigned long long)hm_hi_19 << 32 | (unsigned long long)hm_lo_19;
                            unsigned long long hm_max_19 = ((hm_x_19 > hm_o_19) ? hm_x_19 : hm_o_19);
                            unsigned long long hm_min_19 = ((hm_x_19 > hm_o_19) ? hm_o_19 : hm_x_19);
                            A3[1] = (((sl & 4) == 0) ? hm_max_19 : hm_min_19);
                            unsigned long long hm_x_20 = A3[0];
                            unsigned int hm_lo_20 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_20, 2);
                            unsigned int hm_hi_20 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_20 >> 32), 2);
                            unsigned long long hm_o_20 = (unsigned long long)hm_hi_20 << 32 | (unsigned long long)hm_lo_20;
                            unsigned long long hm_max_20 = ((hm_x_20 > hm_o_20) ? hm_x_20 : hm_o_20);
                            unsigned long long hm_min_20 = ((hm_x_20 > hm_o_20) ? hm_o_20 : hm_x_20);
                            A3[0] = (((sl & 2) == 0) ? hm_max_20 : hm_min_20);
                            unsigned long long hm_x_21 = A3[1];
                            unsigned int hm_lo_21 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_21, 2);
                            unsigned int hm_hi_21 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_21 >> 32), 2);
                            unsigned long long hm_o_21 = (unsigned long long)hm_hi_21 << 32 | (unsigned long long)hm_lo_21;
                            unsigned long long hm_max_21 = ((hm_x_21 > hm_o_21) ? hm_x_21 : hm_o_21);
                            unsigned long long hm_min_21 = ((hm_x_21 > hm_o_21) ? hm_o_21 : hm_x_21);
                            A3[1] = (((sl & 2) == 0) ? hm_max_21 : hm_min_21);
                            unsigned long long hm_x_22 = A3[0];
                            unsigned int hm_lo_22 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_22, 1);
                            unsigned int hm_hi_22 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_22 >> 32), 1);
                            unsigned long long hm_o_22 = (unsigned long long)hm_hi_22 << 32 | (unsigned long long)hm_lo_22;
                            unsigned long long hm_max_22 = ((hm_x_22 > hm_o_22) ? hm_x_22 : hm_o_22);
                            unsigned long long hm_min_22 = ((hm_x_22 > hm_o_22) ? hm_o_22 : hm_x_22);
                            A3[0] = (((sl & 1) == 0) ? hm_max_22 : hm_min_22);
                            unsigned long long hm_x_23 = A3[1];
                            unsigned int hm_lo_23 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_23, 1);
                            unsigned int hm_hi_23 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_23 >> 32), 1);
                            unsigned long long hm_o_23 = (unsigned long long)hm_hi_23 << 32 | (unsigned long long)hm_lo_23;
                            unsigned long long hm_max_23 = ((hm_x_23 > hm_o_23) ? hm_x_23 : hm_o_23);
                            unsigned long long hm_min_23 = ((hm_x_23 > hm_o_23) ? hm_o_23 : hm_x_23);
                            A3[1] = (((sl & 1) == 0) ? hm_max_23 : hm_min_23);
                            unsigned long long hm_a_3 = A3[0];
                            unsigned long long hm_b_3 = A3[1];
                            A3[0] = ((hm_a_3 > hm_b_3) ? hm_a_3 : hm_b_3);
                            A3[1] = ((hm_a_3 > hm_b_3) ? hm_b_3 : hm_a_3);
                        }
                        used = ((used > taken) ? used : taken);
                    }
                }
                if (used > 1) {
                    unsigned long long hr_x_0 = A1[0];
                    unsigned int hr_lo_0 = __shfl_sync(0xFFFFFFFF, (unsigned int)hr_x_0, 7 - sl, 8);
                    unsigned int hr_hi_0 = __shfl_sync(0xFFFFFFFF, (unsigned int)(hr_x_0 >> 32), 7 - sl, 8);
                    L0[0] = (unsigned long long)hr_hi_0 << 32 | (unsigned long long)hr_lo_0;
                    unsigned long long hr_x_1 = A1[1];
                    unsigned int hr_lo_1 = __shfl_sync(0xFFFFFFFF, (unsigned int)hr_x_1, 7 - sl, 8);
                    unsigned int hr_hi_1 = __shfl_sync(0xFFFFFFFF, (unsigned int)(hr_x_1 >> 32), 7 - sl, 8);
                    L0[1] = (unsigned long long)hr_hi_1 << 32 | (unsigned long long)hr_lo_1;
                    R[0] = ((R[0] > L0[1]) ? R[0] : L0[1]);
                    R[1] = ((R[1] > L0[0]) ? R[1] : L0[0]);
                    unsigned long long hm_x_24 = R[0];
                    unsigned int hm_lo_24 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_24, 4);
                    unsigned int hm_hi_24 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_24 >> 32), 4);
                    unsigned long long hm_o_24 = (unsigned long long)hm_hi_24 << 32 | (unsigned long long)hm_lo_24;
                    unsigned long long hm_max_24 = ((hm_x_24 > hm_o_24) ? hm_x_24 : hm_o_24);
                    unsigned long long hm_min_24 = ((hm_x_24 > hm_o_24) ? hm_o_24 : hm_x_24);
                    R[0] = (((sl & 4) == 0) ? hm_max_24 : hm_min_24);
                    unsigned long long hm_x_25 = R[1];
                    unsigned int hm_lo_25 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_25, 4);
                    unsigned int hm_hi_25 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_25 >> 32), 4);
                    unsigned long long hm_o_25 = (unsigned long long)hm_hi_25 << 32 | (unsigned long long)hm_lo_25;
                    unsigned long long hm_max_25 = ((hm_x_25 > hm_o_25) ? hm_x_25 : hm_o_25);
                    unsigned long long hm_min_25 = ((hm_x_25 > hm_o_25) ? hm_o_25 : hm_x_25);
                    R[1] = (((sl & 4) == 0) ? hm_max_25 : hm_min_25);
                    unsigned long long hm_x_26 = R[0];
                    unsigned int hm_lo_26 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_26, 2);
                    unsigned int hm_hi_26 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_26 >> 32), 2);
                    unsigned long long hm_o_26 = (unsigned long long)hm_hi_26 << 32 | (unsigned long long)hm_lo_26;
                    unsigned long long hm_max_26 = ((hm_x_26 > hm_o_26) ? hm_x_26 : hm_o_26);
                    unsigned long long hm_min_26 = ((hm_x_26 > hm_o_26) ? hm_o_26 : hm_x_26);
                    R[0] = (((sl & 2) == 0) ? hm_max_26 : hm_min_26);
                    unsigned long long hm_x_27 = R[1];
                    unsigned int hm_lo_27 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_27, 2);
                    unsigned int hm_hi_27 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_27 >> 32), 2);
                    unsigned long long hm_o_27 = (unsigned long long)hm_hi_27 << 32 | (unsigned long long)hm_lo_27;
                    unsigned long long hm_max_27 = ((hm_x_27 > hm_o_27) ? hm_x_27 : hm_o_27);
                    unsigned long long hm_min_27 = ((hm_x_27 > hm_o_27) ? hm_o_27 : hm_x_27);
                    R[1] = (((sl & 2) == 0) ? hm_max_27 : hm_min_27);
                    unsigned long long hm_x_28 = R[0];
                    unsigned int hm_lo_28 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_28, 1);
                    unsigned int hm_hi_28 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_28 >> 32), 1);
                    unsigned long long hm_o_28 = (unsigned long long)hm_hi_28 << 32 | (unsigned long long)hm_lo_28;
                    unsigned long long hm_max_28 = ((hm_x_28 > hm_o_28) ? hm_x_28 : hm_o_28);
                    unsigned long long hm_min_28 = ((hm_x_28 > hm_o_28) ? hm_o_28 : hm_x_28);
                    R[0] = (((sl & 1) == 0) ? hm_max_28 : hm_min_28);
                    unsigned long long hm_x_29 = R[1];
                    unsigned int hm_lo_29 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_29, 1);
                    unsigned int hm_hi_29 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_29 >> 32), 1);
                    unsigned long long hm_o_29 = (unsigned long long)hm_hi_29 << 32 | (unsigned long long)hm_lo_29;
                    unsigned long long hm_max_29 = ((hm_x_29 > hm_o_29) ? hm_x_29 : hm_o_29);
                    unsigned long long hm_min_29 = ((hm_x_29 > hm_o_29) ? hm_o_29 : hm_x_29);
                    R[1] = (((sl & 1) == 0) ? hm_max_29 : hm_min_29);
                    unsigned long long hm_a_4 = R[0];
                    unsigned long long hm_b_4 = R[1];
                    R[0] = ((hm_a_4 > hm_b_4) ? hm_a_4 : hm_b_4);
                    R[1] = ((hm_a_4 > hm_b_4) ? hm_b_4 : hm_a_4);
                }
                if (used > 3) {
                    unsigned long long hr_x_2 = A3[0];
                    unsigned int hr_lo_2 = __shfl_sync(0xFFFFFFFF, (unsigned int)hr_x_2, 7 - sl, 8);
                    unsigned int hr_hi_2 = __shfl_sync(0xFFFFFFFF, (unsigned int)(hr_x_2 >> 32), 7 - sl, 8);
                    L0[0] = (unsigned long long)hr_hi_2 << 32 | (unsigned long long)hr_lo_2;
                    unsigned long long hr_x_3 = A3[1];
                    unsigned int hr_lo_3 = __shfl_sync(0xFFFFFFFF, (unsigned int)hr_x_3, 7 - sl, 8);
                    unsigned int hr_hi_3 = __shfl_sync(0xFFFFFFFF, (unsigned int)(hr_x_3 >> 32), 7 - sl, 8);
                    L0[1] = (unsigned long long)hr_hi_3 << 32 | (unsigned long long)hr_lo_3;
                    A2[0] = ((A2[0] > L0[1]) ? A2[0] : L0[1]);
                    A2[1] = ((A2[1] > L0[0]) ? A2[1] : L0[0]);
                    unsigned long long hm_x_30 = A2[0];
                    unsigned int hm_lo_30 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_30, 4);
                    unsigned int hm_hi_30 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_30 >> 32), 4);
                    unsigned long long hm_o_30 = (unsigned long long)hm_hi_30 << 32 | (unsigned long long)hm_lo_30;
                    unsigned long long hm_max_30 = ((hm_x_30 > hm_o_30) ? hm_x_30 : hm_o_30);
                    unsigned long long hm_min_30 = ((hm_x_30 > hm_o_30) ? hm_o_30 : hm_x_30);
                    A2[0] = (((sl & 4) == 0) ? hm_max_30 : hm_min_30);
                    unsigned long long hm_x_31 = A2[1];
                    unsigned int hm_lo_31 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_31, 4);
                    unsigned int hm_hi_31 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_31 >> 32), 4);
                    unsigned long long hm_o_31 = (unsigned long long)hm_hi_31 << 32 | (unsigned long long)hm_lo_31;
                    unsigned long long hm_max_31 = ((hm_x_31 > hm_o_31) ? hm_x_31 : hm_o_31);
                    unsigned long long hm_min_31 = ((hm_x_31 > hm_o_31) ? hm_o_31 : hm_x_31);
                    A2[1] = (((sl & 4) == 0) ? hm_max_31 : hm_min_31);
                    unsigned long long hm_x_32 = A2[0];
                    unsigned int hm_lo_32 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_32, 2);
                    unsigned int hm_hi_32 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_32 >> 32), 2);
                    unsigned long long hm_o_32 = (unsigned long long)hm_hi_32 << 32 | (unsigned long long)hm_lo_32;
                    unsigned long long hm_max_32 = ((hm_x_32 > hm_o_32) ? hm_x_32 : hm_o_32);
                    unsigned long long hm_min_32 = ((hm_x_32 > hm_o_32) ? hm_o_32 : hm_x_32);
                    A2[0] = (((sl & 2) == 0) ? hm_max_32 : hm_min_32);
                    unsigned long long hm_x_33 = A2[1];
                    unsigned int hm_lo_33 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_33, 2);
                    unsigned int hm_hi_33 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_33 >> 32), 2);
                    unsigned long long hm_o_33 = (unsigned long long)hm_hi_33 << 32 | (unsigned long long)hm_lo_33;
                    unsigned long long hm_max_33 = ((hm_x_33 > hm_o_33) ? hm_x_33 : hm_o_33);
                    unsigned long long hm_min_33 = ((hm_x_33 > hm_o_33) ? hm_o_33 : hm_x_33);
                    A2[1] = (((sl & 2) == 0) ? hm_max_33 : hm_min_33);
                    unsigned long long hm_x_34 = A2[0];
                    unsigned int hm_lo_34 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_34, 1);
                    unsigned int hm_hi_34 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_34 >> 32), 1);
                    unsigned long long hm_o_34 = (unsigned long long)hm_hi_34 << 32 | (unsigned long long)hm_lo_34;
                    unsigned long long hm_max_34 = ((hm_x_34 > hm_o_34) ? hm_x_34 : hm_o_34);
                    unsigned long long hm_min_34 = ((hm_x_34 > hm_o_34) ? hm_o_34 : hm_x_34);
                    A2[0] = (((sl & 1) == 0) ? hm_max_34 : hm_min_34);
                    unsigned long long hm_x_35 = A2[1];
                    unsigned int hm_lo_35 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_35, 1);
                    unsigned int hm_hi_35 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_35 >> 32), 1);
                    unsigned long long hm_o_35 = (unsigned long long)hm_hi_35 << 32 | (unsigned long long)hm_lo_35;
                    unsigned long long hm_max_35 = ((hm_x_35 > hm_o_35) ? hm_x_35 : hm_o_35);
                    unsigned long long hm_min_35 = ((hm_x_35 > hm_o_35) ? hm_o_35 : hm_x_35);
                    A2[1] = (((sl & 1) == 0) ? hm_max_35 : hm_min_35);
                    unsigned long long hm_a_5 = A2[0];
                    unsigned long long hm_b_5 = A2[1];
                    A2[0] = ((hm_a_5 > hm_b_5) ? hm_a_5 : hm_b_5);
                    A2[1] = ((hm_a_5 > hm_b_5) ? hm_b_5 : hm_a_5);
                }
                if (used > 2) {
                    unsigned long long hr_x_4 = A2[0];
                    unsigned int hr_lo_4 = __shfl_sync(0xFFFFFFFF, (unsigned int)hr_x_4, 7 - sl, 8);
                    unsigned int hr_hi_4 = __shfl_sync(0xFFFFFFFF, (unsigned int)(hr_x_4 >> 32), 7 - sl, 8);
                    L0[0] = (unsigned long long)hr_hi_4 << 32 | (unsigned long long)hr_lo_4;
                    unsigned long long hr_x_5 = A2[1];
                    unsigned int hr_lo_5 = __shfl_sync(0xFFFFFFFF, (unsigned int)hr_x_5, 7 - sl, 8);
                    unsigned int hr_hi_5 = __shfl_sync(0xFFFFFFFF, (unsigned int)(hr_x_5 >> 32), 7 - sl, 8);
                    L0[1] = (unsigned long long)hr_hi_5 << 32 | (unsigned long long)hr_lo_5;
                    R[0] = ((R[0] > L0[1]) ? R[0] : L0[1]);
                    R[1] = ((R[1] > L0[0]) ? R[1] : L0[0]);
                    unsigned long long hm_x_36 = R[0];
                    unsigned int hm_lo_36 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_36, 4);
                    unsigned int hm_hi_36 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_36 >> 32), 4);
                    unsigned long long hm_o_36 = (unsigned long long)hm_hi_36 << 32 | (unsigned long long)hm_lo_36;
                    unsigned long long hm_max_36 = ((hm_x_36 > hm_o_36) ? hm_x_36 : hm_o_36);
                    unsigned long long hm_min_36 = ((hm_x_36 > hm_o_36) ? hm_o_36 : hm_x_36);
                    R[0] = (((sl & 4) == 0) ? hm_max_36 : hm_min_36);
                    unsigned long long hm_x_37 = R[1];
                    unsigned int hm_lo_37 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_37, 4);
                    unsigned int hm_hi_37 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_37 >> 32), 4);
                    unsigned long long hm_o_37 = (unsigned long long)hm_hi_37 << 32 | (unsigned long long)hm_lo_37;
                    unsigned long long hm_max_37 = ((hm_x_37 > hm_o_37) ? hm_x_37 : hm_o_37);
                    unsigned long long hm_min_37 = ((hm_x_37 > hm_o_37) ? hm_o_37 : hm_x_37);
                    R[1] = (((sl & 4) == 0) ? hm_max_37 : hm_min_37);
                    unsigned long long hm_x_38 = R[0];
                    unsigned int hm_lo_38 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_38, 2);
                    unsigned int hm_hi_38 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_38 >> 32), 2);
                    unsigned long long hm_o_38 = (unsigned long long)hm_hi_38 << 32 | (unsigned long long)hm_lo_38;
                    unsigned long long hm_max_38 = ((hm_x_38 > hm_o_38) ? hm_x_38 : hm_o_38);
                    unsigned long long hm_min_38 = ((hm_x_38 > hm_o_38) ? hm_o_38 : hm_x_38);
                    R[0] = (((sl & 2) == 0) ? hm_max_38 : hm_min_38);
                    unsigned long long hm_x_39 = R[1];
                    unsigned int hm_lo_39 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_39, 2);
                    unsigned int hm_hi_39 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_39 >> 32), 2);
                    unsigned long long hm_o_39 = (unsigned long long)hm_hi_39 << 32 | (unsigned long long)hm_lo_39;
                    unsigned long long hm_max_39 = ((hm_x_39 > hm_o_39) ? hm_x_39 : hm_o_39);
                    unsigned long long hm_min_39 = ((hm_x_39 > hm_o_39) ? hm_o_39 : hm_x_39);
                    R[1] = (((sl & 2) == 0) ? hm_max_39 : hm_min_39);
                    unsigned long long hm_x_40 = R[0];
                    unsigned int hm_lo_40 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_40, 1);
                    unsigned int hm_hi_40 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_40 >> 32), 1);
                    unsigned long long hm_o_40 = (unsigned long long)hm_hi_40 << 32 | (unsigned long long)hm_lo_40;
                    unsigned long long hm_max_40 = ((hm_x_40 > hm_o_40) ? hm_x_40 : hm_o_40);
                    unsigned long long hm_min_40 = ((hm_x_40 > hm_o_40) ? hm_o_40 : hm_x_40);
                    R[0] = (((sl & 1) == 0) ? hm_max_40 : hm_min_40);
                    unsigned long long hm_x_41 = R[1];
                    unsigned int hm_lo_41 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)hm_x_41, 1);
                    unsigned int hm_hi_41 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)(hm_x_41 >> 32), 1);
                    unsigned long long hm_o_41 = (unsigned long long)hm_hi_41 << 32 | (unsigned long long)hm_lo_41;
                    unsigned long long hm_max_41 = ((hm_x_41 > hm_o_41) ? hm_x_41 : hm_o_41);
                    unsigned long long hm_min_41 = ((hm_x_41 > hm_o_41) ? hm_o_41 : hm_x_41);
                    R[1] = (((sl & 1) == 0) ? hm_max_41 : hm_min_41);
                    unsigned long long hm_a_6 = R[0];
                    unsigned long long hm_b_6 = R[1];
                    R[0] = ((hm_a_6 > hm_b_6) ? hm_a_6 : hm_b_6);
                    R[1] = ((hm_a_6 > hm_b_6) ? hm_b_6 : hm_a_6);
                }
            }
        }
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        int g[2];
        int pos[2];
        g[0] = ((R[0] != 0) ? 2147483647 - (int)(R[0] & 2147483647) : 2147483647);
        pos[0] = 0;
        g[1] = ((R[1] != 0) ? 2147483647 - (int)(R[1] & 2147483647) : 2147483647);
        pos[1] = 0;
        int g_i_0 = __shfl_sync(0xFFFFFFFF, g[0], 0, 8);
        pos[0] = pos[0] + (int)(g_i_0 < g[0] || g_i_0 == g[0] && 2 * sl > 0);
        pos[1] = pos[1] + (int)(g_i_0 < g[1] || g_i_0 == g[1] && 2 * sl + 1 > 0);
        int g_i_1 = __shfl_sync(0xFFFFFFFF, g[1], 0, 8);
        pos[0] = pos[0] + (int)(g_i_1 < g[0] || g_i_1 == g[0] && 2 * sl > 1);
        pos[1] = pos[1] + (int)(g_i_1 < g[1] || g_i_1 == g[1] && 2 * sl + 1 > 1);
        int g_i_2 = __shfl_sync(0xFFFFFFFF, g[0], 1, 8);
        pos[0] = pos[0] + (int)(g_i_2 < g[0] || g_i_2 == g[0] && 2 * sl > 2);
        pos[1] = pos[1] + (int)(g_i_2 < g[1] || g_i_2 == g[1] && 2 * sl + 1 > 2);
        int g_i_3 = __shfl_sync(0xFFFFFFFF, g[1], 1, 8);
        pos[0] = pos[0] + (int)(g_i_3 < g[0] || g_i_3 == g[0] && 2 * sl > 3);
        pos[1] = pos[1] + (int)(g_i_3 < g[1] || g_i_3 == g[1] && 2 * sl + 1 > 3);
        int g_i_4 = __shfl_sync(0xFFFFFFFF, g[0], 2, 8);
        pos[0] = pos[0] + (int)(g_i_4 < g[0] || g_i_4 == g[0] && 2 * sl > 4);
        pos[1] = pos[1] + (int)(g_i_4 < g[1] || g_i_4 == g[1] && 2 * sl + 1 > 4);
        int g_i_5 = __shfl_sync(0xFFFFFFFF, g[1], 2, 8);
        pos[0] = pos[0] + (int)(g_i_5 < g[0] || g_i_5 == g[0] && 2 * sl > 5);
        pos[1] = pos[1] + (int)(g_i_5 < g[1] || g_i_5 == g[1] && 2 * sl + 1 > 5);
        int g_i_6 = __shfl_sync(0xFFFFFFFF, g[0], 3, 8);
        pos[0] = pos[0] + (int)(g_i_6 < g[0] || g_i_6 == g[0] && 2 * sl > 6);
        pos[1] = pos[1] + (int)(g_i_6 < g[1] || g_i_6 == g[1] && 2 * sl + 1 > 6);
        int g_i_7 = __shfl_sync(0xFFFFFFFF, g[1], 3, 8);
        pos[0] = pos[0] + (int)(g_i_7 < g[0] || g_i_7 == g[0] && 2 * sl > 7);
        pos[1] = pos[1] + (int)(g_i_7 < g[1] || g_i_7 == g[1] && 2 * sl + 1 > 7);
        int g_i_8 = __shfl_sync(0xFFFFFFFF, g[0], 4, 8);
        pos[0] = pos[0] + (int)(g_i_8 < g[0] || g_i_8 == g[0] && 2 * sl > 8);
        pos[1] = pos[1] + (int)(g_i_8 < g[1] || g_i_8 == g[1] && 2 * sl + 1 > 8);
        int g_i_9 = __shfl_sync(0xFFFFFFFF, g[1], 4, 8);
        pos[0] = pos[0] + (int)(g_i_9 < g[0] || g_i_9 == g[0] && 2 * sl > 9);
        pos[1] = pos[1] + (int)(g_i_9 < g[1] || g_i_9 == g[1] && 2 * sl + 1 > 9);
        int g_i_10 = __shfl_sync(0xFFFFFFFF, g[0], 5, 8);
        pos[0] = pos[0] + (int)(g_i_10 < g[0] || g_i_10 == g[0] && 2 * sl > 10);
        pos[1] = pos[1] + (int)(g_i_10 < g[1] || g_i_10 == g[1] && 2 * sl + 1 > 10);
        int g_i_11 = __shfl_sync(0xFFFFFFFF, g[1], 5, 8);
        pos[0] = pos[0] + (int)(g_i_11 < g[0] || g_i_11 == g[0] && 2 * sl > 11);
        pos[1] = pos[1] + (int)(g_i_11 < g[1] || g_i_11 == g[1] && 2 * sl + 1 > 11);
        int g_i_12 = __shfl_sync(0xFFFFFFFF, g[0], 6, 8);
        pos[0] = pos[0] + (int)(g_i_12 < g[0] || g_i_12 == g[0] && 2 * sl > 12);
        pos[1] = pos[1] + (int)(g_i_12 < g[1] || g_i_12 == g[1] && 2 * sl + 1 > 12);
        int g_i_13 = __shfl_sync(0xFFFFFFFF, g[1], 6, 8);
        pos[0] = pos[0] + (int)(g_i_13 < g[0] || g_i_13 == g[0] && 2 * sl > 13);
        pos[1] = pos[1] + (int)(g_i_13 < g[1] || g_i_13 == g[1] && 2 * sl + 1 > 13);
        int g_i_14 = __shfl_sync(0xFFFFFFFF, g[0], 7, 8);
        pos[0] = pos[0] + (int)(g_i_14 < g[0] || g_i_14 == g[0] && 2 * sl > 14);
        pos[1] = pos[1] + (int)(g_i_14 < g[1] || g_i_14 == g[1] && 2 * sl + 1 > 14);
        int g_i_15 = __shfl_sync(0xFFFFFFFF, g[1], 7, 8);
        pos[0] = pos[0] + (int)(g_i_15 < g[0] || g_i_15 == g[0] && 2 * sl > 15);
        pos[1] = pos[1] + (int)(g_i_15 < g[1] || g_i_15 == g[1] && 2 * sl + 1 > 15);
        long long pb_gen_at = (long long)(slot * 4 + h) * (long long)capacity + (long long)(t_base + t);
        unsigned int pb_gen = 0;
        if (sl == 0) {
            int pb_old = generations[pb_gen_at];
            pb_gen = (unsigned int)pb_old + 1;
            pb_gen = ((pb_gen != 0) ? pb_gen : 1);
        }
        unsigned int _shfl_sync_195 = __shfl_sync(0xFFFFFFFF, pb_gen, 0, 8);
        pb_gen = _shfl_sync_195;
        int st_base = ((int)warp * 4 + h) * 32;
        unsigned int p_o32_0 = (unsigned int)(R[0] >> 31);
        unsigned int p_ob_0 = ((p_o32_0 >= 2147483648u) ? p_o32_0 ^ 2147483648u : ~p_o32_0);
        stage[st_base + (pos[0] & 15)] = ((R[0] != 0) ? (int)p_ob_0 : -8388608);
        stage[st_base + 16 + (pos[0] & 15)] = ((R[0] != 0) ? g[0] : -1);
        unsigned int p_o32_1 = (unsigned int)(R[1] >> 31);
        unsigned int p_ob_1 = ((p_o32_1 >= 2147483648u) ? p_o32_1 ^ 2147483648u : ~p_o32_1);
        stage[st_base + (pos[1] & 15)] = ((R[1] != 0) ? (int)p_ob_1 : -8388608);
        stage[st_base + 16 + (pos[1] & 15)] = ((R[1] != 0) ? g[1] : -1);
        __syncwarp();
        int pb_owner = h / heads_per_rank;
        int pb_hloc = h - pb_owner * heads_per_rank;
        long long pb_row = (long long)((slot * world + rank) * heads_per_rank + pb_hloc) * (long long)capacity + (long long)(t_base + t);
        unsigned long long pb_tag = (unsigned long long)pb_gen << 32;
        if (pb_owner == 0) {
            unsigned long long* pb_dst_0 = recv_0 + pb_row * 32;
            int pb_w_0 = stage[st_base + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_0 + sl)), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_0)) : "memory");
            int pb_w_1 = stage[st_base + 8 + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_0 + (8 + sl))), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_1)) : "memory");
            int pb_w_2 = stage[st_base + 16 + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_0 + (16 + sl))), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_2)) : "memory");
            int pb_w_3 = stage[st_base + 24 + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_0 + (24 + sl))), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_3)) : "memory");
        }
        if (pb_owner == 1) {
            unsigned long long* pb_dst_1 = recv_1 + pb_row * 32;
            int pb_w_4 = stage[st_base + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_1 + sl)), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_4)) : "memory");
            int pb_w_5 = stage[st_base + 8 + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_1 + (8 + sl))), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_5)) : "memory");
            int pb_w_6 = stage[st_base + 16 + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_1 + (16 + sl))), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_6)) : "memory");
            int pb_w_7 = stage[st_base + 24 + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_1 + (24 + sl))), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_7)) : "memory");
        }
        if (pb_owner == 2) {
            unsigned long long* pb_dst_2 = recv_2 + pb_row * 32;
            int pb_w_8 = stage[st_base + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_2 + sl)), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_8)) : "memory");
            int pb_w_9 = stage[st_base + 8 + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_2 + (8 + sl))), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_9)) : "memory");
            int pb_w_10 = stage[st_base + 16 + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_2 + (16 + sl))), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_10)) : "memory");
            int pb_w_11 = stage[st_base + 24 + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_2 + (24 + sl))), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_11)) : "memory");
        }
        if (pb_owner == 3) {
            unsigned long long* pb_dst_3 = recv_3 + pb_row * 32;
            int pb_w_12 = stage[st_base + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_3 + sl)), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_12)) : "memory");
            int pb_w_13 = stage[st_base + 8 + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_3 + (8 + sl))), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_13)) : "memory");
            int pb_w_14 = stage[st_base + 16 + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_3 + (16 + sl))), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_14)) : "memory");
            int pb_w_15 = stage[st_base + 24 + sl];
            asm volatile("st.relaxed.sys.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(pb_dst_3 + (24 + sl))), "l"(static_cast<unsigned long long>(pb_tag | (unsigned long long)(unsigned int)pb_w_15)) : "memory");
        }
        __syncwarp();
        if (sl == 0) {
            generations[pb_gen_at] = (int)pb_gen;
        }
    }
}

} // extern "C"
