/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#define SMEM_S_M_OFF 0
#define SMEM_S_M_STAGE_BYTES 17600
#define SMEM_S_M_STRIDE 17600
#define SMEM_TOTAL 17664
#define THREADS 512

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_dsa_indexer_topk_d194cadc957412227f47(long long* __restrict__ Staging, int* __restrict__ Indices, float* __restrict__ Scores, int top_k, int n_split)
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
    int* s_m = reinterpret_cast<int*>(smem_raw + SMEM_S_M_OFF);
    const int s_m_addr = smem + SMEM_S_M_OFF;

    // === Task calls (dependency order) ===
    int lane_1 = tid % 32;
    int warp_1 = tid / 32;
    int n = n_split * top_k;
    long long base = (long long)bid * (long long)n;
    long long row_base = (long long)bid * (long long)top_k;
    unsigned int one_u = 1;
    unsigned int lower_lanes = (one_u << (unsigned int)lane_1) - 1;
    unsigned int zero_u = 0;
    unsigned int sign_mask = 2147483648;
    unsigned int nan_floor = 2139095040;
    int hist_base = warp_1 * 256;
    int sink_idx = 4352 + lane_1;
    int k_rem = top_k;
    int kept_above = 0;
    unsigned long long prefix = 0;
    unsigned long long edge = 0;
    int c_valid = 0;
    #pragma unroll 1
    for (int p = 0; p < 8; p++) {
        int shift = 56 - 8 * p;
        s_m[hist_base + lane_1 * 8] = 0;
        s_m[hist_base + lane_1 * 8 + 1] = 0;
        s_m[hist_base + lane_1 * 8 + 2] = 0;
        s_m[hist_base + lane_1 * 8 + 3] = 0;
        s_m[hist_base + lane_1 * 8 + 4] = 0;
        s_m[hist_base + lane_1 * 8 + 5] = 0;
        s_m[hist_base + lane_1 * 8 + 6] = 0;
        s_m[hist_base + lane_1 * 8 + 7] = 0;
        __syncwarp();
        #pragma unroll 1
        for (int e0 = 0; e0 < n; e0 += 4096) {
            unsigned long long ent[8];
            #pragma unroll
            for (int u = 0; u < 8; u++) {
                int idx_l = e0 + tid + 512 * u;
                ent[u] = 4294967295;
                if (idx_l < n) {
                    ent[u] = (unsigned long long)Staging[base + (long long)idx_l];
                }
            }
            #pragma unroll
            for (int u_1 = 0; u_1 < 8; u_1++) {
                int kid = (int)(unsigned int)(ent[u_1] & 4294967295);
                int kid_h = kid;
                int valid_h = ((kid_h != -1) ? 1 : 0);
                unsigned int bits = (unsigned int)(ent[u_1] >> 32);
                unsigned int magnitude = bits & 2147483647;
                unsigned int m = ((magnitude != 0) ? bits : zero_u);
                unsigned int key32 = (((m & sign_mask) != 0) ? ~m : m | sign_mask);
                if (magnitude > nan_floor) {
                    key32 = zero_u;
                }
                unsigned long long key64 = (unsigned long long)key32 << 32 | (unsigned long long)(unsigned int)kid_h;
                unsigned long long key_h = key64;
                unsigned long long ks_h = key_h >> (unsigned long long)shift;
                int digit_h = (int)(ks_h & 255);
                int pm_h = ((ks_h >> 8 == prefix) ? 1 : 0);
                int bin_h = (((valid_h & pm_h) != 0) ? hist_base + digit_h : sink_idx);
                atomicAdd(&s_m[bin_h], 1);
            }
        }
        __syncthreads();
        if (tid < 256) {
            int tot = 0;
            tot += s_m[tid];
            tot += s_m[256 + tid];
            tot += s_m[512 + tid];
            tot += s_m[768 + tid];
            tot += s_m[1024 + tid];
            tot += s_m[1280 + tid];
            tot += s_m[1536 + tid];
            tot += s_m[1792 + tid];
            tot += s_m[2048 + tid];
            tot += s_m[2304 + tid];
            tot += s_m[2560 + tid];
            tot += s_m[2816 + tid];
            tot += s_m[3072 + tid];
            tot += s_m[3328 + tid];
            tot += s_m[3584 + tid];
            tot += s_m[3840 + tid];
            s_m[4096 + tid] = tot;
        }
        __syncthreads();
        int lane_bins[8];
        int lane_sum = 0;
        lane_bins[0] = s_m[4096 + lane_1 * 8];
        lane_sum += lane_bins[0];
        lane_bins[1] = s_m[4096 + lane_1 * 8 + 1];
        lane_sum += lane_bins[1];
        lane_bins[2] = s_m[4096 + lane_1 * 8 + 2];
        lane_sum += lane_bins[2];
        lane_bins[3] = s_m[4096 + lane_1 * 8 + 3];
        lane_sum += lane_bins[3];
        lane_bins[4] = s_m[4096 + lane_1 * 8 + 4];
        lane_sum += lane_bins[4];
        lane_bins[5] = s_m[4096 + lane_1 * 8 + 5];
        lane_sum += lane_bins[5];
        lane_bins[6] = s_m[4096 + lane_1 * 8 + 6];
        lane_sum += lane_bins[6];
        lane_bins[7] = s_m[4096 + lane_1 * 8 + 7];
        lane_sum += lane_bins[7];
        int suffix = lane_sum;
        int _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, suffix, 1, 32);
        int above_part = _shfl_down_0;
        if (lane_1 + 1 < 32) {
            suffix += above_part;
        }
        int _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, suffix, 2, 32);
        int above_part_0 = _shfl_down_1;
        if (lane_1 + 2 < 32) {
            suffix += above_part_0;
        }
        int _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, suffix, 4, 32);
        int above_part_1 = _shfl_down_2;
        if (lane_1 + 4 < 32) {
            suffix += above_part_1;
        }
        int _shfl_down_3 = __shfl_down_sync(0xFFFFFFFF, suffix, 8, 32);
        int above_part_2 = _shfl_down_3;
        if (lane_1 + 8 < 32) {
            suffix += above_part_2;
        }
        int _shfl_down_4 = __shfl_down_sync(0xFFFFFFFF, suffix, 16, 32);
        int above_part_3 = _shfl_down_4;
        if (lane_1 + 16 < 32) {
            suffix += above_part_3;
        }
        if (p == 0) {
            int _shfl_0 = __shfl_sync(0xFFFFFFFF, suffix, 0);
            c_valid = _shfl_0;
        }
        if (c_valid <= top_k) {
            edge = 0;
            break;
        }
        int excl = suffix - lane_sum;
        int is_target = ((excl < k_rem && k_rem <= excl + lane_sum) ? 1 : 0);
        int d_sel = 0;
        int above_sel = 0;
        int count_sel = 0;
        int found = 0;
        int cum_above = excl;
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[7]) {
                d_sel = 7;
                above_sel = cum_above;
                count_sel = lane_bins[7];
                found = 1;
            }
        }
        cum_above += lane_bins[7];
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[6]) {
                d_sel = 6;
                above_sel = cum_above;
                count_sel = lane_bins[6];
                found = 1;
            }
        }
        cum_above += lane_bins[6];
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[5]) {
                d_sel = 5;
                above_sel = cum_above;
                count_sel = lane_bins[5];
                found = 1;
            }
        }
        cum_above += lane_bins[5];
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[4]) {
                d_sel = 4;
                above_sel = cum_above;
                count_sel = lane_bins[4];
                found = 1;
            }
        }
        cum_above += lane_bins[4];
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[3]) {
                d_sel = 3;
                above_sel = cum_above;
                count_sel = lane_bins[3];
                found = 1;
            }
        }
        cum_above += lane_bins[3];
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[2]) {
                d_sel = 2;
                above_sel = cum_above;
                count_sel = lane_bins[2];
                found = 1;
            }
        }
        cum_above += lane_bins[2];
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[1]) {
                d_sel = 1;
                above_sel = cum_above;
                count_sel = lane_bins[1];
                found = 1;
            }
        }
        cum_above += lane_bins[1];
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[0]) {
                d_sel = 0;
                above_sel = cum_above;
                count_sel = lane_bins[0];
                found = 1;
            }
        }
        cum_above += lane_bins[0];
        int _warp_redux_i32_0;
        asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_i32_0) : "r"(is_target * lane_1));
        int target_lane = _warp_redux_i32_0;
        int _shfl_1 = __shfl_sync(0xFFFFFFFF, d_sel, target_lane);
        int digit_sel = _shfl_1;
        int _shfl_2 = __shfl_sync(0xFFFFFFFF, above_sel, target_lane);
        int above_cnt = _shfl_2;
        int _shfl_3 = __shfl_sync(0xFFFFFFFF, count_sel, target_lane);
        int bucket_cnt = _shfl_3;
        k_rem = k_rem - above_cnt;
        kept_above += above_cnt;
        prefix = prefix << 8 | (unsigned long long)(unsigned int)(target_lane * 8 + digit_sel);
        edge = prefix << (unsigned long long)shift;
        if (kept_above + bucket_cnt <= top_k) {
            break;
        }
    }
    int my_count = 0;
    #pragma unroll 1
    for (int e0_1 = 0; e0_1 < n; e0_1 += 4096) {
        unsigned long long ent_c[8];
        #pragma unroll
        for (int u_2 = 0; u_2 < 8; u_2++) {
            int idx_l_1 = e0_1 + tid + 512 * u_2;
            ent_c[u_2] = 4294967295;
            if (idx_l_1 < n) {
                ent_c[u_2] = (unsigned long long)Staging[base + (long long)idx_l_1];
            }
        }
        #pragma unroll
        for (int u_3 = 0; u_3 < 8; u_3++) {
            int kid_1 = (int)(unsigned int)(ent_c[u_3] & 4294967295);
            int kid_c = kid_1;
            int keep_c = 0;
            if (kid_c != -1) {
                unsigned int bits_1 = (unsigned int)(ent_c[u_3] >> 32);
                unsigned int magnitude_1 = bits_1 & 2147483647;
                unsigned int m_1 = ((magnitude_1 != 0) ? bits_1 : zero_u);
                unsigned int key32_1 = (((m_1 & sign_mask) != 0) ? ~m_1 : m_1 | sign_mask);
                if (magnitude_1 > nan_floor) {
                    key32_1 = zero_u;
                }
                unsigned long long key64_1 = (unsigned long long)key32_1 << 32 | (unsigned long long)(unsigned int)kid_c;
                keep_c = ((key64_1 >= edge) ? 1 : 0);
            }
            unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, keep_c != 0);
            unsigned int ballot_c = _vote_0;
            int _popc_0 = __popc(ballot_c);
            my_count += _popc_0;
        }
    }
    if (lane_1 == 0) {
        s_m[4384 + warp_1] = my_count;
    }
    __syncthreads();
    int write_pos = 0;
    int n_kept = 0;
    int cnt_w = s_m[4384];
    n_kept += cnt_w;
    write_pos += ((warp_1 > 0) ? cnt_w : 0);
    int cnt_w_0 = s_m[4385];
    n_kept += cnt_w_0;
    write_pos += ((warp_1 > 1) ? cnt_w_0 : 0);
    int cnt_w_1 = s_m[4386];
    n_kept += cnt_w_1;
    write_pos += ((warp_1 > 2) ? cnt_w_1 : 0);
    int cnt_w_2 = s_m[4387];
    n_kept += cnt_w_2;
    write_pos += ((warp_1 > 3) ? cnt_w_2 : 0);
    int cnt_w_3 = s_m[4388];
    n_kept += cnt_w_3;
    write_pos += ((warp_1 > 4) ? cnt_w_3 : 0);
    int cnt_w_4 = s_m[4389];
    n_kept += cnt_w_4;
    write_pos += ((warp_1 > 5) ? cnt_w_4 : 0);
    int cnt_w_5 = s_m[4390];
    n_kept += cnt_w_5;
    write_pos += ((warp_1 > 6) ? cnt_w_5 : 0);
    int cnt_w_6 = s_m[4391];
    n_kept += cnt_w_6;
    write_pos += ((warp_1 > 7) ? cnt_w_6 : 0);
    int cnt_w_7 = s_m[4392];
    n_kept += cnt_w_7;
    write_pos += ((warp_1 > 8) ? cnt_w_7 : 0);
    int cnt_w_8 = s_m[4393];
    n_kept += cnt_w_8;
    write_pos += ((warp_1 > 9) ? cnt_w_8 : 0);
    int cnt_w_9 = s_m[4394];
    n_kept += cnt_w_9;
    write_pos += ((warp_1 > 10) ? cnt_w_9 : 0);
    int cnt_w_10 = s_m[4395];
    n_kept += cnt_w_10;
    write_pos += ((warp_1 > 11) ? cnt_w_10 : 0);
    int cnt_w_11 = s_m[4396];
    n_kept += cnt_w_11;
    write_pos += ((warp_1 > 12) ? cnt_w_11 : 0);
    int cnt_w_12 = s_m[4397];
    n_kept += cnt_w_12;
    write_pos += ((warp_1 > 13) ? cnt_w_12 : 0);
    int cnt_w_13 = s_m[4398];
    n_kept += cnt_w_13;
    write_pos += ((warp_1 > 14) ? cnt_w_13 : 0);
    int cnt_w_14 = s_m[4399];
    n_kept += cnt_w_14;
    write_pos += ((warp_1 > 15) ? cnt_w_14 : 0);
    #pragma unroll 1
    for (int e0_2 = 0; e0_2 < n; e0_2 += 4096) {
        unsigned long long ent_w[8];
        #pragma unroll
        for (int u_4 = 0; u_4 < 8; u_4++) {
            int idx_l_2 = e0_2 + tid + 512 * u_4;
            ent_w[u_4] = 4294967295;
            if (idx_l_2 < n) {
                ent_w[u_4] = (unsigned long long)Staging[base + (long long)idx_l_2];
            }
        }
        #pragma unroll
        for (int u_5 = 0; u_5 < 8; u_5++) {
            int kid_2 = (int)(unsigned int)(ent_w[u_5] & 4294967295);
            int kid_w = kid_2;
            int keep_w = 0;
            if (kid_w != -1) {
                unsigned int bits_2 = (unsigned int)(ent_w[u_5] >> 32);
                unsigned int magnitude_2 = bits_2 & 2147483647;
                unsigned int m_2 = ((magnitude_2 != 0) ? bits_2 : zero_u);
                unsigned int key32_2 = (((m_2 & sign_mask) != 0) ? ~m_2 : m_2 | sign_mask);
                if (magnitude_2 > nan_floor) {
                    key32_2 = zero_u;
                }
                unsigned long long key64_2 = (unsigned long long)key32_2 << 32 | (unsigned long long)(unsigned int)kid_w;
                keep_w = ((key64_2 >= edge) ? 1 : 0);
            }
            unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, keep_w != 0);
            unsigned int ballot_w = _vote_1;
            if (keep_w != 0) {
                int _popc_1 = __popc(ballot_w & lower_lanes);
                long long slot_w = row_base + (long long)(write_pos + _popc_1);
                float sc_w = 0.0f;
                unsigned int bits_3 = (unsigned int)(ent_w[u_5] >> 32);
                sc_w = reinterpret_cast<float*>(&bits_3)[0];
                Indices[slot_w] = kid_w;
                Scores[slot_w] = sc_w;
            }
            int _popc_2 = __popc(ballot_w);
            write_pos += _popc_2;
        }
    }
    unsigned int pad_bits = 4286578688;
    float pad_score = 0.0f;
    pad_score = reinterpret_cast<float*>(&pad_bits)[0];
    #pragma unroll 1
    for (int pos = n_kept + tid; pos < top_k; pos += 512) {
        long long slot_p = row_base + (long long)pos;
        Indices[slot_p] = -1;
        Scores[slot_p] = pad_score;
    }
}

} // extern "C"
