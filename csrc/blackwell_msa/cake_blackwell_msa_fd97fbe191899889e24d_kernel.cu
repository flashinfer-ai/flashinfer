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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_blackwell_msa_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_HIST_OFF 0
#define SMEM_HIST_STAGE_BYTES 65536
#define SMEM_HIST_STRIDE 65536
#define SMEM_HIST_EXT_OFF 0
#define SMEM_HIST_EXT_STAGE_BYTES 65600
#define SMEM_HIST_EXT_STRIDE 65600
#define SMEM_HIST64_OFF 0
#define SMEM_HIST64_STAGE_BYTES 65536
#define SMEM_HIST64_STRIDE 65536
#define SMEM_SEG_SUM_OFF 65536
#define SMEM_SEG_SUM_STAGE_BYTES 2048
#define SMEM_SEG_SUM_STRIDE 2048
#define SMEM_STATE_OFF 67584
#define SMEM_STATE_STAGE_BYTES 1024
#define SMEM_STATE_STRIDE 1024
#define SMEM_CAND_OFF 68608
#define SMEM_CAND_STAGE_BYTES 16384
#define SMEM_CAND_STRIDE 16384
#define SMEM_CAND_CNT_OFF 84992
#define SMEM_CAND_CNT_STAGE_BYTES 128
#define SMEM_CAND_CNT_STRIDE 128
#define SMEM_FLAGS_OFF 85248
#define SMEM_FLAGS_STAGE_BYTES 64
#define SMEM_FLAGS_STRIDE 64
#define SMEM_TOTAL 85376
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_blackwell_msa_fd97fbe191899889e24d(float* __restrict__ max_score, int* __restrict__ output, unsigned long long* __restrict__ partials, unsigned int* __restrict__ counters, unsigned int* __restrict__ block_max, int num_heads, int max_k_tiles, int total_q, int num_valid_pages, int force_begin_blocks, int force_end_blocks, int num_qtiles, int num_slices, int slice_len, int index_bits)
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
    unsigned int* hist = reinterpret_cast<unsigned int*>(smem_raw + SMEM_HIST_OFF);
    const int hist_addr = smem + SMEM_HIST_OFF;
    unsigned int* hist_ext = reinterpret_cast<unsigned int*>(smem_raw + SMEM_HIST_EXT_OFF);
    const int hist_ext_addr = smem + SMEM_HIST_EXT_OFF;
    unsigned long long* hist64 = reinterpret_cast<unsigned long long*>(smem_raw + SMEM_HIST64_OFF);
    const int hist64_addr = smem + SMEM_HIST64_OFF;
    unsigned int* seg_sum = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SEG_SUM_OFF);
    const int seg_sum_addr = smem + SMEM_SEG_SUM_OFF;
    unsigned int* state = reinterpret_cast<unsigned int*>(smem_raw + SMEM_STATE_OFF);
    const int state_addr = smem + SMEM_STATE_OFF;
    unsigned long long* cand = reinterpret_cast<unsigned long long*>(smem_raw + SMEM_CAND_OFF);
    const int cand_addr = smem + SMEM_CAND_OFF;
    unsigned int* cand_cnt = reinterpret_cast<unsigned int*>(smem_raw + SMEM_CAND_CNT_OFF);
    const int cand_cnt_addr = smem + SMEM_CAND_CNT_OFF;
    unsigned int* flags = reinterpret_cast<unsigned int*>(smem_raw + SMEM_FLAGS_OFF);
    const int flags_addr = smem + SMEM_FLAGS_OFF;

    // === Task calls (dependency order) ===
    int tid_0 = tid;
    int lane_1 = lane;
    int warp_2 = warp;
    int bid_3 = bid;
    int qtile = bid_3 % num_qtiles;
    int hs = bid_3 / num_qtiles;
    int s = hs % num_slices;
    int h = hs / num_slices;
    int group = h * num_qtiles + qtile;
    int q_local = lane_1 & 15;
    int q = qtile * 32 + q_local * 2;
    int qvalid = (int)(q < total_q);
    int r_sub = warp_2 * 2 + (lane_1 >> 4);
    int fb = force_begin_blocks;
    int fe = force_end_blocks;
    int r = 16 - fb - fe;
    int _max_0 = ((s * slice_len) > (fb) ? (s * slice_len) : (fb));
    int lo_s = _max_0;
    int _min_0 = (((s + 1) * slice_len) < (num_valid_pages - fe) ? ((s + 1) * slice_len) : (num_valid_pages - fe));
    int hi_s = _min_0;
    int _max_1 = ((hi_s - lo_s) > (0) ? (hi_s - lo_s) : (0));
    int slen = _max_1;
    int iters = slen + 15 >> 4;
    unsigned int one_u = 1;
    unsigned int idx_mask = (one_u << (unsigned int)index_bits) - one_u;
    int _min_1 = ((q) < (total_q - 2) ? (q) : (total_q - 2));
    int q_addr = _min_1;
    int row_base = h * max_k_tiles;
    int row0 = row_base + lo_s + r_sub;
    unsigned long long total_q64 = (unsigned long long)total_q;
    unsigned long long off0 = (unsigned long long)row0 * total_q64 + (unsigned long long)q_addr;
    unsigned long long q_addr64 = (unsigned long long)q_addr;
    unsigned long long row_base64 = (unsigned long long)row_base;
    unsigned long long stride16 = total_q64 * 16;
    int seg = tid_0 >> 4;
    int j = seg;
    int iters_nom = slice_len + 15 >> 4;
    int nchunks_nom = iters_nom + 8 - 1 >> 3;
    int bm_stride = nchunks_nom * 64;
    unsigned long long bm_base = (unsigned long long)bid_3 * (unsigned long long)bm_stride;
    int stage_only = (int)(slen <= 64);
    unsigned int mode0 = 0;
    if (r == 0) {
        mode0 = 2;
    } else if (stage_only != 0) {
        mode0 = 1;
    }
    if (stage_only == 0 && r > 0) {
        #pragma unroll
        for (int z = 0; z < 32; z++) {
            hist64[tid_0 + z * 256] = 0;
        }
    }
    if (tid_0 < 32) {
        cand_cnt[tid_0] = 0;
        unsigned int mode_t = mode0;
        if (qtile * 32 + tid_0 >= total_q) {
            mode_t = 2;
        }
        state[tid_0 * 8] = mode_t;
        state[tid_0 * 8 + 1] = 64;
        state[tid_0 * 8 + 2] = 0;
        state[tid_0 * 8 + 3] = 0;
        state[tid_0 * 8 + 4] = 0;
        state[tid_0 * 8 + 5] = 0;
    }
    __syncthreads();
    unsigned int any_open = (unsigned int)(mode0 != 2);
    float xs[16];
    int blks[8];
    unsigned int subs[4];
    unsigned int pre[2];
    unsigned int res[3];
    int nres[1];
    unsigned int subs1[4];
    unsigned int mode_v[2];
    unsigned int rshift_v[2];
    unsigned int ngt_v[2];
    unsigned int tbin_v[2];
    unsigned long long rprefix_v[2];
    int is_hist_v[2];
    int is_stage_v[2];
    int is_pass0_v[2];
    unsigned int w_v[2];
    unsigned long long dsh_v[2];
    unsigned int dmask_v[2];
    unsigned long long rsh_v[2];
    unsigned long long cthr_v[2];
    for (int p = 0; p < 11; p++) {
        if (any_open != 0) {
            int n_hist = 0;
            int n_stage = 0;
            unsigned int tmin = 1024;
            #pragma unroll
            for (int qq = 0; qq < 32; qq++) {
                unsigned int mq = state[qq * 8];
                n_hist = n_hist + (int)(mq == 0);
                n_stage = n_stage + (int)(mq == 1);
                unsigned int tq = state[qq * 8 + 5];
                unsigned int tcand = ((mq == 1) ? tq : 1024);
                unsigned int _min_2 = ((tmin) < (tcand) ? (tmin) : (tcand));
                tmin = _min_2;
            }
            int qw_s = q_local * 2;
            mode_v[0] = state[qw_s * 8];
            rshift_v[0] = state[qw_s * 8 + 1];
            unsigned int rp_lo_w = state[qw_s * 8 + 2];
            unsigned int rp_hi_w = state[qw_s * 8 + 3];
            ngt_v[0] = state[qw_s * 8 + 4];
            tbin_v[0] = state[qw_s * 8 + 5];
            rprefix_v[0] = (unsigned long long)rp_hi_w << 32 | (unsigned long long)rp_lo_w;
            is_hist_v[0] = (int)(mode_v[0] == 0);
            is_stage_v[0] = (int)(mode_v[0] == 1);
            is_pass0_v[0] = (int)(rshift_v[0] >= 64);
            unsigned int _min_3 = ((10) < (rshift_v[0]) ? (10) : (rshift_v[0]));
            w_v[0] = _min_3;
            if (rshift_v[0] > 22) {
                unsigned int _min_4 = ((10) < (rshift_v[0] - 22) ? (10) : (rshift_v[0] - 22));
                w_v[0] = _min_4;
            }
            dsh_v[0] = (unsigned long long)(rshift_v[0] - w_v[0]);
            dmask_v[0] = (one_u << w_v[0]) - one_u;
            unsigned int _min_5 = ((rshift_v[0]) < (63) ? (rshift_v[0]) : (63));
            rsh_v[0] = (unsigned long long)_min_5;
            unsigned int key_lo_w = 0;
            if (tbin_v[0] == 511) {
                key_lo_w = 1207435264;
            } else if (tbin_v[0] == 512) {
                key_lo_w = 2147483648u;
            } else {
                if (tbin_v[0] == 1023) {
                    key_lo_w = 3354918912u;
                }
            }
            unsigned long long rthr_w = rprefix_v[0] << rsh_v[0];
            unsigned long long klo_w = (unsigned long long)key_lo_w;
            unsigned int ilo_w = ~idx_mask;
            if (rshift_v[0] < 54) {
                klo_w = rthr_w >> 22 & 4294967295;
                ilo_w = (unsigned int)(rthr_w & 4194303) | ~idx_mask;
            }
            cthr_v[0] = klo_w << 32 | (unsigned long long)ilo_w;
            int qw_s_0 = q_local * 2 + 1;
            mode_v[1] = state[qw_s_0 * 8];
            rshift_v[1] = state[qw_s_0 * 8 + 1];
            unsigned int rp_lo_w_1 = state[qw_s_0 * 8 + 2];
            unsigned int rp_hi_w_2 = state[qw_s_0 * 8 + 3];
            ngt_v[1] = state[qw_s_0 * 8 + 4];
            tbin_v[1] = state[qw_s_0 * 8 + 5];
            rprefix_v[1] = (unsigned long long)rp_hi_w_2 << 32 | (unsigned long long)rp_lo_w_1;
            is_hist_v[1] = (int)(mode_v[1] == 0);
            is_stage_v[1] = (int)(mode_v[1] == 1);
            is_pass0_v[1] = (int)(rshift_v[1] >= 64);
            unsigned int _min_6 = ((10) < (rshift_v[1]) ? (10) : (rshift_v[1]));
            w_v[1] = _min_6;
            if (rshift_v[1] > 22) {
                unsigned int _min_7 = ((10) < (rshift_v[1] - 22) ? (10) : (rshift_v[1] - 22));
                w_v[1] = _min_7;
            }
            dsh_v[1] = (unsigned long long)(rshift_v[1] - w_v[1]);
            dmask_v[1] = (one_u << w_v[1]) - one_u;
            unsigned int _min_8 = ((rshift_v[1]) < (63) ? (rshift_v[1]) : (63));
            rsh_v[1] = (unsigned long long)_min_8;
            unsigned int key_lo_w_3 = 0;
            if (tbin_v[1] == 511) {
                key_lo_w_3 = 1207435264;
            } else if (tbin_v[1] == 512) {
                key_lo_w_3 = 2147483648u;
            } else {
                if (tbin_v[1] == 1023) {
                    key_lo_w_3 = 3354918912u;
                }
            }
            unsigned long long rthr_w_4 = rprefix_v[1] << rsh_v[1];
            unsigned long long klo_w_5 = (unsigned long long)key_lo_w_3;
            unsigned int ilo_w_6 = ~idx_mask;
            if (rshift_v[1] < 54) {
                klo_w_5 = rthr_w_4 >> 22 & 4294967295;
                ilo_w_6 = (unsigned int)(rthr_w_4 & 4194303) | ~idx_mask;
            }
            cthr_v[1] = klo_w_5 << 32 | (unsigned long long)ilo_w_6;
            if (n_stage == 0 && p == 0) {
                #pragma unroll 1
                for (int i0 = 0; i0 < iters; i0 += 8) {
                    unsigned long long off = off0 + (unsigned long long)i0 * stride16;
                    int blk = lo_s + i0 * 16 + r_sub;
                    blks[0] = blk;
                    int act = (int)(blk < hi_s) & qvalid;
                    xs[0] = __uint_as_float(2143289344);
                    xs[1] = __uint_as_float(2143289344);
                    if (act != 0) {
                        float _vec_load_0[2];
                        {
                            float2 _v2_0 = *reinterpret_cast<const float2*>(max_score + off);
                            _vec_load_0[0] = _v2_0.x;
                            _vec_load_0[0 + 1] = _v2_0.y;
                        }
                        xs[0] = _vec_load_0[0];
                        xs[1] = _vec_load_0[1];
                    }
                    off = off + stride16;
                    int blk_0 = lo_s + (i0 + 1) * 16 + r_sub;
                    blks[1] = blk_0;
                    int act_1 = (int)(blk_0 < hi_s) & qvalid;
                    xs[2] = __uint_as_float(2143289344);
                    xs[3] = __uint_as_float(2143289344);
                    if (act_1 != 0) {
                        float _vec_load_1[2];
                        {
                            float2 _v2_1 = *reinterpret_cast<const float2*>(max_score + off);
                            _vec_load_1[0] = _v2_1.x;
                            _vec_load_1[0 + 1] = _v2_1.y;
                        }
                        xs[2] = _vec_load_1[0];
                        xs[3] = _vec_load_1[1];
                    }
                    off = off + stride16;
                    int blk_2 = lo_s + (i0 + 2) * 16 + r_sub;
                    blks[2] = blk_2;
                    int act_3 = (int)(blk_2 < hi_s) & qvalid;
                    xs[4] = __uint_as_float(2143289344);
                    xs[5] = __uint_as_float(2143289344);
                    if (act_3 != 0) {
                        float _vec_load_2[2];
                        {
                            float2 _v2_2 = *reinterpret_cast<const float2*>(max_score + off);
                            _vec_load_2[0] = _v2_2.x;
                            _vec_load_2[0 + 1] = _v2_2.y;
                        }
                        xs[4] = _vec_load_2[0];
                        xs[5] = _vec_load_2[1];
                    }
                    off = off + stride16;
                    int blk_4 = lo_s + (i0 + 3) * 16 + r_sub;
                    blks[3] = blk_4;
                    int act_5 = (int)(blk_4 < hi_s) & qvalid;
                    xs[6] = __uint_as_float(2143289344);
                    xs[7] = __uint_as_float(2143289344);
                    if (act_5 != 0) {
                        float _vec_load_3[2];
                        {
                            float2 _v2_3 = *reinterpret_cast<const float2*>(max_score + off);
                            _vec_load_3[0] = _v2_3.x;
                            _vec_load_3[0 + 1] = _v2_3.y;
                        }
                        xs[6] = _vec_load_3[0];
                        xs[7] = _vec_load_3[1];
                    }
                    off = off + stride16;
                    int blk_6 = lo_s + (i0 + 4) * 16 + r_sub;
                    blks[4] = blk_6;
                    int act_7 = (int)(blk_6 < hi_s) & qvalid;
                    xs[8] = __uint_as_float(2143289344);
                    xs[9] = __uint_as_float(2143289344);
                    if (act_7 != 0) {
                        float _vec_load_4[2];
                        {
                            float2 _v2_4 = *reinterpret_cast<const float2*>(max_score + off);
                            _vec_load_4[0] = _v2_4.x;
                            _vec_load_4[0 + 1] = _v2_4.y;
                        }
                        xs[8] = _vec_load_4[0];
                        xs[9] = _vec_load_4[1];
                    }
                    off = off + stride16;
                    int blk_8 = lo_s + (i0 + 5) * 16 + r_sub;
                    blks[5] = blk_8;
                    int act_9 = (int)(blk_8 < hi_s) & qvalid;
                    xs[10] = __uint_as_float(2143289344);
                    xs[11] = __uint_as_float(2143289344);
                    if (act_9 != 0) {
                        float _vec_load_5[2];
                        {
                            float2 _v2_5 = *reinterpret_cast<const float2*>(max_score + off);
                            _vec_load_5[0] = _v2_5.x;
                            _vec_load_5[0 + 1] = _v2_5.y;
                        }
                        xs[10] = _vec_load_5[0];
                        xs[11] = _vec_load_5[1];
                    }
                    off = off + stride16;
                    int blk_10 = lo_s + (i0 + 6) * 16 + r_sub;
                    blks[6] = blk_10;
                    int act_11 = (int)(blk_10 < hi_s) & qvalid;
                    xs[12] = __uint_as_float(2143289344);
                    xs[13] = __uint_as_float(2143289344);
                    if (act_11 != 0) {
                        float _vec_load_6[2];
                        {
                            float2 _v2_6 = *reinterpret_cast<const float2*>(max_score + off);
                            _vec_load_6[0] = _v2_6.x;
                            _vec_load_6[0 + 1] = _v2_6.y;
                        }
                        xs[12] = _vec_load_6[0];
                        xs[13] = _vec_load_6[1];
                    }
                    off = off + stride16;
                    int blk_12 = lo_s + (i0 + 7) * 16 + r_sub;
                    blks[7] = blk_12;
                    int act_13 = (int)(blk_12 < hi_s) & qvalid;
                    xs[14] = __uint_as_float(2143289344);
                    xs[15] = __uint_as_float(2143289344);
                    if (act_13 != 0) {
                        float _vec_load_7[2];
                        {
                            float2 _v2_7 = *reinterpret_cast<const float2*>(max_score + off);
                            _vec_load_7[0] = _v2_7.x;
                            _vec_load_7[0 + 1] = _v2_7.y;
                        }
                        xs[14] = _vec_load_7[0];
                        xs[15] = _vec_load_7[1];
                    }
                    off = off + stride16;
                    int chunk0 = i0 >> 3;
                    unsigned long long bm_idx = bm_base + (unsigned long long)((chunk0 * 16 + r_sub) * 4);
                    int m[8];
                    int mx = 0;
                    unsigned int raw = __as_u32(xs[0]);
                    unsigned int mag = raw & 2147483647;
                    unsigned int nan = (unsigned int)(mag > 2139095040);
                    unsigned int bits = ((mag != 0) ? raw : 0);
                    int em = (int)(bits >> 19 & 4095);
                    int _max_2 = ((em - 1792) > (0) ? (em - 1792) : (0));
                    int _min_9 = ((_max_2) < (511) ? (_max_2) : (511));
                    int raw_14 = _min_9;
                    int neg = (int)(bits >> 31);
                    int h10 = ((neg != 0) ? 511 - raw_14 : 512 + raw_14);
                    int hb_w = ((nan != 0) ? 1024 : h10);
                    atomicAdd(&hist_ext[hb_w * 16 + q_local], 1);
                    int hm_w = ((nan != 0) ? 0 : h10);
                    int _max_3 = ((mx) > (hm_w) ? (mx) : (hm_w));
                    mx = _max_3;
                    unsigned int raw_15 = __as_u32(xs[1]);
                    unsigned int mag_16 = raw_15 & 2147483647;
                    unsigned int nan_17 = (unsigned int)(mag_16 > 2139095040);
                    unsigned int bits_18 = ((mag_16 != 0) ? raw_15 : 0);
                    int em_19 = (int)(bits_18 >> 19 & 4095);
                    int _max_4 = ((em_19 - 1792) > (0) ? (em_19 - 1792) : (0));
                    int _min_10 = ((_max_4) < (511) ? (_max_4) : (511));
                    int raw_20 = _min_10;
                    int neg_21 = (int)(bits_18 >> 31);
                    int h10_22 = ((neg_21 != 0) ? 511 - raw_20 : 512 + raw_20);
                    int hb_w_23 = ((nan_17 != 0) ? 1024 : h10_22);
                    atomicAdd(&hist_ext[hb_w_23 * 16 + q_local], 65536);
                    int hm_w_24 = ((nan_17 != 0) ? 0 : h10_22);
                    int _max_5 = ((mx) > (hm_w_24) ? (mx) : (hm_w_24));
                    mx = _max_5;
                    m[0] = mx;
                    int mx_25 = 0;
                    unsigned int raw_26 = __as_u32(xs[2]);
                    unsigned int mag_27 = raw_26 & 2147483647;
                    unsigned int nan_28 = (unsigned int)(mag_27 > 2139095040);
                    unsigned int bits_29 = ((mag_27 != 0) ? raw_26 : 0);
                    int em_30 = (int)(bits_29 >> 19 & 4095);
                    int _max_22 = ((em_30 - 1792) > (0) ? (em_30 - 1792) : (0));
                    int _min_11 = ((_max_22) < (511) ? (_max_22) : (511));
                    int raw_31 = _min_11;
                    int neg_32 = (int)(bits_29 >> 31);
                    int h10_33 = ((neg_32 != 0) ? 511 - raw_31 : 512 + raw_31);
                    int hb_w_34 = ((nan_28 != 0) ? 1024 : h10_33);
                    atomicAdd(&hist_ext[hb_w_34 * 16 + q_local], 1);
                    int hm_w_35 = ((nan_28 != 0) ? 0 : h10_33);
                    int _max_23 = ((mx_25) > (hm_w_35) ? (mx_25) : (hm_w_35));
                    mx_25 = _max_23;
                    unsigned int raw_36 = __as_u32(xs[3]);
                    unsigned int mag_37 = raw_36 & 2147483647;
                    unsigned int nan_38 = (unsigned int)(mag_37 > 2139095040);
                    unsigned int bits_39 = ((mag_37 != 0) ? raw_36 : 0);
                    int em_40 = (int)(bits_39 >> 19 & 4095);
                    int _max_24 = ((em_40 - 1792) > (0) ? (em_40 - 1792) : (0));
                    int _min_12 = ((_max_24) < (511) ? (_max_24) : (511));
                    int raw_41 = _min_12;
                    int neg_42 = (int)(bits_39 >> 31);
                    int h10_43 = ((neg_42 != 0) ? 511 - raw_41 : 512 + raw_41);
                    int hb_w_44 = ((nan_38 != 0) ? 1024 : h10_43);
                    atomicAdd(&hist_ext[hb_w_44 * 16 + q_local], 65536);
                    int hm_w_45 = ((nan_38 != 0) ? 0 : h10_43);
                    int _max_25 = ((mx_25) > (hm_w_45) ? (mx_25) : (hm_w_45));
                    mx_25 = _max_25;
                    m[1] = mx_25;
                    int mx_46 = 0;
                    unsigned int raw_47 = __as_u32(xs[4]);
                    unsigned int mag_48 = raw_47 & 2147483647;
                    unsigned int nan_49 = (unsigned int)(mag_48 > 2139095040);
                    unsigned int bits_50 = ((mag_48 != 0) ? raw_47 : 0);
                    int em_51 = (int)(bits_50 >> 19 & 4095);
                    int _max_42 = ((em_51 - 1792) > (0) ? (em_51 - 1792) : (0));
                    int _min_13 = ((_max_42) < (511) ? (_max_42) : (511));
                    int raw_52 = _min_13;
                    int neg_53 = (int)(bits_50 >> 31);
                    int h10_54 = ((neg_53 != 0) ? 511 - raw_52 : 512 + raw_52);
                    int hb_w_55 = ((nan_49 != 0) ? 1024 : h10_54);
                    atomicAdd(&hist_ext[hb_w_55 * 16 + q_local], 1);
                    int hm_w_56 = ((nan_49 != 0) ? 0 : h10_54);
                    int _max_43 = ((mx_46) > (hm_w_56) ? (mx_46) : (hm_w_56));
                    mx_46 = _max_43;
                    unsigned int raw_57 = __as_u32(xs[5]);
                    unsigned int mag_58 = raw_57 & 2147483647;
                    unsigned int nan_59 = (unsigned int)(mag_58 > 2139095040);
                    unsigned int bits_60 = ((mag_58 != 0) ? raw_57 : 0);
                    int em_61 = (int)(bits_60 >> 19 & 4095);
                    int _max_44 = ((em_61 - 1792) > (0) ? (em_61 - 1792) : (0));
                    int _min_14 = ((_max_44) < (511) ? (_max_44) : (511));
                    int raw_62 = _min_14;
                    int neg_63 = (int)(bits_60 >> 31);
                    int h10_64 = ((neg_63 != 0) ? 511 - raw_62 : 512 + raw_62);
                    int hb_w_65 = ((nan_59 != 0) ? 1024 : h10_64);
                    atomicAdd(&hist_ext[hb_w_65 * 16 + q_local], 65536);
                    int hm_w_66 = ((nan_59 != 0) ? 0 : h10_64);
                    int _max_45 = ((mx_46) > (hm_w_66) ? (mx_46) : (hm_w_66));
                    mx_46 = _max_45;
                    m[2] = mx_46;
                    int mx_67 = 0;
                    unsigned int raw_68 = __as_u32(xs[6]);
                    unsigned int mag_69 = raw_68 & 2147483647;
                    unsigned int nan_70 = (unsigned int)(mag_69 > 2139095040);
                    unsigned int bits_71 = ((mag_69 != 0) ? raw_68 : 0);
                    int em_72 = (int)(bits_71 >> 19 & 4095);
                    int _max_62 = ((em_72 - 1792) > (0) ? (em_72 - 1792) : (0));
                    int _min_15 = ((_max_62) < (511) ? (_max_62) : (511));
                    int raw_73 = _min_15;
                    int neg_74 = (int)(bits_71 >> 31);
                    int h10_75 = ((neg_74 != 0) ? 511 - raw_73 : 512 + raw_73);
                    int hb_w_76 = ((nan_70 != 0) ? 1024 : h10_75);
                    atomicAdd(&hist_ext[hb_w_76 * 16 + q_local], 1);
                    int hm_w_77 = ((nan_70 != 0) ? 0 : h10_75);
                    int _max_63 = ((mx_67) > (hm_w_77) ? (mx_67) : (hm_w_77));
                    mx_67 = _max_63;
                    unsigned int raw_78 = __as_u32(xs[7]);
                    unsigned int mag_79 = raw_78 & 2147483647;
                    unsigned int nan_80 = (unsigned int)(mag_79 > 2139095040);
                    unsigned int bits_81 = ((mag_79 != 0) ? raw_78 : 0);
                    int em_82 = (int)(bits_81 >> 19 & 4095);
                    int _max_64 = ((em_82 - 1792) > (0) ? (em_82 - 1792) : (0));
                    int _min_16 = ((_max_64) < (511) ? (_max_64) : (511));
                    int raw_83 = _min_16;
                    int neg_84 = (int)(bits_81 >> 31);
                    int h10_85 = ((neg_84 != 0) ? 511 - raw_83 : 512 + raw_83);
                    int hb_w_86 = ((nan_80 != 0) ? 1024 : h10_85);
                    atomicAdd(&hist_ext[hb_w_86 * 16 + q_local], 65536);
                    int hm_w_87 = ((nan_80 != 0) ? 0 : h10_85);
                    int _max_65 = ((mx_67) > (hm_w_87) ? (mx_67) : (hm_w_87));
                    mx_67 = _max_65;
                    m[3] = mx_67;
                    int mx_88 = 0;
                    unsigned int raw_89 = __as_u32(xs[8]);
                    unsigned int mag_90 = raw_89 & 2147483647;
                    unsigned int nan_91 = (unsigned int)(mag_90 > 2139095040);
                    unsigned int bits_92 = ((mag_90 != 0) ? raw_89 : 0);
                    int em_93 = (int)(bits_92 >> 19 & 4095);
                    int _max_82 = ((em_93 - 1792) > (0) ? (em_93 - 1792) : (0));
                    int _min_17 = ((_max_82) < (511) ? (_max_82) : (511));
                    int raw_94 = _min_17;
                    int neg_95 = (int)(bits_92 >> 31);
                    int h10_96 = ((neg_95 != 0) ? 511 - raw_94 : 512 + raw_94);
                    int hb_w_97 = ((nan_91 != 0) ? 1024 : h10_96);
                    atomicAdd(&hist_ext[hb_w_97 * 16 + q_local], 1);
                    int hm_w_98 = ((nan_91 != 0) ? 0 : h10_96);
                    int _max_83 = ((mx_88) > (hm_w_98) ? (mx_88) : (hm_w_98));
                    mx_88 = _max_83;
                    unsigned int raw_99 = __as_u32(xs[9]);
                    unsigned int mag_100 = raw_99 & 2147483647;
                    unsigned int nan_101 = (unsigned int)(mag_100 > 2139095040);
                    unsigned int bits_102 = ((mag_100 != 0) ? raw_99 : 0);
                    int em_103 = (int)(bits_102 >> 19 & 4095);
                    int _max_84 = ((em_103 - 1792) > (0) ? (em_103 - 1792) : (0));
                    int _min_18 = ((_max_84) < (511) ? (_max_84) : (511));
                    int raw_104 = _min_18;
                    int neg_105 = (int)(bits_102 >> 31);
                    int h10_106 = ((neg_105 != 0) ? 511 - raw_104 : 512 + raw_104);
                    int hb_w_107 = ((nan_101 != 0) ? 1024 : h10_106);
                    atomicAdd(&hist_ext[hb_w_107 * 16 + q_local], 65536);
                    int hm_w_108 = ((nan_101 != 0) ? 0 : h10_106);
                    int _max_85 = ((mx_88) > (hm_w_108) ? (mx_88) : (hm_w_108));
                    mx_88 = _max_85;
                    m[4] = mx_88;
                    int mx_109 = 0;
                    unsigned int raw_110 = __as_u32(xs[10]);
                    unsigned int mag_111 = raw_110 & 2147483647;
                    unsigned int nan_112 = (unsigned int)(mag_111 > 2139095040);
                    unsigned int bits_113 = ((mag_111 != 0) ? raw_110 : 0);
                    int em_114 = (int)(bits_113 >> 19 & 4095);
                    int _max_102 = ((em_114 - 1792) > (0) ? (em_114 - 1792) : (0));
                    int _min_19 = ((_max_102) < (511) ? (_max_102) : (511));
                    int raw_115 = _min_19;
                    int neg_116 = (int)(bits_113 >> 31);
                    int h10_117 = ((neg_116 != 0) ? 511 - raw_115 : 512 + raw_115);
                    int hb_w_118 = ((nan_112 != 0) ? 1024 : h10_117);
                    atomicAdd(&hist_ext[hb_w_118 * 16 + q_local], 1);
                    int hm_w_119 = ((nan_112 != 0) ? 0 : h10_117);
                    int _max_103 = ((mx_109) > (hm_w_119) ? (mx_109) : (hm_w_119));
                    mx_109 = _max_103;
                    unsigned int raw_120 = __as_u32(xs[11]);
                    unsigned int mag_121 = raw_120 & 2147483647;
                    unsigned int nan_122 = (unsigned int)(mag_121 > 2139095040);
                    unsigned int bits_123 = ((mag_121 != 0) ? raw_120 : 0);
                    int em_124 = (int)(bits_123 >> 19 & 4095);
                    int _max_104 = ((em_124 - 1792) > (0) ? (em_124 - 1792) : (0));
                    int _min_20 = ((_max_104) < (511) ? (_max_104) : (511));
                    int raw_125 = _min_20;
                    int neg_126 = (int)(bits_123 >> 31);
                    int h10_127 = ((neg_126 != 0) ? 511 - raw_125 : 512 + raw_125);
                    int hb_w_128 = ((nan_122 != 0) ? 1024 : h10_127);
                    atomicAdd(&hist_ext[hb_w_128 * 16 + q_local], 65536);
                    int hm_w_129 = ((nan_122 != 0) ? 0 : h10_127);
                    int _max_105 = ((mx_109) > (hm_w_129) ? (mx_109) : (hm_w_129));
                    mx_109 = _max_105;
                    m[5] = mx_109;
                    int mx_130 = 0;
                    unsigned int raw_131 = __as_u32(xs[12]);
                    unsigned int mag_132 = raw_131 & 2147483647;
                    unsigned int nan_133 = (unsigned int)(mag_132 > 2139095040);
                    unsigned int bits_134 = ((mag_132 != 0) ? raw_131 : 0);
                    int em_135 = (int)(bits_134 >> 19 & 4095);
                    int _max_122 = ((em_135 - 1792) > (0) ? (em_135 - 1792) : (0));
                    int _min_21 = ((_max_122) < (511) ? (_max_122) : (511));
                    int raw_136 = _min_21;
                    int neg_137 = (int)(bits_134 >> 31);
                    int h10_138 = ((neg_137 != 0) ? 511 - raw_136 : 512 + raw_136);
                    int hb_w_139 = ((nan_133 != 0) ? 1024 : h10_138);
                    atomicAdd(&hist_ext[hb_w_139 * 16 + q_local], 1);
                    int hm_w_140 = ((nan_133 != 0) ? 0 : h10_138);
                    int _max_123 = ((mx_130) > (hm_w_140) ? (mx_130) : (hm_w_140));
                    mx_130 = _max_123;
                    unsigned int raw_141 = __as_u32(xs[13]);
                    unsigned int mag_142 = raw_141 & 2147483647;
                    unsigned int nan_143 = (unsigned int)(mag_142 > 2139095040);
                    unsigned int bits_144 = ((mag_142 != 0) ? raw_141 : 0);
                    int em_145 = (int)(bits_144 >> 19 & 4095);
                    int _max_124 = ((em_145 - 1792) > (0) ? (em_145 - 1792) : (0));
                    int _min_22 = ((_max_124) < (511) ? (_max_124) : (511));
                    int raw_146 = _min_22;
                    int neg_147 = (int)(bits_144 >> 31);
                    int h10_148 = ((neg_147 != 0) ? 511 - raw_146 : 512 + raw_146);
                    int hb_w_149 = ((nan_143 != 0) ? 1024 : h10_148);
                    atomicAdd(&hist_ext[hb_w_149 * 16 + q_local], 65536);
                    int hm_w_150 = ((nan_143 != 0) ? 0 : h10_148);
                    int _max_125 = ((mx_130) > (hm_w_150) ? (mx_130) : (hm_w_150));
                    mx_130 = _max_125;
                    m[6] = mx_130;
                    int mx_151 = 0;
                    unsigned int raw_152 = __as_u32(xs[14]);
                    unsigned int mag_153 = raw_152 & 2147483647;
                    unsigned int nan_154 = (unsigned int)(mag_153 > 2139095040);
                    unsigned int bits_155 = ((mag_153 != 0) ? raw_152 : 0);
                    int em_156 = (int)(bits_155 >> 19 & 4095);
                    int _max_142 = ((em_156 - 1792) > (0) ? (em_156 - 1792) : (0));
                    int _min_23 = ((_max_142) < (511) ? (_max_142) : (511));
                    int raw_157 = _min_23;
                    int neg_158 = (int)(bits_155 >> 31);
                    int h10_159 = ((neg_158 != 0) ? 511 - raw_157 : 512 + raw_157);
                    int hb_w_160 = ((nan_154 != 0) ? 1024 : h10_159);
                    atomicAdd(&hist_ext[hb_w_160 * 16 + q_local], 1);
                    int hm_w_161 = ((nan_154 != 0) ? 0 : h10_159);
                    int _max_143 = ((mx_151) > (hm_w_161) ? (mx_151) : (hm_w_161));
                    mx_151 = _max_143;
                    unsigned int raw_162 = __as_u32(xs[15]);
                    unsigned int mag_163 = raw_162 & 2147483647;
                    unsigned int nan_164 = (unsigned int)(mag_163 > 2139095040);
                    unsigned int bits_165 = ((mag_163 != 0) ? raw_162 : 0);
                    int em_166 = (int)(bits_165 >> 19 & 4095);
                    int _max_144 = ((em_166 - 1792) > (0) ? (em_166 - 1792) : (0));
                    int _min_24 = ((_max_144) < (511) ? (_max_144) : (511));
                    int raw_167 = _min_24;
                    int neg_168 = (int)(bits_165 >> 31);
                    int h10_169 = ((neg_168 != 0) ? 511 - raw_167 : 512 + raw_167);
                    int hb_w_170 = ((nan_164 != 0) ? 1024 : h10_169);
                    atomicAdd(&hist_ext[hb_w_170 * 16 + q_local], 65536);
                    int hm_w_171 = ((nan_164 != 0) ? 0 : h10_169);
                    int _max_145 = ((mx_151) > (hm_w_171) ? (mx_151) : (hm_w_171));
                    mx_151 = _max_145;
                    m[7] = mx_151;
                    {
                        int b3 = lane_1 & 8;
                        int b2 = lane_1 & 4;
                        int b1 = lane_1 & 2;
                        int a[4];
                        int send = ((b3 != 0) ? m[0] : m[4]);
                        int keep = ((b3 != 0) ? m[4] : m[0]);
                        int _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, send, 8);
                        int recv = _shfl_xor_126;
                        int _max_146 = ((keep) > (recv) ? (keep) : (recv));
                        a[0] = _max_146;
                        int send_0 = ((b3 != 0) ? m[1] : m[5]);
                        int keep_1 = ((b3 != 0) ? m[5] : m[1]);
                        int _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, send_0, 8);
                        int recv_2 = _shfl_xor_127;
                        int _max_147 = ((keep_1) > (recv_2) ? (keep_1) : (recv_2));
                        a[1] = _max_147;
                        int send_3 = ((b3 != 0) ? m[2] : m[6]);
                        int keep_4 = ((b3 != 0) ? m[6] : m[2]);
                        int _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, send_3, 8);
                        int recv_5 = _shfl_xor_128;
                        int _max_148 = ((keep_4) > (recv_5) ? (keep_4) : (recv_5));
                        a[2] = _max_148;
                        int send_6 = ((b3 != 0) ? m[3] : m[7]);
                        int keep_7 = ((b3 != 0) ? m[7] : m[3]);
                        int _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, send_6, 8);
                        int recv_8 = _shfl_xor_129;
                        int _max_149 = ((keep_7) > (recv_8) ? (keep_7) : (recv_8));
                        a[3] = _max_149;
                        int b[2];
                        int send2 = ((b2 != 0) ? a[0] : a[2]);
                        int keep2 = ((b2 != 0) ? a[2] : a[0]);
                        int _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, send2, 4);
                        int recv2 = _shfl_xor_130;
                        int _max_150 = ((keep2) > (recv2) ? (keep2) : (recv2));
                        b[0] = _max_150;
                        int send2_9 = ((b2 != 0) ? a[1] : a[3]);
                        int keep2_10 = ((b2 != 0) ? a[3] : a[1]);
                        int _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, send2_9, 4);
                        int recv2_11 = _shfl_xor_131;
                        int _max_151 = ((keep2_10) > (recv2_11) ? (keep2_10) : (recv2_11));
                        b[1] = _max_151;
                        int send3 = ((b1 != 0) ? b[0] : b[1]);
                        int keep3 = ((b1 != 0) ? b[1] : b[0]);
                        int _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, send3, 2);
                        int recv3 = _shfl_xor_132;
                        int _max_152 = ((keep3) > (recv3) ? (keep3) : (recv3));
                        int c = _max_152;
                        int _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, c, 1);
                        int recv4 = _shfl_xor_133;
                        int _max_153 = ((c) > (recv4) ? (c) : (recv4));
                        int d = _max_153;
                        int _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, d, 2);
                        int other = _shfl_xor_134;
                        int word = ((b1 == 0) ? d | other << 16 : other | d << 16);
                        block_max[bm_idx + (unsigned long long)(lane_1 >> 2 & 3)] = (unsigned int)word;
                    }
                }
            } else if (n_hist == 0) {
                if (p > 0 && slen > 224) {
                    int nchunks = iters + 8 - 1 >> 3;
                    for (int w0 = 0; w0 < slen; w0 += 8192) {
                        int c_lo = w0 >> 7;
                        int _min_25 = ((nchunks) < (w0 + 8192 >> 7) ? (nchunks) : (w0 + 8192 >> 7));
                        int c_hi = _min_25;
                        int n_ent = (c_hi - c_lo) * 16;
                        int cnt = 0;
                        int masks[4];
                        int js[4];
                        int e = tid_0;
                        int c_1 = c_lo + (e >> 4);
                        int r_0 = e & 15;
                        js[0] = c_1 * 128 + r_0;
                        masks[0] = 0;
                        if (e < n_ent) {
                            unsigned int _vec_load_8[4];
                            {
                                uint4 _uv4_8 = *reinterpret_cast<const uint4*>(block_max + bm_base + (unsigned long long)((c_1 * 16 + r_0) * 4));
                                _vec_load_8[0 + 0] = _uv4_8.x;
                                _vec_load_8[0 + 1] = _uv4_8.y;
                                _vec_load_8[0 + 2] = _uv4_8.z;
                                _vec_load_8[0 + 3] = _uv4_8.w;
                            }
                            int mk = 0;
                            unsigned int mu = _vec_load_8[0] & 65535;
                            int j_0 = js[0];
                            int keep_2 = (int)(mu >= tmin && j_0 < slen);
                            mk = mk | keep_2;
                            unsigned int mu_1 = _vec_load_8[0] >> 16 & 65535;
                            int j_2 = js[0] + 16;
                            int keep_3 = (int)(mu_1 >= tmin && j_2 < slen);
                            mk = mk | keep_3 << 1;
                            unsigned int mu_4 = _vec_load_8[1] & 65535;
                            int j_5 = js[0] + 32;
                            int keep_6 = (int)(mu_4 >= tmin && j_5 < slen);
                            mk = mk | keep_6 << 2;
                            unsigned int mu_7 = _vec_load_8[1] >> 16 & 65535;
                            int j_8 = js[0] + 48;
                            int keep_9 = (int)(mu_7 >= tmin && j_8 < slen);
                            mk = mk | keep_9 << 3;
                            unsigned int mu_10 = _vec_load_8[2] & 65535;
                            int j_11 = js[0] + 64;
                            int keep_12 = (int)(mu_10 >= tmin && j_11 < slen);
                            mk = mk | keep_12 << 4;
                            unsigned int mu_13 = _vec_load_8[2] >> 16 & 65535;
                            int j_14 = js[0] + 80;
                            int keep_15 = (int)(mu_13 >= tmin && j_14 < slen);
                            mk = mk | keep_15 << 5;
                            unsigned int mu_16 = _vec_load_8[3] & 65535;
                            int j_17 = js[0] + 96;
                            int keep_18 = (int)(mu_16 >= tmin && j_17 < slen);
                            mk = mk | keep_18 << 6;
                            unsigned int mu_19 = _vec_load_8[3] >> 16 & 65535;
                            int j_20 = js[0] + 112;
                            int keep_21 = (int)(mu_19 >= tmin && j_20 < slen);
                            mk = mk | keep_21 << 7;
                            masks[0] = mk;
                            int _popc_0 = __popc(mk);
                            cnt = cnt + _popc_0;
                        }
                        int e_1 = tid_0 + 256;
                        int c_2 = c_lo + (e_1 >> 4);
                        int r_3 = e_1 & 15;
                        js[1] = c_2 * 128 + r_3;
                        masks[1] = 0;
                        if (e_1 < n_ent) {
                            unsigned int _vec_load_9[4];
                            {
                                uint4 _uv4_9 = *reinterpret_cast<const uint4*>(block_max + bm_base + (unsigned long long)((c_2 * 16 + r_3) * 4));
                                _vec_load_9[0 + 0] = _uv4_9.x;
                                _vec_load_9[0 + 1] = _uv4_9.y;
                                _vec_load_9[0 + 2] = _uv4_9.z;
                                _vec_load_9[0 + 3] = _uv4_9.w;
                            }
                            int mk_1 = 0;
                            unsigned int mu_2 = _vec_load_9[0] & 65535;
                            int j_0_1 = js[1];
                            int keep_5 = (int)(mu_2 >= tmin && j_0_1 < slen);
                            mk_1 = mk_1 | keep_5;
                            unsigned int mu_1_1 = _vec_load_9[0] >> 16 & 65535;
                            int j_2_1 = js[1] + 16;
                            int keep_3_1 = (int)(mu_1_1 >= tmin && j_2_1 < slen);
                            mk_1 = mk_1 | keep_3_1 << 1;
                            unsigned int mu_4_1 = _vec_load_9[1] & 65535;
                            int j_5_1 = js[1] + 32;
                            int keep_6_1 = (int)(mu_4_1 >= tmin && j_5_1 < slen);
                            mk_1 = mk_1 | keep_6_1 << 2;
                            unsigned int mu_7_1 = _vec_load_9[1] >> 16 & 65535;
                            int j_8_1 = js[1] + 48;
                            int keep_9_1 = (int)(mu_7_1 >= tmin && j_8_1 < slen);
                            mk_1 = mk_1 | keep_9_1 << 3;
                            unsigned int mu_10_1 = _vec_load_9[2] & 65535;
                            int j_11_1 = js[1] + 64;
                            int keep_12_1 = (int)(mu_10_1 >= tmin && j_11_1 < slen);
                            mk_1 = mk_1 | keep_12_1 << 4;
                            unsigned int mu_13_1 = _vec_load_9[2] >> 16 & 65535;
                            int j_14_1 = js[1] + 80;
                            int keep_15_1 = (int)(mu_13_1 >= tmin && j_14_1 < slen);
                            mk_1 = mk_1 | keep_15_1 << 5;
                            unsigned int mu_16_1 = _vec_load_9[3] & 65535;
                            int j_17_1 = js[1] + 96;
                            int keep_18_1 = (int)(mu_16_1 >= tmin && j_17_1 < slen);
                            mk_1 = mk_1 | keep_18_1 << 6;
                            unsigned int mu_19_1 = _vec_load_9[3] >> 16 & 65535;
                            int j_20_1 = js[1] + 112;
                            int keep_21_1 = (int)(mu_19_1 >= tmin && j_20_1 < slen);
                            mk_1 = mk_1 | keep_21_1 << 7;
                            masks[1] = mk_1;
                            int _popc_1 = __popc(mk_1);
                            cnt = cnt + _popc_1;
                        }
                        int e_4 = tid_0 + 512;
                        int c_5 = c_lo + (e_4 >> 4);
                        int r_6 = e_4 & 15;
                        js[2] = c_5 * 128 + r_6;
                        masks[2] = 0;
                        if (e_4 < n_ent) {
                            unsigned int _vec_load_10[4];
                            {
                                uint4 _uv4_10 = *reinterpret_cast<const uint4*>(block_max + bm_base + (unsigned long long)((c_5 * 16 + r_6) * 4));
                                _vec_load_10[0 + 0] = _uv4_10.x;
                                _vec_load_10[0 + 1] = _uv4_10.y;
                                _vec_load_10[0 + 2] = _uv4_10.z;
                                _vec_load_10[0 + 3] = _uv4_10.w;
                            }
                            int mk_2 = 0;
                            unsigned int mu_3 = _vec_load_10[0] & 65535;
                            int j_0_2 = js[2];
                            int keep_8 = (int)(mu_3 >= tmin && j_0_2 < slen);
                            mk_2 = mk_2 | keep_8;
                            unsigned int mu_1_2 = _vec_load_10[0] >> 16 & 65535;
                            int j_2_2 = js[2] + 16;
                            int keep_3_2 = (int)(mu_1_2 >= tmin && j_2_2 < slen);
                            mk_2 = mk_2 | keep_3_2 << 1;
                            unsigned int mu_4_2 = _vec_load_10[1] & 65535;
                            int j_5_2 = js[2] + 32;
                            int keep_6_2 = (int)(mu_4_2 >= tmin && j_5_2 < slen);
                            mk_2 = mk_2 | keep_6_2 << 2;
                            unsigned int mu_7_2 = _vec_load_10[1] >> 16 & 65535;
                            int j_8_2 = js[2] + 48;
                            int keep_9_2 = (int)(mu_7_2 >= tmin && j_8_2 < slen);
                            mk_2 = mk_2 | keep_9_2 << 3;
                            unsigned int mu_10_2 = _vec_load_10[2] & 65535;
                            int j_11_2 = js[2] + 64;
                            int keep_12_2 = (int)(mu_10_2 >= tmin && j_11_2 < slen);
                            mk_2 = mk_2 | keep_12_2 << 4;
                            unsigned int mu_13_2 = _vec_load_10[2] >> 16 & 65535;
                            int j_14_2 = js[2] + 80;
                            int keep_15_2 = (int)(mu_13_2 >= tmin && j_14_2 < slen);
                            mk_2 = mk_2 | keep_15_2 << 5;
                            unsigned int mu_16_2 = _vec_load_10[3] & 65535;
                            int j_17_2 = js[2] + 96;
                            int keep_18_2 = (int)(mu_16_2 >= tmin && j_17_2 < slen);
                            mk_2 = mk_2 | keep_18_2 << 6;
                            unsigned int mu_19_2 = _vec_load_10[3] >> 16 & 65535;
                            int j_20_2 = js[2] + 112;
                            int keep_21_2 = (int)(mu_19_2 >= tmin && j_20_2 < slen);
                            mk_2 = mk_2 | keep_21_2 << 7;
                            masks[2] = mk_2;
                            int _popc_2 = __popc(mk_2);
                            cnt = cnt + _popc_2;
                        }
                        int e_7 = tid_0 + 768;
                        int c_8 = c_lo + (e_7 >> 4);
                        int r_9 = e_7 & 15;
                        js[3] = c_8 * 128 + r_9;
                        masks[3] = 0;
                        if (e_7 < n_ent) {
                            unsigned int _vec_load_11[4];
                            {
                                uint4 _uv4_11 = *reinterpret_cast<const uint4*>(block_max + bm_base + (unsigned long long)((c_8 * 16 + r_9) * 4));
                                _vec_load_11[0 + 0] = _uv4_11.x;
                                _vec_load_11[0 + 1] = _uv4_11.y;
                                _vec_load_11[0 + 2] = _uv4_11.z;
                                _vec_load_11[0 + 3] = _uv4_11.w;
                            }
                            int mk_3 = 0;
                            unsigned int mu_5 = _vec_load_11[0] & 65535;
                            int j_0_3 = js[3];
                            int keep_10 = (int)(mu_5 >= tmin && j_0_3 < slen);
                            mk_3 = mk_3 | keep_10;
                            unsigned int mu_1_3 = _vec_load_11[0] >> 16 & 65535;
                            int j_2_3 = js[3] + 16;
                            int keep_3_3 = (int)(mu_1_3 >= tmin && j_2_3 < slen);
                            mk_3 = mk_3 | keep_3_3 << 1;
                            unsigned int mu_4_3 = _vec_load_11[1] & 65535;
                            int j_5_3 = js[3] + 32;
                            int keep_6_3 = (int)(mu_4_3 >= tmin && j_5_3 < slen);
                            mk_3 = mk_3 | keep_6_3 << 2;
                            unsigned int mu_7_3 = _vec_load_11[1] >> 16 & 65535;
                            int j_8_3 = js[3] + 48;
                            int keep_9_3 = (int)(mu_7_3 >= tmin && j_8_3 < slen);
                            mk_3 = mk_3 | keep_9_3 << 3;
                            unsigned int mu_10_3 = _vec_load_11[2] & 65535;
                            int j_11_3 = js[3] + 64;
                            int keep_12_3 = (int)(mu_10_3 >= tmin && j_11_3 < slen);
                            mk_3 = mk_3 | keep_12_3 << 4;
                            unsigned int mu_13_3 = _vec_load_11[2] >> 16 & 65535;
                            int j_14_3 = js[3] + 80;
                            int keep_15_3 = (int)(mu_13_3 >= tmin && j_14_3 < slen);
                            mk_3 = mk_3 | keep_15_3 << 5;
                            unsigned int mu_16_3 = _vec_load_11[3] & 65535;
                            int j_17_3 = js[3] + 96;
                            int keep_18_3 = (int)(mu_16_3 >= tmin && j_17_3 < slen);
                            mk_3 = mk_3 | keep_18_3 << 6;
                            unsigned int mu_19_3 = _vec_load_11[3] >> 16 & 65535;
                            int j_20_3 = js[3] + 112;
                            int keep_21_3 = (int)(mu_19_3 >= tmin && j_20_3 < slen);
                            mk_3 = mk_3 | keep_21_3 << 7;
                            masks[3] = mk_3;
                            int _popc_3 = __popc(mk_3);
                            cnt = cnt + _popc_3;
                        }
                        int incl = cnt;
                        int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, incl, 1, 32);
                        int up = _shfl_up_0;
                        incl = incl + ((lane_1 >= 1) ? up : 0);
                        int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, incl, 2, 32);
                        int up_10 = _shfl_up_1;
                        incl = incl + ((lane_1 >= 2) ? up_10 : 0);
                        int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, incl, 4, 32);
                        int up_11 = _shfl_up_2;
                        incl = incl + ((lane_1 >= 4) ? up_11 : 0);
                        int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, incl, 8, 32);
                        int up_12 = _shfl_up_3;
                        incl = incl + ((lane_1 >= 8) ? up_12 : 0);
                        int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, incl, 16, 32);
                        int up_13 = _shfl_up_4;
                        incl = incl + ((lane_1 >= 16) ? up_13 : 0);
                        if (lane_1 == 31) {
                            seg_sum[warp_2] = (unsigned int)incl;
                        }
                        __syncthreads();
                        int base = incl - cnt;
                        int total = 0;
                        int tw = (int)seg_sum[0];
                        total = total + tw;
                        base = base + ((warp_2 > 0) ? tw : 0);
                        int tw_14 = (int)seg_sum[1];
                        total = total + tw_14;
                        base = base + ((warp_2 > 1) ? tw_14 : 0);
                        int tw_15 = (int)seg_sum[2];
                        total = total + tw_15;
                        base = base + ((warp_2 > 2) ? tw_15 : 0);
                        int tw_16 = (int)seg_sum[3];
                        total = total + tw_16;
                        base = base + ((warp_2 > 3) ? tw_16 : 0);
                        int tw_17 = (int)seg_sum[4];
                        total = total + tw_17;
                        base = base + ((warp_2 > 4) ? tw_17 : 0);
                        int tw_18 = (int)seg_sum[5];
                        total = total + tw_18;
                        base = base + ((warp_2 > 5) ? tw_18 : 0);
                        int tw_19 = (int)seg_sum[6];
                        total = total + tw_19;
                        base = base + ((warp_2 > 6) ? tw_19 : 0);
                        int tw_20 = (int)seg_sum[7];
                        total = total + tw_20;
                        base = base + ((warp_2 > 7) ? tw_20 : 0);
                        int dummy = 1280 + tid_0;
                        int keep2_1 = masks[0] & 1;
                        int idx = ((keep2_1 != 0) ? 1536 + base : dummy);
                        hist[idx] = (unsigned int)(lo_s + js[0]);
                        base = base + keep2_1;
                        int keep2_21 = masks[0] >> 1 & 1;
                        int idx_22 = ((keep2_21 != 0) ? 1536 + base : dummy);
                        hist[idx_22] = (unsigned int)(lo_s + js[0] + 16);
                        base = base + keep2_21;
                        int keep2_23 = masks[0] >> 2 & 1;
                        int idx_24 = ((keep2_23 != 0) ? 1536 + base : dummy);
                        hist[idx_24] = (unsigned int)(lo_s + js[0] + 32);
                        base = base + keep2_23;
                        int keep2_25 = masks[0] >> 3 & 1;
                        int idx_26 = ((keep2_25 != 0) ? 1536 + base : dummy);
                        hist[idx_26] = (unsigned int)(lo_s + js[0] + 48);
                        base = base + keep2_25;
                        int keep2_27 = masks[0] >> 4 & 1;
                        int idx_28 = ((keep2_27 != 0) ? 1536 + base : dummy);
                        hist[idx_28] = (unsigned int)(lo_s + js[0] + 64);
                        base = base + keep2_27;
                        int keep2_29 = masks[0] >> 5 & 1;
                        int idx_30 = ((keep2_29 != 0) ? 1536 + base : dummy);
                        hist[idx_30] = (unsigned int)(lo_s + js[0] + 80);
                        base = base + keep2_29;
                        int keep2_31 = masks[0] >> 6 & 1;
                        int idx_32 = ((keep2_31 != 0) ? 1536 + base : dummy);
                        hist[idx_32] = (unsigned int)(lo_s + js[0] + 96);
                        base = base + keep2_31;
                        int keep2_33 = masks[0] >> 7 & 1;
                        int idx_34 = ((keep2_33 != 0) ? 1536 + base : dummy);
                        hist[idx_34] = (unsigned int)(lo_s + js[0] + 112);
                        base = base + keep2_33;
                        int keep2_35 = masks[1] & 1;
                        int idx_36 = ((keep2_35 != 0) ? 1536 + base : dummy);
                        hist[idx_36] = (unsigned int)(lo_s + js[1]);
                        base = base + keep2_35;
                        int keep2_37 = masks[1] >> 1 & 1;
                        int idx_38 = ((keep2_37 != 0) ? 1536 + base : dummy);
                        hist[idx_38] = (unsigned int)(lo_s + js[1] + 16);
                        base = base + keep2_37;
                        int keep2_39 = masks[1] >> 2 & 1;
                        int idx_40 = ((keep2_39 != 0) ? 1536 + base : dummy);
                        hist[idx_40] = (unsigned int)(lo_s + js[1] + 32);
                        base = base + keep2_39;
                        int keep2_41 = masks[1] >> 3 & 1;
                        int idx_42 = ((keep2_41 != 0) ? 1536 + base : dummy);
                        hist[idx_42] = (unsigned int)(lo_s + js[1] + 48);
                        base = base + keep2_41;
                        int keep2_43 = masks[1] >> 4 & 1;
                        int idx_44 = ((keep2_43 != 0) ? 1536 + base : dummy);
                        hist[idx_44] = (unsigned int)(lo_s + js[1] + 64);
                        base = base + keep2_43;
                        int keep2_45 = masks[1] >> 5 & 1;
                        int idx_46 = ((keep2_45 != 0) ? 1536 + base : dummy);
                        hist[idx_46] = (unsigned int)(lo_s + js[1] + 80);
                        base = base + keep2_45;
                        int keep2_47 = masks[1] >> 6 & 1;
                        int idx_48 = ((keep2_47 != 0) ? 1536 + base : dummy);
                        hist[idx_48] = (unsigned int)(lo_s + js[1] + 96);
                        base = base + keep2_47;
                        int keep2_49 = masks[1] >> 7 & 1;
                        int idx_50 = ((keep2_49 != 0) ? 1536 + base : dummy);
                        hist[idx_50] = (unsigned int)(lo_s + js[1] + 112);
                        base = base + keep2_49;
                        int keep2_51 = masks[2] & 1;
                        int idx_52 = ((keep2_51 != 0) ? 1536 + base : dummy);
                        hist[idx_52] = (unsigned int)(lo_s + js[2]);
                        base = base + keep2_51;
                        int keep2_53 = masks[2] >> 1 & 1;
                        int idx_54 = ((keep2_53 != 0) ? 1536 + base : dummy);
                        hist[idx_54] = (unsigned int)(lo_s + js[2] + 16);
                        base = base + keep2_53;
                        int keep2_55 = masks[2] >> 2 & 1;
                        int idx_56 = ((keep2_55 != 0) ? 1536 + base : dummy);
                        hist[idx_56] = (unsigned int)(lo_s + js[2] + 32);
                        base = base + keep2_55;
                        int keep2_57 = masks[2] >> 3 & 1;
                        int idx_58 = ((keep2_57 != 0) ? 1536 + base : dummy);
                        hist[idx_58] = (unsigned int)(lo_s + js[2] + 48);
                        base = base + keep2_57;
                        int keep2_59 = masks[2] >> 4 & 1;
                        int idx_60 = ((keep2_59 != 0) ? 1536 + base : dummy);
                        hist[idx_60] = (unsigned int)(lo_s + js[2] + 64);
                        base = base + keep2_59;
                        int keep2_61 = masks[2] >> 5 & 1;
                        int idx_62 = ((keep2_61 != 0) ? 1536 + base : dummy);
                        hist[idx_62] = (unsigned int)(lo_s + js[2] + 80);
                        base = base + keep2_61;
                        int keep2_63 = masks[2] >> 6 & 1;
                        int idx_64 = ((keep2_63 != 0) ? 1536 + base : dummy);
                        hist[idx_64] = (unsigned int)(lo_s + js[2] + 96);
                        base = base + keep2_63;
                        int keep2_65 = masks[2] >> 7 & 1;
                        int idx_66 = ((keep2_65 != 0) ? 1536 + base : dummy);
                        hist[idx_66] = (unsigned int)(lo_s + js[2] + 112);
                        base = base + keep2_65;
                        int keep2_67 = masks[3] & 1;
                        int idx_68 = ((keep2_67 != 0) ? 1536 + base : dummy);
                        hist[idx_68] = (unsigned int)(lo_s + js[3]);
                        base = base + keep2_67;
                        int keep2_69 = masks[3] >> 1 & 1;
                        int idx_70 = ((keep2_69 != 0) ? 1536 + base : dummy);
                        hist[idx_70] = (unsigned int)(lo_s + js[3] + 16);
                        base = base + keep2_69;
                        int keep2_71 = masks[3] >> 2 & 1;
                        int idx_72 = ((keep2_71 != 0) ? 1536 + base : dummy);
                        hist[idx_72] = (unsigned int)(lo_s + js[3] + 32);
                        base = base + keep2_71;
                        int keep2_73 = masks[3] >> 3 & 1;
                        int idx_74 = ((keep2_73 != 0) ? 1536 + base : dummy);
                        hist[idx_74] = (unsigned int)(lo_s + js[3] + 48);
                        base = base + keep2_73;
                        int keep2_75 = masks[3] >> 4 & 1;
                        int idx_76 = ((keep2_75 != 0) ? 1536 + base : dummy);
                        hist[idx_76] = (unsigned int)(lo_s + js[3] + 64);
                        base = base + keep2_75;
                        int keep2_77 = masks[3] >> 5 & 1;
                        int idx_78 = ((keep2_77 != 0) ? 1536 + base : dummy);
                        hist[idx_78] = (unsigned int)(lo_s + js[3] + 80);
                        base = base + keep2_77;
                        int keep2_79 = masks[3] >> 6 & 1;
                        int idx_80 = ((keep2_79 != 0) ? 1536 + base : dummy);
                        hist[idx_80] = (unsigned int)(lo_s + js[3] + 96);
                        base = base + keep2_79;
                        int keep2_81 = masks[3] >> 7 & 1;
                        int idx_82 = ((keep2_81 != 0) ? 1536 + base : dummy);
                        hist[idx_82] = (unsigned int)(lo_s + js[3] + 112);
                        base = base + keep2_81;
                        nres[0] = total;
                        __syncthreads();
                        int n_list = nres[0];
                        int n_it = n_list + 15 >> 4;
                        #pragma unroll 1
                        for (int i0_1 = 0; i0_1 < n_it; i0_1 += 8) {
                            int _max_162 = ((n_list - 1) > (0) ? (n_list - 1) : (0));
                            int last = _max_162;
                            int e_0 = i0_1 * 16 + r_sub;
                            int act_2 = (int)(e_0 < n_list) & qvalid;
                            int _min_26 = ((e_0) < (last) ? (e_0) : (last));
                            int blk_1 = (int)hist[1536 + _min_26];
                            blks[0] = blk_1;
                            unsigned long long off_1 = (row_base64 + (unsigned long long)blk_1) * total_q64 + q_addr64;
                            xs[0] = __uint_as_float(2143289344);
                            xs[1] = __uint_as_float(2143289344);
                            if (act_2 != 0) {
                                float _vec_load_12[2];
                                {
                                    float2 _v2_12 = *reinterpret_cast<const float2*>(max_score + off_1);
                                    _vec_load_12[0] = _v2_12.x;
                                    _vec_load_12[0 + 1] = _v2_12.y;
                                }
                                xs[0] = _vec_load_12[0];
                                xs[1] = _vec_load_12[1];
                            }
                            int e_2 = (i0_1 + 1) * 16 + r_sub;
                            int act_3_1 = (int)(e_2 < n_list) & qvalid;
                            int _min_27 = ((e_2) < (last) ? (e_2) : (last));
                            int blk_4_1 = (int)hist[1536 + _min_27];
                            blks[1] = blk_4_1;
                            unsigned long long off_5 = (row_base64 + (unsigned long long)blk_4_1) * total_q64 + q_addr64;
                            xs[2] = __uint_as_float(2143289344);
                            xs[3] = __uint_as_float(2143289344);
                            if (act_3_1 != 0) {
                                float _vec_load_13[2];
                                {
                                    float2 _v2_13 = *reinterpret_cast<const float2*>(max_score + off_5);
                                    _vec_load_13[0] = _v2_13.x;
                                    _vec_load_13[0 + 1] = _v2_13.y;
                                }
                                xs[2] = _vec_load_13[0];
                                xs[3] = _vec_load_13[1];
                            }
                            int e_6 = (i0_1 + 2) * 16 + r_sub;
                            int act_7_1 = (int)(e_6 < n_list) & qvalid;
                            int _min_28 = ((e_6) < (last) ? (e_6) : (last));
                            int blk_8_1 = (int)hist[1536 + _min_28];
                            blks[2] = blk_8_1;
                            unsigned long long off_9 = (row_base64 + (unsigned long long)blk_8_1) * total_q64 + q_addr64;
                            xs[4] = __uint_as_float(2143289344);
                            xs[5] = __uint_as_float(2143289344);
                            if (act_7_1 != 0) {
                                float _vec_load_14[2];
                                {
                                    float2 _v2_14 = *reinterpret_cast<const float2*>(max_score + off_9);
                                    _vec_load_14[0] = _v2_14.x;
                                    _vec_load_14[0 + 1] = _v2_14.y;
                                }
                                xs[4] = _vec_load_14[0];
                                xs[5] = _vec_load_14[1];
                            }
                            int e_10 = (i0_1 + 3) * 16 + r_sub;
                            int act_11_1 = (int)(e_10 < n_list) & qvalid;
                            int _min_29 = ((e_10) < (last) ? (e_10) : (last));
                            int blk_12_1 = (int)hist[1536 + _min_29];
                            blks[3] = blk_12_1;
                            unsigned long long off_13 = (row_base64 + (unsigned long long)blk_12_1) * total_q64 + q_addr64;
                            xs[6] = __uint_as_float(2143289344);
                            xs[7] = __uint_as_float(2143289344);
                            if (act_11_1 != 0) {
                                float _vec_load_15[2];
                                {
                                    float2 _v2_15 = *reinterpret_cast<const float2*>(max_score + off_13);
                                    _vec_load_15[0] = _v2_15.x;
                                    _vec_load_15[0 + 1] = _v2_15.y;
                                }
                                xs[6] = _vec_load_15[0];
                                xs[7] = _vec_load_15[1];
                            }
                            int e_14 = (i0_1 + 4) * 16 + r_sub;
                            int act_15 = (int)(e_14 < n_list) & qvalid;
                            int _min_30 = ((e_14) < (last) ? (e_14) : (last));
                            int blk_16 = (int)hist[1536 + _min_30];
                            blks[4] = blk_16;
                            unsigned long long off_17 = (row_base64 + (unsigned long long)blk_16) * total_q64 + q_addr64;
                            xs[8] = __uint_as_float(2143289344);
                            xs[9] = __uint_as_float(2143289344);
                            if (act_15 != 0) {
                                float _vec_load_16[2];
                                {
                                    float2 _v2_16 = *reinterpret_cast<const float2*>(max_score + off_17);
                                    _vec_load_16[0] = _v2_16.x;
                                    _vec_load_16[0 + 1] = _v2_16.y;
                                }
                                xs[8] = _vec_load_16[0];
                                xs[9] = _vec_load_16[1];
                            }
                            int e_18 = (i0_1 + 5) * 16 + r_sub;
                            int act_19 = (int)(e_18 < n_list) & qvalid;
                            int _min_31 = ((e_18) < (last) ? (e_18) : (last));
                            int blk_20 = (int)hist[1536 + _min_31];
                            blks[5] = blk_20;
                            unsigned long long off_21 = (row_base64 + (unsigned long long)blk_20) * total_q64 + q_addr64;
                            xs[10] = __uint_as_float(2143289344);
                            xs[11] = __uint_as_float(2143289344);
                            if (act_19 != 0) {
                                float _vec_load_17[2];
                                {
                                    float2 _v2_17 = *reinterpret_cast<const float2*>(max_score + off_21);
                                    _vec_load_17[0] = _v2_17.x;
                                    _vec_load_17[0 + 1] = _v2_17.y;
                                }
                                xs[10] = _vec_load_17[0];
                                xs[11] = _vec_load_17[1];
                            }
                            int e_22 = (i0_1 + 6) * 16 + r_sub;
                            int act_23 = (int)(e_22 < n_list) & qvalid;
                            int _min_32 = ((e_22) < (last) ? (e_22) : (last));
                            int blk_24 = (int)hist[1536 + _min_32];
                            blks[6] = blk_24;
                            unsigned long long off_25 = (row_base64 + (unsigned long long)blk_24) * total_q64 + q_addr64;
                            xs[12] = __uint_as_float(2143289344);
                            xs[13] = __uint_as_float(2143289344);
                            if (act_23 != 0) {
                                float _vec_load_18[2];
                                {
                                    float2 _v2_18 = *reinterpret_cast<const float2*>(max_score + off_25);
                                    _vec_load_18[0] = _v2_18.x;
                                    _vec_load_18[0 + 1] = _v2_18.y;
                                }
                                xs[12] = _vec_load_18[0];
                                xs[13] = _vec_load_18[1];
                            }
                            int e_26 = (i0_1 + 7) * 16 + r_sub;
                            int act_27 = (int)(e_26 < n_list) & qvalid;
                            int _min_33 = ((e_26) < (last) ? (e_26) : (last));
                            int blk_28 = (int)hist[1536 + _min_33];
                            blks[7] = blk_28;
                            unsigned long long off_29 = (row_base64 + (unsigned long long)blk_28) * total_q64 + q_addr64;
                            xs[14] = __uint_as_float(2143289344);
                            xs[15] = __uint_as_float(2143289344);
                            if (act_27 != 0) {
                                float _vec_load_19[2];
                                {
                                    float2 _v2_19 = *reinterpret_cast<const float2*>(max_score + off_29);
                                    _vec_load_19[0] = _v2_19.x;
                                    _vec_load_19[0 + 1] = _v2_19.y;
                                }
                                xs[14] = _vec_load_19[0];
                                xs[15] = _vec_load_19[1];
                            }
                            unsigned int raw_1 = __as_u32(xs[0]);
                            unsigned int mag_1 = raw_1 & 2147483647;
                            unsigned int nan_1 = (unsigned int)(mag_1 > 2139095040);
                            unsigned int bits_1 = ((mag_1 != 0) ? raw_1 : 0);
                            unsigned int neg_mask = -(bits_1 >> 31);
                            unsigned int key = bits_1 ^ (neg_mask | 2147483648u);
                            unsigned int inv = ~(unsigned int)blks[0];
                            unsigned long long comp = (unsigned long long)key << 32 | (unsigned long long)inv;
                            int qw = q_local * 2;
                            if (is_stage_v[0] != 0 && nan_1 == 0 && comp >= cthr_v[0]) {
                                unsigned int _atomic_old_0 = atomicAdd(&cand_cnt[qw], 1);
                                unsigned int slot_w = _atomic_old_0;
                                if (slot_w < 64) {
                                    cand[qw * 64 + (int)slot_w] = comp;
                                }
                            }
                            unsigned int raw_30 = __as_u32(xs[1]);
                            unsigned int mag_31 = raw_30 & 2147483647;
                            unsigned int nan_32 = (unsigned int)(mag_31 > 2139095040);
                            unsigned int bits_33 = ((mag_31 != 0) ? raw_30 : 0);
                            unsigned int neg_mask_34 = -(bits_33 >> 31);
                            unsigned int key_35 = bits_33 ^ (neg_mask_34 | 2147483648u);
                            unsigned int inv_36 = ~(unsigned int)blks[0];
                            unsigned long long comp_37 = (unsigned long long)key_35 << 32 | (unsigned long long)inv_36;
                            int qw_38 = q_local * 2 + 1;
                            if (is_stage_v[1] != 0 && nan_32 == 0 && comp_37 >= cthr_v[1]) {
                                unsigned int _atomic_old_1 = atomicAdd(&cand_cnt[qw_38], 1);
                                unsigned int slot_w_1 = _atomic_old_1;
                                if (slot_w_1 < 64) {
                                    cand[qw_38 * 64 + (int)slot_w_1] = comp_37;
                                }
                            }
                            unsigned int raw_39 = __as_u32(xs[2]);
                            unsigned int mag_40 = raw_39 & 2147483647;
                            unsigned int nan_41 = (unsigned int)(mag_40 > 2139095040);
                            unsigned int bits_42 = ((mag_40 != 0) ? raw_39 : 0);
                            unsigned int neg_mask_43 = -(bits_42 >> 31);
                            unsigned int key_44 = bits_42 ^ (neg_mask_43 | 2147483648u);
                            unsigned int inv_45 = ~(unsigned int)blks[1];
                            unsigned long long comp_46 = (unsigned long long)key_44 << 32 | (unsigned long long)inv_45;
                            int qw_47 = q_local * 2;
                            if (is_stage_v[0] != 0 && nan_41 == 0 && comp_46 >= cthr_v[0]) {
                                unsigned int _atomic_old_2 = atomicAdd(&cand_cnt[qw_47], 1);
                                unsigned int slot_w_2 = _atomic_old_2;
                                if (slot_w_2 < 64) {
                                    cand[qw_47 * 64 + (int)slot_w_2] = comp_46;
                                }
                            }
                            unsigned int raw_48 = __as_u32(xs[3]);
                            unsigned int mag_49 = raw_48 & 2147483647;
                            unsigned int nan_50 = (unsigned int)(mag_49 > 2139095040);
                            unsigned int bits_51 = ((mag_49 != 0) ? raw_48 : 0);
                            unsigned int neg_mask_52 = -(bits_51 >> 31);
                            unsigned int key_53 = bits_51 ^ (neg_mask_52 | 2147483648u);
                            unsigned int inv_54 = ~(unsigned int)blks[1];
                            unsigned long long comp_55 = (unsigned long long)key_53 << 32 | (unsigned long long)inv_54;
                            int qw_56 = q_local * 2 + 1;
                            if (is_stage_v[1] != 0 && nan_50 == 0 && comp_55 >= cthr_v[1]) {
                                unsigned int _atomic_old_3 = atomicAdd(&cand_cnt[qw_56], 1);
                                unsigned int slot_w_3 = _atomic_old_3;
                                if (slot_w_3 < 64) {
                                    cand[qw_56 * 64 + (int)slot_w_3] = comp_55;
                                }
                            }
                            unsigned int raw_57_1 = __as_u32(xs[4]);
                            unsigned int mag_58_1 = raw_57_1 & 2147483647;
                            unsigned int nan_59_1 = (unsigned int)(mag_58_1 > 2139095040);
                            unsigned int bits_60_1 = ((mag_58_1 != 0) ? raw_57_1 : 0);
                            unsigned int neg_mask_61 = -(bits_60_1 >> 31);
                            unsigned int key_62 = bits_60_1 ^ (neg_mask_61 | 2147483648u);
                            unsigned int inv_63 = ~(unsigned int)blks[2];
                            unsigned long long comp_64 = (unsigned long long)key_62 << 32 | (unsigned long long)inv_63;
                            int qw_65 = q_local * 2;
                            if (is_stage_v[0] != 0 && nan_59_1 == 0 && comp_64 >= cthr_v[0]) {
                                unsigned int _atomic_old_4 = atomicAdd(&cand_cnt[qw_65], 1);
                                unsigned int slot_w_4 = _atomic_old_4;
                                if (slot_w_4 < 64) {
                                    cand[qw_65 * 64 + (int)slot_w_4] = comp_64;
                                }
                            }
                            unsigned int raw_66 = __as_u32(xs[5]);
                            unsigned int mag_67 = raw_66 & 2147483647;
                            unsigned int nan_68 = (unsigned int)(mag_67 > 2139095040);
                            unsigned int bits_69 = ((mag_67 != 0) ? raw_66 : 0);
                            unsigned int neg_mask_70 = -(bits_69 >> 31);
                            unsigned int key_71 = bits_69 ^ (neg_mask_70 | 2147483648u);
                            unsigned int inv_72 = ~(unsigned int)blks[2];
                            unsigned long long comp_73 = (unsigned long long)key_71 << 32 | (unsigned long long)inv_72;
                            int qw_74 = q_local * 2 + 1;
                            if (is_stage_v[1] != 0 && nan_68 == 0 && comp_73 >= cthr_v[1]) {
                                unsigned int _atomic_old_5 = atomicAdd(&cand_cnt[qw_74], 1);
                                unsigned int slot_w_5 = _atomic_old_5;
                                if (slot_w_5 < 64) {
                                    cand[qw_74 * 64 + (int)slot_w_5] = comp_73;
                                }
                            }
                            unsigned int raw_75 = __as_u32(xs[6]);
                            unsigned int mag_76 = raw_75 & 2147483647;
                            unsigned int nan_77 = (unsigned int)(mag_76 > 2139095040);
                            unsigned int bits_78 = ((mag_76 != 0) ? raw_75 : 0);
                            unsigned int neg_mask_79 = -(bits_78 >> 31);
                            unsigned int key_80 = bits_78 ^ (neg_mask_79 | 2147483648u);
                            unsigned int inv_81 = ~(unsigned int)blks[3];
                            unsigned long long comp_82 = (unsigned long long)key_80 << 32 | (unsigned long long)inv_81;
                            int qw_83 = q_local * 2;
                            if (is_stage_v[0] != 0 && nan_77 == 0 && comp_82 >= cthr_v[0]) {
                                unsigned int _atomic_old_6 = atomicAdd(&cand_cnt[qw_83], 1);
                                unsigned int slot_w_6 = _atomic_old_6;
                                if (slot_w_6 < 64) {
                                    cand[qw_83 * 64 + (int)slot_w_6] = comp_82;
                                }
                            }
                            unsigned int raw_84 = __as_u32(xs[7]);
                            unsigned int mag_85 = raw_84 & 2147483647;
                            unsigned int nan_86 = (unsigned int)(mag_85 > 2139095040);
                            unsigned int bits_87 = ((mag_85 != 0) ? raw_84 : 0);
                            unsigned int neg_mask_88 = -(bits_87 >> 31);
                            unsigned int key_89 = bits_87 ^ (neg_mask_88 | 2147483648u);
                            unsigned int inv_90 = ~(unsigned int)blks[3];
                            unsigned long long comp_91 = (unsigned long long)key_89 << 32 | (unsigned long long)inv_90;
                            int qw_92 = q_local * 2 + 1;
                            if (is_stage_v[1] != 0 && nan_86 == 0 && comp_91 >= cthr_v[1]) {
                                unsigned int _atomic_old_7 = atomicAdd(&cand_cnt[qw_92], 1);
                                unsigned int slot_w_7 = _atomic_old_7;
                                if (slot_w_7 < 64) {
                                    cand[qw_92 * 64 + (int)slot_w_7] = comp_91;
                                }
                            }
                            unsigned int raw_93 = __as_u32(xs[8]);
                            unsigned int mag_94 = raw_93 & 2147483647;
                            unsigned int nan_95 = (unsigned int)(mag_94 > 2139095040);
                            unsigned int bits_96 = ((mag_94 != 0) ? raw_93 : 0);
                            unsigned int neg_mask_97 = -(bits_96 >> 31);
                            unsigned int key_98 = bits_96 ^ (neg_mask_97 | 2147483648u);
                            unsigned int inv_99 = ~(unsigned int)blks[4];
                            unsigned long long comp_100 = (unsigned long long)key_98 << 32 | (unsigned long long)inv_99;
                            int qw_101 = q_local * 2;
                            if (is_stage_v[0] != 0 && nan_95 == 0 && comp_100 >= cthr_v[0]) {
                                unsigned int _atomic_old_8 = atomicAdd(&cand_cnt[qw_101], 1);
                                unsigned int slot_w_8 = _atomic_old_8;
                                if (slot_w_8 < 64) {
                                    cand[qw_101 * 64 + (int)slot_w_8] = comp_100;
                                }
                            }
                            unsigned int raw_102 = __as_u32(xs[9]);
                            unsigned int mag_103 = raw_102 & 2147483647;
                            unsigned int nan_104 = (unsigned int)(mag_103 > 2139095040);
                            unsigned int bits_105 = ((mag_103 != 0) ? raw_102 : 0);
                            unsigned int neg_mask_106 = -(bits_105 >> 31);
                            unsigned int key_107 = bits_105 ^ (neg_mask_106 | 2147483648u);
                            unsigned int inv_108 = ~(unsigned int)blks[4];
                            unsigned long long comp_109 = (unsigned long long)key_107 << 32 | (unsigned long long)inv_108;
                            int qw_110 = q_local * 2 + 1;
                            if (is_stage_v[1] != 0 && nan_104 == 0 && comp_109 >= cthr_v[1]) {
                                unsigned int _atomic_old_9 = atomicAdd(&cand_cnt[qw_110], 1);
                                unsigned int slot_w_9 = _atomic_old_9;
                                if (slot_w_9 < 64) {
                                    cand[qw_110 * 64 + (int)slot_w_9] = comp_109;
                                }
                            }
                            unsigned int raw_111 = __as_u32(xs[10]);
                            unsigned int mag_112 = raw_111 & 2147483647;
                            unsigned int nan_113 = (unsigned int)(mag_112 > 2139095040);
                            unsigned int bits_114 = ((mag_112 != 0) ? raw_111 : 0);
                            unsigned int neg_mask_115 = -(bits_114 >> 31);
                            unsigned int key_116 = bits_114 ^ (neg_mask_115 | 2147483648u);
                            unsigned int inv_117 = ~(unsigned int)blks[5];
                            unsigned long long comp_118 = (unsigned long long)key_116 << 32 | (unsigned long long)inv_117;
                            int qw_119 = q_local * 2;
                            if (is_stage_v[0] != 0 && nan_113 == 0 && comp_118 >= cthr_v[0]) {
                                unsigned int _atomic_old_10 = atomicAdd(&cand_cnt[qw_119], 1);
                                unsigned int slot_w_10 = _atomic_old_10;
                                if (slot_w_10 < 64) {
                                    cand[qw_119 * 64 + (int)slot_w_10] = comp_118;
                                }
                            }
                            unsigned int raw_120_1 = __as_u32(xs[11]);
                            unsigned int mag_121_1 = raw_120_1 & 2147483647;
                            unsigned int nan_122_1 = (unsigned int)(mag_121_1 > 2139095040);
                            unsigned int bits_123_1 = ((mag_121_1 != 0) ? raw_120_1 : 0);
                            unsigned int neg_mask_124 = -(bits_123_1 >> 31);
                            unsigned int key_125 = bits_123_1 ^ (neg_mask_124 | 2147483648u);
                            unsigned int inv_126 = ~(unsigned int)blks[5];
                            unsigned long long comp_127 = (unsigned long long)key_125 << 32 | (unsigned long long)inv_126;
                            int qw_128 = q_local * 2 + 1;
                            if (is_stage_v[1] != 0 && nan_122_1 == 0 && comp_127 >= cthr_v[1]) {
                                unsigned int _atomic_old_11 = atomicAdd(&cand_cnt[qw_128], 1);
                                unsigned int slot_w_11 = _atomic_old_11;
                                if (slot_w_11 < 64) {
                                    cand[qw_128 * 64 + (int)slot_w_11] = comp_127;
                                }
                            }
                            unsigned int raw_129 = __as_u32(xs[12]);
                            unsigned int mag_130 = raw_129 & 2147483647;
                            unsigned int nan_131 = (unsigned int)(mag_130 > 2139095040);
                            unsigned int bits_132 = ((mag_130 != 0) ? raw_129 : 0);
                            unsigned int neg_mask_133 = -(bits_132 >> 31);
                            unsigned int key_134 = bits_132 ^ (neg_mask_133 | 2147483648u);
                            unsigned int inv_135 = ~(unsigned int)blks[6];
                            unsigned long long comp_136 = (unsigned long long)key_134 << 32 | (unsigned long long)inv_135;
                            int qw_137 = q_local * 2;
                            if (is_stage_v[0] != 0 && nan_131 == 0 && comp_136 >= cthr_v[0]) {
                                unsigned int _atomic_old_12 = atomicAdd(&cand_cnt[qw_137], 1);
                                unsigned int slot_w_12 = _atomic_old_12;
                                if (slot_w_12 < 64) {
                                    cand[qw_137 * 64 + (int)slot_w_12] = comp_136;
                                }
                            }
                            unsigned int raw_138 = __as_u32(xs[13]);
                            unsigned int mag_139 = raw_138 & 2147483647;
                            unsigned int nan_140 = (unsigned int)(mag_139 > 2139095040);
                            unsigned int bits_141 = ((mag_139 != 0) ? raw_138 : 0);
                            unsigned int neg_mask_142 = -(bits_141 >> 31);
                            unsigned int key_143 = bits_141 ^ (neg_mask_142 | 2147483648u);
                            unsigned int inv_144 = ~(unsigned int)blks[6];
                            unsigned long long comp_145 = (unsigned long long)key_143 << 32 | (unsigned long long)inv_144;
                            int qw_146 = q_local * 2 + 1;
                            if (is_stage_v[1] != 0 && nan_140 == 0 && comp_145 >= cthr_v[1]) {
                                unsigned int _atomic_old_13 = atomicAdd(&cand_cnt[qw_146], 1);
                                unsigned int slot_w_13 = _atomic_old_13;
                                if (slot_w_13 < 64) {
                                    cand[qw_146 * 64 + (int)slot_w_13] = comp_145;
                                }
                            }
                            unsigned int raw_147 = __as_u32(xs[14]);
                            unsigned int mag_148 = raw_147 & 2147483647;
                            unsigned int nan_149 = (unsigned int)(mag_148 > 2139095040);
                            unsigned int bits_150 = ((mag_148 != 0) ? raw_147 : 0);
                            unsigned int neg_mask_151 = -(bits_150 >> 31);
                            unsigned int key_152 = bits_150 ^ (neg_mask_151 | 2147483648u);
                            unsigned int inv_153 = ~(unsigned int)blks[7];
                            unsigned long long comp_154 = (unsigned long long)key_152 << 32 | (unsigned long long)inv_153;
                            int qw_155 = q_local * 2;
                            if (is_stage_v[0] != 0 && nan_149 == 0 && comp_154 >= cthr_v[0]) {
                                unsigned int _atomic_old_14 = atomicAdd(&cand_cnt[qw_155], 1);
                                unsigned int slot_w_14 = _atomic_old_14;
                                if (slot_w_14 < 64) {
                                    cand[qw_155 * 64 + (int)slot_w_14] = comp_154;
                                }
                            }
                            unsigned int raw_156 = __as_u32(xs[15]);
                            unsigned int mag_157 = raw_156 & 2147483647;
                            unsigned int nan_158 = (unsigned int)(mag_157 > 2139095040);
                            unsigned int bits_159 = ((mag_157 != 0) ? raw_156 : 0);
                            unsigned int neg_mask_160 = -(bits_159 >> 31);
                            unsigned int key_161 = bits_159 ^ (neg_mask_160 | 2147483648u);
                            unsigned int inv_162 = ~(unsigned int)blks[7];
                            unsigned long long comp_163 = (unsigned long long)key_161 << 32 | (unsigned long long)inv_162;
                            int qw_164 = q_local * 2 + 1;
                            if (is_stage_v[1] != 0 && nan_158 == 0 && comp_163 >= cthr_v[1]) {
                                unsigned int _atomic_old_15 = atomicAdd(&cand_cnt[qw_164], 1);
                                unsigned int slot_w_15 = _atomic_old_15;
                                if (slot_w_15 < 64) {
                                    cand[qw_164 * 64 + (int)slot_w_15] = comp_163;
                                }
                            }
                        }
                    }
                } else {
                    #pragma unroll 1
                    for (int i0_2 = 0; i0_2 < iters; i0_2 += 8) {
                        unsigned long long off_2 = off0 + (unsigned long long)i0_2 * stride16;
                        int blk_3 = lo_s + i0_2 * 16 + r_sub;
                        blks[0] = blk_3;
                        int act_4 = (int)(blk_3 < hi_s) & qvalid;
                        xs[0] = __uint_as_float(2143289344);
                        xs[1] = __uint_as_float(2143289344);
                        if (act_4 != 0) {
                            float _vec_load_20[2];
                            {
                                float2 _v2_20 = *reinterpret_cast<const float2*>(max_score + off_2);
                                _vec_load_20[0] = _v2_20.x;
                                _vec_load_20[0 + 1] = _v2_20.y;
                            }
                            xs[0] = _vec_load_20[0];
                            xs[1] = _vec_load_20[1];
                        }
                        off_2 = off_2 + stride16;
                        int blk_0_1 = lo_s + (i0_2 + 1) * 16 + r_sub;
                        blks[1] = blk_0_1;
                        int act_1_1 = (int)(blk_0_1 < hi_s) & qvalid;
                        xs[2] = __uint_as_float(2143289344);
                        xs[3] = __uint_as_float(2143289344);
                        if (act_1_1 != 0) {
                            float _vec_load_21[2];
                            {
                                float2 _v2_21 = *reinterpret_cast<const float2*>(max_score + off_2);
                                _vec_load_21[0] = _v2_21.x;
                                _vec_load_21[0 + 1] = _v2_21.y;
                            }
                            xs[2] = _vec_load_21[0];
                            xs[3] = _vec_load_21[1];
                        }
                        off_2 = off_2 + stride16;
                        int blk_2_1 = lo_s + (i0_2 + 2) * 16 + r_sub;
                        blks[2] = blk_2_1;
                        int act_3_2 = (int)(blk_2_1 < hi_s) & qvalid;
                        xs[4] = __uint_as_float(2143289344);
                        xs[5] = __uint_as_float(2143289344);
                        if (act_3_2 != 0) {
                            float _vec_load_22[2];
                            {
                                float2 _v2_22 = *reinterpret_cast<const float2*>(max_score + off_2);
                                _vec_load_22[0] = _v2_22.x;
                                _vec_load_22[0 + 1] = _v2_22.y;
                            }
                            xs[4] = _vec_load_22[0];
                            xs[5] = _vec_load_22[1];
                        }
                        off_2 = off_2 + stride16;
                        int blk_4_2 = lo_s + (i0_2 + 3) * 16 + r_sub;
                        blks[3] = blk_4_2;
                        int act_5_1 = (int)(blk_4_2 < hi_s) & qvalid;
                        xs[6] = __uint_as_float(2143289344);
                        xs[7] = __uint_as_float(2143289344);
                        if (act_5_1 != 0) {
                            float _vec_load_23[2];
                            {
                                float2 _v2_23 = *reinterpret_cast<const float2*>(max_score + off_2);
                                _vec_load_23[0] = _v2_23.x;
                                _vec_load_23[0 + 1] = _v2_23.y;
                            }
                            xs[6] = _vec_load_23[0];
                            xs[7] = _vec_load_23[1];
                        }
                        off_2 = off_2 + stride16;
                        int blk_6_1 = lo_s + (i0_2 + 4) * 16 + r_sub;
                        blks[4] = blk_6_1;
                        int act_7_2 = (int)(blk_6_1 < hi_s) & qvalid;
                        xs[8] = __uint_as_float(2143289344);
                        xs[9] = __uint_as_float(2143289344);
                        if (act_7_2 != 0) {
                            float _vec_load_24[2];
                            {
                                float2 _v2_24 = *reinterpret_cast<const float2*>(max_score + off_2);
                                _vec_load_24[0] = _v2_24.x;
                                _vec_load_24[0 + 1] = _v2_24.y;
                            }
                            xs[8] = _vec_load_24[0];
                            xs[9] = _vec_load_24[1];
                        }
                        off_2 = off_2 + stride16;
                        int blk_8_2 = lo_s + (i0_2 + 5) * 16 + r_sub;
                        blks[5] = blk_8_2;
                        int act_9_1 = (int)(blk_8_2 < hi_s) & qvalid;
                        xs[10] = __uint_as_float(2143289344);
                        xs[11] = __uint_as_float(2143289344);
                        if (act_9_1 != 0) {
                            float _vec_load_25[2];
                            {
                                float2 _v2_25 = *reinterpret_cast<const float2*>(max_score + off_2);
                                _vec_load_25[0] = _v2_25.x;
                                _vec_load_25[0 + 1] = _v2_25.y;
                            }
                            xs[10] = _vec_load_25[0];
                            xs[11] = _vec_load_25[1];
                        }
                        off_2 = off_2 + stride16;
                        int blk_10_1 = lo_s + (i0_2 + 6) * 16 + r_sub;
                        blks[6] = blk_10_1;
                        int act_11_2 = (int)(blk_10_1 < hi_s) & qvalid;
                        xs[12] = __uint_as_float(2143289344);
                        xs[13] = __uint_as_float(2143289344);
                        if (act_11_2 != 0) {
                            float _vec_load_26[2];
                            {
                                float2 _v2_26 = *reinterpret_cast<const float2*>(max_score + off_2);
                                _vec_load_26[0] = _v2_26.x;
                                _vec_load_26[0 + 1] = _v2_26.y;
                            }
                            xs[12] = _vec_load_26[0];
                            xs[13] = _vec_load_26[1];
                        }
                        off_2 = off_2 + stride16;
                        int blk_12_2 = lo_s + (i0_2 + 7) * 16 + r_sub;
                        blks[7] = blk_12_2;
                        int act_13_1 = (int)(blk_12_2 < hi_s) & qvalid;
                        xs[14] = __uint_as_float(2143289344);
                        xs[15] = __uint_as_float(2143289344);
                        if (act_13_1 != 0) {
                            float _vec_load_27[2];
                            {
                                float2 _v2_27 = *reinterpret_cast<const float2*>(max_score + off_2);
                                _vec_load_27[0] = _v2_27.x;
                                _vec_load_27[0 + 1] = _v2_27.y;
                            }
                            xs[14] = _vec_load_27[0];
                            xs[15] = _vec_load_27[1];
                        }
                        off_2 = off_2 + stride16;
                        unsigned int raw_2 = __as_u32(xs[0]);
                        unsigned int mag_2 = raw_2 & 2147483647;
                        unsigned int nan_2 = (unsigned int)(mag_2 > 2139095040);
                        unsigned int bits_2 = ((mag_2 != 0) ? raw_2 : 0);
                        unsigned int neg_mask_1 = -(bits_2 >> 31);
                        unsigned int key_1 = bits_2 ^ (neg_mask_1 | 2147483648u);
                        unsigned int inv_1 = ~(unsigned int)blks[0];
                        unsigned long long comp_1 = (unsigned long long)key_1 << 32 | (unsigned long long)inv_1;
                        int qw_1 = q_local * 2;
                        if (is_stage_v[0] != 0 && nan_2 == 0 && comp_1 >= cthr_v[0]) {
                            unsigned int _atomic_old_16 = atomicAdd(&cand_cnt[qw_1], 1);
                            unsigned int slot_w_16 = _atomic_old_16;
                            if (slot_w_16 < 64) {
                                cand[qw_1 * 64 + (int)slot_w_16] = comp_1;
                            }
                        }
                        unsigned int raw_14_1 = __as_u32(xs[1]);
                        unsigned int mag_15 = raw_14_1 & 2147483647;
                        unsigned int nan_16 = (unsigned int)(mag_15 > 2139095040);
                        unsigned int bits_17 = ((mag_15 != 0) ? raw_14_1 : 0);
                        unsigned int neg_mask_18 = -(bits_17 >> 31);
                        unsigned int key_19 = bits_17 ^ (neg_mask_18 | 2147483648u);
                        unsigned int inv_20 = ~(unsigned int)blks[0];
                        unsigned long long comp_21 = (unsigned long long)key_19 << 32 | (unsigned long long)inv_20;
                        int qw_22 = q_local * 2 + 1;
                        if (is_stage_v[1] != 0 && nan_16 == 0 && comp_21 >= cthr_v[1]) {
                            unsigned int _atomic_old_17 = atomicAdd(&cand_cnt[qw_22], 1);
                            unsigned int slot_w_17 = _atomic_old_17;
                            if (slot_w_17 < 64) {
                                cand[qw_22 * 64 + (int)slot_w_17] = comp_21;
                            }
                        }
                        unsigned int raw_23 = __as_u32(xs[2]);
                        unsigned int mag_24 = raw_23 & 2147483647;
                        unsigned int nan_25 = (unsigned int)(mag_24 > 2139095040);
                        unsigned int bits_26 = ((mag_24 != 0) ? raw_23 : 0);
                        unsigned int neg_mask_27 = -(bits_26 >> 31);
                        unsigned int key_28 = bits_26 ^ (neg_mask_27 | 2147483648u);
                        unsigned int inv_29 = ~(unsigned int)blks[1];
                        unsigned long long comp_30 = (unsigned long long)key_28 << 32 | (unsigned long long)inv_29;
                        int qw_31 = q_local * 2;
                        if (is_stage_v[0] != 0 && nan_25 == 0 && comp_30 >= cthr_v[0]) {
                            unsigned int _atomic_old_18 = atomicAdd(&cand_cnt[qw_31], 1);
                            unsigned int slot_w_18 = _atomic_old_18;
                            if (slot_w_18 < 64) {
                                cand[qw_31 * 64 + (int)slot_w_18] = comp_30;
                            }
                        }
                        unsigned int raw_32 = __as_u32(xs[3]);
                        unsigned int mag_33 = raw_32 & 2147483647;
                        unsigned int nan_34 = (unsigned int)(mag_33 > 2139095040);
                        unsigned int bits_35 = ((mag_33 != 0) ? raw_32 : 0);
                        unsigned int neg_mask_36 = -(bits_35 >> 31);
                        unsigned int key_37 = bits_35 ^ (neg_mask_36 | 2147483648u);
                        unsigned int inv_38 = ~(unsigned int)blks[1];
                        unsigned long long comp_39 = (unsigned long long)key_37 << 32 | (unsigned long long)inv_38;
                        int qw_40 = q_local * 2 + 1;
                        if (is_stage_v[1] != 0 && nan_34 == 0 && comp_39 >= cthr_v[1]) {
                            unsigned int _atomic_old_19 = atomicAdd(&cand_cnt[qw_40], 1);
                            unsigned int slot_w_19 = _atomic_old_19;
                            if (slot_w_19 < 64) {
                                cand[qw_40 * 64 + (int)slot_w_19] = comp_39;
                            }
                        }
                        unsigned int raw_41_1 = __as_u32(xs[4]);
                        unsigned int mag_42 = raw_41_1 & 2147483647;
                        unsigned int nan_43 = (unsigned int)(mag_42 > 2139095040);
                        unsigned int bits_44 = ((mag_42 != 0) ? raw_41_1 : 0);
                        unsigned int neg_mask_45 = -(bits_44 >> 31);
                        unsigned int key_46 = bits_44 ^ (neg_mask_45 | 2147483648u);
                        unsigned int inv_47 = ~(unsigned int)blks[2];
                        unsigned long long comp_48 = (unsigned long long)key_46 << 32 | (unsigned long long)inv_47;
                        int qw_49 = q_local * 2;
                        if (is_stage_v[0] != 0 && nan_43 == 0 && comp_48 >= cthr_v[0]) {
                            unsigned int _atomic_old_20 = atomicAdd(&cand_cnt[qw_49], 1);
                            unsigned int slot_w_20 = _atomic_old_20;
                            if (slot_w_20 < 64) {
                                cand[qw_49 * 64 + (int)slot_w_20] = comp_48;
                            }
                        }
                        unsigned int raw_50 = __as_u32(xs[5]);
                        unsigned int mag_51 = raw_50 & 2147483647;
                        unsigned int nan_52 = (unsigned int)(mag_51 > 2139095040);
                        unsigned int bits_53 = ((mag_51 != 0) ? raw_50 : 0);
                        unsigned int neg_mask_54 = -(bits_53 >> 31);
                        unsigned int key_55 = bits_53 ^ (neg_mask_54 | 2147483648u);
                        unsigned int inv_56 = ~(unsigned int)blks[2];
                        unsigned long long comp_57 = (unsigned long long)key_55 << 32 | (unsigned long long)inv_56;
                        int qw_58 = q_local * 2 + 1;
                        if (is_stage_v[1] != 0 && nan_52 == 0 && comp_57 >= cthr_v[1]) {
                            unsigned int _atomic_old_21 = atomicAdd(&cand_cnt[qw_58], 1);
                            unsigned int slot_w_21 = _atomic_old_21;
                            if (slot_w_21 < 64) {
                                cand[qw_58 * 64 + (int)slot_w_21] = comp_57;
                            }
                        }
                        unsigned int raw_59 = __as_u32(xs[6]);
                        unsigned int mag_60 = raw_59 & 2147483647;
                        unsigned int nan_61 = (unsigned int)(mag_60 > 2139095040);
                        unsigned int bits_62 = ((mag_60 != 0) ? raw_59 : 0);
                        unsigned int neg_mask_63 = -(bits_62 >> 31);
                        unsigned int key_64 = bits_62 ^ (neg_mask_63 | 2147483648u);
                        unsigned int inv_65 = ~(unsigned int)blks[3];
                        unsigned long long comp_66 = (unsigned long long)key_64 << 32 | (unsigned long long)inv_65;
                        int qw_67 = q_local * 2;
                        if (is_stage_v[0] != 0 && nan_61 == 0 && comp_66 >= cthr_v[0]) {
                            unsigned int _atomic_old_22 = atomicAdd(&cand_cnt[qw_67], 1);
                            unsigned int slot_w_22 = _atomic_old_22;
                            if (slot_w_22 < 64) {
                                cand[qw_67 * 64 + (int)slot_w_22] = comp_66;
                            }
                        }
                        unsigned int raw_68_1 = __as_u32(xs[7]);
                        unsigned int mag_69_1 = raw_68_1 & 2147483647;
                        unsigned int nan_70_1 = (unsigned int)(mag_69_1 > 2139095040);
                        unsigned int bits_71_1 = ((mag_69_1 != 0) ? raw_68_1 : 0);
                        unsigned int neg_mask_72 = -(bits_71_1 >> 31);
                        unsigned int key_73 = bits_71_1 ^ (neg_mask_72 | 2147483648u);
                        unsigned int inv_74 = ~(unsigned int)blks[3];
                        unsigned long long comp_75 = (unsigned long long)key_73 << 32 | (unsigned long long)inv_74;
                        int qw_76 = q_local * 2 + 1;
                        if (is_stage_v[1] != 0 && nan_70_1 == 0 && comp_75 >= cthr_v[1]) {
                            unsigned int _atomic_old_23 = atomicAdd(&cand_cnt[qw_76], 1);
                            unsigned int slot_w_23 = _atomic_old_23;
                            if (slot_w_23 < 64) {
                                cand[qw_76 * 64 + (int)slot_w_23] = comp_75;
                            }
                        }
                        unsigned int raw_77 = __as_u32(xs[8]);
                        unsigned int mag_78 = raw_77 & 2147483647;
                        unsigned int nan_79 = (unsigned int)(mag_78 > 2139095040);
                        unsigned int bits_80 = ((mag_78 != 0) ? raw_77 : 0);
                        unsigned int neg_mask_81 = -(bits_80 >> 31);
                        unsigned int key_82 = bits_80 ^ (neg_mask_81 | 2147483648u);
                        unsigned int inv_83 = ~(unsigned int)blks[4];
                        unsigned long long comp_84 = (unsigned long long)key_82 << 32 | (unsigned long long)inv_83;
                        int qw_85 = q_local * 2;
                        if (is_stage_v[0] != 0 && nan_79 == 0 && comp_84 >= cthr_v[0]) {
                            unsigned int _atomic_old_24 = atomicAdd(&cand_cnt[qw_85], 1);
                            unsigned int slot_w_24 = _atomic_old_24;
                            if (slot_w_24 < 64) {
                                cand[qw_85 * 64 + (int)slot_w_24] = comp_84;
                            }
                        }
                        unsigned int raw_86 = __as_u32(xs[9]);
                        unsigned int mag_87 = raw_86 & 2147483647;
                        unsigned int nan_88 = (unsigned int)(mag_87 > 2139095040);
                        unsigned int bits_89 = ((mag_87 != 0) ? raw_86 : 0);
                        unsigned int neg_mask_90 = -(bits_89 >> 31);
                        unsigned int key_91 = bits_89 ^ (neg_mask_90 | 2147483648u);
                        unsigned int inv_92 = ~(unsigned int)blks[4];
                        unsigned long long comp_93 = (unsigned long long)key_91 << 32 | (unsigned long long)inv_92;
                        int qw_94 = q_local * 2 + 1;
                        if (is_stage_v[1] != 0 && nan_88 == 0 && comp_93 >= cthr_v[1]) {
                            unsigned int _atomic_old_25 = atomicAdd(&cand_cnt[qw_94], 1);
                            unsigned int slot_w_25 = _atomic_old_25;
                            if (slot_w_25 < 64) {
                                cand[qw_94 * 64 + (int)slot_w_25] = comp_93;
                            }
                        }
                        unsigned int raw_95 = __as_u32(xs[10]);
                        unsigned int mag_96 = raw_95 & 2147483647;
                        unsigned int nan_97 = (unsigned int)(mag_96 > 2139095040);
                        unsigned int bits_98 = ((mag_96 != 0) ? raw_95 : 0);
                        unsigned int neg_mask_99 = -(bits_98 >> 31);
                        unsigned int key_100 = bits_98 ^ (neg_mask_99 | 2147483648u);
                        unsigned int inv_101 = ~(unsigned int)blks[5];
                        unsigned long long comp_102 = (unsigned long long)key_100 << 32 | (unsigned long long)inv_101;
                        int qw_103 = q_local * 2;
                        if (is_stage_v[0] != 0 && nan_97 == 0 && comp_102 >= cthr_v[0]) {
                            unsigned int _atomic_old_26 = atomicAdd(&cand_cnt[qw_103], 1);
                            unsigned int slot_w_26 = _atomic_old_26;
                            if (slot_w_26 < 64) {
                                cand[qw_103 * 64 + (int)slot_w_26] = comp_102;
                            }
                        }
                        unsigned int raw_104_1 = __as_u32(xs[11]);
                        unsigned int mag_105 = raw_104_1 & 2147483647;
                        unsigned int nan_106 = (unsigned int)(mag_105 > 2139095040);
                        unsigned int bits_107 = ((mag_105 != 0) ? raw_104_1 : 0);
                        unsigned int neg_mask_108 = -(bits_107 >> 31);
                        unsigned int key_109 = bits_107 ^ (neg_mask_108 | 2147483648u);
                        unsigned int inv_110 = ~(unsigned int)blks[5];
                        unsigned long long comp_111 = (unsigned long long)key_109 << 32 | (unsigned long long)inv_110;
                        int qw_112 = q_local * 2 + 1;
                        if (is_stage_v[1] != 0 && nan_106 == 0 && comp_111 >= cthr_v[1]) {
                            unsigned int _atomic_old_27 = atomicAdd(&cand_cnt[qw_112], 1);
                            unsigned int slot_w_27 = _atomic_old_27;
                            if (slot_w_27 < 64) {
                                cand[qw_112 * 64 + (int)slot_w_27] = comp_111;
                            }
                        }
                        unsigned int raw_113 = __as_u32(xs[12]);
                        unsigned int mag_114 = raw_113 & 2147483647;
                        unsigned int nan_115 = (unsigned int)(mag_114 > 2139095040);
                        unsigned int bits_116 = ((mag_114 != 0) ? raw_113 : 0);
                        unsigned int neg_mask_117 = -(bits_116 >> 31);
                        unsigned int key_118 = bits_116 ^ (neg_mask_117 | 2147483648u);
                        unsigned int inv_119 = ~(unsigned int)blks[6];
                        unsigned long long comp_120 = (unsigned long long)key_118 << 32 | (unsigned long long)inv_119;
                        int qw_121 = q_local * 2;
                        if (is_stage_v[0] != 0 && nan_115 == 0 && comp_120 >= cthr_v[0]) {
                            unsigned int _atomic_old_28 = atomicAdd(&cand_cnt[qw_121], 1);
                            unsigned int slot_w_28 = _atomic_old_28;
                            if (slot_w_28 < 64) {
                                cand[qw_121 * 64 + (int)slot_w_28] = comp_120;
                            }
                        }
                        unsigned int raw_122 = __as_u32(xs[13]);
                        unsigned int mag_123 = raw_122 & 2147483647;
                        unsigned int nan_124 = (unsigned int)(mag_123 > 2139095040);
                        unsigned int bits_125 = ((mag_123 != 0) ? raw_122 : 0);
                        unsigned int neg_mask_126 = -(bits_125 >> 31);
                        unsigned int key_127 = bits_125 ^ (neg_mask_126 | 2147483648u);
                        unsigned int inv_128 = ~(unsigned int)blks[6];
                        unsigned long long comp_129 = (unsigned long long)key_127 << 32 | (unsigned long long)inv_128;
                        int qw_130 = q_local * 2 + 1;
                        if (is_stage_v[1] != 0 && nan_124 == 0 && comp_129 >= cthr_v[1]) {
                            unsigned int _atomic_old_29 = atomicAdd(&cand_cnt[qw_130], 1);
                            unsigned int slot_w_29 = _atomic_old_29;
                            if (slot_w_29 < 64) {
                                cand[qw_130 * 64 + (int)slot_w_29] = comp_129;
                            }
                        }
                        unsigned int raw_131_1 = __as_u32(xs[14]);
                        unsigned int mag_132_1 = raw_131_1 & 2147483647;
                        unsigned int nan_133_1 = (unsigned int)(mag_132_1 > 2139095040);
                        unsigned int bits_134_1 = ((mag_132_1 != 0) ? raw_131_1 : 0);
                        unsigned int neg_mask_135 = -(bits_134_1 >> 31);
                        unsigned int key_136 = bits_134_1 ^ (neg_mask_135 | 2147483648u);
                        unsigned int inv_137 = ~(unsigned int)blks[7];
                        unsigned long long comp_138 = (unsigned long long)key_136 << 32 | (unsigned long long)inv_137;
                        int qw_139 = q_local * 2;
                        if (is_stage_v[0] != 0 && nan_133_1 == 0 && comp_138 >= cthr_v[0]) {
                            unsigned int _atomic_old_30 = atomicAdd(&cand_cnt[qw_139], 1);
                            unsigned int slot_w_30 = _atomic_old_30;
                            if (slot_w_30 < 64) {
                                cand[qw_139 * 64 + (int)slot_w_30] = comp_138;
                            }
                        }
                        unsigned int raw_140 = __as_u32(xs[15]);
                        unsigned int mag_141 = raw_140 & 2147483647;
                        unsigned int nan_142 = (unsigned int)(mag_141 > 2139095040);
                        unsigned int bits_143 = ((mag_141 != 0) ? raw_140 : 0);
                        unsigned int neg_mask_144 = -(bits_143 >> 31);
                        unsigned int key_145 = bits_143 ^ (neg_mask_144 | 2147483648u);
                        unsigned int inv_146 = ~(unsigned int)blks[7];
                        unsigned long long comp_147 = (unsigned long long)key_145 << 32 | (unsigned long long)inv_146;
                        int qw_148 = q_local * 2 + 1;
                        if (is_stage_v[1] != 0 && nan_142 == 0 && comp_147 >= cthr_v[1]) {
                            unsigned int _atomic_old_31 = atomicAdd(&cand_cnt[qw_148], 1);
                            unsigned int slot_w_31 = _atomic_old_31;
                            if (slot_w_31 < 64) {
                                cand[qw_148 * 64 + (int)slot_w_31] = comp_147;
                            }
                        }
                    }
                }
            } else {
                #pragma unroll 1
                for (int i0_3 = 0; i0_3 < iters; i0_3 += 8) {
                    unsigned long long off_3 = off0 + (unsigned long long)i0_3 * stride16;
                    int blk_5 = lo_s + i0_3 * 16 + r_sub;
                    blks[0] = blk_5;
                    int act_6 = (int)(blk_5 < hi_s) & qvalid;
                    xs[0] = __uint_as_float(2143289344);
                    xs[1] = __uint_as_float(2143289344);
                    if (act_6 != 0) {
                        float _vec_load_28[2];
                        {
                            float2 _v2_28 = *reinterpret_cast<const float2*>(max_score + off_3);
                            _vec_load_28[0] = _v2_28.x;
                            _vec_load_28[0 + 1] = _v2_28.y;
                        }
                        xs[0] = _vec_load_28[0];
                        xs[1] = _vec_load_28[1];
                    }
                    off_3 = off_3 + stride16;
                    int blk_0_2 = lo_s + (i0_3 + 1) * 16 + r_sub;
                    blks[1] = blk_0_2;
                    int act_1_2 = (int)(blk_0_2 < hi_s) & qvalid;
                    xs[2] = __uint_as_float(2143289344);
                    xs[3] = __uint_as_float(2143289344);
                    if (act_1_2 != 0) {
                        float _vec_load_29[2];
                        {
                            float2 _v2_29 = *reinterpret_cast<const float2*>(max_score + off_3);
                            _vec_load_29[0] = _v2_29.x;
                            _vec_load_29[0 + 1] = _v2_29.y;
                        }
                        xs[2] = _vec_load_29[0];
                        xs[3] = _vec_load_29[1];
                    }
                    off_3 = off_3 + stride16;
                    int blk_2_2 = lo_s + (i0_3 + 2) * 16 + r_sub;
                    blks[2] = blk_2_2;
                    int act_3_3 = (int)(blk_2_2 < hi_s) & qvalid;
                    xs[4] = __uint_as_float(2143289344);
                    xs[5] = __uint_as_float(2143289344);
                    if (act_3_3 != 0) {
                        float _vec_load_30[2];
                        {
                            float2 _v2_30 = *reinterpret_cast<const float2*>(max_score + off_3);
                            _vec_load_30[0] = _v2_30.x;
                            _vec_load_30[0 + 1] = _v2_30.y;
                        }
                        xs[4] = _vec_load_30[0];
                        xs[5] = _vec_load_30[1];
                    }
                    off_3 = off_3 + stride16;
                    int blk_4_3 = lo_s + (i0_3 + 3) * 16 + r_sub;
                    blks[3] = blk_4_3;
                    int act_5_2 = (int)(blk_4_3 < hi_s) & qvalid;
                    xs[6] = __uint_as_float(2143289344);
                    xs[7] = __uint_as_float(2143289344);
                    if (act_5_2 != 0) {
                        float _vec_load_31[2];
                        {
                            float2 _v2_31 = *reinterpret_cast<const float2*>(max_score + off_3);
                            _vec_load_31[0] = _v2_31.x;
                            _vec_load_31[0 + 1] = _v2_31.y;
                        }
                        xs[6] = _vec_load_31[0];
                        xs[7] = _vec_load_31[1];
                    }
                    off_3 = off_3 + stride16;
                    int blk_6_2 = lo_s + (i0_3 + 4) * 16 + r_sub;
                    blks[4] = blk_6_2;
                    int act_7_3 = (int)(blk_6_2 < hi_s) & qvalid;
                    xs[8] = __uint_as_float(2143289344);
                    xs[9] = __uint_as_float(2143289344);
                    if (act_7_3 != 0) {
                        float _vec_load_32[2];
                        {
                            float2 _v2_32 = *reinterpret_cast<const float2*>(max_score + off_3);
                            _vec_load_32[0] = _v2_32.x;
                            _vec_load_32[0 + 1] = _v2_32.y;
                        }
                        xs[8] = _vec_load_32[0];
                        xs[9] = _vec_load_32[1];
                    }
                    off_3 = off_3 + stride16;
                    int blk_8_3 = lo_s + (i0_3 + 5) * 16 + r_sub;
                    blks[5] = blk_8_3;
                    int act_9_2 = (int)(blk_8_3 < hi_s) & qvalid;
                    xs[10] = __uint_as_float(2143289344);
                    xs[11] = __uint_as_float(2143289344);
                    if (act_9_2 != 0) {
                        float _vec_load_33[2];
                        {
                            float2 _v2_33 = *reinterpret_cast<const float2*>(max_score + off_3);
                            _vec_load_33[0] = _v2_33.x;
                            _vec_load_33[0 + 1] = _v2_33.y;
                        }
                        xs[10] = _vec_load_33[0];
                        xs[11] = _vec_load_33[1];
                    }
                    off_3 = off_3 + stride16;
                    int blk_10_2 = lo_s + (i0_3 + 6) * 16 + r_sub;
                    blks[6] = blk_10_2;
                    int act_11_3 = (int)(blk_10_2 < hi_s) & qvalid;
                    xs[12] = __uint_as_float(2143289344);
                    xs[13] = __uint_as_float(2143289344);
                    if (act_11_3 != 0) {
                        float _vec_load_34[2];
                        {
                            float2 _v2_34 = *reinterpret_cast<const float2*>(max_score + off_3);
                            _vec_load_34[0] = _v2_34.x;
                            _vec_load_34[0 + 1] = _v2_34.y;
                        }
                        xs[12] = _vec_load_34[0];
                        xs[13] = _vec_load_34[1];
                    }
                    off_3 = off_3 + stride16;
                    int blk_12_3 = lo_s + (i0_3 + 7) * 16 + r_sub;
                    blks[7] = blk_12_3;
                    int act_13_2 = (int)(blk_12_3 < hi_s) & qvalid;
                    xs[14] = __uint_as_float(2143289344);
                    xs[15] = __uint_as_float(2143289344);
                    if (act_13_2 != 0) {
                        float _vec_load_35[2];
                        {
                            float2 _v2_35 = *reinterpret_cast<const float2*>(max_score + off_3);
                            _vec_load_35[0] = _v2_35.x;
                            _vec_load_35[0 + 1] = _v2_35.y;
                        }
                        xs[14] = _vec_load_35[0];
                        xs[15] = _vec_load_35[1];
                    }
                    off_3 = off_3 + stride16;
                    int blk2 = blks[0];
                    unsigned int raw_3 = __as_u32(xs[0]);
                    unsigned int mag_3 = raw_3 & 2147483647;
                    unsigned int nan_3 = (unsigned int)(mag_3 > 2139095040);
                    unsigned int bits_3 = ((mag_3 != 0) ? raw_3 : 0);
                    int em_1 = (int)(bits_3 >> 19 & 4095);
                    int _max_163 = ((em_1 - 1792) > (0) ? (em_1 - 1792) : (0));
                    int _min_34 = ((_max_163) < (511) ? (_max_163) : (511));
                    int raw_14_2 = _min_34;
                    int neg_1 = (int)(bits_3 >> 31);
                    int h10_1 = ((neg_1 != 0) ? 511 - raw_14_2 : 512 + raw_14_2);
                    unsigned int neg_mask_2 = -(bits_3 >> 31);
                    unsigned int key_2 = bits_3 ^ (neg_mask_2 | 2147483648u);
                    unsigned int invm = ~(unsigned int)blk2 & idx_mask;
                    unsigned long long rk = (unsigned long long)h10_1 << 54 | (unsigned long long)key_2 << 22 | (unsigned long long)invm;
                    int in_window_w = is_pass0_v[0] | (int)(rk >> rsh_v[0] == rprefix_v[0]);
                    int d_w = (int)((unsigned int)(rk >> dsh_v[0]) & dmask_v[0]);
                    int qw_2 = q_local * 2;
                    int counted_w = is_hist_v[0] & (int)(nan_3 == 0) & in_window_w;
                    int dd_w = ((counted_w != 0) ? d_w : 1024);
                    atomicAdd(&hist_ext[dd_w * 16 + q_local], 1);
                    unsigned int inv_2 = ~(unsigned int)blk2;
                    unsigned long long comp_2 = (unsigned long long)key_2 << 32 | (unsigned long long)inv_2;
                    if (is_stage_v[0] != 0 && nan_3 == 0 && comp_2 >= cthr_v[0]) {
                        unsigned int _atomic_old_32 = atomicAdd(&cand_cnt[qw_2], 1);
                        unsigned int slot_w_32 = _atomic_old_32;
                        if (slot_w_32 < 64) {
                            cand[qw_2 * 64 + (int)slot_w_32] = comp_2;
                        }
                    }
                    unsigned int raw_15_1 = __as_u32(xs[1]);
                    unsigned int mag_16_1 = raw_15_1 & 2147483647;
                    unsigned int nan_17_1 = (unsigned int)(mag_16_1 > 2139095040);
                    unsigned int bits_18_1 = ((mag_16_1 != 0) ? raw_15_1 : 0);
                    int em_19_1 = (int)(bits_18_1 >> 19 & 4095);
                    int _max_164 = ((em_19_1 - 1792) > (0) ? (em_19_1 - 1792) : (0));
                    int _min_35 = ((_max_164) < (511) ? (_max_164) : (511));
                    int raw_20_1 = _min_35;
                    int neg_21_1 = (int)(bits_18_1 >> 31);
                    int h10_22_1 = ((neg_21_1 != 0) ? 511 - raw_20_1 : 512 + raw_20_1);
                    unsigned int neg_mask_23 = -(bits_18_1 >> 31);
                    unsigned int key_24 = bits_18_1 ^ (neg_mask_23 | 2147483648u);
                    unsigned int invm_25 = ~(unsigned int)blk2 & idx_mask;
                    unsigned long long rk_26 = (unsigned long long)h10_22_1 << 54 | (unsigned long long)key_24 << 22 | (unsigned long long)invm_25;
                    int in_window_w_27 = is_pass0_v[1] | (int)(rk_26 >> rsh_v[1] == rprefix_v[1]);
                    int d_w_28 = (int)((unsigned int)(rk_26 >> dsh_v[1]) & dmask_v[1]);
                    int qw_29 = q_local * 2 + 1;
                    int counted_w_30 = is_hist_v[1] & (int)(nan_17_1 == 0) & in_window_w_27;
                    int dd_w_31 = ((counted_w_30 != 0) ? d_w_28 : 1024);
                    atomicAdd(&hist_ext[dd_w_31 * 16 + q_local], 65536);
                    unsigned int inv_32 = ~(unsigned int)blk2;
                    unsigned long long comp_33 = (unsigned long long)key_24 << 32 | (unsigned long long)inv_32;
                    if (is_stage_v[1] != 0 && nan_17_1 == 0 && comp_33 >= cthr_v[1]) {
                        unsigned int _atomic_old_33 = atomicAdd(&cand_cnt[qw_29], 1);
                        unsigned int slot_w_33 = _atomic_old_33;
                        if (slot_w_33 < 64) {
                            cand[qw_29 * 64 + (int)slot_w_33] = comp_33;
                        }
                    }
                    int blk2_34 = blks[1];
                    unsigned int raw_35 = __as_u32(xs[2]);
                    unsigned int mag_36 = raw_35 & 2147483647;
                    unsigned int nan_37 = (unsigned int)(mag_36 > 2139095040);
                    unsigned int bits_38 = ((mag_36 != 0) ? raw_35 : 0);
                    int em_39 = (int)(bits_38 >> 19 & 4095);
                    int _max_165 = ((em_39 - 1792) > (0) ? (em_39 - 1792) : (0));
                    int _min_36 = ((_max_165) < (511) ? (_max_165) : (511));
                    int raw_40 = _min_36;
                    int neg_41 = (int)(bits_38 >> 31);
                    int h10_42 = ((neg_41 != 0) ? 511 - raw_40 : 512 + raw_40);
                    unsigned int neg_mask_43_1 = -(bits_38 >> 31);
                    unsigned int key_44_1 = bits_38 ^ (neg_mask_43_1 | 2147483648u);
                    unsigned int invm_45 = ~(unsigned int)blk2_34 & idx_mask;
                    unsigned long long rk_46 = (unsigned long long)h10_42 << 54 | (unsigned long long)key_44_1 << 22 | (unsigned long long)invm_45;
                    int in_window_w_47 = is_pass0_v[0] | (int)(rk_46 >> rsh_v[0] == rprefix_v[0]);
                    int d_w_48 = (int)((unsigned int)(rk_46 >> dsh_v[0]) & dmask_v[0]);
                    int qw_49_1 = q_local * 2;
                    int counted_w_50 = is_hist_v[0] & (int)(nan_37 == 0) & in_window_w_47;
                    int dd_w_51 = ((counted_w_50 != 0) ? d_w_48 : 1024);
                    atomicAdd(&hist_ext[dd_w_51 * 16 + q_local], 1);
                    unsigned int inv_52 = ~(unsigned int)blk2_34;
                    unsigned long long comp_53 = (unsigned long long)key_44_1 << 32 | (unsigned long long)inv_52;
                    if (is_stage_v[0] != 0 && nan_37 == 0 && comp_53 >= cthr_v[0]) {
                        unsigned int _atomic_old_34 = atomicAdd(&cand_cnt[qw_49_1], 1);
                        unsigned int slot_w_34 = _atomic_old_34;
                        if (slot_w_34 < 64) {
                            cand[qw_49_1 * 64 + (int)slot_w_34] = comp_53;
                        }
                    }
                    unsigned int raw_54 = __as_u32(xs[3]);
                    unsigned int mag_55 = raw_54 & 2147483647;
                    unsigned int nan_56 = (unsigned int)(mag_55 > 2139095040);
                    unsigned int bits_57 = ((mag_55 != 0) ? raw_54 : 0);
                    int em_58 = (int)(bits_57 >> 19 & 4095);
                    int _max_166 = ((em_58 - 1792) > (0) ? (em_58 - 1792) : (0));
                    int _min_37 = ((_max_166) < (511) ? (_max_166) : (511));
                    int raw_59_1 = _min_37;
                    int neg_60 = (int)(bits_57 >> 31);
                    int h10_61 = ((neg_60 != 0) ? 511 - raw_59_1 : 512 + raw_59_1);
                    unsigned int neg_mask_62 = -(bits_57 >> 31);
                    unsigned int key_63 = bits_57 ^ (neg_mask_62 | 2147483648u);
                    unsigned int invm_64 = ~(unsigned int)blk2_34 & idx_mask;
                    unsigned long long rk_65 = (unsigned long long)h10_61 << 54 | (unsigned long long)key_63 << 22 | (unsigned long long)invm_64;
                    int in_window_w_66 = is_pass0_v[1] | (int)(rk_65 >> rsh_v[1] == rprefix_v[1]);
                    int d_w_67 = (int)((unsigned int)(rk_65 >> dsh_v[1]) & dmask_v[1]);
                    int qw_68 = q_local * 2 + 1;
                    int counted_w_69 = is_hist_v[1] & (int)(nan_56 == 0) & in_window_w_66;
                    int dd_w_70 = ((counted_w_69 != 0) ? d_w_67 : 1024);
                    atomicAdd(&hist_ext[dd_w_70 * 16 + q_local], 65536);
                    unsigned int inv_71 = ~(unsigned int)blk2_34;
                    unsigned long long comp_72 = (unsigned long long)key_63 << 32 | (unsigned long long)inv_71;
                    if (is_stage_v[1] != 0 && nan_56 == 0 && comp_72 >= cthr_v[1]) {
                        unsigned int _atomic_old_35 = atomicAdd(&cand_cnt[qw_68], 1);
                        unsigned int slot_w_35 = _atomic_old_35;
                        if (slot_w_35 < 64) {
                            cand[qw_68 * 64 + (int)slot_w_35] = comp_72;
                        }
                    }
                    int blk2_73 = blks[2];
                    unsigned int raw_74 = __as_u32(xs[4]);
                    unsigned int mag_75 = raw_74 & 2147483647;
                    unsigned int nan_76 = (unsigned int)(mag_75 > 2139095040);
                    unsigned int bits_77 = ((mag_75 != 0) ? raw_74 : 0);
                    int em_78 = (int)(bits_77 >> 19 & 4095);
                    int _max_167 = ((em_78 - 1792) > (0) ? (em_78 - 1792) : (0));
                    int _min_38 = ((_max_167) < (511) ? (_max_167) : (511));
                    int raw_79 = _min_38;
                    int neg_80 = (int)(bits_77 >> 31);
                    int h10_81 = ((neg_80 != 0) ? 511 - raw_79 : 512 + raw_79);
                    unsigned int neg_mask_82 = -(bits_77 >> 31);
                    unsigned int key_83 = bits_77 ^ (neg_mask_82 | 2147483648u);
                    unsigned int invm_84 = ~(unsigned int)blk2_73 & idx_mask;
                    unsigned long long rk_85 = (unsigned long long)h10_81 << 54 | (unsigned long long)key_83 << 22 | (unsigned long long)invm_84;
                    int in_window_w_86 = is_pass0_v[0] | (int)(rk_85 >> rsh_v[0] == rprefix_v[0]);
                    int d_w_87 = (int)((unsigned int)(rk_85 >> dsh_v[0]) & dmask_v[0]);
                    int qw_88 = q_local * 2;
                    int counted_w_89 = is_hist_v[0] & (int)(nan_76 == 0) & in_window_w_86;
                    int dd_w_90 = ((counted_w_89 != 0) ? d_w_87 : 1024);
                    atomicAdd(&hist_ext[dd_w_90 * 16 + q_local], 1);
                    unsigned int inv_91 = ~(unsigned int)blk2_73;
                    unsigned long long comp_92 = (unsigned long long)key_83 << 32 | (unsigned long long)inv_91;
                    if (is_stage_v[0] != 0 && nan_76 == 0 && comp_92 >= cthr_v[0]) {
                        unsigned int _atomic_old_36 = atomicAdd(&cand_cnt[qw_88], 1);
                        unsigned int slot_w_36 = _atomic_old_36;
                        if (slot_w_36 < 64) {
                            cand[qw_88 * 64 + (int)slot_w_36] = comp_92;
                        }
                    }
                    unsigned int raw_93_1 = __as_u32(xs[5]);
                    unsigned int mag_94_1 = raw_93_1 & 2147483647;
                    unsigned int nan_95_1 = (unsigned int)(mag_94_1 > 2139095040);
                    unsigned int bits_96_1 = ((mag_94_1 != 0) ? raw_93_1 : 0);
                    int em_97 = (int)(bits_96_1 >> 19 & 4095);
                    int _max_168 = ((em_97 - 1792) > (0) ? (em_97 - 1792) : (0));
                    int _min_39 = ((_max_168) < (511) ? (_max_168) : (511));
                    int raw_98 = _min_39;
                    int neg_99 = (int)(bits_96_1 >> 31);
                    int h10_100 = ((neg_99 != 0) ? 511 - raw_98 : 512 + raw_98);
                    unsigned int neg_mask_101 = -(bits_96_1 >> 31);
                    unsigned int key_102 = bits_96_1 ^ (neg_mask_101 | 2147483648u);
                    unsigned int invm_103 = ~(unsigned int)blk2_73 & idx_mask;
                    unsigned long long rk_104 = (unsigned long long)h10_100 << 54 | (unsigned long long)key_102 << 22 | (unsigned long long)invm_103;
                    int in_window_w_105 = is_pass0_v[1] | (int)(rk_104 >> rsh_v[1] == rprefix_v[1]);
                    int d_w_106 = (int)((unsigned int)(rk_104 >> dsh_v[1]) & dmask_v[1]);
                    int qw_107 = q_local * 2 + 1;
                    int counted_w_108 = is_hist_v[1] & (int)(nan_95_1 == 0) & in_window_w_105;
                    int dd_w_109 = ((counted_w_108 != 0) ? d_w_106 : 1024);
                    atomicAdd(&hist_ext[dd_w_109 * 16 + q_local], 65536);
                    unsigned int inv_110_1 = ~(unsigned int)blk2_73;
                    unsigned long long comp_111_1 = (unsigned long long)key_102 << 32 | (unsigned long long)inv_110_1;
                    if (is_stage_v[1] != 0 && nan_95_1 == 0 && comp_111_1 >= cthr_v[1]) {
                        unsigned int _atomic_old_37 = atomicAdd(&cand_cnt[qw_107], 1);
                        unsigned int slot_w_37 = _atomic_old_37;
                        if (slot_w_37 < 64) {
                            cand[qw_107 * 64 + (int)slot_w_37] = comp_111_1;
                        }
                    }
                    int blk2_112 = blks[3];
                    unsigned int raw_113_1 = __as_u32(xs[6]);
                    unsigned int mag_114_1 = raw_113_1 & 2147483647;
                    unsigned int nan_115_1 = (unsigned int)(mag_114_1 > 2139095040);
                    unsigned int bits_116_1 = ((mag_114_1 != 0) ? raw_113_1 : 0);
                    int em_117 = (int)(bits_116_1 >> 19 & 4095);
                    int _max_169 = ((em_117 - 1792) > (0) ? (em_117 - 1792) : (0));
                    int _min_40 = ((_max_169) < (511) ? (_max_169) : (511));
                    int raw_118 = _min_40;
                    int neg_119 = (int)(bits_116_1 >> 31);
                    int h10_120 = ((neg_119 != 0) ? 511 - raw_118 : 512 + raw_118);
                    unsigned int neg_mask_121 = -(bits_116_1 >> 31);
                    unsigned int key_122 = bits_116_1 ^ (neg_mask_121 | 2147483648u);
                    unsigned int invm_123 = ~(unsigned int)blk2_112 & idx_mask;
                    unsigned long long rk_124 = (unsigned long long)h10_120 << 54 | (unsigned long long)key_122 << 22 | (unsigned long long)invm_123;
                    int in_window_w_125 = is_pass0_v[0] | (int)(rk_124 >> rsh_v[0] == rprefix_v[0]);
                    int d_w_126 = (int)((unsigned int)(rk_124 >> dsh_v[0]) & dmask_v[0]);
                    int qw_127 = q_local * 2;
                    int counted_w_128 = is_hist_v[0] & (int)(nan_115_1 == 0) & in_window_w_125;
                    int dd_w_129 = ((counted_w_128 != 0) ? d_w_126 : 1024);
                    atomicAdd(&hist_ext[dd_w_129 * 16 + q_local], 1);
                    unsigned int inv_130 = ~(unsigned int)blk2_112;
                    unsigned long long comp_131 = (unsigned long long)key_122 << 32 | (unsigned long long)inv_130;
                    if (is_stage_v[0] != 0 && nan_115_1 == 0 && comp_131 >= cthr_v[0]) {
                        unsigned int _atomic_old_38 = atomicAdd(&cand_cnt[qw_127], 1);
                        unsigned int slot_w_38 = _atomic_old_38;
                        if (slot_w_38 < 64) {
                            cand[qw_127 * 64 + (int)slot_w_38] = comp_131;
                        }
                    }
                    unsigned int raw_132 = __as_u32(xs[7]);
                    unsigned int mag_133 = raw_132 & 2147483647;
                    unsigned int nan_134 = (unsigned int)(mag_133 > 2139095040);
                    unsigned int bits_135 = ((mag_133 != 0) ? raw_132 : 0);
                    int em_136 = (int)(bits_135 >> 19 & 4095);
                    int _max_170 = ((em_136 - 1792) > (0) ? (em_136 - 1792) : (0));
                    int _min_41 = ((_max_170) < (511) ? (_max_170) : (511));
                    int raw_137 = _min_41;
                    int neg_138 = (int)(bits_135 >> 31);
                    int h10_139 = ((neg_138 != 0) ? 511 - raw_137 : 512 + raw_137);
                    unsigned int neg_mask_140 = -(bits_135 >> 31);
                    unsigned int key_141 = bits_135 ^ (neg_mask_140 | 2147483648u);
                    unsigned int invm_142 = ~(unsigned int)blk2_112 & idx_mask;
                    unsigned long long rk_143 = (unsigned long long)h10_139 << 54 | (unsigned long long)key_141 << 22 | (unsigned long long)invm_142;
                    int in_window_w_144 = is_pass0_v[1] | (int)(rk_143 >> rsh_v[1] == rprefix_v[1]);
                    int d_w_145 = (int)((unsigned int)(rk_143 >> dsh_v[1]) & dmask_v[1]);
                    int qw_146_1 = q_local * 2 + 1;
                    int counted_w_147 = is_hist_v[1] & (int)(nan_134 == 0) & in_window_w_144;
                    int dd_w_148 = ((counted_w_147 != 0) ? d_w_145 : 1024);
                    atomicAdd(&hist_ext[dd_w_148 * 16 + q_local], 65536);
                    unsigned int inv_149 = ~(unsigned int)blk2_112;
                    unsigned long long comp_150 = (unsigned long long)key_141 << 32 | (unsigned long long)inv_149;
                    if (is_stage_v[1] != 0 && nan_134 == 0 && comp_150 >= cthr_v[1]) {
                        unsigned int _atomic_old_39 = atomicAdd(&cand_cnt[qw_146_1], 1);
                        unsigned int slot_w_39 = _atomic_old_39;
                        if (slot_w_39 < 64) {
                            cand[qw_146_1 * 64 + (int)slot_w_39] = comp_150;
                        }
                    }
                    int blk2_151 = blks[4];
                    unsigned int raw_152_1 = __as_u32(xs[8]);
                    unsigned int mag_153_1 = raw_152_1 & 2147483647;
                    unsigned int nan_154_1 = (unsigned int)(mag_153_1 > 2139095040);
                    unsigned int bits_155_1 = ((mag_153_1 != 0) ? raw_152_1 : 0);
                    int em_156_1 = (int)(bits_155_1 >> 19 & 4095);
                    int _max_171 = ((em_156_1 - 1792) > (0) ? (em_156_1 - 1792) : (0));
                    int _min_42 = ((_max_171) < (511) ? (_max_171) : (511));
                    int raw_157_1 = _min_42;
                    int neg_158_1 = (int)(bits_155_1 >> 31);
                    int h10_159_1 = ((neg_158_1 != 0) ? 511 - raw_157_1 : 512 + raw_157_1);
                    unsigned int neg_mask_160_1 = -(bits_155_1 >> 31);
                    unsigned int key_161_1 = bits_155_1 ^ (neg_mask_160_1 | 2147483648u);
                    unsigned int invm_162 = ~(unsigned int)blk2_151 & idx_mask;
                    unsigned long long rk_163 = (unsigned long long)h10_159_1 << 54 | (unsigned long long)key_161_1 << 22 | (unsigned long long)invm_162;
                    int in_window_w_164 = is_pass0_v[0] | (int)(rk_163 >> rsh_v[0] == rprefix_v[0]);
                    int d_w_165 = (int)((unsigned int)(rk_163 >> dsh_v[0]) & dmask_v[0]);
                    int qw_166 = q_local * 2;
                    int counted_w_167 = is_hist_v[0] & (int)(nan_154_1 == 0) & in_window_w_164;
                    int dd_w_168 = ((counted_w_167 != 0) ? d_w_165 : 1024);
                    atomicAdd(&hist_ext[dd_w_168 * 16 + q_local], 1);
                    unsigned int inv_169 = ~(unsigned int)blk2_151;
                    unsigned long long comp_170 = (unsigned long long)key_161_1 << 32 | (unsigned long long)inv_169;
                    if (is_stage_v[0] != 0 && nan_154_1 == 0 && comp_170 >= cthr_v[0]) {
                        unsigned int _atomic_old_40 = atomicAdd(&cand_cnt[qw_166], 1);
                        unsigned int slot_w_40 = _atomic_old_40;
                        if (slot_w_40 < 64) {
                            cand[qw_166 * 64 + (int)slot_w_40] = comp_170;
                        }
                    }
                    unsigned int raw_171 = __as_u32(xs[9]);
                    unsigned int mag_172 = raw_171 & 2147483647;
                    unsigned int nan_173 = (unsigned int)(mag_172 > 2139095040);
                    unsigned int bits_174 = ((mag_172 != 0) ? raw_171 : 0);
                    int em_175 = (int)(bits_174 >> 19 & 4095);
                    int _max_172 = ((em_175 - 1792) > (0) ? (em_175 - 1792) : (0));
                    int _min_43 = ((_max_172) < (511) ? (_max_172) : (511));
                    int raw_176 = _min_43;
                    int neg_177 = (int)(bits_174 >> 31);
                    int h10_178 = ((neg_177 != 0) ? 511 - raw_176 : 512 + raw_176);
                    unsigned int neg_mask_179 = -(bits_174 >> 31);
                    unsigned int key_180 = bits_174 ^ (neg_mask_179 | 2147483648u);
                    unsigned int invm_181 = ~(unsigned int)blk2_151 & idx_mask;
                    unsigned long long rk_182 = (unsigned long long)h10_178 << 54 | (unsigned long long)key_180 << 22 | (unsigned long long)invm_181;
                    int in_window_w_183 = is_pass0_v[1] | (int)(rk_182 >> rsh_v[1] == rprefix_v[1]);
                    int d_w_184 = (int)((unsigned int)(rk_182 >> dsh_v[1]) & dmask_v[1]);
                    int qw_185 = q_local * 2 + 1;
                    int counted_w_186 = is_hist_v[1] & (int)(nan_173 == 0) & in_window_w_183;
                    int dd_w_187 = ((counted_w_186 != 0) ? d_w_184 : 1024);
                    atomicAdd(&hist_ext[dd_w_187 * 16 + q_local], 65536);
                    unsigned int inv_188 = ~(unsigned int)blk2_151;
                    unsigned long long comp_189 = (unsigned long long)key_180 << 32 | (unsigned long long)inv_188;
                    if (is_stage_v[1] != 0 && nan_173 == 0 && comp_189 >= cthr_v[1]) {
                        unsigned int _atomic_old_41 = atomicAdd(&cand_cnt[qw_185], 1);
                        unsigned int slot_w_41 = _atomic_old_41;
                        if (slot_w_41 < 64) {
                            cand[qw_185 * 64 + (int)slot_w_41] = comp_189;
                        }
                    }
                    int blk2_190 = blks[5];
                    unsigned int raw_191 = __as_u32(xs[10]);
                    unsigned int mag_192 = raw_191 & 2147483647;
                    unsigned int nan_193 = (unsigned int)(mag_192 > 2139095040);
                    unsigned int bits_194 = ((mag_192 != 0) ? raw_191 : 0);
                    int em_195 = (int)(bits_194 >> 19 & 4095);
                    int _max_173 = ((em_195 - 1792) > (0) ? (em_195 - 1792) : (0));
                    int _min_44 = ((_max_173) < (511) ? (_max_173) : (511));
                    int raw_196 = _min_44;
                    int neg_197 = (int)(bits_194 >> 31);
                    int h10_198 = ((neg_197 != 0) ? 511 - raw_196 : 512 + raw_196);
                    unsigned int neg_mask_199 = -(bits_194 >> 31);
                    unsigned int key_200 = bits_194 ^ (neg_mask_199 | 2147483648u);
                    unsigned int invm_201 = ~(unsigned int)blk2_190 & idx_mask;
                    unsigned long long rk_202 = (unsigned long long)h10_198 << 54 | (unsigned long long)key_200 << 22 | (unsigned long long)invm_201;
                    int in_window_w_203 = is_pass0_v[0] | (int)(rk_202 >> rsh_v[0] == rprefix_v[0]);
                    int d_w_204 = (int)((unsigned int)(rk_202 >> dsh_v[0]) & dmask_v[0]);
                    int qw_205 = q_local * 2;
                    int counted_w_206 = is_hist_v[0] & (int)(nan_193 == 0) & in_window_w_203;
                    int dd_w_207 = ((counted_w_206 != 0) ? d_w_204 : 1024);
                    atomicAdd(&hist_ext[dd_w_207 * 16 + q_local], 1);
                    unsigned int inv_208 = ~(unsigned int)blk2_190;
                    unsigned long long comp_209 = (unsigned long long)key_200 << 32 | (unsigned long long)inv_208;
                    if (is_stage_v[0] != 0 && nan_193 == 0 && comp_209 >= cthr_v[0]) {
                        unsigned int _atomic_old_42 = atomicAdd(&cand_cnt[qw_205], 1);
                        unsigned int slot_w_42 = _atomic_old_42;
                        if (slot_w_42 < 64) {
                            cand[qw_205 * 64 + (int)slot_w_42] = comp_209;
                        }
                    }
                    unsigned int raw_210 = __as_u32(xs[11]);
                    unsigned int mag_211 = raw_210 & 2147483647;
                    unsigned int nan_212 = (unsigned int)(mag_211 > 2139095040);
                    unsigned int bits_213 = ((mag_211 != 0) ? raw_210 : 0);
                    int em_214 = (int)(bits_213 >> 19 & 4095);
                    int _max_174 = ((em_214 - 1792) > (0) ? (em_214 - 1792) : (0));
                    int _min_45 = ((_max_174) < (511) ? (_max_174) : (511));
                    int raw_215 = _min_45;
                    int neg_216 = (int)(bits_213 >> 31);
                    int h10_217 = ((neg_216 != 0) ? 511 - raw_215 : 512 + raw_215);
                    unsigned int neg_mask_218 = -(bits_213 >> 31);
                    unsigned int key_219 = bits_213 ^ (neg_mask_218 | 2147483648u);
                    unsigned int invm_220 = ~(unsigned int)blk2_190 & idx_mask;
                    unsigned long long rk_221 = (unsigned long long)h10_217 << 54 | (unsigned long long)key_219 << 22 | (unsigned long long)invm_220;
                    int in_window_w_222 = is_pass0_v[1] | (int)(rk_221 >> rsh_v[1] == rprefix_v[1]);
                    int d_w_223 = (int)((unsigned int)(rk_221 >> dsh_v[1]) & dmask_v[1]);
                    int qw_224 = q_local * 2 + 1;
                    int counted_w_225 = is_hist_v[1] & (int)(nan_212 == 0) & in_window_w_222;
                    int dd_w_226 = ((counted_w_225 != 0) ? d_w_223 : 1024);
                    atomicAdd(&hist_ext[dd_w_226 * 16 + q_local], 65536);
                    unsigned int inv_227 = ~(unsigned int)blk2_190;
                    unsigned long long comp_228 = (unsigned long long)key_219 << 32 | (unsigned long long)inv_227;
                    if (is_stage_v[1] != 0 && nan_212 == 0 && comp_228 >= cthr_v[1]) {
                        unsigned int _atomic_old_43 = atomicAdd(&cand_cnt[qw_224], 1);
                        unsigned int slot_w_43 = _atomic_old_43;
                        if (slot_w_43 < 64) {
                            cand[qw_224 * 64 + (int)slot_w_43] = comp_228;
                        }
                    }
                    int blk2_229 = blks[6];
                    unsigned int raw_230 = __as_u32(xs[12]);
                    unsigned int mag_231 = raw_230 & 2147483647;
                    unsigned int nan_232 = (unsigned int)(mag_231 > 2139095040);
                    unsigned int bits_233 = ((mag_231 != 0) ? raw_230 : 0);
                    int em_234 = (int)(bits_233 >> 19 & 4095);
                    int _max_175 = ((em_234 - 1792) > (0) ? (em_234 - 1792) : (0));
                    int _min_46 = ((_max_175) < (511) ? (_max_175) : (511));
                    int raw_235 = _min_46;
                    int neg_236 = (int)(bits_233 >> 31);
                    int h10_237 = ((neg_236 != 0) ? 511 - raw_235 : 512 + raw_235);
                    unsigned int neg_mask_238 = -(bits_233 >> 31);
                    unsigned int key_239 = bits_233 ^ (neg_mask_238 | 2147483648u);
                    unsigned int invm_240 = ~(unsigned int)blk2_229 & idx_mask;
                    unsigned long long rk_241 = (unsigned long long)h10_237 << 54 | (unsigned long long)key_239 << 22 | (unsigned long long)invm_240;
                    int in_window_w_242 = is_pass0_v[0] | (int)(rk_241 >> rsh_v[0] == rprefix_v[0]);
                    int d_w_243 = (int)((unsigned int)(rk_241 >> dsh_v[0]) & dmask_v[0]);
                    int qw_244 = q_local * 2;
                    int counted_w_245 = is_hist_v[0] & (int)(nan_232 == 0) & in_window_w_242;
                    int dd_w_246 = ((counted_w_245 != 0) ? d_w_243 : 1024);
                    atomicAdd(&hist_ext[dd_w_246 * 16 + q_local], 1);
                    unsigned int inv_247 = ~(unsigned int)blk2_229;
                    unsigned long long comp_248 = (unsigned long long)key_239 << 32 | (unsigned long long)inv_247;
                    if (is_stage_v[0] != 0 && nan_232 == 0 && comp_248 >= cthr_v[0]) {
                        unsigned int _atomic_old_44 = atomicAdd(&cand_cnt[qw_244], 1);
                        unsigned int slot_w_44 = _atomic_old_44;
                        if (slot_w_44 < 64) {
                            cand[qw_244 * 64 + (int)slot_w_44] = comp_248;
                        }
                    }
                    unsigned int raw_249 = __as_u32(xs[13]);
                    unsigned int mag_250 = raw_249 & 2147483647;
                    unsigned int nan_251 = (unsigned int)(mag_250 > 2139095040);
                    unsigned int bits_252 = ((mag_250 != 0) ? raw_249 : 0);
                    int em_253 = (int)(bits_252 >> 19 & 4095);
                    int _max_176 = ((em_253 - 1792) > (0) ? (em_253 - 1792) : (0));
                    int _min_47 = ((_max_176) < (511) ? (_max_176) : (511));
                    int raw_254 = _min_47;
                    int neg_255 = (int)(bits_252 >> 31);
                    int h10_256 = ((neg_255 != 0) ? 511 - raw_254 : 512 + raw_254);
                    unsigned int neg_mask_257 = -(bits_252 >> 31);
                    unsigned int key_258 = bits_252 ^ (neg_mask_257 | 2147483648u);
                    unsigned int invm_259 = ~(unsigned int)blk2_229 & idx_mask;
                    unsigned long long rk_260 = (unsigned long long)h10_256 << 54 | (unsigned long long)key_258 << 22 | (unsigned long long)invm_259;
                    int in_window_w_261 = is_pass0_v[1] | (int)(rk_260 >> rsh_v[1] == rprefix_v[1]);
                    int d_w_262 = (int)((unsigned int)(rk_260 >> dsh_v[1]) & dmask_v[1]);
                    int qw_263 = q_local * 2 + 1;
                    int counted_w_264 = is_hist_v[1] & (int)(nan_251 == 0) & in_window_w_261;
                    int dd_w_265 = ((counted_w_264 != 0) ? d_w_262 : 1024);
                    atomicAdd(&hist_ext[dd_w_265 * 16 + q_local], 65536);
                    unsigned int inv_266 = ~(unsigned int)blk2_229;
                    unsigned long long comp_267 = (unsigned long long)key_258 << 32 | (unsigned long long)inv_266;
                    if (is_stage_v[1] != 0 && nan_251 == 0 && comp_267 >= cthr_v[1]) {
                        unsigned int _atomic_old_45 = atomicAdd(&cand_cnt[qw_263], 1);
                        unsigned int slot_w_45 = _atomic_old_45;
                        if (slot_w_45 < 64) {
                            cand[qw_263 * 64 + (int)slot_w_45] = comp_267;
                        }
                    }
                    int blk2_268 = blks[7];
                    unsigned int raw_269 = __as_u32(xs[14]);
                    unsigned int mag_270 = raw_269 & 2147483647;
                    unsigned int nan_271 = (unsigned int)(mag_270 > 2139095040);
                    unsigned int bits_272 = ((mag_270 != 0) ? raw_269 : 0);
                    int em_273 = (int)(bits_272 >> 19 & 4095);
                    int _max_177 = ((em_273 - 1792) > (0) ? (em_273 - 1792) : (0));
                    int _min_48 = ((_max_177) < (511) ? (_max_177) : (511));
                    int raw_274 = _min_48;
                    int neg_275 = (int)(bits_272 >> 31);
                    int h10_276 = ((neg_275 != 0) ? 511 - raw_274 : 512 + raw_274);
                    unsigned int neg_mask_277 = -(bits_272 >> 31);
                    unsigned int key_278 = bits_272 ^ (neg_mask_277 | 2147483648u);
                    unsigned int invm_279 = ~(unsigned int)blk2_268 & idx_mask;
                    unsigned long long rk_280 = (unsigned long long)h10_276 << 54 | (unsigned long long)key_278 << 22 | (unsigned long long)invm_279;
                    int in_window_w_281 = is_pass0_v[0] | (int)(rk_280 >> rsh_v[0] == rprefix_v[0]);
                    int d_w_282 = (int)((unsigned int)(rk_280 >> dsh_v[0]) & dmask_v[0]);
                    int qw_283 = q_local * 2;
                    int counted_w_284 = is_hist_v[0] & (int)(nan_271 == 0) & in_window_w_281;
                    int dd_w_285 = ((counted_w_284 != 0) ? d_w_282 : 1024);
                    atomicAdd(&hist_ext[dd_w_285 * 16 + q_local], 1);
                    unsigned int inv_286 = ~(unsigned int)blk2_268;
                    unsigned long long comp_287 = (unsigned long long)key_278 << 32 | (unsigned long long)inv_286;
                    if (is_stage_v[0] != 0 && nan_271 == 0 && comp_287 >= cthr_v[0]) {
                        unsigned int _atomic_old_46 = atomicAdd(&cand_cnt[qw_283], 1);
                        unsigned int slot_w_46 = _atomic_old_46;
                        if (slot_w_46 < 64) {
                            cand[qw_283 * 64 + (int)slot_w_46] = comp_287;
                        }
                    }
                    unsigned int raw_288 = __as_u32(xs[15]);
                    unsigned int mag_289 = raw_288 & 2147483647;
                    unsigned int nan_290 = (unsigned int)(mag_289 > 2139095040);
                    unsigned int bits_291 = ((mag_289 != 0) ? raw_288 : 0);
                    int em_292 = (int)(bits_291 >> 19 & 4095);
                    int _max_178 = ((em_292 - 1792) > (0) ? (em_292 - 1792) : (0));
                    int _min_49 = ((_max_178) < (511) ? (_max_178) : (511));
                    int raw_293 = _min_49;
                    int neg_294 = (int)(bits_291 >> 31);
                    int h10_295 = ((neg_294 != 0) ? 511 - raw_293 : 512 + raw_293);
                    unsigned int neg_mask_296 = -(bits_291 >> 31);
                    unsigned int key_297 = bits_291 ^ (neg_mask_296 | 2147483648u);
                    unsigned int invm_298 = ~(unsigned int)blk2_268 & idx_mask;
                    unsigned long long rk_299 = (unsigned long long)h10_295 << 54 | (unsigned long long)key_297 << 22 | (unsigned long long)invm_298;
                    int in_window_w_300 = is_pass0_v[1] | (int)(rk_299 >> rsh_v[1] == rprefix_v[1]);
                    int d_w_301 = (int)((unsigned int)(rk_299 >> dsh_v[1]) & dmask_v[1]);
                    int qw_302 = q_local * 2 + 1;
                    int counted_w_303 = is_hist_v[1] & (int)(nan_290 == 0) & in_window_w_300;
                    int dd_w_304 = ((counted_w_303 != 0) ? d_w_301 : 1024);
                    atomicAdd(&hist_ext[dd_w_304 * 16 + q_local], 65536);
                    unsigned int inv_305 = ~(unsigned int)blk2_268;
                    unsigned long long comp_306 = (unsigned long long)key_297 << 32 | (unsigned long long)inv_305;
                    if (is_stage_v[1] != 0 && nan_290 == 0 && comp_306 >= cthr_v[1]) {
                        unsigned int _atomic_old_47 = atomicAdd(&cand_cnt[qw_302], 1);
                        unsigned int slot_w_47 = _atomic_old_47;
                        if (slot_w_47 < 64) {
                            cand[qw_302 * 64 + (int)slot_w_47] = comp_306;
                        }
                    }
                }
            }
            __syncthreads();
            int any_hist = is_hist_v[0] | is_hist_v[1];
            if (any_hist != 0) {
                int base_1 = seg * 1024 + q_local;
                subs[0] = 0;
                subs1[0] = 0;
                subs[1] = 0;
                subs1[1] = 0;
                subs[2] = 0;
                subs1[2] = 0;
                subs[3] = 0;
                subs1[3] = 0;
                unsigned int c_3 = hist[base_1];
                subs[0] = subs[0] + (c_3 & 65535);
                subs1[0] = subs1[0] + (c_3 >> 16);
                unsigned int c_0 = hist[base_1 + 16];
                subs[0] = subs[0] + (c_0 & 65535);
                subs1[0] = subs1[0] + (c_0 >> 16);
                unsigned int c_1_1 = hist[base_1 + 32];
                subs[0] = subs[0] + (c_1_1 & 65535);
                subs1[0] = subs1[0] + (c_1_1 >> 16);
                unsigned int c_2_1 = hist[base_1 + 48];
                subs[0] = subs[0] + (c_2_1 & 65535);
                subs1[0] = subs1[0] + (c_2_1 >> 16);
                unsigned int c_3_1 = hist[base_1 + 64];
                subs[0] = subs[0] + (c_3_1 & 65535);
                subs1[0] = subs1[0] + (c_3_1 >> 16);
                unsigned int c_4 = hist[base_1 + 80];
                subs[0] = subs[0] + (c_4 & 65535);
                subs1[0] = subs1[0] + (c_4 >> 16);
                unsigned int c_5_1 = hist[base_1 + 96];
                subs[0] = subs[0] + (c_5_1 & 65535);
                subs1[0] = subs1[0] + (c_5_1 >> 16);
                unsigned int c_6 = hist[base_1 + 112];
                subs[0] = subs[0] + (c_6 & 65535);
                subs1[0] = subs1[0] + (c_6 >> 16);
                unsigned int c_7 = hist[base_1 + 128];
                subs[0] = subs[0] + (c_7 & 65535);
                subs1[0] = subs1[0] + (c_7 >> 16);
                unsigned int c_8_1 = hist[base_1 + 144];
                subs[0] = subs[0] + (c_8_1 & 65535);
                subs1[0] = subs1[0] + (c_8_1 >> 16);
                unsigned int c_9 = hist[base_1 + 160];
                subs[0] = subs[0] + (c_9 & 65535);
                subs1[0] = subs1[0] + (c_9 >> 16);
                unsigned int c_10 = hist[base_1 + 176];
                subs[0] = subs[0] + (c_10 & 65535);
                subs1[0] = subs1[0] + (c_10 >> 16);
                unsigned int c_11 = hist[base_1 + 192];
                subs[0] = subs[0] + (c_11 & 65535);
                subs1[0] = subs1[0] + (c_11 >> 16);
                unsigned int c_12 = hist[base_1 + 208];
                subs[0] = subs[0] + (c_12 & 65535);
                subs1[0] = subs1[0] + (c_12 >> 16);
                unsigned int c_13 = hist[base_1 + 224];
                subs[0] = subs[0] + (c_13 & 65535);
                subs1[0] = subs1[0] + (c_13 >> 16);
                unsigned int c_14 = hist[base_1 + 240];
                subs[0] = subs[0] + (c_14 & 65535);
                subs1[0] = subs1[0] + (c_14 >> 16);
                unsigned int c_15 = hist[base_1 + 256];
                subs[1] = subs[1] + (c_15 & 65535);
                subs1[1] = subs1[1] + (c_15 >> 16);
                unsigned int c_16 = hist[base_1 + 272];
                subs[1] = subs[1] + (c_16 & 65535);
                subs1[1] = subs1[1] + (c_16 >> 16);
                unsigned int c_17 = hist[base_1 + 288];
                subs[1] = subs[1] + (c_17 & 65535);
                subs1[1] = subs1[1] + (c_17 >> 16);
                unsigned int c_18 = hist[base_1 + 304];
                subs[1] = subs[1] + (c_18 & 65535);
                subs1[1] = subs1[1] + (c_18 >> 16);
                unsigned int c_19 = hist[base_1 + 320];
                subs[1] = subs[1] + (c_19 & 65535);
                subs1[1] = subs1[1] + (c_19 >> 16);
                unsigned int c_20 = hist[base_1 + 336];
                subs[1] = subs[1] + (c_20 & 65535);
                subs1[1] = subs1[1] + (c_20 >> 16);
                unsigned int c_21 = hist[base_1 + 352];
                subs[1] = subs[1] + (c_21 & 65535);
                subs1[1] = subs1[1] + (c_21 >> 16);
                unsigned int c_22 = hist[base_1 + 368];
                subs[1] = subs[1] + (c_22 & 65535);
                subs1[1] = subs1[1] + (c_22 >> 16);
                unsigned int c_23 = hist[base_1 + 384];
                subs[1] = subs[1] + (c_23 & 65535);
                subs1[1] = subs1[1] + (c_23 >> 16);
                unsigned int c_24 = hist[base_1 + 400];
                subs[1] = subs[1] + (c_24 & 65535);
                subs1[1] = subs1[1] + (c_24 >> 16);
                unsigned int c_25 = hist[base_1 + 416];
                subs[1] = subs[1] + (c_25 & 65535);
                subs1[1] = subs1[1] + (c_25 >> 16);
                unsigned int c_26 = hist[base_1 + 432];
                subs[1] = subs[1] + (c_26 & 65535);
                subs1[1] = subs1[1] + (c_26 >> 16);
                unsigned int c_27 = hist[base_1 + 448];
                subs[1] = subs[1] + (c_27 & 65535);
                subs1[1] = subs1[1] + (c_27 >> 16);
                unsigned int c_28 = hist[base_1 + 464];
                subs[1] = subs[1] + (c_28 & 65535);
                subs1[1] = subs1[1] + (c_28 >> 16);
                unsigned int c_29 = hist[base_1 + 480];
                subs[1] = subs[1] + (c_29 & 65535);
                subs1[1] = subs1[1] + (c_29 >> 16);
                unsigned int c_30 = hist[base_1 + 496];
                subs[1] = subs[1] + (c_30 & 65535);
                subs1[1] = subs1[1] + (c_30 >> 16);
                unsigned int c_31 = hist[base_1 + 512];
                subs[2] = subs[2] + (c_31 & 65535);
                subs1[2] = subs1[2] + (c_31 >> 16);
                unsigned int c_32 = hist[base_1 + 528];
                subs[2] = subs[2] + (c_32 & 65535);
                subs1[2] = subs1[2] + (c_32 >> 16);
                unsigned int c_33 = hist[base_1 + 544];
                subs[2] = subs[2] + (c_33 & 65535);
                subs1[2] = subs1[2] + (c_33 >> 16);
                unsigned int c_34 = hist[base_1 + 560];
                subs[2] = subs[2] + (c_34 & 65535);
                subs1[2] = subs1[2] + (c_34 >> 16);
                unsigned int c_35 = hist[base_1 + 576];
                subs[2] = subs[2] + (c_35 & 65535);
                subs1[2] = subs1[2] + (c_35 >> 16);
                unsigned int c_36 = hist[base_1 + 592];
                subs[2] = subs[2] + (c_36 & 65535);
                subs1[2] = subs1[2] + (c_36 >> 16);
                unsigned int c_37 = hist[base_1 + 608];
                subs[2] = subs[2] + (c_37 & 65535);
                subs1[2] = subs1[2] + (c_37 >> 16);
                unsigned int c_38 = hist[base_1 + 624];
                subs[2] = subs[2] + (c_38 & 65535);
                subs1[2] = subs1[2] + (c_38 >> 16);
                unsigned int c_39 = hist[base_1 + 640];
                subs[2] = subs[2] + (c_39 & 65535);
                subs1[2] = subs1[2] + (c_39 >> 16);
                unsigned int c_40 = hist[base_1 + 656];
                subs[2] = subs[2] + (c_40 & 65535);
                subs1[2] = subs1[2] + (c_40 >> 16);
                unsigned int c_41 = hist[base_1 + 672];
                subs[2] = subs[2] + (c_41 & 65535);
                subs1[2] = subs1[2] + (c_41 >> 16);
                unsigned int c_42 = hist[base_1 + 688];
                subs[2] = subs[2] + (c_42 & 65535);
                subs1[2] = subs1[2] + (c_42 >> 16);
                unsigned int c_43 = hist[base_1 + 704];
                subs[2] = subs[2] + (c_43 & 65535);
                subs1[2] = subs1[2] + (c_43 >> 16);
                unsigned int c_44 = hist[base_1 + 720];
                subs[2] = subs[2] + (c_44 & 65535);
                subs1[2] = subs1[2] + (c_44 >> 16);
                unsigned int c_45 = hist[base_1 + 736];
                subs[2] = subs[2] + (c_45 & 65535);
                subs1[2] = subs1[2] + (c_45 >> 16);
                unsigned int c_46 = hist[base_1 + 752];
                subs[2] = subs[2] + (c_46 & 65535);
                subs1[2] = subs1[2] + (c_46 >> 16);
                unsigned int c_47 = hist[base_1 + 768];
                subs[3] = subs[3] + (c_47 & 65535);
                subs1[3] = subs1[3] + (c_47 >> 16);
                unsigned int c_48 = hist[base_1 + 784];
                subs[3] = subs[3] + (c_48 & 65535);
                subs1[3] = subs1[3] + (c_48 >> 16);
                unsigned int c_49 = hist[base_1 + 800];
                subs[3] = subs[3] + (c_49 & 65535);
                subs1[3] = subs1[3] + (c_49 >> 16);
                unsigned int c_50 = hist[base_1 + 816];
                subs[3] = subs[3] + (c_50 & 65535);
                subs1[3] = subs1[3] + (c_50 >> 16);
                unsigned int c_51 = hist[base_1 + 832];
                subs[3] = subs[3] + (c_51 & 65535);
                subs1[3] = subs1[3] + (c_51 >> 16);
                unsigned int c_52 = hist[base_1 + 848];
                subs[3] = subs[3] + (c_52 & 65535);
                subs1[3] = subs1[3] + (c_52 >> 16);
                unsigned int c_53 = hist[base_1 + 864];
                subs[3] = subs[3] + (c_53 & 65535);
                subs1[3] = subs1[3] + (c_53 >> 16);
                unsigned int c_54 = hist[base_1 + 880];
                subs[3] = subs[3] + (c_54 & 65535);
                subs1[3] = subs1[3] + (c_54 >> 16);
                unsigned int c_55 = hist[base_1 + 896];
                subs[3] = subs[3] + (c_55 & 65535);
                subs1[3] = subs1[3] + (c_55 >> 16);
                unsigned int c_56 = hist[base_1 + 912];
                subs[3] = subs[3] + (c_56 & 65535);
                subs1[3] = subs1[3] + (c_56 >> 16);
                unsigned int c_57 = hist[base_1 + 928];
                subs[3] = subs[3] + (c_57 & 65535);
                subs1[3] = subs1[3] + (c_57 >> 16);
                unsigned int c_58 = hist[base_1 + 944];
                subs[3] = subs[3] + (c_58 & 65535);
                subs1[3] = subs1[3] + (c_58 >> 16);
                unsigned int c_59 = hist[base_1 + 960];
                subs[3] = subs[3] + (c_59 & 65535);
                subs1[3] = subs1[3] + (c_59 >> 16);
                unsigned int c_60 = hist[base_1 + 976];
                subs[3] = subs[3] + (c_60 & 65535);
                subs1[3] = subs1[3] + (c_60 >> 16);
                unsigned int c_61 = hist[base_1 + 992];
                subs[3] = subs[3] + (c_61 & 65535);
                subs1[3] = subs1[3] + (c_61 >> 16);
                unsigned int c_62 = hist[base_1 + 1008];
                subs[3] = subs[3] + (c_62 & 65535);
                subs1[3] = subs1[3] + (c_62 >> 16);
                if (is_hist_v[0] != 0) {
                    seg_sum[seg * 32 + q_local * 2] = subs[0] + subs[1] + subs[2] + subs[3];
                }
                if (is_hist_v[1] != 0) {
                    seg_sum[seg * 32 + q_local * 2 + 1] = subs1[0] + subs1[1] + subs1[2] + subs1[3];
                }
            }
            __syncthreads();
            unsigned int contrib_a[1];
            contrib_a[0] = 0;
            int qw2a = q_local * 2;
            int qw2b = q_local * 2 + 1;
            if (is_hist_v[0] != 0) {
                unsigned int above = 0;
                unsigned int total_1 = 0;
                unsigned int v = seg_sum[qw2a];
                total_1 = total_1 + v;
                above = above + ((seg < 0) ? v : 0);
                unsigned int v_0 = seg_sum[32 + qw2a];
                total_1 = total_1 + v_0;
                above = above + ((seg < 1) ? v_0 : 0);
                unsigned int v_1 = seg_sum[64 + qw2a];
                total_1 = total_1 + v_1;
                above = above + ((seg < 2) ? v_1 : 0);
                unsigned int v_2 = seg_sum[96 + qw2a];
                total_1 = total_1 + v_2;
                above = above + ((seg < 3) ? v_2 : 0);
                unsigned int v_3 = seg_sum[128 + qw2a];
                total_1 = total_1 + v_3;
                above = above + ((seg < 4) ? v_3 : 0);
                unsigned int v_4 = seg_sum[160 + qw2a];
                total_1 = total_1 + v_4;
                above = above + ((seg < 5) ? v_4 : 0);
                unsigned int v_5 = seg_sum[192 + qw2a];
                total_1 = total_1 + v_5;
                above = above + ((seg < 6) ? v_5 : 0);
                unsigned int v_6 = seg_sum[224 + qw2a];
                total_1 = total_1 + v_6;
                above = above + ((seg < 7) ? v_6 : 0);
                unsigned int v_7 = seg_sum[256 + qw2a];
                total_1 = total_1 + v_7;
                above = above + ((seg < 8) ? v_7 : 0);
                unsigned int v_8 = seg_sum[288 + qw2a];
                total_1 = total_1 + v_8;
                above = above + ((seg < 9) ? v_8 : 0);
                unsigned int v_9 = seg_sum[320 + qw2a];
                total_1 = total_1 + v_9;
                above = above + ((seg < 10) ? v_9 : 0);
                unsigned int v_10 = seg_sum[352 + qw2a];
                total_1 = total_1 + v_10;
                above = above + ((seg < 11) ? v_10 : 0);
                unsigned int v_11 = seg_sum[384 + qw2a];
                total_1 = total_1 + v_11;
                above = above + ((seg < 12) ? v_11 : 0);
                unsigned int v_12 = seg_sum[416 + qw2a];
                total_1 = total_1 + v_12;
                above = above + ((seg < 13) ? v_12 : 0);
                unsigned int v_13 = seg_sum[448 + qw2a];
                total_1 = total_1 + v_13;
                above = above + ((seg < 14) ? v_13 : 0);
                unsigned int v_14 = seg_sum[480 + qw2a];
                total_1 = total_1 + v_14;
                above = above + ((seg < 15) ? v_14 : 0);
                pre[0] = above;
                pre[1] = total_1;
                unsigned int above_15 = pre[0];
                unsigned int total_16 = pre[1];
                unsigned int mine = subs[0] + subs[1] + subs[2] + subs[3];
                unsigned int r_rem = (unsigned int)r - ngt_v[0];
                int crossing = (int)(above_15 < r_rem && r_rem <= above_15 + mine);
                int nocross = (int)(total_16 < r_rem && seg == 0);
                if ((crossing | nocross) != 0) {
                    unsigned int tsel = 0;
                    unsigned int ct = 0;
                    unsigned int nadd = 0;
                    if (crossing != 0) {
                        unsigned int c3 = above_15;
                        unsigned int c2 = c3 + subs[3];
                        unsigned int c1 = c2 + subs[2];
                        unsigned int c0 = c1 + subs[1];
                        int k = 0;
                        unsigned int cum = c0;
                        if (r_rem <= c3 + subs[3]) {
                            k = 3;
                            cum = c3;
                        } else if (r_rem <= c2 + subs[2]) {
                            k = 2;
                            cum = c2;
                        } else {
                            if (r_rem <= c1 + subs[1]) {
                                k = 1;
                                cum = c1;
                            }
                        }
                        int bin0 = seg * 64 + k * 16;
                        int base_2 = bin0 * 16 + q_local;
                        unsigned int found = 0;
                        unsigned int tsel_0 = 0;
                        unsigned int ct_1 = 0;
                        unsigned int nadd_2 = 0;
                        unsigned int cw = hist[base_2 + 240];
                        unsigned int c_63 = cw & 65535;
                        unsigned int take = (unsigned int)(found == 0 && r_rem <= cum + c_63);
                        tsel_0 = ((take != 0) ? (unsigned int)(bin0 + 15) : tsel_0);
                        ct_1 = ((take != 0) ? c_63 : ct_1);
                        nadd_2 = ((take != 0) ? cum : nadd_2);
                        found = found | take;
                        cum = cum + c_63;
                        unsigned int cw_3 = hist[base_2 + 224];
                        unsigned int c_4_1 = cw_3 & 65535;
                        unsigned int take_5 = (unsigned int)(found == 0 && r_rem <= cum + c_4_1);
                        tsel_0 = ((take_5 != 0) ? (unsigned int)(bin0 + 14) : tsel_0);
                        ct_1 = ((take_5 != 0) ? c_4_1 : ct_1);
                        nadd_2 = ((take_5 != 0) ? cum : nadd_2);
                        found = found | take_5;
                        cum = cum + c_4_1;
                        unsigned int cw_6 = hist[base_2 + 208];
                        unsigned int c_7_1 = cw_6 & 65535;
                        unsigned int take_8 = (unsigned int)(found == 0 && r_rem <= cum + c_7_1);
                        tsel_0 = ((take_8 != 0) ? (unsigned int)(bin0 + 13) : tsel_0);
                        ct_1 = ((take_8 != 0) ? c_7_1 : ct_1);
                        nadd_2 = ((take_8 != 0) ? cum : nadd_2);
                        found = found | take_8;
                        cum = cum + c_7_1;
                        unsigned int cw_9 = hist[base_2 + 192];
                        unsigned int c_10_1 = cw_9 & 65535;
                        unsigned int take_11 = (unsigned int)(found == 0 && r_rem <= cum + c_10_1);
                        tsel_0 = ((take_11 != 0) ? (unsigned int)(bin0 + 12) : tsel_0);
                        ct_1 = ((take_11 != 0) ? c_10_1 : ct_1);
                        nadd_2 = ((take_11 != 0) ? cum : nadd_2);
                        found = found | take_11;
                        cum = cum + c_10_1;
                        unsigned int cw_12 = hist[base_2 + 176];
                        unsigned int c_13_1 = cw_12 & 65535;
                        unsigned int take_14 = (unsigned int)(found == 0 && r_rem <= cum + c_13_1);
                        tsel_0 = ((take_14 != 0) ? (unsigned int)(bin0 + 11) : tsel_0);
                        ct_1 = ((take_14 != 0) ? c_13_1 : ct_1);
                        nadd_2 = ((take_14 != 0) ? cum : nadd_2);
                        found = found | take_14;
                        cum = cum + c_13_1;
                        unsigned int cw_15 = hist[base_2 + 160];
                        unsigned int c_16_1 = cw_15 & 65535;
                        unsigned int take_17 = (unsigned int)(found == 0 && r_rem <= cum + c_16_1);
                        tsel_0 = ((take_17 != 0) ? (unsigned int)(bin0 + 10) : tsel_0);
                        ct_1 = ((take_17 != 0) ? c_16_1 : ct_1);
                        nadd_2 = ((take_17 != 0) ? cum : nadd_2);
                        found = found | take_17;
                        cum = cum + c_16_1;
                        unsigned int cw_18 = hist[base_2 + 144];
                        unsigned int c_19_1 = cw_18 & 65535;
                        unsigned int take_20 = (unsigned int)(found == 0 && r_rem <= cum + c_19_1);
                        tsel_0 = ((take_20 != 0) ? (unsigned int)(bin0 + 9) : tsel_0);
                        ct_1 = ((take_20 != 0) ? c_19_1 : ct_1);
                        nadd_2 = ((take_20 != 0) ? cum : nadd_2);
                        found = found | take_20;
                        cum = cum + c_19_1;
                        unsigned int cw_21 = hist[base_2 + 128];
                        unsigned int c_22_1 = cw_21 & 65535;
                        unsigned int take_23 = (unsigned int)(found == 0 && r_rem <= cum + c_22_1);
                        tsel_0 = ((take_23 != 0) ? (unsigned int)(bin0 + 8) : tsel_0);
                        ct_1 = ((take_23 != 0) ? c_22_1 : ct_1);
                        nadd_2 = ((take_23 != 0) ? cum : nadd_2);
                        found = found | take_23;
                        cum = cum + c_22_1;
                        unsigned int cw_24 = hist[base_2 + 112];
                        unsigned int c_25_1 = cw_24 & 65535;
                        unsigned int take_26 = (unsigned int)(found == 0 && r_rem <= cum + c_25_1);
                        tsel_0 = ((take_26 != 0) ? (unsigned int)(bin0 + 7) : tsel_0);
                        ct_1 = ((take_26 != 0) ? c_25_1 : ct_1);
                        nadd_2 = ((take_26 != 0) ? cum : nadd_2);
                        found = found | take_26;
                        cum = cum + c_25_1;
                        unsigned int cw_27 = hist[base_2 + 96];
                        unsigned int c_28_1 = cw_27 & 65535;
                        unsigned int take_29 = (unsigned int)(found == 0 && r_rem <= cum + c_28_1);
                        tsel_0 = ((take_29 != 0) ? (unsigned int)(bin0 + 6) : tsel_0);
                        ct_1 = ((take_29 != 0) ? c_28_1 : ct_1);
                        nadd_2 = ((take_29 != 0) ? cum : nadd_2);
                        found = found | take_29;
                        cum = cum + c_28_1;
                        unsigned int cw_30 = hist[base_2 + 80];
                        unsigned int c_31_1 = cw_30 & 65535;
                        unsigned int take_32 = (unsigned int)(found == 0 && r_rem <= cum + c_31_1);
                        tsel_0 = ((take_32 != 0) ? (unsigned int)(bin0 + 5) : tsel_0);
                        ct_1 = ((take_32 != 0) ? c_31_1 : ct_1);
                        nadd_2 = ((take_32 != 0) ? cum : nadd_2);
                        found = found | take_32;
                        cum = cum + c_31_1;
                        unsigned int cw_33 = hist[base_2 + 64];
                        unsigned int c_34_1 = cw_33 & 65535;
                        unsigned int take_35 = (unsigned int)(found == 0 && r_rem <= cum + c_34_1);
                        tsel_0 = ((take_35 != 0) ? (unsigned int)(bin0 + 4) : tsel_0);
                        ct_1 = ((take_35 != 0) ? c_34_1 : ct_1);
                        nadd_2 = ((take_35 != 0) ? cum : nadd_2);
                        found = found | take_35;
                        cum = cum + c_34_1;
                        unsigned int cw_36 = hist[base_2 + 48];
                        unsigned int c_37_1 = cw_36 & 65535;
                        unsigned int take_38 = (unsigned int)(found == 0 && r_rem <= cum + c_37_1);
                        tsel_0 = ((take_38 != 0) ? (unsigned int)(bin0 + 3) : tsel_0);
                        ct_1 = ((take_38 != 0) ? c_37_1 : ct_1);
                        nadd_2 = ((take_38 != 0) ? cum : nadd_2);
                        found = found | take_38;
                        cum = cum + c_37_1;
                        unsigned int cw_39 = hist[base_2 + 32];
                        unsigned int c_40_1 = cw_39 & 65535;
                        unsigned int take_41 = (unsigned int)(found == 0 && r_rem <= cum + c_40_1);
                        tsel_0 = ((take_41 != 0) ? (unsigned int)(bin0 + 2) : tsel_0);
                        ct_1 = ((take_41 != 0) ? c_40_1 : ct_1);
                        nadd_2 = ((take_41 != 0) ? cum : nadd_2);
                        found = found | take_41;
                        cum = cum + c_40_1;
                        unsigned int cw_42 = hist[base_2 + 16];
                        unsigned int c_43_1 = cw_42 & 65535;
                        unsigned int take_44 = (unsigned int)(found == 0 && r_rem <= cum + c_43_1);
                        tsel_0 = ((take_44 != 0) ? (unsigned int)(bin0 + 1) : tsel_0);
                        ct_1 = ((take_44 != 0) ? c_43_1 : ct_1);
                        nadd_2 = ((take_44 != 0) ? cum : nadd_2);
                        found = found | take_44;
                        cum = cum + c_43_1;
                        unsigned int cw_45 = hist[base_2];
                        unsigned int c_46_1 = cw_45 & 65535;
                        unsigned int take_47 = (unsigned int)(found == 0 && r_rem <= cum + c_46_1);
                        tsel_0 = ((take_47 != 0) ? (unsigned int)bin0 : tsel_0);
                        ct_1 = ((take_47 != 0) ? c_46_1 : ct_1);
                        nadd_2 = ((take_47 != 0) ? cum : nadd_2);
                        found = found | take_47;
                        cum = cum + c_46_1;
                        res[0] = tsel_0;
                        res[1] = ct_1;
                        res[2] = nadd_2;
                        tsel = res[0];
                        ct = res[1];
                        nadd = res[2];
                    } else {
                        unsigned int c0_1 = hist[q_local];
                        ct = c0_1 & 65535;
                        nadd = total_16 - ct;
                    }
                    unsigned int ngt_new = ngt_v[0] + nadd;
                    int over = (int)(ngt_new + ct > 64);
                    unsigned int rsh_new = 0;
                    unsigned long long rp_new = 0;
                    if (is_pass0_v[0] != 0) {
                        int sat = (int)(tsel == 0 || tsel == 511 || tsel == 512 || tsel == 1023);
                        if (sat != 0) {
                            rsh_new = 54;
                            rp_new = (unsigned long long)tsel;
                        } else {
                            rsh_new = 41;
                            unsigned int p13 = tsel + ((tsel >= 512) ? 5376 : 1792);
                            rp_new = (unsigned long long)tsel << 13 | (unsigned long long)p13;
                        }
                        state[qw2a * 8 + 5] = tsel;
                    } else {
                        rp_new = rprefix_v[0] << (unsigned long long)w_v[0] | (unsigned long long)tsel;
                        rsh_new = rshift_v[0] - w_v[0];
                        if (rsh_new == 22 && over != 0) {
                            rp_new = rp_new << (unsigned long long)(22 - index_bits);
                            rsh_new = (unsigned int)index_bits;
                        }
                    }
                    unsigned int mode_new = ((over != 0) ? 0 : 1);
                    state[qw2a * 8] = mode_new;
                    state[qw2a * 8 + 1] = rsh_new;
                    state[qw2a * 8 + 2] = (unsigned int)rp_new;
                    state[qw2a * 8 + 3] = (unsigned int)(rp_new >> 32);
                    state[qw2a * 8 + 4] = ngt_new;
                    contrib_a[0] = 1;
                }
            } else if (is_stage_v[0] != 0) {
                if (seg == 0) {
                    state[qw2a * 8] = 2;
                }
            }
            if (is_hist_v[1] != 0) {
                unsigned int above_1 = 0;
                unsigned int total_2 = 0;
                unsigned int v_15 = seg_sum[qw2b];
                total_2 = total_2 + v_15;
                above_1 = above_1 + ((seg < 0) ? v_15 : 0);
                unsigned int v_0_1 = seg_sum[32 + qw2b];
                total_2 = total_2 + v_0_1;
                above_1 = above_1 + ((seg < 1) ? v_0_1 : 0);
                unsigned int v_1_1 = seg_sum[64 + qw2b];
                total_2 = total_2 + v_1_1;
                above_1 = above_1 + ((seg < 2) ? v_1_1 : 0);
                unsigned int v_2_1 = seg_sum[96 + qw2b];
                total_2 = total_2 + v_2_1;
                above_1 = above_1 + ((seg < 3) ? v_2_1 : 0);
                unsigned int v_3_1 = seg_sum[128 + qw2b];
                total_2 = total_2 + v_3_1;
                above_1 = above_1 + ((seg < 4) ? v_3_1 : 0);
                unsigned int v_4_1 = seg_sum[160 + qw2b];
                total_2 = total_2 + v_4_1;
                above_1 = above_1 + ((seg < 5) ? v_4_1 : 0);
                unsigned int v_5_1 = seg_sum[192 + qw2b];
                total_2 = total_2 + v_5_1;
                above_1 = above_1 + ((seg < 6) ? v_5_1 : 0);
                unsigned int v_6_1 = seg_sum[224 + qw2b];
                total_2 = total_2 + v_6_1;
                above_1 = above_1 + ((seg < 7) ? v_6_1 : 0);
                unsigned int v_7_1 = seg_sum[256 + qw2b];
                total_2 = total_2 + v_7_1;
                above_1 = above_1 + ((seg < 8) ? v_7_1 : 0);
                unsigned int v_8_1 = seg_sum[288 + qw2b];
                total_2 = total_2 + v_8_1;
                above_1 = above_1 + ((seg < 9) ? v_8_1 : 0);
                unsigned int v_9_1 = seg_sum[320 + qw2b];
                total_2 = total_2 + v_9_1;
                above_1 = above_1 + ((seg < 10) ? v_9_1 : 0);
                unsigned int v_10_1 = seg_sum[352 + qw2b];
                total_2 = total_2 + v_10_1;
                above_1 = above_1 + ((seg < 11) ? v_10_1 : 0);
                unsigned int v_11_1 = seg_sum[384 + qw2b];
                total_2 = total_2 + v_11_1;
                above_1 = above_1 + ((seg < 12) ? v_11_1 : 0);
                unsigned int v_12_1 = seg_sum[416 + qw2b];
                total_2 = total_2 + v_12_1;
                above_1 = above_1 + ((seg < 13) ? v_12_1 : 0);
                unsigned int v_13_1 = seg_sum[448 + qw2b];
                total_2 = total_2 + v_13_1;
                above_1 = above_1 + ((seg < 14) ? v_13_1 : 0);
                unsigned int v_14_1 = seg_sum[480 + qw2b];
                total_2 = total_2 + v_14_1;
                above_1 = above_1 + ((seg < 15) ? v_14_1 : 0);
                pre[0] = above_1;
                pre[1] = total_2;
                unsigned int above_15_1 = pre[0];
                unsigned int total_16_1 = pre[1];
                unsigned int mine_1 = subs1[0] + subs1[1] + subs1[2] + subs1[3];
                unsigned int r_rem_1 = (unsigned int)r - ngt_v[1];
                int crossing_1 = (int)(above_15_1 < r_rem_1 && r_rem_1 <= above_15_1 + mine_1);
                int nocross_1 = (int)(total_16_1 < r_rem_1 && seg == 0);
                if ((crossing_1 | nocross_1) != 0) {
                    unsigned int tsel_1 = 0;
                    unsigned int ct_2 = 0;
                    unsigned int nadd_1 = 0;
                    if (crossing_1 != 0) {
                        unsigned int c3_1 = above_15_1;
                        unsigned int c2_1 = c3_1 + subs1[3];
                        unsigned int c1_1 = c2_1 + subs1[2];
                        unsigned int c0_2 = c1_1 + subs1[1];
                        int k_1 = 0;
                        unsigned int cum_1 = c0_2;
                        if (r_rem_1 <= c3_1 + subs1[3]) {
                            k_1 = 3;
                            cum_1 = c3_1;
                        } else if (r_rem_1 <= c2_1 + subs1[2]) {
                            k_1 = 2;
                            cum_1 = c2_1;
                        } else {
                            if (r_rem_1 <= c1_1 + subs1[1]) {
                                k_1 = 1;
                                cum_1 = c1_1;
                            }
                        }
                        int bin0_1 = seg * 64 + k_1 * 16;
                        int base_3 = bin0_1 * 16 + q_local;
                        unsigned int found_1 = 0;
                        unsigned int tsel_0_1 = 0;
                        unsigned int ct_1_1 = 0;
                        unsigned int nadd_2_1 = 0;
                        unsigned int cw_1 = hist[base_3 + 240];
                        unsigned int c_64 = cw_1 >> 16 & 65535;
                        unsigned int take_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_64);
                        tsel_0_1 = ((take_1 != 0) ? (unsigned int)(bin0_1 + 15) : tsel_0_1);
                        ct_1_1 = ((take_1 != 0) ? c_64 : ct_1_1);
                        nadd_2_1 = ((take_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_1;
                        cum_1 = cum_1 + c_64;
                        unsigned int cw_3_1 = hist[base_3 + 224];
                        unsigned int c_4_2 = cw_3_1 >> 16 & 65535;
                        unsigned int take_5_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_4_2);
                        tsel_0_1 = ((take_5_1 != 0) ? (unsigned int)(bin0_1 + 14) : tsel_0_1);
                        ct_1_1 = ((take_5_1 != 0) ? c_4_2 : ct_1_1);
                        nadd_2_1 = ((take_5_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_5_1;
                        cum_1 = cum_1 + c_4_2;
                        unsigned int cw_6_1 = hist[base_3 + 208];
                        unsigned int c_7_2 = cw_6_1 >> 16 & 65535;
                        unsigned int take_8_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_7_2);
                        tsel_0_1 = ((take_8_1 != 0) ? (unsigned int)(bin0_1 + 13) : tsel_0_1);
                        ct_1_1 = ((take_8_1 != 0) ? c_7_2 : ct_1_1);
                        nadd_2_1 = ((take_8_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_8_1;
                        cum_1 = cum_1 + c_7_2;
                        unsigned int cw_9_1 = hist[base_3 + 192];
                        unsigned int c_10_2 = cw_9_1 >> 16 & 65535;
                        unsigned int take_11_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_10_2);
                        tsel_0_1 = ((take_11_1 != 0) ? (unsigned int)(bin0_1 + 12) : tsel_0_1);
                        ct_1_1 = ((take_11_1 != 0) ? c_10_2 : ct_1_1);
                        nadd_2_1 = ((take_11_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_11_1;
                        cum_1 = cum_1 + c_10_2;
                        unsigned int cw_12_1 = hist[base_3 + 176];
                        unsigned int c_13_2 = cw_12_1 >> 16 & 65535;
                        unsigned int take_14_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_13_2);
                        tsel_0_1 = ((take_14_1 != 0) ? (unsigned int)(bin0_1 + 11) : tsel_0_1);
                        ct_1_1 = ((take_14_1 != 0) ? c_13_2 : ct_1_1);
                        nadd_2_1 = ((take_14_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_14_1;
                        cum_1 = cum_1 + c_13_2;
                        unsigned int cw_15_1 = hist[base_3 + 160];
                        unsigned int c_16_2 = cw_15_1 >> 16 & 65535;
                        unsigned int take_17_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_16_2);
                        tsel_0_1 = ((take_17_1 != 0) ? (unsigned int)(bin0_1 + 10) : tsel_0_1);
                        ct_1_1 = ((take_17_1 != 0) ? c_16_2 : ct_1_1);
                        nadd_2_1 = ((take_17_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_17_1;
                        cum_1 = cum_1 + c_16_2;
                        unsigned int cw_18_1 = hist[base_3 + 144];
                        unsigned int c_19_2 = cw_18_1 >> 16 & 65535;
                        unsigned int take_20_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_19_2);
                        tsel_0_1 = ((take_20_1 != 0) ? (unsigned int)(bin0_1 + 9) : tsel_0_1);
                        ct_1_1 = ((take_20_1 != 0) ? c_19_2 : ct_1_1);
                        nadd_2_1 = ((take_20_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_20_1;
                        cum_1 = cum_1 + c_19_2;
                        unsigned int cw_21_1 = hist[base_3 + 128];
                        unsigned int c_22_2 = cw_21_1 >> 16 & 65535;
                        unsigned int take_23_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_22_2);
                        tsel_0_1 = ((take_23_1 != 0) ? (unsigned int)(bin0_1 + 8) : tsel_0_1);
                        ct_1_1 = ((take_23_1 != 0) ? c_22_2 : ct_1_1);
                        nadd_2_1 = ((take_23_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_23_1;
                        cum_1 = cum_1 + c_22_2;
                        unsigned int cw_24_1 = hist[base_3 + 112];
                        unsigned int c_25_2 = cw_24_1 >> 16 & 65535;
                        unsigned int take_26_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_25_2);
                        tsel_0_1 = ((take_26_1 != 0) ? (unsigned int)(bin0_1 + 7) : tsel_0_1);
                        ct_1_1 = ((take_26_1 != 0) ? c_25_2 : ct_1_1);
                        nadd_2_1 = ((take_26_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_26_1;
                        cum_1 = cum_1 + c_25_2;
                        unsigned int cw_27_1 = hist[base_3 + 96];
                        unsigned int c_28_2 = cw_27_1 >> 16 & 65535;
                        unsigned int take_29_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_28_2);
                        tsel_0_1 = ((take_29_1 != 0) ? (unsigned int)(bin0_1 + 6) : tsel_0_1);
                        ct_1_1 = ((take_29_1 != 0) ? c_28_2 : ct_1_1);
                        nadd_2_1 = ((take_29_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_29_1;
                        cum_1 = cum_1 + c_28_2;
                        unsigned int cw_30_1 = hist[base_3 + 80];
                        unsigned int c_31_2 = cw_30_1 >> 16 & 65535;
                        unsigned int take_32_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_31_2);
                        tsel_0_1 = ((take_32_1 != 0) ? (unsigned int)(bin0_1 + 5) : tsel_0_1);
                        ct_1_1 = ((take_32_1 != 0) ? c_31_2 : ct_1_1);
                        nadd_2_1 = ((take_32_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_32_1;
                        cum_1 = cum_1 + c_31_2;
                        unsigned int cw_33_1 = hist[base_3 + 64];
                        unsigned int c_34_2 = cw_33_1 >> 16 & 65535;
                        unsigned int take_35_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_34_2);
                        tsel_0_1 = ((take_35_1 != 0) ? (unsigned int)(bin0_1 + 4) : tsel_0_1);
                        ct_1_1 = ((take_35_1 != 0) ? c_34_2 : ct_1_1);
                        nadd_2_1 = ((take_35_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_35_1;
                        cum_1 = cum_1 + c_34_2;
                        unsigned int cw_36_1 = hist[base_3 + 48];
                        unsigned int c_37_2 = cw_36_1 >> 16 & 65535;
                        unsigned int take_38_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_37_2);
                        tsel_0_1 = ((take_38_1 != 0) ? (unsigned int)(bin0_1 + 3) : tsel_0_1);
                        ct_1_1 = ((take_38_1 != 0) ? c_37_2 : ct_1_1);
                        nadd_2_1 = ((take_38_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_38_1;
                        cum_1 = cum_1 + c_37_2;
                        unsigned int cw_39_1 = hist[base_3 + 32];
                        unsigned int c_40_2 = cw_39_1 >> 16 & 65535;
                        unsigned int take_41_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_40_2);
                        tsel_0_1 = ((take_41_1 != 0) ? (unsigned int)(bin0_1 + 2) : tsel_0_1);
                        ct_1_1 = ((take_41_1 != 0) ? c_40_2 : ct_1_1);
                        nadd_2_1 = ((take_41_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_41_1;
                        cum_1 = cum_1 + c_40_2;
                        unsigned int cw_42_1 = hist[base_3 + 16];
                        unsigned int c_43_2 = cw_42_1 >> 16 & 65535;
                        unsigned int take_44_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_43_2);
                        tsel_0_1 = ((take_44_1 != 0) ? (unsigned int)(bin0_1 + 1) : tsel_0_1);
                        ct_1_1 = ((take_44_1 != 0) ? c_43_2 : ct_1_1);
                        nadd_2_1 = ((take_44_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_44_1;
                        cum_1 = cum_1 + c_43_2;
                        unsigned int cw_45_1 = hist[base_3];
                        unsigned int c_46_2 = cw_45_1 >> 16 & 65535;
                        unsigned int take_47_1 = (unsigned int)(found_1 == 0 && r_rem_1 <= cum_1 + c_46_2);
                        tsel_0_1 = ((take_47_1 != 0) ? (unsigned int)bin0_1 : tsel_0_1);
                        ct_1_1 = ((take_47_1 != 0) ? c_46_2 : ct_1_1);
                        nadd_2_1 = ((take_47_1 != 0) ? cum_1 : nadd_2_1);
                        found_1 = found_1 | take_47_1;
                        cum_1 = cum_1 + c_46_2;
                        res[0] = tsel_0_1;
                        res[1] = ct_1_1;
                        res[2] = nadd_2_1;
                        tsel_1 = res[0];
                        ct_2 = res[1];
                        nadd_1 = res[2];
                    } else {
                        unsigned int c0_3 = hist[q_local];
                        ct_2 = c0_3 >> 16 & 65535;
                        nadd_1 = total_16_1 - ct_2;
                    }
                    unsigned int ngt_new_1 = ngt_v[1] + nadd_1;
                    int over_1 = (int)(ngt_new_1 + ct_2 > 64);
                    unsigned int rsh_new_1 = 0;
                    unsigned long long rp_new_1 = 0;
                    if (is_pass0_v[1] != 0) {
                        int sat_1 = (int)(tsel_1 == 0 || tsel_1 == 511 || tsel_1 == 512 || tsel_1 == 1023);
                        if (sat_1 != 0) {
                            rsh_new_1 = 54;
                            rp_new_1 = (unsigned long long)tsel_1;
                        } else {
                            rsh_new_1 = 41;
                            unsigned int p13_1 = tsel_1 + ((tsel_1 >= 512) ? 5376 : 1792);
                            rp_new_1 = (unsigned long long)tsel_1 << 13 | (unsigned long long)p13_1;
                        }
                        state[qw2b * 8 + 5] = tsel_1;
                    } else {
                        rp_new_1 = rprefix_v[1] << (unsigned long long)w_v[1] | (unsigned long long)tsel_1;
                        rsh_new_1 = rshift_v[1] - w_v[1];
                        if (rsh_new_1 == 22 && over_1 != 0) {
                            rp_new_1 = rp_new_1 << (unsigned long long)(22 - index_bits);
                            rsh_new_1 = (unsigned int)index_bits;
                        }
                    }
                    unsigned int mode_new_1 = ((over_1 != 0) ? 0 : 1);
                    state[qw2b * 8] = mode_new_1;
                    state[qw2b * 8 + 1] = rsh_new_1;
                    state[qw2b * 8 + 2] = (unsigned int)rp_new_1;
                    state[qw2b * 8 + 3] = (unsigned int)(rp_new_1 >> 32);
                    state[qw2b * 8 + 4] = ngt_new_1;
                    contrib_a[0] = 1;
                }
            } else if (is_stage_v[1] != 0) {
                if (seg == 0) {
                    state[qw2b * 8] = 2;
                }
            }
            if (any_hist != 0) {
                int base_4 = seg * 1024 + q_local;
                hist[base_4] = 0;
                hist[base_4 + 16] = 0;
                hist[base_4 + 32] = 0;
                hist[base_4 + 48] = 0;
                hist[base_4 + 64] = 0;
                hist[base_4 + 80] = 0;
                hist[base_4 + 96] = 0;
                hist[base_4 + 112] = 0;
                hist[base_4 + 128] = 0;
                hist[base_4 + 144] = 0;
                hist[base_4 + 160] = 0;
                hist[base_4 + 176] = 0;
                hist[base_4 + 192] = 0;
                hist[base_4 + 208] = 0;
                hist[base_4 + 224] = 0;
                hist[base_4 + 240] = 0;
                hist[base_4 + 256] = 0;
                hist[base_4 + 272] = 0;
                hist[base_4 + 288] = 0;
                hist[base_4 + 304] = 0;
                hist[base_4 + 320] = 0;
                hist[base_4 + 336] = 0;
                hist[base_4 + 352] = 0;
                hist[base_4 + 368] = 0;
                hist[base_4 + 384] = 0;
                hist[base_4 + 400] = 0;
                hist[base_4 + 416] = 0;
                hist[base_4 + 432] = 0;
                hist[base_4 + 448] = 0;
                hist[base_4 + 464] = 0;
                hist[base_4 + 480] = 0;
                hist[base_4 + 496] = 0;
                hist[base_4 + 512] = 0;
                hist[base_4 + 528] = 0;
                hist[base_4 + 544] = 0;
                hist[base_4 + 560] = 0;
                hist[base_4 + 576] = 0;
                hist[base_4 + 592] = 0;
                hist[base_4 + 608] = 0;
                hist[base_4 + 624] = 0;
                hist[base_4 + 640] = 0;
                hist[base_4 + 656] = 0;
                hist[base_4 + 672] = 0;
                hist[base_4 + 688] = 0;
                hist[base_4 + 704] = 0;
                hist[base_4 + 720] = 0;
                hist[base_4 + 736] = 0;
                hist[base_4 + 752] = 0;
                hist[base_4 + 768] = 0;
                hist[base_4 + 784] = 0;
                hist[base_4 + 800] = 0;
                hist[base_4 + 816] = 0;
                hist[base_4 + 832] = 0;
                hist[base_4 + 848] = 0;
                hist[base_4 + 864] = 0;
                hist[base_4 + 880] = 0;
                hist[base_4 + 896] = 0;
                hist[base_4 + 912] = 0;
                hist[base_4 + 928] = 0;
                hist[base_4 + 944] = 0;
                hist[base_4 + 960] = 0;
                hist[base_4 + 976] = 0;
                hist[base_4 + 992] = 0;
                hist[base_4 + 1008] = 0;
            }
            uint32_t _cta_or_0 = __syncthreads_or(contrib_a[0] != 0);
            any_open = _cta_or_0;
        }
    }
    unsigned long long ovf_c = -1;
    int qw3 = q_local * 2;
    unsigned int cnt_w = cand_cnt[qw3];
    unsigned int _min_50 = ((cnt_w) < (64) ? (cnt_w) : (64));
    int n_w = (int)_min_50;
    unsigned long long cs[4];
    int ranks[4];
    int i = j;
    cs[0] = 0;
    if (i < n_w) {
        cs[0] = cand[qw3 * 64 + i];
    }
    ranks[0] = 0;
    int i_4 = j + 16;
    cs[1] = 0;
    if (i_4 < n_w) {
        cs[1] = cand[qw3 * 64 + i_4];
    }
    ranks[1] = 0;
    int i_5 = j + 32;
    cs[2] = 0;
    if (i_5 < n_w) {
        cs[2] = cand[qw3 * 64 + i_5];
    }
    ranks[2] = 0;
    int i_6 = j + 48;
    cs[3] = 0;
    if (i_6 < n_w) {
        cs[3] = cand[qw3 * 64 + i_6];
    }
    ranks[3] = 0;
    for (int m_1 = 0; m_1 < n_w; m_1++) {
        unsigned long long cm = cand[qw3 * 64 + m_1];
        ranks[0] = ranks[0] + (int)(cm > cs[0]);
        ranks[1] = ranks[1] + (int)(cm > cs[1]);
        ranks[2] = ranks[2] + (int)(cm > cs[2]);
        ranks[3] = ranks[3] + (int)(cm > cs[3]);
    }
    int i2 = j;
    int live = (int)(i2 < n_w) & (int)(ranks[0] < r);
    if (live != 0) {
        hist64[qw3 * 16 + ranks[0]] = cs[0];
    }
    int i2_7 = j + 16;
    int live_8 = (int)(i2_7 < n_w) & (int)(ranks[1] < r);
    if (live_8 != 0) {
        hist64[qw3 * 16 + ranks[1]] = cs[1];
    }
    int i2_9 = j + 32;
    int live_10 = (int)(i2_9 < n_w) & (int)(ranks[2] < r);
    if (live_10 != 0) {
        hist64[qw3 * 16 + ranks[2]] = cs[2];
    }
    int i2_11 = j + 48;
    int live_12 = (int)(i2_11 < n_w) & (int)(ranks[3] < r);
    if (live_12 != 0) {
        hist64[qw3 * 16 + ranks[3]] = cs[3];
    }
    if (j >= n_w && j < r) {
        hist64[qw3 * 16 + j] = 0;
    }
    int qw3_13 = q_local * 2 + 1;
    unsigned int cnt_w_14 = cand_cnt[qw3_13];
    unsigned int _min_51 = ((cnt_w_14) < (64) ? (cnt_w_14) : (64));
    int n_w_15 = (int)_min_51;
    unsigned long long cs_16[4];
    int ranks_17[4];
    int i_18 = j;
    cs_16[0] = 0;
    if (i_18 < n_w_15) {
        cs_16[0] = cand[qw3_13 * 64 + i_18];
    }
    ranks_17[0] = 0;
    int i_19 = j + 16;
    cs_16[1] = 0;
    if (i_19 < n_w_15) {
        cs_16[1] = cand[qw3_13 * 64 + i_19];
    }
    ranks_17[1] = 0;
    int i_20 = j + 32;
    cs_16[2] = 0;
    if (i_20 < n_w_15) {
        cs_16[2] = cand[qw3_13 * 64 + i_20];
    }
    ranks_17[2] = 0;
    int i_21 = j + 48;
    cs_16[3] = 0;
    if (i_21 < n_w_15) {
        cs_16[3] = cand[qw3_13 * 64 + i_21];
    }
    ranks_17[3] = 0;
    for (int m_2 = 0; m_2 < n_w_15; m_2++) {
        unsigned long long cm_1 = cand[qw3_13 * 64 + m_2];
        ranks_17[0] = ranks_17[0] + (int)(cm_1 > cs_16[0]);
        ranks_17[1] = ranks_17[1] + (int)(cm_1 > cs_16[1]);
        ranks_17[2] = ranks_17[2] + (int)(cm_1 > cs_16[2]);
        ranks_17[3] = ranks_17[3] + (int)(cm_1 > cs_16[3]);
    }
    int i2_22 = j;
    int live_23 = (int)(i2_22 < n_w_15) & (int)(ranks_17[0] < r);
    if (live_23 != 0) {
        hist64[qw3_13 * 16 + ranks_17[0]] = cs_16[0];
    }
    int i2_24 = j + 16;
    int live_25 = (int)(i2_24 < n_w_15) & (int)(ranks_17[1] < r);
    if (live_25 != 0) {
        hist64[qw3_13 * 16 + ranks_17[1]] = cs_16[1];
    }
    int i2_26 = j + 32;
    int live_27 = (int)(i2_26 < n_w_15) & (int)(ranks_17[2] < r);
    if (live_27 != 0) {
        hist64[qw3_13 * 16 + ranks_17[2]] = cs_16[2];
    }
    int i2_28 = j + 48;
    int live_29 = (int)(i2_28 < n_w_15) & (int)(ranks_17[3] < r);
    if (live_29 != 0) {
        hist64[qw3_13 * 16 + ranks_17[3]] = cs_16[3];
    }
    if (j >= n_w_15 && j < r) {
        hist64[qw3_13 * 16 + j] = 0;
    }
    __syncthreads();
    int is_last = 0;
    if (num_slices > 1) {
        int qm0 = warp_2 * 2 + (lane_1 >> 4);
        int jm0 = lane_1 & 15;
        int qm_h = qm0;
        unsigned long long pv_h = 0;
        if (jm0 < r) {
            pv_h = hist64[qm_h * 16 + jm0];
        }
        unsigned int cnt_h = cand_cnt[qm_h];
        unsigned int mode_h = state[qm_h * 8];
        int bad_h = (int)(cnt_h > 64) | (int)(mode_h != 2);
        if (jm0 == 0 && bad_h != 0) {
            pv_h = ovf_c;
        }
        partials[((group * num_slices + s) * 32 + qm_h) * 16 + jm0] = pv_h;
        int qm_h_0 = qm0 + 16;
        unsigned long long pv_h_1 = 0;
        if (jm0 < r) {
            pv_h_1 = hist64[qm_h_0 * 16 + jm0];
        }
        unsigned int cnt_h_2 = cand_cnt[qm_h_0];
        unsigned int mode_h_3 = state[qm_h_0 * 8];
        int bad_h_4 = (int)(cnt_h_2 > 64) | (int)(mode_h_3 != 2);
        if (jm0 == 0 && bad_h_4 != 0) {
            pv_h_1 = ovf_c;
        }
        partials[((group * num_slices + s) * 32 + qm_h_0) * 16 + jm0] = pv_h_1;
        __syncthreads();
        if (tid_0 == 0) {
            uint32_t _atomic_inc_old_0;
            asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                : "=r"(_atomic_inc_old_0) : "l"(&counters[group]), "r"(static_cast<uint32_t>(num_slices - 1)) : "memory");
            unsigned int old2 = _atomic_inc_old_0;
            flags[0] = (unsigned int)(old2 == (unsigned int)(num_slices - 1));
        }
        __syncthreads();
        unsigned int last_flag2 = flags[0];
        is_last = (int)(last_flag2 != 0);
        if (is_last != 0) {
            int qm_h2 = qm0;
            unsigned long long _vec_load_36[1];
            {
                uint64_t _scalar_bits_36;
                asm volatile("ld.global.cg.b64 %0, [%1];"
                    : "=l"(_scalar_bits_36) : "l"((const void*)(partials + ((group * num_slices * 32 + qm_h2) * 16 + jm0))) : "memory");
                _vec_load_36[0] = (unsigned long long)_scalar_bits_36;
            }
            unsigned long long vm_h = _vec_load_36[0];
            for (int sl = 1; sl < num_slices; sl++) {
                unsigned long long _vec_load_37[1];
                {
                    uint64_t _scalar_bits_37;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_37) : "l"((const void*)(partials + (((group * num_slices + sl) * 32 + qm_h2) * 16 + (15 - jm0)))) : "memory");
                    _vec_load_37[0] = (unsigned long long)_scalar_bits_37;
                }
                unsigned long long bv_h = _vec_load_37[0];
                vm_h = ((vm_h > bv_h) ? vm_h : bv_h);
                unsigned long long cur = vm_h;
                unsigned long long _shfl_xor_144 = __shfl_xor_sync(0xFFFFFFFF, cur, 8);
                unsigned long long vp = _shfl_xor_144;
                int take_max = (int)((jm0 & 8) == 0);
                int gt = (int)(cur > vp);
                cur = ((gt == take_max) ? cur : vp);
                unsigned long long _shfl_xor_145 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
                unsigned long long vp_0 = _shfl_xor_145;
                int take_max_1 = (int)((jm0 & 4) == 0);
                int gt_2 = (int)(cur > vp_0);
                cur = ((gt_2 == take_max_1) ? cur : vp_0);
                unsigned long long _shfl_xor_146 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
                unsigned long long vp_3 = _shfl_xor_146;
                int take_max_4 = (int)((jm0 & 2) == 0);
                int gt_5 = (int)(cur > vp_3);
                cur = ((gt_5 == take_max_4) ? cur : vp_3);
                unsigned long long _shfl_xor_147 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
                unsigned long long vp_6 = _shfl_xor_147;
                int take_max_7 = (int)((jm0 & 1) == 0);
                int gt_8 = (int)(cur > vp_6);
                cur = ((gt_8 == take_max_7) ? cur : vp_6);
                vm_h = cur;
            }
            hist64[qm_h2 * 16 + jm0] = vm_h;
            int qm_h2_0 = qm0 + 16;
            unsigned long long _vec_load_38[1];
            {
                uint64_t _scalar_bits_38;
                asm volatile("ld.global.cg.b64 %0, [%1];"
                    : "=l"(_scalar_bits_38) : "l"((const void*)(partials + ((group * num_slices * 32 + qm_h2_0) * 16 + jm0))) : "memory");
                _vec_load_38[0] = (unsigned long long)_scalar_bits_38;
            }
            unsigned long long vm_h_1 = _vec_load_38[0];
            for (int sl_1 = 1; sl_1 < num_slices; sl_1++) {
                unsigned long long _vec_load_39[1];
                {
                    uint64_t _scalar_bits_39;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_39) : "l"((const void*)(partials + (((group * num_slices + sl_1) * 32 + qm_h2_0) * 16 + (15 - jm0)))) : "memory");
                    _vec_load_39[0] = (unsigned long long)_scalar_bits_39;
                }
                unsigned long long bv_h_1 = _vec_load_39[0];
                vm_h_1 = ((vm_h_1 > bv_h_1) ? vm_h_1 : bv_h_1);
                unsigned long long cur_1 = vm_h_1;
                unsigned long long _shfl_xor_148 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 8);
                unsigned long long vp_1 = _shfl_xor_148;
                int take_max_2 = (int)((jm0 & 8) == 0);
                int gt_1 = (int)(cur_1 > vp_1);
                cur_1 = ((gt_1 == take_max_2) ? cur_1 : vp_1);
                unsigned long long _shfl_xor_149 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 4);
                unsigned long long vp_0_1 = _shfl_xor_149;
                int take_max_1_1 = (int)((jm0 & 4) == 0);
                int gt_2_1 = (int)(cur_1 > vp_0_1);
                cur_1 = ((gt_2_1 == take_max_1_1) ? cur_1 : vp_0_1);
                unsigned long long _shfl_xor_150 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 2);
                unsigned long long vp_3_1 = _shfl_xor_150;
                int take_max_4_1 = (int)((jm0 & 2) == 0);
                int gt_5_1 = (int)(cur_1 > vp_3_1);
                cur_1 = ((gt_5_1 == take_max_4_1) ? cur_1 : vp_3_1);
                unsigned long long _shfl_xor_151 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 1);
                unsigned long long vp_6_1 = _shfl_xor_151;
                int take_max_7_1 = (int)((jm0 & 1) == 0);
                int gt_8_1 = (int)(cur_1 > vp_6_1);
                cur_1 = ((gt_8_1 == take_max_7_1) ? cur_1 : vp_6_1);
                vm_h_1 = cur_1;
            }
            hist64[qm_h2_0 * 16 + jm0] = vm_h_1;
            __syncthreads();
        }
    }
    int do_out = (int)(num_slices == 1) | is_last;
    if (do_out != 0) {
        int k_2 = lane_1 & 15;
        int qo_local = warp_2 * 2 + (lane_1 >> 4);
        int qo = qtile * 32 + qo_local;
        unsigned int cnt_o = cand_cnt[qo_local];
        unsigned int mode_o = state[qo_local * 8];
        unsigned long long top_o = hist64[qo_local * 16];
        int ovf_o = (int)(r > 0) & ((int)(cnt_o > 64) | (int)(mode_o != 2) | (int)(top_o == ovf_c));
        int value = max_k_tiles;
        if (k_2 < fb) {
            value = k_2;
        } else if (k_2 < fb + fe) {
            value = num_valid_pages - fe + (k_2 - fb);
        } else {
            unsigned long long comp_o = hist64[qo_local * 16 + (k_2 - fb - fe)];
            if (comp_o != 0) {
                value = (int)~(unsigned int)comp_o;
            }
        }
        int _shfl_xor_152 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_52 = ((value) < (_shfl_xor_152) ? (value) : (_shfl_xor_152));
        int lo = _min_52;
        int _max_179 = ((value) > (_shfl_xor_152) ? (value) : (_shfl_xor_152));
        int hi = _max_179;
        int ascending = (int)((k_2 & 2) == 0);
        int lower_lane = (int)((k_2 & 1) == 0);
        int take_low = (int)(ascending == lower_lane);
        value = ((take_low == 1) ? lo : hi);
        int _shfl_xor_153 = __shfl_xor_sync(0xFFFFFFFF, value, 2);
        int _min_53 = ((value) < (_shfl_xor_153) ? (value) : (_shfl_xor_153));
        int lo_0 = _min_53;
        int _max_180 = ((value) > (_shfl_xor_153) ? (value) : (_shfl_xor_153));
        int hi_1 = _max_180;
        int ascending_2 = (int)((k_2 & 4) == 0);
        int lower_lane_3 = (int)((k_2 & 2) == 0);
        int take_low_4 = (int)(ascending_2 == lower_lane_3);
        value = ((take_low_4 == 1) ? lo_0 : hi_1);
        int _shfl_xor_154 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_54 = ((value) < (_shfl_xor_154) ? (value) : (_shfl_xor_154));
        int lo_5 = _min_54;
        int _max_181 = ((value) > (_shfl_xor_154) ? (value) : (_shfl_xor_154));
        int hi_6 = _max_181;
        int ascending_7 = (int)((k_2 & 4) == 0);
        int lower_lane_8 = (int)((k_2 & 1) == 0);
        int take_low_9 = (int)(ascending_7 == lower_lane_8);
        value = ((take_low_9 == 1) ? lo_5 : hi_6);
        int _shfl_xor_155 = __shfl_xor_sync(0xFFFFFFFF, value, 4);
        int _min_55 = ((value) < (_shfl_xor_155) ? (value) : (_shfl_xor_155));
        int lo_10 = _min_55;
        int _max_182 = ((value) > (_shfl_xor_155) ? (value) : (_shfl_xor_155));
        int hi_11 = _max_182;
        int ascending_12 = (int)((k_2 & 8) == 0);
        int lower_lane_13 = (int)((k_2 & 4) == 0);
        int take_low_14 = (int)(ascending_12 == lower_lane_13);
        value = ((take_low_14 == 1) ? lo_10 : hi_11);
        int _shfl_xor_156 = __shfl_xor_sync(0xFFFFFFFF, value, 2);
        int _min_56 = ((value) < (_shfl_xor_156) ? (value) : (_shfl_xor_156));
        int lo_15 = _min_56;
        int _max_183 = ((value) > (_shfl_xor_156) ? (value) : (_shfl_xor_156));
        int hi_16 = _max_183;
        int ascending_17 = (int)((k_2 & 8) == 0);
        int lower_lane_18 = (int)((k_2 & 2) == 0);
        int take_low_19 = (int)(ascending_17 == lower_lane_18);
        value = ((take_low_19 == 1) ? lo_15 : hi_16);
        int _shfl_xor_157 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_57 = ((value) < (_shfl_xor_157) ? (value) : (_shfl_xor_157));
        int lo_20 = _min_57;
        int _max_184 = ((value) > (_shfl_xor_157) ? (value) : (_shfl_xor_157));
        int hi_21 = _max_184;
        int ascending_22 = (int)((k_2 & 8) == 0);
        int lower_lane_23 = (int)((k_2 & 1) == 0);
        int take_low_24 = (int)(ascending_22 == lower_lane_23);
        value = ((take_low_24 == 1) ? lo_20 : hi_21);
        int _shfl_xor_158 = __shfl_xor_sync(0xFFFFFFFF, value, 8);
        int _min_58 = ((value) < (_shfl_xor_158) ? (value) : (_shfl_xor_158));
        int lo_25 = _min_58;
        int _max_185 = ((value) > (_shfl_xor_158) ? (value) : (_shfl_xor_158));
        int hi_26 = _max_185;
        int ascending_27 = (int)((k_2 & 16) == 0);
        int lower_lane_28 = (int)((k_2 & 8) == 0);
        int take_low_29 = (int)(ascending_27 == lower_lane_28);
        value = ((take_low_29 == 1) ? lo_25 : hi_26);
        int _shfl_xor_159 = __shfl_xor_sync(0xFFFFFFFF, value, 4);
        int _min_59 = ((value) < (_shfl_xor_159) ? (value) : (_shfl_xor_159));
        int lo_30 = _min_59;
        int _max_186 = ((value) > (_shfl_xor_159) ? (value) : (_shfl_xor_159));
        int hi_31 = _max_186;
        int ascending_32 = (int)((k_2 & 16) == 0);
        int lower_lane_33 = (int)((k_2 & 4) == 0);
        int take_low_34 = (int)(ascending_32 == lower_lane_33);
        value = ((take_low_34 == 1) ? lo_30 : hi_31);
        int _shfl_xor_160 = __shfl_xor_sync(0xFFFFFFFF, value, 2);
        int _min_60 = ((value) < (_shfl_xor_160) ? (value) : (_shfl_xor_160));
        int lo_35 = _min_60;
        int _max_187 = ((value) > (_shfl_xor_160) ? (value) : (_shfl_xor_160));
        int hi_36 = _max_187;
        int ascending_37 = (int)((k_2 & 16) == 0);
        int lower_lane_38 = (int)((k_2 & 2) == 0);
        int take_low_39 = (int)(ascending_37 == lower_lane_38);
        value = ((take_low_39 == 1) ? lo_35 : hi_36);
        int _shfl_xor_161 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_61 = ((value) < (_shfl_xor_161) ? (value) : (_shfl_xor_161));
        int lo_40 = _min_61;
        int _max_188 = ((value) > (_shfl_xor_161) ? (value) : (_shfl_xor_161));
        int hi_41 = _max_188;
        int ascending_42 = (int)((k_2 & 16) == 0);
        int lower_lane_43 = (int)((k_2 & 1) == 0);
        int take_low_44 = (int)(ascending_42 == lower_lane_43);
        value = ((take_low_44 == 1) ? lo_40 : hi_41);
        if (qo < total_q) {
            int stored = value;
            if (value >= num_valid_pages) {
                stored = -1;
            }
            if (ovf_o != 0) {
                int dbg = -2;
                if (k_2 == 1) {
                    dbg = (int)cnt_o;
                } else if (k_2 == 2) {
                    dbg = (int)state[qo_local * 8 + 1];
                } else {
                    if (k_2 == 3) {
                        dbg = (int)state[qo_local * 8 + 2];
                    } else if (k_2 == 4) {
                        dbg = (int)state[qo_local * 8 + 3];
                    } else {
                        if (k_2 == 5) {
                            dbg = (int)state[qo_local * 8 + 4];
                        } else if (k_2 == 6) {
                            dbg = (int)state[qo_local * 8 + 5];
                        } else {
                            if (k_2 == 7) {
                                dbg = (int)state[qo_local * 8];
                            }
                        }
                    }
                }
                stored = dbg;
            }
            output[(qo * num_heads + h) * 16 + k_2] = stored;
        }
        int k_45 = lane_1 & 15;
        int qo_local_46 = warp_2 * 2 + (lane_1 >> 4) + 16;
        int qo_47 = qtile * 32 + qo_local_46;
        unsigned int cnt_o_48 = cand_cnt[qo_local_46];
        unsigned int mode_o_49 = state[qo_local_46 * 8];
        unsigned long long top_o_50 = hist64[qo_local_46 * 16];
        int ovf_o_51 = (int)(r > 0) & ((int)(cnt_o_48 > 64) | (int)(mode_o_49 != 2) | (int)(top_o_50 == ovf_c));
        int value_52 = max_k_tiles;
        if (k_45 < fb) {
            value_52 = k_45;
        } else if (k_45 < fb + fe) {
            value_52 = num_valid_pages - fe + (k_45 - fb);
        } else {
            unsigned long long comp_o_1 = hist64[qo_local_46 * 16 + (k_45 - fb - fe)];
            if (comp_o_1 != 0) {
                value_52 = (int)~(unsigned int)comp_o_1;
            }
        }
        int _shfl_xor_162 = __shfl_xor_sync(0xFFFFFFFF, value_52, 1);
        int _min_62 = ((value_52) < (_shfl_xor_162) ? (value_52) : (_shfl_xor_162));
        int lo_53 = _min_62;
        int _max_189 = ((value_52) > (_shfl_xor_162) ? (value_52) : (_shfl_xor_162));
        int hi_54 = _max_189;
        int ascending_55 = (int)((k_45 & 2) == 0);
        int lower_lane_56 = (int)((k_45 & 1) == 0);
        int take_low_57 = (int)(ascending_55 == lower_lane_56);
        value_52 = ((take_low_57 == 1) ? lo_53 : hi_54);
        int _shfl_xor_163 = __shfl_xor_sync(0xFFFFFFFF, value_52, 2);
        int _min_63 = ((value_52) < (_shfl_xor_163) ? (value_52) : (_shfl_xor_163));
        int lo_58 = _min_63;
        int _max_190 = ((value_52) > (_shfl_xor_163) ? (value_52) : (_shfl_xor_163));
        int hi_59 = _max_190;
        int ascending_60 = (int)((k_45 & 4) == 0);
        int lower_lane_61 = (int)((k_45 & 2) == 0);
        int take_low_62 = (int)(ascending_60 == lower_lane_61);
        value_52 = ((take_low_62 == 1) ? lo_58 : hi_59);
        int _shfl_xor_164 = __shfl_xor_sync(0xFFFFFFFF, value_52, 1);
        int _min_64 = ((value_52) < (_shfl_xor_164) ? (value_52) : (_shfl_xor_164));
        int lo_63 = _min_64;
        int _max_191 = ((value_52) > (_shfl_xor_164) ? (value_52) : (_shfl_xor_164));
        int hi_64 = _max_191;
        int ascending_65 = (int)((k_45 & 4) == 0);
        int lower_lane_66 = (int)((k_45 & 1) == 0);
        int take_low_67 = (int)(ascending_65 == lower_lane_66);
        value_52 = ((take_low_67 == 1) ? lo_63 : hi_64);
        int _shfl_xor_165 = __shfl_xor_sync(0xFFFFFFFF, value_52, 4);
        int _min_65 = ((value_52) < (_shfl_xor_165) ? (value_52) : (_shfl_xor_165));
        int lo_68 = _min_65;
        int _max_192 = ((value_52) > (_shfl_xor_165) ? (value_52) : (_shfl_xor_165));
        int hi_69 = _max_192;
        int ascending_70 = (int)((k_45 & 8) == 0);
        int lower_lane_71 = (int)((k_45 & 4) == 0);
        int take_low_72 = (int)(ascending_70 == lower_lane_71);
        value_52 = ((take_low_72 == 1) ? lo_68 : hi_69);
        int _shfl_xor_166 = __shfl_xor_sync(0xFFFFFFFF, value_52, 2);
        int _min_66 = ((value_52) < (_shfl_xor_166) ? (value_52) : (_shfl_xor_166));
        int lo_73 = _min_66;
        int _max_193 = ((value_52) > (_shfl_xor_166) ? (value_52) : (_shfl_xor_166));
        int hi_74 = _max_193;
        int ascending_75 = (int)((k_45 & 8) == 0);
        int lower_lane_76 = (int)((k_45 & 2) == 0);
        int take_low_77 = (int)(ascending_75 == lower_lane_76);
        value_52 = ((take_low_77 == 1) ? lo_73 : hi_74);
        int _shfl_xor_167 = __shfl_xor_sync(0xFFFFFFFF, value_52, 1);
        int _min_67 = ((value_52) < (_shfl_xor_167) ? (value_52) : (_shfl_xor_167));
        int lo_78 = _min_67;
        int _max_194 = ((value_52) > (_shfl_xor_167) ? (value_52) : (_shfl_xor_167));
        int hi_79 = _max_194;
        int ascending_80 = (int)((k_45 & 8) == 0);
        int lower_lane_81 = (int)((k_45 & 1) == 0);
        int take_low_82 = (int)(ascending_80 == lower_lane_81);
        value_52 = ((take_low_82 == 1) ? lo_78 : hi_79);
        int _shfl_xor_168 = __shfl_xor_sync(0xFFFFFFFF, value_52, 8);
        int _min_68 = ((value_52) < (_shfl_xor_168) ? (value_52) : (_shfl_xor_168));
        int lo_83 = _min_68;
        int _max_195 = ((value_52) > (_shfl_xor_168) ? (value_52) : (_shfl_xor_168));
        int hi_84 = _max_195;
        int ascending_85 = (int)((k_45 & 16) == 0);
        int lower_lane_86 = (int)((k_45 & 8) == 0);
        int take_low_87 = (int)(ascending_85 == lower_lane_86);
        value_52 = ((take_low_87 == 1) ? lo_83 : hi_84);
        int _shfl_xor_169 = __shfl_xor_sync(0xFFFFFFFF, value_52, 4);
        int _min_69 = ((value_52) < (_shfl_xor_169) ? (value_52) : (_shfl_xor_169));
        int lo_88 = _min_69;
        int _max_196 = ((value_52) > (_shfl_xor_169) ? (value_52) : (_shfl_xor_169));
        int hi_89 = _max_196;
        int ascending_90 = (int)((k_45 & 16) == 0);
        int lower_lane_91 = (int)((k_45 & 4) == 0);
        int take_low_92 = (int)(ascending_90 == lower_lane_91);
        value_52 = ((take_low_92 == 1) ? lo_88 : hi_89);
        int _shfl_xor_170 = __shfl_xor_sync(0xFFFFFFFF, value_52, 2);
        int _min_70 = ((value_52) < (_shfl_xor_170) ? (value_52) : (_shfl_xor_170));
        int lo_93 = _min_70;
        int _max_197 = ((value_52) > (_shfl_xor_170) ? (value_52) : (_shfl_xor_170));
        int hi_94 = _max_197;
        int ascending_95 = (int)((k_45 & 16) == 0);
        int lower_lane_96 = (int)((k_45 & 2) == 0);
        int take_low_97 = (int)(ascending_95 == lower_lane_96);
        value_52 = ((take_low_97 == 1) ? lo_93 : hi_94);
        int _shfl_xor_171 = __shfl_xor_sync(0xFFFFFFFF, value_52, 1);
        int _min_71 = ((value_52) < (_shfl_xor_171) ? (value_52) : (_shfl_xor_171));
        int lo_98 = _min_71;
        int _max_198 = ((value_52) > (_shfl_xor_171) ? (value_52) : (_shfl_xor_171));
        int hi_99 = _max_198;
        int ascending_100 = (int)((k_45 & 16) == 0);
        int lower_lane_101 = (int)((k_45 & 1) == 0);
        int take_low_102 = (int)(ascending_100 == lower_lane_101);
        value_52 = ((take_low_102 == 1) ? lo_98 : hi_99);
        if (qo_47 < total_q) {
            int stored_1 = value_52;
            if (value_52 >= num_valid_pages) {
                stored_1 = -1;
            }
            if (ovf_o_51 != 0) {
                int dbg_1 = -2;
                if (k_45 == 1) {
                    dbg_1 = (int)cnt_o_48;
                } else if (k_45 == 2) {
                    dbg_1 = (int)state[qo_local_46 * 8 + 1];
                } else {
                    if (k_45 == 3) {
                        dbg_1 = (int)state[qo_local_46 * 8 + 2];
                    } else if (k_45 == 4) {
                        dbg_1 = (int)state[qo_local_46 * 8 + 3];
                    } else {
                        if (k_45 == 5) {
                            dbg_1 = (int)state[qo_local_46 * 8 + 4];
                        } else if (k_45 == 6) {
                            dbg_1 = (int)state[qo_local_46 * 8 + 5];
                        } else {
                            if (k_45 == 7) {
                                dbg_1 = (int)state[qo_local_46 * 8];
                            }
                        }
                    }
                }
                stored_1 = dbg_1;
            }
            output[(qo_47 * num_heads + h) * 16 + k_45] = stored_1;
        }
    }
}

} // extern "C"
